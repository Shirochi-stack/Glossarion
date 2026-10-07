"""Manga batch runner: start, per-image worker, stop, glossary workflow (GUI-free).

Shared GUI-free core (Glossarion mobile rewrite, milestone U8). ``MangaTranslationTab``
(manga_integration.py) inherits :class:`MangaRunMixin`, whose methods were moved there
verbatim (``_start_translation_heavy`` calls the three env steps split out into
``manga_env``). :class:`HeadlessMangaRunner` is the mobile "tab": the same mixins, the
non-GUI state of ``MangaTranslationTab.__init__`` and a ``HeadlessOwner`` as ``main_gui``
(the duck-typed main GUI ``MangaTranslator`` reads); :func:`run_manga_batch` is what the
MANGA job adapter calls on its job thread.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import concurrent.futures
import ctypes
import json
import os
import platform
import threading
import time
import traceback
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

# Optional: psutil/ctypes helpers to reduce GUI lag by lowering background thread priority
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from manga_translator import MangaTranslator
from manga_env import HeadlessMangaState, MangaEnvMixin, MangaOcrSessionMixin
from manga_files_core import (FileListShim, ImageRenderer, MangaFilesMixin, MangaHooksMixin,
                              _get_app_dir, _natural_sort_key)

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




class MangaRunMixin:
    """MangaTranslationTab's batch run: start, worker, stop and glossary workflow (moved verbatim, U8)."""

    # Class-level cancellation flag for all instances
    _global_cancelled = False
    _global_cancel_lock = threading.RLock()

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

    def _is_stop_requested(self) -> bool:
        """Check if stop has been requested using multiple sources"""
        # During graceful stop, don't consider it "stopped" for log suppression
        # We want to keep showing logs until the API call actually completes
        if os.environ.get('GRACEFUL_STOP') == '1':
            return False
        
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
        
        # CRITICAL: Also reset MangaTranslator class flag
        try:
            from manga_translator import MangaTranslator
            MangaTranslator.set_global_cancellation(False)
            # CRITICAL: Force-release any stale pool checkouts from previous interrupted translations
            # This ensures inpainters/detectors are available for the new translation
            MangaTranslator.force_release_all_pool_checkouts()
        except ImportError:
            pass
        
        # CRITICAL: Also reset UnifiedClient flag
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient.set_global_cancellation(False)
        except ImportError:
            pass
        
        # CRITICAL: Reset module-level global_stop_flag in unified_api_client
        # This is separate from class-level cancellation and must be reset
        try:
            from unified_api_client import set_stop_flag
            set_stop_flag(False)
        except ImportError:
            pass

    def reset_stop_flags(self):
        """Reset all stop flags when starting new translation"""
        self.is_running = False
        if hasattr(self, 'stop_flag'):
            self.stop_flag.clear()
        self._reset_global_cancellation()
        self._log("🔄 Stop flags reset for new translation", "debug")

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
            if not ocr_settings.get('bubble_detection_enabled', True):
                return False
            det_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
            model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
            # Sanitize model_id to avoid unrelated JSON paths
            try:
                import os
                if model_id and model_id.lower().endswith('.json'):
                    model_id = ''
                if det_type in ('rtdetr', 'rtdetr_onnx') and model_id and os.path.isfile(model_id):
                    model_id = ''
            except Exception:
                pass

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

    def _is_translation_start_cancelled(self, start_token=None) -> bool:
        """Return True if the pending startup was superseded or stopped."""
        try:
            if start_token is not None and getattr(self, '_translation_start_token', None) != start_token:
                return True
            if getattr(self, '_translation_start_cancel_requested', False):
                return True
            if not getattr(self, 'is_running', False):
                return True
        except Exception:
            return True
        return False

    def _wait_for_previous_translation_to_finish(self, previous_future=None, previous_thread=None):
        """Cancel/wait for stale workers without blocking the GUI thread."""
        try:
            if previous_future:
                try:
                    if not previous_future.done():
                        self._log("Canceling previous translation future...", "info")
                        previous_future.cancel()
                        try:
                            previous_future.result(timeout=3.0)
                        except Exception:
                            pass
                    if getattr(self, 'translation_future', None) is previous_future:
                        self.translation_future = None
                except Exception as fut_err:
                    self._log(f"Warning: error canceling previous translation future: {fut_err}", "debug")

            if previous_thread and previous_thread.is_alive():
                try:
                    self._log("Waiting for previous translation thread to finish...", "info")
                    previous_thread.join(timeout=5.0)
                    if previous_thread.is_alive():
                        self._log("Previous translation thread did not stop cleanly", "warning")
                    elif getattr(self, 'translation_thread', None) is previous_thread:
                        self.translation_thread = None
                except Exception as thread_err:
                    self._log(f"Warning: error waiting for previous translation thread: {thread_err}", "debug")
        except Exception as e:
            self._log(f"Warning: error checking previous translation worker: {e}", "debug")

    def _start_translation_heavy(self, previous_future=None, previous_thread=None, start_token=None):
        """Heavy part of start: build configs, init client/translator, and launch worker (runs off-main-thread)."""
        import os
        try:
            # Lower priority & restrict affinity for this launcher thread (Windows)
            try:
                _lower_current_thread_priority_and_affinity('MANGA_RESERVE_CORES')
            except Exception:
                pass

            self._wait_for_previous_translation_to_finish(previous_future, previous_thread)
            if self._is_translation_start_cancelled(start_token):
                self._log("Translation startup canceled before worker launch", "warning")
                self._translation_startup_pending = False
                self.update_queue.put(('ui_state', 'translation_complete', start_token))
                return

            # Reset graceful stop mode from any previous translation only after
            # stale workers have had a chance to observe the old stop flags.
            self._reset_manga_graceful_stop_env()
            
            # CRITICAL: Reset ALL cancellation flags including inpainter worker restart.
            # After stop+resume the inpainter worker is dead (killed by psutil cleanup).
            # This proactively restarts it and reloads the model so inpainting doesn't stall.
            try:
                import ImageRenderer
                ImageRenderer._reset_cancellation_flags(self)
            except Exception as e:
                print(f"[START_HEAVY] _reset_cancellation_flags failed: {e}")
            if self._is_translation_start_cancelled(start_token):
                self._log("Translation startup canceled before configuration", "warning")
                self._translation_startup_pending = False
                self.update_queue.put(('ui_state', 'translation_complete', start_token))
                return
            prepared = self._prepare_manga_run_env(start_token)
            if prepared is None:
                return
            ocr_config, api_key, model, needs_new_client = prepared
        except Exception as e:
            # Surface any startup error and reset UI so the app doesn't look stuck
            try:
                import traceback
                self._log(f"❌ Startup error: {e}", "error")
                self._log(traceback.format_exc(), "debug")
            except Exception:
                pass
            self._stop_startup_heartbeat()
            self._reset_ui_state(start_token)
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
            
            # CRITICAL: Clean up old translator's futures/executors before creating new one
            # This prevents orphaned futures from blocking or interfering
            if hasattr(self, 'translator') and self.translator:
                try:
                    old_translator = self.translator
                    # Clear early inpainting futures
                    if hasattr(old_translator, '_inpainting_future') and old_translator._inpainting_future:
                        try:
                            old_translator._inpainting_future.cancel()
                        except Exception:
                            pass
                        old_translator._inpainting_future = None
                    # Shutdown early inpainting executor
                    if hasattr(old_translator, '_inpainting_executor') and old_translator._inpainting_executor:
                        try:
                            old_translator._inpainting_executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            old_translator._inpainting_executor.shutdown(wait=False)
                        except Exception:
                            pass
                        old_translator._inpainting_executor = None
                    # Clear checkout references
                    if hasattr(old_translator, '_clear_checkout_references'):
                        old_translator._clear_checkout_references()
                    self._log("🧹 Cleaned up previous translator state", "debug")
                except Exception as cleanup_err:
                    self._log(f"⚠️ Error cleaning up old translator: {cleanup_err}", "debug")
            
            # CRITICAL: Set batch environment variables BEFORE creating translator
            # This ensures MangaTranslator picks up the batch settings on initialization
            self._apply_manga_batch_env()
            
            try:
                self.translator = MangaTranslator(
                    ocr_config,
                    self.main_gui.client,
                    self.main_gui,
                    log_callback=self._log
                )
                self._configure_manga_ocr_io(self.translator)
                
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
                self._reset_ui_state(start_token)
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
                
                # Define helper function first
                def _get_gui_val(var, default):
                    try:
                        if hasattr(var, 'get'):
                            return var.get()
                        return var
                    except Exception:
                        return default

                # Update environment variables and translator attributes
                mode_val = str(_get_gui_val(self.main_gui.batch_mode_var, 'direct')).strip().lower() if hasattr(self.main_gui, 'batch_mode_var') else str(self.main_gui.config.get('batching_mode', 'direct')).strip().lower()
                group_val = 3
                try:
                    if hasattr(self.main_gui, 'batch_group_size_var'):
                        group_val = int(_get_gui_val(self.main_gui.batch_group_size_var, 3) or 3)
                    else:
                        group_val = int(self.main_gui.config.get('batch_group_size', 3) or 3)
                except Exception:
                    group_val = 3

                if batch_translation_enabled:
                    os.environ['BATCH_TRANSLATION'] = '1'
                    os.environ['BATCH_SIZE'] = str(max(1, batch_size_value))
                    os.environ['BATCHING_MODE'] = mode_val
                    os.environ['BATCH_GROUP_SIZE'] = str(max(1, group_val))
                    os.environ['CONSERVATIVE_BATCHING'] = '1' if mode_val == 'conservative' else '0'
                    self.translator.batch_mode = True
                    self.translator.batch_size = max(1, batch_size_value)
                    self.translator.batching_mode = mode_val
                    self.translator.batch_group_size = max(1, group_val)
                    self._log(f"📦 Batch Translation UPDATED: {batch_size_value} concurrent API calls (mode={mode_val}, group={group_val})", "info")
                else:
                    os.environ['BATCH_TRANSLATION'] = '0'
                    os.environ['BATCH_SIZE'] = '1'
                    os.environ['BATCHING_MODE'] = mode_val
                    os.environ['BATCH_GROUP_SIZE'] = str(max(1, group_val))
                    os.environ['CONSERVATIVE_BATCHING'] = '1' if mode_val == 'conservative' else '0'
                    self.translator.batch_mode = False
                    self.translator.batch_size = 1
                    self.translator.batching_mode = mode_val
                    self.translator.batch_group_size = max(1, group_val)
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
            
            # Ensure inpainting settings are properly synchronized.
            # PySide6 build: drive this from the Skip Inpainter checkbox
            # (self.skip_inpainting_value) and the inpaint method radios
            # (self.inpaint_method_value: 'local' | 'cloud' | 'hybrid').
            # The legacy Tkinter `inpainting_mode_var` no longer exists.
            skip_inpainting_flag = bool(
                getattr(self, 'skip_inpainting_value',
                        self.main_gui.config.get('manga_skip_inpainting', False))
            )
            inpaint_method = str(
                getattr(self, 'inpaint_method_value',
                        self.main_gui.config.get('manga_inpaint_method', 'local'))
            ).lower()

            if skip_inpainting_flag:
                self.translator.skip_inpainting = True
                self.translator.use_cloud_inpainting = False
                self.translator.inpaint_mode = 'skip'
                self._log("Inpainting: SKIP (toggle enabled)", "info")
            elif inpaint_method == 'cloud':
                self.translator.skip_inpainting = False
                saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                if saved_api_key:
                    self.translator.use_cloud_inpainting = True
                    self.translator.replicate_api_key = saved_api_key
                    self.translator.inpaint_mode = 'cloud'
                    self._log("Inpainting: CLOUD (Replicate)", "debug")
                else:
                    self.translator.use_cloud_inpainting = False
                    self.translator.inpaint_mode = 'local'
                    self._log("Inpainting: LOCAL (no Replicate key, fallback)", "warning")
            elif inpaint_method == 'hybrid':
                self.translator.skip_inpainting = False
                self.translator.use_cloud_inpainting = False
                self.translator.inpaint_mode = 'hybrid'
                self._log("Inpainting: HYBRID", "debug")
            else:
                # Local (default)
                self.translator.skip_inpainting = False
                self.translator.use_cloud_inpainting = False
                self.translator.inpaint_mode = 'local'
                self._log("Inpainting: LOCAL", "debug")

            # Double-check the settings are applied correctly
            self._log(f"Inpainting final status:", "debug")
            self._log(f"  - Skip: {self.translator.skip_inpainting}", "debug")
            self._log(f"  - Cloud: {self.translator.use_cloud_inpainting}", "debug")
            self._log(f"  - Mode: {'SKIP' if self.translator.skip_inpainting else 'CLOUD' if self.translator.use_cloud_inpainting else 'LOCAL'}", "debug")
        
        # Preflight RT-DETR to avoid first-page fallback after aggressive cleanup
        try:
            ocr_set = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) or {}
            if ocr_set.get('bubble_detection_enabled', True):
                # Ensure a default RT-DETR model id exists when required
                if ocr_set.get('detector_type', 'rtdetr') in ('rtdetr', 'auto'):
                    if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                        ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                self._preflight_bubble_detector(ocr_set)
        except Exception:
            pass
        
        # Reset progress using the visible-order image range, if one is active.
        processing_files, range_error = self._manga_range_filtered_files()
        if range_error or not processing_files:
            self._log(f"Invalid image range: {range_error or 'range matched no files'}", "error")
            self.update_queue.put(('ui_state', 'translation_complete', start_token))
            return
        self._manga_processing_files = processing_files
        range_text = str(getattr(self, 'manga_image_range_value', '') or '').strip()
        skipped_count = max(0, len(self.selected_files) - len(processing_files))
        if skipped_count:
            detail = f" ({range_text})" if range_text else ""
            self._log(
                f"Processing {len(processing_files)}/{len(self.selected_files)} visible images; skipping {skipped_count}{detail}",
                "info"
            )

        # Reset progress
        self.total_files = len(processing_files)
        self.completed_files = 0
        self.failed_files = 0
        self.current_file_index = 0
        
        if self._is_translation_start_cancelled(start_token):
            self._log("Translation startup canceled before file processing", "warning")
            self._translation_startup_pending = False
            self.update_queue.put(('ui_state', 'translation_complete', start_token))
            return
        # Queue UI updates to be processed by main thread (just for file list disable)
        self.update_queue.put(('ui_state', 'translation_started', start_token))
        
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
            rolling_enabled = True
            
            self._log(f"Contextual: {'Enabled' if contextual_enabled else 'Disabled'}", "info")
            self._log(f"History limit: {trans_history} exchanges", "info")
            self._log(f"Rolling history: {'Enabled' if rolling_enabled else 'Disabled'}", "info")
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_value else 'Disabled'}", "info")
        
        # Stop heartbeat before launching worker; now regular progress takes over
        self._stop_startup_heartbeat()
        
        
        # Update progress to show we're starting the translation worker
        self._log("🚀 Launching translation worker...", "info")
        self._update_progress(0, self.total_files, "Starting translation...")

        if self._is_translation_start_cancelled(start_token):
            self._log("Translation startup canceled before worker launch", "warning")
            self._translation_startup_pending = False
            self.update_queue.put(('ui_state', 'translation_complete', start_token))
            return
        self._translation_startup_pending = False

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
                self.translation_future = self.executor.submit(self._translation_worker, start_token)
            else:
                # Fallback to dedicated thread
                self.translation_thread = threading.Thread(
                    target=self._translation_worker,
                    args=(start_token,),
                    daemon=True
                )
                self.translation_thread.start()
        except Exception:
            # Last resort fallback to thread
            self.translation_thread = threading.Thread(
                target=self._translation_worker,
                args=(start_token,),
                daemon=True
            )
            self.translation_thread.start()

    def _translation_worker(self, start_token=None):
        """Worker thread for translation"""
        # Defensive imports at function start to prevent UnboundLocalError
        import os
        import traceback
        import time
        
        # Track start time for performance reporting
        translation_start_time = time.time()
        
        try:
            # Defensive: ensure translator exists before using it (legacy callers may start this worker early)
            if not hasattr(self, 'translator') or self.translator is None:
                self._log("⚠️ Translator not initialized yet; skipping worker start", "warning")
                return
            if hasattr(self.translator, 'set_stop_flag'):
                self.translator.set_stop_flag(self.stop_flag)
            run_files = self._current_manga_processing_files()
            if not run_files:
                self._log("No manga files selected for this run", "warning")
                return
            
            # Ensure API parallelism (batch API calls) is controlled independently of local parallel processing.
            # Propagate the GUI "Batch Translation" toggle into environment so Unified API Client applies it globally
            # for all providers (including custom endpoints).
            try:
                # Support both Tkinter (with .get()) and PySide6 (plain value)
                batch_enabled = False
                if hasattr(self.main_gui, 'batch_translation_var'):
                    if hasattr(self.main_gui.batch_translation_var, 'get'):
                        batch_enabled = bool(self.main_gui.batch_translation_var.get())
                    else:
                        batch_enabled = bool(self.main_gui.batch_translation_var)
                os.environ['BATCH_TRANSLATION'] = '1' if batch_enabled else '0'
                
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
                os.environ['BATCH_SIZE'] = bs_val or os.environ.get('BATCH_SIZE', '3')
            except Exception:
                # Non-fatal if env cannot be set
                pass
            
            # Panel-level parallelization setting (LOCAL threading for panels)
            advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
            panel_parallel = bool(advanced.get('parallel_panel_translation', False))
            requested_panel_workers = int(advanced.get('panel_max_workers', 2))

            # Decouple from global parallel processing: panel concurrency is governed ONLY by panel settings
            effective_workers = requested_panel_workers if (panel_parallel and len(run_files) > 1) else 1

            # Model preloading phase
            self._log("🔧 Model preloading phase", "info")
            # Log current counters (diagnostic)
            try:
                st = self.translator.get_preload_counters() if hasattr(self.translator, 'get_preload_counters') else None
                if st:
                    self._log(f"   Preload status: Inpainter [{st.get('inpaint_checked_out',0)}/{st.get('inpaint_spares',0)} in use, {st.get('inpaint_available',0)} available] | Detector [{st.get('detector_checked_out',0)}/{st.get('detector_spares',0)} in use, {st.get('detector_available',0)} available]", "debug")
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
                    desired_bd = min(int(effective_workers), max(1, int(len(run_files) or 1)))
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
            # 2) Preload LOCAL inpainting instances for panel parallelism.
            # Honor the Skip Inpainter toggle: if it's on, don't preload at all.
            inpaint_preload_event = None
            try:
                inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'cloud')
                skip_inpainting_flag = bool(
                    getattr(self, 'skip_inpainting_value',
                            self.main_gui.config.get('manga_skip_inpainting', False))
                )
                if skip_inpainting_flag:
                    self._log("🚫 Skip Inpainter enabled — skipping panel inpainter preload", "info")
                elif (
                    effective_workers > 1
                    and inpaint_method == 'local'
                    and hasattr(self, 'translator')
                    and self.translator
                ):
                    local_method = self.main_gui.config.get('manga_local_inpaint_model', 'anime')
                    model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')
                    if not model_path:
                        model_path = self.main_gui.config.get(f'{local_method}_model_path', '')
                    if str(local_method or '').lower() == 'custom-image-edit':
                        endpoint = (
                            str(getattr(self.main_gui, 'custom_image_edit_endpoint_var', '') or '').strip()
                            or str(self.main_gui.config.get('custom_image_edit_endpoint', '') or '').strip()
                        )
                        if endpoint:
                            model_path = endpoint
                    
                    # Preload one shared instance plus spares for parallel panel processing
                    # Constrain to actual number of files (no need for more workers than files)
                    desired_inp = min(int(effective_workers), max(1, int(len(run_files) or 1)))
                    self._log(f"🧰 Preloading {desired_inp} local inpainting instance(s) for panel workers...", "info")
                    try:
                        import time as _time
                        t0 = _time.time()
                        # Use concurrent preload for faster startup when loading multiple instances
                        self.translator.preload_local_inpainters_concurrent(local_method, model_path, desired_inp)
                        dt = _time.time() - t0
                        self._log(f"⏱️ Local inpainting preload finished in {dt:.2f}s", "info")
                    except Exception as _e:
                        self._log(f"⚠️ Local inpainting preload failed: {_e}", "warning")
                        self._log(traceback.format_exc(), "debug")
            except Exception as preload_err:
                self._log(f"⚠️ Inpainting preload setup failed: {preload_err}", "warning")
            
            # Log updated counters (diagnostic)
            try:
                st2 = self.translator.get_preload_counters() if hasattr(self.translator, 'get_preload_counters') else None
                if st2:
                    self._log(f"   Preload status: Inpainter [{st2.get('inpaint_checked_out',0)}/{st2.get('inpaint_spares',0)} in use, {st2.get('inpaint_available',0)} available] | Detector [{st2.get('detector_checked_out',0)}/{st2.get('detector_spares',0)} in use, {st2.get('detector_available',0)} available]", "debug")
            except Exception:
                pass

            glossary_only_run = bool(getattr(self, '_manga_glossary_only_run', False))

            if self._manga_glossary_workflow_enabled():
                self._run_manga_glossary_workflow(glossary_only_run=glossary_only_run)

            elif panel_parallel and len(run_files) > 1 and effective_workers > 1:
                self._log(f"🚀 Parallel PANEL translation ENABLED ({effective_workers} workers)", "info")
                
                import concurrent.futures
                import threading as _threading
                progress_lock = _threading.Lock()
                counters = {
                    'started': 0,
                    'done': 0,
                    'failed': 0
                }
                total = self.total_files
                
                def process_single(idx, filepath):
                    # Graceful stop check: if an API call completed during graceful stop, stop now
                    if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                        self._log("✅ Graceful stop: Image completed, stopping...", "info")
                        return False
                    
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
                        self._configure_manga_ocr_io(translator)
                        translator.set_stop_flag(self.stop_flag)
                        
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
                            # CRITICAL FIX: Create separate folder for EACH image to prevent overlays
                            # This enables thread-safe parallel translation and instant feedback
                            base_name = os.path.splitext(filename)[0]
                            
                            # Check for OUTPUT_DIRECTORY override
                            override_dir = os.environ.get('OUTPUT_DIRECTORY')
                            if override_dir:
                                parent_dir = override_dir
                            else:
                                parent_dir = os.path.dirname(filepath)
                            
                            # Create unique folder per image for isolation
                            output_dir = os.path.join(parent_dir, f"{base_name}_translated")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, filename)
                        
                        # Start monitoring this image for auto-updating preview
                        self._monitor_translation_output(filepath)
                        
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
                        
                        # Check if translation was stopped before processing results
                        if self.stop_flag.is_set():
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
                                translated_count = sum(1 for r in regions if (r.get('translated_text') or '').strip())
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
                        
                        # Update counters and progress (thread-safe)
                        with progress_lock:
                            if translation_successful:
                                self.completed_files += 1
                                self._log(f"✅ Translation completed: {filename}", "success")
                                
                                # Save cleaned image path to state if available
                                if result.get('cleaned_image_path'):
                                    try:
                                        if hasattr(self, 'image_state_manager') and self.image_state_manager:
                                            self.image_state_manager.update_state(filepath, {
                                                'cleaned_image_path': result.get('cleaned_image_path')
                                            })
                                            print(f"[CLEANED] Saved cleaned image path to state: {os.path.basename(result.get('cleaned_image_path'))}")
                                    except Exception as e:
                                        print(f"[CLEANED] Failed to save cleaned path to state: {e}")
                                
                                # Update image preview to show translated output if this is the current file
                                # Use thread-safe queue to update preview from parallel worker
                                if hasattr(self, 'image_preview_widget'):
                                    current_path = self.image_preview_widget.current_image_path
                                    print(f"[PREVIEW_UPDATE]   Translated output: {result.get('output_path')}")
                                    
                                    # Normalize paths for comparison (fix slash mixing on Windows)
                                    norm_current = os.path.normpath(current_path) if current_path else None
                                    norm_source = os.path.normpath(filepath)
                                    
                                    # Match if showing original OR any translated version of this image
                                    is_current = (norm_current == norm_source or 
                                                 (norm_current and os.path.dirname(norm_current).endswith(f"{os.path.splitext(os.path.basename(filepath))[0]}_translated")))
                                    
                                    if is_current:
                                        # Use queue for thread-safe GUI update with source path for persistence
                                        self.update_queue.put(('preview_update', {
                                            'translated_path': result.get('output_path'),
                                            'source_path': filepath,
                                            'switch_to_output': True,
                                        }))
                                        self._log(f"🖼️ Queued preview update to show translated image", "debug")
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
                    for idx, filepath in enumerate(run_files):
                        # Graceful stop check: if an API call completed, stop submitting new work
                        if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                            self._log("✅ Graceful stop: Image completed, stopping...", "info")
                            break
                        if self.stop_flag.is_set():
                            break
                        futures.append(executor.submit(process_single, idx, filepath))
                        if stagger_ms > 0:
                            time.sleep(stagger_ms / 1000.0)
                    
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
                for index, filepath in enumerate(run_files):
                    # Graceful stop check: if an API call completed, stop now
                    if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                        self._log("✅ Graceful stop: Image completed, stopping...", "info")
                        break
                    if self.stop_flag.is_set():
                        self._log("\n⏹️ Translation stopped by user", "warning")
                        break
                    
                    # IMPORTANT: Reset translator state for each new image
                    if hasattr(self.translator, 'reset_for_new_image'):
                        self.translator.reset_for_new_image()
                    
                    self.current_file_index = index
                    filename = os.path.basename(filepath)
                    
                    # Start monitoring this image for auto-updating preview
                    self._monitor_translation_output(filepath)
                    
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
                            # CRITICAL FIX: Create separate folder for EACH image to prevent overlays
                            # This enables thread-safe parallel translation and instant feedback
                            # Each image gets its own isolated folder like: image001_translated/image001.png
                            base_name = os.path.splitext(filename)[0]
                            
                            # Check for OUTPUT_DIRECTORY override
                            override_dir = os.environ.get('OUTPUT_DIRECTORY')
                            if override_dir:
                                parent_dir = override_dir
                            else:
                                parent_dir = os.path.dirname(filepath)
                            
                            # Create unique folder per image for isolation
                            output_dir = os.path.join(parent_dir, f"{base_name}_translated")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, filename)
                        
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
                                
                                # Save cleaned image path to state if available
                                if result.get('cleaned_image_path'):
                                    try:
                                        if hasattr(self, 'image_state_manager') and self.image_state_manager:
                                            self.image_state_manager.update_state(filepath, {
                                                'cleaned_image_path': result.get('cleaned_image_path')
                                            })
                                            print(f"[CLEANED] Saved cleaned image path to state: {os.path.basename(result.get('cleaned_image_path'))}")
                                    except Exception as e:
                                        print(f"[CLEANED] Failed to save cleaned path to state: {e}")
                                
                                # Update image preview to show translated output if this is the current file  
                                # Use thread-safe queue to update preview from sequential worker
                                if hasattr(self, 'image_preview_widget'):
                                    current_path = self.image_preview_widget.current_image_path
                                    #print(f"[PREVIEW_UPDATE] Translation completed for: {filename}")
                                    #print(f"[PREVIEW_UPDATE]   Current preview path: {current_path}")
                                    #print(f"[PREVIEW_UPDATE]   Source filepath: {filepath}")
                                    print(f"[PREVIEW_UPDATE]   Translated output: {result.get('output_path')}")
                                    
                                    # Normalize paths for comparison (fix slash mixing on Windows)
                                    norm_current = os.path.normpath(current_path) if current_path else None
                                    norm_source = os.path.normpath(filepath)
                                    
                                    # Match if showing original OR any translated version of this image
                                    is_current = (norm_current == norm_source or 
                                                 (norm_current and os.path.dirname(norm_current).endswith(f"{os.path.splitext(os.path.basename(filepath))[0]}_translated")))
                                    #print(f"[PREVIEW_UPDATE]   Normalized current: {norm_current}")
                                    #print(f"[PREVIEW_UPDATE]   Normalized source: {norm_source}")
                                    #print(f"[PREVIEW_UPDATE]   Is current image: {is_current}")
                                    
                                    if is_current:
                                        # Use queue for thread-safe GUI update
                                        self.update_queue.put(('preview_update', {
                                            'translated_path': result.get('output_path'),
                                            'source_path': filepath,
                                            'switch_to_output': True,
                                        }))
                                        self._log(f"🖼️ Queued preview update to show translated image", "debug")
                                
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
                        
            
            if not bool(getattr(self, '_manga_glossary_only_run', False)):
                # Finalize CBZ packaging (both modes)
                try:
                    self._finalize_cbz_jobs()
                except Exception:
                    pass

                # Create CBZ from isolated folders if enabled
                if hasattr(self, 'create_cbz_at_end_value') and self.create_cbz_at_end_value:
                    try:
                        self._create_cbz_from_isolated_folders()
                    except Exception as e:
                        self._log(f"⚠️ Failed to create CBZ file: {e}", "warning")

                # Auto-consolidate translated images (silent, no message box) if enabled
                if hasattr(self, 'auto_consolidate_images_value') and self.auto_consolidate_images_value:
                    try:
                        if hasattr(self, 'image_preview_widget') and self.image_preview_widget:
                            self._log("📥 Consolidating translated images...", "info")
                            self.image_preview_widget._on_download_images_clicked(silent=True)
                            self._log("✅ Images consolidated successfully", "success")
                    except Exception as e:
                        self._log(f"⚠️ Image consolidation failed: {e}", "warning")
            
            # Final summary - only if not stopped
            if not self.stop_flag.is_set():
                # Calculate elapsed time
                elapsed_time = time.time() - translation_start_time
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60
                
                summary_title = "Glossary Summary" if bool(getattr(self, '_manga_glossary_only_run', False)) else "Translation Summary"
                self._log(f"\n{'='*60}", "info")
                self._log(f"📊 {summary_title}:", "info")
                self._log(f"   Total files: {self.total_files}", "info")
                self._log(f"   ✅ Successful: {self.completed_files}", "success")
                self._log(f"   ❌ Failed: {self.failed_files}", "error" if self.failed_files > 0 else "info")
                if minutes > 0:
                    self._log(f"   ⏱️ Total time: {minutes}m {seconds:.1f}s", "info")
                else:
                    self._log(f"   ⏱️ Total time: {seconds:.1f}s", "info")
                self._log(f"{'='*60}\n", "info")
                
                complete_label = (
                    f"Glossary complete! {self.completed_files} pages used, {self.failed_files} failed"
                    if bool(getattr(self, '_manga_glossary_only_run', False))
                    else f"Complete! {self.completed_files} successful, {self.failed_files} failed"
                )
                self._update_progress(
                    self.total_files,
                    self.total_files,
                    complete_label
                )
                
                # Enable download button and preview mode if translation succeeded
                if (not bool(getattr(self, '_manga_glossary_only_run', False))
                        and self.completed_files > 0 and hasattr(self, 'image_preview_widget')):
                    # Determine translated folder path (use parent directory for isolated folders)
                    translated_folder = None
                    run_files_for_preview = self._current_manga_processing_files()
                    if run_files_for_preview:
                        first_file = run_files_for_preview[0]
                        # Always use parent directory now - isolated folders are children of this
                        translated_folder = os.path.dirname(first_file)
                    
                    # Set translated folder via update queue (main thread) for GUI updates
                    if translated_folder and os.path.exists(translated_folder):
                        try:
                            self.update_queue.put(('set_translated_folder', translated_folder))
                            self._log(f"💻 Preview mode will be updated with isolated translated images", "info")
                        except Exception as e:
                            self._log(f"⚠️ Failed to queue preview mode update: {e}", "warning")
            
        except Exception as e:
            self._log(f"\n❌ Translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
        
        finally:
            self._manga_glossary_only_run = False
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
            
            self._manga_processing_files = None
            self._update_manga_image_range_display()

            # Reset UI state (PySide6 - must call on main thread)
            try:
                # Use the existing update_queue to schedule UI reset on main thread
                # This queue is processed by the main thread's timer
                self.update_queue.put(('ui_state', 'translation_complete', start_token))
            except Exception as e:
                self._log(f"Error resetting UI: {e}", "warning")

    def _stop_translation(self):
        """Stop the translation process"""
        if (
            self.is_running
            or getattr(self, '_graceful_stop_pending', False)
            or getattr(self, '_translation_startup_pending', False)
        ):
            startup_pending = bool(getattr(self, '_translation_startup_pending', False))
            if startup_pending:
                self._translation_start_cancel_requested = True
            # Check if graceful stop is enabled (from main GUI settings)
            graceful_stop = False
            try:
                if hasattr(self, 'main_gui') and self.main_gui:
                    graceful_stop = getattr(self.main_gui, 'graceful_stop_var', False)
            except Exception:
                pass
            if startup_pending:
                graceful_stop = False
            
            # Double-click detection for force stop during graceful stop
            import time as _time
            current_time = _time.time()
            if not hasattr(self, '_stop_click_times'):
                self._stop_click_times = []
            
            # Add current click and remove clicks older than 1 second
            self._stop_click_times.append(current_time)
            self._stop_click_times = [t for t in self._stop_click_times if current_time - t < 1.0]
            
            # Force stop on double-click (2+ clicks within 1s) while graceful stop is pending
            if len(self._stop_click_times) >= 2 and getattr(self, '_graceful_stop_pending', False):
                self._log("⚡ Double-click detected — forcing immediate stop!", "warning")
                graceful_stop = False  # Override to force immediate stop
                self._stop_click_times = []  # Reset click counter
                self._graceful_stop_pending = False
            
            # Set graceful stop mode in environment so API client knows to show logs
            os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'
            
            # Set graceful_stop_active on main GUI so stop_callbacks return False during graceful stop
            try:
                if hasattr(self, 'main_gui') and self.main_gui:
                    self.main_gui.graceful_stop_active = graceful_stop
            except Exception:
                pass
            
            # For graceful stop: keep is_running True so the toggle routes back here on next click
            # For immediate/force stop: set is_running False immediately
            if graceful_stop:
                self._graceful_stop_pending = True
            else:
                self.is_running = False
                self._graceful_stop_pending = False
            
            # For graceful stop, don't set stop_flag yet - let current image complete
            # The GRACEFUL_STOP_COMPLETED flag will be set when the API call finishes
            # For immediate stop, set stop_flag immediately
            if not graceful_stop:
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
            
            # Update button to show "Stopping..." or "Finishing..." state
            try:
                if hasattr(self, 'start_button') and self.start_button:
                    # Clear focus from button to prevent scroll
                    self.start_button.clearFocus()
                    
                    if graceful_stop:
                        # Graceful stop — keep button enabled so user can double-click to force stop
                        self.start_button.setEnabled(True)
                    else:
                        # Immediate/force stop — disable button
                        self.start_button.setEnabled(False)
                    # Update text label instead of button text
                    if hasattr(self, 'start_button_text'):
                        if graceful_stop:
                            self.start_button_text.setText("Finishing...")
                        else:
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
            
            # For graceful stop: DON'T abort in-flight API calls, let them finish
            # For immediate stop: abort everything aggressively
            if not graceful_stop:
                # Set global cancellation flags for coordinated stopping
                self.set_global_cancellation(True)
                
                # Also propagate to MangaTranslator class
                try:
                    from manga_translator import MangaTranslator
                    MangaTranslator.set_global_cancellation(True)
                    # CRITICAL: Force-release all pool checkouts so resources are available
                    # for subsequent translations (interrupted translations leave stale checkouts)
                    # Note: restart_workers=False (default) to avoid GUI lag. Stuck workers
                    # will be lazily restarted on next use via _mp_load_model retry logic.
                    MangaTranslator.force_release_all_pool_checkouts()
                    
                    # CRITICAL: Trigger UI pool tracker update after releasing checkouts
                    try:
                        if hasattr(self, 'update_queue') and self.update_queue:
                            self.update_queue.put(('update_pool_tracker',))
                    except Exception:
                        pass
                except ImportError:
                    pass
                
                # Also propagate to UnifiedClient if available
                try:
                    from unified_api_client import UnifiedClient
                    UnifiedClient.set_global_cancellation(True)
                except ImportError:
                    pass
                
                # Hard cancel: close active HTTP sessions to abort in-flight requests
                try:
                    import unified_api_client
                    if hasattr(unified_api_client, 'hard_cancel_all'):
                        unified_api_client.hard_cancel_all()
                except Exception:
                    pass
                
                # Terminate any background processes (like stuck streaming instances)
                try:
                    import psutil
                    current_process = psutil.Process(os.getpid())
                    children = current_process.children(recursive=True)
                    
                    # Collect inpainter worker PIDs to protect from kill.
                    # These workers are expensive to restart (DLL + model reload);
                    # they survive stop/resume and are reused on next translation.
                    _protected_pids = set()
                    try:
                        from manga_translator import MangaTranslator
                        if hasattr(MangaTranslator, '_inpaint_pool') and MangaTranslator._inpaint_pool:
                            for _key, _rec in MangaTranslator._inpaint_pool.items():
                                if _rec and 'spares' in _rec:
                                    for _inp in _rec['spares']:
                                        if _inp and getattr(_inp, '_mp_worker', None):
                                            try:
                                                _pid = _inp._mp_worker.pid
                                                if _pid:
                                                    _protected_pids.add(_pid)
                                            except Exception:
                                                pass
                    except Exception:
                        pass
                    
                    # Filter out important GUI processes we should keep
                    processes_to_terminate = []
                    for child in children:
                        try:
                            # Get process name to avoid killing important processes
                            name = child.name().lower()
                            # Only terminate Python/script processes, not system processes
                            # Skip inpainter workers — they survive stop and are reused
                            if ('python' in name or 'glossarion' in name) and child.pid not in _protected_pids:
                                processes_to_terminate.append(child)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    if processes_to_terminate:
                        self._log(f"🔧 Terminating {len(processes_to_terminate)} background process(es)...", "info")
                        for proc in processes_to_terminate:
                            try:
                                proc.terminate()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # Wait for processes to terminate, then force kill if needed
                        gone, alive = psutil.wait_procs(processes_to_terminate, timeout=2)
                        for proc in alive:
                            try:
                                proc.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                except Exception as e:
                    self._log(f"Warning: Child process termination failed: {e}", "debug")
            
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
            
            # Log message depends on stop mode
            if graceful_stop:
                try:
                    wait_for_chunks = os.environ.get('WAIT_FOR_CHUNKS') == '1'
                except Exception:
                    wait_for_chunks = False
                if wait_for_chunks:
                    self._log("\n⏳ Graceful stop — waiting for in-flight API calls to complete...", "info")
                else:
                    self._log("\n🛑 Stop requested — cancelling queued/in-flight API calls (WAIT_FOR_CHUNKS=0)", "info")
                # For graceful stop: DON'T schedule UI reset here
                # The UI will be reset when the API call actually completes
                # via the GRACEFUL_STOP_COMPLETED check in the worker loop
            else:
                self._log("\n⏹️ Translation stopped by user", "warning")
                # Schedule UI reset after a delay for immediate stop only
                try:
                    from PySide6.QtCore import QTimer
                    # Wait 2 seconds for cleanup to allow "Stopping..." to be visible
                    stopped_token = getattr(self, '_translation_start_token', None)
                    QTimer.singleShot(
                        2000,
                        lambda token=stopped_token: self._reset_ui_state(token),
                    )
                except Exception:
                    pass

    def _generate_manga_glossary_from_ocr_pages(self, ocr_pages: List[Dict[str, Any]]) -> Optional[str]:
        """Generate, save, and return prompt-ready glossary text from OCR pages."""
        combined_text = self._build_manga_glossary_input(ocr_pages)
        if not combined_text.strip():
            self._log("⚠️ No OCR text available for manga glossary generation", "warning")
            return None

        previous_env = self._prepare_manga_glossary_env()
        try:
            from extract_glossary_from_epub import (
                build_prompt,
                parse_api_response,
                validate_extracted_entry,
                skip_duplicate_entries,
                save_glossary_csv,
                save_glossary_json,
            )
            from TransateKRtoEN import send_with_interrupt

            system_prompt, user_prompt = build_prompt(combined_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            temperature = float(self.main_gui.config.get('manual_glossary_temperature', 0.1))
            glossary_token_cfg = self.main_gui.config.get('glossary_max_output_tokens', -1)
            if str(glossary_token_cfg) == '-1':
                max_tokens = int(getattr(self.main_gui, 'max_output_tokens', 8192) or 8192)
            else:
                max_tokens = int(glossary_token_cfg)

            total_regions = sum(len(page.get('texts', [])) for page in ocr_pages)
            self._log(
                f"📚 Generating manga glossary from {len(ocr_pages)} pages / {total_regions} OCR regions",
                "info",
            )

            client = self.main_gui.client
            previous_client_context = getattr(client, 'context', None)
            previous_session_context = getattr(client, 'current_session_context', None)
            try:
                response, _finish_reason, _raw_obj = send_with_interrupt(
                    messages=messages,
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_check_fn=lambda: self.stop_flag.is_set(),
                    context='glossary',
                )
            finally:
                try:
                    client.context = previous_client_context
                    client.current_session_context = previous_session_context
                except Exception:
                    pass
            response_text = self._normalize_manga_api_response_text(response)
            self._log(f"📥 Manga glossary response received ({len(response_text)} chars)", "info")

            parsed_entries = parse_api_response(response_text)
            entries = []
            for entry in parsed_entries:
                try:
                    if validate_extracted_entry(entry):
                        entries.append(entry)
                except Exception:
                    continue

            self._save_manga_glossary_generation_debug(
                ocr_pages,
                combined_text,
                response_text,
                parsed_entries,
                entries,
            )

            if entries:
                try:
                    manga_glossary_path = self._manga_glossary_output_json_path()
                    entries = skip_duplicate_entries(
                        entries,
                        output_dir=os.path.dirname(manga_glossary_path),
                        glossary_path=manga_glossary_path,
                    )
                except Exception as dedupe_err:
                    self._log(f"⚠️ Manga glossary dedupe skipped: {dedupe_err}", "warning")

            output_json = self._manga_glossary_output_json_path()
            output_csv = output_json.replace('.json', '.csv')
            backup_json = self._manga_glossary_backup_json_path()
            backup_csv = backup_json.replace('.json', '.csv')
            output_legacy_json = os.environ.get('GLOSSARY_OUTPUT_LEGACY_JSON', '0') == '1'

            prompt_glossary_text = ""
            if entries:
                if output_legacy_json:
                    save_glossary_json(entries, output_json)
                    save_glossary_json(entries, backup_json)
                save_glossary_csv(entries, output_json)
                save_glossary_csv(entries, backup_json)
                if os.path.exists(output_csv):
                    with open(output_csv, 'r', encoding='utf-8') as f:
                        prompt_glossary_text = f.read().strip()
                saved_paths = [output_csv, backup_csv]
                if output_legacy_json and os.path.exists(output_json):
                    saved_paths.append(output_json)
                if output_legacy_json and os.path.exists(backup_json):
                    saved_paths.append(backup_json)
                self._log(f"✅ Manga glossary saved: {', '.join(saved_paths)}", "success")
                self._log(f"📚 Manga glossary entries: {len(entries)}", "success")
            else:
                self._log("⚠️ Manga glossary parser found no valid entries; continuing without embedded glossary", "warning")

            self.manga_generated_glossary_text = prompt_glossary_text
            self.manga_generated_glossary_entries = entries
            self.manga_generated_glossary_path = output_json if output_legacy_json and os.path.exists(output_json) else output_csv
            self.manga_glossary_auto_load_suppressed = False
            self.manga_glossary_auto_load_suppressed_root = ''
            self.manga_loaded_glossary_text = prompt_glossary_text
            self.main_gui.config['manga_generated_glossary_path'] = self.manga_generated_glossary_path
            self.main_gui.config['manga_glossary_auto_load_suppressed'] = False
            self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = ''
            setattr(self.main_gui, 'manga_generated_glossary_text', prompt_glossary_text)
            setattr(self.main_gui, 'manga_generated_glossary_entries', entries)
            setattr(self.main_gui, 'manga_generated_glossary_path', self.manga_generated_glossary_path)
            if hasattr(self, 'translator') and self.translator:
                self.translator.manga_generated_glossary_text = prompt_glossary_text
                self.translator._manga_glossary_prompt_logged = False
            self._update_manga_glossary_status_label()
            try:
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
            except Exception:
                pass
            return prompt_glossary_text

        except Exception as e:
            self._log(f"❌ Manga glossary generation failed: {e}", "error")
            self._log(traceback.format_exc(), "debug")
            return None
        finally:
            self._restore_manga_glossary_env(previous_env)

    def _handle_manga_translation_result(self, filepath: str, result: Dict[str, Any]) -> None:
        """Apply the normal sequential result handling for glossary workflow pages."""
        filename = os.path.basename(filepath)
        if result.get('interrupted', False):
            self._log(f"⏸️ Translation of {filename} was interrupted", "warning")
            self.failed_files += 1
            return

        if not result.get('success', False):
            self.failed_files += 1
            errors = '\n'.join(result.get('errors', ['Unknown error']))
            self._log(f"❌ Translation failed: {filename}\n{errors}", "error")
            return

        output_exists = result.get('output_path') and os.path.exists(result.get('output_path', ''))
        has_translations = any(r.get('translated_text', '') for r in result.get('regions', []))
        if output_exists and has_translations:
            self.completed_files += 1
            self._log(f"✅ Translation completed: {filename}", "success")

            if result.get('cleaned_image_path'):
                try:
                    if hasattr(self, 'image_state_manager') and self.image_state_manager:
                        self.image_state_manager.update_state(filepath, {
                            'cleaned_image_path': result.get('cleaned_image_path')
                        })
                except Exception as e:
                    print(f"[CLEANED] Failed to save cleaned path to state: {e}")

            if hasattr(self, 'image_preview_widget'):
                current_path = self.image_preview_widget.current_image_path
                norm_current = os.path.normpath(current_path) if current_path else None
                norm_source = os.path.normpath(filepath)
                is_current = (
                    norm_current == norm_source or
                    (norm_current and os.path.dirname(norm_current).endswith(f"{os.path.splitext(os.path.basename(filepath))[0]}_translated"))
                )
                if is_current:
                    self.update_queue.put(('preview_update', {
                        'translated_path': result.get('output_path'),
                        'source_path': filepath,
                        'switch_to_output': True,
                    }))
                    self._log("🖼️ Queued preview update to show translated image", "debug")
        else:
            self.failed_files += 1
            if not output_exists:
                self._log(f"❌ Output file not created: {filename}", "error")
            else:
                self._log(f"❌ No text was translated: {filename}", "error")

    def _create_manga_glossary_worker_translator(self, glossary_text: str) -> MangaTranslator:
        """Create an isolated translator for one parallel manga glossary page."""
        translator = MangaTranslator(
            self._build_manga_worker_ocr_config(),
            getattr(self.main_gui, 'client', None),
            self.main_gui,
            log_callback=self._log
        )
        self._configure_manga_ocr_io(translator)
        translator.set_stop_flag(self.stop_flag)
        translator.manga_generated_glossary_text = glossary_text
        translator._manga_glossary_prompt_logged = False

        try:
            enabled = (
                bool(self.context_checkbox.isChecked())
                if hasattr(self, 'context_checkbox')
                else bool(getattr(self, 'full_page_context_value', True))
            )
            translator.set_full_page_context(
                enabled=enabled,
                custom_prompt=getattr(self, 'full_page_context_prompt', None)
            )
        except Exception:
            pass

        try:
            advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
            if advanced.get('parallel_panel_translation', False):
                panel_workers = max(1, int(advanced.get('panel_max_workers', 2)))
                translator.manga_settings.setdefault('advanced', {})['parallel_processing'] = True
                translator.manga_settings.setdefault('advanced', {})['max_workers'] = panel_workers
                translator.parallel_processing = True
                translator.max_workers = panel_workers
        except Exception as e:
            self._log(f"Warning: failed to configure manga glossary worker parallel settings: {e}", "warning")

        return translator

    def _run_parallel_manga_glossary_ocr_pass(self):
        """Collect OCR for manga glossary generation using page-level concurrency."""
        advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
        panel_parallel = bool(advanced.get('parallel_panel_translation', False))
        files = self._current_manga_processing_files()
        try:
            requested_workers = max(1, int(advanced.get('panel_max_workers', 2)))
        except Exception:
            requested_workers = 2

        effective_workers = min(requested_workers, len(files)) if panel_parallel and len(files) > 1 else 1
        if effective_workers <= 1:
            return None

        self._log(f"Parallel manga glossary OCR ENABLED ({effective_workers} workers)", "info")

        progress_lock = threading.Lock()
        counters = {'started': 0, 'done': 0}
        total = len(files)
        ocr_pages: List[Dict[str, Any]] = []
        precomputed_regions: Dict[str, List[Any]] = {}

        def process_single(index: int, filepath: str) -> bool:
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                return False

            filename = os.path.basename(filepath)
            translator = None
            try:
                translator = self._create_manga_glossary_worker_translator("")
                translator.skip_inpainting = True
                translator.use_cloud_inpainting = False
                translator.inpaint_mode = 'skip'
                output_path = self._get_manga_output_path_for_file(filepath)

                with progress_lock:
                    counters['started'] += 1
                    self.current_file_index = index
                    self._monitor_translation_output(filepath)
                    self._update_current_file(filename)
                    self._update_progress(
                        counters['done'],
                        total,
                        f"OCR for glossary {counters['started']}/{total}: {filename}"
                    )

                result = translator.process_image(
                    filepath,
                    output_path,
                    batch_index=index + 1,
                    batch_total=total,
                    ocr_only=True
                )

                with progress_lock:
                    counters['done'] += 1
                    if result.get('interrupted', False):
                        self.failed_files += 1
                        self._log(f"Translation of {filename} was interrupted during glossary OCR", "warning")
                    elif not result.get('success', False):
                        self.failed_files += 1
                        errors = '\n'.join(result.get('errors', ['Unknown OCR error']))
                        self._log(f"OCR failed for glossary: {filename}\n{errors}", "error")
                    else:
                        region_objects = result.get('_region_objects') or []
                        texts = [
                            getattr(region, 'text', '')
                            for region in region_objects
                            if getattr(region, 'text', '').strip()
                        ]
                        precomputed_regions[filepath] = region_objects
                        ocr_pages.append({
                            'index': index + 1,
                            'path': filepath,
                            'regions': region_objects,
                            'texts': texts,
                        })
                        self._log(f"OCR captured for glossary: {filename} ({len(texts)} text regions)", "success")
                    self._update_progress(
                        counters['done'],
                        total,
                        f"OCR for glossary complete {counters['done']}/{total}"
                    )
                return bool(result.get('success', False))
            except Exception as e:
                with progress_lock:
                    self.failed_files += 1
                    counters['done'] += 1
                    self._log(f"OCR glossary pass error for {filename}: {e}", "error")
                    self._log(traceback.format_exc(), "debug")
                    self._update_progress(
                        counters['done'],
                        total,
                        f"OCR for glossary complete {counters['done']}/{total}"
                    )
                return False
            finally:
                try:
                    if translator:
                        if hasattr(translator, '_return_inpainter_to_pool'):
                            translator._return_inpainter_to_pool()
                        if hasattr(translator, '_return_bubble_detector_to_pool'):
                            translator._return_bubble_detector_to_pool()
                        if hasattr(translator, 'clear_internal_state'):
                            translator.clear_internal_state()
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = []
            try:
                ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {})
                stagger_ms = int(ocr_settings.get('ocr_request_delay_ms', 100))
            except Exception:
                stagger_ms = 100

            for index, filepath in enumerate(files):
                if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                    break
                futures.append(executor.submit(process_single, index, filepath))
                if stagger_ms > 0:
                    time.sleep(stagger_ms / 1000.0)

            for future in concurrent.futures.as_completed(futures):
                if self.stop_flag.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                try:
                    future.result()
                except Exception:
                    pass

        ocr_pages.sort(key=lambda page: page.get('index', 0))
        return ocr_pages, precomputed_regions

    def _run_parallel_manga_glossary_translation(
        self,
        glossary_text: str,
        *,
        precomputed_regions: Optional[Dict[str, List[Any]]] = None,
        progress_label: str = "Translating with glossary"
    ) -> bool:
        """Run a manga glossary translation pass using panel-level page concurrency."""
        advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
        panel_parallel = bool(advanced.get('parallel_panel_translation', False))
        files = self._current_manga_processing_files()
        try:
            requested_workers = max(1, int(advanced.get('panel_max_workers', 2)))
        except Exception:
            requested_workers = 2

        effective_workers = min(requested_workers, len(files)) if panel_parallel and len(files) > 1 else 1
        if effective_workers <= 1:
            return False

        self._log(f"Parallel manga glossary translation ENABLED ({effective_workers} workers)", "info")

        progress_lock = threading.Lock()
        counters = {'started': 0, 'done': 0}
        total = len(files)

        def process_single(index: int, filepath: str) -> bool:
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                return False

            filename = os.path.basename(filepath)
            regions = None
            if precomputed_regions is not None:
                regions = precomputed_regions.get(filepath) or []
                if not regions:
                    with progress_lock:
                        self.failed_files += 1
                        counters['done'] += 1
                        self._log(f"No precomputed OCR regions for {filename}; skipping translation", "warning")
                        self._update_progress(
                            counters['done'],
                            total,
                            f"{progress_label} complete {counters['done']}/{total}"
                        )
                    return False

            translator = None
            try:
                translator = self._create_manga_glossary_worker_translator(glossary_text)
                output_path = self._get_manga_output_path_for_file(filepath)

                with progress_lock:
                    counters['started'] += 1
                    self.current_file_index = index
                    self._monitor_translation_output(filepath)
                    self._update_current_file(filename)
                    self._update_progress(
                        counters['done'],
                        total,
                        f"{progress_label} {counters['started']}/{total}: {filename}"
                    )

                result = translator.process_image(
                    filepath,
                    output_path,
                    batch_index=index + 1,
                    batch_total=total,
                    precomputed_regions=regions
                )

                with progress_lock:
                    self._handle_manga_translation_result(filepath, result)
                    counters['done'] += 1
                    self._update_progress(
                        counters['done'],
                        total,
                        f"{progress_label} complete {counters['done']}/{total}"
                    )
                return bool(result.get('success', False))
            except Exception as e:
                with progress_lock:
                    self.failed_files += 1
                    counters['done'] += 1
                    self._log(f"Translation error for {filename}: {e}", "error")
                    self._log(traceback.format_exc(), "debug")
                    self._update_progress(
                        counters['done'],
                        total,
                        f"{progress_label} complete {counters['done']}/{total}"
                    )
                return False
            finally:
                try:
                    if translator:
                        if hasattr(translator, '_return_inpainter_to_pool'):
                            translator._return_inpainter_to_pool()
                        if hasattr(translator, '_return_bubble_detector_to_pool'):
                            translator._return_bubble_detector_to_pool()
                        if hasattr(translator, 'clear_internal_state'):
                            translator.clear_internal_state()
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = []
            try:
                stagger_ms = int(advanced.get('panel_start_stagger_ms', 30))
            except Exception:
                stagger_ms = 30

            for index, filepath in enumerate(files):
                if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                    break
                futures.append(executor.submit(process_single, index, filepath))
                if stagger_ms > 0:
                    time.sleep(stagger_ms / 1000.0)

            for future in concurrent.futures.as_completed(futures):
                if self.stop_flag.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                try:
                    future.result()
                except Exception:
                    pass

        try:
            self._finalize_cbz_jobs()
        except Exception:
            pass
        return True

    def _run_manga_loaded_glossary_translation(self, glossary_text: str) -> None:
        """Translate selected pages using a loaded custom glossary without generating a new one."""
        glossary_text = (glossary_text or '').strip()
        if not glossary_text:
            self._log("⚠️ No loaded manga glossary text available", "warning")
            return

        self.manga_generated_glossary_text = glossary_text
        setattr(self.main_gui, 'manga_generated_glossary_text', glossary_text)
        if hasattr(self, 'translator') and self.translator:
            self.translator.manga_generated_glossary_text = glossary_text
            self.translator._manga_glossary_prompt_logged = False

        files = self._current_manga_processing_files()
        total = len(files)
        entry_count = sum(1 for line in glossary_text.splitlines() if line.lstrip().startswith("* "))
        source_path = getattr(self, 'manga_custom_glossary_path', '') or self.main_gui.config.get('manga_custom_glossary_path', '')
        source_name = os.path.basename(source_path) if source_path else "loaded glossary"
        self._log(f"📚 Translating with custom manga glossary: {source_name} ({entry_count} entries)", "info")
        if self._run_parallel_manga_glossary_translation(
            glossary_text,
            progress_label="Translating with loaded glossary"
        ):
            return

        for index, filepath in enumerate(files):
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                self._log("⏹️ Custom glossary translation pass stopped", "warning")
                break

            if hasattr(self.translator, 'reset_for_new_image'):
                self.translator.reset_for_new_image()
            self.translator.manga_generated_glossary_text = glossary_text

            filename = os.path.basename(filepath)
            self.current_file_index = index
            self._monitor_translation_output(filepath)
            self._update_current_file(filename)
            self._update_progress(index, total, f"Translating with loaded glossary {index + 1}/{total}: {filename}")

            try:
                output_path = self._get_manga_output_path_for_file(filepath)
                result = self.translator.process_image(
                    filepath,
                    output_path,
                    batch_index=index + 1,
                    batch_total=total
                )
                self._handle_manga_translation_result(filepath, result)
                time.sleep(0.1)
            except Exception as e:
                self.failed_files += 1
                self._log(f"❌ Translation error for {filename}: {e}", "error")
                self._log(traceback.format_exc(), "debug")

    def _run_manga_glossary_workflow(self, glossary_only_run: bool = False) -> None:
        """Run manga glossary workflow once per computed process group."""
        all_files = self._current_manga_processing_files()
        groups = self._manga_process_groups_for_paths(all_files)
        if not groups:
            self._log("No manga files selected for glossary workflow", "warning")
            return

        original_processing_files = getattr(self, '_manga_processing_files', None)
        original_active_group = getattr(self, '_manga_active_process_group', None)
        custom_glossary_path = getattr(self, 'manga_custom_glossary_path', '') or self.main_gui.config.get('manga_custom_glossary_path', '')

        try:
            for group_index, group in enumerate(groups):
                if self.stop_flag.is_set() or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                    break

                group_files = list(group.get('files', []) or [])
                if not group_files:
                    continue

                self._manga_active_process_group = group
                self._manga_processing_files = group_files
                self.total_files = len(group_files)
                self.current_file_index = 0

                if len(groups) > 1:
                    self._log(
                        f"📚 Manga process {group_index + 1}/{len(groups)}: {group.get('root', '')} ({len(group_files)} images)",
                        "info",
                    )

                if not custom_glossary_path:
                    self.manga_loaded_glossary_text = ''
                    self.manga_generated_glossary_text = ''
                    setattr(self.main_gui, 'manga_generated_glossary_text', '')
                    if getattr(self, 'translator', None):
                        self.translator.manga_generated_glossary_text = ''
                        self.translator._manga_glossary_prompt_logged = False

                loaded_glossary_text = "" if glossary_only_run else self._get_loaded_manga_glossary_text()
                if loaded_glossary_text:
                    self._run_manga_loaded_glossary_translation(loaded_glossary_text)
                else:
                    self._run_manga_glossary_batch(glossary_only=glossary_only_run)
        finally:
            self._manga_processing_files = original_processing_files
            self._manga_active_process_group = original_active_group
            self.total_files = len(all_files)
            self._update_manga_glossary_status_label()

    def _run_manga_glossary_batch(self, glossary_only: bool = False) -> None:
        """Run OCR for all pages, generate one glossary, then optionally translate."""
        self._log("📚 Manga glossary workflow enabled", "info")
        if getattr(self, 'translator', None):
            self.translator.manga_generated_glossary_text = ""
        setattr(self.main_gui, 'manga_generated_glossary_text', "")

        ocr_pages: List[Dict[str, Any]] = []
        precomputed_regions: Dict[str, List[Any]] = {}
        files = self._current_manga_processing_files()
        total = len(files)

        parallel_ocr_result = self._run_parallel_manga_glossary_ocr_pass()
        if parallel_ocr_result is not None:
            ocr_pages, precomputed_regions = parallel_ocr_result
            if self.stop_flag.is_set() or not ocr_pages:
                return

            self._update_progress(len(ocr_pages), total, "Generating manga glossary...")
            glossary_text = self._generate_manga_glossary_from_ocr_pages(ocr_pages)
            if glossary_text is None:
                self._log("⚠️ Manga glossary generation failed; translation pass skipped", "warning")
                return

            if glossary_only:
                self.completed_files += len(ocr_pages)
                self._update_progress(total, total, f"Glossary generated from {len(ocr_pages)} pages")
                return

            self._log("➡️ Starting translation pass with generated manga glossary", "info")
            if self._run_parallel_manga_glossary_translation(
                glossary_text,
                precomputed_regions=precomputed_regions,
                progress_label="Translating with glossary"
            ):
                return

        original_skip_inpainting = getattr(self.translator, 'skip_inpainting', False)
        try:
            self.translator.skip_inpainting = True
        except Exception:
            pass

        for index, filepath in enumerate(files):
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                self._log("⏹️ Manga glossary OCR pass stopped", "warning")
                break

            if hasattr(self.translator, 'reset_for_new_image'):
                self.translator.reset_for_new_image()

            filename = os.path.basename(filepath)
            self.current_file_index = index
            self._monitor_translation_output(filepath)
            self._update_current_file(filename)
            self._update_progress(index, total, f"OCR for glossary {index + 1}/{total}: {filename}")

            try:
                output_path = self._get_manga_output_path_for_file(filepath)
                result = self.translator.process_image(
                    filepath,
                    output_path,
                    batch_index=index + 1,
                    batch_total=total,
                    ocr_only=True
                )
                if result.get('interrupted', False):
                    self.failed_files += 1
                    if self.stop_flag.is_set():
                        break
                    continue
                if not result.get('success', False):
                    self.failed_files += 1
                    errors = '\n'.join(result.get('errors', ['Unknown OCR error']))
                    self._log(f"❌ OCR failed for glossary: {filename}\n{errors}", "error")
                    continue

                region_objects = result.get('_region_objects') or []
                texts = [getattr(region, 'text', '') for region in region_objects if getattr(region, 'text', '').strip()]
                precomputed_regions[filepath] = region_objects
                ocr_pages.append({
                    'index': index + 1,
                    'path': filepath,
                    'regions': region_objects,
                    'texts': texts,
                })
                self._log(f"✅ OCR captured for glossary: {filename} ({len(texts)} text regions)", "success")
            except Exception as e:
                self.failed_files += 1
                self._log(f"❌ OCR glossary pass error for {filename}: {e}", "error")
                self._log(traceback.format_exc(), "debug")

        try:
            self.translator.skip_inpainting = original_skip_inpainting
        except Exception:
            pass

        if self.stop_flag.is_set() or not ocr_pages:
            return

        self._update_progress(len(ocr_pages), total, "Generating manga glossary...")
        glossary_text = self._generate_manga_glossary_from_ocr_pages(ocr_pages)
        if glossary_text is None:
            self._log("⚠️ Manga glossary generation failed; translation pass skipped", "warning")
            return

        if glossary_only:
            self.completed_files += len(ocr_pages)
            self._update_progress(total, total, f"Glossary generated from {len(ocr_pages)} pages")
            return

        self._log("➡️ Starting translation pass with generated manga glossary", "info")
        if self._run_parallel_manga_glossary_translation(
            glossary_text,
            precomputed_regions=precomputed_regions,
            progress_label="Translating with glossary"
        ):
            return

        for index, filepath in enumerate(files):
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1' or self.stop_flag.is_set():
                self._log("⏹️ Manga glossary translation pass stopped", "warning")
                break

            regions = precomputed_regions.get(filepath) or []
            filename = os.path.basename(filepath)
            self.current_file_index = index
            self._monitor_translation_output(filepath)
            self._update_current_file(filename)
            self._update_progress(index, total, f"Translating with glossary {index + 1}/{total}: {filename}")

            if not regions:
                self.failed_files += 1
                self._log(f"⚠️ No precomputed OCR regions for {filename}; skipping translation", "warning")
                continue

            try:
                if hasattr(self.translator, 'reset_for_new_image'):
                    self.translator.reset_for_new_image()
                self.translator.manga_generated_glossary_text = glossary_text
                output_path = self._get_manga_output_path_for_file(filepath)
                result = self.translator.process_image(
                    filepath,
                    output_path,
                    batch_index=index + 1,
                    batch_total=total,
                    precomputed_regions=regions
                )
                self._handle_manga_translation_result(filepath, result)
                time.sleep(0.1)
            except Exception as e:
                self.failed_files += 1
                self._log(f"❌ Translation error for {filename}: {e}", "error")
                self._log(traceback.format_exc(), "debug")


# ---------------------------------------------------------------------------
# Headless batch runner (the MANGA job adapter's entry point)
# ---------------------------------------------------------------------------


class MangaRunError(RuntimeError):
    """The desktop Start button would have refused (no files, invalid image range)."""


class HeadlessMangaRunner(MangaRunMixin, HeadlessMangaState):
    """The manga translator tab without Qt: Start / Stop / batch worker on a job thread.

    ``main_gui`` is the job's ``HeadlessOwner`` (the duck-typed main GUI ``MangaTranslator``
    and the moved tab code read). The run is the desktop's: :meth:`run` replays the GUI-free
    lines of ``MangaTranslationTab._start_translation`` (the Start click: range check, the
    automatic OCR export, imported-OCR matching, the run token and flags), then calls
    ``_start_translation_heavy`` on the calling thread (desktop: its "MangaStartHeavy"
    thread), which launches ``_translation_worker`` on a thread exactly like the desktop when
    the main GUI has no executor; :meth:`run` waits for it, forwarding the tab's update
    queue (progress, current file, UI state) to ``progress`` / the job host.

    Selection (what ``services.manga`` records for a run): *files* (image paths in run
    order; CBZ archives already extracted, mapped by *cbz_jobs* / *cbz_image_to_job*),
    *folder_roots*, *split_first_level*, *image_range* (1-based, visible order), *skipped*
    (paths not to process). *glossary_only* runs the OCR + glossary pass only (the
    "Generate glossary" button). *output_root* sets OUTPUT_DIRECTORY when the env has none.
    """

    def __init__(self, main_gui, *, host=None, files=None, image_range='', folder_roots=None,
                 split_first_level=None, skipped=None, cbz_jobs=None, cbz_image_to_job=None,
                 glossary_only=False, progress=None, output_root=None, ocr_provider=None,
                 image_state_manager=None, stop_event=None, **_ignored):
        super().__init__(main_gui, host=host, ocr_provider=ocr_provider, image_state_manager=image_state_manager)
        self._progress_callback = progress if callable(progress) else None
        self._external_stop_event = stop_event
        self._started_files = []
        self._run_files = []
        if output_root and not os.environ.get('OUTPUT_DIRECTORY'):
            os.environ['OUTPUT_DIRECTORY'] = os.path.abspath(output_root)
        self.set_selection(files or [], image_range=image_range, folder_roots=folder_roots,
                           split_first_level=split_first_level, skipped=skipped, cbz_jobs=cbz_jobs,
                           cbz_image_to_job=cbz_image_to_job)
        self._manga_glossary_only_run = bool(glossary_only)

    # ---- selection --------------------------------------------------------------------------
    def set_selection(self, files, *, image_range=None, folder_roots=None, split_first_level=None,
                      skipped=None, cbz_jobs=None, cbz_image_to_job=None):
        """Replace the file list (the desktop list widget) with *files*, in this order."""
        self.selected_files = [os.fspath(path) for path in files if path]
        if image_range is not None:
            self.manga_image_range_value = str(image_range or '').strip()
        if folder_roots is not None:
            self.manga_selected_folder_roots = [os.path.abspath(p) for p in folder_roots if p]
        if split_first_level is not None:
            self.manga_split_first_level_subfolders_value = bool(split_first_level)
        if skipped is not None:
            self.skipped_processing_files = {self._skip_key_for_path(p) for p in skipped if p}
        if cbz_jobs is not None:
            self.cbz_jobs = dict(cbz_jobs)
        if cbz_image_to_job is not None:
            self.cbz_image_to_job = dict(cbz_image_to_job)
        self.file_listbox.setCurrentRow(0 if self.selected_files else -1)

    # ---- tab callbacks ----------------------------------------------------------------------
    def _monitor_translation_output(self, image_path):
        """Desktop: watches the page's output folder for the preview. Here: remember the page."""
        self._started_files.append(image_path)
        emit = getattr(getattr(self, 'host', None), 'emit', None)
        if callable(emit):
            try:
                emit('manga_page', image=image_path)
            except Exception:
                pass
        return None

    def _forward_update(self, update):
        kind = update[0] if update else None
        if kind == 'progress':
            _kind, current, total, status = update
            if self._progress_callback is not None:
                self._progress_callback(current, total, label=status, failed=self.failed_files)
            else:
                emit = getattr(getattr(self, 'host', None), 'emit', None)
                if callable(emit):
                    emit('progress', total=total, completed=current, failed=self.failed_files, label=status)
        elif kind == 'current_file':
            emit = getattr(getattr(self, 'host', None), 'emit', None)
            if callable(emit):
                emit('phase', label=str(update[1]))
        elif kind == 'log':
            self._log(update[1], update[2] if len(update) > 2 else 'info')
        elif kind == 'ui_state':
            state = update[1]
            token = update[2] if len(update) > 2 else None
            if state == 'translation_complete':
                self._reset_ui_state(token)
        elif kind == 'call_method':
            _kind, method, args = update
            try:
                method(*args)
            except Exception as exc:
                self._log(f"Deferred call failed: {exc}", "debug")

    def _drain_updates(self):
        while True:
            try:
                update = self.update_queue.get_nowait()
            except Empty:
                return
            try:
                self._forward_update(update)
            except Exception as exc:
                self._log(f"Progress update failed: {exc}", "debug")

    # ---- Start / Stop -----------------------------------------------------------------------
    def _reset_headless_cancellation(self):
        """Clear the stop / cancellation flags a previous run left behind.

        The desktop start does this inside ``_start_translation_heavy`` through
        ``ImageRenderer._reset_cancellation_flags`` (a Qt module, not importable without
        PySide6), so a headless start runs the GUI-free implementation first:
        ``manga_editor_core._reset_cancellation_flags`` (env stop flags, ``stop_flag``,
        MangaTranslator / UnifiedClient / TransateKRtoEN flags, stale pool checkouts), else the
        tab's own ``_reset_global_cancellation``. Calling it again is harmless on the desktop.
        """
        try:
            from manga_editor_core import _reset_cancellation_flags
        except Exception:
            _reset_cancellation_flags = None
        done = False
        if _reset_cancellation_flags is not None:
            try:
                _reset_cancellation_flags(self)
                done = True
            except Exception as exc:
                self._log(f"Cancellation reset failed: {exc}", "debug")
        if not done:
            try:
                self.stop_flag.clear()
                self._reset_global_cancellation()
            except Exception as exc:
                self._log(f"Cancellation reset failed: {exc}", "debug")
        try:
            self.set_global_cancellation(False)
        except Exception:
            pass

    def start(self):
        """The GUI-free lines of ``MangaTranslationTab._start_translation`` + ``_start_translation_heavy``.

        Returns False when the start was refused or aborted (see ``_last_error``).
        """
        if not self.selected_files:
            raise MangaRunError("Please select manga images to translate.")
        processing_files, range_error = self._manga_range_filtered_files()
        if range_error:
            raise MangaRunError(range_error)
        if not processing_files:
            raise MangaRunError("The image range does not include any loaded rows.")
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
        self._run_files = list(processing_files)
        self._started_files = []
        self._reset_headless_cancellation()
        previous_future = getattr(self, 'translation_future', None)
        previous_thread = getattr(self, 'translation_thread', None)
        self._translation_start_token = int(getattr(self, '_translation_start_token', 0) or 0) + 1
        start_token = self._translation_start_token
        self._translation_startup_pending = True
        self._translation_start_cancel_requested = False
        self.is_running = True
        self.translation_thread = None
        self.translation_future = None
        self._start_translation_heavy(previous_future, previous_thread, start_token)
        return bool(self.translation_thread is not None or self.translation_future is not None)

    def wait(self, poll=0.2):
        """Block until the worker finished, forwarding the update queue meanwhile."""
        while True:
            worker = getattr(self, 'translation_thread', None)
            future = getattr(self, 'translation_future', None)
            alive = (worker is not None and worker.is_alive()) or (future is not None and not future.done())
            self._drain_updates()
            if not alive:
                break
            time.sleep(poll)
        self._drain_updates()

    def run(self, files=None, glossary_only=None, **_ignored):
        """Start, wait, and report: ``{ok, completed, failed, total, outputs, cbz_paths, stopped, error}``."""
        if files is not None:
            self.set_selection(files)
        if glossary_only is not None:
            self._manga_glossary_only_run = bool(glossary_only)
        self._last_glossary_only = bool(getattr(self, '_manga_glossary_only_run', False))
        launched = False
        try:
            launched = self.start()
            if launched:
                self.wait()
            else:
                self._drain_updates()
        finally:
            self.is_running = False
        return self.summary(launched=launched)

    def request_stop(self, graceful=None, force=False):
        """The desktop Stop button (``_stop_translation``); *force* is the double-click.

        *graceful* None keeps the owner's ``graceful_stop`` setting (what the desktop reads).
        """
        if graceful is not None and not force:
            try:
                self.main_gui.graceful_stop_var = bool(graceful)
            except Exception:
                pass
        if force:
            # A second click within a second while the graceful stop is pending forces it.
            self._stop_click_times = [time.time()]
            if not getattr(self, '_graceful_stop_pending', False):
                try:
                    self.main_gui.graceful_stop_var = False
                except Exception:
                    pass
        self._stop_translation()

    stop = request_stop

    # ---- results ----------------------------------------------------------------------------
    def outputs(self):
        """Rendered pages of the last run that exist (desktop per-image output routing)."""
        found = []
        for path in self._started_files:
            try:
                output = self._get_manga_output_path_for_file(path)
            except Exception:
                continue
            if output and os.path.isfile(output) and output not in found:
                found.append(output)
        return found

    def cbz_paths(self):
        """CBZ archives the run packaged: per imported CBZ (``_finalize_cbz_jobs``) and the
        "Create CBZ at end" archive (``_create_cbz_from_isolated_folders``)."""
        candidates = []
        for cbz_file in (getattr(self, 'cbz_jobs', {}) or {}):
            base = os.path.splitext(os.path.basename(cbz_file))[0]
            candidates.append(os.path.join(os.path.dirname(cbz_file), f"{base}_translated.cbz"))
        parents = []
        override_dir = ((getattr(self.main_gui, 'config', {}) or {}).get('output_directory', '')
                        or os.environ.get('OUTPUT_DIRECTORY', ''))
        if override_dir and os.path.isdir(override_dir):
            parents.append(override_dir)
        parents.extend(os.path.dirname(path) for path in self._run_files[:1])
        for parent_dir in parents:
            candidates.append(os.path.join(parent_dir, f"{os.path.basename(parent_dir)}_translated.cbz"))
        paths = []
        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and candidate not in paths:
                paths.append(candidate)
        return paths

    def summary(self, launched=True):
        stopped = bool(self.stop_flag.is_set()) or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1'
        error = '' if launched else (self._last_error or 'manga run start aborted')
        outputs = self.outputs()
        glossary_only = bool(getattr(self, '_last_glossary_only', False))
        ok = bool(launched) and not stopped and (self.failed_files == 0 or bool(outputs))
        return {
            'ok': ok,
            'completed': int(self.completed_files),
            'failed': int(self.failed_files),
            'total': int(self.total_files),
            'outputs': outputs,
            'cbz_paths': self.cbz_paths(),
            'stopped': stopped,
            'error': error,
            'glossary_path': getattr(self, 'manga_generated_glossary_path', '') or '',
            'glossary_only': glossary_only,
        }


def run_manga_batch(owner, **kwargs):
    """``HeadlessMangaRunner(owner, **kwargs).run()`` (see the class)."""
    return HeadlessMangaRunner(owner, **kwargs).run()


__all__ = [
    "HeadlessMangaRunner",
    "MangaRunError",
    "MangaRunMixin",
    "_IS_WINDOWS",
    "_demote_non_main_threads",
    "_lower_current_thread_priority_and_affinity",
    "run_manga_batch",
]
