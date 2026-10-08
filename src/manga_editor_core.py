"""The GUI-free half of the manga editor, shared by the desktop and Glossarion Mobile (U8).

Detect / Clean / Recognize / Translate / Translate All, the per-box actions, Save & Update
Overlay (re-render) and the per-image editor state of the desktop manga tab
(``ImageRenderer`` + ``manga_image_preview`` + ``manga_integration``) live here.

Layout
------
* **Moved verbatim from ImageRenderer.py** (module functions that take the manga tab as
  ``self``): ``EDITOR_FUNCTIONS``. Nine carry listed edits (``EDITED_FUNCTIONS``):
  Qt-only lines became hooks (``_style_recognized_rectangle``,
  ``_confirm_clean_excluded_rectangle``, ``_schedule_rendered_output_refresh``), the editor
  OCR's ``from google.cloud import vision`` falls back to ``google_vision_rest`` when the SDK
  is missing (``_import_google_vision``), and the six page reads go through
  ``safe_image.cv2_imread`` and the render's page open through ``safe_image.open_page_image``
  (the owner-approved OpenCV decoder gate, 2026-10-08; ImageRenderer imports both, so the
  re-bound bodies resolve them there).
  The desktop does not import these as plain names: ``bind_editor_namespace(globals())`` in
  ImageRenderer re-binds the same code objects to ImageRenderer's namespace, so on the desktop
  they still call ImageRenderer's Qt helpers (box drawing, pulses, overlays, button states) and
  stay patchable as ``ImageRenderer.<name>``, exactly as before the move.
* **Split out of ImageRenderer's Qt dialogs** (bodies verbatim; the dialogs keep their widget
  code and call these): ``_manual_translate_prompt``, ``_apply_ocr_text_edit``,
  ``_apply_translation_text_edit``, ``_persist_translation_text_edit``,
  ``_apply_inpaint_iterations``; out of manga_image_preview's Stop button:
  ``_request_force_stop`` / ``_request_graceful_stop``.
* **Moved verbatim from manga_integration.py**: ``ImageStateManager`` with its spawn worker
  ``_state_manager_worker_process`` (its debug print comes from manga_files_core). The worker
  process is gated by ``mobile_runtime.processes_available()``: on mobile the state stays in
  this process and ``flush()`` writes it.
* **Hook stubs**: GUI-free stand-ins for the Qt-only ImageRenderer helpers the moved code calls.
  They forward to the host's ``_manga_editor_hook(name, *args)`` (``MangaEditorSession``
  answers them with its box model) and do nothing for any other host.
* **Mobile host**: ``MangaEditorSession`` (+ ``EditorBox`` / ``EditorViewer`` /
  ``EditorPreview``, the duck-typed stand-ins for the preview widget the moved code reads) with
  an explicit-parameter API: ``open_page``, ``detect``, ``clean``, ``recognize``, ``translate``,
  ``translate_all``, the per-box actions, ``save_and_update_overlay``, ``stop`` and the OCR
  JSON ``export_ocr`` / ``import_ocr`` (``manga_ocr_io``).

Rules: Python 3.10 compatible; never imports PySide6, translator_gui, dpi_setup,
ImageRenderer or manga_integration (the heavy manga modules are imported lazily, inside the
functions, exactly as ImageRenderer did).
"""

import os
import sys
import json
import copy
import threading
import time
import types
import logging
import traceback
from queue import Queue, Empty
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mobile_runtime
import manga_ocr_io
# ImageStateManager's debug print (manga_integration's helper; manga_files_core holds it since U8)
from manga_files_core import _manga_cmd_debug_print
from safe_image import open_image, open_page_image, cv2_imread

# Same optional import as ImageRenderer (_manga_output_text and _translate_this_text_background use it).
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

_REGION_METADATA_KEYS = ('bubble_type', 'region_type', 'bubble_bounds')


def _import_google_vision():
    """``google.cloud.vision`` (the SDK the desktop ships), or ``google_vision_rest``'s
    SDK-shaped ``vision`` when the SDK is not installed (mobile)."""
    try:
        from google.cloud import vision
    except ImportError:
        from google_vision_rest import vision
    return vision


# ===========================================================================
# Moved verbatim from ImageRenderer.py (EDITOR_FUNCTIONS; see the module docstring)
# ===========================================================================

def _manga_debug_logging_enabled() -> bool:
    return (
        os.environ.get('DEBUG_MODE', '0') == '1'
        or os.environ.get('SHOW_DEBUG_BUTTONS', '0') == '1'
        or os.environ.get('MANGA_DEBUG_MODE', '0') == '1'
        or os.environ.get('DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE', '0') == '1'
    )


def _manga_debug_print(*args, **kwargs):
    try:
        first_arg = str(args[0]) if args else ''
        important = any(term in first_arg.lower() for term in ('error', 'failed', 'failure', 'exception', 'warning'))
        debug_prefixes = (
            '[STATE]',
            '[STATE DEBUG]',
            '[STATE_ISOLATION]',
            '[STATE_CLEAN]',
            '[RECT_0_DEBUG]',
            '[EXCLUDE_RESTORE]',
            '[ITERATIONS_RESTORE]',
        )
        if first_arg.startswith(debug_prefixes) and not important and not _manga_debug_logging_enabled():
            return
    except Exception:
        pass
    print(*args, **kwargs)


def _normalize_region_kind(value) -> str:
    try:
        return str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    except Exception:
        return ''


def _is_free_text_region_metadata(region=None, rect_item=None) -> bool:
    candidates = []
    for source in (region, rect_item):
        if not source:
            continue
        try:
            if isinstance(source, dict):
                candidates.extend([source.get('bubble_type'), source.get('region_type')])
            else:
                candidates.extend([getattr(source, 'bubble_type', None), getattr(source, 'region_type', None)])
        except Exception:
            continue
    return any(_normalize_region_kind(value) in ('free_text', 'text_free', 'freetext') for value in candidates)


def _preserve_free_text_inpaint_enabled(self) -> bool:
    values = []
    try:
        if hasattr(self, 'free_text_only_bg_opacity_value'):
            values.append(bool(self.free_text_only_bg_opacity_value))
    except Exception:
        pass
    try:
        translator = getattr(self, 'translator', None)
        if translator is not None and hasattr(translator, 'free_text_only_bg_opacity'):
            values.append(bool(translator.free_text_only_bg_opacity))
    except Exception:
        pass
    try:
        if getattr(self, 'main_gui', None):
            values.append(bool(self.main_gui.config.get('manga_free_text_only_bg_opacity', False)))
    except Exception:
        pass
    try:
        if hasattr(self, 'ft_only_checkbox'):
            values.append(bool(self.ft_only_checkbox.isChecked()))
    except Exception:
        pass
    return any(values)


def _copy_region_metadata_to_item(item, *sources):
    for source in sources:
        if not source:
            continue
        for key in _REGION_METADATA_KEYS:
            try:
                value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
            except Exception:
                value = None
            if value is not None:
                try:
                    setattr(item, key, value)
                except Exception:
                    pass


def _saved_rect_metadata(state: dict, index: int, rect_data: dict) -> dict:
    metadata = {}
    if isinstance(rect_data, dict):
        for key in _REGION_METADATA_KEYS:
            if rect_data.get(key) is not None:
                metadata[key] = rect_data.get(key)
    try:
        detection_regions = state.get('detection_regions') or []
        if 0 <= index < len(detection_regions) and isinstance(detection_regions[index], dict):
            det = detection_regions[index]
            for key in _REGION_METADATA_KEYS:
                if metadata.get(key) is None and det.get(key) is not None:
                    metadata[key] = det.get(key)
    except Exception:
        pass
    return metadata


def _manga_output_text(value):
    """Keep API failure messages out of manga OCR and translation output."""
    value = str(value or '')
    text = value.strip()
    if text.upper() == '[API RESPONSE UNAVAILABLE]':
        return ''
    if UnifiedClient and UnifiedClient._is_api_error_placeholder(text):
        return ''
    if text.startswith(('[Full Page Context Error:', '[Translation Error:', '[Individual Translation Error:')):
        return ''
    return value


def _regions_with_ocr_text(regions, recognized_texts):
    """Keep detected regions whose matching OCR result contains real text."""
    valid_indexes = set()
    valid_boxes = set()
    for item in recognized_texts or []:
        if not isinstance(item, dict) or not str(_manga_output_text(item.get('text'))).strip():
            continue
        try:
            valid_indexes.add(int(item['region_index']))
        except (KeyError, TypeError, ValueError):
            pass
        valid_boxes.add(tuple(item.get('bbox') or ()))
    selected = []
    for index, region in enumerate(regions or []):
        if index not in valid_indexes and not (
            isinstance(region, dict) and tuple(region.get('bbox') or ()) in valid_boxes
        ):
            continue
        if isinstance(region, dict):
            original_index = region.get('rect_index')
            region = {**region, 'rect_index': index if original_index is None else original_index}
        selected.append(region)
    return selected


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

        # CRITICAL: Release stale pool checkouts from interrupted translations.
        # When the user hits Stop mid-translation, the inpainter/detector that
        # was in use never reaches `_return_inpainter_to_pool()`, so its
        # reference stays in `rec['checked_out']`. The next Translate /
        # Translate All click would then see `spare in checked_out == True`
        # for every spare and fall into the long wait loop
        # ("⏳ All inpainter instances in use… waiting up to 1800s").
        # The Start Translation path already calls this via
        # `_reset_global_cancellation`; mirror that behavior here so the
        # workflow buttons recover the pool after a stop+restart too.
        try:
            from manga_translator import MangaTranslator
            released_inp, released_det = MangaTranslator.force_release_all_pool_checkouts()
            if released_inp or released_det:
                print(
                    f"[CANCEL_RESET] Force-released stale pool checkouts: "
                    f"{released_inp} inpainter(s), {released_det} detector(s)"
                )
        except Exception as e:
            print(f"[CANCEL_RESET] force_release_all_pool_checkouts failed: {e}")
        
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
            temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=None, skip_inpainter_init=True)
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
        image = cv2_imread(image_path)
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
    print = _manga_debug_print
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
        
        # Clear current detection regions to prevent contaminating other images
        if hasattr(self, '_current_regions'):
            old_count = len(self._current_regions) if self._current_regions else 0
            self._current_regions = []
            if old_count > 0:
                print(f"[STATE_ISOLATION] Cleared {old_count} detection regions")

        # Clear original image tracking used by clean/preview workflows. This
        # must not survive Clear All or file-list replacement.
        if hasattr(self, '_original_image_path'):
            old_original = getattr(self, '_original_image_path', None)
            self._original_image_path = None
            if old_original:
                print(f"[STATE_ISOLATION] Cleared original image path (was: {os.path.basename(old_original)})")
        
        print(f"[STATE_ISOLATION] Cross-image state isolation completed")
        
    except Exception as e:
        print(f"[STATE_ISOLATION] Failed to clear cross-image state: {e}")


def _persist_current_image_state(self):
    """Persist the current image's state (rectangles, overlays, paths) before switching.
    IMPORTANT: Merge into existing state to avoid wiping recognized/translated texts.
    Also cleans up old overlays to prevent RAM accumulation.
    """
    print = _manga_debug_print
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
                for key in _REGION_METADATA_KEYS:
                    value = getattr(rect_item, key, None)
                    if value is not None:
                        entry[key] = value
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
    print = _manga_debug_print
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
                if not _manga_output_text(r):
                    continue
                active_rec.append({'text': r, 'bbox': [0, 0, 100, 100], 'region_index': i})
                recognition_data[int(i)] = {'text': r, 'bbox': [0, 0, 100, 100]}
            elif isinstance(r, dict) and 'text' in r:
                if not _manga_output_text(r.get('text')):
                    continue
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
                t = {**t, 'translation': _manga_output_text(t.get('translation'))}
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
    
    OCR, translated text, and editor mappings are portable source data. They
    remain valid even when cleaned/rendered image files were not exported or
    have moved, so only stale file-path fields may be removed here.
    """
    print = _manga_debug_print
    try:
        if not hasattr(self, 'image_state_manager') or not self.image_state_manager:
            return
        
        state = self.image_state_manager.get_state(image_path)
        if not state:
            return
        
        state_changed = False
        # Check cleaned_image_path
        cleaned_path = state.get('cleaned_image_path')
        if cleaned_path:
            if os.path.exists(cleaned_path):
                print(f"[STATE_CLEAN] Cleaned image exists: {os.path.basename(cleaned_path)}")
            else:
                print(f"[STATE_CLEAN] Cleaned image no longer exists: {os.path.basename(cleaned_path)}")
                state.pop('cleaned_image_path', None)
                state_changed = True
        
        # Check rendered_image_path (translated output)
        rendered_path = state.get('rendered_image_path')
        if rendered_path:
            if os.path.exists(rendered_path):
                print(f"[STATE_CLEAN] Rendered/translated image exists: {os.path.basename(rendered_path)}")
            else:
                print(f"[STATE_CLEAN] Rendered/translated image no longer exists: {os.path.basename(rendered_path)}")
                state.pop('rendered_image_path', None)
                state_changed = True
        
        # translated_texts intentionally survives missing output files. A
        # portable editor import may not include rendered images, and the text
        # is required if the user chooses to render the page again.
        
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


def _on_clean_image_clicked(self):
    """Clean button: ensure regions exist (auto-detect if needed) then run inpainting in background."""
    try:
        # Get current image path
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for cleaning", "warning")
            return

        # Disable all workflow buttons and show stop button
        _disable_workflow_buttons(self)
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'clean_btn'):
            self.image_preview_widget.clean_btn.setText("Cleaning...")

        # Determine base image path from the preview's current image. Do not
        # reuse _original_image_path here because it may belong to a previously
        # cleared or removed manga image.
        image_path = self.image_preview_widget.current_image_path
        
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
                for key in _REGION_METADATA_KEYS:
                    value = getattr(rect_item, key, None)
                    if value is not None:
                        region_dict[key] = value
                try:
                    if not region_dict.get('bubble_type') and hasattr(self, '_current_regions') and 0 <= i < len(self._current_regions):
                        _current = self._current_regions[i]
                        if isinstance(_current, dict):
                            for key in _REGION_METADATA_KEYS:
                                if region_dict.get(key) is None and _current.get(key) is not None:
                                    region_dict[key] = _current.get(key)
                except Exception:
                    pass
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
    previous_inpainter_log_callback = None
    custom_image_edit_log_attached = False
    
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
        image = cv2_imread(image_path)
        if image is None:
            self._log(f"❌ Failed to load image: {os.path.basename(image_path)}", "error")
            self.update_queue.put(('clean_button_restore', None))
            return
        
        # Get exclusion list directly from rectangle objects (session-only)
        excluded_regions = []
        rectangles = []
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
        free_text_skipped_count = 0
        preserve_free_text = _preserve_free_text_inpaint_enabled(self)
        
        print(f"[CLEAN_DEBUG] Processing {len(regions)} regions for exclusion filtering")
        for i, region in enumerate(regions):
            # Check if this region should be excluded (using rect_index if available)
            rect_index = region.get('rect_index', None)
            rect_item = None
            try:
                lookup_index = rect_index if rect_index is not None else i
                if rectangles and 0 <= int(lookup_index) < len(rectangles):
                    rect_item = rectangles[int(lookup_index)]
            except Exception:
                rect_item = None
            print(f"[CLEAN_DEBUG] Region {i}: rect_index={rect_index}, excluded_regions={excluded_regions}")
            
            if rect_index is not None and rect_index in excluded_regions:
                excluded_count += 1
                print(f"[CLEAN_DEBUG] EXCLUDING region {i} (rect_index={rect_index})")
                self._log(f"🚫 Skipping region {rect_index} (excluded from clean)", "info")
                continue
            if preserve_free_text and _is_free_text_region_metadata(region, rect_item):
                free_text_skipped_count += 1
                print(f"[CLEAN_DEBUG] SKIPPING free-text region {i} (rect_index={rect_index})")
                self._log(f"Skipping free-text region {rect_index if rect_index is not None else i} (preserve free text enabled)", "info")
                continue
            else:
                print(f"[CLEAN_DEBUG] INCLUDING region {i} (rect_index={rect_index})")
            
            filtered_regions.append(region)

        if not filtered_regions:
            self._log("⏭️ No eligible regions to clean; preserving original image", "info")
            return
        
        self._log(f"🎨 Creating mask from {len(filtered_regions)} regions ({excluded_count} excluded)", "info")
        
        if free_text_skipped_count:
            print(f"[CLEAN_DEBUG] Preserved {free_text_skipped_count} free-text regions during clean mask creation")
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

        if not np.any(mask):
            self._log("⏭️ No eligible text mask; preserving original image", "info")
            return
        
        # Get inpainting settings from manga integration config
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        is_custom_image_edit = inpaint_method == 'local' and str(local_model or '').lower() == 'custom-image-edit'
        
        if inpaint_method == 'local':
            # Use local inpainter with the same method as manga_translator
            self._log(f"🖼️ Using local inpainter: {local_model}", "info")
            
            # Get model path from config (same way as manga_translator)
            model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
            if is_custom_image_edit and not model_path:
                model_path = getattr(self.main_gui, 'custom_image_edit_endpoint_var', '') or self.main_gui.config.get('custom_image_edit_endpoint', '')
            try:
                if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                    model_path = ''
            except Exception:
                pass
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if (not is_custom_image_edit) and (not resolved_model_path or not os.path.exists(resolved_model_path)):
                try:
                    from local_inpainter import LocalInpainter
                    self._log(f"📥 Downloading {local_model} model...", "info")
                    temp_inp = LocalInpainter()
                    resolved_model_path = temp_inp.download_jit_model(local_model)
                except Exception as e:
                    self._log(f"⚠️ Model download failed: {e}", "warning")
                    resolved_model_path = None
            
            # Use shared inpainter from pool - track the temporary translator for cleanup
            if is_custom_image_edit or (resolved_model_path and os.path.exists(resolved_model_path)):
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
                    temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=None, skip_inpainter_init=True)
                    
                    # POLL for inpainter with timeout (same as translator initialization)
                    inpainter = None
                    poll_timeout = 30  # 30 seconds
                    poll_interval = 0.5  # Check every 500ms
                    start_time = time.time()
                    
                    while time.time() - start_time < poll_timeout:
                        inpainter = temp_translator._get_or_init_shared_local_inpainter(local_model, resolved_model_path or '', force_reload=False)
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

            if is_custom_image_edit:
                previous_inpainter_log_callback = getattr(inpainter, 'log_callback', None)
                inpainter.set_log_callback(self._log)
                custom_image_edit_log_attached = True
            
            if custom_iterations:
                iterations_str = ', '.join([f"{region}:{iters}" for region, iters in custom_iterations.items()])
                self._log(f"🧽 Running local inpainting with custom iterations: {iterations_str}", "info")
                # For now, use the first custom iteration value found
                # TODO: Implement per-region inpainting with different iterations
                first_iteration_value = next(iter(custom_iterations.values()))
                disable_performance_mode = not is_custom_image_edit and bool(self.main_gui.config.get('manga_disable_inpaint_performance_mode', False))
                cleaned_image = inpainter.inpaint(
                    image,
                    mask,
                    iterations=first_iteration_value,
                    _skip_hd=disable_performance_mode,
                    _skip_tiling=disable_performance_mode,
                )
            else:
                self._log("🧽 Running local inpainting with auto iterations", "info")
                disable_performance_mode = not is_custom_image_edit and bool(self.main_gui.config.get('manga_disable_inpaint_performance_mode', False))
                cleaned_image = inpainter.inpaint(
                    image,
                    mask,
                    _skip_hd=disable_performance_mode,
                    _skip_tiling=disable_performance_mode,
                )
            
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
        if custom_image_edit_log_attached and inpainter is not None:
            inpainter.set_log_callback(previous_inpainter_log_callback)
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
    """Get inpainting configuration from settings.

    Respects the Skip Inpainter toggle (self.skip_inpainting_value in PySide6
    build, mirrored to config['manga_skip_inpainting']). When skipped, the
    returned method is 'none' and a 'skip' flag is set so callers can early-out.
    """
    # Prefer the live GUI state, fall back to persisted config
    skip = False
    try:
        if hasattr(self, 'skip_inpainting_value'):
            skip = bool(self.skip_inpainting_value)
        elif hasattr(self, 'main_gui') and getattr(self.main_gui, 'config', None):
            skip = bool(self.main_gui.config.get('manga_skip_inpainting', False))
    except Exception:
        skip = False

    method = 'none' if skip else (
        self.inpaint_method_value if hasattr(self, 'inpaint_method_value') else 'none'
    )

    inpaint_config = {
        'skip': skip,
        'method': method,
        'model_type': self.local_model_type_value if hasattr(self, 'local_model_type_value') else 'lama',
        'model_path': self.local_model_path_value if hasattr(self, 'local_model_path_value') else '',
        'quality': self.inpaint_quality_value if hasattr(self, 'inpaint_quality_value') else 'high',
        'dilation': self.inpaint_dilation_value if hasattr(self, 'inpaint_dilation_value') else 0,
        'passes': self.inpaint_passes_value if hasattr(self, 'inpaint_passes_value') else 2
    }
    return inpaint_config


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
            temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=None, skip_inpainter_init=True)
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
        image = cv2_imread(image_path)
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


def _run_inpainting_sync(
    self,
    image_path: str,
    regions: list,
    save_as: str = 'cleaned',
    custom_image_edit_system_prompt: str = None,
) -> str:
    """Run inpainting synchronously (for Translate button) and return cleaned image path
    
    Args:
        image_path: Path to the original image
        regions: List of region dictionaries with 'bbox' keys
        
    Returns:
        str: Path to cleaned/translated image, or None if inpainting failed
    """
    temp_translator = None  # Track temporary translator for pool cleanup
    try:
        # ===== CANCELLATION CHECK: At start of sync inpainting =====
        # Only abort on FORCE stop — graceful stop should let inpainting finish
        # since it runs concurrently with translation and aborting wastes the work.
        if _is_translation_cancelled(self) and os.environ.get('GRACEFUL_STOP') != '1':
            print(f"[INPAINT_SYNC] Force-cancelled at start")
            return None
        
        import cv2
        import numpy as np
        from local_inpainter import LocalInpainter
        
        # Load image
        image = cv2_imread(image_path)
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
        free_text_skipped_count = 0
        preserve_free_text = _preserve_free_text_inpaint_enabled(self)
        
        print(f"[INPAINT_SYNC] Processing {len(regions)} regions for inpainting")
        for i, region in enumerate(regions):
            # Check if this region should be excluded
            original_index = region.get('rect_index') if isinstance(region, dict) else None
            region_index = i if original_index is None else original_index
            if region_index in excluded_regions:
                excluded_count += 1
                print(f"[INPAINT_SYNC] Skipping region {region_index} (excluded from clean)")
                continue
            if preserve_free_text and _is_free_text_region_metadata(region):
                free_text_skipped_count += 1
                print(f"[INPAINT_SYNC] Skipping region {i} (free text preserved)")
                continue
            if isinstance(region, dict) and 'text' in region and not str(_manga_output_text(region['text'])).strip():
                print(f"[INPAINT_SYNC] Skipping region {i} (no OCR text)")
                continue
            
            regions_to_inpaint.append((region_index, region))
        
        print(f"[INPAINT_SYNC] Creating mask from {len(regions_to_inpaint)} regions ({excluded_count} excluded, {free_text_skipped_count} free-text preserved)")
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

        if not np.any(mask):
            print("[INPAINT_SYNC] No eligible text regions; skipping inpainting")
            return None
        
        # Get inpainting settings from manga integration config
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        
        is_custom_image_edit = False
        if inpaint_method == 'local':
            # Use local inpainter with the same method as manga_translator
            print(f"[INPAINT_SYNC] Using local inpainter: {local_model}")
            
            # Get model path from config (same way as manga_translator)
            model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
            is_custom_image_edit = str(local_model or '').lower() == 'custom-image-edit'
            if is_custom_image_edit and not model_path:
                model_path = getattr(self.main_gui, 'custom_image_edit_endpoint_var', '') or self.main_gui.config.get('custom_image_edit_endpoint', '')
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if (not is_custom_image_edit) and (not resolved_model_path or not os.path.exists(resolved_model_path)):
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
            if is_custom_image_edit or (resolved_model_path and os.path.exists(resolved_model_path)):
                try:
                    from manga_translator import MangaTranslator
                    from local_inpainter import LocalInpainter
                    import time
                    
                    # Normalize model path to match pool key. Custom image edit is
                    # endpoint-backed, so a blank path means the default image endpoint.
                    if is_custom_image_edit:
                        key = (local_model, resolved_model_path or '__default_image_edit_endpoint__')
                    else:
                        resolved_model_path = os.path.abspath(os.path.normpath(resolved_model_path))
                        key = (local_model, resolved_model_path)
                    
                    # Poll for inpainter from pool with timeout
                    # This waits for preloading to complete instead of creating a new instance
                    inpainter = None
                    poll_timeout = 0 if is_custom_image_edit else 60  # Endpoint-backed edit can fake-load immediately
                    poll_interval = 0.5  # Check every 500ms
                    start_time = time.time()
                    attempt = 0
                    
                    while time.time() - start_time < poll_timeout:
                        attempt += 1
                        
                        # Check for cancellation while polling — only abort on force stop
                        if _is_translation_cancelled(self) and os.environ.get('GRACEFUL_STOP') != '1':
                            print(f"[INPAINT_SYNC] Force-cancelled while waiting for inpainter")
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
                        try:
                            if is_custom_image_edit and hasattr(self, 'main_gui') and getattr(self.main_gui, 'config', None):
                                new_inpainter.config.update(self.main_gui.config)
                                default_endpoint = ''
                                use_custom_openai = bool(
                                    getattr(self.main_gui, 'use_custom_openai_endpoint_var', False)
                                    or self.main_gui.config.get('use_custom_openai_endpoint', False)
                                )
                                if use_custom_openai:
                                    default_endpoint = self.main_gui.config.get('openai_base_url', '')
                                    default_endpoint = default_endpoint or getattr(self.main_gui, 'openai_base_url_var', '') or ''
                                new_inpainter.config['custom_image_edit_default_endpoint'] = default_endpoint
                                new_inpainter.config['openai_base_url'] = default_endpoint
                                new_inpainter.config['use_custom_openai_endpoint'] = use_custom_openai
                        except Exception:
                            pass
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
            
            # ===== CANCELLATION CHECK: Before running inpainting — only abort on force stop =====
            if _is_translation_cancelled(self) and os.environ.get('GRACEFUL_STOP') != '1':
                print(f"[INPAINT_SYNC] Force-cancelled before inpainting")
                return None
            
            if inpainter is not None:
                try:
                    cfg = getattr(self.main_gui, 'config', {}) if hasattr(self, 'main_gui') else {}
                    disable_performance_mode = not is_custom_image_edit and bool(
                        getattr(self.main_gui, 'manga_disable_inpaint_performance_mode_var', False)
                        or (cfg.get('manga_disable_inpaint_performance_mode', False) if isinstance(cfg, dict) else False)
                    )
                    if hasattr(inpainter, 'config') and isinstance(inpainter.config, dict):
                        inpainter.config['manga_disable_inpaint_performance_mode'] = disable_performance_mode
                except Exception:
                    pass

            if is_custom_image_edit and inpainter is not None:
                try:
                    cfg = getattr(self.main_gui, 'config', {}) if hasattr(self, 'main_gui') else {}
                    if isinstance(cfg, dict):
                        inpainter.config.update(cfg)
                        live_batch_size = getattr(self.main_gui, 'batch_size_var', cfg.get('batch_size', 1))
                        inpainter.config['batch_size'] = live_batch_size.get() if hasattr(live_batch_size, 'get') else live_batch_size
                    override_enabled = bool(
                        getattr(self.main_gui, 'use_custom_image_edit_endpoint_var', False)
                        or (cfg.get('use_custom_image_edit_endpoint', False) if isinstance(cfg, dict) else False)
                    )
                    endpoint = ''
                    if override_enabled:
                        endpoint = str(
                            getattr(self.main_gui, 'custom_image_edit_endpoint_var', '')
                            or (cfg.get('custom_image_edit_endpoint', '') if isinstance(cfg, dict) else '')
                            or ''
                        ).strip()
                    inpainter.config['use_custom_image_edit_endpoint'] = bool(override_enabled)
                    inpainter.config['custom_image_edit_endpoint'] = endpoint
                    inpainter._custom_image_edit_use_current_provider = not bool(endpoint)
                    inpainter._custom_image_edit_endpoint = endpoint if endpoint else 'current-provider'
                    model_name = (
                        getattr(self.main_gui, 'model_var', '')
                        or (cfg.get('model', '') if isinstance(cfg, dict) else '')
                    )
                    if model_name:
                        inpainter.config['model'] = model_name
                        inpainter.config['custom_image_edit_model'] = model_name
                        inpainter._custom_image_edit_model_ref = model_name
                    disable_performance_mode = not is_custom_image_edit and bool(
                        getattr(self.main_gui, 'manga_disable_inpaint_performance_mode_var', False)
                        or (cfg.get('manga_disable_inpaint_performance_mode', False) if isinstance(cfg, dict) else False)
                    )
                    inpainter.config['manga_disable_inpaint_performance_mode'] = disable_performance_mode
                    image_edit_system_prompt = (
                        getattr(self.main_gui, 'custom_image_edit_system_prompt_var', '')
                        or getattr(self.main_gui, 'custom_image_edit_prompt_var', '')
                        or (cfg.get('custom_image_edit_system_prompt', '') if isinstance(cfg, dict) else '')
                        or (cfg.get('custom_image_edit_prompt', '') if isinstance(cfg, dict) else '')
                    )
                    image_edit_user_prompt = (
                        getattr(self.main_gui, 'custom_image_edit_user_prompt_var', '')
                        or (cfg.get('custom_image_edit_user_prompt', '') if isinstance(cfg, dict) else '')
                    )
                    if image_edit_system_prompt:
                        inpainter.config['custom_image_edit_system_prompt'] = image_edit_system_prompt
                        inpainter.config['custom_image_edit_prompt'] = image_edit_system_prompt
                    inpainter.config['custom_image_edit_user_prompt'] = image_edit_user_prompt or ''
                    _fp_raw = getattr(self.main_gui, 'custom_image_edit_full_page_output_var', 10)
                    if isinstance(_fp_raw, bool):
                        _fp_raw = 100 if _fp_raw else 0
                    elif not isinstance(_fp_raw, int):
                        try:
                            _fp_cfg = cfg.get('custom_image_edit_full_page_output', 10) if isinstance(cfg, dict) else 10
                            _fp_raw = int(_fp_cfg) if not isinstance(_fp_cfg, bool) else (100 if _fp_cfg else 0)
                        except (ValueError, TypeError):
                            _fp_raw = 10
                    full_page_output = max(0, min(100, int(_fp_raw)))
                    inpainter.config['custom_image_edit_full_page_output'] = full_page_output
                    if not custom_image_edit_system_prompt and hasattr(inpainter, '_custom_image_edit_request_system_prompt'):
                        delattr(inpainter, '_custom_image_edit_request_system_prompt')
                    if not endpoint:
                        print("[INPAINT_SYNC] Custom image edit URL is blank; using current provider/model")
                except Exception as e:
                    print(f"[INPAINT_SYNC] Failed to refresh custom image edit endpoint mode: {e}")

            old_custom_image_edit_prompts = None
            old_custom_image_edit_request_prompt = None
            had_custom_image_edit_request_prompt = False
            old_mp_enabled = None
            if is_custom_image_edit and custom_image_edit_system_prompt:
                old_custom_image_edit_prompts = (
                    inpainter.config.get('custom_image_edit_system_prompt'),
                    inpainter.config.get('custom_image_edit_prompt'),
                    inpainter.config.get('custom_image_edit_user_prompt'),
                )
                had_custom_image_edit_request_prompt = hasattr(inpainter, '_custom_image_edit_request_system_prompt')
                old_custom_image_edit_request_prompt = getattr(inpainter, '_custom_image_edit_request_system_prompt', None)
                inpainter.config['custom_image_edit_system_prompt'] = custom_image_edit_system_prompt
                inpainter.config['custom_image_edit_prompt'] = custom_image_edit_system_prompt
                inpainter.config['custom_image_edit_user_prompt'] = ''
                inpainter._custom_image_edit_request_system_prompt = custom_image_edit_system_prompt

                # The image-edit worker owns its own config snapshot. For per-click
                # Translate prompts, run endpoint-backed editing in-process so the
                # prompt from translator_gui.py is the one actually sent.
                old_mp_enabled = getattr(inpainter, '_mp_enabled', None)
                if old_mp_enabled is not None:
                    inpainter._mp_enabled = False
                try:
                    prompt_preview = " ".join(str(custom_image_edit_system_prompt).split())[:160]
                    print(f"[INPAINT_SYNC] Custom image edit translate prompt active ({len(str(custom_image_edit_system_prompt))} chars): {prompt_preview}")
                except Exception:
                    pass

            # Run inpainting/image editing
            print(f"[INPAINT_SYNC] Running local inpainting...")
            previous_inpainter_log_callback = None
            try:
                if is_custom_image_edit:
                    previous_inpainter_log_callback = getattr(inpainter, 'log_callback', None)
                    inpainter.set_log_callback(self._log)
                disable_performance_mode = not is_custom_image_edit and bool(
                    getattr(inpainter, 'config', {}).get('manga_disable_inpaint_performance_mode', False)
                    if hasattr(inpainter, 'config') else False
                )
                cleaned_image = inpainter.inpaint(
                    image,
                    mask,
                    _skip_hd=disable_performance_mode,
                    _skip_tiling=disable_performance_mode,
                )
            finally:
                if is_custom_image_edit:
                    inpainter.set_log_callback(previous_inpainter_log_callback)
                if old_mp_enabled is not None:
                    try:
                        inpainter._mp_enabled = old_mp_enabled
                    except Exception:
                        pass
                if old_custom_image_edit_prompts is not None:
                    if had_custom_image_edit_request_prompt:
                        inpainter._custom_image_edit_request_system_prompt = old_custom_image_edit_request_prompt
                    else:
                        try:
                            delattr(inpainter, '_custom_image_edit_request_system_prompt')
                        except Exception:
                            pass
                if old_custom_image_edit_prompts is not None:
                    old_system, old_prompt, old_user = old_custom_image_edit_prompts
                    if old_system is None:
                        inpainter.config.pop('custom_image_edit_system_prompt', None)
                    else:
                        inpainter.config['custom_image_edit_system_prompt'] = old_system
                    if old_prompt is None:
                        inpainter.config.pop('custom_image_edit_prompt', None)
                    else:
                        inpainter.config['custom_image_edit_prompt'] = old_prompt
                    if old_user is None:
                        inpainter.config.pop('custom_image_edit_user_prompt', None)
                    else:
                        inpainter.config['custom_image_edit_user_prompt'] = old_user
            
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
        
        if (
            is_custom_image_edit
            and cleaned_image is not None
            and os.environ.get('GRACEFUL_STOP') != '1'
            and _is_translation_cancelled(self)
        ):
            print("[INPAINT_SYNC] Force-cancelled before saving custom image edit result")
            return None

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
            if save_as == 'translated':
                cleaned_path = os.path.join(output_dir, filename)
            else:
                cleaned_path = os.path.join(output_dir, f"{base}_cleaned{ext}")

            if is_custom_image_edit and os.environ.get('GRACEFUL_STOP') != '1' and _is_translation_cancelled(self):
                print("[INPAINT_SYNC] Force-cancelled before writing custom image edit result")
                return None
            cv2.imwrite(cleaned_path, cleaned_image)
            if save_as == 'translated':
                print(f"[INPAINT_SYNC] Saved translated image to: {cleaned_path}")
            else:
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
        import concurrent.futures
        
        # Load image
        image = cv2_imread(image_path)
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
                        import large_env
                        for key, value in env_vars.items():
                            if key == 'SYSTEM_PROMPT':
                                # DON'T SET THE TRANSLATION SYSTEM PROMPT FOR OCR
                                continue
                            large_env.set_env(key, str(value))
                        try:
                            import json as _json
                            from unified_api_client import UnifiedClient as _UnifiedClient
                            vision_keys = self.main_gui.config.get('qa_scan_keys', []) or []
                            use_vision_keys = getattr(
                                self.main_gui,
                                'use_qa_scan_keys_var',
                                self.main_gui.config.get('use_qa_scan_keys', False)
                            )
                            if hasattr(use_vision_keys, 'isChecked'):
                                use_vision_keys = use_vision_keys.isChecked()
                            elif hasattr(use_vision_keys, 'get'):
                                use_vision_keys = use_vision_keys.get()
                            use_vision_keys = bool(use_vision_keys)
                            self.main_gui.config['use_qa_scan_keys'] = use_vision_keys
                            os.environ['USE_VISION_KEYS'] = '1' if use_vision_keys else '0'
                            os.environ['USE_QA_SCAN_KEYS'] = os.environ['USE_VISION_KEYS']
                            os.environ['VISION_API_KEYS'] = _json.dumps(vision_keys)
                            os.environ['QA_SCAN_API_KEYS'] = os.environ['VISION_API_KEYS']
                            if use_vision_keys and vision_keys:
                                _UnifiedClient.set_in_memory_vision_keys(
                                    vision_keys,
                                    force_rotation=self.main_gui.config.get('force_key_rotation', True),
                                    rotation_frequency=self.main_gui.config.get('rotation_frequency', 1),
                                )
                            else:
                                _UnifiedClient.clear_in_memory_vision_keys()
                        except Exception as vision_err:
                            print(f"[OCR_REGIONS] Failed to sync Vision keys for custom-api OCR: {vision_err}")
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
                        os.environ['OCR_SYSTEM_PROMPT'] = self._default_manga_ocr_prompt()
                        self._log("✅ Using default OCR prompt", "info")
                except Exception:
                    pass

                # 4) Respect user settings: set non-intrusive defaults only when bubble detection is OFF
                try:
                    ms = self.main_gui.config.setdefault('manga_settings', {})
                    ocr_set = ms.setdefault('ocr', {})
                    changed = False
                    bubble_enabled = bool(ocr_set.get('bubble_detection_enabled', True))
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

                # 5) Load custom-api provider - ALWAYS reload to pick up current multi-key state
                # Force fresh UnifiedClient creation so toggling multi-key mode takes effect
                provider_obj = self.ocr_manager.get_provider(provider)
                if provider_obj:
                    provider_obj.is_loaded = False  # force reload with current env vars
                if not self.ocr_manager.get_provider(provider).is_loaded:
                    print(f"[OCR_REGIONS] Loading OCR provider: {provider}")
                    load_kwargs = {}
                    if api_key:
                        load_kwargs['api_key'] = api_key
                        print(f"[OCR_REGIONS] Got API key from GUI")
                    load_kwargs['context'] = 'manga_ocr'
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
                
                def _custom_api_ocr_batch_size():
                    try:
                        if not bool(ocr_config.get('custom_api_ocr_batch_enabled', True)):
                            return 1
                        return max(1, int(ocr_config.get('custom_api_ocr_batch_size', 5)))
                    except Exception:
                        return 1

                if provider == 'custom-api' and len(regions) > 1:
                    ocr_workers = min(_custom_api_ocr_batch_size(), len(regions))
                    if ocr_workers > 1:
                        self._log(f"Using PARALLEL OCR for {len(regions)} regions (custom-api; workers={ocr_workers})", "info")
                        keys = (
                            'BATCH_TRANSLATION',
                            'BATCH_SIZE',
                            'MANGA_OCR_THINKING_OVERRIDE_ACTIVE',
                            'ENABLE_ANTHROPIC_THINKING',
                            'ENABLE_GEMINI_THINKING',
                            'ENABLE_DEEPSEEK_THINKING',
                            'GEMINI_THINKING_LEVEL',
                            'THINKING_BUDGET',
                        )
                        original_env = {key: os.environ.get(key) for key in keys}
                        os.environ['BATCH_TRANSLATION'] = '1'
                        os.environ['BATCH_SIZE'] = str(ocr_workers)
                        if str(os.environ.get('MANGA_OCR_DISABLE_THINKING', '1')).strip().lower() in ('1', 'true', 'yes', 'on'):
                            os.environ['MANGA_OCR_THINKING_OVERRIDE_ACTIVE'] = '1'
                            os.environ['ENABLE_ANTHROPIC_THINKING'] = '0'
                            os.environ['ENABLE_GEMINI_THINKING'] = '0'
                            os.environ['ENABLE_DEEPSEEK_THINKING'] = '0'
                            os.environ['GEMINI_THINKING_LEVEL'] = 'minimal'
                            os.environ.pop('THINKING_BUDGET', None)

                        def _process_custom_api_region(item):
                            i, region = item
                            if _is_translation_cancelled(self):
                                return None
                            bbox = region.get('bbox', [])
                            if len(bbox) < 4:
                                return None
                            region_x, region_y, region_w, region_h = bbox
                            cropped_region = image[region_y:region_y+region_h, region_x:region_x+region_w]
                            ocr_results = self.ocr_manager.detect_text(cropped_region, provider, confidence=0.5)
                            if not ocr_results:
                                return None
                            region_text = " ".join([ocr.text.strip() for ocr in ocr_results if ocr.text.strip()])
                            if not region_text:
                                return None
                            return {
                                'region_index': i,
                                'bbox': bbox,
                                'text': region_text.strip(),
                                'confidence': region.get('confidence', 1.0),
                                'bubble_type': region.get('bubble_type'),
                                'region_type': region.get('region_type'),
                                'bubble_bounds': region.get('bubble_bounds', bbox)
                            }

                        try:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=ocr_workers) as executor:
                                future_map = {
                                    executor.submit(_process_custom_api_region, item): item[0]
                                    for item in enumerate(regions)
                                }
                                parallel_results = []
                                for future in concurrent.futures.as_completed(future_map):
                                    result = future.result()
                                    if result:
                                        parallel_results.append(result)
                                        print(f"[OCR_REGIONS] Region {result['region_index']+1}: '{result['text']}'")
                                recognized_texts.extend(sorted(parallel_results, key=lambda r: r.get('region_index', 0)))
                        finally:
                            for key, value in original_env.items():
                                if value is None:
                                    os.environ.pop(key, None)
                                else:
                                    os.environ[key] = value

                        print(f"[OCR_REGIONS] custom-api recognized text in {len(recognized_texts)}/{len(regions)} regions")
                        return recognized_texts

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
                vision = _import_google_vision()
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
        
        # Disable all workflow buttons and show stop button
        _disable_workflow_buttons(self)
        print(f"[DEBUG] Has recognize_btn: {hasattr(self.image_preview_widget, 'recognize_btn')}")
        if hasattr(self.image_preview_widget, 'recognize_btn'):
            print(f"[DEBUG] Disabling recognize button")
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


def _refresh_live_gui_state_for_manga_action(self):
    """Sync manga workflow actions with the current main GUI state.

    The Start Translation button in `manga_integration.py` explicitly triggers
    a refresh from the main GUI before kicking off work, which makes it pick up
    the latest API key / model selection. The image-preview workflow buttons
    (Translate / Translate All) skipped that step, and the full-page-context
    path also reuses a cached `_manga_translator` whose `UnifiedClient` was
    built from whatever key/model were present on the first run.

    Result: changing the API key or model in the main GUI had no effect for the
    workflow buttons until the whole GUI was restarted.

    This helper mirrors the Start Translation behaviour and additionally drops
    the cached translator/client instances so the next request rebuilds against
    the live GUI values.
    """
    # 1) Re-read / refresh from the main GUI if available
    try:
        if hasattr(self, '_refresh_context_settings'):
            self._refresh_context_settings()
        elif hasattr(self, 'refresh_btn') and self.refresh_btn:
            self.refresh_btn.click()
    except Exception as e:
        try:
            self._log(f"⚠️ Warning: Could not refresh from main GUI: {e}", "debug")
        except Exception:
            pass

    # 2) Invalidate the cached MangaTranslator so full-page-context rebuilds a
    #    fresh UnifiedClient using the current API key / model.
    try:
        if hasattr(self, '_manga_translator') and self._manga_translator is not None:
            try:
                if hasattr(self._manga_translator, 'restore_print'):
                    self._manga_translator.restore_print()
            except Exception:
                pass
            self._manga_translator = None
            try:
                self._log("🔄 Cleared cached manga translator to use current API key/model", "debug")
            except Exception:
                pass
    except Exception:
        pass

    # 3) Drop the main GUI's cached client as well so any code path that uses
    #    `main_gui.client` will recreate it from the live GUI values.
    try:
        if hasattr(self, 'main_gui') and self.main_gui is not None:
            if getattr(self.main_gui, 'client', None) is not None:
                self.main_gui.client = None
                try:
                    self._log("🔄 Cleared cached main GUI client to use current API key/model", "debug")
                except Exception:
                    pass
    except Exception:
        pass


def _get_loaded_manga_glossary_for_workflow(self) -> str:
    """Return the manga glossary text selected by the manga workflow toggle."""
    try:
        if hasattr(self, '_manga_glossary_workflow_enabled'):
            enabled = bool(self._manga_glossary_workflow_enabled())
        elif hasattr(self, 'manga_glossary_checkbox'):
            enabled = bool(self.manga_glossary_checkbox.isChecked())
        else:
            enabled = bool(getattr(self.main_gui, 'config', {}).get('manga_glossary_enabled', False))
    except Exception:
        enabled = False
    if not enabled:
        return ""

    glossary_text = ""
    try:
        if hasattr(self, '_get_loaded_manga_glossary_text'):
            glossary_text = self._get_loaded_manga_glossary_text()
    except Exception as err:
        try:
            self._log(f"Warning: Could not load manga glossary for workflow button: {err}", "warning")
        except Exception:
            pass
        glossary_text = ""

    if not glossary_text:
        for source in (self, getattr(self, 'main_gui', None)):
            try:
                for attr in ('manga_loaded_glossary_text', 'manga_generated_glossary_text'):
                    value = getattr(source, attr, '')
                    if isinstance(value, str) and value.strip():
                        glossary_text = value.strip()
                        break
                if glossary_text:
                    break
            except Exception:
                pass

    glossary_text = str(glossary_text or '').strip()
    if not glossary_text:
        return ""

    try:
        self.manga_loaded_glossary_text = glossary_text
        self.manga_generated_glossary_text = glossary_text
        setattr(self.main_gui, 'manga_generated_glossary_text', glossary_text)
        if hasattr(self, '_manga_translator') and self._manga_translator:
            self._manga_translator.manga_generated_glossary_text = glossary_text
            self._manga_translator._manga_glossary_prompt_logged = False
        if hasattr(self, 'translator') and self.translator:
            self.translator.manga_generated_glossary_text = glossary_text
            self.translator._manga_glossary_prompt_logged = False
    except Exception:
        pass

    return glossary_text


def _manga_workflow_glossary_compression_enabled(self) -> bool:
    env_value = os.getenv("COMPRESS_GLOSSARY_PROMPT")
    if env_value is not None:
        return str(env_value).strip().lower() in ("1", "true", "yes", "on")
    try:
        return bool(getattr(self.main_gui, 'config', {}).get('compress_glossary_prompt', True))
    except Exception:
        return True


def _compress_loaded_manga_glossary_for_workflow(self, glossary_text: str, source_text: str = "", image_path: str = None) -> str:
    if not glossary_text or not source_text or not _manga_workflow_glossary_compression_enabled(self):
        return glossary_text
    try:
        from glossary_compressor import compress_glossary
        glossary_path = ""
        try:
            glossary_path = (
                getattr(self, 'manga_generated_glossary_path', '')
                or getattr(self, 'manga_custom_glossary_path', '')
                or getattr(self.main_gui, 'manga_generated_glossary_path', '')
                or getattr(self.main_gui, 'manga_custom_glossary_path', '')
                or self.main_gui.config.get('manga_generated_glossary_path', '')
                or self.main_gui.config.get('manga_custom_glossary_path', '')
            )
        except Exception:
            glossary_path = ""
        original_length = len(glossary_text)
        compressed = compress_glossary(
            glossary_text,
            source_text,
            glossary_format='auto',
            glossary_path=glossary_path,
            chapter_ref={
                "chapter_num": None,
                "chapter_file": os.path.basename(image_path) if image_path else os.getenv("CURRENT_CHAPTER_FILE", ""),
            },
        )
        compressed = compressed if isinstance(compressed, str) else str(compressed or "")
        reduction_pct = ((original_length - len(compressed)) / original_length * 100) if original_length else 0
        try:
            self._log(
                f"🗜️ Manga glossary: {original_length:,}→{len(compressed):,} chars ({reduction_pct:.1f}%)",
                "info",
            )
        except Exception:
            pass
        return compressed
    except Exception as err:
        try:
            self._log(f"Warning: Manga glossary compression failed for preview: {err}", "warning")
        except Exception:
            pass
        return glossary_text


def _append_loaded_manga_glossary_to_system_prompt(self, system_prompt: str, source_text: str = "", image_path: str = None) -> str:
    """Append the currently loaded manga glossary to direct preview translations."""
    glossary_text = _get_loaded_manga_glossary_for_workflow(self)
    if not glossary_text:
        return system_prompt or ""
    glossary_text = _compress_loaded_manga_glossary_for_workflow(self, glossary_text, source_text, image_path)
    if not glossary_text or not glossary_text.strip():
        try:
            self._log("Manga glossary skipped for preview translation (no matching entries after compression)", "info")
        except Exception:
            pass
        return system_prompt or ""

    default_append_prompt = "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
    try:
        append_prompt = (
            getattr(self.main_gui, 'append_glossary_prompt', None)
            or self.main_gui.config.get('append_glossary_prompt', default_append_prompt)
        )
    except Exception:
        append_prompt = default_append_prompt

    if not getattr(self, '_workflow_manga_glossary_prompt_logged', False):
        try:
            entry_count = sum(1 for line in glossary_text.splitlines() if line.lstrip().startswith("* "))
            self._log(f"Embedding loaded manga glossary in preview translation prompt ({entry_count} entries)", "info")
        except Exception:
            pass
        self._workflow_manga_glossary_prompt_logged = True

    glossary_block = f"{str(append_prompt).rstrip()}\n{glossary_text}"
    try:
        self._log(f"✅ Manga glossary appended ({len(glossary_text):,} characters)", "info")
    except Exception:
        pass
    if system_prompt:
        return f"{system_prompt}\n\n{glossary_block}"
    return glossary_block


def _on_translate_text_clicked(self):
    """Translate recognized text using the selected API - runs full pipeline if needed"""
    self._log("🐛 Translate button clicked - starting translation", "info")

    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)

    # Refresh from the main GUI so this button picks up the CURRENT api_key
    # and model without needing a GUI restart (same behaviour as the
    # Start Translation button in manga_integration._start_translation).
    _refresh_live_gui_state_for_manga_action(self)
    
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


def _manual_translate_full_page_context_enabled(self) -> bool:
    """Read the live manual-editor setting without reusing batch snapshots."""
    if hasattr(self, 'full_page_context_value'):
        return bool(self.full_page_context_value)
    try:
        return bool(self.main_gui.config.get('manga_full_page_context', False))
    except Exception:
        return False


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
                    if not str(_manga_output_text(text_data.get('text'))).strip():
                        continue
                    bbox = text_data['bbox']
                    region_dict = {
                        'bbox': bbox,
                        'text': text_data['text'],
                        'rect_index': text_data.get('region_index'),
                        'coords': [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]],
                        'confidence': text_data.get('confidence', 1.0),
                        'bubble_type': text_data.get('bubble_type'),
                        'region_type': text_data.get('region_type'),
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

        # Respect Skip Inpainter toggle: don't spawn the inpainting thread at all.
        _inpaint_cfg = _get_inpaint_config(self)
        if _inpaint_cfg.get('skip') or _inpaint_cfg.get('method') == 'none':
            self._log("🚫 Skip Inpainter enabled — skipping inpainting for this image", "info")
            print("[TRANSLATE_CONCURRENT] Inpainting skipped (Skip Inpainter toggle ON)")
            with inpaint_lock:
                inpaint_result['completed'] = True
                inpaint_result['cleaned_path'] = None
            inpaint_thread = None
        else:
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
        # This function is the single-page workflow. Never read the Translate
        # All snapshot here: it can outlive a previous batch and override the
        # checkbox currently shown in the manual editor.
        full_page_context_enabled = _manual_translate_full_page_context_enabled(self)


        if full_page_context_enabled:
            self._log(f"📄 Using full page context translation for {len(recognized_texts)} regions", "info")
            translated_texts = _translate_with_full_page_context(self, recognized_texts, image_path)
        else:
            self._log(f"📝 Using individual translation for {len(recognized_texts)} regions", "info")
            translated_texts = _translate_individually(self, recognized_texts, image_path)
        
        print(f"[TRANSLATE_CONCURRENT] Translation completed, checking inpainting status...")
        
        # Wait for inpainting to complete (if not already done) to determine render path
        # This doesn't block the translation API calls - those already happened above
        # Even if stop was requested, we wait for inpainting since discarding a
        # completed clean is equally wasteful.
        if inpaint_thread is not None:
            inpaint_wait_timeout = 30
            try:
                cfg = getattr(getattr(self, 'main_gui', None), 'config', {}) or {}
                if (
                    str(cfg.get('manga_inpaint_method', 'local') or '').lower() == 'local'
                    and str(cfg.get('manga_local_inpaint_model', '') or '').lower() == 'custom-image-edit'
                ):
                    inpaint_wait_timeout = None
            except Exception:
                pass
            if inpaint_wait_timeout is None:
                print("[TRANSLATE_CONCURRENT] Waiting for custom-image-edit inpainting before rendering")
                inpaint_thread.join()
            else:
                inpaint_thread.join(timeout=inpaint_wait_timeout)

        with inpaint_lock:
            cleaned_image_path = inpaint_result.get('cleaned_path')
            if not inpaint_result['completed']:
                print(f"[TRANSLATE_CONCURRENT] Inpainting still running after translation, will use original image")
                self._log(f"⏱️ Inpainting still running, rendering on original image", "info")
        
        # NOTE: We intentionally do NOT check cancellation here if we already
        # have results.  The API call consumed quota; discarding the response
        # is wasteful.  Only bail if the translate function itself returned
        # nothing (meaning it was cancelled before any tokens came back).
        if not translated_texts:
            self._log(f"⏹ Translation returned no results (likely cancelled)", "warning")
            print(f"[TRANSLATE_CONCURRENT] No translation results - skipping render")
            # Even with no translations, if inpainting completed, show the cleaned image
            if cleaned_image_path:
                self.update_queue.put(('preview_update', {
                    'translated_path': cleaned_image_path,
                    'source_path': image_path
                }))
                print(f"[TRANSLATE_CONCURRENT] Inpainting result preserved despite no translation")
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
            
            # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
            if not api_key:
                if not UnifiedClient._model_needs_api_key(model):
                    api_key = 'own-auth'
            
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
                import large_env
                for key, value in env_vars.items():
                    if key == 'SYSTEM_PROMPT':
                        continue  # Don't set translation prompt for OCR
                    large_env.set_env(key, str(value))
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
        
        _get_loaded_manga_glossary_for_workflow(self)
        translations_dict = self._manga_translator.translate_full_page_context(regions, image_path)
        print(f"[DEBUG] Got translations dict: {list(translations_dict.keys()) if translations_dict else 'None'}")
        
        # NOTE: Do NOT discard results here — the API call already completed
        # and consumed quota.  Always process and return whatever came back.
        
        # Convert the results back to the expected format
        translated_texts = []
        for i, (region, text_data) in enumerate(zip(regions, recognized_texts)):
            if hasattr(region, 'translated_text') and region.translated_text:
                translation = _manga_output_text(region.translated_text)
                print(f"[DEBUG] Region {i+1} translated: '{region.text[:20]}...' -> '{translation[:20]}...'")
            else:
                translation = ''
                print(f"[DEBUG] Region {i+1} has no translation; leaving output blank")
            
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
        # Keep failed regions blank in the manga output.
        return [{
            'original': text_data,
            'translation': '',
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
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not api_key:
            if not UnifiedClient._model_needs_api_key(model):
                api_key = 'own-auth'
        
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
        glossary_source_text = "\n".join(
            str(text_data.get('text', '') or '') for text_data in recognized_texts
            if str(text_data.get('text', '') or '').strip()
        )
        system_prompt = _append_loaded_manga_glossary_to_system_prompt(
            self,
            system_prompt,
            source_text=glossary_source_text,
            image_path=image_path,
        )
        
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
                self._log(f"⏹ Translation stopped at text {i+1}/{len(recognized_texts)} — returning {len(translated_texts)} already-translated results", "warning")
                print(f"[TRANSLATE_INDIVIDUAL] Stopped at text {i+1}, returning {len(translated_texts)} partial results")
                # Return whatever we already translated — don't waste completed API calls
                return translated_texts
            
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
                    finish_reason = getattr(response, 'finish_reason', None)
                elif isinstance(response, tuple) and len(response) >= 1:
                    translated_text = response[0]  # (content, finish_reason)
                    finish_reason = response[1] if len(response) > 1 else None
                else:
                    translated_text = response
                    finish_reason = None

                if UnifiedClient._is_failed_finish_reason(finish_reason):
                    translated_text = ''
                else:
                    translated_text = _manga_output_text(translated_text)
                
                print(f"[DEBUG] Processed response: '{translated_text[:50]}...'")
                
                # NOTE: Do NOT discard the response — the API call already
                # completed and consumed quota.  We append the result below
                # and the next iteration's cancellation check will return
                # the partial list if stop is still requested.
                
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
                    'translation': '',
                    'bbox': text_data['bbox']
                })
        
        return translated_texts
        
    except Exception as e:
        # traceback already imported at top of function
        self._log(f"❌ Individual translation failed: {str(e)}", "error")
        print(f"[DEBUG] Individual translation error traceback: {traceback.format_exc()}")
        # Keep failed regions blank in the manga output.
        return [{
            'original': text_data,
            'translation': '',
            'bbox': text_data['bbox']
        } for text_data in recognized_texts]


def _get_system_prompt_from_gui(self) -> str:
    """Get the currently visible system prompt from translator_gui.py."""
    try:
        # First use the live prompt editor. This is the field the user is
        # actually looking at/editing, and it may differ from the saved profile
        # dict until autosave runs.
        try:
            prompt_widget = getattr(self.main_gui, 'prompt_text', None)
            if prompt_widget is not None and hasattr(prompt_widget, 'toPlainText'):
                live_prompt = prompt_widget.toPlainText().strip()
                if live_prompt:
                    print("[DEBUG] Using live system prompt from translator GUI")
                    return live_prompt
        except Exception:
            pass

        config_prompt = str(getattr(self.main_gui, 'config', {}).get('system_prompt', '') or '').strip()
        if config_prompt:
            print("[DEBUG] Using system prompt from translator GUI config")
            return config_prompt

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
        
        # Last resort: saved active profile content.
        system_prompt = ''
        if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
            profile_data = self.main_gui.prompt_profiles[profile_name]
            if isinstance(profile_data, dict):
                system_prompt = profile_data.get('prompt', '')
            else:
                system_prompt = profile_data
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
            if not _manga_output_text(text_data.get('text')):
                continue
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
                _style_recognized_rectangle(rect_item)
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

        recognized_texts = [
            result for result in (recognized_texts or [])
            if _manga_output_text(result.get('text'))
        ]
        
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
            _style_recognized_rectangle(rect_item)
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


def _handle_toggle_free_text_region(self, region_index: int, rect_item):
    """Toggle whether a preview rectangle should be treated as free text."""
    try:
        make_free_text = not _is_free_text_region_metadata(rect_item=rect_item)
        bubble_type = 'free_text' if make_free_text else 'text_bubble'
        region_type = 'free_text' if make_free_text else 'text_bubble'

        rect_item.bubble_type = bubble_type
        rect_item.region_type = region_type

        try:
            br = rect_item.sceneBoundingRect()
            rect_item.bubble_bounds = [int(br.x()), int(br.y()), int(br.width()), int(br.height())]
        except Exception:
            pass

        try:
            if hasattr(self, '_current_regions') and 0 <= region_index < len(self._current_regions):
                region = self._current_regions[region_index]
                if isinstance(region, dict):
                    region['bubble_type'] = bubble_type
                    region['region_type'] = region_type
                    if getattr(rect_item, 'bubble_bounds', None) is not None:
                        region['bubble_bounds'] = rect_item.bubble_bounds
        except Exception:
            pass

        try:
            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
            if current_image and hasattr(self, 'image_state_manager'):
                state = self.image_state_manager.get_state(current_image) or {}
                for key in ('detection_regions', 'viewer_rectangles'):
                    entries = state.get(key) or []
                    if 0 <= region_index < len(entries) and isinstance(entries[region_index], dict):
                        entries[region_index]['bubble_type'] = bubble_type
                        entries[region_index]['region_type'] = region_type
                        if getattr(rect_item, 'bubble_bounds', None) is not None:
                            entries[region_index]['bubble_bounds'] = rect_item.bubble_bounds
                self.image_state_manager.set_state(current_image, state, save=True)
        except Exception as state_err:
            print(f"[FREE_TEXT_TOGGLE] Failed to persist region type: {state_err}")

        _apply_rectangle_clean_style(
            rect_item,
            is_recognized=getattr(rect_item, 'is_recognized', False),
            excluded=getattr(rect_item, 'exclude_from_clean', False)
        )
        label = "free text" if make_free_text else "bubble text"
        self._log(f"Rectangle {region_index} marked as {label}", "info")
        print(f"[FREE_TEXT_TOGGLE] Rectangle {region_index} marked as {label}")
    except Exception as e:
        print(f"[FREE_TEXT_TOGGLE] Error toggling free-text status: {e}")
        self._log(f"Failed to update rectangle type: {str(e)}", "error")


def _handle_clean_this_rectangle(self, region_index: int, rect_item):
    """Handle cleaning/inpainting a specific rectangle region in background thread"""
    try:
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
            if not _confirm_clean_excluded_rectangle(self, region_index):
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
                original_image = cv2_imread(base_image_path)
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
        # Normalize model_path to match pool keys used by MangaTranslator.
        # Custom image edit is endpoint-backed, so blank/default is a valid path.
        if model_path and str(method or '').lower() != 'custom-image-edit':
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
    """Preload inpainter into the pool for fast access on first use.

    Short-circuits when the Skip Inpainter toggle is on (live value on the
    MangaTranslationTab instance, falling back to persisted
    config['manga_skip_inpainting']) so we don't waste time loading a model
    the user explicitly asked us to skip.
    """
    try:
        import os
        from manga_translator import MangaTranslator

        # Respect Skip Inpainter toggle before anything else
        skip_inpainting = False
        try:
            if hasattr(self, 'skip_inpainting_value'):
                skip_inpainting = bool(self.skip_inpainting_value)
            elif hasattr(self, 'main_gui') and getattr(self.main_gui, 'config', None):
                skip_inpainting = bool(self.main_gui.config.get('manga_skip_inpainting', False))
        except Exception:
            skip_inpainting = False
        if skip_inpainting:
            print("[PRELOAD_INPAINTER] Skipping preload - Skip Inpainter toggle is ON")
            return None

        # Get current inpainting settings
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')

        if inpaint_method != 'local':
            print(f"[PRELOAD_INPAINTER] Skipping preload - method is {inpaint_method}, not local")
            return None
        
        # Get model path
        model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
        is_custom_image_edit = str(local_model or '').lower() == 'custom-image-edit'
        if is_custom_image_edit and not model_path:
            model_path = getattr(self.main_gui, 'custom_image_edit_endpoint_var', '') or self.main_gui.config.get('custom_image_edit_endpoint', '')
        try:
            if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                model_path = ''
        except Exception:
            pass
        
        # Normalize path
        if model_path and not is_custom_image_edit:
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
                return True
        
        # Try to download if path not found
        if (not is_custom_image_edit) and (not model_path or not os.path.exists(model_path)):
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
                return False
        if is_custom_image_edit and not model_path:
            print("[PRELOAD_INPAINTER] Custom image edit endpoint missing; skipping fake preload")
            return False
        
        if model_path and (is_custom_image_edit or os.path.exists(model_path)):
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
                return True
            else:
                print(f"[PRELOAD_INPAINTER] No instances preloaded (may already exist in pool)")
                return False

        print(f"[PRELOAD_INPAINTER] Model path missing after resolve for {local_model}")
        return False
        
    except Exception as e:
        print(f"[PRELOAD_INPAINTER] Error during preload: {e}")
        import traceback
        print(f"[PRELOAD_INPAINTER] Traceback: {traceback.format_exc()}")
        return False


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
            is_custom_image_edit = str(local_model or '').lower() == 'custom-image-edit'
            if is_custom_image_edit and not model_path:
                model_path = getattr(self.main_gui, 'custom_image_edit_endpoint_var', '') or self.main_gui.config.get('custom_image_edit_endpoint', '')
            try:
                if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                    model_path = ''
            except Exception:
                pass
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if (not is_custom_image_edit) and (not resolved_model_path or not os.path.exists(resolved_model_path)):
                try:
                    print(f"[INPAINT_REGION] Model path not found, downloading {local_model} model...")
                    from local_inpainter import LocalInpainter
                    temp_inpainter = LocalInpainter()
                    resolved_model_path = temp_inpainter.download_jit_model(local_model)
                except Exception as e:
                    print(f"[INPAINT_REGION] Model download failed: {e}")
                    resolved_model_path = None
            
            if (not is_custom_image_edit) and (not resolved_model_path or not os.path.exists(resolved_model_path)):
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
                disable_performance_mode = not is_custom_image_edit and bool(self.main_gui.config.get('manga_disable_inpaint_performance_mode', False))
                cleaned_image = inpainter.inpaint(
                    image,
                    mask,
                    iterations=custom_iterations,
                    _skip_hd=disable_performance_mode,
                    _skip_tiling=disable_performance_mode,
                )
            else:
                print(f"[INPAINT_REGION] Using auto iterations")
                disable_performance_mode = not is_custom_image_edit and bool(self.main_gui.config.get('manga_disable_inpaint_performance_mode', False))
                cleaned_image = inpainter.inpaint(
                    image,
                    mask,
                    _skip_hd=disable_performance_mode,
                    _skip_tiling=disable_performance_mode,
                )
            
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


def _apply_translate_this_text_thinking_override(self) -> Optional[Dict[str, Optional[str]]]:
    """Temporarily disable thinking for the manual Translate This Text action."""
    try:
        manual_edit = ((self.main_gui.config.get('manga_settings', {}) or {}).get('manual_edit', {}) or {})
        disable_thinking = bool(manual_edit.get('translate_this_text_disable_thinking', True))
    except Exception:
        disable_thinking = True
    if not disable_thinking:
        return None
    if os.environ.get('MANGA_MANUAL_TRANSLATE_THINKING_OVERRIDE_ACTIVE') == '1':
        return None

    keys = (
        'MANGA_MANUAL_TRANSLATE_THINKING_OVERRIDE_ACTIVE',
        'ENABLE_ANTHROPIC_THINKING',
        'ENABLE_GEMINI_THINKING',
        'ENABLE_DEEPSEEK_THINKING',
        'ENABLE_GPT_THINKING',
        'GEMINI_THINKING_LEVEL',
        'THINKING_BUDGET',
    )
    original = {key: os.environ.get(key) for key in keys}
    os.environ['MANGA_MANUAL_TRANSLATE_THINKING_OVERRIDE_ACTIVE'] = '1'
    os.environ['ENABLE_ANTHROPIC_THINKING'] = '0'
    os.environ['ENABLE_GEMINI_THINKING'] = '0'
    os.environ['ENABLE_DEEPSEEK_THINKING'] = '0'
    os.environ['ENABLE_GPT_THINKING'] = '0'
    os.environ['GEMINI_THINKING_LEVEL'] = 'minimal'
    os.environ.pop('THINKING_BUDGET', None)
    return original


def _restore_translate_this_text_thinking_override(original: Optional[Dict[str, Optional[str]]]) -> None:
    if original is None:
        return
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _translate_this_text_background(self, message: str, region_index: int):
    """Send text for translation using MangaTranslator in background"""
    try:
        # CRITICAL: Reset stale cancellation flags from previous stop operations
        _reset_cancellation_flags(self)
        
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
        
        # Get model (need it before API key check for prefix detection)
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
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not api_key:
            if not UnifiedClient._model_needs_api_key(model):
                api_key = 'own-auth'
        
        if not api_key:
            self._log("❌ No API key configured", "error")
            return
        
        # Apply GUI environment variables like Start Translation does
        
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

        # Apply multi-key settings before constructing UnifiedClient. This
        # button bypasses the normal manga Start Translation setup path.
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
                try:
                    UnifiedClient.set_in_memory_multi_keys(
                        mk_list,
                        force_rotation=force_rotation,
                        rotation_frequency=rotation_frequency,
                    )
                except Exception as pool_err:
                    print(f"[TRANSLATE_THIS] Failed to load multi-key pool: {pool_err}")
            else:
                os.environ['USE_MULTI_API_KEYS'] = '0'
                os.environ['USE_MULTI_KEYS'] = '0'
                try:
                    UnifiedClient.clear_in_memory_multi_keys()
                except Exception:
                    pass
        except Exception as mk_err:
            print(f"[TRANSLATE_THIS] Failed to apply multi-key settings: {mk_err}")
        
        # Create unified client with main GUI config
        unified_client = UnifiedClient(model=model, api_key=api_key)
        try:
            if use_mk and mk_list:
                UnifiedClient.setup_multi_key_pool(
                    mk_list,
                    force_rotation=force_rotation,
                    rotation_frequency=rotation_frequency,
                )
        except Exception as pool_err:
            print(f"[TRANSLATE_THIS] setup_multi_key_pool failed: {pool_err}")
        
        manga_settings = {}
        manual_edit = {}
        try:
            manga_settings = self.main_gui.config.get('manga_settings', {}) or {}
            manual_edit = manga_settings.get('manual_edit', {}) or {}
        except Exception:
            manga_settings = {}
            manual_edit = {}

        # Get token limit from manga settings
        max_tokens = 2048  # default
        try:
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
        original_thinking_env = _apply_translate_this_text_thinking_override(self)
        try:
            translation_result, finish_reason = unified_client.send(
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=max_tokens
            )
        finally:
            _restore_translate_this_text_thinking_override(original_thinking_env)
        
        # Check if we got a valid translation
        if (UnifiedClient._is_failed_finish_reason(finish_reason)
                or not _manga_output_text(translation_result)):
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
                src_w, src_h = open_image(current_image).size
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
                    base_w, base_h = open_image(base_image).size
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


def render_persisted_translation_state(self, image_path: str, refresh_preview: bool = False):
    """Render one image entirely from persisted editor state.

    Unlike ``save_positions_and_rerender``, this does not read rectangles from
    the currently visible page, so it is safe for background session imports.
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            return None
        state_manager = getattr(self, 'image_state_manager', None)
        state = state_manager.get_state(image_path) if state_manager else {}
        state = state or {}
        translated_texts = state.get('translated_texts') or []
        last_positions = state.get('last_render_positions') or {}
        regions = []
        from manga_translator import TextRegion

        for fallback_index, result in enumerate(translated_texts):
            if not isinstance(result, dict) or result.get('deleted'):
                continue
            translation = str(result.get('translation') or '').strip()
            if not translation:
                continue
            original = result.get('original') or {}
            try:
                region_index = int(original.get('region_index', fallback_index))
            except (TypeError, ValueError):
                region_index = fallback_index
            bbox = last_positions.get(str(region_index)) or result.get('bbox') or []
            if len(bbox) < 4:
                continue
            x, y, width, height = [int(value) for value in bbox[:4]]
            region = TextRegion(
                text=str(original.get('text') or ''),
                vertices=[
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                ],
                bounding_box=(x, y, width, height),
                confidence=1.0,
                region_type='text_block',
                translated_text=translation,
            )
            regions.append(region)

        if not regions:
            return None

        base_image = _resolve_cleaned_image_for_render(self, image_path) or image_path
        try:
            from PIL import Image as _PIL
            source_width, source_height = open_image(image_path).size
            base_width, base_height = open_image(base_image).size
            if (source_width, source_height) != (base_width, base_height):
                scale_x = base_width / max(1, float(source_width))
                scale_y = base_height / max(1, float(source_height))
                for region in regions:
                    x, y, width, height = region.bounding_box
                    x = int(round(x * scale_x))
                    y = int(round(y * scale_y))
                    width = int(round(width * scale_x))
                    height = int(round(height * scale_y))
                    region.bounding_box = (x, y, width, height)
                    region.vertices = [
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y + height),
                    ]
        except Exception:
            pass

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        override_dir = ''
        try:
            override_dir = getattr(self.main_gui, 'config', {}).get('output_directory', '') or ''
        except Exception:
            pass
        override_dir = override_dir or os.environ.get('OUTPUT_DIRECTORY', '')
        output_root = override_dir or os.path.dirname(image_path)
        output_dir = os.path.join(output_root, f"{base_name}_translated")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        return _render_with_manga_translator(
            self,
            base_image,
            regions,
            output_path=output_path,
            original_image_path=image_path,
            switch_tab=False,
            refresh_preview=refresh_preview,
            use_viewer_exclusions=False,
            use_existing_translator=False,
        )
    except Exception as error:
        self._log(f"Failed to render imported session page {os.path.basename(image_path)}: {error}", "warning")
        return None


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
            src_w, src_h = open_image(current_image).size
            base_w, base_h = open_image(base_image).size
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


def _render_with_manga_translator(
    self,
    image_path: str,
    regions,
    output_path: str = None,
    image_bgr=None,
    original_image_path: str = None,
    switch_tab: bool = True,
    refresh_preview: bool = True,
    use_viewer_exclusions: bool = True,
    use_existing_translator: bool = True,
):
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
            if use_existing_translator and hasattr(self, 'translator') and self.translator:
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
                # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
                _uses_own_auth = not UnifiedClient._model_needs_api_key(model)
                if not api_key and not _uses_own_auth:
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
            pil_image = open_page_image(image_path)
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
                        pil_clean = open_image(cleaned_path).convert('RGB')
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
            if use_viewer_exclusions and hasattr(self.image_preview_widget, 'viewer') and self.image_preview_widget.viewer.rectangles:
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
            if refresh_preview and hasattr(self, 'main_gui') and hasattr(self.main_gui, 'refresh_preview_signal'):
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
            if use_viewer_exclusions and original_image_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
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
        if refresh_preview:
            _schedule_rendered_output_refresh(self, rendered_pil, output_path, switch_tab)
        
        print(f"[RENDER] GUI method call completed")
        
        self._log(f"✅ Rendered to: {output_filename}", "success")
        
        print(f"[RENDER] _render_with_manga_translator COMPLETED SUCCESSFULLY\n{'='*80}\n")
        return output_path
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[RENDER] ERROR in _render_with_manga_translator")
        print(f"[RENDER] Error: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[RENDER] Traceback:\n{traceback_str}")
        print(f"{'='*80}\n")
        self._log(f"❌ Rendering error: {str(e)}", "error")
        return None


def _on_translate_all_clicked(self):
    """Translate all images in the preview list"""
    self._log("🚀 Starting batch translation of all images", "info")

    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)

    # Refresh from the main GUI so this button picks up the CURRENT api_key
    # and model without needing a GUI restart (same behaviour as the
    # Start Translation button in manga_integration._start_translation).
    _refresh_live_gui_state_for_manga_action(self)
    
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
                
                # Reuse imported/saved OCR when it is available. This is what
                # makes the manual editor's Import action useful for Translate
                # All; pages without saved OCR retain the normal detection path.
                saved_state = self.image_state_manager.get_state(image_path) or {}
                saved_regions = manga_ocr_io.canonical_regions_from_editor_state(saved_state)
                saved_recognized = list(saved_state.get('recognized_texts') or [])
                use_saved_ocr = bool(saved_regions and saved_recognized)

                if use_saved_ocr:
                    synthesized = manga_ocr_io.editor_state_from_page({'regions': saved_regions})
                    regions = synthesized.get('detection_regions') or []
                    recognized_texts = saved_recognized
                    self._log(
                        f"📥 [{idx}/{total}] Reusing imported/saved OCR "
                        f"({len(recognized_texts)} text regions)",
                        "info",
                    )
                else:
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
                
                # Step 2: Run OCR only when no imported/saved OCR was found.
                if not use_saved_ocr:
                    # ===== CANCELLATION CHECK: Before OCR =====
                    if _is_translation_cancelled(self):
                        self._log(f"⏹ Translation cancelled before OCR on image {idx}/{total}", "warning")
                        print(f"[TRANSLATE_ALL] Cancelled before OCR on image {idx}")
                        break

                    ocr_config = _get_ocr_config(self, )
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
                    inpaint_skipped = bool(inpaint_config.get('skip', False))

                    # Respect Skip Inpainter toggle first
                    if inpaint_skipped:
                        self._log(f"🚫 [{idx}/{total}] Skip Inpainter enabled — skipping cleaning", "info")
                        image_path_for_rendering = image_path
                        self._cleaned_image_path = None
                    # Only run inpainting if method is not 'none' and is 'local' or 'hybrid'
                    elif inpaint_method in ['local', 'hybrid']:
                        self._log(f"🧹 [{idx}/{total}] Cleaning image...", "info")
                        clean_regions = _regions_with_ocr_text(regions, recognized_texts)
                        cleaned_path = _run_inpainting_sync(self, image_path, clean_regions) if clean_regions else None
                        
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
                
                # NOTE: We intentionally do NOT discard results after translation
                # completes with data — the API call already consumed quota.
                # Only bail if the translate function returned nothing (cancelled mid-call).
                
                if not translated_texts:
                    self._log(f"⚠️ [{idx}/{total}] Translation failed", "warning")
                    failed_count += 1
                    continue
                
                self._log(f"✅ [{idx}/{total}] Translated {len(translated_texts)} regions", "success")
                
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
    debug_enabled = (
        os.getenv('DEBUG_MODE', '0') == '1'
        or os.getenv('SHOW_DEBUG_BUTTONS', '0') == '1'
        or os.getenv('MANGA_DEBUG_MODE', '0') == '1'
        or os.getenv('DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE', '0') == '1'
    )
    import builtins as _builtins

    def print(*args, **kwargs):
        try:
            first_arg = str(args[0]) if args else ''
            if first_arg.startswith('[DEBUG]') and not debug_enabled:
                return
        except Exception:
            pass
        return _builtins.print(*args, **kwargs)

    print(f"[DEBUG] Building OCR config for provider: {provider}")
    config = {'provider': provider}
    try:
        cfg = self.main_gui.config if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config') else {}
        config['custom_api_ocr_batch_enabled'] = bool(cfg.get('manga_custom_api_ocr_batch_enabled', True))
        config['custom_api_ocr_batch_size'] = max(1, int(cfg.get('manga_custom_api_ocr_batch_size', 5)))
    except Exception:
        config['custom_api_ocr_batch_enabled'] = True
        config['custom_api_ocr_batch_size'] = 5
    
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
        manga_ocr_disable_thinking = bool(
            ((self.main_gui.config.get('manga_settings', {}) or {}).get('ocr', {}) or {}).get('manga_ocr_disable_thinking', True)
        )
        config['manga_ocr_disable_thinking'] = manga_ocr_disable_thinking
        os.environ['MANGA_OCR_DISABLE_THINKING'] = '1' if manga_ocr_disable_thinking else '0'
        # For custom-api provider, we need to ensure the API key is available
        api_key = os.environ.get('API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
        # Also check the main GUI config for API key
        if not api_key and hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
            api_key = self.main_gui.config.get('api_key', '')
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        _model = ''
        if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
            _model = (self.main_gui.config.get('model', '') or '').lower()
        try:
            from unified_api_client import UnifiedClient as _UC
            _uses_own_auth = not _UC._model_needs_api_key(_model)
        except Exception:
            _uses_own_auth = False
        if api_key:
            print(f"[DEBUG] Using custom-api provider (API key available)")
            print(f"[DEBUG] API key available for custom-api OCR")
        elif _uses_own_auth:
            print(f"[DEBUG] Using custom-api provider (own-auth model: {_model})")
        else:
            print(f"[DEBUG] Using custom-api provider (requires API key)")
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
    print = _manga_debug_print
    # ===== CANCELLATION CHECK: Discard results if stop was clicked =====
    if _is_translation_cancelled(self):
        print(f"[RECOG_RESULTS] Discarding results - stop was clicked")
        return
    
    try:
        recognized_texts = [
            item for item in results['recognized_texts']
            if _manga_output_text(item.get('text'))
        ]
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
    # Discard results on FORCE stop only — graceful stop should keep them
    # since the API call completed and consumed quota.
    if _is_translation_cancelled(self) and os.environ.get('GRACEFUL_STOP') != '1':
        print(f"[TRANSLATE_RESULTS] Discarding results - force stop was clicked")
        return
    
    try:
        translated_texts = [
            {**item, 'translation': _manga_output_text(item.get('translation'))}
            for item in results['translated_texts']
        ]
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


# ===========================================================================
# Split out of ImageRenderer's Qt dialogs (bodies verbatim; the dialogs call these)
# ===========================================================================

def _manual_translate_prompt(self):
    """'Translate This Text' prompt + target language from manga_settings.manual_edit (lifted from
    ImageRenderer._add_context_menu_to_rectangle, U8)."""
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
    return actual_prompt, target_language


def _apply_ocr_text_edit(self, region_index, ocr_text, new_text):
    """Save an edited OCR text into the editor's text maps (the Save step of ImageRenderer._show_ocr_popup, U8)."""
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


def _apply_translation_text_edit(self, region_index, original, translation, new_original, new_translation):
    """Apply an edited original/translation to the editor's text maps; True when anything changed
    (the first Save step of ImageRenderer._show_translation_popup, U8)."""
    changed = False
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
    return changed


def _persist_translation_text_edit(self, region_index, new_original, new_translation):
    """Persist an edited translation into the page's translated_texts; False when the edit belongs
    to another image (the dialog then closes without re-rendering). The second Save step of
    ImageRenderer._show_translation_popup (U8)."""
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
            return False

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
    return True


def _apply_inpaint_iterations(self, region_index, rect_item, value):
    """Set a rectangle's inpainting iterations (-1 = auto) and persist them to the page state
    (ImageRenderer._handle_set_inpainting_iterations after its input dialog, U8)."""
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


# ===========================================================================
# Split out of manga_image_preview's Stop button (bodies verbatim)
# ===========================================================================

def _request_force_stop(mi):
    """Force stop of the manga editor workflow (manga_image_preview's double-click Stop): stop flags,
    env and the module-level hard cancel of MangaTranslator / unified_api_client / TransateKRtoEN."""
    if mi and hasattr(mi, '_log'):
        mi._log("⚡ Double-click detected — forcing immediate stop!", "warning")

    os.environ['TRANSLATION_CANCELLED'] = '1'
    os.environ['GRACEFUL_STOP'] = '0'
    os.environ['WAIT_FOR_CHUNKS'] = '0'

    if mi:
        if hasattr(mi, 'is_running'):
            mi.is_running = False
        if hasattr(mi, 'stop_flag') and mi.stop_flag:
            mi.stop_flag.set()
        if hasattr(mi, '_batch_mode_active'):
            mi._batch_mode_active = False
        if hasattr(mi, 'set_global_cancellation'):
            mi.set_global_cancellation(True)

    # Hard cancel on MangaTranslator
    try:
        from manga_translator import MangaTranslator
        MangaTranslator.set_global_cancellation(True)
        if hasattr(MangaTranslator, 'hard_cancel_all'):
            MangaTranslator.hard_cancel_all()
        print("[STOP] MangaTranslator hard_cancel_all()")
    except ImportError:
        pass

    # === CRITICAL: Module-level stop — matches main stop_translation ===
    # This sets global_stop_flag, closes HTTP sessions/httpx/OpenAI SDK
    # clients, and cancels AuthGPT/AuthGem/Antigravity SSE streams.
    try:
        import unified_api_client
        if hasattr(unified_api_client, 'set_stop_flag'):
            unified_api_client.set_stop_flag(True)
        if hasattr(unified_api_client, 'global_stop_flag'):
            unified_api_client.global_stop_flag = True
        if hasattr(unified_api_client, 'UnifiedClient'):
            unified_api_client.UnifiedClient._global_cancelled = True
        # Hard cancel: close active HTTP sessions to abort in-flight requests
        if hasattr(unified_api_client, 'hard_cancel_all'):
            unified_api_client.hard_cancel_all()
        print("[STOP] unified_api_client hard_cancel_all() + set_stop_flag(True)")
    except Exception as e:
        print(f"[STOP] unified_api_client force-cancel failed: {e}")

    # Also set TransateKRtoEN stop flag
    try:
        import TransateKRtoEN
        if hasattr(TransateKRtoEN, 'set_stop_flag'):
            TransateKRtoEN.set_stop_flag(True)
    except ImportError:
        pass


def _request_graceful_stop(mi):
    """Graceful stop of the manga editor workflow (manga_image_preview's single-click Stop): the
    editor-level flags only, so the in-flight API call finishes."""
    if hasattr(mi, '_log'):
        mi._log("🛑 Graceful stop requested — waiting for in-flight API call to finish", "warning")

    # Set GRACEFUL_STOP env so background threads know this is graceful
    os.environ['GRACEFUL_STOP'] = '1'
    os.environ['WAIT_FOR_CHUNKS'] = '1'

    # Set is_running to False (prevents new operations from starting)
    if hasattr(mi, 'is_running'):
        mi.is_running = False

    # Set stop_flag (checked by _is_translation_cancelled)
    if hasattr(mi, 'stop_flag') and mi.stop_flag:
        mi.stop_flag.set()
        print("[STOP] Set stop_flag")

    # Clear batch mode flag (prevents next image in batch)
    if hasattr(mi, '_batch_mode_active'):
        mi._batch_mode_active = False
        print("[STOP] Cleared batch mode flag")

    # Set _global_cancellation on manga_integration instance
    # This is checked by _is_translation_cancelled() in ImageRenderer
    if hasattr(mi, '_global_cancellation'):
        mi._global_cancellation = True
        print("[STOP] Set _global_cancellation on manga_integration")


def _delete_translated_outputs(self):
    """Clear Boxes: delete the page's translated output image (never the cleaned one) from the
    ``<name>_translated`` folder under the OUTPUT_DIRECTORY override and the source folder (lifted
    verbatim from manga_image_preview's ``_on_clear_boxes_clicked`` in U9; ``self`` is the preview
    widget: ``current_image_path`` and ``main_gui``)."""
    if self.current_image_path:
        # Get OUTPUT_DIRECTORY override if set
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')

        source_dir = os.path.dirname(self.current_image_path)
        source_filename = os.path.basename(self.current_image_path)
        source_name_no_ext = os.path.splitext(source_filename)[0]

        # Build list of directories to check (override dir first, then source dir)
        search_dirs = []
        if override_dir:
            search_dirs.append(override_dir)
            print(f"[CLEAR] Checking OUTPUT_DIRECTORY override: {override_dir}")
        search_dirs.append(source_dir)

        # Check each directory for translated folder
        for check_dir in search_dirs:
            translated_folder = os.path.join(check_dir, f"{source_name_no_ext}_translated")

            # Delete translated output file (non-cleaned file) from isolated folder
            if os.path.exists(translated_folder):
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
                for filename in os.listdir(translated_folder):
                    name_lower = filename.lower()
                    # Find and delete files that match the source name but NOT cleaned files
                    if (name_lower.startswith(source_name_no_ext.lower()) and
                        name_lower.endswith(image_extensions) and
                        '_cleaned' not in name_lower):
                        translated_path = os.path.join(translated_folder, filename)
                        try:
                            os.remove(translated_path)
                            print(f"[CLEAR] Deleted translated output: {os.path.basename(translated_path)}")
                        except Exception as e:
                            print(f"[CLEAR] Failed to delete translated output: {e}")


def _clear_saved_page_state(self):
    """Clear Boxes: drop the page's saved OCR / translation / overlay / box state and flush it (lifted
    verbatim from manga_image_preview's ``_on_clear_boxes_clicked`` in U9; ``self`` is the preview
    widget: ``manga_integration`` and ``current_image_path``)."""
    if hasattr(self.manga_integration, 'image_state_manager') and self.manga_integration.image_state_manager and self.current_image_path:
        st = self.manga_integration.image_state_manager.get_state(self.current_image_path) or {}
        # Clear ALL saved state data - OCR, translations, overlays, etc.
        st.pop('overlay_offsets', None)
        st.pop('last_render_positions', None)
        st.pop('translated_texts', None)
        st.pop('recognized_texts', None)  # Clear OCR data
        st.pop('detection_regions', None)  # Clear detection data
        st.pop('viewer_rectangles', None)  # Clear rectangle data
        self.manga_integration.image_state_manager.set_state(self.current_image_path, st, save=True)
        # Force immediate flush to disk to ensure deletion persists
        self.manga_integration.image_state_manager.flush()
        print(f"[CLEAR] Flushed cleared state to disk for {os.path.basename(self.current_image_path)}")


# ===========================================================================
# Moved from manga_integration.py: the per-image editor state (worker gated for mobile)
# ===========================================================================

# Module-level worker function for state management (must be picklable)
def _state_manager_worker_process(task_queue, result_queue, state_file_path):
    """Worker process for async state persistence operations.
    
    Runs in separate process to avoid blocking UI during JSON I/O.
    """
    import os
    import json
    from queue import Empty
    
    # Load initial state
    states = {}
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f:
                states = json.load(f)
        except Exception:
            states = {}
    
    while True:
        try:
            task = task_queue.get(timeout=0.5)
            
            if task is None or task.get('type') == 'shutdown':
                # Final save before shutdown
                try:
                    os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
                    temp_path = state_file_path + '.tmp'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(states, f, indent=2)
                    if os.path.exists(state_file_path):
                        os.remove(state_file_path)
                    os.rename(temp_path, state_file_path)
                except Exception:
                    pass
                break
            
            task_type = task.get('type')
            
            if task_type == 'update_state':
                image_path = task.get('image_path')
                updates = task.get('updates', {})
                if image_path:
                    if image_path not in states:
                        states[image_path] = {}
                    states[image_path].update(updates)
                    result_queue.put({'type': 'state_updated', 'image_path': image_path})
            
            elif task_type == 'set_state':
                image_path = task.get('image_path')
                state = task.get('state', {})
                if image_path:
                    states[image_path] = state
                    result_queue.put({'type': 'state_set', 'image_path': image_path})
            
            elif task_type == 'clear_state':
                image_path = task.get('image_path')
                if image_path and image_path in states:
                    del states[image_path]
                    result_queue.put({'type': 'state_cleared', 'image_path': image_path})
            
            elif task_type == 'clear_all_states':
                states.clear()
                result_queue.put({'type': 'all_states_cleared'})
            
            elif task_type == 'save_now':
                # Save to disk (legacy - kept for compatibility)
                try:
                    os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
                    temp_path = state_file_path + '.tmp'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(states, f, indent=2)
                    if os.path.exists(state_file_path):
                        os.remove(state_file_path)
                    os.rename(temp_path, state_file_path)
                    result_queue.put({'type': 'saved', 'count': len(states)})
                except Exception as e:
                    result_queue.put({'type': 'save_error', 'error': str(e)})
            
            elif task_type == 'save_state_snapshot':
                # Save a state snapshot from main thread
                state_snapshot = task.get('state', {})
                try:
                    os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
                    temp_path = state_file_path + '.tmp'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(state_snapshot, f, indent=2)
                    if os.path.exists(state_file_path):
                        os.remove(state_file_path)
                    os.rename(temp_path, state_file_path)
                    # Update worker's own state to match
                    states = state_snapshot
                    result_queue.put({'type': 'saved', 'count': len(states)})
                except Exception as e:
                    result_queue.put({'type': 'save_error', 'error': str(e)})
            
            elif task_type == 'get_state':
                image_path = task.get('image_path')
                state = states.get(image_path, {})
                # Remove exclusions on read
                if 'excluded_from_clean' in state:
                    state = dict(state)
                    del state['excluded_from_clean']
                    states[image_path] = state
                result_queue.put({'type': 'state_result', 'image_path': image_path, 'state': state})
        
        except Empty:
            continue
        except Exception:
            continue


class ImageStateManager:
    """
    Manages per-image state persistence (detection rects, recognition overlays, paths, status).
    Saves state to JSON with debounced writes using worker process to avoid UI blocking.
    """
    def __init__(self, state_file_path: str):
        self.state_file_path = state_file_path
        self._states: Dict[str, Dict[str, Any]] = {}  # Local cache
        self._dirty = False
        self._save_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._async_flush_lock = threading.Lock()
        self._async_flush_requested = False
        self._async_flush_thread = None

        # Worker process for async state operations
        self._mp_enabled = True
        self._mp_ctx = None
        self._mp_task_q = None
        self._mp_result_q = None
        self._mp_worker = None
        self._mp_receiver_thread = None

        # Start worker
        if self._mp_enabled:
            self._start_worker()

        # Load initial state synchronously
        self._load_state()
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path to forward slashes to avoid duplicates from mixed separators"""
        return path.replace('\\', '/')
    def _start_worker(self):
        """Start worker process for async state operations"""
        try:
            # Don't start worker if already shutting down
            if getattr(self, '_mp_worker', None) and self._mp_worker.is_alive():
                return
            
            # Mobile (U8): no worker process on Android/iOS. The state stays in this process and
            # flush() / flush_async() write it (the worker only mirrored it).
            if not mobile_runtime.processes_available():
                self._mp_enabled = False
                return
            
            import multiprocessing as mp
            try:
                self._mp_ctx = mp.get_context('spawn')
            except Exception:
                self._mp_ctx = mp
            
            self._mp_task_q = self._mp_ctx.Queue()
            self._mp_result_q = self._mp_ctx.Queue()
            
            self._mp_worker = self._mp_ctx.Process(
                target=_state_manager_worker_process,
                args=(self._mp_task_q, self._mp_result_q, self.state_file_path),
                daemon=True
            )
            self._mp_worker.start()
            
            # Receiver thread
            def _receiver():
                while True:
                    try:
                        from queue import Empty
                        msg = self._mp_result_q.get(timeout=0.1)
                        # Process results if needed
                        if msg.get('type') == 'saved':
                            print(f"[STATE] Worker saved {msg.get('count', 0)} states")
                    except Empty:
                        continue
                    except Exception:
                        break
            
            self._mp_receiver_thread = threading.Thread(target=_receiver, daemon=True)
            self._mp_receiver_thread.start()
            
        except Exception as e:
            print(f"[STATE] Failed to start worker: {e}")
            self._mp_enabled = False
    
    def _stop_worker(self):
        """Stop worker process"""
        try:
            # Disable multiprocessing first to prevent new tasks
            self._mp_enabled = False
            
            # Send shutdown signal
            if self._mp_task_q:
                try:
                    self._mp_task_q.put({'type': 'shutdown'}, timeout=0.5)
                except Exception:
                    pass
            
            # Wait for worker to finish
            if self._mp_worker and self._mp_worker.is_alive():
                try:
                    self._mp_worker.join(timeout=1.0)
                except Exception:
                    pass
                
                # Force terminate if still alive
                if self._mp_worker.is_alive():
                    try:
                        self._mp_worker.terminate()
                        self._mp_worker.join(timeout=0.5)
                    except Exception:
                        pass
            
            # Close queues to release handles
            if self._mp_task_q:
                try:
                    self._mp_task_q.close()
                    self._mp_task_q.join_thread()
                except Exception:
                    pass
            
            if self._mp_result_q:
                try:
                    self._mp_result_q.close()
                    self._mp_result_q.join_thread()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"[STATE] Error stopping worker: {e}")
        finally:
            self._mp_enabled = False
            self._mp_worker = None
            self._mp_task_q = None
            self._mp_result_q = None
    
    def _load_state(self):
        """Load state from JSON file if it exists (synchronous on startup)"""
        print = _manga_cmd_debug_print
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    raw_states = json.load(f)
                
                # Normalize all paths and merge duplicates
                self._states = {}
                duplicates_merged = 0
                for img_path, state in raw_states.items():
                    normalized = self._normalize_path(img_path)
                    if normalized in self._states:
                        # Merge: prefer state with OCR data
                        existing = self._states[normalized]
                        if 'recognized_texts' not in existing or not existing.get('recognized_texts'):
                            # Existing has no OCR, use new state
                            self._states[normalized] = state
                            duplicates_merged += 1
                        elif 'recognized_texts' in state and state.get('recognized_texts'):
                            # Both have OCR, keep the one with more data
                            if len(state.get('recognized_texts', [])) > len(existing.get('recognized_texts', [])):
                                self._states[normalized] = state
                                duplicates_merged += 1
                        # Otherwise keep existing
                    else:
                        self._states[normalized] = state
                
                print(f"[STATE] Loaded state for {len(self._states)} images from {self.state_file_path}")
                if duplicates_merged > 0:
                    print(f"[STATE] Merged {duplicates_merged} duplicate path entries")
                    self._dirty = True  # Mark for save to clean up duplicates
                
                # Debug: print first image's keys
                if self._states:
                    first_img = next(iter(self._states.keys()))
                    print(f"[STATE DEBUG] First image keys: {list(self._states[first_img].keys())}")
            except Exception as e:
                print(f"[STATE] Failed to load state: {e}")
                self._states = {}
        else:
            print(f"[STATE DEBUG] No state file found at {self.state_file_path}")
    
    def _save_state_now(self):
        """Debounced save - does nothing. Only flush() saves to avoid race conditions."""
        # DO NOT SAVE HERE - race conditions with worker updates
        # Only flush() does synchronous save to guarantee consistency
        pass
    
    def _schedule_save(self):
        """Schedule a debounced save (wait 2 seconds after last change)"""
        self._dirty = True
        
        with self._timer_lock:
            # Cancel existing timer if any
            if self._save_timer is not None:
                self._save_timer.cancel()
            
            # Create new timer for debounced save
            self._save_timer = threading.Timer(2.0, self._save_state_now)
            self._save_timer.daemon = True
            self._save_timer.start()
    
    def get_state(self, image_path: str) -> Dict[str, Any]:
        """Get state for an image (returns empty dict if not found)"""
        image_path = self._normalize_path(image_path)
        state = self._states.get(image_path, {})
        
        # EXCLUSION PERSISTENCE REMOVAL: Always start with no exclusions across new sessions
        # Remove any saved exclusion state to ensure exclusions don't persist
        if 'excluded_from_clean' in state:
            print(f"[EXCLUSION_RESET] Removing saved exclusion state - exclusions always start disabled")
            del state['excluded_from_clean']
            # Update the saved state to remove persistent exclusions
            self._states[image_path] = state
            self._schedule_save()
        
        return state
    
    def set_state(self, image_path: str, state: Dict[str, Any], save: bool = True):
        """Set state for an image and optionally schedule save"""
        image_path = self._normalize_path(image_path)
        self._states[image_path] = state
        if getattr(self, '_mp_enabled', False) and self._mp_task_q and save:
            try:
                self._mp_task_q.put({'type': 'set_state', 'image_path': image_path, 'state': state}, timeout=0.1)
            except Exception:
                pass
        if save:
            self._schedule_save()
    
    def update_state(self, image_path: str, updates: Dict[str, Any], save: bool = True):
        """Update specific fields in image state"""
        print = _manga_cmd_debug_print
        image_path = self._normalize_path(image_path)
        if image_path not in self._states:
            self._states[image_path] = {}
        self._states[image_path].update(updates)
        print(f"[STATE DEBUG] update_state: {image_path} keys={list(updates.keys())}")
        if getattr(self, '_mp_enabled', False) and self._mp_task_q and save:
            try:
                self._mp_task_q.put({'type': 'update_state', 'image_path': image_path, 'updates': updates}, timeout=0.1)
            except Exception:
                pass
        if save:
            self._schedule_save()
    
    def clear_state(self, image_path: str, save: bool = True):
        """Clear state for an image"""
        image_path = self._normalize_path(image_path)
        if image_path in self._states:
            del self._states[image_path]
            if getattr(self, '_mp_enabled', False) and self._mp_task_q:
                try:
                    self._mp_task_q.put({'type': 'clear_state', 'image_path': image_path}, timeout=0.1)
                except Exception:
                    pass
            if save:
                self._schedule_save()
    
    def clear_all_states(self, save: bool = True):
        """Clear all saved states for all images"""
        print = _manga_cmd_debug_print
        self._states.clear()
        if getattr(self, '_mp_enabled', False) and self._mp_task_q:
            try:
                self._mp_task_q.put({'type': 'clear_all_states'}, timeout=0.1)
            except Exception:
                pass
        print(f"[STATE] Cleared all saved states")
        if save:
            self._schedule_save()
    
    def flush(self):
        """Force immediate save if dirty - saves synchronously to guarantee persistence"""
        print = _manga_cmd_debug_print
        print(f"[STATE DEBUG] flush() called, dirty={self._dirty}, images={len(self._states)}")
        
        # Debug: Show ALL images with recognized_texts
        if self._states:
            images_with_ocr = []
            for img_path, state in self._states.items():
                if 'recognized_texts' in state and state['recognized_texts']:
                    ocr_count = len(state['recognized_texts'])
                    # Show last 60 chars of path to distinguish different directories
                    display_path = img_path if len(img_path) <= 60 else '...' + img_path[-57:]
                    images_with_ocr.append((display_path, ocr_count))
            
            if images_with_ocr:
                print(f"[STATE DEBUG] Found {len(images_with_ocr)} images with OCR data")
                for path, count in images_with_ocr[:5]:
                    print(f"[STATE DEBUG]   - {path}: {count} recognized texts")
                if len(images_with_ocr) > 5:
                    print(f"[STATE DEBUG]   ... {len(images_with_ocr) - 5} more omitted")
            else:
                print(f"[STATE DEBUG] NO images have OCR data!")
            
            # Also show first 3 images for context
            print(f"[STATE DEBUG] First 3 images in state:")
            for idx, (img_path, state) in enumerate(list(self._states.items())[:3]):
                keys = list(state.keys())
                has_ocr = 'recognized_texts' in state
                ocr_count = len(state.get('recognized_texts', [])) if has_ocr else 0
                print(f"[STATE DEBUG] Image {idx+1}: {os.path.basename(img_path)}")
                print(f"[STATE DEBUG]   Keys: {keys}")
                print(f"[STATE DEBUG]   Has recognized_texts: {has_ocr}, count: {ocr_count}")
        
        if self._dirty:
            # Cancel pending timer
            with self._timer_lock:
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
            
            # Save synchronously from main thread to guarantee data persistence
            # Don't rely on worker queue during shutdown
            try:
                os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
                temp_path = self.state_file_path + '.tmp'
                print(f"[STATE DEBUG] Writing {len(self._states)} states to {temp_path}")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._states, f, indent=2)
                if os.path.exists(self.state_file_path):
                    os.remove(self.state_file_path)
                os.rename(temp_path, self.state_file_path)
                self._dirty = False
                print(f"[STATE] Flushed state for {len(self._states)} images to {self.state_file_path}")
            except Exception as e:
                print(f"[STATE] Failed to flush state: {e}")
        else:
            print(f"[STATE DEBUG] flush() skipped - not dirty")

    def flush_async(self):
        """Persist the current state snapshot without blocking the GUI thread."""
        self._dirty = True
        with self._async_flush_lock:
            self._async_flush_requested = True
            if self._async_flush_thread and self._async_flush_thread.is_alive():
                return

            def _flush_worker():
                while True:
                    with self._async_flush_lock:
                        self._async_flush_requested = False
                    self.flush()
                    with self._async_flush_lock:
                        if not self._async_flush_requested:
                            break

            self._async_flush_thread = threading.Thread(
                target=_flush_worker,
                name="MangaStateFlush",
                daemon=True,
            )
            self._async_flush_thread.start()
    
    def __del__(self):
        """Cleanup: flush state, stop worker, and cancel timer on deletion"""
        try:
            self.flush()
            self._stop_worker()
        except:
            pass


# ===========================================================================
# Registry
# ===========================================================================

#: ImageRenderer functions whose bodies live here (moved verbatim; see EDITED_FUNCTIONS) plus the
#: helpers split out of ImageRenderer's dialogs. bind_editor_namespace re-binds all of them.
EDITOR_FUNCTIONS = (
    '_manga_debug_logging_enabled',
    '_manga_debug_print',
    '_normalize_region_kind',
    '_is_free_text_region_metadata',
    '_preserve_free_text_inpaint_enabled',
    '_copy_region_metadata_to_item',
    '_saved_rect_metadata',
    '_manga_output_text',
    '_regions_with_ocr_text',
    '_reset_cancellation_flags',
    '_is_translation_cancelled',
    '_on_detect_text_clicked',
    '_run_detect_background',
    '_clear_detection_state_for_image',
    '_clear_cross_image_state',
    '_persist_current_image_state',
    '_rehydrate_text_state_from_persisted',
    '_validate_and_clean_stale_state',
    '_process_detect_results',
    '_on_clean_image_clicked',
    '_extract_regions_from_preview',
    '_run_clean_background',
    '_get_detection_config',
    '_get_inpaint_config',
    '_run_detection_sync',
    '_run_inpainting_sync',
    '_run_ocr_on_regions',
    '_on_recognize_text_clicked',
    '_run_recognize_background',
    '_refresh_live_gui_state_for_manga_action',
    '_get_loaded_manga_glossary_for_workflow',
    '_manga_workflow_glossary_compression_enabled',
    '_compress_loaded_manga_glossary_for_workflow',
    '_append_loaded_manga_glossary_to_system_prompt',
    '_on_translate_text_clicked',
    '_run_full_translate_pipeline',
    '_manual_translate_full_page_context_enabled',
    '_run_translate_background',
    '_translate_with_full_page_context',
    '_translate_individually',
    '_get_system_prompt_from_gui',
    '_update_rectangles_with_recognition',
    '_remove_processing_overlay',
    '_handle_ocr_this_text',
    '_process_ocr_result',
    '_handle_ocr_error',
    '_handle_delete_rectangle',
    '_handle_toggle_free_text_region',
    '_handle_clean_this_rectangle',
    '_get_or_create_shared_inpainter',
    '_preload_shared_bubble_detector',
    '_preload_shared_inpainter',
    '_run_inpainting_on_region',
    '_update_image_preview_with_result',
    '_get_custom_iterations_for_regions',
    '_handle_translate_this_text',
    '_apply_translate_this_text_thinking_override',
    '_restore_translate_this_text_thinking_override',
    '_translate_this_text_background',
    '_clean_up_deleted_rectangle_overlays',
    '_get_translation_text_for_region',
    '_resolve_cleaned_image_for_render',
    '_update_single_text_overlay',
    'render_persisted_translation_state',
    'save_positions_and_rerender',
    '_render_with_manga_translator',
    '_on_translate_all_clicked',
    '_run_translate_all_background',
    '_get_ocr_config',
    '_process_recognize_results',
    '_process_translate_results',
    '_manual_translate_prompt',
    '_apply_ocr_text_edit',
    '_apply_translation_text_edit',
    '_persist_translation_text_edit',
    '_apply_inpaint_iterations',
)
#: Moved functions with a listed edit (Qt lines -> hooks, the google_vision_rest fallback, the
#: safe_image page reads: cv2_imread, open_page_image).
EDITED_FUNCTIONS = ('_run_ocr_on_regions', '_update_rectangles_with_recognition', '_process_ocr_result', '_handle_clean_this_rectangle', '_render_with_manga_translator', '_run_detect_background', '_run_clean_background', '_run_detection_sync', '_run_inpainting_sync',)
#: Split helpers (bodies lifted verbatim out of ImageRenderer / manga_image_preview handlers).
SPLIT_HELPERS = ('_manual_translate_prompt', '_apply_ocr_text_edit', '_apply_translation_text_edit', '_persist_translation_text_edit', '_apply_inpaint_iterations', '_request_force_stop', '_request_graceful_stop',)
#: Plain names the desktop namespace receives as well (constants / helpers without a Qt twin).
SHARED_NAMES = ('_REGION_METADATA_KEYS', '_import_google_vision')


# ===========================================================================
# Hook stubs: GUI-free stand-ins for the Qt-only ImageRenderer helpers the moved code calls.
# On the desktop the moved functions run in ImageRenderer's namespace and call the real Qt
# helpers; here (the mobile session) they reach the host's ``_manga_editor_hook``.
# ===========================================================================

def _editor_hook(self, name, *args, **kwargs):
    """Forward a Qt-only editor step to the host's ``_manga_editor_hook`` (None without one)."""
    handler = getattr(self, '_manga_editor_hook', None)
    if callable(handler):
        return handler(name, *args, **kwargs)
    return None


def _add_context_menu_to_rectangle(self, rect_item, region_index):
    return _editor_hook(self, '_add_context_menu_to_rectangle', rect_item, region_index)


def _add_processing_overlay(self):
    return _editor_hook(self, '_add_processing_overlay')


def _add_rectangle_pulse_effect(self, rect_item, region_index, auto_remove=False):
    return _editor_hook(self, '_add_rectangle_pulse_effect', rect_item, region_index, auto_remove=auto_remove)


def _remove_rectangle_pulse_effect(self, rect_item, region_index):
    return _editor_hook(self, '_remove_rectangle_pulse_effect', rect_item, region_index)


def _add_text_overlay_to_viewer(self, translated_texts):
    return _editor_hook(self, '_add_text_overlay_to_viewer', translated_texts)


def _attach_move_sync_to_rectangle(self, rect_item, region_index):
    return _editor_hook(self, '_attach_move_sync_to_rectangle', rect_item, region_index)


def _apply_rectangle_clean_style(rect_item, *, is_recognized=False, excluded=False):
    """Qt pen/brush styling on the desktop; the mobile canvas derives colours from the box flags."""
    return None


def _style_recognized_rectangle(rect_item):
    """Qt pen/brush styling on the desktop; the moved code sets ``is_recognized`` itself."""
    return None


def _disable_workflow_buttons(self, exclude=None, show_stop_button=True):
    return _editor_hook(self, '_disable_workflow_buttons', exclude=exclude, show_stop_button=show_stop_button)


def _draw_detection_boxes_on_preview(self):
    return _editor_hook(self, '_draw_detection_boxes_on_preview')


def _restore_detect_button(self):
    return _editor_hook(self, '_restore_detect_button')


def _restore_clean_button(self):
    return _editor_hook(self, '_restore_clean_button')


def _restore_recognize_button(self):
    return _editor_hook(self, '_restore_recognize_button')


def _restore_translate_button(self):
    return _editor_hook(self, '_restore_translate_button')


def _restore_translate_all_button(self):
    return _editor_hook(self, '_restore_translate_all_button')


def _confirm_clean_excluded_rectangle(self, region_index):
    """Desktop: a Yes/No dialog. Mobile: the host's answer (the app confirms before it asks), default yes."""
    answer = _editor_hook(self, '_confirm_clean_excluded_rectangle', region_index)
    return True if answer is None else bool(answer)


def _schedule_rendered_output_refresh(self, rendered_pil, output_path, switch_tab):
    return _editor_hook(self, '_schedule_rendered_output_refresh', rendered_pil, output_path, switch_tab)


#: Qt-only ImageRenderer helpers the moved code calls (stubs above); the desktop has the real ones.
QT_HELPER_STUBS = (
    '_add_context_menu_to_rectangle', '_add_processing_overlay', '_add_rectangle_pulse_effect',
    '_remove_rectangle_pulse_effect', '_add_text_overlay_to_viewer', '_attach_move_sync_to_rectangle',
    '_apply_rectangle_clean_style', '_disable_workflow_buttons', '_draw_detection_boxes_on_preview',
    '_restore_detect_button', '_restore_clean_button', '_restore_recognize_button',
    '_restore_translate_button', '_restore_translate_all_button',
)
#: Hooks that replaced Qt lines inside moved bodies; ImageRenderer defines their desktop halves.
DESKTOP_HOOKS = ('_style_recognized_rectangle', '_confirm_clean_excluded_rectangle', '_schedule_rendered_output_refresh')


# ===========================================================================
# Desktop binding
# ===========================================================================

def _rebind(function, namespace):
    """``function``'s code object as a function of ``namespace`` (same behaviour, other globals)."""
    clone = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = dict(function.__kwdefaults__) if function.__kwdefaults__ else None
    clone.__doc__ = function.__doc__
    clone.__annotations__ = dict(function.__annotations__)
    clone.__qualname__ = function.__qualname__
    clone.__dict__.update(function.__dict__)
    clone.__module__ = namespace.get('__name__', function.__module__)
    return clone


def bind_editor_namespace(namespace):
    """Install ``EDITOR_FUNCTIONS`` into a desktop module namespace (``ImageRenderer.globals()``).

    The functions are re-bound, not imported: they resolve every global name in ``namespace``,
    so on the desktop they call ImageRenderer's Qt helpers and the tests' monkeypatches of
    ``ImageRenderer.<name>`` reach them, exactly as before the move. Returns ``namespace``.
    """
    current = globals()
    for name in EDITOR_FUNCTIONS:
        namespace[name] = _rebind(current[name], namespace)
    for name in SHARED_NAMES:
        namespace[name] = current[name]
    return namespace


# ===========================================================================
# The manga tab's settings the moved functions read
# ===========================================================================

#: MangaTranslationTab attributes the moved editor functions read (``getattr(self, ...)``): the
#: inpainting method / model / skip, text rendering, full-page context, the OCR prompt / provider
#: and the glossary paths. The session takes them from manga_env.HeadlessMangaState, which runs
#: the tab's own (moved) ``_load_rendering_settings`` / start-up steps on the job's owner.
TAB_ATTRIBUTES = (
    'inpaint_method_value', 'local_model_type_value', 'local_model_path_value', 'skip_inpainting_value',
    'inpaint_quality_value', 'inpaint_dilation_value', 'inpaint_passes_value',
    'bg_opacity_value', 'free_text_only_bg_opacity_value', 'bg_style_value', 'bg_reduction_value',
    'font_size_value', 'selected_font_path', 'font_size_mode_value', 'font_size_multiplier_value',
    'auto_min_size_value', 'max_font_size_value', 'force_caps_lock_value', 'constrain_to_bubble_value',
    'strict_text_wrapping_value', 'safe_area_enabled_value', 'safe_area_scale_value',
    'text_color_r_value', 'text_color_g_value', 'text_color_b_value', 'shadow_enabled_value',
    'shadow_color_r_value', 'shadow_color_g_value', 'shadow_color_b_value', 'shadow_offset_x_value',
    'shadow_offset_y_value', 'shadow_blur_value', 'font_style_value', 'full_page_context_value',
    'ocr_prompt', 'ocr_provider_value', 'manga_custom_glossary_path', 'manga_generated_glossary_path',
)


class _TabLogHost:
    """job_runner.JobHost-shaped log sink for HeadlessMangaState (its lines go to the session log)."""

    def __init__(self, session):
        self._session = session

    def log(self, message, level='info'):
        self._session._log(message, level)


def default_state_file():
    """Where the per-image editor state lives: the desktop's ``<src>/../.glossarion/image_state.json``
    (manga_integration.MangaTranslationTab), under ``GLOSSARION_DATA_DIR`` when that is set (mobile)."""
    data_dir = mobile_runtime.data_dir('')
    if data_dir:
        return os.path.join(data_dir, '.glossarion', 'image_state.json')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.glossarion', 'image_state.json')


# ===========================================================================
# Mobile host: the preview-widget stand-ins the moved code reads
# ===========================================================================

class _Rect:
    """QRectF stand-in (``x()``, ``y()``, ``width()``, ``height()``) in image pixels."""

    __slots__ = ('_x', '_y', '_w', '_h')

    def __init__(self, x, y, width, height):
        self._x, self._y, self._w, self._h = float(x), float(y), float(width), float(height)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h

    def __eq__(self, other):
        return isinstance(other, _Rect) and (self._x, self._y, self._w, self._h) == (other._x, other._y, other._w, other._h)

    def __repr__(self):
        return f"_Rect({self._x:g}, {self._y:g}, {self._w:g}, {self._h:g})"


class _Point:
    """QPointF stand-in."""

    __slots__ = ('_x', '_y')

    def __init__(self, x, y):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


class _PolygonPath:
    """QPainterPath stand-in for lasso boxes (``toFillPolygon()``)."""

    def __init__(self, points):
        self._points = [(float(px), float(py)) for px, py in points]

    def toFillPolygon(self):
        return [_Point(px, py) for px, py in self._points]


class _NullScene:
    """QGraphicsScene stand-in: the moved code adds / removes Qt items and repaints; nothing to do."""

    def addItem(self, item):
        return None

    def removeItem(self, item):
        return None

    def update(self, *args):
        return None


class EditorBox:
    """One editor shape: the stand-in for manga_image_preview's Moveable{Rect,Ellipse,Path}Item.

    Geometry is in image pixels (the desktop viewer's scene coordinates). ``shape_type`` is
    ``'rect'``, ``'ellipse'`` or ``'polygon'`` (``polygon`` holds the lasso points). The flags the
    moved code reads and writes are plain attributes: ``region_index``, ``is_recognized``,
    ``exclude_from_clean``, ``inpaint_iterations`` and the detector metadata
    ``bubble_type`` / ``region_type`` / ``bubble_bounds``.
    """

    def __init__(self, x, y, width, height, *, shape='rect', polygon=None, region_index=None,
                 bubble_type=None, region_type=None, bubble_bounds=None):
        self.shape_type = shape if shape in ('rect', 'ellipse', 'polygon') else 'rect'
        self.polygon = None
        self.region_index = region_index
        self.is_recognized = False
        self.exclude_from_clean = False
        self.inpaint_iterations = None
        self.bubble_type = bubble_type
        self.region_type = region_type
        self.bubble_bounds = bubble_bounds
        self._x = self._y = self._w = self._h = 0.0
        if self.shape_type == 'polygon' and polygon and len(polygon) >= 3:
            self.polygon = [[float(px), float(py)] for px, py in polygon]
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            self._x, self._y = min(xs), min(ys)
            self._w, self._h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            if self.shape_type == 'polygon':
                self.shape_type = 'rect'
            self.set_geometry(x, y, width, height)

    # ---- the QGraphicsItem surface the moved code reads ----
    def sceneBoundingRect(self):
        return _Rect(self._x, self._y, self._w, self._h)

    def rect(self):
        return self.sceneBoundingRect()

    def path(self):
        if self.polygon:
            return _PolygonPath(self.polygon)
        x, y, w, h = self._x, self._y, self._w, self._h
        return _PolygonPath([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    def mapToScene(self, polygon):
        return polygon  # boxes live in scene (image) coordinates

    def setPen(self, *args):
        return None

    def setBrush(self, *args):
        return None

    # ---- editing ----
    def set_geometry(self, x, y, width, height):
        """Move / resize; a lasso polygon is translated and scaled with its bounding box."""
        x, y = float(x), float(y)
        width, height = max(1.0, float(width)), max(1.0, float(height))
        if self.polygon and self._w > 0 and self._h > 0:
            sx, sy = width / self._w, height / self._h
            self.polygon = [[x + (px - self._x) * sx, y + (py - self._y) * sy] for px, py in self.polygon]
        self._x, self._y, self._w, self._h = x, y, width, height

    @property
    def bbox(self):
        """``[x, y, width, height]`` in whole pixels (the moved code's int() of sceneBoundingRect)."""
        return [int(self._x), int(self._y), int(self._w), int(self._h)]

    @property
    def is_free_text(self):
        return _is_free_text_region_metadata(rect_item=self)

    def state_entry(self):
        """The ``viewer_rectangles`` record manga_image_preview._persist_rectangles_state writes."""
        entry = {'x': self._x, 'y': self._y, 'width': self._w, 'height': self._h, 'shape': self.shape_type}
        for key in _REGION_METADATA_KEYS:
            value = getattr(self, key, None)
            if value is not None:
                entry[key] = value
        if self.shape_type == 'polygon' and self.polygon and len(self.polygon) >= 3:
            entry['polygon'] = [[float(px), float(py)] for px, py in self.polygon]
        return entry

    def to_dict(self):
        """Box snapshot for the UI."""
        return {
            'index': self.region_index, 'x': self._x, 'y': self._y, 'width': self._w, 'height': self._h,
            'shape': self.shape_type, 'polygon': copy.deepcopy(self.polygon),
            'is_recognized': bool(self.is_recognized), 'exclude_from_clean': bool(self.exclude_from_clean),
            'inpaint_iterations': self.inpaint_iterations, 'free_text': bool(self.is_free_text),
            'bubble_type': self.bubble_type, 'region_type': self.region_type,
        }

    def __repr__(self):
        return f"EditorBox({self.region_index}, {self.shape_type}, {self.bbox})"


class EditorViewer:
    """The preview viewer the moved code reads: ``rectangles`` (EditorBox list), ``overlay_rects``,
    ``clear_rectangles()``, ``sceneRect()`` (the page size) and a no-op scene."""

    def __init__(self):
        self.rectangles = []
        self.overlay_rects = []
        self._scene = _NullScene()
        self.image_size = (0, 0)

    def clear_rectangles(self):
        del self.rectangles[:]

    def sceneRect(self):
        width, height = self.image_size
        return _Rect(0, 0, width, height)

    def viewport(self):
        return self

    def update(self, *args):
        return None

    def repaint(self):
        return None


def _image_size(path):
    """Page size in pixels (untrusted file: safe_image.open_image), (0, 0) when unreadable."""
    try:
        with open_image(path) as image:
            return image.size
    except Exception:
        return (0, 0)


class EditorPreview:
    """The MangaImagePreviewWidget surface the moved code reads (no widgets): the current page,
    its translated output, the page list (Translate All), the display mode and the viewer."""

    def __init__(self, session):
        self._session = session
        self.viewer = EditorViewer()
        self.current_image_path = None
        self.current_translated_path = None
        self.image_paths = []
        self.source_display_mode = 'translated'
        self.cleaned_images_enabled = True

    def load_image(self, image_path, preserve_rectangles=False, preserve_text_overlays=False):
        """MangaImagePreviewWidget.load_image without pixels: switch page; restore its boxes from the
        page state unless the caller preserves them (the desktop's load + loaded-success handlers)."""
        self.current_image_path = image_path
        self.viewer.image_size = _image_size(image_path)
        if not preserve_rectangles:
            self.viewer.clear_rectangles()
            if not preserve_text_overlays:
                self._session._restore_page_state(image_path)
        return True

    @property
    def manga_integration(self):
        """The session (the desktop widget's ``manga_integration``, read by the Clear Boxes helpers)."""
        return self._session

    @property
    def main_gui(self):
        return getattr(self._session, 'main_gui', None)

    def _persist_rectangles_state(self):
        """manga_image_preview's _persist_rectangles_state: write ``viewer_rectangles`` only."""
        try:
            image_path = self.current_image_path
            manager = getattr(self._session, 'image_state_manager', None)
            if not image_path or manager is None:
                return
            prev = manager.get_state(image_path) or {}
            prev['viewer_rectangles'] = [box.state_entry() for box in self.viewer.rectangles]
            manager.set_state(image_path, prev, save=True)
        except Exception:
            pass


class _QueuedSignal:
    """Qt signal stand-in: ``emit`` queues the slot for the session's update pump (the GUI thread
    of the desktop)."""

    def __init__(self, update_queue, slot):
        self._queue = update_queue
        self._slot = slot

    def emit(self, *args):
        self._queue.put(('call_method', self._slot, args))

    def connect(self, slot):
        self._slot = slot


# Worker threads the moved click handlers start (``Thread-N (<target>)`` names, Python >= 3.10).
_DETECT_WORKERS = ('_run_detect_background',)
_CLEAN_WORKERS = ('_run_clean_background',)
_RECOGNIZE_WORKERS = ('_run_recognize_background',)
_TRANSLATE_WORKERS = ('_run_translate_background', '_run_full_translate_pipeline', 'run_inpainting_concurrent')
_TRANSLATE_ALL_WORKERS = ('_run_translate_all_background', 'run_inpainting_concurrent')
_BOX_OCR_WORKERS = ('ocr_background',)
_BOX_CLEAN_WORKERS = ('run_single_rect_clean',)
_BOX_TRANSLATE_WORKERS = ('_translate_this_text_background',)


class MangaEditorSession:
    """Headless host of the editor functions: the mobile counterpart of the manga tab's editor half.

    It carries what the moved functions read from the desktop's MangaTranslationTab:
    ``main_gui`` (a HeadlessOwner on mobile), ``update_queue``, ``stop_flag``,
    ``image_state_manager`` (no worker process on mobile), ``image_preview_widget`` (an
    EditorPreview with EditorBox shapes), the tab settings (``TAB_ATTRIBUTES``, from
    manga_env.HeadlessMangaState) and the OCR signals. The workflow methods run the desktop's click handlers and
    then pump ``update_queue`` on the calling thread (the desktop's GUI thread role) until the
    worker threads end, dispatching to the same result handlers the desktop's update loop calls.

    One step at a time (``_op_lock``); ``stop()`` may be called from another thread. The session
    is meant to live as long as the editor screen; jobs pass their HeadlessOwner as ``owner=``.
    """

    def __init__(self, main_gui=None, *, state_file=None, image_paths=(), log_callback=None,
                 event_callback=None, ocr_prompt=None, default_ocr_prompt=None, use_circle_shapes=False):
        self.main_gui = main_gui
        self.update_queue = Queue()
        self.stop_flag = threading.Event()
        self.is_running = False
        self._global_cancellation = False
        self._batch_mode_active = False
        self.dialog = None
        self.translator = None
        self._manga_translator = None
        self.ocr_manager = None
        self._shared_inpainter = None
        self._use_circle_shapes = bool(use_circle_shapes)
        self._current_image_path = None
        self._current_regions = []
        self._rendered_images_map = {}
        self.output_revision = 0
        self.logs = []
        self._log_callback = log_callback
        self._event_callback = event_callback
        self._explicit_ocr_prompt = ocr_prompt
        self._default_ocr_prompt = default_ocr_prompt
        self._tab = None
        self._op_lock = threading.RLock()
        self.image_state_manager = ImageStateManager(state_file or default_state_file())
        self.image_preview_widget = EditorPreview(self)
        self.set_pages(image_paths)
        self.ocr_result_signal = _QueuedSignal(self.update_queue, lambda *args: _process_ocr_result(self, *args))
        self.ocr_error_signal = _QueuedSignal(self.update_queue, lambda *args: _handle_ocr_error(self, *args))
        self.ocr_prompt = ocr_prompt or ''
        if main_gui is not None:
            self.apply_settings(main_gui)

    # ---------------------------------------------------------------- host surface
    def _log(self, message, level='info'):
        """MangaTranslationTab._log: the session keeps the lines and forwards them."""
        text = str(message)
        self.logs.append((level, text))
        if len(self.logs) > 2000:
            del self.logs[:-1000]
        callback = self._log_callback
        if callable(callback):
            try:
                callback(text, level)
            except Exception:
                pass

    def _emit(self, kind, **data):
        callback = self._event_callback
        if callable(callback):
            try:
                callback(kind, data)
            except Exception:
                pass

    def _default_manga_ocr_prompt(self):
        """The OCR system prompt default: the caller's, else manga_env's (desktop
        MangaTranslationTab._default_manga_ocr_prompt), else empty."""
        if self._default_ocr_prompt is not None:
            return self._default_ocr_prompt
        try:
            import manga_env
            provider = getattr(manga_env, 'default_manga_ocr_prompt', None)
            if callable(provider):
                return provider()
        except Exception:
            pass
        return ''

    def set_global_cancellation(self, cancelled):
        self._global_cancellation = bool(cancelled)

    def is_globally_cancelled(self):
        return bool(self._global_cancellation)

    def clear_text_overlays_for_image(self, image_path=None):
        """Qt overlays on the desktop; the mobile page shows the rendered output instead."""
        return None

    # ---------------------------------------------------------------- settings / pages
    def apply_settings(self, owner=None):
        """Load the manga tab's settings for ``owner`` (default ``main_gui``): a
        manga_env.HeadlessMangaState runs the tab's own settings load on the owner's config and
        the session takes ``TAB_ATTRIBUTES`` from it (call it with each job's owner)."""
        owner = owner if owner is not None else self.main_gui
        if owner is None:
            return None
        import manga_env
        tab = manga_env.HeadlessMangaState(owner, host=_TabLogHost(self), image_state_manager=self.image_state_manager)
        self._tab = tab
        for name in TAB_ATTRIBUTES:
            if hasattr(tab, name):
                setattr(self, name, getattr(tab, name))
        if self._explicit_ocr_prompt:
            self.ocr_prompt = self._explicit_ocr_prompt
        elif not getattr(self, 'ocr_prompt', ''):
            self.ocr_prompt = self._default_manga_ocr_prompt() or ''
        return tab

    def set_owner(self, owner):
        """Use ``owner`` (a HeadlessOwner built on the job thread) as ``main_gui``."""
        if owner is not None:
            self.main_gui = owner
            self.apply_settings(owner)

    # The tab methods the moved glossary helpers call (hasattr-guarded on the desktop tab).
    def _manga_glossary_workflow_enabled(self):
        tab = self._tab
        return bool(tab._manga_glossary_workflow_enabled()) if tab is not None else False

    def _get_loaded_manga_glossary_text(self):
        tab = self._tab
        return tab._get_loaded_manga_glossary_text() if tab is not None else ''

    def set_pages(self, image_paths):
        """The page list (the preview's thumbnails; Translate All runs over it)."""
        pages = [os.path.abspath(path) for path in (image_paths or [])]
        self.image_preview_widget.image_paths = pages
        self.selected_files = list(pages)

    @property
    def current_page(self):
        return self.image_preview_widget.current_image_path

    @property
    def boxes(self):
        return list(self.image_preview_widget.viewer.rectangles)

    def open_page(self, image_path):
        """Show a page: persist the page being left, then restore the new page's boxes and texts
        (the desktop's file-selection handler + MangaImagePreviewWidget.load_image)."""
        image_path = os.path.abspath(image_path)
        with self._op_lock:
            previous = self.image_preview_widget.current_image_path
            if previous and os.path.normcase(previous) != os.path.normcase(image_path):
                _persist_current_image_state(self)
                _clear_cross_image_state(self)
            self._current_image_path = image_path
            self.image_preview_widget.current_translated_path = None
            self.image_preview_widget.load_image(image_path)
            return self.page_snapshot()

    def _restore_page_state(self, image_path):
        """Boxes, texts and output paths of a page from its persisted state (the GUI-free steps of
        ImageRenderer._restore_image_state_overlays_only + show_recognized_overlays_for_image)."""
        if getattr(self, '_current_state_image_path', None) and self._current_state_image_path != image_path:
            _clear_cross_image_state(self)
        self._translation_data_image_path = image_path
        self._translating_image_path = image_path
        self._current_state_image_path = image_path
        _validate_and_clean_stale_state(self, image_path)
        state = self.image_state_manager.get_state(image_path) or {}
        viewer = self.image_preview_widget.viewer
        viewer.clear_rectangles()
        recognized_texts = state.get('recognized_texts') or []
        translated_texts = state.get('translated_texts') or []

        def _live(entries, idx):
            return idx < len(entries) and not (isinstance(entries[idx], dict) and entries[idx].get('deleted'))

        if state.get('viewer_rectangles'):
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                if not isinstance(rect_data, dict):
                    continue
                box = EditorBox(rect_data.get('x', 0), rect_data.get('y', 0), rect_data.get('width', 1),
                                rect_data.get('height', 1), shape=rect_data.get('shape', 'rect'),
                                polygon=rect_data.get('polygon'), region_index=idx)
                _copy_region_metadata_to_item(box, _saved_rect_metadata(state, idx, rect_data))
                box.is_recognized = _live(recognized_texts, idx) or _live(translated_texts, idx)
                viewer.rectangles.append(box)
        elif state.get('detection_regions'):
            self._current_regions = state['detection_regions']
            _draw_detection_boxes_on_preview(self)
            for idx, box in enumerate(viewer.rectangles):
                box.is_recognized = _live(recognized_texts, idx) or _live(translated_texts, idx)
        if state:
            ocr_count, _trans_count = _rehydrate_text_state_from_persisted(self, image_path)
            if ocr_count:
                _update_rectangles_with_recognition(self, self._recognized_texts)
        for key, iterations in (state.get('inpaint_iterations') or {}).items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(viewer.rectangles):
                viewer.rectangles[index].inpaint_iterations = iterations
        resolved_cleaned = _resolve_cleaned_image_for_render(self, image_path)
        if resolved_cleaned:
            self._cleaned_image_path = resolved_cleaned
        rendered_path = state.get('rendered_image_path')
        if rendered_path and os.path.exists(rendered_path):
            self.image_preview_widget.current_translated_path = rendered_path
            self._rendered_images_map[image_path] = rendered_path

    def page_snapshot(self, image_path=None):
        """What the editor screen draws for the current page."""
        current = self.image_preview_widget.current_image_path
        image_path = os.path.abspath(image_path) if image_path else current
        state = self.image_state_manager.get_state(image_path) or {} if image_path else {}
        recognition = getattr(self, '_recognition_data', {}) or {}
        translation = getattr(self, '_translation_data', {}) or {}
        boxes = []
        if image_path and current and os.path.normcase(image_path) == os.path.normcase(current):
            for idx, box in enumerate(self.image_preview_widget.viewer.rectangles):
                record = box.to_dict()
                record['index'] = idx
                key = box.region_index if box.region_index is not None else idx
                record['ocr_text'] = (recognition.get(key) or {}).get('text', '')
                record['translation'] = (translation.get(key) or {}).get('translation', '')
                boxes.append(record)
        return {
            'image_path': image_path,
            'boxes': boxes,
            'cleaned_path': state.get('cleaned_image_path'),
            'rendered_path': state.get('rendered_image_path') or self._rendered_images_map.get(image_path),
            'translated_path': self.image_preview_widget.current_translated_path if image_path == current else None,
            'revision': self.output_revision,
        }

    # ---------------------------------------------------------------- the update pump
    def _manga_editor_hook(self, name, *args, **kwargs):
        handler = getattr(self, '_hook' + name, None)
        if callable(handler):
            return handler(*args, **kwargs)
        return None

    def _hook_draw_detection_boxes_on_preview(self):
        """EditorBox shapes for ``_current_regions`` (ImageRenderer._draw_detection_boxes_on_preview)."""
        regions = getattr(self, '_current_regions', None)
        if not regions:
            return
        viewer = self.image_preview_widget.viewer
        if viewer.rectangles:
            return
        for i, region in enumerate(regions):
            bbox = region.get('bbox', []) if isinstance(region, dict) else []
            if len(bbox) >= 4:
                x, y, width, height = bbox[:4]
                box = EditorBox(x, y, width, height, shape='ellipse' if self._use_circle_shapes else 'rect')
                box.region_index = i
                _copy_region_metadata_to_item(box, region)
                viewer.rectangles.append(box)

    def _hook_add_context_menu_to_rectangle(self, rect_item, region_index):
        try:
            rect_item.region_index = region_index
        except Exception:
            pass

    def _hook_schedule_rendered_output_refresh(self, rendered_pil, output_path, switch_tab):
        self.image_preview_widget.current_translated_path = output_path
        self.output_revision += 1
        self._emit('rendered', output_path=output_path)

    def _hook_confirm_clean_excluded_rectangle(self, region_index):
        return True

    def _dispatch_update(self, update):
        """One update-queue message (the desktop's MangaTranslationTab._process_updates branches)."""
        kind = update[0]
        data = update[1] if len(update) > 1 else None
        try:
            if kind == 'detect_results':
                _process_detect_results(self, data)
            elif kind == 'recognize_results':
                _process_recognize_results(self, data)
            elif kind == 'translate_results':
                _process_translate_results(self, data)
            elif kind == 'single_clean_complete':
                self._on_single_clean_complete(data)
            elif kind == 'single_clean_error':
                self._log(f"❌ Rectangle {data['region_index']} cleaning failed: {data['error']}", "error")
            elif kind == 'translate_this_text_result':
                self._on_translate_this_text_result(data)
            elif kind == 'parallel_gui_update':
                _update_single_text_overlay(self, data['region_index'], data['trans_text'])
            elif kind == 'preview_update':
                self._on_preview_update(data)
            elif kind == 'load_preview_image':
                self._on_load_preview_image(data)
            elif kind == 'switch_to_translated_mode':
                self.image_preview_widget.source_display_mode = 'translated'
                self.image_preview_widget.cleaned_images_enabled = True
            elif kind == 'translate_all_progress':
                if not (self.stop_flag.is_set() or self._global_cancellation):
                    self._emit('progress', current=data['current'], total=data['total'])
            elif kind == 'sync_file_selection':
                self._emit('page', image_path=(data or {}).get('image_path'))
            elif kind == 'call_method':
                update[1](*update[2])
            elif kind == 'log':
                self._log(*update[1:])
            # button restores, overlays, pool tracker, parallel button state: no widgets on mobile
        except Exception as error:
            self._log(f"❌ Failed to process {kind}: {error}", "error")

    def _on_single_clean_complete(self, data):
        original_path = data['original_path']
        current = self.image_preview_widget.current_image_path
        if current and original_path and os.path.abspath(original_path) != os.path.abspath(current):
            return
        _update_image_preview_with_result(self, data['result_image'], original_path)
        self.output_revision += 1
        self._log(f"✅ Successfully cleaned rectangle {data['region_index']}", "success")

    def _on_translate_this_text_result(self, data):
        """The GUI-free steps of the desktop's translate_this_text_result branch: keep the text,
        persist the page's translated_texts entry, re-render (its Save & Update Overlay)."""
        region_index = int(data.get('region_index'))
        original_text = data.get('original_text', '')
        translation_result = data.get('translation_result', '')
        bbox = data.get('bbox') or [0, 0, 100, 100]
        if not hasattr(self, '_translation_data'):
            self._translation_data = {}
        self._translation_data[region_index] = {'original': original_text, 'translation': translation_result}
        current_image = self.image_preview_widget.current_image_path
        if current_image:
            state = self.image_state_manager.get_state(current_image) or {}
            tlist = state.get('translated_texts') or []
            if len(tlist) <= region_index:
                tlist = list(tlist) + [{} for _ in range(region_index + 1 - len(tlist))]
            tlist[region_index] = {
                'original': {'text': original_text, 'region_index': region_index},
                'translation': translation_result,
                'bbox': bbox or [0, 0, 100, 100],
            }
            state['translated_texts'] = tlist
            self.image_state_manager.set_state(current_image, state, save=True)
            if self._translation_data:
                _update_single_text_overlay(self, region_index, translation_result)
                self.image_preview_widget.source_display_mode = 'translated'

    def _on_preview_update(self, data):
        if not isinstance(data, dict):
            return
        translated_path = data.get('translated_path')
        source_path = data.get('source_path')
        current = self.image_preview_widget.current_image_path
        if translated_path and source_path and current and os.path.normcase(os.path.abspath(source_path)) == os.path.normcase(os.path.abspath(current)):
            self.image_preview_widget.current_translated_path = translated_path
        self.output_revision += 1
        self._emit('output', source_path=source_path, output_path=translated_path)

    def _on_load_preview_image(self, data):
        if isinstance(data, dict):
            image_path = data.get('path')
            preserve_rectangles = data.get('preserve_rectangles', False)
            preserve_overlays = data.get('preserve_overlays', False)
        else:
            image_path, preserve_rectangles, preserve_overlays = data, False, False
        if not image_path:
            return
        current = self.image_preview_widget.current_image_path
        if preserve_rectangles and current:
            # Translate All shows the page's cleaned image in the desktop source viewer (same boxes).
            # The mobile editor draws the cleaned / rendered output from the page state, so the page
            # stays open (no state entry keyed by the cleaned file).
            self.output_revision += 1
            self._emit('output', source_path=current, output_path=image_path)
            return
        if not self._batch_mode_active:
            if current and os.path.normcase(os.path.normpath(image_path)) != os.path.normcase(os.path.normpath(current)):
                return
        self.image_preview_widget.load_image(image_path, preserve_rectangles=preserve_rectangles,
                                             preserve_text_overlays=preserve_overlays)
        state = self.image_state_manager.get_state(image_path) or {}
        rendered = state.get('rendered_image_path') or self._rendered_images_map.get(image_path)
        if rendered and os.path.exists(rendered):
            self.image_preview_widget.current_translated_path = rendered
        self._current_image_path = image_path
        self._emit('page', image_path=image_path)

    def _workers_alive(self, before, names):
        suffixes = tuple(f"({name})" for name in names)
        return any(
            thread.is_alive() and thread not in before and thread.name.endswith(suffixes)
            for thread in threading.enumerate()
        )

    def pump_updates(self, before=None, names=()):
        """Dispatch queued updates until the named worker threads (started after ``before``) end
        and the queue is empty."""
        before = before if before is not None else set(threading.enumerate())
        while True:
            alive = self._workers_alive(before, names) if names else False
            try:
                update = self.update_queue.get(timeout=0.05 if alive else 0)
            except Empty:
                if not alive:
                    return
                continue
            self._dispatch_update(update)

    def _run_action(self, action, args=(), names=()):
        """Run a desktop click handler, then pump its worker's updates on this thread."""
        before = set(threading.enumerate())
        action(self, *args)
        self.pump_updates(before, names)

    def _begin(self, owner, image_path):
        if owner is not None:
            self.set_owner(owner)
        elif self.main_gui is not None:
            self.apply_settings(self.main_gui)
        if self.main_gui is None:
            raise ValueError("The manga editor needs an owner (main_gui) for this step")
        if image_path:
            image_path = os.path.abspath(image_path)
            current = self.image_preview_widget.current_image_path
            if not current or os.path.normcase(current) != os.path.normcase(image_path):
                self.open_page(image_path)
        if not self.image_preview_widget.current_image_path:
            raise ValueError("No manga page is open")
        self._rendering_in_progress = False
        return self.image_preview_widget.current_image_path

    def _finish(self):
        try:
            _persist_current_image_state(self)
        except Exception:
            pass
        try:
            self.image_state_manager.flush()
        except Exception:
            pass

    # ---------------------------------------------------------------- workflow steps
    def detect(self, image_path=None, *, owner=None):
        """Detect Text: the page's detection regions (also drawn as boxes)."""
        with self._op_lock:
            page = self._begin(owner, image_path)
            self._run_action(_on_detect_text_clicked, (), _DETECT_WORKERS)
            self._finish()
            state = self.image_state_manager.get_state(page) or {}
            return list(getattr(self, '_current_regions', None) or state.get('detection_regions') or [])

    def clean(self, image_path=None, *, owner=None):
        """Clean: inpaint the page's boxes (detecting first when there are none); the cleaned path."""
        with self._op_lock:
            page = self._begin(owner, image_path)
            self._run_action(_on_clean_image_clicked, (), _CLEAN_WORKERS)
            self._finish()
            return (self.image_state_manager.get_state(page) or {}).get('cleaned_image_path')

    def recognize(self, image_path=None, *, owner=None):
        """Recognize Text (OCR) on the boxes (detecting first when there are none)."""
        with self._op_lock:
            page = self._begin(owner, image_path)
            self._run_action(_on_recognize_text_clicked, (), _RECOGNIZE_WORKERS)
            self._finish()
            return list((self.image_state_manager.get_state(page) or {}).get('recognized_texts') or [])

    def translate(self, image_path=None, *, owner=None):
        """Translate: detect / recognize when needed, inpaint, translate and render the page."""
        with self._op_lock:
            page = self._begin(owner, image_path)
            self._run_action(_on_translate_text_clicked, (), _TRANSLATE_WORKERS)
            self._finish()
            state = self.image_state_manager.get_state(page) or {}
            return {
                'translated_texts': list(state.get('translated_texts') or []),
                'rendered_path': state.get('rendered_image_path') or self._rendered_images_map.get(page),
            }

    def translate_all(self, image_paths=None, *, owner=None):
        """Translate All over ``image_paths`` (default: the page list), page by page."""
        with self._op_lock:
            if image_paths is not None:
                self.set_pages(image_paths)
            pages = list(self.image_preview_widget.image_paths)
            if not pages:
                raise ValueError("No manga pages to translate")
            self._begin(owner, self.image_preview_widget.current_image_path or pages[0])
            self._run_action(_on_translate_all_clicked, (), _TRANSLATE_ALL_WORKERS)
            self._finish()
            return {page: dict(self.image_state_manager.get_state(page) or {}).get('rendered_image_path') for page in pages}

    def save_and_update_overlay(self, image_path=None, *, owner=None):
        """Save & Update Overlay: persist the boxes and re-render the page with its translations."""
        with self._op_lock:
            page = self._begin(owner, image_path)
            self.image_preview_widget._persist_rectangles_state()
            save_positions_and_rerender(self)
            self.pump_updates()
            self._finish()
            self.output_revision += 1
            return (self.image_state_manager.get_state(page) or {}).get('rendered_image_path')

    def render_page(self, image_path, *, owner=None):
        """Render a page from its persisted state only (imported sessions; no box reads)."""
        with self._op_lock:
            if owner is not None:
                self.set_owner(owner)
            output = render_persisted_translation_state(self, os.path.abspath(image_path), refresh_preview=False)
            self.image_state_manager.flush()
            if output:
                self.output_revision += 1
            return output

    # ---------------------------------------------------------------- per-box actions
    def _box(self, index):
        boxes = self.image_preview_widget.viewer.rectangles
        if not 0 <= int(index) < len(boxes):
            raise IndexError(f"No box {index} on this page")
        return boxes[int(index)]

    def add_box(self, x, y, width, height, *, shape='rect', polygon=None):
        """A new box (manga_image_preview's rectangle-created handler)."""
        with self._op_lock:
            box = EditorBox(x, y, width, height, shape=shape, polygon=polygon)
            boxes = self.image_preview_widget.viewer.rectangles
            boxes.append(box)
            idx = len(boxes) - 1
            box.region_index = idx
            box.is_recognized = False
            _add_context_menu_to_rectangle(self, box, idx)
            self.image_preview_widget._persist_rectangles_state()
            self.image_state_manager.flush()
            return box

    def update_box(self, index, x, y, width, height, *, owner=None, rerender=True):
        """Move / resize a box; a translated box is re-rendered at its new place (the desktop's
        rectangle-moved auto save: _get_translation_text_for_region + _update_single_text_overlay)."""
        with self._op_lock:
            if owner is not None:
                self.set_owner(owner)
            box = self._box(index)
            box.set_geometry(x, y, width, height)
            self.image_preview_widget._persist_rectangles_state()
            output = None
            if rerender:
                trans_text = _get_translation_text_for_region(self, int(index))
                if trans_text:
                    _update_single_text_overlay(self, int(index), trans_text)
                    self.pump_updates()
                    output = self.image_preview_widget.current_translated_path
                    self.output_revision += 1
            self.image_state_manager.flush()
            return output

    def delete_box(self, index):
        """Delete Selected (ImageRenderer._handle_delete_rectangle)."""
        with self._op_lock:
            _handle_delete_rectangle(self, int(index), self._box(index))
            self._finish()

    def clear_page(self, image_path=None):
        """Clear Boxes (manga_image_preview's ``_on_clear_boxes_clicked``): every box of the page, its
        detection / OCR / translation state and its translated output image (the cleaned image stays),
        then the page reloads from the cleaned or source image."""
        with self._op_lock:
            preview = self.image_preview_widget
            page = os.path.abspath(image_path) if image_path else preview.current_image_path
            if not page:
                return None
            if not preview.current_image_path or os.path.normcase(page) != os.path.normcase(preview.current_image_path):
                self.open_page(page)
            preview.viewer.clear_rectangles()
            _clear_detection_state_for_image(self, page)
            self.clear_text_overlays_for_image(page)
            for name in ('_translation_data', '_recognition_data'):
                data = getattr(self, name, None)
                if hasattr(data, 'clear'):
                    data.clear()
            try:
                _delete_translated_outputs(preview)
            except Exception as e:
                self._log(f"[CLEAR] Error deleting translated output: {e}", "warning")
            try:
                if self.image_state_manager:
                    _clear_saved_page_state(preview)
            except Exception:
                pass
            preview._persist_rectangles_state()
            self._current_regions = []
            self._rendered_images_map.pop(page, None)
            preview.current_translated_path = None
            self.image_state_manager.flush()
            self.output_revision += 1
            return self.page_snapshot(page)

    def set_box_free_text(self, index, free_text=None):
        """Mark as free text / bubble text (ImageRenderer._handle_toggle_free_text_region); None toggles."""
        with self._op_lock:
            box = self._box(index)
            if free_text is None or bool(free_text) != box.is_free_text:
                _handle_toggle_free_text_region(self, int(index), box)
            self.image_state_manager.flush()
            return box.is_free_text

    def set_box_excluded(self, index, excluded):
        """Exclude from Clean (session-only on the desktop too: never persisted)."""
        with self._op_lock:
            box = self._box(index)
            box.exclude_from_clean = bool(excluded)
            if box.exclude_from_clean:
                self._log(f"❌ Rectangle {index} excluded from inpainting", "info")
            else:
                self._log(f"✅ Rectangle {index} included in inpainting", "info")
            return box.exclude_from_clean

    def set_box_iterations(self, index, value):
        """Set Inpainting Iterations: -1 = auto, 0-50 = custom (the dialog's range)."""
        value = int(value)
        if value < -1 or value > 50:
            raise ValueError("Inpainting iterations must be between -1 and 50 (-1 = Auto)")
        with self._op_lock:
            _apply_inpaint_iterations(self, int(index), self._box(index), value)
            self.image_state_manager.flush()
            return self._box(index).inpaint_iterations

    def edit_box_text(self, index, *, ocr_text=None, translation=None):
        """Edit a box's OCR text and/or translation (the Edit OCR / Edit Translation dialogs' Save).

        A translation edit is persisted to the page state like the desktop dialog does; call
        ``save_and_update_overlay`` (or ``update_box``) to re-render. An OCR-only edit stays in the
        text maps on the desktop; the mobile session also writes it to ``recognized_texts`` so it
        survives the app being closed. Returns True when anything changed.
        """
        index = int(index)
        with self._op_lock:
            recognition = getattr(self, '_recognition_data', None) or {}
            current_ocr = (recognition.get(index) or {}).get('text', '')
            changed = False
            if translation is None:
                if ocr_text is not None and ocr_text != current_ocr:
                    _apply_ocr_text_edit(self, index, current_ocr, ocr_text)  # Edit OCR dialog
                    self._persist_recognized_text(index, ocr_text)
                    changed = True
            else:
                if not isinstance(getattr(self, '_translation_data', None), dict):
                    self._translation_data = {}
                entry = self._translation_data.setdefault(index, {'original': current_ocr, 'translation': ''})
                original = entry.get('original', '')
                old_translation = entry.get('translation', '')
                new_original = original if ocr_text is None else ocr_text
                # Edit Translation dialog: text maps, then the page's translated_texts entry
                if _apply_translation_text_edit(self, index, original, old_translation, new_original, translation):
                    _persist_translation_text_edit(self, index, new_original, translation)
                    if new_original != original:
                        self._persist_recognized_text(index, new_original)
                    changed = True
            self.image_state_manager.flush()
            return changed

    def _persist_recognized_text(self, index, text):
        page = self.image_preview_widget.current_image_path
        if not page:
            return
        state = self.image_state_manager.get_state(page) or {}
        rec_list = list(state.get('recognized_texts') or [])
        for entry in rec_list:
            if isinstance(entry, dict) and entry.get('region_index') == index and not entry.get('deleted'):
                entry['text'] = text
                break
        else:
            while len(rec_list) <= index:
                rec_list.append({'deleted': True})
            bbox = (getattr(self, '_recognition_data', {}).get(index) or {}).get('bbox') or self._box(index).bbox
            rec_list[index] = {'text': text, 'bbox': bbox, 'region_index': index}
        state['recognized_texts'] = rec_list
        self.image_state_manager.set_state(page, state, save=True)
        for entry in getattr(self, '_recognized_texts', None) or []:
            if isinstance(entry, dict) and entry.get('region_index') == index:
                entry['text'] = text

    def ocr_box(self, index, *, owner=None):
        """OCR This Text (ImageRenderer._handle_ocr_this_text): the box's recognized text or ''."""
        with self._op_lock:
            self._begin(owner, None)
            self._run_action(_handle_ocr_this_text, (int(index), self._box(index)), _BOX_OCR_WORKERS)
            self._finish()
            return (getattr(self, '_recognition_data', {}).get(int(index)) or {}).get('text', '')

    def translate_box(self, index, *, owner=None):
        """Translate This Text (ImageRenderer._handle_translate_this_text) + its re-render."""
        with self._op_lock:
            self._begin(owner, None)
            prompt, _target_language = _manual_translate_prompt(self)
            self._run_action(_handle_translate_this_text, (int(index), prompt), _BOX_TRANSLATE_WORKERS)
            self._finish()
            return (getattr(self, '_translation_data', {}).get(int(index)) or {}).get('translation', '')

    def clean_box(self, index, *, owner=None):
        """Clean This Rectangle (ImageRenderer._handle_clean_this_rectangle): the cleaned page path."""
        with self._op_lock:
            page = self._begin(owner, None)
            self._run_action(_handle_clean_this_rectangle, (int(index), self._box(index)), _BOX_CLEAN_WORKERS)
            self._finish()
            return (self.image_state_manager.get_state(page) or {}).get('cleaned_image_path')

    # ---------------------------------------------------------------- stop
    def stop(self, force=False):
        """Stop the running step: graceful (finish the in-flight API call) or force (hard cancel);
        the desktop preview's Stop button / double-click."""
        if force:
            _request_force_stop(self)
        else:
            _request_graceful_stop(self)

    # ---------------------------------------------------------------- OCR import / export
    def export_ocr(self, destination, image_paths=None, *, source_root=None):
        """Export the pages' OCR, translations and box mappings as a manga_ocr_io document."""
        with self._op_lock:
            if self.image_preview_widget.current_image_path:
                _persist_current_image_state(self)
            files = [os.path.abspath(path) for path in (image_paths or self.image_preview_widget.image_paths or [])]
            pages = []
            translated_region_count = 0
            for index, image_path in enumerate(files, start=1):
                state = dict(self.image_state_manager.get_state(image_path) or {})
                regions = manga_ocr_io.canonical_regions_from_editor_state(state)
                if not regions and not state.get('recognized_texts'):
                    continue
                translated_region_count += sum(1 for region in regions if str(region.get('translated_text') or '').strip())
                pages.append(manga_ocr_io.make_page(image_path, regions, index=index, source_root=source_root,
                                                    editor_state=state))
            if not pages:
                return {'path': None, 'pages': 0, 'translated_regions': 0}
            document = manga_ocr_io.create_document(pages, workflow="manual-editor", source_root=source_root)
            path = manga_ocr_io.write_document(destination, document)
            return {'path': path, 'pages': len(pages), 'translated_regions': translated_region_count}

    def import_ocr(self, path, image_paths=None, *, owner=None, render=True):
        """Import a manga_ocr_io document onto the pages (matched by path / relative path / name);
        pages with translations are rendered from the imported state when ``render``."""
        with self._op_lock:
            if owner is not None:
                self.set_owner(owner)
            document = manga_ocr_io.load_document(path)
            files = [os.path.abspath(p) for p in (image_paths or self.image_preview_widget.image_paths or [])]
            matches = manga_ocr_io.match_document_pages(document, files)
            translated_region_count = 0
            for image_path, page in matches.items():
                imported_state = manga_ocr_io.editor_state_from_page(page)
                existing_state = self.image_state_manager.get_state(image_path) or {}
                existing_state.update(imported_state)
                self.image_state_manager.set_state(image_path, existing_state, save=False)
                translated_region_count += sum(
                    1 for region in (page.get('regions') or [])
                    if isinstance(region, dict) and str(region.get('translated_text') or '').strip()
                )
            self.image_state_manager.flush()
            rendered = {}
            if render:
                for image_path, page in matches.items():
                    if any(isinstance(region, dict) and str(region.get('translated_text') or '').strip()
                           for region in (page.get('regions') or [])):
                        output = render_persisted_translation_state(self, image_path, refresh_preview=False)
                        if output:
                            rendered[image_path] = output
                self.image_state_manager.flush()
            current = self.image_preview_widget.current_image_path
            if current and current in matches:
                _clear_cross_image_state(self)
                self.image_preview_widget.load_image(current)
            if rendered:
                self.output_revision += 1
            return {'matched': len(matches), 'files': len(files), 'translated_regions': translated_region_count,
                    'rendered': rendered}

    # ---------------------------------------------------------------- lifecycle
    def close(self):
        """Persist the open page and write the state file."""
        with self._op_lock:
            if self.image_preview_widget.current_image_path:
                try:
                    _persist_current_image_state(self)
                except Exception:
                    pass
            self.image_state_manager.flush()
