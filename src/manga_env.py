"""Manga run environment, settings state and OCR session (GUI-free).

Shared GUI-free core (Glossarion mobile rewrite, milestone U8). ``MangaTranslationTab``
(manga_integration.py) inherits :class:`MangaEnvMixin` and :class:`MangaOcrSessionMixin`;
their methods were moved there verbatim (three of them are split verbatim out of
``_start_translation_heavy``: ``_reset_manga_graceful_stop_env``, ``_prepare_manga_run_env``,
``_apply_manga_batch_env``). Mobile drives the same code through
``manga_runner.HeadlessMangaRunner`` with a ``HeadlessOwner`` as ``main_gui``.

Contract functions (mobile, tests, host tools):

* :func:`build_manga_run_env` -> the env delta a desktop batch start writes (thread limits,
  OCR config / credentials, API client + multi-key / image / fallback / Vision key pools,
  custom-api OCR env from ``owner._get_environment_variables`` minus SYSTEM_PROMPT,
  ``OCR_SYSTEM_PROMPT``, ``MANGA_IMAGE_REQUEST_*``, ``BATCH_*``, ``GRACEFUL_STOP*``);
* :func:`prepare_manga_glossary_env` (+ restore) for the manga glossary workflow;
* :func:`apply_rendering_settings` (translator, owner) = ``_apply_rendering_settings``;
* :func:`build_ocr_config` (config) = ``_build_manga_worker_ocr_config``;
* :func:`font_preset_updates` (preset, config) -> the config writes of a font-size preset
  button (``_set_font_preset``, moved here verbatim);
* :func:`apply_manga_startup_thread_limits`, the default prompts.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup; importing
this module does not import manga_translator / cv2 (the run code is in manga_runner).
"""

import json
import os
import re
import shutil
import threading
import time
import traceback
from queue import Queue
from typing import Any, Dict, List, Optional

import manga_ocr_io
from glossary_paths import get_book_glossary_dir, migrate_legacy_named_files
from manga_files_core import FileListShim, ImageRenderer, MangaFilesMixin, MangaHooksMixin, _get_app_dir


def apply_manga_startup_thread_limits(main_gui):
    """``MangaTranslationTab.__init__``'s first block (moved verbatim): single-thread the
    numeric libraries unless ``manga_settings.advanced.parallel_processing`` is on."""
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



class MangaEnvMixin:
    """MangaTranslationTab's settings state, run-start env and glossary paths (moved verbatim, U8)."""

    def _init_manga_run_state(self):
        """``MangaTranslationTab.__init__``'s run-state block (moved verbatim)."""
        # Translation state
        self.translator = None
        self.is_running = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.translation_future = None
        self._translation_start_token = 0
        self._translation_startup_pending = False
        self._translation_start_cancel_requested = False
        # Shared executor from main GUI if available
        try:
            if hasattr(self.main_gui, 'executor') and self.main_gui.executor:
                self.executor = self.main_gui.executor
            else:
                self.executor = None
        except Exception:
            self.executor = None
        self.selected_files = []
        self._manga_file_sort = ('numeric', False)
        self.skipped_processing_files = set()
        self.manga_image_range_value = ""
        self._manga_processing_files = None
        self._ocr_io_lock = threading.RLock()
        self._imported_ocr_document = None
        self._imported_ocr_page_map = {}
        self._automatic_ocr_document = None
        self._automatic_ocr_export_path = None
        self.current_file_index = 0
        self.font_mapping = {}  # Initialize font mapping dictionary

        # Shared inpainting model instance for reuse
        self._shared_inpainter = None
        self._shared_inpainter_method = None
        self._shared_inpainter_path = None
        
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
        # Do NOT preload big local models by default to avoid startup crashes
        self.preload_local_models_on_open = bool(adv_cfg.get('preload_local_models_on_open', False))
        
        # Initialize experimental features env var from config
        experimental_enabled = self.main_gui.config.get('experimental_translate_all', False)
        os.environ['EXPERIMENTAL_TRANSLATE_ALL'] = '1' if experimental_enabled else '0'
        
        # Queue for thread-safe GUI updates
        self.update_queue = Queue()

    def _init_manga_prompt_state(self):
        """``MangaTranslationTab.__init__``'s prompt block after ``_load_rendering_settings`` (moved verbatim)."""
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
        self.ocr_prompt = self.main_gui.config.get('manga_ocr_prompt', self._default_manga_ocr_prompt())

    def _reset_manga_graceful_stop_env(self):
        """Run start: clear the graceful-stop env and flag (split verbatim from
        ``_start_translation_heavy``, U8)."""
        os.environ['GRACEFUL_STOP'] = '0'
        os.environ['GRACEFUL_STOP_COMPLETED'] = '0'
        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui.graceful_stop_active = False

    def _prepare_manga_run_env(self, start_token=None):
        """Run start: thread limits, OCR config + credentials, API client and key pools,
        Vision keys, custom-api OCR env (split verbatim from ``_start_translation_heavy``, U8).

        Returns ``(ocr_config, api_key, model, needs_new_client)``, or None when the start
        was aborted (already logged, heartbeat stopped and UI reset, exactly as before).
        """
        import os
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
        
        # Log AI Bubble Detection settings for debugging
        try:
            manga_settings = self.main_gui.config.get('manga_settings', {})
            ocr_settings = manga_settings.get('ocr', {})
            bubble_enabled = ocr_settings.get('bubble_detection_enabled', True)
            detect_empty = ocr_settings.get('detect_empty_bubbles', False)
            use_rtdetr_for_ocr = ocr_settings.get('use_rtdetr_for_ocr_regions', True)
            detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
            model_url = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path', '')
            
            self._log(f"🤖 AI Bubble Detection: {'Enabled' if bubble_enabled else 'Disabled'}", "info")
            if bubble_enabled:
                self._log(f"  • Detector Type: {detector_type}", "info")
                self._log(f"  • Model: {model_url if model_url else 'Default'}", "info")
                self._log(f"  • Use RT-DETR as Guide: {'Yes' if use_rtdetr_for_ocr else 'No'}", "info")
                self._log(f"  • Detect Empty Bubbles: {'Yes' if detect_empty else 'No'}", "info")
        except Exception as e:
            self._log(f"⚠️ Warning: Could not log bubble detection settings: {e}", "debug")
        
        # Early feedback
        self._log("⏳ Preparing configuration...", "info")
        
        # Reload OCR prompt from config (in case it was edited in the dialog)
        print(f"[OCR_PROMPT_LOAD] Keys in config: {list(self.main_gui.config.keys())[:20]}...")  # Debug
        print(f"[OCR_PROMPT_LOAD] Checking for 'manga_ocr_prompt' in config...")
        if 'manga_ocr_prompt' in self.main_gui.config:
            self.ocr_prompt = self.main_gui.config['manga_ocr_prompt']
            self._log(f"✅ Loaded OCR prompt from config ({len(self.ocr_prompt)} chars)", "info")
            self._log(f"OCR Prompt preview: {self.ocr_prompt[:100]}...", "debug")
            print(f"[OCR_PROMPT_LOAD] Successfully loaded OCR prompt: {len(self.ocr_prompt)} chars")
        else:
            self._log("⚠️ manga_ocr_prompt not found in config, using default", "warning")
            print(f"[OCR_PROMPT_LOAD] manga_ocr_prompt key NOT FOUND in config!")
            print(f"[OCR_PROMPT_LOAD] Using instance ocr_prompt: {len(getattr(self, 'ocr_prompt', '')) if hasattr(self, 'ocr_prompt') else 0} chars")
        
        # Build OCR configuration
        ocr_config = {'provider': self.ocr_provider_value}

        if ocr_config['provider'] == 'Qwen2-VL':
            qwen_provider = self._ensure_ocr_manager().get_provider('Qwen2-VL')
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
                self._reset_ui_state(start_token)
                return None
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
                self._reset_ui_state(start_token)
                return None
            
            # Save Azure settings
            self.main_gui.config['azure_vision_key'] = azure_key
            self.main_gui.config['azure_vision_endpoint'] = azure_endpoint
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            
            ocr_config['azure_key'] = azure_key
            ocr_config['azure_endpoint'] = azure_endpoint
        
        elif ocr_config['provider'] == 'azure-document-intelligence':
            # Azure Document Intelligence uses same credentials as azure
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
                self._reset_ui_state(start_token)
                return None
            
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
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not api_key:
            try:
                from unified_api_client import UnifiedClient as _UC
                if not _UC._model_needs_api_key(model):
                    api_key = 'own-auth'  # placeholder — actual auth handled by provider/local endpoint
            except Exception:
                pass
        if not api_key and ocr_config.get('provider') == 'custom-api':
            api_key = 'dummy-key-for-custom-api'
        
        if not api_key:
            self._log("❌ API key not found. Please configure your API key in the main settings.", "error")
            self._stop_startup_heartbeat()
            self._reset_ui_state(start_token)
            return None
        
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
                    # Validate multi-key configuration before applying
                    valid_keys = 0
                    for key_data in mk_list:
                        if isinstance(key_data, dict) and key_data.get('api_key') and key_data.get('model'):
                            valid_keys += 1
                    
                    if valid_keys == 0:
                        self._log("❌ Multi-key mode is enabled but no valid keys are configured!", "error")
                        self._log("   Each key must have both 'api_key' and 'model' fields.", "error")
                        self._log("   Please check your multi-key configuration in Settings.", "error")
                        self._stop_startup_heartbeat()
                        self._reset_ui_state(start_token)
                        return None
                    
                    os.environ['USE_MULTI_API_KEYS'] = '1'
                    os.environ['USE_MULTI_KEYS'] = '1'  # backward-compat for retry paths
                    os.environ['FORCE_KEY_ROTATION'] = '1' if self.main_gui.config.get('force_key_rotation', True) else '0'
                    os.environ['ROTATION_FREQUENCY'] = str(self.main_gui.config.get('rotation_frequency', 1))

                    # Avoid Windows env var length limit by keeping keys in memory
                    try:
                        from unified_api_client import UnifiedClient
                        UnifiedClient.set_in_memory_multi_keys(
                            mk_list,
                            force_rotation=self.main_gui.config.get('force_key_rotation', True),
                            rotation_frequency=self.main_gui.config.get('rotation_frequency', 1),
                        )
                    except Exception:
                        pass

                    self._log(f"🔑 Multi-key mode ENABLED for manga translator ({valid_keys} valid keys)", "info")
                else:
                    # Explicitly disable if not configured
                    os.environ['USE_MULTI_API_KEYS'] = '0'
                    os.environ['USE_MULTI_KEYS'] = '0'

                # Dedicated Image gen/edit pool for manga custom-image-edit and image output requests.
                inpainter_keys = self.main_gui.config.get('inpainter_keys', []) or []
                use_inpainter_keys = bool(self.main_gui.config.get('use_inpainter_keys', False))
                os.environ['USE_INPAINTER_KEYS'] = '1' if use_inpainter_keys else '0'
                os.environ['INPAINTER_API_KEYS'] = json.dumps(inpainter_keys)
                try:
                    from unified_api_client import UnifiedClient
                    if use_inpainter_keys and inpainter_keys:
                        UnifiedClient.set_in_memory_inpainter_keys(
                            inpainter_keys,
                            force_rotation=self.main_gui.config.get('force_key_rotation', True),
                            rotation_frequency=self.main_gui.config.get('rotation_frequency', 1),
                        )
                        self._log(f"Image gen/edit key pool ENABLED ({len(inpainter_keys)} keys)", "info")
                    else:
                        UnifiedClient.clear_in_memory_inpainter_keys()
                except Exception:
                    pass

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
                self._reset_ui_state(start_token)
                return None
        
        # Vision keys are used by custom-api manga OCR (context=manga_ocr).
        # Sync them on every manga run, even when the main UnifiedClient is
        # reused, so OCR region threads don't silently fall back to the main
        # GUI key/model.
        try:
            import os as _os
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
            _os.environ['USE_VISION_KEYS'] = '1' if use_vision_keys else '0'
            _os.environ['USE_QA_SCAN_KEYS'] = _os.environ['USE_VISION_KEYS']
            _os.environ['VISION_API_KEYS'] = _json.dumps(vision_keys)
            _os.environ['QA_SCAN_API_KEYS'] = _os.environ['VISION_API_KEYS']

            if use_vision_keys and vision_keys:
                _UnifiedClient.set_in_memory_vision_keys(
                    vision_keys,
                    force_rotation=self.main_gui.config.get('force_key_rotation', True),
                    rotation_frequency=self.main_gui.config.get('rotation_frequency', 1),
                )
                self._log(f"Vision key pool ENABLED for manga OCR ({len(vision_keys)} keys)", "info")
            else:
                _UnifiedClient.clear_in_memory_vision_keys()
        except Exception as vision_err:
            self._log(f"⚠️ Failed to sync Vision keys for manga OCR: {vision_err}", "warning")

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
            import large_env
            for key, value in env_vars.items():
                if key == 'SYSTEM_PROMPT':
                    # DON'T SET THE TRANSLATION SYSTEM PROMPT FOR OCR
                    continue
                large_env.set_env(key, str(value))
            self._sync_manga_image_request_quality_env()
            
            # Use custom OCR prompt from GUI if available, otherwise use default
            if hasattr(self, 'ocr_prompt') and self.ocr_prompt:
                os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
                self._log(f"✅ Using custom OCR prompt from GUI ({len(self.ocr_prompt)} chars)", "info")
                self._log(f"OCR Prompt being set: {self.ocr_prompt[:150]}...", "debug")
            else:
                # Fallback to default OCR prompt
                os.environ['OCR_SYSTEM_PROMPT'] = self._default_manga_ocr_prompt()
                self._log("✅ Using default OCR prompt", "info")
            
            self._log("✅ Set environment variables for custom-api OCR (excluded SYSTEM_PROMPT)")
            
            # Respect user settings: only set default detector values when bubble detection is OFF.
            try:
                ms = self.main_gui.config.setdefault('manga_settings', {})
                ocr_set = ms.setdefault('ocr', {})
                changed = False
                bubble_enabled = bool(ocr_set.get('bubble_detection_enabled', True))
                
                if not bubble_enabled:
                    # User has bubble detection OFF -> set non-intrusive defaults only
                    if 'detector_type' not in ocr_set:
                        ocr_set['detector_type'] = 'rtdetr_onnx'
                        changed = True
                    if 'rtdetr_onnx_variant' not in ocr_set:
                        ocr_set['rtdetr_onnx_variant'] = 'detector.onnx'
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
        return ocr_config, api_key, model, needs_new_client

    def _apply_manga_batch_env(self):
        """BATCH_* env before a new translator is created (split verbatim from
        ``_start_translation_heavy``, U8)."""
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
            
            # Determine batching mode & group size with config/GUI fallbacks
            def _get_gui_val(var, default):
                try:
                    if hasattr(var, 'get'):
                        return var.get()
                    return var
                except Exception:
                    return default
            mode_val = 'direct'
            if hasattr(self.main_gui, 'batch_mode_var'):
                mode_val = str(_get_gui_val(self.main_gui.batch_mode_var, 'direct')).strip().lower() or 'direct'
            else:
                try:
                    mode_val = str(self.main_gui.config.get('batching_mode', 'direct')).strip().lower()
                except Exception:
                    pass
            group_val = 3
            if hasattr(self.main_gui, 'batch_group_size_var'):
                try:
                    group_val = int(_get_gui_val(self.main_gui.batch_group_size_var, 3) or 3)
                except Exception:
                    group_val = 3
            else:
                try:
                    group_val = int(self.main_gui.config.get('batch_group_size', 3) or 3)
                except Exception:
                    group_val = 3

            # Set environment variables for the translator to pick up
            if batch_translation_enabled:
                os.environ['BATCH_TRANSLATION'] = '1'
                os.environ['BATCH_SIZE'] = str(max(1, batch_size_value))
                os.environ['BATCHING_MODE'] = mode_val
                os.environ['BATCH_GROUP_SIZE'] = str(max(1, group_val))
                # Legacy compatibility
                os.environ['CONSERVATIVE_BATCHING'] = '1' if mode_val == 'conservative' else '0'
                self._log(f"📦 Batch Translation ENABLED: {batch_size_value} concurrent API calls (mode={mode_val}, group={group_val})", "info")
            else:
                os.environ['BATCH_TRANSLATION'] = '0'
                os.environ['BATCH_SIZE'] = '1'
                os.environ['BATCHING_MODE'] = mode_val
                os.environ['BATCH_GROUP_SIZE'] = str(max(1, group_val))
                os.environ['CONSERVATIVE_BATCHING'] = '1' if mode_val == 'conservative' else '0'
                self._log("📦 Batch Translation DISABLED: Sequential API calls", "info")
        except Exception as e:
            self._log(f"⚠️ Warning: Could not set batch settings: {e}", "warning")
            os.environ['BATCH_TRANSLATION'] = '0'
            os.environ['BATCH_SIZE'] = '1'

    def _sync_manga_image_request_quality_env(self):
        """Mirror manga image request quality settings into process env for request builders."""
        try:
            ms = self.main_gui.config.setdefault('manga_settings', {})
            comp = ms.setdefault('compression', {})
            enabled = bool(comp.get('enabled', False))
            fmt = str(comp.get('format', 'jpeg') or 'jpeg').strip().lower()
            if fmt not in ('jpeg', 'jpg', 'png', 'webp'):
                fmt = 'jpeg'
            os.environ['MANGA_IMAGE_REQUEST_QUALITY_ENABLED'] = '1' if enabled else '0'
            os.environ['MANGA_IMAGE_REQUEST_FORMAT'] = fmt
            os.environ['MANGA_IMAGE_REQUEST_WEBP_QUALITY'] = str(int(comp.get('webp_quality', 85)))
            os.environ['MANGA_IMAGE_REQUEST_JPEG_QUALITY'] = str(int(comp.get('jpeg_quality', 85)))
            os.environ['MANGA_IMAGE_REQUEST_PNG_COMPRESSION'] = str(int(comp.get('png_compress_level', 6)))

            # Keep this setting manga-scoped. The generic IMAGE_COMPRESSION_*
            # variables belong to the main translator GUI and are used by
            # non-manga vision requests.
        except Exception as e:
            try:
                self._log(f"⚠️ Failed to sync manga image request quality env: {e}", "warning")
            except Exception:
                pass

    def _set_custom_image_edit_env(self):
        try:
            enabled = bool(getattr(self, 'use_custom_image_edit_endpoint_value', False))
            url = str(getattr(self, 'custom_image_edit_endpoint_value', '') or '').strip()
            os.environ['USE_CUSTOM_IMAGE_EDIT_ENDPOINT'] = '1' if enabled else '0'
            os.environ['CUSTOM_IMAGE_EDIT_BASE_URL'] = url if enabled else ''
            os.environ['OPENAI_IMAGE_EDIT_BASE_URL'] = url if enabled else ''
            os.environ['CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT'] = str(
                int(getattr(self, 'custom_image_edit_full_page_output_value', 10))
            )
        except Exception:
            pass

    def _default_custom_image_edit_system_prompt(self):
        return (
            "Remove the written text from this image. Redraw only the areas it covered to match their immediate surroundings. "
            "Keep speech-bubble outlines, text-box borders, panel frames, other artwork, and image dimensions unchanged. "
            "Do not add text. Return only the edited image."
        )

    def _default_manga_ocr_prompt(self):
        return (
            "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
            "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself.\n\n"
            "ABSOLUTE RULES:\n"
            "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\n"
            "2. NEVER TRANSLATE OR MODIFY\n"
            "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\n"
            "4. NEVER SAY \"I can't\" or \"I cannot\" or \"no text\" or \"blank image\"\n"
            "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\n"
            "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\n"
            "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\n"
            "8. IF THERE ARE NO VISIBLE CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]\n\n"
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
            "If image has no visible text or symbols → Output: [AI RESPONSE UNAVAILABLE]"
        )

    @staticmethod
    def _migrate_legacy_manga_ocr_prompt(prompt):
        """Update only saved prompts with the old empty-response OCR rule."""
        legacy_rule = "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)"
        if not isinstance(prompt, str) or legacy_rule not in prompt:
            return prompt

        prompt = prompt.replace(
            legacy_rule,
            "8. IF THERE ARE NO VISIBLE CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]",
        ).replace(
            "If image is truly blank → Output: [empty/no response]",
            "If image has no visible text or symbols → Output: [AI RESPONSE UNAVAILABLE]",
        )
        detector_note = "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself."
        if detector_note not in prompt:
            intro = "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
            prompt = prompt.replace(intro, intro + detector_note + "\n\n", 1) if intro in prompt else detector_note + "\n\n" + prompt
        return prompt

    def _default_manga_glossary_prompt(self) -> str:
        """Default prompt for manga glossary generation, based on the EPUB glossary extractor."""
        try:
            from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT
            base_prompt = DEFAULT_GLOSSARY_PROMPT.replace(
                "You are a novel glossary extraction assistant.",
                "You are a manga/manhwa/manhua glossary extraction assistant."
            )
        except Exception:
            base_prompt = (
                "You are a manga/manhwa/manhua glossary extraction assistant.\n\n"
                "Return ONLY CSV rows with columns: type,raw_name,translated_name,gender\n"
                "Extract character names, terms, places, organizations, abilities, items, and titles.\n"
                "Do not extract dialogue lines or full sentences."
            )

        manga_rules = (
            "\n\nMANGA OCR SOURCE RULES:\n"
            "- You will receive OCR text from every selected manga page in one request.\n"
            "- Treat all pages as one continuous chapter/scene for name and term consistency.\n"
            "- OCR can contain short bubbles, sound effects, broken line order, and repeated fragments.\n"
            "- Prefer recurring proper nouns and story-specific terms over ordinary dialogue words.\n"
            "- Keep raw_name exactly in the source script when possible."
        )
        return f"{base_prompt}{manga_rules}"

    def _get_compress_glossary_prompt_value(self) -> bool:
        """Read the shared glossary compression toggle from live UI/config."""
        value = True
        try:
            value = self.main_gui.config.get('compress_glossary_prompt', True)
        except Exception:
            value = True

        try:
            live_var = getattr(self.main_gui, 'compress_glossary_prompt_var', None)
            if live_var is not None:
                value = live_var.get() if hasattr(live_var, 'get') else live_var
        except Exception:
            pass

        try:
            glossary_checkbox = getattr(self.main_gui, 'compress_glossary_checkbox', None)
            if glossary_checkbox is not None and hasattr(glossary_checkbox, 'isChecked'):
                value = glossary_checkbox.isChecked()
        except Exception:
            pass

        return bool(value)

    def _set_checkbox_checked_safely(self, checkbox, checked: bool, *, block_signals: bool = True) -> None:
        if checkbox is None or not hasattr(checkbox, 'setChecked'):
            return
        try:
            if hasattr(checkbox, 'isChecked') and bool(checkbox.isChecked()) == bool(checked):
                self._refresh_synced_checkbox_style(checkbox)
                return
        except Exception:
            pass
        previous = None
        try:
            if block_signals and hasattr(checkbox, 'blockSignals'):
                previous = checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
        finally:
            self._refresh_synced_checkbox_style(checkbox)
            try:
                if previous is not None and hasattr(checkbox, 'blockSignals'):
                    checkbox.blockSignals(previous)
            except Exception:
                pass

    def _refresh_synced_checkbox_style(self, checkbox) -> None:
        """Refresh custom checkbox visuals after blocked programmatic updates."""
        try:
            if hasattr(checkbox, 'style'):
                checkbox.style().unpolish(checkbox)
                checkbox.style().polish(checkbox)
            update_checkmark = getattr(checkbox, '_update_checkmark', None)
            if callable(update_checkmark):
                update_checkmark()
            checkmark = getattr(checkbox, '_checkmark_label', None)
            if bool(getattr(checkbox, 'isChecked', lambda: False)()) and checkmark is not None:
                if hasattr(checkmark, 'show'):
                    checkmark.show()
                if hasattr(checkmark, 'raise_'):
                    checkmark.raise_()
                if hasattr(checkmark, 'update'):
                    checkmark.update()
            if hasattr(checkbox, 'update'):
                checkbox.update()
        except Exception:
            pass

    def _sync_compress_glossary_prompt_setting(self, enabled: bool) -> None:
        """Keep manga and Glossary Manager compression toggles on the same setting."""
        enabled = bool(enabled)
        self.compress_glossary_prompt_value = enabled
        try:
            self.main_gui.config['compress_glossary_prompt'] = enabled
        except Exception:
            pass
        try:
            live_var = getattr(self.main_gui, 'compress_glossary_prompt_var', None)
            if live_var is not None and hasattr(live_var, 'set'):
                live_var.set(enabled)
            else:
                setattr(self.main_gui, 'compress_glossary_prompt_var', enabled)
        except Exception:
            pass
        try:
            os.environ['COMPRESS_GLOSSARY_PROMPT'] = '1' if enabled else '0'
        except Exception:
            pass

        self._set_checkbox_checked_safely(
            getattr(self, 'manga_compress_glossary_checkbox', None),
            enabled,
            block_signals=True,
        )
        self._set_checkbox_checked_safely(
            getattr(self.main_gui, 'compress_glossary_checkbox', None),
            enabled,
            block_signals=False,
        )

    def _manga_glossary_debug_ocr_enabled(self) -> bool:
        """Return True when manga glossary generation should save OCR/debug artifacts."""
        try:
            if hasattr(self, 'manga_glossary_debug_ocr_checkbox'):
                return bool(self.manga_glossary_debug_ocr_checkbox.isChecked())
        except Exception:
            pass
        return bool(getattr(self, 'manga_glossary_debug_ocr_text_value', self.main_gui.config.get('manga_glossary_debug_ocr_text', False)))

    def _manga_glossary_workflow_enabled(self) -> bool:
        """Return True when the new OCR -> glossary -> translation workflow should run."""
        if bool(getattr(self, '_manga_glossary_only_run', False)):
            return True
        try:
            if hasattr(self, 'manga_glossary_checkbox'):
                return bool(self.manga_glossary_checkbox.isChecked())
        except Exception:
            pass
        return bool(getattr(self, 'manga_glossary_enabled_value', self.main_gui.config.get('manga_glossary_enabled', False)))

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
        _cleared_invalid_inpaint_paths = False
        for model_type in  ['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'custom-image-edit', 'mat', 'ollama', 'sd_local']:
            path = inpaint_settings.get(f'{model_type}_model_path', '')
            try:
                if isinstance(path, str) and path.lower().endswith('.json'):
                    path = ''
                    inpaint_settings[f'{model_type}_model_path'] = ''
                    # Also clear any top-level mirror if present
                    if hasattr(self, 'main_gui') and getattr(self, 'main_gui', None) and hasattr(self.main_gui, 'config'):
                        self.main_gui.config[f'manga_{model_type}_model_path'] = ''
                    _cleared_invalid_inpaint_paths = True
            except Exception:
                pass
            if model_type == self.local_model_type_value:
                self.local_model_path_value = path
        if _cleared_invalid_inpaint_paths and hasattr(self, 'main_gui') and hasattr(self.main_gui, 'save_config'):
            try:
                self.main_gui.save_config(show_message=False)
            except Exception:
                pass
        
        # Initialize with defaults (plain Python values, no Tkinter variables)
        self.bg_opacity_value = config.get('manga_bg_opacity', 0)
        self.free_text_only_bg_opacity_value = config.get('manga_free_text_only_bg_opacity', False)
        self.bg_style_value = config.get('manga_bg_style', 'circle')
        self.bg_reduction_value = config.get('manga_bg_reduction', 1.0)
        self.font_size_value = config.get('manga_font_size', 0)
        
        self.selected_font_path = config.get('manga_font_path', None)
        self.skip_inpainting_value = config.get('manga_skip_inpainting', False)
        self.inpaint_quality_value = config.get('manga_inpaint_quality', 'high')
        self.inpaint_dilation_value = config.get('manga_inpaint_dilation', 15)
        self.inpaint_passes_value = config.get('manga_inpaint_passes', 2)
        self.disable_inpaint_performance_mode_value = bool(config.get('manga_disable_inpaint_performance_mode', False))
        
        self.font_size_mode_value = config.get('manga_font_size_mode', 'fixed')
        self.font_size_multiplier_value = config.get('manga_font_size_multiplier', 1.0)
        
        # Auto fit style for auto mode
        try:
            rend_cfg = (config.get('manga_settings', {}) or {}).get('rendering', {})
        except Exception:
            rend_cfg = {}
        self.auto_fit_style_value = rend_cfg.get('auto_fit_style', 'compact')
        
        # Auto minimum font size (from rendering or font_sizing)
        try:
            font_cfg = (config.get('manga_settings', {}) or {}).get('font_sizing', {})
        except Exception:
            font_cfg = {}
        auto_min_default = rend_cfg.get('auto_min_size', font_cfg.get('min_size', 8))
        self.auto_min_size_value = int(auto_min_default)
        
        self.force_caps_lock_value = config.get('manga_force_caps_lock', True)
        self.constrain_to_bubble_value = config.get('manga_constrain_to_bubble', True)
        
        # Advanced font sizing (from manga_settings.font_sizing)
        font_settings = (config.get('manga_settings', {}) or {}).get('font_sizing', {})
        self.font_algorithm_value = str(font_settings.get('algorithm', 'smart'))
        self.prefer_larger_value = bool(font_settings.get('prefer_larger', True))
        self.bubble_size_factor_value = bool(font_settings.get('bubble_size_factor', True))
        self.line_spacing_value = float(font_settings.get('line_spacing', 1.3))
        
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
        
        # Safe area controls
        self.safe_area_enabled_value = bool(config.get('manga_safe_area_enabled', False))
        try:
            self.safe_area_scale_value = float(config.get('manga_safe_area_scale', 1.0))
        except Exception:
            self.safe_area_scale_value = 1.0
        
        # Font color settings
        manga_text_color = config.get('manga_text_color', [102, 0, 0])
        self.text_color_r_value = manga_text_color[0]
        self.text_color_g_value = manga_text_color[1]
        self.text_color_b_value = manga_text_color[2]
        
        # Shadow settings
        self.shadow_enabled_value = config.get('manga_shadow_enabled', True)
        
        manga_shadow_color = config.get('manga_shadow_color', [255, 255, 255])
        self.shadow_color_r_value = manga_shadow_color[0]
        self.shadow_color_g_value = manga_shadow_color[1]
        self.shadow_color_b_value = manga_shadow_color[2]
        
        self.shadow_offset_x_value = config.get('manga_shadow_offset_x', 2)
        self.shadow_offset_y_value = config.get('manga_shadow_offset_y', 2)
        self.shadow_blur_value = config.get('manga_shadow_blur', 0)
        
        # Initialize font_style with saved value or default
        self.font_style_value = config.get('manga_font_style', 'Default')
        
        # Full page context settings
        self.full_page_context_value = config.get('manga_full_page_context', True)
        
        full_page_context_prompt_default = (
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
        self.full_page_context_prompt = config.get('manga_full_page_context_prompt', full_page_context_prompt_default)
        
        # If full page context prompt wasn't in config, save it now
        if 'manga_full_page_context_prompt' not in config:
            self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            print("[MANGA_INIT] Saved default full page context prompt to config")

        # Manga glossary workflow settings. When enabled, the batch worker OCRs
        # every selected page first, generates one glossary from all OCR text,
        # then translates using that generated glossary.
        self.manga_glossary_enabled_value = config.get('manga_glossary_enabled', False)
        self.manga_glossary_prompt = config.get('manga_glossary_prompt', self._default_manga_glossary_prompt())
        self.manga_custom_glossary_path = config.get('manga_custom_glossary_path', '')
        self.manga_generated_glossary_path = config.get('manga_generated_glossary_path', '')
        self.manga_glossary_auto_load_suppressed = config.get('manga_glossary_auto_load_suppressed', False)
        self.manga_glossary_auto_load_suppressed_root = config.get('manga_glossary_auto_load_suppressed_root', '')
        self.manga_split_first_level_subfolders_value = config.get('manga_split_first_level_subfolders', False)
        self.manga_glossary_debug_ocr_text_value = config.get('manga_glossary_debug_ocr_text', False)
        self.compress_glossary_prompt_value = self._get_compress_glossary_prompt_value()
        self.manga_loaded_glossary_text = ''
        self.manga_generated_glossary_text = ''
 
        # Load OCR prompt
        ocr_prompt_default = self._default_manga_ocr_prompt()
        self.ocr_prompt = config.get('manga_ocr_prompt', ocr_prompt_default)
        migrated_ocr_prompt = self._migrate_legacy_manga_ocr_prompt(self.ocr_prompt)
        if migrated_ocr_prompt != self.ocr_prompt:
            self.ocr_prompt = migrated_ocr_prompt
            config['manga_ocr_prompt'] = migrated_ocr_prompt
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            print("[MANGA_INIT] Migrated legacy OCR no-text rule to [AI RESPONSE UNAVAILABLE]")
        
        # If OCR prompt wasn't in config, save it now
        if 'manga_ocr_prompt' not in config:
            self.main_gui.config['manga_ocr_prompt'] = self.ocr_prompt
            print("[MANGA_INIT] Saved default OCR prompt to config")
        # Visual context setting
        self.visual_context_enabled_value = self.main_gui.config.get('manga_visual_context_enabled', True)
        self.qwen2vl_model_size = config.get('qwen2vl_model_size', '1')  # Default to '1' (2B)
        
        # Initialize RapidOCR settings
        self.rapidocr_use_recognition_value = self.main_gui.config.get('rapidocr_use_recognition', True)
        self.rapidocr_language_value = self.main_gui.config.get('rapidocr_language', 'auto')
        self.rapidocr_detection_mode_value = self.main_gui.config.get('rapidocr_detection_mode', 'document')
        self.custom_api_ocr_batch_enabled_value = bool(config.get('manga_custom_api_ocr_batch_enabled', True))
        try:
            self.custom_api_ocr_batch_size_value = max(1, int(config.get('manga_custom_api_ocr_batch_size', 5)))
        except Exception:
            self.custom_api_ocr_batch_size_value = 5
        self.batch_image_requests_enabled_value = bool(config.get('manga_batch_image_requests_enabled', True))
        try:
            self.batch_image_requests_size_value = max(1, int(config.get('manga_batch_image_requests_size', 5)))
        except (TypeError, ValueError):
            self.batch_image_requests_size_value = 5
        self.manga_ocr_disable_thinking_value = bool(
            ((config.get('manga_settings') or {}).get('ocr') or {}).get('manga_ocr_disable_thinking', True)
        )
        self.manga_image_request_quality_enabled_value = bool(
            ((config.get('manga_settings') or {}).get('compression') or {}).get('enabled', False)
        )
        self._sync_manga_image_request_quality_env()

        # Output settings
        self.create_cbz_at_end_value = config.get('manga_create_cbz_at_end', True)
        self.auto_consolidate_images_value = config.get('manga_auto_consolidate_images', True)

    def _save_rendering_settings(self):
        """Save rendering settings with validation"""
        # Don't save during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
        
        # Before saving, refresh key toggle values from widgets if present
        try:
            if hasattr(self, 'context_checkbox'):
                self.full_page_context_value = bool(self.context_checkbox.isChecked())
            if hasattr(self, 'manga_glossary_checkbox'):
                self.manga_glossary_enabled_value = bool(self.manga_glossary_checkbox.isChecked())
            if hasattr(self, 'manga_compress_glossary_checkbox'):
                self.compress_glossary_prompt_value = bool(self.manga_compress_glossary_checkbox.isChecked())
            if hasattr(self, 'manga_glossary_debug_ocr_checkbox'):
                self.manga_glossary_debug_ocr_text_value = bool(self.manga_glossary_debug_ocr_checkbox.isChecked())
            if hasattr(self, 'visual_context_checkbox'):
                self.visual_context_enabled_value = bool(self.visual_context_checkbox.isChecked())
            if hasattr(self, 'manga_ocr_disable_thinking_checkbox'):
                self.manga_ocr_disable_thinking_value = bool(self.manga_ocr_disable_thinking_checkbox.isChecked())
            if hasattr(self, 'custom_api_ocr_batch_checkbox'):
                self.custom_api_ocr_batch_enabled_value = bool(self.custom_api_ocr_batch_checkbox.isChecked())
            if hasattr(self, 'custom_api_ocr_batch_size_spinbox'):
                self.custom_api_ocr_batch_size_value = int(self.custom_api_ocr_batch_size_spinbox.value())
            if hasattr(self, 'batch_image_requests_checkbox'):
                self.batch_image_requests_enabled_value = bool(self.batch_image_requests_checkbox.isChecked())
            if hasattr(self, 'batch_image_requests_spinbox'):
                self.batch_image_requests_size_value = int(self.batch_image_requests_spinbox.value())
            if hasattr(self, 'manga_image_request_quality_checkbox'):
                self.manga_image_request_quality_enabled_value = bool(self.manga_image_request_quality_checkbox.isChecked())
            if hasattr(self, 'manga_image_request_format_combo'):
                ms = self.main_gui.config.setdefault('manga_settings', {})
                comp = ms.setdefault('compression', {})
                comp['format'] = str(self.manga_image_request_format_combo.currentText() or 'jpeg').strip().lower()
            if hasattr(self, 'manga_image_request_quality_spinbox'):
                ms = self.main_gui.config.setdefault('manga_settings', {})
                comp = ms.setdefault('compression', {})
                fmt = str(comp.get('format', 'jpeg') or 'jpeg').strip().lower()
                if fmt == 'png':
                    comp['png_compress_level'] = int(self.manga_image_request_quality_spinbox.value())
                elif fmt == 'webp':
                    comp['webp_quality'] = int(self.manga_image_request_quality_spinbox.value())
                else:
                    comp['jpeg_quality'] = int(self.manga_image_request_quality_spinbox.value())
            if hasattr(self, 'create_cbz_checkbox'):
                self.create_cbz_at_end_value = bool(self.create_cbz_checkbox.isChecked())
            if hasattr(self, 'auto_consolidate_checkbox'):
                self.auto_consolidate_images_value = bool(self.auto_consolidate_checkbox.isChecked())
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
            if hasattr(self, 'custom_image_edit_endpoint_value'):
                self.main_gui.config['custom_image_edit_endpoint'] = self.custom_image_edit_endpoint_value
                self.main_gui.config['manga_custom-image-edit_model_path'] = self.custom_image_edit_endpoint_value
            if hasattr(self, 'use_custom_image_edit_endpoint_value'):
                self.main_gui.config['use_custom_image_edit_endpoint'] = bool(self.use_custom_image_edit_endpoint_value)
                self._set_custom_image_edit_env()
            if hasattr(self, 'disable_inpaint_performance_mode_value'):
                self.main_gui.config['manga_disable_inpaint_performance_mode'] = bool(self.disable_inpaint_performance_mode_value)
                self.main_gui.manga_disable_inpaint_performance_mode_var = bool(self.disable_inpaint_performance_mode_value)
                inpaint['disable_performance_mode'] = bool(self.disable_inpaint_performance_mode_value)
            if hasattr(self, 'custom_image_edit_system_prompt_value'):
                self.main_gui.config['custom_image_edit_system_prompt'] = (
                    self.custom_image_edit_system_prompt_value or self._default_custom_image_edit_system_prompt()
                )
                self.main_gui.custom_image_edit_system_prompt_var = self.main_gui.config['custom_image_edit_system_prompt']
            if hasattr(self, 'custom_image_edit_user_prompt_value'):
                self.main_gui.config['custom_image_edit_user_prompt'] = self.custom_image_edit_user_prompt_value or ''
                self.main_gui.custom_image_edit_user_prompt_var = self.main_gui.config['custom_image_edit_user_prompt']
            if hasattr(self, 'custom_image_edit_full_page_output_value'):
                self.main_gui.config['custom_image_edit_full_page_output'] = int(self.custom_image_edit_full_page_output_value)
                self.main_gui.custom_image_edit_full_page_output_var = int(self.custom_image_edit_full_page_output_value)
            
            # Add new inpainting settings
            if hasattr(self, 'inpaint_method_value'):
                self.main_gui.config['manga_inpaint_method'] = self.inpaint_method_value
            if hasattr(self, 'local_model_type_value'):
                self.main_gui.config['manga_local_inpaint_model'] = self.local_model_type_value
            
            # Save model paths for each type
            for model_type in  ['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'custom-image-edit', 'mat', 'ollama', 'sd_local']:
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
            
            # Save safe area settings
            if hasattr(self, 'safe_area_enabled_value'):
                self.main_gui.config['manga_safe_area_enabled'] = bool(self.safe_area_enabled_value)
            if hasattr(self, 'safe_area_scale_value'):
                try:
                    self.main_gui.config['manga_safe_area_scale'] = float(self.safe_area_scale_value)
                except Exception:
                    pass
            
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
            
            # CRITICAL: Invalidate cached MangaTranslator instance to force fresh settings on next render
            # This ensures font algorithm and auto_fit_style changes take effect immediately
            try:
                if hasattr(self, '_manga_translator'):
                    self._manga_translator = None
                    print("[SETTINGS] Cleared cached _manga_translator to apply new font settings")
            except Exception as e:
                print(f"[SETTINGS] Failed to clear _manga_translator: {e}")
            
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
            if hasattr(self, 'create_cbz_at_end_value'):
                self.main_gui.config['manga_create_cbz_at_end'] = self.create_cbz_at_end_value
            if hasattr(self, 'auto_consolidate_images_value'):
                self.main_gui.config['manga_auto_consolidate_images'] = self.auto_consolidate_images_value
            
            # Save full page context settings
            if hasattr(self, 'full_page_context_value'):
                self.main_gui.config['manga_full_page_context'] = self.full_page_context_value
            if hasattr(self, 'full_page_context_prompt'):
                self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            if hasattr(self, 'manga_glossary_checkbox'):
                self.manga_glossary_enabled_value = bool(self.manga_glossary_checkbox.isChecked())
            if hasattr(self, 'manga_glossary_enabled_value'):
                self.main_gui.config['manga_glossary_enabled'] = bool(self.manga_glossary_enabled_value)
            if hasattr(self, 'manga_glossary_prompt'):
                self.main_gui.config['manga_glossary_prompt'] = self.manga_glossary_prompt
            if hasattr(self, 'manga_custom_glossary_path'):
                self.main_gui.config['manga_custom_glossary_path'] = self.manga_custom_glossary_path
            if hasattr(self, 'manga_generated_glossary_path'):
                self.main_gui.config['manga_generated_glossary_path'] = self.manga_generated_glossary_path
            if hasattr(self, 'manga_glossary_auto_load_suppressed'):
                self.main_gui.config['manga_glossary_auto_load_suppressed'] = bool(self.manga_glossary_auto_load_suppressed)
            if hasattr(self, 'manga_glossary_auto_load_suppressed_root'):
                self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = self.manga_glossary_auto_load_suppressed_root
            if hasattr(self, 'compress_glossary_prompt_value'):
                self._sync_compress_glossary_prompt_setting(self.compress_glossary_prompt_value)
            if hasattr(self, 'manga_glossary_debug_ocr_text_value'):
                self.main_gui.config['manga_glossary_debug_ocr_text'] = bool(self.manga_glossary_debug_ocr_text_value)
            if hasattr(self, 'manga_split_first_level_subfolders_value'):
                self.main_gui.config['manga_split_first_level_subfolders'] = bool(self.manga_split_first_level_subfolders_value)
            
            # Persist visual context setting alongside other toggles
            if hasattr(self, 'visual_context_enabled_value'):
                self.main_gui.config['manga_visual_context_enabled'] = self.visual_context_enabled_value
            if hasattr(self, 'manga_ocr_disable_thinking_value'):
                ms = self.main_gui.config.setdefault('manga_settings', {})
                ocr_set = ms.setdefault('ocr', {})
                ocr_set['manga_ocr_disable_thinking'] = bool(self.manga_ocr_disable_thinking_value)
                os.environ['MANGA_OCR_DISABLE_THINKING'] = '1' if self.manga_ocr_disable_thinking_value else '0'
            if hasattr(self, 'custom_api_ocr_batch_enabled_value'):
                self.main_gui.config['manga_custom_api_ocr_batch_enabled'] = bool(self.custom_api_ocr_batch_enabled_value)
            if hasattr(self, 'custom_api_ocr_batch_size_value'):
                self.main_gui.config['manga_custom_api_ocr_batch_size'] = int(self.custom_api_ocr_batch_size_value)
            if hasattr(self, 'batch_image_requests_enabled_value'):
                self.main_gui.config['manga_batch_image_requests_enabled'] = bool(self.batch_image_requests_enabled_value)
            if hasattr(self, 'batch_image_requests_size_value'):
                self.main_gui.config['manga_batch_image_requests_size'] = int(self.batch_image_requests_size_value)
            if hasattr(self, 'manga_image_request_quality_enabled_value'):
                ms = self.main_gui.config.setdefault('manga_settings', {})
                comp = ms.setdefault('compression', {})
                comp['enabled'] = bool(self.manga_image_request_quality_enabled_value)
                self._sync_manga_image_request_quality_env()
            
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
        # Also propagate safe area controls
        try:
            self.translator.safe_area_enabled = bool(getattr(self, 'safe_area_enabled_value', True))
            self.translator.safe_area_scale = float(getattr(self, 'safe_area_scale_value', 1.0))
        except Exception:
            pass
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
        
        # Handle inpainting mode.
        # PySide6 build: derive mode from the Skip Inpainter checkbox
        # (self.skip_inpainting_value) and the method radios
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
            mode = 'skip'
        elif inpaint_method in ('cloud', 'hybrid'):
            mode = inpaint_method
        else:
            mode = 'local'

        # Persist selected mode on translator
        self.translator.inpaint_mode = mode

        if mode == 'skip':
            self.translator.skip_inpainting = True
            self.translator.use_cloud_inpainting = False
            self._log("  Inpainting: Skipped (toggle enabled)", "info")
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
        #self._log(f"Applied rendering settings:", "info")
        #self._log(f"  Background: {self.bg_style_value} @ {int(self.bg_opacity_value/255*100)}% opacity", "info")
        # os is already imported at module level - no need to import again
       #self._log(f"  Font: {os.path.basename(self.selected_font_path) if self.selected_font_path else 'Default'}", "info")
       #self._log(f"  Minimum Font Size: {self.auto_min_size_value}pt", "info")
       #self._log(f"  Maximum Font Size: {self.max_font_size_value}pt", "info")
       #self._log(f"  Strict Text Wrapping: {'Enabled (force fit)' if self.strict_text_wrapping_value else 'Disabled (allow overflow)'}", "info")
        if self.font_size_mode_value == 'multiplier':
            self._log(f"  Font Size: Dynamic multiplier ({self.font_size_multiplier_value:.1f}x)", "info")
            if hasattr(self, 'constrain_to_bubble_value'):
                constraint_status = "constrained" if self.constrain_to_bubble_value else "unconstrained"
                self._log(f"  Text Constraint: {constraint_status}", "info")
        else:
            size_text = f"{self.font_size_value}pt" if self.font_size_value > 0 else "Auto"
            self._log(f"  Font Size: Fixed ({size_text})", "info")
        #self._log(f"  Text Color: RGB({text_color[0]}, {text_color[1]}, {text_color[2]})", "info")
        #self._log(f"  Shadow: {'Enabled' if self.shadow_enabled_value else 'Disabled'}", "info")
        try:
            self._log(f"  Free-text-only BG opacity: {'Enabled' if getattr(self, 'free_text_only_bg_opacity_value', False) else 'Disabled'}", "info")
        except Exception:
            pass
        try:
            self._log(f"  Safe area: {'Enabled' if bool(getattr(self, 'safe_area_enabled_value', False)) else 'Disabled'} (scale {float(getattr(self, 'safe_area_scale_value', 1.0)):.2f})", "info")
        except Exception:
            pass
        self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_value else 'Disabled'}", "info")
        
        # Refresh text overlays with new settings if translations exist
        try:
            if hasattr(self, '_translated_texts') and self._translated_texts and hasattr(self, 'image_preview_widget'):
                # Rebuild overlays with updated settings
                ImageRenderer._add_text_overlay_to_viewer(self, self._translated_texts)
        except Exception as e:
            # Don't fail if overlay refresh fails - just log it
            pass
        
        # Set OUTPUT_DIRECTORY environment variable from config (like translator_gui.py)
        try:
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.main_gui.config.get('output_directory')
            if override_dir:
                os.environ['OUTPUT_DIRECTORY'] = os.path.abspath(override_dir)
                self._log(f"📁 Using output override: {os.environ['OUTPUT_DIRECTORY']}", "info")
        except Exception as e:
            self._log(f"⚠️ Could not apply OUTPUT_DIRECTORY override: {e}", "warning")

    def _set_font_preset(self, preset: str):
        """Apply font sizing preset (moved from dialog)"""
        try:
            # Determine target values for the preset first
            if preset == 'small':
                self.font_algorithm_value = 'conservative'
                self.auto_min_size_value = 8
                self.max_font_size_value = 48
                self.prefer_larger_value = True
                self.bubble_size_factor_value = True
                self.line_spacing_value = 1.2
                # Small bubbles use strict wrapping and compact fit style
                self.strict_text_wrapping_value = True
                self.auto_fit_style_value = 'compact'
            elif preset == 'balanced':
                self.font_algorithm_value = 'smart'
                self.auto_min_size_value = 12
                self.max_font_size_value = 64
                self.prefer_larger_value = True
                self.bubble_size_factor_value = True
                self.line_spacing_value = 1.3
                # Balanced preset disables strict wrapping and uses balanced fit style
                self.strict_text_wrapping_value = False
                self.auto_fit_style_value = 'balanced'
            elif preset == 'large':
                self.font_algorithm_value = 'aggressive'
                self.auto_min_size_value = 14
                self.max_font_size_value = 96
                self.prefer_larger_value = True
                self.bubble_size_factor_value = False
                self.line_spacing_value = 1.4
                # Large text preset disables strict wrapping and uses readable fit style
                self.strict_text_wrapping_value = False
                self.auto_fit_style_value = 'readable'
            
            # Helper to safely set widget values without emitting signals
            def _safe_set(widget, setter_name, value):
                try:
                    prev = widget.blockSignals(True)
                except Exception:
                    prev = None
                try:
                    getattr(widget, setter_name)(value)
                    # If this is a styled checkbox, manually refresh its checkmark overlay
                    if setter_name == 'setChecked' and hasattr(widget, '_update_checkmark'):
                        try:
                            widget._update_checkmark()
                        except Exception:
                            pass
                finally:
                    try:
                        widget.blockSignals(prev if isinstance(prev, bool) else False)
                    except Exception:
                        pass
            
            # Update spinboxes
            if hasattr(self, 'min_size_spinbox'):
                _safe_set(self.min_size_spinbox, 'setValue', self.auto_min_size_value)
            if hasattr(self, 'max_size_spinbox'):
                _safe_set(self.max_size_spinbox, 'setValue', self.max_font_size_value)
            if hasattr(self, 'line_spacing_spinbox'):
                _safe_set(self.line_spacing_spinbox, 'setValue', self.line_spacing_value)
            
            # Update checkboxes
            if hasattr(self, 'prefer_larger_checkbox'):
                _safe_set(self.prefer_larger_checkbox, 'setChecked', self.prefer_larger_value)
            if hasattr(self, 'bubble_size_factor_checkbox'):
                _safe_set(self.bubble_size_factor_checkbox, 'setChecked', self.bubble_size_factor_value)
            if hasattr(self, 'strict_wrap_checkbox'):
                _safe_set(self.strict_wrap_checkbox, 'setChecked', self.strict_text_wrapping_value)
            
            # Update auto fit style radio buttons
            if hasattr(self, 'auto_fit_style_group'):
                for button in self.auto_fit_style_group.buttons():
                    if button.text().lower() == self.auto_fit_style_value:
                        try:
                            prev = button.blockSignals(True)
                        except Exception:
                            prev = None
                        try:
                            button.setChecked(True)
                        finally:
                            try:
                                button.blockSignals(prev if isinstance(prev, bool) else False)
                            except Exception:
                                pass
                        break
            
            # Update the line spacing label
            if hasattr(self, 'line_spacing_value_label'):
                self.line_spacing_value_label.setText(f"{float(self.line_spacing_value):.2f}")
            
            # Persist/apply once after all UI updates
            self._save_rendering_settings()
            try:
                self._apply_rendering_settings()
            except Exception:
                pass
            try:
                ImageRenderer._relayout_all_overlays_for_current_image(self, )
            except Exception:
                pass
        except Exception as e:
            self._log(f"Error setting preset: {e}", "debug")

    def _ensure_ocr_manager(self):
        """Return a usable OCR manager, recreating it after memory cleanup if needed."""
        if getattr(self, 'ocr_manager', None) is None:
            from ocr_manager import OCRManager
            self.ocr_manager = OCRManager(log_callback=self._log)
        return self.ocr_manager

    def _build_manga_worker_ocr_config(self) -> Dict[str, Any]:
        """Build the OCR config used by isolated manga worker translators."""
        provider = getattr(
            self,
            'ocr_provider_value',
            self.main_gui.config.get('manga_ocr_provider', 'custom-api')
        )
        ocr_config: Dict[str, Any] = {'provider': provider}
        try:
            if provider == 'google':
                google_creds = (
                    self.main_gui.config.get('google_vision_credentials', '') or
                    self.main_gui.config.get('google_cloud_credentials', '')
                )
                if google_creds and os.path.exists(google_creds):
                    ocr_config['google_credentials_path'] = google_creds
            elif provider == 'azure':
                azure_key = self.main_gui.config.get('azure_vision_key', '')
                azure_endpoint = self.main_gui.config.get('azure_vision_endpoint', '')
                if azure_key and azure_endpoint:
                    ocr_config['azure_key'] = azure_key
                    ocr_config['azure_endpoint'] = azure_endpoint
        except Exception:
            pass
        return ocr_config

    def _autoload_manga_glossary_for_selection(self) -> None:
        """Load an existing generated glossary from the selected folder's Glossary subfolder."""
        if getattr(self, 'manga_custom_glossary_path', ''):
            return
        if self._manga_glossary_auto_load_is_suppressed():
            return
        if not getattr(self, 'selected_files', None):
            self._clear_auto_loaded_manga_glossary_cache()
            return
        if len(self._manga_current_process_groups()) != 1:
            self._clear_auto_loaded_manga_glossary_cache()
            return
        previous_path = getattr(self, 'manga_generated_glossary_path', '')
        generated_path = self._find_generated_manga_glossary_path()
        if not generated_path:
            self._clear_auto_loaded_manga_glossary_cache()
            return
        try:
            glossary_text = self._load_manga_glossary_file_as_prompt_text(generated_path)
            if not glossary_text.strip():
                return
            self.manga_loaded_glossary_text = glossary_text
            self.manga_generated_glossary_path = generated_path
            self.manga_generated_glossary_text = glossary_text
            self.manga_glossary_auto_load_suppressed = False
            self.manga_glossary_auto_load_suppressed_root = ''
            self.main_gui.config['manga_generated_glossary_path'] = generated_path
            self.main_gui.config['manga_glossary_auto_load_suppressed'] = False
            self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = ''
            setattr(self.main_gui, 'manga_generated_glossary_text', glossary_text)
            setattr(self.main_gui, 'manga_generated_glossary_path', generated_path)
            if hasattr(self, 'translator') and self.translator:
                self.translator.manga_generated_glossary_text = glossary_text
                self.translator.manga_generated_glossary_path = generated_path
                self.translator._manga_glossary_prompt_logged = False
            if previous_path != generated_path:
                self._log(f"📚 Auto-loaded manga glossary: {os.path.basename(generated_path)}", "info")
        except Exception as err:
            self._log(f"⚠️ Could not auto-load manga glossary: {err}", "warning")

    def _clear_auto_loaded_manga_glossary_cache(self) -> None:
        """Clear cached generated glossary text without touching manual glossary selection."""
        if getattr(self, 'manga_custom_glossary_path', ''):
            return
        self.manga_loaded_glossary_text = ''
        self.manga_generated_glossary_text = ''
        self.manga_generated_glossary_entries = []
        self.manga_generated_glossary_path = ''
        self.main_gui.config['manga_generated_glossary_path'] = ''
        setattr(self.main_gui, 'manga_generated_glossary_text', '')
        setattr(self.main_gui, 'manga_generated_glossary_entries', [])
        setattr(self.main_gui, 'manga_generated_glossary_path', '')
        if hasattr(self, 'translator') and self.translator:
            self.translator.manga_generated_glossary_text = ''
            self.translator.manga_generated_glossary_path = ''
            self.translator._manga_glossary_prompt_logged = False

    def _reset_manga_glossary_selection_for_source_switch(self) -> None:
        """Clear folder-bound glossary state without deleting any glossary files."""
        self.manga_custom_glossary_path = ''
        self.manga_loaded_glossary_text = ''
        self.manga_generated_glossary_text = ''
        self.manga_generated_glossary_entries = []
        self.manga_generated_glossary_path = ''
        self.manga_glossary_auto_load_suppressed = False
        self.manga_glossary_auto_load_suppressed_root = ''
        self.main_gui.config['manga_custom_glossary_path'] = ''
        self.main_gui.config['manga_generated_glossary_path'] = ''
        self.main_gui.config['manga_glossary_auto_load_suppressed'] = False
        self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = ''
        setattr(self.main_gui, 'manga_custom_glossary_path', '')
        setattr(self.main_gui, 'manga_generated_glossary_text', '')
        setattr(self.main_gui, 'manga_generated_glossary_entries', [])
        setattr(self.main_gui, 'manga_generated_glossary_path', '')
        if hasattr(self, 'translator') and self.translator:
            self.translator.manga_generated_glossary_text = ''
            self.translator.manga_generated_glossary_path = ''
            self.translator._manga_glossary_prompt_logged = False
        self._update_manga_glossary_status_label()

    def _manga_glossary_auto_load_is_suppressed(self) -> bool:
        """Return True after the user explicitly clears the manga glossary."""
        try:
            suppressed = bool(
                getattr(self, 'manga_glossary_auto_load_suppressed', False)
                or self.main_gui.config.get('manga_glossary_auto_load_suppressed', False)
            )
            if not suppressed:
                return False
            suppressed_root = (
                getattr(self, 'manga_glossary_auto_load_suppressed_root', '')
                or self.main_gui.config.get('manga_glossary_auto_load_suppressed_root', '')
            )
            if not suppressed_root:
                return False
            current_root = self._current_manga_source_dir()
            if not current_root:
                return False
            same_root = (
                os.path.normcase(os.path.abspath(suppressed_root))
                == os.path.normcase(os.path.abspath(current_root))
            )
            if not same_root:
                return False
            if self._find_generated_manga_glossary_path():
                self.manga_glossary_auto_load_suppressed = False
                self.manga_glossary_auto_load_suppressed_root = ''
                self.main_gui.config['manga_glossary_auto_load_suppressed'] = False
                self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = ''
                return False
            return True
        except Exception:
            return False

    def _find_generated_manga_glossary_path(self) -> str:
        """Find a saved manga glossary generated by this workflow."""
        configured = (
            getattr(self, 'manga_generated_glossary_path', '')
            or self.main_gui.config.get('manga_generated_glossary_path', '')
        )
        exact_candidates = []
        current_glossary_dir = ""
        if getattr(self, 'selected_files', None):
            try:
                safe_name = self._manga_glossary_source_name()
                current_glossary_dir = self._manga_glossary_output_dir()
                output_json = os.path.join(current_glossary_dir, f"{safe_name}_manga_glossary.json")
                backup_json = self._manga_glossary_backup_json_path()
                legacy_backup_json = self._manga_glossary_legacy_backup_json_path()
                exact_candidates.extend([
                    output_json.replace('.json', '.csv'),
                    output_json,
                    backup_json.replace('.json', '.csv'),
                    backup_json,
                    legacy_backup_json.replace('.json', '.csv'),
                    legacy_backup_json,
                ])
            except Exception:
                pass

        existing = [path for path in exact_candidates if path and os.path.exists(path)]
        if existing:
            existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            generated_path = existing[0]
            self.manga_generated_glossary_path = generated_path
            self.main_gui.config['manga_generated_glossary_path'] = generated_path
            return generated_path

        if configured and os.path.exists(configured):
            if exact_candidates:
                try:
                    configured_abs = os.path.normcase(os.path.abspath(configured))
                    exact_abs = {os.path.normcase(os.path.abspath(path)) for path in exact_candidates}
                    if configured_abs not in exact_abs:
                        return ""
                except Exception:
                    return ""
            if current_glossary_dir:
                try:
                    configured_abs = os.path.abspath(configured)
                    allowed_roots = [current_glossary_dir]
                    try:
                        allowed_roots.append(self._manga_glossary_backup_root_dir())
                    except Exception:
                        pass
                    if not any(
                        os.path.commonpath([configured_abs, os.path.abspath(root)]) == os.path.abspath(root)
                        for root in allowed_roots
                        if root
                    ):
                        return ""
                except Exception:
                    return ""
            self.manga_generated_glossary_path = configured
            self.main_gui.config['manga_generated_glossary_path'] = configured
            return configured

        return ""

    def _format_manga_glossary_entries_for_prompt(self, entries: List[Dict[str, Any]]) -> str:
        """Convert loaded glossary entries to the token-efficient prompt format."""
        cleaned_entries = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            raw_name = str(entry.get('raw_name') or entry.get('source') or entry.get('original') or '').strip()
            translated_name = str(entry.get('translated_name') or entry.get('target') or entry.get('translated') or entry.get('name') or '').strip()
            if not raw_name or not translated_name:
                continue
            item = dict(entry)
            item['raw_name'] = raw_name
            item['translated_name'] = translated_name
            item['type'] = str(item.get('type') or 'term').strip() or 'term'
            cleaned_entries.append(item)

        if not cleaned_entries:
            return ""

        custom_fields = getattr(self.main_gui, 'custom_glossary_fields', None)
        if custom_fields is None:
            custom_fields = self.main_gui.config.get('custom_glossary_fields', [])
        if isinstance(custom_fields, str):
            try:
                custom_fields = json.loads(custom_fields)
            except Exception:
                custom_fields = []

        header_cols = ['translated_name', 'raw_name', 'gender']
        if any(any(str(k).strip().lower() == 'description' and str(v or '').strip() for k, v in entry.items()) for entry in cleaned_entries):
            header_cols.append('description')
        for field in custom_fields or []:
            field = str(field).strip()
            if field and field.lower() not in {c.lower() for c in header_cols}:
                header_cols.append(field)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in cleaned_entries:
            entry_type = str(entry.get('type') or 'term').strip() or 'term'
            grouped.setdefault(entry_type, []).append(entry)

        lines = [f"Glossary Columns: {', '.join(header_cols)}", ""]
        for entry_type in sorted(grouped.keys()):
            section = entry_type.upper()
            if not section.endswith('S'):
                section += 'S'
            lines.append(f"=== {section} ===")
            for entry in grouped[entry_type]:
                translated_name = entry.get('translated_name', '')
                raw_name = entry.get('raw_name', '')
                line = f"* {translated_name} ({raw_name})"
                gender = str(entry.get('gender', '') or '').strip()
                if gender and gender.lower() not in {'unknown', 'n/a', 'na', 'none', '-'}:
                    line += f" [{gender}]"

                description = ''
                for key, value in entry.items():
                    if isinstance(key, str) and key.strip().lower() == 'description':
                        description = str(value or '').strip()
                        break

                extra_parts = []
                for field in custom_fields or []:
                    field = str(field).strip()
                    if not field or field.lower() == 'description':
                        continue
                    value = str(entry.get(field, '') or '').strip()
                    if value:
                        extra_parts.append(f"{field}: {value}")

                if description:
                    line += f": {description}"
                if extra_parts:
                    line += f" ({', '.join(extra_parts)})"
                lines.append(line)
            lines.append("")

        return "\n".join(lines).strip()

    def _load_manga_glossary_file_as_prompt_text(self, path: str) -> str:
        """Load JSON/CSV/token-efficient glossary files and return prompt-ready text."""
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf-8-sig') as f:
                raw_text = f.read().strip()

        if raw_text.lower().startswith("glossary columns:") or raw_text.startswith("===") or raw_text.startswith("* "):
            return raw_text

        try:
            from extract_glossary_from_epub import _load_glossary_file
            entries = _load_glossary_file(path)
        except Exception:
            entries = []

        if not entries and raw_text:
            try:
                from extract_glossary_from_epub import parse_api_response
                entries = parse_api_response(raw_text)
            except Exception:
                entries = []

        return self._format_manga_glossary_entries_for_prompt(entries)

    def _copy_loaded_manga_glossary_to_output(self, source_path: str, glossary_text: str) -> str:
        """Copy a manually loaded glossary into the selected manga's Glossary folder."""
        if not source_path or not os.path.exists(source_path):
            return ""
        if not getattr(self, 'selected_files', None):
            return ""
        output_json = self._manga_glossary_output_json_path()
        ext = os.path.splitext(source_path)[1].lower()
        target_path = output_json if ext == '.json' else output_json.replace('.json', '.csv')
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        source_abs = os.path.normcase(os.path.abspath(source_path))
        target_abs = os.path.normcase(os.path.abspath(target_path))
        if source_abs != target_abs:
            if ext in {'.csv', '.json'}:
                shutil.copy2(source_path, target_path)
            else:
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write((glossary_text or '').strip() + "\n")
        os.utime(target_path, None)
        return target_path

    def _get_loaded_manga_glossary_text(self) -> str:
        """Return loaded or generated glossary text, reloading from disk if needed."""
        loaded_text = getattr(self, 'manga_loaded_glossary_text', '')
        if isinstance(loaded_text, str) and loaded_text.strip():
            return loaded_text.strip()

        path = getattr(self, 'manga_custom_glossary_path', '') or self.main_gui.config.get('manga_custom_glossary_path', '')
        if path and os.path.exists(path):
            loaded_text = self._load_manga_glossary_file_as_prompt_text(path)
            self.manga_loaded_glossary_text = loaded_text
            return loaded_text.strip()

        if self._manga_glossary_auto_load_is_suppressed():
            return ""

        generated_text = getattr(self, 'manga_generated_glossary_text', '') or getattr(self.main_gui, 'manga_generated_glossary_text', '')
        if isinstance(generated_text, str) and generated_text.strip():
            self.manga_loaded_glossary_text = generated_text
            return generated_text.strip()

        generated_path = self._find_generated_manga_glossary_path()
        if generated_path:
            loaded_text = self._load_manga_glossary_file_as_prompt_text(generated_path)
            self.manga_loaded_glossary_text = loaded_text
            self.manga_generated_glossary_path = generated_path
            if loaded_text.strip():
                self._update_manga_glossary_status_label()
            return loaded_text.strip()
        return ""

    def _update_manga_glossary_status_label(self):
        """Refresh the small status line for loaded manga glossaries."""
        label = getattr(self, 'manga_glossary_status_label', None)
        if not label:
            return
        path = getattr(self, 'manga_custom_glossary_path', '') or ''
        if path and os.path.exists(path):
            label.setText(f"Loaded glossary: {os.path.basename(path)}")
            label.setToolTip(path)
        elif path:
            label.setText(f"Loaded glossary missing: {os.path.basename(path)}")
            label.setToolTip(path)
        else:
            generated_path = "" if self._manga_glossary_auto_load_is_suppressed() else self._find_generated_manga_glossary_path()
            if generated_path:
                label.setText(f"Generated glossary: {os.path.basename(generated_path)}")
                label.setToolTip(generated_path)
            else:
                label.setText("Loaded glossary: none")
                label.setToolTip("")

    def _build_manga_glossary_input(self, ocr_pages: List[Dict[str, Any]]) -> str:
        """Build the single glossary-generation request body from all OCR pages."""
        lines = [
            "Generate a translation glossary from the OCR text below.",
            "The text is grouped by manga page. Keep page labels only as context; do not output them.",
            ""
        ]
        for page in ocr_pages:
            filename = os.path.basename(page.get('path', 'page'))
            lines.append(f"=== PAGE {page.get('index', '?')} START: {filename} ===")
            for region_idx, text in enumerate(page.get('texts', [])):
                text = str(text or '').strip()
                if text:
                    lines.append(f"[{region_idx}] {text}")
            lines.append(f"=== PAGE {page.get('index', '?')} END ===")
            lines.append("")
        return "\n".join(lines).strip()

    def _manga_glossary_source_name(self) -> str:
        source_dir = self._current_manga_source_dir()
        folder_name = os.path.basename(os.path.normpath(source_dir)) if source_dir else ''
        if not folder_name:
            output_root = os.environ.get('OUTPUT_DIRECTORY') or getattr(self.main_gui, 'config', {}).get('output_directory') or os.getcwd()
            folder_name = os.path.basename(os.path.normpath(output_root)) or 'manga'
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', folder_name).strip('_') or 'manga'

    def _manga_glossary_shared_root(self) -> str:
        """Match the shared glossary root used by regular glossary extraction."""
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or getattr(self.main_gui, 'config', {}).get('output_directory')
        if override_dir:
            return os.path.abspath(override_dir)
        return _get_app_dir()

    def _manga_glossary_output_dir(self) -> str:
        """Primary manga glossary directory used for runtime auto-mapping."""
        source_dir = self._current_manga_source_dir()
        parent_dir = os.environ.get('OUTPUT_DIRECTORY') or source_dir or os.getcwd()
        return os.path.join(parent_dir, "Glossary")

    def _manga_glossary_backup_dir(self) -> str:
        """Per-manga secondary backup location for generated manga glossaries."""
        safe_name = self._manga_glossary_source_name()
        root = self._manga_glossary_backup_root_dir()
        self._migrate_legacy_manga_glossary_backup(root, safe_name)
        return get_book_glossary_dir(root, safe_name)

    def _manga_glossary_backup_root_dir(self) -> str:
        """Root backup folder that contains one subfolder per manga source."""
        return os.path.join(self._manga_glossary_shared_root(), "MangaGlossary_Backup")

    def _manga_glossary_legacy_backup_json_path(self) -> str:
        safe_name = self._manga_glossary_source_name()
        return os.path.join(self._manga_glossary_backup_root_dir(), f"{safe_name}_manga_glossary.json")

    def _migrate_legacy_manga_glossary_backup(self, backup_root: str = None, safe_name: str = None) -> None:
        safe_name = safe_name or self._manga_glossary_source_name()
        backup_root = backup_root or self._manga_glossary_backup_root_dir()
        try:
            migrate_legacy_named_files(
                backup_root,
                safe_name,
                [
                    f"{safe_name}_manga_glossary.json",
                    f"{safe_name}_manga_glossary.csv",
                ],
                logger=lambda msg: self._log(msg, "info"),
            )
        except Exception:
            pass

    def _manga_glossary_candidate_dirs(self) -> List[str]:
        dirs = []
        try:
            dirs.append(self._manga_glossary_output_dir())
        except Exception:
            pass
        seen = set()
        unique_dirs = []
        for path in dirs:
            if not path:
                continue
            norm = os.path.normcase(os.path.abspath(path))
            if norm not in seen:
                seen.add(norm)
                unique_dirs.append(path)
        return unique_dirs

    def _manga_glossary_output_json_path(self) -> str:
        safe_name = self._manga_glossary_source_name()
        glossary_dir = self._manga_glossary_output_dir()
        os.makedirs(glossary_dir, exist_ok=True)
        return os.path.join(glossary_dir, f"{safe_name}_manga_glossary.json")

    def _manga_glossary_backup_json_path(self) -> str:
        safe_name = self._manga_glossary_source_name()
        glossary_dir = self._manga_glossary_backup_dir()
        os.makedirs(glossary_dir, exist_ok=True)
        return os.path.join(glossary_dir, f"{safe_name}_manga_glossary.json")

    def _manga_glossary_debug_dirs(self) -> List[str]:
        """Return debug subfolders beside both generated manga glossary copies."""
        dirs = []
        for base_dir in (self._manga_glossary_output_dir(), self._manga_glossary_backup_dir()):
            try:
                if base_dir:
                    dirs.append(os.path.join(base_dir, "debug"))
            except Exception:
                pass
        seen = set()
        unique_dirs = []
        for path in dirs:
            norm = os.path.normcase(os.path.abspath(path))
            if norm not in seen:
                seen.add(norm)
                unique_dirs.append(path)
        return unique_dirs

    def _manga_glossary_debug_json_safe(self, value: Any) -> Any:
        """Convert parsed glossary/debug values into JSON-safe primitives."""
        if isinstance(value, dict):
            return {str(k): self._manga_glossary_debug_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._manga_glossary_debug_json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write_manga_glossary_debug_file(self, debug_dir: str, filename: str, content: str) -> str:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, filename)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
            if content and not content.endswith('\n'):
                f.write('\n')
        return path

    def _save_manga_glossary_generation_debug(
        self,
        ocr_pages: List[Dict[str, Any]],
        combined_text: str,
        response_text: str,
        parsed_entries: List[Dict[str, Any]],
        valid_entries_before_dedupe: List[Dict[str, Any]],
    ) -> None:
        """Save optional manga glossary generation debug artifacts."""
        if not self._manga_glossary_debug_ocr_enabled():
            return

        debug_dirs = self._manga_glossary_debug_dirs()
        if not debug_dirs:
            return

        ocr_pages_payload = []
        for page in ocr_pages or []:
            ocr_pages_payload.append({
                "index": page.get("index"),
                "path": page.get("path"),
                "filename": os.path.basename(page.get("path", "")),
                "texts": [str(text or "") for text in (page.get("texts", []) or [])],
            })

        total_regions = sum(len(page.get("texts", []) or []) for page in (ocr_pages or []))
        manifest = {
            "source": self._manga_glossary_source_name(),
            "pages": len(ocr_pages or []),
            "ocr_regions": total_regions,
            "raw_response_chars": len(response_text or ""),
            "parsed_undeduped_entries": len(parsed_entries or []),
            "valid_undeduped_entries": len(valid_entries_before_dedupe or []),
            "note": "Raw/valid entries are saved before manga glossary deduplication.",
        }

        written = []
        try:
            for debug_dir in debug_dirs:
                written.append(self._write_manga_glossary_debug_file(debug_dir, "ocr_text.txt", combined_text or ""))
                self._write_manga_glossary_debug_file(
                    debug_dir,
                    "ocr_pages.json",
                    json.dumps(ocr_pages_payload, ensure_ascii=False, indent=2),
                )
                self._write_manga_glossary_debug_file(debug_dir, "raw_undeduped_glossary_output.txt", response_text or "")
                self._write_manga_glossary_debug_file(
                    debug_dir,
                    "parsed_undeduped_entries.json",
                    json.dumps(self._manga_glossary_debug_json_safe(parsed_entries or []), ensure_ascii=False, indent=2),
                )
                self._write_manga_glossary_debug_file(
                    debug_dir,
                    "valid_undeduped_entries.json",
                    json.dumps(self._manga_glossary_debug_json_safe(valid_entries_before_dedupe or []), ensure_ascii=False, indent=2),
                )
                self._write_manga_glossary_debug_file(
                    debug_dir,
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
            if written:
                self._log(f"🧪 Manga glossary debug saved: {', '.join(os.path.dirname(path) for path in written)}", "info")
        except Exception as debug_err:
            self._log(f"⚠️ Could not save manga glossary debug files: {debug_err}", "warning")

    def _normalize_manga_api_response_text(self, response: Any) -> str:
        if hasattr(response, 'content'):
            response = response.content
        elif hasattr(response, 'text'):
            response = response.text
        if isinstance(response, tuple):
            response = response[0] if response else ""
        if isinstance(response, (bytes, bytearray)):
            response = response.decode('utf-8', errors='replace')
        text = str(response or '').strip()
        if text.startswith("('") or text.startswith('("'):
            try:
                import ast
                parsed = ast.literal_eval(text)
                if isinstance(parsed, tuple) and parsed:
                    text = str(parsed[0]).strip()
            except Exception:
                pass
        return text

    def _prepare_manga_glossary_env(self) -> Dict[str, Optional[str]]:
        """Set glossary extractor environment variables and return previous values."""
        keys = [
            'GLOSSARY_SYSTEM_PROMPT',
            'GLOSSARY_TARGET_LANGUAGE',
            'GLOSSARY_CUSTOM_ENTRY_TYPES',
            'GLOSSARY_CUSTOM_FIELDS',
            'GLOSSARY_USE_LEGACY_CSV',
            'GLOSSARY_OUTPUT_LEGACY_JSON',
            'USE_GLOSSARY_KEYS',
            'GLOSSARY_API_KEYS',
            'SAVE_GLOSSARY_IN_OUTPUT',
            'GLOSSARY_OUTPUT_BACKUP_DIR',
        ]
        previous = {key: os.environ.get(key) for key in keys}

        custom_fields = getattr(self.main_gui, 'custom_glossary_fields', None)
        if custom_fields is None:
            custom_fields = self.main_gui.config.get('custom_glossary_fields', self.main_gui.config.get('manual_custom_fields', []))
        if isinstance(custom_fields, str):
            try:
                custom_fields = json.loads(custom_fields)
            except Exception:
                custom_fields = []

        os.environ['GLOSSARY_SYSTEM_PROMPT'] = getattr(self, 'manga_glossary_prompt', self._default_manga_glossary_prompt())
        os.environ['GLOSSARY_TARGET_LANGUAGE'] = (
            self.main_gui.config.get('glossary_target_language')
            or self.main_gui.config.get('output_language')
            or os.environ.get('OUTPUT_LANGUAGE')
            or 'English'
        )
        os.environ['GLOSSARY_CUSTOM_ENTRY_TYPES'] = json.dumps(
            getattr(self.main_gui, 'custom_entry_types', self.main_gui.config.get('custom_entry_types', {}))
        )
        os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(custom_fields or [])
        use_legacy_csv = bool(self.main_gui.config.get('glossary_use_legacy_csv', False))
        output_legacy_json = bool(self.main_gui.config.get('glossary_output_legacy_json', False))
        try:
            if hasattr(self.main_gui, 'use_legacy_csv_var'):
                use_legacy_csv = bool(self.main_gui.use_legacy_csv_var)
            if hasattr(self.main_gui, 'glossary_output_legacy_json_var'):
                output_legacy_json = bool(self.main_gui.glossary_output_legacy_json_var)
        except Exception:
            pass

        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '1' if use_legacy_csv else '0'
        os.environ['GLOSSARY_OUTPUT_LEGACY_JSON'] = '1' if output_legacy_json else '0'
        os.environ['SAVE_GLOSSARY_IN_OUTPUT'] = '0'
        os.environ.pop('GLOSSARY_OUTPUT_BACKUP_DIR', None)

        use_glossary_keys = bool(
            getattr(self.main_gui, 'use_glossary_keys_var', self.main_gui.config.get('use_glossary_keys', False))
        )
        glossary_keys = self.main_gui.config.get('glossary_keys', []) or []
        os.environ['USE_GLOSSARY_KEYS'] = '1' if use_glossary_keys else '0'
        os.environ['GLOSSARY_API_KEYS'] = json.dumps(glossary_keys)
        if use_glossary_keys and glossary_keys:
            try:
                from unified_api_client import UnifiedClient
                UnifiedClient.set_in_memory_glossary_keys(
                    glossary_keys,
                    force_rotation=self.main_gui.config.get('force_key_rotation', True),
                    rotation_frequency=self.main_gui.config.get('rotation_frequency', 1),
                )
            except Exception:
                pass
        return previous

    def _restore_manga_glossary_env(self, previous: Dict[str, Optional[str]]) -> None:
        for key, value in (previous or {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class MangaOcrSessionMixin:
    """MangaTranslationTab's automatic OCR export / imported OCR reuse for batch runs (moved verbatim, U8)."""

    def _manga_ocr_output_dir(self) -> str:
        """Return the manga OCR folder under the application's active output root."""
        override_dir = (
            os.environ.get('OUTPUT_DIRECTORY')
            or getattr(self.main_gui, 'config', {}).get('output_directory')
        )
        # Mirror the EPUB default output root: the executable/app source
        # directory on Windows and the working directory elsewhere.
        output_root = override_dir or _get_app_dir()
        return os.path.join(os.path.abspath(output_root), "OCR Text")

    def _manga_ocr_default_filename(self) -> str:
        return f"{self._manga_glossary_source_name()}_ocr.json"

    def _manga_ocr_timestamped_export_filename(
        self,
        filename: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Return an export filename such as manga_ocr_20260731_193045.json."""
        base_name = os.path.basename(filename or self._manga_ocr_default_filename())
        stem, extension = os.path.splitext(base_name)
        export_timestamp = timestamp or time.strftime('%Y%m%d_%H%M%S')
        return f"{stem}_{export_timestamp}{extension or '.json'}"

    def _manga_ocr_save_dialog_path(self, filename: str) -> str:
        """Return a save-dialog path rooted in the active OCR output folder."""
        output_dir = self._manga_ocr_output_dir()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            # Preserve QFileDialog's normal fallback behavior when the
            # configured output folder is temporarily unavailable.
            return os.path.basename(filename)
        return os.path.join(output_dir, os.path.basename(filename))

    def _prepare_automatic_ocr_export(self, files: List[str]) -> None:
        """Create the per-run OCR manifest before translation starts."""
        source_root = self._current_manga_source_dir()
        if not source_root and files:
            try:
                source_root = os.path.commonpath([os.path.dirname(path) for path in files])
            except ValueError:
                source_root = os.path.dirname(files[0])
        output_dir = self._manga_ocr_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            self._manga_ocr_timestamped_export_filename(),
        )
        document = manga_ocr_io.create_document(
            [], workflow="automatic", source_root=source_root
        )
        with self._ocr_io_lock:
            self._automatic_ocr_document = document
            self._automatic_ocr_export_path = output_path
            manga_ocr_io.write_document(output_path, document)
        self._log(f"OCR text will be saved automatically to: {output_path}", "info")

    def _configure_manga_ocr_io(self, translator) -> None:
        """Attach imported-OCR resolution and incremental export to a translator."""
        if translator is None:
            return
        translator.ocr_import_resolver = self._resolve_imported_ocr_regions
        translator.ocr_export_callback = self._record_automatic_ocr_page

    def _refresh_imported_ocr_page_map(self, files: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        document = getattr(self, '_imported_ocr_document', None)
        if not document:
            self._imported_ocr_page_map = {}
            return {}
        files = list(files or self._current_manga_processing_files() or self.selected_files or [])
        matched = manga_ocr_io.match_document_pages(document, files)
        self._imported_ocr_page_map = {
            os.path.normcase(os.path.abspath(path)): page
            for path, page in matched.items()
        }
        return self._imported_ocr_page_map

    def _resolve_imported_ocr_regions(self, image_path: str):
        """Return fresh TextRegion objects for a matching imported page, or None."""
        document = getattr(self, '_imported_ocr_document', None)
        if not document:
            return None
        normalized = os.path.normcase(os.path.abspath(image_path))
        page = getattr(self, '_imported_ocr_page_map', {}).get(normalized)
        if page is None:
            page = self._refresh_imported_ocr_page_map().get(normalized)
        if page is None:
            return None
        from manga_translator import TextRegion
        regions = [
            manga_ocr_io.region_record_to_text_region(record, TextRegion)
            for record in page.get('regions', [])
        ]
        # Preserve provenance through the glossary workflow's OCR-only pass,
        # which forwards these same objects as explicit precomputed regions.
        for region in regions:
            region._imported_ocr_session = True
        return regions

    def _record_automatic_ocr_page(self, image_path: str, regions) -> None:
        """Incrementally persist OCR and translations without losing newer text."""
        document = getattr(self, '_automatic_ocr_document', None)
        output_path = getattr(self, '_automatic_ocr_export_path', None)
        if not document or not output_path:
            return

        files = list(self._current_manga_processing_files() or self.selected_files or [])
        try:
            page_index = files.index(image_path) + 1
        except ValueError:
            page_index = len(document.get('pages') or []) + 1
        source_root = document.get('source_root')
        page = manga_ocr_io.make_page(
            image_path,
            regions,
            index=page_index,
            source_root=source_root,
        )
        normalized = os.path.normcase(os.path.abspath(image_path))
        with self._ocr_io_lock:
            current_pages = list(document.get('pages') or [])
            replaced = False
            for index, existing in enumerate(current_pages):
                existing_path = existing.get('source_path')
                if existing_path and os.path.normcase(os.path.abspath(existing_path)) == normalized:
                    # A page is saved once immediately after OCR and again after
                    # translation. Never let a late OCR-only callback erase text
                    # already written by the translated callback.
                    existing_regions = list(existing.get('regions') or [])
                    existing_by_rect_index = {
                        region.get('rect_index'): region
                        for region in existing_regions
                        if isinstance(region, dict) and region.get('rect_index') is not None
                    }
                    for region_index, incoming_region in enumerate(page.get('regions') or []):
                        if not isinstance(incoming_region, dict):
                            continue
                        previous_region = existing_by_rect_index.get(incoming_region.get('rect_index'))
                        if previous_region is None and region_index < len(existing_regions):
                            previous_region = existing_regions[region_index]
                        if (
                            isinstance(previous_region, dict)
                            and not str(incoming_region.get('translated_text') or '').strip()
                            and str(previous_region.get('translated_text') or '').strip()
                        ):
                            incoming_region['translated_text'] = previous_region['translated_text']
                    current_pages[index] = page
                    replaced = True
                    break
            if not replaced:
                current_pages.append(page)
            current_pages.sort(key=lambda item: int(item.get('index', 0) or 0))
            document['pages'] = current_pages
            manga_ocr_io.write_document(output_path, document)
        translated_count = sum(
            1
            for region in page.get('regions') or []
            if isinstance(region, dict) and str(region.get('translated_text') or '').strip()
        )
        self._log(
            f"Saved OCR session: {os.path.basename(image_path)} "
            f"({len(page.get('regions') or [])} regions, {translated_count} translated)",
            "debug",
        )

    def _latest_automatic_ocr_path(self) -> Optional[str]:
        current = getattr(self, '_automatic_ocr_export_path', None)
        if current and os.path.isfile(current):
            return current
        output_dir = self._manga_ocr_output_dir()
        try:
            candidates = [
                os.path.join(output_dir, name)
                for name in os.listdir(output_dir)
                if name.lower().endswith('.json')
            ]
            return max(candidates, key=os.path.getmtime) if candidates else None
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Headless tab state (mobile / host tools): the same mixins without Qt
# ---------------------------------------------------------------------------

#: How MangaTranslationTab fills the widget-backed values a headless tab replays (the
#: expressions in ``_build_pyside6_interface``); tests/test_manga_env.py checks the desktop
#: still reads like this.
STARTUP_WIDGET_SOURCES = {
    "ocr_provider_value": (
        "self.ocr_provider_value = (\n"
        "            self.main_gui.config.get('manga_ocr_provider')\n"
        "            or self.main_gui.config.get('ocr_provider')\n"
        "            or 'custom-api'\n"
        "        )"
    ),
    "azure_key_entry": "saved_key = self.main_gui.config.get('azure_vision_key', '')",
    "azure_endpoint_entry": (
        "saved_endpoint = self.main_gui.config.get('azure_vision_endpoint', "
        "'https://YOUR-RESOURCE.cognitiveservices.azure.com/')"
    ),
}
_AZURE_ENDPOINT_PLACEHOLDER = 'https://YOUR-RESOURCE.cognitiveservices.azure.com/'


class HeadlessMangaState(MangaEnvMixin, MangaOcrSessionMixin, MangaFilesMixin, MangaHooksMixin):
    """The non-GUI state of ``MangaTranslationTab`` over a duck-typed ``main_gui``.

    Construction replays ``MangaTranslationTab.__init__`` without its widgets, in order:
    ``apply_manga_startup_thread_limits(main_gui)``, ``_init_manga_run_state()``, the image
    state manager (optional here: mobile passes ``manga_editor_core``'s worker-free one),
    ``_load_rendering_settings()`` + ``_init_manga_prompt_state()`` under ``_initializing``,
    then the values ``_build_pyside6_interface`` puts into the widgets the moved code reads
    (:data:`STARTUP_WIDGET_SOURCES`) and a :class:`~manga_files_core.FileListShim`.

    ``main_gui`` is a ``headless_owner.HeadlessOwner`` on mobile (built on the job thread
    under JOB_LOCK); ``host`` (optional, a ``job_runner.JobHost``) receives the log lines.
    """

    def __init__(self, main_gui, *, host=None, ocr_provider=None, image_state_manager=None):
        from headless_owner import TextShim

        apply_manga_startup_thread_limits(main_gui)
        self.host = host
        self.main_gui = main_gui
        self._init_manga_run_state()
        self.image_state_manager = image_state_manager
        self._current_image_path = None
        self._initializing = True
        self._load_rendering_settings()
        self._init_manga_prompt_state()
        config = self.main_gui.config
        self.ocr_provider_value = ocr_provider or (
            config.get('manga_ocr_provider')
            or config.get('ocr_provider')
            or 'custom-api'
        )
        self.azure_key_entry = TextShim(config.get('azure_vision_key', ''))
        self.azure_endpoint_entry = TextShim(config.get('azure_vision_endpoint', _AZURE_ENDPOINT_PLACEHOLDER))
        self.file_listbox = FileListShim(self)
        self._use_circle_shapes = False
        self._last_error = ''
        self._initializing = False
        # TranslatorGUI.open_manga_translator: ``self.manga_translator = MangaTranslationTab(...)``;
        # MangaTranslator reads the live tab through it (local inpaint method).
        try:
            self.main_gui.manga_translator = self
        except Exception:
            pass

    # ---- tab callbacks -> job host ---------------------------------------------------------
    def _log(self, message, level="info"):
        if level == "error":
            self._last_error = str(message)
        log = getattr(getattr(self, 'host', None), 'log', None)
        if callable(log):
            try:
                log(str(message), level=level)
                return
            except TypeError:
                log(str(message))
                return
        print(message)


class MangaRunEnv(dict):
    """:func:`build_manga_run_env` result: the env delta (``{name: value or None if removed}``).

    Attributes: ``ocr_config``, ``api_key_source`` (``'entry'`` / ``'config'`` /
    ``'own-auth'`` / ``'custom-api'`` / ``''``), ``model``, ``needs_new_client``, ``state``
    (the :class:`HeadlessMangaState` that ran the steps).
    """

    ocr_config: dict = {}
    model: str = ''
    needs_new_client: bool = False
    api_key_source: str = ''
    state: Any = None


class MangaRunEnvError(RuntimeError):
    """The desktop start would have aborted (missing credentials / key); ``str()`` is its log line."""


def _effective_env():
    try:
        import large_env
        store = dict(getattr(large_env, '_store', {}) or {})
    except Exception:
        store = {}
    env = dict(os.environ)
    env.update(store)
    return env


def build_manga_run_env(owner, manga_settings=None, *, ocr_provider=None, ocr_prompt=None, host=None,
                        state=None, include_batch=True):
    """Run a desktop manga batch start's environment steps on a headless tab; return the env delta.

    The steps are ``MangaTranslationTab._start_translation_heavy``'s, verbatim and in order:
    ``_reset_manga_graceful_stop_env`` (GRACEFUL_STOP*), ``_prepare_manga_run_env`` (thread
    limits, OCR config + credential checks, API key / model, a new ``UnifiedClient`` on
    ``owner.client`` with the multi-key / image / fallback pools when the model changed,
    Vision keys, and for custom-api OCR every ``owner._get_environment_variables()`` entry
    except SYSTEM_PROMPT plus ``OCR_SYSTEM_PROMPT`` / ``MANGA_IMAGE_REQUEST_*``) and
    ``_apply_manga_batch_env`` (BATCH_*). Side effects are the desktop's: the process env,
    ``large_env``, the ``UnifiedClient`` class key pools and ``owner.client`` /
    ``owner.config`` are changed, so callers run this on the job thread under JOB_LOCK (or
    inside ``job_runner.scoped_process_state``).

    *manga_settings* replaces ``owner.config['manga_settings']`` (deep copy) first;
    *ocr_provider* overrides ``manga_ocr_provider``; *ocr_prompt* the loaded OCR prompt
    (the step still prefers ``config['manga_ocr_prompt']`` like the desktop).
    Raises :class:`MangaRunEnvError` when the desktop start would have stopped.
    """
    import copy as _copy

    if manga_settings is not None:
        owner.config['manga_settings'] = _copy.deepcopy(manga_settings)
    if state is None:
        state = HeadlessMangaState(owner, host=host, ocr_provider=ocr_provider)
    if ocr_prompt is not None:
        state.ocr_prompt = ocr_prompt
    before = _effective_env()
    state._reset_manga_graceful_stop_env()
    prepared = state._prepare_manga_run_env(None)
    if prepared is None:
        raise MangaRunEnvError(state._last_error or 'manga run start aborted')
    if include_batch:
        state._apply_manga_batch_env()
    after = _effective_env()
    delta = MangaRunEnv(
        (key, after.get(key)) for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)
    )
    ocr_config, api_key, model, needs_new_client = prepared
    delta.ocr_config = dict(ocr_config)
    delta.model = model
    delta.needs_new_client = bool(needs_new_client)
    entry_key = ''
    try:
        entry_key = (owner.api_key_entry.text() or '').strip()
    except Exception:
        entry_key = ''
    if not api_key:
        delta.api_key_source = ''
    elif api_key == 'own-auth':
        delta.api_key_source = 'own-auth'
    elif api_key == 'dummy-key-for-custom-api':
        delta.api_key_source = 'custom-api'
    elif entry_key and api_key == entry_key:
        delta.api_key_source = 'entry'
    else:
        delta.api_key_source = 'config'
    delta.state = state
    return delta


def apply_rendering_settings(translator, owner, *, state=None, host=None):
    """``_apply_rendering_settings`` (moved verbatim) for *translator* with *owner*'s config.

    *owner* is the duck-typed main GUI (``HeadlessOwner`` / anything with ``.config``); a
    plain config dict is accepted too. Pushes fonts, colours, shadow, sizing, safe area,
    caps, wrapping, constrain-to-bubble and the inpainting mode, and sets OUTPUT_DIRECTORY
    from ``config['output_directory']`` when the env has none. Returns the state used.
    """
    if isinstance(owner, dict):
        import types

        owner = types.SimpleNamespace(config=owner)
    if state is None:
        state = HeadlessMangaState(owner, host=host)
    state.translator = translator
    state._apply_rendering_settings()
    return state


class _ConfigView:
    """The two attributes ``_build_manga_worker_ocr_config`` reads."""

    def __init__(self, config, ocr_provider=None):
        import types

        self.main_gui = types.SimpleNamespace(config=config)
        if ocr_provider:
            self.ocr_provider_value = ocr_provider


def build_ocr_config(config, ocr_provider=None):
    """The OCR config the desktop gives isolated worker translators (``_build_manga_worker_ocr_config``).

    ``{'provider': ...}`` plus ``google_credentials_path`` (existing file) or
    ``azure_key`` / ``azure_endpoint`` when configured. *ocr_provider* defaults to what the
    desktop tab sets ``ocr_provider_value`` to before any worker reads it (the provider combo in
    ``_build_pyside6_interface``, :data:`STARTUP_WIDGET_SOURCES`): ``manga_ocr_provider``, then
    ``ocr_provider``, then ``'custom-api'``. The method's own getattr fallback never runs on the
    desktop, so it is never used here either.
    """
    config = config or {}
    if not ocr_provider:
        ocr_provider = (
            config.get('manga_ocr_provider')
            or config.get('ocr_provider')
            or 'custom-api'
        )
    return MangaEnvMixin._build_manga_worker_ocr_config(_ConfigView(config, ocr_provider))


def prepare_manga_glossary_env(owner, *, glossary_prompt=None):
    """``_prepare_manga_glossary_env`` for *owner*: set the glossary extractor env and return the
    previous values (pass them to :func:`restore_manga_glossary_env`)."""
    view = _ConfigView({})
    view.main_gui = owner
    if glossary_prompt is not None:
        view.manga_glossary_prompt = glossary_prompt
    view._default_manga_glossary_prompt = lambda: MangaEnvMixin._default_manga_glossary_prompt(view)
    return MangaEnvMixin._prepare_manga_glossary_env(view)


def restore_manga_glossary_env(previous):
    """Undo :func:`prepare_manga_glossary_env`."""
    return MangaEnvMixin._restore_manga_glossary_env(None, previous)


def default_manga_ocr_prompt():
    """The built-in manga OCR prompt (``_default_manga_ocr_prompt``)."""
    return MangaEnvMixin._default_manga_ocr_prompt(None)


def default_manga_glossary_prompt():
    """The built-in manga glossary prompt (``_default_manga_glossary_prompt``)."""
    return MangaEnvMixin._default_manga_glossary_prompt(None)


def migrate_legacy_manga_ocr_prompt(prompt):
    """``_migrate_legacy_manga_ocr_prompt``: update saved prompts with the old empty-response rule."""
    return MangaEnvMixin._migrate_legacy_manga_ocr_prompt(prompt)


def default_full_page_context_prompt():
    """The full-page-context prompt a manga run uses (``_init_manga_prompt_state``)."""
    import types

    view = types.SimpleNamespace(main_gui=types.SimpleNamespace(config={}),
                                 _default_manga_ocr_prompt=lambda: '')
    MangaEnvMixin._init_manga_prompt_state(view)
    return view.full_page_context_prompt


def default_custom_image_edit_system_prompt():
    """The built-in custom image-edit (inpainting) system prompt."""
    return MangaEnvMixin._default_custom_image_edit_system_prompt(None)


def import_ocr_session(state, path=None, *, document=None, files=None):
    """Load an OCR JSON (``manga_ocr_io``) for reuse by the next batch run on *state*.

    The GUI-free core of the desktop's batch OCR import: the document is validated by
    ``manga_ocr_io.load_document`` and matched to *files* (default: the state's selected
    files) with ``_refresh_imported_ocr_page_map``. Returns the ``{image_path: page}``
    matches; an empty result clears the import (the desktop warns "No Matching Pages").
    """
    if document is None:
        document = manga_ocr_io.load_document(path)
    state._imported_ocr_document = document
    matches = state._refresh_imported_ocr_page_map(files if files is not None else list(state.selected_files))
    if not matches:
        state._imported_ocr_document = None
    return matches


#: The Rendering tab's font-size preset buttons (``MangaTranslationTab._set_font_preset``).
FONT_PRESETS = ('small', 'balanced', 'large')


def _flatten_manga_config(config):
    """``{'key': v, 'manga_settings.<section>.<key>': v}`` view of a config dict."""
    out = {}
    for key, value in (config or {}).items():
        if key == 'manga_settings' and isinstance(value, dict):
            for section, inner in value.items():
                if isinstance(inner, dict):
                    for name, item in inner.items():
                        out[f'manga_settings.{section}.{name}'] = item
                else:
                    out[f'manga_settings.{section}'] = inner
        else:
            out[key] = value
    return out


def _perturbed_scalars(flat):
    """Every bool / int / float value changed (bools flipped, numbers moved); others kept."""
    out = {}
    for key, value in flat.items():
        if isinstance(value, bool):
            out[key] = not value
        elif isinstance(value, int):
            out[key] = value + 7
        elif isinstance(value, float):
            out[key] = round(value + 0.37, 2)
        else:
            out[key] = value
    return out


def _unflatten_manga_config(flat):
    config = {}
    for key, value in flat.items():
        if key.startswith('manga_settings.'):
            parts = key.split('.', 2)
            node = config.setdefault('manga_settings', {})
            if len(parts) == 3:
                node.setdefault(parts[1], {})[parts[2]] = value
            else:
                node[parts[1]] = value
        else:
            config[key] = value
    return config


def font_preset_updates(preset, config=None):
    """The config entries a Rendering font-size preset button writes (``_set_font_preset``).

    The moved desktop method runs on scratch headless tabs (``main_gui`` is a bare namespace
    over a deep copy of *config*: nothing is saved). The result holds EVERY entry the button
    sets, with the preset's value, whatever *config* holds now (applying it reproduces the
    click), keyed like ``'manga_strict_text_wrapping'`` or
    ``'manga_settings.font_sizing.algorithm'``. Which entries those are is measured, not
    listed: against a plain save of the same tab, over the given config and over a copy whose
    flags and numbers were all changed (so a value that already equals the preset's still
    shows up), across every preset. The process environment the scratch tabs touch (thread
    limits, MANGA_IMAGE_REQUEST_*, custom image-edit env) is put back. Unknown presets
    change nothing (``{}``).
    """
    import copy as _copy
    import types

    if preset not in FONT_PRESETS:
        return {}

    def run(base, press):
        owner = types.SimpleNamespace(config=_copy.deepcopy(base))
        state = HeadlessMangaState(owner, host=types.SimpleNamespace(log=lambda *_a, **_k: None))
        if press:
            state._set_font_preset(press)
        else:
            state._save_rendering_settings()
        return _flatten_manga_config(owner.config)

    before_env = dict(os.environ)
    try:
        base = _copy.deepcopy(dict(config or {}))
        saved = run(base, None)
        perturbed = _unflatten_manga_config(_perturbed_scalars(saved))
        saved_perturbed = run(perturbed, None)
        pressed = {name: run(base, name) for name in FONT_PRESETS}
        pressed_perturbed = {name: run(perturbed, name) for name in FONT_PRESETS}
    finally:
        for key in set(os.environ) - set(before_env):
            os.environ.pop(key, None)
        for key, value in before_env.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
    missing = object()
    written = set()
    for name in FONT_PRESETS:
        for reference, result in ((saved, pressed[name]), (saved_perturbed, pressed_perturbed[name])):
            written.update(key for key, value in result.items() if reference.get(key, missing) != value)
    return {key: _copy.deepcopy(pressed[preset][key]) for key in sorted(written) if key in pressed[preset]}

# ---------------------------------------------------------------------------
# U9: Rendering › Reset to Defaults and the custom image-edit endpoint test (shared with mobile)
# ---------------------------------------------------------------------------


def rendering_reset_updates(config=None):
    """The config entries Rendering › Reset to Defaults writes (``_reset_rendering_to_defaults``:
    ``manga_settings_defaults.RENDERING_RESET_VALUES`` on the tab, then ``_save_rendering_settings``).

    Measured like :func:`font_preset_updates`, on scratch headless tabs over a deep copy of *config*
    (nothing is saved): the entries the save writes from the reset attributes are the ones that
    change when those attributes change (each value shifted: flags flipped, numbers moved, strings
    suffixed), listed with their reset values. The process environment the scratch tabs touch is
    put back."""
    import copy as _copy
    import types

    from manga_settings_defaults import RENDERING_RESET_VALUES

    def shifted(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 0.5
        if value is None:
            return "u9-shifted"
        return f"{value}-shifted"

    def run(base, values):
        owner = types.SimpleNamespace(config=_copy.deepcopy(base))
        state = HeadlessMangaState(owner, host=types.SimpleNamespace(log=lambda *_a, **_k: None))
        for attr, value in values.items():
            setattr(state, attr, value)
        state._save_rendering_settings()
        return _flatten_manga_config(owner.config)

    before_env = dict(os.environ)
    try:
        base = _copy.deepcopy(dict(config or {}))
        reset = run(base, dict(RENDERING_RESET_VALUES))
        moved = run(base, {attr: shifted(value) for attr, value in RENDERING_RESET_VALUES.items()})
    finally:
        for key in set(os.environ) - set(before_env):
            os.environ.pop(key, None)
        for key, value in before_env.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
    missing = object()
    written = {key for key in set(reset) | set(moved) if reset.get(key, missing) != moved.get(key, missing)}
    return {key: _copy.deepcopy(reset[key]) for key in sorted(written) if key in reset}


#: The Blank-URL notice of the custom image-edit endpoint test (desktop QMessageBox text).
CUSTOM_IMAGE_EDIT_BLANK_MESSAGE = (
    "Blank URL uses the current main image provider/model. There is no separate custom endpoint URL to test."
)


def normalize_custom_image_edit_url(url):
    """The endpoint URL the test probes (moved from ``_test_custom_image_edit_endpoint``): a bare host
    gets ``http://`` (local hosts) or ``https://``, no trailing slash."""
    if not url.startswith(('http://', 'https://')):
        lower = url.lower()
        url = ('http://' if lower.startswith(('localhost', '127.', '0.0.0.0', '[')) else 'https://') + url
    url = url.rstrip('/')
    return url


def probe_custom_image_edit_endpoint(url, config, http_get=None):
    """``GET {url}/models`` with the desktop key order (CUSTOM_IMAGE_EDIT_API_KEY, OPENAI_API_KEY, the
    config ``api_key``, ``sk-local``); returns ``(box, status_text, colour, title, message)`` with the
    desktop status label / message box texts (moved from ``_test_custom_image_edit_endpoint``)."""
    if http_get is None:
        import requests
        http_get = requests.get
    headers = {
        'Authorization': f"Bearer {os.environ.get('CUSTOM_IMAGE_EDIT_API_KEY') or os.environ.get('OPENAI_API_KEY') or config.get('api_key', '') or 'sk-local'}"
    }
    resp = http_get(f"{url}/models", headers=headers, timeout=10)
    if resp.status_code in (200, 201):
        return ("information", "Image edit endpoint reachable", "green", "Image Edit Endpoint",
                f"Endpoint is reachable:\n{url}")
    elif resp.status_code in (401, 403):
        return ("warning", "Endpoint reached, but authentication failed", "orange", "Custom Image Edit Endpoint",
                f"Endpoint responded with authentication error ({resp.status_code}).")
    return ("information", f"Endpoint responded: HTTP {resp.status_code}", "orange", "Custom Image Edit Endpoint",
            f"Endpoint responded with HTTP {resp.status_code}.")


def test_custom_image_edit_endpoint(config, http_get=None):
    """Glossarion Mobile's Inpainting › custom image edit › Test: ``(ok, status_text, message)`` with the
    desktop texts (blank URL / reachable / auth error / HTTP status / "Test failed")."""
    config = config if isinstance(config, dict) else dict(config or {})
    url = str(config.get('custom_image_edit_endpoint') or '').strip()
    enabled = bool(config.get('use_custom_image_edit_endpoint', False))
    if not enabled or not url:
        return True, "Using current image provider/model", CUSTOM_IMAGE_EDIT_BLANK_MESSAGE
    try:
        url = normalize_custom_image_edit_url(url)
        _box, status, color, _title, message = probe_custom_image_edit_endpoint(url, config, http_get=http_get)
    except Exception as e:
        return False, "Custom Image Edit Endpoint test failed", f"Test failed:\n{e}"
    return color == "green", status, message


__all__ = [
    "CUSTOM_IMAGE_EDIT_BLANK_MESSAGE",
    "FONT_PRESETS",
    "HeadlessMangaState",
    "MangaEnvMixin",
    "MangaOcrSessionMixin",
    "MangaRunEnv",
    "MangaRunEnvError",
    "STARTUP_WIDGET_SOURCES",
    "apply_manga_startup_thread_limits",
    "apply_rendering_settings",
    "build_manga_run_env",
    "build_ocr_config",
    "default_custom_image_edit_system_prompt",
    "default_full_page_context_prompt",
    "default_manga_glossary_prompt",
    "default_manga_ocr_prompt",
    "font_preset_updates",
    "import_ocr_session",
    "normalize_custom_image_edit_url",
    "probe_custom_image_edit_endpoint",
    "rendering_reset_updates",
    "test_custom_image_edit_endpoint",
    "migrate_legacy_manga_ocr_prompt",
    "prepare_manga_glossary_env",
    "restore_manga_glossary_env",
]
