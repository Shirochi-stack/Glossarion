"""image_job: the desktop image / video input runner and the prompt-only generation run (ImageJobMixin).

Shared GUI-free core (Glossarion mobile rewrite, milestone U7). Both runners moved
verbatim out of ``TranslatorGUI`` (``translator_gui.py`` @ 41814faa). They are reached
from ``run_translation_direct`` (``translation_pipeline``), whose mixin inherits
``ImageJobMixin`` in place of the U3 placeholders, so ``TranslatorGUI`` and ``HeadlessOwner``
both run this code:

* ``_process_image_file(image_path, combined_output_dir=None)`` (23705-24851): one image or
  video input of a run (vision OCR / image edit / video / audio output modes alike: the
  client routes by the output-mode environment). The nested ``ImageProgressManager``
  (``translation_progress.json`` with ``images`` / ``content_hashes``), the per-purpose key
  pools, the prompt profile + appended glossary, the Direct Text prompt overrides,
  ``Payloads/image`` request/response files, ``UnifiedClient.send_image`` through
  ``TransateKRtoEN.send_with_interrupt``, generated media (``data:image/...;base64`` and
  ``[GENERATED_IMAGE:<path>]`` responses) and the translated-title HTML page;
* ``_run_generative_prompt_mode()`` (23578-23703): the one image / video / audio generation
  call of a run without an input file (``run_translation_direct``'s generative-only branch,
  reached through the ``GENERATIVE_MODE_SENTINEL`` selection).

Edits made while moving (everything else is byte-for-byte):

* ``_run_generative_prompt_mode`` reads the prompt editor through the
  ``_generative_prompt_source()`` hook; its default is the moved lines (the desktop's main
  prompt editor, ``prompt_text``) unless the owner attribute ``GENERATIVE_PROMPT_ATTR``
  (``_generative_prompt_override``) holds text: mobile sets the chat composer text there
  for "Generate from prompt" (UI_SPEC section 2.6). The desktop never sets it;
* its ``Generated_Media`` folder is ``mobile_runtime.data_dir(<this folder>)``: unchanged on
  desktop (``GLOSSARION_DATA_DIR`` unset), the writable app-data folder on mobile, where the
  bundled code folder is read-only. ``__file__`` is now image_job.py, which sits in the same
  folder as translator_gui.py (desktop builds ship both side by side).

Mobile composition (the job runs the desktop worker, which dispatches here)::

    owner._generative_prompt_override = prompt         # "Generate from prompt" only
    request = owner._prepare_translation_run([GENERATIVE_MODE_SENTINEL])  # or [image_path]
    owner._translation_worker(request)

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import os

from mobile_runtime import data_dir

__all__ = ["GENERATIVE_MODE_SENTINEL", "GENERATIVE_PROMPT_ATTR", "ImageJobMixin"]

#: ``selected_files`` entry of a run without an input file (``_prepare_translation_run`` sets it
#: for an image/video generation model; a job may pass it to reach the generative-only branch).
GENERATIVE_MODE_SENTINEL = "__generative_mode__"
#: Owner attribute holding the prompt of a generative-only run (mobile: the composer text).
GENERATIVE_PROMPT_ATTR = "_generative_prompt_override"


class ImageJobMixin:
    """Image / video inputs and the prompt-only generation run (moved verbatim; see the module docstring)."""

    def _generative_prompt_source(self):
        """Prompt-editor text the generative-only run starts from.

        Desktop: the main window's prompt editor (the moved lines). Mobile: the text in
        ``GENERATIVE_PROMPT_ATTR`` (the chat composer) when it holds any.
        """
        override = getattr(self, GENERATIVE_PROMPT_ATTR, None)
        if isinstance(override, str) and override.strip():
            return override.strip()
        system_prompt = ''
        try:
            system_prompt = self.prompt_text.toPlainText().strip()
        except Exception:
            pass
        return system_prompt

    def _run_generative_prompt_mode(self):
        """Run a single image/video generation call when no input file was provided.

        The user prompt is taken from:
          1. The 'translation_chunk_prompt' config value (the main prompt field), or
          2. The system prompt, or
          3. A sensible fallback.

        The result (image URL, video URL, or generated text) is logged to the
        GUI and also saved to a timestamped .txt file next to the executable.
        """
        try:
            model = str(getattr(self, 'model_var', '')).strip()
            self.append_log(f"🎨 Generative mode: sending prompt to {model}\u2026")

            # Build the user prompt from available config fields.
            # The system prompt lives in the prompt_text QTextEdit widget.
            system_prompt = self._generative_prompt_source()
            if not system_prompt:
                system_prompt = str(self.config.get('system_prompt', '') or '').strip()

            user_prompt = ''
            # 1. Prefer system prompt — it's the actual descriptive prompt the user wrote
            if system_prompt:
                user_prompt = system_prompt
            # 2. Try image-specific chunk prompt (only if not a template)
            if not user_prompt:
                val = getattr(self, 'image_chunk_prompt', '') or self.config.get('image_chunk_prompt', '')
                if val and val.strip() and '{chunk_html}' not in val and '{chunk_idx}' not in val:
                    user_prompt = val.strip()
            # 3. Try translation chunk prompt only if it's not a template
            if not user_prompt:
                val = getattr(self, 'translation_chunk_prompt', '') or self.config.get('translation_chunk_prompt', '')
                if val and val.strip() and '{chunk_html}' not in val and '{chunk_idx}' not in val:
                    user_prompt = val.strip()

            if not user_prompt:
                raise RuntimeError(
                    "No prompt found for generative mode. "
                    "Please enter a system prompt before running."
                )

            messages = []
            if system_prompt and system_prompt != user_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': user_prompt})

            self.append_log(f"📝 Prompt: {user_prompt[:200]}{'...' if len(user_prompt) > 200 else ''}")

            # Push critical output-mode env vars so the client picks them up
            os.environ['ENABLE_IMAGE_OUTPUT_MODE'] = self._get_allowed_image_output_mode()
            os.environ['ENABLE_VIDEO_OUTPUT_MODE'] = self._get_allowed_video_output_mode()
            os.environ['NANOGPT_VIDEO_DURATION'] = str(getattr(self, 'nanogpt_video_duration_var', '60')) + 's'
            os.environ['NANOGPT_VIDEO_RESOLUTION'] = str(getattr(self, 'nanogpt_video_resolution_var', '720p'))

            # Instantiate a UnifiedClient for this call
            from unified_api_client import UnifiedClient
            try:
                api_key = self.api_key_entry.text().strip()
            except Exception:
                api_key = str(getattr(self, 'api_key_var', '') or '').strip()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

            try:
                temperature = float(self.trans_temp.text())
            except Exception:
                temperature = 1.0

            client = UnifiedClient(
                api_key=api_key,
                model=model,
            )

            result_text, _finish = client.send(
                messages,
                temperature=temperature,
                max_tokens=int(getattr(self, 'max_output_tokens', 4096) or 4096),
            )
            result_text = (result_text or '').strip()

            self.append_log(f"\n\u2705 Generation complete!")
            self.append_log(f"🔗 Result: {result_text}")

            # Check if it's a generated media sentinel
            import re, shutil
            match = re.search(r'\[GENERATED_IMAGE:(.+?)\]', result_text)
            
            try:
                import datetime, pathlib
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_model = model.replace('/', '_').replace('\\', '_')

                # All generative output goes to Generated_Media
                out_dir = pathlib.Path(data_dir(os.path.dirname(os.path.abspath(__file__)))) / 'Generated_Media'
                out_dir.mkdir(parents=True, exist_ok=True)

                if match:
                    # It's a media file — already saved in Generated_Media, just log it
                    generated_media_path = match.group(1)
                    if os.path.exists(generated_media_path):
                        self.append_log(f"📄 Media saved to: {generated_media_path}")
                        result_text = generated_media_path
                else:
                    # Standard text response, save as .txt
                    fname = f"generated_{safe_model}_{ts}.txt"
                    out_path = out_dir / fname
                    out_path.write_text(result_text, encoding='utf-8')
                    self.append_log(f"📄 Saved to: {out_path}")
            except Exception as save_err:
                self.append_log(f"\u26a0\ufe0f Could not save result file: {save_err}")

            return True

        except Exception as exc:
            self.append_log(f"\u274c Generative mode error: {exc}")
            # Don't dump traceback for user-initiated cancellations / stops
            exc_lower = str(exc).lower()
            if not any(k in exc_lower for k in ('cancelled', 'canceled', 'graceful stop', 'stop requested')):
                import traceback
                self.append_log(traceback.format_exc())
            return False

    def _process_image_file(self, image_path, combined_output_dir=None):
        """Process a single image file using the direct image translation API with progress tracking"""
        try:
            import time
            import shutil
            import hashlib
            import os
            import json
            
            # Determine output directory early for progress tracking
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            if combined_output_dir:
                output_dir = combined_output_dir
            else:
                # Check for output directory override
                override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
                if override_dir:
                    output_dir = os.path.join(override_dir, base_name)
                else:
                    output_dir = base_name
            
            # Initialize progress manager if not already done
            if not hasattr(self, 'image_progress_manager'):
                # Use the determined output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Import or define a simplified ImageProgressManager
                class ImageProgressManager:
                    def __init__(self, output_dir=None):
                        self.output_dir = output_dir
                        if output_dir:
                            self.PROGRESS_FILE = os.path.join(output_dir, "translation_progress.json")
                            self.prog = self._init_or_load()
                        else:
                            self.PROGRESS_FILE = None
                            self.prog = {"images": {}, "content_hashes": {}, "version": "1.0"}
                    
                    def set_output_dir(self, output_dir):
                        """Set or update the output directory and load progress"""
                        self.output_dir = output_dir
                        self.PROGRESS_FILE = os.path.join(output_dir, "translation_progress.json")
                        self.prog = self._init_or_load()
                    
                    def _init_or_load(self):
                        """Initialize or load progress tracking"""
                        if os.path.exists(self.PROGRESS_FILE):
                            try:
                                with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                                    return json.load(pf)
                            except Exception as e:
                                if hasattr(self, 'append_log'):
                                    self.append_log(f"⚠️ Creating new progress file due to error: {e}")
                                return {"images": {}, "content_hashes": {}, "version": "1.0"}
                        else:
                            return {"images": {}, "content_hashes": {}, "version": "1.0"}
                    
                    def save(self):
                        """Save progress to file atomically with retry for file locks"""
                        if not self.PROGRESS_FILE:
                            return
                        try:
                            import time as _time
                            import threading as _threading
                            import uuid as _uuid
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(self.PROGRESS_FILE), exist_ok=True)
                            
                            temp_file = (
                                f"{self.PROGRESS_FILE}."
                                f"{os.getpid()}."
                                f"{_threading.get_ident()}."
                                f"{_uuid.uuid4().hex}.tmp"
                            )
                            max_retries = 5
                            for attempt in range(max_retries):
                                try:
                                    with open(temp_file, "w", encoding="utf-8") as pf:
                                        json.dump(self.prog, pf, ensure_ascii=False, indent=2)
                                    break
                                except PermissionError:
                                    if attempt < max_retries - 1:
                                        _time.sleep(0.1 * (2 ** attempt))
                                    else:
                                        raise
                            
                            for attempt in range(max_retries):
                                try:
                                    os.replace(temp_file, self.PROGRESS_FILE)
                                    break
                                except PermissionError:
                                    if attempt < max_retries - 1:
                                        _time.sleep(0.1 * (2 ** attempt))
                                    else:
                                        raise
                        except Exception as e:
                            if hasattr(self, 'append_log'):
                                self.append_log(f"⚠️ Failed to save progress: {e}")
                            else:
                                print(f"⚠️ Failed to save progress: {e}")
                    
                    def get_content_hash(self, file_path):
                        """Generate content hash for a file"""
                        hasher = hashlib.sha256()
                        with open(file_path, 'rb') as f:
                            # Read in chunks to handle large files
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
                        return hasher.hexdigest()
                    
                    def check_image_status(self, image_path, content_hash):
                        """Check if an image needs translation"""
                        image_name = os.path.basename(image_path)
                        
                        # NEW: Check for skip markers created by "Mark as Skipped" button
                        skip_key = f"skip_{image_name}"
                        if skip_key in self.prog:
                            skip_info = self.prog[skip_key]
                            if skip_info.get('status') == 'skipped':
                                return False, f"Image marked as skipped", None
                        
                        # NEW: Check if image already exists in images folder (marked as skipped)
                        if self.output_dir:
                            images_dir = os.path.join(self.output_dir, "images")
                            dest_image_path = os.path.join(images_dir, image_name)
                            
                            if os.path.exists(dest_image_path):
                                return False, f"Image in skipped folder", None
                        
                        # Check if image has already been processed
                        if content_hash in self.prog["images"]:
                            image_info = self.prog["images"][content_hash]
                            status = image_info.get("status")
                            output_file = image_info.get("output_file")
                            
                            if status == "completed" and output_file:
                                # Check if output file exists
                                if output_file and os.path.exists(output_file):
                                    return False, f"Image already translated: {output_file}", output_file
                                else:
                                    # Output file missing, mark for retranslation
                                    image_info["status"] = "file_deleted"
                                    image_info["deletion_detected"] = time.time()
                                    self.save()
                                    return True, None, None
                            
                            elif status == "skipped_cover":
                                return False, "Cover image - skipped", None
                            
                            elif status == "error":
                                # Previous error, retry
                                return True, None, None
                        
                        return True, None, None
                    
                    def update(self, image_path, content_hash, output_file=None, status="in_progress", error=None):
                        """Update progress for an image"""
                        image_name = os.path.basename(image_path)
                        
                        image_info = {
                            "name": image_name,
                            "path": image_path,
                            "content_hash": content_hash,
                            "status": status,
                            "last_updated": time.time()
                        }
                        
                        if output_file:
                            image_info["output_file"] = output_file
                        
                        if error:
                            image_info["error"] = str(error)
                        
                        self.prog["images"][content_hash] = image_info
                        
                        # Update content hash index for duplicates
                        if status == "completed" and output_file:
                            self.prog["content_hashes"][content_hash] = {
                                "original_name": image_name,
                                "output_file": output_file
                            }
                        
                        self.save()
                
                # Initialize the progress manager
                self.image_progress_manager = ImageProgressManager(output_dir)
                # Add append_log reference for the progress manager
                self.image_progress_manager.append_log = self.append_log
                self.append_log(f"📊 Progress tracking in: {os.path.join(output_dir, 'translation_progress.json')}")
            
            # Check for stop request early
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled by user")
                return False
            
            # Get content hash for the image
            try:
                content_hash = self.image_progress_manager.get_content_hash(image_path)
            except Exception as e:
                self.append_log(f"⚠️ Could not generate content hash: {e}")
                # Fallback to using file path as identifier
                content_hash = hashlib.sha256(image_path.encode()).hexdigest()
            
            # Check if image needs translation
            needs_translation, skip_reason, existing_output = self.image_progress_manager.check_image_status(
                image_path, content_hash
            )
            
            if not needs_translation:
                self.append_log(f"⏭️ {skip_reason}")
                
                # NEW: If image is marked as skipped but not in images folder yet, copy it there
                if "marked as skipped" in skip_reason and combined_output_dir:
                    images_dir = os.path.join(combined_output_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    dest_image = os.path.join(images_dir, image_name)
                    if not os.path.exists(dest_image):
                        shutil.copy2(image_path, dest_image)
                        self.append_log(f"📁 Copied skipped image to: {dest_image}")
                
                return True
            
            # Update progress to "in_progress"
            self.image_progress_manager.update(image_path, content_hash, status="in_progress")
            
            # Check if image translation is enabled (always allow for direct image/video file input)
            _direct_media_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg', '.ico', '.heic', '.heif', '.avif', '.jxl',
                                  '.mp4', '.mov', '.avi', '.mkv', '.webm'}
            _is_direct_media = os.path.splitext(image_path)[1].lower() in _direct_media_exts
            if not _is_direct_media and (not hasattr(self, 'enable_image_translation_var') or not self.enable_image_translation_var):
                self.append_log(f"⚠️ Image translation not enabled. Enable it in settings to translate images.")
                return False
            
            # Check for cover images
            if 'cover' in image_name.lower():
                self.append_log(f"⏭️ Skipping cover image: {image_name}")
                
                # Update progress for cover
                self.image_progress_manager.update(image_path, content_hash, status="skipped_cover")
                
                # Copy cover image to images folder if using combined output
                if combined_output_dir:
                    images_dir = os.path.join(combined_output_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    dest_image = os.path.join(images_dir, image_name)
                    if not os.path.exists(dest_image):
                        shutil.copy2(image_path, dest_image)
                        self.append_log(f"📁 Copied cover to: {dest_image}")
                
                return True  # Return True to indicate successful skip (not an error)
            
            # Check for stop before processing
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before processing")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Get the file index for numbering
            file_index = getattr(self, 'current_file_index', 0) + 1
            
            # Get API key and model
            api_key = self.api_key_entry.text().strip()
            model = self.model_var.strip()
            
            # Check if model needs API key (delegates to UnifiedClient's authoritative list)
            try:
                from unified_api_client import UnifiedClient as _UC
                model_needs_api_key = _UC._model_needs_api_key(model)
            except Exception:
                model_needs_api_key = bool(model)  # safe fallback
            
            if model_needs_api_key and not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                self.image_progress_manager.update(image_path, content_hash, status="error", error="No API key")
                return False
            
            if not model:
                self.append_log("❌ Error: Please select a model.")
                self.image_progress_manager.update(image_path, content_hash, status="error", error="No model selected")
                return False
            
            self.append_log(f"🖼️ Processing image: {os.path.basename(image_path)}")
            self.append_log(f"🤖 Using model: {model}")
            # Ensure image output mode/env reflect current settings before client call
            # Use _get_allowed_image_output_mode() so generative models (gpt-image-*, etc.)
            # auto-force ENABLE_IMAGE_OUTPUT_MODE=1 regardless of the toggle state.
            try:
                os.environ['ENABLE_IMAGE_OUTPUT_MODE'] = self._get_allowed_image_output_mode()
                os.environ['ENABLE_VIDEO_OUTPUT_MODE'] = self._get_allowed_video_output_mode()
                os.environ['IMAGE_OUTPUT_RESOLUTION'] = str(getattr(self, 'image_output_resolution_var', '1K')).upper()
                # Tell the video API where the source file is so it can probe metadata
                _video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
                if os.path.splitext(image_path)[1].lower() in _video_exts:
                    os.environ['NANOGPT_SOURCE_VIDEO_PATH'] = image_path
                else:
                    os.environ.pop('NANOGPT_SOURCE_VIDEO_PATH', None)
            except Exception:
                pass
            
            # Check if it's a vision-capable model
            vision_models = [
                'claude-opus-4-20250514', 'claude-sonnet-4-20250514',
                'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-5-mini','gpt-5','gpt-5-nano',
                'gpt-4-vision-preview',
                'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp',
                'gemini-2.5-pro', 'gemini-2.5-flash',
                'llama-3.2-11b-vision', 'llama-3.2-90b-vision',
                'gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview',
                'image-preview',  # catch-all for future image-generating Gemini models
                'eh/gemini-2.5-flash', 'eh/gemini-1.5-flash', 'eh/gpt-4o' # ElectronHub variants
            ]
            
            # Check for stop before API initialization
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before API initialization")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Apply multi-key settings for image translation (same as main translation path)
            try:
                use_mk = bool(self.config.get('use_multi_api_keys', False))
                mk_list = self.config.get('multi_api_keys', []) or []
                force_rotation = bool(self.config.get('force_key_rotation', True))
                rotation_frequency = int(self.config.get('rotation_frequency', 1))
                if use_mk and mk_list:
                    os.environ['USE_MULTI_API_KEYS'] = '1'
                    os.environ['USE_MULTI_KEYS'] = '1'
                    os.environ['FORCE_KEY_ROTATION'] = '1' if force_rotation else '0'
                    os.environ['ROTATION_FREQUENCY'] = str(rotation_frequency)
                    try:
                        from unified_api_client import UnifiedClient
                        UnifiedClient.set_in_memory_multi_keys(
                            mk_list,
                            force_rotation=force_rotation,
                            rotation_frequency=rotation_frequency,
                        )
                    except Exception:
                        pass
                    self.append_log(f"🔑 Multi-key mode ENABLED for image translation ({len(mk_list)} keys)")
                else:
                    os.environ['USE_MULTI_API_KEYS'] = '0'
                    os.environ['USE_MULTI_KEYS'] = '0'
                    try:
                        from unified_api_client import UnifiedClient
                        UnifiedClient.clear_in_memory_multi_keys()
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Configure glossary key pool in memory (mirrors multi-key setup)
            try:
                from unified_api_client import UnifiedClient
                if self.config.get('use_glossary_keys', False) and self.config.get('glossary_keys', []):
                    UnifiedClient.set_in_memory_glossary_keys(
                        self.config.get('glossary_keys', []),
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                else:
                    UnifiedClient.clear_in_memory_glossary_keys()
                refinement_keys_enabled = bool(self.config.get('use_glossary_refinement_keys', False))
                refinement_keys = self.config.get('glossary_refinement_keys', []) or []
                os.environ['USE_GLOSSARY_REFINEMENT_KEYS'] = '1' if refinement_keys_enabled else '0'
                os.environ['GLOSSARY_REFINEMENT_API_KEYS'] = json.dumps(refinement_keys)
                if refinement_keys_enabled and refinement_keys:
                    UnifiedClient.set_in_memory_glossary_refinement_keys(
                        refinement_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                else:
                    UnifiedClient.clear_in_memory_glossary_refinement_keys()
            except Exception:
                pass

            # Configure rolling summary key pool for memory summary generation calls.
            try:
                from unified_api_client import UnifiedClient
                rolling_summary_keys_enabled = bool(self.config.get('use_rolling_summary_keys', False))
                rolling_summary_keys = self.config.get('rolling_summary_keys', []) or []
                os.environ['USE_ROLLING_SUMMARY_KEYS'] = '1' if rolling_summary_keys_enabled else '0'
                os.environ['ROLLING_SUMMARY_API_KEYS'] = json.dumps(rolling_summary_keys)
                if rolling_summary_keys_enabled and rolling_summary_keys:
                    UnifiedClient.set_in_memory_rolling_summary_keys(
                        rolling_summary_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    self.append_log(f"[RollingSummary] Key pool ENABLED for rolling summary generation ({len(rolling_summary_keys)} keys)")
                else:
                    UnifiedClient.clear_in_memory_rolling_summary_keys()
                    if rolling_summary_keys_enabled:
                        self.append_log("[RollingSummary] Enabled but no keys configured")
            except Exception:
                pass

            # Configure truncation retry key pool for RETRY_TRUNCATED attempts.
            try:
                from unified_api_client import UnifiedClient
                truncation_retry_keys_enabled = bool(self.config.get('use_truncation_retry_keys', False))
                truncation_retry_keys = self.config.get('truncation_retry_keys', []) or []
                os.environ['USE_TRUNCATION_RETRY_KEYS'] = '1' if truncation_retry_keys_enabled else '0'
                os.environ['TRUNCATION_RETRY_API_KEYS'] = json.dumps(truncation_retry_keys)
                if truncation_retry_keys_enabled and truncation_retry_keys:
                    UnifiedClient.set_in_memory_truncation_retry_keys(
                        truncation_retry_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    self.append_log(f"[TruncationRetry] Key pool ENABLED for truncation retries ({len(truncation_retry_keys)} keys)")
                else:
                    UnifiedClient.clear_in_memory_truncation_retry_keys()
                    if truncation_retry_keys_enabled:
                        self.append_log("[TruncationRetry] Enabled but no keys configured")
            except Exception:
                pass

            # Configure Image Gen/Edit key pool for image output mode requests.
            try:
                from unified_api_client import UnifiedClient
                inpainter_keys_enabled = bool(self.config.get('use_inpainter_keys', False))
                inpainter_keys = self.config.get('inpainter_keys', []) or []
                os.environ['USE_INPAINTER_KEYS'] = '1' if inpainter_keys_enabled else '0'
                os.environ['INPAINTER_API_KEYS'] = json.dumps(inpainter_keys)
                if inpainter_keys_enabled and inpainter_keys:
                    UnifiedClient.set_in_memory_inpainter_keys(
                        inpainter_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    self.append_log(f"[ImageGenEdit] Key pool ENABLED for image output ({len(inpainter_keys)} keys)")
                else:
                    UnifiedClient.clear_in_memory_inpainter_keys()
                    if inpainter_keys_enabled:
                        self.append_log("[ImageGenEdit] Enabled but no keys configured")
            except Exception:
                pass

            # Configure Audio / TTS key pool for audio output mode (context "tts").
            try:
                from unified_api_client import UnifiedClient
                tts_keys_enabled = bool(self.config.get('use_tts_keys', False))
                tts_keys = self.config.get('tts_keys', []) or []
                os.environ['USE_TTS_KEYS'] = '1' if tts_keys_enabled else '0'
                os.environ['TTS_API_KEYS'] = json.dumps(tts_keys)
                if tts_keys_enabled and tts_keys:
                    UnifiedClient.set_in_memory_tts_keys(
                        tts_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    self.append_log(f"[AudioTTS] Key pool ENABLED for audio output ({len(tts_keys)} keys)")
                else:
                    UnifiedClient.clear_in_memory_tts_keys()
                    if tts_keys_enabled:
                        self.append_log("[AudioTTS] Enabled but no keys configured")
            except Exception:
                pass
            
            # Initialize API client with output_dir to enable multi-key mode from environment
            try:
                from unified_api_client import UnifiedClient
                # Pass output_dir to enable environment-based multi-key initialization
                client = UnifiedClient(model=model, api_key=api_key, output_dir=output_dir)
                
                # Set stop flag if the client supports it
                if hasattr(client, 'set_stop_flag'):
                    client.set_stop_flag(self.stop_requested)
                elif hasattr(client, 'stop_flag'):
                    client.stop_flag = self.stop_requested
                    
            except Exception as e:
                self.append_log(f"❌ Failed to initialize API client: {str(e)}")
                self.image_progress_manager.update(image_path, content_hash, status="error", error=f"API client init failed: {e}")
                return False
            
            # Read the image
            try:
                # Get image name for payload naming
                base_name = os.path.splitext(image_name)[0]
                
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Convert to base64
                import base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Check image size
                size_mb = len(image_data) / (1024 * 1024)
                self.append_log(f"📊 Image size: {size_mb:.2f} MB")
                
            except Exception as e:
                self.append_log(f"❌ Failed to read image: {str(e)}")
                self.image_progress_manager.update(image_path, content_hash, status="error", error=f"Failed to read image: {e}")
                return False
            
            # Get system prompt from configuration
            profile_name = self.config.get('active_profile', 'Korean_BeautifulSoup')
            prompt_profiles = self.config.get('prompt_profiles', {})
            
            # Get the main translation prompt
            system_prompt = ""
            if isinstance(prompt_profiles, dict) and profile_name in prompt_profiles:
                profile_data = prompt_profiles[profile_name]
                if isinstance(profile_data, str):
                    # Old format: prompt_profiles[profile_name] = "prompt text"
                    system_prompt = profile_data
                elif isinstance(profile_data, dict):
                    # New format: prompt_profiles[profile_name] = {"prompt": "...", "book_title_prompt": "..."}
                    system_prompt = profile_data.get('prompt', '')
            else:
                # Fallback to check if prompt is stored directly in config
                system_prompt = self.config.get(profile_name, '')
            
            if not system_prompt:
                # Last fallback - empty string
                system_prompt = ""

            if getattr(self, '_input_output_run_active', False):
                profile_as_user = bool(
                    getattr(self, 'system_prompt_to_user_var', False)
                )
                skip_prompt_profile = bool(
                    getattr(
                        self,
                        '_direct_text_skip_prompt_profile',
                        False,
                    )
                )
                if profile_as_user or skip_prompt_profile:
                    system_prompt = ""

            # Replace split marker instruction placeholder
            split_instr = ""
            if getattr(self, 'request_merging_enabled_var', False):
                split_instr = "- CRITICAL Requirement: If you see any HTML tags containing 'SPLIT MARKER' (Example: <h1 id=\"split-1\">SPLIT MARKER: Do Not Remove This Tag</h1>), you MUST preserve them EXACTLY as they appear. Do not translate, modify, or remove these markers."
            
            # Use a regex to replace the placeholder regardless of surrounding whitespace or newlines
            import re
            system_prompt = re.sub(r'\s*\{split_marker_instruction\}\s*', lambda m: '\n' + split_instr if split_instr else '', system_prompt)

            # Check if we should append glossary to the prompt
            append_glossary = self.config.get('append_glossary', True)  # Default to True
            if hasattr(self, 'append_glossary_var'):
                append_glossary = self.append_glossary_var
            
            # Check if automatic glossary is enabled (only minimal mode uses the subprocess path)
            auto_glossary_mode = self.config.get('auto_glossary_mode', None)
            if auto_glossary_mode is None:
                auto_glossary_mode = 'minimal' if self.config.get('enable_auto_glossary', False) else 'off'
            enable_auto_glossary_subprocess = (auto_glossary_mode == 'minimal')
            
            # "No Glossary" mode overrides append_glossary for all file types
            if auto_glossary_mode == 'no_glossary':
                append_glossary = False
            
            if append_glossary:
                # Check for manual glossary
                manual_glossary_path = os.getenv('MANUAL_GLOSSARY')
                if not manual_glossary_path and hasattr(self, 'manual_glossary_path'):
                    manual_glossary_path = self.manual_glossary_path
                # If the file exists but is empty, treat it as missing so auto generation can run
                if manual_glossary_path and os.path.exists(manual_glossary_path):
                    try:
                        if os.path.getsize(manual_glossary_path) == 0:
                            self.append_log("ℹ️ Glossary file is empty; treating as missing so automatic glossary generation can run.")
                            manual_glossary_path = None
                            # Clear env/config so workers don't think a glossary exists
                            os.environ.pop('MANUAL_GLOSSARY', None)
                            if hasattr(self, 'manual_glossary_path'):
                                self.manual_glossary_path = ''
                            self.config['manual_glossary_path'] = ''
                    except Exception:
                        pass
                
                # If minimal mode automatic glossary is enabled and no manual glossary exists, defer appending
                if enable_auto_glossary_subprocess and (not manual_glossary_path or not os.path.exists(manual_glossary_path)):
                    self.append_log(f"📑 Automatic glossary enabled (minimal mode) - glossary will be appended after generation")
                    # Set a flag to indicate deferred glossary appending
                    os.environ['DEFER_GLOSSARY_APPEND'] = '1'
                    # Store the append prompt for later use
                    glossary_prompt = self.config.get('append_glossary_prompt', 
                        "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                    os.environ['GLOSSARY_APPEND_PROMPT'] = glossary_prompt
                else:
                    # Original behavior - append manual glossary immediately
                    if manual_glossary_path and os.path.exists(manual_glossary_path):
                        try:
                            self.append_log(f"📑 Loading glossary for system prompt: {os.path.basename(manual_glossary_path)}")
                            
                            # Copy to output as the same extension, and prefer CSV naming
                            ext = os.path.splitext(manual_glossary_path)[1].lower()
                            out_name = "glossary.csv" if ext == ".csv" else "glossary.json"
                            output_glossary_path = os.path.join(output_dir, out_name)
                            try:
                                import shutil as _shutil
                                _shutil.copy(manual_glossary_path, output_glossary_path)
                                self.append_log(f"💾 Saved glossary to output folder for auto-loading: {out_name}")
                            except Exception as copy_err:
                                self.append_log(f"⚠️ Could not copy glossary into output: {copy_err}")
                            
                            # Append to prompt
                            if ext == ".csv":
                                with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                                    csv_text = f.read()
                                if system_prompt:
                                    system_prompt += "\n\n"
                                glossary_prompt = self.config.get('append_glossary_prompt', 
                                    "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                                system_prompt += f"{glossary_prompt}\n{csv_text}"
                                self.append_log(f"✅ Appended CSV glossary to system prompt")
                            else:
                                with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                                    glossary_data = json.load(f)
                                
                                formatted_entries = {}
                                if isinstance(glossary_data, list):
                                    for char in glossary_data:
                                        if not isinstance(char, dict):
                                            continue
                                        original = char.get('original_name', '')
                                        translated = char.get('name', original)
                                        if original and translated:
                                            formatted_entries[original] = translated
                                        title = char.get('title')
                                        if title and original:
                                            formatted_entries[f"{original} ({title})"] = f"{translated} ({title})"
                                        refer_map = char.get('how_they_refer_to_others', {})
                                        if isinstance(refer_map, dict):
                                            for other_name, reference in refer_map.items():
                                                if other_name and reference:
                                                    formatted_entries[f"{original} → {other_name}"] = f"{translated} → {reference}"
                                elif isinstance(glossary_data, dict):
                                    if "entries" in glossary_data and isinstance(glossary_data["entries"], dict):
                                        formatted_entries = glossary_data["entries"]
                                    else:
                                        formatted_entries = {k: v for k, v in glossary_data.items() if k != "metadata"}
                                if formatted_entries:
                                    glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                                    if system_prompt:
                                        system_prompt += "\n\n"
                                    glossary_prompt = self.config.get('append_glossary_prompt', 
                                        "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                                    system_prompt += f"{glossary_prompt}\n{glossary_block}"
                                    self.append_log(f"✅ Added {len(formatted_entries)} glossary entries to system prompt")
                                else:
                                    self.append_log(f"⚠️ Glossary file has no valid entries")
                                
                        except Exception as e:
                            self.append_log(f"⚠️ Failed to append glossary to prompt: {str(e)}")
                    else:
                        self.append_log(f"ℹ️ No glossary file found to append to prompt")
            else:
                self.append_log(f"ℹ️ Glossary appending disabled in settings")
                # Clear any deferred append flag
                if 'DEFER_GLOSSARY_APPEND' in os.environ:
                    del os.environ['DEFER_GLOSSARY_APPEND']
            
            # Get temperature and max tokens from GUI
            temperature = float(self.trans_temp.text()) if hasattr(self, 'trans_temp') else 0.3
            max_tokens = self.max_output_tokens
            
            # Initialize history manager for contextual translation if enabled
            history_manager = None
            # Check both checkbox state (runtime) and config variable (initial)
            contextual_enabled = bool(getattr(self, 'contextual_var', False))
            if contextual_enabled:
                try:
                    from history_manager import HistoryManager
                    history_manager = HistoryManager(output_dir)
                    
                    # Add previous context to messages if available
                    history_limit = int(self.trans_history.text()) if hasattr(self, 'trans_history') else 3
                    context_messages = history_manager.load_history()
                    
                    # Limit to the most recent exchanges
                    if context_messages:
                        # Each exchange is 2 messages (user + assistant)
                        messages_to_keep = history_limit * 2
                        context_messages = context_messages[-messages_to_keep:] if len(context_messages) > messages_to_keep else context_messages
                    
                    # Build messages with history. A skipped profile is omitted
                    # entirely; do not send an empty system-role placeholder.
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.extend(context_messages)
                    
                    if context_messages:
                        num_exchanges = len(context_messages) // 2
                        self.append_log(f"📚 Using {num_exchanges} previous image(s) for context")
                except Exception as e:
                    self.append_log(f"⚠️ Failed to initialize history manager: {e}")
                    import traceback
                    self.append_log(traceback.format_exc())
                    messages = (
                        [{"role": "system", "content": system_prompt}]
                        if system_prompt else []
                    )
            else:
                # Build messages for vision API without history
                messages = (
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt else []
                )

            if getattr(self, '_input_output_run_active', False):
                try:
                    from TransateKRtoEN import (
                        _apply_direct_text_prompt_overrides_to_messages,
                    )
                    messages = _apply_direct_text_prompt_overrides_to_messages(
                        messages
                    )
                except Exception as prompt_error:
                    self.append_log(
                        f"⚠️ Could not apply Direct Text attachment prompt: "
                        f"{prompt_error}"
                    )
            
            self.append_log(f"🌐 Sending image to vision API...")
            self.append_log(f"   System prompt length: {len(system_prompt)} chars")
            self.append_log(f"   Temperature: {temperature}")
            self.append_log(f"   Max tokens: {max_tokens}")          
            
            # Debug: Show first 200 chars of system prompt
            if system_prompt:
                preview = system_prompt[:] if len(system_prompt) > 200 else system_prompt
                prompt_type = "User prompt" if os.environ.get('SYSTEM_PROMPT_TO_USER', '0') == '1' else "System prompt"
                self.append_log(f"   {prompt_type}: {preview}")
            
            # Check stop before making API call
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before API call")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Make the API call
            try:
                # Create Payloads/image directory for API response tracking
                # Route to the image subfolder to match unified_api_client routing
                payloads_dir = os.path.join("Payloads", "image")
                try:
                    os.makedirs(payloads_dir, exist_ok=True)
                except (PermissionError, OSError):
                    # Fall back to temp directory if CWD is not writable
                    import tempfile
                    payloads_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "image")
                    try:
                        os.makedirs(payloads_dir, exist_ok=True)
                    except Exception:
                        payloads_dir = None  # Skip payload saving entirely
                
                # Create timestamp for unique filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if payloads_dir:
                    payload_file = os.path.join(payloads_dir, f"image_api_{timestamp}_{base_name}.json")
                
                    # Save the request payload
                    request_payload = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "image_file": image_name,
                        "image_size_mb": size_mb,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "image_base64": image_base64  # Full payload without truncation
                    }
                
                    with open(payload_file, 'w', encoding='utf-8') as f:
                        json.dump(request_payload, f, ensure_ascii=False, indent=2)
                
                    self.append_log(f"📝 Saved request payload: {payload_file}")
                
                # Call the vision API with interrupt support
                # Check if the client supports a stop_callback parameter
                # Import the send_with_interrupt function from TransateKRtoEN
                try:
                    from TransateKRtoEN import send_with_interrupt
                except ImportError:
                    self.append_log("⚠️ send_with_interrupt not available, using direct call")
                    send_with_interrupt = None
                
                # Initialize raw_obj to None for both paths
                raw_obj = None
                
                # Call the vision API with interrupt support
                if send_with_interrupt:
                    # For image calls, we need a wrapper since send_with_interrupt expects client.send()
                    # Create a temporary wrapper client that handles image calls
                    class ImageClientWrapper:
                        def __init__(self, real_client, image_data, response_name):
                            self.real_client = real_client
                            self.image_data = image_data
                            self.response_name = response_name
                        
                        def send(self, messages, temperature, max_tokens):
                            return self.real_client.send_image(messages, self.image_data, temperature=temperature, max_tokens=max_tokens, response_name=self.response_name if hasattr(self, 'response_name') else None)
                        
                        def __getattr__(self, name):
                            return getattr(self.real_client, name)
                    
                    # Create wrapped client with source filename
                    wrapped_client = ImageClientWrapper(client, image_base64, image_name)
                    
                    # Use send_with_interrupt
                    response, finish_reason_from_send, raw_obj = send_with_interrupt(
                        messages,
                        wrapped_client,
                        temperature,
                        max_tokens,
                        lambda: self.stop_requested,
                        chunk_timeout=self.config.get('chunk_timeout', 1200)  # 20 min default
                    )
                else:
                    # Fallback to direct call
                    response = client.send_image(
                        messages,
                        image_base64,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_name=image_name
                    )
                    # Try to get raw_obj from response if available
                    if hasattr(response, 'raw_content_object'):
                        raw_obj = response.raw_content_object
                
                # Check if stopped after API call
                if self.stop_requested:
                    self.append_log("⏹️ Image translation stopped after API call")
                    self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                    return False
                
                # Extract content and finish reason from response
                response_content = None
                finish_reason = None
                
                if hasattr(response, 'content'):
                    response_content = response.content
                    finish_reason = response.finish_reason if hasattr(response, 'finish_reason') else 'unknown'
                elif isinstance(response, tuple) and len(response) >= 2:
                    # Handle tuple response (content, finish_reason)
                    response_content, finish_reason = response
                elif isinstance(response, str):
                    # Handle direct string response
                    response_content = response
                    finish_reason = 'complete'
                else:
                    self.append_log(f"❌ Unexpected response type: {type(response)}")
                    self.append_log(f"   Response: {response}")
                
                # Save the response payload
                response_payload = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_content": response_content,
                    "finish_reason": finish_reason,
                    "content_length": len(response_content) if response_content else 0
                }
                
                if payloads_dir:
                    response_file = os.path.join(payloads_dir, f"image_api_response_{timestamp}_{base_name}.json")
                    with open(response_file, 'w', encoding='utf-8') as f:
                        json.dump(response_payload, f, ensure_ascii=False, indent=2)
                
                    self.append_log(f"📝 Saved response payload: {response_file}")
                
                # Check if we got valid content
                if not response_content or response_content.strip() == "[IMAGE TRANSLATION FAILED]":
                    self.append_log(f"❌ Image translation failed - no text extracted from image")
                    self.append_log(f"   This may mean:")
                    self.append_log(f"   - The image doesn't contain readable text")
                    self.append_log(f"   - The model couldn't process the image")
                    self.append_log(f"   - The image format is not supported")
                    
                    # Try to get more info about the failure
                    if hasattr(response, 'error_details'):
                        self.append_log(f"   Error details: {response.error_details}")
                    
                    self.image_progress_manager.update(image_path, content_hash, status="error", error="No text extracted")
                    return False
                
                if response_content:
                    self.append_log(f"✅ Received translation from API")
                    
                    # ── Decode data:image/...;base64, responses (gpt-image-2, etc.) ──────
                    if response_content.startswith("data:image/"):
                        try:
                            # Parse   data:<mime>;base64,<data>
                            header, b64data = response_content.split(",", 1)
                            # Determine extension from mime type
                            mime = header.split(";")[0].split(":")[1].lower()  # e.g. image/png
                            ext  = mime.split("/")[-1]  # png / jpeg / webp …
                            if ext == "jpeg":
                                ext = "jpg"

                            # Build output path next to other translated images
                            os.makedirs(output_dir, exist_ok=True)
                            base_name = os.path.splitext(image_name)[0]
                            generated_image_path = os.path.join(output_dir, f"{base_name}_generated.{ext}")

                            import base64 as _b64
                            with open(generated_image_path, "wb") as _f:
                                _f.write(_b64.b64decode(b64data))

                            self.append_log(f"✅ Generated image decoded and saved: {os.path.basename(generated_image_path)}")

                            # Track for CBZ compilation
                            if not hasattr(self, 'generated_images'):
                                self.generated_images = []
                            self.generated_images.append(generated_image_path)

                            self.image_progress_manager.update(image_path, content_hash, output_file=generated_image_path, status="completed")

                            self.append_log(f"💾 Image saved to: {generated_image_path}")
                            self.append_log(f"📁 Output directory: {output_dir}")
                            return True
                        except Exception as _dec_err:
                            self.append_log(f"⚠️ Failed to decode base64 image: {_dec_err} — saving as text")
                            # Fall through to normal text handling below

                    # Check if this is a generated image response (Gemini-style sentinel)
                    if response_content.startswith("[GENERATED_IMAGE:"):

                        # Extract the image path
                        import re
                        match = re.search(r'\[GENERATED_IMAGE:(.+?)\]', response_content)
                        if match:
                            generated_image_path = match.group(1)
                            if os.path.exists(generated_image_path):
                                # Move the generated file to the actual output folder with the proper name
                                import shutil
                                _, ext = os.path.splitext(generated_image_path)
                                final_media_name = f"response_{file_index:03d}_{base_name}{ext}"
                                final_media_path = os.path.join(output_dir, final_media_name)
                                os.makedirs(output_dir, exist_ok=True)
                                shutil.move(generated_image_path, final_media_path)
                                generated_image_path = final_media_path
                                
                                self.append_log(f"✅ Generated media saved directly as: {final_media_name}")
                                
                                # Track this image for CBZ compilation
                                if not hasattr(self, 'generated_images'):
                                    self.generated_images = []
                                self.generated_images.append(generated_image_path)
                                
                                # Update progress as completed (no HTML file needed)
                                self.image_progress_manager.update(image_path, content_hash, output_file=generated_image_path, status="completed")
                                
                                # Save to history manager if contextual translation is enabled
                                if history_manager:
                                    try:
                                        # Use raw_obj from send_with_interrupt or response.raw_content_object
                                        thought_signature = raw_obj if raw_obj else None
                                        
                                        # Use microsecond lock for thread safety
                                        import threading
                                        if not hasattr(self, '_history_lock'):
                                            self._history_lock = threading.Lock()
                                        
                                        with self._history_lock:
                                            # For generated images, save only the assistant message with image
                                            # No user message needed - just the generated image as context
                                            history_limit = int(self.trans_history.text()) if hasattr(self, 'trans_history') else 3
                                            rolling_history = True
                                            
                                            # Store structured payload for image exchange
                                            assistant_payload = {
                                                "type": "image_exchange",
                                                "version": 1,
                                                "image_path": generated_image_path,
                                                "image_name": os.path.basename(generated_image_path)
                                            }
                                            
                                            history_manager.append_to_history(
                                                user_content="",  # Empty - no need to send source image name to API
                                                assistant_content=assistant_payload,
                                                hist_limit=history_limit,
                                                reset_on_limit=not rolling_history,
                                                rolling_window=rolling_history,
                                                raw_assistant_object=thought_signature
                                            )
                                        self.append_log(f"📚 Saved to translation history with image context")
                                    except Exception as e:
                                        self.append_log(f"⚠️ Failed to save to history: {e}")
                                        import traceback
                                        self.append_log(traceback.format_exc())
                                
                                # Skip HTML generation for generated images
                                self.append_log(f"💾 Image saved to: {generated_image_path}")
                                self.append_log(f"📁 Output directory: {output_dir}")
                                return True
                    
                    # We already have output_dir defined at the top
                    # Copy original image to the output directory if not using combined output
                    if not combined_output_dir and not os.path.exists(os.path.join(output_dir, image_name)):
                        os.makedirs(output_dir, exist_ok=True)
                        shutil.copy2(image_path, os.path.join(output_dir, image_name))
                    
                    # Get book title prompt for translating the filename
                    book_title_prompt = self.config.get('book_title_prompt', '')
                    book_title_system_prompt = self.config.get('book_title_system_prompt', '')
                    
                    # If no book title prompt in main config, check in profile
                    if not book_title_prompt and isinstance(prompt_profiles, dict) and profile_name in prompt_profiles:
                        profile_data = prompt_profiles[profile_name]
                        if isinstance(profile_data, dict):
                            book_title_prompt = profile_data.get('book_title_prompt', '')
                            # Also check for system prompt in profile
                            if 'book_title_system_prompt' in profile_data:
                                book_title_system_prompt = profile_data['book_title_system_prompt']
                    
                    # If still no book title prompt, use the main system prompt
                    if not book_title_prompt:
                        book_title_prompt = system_prompt
                    
                    # If no book title system prompt configured, use the main system prompt
                    if not book_title_system_prompt:
                        book_title_system_prompt = system_prompt
                    
                    # Translate the image filename/title (unless skipped)
                    skip_img_title = bool(getattr(self, 'skip_image_title_translation_var', True) or self.config.get('skip_image_title_translation', True) or os.environ.get('SKIP_IMAGE_TITLE_TRANSLATION') == '1')
                    if skip_img_title:
                        self.append_log("⏭️ Skipping image title translation (setting enabled)")
                        translated_title = base_name
                    else:
                        self.append_log(f"📝 Translating image title...")
                    
                    # Replace {target_lang} variable in both system and user prompts with output language
                    output_lang = self.config.get('output_language', 'English')
                    book_title_system_prompt_formatted = book_title_system_prompt.replace('{target_lang}', output_lang)
                    book_title_prompt_formatted = book_title_prompt.replace('{target_lang}', output_lang)
                    
                    title_messages = [
                        {"role": "system", "content": book_title_system_prompt_formatted},
                        {"role": "user", "content": f"{book_title_prompt_formatted}\n\n{base_name}" if book_title_prompt != system_prompt else base_name}
                    ]
                    
                    if not skip_img_title:
                        try:
                            # Check for stop before title translation
                            graceful_stop_active = os.environ.get('GRACEFUL_STOP') == '1'
                            if self.stop_requested or graceful_stop_active:
                                self.append_log("⏹️ Image translation cancelled before title translation")
                                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                                return False
                            
                            title_response = client.send(
                                title_messages,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            
                            # Extract title translation
                            if hasattr(title_response, 'content'):
                                translated_title = title_response.content.strip() if title_response.content else base_name
                            else:
                                # Handle tuple response
                                title_content, *_ = title_response
                                translated_title = title_content.strip() if title_content else base_name
                        except Exception as e:
                            # If stop/graceful stop toggled during title call, don't treat as failure
                            if self.stop_requested or os.environ.get('GRACEFUL_STOP') == '1' or "cancelled" in str(e).lower():
                                self.append_log("⏹️ Image title translation cancelled")
                                translated_title = base_name
                            else:
                                self.append_log(f"⚠️ Title translation failed: {str(e)}")
                                translated_title = base_name  # Fallback to original if translation fails
                    
                    # Create clean HTML content with just the translated title and content
                    html_content = f'''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>{translated_title}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 40px;
                max-width: 800px;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{translated_title}</h1>
        {response_content}
    </body>
    </html>'''
                    
                    # Save HTML file with proper numbering
                    html_file = os.path.join(output_dir, f"response_{file_index:03d}_{base_name}.html")
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    # Copy original image to the output directory (for reference, not displayed)
                    if not combined_output_dir:
                        shutil.copy2(image_path, os.path.join(output_dir, image_name))
                    
                    # Update progress to completed
                    self.image_progress_manager.update(image_path, content_hash, output_file=html_file, status="completed")
                    
                    # Show preview
                    if response_content and response_content.strip():
                        preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                        self.append_log(f"📝 Translation preview:")
                        self.append_log(f"{preview}")
                    else:
                        self.append_log(f"⚠️ Translation appears to be empty")
                    
                    self.append_log(f"✅ Translation saved to: {html_file}")
                    self.append_log(f"📁 Output directory: {output_dir}")
                    
                    return True
                else:
                    self.append_log(f"❌ No translation received from API")
                    if finish_reason:
                        self.append_log(f"   Finish reason: {finish_reason}")
                    self.image_progress_manager.update(image_path, content_hash, status="error", error="No response from API")
                    return False
                    
            except Exception as e:
                # Check if this was a stop/interrupt exception
                if "stop" in str(e).lower() or "interrupt" in str(e).lower() or self.stop_requested:
                    self.append_log("⏹️ Image translation interrupted")
                    self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                    return False
                else:
                    self.append_log(f"❌ API call failed: {str(e)}")
                    import traceback
                    self.append_log(f"❌ Full error: {traceback.format_exc()}")
                    self.image_progress_manager.update(image_path, content_hash, status="error", error=f"API call failed: {e}")
                    return False
            
        except Exception as e:
            self.append_log(f"❌ Error processing image: {str(e)}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False
