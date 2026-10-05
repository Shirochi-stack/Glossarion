"""text_jobs: the desktop's per-file translation, glossary, metadata and compile runners (TextJobsMixin).

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). These methods moved
verbatim out of ``TranslatorGUI`` (``translator_gui.py`` @ 1719fb59), which now
inherits them (TextJobsMixin is its first base); ``HeadlessOwner`` runs the same code
on mobile:

* ``_process_text_file`` (32196-32541): one EPUB/TXT/PDF/SDLXLIFF/subtitle file through
  ``TransateKRtoEN.main``;
* ``_extract_glossary_from_text_file`` (33823-34064): one source through
  ``extract_glossary_from_epub.main``;
* ``_run_parallel_metadata_files`` (32011-32194): metadata-only batch over EPUBs;
* ``_run_epub_compile`` / ``_run_pdf_compile``: the try/except bodies of
  ``run_epub_converter_direct`` (34225-34278) / ``run_pdf_converter_direct``
  (34106-34162); the desktop runners keep their thread/stop/button reset and call these.

Edits made while moving (everything else is byte-for-byte):

* the inline argv/environment snapshot + restore is ``job_runner.scoped_process_state``
  (``snapshot()`` where the old code took its copies, ``restore()`` in the same
  ``finally``; same restore order: argv, ``os.environ.clear()`` + ``update``, then
  ``large_env.clear_store()`` for translation);
* the module globals ``translation_main`` / ``glossary_main`` / ``fallback_compile_epub``
  (lazily loaded by the desktop) are read through the ``_backend_entry`` hook at the
  same point; message boxes are the ``_notify_compile_result`` hook;
* the glossary stop callback is ``stop_control.make_glossary_stop_callback``;
* ``is_traditional_translation_api`` (translator_gui 1561) lives here; translator_gui
  re-imports it.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import concurrent.futures
import json
import os
import sys
from dataclasses import dataclass

from app_paths import CONFIG_FILE
from job_runner import JobHooksMixin, scoped_process_state
from stop_control import make_glossary_stop_callback

__all__ = ["CompileResult", "TextJobsMixin", "is_traditional_translation_api"]


def is_traditional_translation_api(model: str) -> bool:
    """Check if the model is a traditional translation API"""
    return model in ['deepl', 'google-translate', 'google-translate-free'] or model.startswith('deepl/') or model.startswith('google-translate/')


@dataclass
class CompileResult:
    """Outcome of ``_run_epub_compile`` / ``_run_pdf_compile``.

    ``path``: the compiled file the desktop announces (None when nothing was
    announced); ``error``: the error text of a failed compile; ``stopped``: Stop was
    requested before the EPUB compile finished.
    """

    kind: str
    path: str = None
    error: str = None
    stopped: bool = False

    @property
    def ok(self):
        return self.error is None and not self.stopped


class TextJobsMixin(JobHooksMixin):
    """Per-file job runners shared by TranslatorGUI and HeadlessOwner (moved verbatim)."""

    def _run_parallel_metadata_files(self, files: list[str]) -> bool:
        """Translate multiple EPUB metadata sets in an in-process thread pool."""
        from metadata_translation_worker import run_metadata_translation_job

        files = [
            os.path.abspath(path) for path in files
            if path and path.lower().endswith('.epub') and os.path.isfile(path)
        ]
        if not files:
            self.append_log("❌ No valid EPUBs were available for metadata translation")
            return False

        api_key = self.api_key_entry.text()
        model = self.model_var
        try:
            from unified_api_client import UnifiedClient as _UC
            model_needs_api_key = _UC._model_needs_api_key(model)
        except Exception:
            model_needs_api_key = bool(model)

        if '@' in model or model.startswith('vertex/'):
            google_creds = self.config.get('google_cloud_credentials')
            if not google_creds or not os.path.exists(google_creds):
                self.append_log(
                    "❌ Error: Google Cloud credentials required for Vertex AI models."
                )
                return False
            if not api_key:
                try:
                    with open(google_creds, 'r', encoding='utf-8') as creds_file:
                        api_key = json.load(creds_file).get(
                            'project_id', 'vertex-ai-project'
                        )
                except Exception:
                    api_key = 'vertex-ai-project'
        elif model_needs_api_key and not api_key:
            self.append_log("❌ Error: Please enter your API key.")
            return False

        shared_env = self._get_environment_variables(files[0], api_key)
        prepared_jobs: list[tuple[str, dict]] = []
        for file_path in files:
            env_vars = self._metadata_only_environment_for_file(
                file_path,
                api_key,
                base_env=shared_env,
            )
            prepared_jobs.append((file_path, env_vars))

        try:
            batch_size = max(1, int(getattr(self, 'batch_size_var', 5)))
        except (TypeError, ValueError):
            batch_size = 5
        worker_count = min(batch_size, len(prepared_jobs))
        self.append_log(
            f"⚡ Metadata batch mode: translating {len(prepared_jobs)} EPUBs "
            f"with {worker_count} thread"
            f"{'s' if worker_count != 1 else ''}"
        )

        # Provider/model settings are identical for every selected EPUB. Apply
        # them once before starting the pool. Book-specific paths and metadata
        # selections stay in each job dict and are never written to os.environ
        # by worker threads.
        per_book_env_keys = {
            'EPUB_PATH',
            'GLOSSARY_SOURCE_PATH',
            'MANUAL_GLOSSARY',
            'OUTPUT_DIRECTORY',
            'OUTPUT_DIR',
            'EPUB_OUTPUT_DIR',
            'TRANSLATE_METADATA_FIELDS',
            'SUBTITLE_OUTPUT_GROUP_DIR',
            'SUBTITLE_OUTPUT_FILE',
            'SUBTITLE_WORK_DIR',
            'SUBTITLE_BUNDLE_FILES_JSON',
            'SUBTITLE_BUNDLE_OUTPUTS_JSON',
            'SUBTITLE_BUNDLE_WORK_DIR',
        }
        first_env = prepared_jobs[0][1]
        common_env = {
            key: value
            for key, value in first_env.items()
            if (
                key not in per_book_env_keys
                and all(job_env.get(key) == value for _, job_env in prepared_jobs)
            )
        }
        try:
            import large_env

            large_env.update_env(common_env)
        except Exception:
            for key, value in common_env.items():
                try:
                    os.environ[str(key)] = (
                        '' if value is None else str(value)
                    )
                except (OSError, ValueError):
                    pass
        os.environ.pop('TRANSLATION_CANCELLED', None)
        os.environ['GRACEFUL_STOP'] = '0'
        os.environ['GRACEFUL_STOP_COMPLETED'] = '0'
        self._metadata_worker_processes = set()

        total_jobs = len(prepared_jobs)

        def _run_one(
            file_path: str,
            env_vars: dict,
            job_index: int,
        ) -> tuple[str, bool]:
            if self.stop_requested:
                return file_path, False

            worker_tag = f"M{job_index}"
            display_name = os.path.splitext(
                os.path.basename(file_path)
            )[0]
            self.append_log(
                f"▶ [{worker_tag}] Metadata {job_index}/{total_jobs}: "
                f"{display_name}"
            )

            try:
                succeeded = run_metadata_translation_job(
                    file_path,
                    env_vars,
                    log_callback=lambda message: self.append_log(
                        f"[{worker_tag}] {message}"
                    ),
                    stop_check_fn=lambda: bool(self.stop_requested),
                )
                self.append_log(
                    f"{'✅' if succeeded else '❌'} "
                    f"[{worker_tag}] Metadata "
                    f"{'complete' if succeeded else 'failed'}"
                )
                return file_path, succeeded
            except Exception as exc:
                self.append_log(
                    f"❌ Metadata worker failed for "
                    f"{os.path.basename(file_path)}: {exc}"
                )
                return file_path, False

        successful = 0
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix='MetadataBook',
        ) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    file_path,
                    env_vars,
                    job_index,
                )
                for job_index, (file_path, env_vars)
                in enumerate(prepared_jobs, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    _file_path, succeeded = future.result()
                except Exception as exc:
                    self.append_log(
                        f"❌ Metadata thread failed unexpectedly: {exc}"
                    )
                    succeeded = False
                if succeeded:
                    successful += 1
                else:
                    failed += 1

        self._metadata_worker_processes = set()
        self.append_log("\n" + "=" * 60)
        self.append_log("🌐 Metadata Translation Summary:")
        self.append_log(f"   ✅ Successful: {successful} EPUBs")
        if failed:
            self.append_log(f"   ❌ Failed: {failed} EPUBs")
        self.append_log(f"   📁 Total: {len(prepared_jobs)} EPUBs")
        self.append_log("=" * 60)
        return successful > 0 and not self.stop_requested

    def _process_text_file(self, file_path):
        """Process EPUB, text-like, PDF, or SDLXLIFF files."""
        try:
            translation_main = self._backend_entry('translation_main')
            if translation_main is None:
                self.append_log("❌ Translation module is not available")
                return False

            api_key = self.api_key_entry.text()
            model = self.model_var
            
            # Check if model needs API key (delegates to UnifiedClient's authoritative list)
            try:
                from unified_api_client import UnifiedClient as _UC
                model_needs_api_key = _UC._model_needs_api_key(model)
            except Exception:
                model_needs_api_key = bool(model)  # safe fallback
            
            # Validate API key and model (same as original)
            if '@' in model or model.startswith('vertex/'):
                google_creds = self.config.get('google_cloud_credentials')
                if not google_creds or not os.path.exists(google_creds):
                    self.append_log("❌ Error: Google Cloud credentials required for Vertex AI models.")
                    return False
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                self.append_log(f"🔑 Using Google Cloud credentials: {os.path.basename(google_creds)}")
                
                if not api_key:
                    try:
                        with open(google_creds, 'r') as f:
                            creds_data = json.load(f)
                            api_key = creds_data.get('project_id', 'vertex-ai-project')
                            self.append_log(f"🔑 Using project ID as API key: {api_key}")
                    except:
                        api_key = 'vertex-ai-project'
            elif model_needs_api_key and not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return False

            # ``source_epub.txt`` is the legacy source pointer used by all
            # EPUB/PDF/TXT workspaces. Format collisions have already been
            # resolved by renaming the input in the common selection handler.
            from output_workspace import (
                source_format_label,
                write_workspace_source_reference,
            )
            source_format = source_format_label(file_path)
            output_dir = None
            if source_format:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                metadata_roots = (
                    getattr(self, '_metadata_output_roots', {}) or {}
                    if getattr(self, '_metadata_only_run', False)
                    else {}
                )
                metadata_root = metadata_roots.get(
                    os.path.normcase(os.path.abspath(file_path))
                )
                if metadata_root:
                    output_dir = os.path.join(metadata_root, base_name)
                else:
                    output_dir = self._resolve_translation_output_dir(file_path)
                try:
                    write_workspace_source_reference(output_dir, file_path)
                    self.append_log(
                        f"📚 Saved {source_format} source reference in "
                        f"{os.path.basename(os.path.normpath(output_dir))}"
                    )
                except Exception as e:
                    self.append_log(f"⚠️ Could not save source reference: {e}")

            if file_path.lower().endswith('.epub'):
                # Set EPUB_PATH in environment for immediate use
                os.environ['EPUB_PATH'] = file_path

                # Rename existing output files to match current retain-source-extension toggle
                # This must run before translation starts so the progress tracker sees correct filenames
                try:
                    from output_naming import _rename_output_files_for_retain
                    retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                    _rename_output_files_for_retain(self, retain, output_dir=output_dir)
                except Exception as e:
                    self.append_log(f"⚠️ Could not sync output filenames: {e}")

                
            process_state = scoped_process_state().snapshot()
            

            try:
                # Set up environment (same as original)
                self.append_log(f"🔧 Setting up environment variables...")
                self.append_log(f"📖 File: {os.path.basename(file_path)}")
                self.append_log(f"🤖 Model: {self.model_var}")
                
                # Get the system prompt and log first 100 characters
                system_prompt = self.prompt_text.toPlainText().strip()
                
                # Replace split marker instruction placeholder
                split_instr = ""
                if getattr(self, 'request_merging_enabled_var', False):
                    split_instr = "- CRITICAL Requirement: If you see any HTML tags containing 'SPLIT MARKER' (Example: <h1 id=\"split-1\">SPLIT MARKER: Do Not Remove This Tag</h1>), you MUST preserve them EXACTLY as they appear. Do not translate, modify, or remove these markers."
                
                # Use a regex to replace the placeholder regardless of surrounding whitespace or newlines
                import re
                system_prompt = re.sub(r'\s*\{split_marker_instruction\}\s*', lambda m: '\n' + split_instr if split_instr else '', system_prompt)

                prompt_preview = system_prompt[:] if len(system_prompt) > 100 else system_prompt
                prompt_type = "User prompt" if os.environ.get('SYSTEM_PROMPT_TO_USER', '0') == '1' else "System prompt"
                self.append_log(f"📝 {prompt_type}: {prompt_preview}")
                self.append_log(f"📏 {prompt_type} length: {len(system_prompt)} characters")
                
                # Log assistant prompt if set
                if hasattr(self, 'assistant_prompt') and self.assistant_prompt and self.assistant_prompt.strip():
                    self.append_log(f"🤖 Assistant Prompt: {self.assistant_prompt}")
                
                
                
                # Log glossary status
                if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                    glossary_label = (
                        "Manual"
                        if getattr(self, 'manual_glossary_manually_loaded', False)
                        else "Auto-mapped"
                    )
                    self.append_log(f"📑 {glossary_label} glossary loaded: {os.path.basename(self.manual_glossary_path)}")
                else:
                    self.append_log(f"📑 No manual glossary loaded")
                
                # IMPORTANT: Set IS_TEXT_FILE_TRANSLATION flag for text files
                if file_path.lower().endswith(('.txt', '.csv', '.json', '.pdf', '.sdlxliff', '.srt', '.ass', '.lrc')):
                    os.environ['IS_TEXT_FILE_TRANSLATION'] = '1'
                    self.append_log("📄 Processing as text file")
                
                # Set environment variables
                multipass_enabled, multipass_refinement_mode = self._export_multipass_runtime_env()
                env_vars = self._get_environment_variables(file_path, api_key)
                env_vars['MULTIPASS_MODE'] = '1' if multipass_enabled else '0'
                env_vars['MULTIPASS_REFINEMENT_MODE'] = multipass_refinement_mode

                if getattr(self, '_metadata_only_run', False):
                    env_vars = self._metadata_only_environment_for_file(
                        file_path,
                        api_key,
                        base_env=env_vars,
                    )
                
                # Metadata-only runs read the OPF metadata directly and never
                # start chapter extraction.
                if getattr(self, '_metadata_only_run', False):
                    env_vars['USE_ASYNC_CHAPTER_EXTRACTION'] = '0'
                    self.append_log(
                        "⚡ Reading metadata directly from the source EPUB "
                        "(chapter extraction bypassed)"
                    )
                # Enable async chapter extraction for EPUBs, PDFs, and SDLXLIFF to prevent GUI freezing
                elif file_path.lower().endswith(('.epub', '.pdf', '.sdlxliff')):
                    env_vars['USE_ASYNC_CHAPTER_EXTRACTION'] = '1'
                    self.append_log("🚀 Using async chapter extraction (subprocess mode)")
                
                import large_env
                large_env.update_env(env_vars)
                # ``_get_environment_variables`` reflects persisted checkboxes.
                # Live input/output and reader runs intentionally override those
                # values, so re-assert the forced flags after the bulk export.
                if getattr(self, '_force_stream_all', False):
                    self._apply_forced_streaming_environment()
                if getattr(self, '_input_output_run_active', False):
                    self._apply_direct_text_runtime_environment()
                
                # Re-export the exact live scope after the bulk environment
                # update so every extraction/translation/multipass phase agrees.
                chap_range, _parsed_range, use_spine_order = (
                    self._export_chapter_range_runtime_env()
                )
                if chap_range:
                    if use_spine_order:
                        self.append_log(f"📊 Chapter Range (Spine Order): {chap_range}")
                    else:
                        self.append_log(f"📊 Chapter Range: {chap_range}")
                
                # Set other environment variables (token limits, etc.)
                if hasattr(self, 'token_limit_disabled') and self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                else:
                    token_val = self.token_limit_entry.text().replace(',', '').strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '1000000'
                
                # Validate glossary path
                if self._current_auto_glossary_mode() == 'no_glossary':
                    # A manual glossary can be reattached during normal text-file
                    # setup. Keep No Glossary authoritative for this run.
                    os.environ.pop('MANUAL_GLOSSARY', None)
                else:
                    # Check per-file mapping first (set by the main loop before calling this method)
                    _mapped_gp = os.environ.get('MANUAL_GLOSSARY', '')
                    if _mapped_gp and os.path.exists(_mapped_gp):
                        # Already set by the per-EPUB mapping loop – keep it
                        pass
                    elif hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                        if (hasattr(self, 'auto_loaded_glossary_path') and
                            self.manual_glossary_path == self.auto_loaded_glossary_path):
                            if (hasattr(self, 'auto_loaded_glossary_for_file') and
                                hasattr(self, 'file_path') and
                                self.file_path == self.auto_loaded_glossary_for_file):
                                os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                                self.append_log(f"📑 Using auto-loaded glossary: {os.path.basename(self.manual_glossary_path)}")
                        else:
                            os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                            self.append_log(f"📑 Using manual glossary: {os.path.basename(self.manual_glossary_path)}")
                
                # ── Single-chapter mode (Library / Reader "Translate") ──
                # Only the targeted HTML file is extracted from the EPUB and
                # the run jumps straight to the translation phase. The filter
                # is consumed by Chapter_Extractor._extract_chapters_universal.
                _sc_filter = getattr(self, '_single_chapter_filter', None)
                if getattr(self, '_metadata_only_run', False):
                    os.environ['METADATA_ONLY'] = '1'
                    os.environ.pop('SINGLE_CHAPTER_FILTER', None)
                    os.environ.pop('CHAPTER_RANGE', None)
                    self.append_log(
                        "🌐 Metadata-only mode: running the normal metadata "
                        "phase and skipping chapter translation"
                    )
                elif _sc_filter:
                    os.environ['SINGLE_CHAPTER_FILTER'] = str(_sc_filter)
                    # Targeted extraction is tiny — run it in-process.
                    os.environ['USE_ASYNC_CHAPTER_EXTRACTION'] = '0'
                    # A leftover chapter range could exclude the target file.
                    os.environ.pop('CHAPTER_RANGE', None)
                    # Re-assert forced streaming AFTER large_env.update_env —
                    # _get_environment_variables exports the user's toggle
                    # values, which would otherwise override the live view.
                    if getattr(self, '_force_stream_all', False):
                        self._apply_forced_streaming_environment()
                    self.append_log(
                        f"🎯 Single-chapter mode: {os.path.basename(str(_sc_filter))} "
                        "(skipping full extraction, jumping to translation)")
                else:
                    os.environ.pop('SINGLE_CHAPTER_FILTER', None)

                # Set sys.argv to match what TransateKRtoEN.py expects
                sys.argv = ['TransateKRtoEN.py', file_path]

                if getattr(self, '_translation_run_is_multipass_qa_refinement', False):
                    mode_label = str(getattr(self, '_translation_run_qa_refinement_mode', '') or 'multipass').title()
                    self.append_log(f"🚀 Starting {mode_label} multipass refinement...")
                elif getattr(self, '_metadata_only_run', False):
                    self.append_log("🌐 Starting metadata translation phase...")
                else:
                    self.append_log("🚀 Starting translation...")
                
                # Ensure Payloads directory exists (non-fatal if it fails)
                try:
                    os.makedirs("Payloads", exist_ok=True)
                except (PermissionError, OSError):
                    pass  # Payload saving handled by unified_api_client fallback
                
                # Run translation
                translation_result = translation_main(
                    log_callback=self.append_log,
                    stop_callback=lambda: self.stop_requested
                )

                if translation_result is False:
                    self.append_log("❌ Translation did not complete successfully.")
                    return False
                
                if not self.stop_requested:
                    if getattr(self, '_translation_run_is_multipass_qa_refinement', False):
                        mode_label = str(getattr(self, '_translation_run_qa_refinement_mode', '') or 'Multipass').title()
                        self.append_log(f"✅ {mode_label} multipass refinement completed successfully!")
                    elif getattr(self, '_metadata_only_run', False):
                        self.append_log("✅ Metadata translation completed successfully!")
                    else:
                        self.append_log("✅ Translation completed successfully!")
                    return True
                else:
                    return False
                    
            except ValueError as e:
                # ValueError is used for user-facing errors like invalid chapter range
                # These already have clear error messages, so no need for traceback
                error_msg = str(e)
                self.append_log(f"[DEBUG] ValueError caught in _process_text_file: {error_msg}")
                if "Chapter range" not in error_msg:
                    # If it's not a chapter range error, show the message
                    self.append_log(f"❌ Translation error: {e}")
                # Don't show traceback for user-friendly errors
                return False
                
            except Exception as e:
                # Suppress noisy traceback when user stops/cancels (including graceful stop).
                err_str = str(e).lower()

                # Prefer structured cancellation detection when available.
                is_cancelled = False
                is_config_error = False
                try:
                    from unified_api_client import UnifiedClientError
                    if isinstance(e, UnifiedClientError):
                        is_cancelled = getattr(e, 'error_type', None) == 'cancelled'
                        is_config_error = getattr(e, 'error_type', None) == 'config_error'
                except Exception:
                    pass

                # Fallback to string matching for legacy/foreign exceptions.
                if (
                    "cancelled by user" in err_str or "canceled by user" in err_str or
                    "operation cancelled" in err_str or "operation canceled" in err_str or
                    "graceful stop active" in err_str
                ):
                    is_cancelled = True

                if is_cancelled:
                    # Keep messaging user-friendly and avoid traceback spam.
                    if "graceful stop" in err_str:
                        self.append_log("⏹️ Graceful stop: not starting new API call")
                    else:
                        self.append_log("❌ Translation stopped by user")
                else:
                    # Expected API setup errors already contain an actionable message.
                    self.append_log(f"❌ Translation error: {e}")
                    if hasattr(self, 'append_log_with_api_error_detection'):
                        self.append_log_with_api_error_detection(str(e))
                    if not is_config_error:
                        import traceback
                        self.append_log(f"❌ Full error: {traceback.format_exc()}")
                return False
            
            finally:
                process_state.restore()
                
        except Exception as e:
            self.append_log(f"❌ Error in text file processing: {str(e)}")
            return False

    def _extract_glossary_from_text_file(self, file_path, force_balanced_request_merging=False):
        """Extract a glossary from supported document or subtitle sources."""
        # Skip glossary extraction for traditional APIs
        try:
            api_key = self.api_key_entry.text()
            model = str(getattr(self, 'model_var', '') or '').strip()
            if not model:
                self.append_log("❌ Glossary extraction stopped: no model is selected.")
                return False
            if is_traditional_translation_api(model):
               self.append_log("ℹ️ Skipping automatic glossary extraction (not supported by Google Translate / DeepL translation APIs)")
               return {}
            
            # Check if model needs API key (delegates to UnifiedClient's authoritative list)
            try:
                from unified_api_client import UnifiedClient as _UC
                model_needs_api_key = _UC._model_needs_api_key(model)
            except Exception:
                model_needs_api_key = bool(model)  # safe fallback
            
            # Validate Vertex AI credentials if needed
            if '@' in model or model.startswith('vertex/'):
                google_creds = self.config.get('google_cloud_credentials')
                if not google_creds or not os.path.exists(google_creds):
                    self.append_log("❌ Error: Google Cloud credentials required for Vertex AI models.")
                    return False
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                self.append_log(f"🔑 Using Google Cloud credentials: {os.path.basename(google_creds)}")
                
                if not api_key:
                    try:
                        with open(google_creds, 'r') as f:
                            creds_data = json.load(f)
                            api_key = creds_data.get('project_id', 'vertex-ai-project')
                            self.append_log(f"🔑 Using project ID as API key: {api_key}")
                    except:
                        api_key = 'vertex-ai-project'
            elif model_needs_api_key and not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return False
            
            process_state = scoped_process_state(clear_large_env=False).snapshot()
            
            (
                shared_glossary_dir,
                output_path,
                output_side_backup_dir,
                save_glossary_in_output,
            ) = self._glossary_extraction_paths(file_path)
            
            try:
                (
                    env_updates,
                    resolved_glossary_tokens,
                    glossary_token_cfg,
                    cjk_script_filter_enabled,
                ) = self._build_glossary_extraction_env(
                    file_path,
                    api_key,
                    model,
                    shared_glossary_dir=shared_glossary_dir,
                    save_glossary_in_output=save_glossary_in_output,
                    output_side_backup_dir=output_side_backup_dir,
                    force_balanced_request_merging=force_balanced_request_merging,
                )
                
                # Propagate multi-key toggles so retry logic can engage
                # Both must be enabled for main-then-fallback retry
                try:
                    if self.config.get('use_multi_api_keys', False):
                        os.environ['USE_MULTI_KEYS'] = '1'
                    else:
                        os.environ['USE_MULTI_KEYS'] = '0'
                    if self.config.get('use_fallback_keys', False):
                        os.environ['USE_FALLBACK_KEYS'] = '1'
                    else:
                        os.environ['USE_FALLBACK_KEYS'] = '0'
                    if self.config.get('use_glossary_keys', False):
                        os.environ['USE_GLOSSARY_KEYS'] = '1'
                    else:
                        os.environ['USE_GLOSSARY_KEYS'] = '0'
                    os.environ['USE_GLOSSARY_REFINEMENT_KEYS'] = '1' if self.config.get('use_glossary_refinement_keys', False) else '0'
                    os.environ['GLOSSARY_REFINEMENT_API_KEYS'] = json.dumps(self.config.get('glossary_refinement_keys', []))
                except Exception:
                    # Keep going even if we can't set env for some reason
                    pass

                os.environ.update(env_updates)
                
                chap_range = self.chapter_range_entry.text().strip()
                if chap_range:
                    self.append_log(f"📊 Chapter Range: {chap_range} (glossary extraction will only process these chapters)")
                
                if self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    token_val = self.token_limit_entry.text().replace(',', '').strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                        self.append_log(f"🎯 Input Token Limit: {int(token_val):,}")
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '50000'
                        self.append_log(f"🎯 Input Token Limit: 50000 (default)")
                
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', file_path,
                    '--output', output_path,
                    '--config', CONFIG_FILE
                ]
                
                self.append_log(f"🚀 Extracting glossary from: {os.path.basename(file_path)}")
                self.append_log(f"📤 Output Token Limit: {resolved_glossary_tokens} ({'glossary override' if str(glossary_token_cfg) != '-1' else 'global'})")
                self.append_log(
                    f"[DEBUG] GLOSSARY_CJK_SCRIPT_FILTER = {os.environ.get('GLOSSARY_CJK_SCRIPT_FILTER', '<NOT SET>')} "
                    f"(enabled: {cjk_script_filter_enabled}, target: {os.environ.get('GLOSSARY_TARGET_LANGUAGE', '<NOT SET>')})"
                )
                format_parts = ["type", "raw_name", "translated_name", "gender"]
                custom_fields_json = self.config.get('manual_custom_fields', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                    if custom_fields:
                        format_parts.extend(custom_fields)
                except:
                    custom_fields = []
                self.append_log(f"   Format: Simple ({', '.join(format_parts)})")
                
                # Check honorifics filter
                if self.config.get('glossary_disable_honorifics_filter', False):
                    self.append_log(f"📑 Honorifics Filter: ❌ DISABLED")
                else:
                    self.append_log(f"📑 Honorifics Filter: ✅ ENABLED")
                
                os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
                
                # Enhanced stop callback that checks both flags
                enhanced_stop_callback = make_glossary_stop_callback(
                    lambda: self.stop_requested,
                    lambda: getattr(self, 'graceful_stop_active', False),
                )

                try:
                    # Import traceback for better error info
                    import traceback
                    
                    # Run glossary extraction with enhanced stop callback
                    glossary_main = self._backend_entry('glossary_main')
                    glossary_main(
                        log_callback=self.append_log,
                        stop_callback=enhanced_stop_callback
                    )
                except Exception as e:
                    # Get the full traceback
                    tb_lines = traceback.format_exc()
                    self.append_log(f"❌ FULL ERROR TRACEBACK:\n{tb_lines}")
                    self.append_log(f"❌ Error extracting glossary from {os.path.basename(file_path)}: {e}")
                    return False
                
                # If stopped:
                # - Immediate stop: treat as cancelled
                # - Graceful stop: allow partial output detection below
                if self.stop_requested and not (bool(getattr(self, 'graceful_stop_active', False)) or (os.environ.get('GRACEFUL_STOP') == '1')):
                    self.append_log("⏹️ Glossary extraction was stopped")
                    try:
                        self._reset_api_watchdog_progress(clear_stale_external_files=True)
                    except Exception:
                        pass
                    return False
                # Check if output file exists - check both JSON and CSV
                # Even if stopped, we consider it a partial success if the file exists and has content
                
                has_content = False
                success_type = "Full"
                
                # Check explicit output path (likely JSON)
                if os.path.exists(output_path):
                    has_content = True
                
                # Check CSV variant
                csv_path = os.path.splitext(output_path)[0] + '.csv'
                if os.path.exists(csv_path):
                    has_content = True
                    
                # Check in Glossary subfolder if not found
                if not has_content:
                    glossary_json_sub = os.path.join(shared_glossary_dir, os.path.basename(output_path))
                    glossary_csv_sub = os.path.splitext(glossary_json_sub)[0] + '.csv'
                    
                    if os.path.exists(glossary_json_sub):
                        has_content = True
                    if os.path.exists(glossary_csv_sub):
                        has_content = True
                
                if has_content:
                    if self.stop_requested:
                        self.append_log(f"⚠️ Partial glossary saved (stopped by user): {output_path}")
                        # Don't return True here so it doesn't count as fully successful, 
                        # but we can track it as partial if needed
                        return False 
                    else:
                        self.append_log(f"✅ Glossary saved to: {output_path}")
                        return True
                else:
                    return False
                
            finally:
                process_state.restore()
                
        except Exception as e:
            self.append_log(f"❌ Error extracting glossary from {os.path.basename(file_path)}: {e}")
            return False

    def _run_pdf_compile(self, folder=None):
        """Compile a PDF workspace: run_pdf_converter_direct's try/except, GUI-free.

        *folder* (default: ``self.pdf_folder``) becomes ``self.pdf_folder``. The
        success / failure message is the ``_notify_compile_result`` hook (desktop:
        message boxes); the desktop runner keeps its thread/stop/button reset.
        Returns a ``CompileResult``.
        """
        if folder is not None:
            self.pdf_folder = folder
        result = CompileResult('pdf')
        try:
            from pdf_workspace_compiler import compile_pdf_workspace

            self._build_pdf_compile_env()

            pdf_api_client = None
            if (
                os.environ['USE_TOC_NCX'] == '1'
                or os.environ['BATCH_TRANSLATE_HEADERS'] == '1'
            ):
                from unified_api_client import UnifiedClient

                model = str(getattr(self, 'model_var', '') or '').strip()
                api_key = self.api_key_entry.text().strip()
                if model:
                    os.environ['MODEL'] = model
                if api_key:
                    os.environ['API_KEY'] = api_key
                needs_key = UnifiedClient._model_needs_api_key(model)
                if model and (api_key or not needs_key):
                    pdf_api_client = UnifiedClient(
                        api_key=api_key or 'dummy-key-not-required',
                        model=model,
                        output_dir=self.pdf_folder,
                    )
                else:
                    self.append_log(
                        "⚠️ PDF bookmark/header translation skipped: "
                        "the selected model/API key is not available"
                    )

            compiled_path = compile_pdf_workspace(
                self.pdf_folder,
                log_callback=self.append_log,
                stop_callback=lambda: bool(self.stop_requested),
                api_client=pdf_api_client,
            )
            if compiled_path and os.path.isfile(compiled_path):
                result.path = compiled_path
                self._notify_compile_result('pdf', path=compiled_path)
        except Exception as exc:
            self.append_log(f"❌ PDF Compiler error: {exc}")
            result.error = str(exc)
            self._notify_compile_result('pdf', error=result.error)
        return result

    def _run_epub_compile(self, folder=None):
        """Compile an EPUB workspace: run_epub_converter_direct's try/except, GUI-free.

        *folder* (default: ``self.epub_folder``) becomes ``self.epub_folder``. The
        success / failure message is the ``_notify_compile_result`` hook (desktop:
        message boxes); the desktop runner keeps its thread/stop/button reset.
        Returns a ``CompileResult`` (``stopped`` when Stop was requested).
        """
        if folder is not None:
            self.epub_folder = folder
        result = CompileResult('epub')
        try:
            # Reset stop flag at the start
            try:
                import epub_converter
                if hasattr(epub_converter, 'set_stop_flag'):
                    epub_converter.set_stop_flag(False)
            except Exception:
                pass
            
            folder = self.epub_folder
            self.append_log("📦 Starting EPUB Converter...")
            
            self._build_epub_compile_env(folder)

            fallback_compile_epub = self._backend_entry('fallback_compile_epub')
            compiled_path = fallback_compile_epub(folder, log_callback=self.append_log)
            
            if not self.stop_requested:
                self.append_log("✅ EPUB Converter completed successfully!")
                
                if compiled_path and os.path.isfile(compiled_path):
                    result.path = compiled_path
                    self._notify_compile_result('epub', path=compiled_path)
                else:
                    epub_files = [f for f in os.listdir(folder) if f.endswith('.epub')]
                    if epub_files:
                        epub_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
                        out_file = os.path.join(folder, epub_files[0])
                        result.path = out_file
                        self._notify_compile_result('epub', path=out_file)
                    else:
                        self.append_log("⚠️ EPUB file was not created. Check the logs for details.")
            else:
                result.stopped = True

        except Exception as e:
            error_str = str(e)
            self.append_log(f"❌ EPUB Converter error: {error_str}")
            result.error = error_str

            if "Document is empty" not in error_str:
                self._notify_compile_result('epub', error=error_str)
            else:
                self.append_log("📋 Check the log above for details about what went wrong.")
        return result
