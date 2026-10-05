"""ConfigStateMixin: the desktop owner state, moved verbatim out of TranslatorGUI.

Shared GUI-free core (Glossarion mobile rewrite, milestone U2). Holds the code that
turns a loaded config dict into the desktop's ``*_var`` attributes and startup
environment, plus the GUI-backed startup steps the mobile ``HeadlessOwner`` must
replay to behave like a freshly started desktop:

* ``_init_config_state()`` = the ``__init__`` config block (frozen by the parity
  oracle as ``legacy_init_block``); ``_init_default_prompt_profiles()`` = the
  ``__init__`` default-prompt block; ``_init_watchdog_dir()`` = the ``__init__``
  watchdog-dir export; ``_init_variables()`` / ``_init_default_prompts()`` and the
  sanitizers/migrations are whole-method moves;
* ``_init_gui_backed_state()`` = the state assignments the ``_setup_gui`` section
  builders made while creating widgets (vertex_location_var, deep_scan_var,
  model_var, context_mode_var, translation_history_rolling). They only read
  config/vars and nothing reads them before their old position (checked when
  moving), so TranslatorGUI.__init__ now calls this right before ``_setup_gui``;
* the startup handlers desktop fires once widgets exist (AuthGem project restore,
  temperature toggle, glossary-mode shortcut handler + save_config, context mode,
  target language, active profile prompt, auto compression factor) are mixin
  methods the desktop section builders call at the same points;
  ``_replay_gui_startup_handlers()`` runs them in the same order for owners
  without Qt widgets;
* ``_auto_encrypt_api_keys()`` = the last step of ``__init__`` (save_config when
  the decrypted config holds a plain API key).

Desktop-only side effects are hooks with GUI-free defaults (TranslatorGUI
overrides them with the original code): ``_hook_persist_sanitized_config``,
``_hook_save_default_config`` (both write config.json), ``_hook_ensure_executor``,
``_hook_metadata_defaults`` (MetadataBatchTranslatorUI) and
``_hook_context_mode_layout`` (Qt grid placement).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import os
import time

from prompt_defaults import (
    DEFAULT_ASSISTANT_PROMPT,
    DEFAULT_IMAGE_CHUNK_PROMPT,
    DEFAULT_ROLLING_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_ROLLING_SUMMARY_USER_PROMPT,
    DEFAULT_TRANSLATION_CHUNK_PROMPT,
    DEFAULT_VISION_OCR_COMBINED_CONTEXT_PROMPT,
    DEFAULT_VISION_OCR_PROMPT,
    DEFAULT_VISION_OCR_TRANSLATION_USER_PROMPT,
    DEFAULT_VISION_OCR_USER_PROMPT,
    sanitize_prompt_profiles,
)
from refinement_prompts import (
    DEFAULT_REFINEMENT_FAILED_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_FAILED_USER_PROMPT,
    DEFAULT_REFINEMENT_FULL_WITH_RAW_RAW_FOOTER,
    DEFAULT_REFINEMENT_FULL_WITH_RAW_RAW_HEADER,
    DEFAULT_REFINEMENT_FULL_WITH_RAW_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_FULL_WITH_RAW_USER_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_B2_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_B2_USER_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_B_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_B_USER_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_PARTIAL_USER_PROMPT,
    DEFAULT_REFINEMENT_QA_ISSUE_PROMPT,
    DEFAULT_REFINEMENT_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_USER_PROMPT,
)
from run_env import MULTIPASS_REFINEMENT_MODES, REFINEMENT_RAW_PROMPT_ROLES, _format_plain_decimal_setting
from title_tag_translation import DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT


# Moved verbatim from other_settings.py (re-exported there): setup_other_settings_methods
# runs it first so profile switching sees these vars; HeadlessOwner runs it at the same point.
def initialize_extraction_variables(gui_instance):
    """Initialize extraction-related variables early so profile switching works"""
    # Initialize text_extraction_method_var if it doesn't exist
    if not hasattr(gui_instance, 'text_extraction_method_var'):
        # Check config for saved value, or use default
        if gui_instance.config.get('extraction_mode') == 'enhanced':
            gui_instance.text_extraction_method_var = 'enhanced'
        else:
            gui_instance.text_extraction_method_var = gui_instance.config.get('text_extraction_method', 'standard')
    
    # Initialize file_filtering_level_var if it doesn't exist
    if not hasattr(gui_instance, 'file_filtering_level_var'):
        gui_instance.file_filtering_level_var = gui_instance.config.get('file_filtering_level', 'smart')


class ConfigStateMixin:
    """Desktop owner state shared by TranslatorGUI and HeadlessOwner (see module docstring)."""

    # ---- hooks (GUI-free defaults; TranslatorGUI overrides with the original code) ----
    def _hook_persist_sanitized_config(self, updates_made):
        """Persist _sanitize_config_prompts fixes. Default: in memory only (never writes config.json)."""
        return None

    def _hook_save_default_config(self):
        """Persist the first-run auto_update_check default. Default: in memory only."""
        return None

    def _hook_ensure_executor(self):
        """Create the desktop background executor. Default: nothing (owners run on the caller's thread)."""
        return None

    def _hook_metadata_defaults(self):
        """Seed the metadata/batch-header prompt defaults into config.

        Desktop constructs MetadataBatchTranslatorUI(self), whose __init__ runs exactly
        metadata_defaults.ensure_metadata_prompt_defaults(config)."""
        from metadata_defaults import ensure_metadata_prompt_defaults
        ensure_metadata_prompt_defaults(self.config)

    def _hook_context_mode_layout(self, has_context_options):
        """Re-grid the widgets below the Context Mode options. Default: no widget grid."""
        return None

    def _sanitize_config_prompts(self):
        """Auto-fix known issues in user prompts from older versions."""
        if not hasattr(self, 'config'):
            return
        # Fix + flag in place (prompt_defaults.sanitize_prompt_profiles); None = nothing ran
        updates_made = sanitize_prompt_profiles(self.config)
        if updates_made is None:
            return

        # Save if updates were made or just to persist the flag (desktop hook writes config.json)
        self._hook_persist_sanitized_config(updates_made)

    def _get_protected_prompt_profiles(self):
        """Profiles that are treated as mandatory/built-in (cannot be deleted)."""
        # Keep this list in sync with the always-include logic in _init_variables.
        protected = {
            "Universal",
            "Refinement",
            "Korean_BeautifulSoup",
            "Japanese_BeautifulSoup",
            "Chinese_BeautifulSoup",
            "Korean_html2text",
            "Japanese_html2text",
            "Chinese_html2text",
            "RPGMaker_GTool",
            "RPGMaker_GTool_Image",
            "NanoBanana_Image",
            "SDLXLIFF Editing v2",
            "Subtitle Translation",
        }
        return protected

    def _reset_prompt_profile_to_default(self, name: str) -> bool:
        """Reset a built-in profile's prompt text back to the latest built-in defaults."""
        try:
            if not hasattr(self, 'default_prompts'):
                return False
            if name not in getattr(self, 'default_prompts', {}):
                return False
            if not hasattr(self, 'prompt_profiles'):
                return False

            default_text = self.default_prompts[name]
            self.prompt_profiles[name] = default_text
            try:
                self.config['prompt_profiles'] = self.prompt_profiles
            except Exception:
                pass

            # If the user is currently editing/using this profile, refresh the editor.
            try:
                current_name = self.profile_menu.currentText().strip() if hasattr(self, 'profile_menu') else None
            except Exception:
                current_name = None

            if current_name == name and hasattr(self, 'prompt_text'):
                try:
                    self.prompt_text.blockSignals(True)
                except Exception:
                    pass
                try:
                    self.prompt_text.setPlainText(default_text)
                except Exception:
                    try:
                        self.prompt_text.setText(default_text)
                    except Exception:
                        pass
                try:
                    self.prompt_text.blockSignals(False)
                except Exception:
                    pass

                # Keep original-content cache in sync if present
                try:
                    if not hasattr(self, '_original_profile_content'):
                        self._original_profile_content = {}
                    self._original_profile_content[name] = default_text
                except Exception:
                    pass

            return True
        except Exception:
            return False

    def _init_config_state(self):
        """Config-backed attributes and startup env exports (the TranslatorGUI.__init__ config block)."""
        # Load any persisted per-input glossary mapping
        try:
            mgm = self.config.get('manual_glossary_map', {})
            self.manual_glossary_map = mgm if isinstance(mgm, dict) else {}
        except Exception:
            self.manual_glossary_map = {}

        # Ensure default values exist
        if 'auto_update_check' not in self.config:
            self.config['auto_update_check'] = True
            # Save the default config immediately so it exists (desktop hook)
            self._hook_save_default_config()

        # After loading config, check for Google Cloud credentials
        if self.config.get('google_cloud_credentials'):
            creds_path = self.config['google_cloud_credentials']
            if os.path.exists(creds_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                # Log will be added after GUI is created
            
        if 'force_ncx_only' not in self.config:
            self.config['force_ncx_only'] = True

        # Initialize OpenRouter transport/compression toggles early so they're available
        # before the settings UI creates these variables. This prevents attribute errors
        # when features (like glossary extraction) access them at startup.
        try:
            self.openrouter_http_only_var = self.config.get('openrouter_use_http_only', False)
        except Exception:
            self.openrouter_http_only_var = False
        
        try:
            self.openrouter_accept_identity_var = self.config.get('openrouter_accept_identity', False)
        except Exception:
            self.openrouter_accept_identity_var = False

        # Initialize OpenRouter preferred provider early (avoid blank UI when config key is missing/empty)
        try:
            _orp = self.config.get('openrouter_preferred_provider', 'Auto')
            self.openrouter_preferred_provider_var = (_orp or '').strip() or 'Auto'
            # Keep config aligned
            self.config['openrouter_preferred_provider'] = self.openrouter_preferred_provider_var
        except Exception:
            self.openrouter_preferred_provider_var = 'Auto'
            try:
                self.config['openrouter_preferred_provider'] = 'Auto'
            except Exception:
                pass
            
        # Initialize EPUB utility environment flags on startup
        try:
            os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.config.get('retain_source_extension', False) else '0'
            os.environ['DOWNLOAD_REMOTE_IMAGE_URLS'] = '1' if self.config.get('download_remote_image_urls', False) else '0'
            os.environ['REMOTE_IMAGE_DOWNLOAD_WORKERS'] = str(max(
                1, min(32, int(float(
                    self.config.get('remote_image_download_workers', 4)
                )))
            ))
            remote_image_download_interval = max(
                0.0, min(60.0, float(
                    self.config.get('remote_image_download_interval', 0.5)
                ))
            )
            os.environ['REMOTE_IMAGE_DOWNLOAD_INTERVAL'] = f"{remote_image_download_interval:g}"
        except Exception:
            pass
        
        # Force safe ratios is not needed in PySide6
        # Window sizing is handled by Qt's layout system
    
        # Initialize auto-update check and other variables (converted from Tkinter to Python vars)
        self.auto_update_check_var = self.config.get('auto_update_check', True)
        self.auto_dpi_scale_var = self.config.get('auto_dpi_scale', True)
        self.gui_scale_factor_var = self.config.get('gui_scale_factor', 1.0)
        self.gui_font_scale_var = self.config.get('gui_font_scale', 1.0)
        self.force_ncx_only_var = self.config.get('force_ncx_only', True)
        self.use_p_tag_toc_fallback_var = self.config.get('use_p_tag_toc_fallback', False)
        self.deduplicate_toc_var = self.config.get('deduplicate_toc', False)
        self.deduplicate_toc_use_translated_var = self.config.get('deduplicate_toc_use_translated', False)
        self.skip_duplicate_toc_translation_var = self.config.get('skip_duplicate_toc_translation', False)
        try:
            os.environ['USE_P_TAG_TOC_FALLBACK'] = '1' if self.use_p_tag_toc_fallback_var else '0'
        except Exception:
            pass
        try:
            os.environ['DEDUPLICATE_TOC'] = '1' if self.deduplicate_toc_var else '0'
        except Exception:
            pass
        try:
            os.environ['DEDUPLICATE_TOC_USE_TRANSLATED'] = '1' if self.deduplicate_toc_use_translated_var else '0'
        except Exception:
            pass
        try:
            os.environ['SKIP_DUPLICATE_TOC_TRANSLATION'] = '1' if self.skip_duplicate_toc_translation_var else '0'
        except Exception:
            pass
        self.single_api_image_chunks_var = False
        self.vision_ocr_batch_translation_var = self.config.get('vision_ocr_batch_translation', True)
        self.vision_ocr_batch_size_var = str(self.config.get('vision_ocr_batch_size', '-1'))
        self.vision_ocr_skip_translation_var = self.config.get('vision_ocr_skip_translation', False)
        self.vision_ocr_source_prepass_var = str(self.config.get('vision_ocr_source_prepass', 'auto'))
        os.environ['VISION_OCR_SKIP_TRANSLATION'] = '1' if self.vision_ocr_skip_translation_var else '0'
        self.enable_gemini_thinking_var = self.config.get('enable_gemini_thinking', True)
        self.thinking_budget_var = str(self.config.get('thinking_budget', '-1'))
        self.thinking_level_var = self.config.get('thinking_level', 'high')
        self.enable_thoughts_var = self.config.get('enable_thoughts', True)
        self.gemini_service_tier_var = str(self.config.get('gemini_service_tier', 'off') or 'off')
        os.environ['GEMINI_SERVICE_TIER'] = self.gemini_service_tier_var
        self.force_service_tier_unknown_routes_var = bool(
            self.config.get('force_service_tier_unknown_routes', False)
        )
        os.environ['FORCE_SERVICE_TIER_UNKNOWN_ROUTES'] = (
            '1' if self.force_service_tier_unknown_routes_var else '0'
        )
        # NEW: GPT/OpenRouter reasoning controls
        self.enable_gpt_thinking_var = self.config.get('enable_gpt_thinking', True)
        self.gpt_reasoning_tokens_var = str(self.config.get('gpt_reasoning_tokens', '2000'))
        self.gpt_effort_var = self.config.get('gpt_effort', 'medium')
        self.openrouter_use_reasoning_tokens_var = bool(
            self.config.get('openrouter_use_reasoning_tokens', False)
        )
        self.pass_thinking_all_openai_var = self.config.get('pass_thinking_all_openai', False)
        # NEW: DeepSeek thinking (OpenAI-compatible extra_body)
        self.enable_deepseek_thinking_var = self.config.get('enable_deepseek_thinking', True)
        self.deepseek_effort_var = self.config.get('deepseek_effort', 'high')
        self.deepseek_use_responses_api_var = self.config.get('deepseek_use_responses_api', False)
        # NEW: Anthropic extended/adaptive thinking
        self.enable_anthropic_thinking_var = self.config.get('enable_anthropic_thinking', False)
        self.anthropic_thinking_budget_var = str(self.config.get('anthropic_thinking_budget', '10000'))
        self.anthropic_force_adaptive_var = self.config.get('anthropic_force_adaptive', False)
        self.anthropic_effort_var = self.config.get('anthropic_effort', 'medium')
        # Skip thinking for lightweight tasks
        self.skip_book_title_thinking_var = self.config.get('skip_book_title_thinking', True)
        self.skip_metadata_thinking_var = self.config.get('skip_metadata_thinking', True)
        self.skip_toc_thinking_var = self.config.get('skip_toc_thinking', False)
        self.lightweight_thinking_level_var = self.config.get('lightweight_thinking_level', 1)
        self.thread_delay_var = _format_plain_decimal_setting(self.config.get('thread_submission_delay', '0.0001'))
        _raw_artifacts = os.getenv("REMOVE_AI_ARTIFACTS", "off")
        if _raw_artifacts == "0": _raw_artifacts = "off"
        elif _raw_artifacts == "1": _raw_artifacts = "medium"
        self.remove_ai_artifacts = _raw_artifacts if _raw_artifacts in ("off", "low", "medium", "high") else "off"
        print(f"   🎨 Remove AI Artifacts: {self.remove_ai_artifacts.upper()}")
        self.disable_chapter_merging_var = self.config.get('disable_chapter_merging', True)
        # Review settings
        self.review_system_prompt_var = self.config.get('review_system_prompt', '')
        self.review_spoiler_mode_var = self.config.get('review_spoiler_mode', False)
        self.review_chunk_mode_var = self.config.get('review_chunk_mode', False)
        self.review_chunk_wrap_var = self.config.get('review_chunk_wrap', True)
        self.review_volume_mode_var = self.config.get('review_volume_mode', False)
        self.review_final_prompt_var = self.config.get('review_final_prompt', '')
        # Request merging - combine multiple chapters into single API request
        self.request_merging_enabled_var = self.config.get('request_merging_enabled', False)
        self.request_merge_count_var = str(self.config.get('request_merge_count', 3))
        self.split_the_merge_var = self.config.get('split_the_merge', True)
        self.disable_merge_fallback_var = self.config.get('disable_merge_fallback', True)
        # Synthetic headers helper for merged requests (Split-the-Merge aid)
        self.synthetic_merge_headers_var = self.config.get('synthetic_merge_headers', True)
        self.selected_files = []
        self.current_file_index = 0
        self.use_gemini_openai_endpoint_var = self.config.get('use_gemini_openai_endpoint', False)
        self.gemini_openai_endpoint_var = self.config.get('gemini_openai_endpoint', 'generativelanguage.googleapis.com')
        # Override Gemma routing: when enabled (default), Gemma models use the OpenAI custom endpoint
        # instead of the Gemini custom endpoint, so they can run on local LLM servers (Ollama/LM Studio/etc.)
        self.override_gemma_for_custom_endpoint_var = self.config.get('override_gemma_for_custom_endpoint', True)
        try:
            os.environ['OVERRIDE_GEMMA_FOR_CUSTOM_ENDPOINT'] = '1' if self.override_gemma_for_custom_endpoint_var else '0'
        except Exception:
            pass
        self.force_native_anthropic_var = self.config.get('force_native_anthropic', False)
        self.anthropic_base_url_var = self.config.get('anthropic_base_url', '')
        self.azure_api_version_var = self.config.get('azure_api_version', '2025-01-01-preview')
        # Set initial Azure API version environment variable
        azure_version = self.config.get('azure_api_version', '2025-01-01-preview')
        os.environ['AZURE_API_VERSION'] = azure_version
        print(f"🔧 Initial Azure API Version set: {azure_version}")
        self.use_fallback_keys_var = self.config.get('use_fallback_keys', False)
        self.use_glossary_keys_var = self.config.get('use_glossary_keys', False)
        self.use_glossary_refinement_keys_var = self.config.get('use_glossary_refinement_keys', False)
        self.use_metadata_keys_var = self.config.get('use_metadata_keys', False)
        self.use_qa_scan_keys_var = self.config.get('use_qa_scan_keys', False)
        self.use_ai_truncation_detection_keys_var = self.config.get('use_ai_truncation_detection_keys', False)
        self.use_rolling_summary_keys_var = self.config.get('use_rolling_summary_keys', False)
        self.use_truncation_retry_keys_var = self.config.get('use_truncation_retry_keys', False)
        self.use_inpainter_keys_var = self.config.get('use_inpainter_keys', False)
        self.use_tts_keys_var = self.config.get('use_tts_keys', False)
        self.multipass_mode_var = self.config.get('multipass_mode', False)
        self.multipass_refinement_mode_var = str(self.config.get('multipass_refinement_mode', 'full') or 'full').strip().lower()
        if self.multipass_refinement_mode_var not in MULTIPASS_REFINEMENT_MODES:
            self.multipass_refinement_mode_var = 'full'
        self.refinement_full_with_raw_raw_role_var = str(
            self.config.get('refinement_full_with_raw_raw_role', 'assistant') or 'assistant'
        ).strip().lower()
        if self.refinement_full_with_raw_raw_role_var not in REFINEMENT_RAW_PROMPT_ROLES:
            self.refinement_full_with_raw_raw_role_var = 'assistant'

        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = self.config.get('glossary_fuzzy_threshold', 0.90)
        self.use_legacy_csv_var = self.config.get('glossary_use_legacy_csv', False)
        self.save_glossary_in_output_var = self.config.get('save_glossary_in_output', False)
        # Legacy JSON output toggle (was not persisted previously)
        self.glossary_output_legacy_json_var = self.config.get('glossary_output_legacy_json', False)
        # Dynamic limit expansion toggle (include all characters)
        self.glossary_include_all_characters_var = self.config.get('glossary_include_all_characters', False)
        # Skip identical entries toggle (translated_name == raw_name)
        self.glossary_skip_identical_entries_var = self.config.get('glossary_skip_identical_entries', True)
        # CJK script filter (auto-rejects CJK in translated_name when output is non-CJK)
        self.glossary_cjk_script_filter_var = self.config.get('glossary_cjk_script_filter', False)
        # Skip gender tracker sidecar generation/use
        self.glossary_skip_gender_tracking_var = self.config.get('glossary_skip_gender_tracking', False)
        self.glossary_gender_noise_threshold_var = self.config.get('glossary_gender_noise_threshold', 10)
        self.glossary_gender_tracking_bias_var = self.config.get('glossary_gender_tracking_bias', 'none')
        self.glossary_partial_ratio_gender_only_var = self.config.get('glossary_partial_ratio_gender_only', False)
        self.glossary_alias_aware_name_matching_var = self.config.get('glossary_alias_aware_name_matching', False)
        self.glossary_alias_aware_gender_only_var = self.config.get('glossary_alias_aware_gender_only', True)
        # Entry type filter mode
        self.glossary_entry_type_filter_mode_var = self.config.get('glossary_entry_type_filter_mode', 'none')

        
        # Initialize the variables with default values
        self.enable_parallel_extraction_var = self.config.get('enable_parallel_extraction', True)
        self.extraction_workers_var = self.config.get(
            'extraction_workers',
            min(8, max(2, (os.cpu_count() or 4) // 2)),
        )
        # GUI yield toggle - disabled by default for maximum speed
        self.enable_gui_yield_var = self.config.get('enable_gui_yield', True)
        # Thread pool extraction toggle - faster on Windows
        self.use_thread_pool_extraction_var = self.config.get('use_thread_pool_extraction', False)

        # Set initial environment variable and ensure executor
        if self.enable_parallel_extraction_var:
            # Set workers for glossary extraction optimization
            workers = self.extraction_workers_var
            os.environ["EXTRACTION_WORKERS"] = str(workers)
            # Also enable glossary parallel processing explicitly
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "1"
            print(f"✅ Parallel extraction enabled with {workers} workers")
        else:
            os.environ["EXTRACTION_WORKERS"] = "1"
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "0"
        
        # Set GUI yield environment variable (disabled by default for maximum speed)
        os.environ['ENABLE_GUI_YIELD'] = '1' if self.enable_gui_yield_var else '0'
        print(f"⚡ GUI yield: {'ENABLED (responsive)' if self.enable_gui_yield_var else 'DISABLED (maximum speed)'}")
        # Set thread pool extraction env var
        os.environ['USE_THREAD_POOL_EXTRACTION'] = '1' if self.use_thread_pool_extraction_var else '0'
        if self.use_thread_pool_extraction_var:
            print(f"⚡ Thread pool extraction: ENABLED (fast I/O mode)")
        # Sync ENABLE_THOUGHTS env with config on startup
        # Force thoughts ON when stream thinking logs are enabled (matches UI lock behavior)
        if getattr(self, 'stream_thinking_logs_var', False):
            self.enable_thoughts_var = True
            self.config['enable_thoughts'] = True
        os.environ['ENABLE_THOUGHTS'] = '1' if self.enable_thoughts_var else '0'
        
        # Initialize the executor based on current settings
        try:
            self._hook_ensure_executor()
        except Exception:
            pass


        # Track original profile content for reverting unsaved changes
        self._original_profile_content = {}
        # Track the currently active profile to prevent cross-profile saves
        self._active_profile_for_autosave = None
        self._profile_user_selected_this_run = False
        
        # System prompt to user message toggle
        self.system_prompt_to_user_var = self.config.get('system_prompt_to_user', False)
        os.environ['SYSTEM_PROMPT_TO_USER'] = '1' if self.system_prompt_to_user_var else '0'
        
        # Initialize compression-related variables
        self.enable_image_compression_var = self.config.get('enable_image_compression', False)
        self.auto_compress_enabled_var = self.config.get('auto_compress_enabled', True)
        self.target_image_tokens_var = str(self.config.get('target_image_tokens', 1000))
        self.image_format_var = self.config.get('image_compression_format', 'auto')
        self.webp_quality_var = self.config.get('webp_quality', 85)
        self.jpeg_quality_var = self.config.get('jpeg_quality', 85)
        self.png_compression_var = self.config.get('png_compression', 6)
        self.max_image_dimension_var = str(self.config.get('max_image_dimension', 2048))
        self.max_image_size_mb_var = str(self.config.get('max_image_size_mb', 10))
        self.preserve_transparency_var = self.config.get('preserve_transparency', False)
        self.preserve_original_format_var = self.config.get('preserve_original_format', False)
        self.optimize_for_ocr_var = self.config.get('optimize_for_ocr', True)
        self.progressive_encoding_var = self.config.get('progressive_encoding', True)
        self.save_compressed_images_var = self.config.get('save_compressed_images', False)
        self.image_chunk_overlap_var = str(self.config.get('image_chunk_overlap', '3'))
        try:
            _min_overlap_px = max(80, int(float(self.config.get('image_chunk_min_overlap_pixels', 80))))
        except Exception:
            _min_overlap_px = 80
        self.image_chunk_min_overlap_pixels_var = str(_min_overlap_px)
        self.vision_ocr_fuzzy_chunk_dedupe_var = self.config.get('vision_ocr_fuzzy_chunk_dedupe', False)
        self.image_smart_chunking_var = self.config.get('image_smart_chunking', True)

        # Glossary-related variables (existing)
        self.append_glossary_var = self.config.get('append_glossary', False)

        # Auto-Mapping (Auto-Fill) toggle
        # NOTE: this controls automatic mapping only; it does not control whether a manually loaded glossary is used.
        if 'append_glossary_auto_load' not in self.config:
            self.config['append_glossary_auto_load'] = False
        self.append_glossary_auto_load_var = self.config.get('append_glossary_auto_load', False)
        self.fuzzy_auto_mapping_var = self.config.get('fuzzy_auto_mapping', False)
        self.fuzzy_auto_mapping_threshold_var = self.config.get('fuzzy_auto_mapping_threshold', 80)

        # Force-sync auto-mapping/fuzzy states based on auto_glossary_mode
        _agm = self.config.get('auto_glossary_mode', 'off')
        if _agm in ('off', 'off_fuzzy_automap', 'balanced', 'full', 'single_pass'):
            self.append_glossary_auto_load_var = True
            self.config['append_glossary_auto_load'] = True
        elif _agm in ('off_no_automap', 'minimal'):
            self.append_glossary_auto_load_var = False
            self.config['append_glossary_auto_load'] = False
        if _agm == 'off_fuzzy_automap':
            self.fuzzy_auto_mapping_var = True
            self.config['fuzzy_auto_mapping'] = True
        elif _agm != 'off_fuzzy_automap':
            self.fuzzy_auto_mapping_var = False
            self.config['fuzzy_auto_mapping'] = False

        self.add_additional_glossary_var = self.config.get('add_additional_glossary', False)
        self.enable_unified_glossary_var = self.config.get('enable_unified_glossary', False)
        self.generate_unified_glossary_var = self.config.get('generate_unified_glossary', False)
        self.unified_glossary_source_language_var = str(self.config.get('unified_glossary_source_language', 'auto') or 'auto')
        self.unified_glossary_combine_all_languages_var = self.config.get('unified_glossary_combine_all_languages', False)
        self.unified_glossary_exclude_gender_entries_var = self.config.get('unified_glossary_exclude_gender_entries', True)
        self.glossary_use_smart_filter_var = self.config.get('glossary_use_smart_filter', True)
        self.glossary_min_frequency_var = str(self.config.get('glossary_min_frequency', 2))
        self.glossary_max_names_var = str(self.config.get('glossary_max_names', 50))
        self.glossary_max_titles_var = str(self.config.get('glossary_max_titles', 30))
        self.context_window_size_var = str(self.config.get('context_window_size', 5))
        self.glossary_max_text_size_var = str(self.config.get('glossary_max_text_size', 0))
        self.glossary_chapter_split_threshold_var = self.config.get('glossary_chapter_split_threshold', '0')
        self.glossary_max_sentences_var = str(self.config.get('glossary_max_sentences', 200))
        self.glossary_filter_mode_var = self.config.get('glossary_filter_mode', 'all')

        
        # NEW: Additional glossary settings
        self.strip_honorifics_var = self.config.get('strip_honorifics', True)
        self.disable_honorifics_var = self.config.get('glossary_disable_honorifics_filter', False)
        self.manual_temp_var = str(self.config.get('manual_glossary_temperature', 0.3))
        self.manual_context_var = str(self.config.get('manual_context_limit', 5))
        
        # Custom glossary fields and entry types
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])

        # Seed 'description' as the default custom field on first run so the
        # {description_mandatory}, {description_detailed}, {description_in_language}
        # and {description_excluded_note} placeholders in the fallback glossary
        # prompts (see _init_default_prompts) expand correctly even if the user
        # clicks Extract Glossary before opening Glossary Manager. Mirrors the
        # logic in GlossaryManager_GUI.py so the two paths agree, and respects
        # the 'custom_field_description_removed' flag if the user intentionally
        # removed it.
        if (
            not self.custom_glossary_fields
            and not self.config.get('custom_field_description_removed', False)
        ):
            self.custom_glossary_fields = ['description']
            self.config['custom_glossary_fields'] = self.custom_glossary_fields

        self.custom_entry_types = self.config.get('custom_entry_types', {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False},
            'surnames': {'enabled': True, 'has_gender': False},
            'titles': {'enabled': True, 'has_gender': True},
            'locations': {'enabled': True, 'has_gender': False},
            'nicknames': {'enabled': True, 'has_gender': True}
        })
        
        # Initialize default prompts BEFORE using them
        self._init_default_prompts()
        
        # Glossary prompts — canonical defaults imported from extract_glossary_from_epub
        try:
            from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT
            _default_manual = DEFAULT_GLOSSARY_PROMPT
        except ImportError:
            _default_manual = ""
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt3', _default_manual)
        if not self.manual_glossary_prompt or not self.manual_glossary_prompt.strip():
            self.manual_glossary_prompt = _default_manual
        
        # Note: Ignoring old 'auto_glosary_prompt2' key to force update to new prompt
        # Also treat empty strings as missing to ensure users get the new default
        try:
            from extract_glossary_from_epub import DEFAULT_AUTO_GLOSSARY_PROMPT
            _default_auto = DEFAULT_AUTO_GLOSSARY_PROMPT
        except ImportError:
            _default_auto = ""
        unified_prompt_from_config = self.config.get('unified_auto_glosary_prompt3', _default_auto)

        if not unified_prompt_from_config or not unified_prompt_from_config.strip():
            self.unified_auto_glosary_prompt3 = self.default_unified_auto_glosary_prompt3
        else:
            self.unified_auto_glosary_prompt3 = unified_prompt_from_config
        
        # Get append_glossary_prompt from config, but treat empty string as missing
        default_append_prompt = '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'
        append_prompt_from_config = self.config.get('append_glossary_prompt', default_append_prompt)
        if not append_prompt_from_config or not append_prompt_from_config.strip():
            self.append_glossary_prompt = default_append_prompt
        else:
            self.append_glossary_prompt = append_prompt_from_config
        
        self.glossary_translation_prompt = self.config.get('glossary_translation_prompt', 
            """
You are translating {language} character names and important terms to English.
For character names, provide English transliterations or keep as romanized.
Keep honorifics/suffixes only if they are integral to the name.
Respond with the same numbered format.

Terms to translate:
{terms_list}

Provide translations in the same numbered format.""")
        self.glossary_format_instructions = self.config.get('glossary_format_instructions', 
            """
You must return the results in CSV format with columns separated by commas to separate columns. Wrap a field value in double quotes ONLY when the value itself contains a comma.


For example:
character,김상현,Kim Sang-hyu
character,갈편제,Gale Hardest  
character,디히릿 아데,Dihirit Ade

Only include terms that actually appear in the text.

Text to analyze:
{text_sample}""")  
        
        # Initialize custom API endpoint variables
        self.openai_base_url_var = self.config.get('openai_base_url', '')
        self.custom_image_edit_endpoint_var = self.config.get('custom_image_edit_endpoint', '')
        self.custom_image_edit_system_prompt_var = self.config.get(
            'custom_image_edit_system_prompt',
            self.config.get(
                'custom_image_edit_prompt',
                "Remove the written text from this image. Redraw only the areas it covered to match their immediate surroundings. "
                "Keep speech-bubble outlines, text-box borders, panel frames, other artwork, and image dimensions unchanged. "
                "Do not add text. Return only the edited image."
            )
        )

        self.custom_image_edit_user_prompt_var = self.config.get('custom_image_edit_user_prompt', '')
        self.custom_image_edit_prompt_var = self.custom_image_edit_system_prompt_var
        _raw_fp = self.config.get('custom_image_edit_full_page_output', 10)
        if isinstance(_raw_fp, bool):
            _raw_fp = 100 if _raw_fp else 10
        self.custom_image_edit_full_page_output_var = max(0, min(100, int(_raw_fp)))
        self.manga_disable_inpaint_performance_mode_var = bool(self.config.get('manga_disable_inpaint_performance_mode', False))
        self.use_custom_image_edit_endpoint_var = self.config.get(
            'use_custom_image_edit_endpoint',
            False
        )
        self.openai_tts_endpoint_var = self.config.get('openai_tts_endpoint', '')
        self.tts_voice_var = self.config.get('tts_voice', '')
        self.groq_base_url_var = self.config.get('groq_base_url', '')
        self.fireworks_base_url_var = self.config.get('fireworks_base_url', '')
        self.use_custom_openai_endpoint_var = self.config.get('use_custom_openai_endpoint', False)
        self.authza_use_general_api_var = bool(
            self.config.get('authza_use_general_api', False)
        )
        # Polling runs in this GUI process, not only in translation workers.
        # Apply the persisted AuthZA mode immediately so login-plan and General
        # API catalogs cannot be confused after an application restart.
        try:
            from glm_proxy import set_general_api_mode
            set_general_api_mode(self.authza_use_general_api_var)
        except Exception:
            os.environ['AUTHZA_USE_GENERAL_API'] = (
                '1' if self.authza_use_general_api_var else '0'
            )
        self.custom_prefix_routes = self._normalize_custom_prefix_routes(
            self.config.get('custom_prefix_routes', [])
        )
        self.config['custom_prefix_routes'] = self.custom_prefix_routes
        self._sync_custom_prefix_routes_env()
        
        # Initialize metadata/batch variables the same way.
        #
        # Seed the default field selection (description + subject) when
        # ``translate_metadata_fields`` hasn't been written to config yet
        # \u2014 previously the attribute sat at ``{}`` until the user
        # opened "Configure Metadata Translation" and clicked Save,
        # and every downstream gate (``any(translate_metadata_fields
        # .values())`` in :func:`metadata_batch_translator
        # .enhance_epub_compiler` + :meth:`EPUBCompiler.compile`) read
        # that empty dict as "nothing to translate" and silently
        # skipped the metadata pass. The Book Title path stayed
        # working because it's driven by a separate
        # ``translate_book_title_var`` flag. Mirroring the dialog's
        # ``default_enabled_fields`` here makes a fresh install /
        # stale config behave the way the UI advertises. A config
        # entry with any recognisable field keys \u2014 including all-
        # False, which means the user explicitly deselected \u2014 is
        # left untouched so we never override a real choice. The
        # ``_per_epub`` scaffolding key is ignored for this purpose
        # so a config that only holds per-EPUB overrides still
        # triggers the seeding path.
        _cfg_mf = self.config.get('translate_metadata_fields', None)
        if not isinstance(_cfg_mf, dict):
            _cfg_mf = {}
        _configured_keys = {k for k in _cfg_mf.keys() if k != '_per_epub'}
        if not _configured_keys:
            _cfg_mf = dict(_cfg_mf)
            _cfg_mf.update({'description': True, 'subject': True})
            # Persist immediately so the env-var payload + any
            # downstream consumer sees the same defaults the dialog
            # would have written, without waiting for the next
            # ``save_config`` tick.
            try:
                self.config['translate_metadata_fields'] = _cfg_mf
            except Exception:
                pass
        # Title is now configurable alongside the other metadata fields.  It
        # defaults on so existing configs retain the former always-translate
        # behavior until the user explicitly disables it.
        if 'title' not in _cfg_mf:
            _cfg_mf = dict(_cfg_mf)
            _cfg_mf['title'] = True
            self.config['translate_metadata_fields'] = _cfg_mf
        self.translate_metadata_fields = _cfg_mf
        # Initialize metadata translation UI and prompts
        self._hook_metadata_defaults()
        self.batch_translate_headers_var = self.config.get('batch_translate_headers', True)
        self.headers_per_batch_var = self.config.get('headers_per_batch', '-1')
        self.toc_ncx_per_batch_var = self.config.get('toc_ncx_per_batch', '-1')
        try:
            self.failed_translation_retry_attempts_var = min(
                20,
                max(
                    0,
                    int(
                        self.config.get(
                            'failed_translation_retry_attempts', 3
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.failed_translation_retry_attempts_var = 3
        self.partial_b2_entries_per_request_var = self.config.get('partial_b2_entries_per_request', '-1')
        self.update_html_headers_var = self.config.get('update_html_headers', True)
        self.save_header_translations_var = self.config.get('save_header_translations', True)
        self.ignore_header_var = self.config.get('ignore_header', False)
        self.allow_ai_markdown_headers_var = self.config.get('allow_ai_markdown_headers', False)
        self.skip_title_tag_translation_var = bool(
            self.config.get('skip_title_tag_translation', False)
        )
        # Retain the inverse legacy value for compatibility with old readers.
        self.use_title_var = not self.skip_title_tag_translation_var
        self.remove_duplicate_h1_p_var = self.config.get('remove_duplicate_h1_p', False)
        self.use_sorted_fallback_var = self.config.get('use_sorted_fallback', False)  # Disabled by default
        self.attach_css_to_chapters_var = self.config.get('attach_css_to_chapters', False)
        self.epub_use_html_method_var = self.config.get('epub_use_html_method', False)
        # CSS override path — initialize from config so the converter can
        # pick it up without requiring the Other Settings dialog to be opened.
        self.epub_css_override_path_var = self.config.get('epub_css_override_path', '')
        if self.epub_css_override_path_var:
            os.environ['EPUB_CSS_OVERRIDE_PATH'] = self.epub_css_override_path_var
        
        # Retain exact source extension and disable 'response_' prefix
        self.retain_source_extension_var = self.config.get('retain_source_extension', False)
        self.download_remote_image_urls_var = self.config.get(
            'download_remote_image_urls', False
        )
        try:
            self.remote_image_download_workers_var = max(
                1, min(32, int(float(
                    self.config.get('remote_image_download_workers', 4)
                )))
            )
        except (TypeError, ValueError):
            self.remote_image_download_workers_var = 4
        try:
            self.remote_image_download_interval_var = max(
                0.0, min(60.0, float(
                    self.config.get('remote_image_download_interval', 0.5)
                ))
            )
        except (TypeError, ValueError):
            self.remote_image_download_interval_var = 0.5
        
        # Initialize extraction settings (from Other Settings)
        self.force_bs_for_traditional_var = self.config.get('force_bs_for_traditional', True)
        # Fix Empty Attribute Tags toggles (persisted in config). The extraction
        # fix is now canonical for both BeautifulSoup and html2text; keep the
        # old BS flag mirrored for compatibility with older code paths/configs.
        self.fix_empty_attr_tags_epub_var = self.config.get('fix_empty_attr_tags_epub', False)
        self.fix_empty_attr_tags_extract_var = bool(
            self.config.get('fix_empty_attr_tags_extract', False)
            or self.config.get('fix_empty_attr_tags_bs', False)
        )
        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var
        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var
        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)
        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)
        self.output_sdlxliff_var = self.config.get('output_sdlxliff', True)
        self.output_md_var = self.config.get('output_md', False)
        self.output_txt_var = self.config.get('output_txt', False)
        _raw_ns = self.config.get('number_spacing_token_fix', '0')
        if isinstance(_raw_ns, bool):
            _raw_ns = '1' if _raw_ns else '0'
        self.number_spacing_token_fix_var = str(_raw_ns)
        # Sync environment on startup for downstream components
        os.environ['FIX_EMPTY_ATTR_TAGS_EPUB'] = '1' if self.fix_empty_attr_tags_epub_var else '0'
        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'
        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'
        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'
        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'
        os.environ['OUTPUT_SDLXLIFF'] = '1' if self.output_sdlxliff_var else '0'
        os.environ['OUTPUT_MD'] = '1' if self.output_md_var else '0'
        os.environ['OUTPUT_TXT'] = '1' if self.output_txt_var else '0'
        os.environ['NUMBER_SPACING_TOKEN_FIX'] = self.number_spacing_token_fix_var
        
        # Graceful stop - wait for in-flight API calls to complete instead of aborting them
        self.graceful_stop_var = self.config.get('graceful_stop', True)
        # Wait for chunks - when graceful stop is active, wait for all chunks of a chapter to complete
        self.wait_for_chunks_var = self.config.get('wait_for_chunks', True)
        try:
            self.dispatch_order_timeout_var = min(
                120,
                max(
                    0,
                    int(float(self.config.get('dispatch_order_timeout', 3))),
                ),
            )
        except (TypeError, ValueError):
            self.dispatch_order_timeout_var = 3
        os.environ['ORDERED_BATCH_DISPATCH_TIMEOUT'] = str(
            self.dispatch_order_timeout_var
        )
        
        # Initialize HTTP/Network tuning variables (from Other Settings)
        self.enable_http_tuning_var = self.config.get('enable_http_tuning', False)
        self.connect_timeout_var = str(self.config.get('connect_timeout', 10))
        self.read_timeout_var = str(self.config.get('read_timeout', 180))
        self.http_pool_connections_var = str(self.config.get('http_pool_connections', 20))
        self.http_pool_maxsize_var = str(self.config.get('http_pool_maxsize', 50))
        self.ignore_retry_after_var = self.config.get('ignore_retry_after', False)
        self.max_retries_var = str(self.config.get('max_retries', 7))
        
        # Initialize anti-duplicate parameters (from Other Settings)
        self.enable_anti_duplicate_var = self.config.get('enable_anti_duplicate', False)
        self.top_p_var = self.config.get('top_p', 1.0)
        self.min_p_var = self.config.get('min_p', 0.0)
        self.bypass_min_p_allowlist_var = self.config.get('bypass_min_p_allowlist', False)
        self.top_k_var = self.config.get('top_k', 0)
        self.frequency_penalty_var = self.config.get('frequency_penalty', 0.0)
        self.presence_penalty_var = self.config.get('presence_penalty', 0.0)
        self.repetition_penalty_var = self.config.get('repetition_penalty', 1.0)
        self.candidate_count_var = self.config.get('candidate_count', 1)
        self.custom_stop_sequences_var = self.config.get('custom_stop_sequences', '')
        self.logit_bias_enabled_var = self.config.get('logit_bias_enabled', False)
        self.logit_bias_strength_var = self.config.get('logit_bias_strength', 1.0)
        self.bias_common_words_var = self.config.get('bias_common_words', False)
        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)

        
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        # NOTE: on_model_change() is already called at the end of
        # _create_model_section(); no need for a delayed duplicate call here.

        
        
        # Async processing settings
        self.async_wait_for_completion_var = self.config.get('async_wait_for_completion', False)
        self.async_poll_interval_var = self.config.get('async_poll_interval', 60)
        
        # PDF settings
        self.pdf_output_format_var = self.config.get('pdf_output_format', 'pdf')
        _stored_pdf_render_mode = self.config.get('pdf_render_mode', 'fast_semantic')
        if (
            str(_stored_pdf_render_mode).lower() in ('xhtml', 'html')
            and not self.config.get('pdf_fast_engine_migrated', False)
        ):
            # XHTML was the historical default. Migrate it once so existing
            # installations receive the new default while preserving an
            # explicit Legacy Layout choice made afterward.
            _stored_pdf_render_mode = 'fast_semantic'
            self.config['pdf_render_mode'] = 'fast_semantic'
            self.config['pdf_fast_engine_migrated'] = True
        self.pdf_render_mode_var = _stored_pdf_render_mode
        self.pdf_use_toc_sections_var = self.config.get('pdf_use_toc_sections', True)
        self.pdf_async_page_threshold_var = str(self.config.get('pdf_async_page_threshold', '100'))
        self.pdf_extraction_workers_var = str(
            self.config.get('pdf_extraction_workers', 'auto') or 'auto'
        )
        self.pdf_paragraph_alignment_var = str(
            self.config.get('pdf_paragraph_alignment', 'source') or 'source'
        ).strip().lower()
        if self.pdf_paragraph_alignment_var == 'centre':
            self.pdf_paragraph_alignment_var = 'center'
        if self.pdf_paragraph_alignment_var not in {
            'source', 'left', 'center', 'right'
        }:
            self.pdf_paragraph_alignment_var = 'source'
        self.pdf_header_alignment_var = str(
            self.config.get('pdf_header_alignment', 'source') or 'source'
        ).strip().lower()
        if self.pdf_header_alignment_var == 'centre':
            self.pdf_header_alignment_var = 'center'
        if self.pdf_header_alignment_var not in {
            'source', 'left', 'center', 'right'
        }:
            self.pdf_header_alignment_var = 'source'
        self.pdf_paragraph_justification_var = str(
            self.config.get('pdf_paragraph_justification', 'source') or 'source'
        ).strip().lower()
        if self.pdf_paragraph_justification_var == 'justified':
            self.pdf_paragraph_justification_var = 'justify'
        if self.pdf_paragraph_justification_var not in {
            'source', 'justify', 'none'
        }:
            self.pdf_paragraph_justification_var = 'source'
        self.pdf_rtl_paragraph_layout_var = bool(
            self.config.get('pdf_rtl_paragraph_layout', False)
        )
        os.environ['PDF_EXTRACTION_WORKERS'] = self.pdf_extraction_workers_var
        os.environ['PDF_PARAGRAPH_ALIGNMENT'] = self.pdf_paragraph_alignment_var
        os.environ['PDF_HEADER_ALIGNMENT'] = self.pdf_header_alignment_var
        os.environ['PDF_PARAGRAPH_JUSTIFICATION'] = (
            self.pdf_paragraph_justification_var
        )
        os.environ['PDF_RTL_PARAGRAPH_LAYOUT'] = (
            '1' if self.pdf_rtl_paragraph_layout_var else '0'
        )
        
         # Enhanced filtering level
        if not hasattr(self, 'enhanced_filtering_var'):
            self.enhanced_filtering_var = self.config.get('enhanced_filtering', 'smart')
        
        # Preserve structure toggle
        if not hasattr(self, 'enhanced_preserve_structure_var'):
            self.enhanced_preserve_structure_var = self.config.get('enhanced_preserve_structure', True)
        
        # Single line break toggle
        if not hasattr(self, 'enhanced_single_line_break_var'):
            self.enhanced_single_line_break_var = self.config.get('enhanced_single_line_break', False)

        # Convert Markdown-generated <br> boundaries to sibling paragraphs.
        if not hasattr(self, 'convert_br_to_paragraphs_var'):
            self.convert_br_to_paragraphs_var = self.config.get(
                'convert_br_to_paragraphs', True
            )
        os.environ['CONVERT_BR_TO_PARAGRAPHS'] = (
            '1' if self.convert_br_to_paragraphs_var else '0'
        )

        if not hasattr(self, 'preserve_asterisk_separator_lines_var'):
            self.preserve_asterisk_separator_lines_var = self.config.get(
                'preserve_asterisk_separator_lines', True
            )
        os.environ['PRESERVE_ASTERISK_SEPARATOR_LINES'] = (
            '1' if self.preserve_asterisk_separator_lines_var else '0'
        )

        # Skip markdown -> HTML tag conversion toggle (default OFF)
        if not hasattr(self, 'skip_markdown_to_html_var'):
            self.skip_markdown_to_html_var = self.config.get('skip_markdown_to_html', False)
        os.environ['SKIP_MARKDOWN_TO_HTML'] = '1' if self.skip_markdown_to_html_var else '0'

        # Markdown2 converter toggle (default OFF to avoid legacy converter)
        if not hasattr(self, 'use_markdown2_converter_var'):
            self.use_markdown2_converter_var = self.config.get('use_markdown2_converter', False)

    def _init_default_prompt_profiles(self):
        """default_* prompt attributes and the built-in prompt profiles (TranslatorGUI.__init__ block)."""
        # Default prompts
        self.default_translation_chunk_prompt = DEFAULT_TRANSLATION_CHUNK_PROMPT
        self.default_image_chunk_prompt = DEFAULT_IMAGE_CHUNK_PROMPT
        self.default_image_only_title_tag_system_prompt = (
            DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT
        )
        self.default_vision_ocr_prompt = DEFAULT_VISION_OCR_PROMPT
        self.default_vision_ocr_user_prompt = DEFAULT_VISION_OCR_USER_PROMPT
        self.default_vision_ocr_combined_context_prompt = DEFAULT_VISION_OCR_COMBINED_CONTEXT_PROMPT
        self.default_vision_ocr_translation_user_prompt = DEFAULT_VISION_OCR_TRANSLATION_USER_PROMPT
        from subtitle_processor import DEFAULT_SUBTITLE_TRANSLATION_PROMPT

        self.default_prompts = {
            "Universal": (
                "You are a professional novel translator. You MUST translate the following text to {target_lang}.\n"
                "- You MUST output ONLY in {target_lang}. No other languages are permitted.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, <img>, <ruby>, etc.\n"
                "{split_marker_instruction}\n"
                "- Preserve any Markdown formatting if present (e.g., headings '#', '##', '###', bold '**text**', italic '*text*', lists '- item'/'1. item', blockquotes '> quote', links '[text](url)', images '![alt](url)', inline code '`code`').\n"
                "- Preserve every line-break boundary and every bullet-point marker exactly as they appear in the source. Do not add, remove, merge, split, or move line breaks or bullet markers.\n"
                "- Maintain the original meaning, tone, and style.\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process: track omitted or ambiguous subjects/pronouns from surrounding context, titles, relationships, dialogue, and repeated mentions so pronouns stay consistent instead of defaulting to 'he', 'she', or 'it'.\n"
                "- Output ONLY the translated text in {target_lang}. Do not add any explanations, notes, or conversational filler.\n"
            ),
            "Refinement": self.default_refinement_system_prompt,
            "Korean_BeautifulSoup": (
                "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Retain Korean familial address terms in romanized form rather than translating them (examples: oppa, eonni, hyung, noona, omma, appa, halabeoji, halmeoni), preserving their nuance and relationship context instead of converting them to English equivalents like brother, sister, mom, or dad.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since the Korean language frequently omits subjects and pronouns. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (\" \", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, <ruby>, etc.\n"
                "- Do not leave stray raw text like \"ㅋ\", They must be translated to an english equivalent. \n"
                "{split_marker_instruction}\n"
            ),
            "Japanese_BeautifulSoup": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Retain Japanese familial address terms in romanized form rather than translating them (examples: onii-san, onii-sama, onii-chan, onii-tan, onee-san, onee-sama, okaasan, otousan, imouto, ani, ane), preserving their nuance and level of affection instead of converting them to English equivalents like brother or sister.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since Japanese frequently omits subjects and pronouns. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」 and 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, <ruby>, etc.\n"
                "- Do not leave stray raw text like \"笑\", They must be translated to an english equivalent. \n"
                "{split_marker_instruction}\n"
            ),
            "Chinese_BeautifulSoup": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Chinese titles and respectful forms of address in romanized form, including but not limited to: laoban, laoshi, shifu, xiaojie, xiansheng, taitai, daren, qianbei. For archaic/classical Chinese respectful forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 法术 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since Chinese frequently omits subjects and pronouns, and spoken Chinese pronouns do not reliably indicate gender. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」 for dialogue, 《》 for titles) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, <ruby>, etc.\n"
                "- Do not leave stray raw text like \"哈\", They must be translated to an english equivalent. \n"
                "{split_marker_instruction}\n"
            ),
            "Korean_html2text": (
                "You are a professional Korean to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Retain Korean familial address terms in romanized form rather than translating them (examples: oppa, eonni, hyung, noona, omma, appa, halabeoji, halmeoni), preserving their nuance and relationship context instead of converting them to English equivalents like brother, sister, mom, or dad.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since the Korean language frequently omits subjects and pronouns. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (\" \", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory. When you see [생활관], write [Dormitory]. Do not write [생활관] anywhere in your output - this is forbidden. Apply this rule to every single Asian character - convert them all to English.\n"
                "- Preserve every line-break boundary and every bullet-point marker exactly as they appear in the source. Do not add, remove, merge, split, or move line breaks or bullet markers.\n"
                "- Preserve all Markdown present (e.g., headings '#', '##', '###', bold '**text**', italic '*text*', lists '- item'/'1. item', blockquotes '> quote', links '[text](url)', images '![alt](url)', inline code '`code`').\n"
                "- Preserve any HTML image tags (<img>, <svg>, <picture>, <figure>) and furigana <ruby> tags exactly as they appear (e.g. <ruby>体力<rp>(</rp><rt>HP</rt><rp>)</rp></ruby>). Do not add or preserve any other HTML tags.\n"
                "- Do not leave stray raw text like \"ㅋ\", They must be translated to an english equivalent.\n"
                "{split_marker_instruction}\n"
            ),
            "Japanese_html2text": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Retain Japanese familial address terms in romanized form rather than translating them (examples: onii-san, onii-sama, onii-chan, onii-tan, onee-san, onee-sama, okaasan, otousan, imouto, ani, ane), preserving their nuance and level of affection instead of converting them to English equivalents like brother or sister.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since Japanese frequently omits subjects and pronouns. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」 and 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve every line-break boundary and every bullet-point marker exactly as they appear in the source. Do not add, remove, merge, split, or move line breaks or bullet markers.\n"
                "- Preserve all Markdown present (e.g., headings '#', '##', '###', bold '**text**', italic '*text*', lists '- item'/'1. item', blockquotes '> quote', links '[text](url)', images '![alt](url)', inline code '`code`').\n"
                "- Preserve any HTML image tags (<img>, <svg>, <picture>, <figure>) and furigana <ruby> tags exactly as they appear (e.g. <ruby>体力<rp>(</rp><rt>HP</rt><rp>)</rp></ruby>). Do not add or preserve any other HTML tags.\n"
                "- Do not leave stray raw text like \"笑\", They must be translated to an english equivalent.\n"
                "{split_marker_instruction}\n"
            ),
            "Chinese_html2text": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Chinese titles and respectful forms of address in romanized form, including but not limited to: laoban, laoshi, shifu, xiaojie, xiansheng, taitai, daren, qianbei. For archaic/classical Chinese respectful forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 法术 = magic).\n"
                "- Strictly follow a Subject Tracking & Pronoun Resolution process, since Chinese frequently omits subjects and pronouns, and spoken Chinese pronouns do not reliably indicate gender. DO NOT default to 'he' or 'it' for omitted subjects. Instead, actively track the acting subject from preceding sentences. Deduce gender from context clues (titles, relationships, dialogue) and maintain absolute pronoun consistency for each character throughout the scene.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」 for dialogue, 《》 for titles) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve every line-break boundary and every bullet-point marker exactly as they appear in the source. Do not add, remove, merge, split, or move line breaks or bullet markers.\n"
                "- Preserve all Markdown present (e.g., headings '#', '##', '###', bold '**text**', italic '*text*', lists '- item'/'1. item', blockquotes '> quote', links '[text](url)', images '![alt](url)', inline code '`code`').\n"
                "- Preserve any HTML image tags (<img>, <svg>, <picture>, <figure>) and furigana <ruby> tags exactly as they appear (e.g. <ruby>体力<rp>(</rp><rt>HP</rt><rp>)</rp></ruby>). Do not add or preserve any other HTML tags.\n"
                "- Do not leave stray raw text like \"哈\", They must be translated to an english equivalent.\n"
                "{split_marker_instruction}\n"
            ),
            "Manga_JP": (
                "You are a professional Japanese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」, 『』) as-is without converting to English quotes.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_KR": (
                "You are a professional Korean to English Manhwa translator.\n"
                "You have both the image of the Manhwa panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (\" \", ' ', 「」, 『』) as-is without converting to English quotes.\\n\\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_CN": (
                "You are a professional Chinese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」, 『』) as-is without converting to English quotes.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Glossary_Editor": (
                "I have a messy character glossary from a Korean web novel that needs to be cleaned up and restructured. Please Output only JSON entries while creating a clean JSON glossary with the following requirements:\n"
                "1. Merge duplicate character entries - Some characters appear multiple times (e.g., Noah, Ichinose family members).\n"
                "2. Separate mixed character data - Some entries incorrectly combine multiple characters' information.\n"
                "3. Use 'Korean = English' format - Replace all parentheses with equals signs (e.g., '이로한 = Lee Rohan' instead of '이로한 (Lee Rohan)').\n"
                "4. Merge original_name fields - Combine original Korean names with English names in the name field.\n"
                "5. Remove empty fields - Don't include empty arrays or objects.\n"
                "6. Fix gender inconsistencies - Correct based on context from aliases.\n"

            ),
            "RPGMaker_GTool": (
                "You are a game translator. Translate every numbered entry below to {target_lang}. "
                "Output ONLY the [N] tag followed by the translation. "
                "No original text, no arrows, no quotes, no commentary. "
                "Do NOT skip any entry. Do NOT leave any entry blank.\n\n"
                "EXAMPLE INPUT:\n"
                "[1] 薬草\n"
                "[2] HPを少し回復する。\n"
                "[3] %1は%2を唱えた！\n"
                "[4] 逃げられない！\n"
                "[5] 薬草\n\n"
                "CORRECT OUTPUT:\n"
                "[1] Herb\n"
                "[2] Restores a small amount of HP.\n"
                "[3] %1 cast %2!\n"
                "[4] Cannot escape!\n"
                "[5] Herb\n\n"
                "WRONG OUTPUT (do NOT do this):\n"
                "[1] 薬草 -> Herb\n"
                "[2] \"Restores a small amount of HP.\"\n"
                "[3] %1は%2を唱えた！ — %1 cast %2!\n"
                "[4]\n"
                "[5] Same as [1]\n\n"
                "RULES:\n"
                "- One [N] per line. Every [N] on its own new line.\n"
                "- EVERY entry MUST have a full translation. Even if entries are duplicates, write the full translation each time.\n"
                "- NEVER write 'repeat of', 'same as', 'see above', or reference another entry number. Always write the full translated text.\n"
                "- NEVER include the original text alongside the translation. NEVER use '->' arrows.\n"
                "- NEVER wrap translations in quotes.\n"
                "- NEVER leave an entry empty or skip it.\n"
                "- Keep %1, %2, \\V[1], \\N[2], \\C[3] and other RPG Maker codes exactly as-is.\n"
                "- If an entry has multiple lines (newlines), keep the same number of lines.\n"
                "- No explanations. No original text. Just [N] and the translation.\n"
            ),
            "RPGMaker_GTool_Image": (
                "You are a game UI image editor specializing in RPG Maker games.\n\n"
                "TASK:\n"
                "Generate a new version of this game image with all visible text translated to {target_lang}.\n\n"
                "RULES:\n"
                "- Translate ALL readable text in the image (menus, labels, titles, buttons, tooltips).\n"
                "- Preserve the original visual style exactly: background art, colors, gradients, effects, and layout.\n"
                "- Match the original font style as closely as possible (weight, size, shadow, outline, glow).\n"
                "- Keep text positioning identical — translated text should occupy the same regions.\n"
                "- If text is part of a decorative element (e.g. stylized title logo), recreate the decoration with the translated text.\n"
                "- Do NOT add, remove, or reposition any non-text visual elements.\n"
                "- Do NOT add watermarks, signatures, or any extra markings.\n"
                "- Output the translated image at the same resolution as the input.\n"
            ),
            "NanoBanana_Image": (
                "This is an image editing task. "
                "Edit this image by replacing all foreign-language text with its {target_lang} translation. "
                "Do NOT return plain text or OCR — you MUST return the generated edited image. "
                "If the image has no translatable text, reply exactly: No\n"
            ),
            "Original": "Return everything exactly as seen on the source.",
            "SDLXLIFF Editing v2": (
                "You are editing SDLXLIFF JSON batch records. Translate each source value to {target_lang}.\n"
                "- Input is a JSON array of objects with id and source fields.\n"
                "- Output only a valid JSON array. No markdown fences, explanations, XML wrappers, or extra fields.\n"
                "- Each output object must have exactly these fields: id and target.\n"
                "- Preserve every input id exactly once and in the same order.\n"
                "- Translate only source into target; do not output XLIFF tags, comments, or notes.\n"
                "- Preserve every placeholder token exactly as written, including tokens like [[XLIFF_TAG_000001_0000]].\n"
                "- Do not add, remove, duplicate, reorder, or translate placeholder tokens.\n"
                "- Preserve variables, formatting markers, accelerator keys, punctuation that functions as markup, and line breaks where meaningful.\n"
            ),
            "Subtitle Translation": DEFAULT_SUBTITLE_TRANSLATION_PROMPT,
        }

    def _init_watchdog_dir(self):
        """Export GLOSSARION_WATCHDOG_DIR (TranslatorGUI.__init__, right after _setup_gui)."""
        # Shared watchdog directory for cross-process API tracking (e.g., glossary subprocess)
        try:
            import tempfile
            base_dir = os.path.join(tempfile.gettempdir(), "glossarion_watchdog")
            unique_dir = f"{os.getpid()}_{int(time.time() * 1000)}"
            watchdog_dir = os.path.join(base_dir, unique_dir)
            os.makedirs(watchdog_dir, exist_ok=True)
            os.environ["GLOSSARION_WATCHDOG_DIR"] = watchdog_dir
        except Exception:
            pass

    def _init_default_prompts(self):
        """Initialize all default prompt templates"""
        # Glossary prompt defaults — single source of truth in extract_glossary_from_epub
        try:
            from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT, DEFAULT_AUTO_GLOSSARY_PROMPT
            from glossary_refinement import DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT, DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT
            self.default_manual_glossary_prompt = DEFAULT_GLOSSARY_PROMPT
            self.default_unified_auto_glosary_prompt3 = DEFAULT_AUTO_GLOSSARY_PROMPT
            self.default_glossary_refinement_system_prompt = DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
            self.default_glossary_refinement_user_prompt = DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT
        except ImportError:
            self.default_manual_glossary_prompt = ""
            self.default_unified_auto_glosary_prompt3 = ""
            self.default_glossary_refinement_system_prompt = ""
            self.default_glossary_refinement_user_prompt = ""
        
        self.default_rolling_summary_system_prompt = DEFAULT_ROLLING_SUMMARY_SYSTEM_PROMPT
        
        # Default assistant prompt (empty by default - user can optionally set this to prefill)
        self.default_assistant_prompt = DEFAULT_ASSISTANT_PROMPT

        self.default_refinement_system_prompt = DEFAULT_REFINEMENT_SYSTEM_PROMPT
        self.default_refinement_user_prompt = DEFAULT_REFINEMENT_USER_PROMPT
        self.default_refinement_qa_issue_prompt = DEFAULT_REFINEMENT_QA_ISSUE_PROMPT
        self.default_refinement_full_with_raw_system_prompt = DEFAULT_REFINEMENT_FULL_WITH_RAW_SYSTEM_PROMPT
        self.default_refinement_full_with_raw_user_prompt = DEFAULT_REFINEMENT_FULL_WITH_RAW_USER_PROMPT
        self.default_refinement_full_with_raw_raw_header = DEFAULT_REFINEMENT_FULL_WITH_RAW_RAW_HEADER
        self.default_refinement_full_with_raw_raw_footer = DEFAULT_REFINEMENT_FULL_WITH_RAW_RAW_FOOTER
        self.default_refinement_failed_system_prompt = DEFAULT_REFINEMENT_FAILED_SYSTEM_PROMPT
        self.default_refinement_failed_user_prompt = DEFAULT_REFINEMENT_FAILED_USER_PROMPT
        self.default_refinement_partial_system_prompt = DEFAULT_REFINEMENT_PARTIAL_SYSTEM_PROMPT
        self.default_refinement_partial_user_prompt = DEFAULT_REFINEMENT_PARTIAL_USER_PROMPT
        self.default_refinement_partial_b_system_prompt = DEFAULT_REFINEMENT_PARTIAL_B_SYSTEM_PROMPT
        self.default_refinement_partial_b_user_prompt = DEFAULT_REFINEMENT_PARTIAL_B_USER_PROMPT
        self.default_refinement_partial_b2_system_prompt = DEFAULT_REFINEMENT_PARTIAL_B2_SYSTEM_PROMPT
        self.default_refinement_partial_b2_user_prompt = DEFAULT_REFINEMENT_PARTIAL_B2_USER_PROMPT
        
        self.default_rolling_summary_user_prompt = DEFAULT_ROLLING_SUMMARY_USER_PROMPT

    def _init_variables(self):
        """Initialize all configuration variables"""
        # Load saved prompts
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt3', self.default_manual_glossary_prompt)
        # Note: Ignoring old 'auto_glosary_prompt2' key to force update to new prompt
        # Also treat empty strings as missing to ensure users get the new default
        unified_prompt_temp = self.config.get('unified_auto_glosary_prompt3', self.default_unified_auto_glosary_prompt3)
        if not unified_prompt_temp or not unified_prompt_temp.strip():
            self.unified_auto_glosary_prompt3 = self.default_unified_auto_glosary_prompt3
        else:
            self.unified_auto_glosary_prompt3 = unified_prompt_temp
        self.glossary_refinement_system_prompt = self.config.get(
            'glossary_refinement_system_prompt',
            getattr(self, 'default_glossary_refinement_system_prompt', '')
        )
        if not self.glossary_refinement_system_prompt or not str(self.glossary_refinement_system_prompt).strip():
            self.glossary_refinement_system_prompt = getattr(self, 'default_glossary_refinement_system_prompt', '')
        self.glossary_refinement_user_prompt = self.config.get(
            'glossary_refinement_user_prompt',
            getattr(self, 'default_glossary_refinement_user_prompt', '')
        )
        self.refinement_system_prompt = self.config.get(
            'refinement_system_prompt',
            getattr(self, 'default_refinement_system_prompt', '')
        )
        if not self.refinement_system_prompt or not str(self.refinement_system_prompt).strip():
            self.refinement_system_prompt = getattr(self, 'default_refinement_system_prompt', '')
        self.refinement_user_prompt = self.config.get(
            'refinement_user_prompt',
            getattr(self, 'default_refinement_user_prompt', '')
        )
        self.refinement_full_with_raw_system_prompt = self.config.get(
            'refinement_full_with_raw_system_prompt',
            getattr(self, 'default_refinement_full_with_raw_system_prompt', self.refinement_system_prompt)
        )
        if not self.refinement_full_with_raw_system_prompt or not str(self.refinement_full_with_raw_system_prompt).strip():
            self.refinement_full_with_raw_system_prompt = getattr(
                self,
                'default_refinement_full_with_raw_system_prompt',
                self.refinement_system_prompt,
            )
        self.refinement_full_with_raw_user_prompt = self.config.get(
            'refinement_full_with_raw_user_prompt',
            getattr(self, 'default_refinement_full_with_raw_user_prompt', '')
        )
        if 'refinement_full_with_raw_raw_header' in self.config:
            self.refinement_full_with_raw_raw_header = str(
                self.config.get('refinement_full_with_raw_raw_header', '') or ''
            )
        else:
            self.refinement_full_with_raw_raw_header = getattr(
                self, 'default_refinement_full_with_raw_raw_header', ''
            )
        if 'refinement_full_with_raw_raw_footer' in self.config:
            self.refinement_full_with_raw_raw_footer = str(
                self.config.get('refinement_full_with_raw_raw_footer', '') or ''
            )
        else:
            self.refinement_full_with_raw_raw_footer = getattr(
                self, 'default_refinement_full_with_raw_raw_footer', ''
            )
        self.refinement_failed_system_prompt = self.config.get(
            'refinement_failed_system_prompt',
            getattr(self, 'default_refinement_failed_system_prompt', self.refinement_system_prompt)
        )
        if not self.refinement_failed_system_prompt or not str(self.refinement_failed_system_prompt).strip():
            self.refinement_failed_system_prompt = getattr(self, 'default_refinement_failed_system_prompt', self.refinement_system_prompt)
        self.refinement_failed_user_prompt = self.config.get(
            'refinement_failed_user_prompt',
            getattr(self, 'default_refinement_failed_user_prompt', '')
        )
        self.refinement_partial_system_prompt = self.config.get(
            'refinement_partial_system_prompt',
            getattr(self, 'default_refinement_partial_system_prompt', self.refinement_system_prompt)
        )
        if not self.refinement_partial_system_prompt or not str(self.refinement_partial_system_prompt).strip():
            self.refinement_partial_system_prompt = getattr(self, 'default_refinement_partial_system_prompt', self.refinement_system_prompt)
        self.refinement_partial_user_prompt = self.config.get(
            'refinement_partial_user_prompt',
            getattr(self, 'default_refinement_partial_user_prompt', '')
        )
        if 'refinement_partial_b_system_prompt' in self.config:
            self.refinement_partial_b_system_prompt = str(self.config.get('refinement_partial_b_system_prompt', '') or '')
        else:
            self.refinement_partial_b_system_prompt = getattr(self, 'default_refinement_partial_b_system_prompt', '')
        self.refinement_partial_b_user_prompt = self.config.get(
            'refinement_partial_b_user_prompt',
            getattr(self, 'default_refinement_partial_b_user_prompt', '')
        )
        if 'refinement_partial_b2_system_prompt' in self.config:
            self.refinement_partial_b2_system_prompt = str(self.config.get('refinement_partial_b2_system_prompt', '') or '')
        else:
            self.refinement_partial_b2_system_prompt = getattr(self, 'default_refinement_partial_b2_system_prompt', '')
        self.refinement_partial_b2_user_prompt = self.config.get(
            'refinement_partial_b2_user_prompt',
            getattr(self, 'default_refinement_partial_b2_user_prompt', '')
        )
        self.rolling_summary_system_prompt = self.config.get('rolling_summary_system_prompt', self.default_rolling_summary_system_prompt)
        self.rolling_summary_user_prompt = self.config.get('rolling_summary_user_prompt', self.default_rolling_summary_user_prompt)
        self.append_glossary_prompt = self.config.get('append_glossary_prompt', "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
        self.translation_chunk_prompt = self.config.get('translation_chunk_prompt', self.default_translation_chunk_prompt)
        self.enable_translation_chunk_prompt_var = bool(self.config.get('enable_translation_chunk_prompt', False))
        self.include_previous_chunk_var = bool(self.config.get('include_previous_chunk', False))
        try:
            self.previous_chunk_context_limit_var = max(-1, int(self.config.get('previous_chunk_context_limit', 3)))
        except Exception:
            self.previous_chunk_context_limit_var = 3
        self.translation_chunk_prompt_role_var = str(self.config.get('translation_chunk_prompt_role', 'assistant') or 'assistant').strip().lower()
        if self.translation_chunk_prompt_role_var not in {'system', 'assistant', 'user'}:
            self.translation_chunk_prompt_role_var = 'assistant'
        self.image_chunk_prompt = self.config.get('image_chunk_prompt', self.default_image_chunk_prompt)
        self.image_only_title_tag_system_prompt = str(self.config.get(
            'image_only_title_tag_system_prompt',
            self.default_image_only_title_tag_system_prompt,
        ) or self.default_image_only_title_tag_system_prompt).strip()
        self.vision_ocr_prompt = self.config.get('vision_ocr_prompt', self.default_vision_ocr_prompt)
        if not self.vision_ocr_prompt or not str(self.vision_ocr_prompt).strip():
            self.vision_ocr_prompt = self.default_vision_ocr_prompt
        elif (
            "Preserve the original line breaks as faithfully as possible" in str(self.vision_ocr_prompt)
            or "do not collapse separate visual lines into one paragraph" in str(self.vision_ocr_prompt)
            or "Return only the base source text. Preserve paragraph breaks and intentional textual layout" in str(self.vision_ocr_prompt)
            or "otherwise not a page of readable story text" in str(self.vision_ocr_prompt)
            or "headings with #" in str(self.vision_ocr_prompt)
            or "Do not convert titles, centered text, chapter names, large text, or standalone numbers into Markdown headings" in str(self.vision_ocr_prompt)
            or "Do not add # unless a # character is visibly present in the image" in str(self.vision_ocr_prompt)
            or "If any readable text is present, output it" in str(self.vision_ocr_prompt)
        ):
            self.vision_ocr_prompt = self.default_vision_ocr_prompt
            self.config['vision_ocr_prompt'] = self.default_vision_ocr_prompt
        self.vision_ocr_user_prompt = self.config.get('vision_ocr_user_prompt', self.default_vision_ocr_user_prompt)
        if not self.vision_ocr_user_prompt or not str(self.vision_ocr_user_prompt).strip():
            self.vision_ocr_user_prompt = self.default_vision_ocr_user_prompt
        elif (
            "OCR this image/chunk. Return only the main/base source text." in str(self.vision_ocr_user_prompt)
            or "has no readable story text" in str(self.vision_ocr_user_prompt)
            or "If any readable text is present, output it" in str(self.vision_ocr_user_prompt)
        ):
            self.vision_ocr_user_prompt = self.default_vision_ocr_user_prompt
            self.config['vision_ocr_user_prompt'] = self.default_vision_ocr_user_prompt
        self.vision_ocr_combined_context_prompt = self.config.get('vision_ocr_combined_context_prompt', self.default_vision_ocr_combined_context_prompt)
        if not self.vision_ocr_combined_context_prompt or not str(self.vision_ocr_combined_context_prompt).strip():
            self.vision_ocr_combined_context_prompt = self.default_vision_ocr_combined_context_prompt
        elif (
            "The OCR text below was assembled from" in str(self.vision_ocr_combined_context_prompt)
            or (
                "removing OCR-only duplication from chunk overlap" in str(self.vision_ocr_combined_context_prompt)
                and "{ocr_overlap_instruction}" not in str(self.vision_ocr_combined_context_prompt)
            )
        ):
            self.vision_ocr_combined_context_prompt = self.default_vision_ocr_combined_context_prompt
            self.config['vision_ocr_combined_context_prompt'] = self.default_vision_ocr_combined_context_prompt
        self.vision_ocr_translation_user_prompt = self.config.get('vision_ocr_translation_user_prompt', self.default_vision_ocr_translation_user_prompt)
        if not self.vision_ocr_translation_user_prompt or not str(self.vision_ocr_translation_user_prompt).strip():
            self.vision_ocr_translation_user_prompt = self.default_vision_ocr_translation_user_prompt
        elif "Translate the following OCR text according to the system prompt." in str(self.vision_ocr_translation_user_prompt):
            self.vision_ocr_translation_user_prompt = self.default_vision_ocr_translation_user_prompt
            self.config['vision_ocr_translation_user_prompt'] = self.default_vision_ocr_translation_user_prompt
        # Optional assistant prefill prompt (disabled by default)
        self.assistant_prompt = self.config.get('assistant_prompt', self.default_assistant_prompt)
        
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        self.token_limit_disabled = self.config.get('token_limit_disabled', True)
        self.disable_temperature_var = self._coerce_live_bool(
            self.config.get('disable_temperature', False),
            False,
        )
        self.api_key_visible = False  # Default to hidden
        
        if 'glossary_duplicate_key_mode' not in self.config:
            self.config['glossary_duplicate_key_mode'] = 'fuzzy'
        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = self.config.get('glossary_fuzzy_threshold', 0.90)
        if not hasattr(self, 'glossary_entry_type_filter_mode_var'):
            self.glossary_entry_type_filter_mode_var = self.config.get('glossary_entry_type_filter_mode', 'none')
        
        # Legacy migration: map old conservative_batching flag to new batching_mode/batch_group_size
        if 'batching_mode' not in self.config:
            if self.config.get('conservative_batching', False):
                self.config['batching_mode'] = 'conservative'
            else:
                self.config['batching_mode'] = 'aggressive'
        if 'batch_group_size' not in self.config:
            # legacy hardcoded multiplier was 3
            self.config['batch_group_size'] = str(self.config.get('conservative_batch_multiplier', 3) or 3)

        # Legacy image cap was stored under max_images_per_chapter with a default of 1.
        # The v2 key defaults to -1, which means no per-chapter image limit.
        if 'max_images_per_chapter_v2' not in self.config:
            self.config['max_images_per_chapter_v2'] = '-1'
        self.config.pop('max_images_per_chapter', None)
        # Legacy header-derived output filenames are intentionally disabled.
        self.config['use_header_as_output'] = False
        
        # Create all config variables with helper
        def create_var(var_type, key, default):
            # For PySide6 conversion: just return the value directly
            return self.config.get(key, default)
                
        # Boolean variables
        bool_vars = [
            ('rolling_summary_var', 'use_rolling_summary', False),
            # Controls whether previous source text (user messages) are reused as memory
            ('include_source_in_history_var', 'include_source_in_history', False),
            ('translation_history_rolling_var', 'translation_history_rolling', True),
            ('glossary_history_rolling_var', 'glossary_history_rolling', True),
            ('disable_glossary_history_var', 'disable_glossary_history', True),
            ('glossary_skip_title_header_only_var', 'glossary_skip_title_header_only', True),
            ('glossary_add_minimal_pass_var', 'glossary_add_minimal_pass', False),
            ('translate_book_title_var', 'translate_book_title', True),
            ('skip_txt_title_translation_var', 'skip_txt_title_translation', True),
            ('skip_pdf_title_translation_var', 'skip_pdf_title_translation', False),
            ('include_book_title_glossary_var', 'include_book_title_glossary', False),
            ('enable_auto_glossary_var', 'enable_auto_glossary', True),
            ('append_glossary_var', 'append_glossary', True),
            ('include_gender_context_var', 'include_gender_context', True),
            ('enable_gender_nuance_var', 'enable_gender_nuance', True),
            ('include_description_var', 'include_description', True),
            ('retry_truncated_var', 'retry_truncated', True),
            # Char-ratio truncation (silent truncation detector)
            ('char_ratio_truncation_var', 'char_ratio_truncation_enabled', False),
            ('retry_split_failed_var', 'retry_split_failed', True),
            ('retry_duplicate_var', 'retry_duplicate_bodies', False),
            ('preserve_original_text_var', 'preserve_original_text_on_failure', False),
            ('save_partial_results_var', 'save_partial_results', True),
            ('save_prohibited_results_var', 'save_prohibited_results', False),
            ('disable_empty_safety_heuristic_var', 'disable_empty_safety_heuristic', True),
            ('unknown_finish_as_prohibited_var', 'missing_finish_as_prohibited', self.config.get('unknown_finish_as_prohibited', False)),
            ('disable_qa_marker_checks_var', 'disable_qa_marker_checks', True),
            ('qa_marker_length_limit_var', 'qa_marker_length_limit', '500'),
            ('disable_refusal_checks_var', 'disable_refusal_checks', True),
            ('refusal_pattern_length_limit_var', 'refusal_pattern_length_limit', '1000'),
            # NEW: QA scanning helpers
            ('qa_auto_search_output_var', 'qa_auto_search_output', True),
            ('scan_phase_enabled_var', 'scan_phase_enabled', True),
            ('indefinite_rate_limit_retry_var', 'indefinite_rate_limit_retry', False),
            # Keep existing variables intact
            ('enable_image_translation_var', 'enable_image_translation', False),
            ('process_webnovel_images_var', 'process_webnovel_images', True),
            # REMOVED: ('comprehensive_extraction_var', 'comprehensive_extraction', False),
            ('hide_image_translation_label_var', 'hide_image_translation_label', True),
            ('retry_timeout_var', 'retry_timeout', False),
            ('batch_translation_var', 'batch_translation', True),
            ('enable_chunk_progress_var', 'enable_chunk_progress', True),
            ('disable_epub_gallery_var', 'disable_epub_gallery', True),
            ('skip_non_spine_special_files_var', 'skip_non_spine_special_files', False),
            ('skip_unreferenced_epub_images_var', 'skip_unreferenced_epub_images', False),
            # NEW: Disable automatic cover creation (affects extraction and EPUB cover page)
            ('disable_automatic_cover_creation_var', 'disable_automatic_cover_creation', True),
            ('disable_zero_detection_var', 'disable_zero_detection', True),
            ('use_header_as_output_var', 'use_header_as_output', False),
            ('emergency_restore_var', 'emergency_paragraph_restore', False),
            ('emergency_image_restore_var', 'emergency_image_restore', False),
            ('emergency_glossary_compliance_var', 'emergency_glossary_compliance', False),
            ('contextual_var', 'contextual', False),
            ('enable_watermark_removal_var', 'enable_watermark_removal', True),
            ('save_cleaned_images_var', 'save_cleaned_images', False),
            ('advanced_watermark_removal_var', 'advanced_watermark_removal', False),
            ('enable_decimal_chapters_var', 'enable_decimal_chapters', True),
            ('disable_gemini_safety_var', 'disable_gemini_safety', True),
            ('single_api_image_chunks_var', 'single_api_image_chunks', False),
            ('vision_ocr_batch_translation_var', 'vision_ocr_batch_translation', True),
            ('vision_ocr_skip_translation_var', 'vision_ocr_skip_translation', False),
            ('vision_ocr_keep_images_var', 'vision_ocr_keep_images', False),
            ('enable_image_output_mode_var', 'enable_image_output_mode', False),
            ('enable_video_output_mode_var', 'enable_video_output_mode', False),
            ('enable_audio_output_mode_var', 'enable_audio_output_mode', False),
            ('enable_refinement_output_mode_var', 'enable_refinement_output_mode', False),
            ('enable_streaming_var', 'enable_streaming', False),
            # Preserve streaming logs during batch mode; must be initialized here so save_config
            # keeps the user's choice even if the Other Settings dialog is never opened.
            ('allow_batch_stream_logs_var', 'allow_batch_stream_logs', False),
            ('stream_thinking_logs_var', 'stream_thinking_logs', False),
            ('html2text_escape_snob_var', 'html2text_escape_snob', False),

        ]
        
        for var_name, key, default in bool_vars:
            setattr(self, var_name, create_var(bool, key, default))
        self.use_header_as_output_var = False
        
        self.translate_special_files_var = self.config.get('translate_special_files', False)
        self.skip_image_title_translation_var = bool(self.config.get('skip_image_title_translation', True))
        self.skip_title_tag_translation_var = bool(
            self.config.get('skip_title_tag_translation', False)
        )
        self.use_title_var = not self.skip_title_tag_translation_var
        # Custom special file keywords (comma-separated)
        _DEFAULT_SPECIAL_KEYWORDS = 'title, toc, copyright, preface, nav, message, notice, colophon, dedication, epigraph, foreword, acknowledgment, author, appendix, bibliography'
        _DEFAULT_SPECIAL_EXACT = 'index, glossary, glossary_extension, glossary_unified'
        self.special_file_keywords_var = self.config.get('special_file_keywords', _DEFAULT_SPECIAL_KEYWORDS)
        self._migrate_strict_matching_config()
        self.special_file_exact_var = self._upgrade_special_file_exact(
            self.config.get('special_file_exact', _DEFAULT_SPECIAL_EXACT)
        )
        # Numbered HTML override (must be available before Progress Manager opens)
        self.translate_all_numbered_html_var = self.config.get('translate_all_numbered_html', True)
        self.never_consider_in_between_files_as_special_var = self.config.get(
            'never_consider_in_between_files_as_special', True
        )
        
        # String variables
        str_vars = [
            ('REMOVE_AI_ARTIFACTS_var', 'REMOVE_AI_ARTIFACTS', 'off'),
            ('summary_role_var', 'summary_role', 'system'),
            ('rolling_summary_exchanges_var', 'rolling_summary_exchanges', '5'),
            ('rolling_summary_mode_var', 'rolling_summary_mode', 'replace'),
            # New: how many summaries to retain in append mode
            ('rolling_summary_max_entries_var', 'rolling_summary_max_entries', '5'),
            # New: max tokens for rolling summary generation
            # -1 means: use the main MAX_OUTPUT_TOKENS value
            ('rolling_summary_max_tokens_var', 'rolling_summary_max_tokens', '-1'),

            ('max_retry_tokens_var', 'max_retry_tokens', '-1'),
            ('truncation_retry_attempts_var', 'truncation_retry_attempts', '3'),
            # Char-ratio truncation (silent truncation detector)
            ('char_ratio_truncation_percent_var', 'char_ratio_truncation_percent', '50'),
            ('char_ratio_truncation_attempts_var', 'char_ratio_truncation_attempts', '1'),
            ('char_ratio_min_output_chars_var', 'char_ratio_min_output_chars', '100'),
            ('split_failed_retry_attempts_var', 'split_failed_retry_attempts', '1'),
            ('duplicate_lookback_var', 'duplicate_lookback_chapters', '5'),
            ('glossary_min_frequency_var', 'glossary_min_frequency', '2'),
            ('glossary_max_names_var', 'glossary_max_names', '50'),
            ('glossary_max_titles_var', 'glossary_max_titles', '30'),
            ('context_window_size_var', 'context_window_size', '5'),
            ('webnovel_min_height_var', 'webnovel_min_height', '1000'),
            ('max_images_per_chapter_var', 'max_images_per_chapter_v2', '-1'),
            ('image_chunk_height_var', 'image_chunk_height', '1500'),
            ('image_output_resolution_var', 'image_output_resolution', '1K'),
            ('nanogpt_video_duration_var', 'nanogpt_video_duration', '60'),
            ('nanogpt_video_resolution_var', 'nanogpt_video_resolution', '720p'),
            ('chunk_timeout_var', 'chunk_timeout', '1800'),
            ('timeout_retry_attempts_var', 'timeout_retry_attempts', '2'),
            ('batch_size_var', 'batch_size', '5'),
            ('api_queue_var', 'api_queue', '4'),
            ('vision_ocr_batch_size_var', 'vision_ocr_batch_size', '-1'),
            ('batch_mode_var', 'batching_mode', 'aggressive'),
            ('batch_group_size_var', 'batch_group_size', '3'),
            ('chapter_number_offset_var', 'chapter_number_offset', '0'),
            ('compression_factor_var', 'compression_factor', '3.0'),
            # NEW: scanning phase mode (quick-scan/aggressive/ai-hunter/custom)
            ('scan_phase_mode_var', 'scan_phase_mode', 'quick-scan'),
            ('break_split_count_var', 'break_split_count', ''),
            ('auto_glossary_mode_var', 'auto_glossary_mode', 'balanced'),
            ('emergency_glossary_compliance_mode_var', 'emergency_glossary_compliance_mode', 'characters'),
            ('gemini_safety_threshold_var', 'gemini_safety_threshold', 'BLOCK_NONE'),
        ]
        
        for var_name, key, default in str_vars:
            setattr(self, var_name, create_var(str, key, str(default)))
        
        # Backward compat: old configs stored boolean True/False or "0"/"1"
        _rav = self.REMOVE_AI_ARTIFACTS_var
        if _rav in (True, 'True', '1'):
            self.REMOVE_AI_ARTIFACTS_var = 'medium'
        elif _rav in (False, 'False', '0'):
            self.REMOVE_AI_ARTIFACTS_var = 'off'
        elif _rav not in ('off', 'low', 'medium', 'high'):
            self.REMOVE_AI_ARTIFACTS_var = 'off'
        
        # Emergency glossary compliance custom types (list)
        self.emergency_glossary_compliance_custom_types_var = self.config.get('emergency_glossary_compliance_custom_types', [])
        try:
            self.emergency_glossary_compliance_min_chars_var = max(0, min(10, int(self.config.get('emergency_glossary_compliance_min_chars', 3))))
        except (TypeError, ValueError):
            self.emergency_glossary_compliance_min_chars_var = 3
        
        # NEW: Initialize extraction mode variable
        self.extraction_mode_var = self.config.get('extraction_mode', 'smart')
        
        self.book_title_prompt = self.config.get('book_title_prompt', 
            "Translate this book title to {target_lang} while retaining any acronyms:")
        # Initialize book title system prompt
        if 'book_title_system_prompt' not in self.config:
            self.config['book_title_system_prompt'] = "Translate this book title to {target_lang} while retaining any acronyms. Do not output anything other than the translated text."
        
        # Profiles
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        self.config['prompt_profiles'] = self.prompt_profiles
        
        # Ensure Universal profile and all extraction mode profiles exist and are up to date
        # Define profiles that should always be included (in order of priority)
        always_include_profiles = [
            "Universal",
            "Refinement",
            "Korean_BeautifulSoup",
            "Japanese_BeautifulSoup",
            "Chinese_BeautifulSoup",
            "Korean_html2text",
            "Japanese_html2text",
            "Chinese_html2text",
            # RPG Maker GTool profiles
            "RPGMaker_GTool",
            "RPGMaker_GTool_Image",
            "NanoBanana_Image",
            "SDLXLIFF Editing v2",
            "Subtitle Translation",
        ]
        
        # Add missing required profiles while preserving existing profile positions
        if any(profile_name not in self.prompt_profiles 
               for profile_name in always_include_profiles 
               if profile_name in self.default_prompts):
            
            new_profiles = {}
            
            # First pass: Add profiles from always_include_profiles in their priority order
            # but only if they already exist in user's profiles (preserving their original position)
            for profile_name in always_include_profiles:
                if profile_name in self.default_prompts and profile_name in self.prompt_profiles:
                    new_profiles[profile_name] = self.prompt_profiles[profile_name]
            
            # Second pass: Add any missing required profiles that weren't already in user's profiles
            for profile_name in always_include_profiles:
                if profile_name in self.default_prompts and profile_name not in self.prompt_profiles:
                    new_profiles[profile_name] = self.default_prompts[profile_name]
            
            # Third pass: Add all other user profiles that aren't in the required list
            for profile_name, profile_content in self.prompt_profiles.items():
                if profile_name not in new_profiles:
                    new_profiles[profile_name] = profile_content
            
            self.prompt_profiles = new_profiles

        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        if active not in self.prompt_profiles:
            active = next(iter(self.prompt_profiles))
            self.config['active_profile'] = active
        self.profile_var = active
        # Initialize lang_var to the actual target language from config, not the profile name
        # This will be properly synced when update_target_language is called during GUI setup
        self.lang_var = self.config.get('output_language') or self.config.get('glossary_target_language') or 'English'
        
        # Detection mode
        self.duplicate_detection_mode_var = self.config.get('duplicate_detection_mode', 'basic')

    def _update_auto_compression_factor(self):
        """Update compression factor based on output token limit when auto is enabled"""
        try:
            # Check if auto compression is enabled
            if not self.config.get('auto_compression_factor', True):
                if hasattr(self, '_update_compression_token_budget_label'):
                    try:
                        self._update_compression_token_budget_label()
                    except Exception:
                        pass
                return
            
            # Get current output token limit
            output_tokens = int(getattr(self, 'max_output_tokens', 128000))
            
            # Determine input/output token factor based on output token limit
            if output_tokens < 16379:
                factor = 1.5
            elif output_tokens < 32769:
                factor = 2.0
            elif output_tokens < 65536:
                factor = 2.5
            else:  # 65536 or above
                factor = 3.0
            
            # Update the variable
            self.compression_factor_var = str(factor)
            
            # Also update config so it persists
            self.config['compression_factor'] = factor

            if hasattr(self, '_update_compression_token_budget_label'):
                try:
                    self._update_compression_token_budget_label()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error updating auto compression factor: {e}")
        finally:
            self._sync_chunk_size_entry()

    def _sync_chunk_size_entry(self):
        """Show "Auto" or the max input chunk budget in the main-window Chunk Size field."""
        entry = getattr(self, 'chunk_size_entry', None)
        if entry is None:
            return
        if self.config.get('auto_compression_factor', True):
            text = "Auto"
        else:
            budget = self._compression_chunk_budget()
            text = f"{budget:,}" if budget is not None else "Auto"
        try:
            if entry.text() != text:
                entry.blockSignals(True)
                entry.setText(text)
                entry.blockSignals(False)
        except RuntimeError:
            pass

    def _init_gui_backed_state(self):
        """GUI-backed state the desktop section builders create (moved verbatim, desktop order).

        TranslatorGUI.__init__ calls this right before _setup_gui(); HeadlessOwner at the same point.
        The statements only read config/vars and nothing reads these attributes earlier in
        _setup_gui, so hoisting them out of the builders does not change desktop behaviour.
        """
        # create_file_section: Vertex AI Location text entry
        self.vertex_location_var = self.config.get('vertex_ai_location', 'global')
        # create_file_section: Deep scan option for folders
        self.deep_scan_var = self.config.get('deep_scan', False)
        # _create_model_section: Get default model
        default_model = self.config.get('model', 'authgpt/gpt-6-luna')
        self.model_var = default_model
        # _create_settings_section: Context Mode combo
        self.context_mode_var = self._context_mode_from_flags()
        # _create_settings_section: history rolling is always on
        self.translation_history_rolling_var = True
        self.config['translation_history_rolling'] = True

    def _restore_authgem_project_selection(self):
        """Restore the saved AuthGem GCP project at startup (moved verbatim from _create_model_section).

        Seeds authgem_auth's module cache (account 0) and GOOGLE_CLOUD_PROJECT. TranslatorGUI calls
        it from _create_model_section at the original point; _replay_gui_startup_handlers() first.
        """
        # Restore saved project selection on startup
        saved_project = self.config.get('authgem_project', '')
        if saved_project:
            try:
                import authgem_auth
                authgem_auth._cached_project_id[0] = saved_project
                authgem_auth._project_set_by_gui[0] = True
                import os
                os.environ['GOOGLE_CLOUD_PROJECT'] = saved_project
            except Exception:
                pass

    def _saved_auto_glossary_shortcut_index(self):
        """Initial index of the main-window glossary-mode shortcut combo (_create_settings_section)."""
        _auto_mode = self.config.get('auto_glossary_mode', None)
        if _auto_mode is None:
            _auto_mode = 'minimal' if self.config.get('enable_auto_glossary', False) else 'off'
        _mode_idx = {'off': 0, 'off_fuzzy_automap': 1, 'off_no_automap': 2, 'no_glossary': 3, 'minimal': 4, 'balanced': 5, 'full': 6, 'single_pass': 7}.get(_auto_mode.lower(), 0)
        return _mode_idx

    def _on_auto_glossary_shortcut_changed(self, index):
        """Sync shortcut dropdown → main auto_glossary_mode_combo."""
        mode_map = {0: 'off', 1: 'off_fuzzy_automap', 2: 'off_no_automap', 3: 'no_glossary', 4: 'minimal', 5: 'balanced', 6: 'full', 7: 'single_pass'}
        new_mode = mode_map.get(index, 'off')
        is_on = new_mode not in ('off', 'off_fuzzy_automap', 'off_no_automap', 'no_glossary')
        self.config['auto_glossary_mode'] = new_mode
        self.config['enable_auto_glossary'] = is_on
        self.enable_auto_glossary_var = is_on
        self.auto_glossary_mode_var = new_mode
        # Auto-enable append glossary when off/minimal/balanced/full is selected
        if new_mode != 'no_glossary':
            self.config['append_glossary'] = True
            self.append_glossary_var = True
            if hasattr(self, 'append_glossary_checkbox'):
                self.append_glossary_checkbox.blockSignals(True)
                self.append_glossary_checkbox.setChecked(True)
                self.append_glossary_checkbox.blockSignals(False)
                self.append_glossary_checkbox.style().unpolish(self.append_glossary_checkbox)
                self.append_glossary_checkbox.style().polish(self.append_glossary_checkbox)
                self.append_glossary_checkbox.update()
        # Auto-enable auto map when off/off_fuzzy_automap/balanced/full is selected
        if new_mode in ('off', 'off_fuzzy_automap', 'balanced', 'full', 'single_pass'):
            self.config['append_glossary_auto_load'] = True
            self.append_glossary_auto_load_var = True
            if hasattr(self, 'append_glossary_auto_load_checkbox'):
                self.append_glossary_auto_load_checkbox.blockSignals(True)
                self.append_glossary_auto_load_checkbox.setChecked(True)
                self.append_glossary_auto_load_checkbox.blockSignals(False)
                self.append_glossary_auto_load_checkbox.style().unpolish(self.append_glossary_auto_load_checkbox)
                self.append_glossary_auto_load_checkbox.style().polish(self.append_glossary_auto_load_checkbox)
                self.append_glossary_auto_load_checkbox.update()
        # Physically disable auto-mapping when "Off (No Auto-Mapping)" or "Minimal" is selected
        if new_mode in ('off_no_automap', 'minimal'):
            self.config['append_glossary_auto_load'] = False
            self.append_glossary_auto_load_var = False
            if hasattr(self, 'append_glossary_auto_load_checkbox'):
                self.append_glossary_auto_load_checkbox.blockSignals(True)
                self.append_glossary_auto_load_checkbox.setChecked(False)
                self.append_glossary_auto_load_checkbox.blockSignals(False)
                self.append_glossary_auto_load_checkbox.style().unpolish(self.append_glossary_auto_load_checkbox)
                self.append_glossary_auto_load_checkbox.style().polish(self.append_glossary_auto_load_checkbox)
                self.append_glossary_auto_load_checkbox.update()
        # Fuzzy auto-mapping: only enable when off_fuzzy_automap, disable for all other modes
        if new_mode == 'off_fuzzy_automap':
            self.config['fuzzy_auto_mapping'] = True
            self.fuzzy_auto_mapping_var = True
            if hasattr(self, 'fuzzy_auto_mapping_checkbox'):
                self.fuzzy_auto_mapping_checkbox.blockSignals(True)
                self.fuzzy_auto_mapping_checkbox.setChecked(True)
                self.fuzzy_auto_mapping_checkbox.blockSignals(False)
        else:
            self.config['fuzzy_auto_mapping'] = False
            self.fuzzy_auto_mapping_var = False
            if hasattr(self, 'fuzzy_auto_mapping_checkbox'):
                self.fuzzy_auto_mapping_checkbox.blockSignals(True)
                self.fuzzy_auto_mapping_checkbox.setChecked(False)
                self.fuzzy_auto_mapping_checkbox.blockSignals(False)
        if hasattr(self, 'auto_glossary_mode_combo'):
            self.auto_glossary_mode_combo.blockSignals(True)
            self.auto_glossary_mode_combo.setCurrentIndex(index)
            self.auto_glossary_mode_combo.blockSignals(False)

        # Real-time glossary path switching based on auto-mapping state
        # (blockSignals above prevents _on_auto_mapping_toggled from firing,
        #  so we do the path switch inline here.)
        try:
            automap_now = new_mode in ('balanced', 'full', 'single_pass') or (
                new_mode != 'minimal' and
                hasattr(self, 'append_glossary_auto_load_checkbox') and self.append_glossary_auto_load_checkbox.isChecked()
            )
            epub_path = None
            files = list(getattr(self, 'selected_files', []) or [])
            epubs = [p for p in files if str(p).lower().endswith('.epub')]
            if len(epubs) == 1:
                epub_path = epubs[0]
            if epub_path:
                # Clear current glossary state for a clean switch
                # (don't log here — _autofill / auto_load_glossary_for_file will log the new mapping)
                self.manual_glossary_path = None
                self.auto_loaded_glossary_path = None
                self.auto_loaded_glossary_for_file = None
                self.manual_glossary_manually_loaded = False
                if automap_now:
                    if hasattr(self, '_autofill_glossary_for_current_selection'):
                        self._autofill_glossary_for_current_selection()
                else:
                    self.auto_load_glossary_for_file(epub_path)
                # Refresh glossary editor path (always sync, not just when tab visible)
                if hasattr(self, 'editor_file_entry'):
                    new_path = getattr(self, 'auto_loaded_glossary_path', None) or getattr(self, 'manual_glossary_path', None)
                    if new_path and os.path.exists(new_path):
                        self.editor_file_entry.setText(new_path)
                        _log_msg = f"📑 Glossary editor switched to: {os.path.basename(new_path)}"
                        if getattr(self, '_last_editor_switch_log', '') != _log_msg:
                            self.append_log(_log_msg)
                            self._last_editor_switch_log = _log_msg
        except Exception:
            pass

        self.save_config(show_message=False)

    def _resolve_startup_target_language(self):
        """Target language the desktop selects at startup (_create_prompt_section)."""
        # Set initial value from config - prioritize glossary_target_language for consistency
        # If both exist but differ, use output_language and sync glossary_target_language
        saved_lang = self.config.get('output_language')
        glossary_lang = self.config.get('glossary_target_language')
        
        # Determine which value to use
        if saved_lang and glossary_lang and saved_lang != glossary_lang:
            # They're out of sync - use output_language as the source of truth
            final_lang = saved_lang
        elif glossary_lang:
            # Use glossary_target_language if it exists
            final_lang = glossary_lang
        elif saved_lang:
            # Use output_language if it exists
            final_lang = saved_lang
        else:
            # Neither exists - default to English
            final_lang = 'English'
        return final_lang

    def _init_active_profile_prompt(self):
        """Fill the prompt editor with the active profile's text (_setup_gui)."""
        if hasattr(self, 'profile_var') and self.profile_var in self.prompt_profiles:
            initial_prompt = self.prompt_profiles[self.profile_var]
            self.prompt_text.setPlainText(initial_prompt)
            # Set the initial active profile for autosave
            self._active_profile_for_autosave = self.profile_var
            
            # Store original content for revert capability
            if not hasattr(self, '_original_profile_content'):
                self._original_profile_content = {}
            self._original_profile_content[self.profile_var] = initial_prompt

    def _replay_gui_startup_handlers(self):
        """Run the startup handlers desktop _setup_gui fires once its widgets exist, in desktop order.

        TranslatorGUI calls each of these from its section builders (_create_model_section,
        _create_settings_section, _create_prompt_section, _setup_gui); owners without Qt widgets
        (HeadlessOwner) call this after installing their widget shims. tests/test_headless_owner.py
        checks the order against the desktop builders.
        """
        self._restore_authgem_project_selection()
        self._on_disable_temperature_toggle()
        try:
            self._on_auto_glossary_shortcut_changed(self.auto_glossary_shortcut_combo.currentIndex())
        except Exception:
            pass
        self._on_context_mode_changed()
        final_lang = self._resolve_startup_target_language()
        self.update_target_language(final_lang)
        self._init_active_profile_prompt()
        self._update_auto_compression_factor()

    def _auto_encrypt_api_keys(self):
        """Last step of TranslatorGUI.__init__ (moved verbatim): save_config when API keys are plain.

        config_store.load_config decrypts, so every config with an api_key / replicate_api_key runs
        this save at startup; it re-exports the saved settings after initialize_environment_variables.
        HeadlessOwner calls it at the end of its __init__ with its in-memory save_config.
        """
        try:
            needs_encryption = False
            if 'api_key' in self.config and self.config['api_key']:
                if not self.config['api_key'].startswith('ENC:'):
                    needs_encryption = True
            if 'replicate_api_key' in self.config and self.config['replicate_api_key']:
                if not self.config['replicate_api_key'].startswith('ENC:'):
                    needs_encryption = True

            if needs_encryption:
                # Auto-migrate to encrypted format
                print("Auto-encrypting API keys...")
                self.save_config(show_message=False)
                print("API keys encrypted successfully!")
        except Exception as e:
            print(f"Auto-encryption check failed: {e}")

    def _set_batching_mode(self, mode):
        """Update batching mode state and any visible radio controls."""
        mode = str(mode or 'direct').strip().lower()
        if mode not in ('direct', 'conservative', 'aggressive'):
            mode = 'direct'
        self.batch_mode_var = mode
        self.config['batching_mode'] = mode
        if mode in ('direct', 'conservative'):
            self._context_last_batched_mode = mode

        radio_map = (
            ('batch_conservative_radio', mode == 'conservative'),
            ('batch_direct_radio', mode == 'direct'),
            ('batch_no_batching_radio', mode == 'aggressive'),
        )
        for attr, checked in radio_map:
            if not hasattr(self, attr):
                continue
            radio = getattr(self, attr)
            try:
                radio.blockSignals(True)
                radio.setChecked(checked)
                radio.blockSignals(False)
            except Exception:
                pass

    def _refresh_context_batching_controls(self):
        """Reflect context-mode batching constraints in the Other Settings radios."""
        controls = (
            ('batch_conservative_radio', 'Conservative batching'),
            ('batch_direct_radio', 'Direct batching'),
            ('batch_no_batching_radio', 'No batching'),
        )
        if not any(hasattr(self, attr) for attr, _label in controls):
            return

        mode = getattr(self, 'context_mode_var', None) or self._context_mode_from_flags()
        purple = "#b388ff"
        muted = "#777"
        normal = "#ddd"

        def apply_radio(attr, label, enabled, locked=False):
            radio = getattr(self, attr, None)
            if radio is None:
                return
            try:
                base_label = getattr(radio, '_context_base_text', None)
                if not base_label:
                    radio._context_base_text = label
                    base_label = label
                radio.setText(f"🔒  {base_label}" if locked else base_label)
                radio.setEnabled(enabled)
                if locked:
                    radio.setStyleSheet(
                        f"QRadioButton {{ color: {purple}; }}"
                        f"QRadioButton:disabled {{ color: {purple}; }}"
                    )
                elif enabled:
                    radio.setStyleSheet(f"QRadioButton {{ color: {normal}; }}")
                else:
                    radio.setStyleSheet(f"QRadioButton {{ color: {muted}; }}")
            except Exception:
                pass

        if mode == 'off':
            apply_radio('batch_conservative_radio', 'Conservative batching', False, False)
            apply_radio('batch_direct_radio', 'Direct batching', False, False)
            apply_radio('batch_no_batching_radio', 'No batching', True, True)
            status = "ℹ️ Context Mode Off uses No batching for translation.\n   Glossary extraction also uses No batching."
        else:
            apply_radio('batch_conservative_radio', 'Conservative batching', True, True)
            apply_radio('batch_direct_radio', 'Direct batching', True, True)
            apply_radio('batch_no_batching_radio', 'No batching', False, False)
            if mode == 'contextual_history':
                status = "ℹ️ Contextual History requires Direct or Conservative batching.\n   Glossary extraction uses the selected batching mode so history order stays aligned."
            else:
                status = "ℹ️ Rolling Summary requires Direct or Conservative batching for translation.\n   Glossary extraction stays on No batching and does not use the selected batching mode."

        label = getattr(self, 'batch_context_lock_label', None)
        if label is not None:
            try:
                label.setText(status)
                label.setStyleSheet("color: #17a2b8; font-size: 10pt;")
                label.setVisible(True)
            except Exception:
                pass

    def _enforce_context_batching_mode(self):
        """Apply context-mode batching constraints and keep the visible controls in sync."""
        mode = getattr(self, 'context_mode_var', None) or self._context_mode_from_flags()
        current = str(getattr(self, 'batch_mode_var', 'aggressive') or 'aggressive').strip().lower()

        if mode == 'off':
            if current in ('direct', 'conservative'):
                self._context_last_batched_mode = current
            self._set_batching_mode('aggressive')
        else:
            if current not in ('direct', 'conservative'):
                self._set_batching_mode(getattr(self, '_context_last_batched_mode', 'direct'))

        self._refresh_context_batching_controls()

    def _coerce_live_bool(self, value, default=False):
        if value is None:
            return bool(default)
        if hasattr(value, 'isChecked'):
            try:
                return bool(value.isChecked())
            except Exception:
                return bool(default)
        if hasattr(value, 'get'):
            try:
                return self._coerce_live_bool(value.get(), default)
            except Exception:
                return bool(default)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('1', 'true', 'yes', 'on', 'checked'):
                return True
            if normalized in ('0', 'false', 'no', 'off', 'unchecked', ''):
                return False
        return bool(value)

    def _on_disable_temperature_toggle(self, state=None):
        """Keep the live request flag and temperature field in sync."""
        checkbox = getattr(self, 'disable_temperature_checkbox', None)
        if checkbox is not None:
            disabled = bool(checkbox.isChecked())
        else:
            disabled = self._coerce_live_bool(
                getattr(self, 'disable_temperature_var', self.config.get('disable_temperature', False)),
                False,
            )
        self.disable_temperature_var = disabled
        self.config['disable_temperature'] = disabled
        os.environ['DISABLE_TEMPERATURE'] = '1' if disabled else '0'
        temperature_entry = getattr(self, 'trans_temp', None)
        if temperature_entry is not None:
            temperature_entry.setEnabled(not disabled)

    def _migrate_strict_matching_config(self):
        """One-time upgrade from the retired Strict Gender Entry Matching toggle.

        The toggle is gone; its scope dropdown moved onto Precise Term
        Matching with All as the default. A config that still carries the
        old on/off key is upgraded once: ON keeps the scope it had (the old
        'characters' is spelled 'gender' now), OFF means the saved scope was
        never in effect, so the new default applies. The key is removed so
        a later deliberate choice is never overridden again.
        """
        if 'compress_glossary_strict_gender_matching' not in self.config:
            return
        was_on = bool(self.config.pop('compress_glossary_strict_gender_matching'))
        mode = str(self.config.get('compress_glossary_strict_matching_mode', '') or '').strip().lower()
        if not was_on or not mode:
            mode = 'all'
        elif mode in ('characters', 'character'):
            mode = 'gender'
        self.config['compress_glossary_strict_matching_mode'] = mode
        self.compress_glossary_strict_matching_mode_var = mode
        if hasattr(self, 'compress_glossary_strict_gender_matching_var'):
            del self.compress_glossary_strict_gender_matching_var

    _LEGACY_SPECIAL_FILE_EXACT_TOKENS = frozenset({'index', 'glossary', 'glossary_extension'})

    def _upgrade_special_file_exact(self, value):
        """Upgrade a saved exact-match list that is still the old default.

        The default gained 'glossary_unified' with the unified glossary.
        Anyone who opened Other Settings has the old default persisted in
        config.json and exported to SPECIAL_FILE_EXACT, which would shadow
        the new default for good. Customised lists are left untouched.
        """
        text = str(value or '')
        tokens = {token.strip().lower() for token in text.split(',') if token.strip()}
        if tokens == self._LEGACY_SPECIAL_FILE_EXACT_TOKENS:
            return 'index, glossary, glossary_extension, glossary_unified'
        return text

    def _on_context_mode_changed(self, index=None):
        """Map the Context Mode combo onto the existing runtime config flags."""
        mode = 'off'
        if hasattr(self, 'context_mode_combo'):
            mode = self.context_mode_combo.currentData() or 'off'
        self.context_mode_var = mode

        is_contextual = mode == 'contextual_history'
        is_summary = mode in ('rolling_summary_replace', 'rolling_summary_append')
        is_summary_append = mode == 'rolling_summary_append'
        has_context_options = is_contextual or is_summary

        self.contextual_var = is_contextual
        self.rolling_summary_var = is_summary
        if is_summary:
            self.rolling_summary_mode_var = 'append' if mode == 'rolling_summary_append' else 'replace'

        self.config['contextual'] = self.contextual_var
        self.config['use_rolling_summary'] = self.rolling_summary_var
        self.config['rolling_summary_mode'] = self.rolling_summary_mode_var
        self._enforce_context_batching_mode()

        if hasattr(self, 'contextual_warning_label'):
            self.contextual_warning_label.setVisible(is_contextual)
        if hasattr(self, 'context_options_container'):
            self.context_options_container.setVisible(has_context_options)
        if hasattr(self, 'rolling_summary_keys_btn'):
            self.rolling_summary_keys_btn.setVisible(is_summary)
        if hasattr(self, 'trans_history_label'):
            self.trans_history_label.setVisible(is_contextual)
        if hasattr(self, 'trans_history'):
            self.trans_history.setVisible(is_contextual)
            self.trans_history.setEnabled(is_contextual)
        for attr in (
            'rolling_summary_exchanges_label',
            'rolling_summary_exchanges_edit',
        ):
            if hasattr(self, attr):
                getattr(self, attr).setVisible(is_summary)
        for attr in (
            'rolling_summary_retain_label',
            'rolling_summary_retain_edit',
        ):
            if hasattr(self, attr):
                getattr(self, attr).setVisible(is_summary_append)

        self._hook_context_mode_layout(has_context_options)

        self.translation_history_rolling_var = True

    def update_target_language(self, text):
        """Update target language and sync across all UI components"""
        self.config['output_language'] = text
        # Also update environment variable if needed
        os.environ['OUTPUT_LANGUAGE'] = text
        # Update lang_var which is used by _get_environment_variables
        self.lang_var = text
        
        # Sync with main dropdown if called externally
        if hasattr(self, 'target_lang_combo') and self.target_lang_combo.currentText() != text:
            self.target_lang_combo.blockSignals(True)
            index = self.target_lang_combo.findText(text)
            if index >= 0:
                self.target_lang_combo.setCurrentIndex(index)
            else:
                self.target_lang_combo.setCurrentText(text)
            self.target_lang_combo.blockSignals(False)
        
        # Sync with glossary manager dropdowns if they exist
        self.config['glossary_target_language'] = text
        # IMPORTANT: Also update environment variable immediately so glossary manager sees the change
        os.environ['GLOSSARY_TARGET_LANGUAGE'] = text
        
        if hasattr(self, 'manual_target_language_combo') and self.manual_target_language_combo:
            if self.manual_target_language_combo.currentText() != text:
                self.manual_target_language_combo.blockSignals(True)
                index = self.manual_target_language_combo.findText(text)
                if index >= 0:
                    self.manual_target_language_combo.setCurrentIndex(index)
                else:
                    self.manual_target_language_combo.setCurrentText(text)
                self.manual_target_language_combo.blockSignals(False)
                    
        if hasattr(self, 'glossary_target_language_combo') and self.glossary_target_language_combo:
            if self.glossary_target_language_combo.currentText() != text:
                self.glossary_target_language_combo.blockSignals(True)
                index = self.glossary_target_language_combo.findText(text)
                if index >= 0:
                    self.glossary_target_language_combo.setCurrentIndex(index)
                else:
                    self.glossary_target_language_combo.setCurrentText(text)
                self.glossary_target_language_combo.blockSignals(False)
                
        # Sync with metadata batch UI if dialog is open
        if hasattr(self, 'metadata_batch_ui') and hasattr(self.metadata_batch_ui, 'output_lang_combo'):
            try:
                # Check if widget is valid (not deleted)
                if self.metadata_batch_ui.output_lang_combo.isVisible():
                    if self.metadata_batch_ui.output_lang_combo.currentText() != text:
                        self.metadata_batch_ui.output_lang_combo.blockSignals(True)
                        index = self.metadata_batch_ui.output_lang_combo.findText(text)
                        if index >= 0:
                            self.metadata_batch_ui.output_lang_combo.setCurrentIndex(index)
                        else:
                            self.metadata_batch_ui.output_lang_combo.setCurrentText(text)
                        self.metadata_batch_ui.output_lang_combo.blockSignals(False)
            except RuntimeError:
                # Widget might be deleted if dialog was closed
                pass
        
        # Sync with manga settings dialog if open
        if hasattr(self, 'manga_settings_dialog') and self.manga_settings_dialog:
            try:
                # Check if widget is valid (not deleted)
                if hasattr(self.manga_settings_dialog, 'manual_translate_language'):
                    combo = self.manga_settings_dialog.manual_translate_language
                    if combo.currentText() != text:
                        combo.blockSignals(True)
                        index = combo.findText(text)
                        if index >= 0:
                            combo.setCurrentIndex(index)
                        else:
                            combo.setCurrentText(text)
                        combo.blockSignals(False)
            except RuntimeError:
                # Widget might be deleted if dialog was closed
                pass
        
        # Update manga settings in config as well
        if 'manga_settings' not in self.config:
            self.config['manga_settings'] = {}
        if 'manual_edit' not in self.config['manga_settings']:
            self.config['manga_settings']['manual_edit'] = {}
        self.config['manga_settings']['manual_edit']['translate_target_language'] = text
        
        # Sync with AI Hunter config
        if 'ai_hunter_config' in self.config:
            if 'language_detection' in self.config['ai_hunter_config']:
                self.config['ai_hunter_config']['language_detection']['target_language'] = text
