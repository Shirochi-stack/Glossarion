"""SettingsPersistenceMixin: save_config's settings collection and env export, moved verbatim.

Shared GUI-free core (Glossarion mobile rewrite, milestone U2). TranslatorGUI.save_config
keeps its validation dialogs, backup, file write and messages and calls:

* ``_apply_live_settings_to_config()`` - save_config sections 2-3 (the data-driven
  ``settings_map`` with widget-first precedence plus the special cases), mutating
  ``self.config`` in place exactly as before;
* ``_export_settings_env(show_message, debug_enabled)`` - save_config section 4.

``_collect_live_settings()`` runs the same section 2-3 code on a deep copy and
returns the config dict save_config would persist, without touching the owner
(the HeadlessOwner round-trip bridge: ``HeadlessOwner(gui._collect_live_settings())``
must build the same environment as the live desktop owner).

``safe_int`` / ``safe_float`` were save_config's local helpers (moved to module level;
translator_gui imports them for save_config's verification block).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import os

from emoticon_patterns import DEFAULT_EMOTICON_PATTERNS
from run_env import (
    MULTIPASS_REFINEMENT_MODES,
    REFINEMENT_RAW_PROMPT_ROLES,
    _authnd_auto_token_limits,
    _format_plain_decimal_setting,
)
from title_tag_translation import DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT

# Helper functions for safe type conversion (were local to save_config)
def safe_int(value, default):
    try: return int(value)
    except (ValueError, TypeError): return default

def safe_float(value, default):
    try: return float(value)
    except (ValueError, TypeError): return default


#: Owner attributes _apply_live_settings_to_config rebinds or mutates besides self.config
#: (_collect_live_settings restores them, so collecting stays side-effect free).
_LIVE_SETTINGS_OWNER_ATTRS = ('use_header_as_output_var', 'custom_prefix_routes', 'custom_entry_types')


def _noop_sync_custom_prefix_routes_env():
    """Stand-in for _sync_custom_prefix_routes_env while collecting (no os.environ writes)."""
    return None


class SettingsPersistenceMixin:
    """save_config pieces shared by TranslatorGUI and HeadlessOwner (see module docstring)."""

    def _apply_live_settings_to_config(self):
        """save_config sections 2-3: settings_map (widget-first) + special cases, into self.config in place."""
        # --- 2. Data-Driven Configuration Mapping ---
        # Helper to get value from a source (widget or variable)
        def _get_value(source_attr):
            if not hasattr(self, source_attr):
                return None

            attr = getattr(self, source_attr)
            if hasattr(attr, 'isChecked'): return attr.isChecked()
            if hasattr(attr, 'toPlainText'): return attr.toPlainText().strip()
            if hasattr(attr, 'text'): return attr.text().strip()
            if hasattr(attr, 'currentData'):
                data = attr.currentData()
                if data is not None:
                    return data
                return attr.currentIndex()
            return attr

        # Central mapping of configuration settings
        # format: (config_key, [source_attributes_in_priority_order], default_value, type_converter_func)
        # U9 P5b: built by the settings schema, the single source of this table (the literal
        # lives on, line for line, in src/mobile/tools/frozen_desktop_tables.py).
        from settings_schema import desktop_settings_map
        settings_map = desktop_settings_map(self)

        # Process the settings map to populate self.config
        for key, sources, default, converter in settings_map:
            final_value = None
            found = False
            for source in sources:
                if isinstance(source, tuple): # Handle special config source
                    val = self.config.get(source[1])
                else:
                    val = _get_value(source)

                if val is not None:
                    final_value = val
                    found = True
                    break

            if found:
                converted_value = converter(final_value) if converter else final_value
                self.config[key] = converted_value
            elif default is not None and key not in self.config:
                # Only apply default if key doesn't already exist in config
                # (prevents wiping loaded values when save_config is called before widgets exist)
                self.config[key] = default

        self.config['use_header_as_output'] = False
        self.use_header_as_output_var = False

        if hasattr(self, 'custom_prefix_routes'):
            self.custom_prefix_routes = self._normalize_custom_prefix_routes(self.custom_prefix_routes)
            self.config['custom_prefix_routes'] = self.custom_prefix_routes
            self._sync_custom_prefix_routes_env()

        # --- 3. Handle Special Cases and Complex Logic ---

        # Fuzzy matching threshold with range validation
        # Check slider first (created in glossary settings dialog), then fallback to var
        if hasattr(self, 'fuzzy_threshold_slider'):
            fuzzy_val = self.fuzzy_threshold_slider.value() / 100.0
            self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90
        elif hasattr(self, 'fuzzy_threshold_value'):
            fuzzy_val = self.fuzzy_threshold_value
            self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90
        elif hasattr(self, 'fuzzy_threshold_var'):
            fuzzy_val = self.fuzzy_threshold_var
            self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90

        # Glossary filter mode from radio buttons
        if hasattr(self, 'glossary_filter_mode_buttons'):
            for mode_key, radio_button in self.glossary_filter_mode_buttons.items():
                if radio_button.isChecked():
                    self.config['glossary_filter_mode'] = mode_key
                    break

        # Duplicate algorithm from combo box
        if hasattr(self, 'duplicate_algo_combo'):
            algo_reverse_map = {0: 'auto', 1: 'strict', 2: 'balanced', 3: 'aggressive', 4: 'basic'}
            self.config['glossary_duplicate_algorithm'] = algo_reverse_map.get(self.duplicate_algo_combo.currentIndex(), 'auto')
        elif hasattr(self, 'glossary_duplicate_algorithm_var'):
            self.config['glossary_duplicate_algorithm'] = self.glossary_duplicate_algorithm_var

        # Partial ratio weight (0 disables substring matching)
        if hasattr(self, 'partial_ratio_slider'):
            weight = self.partial_ratio_slider.value() / 100.0
        elif hasattr(self, 'partial_ratio_weight'):
            weight = self.partial_ratio_weight
        else:
            weight = self.config.get('glossary_partial_ratio_weight', 0.45)
        self.config['glossary_partial_ratio_weight'] = max(0.0, min(1.0, float(weight or 0.45)))
        if hasattr(self, 'partial_ratio_gender_only_checkbox'):
            self.config['glossary_partial_ratio_gender_only'] = bool(self.partial_ratio_gender_only_checkbox.isChecked())
        elif hasattr(self, 'glossary_partial_ratio_gender_only_var'):
            self.config['glossary_partial_ratio_gender_only'] = bool(self.glossary_partial_ratio_gender_only_var)
        if hasattr(self, 'alias_aware_name_matching_checkbox'):
            self.config['glossary_alias_aware_name_matching'] = bool(self.alias_aware_name_matching_checkbox.isChecked())
        elif hasattr(self, 'glossary_alias_aware_name_matching_var'):
            self.config['glossary_alias_aware_name_matching'] = bool(self.glossary_alias_aware_name_matching_var)
        if hasattr(self, 'alias_aware_gender_only_checkbox'):
            self.config['glossary_alias_aware_gender_only'] = bool(self.alias_aware_gender_only_checkbox.isChecked())
        elif hasattr(self, 'glossary_alias_aware_gender_only_var'):
            self.config['glossary_alias_aware_gender_only'] = bool(self.glossary_alias_aware_gender_only_var)

        # Target language from combo box
        if hasattr(self, 'glossary_target_language_combo'):
            self.config['glossary_target_language'] = self.glossary_target_language_combo.currentText()

        # Custom glossary data structures
        if hasattr(self, 'custom_glossary_fields'):
            self.config['custom_glossary_fields'] = self.custom_glossary_fields
        # Update enabled status from checkboxes (try both possible attribute names)
        if hasattr(self, 'type_enabled_checks'):
            for type_name, checkbox in self.type_enabled_checks.items():
                if type_name in self.custom_entry_types:
                    self.custom_entry_types[type_name]['enabled'] = checkbox.isChecked()
        elif hasattr(self, 'type_enabled_checkboxes'):
            for type_name, checkbox in self.type_enabled_checkboxes.items():
                if type_name in self.custom_entry_types:
                    self.custom_entry_types[type_name]['enabled'] = checkbox.isChecked()
        if hasattr(self, 'custom_entry_types'):
            self.config['custom_entry_types'] = self.custom_entry_types

        # Backward compatibility for translate_special_files
        if hasattr(self, 'translate_special_files_var'):
            self.config['translate_special_files'] = self.translate_special_files_var

        # Translate all numbered HTML files override
        if hasattr(self, 'translate_all_numbered_html_var'):
            self.config['translate_all_numbered_html'] = self.translate_all_numbered_html_var

        if hasattr(self, 'never_consider_in_between_files_as_special_var'):
            self.config['never_consider_in_between_files_as_special'] = bool(
                self.never_consider_in_between_files_as_special_var
            )

        # Custom special file keywords
        if hasattr(self, 'special_file_keywords_var'):
            self.config['special_file_keywords'] = self.special_file_keywords_var
        if hasattr(self, 'special_file_exact_var'):
            self.config['special_file_exact'] = self.special_file_exact_var

        # Backward compatibility for extraction_mode
        if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'file_filtering_level_var'):
            if self.text_extraction_method_var == 'enhanced':
                self.config['extraction_mode'] = 'enhanced'
                self.config['enhanced_filtering'] = self.file_filtering_level_var
            else:
                self.config['extraction_mode'] = self.file_filtering_level_var
        elif hasattr(self, 'extraction_mode_var'):
            self.config['extraction_mode'] = self.extraction_mode_var

        # Token limit
        _tl = self.token_limit_entry.text().replace(',', '').strip()
        self.config['token_limit'] = int(_tl) if _tl.isdigit() else None

        # Update last update check time
        if hasattr(self, 'update_manager') and self.update_manager:
            self.config['last_update_check_time'] = self.update_manager._last_check_time

        # Save prompts from text widgets
        prompt_widgets = {
            'manual_glossary_prompt': 'manual_prompt_text',
            'unified_auto_glosary_prompt3': 'auto_prompt_text',
            'append_glossary_prompt': 'append_prompt_text',
            'single_pass_glossary_header_prompt': 'single_pass_header_prompt_text',
            'glossary_refinement_system_prompt': 'glossary_refinement_system_prompt_text',
            'glossary_refinement_user_prompt': 'glossary_refinement_user_prompt_text',
            'glossary_translation_prompt': 'translation_prompt_text',
            'glossary_format_instructions': 'format_instructions_text',
        }
        for key, widget_name in prompt_widgets.items():
            if hasattr(self, widget_name):
                try:
                    self.config[key] = getattr(self, widget_name).toPlainText().strip()
                except Exception:
                    pass

        # Set defaults for settings that might not exist yet
        self.config.setdefault('glossary_auto_backup', True)
        self.config.setdefault('glossary_max_backups', 50)
        default_qa_settings = {'foreign_char_threshold': 0, 'excluded_characters': '', 'target_language': 'english', 'check_encoding_issues': False, 'check_repetition': True, 'check_translation_artifacts': True, 'check_glossary_leakage': True, 'min_file_length': 0, 'report_format': 'detailed', 'auto_save_report': True, 'check_word_count_ratio': True, 'check_multiple_headers': True, 'warn_name_mismatch': True, 'check_missing_html_tag': True, 'check_missing_beautifulsoup_tags': False, 'sdlxliff_tag_retention_threshold': 0.9, 'sdlxliff_tag_surplus_tolerance': 0.05, 'sdlxliff_min_source_paragraph_tags': 20, 'check_invalid_nesting': False, 'cache_enabled': True, 'cache_auto_size': False, 'cache_show_stats': False}
        default_qa_settings.update({
            'whitelist_emoticon_patterns': False,
            'exclude_ruby_tags': False,
            'emoticon_patterns': list(DEFAULT_EMOTICON_PATTERNS),
            'emoticon_patterns_are_regex': False,
        })
        self.config.setdefault('qa_scanner_settings', default_qa_settings)
        self.config.setdefault('ai_hunter_config', {}).setdefault(
            'ai_hunter_max_workers', max(1, (os.cpu_count() or 4) // 2))
        self.config.setdefault('save_partial_results', True)
        self.config.setdefault('save_prohibited_results', False)
        self.config.setdefault('disable_empty_safety_heuristic', True)
        self.config.setdefault('missing_finish_as_prohibited', self.config.get('unknown_finish_as_prohibited', False))
        self.config.setdefault('enable_translation_chunk_prompt', False)
        self.config.setdefault('include_previous_chunk', False)
        self.config.setdefault('previous_chunk_context_limit', 3)
        self.config.setdefault('translation_chunk_prompt_role', 'assistant')
        self.config.setdefault(
            'image_only_title_tag_system_prompt',
            DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
        )
        # Image compression defaults
        compression_defaults = {'enable_image_compression': False, 'auto_compress_enabled': True, 'target_image_tokens': 1000, 'image_compression_format': 'auto', 'webp_quality': 85, 'jpeg_quality': 85, 'png_compression': 6, 'max_image_dimension': 2048, 'max_image_size_mb': 10, 'preserve_transparency': False, 'preserve_original_format': False, 'optimize_for_ocr': True, 'progressive_encoding': True, 'save_compressed_images': False}
        for key, val in compression_defaults.items():
            self.config.setdefault(key, val)

    def _export_settings_env(self, show_message=False, debug_enabled=False):
        """save_config section 4: export the saved settings to os.environ; returns the keys set."""
        # --- 4. Update Environment Variables ---
        def _update_env(key, new_val, is_bool=False):
            val_to_set = str(new_val)
            if is_bool:
                val_to_set = '1' if new_val else '0'

            old_val = os.environ.get(key, '<NOT SET>')
            os.environ[key] = val_to_set
            if show_message and debug_enabled and old_val != val_to_set:
                self.append_log(f"🔍 [DEBUG] ENV {key}: '{old_val}' → '{val_to_set}'")
            return key

        def _config_bool(key, default=False):
            value = self.config.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(value)

        def _config_int(key, default, minimum=0, maximum=None):
            try:
                numeric_value = int(float(self.config.get(key, default)))
            except (TypeError, ValueError):
                numeric_value = int(default)
            numeric_value = max(int(minimum), numeric_value)
            if maximum is not None:
                numeric_value = min(int(maximum), numeric_value)
            return numeric_value

        def _config_float(key, default, minimum=0.0, maximum=None):
            try:
                numeric_value = float(self.config.get(key, default))
            except (TypeError, ValueError):
                numeric_value = float(default)
            numeric_value = max(float(minimum), numeric_value)
            if maximum is not None:
                numeric_value = min(float(maximum), numeric_value)
            return f"{numeric_value:g}"

        def _config_choice(key, default, allowed):
            value = str(self.config.get(key, default) or default).strip().lower()
            return value if value in allowed else default

        env_vars_set = []
        # Standard env vars
        env_vars_set.append(_update_env('OPENROUTER_USE_HTTP_ONLY', self.config.get('openrouter_use_http_only'), is_bool=True))
        env_vars_set.append(_update_env('OPENROUTER_ACCEPT_IDENTITY', self.config.get('openrouter_accept_identity'), is_bool=True))
        env_vars_set.append(_update_env('OPENROUTER_PREFERRED_PROVIDER', (str(self.config.get('openrouter_preferred_provider', 'Auto') or '').strip() or 'Auto')))
        env_vars_set.append(_update_env('RETAIN_SOURCE_EXTENSION', self.config.get('retain_source_extension'), is_bool=True))
        env_vars_set.append(_update_env('DOWNLOAD_REMOTE_IMAGE_URLS', self.config.get('download_remote_image_urls'), is_bool=True))
        env_vars_set.append(_update_env('REMOTE_IMAGE_DOWNLOAD_WORKERS', _config_int('remote_image_download_workers', 4, 1, 32)))
        env_vars_set.append(_update_env('REMOTE_IMAGE_DOWNLOAD_INTERVAL', _config_float('remote_image_download_interval', 0.5, 0.0, 60.0)))
        env_vars_set.append(_update_env('ENABLE_GUI_YIELD', self.config.get('enable_gui_yield'), is_bool=True))
        env_vars_set.append(_update_env('PARTIAL_B2_ENTRIES_PER_REQUEST', self.config.get('partial_b2_entries_per_request', -1)))
        authnd_auto_enabled = bool(self.config.get('authnd_token_concurrency_auto', True))
        if authnd_auto_enabled:
            authnd_token_limit, authnd_subprocess_limit, _authnd_cores = _authnd_auto_token_limits()
        else:
            authnd_token_limit = max(1, safe_int(self.config.get('authnd_token_concurrency', 1), 1))
            authnd_subprocess_limit = max(1, safe_int(self.config.get('authnd_token_subprocess_concurrency', 1), 1))
        env_vars_set.append(_update_env('AUTHND_TOKEN_CONCURRENCY_AUTO', authnd_auto_enabled, is_bool=True))
        env_vars_set.append(_update_env('AUTHND_TOKEN_CONCURRENCY', authnd_token_limit))
        env_vars_set.append(_update_env('AUTHND_TOKEN_SUBPROCESS_CONCURRENCY', authnd_subprocess_limit))
        env_vars_set.append(_update_env('AUTHND_TOKEN_TIMEOUT', _config_int('authnd_token_timeout', 180, 30, 600)))
        env_vars_set.append(_update_env('ORDERED_BATCH_DISPATCH_TIMEOUT', _config_int('dispatch_order_timeout', 3, 0, 120)))
        env_vars_set.append(_update_env('GEMINI_FREE_ADAPTIVE_SPLIT', _config_bool('gemini_free_adaptive_split', True), is_bool=True))
        env_vars_set.append(_update_env('GLOSSARION_TOR_ENABLED', _config_bool('tor_proxy_enabled', False), is_bool=True))
        env_vars_set.append(_update_env('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', _config_bool('gemini_free_html_text_node_transport', True), is_bool=True))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _config_int('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_URL_CHARS', _config_int('gemini_free_subchunk_url_chars', 14500, 1000, 200000)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _config_int('gemini_free_subchunk_safety_chars', 600, 0, 20000)))
        env_vars_set.append(_update_env('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _config_int('gemini_free_min_subchunk_body_chars', 80, 1, 50000)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _config_int('gemini_free_subchunk_concurrency', 0, 0, 64)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_START_DELAY', _config_float('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_TIMEOUT', _config_int('gemini_free_subchunk_timeout', 0, 0, 7200)))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _config_choice('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})))
        env_vars_set.append(_update_env('GEMINI_FREE_HTML_SPLITTER', _config_choice('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})))
        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_BALANCER', _config_choice('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})))

        # Extraction workers env var
        new_workers = str(self.config['extraction_workers']) if self.config['enable_parallel_extraction'] else "1"
        env_vars_set.append(_update_env('EXTRACTION_WORKERS', new_workers))
        env_vars_set.append(_update_env(
            'PDF_EXTRACTION_WORKERS',
            str(self.config.get('pdf_extraction_workers', 'auto') or 'auto'),
        ))
        env_vars_set.append(_update_env(
            'PDF_PARAGRAPH_ALIGNMENT',
            str(self.config.get('pdf_paragraph_alignment', 'source') or 'source'),
        ))
        env_vars_set.append(_update_env(
            'PDF_HEADER_ALIGNMENT',
            str(self.config.get('pdf_header_alignment', 'source') or 'source'),
        ))
        env_vars_set.append(_update_env(
            'PDF_PARAGRAPH_JUSTIFICATION',
            str(self.config.get('pdf_paragraph_justification', 'source') or 'source'),
        ))
        env_vars_set.append(_update_env(
            'PDF_RTL_PARAGRAPH_LAYOUT',
            self.config.get('pdf_rtl_paragraph_layout', False),
            is_bool=True,
        ))
        env_vars_set.append(_update_env('USE_THREAD_POOL_EXTRACTION', self.config.get('use_thread_pool_extraction'), is_bool=True))

        # Wire debug payload saving to GUI debug mode
        os.environ['DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE'] = '1' if debug_enabled else '0'
        os.environ['SHOW_DEBUG_BUTTONS'] = '1' if debug_enabled else '0'
        os.environ['DEBUG_SAVE_REQUEST_PAYLOADS'] = '1'

        # Glossary-related environment variables
        if show_message and debug_enabled: self.append_log("🔍 [DEBUG] Setting glossary environment variables...")
        try:
            for env_key, env_value in self._glossary_env_mappings():
                env_vars_set.append(_update_env(env_key, env_value))
        except Exception as e:
            if show_message and debug_enabled: self.append_log(f"❌ [DEBUG] Glossary environment variable setup failed: {e}")

        if show_message and debug_enabled:
            self.append_log(f"🔍 [DEBUG] Set {len(env_vars_set)} environment variables.")
        return env_vars_set

    def _collect_live_settings(self):
        """The config dict save_config would persist, computed without side effects.

        Runs the verbatim save_config sections 2-3 (``_apply_live_settings_to_config``)
        against a deep copy of ``self.config``: widget-first precedence, converters and
        special cases are exactly save_config's. The owner's config, the attributes
        those sections rebind (``_LIVE_SETTINGS_OWNER_ATTRS``) and os.environ (the
        custom-prefix-route export) are left untouched. save_config's backup,
        validation dialogs, ``_on_context_mode_changed`` refresh, env export and file
        write are not part of collecting.
        """
        missing = object()
        original_config = self.config
        saved = {name: self.__dict__.get(name, missing) for name in _LIVE_SETTINGS_OWNER_ATTRS}
        had_sync_override = '_sync_custom_prefix_routes_env' in self.__dict__
        saved_sync = self.__dict__.get('_sync_custom_prefix_routes_env', missing)
        self.config = copy.deepcopy(original_config)
        if saved['custom_entry_types'] is not missing:
            self.custom_entry_types = copy.deepcopy(saved['custom_entry_types'])
        self._sync_custom_prefix_routes_env = _noop_sync_custom_prefix_routes_env
        try:
            self._apply_live_settings_to_config()
            return copy.deepcopy(self.config)
        finally:
            self.config = original_config
            for name, value in saved.items():
                if value is missing:
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = value
            if had_sync_override:
                self.__dict__['_sync_custom_prefix_routes_env'] = saved_sync
            else:
                self.__dict__.pop('_sync_custom_prefix_routes_env', None)
