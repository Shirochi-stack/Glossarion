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
        settings_map = [
            # Basic settings
            ('model', ['model_var'], None, str),
            ('active_profile', ['profile_var'], None, str),
            ('prompt_profiles', ['prompt_profiles'], {}, dict),
            ('contextual', ['contextual_var'], None, bool),
            ('api_key', ['api_key_entry'], '', str),
            ('chapter_range', ['chapter_range_entry'], '', str),
            ('use_spine_order', ['use_spine_order_checkbox'], False, bool),

            # Numeric settings
            ('delay', ['delay_entry'], 5.0, lambda v: safe_float(v, 5.0)),
            ('api_queue', ['api_queue_entry', 'api_queue_var'], 4, lambda v: max(-1, safe_int(v, 4))),
            ('thread_submission_delay', ['thread_delay_entry'], '0.0001', _format_plain_decimal_setting),
            ('translation_temperature', ['trans_temp'], 0.3, lambda v: safe_float(v, 0.3)),
            ('disable_temperature', ['disable_temperature_checkbox', 'disable_temperature_var'], False, bool),
            ('translation_history_limit', ['trans_history'], 2, lambda v: safe_int(v, 2)),

            ('break_split_count', ['break_split_count_var'], '', str),
            ('duplicate_lookback_chapters', ['duplicate_lookback_var'], 5, lambda v: safe_int(v, 5)),

            # Boolean toggles - prioritize checkboxes over vars
            ('REMOVE_AI_ARTIFACTS', ['remove_artifacts_combo', 'REMOVE_AI_ARTIFACTS_var'], 'off', str),
            ('attach_css_to_chapters', ['attach_css_to_chapters_var'], False, bool),
            ('epub_use_html_method', ['epub_use_html_method_var'], False, bool),
            # Optional path to a user-selected CSS file for EPUB converter
            ('epub_css_override_path', ['epub_css_override_path_var'], '', str),
            ('use_rolling_summary', ['rolling_summary_var'], False, bool),
            # Whether to reuse previous source text as memory/history context
            ('include_source_in_history', ['include_source_in_history_var'], False, bool),
            ('translate_book_title', ['translate_book_title_var'], False, bool),
            ('skip_txt_title_translation', ['skip_txt_title_translation_var'], False, bool),
            ('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], False, bool),
            ('skip_image_title_translation', ['skip_image_title_translation_var'], True, bool),
            ('skip_title_tag_translation', ['skip_title_tag_translation_var'], False, bool),
            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),
            ('emergency_image_restore', ['emergency_image_restore_var'], False, bool),
            ('emergency_glossary_compliance', ['emergency_glossary_compliance_var'], False, bool),
            ('emergency_glossary_compliance_mode', ['emergency_glossary_compliance_mode_var'], 'characters', str),
            ('emergency_glossary_compliance_custom_types', ['emergency_glossary_compliance_custom_types_var'], [], list),
            ('emergency_glossary_compliance_min_chars', ['emergency_glossary_compliance_min_chars_var'], 3, lambda v: max(0, min(10, safe_int(v, 3)))),
            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),
            ('token_limit_disabled', ['token_limit_disabled'], False, bool),
            ('enable_thoughts', ['enable_thoughts_var'], True, bool),
            ('gemini_service_tier', ['gemini_service_tier_var'], 'off', str),
            ('translation_history_rolling', ['translation_history_rolling_var'], True, bool),
            ('disable_epub_gallery', ['disable_epub_gallery_var'], True, bool),
            ('skip_non_spine_special_files', ['skip_non_spine_special_files_var'], False, bool),
            ('skip_unreferenced_epub_images', ['skip_unreferenced_epub_images_var'], False, bool),
            ('disable_automatic_cover_creation', ['disable_automatic_cover_creation_var'], True, bool),
            ('use_toc_ncx', ['use_toc_ncx_var'], True, bool),
            ('translate_toc_ncx', ['translate_toc_ncx_var'], True, bool),
            ('use_p_tag_toc_fallback', ['use_p_tag_toc_fallback_var'], False, bool),
            ('deduplicate_toc', ['deduplicate_toc_var'], False, bool),
            ('deduplicate_toc_use_translated', ['deduplicate_toc_use_translated_var'], False, bool),
            ('skip_duplicate_toc_translation', ['skip_duplicate_toc_translation_var'], False, bool),
            ('duplicate_detection_mode', ['duplicate_detection_mode_var'], 'off', str),
            ('use_header_as_output', [], False, bool),
            ('enable_decimal_chapters', ['enable_decimal_chapters_var'], False, bool),
            ('force_ncx_only', ['force_ncx_only_var'], False, bool),
            ('batch_translate_headers', ['batch_translate_headers_var'], True, bool),
            ('enable_chunk_progress', ['enable_chunk_progress_checkbox', 'enable_chunk_progress_var'], True, bool),
            ('update_html_headers', ['update_html_headers_var'], False, bool),
            ('save_header_translations', ['save_header_translations_var'], False, bool),
            ('use_sorted_fallback', ['use_sorted_fallback_var'], False, bool),
            ('allow_ai_markdown_headers', ['allow_ai_markdown_headers_var'], False, bool),
            ('enable_translation_chunk_prompt', ['enable_translation_chunk_prompt_checkbox', 'enable_translation_chunk_prompt_var'], False, bool),
            ('include_previous_chunk', ['include_previous_chunk_checkbox', 'include_previous_chunk_var'], False, bool),
            ('previous_chunk_context_limit', ['previous_chunk_context_limit_spin', 'previous_chunk_context_limit_var'], 3, lambda v: max(-1, safe_int(v, 3))),
            ('translation_chunk_prompt_role', ['translation_chunk_prompt_role_var'], 'assistant', str),
            ('single_api_image_chunks', ['single_api_image_chunks_var'], False, bool),
            ('vision_ocr_batch_translation', ['vision_ocr_batch_translation_var'], True, bool),
            ('vision_ocr_skip_translation', ['vision_ocr_skip_translation_var'], False, bool),
            ('vision_ocr_keep_images', ['vision_ocr_keep_images_var'], False, bool),
            ('vision_ocr_source_prepass', ['vision_ocr_source_prepass_var'], 'auto', str),
            ('use_custom_openai_endpoint', ['use_custom_openai_endpoint_var'], False, bool),
            ('authza_use_general_api', ['authza_use_general_api_var'], False, bool),
            ('use_custom_image_edit_endpoint', ['use_custom_image_edit_endpoint_var'], False, bool),
            ('custom_image_edit_full_page_output', ['custom_image_edit_full_page_output_var'], 10, lambda v: max(0, min(100, safe_int(v, 10)))),
            ('manga_disable_inpaint_performance_mode', ['manga_disable_inpaint_performance_mode_var'], False, bool),
            ('use_inpainter_keys', ['use_inpainter_keys_var'], False, bool),
            ('disable_chapter_merging', ['disable_chapter_merging_var'], False, bool),
            # Request merging settings
            ('request_merging_enabled', ['request_merging_enabled_var'], False, bool),
            ('request_merge_count', ['request_merge_count_var'], 3, lambda v: safe_int(v, 3)),
            ('split_the_merge', ['split_the_merge_var'], True, bool),
            ('disable_merge_fallback', ['disable_merge_fallback_var'], True, bool),
            ('synthetic_merge_headers', ['synthetic_merge_headers_var'], True, bool),
            ('use_gemini_openai_endpoint', ['use_gemini_openai_endpoint_var'], False, bool),
            ('use_fallback_keys', ['use_fallback_keys_var'], False, bool),
            ('use_glossary_keys', ['use_glossary_keys_var'], False, bool),
            ('use_glossary_refinement_keys', ['use_glossary_refinement_keys_var'], False, bool),
            ('use_metadata_keys', ['use_metadata_keys_var'], False, bool),
            ('use_qa_scan_keys', ['use_qa_scan_keys_var'], False, bool),
            ('use_ai_truncation_detection_keys', ['use_ai_truncation_detection_keys_var'], False, bool),
            ('use_rolling_summary_keys', ['use_rolling_summary_keys_var'], False, bool),
            ('use_truncation_retry_keys', ['use_truncation_retry_keys_var'], False, bool),
            ('auto_update_check', ['auto_update_check_var'], True, bool),
            ('auto_dpi_scale', ['auto_dpi_scale_var'], True, bool),
            ('gui_scale_factor', ['gui_scale_factor_var'], 1.0, lambda v: safe_float(v, 1.0)),
            ('gui_font_scale', ['gui_font_scale_var'], 1.0, lambda v: safe_float(v, 1.0)),
            ('ignore_header', ['ignore_header_var'], False, bool),
            ('use_title', ['use_title_var'], True, bool),
            ('scan_phase_enabled', ['scan_phase_enabled_var'], True, bool),
            ('disable_qa_marker_checks', ['disable_qa_marker_checks_var'], True, bool),
            ('qa_marker_length_limit', ['qa_marker_length_limit_var'], 500, lambda v: safe_int(v, 500)),
            ('disable_refusal_checks', ['disable_refusal_checks_var'], True, bool),
            ('refusal_pattern_length_limit', ['refusal_pattern_length_limit_var'], 1000, lambda v: safe_int(v, 1000)),
            ('save_partial_results', ['save_partial_results_var'], True, bool),
            ('save_prohibited_results', ['save_prohibited_results_var'], False, bool),
            ('disable_empty_safety_heuristic', ['disable_empty_safety_heuristic_var'], True, bool),
            ('missing_finish_as_prohibited', ['unknown_finish_as_prohibited_var'], False, bool),

            # Prompts and text fields
            ('summary_role', ['summary_role_var'], '', str),
            ('book_title_prompt', ['book_title_prompt'], '', str),
            ('translation_chunk_prompt', ['translation_chunk_prompt'], '', str),
            ('image_chunk_prompt', ['image_chunk_prompt'], '', str),
            ('image_only_title_tag_system_prompt', ['image_only_title_tag_system_prompt'], DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT, str),
            ('vision_ocr_prompt', ['vision_ocr_prompt'], getattr(self, 'default_vision_ocr_prompt', ''), str),
            ('vision_ocr_user_prompt', ['vision_ocr_user_prompt'], getattr(self, 'default_vision_ocr_user_prompt', ''), str),
            ('vision_ocr_combined_context_prompt', ['vision_ocr_combined_context_prompt'], getattr(self, 'default_vision_ocr_combined_context_prompt', ''), str),
            ('vision_ocr_translation_user_prompt', ['vision_ocr_translation_user_prompt'], getattr(self, 'default_vision_ocr_translation_user_prompt', ''), str),
            ('assistant_prompt', ['assistant_prompt'], '', str),  # Optional assistant prefill
            ('vertex_ai_location', ['vertex_location_entry', 'vertex_location_var'], 'global', str),
            ('openai_base_url', ['openai_base_url_var'], '', str),
            ('custom_image_edit_endpoint', ['custom_image_edit_endpoint_var'], '', str),
            ('custom_image_edit_system_prompt', ['custom_image_edit_system_prompt_var'], '', str),
            ('custom_image_edit_user_prompt', ['custom_image_edit_user_prompt_var'], '', str),
            ('refinement_system_prompt', ['refinement_system_prompt'], getattr(self, 'default_refinement_system_prompt', ''), str),
            ('refinement_user_prompt', ['refinement_user_prompt'], '', str),
            ('refinement_full_with_raw_system_prompt', ['refinement_full_with_raw_system_prompt'], getattr(self, 'default_refinement_full_with_raw_system_prompt', ''), str),
            ('refinement_full_with_raw_user_prompt', ['refinement_full_with_raw_user_prompt'], getattr(self, 'default_refinement_full_with_raw_user_prompt', ''), str),
            ('refinement_full_with_raw_raw_role', ['refinement_full_with_raw_raw_role_var'], 'assistant', lambda v: str(v).strip().lower() if str(v).strip().lower() in REFINEMENT_RAW_PROMPT_ROLES else 'assistant'),
            ('refinement_full_with_raw_raw_header', ['refinement_full_with_raw_raw_header'], getattr(self, 'default_refinement_full_with_raw_raw_header', ''), str),
            ('refinement_full_with_raw_raw_footer', ['refinement_full_with_raw_raw_footer'], getattr(self, 'default_refinement_full_with_raw_raw_footer', ''), str),
            ('refinement_failed_system_prompt', ['refinement_failed_system_prompt'], getattr(self, 'default_refinement_failed_system_prompt', ''), str),
            ('refinement_failed_user_prompt', ['refinement_failed_user_prompt'], getattr(self, 'default_refinement_failed_user_prompt', ''), str),
            ('refinement_partial_system_prompt', ['refinement_partial_system_prompt'], getattr(self, 'default_refinement_partial_system_prompt', ''), str),
            ('refinement_partial_user_prompt', ['refinement_partial_user_prompt'], getattr(self, 'default_refinement_partial_user_prompt', ''), str),
            ('refinement_partial_b_system_prompt', ['refinement_partial_b_system_prompt'], getattr(self, 'default_refinement_partial_b_system_prompt', ''), str),
            ('refinement_partial_b_user_prompt', ['refinement_partial_b_user_prompt'], getattr(self, 'default_refinement_partial_b_user_prompt', ''), str),
            ('refinement_partial_b2_system_prompt', ['refinement_partial_b2_system_prompt'], getattr(self, 'default_refinement_partial_b2_system_prompt', ''), str),
            ('refinement_partial_b2_user_prompt', ['refinement_partial_b2_user_prompt'], getattr(self, 'default_refinement_partial_b2_user_prompt', ''), str),
            ('inpainter_keys', ['inpainter_keys_var'], [], list),
            ('rolling_summary_keys', ['rolling_summary_keys_var'], [], list),
            ('truncation_retry_keys', ['truncation_retry_keys_var'], [], list),
            ('openai_tts_endpoint', ['openai_tts_endpoint_var'], '', str),
            ('tts_voice', ['tts_voice_var'], '', str),
            ('groq_base_url', ['groq_base_url_var'], '', str),
            ('fireworks_base_url', ['fireworks_base_url_var'], '', str),
            ('gemini_openai_endpoint', ['gemini_openai_endpoint_var'], 'generativelanguage.googleapis.com', str),
            ('force_native_anthropic', ['force_native_anthropic_var'], False, bool),
            ('anthropic_base_url', ['anthropic_base_url_var'], '', str),
            ('fuzzy_auto_mapping', ['fuzzy_auto_mapping_var'], False, bool),
            ('fuzzy_auto_mapping_threshold', ['fuzzy_auto_mapping_threshold_var'], 80, lambda v: safe_int(v, 80)),

            # Review settings
            ('review_system_prompt', ['review_system_prompt_var'], '', str),
            ('review_spoiler_mode', ['review_spoiler_mode_var'], False, bool),
            ('review_chunk_mode', ['review_chunk_mode_var'], False, bool),
            ('review_chunk_wrap', ['review_chunk_wrap_var'], True, bool),
            ('review_volume_mode', ['review_volume_mode_var'], False, bool),
            ('review_final_prompt', ['review_final_prompt_var'], '', str),

            # Image settings
            ('enable_image_translation', ['enable_image_translation_var'], False, bool),
            ('process_webnovel_images', ['process_webnovel_images_var'], False, bool),
            ('webnovel_min_height', ['webnovel_min_height_var'], 1000, lambda v: safe_int(v, 1000)),
            ('max_images_per_chapter_v2', ['max_images_per_chapter_var'], -1, lambda v: safe_int(v, -1)),
            ('enable_watermark_removal', ['enable_watermark_removal_var'], False, bool),
            ('save_cleaned_images', ['save_cleaned_images_var'], False, bool),
            ('advanced_watermark_removal', ['advanced_watermark_removal_var'], False, bool),
            # Image output mode
            ('enable_image_output_mode', ['enable_image_output_mode_var'], False, bool),
            ('enable_video_output_mode', ['enable_video_output_mode_var'], False, bool),
            ('enable_audio_output_mode', ['enable_audio_output_mode_var'], False, bool),
            ('enable_refinement_output_mode', ['enable_refinement_output_mode_var'], False, bool),
            ('output_mode', ['output_mode_var'], 'text', str),
            ('image_output_resolution', ['image_output_resolution_var'], '1K', str),
            ('nanogpt_video_duration', ['nanogpt_video_duration_var'], '60', str),
            ('nanogpt_video_resolution', ['nanogpt_video_resolution_var'], '720p', str),
            ('compression_factor', ['compression_factor_var'], 3.0, float),
            ('image_chunk_overlap', ['image_chunk_overlap_var'], 3.0, lambda v: safe_float(v, 3.0)),
            ('image_chunk_min_overlap_pixels', ['image_chunk_min_overlap_pixels_var'], 80, lambda v: max(80, safe_int(v, 80))),
            ('image_smart_chunking', ['image_smart_chunking_var'], True, bool),
            ('vision_ocr_fuzzy_chunk_dedupe', ['vision_ocr_fuzzy_chunk_dedupe_var'], False, bool),

            # Batching
            ('batch_translation', ['batch_checkbox', 'batch_translation_var'], True, bool),
            ('batch_size', ['batch_size_entry', 'batch_size_var'], 5, lambda v: safe_int(v, 5)),
            ('multipass_mode', ['multipass_checkbox', 'multipass_mode_var'], False, bool),
            ('multipass_refinement_mode', ['multipass_refinement_mode_combo', 'multipass_refinement_mode_var'], 'full', lambda v: str(v).strip().lower() if str(v).strip().lower() in MULTIPASS_REFINEMENT_MODES else 'full'),
            ('vision_ocr_batch_size', ['vision_ocr_batch_size_var'], -1, lambda v: max(-1, safe_int(v, -1))),
            ('batching_mode', ['batch_mode_var'], 'aggressive', str),
            ('batch_group_size', ['batch_group_size_var'], 3, lambda v: safe_int(v, 3)),
            ('headers_per_batch', ['headers_per_batch_var'], -1, lambda v: safe_int(v, -1)),
            ('toc_ncx_per_batch', ['toc_ncx_per_batch_var'], -1, lambda v: safe_int(v, -1)),
            ('failed_translation_retry_attempts', ['failed_translation_retry_attempts_var'], 3, lambda v: min(20, max(0, safe_int(v, 3)))),
            ('partial_b2_entries_per_request', ['partial_b2_entries_per_request_var'], -1, lambda v: safe_int(v, -1)),

            # NIM/AuthND runtime settings
            ('authnd_token_concurrency_auto', ['authnd_token_concurrency_auto_checkbox', 'authnd_token_concurrency_auto_var'], True, bool),
            ('authnd_token_concurrency', ['authnd_token_concurrency_var'], 1, lambda v: max(1, safe_int(v, 1))),
            ('authnd_token_subprocess_concurrency', ['authnd_token_subprocess_concurrency_var'], 1, lambda v: max(1, safe_int(v, 1))),
            ('authnd_token_timeout', ['authnd_token_timeout_var'], 180, lambda v: min(600, max(30, safe_int(v, 180)))),
            ('dispatch_order_timeout', ['dispatch_order_timeout_var'], 3, lambda v: min(120, max(0, safe_int(v, 3)))),
            ('tor_proxy_enabled', ['tor_proxy_enabled_checkbox', 'tor_proxy_enabled_var'], False, bool),
            ('gemini_free_adaptive_split', ['gemini_free_adaptive_split_checkbox', 'gemini_free_adaptive_split_var'], True, bool),
            ('gemini_free_html_text_node_transport', ['gemini_free_html_text_node_transport_checkbox', 'gemini_free_html_text_node_transport_var'], True, bool),
            ('gemini_free_subchunk_prompt_chars', ['gemini_free_subchunk_prompt_chars_var'], 7000, lambda v: min(7000, max(300, safe_int(v, 7000)))),
            ('gemini_free_subchunk_url_chars', ['gemini_free_subchunk_url_chars_var'], 14500, lambda v: min(200000, max(1000, safe_int(v, 14500)))),
            ('gemini_free_subchunk_safety_chars', ['gemini_free_subchunk_safety_chars_var'], 600, lambda v: min(20000, max(0, safe_int(v, 600)))),
            ('gemini_free_min_subchunk_body_chars', ['gemini_free_min_subchunk_body_chars_var'], 80, lambda v: min(50000, max(1, safe_int(v, 80)))),
            ('gemini_free_subchunk_concurrency', ['gemini_free_subchunk_concurrency_var'], 0, lambda v: min(64, max(0, safe_int(v, 0)))),
            ('gemini_free_subchunk_start_delay', ['gemini_free_subchunk_start_delay_var'], 5.0, lambda v: min(60.0, max(0.0, safe_float(v, 5.0)))),
            ('gemini_free_subchunk_timeout', ['gemini_free_subchunk_timeout_var'], 0, lambda v: min(7200, max(0, safe_int(v, 0)))),
            ('gemini_free_subchunk_payload_format', ['gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format_combo'], 'auto', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('auto', 'html', 'text') else 'auto'),
            ('gemini_free_html_splitter', ['gemini_free_html_splitter_var', 'gemini_free_html_splitter_combo'], 'beautifulsoup4', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('beautifulsoup4', 'regex') else 'beautifulsoup4'),
            ('gemini_free_subchunk_balancer', ['gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer_combo'], 'balanced', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('balanced', 'greedy') else 'balanced'),

            # Gemini/GPT/DeepSeek Thinking
            ('enable_gemini_thinking', ['enable_gemini_thinking_var'], False, bool),
            ('thinking_budget', ['thinking_budget_var'], 0, lambda v: int(v) if str(v).lstrip('-').isdigit() else 0),
            ('thinking_level', ['thinking_level_var'], 'high', str),
            ('force_service_tier_unknown_routes', ['force_service_tier_unknown_routes_var'], False, bool),
            ('enable_gpt_thinking', ['enable_gpt_thinking_var'], False, bool),
            ('gpt_reasoning_tokens', ['gpt_reasoning_tokens_var'], 0, lambda v: int(v) if str(v).lstrip('-').isdigit() else 0),
            ('gpt_effort', ['gpt_effort_var'], 'auto', str),
            ('openrouter_use_reasoning_tokens', ['openrouter_use_reasoning_tokens_var'], False, bool),
            ('pass_thinking_all_openai', ['pass_thinking_all_openai_var'], False, bool),
            ('enable_deepseek_thinking', ['enable_deepseek_thinking_var'], True, bool),
            ('deepseek_effort', ['deepseek_effort_var'], 'high', str),
            ('deepseek_use_responses_api', ['deepseek_use_responses_api_var'], False, bool),
            # Anthropic extended/adaptive thinking
            ('enable_anthropic_thinking', ['enable_anthropic_thinking_var'], False, bool),
            ('anthropic_thinking_budget', ['anthropic_thinking_budget_var'], 10000, lambda v: int(v) if str(v).lstrip('-').isdigit() else 10000),
            ('anthropic_force_adaptive', ['anthropic_force_adaptive_var'], False, bool),
            ('anthropic_effort', ['anthropic_effort_var'], 'medium', str),
            # Skip thinking for lightweight tasks
            ('skip_book_title_thinking', ['skip_book_title_thinking_var'], True, bool),
            ('skip_metadata_thinking', ['skip_metadata_thinking_var'], True, bool),
            ('skip_toc_thinking', ['skip_toc_thinking_var'], False, bool),
            ('lightweight_thinking_level', ['lightweight_thinking_level_var'], 1, int),

            # Chapter processing
            ('chapter_number_offset', ['chapter_number_offset_var'], 0, lambda v: safe_int(v, 0)),
            ('max_output_tokens', ['max_output_tokens'], 128000, int),

            # Glossary Settings
            ('append_glossary', ['append_glossary_checkbox', 'append_glossary_var'], False, bool),
            ('append_glossary_auto_load', ['append_glossary_auto_load_checkbox', 'append_glossary_auto_load_var'], False, bool),
            ('add_additional_glossary', ['add_additional_glossary_checkbox', 'add_additional_glossary_var'], False, bool),
            ('additional_glossary_path', [('config', 'additional_glossary_path')], '', str),
            ('enable_unified_glossary', ['enable_unified_glossary_checkbox', 'enable_unified_glossary_var'], False, bool),
            ('generate_unified_glossary', ['generate_unified_glossary_checkbox', 'generate_unified_glossary_var'], False, bool),
            ('unified_glossary_source_language', ['unified_glossary_source_language_var'], 'auto', str),
            ('unified_glossary_combine_all_languages', ['unified_combine_all_languages_checkbox', 'unified_glossary_combine_all_languages_var'], False, bool),
            ('unified_glossary_exclude_gender_entries', ['unified_exclude_gender_entries_checkbox', 'unified_glossary_exclude_gender_entries_var'], True, bool),
            ('compress_glossary_prompt', ['compress_glossary_checkbox', 'compress_glossary_prompt_var'], True, bool),
            ('compress_glossary_strict_matching_mode', ['compress_glossary_strict_matching_mode_var'], 'all', str),
            ('compress_glossary_strict_matching_custom_types', ['compress_glossary_strict_matching_custom_types_var'], [], list),
            ('compress_glossary_consider_translated_column', ['consider_translated_compression_checkbox', 'compress_glossary_consider_translated_column_var'], False, bool),
            ('compress_glossary_multipass_exclude_matching', ['multipass_exclude_matching_checkbox', 'compress_glossary_multipass_exclude_matching_var'], True, bool),
            ('save_glossary_in_output', ['save_glossary_in_output_checkbox', 'save_glossary_in_output_var'], False, bool),
            ('include_gender_context', ['include_gender_context_checkbox', 'include_gender_context_var'], False, bool),
            ('enable_gender_nuance', ['enable_gender_nuance_checkbox', 'enable_gender_nuance_var'], True, bool),
            ('include_description', ['include_description_checkbox', 'include_description_var'], False, bool),
            ('glossary_use_smart_filter', ['glossary_use_smart_filter_var'], True, bool),
            ('glossary_min_frequency', ['glossary_min_frequency_entry', 'glossary_min_frequency_var'], 2, lambda v: safe_int(v, 2)),
            ('glossary_max_names', ['glossary_max_names_entry', 'glossary_max_names_var'], 50, lambda v: safe_int(v, 50)),
            ('glossary_max_titles', ['glossary_max_titles_entry', 'glossary_max_titles_var'], 30, lambda v: safe_int(v, 30)),
            ('context_window_size', ['context_window_size_entry', 'context_window_size_var'], 5, lambda v: safe_int(v, 5)),
            ('glossary_max_text_size', ['glossary_max_text_size_entry', 'glossary_max_text_size_var'], 50000, lambda v: safe_int(v, 50000)),
            ('glossary_chapter_split_threshold', ['glossary_chapter_split_threshold_entry', 'glossary_chapter_split_threshold_var'], 8192, lambda v: safe_int(v, 8192)),
            ('glossary_max_sentences', ['glossary_max_sentences_entry', 'glossary_max_sentences_var'], 200, lambda v: safe_int(v, 200)),
            ('strip_honorifics', ['strip_honorifics_checkbox', 'strip_honorifics_var'], False, bool),
            ('glossary_disable_honorifics_filter', ['disable_honorifics_checkbox', 'disable_honorifics_var'], False, bool),
            ('manual_glossary_temperature', ['manual_temp_entry', 'manual_temp_var'], 0.3, lambda v: safe_float(v, 0.3)),
            ('manual_context_limit', ['manual_context_entry', 'manual_context_var'], 5, lambda v: safe_int(v, 5)),
            ('glossary_history_rolling', ['glossary_history_rolling_var'], True, bool),
            ('disable_glossary_history', ['disable_glossary_history_checkbox', 'disable_glossary_history_var'], True, bool),
            ('glossary_skip_title_header_only', ['glossary_skip_title_header_only_checkbox', 'glossary_skip_title_header_only_var'], True, bool),
            ('enable_auto_glossary', ['enable_auto_glossary_checkbox', 'enable_auto_glossary_var'], False, bool),
            ('auto_glossary_mode', ['auto_glossary_mode_var'], 'balanced', str),
            ('glossary_use_legacy_csv', ['use_legacy_csv_checkbox', 'use_legacy_csv_var'], False, bool),
            ('glossary_output_legacy_json', ['glossary_output_legacy_json_var'], False, bool),
            ('glossary_include_all_characters', ['glossary_include_all_characters_var'], False, bool),
            ('glossary_skip_identical_entries', ['glossary_skip_identical_entries_var'], True, bool),
            ('glossary_cjk_script_filter', ['glossary_cjk_script_filter_var'], False, bool),
            ('glossary_skip_gender_tracking', ['skip_gender_tracking_checkbox', 'glossary_skip_gender_tracking_var'], False, bool),
            ('glossary_gender_noise_threshold', ['glossary_gender_noise_threshold_var'], 10, lambda v: safe_int(v, 10)),
            ('glossary_gender_tracking_bias', ['glossary_gender_tracking_bias_var'], 'none', str),
            ('glossary_filter_mode', ['glossary_filter_mode_var'], 'strict', str),
            ('scan_phase_mode', ['scan_phase_mode_var'], 'quick-scan', str),

            # EPUB layout mode
            ('epub_layout_mode', ['epub_layout_mode_var', ('config', 'epub_layout_mode')], 'auto', str),

            # EPUB reader settings (persisted by EpubReaderDialog.closeEvent)
            ('epub_reader_font_size', [('config', 'epub_reader_font_size')], 12, int),
            ('epub_reader_line_spacing', [('config', 'epub_reader_line_spacing')], 1.6, float),
            ('epub_reader_theme', [('config', 'epub_reader_theme')], 0, int),
            ('epub_reader_layout', [('config', 'epub_reader_layout')], 'single_page', str),
            ('epub_reader_native_toc', [('config', 'epub_reader_native_toc')], False, bool),
            # EPUB library settings (persisted by EpubLibraryDialog.closeEvent)
            ('epub_library_sort', [('config', 'epub_library_sort')], 'date', str),
            ('epub_library_card_size', [('config', 'epub_library_card_size')], 'compact', str),

            # Extraction settings - NOTE: these are only created in Other Settings dialog
            ('enable_parallel_extraction', ['enable_parallel_extraction_var'], False, bool),
            ('extraction_workers', ['extraction_workers_var'], 1, int),
            ('text_extraction_method', ['text_extraction_method_var'], 'standard', str),
            ('file_filtering_level', ['file_filtering_level_var'], 'smart', str),
            ('enhanced_preserve_structure', ['enhanced_preserve_structure_var'], True, bool),
            ('enhanced_single_line_break', ['enhanced_single_line_break_var'], False, bool),
            ('convert_br_to_paragraphs', ['convert_br_to_paragraphs_var'], True, bool),
            ('preserve_asterisk_separator_lines', ['preserve_asterisk_separator_lines_var'], True, bool),
            ('skip_markdown_to_html', ['skip_markdown_to_html_var'], False, bool),
            ('html2text_escape_snob', ['html2text_escape_snob_var'], False, bool),
            ('use_markdown2_converter', ['use_markdown2_converter_var'], False, bool),
            ('enhanced_filtering', ['enhanced_filtering_var'], 'smart', str), # Backwards compatibility
            ('force_bs_for_traditional', ['force_bs_for_traditional_var'], True, bool),  # Updated by other_settings.py
            ('output_sdlxliff', ['output_sdlxliff_var'], True, bool),
            ('output_md', ['output_md_var'], False, bool),
            ('output_txt', ['output_txt_var'], False, bool),
            ('fix_stray_p_gt_epub', ['fix_stray_p_gt_epub_var'], False, bool),
            ('fix_stray_p_gt_bs', ['fix_stray_p_gt_bs_var'], False, bool),

            # Stop behavior
            ('graceful_stop', ['graceful_stop_checkbox', 'graceful_stop_var'], False, bool),
            ('wait_for_chunks', ['wait_for_chunks_checkbox', 'wait_for_chunks_var'], True, bool),
            ('save_partial_results', ['save_partial_results_checkbox', 'save_partial_results_var'], True, bool),
            ('save_prohibited_results', ['save_prohibited_results_checkbox', 'save_prohibited_results_var'], False, bool),
            ('disable_empty_safety_heuristic', ['disable_empty_safety_heuristic_checkbox', 'disable_empty_safety_heuristic_var'], True, bool),
            ('missing_finish_as_prohibited', ['unknown_finish_as_prohibited_checkbox', 'unknown_finish_as_prohibited_var'], False, bool),

            # HTTP/Network tuning - prioritize entry widgets over vars
            ('chunk_timeout', ['chunk_timeout_var'], 1800, lambda v: safe_int(v, 1800)),
            ('enable_http_tuning', ['http_tuning_checkbox', 'enable_http_tuning_var'], False, bool),
            ('connect_timeout', ['connect_timeout_entry', 'connect_timeout_var'], 10.0, lambda v: safe_float(v, 10.0)),
            ('read_timeout', ['read_timeout_entry', 'read_timeout_var'], 180.0, lambda v: safe_float(v, 180.0)),
            ('http_pool_connections', ['http_pool_connections_entry', 'http_pool_connections_var'], 20, lambda v: safe_int(v, 20)),
            ('http_pool_maxsize', ['http_pool_maxsize_entry', 'http_pool_maxsize_var'], 50, lambda v: safe_int(v, 50)),
            ('ignore_retry_after', ['ignore_retry_after_checkbox', 'ignore_retry_after_var'], False, bool),
            ('enable_streaming', ['enable_streaming_checkbox', 'enable_streaming_var'], False, bool),
            ('allow_batch_stream_logs', ['allow_batch_stream_logs_checkbox', 'allow_batch_stream_logs_var'], False, bool),
            ('stream_thinking_logs', ['stream_thinking_logs_checkbox', 'stream_thinking_logs_var'], False, bool),
            ('max_retries', ['max_retries_var'], 7, lambda v: safe_int(v, 7)),
            ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),

            # Retry settings
            ('retry_truncated', ['retry_truncated_var'], False, bool),
            # Char-ratio truncation (silent truncation detector)
            ('char_ratio_truncation_enabled', ['char_ratio_truncation_var'], True, bool),
            ('char_ratio_truncation_percent', ['char_ratio_truncation_percent_var'], 50, lambda v: safe_int(v, 50)),
            ('char_ratio_truncation_attempts', ['char_ratio_truncation_attempts_var'], 1, lambda v: safe_int(v, 1)),
            ('char_ratio_min_output_chars', ['char_ratio_min_output_chars_var'], 100, lambda v: safe_int(v, 100)),
            ('retry_split_failed', ['retry_split_failed_var'], True, bool),
            ('max_retry_tokens', ['max_retry_tokens_var'], -1, lambda v: safe_int(v, -1)),
            ('truncation_retry_attempts', ['truncation_retry_attempts_var'], 3, lambda v: safe_int(v, 3)),
            ('split_failed_retry_attempts', ['split_failed_retry_attempts_var'], 1, lambda v: safe_int(v, 1)),
            ('retry_timeout', ['retry_timeout_var'], False, bool),
            ('timeout_retry_attempts', ['timeout_retry_attempts_var'], 2, lambda v: safe_int(v, 2)),
            ('preserve_original_text_on_failure', ['preserve_original_text_var'], False, bool),

            # Rolling summary
            ('rolling_summary_exchanges', ['rolling_summary_exchanges_edit', 'rolling_summary_exchanges_var'], 5, lambda v: safe_int(v, 5)),
            ('rolling_summary_mode', ['rolling_summary_mode_var'], 'replace', str),
            ('rolling_summary_max_entries', ['rolling_summary_retain_edit', 'rolling_summary_max_entries_var'], 10, lambda v: safe_int(v, 10)),
            ('rolling_summary_max_tokens', ['rolling_summary_max_tokens_var'], 8192, lambda v: safe_int(v, 8192)),

            # QA/Scanning
            ('qa_auto_search_output', ['qa_auto_search_output_checkbox', 'qa_auto_search_output_var'], True, bool),
            ('disable_zero_detection', ['disable_zero_detection_var'], False, bool),
            ('disable_gemini_safety', ['disable_gemini_safety_var'], False, bool),
            ('gemini_safety_threshold', ['gemini_safety_threshold_var'], 'BLOCK_NONE', str),

            # Anti-duplicate parameters - all vars updated by other_settings.py callbacks
            ('enable_anti_duplicate', ['enable_anti_duplicate_var'], False, bool),
            ('top_p', ['top_p_var'], 1.0, float),
            ('min_p', ['min_p_var'], 0.0, float),
            ('bypass_min_p_allowlist', ['bypass_min_p_allowlist_var'], False, bool),
            ('top_k', ['top_k_var'], 50, int),
            ('frequency_penalty', ['frequency_penalty_var'], 0.0, float),
            ('presence_penalty', ['presence_penalty_var'], 0.0, float),
            ('repetition_penalty', ['repetition_penalty_var'], 1.0, float),
            ('candidate_count', ['candidate_count_var'], 1, int),
            ('custom_stop_sequences', ['custom_stop_sequences_var'], '', str),
            ('logit_bias_enabled', ['logit_bias_enabled_var'], False, bool),
            ('logit_bias_strength', ['logit_bias_strength_var'], 1.0, float),
            ('bias_common_words', ['bias_common_words_var'], False, bool),
            ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),

            # Glossary anti-duplicate parameters (separate from translation)
            ('glossary_enable_anti_duplicate', ['glossary_enable_anti_duplicate_var', ('config', 'glossary_enable_anti_duplicate')], False, bool),
            ('glossary_top_p', ['glossary_top_p_var', ('config', 'glossary_top_p')], 1.0, float),
            ('glossary_min_p', ['glossary_min_p_var', ('config', 'glossary_min_p')], 0.0, float),
            ('glossary_bypass_min_p_allowlist', ['glossary_bypass_min_p_allowlist_var', ('config', 'glossary_bypass_min_p_allowlist')], False, bool),
            ('glossary_top_k', ['glossary_top_k_var', ('config', 'glossary_top_k')], 0, int),
            ('glossary_frequency_penalty', ['glossary_frequency_penalty_var', ('config', 'glossary_frequency_penalty')], 0.0, float),
            ('glossary_presence_penalty', ['glossary_presence_penalty_var', ('config', 'glossary_presence_penalty')], 0.0, float),
            ('glossary_repetition_penalty', ['glossary_repetition_penalty_var', ('config', 'glossary_repetition_penalty')], 1.0, float),
            ('glossary_candidate_count', ['glossary_candidate_count_var', ('config', 'glossary_candidate_count')], 1, int),
            ('glossary_custom_stop_sequences', ['glossary_custom_stop_sequences_var', ('config', 'glossary_custom_stop_sequences')], '', str),
            ('glossary_logit_bias_enabled', ['glossary_logit_bias_enabled_var', ('config', 'glossary_logit_bias_enabled')], False, bool),
            ('glossary_logit_bias_strength', ['glossary_logit_bias_strength_var', ('config', 'glossary_logit_bias_strength')], -0.5, float),
            ('glossary_bias_common_words', ['glossary_bias_common_words_var', ('config', 'glossary_bias_common_words')], False, bool),
            ('glossary_bias_repetitive_phrases', ['glossary_bias_repetitive_phrases_var', ('config', 'glossary_bias_repetitive_phrases')], False, bool),

            # OpenRouter
            ('openrouter_use_http_only', ['openrouter_http_only_var'], False, bool),
            ('openrouter_accept_identity', ['openrouter_accept_identity_var'], False, bool),
            ('openrouter_preferred_provider', ['openrouter_preferred_provider_var', ('config', 'openrouter_preferred_provider')], 'Auto', lambda v: (str(v).strip() if v is not None else '') or 'Auto'),

            # Environment-backed settings
            ('retain_source_extension', ['retain_source_extension_var'], False, bool),
            ('download_remote_image_urls', ['download_remote_image_urls_var'], False, bool),
            ('remote_image_download_workers', ['remote_image_download_workers_var'], 4, lambda v: max(1, min(32, safe_int(v, 4)))),
            ('remote_image_download_interval', ['remote_image_download_interval_var'], 0.5, lambda v: max(0.0, min(60.0, safe_float(v, 0.5)))),
            ('enable_gui_yield', ['enable_gui_yield_var'], True, bool),
            ('use_thread_pool_extraction', ['use_thread_pool_extraction_var'], False, bool),

            # File selection settings
            ('deep_scan', ['deep_scan_check', 'deep_scan_var'], False, bool),

            # Async processing settings
            ('async_wait_for_completion', ['async_wait_for_completion_var'], False, bool),
            ('async_poll_interval', ['async_poll_interval_var'], 60, lambda v: safe_int(v, 60)),

            # PDF settings
            ('pdf_output_format', ['pdf_output_format_var'], 'pdf', str),
            ('pdf_render_mode', ['pdf_render_mode_var'], 'fast_semantic', str),
            ('pdf_use_toc_sections', ['pdf_use_toc_sections_var'], True, bool),
            ('pdf_async_page_threshold', ['pdf_async_page_threshold_var'], '100', str),
            ('pdf_extraction_workers', ['pdf_extraction_workers_var'], 'auto', str),
            ('pdf_paragraph_alignment', ['pdf_paragraph_alignment_var'], 'source', str),
            ('pdf_header_alignment', ['pdf_header_alignment_var'], 'source', str),
            ('pdf_paragraph_justification', ['pdf_paragraph_justification_var'], 'source', str),
            ('pdf_rtl_paragraph_layout', ['pdf_rtl_paragraph_layout_var'], False, bool),
        ]

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
