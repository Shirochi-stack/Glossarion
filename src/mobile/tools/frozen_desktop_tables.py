"""Frozen copy of the three desktop settings tables (U9 P5b).

Until U9 the desktop held these tables as literals: ``settings_map`` in
``settings_persistence.SettingsPersistenceMixin._apply_live_settings_to_config`` (save_config
sections 2-3) and ``bool_vars`` / ``str_vars`` in ``owner_state.ConfigStateMixin._init_variables``
(U2 moved them there verbatim from ``translator_gui.TranslatorGUI``). P5b made the settings
schema their single source: those methods now call ``settings_schema.desktop_settings_map(self)``,
``desktop_bool_vars(self)`` and ``desktop_str_vars(self)``, which build the same tuples, in the
same order, from the generated ``settings_schema_data.DESKTOP_SETTINGS_MAP`` /
``DESKTOP_BOOL_VARS`` / ``DESKTOP_STR_VARS``.

The statements below are the literals copied line for line from 1cd681786382 (the last commit
that held them in owner_state.py / settings_persistence.py). They are

* the generator's input for the three tables: ``src/mobile/tools/schema_extract.py`` splices each
  literal back in place of its ``desktop_*()`` call before it reads the owner modules, so the
  generated schema data is exactly what the literals produced;
* the oracle of ``tests/test_schema_p5b.py`` (pinned to the literals at 1cd681786382; tuple
  equality, converter behaviour on fuzz inputs) and of the tier C converter check in
  ``tests/test_settings_schema.py``.

Changing a desktop table is a desktop change: edit the row here, regenerate
(``python src/mobile/tools/schema_extract.py``), and move the pins the tests name
(``P5B_BASE_SHA`` in tests/test_schema_p5b.py; ``_init_variables`` is also pinned by
tests/test_headless_owner.py::test_moved_methods_are_verbatim).

Never imported by the desktop or the mobile app (tests and the generator only). Python 3.10.
"""
# The converter lambdas reference these names exactly as settings_persistence.py did.
from run_env import MULTIPASS_REFINEMENT_MODES, REFINEMENT_RAW_PROMPT_ROLES, _format_plain_decimal_setting
from title_tag_translation import DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT

FROZEN_SHA = "1cd681786382abed23069f4c06cb583948371d1a"


# settings_persistence.safe_int / safe_float (save_config's local helpers), verbatim.
def safe_int(value, default):
    try: return int(value)
    except (ValueError, TypeError): return default

def safe_float(value, default):
    try: return float(value)
    except (ValueError, TypeError): return default


class FrozenDesktopTables:
    """Each method returns one table, built exactly as the desktop method built it
    (``self`` is the owner: TranslatorGUI / HeadlessOwner or a test double)."""

    def settings_map(self):
        """SettingsPersistenceMixin._apply_live_settings_to_config: the settings_map literal."""
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
        return settings_map

    def bool_vars(self):
        """ConfigStateMixin._init_variables: the bool_vars literal."""
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
        return bool_vars

    def str_vars(self):
        """ConfigStateMixin._init_variables: the str_vars literal."""
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
        return str_vars
