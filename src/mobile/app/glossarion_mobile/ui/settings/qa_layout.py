"""The desktop QA Scanner's control order, divided into four mobile tabs."""

SDLXLIFF_TAG_FIELDS = (
    "qa_scanner_settings.sdlxliff_tag_retention_threshold",
    "qa_scanner_settings.sdlxliff_tag_surplus_tolerance",
    "qa_scanner_settings.sdlxliff_min_source_paragraph_tags",
)

QA_TABS = (
    ("Detection", (
        ("Foreign character detection", """
            source_language target_language foreign_char_threshold whitelist_emoticon_patterns
            exclude_ruby_tags emoticon_patterns emoticon_patterns_are_regex excluded_characters
        """),
        ("Detection options", """
            check_encoding_issues check_repetition check_translation_artifacts
            check_ai_artifacts ai_artifact_patterns ai_artifact_patterns_are_regex
            check_ai_thinking_preamble ai_thinking_preamble_patterns ai_thinking_preamble_patterns_are_regex
            ai_thinking_preamble_sample_size check_punctuation_mismatch punctuation_loss_threshold
            flag_excess_punctuation excess_punctuation_threshold check_quotation_mismatch
            ignore_excess_quotation_marks only_check_incomplete_quotations ignore_consecutive_missing_quotations
            skip_stylistic_single_quotes include_square_brackets_as_quotations
        """),
    )),
    ("Files & tags", (
        ("Glossary and tag checks", """
            check_glossary_leakage check_glossary_names check_potential_truncation check_missing_beautifulsoup_tags
            sdlxliff_tag_retention_threshold sdlxliff_tag_surplus_tolerance sdlxliff_min_source_paragraph_tags
        """),
        ("File processing", "min_file_length min_duplicate_word_count min_text_length_for_spacing"),
        ("Word count analysis", """
            check_word_count_ratio counting_mode use_auto_multipliers word_count_multipliers
            warn_name_mismatch check_missing_images
        """),
        ("Duplicate checks", """
            quick_scan_sample_size custom_mode_settings.check_all_pairs custom_mode_settings.consecutive_chapters
            custom_mode_settings.min_duplicate_word_count custom_mode_settings.min_text_length
            custom_mode_settings.sample_size custom_mode_settings.thresholds.minhash_threshold
            custom_mode_settings.thresholds.semantic custom_mode_settings.thresholds.similarity
            custom_mode_settings.thresholds.structural custom_mode_settings.thresholds.word_overlap
        """),
    )),
    ("More checks", (
        ("Additional checks", """
            check_missing_header_tags check_multiple_headers check_missing_html_tag check_body_tag
            check_all_text_in_header check_invalid_tag_mismatch check_invalid_nesting check_missing_tag_bracket
            check_silent_truncation truncation_length_threshold truncation_cheap_threshold
            truncation_embed_threshold truncation_borderline_score check_ai_truncation_detection
            ai_truncation_tail_chars ai_truncation_prompt_role ai_truncation_prompt
        """),
        ("AI truncation detector", """
            ai_truncation_api_key ai_truncation_model ai_truncation_temperature ai_truncation_max_tokens
            ai_truncation_api_call_delay ai_truncation_api_call_delay_max
            ai_truncation_endpoint_url ai_truncation_disable_thinking
        """),
    )),
    ("Report & speed", (
        ("Report settings", "report_format auto_save_report word_count_min_ratio word_count_max_ratio custom_output_suffixes"),
        ("Performance cache", """
            cache_enabled use_thread_executor cache_content_hashes cache_normalize_text cache_semantic_fingerprint
            cache_similarity_ratio cache_structural_signature cache_translation_artifacts cache_auto_size cache_show_stats
        """),
    )),
)


def qa_settings_layout(specs):
    by_key = {spec.key: spec for spec in specs}
    ordered, headings, tabs = [], {}, []
    for title, groups in QA_TABS:
        start = len(ordered)
        for heading, names in groups:
            for name in names.split():
                key = f"qa_scanner_settings.{name}"
                if key in by_key:
                    ordered.append(by_key.pop(key))
                    headings[key] = heading
        if title == QA_TABS[-1][0]:
            ordered.extend(by_key.values())
            headings.update(dict.fromkeys(by_key, "Other settings"))
        if start < len(ordered):
            tabs.append((title, start, len(ordered)))
    return ordered, headings, tabs
