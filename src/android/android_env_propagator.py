# android_env_propagator.py
"""
Environment variable propagation for Glossarion Android.

Mirrors the logic from:
  - app.py set_all_environment_variables() (lines 606-800)
  - translator_gui.py run_translation_thread() (lines 9073-9086)
  
This ensures the translation engine (TransateKRtoEN.py) and unified_api_client
receive identical configuration regardless of which entry point is used.
"""

import os
import json
import logging
import uuid
import time

logger = logging.getLogger(__name__)


def set_all_env_vars(config):
    """Set all environment variables required by the translation engine.
    
    This is the Android equivalent of app.py's set_all_environment_variables()
    combined with translator_gui.py's per-run env vars.
    
    Args:
        config: dict — the current app configuration (from android_config)
    """
    def _get(key, default=None):
        return config.get(key, default)

    # ── Core translation settings ──
    os.environ['MODEL'] = str(_get('model', 'authgpt/gpt-5.2'))
    os.environ['API_KEY'] = str(_get('api_key', ''))
    os.environ['OPENAI_API_KEY'] = str(_get('api_key', ''))
    os.environ['TRANSLATION_TEMPERATURE'] = str(_get('translation_temperature', 0.3))
    os.environ['MAX_OUTPUT_TOKENS'] = str(_get('max_output_tokens', 128000))
    os.environ['OUTPUT_LANGUAGE'] = str(_get('output_language', 'English'))
    os.environ['SEND_INTERVAL_SECONDS'] = str(_get('api_call_delay', 0.5))
    os.environ['DELAY'] = str(_get('api_call_delay', 0.5))  # Backward compat

    # ── System prompt (from active profile) ──
    system_prompt = str(_get('active_system_prompt', ''))
    if not system_prompt:
        # Fall back to profile lookup
        profile_name = _get('active_profile', 'Universal')
        profiles = _get('prompt_profiles', {})
        system_prompt = profiles.get(profile_name, '')
    
    # Use large_env if prompt is very long (Windows 32k limit doesn't apply on Android,
    # but keep the pattern for consistency)
    os.environ['SYSTEM_PROMPT'] = system_prompt
    os.environ['PROFILE_NAME'] = str(_get('active_profile', 'Universal'))

    # ── Batch translation ──
    os.environ['BATCH_TRANSLATION'] = '1' if _get('batch_translation', False) else '0'
    os.environ['BATCH_SIZE'] = str(_get('batch_size', 10))
    batching_mode = _get('batching_mode', 'direct')
    os.environ['BATCHING_MODE'] = batching_mode
    os.environ['BATCH_GROUP_SIZE'] = str(_get('batch_group_size', 3))
    os.environ['USE_CONSERVATIVE_BATCHING'] = '1' if batching_mode == 'conservative' else '0'

    # ── Chapter extraction ──
    text_extraction_method = _get('text_extraction_method', 'standard')
    file_filtering_level = _get('file_filtering_level', 'smart')
    os.environ['TEXT_EXTRACTION_METHOD'] = text_extraction_method
    os.environ['FILE_FILTERING_LEVEL'] = file_filtering_level

    if text_extraction_method == 'enhanced':
        os.environ['EXTRACTION_MODE'] = 'enhanced'
    else:
        os.environ['EXTRACTION_MODE'] = file_filtering_level

    os.environ['ENHANCED_FILTERING'] = file_filtering_level
    os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1' if _get('enhanced_preserve_structure', True) else '0'
    os.environ['FORCE_BS_FOR_TRADITIONAL'] = '1' if _get('force_bs_for_traditional', True) else '0'

    # ── API / rate limiting ──
    os.environ['INDEFINITELY_RETRY_RATE_LIMIT'] = '1' if _get('indefinitely_retry_rate_limit', False) else '0'
    os.environ['THREAD_SUBMISSION_DELAY'] = str(_get('thread_submission_delay', 0.1))

    # ── Contextual / history ──
    os.environ['CONTEXTUAL'] = '1' if _get('contextual', False) else '0'
    os.environ['TRANSLATION_HISTORY_LIMIT'] = str(_get('translation_history_limit', 2))
    os.environ['TRANSLATION_HISTORY_ROLLING'] = '1' if _get('translation_history_rolling', False) else '0'

    # ── Streaming ──
    os.environ['ENABLE_STREAMING'] = '1' if _get('enable_streaming', False) else '0'
    os.environ['ALLOW_BATCH_STREAM_LOGS'] = '1' if _get('allow_batch_stream_logs', False) else '0'
    os.environ['ALLOW_AUTHGPT_BATCH_STREAM_LOGS'] = '1' if _get('allow_authgpt_batch_stream_logs', False) else '0'
    os.environ['LOG_STREAM_CHUNKS'] = '1'
    os.environ['ENABLE_THOUGHTS'] = '0'  # No thought display on mobile

    # ── Chapter processing ──
    os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if _get('batch_translate_headers', False) else '0'
    os.environ['HEADERS_PER_BATCH'] = str(_get('headers_per_batch', -1))
    os.environ['IGNORE_HEADER'] = '1' if _get('ignore_header', False) else '0'
    os.environ['USE_TITLE'] = '1' if _get('use_title', False) else '0'
    os.environ['USE_NCX_NAVIGATION'] = '1' if _get('use_ncx_navigation', False) else '0'
    os.environ['ATTACH_CSS_TO_CHAPTERS'] = '1' if _get('attach_css_to_chapters', False) else '0'
    os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if _get('retain_source_extension', True) else '0'
    os.environ['REMOVE_DUPLICATE_H1_P'] = '1' if _get('remove_duplicate_h1_p', False) else '0'
    os.environ['USE_SORTED_FALLBACK'] = '1' if _get('use_sorted_fallback', False) else '0'

    # ── Token limits ──
    os.environ['TOKEN_LIMIT'] = str(_get('token_limit', 200000))
    os.environ['TOKEN_LIMIT_DISABLED'] = '1' if _get('token_limit_disabled', False) else '0'
    os.environ['DISABLE_INPUT_TOKEN_LIMIT'] = '1' if _get('token_limit_disabled', False) else '0'
    os.environ['CHAPTER_RANGE'] = _get('chapter_range', '')

    # ── Glossary settings ──
    os.environ['ENABLE_AUTO_GLOSSARY'] = '1' if _get('enable_auto_glossary', True) else '0'
    os.environ['APPEND_GLOSSARY_TO_PROMPT'] = '1' if _get('append_glossary_to_prompt', True) else '0'
    os.environ['GLOSSARY_MIN_FREQUENCY'] = str(_get('glossary_min_frequency', 2))
    os.environ['GLOSSARY_MAX_NAMES'] = str(_get('glossary_max_names', 50))
    os.environ['GLOSSARY_MAX_TITLES'] = str(_get('glossary_max_titles', 30))
    os.environ['GLOSSARY_BATCH_SIZE'] = str(_get('glossary_batch_size', 50))
    os.environ['GLOSSARY_FILTER_MODE'] = _get('glossary_filter_mode', 'all')
    os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(_get('glossary_fuzzy_threshold', 0.90))

    # Manual glossary
    os.environ['MANUAL_GLOSSARY_MIN_FREQUENCY'] = str(_get('manual_glossary_min_frequency', 2))
    os.environ['MANUAL_GLOSSARY_MAX_NAMES'] = str(_get('manual_glossary_max_names', 50))
    os.environ['MANUAL_GLOSSARY_MAX_TITLES'] = str(_get('manual_glossary_max_titles', 30))
    os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(_get('glossary_max_text_size', 0))
    os.environ['GLOSSARY_MAX_SENTENCES'] = str(_get('glossary_max_sentences', 200))
    os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(_get('glossary_chapter_split_threshold', 0))
    os.environ['MANUAL_GLOSSARY_FILTER_MODE'] = _get('manual_glossary_filter_mode', 'all')
    os.environ['STRIP_HONORIFICS'] = '1' if _get('strip_honorifics', True) else '0'
    os.environ['MANUAL_GLOSSARY_FUZZY_THRESHOLD'] = str(_get('manual_glossary_fuzzy_threshold', 0.90))
    os.environ['GLOSSARY_USE_LEGACY_CSV'] = '1' if _get('glossary_use_legacy_csv', False) else '0'
    os.environ['COMPRESS_GLOSSARY_PROMPT'] = '1' if _get('compress_glossary_prompt', True) else '0'
    os.environ['GLOSSARY_INCLUDE_ALL_CHARACTERS'] = '1' if _get('glossary_include_all_characters', True) else '0'

    # Glossary append prompt
    append_gloss = _get('append_glossary_prompt', '')
    if not append_gloss:
        append_gloss = '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'
    os.environ['APPEND_GLOSSARY_PROMPT'] = append_gloss

    # Auto glossary prompt
    auto_gloss = _get('unified_auto_glosary_prompt3', '')
    if auto_gloss:
        os.environ['AUTO_GLOSSARY_PROMPT'] = auto_gloss

    # ── Thinking mode ──
    os.environ['ENABLE_GPT_THINKING'] = '1' if _get('enable_gpt_thinking', True) else '0'
    os.environ['GPT_THINKING_EFFORT'] = _get('gpt_thinking_effort', 'medium')
    os.environ['OR_THINKING_TOKENS'] = str(_get('or_thinking_tokens', 2000))
    os.environ['ENABLE_GEMINI_THINKING'] = '1' if _get('enable_gemini_thinking', False) else '0'
    os.environ['GEMINI_THINKING_BUDGET'] = str(_get('gemini_thinking_budget', 0))
    os.environ['THINKING_BUDGET'] = str(_get('gemini_thinking_budget', 0))

    # ── Provider-specific ──
    os.environ['DISABLE_GEMINI_SAFETY'] = '1' if _get('disable_gemini_safety', False) else '0'
    os.environ['USE_HTTP_OPENROUTER'] = '1' if _get('use_http_openrouter', False) else '0'
    os.environ['DISABLE_OPENROUTER_COMPRESSION'] = '1' if _get('disable_openrouter_compression', False) else '0'

    # ── Multi API key support ──
    os.environ['USE_MULTI_API_KEYS'] = '1' if _get('use_multi_api_keys', False) else '0'
    multi_keys = _get('multi_api_keys', [])
    if multi_keys:
        try:
            os.environ['MULTI_API_KEYS'] = json.dumps(multi_keys)
        except Exception:
            os.environ['MULTI_API_KEYS'] = '[]'
    else:
        os.environ['MULTI_API_KEYS'] = '[]'

    # ── Batch header translation prompts ──
    output_lang = _get('output_language', 'English')
    batch_header_sys = _get('batch_header_system_prompt', '').replace('{target_lang}', output_lang)
    batch_header_prompt = _get('batch_header_prompt', '').replace('{target_lang}', output_lang)
    os.environ['BATCH_HEADER_SYSTEM_PROMPT'] = batch_header_sys
    os.environ['BATCH_HEADER_PROMPT'] = batch_header_prompt

    # ── Concise logs ──
    os.environ['CONCISE_PIPELINE_LOGS'] = '1' if _get('concise_pipeline_logs', False) else '0'

    logger.info("All environment variables set for translation engine")


def set_per_run_env_vars():
    """Set environment variables that must be fresh for each translation run.
    
    These mirror translator_gui.py's run_translation_thread() per-run setup.
    """
    # Reset stop flags
    os.environ['GRACEFUL_STOP'] = '0'
    os.environ['GRACEFUL_STOP_COMPLETED'] = '0'

    # Assign a unique run ID for log filtering
    os.environ['GLOSSARION_RUN_ID'] = uuid.uuid4().hex[:10]

    # Create stop file for glossary workers
    import tempfile
    stop_file = os.path.join(tempfile.gettempdir(), f"glossarion_glossary_stop_{os.getpid()}.flag")
    os.environ['GLOSSARY_STOP_FILE'] = stop_file
    if os.path.exists(stop_file):
        try:
            os.remove(stop_file)
        except OSError:
            pass

    logger.info(f"Per-run env vars set (run_id={os.environ['GLOSSARION_RUN_ID']})")


def set_input_file_env(file_path, output_dir=None):
    """Set environment variables for the input file being translated.
    
    Args:
        file_path: Path to the EPUB or TXT file
        output_dir: Optional explicit output directory
    """
    os.environ['input_path'] = file_path

    if output_dir:
        os.environ['OUTPUT_DIRECTORY'] = os.path.abspath(output_dir)
    elif 'OUTPUT_DIRECTORY' in os.environ:
        del os.environ['OUTPUT_DIRECTORY']

    # Clear any previous manual glossary
    os.environ.pop('MANUAL_GLOSSARY', None)

    logger.info(f"Input file env set: {os.path.basename(file_path)}")
