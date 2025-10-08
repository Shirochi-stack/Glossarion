#!/usr/bin/env python3
"""
Environment Variable Debugging Helper
This script helps you debug environment variable initialization issues.
"""

import os
import sys
import json
from typing import Set

def _mask_value(name: str, value: str) -> str:
    """Mask secrets for display while preserving basic diagnostics."""
    if any(tok in name.upper() for tok in ["KEY", "TOKEN", "SECRET", "PASSWORD"]):
        return f"<REDACTED> ({len(str(value))} chars)"
    return str(value)


def print_env_var_status():
    """Print the current status of all critical environment variables."""
    print("🔍 Current Environment Variable Status:")
    print("=" * 60)
    
    # Critical environment variables that should always be set
    critical_env_vars = {
        # Glossary-related
        'GLOSSARY_SYSTEM_PROMPT': 'Manual glossary extraction prompt',
        'AUTO_GLOSSARY_PROMPT': 'Auto glossary generation prompt', 
        'GLOSSARY_CUSTOM_ENTRY_TYPES': 'Custom entry types configuration (JSON)',
        'GLOSSARY_DISABLE_HONORIFICS_FILTER': 'Honorifics filter disable flag',
        'GLOSSARY_STRIP_HONORIFICS': 'Strip honorifics flag',
        'GLOSSARY_FUZZY_THRESHOLD': 'Fuzzy matching threshold',
        'GLOSSARY_USE_LEGACY_CSV': 'Legacy CSV format flag',
        'GLOSSARY_CHAPTER_SPLIT_THRESHOLD': 'Chapter split threshold for large texts',
        'GLOSSARY_MAX_SENTENCES': 'Maximum sentences for glossary processing',
        
        # OpenRouter settings
        'OPENROUTER_USE_HTTP_ONLY': 'OpenRouter HTTP-only transport',
        'OPENROUTER_ACCEPT_IDENTITY': 'OpenRouter identity encoding',
        'OPENROUTER_PREFERRED_PROVIDER': 'OpenRouter preferred provider',
        
        # General application settings
        'EXTRACTION_WORKERS': 'Number of extraction worker threads',
        'ENABLE_GUI_YIELD': 'GUI yield during processing',
        'RETAIN_SOURCE_EXTENSION': 'Retain source file extension',
    }
    
    # Optional environment variables
    optional_env_vars = {
        'GLOSSARY_CUSTOM_FIELDS': 'Custom glossary fields (JSON)',
        'GLOSSARY_TRANSLATION_PROMPT': 'Glossary translation prompt',
        'GLOSSARY_FORMAT_INSTRUCTIONS': 'Glossary formatting instructions',
    }

    # Manga-related environment variables (Settings Dialog + Integration)
    manga_env_vars = {
        'MANGA_FULL_PAGE_CONTEXT': 'Enable full page context translation',
        'MANGA_VISUAL_CONTEXT_ENABLED': 'Include page image in requests',
        'MANGA_CREATE_SUBFOLDER': "Create 'translated' subfolder for output",
        'MANGA_BG_OPACITY': 'Background opacity (0-255)',
        'MANGA_BG_STYLE': 'Background style (box/circle/wrap)',
        'MANGA_BG_REDUCTION': 'Background reduction factor',
        'MANGA_FONT_SIZE': 'Fixed font size (0=auto)',
        'MANGA_FONT_STYLE': 'Font style name',
        'MANGA_FONT_PATH': 'Selected font path',
        'MANGA_FONT_SIZE_MODE': 'Font size mode (fixed/multiplier)',
        'MANGA_FONT_SIZE_MULTIPLIER': 'Font size multiplier (for multiplier mode)',
        'MANGA_MAX_FONT_SIZE': 'Maximum font size',
        'MANGA_AUTO_MIN_SIZE': 'Automatic minimum readable font size',
        'MANGA_FREE_TEXT_ONLY_BG_OPACITY': 'Apply BG opacity only to free text',
        'MANGA_FORCE_CAPS_LOCK': 'Force caps lock',
        'MANGA_STRICT_TEXT_WRAPPING': 'Strict text wrapping (force fit)',
        'MANGA_CONSTRAIN_TO_BUBBLE': 'Constrain text to bubble bounds',
        'MANGA_TEXT_COLOR': 'Text color RGB (R,G,B)',
        'MANGA_SHADOW_ENABLED': 'Shadow enabled',
        'MANGA_SHADOW_COLOR': 'Shadow color RGB (R,G,B)',
        'MANGA_SHADOW_OFFSET_X': 'Shadow offset X',
        'MANGA_SHADOW_OFFSET_Y': 'Shadow offset Y',
        'MANGA_SHADOW_BLUR': 'Shadow blur radius',
        'MANGA_INPAINT_SKIP': 'Skip inpainting',
        'MANGA_INPAINT_QUALITY': 'Inpainting quality preset',
        'MANGA_INPAINT_DILATION': 'Inpainting dilation (px)',
        'MANGA_INPAINT_PASSES': 'Inpainting passes',
        'MANGA_INPAINT_METHOD': 'Inpainting method (local/cloud/hybrid/skip)',
        'MANGA_LOCAL_INPAINT_METHOD': 'Local inpainting model type',
        'MANGA_FONT_ALGORITHM': 'Font sizing algorithm preset',
        'MANGA_PREFER_LARGER': 'Prefer larger font sizing',
        'MANGA_BUBBLE_SIZE_FACTOR': 'Use bubble size factor for sizing',
        'MANGA_LINE_SPACING': 'Line spacing multiplier',
        'MANGA_MAX_LINES': 'Maximum lines per bubble',
        'MANGA_QWEN2VL_MODEL_SIZE': 'Qwen2-VL model size selection',
        'MANGA_RAPIDOCR_USE_RECOGNITION': 'RapidOCR: use recognition step',
        'MANGA_RAPIDOCR_LANGUAGE': 'RapidOCR detection language',
        'MANGA_RAPIDOCR_DETECTION_MODE': 'RapidOCR detection mode',
        'MANGA_FULL_PAGE_CONTEXT_PROMPT_LEN': 'Length of full page context prompt',
        'MANGA_OCR_PROMPT_LEN': 'Length of OCR system prompt',
    }
    
    missing_critical = set()
    empty_critical = set()
    set_critical = set()
    
    print("\n📋 CRITICAL ENVIRONMENT VARIABLES:")
    print("-" * 40)
    
    all_critical = set(critical_env_vars.keys())
    for var_name, description in critical_env_vars.items():
        value = os.environ.get(var_name)
        
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ MISSING: {var_name}")
            print(f"   Description: {description}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  EMPTY: {var_name}")
            print(f"   Description: {description}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
    
    print(f"\n📋 OPTIONAL ENVIRONMENT VARIABLES:")
    print("-" * 40)
    
    for var_name, description in optional_env_vars.items():
        value = os.environ.get(var_name)
        if value is None:
            print(f"🔍 Not set: {var_name}")
        elif not value.strip():
            print(f"⚠️  Empty: {var_name}")
        else:
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"🔍 {var_name}: {value_preview}")
    
    known_vars: Set[str] = set()

    # Manga section
    print(f"\n🎎 MANGA ENVIRONMENT VARIABLES:")
    print("-" * 40)
    manga_set = 0
    for var_name, description in manga_env_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            manga_set += 1

    # Rolling summary variables
    rolling_vars = {
        'USE_ROLLING_SUMMARY': 'Enable rolling summary',
        'SUMMARY_ROLE': 'Summary role (user/system)',
        'ROLLING_SUMMARY_EXCHANGES': 'Max exchanges before roll',
        'ROLLING_SUMMARY_MODE': 'Rolling mode (append/replace)',
        'ROLLING_SUMMARY_SYSTEM_PROMPT': 'System prompt for rolling summary',
        'ROLLING_SUMMARY_USER_PROMPT': 'User prompt template for rolling summary',
        'ROLLING_SUMMARY_MAX_ENTRIES': 'Max retained summaries in append mode',
    }
    print(f"\n🧾 ROLLING SUMMARY VARIABLES:")
    print("-" * 40)
    rolling_set = 0
    for var_name, description in rolling_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            rolling_set += 1

    # Retry and network variables
    retry_net_vars = {
        'RETRY_TRUNCATED': 'Retry truncated responses',
        'MAX_RETRY_TOKENS': 'Max tokens for retry budget',
        'RETRY_DUPLICATE_BODIES': 'Retry duplicate outputs',
        'DUPLICATE_LOOKBACK_CHAPTERS': 'Duplicate lookback chapters',
        'RETRY_TIMEOUT': 'Enable retry on timeouts',
        'CHUNK_TIMEOUT': 'Per-chunk timeout (sec)',
        'ENABLE_HTTP_TUNING': 'Enable HTTP tuning',
        'CONNECT_TIMEOUT': 'Connect timeout (sec)',
        'READ_TIMEOUT': 'Read timeout (sec)',
        'HTTP_POOL_CONNECTIONS': 'HTTP pool connections',
        'HTTP_POOL_MAXSIZE': 'HTTP pool maxsize',
        'IGNORE_RETRY_AFTER': 'Ignore Retry-After header',
        'MAX_RETRIES': 'Max retries',
    }
    print(f"\n🌐 RETRY/NETWORK VARIABLES:")
    print("-" * 40)
    retry_set = 0
    for var_name, description in retry_net_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            retry_set += 1

    # QA/meta variables
    qa_meta_vars = {
'QA_AUTO_SEARCH_OUTPUT': 'QA auto-search output folder',
        'INDEFINITE_RATE_LIMIT_RETRY': 'Indefinite rate limit retries',
        'REINFORCEMENT_FREQUENCY': 'Prompt reinforcement frequency',
        'SCAN_PHASE_ENABLED': 'Enable post-translation scanning phase',
        'SCAN_PHASE_MODE': 'Scanning mode (quick-scan/aggressive/ai-hunter/custom)',
    }
    print(f"\n🧪 QA/META VARIABLES:")
    print("-" * 40)
    qa_set = 0
    for var_name, description in qa_meta_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            qa_set += 1

    # Book title variables
    book_vars = {
        'TRANSLATE_BOOK_TITLE': 'Translate book title',
        'BOOK_TITLE_PROMPT': 'Book title prompt',
    }
    print(f"\n📚 BOOK TITLE VARIABLES:")
    print("-" * 40)
    book_set = 0
    for var_name, description in book_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            book_set += 1

    # Image translation/EPUB variables
    image_vars = {
        'ENABLE_IMAGE_TRANSLATION': 'Enable image translation',
        'PROCESS_WEBNOVEL_IMAGES': 'Process images from HTML',
        'WEBNOVEL_MIN_HEIGHT': 'Minimum image height',
        'MAX_IMAGES_PER_CHAPTER': 'Max images per chapter',
        'IMAGE_CHUNK_HEIGHT': 'Image chunk height',
        'HIDE_IMAGE_TRANSLATION_LABEL': 'Hide image translation label',
        'DISABLE_EPUB_GALLERY': 'Disable EPUB gallery',
        'DISABLE_AUTOMATIC_COVER_CREATION': 'Disable automatic cover creation',
        'TRANSLATE_COVER_HTML': 'Translate cover.html',
        'DISABLE_ZERO_DETECTION': 'Disable zero detection',
        'DUPLICATE_DETECTION_MODE': 'Duplicate detection mode',
        'ENABLE_DECIMAL_CHAPTERS': 'Enable decimal chapter numbers',
    }
    print(f"\n🖼️ IMAGE/EPUB VARIABLES:")
    print("-" * 40)
    image_set = 0
    for var_name, description in image_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            image_set += 1

    # Watermark/image cleaning
    watermark_vars = {
        'ENABLE_WATERMARK_REMOVAL': 'Enable watermark removal',
        'SAVE_CLEANED_IMAGES': 'Save cleaned images',
    }
    print(f"\n💧 IMAGE CLEANING VARIABLES:")
    print("-" * 40)
    watermark_set = 0
    for var_name, description in watermark_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
            watermark_set += 1

    # Prompts and safety
    prompt_vars = {
        'TRANSLATION_CHUNK_PROMPT': 'Translation chunk prompt',
        'IMAGE_CHUNK_PROMPT': 'Image chunk prompt',
        'DISABLE_GEMINI_SAFETY': 'Disable Gemini safety checks',
    }
    print(f"\n📝 PROMPT/Safety VARIABLES:")
    print("-" * 40)
    prompt_set = 0
    for var_name, description in prompt_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            prompt_set += 1

    # Thinking features
    thinking_vars = {
        'ENABLE_GEMINI_THINKING': 'Enable Gemini thinking',
        'THINKING_BUDGET': 'Thinking token budget',
        'ENABLE_GPT_THINKING': 'Enable GPT thinking',
        'GPT_REASONING_TOKENS': 'Reasoning tokens for GPT',
        'GPT_EFFORT': 'GPT effort level',
    }
    print(f"\n🧠 THINKING VARIABLES:")
    print("-" * 40)
    thinking_set = 0
    for var_name, description in thinking_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            thinking_set += 1

    # API endpoints
    api_vars = {
        'OPENAI_CUSTOM_BASE_URL': 'Custom OpenAI base URL',
        'GROQ_API_URL': 'Groq API base URL',
        'FIREWORKS_API_URL': 'Fireworks API base URL',
        'USE_CUSTOM_OPENAI_ENDPOINT': 'Use custom OpenAI-compatible endpoint',
        'USE_GEMINI_OPENAI_ENDPOINT': 'Use Gemini-compatible OpenAI endpoint',
        'GEMINI_OPENAI_ENDPOINT': 'Gemini OpenAI endpoint URL',
    }
    print(f"\n🔌 API ENDPOINT VARIABLES:")
    print("-" * 40)
    api_set = 0
    for var_name, description in api_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            api_set += 1

    # Image compression
    compression_vars = {
        'ENABLE_IMAGE_COMPRESSION': 'Enable image compression',
        'AUTO_COMPRESS_ENABLED': 'Auto compression enabled',
        'TARGET_IMAGE_TOKENS': 'Target image tokens',
        'IMAGE_COMPRESSION_FORMAT': 'Image compression format',
        'WEBP_QUALITY': 'WEBP quality',
        'JPEG_QUALITY': 'JPEG quality',
        'PNG_COMPRESSION': 'PNG compression level',
        'MAX_IMAGE_DIMENSION': 'Max image dimension',
        'MAX_IMAGE_SIZE_MB': 'Max image size MB',
        'PRESERVE_TRANSPARENCY': 'Preserve transparency',
        'OPTIMIZE_FOR_OCR': 'Optimize for OCR',
        'PROGRESSIVE_ENCODING': 'Progressive encoding',
        'SAVE_COMPRESSED_IMAGES': 'Save compressed images',
        'USE_FALLBACK_KEYS': 'Use fallback API keys',
        'FALLBACK_KEYS': 'Fallback key list (JSON)',
        'IMAGE_CHUNK_OVERLAP_PERCENT': 'Image chunk overlap percent',
    }
    print(f"\n🗜️ IMAGE COMPRESSION VARIABLES:")
    print("-" * 40)
    comp_set = 0
    for var_name, description in compression_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            comp_set += 1

    # Metadata and headers
    metadata_vars = {
        'TRANSLATE_METADATA_FIELDS': 'Translate metadata fields (JSON)',
        'METADATA_TRANSLATION_MODE': 'Metadata translation mode',
        'BATCH_TRANSLATE_HEADERS': 'Batch translate headers',
        'HEADERS_PER_BATCH': 'Headers per batch',
        'UPDATE_HTML_HEADERS': 'Update HTML headers',
        'SAVE_HEADER_TRANSLATIONS': 'Save header translations',
        'IGNORE_HEADER': 'Ignore header',
        'IGNORE_TITLE': 'Ignore title',
    }
    print(f"\n🏷️ METADATA/HEADERS VARIABLES:")
    print("-" * 40)
    meta_set = 0
    for var_name, description in metadata_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            meta_set += 1

    # Extraction mode and anti-duplicate
    extraction_vars = {
        'TEXT_EXTRACTION_METHOD': 'Text extraction method',
        'FILE_FILTERING_LEVEL': 'File filtering level',
        'EXTRACTION_MODE': 'Extraction mode',
        'ENHANCED_FILTERING': 'Enhanced filtering preset',
        'ENABLE_ANTI_DUPLICATE': 'Enable anti-duplicate',
        'TOP_P': 'Top-p sampling',
        'TOP_K': 'Top-k sampling',
        'FREQUENCY_PENALTY': 'Frequency penalty',
        'PRESENCE_PENALTY': 'Presence penalty',
        'REPETITION_PENALTY': 'Repetition penalty',
        'CANDIDATE_COUNT': 'Candidate count',
        'CUSTOM_STOP_SEQUENCES': 'Custom stop sequences',
        'LOGIT_BIAS_ENABLED': 'Logit bias enabled',
        'LOGIT_BIAS_STRENGTH': 'Logit bias strength',
        'BIAS_COMMON_WORDS': 'Bias common words',
        'BIAS_REPETITIVE_PHRASES': 'Bias repetitive phrases',
    }
    print(f"\n🧩 EXTRACTION/ANTI-DUPLICATE VARIABLES:")
    print("-" * 40)
    extract_set = 0
    for var_name, description in extraction_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            extract_set += 1

    # Azure API version
    azure_vars = {
        'AZURE_API_VERSION': 'Azure API version',
    }
    print(f"\n☁️ AZURE VARIABLES:")
    print("-" * 40)
    azure_set = 0
    for var_name, description in azure_vars.items():
        known_vars.add(var_name)
        value = os.environ.get(var_name)
        all_critical.add(var_name)
        if value is None:
            missing_critical.add(var_name)
            print(f"❌ Not set: {var_name}")
        elif not str(value).strip():
            empty_critical.add(var_name)
            print(f"⚠️  Empty: {var_name}")
        else:
            set_critical.add(var_name)
            value_preview = (str(value)[:80] + ('...' if len(str(value)) > 80 else ''))
            print(f"✅ {var_name}: {value_preview}")
            azure_set += 1
    
    # Summary
    total_critical = len(all_critical)
    print(f"\n📊 SUMMARY:")
    print("-" * 20)
    print(f"Critical variables set: {len(set_critical)}/{total_critical}")
    print(f"Manga variables set: {manga_set}/{len(manga_env_vars)}")
    print(f"Rolling summary set: {rolling_set}/{len(rolling_vars)}")
    print(f"Retry/network set: {retry_set}/{len(retry_net_vars)}")
    print(f"QA/meta set: {qa_set}/{len(qa_meta_vars)}")
    print(f"Book title set: {book_set}/{len(book_vars)}")
    print(f"Image/EPUB set: {image_set}/{len(image_vars)}")
    print(f"Cleaning set: {watermark_set}/{len(watermark_vars)}")
    print(f"Prompts set: {prompt_set}/{len(prompt_vars)}")
    print(f"Thinking set: {thinking_set}/{len(thinking_vars)}")
    print(f"API endpoints set: {api_set}/{len(api_vars)}")
    print(f"Compression set: {comp_set}/{len(compression_vars)}")
    print(f"Metadata set: {meta_set}/{len(metadata_vars)}")
    print(f"Extraction/Anti-dup set: {extract_set}/{len(extraction_vars)}")
    print(f"Azure set: {azure_set}/{len(azure_vars)}")

    # Catch-all app variables by prefix (not yet listed)
    prefixes = [
        'GLOSSARY_', 'MANGA_', 'QA_', 'SCAN_PHASE_', 'TRANSLATE_', 'TRANSLATION_', 'BOOK_',
        'OPENROUTER_', 'ENABLE_', 'DISABLE_', 'EXTRACTION_', 'METADATA_', 'BATCH_', 'HEADERS_',
        'HTTP_', 'CONNECT_TIMEOUT', 'READ_TIMEOUT', 'MAX_RETRIES', 'IGNORE_RETRY_AFTER',
        'TEXT_EXTRACTION_', 'FILE_FILTERING_', 'ENHANCED_FILTERING', 'AZURE_', 'OPENAI_',
        'GROQ_', 'FIREWORKS_', 'USE_CUSTOM_', 'USE_GEMINI_', 'GEMINI_', 'IMAGE_', 'WEBP_',
        'JPEG_', 'PNG_', 'TARGET_IMAGE_TOKENS', 'MAX_IMAGE_', 'PRESERVE_', 'OPTIMIZE_',
        'PROGRESSIVE_', 'SAVE_COMPRESSED_IMAGES', 'USE_FALLBACK_KEYS', 'FALLBACK_KEYS',
        'TOP_P', 'TOP_K', 'FREQUENCY_PENALTY', 'PRESENCE_PENALTY', 'REPETITION_PENALTY',
        'CANDIDATE_COUNT', 'CUSTOM_STOP_SEQUENCES', 'LOGIT_BIAS_', 'BIAS_', 'THINKING_', 'GPT_',
        'AI_HUNTER_', 'SYSTEM_PROMPT', 'MODEL', 'RETAIN_SOURCE_EXTENSION', 'auto_update_check',
        'FORCE_NCX_ONLY', 'SINGLE_API_IMAGE_CHUNKS'
    ]
    print(f"\n🧾 OTHER APP VARIABLES:")
    print("-" * 40)
    other_set = 0
    for k, v in sorted(os.environ.items()):
        if k in known_vars:
            continue
        if any(k.startswith(pfx) for pfx in prefixes):
            masked = _mask_value(k, v)
            all_critical.add(k)
            if v is None or not str(v).strip():
                if v is None:
                    missing_critical.add(k)
                else:
                    empty_critical.add(k)
                print(f"❌ {k}: not set/empty")
            else:
                set_critical.add(k)
                print(f"✅ {k}: {masked[:80]}{'...' if len(masked) > 80 else ''}")
            other_set += 1
    print(f"Other app vars listed: {other_set}")
    
    if missing_critical:
        print(f"❌ Missing ({len(missing_critical)}): {', '.join(sorted(missing_critical))}")
        
    if empty_critical:
        print(f"⚠️  Empty ({len(empty_critical)}): {', '.join(sorted(empty_critical))}")
    
    if not missing_critical and not empty_critical:
        print("✅ All critical environment variables are properly set!")
        return True
    else:
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Run the Glossary Manager and save settings")
        print("2. Check that all GUI variables are properly initialized")
        print("3. Call self.initialize_environment_variables() on app startup")
        print("4. Call self.debug_environment_variables() to see detailed debugging")
        return False

def test_json_env_vars():
    """Test JSON environment variables for validity."""
    print("\n🧪 JSON Environment Variable Validation:")
    print("=" * 50)
    
    json_vars = ['GLOSSARY_CUSTOM_ENTRY_TYPES', 'GLOSSARY_CUSTOM_FIELDS']
    
    for var_name in json_vars:
        value = os.environ.get(var_name)
        
        if not value:
            print(f"🔍 {var_name}: Not set")
            continue
            
        try:
            parsed_json = json.loads(value)
            print(f"✅ {var_name}: Valid JSON ({len(value)} chars)")
            print(f"   Content type: {type(parsed_json).__name__}")
            if isinstance(parsed_json, dict):
                print(f"   Keys: {list(parsed_json.keys())}")
            elif isinstance(parsed_json, list):
                print(f"   Items: {len(parsed_json)}")
        except json.JSONDecodeError as e:
            print(f"❌ {var_name}: Invalid JSON")
            print(f"   Error: {e}")
            print(f"   Value preview: {value[:100]}...")

if __name__ == "__main__":
    print("🚀 Environment Variable Debugging Tool")
    print("This tool helps debug Glossarion environment variable issues")
    
    # Check current status
    success = print_env_var_status()
    
    # Test JSON variables
    test_json_env_vars()
    
    if not success:
        print(f"\n💡 TIP: Add these methods to your TranslatorGUI class:")
        print("  • self.initialize_environment_variables() - Call on startup")
        print("  • self.debug_environment_variables() - Call for debugging")
        
    print("\n" + "=" * 60)
    print("🎯 To use the new debugging features in your app:")
    print("   1. self.initialize_environment_variables()  # On startup")
    print("   2. self.debug_environment_variables()       # For debugging")  
    print("   3. Enhanced save_glossary_settings() now has full debugging")
    print("   4. Enhanced save_config() now has full debugging")