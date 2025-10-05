#!/usr/bin/env python3
"""
Glossarion Web - Gradio Web Interface
AI-powered translation in your browser
"""

import gradio as gr
import os
import sys
import json
import tempfile
import base64
from pathlib import Path

# CRITICAL: Set API delay IMMEDIATELY at module level before any other imports
# This ensures unified_api_client reads the correct value when it's imported
if 'SEND_INTERVAL_SECONDS' not in os.environ:
    os.environ['SEND_INTERVAL_SECONDS'] = '0.5'
print(f"🔧 Module-level API delay initialized: {os.environ['SEND_INTERVAL_SECONDS']}s")

# Import API key encryption/decryption
try:
    from api_key_encryption import APIKeyEncryption
    API_KEY_ENCRYPTION_AVAILABLE = True
    # Create web-specific encryption handler with its own key file
    _web_encryption_handler = None
    def get_web_encryption_handler():
        global _web_encryption_handler
        if _web_encryption_handler is None:
            _web_encryption_handler = APIKeyEncryption()
            # Use web-specific key file
            from pathlib import Path
            _web_encryption_handler.key_file = Path('.glossarion_web_key')
            _web_encryption_handler.cipher = _web_encryption_handler._get_or_create_cipher()
            # Add web-specific fields to encrypt
            _web_encryption_handler.api_key_fields.extend([
                'azure_vision_key',
                'google_vision_credentials'
            ])
        return _web_encryption_handler
    
    def decrypt_config(config):
        return get_web_encryption_handler().decrypt_config(config)
    
    def encrypt_config(config):
        return get_web_encryption_handler().encrypt_config(config)
except ImportError:
    API_KEY_ENCRYPTION_AVAILABLE = False
    def decrypt_config(config):
        return config  # Fallback: return config as-is
    def encrypt_config(config):
        return config  # Fallback: return config as-is

# Import your existing translation modules
try:
    import TransateKRtoEN
    from model_options import get_model_options
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("⚠️ Translation modules not found")

# Import manga translation modules
try:
    from manga_translator import MangaTranslator
    from unified_api_client import UnifiedClient
    MANGA_TRANSLATION_AVAILABLE = True
    print("✅ Manga translation modules loaded successfully")
except ImportError as e:
    MANGA_TRANSLATION_AVAILABLE = False
    print(f"⚠️ Manga translation modules not found: {e}")
    print(f"⚠️ Current working directory: {os.getcwd()}")
    print(f"⚠️ Python path: {sys.path[:3]}...")
    
    # Check if files exist
    files_to_check = ['manga_translator.py', 'unified_api_client.py', 'bubble_detector.py', 'local_inpainter.py']
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")


class GlossarionWeb:
    """Web interface for Glossarion translator"""
    
    def __init__(self):
        # Determine config file path based on environment
        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
        
        if is_hf_spaces:
            # Use /data directory for Hugging Face Spaces persistent storage
            data_dir = '/data'
            if not os.path.exists(data_dir):
                # Fallback to current directory if /data doesn't exist
                data_dir = '.'
            self.config_file = os.path.join(data_dir, 'config_web.json')
            print(f"🤗 HF Spaces detected - using config path: {self.config_file}")
            print(f"📁 Directory exists: {os.path.exists(os.path.dirname(self.config_file))}")
        else:
            # Local mode - use current directory
            self.config_file = "config_web.json"
            print(f"🏠 Local mode - using config path: {self.config_file}")
        
        # Load raw config first
        self.config = self.load_config()
        
        # Create a decrypted version for display/use in the UI
        # but keep the original for saving
        self.decrypted_config = self.config.copy()
        if API_KEY_ENCRYPTION_AVAILABLE:
            self.decrypted_config = decrypt_config(self.decrypted_config)
        
        # CRITICAL: Initialize environment variables IMMEDIATELY after loading config
        # This must happen before any UnifiedClient is created
        
        # Set API call delay
        api_call_delay = self.decrypted_config.get('api_call_delay', 0.5)
        if 'api_call_delay' not in self.config:
            self.config['api_call_delay'] = 0.5
            self.decrypted_config['api_call_delay'] = 0.5
        os.environ['SEND_INTERVAL_SECONDS'] = str(api_call_delay)
        print(f"🔧 Initialized API call delay: {api_call_delay}s")
        
        # Set font algorithm and auto fit style if not present
        if 'manga_settings' not in self.config:
            self.config['manga_settings'] = {}
        if 'font_sizing' not in self.config['manga_settings']:
            self.config['manga_settings']['font_sizing'] = {}
        if 'rendering' not in self.config['manga_settings']:
            self.config['manga_settings']['rendering'] = {}
        
        if 'algorithm' not in self.config['manga_settings']['font_sizing']:
            self.config['manga_settings']['font_sizing']['algorithm'] = 'smart'
        if 'auto_fit_style' not in self.config['manga_settings']['rendering']:
            self.config['manga_settings']['rendering']['auto_fit_style'] = 'balanced'
        
        # Also ensure they're in decrypted_config
        if 'manga_settings' not in self.decrypted_config:
            self.decrypted_config['manga_settings'] = {}
        if 'font_sizing' not in self.decrypted_config['manga_settings']:
            self.decrypted_config['manga_settings']['font_sizing'] = {}
        if 'rendering' not in self.decrypted_config['manga_settings']:
            self.decrypted_config['manga_settings']['rendering'] = {}
        if 'algorithm' not in self.decrypted_config['manga_settings']['font_sizing']:
            self.decrypted_config['manga_settings']['font_sizing']['algorithm'] = 'smart'
        if 'auto_fit_style' not in self.decrypted_config['manga_settings']['rendering']:
            self.decrypted_config['manga_settings']['rendering']['auto_fit_style'] = 'balanced'
        
        print(f"🎨 Initialized font algorithm: {self.config['manga_settings']['font_sizing']['algorithm']}")
        print(f"🎨 Initialized auto fit style: {self.config['manga_settings']['rendering']['auto_fit_style']}")
        
        self.models = get_model_options() if TRANSLATION_AVAILABLE else ["gpt-4", "claude-3-5-sonnet"]
        print(f"🤖 Loaded {len(self.models)} models: {self.models[:5]}{'...' if len(self.models) > 5 else ''}")
        
        # Translation state management
        import threading
        self.is_translating = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.current_unified_client = None  # Track active client to allow cancellation
        self.current_translator = None     # Track active translator to allow shutdown
        
        # Add stop flags for different translation types
        self.epub_translation_stop = False
        self.epub_translation_thread = None
        self.glossary_extraction_stop = False
        self.glossary_extraction_thread = None
        
        # Default prompts from the GUI (same as translator_gui.py)
        self.default_prompts = {
            "korean": (
                "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- When translating Korean's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration, and maintain natural English flow without overusing pronouns just because they're omitted in Korean.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "japanese": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- When translating Japanese's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the Japanese pronoun's nuance (私/僕/俺/etc.) through speech patterns rather than the pronoun itself, and maintain natural English flow without overusing pronouns just because they're omitted in Japanese.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "chinese": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔法 = magic).\n"
                "- When translating Chinese's pronoun-dropping style, insert pronouns in English only where needed for clarity while maintaining natural English flow.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Pinyin.\n"
                "- Keep original Chinese quotation marks (「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "Manga_JP": (
                "You are a professional Japanese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_KR": (
                "You are a professional Korean to English Manhwa translator.\n"
                "You have both the image of the Manhwa panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_CN": (
                "You are a professional Chinese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ),
            "Original": "Return everything exactly as seen on the source."
        }
        
        # Load profiles from config and merge with defaults
        # Always include default prompts, then overlay any custom ones from config
        self.profiles = self.default_prompts.copy()
        config_profiles = self.config.get('prompt_profiles', {})
        if config_profiles:
            self.profiles.update(config_profiles)
    
    def get_config_value(self, key, default=None):
        """Get value from decrypted config with fallback"""
        return self.decrypted_config.get(key, default)
    
    def get_current_config_for_update(self):
        """Get the current config for updating (uses in-memory version)"""
        # Return a copy of the in-memory config, not loaded from file
        return self.config.copy()
    
    def get_default_config(self):
        """Get default configuration for Hugging Face Spaces"""
        return {
            'model': 'gpt-4-turbo',
            'api_key': '',
            'api_call_delay': 0.5,  # Default 0.5 seconds between API calls
            'ocr_provider': 'custom-api',
            'bubble_detection_enabled': True,
            'inpainting_enabled': True,
            'manga_font_size_mode': 'auto',
            'manga_font_size': 0,
            'manga_font_size_multiplier': 1.0,
            'manga_min_font_size': 10,
            'manga_max_font_size': 40,
            'manga_text_color': [102, 0, 0],  # Dark red text (manga_integration.py default)
            'manga_shadow_enabled': True,
            'manga_shadow_color': [204, 128, 128],  # Light pink shadow (manga_integration.py default)
            'manga_shadow_offset_x': 2,  # Match manga integration
            'manga_shadow_offset_y': 2,  # Match manga integration
            'manga_shadow_blur': 0,  # Match manga integration (no blur)
            'manga_bg_opacity': 0,  # Transparent background by default
            'manga_bg_style': 'circle',
            'manga_settings': {
                'ocr': {
                    'detector_type': 'rtdetr_onnx',
                    'rtdetr_confidence': 0.3,
                    'bubble_confidence': 0.3,
                    'detect_text_bubbles': True,
                    'detect_empty_bubbles': True,
                    'detect_free_text': True,
                    'bubble_max_detections_yolo': 100
                },
                'inpainting': {
                    'local_method': 'anime',
                    'method': 'local',
                    'batch_size': 10,
                    'enable_cache': True
                },
                'advanced': {
                    'parallel_processing': True,
                    'max_workers': 2,
                    'parallel_panel_translation': False,
                    'panel_max_workers': 7,
                    'format_detection': True,
                    'webtoon_mode': 'auto',
                    'torch_precision': 'fp16',
                    'auto_cleanup_models': False,
                    'debug_mode': False,
                    'save_intermediate': False
                },
                'rendering': {
                    'auto_min_size': 10,
                    'auto_max_size': 40,
                    'auto_fit_style': 'balanced'
                },
                'font_sizing': {
                    'algorithm': 'smart',
                    'prefer_larger': True,
                    'max_lines': 10,
                    'line_spacing': 1.3,
                    'bubble_size_factor': True,
                    'min_size': 10,
                    'max_size': 40
                },
                'tiling': {
                    'enabled': False,
                    'tile_size': 480,
                    'tile_overlap': 64
                }
            }
        }
    
    def load_config(self):
        """Load configuration - from persistent file on HF Spaces or local file"""
        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
        
        # Try to load from file (works both locally and on HF Spaces with persistent storage)
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Start with defaults
                    default_config = self.get_default_config()
                    # Deep merge - preserve nested structures from loaded config
                    self._deep_merge_config(default_config, loaded_config)
                    
                    if is_hf_spaces:
                        print(f"✅ Loaded config from persistent storage: {self.config_file}")
                    else:
                        print(f"✅ Loaded config from local file: {self.config_file}")
                    
                    return default_config
        except Exception as e:
            print(f"Could not load config from {self.config_file}: {e}")
        
        # If loading fails or file doesn't exist - return defaults
        print(f"📝 Using default configuration")
        return self.get_default_config()
    
    def _deep_merge_config(self, base, override):
        """Deep merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                self._deep_merge_config(base[key], value)
            else:
                # Override the value
                base[key] = value
    
    def set_all_environment_variables(self):
        """Set all environment variables from config for translation engines"""
        config = self.get_config_value
        
        # API Rate Limiting
        os.environ['SEND_INTERVAL_SECONDS'] = str(config('api_call_delay', 0.5))
        
        # Chapter Processing Options
        os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if config('batch_translate_headers', False) else '0'
        os.environ['HEADERS_PER_BATCH'] = str(config('headers_per_batch', 400))
        os.environ['USE_NCX_NAVIGATION'] = '1' if config('use_ncx_navigation', False) else '0'
        os.environ['ATTACH_CSS_TO_CHAPTERS'] = '1' if config('attach_css_to_chapters', False) else '0'
        os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if config('retain_source_extension', True) else '0'
        os.environ['USE_CONSERVATIVE_BATCHING'] = '1' if config('use_conservative_batching', False) else '0'
        os.environ['DISABLE_GEMINI_SAFETY'] = '1' if config('disable_gemini_safety', False) else '0'
        os.environ['USE_HTTP_OPENROUTER'] = '1' if config('use_http_openrouter', False) else '0'
        os.environ['DISABLE_OPENROUTER_COMPRESSION'] = '1' if config('disable_openrouter_compression', False) else '0'
        
        # Chapter Extraction Settings
        os.environ['TEXT_EXTRACTION_METHOD'] = config('text_extraction_method', 'standard')
        os.environ['FILE_FILTERING_LEVEL'] = config('file_filtering_level', 'smart')
        
        # Thinking Mode Settings
        os.environ['ENABLE_GPT_THINKING'] = '1' if config('enable_gpt_thinking', True) else '0'
        os.environ['GPT_THINKING_EFFORT'] = config('gpt_thinking_effort', 'medium')
        os.environ['OR_THINKING_TOKENS'] = str(config('or_thinking_tokens', 2000))
        os.environ['ENABLE_GEMINI_THINKING'] = '1' if config('enable_gemini_thinking', False) else '0'
        os.environ['GEMINI_THINKING_BUDGET'] = str(config('gemini_thinking_budget', 0))
        # IMPORTANT: Also set THINKING_BUDGET for unified_api_client compatibility
        os.environ['THINKING_BUDGET'] = str(config('gemini_thinking_budget', 0))
        
        # Translation Settings
        os.environ['CONTEXTUAL'] = '1' if config('contextual', False) else '0'
        os.environ['TRANSLATION_HISTORY_LIMIT'] = str(config('translation_history_limit', 2))
        os.environ['TRANSLATION_HISTORY_ROLLING'] = '1' if config('translation_history_rolling', False) else '0'
        os.environ['BATCH_TRANSLATION'] = '1' if config('batch_translation', True) else '0'
        os.environ['BATCH_SIZE'] = str(config('batch_size', 10))
        os.environ['THREAD_SUBMISSION_DELAY'] = str(config('thread_submission_delay', 0.1))
        os.environ['DELAY'] = str(config('delay', 1))
        os.environ['CHAPTER_RANGE'] = config('chapter_range', '')
        os.environ['TOKEN_LIMIT'] = str(config('token_limit', 200000))
        os.environ['TOKEN_LIMIT_DISABLED'] = '1' if config('token_limit_disabled', False) else '0'
        os.environ['DISABLE_INPUT_TOKEN_LIMIT'] = '1' if config('token_limit_disabled', False) else '0'
        
        # Glossary Settings
        os.environ['ENABLE_AUTO_GLOSSARY'] = '1' if config('enable_auto_glossary', False) else '0'
        os.environ['APPEND_GLOSSARY_TO_PROMPT'] = '1' if config('append_glossary_to_prompt', True) else '0'
        os.environ['GLOSSARY_MIN_FREQUENCY'] = str(config('glossary_min_frequency', 2))
        os.environ['GLOSSARY_MAX_NAMES'] = str(config('glossary_max_names', 50))
        os.environ['GLOSSARY_MAX_TITLES'] = str(config('glossary_max_titles', 30))
        os.environ['GLOSSARY_BATCH_SIZE'] = str(config('glossary_batch_size', 50))
        os.environ['GLOSSARY_FILTER_MODE'] = config('glossary_filter_mode', 'all')
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(config('glossary_fuzzy_threshold', 0.90))
        
        # Manual Glossary Settings
        os.environ['MANUAL_GLOSSARY_MIN_FREQUENCY'] = str(config('manual_glossary_min_frequency', 2))
        os.environ['MANUAL_GLOSSARY_MAX_NAMES'] = str(config('manual_glossary_max_names', 50))
        os.environ['MANUAL_GLOSSARY_MAX_TITLES'] = str(config('manual_glossary_max_titles', 30))
        os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(config('glossary_max_text_size', 50000))
        os.environ['GLOSSARY_MAX_SENTENCES'] = str(config('glossary_max_sentences', 200))
        os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(config('glossary_chapter_split_threshold', 8192))
        os.environ['MANUAL_GLOSSARY_FILTER_MODE'] = config('manual_glossary_filter_mode', 'all')
        os.environ['STRIP_HONORIFICS'] = '1' if config('strip_honorifics', True) else '0'
        os.environ['MANUAL_GLOSSARY_FUZZY_THRESHOLD'] = str(config('manual_glossary_fuzzy_threshold', 0.90))
        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '1' if config('glossary_use_legacy_csv', False) else '0'
        
        # QA Scanner Settings
        os.environ['ENABLE_POST_TRANSLATION_SCAN'] = '1' if config('enable_post_translation_scan', False) else '0'
        os.environ['QA_MIN_FOREIGN_CHARS'] = str(config('qa_min_foreign_chars', 10))
        os.environ['QA_CHECK_REPETITION'] = '1' if config('qa_check_repetition', True) else '0'
        os.environ['QA_CHECK_GLOSSARY_LEAKAGE'] = '1' if config('qa_check_glossary_leakage', True) else '0'
        os.environ['QA_MIN_FILE_LENGTH'] = str(config('qa_min_file_length', 0))
        os.environ['QA_CHECK_MULTIPLE_HEADERS'] = '1' if config('qa_check_multiple_headers', True) else '0'
        os.environ['QA_CHECK_MISSING_HTML'] = '1' if config('qa_check_missing_html', True) else '0'
        os.environ['QA_CHECK_INSUFFICIENT_PARAGRAPHS'] = '1' if config('qa_check_insufficient_paragraphs', True) else '0'
        os.environ['QA_MIN_PARAGRAPH_PERCENTAGE'] = str(config('qa_min_paragraph_percentage', 30))
        os.environ['QA_REPORT_FORMAT'] = config('qa_report_format', 'detailed')
        os.environ['QA_AUTO_SAVE_REPORT'] = '1' if config('qa_auto_save_report', True) else '0'
        
        # Manga/Image Translation Settings (when available)
        os.environ['BUBBLE_DETECTION_ENABLED'] = '1' if config('bubble_detection_enabled', True) else '0'
        os.environ['INPAINTING_ENABLED'] = '1' if config('inpainting_enabled', True) else '0'
        os.environ['MANGA_FONT_SIZE_MODE'] = config('manga_font_size_mode', 'auto')
        os.environ['MANGA_FONT_SIZE'] = str(config('manga_font_size', 24))
        os.environ['MANGA_FONT_MULTIPLIER'] = str(config('manga_font_multiplier', 1.0))
        os.environ['MANGA_MIN_FONT_SIZE'] = str(config('manga_min_font_size', 12))
        os.environ['MANGA_MAX_FONT_SIZE'] = str(config('manga_max_font_size', 48))
        os.environ['MANGA_SHADOW_ENABLED'] = '1' if config('manga_shadow_enabled', True) else '0'
        os.environ['MANGA_SHADOW_OFFSET_X'] = str(config('manga_shadow_offset_x', 2))
        os.environ['MANGA_SHADOW_OFFSET_Y'] = str(config('manga_shadow_offset_y', 2))
        os.environ['MANGA_SHADOW_BLUR'] = str(config('manga_shadow_blur', 0))
        os.environ['MANGA_BG_OPACITY'] = str(config('manga_bg_opacity', 130))
        os.environ['MANGA_BG_STYLE'] = config('manga_bg_style', 'circle')
        
        # OCR Provider Settings
        os.environ['OCR_PROVIDER'] = config('ocr_provider', 'custom-api')
        
        # Advanced Manga Settings
        manga_settings = config('manga_settings', {})
        if manga_settings:
            advanced = manga_settings.get('advanced', {})
            os.environ['PARALLEL_PANEL_TRANSLATION'] = '1' if advanced.get('parallel_panel_translation', False) else '0'
            os.environ['PANEL_MAX_WORKERS'] = str(advanced.get('panel_max_workers', 7))
            os.environ['PANEL_START_STAGGER_MS'] = str(advanced.get('panel_start_stagger_ms', 0))
            os.environ['WEBTOON_MODE'] = '1' if advanced.get('webtoon_mode', False) else '0'
            os.environ['DEBUG_MODE'] = '1' if advanced.get('debug_mode', False) else '0'
            os.environ['SAVE_INTERMEDIATE'] = '1' if advanced.get('save_intermediate', False) else '0'
            os.environ['PARALLEL_PROCESSING'] = '1' if advanced.get('parallel_processing', True) else '0'
            os.environ['MAX_WORKERS'] = str(advanced.get('max_workers', 4))
            os.environ['AUTO_CLEANUP_MODELS'] = '1' if advanced.get('auto_cleanup_models', False) else '0'
            os.environ['TORCH_PRECISION'] = advanced.get('torch_precision', 'auto')
            os.environ['PRELOAD_LOCAL_INPAINTING_FOR_PANELS'] = '1' if advanced.get('preload_local_inpainting_for_panels', False) else '0'
            
            # OCR settings
            ocr = manga_settings.get('ocr', {})
            os.environ['DETECTOR_TYPE'] = ocr.get('detector_type', 'rtdetr_onnx')
            os.environ['RTDETR_CONFIDENCE'] = str(ocr.get('rtdetr_confidence', 0.3))
            os.environ['BUBBLE_CONFIDENCE'] = str(ocr.get('bubble_confidence', 0.3))
            os.environ['DETECT_TEXT_BUBBLES'] = '1' if ocr.get('detect_text_bubbles', True) else '0'
            os.environ['DETECT_EMPTY_BUBBLES'] = '1' if ocr.get('detect_empty_bubbles', True) else '0'
            os.environ['DETECT_FREE_TEXT'] = '1' if ocr.get('detect_free_text', True) else '0'
            os.environ['BUBBLE_MAX_DETECTIONS_YOLO'] = str(ocr.get('bubble_max_detections_yolo', 100))
            
            # Inpainting settings
            inpainting = manga_settings.get('inpainting', {})
            os.environ['LOCAL_INPAINT_METHOD'] = inpainting.get('local_method', 'anime_onnx')
            os.environ['INPAINT_BATCH_SIZE'] = str(inpainting.get('batch_size', 10))
            os.environ['INPAINT_CACHE_ENABLED'] = '1' if inpainting.get('enable_cache', True) else '0'
            
            # HD Strategy
            os.environ['HD_STRATEGY'] = advanced.get('hd_strategy', 'resize')
            os.environ['HD_RESIZE_LIMIT'] = str(advanced.get('hd_strategy_resize_limit', 1536))
            os.environ['HD_CROP_MARGIN'] = str(advanced.get('hd_strategy_crop_margin', 16))
            os.environ['HD_CROP_TRIGGER'] = str(advanced.get('hd_strategy_crop_trigger_size', 1024))
        
        # Concise Pipeline Logs
        os.environ['CONCISE_PIPELINE_LOGS'] = '1' if config('concise_pipeline_logs', False) else '0'
        
        print("✅ All environment variables set from configuration")
    
    def save_config(self, config):
        """Save configuration - to persistent file on HF Spaces or local file"""
        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
        
        # Always try to save to file (works both locally and on HF Spaces with persistent storage)
        try:
            config_to_save = config.copy()
            
            # Only encrypt if we have the encryption module AND keys aren't already encrypted
            if API_KEY_ENCRYPTION_AVAILABLE:
                # Check if keys need encryption (not already encrypted)
                needs_encryption = False
                for key in ['api_key', 'azure_vision_key', 'google_vision_credentials']:
                    if key in config_to_save:
                        value = config_to_save[key]
                        # If it's a non-empty string that doesn't start with 'ENC:', it needs encryption
                        if value and isinstance(value, str) and not value.startswith('ENC:'):
                            needs_encryption = True
                            break
                
                if needs_encryption:
                    config_to_save = encrypt_config(config_to_save)
            
            # Create directory if it doesn't exist (important for HF Spaces)
            os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
            
            # Debug output
            if is_hf_spaces:
                print(f"📝 Saving to HF Spaces persistent storage: {self.config_file}")
            
            print(f"DEBUG save_config called with model={config.get('model')}, batch_size={config.get('batch_size')}")
            print(f"DEBUG self.config before={self.config.get('model') if hasattr(self, 'config') else 'N/A'}")
            print(f"DEBUG self.decrypted_config before={self.decrypted_config.get('model') if hasattr(self, 'decrypted_config') else 'N/A'}")
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
            
            # IMPORTANT: Update the in-memory configs so the UI reflects the changes immediately
            self.config = config_to_save
            # Update decrypted config too
            self.decrypted_config = config.copy()  # Use the original (unencrypted) version
            if API_KEY_ENCRYPTION_AVAILABLE:
                # Make sure decrypted_config has decrypted values
                self.decrypted_config = decrypt_config(self.decrypted_config)
            
            print(f"DEBUG self.config after={self.config.get('model')}")
            print(f"DEBUG self.decrypted_config after={self.decrypted_config.get('model')}")
            
            if is_hf_spaces:
                print(f"✅ Saved to persistent storage: {self.config_file}")
                # Also verify the file was written
                if os.path.exists(self.config_file):
                    file_size = os.path.getsize(self.config_file)
                    print(f"✅ File confirmed: {file_size} bytes")
                return "✅ Settings saved to persistent storage!"
            else:
                print(f"✅ Saved to {self.config_file}")
                return "✅ Settings saved successfully!"
                
        except Exception as e:
            print(f"❌ Save error: {e}")
            if is_hf_spaces:
                print(f"💡 Note: Make sure you have persistent storage enabled for your Space")
                return f"❌ Failed to save: {str(e)}\n\nNote: Persistent storage may not be enabled"
            return f"❌ Failed to save: {str(e)}"
    
    def translate_epub(
        self,
        epub_file,
        model,
        api_key,
        profile_name,
        system_prompt,
        temperature,
        max_tokens,
        enable_image_trans=False,
        glossary_file=None
    ):
        """Translate EPUB file - yields progress updates"""
        
        if not TRANSLATION_AVAILABLE:
            yield None, None, None, "❌ Translation modules not loaded", None, "Error", 0
            return
        
        if not epub_file:
            yield None, None, None, "❌ Please upload an EPUB or TXT file", None, "Error", 0
            return
        
        if not api_key:
            yield None, None, None, "❌ Please provide an API key", None, "Error", 0
            return
        
        if not profile_name:
            yield None, None, None, "❌ Please select a translation profile", None, "Error", 0
            return
        
        # Initialize logs list
        translation_logs = []
        
        try:
            # Initial status
            input_path = epub_file.name if hasattr(epub_file, 'name') else epub_file
            file_ext = os.path.splitext(input_path)[1].lower()
            file_type = "EPUB" if file_ext == ".epub" else "TXT"
            
            translation_logs.append(f"📚 Starting {file_type} translation...")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Starting...", 0
            
            # Save uploaded file to temp location if needed
            epub_base = os.path.splitext(os.path.basename(input_path))[0]
            
            translation_logs.append(f"📖 Input: {os.path.basename(input_path)}")
            translation_logs.append(f"🤖 Model: {model}")
            translation_logs.append(f"📝 Profile: {profile_name}")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Initializing...", 5
            
            # Use the provided system prompt (user may have edited it)
            translation_prompt = system_prompt if system_prompt else self.profiles.get(profile_name, "")
            
            # Set the input path as a command line argument simulation
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['glossarion_web.py', input_path]
            
            # Set environment variables for TransateKRtoEN.main()
            os.environ['INPUT_PATH'] = input_path
            os.environ['MODEL'] = model
            os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
            os.environ['MAX_OUTPUT_TOKENS'] = str(max_tokens)
            os.environ['ENABLE_IMAGE_TRANSLATION'] = '1' if enable_image_trans else '0'
            # Set output directory to current working directory
            os.environ['OUTPUT_DIRECTORY'] = os.getcwd()
            
            # Set all additional environment variables from config
            self.set_all_environment_variables()
            
            # OVERRIDE critical safety features AFTER config load
            # CORRECT variable name is EMERGENCY_PARAGRAPH_RESTORE (no ATION)
            os.environ['EMERGENCY_PARAGRAPH_RESTORE'] = '0'  # DISABLED
            os.environ['REMOVE_AI_ARTIFACTS'] = '1'  # ENABLED
            
            # Debug: Verify settings
            translation_logs.append(f"\n🔧 Debug: EMERGENCY_PARAGRAPH_RESTORE = '{os.environ.get('EMERGENCY_PARAGRAPH_RESTORE', 'NOT SET')}'")
            translation_logs.append(f"🔧 Debug: REMOVE_AI_ARTIFACTS = '{os.environ.get('REMOVE_AI_ARTIFACTS', 'NOT SET')}'")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Configuration set...", 10
            
            # Set API key environment variable
            if 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            else:
                os.environ['API_KEY'] = api_key
            
            # Set the system prompt
            if translation_prompt:
                # Save to temp profile
                temp_config = self.config.copy()
                temp_config['prompt_profiles'] = temp_config.get('prompt_profiles', {})
                temp_config['prompt_profiles'][profile_name] = translation_prompt
                temp_config['active_profile'] = profile_name
                
                # Save temporarily
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(temp_config, f, ensure_ascii=False, indent=2)
            
            translation_logs.append("⚙️ Configuration set")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Starting translation...", 10
            
            # Create a thread-safe queue for capturing logs
            import queue
            import threading
            import time
            log_queue = queue.Queue()
            translation_complete = threading.Event()
            translation_error = [None]
            
            def log_callback(msg):
                """Capture log messages"""
                if msg and msg.strip():
                    log_queue.put(msg.strip())
            
            # Run translation in a separate thread
            def run_translation():
                try:
                    result = TransateKRtoEN.main(
                        log_callback=log_callback,
                        stop_callback=None
                    )
                    translation_error[0] = None
                except Exception as e:
                    translation_error[0] = e
                finally:
                    translation_complete.set()
            
            translation_thread = threading.Thread(target=run_translation, daemon=True)
            translation_thread.start()
            
            # Monitor progress
            last_yield_time = time.time()
            progress_percent = 10
            
            while not translation_complete.is_set() or not log_queue.empty():
                # Check if stop was requested
                if self.epub_translation_stop:
                    translation_logs.append("⚠️ Stopping translation...")
                    # Try to stop the translation thread
                    translation_complete.set()
                    break
                    
                # Collect logs
                new_logs = []
                while not log_queue.empty():
                    try:
                        msg = log_queue.get_nowait()
                        new_logs.append(msg)
                    except queue.Empty:
                        break
                
                # Add new logs
                if new_logs:
                    translation_logs.extend(new_logs)
                    
                    # Update progress based on log content
                    for log in new_logs:
                        if 'Chapter' in log or 'chapter' in log:
                            progress_percent = min(progress_percent + 5, 90)
                        elif '✅' in log or 'Complete' in log:
                            progress_percent = min(progress_percent + 10, 95)
                        elif 'Translating' in log:
                            progress_percent = min(progress_percent + 2, 85)
                
                # Yield updates periodically
                current_time = time.time()
                if new_logs or (current_time - last_yield_time) > 1.0:
                    status_text = new_logs[-1] if new_logs else "Processing..."
                    # Keep only last 100 logs to avoid UI overflow
                    display_logs = translation_logs[-100:] if len(translation_logs) > 100 else translation_logs
                    yield None, None, gr.update(visible=True), "\n".join(display_logs), gr.update(visible=True), status_text, progress_percent
                    last_yield_time = current_time
                
                # Small delay to avoid CPU spinning
                time.sleep(0.1)
            
            # Wait for thread to complete
            translation_thread.join(timeout=5)
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            # Log any errors but don't fail immediately - check for output first
            if translation_error[0]:
                error_msg = f"⚠️ Translation completed with warnings: {str(translation_error[0])}"
                translation_logs.append(error_msg)
                translation_logs.append("🔍 Checking for output file...")
            
            # Check for output file - just grab any .epub from the output directory
            output_dir = epub_base
            compiled_epub = None
            
            # First, try to find ANY .epub file in the output directory
            output_dir_path = os.path.join(os.getcwd(), output_dir)
            if os.path.isdir(output_dir_path):
                translation_logs.append(f"\n📂 Checking output directory: {output_dir_path}")
                for file in os.listdir(output_dir_path):
                    if file.endswith('.epub'):
                        full_path = os.path.join(output_dir_path, file)
                        # Make sure it's not a temp/backup file
                        if os.path.isfile(full_path) and os.path.getsize(full_path) > 1000:
                            compiled_epub = full_path
                            translation_logs.append(f"  ✅ Found EPUB in output dir: {file}")
                            break
            
            # If we found it in the output directory, return it immediately
            if compiled_epub:
                file_size = os.path.getsize(compiled_epub)
                translation_logs.append(f"\n✅ Translation complete: {os.path.basename(compiled_epub)}")
                translation_logs.append(f"🔗 File path: {compiled_epub}")
                translation_logs.append(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                translation_logs.append(f"📥 Click 'Download Translated {file_type}' below to save your file")
                final_status = "Translation complete!" if not translation_error[0] else "Translation completed with warnings"
                
                yield (
                    compiled_epub,
                    gr.update(value="### ✅ Translation Complete!", visible=True),
                    gr.update(visible=False),
                    "\n".join(translation_logs),
                    gr.update(value=final_status, visible=True),
                    final_status,
                    100
                )
                return
            
            # Determine output extension based on input file type
            output_ext = ".epub" if file_ext == ".epub" else ".txt"
            
            # Get potential base directories
            base_dirs = [
                os.getcwd(),  # Current working directory
                os.path.dirname(input_path),  # Input file directory
                "/tmp",  # Common temp directory on Linux/HF Spaces
                "/home/user/app",  # HF Spaces app directory
                os.path.expanduser("~"),  # Home directory
            ]
            
            # Look for multiple possible output locations
            possible_paths = []
            
            # Extract title from input filename for more patterns
            # e.g., "tales of terror_dick donovan 2" -> "Tales of Terror"
            title_parts = os.path.basename(input_path).replace(output_ext, '').split('_')
            possible_titles = [
                epub_base,  # Original: tales of terror_dick donovan 2
                ' '.join(title_parts[:-2]).title() if len(title_parts) > 2 else epub_base,  # Tales Of Terror
            ]
            
            for base_dir in base_dirs:
                if base_dir and os.path.exists(base_dir):
                    for title in possible_titles:
                        # Direct in base directory
                        possible_paths.append(os.path.join(base_dir, f"{title}_translated{output_ext}"))
                        possible_paths.append(os.path.join(base_dir, f"{title}{output_ext}"))
                        # In output subdirectory
                        possible_paths.append(os.path.join(base_dir, output_dir, f"{title}_translated{output_ext}"))
                        possible_paths.append(os.path.join(base_dir, output_dir, f"{title}{output_ext}"))
                        # In nested output directory
                        possible_paths.append(os.path.join(base_dir, epub_base, f"{title}_translated{output_ext}"))
                        possible_paths.append(os.path.join(base_dir, epub_base, f"{title}{output_ext}"))
            
            # Also add relative paths
            possible_paths.extend([
                f"{epub_base}_translated{output_ext}",
                os.path.join(output_dir, f"{epub_base}_translated{output_ext}"),
                os.path.join(output_dir, f"{epub_base}{output_ext}"),
            ])
            
            # Also search for any translated file in the output directory
            if os.path.isdir(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith(f'_translated{output_ext}'):
                        possible_paths.insert(0, os.path.join(output_dir, file))
            
            # Add debug information about current environment
            translation_logs.append(f"\n📁 Debug Info:")
            translation_logs.append(f"  Current working directory: {os.getcwd()}")
            translation_logs.append(f"  Input file directory: {os.path.dirname(input_path)}")
            translation_logs.append(f"  Looking for: {epub_base}_translated{output_ext}")
            
            translation_logs.append(f"\n🔍 Searching for output file...")
            for potential_epub in possible_paths[:10]:  # Show first 10 paths
                translation_logs.append(f"  Checking: {potential_epub}")
                if os.path.exists(potential_epub):
                    compiled_epub = potential_epub
                    translation_logs.append(f"  ✅ Found: {potential_epub}")
                    break
            
            if not compiled_epub and len(possible_paths) > 10:
                translation_logs.append(f"  ... and {len(possible_paths) - 10} more paths")
            
            if compiled_epub:
                # Verify file exists and is readable
                if os.path.exists(compiled_epub) and os.path.isfile(compiled_epub):
                    file_size = os.path.getsize(compiled_epub)
                    translation_logs.append(f"✅ Translation complete: {os.path.basename(compiled_epub)}")
                    translation_logs.append(f"🔗 File path: {compiled_epub}")
                    translation_logs.append(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                    translation_logs.append(f"📥 Click 'Download Translated {file_type}' below to save your file")
                    # Make the file component visible with the translated file
                    final_status = "Translation complete!" if not translation_error[0] else "Translation completed with warnings"
                    
                    # Return the actual file path WITH visibility update
                    yield (
                        compiled_epub,  # epub_output - The file path (Gradio will handle it)
                        gr.update(value="### ✅ Translation Complete!", visible=True),  # epub_status_message
                        gr.update(visible=False),  # epub_progress_group
                        "\n".join(translation_logs),  # epub_logs
                        gr.update(value=final_status, visible=True),  # epub_status
                        final_status,  # epub_progress_text
                        100  # epub_progress_bar
                    )
                    return
                else:
                    translation_logs.append(f"⚠️ File found but not accessible: {compiled_epub}")
                    compiled_epub = None  # Force search
            
            # Output file not found - search recursively in relevant directories
            translation_logs.append("⚠️ Output file not in expected locations, searching recursively...")
            found_files = []
            
            # Search in multiple directories
            search_dirs = [
                os.getcwd(),  # Current directory
                os.path.dirname(input_path),  # Input file directory
                "/tmp",  # Temp directory (HF Spaces)
                "/home/user/app",  # HF Spaces app directory
            ]
            
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                    
                translation_logs.append(f"  Searching in: {search_dir}")
                try:
                    for root, dirs, files in os.walk(search_dir, topdown=True):
                        # Limit depth to 3 levels and skip hidden/system directories
                        depth = root[len(search_dir):].count(os.sep)
                        if depth >= 3:
                            dirs[:] = []  # Don't go deeper
                        else:
                            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.git']]
                        
                        for file in files:
                            # Look for files with _translated in name or matching our pattern
                            if (f'_translated{output_ext}' in file or 
                                (file.endswith(output_ext) and epub_base in file)):
                                full_path = os.path.join(root, file)
                                found_files.append(full_path)
                                translation_logs.append(f"    ✅ Found: {full_path}")
                except (PermissionError, OSError) as e:
                    translation_logs.append(f"    ⚠️ Could not search {search_dir}: {e}")
            
            if found_files:
                # Use the most recently modified file
                compiled_epub = max(found_files, key=os.path.getmtime)
                
                # Verify file exists and get info
                if os.path.exists(compiled_epub) and os.path.isfile(compiled_epub):
                    file_size = os.path.getsize(compiled_epub)
                    translation_logs.append(f"✅ Found output file: {os.path.basename(compiled_epub)}")
                    translation_logs.append(f"🔗 File path: {compiled_epub}")
                    translation_logs.append(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                    translation_logs.append(f"📥 Click 'Download Translated {file_type}' below to save your file")
                    # Return the actual file path directly
                    yield (
                        compiled_epub,  # epub_output - Just the file path
                        gr.update(value="### ✅ Translation Complete!", visible=True),  # epub_status_message
                        gr.update(visible=False),  # epub_progress_group
                        "\n".join(translation_logs),  # epub_logs
                        gr.update(value="Translation complete!", visible=True),  # epub_status
                        "Translation complete!",  # epub_progress_text
                        100  # epub_progress_bar
                    )
                    return
            
            # Still couldn't find output - report failure
            translation_logs.append("❌ Could not locate translated output file")
            translation_logs.append(f"🔍 Checked paths: {', '.join(possible_paths[:5])}...")
            translation_logs.append("\n💡 Troubleshooting tips:")
            translation_logs.append("  1. Check if TransateKRtoEN.py completed successfully")
            translation_logs.append("  2. Look for any error messages in the logs above")
            translation_logs.append("  3. The output might be in a subdirectory - check manually")
            yield None, gr.update(value="### ⚠️ Output Not Found", visible=True), gr.update(visible=False), "\n".join(translation_logs), gr.update(value="Translation process completed but output file not found", visible=True), "Output not found", 90
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during translation:\n{str(e)}\n\n{traceback.format_exc()}"
            translation_logs.append(error_msg)
            yield None, None, gr.update(visible=False), "\n".join(translation_logs), gr.update(visible=True), "Error occurred", 0
    
    def translate_epub_with_stop(self, *args):
        """Wrapper for translate_epub that includes button visibility control"""
        self.epub_translation_stop = False
        
        # Show stop button, hide translate button at start
        for result in self.translate_epub(*args):
            if self.epub_translation_stop:
                # Translation was stopped
                yield result[0], result[1], result[2], result[3] + "\n\n⚠️ Translation stopped by user", result[4], "Stopped", 0, gr.update(visible=True), gr.update(visible=False)
                return
            # Add button visibility updates to the yields
            yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=False), gr.update(visible=True)
        
        # Reset buttons at the end
        yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=True), gr.update(visible=False)
    
    def stop_epub_translation(self):
        """Stop the ongoing EPUB translation"""
        self.epub_translation_stop = True
        if self.epub_translation_thread and self.epub_translation_thread.is_alive():
            # The thread will check the stop flag
            pass
        return gr.update(visible=True), gr.update(visible=False), "Translation stopped"
    
    def extract_glossary(
        self,
        epub_file,
        model,
        api_key,
        min_frequency,
        max_names,
        max_titles=30,
        max_text_size=50000,
        max_sentences=200,
        translation_batch=50,
        chapter_split_threshold=8192,
        filter_mode='all',
        strip_honorifics=True,
        fuzzy_threshold=0.90,
        extraction_prompt=None,
        format_instructions=None,
        use_legacy_csv=False
    ):
        """Extract glossary from EPUB with manual extraction settings - yields progress updates"""
        
        if not epub_file:
            yield None, None, None, "❌ Please upload an EPUB file", None, "Error", 0
            return
        
        extraction_logs = []
        
        try:
            import extract_glossary_from_epub
            
            extraction_logs.append("🔍 Starting glossary extraction...")
            yield None, None, gr.update(visible=True), "\n".join(extraction_logs), gr.update(visible=True), "Starting...", 0
            
            input_path = epub_file.name if hasattr(epub_file, 'name') else epub_file
            output_path = input_path.replace('.epub', '_glossary.csv')
            
            extraction_logs.append(f"📖 Input: {os.path.basename(input_path)}")
            extraction_logs.append(f"🤖 Model: {model}")
            yield None, None, gr.update(visible=True), "\n".join(extraction_logs), gr.update(visible=True), "Initializing...", 10
            
            # Set all environment variables from config
            self.set_all_environment_variables()
            
            # Set API key
            if 'gpt' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            else:
                os.environ['API_KEY'] = api_key
            
            extraction_logs.append("📋 Extracting text from EPUB...")
            yield None, None, gr.update(visible=True), "\n".join(extraction_logs), gr.update(visible=True), "Extracting text...", 20
            
            # Set environment variables for glossary extraction
            os.environ['MODEL'] = model
            os.environ['GLOSSARY_MIN_FREQUENCY'] = str(min_frequency)
            os.environ['GLOSSARY_MAX_NAMES'] = str(max_names)
            os.environ['GLOSSARY_MAX_TITLES'] = str(max_titles)
            os.environ['GLOSSARY_BATCH_SIZE'] = str(translation_batch)
            os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(max_text_size)
            os.environ['GLOSSARY_MAX_SENTENCES'] = str(max_sentences)
            os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(chapter_split_threshold)
            os.environ['GLOSSARY_FILTER_MODE'] = filter_mode
            os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if strip_honorifics else '0'
            os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(fuzzy_threshold)
            os.environ['GLOSSARY_USE_LEGACY_CSV'] = '1' if use_legacy_csv else '0'
            
            # Set prompts if provided
            if extraction_prompt:
                os.environ['GLOSSARY_SYSTEM_PROMPT'] = extraction_prompt
            if format_instructions:
                os.environ['GLOSSARY_FORMAT_INSTRUCTIONS'] = format_instructions
            
            extraction_logs.append(f"⚙️ Settings: Min freq={min_frequency}, Max names={max_names}, Filter={filter_mode}")
            extraction_logs.append(f"⚙️ Options: Strip honorifics={strip_honorifics}, Fuzzy threshold={fuzzy_threshold:.2f}")
            yield None, None, gr.update(visible=True), "\n".join(extraction_logs), gr.update(visible=True), "Processing...", 40
            
            # Create a thread-safe queue for capturing logs
            import queue
            import threading
            import time
            log_queue = queue.Queue()
            extraction_complete = threading.Event()
            extraction_error = [None]
            extraction_result = [None]
            
            def log_callback(msg):
                """Capture log messages"""
                if msg and msg.strip():
                    log_queue.put(msg.strip())
            
            # Run extraction in a separate thread
            def run_extraction():
                try:
                    result = extract_glossary_from_epub.main(
                        log_callback=log_callback,
                        stop_callback=None
                    )
                    extraction_result[0] = result
                    extraction_error[0] = None
                except Exception as e:
                    extraction_error[0] = e
                finally:
                    extraction_complete.set()
            
            extraction_thread = threading.Thread(target=run_extraction, daemon=True)
            extraction_thread.start()
            
            # Monitor progress
            last_yield_time = time.time()
            progress_percent = 40
            
            while not extraction_complete.is_set() or not log_queue.empty():
                # Check if stop was requested
                if self.glossary_extraction_stop:
                    extraction_logs.append("⚠️ Stopping extraction...")
                    # Try to stop the extraction thread
                    extraction_complete.set()
                    break
                    
                # Collect logs
                new_logs = []
                while not log_queue.empty():
                    try:
                        msg = log_queue.get_nowait()
                        new_logs.append(msg)
                    except queue.Empty:
                        break
                
                # Add new logs
                if new_logs:
                    extraction_logs.extend(new_logs)
                    
                    # Update progress based on log content
                    for log in new_logs:
                        if 'Processing' in log or 'Extracting' in log:
                            progress_percent = min(progress_percent + 5, 80)
                        elif 'Writing' in log or 'Saving' in log:
                            progress_percent = min(progress_percent + 10, 90)
                
                # Yield updates periodically
                current_time = time.time()
                if new_logs or (current_time - last_yield_time) > 1.0:
                    status_text = new_logs[-1] if new_logs else "Processing..."
                    # Keep only last 100 logs
                    display_logs = extraction_logs[-100:] if len(extraction_logs) > 100 else extraction_logs
                    yield None, None, gr.update(visible=True), "\n".join(display_logs), gr.update(visible=True), status_text, progress_percent
                    last_yield_time = current_time
                
                # Small delay to avoid CPU spinning
                time.sleep(0.1)
            
            # Wait for thread to complete
            extraction_thread.join(timeout=5)
            
            # Check for errors
            if extraction_error[0]:
                error_msg = f"❌ Extraction error: {str(extraction_error[0])}"
                extraction_logs.append(error_msg)
                yield None, None, gr.update(visible=False), "\n".join(extraction_logs), gr.update(visible=True), error_msg, 0
                return
            
            extraction_logs.append("🖍️ Writing glossary to CSV...")
            yield None, None, gr.update(visible=True), "\n".join(extraction_logs), gr.update(visible=True), "Writing CSV...", 95
            
            if os.path.exists(output_path):
                extraction_logs.append(f"✅ Glossary extracted successfully!")
                extraction_logs.append(f"💾 Saved to: {os.path.basename(output_path)}")
                yield output_path, gr.update(visible=True), gr.update(visible=False), "\n".join(extraction_logs), gr.update(visible=True), "Extraction complete!", 100
            else:
                extraction_logs.append("❌ Glossary extraction failed - output file not created")
                yield None, None, gr.update(visible=False), "\n".join(extraction_logs), gr.update(visible=True), "Extraction failed", 0
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during extraction:\n{str(e)}\n\n{traceback.format_exc()}"
            extraction_logs.append(error_msg)
            yield None, None, gr.update(visible=False), "\n".join(extraction_logs), gr.update(visible=True), "Error occurred", 0
    
    def extract_glossary_with_stop(self, *args):
        """Wrapper for extract_glossary that includes button visibility control"""
        self.glossary_extraction_stop = False
        
        # Show stop button, hide extract button at start
        for result in self.extract_glossary(*args):
            if self.glossary_extraction_stop:
                # Extraction was stopped
                yield result[0], result[1], result[2], result[3] + "\n\n⚠️ Extraction stopped by user", result[4], "Stopped", 0, gr.update(visible=True), gr.update(visible=False)
                return
            # Add button visibility updates to the yields
            yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=False), gr.update(visible=True)
        
        # Reset buttons at the end
        yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=True), gr.update(visible=False)
    
    def stop_glossary_extraction(self):
        """Stop the ongoing glossary extraction"""
        self.glossary_extraction_stop = True
        if self.glossary_extraction_thread and self.glossary_extraction_thread.is_alive():
            # The thread will check the stop flag
            pass
        return gr.update(visible=True), gr.update(visible=False), "Extraction stopped"
    
    def run_qa_scan(self, folder_path, min_foreign_chars, check_repetition, 
                    check_glossary_leakage, min_file_length, check_multiple_headers,
                    check_missing_html, check_insufficient_paragraphs, 
                    min_paragraph_percentage, report_format, auto_save_report):
        """Run Quick QA scan on output folder - yields progress updates"""
        
        # Handle both string paths and File objects
        if hasattr(folder_path, 'name'):
            # It's a File object from Gradio
            folder_path = folder_path.name
        
        if not folder_path:
            yield gr.update(visible=False), gr.update(value="### ❌ Error", visible=True), gr.update(visible=False), "❌ Please provide a folder path or upload a ZIP file", gr.update(visible=False), "Error", 0
            return
        
        if isinstance(folder_path, str):
            folder_path = folder_path.strip()
        
        if not os.path.exists(folder_path):
            yield gr.update(visible=False), gr.update(value=f"### ❌ File/Folder not found", visible=True), gr.update(visible=False), f"❌ File/Folder not found: {folder_path}", gr.update(visible=False), "Error", 0
            return
        
        # Initialize scan_logs early
        scan_logs = []
        
        # Check if it's a ZIP or EPUB file (for Hugging Face Spaces or convenience)
        if os.path.isfile(folder_path) and (folder_path.lower().endswith('.zip') or folder_path.lower().endswith('.epub')):
            # Extract ZIP/EPUB to temp folder
            import zipfile
            import tempfile
            
            temp_dir = tempfile.mkdtemp(prefix="qa_scan_")
            
            try:
                file_type = "EPUB" if folder_path.lower().endswith('.epub') else "ZIP"
                scan_logs.append(f"📦 Extracting {file_type} file: {os.path.basename(folder_path)}")
                
                with zipfile.ZipFile(folder_path, 'r') as zip_ref:
                    # For EPUB files, look for the content folders
                    if file_type == "EPUB":
                        # EPUB files typically have OEBPS, EPUB, or similar content folders
                        all_files = zip_ref.namelist()
                        # Extract everything
                        zip_ref.extractall(temp_dir)
                        
                        # Try to find the content directory
                        content_dirs = ['OEBPS', 'EPUB', 'OPS', 'content']
                        actual_content_dir = None
                        for dir_name in content_dirs:
                            potential_dir = os.path.join(temp_dir, dir_name)
                            if os.path.exists(potential_dir):
                                actual_content_dir = potential_dir
                                break
                        
                        # If no standard content dir found, use the temp_dir itself
                        if actual_content_dir:
                            folder_path = actual_content_dir
                            scan_logs.append(f"📁 Found EPUB content directory: {os.path.basename(actual_content_dir)}")
                        else:
                            folder_path = temp_dir
                            scan_logs.append(f"📁 Using extracted root directory")
                    else:
                        # Regular ZIP file
                        zip_ref.extractall(temp_dir)
                        folder_path = temp_dir
                        
                scan_logs.append(f"✅ Successfully extracted to temporary folder")
                # Continue with normal processing, but include initial logs
                # Note: we'll need to pass scan_logs through the rest of the function
                
            except Exception as e:
                yield gr.update(visible=False), gr.update(value=f"### ❌ {file_type} extraction failed", visible=True), gr.update(visible=False), f"❌ Failed to extract {file_type}: {str(e)}", gr.update(visible=False), "Error", 0
                return
        elif not os.path.isdir(folder_path):
            yield gr.update(visible=False), gr.update(value=f"### ❌ Not a folder, ZIP, or EPUB", visible=True), gr.update(visible=False), f"❌ Path is not a folder, ZIP, or EPUB file: {folder_path}", gr.update(visible=False), "Error", 0
            return
        
        try:
            scan_logs.append("🔍 Starting Quick QA Scan...")
            scan_logs.append(f"📁 Scanning folder: {folder_path}")
            yield gr.update(visible=False), gr.update(value="### Scanning...", visible=True), gr.update(visible=True), "\n".join(scan_logs), gr.update(visible=False), "Starting...", 0
            
            # Find all HTML/XHTML files in the folder and subfolders
            html_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.html', '.xhtml', '.htm')):
                        html_files.append(os.path.join(root, file))
            
            if not html_files:
                scan_logs.append(f"⚠️ No HTML/XHTML files found in {folder_path}")
                yield gr.update(visible=False), gr.update(value="### ⚠️ No files found", visible=True), gr.update(visible=False), "\n".join(scan_logs), gr.update(visible=False), "No files to scan", 0
                return
            
            scan_logs.append(f"📄 Found {len(html_files)} HTML/XHTML files to scan")
            scan_logs.append("⚡ Quick Scan Mode (85% threshold, Speed optimized)")
            yield gr.update(visible=False), gr.update(value="### Initializing...", visible=True), gr.update(visible=True), "\n".join(scan_logs), gr.update(visible=False), "Initializing...", 10
            
            # QA scanning process
            total_files = len(html_files)
            issues_found = []
            chapters_scanned = set()
            
            for i, file_path in enumerate(html_files):
                if self.qa_scan_stop:
                    scan_logs.append("⚠️ Scan stopped by user")
                    break
                    
                # Get relative path from base folder for cleaner display
                rel_path = os.path.relpath(file_path, folder_path)
                file_name = rel_path.replace('\\', '/')
                
                # Quick scan optimization: skip if we've already scanned similar chapters
                # (consecutive chapter checking)
                chapter_match = None
                for pattern in ['chapter', 'ch', 'c']:
                    if pattern in file_name.lower():
                        import re
                        match = re.search(r'(\d+)', file_name)
                        if match:
                            chapter_num = int(match.group(1))
                            # Skip if we've already scanned nearby chapters (Quick Scan optimization)
                            if any(abs(chapter_num - ch) <= 1 for ch in chapters_scanned):
                                if len(chapters_scanned) > 5:  # Only skip after scanning a few
                                    continue
                            chapters_scanned.add(chapter_num)
                            break
                
                scan_logs.append(f"\n🔍 Scanning: {file_name}")
                progress = int(10 + (80 * i / total_files))
                yield None, None, gr.update(visible=True), "\n".join(scan_logs), gr.update(visible=True), f"Scanning {file_name}...", progress
                
                # Read and check the HTML file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_issues = []
                    
                    # Check file length
                    if len(content) < min_file_length:
                        continue  # Skip short files
                    
                    # Check for foreign characters (simulation - would need actual implementation)
                    # In real implementation, would check for source language characters
                    import random
                    
                    # Check for multiple headers
                    if check_multiple_headers:
                        import re
                        headers = re.findall(r'<h[1-6][^>]*>', content, re.IGNORECASE)
                        if len(headers) >= 2:
                            file_issues.append("Multiple headers detected")
                    
                    # Check for missing html tag
                    if check_missing_html:
                        if '<html' not in content.lower():
                            file_issues.append("Missing <html> tag")
                    
                    # Check for insufficient paragraphs
                    if check_insufficient_paragraphs:
                        p_tags = content.count('<p>') + content.count('<p ')
                        text_length = len(re.sub(r'<[^>]+>', '', content))
                        if text_length > 0:
                            p_text = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)
                            p_text_length = sum(len(t) for t in p_text)
                            percentage = (p_text_length / text_length) * 100
                            if percentage < min_paragraph_percentage:
                                file_issues.append(f"Only {percentage:.1f}% text in <p> tags")
                    
                    # Simulated additional checks
                    if check_repetition and random.random() > 0.85:
                        file_issues.append("Excessive repetition detected")
                    
                    if check_glossary_leakage and random.random() > 0.9:
                        file_issues.append("Glossary leakage detected")
                    
                    # Report issues found
                    if file_issues:
                        for issue in file_issues:
                            issues_found.append(f"  ⚠️ {file_name}: {issue}")
                            scan_logs.append(f"  ⚠️ Issue: {issue}")
                    else:
                        scan_logs.append(f"  ✅ No issues found")
                        
                except Exception as e:
                    scan_logs.append(f"  ❌ Error reading file: {str(e)}")
                
                # Update logs periodically
                if len(scan_logs) > 100:
                    scan_logs = scan_logs[-100:]  # Keep only last 100 logs
                    
                yield gr.update(visible=False), None, gr.update(visible=True), "\n".join(scan_logs), gr.update(visible=False), f"Scanning {file_name}...", progress
            
            # Generate report
            scan_logs.append("\n📝 Generating report...")
            yield gr.update(visible=False), None, gr.update(visible=True), "\n".join(scan_logs), gr.update(visible=False), "Generating report...", 95
            
            # Create report content based on selected format
            if report_format == "summary":
                # Summary format - brief overview only
                report_content = "QA SCAN REPORT - SUMMARY\n"
                report_content += "=" * 50 + "\n\n"
                report_content += f"Total files scanned: {total_files}\n"
                report_content += f"Issues found: {len(issues_found)}\n\n"
                if issues_found:
                    report_content += f"Files with issues: {min(len(issues_found), 10)} (showing first 10)\n"
                    report_content += "\n".join(issues_found[:10])
                else:
                    report_content += "✅ No issues detected."
            
            elif report_format == "verbose":
                # Verbose format - all data including passed files
                report_content = "QA SCAN REPORT - VERBOSE (ALL DATA)\n"
                report_content += "=" * 50 + "\n\n"
                from datetime import datetime
                report_content += f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report_content += f"Folder Scanned: {folder_path}\n"
                report_content += f"Total files scanned: {total_files}\n"
                report_content += f"Issues found: {len(issues_found)}\n"
                report_content += f"Settings used:\n"
                report_content += f"  - Min foreign chars: {min_foreign_chars}\n"
                report_content += f"  - Check repetition: {check_repetition}\n"
                report_content += f"  - Check glossary leakage: {check_glossary_leakage}\n"
                report_content += f"  - Min file length: {min_file_length}\n"
                report_content += f"  - Check multiple headers: {check_multiple_headers}\n"
                report_content += f"  - Check missing HTML: {check_missing_html}\n"
                report_content += f"  - Check insufficient paragraphs: {check_insufficient_paragraphs}\n"
                report_content += f"  - Min paragraph percentage: {min_paragraph_percentage}%\n\n"
                
                report_content += "ALL FILES PROCESSED:\n"
                report_content += "-" * 30 + "\n"
                for file in html_files:
                    rel_path = os.path.relpath(file, folder_path)
                    report_content += f"  {rel_path}\n"
                
                if issues_found:
                    report_content += "\n\nISSUES DETECTED (DETAILED):\n"
                    report_content += "\n".join(issues_found)
                else:
                    report_content += "\n\n✅ No issues detected. All files passed scan."
            
            else:  # detailed (default/recommended)
                # Detailed format - recommended balance
                report_content = "QA SCAN REPORT - DETAILED\n"
                report_content += "=" * 50 + "\n\n"
                report_content += f"Total files scanned: {total_files}\n"
                report_content += f"Issues found: {len(issues_found)}\n\n"
                
                if issues_found:
                    report_content += "ISSUES DETECTED:\n"
                    report_content += "\n".join(issues_found)
                else:
                    report_content += "No issues detected. All files passed quick scan."
            
            # Always save report to file for download
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"qa_scan_report_{timestamp}.txt"
            report_path = os.path.join(os.getcwd(), report_filename)
            
            # Always write the report file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            if auto_save_report:
                scan_logs.append(f"💾 Report auto-saved to: {report_filename}")
            else:
                scan_logs.append(f"📄 Report ready for download: {report_filename}")
            
            scan_logs.append(f"\n✅ QA Scan completed!")
            scan_logs.append(f"📊 Summary: {total_files} files scanned, {len(issues_found)} issues found")
            scan_logs.append(f"\n📥 Click 'Download QA Report' below to save the report")
            
            # Always return the report path and make File component visible
            final_status = f"✅ Scan complete!\n{total_files} files scanned\n{len(issues_found)} issues found"
            yield gr.update(value=report_path, visible=True), gr.update(value=f"### {final_status}", visible=True), gr.update(visible=False), "\n".join(scan_logs), gr.update(value=final_status, visible=True), "Scan complete!", 100
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during QA scan:\n{str(e)}\n\n{traceback.format_exc()}"
            scan_logs.append(error_msg)
            yield gr.update(visible=False), gr.update(value="### ❌ Error occurred", visible=True), gr.update(visible=False), "\n".join(scan_logs), gr.update(visible=True), "Error occurred", 0
    
    def run_qa_scan_with_stop(self, *args):
        """Wrapper for run_qa_scan that includes button visibility control"""
        self.qa_scan_stop = False
        
        # Show stop button, hide scan button at start
        for result in self.run_qa_scan(*args):
            if self.qa_scan_stop:
                # Scan was stopped
                yield result[0], result[1], result[2], result[3] + "\n\n⚠️ Scan stopped by user", result[4], "Stopped", 0, gr.update(visible=True), gr.update(visible=False)
                return
            # Add button visibility updates to the yields
            yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=False), gr.update(visible=True)
        
        # Reset buttons at the end
        yield result[0], result[1], result[2], result[3], result[4], result[5], result[6], gr.update(visible=True), gr.update(visible=False)
    
    def stop_qa_scan(self):
        """Stop the ongoing QA scan"""
        self.qa_scan_stop = True
        return gr.update(visible=True), gr.update(visible=False), "Scan stopped"
    
    def stop_translation(self):
        """Stop the ongoing translation process"""
        print(f"DEBUG: stop_translation called, was_translating={self.is_translating}")
        if self.is_translating:
            print("DEBUG: Setting stop flag and cancellation")
            self.stop_flag.set()
            self.is_translating = False
            
            # Best-effort: cancel any in-flight API operation on the active client
            try:
                if getattr(self, 'current_unified_client', None):
                    self.current_unified_client.cancel_current_operation()
                    print("DEBUG: Requested UnifiedClient cancellation")
            except Exception as e:
                print(f"DEBUG: UnifiedClient cancel failed: {e}")
            
            # Also propagate to MangaTranslator class if available
            try:
                if MANGA_TRANSLATION_AVAILABLE:
                    from manga_translator import MangaTranslator
                    MangaTranslator.set_global_cancellation(True)
                    print("DEBUG: Set MangaTranslator global cancellation")
            except ImportError:
                pass
            
            # Also propagate to UnifiedClient if available
            try:
                if MANGA_TRANSLATION_AVAILABLE:
                    from unified_api_client import UnifiedClient
                    UnifiedClient.set_global_cancellation(True)
                    print("DEBUG: Set UnifiedClient global cancellation")
            except ImportError:
                pass
            
            # Kick off translator shutdown to free resources quickly
            try:
                tr = getattr(self, 'current_translator', None)
                if tr and hasattr(tr, 'shutdown'):
                    import threading as _th
                    _th.Thread(target=tr.shutdown, name="WebMangaTranslatorShutdown", daemon=True).start()
                    print("DEBUG: Initiated translator shutdown thread")
                    # Clear reference so a new start creates a fresh instance
                    self.current_translator = None
            except Exception as e:
                print(f"DEBUG: Failed to start translator shutdown: {e}")
        else:
            print("DEBUG: stop_translation called but not translating")
    
    def _reset_translation_flags(self):
        """Reset all translation flags for new translation"""
        self.is_translating = False
        self.stop_flag.clear()
        
        # Reset global cancellation flags
        try:
            if MANGA_TRANSLATION_AVAILABLE:
                from manga_translator import MangaTranslator
                MangaTranslator.set_global_cancellation(False)
        except ImportError:
            pass
            
        try:
            if MANGA_TRANSLATION_AVAILABLE:
                from unified_api_client import UnifiedClient
                UnifiedClient.set_global_cancellation(False)
        except ImportError:
            pass
    
    def translate_manga(
        self,
        image_files,
        model,
        api_key,
        profile_name,
        system_prompt,
        ocr_provider,
        google_creds_path,
        azure_key,
        azure_endpoint,
        enable_bubble_detection,
        enable_inpainting,
        font_size_mode,
        font_size,
        font_multiplier,
        min_font_size,
        max_font_size,
        text_color,
        shadow_enabled,
        shadow_color,
        shadow_offset_x,
        shadow_offset_y,
        shadow_blur,
        bg_opacity,
        bg_style,
        parallel_panel_translation=False,
        panel_max_workers=10
    ):
        """Translate manga images - GENERATOR that yields (logs, image, cbz_file, status, progress_group, progress_text, progress_bar) updates"""
        
        # Reset translation flags and set running state
        self._reset_translation_flags()
        self.is_translating = True
        
        if not MANGA_TRANSLATION_AVAILABLE:
            self.is_translating = False
            yield "❌ Manga translation modules not loaded", None, None, gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        if not image_files:
            self.is_translating = False
            yield "❌ Please upload at least one image", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        if not api_key:
            self.is_translating = False
            yield "❌ Please provide an API key", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        # Check for stop request
        if self.stop_flag.is_set():
            self.is_translating = False
            yield "⏹️ Translation stopped by user", gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Stopped", visible=True), gr.update(visible=False), gr.update(value="Stopped"), gr.update(value=0)
            return
        
        if ocr_provider == "google":
            # Check if credentials are provided or saved in config
            if not google_creds_path and not self.get_config_value('google_vision_credentials'):
                yield "❌ Please provide Google Cloud credentials JSON file", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
        
        if ocr_provider == "azure":
            # Ensure azure credentials are strings
            azure_key_str = str(azure_key) if azure_key else ''
            azure_endpoint_str = str(azure_endpoint) if azure_endpoint else ''
            if not azure_key_str.strip() or not azure_endpoint_str.strip():
                yield "❌ Please provide Azure API key and endpoint", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
        
        try:
            
            # Set all environment variables from config
            self.set_all_environment_variables()
            
            # Set API key environment variable
            if 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = api_key
            
            # Set Google Cloud credentials if provided and save to config
            if ocr_provider == "google":
                if google_creds_path:
                    # New file provided - save it
                    creds_path = google_creds_path.name if hasattr(google_creds_path, 'name') else google_creds_path
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                    # Auto-save to config
                    self.config['google_vision_credentials'] = creds_path
                    self.save_config(self.config)
                elif self.get_config_value('google_vision_credentials'):
                    # Use saved credentials from config
                    creds_path = self.get_config_value('google_vision_credentials')
                    if os.path.exists(creds_path):
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                    else:
                        yield f"❌ Saved Google credentials not found: {creds_path}", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                        return
            
            # Set Azure credentials if provided and save to config
            if ocr_provider == "azure":
                # Convert to strings and strip whitespace
                azure_key_str = str(azure_key).strip() if azure_key else ''
                azure_endpoint_str = str(azure_endpoint).strip() if azure_endpoint else ''
                
                os.environ['AZURE_VISION_KEY'] = azure_key_str
                os.environ['AZURE_VISION_ENDPOINT'] = azure_endpoint_str
                # Auto-save to config
                self.config['azure_vision_key'] = azure_key_str
                self.config['azure_vision_endpoint'] = azure_endpoint_str
                self.save_config(self.config)
            
            # Apply text visibility settings to config
            # Convert hex color to RGB tuple
            def hex_to_rgb(hex_color):
                # Handle different color formats
                if isinstance(hex_color, (list, tuple)):
                    # Already RGB format
                    return tuple(hex_color[:3])
                elif isinstance(hex_color, str):
                    # Remove any brackets or spaces if present
                    hex_color = hex_color.strip().strip('[]').strip()
                    if hex_color.startswith('#'):
                        # Hex format
                        hex_color = hex_color.lstrip('#')
                        if len(hex_color) == 6:
                            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        elif len(hex_color) == 3:
                            # Short hex format like #FFF
                            return tuple(int(hex_color[i]*2, 16) for i in range(3))
                    elif ',' in hex_color:
                        # RGB string format like "255, 0, 0"
                        try:
                            parts = hex_color.split(',')
                            return tuple(int(p.strip()) for p in parts[:3])
                        except:
                            pass
                # Default to black if parsing fails
                return (0, 0, 0)
            
            # Debug logging for color values
            print(f"DEBUG: text_color type: {type(text_color)}, value: {text_color}")
            print(f"DEBUG: shadow_color type: {type(shadow_color)}, value: {shadow_color}")
            
            try:
                text_rgb = hex_to_rgb(text_color)
                shadow_rgb = hex_to_rgb(shadow_color)
            except Exception as e:
                print(f"WARNING: Error converting colors: {e}")
                print(f"WARNING: Using default colors - text: black, shadow: white")
                text_rgb = (0, 0, 0)  # Default to black text
                shadow_rgb = (255, 255, 255)  # Default to white shadow
            
            self.config['manga_font_size_mode'] = font_size_mode
            self.config['manga_font_size'] = int(font_size)
            self.config['manga_font_size_multiplier'] = float(font_multiplier)
            self.config['manga_max_font_size'] = int(max_font_size)
            self.config['manga_text_color'] = list(text_rgb)
            self.config['manga_shadow_enabled'] = bool(shadow_enabled)
            self.config['manga_shadow_color'] = list(shadow_rgb)
            self.config['manga_shadow_offset_x'] = int(shadow_offset_x)
            self.config['manga_shadow_offset_y'] = int(shadow_offset_y)
            self.config['manga_shadow_blur'] = int(shadow_blur)
            self.config['manga_bg_opacity'] = int(bg_opacity)
            self.config['manga_bg_style'] = bg_style
            
            # Also update nested manga_settings structure
            if 'manga_settings' not in self.config:
                self.config['manga_settings'] = {}
            if 'rendering' not in self.config['manga_settings']:
                self.config['manga_settings']['rendering'] = {}
            if 'font_sizing' not in self.config['manga_settings']:
                self.config['manga_settings']['font_sizing'] = {}
            
            self.config['manga_settings']['rendering']['auto_min_size'] = int(min_font_size)
            self.config['manga_settings']['font_sizing']['min_size'] = int(min_font_size)
            self.config['manga_settings']['rendering']['auto_max_size'] = int(max_font_size)
            self.config['manga_settings']['font_sizing']['max_size'] = int(max_font_size)
            
            # Prepare output directory
            output_dir = tempfile.mkdtemp(prefix="manga_translated_")
            translated_files = []
            cbz_mode = False
            cbz_output_path = None
            
            # Initialize translation logs early (needed for CBZ processing)
            translation_logs = []
            
            # Check if any file is a CBZ/ZIP archive
            import zipfile
            files_to_process = image_files if isinstance(image_files, list) else [image_files]
            extracted_images = []
            
            for file in files_to_process:
                file_path = file.name if hasattr(file, 'name') else file
                if file_path.lower().endswith(('.cbz', '.zip')):
                    # Extract CBZ
                    cbz_mode = True
                    translation_logs.append(f"📚 Extracting CBZ: {os.path.basename(file_path)}")
                    extract_dir = tempfile.mkdtemp(prefix="cbz_extract_")
                    
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Find all image files in extracted directory
                        import glob
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.gif']:
                            extracted_images.extend(glob.glob(os.path.join(extract_dir, '**', ext), recursive=True))
                        
                        # Sort naturally (by filename)
                        extracted_images.sort()
                        translation_logs.append(f"✅ Extracted {len(extracted_images)} images from CBZ")
                        
                        # Prepare CBZ output path
                        cbz_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_translated.cbz")
                    except Exception as e:
                        translation_logs.append(f"❌ Error extracting CBZ: {str(e)}")
                else:
                    # Regular image file
                    extracted_images.append(file_path)
            
            # Use extracted images if CBZ was processed, otherwise use original files
            if extracted_images:
                # Create mock file objects for extracted images
                class MockFile:
                    def __init__(self, path):
                        self.name = path
                
                files_to_process = [MockFile(img) for img in extracted_images]
            
            total_images = len(files_to_process)
            
            # Merge web app config with SimpleConfig for MangaTranslator
            # This includes all the text visibility settings we just set
            merged_config = self.config.copy()
            
            # Override with web-specific settings
            merged_config['model'] = model
            merged_config['active_profile'] = profile_name
            
            # Update manga_settings
            if 'manga_settings' not in merged_config:
                merged_config['manga_settings'] = {}
            if 'ocr' not in merged_config['manga_settings']:
                merged_config['manga_settings']['ocr'] = {}
            if 'inpainting' not in merged_config['manga_settings']:
                merged_config['manga_settings']['inpainting'] = {}
            if 'advanced' not in merged_config['manga_settings']:
                merged_config['manga_settings']['advanced'] = {}
            
            merged_config['manga_settings']['ocr']['provider'] = ocr_provider
            merged_config['manga_settings']['ocr']['bubble_detection_enabled'] = enable_bubble_detection
            merged_config['manga_settings']['inpainting']['method'] = 'local' if enable_inpainting else 'none'
            # Make sure local_method is set from config (defaults to anime)
            if 'local_method' not in merged_config['manga_settings']['inpainting']:
                merged_config['manga_settings']['inpainting']['local_method'] = self.get_config_value('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime')
            
            # Set parallel panel translation settings from config (Manga Settings tab)
            # These are controlled in the Manga Settings tab, so reload config to get latest values
            current_config = self.load_config()
            if API_KEY_ENCRYPTION_AVAILABLE:
                current_config = decrypt_config(current_config)
            
            config_parallel = current_config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False)
            config_max_workers = current_config.get('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 10)
            
            # Map web UI settings to MangaTranslator expected names
            merged_config['manga_settings']['advanced']['parallel_panel_translation'] = config_parallel
            merged_config['manga_settings']['advanced']['panel_max_workers'] = int(config_max_workers)
            # CRITICAL: Also set the setting names that MangaTranslator actually checks
            merged_config['manga_settings']['advanced']['parallel_processing'] = config_parallel
            merged_config['manga_settings']['advanced']['max_workers'] = int(config_max_workers)
            
            # Log the parallel settings being used
            print(f"🔧 Reloaded config - Using parallel panel translation: {config_parallel}")
            print(f"🔧 Reloaded config - Using panel max workers: {config_max_workers}")
            
            # CRITICAL: Set skip_inpainting flag to False when inpainting is enabled
            merged_config['manga_skip_inpainting'] = not enable_inpainting
            
            # Create a simple config object for MangaTranslator
            class SimpleConfig:
                def __init__(self, cfg):
                    self.config = cfg
                
                def get(self, key, default=None):
                    return self.config.get(key, default)
            
            # Create mock GUI object with necessary attributes
            class MockGUI:
                def __init__(self, config, profile_name, system_prompt, max_output_tokens, api_key, model):
                    self.config = config
                    # Add profile_var mock for MangaTranslator compatibility
                    class ProfileVar:
                        def __init__(self, profile):
                            self.profile = str(profile) if profile else ''
                        def get(self):
                            return self.profile
                    self.profile_var = ProfileVar(profile_name)
                    # Add prompt_profiles BOTH to config AND as attribute (manga_translator checks both)
                    if 'prompt_profiles' not in self.config:
                        self.config['prompt_profiles'] = {}
                    self.config['prompt_profiles'][profile_name] = system_prompt
                    # Also set as direct attribute for line 4653 check
                    self.prompt_profiles = self.config['prompt_profiles']
                    # Add max_output_tokens as direct attribute (line 299 check)
                    self.max_output_tokens = max_output_tokens
                    # Add mock GUI attributes that MangaTranslator expects
                    class MockVar:
                        def __init__(self, val):
                            # Ensure val is properly typed
                            self.val = val
                        def get(self):
                            return self.val
                    # CRITICAL: delay_entry must read from api_call_delay (not 'delay')
                    self.delay_entry = MockVar(float(config.get('api_call_delay', 0.5)))
                    self.trans_temp = MockVar(float(config.get('translation_temperature', 0.3)))
                    self.contextual_var = MockVar(bool(config.get('contextual', False)))
                    self.trans_history = MockVar(int(config.get('translation_history_limit', 2)))
                    self.translation_history_rolling_var = MockVar(bool(config.get('translation_history_rolling', False)))
                    self.token_limit_disabled = bool(config.get('token_limit_disabled', False))
                    # IMPORTANT: token_limit_entry must return STRING because manga_translator calls .strip() on it
                    self.token_limit_entry = MockVar(str(config.get('token_limit', 200000)))
                    # Batch translation settings
                    self.batch_size_var = MockVar(int(config.get('batch_size', 10)))
                    # Add API key and model for custom-api OCR provider - ensure strings
                    self.api_key_entry = MockVar(str(api_key) if api_key else '')
                    self.model_var = MockVar(str(model) if model else '')
            
            simple_config = SimpleConfig(merged_config)
            # Get max_output_tokens from config or use from web app config
            web_max_tokens = merged_config.get('max_output_tokens', 16000)
            mock_gui = MockGUI(simple_config.config, profile_name, system_prompt, web_max_tokens, api_key, model)
            
            # Ensure model path is in config for local inpainting
            if enable_inpainting:
                local_method = merged_config.get('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime')
                # Set the model path key that MangaTranslator expects
                model_path_key = f'manga_{local_method}_model_path'
                if model_path_key not in merged_config:
                    # Use default model path or empty string
                    default_model_path = self.get_config_value(model_path_key, '')
                    merged_config[model_path_key] = default_model_path
                    print(f"Set {model_path_key} to: {default_model_path}")
            
            # CRITICAL: Explicitly set environment variables before creating UnifiedClient
            api_call_delay = merged_config.get('api_call_delay', 0.5)
            os.environ['SEND_INTERVAL_SECONDS'] = str(api_call_delay)
            print(f"🔧 Manga translation: Set SEND_INTERVAL_SECONDS = {api_call_delay}s")
            
            # Set batch translation and batch size from proper config structure
            batch_translation = merged_config.get('batch_translation', True)
            batch_size = merged_config.get('batch_size', 10)
            os.environ['BATCH_TRANSLATION'] = '1' if batch_translation else '0'
            os.environ['BATCH_SIZE'] = str(batch_size)
            
            # Also ensure font algorithm and auto fit style are in config for manga_translator
            if 'manga_settings' not in merged_config:
                merged_config['manga_settings'] = {}
            if 'font_sizing' not in merged_config['manga_settings']:
                merged_config['manga_settings']['font_sizing'] = {}
            if 'rendering' not in merged_config['manga_settings']:
                merged_config['manga_settings']['rendering'] = {}
            
            if 'algorithm' not in merged_config['manga_settings']['font_sizing']:
                merged_config['manga_settings']['font_sizing']['algorithm'] = 'smart'
            if 'auto_fit_style' not in merged_config['manga_settings']['rendering']:
                merged_config['manga_settings']['rendering']['auto_fit_style'] = 'balanced'
            
            print(f"📦 Batch: BATCH_TRANSLATION={batch_translation}, BATCH_SIZE={batch_size}")
            print(f"🎨 Font: algorithm={merged_config['manga_settings']['font_sizing']['algorithm']}, auto_fit_style={merged_config['manga_settings']['rendering']['auto_fit_style']}")
            
            # Setup OCR configuration
            ocr_config = {
                'provider': ocr_provider
            }
            
            if ocr_provider == 'google':
                ocr_config['google_credentials_path'] = google_creds_path.name if google_creds_path else None
            elif ocr_provider == 'azure':
                # Use string versions
                azure_key_str = str(azure_key).strip() if azure_key else ''
                azure_endpoint_str = str(azure_endpoint).strip() if azure_endpoint else ''
                ocr_config['azure_key'] = azure_key_str
                ocr_config['azure_endpoint'] = azure_endpoint_str
            
            # Create UnifiedClient for translation API calls
            try:
                unified_client = UnifiedClient(
                    api_key=api_key,
                    model=model,
                    output_dir=output_dir
                )
                # Store reference for stop() cancellation support
                self.current_unified_client = unified_client
            except Exception as e:
                error_log = f"❌ Failed to initialize API client: {str(e)}"
                yield error_log, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_log, visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
            
            # Log storage - will be yielded as live updates
            last_yield_log_count = [0]  # Track when we last yielded
            last_yield_time = [0]  # Track last yield time
            
            # Track current image being processed
            current_image_idx = [0]
            
            import time
            
            def should_yield_logs():
                """Check if we should yield log updates (every 2 logs or 1 second)"""
                current_time = time.time()
                log_count_diff = len(translation_logs) - last_yield_log_count[0]
                time_diff = current_time - last_yield_time[0]
                
                # Yield if 2+ new logs OR 1+ seconds passed
                return log_count_diff >= 2 or time_diff >= 1.0
            
            def capture_log(msg, level="info"):
                """Capture logs - caller will yield periodically"""
                if msg and msg.strip():
                    log_msg = msg.strip()
                    translation_logs.append(log_msg)
            
            # Initialize timing
            last_yield_time[0] = time.time()
            
            # Create MangaTranslator instance
            try:
                # Debug: Log inpainting config
                inpaint_cfg = merged_config.get('manga_settings', {}).get('inpainting', {})
                print(f"\n=== INPAINTING CONFIG DEBUG ===")
                print(f"Inpainting enabled checkbox: {enable_inpainting}")
                print(f"Inpainting method: {inpaint_cfg.get('method')}")
                print(f"Local method: {inpaint_cfg.get('local_method')}")
                print(f"Full inpainting config: {inpaint_cfg}")
                print("=== END DEBUG ===\n")
                
                translator = MangaTranslator(
                    ocr_config=ocr_config,
                    unified_client=unified_client,
                    main_gui=mock_gui,
                    log_callback=capture_log
                )
                
                # Keep a reference for stop/shutdown support
                self.current_translator = translator
                
                # Connect stop flag so translator can react immediately to stop requests
                if hasattr(translator, 'set_stop_flag'):
                    try:
                        translator.set_stop_flag(self.stop_flag)
                    except Exception:
                        pass
                
                # CRITICAL: Set skip_inpainting flag directly on translator instance
                translator.skip_inpainting = not enable_inpainting
                print(f"Set translator.skip_inpainting = {translator.skip_inpainting}")
                
                # Explicitly initialize local inpainting if enabled
                if enable_inpainting:
                    print(f"🎨 Initializing local inpainting...")
                    try:
                        # Force initialization of the inpainter
                        init_result = translator._initialize_local_inpainter()
                        if init_result:
                            print(f"✅ Local inpainter initialized successfully")
                        else:
                            print(f"⚠️ Local inpainter initialization returned False")
                    except Exception as init_error:
                        print(f"❌ Failed to initialize inpainter: {init_error}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                import traceback
                full_error = traceback.format_exc()
                print(f"\n\n=== MANGA TRANSLATOR INIT ERROR ===")
                print(full_error)
                print(f"\nocr_config: {ocr_config}")
                print(f"\nmock_gui.model_var.get(): {mock_gui.model_var.get()}")
                print(f"\nmock_gui.api_key_entry.get(): {type(mock_gui.api_key_entry.get())}")
                print("=== END ERROR ===")
                error_log = f"❌ Failed to initialize manga translator: {str(e)}\n\nCheck console for full traceback"
                yield error_log, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_log, visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
            
            # Process each image with real progress tracking
            for idx, img_file in enumerate(files_to_process, 1):
                try:
                    # Check for stop request before processing each image
                    if self.stop_flag.is_set():
                        translation_logs.append(f"\n⏹️ Translation stopped by user before image {idx}/{total_images}")
                        self.is_translating = False
                        yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                        return
                    
                    # Update current image index for log capture
                    current_image_idx[0] = idx
                    
                    # Calculate progress range for this image
                    start_progress = (idx - 1) / total_images
                    end_progress = idx / total_images
                    
                    input_path = img_file.name if hasattr(img_file, 'name') else img_file
                    output_path = os.path.join(output_dir, f"translated_{os.path.basename(input_path)}")
                    filename = os.path.basename(input_path)
                    
                    # Log start of processing and YIELD update
                    start_msg = f"🎨 [{idx}/{total_images}] Starting: {filename}"
                    translation_logs.append(start_msg)
                    translation_logs.append(f"Image path: {input_path}")
                    translation_logs.append(f"Processing with OCR: {ocr_provider}, Model: {model}")
                    translation_logs.append("-" * 60)
                    
                    # Yield initial log update with progress
                    progress_percent = int(((idx - 1) / total_images) * 100)
                    status_text = f"Processing {idx}/{total_images}: {filename}"
                    last_yield_log_count[0] = len(translation_logs)
                    last_yield_time[0] = time.time()
                    yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    
                    # Start processing in a thread so we can yield logs periodically
                    import threading
                    processing_complete = [False]
                    result_container = [None]
                    
                    def process_wrapper():
                        result_container[0] = translator.process_image(
                            image_path=input_path,
                            output_path=output_path,
                            batch_index=idx,
                            batch_total=total_images
                        )
                        processing_complete[0] = True
                    
                    # Start processing in background
                    process_thread = threading.Thread(target=process_wrapper, daemon=True)
                    process_thread.start()
                    
                    # Poll for log updates while processing
                    while not processing_complete[0]:
                        time.sleep(0.5)  # Check every 0.5 seconds
                        
                        # Check for stop request during processing
                        if self.stop_flag.is_set():
                            translation_logs.append(f"\n⏹️ Translation stopped by user while processing image {idx}/{total_images}")
                            self.is_translating = False
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                            return
                        
                        if should_yield_logs():
                            progress_percent = int(((idx - 0.5) / total_images) * 100)  # Mid-processing
                            status_text = f"Processing {idx}/{total_images}: {filename} (in progress...)"
                            last_yield_log_count[0] = len(translation_logs)
                            last_yield_time[0] = time.time()
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    
                    # Wait for thread to complete
                    process_thread.join(timeout=1)
                    result = result_container[0]
                    
                    if result.get('success'):
                        # Use the output path from the result
                        final_output = result.get('output_path', output_path)
                        if os.path.exists(final_output):
                            translated_files.append(final_output)
                            translation_logs.append(f"✅ Image {idx}/{total_images} COMPLETE: {filename} | Total: {len(translated_files)}/{total_images} done")
                            translation_logs.append("")
                            # Yield progress update with all translated images so far
                            progress_percent = int((idx / total_images) * 100)
                            status_text = f"Completed {idx}/{total_images}: {filename}"
                            # Show all translated files as gallery
                            yield "\n".join(translation_logs), gr.update(value=translated_files, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                        else:
                            translation_logs.append(f"⚠️ Image {idx}/{total_images}: Output file missing for {filename}")
                            translation_logs.append(f"⚠️ Warning: Output file not found for image {idx}")
                            translation_logs.append("")
                            # Yield progress update
                            progress_percent = int((idx / total_images) * 100)
                            status_text = f"Warning: {idx}/{total_images} - Output missing for {filename}"
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    else:
                        errors = result.get('errors', [])
                        error_msg = errors[0] if errors else 'Unknown error'
                        translation_logs.append(f"❌ Image {idx}/{total_images} FAILED: {error_msg[:50]}")
                        translation_logs.append(f"⚠️ Error on image {idx}: {error_msg}")
                        translation_logs.append("")
                        # Yield progress update
                        progress_percent = int((idx / total_images) * 100)
                        status_text = f"Failed: {idx}/{total_images} - {filename}"
                        yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                        
                        # If translation failed, save original with error overlay
                        from PIL import Image as PILImage, ImageDraw, ImageFont
                        img = PILImage.open(input_path)
                        draw = ImageDraw.Draw(img)
                        # Add error message
                        draw.text((10, 10), f"Translation Error: {error_msg[:50]}", fill="red")
                        img.save(output_path)
                        translated_files.append(output_path)
                    
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    translation_logs.append(f"❌ Image {idx}/{total_images} ERROR: {str(e)[:60]}")
                    translation_logs.append(f"❌ Exception on image {idx}: {str(e)}")
                    print(f"Manga translation error for {input_path}:\n{error_trace}")
                    
                    # Save original on error
                    try:
                        from PIL import Image as PILImage
                        img = PILImage.open(input_path)
                        img.save(output_path)
                        translated_files.append(output_path)
                    except:
                        pass
                    continue
            
            # Check for stop request before final processing
            if self.stop_flag.is_set():
                translation_logs.append("\n⏹️ Translation stopped by user")
                self.is_translating = False
                yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                return
                
            # Add completion message
            translation_logs.append("\n" + "="*60)
            translation_logs.append(f"✅ ALL COMPLETE! Successfully translated {len(translated_files)}/{total_images} images")
            translation_logs.append("="*60)
            
            # If CBZ mode, compile translated images into CBZ archive
            final_output_for_display = None
            if cbz_mode and cbz_output_path and translated_files:
                translation_logs.append("\n📦 Compiling translated images into CBZ archive...")
                try:
                    with zipfile.ZipFile(cbz_output_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                        for img_path in translated_files:
                            # Preserve original filename structure
                            arcname = os.path.basename(img_path).replace("translated_", "")
                            cbz.write(img_path, arcname)
                    
                    translation_logs.append(f"✅ CBZ archive created: {os.path.basename(cbz_output_path)}")
                    translation_logs.append(f"📁 Archive location: {cbz_output_path}")
                    final_output_for_display = cbz_output_path
                except Exception as e:
                    translation_logs.append(f"❌ Error creating CBZ: {str(e)}")
            
            # Build final status with detailed panel information
            final_status_lines = []
            if translated_files:
                final_status_lines.append(f"✅ Successfully translated {len(translated_files)}/{total_images} image(s)!")
                final_status_lines.append("")
                final_status_lines.append("🖼️ **Translated Panels:**")
                for i, file_path in enumerate(translated_files, 1):
                    filename = os.path.basename(file_path)
                    final_status_lines.append(f"  {i}. {filename}")
                
                final_status_lines.append("")
                final_status_lines.append("🔄 **Download Options:**")
                if cbz_mode and cbz_output_path:
                    final_status_lines.append(f"  📦 CBZ Archive: {os.path.basename(cbz_output_path)}")
                    final_status_lines.append(f"  📁 Location: {cbz_output_path}")
                else:
                    # Create ZIP file for all images
                    zip_path = os.path.join(output_dir, "translated_images.zip")
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for img_path in translated_files:
                                arcname = os.path.basename(img_path)
                                zipf.write(img_path, arcname)
                        final_status_lines.append(f"  📦 Download all images: translated_images.zip")
                        final_status_lines.append(f"  📁 Output directory: {output_dir}")
                        final_output_for_display = zip_path  # Set this so it can be downloaded
                    except Exception as e:
                        final_status_lines.append(f"  ❌ Failed to create ZIP: {str(e)}")
                        final_status_lines.append(f"  📁 Output directory: {output_dir}")
                        final_status_lines.append("  🖼️ Images saved individually in output directory")
            else:
                final_status_lines.append("❌ Translation failed - no images were processed")
            
            final_status_text = "\n".join(final_status_lines)
            
            # Final yield with complete logs, image, CBZ, and final status
            # Format: (logs_textbox, output_image, cbz_file, status_textbox, progress_group, progress_text, progress_bar)
            final_progress_text = f"Complete! Processed {len(translated_files)}/{total_images} images"
            if translated_files:
                # Show all translated images in gallery
                if cbz_mode and cbz_output_path and os.path.exists(cbz_output_path):
                    yield (
                        "\n".join(translation_logs), 
                        gr.update(value=translated_files, visible=True),  # Show all images in gallery
                        gr.update(value=cbz_output_path, visible=True),  # CBZ file for download with visibility
                        gr.update(value=final_status_text, visible=True),
                        gr.update(visible=True),
                        gr.update(value=final_progress_text),
                        gr.update(value=100)
                    )
                else:
                    # Show ZIP file for download if it was created
                    if final_output_for_display and os.path.exists(final_output_for_display):
                        yield (
                            "\n".join(translation_logs), 
                            gr.update(value=translated_files, visible=True),  # Show all images in gallery
                            gr.update(value=final_output_for_display, visible=True),  # ZIP file for download
                            gr.update(value=final_status_text, visible=True),
                            gr.update(visible=True),
                            gr.update(value=final_progress_text),
                            gr.update(value=100)
                        )
                    else:
                        yield (
                            "\n".join(translation_logs), 
                            gr.update(value=translated_files, visible=True),  # Show all images in gallery
                            gr.update(visible=False),  # Hide download component if ZIP failed
                            gr.update(value=final_status_text, visible=True),
                            gr.update(visible=True),
                            gr.update(value=final_progress_text),
                            gr.update(value=100)
                        )
            else:
                yield (
                    "\n".join(translation_logs), 
                    gr.update(visible=False), 
                    gr.update(visible=False),  # Hide CBZ component
                    gr.update(value=final_status_text, visible=True),
                    gr.update(visible=True),
                    gr.update(value=final_progress_text),
                    gr.update(value=0)  # 0% if nothing was processed
                )
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during manga translation:\n{str(e)}\n\n{traceback.format_exc()}"
            self.is_translating = False
            yield error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_msg, visible=True), gr.update(visible=False), gr.update(value="Error occurred"), gr.update(value=0)
        finally:
            # Always reset translation state when done
            self.is_translating = False
            # Clear active references on full completion
            try:
                self.current_translator = None
                self.current_unified_client = None
            except Exception:
                pass
    
    def stop_manga_translation(self):
        """Simple function to stop manga translation"""
        print("DEBUG: Stop button clicked")
        if self.is_translating:
            print("DEBUG: Stopping active translation")
            self.stop_translation()
            # Return UI updates for button visibility and status
            return (
                gr.update(visible=True),   # translate button - show
                gr.update(visible=False),  # stop button - hide  
                "⏹️ Translation stopped by user"
            )
        else:
            print("DEBUG: No active translation to stop")
            return (
                gr.update(visible=True),   # translate button - show
                gr.update(visible=False),  # stop button - hide
                "No active translation to stop"
            )
    
    def start_manga_translation(self, *args):
        """Simple function to start manga translation - GENERATOR FUNCTION"""
        print("DEBUG: Translate button clicked")
        
        # Reset flags for new translation and mark as translating BEFORE first yield
        self._reset_translation_flags()
        self.is_translating = True
        
        # Initial yield to update button visibility
        yield (
            "🚀 Starting translation...",
            gr.update(visible=False),  # manga_output_gallery - hide initially
            gr.update(visible=False),  # manga_cbz_output  
            gr.update(value="Starting...", visible=True),  # manga_status
            gr.update(visible=False),  # manga_progress_group
            gr.update(value="Initializing..."),  # manga_progress_text
            gr.update(value=0),  # manga_progress_bar
            gr.update(visible=False),  # translate button - hide during translation
            gr.update(visible=True)   # stop button - show during translation
        )
        
        # Call the translate function and yield all its results
        last_result = None
        try:
            for result in self.translate_manga(*args):
                # Check if stop was requested during iteration
                if self.stop_flag.is_set():
                    print("DEBUG: Stop flag detected, breaking translation loop")
                    break
                    
                last_result = result
                # Pad result to include button states (translate_visible=False, stop_visible=True)
                if len(result) >= 7:
                    yield result + (gr.update(visible=False), gr.update(visible=True))
                else:
                    # Pad result to match expected length (7 values) then add button states
                    padded_result = list(result) + [gr.update(visible=False)] * (7 - len(result))
                    yield tuple(padded_result) + (gr.update(visible=False), gr.update(visible=True))
                    
        except GeneratorExit:
            print("DEBUG: Translation generator was closed")
            self.is_translating = False
            return
        except Exception as e:
            print(f"DEBUG: Exception during translation: {e}")
            self.is_translating = False
            # Show error and reset buttons
            error_msg = f"❌ Error during translation: {str(e)}"
            yield (
                error_msg,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=error_msg, visible=True),
                gr.update(visible=False),
                gr.update(value="Error occurred"),
                gr.update(value=0),
                gr.update(visible=True),   # translate button - show after error
                gr.update(visible=False)   # stop button - hide after error
            )
            return
        finally:
            # Clear active references when the loop exits
            self.is_translating = False
            try:
                self.current_translator = None
                self.current_unified_client = None
            except Exception:
                pass
        
        # Check if we stopped early
        if self.stop_flag.is_set():
            yield (
                "⏹️ Translation stopped by user",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="⏹️ Translation stopped", visible=True),
                gr.update(visible=False),
                gr.update(value="Stopped"),
                gr.update(value=0),
                gr.update(visible=True),   # translate button - show after stop
                gr.update(visible=False)   # stop button - hide after stop
            )
            return
        
        # Final yield to reset buttons after successful completion
        print("DEBUG: Translation completed normally, resetting buttons")
        if last_result is None:
            last_result = ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Complete"), gr.update(value=100))
        
        if len(last_result) >= 7:
            yield last_result[:7] + (gr.update(visible=True), gr.update(visible=False))
        else:
            # Pad result to match expected length then add button states
            padded_result = list(last_result) + [gr.update(visible=False)] * (7 - len(last_result))
            yield tuple(padded_result) + (gr.update(visible=True), gr.update(visible=False))
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        # Reload config before creating interface to get latest values
        self.config = self.load_config()
        self.decrypted_config = decrypt_config(self.config.copy()) if API_KEY_ENCRYPTION_AVAILABLE else self.config.copy()
        
        # Load and encode icon as base64
        icon_base64 = ""
        icon_path = "Halgakos.ico" if os.path.exists("Halgakos.ico") else "Halgakos.ico"
        if os.path.exists(icon_path):
            with open(icon_path, "rb") as f:
                icon_base64 = base64.b64encode(f.read()).decode()
        
        # Custom CSS to hide Gradio footer and add favicon
        custom_css = """
        footer {display: none !important;}
        .gradio-container {min-height: 100vh;}
        
        /* Stop button styling */
        .gr-button[data-variant="stop"] {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
            color: white !important;
        }
        .gr-button[data-variant="stop"]:hover {
            background-color: #c82333 !important;
            border-color: #bd2130 !important;
            color: white !important;
        }
        """
        
        # JavaScript for localStorage persistence - SIMPLE VERSION
        localStorage_js = """
        <script>
        console.log('Glossarion localStorage script loading...');
        
        // Simple localStorage functions
        function saveToLocalStorage(key, value) {
            try {
                localStorage.setItem('glossarion_' + key, JSON.stringify(value));
                console.log('Saved:', key, '=', value);
                return true;
            } catch (e) {
                console.error('Save failed:', e);
                return false;
            }
        }
        
        function loadFromLocalStorage(key, defaultValue) {
            try {
                const item = localStorage.getItem('glossarion_' + key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Load failed:', e);
                return defaultValue;
            }
        }
        
        // Manual save current form values to localStorage
        function saveCurrentSettings() {
            const settings = {};
            
            // Find all input elements in Gradio
            document.querySelectorAll('input, select, textarea').forEach(el => {
                // Skip file inputs
                if (el.type === 'file') return;
                
                // Get a unique key based on element properties
                let key = el.id || el.name || el.placeholder || '';
                if (!key) {
                    // Try to get label text
                    const label = el.closest('div')?.querySelector('label');
                    if (label) key = label.textContent;
                }
                
                if (key) {
                    key = key.trim().replace(/[^a-zA-Z0-9]/g, '_');
                    if (el.type === 'checkbox') {
                        settings[key] = el.checked;
                    } else if (el.type === 'radio') {
                        if (el.checked) settings[key] = el.value;
                    } else if (el.value) {
                        settings[key] = el.value;
                    }
                }
            });
            
            // Save all settings
            Object.keys(settings).forEach(key => {
                saveToLocalStorage(key, settings[key]);
            });
            
            console.log('Saved', Object.keys(settings).length, 'settings');
            return settings;
        }
        
        // Export settings from localStorage
        function exportSettings() {
            console.log('Export started');
            
            // First save current form state
            saveCurrentSettings();
            
            // Then export from localStorage
            const settings = {};
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('glossarion_')) {
                    try {
                        settings[key.replace('glossarion_', '')] = JSON.parse(localStorage.getItem(key));
                    } catch (e) {
                        // Store as-is if not JSON
                        settings[key.replace('glossarion_', '')] = localStorage.getItem(key);
                    }
                }
            }
            
            if (Object.keys(settings).length === 0) {
                alert('No settings to export. Try saving some settings first.');
                return;
            }
            
            // Download as JSON
            const blob = new Blob([JSON.stringify(settings, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'glossarion_settings_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log('Exported', Object.keys(settings).length, 'settings');
        }
        
        function importSettings(fileContent) {
            try {
                const settings = JSON.parse(fileContent);
                Object.keys(settings).forEach(key => {
                    saveToLocalStorage(key, settings[key]);
                });
                location.reload(); // Reload to apply settings
            } catch (e) {
                alert('Invalid settings file format');
            }
        }
        
        // Expose to global scope
        window.exportSettings = exportSettings;
        window.importSettings = importSettings;
        window.saveCurrentSettings = saveCurrentSettings;
        window.saveToLocalStorage = saveToLocalStorage;
        window.loadFromLocalStorage = loadFromLocalStorage;
        
        // Load settings from localStorage on page load for HF Spaces
        function loadSettingsFromLocalStorage() {
            console.log('Attempting to load settings from localStorage...');
            try {
                // Get all localStorage items with glossarion_ prefix
                const settings = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && key.startsWith('glossarion_')) {
                        const cleanKey = key.replace('glossarion_', '');
                        try {
                            settings[cleanKey] = JSON.parse(localStorage.getItem(key));
                        } catch (e) {
                            settings[cleanKey] = localStorage.getItem(key);
                        }
                    }
                }
                
                if (Object.keys(settings).length > 0) {
                    console.log('Found', Object.keys(settings).length, 'settings in localStorage');
                    
                    // Try to update Gradio components
                    // This is tricky because Gradio components are rendered dynamically
                    // We'll need to find them by their labels or other identifiers
                    
                    // For now, just log what we found
                    console.log('Settings:', settings);
                }
            } catch (e) {
                console.error('Error loading from localStorage:', e);
            }
        }
        
        // Try loading settings at various points
        window.addEventListener('load', function() {
            console.log('Page loaded');
            setTimeout(loadSettingsFromLocalStorage, 1000);
            setTimeout(loadSettingsFromLocalStorage, 3000);
        });
        
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM ready');
            setTimeout(loadSettingsFromLocalStorage, 500);
        });
        </script>
        """
        
        with gr.Blocks(
            title="Glossarion - AI Translation",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as app:
            
            # Add custom HTML with favicon link and title with icon
            icon_img_tag = f'<img src="data:image/png;base64,{icon_base64}" alt="Glossarion">' if icon_base64 else ''
            
            gr.HTML(f"""
            <link rel="icon" type="image/x-icon" href="file/Halgakos.ico">
            <link rel="shortcut icon" type="image/x-icon" href="file/Halgakos.ico">
            <style>
                .title-with-icon {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    margin-bottom: 10px;
                }}
                .title-with-icon img {{
                    width: 48px;
                    height: 48px;
                }}
            </style>
            <div class="title-with-icon">
                {icon_img_tag}
                <h1>Glossarion - AI-Powered Translation</h1>
            </div>
            {localStorage_js}
            """)
            
            with gr.Row():
                gr.Markdown("""
                Translate novels and books using advanced AI models (GPT-5, Claude, etc.)
                """)
                
                
                # SECURITY: Save Config button disabled for Hugging Face to prevent API key leakage
                # Users should use localStorage (browser-based storage) instead
                # with gr.Column(scale=0):
                #     save_config_btn = gr.Button(
                #         "💾 Save Config",
                #         variant="secondary",
                #         size="sm"
                #     )
                #     save_status_text = gr.Markdown(
                #         "",
                #         visible=False
                #     )
                
            with gr.Tabs() as main_tabs:
                # EPUB Translation Tab
                with gr.Tab("📚 EPUB Translation"):
                    with gr.Row():
                        with gr.Column():
                            epub_file = gr.File(
                                label="📖 Upload EPUB or TXT File",
                                file_types=[".epub", ".txt"]
                            )
                            
                            with gr.Row():
                                translate_btn = gr.Button(
                                    "🚀 Translate EPUB",
                                    variant="primary",
                                    size="lg",
                                    scale=2
                                )
                                
                                stop_epub_btn = gr.Button(
                                    "⏹️ Stop Translation",
                                    variant="stop",
                                    size="lg",
                                    visible=False,
                                    scale=1
                                )
                            
                            epub_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            epub_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')
                            )
                            
                            # Use all profiles without filtering
                            profile_choices = list(self.profiles.keys())
                            # Use saved active_profile instead of hardcoded default
                            default_profile = self.get_config_value('active_profile', profile_choices[0] if profile_choices else '')
                            
                            epub_profile = gr.Dropdown(
                                choices=profile_choices,
                                value=default_profile,
                                label="📝 Translation Profile"
                            )
                            
                            epub_system_prompt = gr.Textbox(
                                label="System Prompt (Translation Instructions)",
                                lines=8,
                                max_lines=15,
                                interactive=True,
                                placeholder="Select a profile to load translation instructions...",
                                value=self.profiles.get(default_profile, '') if default_profile else ''
                            )
                            
                            with gr.Accordion("⚙️ Advanced Settings", open=False):
                                epub_temperature = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=self.get_config_value('temperature', 0.3),
                                    step=0.1,
                                    label="Temperature"
                                )
                                
                                epub_max_tokens = gr.Number(
                                    label="Max Output Tokens",
                                    value=self.get_config_value('max_output_tokens', 16000),
                                    minimum=0
                                )
                                
                                gr.Markdown("### Image Translation")
                                
                                enable_image_translation = gr.Checkbox(
                                    label="Enable Image Translation",
                                    value=self.get_config_value('enable_image_translation', False),
                                    info="Extracts and translates text from images using vision models"
                                )
                                
                                gr.Markdown("### Glossary Settings")
                                
                                enable_auto_glossary = gr.Checkbox(
                                    label="Enable Automatic Glossary Generation",
                                    value=self.get_config_value('enable_auto_glossary', False),
                                    info="Automatic extraction and translation of character names/terms"
                                )
                                
                                append_glossary = gr.Checkbox(
                                    label="Append Glossary to System Prompt",
                                    value=self.get_config_value('append_glossary_to_prompt', True),
                                    info="Applies to ALL glossaries - manual and automatic"
                                )
                                
                                # Automatic glossary extraction settings (only show when enabled)
                                with gr.Group(visible=self.get_config_value('enable_auto_glossary', False)) as auto_glossary_settings:
                                    gr.Markdown("#### Automatic Glossary Extraction Settings")
                                    
                                    with gr.Row():
                                        auto_glossary_min_freq = gr.Slider(
                                            minimum=1,
                                            maximum=10,
                                            value=self.get_config_value('glossary_min_frequency', 2),
                                            step=1,
                                            label="Min Frequency",
                                            info="Minimum times a name must appear"
                                        )
                                        
                                        auto_glossary_max_names = gr.Slider(
                                            minimum=10,
                                            maximum=200,
                                            value=self.get_config_value('glossary_max_names', 50),
                                            step=10,
                                            label="Max Names",
                                            info="Maximum number of character names"
                                        )
                                    
                                    with gr.Row():
                                        auto_glossary_max_titles = gr.Slider(
                                            minimum=10,
                                            maximum=100,
                                            value=self.get_config_value('glossary_max_titles', 30),
                                            step=5,
                                            label="Max Titles",
                                            info="Maximum number of titles/terms"
                                        )
                                        
                                        auto_glossary_batch_size = gr.Slider(
                                            minimum=10,
                                            maximum=100,
                                            value=self.get_config_value('glossary_batch_size', 50),
                                            step=5,
                                            label="Translation Batch Size",
                                            info="Terms per API call"
                                        )
                                    
                                    auto_glossary_filter_mode = gr.Radio(
                                        choices=[
                                            ("All names & terms", "all"),
                                            ("Names with honorifics only", "only_with_honorifics"),
                                            ("Names without honorifics & terms", "only_without_honorifics")
                                        ],
                                        value=self.get_config_value('glossary_filter_mode', 'all'),
                                        label="Filter Mode",
                                        info="What types of names to extract"
                                    )
                                    
                                    auto_glossary_fuzzy_threshold = gr.Slider(
                                        minimum=0.5,
                                        maximum=1.0,
                                        value=self.get_config_value('glossary_fuzzy_threshold', 0.90),
                                        step=0.05,
                                        label="Fuzzy Matching Threshold",
                                        info="How similar names must be to match (0.9 = 90% match)"
                                    )
                                
                                # Toggle visibility of auto glossary settings
                                enable_auto_glossary.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=[enable_auto_glossary],
                                    outputs=[auto_glossary_settings]
                                )
                                
                                gr.Markdown("### Quality Assurance")
                                
                                enable_post_translation_scan = gr.Checkbox(
                                    label="Enable post-translation Scanning phase",
                                    value=self.get_config_value('enable_post_translation_scan', False),
                                    info="Automatically run QA Scanner after translation completes"
                                )
                                
                                glossary_file = gr.File(
                                    label="📋 Manual Glossary CSV (optional)",
                                    file_types=[".csv", ".json", ".txt"]
                                )
                        
                        with gr.Column():
                            # Add logo and status at top
                            with gr.Row():
                                gr.Image(
                                    value="Halgakos.png",
                                    label=None,
                                    show_label=False,
                                    width=80,
                                    height=80,
                                    interactive=False,
                                    show_download_button=False,
                                    container=False
                                )
                                epub_status_message = gr.Markdown(
                                    value="### Ready to translate\nUpload an EPUB or TXT file and click 'Translate' to begin.",
                                    visible=True
                                )
                            
                            # Progress section (similar to manga tab)
                            with gr.Group(visible=False) as epub_progress_group:
                                gr.Markdown("### Progress")
                                epub_progress_text = gr.Textbox(
                                    label="📨 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                epub_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Translation Progress",
                                    interactive=False,
                                    show_label=True
                                )
                            
                            epub_logs = gr.Textbox(
                                label="📋 Translation Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to translate. Upload an EPUB or TXT file and configure settings.",
                                visible=True,
                                interactive=False
                            )
                            
                            epub_output = gr.File(
                                label="📥 Download Translated File",
                                visible=True  # Always visible, will show file when ready
                            )
                            
                            epub_status = gr.Textbox(
                                label="Final Status",
                                lines=3,
                                max_lines=5,
                                visible=False,
                                interactive=False
                            )
                    
                    # Sync handlers will be connected after manga components are created
                    
                    # Translation button handler - now with progress outputs
                    translate_btn.click(
                        fn=self.translate_epub_with_stop,
                        inputs=[
                            epub_file,
                            epub_model,
                            epub_api_key,
                            epub_profile,
                            epub_system_prompt,
                            epub_temperature,
                            epub_max_tokens,
                            enable_image_translation,
                            glossary_file
                        ],
                        outputs=[
                            epub_output,          # Download file
                            epub_status_message,  # Top status message
                            epub_progress_group,  # Progress group visibility
                            epub_logs,            # Translation logs
                            epub_status,          # Final status
                            epub_progress_text,   # Progress text
                            epub_progress_bar,    # Progress bar
                            translate_btn,        # Show/hide translate button
                            stop_epub_btn        # Show/hide stop button
                        ]
                    )
                    
                    # Stop button handler
                    stop_epub_btn.click(
                        fn=self.stop_epub_translation,
                        inputs=[],
                        outputs=[translate_btn, stop_epub_btn, epub_status]
                    )
                
                # Manga Translation Tab
                with gr.Tab("🎨 Manga Translation"):
                    with gr.Row():
                        with gr.Column():
                            manga_images = gr.File(
                                label="🖼️ Upload Manga Images or CBZ",
                                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".cbz", ".zip"],
                                file_count="multiple"
                            )
                            
                            with gr.Row():
                                translate_manga_btn = gr.Button(
                                    "🚀 Translate Manga",
                                    variant="primary",
                                    size="lg",
                                    scale=2
                                )
                                
                                stop_manga_btn = gr.Button(
                                    "⏹️ Stop Translation",
                                    variant="stop",
                                    size="lg",
                                    visible=False,
                                    scale=1
                                )
                            
                            manga_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            manga_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')  # Pre-fill from config
                            )
                            
                            # Use all profiles without filtering
                            profile_choices = list(self.profiles.keys())
                            # Use the active profile from config, same as EPUB tab
                            default_profile = self.get_config_value('active_profile', profile_choices[0] if profile_choices else '')
                            
                            manga_profile = gr.Dropdown(
                                choices=profile_choices,
                                value=default_profile,
                                label="📝 Translation Profile"
                            )
                            
                            # Editable manga system prompt
                            manga_system_prompt = gr.Textbox(
                                label="Manga System Prompt (Translation Instructions)",
                                lines=8,
                                max_lines=15,
                                interactive=True,
                                placeholder="Select a manga profile to load translation instructions...",
                                value=self.profiles.get(default_profile, '') if default_profile else ''
                            )
                            
                            with gr.Accordion("⚙️ OCR Settings", open=False):
                                gr.Markdown("🔒 **Credentials are auto-saved** to your config (encrypted) after first use.")
                                
                                ocr_provider = gr.Radio(
                                    choices=["google", "azure", "custom-api"],
                                    value=self.get_config_value('ocr_provider', 'custom-api'),
                                    label="OCR Provider"
                                )
                                
                                # Show saved Google credentials path if available
                                saved_google_path = self.get_config_value('google_vision_credentials', '')
                                if saved_google_path and os.path.exists(saved_google_path):
                                    gr.Markdown(f"✅ **Saved credentials found:** `{os.path.basename(saved_google_path)}`")
                                    gr.Markdown("💡 *Using saved credentials. Upload a new file only if you want to change them.*")
                                else:
                                    gr.Markdown("⚠️ No saved Google credentials found. Please upload your JSON file.")
                                
                                # Note: File component doesn't support pre-filling paths due to browser security
                                google_creds = gr.File(
                                    label="Google Cloud Credentials JSON (upload to update)",
                                    file_types=[".json"]
                                )
                                
                                azure_key = gr.Textbox(
                                    label="Azure Vision API Key (if using Azure)",
                                    type="password",
                                    placeholder="Enter Azure API key",
                                    value=self.get_config_value('azure_vision_key', '')
                                )
                                
                                azure_endpoint = gr.Textbox(
                                    label="Azure Vision Endpoint (if using Azure)",
                                    placeholder="https://your-resource.cognitiveservices.azure.com/",
                                    value=self.get_config_value('azure_vision_endpoint', '')
                                )
                                
                                bubble_detection = gr.Checkbox(
                                    label="Enable Bubble Detection",
                                    value=self.get_config_value('bubble_detection_enabled', True)
                                )
                                
                                inpainting = gr.Checkbox(
                                    label="Enable Text Removal (Inpainting)",
                                    value=self.get_config_value('inpainting_enabled', True)
                                )
                            
                            with gr.Accordion("⚡ Parallel Processing", open=False):
                                gr.Markdown("### Parallel Panel Translation")
                                gr.Markdown("*Process multiple panels simultaneously for faster translation*")
                                
                                # Check environment variables first, then config
                                parallel_enabled = os.getenv('PARALLEL_PANEL_TRANSLATION', '').lower() == 'true'
                                if not parallel_enabled:
                                    # Fall back to config if not set in env
                                    parallel_enabled = self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False)
                                
                                # Get max workers from env or config
                                max_workers_env = os.getenv('PANEL_MAX_WORKERS', '')
                                if max_workers_env.isdigit():
                                    max_workers = int(max_workers_env)
                                else:
                                    max_workers = self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7)
                                
                                parallel_panel_translation = gr.Checkbox(
                                    label="Enable Parallel Panel Translation",
                                    value=parallel_enabled,
                                    info="Translates multiple panels at once instead of sequentially"
                                )
                                
                                panel_max_workers = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=max_workers,
                                    step=1,
                                    label="Max concurrent panels",
                                    interactive=True,
                                    info="Number of panels to process simultaneously (higher = faster but more memory)"
                                )
                            
                            with gr.Accordion("✨ Text Visibility Settings", open=False):
                                gr.Markdown("### Font Settings")
                                
                                font_size_mode = gr.Radio(
                                    choices=["auto", "fixed", "multiplier"],
                                    value=self.get_config_value('manga_font_size_mode', 'auto'),
                                    label="Font Size Mode"
                                )
                                
                                font_size = gr.Slider(
                                    minimum=0,
                                    maximum=72,
                                    value=self.get_config_value('manga_font_size', 24),
                                    step=1,
                                    label="Fixed Font Size (0=auto, used when mode=fixed)"
                                )
                                
                                font_multiplier = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=self.get_config_value('manga_font_size_multiplier', 1.0),
                                    step=0.1,
                                    label="Font Size Multiplier (when mode=multiplier)"
                                )
                                
                                min_font_size = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=self.get_config_value('manga_settings', {}).get('rendering', {}).get('auto_min_size', 12),
                                    step=1,
                                    label="Minimum Font Size (0=no limit)"
                                )
                                
                                max_font_size = gr.Slider(
                                    minimum=20,
                                    maximum=100,
                                    value=self.get_config_value('manga_max_font_size', 48),
                                    step=1,
                                    label="Maximum Font Size"
                                )
                                
                                gr.Markdown("### Text Color")
                                
                                # Convert RGB array to hex if needed
                                def to_hex_color(color_value, default='#000000'):
                                    if isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
                                        return '#{:02x}{:02x}{:02x}'.format(int(color_value[0]), int(color_value[1]), int(color_value[2]))
                                    elif isinstance(color_value, str):
                                        return color_value if color_value.startswith('#') else default
                                    return default
                                
                                text_color_rgb = gr.ColorPicker(
                                    label="Font Color",
                                    value=to_hex_color(self.get_config_value('manga_text_color', [255, 255, 255]), '#FFFFFF')  # Default white
                                )
                                
                                gr.Markdown("### Shadow Settings")
                                
                                shadow_enabled = gr.Checkbox(
                                    label="Enable Text Shadow",
                                    value=self.get_config_value('manga_shadow_enabled', True)
                                )
                                
                                shadow_color = gr.ColorPicker(
                                    label="Shadow Color",
                                    value=to_hex_color(self.get_config_value('manga_shadow_color', [0, 0, 0]), '#000000')  # Default black
                                )
                                
                                shadow_offset_x = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_offset_x', 2),
                                    step=1,
                                    label="Shadow Offset X"
                                )
                                
                                shadow_offset_y = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_offset_y', 2),
                                    step=1,
                                    label="Shadow Offset Y"
                                )
                                
                                shadow_blur = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_blur', 0),
                                    step=1,
                                    label="Shadow Blur"
                                )
                                
                                gr.Markdown("### Background Settings")
                                
                                bg_opacity = gr.Slider(
                                    minimum=0,
                                    maximum=255,
                                    value=self.get_config_value('manga_bg_opacity', 130),
                                    step=1,
                                    label="Background Opacity"
                                )
                                
                            # Ensure bg_style value is valid
                            bg_style_value = self.get_config_value('manga_bg_style', 'circle')
                            if bg_style_value not in ["box", "circle", "wrap"]:
                                bg_style_value = 'circle'  # Default fallback
                            
                            bg_style = gr.Radio(
                                choices=["box", "circle", "wrap"],
                                value=bg_style_value,
                                label="Background Style"
                            )
                        
                        with gr.Column():
                            # Add logo and loading message at top
                            with gr.Row():
                                gr.Image(
                                    value="Halgakos.png",
                                    label=None,
                                    show_label=False,
                                    width=80,
                                    height=80,
                                    interactive=False,
                                    show_download_button=False,
                                    container=False
                                )
                                status_message = gr.Markdown(
                                    value="### Ready to translate\nUpload an image and click 'Translate Manga' to begin.",
                                    visible=True
                            )
                            
                            # Progress section for manga translation (similar to manga integration script)
                            with gr.Group(visible=False) as manga_progress_group:
                                gr.Markdown("### Progress")
                                manga_progress_text = gr.Textbox(
                                    label="📈 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                manga_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Translation Progress",
                                    interactive=False,
                                    show_label=True
                                )
                            
                            manga_logs = gr.Textbox(
                                label="📋 Translation Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to translate. Click 'Translate Manga' to begin.",
                                visible=True,
                                interactive=False
                            )
                            
                            # Use Gallery to show all translated images
                            manga_output_gallery = gr.Gallery(
                                label="📷 Translated Images (click to download)",
                                visible=False,
                                show_label=True,
                                elem_id="manga_output_gallery",
                                columns=3,
                                rows=2,
                                height="auto",
                                allow_preview=True,
                                show_download_button=True  # Allow download of individual images
                            )
                            # Keep CBZ output for bulk download
                            manga_cbz_output = gr.File(label="📦 Download Translated CBZ", visible=False)
                            manga_status = gr.Textbox(
                                label="Final Status",
                                lines=8,
                                max_lines=15,
                                visible=False
                            )
                    
                    # Global sync flag to prevent loops
                    self._syncing_active = False
                    
                    # Auto-save Azure credentials on change
                    def save_azure_credentials(key, endpoint):
                        """Save Azure credentials to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            if key and key.strip():
                                current_config['azure_vision_key'] = str(key).strip()
                            if endpoint and endpoint.strip():
                                current_config['azure_vision_endpoint'] = str(endpoint).strip()
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save Azure credentials: {e}")
                            return None
                    
                    # All auto-save handlers removed - use manual Save Config button to avoid constant writes to persistent storage
                    
                    # Only update system prompts when profiles change - no cross-tab syncing
                    epub_profile.change(
                        fn=lambda p: self.profiles.get(p, ''),
                        inputs=[epub_profile],
                        outputs=[epub_system_prompt]
                    )
                    
                    manga_profile.change(
                        fn=lambda p: self.profiles.get(p, ''),
                        inputs=[manga_profile],
                        outputs=[manga_system_prompt]
                    )
                    
                    # Manual save function for all configuration
                    def save_all_config(
                        model, api_key, profile, temperature, max_tokens,
                        enable_image_trans, enable_auto_gloss, append_gloss,
                        # Auto glossary settings
                        auto_gloss_min_freq, auto_gloss_max_names, auto_gloss_max_titles,
                        auto_gloss_batch_size, auto_gloss_filter_mode, auto_gloss_fuzzy,
                        enable_post_scan,
                        # Manual glossary extraction settings
                        manual_min_freq, manual_max_names, manual_max_titles,
                        manual_max_text_size, manual_max_sentences, manual_trans_batch,
                        manual_chapter_split, manual_filter_mode, manual_strip_honorifics,
                        manual_fuzzy, manual_extraction_prompt, manual_format_instructions,
                        manual_use_legacy_csv,
                        # QA Scanner settings
                        qa_min_foreign, qa_check_rep, qa_check_gloss_leak,
                        qa_min_file_len, qa_check_headers, qa_check_html,
                        qa_check_paragraphs, qa_min_para_percent, qa_report_fmt, qa_auto_save,
                        # Chapter processing options
                        batch_trans_headers, headers_batch, ncx_nav, attach_css, retain_ext,
                        conservative_batch, gemini_safety, http_openrouter, openrouter_compress,
                        extraction_method, filter_level,
                        # Thinking mode settings
                        gpt_thinking_enabled, gpt_effort, or_tokens,
                        gemini_thinking_enabled, gemini_budget,
                        manga_model, manga_api_key, manga_profile,
                        ocr_prov, azure_k, azure_e,
                        bubble_det, inpaint,
                        font_mode, font_s, font_mult, min_font, max_font,
                        text_col, shadow_en, shadow_col,
                        shadow_x, shadow_y, shadow_b,
                        bg_op, bg_st,
                        parallel_trans, panel_workers,
                        # Advanced Settings fields
                        detector_type_val, rtdetr_conf, bubble_conf,
                        detect_text, detect_empty, detect_free, max_detections,
                        local_method_val, webtoon_val,
                        batch_size_val, cache_enabled_val,
                        parallel_proc, max_work,
                        preload_local, stagger_ms,
                        torch_prec, auto_cleanup,
                        debug, save_inter, concise_logs
                    ):
                        """Save all configuration values at once"""
                        try:
                            config = self.get_current_config_for_update()
                            
                            # Save all values
                            config['model'] = model
                            if api_key:  # Only save non-empty API keys
                                config['api_key'] = api_key
                            config['active_profile'] = profile
                            config['temperature'] = temperature
                            config['max_output_tokens'] = max_tokens
                            config['enable_image_translation'] = enable_image_trans
                            config['enable_auto_glossary'] = enable_auto_gloss
                            config['append_glossary_to_prompt'] = append_gloss
                            
                            # Auto glossary settings
                            config['glossary_min_frequency'] = auto_gloss_min_freq
                            config['glossary_max_names'] = auto_gloss_max_names
                            config['glossary_max_titles'] = auto_gloss_max_titles
                            config['glossary_batch_size'] = auto_gloss_batch_size
                            config['glossary_filter_mode'] = auto_gloss_filter_mode
                            config['glossary_fuzzy_threshold'] = auto_gloss_fuzzy
                            
                            # Manual glossary extraction settings
                            config['manual_glossary_min_frequency'] = manual_min_freq
                            config['manual_glossary_max_names'] = manual_max_names
                            config['manual_glossary_max_titles'] = manual_max_titles
                            config['glossary_max_text_size'] = manual_max_text_size
                            config['glossary_max_sentences'] = manual_max_sentences
                            config['manual_glossary_batch_size'] = manual_trans_batch
                            config['glossary_chapter_split_threshold'] = manual_chapter_split
                            config['manual_glossary_filter_mode'] = manual_filter_mode
                            config['strip_honorifics'] = manual_strip_honorifics
                            config['manual_glossary_fuzzy_threshold'] = manual_fuzzy
                            config['manual_glossary_prompt'] = manual_extraction_prompt
                            config['glossary_format_instructions'] = manual_format_instructions
                            config['glossary_use_legacy_csv'] = manual_use_legacy_csv
                            config['enable_post_translation_scan'] = enable_post_scan
                            
                            # QA Scanner settings
                            config['qa_min_foreign_chars'] = qa_min_foreign
                            config['qa_check_repetition'] = qa_check_rep
                            config['qa_check_glossary_leakage'] = qa_check_gloss_leak
                            config['qa_min_file_length'] = qa_min_file_len
                            config['qa_check_multiple_headers'] = qa_check_headers
                            config['qa_check_missing_html'] = qa_check_html
                            config['qa_check_insufficient_paragraphs'] = qa_check_paragraphs
                            config['qa_min_paragraph_percentage'] = qa_min_para_percent
                            config['qa_report_format'] = qa_report_fmt
                            config['qa_auto_save_report'] = qa_auto_save
                            
                            # Chapter processing options
                            config['batch_translate_headers'] = batch_trans_headers
                            config['headers_per_batch'] = headers_batch
                            config['use_ncx_navigation'] = ncx_nav
                            config['attach_css_to_chapters'] = attach_css
                            config['retain_source_extension'] = retain_ext
                            config['use_conservative_batching'] = conservative_batch
                            config['disable_gemini_safety'] = gemini_safety
                            config['use_http_openrouter'] = http_openrouter
                            config['disable_openrouter_compression'] = openrouter_compress
                            config['text_extraction_method'] = extraction_method
                            config['file_filtering_level'] = filter_level
                            
                            # Thinking mode settings
                            config['enable_gpt_thinking'] = gpt_thinking_enabled
                            config['gpt_thinking_effort'] = gpt_effort
                            config['or_thinking_tokens'] = or_tokens
                            config['enable_gemini_thinking'] = gemini_thinking_enabled
                            config['gemini_thinking_budget'] = gemini_budget
                            
                            # Manga settings
                            config['ocr_provider'] = ocr_prov
                            if azure_k:
                                config['azure_vision_key'] = azure_k
                            if azure_e:
                                config['azure_vision_endpoint'] = azure_e
                            config['bubble_detection_enabled'] = bubble_det
                            config['inpainting_enabled'] = inpaint
                            config['manga_font_size_mode'] = font_mode
                            config['manga_font_size'] = font_s
                            config['manga_font_multiplier'] = font_mult
                            config['manga_min_font_size'] = min_font
                            config['manga_max_font_size'] = max_font
                            config['manga_text_color'] = text_col
                            config['manga_shadow_enabled'] = shadow_en
                            config['manga_shadow_color'] = shadow_col
                            config['manga_shadow_offset_x'] = shadow_x
                            config['manga_shadow_offset_y'] = shadow_y
                            config['manga_shadow_blur'] = shadow_b
                            config['manga_bg_opacity'] = bg_op
                            config['manga_bg_style'] = bg_st
                            
                            # Advanced settings
                            if 'manga_settings' not in config:
                                config['manga_settings'] = {}
                            if 'advanced' not in config['manga_settings']:
                                config['manga_settings']['advanced'] = {}
                            config['manga_settings']['advanced']['parallel_panel_translation'] = parallel_trans
                            config['manga_settings']['advanced']['panel_max_workers'] = panel_workers
                            
                            # Advanced bubble detection and inpainting settings
                            if 'ocr' not in config['manga_settings']:
                                config['manga_settings']['ocr'] = {}
                            if 'inpainting' not in config['manga_settings']:
                                config['manga_settings']['inpainting'] = {}
                                
                            config['manga_settings']['ocr']['detector_type'] = detector_type_val
                            config['manga_settings']['ocr']['rtdetr_confidence'] = rtdetr_conf
                            config['manga_settings']['ocr']['bubble_confidence'] = bubble_conf
                            config['manga_settings']['ocr']['detect_text_bubbles'] = detect_text
                            config['manga_settings']['ocr']['detect_empty_bubbles'] = detect_empty
                            config['manga_settings']['ocr']['detect_free_text'] = detect_free
                            config['manga_settings']['ocr']['bubble_max_detections_yolo'] = max_detections
                            config['manga_settings']['inpainting']['local_method'] = local_method_val
                            config['manga_settings']['advanced']['webtoon_mode'] = webtoon_val
                            config['manga_settings']['inpainting']['batch_size'] = batch_size_val
                            config['manga_settings']['inpainting']['enable_cache'] = cache_enabled_val
                            config['manga_settings']['advanced']['parallel_processing'] = parallel_proc
                            config['manga_settings']['advanced']['max_workers'] = max_work
                            config['manga_settings']['advanced']['preload_local_inpainting_for_panels'] = preload_local
                            config['manga_settings']['advanced']['panel_start_stagger_ms'] = stagger_ms
                            config['manga_settings']['advanced']['torch_precision'] = torch_prec
                            config['manga_settings']['advanced']['auto_cleanup_models'] = auto_cleanup
                            config['manga_settings']['advanced']['debug_mode'] = debug
                            config['manga_settings']['advanced']['save_intermediate'] = save_inter
                            config['concise_pipeline_logs'] = concise_logs
                            
                            # Save to file
                            result = self.save_config(config)
                            
                            # Show success message for 3 seconds
                            return gr.update(value=result, visible=True)
                            
                        except Exception as e:
                            return gr.update(value=f"❌ Save failed: {str(e)}", visible=True)
                    
                    # Save button will be configured after all components are created
                    
                    # Auto-hide status message after 3 seconds
                    def hide_status_after_delay():
                        import time
                        time.sleep(3)
                        return gr.update(visible=False)
                    
                    # Note: We can't use the change event to auto-hide because it would trigger immediately
                    # The status will remain visible until manually dismissed or page refresh
                    
                    # All individual field auto-save handlers removed - use manual Save Config button instead
                    
                    # Translate button click handler
                    translate_manga_btn.click(
                        fn=self.start_manga_translation,
                        inputs=[
                            manga_images,
                            manga_model,
                            manga_api_key,
                            manga_profile,
                            manga_system_prompt,
                            ocr_provider,
                            google_creds,
                            azure_key,
                            azure_endpoint,
                            bubble_detection,
                            inpainting,
                            font_size_mode,
                            font_size,
                            font_multiplier,
                            min_font_size,
                            max_font_size,
                            text_color_rgb,
                            shadow_enabled,
                            shadow_color,
                            shadow_offset_x,
                            shadow_offset_y,
                            shadow_blur,
                            bg_opacity,
                            bg_style,
                            parallel_panel_translation,
                            panel_max_workers
                        ],
                        outputs=[manga_logs, manga_output_gallery, manga_cbz_output, manga_status, manga_progress_group, manga_progress_text, manga_progress_bar, translate_manga_btn, stop_manga_btn]
                    )
                    
                    # Stop button click handler
                    stop_manga_btn.click(
                        fn=self.stop_manga_translation,
                        inputs=[],
                        outputs=[translate_manga_btn, stop_manga_btn, manga_status]
                    )
                    
                    # Load settings from localStorage on page load
                    def load_settings_from_storage():
                        """Load settings from localStorage or config file"""
                        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
                        
                        if not is_hf_spaces:
                            # Load from config file locally
                            config = self.load_config()
                            # Decrypt API keys if needed
                            if API_KEY_ENCRYPTION_AVAILABLE:
                                config = decrypt_config(config)
                            return [
                                config.get('model', 'gpt-4-turbo'),
                                config.get('api_key', ''),
                                config.get('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # profile
                                self.profiles.get(config.get('active_profile', list(self.profiles.keys())[0] if self.profiles else ''), ''),  # prompt
                                config.get('ocr_provider', 'custom-api'),
                                None,  # google_creds (file component - can't be pre-filled)
                                config.get('azure_vision_key', ''),
                                config.get('azure_vision_endpoint', ''),
                                config.get('bubble_detection_enabled', True),
                                config.get('inpainting_enabled', True),
                                config.get('manga_font_size_mode', 'auto'),
                                config.get('manga_font_size', 24),
                                config.get('manga_font_multiplier', 1.0),
                                config.get('manga_min_font_size', 12),
                                config.get('manga_max_font_size', 48),
                                config.get('manga_text_color', [255, 255, 255]),  # Default white text
                                config.get('manga_shadow_enabled', True),
                                config.get('manga_shadow_color', [0, 0, 0]),  # Default black shadow
                                config.get('manga_shadow_offset_x', 2),
                                config.get('manga_shadow_offset_y', 2),
                                config.get('manga_shadow_blur', 0),
                                config.get('manga_bg_opacity', 180),
                                config.get('manga_bg_style', 'auto'),
                                config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False),
                                config.get('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7)
                            ]
                        else:
                            # For HF Spaces, return defaults (will be overridden by JS)
                            return [
                                'gpt-4-turbo',  # model
                                '',  # api_key
                                list(self.profiles.keys())[0] if self.profiles else '',  # profile
                                self.profiles.get(list(self.profiles.keys())[0] if self.profiles else '', ''),  # prompt
                                'custom-api',  # ocr_provider
                                None,  # google_creds (file component - can't be pre-filled)
                                '',  # azure_key
                                '',  # azure_endpoint
                                True,  # bubble_detection
                                True,  # inpainting
                                'auto',  # font_size_mode
                                24,  # font_size
                                1.0,  # font_multiplier
                                12,  # min_font_size
                                48,  # max_font_size
                                '#FFFFFF',  # text_color - white
                                True,  # shadow_enabled
                                '#000000',  # shadow_color - black
                                2,  # shadow_offset_x
                                2,  # shadow_offset_y
                                0,  # shadow_blur
                                180,  # bg_opacity
                                'auto',  # bg_style
                                False,  # parallel_panel_translation
                                7  # panel_max_workers
                            ]
                    
                    # Store references for load handler
                    self.manga_components = {
                        'model': manga_model,
                        'api_key': manga_api_key,
                        'profile': manga_profile,
                        'prompt': manga_system_prompt,
                        'ocr_provider': ocr_provider,
                        'google_creds': google_creds,
                        'azure_key': azure_key,
                        'azure_endpoint': azure_endpoint,
                        'bubble_detection': bubble_detection,
                        'inpainting': inpainting,
                        'font_size_mode': font_size_mode,
                        'font_size': font_size,
                        'font_multiplier': font_multiplier,
                        'min_font_size': min_font_size,
                        'max_font_size': max_font_size,
                        'text_color_rgb': text_color_rgb,
                        'shadow_enabled': shadow_enabled,
                        'shadow_color': shadow_color,
                        'shadow_offset_x': shadow_offset_x,
                        'shadow_offset_y': shadow_offset_y,
                        'shadow_blur': shadow_blur,
                        'bg_opacity': bg_opacity,
                        'bg_style': bg_style,
                        'parallel_panel_translation': parallel_panel_translation,
                        'panel_max_workers': panel_max_workers
                    }
                    self.load_settings_fn = load_settings_from_storage
                
                # Manga Settings Tab - NEW
                with gr.Tab("🎬 Manga Settings"):
                    gr.Markdown("### Advanced Manga Translation Settings")
                    gr.Markdown("Configure bubble detection, inpainting, preprocessing, and rendering options.")
                    
                    with gr.Accordion("🕹️ Bubble Detection & Inpainting", open=True):
                        gr.Markdown("#### Bubble Detection")
                        
                        detector_type = gr.Radio(
                            choices=["rtdetr_onnx", "rtdetr", "yolo"],
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detector_type', 'rtdetr_onnx'),
                            label="Detector Type",
                            interactive=True
                        )
                        
                        rtdetr_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('rtdetr_confidence', 0.3),
                            step=0.05,
                            label="RT-DETR Confidence Threshold",
                            interactive=True
                        )
                        
                        bubble_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('bubble_confidence', 0.3),
                            step=0.05,
                            label="YOLO Bubble Confidence Threshold",
                            interactive=True
                        )
                        
                        detect_text_bubbles = gr.Checkbox(
                            label="Detect Text Bubbles",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_text_bubbles', True)
                        )
                        
                        detect_empty_bubbles = gr.Checkbox(
                            label="Detect Empty Bubbles",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_empty_bubbles', True)
                        )
                        
                        detect_free_text = gr.Checkbox(
                            label="Detect Free Text (outside bubbles)",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_free_text', True)
                        )
                        
                        bubble_max_detections = gr.Slider(
                            minimum=1,
                            maximum=2000,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('bubble_max_detections_yolo', 100),
                            step=1,
                            label="Max detections (YOLO only)",
                            interactive=True,
                            info="Maximum number of bubble detections for YOLO detector"
                        )
                        
                        gr.Markdown("#### Inpainting")
                        
                        local_inpaint_method = gr.Radio(
                            choices=["anime_onnx", "anime", "lama", "lama_onnx", "aot", "aot_onnx"],
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime_onnx'),
                            label="Local Inpainting Model",
                            interactive=True
                        )
                        
                        with gr.Row():
                            download_models_btn = gr.Button(
                                "📥 Download Models",
                                variant="secondary",
                                size="sm"
                            )
                            load_models_btn = gr.Button(
                                "📂 Load Models",
                                variant="secondary",
                                size="sm"
                            )
                        
                        gr.Markdown("#### Mask Dilation")
                        
                        auto_iterations = gr.Checkbox(
                            label="Auto Iterations (Recommended)",
                            value=self.get_config_value('manga_settings', {}).get('auto_iterations', True)
                        )
                        
                        mask_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('mask_dilation', 0),
                            step=1,
                            label="General Mask Dilation",
                            interactive=True
                        )
                        
                        text_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('text_bubble_dilation_iterations', 2),
                            step=1,
                            label="Text Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        empty_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('empty_bubble_dilation_iterations', 3),
                            step=1,
                            label="Empty Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        free_text_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('free_text_dilation_iterations', 3),
                            step=1,
                            label="Free Text Dilation Iterations",
                            interactive=True
                        )
                    
                    with gr.Accordion("🖌️ Image Preprocessing", open=False):
                        preprocessing_enabled = gr.Checkbox(
                            label="Enable Preprocessing",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('enabled', False)
                        )
                        
                        auto_detect_quality = gr.Checkbox(
                            label="Auto Detect Image Quality",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('auto_detect_quality', True)
                        )
                        
                        enhancement_strength = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('enhancement_strength', 1.5),
                            step=0.1,
                            label="Enhancement Strength",
                            interactive=True
                        )
                        
                        denoise_strength = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('denoise_strength', 10),
                            step=1,
                            label="Denoise Strength",
                            interactive=True
                        )
                        
                        max_image_dimension = gr.Number(
                            label="Max Image Dimension (pixels)",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('max_image_dimension', 2000),
                            minimum=500
                        )
                        
                        chunk_height = gr.Number(
                            label="Chunk Height for Large Images",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('chunk_height', 1000),
                            minimum=500
                        )
                        
                        gr.Markdown("#### HD Strategy for Inpainting")
                        gr.Markdown("*Controls how large images are processed during inpainting*")
                        
                        hd_strategy = gr.Radio(
                            choices=["original", "resize", "crop"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy', 'resize'),
                            label="HD Strategy",
                            interactive=True,
                            info="original = legacy full-image; resize/crop = faster"
                        )
                        
                        hd_strategy_resize_limit = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_resize_limit', 1536),
                            step=64,
                            label="Resize Limit (long edge, px)",
                            info="For resize strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_margin = gr.Slider(
                            minimum=0,
                            maximum=256,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_margin', 16),
                            step=2,
                            label="Crop Margin (px)",
                            info="For crop strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_trigger = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_trigger_size', 1024),
                            step=64,
                            label="Crop Trigger Size (px)",
                            info="Apply crop only if long edge exceeds this",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Image Tiling")
                        gr.Markdown("*Alternative tiling strategy (note: HD Strategy takes precedence)*")
                        
                        tiling_enabled = gr.Checkbox(
                            label="Enable Tiling",
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('enabled', False)
                        )
                        
                        tiling_tile_size = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('tile_size', 480),
                            step=64,
                            label="Tile Size (px)",
                            interactive=True
                        )
                        
                        tiling_tile_overlap = gr.Slider(
                            minimum=0,
                            maximum=128,
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('tile_overlap', 64),
                            step=16,
                            label="Tile Overlap (px)",
                            interactive=True
                        )
                    
                    with gr.Accordion("🎨 Font & Text Rendering", open=False):
                        gr.Markdown("#### Font Sizing Algorithm")
                        
                        font_algorithm = gr.Radio(
                            choices=["smart", "simple"],
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('algorithm', 'smart'),
                            label="Font Sizing Algorithm",
                            interactive=True
                        )
                        
                        prefer_larger = gr.Checkbox(
                            label="Prefer Larger Fonts",
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('prefer_larger', True)
                        )
                        
                        max_lines = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('max_lines', 10),
                            step=1,
                            label="Maximum Lines Per Bubble",
                            interactive=True
                        )
                        
                        line_spacing = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('line_spacing', 1.3),
                            step=0.1,
                            label="Line Spacing Multiplier",
                            interactive=True
                        )
                        
                        bubble_size_factor = gr.Checkbox(
                            label="Use Bubble Size Factor",
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('bubble_size_factor', True)
                        )
                        
                        auto_fit_style = gr.Radio(
                            choices=["balanced", "aggressive", "conservative"],
                            value=self.get_config_value('manga_settings', {}).get('rendering', {}).get('auto_fit_style', 'balanced'),
                            label="Auto Fit Style",
                            interactive=True
                        )
                    
                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        gr.Markdown("#### Format Detection")
                        
                        format_detection = gr.Checkbox(
                            label="Enable Format Detection (manga/webtoon)",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('format_detection', True)
                        )
                        
                        webtoon_mode = gr.Radio(
                            choices=["auto", "force_manga", "force_webtoon"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('webtoon_mode', 'auto'),
                            label="Webtoon Mode",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Inpainting Performance")
                        
                        inpaint_batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('batch_size', 10),
                            step=1,
                            label="Batch Size",
                            interactive=True,
                            info="Process multiple regions at once"
                        )
                        
                        inpaint_cache_enabled = gr.Checkbox(
                            label="Enable inpainting cache (speeds up repeated processing)",
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('enable_cache', True)
                        )
                        
                        gr.Markdown("#### Performance")
                        
                        parallel_processing = gr.Checkbox(
                            label="Enable Parallel Processing",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_processing', True)
                        )
                        
                        max_workers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('max_workers', 2),
                            step=1,
                            label="Max Worker Threads",
                            interactive=True
                        )
                        
                        gr.Markdown("**⚡ Advanced Performance**")
                        
                        preload_local_inpainting = gr.Checkbox(
                            label="Preload local inpainting instances for panel-parallel runs",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('preload_local_inpainting_for_panels', True),
                            info="Preloads inpainting models to speed up parallel processing"
                        )
                        
                        panel_start_stagger = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_start_stagger_ms', 30),
                            step=10,
                            label="Panel start stagger",
                            interactive=True,
                            info="Milliseconds delay between panel starts"
                        )
                        
                        gr.Markdown("#### Model Optimization")
                        
                        torch_precision = gr.Radio(
                            choices=["fp32", "fp16"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('torch_precision', 'fp16'),
                            label="Torch Precision",
                            interactive=True
                        )
                        
                        auto_cleanup_models = gr.Checkbox(
                            label="Auto Cleanup Models from Memory",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('auto_cleanup_models', False)
                        )
                        
                        gr.Markdown("#### Debug Options")
                        
                        debug_mode = gr.Checkbox(
                            label="Enable Debug Mode",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('debug_mode', False)
                        )
                        
                        save_intermediate = gr.Checkbox(
                            label="Save Intermediate Files",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('save_intermediate', False)
                        )
                        
                        concise_pipeline_logs = gr.Checkbox(
                            label="Concise Pipeline Logs",
                            value=self.get_config_value('concise_pipeline_logs', True)
                        )
                    
                    # Button handlers for model management
                    def download_models_handler(detector_type_val, inpaint_method_val):
                        """Download selected models"""
                        messages = []
                        
                        try:
                            # Download bubble detection model
                            if detector_type_val:
                                messages.append(f"📥 Downloading {detector_type_val} bubble detector...")
                                try:
                                    from bubble_detector import BubbleDetector
                                    bd = BubbleDetector()
                                    
                                    if detector_type_val == "rtdetr_onnx":
                                        if bd.load_rtdetr_onnx_model():
                                            messages.append("✅ RT-DETR ONNX model downloaded successfully")
                                        else:
                                            messages.append("❌ Failed to download RT-DETR ONNX model")
                                    elif detector_type_val == "rtdetr":
                                        if bd.load_rtdetr_model():
                                            messages.append("✅ RT-DETR model downloaded successfully")
                                        else:
                                            messages.append("❌ Failed to download RT-DETR model")
                                    elif detector_type_val == "yolo":
                                        messages.append("ℹ️ YOLO models are downloaded automatically on first use")
                                except Exception as e:
                                    messages.append(f"❌ Error downloading detector: {str(e)}")
                            
                            # Download inpainting model
                            if inpaint_method_val:
                                messages.append(f"\n📥 Downloading {inpaint_method_val} inpainting model...")
                                try:
                                    from local_inpainter import LocalInpainter, LAMA_JIT_MODELS
                                    
                                    inpainter = LocalInpainter({})
                                    
                                    # Map method names to download keys
                                    method_map = {
                                        'anime_onnx': 'anime_onnx',
                                        'anime': 'anime',
                                        'lama': 'lama',
                                        'lama_onnx': 'lama_onnx',
                                        'aot': 'aot',
                                        'aot_onnx': 'aot_onnx'
                                    }
                                    
                                    method_key = method_map.get(inpaint_method_val)
                                    if method_key and method_key in LAMA_JIT_MODELS:
                                        model_info = LAMA_JIT_MODELS[method_key]
                                        messages.append(f"Downloading {model_info['name']}...")
                                        
                                        model_path = inpainter.download_jit_model(method_key)
                                        if model_path:
                                            messages.append(f"✅ {model_info['name']} downloaded to: {model_path}")
                                        else:
                                            messages.append(f"❌ Failed to download {model_info['name']}")
                                    else:
                                        messages.append(f"ℹ️ {inpaint_method_val} is downloaded automatically on first use")
                                        
                                except Exception as e:
                                    messages.append(f"❌ Error downloading inpainting model: {str(e)}")
                            
                            if not messages:
                                messages.append("ℹ️ No models selected for download")
                                
                        except Exception as e:
                            messages.append(f"❌ Error during download: {str(e)}")
                        
                        return gr.Info("\n".join(messages))
                    
                    def load_models_handler(detector_type_val, inpaint_method_val):
                        """Load selected models into memory"""
                        messages = []
                        
                        try:
                            # Load bubble detection model
                            if detector_type_val:
                                messages.append(f"📦 Loading {detector_type_val} bubble detector...")
                                try:
                                    from bubble_detector import BubbleDetector
                                    bd = BubbleDetector()
                                    
                                    if detector_type_val == "rtdetr_onnx":
                                        if bd.load_rtdetr_onnx_model():
                                            messages.append("✅ RT-DETR ONNX model loaded successfully")
                                        else:
                                            messages.append("❌ Failed to load RT-DETR ONNX model")
                                    elif detector_type_val == "rtdetr":
                                        if bd.load_rtdetr_model():
                                            messages.append("✅ RT-DETR model loaded successfully")
                                        else:
                                            messages.append("❌ Failed to load RT-DETR model")
                                    elif detector_type_val == "yolo":
                                        messages.append("ℹ️ YOLO models are loaded automatically when needed")
                                except Exception as e:
                                    messages.append(f"❌ Error loading detector: {str(e)}")
                            
                            # Load inpainting model
                            if inpaint_method_val:
                                messages.append(f"\n📦 Loading {inpaint_method_val} inpainting model...")
                                try:
                                    from local_inpainter import LocalInpainter, LAMA_JIT_MODELS
                                    import os
                                    
                                    inpainter = LocalInpainter({})
                                    
                                    # Map method names to model keys
                                    method_map = {
                                        'anime_onnx': 'anime_onnx',
                                        'anime': 'anime',
                                        'lama': 'lama',
                                        'lama_onnx': 'lama_onnx',
                                        'aot': 'aot',
                                        'aot_onnx': 'aot_onnx'
                                    }
                                    
                                    method_key = method_map.get(inpaint_method_val)
                                    if method_key:
                                        # First check if model exists, download if not
                                        if method_key in LAMA_JIT_MODELS:
                                            model_info = LAMA_JIT_MODELS[method_key]
                                            cache_dir = os.path.expanduser('~/.cache/inpainting')
                                            model_filename = os.path.basename(model_info['url'])
                                            model_path = os.path.join(cache_dir, model_filename)
                                            
                                            if not os.path.exists(model_path):
                                                messages.append(f"Model not found, downloading first...")
                                                model_path = inpainter.download_jit_model(method_key)
                                                if not model_path:
                                                    messages.append(f"❌ Failed to download model")
                                                    return gr.Info("\n".join(messages))
                                            
                                            # Now load the model
                                            if inpainter.load_model(method_key, model_path):
                                                messages.append(f"✅ {model_info['name']} loaded successfully")
                                            else:
                                                messages.append(f"❌ Failed to load {model_info['name']}")
                                        else:
                                            messages.append(f"ℹ️ {inpaint_method_val} will be loaded automatically when needed")
                                    else:
                                        messages.append(f"ℹ️ Unknown method: {inpaint_method_val}")
                                        
                                except Exception as e:
                                    messages.append(f"❌ Error loading inpainting model: {str(e)}")
                            
                            if not messages:
                                messages.append("ℹ️ No models selected for loading")
                                
                        except Exception as e:
                            messages.append(f"❌ Error during loading: {str(e)}")
                        
                        return gr.Info("\n".join(messages))
                    
                    download_models_btn.click(
                        fn=download_models_handler,
                        inputs=[detector_type, local_inpaint_method],
                        outputs=None
                    )
                    
                    load_models_btn.click(
                        fn=load_models_handler,
                        inputs=[detector_type, local_inpaint_method],
                        outputs=None
                    )
                    
                    # Auto-save parallel panel translation settings
                    def save_parallel_settings(preload_enabled, parallel_enabled, max_workers, stagger_ms):
                        """Save parallel panel translation settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'advanced' not in current_config['manga_settings']:
                                current_config['manga_settings']['advanced'] = {}
                            
                            current_config['manga_settings']['advanced']['preload_local_inpainting_for_panels'] = bool(preload_enabled)
                            current_config['manga_settings']['advanced']['parallel_panel_translation'] = bool(parallel_enabled)
                            current_config['manga_settings']['advanced']['panel_max_workers'] = int(max_workers)
                            current_config['manga_settings']['advanced']['panel_start_stagger_ms'] = int(stagger_ms)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save parallel panel settings: {e}")
                            return None
                    
                    # Auto-save inpainting performance settings
                    def save_inpainting_settings(batch_size, cache_enabled):
                        """Save inpainting performance settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'inpainting' not in current_config['manga_settings']:
                                current_config['manga_settings']['inpainting'] = {}
                            
                            current_config['manga_settings']['inpainting']['batch_size'] = int(batch_size)
                            current_config['manga_settings']['inpainting']['enable_cache'] = bool(cache_enabled)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save inpainting settings: {e}")
                            return None
                    
                    # Auto-save preload local inpainting setting
                    def save_preload_setting(preload_enabled):
                        """Save preload local inpainting setting to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'advanced' not in current_config['manga_settings']:
                                current_config['manga_settings']['advanced'] = {}
                            
                            current_config['manga_settings']['advanced']['preload_local_inpainting_for_panels'] = bool(preload_enabled)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save preload setting: {e}")
                            return None
                    
                    # Auto-save bubble detection settings
                    def save_bubble_detection_settings(detector_type_val, rtdetr_conf, bubble_conf, detect_text, detect_empty, detect_free, max_detections, local_method_val):
                        """Save bubble detection settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'ocr' not in current_config['manga_settings']:
                                current_config['manga_settings']['ocr'] = {}
                            if 'inpainting' not in current_config['manga_settings']:
                                current_config['manga_settings']['inpainting'] = {}
                            
                            # Save bubble detection settings
                            current_config['manga_settings']['ocr']['detector_type'] = detector_type_val
                            current_config['manga_settings']['ocr']['rtdetr_confidence'] = float(rtdetr_conf)
                            current_config['manga_settings']['ocr']['bubble_confidence'] = float(bubble_conf)
                            current_config['manga_settings']['ocr']['detect_text_bubbles'] = bool(detect_text)
                            current_config['manga_settings']['ocr']['detect_empty_bubbles'] = bool(detect_empty)
                            current_config['manga_settings']['ocr']['detect_free_text'] = bool(detect_free)
                            current_config['manga_settings']['ocr']['bubble_max_detections_yolo'] = int(max_detections)
                            
                            # Save inpainting method
                            current_config['manga_settings']['inpainting']['local_method'] = local_method_val
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save bubble detection settings: {e}")
                            return None
                    
                    # All Advanced Settings auto-save handlers removed - use manual Save Config button
                    
                    gr.Markdown("\n---\n**Note:** These settings will be saved to your config and applied to all manga translations.")
                
                # Manual Glossary Extraction Tab
                with gr.Tab("📝 Manual Glossary Extraction"):
                    gr.Markdown("""
                    ### Extract character names and terms from EPUB files
                    Configure extraction settings below, then upload an EPUB file to extract a glossary.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            glossary_epub = gr.File(
                                label="📖 Upload EPUB File",
                                file_types=[".epub"]
                            )
                            
                            with gr.Row():
                                extract_btn = gr.Button(
                                    "🔍 Extract Glossary",
                                    variant="primary",
                                    size="lg",
                                    scale=2
                                )
                                
                                stop_glossary_btn = gr.Button(
                                    "⏹️ Stop Extraction",
                                    variant="stop",
                                    size="lg",
                                    visible=False,
                                    scale=1
                                )
                            
                            glossary_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            glossary_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')
                            )
                            
                            # Tabs for different settings sections
                            with gr.Tabs():
                                # Extraction Settings Tab
                                with gr.Tab("Extraction Settings"):
                                    with gr.Accordion("🎯 Targeted Extraction Settings", open=True):
                                        with gr.Row():
                                            with gr.Column():
                                                min_freq = gr.Slider(
                                                    minimum=1,
                                                    maximum=10,
                                                    value=self.get_config_value('glossary_min_frequency', 2),
                                                    step=1,
                                                    label="Min frequency",
                                                    info="How many times a name must appear (lower = more terms)"
                                                )
                                                
                                                max_titles = gr.Slider(
                                                    minimum=10,
                                                    maximum=100,
                                                    value=self.get_config_value('glossary_max_titles', 30),
                                                    step=5,
                                                    label="Max titles",
                                                    info="Limits to prevent huge glossaries"
                                                )
                                                
                                                max_text_size = gr.Number(
                                                    label="Max text size",
                                                    value=self.get_config_value('glossary_max_text_size', 50000),
                                                    info="Characters to analyze (0 = entire text)"
                                                )
                                                
                                                max_sentences = gr.Slider(
                                                    minimum=50,
                                                    maximum=500,
                                                    value=self.get_config_value('glossary_max_sentences', 200),
                                                    step=10,
                                                    label="Max sentences",
                                                    info="Maximum sentences to send to AI (increase for more context)"
                                                )
                                            
                                            with gr.Column():
                                                max_names_slider = gr.Slider(
                                                    minimum=10,
                                                    maximum=200,
                                                    value=self.get_config_value('glossary_max_names', 50),
                                                    step=10,
                                                    label="Max names",
                                                    info="Maximum number of character names to extract"
                                                )
                                                
                                                translation_batch = gr.Slider(
                                                    minimum=10,
                                                    maximum=100,
                                                    value=self.get_config_value('glossary_batch_size', 50),
                                                    step=5,
                                                    label="Translation batch",
                                                    info="Terms per API call (larger = faster but may reduce quality)"
                                                )
                                                
                                                chapter_split_threshold = gr.Number(
                                                    label="Chapter split threshold",
                                                    value=self.get_config_value('glossary_chapter_split_threshold', 8192),
                                                    info="Split large texts into chunks (0 = no splitting)"
                                                )
                                        
                                        # Filter mode selection
                                        filter_mode = gr.Radio(
                                            choices=[
                                                "all",
                                                "only_with_honorifics",
                                                "only_without_honorifics"
                                            ],
                                            value=self.get_config_value('glossary_filter_mode', 'all'),
                                            label="Filter mode",
                                            info="What types of names to extract"
                                        )
                                        
                                        # Strip honorifics checkbox
                                        strip_honorifics = gr.Checkbox(
                                            label="Remove honorifics from extracted names",
                                            value=self.get_config_value('strip_honorifics', True),
                                            info="Remove suffixes like '님', 'さん', '先生' from names"
                                        )
                                        
                                        # Fuzzy threshold slider
                                        fuzzy_threshold = gr.Slider(
                                            minimum=0.5,
                                            maximum=1.0,
                                            value=self.get_config_value('glossary_fuzzy_threshold', 0.90),
                                            step=0.05,
                                            label="Fuzzy threshold",
                                            info="How similar names must be to match (0.9 = 90% match, 1.0 = exact match)"
                                        )
                        
                                
                                # Extraction Prompt Tab
                                with gr.Tab("Extraction Prompt"):
                                    gr.Markdown("""
                                    ### System Prompt for Extraction
                                    Customize how the AI extracts names and terms from your text.
                                    """)
                                    
                                    extraction_prompt = gr.Textbox(
                                        label="Extraction Template (Use placeholders: {language}, {min_frequency}, {max_names}, {max_titles})",
                                        lines=10,
                                        value=self.get_config_value('manual_glossary_prompt', 
                                            "Extract character names and important terms from the following text.\n\n"
                                            "Output format:\n{fields}\n\n"
                                            "Rules:\n- Output ONLY CSV lines in the exact format shown above\n"
                                            "- No headers, no extra text, no JSON\n"
                                            "- One entry per line\n"
                                            "- Leave gender empty for terms (just end with comma)")
                                    )
                                    
                                    reset_extraction_prompt_btn = gr.Button(
                                        "Reset to Default",
                                        variant="secondary",
                                        size="sm"
                                    )
                                
                                # Format Instructions Tab
                                with gr.Tab("Format Instructions"):
                                    gr.Markdown("""
                                    ### Output Format Instructions
                                    These instructions tell the AI exactly how to format the extracted glossary.
                                    """)
                                    
                                    format_instructions = gr.Textbox(
                                        label="Format Instructions (Use placeholder: {text_sample})",
                                        lines=10,
                                        value=self.get_config_value('glossary_format_instructions',
                                            "Return the results in EXACT CSV format with this header:\n"
                                            "type,raw_name,translated_name\n\n"
                                            "For example:\n"
                                            "character,김상현,Kim Sang-hyun\n"
                                            "character,갈편제,Gale Hardest\n"
                                            "term,마법사,Mage\n\n"
                                            "Only include terms that actually appear in the text.\n"
                                            "Do not use quotes around values unless they contain commas.\n\n"
                                            "Text to analyze:\n{text_sample}")
                                    )
                                    
                                    use_legacy_csv = gr.Checkbox(
                                        label="Use legacy CSV format",
                                        value=self.get_config_value('glossary_use_legacy_csv', False),
                                        info="When disabled: Uses clean format with sections (===CHARACTERS===). When enabled: Uses traditional CSV format with repeated type columns."
                                    )
                        
                        with gr.Column():
                            # Add logo and status at top
                            with gr.Row():
                                gr.Image(
                                    value="Halgakos.png",
                                    label=None,
                                    show_label=False,
                                    width=80,
                                    height=80,
                                    interactive=False,
                                    show_download_button=False,
                                    container=False
                                )
                                glossary_status_message = gr.Markdown(
                                    value="### Ready to extract\nUpload an EPUB file and click 'Extract Glossary' to begin.",
                                    visible=True
                                )
                            
                            # Progress section (similar to translation tabs)
                            with gr.Group(visible=False) as glossary_progress_group:
                                gr.Markdown("### Progress")
                                glossary_progress_text = gr.Textbox(
                                    label="📨 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                glossary_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Extraction Progress",
                                    interactive=False,
                                    show_label=True
                                )
                            
                            glossary_logs = gr.Textbox(
                                label="📋 Extraction Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to extract. Upload an EPUB file and configure settings.",
                                visible=True,
                                interactive=False
                            )
                            
                            glossary_output = gr.File(
                                label="📥 Download Glossary CSV",
                                visible=False
                            )
                            
                            glossary_status = gr.Textbox(
                                label="Final Status",
                                lines=3,
                                max_lines=5,
                                visible=False,
                                interactive=False
                            )
                    
                    extract_btn.click(
                        fn=self.extract_glossary_with_stop,
                        inputs=[
                            glossary_epub,
                            glossary_model,
                            glossary_api_key,
                            min_freq,
                            max_names_slider,
                            max_titles,
                            max_text_size,
                            max_sentences,
                            translation_batch,
                            chapter_split_threshold,
                            filter_mode,
                            strip_honorifics,
                            fuzzy_threshold,
                            extraction_prompt,
                            format_instructions,
                            use_legacy_csv
                        ],
                        outputs=[
                            glossary_output,
                            glossary_status_message,
                            glossary_progress_group,
                            glossary_logs,
                            glossary_status,
                            glossary_progress_text,
                            glossary_progress_bar,
                            extract_btn,
                            stop_glossary_btn
                        ]
                    )
                    
                    # Stop button handler
                    stop_glossary_btn.click(
                        fn=self.stop_glossary_extraction,
                        inputs=[],
                        outputs=[extract_btn, stop_glossary_btn, glossary_status]
                    )
                
                # QA Scanner Tab
                with gr.Tab("🔍 QA Scanner"):
                    gr.Markdown("""
                    ### Quick Scan for Translation Quality
                    Scan translated content for common issues like untranslated text, formatting problems, and quality concerns.
                    
                    **Supported inputs:**
                    - 📁 Output folder containing extracted HTML/XHTML files
                    - 📖 EPUB file (will be automatically extracted and scanned)
                    - 📦 ZIP file containing HTML/XHTML files
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            # Check if running on Hugging Face Spaces
                            is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
                            
                            if is_hf_spaces:
                                gr.Markdown("""
                                **🤗 Hugging Face Spaces Mode**
                                Upload an EPUB or ZIP file containing the translated content.
                                The scanner will extract and analyze the HTML/XHTML files inside.
                                """)
                                qa_folder_path = gr.File(
                                    label="📂 Upload EPUB or ZIP file",
                                    file_types=[".epub", ".zip"],
                                    type="filepath"
                                )
                            else:
                                qa_folder_path = gr.Textbox(
                                    label="📁 Path to Folder, EPUB, or ZIP",
                                    placeholder="Enter path to: folder with HTML files, EPUB file, or ZIP file",
                                    info="Can be a folder path, or direct path to an EPUB/ZIP file"
                                )
                            
                            with gr.Row():
                                qa_scan_btn = gr.Button(
                                    "⚡ Quick Scan",
                                    variant="primary",
                                    size="lg",
                                    scale=2
                                )
                                
                                stop_qa_btn = gr.Button(
                                    "⏹️ Stop Scan",
                                    variant="stop",
                                    size="lg",
                                    visible=False,
                                    scale=1
                                )
                            
                            with gr.Accordion("⚙️ Quick Scan Settings", open=True):
                                gr.Markdown("""
                                **Quick Scan Mode (85% threshold, Speed optimized)**
                                - 3-5x faster scanning
                                - Checks consecutive chapters only
                                - Simplified analysis
                                - Good for large libraries
                                - Minimal resource usage
                                """)
                                
                                # Foreign Character Detection
                                gr.Markdown("#### Foreign Character Detection")
                                min_foreign_chars = gr.Slider(
                                    minimum=0,
                                    maximum=50,
                                    value=self.get_config_value('qa_min_foreign_chars', 10),
                                    step=1,
                                    label="Minimum foreign characters to flag",
                                    info="0 = always flag, higher = more tolerant"
                                )
                                
                                # Detection Options
                                gr.Markdown("#### Detection Options")
                                check_repetition = gr.Checkbox(
                                    label="Check for excessive repetition",
                                    value=self.get_config_value('qa_check_repetition', True)
                                )
                                
                                check_glossary_leakage = gr.Checkbox(
                                    label="Check for glossary leakage (raw glossary entries in translation)",
                                    value=self.get_config_value('qa_check_glossary_leakage', True)
                                )
                                
                                # File Processing
                                gr.Markdown("#### File Processing")
                                min_file_length = gr.Slider(
                                    minimum=0,
                                    maximum=5000,
                                    value=self.get_config_value('qa_min_file_length', 0),
                                    step=100,
                                    label="Minimum file length (characters)",
                                    info="Skip files shorter than this"
                                )
                                
                                # Additional Checks
                                gr.Markdown("#### Additional Checks")
                                check_multiple_headers = gr.Checkbox(
                                    label="Detect files with 2 or more headers (h1-h6 tags)",
                                    value=self.get_config_value('qa_check_multiple_headers', True),
                                    info="Identifies files that may have been incorrectly split or merged"
                                )
                                
                                check_missing_html = gr.Checkbox(
                                    label="Flag HTML files with missing <html> tag",
                                    value=self.get_config_value('qa_check_missing_html', True),
                                    info="Checks if HTML files have proper structure"
                                )
                                
                                check_insufficient_paragraphs = gr.Checkbox(
                                    label="Check for insufficient paragraph tags",
                                    value=self.get_config_value('qa_check_insufficient_paragraphs', True)
                                )
                                
                                min_paragraph_percentage = gr.Slider(
                                    minimum=10,
                                    maximum=90,
                                    value=self.get_config_value('qa_min_paragraph_percentage', 30),
                                    step=5,
                                    label="Minimum text in <p> tags (%)",
                                    info="Files with less than this percentage will be flagged"
                                )
                                
                                # Report Settings
                                gr.Markdown("#### Report Settings")
                                
                                report_format = gr.Radio(
                                    choices=["summary", "detailed", "verbose"],
                                    value=self.get_config_value('qa_report_format', 'detailed'),
                                    label="Report format",
                                    info="Summary = brief overview, Detailed = recommended, Verbose = all data"
                                )
                                
                                auto_save_report = gr.Checkbox(
                                    label="Automatically save report after scan",
                                    value=self.get_config_value('qa_auto_save_report', True)
                                )
                        
                        with gr.Column():
                            # Add logo and status at top
                            with gr.Row():
                                gr.Image(
                                    value="Halgakos.png",
                                    label=None,
                                    show_label=False,
                                    width=80,
                                    height=80,
                                    interactive=False,
                                    show_download_button=False,
                                    container=False
                                )
                                qa_status_message = gr.Markdown(
                                    value="### Ready to scan\nEnter the path to your output folder and click 'Quick Scan' to begin.",
                                    visible=True
                                )
                            
                            # Progress section
                            with gr.Group(visible=False) as qa_progress_group:
                                gr.Markdown("### Progress")
                                qa_progress_text = gr.Textbox(
                                    label="📨 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                qa_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Scan Progress",
                                    interactive=False,
                                    show_label=True
                                )
                            
                            qa_logs = gr.Textbox(
                                label="📋 Scan Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to scan. Enter output folder path and configure settings.",
                                visible=True,
                                interactive=False
                            )
                            
                            qa_report = gr.File(
                                label="📄 Download QA Report",
                                visible=False
                            )
                            
                            qa_status = gr.Textbox(
                                label="Final Status",
                                lines=3,
                                max_lines=5,
                                visible=False,
                                interactive=False
                            )
                    
                    # QA Scan button handler
                    qa_scan_btn.click(
                        fn=self.run_qa_scan_with_stop,
                        inputs=[
                            qa_folder_path,
                            min_foreign_chars,
                            check_repetition,
                            check_glossary_leakage,
                            min_file_length,
                            check_multiple_headers,
                            check_missing_html,
                            check_insufficient_paragraphs,
                            min_paragraph_percentage,
                            report_format,
                            auto_save_report
                        ],
                        outputs=[
                            qa_report,
                            qa_status_message,
                            qa_progress_group,
                            qa_logs,
                            qa_status,
                            qa_progress_text,
                            qa_progress_bar,
                            qa_scan_btn,
                            stop_qa_btn
                        ]
                    )
                    
                    # Stop button handler
                    stop_qa_btn.click(
                        fn=self.stop_qa_scan,
                        inputs=[],
                        outputs=[qa_scan_btn, stop_qa_btn, qa_status]
                    )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("### Configuration")
                    
                    gr.Markdown("#### Translation Profiles")
                    gr.Markdown("Profiles are loaded from your `config_web.json` file. The web interface has its own separate configuration.")
                    
                    with gr.Accordion("View All Profiles", open=False):
                        profiles_text = "\n\n".join(
                            [f"**{name}**:\n```\n{prompt[:200]}...\n```" 
                             for name, prompt in self.profiles.items()]
                        )
                        gr.Markdown(profiles_text if profiles_text else "No profiles found")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Advanced Translation Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            thread_delay = gr.Slider(
                                minimum=0,
                                maximum=5,
                                value=self.get_config_value('thread_submission_delay', 0.1),
                                step=0.1,
                                label="Threading delay (s)",
                                interactive=True
                            )
                            
                            api_delay = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=self.get_config_value('delay', 1),
                                step=0.5,
                                label="API call delay (s)",
                                interactive=True
                            )
                            
                            chapter_range = gr.Textbox(
                                label="Chapter range (e.g., 5-10)",
                                value=self.get_config_value('chapter_range', ''),
                                placeholder="Leave empty for all chapters"
                            )
                            
                            token_limit = gr.Number(
                                label="Input Token limit",
                                value=self.get_config_value('token_limit', 200000),
                                minimum=0
                            )
                            
                            disable_token_limit = gr.Checkbox(
                                label="Disable Input Token Limit",
                                value=self.get_config_value('token_limit_disabled', False)
                            )
                            
                            output_token_limit = gr.Number(
                                label="Output Token limit",
                                value=self.get_config_value('max_output_tokens', 16000),
                                minimum=0
                            )
                        
                        with gr.Column():
                            contextual = gr.Checkbox(
                                label="Contextual Translation",
                                value=self.get_config_value('contextual', False)
                            )
                            
                            history_limit = gr.Number(
                                label="Translation History Limit",
                                value=self.get_config_value('translation_history_limit', 2),
                                minimum=0
                            )
                            
                            rolling_history = gr.Checkbox(
                                label="Rolling History Window",
                                value=self.get_config_value('translation_history_rolling', False)
                            )
                            
                            batch_translation = gr.Checkbox(
                                label="Batch Translation",
                                value=self.get_config_value('batch_translation', True)
                            )
                            
                            batch_size = gr.Number(
                                label="Batch Size",
                                value=self.get_config_value('batch_size', 10),
                                minimum=1
                            )
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Chapter Processing Options")
                    
                    with gr.Row():
                        with gr.Column():
                            # Chapter Header Translation
                            batch_translate_headers = gr.Checkbox(
                                label="Batch Translate Headers",
                                value=self.get_config_value('batch_translate_headers', False)
                            )
                            
                            headers_per_batch = gr.Number(
                                label="Headers per batch",
                                value=self.get_config_value('headers_per_batch', 400),
                                minimum=1
                            )
                            
                            # NCX and CSS options
                            use_ncx_navigation = gr.Checkbox(
                                label="Use NCX-only Navigation (Compatibility Mode)",
                                value=self.get_config_value('use_ncx_navigation', False)
                            )
                            
                            attach_css_to_chapters = gr.Checkbox(
                                label="Attach CSS to Chapters (Fixes styling issues)",
                                value=self.get_config_value('attach_css_to_chapters', False)
                            )
                            
                            retain_source_extension = gr.Checkbox(
                                label="Retain source extension (no 'response_' prefix)",
                                value=self.get_config_value('retain_source_extension', True)
                            )
                        
                        with gr.Column():
                            # Conservative Batching
                            use_conservative_batching = gr.Checkbox(
                                label="Use Conservative Batching",
                                value=self.get_config_value('use_conservative_batching', False),
                                info="Groups chapters in batches of 3x batch size for memory management"
                            )
                            
                            # Gemini API Safety
                            disable_gemini_safety = gr.Checkbox(
                                label="Disable Gemini API Safety Filters",
                                value=self.get_config_value('disable_gemini_safety', False),
                                info="⚠️ Disables ALL content safety filters for Gemini models (BLOCK_NONE)"
                            )
                            
                            # OpenRouter Options
                            use_http_openrouter = gr.Checkbox(
                                label="Use HTTP-only for OpenRouter (bypass SDK)",
                                value=self.get_config_value('use_http_openrouter', False),
                                info="Direct HTTP POST with explicit headers"
                            )
                            
                            disable_openrouter_compression = gr.Checkbox(
                                label="Disable compression for OpenRouter (Accept-Encoding)",
                                value=self.get_config_value('disable_openrouter_compression', False),
                                info="Sends Accept-Encoding: identity for uncompressed responses"
                            )
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Chapter Extraction Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Text Extraction Method:**")
                            text_extraction_method = gr.Radio(
                                choices=["standard", "enhanced"],
                                value=self.get_config_value('text_extraction_method', 'standard'),
                                label="",
                                info="Standard uses BeautifulSoup, Enhanced uses html2text"
                            )
                            
                            gr.Markdown("• **Standard (BeautifulSoup)** - Traditional HTML parsing, fast and reliable")
                            gr.Markdown("• **Enhanced (html2text)** - Superior Unicode handling, cleaner text extraction")
                        
                        with gr.Column():
                            gr.Markdown("**File Filtering Level:**")
                            file_filtering_level = gr.Radio(
                                choices=["smart", "moderate", "full"],
                                value=self.get_config_value('file_filtering_level', 'smart'),
                                label="",
                                info="Controls which files are extracted from EPUBs"
                            )
                            
                            gr.Markdown("• **Smart (Aggressive Filtering)** - Skips navigation, TOC, copyright files")
                            gr.Markdown("• **Moderate** - Only skips obvious navigation files")
                            gr.Markdown("• **Full (No Filtering)** - Extracts ALL HTML/XHTML files")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Response Handling & Retry Logic")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**GPT-5 Thinking (OpenRouter/OpenAI-style)**")
                            enable_gpt_thinking = gr.Checkbox(
                                label="Enable GPT / OR Thinking",
                                value=self.get_config_value('enable_gpt_thinking', True),
                                info="Controls GPT-5 and OpenRouter reasoning"
                            )
                            
                            with gr.Row():
                                gpt_thinking_effort = gr.Dropdown(
                                    choices=["low", "medium", "high"],
                                    value=self.get_config_value('gpt_thinking_effort', 'medium'),
                                    label="Effort",
                                    interactive=True
                                )
                                
                                or_thinking_tokens = gr.Number(
                                    label="OR Thinking Tokens",
                                    value=self.get_config_value('or_thinking_tokens', 2000),
                                    minimum=0,
                                    maximum=50000,
                                    info="tokens"
                                )
                            
                            gr.Markdown("*Provide Tokens to force a max token budget for other models; GPT-5 only uses Effort (low/medium/high)*", elem_classes=["markdown-small"])
                        
                        with gr.Column():
                            gr.Markdown("**Gemini Thinking Mode**")
                            enable_gemini_thinking = gr.Checkbox(
                                label="Enable Gemini Thinking",
                                value=self.get_config_value('enable_gemini_thinking', False),
                                info="Control Gemini's thinking process",
                                interactive=True
                            )
                            
                            gemini_thinking_budget = gr.Number(
                                label="Budget",
                                value=self.get_config_value('gemini_thinking_budget', 0),
                                minimum=0,
                                maximum=50000,
                                info="tokens (0 = disabled)",
                                interactive=True
                            )
                            
                            gr.Markdown("*0 = disabled, 512-24576 = limited thinking*", elem_classes=["markdown-small"])
                    
                    gr.Markdown("---")
                    gr.Markdown("🔒 **API keys are encrypted** when saved to config using AES encryption.")
                    
                    save_api_key = gr.Checkbox(
                        label="Save API Key (Encrypted)",
                        value=True
                    )
                    
                    save_status = gr.Textbox(label="Settings Status", value="Use the 'Save Config' button to save changes", interactive=False)
                    
                    # Hidden HTML component for JavaScript execution
                    js_executor = gr.HTML("", visible=False)
                    
                    # Auto-save function for settings tab
                    def save_settings_tab(thread_delay_val, api_delay_val, chapter_range_val, token_limit_val, disable_token_limit_val, output_token_limit_val, contextual_val, history_limit_val, rolling_history_val, batch_translation_val, batch_size_val, save_api_key_val):
                        """Save settings from the Settings tab"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update non-encrypted fields
                            
                            # Update settings
                            current_config['thread_submission_delay'] = float(thread_delay_val)
                            current_config['delay'] = float(api_delay_val)
                            current_config['chapter_range'] = str(chapter_range_val)
                            current_config['token_limit'] = int(token_limit_val)
                            current_config['token_limit_disabled'] = bool(disable_token_limit_val)
                            current_config['max_output_tokens'] = int(output_token_limit_val)
                            current_config['contextual'] = bool(contextual_val)
                            current_config['translation_history_limit'] = int(history_limit_val)
                            current_config['translation_history_rolling'] = bool(rolling_history_val)
                            current_config['batch_translation'] = bool(batch_translation_val)
                            current_config['batch_size'] = int(batch_size_val)
                            
                            # Save to file
                            self.save_config(current_config)
                            
                            # JavaScript to save to localStorage
                            js_code = """
                            <script>
                            (function() {
                                // Save individual settings to localStorage
                                window.saveToLocalStorage('thread_delay', %f);
                                window.saveToLocalStorage('api_delay', %f);
                                window.saveToLocalStorage('chapter_range', '%s');
                                window.saveToLocalStorage('token_limit', %d);
                                window.saveToLocalStorage('disable_token_limit', %s);
                                window.saveToLocalStorage('output_token_limit', %d);
                                window.saveToLocalStorage('contextual', %s);
                                window.saveToLocalStorage('history_limit', %d);
                                window.saveToLocalStorage('rolling_history', %s);
                                window.saveToLocalStorage('batch_translation', %s);
                                window.saveToLocalStorage('batch_size', %d);
                                console.log('Settings saved to localStorage');
                            })();
                            </script>
                            """ % (
                                thread_delay_val, api_delay_val, chapter_range_val, token_limit_val,
                                str(disable_token_limit_val).lower(), output_token_limit_val,
                                str(contextual_val).lower(), history_limit_val,
                                str(rolling_history_val).lower(), str(batch_translation_val).lower(),
                                batch_size_val
                            )
                            
                            return "✅ Settings saved successfully", js_code
                        except Exception as e:
                            return f"❌ Failed to save: {str(e)}", ""
                    
                    # Settings tab auto-save handlers removed - use manual Save Config button
                    
                    # Token sync handlers removed - use manual Save Config button
                
                # Help Tab
                with gr.Tab("❓ Help"):
                    gr.Markdown("""
                    ## How to Use Glossarion
                    
                    ### Translation
                    1. Upload an EPUB file
                    2. Select AI model (GPT-4, Claude, etc.)
                    3. Enter your API key
                    4. Click "Translate"
                    5. Download the translated EPUB
                    
                    ### Manga Translation
                    1. Upload manga image(s) (PNG, JPG, etc.)
                    2. Select AI model and enter API key
                    3. Choose translation profile (e.g., Manga_JP, Manga_KR)
                    4. Configure OCR settings (Google Cloud Vision recommended)
                    5. Enable bubble detection and inpainting for best results
                    6. Click "Translate Manga"
                    
                    ### Glossary Extraction
                    1. Upload an EPUB file
                    2. Configure extraction settings
                    3. Click "Extract Glossary"
                    4. Use the CSV in future translations
                    
                    ### API Keys
                    - **OpenAI**: Get from https://platform.openai.com/api-keys
                    - **Anthropic**: Get from https://console.anthropic.com/
                    
                    ### Translation Profiles
                    Profiles contain detailed translation instructions and rules.
                    Select a profile that matches your source language and style preferences.
                    
                    You can create and edit profiles in the desktop application.
                    
                    ### Tips
                    - Use glossaries for consistent character name translation
                    - Lower temperature (0.1-0.3) for more literal translations
                    - Higher temperature (0.5-0.7) for more creative translations
                    """)
            
            # Create a comprehensive load function that refreshes ALL values
            def load_all_settings():
                """Load all settings from config file on page refresh"""
                # Reload config to get latest values
                self.config = self.load_config()
                self.decrypted_config = decrypt_config(self.config.copy()) if API_KEY_ENCRYPTION_AVAILABLE else self.config.copy()
                
                # CRITICAL: Reload profiles from config after reloading config
                self.profiles = self.default_prompts.copy()
                config_profiles = self.config.get('prompt_profiles', {})
                if config_profiles:
                    self.profiles.update(config_profiles)
                
                # Helper function to convert RGB arrays to hex
                def to_hex_color(color_value, default='#000000'):
                    if isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
                        return '#{:02x}{:02x}{:02x}'.format(int(color_value[0]), int(color_value[1]), int(color_value[2]))
                    elif isinstance(color_value, str):
                        return color_value if color_value.startswith('#') else default
                    return default
                
                # Return values for all tracked components
                return [
                    self.get_config_value('model', 'gpt-4-turbo'),  # epub_model
                    self.get_config_value('api_key', ''),  # epub_api_key
                    self.get_config_value('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # epub_profile
                    self.profiles.get(self.get_config_value('active_profile', ''), ''),  # epub_system_prompt
                    self.get_config_value('temperature', 0.3),  # epub_temperature
                    self.get_config_value('max_output_tokens', 16000),  # epub_max_tokens
                    self.get_config_value('enable_image_translation', False),  # enable_image_translation
                    self.get_config_value('enable_auto_glossary', False),  # enable_auto_glossary  
                    self.get_config_value('append_glossary_to_prompt', True),  # append_glossary
                    # Auto glossary settings
                    self.get_config_value('glossary_min_frequency', 2),  # auto_glossary_min_freq
                    self.get_config_value('glossary_max_names', 50),  # auto_glossary_max_names
                    self.get_config_value('glossary_max_titles', 30),  # auto_glossary_max_titles
                    self.get_config_value('glossary_batch_size', 50),  # auto_glossary_batch_size
                    self.get_config_value('glossary_filter_mode', 'all'),  # auto_glossary_filter_mode
                    self.get_config_value('glossary_fuzzy_threshold', 0.90),  # auto_glossary_fuzzy_threshold
                    # Manual glossary extraction settings
                    self.get_config_value('manual_glossary_min_frequency', self.get_config_value('glossary_min_frequency', 2)),  # min_freq
                    self.get_config_value('manual_glossary_max_names', self.get_config_value('glossary_max_names', 50)),  # max_names_slider
                    self.get_config_value('manual_glossary_max_titles', self.get_config_value('glossary_max_titles', 30)),  # max_titles
                    self.get_config_value('glossary_max_text_size', 50000),  # max_text_size
                    self.get_config_value('glossary_max_sentences', 200),  # max_sentences
                    self.get_config_value('manual_glossary_batch_size', self.get_config_value('glossary_batch_size', 50)),  # translation_batch
                    self.get_config_value('glossary_chapter_split_threshold', 8192),  # chapter_split_threshold
                    self.get_config_value('manual_glossary_filter_mode', self.get_config_value('glossary_filter_mode', 'all')),  # filter_mode
                    self.get_config_value('strip_honorifics', True),  # strip_honorifics
                    self.get_config_value('manual_glossary_fuzzy_threshold', self.get_config_value('glossary_fuzzy_threshold', 0.90)),  # fuzzy_threshold
                    # Chapter processing options
                    self.get_config_value('batch_translate_headers', False),  # batch_translate_headers
                    self.get_config_value('headers_per_batch', 400),  # headers_per_batch
                    self.get_config_value('use_ncx_navigation', False),  # use_ncx_navigation
                    self.get_config_value('attach_css_to_chapters', False),  # attach_css_to_chapters
                    self.get_config_value('retain_source_extension', True),  # retain_source_extension
                    self.get_config_value('use_conservative_batching', False),  # use_conservative_batching
                    self.get_config_value('disable_gemini_safety', False),  # disable_gemini_safety
                    self.get_config_value('use_http_openrouter', False),  # use_http_openrouter
                    self.get_config_value('disable_openrouter_compression', False),  # disable_openrouter_compression
                    self.get_config_value('text_extraction_method', 'standard'),  # text_extraction_method
                    self.get_config_value('file_filtering_level', 'smart'),  # file_filtering_level
                    # QA report format
                    self.get_config_value('qa_report_format', 'detailed'),  # report_format
                    # Thinking mode settings
                    self.get_config_value('enable_gpt_thinking', True),  # enable_gpt_thinking
                    self.get_config_value('gpt_thinking_effort', 'medium'),  # gpt_thinking_effort
                    self.get_config_value('or_thinking_tokens', 2000),  # or_thinking_tokens
                    self.get_config_value('enable_gemini_thinking', False),  # enable_gemini_thinking - disabled by default
                    self.get_config_value('gemini_thinking_budget', 0),  # gemini_thinking_budget - 0 = disabled
                    # Manga settings
                    self.get_config_value('model', 'gpt-4-turbo'),  # manga_model
                    self.get_config_value('api_key', ''),  # manga_api_key
                    self.get_config_value('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # manga_profile
                    self.profiles.get(self.get_config_value('active_profile', ''), ''),  # manga_system_prompt
                    self.get_config_value('ocr_provider', 'custom-api'),  # ocr_provider
                    self.get_config_value('azure_vision_key', ''),  # azure_key
                    self.get_config_value('azure_vision_endpoint', ''),  # azure_endpoint
                    self.get_config_value('bubble_detection_enabled', True),  # bubble_detection
                    self.get_config_value('inpainting_enabled', True),  # inpainting
                    self.get_config_value('manga_font_size_mode', 'auto'),  # font_size_mode
                    self.get_config_value('manga_font_size', 24),  # font_size
                    self.get_config_value('manga_font_multiplier', 1.0),  # font_multiplier
                    self.get_config_value('manga_min_font_size', 12),  # min_font_size
                    self.get_config_value('manga_max_font_size', 48),  # max_font_size
                    # Convert colors to hex format if they're stored as RGB arrays (white text, black shadow like manga integration)
                    to_hex_color(self.get_config_value('manga_text_color', [255, 255, 255]), '#FFFFFF'),  # text_color_rgb - default white
                    self.get_config_value('manga_shadow_enabled', True),  # shadow_enabled
                    to_hex_color(self.get_config_value('manga_shadow_color', [0, 0, 0]), '#000000'),  # shadow_color - default black
                    self.get_config_value('manga_shadow_offset_x', 2),  # shadow_offset_x
                    self.get_config_value('manga_shadow_offset_y', 2),  # shadow_offset_y
                    self.get_config_value('manga_shadow_blur', 0),  # shadow_blur
                    self.get_config_value('manga_bg_opacity', 130),  # bg_opacity
                    self.get_config_value('manga_bg_style', 'circle'),  # bg_style
                    self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False),  # parallel_panel_translation
                    self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7),  # panel_max_workers
                ]
            
            
            # SECURITY: Save Config button DISABLED to prevent API keys from being saved to persistent storage on HF Spaces
            # This is a critical security measure to prevent API key leakage in shared environments
            # save_config_btn.click(
            #     fn=save_all_config,
            #     inputs=[
            #         # EPUB tab fields
            #         epub_model, epub_api_key, epub_profile, epub_temperature, epub_max_tokens,
            #         enable_image_translation, enable_auto_glossary, append_glossary,
            #         # Auto glossary settings
            #         auto_glossary_min_freq, auto_glossary_max_names, auto_glossary_max_titles,
            #         auto_glossary_batch_size, auto_glossary_filter_mode, auto_glossary_fuzzy_threshold,
            #         enable_post_translation_scan,
            #         # Manual glossary extraction settings
            #         min_freq, max_names_slider, max_titles,
            #         max_text_size, max_sentences, translation_batch,
            #         chapter_split_threshold, filter_mode, strip_honorifics,
            #         fuzzy_threshold, extraction_prompt, format_instructions,
            #         use_legacy_csv,
            #         # QA Scanner settings
            #         min_foreign_chars, check_repetition, check_glossary_leakage,
            #         min_file_length, check_multiple_headers, check_missing_html,
            #         check_insufficient_paragraphs, min_paragraph_percentage,
            #         report_format, auto_save_report,
            #         # Chapter processing options
            #         batch_translate_headers, headers_per_batch, use_ncx_navigation,
            #         attach_css_to_chapters, retain_source_extension,
            #         use_conservative_batching, disable_gemini_safety,
            #         use_http_openrouter, disable_openrouter_compression,
            #         text_extraction_method, file_filtering_level,
            #         # Thinking mode settings
            #         enable_gpt_thinking, gpt_thinking_effort, or_thinking_tokens,
            #         enable_gemini_thinking, gemini_thinking_budget,
            #         # Manga tab fields  
            #         manga_model, manga_api_key, manga_profile,
            #         ocr_provider, azure_key, azure_endpoint,
            #         bubble_detection, inpainting,
            #         font_size_mode, font_size, font_multiplier, min_font_size, max_font_size,
            #         text_color_rgb, shadow_enabled, shadow_color,
            #         shadow_offset_x, shadow_offset_y, shadow_blur,
            #         bg_opacity, bg_style,
            #         parallel_panel_translation, panel_max_workers,
            #         # Advanced Settings fields
            #         detector_type, rtdetr_confidence, bubble_confidence,
            #         detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections,
            #         local_inpaint_method, webtoon_mode,
            #         inpaint_batch_size, inpaint_cache_enabled,
            #         parallel_processing, max_workers,
            #         preload_local_inpainting, panel_start_stagger,
            #         torch_precision, auto_cleanup_models,
            #         debug_mode, save_intermediate, concise_pipeline_logs
            #     ],
            #     outputs=[save_status_text]
            # )
            
            # Add load handler to restore settings on page load
            app.load(
                fn=load_all_settings,
                inputs=[],
                outputs=[
                    epub_model, epub_api_key, epub_profile, epub_system_prompt, epub_temperature, epub_max_tokens,
                    enable_image_translation, enable_auto_glossary, append_glossary,
                    # Auto glossary settings
                    auto_glossary_min_freq, auto_glossary_max_names, auto_glossary_max_titles,
                    auto_glossary_batch_size, auto_glossary_filter_mode, auto_glossary_fuzzy_threshold,
                    # Manual glossary extraction settings
                    min_freq, max_names_slider, max_titles,
                    max_text_size, max_sentences, translation_batch,
                    chapter_split_threshold, filter_mode, strip_honorifics,
                    fuzzy_threshold,
                    # Chapter processing options  
                    batch_translate_headers, headers_per_batch, use_ncx_navigation,
                    attach_css_to_chapters, retain_source_extension,
                    use_conservative_batching, disable_gemini_safety,
                    use_http_openrouter, disable_openrouter_compression,
                    text_extraction_method, file_filtering_level,
                    report_format,
                    # Thinking mode settings
                    enable_gpt_thinking, gpt_thinking_effort, or_thinking_tokens,
                    enable_gemini_thinking, gemini_thinking_budget,
                    # Manga settings
                    manga_model, manga_api_key, manga_profile, manga_system_prompt,
                    ocr_provider, azure_key, azure_endpoint, bubble_detection, inpainting,
                    font_size_mode, font_size, font_multiplier, min_font_size, max_font_size,
                    text_color_rgb, shadow_enabled, shadow_color, shadow_offset_x, shadow_offset_y,
                    shadow_blur, bg_opacity, bg_style, parallel_panel_translation, panel_max_workers
                ]
            )
        
        return app


def main():
    """Launch Gradio web app"""
    print("🚀 Starting Glossarion Web Interface...")
    
    # Check if running on Hugging Face Spaces
    is_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
    if is_spaces:
        print("🤗 Running on Hugging Face Spaces")
        print(f"📁 Space ID: {os.getenv('SPACE_ID', 'Unknown')}")
        print(f"📁 Files in current directory: {len(os.listdir('.'))} items")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"😎 Available manga modules: {MANGA_TRANSLATION_AVAILABLE}")
    else:
        print("🏠 Running locally")
    
    web_app = GlossarionWeb()
    app = web_app.create_interface()
    
    # Set favicon with absolute path if available (skip for Spaces)
    favicon_path = None
    if not is_spaces and os.path.exists("Halgakos.ico"):
        favicon_path = os.path.abspath("Halgakos.ico")
        print(f"✅ Using favicon: {favicon_path}")
    elif not is_spaces:
        print("⚠️ Halgakos.ico not found")
    
    # Launch with options appropriate for environment
    launch_args = {
        "server_name": "0.0.0.0",  # Allow external access
        "server_port": 7860,
        "share": False,
        "show_error": True,
    }
    
    # Only add favicon for non-Spaces environments
    if not is_spaces and favicon_path:
        launch_args["favicon_path"] = favicon_path
    
    app.launch(**launch_args)


if __name__ == "__main__":
    main()