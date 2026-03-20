# android_config.py
"""
Configuration management for Glossarion Android.
Loads/saves config from the app's private data directory.
Same JSON schema as config.json / config_web.json.
"""

import json
import os
import logging

from android_file_utils import get_config_path, get_app_data_dir

logger = logging.getLogger(__name__)


# Default values matching app.py's get_default_config()
DEFAULT_CONFIG = {
    'model': 'authgpt/gpt-5.2',
    'api_key': '',
    'api_call_delay': 0.5,
    'batch_translation': False,
    'batch_size': 10,
    'batching_mode': 'direct',
    'batch_group_size': 3,
    'text_extraction_method': 'standard',
    'file_filtering_level': 'smart',
    'enhanced_preserve_structure': True,
    'force_bs_for_traditional': True,
    'indefinitely_retry_rate_limit': False,
    'thread_submission_delay': 0.1,
    'active_profile': 'Universal',
    'output_language': 'English',
    'max_output_tokens': 128000,
    'translation_temperature': 0.3,
    'enable_auto_glossary': True,
    'contextual': False,
    'translation_history_limit': 2,
    'translation_history_rolling': False,
    'enable_streaming': False,
    'batch_translate_headers': False,
    'use_multi_api_keys': False,
    'multi_api_keys': [],
    'rotation_frequency': 1,

    # Reader settings
    'reader_font_size': 18,
    'reader_font_family': 'sans-serif',
    'reader_line_spacing': 1.5,
    'reader_margin': 'medium',
    'reader_text_align': 'left',
    'reader_theme': 'dark',
    'reader_brightness': 1.0,
    'reader_auto_scroll_speed': 0,
    'reader_orientation_lock': False,
    'bookmarks': {},        # {file_path: [{chapter, position, timestamp, label}]}
    'reading_progress': {},  # {file_path: {chapter, scroll_pos, percent}}

    # Library settings
    'library_scan_dirs': [],
    'library_sort_by': 'modified',

    # Glossary settings
    'enable_auto_glossary': True,
    'append_glossary_to_prompt': True,
    'glossary_min_frequency': 2,
    'glossary_max_names': 50,
    'glossary_max_titles': 30,
    'glossary_batch_size': 50,
    'glossary_filter_mode': 'all',
    'glossary_fuzzy_threshold': 0.90,

    # Thinking mode
    'enable_gpt_thinking': True,
    'gpt_thinking_effort': 'medium',
    'enable_gemini_thinking': False,
    'gemini_thinking_budget': 0,

    # Prompt profiles (merged with built-in defaults at load time)
    'prompt_profiles': {},
}


def _deep_merge(base, override):
    """Deep merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config():
    """Load configuration from disk, merged with defaults.
    
    Returns:
        dict: The merged configuration
    """
    config = DEFAULT_CONFIG.copy()
    config_path = get_config_path()

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            _deep_merge(config, saved)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.info("No saved config found, using defaults")
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")

    # Try to decrypt API keys
    try:
        import sys
        # Add parent dir (src/) to path so we can import api_key_encryption
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from api_key_encryption import decrypt_config
        config = decrypt_config(config)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"API key decryption failed: {e}")

    return config


def save_config(config):
    """Save configuration to disk.
    
    Args:
        config: dict to save
    """
    config_path = get_config_path()

    # Try to encrypt API keys before saving
    save_data = config.copy()
    try:
        import sys
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from api_key_encryption import encrypt_config
        save_data = encrypt_config(save_data)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"API key encryption failed: {e}")

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        # Atomic write: write to temp then rename
        tmp_path = config_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, config_path)
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        # Fallback direct write
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def update_config(config, key, value):
    """Update a single config key and save.
    
    Args:
        config: The config dict (modified in place)
        key: Config key (supports dot notation like 'reader.font_size')
        value: New value
    """
    parts = key.split('.')
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value
    save_config(config)


def get_reading_progress(config, file_path):
    """Get reading progress for a specific file.
    
    Returns:
        dict: {chapter: int, scroll_pos: float, percent: float} or None
    """
    progress = config.get('reading_progress', {})
    return progress.get(os.path.abspath(file_path))


def save_reading_progress(config, file_path, chapter, scroll_pos, percent):
    """Save reading progress for a specific file."""
    if 'reading_progress' not in config:
        config['reading_progress'] = {}
    config['reading_progress'][os.path.abspath(file_path)] = {
        'chapter': chapter,
        'scroll_pos': scroll_pos,
        'percent': round(percent, 1),
    }
    save_config(config)


def get_bookmarks(config, file_path):
    """Get bookmarks for a specific file.
    
    Returns:
        list: List of bookmark dicts
    """
    bookmarks = config.get('bookmarks', {})
    return bookmarks.get(os.path.abspath(file_path), [])


def add_bookmark(config, file_path, chapter, position, label=''):
    """Add a bookmark for a specific file."""
    import time
    if 'bookmarks' not in config:
        config['bookmarks'] = {}
    key = os.path.abspath(file_path)
    if key not in config['bookmarks']:
        config['bookmarks'][key] = []
    config['bookmarks'][key].append({
        'chapter': chapter,
        'position': position,
        'timestamp': time.time(),
        'label': label,
    })
    save_config(config)


def remove_bookmark(config, file_path, index):
    """Remove a bookmark by index."""
    key = os.path.abspath(file_path)
    bookmarks = config.get('bookmarks', {}).get(key, [])
    if 0 <= index < len(bookmarks):
        bookmarks.pop(index)
        save_config(config)
