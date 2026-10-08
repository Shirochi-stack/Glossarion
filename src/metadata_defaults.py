"""Default metadata / batch-header prompts written into the translator config.

``ensure_metadata_prompt_defaults(config)`` is the verbatim body of
``MetadataBatchTranslatorUI._initialize_default_prompts`` (metadata_batch_translator.py)
with ``self.gui.config`` passed in as ``config``. Only missing keys are added;
existing user values are never touched. The desktop UI method now calls it.

GUI-free; must stay importable on Python 3.10 without Qt.
"""


def ensure_metadata_prompt_defaults(config):
    """Initialize all default prompts in config if not present.

    Returns True when at least one default was added.
    """
    count_before = len(config)
    # Book title system prompt
    if 'book_title_system_prompt' not in config:
        config['book_title_system_prompt'] = (
            "Translate this book title to {target_lang} while retaining any acronyms. Do not output anything other than the translated text."
        )

    # Book title user prompt
    if 'book_title_prompt' not in config:
        config['book_title_prompt'] = (
            ""
        )

    # Batch header system prompt
    if 'batch_header_system_prompt' not in config:
        config['batch_header_system_prompt'] = (
            "You are a professional translator specializing in novel chapter titles. "
            "You must translate the chapter titles to {target_lang}. "
            "Respond with only the translated JSON, nothing else. "
            "Maintain the original tone and style while making titles natural in the target language."
        )

    # Batch header user prompt (existing)
    if 'batch_header_prompt' not in config:
        config['batch_header_prompt'] = (
            "Translate these chapter titles to {target_lang}.\n"
            "- For titles with parenthetical text, translate both the main title and the parenthetical content.\n"
            "- Translate the meaning accurately - don't use overly dramatic words unless the original implies them.\n"
            "- Preserve the chapter number format exactly as shown.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"
        )

    # Metadata batch prompt
    if 'metadata_batch_prompt' not in config:
        config['metadata_batch_prompt'] = (
            "Translate the following metadata fields to {target_lang}.\n"
            "Output ONLY a JSON object with the same field names as keys."
        )

    # Field-specific prompts
    if 'metadata_field_prompts' not in config:
        config['metadata_field_prompts'] = {
            'creator': "Romanize this author name. Do not output anything other than the romanized text.",
            'publisher': "Romanize this publisher name. Do not output anything other than the romanized text.",
            'subject': "Translate this book genre/subject to {target_lang}. Do not output anything other than the translated text.",
            'description': "Translate this book description to {target_lang}. Do not output anything other than the translated text.",
            'series': "Translate this series name to {target_lang}. Do not output anything other than the translated text.",
            '_default': "Translate this text to {target_lang}. Do not output anything other than the translated text."
        }
    return len(config) != count_before


# The prompt-related keys "Configure All" › "Reset all prompts to defaults" removes
# (MetadataBatchTranslatorUI._reset_all_prompts_to_defaults; the desktop method uses this list).
METADATA_PROMPT_RESET_KEYS = (
    'book_title_system_prompt', 'book_title_prompt',
    'metadata_system_prompt',
    'batch_header_system_prompt',
    'batch_header_prompt', 'batch_header_prepend_number_pattern',
    'metadata_batch_prompt',
    'metadata_field_prompts', 'lang_prompt_behavior',
    'forced_source_lang', 'output_language'
)


def reset_metadata_prompts(config):
    """The config side of "Reset all prompts to defaults": remove every
    ``METADATA_PROMPT_RESET_KEYS`` key, blank ``book_title_prompt`` and re-seed the default
    prompts (``ensure_metadata_prompt_defaults``). Returns the keys removed."""
    removed = [key for key in METADATA_PROMPT_RESET_KEYS if key in config]
    for key in removed:
        del config[key]
    # Force set book title prompt to new default
    config['book_title_prompt'] = ""
    ensure_metadata_prompt_defaults(config)
    return removed


# Chapter Header Translation - Detailed Guide (other_settings.HeaderTranslationHelpDialog, moved in U9;
# Glossarion Mobile shows the same sections in Tools › Headers & metadata ⓘ).
HEADER_HELP_TITLE = "Chapter Header Translation - Detailed Guide"
HEADER_HELP_SECTIONS = [
    {
        "title": "🔄 Translation Modes",
        "content": [
            "• OFF: Use existing headers from already translated chapters",
            "• ON: Extract all headers → Translate in batch → Update files"
        ]
    },
    {
        "title": "⚙️ Options Explained",
        "content": [
            "• Update headers in HTML files: Modifies the actual chapter files with translated headers",
            "• Save translations to .txt: Creates backup files with translation mappings",
            "• Headers per batch: Number of headers to translate simultaneously (affects API usage)"
        ]
    },
    {
        "title": "🚫 Ignore Options",
        "content": [
            "• Ignore header: Skip h1/h2/h3 tags (prevents re-translation of visible headers)",
            "• Skip title tag translation: Preserve <title> tags without translating them"
        ]
    },
    {
        "title": "⚠️ Fallback System",
        "content": [
            "• Use Sorted Fallback: If OPF-based matching fails, use sorted index matching",
            "• WARNING: Less accurate - may mismatch chapters if file order differs from OPF spine",
            "• Only use if you're experiencing matching issues with standard mode"
        ]
    },
    {
        "title": "📂 Standalone Mode",
        "content": [
            "• Uses OPF-based exact mapping for precise chapter matching",
            "• Translates chapters with matching names (ignores 'response_' prefix and extensions)",
            "• The regular translation logic uses this logic as well"
        ]
    },
    {
        "title": "🗑️ File Management",
        "content": [
            "• Delete Header Files: Removes translated_headers.txt files for all selected EPUBs",
            "• Use this to reset translation state or clean up after testing",
            "• Safe operation - only removes translation cache files, not original content"
        ]
    },
    {
        "title": "💡 Best Practices",
        "content": [
            "• Test with a small batch first to verify settings work correctly",
            "• Enable 'Save translations to .txt' for backup and debugging",
            "• Use 'Ignore header' if chapters already have translated visible titles",
            "• Keep 'Headers per batch' moderate to be within your output token limit"
        ]
    }
]

