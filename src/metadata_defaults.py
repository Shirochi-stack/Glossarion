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
