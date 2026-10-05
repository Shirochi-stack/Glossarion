"""Built-in default prompt texts shared by desktop and mobile.

Moved verbatim from translator_gui.py: the ``default_*`` prompt attributes
assigned in ``TranslatorGUI.__init__`` (chunk / image-chunk / Vision OCR) and in
``_init_default_prompts`` (rolling summary, assistant prefill), plus the core of
``_sanitize_config_prompts``. translator_gui assigns these constants to the
same attributes at the same points, so attribute values are unchanged.

Still in translator_gui (source-checked by tests/test_sdlxliff_support.py and
tests/test_subtitle_processor.py): the ``default_prompts`` profile dict, the
protected-profile set and ``always_include_profiles``. The image-only title
tag prompt stays in title_tag_translation (DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT).

GUI-free; must stay importable on Python 3.10 without Qt.
"""

# ---- TranslatorGUI.__init__ ("# Default prompts") ----
DEFAULT_TRANSLATION_CHUNK_PROMPT = "[This is part {chunk_idx}/{total_chunks}]. You must maintain the narrative flow with the previous chunks while following all system prompt guidelines previously mentioned."
DEFAULT_IMAGE_CHUNK_PROMPT = "This is part {chunk_idx} of {total_chunks} of a longer image. You must maintain the narrative flow with the previous chunks while following all system prompt guidelines previously mentioned. {context}"
DEFAULT_VISION_OCR_PROMPT = (
    "Extract all readable text that is physically present in the image, in natural reading order. Return Markdown only, not HTML. "
    "Output plain text by default. Use Markdown only to preserve visible source structure or styling when it is actually present in the image: paragraph breaks, meaningful line breaks, bullet lists, numbered lists, blockquotes, tables, bold, italic, strikethrough/deleted text, inline code/code blocks, or visibly printed Markdown characters. "
    "Do not invent Markdown formatting. "
    "If the image is primarily cover art, character art, scene illustration, splash art, decorative art, a poster, or a promotional image, reply exactly No when the only readable text is a logo, watermark, title/author/credit text, short decorative words, background writing, or other incidental non-story text. "
    "Do not OCR incidental text from illustrated covers or splash images. "
    "If the image is primarily a text page, title page, chapter title page, document/table/list page, speech-bubble comic page, or mostly blank page with readable non-decorative text, output the readable text. "
    "Reply exactly No only when there is no readable text, or when the image is illustration/decorative/cover art whose readable text is only incidental. "
    "Do not reproduce every visual wrap from the image; merge wrapped lines that belong to the same sentence or paragraph unless the line break is semantically intentional. "
    "Preserve visible textual marks when possible, including brackets, parentheses, quote marks, symbols, and emotes/emoticons. "
    "For Chinese/Japanese/Korean text with small pronunciation guides above or beside the main characters, OCR only the main/base characters and ignore the pronunciation guides. "
    "For pinyin-over-Chinese images, output the Chinese characters only; do not output the pinyin unless the pinyin is standalone text with no matching Chinese base text. "
    "Do not translate, summarize, explain, annotate, transliterate, romanize, or add pronunciation guides. "
    "Do not output duplicate reading lines such as pinyin, romaji, furigana, Jyutping, or Latin readings when they are attached to the same base text."
)
DEFAULT_VISION_OCR_USER_PROMPT = (
    "OCR this image/chunk. Return Markdown only with the literal main/base source text. "
    "If this is primarily cover/illustration/splash/decorative art and the readable text is only logo, watermark, title/author/credit text, short decorative words, or background writing, reply exactly No. "
    "If this is primarily a text/title/chapter/document/comic page with readable non-decorative text, output it. Reply exactly No only when there is no readable text or only incidental cover/illustration text. "
    "Ignore pinyin/romaji/furigana/Jyutping pronunciation guides attached to base characters. Do not translate."
    "\n\nContext:\n{context}"
)
DEFAULT_VISION_OCR_COMBINED_CONTEXT_PROMPT = (
    "The Markdown OCR text below was assembled from {chunk_count} tall-image chunk(s). "
    "Translate it as one continuous passage, preserving narrative flow and Markdown structure. {ocr_overlap_instruction}"
)
DEFAULT_VISION_OCR_TRANSLATION_USER_PROMPT = (
    "{context}\n\n"
    "Translate the following Markdown OCR text according to the system prompt. "
    "Return only the translated text. Preserve the Markdown paragraph, heading, list, table, blockquote, emphasis, and line-break structure.\n\n"
    "<OCR_TEXT>\n{ocr_text}\n</OCR_TEXT>"
)

# ---- TranslatorGUI._init_default_prompts ----
DEFAULT_ROLLING_SUMMARY_SYSTEM_PROMPT = """You are a context summarization assistant. Create concise, informative summaries that preserve key story elements for translation continuity."""

# Default assistant prompt (empty by default - user can optionally set this to prefill)
DEFAULT_ASSISTANT_PROMPT = ""

DEFAULT_ROLLING_SUMMARY_USER_PROMPT = """Analyze the recent translation exchanges and create a structured summary for context continuity.

Focus on extracting and preserving:
1. **Character Information**: Names (with original forms), relationships, roles, and important character developments
2. **Plot Points**: Key events, conflicts, and story progression
3. **Locations**: Important places and settings
4. **Terminology**: Special terms, abilities, items, or concepts (with original forms)
5. **Tone & Style**: Writing style, mood, and any notable patterns
6. **Unresolved Elements**: Questions, mysteries, or ongoing situations

Format the summary clearly with sections. Be concise but comprehensive.

Recent translations to summarize:
{translations}
        """


def sanitize_prompt_profiles(config):
    """Auto-fix known issues in user prompts from older versions.

    Returns None when nothing ran (no ``prompt_profiles`` or already sanitized).
    Otherwise fixes the profiles in place, sets ``sanitization_korean_quotes_fixed``
    and returns whether any profile text changed; the caller then persists the
    config (desktop saves even when only the flag changed).
    """
    if 'prompt_profiles' not in config:
        return None

    # Check if already ran
    if config.get('sanitization_korean_quotes_fixed', False):
        return None

    updates_made = False
    profiles = config['prompt_profiles']

    # The specific broken pattern (missing the double quote pair)
    # We look for the substring where " " is missing before the comma
    broken_fragment = "Korean quotation marks (, ' ', 「」, 『』)"
    fixed_fragment = "Korean quotation marks (\" \", ' ', 「」, 『』)"

    for profile_name, profile_data in profiles.items():
        # profile_data can be a string or a dict
        prompt_text = ""
        if isinstance(profile_data, str):
            prompt_text = profile_data
        elif isinstance(profile_data, dict):
            prompt_text = profile_data.get('prompt', '')

        if broken_fragment in prompt_text:
            fixed_text = prompt_text.replace(broken_fragment, fixed_fragment)

            if isinstance(profile_data, str):
                profiles[profile_name] = fixed_text
            elif isinstance(profile_data, dict):
                profiles[profile_name]['prompt'] = fixed_text

            updates_made = True
            print(f"[Sanitizer] Fixed malformed Korean quotes in profile: {profile_name}")

    # Always set flag to avoid re-running
    config['sanitization_korean_quotes_fixed'] = True
    config['prompt_profiles'] = profiles
    return updates_made
