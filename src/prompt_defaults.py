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

import re

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


# ---- {split_marker_instruction} (request merging) ----
SPLIT_MARKER_PLACEHOLDER = "{split_marker_instruction}"
SPLIT_MARKER_INSTRUCTION = "- CRITICAL Requirement: If you see any HTML tags containing 'SPLIT MARKER' (Example: <h1 id=\"split-1\">SPLIT MARKER: Do Not Remove This Tag</h1>), you MUST preserve them EXACTLY as they appear. Do not translate, modify, or remove these markers."


def apply_split_marker_instruction(prompt, instruction=""):
    """Fill ``{split_marker_instruction}`` where the user put it.

    A non-empty *instruction* replaces each placeholder in place. An empty one
    removes the placeholder: a line holding only the placeholder is dropped
    together with its line break, so the lines around it stay separate lines;
    an inline placeholder is removed and the words around it keep one space.
    """
    if not prompt or SPLIT_MARKER_PLACEHOLDER not in prompt:
        return prompt
    if instruction:
        return prompt.replace(SPLIT_MARKER_PLACEHOLDER, instruction)
    lines = prompt.splitlines(keepends=True)
    kept = []
    for line in lines:
        if line.strip() == SPLIT_MARKER_PLACEHOLDER:
            continue
        kept.append(re.sub(r'[ \t]*\{split_marker_instruction\}[ \t]*', _inline_gap, line))
    text = "".join(kept)
    if lines and lines[-1].strip() == SPLIT_MARKER_PLACEHOLDER and not prompt.endswith(("\n", "\r")):
        # the placeholder was the last line: drop the break that led to it
        text = text.rstrip("\r\n")
    return text


def _inline_gap(match):
    line = match.string
    before = line[:match.start()]
    after = line[match.end():]
    return " " if before.strip() and after.strip() else ""


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
        # the quote fix ran once; an older built-in default may still need its current lines
        return True if migrate_legacy_default_profiles(config['prompt_profiles']) else None

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
    return bool(migrate_legacy_default_profiles(profiles)) or updates_made


#: Lines an older built-in profile default saved into config.json -> the current default lines.
#: 2026-11-06: Manga_KR keeps Korean honorifics and family address terms the way the Korean text
#: profiles ask (it said "Retain honorifics ... in Romaji"), and its quotation-mark line ends in real
#: line breaks (it ended in a literal backslash-n pair).
LEGACY_DEFAULT_PROFILE_LINES = {
    "Manga_KR": (
        (
            "- Retain honorifics and onomatopoeia in Romaji.\n"
            "- Keep original Korean quotation marks (\" \", ' ', 「」, 『』) as-is without converting to English quotes.\\n\\n",
            "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
            "- Retain Korean familial address terms in romanized form rather than translating them (examples: oppa, eonni, hyung, noona, omma, appa, halabeoji, halmeoni), preserving their nuance and relationship context instead of converting them to English equivalents like brother, sister, mom, or dad.\n"
            "- Retain onomatopoeia in Romaji.\n"
            "- Keep original Korean quotation marks (\" \", ' ', 「」, 『』) as-is without converting to English quotes.\n\n",
        ),
    ),
}


def migrate_legacy_default_profiles(profiles):
    """Update built-in profiles that still hold an older default's exact lines (LEGACY_DEFAULT_PROFILE_LINES).

    Only the built-in profile of that name, and only lines that are still the old default text:
    a profile the user edited there, or any profile of their own, is left alone. Returns the names
    of the profiles changed (in place)."""
    changed = []
    if not isinstance(profiles, dict):
        return changed
    for name, replacements in LEGACY_DEFAULT_PROFILE_LINES.items():
        data = profiles.get(name)
        text = data if isinstance(data, str) else data.get('prompt', '') if isinstance(data, dict) else ''
        new_text = text
        for old, new in replacements:
            if isinstance(new_text, str) and old in new_text:
                new_text = new_text.replace(old, new)
        if new_text != text:
            if isinstance(data, dict):
                data['prompt'] = new_text
            else:
                profiles[name] = new_text
            changed.append(name)
            print(f"[Sanitizer] Updated the built-in '{name}' profile to its current default rules")
    return changed


# ---- AI prompt placeholders shown like links (display only) ----
# The desktop marks them in its prompt editors and shows what each says on hover
# (qt_widget_helpers.link_placeholders); mobile marks them and shows it in a sheet
# (glossarion_mobile.ui.components.placeholder_links). Nothing here changes a prompt or a request.

#: One ``{name}`` token.
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: The placeholders TransateKRtoEN replaces: in the translation system prompt (build_system_prompt, request
#: merging) and in the refinement prompts (_format_refinement_prompt; {html}/{content} in user prompts).
SYSTEM_PROMPT_PLACEHOLDERS = frozenset({'target_lang', 'split_marker_instruction'})
REFINEMENT_SYSTEM_PLACEHOLDERS = frozenset({'target_lang', 'QA_Issues', 'QA_ISSUES', 'qa_issues'})
REFINEMENT_USER_PLACEHOLDERS = REFINEMENT_SYSTEM_PLACEHOLDERS | {'html', 'content'}

#: The Chunk Prompt Template placeholders (TransateKRtoEN fills them for each chunk) and what they are.
TRANSLATION_CHUNK_PLACEHOLDER_NOTES = {
    'chunk_idx': "Current chunk number (1-based)",
    'total_chunks': "Total number of chunks",
}
#: The Image Chunk Prompt / Vision OCR prompt placeholders (image_translator fills them for each request).
IMAGE_CHUNK_PLACEHOLDER_NOTES = {
    'chunk_idx': "Current chunk number (1-based)",
    'total_chunks': "Total number of chunks",
    'context': "Additional context (e.g., chapter info)",
    'chunk_count': "Number of OCR chunks with text",
    'ocr_overlap_instruction': "Remove OCR-only duplicated text caused by the overlapping pixels between adjacent "
                               "image chunks.",
    'ocr_text': "Combined OCR text (translation prompt only)",
}
#: Placeholders filled with data of each request, by the prompt (config key) whose request replaces them:
#: what each is filled with.
PROMPT_INPUT_PLACEHOLDERS = {
    'rolling_summary_user_prompt': {
        'translations': "the translated text of the chapters being summarized, joined by \"---\" lines.",
    },
    'translation_chunk_prompt': dict(TRANSLATION_CHUNK_PLACEHOLDER_NOTES),
    'image_chunk_prompt': {name: IMAGE_CHUNK_PLACEHOLDER_NOTES[name]
                           for name in ('chunk_idx', 'total_chunks', 'context')},
    'vision_ocr_combined_context_prompt': {name: IMAGE_CHUNK_PLACEHOLDER_NOTES[name]
                                           for name in ('chunk_count', 'total_chunks', 'context',
                                                        'ocr_overlap_instruction')},
    'vision_ocr_translation_user_prompt': {name: IMAGE_CHUNK_PLACEHOLDER_NOTES[name]
                                           for name in ('context', 'ocr_text')},
    'vision_ocr_user_prompt': {
        'context': "the chapter/image context or alt text of the image being read (empty when none).",
    },
    'parallel_epub_glossary_wrapper_prompt': {
        'raw_text': "the raw chapter text of each mapped pair.",
        'translated_text': "the translated chapter text of each mapped pair.",
        'raw_filename': "the raw HTML file name of each mapped pair.",
        'translated_filename': "the translated HTML file name of each mapped pair.",
    },
}


def placeholder_names(names):
    """The names valid now: ``names`` is a callable (called now) or a collection."""
    try:
        value = names() if callable(names) else names
    except Exception:
        return frozenset()
    return value if isinstance(value, frozenset) else frozenset(value or ())


def placeholder_spans(text, names):
    """``[(start, end, name)]`` of each ``{name}`` in ``text`` whose name is valid (``names``)."""
    if not text or "{" not in text:
        return []
    valid = placeholder_names(names)
    if not valid:
        return []
    return [(m.start(), m.end(), m.group(1)) for m in PLACEHOLDER_RE.finditer(text) if m.group(1) in valid]


def described_placeholder_value(descriptions, name):
    """The value of a placeholder that holds data of each request (``descriptions``: name -> what it holds)."""
    return {'per_request': True, 'text': descriptions[name]}


def translation_placeholder_value(name, *, target_lang='', request_merging=False):
    """What a translation / refinement prompt placeholder says now (TransateKRtoEN replaces it), from the
    live ``target_lang`` and Request Merging state: a str or a value dict (see placeholder_value_parts)."""
    if name == 'target_lang':
        return str(target_lang or '')
    if name == 'split_marker_instruction':
        if not request_merging:
            return {'active': False,
                    'inactive': "Inactive right now: Request Merging is off, so its line is removed from the prompt."}
        return {'text': SPLIT_MARKER_INSTRUCTION,
                'extra': "Sent in requests that merge more than one chapter; removed from the others."}
    if name.lower() == 'qa_issues':
        return {'per_request': True,
                'text': "\"QA issue(s) to address: \" followed by the chapter's current character-found QA "
                        "issues; without QA issues its line is removed."}
    if name in ('html', 'content'):
        return {'per_request': True, 'text': "the translated HTML of the chapter being refined."}
    return ''


def assistant_placeholder_value(name, *, target_lang=''):
    """The Assistant Prompt's placeholders: only refinement requests replace them (_format_refinement_prompt
    without QA issues); translation requests send the prefill as written."""
    if name == 'target_lang':
        return {'text': str(target_lang or ''),
                'extra': "Replaced in refinement requests only; translation requests send {target_lang} as written."}
    return {'active': False,
            'inactive': "Inactive in refinement requests (no QA issues are given to the assistant prompt): its "
                        "line is removed. Translation requests send it as written."}


def placeholder_value_parts(value):
    """What one placeholder value shows (desktop hover tooltip, mobile value sheet): ``(text, notes)``.

    ``value`` is a str (its expansion) or a dict with ``text`` and the optional keys ``active`` (False: removed
    from the prompt now), ``inactive`` (what removed means there), ``per_request`` (``text`` describes data of
    each request), ``note`` (shown under an inactive value), ``extra`` (a short line under the value),
    ``error``. ``text`` is the expansion, verbatim ('' when the placeholder has none now); ``notes`` are the
    short lines shown muted.
    """
    if not isinstance(value, dict):
        text = str(value if value is not None else "")
        return text, ([] if text else ["(empty)"])
    text = str(value.get("text") or "")
    note = str(value.get("note") or "")
    if value.get("error"):
        return "", [f"The current value could not be read: {value['error']}"]
    if value.get("per_request"):
        return "", ["Filled in for each request: " + (text or note)]
    if value.get("active") is False:
        return "", [str(value.get("inactive") or "Inactive right now: removed from the prompt.")] + (
            [note] if note else [])
    extra = [str(value["extra"])] if value.get("extra") else []
    return text, (extra if text else ["(empty)"] + extra)
