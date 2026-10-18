"""Unicode script ranges per writing system and the scripts each target language allows.

Shared by the QA scanner (``scan_html_folder`` non-target-language check) and the SDLXLIFF
review editor (unchanged text that is already in the output language is not "untranslated").
GUI-free, standard library only, Python 3.10 compatible.
"""

# Non-Latin scripts the QA scanner can report as foreign text.
SCRIPT_RANGES = [
    (0xAC00, 0xD7AF, 'Korean'), (0x1100, 0x11FF, 'Korean'),
    (0x3130, 0x318F, 'Korean'), (0xA960, 0xA97F, 'Korean'),
    (0xD7B0, 0xD7FF, 'Korean'), (0x3040, 0x309F, 'Japanese'),
    (0x30A0, 0x30FF, 'Japanese'), (0x31F0, 0x31FF, 'Japanese'),
    (0xFF65, 0xFF9F, 'Japanese'), (0x4E00, 0x9FFF, 'Chinese'),
    (0x3400, 0x4DBF, 'Chinese'), (0x20000, 0x2A6DF, 'Chinese'),
    (0x2A700, 0x2B73F, 'Chinese'), (0x0590, 0x05FF, 'Hebrew'),
    (0x0600, 0x06FF, 'Arabic'), (0x0700, 0x074F, 'Syriac'),
    (0x0750, 0x077F, 'Arabic'), (0x0E00, 0x0E7F, 'Thai'),
    (0x0400, 0x04FF, 'Cyrillic'), (0x0500, 0x052F, 'Cyrillic'),
]

# Latin letters, including the fullwidth forms CJK books use for English (ＹＯＵ).
LATIN_RANGES = [
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF),
    (0xFF21, 0xFF3A), (0xFF41, 0xFF5A),
]

# Define which scripts are allowed for each target language
# Latin-based languages allow basic Latin + extensions
LANGUAGE_SCRIPTS = {
    'english': [],  # Only Latin (default - everything else is foreign)
    'spanish': [],  # Latin with Spanish diacritics
    'french': [],   # Latin with French diacritics
    'german': [],   # Latin with German diacritics
    'portuguese': [],  # Latin with Portuguese diacritics
    'italian': [],  # Latin with Italian diacritics
    'russian': ['Cyrillic'],  # Cyrillic only
    'japanese': ['Japanese', 'Chinese'],  # Japanese uses kanji too
    'korean': ['Korean'],
    'chinese': ['Chinese'],
    'arabic': ['Arabic', 'Syriac'],
    'hebrew': ['Hebrew'],
    'thai': ['Thai'],
}

# Borrowed scripts that count as the language's own only next to its main script:
# Hanja inside Korean text. (Kanji inside Japanese text is already allowed above.)
BORROWED_SCRIPTS = {'korean': ['Chinese']}


def script_of_char(char):
    """'Latin', a ``SCRIPT_RANGES`` name, or None for anything else (digits, punctuation, Greek...)."""
    code_point = ord(char)
    for start, end in LATIN_RANGES:
        if start <= code_point <= end:
            return 'Latin'
    for start, end, script_name in SCRIPT_RANGES:
        if start <= code_point <= end:
            return script_name
    return None


def language_script_key(language):
    """Normalise an output-language setting ("Chinese (Traditional)", "Simplified Chinese")
    to a ``LANGUAGE_SCRIPTS`` key, or '' when the language is unknown."""
    name = str(language or '').strip().lower()
    if name in LANGUAGE_SCRIPTS:
        return name
    for key in LANGUAGE_SCRIPTS:
        if key in name:
            return key
    return ''


def text_is_in_language_script(text, language):
    """True when every letter of ``text`` is written in ``language``'s script, e.g. English
    words kept in a Korean source when the output is English.

    A language's own main script must appear at least once, so a line that is only Hanja
    or only kanji (which may be untranslated Chinese) does not count for Korean or Japanese.
    """
    key = language_script_key(language)
    if not key:
        return False
    own = LANGUAGE_SCRIPTS[key] or ['Latin']
    allowed = set(own) | set(BORROWED_SCRIPTS.get(key, ()))
    has_main = False
    has_letter = False
    for char in str(text or ''):
        if not char.isalpha():
            continue
        has_letter = True
        script = script_of_char(char)
        if script not in allowed:
            return False
        if script == own[0]:
            has_main = True
    return has_letter and has_main
