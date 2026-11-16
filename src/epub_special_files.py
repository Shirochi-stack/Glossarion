"""Apply special-file rules to an EPUB's ordered document list."""

import os
import re
from pathlib import PurePosixPath


def special_file_flags(filenames, predicate, *, protect_interior=False):
    """Optionally protect interior documents and numbered trailing files.

    The original first and last non-special documents are the boundaries.
    Leading files keep their classification, even when their names have digits.
    """
    filenames = list(filenames)
    flags = [bool(predicate(filename)) for filename in filenames]
    if protect_interior:
        first = next((index for index, flag in enumerate(flags) if not flag), None)
        if first is not None:
            last = next(index for index in range(len(flags) - 1, first - 1, -1) if not flags[index])
            flags[first:last + 1] = [False] * (last - first + 1)
            # Only the original trailing block gets this numbered-file override.
            # Do not turn exempted files into new anchors that protect other tail files.
            for index in range(last + 1, len(flags)):
                stem = PurePosixPath(str(filenames[index]).replace("\\", "/")).stem
                if re.search(r"\d", stem):
                    flags[index] = False
    return flags


# ---------------------------------------------------------------------------
# The glossary extractor's special-file decision (extract_glossary_from_epub), shared so that a list of the
# files (the mobile Glossary progress) can ask the same question from the user's settings instead of the
# process environment.
# ---------------------------------------------------------------------------

#: The extractor's lists when ``SPECIAL_FILE_KEYWORDS`` / ``SPECIAL_FILE_EXACT`` are empty.
GLOSSARY_DEFAULT_SPECIAL_FILE_KEYWORDS = (
    'title', 'toc', 'copyright', 'preface', 'nav',
    'message', 'notice', 'colophon', 'dedication', 'epigraph',
    'foreword', 'acknowledgment', 'author', 'appendix', 'bibliography',
)
GLOSSARY_DEFAULT_SPECIAL_FILE_EXACT = ('index', 'glossary', 'glossary_extension', 'glossary_unified')


def glossary_special_file_settings(environ=None, *, include_special_files=False):
    """The settings the glossary extractor builds its document records with (its text-cache signature), read
    from ``environ`` (default ``os.environ``, the run's environment)."""
    env = os.environ if environ is None else environ
    keywords = env.get('SPECIAL_FILE_KEYWORDS', '') or ''
    exact = env.get('SPECIAL_FILE_EXACT', '') or ''
    translate_special = include_special_files or env.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
    return {
        'parser': 'html.parser',
        'translate_special_files': translate_special,
        'protect_interior_special_files': not translate_special and env.get(
            'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL', '1',
        ) == '1',
        'special_file_keywords': [] if translate_special else sorted(set(
            [k.strip().lower() for k in keywords.split(',') if k.strip()]
            if keywords else list(GLOSSARY_DEFAULT_SPECIAL_FILE_KEYWORDS)
        )),
        'special_file_exact': [] if translate_special else sorted(set(
            [k.strip().lower() for k in exact.split(',') if k.strip()]
            if exact else list(GLOSSARY_DEFAULT_SPECIAL_FILE_EXACT)
        )),
    }


def glossary_special_file_flags(filenames, signature):
    """Which of an EPUB's documents (reading order) glossary extraction skips as special files, for the
    settings ``signature`` (``glossary_special_file_settings``); extraction and numbering use the same flags."""
    def is_special(filename):
        if signature['translate_special_files']:
            return False
        name = os.path.splitext(os.path.basename(filename or ''))[0].lower()
        if not name:
            return False
        if name in signature['special_file_exact']:
            return True
        keywords = signature['special_file_keywords']
        if not any(keyword in name for keyword in keywords):
            return False
        # Preserve the configured matching rules for numbered variants.
        stripped = re.sub(r'\d+$', '', name).rstrip('_- ')
        return not re.search(r'\d', name) or any(keyword in stripped for keyword in keywords)

    return special_file_flags(
        filenames, is_special,
        protect_interior=signature['protect_interior_special_files'],
    )


def glossary_special_file_env(owner):
    """The special-file environment a run of ``owner`` gets from its settings: the rows
    ``run_env.initialize_environment_variables`` / ``_build_glossary_extraction_env`` export from the owner's
    start-up values (``translate_special_files_var`` …, seeded from the config like the desktop's)."""
    config = getattr(owner, 'config', None) or {}
    return {
        'TRANSLATE_SPECIAL_FILES': '1' if getattr(owner, 'translate_special_files_var', False) else '0',
        'SPECIAL_FILE_KEYWORDS': str(getattr(owner, 'special_file_keywords_var', '') or ''),
        'SPECIAL_FILE_EXACT': str(getattr(owner, 'special_file_exact_var', '') or ''),
        'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL': '1' if getattr(
            owner, 'never_consider_in_between_files_as_special_var',
            config.get('never_consider_in_between_files_as_special', True),
        ) else '0',
    }


def glossary_special_file_settings_for_owner(owner):
    """``glossary_special_file_settings`` from ``owner``'s settings rather than this process's environment."""
    return glossary_special_file_settings(glossary_special_file_env(owner))
