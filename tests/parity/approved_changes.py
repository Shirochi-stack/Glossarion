"""Owner-approved desktop default changes made after the goldens and frozen literals were captured.

The parity tests replay the desktop as it was at their pinned commits (``git show <sha>:...``)
and compare it with recorded goldens. A deliberate default change (DISCREPANCIES.md: "each
needs a separate, user-approved change that also updates the goldens") is applied here, to the
legacy source the tests read and to the goldens they load, so every other behaviour is still
checked exactly. Each entry names the source rows and the recorded values it changes.
"""
from __future__ import annotations


def _after(line, *added):
    """Insert *added* lines after *line* (both without their newline)."""
    return (line + "\n", line + "\n" + "".join(text + "\n" for text in added))


# 2026-10-17 "Always check and respect Retry-After" (respect_retry_after / RESPECT_RETRY_AFTER)
_RESPECT_SETTINGS_ROW = _after(
    "            ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),",
    "            ('respect_retry_after', ['respect_retry_after_var'], True, bool),")
_RESPECT_SETTINGS_ROW_GUI = _after(
    "                ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),",
    "                ('respect_retry_after', ['respect_retry_after_var'], True, bool),")
_RESPECT_BOOL_ROW = _after(
    "            ('indefinite_rate_limit_retry_var', 'indefinite_rate_limit_retry', False),",
    "            ('respect_retry_after_var', 'respect_retry_after', True),")
_RESPECT_MIGRATION = _after(
    "            setattr(self, var_name, create_var(bool, key, default))\n"
    "        self.use_header_as_output_var = False",
    "        if self.respect_retry_after_var and getattr(self, 'ignore_retry_after_var', False):",
    "            # 'Always check and respect Retry-After' contradicts 'Ignore server Retry-After header':",
    "            # a config holding both keeps the explicit ignore choice.",
    "            self.respect_retry_after_var = False")
_RESPECT_ENV_DICT = _after(
    "            'INDEFINITE_RATE_LIMIT_RETRY': '1' if self.config.get('indefinite_rate_limit_retry', False) else '0',",
    "            'RESPECT_RETRY_AFTER': '1' if getattr(self, 'respect_retry_after_var', True) else '0',")
_RESPECT_ENV_PAIR = _after(
    "                ('INDEFINITE_RATE_LIMIT_RETRY', '1' if getattr(self, 'indefinite_rate_limit_retry_var', False) else '0'),",
    "                ('RESPECT_RETRY_AFTER', '1' if getattr(self, 'respect_retry_after_var', True) else '0'),")

# 2026-10-18 per-category Gemini safety thresholds (gemini_safety_category_thresholds /
# GEMINI_SAFETY_CATEGORY_THRESHOLDS)
_SAFETY_CATEGORY_ROW = _after(
    "            ('gemini_safety_threshold', ['gemini_safety_threshold_var'], 'BLOCK_NONE', str),",
    "            ('gemini_safety_category_thresholds', ['gemini_safety_category_thresholds_var'], {}, dict),")
_SAFETY_CATEGORY_ROW_GUI = _after(
    "                ('gemini_safety_threshold', ['gemini_safety_threshold_var'], 'BLOCK_NONE', str),",
    "                ('gemini_safety_category_thresholds', ['gemini_safety_category_thresholds_var'], {}, dict),")
_SAFETY_CATEGORY_ENV_DICT = _after(
    "            'GEMINI_SAFETY_THRESHOLD': str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE')),",
    "            'GEMINI_SAFETY_CATEGORY_THRESHOLDS': json.dumps(self.config.get('gemini_safety_category_thresholds') or {}),")
_SAFETY_CATEGORY_ENV_PAIR = _after(
    "                ('GEMINI_SAFETY_THRESHOLD', str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE'))),",
    "                ('GEMINI_SAFETY_CATEGORY_THRESHOLDS', json.dumps(self.config.get('gemini_safety_category_thresholds') or {})),")

# 2026-10-18 {split_marker_instruction} stays where the user put it: filled in place, or its own
# line removed without joining the lines around it (prompt_defaults.apply_split_marker_instruction)
_SPLIT_MARKER_OLD_SUB = ("re.sub(r'\\s*\\{split_marker_instruction\\}\\s*', "
                         "lambda m: '\\n' + split_instr if split_instr else '', system_prompt)")


def _split_marker_fill(indent):
    return (indent + "# Use a regex to replace the placeholder regardless of surrounding whitespace or newlines\n"
            + indent + "import re\n"
            + indent + "system_prompt = " + _SPLIT_MARKER_OLD_SUB + "\n",
            indent + "# Fill the placeholder where the user put it; without an instruction its line goes away\n"
            + indent + "from prompt_defaults import apply_split_marker_instruction\n"
            + indent + "system_prompt = apply_split_marker_instruction(system_prompt, split_instr)\n")


_SPLIT_MARKER_TEXT_FILE = _split_marker_fill(" " * 16)   # _process_text_file (text_jobs)
_SPLIT_MARKER_IMAGE_FILE = _split_marker_fill(" " * 12)  # _process_image_file (image_job)
_SPLIT_MARKER_RPGMAKER = (                               # _process_rpgmaker_game (rpgmaker_job)
    "            import re\n"
    "            system_prompt = re.sub(r'\\s*\\{split_marker_instruction\\}\\s*', '', system_prompt)\n",
    "            from prompt_defaults import apply_split_marker_instruction\n"
    "            system_prompt = apply_split_marker_instruction(system_prompt, \"\")\n")

APPROVED = (
    {
        "date": "2026-10-13",
        "what": "Skip .pdf book title translation is on by default (like .txt titles)",
        "source": {
            "src/owner_state.py": [
                ("('skip_pdf_title_translation_var', 'skip_pdf_title_translation', False)",
                 "('skip_pdf_title_translation_var', 'skip_pdf_title_translation', True)"),
            ],
            "src/translator_gui.py": [
                ("('skip_pdf_title_translation_var', 'skip_pdf_title_translation', False)",
                 "('skip_pdf_title_translation_var', 'skip_pdf_title_translation', True)"),
                ("('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], False, bool)",
                 "('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], True, bool)"),
                # run_env's environment builders lived in translator_gui before U2
                ("getattr(self, 'skip_pdf_title_translation_var', False)",
                 "getattr(self, 'skip_pdf_title_translation_var', True)"),
            ],
            "src/settings_persistence.py": [
                ("('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], False, bool)",
                 "('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], True, bool)"),
            ],
            "src/other_settings.py": [
                ("self.config.get('skip_pdf_title_translation', False)", "self.config.get('skip_pdf_title_translation', True)"),
                ("self.config['skip_pdf_title_translation'] = False", "self.config['skip_pdf_title_translation'] = True"),
            ],
            "src/run_env.py": [
                ("getattr(self, 'skip_pdf_title_translation_var', False)",
                 "getattr(self, 'skip_pdf_title_translation_var', True)"),
            ],
            "src/progress_core.py": [
                ("config.get('skip_pdf_title_translation', False)", "config.get('skip_pdf_title_translation', True)"),
            ],
            "src/TransateKRtoEN.py": [
                ("os.getenv('SKIP_PDF_TITLE_TRANSLATION', '0')", "os.getenv('SKIP_PDF_TITLE_TRANSLATION', '1')"),
            ],
        },
        # recorded values (config key, owner attribute, exported env var): old -> new
        "values": {
            "skip_pdf_title_translation": (False, True),
            "skip_pdf_title_translation_var": (False, True),
            "SKIP_PDF_TITLE_TRANSLATION": ("0", "1"),
        },
    },
    {
        "date": "2026-10-17",
        "what": "New 'Always check and respect Retry-After' toggle (on by default, below Indefinite Rate "
                "Limit Retry); never on together with 'Ignore server Retry-After header'",
        "source": {
            "src/settings_persistence.py": [_RESPECT_SETTINGS_ROW],
            "src/owner_state.py": [_RESPECT_BOOL_ROW, _RESPECT_MIGRATION],
            "src/translator_gui.py": [_RESPECT_SETTINGS_ROW_GUI, _RESPECT_BOOL_ROW, _RESPECT_MIGRATION,
                                      _RESPECT_ENV_DICT, _RESPECT_ENV_PAIR],
            "src/run_env.py": [_RESPECT_ENV_DICT, _RESPECT_ENV_PAIR],
        },
        # a new key: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-18",
        "what": "Per-category Gemini safety thresholds (Other Settings › API Safety Settings); a "
                "category set to 'Same as threshold' follows the main Threshold",
        "source": {
            "src/settings_persistence.py": [_SAFETY_CATEGORY_ROW],
            "src/translator_gui.py": [_SAFETY_CATEGORY_ROW_GUI, _SAFETY_CATEGORY_ENV_DICT, _SAFETY_CATEGORY_ENV_PAIR],
            "src/run_env.py": [_SAFETY_CATEGORY_ENV_DICT, _SAFETY_CATEGORY_ENV_PAIR],
        },
        # a new key: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-18",
        "what": "Prompt placeholders stay where the user put them: {split_marker_instruction} is filled "
                "in place, and without an instruction its line is removed without joining its neighbours",
        "source": {
            # the runners lived in translator_gui before U3 / U7
            "src/translator_gui.py": [_SPLIT_MARKER_TEXT_FILE, _SPLIT_MARKER_IMAGE_FILE, _SPLIT_MARKER_RPGMAKER],
            "src/text_jobs.py": [_SPLIT_MARKER_TEXT_FILE],
            "src/image_job.py": [_SPLIT_MARKER_IMAGE_FILE],
            "src/rpgmaker_job.py": [_SPLIT_MARKER_RPGMAKER],
        },
        # prompt text only (the logged preview / length follow from the source change)
        "values": {},
    },
)


def apply_to_source(relpath: str, text: str) -> str:
    """Legacy source ``relpath`` (as ``git show`` returns it) with the approved changes applied."""
    relpath = str(relpath).replace("\\", "/")
    for change in APPROVED:
        for old, new in change["source"].get(relpath, ()):
            text = text.replace(old, new)
    return text


def _changed(value, old, new):
    if value == old and type(value) is type(old):
        return new
    if isinstance(value, str) and value == str(old):  # values recorded as text
        return str(new)
    return value


def apply_to_golden(data):
    """A loaded golden (nested dicts / lists) with the approved recorded values changed."""
    values = {}
    for change in APPROVED:
        values.update(change["values"])

    def walk(node):
        if isinstance(node, dict):
            return {key: (_changed(walk(value), *values[key]) if key in values else walk(value))
                    for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        return node

    return walk(data)
