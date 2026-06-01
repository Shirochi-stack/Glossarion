# default_prompts.py
"""Default prompt access backed by translator_gui.py.

The desktop GUI's ``self.default_prompts`` block is the source of truth. This
module extracts that dictionary without importing the GUI, so Android helpers
can reuse the same prompt text without carrying a second copy.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from refinement_prompts import DEFAULT_REFINEMENT_SYSTEM_PROMPT
except Exception:
    DEFAULT_REFINEMENT_SYSTEM_PROMPT = ""


class _DefaultPromptTransformer(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "default_refinement_system_prompt"
        ):
            return ast.copy_location(ast.Constant(DEFAULT_REFINEMENT_SYSTEM_PROMPT), node)
        return self.generic_visit(node)


def _translator_gui_candidates():
    here = Path(__file__).resolve()
    yield here.parents[1] / "translator_gui.py"
    yield Path.cwd() / "translator_gui.py"
    if getattr(sys, "frozen", False):
        bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        yield bundle_dir / "translator_gui.py"


def _find_translator_gui():
    for candidate in _translator_gui_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find translator_gui.py for default prompts")


def _extract_default_prompts():
    translator_gui_path = _find_translator_gui()
    source = translator_gui_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(translator_gui_path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "default_prompts"
            ):
                prompt_node = _DefaultPromptTransformer().visit(node.value)
                ast.fix_missing_locations(prompt_node)
                return ast.literal_eval(prompt_node)

    raise ValueError("Could not find self.default_prompts in translator_gui.py")


def get_default_prompts():
    return _extract_default_prompts().copy()


DEFAULT_PROMPTS = get_default_prompts()


def get_prompt(profile_name, target_lang="English"):
    """Get the default prompt for a profile, with placeholders replaced."""
    prompt = DEFAULT_PROMPTS.get(profile_name, "")
    if not prompt:
        return ""

    prompt = prompt.replace("{target_lang}", target_lang)
    prompt = re.sub(r"\s*\{split_marker_instruction\}\s*", "\n", prompt)

    while "\n\n\n" in prompt:
        prompt = prompt.replace("\n\n\n", "\n\n")

    return prompt.strip()
