"""Desktop GUI source corpus for tests that read translator_gui.py.

The mobile rewrite (milestone U2) moved TranslatorGUI methods verbatim into shared
GUI-free mixins that TranslatorGUI inherits (``settings_persistence``, ``run_env``,
``owner_state``) and the Direct Text run attributes into ``headless_owner``; U3 moved
the per-file job runners (``text_jobs``), input preparation (``input_preparation``),
the job hooks (``job_runner``), the stop protocol (``stop_control``) and the translation /
glossary pipelines (``translation_pipeline``: run set-up + worker, run_translation_direct,
QA/multipass planning, glossary extraction and auto-loading) and the Direct Text dialog's
chat persistence / log-stream model (``direct_text_store``, ``direct_text_stream``). Tests
that grep translator_gui.py for a code fragment, or look a TranslatorGUI method up by
AST, search the whole corpus instead so they keep checking the same code wherever
it now lives.

    from _src_corpus import desktop_gui_source, find_method, method_source

``find_method(name)`` follows TranslatorGUI's precedence: its own class body first,
then the shared mixins in MRO order.
"""

from __future__ import annotations

import ast
import textwrap
from functools import lru_cache
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

#: translator_gui plus the shared modules holding code that moved out of it (MRO order).
DESKTOP_GUI_MODULES = ("translator_gui", "translation_pipeline", "text_jobs", "input_preparation", "job_runner", "settings_persistence",
                       "run_env", "owner_state", "headless_owner", "stop_control", "direct_text_store",
                       "direct_text_stream")

#: Classes searched by find_method, in TranslatorGUI's method resolution order.
DESKTOP_GUI_CLASSES = (
    ("translator_gui", "TranslatorGUI"),
    ("translation_pipeline", "TranslationPipelineMixin"),
    ("translation_pipeline", "GlossaryPipelineMixin"),
    ("translation_pipeline", "PipelineHooksMixin"),
    ("text_jobs", "TextJobsMixin"),
    ("input_preparation", "InputPreparationMixin"),
    ("job_runner", "JobHooksMixin"),
    ("settings_persistence", "SettingsPersistenceMixin"),
    ("run_env", "RunEnvMixin"),
    ("owner_state", "ConfigStateMixin"),
)

#: Methods holding statements that used to be inline in TranslatorGUI.__init__.
INIT_BLOCK_METHODS = (
    ("translator_gui", "TranslatorGUI", "__init__"),
    ("owner_state", "ConfigStateMixin", "_init_config_state"),
    ("owner_state", "ConfigStateMixin", "_init_default_prompt_profiles"),
    ("owner_state", "ConfigStateMixin", "_init_watchdog_dir"),
    ("owner_state", "ConfigStateMixin", "_auto_encrypt_api_keys"),  # last __init__ step
)


@lru_cache(maxsize=None)
def module_source(module: str) -> str:
    """src/<module>.py decoded with utf-8-sig (translator_gui has a BOM), LF line endings."""
    return (SRC_DIR / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


@lru_cache(maxsize=None)
def module_tree(module: str) -> ast.Module:
    return ast.parse(module_source(module))


def desktop_gui_files() -> list:
    return [SRC_DIR / f"{m}.py" for m in DESKTOP_GUI_MODULES if (SRC_DIR / f"{m}.py").exists()]


def desktop_gui_source() -> str:
    """translator_gui.py followed by the shared modules (one string, LF)."""
    return "\n".join(module_source(m) for m in DESKTOP_GUI_MODULES if (SRC_DIR / f"{m}.py").exists())


def _class_node(module: str, class_name: str):
    if not (SRC_DIR / f"{module}.py").exists():
        return None
    for node in module_tree(module).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def find_method_with_owner(name: str):
    """(module, class_name, FunctionDef) of the definition TranslatorGUI resolves *name* to."""
    for module, class_name in DESKTOP_GUI_CLASSES:
        cls = _class_node(module, class_name)
        if cls is None:
            continue
        found = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if found:
            return module, class_name, found[-1]
    raise KeyError(f"{name} is not defined by TranslatorGUI or its shared mixins")


def find_method(name: str):
    """FunctionDef of the definition TranslatorGUI resolves *name* to."""
    return find_method_with_owner(name)[2]


def method_source(name: str) -> str:
    """Dedented source text of the method TranslatorGUI resolves *name* to."""
    module, _cls, node = find_method_with_owner(name)
    lines = module_source(module).split("\n")
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))


def init_statements() -> list:
    """Top-level statements of TranslatorGUI.__init__ plus the init blocks moved out of it."""
    out = []
    for module, class_name, method in INIT_BLOCK_METHODS:
        cls = _class_node(module, class_name)
        if cls is None:
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == method:
                out.extend(node.body)
    return out


__all__ = [
    "DESKTOP_GUI_CLASSES",
    "DESKTOP_GUI_MODULES",
    "INIT_BLOCK_METHODS",
    "SRC_DIR",
    "desktop_gui_files",
    "desktop_gui_source",
    "find_method",
    "find_method_with_owner",
    "init_statements",
    "method_source",
    "module_source",
    "module_tree",
]
