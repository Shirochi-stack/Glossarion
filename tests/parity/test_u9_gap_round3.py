"""U9 gap closures (audit round 3): desktop logic moved into shared GUI-free modules.

The oracle is the desktop at ``U9_BASE_SHA`` (main 96da1ec6, the parent of the moves), read with
``git show``; the frozen code runs against the shared function (and the rewired desktop code) on the
same inputs.

* GlossaryManager_GUI ``_setup_manual_glossary_tab`` › ``_update_glossary_compression`` (the
  Balanced/Full "Compression Factor" + "Auto" closure) -> ``settings_rules.glossary_output_limit`` /
  ``glossary_auto_compression_factor`` (the closure keeps its widget work, in the same order);
  ``settings_rules.apply_glossary_auto_compression`` is its config side for the mobile app;
* metadata_batch_translator ``_reset_all_prompts_to_defaults``'s key list ->
  ``metadata_defaults.METADATA_PROMPT_RESET_KEYS``; its config side (remove those keys, blank
  ``book_title_prompt``, re-seed the defaults) -> ``metadata_defaults.reset_metadata_prompts``, which the
  desktop method calls (the frozen and the live method run on the same configs);
* the new ``settings_rules`` lock / change rules for the compression factors run the desktop's own
  threshold tables (ConfigStateMixin._update_auto_compression_factor and the Glossary Manager closure).
"""

from __future__ import annotations

import ast
import copy
import itertools
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

U9_BASE_SHA = "96da1ec6ccb34828c3ebd6e7f1c842796639a8e8"


def frozen(relpath: str) -> str:
    try:
        raw = subprocess.check_output(["git", "show", f"{U9_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                                      stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(f"git show {U9_BASE_SHA}:{relpath} unavailable: {exc} (CI must fetch the base commit)")
        pytest.skip(f"git show {U9_BASE_SHA}:{relpath} unavailable: {exc}")
    return raw.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")


def current(name: str) -> str:
    return (SRC / name).read_bytes().decode("utf-8").lstrip("﻿").replace("\r\n", "\n")


def _method(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name][0]


def _nested(method: ast.FunctionDef, name: str) -> ast.FunctionDef:
    return [n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n.name == name][0]


# ---------------------------------------------------------------------------
# Glossary Manager: Compression Factor + Auto
# ---------------------------------------------------------------------------

class _Entry:
    def __init__(self, text: str = "") -> None:
        self._text = text
        self.calls: list = []

    def text(self) -> str:
        return self._text

    def setText(self, value) -> None:  # noqa: N802 - Qt API
        self.calls.append(("setText", value))
        self._text = value

    def setEnabled(self, value) -> None:  # noqa: N802 - Qt API
        self.calls.append(("setEnabled", value))


class _Check:
    def __init__(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:  # noqa: N802 - Qt API
        return self.checked


def _closure(source: str, namespace: dict):
    """``_update_glossary_compression`` of ``_setup_manual_glossary_tab`` as ``make(self) -> closure``."""
    method = _method(source, "_setup_manual_glossary_tab")
    node = _nested(method, "_update_glossary_compression")
    lines = source.splitlines()
    body = textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))
    code = "def make(self):\n" + textwrap.indent(body, "    ") + "\n    return _update_glossary_compression\n"
    exec(compile(code, "<_update_glossary_compression>", "exec"), namespace)
    return namespace["make"]


class _Owner:
    def __init__(self, limit_text: str, auto: bool, max_tokens) -> None:
        self.glossary_output_token_limit_entry = _Entry(limit_text)
        self.token_limit_helper = _Entry()
        self.glossary_auto_compression_checkbox = _Check(auto)
        self.glossary_compression_factor_entry = _Entry("1.0")
        if max_tokens is not None:
            self.max_output_tokens = max_tokens

    def record(self) -> tuple:
        return (tuple(self.token_limit_helper.calls), tuple(self.glossary_compression_factor_entry.calls))


LIMIT_TEXTS = ("-1", "0", "1", "16378", "16379", "32768", "32769", "65535", "65536", "128000", "abc", "", " 64 ",
               "1e3", "-2", "999999999")
MAX_TOKENS = (None, 8000, 16378, 16379, 32769, 65536, 128000)


def test_glossary_compression_closure_is_the_shared_rule():
    import settings_rules

    printed: list = []
    legacy_make = _closure(frozen("src/GlossaryManager_GUI.py"), {"print": printed.append})
    new_make = _closure(current("GlossaryManager_GUI.py"), {"print": printed.append, "settings_rules": settings_rules})
    cases = 0
    for limit, auto, tokens in itertools.product(LIMIT_TEXTS, (True, False), MAX_TOKENS):
        legacy, new = _Owner(limit, auto, tokens), _Owner(limit, auto, tokens)
        legacy_make(legacy)()
        new_make(new)()
        assert new.record() == legacy.record(), (limit, auto, tokens)
        # the shared helpers alone give the same helper label and factor
        actual, helper = settings_rules.glossary_output_limit(limit, tokens if tokens is not None else 65536)
        assert ("setText", helper) == legacy.token_limit_helper.calls[-1]
        if auto:
            factor = settings_rules.glossary_auto_compression_factor(actual)
            assert legacy.glossary_compression_factor_entry.calls == [("setEnabled", False), ("setText", str(factor))]
        cases += 1
    assert cases == len(LIMIT_TEXTS) * 2 * len(MAX_TOKENS)
    assert not printed  # no case raised inside the closure


def test_apply_glossary_auto_compression_is_the_closure_then_the_save():
    """Mobile: the config the Glossary Manager's save writes after the closure ran
    (``config['glossary_compression_factor'] = float(entry.text())``)."""
    import settings_rules

    printed: list = []
    legacy_make = _closure(frozen("src/GlossaryManager_GUI.py"), {"print": printed.append})
    for limit, auto, tokens in itertools.product((-1, "-1", 8000, 20000, 40000, 70000, "abc"), (True, False, None),
                                                 (8000, 30000, 65536, 200000)):
        config = {"max_output_tokens": tokens, "glossary_max_output_tokens": limit, "glossary_compression_factor": 2.5}
        if auto is not None:
            config["glossary_auto_compression"] = auto
        owner = _Owner(str(limit), True if auto is None else auto, tokens)
        legacy_make(owner)()
        expected = float(owner.glossary_compression_factor_entry.text())
        result = settings_rules.apply_glossary_auto_compression(config)
        if auto is False:
            assert result is None and config["glossary_compression_factor"] == 2.5
        else:
            assert result == expected and config["glossary_compression_factor"] == expected
    assert not printed


def test_glossary_compression_lock_and_change_rules():
    import settings_rules

    assert settings_rules.evaluate("lock:glossary_compression_factor", {})  # Auto starts ticked
    assert settings_rules.evaluate("lock:glossary_compression_factor", {"glossary_auto_compression": False}) == ""
    config = {"max_output_tokens": 20000, "glossary_compression_factor": 3.0}
    changed, env = settings_rules.apply_change(config, "glossary_max_output_tokens", -1)
    assert changed == {"glossary_max_output_tokens": -1, "glossary_compression_factor": 1.2} and env == {}
    changed, _env = settings_rules.apply_change(config, "glossary_auto_compression", False)
    assert changed == {"glossary_auto_compression": False}


# ---------------------------------------------------------------------------
# Main compression factor (ConfigStateMixin._update_auto_compression_factor)
# ---------------------------------------------------------------------------

def test_auto_compression_lock_and_change_rules_use_the_desktop_table():
    import settings_rules
    from owner_state import ConfigStateMixin

    assert settings_rules.evaluate("lock:compression_factor", {})
    assert settings_rules.evaluate("lock:compression_factor", {"auto_compression_factor": False}) == ""
    for tokens in (1000, 16378, 16379, 32768, 32769, 65535, 65536, 128000):
        config = {"auto_compression_factor": True}
        settings_rules.apply_change(config, "max_output_tokens", tokens)
        owner = type("Owner", (ConfigStateMixin,), {})()
        owner.config = {"auto_compression_factor": True}
        owner.max_output_tokens = tokens
        owner.compression_factor_var = "3.0"
        ConfigStateMixin._update_auto_compression_factor(owner)
        assert config["compression_factor"] == owner.config["compression_factor"], tokens
    # off: the manual factor stays and the current budget becomes the held chunk size (desktop toggle)
    config = {"compression_factor": 2.0, "max_output_tokens": 40000}
    changed, _env = settings_rules.apply_change(config, "auto_compression_factor", False)
    assert changed["auto_compression_factor"] is False and "compression_factor" not in changed
    assert changed["manual_chunk_size"] == settings_rules.chunk_budget(40000, 2.0)


# ---------------------------------------------------------------------------
# Configure All › Reset all prompts to defaults
# ---------------------------------------------------------------------------

def _frozen_reset_keys() -> list:
    method = _method(frozen("src/metadata_batch_translator.py"), "_reset_all_prompts_to_defaults")
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "prompt_keys":
            return ast.literal_eval(node.value)
    raise AssertionError("prompt_keys literal not found")


def test_reset_key_list_is_the_frozen_literal_and_the_desktop_calls_the_shared_reset():
    from metadata_defaults import METADATA_PROMPT_RESET_KEYS

    assert list(METADATA_PROMPT_RESET_KEYS) == _frozen_reset_keys()
    method = _method(current("metadata_batch_translator.py"), "_reset_all_prompts_to_defaults")
    calls = [ast.unparse(n) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert "reset_metadata_prompts(self.gui.config)" in calls
    assert not any("prompt_keys" in ast.unparse(n) for n in ast.walk(method) if isinstance(n, ast.Assign))


class _StopAtSave(Exception):
    """Raised by the fake gui's save_config: the config side of the reset ends there (save / reload /
    widget updates follow)."""


def _run_reset_config_side(source: str, namespace: dict, config: dict, *, has_attr: bool) -> tuple:
    from metadata_defaults import ensure_metadata_prompt_defaults

    method = _method(source, "_reset_all_prompts_to_defaults")
    lines = source.splitlines()
    code = textwrap.dedent("\n".join(lines[method.lineno - 1:method.end_lineno]))
    exec(compile(code, "<_reset_all_prompts_to_defaults>", "exec"), namespace)
    calls: list = []

    class Gui:
        def save_config(self, show_message=True):
            calls.append(("save_config", show_message))
            raise _StopAtSave

    gui = Gui()
    gui.config = config
    if has_attr:
        gui.book_title_prompt = "custom title prompt"
    owner = type("Owner", (), {})()
    owner.gui = gui
    owner._initialize_default_prompts = lambda: ensure_metadata_prompt_defaults(gui.config)
    with pytest.raises(_StopAtSave):
        namespace["_reset_all_prompts_to_defaults"](owner)
    return gui.config, getattr(gui, "book_title_prompt", None), calls


def test_live_reset_config_side_equals_the_frozen_method():
    import metadata_defaults

    keys = _frozen_reset_keys()
    configs = [
        {},
        {key: f"custom {key}" for key in keys},
        {**{key: f"custom {key}" for key in keys[::2]}, "model": "m", "metadata_field_prompts": {"creator": "x"}},
        {"book_title_prompt": "keep?", "output_language": "Korean", "unrelated": [1, 2]},
    ]
    for config in configs:
        for has_attr in (True, False):
            legacy = _run_reset_config_side(frozen("src/metadata_batch_translator.py"), {}, copy.deepcopy(config),
                                            has_attr=has_attr)
            live = _run_reset_config_side(current("metadata_batch_translator.py"),
                                          {"reset_metadata_prompts": metadata_defaults.reset_metadata_prompts},
                                          copy.deepcopy(config), has_attr=has_attr)
            assert live == legacy, (config, has_attr)


def test_reset_metadata_prompts_matches_the_desktop_config_steps():
    from metadata_defaults import ensure_metadata_prompt_defaults, reset_metadata_prompts

    keys = _frozen_reset_keys()
    base = {key: f"custom {key}" for key in keys}
    base.update({"metadata_field_prompts": {"creator": "x"}, "model": "m", "output_language": "Korean"})
    legacy = copy.deepcopy(base)
    for key in keys:  # the frozen method's config steps (before its save / reload)
        if key in legacy:
            del legacy[key]
    legacy["book_title_prompt"] = ""
    ensure_metadata_prompt_defaults(legacy)
    new = copy.deepcopy(base)
    removed = reset_metadata_prompts(new)
    assert new == legacy and set(removed) == set(keys)
