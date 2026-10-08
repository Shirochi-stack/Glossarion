"""U9 gap closures (audit round 2): desktop code moved into shared GUI-free modules, and one seam fix.

The oracle is the desktop at ``U9_BASE_SHA`` (main 96da1ec6, the parent of the moves), read with
``git show``; the frozen code is executed against the shared function on the same inputs.

* multi_api_key_manager ``MultiAPIKeyDialog._refresh_key_list``'s Status rule ->
  ``key_pool_service.key_tree_status`` (the tree calls it; the mobile key cards read it);
* GlossaryManager_GUI Entry Type Configuration (legacy ``term`` normalisation, list order, Add Type,
  the built-in guard of × Remove) and Custom Fields (the ``custom_field_description_removed`` flag of
  Add / Remove) -> ``glossary_document.normalize_legacy_entry_types`` / ``sorted_entry_types`` /
  ``add_entry_type`` / ``entry_type_remove_warning`` / ``description_removed_flag``;
* ``TranslatorGUI._rename_input_for_existing_workspace_collision`` -> ``run_env.RunEnvMixin`` (verbatim;
  the desktop file selection still calls it on ``self``);
* seam: ``epub_converter._glossarion_library_dir`` and TransateKRtoEN's two Library registry lookups
  honour ``GLOSSARION_LIBRARY_DIR`` through ``library_core.library_root_path()`` (the same path on
  desktop when it is unset), so Compile EPUB replaces the organized Library copy on mobile too;
* ``settings_rules.apply_change(config, 'output_language', …)`` is the target-language fan-out.
"""

from __future__ import annotations

import ast
import copy
import itertools
import os
import subprocess
import sys
import textwrap
import time
import types
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


def class_method(source: str, class_name: str, name: str):
    tree = ast.parse(source)
    cls = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    return [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name][0]


def nested_function(method: ast.FunctionDef, name: str) -> ast.FunctionDef:
    return [n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n.name == name][0]


# ---------------------------------------------------------------------------
# key list status
# ---------------------------------------------------------------------------

def _legacy_status_function():
    """The frozen if/elif chain of ``_refresh_key_list`` as ``legacy(key) -> (status, tags)``."""
    source = frozen("src/multi_api_key_manager.py")
    start = source.index("            # Determine status based on test results and current state\n")
    end = source.index("            # Times used (counter)\n", start)
    block = textwrap.dedent(source[start:end])
    code = "def legacy(key):\n" + textwrap.indent(block, "    ") + "    return status, tags\n"
    namespace = {"time": time}
    exec(compile(code, "<frozen _refresh_key_list status>", "exec"), namespace)
    return namespace["legacy"]


def test_key_tree_status_matches_the_frozen_tree_rule(monkeypatch):
    import key_pool_service as kps

    legacy = _legacy_status_function()
    now = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    results = (None, "passed", "failed", "timeout", "rate_limited", "error", "other")
    messages = (None, "", "boom" * 10)
    checked = 0
    for result, message, enabled, cooling, last_error, testing in itertools.product(
            results, messages, (True, False), (True, False), (None, now - 10, now - 500), (False, True)):
        def make():
            key = types.SimpleNamespace(last_test_result=result, last_test_message=message, enabled=enabled,
                                        is_cooling_down=cooling, last_error_time=last_error, cooldown=60)
            if testing:
                key._testing = True
            return key
        old_key, new_key = make(), make()
        assert kps.key_tree_status(new_key) == legacy(old_key)
        assert vars(new_key) == vars(old_key)  # an expired cooldown is cleared the same way
        checked += 1
    assert checked > 500


def test_desktop_tree_calls_the_shared_status_rule():
    method = ast.unparse(class_method(current("multi_api_key_manager.py"), "MultiAPIKeyDialog", "_refresh_key_list"))
    assert "key_pool_service.key_tree_status(key)" in method  # (3.10's unparse brackets the tuple target)
    assert "elif key.last_test_result == 'rate_limited':" not in method


# ---------------------------------------------------------------------------
# Glossary Manager entry types / custom fields
# ---------------------------------------------------------------------------

class _Box:
    """QMessageBox stand-in recording warnings (the closures' only Qt call before the rule ends)."""

    Yes, No = 1, 2

    def __init__(self):
        self.warnings = []

    def warning(self, _parent, title, text):
        self.warnings.append((title, text))

    def question(self, *a, **k):
        return self.Yes


def _legacy_closure(name: str, free: dict):
    method = class_method(frozen("src/GlossaryManager_GUI.py"), "GlossaryManagerMixin", "_setup_manual_glossary_tab") \
        if name in ("add_custom_type", "remove_type", "add_custom_field", "remove_custom_field") else None
    assert method is not None
    fn = nested_function(method, name)
    namespace = dict(free)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"<frozen {name}>", "exec"), namespace)
    return namespace[name]


def _owner(types_):
    owner = types.SimpleNamespace(custom_entry_types=copy.deepcopy(types_), type_enabled_checkboxes={}, logs=[],
                                  config={}, saved=0, custom_glossary_fields=[])
    owner.append_log = owner.logs.append
    owner.save_config = lambda show_message=False: setattr(owner, "saved", owner.saved + 1)
    return owner


class _Line:
    def __init__(self, text):
        self.value = text

    def text(self):
        return self.value

    def clear(self):
        self.value = ""


class _Check:
    def __init__(self, on):
        self.on = on

    def isChecked(self):
        return self.on

    def setChecked(self, on):
        self.on = on


def test_add_type_and_remove_guard_match_the_frozen_closures():
    import glossary_document as gd

    base = {"character": {"enabled": True, "has_gender": True}, "terms": {"enabled": True, "has_gender": False}}
    for text, gender in itertools.product(("", "   ", "Skills", "  ITEMS ", "character", "terms", "Terms", "a&b"),
                                          (True, False)):
        owner = _owner(base)
        box = _Box()
        add = _legacy_closure("add_custom_type", {
            "self": owner, "new_type_entry": _Line(text), "has_gender_checkbox": _Check(gender), "parent": None,
            "QMessageBox": box, "update_type_checkboxes": lambda: None})
        add()
        legacy_types = owner.custom_entry_types
        mine = copy.deepcopy(base)
        type_name, warning = gd.add_entry_type(mine, text, gender)
        assert mine == legacy_types, text
        assert ([warning] if warning else []) == box.warnings, text
    for name in ("character", "term", "terms", "skills"):
        owner = _owner(dict(base, skills={"enabled": True, "has_gender": False}))
        box = _Box()
        remove = _legacy_closure("remove_type", {"self": owner, "parent": None, "QMessageBox": box,
                                                 "update_type_checkboxes": lambda: None})
        try:
            remove(name)
        except KeyError:  # the frozen closure deletes a type that is not there ("term"): not reached by the UI
            pass
        warning = gd.entry_type_remove_warning(name)
        assert ([warning] if warning else []) == box.warnings, name


def test_type_list_order_and_legacy_normalisation_are_the_frozen_statements():
    import glossary_document as gd

    source = frozen("src/GlossaryManager_GUI.py")
    assert "key=lambda x: (x[0] not in ['character', 'terms'], x[0]))" in source
    for types_ in ({"term": {"enabled": True}}, {"term": {"enabled": False}, "terms": {"enabled": True}},
                   {"zeta": {}, "character": {}, "alpha": {}, "terms": {}}):
        legacy = copy.deepcopy(types_)
        if 'term' in legacy and 'terms' not in legacy:
            legacy['terms'] = legacy.pop('term')
        if 'term' in legacy and 'terms' in legacy:
            legacy.pop('term', None)
        mine = gd.normalize_legacy_entry_types(copy.deepcopy(types_))
        assert mine == legacy
        assert gd.sorted_entry_types(mine) == sorted(mine.items(), key=lambda x: (x[0] not in ['character', 'terms'], x[0]))
    method = ast.unparse(class_method(current("GlossaryManager_GUI.py"), "GlossaryManagerMixin",
                                      "_setup_manual_glossary_tab"))
    for call in ("glossary_document.normalize_legacy_entry_types(self.custom_entry_types)",
                 "glossary_document.sorted_entry_types(self.custom_entry_types)",
                 "glossary_document.add_entry_type(", "glossary_document.entry_type_remove_warning(type_name)",
                 "glossary_document.description_removed_flag('add', field)",
                 "glossary_document.description_removed_flag('remove', field)"):
        assert call in method, call


def test_custom_field_flag_matches_the_frozen_add_and_remove():
    import glossary_document as gd

    class ListBox:
        def __init__(self, items):
            self.items = list(items)
            self.row = 0

        def addItem(self, text):
            self.items.append(text)

        def currentRow(self):
            return self.row

        def item(self, row):
            return types.SimpleNamespace(text=lambda: self.items[row])

        def takeItem(self, row):
            self.items.pop(row)

    for field in ("description", "Description", "notes", ""):
        owner = _owner({})
        owner.custom_glossary_fields = ["notes"]
        owner.custom_fields_listbox = ListBox(["notes"])
        owner.custom_field_entry = _Line(field)
        add = _legacy_closure("add_custom_field", {"self": owner})
        add()
        expected = owner.config.get("custom_field_description_removed")
        flag = gd.description_removed_flag("add", field.strip()) if field.strip() else None
        assert flag == expected, field
        owner = _owner({})
        owner.custom_glossary_fields = [field or "x"]
        owner.custom_fields_listbox = ListBox([field or "x"])
        remove = _legacy_closure("remove_custom_field", {"self": owner})
        remove()
        assert gd.description_removed_flag("remove", field or "x") == owner.config.get("custom_field_description_removed")


# ---------------------------------------------------------------------------
# workspace-collision rename -> RunEnvMixin
# ---------------------------------------------------------------------------

def test_rename_input_method_moved_verbatim_into_run_env():
    legacy = class_method(frozen("src/translator_gui.py"), "TranslatorGUI",
                          "_rename_input_for_existing_workspace_collision")
    moved = class_method(current("run_env.py"), "RunEnvMixin", "_rename_input_for_existing_workspace_collision")
    assert ast.unparse(moved) == ast.unparse(legacy)
    tg = current("translator_gui.py")
    assert "def _rename_input_for_existing_workspace_collision" not in tg
    assert "self._rename_input_for_existing_workspace_collision(path)" in tg


# ---------------------------------------------------------------------------
# seam: the Library folder of the compile / source lookups
# ---------------------------------------------------------------------------

def test_library_dir_seam_is_the_old_path_without_the_override(tmp_path, monkeypatch):
    import epub_converter
    import TransateKRtoEN

    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.delenv("GLOSSARION_LIBRARY_DIR", raising=False)
    legacy = os.path.join(os.path.expanduser("~"), "Documents", "Glossarion", "Library")
    assert os.path.normpath(epub_converter._glossarion_library_dir()) == os.path.normpath(legacy)
    assert os.path.normpath(TransateKRtoEN._glossarion_library_dir()) == os.path.normpath(legacy)
    # the frozen expression is what the fallback keeps
    assert 'return os.path.join(os.path.expanduser("~"), "Documents", "Glossarion", "Library")' in \
        frozen("src/epub_converter.py")
    for fn in ("_library_origins_raw_epubs_for_stem", "_library_raw_inputs_epubs_for_stem"):
        old = [n for n in ast.parse(frozen("src/TransateKRtoEN.py")).body if isinstance(n, ast.FunctionDef)
               and n.name == fn][0]
        new = [n for n in ast.parse(current("TransateKRtoEN.py")).body if isinstance(n, ast.FunctionDef)
               and n.name == fn][0]
        old_text = ast.unparse(old).replace(
            "os.path.join(os.path.expanduser('~'), 'Documents', 'Glossarion', 'Library')", "_glossarion_library_dir()")
        assert ast.unparse(new) == old_text, fn


def test_compile_replaces_the_organized_library_copy_under_the_library_override(tmp_path, monkeypatch):
    import json

    import epub_converter
    import TransateKRtoEN

    home = tmp_path / "home"
    lib = tmp_path / "lib"
    out = tmp_path / "out" / "Book"
    for folder in (home, lib / "Translated", lib / "Raw", out):
        folder.mkdir(parents=True)
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(lib))
    (lib / "Translated" / "Book.epub").write_bytes(b"PK old")
    compiled = out / "Book.epub"
    compiled.write_bytes(b"PK new")
    origins = {"version": 3, "raw": {}, "translated": {"Book.epub": str(compiled)}, "pairs": {}}
    (lib / "library_origins.txt").write_text(json.dumps(origins), encoding="utf-8")
    assert os.path.normpath(epub_converter._glossarion_library_dir()) == os.path.normpath(str(lib))
    target = epub_converter._organized_library_replacement_target(str(out))
    assert target is not None and os.path.normcase(os.path.normpath(target)) == \
        os.path.normcase(os.path.normpath(str(lib / "Translated" / "Book.epub")))
    # the source lookups read the same Library
    raw = lib / "Raw" / "Book.epub"
    raw.write_bytes(b"PK raw")
    (lib / "library_raw_inputs.txt").write_text(str(raw) + "\n", encoding="utf-8")
    assert [os.path.normcase(p) for p in TransateKRtoEN._library_raw_inputs_epubs_for_stem(os.path.normcase("Book"))] \
        == [os.path.normcase(str(raw))]


# ---------------------------------------------------------------------------
# settings_rules: the target-language change rule
# ---------------------------------------------------------------------------

def test_apply_change_output_language_is_the_fan_out():
    import settings_rules as sr

    start = {"output_language": "English", "ai_hunter_config": {"language_detection": {"target_language": "x"}},
             "manga_settings": {"manual_edit": {"translate_target_language": "x"}}}
    expected = copy.deepcopy(start)
    env = sr.fan_out_target_language(expected, "Korean")
    config = copy.deepcopy(start)
    changed, changed_env = sr.apply_change(config, "output_language", "Korean")
    assert config == expected and changed_env == env
    assert set(changed) == {"output_language", "glossary_target_language", "manga_settings", "ai_hunter_config"}
