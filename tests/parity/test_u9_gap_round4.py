"""U9 gap closures (audit round 4): desktop rules shared with the mobile app.

The oracle is the desktop at ``U9_BASE_SHA`` (main 96da1ec6, the parent of the moves), read with
``git show``; the frozen code runs against the shared function (and the rewired desktop code) on the
same inputs.

* individual_endpoint_dialog ``IndividualEndpointDialog._validate`` / ``_is_azure_endpoint`` ->
  ``key_pool_service.individual_endpoint_error`` / ``is_azure_endpoint`` (the dialog calls them; Glossarion
  Mobile's KeyEditor and ``validate_entry`` use the same messages);
* multi_api_key_manager ``_edit_selected_key_contexts``'s shortcut buttons -> ``key_contexts.context_presets``
  (the dialog loops over it; the mobile KeyEditor / bulk context sheet show the same presets);
* the desktop Google credential pickers' check (``_browse_google_credentials``) ->
  ``settings_rules.google_credentials_error`` (mobile KeyEditor / Settings path tile);
* other_settings ``on_extraction_method_change`` -> the ``extraction:standard`` / ``extraction:enhanced``
  visibility rules (``settings_rules.text_extraction_method`` over owner_state's own initialisation);
* the Multi-Key Manager's Test all filter ("Only test enabled keys") that the mobile Keys screen applies.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
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


def _function(source: str, name: str, namespace: dict, *, owner: str = ""):
    """``name`` (a method of class ``owner``, or a module function) compiled into ``namespace``."""
    tree = ast.parse(source)
    scope = tree
    if owner:
        scope = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines()
    code = textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))
    exec(compile(code, f"<{name}>", "exec"), namespace)
    return namespace[name]


class _Box:
    """QMessageBox stand-in: records (kind, title, text)."""

    def __init__(self) -> None:
        self.shown: list = []

    def critical(self, _parent, title, text):
        self.shown.append(("critical", title, text))

    def warning(self, _parent, title, text):
        self.shown.append(("warning", title, text))


class _Text:
    def __init__(self, value: str) -> None:
        self.value = value

    def text(self) -> str:
        return self.value

    def currentText(self) -> str:  # noqa: N802 - Qt API
        return self.value


class _Check:
    def __init__(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:  # noqa: N802 - Qt API
        return self.checked


# ---------------------------------------------------------------------------
# Individual Endpoint dialog
# ---------------------------------------------------------------------------

ENDPOINT_CASES = [
    (enabled, url, version)
    for enabled in (True, False)
    for url in ("", "  ", "http://localhost:11434/v1", "https://api.example.com/v1", "192.168.1.5:11434/v1",
                "ftp://x", " https://res.openai.azure.com/ ", "https://x.azure.com/openai/deployments/d",
                "HTTPS://upper.example", "http://h/openai/deployments/x")
    for version in ("", "  ", "2025-01-01-preview")
]


def test_individual_endpoint_validation_is_the_dialog_rule():
    import key_pool_service

    def run(validate, is_azure, enabled, url, version):
        box = _Box()
        self = types.SimpleNamespace(enable_checkbox=_Check(enabled), endpoint_entry=_Text(url),
                                     api_version_combo=_Text(version))
        self._is_azure_endpoint = types.MethodType(is_azure, self)
        validate.__globals__["QMessageBox"] = box
        ok = validate(self)
        return ok, [text for _k, _t, text in box.shown]

    old_src = frozen("src/individual_endpoint_dialog.py")
    new_src = current("individual_endpoint_dialog.py")
    old_validate = _function(old_src, "_validate", {}, owner="IndividualEndpointDialog")
    old_azure = _function(old_src, "_is_azure_endpoint", {}, owner="IndividualEndpointDialog")
    new_ns = {"key_pool_service": key_pool_service}
    new_validate = _function(new_src, "_validate", new_ns, owner="IndividualEndpointDialog")
    new_azure = _function(new_src, "_is_azure_endpoint", dict(new_ns), owner="IndividualEndpointDialog")
    for enabled, url, version in ENDPOINT_CASES:
        legacy = run(old_validate, old_azure, enabled, url, version)
        assert run(new_validate, new_azure, enabled, url, version) == legacy, (enabled, url, version)
        error = key_pool_service.individual_endpoint_error(enabled, url, version)
        assert (error is None) == legacy[0] and (legacy[1] == ([error] if error else [])), (enabled, url, version)
        stub = types.SimpleNamespace()
        assert key_pool_service.is_azure_endpoint(url) == old_azure(stub, url)
    assert "_is_azure_endpoint" in new_src and "key_pool_service.individual_endpoint_error" in new_src


# ---------------------------------------------------------------------------
# Key request context presets
# ---------------------------------------------------------------------------

def _frozen_presets(routes):
    """The preset statements of the frozen ``_edit_selected_key_contexts`` (image_routes / presets / if)."""
    from key_contexts import POOL_CONTEXTS

    source = frozen("src/multi_api_key_manager.py")
    method = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "_edit_selected_key_contexts")
    picked = []
    for node in method.body:
        text = ast.get_source_segment(source, node) or ""
        if text.startswith(("image_routes =", "presets =", "if image_routes.intersection")):
            picked.append(text)
    assert len(picked) == 3, picked
    namespace = {"POOL_CONTEXTS": POOL_CONTEXTS, "routes": routes}
    exec(compile("\n".join(picked), "<presets>", "exec"), namespace)
    return namespace["presets"]


def test_context_presets_are_the_dialog_buttons():
    from key_contexts import POOL_CONTEXTS, context_presets

    for pool, routes in POOL_CONTEXTS.items():
        assert context_presets(routes) == _frozen_presets(routes), pool
    assert "for label, allowed in context_presets(routes):" in current("multi_api_key_manager.py")


# ---------------------------------------------------------------------------
# Google Cloud credentials picker check
# ---------------------------------------------------------------------------

def test_google_credentials_check_is_the_picker_check(tmp_path):
    import settings_rules

    files = {
        "sa": {"type": "service_account", "project_id": "p", "private_key": "k"},
        "no_project": {"type": "service_account"},
        "no_type": {"project_id": "p"},
        "list": ["type", "project_id"],  # the desktop membership test passes a list too
        "other": {"installed": {}},
    }
    paths = []
    for name, data in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths.append(path)
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    paths += [broken, tmp_path / "missing.json"]
    source = frozen("src/multi_api_key_manager.py")
    for path in paths:
        box = _Box()
        picked: list = []
        file_dialog = types.SimpleNamespace(getOpenFileName=lambda *a, p=str(path): (p, ""))
        browse = _function(source, "_browse_google_credentials",
                           {"QFileDialog": file_dialog, "QMessageBox": box, "json": json, "os": os})
        self = types.SimpleNamespace(google_creds_entry=types.SimpleNamespace(setText=picked.append),
                                     _sync_parent_google_credentials=lambda f: None, _show_status=lambda s: None)
        browse(self)
        error = settings_rules.google_credentials_error(str(path))
        if error is None:
            assert picked == [str(path)] and not box.shown, path.name
        else:
            assert not picked and box.shown and box.shown[0][2] == error, (path.name, box.shown, error)


class _Recorder:
    """Any attribute / call chain of a picker's ``self`` is recorded; names in ``fail`` raise."""

    def __init__(self, log: list, name: str, fail: frozenset) -> None:
        self._log, self._name, self._fail = log, name, fail

    def __getattr__(self, attr):
        return _Recorder(self._log, f"{self._name}.{attr}", self._fail)

    def __call__(self, *args, **kwargs):
        self._log.append((self._name, args, kwargs))
        if self._name in self._fail:
            raise RuntimeError(f"{self._name} failed")
        return _Recorder(self._log, f"{self._name}()", self._fail)


#: (file, owner class, method, extra call args, the success call that may fail)
LIVE_PICKERS = (
    ("translator_gui.py", "TranslatorGUI", "select_google_credentials", (), "self.save_config"),
    ("multi_api_key_manager.py", "MultiAPIKeyDialog", "_browse_google_credentials", (),
     "self._sync_parent_google_credentials"),
    ("multi_api_key_manager.py", "MultiAPIKeyDialog", "_browse_fallback_google_credentials", (),
     "self._sync_parent_google_credentials"),
    ("multi_api_key_manager.py", "MultiAPIKeyDialog", "_browse_glossary_google_credentials", (),
     "self._sync_parent_google_credentials"),
    ("multi_api_key_manager.py", "MultiAPIKeyDialog", "_dedicated_browse_google_credentials", ("metadata",),
     "self._sync_parent_google_credentials"),
)


def test_live_desktop_pickers_equal_the_frozen_ones(tmp_path, monkeypatch):
    """The desktop pickers call settings_rules' check and messages (U9 review): the live methods do
    exactly what the frozen ones did, on readable, refused, unreadable, scalar and missing files and when a
    success step raises."""
    files = {"sa": {"type": "service_account", "project_id": "p"}, "no_type": {"project_id": "p"},
             "list": ["type", "project_id"], "scalar": 42}
    paths = []
    for name, data in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths.append(path)
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    paths += [broken, tmp_path / "missing.json"]

    def run(source, owner, method, args, path, fail):
        box, log = _Box(), []
        file_dialog = types.SimpleNamespace(getOpenFileName=lambda *a, p=str(path): (p, ""))
        picker = _function(source, method, {"QFileDialog": file_dialog, "QMessageBox": box, "json": json, "os": os},
                           owner=owner)
        self = _Recorder(log, "self", fail)
        config: dict = {}
        self.__dict__["config"] = config  # translator_gui writes self.config[...]
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "unchanged")
        picker(self, *args)
        return box.shown, log, config, os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    for relpath, owner, method, args, success_step in LIVE_PICKERS:
        old_src, new_src = frozen(f"src/{relpath}"), current(relpath)
        assert "is_google_service_account(creds_data)" in ast.unparse(
            next(n for n in ast.walk(ast.parse(new_src)) if isinstance(n, ast.FunctionDef) and n.name == method))
        for path in paths:
            for fail in (frozenset(), frozenset({success_step})):
                legacy = run(old_src, owner, method, args, path, fail)
                assert run(new_src, owner, method, args, path, fail) == legacy, (method, path.name, fail)


# ---------------------------------------------------------------------------
# Text Extraction Method: the Standard / Enhanced option frames
# ---------------------------------------------------------------------------

class _Frame:
    def __init__(self) -> None:
        self.visible = None

    def setVisible(self, value) -> None:  # noqa: N802 - Qt API
        self.visible = bool(value)


def test_extraction_visibility_rules_follow_the_desktop_frames():
    import settings_rules
    from owner_state import initialize_extraction_variables

    handler = _function(frozen("src/other_settings.py"), "on_extraction_method_change", {})
    configs = [{}, {"extraction_mode": "enhanced"}, {"extraction_mode": "smart"},
               {"text_extraction_method": "enhanced"}, {"text_extraction_method": "standard"},
               {"extraction_mode": "enhanced", "text_extraction_method": "standard"},
               {"extraction_mode": "full", "text_extraction_method": "enhanced"}]
    for config in configs:
        owner = types.SimpleNamespace(config=dict(config))
        initialize_extraction_variables(owner)  # the desktop dialog's starting radio
        owner.enhanced_options_frame, owner.bs_options_frame = _Frame(), _Frame()
        handler(owner)
        assert settings_rules.evaluate("extraction:enhanced", config) == owner.enhanced_options_frame.visible, config
        assert settings_rules.evaluate("extraction:standard", config) == owner.bs_options_frame.visible, config


# ---------------------------------------------------------------------------
# Multi-Key Manager: Test all (Translation pool)
# ---------------------------------------------------------------------------

def test_test_all_filter_matches_the_desktop():
    source = frozen("src/multi_api_key_manager.py")
    method = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "_test_all")
    text = ast.get_source_segment(source, method)
    assert "indices = [i for i, key in enumerate(self.key_pool.keys) if key.enabled]" in text
    assert '"No enabled keys to test"' in text
    mobile = (SRC / "mobile" / "app" / "glossarion_mobile" / "ui" / "screens" / "keys.py").read_text(encoding="utf-8")
    assert 'if entry.get("enabled", True)]' in mobile and '"No enabled keys to test"' in mobile
