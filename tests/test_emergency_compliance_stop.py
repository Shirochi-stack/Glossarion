"""Exercise cancellation without importing the desktop GUI or provider clients."""

import ast
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
FLAGS = ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED")


def load_functions(filename, names, namespace, class_name=None):
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8-sig"))
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.fixture
def backend(monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    ns = load_functions("TransateKRtoEN.py", {
        "apply_emergency_glossary_compliance", "_translation_prequeue_stop_mode",
        "is_stop_requested", "set_stop_flag",
    }, {"os": os, "json": json, "re": re, "_stop_requested": False,
        "_request_glossary_setting": lambda settings, key, default: settings.get(key, default)})
    monkeypatch.setattr(os.path, "exists", lambda _: True)
    ns["apply_emergency_glossary_compliance"]._logged_paths = {
        os.path.abspath("glossary.json")}
    return ns


def apply(backend, content="Alice Bob"):
    return backend["apply_emergency_glossary_compliance"](
        content, ".", "glossary.json",
        settings={"EMERGENCY_GLOSSARY_COMPLIANCE": "1"})


@pytest.mark.parametrize("flag", (*FLAGS, "module"))
def test_stop_before_scan_skips_file_and_logs(backend, monkeypatch, capsys, flag):
    if flag == "module":
        backend["set_stop_flag"](True)
    else:
        monkeypatch.setenv(flag, "1")
    reader = mock_open(read_data="[]")
    monkeypatch.setattr("builtins.open", reader)
    assert apply(backend) == "Alice Bob"
    reader.assert_not_called()
    assert capsys.readouterr().out == ""


def test_stop_during_read_skips_parse_and_logs(backend, monkeypatch, capsys):
    reader = mock_open()
    def read():
        monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "1")
        return "invalid JSON"
    reader.return_value.read.side_effect = read
    monkeypatch.setattr("builtins.open", reader)
    assert apply(backend) == "Alice Bob"
    assert capsys.readouterr().out == ""


def test_stop_during_replacement_discards_partial_result(backend, monkeypatch, capsys):
    raw = json.dumps({"Alice": {"type": "character", "translated": "Alicia"},
                      "Bob": {"type": "character", "translated": "Robert"}})
    monkeypatch.setattr("builtins.open", mock_open(read_data=raw))
    class StopOnReplace(str):
        def replace(self, old, new):
            monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
            return super().replace(old, new)
    original = StopOnReplace("Alice Bob")
    assert apply(backend, original) is original
    assert capsys.readouterr().out == ""


def test_new_run_resumes_normal_replacement(backend, monkeypatch, capsys):
    monkeypatch.setattr("builtins.open", mock_open(read_data=json.dumps([
        {"type": "character", "raw_name": "Alice", "translated": "Alicia"}])))
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    assert apply(backend) == "Alice Bob"
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    assert apply(backend) == "Alicia Bob"
    assert "replaced 1 entries" in capsys.readouterr().out


@pytest.mark.parametrize("flag", (*FLAGS, "owner"))
def test_queued_compliance_log_is_hidden_after_stop(monkeypatch, flag):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    ns = load_functions("translator_gui.py", {
        "_direct_log_message_is_suppressed", "_append_gui_log_batch",
    }, {"os": os}, class_name="TranslatorGUI")
    visible = []
    owner = SimpleNamespace(stop_requested=False,
        log_text=SimpleNamespace(document=lambda: object(), appendPlainText=visible.append),
        _schedule_log_autoscroll=lambda: None)
    owner._direct_log_message_is_suppressed = lambda message: ns[
        "_direct_log_message_is_suppressed"](owner, message)
    message = "Emergency Glossary Compliance: 0 matches found in content"
    ns["_append_gui_log_batch"](owner, [message])
    assert visible == [message]
    visible.clear()
    if flag == "owner":
        owner.stop_requested = True
    else:
        monkeypatch.setenv(flag, "1")
    ns["_append_gui_log_batch"](owner, [message, "Translation stopped"])
    assert visible == ["Translation stopped"]
