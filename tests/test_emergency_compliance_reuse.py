"""Verify glossary reuse, text equivalence, and request/run isolation."""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from glossary_matching import is_entry_type_token
from test_emergency_compliance_stop import FLAGS, load_functions


@pytest.fixture
def backend(monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    ns = load_functions("TransateKRtoEN.py", {
        "_EmergencyGlossaryComplianceCache",
        "_prepare_emergency_glossary_compliance",
        "_apply_prepared_emergency_glossary_compliance",
        "apply_emergency_glossary_compliance", "prepare_glossary_aware_request",
        "_request_glossary_setting", "_translation_prequeue_stop_mode",
        "is_stop_requested", "set_stop_flag",
    }, {"os": os, "json": json, "re": re, "threading": threading,
        "is_entry_type_token": is_entry_type_token, "_stop_requested": False})
    ns["find_glossary_file"] = lambda *_a, **_kw: "glossary.json"
    ns["build_system_prompt"] = lambda prompt, *_a, **_kw: prompt
    monkeypatch.setattr(os.path, "exists", lambda _: True)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)
    revision = [0]
    monkeypatch.setattr(ns["_EmergencyGlossaryComplianceCache"], "_file_signature",
                        staticmethod(lambda path: (os.path.abspath(path), revision[0])))
    ns["revision"] = revision
    ns["apply_emergency_glossary_compliance"]._logged_paths = set()
    return ns


ENTRIES = [
    ("character", "Alice Cooper", "Alice"),
    ("character", "Alice", "Bob"),
    ("character", "Alice", "Wrong duplicate"),
    ("character", "Bob", "Robert"),
    ("term", "词", "Word"),
    ("location", "塔", "Tower"),
]


def glossary(fmt):
    if fmt == "json-list":
        return "glossary.json", json.dumps([
            {"type": t, "raw_name": r, "translated_name": tr}
            for t, r, tr in ENTRIES])
    if fmt == "json-dict":
        return "glossary.json", json.dumps({str(i): {
            "type": t, "raw": r, "translated": tr}
            for i, (t, r, tr) in enumerate(ENTRIES)})
    if fmt in ("csv", "unit-separator"):
        sep = "," if fmt == "csv" else "\x1f"
        return "glossary.csv", "\n".join(
            [sep.join(("type", "raw_name", "translated_name"))]
            + [sep.join(entry) for entry in ENTRIES])
    lines = []
    for t, r, tr in ENTRIES:
        lines.append(f"=== {t.upper()}S ===")
        entry = f"{tr} ({r})" if fmt == "legacy-token" else f"{r} = {tr}"
        lines.append(f"* {entry} [male] (role: hero): Description")
    return "glossary.txt", "\n".join(lines)


def settings(**extra):
    return {"EMERGENCY_GLOSSARY_COMPLIANCE": "1",
            "EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "all",
            "EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": 0, **extra}


def apply(ns, text, cache=None, path="glossary.json", options=None):
    return ns["apply_emergency_glossary_compliance"](
        text, ".", path, settings=options or settings(), _cache=cache)


@pytest.mark.parametrize("fmt", ["json-list", "json-dict", "csv",
                                    "unit-separator", "token", "legacy-token"])
@pytest.mark.parametrize(("extra", "expected"), [
    ({}, "Robert Robert Robert Word Tower"),
    ({"EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "characters"},
     "Robert Robert Robert 词 塔"),
    ({"EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": 4}, "Bob Bob Bob 词 塔"),
    ({"EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "custom",
      "EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES": '["terms", "locations"]'},
     "Alice Cooper Alice Bob Word Tower"),
])
def test_reuse_preserves_formats_duplicates_overlap_and_cascades(
        backend, monkeypatch, fmt, extra, expected):
    path, raw = glossary(fmt)
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    text = "Alice Cooper Alice Bob 词 塔"
    assert apply(backend, text, path=path, options=settings(**extra)) == expected
    reader.reset_mock()
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    for _ in range(3):
        assert apply(backend, text, cache, path, settings(**extra)) == expected
    assert reader.call_count == 1


def test_nested_batch_reads_once_summarizes_and_keeps_original_for_compression(
        backend, monkeypatch, capsys):
    path, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    captured = {}

    def build(prompt, path, **kwargs):
        captured.update(kwargs)
        return prompt

    backend["build_system_prompt"] = build
    payload = {1: "Alice", 2: ["Bob", ("词", 7, None)], "Alice": "untouched"}
    _, result, _ = backend["prepare_glossary_aware_request"](
        "system", payload, glossary_path=path, settings=settings())
    assert result == {1: "Robert", 2: ["Robert", ("Word", 7, None)],
                      "Alice": "untouched"}
    assert payload[1] == "Alice"
    assert captured["source_text"] == json.dumps(payload, ensure_ascii=False, indent=2)
    assert reader.call_count == 1
    messages = capsys.readouterr().out
    assert messages.count("Emergency Glossary Compliance: replaced") == 1
    assert "across 4 text value(s)" in messages
    assert "0 matches" not in messages


def test_cache_refreshes_for_file_path_and_options_and_new_runs(backend, monkeypatch):
    _, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    assert apply(backend, "Alice", cache) == "Robert"
    assert apply(backend, "Alice", cache) == "Robert"
    reader.return_value.read.return_value = json.dumps({"Alice": "New"})
    backend["revision"][0] += 1
    assert apply(backend, "Alice", cache) == "New"
    assert apply(backend, "Alice", cache, options=settings(
        EMERGENCY_GLOSSARY_COMPLIANCE_MODE="characters")) == "Alice"
    assert apply(backend, "Alice", cache, path="other-book.json") == "New"
    assert apply(backend, "Alice", backend["_EmergencyGlossaryComplianceCache"]()) == "New"
    assert reader.call_count == 5


@pytest.mark.parametrize("disabled", [
    {"EMERGENCY_GLOSSARY_COMPLIANCE": "0"},
    {"AUTO_GLOSSARY_MODE": "no_glossary"},
])
def test_disabling_compliance_bypasses_warm_cache(backend, monkeypatch, disabled):
    _, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    assert apply(backend, "Alice", cache) == "Robert"
    assert apply(backend, "Alice", cache, options=settings(**disabled)) == "Alice"
    assert reader.call_count == 1


def test_parallel_chapters_prepare_once_and_do_not_mix_content(backend, monkeypatch):
    _, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda i: apply(backend, f"Alice {i}", cache), range(40)))
    assert results == [f"Robert {i}" for i in range(40)]
    assert reader.call_count == 1


@pytest.mark.parametrize("flag", FLAGS)
def test_stop_bypasses_warm_cache_and_resume_works(backend, monkeypatch, capsys, flag):
    _, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    monkeypatch.setattr("builtins.open", reader)
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    assert apply(backend, "Alice", cache) == "Robert"
    capsys.readouterr()
    monkeypatch.setenv(flag, "1")
    assert apply(backend, "Alice", cache) == "Alice"
    assert capsys.readouterr().out == ""
    monkeypatch.setenv(flag, "0")
    assert apply(backend, "Alice", cache) == "Robert"
    assert reader.call_count == 1


def test_stop_while_waiting_for_cache_does_not_wait_for_loader(backend, monkeypatch):
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    cache._lock.acquire()
    entered = threading.Event()

    def wait():
        entered.set()
        return cache.prepare("glossary.json", (), lambda: pytest.fail("must not load"))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(wait)
            assert entered.wait(timeout=1)
            monkeypatch.setenv("GRACEFUL_STOP", "1")
            assert result.result(timeout=1) is None
    finally:
        cache._lock.release()


def test_failed_read_is_not_cached(backend, monkeypatch):
    _, raw = glossary("json-list")
    reader = mock_open(read_data=raw)
    reader.side_effect = OSError("temporary read failure")
    monkeypatch.setattr("builtins.open", reader)
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    assert apply(backend, "Alice", cache) == "Alice"
    reader.side_effect = None
    assert apply(backend, "Alice", cache) == "Robert"
    assert reader.call_count == 2


def test_file_changed_during_load_is_not_cached(backend):
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    loads = []

    def load():
        loads.append(True)
        backend["revision"][0] += 1
        return "all", (("Alice", "Bob"),)

    cache.prepare("glossary.json", (), load)
    cache.prepare("glossary.json", (), load)
    assert len(loads) == 2


def test_cancelled_preparation_is_not_cached(backend, monkeypatch):
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    loads = []

    def load():
        loads.append(True)
        monkeypatch.setenv("GRACEFUL_STOP", "1")
        return "all", (("Alice", "Bob"),)

    cache.prepare("glossary.json", (), load)
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    cache.prepare("glossary.json", (), load)
    assert len(loads) == 2


def test_stop_during_cached_scan_discards_partial_text(backend, monkeypatch, capsys):
    _, raw = glossary("json-list")
    monkeypatch.setattr("builtins.open", mock_open(read_data=raw))
    cache = backend["_EmergencyGlossaryComplianceCache"]()
    assert apply(backend, "Alice", cache) == "Robert"
    capsys.readouterr()

    class StopOnReplace(str):
        def replace(self, old, new):
            monkeypatch.setenv("GRACEFUL_STOP", "1")
            return super().replace(old, new)

    original = StopOnReplace("Alice Cooper Alice")
    assert apply(backend, original, cache) is original
    assert capsys.readouterr().out == ""


def test_next_artifact_request_reloads_glossary_without_retaining_previous_plan(
        backend, monkeypatch):
    reader = mock_open(read_data=json.dumps({"Alice": "First"}))
    monkeypatch.setattr("builtins.open", reader)
    prepare = backend["prepare_glossary_aware_request"]
    _, first, _ = prepare("system", {1: "Alice"}, settings=settings())
    reader.return_value.read.return_value = json.dumps({"Alice": "Second"})
    _, second, _ = prepare("system", {1: "Alice"}, settings=settings())
    assert first == {1: "First"}
    assert second == {1: "Second"}
    assert reader.call_count == 2


def test_file_signature_includes_timestamp_size_and_identity():
    # Exercise the real signature method without modifying global filesystem calls.
    ns = load_functions("TransateKRtoEN.py", {"_EmergencyGlossaryComplianceCache"},
                        {"threading": threading})
    info = SimpleNamespace(st_dev=1, st_ino=2, st_size=3,
                           st_mtime_ns=4, st_ctime_ns=5)
    ns["os"] = SimpleNamespace(path=os.path, stat=lambda _: info)
    cls = ns["_EmergencyGlossaryComplianceCache"]
    baseline = cls._file_signature("glossary.json")
    for attribute in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        before = getattr(info, attribute)
        setattr(info, attribute, before + 1)
        assert cls._file_signature("glossary.json") != baseline
        setattr(info, attribute, before)
