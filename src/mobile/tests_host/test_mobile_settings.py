"""Host tests for U2 settings: MobileConfigStore, Prefs, schema-driven Settings, Env preview.

Run from src/mobile with the mobile venv (Flet installed):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_mobile_settings.py

* Store / Prefs / pure helpers need only Python + ``cryptography`` (the
  store goes through the shared ``src/config_store.py``).
* Renderer tests build controls into the in-memory fake Flet session of
  ``test_bootstrap`` against a fake schema module that implements the
  ``settings_schema`` contract, so they run before the real schema lands;
  the tests that use the real ``settings_schema`` / ``headless_owner`` /
  ``run_env`` modules skip cleanly while those are not importable.
* Every config.json lives in a temp dir; the API-key cipher is a
  process-local test key (``set_key_material``), never a key file in src/.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))  # the shared backend (config_store, settings_schema, ...)

from glossarion_mobile.state.config_store import MISSING, DebouncedSaver, MobileConfigStore, same_value  # noqa: E402
from glossarion_mobile.state.prefs import Prefs, atomic_write_json, file_ref_id  # noqa: E402
from glossarion_mobile.ui.router import ROUTES_BY_NAME, build_route, parse_route  # noqa: E402
from glossarion_mobile.ui.screens import env_preview as ep  # noqa: E402
from glossarion_mobile.ui.settings import model  # noqa: E402
from glossarion_mobile.ui.settings.schema_access import SchemaAccess  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
needs_crypto = pytest.mark.skipif(not _has("cryptography"), reason="cryptography not installed")

# The fake Flet session and the app fixtures are shared with test_bootstrap.py (one copy).
_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_settings", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env
_fake_session = _TB._fake_session
_route = _TB._route


@pytest.fixture
def test_key():
    """A process-local Fernet key for api_key_encryption (no key file is ever created)."""
    fernet = pytest.importorskip("cryptography.fernet")
    import api_key_encryption

    key = fernet.Fernet.generate_key()
    api_key_encryption.set_key_material(key)
    yield key
    api_key_encryption.set_key_material(None)


def _desktop_config() -> dict:
    return {
        "model": "gemini-3.5-flash",
        "api_key": "sk-test-1234567890abcdefghij",
        "temperature": 0.3,
        "batch_size": 10,
        "output_language": "English",
        "translation_history_rolling": True,
        "multi_api_keys": [
            {"api_key": "AIzaSyTestKeyNumberOne1234567890", "model": "gemini-3.5-flash", "enabled": True},
            {"api_key": "sk-second-0987654321zyxwvutsrq", "model": "gpt-6", "enabled": False},
        ],
        "prompt_profiles": {"Universal": "Translate to {target_lang}.\nKeep honorifics: 씨, さん."},
        "custom_entry_types": {"character": {"enabled": True, "has_gender": True}},
        "delay": 2.0,
        "chapter_range": "",
        "large_float": 1e-05,
    }


def _write_desktop_config(path: Path, cfg: Optional[dict] = None) -> bytes:
    """config.json exactly as the desktop writes it (shared config_store.save_config_file)."""
    import config_store

    config_store.save_config_file(dict(cfg or _desktop_config()), str(path), backup=False)
    return path.read_bytes()


# ==========================================================================
# MobileConfigStore
# ==========================================================================


def test_same_value_compares_types():
    assert same_value({"a": [1, 2.0]}, {"a": [1, 2.0]})
    assert not same_value(1, True)
    assert not same_value(5, 5.0)
    assert not same_value({"a": 1}, {"a": 1, "b": 2})
    assert same_value("x", "x") and not same_value("x", None)


@needs_crypto
def test_store_round_trips_desktop_config_byte_identically(tmp_path, test_key):
    path = tmp_path / "config.json"
    original = _write_desktop_config(path)
    raw = json.loads(original.decode("utf-8"))
    assert raw["api_key"].startswith("ENC:") and raw["multi_api_keys"][0]["api_key"].startswith("ENC:")
    mtime = path.stat().st_mtime_ns
    writes = []
    store = MobileConfigStore(path, debounce=0.05, defaults=lambda key: {"glossary_mode": "balanced"}.get(key))
    try:
        snapshot = store.load()
        assert snapshot == _desktop_config()  # decrypted
        assert store.get("api_key") == "sk-test-1234567890abcdefghij"
        assert store.undecryptable_keys() == []
        store.observe_saves(lambda ok, err: writes.append(ok))
        # nothing changed -> nothing written
        assert store.flush() is False
        # same value again -> still nothing
        assert store.set("temperature", 0.3) is False
        assert store.set_many({"batch_size": 10, "model": "gemini-3.5-flash"}) == []
        # changed, then put back -> nothing written
        assert store.set("temperature", 0.9) is True
        assert store.set("temperature", 0.3) is True
        assert store.flush() is False
        assert store.wait_idle(2)
        # display defaults are never written
        assert store.effective("glossary_mode") == "balanced" and not store.has("glossary_mode")
        assert store.effective("model") == "gemini-3.5-flash"
        assert not store.is_modified("glossary_mode")
        store.close()
        assert path.read_bytes() == original
        assert path.stat().st_mtime_ns == mtime
        assert writes == []
        assert not (tmp_path / "config_backups").exists()
    finally:
        store.close()


@needs_crypto
def test_store_sparse_write_keeps_other_keys_and_encrypted_values(tmp_path, test_key):
    path = tmp_path / "config.json"
    original = _write_desktop_config(path)
    before = json.loads(original.decode("utf-8"))
    store = MobileConfigStore(path, debounce=10)
    store.load()
    store.set("temperature", 0.7)
    assert store.flush() is True
    after_bytes = path.read_bytes()
    after = json.loads(after_bytes.decode("utf-8"))
    assert list(after.keys()) == list(before.keys())  # order kept, no defaults added
    for key in before:
        if key != "temperature":
            assert after[key] == before[key], key  # ENC: blobs byte-for-byte, not re-encrypted
    assert after["temperature"] == 0.7
    old_lines = original.decode("utf-8").splitlines()
    new_lines = after_bytes.decode("utf-8").splitlines()
    assert len(old_lines) == len(new_lines)
    assert [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b] == [old_lines.index('  "temperature": 0.3,')]
    # a first save of the session makes the usual config_backups copy, once
    backups = list((tmp_path / "config_backups").glob("config_*.json.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == original

    # changing a secret re-encrypts only that one
    store.set("api_key", "sk-brand-new-key-0000000000")
    store.flush()
    third = json.loads(path.read_text(encoding="utf-8"))
    assert third["api_key"].startswith("ENC:") and third["api_key"] != before["api_key"]
    assert third["multi_api_keys"] == before["multi_api_keys"]
    # new keys are appended, unset removes them again
    store.set("epub_details_show_special_files", True)
    store.flush()
    assert list(json.loads(path.read_text(encoding="utf-8")).keys())[-1] == "epub_details_show_special_files"
    store.unset("epub_details_show_special_files")
    store.flush()
    assert "epub_details_show_special_files" not in json.loads(path.read_text(encoding="utf-8"))
    # a fresh store decrypts what was written
    again = MobileConfigStore(path)
    assert again.load()["api_key"] == "sk-brand-new-key-0000000000"
    assert again.get("multi_api_keys") == _desktop_config()["multi_api_keys"]
    store.close()
    assert len(list((tmp_path / "config_backups").glob("config_*.json.bak"))) == 1


@needs_crypto
def test_store_debounce_and_synchronous_flush(tmp_path, test_key):
    path = tmp_path / "config.json"
    _write_desktop_config(path)
    calls: list[tuple[str, bool]] = []

    def writer(disk, target, backup=False):
        calls.append((threading.current_thread().name, backup))
        import config_store

        config_store.save_config_file(disk, target, backup=backup)

    store = MobileConfigStore(path, debounce=0.25, writer=writer)
    store.load()
    original = path.read_bytes()
    for value in (11, 12, 13):
        store.set("batch_size", value)
        time.sleep(0.02)
    assert store.save_pending and store.dirty
    assert path.read_bytes() == original  # still debouncing
    assert store.wait_idle(5)
    assert len(calls) == 1 and calls[0] == ("gl-config-save", True)  # one write, on the worker, with the backup
    assert json.loads(path.read_text(encoding="utf-8"))["batch_size"] == 13
    assert store.save_count == 1 and not store.dirty
    # flush() writes synchronously on the caller's thread and cancels the pending save
    store.set("batch_size", 14)
    assert store.flush() is True
    assert calls[-1] == (threading.current_thread().name, False)
    assert json.loads(path.read_text(encoding="utf-8"))["batch_size"] == 14
    time.sleep(0.4)
    assert len(calls) == 2  # the debounced save did not write again
    store.close()


@needs_crypto
def test_store_concurrent_writers_end_consistent(tmp_path, test_key):
    path = tmp_path / "config.json"
    store = MobileConfigStore(path, debounce=0.01)
    store.load()

    def worker(n):
        for i in range(30):
            store.set(f"k{n}", i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.close()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"k0": 29, "k1": 29, "k2": 29, "k3": 29}


@needs_crypto
def test_store_observers_snapshot_and_job_flag(tmp_path, test_key):
    store = MobileConfigStore(tmp_path / "config.json", debounce=10)
    store.load()
    seen, everything, jobs = [], [], []
    unsub = store.observe("model", lambda k, v: seen.append(v))
    store.observe_all(lambda k, v: everything.append(k))
    store.observe_job(jobs.append)
    store.set("model", "gpt-6")
    store.set("batch_size", 5)
    store.unset("model")
    assert seen == ["gpt-6", MISSING] and everything == ["model", "batch_size", "model"]
    unsub()
    store.set("model", "x")
    assert seen == ["gpt-6", MISSING]
    snap = store.snapshot()
    snap["batch_size"] = 99
    assert store.get("batch_size") == 5
    lst = [1]
    store.set("list", lst)
    lst.append(2)
    assert store.get("list") == [1]  # stored by value
    assert not store.job_running and store.changed_during_job == frozenset()
    store.set_job_running(True)
    store.set("temperature", 1.0)
    assert store.changed_during_job == {"temperature"} and jobs == [True]
    store.set_job_running(False)
    assert jobs == [True, False] and store.changed_during_job == frozenset()
    # revert_to (Discard changes since opening) notifies the changed keys only
    everything.clear()
    store.revert_to({"batch_size": 5})
    assert sorted(everything) == ["list", "model", "temperature"]
    assert store.snapshot() == {"batch_size": 5}
    store._saver.close()


@needs_crypto
def test_store_missing_and_corrupt_files(tmp_path, test_key):
    missing = tmp_path / "fresh" / "config.json"
    store = MobileConfigStore(missing, debounce=10)
    assert store.load() == {} and not store.exists and store.load_error is None
    assert store.flush() is False and not missing.exists()  # a fresh install writes nothing
    store.set("model", "gpt-6")
    assert store.flush() is True and json.loads(missing.read_text(encoding="utf-8")) == {"model": "gpt-6"}
    store.close()

    corrupt = tmp_path / "bad" / "config.json"
    corrupt.parent.mkdir()
    corrupt.write_text('{"model": "x",', encoding="utf-8")
    store = MobileConfigStore(corrupt, debounce=10)
    assert store.load() == {}
    assert store.load_error and "could not be parsed" in store.load_error
    assert store.corrupt_backup and Path(store.corrupt_backup).read_text(encoding="utf-8") == '{"model": "x",'
    store.set("model", "y")
    store.close()
    assert json.loads(corrupt.read_text(encoding="utf-8")) == {"model": "y"}


@needs_crypto
def test_store_keeps_undecryptable_keys_untouched(tmp_path, test_key):
    from cryptography.fernet import Fernet

    import api_key_encryption

    path = tmp_path / "config.json"
    _write_desktop_config(path)
    before = json.loads(path.read_text(encoding="utf-8"))
    api_key_encryption.set_key_material(Fernet.generate_key())  # e.g. a desktop config without its .glossarion_key
    store = MobileConfigStore(path, debounce=10)
    store.load()
    assert store.get("api_key", "").startswith("ENC:")
    assert store.undecryptable_keys() == ["api_key", "multi_api_keys"]
    store.set("batch_size", 3)
    store.close()
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["api_key"] == before["api_key"] and after["multi_api_keys"] == before["multi_api_keys"]


@needs_crypto
def test_store_nested_paths_write_one_value(tmp_path, test_key):
    path = tmp_path / "config.json"
    cfg = dict(_desktop_config(), qa_scanner_settings={"min_file_length": 200, "report_format": "detailed"})
    original = _write_desktop_config(path, cfg)
    store = MobileConfigStore(path, debounce=10, defaults=lambda k: {"qa_scanner_settings.min_file_length": 200,
                                                                     "ai_hunter_config.edge_filters.min": 0.7}.get(k))
    store.load()
    seen = []
    store.observe("qa_scanner_settings.min_file_length", lambda k, v: seen.append((k, v)))
    store.observe("qa_scanner_settings", lambda k, v: seen.append((k, v)))
    qa = ("qa_scanner_settings", "min_file_length")
    assert store.get(qa) == 200 and store.effective(("ai_hunter_config", "edge_filters", "min")) == 0.7
    assert not store.is_modified(qa)
    assert store.set(qa, 300)
    assert seen == [("qa_scanner_settings.min_file_length", 300),
                    ("qa_scanner_settings", {"min_file_length": 300, "report_format": "detailed"})]
    assert store.is_modified(qa)
    store.flush()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["qa_scanner_settings"] == {"min_file_length": 300, "report_format": "detailed"}
    # a nested value under an absent parent creates it; resetting it removes what the edit created
    deep = ("ai_hunter_config", "edge_filters", "min")
    assert store.set(deep, 0.5) and store.get("ai_hunter_config") == {"edge_filters": {"min": 0.5}}
    assert store.unset(deep) and not store.has("ai_hunter_config")
    # back to the original value -> the file matches the original bytes again
    store.set(qa, 200)
    store.flush()
    assert path.read_bytes() == original
    # resetting a nested value keeps its siblings (and the existing parent object)
    assert store.unset(qa) and store.get("qa_scanner_settings") == {"report_format": "detailed"}
    assert store.unset(("qa_scanner_settings", "report_format")) and store.get("qa_scanner_settings") == {}
    with pytest.raises(ValueError):
        store.set(("batch_size", "inner"), 1)  # the parent is not an object
    store.close()


def test_debounced_saver_coalesces_and_closes():
    runs = []
    saver = DebouncedSaver(lambda: runs.append(time.monotonic()), delay=0.1)
    for _ in range(5):
        saver.schedule()
    assert saver.pending
    assert saver.wait_idle(3)
    assert len(runs) == 1
    saver.schedule()
    saver.cancel()
    time.sleep(0.2)
    assert len(runs) == 1
    saver.close()
    saver.schedule()  # ignored once closed
    time.sleep(0.15)
    assert len(runs) == 1


# ==========================================================================
# Prefs (mobile_state.json)
# ==========================================================================


def test_prefs_round_trip_and_debounce(tmp_path):
    path = tmp_path / "mobile_state.json"
    prefs = Prefs(path, debounce=0.1)
    assert prefs.load()["version"] == 1 and not path.exists()
    pos = prefs.set_reader_position("ab12cd34ef56", "ch012.xhtml", 1.7, page=3, mode="scroll")
    assert pos["fraction"] == 1.0
    prefs.add_bookmark("ab12cd34ef56", "ch003.xhtml", 0.25, "Battle")
    prefs.set("recent_languages", ["English", "Korean"])
    assert prefs.set_last_route("/settings/s/response_handling#retry_timeout") is True
    assert prefs.set_last_route("/document/raw%3A%2Fx.epub") is False  # never a non-whitelisted route
    assert prefs.set_last_route("/__selftest__?suite=smoke") is False  # nor a handled one
    assert not path.exists()  # debounced
    assert prefs.wait_idle(3) and path.exists()
    again = Prefs(path)
    data = again.load()
    assert data["reader_positions"]["ab12cd34ef56"]["href"] == "ch012.xhtml"
    assert again.bookmarks("ab12cd34ef56")[0]["label"] == "Battle"
    assert again.get("recent_languages") == ["English", "Korean"]
    assert again.last_route() == "/settings/s/response_handling#retry_timeout"
    assert again.remove_bookmark("ab12cd34ef56", 0) and again.bookmarks("ab12cd34ef56") == []
    with pytest.raises(ValueError):
        again.set("file_refs", {})
    prefs.close()
    again.close()


def test_prefs_atomic_write_never_tears_the_file(tmp_path, monkeypatch):
    path = tmp_path / "mobile_state.json"
    prefs = Prefs(path, debounce=10)
    prefs.load()
    prefs.set("tips", ["a"])
    assert prefs.flush()
    good = path.read_bytes()
    prefs.set("tips", ["b"])

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    assert prefs.flush() is False and prefs.save_error
    monkeypatch.undo()
    assert path.read_bytes() == good  # the old file is intact
    assert [p.name for p in tmp_path.iterdir()] == ["mobile_state.json"]  # no temp file left behind
    assert prefs.flush() is True  # still dirty -> retried
    assert json.loads(path.read_text(encoding="utf-8"))["tips"] == ["b"]

    # a crash while serialising leaves the previous file too
    import glossarion_mobile.state.prefs as prefs_module

    def boom(*a, **k):
        raise RuntimeError("serialiser crashed")

    monkeypatch.setattr(prefs_module.json, "dump", boom)
    with pytest.raises(RuntimeError):
        atomic_write_json(path, {"x": 1})
    monkeypatch.undo()
    assert json.loads(path.read_text(encoding="utf-8"))["tips"] == ["b"]
    assert [p.name for p in tmp_path.iterdir()] == ["mobile_state.json"]
    prefs.close()


def test_prefs_corrupt_file_is_moved_aside(tmp_path):
    path = tmp_path / "mobile_state.json"
    path.write_text("{not json", encoding="utf-8")
    prefs = Prefs(path)
    assert prefs.load()["reader_positions"] == {}
    assert prefs.load_error
    assert not path.exists() and len(list(tmp_path.glob("mobile_state.corrupt-*.json"))) == 1
    prefs.close()


def test_prefs_file_refs_are_opaque_bounded_lru(tmp_path):
    prefs = Prefs(tmp_path / "mobile_state.json", debounce=10, max_file_refs=3)
    prefs.load()
    paths = [str(tmp_path / f"book{i}.epub") for i in range(4)]
    ids = [prefs.file_ref(p, kind="epub") for p in paths[:3]]
    assert all(re.fullmatch(r"[0-9a-f]{12}", fid) for fid in ids)
    assert build_route("tools.text", {"fid": ids[0]}) == f"/tools/text/{ids[0]}"  # route-safe
    assert ids[0] == file_ref_id(paths[0]) and prefs.file_ref(paths[0]) == ids[0]  # stable
    assert prefs.resolve_file_ref(ids[1]) == os.path.abspath(paths[1])  # touch -> most recent
    fourth = prefs.file_ref(paths[3])
    assert prefs.file_ref_count() == 3
    assert prefs.resolve_file_ref(ids[2]) is None  # least recently used was evicted
    assert prefs.resolve_file_ref(fourth) == os.path.abspath(paths[3])
    assert prefs.forget_file_ref(fourth) and prefs.resolve_file_ref(fourth) is None
    prefs.flush()
    reloaded = Prefs(tmp_path / "mobile_state.json")
    reloaded.load()
    assert reloaded.resolve_file_ref(ids[1], touch=False) == os.path.abspath(paths[1])
    prefs.close()
    reloaded.close()


# ==========================================================================
# Pure helpers and the schema access layer (fake schema implementing the contract)
# ==========================================================================


@dataclass(frozen=True)
class FakeSpec:
    key: str
    type: str = "str"
    default: Any = None
    save_default: Any = None
    converter: Any = None
    var_names: tuple = ()
    widget_sources: tuple = ()
    env: tuple = ()
    section: str = ""
    label: str = ""
    tooltip: str = ""
    choices: Any = None
    minimum: Any = None
    maximum: Any = None
    visible_if: Any = None
    locked_if: Any = None
    platforms: frozenset = frozenset({"desktop", "mobile"})
    discrepancies: tuple = ()
    parent: Any = None


@dataclass(frozen=True)
class FakeSection:
    id: str
    title: str
    keys: tuple
    group: str


def make_fake_schema(specs: list, sections: list, *, with_search: bool = True) -> types.ModuleType:
    by_key = {s.key: s for s in specs}
    mod = types.ModuleType("fake_settings_schema")

    def spec(key):
        return by_key[key]

    def effective_default(key):
        default = by_key[key].default
        if isinstance(default, dict) and "$ref" in default:
            return "Resolved " + default["$ref"]  # the real schema imports the module here
        if isinstance(default, dict) and "$expr" in default:
            return None
        return default

    def coerce(key, value):
        kind = by_key[key].type
        if kind == "int":
            if isinstance(value, bool):
                raise ValueError("expected a whole number")
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "bool":
            return bool(value)
        if kind == "choice":
            allowed = [c[0] if isinstance(c, tuple) else c for c in by_key[key].choices]
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")
        return value

    def is_available(key, platform="mobile"):
        s = by_key[key]
        if platform not in s.platforms:
            return False, "Not available on mobile"
        if key == "use_tor":
            return False, "Tor is not available on mobile"
        return True, None

    def evaluate_rule(rule_id, cfg):
        if rule_id == "glossary_off":
            return "Locked by mode: Off" if cfg.get("glossary_mode", "balanced") == "off" else False
        if rule_id == "rolling_on":
            return bool(cfg.get("translation_history_rolling", False))
        raise KeyError(rule_id)

    def search(query):
        q = query.casefold()
        return [s for s in specs if q in s.key.casefold() or q in s.label.casefold() or q in s.tooltip.casefold()
                or any(q in str(e).casefold() for e in s.env)]

    mod.all_specs = lambda: list(specs)
    mod.spec = spec
    mod.sections = lambda: list(sections)
    mod.effective_default = effective_default
    mod.coerce = coerce
    mod.is_available = is_available
    mod.evaluate_rule = evaluate_rule
    if with_search:
        mod.search = search
    return mod


LONG_PROMPT = "You are a professional translator.\nTranslate {raw_text} into {target_lang}.\n" * 3

FAKE_SPECS = [
    FakeSpec("translation_history_rolling", "bool", False, section="context", label="Rolling history",
             tooltip="Keep a rolling translation history.\nOlder entries are summarised.", env=("TRANSLATION_HISTORY_ROLLING",)),
    FakeSpec("batch_size", "int", 10, section="context", label="Batch size", minimum=1, maximum=500,
             env=("BATCH_SIZE",), discrepancies=("BATCH_SIZE: settings_map default 3",)),
    FakeSpec("temperature", "float", 0.3, section="context", label="Temperature", minimum=0.0, maximum=2.0),
    FakeSpec("contextual_window", "int", 5, section="context", label="Context window", minimum=0, maximum=20),
    FakeSpec("glossary_mode", "choice", "balanced", section="glossary", label="Glossary mode",
             choices=(("off", "Off"), ("minimal", "Minimal"), ("balanced", "Balanced"), ("full", "Full"))),
    FakeSpec("output_language", "str", "English", section="context", label="Target language",
             choices=("English", "Korean", "Japanese", "Chinese (Simplified)", "Spanish")),
    FakeSpec("glossary_fuzzy", "bool", True, section="glossary", label="Fuzzy matching", locked_if="glossary_off"),
    FakeSpec("glossary_name", "str", "", section="glossary", label="Glossary name"),
    FakeSpec("system_prompt", "str", LONG_PROMPT, section="prompts", label="System prompt"),
    FakeSpec("replicate_api_key", "secret", "", section="prompts", label="Replicate key"),
    FakeSpec("epub_css_override_path", "path", "", section="prompts", label="CSS override"),
    FakeSpec("stop_sequences", "list", ["</s>"], section="prompts", label="Stop sequences"),
    FakeSpec("custom_entry_types", "dict", {"character": {"enabled": True}}, section="prompts", label="Entry types"),
    FakeSpec("use_tor", "bool", False, section="context", label="Use Tor"),
    FakeSpec("dpi_scaling", "float", 1.0, section="context", label="Auto DPI scale", platforms=frozenset({"desktop"})),
    FakeSpec("summary_role", "str", "user", section="context", label="Summary role", visible_if="rolling_on"),
    # nested settings live inside their parent object in config.json
    FakeSpec("qa_scanner_settings.min_file_length", "int", 200, section="qa.settings", label="Min file length",
             minimum=0, maximum=100000, parent="qa_scanner_settings",
             tooltip="<qt><p>Files shorter than this are <b>flagged</b>.<br>0 disables &amp; skips.</p></qt>"),
    FakeSpec("qa_scanner_settings.report_format", "choice", "detailed", section="qa.settings", label="Report format",
             choices=("summary", "detailed", "verbose"), parent="qa_scanner_settings"),
    # lazy defaults: $ref (imports a module) and $expr (computed at run time)
    FakeSpec("assistant_prompt", "str", {"$ref": "glossarion_fake_heavy_module:PROMPT"}, section="qa.settings",
             label="Assistant prompt"),
    FakeSpec("extraction_workers", "int", {"$expr": "min(8, os.cpu_count())"}, section="qa.settings",
             label="Extraction workers"),
    # mis-typed by the generator (name heuristics): a pool list and a flag typed "secret"
    FakeSpec("multi_api_keys", "secret", [], section="qa.settings", label="Multi API keys"),
    FakeSpec("use_multi_api_keys", "secret", False, section="qa.settings", label="Use multi keys"),
]
FAKE_SECTIONS = [
    FakeSection("context", "Context & memory", tuple(s.key for s in FAKE_SPECS if s.section == "context"), "Translation"),
    FakeSection("glossary", "Glossary general", tuple(s.key for s in FAKE_SPECS if s.section == "glossary"), "Glossary"),
    FakeSection("prompts", "Prompts", tuple(s.key for s in FAKE_SPECS if s.section == "prompts"), "Translation"),
    FakeSection("qa.settings", "QA Scanner", tuple(s.key for s in FAKE_SPECS if s.section == "qa.settings"), "qa"),
    FakeSection("appearance", "Appearance", (), "General"),
]


def _fake_schema(**kw) -> SchemaAccess:
    return SchemaAccess(make_fake_schema(FAKE_SPECS, FAKE_SECTIONS, **kw))


def _spec(key):
    return next(s for s in FAKE_SPECS if s.key == key)


def test_tile_kind_mapping():
    expected = {
        "translation_history_rolling": "switch",
        "batch_size": "number",  # range too wide for a slider
        "temperature": "slider",
        "contextual_window": "slider",
        "glossary_mode": "segmented",
        "output_language": "dropdown",  # 5 choices
        "glossary_name": "text",
        "system_prompt": "prompt",
        "replicate_api_key": "secret",
        "epub_css_override_path": "path",
        "stop_sequences": "list",
        "custom_entry_types": "json",
    }
    assert {k: model.tile_kind(_spec(k)) for k in expected} == expected
    assert set(expected.values()) == set(model.TILE_KINDS) - set()  # every tile kind is covered
    assert model.tile_kind(FakeSpec("api_key", "str")) == "secret"
    assert model.tile_kind(FakeSpec("translation_chunk_prompt", "str", "x")) == "prompt"
    assert model.tile_kind(FakeSpec("mystery", "frobnicate")) == "text"
    assert model.tile_kind(FakeSpec("pairs", "list", [{"a": 1}])) == "json"
    assert model.tile_kind(FakeSpec("unknown_list", "list")) == "json"  # never stringify unknown items
    assert model.tile_kind(FakeSpec("names", "list"), ["a", "b"]) == "list"  # the stored value decides
    # the generator types some non-strings as "secret" by name; their value type wins
    assert model.tile_kind(_spec("multi_api_keys")) == "json" and model.tile_kind(_spec("use_multi_api_keys")) == "switch"
    assert model.tile_kind(FakeSpec("replicate_api_key", "secret")) == "secret"
    assert model.config_path(_spec("qa_scanner_settings.min_file_length")) == ("qa_scanner_settings", "min_file_length")
    assert model.config_path(_spec("batch_size")) == ("batch_size",)
    assert model.label_for(FakeSpec("ai_hunter_config.edge_filters.min_ratio", parent="ai_hunter_config")) == "Min ratio"
    assert model.slider_params(_spec("temperature")) == (0.0, 2.0, 20)
    assert model.slider_params(_spec("contextual_window")) == (0.0, 20.0, 20)


def test_model_summaries_masks_and_windows():
    assert model.mask_secret("sk-abcdefghijklmnop1234") == "sk-…1234"
    assert model.mask_secret("short") == "•••••" and model.mask_secret("") == "Not set"
    assert model.mask_secret("ENC:xyz") == "Encrypted (key unavailable)"
    assert model.summarize(True, "switch") == "On" and model.summarize(0.30000, "slider") == "0.3"
    assert model.summarize("balanced", "segmented", _spec("glossary_mode")) == "Balanced"
    assert model.summarize(["a", "b", "c", "d"], "list") == "4 items: a, b, c…"
    assert model.summarize({"a": 1}, "json") == "1 entry"
    assert model.summarize("C:\\styles\\book.css", "path") == "book.css"
    assert model.summarize(LONG_PROMPT, "prompt").startswith("You are a professional translator. · Translate")
    assert model.window_bounds(5, 10, 60) == (0, 10)
    assert model.window_bounds(90, 100, 20) == (80, 100)
    assert model.window_bounds(50, 100, 20) == (40, 60)
    assert model.window_bounds(0, 100, 20) == (0, 20)
    assert model.ordered_groups(["Data", "Translation", "Custom", "General", "Translation"]) == [
        "General", "Translation", "Data", "Custom"]
    assert model.env_names(FakeSpec("x", env=("A", ("B", "translation"), types.SimpleNamespace(name="C"), "A"))) == ["A", "B", "C"]
    assert model.help_line(_spec("translation_history_rolling")) == "Keep a rolling translation history."
    # Qt rich-text tooltips (the generated schema keeps desktop setToolTip HTML)
    assert model.plain_text(_spec("qa_scanner_settings.min_file_length").tooltip) == (
        "Files shorter than this are flagged.\n0 disables & skips.")
    assert model.help_line(_spec("qa_scanner_settings.min_file_length")) == "Files shorter than this are flagged."
    assert model.plain_text("a < b and c > d") == "a < b and c > d"
    assert model.plain_text("<ul><li>One</li><li>Two</li></ul>") == "• One\n• Two"
    assert model.group_title("main.model", "main") == "Models & keys"
    assert model.group_title("other.context", "other_settings") == "Translation"
    assert model.group_title("qa.settings", "qa") == "QA" and model.group_title("x", "Custom") == "Custom"
    assert model.group_title("internal.state", "internal") == "Advanced"


def test_schema_access_contract_and_fallbacks():
    schema = _fake_schema()
    assert schema.available
    assert [s.id for s in schema.sections()] == ["context", "glossary", "prompts", "qa.settings", "appearance"]
    assert [g for g, _s in schema.groups()] == ["General", "Translation", "Glossary", "QA"]
    assert schema.section_for_key("glossary_fuzzy").id == "glossary"
    assert schema.path_of("qa_scanner_settings.min_file_length") == ("qa_scanner_settings", "min_file_length")
    # lazy defaults never import a module on the UI loop; warm_defaults (worker thread) resolves them
    assert "glossarion_fake_heavy_module" not in sys.modules
    assert schema.effective_default("assistant_prompt") is None
    assert schema.default_note("assistant_prompt") == "loading default…"
    assert schema.default_note("extraction_workers") == "computed when a run starts"
    assert schema.effective_default("batch_size") == 10 and schema.default_note("batch_size") is None
    assert schema.warm_defaults() == len(FAKE_SPECS)
    assert schema.effective_default("assistant_prompt") == "Resolved glossarion_fake_heavy_module:PROMPT"
    assert schema.default_note("assistant_prompt") is None
    defaults = schema.effective_default("stop_sequences")
    defaults.append("mutated")
    assert schema.effective_default("stop_sequences") == ["</s>"]  # cached defaults are copied
    assert schema.spec("nope") is None and schema.effective_default("nope") is None
    assert schema.coerce("batch_size", "12") == 12
    with pytest.raises(ValueError):
        schema.coerce("glossary_mode", "weird")
    assert schema.availability("use_tor") == (False, "Tor is not available on mobile")
    assert schema.availability("dpi_scaling") == (False, "Not available on mobile")
    assert schema.availability("batch_size") == (True, None)
    assert schema.lock_reason(_spec("glossary_fuzzy"), {"glossary_mode": "off"}) == "Locked by mode: Off"
    assert schema.lock_reason(_spec("glossary_fuzzy"), {}) is None
    assert schema.hidden_reason(_spec("summary_role"), {}) == "Not used with the current settings"
    assert schema.hidden_reason(_spec("summary_role"), {"translation_history_rolling": True}) is None
    hits = schema.search("BATCH_SIZE")
    assert [h.key for h in hits] == ["batch_size"] and hits[0].breadcrumb == "Translation › Context & memory"
    # without a schema search() the local index ranks key/label/env/help/section matches
    local = _fake_schema(with_search=False)
    assert [h.key for h in local.search("temperature")][0] == "temperature"
    assert "glossary_fuzzy" in [h.key for h in local.search("glossary general")]
    # a missing schema module degrades to "unavailable" instead of raising
    missing = SchemaAccess(module_name="settings_schema_that_does_not_exist")
    assert not missing.available and missing.error and missing.sections() == [] and missing.search("x") == []
    assert missing.availability("x") == (True, None) and missing.coerce("x", 5) == 5


# ==========================================================================
# Renderer (fake Flet session)
# ==========================================================================


def _ctx(store, schema=None, page=None, **kw):
    from glossarion_mobile.ui.settings.context import SettingsContext

    navigated = kw.pop("navigated", None)
    notes = kw.pop("notes", None)
    return SettingsContext(
        page=page,
        store=store,
        schema=schema or _fake_schema(),
        navigate_route=(navigated.append if navigated is not None else None),
        notify=(lambda msg, *a: notes.append(msg)) if notes is not None else None,
        **kw,
    )


def _memory_store(tmp_path, data: Optional[dict] = None, schema: Optional[SchemaAccess] = None) -> MobileConfigStore:
    path = tmp_path / "config.json"
    if data is not None:
        path.write_text(json.dumps(data), encoding="utf-8")
    schema = schema or _fake_schema()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _match(route):
    match = parse_route(route)
    assert match is not None, route
    return match


@needs_flet
def test_renderer_builds_every_section_and_tile_kind(tmp_path):
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.tiles import TILE_CLASSES

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        schema = _fake_schema()
        store = _memory_store(tmp_path, {"batch_size": 25}, schema)
        ctx = _ctx(store, schema, page)
        kinds = set()
        for section in schema.sections():
            screen = SectionPage(_match(f"/settings/s/{section.id}"), ctx)
            body = screen.get_body()
            screen.actions()
            page.views[0].controls.append(body)
            kinds.update(t.kind for t in screen.tiles.values())
            assert screen.title == section.title
            assert screen.visible_keys == list(section.keys)
            for key, tile in screen.tiles.items():
                assert tile.control.key == __import__("flet").ScrollKey(key)
        page.update()
        assert conn.bytes_sent > 0
        assert kinds == set(TILE_CLASSES)
        unknown = SectionPage(_match("/settings/s/nope"), ctx)
        assert unknown.get_body().title == "Unknown settings section"
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_tiles_edit_validate_reset_and_show_reasons(tmp_path):
    import flet as ft

    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.settings.section_page import SectionPage

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        schema = _fake_schema()
        store = _memory_store(tmp_path, {"batch_size": 25, "glossary_mode": "off"}, schema)
        notes = []
        ctx = _ctx(store, schema, page, notes=notes)
        context = SectionPage(_match("/settings/s/context"), ctx)
        glossary = SectionPage(_match("/settings/s/glossary"), ctx)
        prompts = SectionPage(_match("/settings/s/prompts"), ctx)
        page.views[0].controls.extend([context.get_body(), glossary.get_body(), prompts.get_body()])
        page.update()
        for screen in (context, glossary, prompts):
            screen.did_show()  # store observers: a change re-evaluates every rendered tile's rules
        t = {**context.tiles, **glossary.tiles, **prompts.tiles}

        # effective values: stored vs display default (never written)
        assert t["batch_size"].value_text.value == "25" and t["batch_size"].modified_dot.visible
        assert t["temperature"].value_text.value == "0.3 · default" and not store.has("temperature")
        assert t["translation_history_rolling"].switch.value is False

        # switch, slider, number (+ steppers, validation), segmented, dropdown, text
        t["translation_history_rolling"].activate()
        assert store.get("translation_history_rolling") is True and t["translation_history_rolling"].switch.value
        t["temperature"]._on_slide_end(types.SimpleNamespace(control=types.SimpleNamespace(value=0.7000001)))
        assert store.get("temperature") == 0.7
        assert t["batch_size"].apply("40") and store.get("batch_size") == 40
        assert t["batch_size"].step(1) and store.get("batch_size") == 41
        assert not t["batch_size"].apply("abc") and t["batch_size"].error == "Enter a whole number"
        assert store.get("batch_size") == 41
        assert not t["batch_size"].apply("9999") and t["batch_size"].error == "Must be at most 500"
        assert t["batch_size"].apply("500") and t["batch_size"].error is None
        assert t["glossary_mode"].choose(1) and store.get("glossary_mode") == "minimal"
        assert t["glossary_mode"].segmented.selected == ["1"]
        dropdown = t["output_language"]
        assert dropdown.choose(1) and store.get("output_language") == "Korean"
        assert dropdown.dropdown.value == "1"
        # an option settings_schema marks unavailable here (is_value_available) is listed, never stored
        schema.module.is_value_available = (
            lambda key, value, platform="mobile": (False, "Needs PyTorch") if (key, value) == ("output_language", "Japanese")
            else (True, ""))
        japanese = next(i for i, (v, _l) in enumerate(dropdown.options()) if v == "Japanese")
        assert not dropdown.choose(japanese) and dropdown.error == "Needs PyTorch"
        assert store.get("output_language") == "Korean" and dropdown.dropdown.value == "1"
        assert dropdown.choose(1) and dropdown.error is None
        del schema.module.is_value_available
        t["glossary_name"].field.value = "My glossary"
        t["glossary_name"]._on_submit()
        assert store.get("glossary_name") == "My glossary"

        # locks follow other keys: Off locks fuzzy matching
        assert t["glossary_mode"].choose(0)
        glossary._on_config_change("glossary_mode")
        fuzzy = t["glossary_fuzzy"]
        assert fuzzy.lock_reason == "Locked by mode: Off" and fuzzy.switch.disabled
        assert not fuzzy.activate() and not store.has("glossary_fuzzy")
        assert any(isinstance(b, ft.Container) and b.tooltip == "Locked by mode: Off" for b in fuzzy.badges.controls)
        t["glossary_mode"].choose(2)
        glossary._on_config_change("glossary_mode")
        assert fuzzy.lock_reason is None and not fuzzy.switch.disabled

        # unavailable rows stay visible, disabled, with a ReasonChip; their values are untouched
        tor = t["use_tor"]
        assert not tor.editable and tor.switch.disabled
        assert isinstance(tor.badges.controls[0], ReasonChip) and tor.badges.controls[0].reason == "Tor is not available on mobile"
        assert not tor.apply(True) and not store.has("use_tor")
        assert isinstance(t["dpi_scaling"].badges.controls[0], ReasonChip)
        assert t["summary_role"].hidden_reason is None  # rolling history is on now

        # long-press reset removes the key (default applies again) and offers Undo
        assert t["batch_size"]._on_long_press() is None and not store.has("batch_size")
        assert notes[-1] == "Batch size reset to default"
        assert t["batch_size"].value_text.value == "10 · default"

        # help sheet: full tooltip, key, env, default, desktop discrepancy note
        sheet = t["batch_size"].open_help()
        assert "Config key: batch_size" in sheet.body and "Environment: BATCH_SIZE" in sheet.body
        assert "Desktop note: BATCH_SIZE: settings_map default 3" in sheet.body and "Default: 10" in sheet.body

        # full-screen editors: prompt, secret, path, list, json
        editor = t["system_prompt"].activate()
        assert editor.sheet.fullscreen and editor.field.value == LONG_PROMPT
        editor.field.value = "Short prompt"
        editor._on_change()
        assert editor.counter.value == "12 chars · ≈ 3 tokens"
        assert editor.save() and store.get("system_prompt") == "Short prompt"
        editor = t["system_prompt"].activate()
        editor._on_reset()
        assert editor.field.value == LONG_PROMPT and editor.save() and store.get("system_prompt") == LONG_PROMPT
        secret = t["replicate_api_key"].activate()
        assert secret.field.password and secret.field.can_reveal_password
        secret.field.value = "r8_abcdefghijklmnopqrstuvwxyz"
        assert secret.save() and t["replicate_api_key"].value_text.value == "r8_…wxyz"
        path_editor = t["epub_css_override_path"].activate()
        path_editor.field.value = "/data/imports/book.css"
        assert path_editor.save() and store.get("epub_css_override_path") == "/data/imports/book.css"
        lst = t["stop_sequences"].activate()
        lst.add("<|end|>")
        lst.move(1, -1)
        assert lst.save() and store.get("stop_sequences") == ["<|end|>", "</s>"]
        js = t["custom_entry_types"].activate()
        js.field.value = "[1, 2]"
        assert not js.save() and js.error_text.value == "Expected a JSON object ({…})"
        js.field.value = '{"term": {"enabled": false}}'
        assert js.save() and store.get("custom_entry_types") == {"term": {"enabled": False}}

        # Discard changes since opening restores the section's starting point
        changed = prompts.discard_changes()
        assert set(changed) >= {"system_prompt", "replicate_api_key", "stop_sequences"}
        assert not store.has("system_prompt") and store.get("batch_size") == 25  # as when the page opened
        page.update()
        assert conn.bytes_sent > 0
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_nested_lazy_default_and_key_pool_tiles(tmp_path):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.tiles import JsonTile, SwitchTile

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        schema = _fake_schema()
        store = _memory_store(tmp_path, {"multi_api_keys": [{"api_key": "sk-pool-1234567890abcdef", "model": "x"}],
                                         "qa_scanner_settings": {"report_format": "summary"}}, schema)
        navigated: list = []
        ctx = _ctx(store, schema, page, navigated=navigated)
        qa = SectionPage(_match("/settings/s/qa.settings"), ctx)
        page.views[0].controls.append(qa.get_body())
        page.update()
        qa.did_show()
        t = qa.tiles
        nested = t["qa_scanner_settings.min_file_length"]
        assert nested.path == ("qa_scanner_settings", "min_file_length")
        assert nested.value_text.value == "200 · default" and not nested.stored
        assert nested.apply("300") and store.get("qa_scanner_settings") == {"report_format": "summary", "min_file_length": 300}
        assert t["qa_scanner_settings.report_format"].value_text.value == "summary"
        sheet = nested.open_help()
        assert "Config key: qa_scanner_settings › min_file_length" in sheet.body
        assert "Files shorter than this are flagged.\n0 disables & skips." in sheet.body and "<b>" not in sheet.body
        assert nested.reset() and store.get("qa_scanner_settings") == {"report_format": "summary"}
        # lazy defaults: shown as notes until resolved, never imported on the loop
        assert t["assistant_prompt"].value_text.value == "Loading default…"
        assert t["extraction_workers"].value_text.value == "Computed when a run starts"
        assert "Default: computed when a run starts." in t["extraction_workers"].help_body()
        await ctx.run_io(schema.warm_defaults)
        qa.refresh_all()
        assert t["assistant_prompt"].value_text.value.startswith("Resolved glossarion_fake_heavy_module")
        # a key pool typed "secret" is shown read-only (count only), a mis-typed flag is a switch
        pool = t["multi_api_keys"]
        assert isinstance(pool, JsonTile) and not pool.editable and pool.activate() is None
        assert pool.value_text.value == "1 key" and "sk-pool" not in pool.value_text.value
        assert isinstance(pool.badges.controls[0], ReasonChip) and pool.badges.controls[0].reason == "Edited in API keys"
        # tapping it opens that pool in Settings › API keys (key_pool_service.POOL_SPECS: multi_api_keys = Translation)
        if pool.pool_slug() is not None:  # key_pool_service importable (backend on sys.path)
            assert pool._on_tap() == "/settings/keys/translation" and navigated[-1] == "/settings/keys/translation"
        assert isinstance(t["use_multi_api_keys"], SwitchTile)
        assert t["use_multi_api_keys"].activate() and store.get("use_multi_api_keys") is True
        page.update()
        qa.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_section_window_recentres_and_scrolls_to_jump_target(tmp_path, monkeypatch):
    from glossarion_mobile.ui.settings import section_page as sp

    monkeypatch.setattr(sp, "HIGHLIGHT_SECONDS", 0.01)
    specs = [FakeSpec(f"k_{i:03d}", "bool", False, section="big", label=f"Toggle {i}") for i in range(100)]
    schema = SchemaAccess(make_fake_schema(specs, [FakeSection("big", "Big", tuple(s.key for s in specs), "Translation")]))

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _memory_store(tmp_path, {}, schema)
        ctx = _ctx(store, schema, page)
        # a #fragment opens the window around its target
        screen = sp.SectionPage(_match("/settings/s/big#k_090"), ctx, window_size=20)
        page.views[0].controls.append(screen.get_body())
        page.update()
        assert screen.window == (80, 100) and "k_090" in screen.visible_keys
        assert screen.list_view.controls[0] is screen.earlier_button
        screen.did_show()
        await asyncio.sleep(0.2)
        assert screen.jumps == ["k_090"]
        assert "scroll_to" in conn.invoked()
        assert not screen.tiles["k_090"].highlighted  # 1.5 s highlight (shortened here) ended
        # an in-page jump outside the window re-centres it first
        assert await screen.focus_key("k_005", highlight=False)
        assert screen.window == (0, 20) and screen.list_view.controls[-1] is screen.later_button
        assert await screen.focus_key("k_050", highlight=False) and screen.window == (40, 60)
        assert not await screen.focus_key("missing")
        screen._on_later()
        assert screen.window == (50, 70)
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_settings_search_filters_and_opens_fragment_routes(tmp_path):
    from glossarion_mobile.ui.settings.search import SettingsSearch, find_settings

    async def scenario():
        conn, session = _fake_session("android")
        schema = _fake_schema()
        store = _memory_store(tmp_path, {"batch_size": 25, "glossary_mode": "off"}, schema)
        navigated = []
        ctx = _ctx(store, schema, session.page, navigated=navigated)
        assert [h.key for h in find_settings(ctx, "temperature")] == ["temperature"]
        assert [h.key for h in find_settings(ctx, "", ["modified"])] == ["batch_size", "glossary_mode"]
        assert [h.key for h in find_settings(ctx, "", ["locked"])] == ["glossary_fuzzy"]
        assert [h.key for h in find_settings(ctx, "", ["unavailable"])] == ["use_tor", "dpi_scaling"]
        assert find_settings(ctx, "") == []
        opened = []
        search = SettingsSearch(ctx, on_open=lambda hit: opened.append(ctx.open_setting(hit.section_id, hit.key)))
        session.page.views[0].controls.extend([search.field, search.filter_row, search.results])
        session.page.update()
        search.set_query("batch")
        assert list(search.result_rows) == ["batch_size"]
        assert search.results.controls[0].content.value == "Translation › Context & memory"
        search.toggle_filter("unavailable")
        assert search.result_rows == {} and "No settings match" in search.results.controls[0].content.value
        search.toggle_filter("unavailable")
        search.result_rows["batch_size"].on_click(None)
        assert navigated == ["/settings/s/context#batch_size"] == opened
        assert parse_route(navigated[0]).fragment == "batch_size"
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_settings_home_groups_chips_banners(tmp_path, test_key):
    import flet as ft

    from glossarion_mobile.ui.screens.base import HubScreen
    from glossarion_mobile.ui.settings.integration import IMPLEMENTED_ROUTES
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        schema = _fake_schema()
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"batch_size": 25, "api_key": "ENC:not-decryptable"}), encoding="utf-8")
        store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default)
        store.load()
        navigated, notes = [], []
        ctx = _ctx(store, schema, page, navigated=navigated, notes=notes)
        home = SettingsHome(_match("/settings"), ctx, implemented_routes=IMPLEMENTED_ROUTES)
        assert isinstance(home, HubScreen)
        page.views[0].controls.append(home.get_body())
        page.update()
        home.did_show()
        assert home.group_titles == ["General", "Translation", "Models & keys", "Glossary", "QA", "Data", "About"]
        assert home.section_tiles["context"].subtitle.value == "8 settings · 1 changed"
        assert "settings.appearance" not in home.route_tiles  # the schema section "appearance" covers it
        assert isinstance(home.route_tiles["settings.logs"].trailing, ft.Icon)
        assert home.route_tiles["settings.env_preview"].trailing.icon == ft.Icons.CHEVRON_RIGHT
        assert home.route_tiles["settings.models"].trailing.reason == "Arrives in U4"
        assert [n.key for n in home.notices.controls] == ["settings-notice-keys"]  # api_key kept, re-enter notice
        home.section_tiles["glossary"].on_click(None)
        home.route_tiles["settings.env_preview"].on_click(None)
        assert navigated == ["/settings/s/glossary", "/settings/logs/env"]
        # job banner
        assert not home.banner.visible
        store.set_job_running(True)
        assert home.banner.visible
        store.set_job_running(False)
        assert not home.banner.visible
        # quick chips: Save now, Backup, Import / Export profiles
        store.set("batch_size", 30)
        assert await home._on_save_now() is True and notes[-1] == "Settings saved"
        assert await home._on_save_now() is False and notes[-1] == "No unsaved changes"
        backup = await home._on_backup()
        assert backup and Path(backup).exists() and notes[-1].startswith("Backup created: config_")
        assert home._on_profiles().title == "Import / Export profiles"
        # the search field swaps the list for results
        home.search.set_query("temperature")
        assert "temperature" in home.search.result_rows and home.content.controls is home.search.results.controls
        home.search.set_query("")
        assert isinstance(home.content.controls[0], type(home.group_cards()[0]))
        assert json.loads(path.read_text(encoding="utf-8"))["api_key"] == "ENC:not-decryptable"
        home.dispose()
        store.close()

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(not _importable("settings_schema"), reason="settings_schema is not importable yet (U2 schema agent)")
def test_real_schema_renders_every_section(tmp_path):
    """The generated schema: a page builds for every section without exceptions (store empty = fresh install)."""
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.tiles import TILE_CLASSES

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        schema = SchemaAccess()
        assert schema.available, schema.error
        sections = schema.sections()
        assert sections, "settings_schema.sections() returned nothing"
        store = _memory_store(tmp_path, {}, schema)
        ctx = _ctx(store, schema, page)
        built = 0
        for section in sections:
            screen = SectionPage(_match(build_route("settings.section", {"section": section.id})), ctx, window_size=400)
            page.views[0].controls[:] = [screen.get_body()]
            page.update()
            for key in section.keys:
                if schema.spec(key) is not None:
                    tile = screen.tiles[key]
                    assert tile.kind in TILE_CLASSES
                    available, reason = schema.availability(key)
                    assert tile.available is available and (available or reason)
                    built += 1
        assert built > 0 and not store.keys()  # rendering never writes defaults
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# Env preview
# ==========================================================================


def test_redaction_rules():
    env = {
        "API_KEY": "sk-live-abcdefghijklmnop",
        "MULTI_API_KEYS": json.dumps([{"api_key": "AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAA", "model": "x"}]),
        "USE_MULTI_API_KEYS": "1",  # trivial flag stays visible
        "SEND_INTERVAL_SECONDS": "2",
        "MODEL": "gemini-3.5-flash",
        "CUSTOM_HEADER": "Bearer abcdefghijklmnopqrstuv",
        "GOOGLE_APPLICATION_CREDENTIALS": "/data/creds.json",
        "ECHO": "my secret is hunter2-hunter2",
        "SYSTEM_PROMPT": "x" * 1000,
        "AUTH_COOKIE": "session=abc123def456",
        "EMPTY_TOKEN": "",
        "ENCRYPTED": "ENC:gAAAAAB",
        "SESSION_TOKEN": "abc123def456ghi",
        "GLOSSARY_DUPLICATE_KEY_MODE": "fuzzy",  # "KEY" mid-name is not a secret
        "MAX_OUTPUT_TOKENS": "128000",
        "AZURE_PRIVATE_KEY_PATH_INFO": "pem-content-here",
    }
    rows = {r.key: r for r in ep.redact_env(env, secrets={"hunter2-hunter2"})}
    for name in ("API_KEY", "MULTI_API_KEYS", "CUSTOM_HEADER", "ECHO", "AUTH_COOKIE", "ENCRYPTED", "SESSION_TOKEN",
                 "AZURE_PRIVATE_KEY_PATH_INFO"):
        assert rows[name].redacted and rows[name].value == f"<REDACTED> ({len(env[name])} chars)", name
    for name in ("USE_MULTI_API_KEYS", "SEND_INTERVAL_SECONDS", "MODEL", "GOOGLE_APPLICATION_CREDENTIALS", "EMPTY_TOKEN",
                 "GLOSSARY_DUPLICATE_KEY_MODE", "MAX_OUTPUT_TOKENS"):
        assert not rows[name].redacted and rows[name].value == env[name], name
    assert rows["SYSTEM_PROMPT"].value.endswith("… (1,000 chars)") and rows["SYSTEM_PROMPT"].length == 1000
    assert list(rows) == sorted(rows, key=str.upper)
    assert [r.key for r in ep.filter_rows(rows.values(), "model")] == ["MODEL"]
    assert [r.key for r in ep.filter_rows(rows.values(), "hunter2")] == []  # redacted values are not searchable
    text = ep.rows_as_text(rows.values())
    assert "sk-live" not in text and "AIzaSy" not in text and "hunter2" not in text
    cfg = {"api_key": "sk-cfg-1234567890", "multi_api_keys": [{"api_key": "pool-key-123456"}], "model": "x",
           "fallback_keys": [{"api_key": "fb-key-123456", "note": "n"}]}
    assert ep.config_secrets(cfg) == {"sk-cfg-1234567890", "pool-key-123456", "fb-key-123456"}


def test_build_env_preview_isolates_process_state_and_holds_the_lock(monkeypatch, tmp_path):
    monkeypatch.delenv("GL_PREVIEW_PROBE", raising=False)
    large = types.ModuleType("large_env")
    large._store = {"KEEP": "1"}
    monkeypatch.setitem(sys.modules, "large_env", large)
    lock = threading.Lock()
    seen = {}

    class FakeOwner:
        def __init__(self, config, *, host, api_key):
            os.environ["GL_PREVIEW_PROBE"] = "written by owner init"  # like the desktop __init__ block
            large._store["BIG"] = "x" * 10
            sys.argv.append("--fake")
            host.log("owner ready")
            self.config, self.api_key = config, api_key

    def env_builder(owner, input_path, api_key):
        seen["locked"] = lock.locked()
        seen["path"] = input_path
        config_key = owner.config["api_key"]
        return {"API_KEY": api_key, "MODEL": owner.config["model"], "PROBE": os.environ["GL_PREVIEW_PROBE"],
                "OTHER": f"prefix {config_key} suffix"}

    argv = list(sys.argv)
    config = {"model": "gpt-6", "api_key": "sk-preview-key-123456"}
    result = ep.build_env_preview(config, input_path=ep.preview_input_path("txt", str(tmp_path)), lock=lock,
                                  owner_factory=FakeOwner, env_builder=env_builder)
    assert result.ok, result.error
    assert seen == {"locked": True, "path": str(tmp_path / "Inbox" / "env-preview.txt")}
    assert not lock.locked()
    assert "GL_PREVIEW_PROBE" not in os.environ and sys.argv == argv and large._store == {"KEEP": "1"}
    rows = {r.key: r for r in result.rows}
    assert rows["API_KEY"].redacted and rows["OTHER"].redacted and not rows["MODEL"].redacted
    assert rows["PROBE"].value == "written by owner init" and result.logs == ["owner ready"]
    assert config == {"model": "gpt-6", "api_key": "sk-preview-key-123456"}  # the snapshot is copied

    # busy lock -> friendly error, no owner built
    lock.acquire()
    try:
        busy = ep.build_env_preview(config, input_path="x.epub", lock=lock, lock_timeout=0.05,
                                    owner_factory=lambda *a, **k: pytest.fail("built while busy"))
    finally:
        lock.release()
    assert not busy.ok and busy.error.startswith("Busy")

    # missing shared modules / failing builders are reported, env still restored
    def missing(*a, **k):
        raise ImportError("No module named 'headless_owner'")

    gone = ep.build_env_preview(config, input_path="x.epub", lock=lock, owner_factory=missing)
    assert not gone.ok and "headless_owner" in gone.error

    def explode(owner, path, key):
        os.environ["GL_PREVIEW_PROBE"] = "half way"
        raise RuntimeError("boom")

    failed = ep.build_env_preview(config, input_path="x.epub", lock=lock, owner_factory=FakeOwner, env_builder=explode)
    assert not failed.ok and "RuntimeError: boom" in failed.error
    assert "GL_PREVIEW_PROBE" not in os.environ and not lock.locked()


@pytest.mark.skipif(not (_importable("headless_owner") and _importable("run_env")),
                    reason="headless_owner / run_env are not importable yet (U2 shared-core agents)")
def test_env_preview_with_real_headless_owner(tmp_path, monkeypatch, test_key):
    import app_paths

    # any config write by the owner (sanitizer persistence) must land in tmp, never in src/config.json
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.chdir(tmp_path)
    before = dict(os.environ)
    config = {"api_key": "sk-real-preview-0123456789", "use_multi_api_keys": True,
              "multi_api_keys": [{"api_key": "sk-pool-preview-0123456789", "model": "gpt-4o"}]}
    try:
        import unified_api_client

        uc = unified_api_client.UnifiedClient
    except Exception:  # backend dependencies missing in this venv: the pools are not reachable
        uc = None
    pools_before = {n: v for n, v in vars(uc).items() if ep._is_pool_state_attr(n)} if uc else {}
    result = ep.build_env_preview(config, input_path=ep.preview_input_path("epub", str(tmp_path)))
    assert result.ok, result.error
    assert dict(os.environ) == before
    if uc is not None:
        # the preview applied the snapshot's multi-key pool, then put the previous pool state back
        assert {n: v for n, v in vars(uc).items() if ep._is_pool_state_attr(n)} == pools_before
        assert uc._in_memory_multi_keys is pools_before.get("_in_memory_multi_keys")
    rows = {r.key: r for r in result.rows}
    assert result.count > 50 and "EPUB_PATH" in rows
    assert all("sk-real-preview-0123456789" not in r.value for r in result.rows)


def test_isolated_key_pools_never_mutates_the_previous_pools(monkeypatch):
    import threading as _threading

    class Pool:
        def __init__(self, keys):
            self.keys = list(keys)

    class UnifiedClient:
        _in_memory_multi_keys = ["old-key"]
        _in_memory_multi_keys_lock = _threading.RLock()
        _api_key_pool = Pool(["old-key"])
        _glossary_key_pool = None
        _glossary_pool_logged = True

        @classmethod
        def set_in_memory_multi_keys(cls, keys):  # like UnifiedClient.setup_multi_key_pool
            cls._in_memory_multi_keys = keys
            if cls._api_key_pool is None:
                cls._api_key_pool = Pool([])
            cls._api_key_pool.keys.clear()
            cls._api_key_pool.keys.extend(keys)
            cls._force_rotation = False  # added on first setup (hasattr-guarded upstream)
            cls._glossary_pool_logged = False

    fake = types.ModuleType("unified_api_client")
    fake.UnifiedClient = UnifiedClient
    monkeypatch.setitem(sys.modules, "unified_api_client", fake)
    old_pool, old_lock = UnifiedClient._api_key_pool, UnifiedClient._in_memory_multi_keys_lock
    with ep.scoped_process_env():
        assert UnifiedClient._api_key_pool is None  # detached: setup builds a fresh pool
        UnifiedClient.set_in_memory_multi_keys(["preview-key"])
        assert UnifiedClient._api_key_pool.keys == ["preview-key"]
    assert UnifiedClient._api_key_pool is old_pool and old_pool.keys == ["old-key"]
    assert UnifiedClient._in_memory_multi_keys == ["old-key"] and UnifiedClient._glossary_pool_logged is True
    assert UnifiedClient._in_memory_multi_keys_lock is old_lock
    assert "_force_rotation" not in vars(UnifiedClient)


def test_scoped_process_env_restores_key_by_key_and_never_clears(monkeypatch):
    monkeypatch.setenv("GL_SCOPE_KEEP", "stay")
    monkeypatch.setenv("GL_SCOPE_CHANGE", "before")
    monkeypatch.delenv("GL_SCOPE_ADDED", raising=False)
    target = {"KEEP": "1", "CHANGE": "a"}
    keep_value = target["KEEP"]
    ep.restore_mapping(target, {"KEEP": keep_value, "CHANGE": "b", "BACK": "c"})
    assert target == {"KEEP": "1", "CHANGE": "b", "BACK": "c"}

    seen_missing = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            if os.environ.get("GL_SCOPE_KEEP") != "stay":
                seen_missing.append(True)

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for _ in range(20):
            with ep.scoped_process_env():
                os.environ["GL_SCOPE_CHANGE"] = "during"
                os.environ["GL_SCOPE_ADDED"] = "new"
                for i in range(50):
                    os.environ[f"GL_SCOPE_PAD_{i}"] = "x"
    finally:
        stop.set()
        thread.join(5)
        sys.setswitchinterval(old_interval)
    assert not seen_missing  # an unchanged variable never disappears while the scope restores
    assert os.environ["GL_SCOPE_CHANGE"] == "before" and "GL_SCOPE_ADDED" not in os.environ
    assert not any(k.startswith("GL_SCOPE_PAD_") for k in os.environ)


@needs_flet
def test_env_preview_screen_builds_filters_and_copies(tmp_path):
    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        store = _memory_store(tmp_path, {"model": "gpt-6", "api_key": "sk-screen-key-0123456"})
        calls, copied = [], []

        def builder(snapshot, *, input_path):
            calls.append((snapshot, input_path))
            env = {"API_KEY": snapshot["api_key"], "MODEL": snapshot["model"], "BATCH_SIZE": "10"}
            return ep.EnvPreviewResult(True, rows=ep.redact_env(env, ep.config_secrets(snapshot)), secs=0.01,
                                       input_path=input_path)

        screen = ep.EnvPreviewScreen(_match("/settings/logs/env"), store=store, page=page, data_dir=str(tmp_path),
                                     builder=builder, copy_handler=copied.append)
        page.views[0].controls.append(screen.get_body())
        page.update()
        screen.did_show()
        screen._on_kind(types.SimpleNamespace(control=types.SimpleNamespace(selected=["pdf"])))
        result = await screen.run_preview()
        assert result.ok and calls[0][1].endswith("env-preview.pdf") and calls[0][0]["model"] == "gpt-6"
        assert screen.status.value.startswith("3 variables for env-preview.pdf") and "1 redacted" in screen.status.value
        assert [c.key for c in screen.rows_view.controls] == ["env-API_KEY", "env-BATCH_SIZE", "env-MODEL"]
        screen.set_query("model")
        assert [c.key for c in screen.rows_view.controls] == ["env-MODEL"]
        screen.set_query("")
        text = await screen._on_copy()
        assert copied == [text] and "sk-screen" not in text and "MODEL=gpt-6" in text
        store.set_job_running(True)
        assert screen.build_button.disabled and await screen.run_preview() is None
        store.set_job_running(False)
        assert not screen.build_button.disabled
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# Wiring into the running app
# ==========================================================================


def _load_main_module():
    spec = importlib.util.spec_from_file_location("glossarion_mobile_app_main_u2", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


@needs_flet
def test_settings_feature_wires_screens_lifecycle_and_job_banner(app_env, monkeypatch):
    from glossarion_mobile.state.app_state import JobStripModel
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen
    from glossarion_mobile.ui.screens.env_preview import EnvPreviewScreen
    from glossarion_mobile.ui.settings.integration import SettingsFeature
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    monkeypatch.setitem(sys.modules, "settings_schema", make_fake_schema(FAKE_SPECS, FAKE_SECTIONS))

    async def scenario():
        main_module = _load_main_module()
        conn, session = _fake_session("android")
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        try:
            # GlossarionApp.start installs the feature itself, before the initial route dispatch
            feature = app.settings
            assert isinstance(feature, SettingsFeature) and feature.app is app
            assert feature.screens_built == []  # the initial "/" route is the chat root
            assert app.config_store is feature.store and feature.store.loaded and feature.prefs.loaded
            config_path = Path(feature.store.path)
            assert config_path == app.paths.config_file and not config_path.exists()
            await _route(session, "/settings")
            assert isinstance(app.shell.top_screen, SettingsHome)
            await _route(session, "/settings/s/context#temperature")
            assert [type(e.screen).__name__ for e in app.shell.stack] == ["SettingsHome", "SectionPage"]
            section = app.shell.top_screen
            assert isinstance(section, SectionPage) and section.focus_target == "temperature"
            assert await _wait(lambda: section.jumps == ["temperature"])
            await _route(session, "/settings/logs/env")
            assert isinstance(app.shell.top_screen, EnvPreviewScreen)
            assert isinstance(app.shell.stack[-2].screen, DiagnosticsScreen)  # app's own screen still used
            assert feature.screens_built == ["settings", "settings.section", "settings.env_preview"]
            assert feature.prefs.last_route() == "/settings/logs/env"

            # edits are debounced; the app going to the background flushes them synchronously
            feature.store.set("batch_size", 42)
            assert feature.store.dirty and not config_path.exists()
            await session.dispatch_event(page._i, "app_lifecycle_state_change", {"state": "hide"})
            assert json.loads(config_path.read_text(encoding="utf-8")) == {"batch_size": 42}
            assert app.lifecycle and app.lifecycle[-1].endswith("hide")  # the app's own handler still ran

            # JobStrip -> "Changes apply to the next run"; the model key -> chat header context
            app.state.job_strip.set(JobStripModel("Translating · Book.epub"))
            assert feature.store.job_running
            app.state.job_strip.set(None)
            assert not feature.store.job_running
            feature.store.set("model", "gemini-3.5-flash")
            assert app.state.chat_context.value.model == "gemini-3.5-flash"
            feature.close()
        finally:
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


@needs_flet
def test_cold_start_settings_deep_link_opens_the_settings_home(app_env, monkeypatch):
    """The feature is installed before the initial route, so /settings at launch is SettingsHome."""
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    monkeypatch.setitem(sys.modules, "settings_schema", make_fake_schema(FAKE_SPECS, FAKE_SECTIONS))

    async def scenario():
        main_module = _load_main_module()
        conn, session = _fake_session("android")
        page = session.page
        session.apply_page_patch({"route": "/settings"})
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        try:
            assert app.initial_route == "/settings"
            assert isinstance(app.shell.top_screen, SettingsHome)
            assert app.settings.screens_built == ["settings"]
        finally:
            if app.settings is not None:
                app.settings.close()
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_route_and_import_hygiene():
    import ast
    import subprocess

    spec = ROUTES_BY_NAME["settings.env_preview"]
    assert spec.parent == "settings.logs" and parse_route("/settings/logs/env").name == "settings.env_preview"
    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "import glossarion_mobile.state.config_store, glossarion_mobile.state.prefs\n"
        "import glossarion_mobile.ui.settings.model, glossarion_mobile.ui.settings.schema_access\n"
        "print(json.dumps(sorted(m for m in ('flet', 'PySide6', 'translator_gui', 'config_store', 'settings_schema')"
        " if m in sys.modules)))\n"
    ) % str(APP_DIR)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"  # pure state/model modules import nothing heavy
    for path in sorted((APP_DIR / "glossarion_mobile").rglob("*.py")):
        if "settings" in path.parts or path.name in ("config_store.py", "prefs.py", "env_preview.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
