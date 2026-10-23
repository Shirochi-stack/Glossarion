"""logs/Logistics.json: one compact record per API request (api_logistics; desktop only)."""
import json
import os
import sys
import threading

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import api_logistics  # noqa: E402


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("GLOSSARION_LOGISTICS", raising=False)
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    return tmp_path


def read(logs):
    with open(logs / "Logistics.json", encoding="utf-8") as f:
        return json.load(f)


def test_records_append_to_a_valid_compact_json_array(logs):
    api_logistics.record({"key": "Key#1 (gpt-5)", "params": {"temperature": 0.3}, "empty": "", "none": None})
    api_logistics.record({"key": "Key#2 (gpt-5)", "params": {}})
    assert read(logs) == [{"key": "Key#1 (gpt-5)", "params": {"temperature": 0.3}}, {"key": "Key#2 (gpt-5)"}]
    text = (logs / "Logistics.json").read_text(encoding="utf-8")
    assert text.count("\n") == 4 and '"params":{"temperature":0.3}' in text  # one compact line per request


def test_concurrent_writers_keep_every_record(logs):
    threads = [threading.Thread(target=lambda i=i: [api_logistics.record({"n": i * 100 + j}) for j in range(20)])
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r["n"] for r in read(logs)) == sorted(i * 100 + j for i in range(5) for j in range(20))


def test_rotation_and_switches(logs, monkeypatch):
    monkeypatch.setattr(api_logistics, "MAX_BYTES", 10)
    api_logistics.record({"n": 1})
    api_logistics.record({"n": 2})
    api_logistics.record({"n": 3})
    assert read(logs) == [{"n": 3}] and any(p.name.startswith("Logistics_") for p in logs.iterdir())
    monkeypatch.setenv("GLOSSARION_LOGISTICS", "0")
    assert api_logistics.record({"n": 4}) is None
    monkeypatch.delenv("GLOSSARION_LOGISTICS")
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert api_logistics.record({"n": 5}) is None and read(logs) == [{"n": 3}]


def test_masked_key_never_contains_the_key():
    assert api_logistics.mask_key("AIzaSyABCDEFGHIJKLMNOPx9Qk") == "AIzaSyAB...x9Qk"
    assert api_logistics.mask_key("short") == "sh...rt" and api_logistics.mask_key("") is None


def test_client_records_key_params_anti_duplicate_output_and_payload(logs, tmp_path, monkeypatch):
    import unified_api_client as api
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("GLOSSARY_ENABLE_ANTI_DUPLICATE", "1")
    monkeypatch.setenv("GLOSSARY_TOP_P", "0.9")
    monkeypatch.setenv("GLOSSARY_FREQUENCY_PENALTY", "0.0")
    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "0")
    client = UnifiedClient.__new__(UnifiedClient)
    client.client_type, client.model, client.context = "openai", "gpt-5", "glossary"
    client.key_identifier, client.current_key_index = "GlossaryKey#2 (gpt-5)", 1
    client.api_key = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
    client.output_dir = str(tmp_path / "Book")
    client._get_active_request_model = lambda: "gpt-5"
    payload = tmp_path / "Payloads" / "glossary" / "chapter_1.json"
    client._record_logistics(str(payload), {"temperature": 0.3, "max_tokens": 8192, "unused": None})
    [entry] = read(logs)
    assert entry["key"] == "GlossaryKey#2 (gpt-5)" and entry["key_index"] == 1
    assert entry["api_key"] == "sk-proj-...1234" and "ABCDEFGH" not in json.dumps(entry)
    assert entry["provider"] == "openai" and entry["model"] == "gpt-5" and entry["context"] == "glossary"
    assert entry["params"] == {"temperature": 0.3, "max_tokens": 8192}
    assert entry["anti_duplicate"] == {"section": "glossary", "enabled": True, "settings": {"top_p": "0.9"},
                                       "sent": {"top_p": 0.9}}
    assert entry["output_dir"] == str(tmp_path / "Book") and entry["payload"] == str(payload)
    # translation requests use the translation section
    client.context = "translation"
    client._record_logistics(None, {"temperature": 0.3})
    assert read(logs)[1]["anti_duplicate"] == {"section": "translation", "enabled": False}


def test_payload_cleanup_limit_setting(tmp_path):
    import shutdown_utils

    config = tmp_path / "config.json"
    default = 400 * 1024 * 1024
    assert shutdown_utils.payload_cleanup_limit_bytes(default, str(config)) == default  # no config yet
    config.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert shutdown_utils.payload_cleanup_limit_bytes(default, str(config)) == default  # key not saved
    config.write_text(json.dumps({"payload_cleanup_limit_mb": 50}), encoding="utf-8")
    assert shutdown_utils.payload_cleanup_limit_bytes(default, str(config)) == 50 * 1024 * 1024
    config.write_text(json.dumps({"payload_cleanup_limit_mb": 0}), encoding="utf-8")
    assert shutdown_utils.payload_cleanup_limit_bytes(default, str(config)) is None  # 0 = never clean up


def test_sweep_honors_never_clean_up(tmp_path, monkeypatch):
    import shutdown_utils

    swept = []
    monkeypatch.setattr(shutdown_utils, "payload_cleanup_limit_bytes", lambda default=None, config_file=None: None)
    monkeypatch.setattr(shutdown_utils, "sweep_size_capped_dir", lambda *a, **k: swept.append(a) or (0, 0, 0, 0))
    shutdown_utils.sweep_large_caches(phase="startup", extra_roots=[str(tmp_path)])
    assert swept == []


def test_payload_snapshot_matches_the_section(monkeypatch):
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("ENABLE_ANTI_DUPLICATE", "1")
    monkeypatch.setenv("TOP_K", "40")
    monkeypatch.setenv("GLOSSARY_ENABLE_ANTI_DUPLICATE", "0")
    client = UnifiedClient.__new__(UnifiedClient)
    client.client_type, client.model, client.context = "gemini", "gemini-2.5-pro", "translation"
    client._get_active_request_model = lambda: "gemini-2.5-pro"
    assert client._anti_duplicate_snapshot(0.3) == {"section": "translation", "enabled": True,
                                                    "settings": {"top_k": "40"}, "sent": {"top_k": 40}}
    client.context = "glossary"
    assert client._anti_duplicate_snapshot(0.3) == {"section": "glossary", "enabled": False}


def test_outgoing_request_is_saved_with_headers_redacted(tmp_path, monkeypatch):
    """2026-10-23: _save_outgoing_request referenced ``self`` at module level and never wrote a file."""
    import unified_api_client as api

    monkeypatch.delenv("DEBUG_SAVE_REQUEST_PAYLOADS", raising=False)
    api._save_outgoing_request("Anthropic", "POST", "https://api.anthropic.com/v1/messages",
                               {"x-api-key": "sk-ant-SECRET123456", "content-type": "application/json"},
                               {"model": "claude-sonnet-5-5", "max_tokens": 128000}, request_id="req_test",
                               out_dir=str(tmp_path))
    saved = json.loads((tmp_path / "req_test_request.json").read_text(encoding="utf-8"))
    assert saved["body"] == {"model": "claude-sonnet-5-5", "max_tokens": 128000}
    assert "SECRET" not in json.dumps(saved)
