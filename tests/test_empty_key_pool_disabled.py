"""A dedicated key pool with no enabled key counts as switched off.

Owner: "all keys are disabled, its an empty pool, and empty pool in general should not be treated
as enabled." A pool whose ``use_*`` toggle is on while every key in it is disabled (or the list is
empty) used to fail requests with "Image gen/edit key pool ... has no available keys; refusing
fallback to another pool"; now the request takes the normal route, as if the toggle were off. The
saved toggle is left as the user set it.

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_empty_key_pool_disabled.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import instance_plan as ip  # noqa: E402

# (switch variable, key-list variable, UnifiedClient in-memory setter or None)
POOL_ENVS = [
    ("USE_INPAINTER_KEYS", "INPAINTER_API_KEYS", "set_in_memory_inpainter_keys"),
    ("USE_GLOSSARY_KEYS", "GLOSSARY_API_KEYS", "set_in_memory_glossary_keys"),
    ("USE_ROLLING_SUMMARY_KEYS", "ROLLING_SUMMARY_API_KEYS", "set_in_memory_rolling_summary_keys"),
    ("USE_VISION_KEYS", "VISION_API_KEYS", "set_in_memory_vision_keys"),
    ("USE_TTS_KEYS", "TTS_API_KEYS", "set_in_memory_tts_keys"),
    ("USE_METADATA_KEYS", "METADATA_API_KEYS", None),
    ("USE_FALLBACK_KEYS", "FALLBACK_KEYS", None),
]
ALL_POOL_ENV = sorted({name for spec in ip.POOLS.values() for name in spec[2] + spec[3]})


def _key(name="k", enabled=True):
    return {"api_key": name, "model": "gpt-test", "enabled": enabled}


@pytest.fixture
def clean_env(monkeypatch):
    """No key scope, no pool variables and no in-memory pools from other tests."""
    for name in ALL_POOL_ENV + [ip.ENV_KEY_INSTANCE, ip.ENV_INSTANCE]:
        monkeypatch.delenv(name, raising=False)
    from unified_api_client import UnifiedClient

    for attr in ("_in_memory_inpainter_keys", "_in_memory_glossary_keys", "_in_memory_qa_scan_keys",
                 "_in_memory_rolling_summary_keys", "_in_memory_tts_keys",
                 "_inpainter_key_pool", "_glossary_key_pool", "_qa_scan_key_pool",
                 "_rolling_summary_key_pool", "_tts_key_pool"):
        monkeypatch.setattr(UnifiedClient, attr, None)
    return monkeypatch


# --------------------------------------------------------------------------
# instance_plan.pool_switch / has_enabled_key
# --------------------------------------------------------------------------

def test_has_enabled_key():
    class Entry:
        api_key = "x"

        def __init__(self, enabled):
            self.enabled = enabled

    assert ip.has_enabled_key([_key()])
    assert ip.has_enabled_key([_key(enabled=False), {"api_key": "y", "model": "m"}])  # default: enabled
    assert ip.has_enabled_key([Entry(False), Entry(True)])
    assert not ip.has_enabled_key([])
    assert not ip.has_enabled_key(None)
    assert not ip.has_enabled_key([_key(enabled=False), Entry(False)])
    assert not ip.has_enabled_key(["not-a-row", 3])


@pytest.mark.parametrize("use_env,keys_env,_setter", POOL_ENVS)
def test_pool_switch_reads_off_for_an_empty_or_all_disabled_pool(use_env, keys_env, _setter):
    enabled = json.dumps([_key("a", enabled=False), _key("b")])
    all_disabled = json.dumps([_key("a", enabled=False), _key("b", enabled=False)])
    assert ip.pool_switch(use_env, "1", environ={keys_env: enabled}) == "1"
    assert ip.pool_switch(use_env, "1", environ={keys_env: all_disabled}) == "0"
    assert ip.pool_switch(use_env, "1", environ={keys_env: "[]"}) == "0"
    assert ip.pool_switch(use_env, "1", environ={}) == "0"
    assert ip.pool_switch(use_env, "1", environ={keys_env: "{broken"}) == "0"
    # the client's in-memory list counts as well
    memory = lambda pool: [_key("m")]  # noqa: E731
    assert ip.pool_switch(use_env, "1", memory, environ={keys_env: all_disabled}) == "1"
    # a switch that is off, or unset, stays as it is
    assert ip.pool_switch(use_env, "0", environ={keys_env: enabled}) == "0"
    assert ip.pool_switch(use_env, None, environ={keys_env: enabled}) is None


def test_pool_switch_leaves_the_translation_pool_and_other_variables_alone():
    assert ip.pool_switch("USE_MULTI_API_KEYS", "1", environ={}) == "1"
    assert ip.pool_switch("USE_MULTI_KEYS", "1", environ={"MULTI_API_KEYS": "[]"}) == "1"
    assert ip.pool_switch("INPAINTER_API_KEYS", "[]", environ={}) == "[]"
    assert ip.pool_switch("MODEL", "gpt", environ={}) == "gpt"


def test_both_vision_switches_follow_both_vision_lists():
    rows = json.dumps([_key()])
    for name in ("USE_VISION_KEYS", "USE_QA_SCAN_KEYS"):
        assert ip.pool_switch(name, "1", environ={"QA_SCAN_API_KEYS": rows}) == "1"
        assert ip.pool_switch(name, "1", environ={"VISION_API_KEYS": rows}) == "1"
        assert ip.pool_switch(name, "1", environ={"VISION_API_KEYS": "[]", "QA_SCAN_API_KEYS": "[]"}) == "0"


# --------------------------------------------------------------------------
# unified_api_client._key_pool_env: every routing decision reads the switch through it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("use_env,keys_env,setter", POOL_ENVS)
@pytest.mark.parametrize("rows", ([], [_key("a", enabled=False), _key("b", enabled=False)]),
                         ids=("empty", "all-disabled"))
def test_client_reads_an_empty_pool_as_switched_off(clean_env, use_env, keys_env, setter, rows):
    import unified_api_client as uac

    clean_env.setenv(use_env, "1")
    clean_env.setenv(keys_env, json.dumps(rows))
    if setter and rows:
        getattr(uac.UnifiedClient, setter)(rows)
    assert uac._key_pool_env(use_env, "0") == "0"
    # the environment (the user's toggle as exported) is not rewritten
    assert os.environ[use_env] == "1"


@pytest.mark.parametrize("use_env,keys_env,setter", POOL_ENVS)
def test_client_keeps_a_pool_with_an_enabled_key(clean_env, use_env, keys_env, setter):
    import unified_api_client as uac

    rows = [_key("a", enabled=False), _key("b")]
    clean_env.setenv(use_env, "1")
    if setter:
        # in-memory only (the Image gen/edit, Glossary ... pools skip the env list)
        clean_env.setenv(keys_env, "[]")
        getattr(uac.UnifiedClient, setter)(rows)
    else:
        clean_env.setenv(keys_env, json.dumps(rows))
    assert uac._key_pool_env(use_env, "0") == "1"


def test_client_translation_pool_switch_is_unchanged(clean_env):
    import unified_api_client as uac

    clean_env.setenv("USE_MULTI_API_KEYS", "1")
    clean_env.setenv("MULTI_API_KEYS", "[]")
    assert uac._key_pool_env("USE_MULTI_API_KEYS", "0") == "1"


# --------------------------------------------------------------------------
# routing: an image request with an all-disabled Image gen/edit pool uses the main route
# --------------------------------------------------------------------------

def _main_route_client(monkeypatch):
    import unified_api_client as uac
    from unified_api_client import UnifiedClient

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("USE_MULTI_API_KEYS", "0")
    monkeypatch.setenv("USE_MULTI_KEYS", "0")
    monkeypatch.setattr(UnifiedClient, "_setup_client", lambda self: setattr(self, "client_type", "test"))
    client = UnifiedClient(api_key="main-key", model="gpt-test")
    monkeypatch.setattr(client, "reset_cleanup_state", lambda: None)
    monkeypatch.setattr(client, "_log_pre_stagger", lambda *_args: None)
    monkeypatch.setattr(client, "_apply_thread_submission_delay", lambda: None)
    monkeypatch.setattr(client, "_should_abort_retry", lambda: False)
    monkeypatch.setattr(client, "_extract_chapter_label", lambda _messages: None)
    monkeypatch.setattr(uac, "_api_watchdog_started", lambda *_a, **_k: None)
    monkeypatch.setattr(uac, "_api_watchdog_finished", lambda *_a, **_k: None)
    sent = []

    def fake_send_internal(self, *_args, **_kwargs):
        sent.append((self.api_key, getattr(self, "key_identifier", None)))
        return "ok", "stop"

    monkeypatch.setattr(UnifiedClient, "_send_internal", fake_send_internal)
    return client, sent


@pytest.mark.parametrize("rows", ([], [_key("img-a", enabled=False), _key("img-b", enabled=False)]),
                         ids=("empty", "all-disabled"))
def test_image_request_with_an_empty_image_pool_takes_the_main_route(clean_env, rows):
    from unified_api_client import UnifiedClient

    clean_env.setenv("USE_INPAINTER_KEYS", "1")
    clean_env.setenv("INPAINTER_API_KEYS", json.dumps(rows))
    if rows:
        UnifiedClient.set_in_memory_inpainter_keys(rows)
    client, sent = _main_route_client(clean_env)

    result = client._send_core([{"role": "user", "content": "edit"}], 0.0, 100, None, "image_edit")

    assert result[0] == "ok"
    assert sent and sent[0][0] == "main-key"


def test_image_request_with_an_enabled_image_key_still_uses_the_image_pool(clean_env):
    from unified_api_client import UnifiedClient

    rows = [_key("img-a", enabled=False), _key("img-b")]
    clean_env.setenv("USE_INPAINTER_KEYS", "1")
    clean_env.setenv("INPAINTER_API_KEYS", "[]")
    UnifiedClient.set_in_memory_inpainter_keys(rows)
    client, sent = _main_route_client(clean_env)

    result = client._send_core([{"role": "user", "content": "edit"}], 0.0, 100, None, "image_edit")

    assert result[0] == "ok"
    assert sent and sent[0][0] == "img-b"


# --------------------------------------------------------------------------
# glossary extraction: an all-disabled Glossary pool falls back to the Translation keys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("glossary_keys", ([], [_key("g", enabled=False)]), ids=("empty", "all-disabled"))
def test_glossary_extraction_skips_an_empty_glossary_pool(clean_env, tmp_path, glossary_keys):
    import extract_glossary_from_epub as extractor

    calls = []

    class FakeClient:
        def __init__(self, api_key, model, output_dir):
            self.context = None

        @classmethod
        def set_in_memory_glossary_keys(cls, keys, **kwargs):
            calls.append(("glossary", keys))

        @classmethod
        def set_in_memory_multi_keys(cls, keys, **kwargs):
            calls.append(("translation", keys))

    translation_keys = [_key("translation")]
    clean_env.setattr(extractor, "UnifiedClient", FakeClient)
    clean_env.setenv("USE_GLOSSARY_KEYS", "1")
    clean_env.setenv("GLOSSARY_API_KEYS", "[]")
    for name in ("USE_MULTI_API_KEYS", "USE_MULTI_KEYS"):
        clean_env.delenv(name, raising=False)

    extractor.create_client_with_multi_key_support(
        "main-key", "main-model", str(tmp_path),
        {"use_multi_api_keys": True, "multi_api_keys": translation_keys, "glossary_keys": glossary_keys},
        context="glossary",
    )

    assert calls == [("translation", translation_keys)]
