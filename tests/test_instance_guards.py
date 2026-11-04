"""Multi-instance translation: the request-time key guard, the main window's instance 1 key scope and the
QA scans of a multi-instance run (owner decisions after the review, 2026-11-04).

Owner: "we must not let instance 2 keys get used by instance 1"; an instance >= 2 never uses instance 1's keys
nor the main GUI key.

* (A) Under a key scope the real API client refuses, at request time, a key that belongs to another instance only
  (or, inside an instance >= 2, the main GUI key), whatever path chose it: a client built by hand, a key pool
  built before the scope; keys no pool of any instance holds pass; without a scope nothing changes.
* (B) While an instance >= 2 is alive, everything the main window starts (a review, the Manga translator, a QA
  scan) uses instance 1's keys only; every key again once the last instance ended.
* (C) The QA scanner's AI truncation check keeps to the scanning instance: an instance 1 scan never sends an
  instance 2 key, instance 2's post-translation scan uses its own key and config only.
* The stop sweeps of the main window spare the instance processes (and their helpers); an instance process
  whose main window is already gone does not start; an instance's glossary / image payload folders are its own.

The recording fake OpenAI-compatible server (tests/_fake_key_server.py) sees every request's key.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_instance_guards.py
"""

import base64
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _path in (str(TESTS_DIR), str(SRC_DIR)):
    if _path not in sys.path:  # CI has no tests/conftest.py
        sys.path.insert(0, _path)
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")  # before unified_api_client is imported

import instance_plan  # noqa: E402
import instance_worker  # noqa: E402
from test_instance_key_isolation import (  # noqa: E402,F401  (server / runtime are fixtures)
    MAIN_KEY,
    MESSAGES,
    MODEL,
    ONE,
    TWO,
    _key,
    _send,
    full_config,
    runtime,
    server,
)

#: a 1x1 PNG (the Manga translator sends images)
PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()


@pytest.fixture(autouse=True)
def fresh_key_owners():
    """What this process knows about whose key is whose: nothing before and after each test."""
    previous = instance_plan.set_key_owners({})
    yield
    instance_plan.set_key_owners(previous)


def _try_send(client, context="translation"):
    try:
        return _send(client, context)
    except Exception as exc:  # a refused key fails the request
        return str(exc), "error"


def _single_key_env(runtime, server):
    """The run environment of a config without key pools: a client sends exactly the key it was built with."""
    config = full_config(server.base_url, use_multi_api_keys=False, use_fallback_keys=False,
                         use_main_key_fallback=False)
    for _pool, (config_key, toggle_key, _use, _keys) in instance_plan.POOLS.items():
        config[toggle_key] = False
    runtime.start(runtime.run_env(config), config)
    return config


# ---------------------------------------------------------------------------
# (A) the request guard
# ---------------------------------------------------------------------------

def test_whose_key_is_whose(monkeypatch):
    """``key_owners`` of the main window's config (fingerprints only), ``foreign_key_owners`` per scope."""
    config = full_config("http://127.0.0.1:9/v1", glossary_keys=[_key("shared"), _key("shared", 2)])
    owners = instance_plan.key_owners(config)
    assert all(len(fp) == 32 and "tr-" not in fp for fp in owners)  # never the keys themselves
    assert owners[instance_plan.key_fingerprint(MAIN_KEY)] == [1]
    assert owners[instance_plan.key_fingerprint("tr-two")] == [2]
    assert owners[instance_plan.key_fingerprint("shared")] == [1, 2]
    instance_plan.set_key_owners(owners)
    assert instance_plan.foreign_key_owners("tr-two") == ()  # no scope: no lookup, nothing is foreign
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "2")
    assert instance_plan.foreign_key_owners(MAIN_KEY) == (1,)  # the main GUI key is instance 1's
    assert instance_plan.foreign_key_owners("tr-one") == (1,) and instance_plan.foreign_key_owners("fb-one") == (1,)
    assert instance_plan.foreign_key_owners("tr-two") == () and instance_plan.foreign_key_owners("shared") == ()
    assert instance_plan.foreign_key_owners("sk-nobody") == () and instance_plan.foreign_key_owners("") == ()
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "1")
    assert instance_plan.foreign_key_owners("tr-two") == (2,) and instance_plan.foreign_key_owners(MAIN_KEY) == ()
    # a key list narrowed under a scope teaches the guard whose its rows are
    instance_plan.set_key_owners({})
    assert instance_plan.scope_keys([_key("late-one"), _key("late-two", 2)]) == [_key("late-one")]
    assert instance_plan.foreign_key_owners("late-two") == (2,) and instance_plan.foreign_key_owners("late-one") == ()


def test_hand_built_clients_never_send_a_foreign_key(server, runtime):
    """A client built by hand with another instance's key (or, in an instance >= 2, the main GUI key) under a key
    scope: the request is refused before it is sent. Its own keys and keys of no pool go out; no scope: any key."""
    config = full_config(server.base_url)
    _single_key_env(runtime, server)
    restore = instance_worker.restrict_keys(1, config)  # the main window while instance 2 is alive
    try:
        content, finish = _try_send(runtime.client("tr-two"))
        assert finish == "error" and "refused" in content and "instance 2" in content, content
        _content, finish = _try_send(runtime.client("tr-one"))
        assert finish == "stop"
        _content, finish = _try_send(runtime.client(MAIN_KEY))  # the main GUI key is instance 1's
        assert finish == "stop"
    finally:
        restore()
    assert server.keys() == ["tr-one", MAIN_KEY]
    server.clear()

    restore = instance_worker.restrict_keys(2, instance_plan.config_for_instance(config, 2),
                                            key_owners=instance_plan.key_owners(config))  # an instance 2 process
    try:
        for key in (MAIN_KEY, "tr-one", "fb-one", "gl-one"):
            content, finish = _try_send(runtime.client(key))
            assert finish == "error" and "instance 1" in content, (key, content)
        for key in ("tr-two", "fb-two", "sk-nobody"):  # its own keys, and a key no pool holds (an endpoint's)
            _content, finish = _try_send(runtime.client(key))
            assert finish == "stop", key
    finally:
        restore()
    assert server.keys() == ["tr-two", "fb-two", "sk-nobody"]
    server.clear()

    # no scope: a single-instance run sends whatever it is given, as before
    assert instance_plan.key_scope() is None
    for key in ("tr-two", MAIN_KEY, "tr-one"):
        assert _try_send(runtime.client(key))[1] == "stop"
    assert server.keys() == ["tr-two", MAIN_KEY, "tr-one"]


def test_a_pool_built_before_the_scope_cannot_leak_into_it(server, runtime, monkeypatch):
    """The backstop: the Translation pool holds every instance's keys (built before the scope, never narrowed);
    under instance 2's scope its rotation may pick instance 1's keys, and those requests are refused instead of
    sent. Only instance 2's keys ever reach the server."""
    from multi_api_key_manager import APIKeyPool

    config = full_config(server.base_url, use_fallback_keys=False, use_main_key_fallback=False)
    runtime.start(runtime.run_env(config), config)
    pool = APIKeyPool()  # a pool object of every key that no narrowing reached
    pool.load_from_list(copy.deepcopy(config["multi_api_keys"]))
    assert sorted(k.api_key for k in pool.keys) == ["tr-one", "tr-one-b", "tr-two", "tr-two-b"]
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "2")
    instance_plan.set_key_owners(instance_plan.key_owners(config))
    outcomes = []
    for _ in range(8):
        client = runtime.client("tr-two")
        client._api_key_pool = pool  # its rotation picks from all four keys
        outcomes.append(_try_send(client)[1])
    keys = server.keys()
    assert "stop" in outcomes and keys and set(keys) <= TWO, (outcomes, keys)


def test_a_tts_request_with_a_foreign_key_is_refused_before_it_is_sent(server, runtime, monkeypatch, tmp_path):
    """Text to speech has its own dispatcher: the guard is there too."""
    import unified_api_client

    _single_key_env(runtime, server)
    monkeypatch.setenv("OPENAI_TTS_ENDPOINT", server.base_url + "/audio/speech")
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "2")
    instance_plan.set_key_owners(instance_plan.key_owners(full_config(server.base_url)))
    client = runtime.client("tr-one")
    with pytest.raises(unified_api_client.UnifiedClientError, match="refused"):
        client._dispatch_tts_request_inner("hello", str(tmp_path / "a.mp3"), None, None, "tts-1")
    assert server.keys() == []


# ---------------------------------------------------------------------------
# (B) the main window while an instance >= 2 is alive
# ---------------------------------------------------------------------------

def test_main_window_requests_keep_to_instance_one_while_an_instance_runs(server, runtime):
    """``restrict_keys(1, config)``, which the main window holds from the start of a multi-instance run until its
    last instance process ended (instance_runs): a review, the Manga translator's image request and a QA-style
    request (each one's Translation keys blocked, so every fallback path is tried) use instance 1's keys only.
    Afterwards every key again."""
    from extract_glossary_from_epub import create_client_with_multi_key_support

    config = full_config(server.base_url)
    env = runtime.run_env(config)  # the main window exports every instance's keys
    runtime.start(env, config)
    restore = instance_worker.restrict_keys(1, config)
    try:
        runtime.start(env, config)  # a run started meanwhile exports the full environment again
        server.behaviours.update({"tr-one": "prohibited", "tr-one-b": "prohibited"})
        review = create_client_with_multi_key_support(MAIN_KEY, MODEL, str(runtime.tmp / "review"),
                                                      copy.deepcopy(config), context="review")
        _try_send(review, "review")
        manga = runtime.client(MAIN_KEY)
        try:
            manga.send_image(copy.deepcopy(MESSAGES), PNG)
        except Exception:
            pass
        _try_send(runtime.client(MAIN_KEY), "qa_truncation")
        keys = server.keys()
        assert keys and set(keys) <= ONE | {MAIN_KEY}, keys
        # a client built by hand with an instance 2 key: that key is refused (its own keys may still answer)
        _try_send(runtime.client("tr-two"))
        assert set(server.keys()) <= ONE | {MAIN_KEY}, server.keys()
    finally:
        restore()
    server.clear()
    server.behaviours.clear()
    _single_key_env(runtime, server)
    assert _try_send(runtime.client("tr-two"))[1] == "stop" and server.keys() == ["tr-two"]


# ---------------------------------------------------------------------------
# (C) the QA scanner's AI checks
# ---------------------------------------------------------------------------

class _Owner:
    """The main window as the QA scanner reads it (``qa_scan_runtime._live_config_from_owner``)."""

    class _Entry:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

    def __init__(self, config):
        self.config = config
        self.api_key_entry = self._Entry(config.get("api_key", ""))
        self.model_var = config.get("model")


def _ai_scan(tmp_path, config, source="owner"):
    """A real QA scan with the AI truncation check of a 3-chapter book; returns (settings, log lines). Its
    settings are prepared as ``run_qa_scan`` prepares them from the main window (``source="owner"``), or from the
    run's environment like a QA job without a window (``"worker"``: its lists are read as exported)."""
    import qa_scan_runtime
    import scan_html_folder as shf
    from test_qa_stop_instant import _book

    folder, epub = _book(tmp_path, 3)
    owner = _Owner(copy.deepcopy(config)) if source == "owner" else None
    settings = qa_scan_runtime.prepare_qa_scan_settings({"check_ai_truncation_detection": True}, owner=owner)
    settings.update(use_thread_executor=True, check_ai_truncation_detection=True,
                    custom_mode_settings={"sample_size": 0})
    lines = []
    shf.scan_html_folder(str(folder), log=lines.append, mode="custom", qa_settings=settings, epub_path=str(epub))
    for name in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED", "GRACEFUL_STOP_COMPLETED"):
        os.environ.pop(name, None)
    return settings, lines


@pytest.mark.parametrize("source", ["owner", "worker"])
def test_qa_ai_check_in_instance_one_never_sends_an_instance_two_key(server, runtime, tmp_path, monkeypatch, source):
    """Instance 1's scope; the AI truncation pool holds only an instance 2 key: for instance 1 that pool is off
    (its Translation keys / the main GUI key do the work) and the instance 2 key is never sent. ``worker``: the
    settings come from the exported environment (every instance's AI keys); the scanner keeps the scope's rows."""
    config = full_config(server.base_url, ai_truncation_detection_keys=[_key("ai-two", 2)])
    runtime.start(runtime.run_env(config), config)
    if source == "worker":  # what an earlier, unscoped scan exported
        monkeypatch.setenv("USE_AI_TRUNCATION_DETECTION_KEYS", "1")
        monkeypatch.setenv("AI_TRUNCATION_DETECTION_API_KEYS", json.dumps(config["ai_truncation_detection_keys"]))
    restore = instance_worker.restrict_keys(1, config)
    try:
        settings, lines = _ai_scan(tmp_path, config, source)
    finally:
        restore()
    keys = server.keys()
    assert keys and set(keys) <= ONE | {MAIN_KEY}, (keys, lines[-15:])
    assert settings["_live_api_key"] == MAIN_KEY
    assert any("AI truncation" in line for line in lines)
    # the scanner itself sees no AI truncation key of instance 1: the pool is off for it (no seed key from it)
    assert not any("AI truncation detection key pool enabled" in line for line in lines), lines


def test_instance_two_qa_scan_uses_only_instance_two_keys(server, runtime, tmp_path):
    """The post-translation QA scan an instance process asks for runs in the main window under that instance's
    scope (instance_runs): its live config is instance 2's (``config_for_instance``: its own key in place of the
    main GUI key) and the AI truncation pool keeps only its row. Only instance 2's keys reach the server."""
    config = full_config(server.base_url)  # AI pool: ai-one and ai-two
    runtime.start(runtime.run_env(config), config)  # the main window's environment: every key
    restore = instance_worker.restrict_keys(2, instance_plan.config_for_instance(config, 2),
                                            key_owners=instance_plan.key_owners(config))
    try:
        settings, lines = _ai_scan(tmp_path, config)
    finally:
        restore()
    keys = server.keys()
    assert settings["_live_api_key"] == "tr-two" and MAIN_KEY not in json.dumps(settings["_live_config"])
    assert keys and set(keys) <= TWO, (keys, lines[-15:])
    assert "ai-two" in keys


def test_single_instance_qa_ai_check_is_unchanged(server, runtime, tmp_path):
    """No scope: the AI truncation pool with every key is used as before."""
    config = full_config(server.base_url)
    runtime.start(runtime.run_env(config), config)
    settings, lines = _ai_scan(tmp_path, config)
    keys = server.keys()
    assert settings["_live_api_key"] == MAIN_KEY and settings["_live_config"]["ai_truncation_detection_keys"] == [
        _key("ai-one"), _key("ai-two", 2)]
    assert keys and set(keys) <= {"ai-one", "ai-two"}, (keys, lines[-15:])


# ---------------------------------------------------------------------------
# Stop sweeps, the parent window, payload folders, HTTP log filters
# ---------------------------------------------------------------------------

def test_stop_sweeps_spare_the_instance_processes(tmp_path):
    """``kill_helper_subprocesses`` (an immediate Stop in the main window) ends a chapter-extraction helper of
    this window, but not one an instance process started (``in_instance_process_tree``)."""
    psutil = pytest.importorskip("psutil")
    import stop_control

    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', '--run-chapter-extraction']); "
            "time.sleep(60)")
    instance = subprocess.Popen([sys.executable, "-c", code])
    own = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "--run-chapter-extraction"])
    try:
        deadline = time.time() + 20
        helper = None
        while time.time() < deadline and helper is None:
            kids = psutil.Process(instance.pid).children()
            helper = kids[0] if kids else None
            time.sleep(0.1)
        assert helper is not None
        instance_plan.register_instance_process(instance.pid)
        assert instance_plan.in_instance_process_tree(helper)
        assert instance_plan.in_instance_process_tree(psutil.Process(instance.pid))
        assert not instance_plan.in_instance_process_tree(psutil.Process(own.pid))
        stop_control.kill_helper_subprocesses()
        assert own.wait(10) is not None  # this window's helper is gone
        assert helper.is_running() and instance.poll() is None  # the instance's helper and the instance are not
    finally:
        instance_plan.unregister_instance_process(instance.pid)
        for proc in (own, instance):
            if proc.poll() is None:
                proc.kill()
        try:
            for child in psutil.Process(instance.pid).children(recursive=True):
                child.kill()
        except Exception:
            pass
        try:
            helper.kill()
        except Exception:
            pass
    assert not instance_plan.in_instance_process_tree(psutil.Process(os.getpid()))


@pytest.mark.parametrize("module, function", [
    ("TransateKRtoEN.py", None), ("manga_runner.py", None), ("stop_control.py", "kill_helper_subprocesses"),
    ("stop_control.py", "kill_glossary_helper_subprocesses"),
])
def test_every_kill_all_children_sweep_asks_about_instance_processes(module, function):
    """Each "end every child of this process" sweep the main window runs on Stop skips instance processes."""
    source = (SRC_DIR / module).read_text(encoding="utf-8")
    if function is not None:
        start = source.index(f"def {function}(")
        source = source[start:source.index("\ndef ", start + 10)]
    assert "in_instance_process_tree(" in source


def test_an_instance_whose_main_window_is_gone_does_not_start(tmp_path, monkeypatch):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    parent = instance_worker._ParentProcess(dead.pid)
    assert parent.gone and not parent.running()
    assert instance_worker._ParentProcess(os.getpid()).running()
    started = []
    monkeypatch.setattr(instance_worker, "run_job", lambda *a, **k: started.append(1) or instance_worker.EXIT_OK)
    for name in (instance_plan.ENV_INSTANCE, instance_plan.ENV_KEY_INSTANCE):
        monkeypatch.delenv(name, raising=False)
    job = {"version": instance_worker.JOB_VERSION, "instance": 2, "files": ["a.epub"],
           "config_path": str(tmp_path / "c.json"), "parent_pid": dead.pid}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    assert instance_worker.main(str(path), hard_exit=False) == instance_worker.EXIT_STOPPED
    assert started == []


def test_payload_folders_of_an_instance_are_its_own(monkeypatch):
    monkeypatch.delenv(instance_plan.ENV_INSTANCE, raising=False)
    assert instance_plan.instance_payloads_dir("Payloads") == "Payloads"
    monkeypatch.setenv(instance_plan.ENV_INSTANCE, "2")
    assert instance_plan.instance_payloads_dir("Payloads") == os.path.join("Payloads", "Instance 2")
    for module in ("extract_glossary_from_epub.py", "translation_pipeline.py", "image_job.py", "rpgmaker_handler.py"):
        source = (SRC_DIR / module).read_text(encoding="utf-8")
        assert 'os.path.join("Payloads", "glossary"' not in source and 'os.path.join("Payloads", "image"' not in source
        assert 'instance_payloads_dir("Payloads")' in source, module


def test_http_logging_set_up_twice_adds_one_emoji_filter():
    """A second set-up (each multi-instance copy of the client runs it at import) adds no second 🌐 filter."""
    import unified_api_client

    unified_api_client.setup_http_logging()
    unified_api_client.setup_http_logging()
    names = [type(f).__name__ for f in logging.getLogger("httpx").filters]
    assert names.count("_HttpEmojiFilter") == 1
