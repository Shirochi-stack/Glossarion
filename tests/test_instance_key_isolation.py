"""Multi-instance translation: no API key of one instance reaches another instance's requests.

Owner requirement: "instance 2 must never use the fallback key of instance 1, we must not have
instance contamination" and "the main gui key is always instance 1". Every key pool is split by
instance (``instance_plan.config_for_instance``); inside an instance the real API client narrows
every key list it is given or reads (``instance_plan.key_scope``), whoever exported it.

The real ``unified_api_client`` and its real retry / fallback logic run against a fake
OpenAI-compatible server (tests/_fake_key_server.py) that records the key of every request and fails
chosen keys (an empty ``content_filter`` answer = prohibited content; HTTP 429 = rate limit). The
run's environment comes from the real ``run_env`` (a HeadlessOwner built from the config):

* (e) a single-instance run is unchanged: every key, the main GUI key retry, every fallback key;
* (a) instance 2 whose Translation key is blocked tries only its own fallback keys, never
  instance 1's nor the main GUI key; also as a real ``instance_worker`` child process started
  with the main window's keys in its parent environment;
* (b) instance 2 without a fallback key of its own: retries stay within its own keys;
* (c) a glossary / metadata / rolling summary / QA / refinement / Vision request of instance 2
  uses only its own keys of that pool, or its own Translation keys when it has none there;
* (d) instance 1 in the main window (``instance_worker.restrict_keys``) never uses instance 2's
  keys of any pool, and the config and the client's pools are unchanged afterwards.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_instance_key_isolation.py
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _path in (str(TESTS_DIR), str(SRC_DIR)):
    if _path not in sys.path:  # CI has no tests/conftest.py
        sys.path.insert(0, _path)
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")  # before unified_api_client is imported

import instance_plan  # noqa: E402
from _fake_key_server import FakeKeyServer  # noqa: E402

MODEL = "gpt-4o-mini"
MAIN_KEY = "sk-main-gui-key"
ONE = {"tr-one", "tr-one-b", "fb-one", "gl-one", "rf-one", "md-one", "ai-one", "rs-one", "qa-one"}
TWO = {"tr-two", "tr-two-b", "fb-two", "gl-two", "rf-two", "md-two", "ai-two", "rs-two", "qa-two"}
#: dedicated pools with chat requests: (pool, request context, key prefix)
DEDICATED = [
    ("glossary", "glossary", "gl"),
    ("glossary_refinement", "refinement", "rf"),
    ("metadata", "book_title", "md"),
    ("ai_truncation_detection", "qa_truncation", "ai"),
    ("rolling_summary", "summary", "rs"),
    ("qa_scan", "truncation", "qa"),
]


def _key(name, instance=None, **extra):
    entry = {"api_key": name, "model": MODEL, "enabled": True, "cooldown": 1}
    if instance is not None:
        entry["instance"] = instance
    entry.update(extra)
    return entry


def full_config(base_url, **overrides):
    """The main window's config: two instances' keys in every pool, the main GUI key, its fallback."""
    config = {
        "model": MODEL, "api_key": MAIN_KEY, "use_main_key_fallback": True,
        "use_custom_openai_endpoint": True, "openai_base_url": base_url,
        "use_multi_api_keys": True, "force_key_rotation": True, "rotation_frequency": 1,
        "multi_api_keys": [_key("tr-one"), _key("tr-two", 2), _key("tr-two-b", 2), _key("tr-one-b")],
        "use_fallback_keys": True, "fallback_keys": [_key("fb-one"), _key("fb-two", 2)],
        "delay": 0, "thread_submission_delay": 0, "batch_translation": False,
    }
    for pool, _context, prefix in DEDICATED:
        config_key, toggle_key, _use_envs, _keys_envs = instance_plan.POOLS[pool]
        config[config_key] = [_key(f"{prefix}-one"), _key(f"{prefix}-two", 2)]
        config[toggle_key] = True
    config.update(overrides)
    return config


MESSAGES = [{"role": "system", "content": "Translate to English."},
            {"role": "user", "content": "<p>그는 천천히 문을 열었다.</p>"}]


@pytest.fixture
def server():
    with FakeKeyServer() as fake:
        yield fake


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """The real client in a sandbox: environment and key pools restored afterwards, payloads in
    tmp_path, no key scope (each test sets its own)."""
    import job_runner
    import unified_api_client

    for name in instance_plan.KEY_ENV + (instance_plan.ENV_INSTANCE, instance_plan.ENV_KEY_INSTANCE):
        monkeypatch.delenv(name, raising=False)
    payloads = tmp_path / "Payloads"
    payloads.mkdir()
    monkeypatch.setattr(unified_api_client, "_payloads_resolved_dir", str(payloads))
    monkeypatch.setenv("GLOSSARION_LOGISTICS", "0")
    with job_runner.scoped_process_state(keywise=True, isolate_key_pools=True, restore_cwd=True):
        yield _Runtime(tmp_path, monkeypatch)


class _Runtime:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.builds = 0

    def run_env(self, config):
        """``run_env.build_translation_env`` of a HeadlessOwner built from *config* (the env a
        translation run exports), built in a scrubbed sandbox."""
        from _headless_env import run_envs

        self.builds += 1
        return run_envs(self.tmp / f"owner{self.builds}", self.monkeypatch, copy.deepcopy(config),
                        api_key=config.get("api_key", ""))["translation"]

    def start(self, env, config):
        """Export *env* and the key pools of *config* as the run does (``_get_environment_variables``
        + ``key_pools.apply_key_pools_to_runtime``); quick retries."""
        import key_pools
        import large_env

        large_env.update_env(env)
        os.environ.update({"SEND_INTERVAL_SECONDS": "0", "SEND_INTERVAL_MAX_SECONDS": "",
                           "THREAD_SUBMISSION_DELAY_SECONDS": "0", "MAX_RETRIES": "2", "RETRY_TIMEOUT": "0"})
        key_pools.apply_key_pools_to_runtime(config, export_env=False)

    def client(self, api_key):
        from unified_api_client import UnifiedClient

        return UnifiedClient(api_key=api_key, model=MODEL, output_dir=str(self.tmp / "out"))


def _scope(monkeypatch, instance):
    """Inside instance *instance* of a multi-instance run (what instance_worker sets up)."""
    monkeypatch.setenv(instance_plan.ENV_INSTANCE, str(instance))
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, str(instance))


def _send(client, context="translation"):
    content, finish = client.send(copy.deepcopy(MESSAGES), context=context)
    return content, finish


# ---------------------------------------------------------------------------
# (e) single-instance runs: unchanged
# ---------------------------------------------------------------------------

def test_single_instance_run_uses_every_key_and_the_main_key_retry(server, runtime):
    """No key scope: the instance numbers are ignored. A blocked Translation key retries with the
    main GUI key, then every fallback key in order (instance 1's and 2's alike), as before."""
    config = full_config(server.base_url)
    env = runtime.run_env(config)
    assert json.loads(env["FALLBACK_KEYS"]) == config["fallback_keys"] and env["API_KEY"] == MAIN_KEY
    runtime.start(env, config)
    for key in ("tr-one", "tr-two", "tr-two-b", "tr-one-b", MAIN_KEY, "fb-one"):
        server.behaviours[key] = "prohibited"
    content, finish = _send(runtime.client(MAIN_KEY))
    assert finish == "stop" and "fb-two" in content
    keys = server.keys()
    assert keys[1:] == [MAIN_KEY, "fb-one", "fb-two"] and keys[0].startswith("tr-")
    # the client reads the environment as it is
    import unified_api_client

    assert unified_api_client._key_pool_env("FALLBACK_KEYS") == os.environ["FALLBACK_KEYS"]
    assert unified_api_client._key_pool_env("USE_MAIN_KEY_FALLBACK") == "1"
    # a glossary request uses the whole Glossary pool
    server.clear()
    content, _finish = _send(runtime.client(MAIN_KEY), "glossary")
    assert server.keys()[0] in ("gl-one", "gl-two")
    pool = unified_api_client.UnifiedClient._glossary_key_pool
    assert sorted(k.api_key for k in pool.keys) == ["gl-one", "gl-two"]


# ---------------------------------------------------------------------------
# (a) / (b) instance 2: the Fallback pool and the main GUI key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["snapshot", "leaked"])
def test_instance_two_falls_back_only_to_its_own_fallback_keys(server, runtime, monkeypatch, source):
    """Instance 2's Translation keys are blocked: it tries its own fallback key only.

    ``snapshot``: the run's environment comes from instance 2's config snapshot (the desktop
    instance process). ``leaked``: it comes from the main window's full config and the client is
    even built with the main GUI key (anything that might leak in): the client's key scope still
    keeps every other instance's key and the main GUI key out."""
    config = full_config(server.base_url)
    run_config = instance_plan.config_for_instance(config, 2) if source == "snapshot" else config
    env = runtime.run_env(run_config)
    _scope(monkeypatch, 2)
    runtime.start(env, run_config)
    for key in ("tr-two", "tr-two-b", "tr-one", "tr-one-b"):
        server.behaviours[key] = "prohibited"
    content, finish = _send(runtime.client(run_config["api_key"]))
    assert finish == "stop" and "fb-two" in content
    keys = server.keys()
    assert set(keys) <= TWO and "fb-two" in keys and keys[0] in ("tr-two", "tr-two-b"), keys
    assert MAIN_KEY not in keys and "fb-one" not in keys


@pytest.mark.parametrize("source", ["snapshot", "leaked"])
def test_instance_two_without_a_fallback_key_stays_within_its_own_keys(server, runtime, monkeypatch, source):
    """Instance 2 has no fallback key: a blocked request is not retried with instance 1's fallback
    key nor the main GUI key; a rate limit rotates only among its own Translation keys."""
    config = full_config(server.base_url, fallback_keys=[_key("fb-one")])
    run_config = instance_plan.config_for_instance(config, 2) if source == "snapshot" else config
    if source == "snapshot":
        assert run_config["use_fallback_keys"] is False and run_config["fallback_keys"] == []
        assert run_config["api_key"] == "tr-two" and run_config["use_main_key_fallback"] is False
    env = runtime.run_env(run_config)
    _scope(monkeypatch, 2)
    runtime.start(env, run_config)
    server.behaviours.update({"tr-two": "prohibited", "tr-two-b": "prohibited"})
    _content, finish = _send(runtime.client(run_config["api_key"]))
    assert finish != "stop"
    assert set(server.keys()) <= {"tr-two", "tr-two-b"}, server.keys()
    # a rate limit: rotation inside instance 2's Translation pool
    server.clear()
    server.behaviours.update({"tr-two": "429", "tr-two-b": "ok"})
    content, finish = _send(runtime.client(run_config["api_key"]))
    assert finish == "stop" and "tr-two-b" in content
    assert set(server.keys()) <= {"tr-two", "tr-two-b"} and server.keys()[-1] == "tr-two-b", server.keys()


def test_the_same_config_without_a_scope_does_reach_the_main_key(server, runtime):
    """Control for the two tests above: without the instance scope the very same blocked request
    goes on to the main GUI key and instance 1's fallback key (the paths the scope closes)."""
    config = full_config(server.base_url, fallback_keys=[_key("fb-one")])
    env = runtime.run_env(config)
    runtime.start(env, config)
    for key in ("tr-one", "tr-two", "tr-two-b", "tr-one-b", MAIN_KEY):
        server.behaviours[key] = "prohibited"
    _send(runtime.client(MAIN_KEY))
    assert MAIN_KEY in server.keys() and "fb-one" in server.keys()


# ---------------------------------------------------------------------------
# (c) dedicated pools inside instance 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["snapshot", "leaked"])
@pytest.mark.parametrize("pool, context, prefix", DEDICATED)
def test_instance_two_dedicated_pool_requests(server, runtime, monkeypatch, source, pool, context, prefix):
    """A request of a dedicated pool's context uses instance 2's key of that pool; with none of
    its own there (the pool is off for it) it uses instance 2's Translation keys, never instance 1's."""
    config_key, toggle_key, _use_envs, _keys_envs = instance_plan.POOLS[pool]
    for own in (True, False):
        server.clear()
        config = full_config(server.base_url)
        if not own:
            config[config_key] = [_key(f"{prefix}-one")]
        run_config = instance_plan.config_for_instance(config, 2) if source == "snapshot" else config
        env = runtime.run_env(run_config)
        _scope(monkeypatch, 2)
        runtime.start(env, run_config)
        content, finish = _send(runtime.client(run_config["api_key"]), context)
        monkeypatch.delenv(instance_plan.ENV_KEY_INSTANCE)
        monkeypatch.delenv(instance_plan.ENV_INSTANCE)
        keys = server.keys()
        assert finish == "stop", (pool, own, content)
        if own:
            assert keys == [f"{prefix}-two"], (pool, keys)
        else:
            assert keys and set(keys) <= {"tr-two", "tr-two-b"}, (pool, keys)


def test_instance_two_glossary_prohibited_content_stays_in_instance_two(server, runtime, monkeypatch):
    """A blocked glossary request: the glossary / fallback retries use instance 2's keys only."""
    config = full_config(server.base_url)
    env = runtime.run_env(config)  # the main window's full environment (worst case)
    _scope(monkeypatch, 2)
    runtime.start(env, config)
    server.behaviours.update({"gl-two": "prohibited", "gl-one": "ok"})
    _send(runtime.client(MAIN_KEY), "glossary")
    keys = server.keys()
    assert keys[0] == "gl-two" and set(keys) <= TWO, keys


# ---------------------------------------------------------------------------
# (d) instance 1 in the main window
# ---------------------------------------------------------------------------

def test_instance_one_in_the_main_window_never_uses_instance_two_keys(server, runtime):
    """``instance_worker.restrict_keys(1, config)`` (instance_runs, for instance 1's run in the main
    window): every request of every pool uses instance 1's keys (the main GUI key is instance 1's);
    afterwards the config and the client's pools are exactly as before."""
    import instance_worker
    import key_pools
    import unified_api_client

    client_cls = unified_api_client.UnifiedClient
    config = full_config(server.base_url, use_tts_keys=True, tts_keys=[_key("tts-two", 2)])
    saved = copy.deepcopy(config)
    env = runtime.run_env(config)  # what the main window exports: every instance's keys
    runtime.start(env, config)
    key_pools.apply_key_pools_to_runtime(config)
    attrs = ("_api_key_pool", "_glossary_key_pool", "_glossary_refinement_key_pool", "_qa_scan_key_pool",
             "_rolling_summary_key_pool", "_in_memory_multi_keys", "_in_memory_glossary_keys")
    before = {name: getattr(client_cls, name) for name in attrs}
    assert sorted(k.api_key for k in before["_api_key_pool"].keys) == ["tr-one", "tr-one-b", "tr-two", "tr-two-b"]
    env_before = {name: os.environ.get(name) for name in instance_plan.KEY_ENV}

    restore = instance_worker.restrict_keys(1, config)
    try:
        assert instance_plan.key_scope() == 1
        # the run exports the full environment again (run_env reads the main window's config)
        runtime.start(env, config)
        assert client_cls._api_key_pool is not before["_api_key_pool"]
        assert sorted(k.api_key for k in client_cls._api_key_pool.keys) == ["tr-one", "tr-one-b"]
        # Translation keys blocked: the main GUI key (instance 1's) and instance 1's fallback key
        server.behaviours.update({"tr-one": "prohibited", "tr-one-b": "prohibited", MAIN_KEY: "prohibited"})
        content, finish = _send(runtime.client(MAIN_KEY))
        assert finish == "stop" and "fb-one" in content
        assert server.keys()[-2:] == [MAIN_KEY, "fb-one"]
        server.behaviours.clear()
        for _pool, context, prefix in DEDICATED:
            _send(runtime.client(MAIN_KEY), context)
            assert server.keys()[-1] == f"{prefix}-one", context
        # a pool with no instance 1 key (Audio / TTS holds only an instance 2 key) is off for it
        os.environ["USE_TTS_KEYS"] = "1"
        assert unified_api_client._key_pool_env("USE_TTS_KEYS", "0") == "0"
        assert client_cls._in_memory_tts_keys == []
        assert set(server.keys()) <= ONE | {MAIN_KEY}, server.keys()
        # the Multi API Key Manager opened now would keep its own pool (test_key_instance_column)
    finally:
        restore()
    # everything is back: the scope, the very same pool objects, the config, the environment
    assert instance_plan.key_scope() is None
    assert {name: getattr(client_cls, name) for name in attrs} == before
    assert all(getattr(client_cls, name) is value for name, value in before.items())
    assert sorted(k.api_key for k in client_cls._api_key_pool.keys) == ["tr-one", "tr-one-b", "tr-two", "tr-two-b"]
    assert config == saved
    os.environ.pop("USE_TTS_KEYS", None)
    env_before.pop("USE_TTS_KEYS", None)
    assert {name: os.environ.get(name) for name in instance_plan.KEY_ENV if name != "USE_TTS_KEYS"} == env_before
    # and the client reads every key again
    assert unified_api_client._key_pool_env("FALLBACK_KEYS") == os.environ["FALLBACK_KEYS"]
    assert sorted(k.api_key for k in client_cls._glossary_key_pool.keys) == ["gl-one", "gl-two"]


# ---------------------------------------------------------------------------
# (a) a real instance process
# ---------------------------------------------------------------------------

#: Runs instance_worker.main in its own process: the real command-line path, only the payload
#: folder is the test's (a source checkout writes it next to unified_api_client.py otherwise).
CHILD_BOOT = r'''
import sys
payloads, job_path, src = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
import instance_worker

_run_job = instance_worker.run_job


def run_job(job, host=None, parent=None):
    import unified_api_client

    unified_api_client._payloads_resolved_dir = payloads
    return _run_job(job, host, parent=parent)


instance_worker.run_job = run_job
instance_worker.main(job_path)
'''


def _write_epub(path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("mi-key-isolation")
    book.set_title("책 제목")
    book.set_language("ko")
    chapter = epub.EpubHtml(title="제1장", file_name="chapter_001.xhtml", lang="ko")
    chapter.content = "<h1>제1장</h1>\n<p>그는 천천히 문을 열고 어두운 복도를 따라 걸어갔다.</p>"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


@pytest.mark.parametrize("own_fallback", [True, False])
def test_instance_process_never_reaches_instance_one_keys(server, tmp_path, own_fallback):
    """Instance 2 as a real instance_worker process whose Translation keys are blocked, started
    from a parent environment that holds the main window's keys (main GUI key, every pool's list):
    only instance 2's keys ever reach the server; its own fallback key rescues the chapter."""
    pytest.importorskip("ebooklib")
    import instance_worker
    from config_store import save_config_file

    for sub in ("app", "data", "logs", "library", "out", "cwd", "job", "wd", "payloads"):
        (tmp_path / sub).mkdir()
    book = tmp_path / "book.epub"
    _write_epub(book)
    fallback = [_key("fb-one"), _key("fb-two", 2)] if own_fallback else [_key("fb-one")]
    config = full_config(server.base_url, fallback_keys=fallback, output_directory=str(tmp_path / "out"),
                         auto_update_check=False, enable_auto_glossary=False, auto_glossary_mode="off",
                         scan_phase_enabled=False, contextual=False, token_limit=200000,
                         unified_glossary_enabled=False, enable_unified_glossary=False)
    snapshot = instance_plan.config_for_instance(config, 2)
    assert MAIN_KEY not in json.dumps(snapshot)
    save_config_file(snapshot, str(tmp_path / "job" / "config.json"), backup=False)
    job = {"version": instance_worker.JOB_VERSION, "instance": 2, "files": [str(book)],
           "config_path": str(tmp_path / "job" / "config.json"), "env": {"FALLBACK_KEYS": json.dumps(config["fallback_keys"])},
           "watchdog_dir": str(tmp_path / "wd" / "instance_2"), "stop_file": str(tmp_path / "job" / "stop.flag"),
           "log_path": str(tmp_path / "logs" / "run_instance_2.log"), "parent_pid": os.getpid()}
    job_path = tmp_path / "job" / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    (tmp_path / "boot.py").write_text(CHILD_BOOT, encoding="utf-8")
    # the main window's environment: its main GUI key and every pool's keys
    parent = dict(os.environ)
    parent.update({"API_KEY": MAIN_KEY, "OPENAI_API_KEY": MAIN_KEY, "GEMINI_API_KEY": MAIN_KEY,
                   "OPENAI_OR_Gemini_API_KEY": MAIN_KEY, "USE_MAIN_KEY_FALLBACK": "1",
                   "USE_FALLBACK_KEYS": "1", "FALLBACK_KEYS": json.dumps(config["fallback_keys"]),
                   "USE_GLOSSARY_KEYS": "1", "GLOSSARY_API_KEYS": json.dumps(config["glossary_keys"]),
                   "MULTI_API_KEYS": json.dumps(config["multi_api_keys"]), "GLOSSARION_KEY_INSTANCE": "1",
                   "GLOSSARION_APP_DIR": str(tmp_path / "app"), "GLOSSARION_DATA_DIR": str(tmp_path / "data"),
                   "GLOSSARION_LOG_DIR": str(tmp_path / "logs"), "GLOSSARION_LIBRARY_DIR": str(tmp_path / "library"),
                   "GLOSSARION_HTTP_LOG": "0", "GLOSSARION_LOGISTICS": "0"})
    env = instance_worker.child_environment(parent, 2)
    for key in ("tr-two", "tr-two-b"):
        server.behaviours[key] = "prohibited"
    proc = subprocess.run([sys.executable, str(tmp_path / "boot.py"), str(tmp_path / "payloads"), str(job_path),
                           str(SRC_DIR)], env=env, cwd=str(tmp_path / "cwd"), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=480)
    keys = server.keys()
    log = proc.stdout[-6000:]
    print(f"instance 2 process (own fallback key: {own_fallback}): exit {proc.returncode}, keys {keys}")
    assert keys, log
    assert set(keys) <= TWO, (keys, log)
    assert not set(keys) & (ONE | {MAIN_KEY}), (keys, log)
    assert {"tr-two", "tr-two-b"} & set(keys), (keys, log)
    if own_fallback:
        assert "fb-two" in keys, (keys, log)
        assert proc.returncode == instance_worker.EXIT_OK, log
    else:
        assert "fb-two" not in keys, (keys, log)
