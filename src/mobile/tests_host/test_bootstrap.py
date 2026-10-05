"""Host tests for the mobile runtime bootstrap, router, self-test and app page.

Run from src/mobile:  python -m pytest -p no:cacheprovider tests_host/test_bootstrap.py -q

Nothing here opens a window: the app page is built against an in-memory fake
Flet client (messages are msgpack-encoded exactly like the real transport).
Parts whose packages are missing on the host are skipped.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import webbrowser
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import runtime_bootstrap as rb  # noqa: E402
from glossarion_mobile.services import secure_keys  # noqa: E402
from glossarion_mobile.ui.router import Router, parse_route  # noqa: E402

CONTRACT_KEYS = (
    "GLOSSARION_MOBILE",
    "GLOSSARION_NO_PROCESSES",
    "GLOSSARION_HEADLESS_KEY_MANAGER",
    "GLOSSARION_APP_DIR",
    "GLOSSARION_DATA_DIR",
    "CONFIG_FILE",
    "HOME",
    "OUTPUT_DIRECTORY",
    "GLOSSARION_LIBRARY_DIR",
    "GLOSSARION_LOG_DIR",
    "GLOSSARION_MODEL_CATALOG_CACHE",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "TIKTOKEN_CACHE_DIR",
    "GLOSSARION_HTTP_LOG",
    "GLOSSARION_OAUTH_RETURN_URL",
    "USE_ASYNC_CHAPTER_EXTRACTION",
    "PDF_EXTRACTION_WORKERS",
    "QA_USE_THREAD_EXECUTOR",
)

_ENV_INPUTS = (
    "GLOSSARION_BACKEND_DIR",
    "GLOSSARION_SIMULATE_PLATFORM",
    "GLOSSARION_ORIGINAL_HOME",
    "FLET_ASSETS_DIR",
    "FLET_PLATFORM",
    "GLOSSARION_FORCE_NATIVE_STUB",
)


# Blocking helpers the app runs on UiDispatcher.run_in_thread threads (sign-in status reads, folder
# sizes, config/chat I/O). They may still be running when a test's app stops.
_APP_IO_THREADS = ("gl-settings-io", "gl-chat-io", "gl-models-io", "gl-files", "gl-worker")


def _join_app_io_threads(timeout: float = 15.0) -> None:
    """Let in-flight app I/O threads finish before the test's env is undone: a status read that ran
    after ``rb.reset(restore_env=True)`` would resolve the auth token stores (``AUTH*_TOKEN_FILE``,
    ``GLOSSARION_TOKEN_DIR``) to the developer's real ~/.glossarion."""
    deadline = time.monotonic() + timeout
    for thread in list(threading.enumerate()):
        if thread is threading.current_thread() or thread.name not in _APP_IO_THREADS:
            continue
        thread.join(max(0.0, deadline - time.monotonic()))


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Temp FLET_APP_STORAGE_* dirs; bootstrap() (and any installed encryption keys) undone after the test."""
    rb.reset(restore_env=True)
    secure_keys.reset()
    dirs = {}
    for name in ("data", "cache", "temp"):
        d = tmp_path / "storage" / name
        d.mkdir(parents=True)
        dirs[name] = d
        monkeypatch.setenv(f"FLET_APP_STORAGE_{name.upper()}", str(d))
    for key in _ENV_INPUTS:
        monkeypatch.delenv(key, raising=False)
    yield dirs
    _join_app_io_threads()
    secure_keys.reset()
    rb.reset(restore_env=True)


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


# --------------------------------------------------------------------------
# Env contract
# --------------------------------------------------------------------------


def test_env_contract_applied(storage, capsys):
    paths = rb.bootstrap(app_dir=APP_DIR, force=True)
    data, cache, temp = storage["data"], storage["cache"], storage["temp"]

    assert paths.platform == "desktop"
    assert _same(paths.data, data) and _same(paths.cache, cache) and _same(paths.temp, temp)
    assert not paths.dev_storage
    expected = {
        "GLOSSARION_MOBILE": "1",
        "GLOSSARION_NO_PROCESSES": "1",
        "GLOSSARION_HEADLESS_KEY_MANAGER": "1",
        "GLOSSARION_APP_DIR": str(paths.data),
        "GLOSSARION_DATA_DIR": str(paths.data),
        "CONFIG_FILE": str(paths.data / "config.json"),
        "HOME": str(paths.data / "home"),
        "OUTPUT_DIRECTORY": str(paths.data / "Output"),  # desktop/Android: docs == data
        "GLOSSARION_LIBRARY_DIR": str(paths.data / "Library"),
        "GLOSSARION_LOG_DIR": str(paths.data / "logs"),
        "GLOSSARION_MODEL_CATALOG_CACHE": str(paths.cache / "model_catalog_cache.json"),
        "XDG_CACHE_HOME": str(paths.cache),
        "TMPDIR": str(paths.temp),
        "TEMP": str(paths.temp),
        "TMP": str(paths.temp),
        "TIKTOKEN_CACHE_DIR": str(paths.cache / "tiktoken"),
        "GLOSSARION_HTTP_LOG": "0",
        "GLOSSARION_OAUTH_RETURN_URL": "glossarion://app/oauth/return",
        "USE_ASYNC_CHAPTER_EXTRACTION": "0",
        "PDF_EXTRACTION_WORKERS": "1",
        "QA_USE_THREAD_EXECUTOR": "1",
    }
    assert set(expected) == set(CONTRACT_KEYS)
    for key, value in expected.items():
        assert os.environ.get(key) == value, key
    assert paths.env_contract().items() >= expected.items()

    try:
        import certifi
    except ImportError:
        certifi = None
    if certifi is not None:
        assert os.environ["SSL_CERT_FILE"] == certifi.where()
        assert os.environ["REQUESTS_CA_BUNDLE"] == certifi.where()

    assert os.path.samefile(os.getcwd(), paths.data)
    assert rb.current_thread_stack_size() == rb.THREAD_STACK_SIZE
    assert rb.current_thread_stack_size() == rb.THREAD_STACK_SIZE  # reading must not reset it
    for directory in paths.writable_dirs().values():
        assert directory.is_dir()
    assert str(paths.backend_dir) == sys.path[0]

    err = capsys.readouterr().err
    boot_lines = [ln for ln in err.splitlines() if ln.startswith("GLOSSARION_BOOT ")]
    assert len(boot_lines) == 1
    payload = json.loads(boot_lines[0].split(" ", 1)[1])
    assert payload["platform"] == "desktop" and payload["backend"] == "repo"
    assert payload["version"] and payload["build"] == _expected_build(payload["version"])
    assert len(boot_lines[0]) <= rb.MARKER_MAX_CHARS + 32

    rb.flush_logs()
    run_log = (paths.logs / "run.log").read_text(encoding="utf-8")
    assert "GLOSSARION_BOOT" in run_log
    assert (paths.logs / "crash.log").read_text(encoding="utf-8").startswith("=== boot ")


def _expected_build(version: str) -> int:
    major, minor, patch = (int(p) for p in version.split(".")[:3])
    return major * 1_000_000 + minor * 10_000 + patch * 100


def test_app_version_matches_src():
    text = (SRC_DIR / "app_version.py").read_text(encoding="utf-8-sig")
    info = rb.app_version(SRC_DIR)
    assert f'APP_VERSION = "{info["version"]}"' in text
    assert info["build"] == _expected_build(info["version"])


def test_app_version_from_pyc_only_bundle(tmp_path):
    """flet build compiles the app dir (compileall -b) and deletes the .py sources."""
    import compileall

    (tmp_path / "app_version.py").write_text('APP_VERSION = "9.13.6"\n', encoding="utf-8")
    (tmp_path / "_bundle_info.py").write_text(
        "BUILD_VERSION = '9.13.6'\nBUNDLE_SHA256 = '" + "ab" * 32 + "'\n", encoding="utf-8"
    )
    compileall.compile_dir(str(tmp_path), quiet=1, legacy=True)
    for source in tmp_path.glob("*.py"):
        source.unlink()
    before = set(sys.modules)
    assert rb.app_version(tmp_path) == {"version": "9.13.6", "build": 9130600, "bundle": "ab" * 6}
    assert set(sys.modules) == before  # read without importing


def test_bootstrap_is_idempotent_and_reset_restores(storage):
    before_env = dict(os.environ)
    before_cwd = os.getcwd()
    first = rb.bootstrap(app_dir=APP_DIR, force=True)
    again = rb.bootstrap(app_dir=APP_DIR)  # Android process reuse: no second setup
    assert again is first
    rb.reset(restore_env=True)
    assert dict(os.environ) == before_env
    assert os.getcwd() == before_cwd
    assert rb.get_state() is None
    assert "glossarion-inapp" not in (webbrowser._tryorder or [])  # type: ignore[attr-defined]


def test_ios_docs_use_original_home_documents(storage, tmp_path, monkeypatch):
    home = tmp_path / "sandbox"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GLOSSARION_SIMULATE_PLATFORM", "ios")
    paths = rb.bootstrap(app_dir=APP_DIR, force=True)
    docs = home / "Documents" / "Glossarion"
    assert paths.platform == "ios" and paths.is_device
    assert _same(paths.docs, docs)
    assert os.environ["OUTPUT_DIRECTORY"] == str(paths.docs / "Output")
    assert os.environ["GLOSSARION_LIBRARY_DIR"] == str(paths.docs / "Library")
    assert os.environ["HOME"] == str(paths.data / "home")
    assert os.environ["GLOSSARION_ORIGINAL_HOME"] == str(home)
    # A second (forced) bootstrap must still find Documents via the captured HOME.
    again = rb.bootstrap(app_dir=APP_DIR, force=True)
    assert _same(again.docs, docs)


def test_dev_storage_fallback(tmp_path, monkeypatch):
    for name in ("DATA", "CACHE", "TEMP"):
        monkeypatch.delenv(f"FLET_APP_STORAGE_{name}", raising=False)
    monkeypatch.delenv("GLOSSARION_BACKEND_DIR", raising=False)
    fake_app = tmp_path / "x" / "mobile" / "app"
    fake_app.mkdir(parents=True)
    paths = rb.resolve_paths(fake_app, platform="desktop")
    assert paths.dev_storage
    assert _same(paths.data, tmp_path / "x" / "mobile" / "storage" / "data")
    assert _same(paths.temp, tmp_path / "x" / "mobile" / "storage" / "temp")


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------


def test_backend_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARION_BACKEND_DIR", raising=False)
    # dev: the repo src/ (main.py's parents[2] has TransateKRtoEN.py)
    backend, source = rb.resolve_backend_dir(APP_DIR)
    assert source == "repo" and _same(backend, SRC_DIR)

    # bundle: <app>/backend when not inside the repo
    fake_app = tmp_path / "a" / "b" / "app"
    fake_app.mkdir(parents=True)
    assert rb.resolve_backend_dir(fake_app) == (None, "missing")
    (fake_app / "backend").mkdir()
    backend, source = rb.resolve_backend_dir(fake_app)
    assert source == "bundle" and _same(backend, fake_app / "backend")

    # env override wins over both
    override = tmp_path / "collected"
    override.mkdir()
    monkeypatch.setenv("GLOSSARION_BACKEND_DIR", str(override))
    backend, source = rb.resolve_backend_dir(APP_DIR)
    assert source == "env" and _same(backend, override)

    # a missing override is an error, never a silent fallback
    monkeypatch.setenv("GLOSSARION_BACKEND_DIR", str(tmp_path / "nope"))
    assert rb.resolve_backend_dir(APP_DIR) == (None, "env-missing")


def test_no_backend_or_flet_import_at_import_time(tmp_path):
    """Importing the bootstrap/router/self-test modules and running bootstrap()
    must not import any backend module (or Flet)."""
    env = dict(os.environ)
    for name in ("data", "cache", "temp"):
        (tmp_path / name).mkdir()
        env[f"FLET_APP_STORAGE_{name.upper()}"] = str(tmp_path / name)
    env.pop("GLOSSARION_BACKEND_DIR", None)
    env["PYTHONIOENCODING"] = "utf-8"
    script = textwrap.dedent(
        f"""
        import sys, json
        sys.path.insert(0, {str(APP_DIR)!r})
        import glossarion_mobile
        from glossarion_mobile import runtime_bootstrap as rb
        import glossarion_mobile.ui.router, glossarion_mobile.diagnostics.selftest
        import glossarion_mobile.services.native
        rb.bootstrap()
        watch = list(rb.WARM_IMPORT_MODULES) + ["flet", "flet_glossarion_native", "PySide6", "translator_gui"]
        print(json.dumps(sorted(m for m in watch if m in sys.modules)))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
    assert "GLOSSARION_BOOT " in out.stderr


# --------------------------------------------------------------------------
# tiktoken seeding
# --------------------------------------------------------------------------


def _write_manifest(directory: Path, entries: dict[str, bytes]) -> None:
    lines = ["format = 1", ""]
    for index, (name, data) in enumerate(entries.items()):
        lines += [
            f"[encodings.enc{index}]",
            f'cache_key = "{name}"',
            f'file = "{name}"',
            f'sha256 = "{hashlib.sha256(data).hexdigest()}"',
            "",
        ]
    (directory / rb.TIKTOKEN_MANIFEST).write_text("\n".join(lines), encoding="utf-8")


def test_seed_tiktoken_cache_with_manifest(tmp_path):
    src = tmp_path / "assets"
    dst = tmp_path / "cache" / "tiktoken"
    src.mkdir()
    files = {"a" * 40: b"first bpe", "b" * 40: b"second bpe file"}
    for name, data in files.items():
        (src / name).write_bytes(data)
    _write_manifest(src, files)

    report = rb.seed_tiktoken_cache(src, dst)
    assert report["manifest"] and report["copied"] == 2 and not report["bad_hash"]
    assert {p.name for p in dst.iterdir()} == set(files)
    again = rb.seed_tiktoken_cache(src, dst)
    assert again["copied"] == 0 and again["present"] == 2

    # a corrupted asset never replaces/creates a cache entry
    (dst / ("a" * 40)).unlink()
    (src / ("a" * 40)).write_bytes(b"corrupted!")
    bad = rb.seed_tiktoken_cache(src, dst)
    assert bad["bad_hash"] == ["a" * 40]
    assert not (dst / ("a" * 40)).exists()


def test_seed_tiktoken_cache_real_assets(tmp_path):
    src = APP_DIR / "assets" / "tiktoken"
    if not (src / rb.TIKTOKEN_MANIFEST).is_file():
        pytest.skip("app/assets/tiktoken not generated (run tools/prepare_assets.py)")
    report = rb.seed_tiktoken_cache(src, tmp_path / "tiktoken")
    assert report["copied"] == 2 and not report["missing"] and not report["bad_hash"], report
    for url, digest in rb.TIKTOKEN_ENCODINGS.values():
        target = tmp_path / "tiktoken" / rb.tiktoken_cache_key(url)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest


# --------------------------------------------------------------------------
# webbrowser controller
# --------------------------------------------------------------------------


def test_webbrowser_controller_marshals_to_opener(storage):
    rb.bootstrap(app_dir=APP_DIR, force=True)
    assert webbrowser.get() is rb.IN_APP_BROWSER
    assert webbrowser.open("https://example.invalid/early") is True  # queued: no opener yet
    assert rb.IN_APP_BROWSER.pending == ["https://example.invalid/early"]

    opened: list[tuple[str, str]] = []
    rb.set_url_opener(lambda url: opened.append((url, threading.current_thread().name)))
    assert [u for u, _ in opened] == ["https://example.invalid/early"]

    worker = threading.Thread(target=webbrowser.open, args=("http://127.0.0.1:1455/auth",), name="oauth-worker")
    worker.start()
    worker.join(10)
    assert opened[-1] == ("http://127.0.0.1:1455/auth", "oauth-worker")
    assert rb.IN_APP_BROWSER.pending == []


# --------------------------------------------------------------------------
# Router whitelist
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, path, query",
    [
        (None, "/", {}),
        ("", "/", {}),
        ("/", "/", {}),
        ("/__selftest__?suite=smoke", "/__selftest__", {"suite": "smoke"}),
        ("/__selftest__/", "/__selftest__", {}),
        ("glossarion://app/__selftest__?suite=smoke", "/__selftest__", {"suite": "smoke"}),
        ("GLOSSARION://APP/oauth/return?p=spike&nonce=ab12", "/oauth/return", {"p": "spike", "nonce": "ab12"}),
        ("/oauth/return?p=authgpt", "/oauth/return", {"p": "authgpt"}),
    ],
)
def test_router_accepts_whitelisted(raw, path, query):
    match = parse_route(raw)
    assert match is not None
    assert match.path == path
    assert match.query == query


@pytest.mark.parametrize(
    "raw",
    [
        "content://com.android.providers.downloads.documents/document/raw%3A%2Fstorage%2Fx.epub",
        "file:///storage/emulated/0/Download/book.epub",
        "FILE:///private/var/mobile/x.pdf",
        "intent://scan/#Intent;scheme=zxing;end",
        "javascript:alert(1)",
        "https://example.com/__selftest__",
        "http://127.0.0.1:1455/auth/callback?code=x",
        "otherapp://app/__selftest__",
        "glossarion://evil/__selftest__",
        "glossarion://app:99/__selftest__",
        "/document/raw%3A%2Fstorage%2Femulated%2F0%2Fx.epub",  # Flet's path-only form of a content URI
        "/storage/emulated/0/Download/book.epub",  # path-only form of a file URI
        "/libraryx",  # unknown path ("/library" is whitelisted since U1)
        "/__selftest__/../settings",
        "/oauth%2Freturn",
        "//evil.example/oauth/return",
        "relative/path",
        "/a\\b",
        "/" + "x" * 3000,
    ],
)
def test_router_ignores_everything_else(raw):
    assert parse_route(raw) is None


def test_router_history_records_accept_and_reject():
    router = Router()
    t0 = time.time()
    assert router.handle("/") is not None
    assert router.handle("content://x/y") is None
    history = router.history
    assert [r.accepted for r in history] == [True, False]
    assert router.rejected_since(t0)[0].reason.startswith("blocked scheme")
    assert len(router.seen_since(t0)) == 2


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def test_selftest_smoke_on_host(storage, capsys):
    from glossarion_mobile.diagnostics import selftest

    paths = rb.bootstrap(app_dir=APP_DIR, force=True)
    capsys.readouterr()
    result = selftest.run_selftest("smoke", strict=False)
    json.dumps(result)  # JSON-serialisable

    by_name = {c["name"]: c for c in result["checks"]}
    assert set(by_name) == {name for name, _ in selftest.SUITES["smoke"]}
    failed = {n: c.get("error") for n, c in by_name.items() if c["status"] == "fail"}
    assert not failed, failed
    for name in ("env_contract", "writable_dirs", "thread_stack"):
        assert by_name[name]["status"] == "pass", by_name[name]
    assert result["ok"] is True

    err = capsys.readouterr().err
    marker = [ln for ln in err.splitlines() if ln.startswith("GLOSSARION_SELFTEST ")]
    assert len(marker) == 1 and marker[0].startswith("GLOSSARION_SELFTEST PASS {")
    summary = json.loads(marker[0].split(" ", 2)[2])
    assert summary["passed"] == result["passed"] and summary["failed"] == 0
    report = json.loads((paths.logs / "selftest-last.json").read_text(encoding="utf-8"))
    assert report["suite"] == "smoke"


def test_selftest_unknown_suite_fails(storage, capsys):
    from glossarion_mobile.diagnostics import selftest

    rb.bootstrap(app_dir=APP_DIR, force=True)
    capsys.readouterr()
    result = selftest.run_selftest("nope")
    assert result["ok"] is False and "unknown suite" in result["error"]
    assert "GLOSSARION_SELFTEST FAIL " in capsys.readouterr().err


def test_selftest_without_bootstrap_fails_env_checks(monkeypatch):
    from glossarion_mobile.diagnostics import selftest

    rb.reset(restore_env=True)
    result = selftest.run_selftest("smoke", strict=False, emit=False, write_report=False, only={"env_contract"})
    assert result["failed"] == 1 and "bootstrap" in result["checks"][0]["error"]


def test_selftest_marker_is_short():
    from glossarion_mobile.diagnostics import selftest

    checks = [{"name": f"check_{i}", "status": "fail", "error": "E" * 500} for i in range(30)]
    line = selftest.summary_line(
        {"suite": "smoke", "ok": False, "passed": 0, "failed": 30, "skipped": 0, "secs": 1.0, "checks": checks}
    )
    assert line.startswith("GLOSSARION_SELFTEST FAIL {")
    assert len(line) <= rb.MARKER_MAX_CHARS + 40
    json.loads(line.split(" ", 2)[2])


# --------------------------------------------------------------------------
# Warm import markers
# --------------------------------------------------------------------------


def test_warm_import_markers(storage, tmp_path, monkeypatch, capsys):
    backend = tmp_path / "backend_fake"
    backend.mkdir()
    (backend / "glfake_alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    (backend / "glfake_beta.py").write_text("import glfake_alpha\n", encoding="utf-8")
    (backend / "glfake_broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.setenv("GLOSSARION_BACKEND_DIR", str(backend))
    try:
        paths = rb.bootstrap(app_dir=APP_DIR, force=True)
        assert paths.backend_source == "env"
        capsys.readouterr()

        done = threading.Event()
        results = []
        thread = rb.start_warm_import(("glfake_alpha", "glfake_beta"), on_done=lambda r: (results.append(r), done.set()))
        assert done.wait(30)
        thread.join(5)
        assert results[0]["ok"] and results[0]["modules"] == 2
        assert rb.backend_ready(1) == results[0]
        err = capsys.readouterr().err
        assert any(ln.startswith("GLOSSARION_BACKEND_READY modules=2 secs=") for ln in err.splitlines())

        bad = rb.warm_import(("glfake_broken",))
        assert not bad["ok"] and "RuntimeError" in bad["failed"]["glfake_broken"]
        err = capsys.readouterr().err
        assert "GLOSSARION_BACKEND_FAIL {" in err and "GLOSSARION_BACKEND_READY" not in err
    finally:
        for name in ("glfake_alpha", "glfake_beta", "glfake_broken"):
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------
# App page (offline, fake Flet client)
# --------------------------------------------------------------------------


def _load_main_module():
    spec = importlib.util.spec_from_file_location("glossarion_mobile_app_main", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spike_main():
    """The U0 device-checks screen run standalone (since U1 main.py's ``main`` is the chat
    shell; tests_host/test_ui_foundations.py covers that and the embedded device checks)."""
    from glossarion_mobile.ui import spike

    return spike.main


def _fake_session(platform: str):
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.controls.context import _context_page
    from flet.messaging.connection import Connection
    from flet.messaging.protocol import MessageAction, configure_encode_object_for_msgpack
    from flet.messaging.session import Session
    from flet.pubsub.pubsub_hub import PubSubHub

    encode = configure_encode_object_for_msgpack(BaseControl)

    class FakeConnection(Connection):
        """In-memory client: encodes every message like the socket transport
        (which also maintains Flet's diff snapshots) and answers every
        invoke_method with ``None``."""

        def __init__(self, loop):
            super().__init__()
            from concurrent.futures import ThreadPoolExecutor

            self.loop = loop
            self.executor = ThreadPoolExecutor(2)
            self.pubsubhub = PubSubHub(loop=loop, executor=self.executor)
            self.page_url = "test://fake"
            self.messages = []
            self.bytes_sent = 0
            self.session = None

        def send_message(self, message):
            self.messages.append(message)
            self.bytes_sent += len(msgpack.packb([message.action, message.body], default=encode))
            if message.action == MessageAction.INVOKE_METHOD and self.session is not None:
                body = message.body
                self.loop.call_soon_threadsafe(
                    self.session.handle_invoke_method_results, body.control_id, body.call_id, None, None
                )

        def invoked(self):
            return [m.body.name for m in self.messages if m.action == MessageAction.INVOKE_METHOD]

    conn = FakeConnection(asyncio.get_running_loop())
    session = Session(conn)
    conn.session = session
    session.apply_page_patch({"route": "/", "platform": platform, "width": 412, "height": 860})
    msgpack.packb(session.get_page_patch(), default=encode)  # what REGISTER_CLIENT sends
    _context_page.set(session.page)
    return conn, session


@pytest.fixture
def app_env(storage, monkeypatch):
    pytest.importorskip("flet")
    pytest.importorskip("msgpack")
    rb.bootstrap(app_dir=APP_DIR, force=True)
    calls = []

    def fake_warm(modules=rb.WARM_IMPORT_MODULES, *, on_done=None):
        calls.append(modules)
        if on_done is not None:
            on_done({"ok": True, "modules": 0, "secs": 0.0, "failed": {}, "requested": list(modules)})

    monkeypatch.setattr(rb, "start_warm_import", fake_warm)
    from glossarion_mobile.diagnostics import selftest

    def fake_selftest(suite="smoke", **kwargs):
        return {"suite": suite, "ok": True, "passed": 1, "failed": 0, "skipped": 0, "secs": 0.0,
                "strict": False, "checks": [{"name": "fake", "status": "pass", "secs": 0.0}]}

    monkeypatch.setattr(selftest, "run_selftest", fake_selftest)
    # Hermetic token stores for the app's sign-in status (U3 ChatFeature: ChatGPT; U4 Accounts: every
    # provider slot): the auth modules resolve their token folder once at import, so a module imported
    # earlier in this process (before any bootstrap, or by a previous test) would still point at the
    # developer's real ~/.glossarion (on Windows expanduser ignores the HOME override).
    token_dir = os.path.join(str(storage["data"]), "home", ".glossarion")
    for name in ("authgpt_auth", "authgem_auth", "authgrok_auth", "authcd_auth"):
        try:
            module = __import__(name)
        except Exception:  # pragma: no cover - backend not importable
            continue
        token_file = os.path.join(token_dir, name.replace("_auth", "_tokens.json"))
        monkeypatch.setattr(module, "_DEFAULT_TOKEN_DIR", token_dir)
        if hasattr(module, "_DEFAULT_TOKEN_FILE"):
            monkeypatch.setattr(module, "_DEFAULT_TOKEN_FILE", token_file)
        if hasattr(module, "_default_store"):
            monkeypatch.setattr(module, "_default_store", None)
        if hasattr(module, "_account_stores"):
            monkeypatch.setattr(module, "_account_stores", {})
        if hasattr(module, "_CLIENT_VERSION_FILE"):  # authcd: adopted Claude Code client version
            monkeypatch.setattr(module, "_CLIENT_VERSION_FILE", os.path.join(token_dir, "authcd_client_version.json"))
        if hasattr(module, "_OFFICIAL_GROK_AUTH_FILE"):  # authgrok: the Grok CLI login (slot 0 import)
            monkeypatch.setattr(module, "_OFFICIAL_GROK_AUTH_FILE",
                                os.path.join(str(storage["data"]), "home", ".grok", "auth.json"))
        monkeypatch.setenv(name.replace("_auth", "_TOKEN_FILE").upper(), token_file)
    # A returning user: the first-run Welcome flow (U3) is done, so it does not cover the chat home.
    # (test_ui_foundations._start(first_run=True) removes this for the first-run tests.)
    (storage["data"] / "mobile_state.json").write_text(json.dumps({"welcome_completed": True}), encoding="utf-8")
    return calls


async def _route(session, route: str) -> None:
    """What the Flutter client does on a route change: patch page.route, then fire the event."""
    session.apply_page_patch({"route": route})
    await session.dispatch_event(session.page._i, "route_change", {"route": route})


async def _settle(app, seconds: float = 0.3):
    await asyncio.sleep(seconds)


async def _shutdown(app):
    app._fgs_stop.set()
    app._cp_stop.set()
    app._close_oauth()
    for task in list(app._tasks):
        task.cancel()
    await asyncio.sleep(0)


def test_app_page_builds_offline(app_env, capsys):
    import flet as ft

    main_module = _load_main_module()
    assert main_module.PATHS is rb.get_paths()  # main.py reused the existing bootstrap
    spike_main = _spike_main()

    async def scenario():
        conn, session = _fake_session("windows")
        page = session.page
        await spike_main(page)
        await session.after_event(page)
        app = page.data
        try:
            assert type(app).__name__ == "SpikeApp"
            assert app.native.is_stub  # desktop never constructs the extension service
            assert page.theme.color_scheme_seed == "#E18F98"
            assert page.theme.visual_density == ft.VisualDensity.COMPACT
            assert set(app.cards) >= {"selftest", "secure", "fgs", "notify", "share", "oauth", "stack", "background"}
            assert page.views[0].controls, "page has no content"
            assert conn.bytes_sent > 0
            assert app_env, "warm import was not started"

            # Open-with paths and foreign links are ignored and recorded.
            await _route(session, "/document/raw%3A%2Fx.epub")
            await _route(session, "/oauth/return?p=spike&nonce=nope")
            # The self-test deep link runs the suite on a worker thread, then resets the route.
            await _route(session, "/__selftest__?suite=smoke")
            await _settle(app, 0.5)
            accepted = [(r.raw, r.accepted) for r in app.router.history]
            assert ("/document/raw%3A%2Fx.epub", False) in accepted
            assert ("/__selftest__?suite=smoke", True) in accepted
            assert app.cards["selftest"].status == "pass"
            assert app.cards["oauth"].status == "info"  # return without a pending test
            assert "push_route" in conn.invoked()

            await app.stack_test()
            assert app.cards["stack"].status == "pass", app.cards["stack"].result.value
            await app.bg_begin()
            assert app.cards["background"].status == "n/a"
        finally:
            await _shutdown(app)

    asyncio.run(scenario())
    err = capsys.readouterr().err
    assert "GLOSSARION_READY" in err.splitlines()


def test_app_oauth_loopback_flow_offline(app_env):
    import urllib.request

    spike_main = _spike_main()

    async def scenario():
        conn, session = _fake_session("windows")
        page = session.page
        await spike_main(page)
        await session.after_event(page)
        app = page.data
        try:
            opened = []
            rb.set_url_opener(opened.append)  # stand-in for UrlLauncher
            await app.oauth_test()
            assert opened and opened[0].startswith("http://127.0.0.1:")
            # the "browser" loads the loopback page...
            body = await asyncio.to_thread(lambda: urllib.request.urlopen(opened[0], timeout=10).read().decode())
            assert "glossarion://app/oauth/return?p=spike&amp;nonce=" in body
            await _settle(app, 0.2)
            nonce = app._oauth["nonce"]
            # ...which sends the app the return deep link (Flet hands Python the path form).
            await _route(session, f"/oauth/return?p=spike&nonce={nonce}")
            await _settle(app, 0.2)
            assert app.cards["oauth"].status == "pass", app.cards["oauth"].result.value
            assert app._oauth is None  # server closed
        finally:
            await _shutdown(app)

    asyncio.run(scenario())


def test_app_with_native_extension_on_android(app_env, monkeypatch):
    if not (EXTENSION_SRC / "flet_glossarion_native" / "__init__.py").is_file():
        pytest.skip("flet_glossarion_native extension sources not present")
    monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    spike_main = _spike_main()

    async def scenario():
        conn, session = _fake_session("android")
        page = session.page
        await spike_main(page)
        await session.after_event(page)
        app = page.data
        try:
            native = app.native.native
            assert type(native).__name__ == "GlossarionNative" and not app.native.is_stub
            for event in ("share", "foreground", "background_task", "notification"):
                assert getattr(native, f"on_{event}") is not None
            await _settle(app, 0.2)
            assert {"get_platform_info", "get_initial_shared"} <= set(conn.invoked())

            # FGS ticker; the notification Stop button must reach the Python thread.
            await app.fgs_start()
            await _settle(app, 1.5)
            await session.dispatch_event(native._i, "foreground", {"type": "button", "button_id": "stop"})
            for _ in range(50):
                if app.cards["fgs"].status != "running":
                    break
                await asyncio.sleep(0.1)
            assert app.cards["fgs"].status == "pass", app.cards["fgs"].result.value
            assert "notification Stop button" in app.cards["fgs"].result.value
            assert "stop_job_service" in conn.invoked()

            # Open-with arrives as a share event and must not touch page.route.
            await session.dispatch_event(
                native._i, "share", {"items": [{"id": "1", "kind": "file", "path": "/x/book.epub", "name": "book.epub"}]}
            )
            await asyncio.sleep(2.8)
            assert app.cards["share"].status == "pass", app.cards["share"].result.value
            assert page.route == "/"
        finally:
            await _shutdown(app)

    asyncio.run(scenario())
