"""Host tests for tools/host_smoke.py (the read-only bundle import smoke).

Run from src/mobile:  python -m pytest -p no:cacheprovider tests_host/test_host_smoke.py -q

host_smoke patches process-global state (subprocess, os, sockets, sys.meta_path,
the environment), so every scenario runs it as a subprocess against a small fake
bundle whose modules each trigger one guard. The real collected bundle is smoked
by CI directly (build-mobile.yml prepare, python-app.yml mobile-backend-check).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = MOBILE_DIR / "tools"
SRC_DIR = MOBILE_DIR.parent
HOST_SMOKE = TOOLS_DIR / "host_smoke.py"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import host_smoke  # noqa: E402

BUNDLE_INFO = 'BUILD_VERSION = "0.0.0"\nGIT_SHA = "hostsmoketest"\nBUNDLE_SHA256 = "' + "0" * 64 + '"\nMODULE_COUNT = 0\n'

VENDOR_LIB = """
import subprocess
import sys


def probe_handled():
    try:
        subprocess.run([sys.executable, "-c", "pass"], check=False)
    except OSError:
        return "fallback"
    return "spawned"


def probe_unhandled():
    subprocess.run([sys.executable, "-c", "pass"], check=False)
"""

CLEAN_MODULES = {
    "env_contract_probe": """
        import importlib.util
        import os
        import sys

        import mobile_runtime

        assert os.environ["GLOSSARION_MOBILE"] == "1"
        assert os.environ["GLOSSARION_NO_PROCESSES"] == "1"
        assert mobile_runtime.is_mobile() and not mobile_runtime.processes_available()
        assert not mobile_runtime.subprocesses_available()
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9" == os.environ["HTTP_PROXY"]
        assert "NO_PROXY" not in os.environ
        data = os.path.normcase(os.environ["GLOSSARION_DATA_DIR"])
        assert os.path.normcase(os.getcwd()) == data
        assert os.path.normcase(os.environ["CONFIG_FILE"]) == os.path.normcase(os.path.join(data, "config.json"))
        assert os.path.normcase(os.path.expanduser("~")) == os.path.normcase(os.environ["HOME"])
        assert os.path.normcase(os.environ["HOME"]).startswith(data)
        assert sys.dont_write_bytecode
        assert os.path.basename(sys.argv[0]) == "main.py"
    """,
    "pool_thread_mode": """
        import os

        import mobile_runtime

        def _process_only_initializer():
            raise RuntimeError("a process-only initializer must not run in thread mode")

        with mobile_runtime.make_pool_executor(2, initializer=_process_only_initializer) as executor:
            assert type(executor).__name__ == "ThreadPoolExecutor"
            assert executor.submit(os.getpid).result() == os.getpid()
    """,
    "guarded_blocked": """
        import importlib.util

        try:
            import jellyfish  # noqa: F401
            raise AssertionError("jellyfish must be blocked")
        except ImportError:
            pass
        try:
            import PySide6.QtCore  # noqa: F401
            raise AssertionError("PySide6 must be blocked")
        except ImportError:
            pass
        assert importlib.util.find_spec("torch") is None
        assert importlib.util.find_spec("tkinter") is None
        assert importlib.util.find_spec("json") is not None
    """,
    "wants_psutil": """
        try:
            import psutil  # noqa: F401
        except ImportError:
            psutil = None
    """,
    "replaces_stdout": """
        import io
        import sys

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print("worker-script style stdout re-wrap")
    """,
    "indirect_handled": """
        import os
        import sys

        sys.path.append(os.environ["HOST_SMOKE_TEST_VENDOR"])
        import fakevendor

        assert fakevendor.probe_handled() == "fallback"
    """,
}

VIOLATION_MODULES = {
    "spawn_direct": """
        import subprocess
        import sys

        subprocess.run([sys.executable, "-c", "pass"], check=False)
    """,
    "spawn_handled": """
        import os

        try:
            os.system("echo host-smoke")
        except OSError:
            HANDLED = True
    """,
    "pool_direct": """
        from concurrent.futures import ProcessPoolExecutor

        try:
            ProcessPoolExecutor(max_workers=2)
        except OSError:
            pass
    """,
    "mp_queue": """
        import multiprocessing

        try:
            multiprocessing.Queue()
        except (OSError, ImportError):
            pass
    """,
    "network_probe": """
        import socket

        try:
            socket.create_connection(("203.0.113.7", 80), timeout=0.5)
        except OSError:
            pass
    """,
    "writes_next_to_file": """
        import os

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json"), "w") as fh:
            fh.write("{}")
    """,
    "indirect_unhandled": """
        import os
        import sys

        sys.path.append(os.environ["HOST_SMOKE_TEST_VENDOR"])
        import fakevendor

        fakevendor.probe_unhandled()
    """,
    "blocked_unguarded": """
        import weasyprint  # noqa: F401
    """,
}

PYC_MODULES = {
    "pyc_layout": """
        import os

        assert __file__.endswith(".pyc"), __file__
        assert not os.path.exists(__file__[:-1]), "the .py source must not ship with --pyc"
    """,
    "wants_psutil": CLEAN_MODULES["wants_psutil"],
}


def _make_bundle(root: Path, modules: dict) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "_bundle_info.py").write_text(BUNDLE_INFO, encoding="utf-8")
    (bundle / "mobile_runtime.py").write_bytes((SRC_DIR / "mobile_runtime.py").read_bytes())
    for name, source in modules.items():
        (bundle / f"{name}.py").write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return bundle


def _vendor(root: Path) -> Path:
    site = root / "vendor" / "site-packages"
    site.mkdir(parents=True)
    (site / "fakevendor.py").write_text(VENDOR_LIB.lstrip(), encoding="utf-8")
    return site


def _run(tmp_path: Path, bundle: Path, *extra: str, env_extra: dict | None = None):
    report_path = tmp_path / "report.json"
    assets = tmp_path / "no-assets"
    assets.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop("GITHUB_STEP_SUMMARY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra or {})
    cmd = [
        sys.executable, str(HOST_SMOKE),
        "--bundle", str(bundle), "--assets", str(assets), "--checks", "none",
        "--work-dir", str(tmp_path / "work"), *extra,
    ]
    if "--json" not in extra:
        cmd += ["--json", str(report_path)]
    proc = subprocess.run(
        cmd, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        timeout=300,
    )
    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return proc, report


def _events_for(report: dict, module: str) -> list:
    return [
        e for e in report["tripwire"]
        if module in json.dumps([e.get("caller"), e.get("bundle_frame"), e.get("phase")])
    ]


# --------------------------------------------------------------------------
# End to end (subprocess)
# --------------------------------------------------------------------------


def test_clean_bundle_passes_with_contract_blocker_and_warnings(tmp_path):
    bundle = _make_bundle(tmp_path, CLEAN_MODULES)
    summary = tmp_path / "step_summary.md"
    proc, report = _run(
        tmp_path, bundle, "--simulate", "android",
        env_extra={"HOST_SMOKE_TEST_VENDOR": str(_vendor(tmp_path)), "GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert report is not None, proc.stdout + proc.stderr
    assert proc.returncode == 0, json.dumps(report["failures"], indent=1) + proc.stdout[-3000:]
    assert report["ok"] is True
    assert report["simulate"] == "android"
    assert report["bootstrap"]["backend_source"] == "bundle"
    assert report["bootstrap"]["platform"] == "android"
    imports = report["imports"]
    assert imports["failed"] == {} and imports["outside_bundle"] == {}
    assert imports["ok"] == imports["total"] == len(CLEAN_MODULES) + 2  # + mobile_runtime, _bundle_info
    assert report["network"] == [] and report["stray_writes"] == []
    blocked = report["blocked_imports"]
    assert {"jellyfish", "PySide6", "torch", "tkinter"} <= set(blocked)
    assert blocked["jellyfish"]["first_importer"].startswith("<bundle>/guarded_blocked.py:")
    assert "psutil" not in blocked  # psutil ships on Android
    # A spawn inside a third-party helper that handles the OSError itself is a warning.
    assert len(report["tripwire"]) == 1
    event = report["tripwire"][0]
    assert event["api"] == "subprocess.Popen" and event["direct"] is False and event["propagated"] is False
    assert event["caller"]["where"] == "third_party"
    assert event["bundle_frame"]["file"] == "<bundle>/indirect_handled.py"
    assert any("handled by the caller" in w for w in report["warnings"])
    assert any("replaces_stdout replaced sys.stdout" in w for w in report["warnings"])
    assert "host_smoke (android" in proc.stdout and ": PASS" in proc.stdout
    assert "## Host smoke (android" in summary.read_text(encoding="utf-8")
    assert not Path(report["root"]).exists(), "the temp root must be removed"


def test_violations_are_reported_precisely(tmp_path):
    bundle = _make_bundle(tmp_path, VIOLATION_MODULES)
    proc, report = _run(
        tmp_path, bundle, "--simulate", "android",
        env_extra={"HOST_SMOKE_TEST_VENDOR": str(_vendor(tmp_path))},
    )
    assert report is not None, proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert report["ok"] is False
    failed = report["imports"]["failed"]

    # Unguarded direct spawn: the import fails with the tripwire's ENOTSUP OSError.
    assert "spawn_direct" in failed and "host_smoke tripwire" in failed["spawn_direct"]["error"]
    (direct,) = _events_for(report, "spawn_direct")
    assert direct["api"] == "subprocess.Popen" and direct["direct"] is True and direct["propagated"] is True
    assert direct["caller"]["file"] == "<bundle>/spawn_direct.py" and direct["caller"]["line"] == 4
    assert direct["phase"] == "import:spawn_direct"
    assert any(line.startswith("<bundle>/spawn_direct.py:4") for line in direct["stack"])

    # A direct spawn fails the smoke even when the module handles the OSError.
    (handled,) = _events_for(report, "spawn_handled")
    assert handled["api"] == "os.system" and handled["direct"] is True and handled["propagated"] is False
    assert "spawn_handled" not in failed

    (pool,) = _events_for(report, "pool_direct")
    assert pool["api"] == "concurrent.futures.ProcessPoolExecutor" and pool["direct"] is True

    queue_events = _events_for(report, "mp_queue")
    assert queue_events and "SemLock" in queue_events[0]["api"] and queue_events[0]["direct"] is True

    # An indirect spawn that propagates out of the import fails it.
    assert "indirect_unhandled" in failed
    (indirect,) = _events_for(report, "indirect_unhandled")
    assert indirect["direct"] is False and indirect["propagated"] is True
    assert indirect["caller"]["where"] == "third_party"

    (network,) = report["network"]
    assert network["target"] == "203.0.113.7:80" and network["phase"] == "import:network_probe"

    # Unguarded import of a package the phone does not have.
    assert "blocked_unguarded" in failed and "weasyprint" in failed["blocked_unguarded"]["error"]

    # Writing next to __file__: refused (POSIX read-only dir) or caught as a stray write (Windows).
    stray = [w["path"] for w in report["stray_writes"]]
    assert "writes_next_to_file" in failed or any(p.endswith("state.json") for p in stray)

    failures = "\n".join(report["failures"])
    for needle in ("import spawn_direct", "tripwire #", "network #1", "import blocked_unguarded"):
        assert needle in failures
    assert "FAIL" in proc.stdout and "spawn_direct.py:4" in proc.stdout


def test_ios_simulation_blocks_psutil_and_ships_pyc_only(tmp_path):
    bundle = _make_bundle(tmp_path, PYC_MODULES)
    proc, _ = _run(tmp_path, bundle, "--simulate", "ios", "--pyc", "--json", "-")
    report = json.loads(proc.stdout)
    assert proc.returncode == 0, json.dumps(report["failures"], indent=1) + proc.stderr[-3000:]
    assert report["simulate"] == "ios" and report["pyc"] is True
    assert report["bundle"]["pyc_files"] == len(PYC_MODULES) + 2
    assert report["imports"]["ok"] == report["imports"]["total"]
    assert "psutil" in report["blocked_imports"]
    assert "psutil" in report["blocked_packages"]
    assert "host_smoke (ios" in proc.stderr  # the human summary moves to stderr with --json -


def test_setup_errors_exit_2(tmp_path):
    proc, report = _run(tmp_path, tmp_path / "missing-bundle")
    assert proc.returncode == 2
    assert any("no collected bundle" in f for f in report["failures"])

    bundle = _make_bundle(tmp_path / "ok", {"plain": "VALUE = 1\n"})
    proc, report = _run(tmp_path, bundle, "--checks", "no_such_check")
    assert proc.returncode == 2
    assert any("unknown checks" in f for f in report["failures"])


# --------------------------------------------------------------------------
# Pure helpers (in-process; nothing global is patched)
# --------------------------------------------------------------------------


def test_blocked_packages_merge_spec_list_and_manifest():
    android = host_smoke.blocked_packages("android")
    ios = host_smoke.blocked_packages("ios")
    spec = {"PySide6", "shiboken6", "tkinter", "weasyprint", "xhtml2pdf", "torch", "transformers", "jellyfish",
            "cohere", "mistralai", "vertexai", "sklearn"}
    assert spec <= set(android)
    assert "psutil" not in android and "psutil" in ios
    # From backend_manifest.toml: [gui].packages and [thirdparty.unavailable] via [thirdparty.map].
    assert {"PyQt5", "gradio", "huggingface_hub", "Crypto"} <= set(android)
    # Packages the phone does ship are never blocked (grpcio / google-cloud-vision: Tier B, U9).
    for shipped in ("onnxruntime", "fitz", "lxml", "tiktoken", "cryptography", "rapidfuzz", "google.genai",
                    "grpc", "google.cloud.vision"):
        assert shipped not in android


def test_tripwire_ids_follow_exception_chains():
    marked = OSError(95, "tripwire")
    marked._host_smoke_event = 7
    try:
        try:
            raise marked
        except OSError as exc:
            raise RuntimeError("wrapped") from exc
    except RuntimeError as outer:
        assert host_smoke._tripwire_ids(outer) == {7}
    assert host_smoke._tripwire_ids(ValueError("unrelated")) == set()
    assert host_smoke._tripwire_ids(None) == set()


def test_snapshot_diff_ignores_writable_dirs(tmp_path):
    (tmp_path / "ro").mkdir()
    (tmp_path / "ro" / "kept.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ro" / "gone.txt").write_text("b", encoding="utf-8")
    (tmp_path / "rw").mkdir()
    before = host_smoke._snapshot(tmp_path)
    (tmp_path / "ro" / "new.txt").write_text("c", encoding="utf-8")
    (tmp_path / "ro" / "gone.txt").unlink()
    (tmp_path / "rw" / "allowed.txt").write_text("d", encoding="utf-8")
    after = host_smoke._snapshot(tmp_path)
    changes = host_smoke._diff_snapshots(tmp_path, before, after, [host_smoke._norm(tmp_path / "rw")])
    assert sorted((Path(c["path"]).name, c["change"]) for c in changes) == [("gone.txt", "deleted"), ("new.txt", "created")]
    created_only = host_smoke._diff_snapshots(tmp_path, before, after, [], created_only=True)
    assert {Path(c["path"]).name for c in created_only} == {"new.txt", "allowed.txt"}


def test_summary_and_markdown_render_failures():
    report = {
        "ok": False, "simulate": "ios", "python": "3.13.0", "host_platform": "linux",
        "imports": {"ok": 1, "total": 2, "secs": 0.1, "failed": {"x": {"error": "OSError: boom"}}},
        "checks": [{"name": "jaro_winkler", "status": "fail", "error": "AssertionError: P26"}],
        "tripwire": [{"id": 1, "api": "os.system", "phase": "import:x", "direct": True, "propagated": False,
                      "thread": "MainThread", "caller": {"file": "<bundle>/x.py", "line": 3, "function": "<module>"},
                      "bundle_frame": {"file": "<bundle>/x.py", "line": 3, "function": "<module>"},
                      "stack": ["<bundle>/x.py:3 in <module>"]}],
        "network": [], "stray_writes": [], "failures": ["import x: OSError: boom"], "warnings": [],
    }
    text = host_smoke.format_summary(report)
    assert "FAIL" in text and "<bundle>/x.py:3" in text and "jaro_winkler" in text
    markdown = host_smoke.format_markdown(report)
    assert markdown.startswith("## Host smoke (ios") and "1 event(s), 1 direct" in markdown


@pytest.mark.parametrize("host,local", [("127.0.0.1", True), ("localhost", True), ("::1", True),
                                        ("203.0.113.7", False), ("example.com", False), (None, True)])
def test_loopback_classification(host, local):
    assert host_smoke.Tripwires._is_local_host(host) is local
