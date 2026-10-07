#!/usr/bin/env python3
"""Host smoke test of the collected mobile backend, run the way the phone runs it.

CI runs this on Python 3.13 (the interpreter Flet embeds) in ``build-mobile.yml``
(prepare: ``--simulate android`` and ``--simulate ios``) and in ``python-app.yml``
(mobile-backend-check). Usage, from ``src/mobile``::

    python tools/host_smoke.py --simulate android              # bundle: app/backend
    python tools/host_smoke.py --simulate ios --bundle app/backend --json build/host_smoke_ios.json
    python tools/host_smoke.py --collect --simulate android    # collect a fresh bundle first
    python tools/host_smoke.py --checks none                   # imports only

What it does (plan section 9, build-ci design section 7.4):

1. **Device layout.** A temp root gets a *read-only* app dir (``app/main.py``,
   ``app/glossarion_mobile``, ``app/assets/{tiktoken,selftest}``, ``app/backend`` =
   copy of the collected bundle; ``--pyc`` ships the backend as legacy ``.pyc`` only,
   like ``flet build``) and separate writable ``FLET_APP_STORAGE_DATA/CACHE/TEMP``
   dirs. ``runtime_bootstrap.bootstrap()`` then applies the real env contract
   (``GLOSSARION_MOBILE=1``, ``GLOSSARION_NO_PROCESSES=1``, ``HOME``, ``TMPDIR``,
   ``GLOSSARION_DATA_DIR``, ``CONFIG_FILE``, ``TIKTOKEN_CACHE_DIR`` seeded from the
   assets, ...), chdirs into the data dir and resolves the backend from
   ``app/backend`` exactly as on a device (``backend_source == "bundle"``).
2. **Guards**, installed before anything is imported:

   * a meta-path blocker for packages that do not exist on the phone (PySide6,
     shiboken6, tkinter, weasyprint, xhtml2pdf, torch, transformers, jellyfish, cohere,
     mistralai, vertexai, sklearn, plus ``[gui].packages`` and
     ``[thirdparty.unavailable]`` from ``backend_manifest.toml``; psutil too for
     ``--simulate ios``). ``importlib.util.find_spec`` reports them as missing;
   * a **process tripwire**: ``ProcessPoolExecutor.__init__``, multiprocessing
     ``Process.start``/``Pool``/``ThreadPool``/``SemLock`` (no ``sem_open`` on
     Android), ``subprocess.Popen.__init__``, ``os.fork``, ``os.posix_spawn``,
     ``os.system`` and the other ``os`` spawn/exec functions raise
     ``OSError(ENOTSUP)`` (as on iOS) and record the stack;
   * offline networking: ``HTTP(S)_PROXY=http://127.0.0.1:9`` and a socket guard that
     refuses non-loopback ``connect``/``getaddrinfo``.

3. **Key setters** (device order, before the warm import):
   ``api_key_encryption.set_key_material`` and ``token_encryption.set_symmetric_key``
   get fresh random keys through the app's own installer
   (``glossarion_mobile.services.secure_keys.install_backend_keys``, which
   ``GlossarionApp.start`` feeds from SecureStorage).
4. **Imports every bundled module** (warm-import modules first) and fails on any
   exception, on a module loaded from outside the bundle, and reports modules that
   replace ``sys.stdout``/``sys.stderr``.
5. **Functional checks**: the device self-test suite
   (``glossarion_mobile.diagnostics.selftest``: env contract, writable dirs, offline
   tiktoken for both encodings, ebooklib/lxml on the self-test EPUB, Fernet, the
   installed encryption keys, openai/pydantic/jiter, PyMuPDF, cv2, onnxruntime,
   16 MiB thread stacks, and ``library_reader``: a partly translated workspace of the
   self-test EPUB opened through the Library scan, Book page, Chapters tab and Reader
   page builder over the bundled ``library_core`` / ``progress_core`` / ``reader_doc``, and
   ``glossary_qa``: a token-CSV glossary parsed, edited, saved and re-parsed through
   ``glossary_document.GlossaryDocument`` plus a QA quick scan through
   ``qa_scan_runtime.run_qa_scan_path``, which must run on threads)
   plus host checks:
   ``key_material`` (Fernet round trip through ``api_key_encryption`` with the injected
   key, no key file written), ``chapter_extractor_pool`` (``extract_chapters`` on the
   13-file self-test EPUB, which takes the worker-pool path, in thread mode),
   ``jaro_winkler`` (the RapidFuzz Jaro-Winkler path with jellyfish blocked),
   ``pdf_mupdf_html`` (the WeasyPrint-subset shim; skipped until it is bundled) and
   ``headless_owner_env`` (the app's Env preview path: a ``HeadlessOwner`` built from a
   fresh-install config and ``run_env.build_translation_env``; the desktop defaults
   ``authgpt/gpt-6-luna`` / ``AUTO_GLOSSARY_MODE=off`` must come out, and the process
   env, cwd and config.json must be left as they were) and ``manga_pipeline`` (U8: RT-DETR
   through the bundled ``bubble_detector``'s Python onnxruntime path on a tiny synthetic
   export planted in ``BUBBLE_CACHE_DIR``, then one fixture page through
   ``manga_runner.HeadlessMangaRunner`` with custom-api OCR and translation answered by the
   loopback fake LLM server, inpainting skipped; env, cwd and config.json unchanged).
6. **Verdict.** Fails (exit 1) on an import or check failure, on any tripwire event
   raised directly by bundled code (or one that propagated out of a phase), on any
   network attempt, and on any file created, changed or deleted outside the writable
   dirs. Spawns attempted by stdlib/third-party helpers that handled the OSError
   themselves are reported as warnings. ``--json`` writes the full report (every
   event with its stack); ``$GITHUB_STEP_SUMMARY`` gets a Markdown summary.

Exit codes: 0 pass, 1 smoke failure, 2 usage/setup error. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import errno
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
import platform as _platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
MOBILE_DIR = TOOLS_DIR.parent
APP_SRC_DIR = MOBILE_DIR / "app"
DEFAULT_BUNDLE = APP_SRC_DIR / "backend"
DEFAULT_ASSETS = APP_SRC_DIR / "assets"
MANIFEST_PATH = MOBILE_DIR / "backend_manifest.toml"
COLLECTOR = TOOLS_DIR / "collect_backend.py"

SIMULATED_PLATFORMS = ("android", "ios")
OFFLINE_PROXY = "http://127.0.0.1:9"
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
ENOTSUP = getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", 95))
ENETUNREACH = getattr(errno, "ENETUNREACH", 101)
ASSET_GROUPS = ("tiktoken", "selftest")

# Packages that never exist on the phone (build-ci design section 7.4). The manifest's
# [gui].packages and [thirdparty.unavailable] are merged in at runtime.
BLOCKED_PACKAGES: Dict[str, str] = {
    "PySide6": "Qt desktop GUI",
    "shiboken6": "Qt desktop GUI",
    "tkinter": "Tk desktop GUI",
    "_tkinter": "Tk desktop GUI",
    "weasyprint": "pango; pdf_mupdf_html shim on mobile",
    "xhtml2pdf": "python-bidi has no mobile wheel",
    "torch": "no Android/iOS wheel",
    "transformers": "torch stack",
    "jellyfish": "no wheel; RapidFuzz Jaro-Winkler fallback",
    "cohere": "fastavro has no wheel; HTTP fallback",
    "mistralai": "httpx2 + opentelemetry; HTTP fallback",
    "vertexai": "gRPC SDK; Vertex via REST",
    "sklearn": "scikit-learn (sentence-transformers only)",
}
BLOCKED_IOS_ONLY: Dict[str, str] = {"psutil": "no iOS wheel (Android-only dependency)"}

# Stdlib modules that implement a spawn API; frames inside them are skipped when
# deciding who asked for the process.
_SPAWN_IMPL = frozenset({"subprocess.py", "os.py", "concurrent", "multiprocessing", "asyncio", "pty.py"})
_HARNESS_FILE = os.path.normcase(os.path.abspath(__file__))

# Selftest checks the CLI accepts besides the host checks below.
HOST_CHECKS = ("key_material", "chapter_extractor_pool", "jaro_winkler", "pdf_mupdf_html", "headless_owner_env",
               "manga_pipeline")
# What a desktop fresh install runs with (U0 oracle; HeadlessOwner replays the desktop startup).
FRESH_INSTALL_ENV = {"MODEL": "authgpt/gpt-6-luna", "AUTO_GLOSSARY_MODE": "off"}
# A tiny RT-DETR-shaped ONNX export (inputs images [1,3,640,640] / orig_target_sizes [1,2]; three
# constant detections: a bubble, a text bubble and one below the 0.3 threshold), built with
# onnx.helper as tests/test_mobile_compat_patches_manga.py does (the onnx package is not on the
# phone, so the bytes are embedded; that test pins them to their builder).
SYNTHETIC_RTDETR_ONNX_B64 = (
    "CAg6ggMKNRIGbGFiZWxzIghDb25zdGFudCohCgV2YWx1ZSoVCAEIAxAHOgMAAQJCCGxhYmVsc192oAEECmISBWJveGVz"
    "IghDb25zdGFudCpPCgV2YWx1ZSpDCAEIAwgEEAEiMAAAIEEAAKBBAADcQgAAXEMAAJZDAAAgQgAAyEMAALRCAABIQgAA"
    "+kMAABZDAAAMREIHYm94ZXNfdqABBAo+EgZzY29yZXMiCENvbnN0YW50KioKBXZhbHVlKh4IAQgDEAEiDGZmZj/NzEw/"
    "zczMPUIIc2NvcmVzX3agAQQSC2Zha2VfcnRkZXRyWiIKBmltYWdlcxIYChYIARISCgIIAQoCCAMKAwiABQoDCIAFWiMK"
    "EW9yaWdfdGFyZ2V0X3NpemVzEg4KDAgHEggKAggBCgIIAmIYCgZsYWJlbHMSDgoMCAcSCAoCCAEKAggDYhsKBWJveGVz"
    "EhIKEAgBEgwKAggBCgIIAwoCCARiGAoGc2NvcmVzEg4KDAgBEggKAggBCgIIA0IECgAQDQ=="
)


class SetupError(Exception):
    """The smoke could not be set up (exit code 2)."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_toml(path: Path) -> Optional[dict]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: fall back to the static lists
        return None
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        return None


def _short(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def blocked_packages(simulate: str, manifest_path: Path = MANIFEST_PATH) -> Dict[str, str]:
    """Import names (top-level or dotted) hidden from the backend -> reason."""
    blocked = dict(BLOCKED_PACKAGES)
    data = _load_toml(manifest_path) or {}
    for name in (data.get("gui") or {}).get("packages", []) or []:
        blocked.setdefault(str(name), "desktop GUI toolkit (backend_manifest [gui].packages)")
    thirdparty = data.get("thirdparty") or {}
    reverse: Dict[str, List[str]] = {}
    for import_name, dist in (thirdparty.get("map") or {}).items():
        reverse.setdefault(_pep503(str(dist)), []).append(str(import_name))
    for dist, reason in (thirdparty.get("unavailable") or {}).items():
        for import_name in reverse.get(_pep503(str(dist)), [str(dist).replace("-", "_")]):
            blocked.setdefault(import_name, f"not installed on mobile: {reason}")
    if simulate == "ios":
        blocked.update(BLOCKED_IOS_ONLY)
    return blocked


class _SitePaths:
    """Classifies a code filename: bundle / harness / app / third_party / stdlib / other."""

    def __init__(self, bundle: Path, app: Path) -> None:
        self.bundle = _norm(bundle)
        self.app = _norm(app)
        paths = sysconfig.get_paths()
        self.stdlib = sorted({_norm(paths[k]) for k in ("stdlib", "platstdlib") if paths.get(k)}, key=len, reverse=True)
        site_dirs = {_norm(paths[k]) for k in ("purelib", "platlib") if paths.get(k)}
        try:
            import site

            site_dirs.update(_norm(p) for p in site.getsitepackages())
            user_site = site.getusersitepackages()
            if user_site:
                site_dirs.add(_norm(user_site))
        except Exception:
            pass
        # site.getsitepackages() also lists sys.prefix on Windows, which holds Lib/ (the stdlib).
        site_dirs = {p for p in site_dirs if os.path.basename(p) in ("site-packages", "dist-packages")}
        self.site = sorted(site_dirs, key=len, reverse=True)

    def where(self, filename: str) -> str:
        if not filename or filename.startswith("<"):
            return "frozen"
        path = _norm(filename)
        if _under(path, self.bundle):
            return "bundle"
        if path == _HARNESS_FILE:
            return "harness"
        if _under(path, self.app):
            return "app"
        if any(_under(path, s) for s in self.site) or "site-packages" in path or "dist-packages" in path:
            return "third_party"
        if any(_under(path, s) for s in self.stdlib):
            return "stdlib"
        return "other"

    def display(self, filename: str) -> str:
        if not filename or filename.startswith("<"):
            return filename or "?"
        path = _norm(filename)
        real = os.path.abspath(filename)  # relpath keeps the file's own spelling (normcase lowercases on Windows)
        for root, label in ((self.bundle, "<bundle>"), (self.app, "<app>")):
            if _under(path, root):
                return label + "/" + os.path.relpath(real, root).replace(os.sep, "/")
        for root in self.site:
            if _under(path, root):
                return "<site>/" + os.path.relpath(real, root).replace(os.sep, "/")
        for root in self.stdlib:
            if _under(path, root):
                return "<stdlib>/" + os.path.relpath(real, root).replace(os.sep, "/")
        return filename

    def is_spawn_impl(self, filename: str) -> bool:
        path = _norm(filename)
        for root in self.stdlib:
            if _under(path, root):
                first = os.path.relpath(path, root).split(os.sep, 1)[0]
                return first in _SPAWN_IMPL
        return False


# --------------------------------------------------------------------------
# Phase tracking (sequential; worker threads inherit the current phase)
# --------------------------------------------------------------------------

_PHASE_LOCK = threading.Lock()
_PHASE = "setup"


def _set_phase(name: str) -> None:
    global _PHASE
    with _PHASE_LOCK:
        _PHASE = name


def _phase() -> str:
    with _PHASE_LOCK:
        return _PHASE


# --------------------------------------------------------------------------
# Import blocker
# --------------------------------------------------------------------------


class ImportBlocker(importlib.abc.MetaPathFinder):
    """Makes packages that the phone does not have look uninstalled."""

    def __init__(self, blocked: Dict[str, str], simulate: str, sites: _SitePaths) -> None:
        self.blocked = dict(blocked)
        self.simulate = simulate
        self.sites = sites
        self._by_top: Dict[str, List[str]] = {}
        for key in self.blocked:
            self._by_top.setdefault(key.split(".", 1)[0], []).append(key)
        self.attempts: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._orig_find_spec: Optional[Callable] = None
        self.evicted: List[str] = []

    def match(self, fullname: str) -> Optional[str]:
        for key in self._by_top.get(fullname.split(".", 1)[0], ()):
            if fullname == key or fullname.startswith(key + "."):
                return key
        return None

    def _importer(self) -> Optional[str]:
        frame = sys._getframe(2)
        while frame is not None:
            filename = frame.f_code.co_filename
            where = self.sites.where(filename)
            if where not in ("frozen", "harness") and not filename.replace("\\", "/").endswith(
                ("importlib/__init__.py", "importlib/util.py")
            ):
                return f"{self.sites.display(filename)}:{frame.f_lineno}"
            frame = frame.f_back
        return None

    def record(self, fullname: str, key: str, via: str) -> None:
        importer = self._importer()
        with self._lock:
            entry = self.attempts.setdefault(
                fullname, {"blocked_as": key, "count": 0, "first_importer": importer, "phase": _phase(), "via": via}
            )
            entry["count"] += 1

    def message(self, fullname: str, key: str) -> str:
        return f"No module named {fullname!r} (host_smoke: {key} is not available on {self.simulate}: {self.blocked[key]})"

    # MetaPathFinder API
    def find_spec(self, fullname, path=None, target=None):  # noqa: D401 - importlib API
        key = self.match(fullname)
        if key is None:
            return None
        self.record(fullname, key, "import")
        raise ModuleNotFoundError(self.message(fullname, key), name=fullname)

    def install(self) -> None:
        for name in list(sys.modules):
            if self.match(name) is not None:
                self.evicted.append(name)
                del sys.modules[name]
        sys.meta_path.insert(0, self)
        original = importlib.util.find_spec
        self._orig_find_spec = original
        blocker = self

        def find_spec(name, package=None):
            fullname = importlib.util.resolve_name(name, package) if name.startswith(".") else name
            key = blocker.match(fullname)
            if key is not None:
                blocker.record(fullname, key, "find_spec")
                if fullname == key:
                    return None  # an uninstalled package
                raise ModuleNotFoundError(blocker.message(fullname, key), name=fullname)
            return original(name, package)

        importlib.util.find_spec = find_spec

    def uninstall(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        if self._orig_find_spec is not None:
            importlib.util.find_spec = self._orig_find_spec
            self._orig_find_spec = None


# --------------------------------------------------------------------------
# Process tripwire + network guard
# --------------------------------------------------------------------------


class Tripwires:
    """Raises OSError(ENOTSUP) on every process-spawn API and records who called it."""

    OS_FUNCTIONS = (
        "fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen", "startfile",
        "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    )

    def __init__(self, sites: _SitePaths, simulate: str) -> None:
        self.sites = sites
        self.simulate = simulate
        self.events: List[dict] = []
        self.network: List[dict] = []
        self._lock = threading.Lock()
        self._originals: List[Tuple[Any, str, Any]] = []

    # -- recording --------------------------------------------------------

    def _frames(self) -> List[traceback.FrameSummary]:
        frames = traceback.extract_stack()
        return [f for f in frames if self.sites.where(f.filename) != "harness"]

    def _frame_dict(self, frame: Optional[traceback.FrameSummary]) -> Optional[dict]:
        if frame is None:
            return None
        return {
            "where": self.sites.where(frame.filename),
            "file": self.sites.display(frame.filename),
            "line": frame.lineno,
            "function": frame.name,
            "code": (frame.line or "").strip()[:200],
        }

    def _stack_lines(self, frames: List[traceback.FrameSummary], limit: int = 30) -> List[str]:
        return [f"{self.sites.display(f.filename)}:{f.lineno} in {f.name}" for f in frames[-limit:]]

    def fire(self, kind: str, api: str) -> OSError:
        frames = self._frames()
        caller = None
        for frame in reversed(frames):
            if self.sites.where(frame.filename) == "stdlib" and self.sites.is_spawn_impl(frame.filename):
                continue
            caller = frame
            break
        bundle_frame = next((f for f in reversed(frames) if self.sites.where(f.filename) == "bundle"), None)
        event = {
            "kind": kind,
            "api": api,
            "phase": _phase(),
            "thread": threading.current_thread().name,
            "caller": self._frame_dict(caller),
            "bundle_frame": self._frame_dict(bundle_frame),
            "direct": bool(caller is not None and self.sites.where(caller.filename) == "bundle"),
            "propagated": None,
            "stack": self._stack_lines(frames),
        }
        with self._lock:
            event["id"] = len(self.events) + 1
            self.events.append(event)
        exc = OSError(
            ENOTSUP,
            f"host_smoke tripwire: {api} is not available on {self.simulate} "
            "(no worker processes on mobile; gate the call with mobile_runtime)",
        )
        exc._host_smoke_event = event["id"]  # type: ignore[attr-defined]
        return exc

    def mark_phase(self, phase: str, exc: Optional[BaseException]) -> None:
        """Mark this phase's events propagated (reached the phase runner) or handled."""
        ids = _tripwire_ids(exc) if exc is not None else set()
        with self._lock:
            for event in self.events:
                if event["phase"] == phase and event["propagated"] is None:
                    event["propagated"] = event["id"] in ids

    def failing_events(self) -> List[dict]:
        return [e for e in self.events if e["direct"] or e["propagated"] is not False]

    def handled_events(self) -> List[dict]:
        return [e for e in self.events if not e["direct"] and e["propagated"] is False]

    # -- installation -----------------------------------------------------

    def _patch(self, owner: Any, attr: str, replacement: Callable) -> None:
        original = getattr(owner, attr, None)
        if original is None:
            return
        try:
            replacement.__name__ = getattr(original, "__name__", attr)
            replacement.__qualname__ = getattr(original, "__qualname__", attr)
        except (AttributeError, TypeError):
            pass
        setattr(owner, attr, replacement)
        self._originals.append((owner, attr, original))

    def _simple(self, kind: str, api: str) -> Callable:
        wire = self

        def tripwire(*_args, **_kwargs):
            raise wire.fire(kind, api)

        return tripwire

    def install(self) -> None:
        import concurrent.futures.process as cf_process
        import multiprocessing.pool as mp_pool
        import multiprocessing.process as mp_process

        self._patch(
            cf_process.ProcessPoolExecutor,
            "__init__",
            self._simple("process_pool", "concurrent.futures.ProcessPoolExecutor"),
        )
        self._patch(
            mp_process.BaseProcess, "start", self._simple("multiprocessing", "multiprocessing Process.start")
        )
        wire = self

        def pool_init(pool_self, *_args, **_kwargs):
            # Pool.__del__ reads these; set them so the refused pool is collected quietly.
            pool_self._pool = []
            pool_self._state = mp_pool.INIT
            if isinstance(pool_self, mp_pool.ThreadPool):
                raise wire.fire("multiprocessing", "multiprocessing.pool.ThreadPool (needs sem_open)")
            raise wire.fire("multiprocessing", "multiprocessing.Pool")

        self._patch(mp_pool.Pool, "__init__", pool_init)
        try:
            import multiprocessing.synchronize as mp_sync
        except ImportError:
            mp_sync = None
        if mp_sync is not None:
            self._patch(
                mp_sync.SemLock,
                "__init__",
                self._simple("multiprocessing", "multiprocessing.synchronize.SemLock (no sem_open on Android)"),
            )
        self._patch(subprocess.Popen, "__init__", self._simple("subprocess", "subprocess.Popen"))
        for name in self.OS_FUNCTIONS:
            if callable(getattr(os, name, None)):
                self._patch(os, name, self._simple("os", f"os.{name}"))
        self._install_network_guard()

    # -- network ----------------------------------------------------------

    @staticmethod
    def _is_local_host(host: Any) -> bool:
        if host is None:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        host = str(host).strip("[]").lower()
        return host in ("", "localhost", "::1", "0.0.0.0", "::") or host.startswith("127.") or host.endswith(".localhost")

    def _record_network(self, api: str, target: str) -> None:
        frames = self._frames()
        bundle_frame = next((f for f in reversed(frames) if self.sites.where(f.filename) == "bundle"), None)
        with self._lock:
            self.network.append(
                {
                    "id": len(self.network) + 1,
                    "api": api,
                    "target": target,
                    "phase": _phase(),
                    "thread": threading.current_thread().name,
                    "bundle_frame": self._frame_dict(bundle_frame),
                    "stack": self._stack_lines(frames),
                }
            )

    def _install_network_guard(self) -> None:
        wire = self
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_getaddrinfo = socket.getaddrinfo

        def _check(sock, address, api):
            family = getattr(sock, "family", None)
            if family in (socket.AF_INET, getattr(socket, "AF_INET6", None)) and isinstance(address, tuple):
                host = address[0]
                if not wire._is_local_host(host):
                    wire._record_network(api, f"{host}:{address[1] if len(address) > 1 else '?'}")
                    raise OSError(ENETUNREACH, f"host_smoke: network access to {host} is blocked (offline smoke)")

        def connect(sock, address):
            _check(sock, address, "socket.connect")
            return original_connect(sock, address)

        def connect_ex(sock, address):
            _check(sock, address, "socket.connect_ex")
            return original_connect_ex(sock, address)

        def getaddrinfo(host, *args, **kwargs):
            if not wire._is_local_host(host) and not _looks_numeric(host):
                wire._record_network("socket.getaddrinfo", str(host))
                raise socket.gaierror(getattr(socket, "EAI_NONAME", -2), f"host_smoke: DNS lookup of {host} is blocked")
            return original_getaddrinfo(host, *args, **kwargs)

        self._patch(socket.socket, "connect", connect)
        self._patch(socket.socket, "connect_ex", connect_ex)
        self._patch(socket, "getaddrinfo", getaddrinfo)

    def uninstall(self) -> None:
        while self._originals:
            owner, attr, original = self._originals.pop()
            try:
                setattr(owner, attr, original)
            except (AttributeError, TypeError):
                pass


def _looks_numeric(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    try:
        import ipaddress

        ipaddress.ip_address(str(host).strip("[]"))
        return True
    except ValueError:
        return False


def _tripwire_ids(exc: Optional[BaseException]) -> set:
    ids = set()
    stack: List[Any] = [exc]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        marker = getattr(current, "_host_smoke_event", None)
        if marker is not None:
            ids.add(marker)
        stack.append(getattr(current, "__cause__", None))
        stack.append(getattr(current, "__context__", None))
        stack.extend(getattr(current, "exceptions", None) or ())
    return ids


# --------------------------------------------------------------------------
# Output capture (backend modules print a lot and some re-wrap sys.stdout)
# --------------------------------------------------------------------------


class _Sink(io.BytesIO):
    def close(self) -> None:  # a module's TextIOWrapper(sys.stdout.buffer) may be collected
        pass


class OutputCapture:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.sink = _Sink()
        self.stream = io.TextIOWrapper(self.sink, encoding="utf-8", errors="replace", line_buffering=True)
        self.real_stdout = sys.stdout
        self.real_stderr = sys.stderr

    def start(self) -> None:
        if self.enabled:
            sys.stdout = self.stream
            sys.stderr = self.stream

    def restore(self) -> Tuple[bool, bool]:
        """Put the capture (or the real streams) back; returns (stdout_replaced, stderr_replaced)."""
        expected_out = self.stream if self.enabled else self.real_stdout
        expected_err = self.stream if self.enabled else self.real_stderr
        out_changed = sys.stdout is not expected_out
        err_changed = sys.stderr is not expected_err
        if self.enabled:
            sys.stdout = self.stream
            sys.stderr = self.stream
        else:
            sys.stdout = self.real_stdout
            sys.stderr = self.real_stderr
        return out_changed, err_changed

    def take(self, max_lines: int = 40) -> List[str]:
        """Return and clear the captured output (last ``max_lines`` lines)."""
        if not self.enabled:
            return []
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass
        text = self.sink.getvalue().decode("utf-8", "replace")
        self.sink.seek(0)
        self.sink.truncate()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return [_short(ln, 240) for ln in lines[-max_lines:]]

    def stop(self) -> None:
        sys.stdout = self.real_stdout
        sys.stderr = self.real_stderr


# --------------------------------------------------------------------------
# Filesystem snapshot (stray writes outside the writable dirs)
# --------------------------------------------------------------------------


def _snapshot(root: Path, recursive: bool = True) -> Dict[str, Tuple[int, int]]:
    """relpath -> (size or -1 for dirs, mtime_ns)."""
    result: Dict[str, Tuple[int, int]] = {}
    if not root.is_dir():
        return result
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = os.path.relpath(entry.path, root)
            if entry.is_dir(follow_symlinks=False):
                result[rel] = (-1, 0)
                if recursive:
                    stack.append(Path(entry.path))
            else:
                result[rel] = (st.st_size, st.st_mtime_ns)
    return result


def _diff_snapshots(
    root: Path, before: Dict[str, Tuple[int, int]], after: Dict[str, Tuple[int, int]], allowed: List[str],
    created_only: bool = False,
) -> List[dict]:
    changes = []
    for rel in sorted(set(before) | set(after)):
        full = _norm(root / rel)
        if any(_under(full, a) for a in allowed):
            continue
        if rel not in before:
            changes.append({"path": str(root / rel), "change": "created"})
        elif created_only:
            continue
        elif rel not in after:
            changes.append({"path": str(root / rel), "change": "deleted"})
        elif before[rel][0] != -1 and before[rel] != after[rel]:
            changes.append({"path": str(root / rel), "change": "modified"})
    return changes


def _make_read_only(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for name in dirnames:
            path = os.path.join(dirpath, name)
            os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(root, stat.S_IMODE(os.stat(root).st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            try:
                os.chmod(os.path.join(dirpath, name), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            except OSError:
                pass
    try:
        os.chmod(root, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass


def _probe_read_only(directory: Path) -> bool:
    probe = directory / f".host-smoke-probe-{os.getpid()}"
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("x")
    except OSError:
        return True
    try:
        os.chmod(probe, stat.S_IWRITE | stat.S_IREAD)
        probe.unlink()
    except OSError:
        pass
    return False


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def _compile_legacy_pyc(directory: Path) -> int:
    """Ship ``.pyc`` only (legacy layout), the way ``flet build`` compiles the app dir."""
    import py_compile

    count = 0
    for path in sorted(directory.rglob("*.py")):
        py_compile.compile(str(path), cfile=str(path.with_suffix(".pyc")), doraise=True)
        path.unlink()
        count += 1
    return count


def build_layout(root: Path, bundle: Path, assets: Path, simulate: str, pyc: bool) -> Tuple[Dict[str, Path], int]:
    """Create the temp device layout under ``root``; returns (key paths, .pyc files compiled)."""
    app = root / "app"
    app.mkdir(parents=True)
    backend = app / "backend"
    _copy_tree(bundle, backend)
    _copy_tree(APP_SRC_DIR / "glossarion_mobile", app / "glossarion_mobile")
    shutil.copy2(APP_SRC_DIR / "main.py", app / "main.py")
    assets_dst = app / "assets"
    assets_dst.mkdir()
    for group in ASSET_GROUPS:
        if (assets / group).is_dir():
            _copy_tree(assets / group, assets_dst / group)
    compiled = _compile_legacy_pyc(backend) if pyc else 0

    storage = root / "storage"
    dirs = {name: storage / name for name in ("data", "cache", "temp")}
    for directory in dirs.values():
        directory.mkdir(parents=True)
    device_home = root / "device_home"  # the original HOME the platform gives the app
    device_home.mkdir()
    if simulate == "ios":
        (device_home / "Documents").mkdir()  # iOS Documents (Files-visible): docs = Documents/Glossarion
    host = root / "host"  # host-only env redirects (APPDATA...); not writable on a device
    host.mkdir()
    _make_read_only(app)
    layout = {"app": app, "backend": backend, "assets": assets_dst, "device_home": device_home, "host": host}
    layout.update(dirs)
    return layout, compiled


def collect_bundle(out: Path) -> None:
    """Run the collector into ``out`` (before any guard is installed)."""
    cmd = [sys.executable, str(COLLECTOR), "--out", str(out), "--no-cache"]
    proc = subprocess.run(cmd, cwd=str(MOBILE_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not (out / "_bundle_info.py").is_file():
        lines = (proc.stdout + proc.stderr).strip().splitlines()
        errors = [ln for ln in lines if ln.startswith("ERROR")][:20]
        tail = "\n".join(errors + lines[-5:])
        raise SetupError(f"collect_backend.py failed (exit {proc.returncode}):\n{tail}")


def bundle_modules(backend: Path, first: Tuple[str, ...] = ()) -> List[str]:
    names = set()
    for entry in os.scandir(backend):
        if entry.is_file() and entry.name.endswith((".py", ".pyc")):
            stem = entry.name.rsplit(".", 1)[0]
            if stem != "__init__" and stem.isidentifier():
                names.add(stem)
        elif entry.is_dir() and entry.name.isidentifier() and (
            os.path.isfile(os.path.join(entry.path, "__init__.py")) or os.path.isfile(os.path.join(entry.path, "__init__.pyc"))
        ):
            names.add(entry.name)
    ordered = [n for n in first if n in names]
    ordered += sorted(names - set(ordered), key=str.lower)
    return ordered


# --------------------------------------------------------------------------
# Host checks (selftest Context API: ctx.need / ctx.paths / ctx.strict)
# --------------------------------------------------------------------------


class KeyState:
    def __init__(self) -> None:
        self.api_key: Optional[bytes] = None
        self.api_form: Optional[str] = None
        self.api_setter = False
        self.token_key: Optional[bytes] = None
        self.token_setter = False
        self.notes: List[str] = []


def _bundled_module(name: str) -> Optional[Any]:
    """Import ``name``; None when the bundle simply does not contain it."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


def apply_key_setters(state: KeyState) -> None:
    """Device order: keys from SecureStorage go into the setters before the warm import.

    Uses the app's own installer (``secure_keys.install_backend_keys``), so the
    self-test's ``encryption_keys`` check sees the same state as on a device.
    """
    aek = _bundled_module("api_key_encryption")
    if aek is None:
        state.notes.append("api_key_encryption is not in the bundle")
    elif not callable(getattr(aek, "set_key_material", None)):
        state.notes.append("api_key_encryption.set_key_material does not exist yet (P24)")
    tok = _bundled_module("token_encryption")
    if tok is None:
        state.notes.append("token_encryption is not in the bundle")
    elif not callable(getattr(tok, "set_symmetric_key", None)):
        state.notes.append("token_encryption.set_symmetric_key does not exist yet (P25)")
    if state.notes:
        return
    from glossarion_mobile.services import secure_keys

    api_key, token_key = secure_keys.new_api_key(), secure_keys.new_token_key()
    status = secure_keys.install_backend_keys(api_key, token_key, source="host_smoke")
    if not status.installed:
        raise AssertionError("; ".join(status.errors))
    state.api_setter = state.token_setter = True
    state.api_key, state.api_form, state.token_key = api_key, "fernet-key", token_key


def make_host_checks(selftest: Any, keys: KeyState) -> Dict[str, Callable]:
    CheckSkipped = selftest.CheckSkipped
    fixtures = selftest.fixtures

    def check_key_material(ctx) -> dict:
        paths = ctx.require_bootstrap()
        if not keys.api_setter:
            raise CheckSkipped("; ".join(keys.notes) or "key setters missing")
        fernet_mod = ctx.need("cryptography.fernet")
        aek = importlib.import_module("api_key_encryption")
        handler = aek.get_handler()
        plain = "sk-host-smoke-\U0001F511-값"
        token = handler.encrypt_value(plain)
        if not isinstance(token, str) or not token.startswith("ENC:"):
            raise AssertionError(f"encrypt_value did not encrypt (got {str(token)[:24]!r})")
        if handler.decrypt_value(token) != plain:
            raise AssertionError("decrypt_value(encrypt_value(x)) != x")
        try:
            injected = fernet_mod.Fernet(keys.api_key).decrypt(base64.b64decode(token[4:])).decode("utf-8")
        except Exception as exc:
            raise AssertionError(f"the value was not encrypted with the key given to set_key_material: {exc!r}")
        if injected != plain:
            raise AssertionError("Fernet(injected key) decrypted a different value")
        key_files = [str(p) for p in (paths.home / ".glossarion_key", paths.home / "glossarion_key.txt") if p.exists()]
        if key_files:
            raise AssertionError(f"a key file was written although set_key_material was used: {key_files}")
        detail = {"api_key_form": keys.api_form, "token_setter": keys.token_setter}
        if keys.token_setter:
            tok = importlib.import_module("token_encryption")
            tokens = {"access_token": "host-smoke", "n": 1}
            if tok.decrypt_tokens(tok.encrypt_tokens(tokens)) != tokens:
                raise AssertionError("token_encryption round trip failed with set_symmetric_key")
            detail["token_round_trip"] = True
        if keys.notes:
            detail["notes"] = keys.notes
        return detail

    def check_chapter_extractor_pool(ctx) -> dict:
        paths = ctx.require_bootstrap()
        for module in ("bs4", "lxml", "ebooklib"):
            ctx.need(module)
        epub = fixtures.find_selftest_epub(paths.assets_dir)
        if epub is None:
            message = "selftest EPUB missing from assets/selftest (run tools/prepare_assets.py)"
            if ctx.strict:
                raise AssertionError(message)
            raise CheckSkipped(message)
        extractor = importlib.import_module("Chapter_Extractor")
        out_dir = paths.temp / "host-smoke-chapters"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        with zipfile.ZipFile(epub) as zf:
            html_files = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        from concurrent.futures import thread as cf_thread

        pools: List[str] = []
        original_init = cf_thread.ThreadPoolExecutor.__init__

        def recording_init(executor, max_workers=None, thread_name_prefix="", *args, **kwargs):
            pools.append(f"{thread_name_prefix or '?'}(max_workers={max_workers})")
            return original_init(executor, max_workers, thread_name_prefix, *args, **kwargs)

        old_workers = os.environ.get("EXTRACTION_WORKERS")
        os.environ["EXTRACTION_WORKERS"] = "2"
        cf_thread.ThreadPoolExecutor.__init__ = recording_init
        try:
            with zipfile.ZipFile(epub) as zf:
                chapters = extractor.extract_chapters(zf, str(out_dir))
        finally:
            cf_thread.ThreadPoolExecutor.__init__ = original_init
            if old_workers is None:
                os.environ.pop("EXTRACTION_WORKERS", None)
            else:
                os.environ["EXTRACTION_WORKERS"] = old_workers
        manifest = selftest.rb._load_toml(epub.parent / "MANIFEST.toml") or {}
        expected = (manifest.get("epub") or {}).get("chapters") or 12
        if not isinstance(chapters, list) or len(chapters) < expected:
            count = len(chapters) if isinstance(chapters, list) else type(chapters).__name__
            raise AssertionError(f"extract_chapters returned {count} chapters, expected >= {expected}")
        empty = [c.get("num") for c in chapters if isinstance(c, dict) and not (c.get("body") or c.get("content"))]
        return {
            "epub": epub.name,
            "html_files": len(html_files),
            "pool_path": len(html_files) > 10,
            "chapters": len(chapters),
            "empty_bodies": empty[:10],
            "thread_pools": pools[:20],
        }

    def check_jaro_winkler(ctx) -> dict:
        jw_mod = ctx.need("rapidfuzz.distance")
        ddc = importlib.import_module("duplicate_detection_config")
        if getattr(ddc, "_HAS_JELLYFISH", False):
            raise AssertionError("duplicate_detection_config imported jellyfish although it is blocked")
        config = {"algorithms": ["jaro_winkler"], "partial_ratio_weight": 1.0}
        pairs = (("kim dokja", "kim dok-ja"), ("yoo joonghyuk", "yu jung-hyeok"), ("이서연", "이서윤"))
        scores = {}
        for left, right in pairs:
            got = ddc.calculate_similarity_with_config(left, right, config)
            want = jw_mod.JaroWinkler.similarity(unicodedata.normalize("NFC", left), unicodedata.normalize("NFC", right))
            if abs(float(got) - float(want)) > 1e-9:
                raise AssertionError(
                    f"jaro_winkler({left!r}, {right!r}) = {got!r}, RapidFuzz JaroWinkler = {want!r}: "
                    "the Jaro-Winkler path is inactive without jellyfish (P26 RapidFuzz fallback missing?)"
                )
            scores[f"{left}|{right}"] = round(float(got), 6)
        info = ddc.get_algorithm_display_info() if hasattr(ddc, "get_algorithm_display_info") else None
        return {"scores": scores, "display": info}

    def check_pdf_mupdf_html(ctx) -> dict:
        paths = ctx.require_bootstrap()
        fitz = ctx.need("fitz")
        try:
            shim = importlib.import_module("pdf_mupdf_html")
        except ModuleNotFoundError as exc:
            if exc.name == "pdf_mupdf_html":
                raise CheckSkipped("pdf_mupdf_html is not in the bundle yet")
            raise
        body = "".join(
            f'<h1 id="ch{i}">Chapter {i}</h1><p>Paragraph {i} of the host smoke PDF.</p>' for i in (1, 2, 3)
        )
        html = f"<html><head><title>Host smoke</title></head><body>{body}</body></html>"
        document = shim.HTML(string=html, base_url=str(paths.temp)).render()
        anchors = set()
        bookmarks = 0
        for page in getattr(document, "pages", []) or []:
            anchors.update((getattr(page, "anchors", None) or {}).keys())
            bookmarks += len(getattr(page, "bookmarks", None) or [])
        target = paths.temp / "host-smoke-shim.pdf"
        result = document.write_pdf(str(target))
        data = target.read_bytes() if target.is_file() else result
        if not isinstance(data, (bytes, bytearray)) or not bytes(data[:5]) == b"%PDF-":
            raise AssertionError("write_pdf produced no PDF")
        doc = fitz.open(stream=bytes(data), filetype="pdf")
        try:
            text = "".join(page.get_text() for page in doc)
            pages = doc.page_count
            toc = doc.get_toc()
        finally:
            doc.close()
        missing = {"ch1", "ch2", "ch3"} - anchors
        if missing:
            raise AssertionError(f"rendered pages have no anchors for {sorted(missing)}")
        if "Chapter 3" not in text:
            raise AssertionError("chapter text missing from the PDF")
        if not bookmarks and not toc:
            raise AssertionError("no bookmarks/outline for the h1 headings")
        return {"pages": pages, "anchors": sorted(anchors), "bookmarks": bookmarks, "toc": len(toc),
                "engine": getattr(shim, "ENGINE_NAME", None)}

    def check_headless_owner_env(ctx) -> dict:
        paths = ctx.require_bootstrap()
        for module in ("headless_owner", "run_env"):
            importlib.import_module(module)  # must come from the bundle (import phase checks the origin)
        from glossarion_mobile.ui.screens import env_preview

        config_file = Path(os.environ.get("CONFIG_FILE") or paths.data / "config.json")
        config_before = config_file.read_bytes() if config_file.is_file() else None
        env_before = dict(os.environ)
        cwd_before = os.getcwd()
        argv_before = list(sys.argv)
        input_path = env_preview.preview_input_path("epub", str(paths.data))
        # The app's Env preview: HeadlessOwner(fresh config) + run_env.build_translation_env,
        # under the preview lock with the process env/argv/cwd restored afterwards.
        result = env_preview.build_env_preview({}, input_path=input_path, api_key="")
        if not result.ok:
            raise AssertionError(f"build_env_preview failed: {result.error}")
        env = {row.key: row.value for row in result.rows}
        wrong = {k: env.get(k) for k, v in FRESH_INSTALL_ENV.items() if env.get(k) != v}
        if wrong:
            raise AssertionError(f"fresh-install run env differs from the desktop: {wrong} (expected {FRESH_INSTALL_ENV})")
        if result.count < 300:
            raise AssertionError(f"only {result.count} variables in the translation env (desktop builds ~400)")
        changed = sorted(k for k in set(env_before) | set(os.environ) if env_before.get(k) != os.environ.get(k))
        if changed:
            raise AssertionError(f"the preview left process env changes behind: {changed[:20]}")
        if os.getcwd() != cwd_before or sys.argv != argv_before:
            raise AssertionError("the preview did not restore the working directory / sys.argv")
        config_after = config_file.read_bytes() if config_file.is_file() else None
        if config_after != config_before:
            raise AssertionError(f"building a HeadlessOwner wrote {config_file} (owners never persist config)")
        return {
            "variables": result.count,
            "redacted": sum(1 for row in result.rows if row.redacted),
            "env": {k: env.get(k) for k in FRESH_INSTALL_ENV},
            "secs": result.secs,
        }

    def check_manga_pipeline(ctx) -> dict:
        """U8 Tools › Manga on the bundle: RT-DETR through Python onnxruntime (a tiny synthetic
        export with three constant boxes, written where the download lands, so nothing is fetched)
        and one page through ``manga_runner.HeadlessMangaRunner`` with a HeadlessOwner, custom-api
        OCR + translation answered by the loopback fake server (OCR-response mode), inpainting
        skipped; process state and config.json must come out unchanged."""
        paths = ctx.require_bootstrap()
        for module in ("numpy", "cv2", "PIL", "onnxruntime"):
            ctx.need(module)
        import numpy

        from glossarion_mobile.diagnostics.fake_llm_server import (FAKE_MANGA_OCR_TEXT, FAKE_MARKER, FAKE_MODEL,
                                                                   FakeLLMServer)

        work = paths.temp / "host-smoke-manga"
        if work.exists():
            shutil.rmtree(work)
        page = work / "pages" / "001.png"
        page.parent.mkdir(parents=True)
        page.write_bytes(fixtures.manga_page_png(1))
        detail: Dict[str, Any] = {}

        # 1. RT-DETR on the device path: the bundled bubble_detector, Python onnxruntime (the C++
        #    backend is excluded), the bootstrap's BUBBLE_CACHE_DIR (<data>/models/detector).
        bd = importlib.import_module("bubble_detector")
        if not bd._rtdetr_python_onnx_allowed():
            raise AssertionError("bubble_detector does not allow the Python onnxruntime RT-DETR path on mobile")
        cache = Path(os.environ.get("BUBBLE_CACHE_DIR") or "")
        if not cache.is_absolute() or not _under(str(cache), str(paths.data)):
            raise AssertionError(f"BUBBLE_CACHE_DIR {cache} is not under the data dir {paths.data}")
        detector = bd.BubbleDetector(config_path=str(work / "bubble_config.json"))
        if _norm(detector.cache_dir) != _norm(cache):
            raise AssertionError(f"BubbleDetector cache dir {detector.cache_dir} != BUBBLE_CACHE_DIR {cache}")
        filename = "detector.onnx"
        cache.mkdir(parents=True, exist_ok=True)
        planted = [cache / filename, cache / "config.json"]
        planted[0].write_bytes(base64.b64decode(SYNTHETIC_RTDETR_ONNX_B64))
        planted[1].write_bytes(b"{}")
        try:
            if not detector.load_rtdetr_onnx_model(onnx_filename=filename, force_reload=True):
                raise AssertionError("load_rtdetr_onnx_model failed on the synthetic export")
            cls = bd.BubbleDetector
            if cls._rtdetr_onnx_use_cpp or detector.rtdetr_onnx_session is None:
                raise AssertionError("RT-DETR did not load through a Python onnxruntime session")
            image = numpy.full((200, 300, 3), 255, dtype=numpy.uint8)
            boxes = detector.detect_bubbles(str(page), confidence=0.3, use_rtdetr=True)
            raw = detector.detect_with_rtdetr_onnx(image=image, confidence=0.3)
            if not boxes or not isinstance(raw, dict) or not any(raw.values()):
                raise AssertionError(f"no detections from the synthetic export: {boxes!r} / {raw!r}")
            detail["rtdetr"] = {"providers": list(cls._rtdetr_onnx_providers or []), "boxes": len(boxes),
                                "classes": {k: len(v) for k, v in raw.items()}}
        finally:
            try:
                cls = bd.BubbleDetector
                for name, value in (("_rtdetr_onnx_shared_session", None), ("_rtdetr_onnx_loaded", False),
                                    ("_rtdetr_onnx_model_key", None), ("_rtdetr_onnx_model_path", None)):
                    if hasattr(cls, name):
                        setattr(cls, name, value)
            except Exception:
                pass
            for path in planted:
                try:
                    path.unlink()
                except OSError:
                    pass

        # 2. One page through the mobile batch runner (the MANGA job's code path) on the bundle.
        job_runner = importlib.import_module("job_runner")
        stop_control = importlib.import_module("stop_control")
        headless_owner = importlib.import_module("headless_owner")
        manga_runner = importlib.import_module("manga_runner")
        config_file = Path(os.environ.get("CONFIG_FILE") or paths.data / "config.json")
        config_before = config_file.read_bytes() if config_file.is_file() else None
        env_before = dict(os.environ)
        cwd_before = os.getcwd()
        lines: List[str] = []

        class _Host:
            def log(self, text="", **_kw):
                lines.append(str(text))

            def is_stop_requested(self) -> bool:
                return False

            def is_graceful_stop(self) -> bool:
                return False

            def emit(self, kind, **data):
                if kind == "log":
                    lines.append(str(data.get("text", "")))

            def ask(self, kind, **data):
                return None

        host = _Host()
        with FakeLLMServer() as server:
            server.ocr_text = FAKE_MANGA_OCR_TEXT
            config = {
                "model": FAKE_MODEL, "api_key": "sk-host-smoke-manga", "use_custom_openai_endpoint": True,
                "openai_base_url": server.url, "output_language": "English", "delay": 0,
                "manga_ocr_provider": "custom-api", "manga_skip_inpainting": True,
                "manga_settings": {"ocr": {"bubble_detection_enabled": False}},
            }
            with job_runner.job_process_state(host.log, lock=job_runner.JOB_LOCK):
                # The smoke points every proxy at a dead port: the loopback fake server must bypass
                # them (os.environ is case-insensitive on Windows); the scope restores the env.
                for key in ("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy"):
                    os.environ[key] = "127.0.0.1,localhost"
                stop_control.reset_for_new_run(kind="translation")
                owner = headless_owner.HeadlessOwner(config, host=host)
                runner = manga_runner.HeadlessMangaRunner(owner, host=host, files=[str(page)],
                                                          output_root=str(work / "Output"))
                summary = dict(runner.run() or {})
            records = server.records()
        outputs = [p for p in (summary.get("outputs") or []) if p]
        kinds = [r.kind for r in records]
        if not summary.get("ok") or int(summary.get("completed") or 0) != 1 or len(outputs) != 1:
            tail = " | ".join(line for line in lines[-12:] if line)
            raise AssertionError(f"the manga run did not translate the page: {summary!r} ({tail[-1500:]})")
        if kinds.count("vision") != 1 or "translation" not in kinds:
            raise AssertionError(f"expected one OCR request and a translation request, got {kinds}")
        if not any(FAKE_MARKER in (r.reply or "") for r in records if r.kind == "translation"):
            raise AssertionError("the translation request did not carry the OCR text")
        if not os.path.isfile(outputs[0]) or os.path.getsize(outputs[0]) == 0:
            raise AssertionError(f"no translated page written: {outputs[0]}")
        changed = sorted(k for k in set(env_before) | set(os.environ) if env_before.get(k) != os.environ.get(k))
        if changed:
            raise AssertionError(f"the manga run left process env changes behind: {changed[:20]}")
        if os.getcwd() != cwd_before:
            raise AssertionError("the manga run did not restore the working directory")
        config_after = config_file.read_bytes() if config_file.is_file() else None
        if config_after != config_before:
            raise AssertionError(f"the manga run wrote {config_file} (owners never persist config)")
        detail["runner"] = {"requests": kinds, "output": os.path.basename(outputs[0]),
                            "cbz": [os.path.basename(p) for p in summary.get("cbz_paths") or []]}
        return detail

    return {
        "key_material": check_key_material,
        "chapter_extractor_pool": check_chapter_extractor_pool,
        "jaro_winkler": check_jaro_winkler,
        "pdf_mupdf_html": check_pdf_mupdf_html,
        "headless_owner_env": check_headless_owner_env,
        "manga_pipeline": check_manga_pipeline,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class Smoke:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.simulate = args.simulate
        self.report: Dict[str, Any] = {
            "ok": False,
            "simulate": self.simulate,
            "python": _platform.python_version(),
            "host_platform": sys.platform,
            "strict": not args.lenient,
            "pyc": bool(args.pyc),
            "failures": [],
            "warnings": [],
        }
        self.capture = OutputCapture(enabled=not args.verbose)
        self.blocker: Optional[ImportBlocker] = None
        self.wires: Optional[Tripwires] = None
        self.rb: Any = None

    # -- phases -----------------------------------------------------------

    def _run(self, phase: str, func: Callable[[], Any]) -> Tuple[str, Any, Optional[BaseException], float]:
        _set_phase(phase)
        t0 = time.monotonic()
        exc: Optional[BaseException] = None
        status, value = "ok", None
        try:
            value = func()
        except BaseException as error:  # SystemExit from a backend import must not end the smoke
            if isinstance(error, KeyboardInterrupt):
                raise
            exc = error
            status = "skip" if self._is_skip(error) else "fail"
        finally:
            if self.wires is not None:
                self.wires.mark_phase(phase, exc)
            _set_phase("between-phases")
        return status, value, exc, round(time.monotonic() - t0, 3)

    _skip_types: Tuple[type, ...] = ()

    def _is_skip(self, error: BaseException) -> bool:
        return bool(self._skip_types) and isinstance(error, self._skip_types)

    @staticmethod
    def _error(exc: BaseException) -> Dict[str, Any]:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {"error": f"{type(exc).__name__}: {_short(exc, 500)}", "traceback": tb[-3000:]}

    # -- main flow --------------------------------------------------------

    def run(self) -> int:
        args = self.args
        work = Path(args.work_dir).resolve() if args.work_dir else None
        if work is not None:
            work.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"glossarion-host-smoke-{self.simulate}-", dir=str(work) if work else None))
        self.report["root"] = str(root)
        original_cwd = Path.cwd()
        try:
            return self._run_in(root, original_cwd)
        finally:
            self._teardown(root)

    def _run_in(self, root: Path, original_cwd: Path) -> int:
        args = self.args
        report = self.report
        if args.collect:
            collected = root / "collected"
            print(f"host_smoke: collecting the backend into {collected} ...", flush=True)
            collect_bundle(collected)
            bundle = collected
        else:
            bundle = Path(args.bundle).resolve()
        if not (bundle / "_bundle_info.py").is_file() and not (bundle / "_bundle_info.pyc").is_file():
            raise SetupError(
                f"no collected bundle at {bundle} (run `python tools/collect_backend.py --out app/backend` "
                "from src/mobile, or pass --collect)"
            )
        assets = Path(args.assets).resolve()
        layout, compiled = build_layout(root, bundle, assets, self.simulate, args.pyc)
        if args.collect:
            shutil.rmtree(root / "collected", ignore_errors=True)
        backend = layout["backend"]
        report["bundle"] = {
            "source": "collected by --collect" if args.collect else str(bundle),
            "copy": str(backend),
            "pyc_files": compiled,
            "read_only": _probe_read_only(backend),
            "info": _bundle_info_summary(bundle),
        }
        if not report["bundle"]["read_only"]:
            note = "the bundle copy is still writable here"
            if os.name == "nt":
                note += " (Windows ignores read-only on directories)"
            elif hasattr(os, "geteuid") and os.geteuid() == 0:
                note += " (running as root)"
            report["warnings"].append(note + "; stray writes are still caught by the snapshot")
        missing_assets = [g for g in ASSET_GROUPS if not (layout["assets"] / g).is_dir()]
        if missing_assets:
            report["warnings"].append(
                f"assets missing: {', '.join(missing_assets)} (run `python tools/prepare_assets.py` from src/mobile)"
            )

        sites = _SitePaths(backend, layout["app"])
        blocked = blocked_packages(self.simulate)
        report["blocked_packages"] = sorted(blocked)

        self._prepare_environment(root, layout)
        before_root = _snapshot(root)
        before_cwd = _snapshot(original_cwd, recursive=False)

        # Guards first: nothing below may spawn, reach the network or see a blocked package.
        sys.dont_write_bytecode = True
        self.blocker = ImportBlocker(blocked, self.simulate, sites)
        self.blocker.install()
        if self.blocker.evicted:
            report["warnings"].append(f"blocked modules were already imported (evicted): {self.blocker.evicted}")
        self.wires = Tripwires(sites, self.simulate)
        self.wires.install()

        # Device bootstrap (env contract, chdir, backend on sys.path, tiktoken seeding).
        status, paths, exc, secs = self._run("bootstrap", lambda: self._bootstrap(layout))
        if status != "ok":
            raise SetupError(f"runtime_bootstrap.bootstrap() failed: {self._error(exc)['traceback']}")
        report["bootstrap"] = {
            "secs": secs,
            "platform": paths.platform,
            "backend_source": paths.backend_source,
            "errors": list(self.rb.get_state().errors),
            "tiktoken": self.rb.get_state().tiktoken,
        }
        writable = {name: str(p) for name, p in paths.writable_dirs().items()}
        report["writable_dirs"] = writable
        if paths.backend_source != "bundle" or _norm(paths.backend_dir) != _norm(backend):
            report["failures"].append(
                f"backend resolved from {paths.backend_dir} ({paths.backend_source}), expected the bundle {backend}"
            )
        for error in report["bootstrap"]["errors"]:
            report["warnings"].append(f"bootstrap: {error}")

        from glossarion_mobile.diagnostics import selftest

        self._skip_types = (selftest.CheckSkipped,)
        self.capture.start()

        keys = KeyState()
        status, _, exc, secs = self._run("key_setters", lambda: apply_key_setters(keys))
        report["key_setters"] = {"status": status, "api_key_encryption.set_key_material": keys.api_setter,
                                 "token_encryption.set_symmetric_key": keys.token_setter, "notes": keys.notes}
        if exc is not None:
            report["key_setters"].update(self._error(exc))
            report["failures"].append(f"key setters: {self._error(exc)['error']}")
        self.capture.take()

        report["imports"] = self._import_all(backend, selftest)
        report["checks"] = self._run_checks(selftest, keys)
        self.capture.stop()

        # Verdict
        writable_norm = [_norm(p) for p in writable.values()]
        stray = _diff_snapshots(root, before_root, _snapshot(root), writable_norm)
        stray += _diff_snapshots(original_cwd, before_cwd, _snapshot(original_cwd, recursive=False), [], created_only=True)
        report["stray_writes"] = stray
        report["tripwire"] = self.wires.events
        report["network"] = self.wires.network
        report["blocked_imports"] = self.blocker.attempts
        self._verdict()
        return 0 if report["ok"] else 1

    def _prepare_environment(self, root: Path, layout: Dict[str, Path]) -> None:
        env = os.environ
        # A phone has none of these. DISPLAY/WAYLAND_DISPLAY would also make webbrowser's
        # register_standard_browsers() (run by bootstrap) spawn xdg-settings on a Linux host.
        for var in ("GLOSSARION_BACKEND_DIR", "GLOSSARION_ORIGINAL_HOME", "CONFIG_FILE", "GLOSSARION_DATA_DIR",
                    "GLOSSARION_APP_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                    "DISPLAY", "WAYLAND_DISPLAY"):
            env.pop(var, None)
        env["FLET_APP_STORAGE_DATA"] = str(layout["data"])
        env["FLET_APP_STORAGE_CACHE"] = str(layout["cache"])
        env["FLET_APP_STORAGE_TEMP"] = str(layout["temp"])
        env["FLET_ASSETS_DIR"] = str(layout["assets"])
        env["FLET_PLATFORM"] = self.simulate
        env["GLOSSARION_SIMULATE_PLATFORM"] = self.simulate
        env["HOME"] = str(layout["device_home"])
        if os.name == "nt":
            # expanduser() reads USERPROFILE on Windows; keep it on the device HOME (set again after bootstrap).
            env["USERPROFILE"] = str(layout["device_home"])
            env["APPDATA"] = str(layout["host"] / "AppData" / "Roaming")
            env["LOCALAPPDATA"] = str(layout["host"] / "AppData" / "Local")
        for var in PROXY_VARS:
            env[var] = OFFLINE_PROXY
        for var in NO_PROXY_VARS:
            env.pop(var, None)
        # On a device sys.argv[0] is the app's main.py inside the read-only app dir.
        sys.argv = [str(layout["app"] / "main.py")]
        # The tools dir must not shadow backend modules.
        tools = _norm(TOOLS_DIR)
        sys.path[:] = [p for p in sys.path if not p or _norm(p) != tools]

    def _bootstrap(self, layout: Dict[str, Path]):
        app = str(layout["app"])
        if app not in sys.path:
            sys.path.insert(0, app)
        from glossarion_mobile import runtime_bootstrap as rb

        self.rb = rb
        paths = rb.bootstrap(app_dir=layout["app"], force=True)
        if os.name == "nt":
            os.environ["USERPROFILE"] = os.environ.get("HOME", str(paths.home))
        backend = _norm(layout["backend"])
        # Only the bundle may provide backend modules (a repo src/ on sys.path would hide gaps).
        sys.path[:] = [
            p for p in sys.path
            if not p or _norm(p) == backend or not os.path.isfile(os.path.join(p, "TransateKRtoEN.py"))
        ]
        return paths

    def _import_all(self, backend: Path, selftest: Any) -> Dict[str, Any]:
        first = tuple(getattr(selftest.rb, "WARM_IMPORT_MODULES", ())) + ("mobile_runtime",)
        modules = bundle_modules(backend, first)
        results: Dict[str, Any] = {"total": len(modules), "ok": 0, "failed": {}, "outside_bundle": {},
                                   "stdio_replaced": [], "slowest": []}
        timings = []
        backend_norm = _norm(backend)
        threads_before = {t.ident for t in threading.enumerate()}
        for name in modules:
            status, module, exc, secs = self._run(f"import:{name}", lambda n=name: importlib.import_module(n))
            out_replaced, err_replaced = self.capture.restore()
            output = self.capture.take()
            timings.append((secs, name))
            if out_replaced or err_replaced:
                which = [s for s, flag in (("stdout", out_replaced), ("stderr", err_replaced)) if flag]
                results["stdio_replaced"].append({"module": name, "streams": which})
            if status != "ok":
                entry = self._error(exc)
                entry["output_tail"] = output[-15:]
                results["failed"][name] = entry
                continue
            filename = getattr(module, "__file__", None) or ""
            if filename and not _under(_norm(filename), backend_norm):
                results["outside_bundle"][name] = filename
                continue
            results["ok"] += 1
        results["secs"] = round(sum(t for t, _ in timings), 2)
        results["slowest"] = [{"module": n, "secs": t} for t, n in sorted(timings, reverse=True)[:8]]
        results["threads_started"] = sorted(
            t.name for t in threading.enumerate() if t.ident not in threads_before and t.is_alive()
        )
        return results

    def _selected_checks(self, selftest: Any) -> List[Tuple[str, str]]:
        available = [("selftest", name) for name, _ in selftest.SUITES["smoke"]] + [("host", n) for n in HOST_CHECKS]
        spec = (self.args.checks or "all").strip().lower()
        if spec == "all":
            return available
        if spec == "none":
            return []
        wanted = [s.strip() for s in spec.split(",") if s.strip()]
        names = {name for _, name in available}
        unknown = [w for w in wanted if w not in names]
        if unknown:
            raise SetupError(f"unknown checks {unknown}; known: {sorted(names)}")
        return [item for item in available if item[1] in wanted]

    def _run_checks(self, selftest: Any, keys: KeyState) -> List[Dict[str, Any]]:
        funcs = dict(selftest.SUITES["smoke"])
        funcs.update(make_host_checks(selftest, keys))
        strict = not self.args.lenient
        results = []
        for group, name in self._selected_checks(selftest):
            ctx = selftest.Context(strict=strict)
            status, detail, exc, secs = self._run(f"check:{name}", lambda f=funcs[name], c=ctx: f(c))
            self.capture.restore()
            output = self.capture.take()
            entry: Dict[str, Any] = {"name": name, "group": group, "status": status, "secs": secs}
            if status == "ok":
                entry["status"] = "pass"
                entry["detail"] = detail
            elif status == "skip":
                entry["reason"] = str(exc)
            else:
                entry.update(self._error(exc))
                entry["output_tail"] = output[-15:]
            results.append(entry)
        return results

    def _verdict(self) -> None:
        report = self.report
        failures = report["failures"]
        imports = report.get("imports") or {}
        for name, entry in (imports.get("failed") or {}).items():
            failures.append(f"import {name}: {entry['error']}")
        for name, filename in (imports.get("outside_bundle") or {}).items():
            failures.append(f"import {name}: loaded from outside the bundle ({filename})")
        for entry in imports.get("stdio_replaced") or []:
            report["warnings"].append(f"importing {entry['module']} replaced sys.{'/'.join(entry['streams'])}")
        for check in report.get("checks") or []:
            if check["status"] == "fail":
                failures.append(f"check {check['name']}: {check['error']}")
        assert self.wires is not None
        for event in self.wires.failing_events():
            failures.append(f"tripwire #{event['id']}: {_describe_event(event)}")
        for event in self.wires.handled_events():
            report["warnings"].append(f"tripwire #{event['id']} (handled by the caller): {_describe_event(event)}")
        for event in self.wires.network:
            failures.append(f"network #{event['id']}: {event['api']} {event['target']} during {event['phase']}")
        for change in report.get("stray_writes") or []:
            failures.append(f"stray write ({change['change']}): {change['path']}")
        report["ok"] = not failures

    def _teardown(self, root: Path) -> None:
        if self.capture is not None:
            self.capture.stop()
        if self.wires is not None:
            self.wires.uninstall()
        if self.blocker is not None:
            self.blocker.uninstall()
        if self.rb is not None:
            try:
                self.rb.reset(restore_env=True)
            except Exception:
                pass
        if self.args.keep:
            self.report["kept_root"] = str(root)
            return
        _make_writable(root)
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            self.report.setdefault("warnings", []).append(f"could not remove {root} (files still open?)")


def _bundle_info_summary(bundle: Path) -> Dict[str, Any]:
    info_file = bundle / "_bundle_info.py"
    summary: Dict[str, Any] = {}
    try:
        text = info_file.read_text(encoding="utf-8")
    except OSError:
        return summary
    for key in ("BUILD_VERSION", "GIT_SHA", "BUNDLE_SHA256", "MODULE_COUNT"):
        match = re.search(rf"^{key}\s*=\s*(.+)$", text, re.MULTILINE)
        if match:
            value = match.group(1).strip().strip("'\"")
            summary[key] = value[:16] if key == "BUNDLE_SHA256" else value
    return summary


def _describe_event(event: Dict[str, Any]) -> str:
    where = event.get("bundle_frame") or event.get("caller") or {}
    caller = event.get("caller") or {}
    site = f"{where.get('file')}:{where.get('line')} in {where.get('function')}" if where else "?"
    via = ""
    if caller and caller is not where and caller.get("file") != where.get("file"):
        via = f" via {caller.get('file')}:{caller.get('line')} in {caller.get('function')}"
    state = "direct" if event.get("direct") else "indirect"
    if event.get("propagated"):
        state += ", propagated"
    return f"{event['api']} during {event['phase']} at {site}{via} [{state}, thread {event.get('thread')}]"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_summary(report: Dict[str, Any]) -> str:
    lines = []
    status = "PASS" if report.get("ok") else "FAIL"
    lines.append(
        f"host_smoke ({report.get('simulate')}, Python {report.get('python')}, host {report.get('host_platform')}): {status}"
    )
    bundle = report.get("bundle") or {}
    if bundle:
        info = bundle.get("info") or {}
        lines.append(
            f"  bundle: {bundle.get('source')} ({info.get('MODULE_COUNT', '?')} modules, git {info.get('GIT_SHA', '?')[:10]}, "
            f"read-only copy: {bundle.get('read_only')}{', .pyc only' if report.get('pyc') else ''})"
        )
    imports = report.get("imports") or {}
    if imports:
        lines.append(f"  imports: {imports.get('ok')}/{imports.get('total')} ok in {imports.get('secs')} s")
        for name, entry in (imports.get("failed") or {}).items():
            lines.append(f"    FAIL {name}: {entry['error']}")
        for name, filename in (imports.get("outside_bundle") or {}).items():
            lines.append(f"    FAIL {name}: loaded from outside the bundle ({filename})")
    checks = report.get("checks") or []
    if checks:
        counts = {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "fail", "skip")}
        lines.append(f"  checks: {counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped")
        for check in checks:
            if check["status"] == "fail":
                lines.append(f"    FAIL {check['name']}: {_short(check['error'], 400)}")
            elif check["status"] == "skip":
                lines.append(f"    skip {check['name']}: {_short(check.get('reason'), 200)}")
    events = report.get("tripwire") or []
    lines.append(f"  process tripwire: {len(events)} event(s)")
    for event in events:
        lines.append(f"    #{event['id']} {_describe_event(event)}")
        for frame in event.get("stack", [])[-8:]:
            lines.append(f"        {frame}")
    network = report.get("network") or []
    lines.append(f"  network: {len(network)} attempt(s)")
    for event in network:
        lines.append(f"    #{event['id']} {event['api']} {event['target']} during {event['phase']}")
    stray = report.get("stray_writes") or []
    lines.append(f"  stray writes outside the writable dirs: {len(stray)}")
    for change in stray[:30]:
        lines.append(f"    {change['change']}: {change['path']}")
    blocked = report.get("blocked_imports") or {}
    if blocked:
        names = ", ".join(f"{k} (x{v['count']})" for k, v in sorted(blocked.items())[:25])
        lines.append(f"  blocked imports (guarded, expected): {names}")
    for warning in report.get("warnings") or []:
        lines.append(f"  warning: {_short(warning, 400)}")
    if not report.get("ok"):
        lines.append("  failures:")
        for failure in report.get("failures") or []:
            lines.append(f"    - {_short(failure, 400)}")
    return "\n".join(lines)


def format_markdown(report: Dict[str, Any]) -> str:
    status = "PASS" if report.get("ok") else "FAIL"
    imports = report.get("imports") or {}
    checks = report.get("checks") or []
    counts = {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "fail", "skip")}
    events = report.get("tripwire") or []
    rows = [
        f"## Host smoke ({report.get('simulate')}, Python {report.get('python')}): {status}",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Imports | {imports.get('ok', 0)}/{imports.get('total', 0)} ok |",
        f"| Checks | {counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped |",
        f"| Process tripwire | {len(events)} event(s), {sum(1 for e in events if e.get('direct'))} direct |",
        f"| Network attempts | {len(report.get('network') or [])} |",
        f"| Stray writes | {len(report.get('stray_writes') or [])} |",
        "",
    ]
    failures = report.get("failures") or []
    if failures:
        rows.append("<details><summary>Failures</summary>\n")
        rows += [f"- `{_short(f, 300)}`" for f in failures[:60]]
        rows.append("\n</details>\n")
    return "\n".join(rows) + "\n"


def _write_outputs(report: Dict[str, Any], json_target: Optional[str]) -> None:
    text = format_summary(report)
    payload = json.dumps(report, indent=1, ensure_ascii=False, default=str)
    if json_target == "-":
        sys.stderr.write(text + "\n")
        sys.stdout.write(payload + "\n")
    else:
        sys.stdout.write(text + "\n")
        if json_target:
            path = Path(json_target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(format_markdown(report))
        except OSError:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and exercise the collected mobile backend under device conditions "
        "(read-only bundle, process tripwires, blocked desktop packages, offline network)."
    )
    parser.add_argument("--simulate", choices=SIMULATED_PLATFORMS, default="android",
                        help="device platform to simulate (ios also blocks psutil); default android")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE),
                        help="collected backend bundle (default: src/mobile/app/backend)")
    parser.add_argument("--collect", action="store_true",
                        help="run tools/collect_backend.py into the temp dir first and use that bundle")
    parser.add_argument("--assets", default=str(DEFAULT_ASSETS),
                        help="assets dir with tiktoken/ and selftest/ (default: src/mobile/app/assets)")
    parser.add_argument("--json", dest="json_out", metavar="PATH",
                        help="write the full JSON report here ('-' prints only JSON to stdout)")
    parser.add_argument("--checks", default="all",
                        help="comma-separated functional checks, 'all' (default) or 'none' (imports only)")
    parser.add_argument("--pyc", action="store_true",
                        help="ship the bundle copy as legacy .pyc only, like `flet build` (no .py sources)")
    parser.add_argument("--lenient", action="store_true",
                        help="missing third-party packages skip checks instead of failing them")
    parser.add_argument("--work-dir", help="parent directory for the temp root (default: the system temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the temp root for inspection")
    parser.add_argument("-v", "--verbose", action="store_true", help="do not capture backend output")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    for attr in ("bundle", "assets", "work_dir"):
        value = getattr(args, attr)
        if value:
            setattr(args, attr, str((cwd / value).resolve()) if not Path(value).is_absolute() else value)
    if args.json_out and args.json_out != "-":
        args.json_out = str((cwd / args.json_out).resolve())
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass
    smoke = Smoke(args)
    code = 2
    try:
        code = smoke.run()
    except SetupError as exc:
        smoke.report["failures"].append(f"setup: {exc}")
        smoke.report["ok"] = False
        code = 2
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # a harness bug still produces a report
        smoke.report["failures"].append(f"host_smoke crashed: {type(exc).__name__}: {exc}")
        smoke.report["traceback"] = traceback.format_exc()[-4000:]
        smoke.report["ok"] = False
        code = 2
    _write_outputs(smoke.report, args.json_out)
    lingering = [t.name for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon and t.is_alive()]
    if lingering:
        # Backend threads must not keep the smoke alive (built apps hard-exit too).
        sys.stderr.write(f"host_smoke: exiting with non-daemon threads still running: {lingering}\n")
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
