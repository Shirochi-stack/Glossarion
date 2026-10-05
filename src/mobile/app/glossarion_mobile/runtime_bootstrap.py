"""Process bootstrap for the Glossarion mobile app.

``bootstrap()`` runs first thing in ``app/main.py``: before Flet starts and before
any backend module (TransateKRtoEN, unified_api_client, ...) is imported. It

* sizes new thread stacks (16 MiB; Android/iOS secondary threads default to
  roughly 1 MiB / 512 KiB, too small for deep bs4/lxml/json recursion),
* resolves the app storage directories (``AppPaths``) from the
  ``FLET_APP_STORAGE_*`` variables that the Flet runtime sets,
* applies the backend environment contract (``AppPaths.env_contract()``),
* chdirs into the data dir and puts the backend on ``sys.path``,
* configures logging (``<data>/logs/run.log``, 2 MB x 5), faulthandler
  (``crash.log``) and the sys/threading excepthooks,
* seeds the tiktoken cache from ``app/assets/tiktoken`` so the first run works
  offline,
* installs a preferred ``webbrowser`` controller whose ``open(url)`` is handed to
  a callback registered by the UI (UrlLauncher IN_APP_BROWSER_VIEW), so the
  unchanged desktop OAuth loopback flows work on a phone,
* prints the ``GLOSSARION_BOOT`` device-log marker.

This module only uses the standard library (plus ``certifi`` when present) and
never imports backend modules at import time; ``warm_import()`` /
``start_warm_import()`` import them later, off the UI loop.

Device log markers go to the *current* ``sys.stderr``. On a device that is the
native-log writer installed by Flet's runtime (``dart_bridge`` +
``_TeeWriter``): logcat tag ``flet.python`` on Android, os_log on iOS, plus
``<cache>/console.log``. ``sys.__stderr__`` is the raw fd 2 there, which is
``/dev/null`` on Android, so it is only used as a fallback.

Built apps hard-exit (no ``atexit``): every handler here flushes per record.
"""

from __future__ import annotations

import faulthandler
import hashlib
import importlib
import json
import logging
import os
import platform as _platform
import re
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = [
    "AppPaths",
    "BootState",
    "InAppBrowser",
    "THREAD_STACK_SIZE",
    "WARM_IMPORT_MODULES",
    "TIKTOKEN_ENCODINGS",
    "bootstrap",
    "reset",
    "get_state",
    "get_paths",
    "detect_platform",
    "resolve_backend_dir",
    "resolve_paths",
    "seed_tiktoken_cache",
    "emit_marker",
    "set_url_opener",
    "warm_import",
    "start_warm_import",
    "backend_ready",
    "app_version",
    "current_thread_stack_size",
    "flush_logs",
]

log = logging.getLogger("glossarion.bootstrap")

# --------------------------------------------------------------------------
# Constants (shared contracts; keep in sync with CI and the self-test)
# --------------------------------------------------------------------------

THREAD_STACK_SIZE = 16 * 1024 * 1024
RECURSION_LIMIT = 5000

OAUTH_RETURN_URL = "glossarion://app/oauth/return"

MARKER_BOOT = "GLOSSARION_BOOT"
MARKER_READY = "GLOSSARION_READY"
MARKER_BACKEND_READY = "GLOSSARION_BACKEND_READY"
MARKER_BACKEND_FAIL = "GLOSSARION_BACKEND_FAIL"
MARKER_SELFTEST = "GLOSSARION_SELFTEST"

# Imported off the UI loop once the page is up (heavy: ~75 modules).
WARM_IMPORT_MODULES = (
    "TransateKRtoEN",
    "unified_api_client",
    "extract_glossary_from_epub",
    "epub_converter",
    "scan_html_folder",
    "qa_scan_runtime",
    # U3: the job owner (pipelines) and the Direct Text chat store + stream the chat calls
    "headless_owner",
    "direct_text_stream",
)

# tiktoken encodings shipped in app/assets/tiktoken (cache files are named
# sha1(blob url), exactly as tiktoken.load.read_file_cached expects).
TIKTOKEN_ENCODINGS = {
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
    "o200k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    ),
}
TIKTOKEN_MANIFEST = "MANIFEST.toml"

# Device log lines are truncated by os_log at ~1 KiB; keep marker payloads short.
MARKER_MAX_CHARS = 900

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


_STACK_LOCK = threading.Lock()


def current_thread_stack_size() -> int:
    """The stack size for new threads, WITHOUT changing it.

    ``threading.stack_size()`` with no argument is ``stack_size(0)``: it resets
    the size to the platform default and returns the old value. Never call it
    bare; use this helper (read + immediate restore).
    """
    with _STACK_LOCK:
        size = threading.stack_size(0)
        if size:
            threading.stack_size(size)
        return size


def tiktoken_cache_key(blob_url: str) -> str:
    """File name tiktoken uses for ``blob_url`` inside ``TIKTOKEN_CACHE_DIR``."""
    return hashlib.sha1(blob_url.encode()).hexdigest()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppPaths:
    """Every directory and file location the mobile runtime uses."""

    platform: str  # "android" | "ios" | "desktop"
    app_dir: Path  # Flet app dir (contains main.py); read-only on device
    assets_dir: Path
    data: Path  # FLET_APP_STORAGE_DATA; process cwd; survives app updates
    cache: Path  # FLET_APP_STORAGE_CACHE; may be purged by the OS
    temp: Path  # FLET_APP_STORAGE_TEMP; may vanish between launches
    docs: Path  # iOS: <original HOME>/Documents/Glossarion (Files-visible); else data
    home: Path  # HOME override: <data>/home
    logs: Path
    output: Path  # OUTPUT_DIRECTORY
    library: Path  # GLOSSARION_LIBRARY_DIR
    config_file: Path
    model_catalog_cache: Path
    tiktoken_cache: Path
    original_home: Optional[str]
    backend_dir: Optional[Path]
    backend_source: str  # "env" | "repo" | "bundle" | "env-missing" | "missing"
    dev_storage: bool = False  # FLET_APP_STORAGE_* were unset; dev fallbacks used

    @property
    def is_device(self) -> bool:
        return self.platform in ("android", "ios")

    def writable_dirs(self) -> dict[str, Path]:
        return {
            "data": self.data,
            "cache": self.cache,
            "temp": self.temp,
            "docs": self.docs,
            "home": self.home,
            "logs": self.logs,
            "output": self.output,
            "library": self.library,
            "tiktoken_cache": self.tiktoken_cache,
        }

    def env_contract(self, ca_bundle: Optional[str] = None) -> dict[str, str]:
        """The exact environment the backend must see (set before any backend import)."""
        env = {
            "GLOSSARION_MOBILE": "1",
            "GLOSSARION_NO_PROCESSES": "1",
            "GLOSSARION_HEADLESS_KEY_MANAGER": "1",
            "GLOSSARION_APP_DIR": str(self.data),
            "GLOSSARION_DATA_DIR": str(self.data),
            "CONFIG_FILE": str(self.config_file),
            "HOME": str(self.home),
            "OUTPUT_DIRECTORY": str(self.output),
            "GLOSSARION_LIBRARY_DIR": str(self.library),
            "GLOSSARION_LOG_DIR": str(self.logs),
            "GLOSSARION_MODEL_CATALOG_CACHE": str(self.model_catalog_cache),
            "XDG_CACHE_HOME": str(self.cache),
            "TMPDIR": str(self.temp),
            "TEMP": str(self.temp),
            "TMP": str(self.temp),
            "TIKTOKEN_CACHE_DIR": str(self.tiktoken_cache),
            "GLOSSARION_HTTP_LOG": "0",
            "GLOSSARION_OAUTH_RETURN_URL": OAUTH_RETURN_URL,
            "USE_ASYNC_CHAPTER_EXTRACTION": "0",
            "PDF_EXTRACTION_WORKERS": "1",
            "QA_USE_THREAD_EXECUTOR": "1",
            # Bookkeeping (not read by the backend):
            "GLOSSARION_PLATFORM": self.platform,
        }
        if ca_bundle:
            env["SSL_CERT_FILE"] = ca_bundle
            env["REQUESTS_CA_BUNDLE"] = ca_bundle
        return env


def _abspath(value: str | os.PathLike[str]) -> Path:
    # abspath, not resolve(): keep the platform's own spelling of app-storage
    # paths (Android /data/user/0 is a symlink to /data/data).
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def detect_platform() -> str:
    """``"android"``, ``"ios"`` or ``"desktop"``.

    ``GLOSSARION_SIMULATE_PLATFORM`` (android|ios) lets host smoke tests run the
    device code paths on a desktop Python.
    """
    simulated = os.environ.get("GLOSSARION_SIMULATE_PLATFORM", "").strip().lower()
    if simulated in ("android", "ios", "desktop"):
        return simulated
    flet_platform = os.environ.get("FLET_PLATFORM", "").strip().lower()
    if sys.platform == "android" or flet_platform == "android":
        return "android"
    if sys.platform == "ios" or flet_platform == "ios":
        return "ios"
    if "ANDROID_DATA" in os.environ and "ANDROID_ROOT" in os.environ:
        return "android"
    return "desktop"


def resolve_backend_dir(app_dir: Path) -> tuple[Optional[Path], str]:
    """Locate the backend modules.

    Order: ``GLOSSARION_BACKEND_DIR`` -> the repo ``src/`` (dev: when
    ``Path(main.py).resolve().parents[2] / "TransateKRtoEN.py"`` exists) ->
    the collected bundle ``<app>/backend``.
    """
    env_dir = os.environ.get("GLOSSARION_BACKEND_DIR", "").strip()
    if env_dir:
        candidate = _abspath(env_dir)
        if candidate.is_dir():
            return candidate, "env"
        # An explicit override that does not exist is an error, never a silent
        # fallback (host smoke uses it to force the collected bundle).
        return None, "env-missing"
    try:
        repo_src = (Path(app_dir) / "main.py").resolve().parents[2]
    except IndexError:
        repo_src = None
    if repo_src is not None and (repo_src / "TransateKRtoEN.py").is_file():
        return repo_src, "repo"
    bundled = Path(app_dir) / "backend"
    if bundled.is_dir():
        return bundled, "bundle"
    return None, "missing"


def resolve_paths(app_dir: Path, platform: Optional[str] = None) -> AppPaths:
    """Compute ``AppPaths`` from the environment without touching the filesystem."""
    app_dir = _abspath(app_dir)
    platform = platform or detect_platform()
    dev_root = app_dir.parent / "storage"  # src/mobile/storage when FLET_APP_STORAGE_* are unset

    dev_storage = False

    def storage(var: str, name: str) -> Path:
        nonlocal dev_storage
        value = os.environ.get(var, "").strip()
        if value:
            return _abspath(value)
        dev_storage = True
        return _abspath(dev_root / name)

    data = storage("FLET_APP_STORAGE_DATA", "data")
    cache = storage("FLET_APP_STORAGE_CACHE", "cache")
    temp = storage("FLET_APP_STORAGE_TEMP", "temp")

    # The original HOME is captured once per process (bootstrap overrides HOME).
    original_home = os.environ.get("GLOSSARION_ORIGINAL_HOME") or os.environ.get("HOME") or None
    docs = data
    if platform == "ios" and original_home:
        documents = Path(original_home) / "Documents"
        if documents.is_dir():
            docs = _abspath(documents / "Glossarion")

    assets_env = os.environ.get("FLET_ASSETS_DIR", "").strip()
    assets_dir = _abspath(assets_env) if assets_env else app_dir / "assets"
    backend_dir, backend_source = resolve_backend_dir(app_dir)

    return AppPaths(
        platform=platform,
        app_dir=app_dir,
        assets_dir=assets_dir,
        data=data,
        cache=cache,
        temp=temp,
        docs=docs,
        home=data / "home",
        logs=data / "logs",
        output=docs / "Output",
        library=docs / "Library",
        config_file=data / "config.json",
        model_catalog_cache=cache / "model_catalog_cache.json",
        tiktoken_cache=cache / "tiktoken",
        original_home=original_home,
        backend_dir=backend_dir,
        backend_source=backend_source,
        dev_storage=dev_storage,
    )


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


def _marker_stream():
    stream = sys.stderr
    if stream is None or getattr(stream, "closed", False):
        stream = sys.__stderr__
    return stream


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, default=str)


def emit_marker(name: str, payload: Any = None) -> str:
    """Print a device-log marker line and mirror it into run.log. Returns the line."""
    if payload is None:
        line = name
    elif isinstance(payload, str):
        line = f"{name} {payload}"
    else:
        line = f"{name} {_compact_json(payload)}"
    stream = _marker_stream()
    try:
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()
    except Exception:
        try:
            if sys.__stderr__ is not None and stream is not sys.__stderr__:
                sys.__stderr__.write(line + "\n")
                sys.__stderr__.flush()
        except Exception:
            pass
    # Mirror into run.log only (not through the root logger, which may also
    # have a stderr handler and would print the marker twice).
    state = _STATE
    handler = state.hooks.handler if state is not None else None
    if handler is not None:
        try:
            handler.handle(
                logging.LogRecord("glossarion.markers", logging.INFO, __file__, 0, line, None, None)
            )
        except Exception:
            pass
    return line


# --------------------------------------------------------------------------
# tiktoken cache seeding
# --------------------------------------------------------------------------


def _load_toml(path: Path) -> Any:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - device/host are 3.12+/3.13
        return None
    with open(path, "rb") as fh:
        return tomllib.load(fh)


_NAME_KEYS = ("cache_key", "cache_file", "file", "filename", "path", "name", "sha1")
_HASH_KEYS = ("sha256", "expected_hash", "hash", "expected_sha256")


def _manifest_entries(manifest: Any, available: set[str]) -> dict[str, Optional[str]]:
    """Map cache file name -> expected sha256 from a MANIFEST.toml of any reasonable shape.

    Accepts ``[[files]] cache_key=.. sha256=..`` tables, ``[encodings.<name>]``
    tables, and flat ``"<file>" = "<sha256>"`` mappings.
    """
    entries: dict[str, Optional[str]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            name = None
            for key in _NAME_KEYS:
                value = node.get(key)
                if isinstance(value, str) and Path(value).name in available:
                    name = Path(value).name
                    break
            if name is not None:
                digest = None
                for key in _HASH_KEYS:
                    value = node.get(key)
                    if isinstance(value, str) and _HEX64.match(value):
                        digest = value.lower()
                        break
                entries.setdefault(name, digest)
            for key, value in node.items():
                if isinstance(key, str) and key in available and isinstance(value, str) and _HEX64.match(value):
                    entries[key] = value.lower()
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(manifest)
    return entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_tiktoken_cache(src: Path, dst: Path) -> dict[str, Any]:
    """Copy the bundled tiktoken BPE files into ``TIKTOKEN_CACHE_DIR``.

    Files already present with the same size are left alone (tiktoken itself
    verifies the hash when reading). Copied files are checked against the
    MANIFEST.toml sha256 (or the well-known tiktoken hashes) before an atomic
    rename, so a bad asset never shadows a good cache entry.
    """
    report: dict[str, Any] = {
        "copied": 0,
        "present": 0,
        "missing": [],
        "bad_hash": [],
        "manifest": False,
        "errors": [],
    }
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        report["errors"].append(f"no asset dir {src}")
        return report

    available = {p.name for p in src.iterdir() if p.is_file() and p.name != TIKTOKEN_MANIFEST}
    known = {tiktoken_cache_key(url): digest for url, digest in TIKTOKEN_ENCODINGS.values()}
    expected: dict[str, Optional[str]] = {}
    manifest_path = src / TIKTOKEN_MANIFEST
    if manifest_path.is_file():
        try:
            expected = _manifest_entries(_load_toml(manifest_path), available)
            report["manifest"] = True
        except Exception as exc:  # malformed manifest: fall back to the files themselves
            report["errors"].append(f"manifest: {type(exc).__name__}: {exc}")
    names = sorted(expected) if expected else sorted(available)
    for name in known:
        if name not in available and name not in names:
            report["missing"].append(name)

    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = src / name
        target = dst / name
        if not source.is_file():
            report["missing"].append(name)
            continue
        want = expected.get(name) or known.get(name)
        try:
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                report["present"] += 1
                continue
            if want and _sha256_file(source) != want:
                report["bad_hash"].append(name)
                continue
            tmp = target.with_name(f"{name}.{os.getpid()}.tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
            report["copied"] += 1
        except OSError as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------
# webbrowser -> in-app browser (UrlLauncher IN_APP_BROWSER_VIEW)
# --------------------------------------------------------------------------


class InAppBrowser(webbrowser.BaseBrowser):
    """Preferred ``webbrowser`` controller for the phone.

    ``open(url)`` hands the URL to the opener registered with
    ``set_url_opener()`` (the UI marshals it to
    ``UrlLauncher.launch_url(url, mode=LaunchMode.IN_APP_BROWSER_VIEW)`` on the
    Flet loop). URLs opened before the UI registers an opener are queued and
    delivered on registration. ``open()`` is safe to call from any thread.
    """

    NAME = "glossarion-inapp"

    def __init__(self) -> None:
        super().__init__(self.NAME)
        self._lock = threading.Lock()
        self._opener: Optional[Callable[[str], Any]] = None
        self._pending: list[str] = []
        self.opened: list[str] = []  # history for diagnostics / the spike screen

    def set_opener(self, opener: Optional[Callable[[str], Any]]) -> None:
        with self._lock:
            self._opener = opener
            pending, self._pending = self._pending, []
        if opener is not None:
            for url in pending:
                self._deliver(opener, url)

    @property
    def pending(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def _deliver(self, opener: Callable[[str], Any], url: str) -> bool:
        try:
            opener(url)
            return True
        except Exception:
            log.exception("in-app browser opener failed for %s", url)
            return False

    def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:  # noqa: A003
        sys.audit("webbrowser.open", url)
        with self._lock:
            self.opened.append(url)
            del self.opened[:-50]
            opener = self._opener
            if opener is None:
                self._pending.append(url)
                log.info("webbrowser.open queued until the UI is ready: %s", url)
                return True
        return self._deliver(opener, url)


IN_APP_BROWSER = InAppBrowser()


def _install_webbrowser() -> None:
    webbrowser.register(InAppBrowser.NAME, None, IN_APP_BROWSER, preferred=True)


def _uninstall_webbrowser() -> None:
    try:
        with webbrowser._lock:  # type: ignore[attr-defined]
            webbrowser._browsers.pop(InAppBrowser.NAME, None)  # type: ignore[attr-defined]
            order = webbrowser._tryorder  # type: ignore[attr-defined]
            while order is not None and InAppBrowser.NAME in order:
                order.remove(InAppBrowser.NAME)
    except Exception:
        pass


def set_url_opener(opener: Optional[Callable[[str], Any]]) -> None:
    """Register the callable that actually opens URLs (called from any thread)."""
    IN_APP_BROWSER.set_opener(opener)


# --------------------------------------------------------------------------
# Logging, faulthandler, excepthooks
# --------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"


@dataclass
class _Hooks:
    handler: Optional[logging.Handler] = None
    crash_file: Any = None
    prev_sys_hook: Any = None
    prev_thread_hook: Any = None
    prev_root_level: Optional[int] = None
    prev_stack_size: Optional[int] = None
    prev_recursion_limit: Optional[int] = None
    prev_faulthandler: bool = False


def _check_previous_crash(crash_log: Path) -> bool:
    """True when the last session wrote something after its boot header."""
    try:
        size = crash_log.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    try:
        with open(crash_log, "rb") as fh:
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return False
    idx = tail.rfind("=== boot ")
    rest = tail[idx:] if idx >= 0 else tail
    lines = [ln for ln in rest.splitlines()[1:] if ln.strip()]
    return bool(lines)


def _configure_logging(paths: AppPaths, hooks: _Hooks) -> bool:
    """Install run.log + crash.log + excepthooks. Returns previous-crash flag."""
    root = logging.getLogger()
    handler = RotatingFileHandler(
        paths.logs / "run.log", maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    hooks.handler = handler
    hooks.prev_root_level = root.level
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    crash_log = paths.logs / "crash.log"
    previous_crash = _check_previous_crash(crash_log)
    crash_file = open(crash_log, "a", encoding="utf-8")
    crash_file.write(f"=== boot {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===\n")
    crash_file.flush()
    hooks.crash_file = crash_file
    hooks.prev_faulthandler = faulthandler.is_enabled()
    try:
        faulthandler.enable(file=crash_file, all_threads=True)
    except Exception as exc:  # pragma: no cover - fileno-less platforms
        log.warning("faulthandler unavailable: %s", exc)

    hooks.prev_sys_hook = sys.excepthook
    hooks.prev_thread_hook = threading.excepthook

    def sys_hook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("glossarion.crash").critical(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )
            _flush_log_handlers()
        prev = hooks.prev_sys_hook or sys.__excepthook__
        prev(exc_type, exc, tb)

    def thread_hook(args):
        if args.exc_type is not SystemExit:
            name = args.thread.name if args.thread is not None else "?"
            logging.getLogger("glossarion.crash").critical(
                "Uncaught exception in thread %s",
                name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _flush_log_handlers()
        prev = hooks.prev_thread_hook or threading.__excepthook__
        prev(args)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook
    return previous_crash


def _flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def flush_logs() -> None:
    """Flush run.log/crash.log (call on lifecycle INACTIVE/HIDE/PAUSE/DETACH)."""
    _flush_log_handlers()
    state = _STATE
    if state is not None and state.hooks.crash_file is not None:
        try:
            state.hooks.crash_file.flush()
        except Exception:
            pass


def _undo_hooks(hooks: _Hooks) -> None:
    root = logging.getLogger()
    if hooks.handler is not None:
        root.removeHandler(hooks.handler)
        try:
            hooks.handler.close()
        except Exception:
            pass
        hooks.handler = None
    if hooks.prev_root_level is not None:
        root.setLevel(hooks.prev_root_level)
        hooks.prev_root_level = None
    if hooks.crash_file is not None:
        try:
            faulthandler.disable()
            if hooks.prev_faulthandler and sys.__stderr__ is not None:
                faulthandler.enable(file=sys.__stderr__, all_threads=True)
        except Exception:
            pass
        try:
            hooks.crash_file.close()
        except Exception:
            pass
        hooks.crash_file = None
    if hooks.prev_sys_hook is not None:
        sys.excepthook = hooks.prev_sys_hook
        hooks.prev_sys_hook = None
    if hooks.prev_thread_hook is not None:
        threading.excepthook = hooks.prev_thread_hook
        hooks.prev_thread_hook = None
    if hooks.prev_stack_size is not None:
        try:
            threading.stack_size(hooks.prev_stack_size)
        except Exception:
            pass
        hooks.prev_stack_size = None
    if hooks.prev_recursion_limit is not None:
        sys.setrecursionlimit(hooks.prev_recursion_limit)
        hooks.prev_recursion_limit = None


# --------------------------------------------------------------------------
# Version info (read from source text; never imports the backend)
# --------------------------------------------------------------------------


def _read_assignment(path: Path, name: str) -> Optional[str]:
    """String constant ``name`` from a constants-only module, without importing it.

    Built apps ship only legacy ``.pyc`` files (``flet build`` compiles the app
    dir with ``compileall -b`` and deletes the ``.py`` sources), so fall back to
    executing the sibling ``.pyc`` in a throwaway module object that is never
    registered in ``sys.modules``.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        text = None
    if text is not None:
        match = re.search(rf"^{name}\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
        return match.group(1) if match else None
    compiled = path.with_suffix(".pyc")
    if not compiled.is_file():
        return None
    try:
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.SourcelessFileLoader(f"_glossarion_const_{path.stem}", str(compiled))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except Exception:
        return None
    value = getattr(module, name, None)
    return value if isinstance(value, str) else None


def app_version(backend_dir: Optional[Path]) -> dict[str, Any]:
    """``{"version": "9.13.6", "build": 9130600, "bundle": "<sha prefix>"|None}``."""
    version = None
    bundle = None
    if backend_dir is not None:
        version = _read_assignment(backend_dir / "app_version.py", "APP_VERSION")
        info = backend_dir / "_bundle_info.py"
        if info.is_file() or info.with_suffix(".pyc").is_file():
            version = version or _read_assignment(info, "BUILD_VERSION")
            sha = _read_assignment(info, "BUNDLE_SHA256")
            bundle = sha[:12] if sha else None
    build = None
    if version:
        parts = re.findall(r"\d+", version)[:3]
        if len(parts) == 3:
            major, minor, patch = (int(p) for p in parts)
            build = major * 1_000_000 + minor * 10_000 + patch * 100
    return {"version": version, "build": build, "bundle": bundle}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


@dataclass
class BootState:
    paths: AppPaths
    started_at: float
    secs: float
    tiktoken: dict[str, Any]
    previous_crash: bool
    errors: list[str]
    version: dict[str, Any]
    hooks: _Hooks = field(default_factory=_Hooks)
    backend_ready: threading.Event = field(default_factory=threading.Event)
    backend_result: Optional[dict[str, Any]] = None


_LOCK = threading.RLock()
_STATE: Optional[BootState] = None
# Pre-bootstrap process state, captured once, restored by reset(restore_env=True).
_ORIGINAL: Optional[dict[str, Any]] = None


def get_state() -> Optional[BootState]:
    return _STATE


def get_paths() -> Optional[AppPaths]:
    return _STATE.paths if _STATE is not None else None


def _default_app_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _absolutize_argv0() -> None:
    # flet resolves a relative assets_dir against dirname(sys.argv[0]); make that
    # independent of the os.chdir() below.
    try:
        if sys.argv and sys.argv[0] and not os.path.isabs(sys.argv[0]) and os.path.exists(sys.argv[0]):
            sys.argv[0] = os.path.abspath(sys.argv[0])
    except Exception:
        pass


def _certifi_where() -> Optional[str]:
    try:
        import certifi

        path = certifi.where()
        return path if path and os.path.isfile(path) else None
    except Exception:
        return None


def bootstrap(
    *,
    app_dir: Optional[str | os.PathLike[str]] = None,
    force: bool = False,
    configure_logging: bool = True,
) -> AppPaths:
    """Prepare the process for the backend. Idempotent unless ``force=True``.

    Safe to call again on Android process reuse (the Dart VM restarts while
    Python keeps running): the second call returns the existing paths.
    """
    global _STATE, _ORIGINAL
    with _LOCK:
        if _STATE is not None and not force:
            return _STATE.paths
        if _STATE is not None:
            _undo_hooks(_STATE.hooks)
            _STATE = None

        t0 = time.monotonic()
        if _ORIGINAL is None:
            _ORIGINAL = {
                "environ": dict(os.environ),
                "cwd": os.getcwd(),
                "sys_path": list(sys.path),
                "tempdir": tempfile.tempdir,
            }
        errors: list[str] = []
        hooks = _Hooks()

        _absolutize_argv0()
        try:
            with _STACK_LOCK:
                hooks.prev_stack_size = threading.stack_size(THREAD_STACK_SIZE)
        except (ValueError, RuntimeError) as exc:
            errors.append(f"stack_size: {exc}")
        hooks.prev_recursion_limit = sys.getrecursionlimit()
        if hooks.prev_recursion_limit < RECURSION_LIMIT:
            sys.setrecursionlimit(RECURSION_LIMIT)

        os.environ.setdefault("GLOSSARION_ORIGINAL_HOME", os.environ.get("HOME", ""))
        if not os.environ["GLOSSARION_ORIGINAL_HOME"]:
            del os.environ["GLOSSARION_ORIGINAL_HOME"]

        paths = resolve_paths(Path(app_dir) if app_dir is not None else _default_app_dir())

        for name, directory in paths.writable_dirs().items():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors.append(f"mkdir {name}: {exc}")

        previous_crash = False
        if configure_logging:
            try:
                previous_crash = _configure_logging(paths, hooks)
            except Exception as exc:
                errors.append(f"logging: {type(exc).__name__}: {exc}")

        ca_bundle = _certifi_where()
        if ca_bundle is None:
            errors.append("certifi unavailable: SSL_CERT_FILE/REQUESTS_CA_BUNDLE not set")
        os.environ.update(paths.env_contract(ca_bundle))
        tempfile.tempdir = None  # recompute from TMPDIR/TEMP/TMP

        try:
            os.chdir(paths.data)
        except OSError as exc:
            errors.append(f"chdir: {exc}")

        if paths.backend_dir is not None:
            backend = str(paths.backend_dir)
            while backend in sys.path:
                sys.path.remove(backend)
            sys.path.insert(0, backend)
        else:
            errors.append(f"backend missing ({paths.backend_source})")

        try:
            tiktoken_report = seed_tiktoken_cache(paths.assets_dir / "tiktoken", paths.tiktoken_cache)
        except Exception as exc:
            tiktoken_report = {"errors": [f"{type(exc).__name__}: {exc}"]}

        _install_webbrowser()
        _prime_platform_processor()

        state = BootState(
            paths=paths,
            started_at=time.time(),
            secs=round(time.monotonic() - t0, 3),
            tiktoken=tiktoken_report,
            previous_crash=previous_crash,
            errors=errors,
            version=app_version(paths.backend_dir),
            hooks=hooks,
        )
        _STATE = state

        for message in errors:
            log.warning("bootstrap: %s", message)
        log.info(
            "bootstrap done: platform=%s data=%s docs=%s backend=%s (%s) tiktoken=%s",
            paths.platform,
            paths.data,
            paths.docs,
            paths.backend_dir,
            paths.backend_source,
            tiktoken_report,
        )
        emit_marker(MARKER_BOOT, _boot_payload(state))
        return paths


def _prime_platform_processor() -> None:
    """Pre-fill platform.uname().processor so nothing ever runs ``uname -p``.

    CPython computes ``uname_result.processor`` lazily (a cached_property) and on
    Linux-like systems (Android) does it by spawning ``uname -p``. SDKs reach it
    through ``platform.platform()`` when they build user-agent headers (e.g. the
    openai client on its first request), which would spawn a process on a worker
    thread. The value is cosmetic there, so seed the cache with the same blank
    result ``uname -p`` gives on Android; leave an already computed value alone.
    """
    try:
        import platform as _platform

        result = _platform.uname()
        if "processor" not in getattr(result, "__dict__", {}):
            result.__dict__["processor"] = ""
    except Exception:
        return
    # platform.platform() takes its generic branch on Android and calls
    # architecture(sys.executable), which runs the `file` command through
    # _syscmd_file. That only refines the linkage string, and CPython already
    # returns the default on iOS, so answer with the default on Android too.
    if sys.platform == "android" or os.environ.get("GLOSSARION_PLATFORM") == "android":
        try:
            def _no_file_command(target: Any, default: str = "") -> str:
                return default

            _platform._syscmd_file = _no_file_command
        except Exception:
            pass


def _flet_version() -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version("flet")
    except Exception:
        return None


def _boot_payload(state: BootState) -> dict[str, Any]:
    paths = state.paths
    tk = state.tiktoken or {}
    payload: dict[str, Any] = {
        "version": state.version.get("version"),
        "build": state.version.get("build"),
        "bundle": state.version.get("bundle"),
        "platform": paths.platform,
        "sys_platform": sys.platform,
        "arch": _platform.machine(),
        "python": _platform.python_version(),
        "flet": _flet_version(),
        "backend": paths.backend_source,
        "tiktoken": {"copied": tk.get("copied"), "present": tk.get("present"), "missing": len(tk.get("missing") or [])},
        "previous_crash": state.previous_crash,
        "dev_storage": paths.dev_storage,
        "secs": state.secs,
        "pid": os.getpid(),
        "errors": [e[:120] for e in state.errors][:5],
        "data": str(paths.data),
        "docs": str(paths.docs),
    }
    if len(_compact_json(payload)) > MARKER_MAX_CHARS:
        payload.pop("data", None)
        payload.pop("docs", None)
        payload["errors"] = [e[:60] for e in state.errors][:3]
    return payload


def reset(*, restore_env: bool = True) -> None:
    """Undo ``bootstrap()`` (host tests only): hooks, handlers, webbrowser and,
    with ``restore_env``, ``os.environ``/cwd/``sys.path`` as they were before the
    first ``bootstrap()`` call."""
    global _STATE, _ORIGINAL
    with _LOCK:
        if _STATE is not None:
            _undo_hooks(_STATE.hooks)
            _STATE = None
        _uninstall_webbrowser()
        IN_APP_BROWSER.set_opener(None)
        with IN_APP_BROWSER._lock:
            IN_APP_BROWSER._pending.clear()
            IN_APP_BROWSER.opened.clear()
        if restore_env and _ORIGINAL is not None:
            os.environ.clear()
            os.environ.update(_ORIGINAL["environ"])
            try:
                os.chdir(_ORIGINAL["cwd"])
            except OSError:
                pass
            sys.path[:] = _ORIGINAL["sys_path"]
            tempfile.tempdir = _ORIGINAL["tempdir"]
            _ORIGINAL = None


# --------------------------------------------------------------------------
# Warm import (off the UI loop)
# --------------------------------------------------------------------------


def _backend_module_count(backend_dir: Optional[Path]) -> int:
    if backend_dir is None:
        return 0
    root = os.path.normcase(str(backend_dir))
    count = 0
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and os.path.normcase(os.path.abspath(filename)).startswith(root + os.sep):
            count += 1
    return count


def warm_import(modules: tuple[str, ...] = WARM_IMPORT_MODULES, *, emit: bool = True) -> dict[str, Any]:
    """Import the heavy backend modules; print GLOSSARION_BACKEND_READY (or _FAIL)."""
    t0 = time.monotonic()
    failed: dict[str, str] = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except BaseException as exc:  # SystemExit from a backend import must not kill the app
            if isinstance(exc, KeyboardInterrupt):
                raise
            failed[name] = f"{type(exc).__name__}: {exc}"
            log.exception("warm import of %s failed", name)
    secs = round(time.monotonic() - t0, 2)
    state = _STATE
    backend_dir = state.paths.backend_dir if state is not None else None
    count = _backend_module_count(backend_dir)
    result = {"ok": not failed, "modules": count, "secs": secs, "failed": failed, "requested": list(modules)}
    if emit:
        if failed:
            emit_marker(
                MARKER_BACKEND_FAIL,
                {"secs": secs, "failed": {k: v[:160] for k, v in list(failed.items())[:4]}},
            )
        else:
            emit_marker(MARKER_BACKEND_READY, f"modules={count} secs={secs}")
    if state is not None:
        state.backend_result = result
        state.backend_ready.set()
    return result


def start_warm_import(
    modules: tuple[str, ...] = WARM_IMPORT_MODULES,
    *,
    on_done: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> threading.Thread:
    """Run ``warm_import`` on a daemon thread (16 MiB stack); ``on_done(result)``
    is called on that thread."""

    def run() -> None:
        result = warm_import(modules)
        if on_done is not None:
            try:
                on_done(result)
            except Exception:
                log.exception("warm import callback failed")

    thread = threading.Thread(target=run, name="gl-warm-import", daemon=True)
    thread.start()
    return thread


def backend_ready(timeout: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Wait for the warm import; returns its result or ``None`` on timeout."""
    state = _STATE
    if state is None:
        return None
    if state.backend_ready.wait(timeout):
        return state.backend_result
    return None
