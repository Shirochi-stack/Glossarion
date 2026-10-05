"""Sandbox, deterministic process state and env/state capture for the parity oracle.

``CaptureContext`` (one per scenario x entry) provides:

* a fresh sandbox (fake app/script dir, cwd, tmp, home, inputs, outputs);
* a scrubbed baseline environment (OS essentials + sandbox HOME/TEMP/APPDATA +
  scenario env) so env deltas never depend on the developer's shell;
* deterministic ``os.cpu_count`` (8), ``os.getpid`` (4242), ``time.time``
  (fixed), ``uuid.uuid4`` (counter) and ``tempfile.tempdir`` (sandbox);
* a recording ``UnifiedClient`` (pool setters + constructor) and recording
  backend entry points (``translation_main``, ``glossary_main``,
  ``fallback_compile_epub``, ``pdf_workspace_compiler.compile_pdf_workspace``)
  that capture the full effective environment (``os.environ`` merged with
  ``large_env._store``), ``sys.argv`` and cwd at call time;
* patch helpers that are undone on exit, and stdout capture.

``normalize_value`` turns any captured value into a deterministic literal:
sandbox/repo/home paths become ``<SANDBOX>``/``<SRC>``/``<REPO>``/``<HOME>``
(raw, forward-slash, JSON-escaped and normcase variants), memory addresses are
masked, sets are sorted, fakes are described and other objects are named.
"""

from __future__ import annotations

import builtins
import collections
import copy
import importlib
import io
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import fakes  # noqa: E402

SANDBOX_TOKEN = "<SANDBOX>"
FIXED_TIME = 1_700_000_000.0
FIXED_PID = 4242
FIXED_CPU_COUNT = 8
#: sys.argv of a desktop source launch (``python translator_gui.py``); the
#: compile runners leave argv untouched, so the harness's own argv must not leak.
BASELINE_ARGV = ("translator_gui.py",)

#: OS variables kept in the scrubbed baseline (everything else is removed).
ESSENTIAL_ENV_KEYS = (
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "PATH", "OS",
    "PROCESSOR_ARCHITECTURE",
)

#: Imported once outside any sandbox so import-time side effects (DPI env,
#: cache sweeps, module-level path constants) are identical for every capture.
PREIMPORT_MODULES = (
    "large_env",
    "api_key_encryption",
    "unified_api_client",
    "glm_proxy",
    "glossary_paths",
    "output_workspace",
    "extract_glossary_from_epub",
    "subtitle_processor",
    "metadata_batch_translator",
    "ollama_settings_dialog",
    "other_settings",  # imports translator_gui (module-level CONFIG_FILE import)
    "epub_library",
    "epub_converter",
    "pdf_workspace_compiler",
)

#: Backend entry points stubbed in the frozen namespace (module globals of translator_gui).
NAMESPACE_BACKEND_STUBS = ("translation_main", "glossary_main", "fallback_compile_epub")
#: Backend entry points stubbed as live module attributes (looked up by local import).
MODULE_BACKEND_STUBS = (("pdf_workspace_compiler", "compile_pdf_workspace"),)

_PREIMPORTED: dict = {}


def sandbox_base_dir() -> Path:
    """Fixed per-machine sandbox base (outside the repo; never random)."""
    return Path(tempfile.gettempdir()).resolve() / "glossarion_parity_sandbox"


_SANDBOX_LOCKS: dict = {}


def hold_sandbox_lock(base) -> None:
    """Hold an exclusive cross-process lock on *base* until this process exits.

    Every capture of a scenario/entry uses the same fixed sandbox path, so two test
    processes capturing at once (e.g. test_shared_core_p1 and test_headless_owner in a
    parallel suite run) would delete each other's sandboxes mid-run. The first capture
    in a process waits for the lock; the OS releases it when the process ends.
    """
    base = Path(base)
    key = os.path.normcase(str(base.resolve()))
    if key in _SANDBOX_LOCKS:
        return
    base.mkdir(parents=True, exist_ok=True)
    handle = open(base / ".capture.lock", "a+b")
    deadline = time.monotonic() + 900  # never wait forever (e.g. a capturing child of a holder)
    while True:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() > deadline:
                print(f"[parity] sandbox lock {base} still busy after 15 min; continuing", file=sys.__stderr__)
                break
            time.sleep(0.5)
    _SANDBOX_LOCKS[key] = handle


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)) or "x"


def preimport_backend_modules() -> dict:
    """Import the backend modules the frozen code reaches (idempotent)."""
    if _PREIMPORTED:
        return _PREIMPORTED
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for name in PREIMPORT_MODULES:
        try:
            importlib.import_module(name)
            _PREIMPORTED[name] = "ok"
        except Exception as exc:  # pragma: no cover - environment dependent
            _PREIMPORTED[name] = f"{type(exc).__name__}: {exc}"
    return _PREIMPORTED


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.app_dir = self.root / "src"  # fake script dir: translator_gui.__file__ lives here
        self.cwd = self.root / "cwd"
        self.tmp = self.root / "tmp"
        self.home = self.root / "home"
        self.appdata = self.root / "appdata"
        self.localappdata = self.root / "localappdata"
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        for d in (self.app_dir, self.cwd, self.tmp, self.home, self.appdata,
                  self.localappdata, self.inputs, self.outputs):
            d.mkdir(parents=True, exist_ok=True)
        self.config_file = self.app_dir / "config.json"

    def src_file(self, name: str) -> Path:
        return self.app_dir / name

    def path(self, rel: str) -> str:
        return str(self.root / rel)

    def resolve(self, obj):
        """Replace the ``<SANDBOX>`` token in every string of *obj* with the sandbox root."""
        root = str(self.root)
        if isinstance(obj, str):
            return obj.replace(SANDBOX_TOKEN, root)
        if isinstance(obj, dict):
            return {self.resolve(k): self.resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.resolve(v) for v in obj)
        return obj


def _path_variants(path: str) -> list:
    out = set()
    for p in {path, os.path.normpath(path)}:
        for v in (p, p.replace("\\", "/"), p.replace("\\", "\\\\"), p.replace("/", "\\")):
            out.add(v)
            out.add(os.path.normcase(v))
            out.add(v.lower())
    return [v for v in out if v]


class PathPlaceholders:
    def __init__(self, mapping: list):
        pairs = []
        for path, token in mapping:
            if not path:
                continue
            for variant in _path_variants(str(path)):
                pairs.append((variant, token))
        # longest first so nested roots (sandbox under temp under home) win
        self.pairs = sorted(set(pairs), key=lambda kv: (-len(kv[0]), kv[0]))

    def apply(self, text: str) -> str:
        for variant, token in self.pairs:
            if variant in text:
                text = text.replace(variant, token)
        return text


_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")


def normalize_value(value, placeholders: PathPlaceholders, _depth=0):
    """Deterministic, literal-only representation of *value*."""
    if _depth > 60:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _ADDR_RE.sub("0x?", placeholders.apply(value))
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Path):
        return normalize_value(str(value), placeholders, _depth + 1)
    if isinstance(value, dict):
        return {
            normalize_value(k, placeholders, _depth + 1): normalize_value(v, placeholders, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize_value(v, placeholders, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_value(v, placeholders, _depth + 1) for v in value)
    if isinstance(value, (set, frozenset)):
        items = [normalize_value(v, placeholders, _depth + 1) for v in value]
        return {"__set__": sorted(items, key=repr)}
    if isinstance(value, collections.deque):
        return {"__deque__": [normalize_value(v, placeholders, _depth + 1) for v in value],
                "maxlen": value.maxlen}
    describe = getattr(value, "describe", None)
    if isinstance(value, fakes.FAKE_WIDGET_TYPES + (fakes.FakeSignal,)) and callable(describe):
        return normalize_value(describe(), placeholders, _depth + 1)
    init_kwargs = getattr(value, "_parity_init_kwargs", None)
    if isinstance(init_kwargs, dict):
        return {"__client__": type(value).__name__,
                "kwargs": normalize_value(init_kwargs, placeholders, _depth + 1)}
    if callable(value):
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or type(value).__name__
        return f"<callable {name}>"
    return f"<object {type(value).__name__}>"


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def effective_env() -> dict:
    """os.environ merged with large_env's in-memory store (Windows >32k values)."""
    env = {(k.upper() if sys.platform == "win32" else k): v for k, v in os.environ.items()}
    try:
        import large_env

        for k, v in large_env._store.items():
            env[k.upper() if sys.platform == "win32" else k] = v
    except Exception:
        pass
    return env


def large_env_store() -> dict:
    try:
        import large_env

        return dict(sorted(large_env._store.items()))
    except Exception:
        return {}


def env_delta(before: dict, after: dict) -> dict:
    changed = {k: after[k] for k in sorted(after) if before.get(k) != after[k]}
    removed = sorted(k for k in before if k not in after)
    return {"set": changed, "removed": removed}


def dict_delta(before: dict, after: dict) -> dict:
    """Top-level key delta of two already-normalized dicts."""
    changed = {k: after[k] for k in after if k not in before or before[k] != after[k]}
    removed = [k for k in before if k not in after]
    return {"set": changed, "removed": removed}


def snapshot_attrs(owner, placeholders) -> dict:
    """Data attributes of *owner* (bound methods from setup_other_settings_methods and
    signals are behaviour, not state, and are left out)."""
    import types

    out = {}
    for name, value in vars(owner).items():
        if name.startswith("_parity_") or name == "config":
            continue
        if isinstance(value, fakes.FakeSignal):
            continue
        if isinstance(value, types.MethodType) and getattr(value, "__self__", None) is owner:
            continue
        out[name] = normalize_value(value, placeholders)
    return out


def snapshot_config(owner, placeholders) -> dict:
    cfg = getattr(owner, "config", None)
    if not isinstance(cfg, dict):
        return {"__missing__": True}
    return normalize_value(cfg, placeholders)


# ---------------------------------------------------------------------------
# Capture context
# ---------------------------------------------------------------------------


class CaptureContext:
    """One scenario x entry run in a fresh sandbox with deterministic process state."""

    def __init__(self, scenario: dict, entry: str, *, record_checkpoints: bool = False,
                 base_dir: str | None = None, backend: bool = True):
        self.scenario = scenario
        self.entry = entry
        self.record_checkpoints = record_checkpoints
        self._base_dir = base_dir
        #: False = no backend preimport / UnifiedClient recorder / backend stubs (toy harness
        #: self-tests in test_parity_tiers.py run without the desktop import closure).
        self.backend = backend
        self.recorder = fakes.CallRecorder()
        self.backend_calls: list = []
        self.checkpoints: list = []
        self._patches: list = []
        self._uuid_counter = 0
        self.backend_stubs = {name: self._make_backend_stub(name) for name in NAMESPACE_BACKEND_STUBS}
        self.sandbox: Sandbox | None = None
        self.placeholders: PathPlaceholders | None = None

    # -- patching -------------------------------------------------------
    def patch(self, obj, attr: str, value):
        missing = object()
        old = getattr(obj, attr, missing)
        self._patches.append(("attr", obj, attr, old, missing))
        setattr(obj, attr, value)

    def patch_dict(self, mapping: dict, key, value):
        missing = object()
        old = mapping.get(key, missing)
        self._patches.append(("dict", mapping, key, old, missing))
        mapping[key] = value

    def patch_module_attr(self, module_name: str, attr: str, value):
        module = importlib.import_module(module_name)
        self.patch(module, attr, value)

    def _undo_patches(self):
        while self._patches:
            kind, obj, key, old, missing = self._patches.pop()
            if kind == "attr":
                if old is missing:
                    try:
                        delattr(obj, key)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, key, old)
            else:
                if old is missing:
                    obj.pop(key, None)
                else:
                    obj[key] = old

    # -- deterministic process state -------------------------------------
    def _fixed_uuid4(self):
        self._uuid_counter += 1
        return uuid.UUID(int=0x5EED0000000000000000000000000000 + self._uuid_counter)

    def _baseline_env(self) -> dict:
        sb = self.sandbox
        env = {k: os.environ[k] for k in ESSENTIAL_ENV_KEYS if k in os.environ}
        env.update({
            "TEMP": str(sb.tmp),
            "TMP": str(sb.tmp),
            "TMPDIR": str(sb.tmp),
            "HOME": str(sb.home),
            "USERPROFILE": str(sb.home),
            "APPDATA": str(sb.appdata),
            "LOCALAPPDATA": str(sb.localappdata),
            "PYTHONIOENCODING": "utf-8",
            "QT_QPA_PLATFORM": "offscreen",
        })
        env.update({k: str(v) for k, v in sb.resolve(self.scenario.get("env") or {}).items()})
        return env

    # -- backend stubs ---------------------------------------------------
    def _make_backend_stub(self, name: str):
        def stub(*args, **kwargs):
            self.backend_calls.append({
                "entry": name,
                "args": list(args),
                "kwargs": dict(sorted(kwargs.items())),
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "env": env_delta(self.baseline_env, effective_env()),
                "large_env": large_env_store(),
            })
            return None

        stub.__name__ = f"stub_{name}"
        return stub

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        if self.backend:
            preimport_backend_modules()
        # Deterministic sandbox location: some desktop helpers hash absolute
        # paths (e.g. subtitle work dirs), so the root must not be random.
        base = Path(self._base_dir) if self._base_dir else sandbox_base_dir()
        hold_sandbox_lock(base)
        root = (base / _safe_name(self.scenario.get("name", "scenario")) / _safe_name(self.entry)).resolve()
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self.sandbox = Sandbox(root / "sb")
        self.placeholders = PathPlaceholders([
            (str(self.sandbox.root), SANDBOX_TOKEN),
            (str(SRC_DIR), "<SRC>"),
            (str(REPO_ROOT), "<REPO>"),
            (os.path.expanduser("~"), "<HOME>"),
        ])
        self._saved_env = dict(os.environ)
        self._saved_cwd = os.getcwd()
        self._saved_argv = list(sys.argv)
        try:
            import large_env

            large_env.clear_store()
        except Exception:
            pass
        self._write_scenario_files()
        os.environ.clear()
        os.environ.update(self._baseline_env())
        self.baseline_env = effective_env()
        os.chdir(self.sandbox.cwd)
        sys.argv = list(BASELINE_ARGV)

        self.patch(os, "cpu_count", lambda: FIXED_CPU_COUNT)
        self.patch(os, "getpid", lambda: FIXED_PID)
        self.patch(time, "time", lambda: FIXED_TIME)
        self.patch(uuid, "uuid4", self._fixed_uuid4)
        self.patch(tempfile, "tempdir", str(self.sandbox.tmp))
        if self.backend:
            self.patch_module_attr(
                "unified_api_client", "UnifiedClient", fakes.make_recording_unified_client(self.recorder)
            )
            for module_name, attr in MODULE_BACKEND_STUBS:
                self.patch_module_attr(module_name, attr, self._make_backend_stub(f"{module_name}.{attr}"))
        self._stdout = io.StringIO()
        self.patch(sys, "stdout", self._stdout)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._undo_patches()
        os.chdir(self._saved_cwd)
        os.environ.clear()
        os.environ.update(self._saved_env)
        sys.argv = self._saved_argv
        try:
            import large_env

            large_env.clear_store()
        except Exception:
            pass
        shutil.rmtree(self._root, ignore_errors=True)
        return False

    def _write_scenario_files(self):
        sb = self.sandbox
        for rel, content in (self.scenario.get("files") or {}).items():
            path = sb.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            content = sb.resolve(content)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(str(content), encoding="utf-8")
        config = self.scenario.get("config")
        if config is not None:
            import json

            sb.config_file.write_text(
                json.dumps(sb.resolve(copy.deepcopy(config)), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # -- capture helpers -------------------------------------------------------
    def norm(self, value):
        return normalize_value(value, self.placeholders)

    def take_stdout(self) -> list:
        text = self._stdout.getvalue()
        self._stdout.seek(0)
        self._stdout.truncate()
        return [self.norm(line) for line in text.splitlines()]

    def state(self, owner) -> dict:
        return {
            "env": effective_env(),
            "config": snapshot_config(owner, self.placeholders),
            "attrs": snapshot_attrs(owner, self.placeholders),
        }

    def checkpoint(self, phase: str, owner, result=None):
        if not self.record_checkpoints:
            return
        current = self.state(owner)
        previous = self.checkpoints[-1]["_state"] if self.checkpoints else {
            "env": self.baseline_env, "config": {}, "attrs": {},
        }
        self.checkpoints.append({
            "phase": phase,
            "result": self.norm(result),
            "env": self.norm(env_delta(previous["env"], current["env"])),
            "config": dict_delta(previous["config"], current["config"]),
            "attrs": dict_delta(previous["attrs"], current["attrs"]),
            "calls": self.norm(list(self.recorder.calls)),
            "stdout": self.take_stdout(),
            "_state": current,
        })
        self.recorder.clear()


__all__ = [
    "CaptureContext",
    "PathPlaceholders",
    "Sandbox",
    "SANDBOX_TOKEN",
    "effective_env",
    "env_delta",
    "dict_delta",
    "normalize_value",
    "preimport_backend_modules",
    "snapshot_attrs",
    "snapshot_config",
]

del builtins
