#!/usr/bin/env python3
"""Tier I: shared GUI-free modules import without Qt and without the desktop GUI.

Each module is imported in its own fresh interpreter with ``sys.modules['PySide6']``
(and ``shiboken6``) set to ``None`` - so ``import PySide6...`` raises
``ImportError`` exactly like on a phone - and the run fails when the import
itself fails or when a forbidden module (``translator_gui``, ``dpi_setup`` by
default) ends up in ``sys.modules``. Guarded Qt imports (``try: import PySide6
except ImportError``) are allowed and reported as ``qt_attempts``.

``parse_errors(path)`` additionally checks the source with
``ast.parse(..., feature_version=(3, 10))``; ``check_imports(..., python=...)``
can run the same probe under a real Python 3.10 interpreter (set
``GLOSSARION_PY310`` to its path; see ``python310()``).

CLI (repository root)::

    python tests/parity/import_hygiene.py run_env owner_state headless_owner
    python tests/parity/import_hygiene.py --all [--python C:/path/to/python310.exe]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

DEFAULT_FORBIDDEN = ("translator_gui", "dpi_setup")

#: Runs inside the child interpreter. argv: src_dir, module, forbidden (comma list).
PROBE = r'''
import builtins, importlib, json, sys, time, traceback
src, module, forbidden = sys.argv[1], sys.argv[2], [f for f in sys.argv[3].split(",") if f]
attempts = []
_real_import = builtins.__import__

def _tripwire_import(name, globals=None, locals=None, fromlist=(), level=0):
    # sys.modules['PySide6'] = None short-circuits meta-path finders, so record at __import__
    if not level and name.split(".")[0] in ("PySide6", "shiboken6"):
        frame = sys._getframe(1)
        attempts.append([name, "%s:%s" % (frame.f_code.co_filename, frame.f_lineno)])
    return _real_import(name, globals, locals, fromlist, level)

sys.modules["PySide6"] = None
sys.modules["shiboken6"] = None
builtins.__import__ = _tripwire_import
sys.path.insert(0, src)
before = set(sys.modules)
started = time.perf_counter()
error = None
try:
    importlib.import_module(module)
except BaseException:
    error = traceback.format_exc()[-4000:]
seconds = time.perf_counter() - started
loaded = sorted(n for n, m in sys.modules.items() if m is not None and n not in before)
leaked = [n for n in loaded if n.split(".")[0] in forbidden or n.split(".")[0] in ("PySide6", "shiboken6")]
print("HYGIENE-RESULT " + json.dumps({
    "error": error, "leaked": leaked, "qt_attempts": attempts, "loaded": len(loaded),
    "seconds": round(seconds, 3), "python": sys.version.split()[0],
}))
'''


@dataclass
class HygieneResult:
    module: str
    python: str = ""
    returncode: int | None = None
    error: str | None = None
    leaked: list = field(default_factory=list)
    qt_attempts: list = field(default_factory=list)
    loaded: int = 0
    seconds: float = 0.0
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error and not self.leaked

    def describe(self) -> str:
        if self.ok:
            extra = f", guarded Qt imports: {self.qt_attempts}" if self.qt_attempts else ""
            return f"{self.module}: ok ({self.loaded} modules, {self.seconds:.2f}s, py{self.python}{extra})"
        parts = [f"{self.module}: FAILED (py{self.python}, rc={self.returncode})"]
        if self.leaked:
            parts.append(f"  forbidden modules imported: {self.leaked}")
        if self.qt_attempts:
            parts.append(f"  Qt import attempts: {self.qt_attempts[:5]}")
        if self.error:
            parts.append("  import error:\n" + "\n".join("    " + line for line in self.error.splitlines()[-12:]))
        elif self.stderr:
            parts.append("  stderr: " + self.stderr[-1500:])
        return "\n".join(parts)


def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    return env


def check_imports(modules, *, src_dir=SRC_DIR, python: str | None = None,
                  forbidden=DEFAULT_FORBIDDEN, timeout: float = 240, parallel: bool = True) -> dict:
    """{module: HygieneResult}; one fresh interpreter per module (run concurrently)."""
    python = python or sys.executable
    procs = {}
    results = {}

    def start(module):
        return subprocess.Popen(
            [python, "-c", PROBE, str(src_dir), module, ",".join(forbidden)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_child_env(),
            cwd=str(src_dir), stdin=subprocess.DEVNULL,
        )

    def finish(module, proc):
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return HygieneResult(module, returncode=None, error=f"timed out after {timeout}s",
                                 stderr=err.decode("utf-8", "replace"))
        text = out.decode("utf-8", "replace")
        res = HygieneResult(module, returncode=proc.returncode, stderr=err.decode("utf-8", "replace"))
        marker = [line for line in text.splitlines() if line.startswith("HYGIENE-RESULT ")]
        if not marker:
            res.error = res.error or f"probe produced no result (stdout tail: {text[-800:]!r})"
            return res
        data = json.loads(marker[-1][len("HYGIENE-RESULT "):])
        res.python = data.get("python", "")
        res.error = data.get("error")
        res.leaked = data.get("leaked", [])
        res.qt_attempts = data.get("qt_attempts", [])
        res.loaded = data.get("loaded", 0)
        res.seconds = data.get("seconds", 0.0)
        return res

    modules = list(modules)
    if parallel:
        for module in modules:
            procs[module] = start(module)
        for module in modules:
            results[module] = finish(module, procs[module])
    else:
        for module in modules:
            results[module] = finish(module, start(module))
    return results


def check_import(module: str, **kwargs) -> HygieneResult:
    return check_imports([module], **kwargs)[module]


def parse_errors(path, feature_version=(3, 10)) -> list:
    """SyntaxErrors of *path* under ``ast.parse(feature_version=...)`` (empty list = ok)."""
    path = Path(path)
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=feature_version)
    except SyntaxError as exc:
        return [f"{path.name}:{exc.lineno}: {exc.msg}"]
    return []


def python310() -> str | None:
    """Path of a Python 3.10 interpreter from ``GLOSSARION_PY310`` (None when unset/missing)."""
    path = os.environ.get("GLOSSARION_PY310")
    if path and Path(path).is_file():
        return path
    return None


def existing(modules, src_dir=SRC_DIR) -> list:
    return [m for m in modules if (Path(src_dir) / f"{m}.py").is_file()]


def main(argv=None) -> int:
    sys.path.insert(0, str(_TESTS_DIR))
    from parity import moved_functions

    parser = argparse.ArgumentParser(description="Tier I import hygiene of shared GUI-free modules")
    parser.add_argument("modules", nargs="*")
    parser.add_argument("--all", action="store_true", help="every existing module in SHARED_MODULES")
    parser.add_argument("--python", default=None, help="interpreter for the probe (default: this one)")
    args = parser.parse_args(argv)
    modules = existing(moved_functions.SHARED_MODULES) if args.all else args.modules
    if not modules:
        parser.error("give module names or --all")
    started = time.perf_counter()
    failures = 0
    for module, res in check_imports(modules, python=args.python).items():
        print(res.describe())
        failures += not res.ok
        for problem in parse_errors(SRC_DIR / f"{module}.py"):
            print(f"  py3.10 syntax: {problem}")
            failures += 1
    print(f"{len(modules)} modules in {time.perf_counter() - started:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
