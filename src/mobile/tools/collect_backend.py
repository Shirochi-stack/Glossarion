#!/usr/bin/env python3
"""Collect the Glossarion backend import closure for the mobile app bundle.

Stdlib only (Python 3.11+; CI runs 3.13). Policy lives in
``src/mobile/backend_manifest.toml``; pins come from ``src/mobile/pyproject.toml``.

What it does
------------
1. Parses ``src/*.py`` with :mod:`ast`. Reading is BOM-safe (translator_gui.py starts
   with a UTF-8 BOM) and honours PEP 263 cookies. Each import gets a *kind* and a
   *guarded* flag:

   * kind ``module``: runs at import time (module scope, class bodies).
   * kind ``conditional``: module scope, but under an ``if``, in an ``except`` handler
     or in the ``else`` of a ``try``.
   * kind ``lazy``: inside a function or lambda; runs only when called.
   * kind ``main``: under ``if __name__ == "__main__"``; never runs on import.
   * kind ``type_checking``: under ``if TYPE_CHECKING``; never runs, so it is ignored.
   * ``guarded``: inside a ``try`` whose handlers catch ImportError,
     ModuleNotFoundError, Exception or BaseException (or a bare ``except``), or inside
     ``with suppress(<one of those>)``. Only guards between the import and its
     enclosing function count.

2. **GUI taint.** A module is tainted if it has an unguarded ``module``-kind import
   of a ``[gui].packages`` entry, of a ``[gui].roots`` module, or of another tainted
   local module. Computed as a fixpoint.

3. **Closure walk.** The walk starts from ``[entry].modules``, ``extra_modules``,
   every ``src/`` module imported by ``scan_package`` (the mobile app) and the
   ``[dynamic_imports]`` targets. It follows ``module``, ``conditional`` and ``lazy``
   edges into local modules that are neither excluded nor tainted. An edge into an
   excluded or tainted module, or into a GUI package:

   * fails when it is ``module``-kind and unguarded;
   * passes when it is guarded;
   * otherwise (lazy, conditional or main, and unguarded) must be listed in
     ``[baseline.lazy_gui_edges]`` or ``[baseline.lazy_excluded_edges]``. Each entry
     records a site ``count`` that may not grow (ratchet).

4. **Checks on every bundled module:**

   a. A string literal naming a GUI or excluded module file (``"translator_gui.py"``)
      inside a call argument or a path join fails, unless it is listed in
      ``[baseline.gui_file_refs]``.
   b. **Process-spawn ratchet.** It flags calls to, and value references of,
      ProcessPoolExecutor, multiprocessing pools, processes and sync primitives,
      ``subprocess.*``, ``os.fork/system/popen/spawn*/exec*/posix_spawn/startfile`` and
      asyncio subprocesses. Import aliases are resolved:
      ``import subprocess as _sp`` then ``_sp.Popen``;
      ``from concurrent.futures import ProcessPoolExecutor as PPE``;
      a module-level ``X = subprocess.Popen``;
      ``mp.get_context("spawn").Pool``. Helpers that wrap subprocess calls are listed in
      ``[spawn].wrappers`` (e.g. ``shutdown_utils.popen_no_window``) and count as spawn
      APIs too. Every ``module:function`` site must be in
      ``[baseline.process_spawn_sites]`` with the same APIs and no more call sites.
   c. ``compile()`` of the AST must succeed on the running interpreter (3.13 in CI).
   d. **Pinned third-party imports.** Every unguarded ``module``-kind third-party
      import must map to a pinned dependency. Mapping comes from ``[thirdparty.map]``,
      falling back to the PEP 503-normalised top-level name. Pins are pyproject
      ``[project].dependencies``, plus Android-only ``[tool.flet.android].dependencies``,
      plus the ``[thirdparty].transitive`` allowlist. ``[thirdparty].unavailable`` and
      ``[platform].unavailable`` imports fail only when module-scope and unguarded.
   e. Imports of stdlib modules removed by Python 3.12/3.13 fail at module scope.
   f. A non-literal ``__import__(x)`` or ``importlib.import_module(x)`` must be declared
      in ``[dynamic_imports]`` along with the modules it can load.

5. **Output.** Without ``--check``, the closure and ``[data].files`` are wiped and
   copied into ``--out`` (default ``app/backend``), together with
   ``_bundle_info.py``. That file holds the version, git sha, per-file sha256, bundle
   sha256, modules, exclusions, edge statuses and spawn sites.

   * ``--report`` writes Markdown and also appends it to ``$GITHUB_STEP_SUMMARY``.
   * ``--json`` writes the machine-readable result.
   * ``--verify`` re-hashes an existing bundle against its ``_bundle_info.py`` and,
     when the repo ``src/`` exists, against the current sources (freshness).

Exit codes: 0 ok; 1 policy failures or verify mismatch; 2 usage/config error.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    sys.stderr.write("collect_backend.py needs Python 3.11+ (tomllib)\n")
    raise SystemExit(2)

COLLECTOR_VERSION = "2"
BUNDLE_INFO_NAME = "_bundle_info.py"

TOOLS_DIR = Path(__file__).resolve().parent
MOBILE_DIR = TOOLS_DIR.parent
DEFAULT_SRC = MOBILE_DIR.parent
DEFAULT_MANIFEST = MOBILE_DIR / "backend_manifest.toml"
DEFAULT_PYPROJECT = MOBILE_DIR / "pyproject.toml"
DEFAULT_OUT = MOBILE_DIR / "app" / "backend"
DEFAULT_CACHE = MOBILE_DIR / "build" / "collector_cache.json"

GUARD_EXCEPTIONS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})
NAMESPACE_PACKAGES = frozenset({"google", "azure", "backports", "jaraco", "zope", "sphinxcontrib"})

# Stdlib modules removed by Python 3.12/3.13 (PEP 594, distutils, imp, asyncore...).
REMOVED_STDLIB = frozenset({
    "aifc", "asynchat", "asyncore", "audioop", "cgi", "cgitb", "chunk", "crypt", "distutils",
    "imghdr", "imp", "lib2to3", "mailcap", "msilib", "nis", "nntplib", "ossaudiodev", "pipes",
    "smtpd", "sndhdr", "spwd", "sunau", "telnetlib", "uu", "xdrlib",
})

# Qualified name -> category. Calls and value references are both reported.
SPAWN_APIS: dict[str, str] = {
    "concurrent.futures.ProcessPoolExecutor": "process-pool",
    "concurrent.futures.process.ProcessPoolExecutor": "process-pool",
    "subprocess.Popen": "subprocess",
    "subprocess.run": "subprocess",
    "subprocess.call": "subprocess",
    "subprocess.check_call": "subprocess",
    "subprocess.check_output": "subprocess",
    "subprocess.getoutput": "subprocess",
    "subprocess.getstatusoutput": "subprocess",
    "asyncio.create_subprocess_exec": "subprocess",
    "asyncio.create_subprocess_shell": "subprocess",
    "pty.spawn": "os-exec",
    "pty.fork": "os-exec",
}
for _n in ("fork", "forkpty", "system", "popen", "startfile", "posix_spawn", "posix_spawnp",
           "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
           "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe"):
    SPAWN_APIS[f"os.{_n}"] = "os-exec"
for _n in ("Pool", "Process", "get_context", "Manager", "set_start_method"):
    SPAWN_APIS[f"multiprocessing.{_n}"] = "multiprocessing"
SPAWN_APIS["multiprocessing.pool.Pool"] = "multiprocessing"
SPAWN_APIS["multiprocessing.context.Process"] = "multiprocessing"
# These need a working sem_open (missing on Android) even though they spawn nothing.
for _n in ("Queue", "JoinableQueue", "SimpleQueue", "Event", "Lock", "RLock", "Semaphore",
           "BoundedSemaphore", "Condition", "Barrier", "Value", "Array"):
    SPAWN_APIS[f"multiprocessing.{_n}"] = "multiprocessing-sync"

DYNAMIC_IMPORT_FUNCS = frozenset({"__import__", "importlib.import_module", "importlib.__import__"})
# Module-valued aliases worth tracking through plain assignments (``sp = subprocess``).
ALIAS_MODULES = frozenset({"subprocess", "multiprocessing", "multiprocessing.pool", "os", "concurrent.futures",
                           "importlib", "asyncio", "pty"})
DEFAULT_GUI_PACKAGES = ("PySide6", "shiboken6", "PyQt5", "PyQt6", "tkinter", "ttkbootstrap")
_PY_FILE_RE = re.compile(r"(?:^|[\\/])([A-Za-z_][A-Za-z0-9_]*)\.pyw?$")


# --------------------------------------------------------------------------- data model
@dataclass
class ImportRef:
    target: str      # absolute dotted name ("from google import genai" -> "google.genai")
    top: str         # first component
    lineno: int
    kind: str        # module | conditional | lazy | main | type_checking
    guarded: bool
    func: str        # enclosing function qualname, "<module>" at module scope
    dynamic: bool = False


@dataclass
class SpawnRef:
    api: str
    category: str
    lineno: int
    func: str
    how: str  # call | reference


@dataclass
class ModuleInfo:
    name: str
    relpath: str
    sha256: str
    size: int
    lines: int
    bom: bool
    imports: list = field(default_factory=list)          # list[ImportRef]
    spawns: list = field(default_factory=list)           # list[SpawnRef]
    dynamic_sites: list = field(default_factory=list)    # {func, lineno, expr}
    file_refs: list = field(default_factory=list)        # {lineno, literal, module, func}
    dunder_file_lines: list = field(default_factory=list)
    error: str = ""          # read/parse error
    compile_error: str = ""  # compile() failure

    @staticmethod
    def from_dict(d: dict) -> "ModuleInfo":
        d = dict(d)
        d["imports"] = [ImportRef(**x) for x in d.get("imports", [])]
        d["spawns"] = [SpawnRef(**x) for x in d.get("spawns", [])]
        return ModuleInfo(**d)


@dataclass
class Finding:
    level: str      # error | warning | info
    code: str
    module: str
    message: str
    lines: list = field(default_factory=list)


# --------------------------------------------------------------------------- helpers
def normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def decode_source(data: bytes) -> str:
    """Decode Python source bytes honouring a UTF-8 BOM and PEP 263 cookies."""
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    text = data.decode(encoding)
    return text[1:] if text.startswith("\ufeff") else text


def read_source(path: Path) -> str:
    return decode_source(path.read_bytes())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dotted(expr: ast.AST) -> str | None:
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def _exc_name(e: ast.AST) -> str | None:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return e.attr
    return None


def _handler_catches_import(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    elts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(_exc_name(e) in GUARD_EXCEPTIONS for e in elts)


def _is_main_test(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    a, b = test.left, test.comparators[0]
    for x, y in ((a, b), (b, a)):
        if isinstance(x, ast.Name) and x.id == "__name__" and isinstance(y, ast.Constant) and y.value == "__main__":
            return True
    return False


def _is_type_checking_test(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


_TRY_TYPES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


# --------------------------------------------------------------------------- per-module analysis
@dataclass(frozen=True)
class _Ctx:
    chain: tuple          # scope keys used for name resolution, innermost last
    func: str             # qualname of the enclosing function, "<module>" at module scope
    qual: str             # qualname prefix for nested definitions
    in_function: bool = False
    guarded: bool = False
    conditional: bool = False
    main: bool = False
    type_checking: bool = False

    def kind(self) -> str:
        if self.type_checking:
            return "type_checking"
        if self.main:
            return "main"
        if self.in_function:
            return "lazy"
        if self.conditional:
            return "conditional"
        return "module"

    def but(self, **kw) -> "_Ctx":
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kw)
        return _Ctx(**d)

    def fn_chain(self) -> tuple:
        # Class scopes are not visible from nested functions (Python scoping rules).
        return tuple(k for k in self.chain if not k.startswith("class:"))


class _Analyzer:
    """One AST pass that records imports, alias bindings, spawn/dynamic calls and string refs."""

    def __init__(self, info: ModuleInfo, local_names: frozenset, spawn_apis: dict | None = None):
        self.info = info
        self.local_names = local_names
        self.spawn_apis = spawn_apis or SPAWN_APIS
        self.bindings: dict[str, dict[str, str]] = {}  # scope key -> {local name: qualified name}
        self.pending: list[tuple[ast.AST, _Ctx, str]] = []  # (expr, ctx, how)
        self._refs_seen: set[tuple[int, str]] = set()
        self._imports_seen: set[tuple] = set()

    # -- name binding / resolution
    def _bind(self, ctx: _Ctx, name: str, qualified: str) -> None:
        self.bindings.setdefault(ctx.chain[-1], {})[name] = qualified

    def resolve(self, expr: ast.AST, chain: tuple) -> str | None:
        if isinstance(expr, ast.Name):
            for key in reversed(chain):
                q = self.bindings.get(key, {}).get(expr.id)
                if q is not None:
                    return q
            return "__import__" if expr.id == "__import__" else None
        if isinstance(expr, ast.Attribute):
            base = self.resolve(expr.value, chain)
            return f"{base}.{expr.attr}" if base else None
        if isinstance(expr, ast.Call):
            # multiprocessing.get_context("spawn").Pool(...) -> multiprocessing.Pool
            if self.resolve(expr.func, chain) == "multiprocessing.get_context":
                return "multiprocessing"
        return None

    # -- imports
    def _add_import(self, ref: ImportRef) -> None:
        key = (ref.target, ref.lineno, ref.kind, ref.guarded, ref.func)
        if key not in self._imports_seen:
            self._imports_seen.add(key)
            self.info.imports.append(ref)

    def _record_import(self, node: ast.AST, ctx: _Ctx) -> None:
        kind = ctx.kind()
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                top = target.split(".")[0]
                self._add_import(ImportRef(target, top, node.lineno, kind, ctx.guarded, ctx.func))
                if alias.asname:
                    self._bind(ctx, alias.asname, target)
                else:
                    self._bind(ctx, top, top)
            return
        if node.level:  # relative import: meaningless for flat src/ modules
            return
        mod = node.module or ""
        top = mod.split(".")[0]
        for alias in node.names:
            full = mod if alias.name == "*" else f"{mod}.{alias.name}"
            # Keep the submodule for namespace packages so google.genai / google.auth map
            # to the right distributions.
            target = full if top in NAMESPACE_PACKAGES else mod
            self._add_import(ImportRef(target, top, node.lineno, kind, ctx.guarded, ctx.func))
            if alias.name != "*":
                self._bind(ctx, alias.asname or alias.name, full)

    # -- traversal
    def walk(self, node: ast.AST, ctx: _Ctx) -> None:  # noqa: C901 - single dispatcher
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                self.walk(d, ctx)
            self._walk_defaults(node.args, ctx)
            qual = f"{ctx.qual}{node.name}"
            inner = _Ctx(chain=ctx.fn_chain() + (qual,), func=qual, qual=f"{qual}.<locals>.",
                         in_function=True, main=ctx.main, type_checking=ctx.type_checking)
            self._bind(ctx, node.name, f"<def>.{node.name}")  # shadows imported names
            for stmt in node.body:
                self.walk(stmt, inner)
            return
        if isinstance(node, ast.Lambda):
            self._walk_defaults(node.args, ctx)
            inner = _Ctx(chain=ctx.fn_chain() + (f"{ctx.qual}<lambda>@{node.lineno}",), func=ctx.func,
                         qual=ctx.qual, in_function=True, main=ctx.main, type_checking=ctx.type_checking)
            self.walk(node.body, inner)
            return
        if isinstance(node, ast.ClassDef):
            for d in node.decorator_list:
                self.walk(d, ctx)
            for b in node.bases:
                self.walk(b, ctx)
            for k in node.keywords:
                self.walk(k.value, ctx)
            qual = f"{ctx.qual}{node.name}"
            inner = ctx.but(chain=ctx.chain + (f"class:{qual}",), qual=f"{qual}.")
            self._bind(ctx, node.name, f"<def>.{node.name}")
            for stmt in node.body:
                self.walk(stmt, inner)
            return
        if isinstance(node, _TRY_TYPES):
            guard = any(_handler_catches_import(h) for h in node.handlers)
            body_ctx = ctx.but(guarded=True) if guard else ctx
            for stmt in node.body:
                self.walk(stmt, body_ctx)
            for h in node.handlers:
                if h.type is not None:
                    self.walk(h.type, ctx)
                for stmt in h.body:
                    self.walk(stmt, ctx.but(conditional=True))
            for stmt in node.orelse:
                self.walk(stmt, ctx.but(conditional=True))
            for stmt in node.finalbody:
                self.walk(stmt, ctx)
            return
        if isinstance(node, ast.If):
            self.walk(node.test, ctx)
            if _is_main_test(node.test):
                body_ctx = ctx.but(main=True)
            elif _is_type_checking_test(node.test):
                body_ctx = ctx.but(type_checking=True)
            else:
                body_ctx = ctx.but(conditional=True)
            for stmt in node.body:
                self.walk(stmt, body_ctx)
            for stmt in node.orelse:
                self.walk(stmt, ctx.but(conditional=True))
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            guard = False
            for item in node.items:
                self.walk(item.context_expr, ctx)
                if item.optional_vars is not None:
                    self.walk(item.optional_vars, ctx)
                ce = item.context_expr
                if isinstance(ce, ast.Call) and (_dotted(ce.func) or "").split(".")[-1] == "suppress":
                    guard = guard or any(_exc_name(a) in GUARD_EXCEPTIONS for a in ce.args)
            body_ctx = ctx.but(guarded=True) if guard else ctx
            for stmt in node.body:
                self.walk(stmt, body_ctx)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._record_import(node, ctx)
            return
        if isinstance(node, ast.Assign):
            self.walk(node.value, ctx)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, (ast.Name, ast.Attribute))):
                self.pending.append((node.value, ctx, "alias:" + node.targets[0].id))
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.walk(node.value, ctx)
            return
        if isinstance(node, ast.Call):
            self._visit_call(node, ctx)
            return
        if isinstance(node, (ast.Name, ast.Attribute)):
            if isinstance(node.ctx, ast.Load):
                self.pending.append((node, ctx, "reference"))
            if isinstance(node, ast.Name):
                if node.id == "__file__":
                    self.info.dunder_file_lines.append(node.lineno)
                return
            inner = node.value
            while isinstance(inner, ast.Attribute):
                inner = inner.value
            if not isinstance(inner, ast.Name):
                self.walk(inner, ctx)
            elif inner.id == "__file__":
                self.info.dunder_file_lines.append(inner.lineno)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for side in (node.left, node.right):  # Path(...) / "x.py"
                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                    self._string_ref(side, ctx)
        if isinstance(node, ast.arg):
            return  # annotations are never spawn sites
        for child in ast.iter_child_nodes(node):
            self.walk(child, ctx)

    def _walk_defaults(self, args: ast.arguments, ctx: _Ctx) -> None:
        for d in list(args.defaults) + [x for x in args.kw_defaults if x is not None]:
            self.walk(d, ctx)

    def _string_ref(self, node: ast.Constant, ctx: _Ctx) -> None:
        value = node.value.strip()
        if len(value) > 300:
            return
        m = _PY_FILE_RE.search(value)
        if m and (node.lineno, node.value) not in self._refs_seen:
            self._refs_seen.add((node.lineno, node.value))
            self.info.file_refs.append({"lineno": node.lineno, "literal": node.value,
                                        "module": m.group(1), "func": ctx.func})

    def _visit_call(self, node: ast.Call, ctx: _Ctx) -> None:
        fname = _dotted(node.func)
        self.pending.append((node.func, ctx, "call"))
        self.pending.append((node, ctx, "dyncheck"))
        if isinstance(node.func, ast.Attribute):
            inner = node.func.value
            while isinstance(inner, ast.Attribute):
                inner = inner.value
            if not isinstance(inner, ast.Name):
                self.walk(inner, ctx)
            elif inner.id == "__file__":
                self.info.dunder_file_lines.append(inner.lineno)
        elif not isinstance(node.func, ast.Name):
            self.walk(node.func, ctx)
        skip_args = fname in ("isinstance", "issubclass")
        for a in list(node.args) + [k.value for k in node.keywords]:
            for sub in ast.walk(a):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    self._string_ref(sub, ctx)
            if not skip_args:
                self.walk(a, ctx)

    def finish(self) -> None:
        # Resolve plain-assignment aliases first (two rounds handle simple chains).
        for _ in range(2):
            for expr, ctx, how in self.pending:
                if how.startswith("alias:"):
                    q = self.resolve(expr, ctx.chain)
                    if q and (q in self.spawn_apis or q in DYNAMIC_IMPORT_FUNCS or q in ALIAS_MODULES):
                        self.bindings.setdefault(ctx.chain[-1], {})[how[6:]] = q
        seen: set[tuple] = set()
        for expr, ctx, how in self.pending:
            if how.startswith("alias:"):
                continue
            if how == "dyncheck":
                q = self.resolve(expr.func, ctx.chain)
                if q in DYNAMIC_IMPORT_FUNCS and expr.args:
                    a0 = expr.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value:
                        self._add_import(ImportRef(a0.value, a0.value.split(".")[0], expr.lineno,
                                                   ctx.kind(), ctx.guarded, ctx.func, dynamic=True))
                    else:
                        self.info.dynamic_sites.append({"func": ctx.func, "lineno": expr.lineno,
                                                        "expr": ast.unparse(a0)[:120]})
                continue
            q = self.resolve(expr, ctx.chain)
            if q in self.spawn_apis:
                key = (q, expr.lineno, ctx.func, how)
                if key not in seen:
                    seen.add(key)
                    self.info.spawns.append(SpawnRef(q, self.spawn_apis[q], expr.lineno, ctx.func,
                                                     "call" if how == "call" else "reference"))
        calls = {(s.api, s.lineno, s.func) for s in self.info.spawns if s.how == "call"}
        self.info.spawns = sorted((s for s in self.info.spawns
                                   if s.how == "call" or (s.api, s.lineno, s.func) not in calls),
                                  key=lambda s: (s.lineno, s.api))
        self.info.dunder_file_lines = sorted(set(self.info.dunder_file_lines))
        self.pending.clear()


def analyze_source(name: str, relpath: str, data: bytes, local_names: frozenset,
                   do_compile: bool = True, spawn_apis: dict | None = None) -> ModuleInfo:
    info = ModuleInfo(name=name, relpath=relpath, sha256=sha256_bytes(data), size=len(data),
                      lines=data.count(b"\n") + 1, bom=data.startswith(b"\xef\xbb\xbf"))
    try:
        with warnings.catch_warnings():  # invalid escape sequences etc. are the desktop code's business
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(decode_source(data), filename=relpath)
    except (SyntaxError, UnicodeDecodeError, LookupError, ValueError) as e:
        info.error = f"{type(e).__name__}: {e}"
        return info
    an = _Analyzer(info, local_names, spawn_apis)
    ctx = _Ctx(chain=("<module>",), func="<module>", qual="")
    for stmt in tree.body:
        an.walk(stmt, ctx)
    an.finish()
    if do_compile:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                compile(tree, relpath, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            info.compile_error = f"{type(e).__name__}: {e}"
    return info


# --------------------------------------------------------------------------- manifest / pins
@dataclass
class Manifest:
    path: Path
    entries: list
    planned: list
    extra_modules: list
    scan_packages: list
    app_local: list
    excluded: dict
    gui_packages: list
    gui_roots: list
    lazy_gui_edges: dict
    lazy_excluded_edges: dict
    spawn_sites: dict
    gui_file_refs: dict
    dynamic_imports: dict
    tp_map: dict
    tp_transitive: dict
    tp_unavailable: dict
    platform_unavailable: dict
    data_files: list
    spawn_wrappers: dict = field(default_factory=dict)


def _baseline_table(raw: dict, key: str) -> dict:
    out = {}
    for k, v in (raw.get("baseline", {}).get(key, {}) or {}).items():
        out[k] = {"count": None, "reason": v} if isinstance(v, str) else dict(v)
    return out


def load_manifest(path: Path) -> Manifest:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    entry = raw.get("entry", {})
    scan = entry.get("scan_package", [])
    gui = raw.get("gui", {})
    tp = raw.get("thirdparty", {})
    transitive = tp.get("transitive", {})
    if isinstance(transitive, list):
        transitive = {name: "" for name in transitive}
    return Manifest(
        path=Path(path),
        entries=list(entry.get("modules", [])),
        planned=list(entry.get("planned", [])),
        extra_modules=list(entry.get("extra_modules", [])),
        scan_packages=[scan] if isinstance(scan, str) else list(scan),
        app_local=list(entry.get("app_local", ["glossarion_mobile", "main", "flet_glossarion_native"])),
        excluded=dict(raw.get("exclude", {})),
        gui_packages=list(gui.get("packages", DEFAULT_GUI_PACKAGES)),
        gui_roots=list(gui.get("roots", [])),
        lazy_gui_edges=_baseline_table(raw, "lazy_gui_edges"),
        lazy_excluded_edges=_baseline_table(raw, "lazy_excluded_edges"),
        spawn_sites=_baseline_table(raw, "process_spawn_sites"),
        gui_file_refs=_baseline_table(raw, "gui_file_refs"),
        dynamic_imports={k: list(v) for k, v in raw.get("dynamic_imports", {}).items()},
        tp_map=dict(tp.get("map", {})),
        tp_transitive=dict(transitive),
        tp_unavailable=dict(tp.get("unavailable", {})),
        platform_unavailable=dict(raw.get("platform", {}).get("unavailable", {})),
        data_files=list(raw.get("data", {}).get("files", [])),
        spawn_wrappers=dict(raw.get("spawn", {}).get("wrappers", {})),
    )


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass
class Pins:
    common: dict          # normalized dist -> requirement string
    android_only: dict
    ios_only: dict
    source_packages: list

    def scope(self, dist: str) -> str | None:
        d = normalize_dist(dist)
        if d in self.common or (d in self.android_only and d in self.ios_only):
            return "common"
        if d in self.android_only:
            return "android"
        if d in self.ios_only:
            return "ios"
        return None


def load_pins(pyproject: Path | None) -> Pins:
    if pyproject is None or not Path(pyproject).exists():
        return Pins({}, {}, {}, [])
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    def reqs(lst):
        out = {}
        for r in lst or []:
            m = _REQ_NAME.match(r)
            if m:
                out[normalize_dist(m.group(1))] = r
        return out

    flet = data.get("tool", {}).get("flet", {})
    return Pins(common=reqs(data.get("project", {}).get("dependencies", [])),
                android_only=reqs(flet.get("android", {}).get("dependencies", [])),
                ios_only=reqs(flet.get("ios", {}).get("dependencies", [])),
                source_packages=list(flet.get("source_packages", [])))


# --------------------------------------------------------------------------- the collector
@dataclass
class EdgeSite:
    src: str
    dst: str
    kind: str
    guarded: bool
    lineno: int
    func: str


@dataclass
class Result:
    src_dir: str
    manifest: str
    python: str
    closure: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)        # module -> why bundled
    tainted: dict = field(default_factory=dict)        # tainted module reached by an edge -> reason
    blocked_edges: list = field(default_factory=list)  # EdgeSite into excluded/tainted/GUI package
    edge_status: dict = field(default_factory=dict)    # "src->dst" -> guarded | baselined | FAIL (...)
    spawn_sites: dict = field(default_factory=dict)    # "mod:func" -> {apis, count, lines, status}
    thirdparty: dict = field(default_factory=dict)     # dist -> {...}
    findings: list = field(default_factory=list)
    data_files: list = field(default_factory=list)
    infos: dict = field(default_factory=dict)
    app_modules: list = field(default_factory=list)
    untracked: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.level == "warning"]


class Collector:
    def __init__(self, src_dir: Path, manifest: Manifest, pins: Pins, *,
                 mobile_dir: Path | None = None, cache_path: Path | None = None):
        self.src = Path(src_dir).resolve()
        self.m = manifest
        self.pins = pins
        self.mobile_dir = Path(mobile_dir or manifest.path.parent).resolve()
        self.cache_path = cache_path
        self.local = {p.stem: p for p in self.src.glob("*.py") if p.stem.isidentifier()}
        self.local_names = frozenset(self.local)
        self.infos: dict[str, ModuleInfo] = {}
        self._cache: dict[str, dict] = {}
        self._cache_dirty = False
        self._taint: dict[str, str | None] = {}
        self._taint_via: dict[str, tuple] = {}
        self.stdlib = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
        self.transitive = {normalize_dist(x) for x in self.m.tp_transitive}
        self.spawn_apis = {**SPAWN_APIS, **self.m.spawn_wrappers}
        self._cache_key = f"{COLLECTOR_VERSION}|{sorted(self.m.spawn_wrappers.items())}"
        self._load_cache()

    # -- parsing with a per-file cache keyed by content hash
    def _load_cache(self) -> None:
        if not self.cache_path or not Path(self.cache_path).exists():
            return
        try:
            data = json.loads(Path(self.cache_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("version") == self._cache_key and data.get("python") == sys.version:
            self._cache = data.get("modules", {})

    def save_cache(self) -> None:
        if not self.cache_path or not self._cache_dirty:
            return
        try:
            path = Path(self.cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": self._cache_key, "python": sys.version,
                                       "modules": self._cache}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    def info(self, name: str) -> ModuleInfo:
        mi = self.infos.get(name)
        if mi is not None:
            return mi
        data = self.local[name].read_bytes()
        digest = sha256_bytes(data)
        cached = self._cache.get(name)
        if cached and cached.get("sha256") == digest:
            mi = ModuleInfo.from_dict(cached)
        else:
            mi = analyze_source(name, self.local[name].name, data, self.local_names, spawn_apis=self.spawn_apis)
            self._cache[name] = asdict(mi)
            self._cache_dirty = True
        self.infos[name] = mi
        return mi

    # -- classification
    def is_excluded(self, name: str) -> bool:
        return name in self.m.excluded

    def is_gui_pkg(self, top: str) -> bool:
        return top in self.m.gui_packages

    def _eager_local(self, name: str):
        for imp in self.info(name).imports:
            if (imp.kind == "module" and not imp.guarded and imp.top in self.local and imp.top != name
                    and not self.is_excluded(imp.top)):
                yield imp

    def taint(self, name: str) -> str | None:
        """Reason string when ``name`` is GUI-tainted (fixpoint over module-scope imports), else None."""
        if name in self._taint:
            return self._taint[name]
        seen: list[str] = []
        stack = [name]
        while stack:
            n = stack.pop()
            if n in seen or n in self._taint:
                continue
            seen.append(n)
            if n in self.m.gui_roots:
                continue
            stack.extend(imp.top for imp in self._eager_local(n))
        via: dict[str, tuple] = {}
        for n in seen:
            if n in self.m.gui_roots:
                via[n] = ("root",)
                continue
            if self.info(n).error:
                continue
            for imp in self.info(n).imports:
                if imp.kind == "module" and not imp.guarded and self.is_gui_pkg(imp.top):
                    via[n] = ("pkg", imp.target, imp.lineno)
                    break
        changed = True
        while changed:
            changed = False
            for n in seen:
                if n in via or self.info(n).error:
                    continue
                for imp in self._eager_local(n):
                    if imp.top in via or self._taint.get(imp.top):
                        via[n] = ("mod", imp.top, imp.lineno)
                        changed = True
                        break
        self._taint_via.update(via)
        for n in seen:
            self._taint[n] = self._describe_taint(n) if n in via else None
        return self._taint[name]

    def _describe_taint(self, name: str) -> str:
        parts, n, guard = [], name, 0
        while guard < 20:
            guard += 1
            v = self._taint_via.get(n)
            if v is None:
                break
            if v[0] == "root":
                parts.append(f"{n} is a GUI root")
                break
            if v[0] == "pkg":
                parts.append(f"{n} imports {v[1]} at module scope (line {v[2]})")
                break
            parts.append(f"{n} imports {v[1]} (line {v[2]})")
            n = v[1]
        return " -> ".join(parts)

    def dist_for(self, target: str) -> str:
        parts = target.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in self.m.tp_map:
                return normalize_dist(self.m.tp_map[key])
        return normalize_dist(parts[0])

    # -- main entry
    def run(self) -> Result:
        res = Result(src_dir=str(self.src), manifest=str(self.m.path), python=sys.version.split()[0])
        F = res.findings
        roots: list[tuple[str, str]] = []
        for e in self.m.entries:
            if e in self.local:
                roots.append((e, "entry"))
            else:
                F.append(Finding("error", "entry-missing", e, f"[entry].modules lists {e}, but src/{e}.py does not exist"))
        for e in self.m.planned:
            if e in self.local:
                roots.append((e, "entry (planned)"))
                F.append(Finding("info", "planned-present", e, f"planned module {e}.py now exists; move it to [entry].modules"))
        for e in self.m.extra_modules:
            if e in self.local:
                roots.append((e, "extra_modules"))
            else:
                F.append(Finding("error", "entry-missing", e, f"[entry].extra_modules lists {e}, but src/{e}.py does not exist"))

        app_infos = self._scan_app(res, roots)

        closure: dict[str, str] = {}
        queue: list[str] = []
        for name, why in roots:
            if name in closure:
                continue
            if self.is_excluded(name):
                F.append(Finding("error", "entry-excluded", name, f"{name} ({why}) is listed in [exclude]"))
                continue
            t = self.taint(name)
            if t:
                F.append(Finding("error", "entry-tainted", name, f"{name} ({why}) is GUI-tainted: {t}"))
                continue
            closure[name] = why
            queue.append(name)

        third: dict[str, dict] = {}
        while queue:
            name = queue.pop(0)
            mi = self.info(name)
            if mi.error:
                F.append(Finding("error", "parse", name, mi.error))
                continue
            imports = list(mi.imports)
            for site, mods in self.m.dynamic_imports.items():
                smod, _, sfunc = site.partition(":")
                if smod == name:
                    line = next((s["lineno"] for s in mi.dynamic_sites if s["func"] == sfunc), 0)
                    imports.extend(ImportRef(t, t.split(".")[0], line, "lazy", False, sfunc, dynamic=True)
                                   for t in mods)
            for imp in imports:
                if imp.kind == "type_checking":
                    continue
                top = imp.top
                if top in self.local:
                    if top == name:
                        continue
                    if self.is_excluded(top) or top in self.m.gui_roots or self.taint(top):
                        res.blocked_edges.append(EdgeSite(name, top, imp.kind, imp.guarded, imp.lineno, imp.func))
                    elif imp.kind != "main" and top not in closure:
                        closure[top] = (f"{name}:{imp.lineno} ({imp.kind}{', guarded' if imp.guarded else ''}"
                                        f"{', dynamic' if imp.dynamic else ''})")
                        queue.append(top)
                    continue
                if self.is_gui_pkg(top):
                    res.blocked_edges.append(EdgeSite(name, top, imp.kind, imp.guarded, imp.lineno, imp.func))
                    continue
                self._classify_external(name, imp, third, F)

        for mi in app_infos:
            for imp in mi.imports:
                if imp.kind == "type_checking" or imp.top in self.m.app_local or imp.top in self.local:
                    continue
                if imp.top in self.m.planned:
                    continue
                if self.is_gui_pkg(imp.top):
                    F.append(Finding("error" if imp.kind == "module" and not imp.guarded else "warning",
                                     "app-gui-import", mi.name, f"app imports GUI package {imp.target} ({imp.kind})",
                                     [imp.lineno]))
                    continue
                self._classify_external(mi.name, imp, third, F)

        res.closure = sorted(closure)
        res.reasons = closure
        res.thirdparty = third
        for e in res.blocked_edges:
            if e.dst in self.local and not self.is_excluded(e.dst):
                res.tainted[e.dst] = self.taint(e.dst) or f"{e.dst} is a GUI root"
        self._check_blocked_edges(res)
        self._check_modules(res)
        self._check_dynamic(res)
        self._check_data(res)
        self._check_tracked(res)
        res.infos = {n: self.infos[n] for n in res.closure}
        self.save_cache()
        return res

    def _scan_app(self, res: Result, roots: list) -> list:
        infos = []
        for pkg in self.m.scan_packages:
            pdir = self.mobile_dir / pkg
            if not pdir.exists():
                res.findings.append(Finding("warning", "scan-package-missing", pkg, f"scan_package {pkg} not found"))
                continue
            files = sorted(pdir.rglob("*.py")) if pdir.is_dir() else [pdir]
            for p in files:
                if "__pycache__" in p.parts:
                    continue
                rel = p.relative_to(self.mobile_dir).as_posix()
                mi = analyze_source(rel, rel, p.read_bytes(), self.local_names, spawn_apis=self.spawn_apis)
                infos.append(mi)
                res.app_modules.append(rel)
                if mi.error:
                    res.findings.append(Finding("error", "app-parse", rel, mi.error))
                for imp in mi.imports:
                    if imp.kind != "type_checking" and imp.top in self.local and imp.top not in self.m.app_local:
                        roots.append((imp.top, f"app {rel}:{imp.lineno} ({imp.kind})"))
        return infos

    # -- rule (d) and friends
    def _classify_external(self, mod: str, imp: ImportRef, third: dict, F: list) -> None:
        top = imp.top
        eager = imp.kind == "module" and not imp.guarded
        where = f"{mod}:{imp.lineno} {imp.kind}{' guarded' if imp.guarded else ''}"
        if top in REMOVED_STDLIB:
            F.append(Finding("error" if eager else "warning", "removed-stdlib", mod,
                             f"imports {imp.target}, which Python 3.13 no longer ships ({imp.kind})", [imp.lineno]))
            return
        if top in self.m.platform_unavailable:
            rec = third.setdefault(f"stdlib:{top}", {"dist": top, "status": "platform-unavailable",
                                                     "reason": self.m.platform_unavailable[top],
                                                     "imports": set(), "sites": [], "eager_sites": []})
            rec["imports"].add(imp.target)
            rec["sites"].append(where)
            if eager:
                rec["eager_sites"].append(where)
                F.append(Finding("error", "platform-import", mod,
                                 f"unguarded module-scope import of {imp.target}, unavailable on Android/iOS "
                                 f"({self.m.platform_unavailable[top]})", [imp.lineno]))
            return
        if top in self.stdlib:
            return
        dist = self.dist_for(imp.target)
        rec = third.setdefault(dist, {"dist": dist, "status": "", "imports": set(), "sites": [], "eager_sites": []})
        rec["imports"].add(".".join(imp.target.split(".")[:3]) if top in NAMESPACE_PACKAGES else top)
        rec["sites"].append(where)
        if eager:
            rec["eager_sites"].append(where)
        unavailable = self.m.tp_unavailable.get(dist) or self.m.tp_unavailable.get(top)
        if unavailable:
            rec["status"], rec["reason"] = "unavailable", unavailable
            if eager:
                F.append(Finding("error", "unavailable-import", mod,
                                 f"unguarded module-scope import of {imp.target} ({dist}: {unavailable})", [imp.lineno]))
            return
        scope = self.pins.scope(dist)
        if scope is None and dist in self.transitive:
            scope = "transitive"
        if scope in ("common", "transitive"):
            rec["status"] = "pinned" if scope == "common" else "transitive"
            return
        if scope in ("android", "ios"):
            rec["status"] = f"{scope}-only"
            if eager:
                F.append(Finding("error", f"{scope}-only-import", mod,
                                 f"unguarded module-scope import of {imp.target}: {dist} is pinned for {scope} only",
                                 [imp.lineno]))
            return
        rec["status"] = rec["status"] or "unpinned"
        if eager:
            F.append(Finding("error", "unpinned-import", mod,
                             f"unguarded module-scope import of {imp.target} maps to '{dist}', which is not pinned "
                             f"(pin it in pyproject, map it in [thirdparty.map] or list it in [thirdparty].transitive)",
                             [imp.lineno]))
        elif imp.kind == "conditional" and not imp.guarded:
            F.append(Finding("warning", "unpinned-conditional", mod,
                             f"conditional module-scope import of {imp.target} ('{dist}') is not pinned", [imp.lineno]))

    # -- edges into GUI / excluded modules
    def _check_blocked_edges(self, res: Result) -> None:
        F = res.findings
        groups: dict[tuple, list] = {}
        for e in res.blocked_edges:
            groups.setdefault((e.src, e.dst), []).append(e)
        used_gui, used_exc = set(), set()
        for (src, dst), sites in sorted(groups.items()):
            key = f"{src}->{dst}"
            excluded = self.is_excluded(dst)
            table = self.m.lazy_excluded_edges if excluded else self.m.lazy_gui_edges
            section = "lazy_excluded_edges" if excluded else "lazy_gui_edges"
            what = (f"excluded module {dst} ({self.m.excluded[dst]})" if excluded else
                    f"GUI package {dst}" if self.is_gui_pkg(dst) else f"GUI-tainted module {dst}")
            eager = [s for s in sites if s.kind == "module" and not s.guarded]
            unguarded = [s for s in sites if not s.guarded]
            if eager:
                F.append(Finding("error", "eager-blocked-edge", src, f"unguarded module-scope import of {what}",
                                 sorted({s.lineno for s in eager})))
                res.edge_status[key] = "FAIL (module-scope)"
                continue
            if not unguarded:
                res.edge_status[key] = "guarded"
                continue
            lines = sorted({s.lineno for s in unguarded})
            count = len(lines)
            entry = table.get(key)
            if entry is None:
                F.append(Finding("error", "unbaselined-edge", src,
                                 f"new unguarded {'/'.join(sorted({s.kind for s in unguarded}))} import of {what}; "
                                 f"guard it with try/except ImportError or add \"{key}\" to [baseline.{section}]",
                                 lines))
                res.edge_status[key] = "FAIL (not baselined)"
                continue
            (used_exc if excluded else used_gui).add(key)
            limit = entry.get("count")
            if limit is not None and count > int(limit):
                F.append(Finding("error", "edge-ratchet", src,
                                 f"{key}: {count} unguarded import sites, baseline allows {limit}", lines))
                res.edge_status[key] = f"FAIL (count {count} > {limit})"
                continue
            if limit is not None and count < int(limit):
                F.append(Finding("info", "edge-ratchet-slack", src,
                                 f"{key}: {count} unguarded import sites, baseline allows {limit}; lower the count"))
            res.edge_status[key] = "baselined"
        for table, used, section in ((self.m.lazy_gui_edges, used_gui, "lazy_gui_edges"),
                                     (self.m.lazy_excluded_edges, used_exc, "lazy_excluded_edges")):
            for key in sorted(set(table) - used):
                F.append(Finding("warning", "stale-baseline", key.split("->")[0],
                                 f"[baseline.{section}] \"{key}\" no longer matches an unguarded edge; remove it"))

    def _check_modules(self, res: Result) -> None:
        F = res.findings
        closure = set(res.closure)
        blocked = set(self.m.excluded) | set(self.m.gui_roots) | set(res.tainted)
        used_spawn, used_refs = set(), set()
        for name in res.closure:
            mi = self.info(name)
            if mi.compile_error:
                F.append(Finding("error", "compile", name, mi.compile_error))
            # (a) string literals naming GUI/excluded module files
            for ref in mi.file_refs:
                target = ref["module"]
                if target == name or target not in self.local:
                    continue
                if target in blocked or (target not in closure and (self.is_excluded(target) or self.taint(target))):
                    key = f"{name}->{target}.py"
                    if key in self.m.gui_file_refs:
                        used_refs.add(key)
                        continue
                    F.append(Finding("error", "gui-file-ref", name,
                                     f"string literal {ref['literal']!r} in {ref['func']} names a GUI/excluded module "
                                     f"file; add \"{key}\" to [baseline.gui_file_refs] if it is harmless",
                                     [ref["lineno"]]))
                elif target not in closure:
                    F.append(Finding("warning", "script-ref-outside-bundle", name,
                                     f"string literal {ref['literal']!r} in {ref['func']} names {target}.py, "
                                     f"which is not bundled", [ref["lineno"]]))
            # (b) process-spawn ratchet
            by_func: dict[str, list] = {}
            for s in mi.spawns:
                by_func.setdefault(s.func, []).append(s)
            for func, sites in sorted(by_func.items()):
                key = f"{name}:{func}"
                apis = sorted({s.api for s in sites})
                lines = sorted({s.lineno for s in sites})
                count = len({(s.api, s.lineno) for s in sites})
                rec = {"apis": apis, "count": count, "lines": lines,
                       "categories": sorted({s.category for s in sites}), "status": ""}
                res.spawn_sites[key] = rec
                base = self.m.spawn_sites.get(key)
                if base is None:
                    rec["status"] = "FAIL (new)"
                    F.append(Finding("error", "spawn-new", name,
                                     f"new process-spawn site {key} ({', '.join(apis)}); gate it with "
                                     f"mobile_runtime.processes_available()/subprocesses_available() and list it in "
                                     f"[baseline.process_spawn_sites]", lines))
                    continue
                used_spawn.add(key)
                b_apis = set(base.get("apis") or [])
                new_apis = sorted(set(apis) - b_apis) if b_apis else []
                if new_apis:
                    rec["status"] = "FAIL (new api)"
                    F.append(Finding("error", "spawn-new-api", name,
                                     f"{key} now also uses {', '.join(new_apis)} (baseline: {', '.join(sorted(b_apis))})",
                                     lines))
                    continue
                b_count = base.get("count")
                if b_count is not None and count > int(b_count):
                    rec["status"] = f"FAIL (count {count} > {b_count})"
                    F.append(Finding("error", "spawn-ratchet", name,
                                     f"{key}: {count} spawn call sites, baseline allows {b_count}", lines))
                    continue
                if b_count is not None and count < int(b_count):
                    F.append(Finding("info", "spawn-ratchet-slack", name,
                                     f"{key}: {count} spawn call sites, baseline allows {b_count}; lower the count"))
                rec["status"] = "baselined"
        for key in sorted(set(self.m.spawn_sites) - used_spawn):
            F.append(Finding("warning", "stale-baseline", key.split(":")[0],
                             f"[baseline.process_spawn_sites] \"{key}\" no longer matches a spawn site; remove it"))
        for key in sorted(set(self.m.gui_file_refs) - used_refs):
            F.append(Finding("warning", "stale-baseline", key.split("->")[0],
                             f"[baseline.gui_file_refs] \"{key}\" no longer matches; remove it"))

    def _check_dynamic(self, res: Result) -> None:
        for name in res.closure:
            for site in self.info(name).dynamic_sites:
                key = f"{name}:{site['func']}"
                if key not in self.m.dynamic_imports:
                    res.findings.append(Finding("error", "dynamic-import", name,
                                                f"non-literal dynamic import {site['expr']!r} in {site['func']}; "
                                                f"declare \"{key}\" = [modules...] in [dynamic_imports]",
                                                [site["lineno"]]))
        for key in self.m.dynamic_imports:
            mod, _, func = key.partition(":")
            if mod not in res.closure:
                res.findings.append(Finding("warning", "stale-dynamic", mod,
                                            f"[dynamic_imports] \"{key}\": {mod} is not bundled"))
            elif func not in {s["func"] for s in self.info(mod).dynamic_sites}:
                res.findings.append(Finding("warning", "stale-dynamic", mod,
                                            f"[dynamic_imports] \"{key}\" no longer matches a dynamic import"))

    def _check_data(self, res: Result) -> None:
        files = set()
        for pattern in self.m.data_files:
            matches = [p for p in self.src.glob(pattern) if p.is_file()]
            if not matches:
                res.findings.append(Finding("error", "data-missing", "-",
                                            f"[data].files pattern {pattern!r} matched nothing"))
            files.update(p.relative_to(self.src).as_posix() for p in matches)
        res.data_files = sorted(files)

    def _check_tracked(self, res: Result) -> None:
        try:
            out = subprocess.run(["git", "-C", str(self.src), "ls-files", "--", "*.py"], capture_output=True,
                                 text=True, timeout=60, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return
        tracked = {Path(line.strip()).stem for line in out.splitlines() if line.strip() and "/" not in line.strip()}
        if not tracked:
            return
        res.untracked = [n for n in res.closure if n not in tracked]
        for n in res.untracked:
            res.findings.append(Finding("warning", "untracked", n,
                                        f"{n}.py is bundled but not tracked by git, so CI will not have it"))


# --------------------------------------------------------------------------- bundle output
def git_sha(src: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=30, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no", "--", "."],
                                    capture_output=True, text=True, timeout=120, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("GITHUB_SHA", ""), False


def read_app_version(src: Path) -> str:
    try:
        tree = ast.parse(read_source(Path(src) / "app_version.py"))
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == "APP_VERSION" for t in node.targets)):
            return str(node.value.value)
    return ""


def bundle_digest(files: dict) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode("utf-8"))
    return h.hexdigest()


def write_bundle(res: Result, out: Path, src: Path, manifest: Manifest) -> dict:
    out, src = Path(out).resolve(), Path(src).resolve()
    if out == src or out in src.parents:
        raise SystemExit(f"refusing to write the bundle into {out}")
    if out.exists():
        if any(out.iterdir()) and not (out / BUNDLE_INFO_NAME).exists():
            raise SystemExit(f"{out} is not empty and has no {BUNDLE_INFO_NAME}; refusing to wipe it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files: dict[str, str] = {}
    for rel in [f"{n}.py" for n in res.closure] + list(res.data_files):
        data = (src / rel).read_bytes()
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        files[rel] = sha256_bytes(data)
    sha, dirty = git_sha(src)
    info = {
        "FORMAT": 1,
        "COLLECTOR_VERSION": COLLECTOR_VERSION,
        "BUILD_VERSION": read_app_version(src),
        "GIT_SHA": sha,
        "GIT_DIRTY": dirty,
        "GENERATED_AT": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "COLLECTED_WITH_PYTHON": sys.version.split()[0],
        "BUNDLE_SHA256": bundle_digest(files),
        "MODULE_COUNT": len(res.closure),
        "TOTAL_BYTES": sum(len((src / r).read_bytes()) for r in files),
        "ENTRY_MODULES": tuple(manifest.entries),
        "MODULES": tuple(res.closure),
        "DATA_FILES": tuple(res.data_files),
        "EXCLUDED": dict(sorted(manifest.excluded.items())),
        "GUI_TAINTED": dict(sorted(res.tainted.items())),
        "BLOCKED_EDGES": dict(sorted(res.edge_status.items())),
        "SPAWN_SITES": {k: {"apis": v["apis"], "count": v["count"], "lines": v["lines"]}
                        for k, v in sorted(res.spawn_sites.items())},
        "FILES": dict(sorted(files.items())),
    }
    lines = ["# Generated by src/mobile/tools/collect_backend.py. Do not edit.",
             "# Mobile backend bundle manifest: read by the About screen and by `collect_backend.py --verify`.", ""]
    lines += [f"{k} = {v!r}" for k, v in info.items()]
    (out / BUNDLE_INFO_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info


def load_bundle_info(out: Path) -> dict:
    tree = ast.parse((Path(out) / BUNDLE_INFO_NAME).read_text(encoding="utf-8"))
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}


def verify_bundle(out: Path, src: Path | None = None) -> list:
    out = Path(out)
    if not (out / BUNDLE_INFO_NAME).exists():
        return [f"{out / BUNDLE_INFO_NAME} is missing (run collect_backend.py first)"]
    try:
        info = load_bundle_info(out)
    except (OSError, SyntaxError, ValueError) as e:
        return [f"cannot read {BUNDLE_INFO_NAME}: {e}"]
    expected: dict = info.get("FILES", {})
    actual: dict = {}
    for p in sorted(out.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(out).as_posix()
        if rel != BUNDLE_INFO_NAME:
            actual[rel] = sha256_bytes(p.read_bytes())
    problems = [f"missing from bundle: {r}" for r in sorted(set(expected) - set(actual))]
    problems += [f"unexpected file in bundle: {r}" for r in sorted(set(actual) - set(expected))]
    problems += [f"hash mismatch: {r}" for r in sorted(set(expected) & set(actual)) if expected[r] != actual[r]]
    if bundle_digest(expected) != info.get("BUNDLE_SHA256"):
        problems.append("BUNDLE_SHA256 does not match FILES")
    if src is not None and Path(src).exists():
        for rel, digest in sorted(expected.items()):
            sp = Path(src) / rel
            if not sp.exists():
                problems.append(f"stale bundle: {rel} no longer exists in {src}")
            elif sha256_bytes(sp.read_bytes()) != digest:
                problems.append(f"stale bundle: {rel} differs from src (re-run collect_backend.py)")
    return problems


# --------------------------------------------------------------------------- report
def _md(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_report(res: Result, manifest: Manifest) -> str:
    total_bytes = sum(res.infos[n].size for n in res.closure if n in res.infos)
    total_lines = sum(res.infos[n].lines for n in res.closure if n in res.infos)
    errs, warns = res.errors, res.warnings
    status_counts = {k: sum(1 for v in res.edge_status.values() if v.startswith(k))
                     for k in ("guarded", "baselined", "FAIL")}
    o: list[str] = []
    w = o.append
    w("# Mobile backend collector report")
    w("")
    w(f"- Python `{res.python}`, collector v{COLLECTOR_VERSION}, manifest `{Path(res.manifest).name}`")
    w(f"- Bundled modules: **{len(res.closure)}** ({total_bytes / 1e6:.1f} MB, {total_lines:,} lines); "
      f"data files: {len(res.data_files)}")
    w(f"- Entry modules: {len(manifest.entries)}; app files scanned: {len(res.app_modules)}")
    w(f"- Excluded modules: {len(manifest.excluded)}; GUI-tainted modules reached by an edge: {len(res.tainted)}")
    w(f"- Edges into GUI/excluded code: {len(res.edge_status)} ({status_counts['guarded']} guarded, "
      f"{status_counts['baselined']} baselined, {status_counts['FAIL']} failing)")
    w(f"- Process-spawn sites: {len(res.spawn_sites)} functions, "
      f"{sum(v['count'] for v in res.spawn_sites.values())} call/reference sites")
    w(f"- Result: **{'FAIL' if errs else 'PASS'}** with {len(errs)} errors and {len(warns)} warnings")
    w("")
    if errs or warns:
        w("## Errors and warnings")
        w("")
        w("| level | check | module | lines | message |")
        w("|---|---|---|---|---|")
        for f in errs + warns:
            w(f"| {f.level} | {f.code} | `{f.module}` | {', '.join(map(str, f.lines[:8]))} | {_md(f.message)} |")
        w("")
    notes = [f for f in res.findings if f.level == "info"]
    if notes:
        w("## Notes")
        w("")
        for f in notes:
            w(f"- `{f.module}` ({f.code}): {_md(f.message)}")
        w("")

    w("## Edges into GUI and excluded code")
    w("")
    w("| edge | status | kinds | lines | why the target is blocked |")
    w("|---|---|---|---|---|")
    groups: dict[str, list] = {}
    for e in res.blocked_edges:
        groups.setdefault(f"{e.src}->{e.dst}", []).append(e)
    for key in sorted(groups):
        sites = groups[key]
        dst = key.split("->")[1]
        kinds = ", ".join(sorted({s.kind + (" guarded" if s.guarded else "") for s in sites}))
        lines = ", ".join(str(x) for x in sorted({s.lineno for s in sites})[:12])
        why = manifest.excluded.get(dst) or res.tainted.get(dst) or ("GUI package" if dst in manifest.gui_packages else "")
        w(f"| `{key}` | {res.edge_status.get(key, '?')} | {kinds} | {lines} | {_md(why)} |")
    w("")

    w("## Process-spawn sites (ratchet)")
    w("")
    w("| site | APIs | sites | lines | status | baseline note |")
    w("|---|---|---|---|---|---|")
    for key, rec in sorted(res.spawn_sites.items()):
        note = manifest.spawn_sites.get(key, {}).get("reason", "")
        w(f"| `{key}` | {', '.join(rec['apis'])} | {rec['count']} | {', '.join(map(str, rec['lines'][:12]))} | "
          f"{rec['status']} | {_md(note)} |")
    w("")

    w("## Third-party imports")
    w("")
    w("| distribution | status | import names | module-scope unguarded sites | other sites |")
    w("|---|---|---|---|---|")
    for dist, rec in sorted(res.thirdparty.items()):
        eager = rec.get("eager_sites", [])
        other = [s for s in rec.get("sites", []) if s not in eager]
        reason = f" ({rec['reason']})" if rec.get("reason") else ""
        w(f"| `{dist}` | {rec.get('status', '')}{_md(reason)} | {_md(', '.join(sorted(rec.get('imports', ()))))} | "
          f"{_md('; '.join(eager[:4]) + (' ...' if len(eager) > 4 else ''))} | {len(other)} |")
    w("")
    risky = [(dist, rec.get("status", ""), s) for dist, rec in sorted(res.thirdparty.items())
             if rec.get("status") not in ("pinned", "transitive")
             for s in rec.get("sites", []) if "guarded" not in s and s not in rec.get("eager_sites", [])]
    if risky:
        w("## Unguarded lazy/conditional imports of packages missing on device")
        w("")
        w("These are allowed: they only fail if the code path runs. Review them when the feature is enabled on mobile.")
        w("")
        w("| distribution | status | site |")
        w("|---|---|---|")
        for dist, status, site in risky:
            w(f"| `{dist}` | {status} | `{site}` |")
        w("")

    w("## GUI-tainted modules reached by an edge")
    w("")
    for name, why in sorted(res.tainted.items()):
        w(f"- `{name}`: {_md(why)}")
    w("")
    w("## Excluded modules")
    w("")
    for name, why in sorted(manifest.excluded.items()):
        w(f"- `{name}`: {_md(why)}")
    w("")

    w("## Bundled modules")
    w("")
    w("| module | KB | lines | spawn sites | `__file__` uses | why bundled |")
    w("|---|---|---|---|---|---|")
    for name in res.closure:
        mi = res.infos.get(name)
        if mi:
            w(f"| `{name}` | {mi.size / 1024:.0f} | {mi.lines} | {len(mi.spawns) or ''} | "
              f"{len(mi.dunder_file_lines) or ''} | {_md(res.reasons.get(name, ''))} |")
    w("")
    if res.data_files:
        w("## Data files")
        w("")
        w("\n".join(f"- `{rel}`" for rel in res.data_files))
        w("")
    return "\n".join(o) + "\n"


def result_to_json(res: Result) -> dict:
    tp = {}
    for k, v in res.thirdparty.items():
        v = dict(v)
        v["imports"] = sorted(v.get("imports", ()))
        tp[k] = v
    return {
        "python": res.python,
        "closure": res.closure,
        "reasons": res.reasons,
        "tainted": res.tainted,
        "edge_status": res.edge_status,
        "blocked_edges": [asdict(e) for e in res.blocked_edges],
        "spawn_sites": res.spawn_sites,
        "thirdparty": tp,
        "findings": [asdict(f) for f in res.findings],
        "data_files": res.data_files,
        "app_modules": res.app_modules,
        "dunder_file": {n: res.infos[n].dunder_file_lines for n in res.closure if n in res.infos},
        "file_refs": {n: res.infos[n].file_refs for n in res.closure if n in res.infos and res.infos[n].file_refs},
    }


# --------------------------------------------------------------------------- CLI
def _path(p: str | None, default: Path | None) -> Path | None:
    if p is None:
        return default
    path = Path(p)
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect the Glossarion backend closure for the mobile bundle "
                                             "(see the module docstring for the rules).")
    ap.add_argument("--src", help="repo src/ directory (default: two levels above src/mobile/tools)")
    ap.add_argument("--manifest", help="policy file (default: src/mobile/backend_manifest.toml)")
    ap.add_argument("--pyproject", help="pins (default: src/mobile/pyproject.toml)")
    ap.add_argument("--out", help="bundle directory (default: src/mobile/app/backend)")
    ap.add_argument("--report", help="write the Markdown report here (also appended to $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--json", dest="json_out", help="write the machine-readable result here")
    ap.add_argument("--check", action="store_true", help="analyse only; do not write the bundle")
    ap.add_argument("--verify", action="store_true",
                    help="verify --out against its _bundle_info.py (and against src/ when present); no analysis")
    ap.add_argument("--no-cache", action="store_true", help="ignore the per-file analysis cache in build/")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print the summary line")
    args = ap.parse_args(argv)

    src = _path(args.src, DEFAULT_SRC).resolve()
    out = _path(args.out, DEFAULT_OUT).resolve()

    if args.verify:
        # Freshness against src/ when it is the repo (or was given explicitly); CI jobs that only
        # downloaded the bundle artifact still get the integrity check.
        fresh_src = src if (args.src or (src / "TransateKRtoEN.py").exists()) else None
        problems = verify_bundle(out, fresh_src)
        if problems:
            print(f"VERIFY FAILED for {out}:", file=sys.stderr)
            for p in problems[:200]:
                print(f"  - {p}", file=sys.stderr)
            return 1
        info = load_bundle_info(out)
        print(f"verify OK: {out} ({info.get('MODULE_COUNT')} modules, bundle sha256 "
              f"{str(info.get('BUNDLE_SHA256', ''))[:16]}..., git {str(info.get('GIT_SHA', ''))[:10]})")
        return 0

    manifest_path = _path(args.manifest, DEFAULT_MANIFEST)
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"cannot read manifest {manifest_path}: {e}", file=sys.stderr)
        return 2
    pins = load_pins(_path(args.pyproject, DEFAULT_PYPROJECT))
    cache = None if args.no_cache else DEFAULT_CACHE
    res = Collector(src, manifest, pins, mobile_dir=manifest_path.resolve().parent, cache_path=cache).run()

    report = render_report(res, manifest)
    if args.report:
        rp = _path(args.report, None)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding="utf-8")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            try:
                with open(summary, "a", encoding="utf-8") as f:
                    f.write(report)
            except OSError:
                pass
    if args.json_out:
        jp = _path(args.json_out, None)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(result_to_json(res), indent=1, sort_keys=True), encoding="utf-8")

    if not args.quiet:
        for f in res.errors + res.warnings:
            loc = f"{f.module}:{f.lines[0]}" if f.lines else f.module
            print(f"{f.level.upper():7} {f.code:24} {loc}: {f.message}")
    print(f"collect_backend: {len(res.closure)} modules, {len(res.errors)} errors, {len(res.warnings)} warnings"
          + (f"; report {args.report}" if args.report else ""))
    if res.errors:
        return 1
    if not args.check:
        info = write_bundle(res, out, src, manifest)
        problems = verify_bundle(out, src)
        if problems:
            print("bundle verification failed right after writing:", *problems[:20], sep="\n  ", file=sys.stderr)
            return 1
        print(f"wrote {info['MODULE_COUNT']} modules ({info['TOTAL_BYTES'] / 1e6:.1f} MB) to {out}; "
              f"BUNDLE_SHA256 {info['BUNDLE_SHA256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
