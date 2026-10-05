#!/usr/bin/env python3
"""Tier D: differential fuzz of moved desktop methods (legacy frozen copy vs new code).

For one moved method the driver

1. derives the *read set* by AST from both implementations and their ``self.``
   call closure: ``self.<attr>`` / ``getattr(self, 'x')`` / ``hasattr(self, 'x')``
   reads (with the widget API used on them), ``self.config.get('k', d)`` /
   ``self.config['k']`` / ``'k' in self.config`` keys (with their literal
   defaults), ``os.environ`` / ``os.getenv`` reads, and the Qt names touched;
2. generates >= 500 seeded owner states: a base (an empty owner, or a desktop
   owner booted from one of the golden scenarios by the frozen legacy boot) plus
   perturbations of read-set items - attributes absent or one of
   ``SCALARS = (True, False, '0', '1', '', 'abc', 0, 5, -1, None)``, values seen
   in other bases, or fake widgets; config keys absent / their call-site default
   / an alternate; env variables unset or set;
3. runs the legacy side and the new side on identical copies of each state
   inside one ``normalize.CaptureContext`` sandbox, resetting the process
   between runs (scrubbed env, cwd, argv, ``large_env`` store, uuid counter,
   ``random`` seed, tempfile names, sandbox files), with recorders for GUI-only
   methods, ``UnifiedClient`` pools, backend entry points, Qt classes and
   process/beep/browser side effects;
4. compares return value (and top-level dict key order), raised exception type,
   ``os.environ`` delta (incl. ``large_env``), final ``self.config``, final
   owner attributes, recorded calls, backend-stub captures, stdout, sandbox file
   changes (JSON decrypted), ``sys.argv`` and cwd. Text is compared with memory
   addresses and traceback frames masked (frames name different files per side).

Safety: ``open()`` refuses integer "paths" while fuzzing (a fuzzed ``True``/``1``
would otherwise open and close the test process's stdout), processes are never
spawned, Qt classes are recording stubs (no dialog can block or abort), every
live module's ``CONFIG_FILE`` points into the sandbox, and a tripwire reports any
change to ``src/`` config/key files.

The *new side* is the working-tree desktop: every name is resolved through
``TranslatorGUI``'s Python MRO (shared mixins listed first, desktop hook
overrides in the class body), exactly what the desktop runs. ``mode='mixins'``
resolves through the shared mixins only (GUI-free hook defaults; debugging aid
when PySide6 is unavailable).

Low-level API (used by the toy self-tests and by later milestones)::

    space = StateSpace(read_set, bases, arg_gen=..., seed=...)
    with FuzzContext('label', backend=False) as ctx:
        report = differential_fuzz(ctx, Side('legacy', OldCls, 'm'), Side('new', NewCls, 'm'), space)

High-level::

    report = fuzz_moved(moved_functions.get('_get_environment_variables'))

Environment: ``PARITY_FUZZ_STATES`` (default 500), ``PARITY_FUZZ_SEED`` is the
``--seed`` default, ``PARITY_FUZZ_TRACE=1`` prints each state to stderr before it
runs (finds a state that hangs or kills the process).

CLI (repository root)::

    python tests/parity/fuzz_moved.py --list
    python tests/parity/fuzz_moved.py _get_environment_variables [--states 500] [--seed N]
    python tests/parity/fuzz_moved.py --all [--legacy-vs-legacy]
    python tests/parity/fuzz_moved.py _get_output_mode --index 17     # replay one state verbosely
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import itertools
import json
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import types
import zlib
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import fakes, moved_functions, normalize  # noqa: E402

DEFAULT_STATES = 500
DEFAULT_SEED = int(os.environ.get("PARITY_FUZZ_SEED", "20261005"))
#: PARITY_FUZZ_TRACE=1 prints every state to stderr before it runs (read at import:
#: the fuzz sandbox scrubs os.environ).
TRACE = os.environ.get("PARITY_FUZZ_TRACE") == "1"
SCALARS = (True, False, "0", "1", "", "abc", 0, 5, -1, None)
ENV_VALUES = ("0", "1", "", "abc", "5", "true")
TEXT_VALUES = ("", "0", "1", "5", "-1", "abc", "2.5", " 3 ", "1-5", "3-9", "1,000", "true", "Off")
#: fields compared between the two sides (``exc_type`` first: the most telling one)
COMPARED_FIELDS = ("exc_type", "result", "env", "config", "attrs", "calls", "backend",
                   "stdout", "fs", "argv", "cwd")


class _Missing:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<absent>"

    def __reduce__(self):
        return (_Missing, ())


MISSING = _Missing()
NO_DEFAULT = "<no-default>"


class Unavailable(Exception):
    """A prerequisite of the high-level driver is missing (skip, not fail)."""


# ---------------------------------------------------------------------------
# AST read sets
# ---------------------------------------------------------------------------

#: widget method -> fake kind (the Qt surface the desktop code touches, see fakes.py)
WIDGET_API = {
    "text": "line", "setText": "line", "clear": "line", "setPlaceholderText": "line",
    "isChecked": "check", "setChecked": "check",
    "toPlainText": "plain", "setPlainText": "plain",
    "currentData": "combo", "currentText": "combo", "currentIndex": "combo", "findData": "combo",
    "findText": "combo", "setCurrentIndex": "combo", "setCurrentText": "combo", "count": "combo",
    "itemData": "combo", "itemText": "combo",
}
_QT_NAME_RE = re.compile(r"^(Q[A-Z]\w*|Qt)$")
_SELF_NAMES = ("self", "cls")


@dataclass
class ReadSet:
    attrs: set = field(default_factory=set)
    attr_defaults: dict = field(default_factory=dict)   # attr -> [literal getattr defaults]
    widgets: dict = field(default_factory=dict)         # attr -> {fake kinds}
    config: dict = field(default_factory=dict)          # key -> [literal defaults | NO_DEFAULT]
    env: set = field(default_factory=set)
    strings: set = field(default_factory=set)           # identifier-like string constants
    calls: set = field(default_factory=set)             # self.<name>(...) callees
    writes: set = field(default_factory=set)
    qt_names: set = field(default_factory=set)
    qt_modules: set = field(default_factory=set)
    functions: list = field(default_factory=list)       # closure members analysed
    #: (callee expression, positional index | keyword name) of calls receiving ``self.config``
    config_passes: list = field(default_factory=list)
    local_imports: dict = field(default_factory=dict)   # local name -> (module, attribute)

    def update(self, other: "ReadSet") -> "ReadSet":
        self.attrs |= other.attrs
        for k, v in other.attr_defaults.items():
            self.attr_defaults.setdefault(k, []).extend(v)
        for k, v in other.widgets.items():
            self.widgets.setdefault(k, set()).update(v)
        for k, v in other.config.items():
            self.config.setdefault(k, []).extend(v)
        self.env |= other.env
        self.strings |= other.strings
        self.calls |= other.calls
        self.writes |= other.writes
        self.qt_names |= other.qt_names
        self.qt_modules |= other.qt_modules
        for f in other.functions:
            if f not in self.functions:
                self.functions.append(f)
        return self

    def drop_provided(self, provided) -> "ReadSet":
        provided = set(provided)
        self.attrs -= provided
        self.attrs.discard("config")
        # harness internals (FakeState's recorder) are never owner state to perturb
        self.attrs = {a for a in self.attrs if not a.startswith("_parity_")}
        for name in list(self.widgets):
            if name in provided:
                del self.widgets[name]
        for name in list(self.attr_defaults):
            if name in provided:
                del self.attr_defaults[name]
        return self

    def sizes(self) -> dict:
        return {"attrs": len(self.attrs), "widgets": len(self.widgets), "config": len(self.config),
                "env": len(self.env), "functions": len(self.functions)}


def _is_self(node) -> bool:
    return isinstance(node, ast.Name) and node.id in _SELF_NAMES


def _const_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return NO_DEFAULT


def _is_config_expr(node, aliases) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "config" and _is_self(node.value):
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
            and len(node.args) >= 2 and _is_self(node.args[0]) and _const_str(node.args[1]) == "config"):
        return True
    if isinstance(node, ast.Name) and node.id in aliases:
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
        return _is_config_expr(node.values[0], aliases)
    return False


def _is_environ(node) -> bool:
    text = None
    if isinstance(node, ast.Attribute):
        text = ast.unparse(node)
    elif isinstance(node, ast.Name):
        text = node.id
    return text in ("os.environ", "environ", "_os.environ")


def _is_env_getter(func) -> bool:
    if isinstance(func, ast.Attribute):
        if func.attr == "get" and _is_environ(func.value):
            return True
        text = ast.unparse(func)
        return text in ("os.getenv", "_os.getenv", "large_env.get_env")
    return isinstance(func, ast.Name) and func.id == "getenv"


def parse_function_source(source: str) -> ast.Module:
    """Parse a method's source even when ``textwrap.dedent`` cannot remove its indentation
    (multi-line string literals at column 0 inside an indented method)."""
    text = textwrap.dedent(source)
    try:
        return ast.parse(text)
    except IndentationError:
        return ast.parse("if True:\n" + source)


def analyze_source(source: str, config_aliases=()) -> ReadSet:
    """Read set of one function source (nested functions and lambdas included).

    *config_aliases* are local names known to hold the owner's config (used when a
    module helper receiving ``self.config`` is analysed, see ``closure_read_set``).
    """
    tree = parse_function_source(source)
    rs = ReadSet()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    aliases: set = set(config_aliases)
    for _ in range(2):  # chained aliases (cfg = self.config; c2 = cfg)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_config_expr(node.value, aliases):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        aliases.add(t.id)

    def widget_use(node, attr_name):
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            grand = parents.get(parent)
            if isinstance(grand, ast.Call) and grand.func is parent and parent.attr in WIDGET_API:
                rs.widgets.setdefault(attr_name, set()).add(WIDGET_API[parent.attr])

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_self(node.value):
            name = node.attr
            if name == "config":
                continue
            parent = parents.get(node)
            if isinstance(node.ctx, ast.Store):
                rs.writes.add(name)
                if isinstance(parent, ast.AugAssign):
                    rs.attrs.add(name)
                continue
            if isinstance(node.ctx, ast.Del):
                rs.writes.add(name)
                continue
            if isinstance(parent, ast.Call) and parent.func is node:
                rs.calls.add(name)
                continue
            rs.attrs.add(name)
            widget_use(node, name)
        elif isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id in ("getattr", "hasattr", "setattr", "delattr")
                    and len(node.args) >= 2 and _is_self(node.args[0])):
                name = _const_str(node.args[1])
                if name is not None and name != "config":
                    if func.id in ("getattr", "hasattr"):
                        rs.attrs.add(name)
                        if func.id == "getattr" and len(node.args) >= 3:
                            default = _literal(node.args[2])
                            if default is not NO_DEFAULT:
                                rs.attr_defaults.setdefault(name, []).append(default)
                            widget_use(node, name)
                        elif func.id == "getattr":
                            widget_use(node, name)
                    else:
                        rs.writes.add(name)
            if (isinstance(func, ast.Attribute) and func.attr in ("get", "setdefault", "pop")
                    and _is_config_expr(func.value, aliases) and node.args):
                key = _const_str(node.args[0])
                if key is not None:
                    default = _literal(node.args[1]) if len(node.args) >= 2 else NO_DEFAULT
                    rs.config.setdefault(key, []).append(default)
            if _is_env_getter(func) and node.args:
                key = _const_str(node.args[0])
                if key is not None:
                    rs.env.add(key)
            if isinstance(func, (ast.Name, ast.Attribute)) and not (
                    isinstance(func, ast.Attribute) and _is_config_expr(func.value, aliases)):
                for i, arg in enumerate(node.args):
                    if _is_config_expr(arg, aliases):
                        rs.config_passes.append((ast.unparse(func), i))
                for kw in node.keywords:
                    if kw.arg and _is_config_expr(kw.value, aliases):
                        rs.config_passes.append((ast.unparse(func), kw.arg))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = _const_str(node.slice)
            if key is not None:
                if _is_config_expr(node.value, aliases):
                    rs.config.setdefault(key, []).append(NO_DEFAULT)
                elif _is_environ(node.value):
                    rs.env.add(key)
        elif isinstance(node, ast.Compare):
            left = _const_str(node.left)
            if left is not None:
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)):
                        if _is_config_expr(comparator, aliases):
                            rs.config.setdefault(left, []).append(NO_DEFAULT)
                        elif _is_environ(comparator):
                            rs.env.add(left)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.isidentifier() and len(node.value) <= 80:
                rs.strings.add(node.value)
        elif isinstance(node, ast.Name) and _QT_NAME_RE.match(node.id):
            rs.qt_names.add(node.id)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("PySide6"):
            rs.qt_modules.add(node.module)
            for alias in node.names:
                if _QT_NAME_RE.match(alias.asname or alias.name):
                    rs.qt_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                rs.local_imports[alias.asname or alias.name] = (node.module, alias.name)
    return rs


def unwrap(obj):
    """Function behind staticmethod / classmethod / property / bound method."""
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    if isinstance(obj, property):
        return obj.fget
    if isinstance(obj, types.MethodType):
        return obj.__func__
    return obj


_SOURCE_CACHE: dict = {}


def source_of(fn) -> str | None:
    fn = unwrap(fn)
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    key = (code.co_filename, code.co_firstlineno, code.co_name)
    if key not in _SOURCE_CACHE:
        try:
            _SOURCE_CACHE[key] = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            _SOURCE_CACHE[key] = None
    return _SOURCE_CACHE[key]


_ANALYSIS_CACHE: dict = {}


def analyze_function(fn, config_aliases=()) -> ReadSet:
    src = source_of(fn)
    if src is None:
        return ReadSet()
    key = (src, tuple(sorted(config_aliases)))
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = analyze_source(src, config_aliases)
    return copy.deepcopy(_ANALYSIS_CACHE[key])


def _resolve_callee(fn, expr: str, local_imports: dict):
    """Python function named by *expr* inside *fn* (globals, local imports, dotted modules)."""
    parts = expr.split(".")
    head = parts[0]
    if head in _SELF_NAMES:
        return None
    globs = getattr(unwrap(fn), "__globals__", {}) or {}
    obj = globs.get(head, MISSING)
    if obj is MISSING and head in local_imports:
        module_name, attr = local_imports[head]
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                return None
        obj = getattr(module, attr, MISSING)
    if obj is MISSING:
        return None
    for part in parts[1:]:
        obj = getattr(obj, part, MISSING)
        if obj is MISSING:
            return None
    obj = unwrap(obj)
    return obj if isinstance(obj, types.FunctionType) else None


def config_flow_read_set(fn, part: ReadSet, depth: int = 3) -> ReadSet:
    """Config keys / env reads of module helpers that receive ``self.config`` from *fn*
    (e.g. ``apply_key_pools_to_runtime(self.config)``), followed *depth* calls deep."""
    out = ReadSet()
    if depth <= 0:
        return out
    for expr, where in part.config_passes:
        callee = _resolve_callee(fn, expr, part.local_imports)
        if callee is None:
            continue
        try:
            params = list(inspect.signature(callee).parameters.values())
        except (TypeError, ValueError):
            continue
        if isinstance(where, int):
            if where >= len(params) or params[where].kind not in (
                    inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                continue
            pname = params[where].name
        else:
            pname = where if any(p.name == where for p in params) else None
        if not pname:
            continue
        sub = analyze_function(callee, (pname,))
        found = ReadSet(config=dict(sub.config), env=set(sub.env))
        out.update(found)
        out.update(config_flow_read_set(callee, sub, depth - 1))
    return out


def class_lookup(cls):
    """name -> class attribute of *cls* (MRO), or None."""
    def lookup(name):
        for klass in cls.__mro__:
            if name in klass.__dict__:
                return klass.__dict__[name]
        return None
    return lookup


def closure_read_set(start_names, lookup) -> ReadSet:
    """Read set of *start_names* and every ``self.<method>`` they reach (via *lookup*)."""
    rs = ReadSet()
    seen = []
    queue = list(start_names)
    provided = set()
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        obj = lookup(name)
        if obj is None:
            continue
        provided.add(name)
        fn = unwrap(obj)
        if not isinstance(fn, types.FunctionType):
            continue
        seen.append(name)
        part = analyze_function(fn)
        part.functions = [name]
        rs.update(part)
        rs.update(config_flow_read_set(fn, part))
        for ref in sorted(part.calls | part.attrs):
            if ref not in seen and lookup(ref) is not None:
                queue.append(ref)
    for name in list(rs.attrs) + list(rs.calls):
        if lookup(name) is not None:
            provided.add(name)
    return rs.drop_provided(provided)


def index_call_sites(sources, limit: int = 40) -> dict:
    """{method: [literal (args, kwargs)]} of every ``self.<method>(...)`` call in *sources*."""
    out: dict = {}
    for src in sources:
        if not src:
            continue
        try:
            tree = parse_function_source(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            target = node.func.value
            via_class = isinstance(target, ast.Name) and target.id == "TranslatorGUI"
            if not (_is_self(target) or via_class):
                continue
            args = node.args[1:] if via_class else node.args
            try:
                values = tuple(ast.literal_eval(a) for a in args)
                kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
            except Exception:
                continue
            if len(node.keywords) != len(kwargs):
                continue
            sites = out.setdefault(node.func.attr, [])
            item = (values, kwargs)
            if item not in sites and len(sites) < limit:
                sites.append(item)
    return out


def harvest_call_sites(sources, method_name: str, limit: int = 40) -> list:
    """Literal ``(args, kwargs)`` of ``self.<method_name>(...)`` calls found in *sources*."""
    return index_call_sites([s for s in sources if s and method_name in s], limit).get(method_name, [])


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

_IMMUTABLE_TYPES = frozenset({str, int, float, bool, type(None), bytes, complex, _Missing})
_WIDGET_TYPES = (fakes.FakeWidget, fakes.FakeLineEdit, fakes.FakeCheck, fakes.FakeTextEdit, fakes.FakeCombo)


def fast_copy(value):
    """Deep copy specialised for literal state (much faster than copy.deepcopy)."""
    t = type(value)
    if t in _IMMUTABLE_TYPES:
        return value
    if t is dict:
        return {k: fast_copy(v) for k, v in value.items()}
    if t is list:
        return [fast_copy(v) for v in value]
    if t is tuple:
        return tuple(fast_copy(v) for v in value)
    if t is set:
        return {fast_copy(v) for v in value}
    if isinstance(value, _WIDGET_TYPES):
        clone = copy.copy(value)
        if isinstance(clone, fakes.FakeCombo):
            clone._items = list(clone._items)
        return clone
    return copy.deepcopy(value)


def is_plain(value, depth=0) -> bool:
    if depth > 12:
        return False
    t = type(value)
    if t in _IMMUTABLE_TYPES:
        return True
    if t in (list, tuple, set, frozenset):
        return all(is_plain(v, depth + 1) for v in value)
    if t is dict:
        return all(is_plain(k, depth + 1) and is_plain(v, depth + 1) for k, v in value.items())
    return isinstance(value, _WIDGET_TYPES)


def map_strings(value, fn):
    """Apply *fn* to every string inside a literal value (fake widgets are copied)."""
    if isinstance(value, str):
        return fn(value)
    t = type(value)
    if t is dict:
        return {map_strings(k, fn): map_strings(v, fn) for k, v in value.items()}
    if t is list:
        return [map_strings(v, fn) for v in value]
    if t is tuple:
        return tuple(map_strings(v, fn) for v in value)
    if t in (set, frozenset):
        return t(map_strings(v, fn) for v in value)
    if isinstance(value, _WIDGET_TYPES):
        clone = copy.copy(value)
        for k, v in list(vars(clone).items()):
            if isinstance(v, (str, list, tuple)):
                setattr(clone, k, map_strings(v, fn))
        return clone
    return value


def dedup(values) -> list:
    out, seen = [], set()
    for v in values:
        key = repr(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
#: traceback frames name the frozen legacy file vs translator_gui.py / the mixin module and
#: their number changes when a hook adds a call level: frames are dropped, the
#: "Traceback (most recent call last):" header and the final exception line stay.
_TB_FRAME_RE = re.compile(r'(?m)^ *File "[^"\n]*", line \d+[^\n]*\n(?:(?: {4,}|\t)[^\n]*\n)*')
_FRAME_RE = re.compile(r'File "[^"\n]+", line \d+')
#: the name suggestion a printed AttributeError/NameError gets (3.10+) depends on how many
#: attributes the owner class has (none above CPython's candidate limit): side-specific
_SUGGESTION_RE = re.compile(r"\. Did you mean: '[^'\n]*'\?")


def mask(text: str) -> str:
    """Side-independent text: memory addresses, traceback frames and name suggestions masked."""
    if "0x" in text:
        text = _ADDR_RE.sub("0x?", text)
    if 'File "' in text:
        text = _TB_FRAME_RE.sub("", text)
        text = _FRAME_RE.sub('File "<src>", line <n>', text)
    if "Did you mean" in text:
        text = _SUGGESTION_RE.sub("", text)
    return text


def canon(value, owner=None, _depth=0):
    """Comparable literal form of a captured value (identity-free, owner-class-free)."""
    if _depth > 40:
        return "<max-depth>"
    t = type(value)
    if t is str:
        return mask(value) if ("0x" in value or 'File "' in value or "Did you mean" in value) else value
    if t is int or t is bool or value is None:
        return value
    if t is float:  # tagged: 5.0 must not compare equal to 5 / True
        return {"__float__": repr(value)}
    if owner is not None and value is owner:
        return "<owner>"
    d = _depth + 1
    if isinstance(value, dict):
        return {canon(k, owner, d): canon(v, owner, d) for k, v in value.items()}
    if isinstance(value, list):
        return [canon(v, owner, d) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__dataclass__": type(value).__name__,
                **{f.name: canon(getattr(value, f.name, None), owner, d) for f in dataclasses.fields(value)}}
    if isinstance(value, tuple):
        return tuple(canon(v, owner, d) for v in value)
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted((canon(v, owner, d) for v in value), key=repr)}
    if isinstance(value, bytes):
        return {"__bytes__": value.hex() if len(value) <= 256 else zlib.crc32(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, collections.deque):
        return {"__deque__": [canon(v, owner, d) for v in value], "maxlen": value.maxlen}
    if isinstance(value, _Missing):
        return "<absent>"
    if isinstance(value, QtStub):
        return value.describe()
    if isinstance(value, _WIDGET_TYPES + (fakes.FakeSignal,)):
        return canon(value.describe(), owner, d)
    if isinstance(value, types.MethodType):
        target = "<owner>" if owner is not None and value.__self__ is owner else type(value.__self__).__name__
        return f"<method {getattr(value.__func__, '__name__', '?')} of {target}>"
    if isinstance(value, functools.partial):
        return {"__partial__": canon(value.func, owner, d), "args": canon(value.args, owner, d),
                "kwargs": canon(value.keywords, owner, d)}
    if isinstance(value, type):
        return f"<class {value.__name__}>"
    if callable(value) and hasattr(value, "__name__"):
        return f"<function {value.__name__}>"
    init_kwargs = getattr(value, "_parity_init_kwargs", None)
    if isinstance(init_kwargs, dict):
        return {"__client__": type(value).__name__, "kwargs": canon(init_kwargs, owner, d)}
    return f"<object {type(value).__name__}>"


# ---------------------------------------------------------------------------
# Qt and side-effect stubs
# ---------------------------------------------------------------------------


class QtStub:
    """Stands in for a Qt class/enum while fuzzing: every call is recorded, never shown."""

    __slots__ = ("_path", "_recorder", "_children")

    def __init__(self, path: str, recorder):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_children", {})

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        children = object.__getattribute__(self, "_children")
        if name not in children:
            children[name] = QtStub(f"{self._path}.{name}", self._recorder)
        return children[name]

    def __setattr__(self, name, value):
        self._children[name] = value

    def __call__(self, *args, **kwargs):
        self._recorder.record(f"qt:{self._path}", args, kwargs)
        return QtStub(f"{self._path}()", self._recorder)

    def __bool__(self):
        return True

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __int__(self):
        return 0

    def __index__(self):
        return 0

    def _combine(self, other, op):
        other_path = other._path if isinstance(other, QtStub) else repr(other)
        return QtStub(f"({self._path}{op}{other_path})", self._recorder)

    def __or__(self, other):
        return self._combine(other, "|")

    __ror__ = __or__

    def __and__(self, other):
        return self._combine(other, "&")

    __rand__ = __and__

    def __invert__(self):
        return QtStub(f"~{self._path}", self._recorder)

    def __repr__(self):
        return f"<qt {self._path}>"

    def describe(self):
        return {"__qt__": self._path}


_PYSIDE_MODULES = ("PySide6.QtWidgets", "PySide6.QtCore", "PySide6.QtGui")


def install_qt_stubs(ctx, names, modules=(), namespaces=()) -> dict:
    """Replace the Qt *names* in *namespaces* and in the PySide6 modules with recording stubs."""
    names = {n for n in names if _QT_NAME_RE.match(n)}
    if not names:
        return {}
    stubs = {n: QtStub(n, ctx.recorder) for n in sorted(names)}
    for ns in namespaces:
        for n, stub in stubs.items():
            if n in ns:
                ctx.patch_dict(ns, n, stub)
    if importlib.util.find_spec("PySide6") is None:
        return stubs
    for module_name in sorted(set(_PYSIDE_MODULES) | set(modules)):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for n, stub in stubs.items():
            if n in vars(module):
                ctx.patch(module, n, stub)
    return stubs


class BlockedPopen:
    """subprocess.Popen while fuzzing: recorded, then refused (no process is spawned)."""

    recorder = None

    def __init__(self, *args, **kwargs):
        if BlockedPopen.recorder is not None:
            BlockedPopen.recorder.record("subprocess.Popen", args[:1], {})
        raise OSError("parity fuzz: process spawning is blocked")


def _guard_fd_open(real_open):
    """``open()`` that refuses integer "paths": fuzzed values such as ``True``/``1`` would
    otherwise open (and, leaving a ``with`` block, close) the test process's own stdout."""
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, int):
            raise OSError(9, f"parity fuzz: refusing to open file descriptor {file!r}")
        return real_open(file, *args, **kwargs)
    guarded_open.__name__ = "open"
    guarded_open.__wrapped__ = real_open
    return guarded_open


def install_fd_guard(ctx) -> None:
    import builtins
    import io

    real_open = getattr(builtins.open, "__wrapped__", builtins.open)
    guarded = _guard_fd_open(real_open)
    ctx.patch(builtins, "open", guarded)
    ctx.patch(io, "open", guarded)
    real_fdopen = getattr(os.fdopen, "__wrapped__", os.fdopen)
    ctx.patch(os, "fdopen", _guard_fd_open(real_fdopen))


def install_clock_stubs(ctx) -> None:
    """Wall-clock reads the capture context's fixed ``time.time`` does not cover.

    Generated files embed the current time (``time.gmtime()`` / ``time.localtime()``
    without an argument, ``datetime.now()`` in the archive -> EPUB converters); two
    sides straddling a second boundary would otherwise write different bytes.
    """
    import datetime as _dt

    real_gmtime, real_localtime = time.gmtime, time.localtime
    fixed = normalize.FIXED_TIME

    def gmtime(secs=None):
        return real_gmtime(fixed if secs is None else secs)

    def localtime(secs=None):
        return real_localtime(fixed if secs is None else secs)

    real_datetime = _dt.datetime

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromtimestamp(fixed, tz)

        @classmethod
        def utcnow(cls):
            return real_datetime.fromtimestamp(fixed, _dt.timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return real_datetime.fromtimestamp(fixed)

    ctx.patch(time, "gmtime", gmtime)
    ctx.patch(time, "localtime", localtime)
    ctx.patch(_dt, "datetime", FixedDatetime)


def install_side_effect_stubs(ctx) -> None:
    rec = ctx.recorder

    def recorder_fn(name):
        def fn(*args, **kwargs):
            rec.record(name, args, kwargs)
            return None
        fn.__name__ = name.rsplit(".", 1)[-1]
        return fn

    try:
        import winsound
        for attr in ("MessageBeep", "PlaySound", "Beep"):
            if hasattr(winsound, attr):
                ctx.patch(winsound, attr, recorder_fn(f"winsound.{attr}"))
    except ImportError:
        pass
    import webbrowser
    for attr in ("open", "open_new", "open_new_tab"):
        ctx.patch(webbrowser, attr, recorder_fn(f"webbrowser.{attr}"))
    if hasattr(os, "startfile"):
        ctx.patch(os, "startfile", recorder_fn("os.startfile"))
    import subprocess
    BlockedPopen.recorder = rec
    ctx.patch(subprocess, "Popen", BlockedPopen)


# ---------------------------------------------------------------------------
# Sandbox / process reset
# ---------------------------------------------------------------------------

_GLOSSARY_CSV = "type,raw_name,translated_name,gender\ncharacter,김상현,Kim Sang-hyun,male\n"

#: Files of every fuzz sandbox (plus the union of the golden scenarios' files).
FUZZ_FILES = {
    "inputs/Fuzz Novel.epub": "PK",
    "inputs/fuzz.txt": "첫 번째 문단.\n\n두 번째 문단.\n",
    "inputs/Fuzz Doc.pdf": "%PDF-1.4\n",
    "inputs/ep01.srt": "1\n00:00:01,000 --> 00:00:02,000\n안녕\n",
    "inputs/fuzz_glossary.csv": _GLOSSARY_CSV,
    "outputs/Fuzz Novel/response_0001_chapter.html": "<html><body><p>x</p></body></html>",
}

def _zip_bytes(members) -> bytes:
    """A ZIP archive (bytes) of ``[(name, data)]`` (stored, deterministic timestamps)."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            archive.writestr(info, data)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    """A valid 1x1 RGB PNG."""
    import struct

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    signature = bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A])
    pixels = bytes([0x00, 0xFF, 0x00, 0x00])  # filter byte + one RGB pixel
    return (signature + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def _archive_fixtures() -> dict:
    """Inputs for the U3 input-preparation entries (one per resolve_input_to_epub branch)."""
    nl = chr(10)
    page = "<html><head><title>{t}</title></head><body><h1>{t}</h1><p>본문 {t}.</p></body></html>"
    png = _png_bytes()
    srt = "1" + nl + "00:00:01,000 --> 00:00:02,000" + nl + "{t}" + nl
    container = ('<?xml version="1.0"?><container version="1.0" '
                 'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                 '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 '</rootfiles></container>')
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
           '<metadata/><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
           '</manifest><spine><itemref idref="c1"/></spine></package>')
    images = _zip_bytes([("001.png", png), ("002.png", png)])
    return {
        "inputs/Fuzz Chapters.zip": _zip_bytes([("ch1.html", page.format(t="One").encode()),
                                                ("ch2.html", page.format(t="Two").encode())]),
        "inputs/Fuzz Images.zip": images,
        "inputs/Fuzz Images.cbz": images,
        "inputs/Fuzz Subs.zip": _zip_bytes([("ep01.srt", srt.format(t="안녕").encode()),
                                            ("ep02.srt", srt.format(t="잘 가").encode())]),
        "inputs/Fuzz Bad.zip": _zip_bytes([("data.bin", bytes([0, 1, 2]))]),
        "inputs/Fuzz Epub.zip": _zip_bytes([("mimetype", b"application/epub+zip"),
                                            ("META-INF/container.xml", container.encode()),
                                            ("OEBPS/content.opf", opf.encode()),
                                            ("OEBPS/c1.xhtml", page.format(t="E").encode())]),
        "inputs/Fuzz Page.html": page.format(t="Page"),
        "inputs/Fuzz Broken.zip": b"PK" + bytes([3, 4]) + b" not really a zip",
    }


#: Archive / HTML inputs (``archive_path`` / ``unique_files`` argument profiles); written
#: into every fuzz sandbox by ``fuzz_template_files``.
FUZZ_ARCHIVES = _archive_fixtures()

#: Path pool for path-like arguments (relative to the sandbox root; None/''/'abc' added).
FUZZ_PATHS = (
    "inputs/Fuzz Novel.epub",
    "inputs/fuzz.txt",
    "inputs/Fuzz Doc.pdf",
    "inputs/ep01.srt",
    "inputs/missing.epub",
    "outputs/Fuzz Novel",
)

_TS_RE = re.compile(r"\d{8}[_-]?\d{6}")


def _scenario_files() -> dict:
    from parity import scenarios

    out = {}
    for scenario in scenarios.SCENARIOS.values():
        out.update(scenario.get("files") or {})
    return out


def _det_tempnames():
    return (f"fz{i:06d}" for i in itertools.count())


class FuzzContext:
    """One sandbox with deterministic process state for a whole fuzz run (both sides).

    Wraps ``normalize.CaptureContext`` (fresh sandbox, scrubbed env, fixed
    time/pid/cpu/uuid/tempdir, recording ``UnifiedClient`` and backend stubs when
    *backend*) and adds a per-run ``reset`` plus sandbox file-change capture/undo.
    """

    def __init__(self, label: str, *, backend: bool = True, files: dict | None = None):
        scenario = {"name": "fuzz", "config": None, "files": dict(files or {})}
        self.cap = normalize.CaptureContext(scenario, f"fuzz-{label}", backend=backend)
        self.backend = backend
        self.violations: list = []

    # -- delegation ------------------------------------------------------------
    @property
    def recorder(self):
        return self.cap.recorder

    @property
    def sandbox(self):
        return self.cap.sandbox

    @property
    def backend_calls(self):
        return self.cap.backend_calls

    @property
    def backend_stubs(self):
        return self.cap.backend_stubs

    def patch(self, obj, attr, value):
        self.cap.patch(obj, attr, value)

    def patch_dict(self, mapping, key, value):
        self.cap.patch_dict(mapping, key, value)

    def resolve(self, value):
        root = str(self.sandbox.root)
        return map_strings(value, lambda s: s.replace(normalize.SANDBOX_TOKEN, root))

    def path(self, rel):
        return None if rel is None else self.sandbox.path(rel)

    # -- lifecycle ---------------------------------------------------------------
    def __enter__(self):
        self.cap.__enter__()
        install_fd_guard(self)
        install_clock_stubs(self)
        self._src_before = _src_tripwire()
        self.base_os_env = dict(os.environ)
        self.base_eff = normalize.effective_env()
        self.base_cwd = os.getcwd()
        self.base_stdout = sys.stdout
        self.rebaseline()
        return self

    def rebaseline(self) -> None:
        """Take the current sandbox tree as the state every run starts from."""
        self._fs_bytes, self._fs_stat, self._fs_dirs = self._scan(read=True)
        self._pending = None

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout = self.base_stdout
            self.cap.__exit__(exc_type, exc, tb)
        finally:
            after = _src_tripwire()
            if after != self._src_before:
                self.violations.append(f"src/ changed during the fuzz run: {_tripwire_diff(self._src_before, after)}")
        return False

    # -- per run -------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        sys.stdout = self.base_stdout
        os.chdir(self.base_cwd)
        os.environ.clear()
        os.environ.update(self.base_os_env)
        large_env = sys.modules.get("large_env")
        if large_env is not None:
            large_env.clear_store()
        sys.argv = list(normalize.BASELINE_ARGV)
        self.cap._uuid_counter = 0
        random.seed(seed)
        tempfile._name_sequence = _det_tempnames()
        self.recorder.clear()
        self.cap.backend_calls.clear()
        self.base_stdout.seek(0)
        self.base_stdout.truncate()

    def stdout_lines(self) -> list:
        return [mask(line) for line in self.base_stdout.getvalue().splitlines()]

    def env_delta(self) -> dict:
        return normalize.env_delta(self.base_eff, normalize.effective_env())

    # -- sandbox files ----------------------------------------------------------
    def _scan(self, read=False):
        root = str(self.sandbox.root)
        files_stat, dirs, data = {}, set(), {}
        stack = [(root, "")]
        while stack:
            path, rel = stack.pop()
            try:
                entries = list(os.scandir(path))
            except OSError:
                continue
            for entry in entries:
                child_rel = f"{rel}/{entry.name}" if rel else entry.name
                if entry.is_dir(follow_symlinks=False):
                    dirs.add(child_rel)
                    stack.append((entry.path, child_rel))
                else:
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    files_stat[child_rel] = (st.st_size, st.st_mtime_ns)
                    if read:
                        with open(entry.path, "rb") as fh:
                            data[child_rel] = fh.read()
        return data, files_stat, dirs

    def _read(self, rel) -> bytes:
        with open(os.path.join(str(self.sandbox.root), rel), "rb") as fh:
            return fh.read()

    def fs_delta(self) -> dict:
        _data, stat, dirs = self._scan()
        added = sorted(r for r in stat if r not in self._fs_stat)
        removed = sorted(r for r in self._fs_stat if r not in stat)
        changed = []
        for rel in stat:
            if rel in self._fs_stat and stat[rel] != self._fs_stat[rel]:
                if self._read(rel) != self._fs_bytes[rel]:
                    changed.append(rel)
                else:
                    self._fs_stat[rel] = stat[rel]
        dirs_added = sorted(dirs - self._fs_dirs)
        dirs_removed = sorted(self._fs_dirs - dirs)
        self._pending = (added, changed, removed, dirs_added, dirs_removed)
        if not (added or changed or removed or dirs_added or dirs_removed):
            return {}
        root = str(self.sandbox.root)
        return {
            "added": {_TS_RE.sub("<TS>", r): canon_file(os.path.join(root, r)) for r in added},
            "changed": {_TS_RE.sub("<TS>", r): canon_file(os.path.join(root, r)) for r in sorted(changed)},
            "removed": [_TS_RE.sub("<TS>", r) for r in removed],
            "dirs_added": [_TS_RE.sub("<TS>", r) for r in dirs_added],
            "dirs_removed": [_TS_RE.sub("<TS>", r) for r in dirs_removed],
        }

    def restore_fs(self) -> None:
        if not self._pending:
            return
        added, changed, removed, dirs_added, dirs_removed = self._pending
        self._pending = None
        root = str(self.sandbox.root)
        for rel in added:
            try:
                os.remove(os.path.join(root, rel))
            except OSError:
                pass
        for rel in sorted(dirs_added, key=lambda r: -r.count("/")):
            shutil.rmtree(os.path.join(root, rel), ignore_errors=True)
        for rel in sorted(dirs_removed, key=lambda r: r.count("/")):
            os.makedirs(os.path.join(root, rel), exist_ok=True)
        for rel in list(changed) + list(removed):
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(self._fs_bytes[rel])
            st = os.stat(path)
            self._fs_stat[rel] = (st.st_size, st.st_mtime_ns)


def canon_file(path: str):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return {"unreadable": type(exc).__name__}
    if len(data) > 512 * 1024:
        return {"crc32": zlib.crc32(data), "size": len(data)}
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"crc32": zlib.crc32(data), "size": len(data)}
    if path.lower().endswith(".json"):
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                try:
                    from api_key_encryption import decrypt_config
                    value = decrypt_config(value)
                except Exception:
                    pass
            return {"json": canon(value)}
        except ValueError:
            pass
    return {"text": mask(text)}


def _src_tripwire() -> dict:
    out = {}
    try:
        for entry in os.scandir(SRC_DIR):
            if entry.is_file() and entry.name.endswith((".json", ".key")) or entry.name.startswith(".glossarion"):
                try:
                    out[entry.name] = entry.stat().st_mtime_ns
                except OSError:
                    pass
            elif entry.is_dir():
                out[entry.name + "/"] = 0
    except OSError:
        pass
    return out


def _tripwire_diff(before: dict, after: dict) -> list:
    keys = sorted(set(before) | set(after))
    return [k for k in keys if before.get(k) != after.get(k)][:10]


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


@dataclass
class Base:
    name: str
    attrs: dict
    config: dict

    def __post_init__(self):
        # owners are built with a C-level dict copy; only these values need a deep copy
        self.mutable_attrs = tuple(k for k, v in self.attrs.items() if type(v) not in _IMMUTABLE_TYPES)
        self.mutable_config = tuple(k for k, v in self.config.items() if type(v) not in _IMMUTABLE_TYPES)


EMPTY_BASE = Base("empty", {}, {})


@dataclass
class State:
    index: int
    seed: int
    base: str
    mode: str
    attrs: dict          # name -> value | MISSING (overrides on top of the base)
    config: dict         # key -> value | MISSING
    env: dict            # name -> value | MISSING
    args: tuple
    kwargs: dict

    def summary(self, limit=12) -> dict:
        def short(d):
            items = list(d.items())
            out = {k: (repr(v) if not isinstance(v, _WIDGET_TYPES) else v.describe()) for k, v in items[:limit]}
            if len(items) > limit:
                out["..."] = f"+{len(items) - limit} more"
            return out
        return {"index": self.index, "seed": self.seed, "base": self.base, "mode": self.mode,
                "attrs": short(self.attrs), "config": short(self.config), "env": short(self.env),
                "args": repr(self.args)[:300], "kwargs": repr(self.kwargs)[:300]}


_COMBO_ITEMS = (("Off", "off"), ("Balanced", "balanced"), ("Full", "full"), ("Partial", "partial"),
                ("Contextual History", "contextual_history"), ("High", "high"), ("Abc", None))


def make_widget(kinds, rng: random.Random):
    kinds = set(kinds or ())
    if "check" in kinds:
        return fakes.FakeCheck(rng.choice((True, False)))
    if "combo" in kinds:
        items = rng.sample(_COMBO_ITEMS, rng.randint(0, len(_COMBO_ITEMS)))
        combo = fakes.FakeCombo(items, index=rng.randint(-1, max(0, len(items) - 1)) if items else 0)
        if rng.random() < 0.15:
            combo.setCurrentText(rng.choice(TEXT_VALUES))
        return combo
    if "plain" in kinds:
        return fakes.FakeTextEdit(rng.choice(TEXT_VALUES))
    return fakes.FakeLineEdit(rng.choice(TEXT_VALUES))


def _variant(value, rng: random.Random):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return rng.choice((value + 1, -value, 0))
    if isinstance(value, float):
        return rng.choice((value * 2, -value, 0.0))
    if isinstance(value, str):
        return rng.choice(("", value.upper(), f" {value} ", value + "x"))
    if isinstance(value, list):
        return rng.choice(([], value[:1], [None]))
    if isinstance(value, dict):
        return rng.choice(({}, {"abc": 1}))
    return rng.choice(("abc", 0, []))


_ATTR_SUFFIXES = ("_var", "_checkbox", "_check", "_entry", "_combo", "_edit")


def promote_string_reads(read_set: ReadSet, bases, provided=frozenset()) -> ReadSet:
    """Treat identifier strings as dynamic reads (``getattr(self, name)`` / ``config.get(key)``
    driven by literal tables such as ``self._live_bool_setting('x_checkbox', 'x_var', 'x')``):
    strings naming an attribute of some base (or shaped like one) join ``attrs``, strings
    naming a config key of some base join ``config``."""
    known_attrs = set().union(*(b.attrs for b in bases)) if bases else set()
    known_config = set().union(*(b.config for b in bases)) if bases else set()
    provided = set(provided) | {"config"}
    for s in read_set.strings:
        if s in provided:
            continue
        if s in known_attrs or s.endswith(_ATTR_SUFFIXES):
            read_set.attrs.add(s)
        if s in known_config:
            read_set.config.setdefault(s, [])
    return read_set


class StateSpace:
    """Seeded generator of owner states over a read set and a list of bases."""

    #: probability that a state perturbs every read item independently (else 1-6 items)
    CHAOS = 0.25

    def __init__(self, read_set: ReadSet, bases, *, arg_gen=None, seed: int = DEFAULT_SEED,
                 empty_weight: float = 0.2, provided=frozenset(), promote_strings: bool = True):
        self.read_set = read_set
        self.bases = list(bases) or [EMPTY_BASE]
        if not any(b.name == "empty" for b in self.bases):
            self.bases.insert(0, EMPTY_BASE)
        if promote_strings:
            promote_string_reads(read_set, self.bases, provided)
        self.seed = int(seed)
        self.empty_weight = empty_weight
        self.arg_gen = arg_gen or (lambda rng: ((), {}))
        self.attrs = sorted(read_set.attrs)
        self.widgets = {k: set(v) for k, v in read_set.widgets.items()}
        self.config_keys = sorted(read_set.config)
        self.env_keys = sorted(read_set.env)
        self.config_defaults = {
            k: dedup(d for d in v if d is not NO_DEFAULT) for k, v in read_set.config.items()
        }
        self.known_attrs = set().union(*(b.attrs for b in self.bases))
        self.known_config = set().union(*(b.config for b in self.bases))
        self.attr_pool = {
            a: dedup([b.attrs[a] for b in self.bases if a in b.attrs] + read_set.attr_defaults.get(a, []))
            for a in self.attrs
        }
        self.config_pool = {
            k: dedup([b.config[k] for b in self.bases if k in b.config] + self.config_defaults.get(k, []))
            for k in self.config_keys
        }
        self._items = ([("attr", a) for a in self.attrs] + [("config", k) for k in self.config_keys]
                       + [("env", e) for e in self.env_keys])

    def _attr_value(self, name, rng):
        r = rng.random()
        pool = self.attr_pool.get(name)
        if pool is None:
            pool = dedup([b.attrs[name] for b in self.bases if name in b.attrs])
        if r < 0.30:
            return MISSING
        if r < 0.62:
            return rng.choice(SCALARS)
        if r < 0.84 and pool:
            return fast_copy(rng.choice(pool))
        if name in self.widgets or name.endswith(("_entry", "_checkbox", "_check", "_combo", "_edit")):
            return make_widget(self.widgets.get(name) or {"line"}, rng)
        return rng.choice(SCALARS)

    def _config_value(self, key, rng, base):
        r = rng.random()
        defaults = self.config_defaults.get(key) or []
        pool = self.config_pool.get(key)
        if pool is None:
            pool = dedup([b.config[key] for b in self.bases if key in b.config])
        if r < 0.25:
            return MISSING
        if r < 0.45 and defaults:
            return fast_copy(rng.choice(defaults))
        if r < 0.70 and pool:
            return fast_copy(rng.choice(pool))
        if r < 0.90:
            return rng.choice(SCALARS)
        current = base.config.get(key, defaults[0] if defaults else None)
        return _variant(current, rng)

    def generate(self, index: int) -> State:
        rng = random.Random(self.seed * 1_000_003 + index)
        if rng.random() < self.empty_weight or len(self.bases) == 1:
            base = self.bases[0]
        else:
            base = rng.choice(self.bases[1:])
        args, kwargs = self.arg_gen(rng)
        dynamic = []
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, str) and value.isidentifier():
                if value in self.known_config:
                    dynamic.append(("config", value))
                if value in self.known_attrs or value.endswith(("_var", "_checkbox", "_check", "_entry",
                                                                 "_combo", "_edit")):
                    dynamic.append(("attr", value))
        items = self._items
        if rng.random() < self.CHAOS:
            mode = "chaos"
            chosen = [it for it in items if rng.random() < 0.5] + [it for it in dynamic if rng.random() < 0.5]
        else:
            mode = "targeted"
            k = rng.randint(1, min(6, len(items))) if items else 0
            chosen = rng.sample(items, k) if k else []
            chosen += [it for it in dynamic if rng.random() < 0.5]
        attrs, config, env = {}, {}, {}
        for kind, name in chosen:
            if kind == "attr":
                attrs[name] = self._attr_value(name, rng)
            elif kind == "config":
                config[name] = self._config_value(name, rng, base)
            else:
                env[name] = rng.choice(ENV_VALUES + (MISSING,))
        run_seed = (self.seed * 7919 + index) & 0x7FFFFFFF
        return State(index, run_seed, base.name, mode, attrs, config, env, tuple(args), dict(kwargs))

    def base_named(self, name) -> Base:
        for b in self.bases:
            if b.name == name:
                return b
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

MODEL_VALUES = ("", None, 5, "gpt-4o", "gpt-4o-mini", "authgpt/gpt-6-luna", "gemini-2.5-flash",
                "gemini-2.5-flash-image", "gpt-image-1", "dall-e-3", "imagen-4.0-generate-001",
                "veo-3.0-generate-001", "sora-2", "vertex/gemini-2.5-pro", "claude-sonnet-4-5",
                "deepseek-chat", "lan/qwen3:8b", "google-translate-free", "deepl")
ROUTE_VALUES = (
    None, [], "abc", {"prefix": "a"}, [{}], [None, "x"], [{"prefix": 5}],
    [{"prefix": "lan", "routing": "http://192.168.1.20:11434/v1/", "endpoint_type": "openai_chat"}],
    [{"prefix": "img/", "routing": "https://images.example.test", "endpoint_type": "openai-images"},
     {"prefix": "bad/", "routing": "ftp://not-http"},
     {"prefix": "LAN/", "routing": "http://duplicate.example.test"},
     {"prefix": "anth", "base_url": "https://anthropic.example.test", "type": "{base_url}/v1/messages"}],
)
ENDPOINT_VALUES = (None, "", "abc", 5, "openai_chat", "openai-chat", "openai_responses", "openai-images",
                   "anthropic", "gemini", "{base_url}/v1/messages", "{base_url}/chat/completions", " OpenAI_Chat ")
ENV_DICT_VALUES = (None, {}, {"OUTPUT_MODE": "vision", "MODEL": "gpt-4o"}, {"X": "1"},
                   {"GLOSSARY_DUPLICATE_ALGORITHM": "auto", "TRANSLATION_ANTI_DUPLICATE": "1"})


class ArgGen:
    """Arguments for one callable: harvested literal call sites + name-based heuristics."""

    def __init__(self, fn=None, *, harvested=(), paths=(), hints=None, profile=None, archives=()):
        self.harvested = list(harvested)
        self.paths = list(paths)
        self.archives = list(archives)
        self.hints = hints or {}
        self.profile = profile
        self.params = []
        fn = unwrap(fn) if fn is not None else None
        if fn is not None:
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                sig = None
            if sig is not None:
                for p in sig.parameters.values():
                    if p.name in _SELF_NAMES:
                        continue
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                        self.params.append(p)

    def __call__(self, rng: random.Random):
        if self.profile is not None:
            return ARG_PROFILES[self.profile](self, rng)
        if self.harvested and rng.random() < 0.5:
            args, kwargs = rng.choice(self.harvested)
            return fast_copy(tuple(args)), fast_copy(dict(kwargs))
        args, kwargs = [], {}
        positional = True
        for p in self.params:
            has_default = p.default is not p.empty
            if has_default and rng.random() < 0.35:
                positional = False
                continue
            value = self.value_for(p.name, rng)
            if positional and p.kind != p.KEYWORD_ONLY and not has_default:
                args.append(value)
            else:
                positional = positional and not has_default
                kwargs[p.name] = value
        return tuple(args), kwargs

    def pick_path(self, rng):
        return rng.choice(self.paths + [None, "", "abc"])

    def value_for(self, name: str, rng: random.Random):
        n = name.lower()
        hints = self.hints
        if n == "files":
            return [rng.choice(self.paths) for _ in range(rng.randint(0, 3))] if self.paths else []
        if n.endswith("_attr"):
            suffixes = {"checkbox_attr": ("_checkbox", "_check"), "var_attr": ("_var",),
                        "widget_attr": ("_entry", "_edit", "_text")}.get(n, ("_var",))
            pool = [a for a in hints.get("attrs", ()) if a.endswith(suffixes)]
            return rng.choice(pool + ["abc", ""]) if pool else rng.choice(("abc", ""))
        if n in ("config_key", "key"):
            pool = list(hints.get("config_keys", ()))
            return rng.choice(pool + ["abc"]) if pool else "abc"
        if n in ("api_key",):
            return rng.choice(("", "sk-fuzz-0000", None, "AIza-fuzz"))
        if "model" in n:
            return rng.choice(tuple(hints.get("models", ())) + MODEL_VALUES)
        if n == "routes":
            return fast_copy(rng.choice(ROUTE_VALUES))
        if n == "endpoint_type":
            return rng.choice(ENDPOINT_VALUES)
        if n in ("env_vars", "base_env", "env"):
            return fast_copy(rng.choice(ENV_DICT_VALUES))
        if n == "name":
            return rng.choice(tuple(hints.get("profile_names", ())) + ("", "abc", None))
        if any(t in n for t in ("path", "file", "folder", "dir", "source", "input")):
            return self.pick_path(rng)
        if "tokens" in n:
            return rng.choice((0, 1, 4096, 65536, 128000, -1, None, "5", "abc"))
        if n.startswith(("force", "show", "is_", "use_", "enable")):
            return rng.choice((True, False, None, 1))
        if n in ("index", "_idx", "idx", "state"):
            return rng.choice((None, 0, 1, 2, -1, True))
        if n in ("value", "text"):
            return rng.choice(TEXT_VALUES + SCALARS)
        return rng.choice(SCALARS + TEXT_VALUES[:4])


def _unique_files_args(gen, rng):
    """``(files,)`` without duplicates (at most one real EPUB: a thread pool of one keeps the
    worker log order fixed)."""
    pool = list(dict.fromkeys(gen.paths + gen.archives))
    files = rng.sample(pool, rng.randint(0, min(3, len(pool)))) if pool else []
    epubs = [p for p in pool if str(p).lower().endswith(".epub") and os.path.isfile(p)]
    if epubs and rng.random() < 0.6 and not any(p in files for p in epubs):
        files.insert(rng.randint(0, len(files)), epubs[0])  # reach the metadata worker more often
    if rng.random() < 0.1:
        files.append(rng.choice((None, "", "abc")))
    return (files,), {}


def _archive_path_args(gen, rng):
    """``(path,)`` from the archive / HTML fixtures (mostly), the plain paths or junk."""
    r = rng.random()
    if r < 0.7 and gen.archives:
        return (rng.choice(gen.archives),), {}
    if r < 0.9:
        return (gen.pick_path(rng),), {}
    return (rng.choice((None, "", "abc", 5, "x.ZIP", "page.HTM")),), {}


ARG_PROFILES = {
    "none": lambda gen, rng: ((), {}),
    "save_config": lambda gen, rng: ((), {"show_message": rng.choice((False, True))}),
    "unique_files": _unique_files_args,
    "archive_path": _archive_path_args,
}


# ---------------------------------------------------------------------------
# Sides, outcomes, comparison
# ---------------------------------------------------------------------------


@dataclass
class Side:
    """One implementation under test: owner class + what to call on it."""

    label: str
    owner_cls: type
    call: object                      # attribute name, tuple of names (run in order), or callable(owner, *a, **k)
    enter: object = None              # callable() run right before the call (e.g. module patches)
    exit: object = None
    bound: tuple = ()                 # per-instance recorder methods (setup_other_settings_methods)
    exclude_attrs: frozenset = frozenset()

    def invoke(self, owner, args, kwargs):
        call = self.call
        if callable(call) and not isinstance(call, str):
            return call(owner, *args, **kwargs)
        if isinstance(call, (tuple, list)):
            result = None
            for i, name in enumerate(call):
                result = getattr(owner, name)(*args, **kwargs) if i == 0 else getattr(owner, name)()
            return result
        return getattr(owner, call)(*args, **kwargs)


@dataclass
class Outcome:
    """What one side did. ``config``/``attrs``/``calls``/``backend`` hold the raw objects
    (compared with ``==`` first; canonicalised only when they differ, see ``compare_outcomes``)."""

    result: object = None
    result_keys: object = None
    exc_type: object = None
    exc_msg: str = ""
    env: dict = field(default_factory=dict)
    config: object = None
    attrs: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    backend: list = field(default_factory=list)
    stdout: list = field(default_factory=list)
    fs: dict = field(default_factory=dict)
    argv: list = field(default_factory=list)
    cwd: str = ""
    seconds: float = 0.0
    owner: object = None

    def canonical(self, name):
        value = getattr(self, name)
        if name in LAZY_FIELDS:
            return canon(value, self.owner)
        return value


_RECORDER_FUNCS: dict = {}


def _recorder_function(name: str):
    if name not in _RECORDER_FUNCS:
        def recorder_method(self, *args, **kwargs):
            self._parity_recorder.record(name, args, kwargs)
            return None
        recorder_method.__name__ = name
        _RECORDER_FUNCS[name] = recorder_method
    return _RECORDER_FUNCS[name]


def build_owner(side: Side, ctx: FuzzContext, base: Base, state: State):
    owner = side.owner_cls(ctx.recorder)
    data = owner.__dict__
    data.update(base.attrs)
    for name in base.mutable_attrs:
        data[name] = fast_copy(data[name])
    for name, value in state.attrs.items():
        if value is MISSING:
            data.pop(name, None)
        else:
            setattr(owner, name, fast_copy(value))
    config = dict(base.config)  # keeps the key order
    for key in base.mutable_config:
        config[key] = fast_copy(config[key])
    for key, value in state.config.items():
        if value is MISSING:
            config.pop(key, None)
        else:
            config[key] = fast_copy(value)
    owner.config = config
    for name in side.bound:
        setattr(owner, name, types.MethodType(_recorder_function(name), owner))
    return owner


def _apply_env(state: State):
    for name, value in state.env.items():
        if value is MISSING:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def run_side(ctx: FuzzContext, side: Side, base: Base, state: State) -> Outcome:
    ctx.reset(state.seed)
    _apply_env(state)
    owner = build_owner(side, ctx, base, state)
    # the env the state starts from is part of the input, not of the effect
    pre_env = normalize.effective_env()
    out = Outcome()
    args, kwargs = fast_copy(state.args), fast_copy(state.kwargs)
    started = time.perf_counter()
    if side.enter:
        side.enter()
    try:
        result = side.invoke(owner, args, kwargs)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - SystemExit from backend code is an outcome too
        result = None
        out.exc_type = type(exc).__name__
        out.exc_msg = mask(str(exc))[:300]
    finally:
        if side.exit:
            side.exit()
    out.seconds = time.perf_counter() - started
    out.owner = owner
    out.result = canon(result, owner)
    if isinstance(result, dict):
        out.result_keys = [canon(k, owner) for k in result]
    out.env = normalize.env_delta(pre_env, normalize.effective_env())
    out.config = owner.__dict__.get("config", MISSING)
    skip = side.exclude_attrs | set(side.bound)
    widget_types = _WIDGET_TYPES
    out.attrs = {
        k: (v.describe() if isinstance(v, widget_types) else v) for k, v in owner.__dict__.items()
        if k != "config" and not k.startswith("_parity_") and k not in skip
        and not isinstance(v, fakes.FakeSignal)
    }
    out.calls = list(ctx.recorder.calls)
    out.backend = list(ctx.backend_calls)
    out.stdout = ctx.stdout_lines()
    out.fs = ctx.fs_delta()
    out.argv = list(sys.argv)
    out.cwd = os.getcwd()
    ctx.restore_fs()
    return out


def _diff(expected, actual, limit=6) -> list:
    from parity import capture_golden

    lines = capture_golden.diff(expected, actual, limit=limit)
    return lines or [f"{str(expected)[:200]!r} != {str(actual)[:200]!r}"]


#: Outcome fields holding raw objects: ``==`` is the fast path, ``canon`` decides on a miss.
#: (``==`` treats 5 == 5.0 == True alike; such a change inside config/attrs only shows
#: where it reaches env, stdout, files or the return value, which are canonical/strings.)
LAZY_FIELDS = frozenset({"config", "attrs", "calls", "backend"})


def _raw_equal(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def compare_outcomes(a: Outcome, b: Outcome, *, fields=COMPARED_FIELDS, strict_order=True) -> dict:
    """{field: [diff lines]} for every field where the two outcomes differ."""
    out = {}
    for name in fields:
        va, vb = getattr(a, name), getattr(b, name)
        if name in LAZY_FIELDS:
            if _raw_equal(va, vb):
                continue
            va, vb = a.canonical(name), b.canonical(name)
        if va != vb:
            out[name] = _diff(va, vb)
    if strict_order and "result" not in out and a.result_keys != b.result_keys:
        out["result_order"] = [f"key order {a.result_keys[:20]!r} != {b.result_keys[:20]!r}"]
    return out


@dataclass
class Mismatch:
    index: int
    fields: dict
    state: dict
    legacy_exc: str
    new_exc: str


@dataclass
class FuzzReport:
    label: str
    states: int = 0
    seed: int = DEFAULT_SEED
    mismatches: list = field(default_factory=list)
    legacy_exceptions: collections.Counter = field(default_factory=collections.Counter)
    new_exceptions: collections.Counter = field(default_factory=collections.Counter)
    modes: collections.Counter = field(default_factory=collections.Counter)
    bases: collections.Counter = field(default_factory=collections.Counter)
    read_set: dict = field(default_factory=dict)
    seconds: float = 0.0
    slowest: float = 0.0
    stopped_early: bool = False
    violations: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.violations

    @property
    def clean_fraction(self) -> float:
        """Share of states where the legacy side returned without an exception (coverage proxy)."""
        if not self.states:
            return 0.0
        raised = sum(self.legacy_exceptions.values())
        return 1.0 - raised / self.states

    def summary(self) -> str:
        top = ", ".join(f"{k} x{v}" for k, v in self.legacy_exceptions.most_common(3))
        return (f"{self.label}: {self.states} states, {len(self.mismatches)} mismatches, "
                f"clean {self.clean_fraction:.0%}, exceptions [{top or '-'}], read set {self.read_set}, "
                f"{self.seconds:.1f}s (slowest call {self.slowest * 1000:.0f} ms)")

    def failure_text(self, limit: int = 3) -> str:
        lines = [self.summary()]
        lines += [f"HARNESS VIOLATION: {v}" for v in self.violations]
        for m in self.mismatches[:limit]:
            lines.append(f"--- state #{m.index} differs in {sorted(m.fields)} "
                         f"(legacy exc: {m.legacy_exc or '-'}; new exc: {m.new_exc or '-'})")
            lines.append(f"    state: {m.state}")
            for fld, diff_lines in m.fields.items():
                for line in diff_lines[:6]:
                    lines.append(f"    [{fld}] {line}")
        if len(self.mismatches) > limit:
            lines.append(f"... {len(self.mismatches) - limit} more mismatching states")
        if self.mismatches:
            lines.append(f"replay: python tests/parity/fuzz_moved.py {self.label} --seed {self.seed} "
                         f"--index {self.mismatches[0].index}")
        return "\n".join(lines)


def differential_fuzz(ctx: FuzzContext, legacy: Side, new: Side, space: StateSpace, *,
                      states: int = DEFAULT_STATES, fields=COMPARED_FIELDS, strict_order: bool = True,
                      stop_after: int = 20, label: str = "fuzz", only_index: int | None = None,
                      on_state=None) -> FuzzReport:
    """Run both sides on *states* seeded states; collect mismatches (stop after *stop_after*)."""
    report = FuzzReport(label=label, seed=space.seed, read_set=space.read_set.sizes())
    ctx.rebaseline()  # setup (patches, imports) may have touched the sandbox
    started = time.perf_counter()
    indices = [only_index] if only_index is not None else range(states)
    trace = TRACE
    for i in indices:
        state = space.generate(i)
        if trace:  # last line before a hard crash names the state (sys.stdout is captured)
            sys.__stderr__.write(f"[fuzz {label}] state #{i} {state.summary(limit=4)}\n")
            sys.__stderr__.flush()
        base = space.base_named(state.base)
        sides = (legacy, new) if i % 2 == 0 else (new, legacy)
        outs = {side.label: run_side(ctx, side, base, state) for side in sides}
        a, b = outs[legacy.label], outs[new.label]
        report.states += 1
        report.modes[state.mode] += 1
        report.bases[state.base] += 1
        report.slowest = max(report.slowest, a.seconds, b.seconds)
        if a.exc_type:
            report.legacy_exceptions[a.exc_type] += 1
        if b.exc_type:
            report.new_exceptions[b.exc_type] += 1
        if on_state is not None:
            on_state(state, a, b)
        diff = compare_outcomes(a, b, fields=fields, strict_order=strict_order)
        if diff:
            report.mismatches.append(Mismatch(
                i, diff, state.summary(),
                f"{a.exc_type}: {a.exc_msg}" if a.exc_type else "",
                f"{b.exc_type}: {b.exc_msg}" if b.exc_type else "",
            ))
            if len(report.mismatches) >= stop_after:
                report.stopped_early = report.states < len(indices)
                break
    report.seconds = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------
# High level: legacy oracle vs working-tree desktop
# ---------------------------------------------------------------------------

#: Python MRO entries that are never resolved (Qt C++ bases).
_FOREIGN_MODULE_PREFIXES = ("PySide6", "shiboken6", "Shiboken", "builtins")


#: Both fuzzed owner classes carry the desktop class name, so messages that embed
#: ``type(self).__name__`` (AttributeError texts the code logs) are side-independent.
OWNER_CLASS_NAME = "TranslatorGUI"


def owner_class(name: str, bases: tuple, namespace: dict, *, tag: str = "") -> type:
    ns = {"__module__": __name__, "_parity_side": tag}
    ns.update(namespace)
    return type(name, bases, ns)


def resolve_python_mro(cls, name):
    for klass in cls.__mro__:
        if (klass.__module__ or "").startswith(_FOREIGN_MODULE_PREFIXES):
            continue
        if name in klass.__dict__:
            return klass.__dict__[name]
    return MISSING


def module_exists(name: str) -> bool:
    return (SRC_DIR / f"{name}.py").is_file()


class FuzzSession:
    """Lazily built shared pieces for the high-level driver (one per pytest session)."""

    def __init__(self, bundle=None):
        self._bundle = bundle
        self._bases = None
        self._tg = None
        self._tg_error = None
        self._call_sites = None

    # -- legacy ---------------------------------------------------------------
    @property
    def bundle(self):
        if self._bundle is None:
            from parity import freeze_legacy
            try:
                self._bundle = freeze_legacy.load_legacy()
            except FileNotFoundError as exc:
                raise Unavailable(f"legacy oracle not frozen ({exc}); run tests/parity/freeze_legacy.py") from exc
        return self._bundle

    @property
    def recorded(self) -> frozenset:
        from parity import freeze_legacy
        return frozenset(self.bundle.manifest.get("recorded_methods", ())) | freeze_legacy.TG_RECORDED_METHODS

    @property
    def bound(self) -> tuple:
        return tuple(self.bundle.manifest.get("other_settings_bound_methods", ()))

    def legacy_class(self, name=None):
        # U2+ oracles carry FROZEN copies of the shared mixins (never the live modules)
        mixins = tuple(fakes.shared_mixin_classes(self.bundle))
        return owner_class(OWNER_CLASS_NAME, (fakes.FakeState, self.bundle.methods) + mixins,
                           {n: _recorder_function(n) for n in self.recorded}, tag=name or "legacy")

    def legacy_names(self) -> set:
        m = self.bundle.manifest
        names = set(m.get("frozen_methods", {})) | set(m.get("class_attrs", {}))
        names |= {n for n, v in vars(self.bundle.methods).items() if n.startswith("legacy_")}
        for provided in (m.get("mixin_methods") or {}).values():  # moved into the frozen mixins (U2+)
            names |= set(provided)
        return names

    def bases(self) -> list:
        if self._bases is None:
            self._bases = legacy_bases(self.bundle)
        return self._bases

    def call_sites(self, name: str) -> list:
        """Literal call sites of ``self.<name>(...)`` in the frozen legacy code (indexed once)."""
        if self._call_sites is None:
            self._call_sites = index_call_sites([self.bundle.path.read_text(encoding="utf-8")])
        return list(self._call_sites.get(name, ()))

    # -- new side -------------------------------------------------------------
    def translator_gui_class(self):
        if self._tg is None and self._tg_error is None:
            if importlib.util.find_spec("PySide6") is None:
                self._tg_error = "PySide6 is not installed (tier D resolves the desktop MRO of TranslatorGUI)"
            else:
                normalize.preimport_backend_modules()
                import translator_gui  # an import error here is a broken desktop: let it fail
                self._tg = translator_gui.TranslatorGUI
        if self._tg is None:
            raise Unavailable(self._tg_error)
        return self._tg

    def mixin_classes(self) -> dict:
        out = {}
        for module_name, cls_name in moved_functions.MIXINS.items():
            if not module_exists(module_name):
                continue
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name, None)
            if cls is not None:
                out[module_name] = cls
        return out

    def new_class(self, mode: str = "desktop", name: str | None = None):
        mixins = self.mixin_classes()
        if mode == "desktop":
            source_cls = self.translator_gui_class()
        elif mode == "mixins":
            if not mixins:
                raise Unavailable("no shared mixin module exists yet")
            source_cls = type("SharedMixins", tuple(mixins.values()), {})
        elif mode == "legacy":
            return self.legacy_class(name or "legacy-2")
        else:
            raise ValueError(mode)
        names = self.legacy_names() | moved_functions.HOOK_NAMES
        for cls in mixins.values():
            for klass in cls.__mro__:
                if klass is not object:
                    names |= set(vars(klass))
        ns = {"__module__": __name__}
        for n in sorted(names):
            if n.startswith("__") or n in fakes.DESKTOP_SIGNALS or n.startswith("legacy_"):
                continue
            value = resolve_python_mro(source_cls, n)
            if value is not MISSING:
                ns[n] = value
        for n in self.recorded:
            ns[n] = _recorder_function(n)
        return owner_class(OWNER_CLASS_NAME, (fakes.FakeState,), ns, tag=name or mode)


def legacy_bases(bundle, scenario_names=None) -> list:
    """Desktop owners booted by the frozen legacy boot, one per golden scenario (paths as <SANDBOX>)."""
    from parity import capture_golden, scenarios

    names = tuple(scenario_names or scenarios.SCENARIO_NAMES)
    factory = fakes.make_legacy_owner_factory(bundle)
    out = [EMPTY_BASE]
    for name in names:
        scenario = scenarios.get(name)
        with normalize.CaptureContext(scenario, "fuzz-base") as ctx:
            owner = factory(scenario, ctx)
            capture_golden._apply_run_attrs(owner, scenario, ctx)
            ph = normalize.PathPlaceholders([(str(ctx.sandbox.root), normalize.SANDBOX_TOKEN)])
            attrs = {}
            for key, value in vars(owner).items():
                if key == "config" or key.startswith("_parity_"):
                    continue
                if isinstance(value, (fakes.FakeSignal, types.MethodType)) or not is_plain(value):
                    continue
                attrs[key] = map_strings(value, ph.apply)
            config = map_strings(copy.deepcopy(owner.config), ph.apply)
        out.append(Base(name, attrs, config))
    return out


def _real_path_globals() -> dict:
    try:
        import app_paths
        return {"CONFIG_FILE": app_paths.CONFIG_FILE, "_APP_DIR": app_paths._APP_DIR}
    except Exception:  # pragma: no cover
        return {"CONFIG_FILE": str(SRC_DIR / "config.json"), "_APP_DIR": str(SRC_DIR)}


def patch_src_module_paths(ctx, extra_file_modules=()) -> None:
    """Point every live src module's CONFIG_FILE/_APP_DIR (and selected __file__) at the sandbox."""
    sb = ctx.sandbox
    real = _real_path_globals()
    real_norm = {k: os.path.normcase(os.path.abspath(v)) for k, v in real.items() if v}
    sandbox_values = {"CONFIG_FILE": str(sb.config_file), "_APP_DIR": str(sb.app_dir)}
    src_norm = os.path.normcase(str(SRC_DIR))
    for mod_name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or not os.path.normcase(os.path.abspath(file)).startswith(src_norm):
            continue
        for attr, real_value in real_norm.items():
            value = module.__dict__.get(attr)
            if isinstance(value, str) and os.path.normcase(os.path.abspath(value)) == real_value:
                ctx.patch(module, attr, sandbox_values[attr])
    for mod_name in ("translator_gui", "app_paths", *extra_file_modules):
        module = sys.modules.get(mod_name)
        if module is not None:
            ctx.patch(module, "__file__", str(sb.src_file(f"{mod_name}.py")))
    for mod_name, module in list(sys.modules.items()):
        value = getattr(module, "CONFIG_FILE", None) if module is not None else None
        file = getattr(module, "__file__", None) if module is not None else None
        if (isinstance(value, str) and file
                and os.path.normcase(os.path.abspath(file)).startswith(src_norm)
                and not os.path.normcase(os.path.abspath(value)).startswith(os.path.normcase(str(sb.root)))):
            raise RuntimeError(f"refusing to fuzz: {mod_name}.CONFIG_FILE {value!r} is outside the sandbox")


def patch_legacy_namespace(ctx, bundle) -> None:
    """The path/backend patches of fakes.make_legacy_owner_factory, applied once per run."""
    ns = bundle.namespace
    sb = ctx.sandbox
    path_globals = {"_APP_DIR": str(sb.app_dir), "CONFIG_FILE": str(sb.config_file)}
    ctx.patch_dict(ns, "__file__", str(sb.src_file("translator_gui.py")))
    for name, value in path_globals.items():
        ctx.patch_dict(ns, name, value)
    for name, stub in ctx.backend_stubs.items():
        if name in ns:
            ctx.patch_dict(ns, name, stub)
    for module, ext_ns in bundle.externals.items():
        ctx.patch_dict(ext_ns, "__file__", str(sb.src_file(f"{module}.py")))
        for name, value in path_globals.items():
            if name in ext_ns:
                ctx.patch_dict(ext_ns, name, value)
    for ext_ns in [ns, *bundle.externals.values()]:
        config_file = ext_ns.get("CONFIG_FILE")
        if config_file is not None and not str(config_file).startswith(str(sb.root)):
            raise RuntimeError(f"refusing to fuzz: legacy CONFIG_FILE {config_file!r} is outside the sandbox")
    fakes.patch_frozen_mixin_paths(ctx, bundle)  # U2+: the frozen shared mixin modules


def patch_translator_gui_backends(ctx) -> None:
    module = sys.modules.get("translator_gui")
    if module is None:
        return
    for name, stub in ctx.backend_stubs.items():
        if name in module.__dict__:
            ctx.patch_dict(module.__dict__, name, stub)


def _module_patch_hooks(bundle):
    """enter/exit pair that swaps the frozen external objects into live modules (legacy side only)."""
    patches = [(importlib.import_module(m), attr, obj) for m, attr, obj in bundle.module_patches()]
    saved: list = []

    def enter():
        saved.clear()
        for module, attr, obj in patches:
            saved.append((module, attr, module.__dict__.get(attr, MISSING)))
            setattr(module, attr, obj)

    def exit_():
        while saved:
            module, attr, old = saved.pop()
            if old is MISSING:
                module.__dict__.pop(attr, None)
            else:
                setattr(module, attr, old)

    return enter, exit_


def fuzz_template_files() -> dict:
    files = _scenario_files()
    files.update(FUZZ_FILES)
    files.update(FUZZ_ARCHIVES)
    return files


def _stub_recorder(recorder, name):
    def stub(*args, **kwargs):
        recorder.record(name, args, kwargs)
        return None

    stub.__name__ = name.rsplit(".", 1)[-1]
    return stub


def _resolve_bases(ctx, bases, setup) -> list:
    out = []
    for base in bases:
        attrs = ctx.resolve(base.attrs)
        if base.name != "empty":
            attrs.update(ctx.resolve(dict(setup or {})))
        out.append(Base(base.name, attrs, ctx.resolve(base.config)))
    return out


def _hints(space_bases, legacy_cls) -> dict:
    attrs, cfg, profiles, models = set(), set(), set(), set()
    for b in space_bases:
        attrs |= set(b.attrs)
        cfg |= set(b.config)
        pp = b.attrs.get("prompt_profiles")
        if isinstance(pp, dict):
            profiles |= {k for k in pp if isinstance(k, str)}
        model = b.attrs.get("model_var")
        if isinstance(model, str):
            models.add(model)
    for alias_attr in ("_IMAGE_MODEL_ALIASES", "_VIDEO_MODEL_ALIASES"):
        value = getattr(legacy_cls, alias_attr, None)
        if isinstance(value, dict):
            models |= {k for k in value if isinstance(k, str)} | {v for v in value.values() if isinstance(v, str)}
        elif isinstance(value, (set, frozenset, list, tuple)):
            models |= {v for v in value if isinstance(v, str)}
    return {"attrs": sorted(attrs), "config_keys": sorted(cfg), "profile_names": sorted(profiles),
            "models": sorted(models)}


def function_seed(name: str, seed: int = DEFAULT_SEED) -> int:
    return (zlib.crc32(name.encode("utf-8")) ^ seed) & 0x7FFFFFFF


def requested_states(default: int = DEFAULT_STATES) -> int:
    try:
        return int(os.environ.get("PARITY_FUZZ_STATES", default))
    except ValueError:
        return default


_SESSION = None
_REPORT_CACHE: dict = {}


def session() -> FuzzSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = FuzzSession()
    return _SESSION


def check_available(spec, sess: FuzzSession, mode: str = "desktop") -> None:
    """Raise Unavailable (skip reason) when tier D cannot run for *spec* yet."""
    if not module_exists(spec.module):
        raise Unavailable(f"src/{spec.module}.py does not exist yet ({spec.milestone} move pending)")
    if not spec.fuzz:
        raise Unavailable(f"tier D disabled for {spec.name}: {spec.reason}")
    mixins = sess.mixin_classes()
    cls = mixins.get(spec.module)
    if cls is None:
        raise Unavailable(f"{spec.module}.{spec.mixin} not defined yet")
    if resolve_python_mro(cls, spec.name) is MISSING:
        msg = f"{spec.mixin}.{spec.name} is not defined"
        if spec.optional:
            raise Unavailable(msg + " (optional move: still a TranslatorGUI method)")
        raise LookupError(msg + " but it is listed in tests/parity/moved_functions.py")
    missing = [n for n in spec.legacy_names if n not in sess.legacy_names()]
    if missing:
        raise Unavailable(f"legacy oracle @ {sess.bundle.sha[:12]} has no {missing}; re-run freeze_legacy.py")
    if mode == "desktop":
        sess.translator_gui_class()


def fuzz_moved(spec, sess: FuzzSession | None = None, *, states: int | None = None, seed: int = DEFAULT_SEED,
               mode: str = "desktop", only_index: int | None = None, stop_after: int = 20,
               check: bool = True, on_state=None, hooks=None) -> FuzzReport:
    """Tier D for one ``moved_functions.Moved`` entry (legacy oracle vs working-tree desktop).

    ``hooks(label) -> (enter, exit)`` (optional) runs around each side's call, inside the
    side's own hooks (tier T trace recorders use it); results are not cached then.

    ``mode='legacy'`` runs the legacy oracle against itself (harness self-check: any
    mismatch means the per-run reset is incomplete or the code is nondeterministic).
    """
    sess = sess or session()
    if check and mode != "legacy":
        check_available(spec, sess, mode)
    states = states or requested_states()
    cache_key = None
    if only_index is None and on_state is None and hooks is None:
        cache_key = (spec.legacy_names, spec.call_name, spec.args, repr(sorted(spec.setup.items())),
                     tuple(getattr(spec, "stubs", ()) or ()), mode, states, seed, stop_after, id(sess))
        if cache_key in _REPORT_CACHE:  # same caller, same states: reuse under this entry's name
            return dataclasses.replace(_REPORT_CACHE[cache_key], label=spec.name)
    bundle = sess.bundle
    legacy_cls = sess.legacy_class()
    new_cls = sess.new_class(mode)
    legacy_names = spec.legacy_names
    new_call = spec.call_name if mode != "legacy" else (legacy_names[0] if len(legacy_names) == 1 else legacy_names)
    legacy_call = legacy_names[0] if len(legacy_names) == 1 else legacy_names

    rs = closure_read_set(legacy_names, class_lookup(legacy_cls))
    rs.update(closure_read_set([new_call] if isinstance(new_call, str) else list(new_call), class_lookup(new_cls)))
    provided = set(sess.bound)
    for cls in (legacy_cls, new_cls):
        provided |= {n for n in rs.attrs if class_lookup(cls)(n) is not None}
    rs.drop_provided(provided)

    sample_fn = class_lookup(legacy_cls)(legacy_names[0])
    if sample_fn is None and isinstance(new_call, str):
        sample_fn = class_lookup(new_cls)(new_call)
    harvested = sess.call_sites(legacy_names[0])
    raw_bases = sess.bases()  # booted in their own sandboxes, never nested in the fuzz sandbox

    report = None
    with FuzzContext(spec.name, backend=True, files=fuzz_template_files()) as ctx:
        patch_legacy_namespace(ctx, bundle)
        file_modules = [m for m in moved_functions.MIXINS if m in sys.modules] + \
                       [m for m in ("headless_owner", "settings_persistence") if m in sys.modules]
        patch_src_module_paths(ctx, extra_file_modules=file_modules)
        patch_translator_gui_backends(ctx)
        install_side_effect_stubs(ctx)
        for module_name, attr in getattr(spec, "stubs", ()) or ():
            ctx.patch(importlib.import_module(module_name), attr,
                      _stub_recorder(ctx.recorder, f"{module_name}.{attr}"))
        namespaces = [bundle.namespace, *bundle.externals.values(), *bundle.mixin_namespaces()]
        if "translator_gui" in sys.modules:
            namespaces.append(sys.modules["translator_gui"].__dict__)
        install_qt_stubs(ctx, rs.qt_names, rs.qt_modules, namespaces)
        bases = _resolve_bases(ctx, raw_bases, spec.setup)
        hints = _hints(bases, legacy_cls)
        paths = [ctx.path(p) for p in FUZZ_PATHS]
        archives = [ctx.path(p) for p in FUZZ_ARCHIVES]
        arg_gen = ArgGen(sample_fn, harvested=harvested, paths=paths, hints=hints, profile=spec.args,
                         archives=archives)
        space = StateSpace(rs, bases, arg_gen=arg_gen, seed=function_seed(spec.via or spec.name, seed),
                           provided=provided)
        enter, exit_ = _module_patch_hooks(bundle)

        def _compose(label, first_enter, first_exit):
            if hooks is None:
                return first_enter, first_exit
            extra_enter, extra_exit = hooks(label)

            def composed_enter():
                if first_enter:
                    first_enter()
                extra_enter()

            def composed_exit():
                try:
                    extra_exit()
                finally:
                    if first_exit:
                        first_exit()

            return composed_enter, composed_exit

        legacy_enter, legacy_exit = _compose("legacy", enter, exit_)
        legacy_side = Side("legacy", legacy_cls, legacy_call, enter=legacy_enter, exit=legacy_exit, bound=sess.bound)
        if mode == "legacy":
            new_enter, new_exit = _compose("new", enter, exit_)
        else:
            new_enter, new_exit = _compose("new", None, None)
        new_side = Side("new", new_cls, new_call, enter=new_enter, exit=new_exit, bound=sess.bound)
        report = differential_fuzz(ctx, legacy_side, new_side, space, states=states, label=spec.name,
                                   only_index=only_index, stop_after=stop_after, on_state=on_state)
        report.notes.append(f"legacy oracle @ {bundle.sha[:12]}; mode={mode}; "
                            f"harvested call sites={len(harvested)}")
    report.violations.extend(ctx.violations)
    if cache_key is not None:
        _REPORT_CACHE[cache_key] = report
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_outcomes(state, a, b):
    out = sys.__stdout__  # sys.stdout is the capture buffer while a state runs
    nl = "\n"
    out.write(json.dumps(state.summary(limit=50), indent=1, ensure_ascii=False, default=repr) + nl)
    for label, res in (("legacy", a), ("new", b)):
        out.write(f"== {label}: exc={res.exc_type} {res.exc_msg[:200]}" + nl)
        out.write(f"   result={str(res.result)[:600]}" + nl)
        out.write(f"   env set={len(res.env.get('set', {}))} removed={len(res.env.get('removed', []))} "
                  f"calls={len(res.calls)} fs={'yes' if res.fs else 'no'} stdout={len(res.stdout)} lines" + nl)
    out.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tier D differential fuzz of moved desktop methods")
    parser.add_argument("names", nargs="*", help="moved names (see --list)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true", help="every entry with tier D enabled")
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=("desktop", "mixins", "legacy"), default="desktop")
    parser.add_argument("--legacy-vs-legacy", action="store_true", help="alias for --mode legacy")
    parser.add_argument("--index", type=int, default=None, help="replay one state verbosely")
    parser.add_argument("--show", type=int, default=3, help="mismatches to print per function")
    parser.add_argument("--force", action="store_true",
                        help="skip the availability checks (e.g. legacy vs the unmodified desktop before a move)")
    args = parser.parse_args(argv)
    mode = "legacy" if args.legacy_vs_legacy else args.mode
    if args.list:
        for m in moved_functions.MOVED:
            status = "exists" if module_exists(m.module) else "pending"
            print(f"{m.name:48s} {m.module}.{m.mixin} [{m.kind}] {status}"
                  + (f" via {m.via}" if m.via else "") + ("" if m.fuzz else " (no fuzz)"))
        return 0
    names = [m.name for m in moved_functions.FUZZED] if args.all else args.names
    if not names:
        parser.error("give moved names, --all or --list")
    sess = session()
    failures = 0
    for name in names:
        spec = moved_functions.get(name)
        try:
            report = fuzz_moved(spec, sess, states=args.states, seed=args.seed, mode=mode,
                                only_index=args.index, check=not args.force,
                                on_state=_print_outcomes if args.index is not None else None)
        except Unavailable as exc:
            print(f"{name}: SKIP ({exc})")
            continue
        except LookupError as exc:
            print(f"{name}: FAIL ({exc})")
            failures += 1
            continue
        except Exception:  # keep going: one broken entry must not hide the others
            import traceback
            print(f"{name}: HARNESS ERROR\n{traceback.format_exc()}")
            failures += 1
            continue
        print(report.failure_text(args.show) if not report.ok else report.summary())
        failures += not report.ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
