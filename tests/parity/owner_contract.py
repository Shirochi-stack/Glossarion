#!/usr/bin/env python3
"""Owner contract: the ``self.<name>`` reads the shared mixins make WITHOUT a guard.

Every attribute the moved code reads unguarded must exist on any owner that
runs it (``TranslatorGUI`` on desktop, ``HeadlessOwner`` on mobile), or the
run fails with ``AttributeError``. This scanner derives that set by AST from
the shared mixin modules (``moved_functions.MIXINS``), independently of
``headless_owner.OWNER_CONTRACT``, so the tests can check both the declared
contract and real owners against it.

A read ``self.x`` counts as *guarded* when it sits

* inside ``if``/``while``/ternary whose test calls ``hasattr(self, 'x')`` or
  ``getattr(self, 'x', ...)`` (body only, not the ``else`` branch);
* after ``hasattr(self, 'x') and ...`` / ``getattr(self, 'x', ...) and ...`` in a
  boolean ``and``;
* in the body of a ``try`` whose handlers catch ``AttributeError``,
  ``Exception``, ``BaseException`` or everything (or ``contextlib.suppress`` of
  those);

``getattr(self, 'x', default)`` / ``hasattr(self, 'x')`` themselves never count.
Names defined by the scanned mixin classes (methods, class attributes, also
inherited between them) are *class provided*; names assigned ``self.x = ...``
earlier in the same method are *locally provided*. Everything else read
unguarded is the contract. ``init_provided`` lists contract names assigned by
the ConfigStateMixin init methods (the owner gets them by running init);
``external`` are the rest (widgets/shims, GUI recorders, hooks, run attrs).

CLI::

    python tests/parity/owner_contract.py [--modules run_env owner_state ...] [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

#: ConfigStateMixin methods that run while an owner is built (their self.x writes
#: are available to every later read).
INIT_METHODS = ("_init_config_state", "_init_variables", "_init_gui_backed_state",
                "_init_default_prompts", "__init__")

#: name -> reason. Contract names an owner only gets at run time (set by the job code
#: right before the moved method runs), so a freshly built owner may lack them.
#: Every entry needs a reason; the owner-contract tests list anything else as missing.
RUNTIME_ATTRS: dict = {}

_GUARD_EXCEPTIONS = {"AttributeError", "Exception", "BaseException"}
_SELF = ("self",)


@dataclass(frozen=True)
class Site:
    module: str
    cls: str
    method: str
    line: int
    kind: str  # 'attr' | 'call' | 'widget'

    def __str__(self):
        return f"{self.module}.{self.cls}.{self.method}:{self.line} ({self.kind})"


@dataclass
class ContractScan:
    unguarded: dict = field(default_factory=dict)       # name -> [Site]
    guarded: dict = field(default_factory=dict)         # name -> [Site]
    locally_provided: dict = field(default_factory=dict)
    class_provided: set = field(default_factory=set)
    init_assigned: set = field(default_factory=set)
    stored: set = field(default_factory=set)            # self.x = ... anywhere in the scanned classes
    classes: dict = field(default_factory=dict)         # module -> [class names]

    def names(self, kinds=("attr", "call", "widget")) -> frozenset:
        """The contract: unguarded reads minus class-provided names."""
        out = set()
        for name, sites in self.unguarded.items():
            if name in self.class_provided or name.startswith("__"):
                continue
            if any(s.kind in kinds for s in sites):
                out.add(name)
        return frozenset(out)

    @property
    def contract(self) -> frozenset:
        return self.names()

    @property
    def init_provided(self) -> frozenset:
        return frozenset(self.contract & self.init_assigned)

    @property
    def external(self) -> frozenset:
        """Contract names the init methods do not assign (shims, GUI recorders, hooks, run attrs)."""
        return frozenset(self.contract - self.init_assigned)

    @property
    def unprovided(self) -> frozenset:
        """Contract names no scanned method ever assigns: the owner class must bring them
        (same scope as ``headless_owner.OWNER_CONTRACT``: reads - stores - class names)."""
        return frozenset(self.contract - self.stored)

    def sites(self, name) -> list:
        return list(self.unguarded.get(name, ()))

    def report(self, verbose=False) -> str:
        lines = [f"contract: {len(self.contract)} names ({len(self.init_provided)} set by init, "
                 f"{len(self.external)} external); class provided: {len(self.class_provided)}"]
        for name in sorted(self.external):
            first = self.unguarded[name][0]
            lines.append(f"  external {name:42s} {first}" + (f" (+{len(self.unguarded[name]) - 1})"
                                                             if len(self.unguarded[name]) > 1 else ""))
        if verbose:
            for name in sorted(self.init_provided):
                lines.append(f"  init     {name:42s} {self.unguarded[name][0]}")
        return "\n".join(lines)


def _guard_names(test) -> set:
    """Attribute names a test expression guards (hasattr/getattr(self, 'x' ...))."""
    out = set()
    for node in ast.walk(test):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id in _SELF
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            out.add(node.args[1].value)
        elif (isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant)
              and isinstance(node.left.value, str)
              and any(isinstance(op, ast.In) for op in node.ops)
              and any(ast.unparse(c) in ("vars(self)", "self.__dict__") for c in node.comparators)):
            out.add(node.left.value)
    return out


def _handler_guards(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types_ = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(ast.unparse(t).split(".")[-1] in _GUARD_EXCEPTIONS for t in types_)


def _suppress_guards(with_node) -> bool:
    for item in with_node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Call) and ast.unparse(expr.func).split(".")[-1] == "suppress":
            if any(ast.unparse(a).split(".")[-1] in _GUARD_EXCEPTIONS for a in expr.args):
                return True
    return False


def _is_guarded(node, name, parents) -> bool:
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, (ast.If, ast.While)) and child in parent.body and name in _guard_names(parent.test):
            return True
        if isinstance(parent, ast.IfExp) and child is parent.body and name in _guard_names(parent.test):
            return True
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            idx = parent.values.index(child) if child in parent.values else -1
            if idx > 0 and any(name in _guard_names(v) for v in parent.values[:idx]):
                return True
        if isinstance(parent, ast.Try) and child in parent.body and any(_handler_guards(h) for h in parent.handlers):
            return True
        if isinstance(parent, ast.With) and child in parent.body and _suppress_guards(parent):
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and parents.get(parent) is None:
            break
        child, parent = parent, parents.get(parent)
    return False


def _class_defined_names(cls: ast.ClassDef) -> set:
    out = set()
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.add(stmt.target.id)
    return out


def scan_source(text: str, module: str, class_names=None, scan: ContractScan | None = None) -> ContractScan:
    """Scan the classes (*class_names*, default: all) of one module source."""
    scan = scan or ContractScan()
    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and (class_names is None or n.name in class_names)]
    scan.classes.setdefault(module, []).extend(c.name for c in classes)
    for cls in classes:
        scan.class_provided |= _class_defined_names(cls)
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            _scan_method(method, module, cls.name, scan)
    return scan


def _scan_method(method, module, cls_name, scan: ContractScan) -> None:
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    assigned_lines: dict = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in _SELF \
                and isinstance(node.ctx, ast.Store):
            assigned_lines.setdefault(node.attr, node.lineno)
            scan.stored.add(node.attr)
            if method.name in INIT_METHODS:
                scan.init_assigned.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
              and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in _SELF
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            assigned_lines.setdefault(node.args[1].value, node.lineno)
            scan.stored.add(node.args[1].value)
            if method.name in INIT_METHODS:
                scan.init_assigned.add(node.args[1].value)
    if method.name in INIT_METHODS:
        # dynamic setattr(self, var_name, ...) driven by literal tables (bool_vars/str_vars)
        for node in ast.walk(method):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_var"):
                scan.init_assigned.add(node.value)
    for node in ast.walk(method):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in _SELF):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        name = node.attr
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            kind = "call"
        elif isinstance(parent, ast.Attribute) and parent.value is node and isinstance(parents.get(parent), ast.Call):
            kind = "widget"
        else:
            kind = "attr"
        site = Site(module, cls_name, method.name, node.lineno, kind)
        if name in assigned_lines and assigned_lines[name] < node.lineno:
            scan.locally_provided.setdefault(name, []).append(site)
        elif _is_guarded(node, name, parents):
            scan.guarded.setdefault(name, []).append(site)
        else:
            scan.unguarded.setdefault(name, []).append(site)


def scan_modules(modules=None, *, src_dir=SRC_DIR) -> ContractScan:
    """Scan the shared mixin modules that exist (default: ``moved_functions.MIXINS``)."""
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from parity import moved_functions

    mixins = dict(moved_functions.MIXINS)
    modules = list(modules or mixins)
    scan = ContractScan()
    for module in modules:
        path = Path(src_dir) / f"{module}.py"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        scan_source(text, module, None, scan)
    return scan


def missing_on(owner, names) -> list:
    """Contract names *owner* does not provide (instance or class attribute)."""
    return sorted(n for n in names if not hasattr(owner, n))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Unguarded self.<attr> reads of the shared mixins")
    parser.add_argument("--modules", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    scan = scan_modules(args.modules)
    if not any(scan.classes.values()):
        print("no shared mixin module exists yet")
        return 0
    print(scan.report(verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
