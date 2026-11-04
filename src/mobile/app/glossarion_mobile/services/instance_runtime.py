"""Instance runtime: an isolated Glossarion instance inside this process (multi-instance translation, design §6).

A phone cannot start processes, and a translation run cannot live in a child interpreter (lxml, ebooklib.epub,
tiktoken, pydantic_core ... refuse to load there), so a multi-instance run's instances 2..N run in this process,
each as if it were a process of its own. Instance 1 is the process as it always was: every thread that belongs to
no instance (the UI loop, ``gl-job``, ``gl-job-stop``, the dispatcher's io threads) is instance 1 and sees exactly
what it saw before. Nothing here is installed until the first job of an instance >= 2 starts (``ensure``); a session
that never runs one keeps the interpreter untouched.

What ``install`` routes (once per process; ``uninstall`` puts it back, tests only):

* ``os.environ`` (and ``os.getenv``, which reads the same module global) becomes a routing mapping: an instance's
  threads read and write that instance's own full copy (``os._Environ`` rules - str keys and values, case-folded on
  Windows - without ``putenv``, which is process-global), every other thread the real environment;
* ``sys.stdout`` / ``sys.stderr`` / ``sys.argv``: ``sys`` becomes a ``ModuleType`` subclass whose three properties
  hold one value per instance (a job's ``job_runner`` scope tees its own stdout, never another instance's), and the
  ``sys`` dict holds a routing stream so C-level writers (``print``) reach the writing thread's stream;
* ``threading.Thread``: a new thread belongs to the instance of the thread that created it, so pools started by an
  instance's code are that instance's; ``ThreadPoolExecutor.submit`` runs each work item as the submitter's
  instance, whichever thread of a shared pool takes it (asyncio's default executor among them);
* ``builtins.__import__`` / ``importlib.import_module``: the per-instance backend namespace below;
* ``logging``: a record made on an instance's thread carries that instance; a handler added there only takes that
  instance's records, and the handlers added outside every instance (the app's log file and buffers) ignore them.
  The app's own loggers ("glossarion", "glossarion.*") stay the app's wherever they log.

The per-instance backend namespace (``InstanceRuntime``): every module of the backend dir (``src/`` on a desktop
Python, ``<app>/backend`` on a phone) and the job-kind adapters (``glossarion_mobile.job_kinds``) get a fresh copy
per instance, executed from the code object the import system loads (cached once, ``_CODE``, so every instance
shares it) into a new module whose ``__builtins__`` is the instance's own. An instance's copy imports the other
backend modules as that instance's copies (by the importing module's marker, ``MARKER``, wherever its code runs),
and code that is not a copy (``services.jobs``, an adapter's lazy import) gets the copies on that instance's
threads (``current``). ``from X import Y`` therefore binds the instance's ``Y``, and exceptions raised by an
instance's code are that instance's classes; ``services.jobs`` (``JobError``) is never copied, so the classes its
``except`` clauses name are the same for every instance. The base copies are never touched: a copy is made after
its base module is imported (``dataclasses`` resolves annotations through ``sys.modules``), and its import-time
statements write the instance's environment and streams.

An instance's copies see their own stand-ins for the standard modules that hold process-wide registrations:
``sys`` (``modules`` lists the instance's copies, so ``sys.modules.get("unified_api_client")`` in a copy of
``job_runner`` finds the instance's client), ``builtins`` (``TransateKRtoEN``'s ``builtins._DISABLE_ZERO_DETECTION``
flag stays the instance's), ``json`` (its per-run ``json.load`` debug wrapper), ``atexit`` (an import-time
registration never runs twice) and ``signal`` (handlers stay the process's). Third-party packages stay shared.

Shared with every instance (``SHARED_MODULES``): the key material and the Library folders the app installs at its
start, the in-app browser driver it registers, the module that patches ``requests`` / ``httpx`` once per process
and the generated settings data. The Unified Glossary module is copied like any other; its lock is a lock file, so
its copies serialise one read -> merge -> write like separate processes do.

Copies are kept for the instance's next job (a destroyed module copy gives nothing back that a reused one costs).
A module that cannot be copied fails the import with ``InstanceIsolationError``: an instance never falls back to
the shared module.

Pure Python, Python 3.10 compatible, no Flet and no backend import at module scope.
"""

from __future__ import annotations

import builtins
import collections.abc
import concurrent.futures
import contextlib
import copy
import importlib
import importlib.util
import logging
import os
import sys
import threading
import types
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

__all__ = [
    "COPIED_APP_PACKAGES",
    "ENV_INSTANCE",
    "ESSENTIAL_MODULES",
    "InstanceIsolationError",
    "InstanceRuntime",
    "MARKER",
    "SHADOWED_MODULES",
    "SHARED_MODULES",
    "activated",
    "base_argv",
    "base_environ",
    "base_stream",
    "copyable",
    "current",
    "ensure",
    "get",
    "install",
    "installed",
    "loaded_module",
    "runtimes",
    "uninstall",
]

log = logging.getLogger("glossarion.instances")

#: The environment variable ``instance_plan.current_instance()`` reads (set in every instance's environment).
ENV_INSTANCE = "GLOSSARION_INSTANCE"
#: The module-dict key that marks an instance's module copy (its instance number).
MARKER = "__glossarion_instance__"
#: Backend modules every instance shares (never copied): process-wide state the app installs at its start
#: (``secure_keys``: the key material; ``services.library``: the pinned Library folders and caches;
#: ``webview_bridge``: the registered browser driver), the app's own paths (``app_paths``: config.json, the
#: one-writer-per-process lock of its atomic writes), the once-per-process ``requests`` / ``httpx`` patch, the
#: desktop's Qt glue and the generated settings data (constants only; one copy is enough).
SHARED_MODULES = frozenset({
    "api_key_encryption", "token_encryption",
    "library_core", "library_covers", "reader_doc",
    "browser_driver",
    "app_paths",
    "http_logger", "dpi_setup",
    "settings_schema_data",
})
#: App packages copied per instance besides the backend dir: the job-kind adapters install callbacks on the
#: backend modules (``TransateKRtoEN.set_glossary_stage_callback`` ...) and keep their own module state.
COPIED_APP_PACKAGES = ("glossarion_mobile.job_kinds",)
#: Standard modules an instance's copies import as the instance's own stand-in (``InstanceRuntime.shadow``).
SHADOWED_MODULES = frozenset({"sys", "builtins", "json", "atexit", "signal"})
#: Copied when an instance is prepared for a job, so a module that cannot be copied fails the job before it starts.
ESSENTIAL_MODULES = ("job_runner", "stop_control", "direct_text_stream", "headless_owner",
                     "glossarion_mobile.job_kinds")
#: The app's own loggers: their records stay the app's on every thread (``_record_init``).
APP_LOGGER = "glossarion"


class InstanceIsolationError(ImportError):
    """An instance's module could not be made its own (the import fails; an instance never shares instead)."""


_LOCK = threading.RLock()
_tls = threading.local()
_UNSET = object()
_INSTALLED = False
#: number -> InstanceRuntime (kept for the instance's next job)
_RUNTIMES: dict = {}
#: what ``install`` replaced, for ``uninstall``
_ORIGINAL: dict = {}
#: instance 1's stdout / stderr (the ``sys`` dict holds the routing streams while installed)
_BASE: dict = {}
#: (module name, origin) -> code object, shared by every instance
_CODE: dict = {}
_BACKEND_NAMES: Optional[frozenset] = None
_BASE_ENVIRON: Any = None
#: the import entry points as they were before this module was imported (``uninstall`` leaves copies on them)
_ORIG_IMPORT = builtins.__import__
_ORIG_IMPORT_MODULE = importlib.import_module


# ---------------------------------------------------------------------------
# Which instance a thread is
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[int]:
    """An instance number >= 2, or None for instance 1 (the process itself)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 2 else None


def current() -> Optional[int]:
    """The instance this thread runs as (``activated``, else the instance its thread belongs to); None for
    instance 1. Never creates a ``_DummyThread`` (it is called inside ``Thread.__init__``)."""
    try:
        return _tls.number
    except AttributeError:
        pass
    thread = threading._active.get(threading.get_ident())  # noqa: SLF001 - current_thread() would make a dummy
    number = getattr(thread, "_gl_instance", None) if thread is not None else None
    if thread is not None:
        _tls.number = number
    return number


def _runtime_now() -> Optional["InstanceRuntime"]:
    number = current()
    return _RUNTIMES.get(number) if number is not None else None


@contextlib.contextmanager
def activated(number: Any) -> Iterator[None]:
    """Run the block as instance ``number`` (None / 1: the process itself) on this thread: the Stop protocol of an
    instance's job runs on ``gl-job-stop``, a gate refinement on an io thread."""
    previous = getattr(_tls, "number", _UNSET)
    _tls.number = _number(number)
    try:
        yield
    finally:
        if previous is _UNSET:
            try:
                del _tls.number
            except AttributeError:
                pass
        else:
            _tls.number = previous


def _as_instance(number: Optional[int], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn`` as instance ``number`` (a ``ThreadPoolExecutor`` work item runs as the instance that submitted it)."""
    with activated(number):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Routed process state
# ---------------------------------------------------------------------------


class _InstanceEnviron(os._Environ):  # noqa: SLF001 - os.environ's own rules for keys and values
    """An instance's environment: ``os._Environ``'s encoding of keys and values (case-folded keys on Windows,
    ``TypeError`` for a value that is not a str) without ``putenv`` / ``unsetenv``, which change the process."""

    def __setitem__(self, key: str, value: str) -> None:
        self._data[self.encodekey(key)] = self.encodevalue(value)

    def __delitem__(self, key: str) -> None:
        try:
            del self._data[self.encodekey(key)]
        except KeyError:
            raise KeyError(key) from None


def _new_environ(items: Mapping) -> _InstanceEnviron:
    base = _BASE_ENVIRON if _BASE_ENVIRON is not None else os.environ
    env = _InstanceEnviron({}, base.encodekey, base.decodekey, base.encodevalue, base.decodevalue)
    for key, value in dict(items).items():
        env[key] = value
    return env


class _RoutedEnviron(os._Environ):  # noqa: SLF001 - isinstance(os.environ, os._Environ) stays true
    """``os.environ`` while installed: the calling thread's instance environment, else the real one."""

    def __init__(self, base: Any) -> None:  # every method delegates: os._Environ's own state is never used
        self._base = base

    def _env(self) -> Any:
        number = current()
        runtime = _RUNTIMES.get(number) if number is not None else None
        return runtime.env if runtime is not None else self._base

    def __getitem__(self, key: str) -> str:
        return self._env()[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._env()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._env()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._env())

    def __len__(self) -> int:
        return len(self._env())

    def __contains__(self, key: object) -> bool:
        return key in self._env()

    def get(self, key: str, default: Any = None) -> Any:
        return self._env().get(key, default)

    def copy(self) -> dict:
        return dict(self._env())

    def setdefault(self, key: str, value: str) -> str:
        return self._env().setdefault(key, value)

    def __repr__(self) -> str:
        return repr(self._env())

    def __ior__(self, other: Any) -> "_RoutedEnviron":
        self._env().update(other)
        return self

    def __or__(self, other: Any) -> Any:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        new = dict(self._env())
        new.update(other)
        return new

    def __ror__(self, other: Any) -> Any:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        new = dict(other)
        new.update(self._env())
        return new

    def __getattr__(self, name: str) -> Any:  # _data, encodekey, ... of the environment in use
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._env(), name)


class _RoutedStream:
    """``sys.__dict__['stdout' / 'stderr']`` while installed: what C code writes (``print``) reaches the stream of
    the writing thread's instance; Python code reads the stream itself through ``sys``'s properties."""

    def __init__(self, name: str) -> None:
        self._name = name

    def _target(self) -> Any:
        runtime = _runtime_now()
        target = getattr(runtime, self._name) if runtime is not None else _BASE.get(self._name)
        if target is self:  # never write to itself
            target = getattr(sys, f"__{self._name}__", None)
        return target

    def write(self, text: str) -> int:
        target = self._target()
        return target.write(text) if target is not None else len(text)

    def flush(self) -> None:
        target = self._target()
        if target is not None:
            target.flush()

    def __getattr__(self, name: str) -> Any:
        target = self._target()
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)


def _routed_attribute(name: str) -> property:
    """A ``sys`` attribute held per instance; instance 1's lives where it always did (``argv`` in the ``sys``
    dict, the streams in ``_BASE`` because the dict holds the routing streams)."""

    def fget(module: Any) -> Any:
        runtime = _runtime_now()
        if runtime is not None:
            return getattr(runtime, name)
        return _BASE[name] if name in _BASE else vars(module)[name]

    def fset(module: Any, value: Any) -> None:
        runtime = _runtime_now()
        if runtime is not None:
            setattr(runtime, name, value)
        elif name in _BASE:
            _BASE[name] = value
        else:
            vars(module)[name] = value

    return property(fget, fset, doc=f"sys.{name} of the calling thread's instance")


class _RoutedSys(types.ModuleType):
    """``type(sys)`` while installed (a module may take a ``ModuleType`` subclass as its class)."""

    stdout = _routed_attribute("stdout")
    stderr = _routed_attribute("stderr")
    argv = _routed_attribute("argv")


# ---------------------------------------------------------------------------
# An instance's stand-ins for the standard modules with process-wide registrations
# ---------------------------------------------------------------------------


class _InstanceModules(collections.abc.MutableMapping):
    """``sys.modules`` as an instance's copies see it: the instance's copies under their names, the shared
    modules as they are."""

    def __init__(self, runtime: "InstanceRuntime") -> None:
        self._runtime = runtime

    def __getitem__(self, name: str) -> Any:
        if copyable(name):
            return self._runtime.modules[name]
        return sys.modules[name]

    def __setitem__(self, name: str, module: Any) -> None:
        if copyable(name):
            self._runtime.modules[name] = module
        else:
            sys.modules[name] = module

    def __delitem__(self, name: str) -> None:
        if copyable(name):
            del self._runtime.modules[name]
        else:
            del sys.modules[name]

    def __contains__(self, name: object) -> bool:
        if isinstance(name, str) and copyable(name):
            return name in self._runtime.modules
        return name in sys.modules

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        own = dict(self._runtime.modules)
        for name in list(sys.modules):
            if not copyable(name) or name in own:
                yield name
        for name in own:
            if name not in sys.modules:
                yield name

    def __len__(self) -> int:
        return sum(1 for _name in self)


class _InstanceSys(types.ModuleType):
    """``sys`` for an instance's copies: ``modules`` (``_InstanceModules``), and ``stdout`` / ``stderr`` /
    ``argv`` of the instance itself on whatever thread its code runs; everything else is the real ``sys``."""

    _OWN = ("stdout", "stderr", "argv")

    def __init__(self, runtime: "InstanceRuntime") -> None:
        super().__init__("sys", sys.__doc__)
        object.__setattr__(self, "_gl_runtime", runtime)
        object.__setattr__(self, "modules", _InstanceModules(runtime))

    stdout = property(lambda self: self._gl_runtime.stdout, lambda self, v: setattr(self._gl_runtime, "stdout", v))
    stderr = property(lambda self: self._gl_runtime.stderr, lambda self, v: setattr(self._gl_runtime, "stderr", v))
    argv = property(lambda self: self._gl_runtime.argv, lambda self, v: setattr(self._gl_runtime, "argv", v))

    def __getattr__(self, name: str) -> Any:
        return getattr(sys, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(sys, name, value)

    def __dir__(self) -> list:
        return dir(sys)


class _InstanceBuiltins(types.ModuleType):
    """``import builtins`` in an instance's copies: the instance's own builtins dict (the copies'
    ``__builtins__``), so a flag or a patched ``print`` set through it stays the instance's."""

    def __init__(self, namespace: dict) -> None:
        super().__init__("builtins", builtins.__doc__)
        object.__setattr__(self, "_gl_namespace", namespace)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._gl_namespace[name]
        except KeyError:
            raise AttributeError(f"module 'builtins' has no attribute {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._gl_namespace[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self._gl_namespace[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list:
        return sorted(self._gl_namespace)


def _overlay_module(name: str, real: Any) -> types.ModuleType:
    """A module whose own attributes (what the instance sets on it) win over the real module's."""
    module = types.ModuleType(name, getattr(real, "__doc__", None))
    module.__dict__["__getattr__"] = lambda attribute: getattr(real, attribute)
    module.__dict__["__dir__"] = lambda: dir(real)
    return module


def _atexit_module(runtime: "InstanceRuntime") -> types.ModuleType:
    import atexit

    module = _overlay_module("atexit", atexit)

    def register(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
        # the base module registered its own function already: the instance's runs once, when the process ends
        if not runtime.atexit_callbacks:
            atexit.register(runtime.run_atexit)
        runtime.atexit_callbacks.append((func, args, kwargs))
        return func

    def unregister(func: Callable[..., Any]) -> None:
        runtime.atexit_callbacks[:] = [entry for entry in runtime.atexit_callbacks if entry[0] != func]

    module.register = register
    module.unregister = unregister
    return module


def _signal_module() -> types.ModuleType:
    import signal

    module = _overlay_module("signal", signal)

    def keep_handler(signalnum: Any, handler: Any) -> Any:
        return signal.getsignal(signalnum)  # signal handlers are the process's (and only the main thread's)

    module.signal = keep_handler
    return module


# ---------------------------------------------------------------------------
# Which modules an instance copies
# ---------------------------------------------------------------------------


def _backend_dir() -> Optional[str]:
    try:
        from glossarion_mobile import runtime_bootstrap

        paths = runtime_bootstrap.get_paths()
        if paths is not None and paths.backend_dir is not None:
            return os.fspath(paths.backend_dir)
    except Exception:
        log.debug("runtime_bootstrap has no backend dir", exc_info=True)
    spec = importlib.util.find_spec("job_runner")
    origin = getattr(spec, "origin", None) if spec is not None else None
    return os.path.dirname(os.path.abspath(origin)) if origin else None


def _backend_names() -> frozenset:
    global _BACKEND_NAMES
    if _BACKEND_NAMES is None:
        folder = _backend_dir()
        names = set()
        if folder and os.path.isdir(folder):
            for entry in os.listdir(folder):
                stem, ext = os.path.splitext(entry)
                if ext in (".py", ".pyc") and stem.isidentifier() and stem != "__init__":
                    names.add(stem)
        if not names:
            raise InstanceIsolationError("the backend modules were not found, so an instance cannot have its own")
        _BACKEND_NAMES = frozenset(names)
    return _BACKEND_NAMES


def copyable(name: str) -> bool:
    """Whether module ``name`` (absolute) is copied per instance: a backend module not in ``SHARED_MODULES``, or
    a module of a ``COPIED_APP_PACKAGES`` package."""
    if "." not in name:
        return name in (_BACKEND_NAMES or ()) and name not in SHARED_MODULES
    return any(name == package or name.startswith(package + ".") for package in COPIED_APP_PACKAGES)


def _code_key(code: Any) -> tuple:
    return getattr(code, "co_qualname", code.co_name), code.co_firstlineno


def _base_code_index(module: Any, filename: str) -> dict:
    """(qualified name, first line) -> code object of the functions base module ``module`` made from ``filename``
    (module functions, methods, properties, class/static methods, wrapped functions and what they define)."""
    index: dict = {}
    seen: set = set()

    def add(code: Any) -> None:
        if id(code) in seen or code.co_filename != filename:
            return
        seen.add(id(code))
        index.setdefault(_code_key(code), code)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                add(const)

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 8 or id(value) in seen:
            return
        function = getattr(value, "__func__", value)
        for candidate in (function, getattr(function, "__wrapped__", None), getattr(value, "fget", None),
                          getattr(value, "fset", None), getattr(value, "fdel", None)):
            code = getattr(candidate, "__code__", None)
            if isinstance(code, types.CodeType):
                add(code)
        if isinstance(value, type) and getattr(value, "__module__", None) == module.__name__:
            seen.add(id(value))
            for member in list(vars(value).values()):
                visit(member, depth + 1)

    for value in list(vars(module).values()):
        visit(value)
    return index


def _share_consts(code: Any, index: dict) -> Any:
    """``code`` with every nested code object that equals the base module's own replaced by it: an instance's
    functions run the very code objects the base module's run (one copy in memory), only their globals differ."""
    consts = list(code.co_consts)
    changed = False
    for position, const in enumerate(consts):
        if not isinstance(const, types.CodeType):
            continue
        base = index.get(_code_key(const))
        if base is not None and base is not const and base == const:
            consts[position] = base
            changed = True
            continue
        shared = _share_consts(const, index)  # a class body: its methods
        if shared is not const:
            consts[position] = shared
            changed = True
    return code.replace(co_consts=tuple(consts)) if changed else code


def _code_for(name: str, spec: Any) -> Any:
    key = (name, getattr(spec, "origin", None))
    code = _CODE.get(key)
    if code is None:
        get_code = getattr(spec.loader, "get_code", None)
        code = get_code(name) if callable(get_code) else None
        if code is None:
            raise InstanceIsolationError(
                f"{name} cannot be copied for an instance ({type(spec.loader).__name__} gives no code)", name=name)
        base = sys.modules.get(name)
        if base is not None:
            try:
                code = _share_consts(code, _base_code_index(base, code.co_filename))
            except Exception:  # sharing only saves memory: the fresh code object is as good
                log.debug("the code of %s is not shared with its base module", name, exc_info=True)
        _CODE[key] = code
    return code


def _package_of(globals_: Mapping) -> str:
    package = globals_.get("__package__")
    if package is not None:
        return package
    spec = globals_.get("__spec__")
    if spec is not None:
        return spec.parent
    name = str(globals_.get("__name__") or "")
    return name if "__path__" in globals_ else name.rpartition(".")[0]


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------


class InstanceRuntime:
    """Instance ``number``'s process state and module copies (kept for its next job)."""

    def __init__(self, number: int) -> None:
        self.number = int(number)
        self.modules: dict = {}
        self._initializing: set = set()
        self._lock = threading.RLock()
        self.env = _new_environ(_BASE_ENVIRON if _BASE_ENVIRON is not None else os.environ)
        self.argv: list = list(vars(sys).get("argv") or [])
        self.stdout: Any = _BASE.get("stdout", sys.__stdout__)
        self.stderr: Any = _BASE.get("stderr", sys.__stderr__)
        #: the copies' ``__builtins__``: their imports always come through the hook, whatever replaces
        #: ``builtins.__import__`` later
        self.builtins_dict: dict = dict(vars(builtins))
        self.builtins_dict["__import__"] = _import_hook
        self.atexit_callbacks: list = []
        self._shadows: dict = {}

    def __repr__(self) -> str:
        return f"<InstanceRuntime {self.number}: {len(self.modules)} module copies>"

    # ---- process state ------------------------------------------------------------------------

    def stop_file(self) -> str:
        """This instance's ``GLOSSARY_STOP_FILE`` (``stop_control.prepare_glossary_stop_file`` keeps a set value):
        the default name holds only the pid, which every instance of this process shares."""
        import tempfile

        return os.path.join(tempfile.gettempdir(), f"glossarion_glossary_stop_{os.getpid()}_instance{self.number}.flag")

    def prepare(self, *, env: Mapping, argv: Optional[Iterable[str]] = None, stdout: Any = None,
                stderr: Any = None) -> None:
        """A job's start: the instance's environment becomes ``env`` (instance 1's, as it is outside any job) plus
        ``GLOSSARION_INSTANCE`` and its own ``GLOSSARY_STOP_FILE``, key by key (like ``job_runner``'s restore:
        threads still running read unchanged keys the whole time); argv and the streams start as instance 1's."""
        wanted = dict(env)
        wanted[ENV_INSTANCE] = str(self.number)
        wanted["GLOSSARY_STOP_FILE"] = self.stop_file()
        current_env = self.env
        for key in [k for k in list(current_env.keys()) if k not in wanted]:
            current_env.pop(key, None)
        for key, value in wanted.items():
            if current_env.get(key) != value:
                current_env[key] = value
        if argv is not None:
            self.argv = list(argv)
        if stdout is not None:
            self.stdout = stdout
        if stderr is not None:
            self.stderr = stderr

    def run_atexit(self) -> None:
        """What the instance's copies registered with ``atexit`` (the process's exit runs it once, latest first)."""
        callbacks, self.atexit_callbacks = list(self.atexit_callbacks), []
        with activated(self.number):
            for func, args, kwargs in reversed(callbacks):
                try:
                    func(*args, **kwargs)
                except Exception:
                    log.debug("an exit callback of instance %d failed", self.number, exc_info=True)

    def thread(self, target: Callable[..., Any], *, name: str, args: tuple = (), kwargs: Optional[dict] = None,
               daemon: bool = True) -> threading.Thread:
        """A thread that belongs to this instance (not started)."""
        thread = threading.Thread(target=target, name=name, args=args, kwargs=kwargs or {}, daemon=daemon)
        thread._gl_instance = self.number
        return thread

    # ---- modules --------------------------------------------------------------------------------

    def shadow(self, name: str) -> types.ModuleType:
        """The instance's stand-in for standard module ``name`` (``SHADOWED_MODULES``)."""
        module = self._shadows.get(name)
        if module is None:
            with self._lock:
                module = self._shadows.get(name)
                if module is None:
                    if name == "sys":
                        module = _InstanceSys(self)
                    elif name == "builtins":
                        module = _InstanceBuiltins(self.builtins_dict)
                    elif name == "atexit":
                        module = _atexit_module(self)
                    elif name == "signal":
                        module = _signal_module()
                    else:
                        module = _overlay_module(name, sys.modules.get(name) or _ORIG_IMPORT_MODULE(name))
                    self._shadows[name] = module
        return module

    def loaded(self, name: str) -> Any:
        """``sys.modules.get(name)`` for this instance (its copy if it made one, else None; a shared module as is)."""
        if copyable(name):
            return self.modules.get(name)
        return sys.modules.get(name)

    def module(self, name: str) -> Any:
        """This instance's copy of module ``name`` (made on first use)."""
        module = self.modules.get(name)
        if module is not None and name not in self._initializing:
            return module
        with self._lock:  # another thread of this instance may be executing it: wait for it
            module = self.modules.get(name)
            if module is not None:
                return module  # complete, or this thread's own circular import (like the import system)
            return self._create(name)

    def warm(self, names: Iterable[str]) -> None:
        """Copy ``names`` now (``ESSENTIAL_MODULES`` when a job is prepared); ``InstanceIsolationError`` names the
        module that could not be copied."""
        for name in names:
            try:
                self.module(name)
            except InstanceIsolationError:
                raise
            except BaseException as exc:
                raise InstanceIsolationError(f"{name}: {type(exc).__name__}: {exc}", name=name) from exc

    def _create(self, name: str) -> Any:
        parent_name, _dot, child = name.rpartition(".")
        parent = self.module(parent_name) if parent_name and copyable(parent_name) else None
        if name not in sys.modules:
            # The base module first: what resolves annotations or pickles by module name (dataclasses, typing)
            # looks in sys.modules, and a native module the copy needs is loaded where the process expects it.
            with activated(None):
                _ORIG_IMPORT_MODULE(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        code = _code_for(name, spec)
        module = types.ModuleType(name, None)
        namespace = module.__dict__
        is_package = spec.submodule_search_locations is not None
        try:
            cached = spec.cached
        except Exception:
            cached = None
        namespace.update({
            "__spec__": copy.copy(spec),
            "__loader__": spec.loader,
            "__file__": spec.origin,
            "__cached__": cached,
            "__package__": name if is_package else parent_name,
            "__builtins__": self.builtins_dict,
            MARKER: self.number,
        })
        if is_package:
            namespace["__path__"] = list(spec.submodule_search_locations)
        self.modules[name] = module
        self._initializing.add(name)
        try:
            with activated(self.number):
                exec(code, namespace)
        except BaseException:
            self.modules.pop(name, None)
            raise
        finally:
            self._initializing.discard(name)
        if parent is not None:
            setattr(parent, child, module)
        log.debug("instance %d: copied %s", self.number, name)
        return module

    def _from_list(self, module: Any, name: str, fromlist: Iterable[Any]) -> None:
        """``from package import submodule`` for a copied package (the import system's ``_handle_fromlist``)."""
        if not hasattr(module, "__path__"):
            return
        for item in fromlist:
            if not isinstance(item, str):
                continue
            if item == "*":
                names = getattr(module, "__all__", None)
                if names:
                    self._from_list(module, name, [n for n in names if n != "*"])
                continue
            if hasattr(module, item):
                continue
            sub = f"{name}.{item}"
            if not copyable(sub):
                continue
            try:
                self.module(sub)
            except ModuleNotFoundError as exc:
                if exc.name != sub:  # like the import system: only a missing submodule itself is skipped
                    raise

    def import_(self, name: str, globals_: Optional[Mapping], locals_: Any, fromlist: Any, level: int,
                from_copy: bool) -> Any:
        """``__import__`` for this instance."""
        if level:
            if not from_copy or globals_ is None:
                return _fallback_import(name, globals_, locals_, fromlist, level)
            absolute = importlib.util.resolve_name("." * level + name, _package_of(globals_))
        else:
            absolute = name
        top = absolute.partition(".")[0]
        if from_copy and not level and top in SHADOWED_MODULES:
            real = _fallback_import(name, globals_, locals_, fromlist, level)  # the real module (and submodules)
            if fromlist and absolute != top:
                return real  # from json import decoder: the real submodule
            return self.shadow(top)
        if not copyable(absolute):
            return _fallback_import(name, globals_, locals_, fromlist, level)
        module = self.module(absolute)
        if fromlist:
            self._from_list(module, absolute, fromlist)
            return module
        if "." in absolute:  # import a.b.c binds a
            return self.module(top) if copyable(top) else _fallback_import(top, globals_, locals_, (), 0)
        return module


# ---------------------------------------------------------------------------
# The process-wide hooks
# ---------------------------------------------------------------------------


def _fallback_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """An import the instances do not route: the ``__import__`` that was in place before ours, so another hook
    keeps working (PySide6's shiboken installs one the first time it is imported). A hook that calls ours again
    (installed after ours, it took ours as its original) gets the plain import system then, never a loop."""
    previous = _ORIGINAL.get("builtins.__import__")
    if previous is None or previous is _import_hook or getattr(_tls, "in_fallback", False):
        return _ORIG_IMPORT(name, globals_, locals_, fromlist, level)
    _tls.in_fallback = True
    try:
        return previous(name, globals_, locals_, fromlist, level)
    finally:
        _tls.in_fallback = False


def _keep_import_hook() -> None:
    """Another hook replaced ``builtins.__import__`` after ``install`` (shiboken does when PySide6 is first
    imported while a job starts): put ours back in front of it; it stays the fallback (``_fallback_import``)."""
    if not _INSTALLED or builtins.__import__ is _import_hook:
        return
    with _LOCK:
        if _INSTALLED and builtins.__import__ is not _import_hook:
            _ORIGINAL["builtins.__import__"] = builtins.__import__
            builtins.__import__ = _import_hook
            log.info("instance runtime: the import hook is back in front of %r", _ORIGINAL["builtins.__import__"])


def _import_hook(name: str, globals: Optional[Mapping] = None, locals: Any = None, fromlist: Any = (),  # noqa: A002
                 level: int = 0) -> Any:
    number = None
    if globals is not None:
        try:
            number = globals.get(MARKER)
        except Exception:
            number = None
    from_copy = number is not None
    if number is None:
        number = current()
        if number is None:
            return _fallback_import(name, globals, locals, fromlist, level)
    runtime = _RUNTIMES.get(number)
    if runtime is None:
        return _fallback_import(name, globals, locals, fromlist, level)
    return runtime.import_(name, globals, locals, fromlist, level, from_copy)


def _import_module(name: str, package: Optional[str] = None) -> Any:
    try:
        number = sys._getframe(1).f_globals.get(MARKER)  # noqa: SLF001 - the caller's module decides, like __import__
    except Exception:
        number = None
    if number is None:
        number = current()
    runtime = _RUNTIMES.get(number) if number is not None else None
    if runtime is None:
        return _ORIG_IMPORT_MODULE(name, package)
    absolute = importlib.util.resolve_name(name, package) if name.startswith(".") else name
    if not copyable(absolute):
        return _ORIG_IMPORT_MODULE(name, package)
    return runtime.module(absolute)


def _thread_init(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL["Thread.__init__"](self, *args, **kwargs)
    self._gl_instance = current()
    if builtins.__import__ is not _import_hook:
        _keep_import_hook()


def _submit(self: Any, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    if builtins.__import__ is not _import_hook:
        _keep_import_hook()
    return _ORIGINAL["ThreadPoolExecutor.submit"](self, _as_instance, current(), fn, *args, **kwargs)


def _record_init(self: logging.LogRecord, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL["LogRecord.__init__"](self, *args, **kwargs)
    number = current()
    if number is not None:
        name = str(getattr(self, "name", "") or "")
        if name != APP_LOGGER and not name.startswith(APP_LOGGER + "."):
            self._gl_instance = number


def _add_handler(self: logging.Logger, handler: logging.Handler) -> None:
    number = current()
    if number is not None and getattr(handler, "_gl_instance", None) is None:
        try:
            handler._gl_instance = number
        except Exception:
            pass
    return _ORIGINAL["Logger.addHandler"](self, handler)


def _handle(self: logging.Handler, record: logging.LogRecord) -> Any:
    if getattr(self, "_gl_instance", None) != getattr(record, "_gl_instance", None):
        return False
    return _ORIGINAL["Handler.handle"](self, record)


def installed() -> bool:
    return _INSTALLED


def install() -> None:
    """Route the process state per instance (idempotent). ``InstanceIsolationError`` when it cannot: nothing stays
    patched then."""
    global _INSTALLED, _BASE_ENVIRON
    with _LOCK:
        if _INSTALLED:
            return
        try:
            _backend_names()
            if not isinstance(os.environ, os._Environ):  # noqa: SLF001
                raise InstanceIsolationError(f"os.environ is a {type(os.environ).__name__}, not os._Environ")
            if type(sys) is not types.ModuleType:
                raise InstanceIsolationError(f"sys is already a {type(sys).__name__}")
            _ORIGINAL.update({
                "environ": os.environ,
                "builtins.__import__": builtins.__import__,
                "importlib.import_module": importlib.import_module,
                "Thread.__init__": threading.Thread.__init__,
                "ThreadPoolExecutor.submit": concurrent.futures.ThreadPoolExecutor.submit,
                "LogRecord.__init__": logging.LogRecord.__init__,
                "Logger.addHandler": logging.Logger.addHandler,
                "Handler.handle": logging.Handler.handle,
            })
            _BASE_ENVIRON = os.environ
            interval = sys.getswitchinterval()
            sys.setswitchinterval(10.0)  # keep the GIL: no other thread may set sys.stdout halfway through
            try:
                _BASE["stdout"] = vars(sys)["stdout"]
                _BASE["stderr"] = vars(sys)["stderr"]
                sys.__class__ = _RoutedSys
                vars(sys)["stdout"] = _RoutedStream("stdout")
                vars(sys)["stderr"] = _RoutedStream("stderr")
                os.environ = _RoutedEnviron(_BASE_ENVIRON)
            finally:
                sys.setswitchinterval(interval)
            builtins.__import__ = _import_hook
            importlib.import_module = _import_module
            threading.Thread.__init__ = _thread_init
            concurrent.futures.ThreadPoolExecutor.submit = _submit
            logging.LogRecord.__init__ = _record_init
            logging.Logger.addHandler = _add_handler
            logging.Handler.handle = _handle
            _INSTALLED = True
        except BaseException as exc:
            _restore()
            if isinstance(exc, InstanceIsolationError):
                raise
            raise InstanceIsolationError(f"the instance runtime could not be installed: {type(exc).__name__}: {exc}") \
                from exc
    log.info("instance runtime installed (%d backend modules can be copied per instance)", len(_BACKEND_NAMES or ()))


def _restore() -> None:
    """Put back what ``install`` replaced (caller holds ``_LOCK``)."""
    global _BASE_ENVIRON
    patches = (
        (builtins, "__import__", "builtins.__import__"),
        (importlib, "import_module", "importlib.import_module"),
        (threading.Thread, "__init__", "Thread.__init__"),
        (concurrent.futures.ThreadPoolExecutor, "submit", "ThreadPoolExecutor.submit"),
        (logging.LogRecord, "__init__", "LogRecord.__init__"),
        (logging.Logger, "addHandler", "Logger.addHandler"),
        (logging.Handler, "handle", "Handler.handle"),
    )
    ours = {"builtins.__import__": _import_hook, "importlib.import_module": _import_module,
            "Thread.__init__": _thread_init, "ThreadPoolExecutor.submit": _submit, "LogRecord.__init__": _record_init,
            "Logger.addHandler": _add_handler, "Handler.handle": _handle}
    for owner, attribute, key in patches:
        # only where ours still stands: a hook installed on top of ours since then stays (it chains to the original)
        if key in _ORIGINAL and getattr(owner, attribute, None) is ours[key]:
            setattr(owner, attribute, _ORIGINAL[key])
    if type(sys) is _RoutedSys:
        interval = sys.getswitchinterval()
        sys.setswitchinterval(10.0)
        try:
            stdout, stderr = _BASE.get("stdout"), _BASE.get("stderr")
            sys.__class__ = types.ModuleType
            if isinstance(vars(sys).get("stdout"), _RoutedStream):
                vars(sys)["stdout"] = stdout
            if isinstance(vars(sys).get("stderr"), _RoutedStream):
                vars(sys)["stderr"] = stderr
        finally:
            sys.setswitchinterval(interval)
    if isinstance(os.environ, _RoutedEnviron) and "environ" in _ORIGINAL:
        os.environ = _ORIGINAL["environ"]
    _ORIGINAL.clear()
    _BASE.clear()
    _BASE_ENVIRON = None


def uninstall() -> None:
    """Undo ``install`` and forget every instance (tests: the app keeps it for its lifetime)."""
    global _INSTALLED
    with _LOCK:
        _restore()
        import atexit

        for runtime in _RUNTIMES.values():
            atexit.unregister(runtime.run_atexit)
        _RUNTIMES.clear()
        _INSTALLED = False


def ensure(number: Any) -> InstanceRuntime:
    """Instance ``number``'s runtime (>= 2), installing the routing first; kept for the instance's next job."""
    wanted = _number(number)
    if wanted is None:
        raise ValueError(f"instance {number!r} is the process itself (instance 1)")
    install()
    _keep_import_hook()
    with _LOCK:
        runtime = _RUNTIMES.get(wanted)
        if runtime is None:
            runtime = _RUNTIMES[wanted] = InstanceRuntime(wanted)
            log.info("instance %d created", wanted)
        return runtime


def get(number: Any) -> Optional[InstanceRuntime]:
    wanted = _number(number)
    return _RUNTIMES.get(wanted) if wanted is not None else None


def runtimes() -> tuple:
    """The instances made so far, in number order."""
    with _LOCK:
        return tuple(_RUNTIMES[number] for number in sorted(_RUNTIMES))


def loaded_module(name: str, number: Any = _UNSET) -> Any:
    """``sys.modules.get(name)`` as the calling thread's instance (or instance ``number``) sees it."""
    wanted = current() if number is _UNSET else _number(number)
    runtime = _RUNTIMES.get(wanted) if wanted is not None else None
    return runtime.loaded(name) if runtime is not None else sys.modules.get(name)


def base_environ() -> Any:
    """Instance 1's environment (the process's own ``os._Environ``) from any thread."""
    return _BASE_ENVIRON if _INSTALLED and _BASE_ENVIRON is not None else os.environ


def base_argv() -> list:
    """Instance 1's ``sys.argv``."""
    return vars(sys)["argv"]


def base_stream(name: str) -> Any:
    """Instance 1's ``sys.stdout`` / ``sys.stderr``."""
    if _INSTALLED and name in _BASE:
        return _BASE[name]
    return vars(sys)[name]
