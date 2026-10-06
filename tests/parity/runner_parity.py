"""Runner-level trace parity for the U7 moves (image_job / rpgmaker_job), CI-capable.

The desktop runners ``TranslatorGUI._run_generative_prompt_mode``, ``_process_image_file`` and
``_process_rpgmaker_game`` moved verbatim into ``image_job.ImageJobMixin`` and
``rpgmaker_job.RpgMakerJobMixin``. This harness drives the LEGACY method (its text at
``LEGACY_SHA`` read with ``git show`` and compiled in a translator_gui-like namespace: ``os``,
``json``, ``__file__``) and the NEW mixin method on the same kind of owner, each in its own
scrubbed sandbox with an identical fixture tree, with every backend a recording stub:

* ``unified_api_client.UnifiedClient``: constructor, every ``set/clear_in_memory_*`` key pool,
  ``send`` and ``send_image`` (the vision / image-edit / video / audio call: the client routes by
  the output-mode environment, which each call records), ``_model_needs_api_key`` delegates to
  the real class;
* ``TransateKRtoEN.send_with_interrupt`` (records its arguments and the stop answer, then calls
  ``client.send`` like the real one);
* ``rpgmaker_handler.translate_game_images`` (image mode; the text mode runs the real
  ``rpgmaker_handler`` extract / chunk / parse / apply functions on the fixture game);
* the clock (``time.time`` / ``time.strftime`` / ``datetime.now``) is fixed.

One run records an ordered event list (log lines, backend calls with the environment delta they
saw), the return value, the final environment delta, owner attributes and the whole sandbox
tree (JSON parsed, text verbatim, binaries hashed), with the sandbox path replaced by
``<ROOT>``. Legacy and new must be equal.

The legacy text needs git history (``fetch-depth: 0`` in CI); no PySide6, no frozen oracle.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
TESTS_DIR = PARITY_DIR.parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The commit the U7 runners were moved from (translator_gui.py before the move).
LEGACY_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"
RUNNERS = ("_run_generative_prompt_mode", "_process_image_file", "_process_rpgmaker_game")
FIXED_TIME = 1_700_000_000.0
ROOT_TOKEN = "<ROOT>"

_LEGACY_CACHE: dict = {}


class Unavailable(RuntimeError):
    """The legacy source cannot be read (no git history)."""


# ---------------------------------------------------------------------------
# legacy source
# ---------------------------------------------------------------------------


def git_show(sha: str, relpath: str) -> str:
    try:
        raw = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT), capture_output=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Unavailable(f"git show {sha[:12]}:{relpath} failed: {exc}") from exc
    return raw.decode("utf-8-sig").replace("\r\n", "\n")


def legacy_translator_gui() -> str:
    if "tg" not in _LEGACY_CACHE:
        _LEGACY_CACHE["tg"] = git_show(LEGACY_SHA, "src/translator_gui.py")
    return _LEGACY_CACHE["tg"]


def _class_methods(text: str, class_name: str) -> dict:
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def node_text(text: str, node) -> str:
    lines = text.split("\n")
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(lines[start - 1:node.end_lineno])


def legacy_method_text(name: str) -> str:
    if "methods" not in _LEGACY_CACHE:
        text = legacy_translator_gui()
        _LEGACY_CACHE["methods"] = {n: node_text(text, node)
                                    for n, node in _class_methods(text, "TranslatorGUI").items()}
    return _LEGACY_CACHE["methods"][name]


def legacy_functions(src_file: str, names=RUNNERS) -> dict:
    """The legacy methods compiled as plain functions; globals = translator_gui's (os, json, __file__).

    They are compiled inside a class body exactly as indented in TranslatorGUI (dedenting
    would also strip the indentation inside their multi-line strings, e.g. the image HTML).
    """
    key = ("code", tuple(names))
    if key not in _LEGACY_CACHE:
        source = "class _LegacyTranslatorGUI:\n" + "\n\n".join(legacy_method_text(n) for n in names) + "\n"
        _LEGACY_CACHE[key] = compile(source, f"<translator_gui@{LEGACY_SHA[:12]}>", "exec")
    ns = {"os": os, "json": json, "__file__": src_file, "__name__": "translator_gui",
          "__builtins__": __builtins__}
    exec(_LEGACY_CACHE[key], ns)
    cls = ns.pop("_LegacyTranslatorGUI")
    return {n: vars(cls)[n] for n in names}


def module_text(module: str) -> str:
    return (SRC_DIR / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self, root: str):
        self.root = str(root)
        self.events: list = []
        self.start_env: dict = {}

    def norm(self, value):
        return normalize(value, self.root)

    def log(self, message):
        self.events.append(("log", self.norm(mask_traceback(str(message)))))

    def call(self, name, payload=None):
        self.events.append((name, self.norm(payload if payload is not None else {})))

    def env_delta(self) -> dict:
        now = dict(os.environ)
        changed = {k: v for k, v in now.items() if self.start_env.get(k) != v}
        removed = sorted(k for k in self.start_env if k not in now)
        return {"set": dict(sorted(changed.items())), "removed": removed}


def mask_traceback(text: str) -> str:
    """A logged traceback without its frames (file names, line numbers and source lines differ:
    the legacy methods are compiled from a string, the new ones live in image_job.py)."""
    if "Traceback (most recent call last):" not in text:
        return text
    out, in_frame = [], False
    for line in text.split("\n"):
        if line.startswith('  File "'):
            if not (out and out[-1] == "  <frames>"):
                out.append("  <frames>")
            in_frame = True
            continue
        if in_frame and line.startswith("    "):
            continue  # a frame's source line or its caret marker
        in_frame = False
        out.append(line)
    return "\n".join(out)


def normalize(value, root: str):
    root = str(root)
    # plain, forward-slash, backslash and repr-escaped (an OSError message quotes the path) forms
    variants = sorted({root, root.replace("\\", "/"), root.replace("/", "\\"),
                       root.replace("/", "\\").replace("\\", "\\\\")}, key=len, reverse=True)

    def fix(s):
        for v in variants:
            s = s.replace(v, ROOT_TOKEN)
        return s.replace(ROOT_TOKEN + "\\", ROOT_TOKEN + "/")

    def walk(v):
        if isinstance(v, str):
            return fix(v)
        if isinstance(v, dict):
            return {walk(k): walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        if isinstance(v, float):
            return {"float": repr(v)}
        if v is None or isinstance(v, (bool, int)):
            return v
        return f"<{type(v).__name__}>"

    return walk(value)


def _summarize_messages(messages):
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({k: (v if not isinstance(v, (bytes, bytearray)) else f"<{len(v)} bytes>") for k, v in m.items()})
        else:
            out.append(repr(m))
    return out


def _stub_response(plan, index, root, payload):
    if index < len(plan):
        response = plan[index]
    else:
        return None
    if isinstance(response, BaseException):
        raise response
    if callable(response):
        return response(root, payload)
    return response


def make_fake_client(recorder: Recorder, scenario):
    """A UnifiedClient stand-in: records construction, key pools and send / send_image calls."""
    import unified_api_client

    real = unified_api_client.UnifiedClient
    counters = {"send": 0, "send_image": 0}
    root = recorder.root

    def pool(name):
        def method(cls, *args, **kwargs):
            recorder.call(f"UnifiedClient.{name}", {"args": list(args), "kwargs": dict(kwargs)})
            return True
        method.__name__ = name
        return classmethod(method)

    def __init__(self, *args, **kwargs):
        recorder.call("UnifiedClient.__init__", {"args": list(args), "kwargs": dict(kwargs)})

    def send(self, messages=None, temperature=None, max_tokens=None, **kwargs):
        index = counters["send"]
        counters["send"] += 1
        payload = {"messages": _summarize_messages(messages), "temperature": temperature,
                   "max_tokens": max_tokens, "kwargs": dict(kwargs), "env": recorder.env_delta()}
        recorder.call("UnifiedClient.send", payload)
        response = _stub_response(scenario.send, index, root, payload)
        if response is None:
            response = f"[stub send {index}]"
        return response if isinstance(response, tuple) else (response, "stop")

    def send_image(self, messages, image_data, temperature=None, max_tokens=None, response_name=None, **kwargs):
        index = counters["send_image"]
        counters["send_image"] += 1
        data = image_data if isinstance(image_data, (bytes, bytearray)) else str(image_data or "").encode("utf-8")
        payload = {"messages": _summarize_messages(messages), "image_b64_sha256": hashlib.sha256(data).hexdigest(),
                   "image_b64_len": len(data), "temperature": temperature, "max_tokens": max_tokens,
                   "response_name": response_name, "kwargs": dict(kwargs), "env": recorder.env_delta()}
        recorder.call("UnifiedClient.send_image", payload)
        response = _stub_response(scenario.send_image, index, root, payload)
        if response is None:
            response = f"<p>stub image response {index}</p>"
        return response if isinstance(response, tuple) else (response, "stop")

    ns = {"__init__": __init__, "send": send, "send_image": send_image,
          "_model_needs_api_key": classmethod(lambda cls, model: real._model_needs_api_key(model))}
    for name in dir(real):
        if name.startswith(("set_in_memory_", "clear_in_memory_")):
            ns[name] = pool(name)
    return type("RecordingUnifiedClient", (), ns)


def make_send_with_interrupt(recorder: Recorder):
    def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None, **kwargs):
        stop = bool(stop_check_fn()) if callable(stop_check_fn) else None
        recorder.call("TransateKRtoEN.send_with_interrupt", {
            "temperature": temperature, "max_tokens": max_tokens, "chunk_timeout": chunk_timeout,
            "kwargs": dict(kwargs), "stop": stop, "client": type(client).__name__})
        result = client.send(messages, temperature, max_tokens)
        if isinstance(result, tuple):
            return (*result, None)
        return result, "stop", None
    return send_with_interrupt


def make_translate_game_images(recorder: Recorder, scenario):
    def translate_game_images(**kwargs):
        shown = {k: (type(v).__name__ if k in ("client",) else ("<callable>" if callable(v) else v))
                 for k, v in kwargs.items()}
        shown["stop_check"] = bool(kwargs["stop_check"]()) if callable(kwargs.get("stop_check")) else None
        recorder.call("rpgmaker_handler.translate_game_images", shown)
        return scenario.images
    return translate_game_images


class _Patches:
    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        missing = object()
        old = getattr(obj, name, missing)
        self._undo.append((obj, name, old, missing))
        setattr(obj, name, value)

    def undo(self):
        while self._undo:
            obj, name, old, missing = self._undo.pop()
            if old is missing:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, old)


@contextlib.contextmanager
def recording_backends(recorder: Recorder, scenario):
    import TransateKRtoEN
    import rpgmaker_handler
    import unified_api_client

    patches = _Patches()
    real_strftime, real_localtime = time.strftime, time.localtime
    real_datetime = _dt.datetime

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromtimestamp(FIXED_TIME, tz)

    try:
        patches.set(unified_api_client, "UnifiedClient", make_fake_client(recorder, scenario))
        patches.set(TransateKRtoEN, "send_with_interrupt", make_send_with_interrupt(recorder))
        patches.set(rpgmaker_handler, "translate_game_images", make_translate_game_images(recorder, scenario))
        # character-estimate token counts (no tiktoken download; same on both sides)
        patches.set(rpgmaker_handler._get_tiktoken_encoder, "_enc", None)
        patches.set(time, "time", lambda: FIXED_TIME)
        patches.set(time, "strftime", lambda fmt, t=None: real_strftime(fmt, real_localtime(FIXED_TIME) if t is None else t))
        patches.set(_dt, "datetime", FixedDatetime)
        yield
    finally:
        patches.undo()


# ---------------------------------------------------------------------------
# owners
# ---------------------------------------------------------------------------


def _owner_base():
    from headless_owner import CheckShim, PlainTextShim, TextShim  # noqa: F401 - shims used below
    from run_env import RunEnvMixin

    class RunnerOwner(RunEnvMixin):
        """The surface the runners read (TranslatorGUI's attributes / widgets / GUI methods)."""

        def __init__(self, recorder, config, attrs):
            self._recorder = recorder
            self.config = copy.deepcopy(config)
            self.stop_requested = False
            self._modules_loaded = True
            self.max_output_tokens = 4096
            self.model_var = self.config.get("model", "gpt-4o-mini")
            self.api_key_entry = TextShim(self.config.get("api_key", "sk-runner-0000"))
            self.trans_temp = TextShim(str(self.config.get("translation_temperature", 0.3)))
            self.trans_history = TextShim(str(self.config.get("translation_history_limit", 2)))
            self.delay_entry = TextShim(str(self.config.get("delay", 2)))
            self.api_queue_entry = TextShim(str(self.config.get("api_queue", 4)))
            self.thread_delay_entry = TextShim(str(self.config.get("thread_delay", 0.1)))
            self.prompt_text = PlainTextShim(self.config.get("prompt_text", ""))
            for key, value in attrs.items():
                setattr(self, key, value)

        def append_log(self, message):
            self._recorder.log(message)

        def _lazy_load_modules(self, splash_callback=None):
            self._recorder.call("owner._lazy_load_modules")
            self._modules_loaded = True
            return True

    return RunnerOwner


def owner_class(side: str, src_file: str):
    base = _owner_base()
    if side == "legacy":
        return type("LegacyRunnerOwner", (base,), legacy_functions(src_file))
    from image_job import ImageJobMixin
    from rpgmaker_job import RpgMakerJobMixin

    return type("NewRunnerOwner", (base, ImageJobMixin, RpgMakerJobMixin), {})


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


@dataclass
class RunnerScenario:
    """One runner call. Strings starting with ``@/`` are paths in the side's sandbox."""

    name: str
    method: str
    args: tuple = ()
    files: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    send: tuple = ()
    send_image: tuple = ()
    images: int = 0
    #: callable(owner) run right before the method (e.g. a Stop mid-run is not needed: flags)
    before: object = None


def resolve(value, root: str):
    if isinstance(value, str) and value.startswith("@/"):
        return os.path.join(root, *value[2:].split("/"))
    if isinstance(value, dict):
        return {k: resolve(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(resolve(v, root) for v in value)
    return value


def write_files(root: str, files: dict):
    for rel, content in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if callable(content):
            content = content(root)
        if isinstance(content, str):
            content = content.encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(content)


def tree_snapshot(root: str) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace("\\", "/")
            data = Path(path).read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                out[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                continue
            if rel.lower().endswith(".json"):
                try:
                    out[rel] = {"json": normalize(json.loads(text), root)}
                    continue
                except ValueError:
                    pass
            out[rel] = {"text": normalize(text, root)}
        for d in dirnames:
            rel = os.path.relpath(os.path.join(dirpath, d), root).replace("\\", "/")
            out.setdefault(rel + "/", {"dir": True})
    return out


_KEEP_ENV = ("SYSTEMROOT", "PATH", "PATHEXT", "COMSPEC", "WINDIR")


def _tiktoken_cache_dir() -> str:
    """tiktoken's cache outside the sandboxes (it defaults to <TEMP>/data-gym-cache)."""
    import tempfile

    for key in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        if os.environ.get(key):
            return os.environ[key]
    return os.path.join(tempfile.gettempdir(), "data-gym-cache")


@contextlib.contextmanager
def _sandbox_process(root: str, env: dict):
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    keep = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
    keep["TIKTOKEN_CACHE_DIR"] = _tiktoken_cache_dir()
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update({"TEMP": root, "TMP": root, "HOME": root, "USERPROFILE": root})
    os.environ.update({k: str(v) for k, v in env.items()})
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)


#: Backend modules imported before any sandbox exists: their import-time side effects (PATH,
#: PySide6 option env, tiktoken cache under TEMP) must not land in one side's record only.
PREIMPORT = ("unified_api_client", "TransateKRtoEN", "rpgmaker_handler", "history_manager", "headless_owner",
             "run_env", "image_job", "rpgmaker_job")


def preimport() -> None:
    import importlib

    for name in PREIMPORT:
        importlib.import_module(name)


def run_side(side: str, scenario: RunnerScenario, root) -> dict:
    """Run *scenario* with the *side* ('legacy' / 'new') runner in sandbox *root*."""
    preimport()
    import image_job

    root = str(Path(root).resolve())
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    write_files(root, scenario.files)
    recorder = Recorder(root)
    # legacy: translator_gui's __file__; new: image_job's (same folder: Generated_Media lands in <root>/src)
    cls = owner_class(side, os.path.join(root, "src", "translator_gui.py"))
    saved_file = image_job.__file__
    image_job.__file__ = os.path.join(root, "src", "image_job.py")
    try:
        with _sandbox_process(root, resolve(scenario.env, root)):
            owner = cls(recorder, resolve(scenario.config, root), resolve(scenario.attrs, root))
            if callable(scenario.before):
                scenario.before(owner)
            recorder.start_env = dict(os.environ)
            result, error = None, None
            with recording_backends(recorder, scenario):
                try:
                    result = getattr(owner, scenario.method)(*resolve(scenario.args, root))
                except BaseException as exc:  # noqa: BLE001 - an escape is an outcome
                    error = f"{type(exc).__name__}: {exc}"
            env = recorder.env_delta()
    finally:
        image_job.__file__ = saved_file
    attrs = {"generated_images": getattr(owner, "generated_images", None),
             "has_image_progress_manager": hasattr(owner, "image_progress_manager"),
             "stop_requested": owner.stop_requested,
             "config": owner.config}
    return normalize({"result": result, "error": error, "events": recorder.events, "env": env,
                      "attrs": attrs, "fs": tree_snapshot(root)}, root)


def compare(legacy: dict, new: dict, limit: int = 30) -> list:
    problems = []

    def walk(a, b, path):
        if len(problems) >= limit:
            return
        if type(a) is not type(b):
            problems.append(f"{path}: {a!r:.300} != {b!r:.300}")
        elif isinstance(a, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a or k not in b:
                    problems.append(f"{path}/{k}: only in {'new' if k in b else 'legacy'}: "
                                    f"{(b if k in b else a)[k]!r:.300}")
                else:
                    walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            if len(a) != len(b):
                problems.append(f"{path}: {len(a)} items != {len(b)} items")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a != b:
            problems.append(f"{path}: {a!r:.300} != {b!r:.300}")

    walk(legacy, new, "")
    return problems


def run_pair(scenario: RunnerScenario, tmp_path) -> tuple:
    base = Path(tmp_path) / scenario.name
    # equal-length sandbox names: recorded lengths of absolute paths (payload content_length) match
    legacy = run_side("legacy", scenario, base / "old")
    new = run_side("new", scenario, base / "new")
    return legacy, new


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

#: a 1x1 PNG
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000100054f6b2a0000000049454e44ae426082")
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32


def generated(rel: str, data: bytes = PNG_BYTES):
    """A response callable: the client wrote a media file and returns its sentinel."""
    def response(root, _payload):
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return f"[GENERATED_IMAGE:{path}]"
    return response


def rpg_echo(prefix: str = "EN"):
    """A response callable for RPG Maker chunks: ``[N] <prefix>(text)`` for every numbered line."""
    import re

    def response(_root, payload):
        messages = payload.get("messages") or []
        user = next((m.get("content", "") for m in reversed(messages) if isinstance(m, dict)
                     and m.get("role") == "user"), "")
        parts = re.split(r"^\[(\d+)\]\s*", str(user), flags=re.M)
        out = []
        for i in range(1, len(parts) - 1, 2):
            text = parts[i + 1].strip()
            out.append(f"[{parts[i]}] {prefix}({text})")
        return "\n".join(out)
    return response


__all__ = [
    "LEGACY_SHA", "MP4_BYTES", "PNG_BYTES", "RUNNERS", "RunnerScenario", "Unavailable", "compare", "generated",
    "git_show", "legacy_functions", "legacy_method_text", "legacy_translator_gui", "module_text", "node_text",
    "normalize", "rpg_echo", "run_pair", "run_side", "tree_snapshot",
]
