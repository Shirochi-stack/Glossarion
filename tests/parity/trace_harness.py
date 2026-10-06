#!/usr/bin/env python3
"""Tier T: pipeline-order and stop-race traces (shared-core design §6, U3).

The desktop run pipeline (``run_translation_thread`` and its ``simple_thread_target``
worker, ``run_translation_direct``, ``run_glossary_extraction_direct``, ``stop_translation``)
is driven end to end on a booted owner while every backend entry point is a recording
stub. A run produces one ordered *trace*:

``append_log``           log lines (and every other GUI recorder call / ``signal:`` / ``qt:``)
``backend:<entry>``      a backend entry point was called: call index, ``sys.argv``, cwd,
                         the environment it sees (delta vs the run's starting env, merged
                         with ``large_env``), stop flags, entry-specific arguments
``check:<entry>``        the stub asked ``stop_callback()`` before a work unit: answer + flags
``stopapi:<fn>``         stop/cancel API calls (``unified_api_client`` stop flag, hard cancel,
                         watchdog reset/clear, API stagger reset, ``TransateKRtoEN`` stop
                         flag / queued-send cancel, glossary/EPUB stop flags) + flags
``hook:<fn>``            other backend hooks (Direct Text approval callback install, ...)
``latch:<attr>``         writes to ``stop_requested`` / ``graceful_stop_active`` /
                         ``_glossary_stop_was_requested`` with the env flags at that moment
                         (proves the stop latch comes after the env mode flags)
``action:<kind>``        harness actions (Stop clicks, clock advances)
``ask``                  the Direct Text glossary approval question (answered from the scenario)
``thread:<event>``       worker threads started/run (run deterministically, see below)

plus the run's final env delta, owner attribute/config deltas, sandbox file changes,
stdout, argv and cwd.

Sides (``mode``):

``legacy``   the frozen oracle: ``TranslatorGUI`` methods + shared mixins as they were at
             the parent commit (``tests/parity/legacy_trace``; frozen with the trace entry
             points as closure seeds, see :func:`freeze_trace_oracle`).
``desktop``  the working-tree desktop: every name resolved through ``TranslatorGUI``'s MRO
             (shared mixins first, desktop hook overrides) onto the same FakeState owner.
             Compared with ``legacy`` on the *full* record.
``mixins``   the mobile path: ``HeadlessOwner``'s bases (+ the U3 pipeline mixins) with
             GUI-free hook defaults, driven as ``_prepare_translation_run(files)`` +
             ``_translation_worker(request)`` inside ``job_runner.scoped_process_state``;
             Stop = ``stop_control.request_stop`` with a ``StopClickTracker`` double-tap.
             Compared with ``legacy`` on the *mobile projection* (backend calls with their
             env/argv, stop checks, stop API calls, the stop latch, approval questions).

Every side starts from the same state: the owner booted once per scenario by the frozen
legacy boot (``fakes.make_legacy_owner_factory``, like the goldens), copied onto the side's
owner class, with the boot's process env and saved config.json restored in the sandbox.

Determinism: threads started by Glossarion code (``threading.Thread.start`` with a target
defined under ``src/`` or in the frozen oracle) are deferred and run synchronously when the
harness drains them (after the entry returns and after every Stop click), and report
``is_alive()`` while pending/running; ``ThreadPoolExecutor.submit`` of such callables runs
inline. ``time.time`` is a harness clock (fixed, advanced only by actions); pid, cpu count,
uuid, temp names and the env are those of ``normalize.CaptureContext``. ``psutil.Process``
is a recorder (helper-process kill never touches real processes).

CLI (repository root)::

    python tests/parity/trace_harness.py --freeze [--sha REV]     # oracle at REV (default HEAD)
    python tests/parity/trace_harness.py --list
    python tests/parity/trace_harness.py graceful_stop [--mode legacy] [--dump]
    python tests/parity/trace_harness.py graceful_stop --compare desktop
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import functools
import importlib
import importlib.util
import inspect
import os
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
TESTS_DIR = PARITY_DIR.parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import fakes, freeze_legacy, normalize  # noqa: E402
from parity import fuzz_moved as fm  # noqa: E402
from parity import trace_scenarios as ts  # noqa: E402

Unavailable = fm.Unavailable

#: the trace oracle lives apart from tests/parity/legacy (different closure seeds)
TRACE_LEGACY_DIR = PARITY_DIR / "legacy_trace"

#: closure seeds: the goldens' entries plus the run pipeline and the stop protocol
TRACE_ENTRY_METHODS = tuple(freeze_legacy.TG_ENTRY_METHODS) + (
    "run_translation_thread",
    "run_translation_direct",
    "stop_translation",
    "run_glossary_extraction_direct",
    "run_glossary_extraction_thread",
    "stop_glossary_extraction",
    "_run_parallel_metadata_files",
    "_convert_zip_input_to_epub_if_needed",
    "_resolve_zip_inputs_for_translation",
    "_reset_stop_flags_if_idle",
)

#: GUI-only methods recorded (never frozen) on top of freeze_legacy.TG_RECORDED_METHODS:
#: _attach_gui_logging_handlers installs handlers on process-wide loggers.
TRACE_RECORDED_METHODS = frozenset({"_attach_gui_logging_handlers"})

#: shared U3 pipeline mixins (contract names of the U3 plan)
U3_MIXINS = (
    ("translation_pipeline", "TranslationPipelineMixin"),
    ("translation_pipeline", "GlossaryPipelineMixin"),
    ("text_jobs", "TextJobsMixin"),
    ("input_preparation", "InputPreparationMixin"),
)
#: unmoved desktop GUI mixins (TranslatorGUI bases after the shared ones): live on every side
GUI_MIXINS = (
    ("QA_Scanner_GUI", "QAScannerMixin"),
    ("Retranslation_GUI", "RetranslationMixin"),
    ("GlossaryManager_GUI", "GlossaryManagerMixin"),
)
U3_MODULES = ("stop_control", "job_runner", "text_jobs", "input_preparation", "translation_pipeline")
#: what a desktop/mobile comparison needs before it can run: the desktop comparison runs as
#: soon as any U3 module exists (from then on the desktop is being rewired onto them)
DESKTOP_REQUIRES_ANY = U3_MODULES
MOBILE_REQUIRES = ("translation_pipeline", "stop_control", "job_runner", "text_jobs", "input_preparation")

STOP_ENV_KEYS = ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED", "WAIT_FOR_CHUNKS",
                 "GRACEFUL_STOP_API_ACTIVE")
MODULE_FLAGS = ("TransateKRtoEN", "extract_glossary_from_epub", "epub_converter", "unified_api_client")
LATCH_ATTRS = ("stop_requested", "graceful_stop_active", "_glossary_stop_was_requested")
#: loggers stop_translation silences during a graceful stop (restored after every run)
SILENCED_LOGGERS = ("httpx", "openai", "google", "google.api_core", "google.generativeai", "urllib3")
#: translator_gui module globals bound by _lazy_load_modules
BACKEND_GLOBALS = {
    "translation_main": (ts.TRANSLATION_MAIN,),
    "translation_stop_flag": ("TransateKRtoEN.set_stop_flag",),
    "translation_stop_check": ("TransateKRtoEN.is_stop_requested",),
    "glossary_main": (ts.GLOSSARY_MAIN,),
    "glossary_stop_flag": ("extract_glossary_from_epub.set_stop_flag",),
    "glossary_stop_check": ("extract_glossary_from_epub.is_stop_requested",),
    "fallback_compile_epub": ("epub_converter.fallback_compile_epub",),
}
#: modules imported before any sandbox exists (import-time side effects stay identical)
PREIMPORT = ("TransateKRtoEN", "extract_glossary_from_epub", "epub_converter", "pdf_workspace_compiler",
             "metadata_translation_worker", "unified_api_client", "antigravity_proxy", "scan_html_folder",
             "glossary_translation_gate", "glossary_paths", "output_workspace", "output_naming",
             "library_core", "headless_owner", "psutil",
             # U7: the image / RPG Maker runners and what they import lazily
             "rpgmaker_handler", "history_manager", "image_job", "rpgmaker_job")
#: live modules whose ``__file__`` the sandbox replaces (they derive output paths from it;
#: U7: image_job's generative-only run saves into <its folder>/Generated_Media)
FILE_MODULES = ("translator_gui", "app_paths", "owner_state", "run_env", "settings_persistence",
                "headless_owner", "library_core", "image_job") + U3_MODULES
#: channels kept by the mobile projection
MOBILE_CHANNELS = ("backend", "check", "stopapi", "hook", "action", "ask")
#: sandbox files left out of the mobile projection's file delta (reason in the comment)
MOBILE_IGNORED_FILES = (
    "src/config.json",  # desktop stop_translation persists config.json; mobile config writes are sparse
)
#: sandbox trees left out of the mobile projection's file delta. Empty since U5: HeadlessOwner
#: records the run's raw inputs in the Library registry (``library_raw_inputs.txt`` + the Library
#: layout under the sandbox home) through ``library_core.record_library_raw_inputs``, exactly as
#: the desktop's ``_record_library_raw_inputs`` hook does through epub_library, so the mobile file
#: delta must equal the desktop one there too (U3 fix pass ignored the tree; U5 integration).
MOBILE_IGNORED_TREES: tuple = ()
#: parents the Library layout creates (ignored as added/removed directories only; none since U5)
MOBILE_IGNORED_PARENT_DIRS: tuple = ()


def _mobile_fs_keep(kind: str, path: str) -> bool:
    if path in MOBILE_IGNORED_FILES:
        return False
    if any(path == tree or path.startswith(tree + "/") for tree in MOBILE_IGNORED_TREES):
        return False
    return not (kind.startswith("dirs_") and path in MOBILE_IGNORED_PARENT_DIRS)
#: recorded on the mobile side even when HeadlessOwner defines them: the API-watchdog
#: progress reset is GUI progress-bar housekeeping (desktop: a recorder), whose inner
#: ``_api_watchdog_reset`` call would otherwise appear on one side only
MOBILE_FORCED_RECORDERS = ("_reset_api_watchdog_progress",)
#: env values that are opaque per-run tokens (the mobile JobService resets once more)
MOBILE_MASKED_ENV = ("GLOSSARION_RUN_ID",)
#: Known, reviewed legacy/new differences: {mode: [(event-name prefix, reason)]} (target: empty)
KNOWN_TRACE_DIVERGENCES: dict = {
    "desktop": [],
    "mixins": [
        ("hook:antigravity_proxy.", "antigravity/ is excluded on mobile; desktop resets its proxy-update retry "
                                    "in run_translation_thread's thread launch, not in the shared worker"),
    ],
}
#: callables(owner) run on the new sides after the state copy, for attributes a U3 desktop
#: __init__ adds that the legacy boot cannot produce (keep empty unless a move needs it)
NEW_SIDE_BOOT_HOOKS: list = []

_ABSENT = "<absent>"


# ===========================================================================
# Oracle: freeze / load
# ===========================================================================


def freeze_trace_oracle(rev: str = "HEAD", out_dir: Path = TRACE_LEGACY_DIR) -> Path:
    """Freeze TranslatorGUI @ *rev* with the trace closure into *out_dir* (``LATEST.txt`` updated)."""
    saved = freeze_legacy.TG_RECORDED_METHODS
    freeze_legacy.TG_RECORDED_METHODS = frozenset(saved | TRACE_RECORDED_METHODS)
    try:
        return freeze_legacy.freeze(rev, out_dir=out_dir, entry_methods=TRACE_ENTRY_METHODS)
    finally:
        freeze_legacy.TG_RECORDED_METHODS = saved


def trace_oracle_sha(out_dir: Path = TRACE_LEGACY_DIR) -> str:
    marker = out_dir / "LATEST.txt"
    if not marker.exists():
        raise FileNotFoundError(
            f"no trace oracle in {out_dir}; run python tests/parity/trace_harness.py --freeze")
    return marker.read_text(encoding="utf-8").strip()


def load_trace_oracle(sha: str | None = None, out_dir: Path = TRACE_LEGACY_DIR):
    """``freeze_legacy.load_legacy`` reading the trace oracle directory."""
    sha = sha or trace_oracle_sha(out_dir)
    saved = freeze_legacy.LEGACY_DIR
    freeze_legacy.LEGACY_DIR = out_dir
    try:
        return freeze_legacy.load_legacy(sha)
    finally:
        freeze_legacy.LEGACY_DIR = saved


def module_exists(name: str) -> bool:
    return (SRC_DIR / f"{name}.py").is_file()


def missing_modules(names) -> list:
    return [n for n in names if not module_exists(n)]


def preimport() -> None:
    normalize.preimport_backend_modules()
    for name in PREIMPORT + tuple(m for m in U3_MODULES if module_exists(m)):
        try:
            importlib.import_module(name)
        except Exception:  # pragma: no cover - environment dependent
            pass


# ===========================================================================
# Deterministic clock and threads
# ===========================================================================


class TraceClock:
    """``time.time`` of a trace run: fixed, moved only by ``('advance', s)`` actions."""

    def __init__(self):
        self.offset = 0.0

    def reset(self):
        self.offset = 0.0

    def time(self):
        return normalize.FIXED_TIME + self.offset

    def advance(self, seconds):
        self.offset += float(seconds)


def _code_file(fn):
    for _ in range(6):
        if isinstance(fn, functools.partial):
            fn = fn.func
        elif inspect.ismethod(fn):
            fn = fn.__func__
        else:
            break
    code = getattr(fn, "__code__", None)
    return code.co_filename if code is not None else None


class ThreadGate:
    """Defers threads/executor tasks started by Glossarion code and runs them on ``drain()``."""

    OWNED_ROOTS = tuple(os.path.normcase(str(p)) for p in (SRC_DIR, TRACE_LEGACY_DIR, PARITY_DIR))

    def __init__(self, tracer):
        self.tracer = tracer
        self.pending: deque = deque()
        self.real_start = threading.Thread.start
        self.real_is_alive = threading.Thread.is_alive
        self.real_join = threading.Thread.join
        self.real_submit = concurrent.futures.ThreadPoolExecutor.submit

    def owns(self, fn) -> bool:
        path = _code_file(fn)
        if not path:
            return False
        path = os.path.normcase(os.path.abspath(path))
        return path.startswith(self.OWNED_ROOTS)

    @staticmethod
    def _target(thread):
        target = getattr(thread, "_target", None)
        if target is None and isinstance(thread, threading.Timer):
            target = thread.function
        return target

    def install(self, ctx) -> None:
        gate = self

        def start(thread):
            if getattr(thread, "_parity_state", None) is None and gate.owns(gate._target(thread)):
                thread._parity_state = "pending"
                gate.pending.append(thread)
                gate.tracer.record("thread:start", [thread.name])
                return None
            return gate.real_start(thread)

        def is_alive(thread):
            state = getattr(thread, "_parity_state", None)
            if state is None:
                return gate.real_is_alive(thread)
            return state in ("pending", "running")

        def join(thread, timeout=None):
            state = getattr(thread, "_parity_state", None)
            if state is None:
                return gate.real_join(thread, timeout)
            if state == "pending":
                gate.run_one(thread)
            return None

        def submit(executor, fn, /, *args, **kwargs):
            if not gate.owns(fn):
                return gate.real_submit(executor, fn, *args, **kwargs)
            future = concurrent.futures.Future()
            gate.tracer.record("thread:submit", [getattr(fn, "__name__", type(fn).__name__)])
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - the future carries it
                future.set_exception(exc)
            return future

        ctx.patch(threading.Thread, "start", start)
        ctx.patch(threading.Thread, "is_alive", is_alive)
        ctx.patch(threading.Thread, "join", join)
        ctx.patch(concurrent.futures.ThreadPoolExecutor, "submit", submit)

    def reset(self):
        self.pending.clear()

    def run_one(self, thread):
        try:
            self.pending.remove(thread)
        except ValueError:
            pass
        if getattr(thread, "_parity_state", None) != "pending":
            return
        thread._parity_state = "running"
        self.tracer.record("thread:run", [thread.name])
        try:
            if isinstance(thread, threading.Timer) and getattr(thread, "_target", None) is None:
                thread.function(*thread.args, **thread.kwargs)
            else:
                thread._target(*thread._args, **thread._kwargs)
        except BaseException as exc:  # noqa: BLE001 - a thread's exception is part of the trace
            self.tracer.record("thread:exception", [thread.name, type(exc).__name__, fm.mask(str(exc))[:300]])
        finally:
            thread._parity_state = "done"

    def drain(self):
        while self.pending:
            self.run_one(self.pending[0])


# ===========================================================================
# Recording backend stubs
# ===========================================================================


class _LatchAttr:
    """Data descriptor recording writes to a stop-latch attribute (value kept in __dict__)."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value
        tracer = obj.__dict__.get("_parity_tracer")
        if tracer is not None:
            tracer.record(f"latch:{self.name}", [value], {"flags": tracer.flags()})

    def __delete__(self, obj):
        try:
            del obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None


class AnsweringSignal(fakes.FakeSignal):
    """``direct_text_glossary_approval_signal``: records the question and answers it at once."""

    def __init__(self, recorder, name, tracer):
        super().__init__(recorder, name)
        self._tracer = tracer

    def emit(self, *args):
        path = args[0] if args else ""
        request = args[1] if len(args) > 1 else None
        answer = self._tracer.next_answer()
        self._tracer.record("ask", [path], {"kind": self._name, "answer": answer})
        if isinstance(request, dict):
            request["accepted"] = answer
            event = request.get("event")
            if event is not None:
                event.set()


class TraceHost:
    """``job_runner.JobHost`` for the mobile side: logs/events/questions go into the trace."""

    def __init__(self, tracer):
        self._tracer = tracer

    def log(self, text, **_kw):
        self._tracer.ctx.recorder.record("append_log", [text], {})

    def is_stop_requested(self):
        return bool(self._tracer.owner_value("stop_requested", False))

    def is_graceful_stop(self):
        return bool(self._tracer.owner_value("graceful_stop_active", False))

    def emit(self, kind, **data):
        self._tracer.record(f"emit:{kind}", [], dict(data))

    def ask(self, kind, **data):
        path = next((v for v in data.values() if isinstance(v, str)), "")
        answer = self._tracer.next_answer()
        self._tracer.record("ask", [path], {"kind": kind, "answer": answer})
        return answer

    def describe(self):
        return {"__trace_host__": True}


class Tracer:
    """Per-context recorder of backend stubs, stop flags, clicks and answers."""

    def __init__(self, ctx, session):
        self.ctx = ctx
        self.session = session
        self.clock = TraceClock()
        self.gate = ThreadGate(self)
        self.owner = None
        self.mode = None
        self.scenario = None
        self.pre_env: dict = {}
        self.flags_state = {m: False for m in MODULE_FLAGS}
        self.counts: dict = {}
        self.answers: list = []
        self.clicks = 0
        self.mobile_tracker = None
        self.approval_callback = None
        self.uac = None

    # -- recording ---------------------------------------------------------------
    def record(self, name, args=(), kwargs=None):
        self.ctx.recorder.record(name, list(args), kwargs or {})

    def owner_value(self, name, default=None):
        owner = self.owner
        return default if owner is None else owner.__dict__.get(name, default)

    def flags(self) -> dict:
        env = os.environ
        out = {k: env.get(k) for k in STOP_ENV_KEYS}
        for name in ("stop_requested", "graceful_stop_active"):
            out[name] = self.owner_value(name, _ABSENT)
        for module in MODULE_FLAGS:
            out[f"module:{module}"] = self.flags_state[module]
        uac = self.uac
        if uac is not None:
            out["UnifiedClient._global_cancelled"] = getattr(uac.UnifiedClient, "_global_cancelled", None)
            out["unified_api_client.global_stop_flag"] = getattr(uac, "global_stop_flag", None)
        stop_file = env.get("GLOSSARY_STOP_FILE") or ""
        out["glossary_stop_file_exists"] = bool(stop_file) and os.path.exists(stop_file)
        return out

    def env_now(self) -> dict:
        return normalize.env_delta(self.pre_env, normalize.effective_env())

    def next_answer(self):
        return self.answers.pop(0) if self.answers else False

    # -- per run ---------------------------------------------------------------------
    def reset(self, scenario, mode):
        self.scenario = scenario
        self.mode = mode
        self.clock.reset()
        self.gate.reset()
        self.flags_state = {m: False for m in MODULE_FLAGS}
        self.counts = {}
        self.answers = list(scenario.get("answers") or ())
        self.clicks = 0
        self.mobile_tracker = None
        self.approval_callback = None
        self.owner = None
        if self.uac is not None:
            recording_cls = self.uac.UnifiedClient
            if "_global_cancelled" in vars(recording_cls):
                delattr(recording_cls, "_global_cancelled")
            if hasattr(self.uac, "global_stop_flag"):
                self.uac.global_stop_flag = False
        import logging

        for name, level in self.session.logger_levels.items():
            logging.getLogger(name).setLevel(level)

    def plan(self, entry: str, index: int) -> ts.CallPlan:
        plans = (self.scenario or {}).get("plan", {}).get(entry) or ()
        return plans[index] if index < len(plans) else ts.CallPlan()

    # -- actions -------------------------------------------------------------------
    def fire(self, action):
        kind = action[0]
        if kind == "advance":
            self.clock.advance(action[1])
            self.record("action:advance", [action[1]])
        elif kind == "click":
            self.clicks += 1
            self.record("action:click", [self.clicks], {"flags": self.flags()})
            DRIVERS[self.mode].click(self)
            self.gate.drain()
        else:
            raise ValueError(f"unknown trace action {action!r}")

    # -- stub bodies ------------------------------------------------------------------
    def _work(self, entry, plan, stop_callback, log_callback):
        for chunk in range(1, plan.chunks + 1):
            stop = bool(stop_callback()) if callable(stop_callback) else False
            self.record(f"check:{entry}", [chunk], {"stop_callback": stop, "flags": self.flags()})
            if stop:
                if callable(log_callback):
                    log_callback(f"[trace] {entry}: stop observed before unit {chunk}")
                return False
            if callable(log_callback):
                log_callback(f"[trace] {entry}: unit {chunk}/{plan.chunks}")
            for action in plan.actions.get(chunk, ()):
                self.fire(action)
        return True

    def _enter(self, entry, extra=None):
        index = self.counts.get(entry, 0)
        self.counts[entry] = index + 1
        data = {"call": index, "argv": list(sys.argv), "cwd": os.getcwd(), "env": self.env_now(),
                "large_env": normalize.large_env_store(), "flags": self.flags()}
        data.update(extra or {})
        self.record(f"backend:{entry}", [], data)
        return index, self.plan(entry, index)

    def translation_main(self, log_callback=None, stop_callback=None):
        _index, plan = self._enter(ts.TRANSLATION_MAIN)
        self._work(ts.TRANSLATION_MAIN, plan, stop_callback, log_callback)
        return plan.result

    def glossary_main(self, log_callback=None, stop_callback=None):
        _index, plan = self._enter(ts.GLOSSARY_MAIN)
        finished = self._work(ts.GLOSSARY_MAIN, plan, stop_callback, log_callback)
        output_path = os.environ.get("OUTPUT_PATH") or ""
        if output_path and (plan.glossary and finished):
            import json

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump([{"type": "character", "raw_name": "김상현", "translated_name": "Kim Sang-hyun",
                            "gender": "male"}], fh, ensure_ascii=False, indent=2)
        if output_path and plan.progress is not None:
            import json

            base = os.path.splitext(os.path.basename(output_path))[0]
            if base.endswith("_glossary"):
                base = base[: -len("_glossary")]
            progress_path = os.path.join(os.path.dirname(output_path), f"{base}_glossary_progress.json")
            with open(progress_path, "w", encoding="utf-8") as fh:
                json.dump(plan.progress, fh, indent=2)
        return plan.result

    def metadata_job(self, source_path, env_vars, *, log_callback=None, stop_check_fn=None,
                     client_factory=None, translator_factory=None):
        _index, plan = self._enter(ts.METADATA_JOB, {
            "source_path": source_path,
            "env_vars": dict(env_vars or {}),
            "factories": [client_factory is not None, translator_factory is not None],
        })
        finished = self._work(ts.METADATA_JOB, plan, stop_check_fn, log_callback)
        return finished if plan.result is None else plan.result

    def compile_epub(self, *args, **kwargs):
        _index, plan = self._enter(ts.EPUB_COMPILE, {"args": list(args), "kwargs": dict(kwargs)})
        return plan.result

    def compile_pdf(self, folder, log_callback=None, stop_callback=None, api_client=None):
        _index, plan = self._enter(ts.PDF_COMPILE, {"folder": folder, "api_client": api_client is not None})
        self._work(ts.PDF_COMPILE, plan, stop_callback, log_callback)
        return plan.result

    # -- U7: the client calls of the image / generative / RPG Maker runners ----------------------
    def _client_result(self, plan, payload, default):
        result = plan.result
        if callable(result):
            result = result(self, payload)
        if isinstance(result, BaseException):
            raise result
        if result is None:
            result = default
        return result if isinstance(result, tuple) else (result, "stop")

    def client_send(self, _client, messages=None, temperature=None, max_tokens=None, **kwargs):
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "kwargs": kwargs}
        index, plan = self._enter(ts.CLIENT_SEND, payload)
        return self._client_result(plan, payload, f"[trace send {index}]")

    def client_send_image(self, _client, messages, image_data, temperature=None, max_tokens=None,
                          response_name=None, **kwargs):
        import hashlib

        data = image_data if isinstance(image_data, (bytes, bytearray)) else str(image_data or "").encode("utf-8")
        payload = {"messages": messages, "image_sha256": hashlib.sha256(data).hexdigest(), "temperature": temperature,
                   "max_tokens": max_tokens, "response_name": response_name, "kwargs": kwargs}
        index, plan = self._enter(ts.IMAGE_SEND, payload)
        return self._client_result(plan, payload, f"<p>trace image response {index}</p>")

    def send_with_interrupt(self, messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None,
                            **kwargs):
        """TransateKRtoEN.send_with_interrupt: records its arguments, then calls ``client.send`` inline."""
        stop = bool(stop_check_fn()) if callable(stop_check_fn) else None
        self.record(f"hook:{ts.SEND_INTERRUPT}", [], {"temperature": temperature, "max_tokens": max_tokens,
                                                      "chunk_timeout": chunk_timeout, "stop": stop,
                                                      "kwargs": dict(kwargs), "client": type(client).__name__})
        result = client.send(messages, temperature, max_tokens)
        return (*result, None) if isinstance(result, tuple) else (result, "stop", None)

    def translate_game_images(self, **kwargs):
        shown = {k: ("<client>" if k == "client" else "<callable>" if callable(v) else v) for k, v in kwargs.items()}
        _index, plan = self._enter(ts.GAME_IMAGES, {"kwargs": shown})
        return 0 if plan.result is None else plan.result

    def stop_api(self, name, *, flag=None, result=None):
        tracer = self

        def fn(*args, **kwargs):
            if flag is not None:
                value = args[0] if args else kwargs.get("value", True)
                tracer.flags_state[flag] = bool(value)
            tracer.record(f"stopapi:{name}", list(args), dict(kwargs, flags=tracer.flags()))
            return result() if callable(result) else result

        fn.__name__ = name.rsplit(".", 1)[-1]
        return fn

    def flag_reader(self, module):
        def fn(*_args, **_kwargs):
            return bool(self.flags_state[module])

        fn.__name__ = "is_stop_requested"
        return fn

    def hook(self, name, *, store=None, result=None):
        tracer = self

        def fn(*args, **kwargs):
            if store:
                setattr(tracer, store, args[0] if args else None)
            tracer.record(f"hook:{name}", list(args), dict(kwargs))
            return result

        fn.__name__ = name.rsplit(".", 1)[-1]
        return fn

    def watchdog_state(self, *_args, **_kwargs):
        return {"in_flight": 0, "queued": 0, "cooldown": 0, "completed": 0, "failed": 0}

    # -- install ------------------------------------------------------------------
    def stub_table(self) -> dict:
        """{module: {attribute: replacement}} (only attributes the live module defines are patched)."""
        return {
            "TransateKRtoEN": {
                "main": self.translation_main,
                "set_stop_flag": self.stop_api("TransateKRtoEN.set_stop_flag", flag="TransateKRtoEN"),
                "is_stop_requested": self.flag_reader("TransateKRtoEN"),
                "cancel_queued_translation_sends": self.stop_api(
                    "TransateKRtoEN.cancel_queued_translation_sends", result=2),
                "set_direct_text_glossary_approval_callback": self.hook(
                    "TransateKRtoEN.set_direct_text_glossary_approval_callback", store="approval_callback"),
                # U7: the image runner's vision call goes through it (a thread + polling in reality)
                "send_with_interrupt": self.send_with_interrupt,
            },
            "rpgmaker_handler": {"translate_game_images": self.translate_game_images},
            "extract_glossary_from_epub": {
                "main": self.glossary_main,
                "set_stop_flag": self.stop_api("extract_glossary_from_epub.set_stop_flag",
                                               flag="extract_glossary_from_epub"),
                "is_stop_requested": self.flag_reader("extract_glossary_from_epub"),
            },
            "epub_converter": {
                "set_stop_flag": self.stop_api("epub_converter.set_stop_flag", flag="epub_converter"),
                "is_stop_requested": self.flag_reader("epub_converter"),
                "compile_epub": self.compile_epub,
                "fallback_compile_epub": self.compile_epub,
            },
            "pdf_workspace_compiler": {"compile_pdf_workspace": self.compile_pdf},
            "metadata_translation_worker": {"run_metadata_translation_job": self.metadata_job},
            "unified_api_client": {
                "set_stop_flag": self.stop_api("unified_api_client.set_stop_flag", flag="unified_api_client"),
                "is_stop_requested": self.flag_reader("unified_api_client"),
                "hard_cancel_all": self.stop_api("unified_api_client.hard_cancel_all"),
                "_api_watchdog_reset": self.stop_api("unified_api_client._api_watchdog_reset"),
                "reset_api_call_stagger": self.stop_api("unified_api_client.reset_api_call_stagger"),
                "_api_watchdog_clear_pending_requests": self.stop_api(
                    "unified_api_client._api_watchdog_clear_pending_requests", result=1),
                "get_api_watchdog_state": self.watchdog_state,
            },
            "antigravity_proxy": {
                "allow_proxy_update_retry_for_new_run": self.hook(
                    "antigravity_proxy.allow_proxy_update_retry_for_new_run"),
            },
        }

    def install(self, namespaces) -> None:
        ctx = self.ctx
        table = self.stub_table()
        resolved = {}
        for module_name, attrs in table.items():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr, stub in attrs.items():
                if attr in vars(module):
                    ctx.patch(module, attr, stub)
                    resolved[f"{module_name}.{attr}"] = stub
        uac = importlib.import_module("unified_api_client")
        self.uac = uac
        if hasattr(uac, "global_stop_flag"):
            ctx.patch(uac, "global_stop_flag", False)
        recording_cls = uac.UnifiedClient  # CaptureContext's RecordingUnifiedClient
        for name in ("hard_cancel_all", "set_global_cancellation", "reset_api_call_stagger"):
            if hasattr(recording_cls, name):
                stub = self.stop_api(f"UnifiedClient.{name}")
                ctx.patch(recording_cls, name, classmethod(lambda cls, *a, _s=stub, **k: _s(*a, **k)))
        # U7: the runners' client calls (the recording client only records its constructor)
        tracer_ = self
        ctx.patch(recording_cls, "send", lambda client, *a, **k: tracer_.client_send(client, *a, **k))
        ctx.patch(recording_cls, "send_image", lambda client, *a, **k: tracer_.client_send_image(client, *a, **k))
        try:
            import rpgmaker_handler

            # character-estimate token budgets (no tiktoken download; identical on every side)
            ctx.patch(rpgmaker_handler._get_tiktoken_encoder, "_enc", None)
        except ImportError:
            pass
        real_strftime, real_localtime = time.strftime, time.localtime
        clock = self.clock
        ctx.patch(time, "strftime", lambda fmt, t=None: real_strftime(
            fmt, real_localtime(clock.time()) if t is None else t))
        # module globals _lazy_load_modules binds (frozen namespace + live translator_gui)
        for ns in namespaces:
            for global_name, (target,) in BACKEND_GLOBALS.items():
                if global_name in ns and target in resolved:
                    ctx.patch_dict(ns, global_name, resolved[target])
            if "scan_html_folder" in ns:  # rebound by _lazy_load_modules: restored on exit
                ctx.patch_dict(ns, "scan_html_folder", ns["scan_html_folder"])
        try:
            import psutil

            tracer = self

            class _FakeProcess:
                def __init__(self, pid=None):
                    self.pid = pid
                    tracer.record("stopapi:psutil.Process", [pid])

                def children(self, recursive=False):
                    tracer.record("stopapi:psutil.Process.children", [], {"recursive": recursive})
                    return []

            ctx.patch(psutil, "Process", _FakeProcess)
        except ImportError:
            pass
        self.gate.install(ctx)
        ctx.patch(time, "time", self.clock.time)


# ===========================================================================
# Drivers (what each side runs)
# ===========================================================================


class DesktopDriver:
    """legacy and desktop sides: the Run button and the desktop worker thread."""

    @staticmethod
    def run(tracer, owner, scenario):
        if scenario["entry"] == ts.GLOSSARY:
            # the Extract Glossary button (U6): run_glossary_extraction_thread resets the stop
            # state (stop env, run id, extractor stop flag, glossary stop file) and hands
            # run_glossary_extraction_direct to the executor, which the gate runs inline
            return owner.run_glossary_extraction_thread()
        return owner.run_translation_thread()

    @staticmethod
    def click(tracer):
        if (tracer.scenario or {}).get("entry") == ts.GLOSSARY:
            # the Extract Glossary button while extraction runs: run_glossary_extraction_thread
            # stops a live glossary worker through stop_glossary_extraction (U6 scenarios)
            tracer.owner.stop_glossary_extraction()
            return
        # the Run/Stop button: run_translation_thread() stops a live worker (stop_translation)
        tracer.owner.run_translation_thread()


class MobileDriver:
    """mixins side: the mobile JobService composition of the shared pipeline + stop_control.

    Run (``services/jobs.py`` ``JobService._execute`` + ``job_kinds/translate.py``): inside
    ``job_runner.scoped_process_state(capture_stdout=host.log, lock=JOB_LOCK)``,
    ``stop_control.reset_for_new_run(kind)``, then ``_prepare_translation_run(files)`` and
    ``_translation_worker(request)``.

    Stop: the *reference* use of ``stop_control.request_stop`` that reproduces desktop
    ``stop_translation``: ``StopClickTracker`` double click = force (a force click first drops
    ``graceful_stop_active`` / ``_last_stop_was_graceful``, as the desktop double-click block
    does before the force flags), the latch sets what the desktop latches
    (``graceful_stop_active``, the click timestamps, ``stop_requested``), the lazily loaded
    ``translation_stop_flag`` is the ``stop_flag_hook`` and the watchdog reset is the owner's
    ``_reset_api_watchdog_progress``.
    """

    @staticmethod
    def run(tracer, owner, scenario):
        import job_runner
        import stop_control

        host = owner.__dict__.get("host")
        capture = getattr(host, "log", None)
        with job_runner.scoped_process_state(capture_stdout=capture, lock=job_runner.JOB_LOCK):
            kind = "glossary" if scenario["entry"] == ts.GLOSSARY else "translation"
            tracer.record("action:job_start", [kind])
            stop_control.reset_for_new_run(kind=kind)
            tracer.record("action:pipeline_start", [kind])
            if scenario["entry"] == ts.GLOSSARY:
                return owner.run_glossary_extraction_direct()
            # (the generative-only sentinel is not a path: a mobile job passes it as is, U7)
            files = [p if p == "__generative_mode__" else os.path.abspath(os.fspath(p))
                     for p in (getattr(owner, "selected_files", None) or []) if p]
            request = owner._prepare_translation_run(files)
            if request is None or request is False:
                return request
            return owner._translation_worker(request)

    @staticmethod
    def click(tracer):
        import stop_control

        owner = tracer.owner
        if (tracer.scenario or {}).get("entry") == ts.GLOSSARY:
            # U6: a glossary job stops through stop_control.request_glossary_stop (the desktop
            # stop_glossary_extraction protocol). The desktop forces an immediate stop only while
            # its button reads "Finishing..." (no such label here), so there is no force click; the
            # latch is stop_requested and the cleanup's run-id guard reads _glossary_run_id.
            if not hasattr(stop_control, "request_glossary_stop"):
                raise Unavailable("stop_control.request_glossary_stop is missing (U6)")
            entry = getattr(owner, "_backend_entry", None)
            if callable(entry):
                glossary_flag = entry("glossary_stop_flag")
            else:  # pre-U3 owner: the desktop's lazily loaded glossary_stop_flag
                glossary_flag = getattr(importlib.import_module("extract_glossary_from_epub"), "set_stop_flag", None)

            def glossary_latch():
                owner.stop_requested = True

            _call_supported(stop_control.request_glossary_stop,
                            graceful=bool(getattr(owner, "graceful_stop_var", False)),
                            set_stop_requested=glossary_latch, log=owner.append_log,
                            glossary_stop_flag=glossary_flag,
                            get_run_id=lambda: getattr(owner, "_glossary_run_id", None))
            return
        if tracer.mobile_tracker is None:
            tracer.mobile_tracker = stop_control.StopClickTracker()
        force = bool(_call_supported(tracer.mobile_tracker.register, now=tracer.clock.time()))
        graceful = bool(getattr(owner, "graceful_stop_var", False)) and not force
        wait = bool(getattr(owner, "wait_for_chunks_var", True)) and graceful
        if force:
            # desktop double click: the graceful state is dropped before the force flags
            owner.graceful_stop_active = False
            owner._last_stop_was_graceful = False

        def latch():
            # desktop stop_translation order: graceful flag, click timestamps, then the latch
            owner.graceful_stop_active = graceful
            owner._last_stop_translation_ts = time.time()
            owner._last_stop_was_graceful = graceful
            owner.stop_requested = True

        reset_watchdog = getattr(owner, "_reset_api_watchdog_progress", None)
        clear = (lambda: reset_watchdog(clear_stale_external_files=True)) if callable(reset_watchdog) else None
        entry = getattr(owner, "_backend_entry", None)
        if callable(entry):
            stop_flag = entry("translation_stop_flag")
        else:  # pre-U3 owner (stop_control mode): the desktop's lazily loaded translation_stop_flag
            stop_flag = getattr(importlib.import_module("TransateKRtoEN"), "set_stop_flag", None)
        hook = (lambda: stop_flag(True)) if callable(stop_flag) else None
        _call_supported(stop_control.request_stop, graceful=graceful, wait_for_chunks=wait, force=force,
                        set_stop_requested=latch, log=owner.append_log, clear_watchdog=clear,
                        stop_flag_hook=hook,
                        cleanup_thread_created=lambda t: setattr(owner, "_translation_stop_cleanup_thread", t))


def _call_supported(fn, **kwargs):
    """Call *fn* with the keyword arguments its signature accepts (contract drift shows in the trace)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


class StopControlDriver:
    """stop_control side: the frozen desktop run, but every Stop click goes through
    ``stop_control.request_stop`` the way the mobile side does (checks the shared stop protocol
    against desktop ``stop_translation`` before the pipeline moves)."""

    run = staticmethod(DesktopDriver.run)
    click = staticmethod(MobileDriver.click)


DRIVERS = {"legacy": DesktopDriver, "desktop": DesktopDriver, "mixins": MobileDriver,
           "stop_control": StopControlDriver}
#: modes that run the frozen legacy code (frozen namespace + module patches)
LEGACY_CODE_MODES = ("legacy", "stop_control")


# ===========================================================================
# Session: oracle, booted bases, owner classes
# ===========================================================================


@dataclass
class BootedBase:
    name: str
    attrs: dict
    config: dict
    os_env: dict
    large_env: dict
    config_text: str | None


def _qt_names(bundle) -> set:
    import ast

    names = {"QApplication", "QDialogButtonBox", "QMessageBox", "QThread", "QTimer", "Qt"}
    tree = ast.parse(bundle.path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and fm._QT_NAME_RE.match(node.id):
            names.add(node.id)
    return names


def _is_qt_object(value) -> bool:
    module = getattr(value, "__module__", None) or type(value).__module__ or ""
    return str(module).startswith(("PySide6", "shiboken6", "Shiboken"))


def _recorder(name):
    return fm._recorder_function(name)


def _latch_namespace() -> dict:
    return {name: _LatchAttr(name) for name in LATCH_ATTRS}


class TraceSession:
    """Lazily built shared pieces (one per pytest session / CLI run)."""

    def __init__(self, sha: str | None = None):
        self._sha = sha
        self._bundle = None
        self._bases: dict = {}
        self._classes: dict = {}
        self._signal_names = tuple(fakes.DESKTOP_SIGNALS)
        import logging

        self.logger_levels = {name: logging.getLogger(name).level for name in SILENCED_LOGGERS}

    # -- oracle ------------------------------------------------------------------
    @property
    def bundle(self):
        if self._bundle is None:
            if importlib.util.find_spec("PySide6") is None:
                raise Unavailable("PySide6 is not installed (the frozen desktop oracle imports it)")
            try:
                self._bundle = load_trace_oracle(self._sha)
            except FileNotFoundError as exc:
                raise Unavailable(f"tier T oracle missing ({exc}); run python tests/parity/trace_harness.py "
                                  "--freeze [--sha REV] (REV = the parent commit of the U3 step)") from exc
            preimport()
        return self._bundle

    @property
    def recorded(self) -> frozenset:
        return (frozenset(self.bundle.manifest.get("recorded_methods", ()))
                | freeze_legacy.TG_RECORDED_METHODS | TRACE_RECORDED_METHODS)

    @property
    def bound(self) -> tuple:
        return tuple(self.bundle.manifest.get("other_settings_bound_methods", ()))

    # -- bases -------------------------------------------------------------------
    def base(self, scenario: dict) -> BootedBase:
        name = scenario["name"]
        if name not in self._bases:
            self._bases[name] = self._boot(scenario)
        return self._bases[name]

    def _boot(self, scenario) -> BootedBase:
        bundle = self.bundle
        factory = fakes.make_legacy_owner_factory(bundle)
        boot = {"name": f"trace_boot_{scenario['name']}", "config": scenario.get("config"), "files": {}}
        with normalize.CaptureContext(boot, "boot") as ctx:
            owner = factory(boot, ctx)
            ph = normalize.PathPlaceholders([(str(ctx.sandbox.root), normalize.SANDBOX_TOKEN)])
            attrs = {}
            for key, value in vars(owner).items():
                if key == "config" or key.startswith("_parity_"):
                    continue
                if isinstance(value, (fakes.FakeSignal, types.MethodType)) or not fm.is_plain(value):
                    continue
                attrs[key] = fm.map_strings(value, ph.apply)
            config = fm.map_strings(copy.deepcopy(owner.config), ph.apply)
            delta = normalize.env_delta(ctx.baseline_env, normalize.effective_env())
            large = normalize.large_env_store()
            large_upper = {k.upper() if sys.platform == "win32" else k for k in large}
            os_env = {k: ph.apply(v) for k, v in delta["set"].items() if k not in large_upper}
            large_env = {k: ph.apply(v) for k, v in large.items()}
            config_text = None
            if ctx.sandbox.config_file.exists():
                config_text = ph.apply(ctx.sandbox.config_file.read_text(encoding="utf-8"))
        return BootedBase(scenario["name"], attrs, config, os_env, large_env, config_text)

    # -- owner classes -------------------------------------------------------------
    def owner_class(self, mode: str):
        if mode not in self._classes:
            builder = {"legacy": self._legacy_class, "stop_control": self._legacy_class,
                       "desktop": self._desktop_class, "mixins": self._mixins_class}[mode]
            self._classes[mode] = builder()
        return self._classes[mode]

    def gui_mixins(self) -> tuple:
        """The unmoved desktop GUI mixins (live: shared by the legacy and desktop sides)."""
        return tuple(getattr(importlib.import_module(m), c) for m, c in GUI_MIXINS)

    def _legacy_class(self):
        bundle = self.bundle
        ns = {n: _recorder(n) for n in self.recorded}
        ns.update(_latch_namespace())
        bases = (fakes.FakeState, bundle.methods) + bundle.mixin_classes() + self.gui_mixins()
        return fm.owner_class(fm.OWNER_CLASS_NAME, bases, ns, tag="legacy")

    def translator_gui_class(self):
        if importlib.util.find_spec("PySide6") is None:
            raise Unavailable("PySide6 is not installed (desktop mode resolves TranslatorGUI's MRO)")
        normalize.preimport_backend_modules()
        import translator_gui

        return translator_gui.TranslatorGUI

    def _desktop_class(self):
        tg = self.translator_gui_class()
        ns: dict = {}
        signals = set(self._signal_names)
        for klass in reversed(tg.__mro__):
            if klass is object or (klass.__module__ or "").startswith(fm._FOREIGN_MODULE_PREFIXES):
                continue
            for name, value in vars(klass).items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                if type(value).__module__.startswith(("PySide6", "shiboken6")) and type(value).__name__ == "Signal":
                    signals.add(name)
                    continue
                ns[name] = value
        for name in signals:
            ns.pop(name, None)
        for name in self.recorded:
            ns[name] = _recorder(name)
        ns.update(_latch_namespace())
        self._signal_names = tuple(sorted(signals))
        return fm.owner_class(fm.OWNER_CLASS_NAME, (fakes.FakeState,), ns, tag="desktop")

    def pipeline_mixins(self) -> list:
        out = []
        for module_name, cls_name in U3_MIXINS:
            if not module_exists(module_name):
                continue
            cls = getattr(importlib.import_module(module_name), cls_name, None)
            if cls is not None:
                out.append(cls)
        return out

    def _mixins_class(self):
        import headless_owner

        ho = headless_owner.HeadlessOwner
        extra = tuple(c for c in self.pipeline_mixins() if c not in ho.__mro__)
        bases = (fakes.FakeState,) + extra + (ho,)
        provided = set()
        for klass in bases[1:]:
            for k in klass.__mro__:
                if k is not object:
                    provided |= set(vars(k))
        ns = {n: _recorder(n) for n in self.recorded if n not in provided}
        ns.update({n: _recorder(n) for n in MOBILE_FORCED_RECORDERS})
        ns.update(_latch_namespace())
        return fm.owner_class(fm.OWNER_CLASS_NAME, bases, ns, tag="mixins")

    # -- availability -------------------------------------------------------------
    def check_mode(self, mode: str) -> None:
        if mode == "legacy":
            return
        if mode == "stop_control":
            if not module_exists("stop_control"):
                raise Unavailable("src/stop_control.py does not exist yet")
            return
        if os.environ.get("PARITY_TRACE_FORCE") != "1":
            if mode == "desktop":
                if not any(module_exists(m) for m in DESKTOP_REQUIRES_ANY):
                    raise Unavailable("tier T desktop side waits for the first U3 shared module (src/"
                                      + ", src/".join(f"{m}.py" for m in DESKTOP_REQUIRES_ANY) + ")")
            missing = [] if mode == "desktop" else missing_modules(MOBILE_REQUIRES)
            if missing:
                raise Unavailable(f"tier T {mode} side waits for the U3 shared modules: src/"
                                  + ", src/".join(f"{m}.py" for m in missing) + " do not exist yet")
        if mode == "desktop":
            self.translator_gui_class()
        if mode == "mixins":
            import headless_owner  # noqa: F401
            for module_name, cls_name in U3_MIXINS[:1]:
                if module_exists(module_name):
                    cls = getattr(importlib.import_module(module_name), cls_name, None)
                    for name in ("_prepare_translation_run", "_translation_worker"):
                        if cls is None or not hasattr(cls, name):
                            raise Unavailable(f"{module_name}.{cls_name}.{name} is not defined yet")


# ===========================================================================
# Running a side
# ===========================================================================


@dataclass
class TraceResult:
    scenario: str
    mode: str
    trace: list = field(default_factory=list)
    result: object = None
    exception: object = None
    env: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    fs: dict = field(default_factory=dict)
    stdout: list = field(default_factory=list)
    argv: list = field(default_factory=list)
    cwd: str = ""
    violations: list = field(default_factory=list)
    #: normalized env the run started from (input, not compared; TraceView.entry_env merges it)
    pre_env: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"trace": self.trace, "result": self.result, "exception": self.exception, "env": self.env,
                "attrs": self.attrs, "config": self.config, "fs": self.fs, "stdout": self.stdout,
                "argv": self.argv, "cwd": self.cwd}


class TraceContext:
    """One sandbox per scenario; every side runs in it after a full process/sandbox reset."""

    def __init__(self, session: TraceSession, scenario: dict):
        self.session = session
        self.scenario = scenario
        bundle = session.bundle
        base = session.base(scenario)
        files = dict(scenario.get("files") or {})
        if base.config_text is not None:
            files["src/config.json"] = base.config_text
        self.fctx = fm.FuzzContext(f"trace-{scenario['name']}", backend=True, files=files)
        self.bundle = bundle
        self.base = base
        self.tracer = None

    # -- lifecycle ----------------------------------------------------------------
    def __enter__(self):
        fctx = self.fctx
        fctx.__enter__()
        try:
            self._setup()
        except BaseException:
            fctx.__exit__(*sys.exc_info())
            raise
        return self

    def _setup(self):
        fctx, bundle = self.fctx, self.bundle
        fm.patch_legacy_namespace(fctx, bundle)
        file_modules = [m for m in FILE_MODULES if m in sys.modules]
        fm.patch_src_module_paths(fctx, extra_file_modules=file_modules)
        fm.patch_translator_gui_backends(fctx)
        fm.install_side_effect_stubs(fctx)
        namespaces = [bundle.namespace, *bundle.externals.values(), *bundle.mixin_namespaces()]
        if "translator_gui" in sys.modules:
            namespaces.append(sys.modules["translator_gui"].__dict__)
        fm.install_qt_stubs(fctx, _qt_names(bundle), (), namespaces)
        # the unmoved GUI mixin modules: every Qt name of their namespaces (never the PySide6 modules)
        for module_name, _cls in GUI_MIXINS:
            module_ns = importlib.import_module(module_name).__dict__
            for name in sorted(module_ns):
                if fm._QT_NAME_RE.match(name) and _is_qt_object(module_ns[name]):
                    fctx.patch_dict(module_ns, name, fm.QtStub(name, fctx.recorder))
        self.tracer = Tracer(self, self.session)
        backend_namespaces = [bundle.namespace]
        if "translator_gui" in sys.modules:
            backend_namespaces.append(sys.modules["translator_gui"].__dict__)
        self.tracer.install(backend_namespaces)
        self._module_enter, self._module_exit = fm._module_patch_hooks(bundle)
        fctx.rebaseline()

    def __exit__(self, exc_type, exc, tb):
        import logging

        try:
            return self.fctx.__exit__(exc_type, exc, tb)
        finally:
            for name, level in self.session.logger_levels.items():
                logging.getLogger(name).setLevel(level)

    # -- delegation ---------------------------------------------------------------
    @property
    def recorder(self):
        return self.fctx.recorder

    @property
    def sandbox(self):
        return self.fctx.sandbox

    def patch(self, obj, attr, value):
        self.fctx.patch(obj, attr, value)

    def patch_dict(self, mapping, key, value):
        self.fctx.patch_dict(mapping, key, value)

    def resolve(self, value):
        return self.fctx.resolve(value)

    # -- one side -----------------------------------------------------------------
    def _build_owner(self, mode, owner_cls=None):
        session, tracer, scenario = self.session, self.tracer, self.scenario
        cls = owner_cls or session.owner_class(mode)
        owner = cls(self.recorder)
        data = owner.__dict__
        for key, value in self.resolve(self.base.attrs).items():
            data[key] = fm.fast_copy(value)
        owner.__dict__["config"] = fm.fast_copy(self.resolve(self.base.config))
        for name in session.bound:
            data[name] = types.MethodType(_recorder(name), owner)
        for name in session._signal_names:
            if name not in data:
                data[name] = fakes.FakeSignal(self.recorder, name)
        data["direct_text_glossary_approval_signal"] = AnsweringSignal(
            self.recorder, "direct_text_glossary_approval_signal", tracer)
        if mode == "mixins":
            data["host"] = TraceHost(tracer)
        if mode not in LEGACY_CODE_MODES:
            for hook in NEW_SIDE_BOOT_HOOKS:
                hook(owner)
        run_attrs = scenario.get("run_attrs") or {}
        if callable(run_attrs):
            run_attrs = run_attrs(self.sandbox)
        for key, value in self.resolve(run_attrs).items():
            data[key] = value
        return owner

    def _apply_boot_env(self):
        env = self.resolve(self.base.os_env)
        for key, value in env.items():
            os.environ[key] = value
        large = self.resolve(self.base.large_env)
        if large:
            import large_env

            for key, value in large.items():
                large_env.set_env(key, value)

    def run(self, mode: str, *, owner_cls=None) -> TraceResult:
        """Run *scenario* on one side. *owner_cls* replaces the side's class (harness self-tests:
        a mutant of the legacy class driven like *mode*)."""
        fctx, tracer, scenario = self.fctx, self.tracer, self.scenario
        driver = DRIVERS[mode]
        fctx.reset(0)
        tracer.reset(scenario, mode)
        self._apply_boot_env()
        owner = self._build_owner(mode, owner_cls)
        tracer.owner = owner
        if mode in LEGACY_CODE_MODES:
            fctx.patch_dict(self.bundle.namespace, "TranslatorGUI", type(owner))
        pre_run = scenario.get("pre_run")
        if pre_run:
            pre_run(owner, self)
        fctx.recorder.clear()
        fctx.base_stdout.seek(0)
        fctx.base_stdout.truncate()
        tracer.pre_env = normalize.effective_env()
        pre_attrs = self._attrs(owner)
        pre_config = copy.deepcopy(owner.__dict__.get("config", {}))
        owner.__dict__["_parity_tracer"] = tracer
        out = TraceResult(scenario["name"], mode)
        if mode in LEGACY_CODE_MODES:
            self._module_enter()
        result = None
        try:
            result = driver.run(tracer, owner, scenario)
            tracer.gate.drain()
            for action in scenario.get("post_actions") or ():
                tracer.fire(action)
            tracer.gate.drain()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - an escaped exception is an outcome
            out.exception = {"type": type(exc).__name__, "message": fm.mask(str(exc))[:300]}
        finally:
            owner.__dict__.pop("_parity_tracer", None)
            if mode in LEGACY_CODE_MODES:
                self._module_exit()
        norm = self._normalizer(owner)
        out.result = norm(result)
        out.trace = norm(list(fctx.recorder.calls))
        out.env = norm(normalize.env_delta(tracer.pre_env, normalize.effective_env()))
        out.attrs = norm(normalize.dict_delta(pre_attrs, self._attrs(owner)))
        post_config = owner.__dict__.get("config", {})
        out.config = norm(normalize.dict_delta(fm.canon(pre_config), fm.canon(post_config)))
        out.fs = norm(fctx.fs_delta())
        out.stdout = norm(fctx.stdout_lines())
        out.argv = norm(list(sys.argv))
        out.cwd = norm(os.getcwd())
        out.violations = list(fctx.violations)
        out.pre_env = norm(dict(tracer.pre_env))
        fctx.restore_fs()
        tracer.owner = None
        return out

    @staticmethod
    def _attrs(owner) -> dict:
        out = {}
        for key, value in owner.__dict__.items():
            if key == "config" or key.startswith("_parity_") or key == "host":
                continue
            if isinstance(value, (fakes.FakeSignal, types.MethodType)):
                continue
            out[key] = fm.canon(value.describe() if isinstance(value, fm._WIDGET_TYPES) else value, owner)
        return out

    def _normalizer(self, owner):
        ph = normalize.PathPlaceholders([
            (str(self.sandbox.root), normalize.SANDBOX_TOKEN),
            (str(TRACE_LEGACY_DIR), "<ORACLE>"),
            (str(SRC_DIR), "<SRC>"),
            (str(REPO_ROOT), "<REPO>"),
            (os.path.expanduser("~"), "<HOME>"),
        ])

        def norm(value):
            return _mask_volatile(fm.map_strings(fm.canon(value, owner), ph.apply))

        return norm


#: integers this large are file timestamps in ns (e.g. the glossary-folder signatures
#: auto-mapping caches in ``_glossary_dir_candidate_cache``): wall clock, not behaviour
_NS_TIMESTAMP_MIN = 10 ** 17


def _mask_volatile(value):
    t = type(value)
    if t is int and abs(value) >= _NS_TIMESTAMP_MIN:
        return "<mtime_ns>"
    if t is dict:
        return {_mask_volatile(k): _mask_volatile(v) for k, v in value.items()}
    if t is list:
        return [_mask_volatile(v) for v in value]
    if t is tuple:
        return tuple(_mask_volatile(v) for v in value)
    return value


# ===========================================================================
# Views, projections and comparison
# ===========================================================================


def channel(event) -> str:
    name = event[0] if isinstance(event, (list, tuple)) and event else ""
    if not isinstance(name, str):
        return "other"
    if ":" in name:
        head = name.split(":", 1)[0]
        return head
    if name == "ask":
        return "ask"
    if name == "append_log":
        return "log"
    return "gui"


class TraceView:
    """Read helpers over a normalized :class:`TraceResult` (scenario expectations)."""

    def __init__(self, result: TraceResult):
        self.result = result
        self.trace = result.trace

    def events(self, prefix):
        return [e for e in self.trace if isinstance(e[0], str) and e[0].startswith(prefix)]

    def entries(self) -> list:
        return [e[0][len("backend:"):] for e in self.events("backend:")]

    def entry_env(self, entry, index):
        """Full environment (run start + delta, large_env merged) the *index*-th call saw."""
        calls = [e for e in self.events(f"backend:{entry}")]
        if index >= len(calls):
            return None
        delta = calls[index][2].get("env", {})
        env = dict(self.result.pre_env)
        for key in delta.get("removed", ()):
            env.pop(key, None)
        env.update(delta.get("set", {}))
        return env

    def logs(self) -> list:
        return [str(e[1][0]) for e in self.trace if e[0] == "append_log" and e[1]]

    def _first_click_index(self):
        for i, e in enumerate(self.trace):
            if e[0] == "action:click":
                return i
        return None

    def first_check_after_action(self):
        start = self._first_click_index()
        if start is None:
            return None
        for e in self.trace[start + 1:]:
            if isinstance(e[0], str) and e[0].startswith("check:"):
                return e[2]
        return None

    @staticmethod
    def flag(check_kwargs, key):
        if key == "stop_callback":
            return check_kwargs.get("stop_callback")
        return (check_kwargs.get("flags") or {}).get(key)

    def stop_api_calls(self, after_action=True) -> list:
        start = self._first_click_index() if after_action else -1
        if start is None:
            return []
        return [e[0][len("stopapi:"):] for e in self.trace[start + 1:] if str(e[0]).startswith("stopapi:")]

    def asks(self) -> int:
        return len(self.events("ask"))


def expectation_problems(scenario: dict, result: TraceResult) -> list:
    problems = []
    if result.exception:
        problems.append(f"escaped exception {result.exception}")
    for e in result.trace:
        if e[0] == "thread:exception":
            problems.append(f"worker thread raised {e[1]}")
    errors = [line for line in TraceView(result).logs() if line.startswith(("❌ Error in thread", "❌ Translation setup error"))]
    problems.extend(f"log: {line[:200]}" for line in errors)
    expect = scenario.get("expect")
    if expect:
        problems.extend(expect(TraceView(result)))
    problems.extend(f"sandbox violation: {v}" for v in result.violations)
    return problems


_FLAG_KEYS_MOBILE = STOP_ENV_KEYS + ("stop_requested", "graceful_stop_active", "glossary_stop_file_exists") + tuple(
    f"module:{m}" for m in MODULE_FLAGS) + ("UnifiedClient._global_cancelled", "unified_api_client.global_stop_flag")


def _mask_env(env):
    if not isinstance(env, dict):
        return env
    out = dict(env)
    for key in ("set",):
        if isinstance(out.get(key), dict):
            out[key] = {k: ("<masked>" if k in MOBILE_MASKED_ENV else v) for k, v in out[key].items()}
    return out


def _mobile_event(event):
    name, args, kwargs = event[0], event[1], event[2]
    kw = dict(kwargs)
    if "flags" in kw and isinstance(kw["flags"], dict):
        kw["flags"] = {k: kw["flags"].get(k) for k in _FLAG_KEYS_MOBILE}
        # The shared pipelines read graceful_stop_active with getattr(..., False). A fresh mobile
        # glossary job owner has no value until a stop latches one, while the desktop Extract
        # Glossary button sets False at run start (U6 glossary stop scenarios): equivalent.
        if kw["flags"].get("graceful_stop_active") == _ABSENT:
            kw["flags"]["graceful_stop_active"] = False
    if "env" in kw:
        kw["env"] = _mask_env(kw["env"])
    if "env_vars" in kw and isinstance(kw["env_vars"], dict):
        kw["env_vars"] = {k: ("<masked>" if k in MOBILE_MASKED_ENV else v) for k, v in kw["env_vars"].items()}
    if name == "ask":
        return ["ask", list(args), {"answer": kw.get("answer")}]
    return [name, list(args), kw]


def _mobile_keep(event, clicked) -> bool:
    name = str(event[0])
    if name in ("action:job_start", "action:pipeline_start"):
        return False  # mobile JobService markers (no desktop counterpart)
    if name.startswith("stopapi:") and not clicked:
        # run-start resets: the desktop runs them inside run_translation_thread, the mobile
        # JobService once more before the pipeline; their outcome is compared through the
        # stop flags every backend entry records
        return False
    return channel(event) in MOBILE_CHANNELS


def project(result: TraceResult, projection: str = "full", mode: str | None = None) -> dict:
    """The comparable part of a result: ``full`` (desktop) or ``mobile`` (mixins)."""
    known = [prefix for prefix, _reason in KNOWN_TRACE_DIVERGENCES.get(mode or "", [])]

    def keep(event):
        return not any(str(event[0]).startswith(p) for p in known)

    if projection == "full":
        rec = result.record()
        rec["trace"] = [e for e in rec["trace"] if keep(e)]
        return rec
    if projection != "mobile":
        raise ValueError(projection)
    events, clicked = [], False
    for e in result.trace:
        clicked = clicked or e[0] == "action:click"
        if keep(e) and _mobile_keep(e, clicked):
            events.append(_mobile_event(e))
    latches = [_mobile_event(e) for e in result.trace
               if e[0] == "latch:stop_requested" and e[1] and e[1][0] is True]
    fs = {}
    for kind, value in (result.fs or {}).items():
        kept = ({k: v for k, v in value.items() if _mobile_fs_keep(kind, k)} if isinstance(value, dict)
                else [k for k in value if _mobile_fs_keep(kind, k)])
        if kept:  # an empty delta and a delta that only held ignored paths compare equal
            fs[kind] = kept
    return {"trace": events, "stop_latch": latches, "fs": fs, "exception": result.exception}


def compare(expected: TraceResult, actual: TraceResult, projection: str = "full", *, limit=40) -> list:
    from parity import capture_golden

    a = project(expected, projection, actual.mode)
    b = project(actual, projection, actual.mode)
    out = []
    for key in a:
        if a[key] != b[key]:
            out.extend(capture_golden.diff(a[key], b[key], path=key, limit=limit - len(out)))
            if key == "trace":
                out.append(first_divergence(a[key], b[key]))
        if len(out) >= limit:
            break
    return out


def first_divergence(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"first trace divergence at event {i}: expected {str(x)[:300]!r} != actual "
                    f"{str(y)[:300]!r}")
    if len(a) != len(b):
        longer, which = (a, "expected") if len(a) > len(b) else (b, "actual")
        return (f"trace lengths differ ({len(a)} vs {len(b)}); {which} continues with "
                f"{str(longer[min(len(a), len(b))])[:300]!r}")
    return "traces equal"


# ===========================================================================
# High-level helpers
# ===========================================================================

_SESSION = None
_CACHE: dict = {}


def session() -> TraceSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = TraceSession(os.environ.get("PARITY_TRACE_SHA") or None)
    return _SESSION


def run_modes(name: str, modes, sess: TraceSession | None = None) -> dict:
    """{mode: TraceResult} for one scenario, all sides in one sandbox (cached per mode tuple)."""
    sess = sess or session()
    key = (name, tuple(modes), id(sess))
    if key not in _CACHE:
        scenario = ts.get(name)
        for mode in modes:
            if mode != "legacy":
                sess.check_mode(mode)
        out = {}
        with TraceContext(sess, scenario) as tctx:
            for i, mode in enumerate(modes):
                out[f"{mode}#{i}" if list(modes).count(mode) > 1 else mode] = tctx.run(mode)
        _CACHE[key] = out
    return _CACHE[key]


def mutant_legacy_class(sess: TraceSession, method: str, old: str, new: str):
    """The legacy owner class with one frozen method recompiled from edited source (self-tests)."""
    import textwrap

    base = sess.owner_class("legacy")
    # resolved like the legacy owner does: the frozen TranslatorGUI body, then (oracles frozen
    # after a move, e.g. U3's _process_text_file) the frozen shared mixin copies
    original = fm.resolve_python_mro(base, method)
    if original is fm.MISSING:
        original = getattr(sess.bundle.methods, method)
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(old) != 1:
        raise ValueError(f"{method}: edit anchor found {source.count(old)} times")
    namespace: dict = {}
    exec(compile(source.replace(old, new), f"<mutant {method}>", "exec"), original.__globals__, namespace)
    return fm.owner_class(fm.OWNER_CLASS_NAME, (base,), {method: namespace[method]}, tag=f"mutant-{method}")


def run_mutant(name: str, mode: str, owner_cls, sess: TraceSession | None = None):
    """(legacy result, mutant result) of scenario *name* in one sandbox (never cached)."""
    sess = sess or session()
    with TraceContext(sess, ts.get(name)) as tctx:
        return tctx.run("legacy"), tctx.run(mode, owner_cls=owner_cls)


def legacy_pair(name: str, sess: TraceSession | None = None):
    res = run_modes(name, ("legacy", "legacy"), sess)
    return res["legacy#0"], res["legacy#1"]


def legacy_vs(name: str, mode: str, sess: TraceSession | None = None):
    res = run_modes(name, ("legacy", mode), sess)
    return res["legacy"], res[mode]


# ===========================================================================
# CLI
# ===========================================================================


def _print_result(res: TraceResult, *, dump=False):
    print(f"== {res.scenario} [{res.mode}] exception={res.exception}")
    for e in res.trace:
        name = e[0]
        if not dump and name == "append_log":
            print(f"   log  {str(e[1][0])[:160]}")
        elif not dump and name.startswith(("backend:", "check:", "stopapi:", "action:", "latch:", "ask",
                                           "hook:", "thread:")):
            kw = dict(e[2])
            env = kw.pop("env", None)
            kw.pop("large_env", None)
            flags = kw.pop("flags", None)
            short_flags = {k: v for k, v in (flags or {}).items() if v not in (None, False, _ABSENT)}
            extra = f" env+{len(env.get('set', {}))}" if isinstance(env, dict) else ""
            print(f"   {name:60s} {e[1]!s:.60} {kw!s:.120}{extra} flags={short_flags}")
        elif dump:
            print(f"   {e!s:.2000}")
    if res.violations:
        print(f"   VIOLATIONS: {res.violations}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tier T trace harness")
    parser.add_argument("scenario", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--freeze", action="store_true", help="freeze the trace oracle and exit")
    parser.add_argument("--sha", default="HEAD", help="commit to freeze (with --freeze)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--mode", default="legacy", choices=sorted(DRIVERS))
    parser.add_argument("--compare", choices=["desktop", "mixins", "stop_control", "legacy"],
                        help="compare legacy with this side (legacy = determinism)")
    parser.add_argument("--dump", action="store_true", help="print every event in full")
    args = parser.parse_args(argv)
    if args.freeze:
        path = freeze_trace_oracle(args.sha)
        bundle = load_trace_oracle(freeze_legacy.resolve_sha(args.sha))
        m = bundle.manifest
        print(f"trace oracle {m['sha']} -> {path.relative_to(REPO_ROOT)}")
        print(f"  frozen methods: {len(m['frozen_methods'])}; recorded: {m['recorded_methods']}")
        print(f"  missing: {m['missing'] or 'none'}")
        return 0
    if args.list:
        for name, scenario in ts.SCENARIOS.items():
            print(f"{name:40s} modes={','.join(scenario['modes'])}  {scenario['description']}")
        return 0
    names = args.scenario or list(ts.SCENARIO_NAMES)
    failures = 0
    for name in names:
        scenario = ts.get(name)
        if args.compare:
            try:
                if args.compare == "legacy":
                    a, b = legacy_pair(name)
                    problems = compare(a, b, "full")
                else:
                    a, b = legacy_vs(name, args.compare)
                    problems = compare(a, b, "full" if args.compare == "desktop" else "mobile")
            except Unavailable as exc:
                print(f"{name:40s} SKIP {exc}")
                continue
            status = "MATCH" if not problems else f"DIFF ({len(problems)})"
            failures += bool(problems)
            print(f"{name:40s} {status}")
            for line in problems[:25]:
                print(f"    {line}")
            continue
        res = run_modes(name, (args.mode,))[args.mode]
        _print_result(res, dump=args.dump)
        problems = expectation_problems(scenario, res) if args.mode == "legacy" else []
        for p in problems:
            print(f"   EXPECTATION: {p}")
        failures += bool(problems)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
