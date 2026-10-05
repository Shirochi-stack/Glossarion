"""Tier T: pipeline order and stop races, legacy desktop vs the U3 shared pipeline.

Drives the frozen desktop run pipeline (``run_translation_thread`` + its worker,
``run_translation_direct``, ``run_glossary_extraction_direct``, ``stop_translation``) and the
new ``TranslationPipelineMixin._translation_worker`` + ``stop_control`` on the same booted
FakeState owner with recording backend stubs, and compares the traces
(see ``trace_harness`` for the event channels and ``trace_scenarios`` for the scenarios).

* Oracle checks: the trace oracle is a verbatim copy of ``git show <sha>`` and closes over
  the pipeline; it is frozen at the parent commit of the U3 step.
* Legacy self-consistency (must pass before any U3 module exists): every scenario traces
  identically twice in one sandbox, reaches the paths it claims (scenario expectations),
  and the comparison catches an injected stop-latch reordering and a backend env change.
* stop_control: the frozen desktop run with Stop clicks routed through
  ``stop_control.request_stop`` (the mobile reference click) reproduces desktop
  ``stop_translation`` (mobile projection).
* Desktop parity: the working-tree TranslatorGUI MRO (shared mixins + hook overrides) must
  reproduce the legacy record exactly.
* Mobile parity: HeadlessOwner's bases + the U3 pipeline mixins (GUI-free hooks) driven as
  ``_prepare_translation_run`` + ``_translation_worker`` with ``stop_control`` clicks must
  reproduce the legacy backend calls, stop checks, stop API calls, stop latch, approval
  questions and sandbox files.
* U3 contracts (names the parallel modules agree on) are checked once each module exists.

Run (repository root)::

    python tests/parity/trace_harness.py --freeze          # once per U3 step (parent commit)
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/parity/test_trace_parity.py

``PARITY_TRACE_FORCE=1`` runs the desktop/mobile comparisons even while U3 modules are
missing (debugging aid); ``PARITY_TRACE_SHA`` selects another frozen trace oracle.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from parity import freeze_legacy  # noqa: E402
from parity import trace_harness as th  # noqa: E402
from parity import trace_scenarios as ts  # noqa: E402

NAMES = ts.SCENARIO_NAMES


def _originals() -> dict:
    import concurrent.futures
    import time

    return {"start": threading.Thread.start, "is_alive": threading.Thread.is_alive,
            "submit": concurrent.futures.ThreadPoolExecutor.submit, "time": time.time}


#: process-wide callables the harness patches during a run (captured before any run)
_ORIGINAL = _originals()


def _skip_unless_modules(*modules):
    missing = th.missing_modules(modules)
    if missing:
        pytest.skip("U3 module(s) not written yet: " + ", ".join(f"src/{m}.py" for m in missing))


@pytest.fixture(scope="module")
def session():
    sess = th.session()
    try:
        sess.bundle  # noqa: B018 - loads the trace oracle
    except th.Unavailable as exc:
        pytest.skip(str(exc))
    return sess


# ===========================================================================
# The trace oracle
# ===========================================================================


def test_trace_oracle_methods_are_verbatim(session):
    import ast as _ast

    bundle = session.bundle
    manifest = bundle.manifest
    original = freeze_legacy.git_show_text(manifest["sha"], manifest["source_file"]).split("\n")
    frozen_lines = bundle.path.read_text(encoding="utf-8").split("\n")
    tree = _ast.parse("\n".join(frozen_lines))
    cls = next(n for n in tree.body if isinstance(n, _ast.ClassDef) and n.name == "LegacyMethods")
    checked = 0
    for node in cls.body:
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        info = manifest["frozen_methods"].get(node.name)
        if info is None:
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        a, b = info["lines"]
        assert frozen_lines[start - 1:node.end_lineno] == original[a - 1:b], f"{node.name} differs from git show"
        checked += 1
    assert checked == len(manifest["frozen_methods"])
    for module, info in (manifest.get("frozen_mixins") or {}).items():
        path = Path(vars(bundle.mixins[module])["__frozen_path__"])
        source = freeze_legacy.frozen_mixin_source(path, info)
        assert source == freeze_legacy.git_show_text(manifest["sha"], info["source_file"]), module


def test_trace_oracle_closes_over_the_pipeline(session):
    manifest = session.bundle.manifest
    frozen = set(manifest["frozen_methods"])
    wanted = {"run_translation_thread", "run_translation_direct", "stop_translation",
              "run_glossary_extraction_direct", "_process_text_file", "_extract_glossary_from_text_file",
              "_run_parallel_metadata_files", "_prepare_multipass_qa_refinement_run",
              "_auto_load_glossary_after_extraction", "_await_direct_text_glossary_approval",
              "_lazy_load_modules"}
    assert not manifest["missing"], manifest["missing"]
    assert wanted <= frozen | set(manifest.get("moved_to_mixins") or ()), sorted(wanted - frozen)
    gui_only = freeze_legacy.TG_RECORDED_METHODS | th.TRACE_RECORDED_METHODS
    assert set(manifest["recorded_methods"]) <= gui_only, sorted(set(manifest["recorded_methods"]) - gui_only)


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def test_trace_oracle_is_frozen_at_parent_commit(session):
    present = [m for m in th.U3_MODULES if th.module_exists(m)]
    if not present:
        pytest.skip("no U3 module exists yet: the trace oracle only has to be frozen before the first move")
    expected = os.environ.get("PARITY_TRACE_SHA") or _git_head()
    if not expected:
        pytest.skip("git unavailable")
    manifest = session.bundle.manifest
    if manifest["sha"] == expected:
        return
    sources = {manifest["source_file"]: manifest["source_sha256"]}
    sources.update({i["source_file"]: i["source_sha256"] for i in manifest["externals"].values()})
    sources.update({i["source_file"]: i["source_sha256"] for i in (manifest.get("frozen_mixins") or {}).values()})

    def digest_at(path):
        try:
            return freeze_legacy._sha256(freeze_legacy.git_show_text(expected, path))
        except subprocess.CalledProcessError:
            return None

    changed = sorted(p for p, d in sources.items() if digest_at(p) != d)
    assert not changed, (f"tier T compares against legacy @ {manifest['sha'][:12]} but {changed} changed by "
                         f"{expected[:12]}; re-run python tests/parity/trace_harness.py --freeze")


# ===========================================================================
# Legacy self-consistency (runs now)
# ===========================================================================


def test_scenarios_cover_the_u3_trace_matrix():
    required = {"plain_translate", "fresh_install_translate", "balanced_pre_glossary",
                "require_complete_gate_blocks", "multipass_refinement_followup", "metadata_only_batch",
                "single_chapter_filter", "multi_epub_glossary_map", "graceful_stop", "immediate_stop",
                "double_click_force_stop", "direct_text_glossary_approved", "direct_text_glossary_rejected"}
    assert required <= set(NAMES), sorted(required - set(NAMES))
    assert {"graceful_stop", "immediate_stop", "double_click_force_stop", "click_after_graceful_finish",
            "immediate_stop_during_pre_glossary"} <= set(ts.STOP_SCENARIOS)
    for name in NAMES:
        assert set(ts.modes(name)) <= set(ts.ALL_MODES), name
        assert ts.get(name).get("expect"), f"{name} has no legacy expectation"


@pytest.mark.parametrize("name", NAMES)
def test_legacy_trace_is_deterministic(session, name):
    first, second = th.legacy_pair(name, session)
    problems = th.compare(first, second, "full")
    assert not problems, "\n".join(problems)
    assert first.trace, "empty trace"


@pytest.mark.parametrize("name", NAMES)
def test_legacy_trace_reaches_the_scenario_paths(session, name):
    first, _second = th.legacy_pair(name, session)
    problems = th.expectation_problems(ts.get(name), first)
    assert not problems, "\n".join(problems)


def test_stop_latch_follows_the_env_mode_flags(session):
    """Desktop race rule (stop_translation): GRACEFUL_STOP/WAIT_FOR_CHUNKS are published before
    stop_requested latches, and an immediate stop sets TRANSLATION_CANCELLED first."""
    graceful, _ = th.legacy_pair("graceful_stop", session)
    latch = next(e for e in graceful.trace if e[0] == "latch:stop_requested" and e[1] == [True])
    assert latch[2]["flags"]["GRACEFUL_STOP"] == "1" and latch[2]["flags"]["WAIT_FOR_CHUNKS"] == "1"
    assert latch[2]["flags"]["TRANSLATION_CANCELLED"] is None
    immediate, _ = th.legacy_pair("immediate_stop", session)
    latch = next(e for e in immediate.trace if e[0] == "latch:stop_requested" and e[1] == [True])
    assert latch[2]["flags"]["TRANSLATION_CANCELLED"] == "1" and latch[2]["flags"]["GRACEFUL_STOP"] == "0"


def test_trace_detects_a_stop_latch_reordering(session):
    """Harness self-test: latching stop_requested before the env flags must be caught by both
    projections (the race the desktop comment at stop_translation warns about)."""
    mutant = th.mutant_legacy_class(
        session, "stop_translation",
        "os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'",
        "self.stop_requested = True; os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'",
    )
    legacy, changed = th.run_mutant("graceful_stop", "legacy", mutant, session)
    assert th.compare(legacy, changed, "full"), "full projection missed the latch reordering"
    mobile = th.compare(legacy, changed, "mobile")
    assert mobile, "mobile projection missed the latch reordering"
    assert any("stop_latch" in line or "latch" in line for line in mobile), mobile


def test_trace_detects_a_backend_env_change(session):
    """Harness self-test: one env value changed on the way to TransateKRtoEN.main is caught by the
    mobile projection (backend env), not only by the full record."""
    mutant = th.mutant_legacy_class(
        session, "_process_text_file",
        "env_vars['MULTIPASS_MODE'] = '1' if multipass_enabled else '0'",
        "env_vars['MULTIPASS_MODE'] = '1' if multipass_enabled else 'no'",
    )
    legacy, changed = th.run_mutant("plain_translate", "legacy", mutant, session)
    problems = th.compare(legacy, changed, "mobile")
    assert any("MULTIPASS_MODE" in p for p in problems), problems


def test_trace_runs_leave_no_process_state_behind(session):
    """After a scenario: thread/executor/clock/psutil patches, stop flags and loggers are restored."""
    import concurrent.futures
    import logging
    import time

    import unified_api_client

    th.legacy_pair("immediate_stop", session)
    assert threading.Thread.start is _ORIGINAL["start"]
    assert threading.Thread.is_alive is _ORIGINAL["is_alive"]
    assert concurrent.futures.ThreadPoolExecutor.submit is _ORIGINAL["submit"]
    assert time.time is _ORIGINAL["time"]
    assert getattr(unified_api_client, "global_stop_flag", False) is False
    for name, level in session.logger_levels.items():
        assert logging.getLogger(name).level == level, name
    assert "TRANSLATION_CANCELLED" not in os.environ


# ===========================================================================
# stop_control vs desktop stop_translation (runs as soon as stop_control exists)
# ===========================================================================

STOP_CONTROL_SCENARIOS = tuple(n for n in ts.STOP_SCENARIOS if "mixins" in ts.modes(n))


@pytest.mark.parametrize("name", STOP_CONTROL_SCENARIOS)
def test_stop_control_reproduces_the_desktop_stop_protocol(session, name):
    """The frozen desktop run with every Stop click going through ``stop_control.request_stop``
    (the mobile reference click) must match the desktop's own ``stop_translation`` on the mobile
    projection: stop API calls and their flag snapshots, the latch, every backend stop check."""
    try:
        session.check_mode("stop_control")
    except th.Unavailable as exc:
        pytest.skip(str(exc))
    legacy, hybrid = th.legacy_vs(name, "stop_control", session)
    problems = th.compare(legacy, hybrid, "mobile")
    assert not problems, "\n".join(problems)


# ===========================================================================
# Desktop parity: working-tree TranslatorGUI vs legacy (full record)
# ===========================================================================


@pytest.mark.parametrize("name", NAMES)
def test_desktop_trace_matches_legacy(session, name):
    if "desktop" not in ts.modes(name):
        pytest.skip(f"{name} is not a desktop scenario")
    try:
        session.check_mode("desktop")
    except th.Unavailable as exc:
        pytest.skip(str(exc))
    legacy, desktop = th.legacy_vs(name, "desktop", session)
    problems = th.compare(legacy, desktop, "full")
    assert not problems, "\n".join(problems)


# ===========================================================================
# Mobile parity: shared mixins + stop_control vs legacy (mobile projection)
# ===========================================================================


@pytest.mark.parametrize("name", NAMES)
def test_mobile_trace_matches_legacy(session, name):
    if "mixins" not in ts.modes(name):
        pytest.skip(f"{name} is a desktop-only race (the Run button re-click after the worker ended)")
    try:
        session.check_mode("mixins")
    except th.Unavailable as exc:
        pytest.skip(str(exc))
    legacy, mobile = th.legacy_vs(name, "mixins", session)
    problems = th.compare(legacy, mobile, "mobile")
    assert not problems, "\n".join(problems)


# ===========================================================================
# U3 contracts (checked once each module exists)
# ===========================================================================


def _params(fn) -> dict:
    return dict(inspect.signature(fn).parameters)


def test_stop_control_contract():
    _skip_unless_modules("stop_control")
    import stop_control

    params = _params(stop_control.reset_for_new_run)
    assert params["kind"].default == "translation"
    params = _params(stop_control.request_stop)
    for name in ("graceful", "wait_for_chunks", "force", "set_stop_requested", "log", "clear_watchdog"):
        assert name in params and params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert params["force"].default is False and params["clear_watchdog"].default is None
    assert list(_params(stop_control.make_glossary_stop_callback))[:2] == ["is_stop_requested", "is_graceful"]
    assert callable(stop_control.kill_helper_subprocesses)
    tracker = stop_control.StopClickTracker()
    assert tracker.register(now=100.0) is False
    assert tracker.register(now=100.3) is True, "two clicks within 1 s are a force stop (desktop 1.0 s window)"
    tracker = stop_control.StopClickTracker()
    tracker.register(now=100.0)
    assert tracker.register(now=101.5) is False


def test_job_runner_contract():
    _skip_unless_modules("job_runner")
    import job_runner
    import large_env

    assert isinstance(job_runner.JOB_LOCK, type(threading.RLock()))
    params = _params(job_runner.scoped_process_state)
    for name in ("argv", "env_updates", "restore_env", "clear_large_env", "capture_stdout", "lock"):
        assert name in params, name
    for name in ("log", "is_stop_requested", "is_graceful_stop", "emit", "ask"):
        assert hasattr(job_runner.JobHost, name), name
    assert list(_params(job_runner.ProgressWatcher))[:2] == ["output_dir_resolver", "host"]
    assert _params(job_runner.ProgressWatcher)["interval"].default == 2.0
    event = job_runner.JobEvent("progress", "job-1", 1.0, {"total": 3})
    assert (event.kind, event.job_id, event.ts, event.data) == ("progress", "job-1", 1.0, {"total": 3})
    # restore order of _process_text_file: argv, env, large_env store
    saved_env, saved_argv = dict(os.environ), list(sys.argv)
    try:
        with job_runner.scoped_process_state(["trace.py", "x"], env_updates={"PARITY_TRACE_SCOPED": "1"},
                                             lock=job_runner.JOB_LOCK):
            assert sys.argv == ["trace.py", "x"] and os.environ["PARITY_TRACE_SCOPED"] == "1"
            os.environ["PARITY_TRACE_LEAK"] = "1"
            large_env.set_env("PARITY_TRACE_LARGE", "x" * 40_000)
        assert "PARITY_TRACE_SCOPED" not in os.environ and "PARITY_TRACE_LEAK" not in os.environ
        assert sys.argv == saved_argv and "PARITY_TRACE_LARGE" not in large_env._store
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        sys.argv = saved_argv
        large_env.clear_store()


def test_translation_pipeline_contract():
    _skip_unless_modules("translation_pipeline")
    import translation_pipeline as tp

    mixin = tp.TranslationPipelineMixin
    assert _params(mixin._prepare_translation_run)["files"].default is None
    assert "request" in _params(mixin._translation_worker)
    for name in ("run_translation_direct", "_prepare_multipass_qa_refinement_run",
                 "_collect_translation_qa_failures", "_clear_translation_run_overrides"):
        assert hasattr(mixin, name), name
    assert hasattr(tp.GlossaryPipelineMixin, "run_glossary_extraction_direct")


def test_text_jobs_and_input_preparation_contract():
    _skip_unless_modules("text_jobs", "input_preparation")
    import input_preparation
    import text_jobs

    for name in ("_process_text_file", "_extract_glossary_from_text_file", "_run_epub_compile",
                 "_run_pdf_compile", "_run_parallel_metadata_files"):
        assert hasattr(text_jobs.TextJobsMixin, name), name
    params = _params(input_preparation.resolve_input_to_epub)
    assert list(params)[0] == "path"
    for name in ("conversion_dir", "should_stop", "log", "set_active"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert hasattr(input_preparation, "InputPreparationMixin")


def test_headless_owner_has_the_u3_pipeline_bases():
    _skip_unless_modules("translation_pipeline", "text_jobs", "input_preparation")
    import headless_owner

    mro = headless_owner.HeadlessOwner.__mro__
    for cls in th.session().pipeline_mixins():  # no oracle needed
        assert cls in mro, f"HeadlessOwner lacks {cls.__module__}.{cls.__name__} as a base"
    assert hasattr(headless_owner.HeadlessOwner, "auto_load_glossary_for_file"), (
        "auto_load_glossary_for_file is in OWNER_CONTRACT (glossary-mode handler with one EPUB selected)")


#: GUI-only names the desktop records (tier T recorders); HeadlessOwner may lack them
_GUI_ONLY = freeze_legacy.TG_RECORDED_METHODS | th.TRACE_RECORDED_METHODS


def _self_calls(fn) -> set:
    import textwrap

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        # a multi-line string at column 0 defeats dedent: parse the method inside a class body
        tree = ast.parse("class _Probe:\n" + source)
    fn_node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    # a class defined inside the method (a nested progress manager): its ``self`` is not the owner
    nested = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.ClassDef):
            nested.update(ast.walk(node))
    # ``hasattr(self, 'x')`` probes: the call is optional (desktop-only GUI helpers)
    probed = {node.args[1].value for node in ast.walk(fn_node)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr"
              and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)}
    out = set()
    for node in ast.walk(fn_node):
        if node in nested:
            continue
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                and node.func.attr not in probed):
            out.add(node.func.attr)
    return out


def test_headless_owner_provides_everything_the_shared_worker_calls():
    """Every ``self.<method>()`` reachable from the mobile entry points exists on HeadlessOwner
    (desktop-only GUI recorders excepted): a missing one is an AttributeError the worker's
    broad ``except`` turns into a silently skipped step on the phone."""
    _skip_unless_modules("translation_pipeline", "text_jobs", "input_preparation")
    import headless_owner

    ho = headless_owner.HeadlessOwner
    mixins = th.session().pipeline_mixins()
    cls = type("_Probe", tuple(c for c in mixins if c not in ho.__mro__) + (ho,), {})
    seen, queue, missing = set(), ["_prepare_translation_run", "_translation_worker",
                                   "run_glossary_extraction_direct"], set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        fn = inspect.getattr_static(cls, name, None)
        if fn is None:
            if name not in _GUI_ONLY:
                missing.add(name)
            continue
        fn = getattr(fn, "__func__", fn)
        if callable(fn):
            queue.extend(_self_calls(fn) - seen)
    assert not missing, f"HeadlessOwner (+U3 mixins) lacks methods the shared pipeline calls: {sorted(missing)}"
