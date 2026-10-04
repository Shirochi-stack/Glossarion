#!/usr/bin/env python3
"""Capture desktop-parity goldens: ``capture(owner_factory, scenario)``.

Each scenario entry runs in its own ``normalize.CaptureContext`` (fresh
sandbox, scrubbed env, deterministic process state) on a freshly booted owner
built by ``owner_factory(scenario, ctx)``. The same ``capture`` drives the
frozen legacy owner (``fakes.make_legacy_owner_factory``) now and the
extracted mixins / HeadlessOwner in U2 (tiers G and D compare their records).

Goldens are Python literal files (``*.json`` is gitignored repo-wide and the
rule is kept): ``tests/parity/golden/<sha12>/<scenario>.py`` with
``GOLDEN = {...}`` plus ``INDEX.py``.

Usage (repository root)::

    python tests/parity/capture_golden.py                 # all scenarios @ frozen SHA
    python tests/parity/capture_golden.py --scenario fresh_install --check
"""

from __future__ import annotations

import argparse
import ast
import pprint
import sys
import time as _time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import fakes, freeze_legacy, normalize, scenarios  # noqa: E402

PARITY_DIR = Path(__file__).resolve().parent
GOLDEN_ROOT = PARITY_DIR / "golden"
GOLDEN_FORMAT = 1


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def _input_path(scenario, ctx) -> str:
    return ctx.sandbox.path(scenario["input"])


def _api_key(owner) -> str:
    return owner.api_key_entry.text()


def entry_translation_env(owner, scenario, ctx):
    return owner._get_environment_variables(_input_path(scenario, ctx), _api_key(owner))


def entry_glossary_env_mappings(owner, scenario, ctx):
    return owner._glossary_env_mappings()


#: Small run_env helpers evaluated in this order on one owner (some mutate config).
RUN_HELPERS = (
    ("resolve_glossary_for_env", lambda o, p: o._resolve_glossary_for_env(p)),
    ("resolve_translation_output_dir", lambda o, p: o._resolve_translation_output_dir(p)),
    ("get_output_base_dir", lambda o, p: o._get_output_base_dir(p)),
    ("output_side_glossary_backup_dir", lambda o, p: o._output_side_glossary_backup_dir_for_source(p)),
    ("subtitle_zip_output_info", lambda o, p: o._subtitle_zip_output_info(p)),
    ("get_output_mode", lambda o, p: o._get_output_mode()),
    ("active_translation_output_mode", lambda o, p: o._active_translation_output_mode()),
    ("is_generative_output_mode", lambda o, p: o._is_generative_output_mode()),
    ("allowed_image_output_mode", lambda o, p: o._get_allowed_image_output_mode()),
    ("allowed_video_output_mode", lambda o, p: o._get_allowed_video_output_mode()),
    ("model_is_image_gen", lambda o, p: o._model_is_image_gen(o.model_var)),
    ("model_is_video_gen", lambda o, p: o._model_is_video_gen(o.model_var)),
    ("current_auto_glossary_mode", lambda o, p: o._current_auto_glossary_mode()),
    ("context_mode_from_flags", lambda o, p: o._context_mode_from_flags()),
    ("translation_batching_mode", lambda o, p: o._translation_batching_mode_for_env()),
    ("glossary_batching_mode", lambda o, p: o._glossary_batching_mode_for_env()),
    ("glossary_request_env", lambda o, p: o._current_glossary_request_env()),
    ("glossary_request_env_forced", lambda o, p: o._current_glossary_request_env(force_balanced_request_merging=True)),
    ("glossary_contextual", lambda o, p: o._glossary_contextual_env_value()),
    ("glossary_skip_title_header_only", lambda o, p: o._glossary_skip_title_header_only_env_value()),
    ("glossary_add_minimal_pass", lambda o, p: o._glossary_add_minimal_pass_env_value()),
    ("glossary_match_engine", lambda o, p: o._glossary_match_engine_env_value()),
    ("glossary_cjk_script_filter", lambda o, p: o._current_glossary_cjk_script_filter_enabled()),
    ("strict_matching_env", lambda o, p: o._strict_matching_env_dict()),
    ("unified_glossary_env", lambda o, p: o._unified_glossary_env_dict()),
    ("custom_prefix_routes_json", lambda o, p: o._custom_prefix_routes_env_json()),
    ("ollama_settings_json", lambda o, p: o._ollama_settings_env_json()),
    ("qa_scanner_settings_json", lambda o, p: o._get_qa_scanner_settings_json()),
    ("scan_phase_mode", lambda o, p: o._get_scan_phase_mode()),
    ("multipass_refinement_mode", lambda o, p: o._get_multipass_refinement_mode()),
    ("resolve_max_retries", lambda o, p: o._resolve_max_retries()),
    ("resolve_max_retry_tokens", lambda o, p: o._resolve_max_retry_tokens(o.max_output_tokens)),
    ("compression_chunk_budget", lambda o, p: o._compression_chunk_budget()),
    ("live_chapter_range_settings", lambda o, p: o._live_chapter_range_settings()),
    ("anti_duplicate_settings_line", lambda o, p: o._format_translation_anti_duplicate_settings()),
)


def entry_run_helpers(owner, scenario, ctx):
    path = _input_path(scenario, ctx)
    out = {}
    for name, fn in RUN_HELPERS:
        try:
            out[name] = {"value": fn(owner, path)}
        except Exception as exc:
            out[name] = {"exception": type(exc).__name__, "message": str(exc)}
    return out


def entry_multipass_runtime_env(owner, scenario, ctx):
    return owner._export_multipass_runtime_env()


def entry_chapter_range_runtime_env(owner, scenario, ctx):
    return owner._export_chapter_range_runtime_env()


def entry_process_text_file(owner, scenario, ctx):
    """Desktop per-file translation path up to the stubbed TransateKRtoEN.main."""
    return owner._process_text_file(_input_path(scenario, ctx))


def entry_extract_glossary(owner, scenario, ctx):
    """Glossary extraction path up to the stubbed extract_glossary_from_epub.main."""
    return owner._extract_glossary_from_text_file(_input_path(scenario, ctx))


def _workspace(owner, scenario, ctx) -> str:
    stem = Path(scenario["input"]).stem
    folder = Path(ctx.sandbox.path("outputs")) / stem
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def entry_epub_compile(owner, scenario, ctx):
    owner.epub_folder = _workspace(owner, scenario, ctx)
    return owner.run_epub_converter_direct()


def entry_pdf_compile(owner, scenario, ctx):
    owner.pdf_folder = _workspace(owner, scenario, ctx)
    return owner.run_pdf_converter_direct()


def entry_metadata_only_env(owner, scenario, ctx):
    return owner._metadata_only_environment_for_file(_input_path(scenario, ctx), _api_key(owner))


def entry_direct_text_env(owner, scenario, ctx):
    return owner._apply_direct_text_runtime_environment()


def entry_forced_streaming_env(owner, scenario, ctx):
    return owner._apply_forced_streaming_environment()


ENTRY_FUNCS = {
    "translation_env": entry_translation_env,
    "glossary_env_mappings": entry_glossary_env_mappings,
    "run_helpers": entry_run_helpers,
    "multipass_runtime_env": entry_multipass_runtime_env,
    "chapter_range_runtime_env": entry_chapter_range_runtime_env,
    "process_text_file": entry_process_text_file,
    "extract_glossary": entry_extract_glossary,
    "epub_compile": entry_epub_compile,
    "pdf_compile": entry_pdf_compile,
    "metadata_only_env": entry_metadata_only_env,
    "direct_text_env": entry_direct_text_env,
    "forced_streaming_env": entry_forced_streaming_env,
}


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _exception_record(exc, ctx) -> dict:
    return {"type": type(exc).__name__, "message": ctx.norm(str(exc))}


def _apply_run_attrs(owner, scenario, ctx):
    run_attrs = scenario.get("run_attrs") or {}
    if callable(run_attrs):
        run_attrs = run_attrs(ctx.sandbox)
    for name, value in ctx.sandbox.resolve(run_attrs).items():
        setattr(owner, name, value)


def run_entry(owner_factory, scenario: dict, entry: str, ctx) -> dict:
    try:
        owner = owner_factory(scenario, ctx)
    except Exception as exc:
        return {"boot_error": _exception_record(exc, ctx), "stdout": ctx.take_stdout(),
                "phases": [{k: v for k, v in cp.items() if k != "_state"} for cp in ctx.checkpoints]}
    if entry == "boot":
        final = ctx.state(owner)
        return {
            "phases": [{k: v for k, v in cp.items() if k != "_state"} for cp in ctx.checkpoints],
            "final": {
                "env": ctx.norm(normalize.env_delta(ctx.baseline_env, final["env"])),
                "config": final["config"],
                "attrs": final["attrs"],
            },
        }
    ctx.take_stdout()
    ctx.recorder.clear()
    ctx.backend_calls.clear()
    _apply_run_attrs(owner, scenario, ctx)
    before = ctx.state(owner)
    exception = None
    result = None
    try:
        result = ENTRY_FUNCS[entry](owner, scenario, ctx)
    except Exception as exc:
        exception = _exception_record(exc, ctx)
    after = ctx.state(owner)
    return {
        "result": ctx.norm(result),
        "exception": exception,
        "env": ctx.norm(normalize.env_delta(before["env"], after["env"])),
        "config": normalize.dict_delta(before["config"], after["config"]),
        "attrs": normalize.dict_delta(before["attrs"], after["attrs"]),
        "calls": ctx.norm(list(ctx.recorder.calls)),
        "backend": ctx.norm(list(ctx.backend_calls)),
        "stdout": ctx.take_stdout(),
    }


def capture(owner_factory, scenario: dict, *, entries=None) -> dict:
    """Run every entry of *scenario* against fresh owners from *owner_factory*."""
    out = {"scenario": scenario["name"], "entries": {}}
    for entry in entries or scenario["entries"]:
        if entry != "boot" and entry not in ENTRY_FUNCS:
            raise KeyError(f"unknown entry {entry!r}")
        with normalize.CaptureContext(scenario, entry, record_checkpoints=(entry == "boot")) as ctx:
            out["entries"][entry] = run_entry(owner_factory, scenario, entry, ctx)
    return out


def summarize(result: dict) -> dict:
    """{entry: 'ok' | 'exception:<Type>' | 'boot_error:<Type>'} for reports."""
    out = {}
    for entry, rec in result["entries"].items():
        if "boot_error" in rec:
            out[entry] = f"boot_error:{rec['boot_error']['type']}"
        elif rec.get("exception"):
            out[entry] = f"exception:{rec['exception']['type']}"
        else:
            helper_errors = []
            if entry == "run_helpers" and isinstance(rec.get("result"), dict):
                helper_errors = sorted(k for k, v in rec["result"].items() if "exception" in v)
            out[entry] = "ok" if not helper_errors else f"ok(helper exceptions: {', '.join(helper_errors)})"
    return out


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


def golden_dir(sha: str) -> Path:
    return GOLDEN_ROOT / sha[:12]


def _format_literal(name: str, value) -> str:
    return f"{name} = " + pprint.pformat(value, width=120, sort_dicts=False) + "\n"


_INTERN_MIN_LEN = 160
_TOKEN_PREFIX = "<<str:"
_TOKEN_SUFFIX = ">>"


def intern_strings(value, table: dict):
    """Replace long strings (prompts, JSON blobs) by ``<<str:hash>>`` tokens stored once in *table*."""
    if isinstance(value, str):
        if len(value) < _INTERN_MIN_LEN:
            return value
        import hashlib

        key = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
        table.setdefault(key, value)
        return f"{_TOKEN_PREFIX}{key}{_TOKEN_SUFFIX}"
    if isinstance(value, dict):
        return {intern_strings(k, table): intern_strings(v, table) for k, v in value.items()}
    if isinstance(value, list):
        return [intern_strings(v, table) for v in value]
    if isinstance(value, tuple):
        return tuple(intern_strings(v, table) for v in value)
    return value


def expand_strings(value, table: dict):
    if isinstance(value, str):
        if value.startswith(_TOKEN_PREFIX) and value.endswith(_TOKEN_SUFFIX):
            return table[value[len(_TOKEN_PREFIX):-len(_TOKEN_SUFFIX)]]
        return value
    if isinstance(value, dict):
        return {expand_strings(k, table): expand_strings(v, table) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_strings(v, table) for v in value]
    if isinstance(value, tuple):
        return tuple(expand_strings(v, table) for v in value)
    return value


def write_golden(result: dict, sha: str, *, owner_kind: str) -> Path:
    path = golden_dir(sha) / f"{result['scenario']}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED by tests/parity/capture_golden.py - review before regenerating.\n"
        f"# BASE_SHA {sha}  owner {owner_kind}  format {GOLDEN_FORMAT}\n"
        "# Python literals (not JSON: *.json stays gitignored). Long strings are stored once in\n"
        "# STRINGS and referenced as '<<str:hash>>'. Load with capture_golden.load_golden().\n"
    )
    table: dict = {}
    interned = intern_strings(result, table)
    text = header + _format_literal("GOLDEN", interned) + "\n" + _format_literal("STRINGS", table)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_index(sha: str, summaries: dict, *, owner_kind: str, seconds: float) -> Path:
    path = golden_dir(sha) / "INDEX.py"
    meta = {
        "base_sha": sha,
        "owner_kind": owner_kind,
        "format": GOLDEN_FORMAT,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "capture_seconds": round(seconds, 1),
        "scenarios": summaries,
    }
    path.write_text(
        "# GENERATED by tests/parity/capture_golden.py\n" + _format_literal("INDEX", meta),
        encoding="utf-8", newline="\n",
    )
    return path


def _literals_from_file(path: Path, *names: str) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    found[t.id] = ast.literal_eval(node.value)
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{missing} not found in {path}")
    return [found[n] for n in names]


def _literal_from_file(path: Path, name: str):
    return _literals_from_file(path, name)[0]


def load_golden(sha: str, scenario: str) -> dict:
    golden, strings = _literals_from_file(golden_dir(sha) / f"{scenario}.py", "GOLDEN", "STRINGS")
    return expand_strings(golden, strings)


def load_index(sha: str) -> dict:
    return _literal_from_file(golden_dir(sha) / "INDEX.py", "INDEX")


def roundtrip(value):
    """Value as it reads back from a golden file (tuples/lists etc. as written)."""
    return ast.literal_eval(pprint.pformat(value, width=120, sort_dicts=False))


def diff(expected, actual, path="", out=None, limit=50) -> list:
    """Readable recursive differences (first *limit*)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in list(expected) + [k for k in actual if k not in expected]:
            if key not in actual:
                out.append(f"{path}/{key}: missing in actual")
            elif key not in expected:
                out.append(f"{path}/{key}: unexpected {str(actual[key])[:120]!r}")
            else:
                diff(expected[key], actual[key], f"{path}/{key}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)) and len(expected) == len(actual):
        for i, (a, b) in enumerate(zip(expected, actual)):
            diff(a, b, f"{path}[{i}]", out, limit)
    elif expected != actual:
        out.append(f"{path}: {str(expected)[:160]!r} != {str(actual)[:160]!r}")
    return out


def legacy_factory(sha: str | None = None):
    bundle = freeze_legacy.load_legacy(sha)
    return bundle, fakes.make_legacy_owner_factory(bundle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Capture legacy desktop-parity goldens")
    parser.add_argument("--sha", default=None, help="frozen legacy SHA (default: legacy/LATEST.txt)")
    parser.add_argument("--scenario", action="append", help="limit to these scenarios")
    parser.add_argument("--check", action="store_true", help="compare with existing goldens instead of writing")
    args = parser.parse_args(argv)
    bundle, factory = legacy_factory(args.sha)
    names = args.scenario or list(scenarios.SCENARIO_NAMES)
    started = _time.monotonic()
    summaries, failures = {}, 0
    for name in names:
        scenario = scenarios.get(name)
        t0 = _time.monotonic()
        result = capture(factory, scenario)
        summaries[name] = summarize(result)
        if args.check:
            golden = load_golden(bundle.sha, name)
            problems = diff(golden, roundtrip(result))
            status = "MATCH" if not problems else f"DIFF ({len(problems)})"
            failures += bool(problems)
            print(f"{name:34s} {status}  {_time.monotonic() - t0:5.1f}s")
            for line in problems[:20]:
                print(f"    {line}")
        else:
            path = write_golden(result, bundle.sha, owner_kind=factory.kind)
            print(f"{name:34s} {path.stat().st_size // 1024:6d} KB  {_time.monotonic() - t0:5.1f}s  "
                  + ", ".join(f"{k}={v}" for k, v in summaries[name].items() if v != "ok"))
    if not args.check:
        write_index(bundle.sha, summaries, owner_kind=factory.kind, seconds=_time.monotonic() - started)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
