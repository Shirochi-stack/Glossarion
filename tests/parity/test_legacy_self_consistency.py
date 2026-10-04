"""Legacy-vs-legacy self-consistency of the desktop parity oracle.

* The frozen methods are byte-for-byte copies of ``git show <sha>:src/...``.
* Capturing every scenario twice yields identical records (no hidden
  nondeterminism from time, pids, cpu count, temp paths or env leakage).
* A fresh capture equals the goldens written by capture_golden.py.
* Every entry of every scenario ran without an escaped exception and reached
  its stubbed backend entry point.
* Desktop facts the mobile plan depends on (recorded, not "fixed").

Run (repository root)::

    python tests/parity/freeze_legacy.py
    python tests/parity/capture_golden.py
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests/parity/test_legacy_self_consistency.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import capture_golden as cg  # noqa: E402
from parity import fakes, freeze_legacy, scenarios  # noqa: E402

NAMES = scenarios.SCENARIO_NAMES
BACKEND_ENTRIES = {
    "process_text_file": "translation_main",
    "extract_glossary": "glossary_main",
    "epub_compile": "fallback_compile_epub",
    "pdf_compile": "pdf_workspace_compiler.compile_pdf_workspace",
}


@pytest.fixture(scope="module")
def legacy():
    try:
        bundle = freeze_legacy.load_legacy()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return bundle, fakes.make_legacy_owner_factory(bundle)


@pytest.fixture(scope="module")
def captures(legacy):
    _bundle, factory = legacy
    first = {name: cg.capture(factory, scenarios.get(name)) for name in NAMES}
    second = {name: cg.capture(factory, scenarios.get(name)) for name in NAMES}
    return first, second


def test_frozen_methods_are_verbatim_copies(legacy):
    bundle, _factory = legacy
    manifest = bundle.manifest
    original = freeze_legacy.git_show_text(manifest["sha"], manifest["source_file"]).split("\n")
    frozen_text = bundle.path.read_text(encoding="utf-8")
    frozen_lines = frozen_text.split("\n")
    tree = ast.parse(frozen_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LegacyMethods")
    checked = 0
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        info = manifest["frozen_methods"].get(node.name)
        if info is None:
            continue  # synthetic legacy_* blocks are checked span by span below
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        frozen = frozen_lines[start - 1:node.end_lineno]
        a, b = info["lines"]
        assert frozen == original[a - 1:b], f"{node.name} differs from git show"
        checked += 1
    assert checked == len(manifest["frozen_methods"])
    blocks_text = "\n".join(frozen_lines)
    for name, spans in manifest["synthetic_blocks"].items():
        for a, b in spans:
            segment = "\n".join(original[a - 1:b])
            assert segment in blocks_text, f"{name} lines {a}-{b} not copied verbatim"


@pytest.mark.parametrize("name", NAMES)
def test_capture_twice_is_identical(captures, name):
    first, second = captures
    problems = cg.diff(first[name], second[name])
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", NAMES)
def test_capture_matches_golden(legacy, captures, name):
    bundle, _factory = legacy
    try:
        golden = cg.load_golden(bundle.sha, name)
    except FileNotFoundError:
        pytest.skip("goldens not captured; run tests/parity/capture_golden.py")
    first, _second = captures
    problems = cg.diff(golden, cg.roundtrip(first[name]))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", NAMES)
def test_every_entry_ran_cleanly(captures, name):
    first, _second = captures
    result = first[name]
    assert list(result["entries"]) == list(scenarios.get(name)["entries"])
    for entry, record in result["entries"].items():
        assert "boot_error" not in record, (entry, record.get("boot_error"))
        assert not record.get("exception"), (entry, record["exception"])
        if entry in BACKEND_ENTRIES:
            calls = record["backend"]
            assert [c["entry"] for c in calls] == [BACKEND_ENTRIES[entry]], (entry, calls)
        if entry == "run_helpers":
            failed = {k: v for k, v in record["result"].items() if "exception" in v}
            assert not failed, failed


def _env_at(record, backend_entry):
    call = next(c for c in record["backend"] if c["entry"] == backend_entry)
    return call["env"]["set"], call["argv"]


def test_fresh_install_desktop_exports(captures):
    """What the desktop really exports on a fresh install (no config.json).

    MODEL comes from _create_model_section's default ('authgpt/gpt-6-luna').
    AUTO_GLOSSARY_MODE is 'off', NOT the _init_variables default 'balanced':
    _create_settings_section fires the glossary shortcut handler once at startup
    with the combo index derived from config (None -> 'off'), which sets
    auto_glossary_mode_var='off' and runs save_config() (verified against a real
    offscreen TranslatorGUI). A first-run dialog then asks the user for a mode.
    """
    first, _second = captures
    entries = first["fresh_install"]["entries"]
    env = entries["translation_env"]["result"]
    assert env["MODEL"] == "authgpt/gpt-6-luna"
    assert env["AUTO_GLOSSARY_MODE"] == "off"
    assert env["API_KEY"] == ""

    main_env, argv = _env_at(entries["process_text_file"], "translation_main")
    assert main_env["MODEL"] == "authgpt/gpt-6-luna"
    assert main_env["AUTO_GLOSSARY_MODE"] == "off"
    assert argv[0] == "TransateKRtoEN.py" and argv[1].endswith("Parity Novel.epub")

    phases = {p["phase"]: p for p in entries["boot"]["phases"]}
    assert phases["init_variables"]["attrs"]["set"]["auto_glossary_mode_var"] == "balanced"
    assert phases["gui_state"]["attrs"]["set"]["model_var"] == "authgpt/gpt-6-luna"
    assert phases["gui_handlers"]["attrs"]["set"]["auto_glossary_mode_var"] == "off"
    assert phases["gui_handlers"]["config"]["set"]["auto_glossary_mode"] == "off"
    # startup save_config persisted the full settings_map
    assert len(entries["boot"]["final"]["config"]) > 400


def test_known_desktop_quirks_are_preserved(captures):
    """Bugs/defaults found while freezing are recorded, never fixed in the oracle."""
    first, _second = captures
    env = first["fresh_install"]["entries"]["translation_env"]["result"]
    # 'PRESERVE_ORIGINAL_FORMAT' 'OPTIMIZE_FOR_OCR' implicit string concatenation (36557-36558)
    assert "PRESERVE_ORIGINAL_FORMATOPTIMIZE_FOR_OCR" in env
    assert "PRESERVE_ORIGINAL_FORMAT" not in env
    assert "OPTIMIZE_FOR_OCR" not in env
    # Vision OCR batch size > 0 forces BATCH_SIZE=1 in vision mode
    vision = first["vision_balanced"]["entries"]["translation_env"]["result"]
    assert vision["OUTPUT_MODE"] == "vision" and vision["BATCH_SIZE"] == "1"
    # formatting comes from where the value was last written: the startup save_config
    # stores safe_float() values (10.0) while the API delay is the raw widget text
    assert env["CONNECT_TIMEOUT"] == "10.0"
    assert env["READ_TIMEOUT"] == "180.0"
    assert env["SEND_INTERVAL_SECONDS"] == "5"
