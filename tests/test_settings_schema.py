"""Settings schema (src/settings_schema.py + generated src/settings_schema_data.py).

* drift: re-running src/mobile/tools/schema_extract.py reproduces the committed data;
* every save_config settings_map key has a spec, in source order;
* tier C: settings_schema.coerce == the desktop settings_map lambda on fuzz inputs
  (the lambdas are compiled from the current source, wherever the shared-core move put
  settings_map: translator_gui.save_config or settings_persistence);
* sections cover every key exactly once; search; platform availability reasons;
* recorded default discrepancies; fresh-install defaults vs the U0 oracle golden
  (skipped when tests/parity/golden is not captured on this machine);
* import hygiene (no PySide6 / translator_gui, Python 3.10 syntax).

Run: python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_settings_schema.py
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
import os
import random
import subprocess
import sys
import textwrap
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = SRC / "mobile" / "tools"
TESTS = ROOT / "tests"
for _path in (str(SRC), str(TESTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import settings_schema as ss  # noqa: E402
import settings_schema_data as ssd  # noqa: E402


def _load_tool():
    spec = importlib.util.spec_from_file_location("schema_extract_under_test", TOOLS / "schema_extract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


se = _load_tool()


# --------------------------------------------------------------------------- drift
@pytest.fixture(scope="module")
def generated():
    return se.generate(SRC)


def test_generated_data_is_fresh(generated):
    committed = se.read_committed(SRC)
    assert committed is not None
    assert committed == generated, "src/settings_schema_data.py is stale: run python src/mobile/tools/schema_extract.py"


# --------------------------------------------------------------------------- settings_map
def _owner_modules():
    out = []
    for name in se.OWNER_MODULES:
        path = SRC / name
        if path.is_file():
            source = se.read_source(path)
            out.append((name, source, ast.parse(source)))
    return out


def _find_settings_map():
    for name, source, tree in _owner_modules():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "settings_map"):
                return name, source, node.value
    raise AssertionError("settings_map not found")


def test_every_settings_map_key_has_a_spec():
    _name, _source, lst = _find_settings_map()
    keys = [ast.literal_eval(elt.elts[0]) for elt in lst.elts]
    assert tuple(keys) == ssd.SETTINGS_MAP_ORDER
    for key in keys:
        item = ss.spec(key)
        assert item.converter is not None and item.converter[0] != "unknown", key
        assert "settings_map" in item.origins
    assert ssd.UNKNOWN_CONVERTERS == {}


# --------------------------------------------------------------------------- tier C: coerce
def _function_source(source, node):
    return textwrap.dedent(ast.get_source_segment(source, node))


def _lambda_namespace():
    """safe_int/safe_float, module constants and named converters, from the desktop source."""
    ns = {"Decimal": Decimal, "InvalidOperation": InvalidOperation}
    found = {}
    for _name, source, tree in _owner_modules():
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    ns.setdefault(node.targets[0].id, ast.literal_eval(node.value))
                except Exception:
                    pass
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("safe_int", "safe_float", "_format_plain_decimal_setting"):
                found.setdefault(node.name, _function_source(source, node))
    for name, text in found.items():
        exec(compile(text, f"<desktop {name}>", "exec"), ns)
    return ns


def _desktop_converters():
    _name, source, lst = _find_settings_map()
    ns = _lambda_namespace()
    converters = {}
    for elt in lst.elts:                                   # last occurrence wins, like the loop
        key = ast.literal_eval(elt.elts[0])
        expr = ast.Expression(body=elt.elts[3])
        ast.fix_missing_locations(expr)
        converters[key] = eval(compile(expr, f"<settings_map {key}>", "eval"), ns)
    return converters


BASE_INPUTS = [
    None, True, False, 0, 1, -1, 2, 5, 10, 11, -2, 0.0, 1.5, -2.7, 0.3, 1e20, -1e20,
    float("nan"), float("inf"), "", " ", "0", "1", "-1", "5", "10", "1.5", "-0.5", " 7 ",
    "abc", "True", "false", "FULL", " Partial.B ", "html", "Auto", "1e3", "1e-5", "0.0001",
    "٣", "--1", "+3", "0x10", [], [1, 2], {}, {"a": 1}, (1, 2), "None",
]


def _inputs_for(item, rng):
    values = list(BASE_INPUTS)
    conv = item.converter
    if conv and conv[0] == "choice":
        for choice in conv[1]:
            values += [choice, choice.upper(), f"  {choice} "]
    if conv and conv[0] in ("safe_int", "safe_float"):
        for bound in (conv[1], conv[2], conv[3]):
            if bound is not None:
                values += [bound, bound - 1, bound + 1, str(bound), str(bound + 1)]
    for _ in range(40):
        kind = rng.randrange(4)
        if kind == 0:
            values.append(rng.randint(-10 ** 6, 10 ** 6))
        elif kind == 1:
            values.append(rng.uniform(-1000, 1000))
        elif kind == 2:
            values.append(str(rng.randint(-500, 500)))
        else:
            values.append("".join(rng.choice("ab-.1 0") for _ in range(rng.randrange(6))))
    return values


def _outcome(func, value):
    try:
        result = func(copy.deepcopy(value))
    except Exception as exc:                 # same exception type is part of the contract
        return ("raise", type(exc).__name__)
    if isinstance(result, float) and math.isnan(result):
        return ("ok", "float", "nan")
    return ("ok", type(result).__name__, repr(result))


def test_coerce_matches_desktop_lambdas():
    converters = _desktop_converters()
    rng = random.Random(20261005)
    checked = 0
    call_ok = True
    try:
        ss.apply_converter(("call", "_format_plain_decimal_setting"), "1")
    except LookupError:
        call_ok = False                      # run_env not importable yet (pre-move layout)
    mismatches = []
    for key, desktop in sorted(converters.items()):
        item = ss.spec(key)
        if item.converter[0] == "call" and not call_ok:
            continue
        for value in _inputs_for(item, rng):
            expected = _outcome(desktop, value)
            actual = _outcome(lambda v, k=key: ss.coerce(k, v), value)
            checked += 1
            if expected != actual:
                mismatches.append((key, value, expected, actual))
    assert not mismatches[:20], mismatches[:20]
    assert checked > 20000


def test_coerce_non_settings_map_keys():
    # bool like save_config._config_bool
    assert ss.coerce("qa_scanner_settings.check_repetition", "yes") is True
    assert ss.coerce("qa_scanner_settings.check_repetition", "0") is False
    assert ss.coerce("qa_scanner_settings.check_repetition", None) is True      # default
    # int falls back to the default on junk
    assert ss.coerce("qa_scanner_settings.min_file_length", "abc") == 0
    assert ss.coerce("qa_scanner_settings.min_file_length", "12") == 12


# --------------------------------------------------------------------------- sections
def test_sections_cover_every_key_exactly_once():
    seen = {}
    for section in ss.sections(include_empty=True):
        for key in section.keys:
            assert key not in seen, f"{key} in {seen[key]} and {section.id}"
            seen[key] = section.id
            assert ss.spec(key).section == section.id
    assert set(seen) == set(ss.keys())
    # catch-all sections exist only with an explanation
    for section_id in ("other.stored", "internal.state"):
        assert ss.section(section_id).description
    assert len(ss.section("other.stored").keys) < 40, "place new settings in a real section"


def test_section_order_mirrors_desktop():
    ids = [s.id for s in ss.sections(include_empty=True)]
    other = [i for i in ids if i.startswith("other.")]
    assert other[:4] == ["other.meta_data", "other.context", "other.response", "other.processing"]
    assert ids.index("other.anti_duplicate.core") < ids.index("other.anti_duplicate.logit_bias") < ids.index("other.endpoints")
    assert ids.index("other.output") < ids.index("other.danger")
    for section_id in ("glossary.general", "glossary.balanced_full", "glossary.minimal", "glossary.refinement",
                       "qa.settings", "manga.settings", "library.reader", "direct_text.settings", "main.run"):
        assert section_id in ids


@pytest.mark.parametrize("key, section_id", [
    ("delay", "main.run"),
    ("model", "main.model"),
    ("auto_glossary_mode", "main.run"),
    ("top_p", "other.anti_duplicate.core"),
    ("repetition_penalty", "other.anti_duplicate.advanced"),
    ("custom_stop_sequences", "other.anti_duplicate.stop"),
    ("logit_bias_enabled", "other.anti_duplicate.logit_bias"),
    ("enable_pdf_output", "other.output"),
    ("qa_scanner_settings.check_repetition", "qa.settings"),
    ("ai_hunter_config.thresholds.text", "qa.ai_hunter"),
    ("epub_reader_theme", "library.reader"),
])
def test_key_sections(key, section_id):
    assert ss.spec(key).section == section_id


# --------------------------------------------------------------------------- search
def test_search():
    assert ss.search("") == []
    assert ss.search("delay")[0].key == "delay"
    assert "delay" in [s.key for s in ss.search("API call delay")]
    assert ss.search("GLOSSARY_MAX_SENTENCES")[0].key == "glossary_max_sentences"   # env name
    keys = [s.key for s in ss.search("temperature")]
    assert "translation_temperature" in keys[:5]
    assert all("tor" in (s.key + s.label + s.tooltip).lower() for s in ss.search("tor proxy"))
    assert ss.search("no such setting at all") == []
    assert len(ss.search("glossary", limit=3)) == 3


# --------------------------------------------------------------------------- availability
@pytest.mark.parametrize("key, needle", [
    ("tor_proxy_enabled", "Tor"),
    ("auto_dpi_scale", "DPI"),
    ("authza_use_general_api", "AuthZA"),
    ("authnd_token_concurrency", "U9"),
    ("gemini_free_adaptive_split", "U9"),
    ("model_mousewheel_locked", "touch"),
    ("enable_gui_yield", "GUI"),
    ("multi_api_key_tree_font_size", "Key-tree"),
    # worker-process settings that mobile_runtime.processes_available() turns into no-ops
    ("pdf_extraction_workers", "single-process"),
    ("pdf_async_page_threshold", "single-process"),
    ("manga_settings.inpainting.disable_worker_process", "in-process"),
])
def test_unavailable_on_mobile_with_reason(key, needle):
    available, reason = ss.is_available(key, "mobile")
    assert available is False and needle in reason
    assert ss.is_available(key, "desktop") == (True, "")
    item = ss.spec(key)
    assert "mobile" not in item.platforms and "desktop" in item.platforms
    assert item.section                                      # still placed: never hidden


def test_available_settings():
    for key in ("model", "delay", "contextual", "qa_scanner_settings.check_repetition"):
        assert ss.is_available(key) == (True, "")
    with pytest.raises(KeyError):
        ss.is_available("definitely_not_a_setting")


# --------------------------------------------------------------------------- defaults
def test_discrepancies_recorded_not_fixed():
    gms = ss.spec("glossary_max_sentences")
    assert ss.effective_default("glossary_max_sentences") == 200
    text = " ".join(gms.discrepancies)
    assert "=10" in text and "=200" in text
    assert any(b.name == "GLOSSARY_MAX_SENTENCES" and b.site == "startup" and b.call_site_default == 10 for b in gms.env)
    agm = ss.spec("auto_glossary_mode")
    assert ss.effective_default("auto_glossary_mode") == "off"     # U0 oracle
    assert agm.init_default == "balanced" and agm.default_source == "oracle"
    assert any("balanced" in d and "oracle" in d for d in agm.discrepancies)


def test_effective_defaults():
    assert ss.effective_default("model") == "authgpt/gpt-6-luna"
    assert ss.effective_default("delay") == 5.0
    assert ss.effective_default("api_queue") == 4
    assert ss.effective_default("vertex_ai_location") == "global"
    import prompt_defaults
    assert ss.effective_default("vision_ocr_prompt") == prompt_defaults.DEFAULT_VISION_OCR_PROMPT
    assert ss.effective_default("prompt_profiles") is None          # computed at runtime
    value = ss.effective_default("custom_glossary_fields")
    value.append("mutated")
    assert ss.effective_default("custom_glossary_fields") == ["description"]   # copies


def _golden_fresh_install():
    golden_root = TESTS / "parity" / "golden"
    if not golden_root.is_dir():
        return None
    candidates = []
    latest = TESTS / "parity" / "legacy" / "LATEST.txt"
    if latest.is_file():
        candidates.append(latest.read_text(encoding="utf-8").strip()[:12])
    candidates += [p.name for p in sorted(golden_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)]
    for sha in candidates:
        if (golden_root / sha / "fresh_install.py").is_file():
            from parity import capture_golden
            return capture_golden.load_golden(sha, "fresh_install")
    return None


def _norm(value):
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("true", "false"):
            return float(text.lower() == "true")
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, tuple):
        return [_norm(v) for v in value]
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


NESTED_PARENTS = ("qa_scanner_settings", "manga_settings", "ai_hunter_config")


def _flatten(config, prefix=""):
    """Config keys as schema keys: nested settings dicts become dotted keys."""
    all_keys = ss.keys()
    for key, value in config.items():
        path = f"{prefix}{key}"
        nested = path in NESTED_PARENTS or (
            prefix and not ss.has_spec(path) and any(k.startswith(path + ".") for k in all_keys))
        if isinstance(value, dict) and nested:
            yield from _flatten(value, path + ".")
        else:
            yield path, value


def test_fresh_install_defaults_match_desktop_oracle():
    golden = _golden_fresh_install()
    if golden is None:
        pytest.skip("tests/parity golden fresh_install not captured on this machine")
    config = golden["entries"]["boot"]["final"]["config"]
    known = set(ss.FRESH_INSTALL_NOTES) | set(ss.COMPUTED_DEFAULTS)
    unknown_keys, mismatches = [], []
    for key, value in _flatten(config):
        if not ss.has_spec(key):
            unknown_keys.append(key)
            continue
        if key in known:
            continue
        if _norm(ss.effective_default(key)) != _norm(value):
            mismatches.append((key, ss.effective_default(key), value))
    assert not unknown_keys, f"fresh-install config keys without a spec: {unknown_keys}"
    assert not mismatches, mismatches[:10]


# --------------------------------------------------------------------------- contract / hygiene
def test_spec_and_section_contract():
    fields = set(ss.SettingSpec.__dataclass_fields__)
    for name in ("key", "type", "default", "save_default", "converter", "var_names", "widget_sources", "env",
                 "section", "label", "tooltip", "choices", "minimum", "maximum", "visible_if", "locked_if",
                 "platforms", "discrepancies"):
        assert name in fields
    assert ss.Section._fields[:4] == ("id", "title", "keys", "group")
    item = ss.spec("delay")
    with pytest.raises(Exception):
        item.default = 1                                     # frozen
    assert item.label == "API call delay (s)"
    assert {s.type for s in ss.all_specs()} <= {"bool", "int", "float", "str", "choice", "list", "dict", "json", "path", "secret"}
    assert ss.spec("api_key").type == "secret"


def test_import_hygiene():
    for name in ("settings_schema.py", "settings_schema_data.py"):
        ast.parse((SRC / name).read_text(encoding="utf-8"), feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, sys.argv[1]);"
        "import settings_schema as ss; ss.all_specs(); ss.sections(); ss.search('delay');"
        "bad = [m for m in ('translator_gui', 'dpi_setup', 'PySide6.QtWidgets') if m in sys.modules and sys.modules[m] is not None];"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert result.returncode == 0, result.stderr[-2000:]
