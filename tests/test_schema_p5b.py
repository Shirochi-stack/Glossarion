"""U9 P5b: the desktop settings tables are built from the settings schema.

``settings_persistence.SettingsPersistenceMixin._apply_live_settings_to_config`` (save_config) and
``owner_state.ConfigStateMixin._init_variables`` held three literal tables: ``settings_map``,
``bool_vars`` and ``str_vars``. They now call ``settings_schema.desktop_settings_map(self)`` /
``desktop_bool_vars(self)`` / ``desktop_str_vars(self)``, which build the tables from the
generated ``settings_schema_data.DESKTOP_*`` rows. The literals live on, line for line, in the
frozen copy ``src/mobile/tools/frozen_desktop_tables.py`` (the generator's input).

What is proven here:

* the frozen copy is the literal at ``P5B_BASE_SHA`` (the last commit with the literals);
* tuple equality: every row, in order, with the same key / sources (lists, ``('config', key)``
  tuples) / default (value AND type; ``getattr(self, 'default_*', '')`` and
  ``self.config.get(...)`` defaults on fuzzed owners; mutable defaults fresh per build) /
  converter (the same builtin or named function object, ``None``, or - for the 75 lambdas -
  the same result or exception type and message on fuzz inputs);
* end to end: a HeadlessOwner whose ``_init_variables`` / ``_apply_live_settings_to_config`` are
  the methods at ``P5B_BASE_SHA`` vs the working tree, on fuzzed configs and owner states:
  same attributes, config, collected settings and startup environment;
* the desktop holds no table literal any more (single source), and the generator splices the
  frozen literal back in place of each call (a fake tree in both forms yields identical data).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_schema_p5b.py
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
TOOLS = SRC / "mobile" / "tools"
FROZEN = TOOLS / "frozen_desktop_tables.py"
for _path in (str(SRC), str(TESTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import settings_schema as ss  # noqa: E402
import settings_schema_data as ssd  # noqa: E402

#: The last commit whose owner_state.py / settings_persistence.py held the literal tables.
P5B_BASE_SHA = "1cd681786382abed23069f4c06cb583948371d1a"
TABLE_HOMES = {
    "settings_map": ("src/settings_persistence.py", "settings_persistence", "_apply_live_settings_to_config"),
    "bool_vars": ("src/owner_state.py", "owner_state", "_init_variables"),
    "str_vars": ("src/owner_state.py", "owner_state", "_init_variables"),
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frozen = _load(FROZEN, "frozen_desktop_tables_under_test")
se = _load(TOOLS / "schema_extract.py", "schema_extract_p5b_under_test")
LEGACY = frozen.FrozenDesktopTables


def _git_show(relpath: str, sha: str = P5B_BASE_SHA) -> str:
    try:
        data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(ROOT),
                              check=True, capture_output=True).stdout
    except Exception as exc:  # pragma: no cover - shallow clone / no git
        message = f"git show {sha}:{relpath} unavailable: {exc}"
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(message + " (CI checks out full history; the P5b pins need the base commit)")
        pytest.skip(message)
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def _assigns(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]


# --------------------------------------------------------------------------- the frozen copy
@pytest.mark.parametrize("table", sorted(TABLE_HOMES))
def test_frozen_copy_is_the_literal_at_the_base_sha(table):
    relpath, _module, method = TABLE_HOMES[table]
    legacy_tree = ast.parse(_git_show(relpath))
    legacy = [node for fn in ast.walk(legacy_tree) if isinstance(fn, ast.FunctionDef) and fn.name == method
              for node in _assigns(fn, table)]
    assert len(legacy) == 1 and isinstance(legacy[0].value, ast.List)
    frozen_nodes = _assigns(ast.parse(FROZEN.read_text(encoding="utf-8")), table)
    assert len(frozen_nodes) == 1
    assert ast.unparse(frozen_nodes[0]) == ast.unparse(legacy[0])
    # line for line, comments included (whitespace and line endings aside)
    legacy_text = ast.get_source_segment(_git_show(relpath), legacy[0])
    frozen_text = ast.get_source_segment(FROZEN.read_text(encoding="utf-8").replace("\r\n", "\n"), frozen_nodes[0])
    assert frozen_text == legacy_text


def test_frozen_helpers_are_the_save_config_helpers():
    legacy = ast.parse(_git_show("src/settings_persistence.py"))
    ours = ast.parse(FROZEN.read_text(encoding="utf-8"))
    for name in ("safe_int", "safe_float"):
        a = next(n for n in legacy.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.unparse(a) == ast.unparse(b), name
    assert frozen.FROZEN_SHA == P5B_BASE_SHA


def test_frozen_copy_is_python_310_and_uniform_line_endings():
    data = FROZEN.read_bytes()
    ast.parse(data.decode("utf-8"), feature_version=(3, 10))
    assert data.count(b"\r\n") in (0, data.count(b"\n"))


# --------------------------------------------------------------------------- single source
def test_desktop_methods_call_the_schema_tables():
    for table, (relpath, module, method) in TABLE_HOMES.items():
        tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8-sig"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method)
        nodes = _assigns(fn, table)
        assert len(nodes) == 1, (module, table)
        value = nodes[0].value
        assert isinstance(value, ast.Call) and ast.unparse(value) == f"desktop_{table}(self)", ast.unparse(value)
        imports = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and n.module == "settings_schema"]
        assert any(a.name == f"desktop_{table}" for n in imports for a in n.names), (module, table)


def test_no_table_literal_is_left_in_the_owner_modules():
    for name in se.OWNER_MODULES:
        path = SRC / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for table in se.DESKTOP_TABLES:
            literals = [n for n in _assigns(tree, table) if isinstance(n.value, ast.List)]
            assert not literals, f"{name} still holds a {table} literal (edit {FROZEN.name} instead)"


# --------------------------------------------------------------------------- tuple equality
DEFAULT_ATTRS = sorted({spec[1] for _k, _s, spec, _c in ssd.DESKTOP_SETTINGS_MAP if spec[0] == "attr"})
CONFIG_VALUES = (True, False, None, 0, 1, "", "0", "1", "yes", [], {}, "x")


class _Owner:
    """Test double: ``config`` plus a random subset of the ``default_*`` prompt attributes."""


def _owners(count, seed):
    rng = random.Random(seed)
    owners = []
    for index in range(count):
        owner = _Owner()
        owner.config = {}
        if index % 3:
            owner.config["unknown_finish_as_prohibited"] = rng.choice(CONFIG_VALUES)
        if index % 5 == 1:
            owner.config["missing_finish_as_prohibited"] = rng.choice(CONFIG_VALUES)
        for attr in DEFAULT_ATTRS:
            mode = rng.randrange(4) if index else 0
            if mode == 1:
                setattr(owner, attr, f"{attr} text {rng.randrange(1000)}")
            elif mode == 2:
                setattr(owner, attr, rng.choice(("", None, 0, ["x"])))
        owners.append(owner)
    return owners


def _is_lambda(func):
    return isinstance(func, types.FunctionType) and func.__name__ == "<lambda>"


def test_settings_map_rows_keys_sources_and_defaults_are_equal():
    for owner in _owners(60, 20261008):
        legacy = LEGACY.settings_map(owner)
        built = ss.desktop_settings_map(owner)
        assert type(built) is list and len(built) == len(legacy) == len(ssd.DESKTOP_SETTINGS_MAP)
        for old, new in zip(legacy, built):
            assert type(new) is tuple and len(new) == 4
            key, sources, default, _conv = new
            assert key == old[0]
            assert type(sources) is list and sources == old[1]
            assert [type(s) for s in sources] == [type(s) for s in old[1]], key
            assert type(default) is type(old[2]) and default == old[2], (key, default, old[2])


def test_settings_map_converters_are_the_same_objects_or_behave_the_same():
    legacy = LEGACY.settings_map(_owners(1, 1)[0])
    built = ss.desktop_settings_map(_owners(1, 1)[0])
    rng = random.Random(55)
    lambdas = checked = 0
    for old, new in zip(legacy, built):
        key, old_conv, new_conv = old[0], old[3], new[3]
        if not _is_lambda(old_conv):
            assert new_conv is old_conv, (key, old_conv, new_conv)     # bool / str / ... / named function
            continue
        lambdas += 1
        assert callable(new_conv) and not _is_lambda(new_conv)
        for value in _inputs(new_conv.conv_spec, rng):
            expected, actual = _outcome(old_conv, value), _outcome(new_conv, value)
            assert expected == actual, (key, value, expected, actual)
            checked += 1
    assert lambdas == 75 and checked > 75 * 120


BASE_INPUTS = [
    None, True, False, 0, 1, -1, 2, 5, 10, 11, -2, 0.0, 1.5, -2.7, 0.3, 1e20, -1e20,
    float("nan"), float("inf"), float("-inf"), "", " ", "0", "1", "-1", "5", "10", "1.5", "-0.5", " 7 ",
    "abc", "True", "false", "FULL", " Partial.B ", "html", "Auto", "1e3", "1e-5", "0.0001", "٣", "--1",
    "+3", "0x10", [], [1, 2], {}, {"a": 1}, (1, 2), "None", b"7", 10 ** 30, -(10 ** 30), "  ", "\t3\n",
]


def _inputs(conv, rng):
    values = list(BASE_INPUTS)
    if conv[0] == "choice":
        for choice in conv[1]:
            values += [choice, choice.upper(), f"  {choice} ", choice.title()]
    if conv[0] in ("safe_int", "safe_float"):
        for bound in conv[1:4]:
            if bound is not None:
                values += [bound, bound - 1, bound + 1, str(bound), str(bound + 1), float(bound) + 0.5]
    if conv[0] == "str_or":
        values += [conv[1], conv[2], f" {conv[2]} "]
    for _ in range(60):
        kind = rng.randrange(5)
        if kind == 0:
            values.append(rng.randint(-10 ** 6, 10 ** 6))
        elif kind == 1:
            values.append(rng.uniform(-100000, 100000))
        elif kind == 2:
            values.append(str(rng.randint(-50000, 50000)))
        elif kind == 3:
            values.append(f"{rng.uniform(-500, 500):.3f}")
        else:
            values.append("".join(rng.choice("ab-.1 0e+") for _ in range(rng.randrange(7))))
    return values


def _outcome(func, value):
    try:
        result = func(copy.deepcopy(value))
    except Exception as exc:                 # exception type and message are part of the contract
        return ("raise", type(exc).__name__, str(exc))
    if isinstance(result, float) and math.isnan(result):
        return ("ok", "float", "nan")
    return ("ok", type(result).__name__, repr(result))


@pytest.mark.parametrize("table", ["bool_vars", "str_vars"])
def test_init_variables_tables_are_equal(table):
    for owner in _owners(80, 77):
        legacy = getattr(LEGACY, table)(owner)
        built = getattr(ss, f"desktop_{table}")(owner)
        assert type(built) is list and built == legacy
        assert all(type(row) is tuple for row in built)
        assert [type(row[2]) for row in built] == [type(row[2]) for row in legacy]
    if table == "str_vars":
        assert ss.desktop_str_vars() == LEGACY.str_vars(None)


def test_missing_finish_default_reads_the_legacy_config_key():
    owner = _Owner()
    owner.config = {"unknown_finish_as_prohibited": True}
    row = next(r for r in ss.desktop_bool_vars(owner) if r[0] == "unknown_finish_as_prohibited_var")
    assert row == ("unknown_finish_as_prohibited_var", "missing_finish_as_prohibited", True)
    owner.config = {}
    row = next(r for r in ss.desktop_bool_vars(owner) if r[0] == "unknown_finish_as_prohibited_var")
    assert row[2] is False


def test_mutable_defaults_are_fresh_for_every_build():
    owner = _owners(1, 3)[0]
    first, second = ss.desktop_settings_map(owner), ss.desktop_settings_map(owner)
    mutable = [(a, b) for a, b in zip(first, second) if isinstance(a[2], (list, dict))]
    assert len(mutable) == 6                  # prompt_profiles {} + five [] defaults
    for a, b in mutable:
        assert a[2] is not b[2] and a[1] is not b[1]
        a[2].append("x") if isinstance(a[2], list) else a[2].update(x=1)
    assert all(not row[2] for row in ss.desktop_settings_map(owner) if isinstance(row[2], (list, dict)))
    for _key, _sources, default, _conv in ssd.DESKTOP_SETTINGS_MAP:
        if default[0] == "value" and isinstance(default[1], (list, dict)):
            assert not default[1]                                     # the data rows stay untouched


def test_image_only_title_default_is_the_shared_constant():
    import title_tag_translation

    row = next(r for r in ss.desktop_settings_map(_Owner()) if r[0] == "image_only_title_tag_system_prompt")
    assert row[2] is title_tag_translation.DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT


def test_desktop_tables_do_not_build_the_spec_index():
    ss._reset_cache()
    ss.desktop_settings_map(_Owner())
    ss.desktop_bool_vars(types.SimpleNamespace(config={}))
    ss.desktop_str_vars()
    assert "built" not in ss._STATE


# --------------------------------------------------------------------------- end to end
def _exec_legacy(relpath: str, name: str):
    module = types.ModuleType(name)
    module.__file__ = str(SRC / Path(relpath).name)
    exec(compile(_git_show(relpath), f"<{P5B_BASE_SHA[:12]}:{relpath}>", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def legacy_owner_class():
    import headless_owner

    owner_state_legacy = _exec_legacy("src/owner_state.py", "owner_state_p5b_legacy")
    persistence_legacy = _exec_legacy("src/settings_persistence.py", "settings_persistence_p5b_legacy")
    methods = {
        "_init_variables": owner_state_legacy.ConfigStateMixin._init_variables,
        "_apply_live_settings_to_config": persistence_legacy.SettingsPersistenceMixin._apply_live_settings_to_config,
    }
    return type("LegacyTablesOwner", (headless_owner.HeadlessOwner,), methods)


TABLE_KEYS = sorted({row[0] for row in ssd.DESKTOP_SETTINGS_MAP} | {row[1] for row in ssd.DESKTOP_BOOL_VARS}
                    | {row[1] for row in ssd.DESKTOP_STR_VARS} | {"unknown_finish_as_prohibited"})
TABLE_SOURCES = sorted({s for row in ssd.DESKTOP_SETTINGS_MAP for s in row[1] if isinstance(s, str)}
                       | {row[0] for row in ssd.DESKTOP_BOOL_VARS} | {row[0] for row in ssd.DESKTOP_STR_VARS})
FUZZ_VALUES = (True, False, None, 0, 1, -1, 3, 2.5, "", "0", "1", "5", "abc", "-1", "full", "AUTO", " 7 ", "1e3",
               [], ["k"], {})
def _fuzz_config(rng):
    config = {}
    for key in rng.sample(TABLE_KEYS, rng.randrange(0, 40)):
        config[key] = rng.choice(FUZZ_VALUES)
    return config


def _norm(value, depth=0):
    """Comparable form of an owner attribute (objects by type and state, never by address)."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    if depth > 4:
        return type(value).__name__
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_norm(v, depth + 1) for v in value]
        return (type(value).__name__, sorted(items) if isinstance(value, (set, frozenset)) else items)
    if isinstance(value, dict):
        return ("dict", [(repr(k), _norm(v, depth + 1)) for k, v in value.items()])
    if callable(value) and not hasattr(value, "__dict__"):
        return type(value).__name__
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return (type(value).__name__, [(k, _norm(v, depth + 1)) for k, v in sorted(state.items())
                                       if not callable(v) or hasattr(v, "__dict__")])
    return type(value).__name__


def _snapshot(owner):
    return {name: _norm(value) for name, value in sorted(vars(owner).items()) if name != "host"}


def _run(owner_class, config, overrides, root, monkeypatch):
    import app_paths
    from _headless_env import scrubbed_env

    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)          # both owners start from the same empty folder
    root.mkdir(parents=True)
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(root / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(root / "app_paths.py"))
    with scrubbed_env(root):
        try:
            owner = owner_class(copy.deepcopy(config), host=types.SimpleNamespace(log=lambda _m: None))
        except Exception as exc:
            return ("init-raise", type(exc).__name__, str(exc))
        environ = {k: v for k, v in os.environ.items() if k not in ("GLOSSARION_WATCHDOG_DIR",)}
        state = _snapshot(owner)
        for name, value in overrides.items():
            setattr(owner, name, copy.deepcopy(value))
        try:
            collected = repr(owner._collect_live_settings())
        except Exception as exc:
            collected = ("raise", type(exc).__name__, str(exc))
    return ("ok", state, environ, collected)


def test_headless_owner_with_legacy_tables_is_identical(legacy_owner_class, tmp_path, monkeypatch):
    import headless_owner

    rng = random.Random(9055)
    states = 36
    raised = 0
    # warm-up: first-time lazy imports during a build (PySide6 when it is installed) edit os.environ
    # (PATH, PYSIDE6_OPTION_PYTHON_ENUM) once per process; the compared builds must not see that
    _run(headless_owner.HeadlessOwner, {}, {}, tmp_path / "owner", monkeypatch)
    for index in range(states):
        config = _fuzz_config(rng) if index else {}
        overrides = {name: rng.choice(FUZZ_VALUES) for name in rng.sample(TABLE_SOURCES, rng.randrange(0, 12))}
        overrides.pop("api_key_entry", None)
        expected = _run(legacy_owner_class, config, overrides, tmp_path / "owner", monkeypatch)
        actual = _run(headless_owner.HeadlessOwner, config, overrides, tmp_path / "owner", monkeypatch)
        if expected[0] != "ok":
            raised += 1
            assert actual == expected, (index, config)
            continue
        assert actual[0] == "ok", (index, actual, config)
        _tag, state_a, env_a, collected_a = expected
        _tag, state_b, env_b, collected_b = actual
        diff = sorted(k for k in set(state_a) | set(state_b) if state_a.get(k) != state_b.get(k))
        assert not diff, (index, diff[:10], config)
        assert env_a == env_b, (index, sorted(k for k in set(env_a) | set(env_b) if env_a.get(k) != env_b.get(k)))
        assert collected_a == collected_b, (index, overrides)
    assert raised < states // 2


def test_end_to_end_harness_catches_a_changed_row(legacy_owner_class, tmp_path, monkeypatch):
    """Self-test: one flipped bool_vars default and one changed clamp are both detected."""
    import headless_owner

    bools = list(ssd.DESKTOP_BOOL_VARS)
    i = next(i for i, row in enumerate(bools) if row[0] == "retry_truncated_var")
    bools[i] = bools[i][:2] + (("value", not bools[i][2][1]),)
    rows = list(ssd.DESKTOP_SETTINGS_MAP)
    j = next(j for j, row in enumerate(rows) if row[0] == "vision_ocr_batch_size")
    assert rows[j][3] == ("safe_int", -1, -1, None, "")
    rows[j] = rows[j][:3] + (("safe_int", -1, 0, None, ""),)
    monkeypatch.setattr(ssd, "DESKTOP_BOOL_VARS", tuple(bools))
    monkeypatch.setattr(ssd, "DESKTOP_SETTINGS_MAP", tuple(rows))
    overrides = {"vision_ocr_batch_size_var": "-5"}
    expected = _run(legacy_owner_class, {}, overrides, tmp_path / "owner", monkeypatch)
    actual = _run(headless_owner.HeadlessOwner, {}, overrides, tmp_path / "owner", monkeypatch)
    assert expected[0] == actual[0] == "ok"
    assert expected[1]["retry_truncated_var"] != actual[1]["retry_truncated_var"]
    assert "'vision_ocr_batch_size': -1" in expected[3] and "'vision_ocr_batch_size': 0" in actual[3]


# --------------------------------------------------------------------------- generator
FAKE_PROMPTS = 'LONG_PROMPT = "{}"\n'.format("p" * 200)
FAKE_GUI = (
    "import os\n"
    "from owner_state import ConfigStateMixin\n"
    "from settings_persistence import SettingsPersistenceMixin\n"
    "\n"
    "\n"
    "class TranslatorGUI(SettingsPersistenceMixin, ConfigStateMixin):\n"
    "    def __init__(self):\n"
    "        self.config = {}\n"
    "        self._init_variables()\n"
)
BOOL_LITERAL = """[
            ('contextual_var', 'contextual', False),
            # a comment inside the table
            ('finish_var', 'missing_finish', self.config.get('unknown_finish', False)),
        ]"""
STR_LITERAL = """[
            ('api_queue_var', 'api_queue', 4),
            ('mode_var', 'refine_mode', 'full'),
        ]"""
MAP_LITERAL = """[
            ('contextual', ['contextual_checkbox', 'contextual_var'], False, bool),
            ('api_queue', ['api_queue_var'], 4, lambda v: max(-1, safe_int(v, 4))),
            ('refine_mode', ['mode_var'], 'full', lambda v: str(v).strip().lower() if str(v).strip().lower() in MODES else 'full'),
            ('vision_prompt', ['vision_prompt'], getattr(self, 'default_vision_prompt', ''), str),
            ('long_prompt', ['long_prompt_var'], LONG_PROMPT, str),
            ('layout', ['layout_var', ('config', 'layout')], 'auto', str),
            ('fields', ['fields_var'], [], list),
        ]"""
OWNER_STATE = '''
MODES = ("full", "partial")


class ConfigStateMixin:
    def _init_variables(self):
        self.vision_prompt = self.config.get('vision_prompt', 'v')

        def create_var(var_type, key, default):
            return self.config.get(key, default)
{bool_import}
        bool_vars = {bool_value}
        for var_name, key, default in bool_vars:
            setattr(self, var_name, create_var(bool, key, default))
        str_vars = {str_value}
        for var_name, key, default in str_vars:
            setattr(self, var_name, create_var(str, key, str(default)))
'''
PERSISTENCE = '''
from fake_prompts import LONG_PROMPT
from owner_state import MODES


def safe_int(value, default):
    try: return int(value)
    except (ValueError, TypeError): return default


class SettingsPersistenceMixin:
    def _apply_live_settings_to_config(self):
{map_import}
        settings_map = {map_value}
        for key, sources, default, converter in settings_map:
            pass
        self.config.setdefault('backup_enabled', True)
'''
FROZEN_FAKE = '''
class FrozenDesktopTables:
    def settings_map(self):
        settings_map = {map_value}
        return settings_map

    def bool_vars(self):
        bool_vars = {bool_value}
        return bool_vars

    def str_vars(self):
        str_vars = {str_value}
        return str_vars
'''


def _fake_tree(root: Path, form: str, frozen_copy=True, bool_literal=BOOL_LITERAL) -> Path:
    src = root / "src"
    (src / "mobile" / "tools").mkdir(parents=True)
    literal = form == "literal"
    files = {
        "fake_prompts.py": FAKE_PROMPTS,
        "translator_gui.py": FAKE_GUI,
        "owner_state.py": OWNER_STATE.format(
            bool_import="" if literal else "        from settings_schema import desktop_bool_vars, desktop_str_vars",
            bool_value=bool_literal if literal else "desktop_bool_vars(self)",
            str_value=STR_LITERAL if literal else "desktop_str_vars(self)"),
        "settings_persistence.py": PERSISTENCE.format(
            map_import="" if literal else "        from settings_schema import desktop_settings_map",
            map_value=MAP_LITERAL if literal else "desktop_settings_map(self)"),
    }
    if frozen_copy:
        files[se.FROZEN_TABLES] = FROZEN_FAKE.format(map_value=MAP_LITERAL, bool_value=bool_literal,
                                                     str_value=STR_LITERAL)
    for name, text in files.items():
        (src / name).write_text(text, encoding="utf-8")
        ast.parse(text)
    return src


def _data(text):
    namespace = {}
    exec(compile(text, "settings_schema_data.py", "exec"), namespace)
    return namespace


def test_generator_splices_the_frozen_literals_back(tmp_path):
    literal = se.generate(_fake_tree(tmp_path / "literal", "literal"))
    spliced = se.generate(_fake_tree(tmp_path / "call", "call"))
    assert spliced == literal
    data = _data(literal)
    assert data["DESKTOP_BOOL_VARS"] == (
        ("contextual_var", "contextual", ("value", False)),
        ("finish_var", "missing_finish", ("config", "unknown_finish", ("value", False))))
    assert data["DESKTOP_STR_VARS"] == (("api_queue_var", "api_queue", ("value", 4)),
                                        ("mode_var", "refine_mode", ("value", "full")))
    rows = {row[0]: row for row in data["DESKTOP_SETTINGS_MAP"]}
    assert [row[0] for row in data["DESKTOP_SETTINGS_MAP"]] == list(data["SETTINGS_MAP_ORDER"])
    assert rows["contextual"] == ("contextual", ("contextual_checkbox", "contextual_var"), ("value", False), ("bool",))
    assert rows["api_queue"][3] == ("safe_int", 4, -1, None, "")
    assert rows["refine_mode"][3] == ("choice", ("full", "partial"), "full")
    assert rows["vision_prompt"][2] == ("attr", "default_vision_prompt", ("value", ""))
    assert rows["long_prompt"][2] == ("ref", "fake_prompts:LONG_PROMPT")
    assert rows["layout"][1] == ("layout_var", ("config", "layout"))
    assert rows["fields"][2] == ("value", [])
    # the records read through the spliced table are the literal's (var map, init defaults)
    assert data["SETTINGS"]["missing_finish"]["init_default"] is False
    assert data["SETTINGS"]["refine_mode"]["var_names"] == ("mode_var",)


def test_generator_without_the_frozen_copy_fails_loudly(tmp_path):
    src = _fake_tree(tmp_path, "call", frozen_copy=False)
    with pytest.raises(SystemExit, match="frozen copy"):
        se.generate(src)


def test_generator_refuses_a_default_the_schema_cannot_rebuild(tmp_path):
    odd = BOOL_LITERAL.replace("self.config.get('unknown_finish', False)", "self.finish_default")
    src = _fake_tree(tmp_path, "literal", bool_literal=odd)
    with pytest.raises(SystemExit, match="cannot be rebuilt"):
        se.generate(src)


def test_data_rows_match_the_frozen_literals_row_for_row():
    """The committed DESKTOP_* rows encode the frozen literals (the drift test proves the rest)."""
    tree = ast.parse(FROZEN.read_text(encoding="utf-8"))
    for table, rows in (("settings_map", ssd.DESKTOP_SETTINGS_MAP), ("bool_vars", ssd.DESKTOP_BOOL_VARS),
                        ("str_vars", ssd.DESKTOP_STR_VARS)):
        node = _assigns(tree, table)[0].value
        assert [ast.literal_eval(elt.elts[0]) for elt in node.elts] == [row[0] for row in rows], table
    assert tuple(row[0] for row in ssd.DESKTOP_SETTINGS_MAP) == ssd.SETTINGS_MAP_ORDER


def test_settings_schema_still_imports_without_qt():
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, sys.argv[1]);"
        "import settings_schema as ss; t = ss.desktop_settings_map(type('O', (), {'config': {}})());"
        "assert len(t) == len(ss._data().DESKTOP_SETTINGS_MAP);"
        "bad = [m for m in ('translator_gui', 'dpi_setup', 'PySide6.QtWidgets') if sys.modules.get(m) is not None];"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert result.returncode == 0, result.stderr[-2000:]
    ast.parse((SRC / "settings_schema.py").read_text(encoding="utf-8"), feature_version=(3, 10))
