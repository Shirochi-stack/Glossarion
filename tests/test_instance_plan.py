"""Unit tests for src/instance_plan.py (multi-instance translation: files -> instances) and the
Translation key's ``instance`` field (multi_api_key_manager.APIKeyEntry).

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_instance_plan.py
"""

import ast
import copy
import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import instance_plan as ip  # noqa: E402


def _key(instance=None, enabled=True, name="k"):
    entry = {"api_key": name, "model": "gpt-x", "enabled": enabled}
    if instance is not None:
        entry["instance"] = instance
    return entry


def _config(*keys, on=True, **extra):
    config = {"use_multi_api_keys": on, "multi_api_keys": list(keys),
              "fallback_keys": [{"api_key": "fb", "model": "m"}], "use_fallback_keys": True}
    config.update(extra)
    return config


# --------------------------------------------------------------------------
# instance numbers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, 1), (1, 1), (2, 2), ("3", 3), (" 4 ", 4), (2.0, 2), ("2.0", 2),
    (0, 1), (-2, 1), ("abc", 1), ("", 1), (2.5, 1), (True, 1), (False, 1),
    (float("nan"), 1), (float("inf"), 1), ([2], 1), ({"a": 1}, 1),
])
def test_coerce_and_key_instance(value, expected):
    assert ip.coerce_instance(value) == expected
    assert ip.key_instance({"instance": value}) == expected


def test_key_instance_defaults_and_objects():
    assert ip.key_instance({}) == 1
    assert ip.key_instance(None) == 1
    assert ip.key_instance(type("Entry", (), {"instance": 3})()) == 3


def test_current_instance_reads_the_environment(monkeypatch):
    monkeypatch.delenv(ip.ENV_INSTANCE, raising=False)
    assert ip.current_instance() == 1
    monkeypatch.setenv(ip.ENV_INSTANCE, "3")
    assert ip.current_instance() == 3
    monkeypatch.setenv(ip.ENV_INSTANCE, "garbage")
    assert ip.current_instance() == 1
    assert ip.ENV_INSTANCE == "GLOSSARION_INSTANCE" and ip.CONFIG_FILE_MAP == "instance_file_map"


# --------------------------------------------------------------------------
# available instances / when multi-instance applies
# --------------------------------------------------------------------------

def test_available_instances_counts_enabled_translation_keys_only():
    config = _config(_key(), _key(3), _key("2"), _key(5, enabled=False), _key(3, name="k2"))
    assert ip.available_instances(config) == [1, 2, 3]
    # the other pools never contribute instance numbers
    config["fallback_keys"] = [{"api_key": "fb", "model": "m", "instance": 9}]
    assert ip.available_instances(config) == [1, 2, 3]


def test_available_instances_is_one_when_multi_keys_are_off_or_empty():
    assert ip.available_instances(_config(_key(2), _key(3), on=False)) == [1]
    assert ip.available_instances({"multi_api_keys": [_key(2), _key(3)]}) == [1]  # toggle missing
    assert ip.available_instances(_config()) == [1]
    assert ip.available_instances(_config(_key(2, enabled=False))) == [1]
    assert ip.available_instances(_config(_key(2), "not-a-row", None)) == [2]
    assert ip.available_instances(None) == [1]
    assert ip.available_instances({"use_multi_api_keys": True, "multi_api_keys": None}) == [1]


def test_is_multi_instance_needs_two_instances_and_two_files():
    two = _config(_key(1), _key(2))
    files = ["a.epub", "b.epub"]
    assert ip.is_multi_instance(two, files) is True
    assert ip.is_multi_instance(two, ["a.epub"]) is False
    assert ip.is_multi_instance(two, ["a.epub", "a.epub", ""]) is False  # one distinct file
    assert ip.is_multi_instance(two, ["a.epub", os.path.abspath("a.epub")]) is False
    assert ip.is_multi_instance(two, []) is False and ip.is_multi_instance(two, None) is False
    assert ip.is_multi_instance(_config(_key(2), _key(2, name="x")), files) is False  # one number
    assert ip.is_multi_instance(_config(_key(1), _key(2), on=False), files) is False
    assert ip.is_multi_instance(_config(_key(1), _key(2, enabled=False)), files) is False


# --------------------------------------------------------------------------
# assignment and plan
# --------------------------------------------------------------------------

def test_default_assignment_round_robin_in_file_order():
    files = ["c.epub", "a.epub", "b.txt", "d.pdf", "e.epub"]
    assert ip.default_assignment(files, [1, 2]) == {
        "c.epub": 1, "a.epub": 2, "b.txt": 1, "d.pdf": 2, "e.epub": 1,
    }
    assert list(ip.default_assignment(files, [1, 2, 3])) == files  # file order kept
    assert ip.default_assignment(files, [2, 3, 3]) == {
        "c.epub": 2, "a.epub": 3, "b.txt": 2, "d.pdf": 3, "e.epub": 2,
    }
    assert ip.default_assignment(files[:2], []) == {"c.epub": 1, "a.epub": 1}
    assert ip.default_assignment([], [1, 2]) == {}


def test_default_assignment_keeps_saved_instances_that_still_exist(tmp_path):
    a, b, c, d = (str(tmp_path / name) for name in ("a.epub", "b.epub", "c.epub", "d.epub"))
    saved = {
        os.path.normcase(os.path.abspath(a)).upper() if os.name == "nt" else a: 3,  # case-insensitive on Windows
        b: 7,          # instance 7 has no key any more -> round-robin
        c: "bad",      # unusable saved value -> round-robin
        "": 2,         # ignored
    }
    result = ip.default_assignment([a, b, c, d, a], [1, 2, 3], saved_map=saved)
    assert result == {a: 3, b: 1, c: 2, d: 3}
    assert ip.default_assignment([a, b], [1, 2], saved_map="not-a-dict") == {a: 1, b: 2}


def test_default_assignment_returns_the_callers_strings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = {str(tmp_path / "book.epub"): 2}
    assert ip.default_assignment(["book.epub", "other.epub"], [1, 2], saved) == {
        "book.epub": 2, "other.epub": 1,
    }


def test_plan_groups_files_per_instance_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = ["a.epub", "b.epub", "c.epub", "d.epub", "a.epub", ""]
    assignment = {"a.epub": 2, "b.epub": "1", str(tmp_path / "c.epub"): 2}
    # d.epub is not in the assignment: instance 1; a.epub only once
    assert ip.plan(files, assignment) == {1: ["b.epub", "d.epub"], 2: ["a.epub", "c.epub"]}
    assert list(ip.plan(["x", "y"], {"x": 3, "y": 1})) == [1, 3]  # instance order
    assert ip.plan([], {"x": 2}) == {}  # instances without files are dropped
    assert ip.plan(["x"], None) == {1: ["x"]}


def test_plan_of_the_default_assignment_covers_every_file_once():
    files = [f"book{i}.epub" for i in range(7)]
    planned = ip.plan(files, ip.default_assignment(files, [1, 2, 3]))
    assert sorted(sum(planned.values(), [])) == sorted(files)
    assert {n: len(v) for n, v in planned.items()} == {1: 3, 2: 2, 3: 2}


# --------------------------------------------------------------------------
# per-instance config snapshot
# --------------------------------------------------------------------------

def test_config_for_instance_keeps_only_that_instances_translation_keys():
    config = _config(_key(name="one"), _key(2, name="two"), _key("2", enabled=False, name="two-off"),
                     _key(3, name="three"), instance_file_map={"a": 2}, model="gpt-x", api_key="main")
    before = copy.deepcopy(config)
    snap = ip.config_for_instance(config, 2)
    assert [k["api_key"] for k in snap["multi_api_keys"]] == ["two", "two-off"]
    one = ip.config_for_instance(config, "1")
    assert [k["api_key"] for k in one["multi_api_keys"]] == ["one"]
    # instance 1 keeps its fallback key and the main GUI key; every other setting is copied
    # unchanged, and the input is not modified
    assert {k: v for k, v in one.items() if k != "multi_api_keys"} == \
        {k: v for k, v in before.items() if k != "multi_api_keys"}
    assert config == before
    one["fallback_keys"][0]["api_key"] = "changed"
    assert config["fallback_keys"][0]["api_key"] == "fb"  # deep copy
    # instance 2 has no fallback key of its own: the pool is off for it
    assert snap["fallback_keys"] == [] and snap["use_fallback_keys"] is False
    assert snap["instance_file_map"] == {"a": 2} and snap["model"] == "gpt-x"


def _pool_key(name, instance=None, enabled=True):
    entry = _key(instance, enabled, name)
    entry["model"] = f"model-{name}"
    return entry


def test_config_for_instance_splits_every_key_pool():
    """Every pool of key_pool_service.POOL_SPECS keeps only the instance's rows; a pool other than
    Translation left without an enabled row of the instance is switched off for it."""
    config = _config(_key(name="t1"), _key(2, name="t2"), model="gpt-x", api_key="main")
    for pool, (config_key, toggle_key, _use, _keys) in ip.POOLS.items():
        if pool == "main":
            continue
        config[config_key] = [_pool_key(f"{pool}-1"), _pool_key(f"{pool}-2", 2),
                              _pool_key(f"{pool}-2off", "2", enabled=False)]
        config[toggle_key] = True
    config["qa_scan_keys"] = [_pool_key("qa-1")]  # instance 2 has no Vision key at all
    config["metadata_keys"] = [_pool_key("meta-2off", 2, enabled=False)]  # only a disabled one
    snap = ip.config_for_instance(config, 2)
    one = ip.config_for_instance(config, 1)
    for pool, (config_key, toggle_key, _use, _keys) in ip.POOLS.items():
        assert all(ip.key_instance(k) == 2 for k in snap[config_key]), pool
        assert all(ip.key_instance(k) == 1 for k in one[config_key]), pool
    assert [k["api_key"] for k in snap["fallback_keys"]] == ["fallback-2", "fallback-2off"]
    assert [k["api_key"] for k in one["glossary_keys"]] == ["glossary-1"]
    assert snap["use_fallback_keys"] is True and snap["use_tts_keys"] is True
    assert snap["qa_scan_keys"] == [] and snap["use_qa_scan_keys"] is False
    assert [k["api_key"] for k in snap["metadata_keys"]] == ["meta-2off"] and snap["use_metadata_keys"] is False
    assert one["use_qa_scan_keys"] is True and one["metadata_keys"] == [] and one["use_metadata_keys"] is False
    # a toggle that was off stays off and is not added where it was missing
    config["use_glossary_keys"] = False
    config.pop("use_tts_keys")
    snap = ip.config_for_instance(config, 2)
    assert snap["use_glossary_keys"] is False and "use_tts_keys" not in snap


@pytest.mark.parametrize("keys, model, expected, expected_model", [
    ([_key(2, name="t2a"), dict(_key(2, name="t2b"), model="other")], "gpt-x", "t2a", "gpt-x"),
    ([dict(_key(2, name="t2a"), model="other"), _key(2, name="t2b")], "gpt-x", "t2b", "gpt-x"),  # the main model's key
    # no key of the instance serves the main model: the stand-in key comes with its own model, so a request that
    # falls back to "the main key" (every pool key cooling down) never sends it to another provider's model
    ([dict(_key(2, name="t2a"), model="other"), dict(_key(2, name="t2b"), model="x")], "gpt-x", "t2a", "other"),
    ([_key(2, enabled=False, name="off"), dict(_key(2, name="on"), model="x")], "gpt-x", "on", "x"),
])
def test_config_for_instance_two_never_holds_the_main_gui_key(keys, model, expected, expected_model):
    config = _config(_key(name="t1"), *keys, model=model, api_key="MAIN-GUI-KEY", use_main_key_fallback=True)
    snap = ip.config_for_instance(config, 2)
    assert snap["api_key"] == expected and snap["use_main_key_fallback"] is False
    assert snap["model"] == expected_model  # the main model while a key of the instance serves it
    assert "MAIN-GUI-KEY" not in repr(snap)
    one = ip.config_for_instance(config, 1)
    assert one["api_key"] == "MAIN-GUI-KEY" and one["use_main_key_fallback"] is True


def test_pools_cover_every_key_pool_service_pool():
    import key_pool_service

    assert set(ip.POOLS) == set(key_pool_service.POOL_SPECS)
    for pool, (config_key, toggle_key, use_envs, keys_envs) in ip.POOLS.items():
        spec = key_pool_service.POOL_SPECS[pool]
        assert (config_key, toggle_key) == (spec["config_key"], spec["toggle_key"]), pool
        assert set(spec["use_envs"]) <= set(use_envs), pool
        assert set(spec["keys_envs"]) <= set(keys_envs), pool
    for name in ("FALLBACK_KEYS", "GLOSSARY_API_KEYS", "MULTI_API_KEYS", "USE_MAIN_KEY_FALLBACK", "API_KEY",
                 "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENAI_OR_Gemini_API_KEY", "USE_GLOSSARY_KEYS"):
        assert name in ip.KEY_ENV, name


# --------------------------------------------------------------------------
# key scope (the API client's view of the keys inside one instance)
# --------------------------------------------------------------------------

@pytest.fixture
def no_scope(monkeypatch):
    monkeypatch.delenv(ip.ENV_INSTANCE, raising=False)
    monkeypatch.delenv(ip.ENV_KEY_INSTANCE, raising=False)
    return monkeypatch


def test_key_scope(no_scope):
    assert ip.key_scope() is None  # single-instance: every key
    no_scope.setenv(ip.ENV_INSTANCE, "1")
    assert ip.key_scope() is None
    no_scope.setenv(ip.ENV_INSTANCE, "3")
    assert ip.key_scope() == 3  # an instance >= 2 always narrows
    no_scope.setenv(ip.ENV_KEY_INSTANCE, "1")
    assert ip.key_scope() == 1
    no_scope.setenv(ip.ENV_KEY_INSTANCE, "bad")
    assert ip.key_scope() == 3


def test_scope_keys(no_scope):
    keys = [_key(name="a"), _key(2, name="b"), _key("2", name="c")]
    assert ip.scope_keys(keys) is keys  # no scope: the very same list
    assert ip.scope_keys(None) is None
    assert [k["api_key"] for k in ip.scope_keys(keys, 2)] == ["b", "c"]
    no_scope.setenv(ip.ENV_KEY_INSTANCE, "1")
    assert [k["api_key"] for k in ip.scope_keys(keys)] == ["a"]
    assert ip.scope_keys("not a list") == "not a list"


def test_scoped_env(no_scope):
    import json

    fallback = json.dumps([_key(name="f1"), _key(2, name="f2")])
    env = {"FALLBACK_KEYS": fallback, "GLOSSARY_API_KEYS": json.dumps([_key(name="g1")]),
           "METADATA_API_KEYS": json.dumps([_key(2, enabled=False, name="m2")])}
    # no scope: everything passes through
    for name, value in [("FALLBACK_KEYS", fallback), ("USE_GLOSSARY_KEYS", "1"), ("USE_MAIN_KEY_FALLBACK", "1")]:
        assert ip.scoped_env(name, value, environ=env) == value
    no_scope.setenv(ip.ENV_KEY_INSTANCE, "2")
    assert [k["api_key"] for k in json.loads(ip.scoped_env("FALLBACK_KEYS", fallback, environ=env))] == ["f2"]
    assert ip.scoped_env("FALLBACK_KEYS", "{broken", environ=env) == "[]"
    assert ip.scoped_env("FALLBACK_KEYS", None, environ=env) is None
    assert ip.scoped_env("USE_FALLBACK_KEYS", "1", environ=env) == "1"
    assert ip.scoped_env("USE_GLOSSARY_KEYS", "1", environ=env) == "0"      # no glossary key of instance 2
    assert ip.scoped_env("USE_METADATA_KEYS", "1", environ=env) == "0"      # only a disabled one
    assert ip.scoped_env("USE_GLOSSARY_KEYS", "0", environ=env) == "0"
    assert ip.scoped_env("USE_MAIN_KEY_FALLBACK", "1", environ=env) == "0"  # the main GUI key is instance 1's
    assert ip.scoped_env("USE_MULTI_API_KEYS", "1", environ=env) == "1"     # Translation: unchanged
    assert ip.scoped_env("MODEL", "gpt", environ=env) == "gpt"
    # the in-memory list of a pool counts too
    assert ip.scoped_env("USE_TTS_KEYS", "1", lambda pool: [_key(2, name="t")] if pool == "tts" else [],
                         environ=env) == "1"
    no_scope.setenv(ip.ENV_KEY_INSTANCE, "1")
    assert ip.scoped_env("USE_MAIN_KEY_FALLBACK", "1", environ=env) == "1"
    assert ip.scoped_env("USE_GLOSSARY_KEYS", "1", environ=env) == "1"


def test_config_for_instance_without_an_enabled_key_raises():
    config = _config(_key(1), _key(2, enabled=False))
    with pytest.raises(ValueError, match="Instance 2"):
        ip.config_for_instance(config, 2)
    with pytest.raises(ValueError):
        ip.config_for_instance(config, 4)
    with pytest.raises(ValueError):
        ip.config_for_instance({}, 1)


# --------------------------------------------------------------------------
# module rules + packaging
# --------------------------------------------------------------------------

def _top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), feature_version=(3, 10))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_instance_plan_is_gui_free_and_spawns_nothing():
    imports = _top_level_imports(SRC_DIR / "instance_plan.py")
    # stdlib only (functools / hashlib / threading: the request guard's key fingerprints)
    assert imports <= {"copy", "functools", "hashlib", "json", "math", "os", "threading"}, imports


def _spec_list(source, name):
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = ["))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "]")
    return ast.literal_eval("\n".join(lines[start:end + 1]).split("=", 1)[1].strip())


def test_instance_plan_ships_in_both_lists_of_every_spec():
    specs = sorted(SRC_DIR.glob("translator*.spec"))
    assert len(specs) == 14, [p.name for p in specs]
    for spec in specs:
        source = spec.read_text(encoding="utf-8")
        files = [Path(entry[0]).stem for entry in _spec_list(source, "app_files")]
        modules = _spec_list(source, "app_modules")
        assert files.count("instance_plan") == 1, (spec.name, "app_files")
        assert modules.count("instance_plan") == 1, (spec.name, "app_modules")


# --------------------------------------------------------------------------
# APIKeyEntry.instance (multi_api_key_manager data model)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def entry_cls():
    import multi_api_key_manager

    return multi_api_key_manager.APIKeyEntry


@pytest.mark.parametrize("value, expected", [(2, 2), ("3", 3), (1, 1), (0, 1), ("x", 1), (None, 1), (-4, 1)])
def test_api_key_entry_instance_is_coerced(entry_cls, value, expected):
    assert entry_cls("k", "m", instance=value).instance == expected
    assert entry_cls.from_dict({"api_key": "k", "model": "m", "instance": value}).instance == expected


def test_api_key_entry_instance_round_trips(entry_cls):
    assert entry_cls("k", "m").instance == 1
    assert entry_cls.from_dict({"api_key": "k", "model": "m"}).instance == 1  # older configs
    data = entry_cls("k", "m", instance=4).to_dict()
    assert data["instance"] == 4
    assert entry_cls.from_dict(data).to_dict() == data
    # the default is not written, so configs of single-instance users stay byte-identical
    assert "instance" not in entry_cls("k", "m").to_dict()
    assert "instance" not in entry_cls.from_dict({"api_key": "k", "model": "m", "instance": "bad"}).to_dict()
    assert ip.key_instance(entry_cls("k", "m", instance=3).to_dict()) == 3
    assert ip.key_instance(entry_cls("k", "m").to_dict()) == 1
