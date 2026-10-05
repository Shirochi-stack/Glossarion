"""Desktop parity tiers for the shared GUI-free core (U2 and later milestones).

Wires the reusable tier modules of tests/parity (shared-core design §6):

* harness self-tests (always run, no desktop import): the differential fuzz driver
  on a toy pair (identical -> pass, each injected difference -> caught), seeded
  state generation, AST read sets, the owner-contract scanner and the import
  hygiene probe on toy modules;
* harness self-tests on the real frozen desktop code (need the legacy oracle):
  legacy-vs-legacy fuzz (any mismatch = incomplete per-run reset) and the tier R
  helper on a desktop restart (records LEGACY_RESTART_DIVERGENCES);
* tier D (``fuzz_moved``): every ``moved_functions.FUZZED`` entry, legacy oracle vs
  the working-tree desktop, >= 500 seeded states;
* tier I (``import_hygiene``): every shared module imports with PySide6 blocked,
  never pulls in translator_gui/dpi_setup, parses as Python 3.10 (and imports
  under a real 3.10 when ``GLOSSARION_PY310`` is set);
* MRO / duplicate / class-attribute checks and the owner contract
  (``owner_contract``) against HeadlessOwner and a booted desktop owner;
* tier R (``roundtrip``): HeadlessOwner(_collect_live_settings(live)) vs the live
  desktop and vs a desktop restarted from the same settings.

Tiers whose module does not exist yet skip with the reason. Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/parity/test_parity_tiers.py

``PARITY_FUZZ_STATES`` changes the number of fuzz states (default 500; the
minimum check follows it), ``PARITY_FUZZ_TRACE=1`` prints every state to stderr.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import fakes, fuzz_moved as fm, import_hygiene, moved_functions as mf, owner_contract  # noqa: E402
from parity import roundtrip as rt  # noqa: E402


def _module_exists(name: str) -> bool:
    return (SRC_DIR / f"{name}.py").is_file()


def _u2_modules_present() -> list:
    return [m for m in mf.MIXINS if _module_exists(m)] + [m for m in ("headless_owner",) if _module_exists(m)]


# ===========================================================================
# Harness self-tests on toy code (fast; no desktop import)
# ===========================================================================


class ToyOwnerCode:
    """The 'legacy' toy method. Every read kind the driver must find is present."""

    LIMIT = 3

    def toy_build(self, path):
        env = {}
        if getattr(self, "flag_var", False):
            env["FLAG"] = "1"
        mode = self.config.get("mode", "off")
        env["MODE"] = str(mode)
        if hasattr(self, "entry"):
            env["TEXT"] = self.entry.text()
        if os.environ.get("TOY_IN") == "1":
            env["ENV_IN"] = "yes"
        if self.count_var == "0":
            env["ZERO"] = "y"
        os.environ["TOY_OUT"] = env["MODE"]
        print(f"toy build {len(env)}")
        with open("toy_out.txt", "w", encoding="utf-8") as fh:
            fh.write(env["MODE"])
        self.append_log(f"built {path}")
        self.config["touched"] = True
        self.last_env = dict(env)
        return env


def _toy_variant(source_edit=None):
    """A copy of ToyOwnerCode.toy_build compiled from (edited) source: a separate function object."""
    import inspect

    src = textwrap.dedent(inspect.getsource(ToyOwnerCode.toy_build))
    if source_edit:
        old, new = source_edit
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = {"os": os}
    exec(compile(src, f"<toy {source_edit!r}>", "exec"), ns)
    return type("ToyVariant", (), {"toy_build": ns["toy_build"], "LIMIT": 3})


#: injected difference -> output field that must catch it
TOY_VARIANTS = {
    "result": (('config.get("mode", "off")', 'config.get("mode", "on")'), "result"),
    "env": (('os.environ["TOY_OUT"] = env["MODE"]', 'os.environ["TOY_OUT"] = env["MODE"] + "!"'), "env"),
    "config": (('self.config["touched"] = True', 'self.config["touched"] = 1.5'), "config"),
    "attrs": (("self.last_env = dict(env)", "self.last_env = dict(env, extra=1)"), "attrs"),
    "calls": (('self.append_log(f"built {path}")', 'self.append_log(f"built  {path}")'), "calls"),
    "exception": (('if self.count_var == "0":', 'if self.__dict__["count_var"] == "0":'), "exc_type"),
    "rare_branch": (('env["ZERO"] = "y"', 'env["ZERO"] = "n"'), "result"),
    "stdout": (('print(f"toy build {len(env)}")', 'print(f"toy build {len(env)} ")'), "stdout"),
    "fs": (('fh.write(env["MODE"])', 'fh.write(env["MODE"] * 2)'), "fs"),
    "key_order": (("env = {}", 'env = {"MODE": None}'), None),
}


def _toy_space(seed=fm.DEFAULT_SEED, *, arg_paths=()):
    rs = fm.analyze_source(textwrap.dedent(__import__("inspect").getsource(ToyOwnerCode.toy_build)))
    rs.drop_provided({"append_log", "LIMIT"})
    bases = [fm.EMPTY_BASE,
             fm.Base("toy", {"flag_var": True, "count_var": "5", "entry": fakes.FakeLineEdit("hi")},
                     {"mode": "balanced"})]
    gen = fm.ArgGen(ToyOwnerCode.toy_build, paths=list(arg_paths))
    return fm.StateSpace(rs, bases, arg_gen=gen, seed=seed)


def _toy_owner_cls(code_cls, tag):
    return fm.owner_class(fm.OWNER_CLASS_NAME, (fakes.FakeState, code_cls),
                          {"append_log": fm._recorder_function("append_log")}, tag=tag)


def _run_toy(variant_cls, *, states=fm.DEFAULT_STATES, stop_after=20):
    legacy = fm.Side("legacy", _toy_owner_cls(ToyOwnerCode, "legacy"), "toy_build")
    new = fm.Side("new", _toy_owner_cls(variant_cls, "new"), "toy_build")
    with fm.FuzzContext("toy", backend=False, files={"inputs/a.txt": "x"}) as ctx:
        space = _toy_space(arg_paths=[ctx.path("inputs/a.txt"), ctx.path("inputs/missing.txt")])
        report = fm.differential_fuzz(ctx, legacy, new, space, states=states, stop_after=stop_after,
                                      label="toy_build")
    report.violations.extend(ctx.violations)
    return report


def test_toy_read_set_extraction():
    import inspect

    rs = fm.analyze_source(textwrap.dedent(inspect.getsource(ToyOwnerCode.toy_build)))
    assert {"flag_var", "entry", "count_var"} <= rs.attrs
    assert rs.widgets == {"entry": {"line"}}
    assert rs.config == {"mode": ["off"]}
    assert rs.env == {"TOY_IN"}
    assert "append_log" in rs.calls
    assert {"last_env"} <= rs.writes
    assert rs.attr_defaults == {"flag_var": [False]}


def test_toy_config_flow_and_aliases_are_followed():
    src = textwrap.dedent('''
        def m(self):
            cfg = self.config
            a = cfg.get("alias_key", 1)
            b = getattr(self, "config", {}).get("getattr_key")
            c = (self.config.get("nested") or {}).get("inner")
            d = "in_key" in self.config
            e = self.config["sub_key"]
            helper(self.config)
            return os.environ.get("ENV_A"), os.getenv("ENV_B"), os.environ["ENV_C"]
    ''')
    rs = fm.analyze_source(src)
    assert {"alias_key", "getattr_key", "nested", "in_key", "sub_key"} <= set(rs.config)
    assert rs.env == {"ENV_A", "ENV_B", "ENV_C"}
    assert rs.config_passes == [("helper", 0)]


def test_toy_state_generation_is_seeded_and_covers_the_value_space():
    space = _toy_space()
    assert space.generate(7).summary() == space.generate(7).summary()
    other = _toy_space(seed=fm.DEFAULT_SEED + 1)
    assert [space.generate(i).summary() for i in range(20)] != [other.generate(i).summary() for i in range(20)]
    seen = {"absent": set(), "scalars": set(), "config": set(), "env": set(), "modes": set(), "bases": set()}
    for i in range(fm.DEFAULT_STATES):
        st = space.generate(i)
        seen["modes"].add(st.mode)
        seen["bases"].add(st.base)
        for name, value in st.attrs.items():
            if value is fm.MISSING:
                seen["absent"].add(name)
            elif not isinstance(value, fm._WIDGET_TYPES):
                seen["scalars"].add(repr(value))
        for key, value in st.config.items():
            seen["config"].add("absent" if value is fm.MISSING else ("default" if value == "off" else "alternate"))
        seen["env"] |= {v if v is not fm.MISSING else "unset" for v in st.env.values()}
    assert seen["absent"] >= {"flag_var", "entry", "count_var"}
    assert seen["scalars"] >= {repr(v) for v in fm.SCALARS}
    assert seen["config"] == {"absent", "default", "alternate"}
    assert "1" in seen["env"] and "unset" in seen["env"]
    assert seen["modes"] == {"chaos", "targeted"} and seen["bases"] == {"empty", "toy"}


def test_toy_identical_pair_passes():
    report = _run_toy(_toy_variant())
    assert report.ok, report.failure_text()
    assert report.states == fm.DEFAULT_STATES
    assert 0.2 < report.clean_fraction < 1.0  # both raising and clean states were exercised
    assert report.legacy_exceptions == report.new_exceptions


@pytest.mark.parametrize("variant", sorted(TOY_VARIANTS))
def test_toy_injected_difference_is_detected(variant):
    edit, field_name = TOY_VARIANTS[variant]
    report = _run_toy(_toy_variant(edit), stop_after=1)
    assert report.mismatches, f"{variant}: injected difference not detected in {report.states} states"
    fields = set(report.mismatches[0].fields)
    if field_name is not None:
        assert field_name in fields, (variant, fields)
    else:
        assert fields == {"result_order"}, fields
    text = report.failure_text()
    assert "replay:" in text and f"state #{report.mismatches[0].index}" in text


def test_text_masking_hides_side_specific_locations():
    import traceback

    try:
        raise ValueError("bad 0x7ffde1234567")
    except ValueError:
        tb = traceback.format_exc()
    assert fm.mask("X: " + tb) == "X: Traceback (most recent call last):\nValueError: bad 0x?\n"
    assert fm.canon({"k": ["see File \"a.py\", line 3"]}) == {"k": ['see File "<src>", line <n>']}


def test_fd_guard_refuses_integer_paths():
    with fm.FuzzContext("fdguard", backend=False):
        with pytest.raises(OSError):
            open(1)  # noqa: SIM115 - fuzzed True/1 must never close the test process's stdout
        with open(os.path.join(os.getcwd(), "ok.txt"), "w", encoding="utf-8") as fh:
            fh.write("x")
    os.fstat(1)


TOY_CONTRACT_MODULE = textwrap.dedent('''
    class ToyMixin:
        CONST = 1

        def helper(self):
            return 1

        def m(self):
            a = self.plain
            b = getattr(self, "g1", None)
            if hasattr(self, "g2"):
                c = self.g2
            d = hasattr(self, "g3") and self.g3.text()
            try:
                e = self.g4
            except AttributeError:
                e = None
            f = self.entry.text()
            self.log_it("x")
            self.helper()
            k = self.CONST
            self.local = 1
            g = self.local
            h = self.g5 if hasattr(self, "g5") else None
            return self.config.get("k")

        def _init_variables(self):
            self.from_init = 1
            for name in ("table_var",):
                setattr(self, name, 0)


    class Other:
        def n(self):
            return self.not_scanned
''')


def test_owner_contract_scanner_on_toy_module():
    scan = owner_contract.scan_source(TOY_CONTRACT_MODULE, "toy", ["ToyMixin"])
    assert scan.contract == {"plain", "entry", "log_it", "config"}
    assert {"g2", "g3", "g4", "g5"} <= set(scan.guarded)
    assert "local" in scan.locally_provided
    assert {"helper", "CONST", "m"} <= scan.class_provided
    assert {"from_init", "table_var"} <= scan.init_assigned
    assert scan.external == {"plain", "entry", "log_it", "config"}
    kinds = {s.kind for s in scan.sites("entry")} | {s.kind for s in scan.sites("log_it")}
    assert kinds == {"widget", "call"}


def test_import_hygiene_probe_on_toy_modules(tmp_path):
    (tmp_path / "translator_gui.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "toy_clean.py").write_text("import json\n", encoding="utf-8")
    (tmp_path / "toy_leaky.py").write_text("import translator_gui\n", encoding="utf-8")
    (tmp_path / "toy_qt_hard.py").write_text("from PySide6 import QtCore\n", encoding="utf-8")
    (tmp_path / "toy_qt_guarded.py").write_text(
        "try:\n    from PySide6 import QtCore\nexcept ImportError:\n    QtCore = None\n", encoding="utf-8")
    (tmp_path / "toy_py311.py").write_text(
        "try:\n    pass\nexcept* ValueError:\n    pass\n", encoding="utf-8")
    res = import_hygiene.check_imports(["toy_clean", "toy_leaky", "toy_qt_hard", "toy_qt_guarded"],
                                       src_dir=tmp_path)
    assert res["toy_clean"].ok, res["toy_clean"].describe()
    assert not res["toy_leaky"].ok and res["toy_leaky"].leaked == ["translator_gui"]
    assert not res["toy_qt_hard"].ok and "PySide6" in (res["toy_qt_hard"].error or "")
    assert res["toy_qt_hard"].qt_attempts
    assert res["toy_qt_guarded"].ok and res["toy_qt_guarded"].qt_attempts
    assert import_hygiene.parse_errors(tmp_path / "toy_clean.py") == []
    assert import_hygiene.parse_errors(tmp_path / "toy_py311.py")


def test_moved_registry_is_consistent():
    names = [m.name for m in mf.MOVED]
    assert len(names) == len(set(names))
    assert all(m.module in mf.MIXINS for m in mf.MOVED)
    assert not set(names) & mf.HOOK_NAMES
    for m in mf.MOVED:
        assert m.fuzz or m.reason, m.name
    assert set(mf.MIXINS) <= set(mf.SHARED_MODULES)


# ===========================================================================
# Shared fixtures for the desktop-backed tiers
# ===========================================================================


@pytest.fixture(scope="module")
def session():
    import importlib.util

    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed: the frozen desktop oracle and TranslatorGUI need it "
                    "(the toy self-tests above still ran)")
    sess = fm.session()
    try:
        sess.bundle  # noqa: B018 - loads the oracle
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    return sess


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def test_moved_registry_matches_legacy_oracle(session):
    names = session.legacy_names()
    missing = sorted({n for m in mf.FUZZED for n in m.legacy_names if n not in names})
    assert not missing, f"legacy oracle @ {session.bundle.sha[:12]} lacks {missing}"


def test_legacy_oracle_is_frozen_at_parent_commit(session):
    present = _u2_modules_present()
    if not present:
        pytest.skip("no U2 module exists yet: the oracle only has to be frozen before the first move")
    expected = os.environ.get("PARITY_LEGACY_SHA") or _git_head()
    if not expected:
        pytest.skip("git unavailable")
    if session.bundle.sha == expected:
        return
    # commits since the freeze that leave every frozen source untouched keep the oracle valid
    from parity import freeze_legacy

    manifest = session.bundle.manifest
    sources = {manifest["source_file"]: manifest["source_sha256"]}
    sources.update({info["source_file"]: info["source_sha256"] for info in manifest["externals"].values()})
    # U2+ oracles freeze the shared mixin modules too (their copies are what the legacy side runs)
    sources.update({info["source_file"]: info["source_sha256"]
                    for info in (manifest.get("frozen_mixins") or {}).values()})

    def digest_at(path):
        try:
            return freeze_legacy._sha256(freeze_legacy.git_show_text(expected, path))
        except subprocess.CalledProcessError:
            return None  # removed since the freeze

    changed = sorted(path for path, digest in sources.items() if digest_at(path) != digest)
    assert not changed, (
        f"tier D compares against legacy @ {session.bundle.sha[:12]} but {changed} changed by "
        f"{expected[:12]} (the step's parent commit; U2 modules present: {present}); re-run "
        f"python tests/parity/freeze_legacy.py (or set PARITY_LEGACY_SHA)")


#: real frozen methods used to prove the per-run reset is complete (legacy vs legacy)
SELF_CHECK = ("_get_environment_variables", "initialize_environment_variables", "_init_variables",
              "_init_config_state", "_sanitize_config_prompts", "_collect_live_settings",
              "_build_epub_compile_env", "_build_glossary_extraction_env")


@pytest.mark.parametrize("name", SELF_CHECK)
def test_fuzz_harness_is_deterministic_on_legacy_code(session, name):
    report = fm.fuzz_moved(mf.get(name), session, states=60, mode="legacy")
    assert report.ok, report.failure_text()
    assert report.states == 60


# ===========================================================================
# Tier D: legacy oracle vs working-tree desktop, >= 500 states per moved function
# ===========================================================================


@pytest.mark.parametrize("name", [m.name for m in mf.FUZZED])
def test_moved_function_matches_legacy(session, name):
    spec = mf.get(name)
    try:
        fm.check_available(spec, session)
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    report = fm.fuzz_moved(spec, session, check=False)
    assert report.ok, report.failure_text()
    wanted = min(fm.DEFAULT_STATES, fm.requested_states())
    assert report.states >= wanted, f"only {report.states} states ran (wanted {wanted})"


def _injected(original, transform):
    """A wrapper of *original* whose result passes through *transform* (rare-branch bug)."""
    def wrapper(self, *args, **kwargs):
        return transform(self, original(self, *args, **kwargs))
    wrapper.__name__ = original.__name__
    return wrapper


INJECTIONS = {
    # a value only some states produce
    "_get_output_mode": lambda self, mode: "text" if mode == "vision" else mode,
    # one env key changed only for gemini models, inside the 800-line builder
    "_get_environment_variables": lambda self, env: (
        dict(env, BATCH_SIZE="99") if "gemini" in str(getattr(self, "model_var", "")) else env),
}


@pytest.mark.parametrize("name", sorted(INJECTIONS))
def test_tier_d_catches_a_difference_injected_into_a_real_moved_method(session, monkeypatch, name):
    """Harness self-test on the real desktop path: the same driver must flag a rare-branch change."""
    spec = mf.get(name)
    try:
        fm.check_available(spec, session)
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    cls = session.mixin_classes()[spec.module]
    monkeypatch.setattr(cls, name, _injected(fm.resolve_python_mro(cls, name), INJECTIONS[name]))
    report = fm.fuzz_moved(spec, session, check=False, stop_after=1)
    assert report.mismatches, f"injected difference in {name} not detected in {report.states} states"
    assert "result" in report.mismatches[0].fields


# ===========================================================================
# MRO / duplicates / class attributes
# ===========================================================================


@pytest.fixture(scope="module")
def desktop_class(session):
    try:
        return session.translator_gui_class()
    except fm.Unavailable as exc:
        pytest.skip(str(exc))


def _gui_mixins(tg, shared):
    out = []
    for klass in tg.__mro__[1:]:
        module = klass.__module__ or ""
        if klass in shared or klass is object or module.startswith(("PySide6", "shiboken6", "Shiboken")):
            continue
        out.append(klass)
    return out


@pytest.mark.parametrize("module", sorted(mf.MIXINS))
def test_moved_names_live_only_in_their_mixin(session, desktop_class, module):
    if not _module_exists(module):
        pytest.skip(f"src/{module}.py does not exist yet")
    mixins = session.mixin_classes()
    cls = mixins.get(module)
    assert cls is not None, f"{module}.{mf.MIXINS[module]} missing"
    gui_mixins = _gui_mixins(desktop_class, set(mixins.values()))
    problems = []
    for spec in mf.by_module(module):
        if fm.resolve_python_mro(cls, spec.name) is fm.MISSING:
            if not spec.optional:
                problems.append(f"{spec.name}: not defined by {spec.mixin}")
            continue
        if spec.name in vars(desktop_class):
            problems.append(f"{spec.name}: TranslatorGUI still defines it (moved bodies must be deleted)")
        for gui in gui_mixins:
            if spec.name in vars(gui):
                problems.append(f"{spec.name}: also defined by GUI mixin {gui.__module__}.{gui.__name__}")
        if desktop_class.__dict__.get(spec.name) is None and \
                fm.resolve_python_mro(desktop_class, spec.name) is not fm.resolve_python_mro(cls, spec.name):
            problems.append(f"{spec.name}: TranslatorGUI's MRO does not resolve it to {spec.mixin}")
    assert not problems, "\n".join(problems)


def test_translator_gui_lists_shared_mixins_first(session, desktop_class):
    mixins = session.mixin_classes()
    if not mixins:
        pytest.skip("no shared mixin module exists yet")
    bases = list(desktop_class.__bases__)
    missing = [c.__name__ for c in mixins.values() if c not in bases]
    assert not missing, f"TranslatorGUI does not inherit {missing}"
    shared_idx = [bases.index(c) for c in mixins.values()]
    other_idx = [i for i, b in enumerate(bases) if b not in mixins.values()]
    assert max(shared_idx) < min(other_idx), [b.__name__ for b in bases]
    design_order = ["SettingsPersistenceMixin", "RunEnvMixin", "ConfigStateMixin"]
    present = [b.__name__ for b in bases if b.__name__ in design_order]
    assert present == [n for n in design_order if n in present], present


@pytest.mark.parametrize("name", [m.name for m in mf.CLASS_ATTRS])
def test_moved_class_attribute_matches_legacy(session, name):
    spec = mf.get(name)
    if not _module_exists(spec.module):
        pytest.skip(f"src/{spec.module}.py does not exist yet")
    cls = session.mixin_classes().get(spec.module)
    value = fm.resolve_python_mro(cls, name) if cls is not None else fm.MISSING
    assert value is not fm.MISSING, f"{spec.mixin}.{name} missing"
    # the frozen TranslatorGUI body, then (U2+ oracles) the frozen mixin copies
    legacy = fm.resolve_python_mro(session.legacy_class(), name)
    assert fm.canon(value) == fm.canon(legacy)


# ===========================================================================
# Tier I: import hygiene / Python 3.10
# ===========================================================================

_EXISTING_SHARED = [m for m in mf.SHARED_MODULES if _module_exists(m)]


@pytest.fixture(scope="module")
def hygiene():
    return import_hygiene.check_imports(_EXISTING_SHARED)


@pytest.mark.parametrize("module", mf.SHARED_MODULES)
def test_shared_module_import_hygiene(hygiene, module):
    if module not in hygiene:
        pytest.skip(f"src/{module}.py does not exist yet")
    res = hygiene[module]
    assert res.ok, res.describe()


@pytest.mark.parametrize("module", mf.SHARED_MODULES)
def test_shared_module_parses_as_python_310(module):
    if not _module_exists(module):
        pytest.skip(f"src/{module}.py does not exist yet")
    assert import_hygiene.parse_errors(SRC_DIR / f"{module}.py") == []


def test_shared_modules_import_on_python_310():
    python = import_hygiene.python310()
    if python is None:
        pytest.skip("set GLOSSARION_PY310 to a Python 3.10 interpreter to run the real 3.10 import probe")
    results = import_hygiene.check_imports(_EXISTING_SHARED, python=python)
    bad = [r.describe() for r in results.values() if not r.ok]
    assert not bad, "\n".join(bad)


# ===========================================================================
# Owner contract
# ===========================================================================


@pytest.fixture(scope="module")
def contract():
    if not any(_module_exists(m) for m in mf.MIXINS):
        pytest.skip("no shared mixin module exists yet")
    return owner_contract.scan_modules()


def _headless_owner_in_sandbox(config=None):
    with fm.FuzzContext("headless-contract", backend=True) as ctx:
        fm.patch_src_module_paths(ctx, extra_file_modules=[m for m in (*mf.MIXINS, "headless_owner")
                                                           if m in sys.modules])
        import headless_owner

        fm.patch_src_module_paths(ctx, extra_file_modules=[m for m in (*mf.MIXINS, "headless_owner")
                                                           if m in sys.modules])
        owner = headless_owner.HeadlessOwner(dict(config or {}), host=rt.RecordingHost(ctx.recorder))
        present = {n for n in dir(owner)}
    return owner, present, ctx.violations


def test_headless_owner_declares_the_scanned_contract(contract):
    if not _module_exists("headless_owner"):
        pytest.skip("src/headless_owner.py does not exist yet")
    import headless_owner

    declared = set(getattr(headless_owner, "OWNER_CONTRACT", ()))
    assert declared, "headless_owner.OWNER_CONTRACT is empty or missing"
    provided = {n for n in contract.unprovided
                if fm.resolve_python_mro(headless_owner.HeadlessOwner, n) is not fm.MISSING}
    undeclared = sorted(contract.unprovided - declared - provided)
    assert not undeclared, (
        "unguarded mixin reads missing from headless_owner.OWNER_CONTRACT:\n"
        + "\n".join(f"  {n}: {contract.sites(n)[0]}" for n in undeclared))


def test_headless_owner_satisfies_the_contract(contract):
    if not _module_exists("headless_owner"):
        pytest.skip("src/headless_owner.py does not exist yet")
    _owner, present, violations = _headless_owner_in_sandbox()
    assert not violations, violations
    missing = sorted(n for n in contract.contract if n not in present and n not in owner_contract.RUNTIME_ATTRS)
    assert not missing, (
        "HeadlessOwner({}) lacks attributes the shared mixins read unguarded:\n"
        + "\n".join(f"  {n}: {contract.sites(n)[0]}" for n in missing))


def test_booted_desktop_owner_satisfies_the_contract(session, contract):
    from parity import normalize, scenarios

    factory = fakes.make_legacy_owner_factory(session.bundle)
    scenario = scenarios.get("fresh_install")
    with normalize.CaptureContext(scenario, "contract-desktop") as ctx:
        owner = factory(scenario, ctx)
        present = set(dir(owner))
    data_names = {n for n in contract.contract
                  if all(s.kind != "call" for s in contract.sites(n))}
    missing = sorted(n for n in data_names if n not in present and n not in owner_contract.RUNTIME_ATTRS)
    assert not missing, (
        "a booted desktop owner lacks attributes the shared mixins read unguarded:\n"
        + "\n".join(f"  {n}: {contract.sites(n)[0]}" for n in missing))


# ===========================================================================
# Tier R: round trip
# ===========================================================================

from parity import scenarios as _scenarios  # noqa: E402

SCENARIOS = list(_scenarios.SCENARIO_NAMES)


@pytest.fixture(scope="module")
def restart_results(session):
    factory = fakes.make_legacy_owner_factory(session.bundle)
    return {
        name: rt.roundtrip(name, collect=rt.collect_from_saved_file, rebuild=rt.rebuild_by_restart(factory),
                           live_factory=factory, known=rt.LEGACY_RESTART_DIVERGENCES)
        for name in SCENARIOS
    }


def _assert_substantial(res):
    assert len(res.live.get("translation_env") or {}) > 100, res.text()
    assert len(res.rebuilt.get("translation_env") or {}) > 100, res.text()
    assert len(res.live.get("startup_env") or {}) > 100, res.text()


@pytest.mark.parametrize("name", SCENARIOS)
def test_roundtrip_helper_on_desktop_restart(restart_results, name):
    res = restart_results[name]
    assert res.ok, res.text()
    assert res.collected_keys > 400
    _assert_substantial(res)


def test_restart_divergence_list_is_current(restart_results):
    observed = set()
    for res in restart_results.values():
        for line in res.allowed:
            entry_key = line.split(":", 1)[0]
            entry, _, key = entry_key.partition("/")
            observed.add((entry, key))
    stale = sorted(set(rt.LEGACY_RESTART_DIVERGENCES) - observed)
    assert not stale, f"LEGACY_RESTART_DIVERGENCES entries no longer observed: {stale}"


def _require_tier_r():
    reason = rt.available()
    if reason:
        pytest.skip(reason)


@pytest.mark.parametrize("name", SCENARIOS)
def test_collect_live_settings_is_pure_and_matches_legacy_save(session, name):
    """R0: on a booted desktop owner, the extracted _collect_live_settings() returns exactly the
    config the frozen save_config() then writes into self.config, and changes nothing itself."""
    if not _module_exists("settings_persistence"):
        pytest.skip("src/settings_persistence.py does not exist yet")
    from parity import capture_golden, normalize

    factory = fakes.make_legacy_owner_factory(session.bundle)
    scenario = _scenarios.get(name)
    with normalize.CaptureContext(scenario, rt.SANDBOX_ENTRY) as ctx:
        owner = factory(scenario, ctx)
        rt._with_shared_mixins(owner)
        before = ctx.state(owner)
        collected = owner._collect_live_settings()
        after = ctx.state(owner)
        assert before == after, "_collect_live_settings() mutated the owner, its config or os.environ"
        session.bundle.methods.save_config(owner, show_message=False)
        problems = capture_golden.diff(ctx.norm(owner.config), ctx.norm(collected), limit=20)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", SCENARIOS)
def test_roundtrip_headless_matches_desktop_restart(session, name):
    _require_tier_r()
    res = rt.roundtrip(name, collect=rt.collect_with_mixin, rebuild=rt.rebuild_headless,
                       bundle=session.bundle, reference="restart")
    assert res.ok, res.text()
    _assert_substantial(res)


@pytest.mark.parametrize("name", SCENARIOS)
def test_roundtrip_headless_matches_live_desktop(session, name):
    _require_tier_r()
    res = rt.roundtrip(name, collect=rt.collect_with_mixin, rebuild=rt.rebuild_headless,
                       bundle=session.bundle, reference="live")
    assert res.ok, res.text()
    _assert_substantial(res)
