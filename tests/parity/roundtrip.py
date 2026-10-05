#!/usr/bin/env python3
"""Tier R: live desktop owner -> saved settings -> rebuilt owner gives the same run env.

The mobile app never sees desktop widgets: it builds a ``HeadlessOwner`` from a
config dict. Tier R proves that this is equivalent to the desktop *after a
save*::

    live    = desktop owner booted from a golden scenario (frozen legacy boot)
    config  = collect(live)        # SettingsPersistenceMixin._collect_live_settings
    rebuilt = rebuild(config)      # HeadlessOwner(config, api_key=..., model=None)
    compare(evaluate(live), evaluate(rebuilt))

``evaluate`` captures, in this order: the process env delta after boot/startup
(``startup_env``), the translation env dict (``translation_env``) and the
glossary env pairs (``glossary_env_mappings``), each normalised with the
context's path placeholders (live and rebuilt run in different sandboxes).

Collect/rebuild strategies are pluggable so the helper is useful before and
after U2:

* ``collect_with_mixin`` + ``rebuild_headless``: the real tier R (needs
  ``settings_persistence`` and ``headless_owner``); divergences allowed only via
  ``KNOWN_ROUNDTRIP_DIVERGENCES`` (target: empty);
* ``collect_from_saved_file`` + ``rebuild_by_restart(factory)``: desktop
  restart from the config.json its startup ``save_config`` wrote (runs today;
  validates the helper and records which keys a desktop restart does not
  reproduce in ``LEGACY_RESTART_DIVERGENCES``).

CLI::

    python tests/parity/roundtrip.py [--restart] [--scenario NAME ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import capture_golden, fakes, moved_functions, normalize  # noqa: E402

ENTRIES = ("startup_env", "translation_env", "glossary_env_mappings")

#: Sandbox entry name shared by every context of one round trip: live, reference and
#: rebuilt owners then see the same absolute paths (some desktop helpers hash them,
#: e.g. SUBTITLE_WORK_DIR = <work>/<stem>_<sha1(abs path)>).
SANDBOX_ENTRY = "roundtrip"

#: (entry, key) -> reason. Allowed differences between ``HeadlessOwner`` built from the
#: collected settings and the desktop reference. Target: empty. A key ``'*'`` allows
#: every key of the entry (never use it for translation_env).
KNOWN_ROUNDTRIP_DIVERGENCES: dict = {}

_FLOAT_TEXT = ("save_config stores safe_float()/safe_int() values, so a session started from a "
               "saved config exports '10.0' where the first session exported the raw default '10'")

#: (entry, key) -> reason. What a desktop *restart* (boot from the config.json written by
#: the first boot's startup save_config) does not reproduce: desktop facts, found by the
#: helper self-test (test_parity_tiers.py::test_roundtrip_helper_on_desktop_restart) and
#: listed in tests/parity/DISCREPANCIES.md. Mobile builds its owner from saved settings,
#: so it matches the desktop *after* a restart on these keys, not its first session.
LEGACY_RESTART_DIVERGENCES: dict = {
    ("startup_env", "CONNECT_TIMEOUT"): _FLOAT_TEXT,
    ("startup_env", "READ_TIMEOUT"): _FLOAT_TEXT,
    ("startup_env", "IMAGE_CHUNK_OVERLAP_PERCENT"): _FLOAT_TEXT,
    ("translation_env", "IMAGE_CHUNK_OVERLAP_PERCENT"): _FLOAT_TEXT,
    ("translation_env", "SEND_INTERVAL_SECONDS"): (
        "delay_entry shows str(config['delay']); save_config stores safe_float(delay), "
        "so '5' becomes '5.0' after a restart"),
    ("translation_env", "GLOSSARY_TRANSLATION_PROMPT"): (
        "first session exports the built-in default (init block: config.get(key, default)); the "
        "startup save_config's _glossary_env_mappings writes config[key] = config.get(key, '') or '' "
        "for the missing key, so every later session exports ''"),
    ("translation_env", "GLOSSARY_FORMAT_INSTRUCTIONS"): (
        "same as GLOSSARY_TRANSLATION_PROMPT: '' is persisted by the first startup save_config"),
    ("startup_env", "EXTRACTION_MODE"): (
        "save_config rewrites config['extraction_mode'] from file_filtering_level_var "
        "(extraction-mode compatibility block), so a restart starts from the rewritten value"),
}


@dataclass
class RoundTrip:
    scenario: str
    reference: str = "live"
    live: dict = field(default_factory=dict)          # the reference evaluation (live or restart)
    rebuilt: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    allowed: list = field(default_factory=list)
    collected_keys: int = 0
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.errors

    def text(self, limit=25) -> str:
        lines = [f"{self.scenario} (vs {self.reference}): {len(self.problems)} problems, "
                 f"{len(self.allowed)} allowed, {self.collected_keys} collected keys"]
        lines += [f"  ERROR {e}" for e in self.errors]
        lines += [f"  {p}" for p in self.problems[:limit]]
        if len(self.problems) > limit:
            lines.append(f"  ... {len(self.problems) - limit} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# collect strategies: live owner -> config dict
# ---------------------------------------------------------------------------


def _with_shared_mixins(owner):
    """Give a legacy owner the LIVE shared mixins (the new code under test).

    Oracle without mixins (frozen before U2): the live mixins go *after* its frozen methods
    (desktop precedence: the frozen bodies win, names that only exist in the mixins come from
    them). Oracle with frozen mixin copies (U2+): the live mixin attributes that the frozen
    TranslatorGUI body (and the recorders) do not define are put in front of the frozen
    copies, so ``_collect_live_settings()`` etc. run the working-tree code, not the oracle's.
    """
    mixins = []
    for module_name, cls_name in moved_functions.MIXINS.items():
        if (SRC_DIR / f"{module_name}.py").is_file():
            module = __import__(module_name)
            cls = getattr(module, cls_name, None)
            if cls is not None and cls not in type(owner).__mro__:
                mixins.append(cls)
    if not mixins:
        return owner
    names = {cls.__name__ for cls in mixins}
    mro = type(owner).__mro__
    frozen = [k for k in mro if k.__name__ in names and (k.__module__ or "").startswith("parity_legacy_")]
    if not frozen:
        owner.__class__ = type(type(owner).__name__, (type(owner), *mixins), {"__module__": __name__})
        return owner
    body = next((k for k in mro if k.__name__ == "LegacyMethods"), None)
    keep = set(vars(body)) if body is not None else set()
    keep |= set(vars(type(owner)))  # recorders
    ns = {"__module__": __name__}
    for cls in mixins:  # the shared mixins define disjoint names (tests/test_headless_owner.py)
        for name, value in vars(cls).items():
            if not name.startswith("__") and name not in keep:
                ns[name] = value
    owner.__class__ = type(type(owner).__name__, (type(owner),), ns)
    return owner


def collect_with_mixin(owner, ctx) -> dict:
    """``SettingsPersistenceMixin._collect_live_settings()`` on the live owner (side-effect free)."""
    _with_shared_mixins(owner)
    return owner._collect_live_settings()


def collect_from_saved_file(owner, ctx) -> dict:
    """config.json written by the live owner's startup save_config (decrypted)."""
    import api_key_encryption

    path = ctx.sandbox.config_file
    if not path.exists():
        raise FileNotFoundError(f"{path} was not written by the live boot")
    return api_key_encryption.decrypt_config(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# rebuild strategies: config dict -> owner (inside a fresh context)
# ---------------------------------------------------------------------------


class RecordingHost:
    """Minimal JobHost for HeadlessOwner: log/emit/ask are recorded, ask answers None."""

    def __init__(self, recorder):
        self._recorder = recorder

    def log(self, message, *args, **kwargs):
        self._recorder.record("host.log", (message,) + args, kwargs)

    def emit(self, kind, **data):
        self._recorder.record("host.emit", (kind,), data)

    def ask(self, question, *args, **kwargs):
        self._recorder.record("host.ask", (question,) + args, kwargs)
        return None

    def is_stop_requested(self):
        return False

    def is_graceful_stop(self):
        return False


def rebuild_headless(config, scenario, ctx, *, api_key="", model=None):
    """``HeadlessOwner(config, host=..., api_key=..., model=...)`` in the rebuilt sandbox."""
    from parity import fuzz_moved

    fuzz_moved.patch_src_module_paths(
        ctx, extra_file_modules=[m for m in (*moved_functions.MIXINS, "headless_owner") if m in sys.modules])
    import headless_owner

    fuzz_moved.patch_src_module_paths(
        ctx, extra_file_modules=[m for m in (*moved_functions.MIXINS, "headless_owner") if m in sys.modules])
    return headless_owner.HeadlessOwner(config, host=RecordingHost(ctx.recorder), api_key=api_key, model=model)


def rebuild_by_restart(factory):
    """Rebuild = a full desktop boot by *factory* from a config.json holding *config*."""
    def rebuild(config, scenario, ctx, *, api_key="", model=None):
        return factory(scenario, ctx)
    rebuild.restart = True
    return rebuild


# ---------------------------------------------------------------------------
# evaluation / comparison
# ---------------------------------------------------------------------------


def _translation_env(owner, path, api_key, headless: bool):
    if headless:
        try:
            import run_env
            if hasattr(run_env, "build_translation_env"):
                return run_env.build_translation_env(owner, path, api_key)
        except ImportError:
            pass
    return owner._get_environment_variables(path, api_key)


def _startup(owner, headless: bool):
    if not headless:
        return None  # the desktop boot already ran initialize_environment_variables
    try:
        import run_env
        if hasattr(run_env, "build_startup_env"):
            return run_env.build_startup_env(owner)
    except ImportError:
        pass
    return owner.initialize_environment_variables()


def _pairs_to_dict(value):
    if isinstance(value, (list, tuple)) and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value):
        return {str(k): v for k, v in value}
    return value


def evaluate(owner, scenario, ctx, *, headless: bool = False, entries=ENTRIES) -> dict:
    out = {}
    path = ctx.sandbox.path(scenario["input"])
    api_key = owner.api_key_entry.text() if hasattr(owner, "api_key_entry") else ""
    for entry in entries:
        try:
            if entry == "startup_env":
                _startup(owner, headless)
                value = normalize.env_delta(ctx.baseline_env, normalize.effective_env())["set"]
            elif entry == "translation_env":
                value = _translation_env(owner, path, api_key, headless)
            elif entry == "glossary_env_mappings":
                value = _pairs_to_dict(owner._glossary_env_mappings())
            else:
                raise KeyError(entry)
            out[entry] = ctx.norm(value)
        except Exception as exc:
            out[entry] = {"__error__": f"{type(exc).__name__}: {exc}"}
    return out


def compare(live: dict, rebuilt: dict, known: dict) -> tuple:
    """(problems, allowed) between two ``evaluate`` results."""
    problems, allowed = [], []
    for entry in sorted(set(live) | set(rebuilt)):
        a, b = live.get(entry), rebuilt.get(entry)
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
                if va == vb:
                    continue
                line = f"{entry}/{key}: live={str(va)[:120]!r} rebuilt={str(vb)[:120]!r}"
                reason = known.get((entry, key)) or (known.get((entry, "*")) if entry != "translation_env" else None)
                (allowed if reason else problems).append(line + (f"  [allowed: {reason}]" if reason else ""))
        elif a != b:
            line = f"{entry}: live={str(a)[:160]!r} rebuilt={str(b)[:160]!r}"
            reason = known.get((entry, "*"))
            (allowed if reason else problems).append(line)
    return problems, allowed


def _rebased(config: dict, ctx) -> dict:
    """Config collected in the live sandbox with its root replaced by the <SANDBOX> token."""
    from parity import fuzz_moved

    ph = normalize.PathPlaceholders([(str(ctx.sandbox.root), normalize.SANDBOX_TOKEN)])
    return fuzz_moved.map_strings(copy.deepcopy(config), ph.apply)


def default_known(reference: str, rebuild) -> dict:
    """Allowed divergences: a headless rebuild compared with the *live* desktop may also
    differ where the desktop's own restart does (LEGACY_RESTART_DIVERGENCES)."""
    known = dict(KNOWN_ROUNDTRIP_DIVERGENCES)
    if reference == "live" or getattr(rebuild, "restart", False):
        known.update(LEGACY_RESTART_DIVERGENCES)
    return known


def roundtrip(scenario_name: str, *, collect, rebuild, live_factory=None, known=None,
              entries=ENTRIES, bundle=None, reference: str = "live") -> RoundTrip:
    """Run one scenario through live boot -> collect -> rebuild and compare the envs.

    *reference* ``'live'`` compares the rebuilt owner with the live desktop session;
    ``'restart'`` compares it with a desktop booted from the same collected config
    (the strict "mobile == desktop after save + restart" statement).
    """
    from parity import freeze_legacy, scenarios

    if reference not in ("live", "restart"):
        raise ValueError(reference)
    scenario = scenarios.get(scenario_name)
    if live_factory is None:
        bundle = bundle or freeze_legacy.load_legacy()
        live_factory = fakes.make_legacy_owner_factory(bundle)
    known = default_known(reference, rebuild) if known is None else known
    result = RoundTrip(scenario_name, reference=reference)
    with normalize.CaptureContext(scenario, SANDBOX_ENTRY) as ctx:
        live = live_factory(scenario, ctx)
        capture_golden._apply_run_attrs(live, scenario, ctx)
        api_key = live.api_key_entry.text() if hasattr(live, "api_key_entry") else ""
        result.live = evaluate(live, scenario, ctx, entries=entries)
        try:
            collected = collect(live, ctx)
        except Exception as exc:
            result.errors.append(f"collect failed: {type(exc).__name__}: {exc}")
            return result
        result.collected_keys = len(collected)
        token_config = _rebased(collected, ctx)
    rebuilt_scenario = dict(scenario, config=token_config)
    if reference == "restart":
        with normalize.CaptureContext(rebuilt_scenario, SANDBOX_ENTRY) as ctx:
            ref_owner = live_factory(rebuilt_scenario, ctx)
            capture_golden._apply_run_attrs(ref_owner, rebuilt_scenario, ctx)
            result.live = evaluate(ref_owner, rebuilt_scenario, ctx, entries=entries)
    headless = not getattr(rebuild, "restart", False)
    with normalize.CaptureContext(rebuilt_scenario, SANDBOX_ENTRY) as ctx:
        try:
            owner = rebuild(ctx.sandbox.resolve(copy.deepcopy(token_config)), rebuilt_scenario, ctx,
                            api_key=api_key, model=None)
        except Exception as exc:
            result.errors.append(f"rebuild failed: {type(exc).__name__}: {exc}")
            return result
        capture_golden._apply_run_attrs(owner, rebuilt_scenario, ctx)
        result.rebuilt = evaluate(owner, rebuilt_scenario, ctx, headless=headless, entries=entries)
    result.problems, result.allowed = compare(result.live, result.rebuilt, known)
    # an entry that raised on both sides compares equal: never let that pass silently
    for label, values in (("reference", result.live), ("rebuilt", result.rebuilt)):
        for entry, value in values.items():
            if isinstance(value, dict) and "__error__" in value:
                result.errors.append(f"{label} {entry}: {value['__error__']}")
    return result


def available() -> str | None:
    """None when the real tier R can run, else the skip reason."""
    missing = [m for m in ("settings_persistence", "headless_owner") if not (SRC_DIR / f"{m}.py").is_file()]
    if missing:
        return f"{', '.join(f'src/{m}.py' for m in missing)} not written yet (U2)"
    return None


def main(argv=None) -> int:
    from parity import freeze_legacy, scenarios

    parser = argparse.ArgumentParser(description="Tier R round trip")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--restart", action="store_true", help="desktop restart variant (no U2 modules needed)")
    parser.add_argument("--reference", choices=("live", "restart"), default="live",
                        help="headless variant: compare with the live session or a desktop restart")
    args = parser.parse_args(argv)
    names = args.scenario or list(scenarios.SCENARIO_NAMES)
    bundle = freeze_legacy.load_legacy()
    factory = fakes.make_legacy_owner_factory(bundle)
    reference = "live"
    if args.restart:
        collect, rebuild = collect_from_saved_file, rebuild_by_restart(factory)
    else:
        reason = available()
        if reason:
            print(f"SKIP: {reason}")
            return 0
        collect, rebuild, reference = collect_with_mixin, rebuild_headless, args.reference
    failures = 0
    for name in names:
        res = roundtrip(name, collect=collect, rebuild=rebuild, live_factory=factory, reference=reference)
        print(res.text())
        failures += not res.ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
