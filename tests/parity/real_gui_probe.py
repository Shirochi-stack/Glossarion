#!/usr/bin/env python3
"""Tier L (manual): compare the legacy boot replay with a REAL offscreen TranslatorGUI.

For each scenario a child process builds the live ``translator_gui.TranslatorGUI()``
with ``QT_QPA_PLATFORM=offscreen`` inside a sandbox laid out like
``normalize.Sandbox`` (``GLOSSARION_APP_DIR`` = sandbox/src, so CONFIG_FILE and
config backups stay in the sandbox; HOME/USERPROFILE/TEMP/cwd sandboxed;
``os.cpu_count`` = 8 like the capture harness; glossary migration and
UnifiedClient key pools stubbed). It applies the scenario's ``run_attrs`` and
records the owner's data attributes, config, ``_get_environment_variables``
result and the process environment right after startup, which are diffed against
the legacy golden (boot ``final`` attrs/config/env + the ``translation_env`` entry).

This validates that the frozen boot (``fakes.LEGACY_BOOT_PHASES``) reproduces
real desktop startup. Expected, allow-listed differences: paths derived from the
real script dir (``_get_app_dir()``/``base_dir`` on Windows), the real thread
executor, UpdateManager's ``last_update_check_time``, and environment variables
the real GUI process sets that the capture sandbox does not model (Qt/DPI).

``--headless`` builds ``headless_owner.HeadlessOwner`` from the same sandbox
config.json in a second child and diffs it against the real GUI directly: config,
translation env, startup environment and the ``authgem_auth`` project cache (the
widget attributes differ by design: shims vs Qt widgets). ``--config FILE`` runs
an ad-hoc config.json (JSON, ``<SANDBOX>`` placeholders allowed) on top of the
``fresh_install`` scenario; it implies ``--headless`` (there is no golden for it).

The live GUI import has the usual side effects of starting the desktop app
(DPI env, http_requests/Payloads size sweep, logs under src/); tiktoken keeps its
real cache so no encoding download is triggered. The developer's src/config.json
is never touched (CONFIG_FILE is asserted to be inside the sandbox). Usage::

    python tests/parity/real_gui_probe.py [--scenario NAME ...] [--headless]
    python tests/parity/real_gui_probe.py --config my_config.json [--config other.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = _TESTS_DIR.parent / "src"
for _p in (str(_TESTS_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: (section, key) pairs that legitimately differ between the sandboxed replay and the live GUI.
ALLOWED_DIFFERENCES = {
    ("translation_env", "EPUB_OUTPUT_DIR"),       # _get_app_dir(): real script dir vs sandbox src
    ("translation_env", "GLOSSARY_SHARED_DIR"),   # idem
    ("translation_env", "GLOSSARY_OUTPUT_BACKUP_DIR"),
    ("attrs", "base_dir"),
    ("attrs", "payloads_dir"),
    ("attrs", "_executor_workers"),               # real ThreadPool executor vs recorder
    ("attrs", "executor"),
    ("config", "last_update_check_time"),         # UpdateManager is not replayed
    ("startup_env", "GLOSSARION_WATCHDOG_DIR"),   # <pid>_<ms timestamp> (the harness fixes both)
    ("startup_env", "GLOSSARY_SHARED_DIR"),       # _get_app_dir(): real script dir vs sandbox src
    ("startup_env", "GLOSSARY_OUTPUT_BACKUP_DIR"),
    ("startup_env", "EPUB_OUTPUT_DIR"),
}

#: Startup-environment variables only a real GUI process / a real child process sets
#: (Qt platform + DPI scaling, the probe's own sandbox variables); never compared.
STARTUP_ENV_IGNORED_PREFIXES = ("QT_", "PYTHON")
STARTUP_ENV_IGNORED = {
    "GLOSSARION_APP_DIR", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
    "TIKTOKEN_CACHE_DIR",
}


def _plain(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _prepare_sandbox(scenario, root: str, config_override=None):
    from parity.normalize import Sandbox

    sandbox = Sandbox(Path(root))
    for rel, content in (scenario.get("files") or {}).items():
        path = sandbox.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sandbox.resolve(content), encoding="utf-8")
    config = config_override if config_override is not None else scenario.get("config")
    if config is not None:
        sandbox.config_file.write_text(
            json.dumps(sandbox.resolve(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return sandbox


def _authgem_state():
    module = sys.modules.get("authgem_auth")
    if module is None:
        return None
    return {
        "cached_project_id": {str(k): v for k, v in dict(getattr(module, "_cached_project_id", {}) or {}).items()},
        "project_set_by_gui": {str(k): v for k, v in dict(getattr(module, "_project_set_by_gui", {}) or {}).items()},
    }


def _child(mode: str, scenario_name: str, root: str, out_path: str, config_path: str = "") -> int:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.cpu_count = lambda: 8  # same as normalize.FIXED_CPU_COUNT
    from parity import scenarios

    scenario = scenarios.get(scenario_name)
    override = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else None
    sandbox = _prepare_sandbox(scenario, root, override)
    import glossary_paths
    import unified_api_client

    glossary_paths.migrate_all_legacy_glossary_files = lambda *a, **k: None
    for name in dir(unified_api_client.UnifiedClient):
        if name.startswith(("set_in_memory_", "clear_in_memory_")):
            setattr(unified_api_client.UnifiedClient, name, classmethod(lambda cls, *a, **k: None))
    if mode == "gui":
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])  # noqa: F841
        import translator_gui

        if not os.path.normcase(translator_gui.CONFIG_FILE).startswith(os.path.normcase(str(sandbox.root))):
            raise SystemExit(f"CONFIG_FILE {translator_gui.CONFIG_FILE} is outside the sandbox; aborting")
        env_before = dict(os.environ)
        owner = translator_gui.TranslatorGUI()
    else:
        import app_paths
        from headless_owner import HeadlessOwner

        if not os.path.normcase(app_paths.CONFIG_FILE).startswith(os.path.normcase(str(sandbox.root))):
            raise SystemExit(f"CONFIG_FILE {app_paths.CONFIG_FILE} is outside the sandbox; aborting")
        env_before = dict(os.environ)
        owner = HeadlessOwner.from_config_store(host=type("Host", (), {"log": staticmethod(print)})(),
                                                path=app_paths.CONFIG_FILE)
    env_after = dict(os.environ)
    authgem = _authgem_state()

    run_attrs = scenario.get("run_attrs") or {}
    if callable(run_attrs):
        run_attrs = run_attrs(sandbox)
    for key, value in sandbox.resolve(run_attrs).items():
        setattr(owner, key, value)
    env = owner._get_environment_variables(sandbox.path(scenario["input"]), owner.api_key_entry.text())

    changed = {k: v for k, v in env_after.items() if env_before.get(k) != v}
    removed = sorted(k for k in env_before if k not in env_after)
    result = {
        "attrs": {k: v for k, v in vars(owner).items() if k != "config" and _plain(v)},
        "config": json.loads(json.dumps(owner.config, default=str)),
        "translation_env": env,
        "startup_env": {"set": changed, "removed": removed, "after": env_after},
        "authgem": authgem,
        "run_attr_keys": sorted(run_attrs),
    }
    Path(out_path).write_text(repr(result), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)


def _placeholders(sandbox_root: Path):
    from parity.normalize import SANDBOX_TOKEN, PathPlaceholders

    return PathPlaceholders([
        (str(sandbox_root), SANDBOX_TOKEN),
        (str(SRC_DIR), "<SRC>"),
        (str(_TESTS_DIR.parent), "<REPO>"),
        (os.path.expanduser("~"), "<HOME>"),
    ])


def _startup_env_ignored(key: str) -> bool:
    return (key in STARTUP_ENV_IGNORED or key.startswith(STARTUP_ENV_IGNORED_PREFIXES)
            or ("startup_env", key) in ALLOWED_DIFFERENCES)


def _compare_startup_env(expected_set: dict, live: dict, cg) -> list:
    """Golden/reference boot env 'set' values vs the live process env after startup."""
    problems = []
    after = live["after"]
    for key in sorted(expected_set):
        if _startup_env_ignored(key):
            continue
        if key not in after:
            problems.append(f"startup_env/{key}: only in reference ({str(expected_set[key])[:80]!r})")
        elif cg.roundtrip(after[key]) != cg.roundtrip(expected_set[key]):
            problems.append(f"startup_env/{key}: reference={str(expected_set[key])[:80]!r} live={str(after[key])[:80]!r}")
    for key in sorted(live["set"]):
        if key not in expected_set and not _startup_env_ignored(key):
            problems.append(f"startup_env/{key}: only set by the live GUI ({str(live['set'][key])[:80]!r})")
    return problems


def _compare(scenario_name: str, live: dict, sandbox_root: Path) -> list:
    from parity import capture_golden as cg
    from parity import freeze_legacy
    from parity.normalize import normalize_value

    live = normalize_value(live, _placeholders(sandbox_root))
    golden = cg.load_golden(freeze_legacy.latest_sha(), scenario_name)
    final = golden["entries"]["boot"]["final"]
    replay = {
        "attrs": final["attrs"],
        "config": final["config"],
        "translation_env": golden["entries"]["translation_env"]["result"],
    }
    problems = []
    run_attr_keys = set(live.get("run_attr_keys", ()))
    for section in ("attrs", "config", "translation_env"):
        a, b = replay[section], live[section]
        for key in sorted(set(a) | set(b)):
            if (section, key) in ALLOWED_DIFFERENCES:
                continue
            if section == "attrs" and key in run_attr_keys:
                continue  # set after boot by the harness on both sides (replay boot state predates them)
            if section == "attrs" and key not in b:
                continue  # live dump keeps JSON-able values only (widgets/objects are skipped)
            if section == "attrs" and key not in a and not key.endswith("_var"):
                continue  # live-only GUI internals (styles, zoom steps, catalog caches, ...)
            if key not in a:
                problems.append(f"{section}/{key}: only in live GUI ({str(b[key])[:80]!r})")
            elif key not in b:
                problems.append(f"{section}/{key}: only in replay ({str(a[key])[:80]!r})")
            elif cg.roundtrip(a[key]) != cg.roundtrip(b[key]):
                problems.append(f"{section}/{key}: replay={str(a[key])[:80]!r} live={str(b[key])[:80]!r}")
    problems += _compare_startup_env(final["env"]["set"], live["startup_env"], cg)
    return problems


def _compare_headless(live: dict, headless: dict, sandbox_root: Path) -> list:
    """Real GUI vs HeadlessOwner built from the same config.json (no golden involved)."""
    from parity import capture_golden as cg
    from parity.normalize import normalize_value

    ph = _placeholders(sandbox_root)
    live, headless = normalize_value(live, ph), normalize_value(headless, ph)
    problems = []
    for section in ("config", "translation_env"):
        a, b = live[section], headless[section]
        for key in sorted(set(a) | set(b)):
            if (section, key) in ALLOWED_DIFFERENCES:
                continue
            if key not in b:
                problems.append(f"{section}/{key}: only in live GUI ({str(a[key])[:80]!r})")
            elif key not in a:
                problems.append(f"{section}/{key}: only in HeadlessOwner ({str(b[key])[:80]!r})")
            elif cg.roundtrip(a[key]) != cg.roundtrip(b[key]):
                problems.append(f"{section}/{key}: gui={str(a[key])[:80]!r} headless={str(b[key])[:80]!r}")
    gui_env, head_env = live["startup_env"], headless["startup_env"]
    for key in sorted(set(gui_env["set"]) | set(head_env["set"])):
        if _startup_env_ignored(key):
            continue
        a, b = gui_env["after"].get(key), head_env["after"].get(key)
        if a != b:
            problems.append(f"startup_env/{key}: gui={str(a)[:80]!r} headless={str(b)[:80]!r}")
    if (live.get("authgem") or {}) != (headless.get("authgem") or {}):
        if live.get("authgem") or headless.get("authgem"):  # None (never imported) == empty cache
            problems.append(f"authgem_auth: gui={live.get('authgem')!r} headless={headless.get('authgem')!r}")
    return problems


def _run_child(mode, name, root, out, config_path, env):
    args = [sys.executable, str(Path(__file__).resolve()), "--child", mode, name, str(root), str(out)]
    if config_path:
        args.append(str(config_path))
    return subprocess.run(
        args, cwd=str(root / "cwd"), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600,
    )


def _child_env(name, root):
    from parity import scenarios

    env = dict(os.environ)
    env.update({
        "GLOSSARION_APP_DIR": str(root / "src"),
        "HOME": str(root / "home"),
        "USERPROFILE": str(root / "home"),
        "APPDATA": str(root / "appdata"),
        "LOCALAPPDATA": str(root / "localappdata"),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "PYTHONIOENCODING": "utf-8",
        "QT_QPA_PLATFORM": "offscreen",
        # keep tiktoken's real cache: a sandboxed TEMP would make the live GUI
        # re-download cl100k_base (and http_logger would log it under src/)
        "TIKTOKEN_CACHE_DIR": os.environ.get("TIKTOKEN_CACHE_DIR")
        or str(Path(tempfile.gettempdir()) / "data-gym-cache"),
    })
    for key in [k for k in env if k.startswith("GLOSSARION_") and k != "GLOSSARION_APP_DIR"]:
        env.pop(key)
    for key, value in (scenarios.get(name).get("env") or {}).items():
        env[key] = str(value).replace("<SANDBOX>", str(root))
    return env


def _fresh_root(base: Path) -> Path:
    shutil.rmtree(base, ignore_errors=True)
    root = base / "sb"
    root.mkdir(parents=True, exist_ok=True)
    (root / "cwd").mkdir(parents=True, exist_ok=True)
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare legacy boot replay with a real offscreen TranslatorGUI")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--headless", action="store_true",
                        help="compare the real GUI with HeadlessOwner (same config.json) instead of the golden")
    parser.add_argument("--config", action="append", default=[],
                        help="ad-hoc config.json on top of fresh_install (implies --headless)")
    parser.add_argument("--child", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return _child(*args.child)

    from parity import scenarios
    from parity.normalize import sandbox_base_dir

    runs = [(name, "") for name in (args.scenario or list(scenarios.SCENARIO_NAMES))] if not args.config else []
    runs += [("fresh_install", str(Path(p).resolve())) for p in args.config]
    headless = args.headless or bool(args.config)
    failures = 0
    for name, config_path in runs:
        label = f"{name} + {Path(config_path).name}" if config_path else name
        # same sandbox path as the capture of the translation_env entry, so values
        # derived from hashed absolute paths (subtitle work dirs) are comparable
        base = sandbox_base_dir() / name / "translation_env"
        results = {}
        for mode in (("gui", "headless") if headless else ("gui",)):
            root = _fresh_root(base)
            out = sandbox_base_dir() / f"real_gui_{name}_{mode}.repr"
            out.unlink(missing_ok=True)
            proc = _run_child(mode, name, root, out, config_path, _child_env(name, root))
            if not out.exists():
                print(f"{label:40s} {mode.upper()} CHILD FAILED (exit {proc.returncode})\n{proc.stderr[-2000:]}")
                break
            results[mode] = ast.literal_eval(out.read_text(encoding="utf-8"))
            out.unlink(missing_ok=True)
        if len(results) != (2 if headless else 1):
            failures += 1
            shutil.rmtree(base, ignore_errors=True)
            continue
        if headless:
            problems = _compare_headless(results["gui"], results["headless"], root)
        else:
            problems = _compare(name, results["gui"], root)
        failures += bool(problems)
        kind = "GUI vs HeadlessOwner" if headless else "GUI vs replay"
        print(f"{label:40s} {kind}: {'MATCH' if not problems else f'DIFF ({len(problems)})'}")
        for line in problems[:30]:
            print(f"    {line}")
        shutil.rmtree(base, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
