#!/usr/bin/env python3
"""Tier L (manual): compare the legacy boot replay with a REAL offscreen TranslatorGUI.

For each scenario a child process builds the live ``translator_gui.TranslatorGUI()``
with ``QT_QPA_PLATFORM=offscreen`` inside a sandbox laid out like
``normalize.Sandbox`` (``GLOSSARION_APP_DIR`` = sandbox/src, so CONFIG_FILE and
config backups stay in the sandbox; HOME/USERPROFILE/TEMP/cwd sandboxed;
``os.cpu_count`` = 8 like the capture harness; glossary migration and
UnifiedClient key pools stubbed). It applies the scenario's ``run_attrs`` and
records the owner's data attributes, config and ``_get_environment_variables``
result, which are diffed against the legacy golden (boot ``final`` + the
``translation_env`` entry).

This validates that the frozen boot (``fakes.LEGACY_BOOT_PHASES``) reproduces
real desktop startup. Expected, allow-listed differences: paths derived from the
real script dir (``_get_app_dir()``/``base_dir`` on Windows), the real thread
executor, UpdateManager's ``last_update_check_time``.

The live GUI import has the usual side effects of starting the desktop app
(DPI env, http_requests/Payloads size sweep, logs under src/); tiktoken keeps its
real cache so no encoding download is triggered. The developer's src/config.json
is never touched (CONFIG_FILE is asserted to be inside the sandbox). Usage::

    python tests/parity/real_gui_probe.py [--scenario NAME ...]
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
}


def _child(scenario_name: str, root: str, out_path: str) -> int:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.cpu_count = lambda: 8  # same as normalize.FIXED_CPU_COUNT
    from parity import scenarios
    from parity.normalize import Sandbox

    scenario = scenarios.get(scenario_name)
    sandbox = Sandbox(Path(root))
    for rel, content in (scenario.get("files") or {}).items():
        path = sandbox.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sandbox.resolve(content), encoding="utf-8")
    if scenario.get("config") is not None:
        sandbox.config_file.write_text(
            json.dumps(sandbox.resolve(scenario["config"]), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841
    import glossary_paths
    import unified_api_client

    glossary_paths.migrate_all_legacy_glossary_files = lambda *a, **k: None
    for name in dir(unified_api_client.UnifiedClient):
        if name.startswith(("set_in_memory_", "clear_in_memory_")):
            setattr(unified_api_client.UnifiedClient, name, classmethod(lambda cls, *a, **k: None))
    import translator_gui

    if not os.path.normcase(translator_gui.CONFIG_FILE).startswith(os.path.normcase(str(sandbox.root))):
        raise SystemExit(f"CONFIG_FILE {translator_gui.CONFIG_FILE} is outside the sandbox; aborting")
    gui = translator_gui.TranslatorGUI()

    run_attrs = scenario.get("run_attrs") or {}
    if callable(run_attrs):
        run_attrs = run_attrs(sandbox)
    for key, value in sandbox.resolve(run_attrs).items():
        setattr(gui, key, value)
    env = gui._get_environment_variables(sandbox.path(scenario["input"]), gui.api_key_entry.text())

    def plain(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    result = {
        "attrs": {k: v for k, v in vars(gui).items() if k != "config" and plain(v)},
        "config": json.loads(json.dumps(gui.config, default=str)),
        "translation_env": env,
        "run_attr_keys": sorted(run_attrs),
    }
    Path(out_path).write_text(repr(result), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)


def _compare(scenario_name: str, live: dict, sandbox_root: Path) -> list:
    from parity import capture_golden as cg
    from parity import freeze_legacy
    from parity.normalize import SANDBOX_TOKEN, PathPlaceholders, normalize_value

    placeholders = PathPlaceholders([
        (str(sandbox_root), SANDBOX_TOKEN),
        (str(SRC_DIR), "<SRC>"),
        (str(_TESTS_DIR.parent), "<REPO>"),
        (os.path.expanduser("~"), "<HOME>"),
    ])
    live = normalize_value(live, placeholders)
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
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare legacy boot replay with a real offscreen TranslatorGUI")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--child", nargs=3, metavar=("SCENARIO", "ROOT", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return _child(*args.child)

    from parity import scenarios
    from parity.normalize import sandbox_base_dir

    names = args.scenario or list(scenarios.SCENARIO_NAMES)
    failures = 0
    for name in names:
        # same sandbox path as the capture of the translation_env entry, so values
        # derived from hashed absolute paths (subtitle work dirs) are comparable
        base = sandbox_base_dir() / name / "translation_env"
        shutil.rmtree(base, ignore_errors=True)
        root = base / "sb"
        root.mkdir(parents=True, exist_ok=True)
        out = sandbox_base_dir() / f"real_gui_{name}.repr"
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
        for key, value in (scenarios.get(name).get("env") or {}).items():
            env[key] = str(value).replace("<SANDBOX>", str(root))
        (root / "cwd").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", name, str(root), str(out)],
            cwd=str(root / "cwd"), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if not out.exists():
            failures += 1
            print(f"{name:34s} CHILD FAILED (exit {proc.returncode})\n{proc.stderr[-2000:]}")
            continue
        problems = _compare(name, ast.literal_eval(out.read_text(encoding="utf-8")), root)
        failures += bool(problems)
        print(f"{name:34s} {'MATCH' if not problems else f'DIFF ({len(problems)})'}")
        for line in problems[:30]:
            print(f"    {line}")
        shutil.rmtree(base, ignore_errors=True)
        out.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
