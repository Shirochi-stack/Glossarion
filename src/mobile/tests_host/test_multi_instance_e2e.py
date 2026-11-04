"""Host test: two translation jobs at once, instance 1 and instance 2, through the real pipeline (design §6).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_multi_instance_e2e.py

``multi_instance_scenario.main`` runs in a subprocess (bootstrap into temp ``FLET_APP_STORAGE_*`` dirs, like
``test_e2e_offline``): the instance runtime is installed for the life of that process only. The scenario checks
the two jobs ran concurrently, each with its own key, environment, argv, stdout and stop flag, that stopping one left
the other running, that their outputs are separate and that the unified glossary kept both books' entries; this
test also reads the result's details.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SCENARIO = Path(__file__).resolve().parent / "multi_instance_scenario.py"
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py (CI: prepare)
DEPENDENCIES = ("ebooklib", "openai", "httpx", "lxml", "bs4", "tiktoken")
CHECKS = ["setup", "concurrent_run", "stop_one_instance", "keys_per_instance", "process_state_per_instance",
          "logs_per_instance", "outputs_separate", "unified_glossary_merged", "hygiene"]


def _missing(*modules: str) -> list:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def run_scenario(tmp_path: Path, script: Path, *args: str) -> tuple:
    """``script``'s ``main`` in a subprocess bootstrapped into temp ``FLET_APP_STORAGE_*`` dirs; asserts it passed and
    returns (its result, the storage dirs)."""
    storage = {name: tmp_path / name for name in ("data", "cache", "temp")}
    for directory in storage.values():
        directory.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLET_", "GLOSSARION_"))}
    env.update({f"FLET_APP_STORAGE_{name.upper()}": str(path) for name, path in storage.items()})
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    result_file = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(result_file), *args],
        cwd=str(APP_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500,
    )
    tail = (proc.stderr or "")[-6000:]
    assert result_file.is_file(), f"no result (exit {proc.returncode}):\n{tail}"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["ok"] and proc.returncode == 0, json.dumps(
        {k: result.get(k) for k in ("checks", "error", "traceback")}, indent=1, ensure_ascii=False) + "\n" + tail
    return result, storage


@pytest.mark.skipif(not SELFTEST_EPUB.is_file(), reason="run tools/prepare_assets.py first")
@pytest.mark.skipif(bool(_missing(*DEPENDENCIES)), reason=f"backend dependencies missing: {_missing(*DEPENDENCIES)}")
def test_two_books_translate_at_once_in_instances_1_and_2(tmp_path):
    result, storage = run_scenario(tmp_path, SCENARIO)
    assert list(result["checks"]) == CHECKS and set(result["checks"].values()) == {"pass"}

    detail = result["detail"]
    assert detail["runtime_installed_by_instance_1"] is False  # instance 1 alone never installs the runtime
    assert detail["parked_a"] >= 1  # book A was mid-run (a chapter request in flight) while book B ran and stopped
    requests = detail["requests"]
    assert requests["keys_A"] == ["sk-glossarion-instance-one"] and requests["keys_B"] == ["sk-glossarion-instance-two"]
    assert requests["A"] >= 12 and requests["B"] >= 5
    assert detail["argv_seen"]["instance_2"] >= 1
    assert detail["progress"]["A"]["completed"] == 12 and detail["progress"]["B"]["completed"] == 5
    assert not detail["unified"]["missing"] and detail["unified"]["entries"] >= 9
    assert detail["instance_2"]["module_copies"] > 20 and detail["instance_2"]["payloads"].endswith("Instance 2")
    # the sandbox is gone after a pass; nothing was written to the app's own data
    sandboxes = storage["temp"] / "glossarion-e2e"
    assert not sandboxes.exists() or not any(sandboxes.iterdir())
    assert not (storage["data"] / "config.json").exists()
