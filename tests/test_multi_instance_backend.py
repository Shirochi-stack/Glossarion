"""Multi-instance translation: the per-instance pieces of the shared backend.

* ``stop_control.reset_api_watchdog``: the desktop GUI (instance 1) still deletes every
  process's watchdog file on Stop; inside instance N > 1 a run that ends or stops deletes only
  its own pid's file, and ``all_processes=`` picks either sweep explicitly;
* ``unified_api_client``: watchdog files carry ``"instance"``; instance N > 1 saves payloads
  under ``Payloads/Instance N/`` (instance 1 unchanged).

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_multi_instance_backend.py
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))

import stop_control  # noqa: E402


# --------------------------------------------------------------------------
# stop_control: watchdog file sweep per instance
# --------------------------------------------------------------------------

@pytest.fixture
def watchdog(tmp_path, monkeypatch):
    """A watchdog folder with this process's file (+ .tmp) and two other processes' files, and
    a stand-in unified_api_client that records counter resets."""
    folder = tmp_path / "glossarion_watchdog"
    folder.mkdir()
    pid = os.getpid()
    names = [f"api_watchdog_{pid}.json", f"api_watchdog_{pid}.json.tmp",
             "api_watchdog_999991.json", "api_watchdog_999992.json.tmp", "unrelated.json"]
    for name in names:
        (folder / name).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GLOSSARION_WATCHDOG_DIR", str(folder))
    monkeypatch.delenv("GLOSSARION_INSTANCE", raising=False)
    resets = []
    fake = types.ModuleType("unified_api_client")
    fake._api_watchdog_reset = lambda: resets.append(1)
    monkeypatch.setitem(sys.modules, "unified_api_client", fake)
    return types.SimpleNamespace(folder=folder, resets=resets, monkeypatch=monkeypatch,
                                 left=lambda: sorted(p.name for p in folder.iterdir()))


def test_instance_one_full_reset_clears_every_watchdog_file(watchdog):
    stop_control.reset_api_watchdog(clear_stale_external_files=True)
    assert watchdog.left() == ["unrelated.json"] and watchdog.resets == [1]


def test_instance_n_clears_only_its_own_watchdog_file(watchdog):
    watchdog.monkeypatch.setenv("GLOSSARION_INSTANCE", "2")
    stop_control.reset_api_watchdog(clear_stale_external_files=True)
    assert watchdog.left() == ["api_watchdog_999991.json", "api_watchdog_999992.json.tmp", "unrelated.json"]
    assert watchdog.resets == [1]  # the counters are reset either way


def test_no_file_sweep_when_not_asked(watchdog):
    for instance in ("1", "3"):
        watchdog.monkeypatch.setenv("GLOSSARION_INSTANCE", instance)
        stop_control.reset_api_watchdog(clear_stale_external_files=False)
    assert len(watchdog.left()) == 5 and watchdog.resets == [1, 1]


def test_all_processes_overrides_the_instance(watchdog):
    stop_control.reset_api_watchdog(all_processes=False)  # instance 1, own file only
    assert watchdog.left() == ["api_watchdog_999991.json", "api_watchdog_999992.json.tmp", "unrelated.json"]
    watchdog.monkeypatch.setenv("GLOSSARION_INSTANCE", "2")
    stop_control.reset_api_watchdog(all_processes=True)  # the GUI's full reset from anywhere
    assert watchdog.left() == ["unrelated.json"]


def test_clear_own_watchdog_file_tolerates_missing_folder(watchdog, tmp_path):
    stop_control.clear_own_watchdog_file()
    assert watchdog.left() == ["api_watchdog_999991.json", "api_watchdog_999992.json.tmp", "unrelated.json"]
    stop_control.clear_own_watchdog_file()  # nothing left to delete
    watchdog.monkeypatch.setenv("GLOSSARION_WATCHDOG_DIR", str(tmp_path / "missing"))
    stop_control.clear_own_watchdog_file()
    watchdog.monkeypatch.delenv("GLOSSARION_WATCHDOG_DIR")
    stop_control.clear_own_watchdog_file()
    assert "clear_own_watchdog_file" in stop_control.__all__


def test_headless_owner_reset_follows_the_instance(watchdog):
    """HeadlessOwner (desktop instance child, mobile job) resets through stop_control."""
    import headless_owner

    watchdog.monkeypatch.setenv("GLOSSARION_INSTANCE", "2")
    headless_owner.HeadlessOwner._reset_api_watchdog_progress(object(), clear_stale_external_files=True)
    assert watchdog.left() == ["api_watchdog_999991.json", "api_watchdog_999992.json.tmp", "unrelated.json"]


# --------------------------------------------------------------------------
# unified_api_client: watchdog "instance" field and per-instance Payloads root
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uac():
    # Import with the HTTP logger off so this process's requests stay unpatched.
    saved = os.environ.get("GLOSSARION_HTTP_LOG")
    os.environ["GLOSSARION_HTTP_LOG"] = "0"
    try:
        module = importlib.import_module("unified_api_client")
    finally:
        if saved is None:
            os.environ.pop("GLOSSARION_HTTP_LOG", None)
        else:
            os.environ["GLOSSARION_HTTP_LOG"] = saved
    return module


@pytest.mark.parametrize("instance, expected", [(None, 1), ("1", 1), ("3", 3)])
def test_watchdog_file_records_the_instance(uac, tmp_path, monkeypatch, instance, expected):
    monkeypatch.setenv("GLOSSARION_WATCHDOG_DIR", str(tmp_path))
    if instance is None:
        monkeypatch.delenv("GLOSSARION_INSTANCE", raising=False)
    else:
        monkeypatch.setenv("GLOSSARION_INSTANCE", instance)
    uac._api_watchdog_external_write(uac.get_api_watchdog_state())
    state = json.loads((tmp_path / f"api_watchdog_{os.getpid()}.json").read_text(encoding="utf-8"))
    assert state["instance"] == expected and state["pid"] == os.getpid()


@pytest.mark.parametrize("instance, sub", [(None, ()), ("1", ()), ("2", ("Instance 2",)), ("12", ("Instance 12",))])
def test_payloads_dir_is_per_instance(uac, tmp_path, monkeypatch, instance, sub):
    assert not getattr(sys, "frozen", False)
    for name in ("GLOSSARION_MOBILE", "GLOSSARION_DATA_DIR", "FLET_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    if instance is None:
        monkeypatch.delenv("GLOSSARION_INSTANCE", raising=False)
    else:
        monkeypatch.setenv("GLOSSARION_INSTANCE", instance)
    monkeypatch.setattr(uac, "_payloads_resolved_dir", None)
    monkeypatch.setattr(uac, "_PAYLOADS_DISABLED", False)
    monkeypatch.setattr(uac, "__file__", str(tmp_path / "unified_api_client.py"))
    expected = tmp_path.joinpath("Payloads", *sub)
    assert uac._payloads_dir() == str(expected)
    assert expected.is_dir()


def test_payloads_temp_fallback_is_per_instance_too(uac, tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_INSTANCE", "2")
    monkeypatch.setattr(uac, "_payloads_resolved_dir", None)
    monkeypatch.setattr(uac, "_PAYLOADS_DISABLED", False)
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    monkeypatch.setattr(uac, "__file__", str(blocker / "unified_api_client.py"))
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    assert uac._payloads_dir() == str(tmp_path / "tmp" / "Glossarion_Payloads" / "Instance 2")
