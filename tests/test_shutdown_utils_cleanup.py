from pathlib import Path
import sys
import types

import pytest

import gemini_free
import shutdown_utils


def _isolated_state_env(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("AUTHND_TOKEN_HELPER", raising=False)
    monkeypatch.delenv("GEMINI_FREE_HELPER", raising=False)
    return home


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_profile_dir(root: Path, parent_name: str, child_name: str = "profile-1") -> Path:
    profile = root / parent_name / child_name
    _touch(profile / "Local Storage" / "leveldb" / "000003.log")
    _touch(profile / "cache" / "Cache" / "data_0")
    return profile


def test_browser_state_cleanup_removes_only_child_profiles(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    root.mkdir()

    generated_profiles = [
        _make_profile_dir(root, "authnd_browser", "authnd-1"),
        _make_profile_dir(root, "gemini_free_browser", "gemini-1"),
        _make_profile_dir(root, "qtwebengine_requests", "request-1"),
    ]
    _touch(root / "authnd_browser" / "root-level-file")
    _touch(root / "gemini_free_browser" / "legacy-root-level-file")

    preserve_dirs = ("cache", "authza_browser", "authza_browser_2")
    for name in preserve_dirs:
        _touch(root / name / "kept.txt")

    preserve_files = (
        "authgpt_tokens.json",
        "authgem_tokens.json",
        "authcd_tokens.json",
        "config_android.json",
    )
    for name in preserve_files:
        _touch(root / name, "{}")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["removed"] == len(generated_profiles)
    assert stats["failed"] == 0
    for profile in generated_profiles:
        assert not profile.exists()
    assert (root / "authnd_browser").is_dir()
    assert (root / "authnd_browser" / "root-level-file").is_file()
    assert (root / "gemini_free_browser").is_dir()
    assert (root / "gemini_free_browser" / "legacy-root-level-file").is_file()
    assert (root / "qtwebengine_requests").is_dir()
    for name in preserve_dirs:
        assert (root / name / "kept.txt").is_file()
    for name in preserve_files:
        assert (root / name).is_file()


def test_browser_state_cleanup_finds_app_data_roots(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    roots = [
        Path(tmp_path / "roaming" / "Glossarion"),
        Path(tmp_path / "local" / "Glossarion"),
        Path(tmp_path / "xdg-data" / "Glossarion"),
        Path(tmp_path / "xdg-config" / "Glossarion"),
        home / "Library" / "Application Support" / "Glossarion",
        home / ".local" / "share" / "Glossarion",
        home / ".config" / "Glossarion",
    ]
    for root in roots:
        _make_profile_dir(root, "qtwebengine_requests", "request-1")
        _touch(root / "cache" / "kept.txt")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["removed"] == len(roots)
    assert stats["failed"] == 0
    for root in roots:
        assert (root / "qtwebengine_requests").is_dir()
        assert not (root / "qtwebengine_requests" / "request-1").exists()
        assert (root / "cache" / "kept.txt").is_file()


@pytest.mark.parametrize("helper_env", ("AUTHND_TOKEN_HELPER", "GEMINI_FREE_HELPER"))
def test_browser_state_cleanup_skips_helper_processes(monkeypatch, tmp_path, helper_env):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    profile = _make_profile_dir(root, "authnd_browser")
    monkeypatch.setenv(helper_env, "1")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["skipped"] == 1
    assert stats["removed"] == 0
    assert profile.is_dir()


def test_browser_state_cleanup_sweeps_stale_handoff(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    stale = root / ".startup_cleanup_deleting" / "authnd_browser-old"
    _touch(stale / "LOCK")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["removed"] == 1
    assert stats["failed"] == 0
    assert not stale.exists()
    assert not (root / ".startup_cleanup_deleting").exists()


def test_browser_state_cleanup_moves_then_retries_locked_target(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    target = _make_profile_dir(root, "authnd_browser", "authnd-locked")
    original_remove = shutdown_utils._remove_path_once

    def remove_with_initial_lock(path):
        if Path(path) == target:
            raise PermissionError("locked")
        return original_remove(path)

    monkeypatch.setattr(shutdown_utils, "_remove_path_once", remove_with_initial_lock)

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["removed"] == 1
    assert stats["moved"] == 1
    assert stats["failed"] == 0
    assert not target.exists()
    assert (root / "authnd_browser").is_dir()
    assert not any((root / ".startup_cleanup_deleting").glob("*"))


def test_cleanup_generated_browser_profile_dir_refuses_parent_and_wrong_parent(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    valid_profile = _make_profile_dir(root, "authnd_browser", "keep-me")
    parent = root / "authnd_browser"
    wrong_parent_profile = _make_profile_dir(root, "authza_browser", "do-not-touch")

    parent_stats = shutdown_utils.cleanup_generated_browser_profile_dir(parent, "authnd_browser")
    wrong_stats = shutdown_utils.cleanup_generated_browser_profile_dir(wrong_parent_profile, "authnd_browser")
    valid_stats = shutdown_utils.cleanup_generated_browser_profile_dir(valid_profile, "authnd_browser")

    assert parent_stats["skipped"] == 1
    assert wrong_stats["skipped"] == 1
    assert valid_stats["removed"] == 1
    assert parent.is_dir()
    assert not valid_profile.exists()
    assert wrong_parent_profile.is_dir()


def test_cleanup_generated_browser_profile_dir_preserves_siblings(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    target = _make_profile_dir(root, "gemini_free_browser", "remove-me")
    sibling = _make_profile_dir(root, "gemini_free_browser", "keep-me")

    stats = shutdown_utils.cleanup_generated_browser_profile_dir(target, "gemini_free_browser")

    assert stats["removed"] == 1
    assert not target.exists()
    assert sibling.is_dir()
    assert (root / "gemini_free_browser").is_dir()


def test_gemini_free_profile_uses_per_request_child_root(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)

    class FakeProfile:
        def __init__(self, name, app):
            self.name = name
            self.app = app
            self.storage_path = ""
            self.cache_path = ""
            self.user_agent = ""
            self.accept_language = ""

        def setHttpUserAgent(self, value):
            self.user_agent = value

        def setPersistentStoragePath(self, value):
            self.storage_path = value

        def setCachePath(self, value):
            self.cache_path = value

        def setHttpAcceptLanguage(self, value):
            self.accept_language = value

    pyside_module = types.ModuleType("PySide6")
    webengine_module = types.ModuleType("PySide6.QtWebEngineCore")
    webengine_module.QWebEngineProfile = FakeProfile
    monkeypatch.setitem(sys.modules, "PySide6", pyside_module)
    monkeypatch.setitem(sys.modules, "PySide6.QtWebEngineCore", webengine_module)
    monkeypatch.setattr(gemini_free.Path, "home", classmethod(lambda cls: home))

    profile, profile_root = gemini_free._create_profile(object())

    assert profile_root.parent == home / ".glossarion" / "gemini_free_browser"
    assert profile_root.name
    assert profile_root.is_dir()
    assert profile.storage_path == str(profile_root)
    assert profile.cache_path == str(profile_root / "cache")


def test_meipass_matching_detects_paths_and_command_lines(tmp_path):
    meipass = tmp_path / "_MEI426002"
    dll_path = meipass / "PySide6" / "Qt6" / "bin" / "Qt6Core.dll"
    outside = tmp_path / "outside" / "helper.exe"

    assert shutdown_utils._path_points_under_meipass(str(dll_path), str(meipass))
    assert not shutdown_utils._path_points_under_meipass(str(outside), str(meipass))
    assert shutdown_utils._text_mentions_meipass(
        f'"{outside}" --library "{dll_path}"',
        str(meipass),
    )
    assert shutdown_utils._process_info_points_to_meipass(
        {"exe": str(outside), "cmdline": [str(outside), "--root", str(meipass)]},
        str(meipass),
    )


def test_psutil_meipass_sweep_kills_only_lock_holders(monkeypatch, tmp_path):
    meipass = tmp_path / "_MEI426002"
    meipass.mkdir()
    monkeypatch.setattr(shutdown_utils.sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(shutdown_utils.os, "getpid", lambda: 100)
    killed_by_taskkill = []
    monkeypatch.setattr(
        shutdown_utils,
        "_taskkill_pid_tree",
        lambda pid, **_kwargs: killed_by_taskkill.append(pid) or True,
    )

    class FakeProc:
        def __init__(self, pid, info, maps=()):
            self.pid = pid
            self.info = {"pid": pid, **info}
            self._maps = list(maps)
            self.terminated = False
            self.killed = False
            self._running = True

        def open_files(self):
            return []

        def memory_maps(self):
            return self._maps

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self._running = False

        def is_running(self):
            return self._running

    own_proc = FakeProc(100, {"exe": str(meipass / "Glossarion.exe")})
    unrelated = FakeProc(101, {"exe": str(tmp_path / "other.exe"), "cmdline": ["other"]})
    exe_holder = FakeProc(102, {"exe": str(meipass / "QtWebEngineProcess.exe")})
    cmdline_holder = FakeProc(103, {"exe": str(tmp_path / "helper.exe"), "cmdline": ["helper", str(meipass)]})
    map_holder = FakeProc(
        104,
        {"exe": str(tmp_path / "native-helper.exe"), "cmdline": ["native-helper"]},
        maps=[types.SimpleNamespace(path=str(meipass / "Qt6Gui.dll"))],
    )

    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs=None: [own_proc, unrelated, exe_holder, cmdline_holder, map_holder],
        wait_procs=lambda procs, timeout=None: (
            [proc for proc in procs if not proc.is_running()],
            [proc for proc in procs if proc.is_running()],
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    count = shutdown_utils._terminate_psutil_meipass_lock_holders(timeout=0.2)

    assert count == 3
    assert not own_proc.terminated
    assert not unrelated.terminated
    for proc in (exe_holder, cmdline_holder, map_holder):
        assert proc.terminated
        assert proc.killed
    assert killed_by_taskkill == [102, 103, 104]
