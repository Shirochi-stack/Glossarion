from pathlib import Path

import pytest

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


def _make_generated_dir(root: Path, name: str) -> Path:
    generated = root / name
    _touch(generated / "Local Storage" / "leveldb" / "000003.log")
    _touch(generated / "cache" / "Cache" / "data_0")
    return generated


def test_browser_state_cleanup_removes_only_generated_dirs(monkeypatch, tmp_path):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    root.mkdir()

    generated_names = ("authnd_browser", "gemini_free_browser", "qtwebengine_requests")
    for name in generated_names:
        _make_generated_dir(root, name)

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

    assert stats["removed"] == len(generated_names)
    assert stats["failed"] == 0
    for name in generated_names:
        assert not (root / name).exists()
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
        _make_generated_dir(root, "qtwebengine_requests")
        _touch(root / "cache" / "kept.txt")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["removed"] == len(roots)
    assert stats["failed"] == 0
    for root in roots:
        assert not (root / "qtwebengine_requests").exists()
        assert (root / "cache" / "kept.txt").is_file()


@pytest.mark.parametrize("helper_env", ("AUTHND_TOKEN_HELPER", "GEMINI_FREE_HELPER"))
def test_browser_state_cleanup_skips_helper_processes(monkeypatch, tmp_path, helper_env):
    home = _isolated_state_env(monkeypatch, tmp_path)
    root = home / ".glossarion"
    _make_generated_dir(root, "authnd_browser")
    monkeypatch.setenv(helper_env, "1")

    stats = shutdown_utils.cleanup_browser_generated_state_for_shutdown()

    assert stats["skipped"] == 1
    assert stats["removed"] == 0
    assert (root / "authnd_browser").is_dir()


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
    target = _make_generated_dir(root, "authnd_browser")
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
    assert not any((root / ".startup_cleanup_deleting").glob("*"))
