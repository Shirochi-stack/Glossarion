"""config.json is never left truncated, and an unreadable one is restored from a backup.

Every test works on a temporary config (``app_paths.CONFIG_FILE`` is patched); the
owner's real config.json is never read or written.
"""
import json
import os
import threading
import time
from pathlib import Path

import pytest

import app_paths
import config_store


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.json"
    path.parent.mkdir()
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(path))
    return path


def _write_backup(config_path, name, text, age_hours):
    backup_dir = Path(config_store.config_backup_dir(str(config_path)))
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / name
    backup.write_text(text, encoding="utf-8")
    stamp = time.time() - age_hours * 3600
    os.utime(backup, (stamp, stamp))
    return backup


def _corrupt_copies(config_path):
    return sorted(config_path.parent.glob("config.corrupt-*.json"))


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

def test_corrupted_config_is_restored_from_the_newest_valid_backup(config_path):
    truncated = '{\n  "model": "current",\n  "prompt_profiles": {"Universal": "tr'
    config_path.write_text(truncated, encoding="utf-8")
    _write_backup(config_path, "config_20261027_090000.json.bak",
                  '{"model": "earliest", "auto_update_check": false}', age_hours=50)
    _write_backup(config_path, "config_20261028_110000.json.bak",
                  '{"model": "newest valid", "auto_update_check": false}', age_hours=20)
    _write_backup(config_path, "config_20261029_070000.json.bak", "[1, 2]", age_hours=2)  # not an object
    _write_backup(config_path, "config_20261029_080000.json.bak", "", age_hours=1)  # empty: skipped

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {"model": "newest valid", "auto_update_check": False}
    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert "config_20261028_110000.json.bak" in notice
    corrupt = _corrupt_copies(config_path)
    assert len(corrupt) == 1 and corrupt[0].read_text(encoding="utf-8") == truncated
    assert corrupt[0].name in notice
    # The backups themselves are untouched (nothing pruned or overwritten).
    assert len(os.listdir(config_store.config_backup_dir(str(config_path)))) == 4


def test_restored_backup_api_keys_are_decrypted(config_path):
    import api_key_encryption

    config_path.write_bytes(b"\x00" * 64)  # zero-filled after a crash
    encrypted = api_key_encryption.encrypt_config({"api_key": "sk-test", "model": "m"})
    _write_backup(config_path, "config_20261028_120000.json.bak", json.dumps(encrypted), age_hours=2)

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {"api_key": "sk-test", "model": "m"}
    assert notice


def test_no_valid_backup_falls_back_to_defaults_and_keeps_the_corrupt_file(config_path):
    config_path.write_text('{"model": ', encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", "{broken", age_hours=2)

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {}
    assert "no valid backup" in notice
    assert not config_path.exists()
    assert [p.read_text(encoding="utf-8") for p in _corrupt_copies(config_path)] == ['{"model": ']


def test_no_backups_at_all_falls_back_to_defaults(config_path):
    config_path.write_text("not json", encoding="utf-8")

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {} and "no valid backup" in notice
    assert len(_corrupt_copies(config_path)) == 1


def test_missing_config_is_a_first_run_not_a_recovery(config_path):
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "old"}', age_hours=2)

    assert config_store.load_config_or_restore_backup(str(config_path)) == ({}, None)
    assert not config_path.exists() and not _corrupt_copies(config_path)


def test_valid_config_loads_without_touching_anything(config_path):
    config_path.write_text('{"model": "current"}', encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "old"}', age_hours=2)

    assert config_store.load_config_or_restore_backup(str(config_path)) == ({"model": "current"}, None)
    assert config_path.read_text(encoding="utf-8") == '{"model": "current"}'
    assert not _corrupt_copies(config_path)


def test_desktop_startup_uses_the_recovering_loader():
    source = (Path(__file__).resolve().parents[1] / "src" / "translator_gui.py").read_text(encoding="utf-8")
    start = source.index("self.config_file_path = CONFIG_FILE")
    block = source[start:source.index("self._init_config_state()", start)]
    assert "load_config_or_restore_backup(CONFIG_FILE)" in block
    # The old block reset the config to {} on any error (a failed prompt fix too), then wrote
    # that over config.json; now only a config that never loaded becomes {}.
    assert "if not isinstance(getattr(self, 'config', None), dict):\n                self.config = {}" in block


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def test_unserializable_value_never_truncates_config(config_path):
    config_path.write_text('{"model": "good"}', encoding="utf-8")

    with pytest.raises(TypeError):
        app_paths._atomic_json_write(str(config_path), {"model": "new", "widget": object()})

    assert config_path.read_text(encoding="utf-8") == '{"model": "good"}'
    assert not list(config_path.parent.glob("*.tmp"))


def test_refused_replace_never_falls_back_to_a_direct_write(config_path, monkeypatch):
    # Windows refuses os.replace onto config.json while another handle has it open.
    config_path.write_text('{"model": "good"}', encoding="utf-8")
    monkeypatch.setattr(app_paths, "_REPLACE_RETRY_SECONDS", 0.2)

    def refuse(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(app_paths.os, "replace", refuse)
    with pytest.raises(PermissionError):
        app_paths._atomic_json_write(str(config_path), {"model": "new"})

    assert config_path.read_text(encoding="utf-8") == '{"model": "good"}'
    assert not list(config_path.parent.glob("*.tmp"))


def test_briefly_locked_config_is_replaced_once_the_lock_is_gone(config_path, monkeypatch):
    config_path.write_text('{"model": "old"}', encoding="utf-8")
    real_replace = os.replace
    attempts = []

    def locked_twice(src, dst):
        attempts.append(dst)
        if len(attempts) <= 2:
            raise PermissionError(32, "The process cannot access the file")
        real_replace(src, dst)

    monkeypatch.setattr(app_paths.os, "replace", locked_twice)
    app_paths._atomic_json_write(str(config_path), {"model": "new"})

    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model": "new"}
    assert len(attempts) == 3


def test_concurrent_writers_always_leave_valid_json(config_path):
    errors = []

    def writer(n):
        payload = {"writer": n, "data": "x" * 20000}
        for _ in range(25):
            try:
                app_paths._atomic_json_write(str(config_path), payload)
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    final = json.loads(config_path.read_text(encoding="utf-8"))
    assert final["writer"] in range(8) and len(final["data"]) == 20000
    assert not list(config_path.parent.glob("*.tmp"))
