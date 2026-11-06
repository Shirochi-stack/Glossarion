"""config.json is never left truncated, and an unreadable one is restored from a backup.

Every test works on a temporary config (``app_paths.CONFIG_FILE`` is patched); the
owner's real config.json is never read or written.
"""
import json
import os
import threading
import time
import types
from pathlib import Path

import pytest

import app_paths
import config_store


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.json"
    path.parent.mkdir()
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(path))
    monkeypatch.setattr(config_store, "READ_RETRY_SECONDS", 0)
    monkeypatch.setattr(app_paths, "_WRITE_PROTECTED_FILES", set())
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
# Looking again before config.json counts as unreadable
# ---------------------------------------------------------------------------

def _fail_reads(monkeypatch, *errors):
    """``load_config`` raises *errors* one per call, then reads the file for real."""
    real_load = config_store.load_config
    pending = list(errors)
    calls = []

    def load(path=None, *, decrypt=True):
        calls.append(path)
        if pending:
            raise pending.pop(0)
        return real_load(path, decrypt=decrypt)

    monkeypatch.setattr(config_store, "load_config", load)
    return calls


@pytest.mark.parametrize("error", [PermissionError(13, "locked"), json.JSONDecodeError("half written", "", 0)])
def test_a_read_that_works_on_a_later_try_touches_nothing(config_path, monkeypatch, error):
    config_path.write_text('{"model": "current"}', encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "old"}', age_hours=2)
    calls = _fail_reads(monkeypatch, error, error)

    assert config_store.load_config_or_restore_backup(str(config_path)) == ({"model": "current"}, None)

    assert len(calls) == 3
    assert config_path.read_text(encoding="utf-8") == '{"model": "current"}' and not _corrupt_copies(config_path)


def test_every_try_is_used_before_a_backup_is_restored(config_path, monkeypatch):
    config_path.write_text("not json", encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "saved"}', age_hours=2)
    real_load = config_store.load_config
    calls = []

    def load(path=None, *, decrypt=True):
        calls.append(path)
        return real_load(path, decrypt=decrypt)

    monkeypatch.setattr(config_store, "load_config", load)
    config, _notice = config_store.load_config_or_restore_backup(str(config_path))
    assert config == {"model": "saved"}
    assert len(calls) == config_store.READ_ATTEMPTS + 1  # and once for the restored file


def test_a_first_run_does_not_wait(config_path, monkeypatch):
    calls = _fail_reads(monkeypatch)
    assert config_store.load_config_or_restore_backup(str(config_path)) == ({}, None)
    assert len(calls) == 1


def test_a_missing_config_with_backups_is_looked_for_again(config_path, monkeypatch):
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "old"}', age_hours=2)
    real_load = config_store.load_config

    def load(path=None, *, decrypt=True):
        if not config_path.exists():
            config_path.write_text('{"model": "back"}', encoding="utf-8")
            raise FileNotFoundError(path)
        return real_load(path, decrypt=decrypt)

    monkeypatch.setattr(config_store, "load_config", load)
    assert config_store.load_config_or_restore_backup(str(config_path)) == ({"model": "back"}, None)


def test_a_locked_config_is_never_overwritten(config_path, monkeypatch):
    original = '{"model": "my settings"}'
    config_path.write_text(original, encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "saved"}', age_hours=2)
    _fail_reads(monkeypatch, *[PermissionError(13, "locked")] * config_store.READ_ATTEMPTS)

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {"model": "saved"}
    assert "config_20261028_120000.json.bak" in notice and "not be saved" in notice
    assert config_path.read_text(encoding="utf-8") == original and not _corrupt_copies(config_path)
    with pytest.raises(PermissionError):
        app_paths._atomic_json_write(str(config_path), {"auto_update_check": True})
    assert config_path.read_text(encoding="utf-8") == original


def test_a_locked_config_without_backups_is_not_overwritten_either(config_path, monkeypatch):
    original = '{"model": "my settings"}'
    config_path.write_text(original, encoding="utf-8")
    _fail_reads(monkeypatch, *[PermissionError(13, "locked")] * config_store.READ_ATTEMPTS)

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {} and "defaults" in notice
    with pytest.raises(PermissionError):
        config_store.save_config_file({"model": "default"}, str(config_path), backup=False)
    assert config_path.read_text(encoding="utf-8") == original


def test_a_corrupt_config_that_cannot_be_kept_is_not_replaced(config_path, monkeypatch):
    config_path.write_text("not json", encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "saved"}', age_hours=2)
    monkeypatch.setattr(config_store, "move_corrupt_config_aside", lambda path=None: None)

    config, notice = config_store.load_config_or_restore_backup(str(config_path))

    assert config == {"model": "saved"} and notice
    assert config_path.read_text(encoding="utf-8") == "not json"


def test_other_files_stay_writable_while_config_is_protected(config_path, monkeypatch):
    config_path.write_text("{}", encoding="utf-8")
    _fail_reads(monkeypatch, *[PermissionError(13, "locked")] * config_store.READ_ATTEMPTS)
    config_store.load_config_or_restore_backup(str(config_path))
    other = config_path.parent / "other.json"
    app_paths._atomic_json_write(str(other), {"a": 1})
    assert json.loads(other.read_text(encoding="utf-8")) == {"a": 1}


def test_a_console_that_cannot_print_does_not_stop_the_recovery(config_path, monkeypatch):
    config_path.write_text("not json", encoding="utf-8")
    _write_backup(config_path, "config_20261028_120000.json.bak", '{"model": "saved"}', age_hours=2)

    def refuse(*_args, **_kwargs):
        raise UnicodeEncodeError("cp1252", "x", 0, 1, "cannot encode")

    monkeypatch.setattr("builtins.print", refuse)
    config, notice = config_store.load_config_or_restore_backup(str(config_path))
    assert config == {"model": "saved"} and notice


def test_desktop_startup_protects_a_config_that_never_loaded():
    source = (Path(__file__).resolve().parents[1] / "src" / "translator_gui.py").read_text(encoding="utf-8")
    start = source.index("self.config_file_path = CONFIG_FILE")
    block = source[start:source.index("self._init_config_state()", start)]
    assert "protect_file_from_writes(CONFIG_FILE)" in block


# ---------------------------------------------------------------------------
# Writers holding part of the settings (bubble_detector, local_inpainter)
# ---------------------------------------------------------------------------

def test_update_config_file_keeps_the_rest_of_the_file(config_path):
    config_path.write_text('{"model": "mine", "api_key": "ENC:x"}', encoding="utf-8")
    config_store.update_config_file({"bubble_model_path": "m.pt"}, str(config_path))
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "model": "mine", "api_key": "ENC:x", "bubble_model_path": "m.pt"}


@pytest.mark.parametrize("content", ["not json", "[1]"])
def test_update_config_file_writes_nothing_over_an_unreadable_file(config_path, content):
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        config_store.update_config_file({"bubble_model_path": "m.pt"}, str(config_path))
    assert config_path.read_text(encoding="utf-8") == content


def test_update_config_file_creates_a_missing_file(config_path):
    config_store.update_config_file({"bubble_model_path": "m.pt"}, str(config_path))
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"bubble_model_path": "m.pt"}


def test_a_bubble_detector_that_failed_to_load_does_not_wipe_the_config(config_path, monkeypatch):
    bubble_detector = pytest.importorskip("bubble_detector")
    config_path.write_text('{"model": "mine", "api_key": "ENC:x"}', encoding="utf-8")
    # The state BubbleDetector._load_config leaves after a failed read, then its model path save.
    detector = types.SimpleNamespace(config={}, config_path=str(config_path))
    bubble_detector.BubbleDetector._save_config(detector, {"bubble_model_path": "m.pt"})
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "model": "mine", "api_key": "ENC:x", "bubble_model_path": "m.pt"}
    assert detector.config == {"bubble_model_path": "m.pt"}
    config_path.write_text("half written", encoding="utf-8")
    bubble_detector.BubbleDetector._save_config(detector, {"bubble_model_path": "m.pt"})
    assert config_path.read_text(encoding="utf-8") == "half written"


def test_an_inpainter_save_does_not_wipe_an_unreadable_config(config_path):
    local_inpainter = pytest.importorskip("local_inpainter")
    config_path.write_text("half written", encoding="utf-8")
    inpainter = types.SimpleNamespace(config={}, config_path=str(config_path))
    local_inpainter.LocalInpainter._save_config(inpainter, {"inpainter_model": "lama"})
    assert config_path.read_text(encoding="utf-8") == "half written"
    config_path.write_text('{"model": "mine"}', encoding="utf-8")
    local_inpainter.LocalInpainter._save_config(inpainter, {"inpainter_model": "lama"})
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model": "mine", "inpainter_model": "lama"}


# ---------------------------------------------------------------------------
# Backups survive
# ---------------------------------------------------------------------------

def _backup_names(config_path):
    return sorted(os.listdir(config_store.config_backup_dir(str(config_path))))


def test_a_save_after_days_away_keeps_the_old_backups_and_the_new_one(config_path):
    # The reported loss: 72 h without opening the app, then the first save of the new version.
    config_path.write_text('{"model": "my settings"}', encoding="utf-8")
    stamp = time.time() - 5 * 86400
    os.utime(config_path, (stamp, stamp))
    first = _write_backup(config_path, "config_20261020_100000.json.bak", '{"model": "first"}', age_hours=150)
    second = _write_backup(config_path, "config_20261021_100000.json.bak", '{"model": "second"}', age_hours=125)

    new_backup = Path(config_store.backup_config_file(str(config_path)))

    assert first.exists() and second.exists() and new_backup.exists()
    assert new_backup.read_text(encoding="utf-8") == '{"model": "my settings"}'
    assert time.time() - new_backup.stat().st_mtime < 60  # its own age, not the config's


def test_many_saves_of_one_config_do_not_push_out_an_older_one(config_path):
    config_path.write_text('{"model": "default"}', encoding="utf-8")
    _write_backup(config_path, "config_20261020_100000.json.bak", '{"model": "my settings"}', age_hours=150)
    for second in range(30):  # saves of the same config: one version, however many copies
        config_store.backup_config_file(str(config_path))
        backup_dir = Path(config_store.config_backup_dir(str(config_path)))
        os.rename(config_store.latest_backup(str(config_path)), backup_dir / f"config_20261102_0000{second:02d}.json.bak")
    texts = [p.read_text(encoding="utf-8") for p in backup_dir.iterdir()]
    assert '{"model": "my settings"}' in texts


def test_backups_past_the_newest_versions_expire(config_path):
    keep = config_store.BACKUP_KEEP_VERSIONS
    config_path.write_text('{"model": "now"}', encoding="utf-8")
    for n in range(keep + 5):
        _write_backup(config_path, f"config_202610{n:02d}_100000.json.bak", f'{{"model": {n}}}', age_hours=100 + n)

    config_store.backup_config_file(str(config_path))

    names = _backup_names(config_path)
    # the new backup and the newest old versions stay; the oldest are gone
    assert len(names) == keep
    assert "config_20261000_100000.json.bak" in names
    assert f"config_202610{keep + 4:02d}_100000.json.bak" not in names


def test_expired_copies_and_broken_backups_do_not_count_as_versions(config_path):
    config_path.write_text('{"model": "now"}', encoding="utf-8")
    keep_me = _write_backup(config_path, "config_20261001_100000.json.bak", '{"model": "older"}', age_hours=200)
    copy_of_new = _write_backup(config_path, "config_20261002_100000.json.bak", '{"model": "now"}', age_hours=100)
    broken = _write_backup(config_path, "config_20261003_100000.json.bak", "{broken", age_hours=90)

    config_store.backup_config_file(str(config_path))

    assert keep_me.exists() and not copy_of_new.exists() and not broken.exists()


def test_recent_backups_are_never_removed(config_path):
    keep = config_store.BACKUP_KEEP_VERSIONS
    config_path.write_text('{"model": "now"}', encoding="utf-8")
    for n in range(keep + 5):
        _write_backup(config_path, f"config_202610{n:02d}_100000.json.bak", "{broken", age_hours=1)
    config_store.backup_config_file(str(config_path))
    assert len(_backup_names(config_path)) == keep + 6


def test_a_failed_backup_removes_nothing(config_path, monkeypatch):
    config_path.write_text('{"model": "now"}', encoding="utf-8")
    old = _write_backup(config_path, "config_20261020_100000.json.bak", "{broken", age_hours=150)

    def refuse(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(config_store, "_replace_config_contents", refuse)
    assert config_store.backup_config_file(str(config_path)) is None
    assert old.exists()


def test_restore_latest_backup_skips_broken_ones(config_path):
    config_path.write_text("not json", encoding="utf-8")
    _write_backup(config_path, "config_20261028_100000.json.bak", '{"model": "saved"}', age_hours=3)
    _write_backup(config_path, "config_20261028_120000.json.bak", "{broken", age_hours=1)

    restored = config_store.restore_latest_backup(str(config_path))

    assert os.path.basename(restored) == "config_20261028_100000.json.bak"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model": "saved"}


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
