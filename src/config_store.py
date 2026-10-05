"""config.json load/save and backup/restore core shared by desktop and mobile.

Moved from translator_gui.py (config load + decrypt in ``TranslatorGUI.__init__``,
the encrypt + atomic write at the end of ``save_config``) and from the
non-Qt parts of config_backup.py (backup creation and 72 h retention, latest
backup lookup, validated atomic restore). config_backup keeps the dialogs and
message boxes and calls these functions.

Rules (see the mobile plan): ``load_config`` only reads and decrypts; it never
sanitizes, migrates or writes defaults. Every function takes the config path
explicitly; ``None`` means ``app_paths.config_file_path()`` read at call time.

GUI-free; must stay importable on Python 3.10 without Qt.
"""

import json
import os
import shutil
import tempfile
import time

import app_paths

BACKUP_DIR_NAME = "config_backups"
BACKUP_PREFIX = "config_"
BACKUP_SUFFIX = ".json.bak"
BACKUP_RETENTION_HOURS = 72


def _resolve(path):
    return path or app_paths.config_file_path()


def _is_backup_name(name):
    return name.startswith(BACKUP_PREFIX) and name.endswith(BACKUP_SUFFIX)


def load_config(path=None, *, decrypt=True):
    """Read config.json and decrypt its API keys.

    Raises (OSError / ValueError) when the file is missing or not JSON; the
    desktop caller keeps its own ``except`` that falls back to ``{}``.
    """
    config_file = _resolve(path)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        # Decrypt API keys
        if decrypt:
            import api_key_encryption
            config = api_key_encryption.decrypt_config(config)
    return config


def save_config_file(config, path=None, *, backup=True):
    """Encrypt API keys and write config.json atomically.

    ``backup=True`` first copies the current file into ``config_backups``
    (desktop ``save_config`` makes that backup itself at the start and passes
    ``backup=False``).
    """
    config_file = _resolve(path)
    if backup:
        backup_config_file(config_file)
    import api_key_encryption
    # --- 5. Final Write to File ---
    google_creds_path = config.get('google_cloud_credentials')
    encrypted_config = api_key_encryption.encrypt_config(config)
    if google_creds_path:
        encrypted_config['google_cloud_credentials'] = google_creds_path

    json.dumps(encrypted_config, ensure_ascii=False, indent=2) # Validation check
    app_paths._atomic_json_write(config_file, encrypted_config)


def config_backup_dir(path=None):
    """``<config dir>/config_backups`` for *path* (not created)."""
    config_file = _resolve(path)
    # Resolve config file path for backup directory
    if os.path.isabs(config_file):
        config_dir = os.path.dirname(config_file)
    else:
        config_dir = os.path.dirname(os.path.abspath(config_file))
    return os.path.join(config_dir, BACKUP_DIR_NAME)


def backup_config_file(path=None, retention_hours=BACKUP_RETENTION_HOURS):
    """Create backup of the existing config file before saving.

    Returns the backup path, or None when there is no config yet or the backup
    failed (failures are printed, never raised, so saving is not interrupted).
    """
    config_file = _resolve(path)
    try:
        # Skip if config file doesn't exist yet
        if not os.path.exists(config_file):
            return None

        # Create backup directory
        backup_dir = config_backup_dir(config_file)
        os.makedirs(backup_dir, exist_ok=True)

        # Create timestamped backup name
        backup_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}.json.bak"
        backup_path = os.path.join(backup_dir, backup_name)

        # Copy the file
        shutil.copy2(config_file, backup_path)

        # Clean backups older than 72 hours
        cutoff_time = time.time() - (retention_hours * 60 * 60)  # 72 hours in seconds
        backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir)
                   if _is_backup_name(f)]

        # Remove backups older than 72 hours
        for backup_file in backups:
            try:
                if os.path.getmtime(backup_file) <= cutoff_time:
                    os.remove(backup_file)
            except Exception:
                pass  # Ignore errors when cleaning old backups

        return backup_path
    except Exception as e:
        # Silent exception - don't interrupt normal operation if backup fails
        print(f"Warning: Could not create config backup: {e}")
        return None


def list_config_backups(path=None):
    """Backups newest first as ``[{'name', 'path', 'mtime', 'size'}]``.

    Returns ``[]`` when the backup folder does not exist. ``size`` is None when
    it cannot be read. Errors while listing/sorting propagate.
    """
    backup_dir = config_backup_dir(path)
    if not os.path.exists(backup_dir):
        return []
    names = [f for f in os.listdir(backup_dir) if _is_backup_name(f)]
    # Sort by modification time (newest first)
    names.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
    entries = []
    for name in names:
        full_path = os.path.join(backup_dir, name)
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            mtime = None
        try:
            size = os.path.getsize(full_path)
        except OSError:
            size = None
        entries.append({'name': name, 'path': full_path, 'mtime': mtime, 'size': size})
    return entries


def latest_backup(path=None):
    """Path of the most recent backup, or None."""
    backup_dir = config_backup_dir(path)

    if not os.path.exists(backup_dir):
        return None

    # Find most recent backup
    backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir)
              if _is_backup_name(f)]

    if not backups:
        return None

    backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return backups[0]


def restore_latest_backup(path=None):
    """Copy the most recent backup over config.json; returns its path or None.

    Errors propagate (the desktop caller reports them in a dialog).
    """
    config_file = _resolve(path)
    latest = latest_backup(config_file)
    if not latest:
        return None
    # Copy backup to config file
    shutil.copy2(latest, config_file)
    return latest


def restore_config_backup_file(path, backup_path, *, safety_backup=None):
    """Validate *backup_path* and atomically replace config.json with it.

    The selected backup is read before the safety backup of the current config
    is made (``safety_backup()`` when given, else ``backup_config_file``),
    because the safety backup's age-based cleanup may prune the selected file.
    Raises on invalid JSON / non-object backups or write errors, leaving the
    current config untouched.
    """
    config_file = _resolve(path)
    # Read first: creating a safety backup may prune the selected old backup.
    with open(backup_path, 'rb') as source:
        contents = source.read()
    if not isinstance(json.loads(contents), dict):
        raise ValueError("The backup must contain a configuration object.")
    if safety_backup is None:
        backup_config_file(config_file)
    else:
        safety_backup()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(os.path.abspath(config_file)), delete=False,
            prefix='.config_restore_', suffix='.tmp',
        ) as target:
            temp_path = target.name
            target.write(contents)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_path, config_file)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
