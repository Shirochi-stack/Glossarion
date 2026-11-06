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
#: Backups past the retention are still kept while they are among the newest of this many different configs.
BACKUP_KEEP_VERSIONS = 10
#: How often config.json is read before it counts as unreadable (antivirus, cloud sync and an
#: older writer hold it or show it half-written for a moment, right after an update too).
READ_ATTEMPTS = 5
READ_RETRY_SECONDS = 0.5


def _resolve(path):
    return path or app_paths.config_file_path()


def _is_backup_name(name):
    return name.startswith(BACKUP_PREFIX) and name.endswith(BACKUP_SUFFIX)


def _report(text):
    """Print a recovery message. A console that cannot show it must not stop the recovery."""
    try:
        print(text)
    except Exception:
        pass


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


def load_config_retrying(path=None, *, decrypt=True):
    """``load_config`` that looks again before giving up (``READ_ATTEMPTS``); raises the last error.

    A config that is not an object raises ValueError. A missing file is only looked for again
    when backups show it existed: a first run starts at once.
    """
    config_file = _resolve(path)
    error = None
    for attempt in range(READ_ATTEMPTS):
        if attempt:
            time.sleep(READ_RETRY_SECONDS)
        try:
            config = load_config(config_file, decrypt=decrypt)
            if isinstance(config, dict):
                return config
            error = ValueError("config.json does not contain a JSON object")
        except FileNotFoundError as e:
            error = e
            if not list_config_backups(config_file):
                break
        except Exception as e:
            error = e
    raise error


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


def update_config_file(values, path=None):
    """Write *values* into config.json atomically, keeping the rest of the file as it is on disk.

    For writers that own a few keys (bubble_detector / local_inpainter model paths): pass only
    the keys you changed, never a whole config dict read earlier. A copy read before the GUI's
    last save writes its old settings back (2026-11-06: ``use_inpainter_keys`` and
    ``active_profile`` reverted whenever a manga model loaded).

    The current file is read and written under the config write lock, so a save from another
    thread of this process cannot land in between. No decrypting or encrypting: for callers
    holding the file's own raw values. Raises, and writes nothing, when the existing file
    cannot be read: it would otherwise be replaced by *values* alone. A missing file is created.
    """
    config_file = _resolve(path)
    with app_paths._ATOMIC_WRITE_LOCK:
        try:
            current = load_config_retrying(config_file, decrypt=False)
        except FileNotFoundError:
            current = {}
        current.update(values)
        app_paths._atomic_json_write(config_file, current)


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

        # Copy the file. A new file on purpose: its age is the backup's, not the time config.json
        # was last saved (copy2 kept that, and a config unused for 72 h had its backup removed at once).
        with open(config_file, 'rb') as source:
            contents = source.read()
        _replace_config_contents(backup_path, contents)

        # Only now that the new backup exists
        try:
            _prune_backups(config_file, retention_hours)
        except Exception:
            pass  # Ignore errors when cleaning old backups

        return backup_path
    except Exception as e:
        # Silent exception - don't interrupt normal operation if backup fails
        _report(f"Warning: Could not create config backup: {e}")
        return None


def _prune_backups(path, retention_hours):
    """Remove backups older than *retention_hours*, except the newest ``BACKUP_KEEP_VERSIONS`` different configs.

    Age alone emptied the folder after 72 h without opening the app, so the first save of a new
    version was left with no backup of the old settings. Copies of one config count once, or a
    few saves in a row would push out everything older. One that cannot be read now is left alone.
    """
    cutoff_time = time.time() - (retention_hours * 60 * 60)
    backups = list_config_backups(path)  # newest first
    if not any(entry['mtime'] is not None and entry['mtime'] <= cutoff_time for entry in backups):
        return
    kept_versions = set()
    for entry in backups:
        try:
            contents = _read_backup_contents(entry['path'])
        except OSError:
            continue
        except ValueError:
            contents = None
        if contents is not None and contents not in kept_versions and len(kept_versions) < BACKUP_KEEP_VERSIONS:
            kept_versions.add(contents)
        elif entry['mtime'] is not None and entry['mtime'] <= cutoff_time:
            try:
                os.remove(entry['path'])
            except OSError:
                pass


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
    """Atomically replace config.json with the most recent valid backup; returns its path or None.

    Errors writing config.json propagate (the desktop caller reports them in a dialog).
    """
    config_file = _resolve(path)
    for entry in list_config_backups(config_file):
        try:
            contents = _read_backup_contents(entry['path'])
        except (OSError, ValueError) as e:
            _report(f"⚠️ Skipping config backup {entry['name']}: {e}")
            continue
        _replace_config_contents(config_file, contents)
        return entry['path']
    return None


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
    contents = _read_backup_contents(backup_path)
    if safety_backup is None:
        backup_config_file(config_file)
    else:
        safety_backup()
    _replace_config_contents(config_file, contents)


def _read_backup_contents(backup_path):
    """Bytes of *backup_path*; raises unless they hold a JSON configuration object."""
    with open(backup_path, 'rb') as source:
        contents = source.read()
    if not isinstance(json.loads(contents), dict):
        raise ValueError("The backup must contain a configuration object.")
    return contents


def _replace_config_contents(config_file, contents):
    """Atomically replace *config_file* with *contents* (temp file + fsync + os.replace)."""
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


# ---------------------------------------------------------------------------
# Startup recovery of an unreadable config.json
# ---------------------------------------------------------------------------

def move_corrupt_config_aside(path=None):
    """Rename an unreadable config.json to ``config.corrupt-<time>.json`` (kept, never deleted).

    Returns the new path, or None when there was nothing to move or the rename failed.
    """
    config_file = _resolve(path)
    if not os.path.exists(config_file):
        return None
    root, ext = os.path.splitext(config_file)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    target = f"{root}.corrupt-{stamp}{ext}"
    counter = 1
    while os.path.exists(target):
        target = f"{root}.corrupt-{stamp}_{counter}{ext}"
        counter += 1
    try:
        os.replace(config_file, target)
    except OSError:
        try:
            # Locked (Windows): keep a copy instead; the restore then replaces the original.
            shutil.copy2(config_file, target)
        except OSError as e:
            _report(f"Warning: Could not keep a copy of the unreadable config: {e}")
            return None
    return target


def load_config_or_restore_backup(path=None, *, decrypt=True):
    """Load config.json; when it exists but cannot be read, restore a backup first.

    Returns ``(config, notice)``. ``notice`` is None when config.json loaded (or does not
    exist yet: first run, ``config`` is ``{}``). It is read several times first
    (``load_config_retrying``). Then ``recover_unreadable_config`` runs, and ``notice``
    says what it did, for the log.
    """
    config_file = _resolve(path)
    try:
        return load_config_retrying(config_file, decrypt=decrypt), None
    except FileNotFoundError:
        return {}, None
    except Exception as e:
        load_error = e
    _report(f"⚠️ config.json could not be loaded: {load_error}")
    recovery = recover_unreadable_config(config_file, load_error, decrypt=decrypt)
    return recovery['config'], recovery['notice']


def recover_unreadable_config(path, load_error, *, decrypt=True):
    """What to run on after config.json could not be read; returns a dict.

    ``state`` is one of:

    * ``'restored'``: the unreadable file is kept aside (``move_corrupt_config_aside``) and the
      newest valid backup (owner's choice: the least lost) atomically became config.json.
    * ``'defaults'``: kept aside too, but no backup is valid, so ``config`` is ``{}``.
    * ``'protected'``: the file still could not be opened or kept. It may well be fine, so it is
      not replaced and this process cannot write it (``app_paths.protect_file_from_writes``);
      ``config`` comes from the newest valid backup, or is ``{}``.

    ``notice`` says what happened, ``corrupt_copy`` is the kept file or None.
    """
    config_file = _resolve(path)
    # Read the backups before anything else writes: a later backup_config_file prunes old ones.
    backups = list_config_backups(config_file)  # newest first
    corrupt_copy = None if isinstance(load_error, OSError) else move_corrupt_config_aside(config_file)
    if isinstance(load_error, OSError) or (corrupt_copy is None and os.path.exists(config_file)):
        return _use_backup_without_writing(config_file, backups, load_error, decrypt)
    kept = (f" The unreadable file was kept as {os.path.basename(corrupt_copy)}."
            if corrupt_copy else "")
    for entry in backups:
        try:
            contents = _read_backup_contents(entry['path'])
            _replace_config_contents(config_file, contents)
            config = load_config(config_file, decrypt=decrypt)
        except Exception as e:
            _report(f"⚠️ Skipping config backup {entry['name']}: {e}")
            continue
        notice = (f"config.json could not be loaded ({load_error}); restored settings from "
                  f"backup {entry['name']}.{kept}")
        _report(f"♻️ {notice}")
        return {'config': config, 'notice': notice, 'state': 'restored', 'corrupt_copy': corrupt_copy,
                'backup': entry['name']}
    notice = (f"config.json could not be loaded ({load_error}) and no valid backup was found; "
              f"settings were reset to defaults.{kept}")
    _report(f"⚠️ {notice}")
    return {'config': {}, 'notice': notice, 'state': 'defaults', 'corrupt_copy': corrupt_copy, 'backup': None}


def _use_backup_without_writing(config_file, backups, load_error, decrypt):
    """``recover_unreadable_config`` for a file that must stay as it is: see ``'protected'`` there."""
    app_paths.protect_file_from_writes(config_file)
    unchanged = ("config.json was not changed, and changes made now will not be saved. "
                 "Close whatever is holding the file and restart Glossarion.")
    for entry in backups:
        try:
            config = json.loads(_read_backup_contents(entry['path']))
            if decrypt:
                import api_key_encryption
                config = api_key_encryption.decrypt_config(config)
        except Exception as e:
            _report(f"⚠️ Skipping config backup {entry['name']}: {e}")
            continue
        notice = (f"config.json could not be read ({load_error}); using settings from backup "
                  f"{entry['name']} for now. {unchanged}")
        _report(f"⚠️ {notice}")
        return {'config': config, 'notice': notice, 'state': 'protected', 'corrupt_copy': None,
                'backup': entry['name']}
    notice = f"config.json could not be read ({load_error}) and no valid backup was found; using defaults for now. {unchanged}"
    _report(f"⚠️ {notice}")
    return {'config': {}, 'notice': notice, 'state': 'protected', 'corrupt_copy': None, 'backup': None}


# ---------------------------------------------------------------------------
# Reset Settings to Defaults (Other Settings > Danger Zone)
# ---------------------------------------------------------------------------

#: What the reset keeps: the list of the desktop confirmation (other_settings
#: prefixes "This will restart the application." and a blank line).
RESET_PRESERVED_TEXT = (
    "The following will be PRESERVED:\n"
    "• Main API Key\n"
    "• Multi-API Keys\n"
    "• Fallback Keys\n"
    "• Replicate API Key\n"
    "• Azure Vision Key & Endpoint\n"
    "• Azure Document Intelligence Key & Endpoint\n"
    "• Google Vision Credentials Path\n"
    "• Selected Model\n"
    "• Prompt Profiles & Active Profile\n"
    "• Multi-Key & Fallback Key Mode Toggles\n"
    "• QA Scanner Excluded Characters\n\n"
    "All other settings (history limits, custom endpoints, etc.) will be lost."
)


def reset_preserved_keys(current_config):
    """The config keys "Reset Settings to Defaults" keeps (``keys_to_preserve``).

    Moved from other_settings ``_reset_config_to_defaults``: API keys and every key pool,
    their toggles, Replicate / Azure / Google credentials, the model, the prompt profiles
    and the active profile, and only ``excluded_characters`` of the QA scanner settings.
    Values are the config's own objects (not copies). The desktop writes the result as the
    new config.json and restarts; the mobile Danger zone writes it through its config store.
    """
    # Preservation logic
    keys_to_preserve = {}

    # 1. Main API Key
    if 'api_key' in current_config:
        keys_to_preserve['api_key'] = current_config['api_key']

    # 2. Multi API Keys
    if 'multi_api_keys' in current_config:
        keys_to_preserve['multi_api_keys'] = current_config['multi_api_keys']

    # 3. Fallback Keys
    if 'fallback_keys' in current_config:
        keys_to_preserve['fallback_keys'] = current_config['fallback_keys']

    # 3b. Dedicated key pools managed by the Multi API Key Manager
    for _pool_key in (
        'glossary_keys',
        'glossary_refinement_keys',
        'metadata_keys',
        'qa_scan_keys',
        'ai_truncation_detection_keys',
        'rolling_summary_keys',
        'truncation_retry_keys',
        'inpainter_keys',
        'tts_keys',
    ):
        if _pool_key in current_config:
            keys_to_preserve[_pool_key] = current_config[_pool_key]

    # 4. Replicate API Key
    if 'replicate_api_key' in current_config:
        keys_to_preserve['replicate_api_key'] = current_config['replicate_api_key']

    # 5. Model Name
    if 'model' in current_config:
        keys_to_preserve['model'] = current_config['model']

    # 6. Azure Computer Vision credentials
    if 'azure_vision_key' in current_config:
        keys_to_preserve['azure_vision_key'] = current_config['azure_vision_key']
    if 'azure_vision_endpoint' in current_config:
        keys_to_preserve['azure_vision_endpoint'] = current_config['azure_vision_endpoint']

    # 7. Azure Document Intelligence credentials
    if 'azure_document_intelligence_key' in current_config:
        keys_to_preserve['azure_document_intelligence_key'] = current_config['azure_document_intelligence_key']
    if 'azure_document_intelligence_endpoint' in current_config:
        keys_to_preserve['azure_document_intelligence_endpoint'] = current_config['azure_document_intelligence_endpoint']

    # 8. Google Vision credentials path
    if 'google_vision_credentials' in current_config:
        keys_to_preserve['google_vision_credentials'] = current_config['google_vision_credentials']
    if 'google_cloud_credentials' in current_config:
        keys_to_preserve['google_cloud_credentials'] = current_config['google_cloud_credentials']

    # 9. Prompt Profiles
    if 'prompt_profiles' in current_config:
        keys_to_preserve['prompt_profiles'] = current_config['prompt_profiles']
    if 'active_profile' in current_config:
        keys_to_preserve['active_profile'] = current_config['active_profile']

    # 10. Multi-Key and Fallback Key Toggle States
    if 'use_multi_api_keys' in current_config:
        keys_to_preserve['use_multi_api_keys'] = current_config['use_multi_api_keys']
    if 'use_fallback_keys' in current_config:
        keys_to_preserve['use_fallback_keys'] = current_config['use_fallback_keys']
    for _toggle_key in (
        'use_glossary_keys',
        'use_glossary_refinement_keys',
        'use_metadata_keys',
        'use_qa_scan_keys',
        'use_ai_truncation_detection_keys',
        'use_rolling_summary_keys',
        'use_truncation_retry_keys',
        'use_inpainter_keys',
        'use_tts_keys',
    ):
        if _toggle_key in current_config:
            keys_to_preserve[_toggle_key] = current_config[_toggle_key]

    # 11. QA Scanner Excluded Characters
    if 'qa_scanner_settings' in current_config:
        qa_settings = current_config['qa_scanner_settings']
        if isinstance(qa_settings, dict) and 'excluded_characters' in qa_settings:
            # Preserve only the excluded_characters field from QA settings
            if 'qa_scanner_settings' not in keys_to_preserve:
                keys_to_preserve['qa_scanner_settings'] = {}
            keys_to_preserve['qa_scanner_settings']['excluded_characters'] = qa_settings['excluded_characters']
    return keys_to_preserve
