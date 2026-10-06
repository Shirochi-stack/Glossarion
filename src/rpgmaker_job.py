"""rpgmaker_job: the desktop RPG Maker game runner (RpgMakerJobMixin) and the mobile game-folder entry.

Shared GUI-free core (Glossarion mobile rewrite, milestone U7). ``_process_rpgmaker_game``
(``translator_gui.py`` @ 41814faa, 24853-25279) moved verbatim out of ``TranslatorGUI``.
``run_translation_direct`` (``translation_pipeline``) dispatches ``.exe`` inputs to it; its
mixin inherits ``RpgMakerJobMixin`` in place of the U3 placeholder, so ``TranslatorGUI`` and
``HeadlessOwner`` both run this code. The runner drives ``rpgmaker_handler`` itself
(``extract_all``, chunking with token budgets, resume/scrub/consistency of
``GTool_Translation/progress.json``, the parallel ``UnifiedClient.send`` loop,
``apply_translations``; image mode: ``translate_game_images``); it never calls
``rpgmaker_handler.process_game``.

Edit made while moving (everything else is byte-for-byte): the game folder is
``rpgmaker_game_dir(exe_path)``: the ``.exe``'s folder exactly as before, or the path itself
when it is a folder (the mobile entry below).

Mobile entry (next to the desktop ``.exe`` path; UI_SPEC section 4.10): a picked game folder
or a ZIP of one. ``apply_translations`` writes into the game's data folder and the runner
keeps its progress in ``<game>/GTool_Translation``, so the game must sit in a writable
folder: ``prepare_rpgmaker_game(source, work_dir)`` extracts a ZIP (or copies a read-only
folder) into *work_dir* and returns the game root (the folder ``rpgmaker_handler`` detects
a version in). Each extraction / copy has a folder of its own, named after the source and
a hash of it (the ZIP's contents, the folder's real path), and records its source in
``SOURCE_MARKER``: another copy of the same ZIP resumes from its ``GTool_Translation``,
while a different game that happens to have the same file or folder name never reuses it.
``_register_rpgmaker_game_input`` does that and registers the folder in
``_rpgmaker_game_inputs``, the only folders ``run_translation_direct`` routes to this runner
(desktop never registers one, so a folder input stays "Unsupported file type" there)::

    game_dir = owner._register_rpgmaker_game_input(source, work_dir)
    request = owner._prepare_translation_run([game_dir])
    owner._translation_worker(request)

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import hashlib
import json
import os
import shutil
import threading
import zipfile

__all__ = [
    "COPY_SUBFOLDER",
    "RPGMAKER_GAME_INPUTS_ATTR",
    "RpgMakerJobMixin",
    "SOURCE_MARKER",
    "ZIP_SUBFOLDER",
    "find_rpgmaker_game_root",
    "prepare_rpgmaker_game",
    "rpgmaker_game_dir",
]

#: Owner attribute listing the game folders the mobile entry registered for this run.
RPGMAKER_GAME_INPUTS_ATTR = "_rpgmaker_game_inputs"
#: Written last into a finished ZIP extraction / folder copy: the source it was made from.
SOURCE_MARKER = ".glossarion_rpg_source.json"
#: Subfolders of the mobile work folder (ZIP extractions and folder copies never share a name).
ZIP_SUBFOLDER = "zips"
COPY_SUBFOLDER = "folders"

_PREPARE_LOCKS = {}
_PREPARE_LOCKS_GUARD = threading.Lock()


def rpgmaker_game_dir(path):
    """Game folder of a run input: the ``.exe``'s folder (desktop), or *path* itself when it is a folder."""
    if os.path.isdir(path):
        return os.path.abspath(path)
    return os.path.dirname(os.path.abspath(path))


def _detects_game(folder):
    import rpgmaker_handler

    try:
        version, _data_dir = rpgmaker_handler.detect_version(folder)
    except OSError:
        return False
    return version != rpgmaker_handler.RPGMakerVersion.UNKNOWN


def find_rpgmaker_game_root(folder, max_depth=3):
    """The shallowest folder under *folder* (itself included) holding an RPG Maker game, or None.

    A ZIP usually wraps the game in one top folder; ``GTool_Translation`` (the runner's own
    output) is never searched.
    """
    folder = os.path.abspath(folder)
    level = [folder]
    for _depth in range(max_depth + 1):
        next_level = []
        for candidate in level:
            if _detects_game(candidate):
                return candidate
            try:
                entries = sorted(os.scandir(candidate), key=lambda e: e.name.lower())
            except OSError:
                continue
            next_level.extend(e.path for e in entries
                              if e.is_dir(follow_symlinks=False) and e.name != "GTool_Translation")
        level = next_level
        if not level:
            break
    return None


def _safe_extract(zip_path, target):
    """Extract *zip_path* into *target*, refusing members that would land outside it."""
    root = os.path.realpath(target)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename.replace("\\", "/")
            dest = os.path.realpath(os.path.join(root, name))
            if dest != root and not dest.startswith(root + os.sep):
                raise ValueError(f"unsafe path in ZIP: {member.filename}")
        zf.extractall(root)


def _is_writable_game(game_dir):
    import rpgmaker_handler

    try:
        _version, data_dir = rpgmaker_handler.detect_version(game_dir)
    except OSError:
        data_dir = ""
    folders = [game_dir] + ([data_dir] if data_dir else [])
    return all(os.access(f, os.W_OK) for f in folders)


def _target_lock(target):
    """One lock per extraction / copy folder: a scan and a job may prepare the same game at once."""
    key = os.path.normcase(os.path.abspath(target))
    with _PREPARE_LOCKS_GUARD:
        lock = _PREPARE_LOCKS.get(key)
        if lock is None:
            lock = _PREPARE_LOCKS[key] = threading.Lock()
        return lock


def _zip_fingerprint(zip_path):
    """The ZIP's contents as its central directory lists them (member names, CRC-32s, sizes).

    Another copy of the same archive (a picker copies the file it hands over) gives the same
    fingerprint; a different game or another release under the same file name does not.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = sorted(zf.infolist(), key=lambda info: info.filename)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP file: {os.path.basename(zip_path)} ({exc})") from exc
    digest = hashlib.sha256()
    for info in members:
        digest.update(f"{info.filename}\0{info.CRC:08x}\0{info.file_size}\n".encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def _folder_fingerprint(folder):
    """A read-only game folder is identified by its real path (its copy keeps the resume progress)."""
    path = os.path.normcase(os.path.realpath(folder))
    return hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest()


def _work_folder(work_dir, subfolder, name, fingerprint):
    """``<work_dir>/<subfolder>/<name>-<first 8 hex digits of fingerprint>``."""
    name = (str(name or "").strip().rstrip(". ") or "game")[:80]
    return os.path.join(os.path.abspath(work_dir), subfolder, f"{name}-{fingerprint[:8]}")


def _marker_matches(folder, fingerprint):
    try:
        with open(os.path.join(folder, SOURCE_MARKER), "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("fingerprint") == fingerprint


def _write_marker(folder, record):
    path = os.path.join(folder, SOURCE_MARKER)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def prepare_rpgmaker_game(source, work_dir, *, log=print, fresh=False):
    """A writable RPG Maker game folder for *source* (a game folder, a ZIP of one, or a game ``.exe``).

    * folder / ``.exe``: the game root itself when it is writable, else a copy in
      ``<work_dir>/folders/<folder name>-<hash of its real path>/<folder name>`` (kept
      between runs: its ``GTool_Translation`` holds the resume progress);
    * ZIP: extracted once into ``<work_dir>/zips/<zip name>-<hash of its contents>`` and the
      game root inside it. Another copy of the same archive resumes there; a different
      archive with the same name gets its own folder.

    An extraction / copy is reused only when its ``SOURCE_MARKER`` (written once it is
    complete) names the same source fingerprint; otherwise, or when *fresh*, it is rebuilt.
    Calls preparing the same folder are serialised.

    Raises ``ValueError`` when *source* does not exist, is not a valid ZIP or holds no RPG
    Maker game.
    """
    source = os.path.abspath(os.fspath(source))
    if not os.path.exists(source):
        raise ValueError(f"Game not found: {source}")
    if os.path.isfile(source) and source.lower().endswith(".zip"):
        if not work_dir:
            raise ValueError("A ZIP game needs a work folder to be extracted into")
        fingerprint = _zip_fingerprint(source)
        name = os.path.splitext(os.path.basename(source))[0] or "game"
        target = _work_folder(work_dir, ZIP_SUBFOLDER, name, fingerprint)
        with _target_lock(target):
            root = None
            if not fresh and _marker_matches(target, fingerprint):
                root = find_rpgmaker_game_root(target)
            if root is None:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                os.makedirs(target, exist_ok=True)
                log(f"📦 Extracting {os.path.basename(source)} into {target}")
                try:
                    _safe_extract(source, target)
                    root = find_rpgmaker_game_root(target)
                    if root is None:
                        raise ValueError(f"No RPG Maker game found in {os.path.basename(source)}")
                except BaseException:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
                _write_marker(target, {"kind": "zip", "source": source, "fingerprint": fingerprint})
        return root
    folder = rpgmaker_game_dir(source)
    root = find_rpgmaker_game_root(folder)
    if root is None:
        raise ValueError(f"No RPG Maker game found in {folder}")
    if _is_writable_game(root) or not work_dir:
        return root
    fingerprint = _folder_fingerprint(root)
    name = os.path.basename(root.rstrip(os.sep)) or "game"
    container = _work_folder(work_dir, COPY_SUBFOLDER, name, fingerprint)
    target = os.path.join(container, name)
    with _target_lock(container):
        if fresh or not (_marker_matches(container, fingerprint) and os.path.isdir(target)):
            if os.path.isdir(container):
                shutil.rmtree(container)
            os.makedirs(container, exist_ok=True)
            log(f"📁 Copying {os.path.basename(root)} into {target} (the game folder is read-only)")
            try:
                shutil.copytree(root, target)
            except BaseException:
                shutil.rmtree(container, ignore_errors=True)
                raise
            _write_marker(container, {"kind": "folder", "source": root, "fingerprint": fingerprint})
    return target


class RpgMakerJobMixin:
    """RPG Maker games through GTool (moved verbatim; see the module docstring)."""

    def _process_rpgmaker_game(self, exe_path):
        """Process an RPG Maker game executable via GTool pipeline.

        Mode behaviour:
          - Text mode  → text-only translation (unchanged)
          - Image mode → image-asset-only translation (no text)
        """
        try:
            import rpgmaker_handler

            self.append_log(f"\n{'='*60}")
            self.append_log(f"🎮 GTool: RPG Maker Game Translation")
            self.append_log(f"📁 Game: {os.path.basename(exe_path)}")
            self.append_log(f"{'='*60}")

            # Load translation modules if needed
            if not self._modules_loaded:
                if not self._lazy_load_modules():
                    self.append_log("❌ Failed to load translation modules")
                    return False

            game_dir = rpgmaker_game_dir(exe_path)

            # Detect version & data dir (needed by both modes)
            version, data_dir, all_strings = rpgmaker_handler.extract_all(
                game_dir, self.append_log)

            if version == rpgmaker_handler.RPGMakerVersion.UNKNOWN:
                self.append_log("❌ Could not detect RPG Maker version")
                return False

            # ── Check output mode ────────────────────────────────
            image_mode = (
                os.environ.get('ENABLE_IMAGE_TRANSLATION', '0') == '1'
                or getattr(self, 'enable_image_translation_var', False)
            )

            # Set up common env / client
            api_key = self.api_key_entry.text()
            output_lang = self.config.get('output_language', 'English')
            from unified_api_client import UnifiedClient
            gtool_out = os.path.join(game_dir, "GTool_Translation")

            # Initialize key pools from config (normally done by _get_environment_variables,
            # but image mode bypasses that path)
            try:
                # Env toggles
                os.environ['USE_MULTI_KEYS'] = '1' if self.config.get('use_multi_api_keys', False) else '0'
                os.environ['USE_FALLBACK_KEYS'] = '1' if self.config.get('use_fallback_keys', False) else '0'
                os.environ['USE_MAIN_KEY_FALLBACK'] = '1' if self.config.get('use_main_key_fallback', True) else '0'
                os.environ['FALLBACK_KEY_SHUFFLE'] = '1' if self.config.get('fallback_key_shuffle', False) else '0'
                os.environ['USE_GLOSSARY_KEYS'] = '1' if self.config.get('use_glossary_keys', False) else '0'
                refinement_keys_enabled = self.config.get('use_glossary_refinement_keys', False)
                refinement_keys = self.config.get('glossary_refinement_keys', [])
                metadata_keys_enabled = self.config.get('use_metadata_keys', False)
                metadata_keys = self.config.get('metadata_keys', [])
                vision_keys_enabled = self.config.get('use_qa_scan_keys', False)
                vision_keys = self.config.get('qa_scan_keys', [])
                rolling_summary_keys_enabled = self.config.get('use_rolling_summary_keys', False)
                rolling_summary_keys = self.config.get('rolling_summary_keys', [])
                truncation_retry_keys_enabled = self.config.get('use_truncation_retry_keys', False)
                truncation_retry_keys = self.config.get('truncation_retry_keys', [])
                inpainter_keys_enabled = self.config.get('use_inpainter_keys', False)
                inpainter_keys = self.config.get('inpainter_keys', [])
                os.environ['USE_VISION_KEYS'] = '1' if vision_keys_enabled else '0'
                os.environ['USE_QA_SCAN_KEYS'] = os.environ['USE_VISION_KEYS']
                os.environ['USE_ROLLING_SUMMARY_KEYS'] = '1' if rolling_summary_keys_enabled else '0'
                os.environ['USE_TRUNCATION_RETRY_KEYS'] = '1' if truncation_retry_keys_enabled else '0'
                os.environ['USE_INPAINTER_KEYS'] = '1' if inpainter_keys_enabled else '0'
                os.environ['FALLBACK_KEYS'] = json.dumps(self.config.get('fallback_keys', []))
                os.environ['GLOSSARY_API_KEYS'] = json.dumps(self.config.get('glossary_keys', []))
                os.environ['USE_GLOSSARY_REFINEMENT_KEYS'] = '1' if refinement_keys_enabled else '0'
                os.environ['GLOSSARY_REFINEMENT_API_KEYS'] = json.dumps(refinement_keys)
                os.environ['USE_METADATA_KEYS'] = '1' if metadata_keys_enabled else '0'
                os.environ['METADATA_API_KEYS'] = json.dumps(metadata_keys)
                os.environ['VISION_API_KEYS'] = json.dumps(vision_keys)
                os.environ['QA_SCAN_API_KEYS'] = os.environ['VISION_API_KEYS']
                os.environ['ROLLING_SUMMARY_API_KEYS'] = json.dumps(rolling_summary_keys)
                os.environ['TRUNCATION_RETRY_API_KEYS'] = json.dumps(truncation_retry_keys)
                os.environ['INPAINTER_API_KEYS'] = json.dumps(inpainter_keys)

                # In-memory key pools
                if self.config.get('use_multi_api_keys', False) and self.config.get('multi_api_keys', []):
                    UnifiedClient.set_in_memory_multi_keys(
                        self.config.get('multi_api_keys', []),
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                if self.config.get('use_glossary_keys', False) and self.config.get('glossary_keys', []):
                    UnifiedClient.set_in_memory_glossary_keys(
                        self.config.get('glossary_keys', []),
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                if refinement_keys_enabled and refinement_keys:
                    UnifiedClient.set_in_memory_glossary_refinement_keys(
                        refinement_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                if vision_keys_enabled and vision_keys:
                    ok = UnifiedClient.set_in_memory_vision_keys(
                        vision_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    pool = getattr(UnifiedClient, '_qa_scan_key_pool', None)
                    configured_count = len(vision_keys)
                    loaded_count = len(getattr(pool, 'keys', [])) if pool else 0
                    load_note = f", {loaded_count} loaded" if loaded_count != configured_count else ""
                    status_detail = str(getattr(UnifiedClient, '_last_qa_scan_pool_setup_status', '') or '').strip()
                    status_note = f": {status_detail}" if (not ok and status_detail) else ""
                    self.append_log(f"[GTool] Vision key pool: {configured_count} entries configured{load_note} (setup={'OK' if ok else 'FAILED'}{status_note})")
                else:
                    if vision_keys_enabled:
                        self.append_log("[GTool] Vision Keys enabled but no keys configured")
                if rolling_summary_keys_enabled and rolling_summary_keys:
                    ok = UnifiedClient.set_in_memory_rolling_summary_keys(
                        rolling_summary_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    pool = getattr(UnifiedClient, '_rolling_summary_key_pool', None)
                    pool_count = len(getattr(pool, 'keys', [])) if pool else 0
                    self.append_log(f"[GTool] Rolling summary key pool: {pool_count} keys loaded (setup={'OK' if ok else 'FAILED'})")
                elif rolling_summary_keys_enabled:
                    self.append_log("[GTool] Rolling Summary Keys enabled but no keys configured")
                if truncation_retry_keys_enabled and truncation_retry_keys:
                    ok = UnifiedClient.set_in_memory_truncation_retry_keys(
                        truncation_retry_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    pool = getattr(UnifiedClient, '_truncation_retry_key_pool', None)
                    pool_count = len(getattr(pool, 'keys', [])) if pool else 0
                    self.append_log(f"[GTool] Truncation retry key pool: {pool_count} keys loaded (setup={'OK' if ok else 'FAILED'})")
                elif truncation_retry_keys_enabled:
                    self.append_log("[GTool] Truncation Retry Keys enabled but no keys configured")
                if inpainter_keys_enabled and inpainter_keys:
                    ok = UnifiedClient.set_in_memory_inpainter_keys(
                        inpainter_keys,
                        force_rotation=self.config.get('force_key_rotation', True),
                        rotation_frequency=self.config.get('rotation_frequency', 1),
                    )
                    pool = getattr(UnifiedClient, '_inpainter_key_pool', None)
                    configured_count = len(inpainter_keys)
                    loaded_count = len(getattr(pool, 'keys', [])) if pool else 0
                    load_note = f", {loaded_count} loaded" if loaded_count != configured_count else ""
                    status_detail = str(getattr(UnifiedClient, '_last_inpainter_pool_setup_status', '') or '').strip()
                    status_note = f": {status_detail}" if (not ok and status_detail) else ""
                    self.append_log(f"[GTool] Image gen/edit key pool: {configured_count} entries configured{load_note} (setup={'OK' if ok else 'FAILED'}{status_note})")
                elif inpainter_keys_enabled:
                    self.append_log("[GTool] Image Gen / Edit Keys enabled but no keys configured")
            except Exception as e:
                self.append_log(f"[GTool] ⚠️ Key pool init error: {e}")

            # ── IMAGE MODE: image-asset-only translation ─────────
            if image_mode:
                # MV/MZ only (older engines don't use JS encryption)
                if version not in (rpgmaker_handler.RPGMakerVersion.MV,
                                   rpgmaker_handler.RPGMakerVersion.MZ):
                    self.append_log("⚠️ Image translation is only supported for MV/MZ games")
                    return False

                self.append_log("🖼️ Output mode: Image — translating game image assets only")

                # Set API call delay from GUI
                os.environ['SEND_INTERVAL_SECONDS'] = str(self.delay_entry.text() or '2.0')
                os.environ['API_QUEUE_SIZE'] = self.api_queue_entry.text().strip() or '4'

                # Determine parallelism from batch settings
                use_batch = getattr(self, 'batch_translation_var', True)
                batch_size = max(1, int(getattr(self, 'batch_size_var', 5))) if use_batch else 1
                if use_batch:
                    self.append_log(f"⚡ Batch mode: {batch_size} parallel workers")

                img_client = UnifiedClient(
                    model=self.model_var,
                    api_key=api_key,
                    output_dir=gtool_out,
                )

                # Load system prompt from the image profile
                img_prompt = ""
                try:
                    profiles = getattr(self, 'prompt_profiles', {})
                    if "RPGMaker_GTool_Image" in profiles:
                        img_prompt = profiles["RPGMaker_GTool_Image"]
                    elif hasattr(self, 'default_prompts') and "RPGMaker_GTool_Image" in self.default_prompts:
                        img_prompt = self.default_prompts["RPGMaker_GTool_Image"]
                except Exception:
                    pass

                # Load filter prompts from settings
                filter_sys = getattr(self, 'gtool_filter_user_prompt_var', '') or ''  # system prompt (legacy var name)
                filter_usr = getattr(self, 'gtool_scan_user_prompt_var', '') or ''    # user prompt

                img_count = rpgmaker_handler.translate_game_images(
                    game_dir=game_dir,
                    data_dir=data_dir,
                    client=img_client,
                    target_lang=output_lang,
                    system_prompt=img_prompt,
                    filter_system_prompt=filter_sys,
                    filter_user_prompt=filter_usr,
                    temperature=float(self.trans_temp.text() or "0.3"),
                    max_tokens=self.max_output_tokens,
                    batch_size=batch_size,
                    api_key=api_key,
                    model=self.model_var,
                    output_dir=gtool_out,
                    log=self.append_log,
                    stop_check=lambda: self.stop_requested,
                )

                self.append_log(f"🎮 GTool: Image translation complete!")
                self.append_log(f"💾 Translation data: {gtool_out}")
                self.append_log(f"📁 Backups: {os.path.join(gtool_out, 'originals_backup')}")
                return img_count > 0

            # ── TEXT MODE: text-only translation (original path) ──
            if not all_strings:
                self.append_log("❌ No translatable strings found - is this an RPG Maker game?")
                return False

            total_strings = sum(len(v) for v in all_strings.values())
            self.append_log(f"📊 Found {total_strings} translatable strings")

            # Build chunks — use tiktoken for accurate token counting.
            # Budget = output_limit / compression_factor (each JP token
            # typically expands to ~compression_factor EN tokens in output).
            compression_factor = float(getattr(self, 'compression_factor_var', '3.0') or '3.0')
            if compression_factor <= 0:
                compression_factor = 3.0
            safety_margin = 200
            max_chunk_tokens = max(500, int((self.max_output_tokens - safety_margin) / compression_factor))
            chunks = rpgmaker_handler.build_translation_chunks(all_strings, max_tokens=max_chunk_tokens)
            self.append_log(f"📦 Split into {len(chunks)} translation chunks "
                          f"(max {max_chunk_tokens:,} tokens/chunk, "
                          f"compression: {compression_factor}x, "
                          f"output limit: {self.max_output_tokens:,})")

            # Load progress for resume
            progress = rpgmaker_handler.load_progress(game_dir)
            # Scrub empty-valued and escape-code-only entries from previous runs
            # (e.g. AI returned empty or bare \\c translation, old code stored it as "done")
            bad_keys = [k for k, v in progress.items()
                        if isinstance(v, str) and (not v or not v.strip()
                                                   or rpgmaker_handler._is_escape_only(v))]
            if bad_keys:
                for k in bad_keys:
                    del progress[k]
                rpgmaker_handler.save_progress(game_dir, progress)
                self.append_log(f"🧹 Cleaned {len(bad_keys)} invalid translations from progress")

            # Detect stale translations from previous runs where originals
            # were corrupted (now restored from backup).
            stale_count = rpgmaker_handler.scrub_stale_progress(
                progress, all_strings, self.append_log)
            if stale_count:
                rpgmaker_handler.save_progress(game_dir, progress)

            # Ensure identical originals get the same translation
            # (e.g. '留奈' must always be 'Runa', never sometimes 'Luna')
            consistency_fixes = rpgmaker_handler.enforce_translation_consistency(
                progress, all_strings, self.append_log)
            if consistency_fixes:
                rpgmaker_handler.save_progress(game_dir, progress)

            if progress:
                self.append_log(f"📋 Resuming: {len(progress)} strings already translated")

            # Create translation map file
            trans_path = rpgmaker_handler.create_translation_file(
                game_dir, all_strings, self.append_log)

            # Set up environment for translation
            system_prompt = self.prompt_text.toPlainText().strip()
            system_prompt = system_prompt.replace('{target_lang}', output_lang)
            import re
            system_prompt = re.sub(r'\s*\{split_marker_instruction\}\s*', '', system_prompt)

            os.environ['API_KEY'] = api_key
            os.environ['SYSTEM_PROMPT'] = system_prompt
            os.environ['MODEL'] = self.model_var
            os.environ['IS_TEXT_FILE_TRANSLATION'] = '1'

            # Set API call delay from GUI (same as image mode + main pipeline)
            os.environ['SEND_INTERVAL_SECONDS'] = str(self.delay_entry.text() or '2.0')
            os.environ['API_QUEUE_SIZE'] = self.api_queue_entry.text().strip() or '4'
            os.environ['THREAD_SUBMISSION_DELAY_SECONDS'] = self.thread_delay_entry.text().strip() or '0.0001'

            # Determine parallelism from batch settings
            use_batch = getattr(self, 'batch_translation_var', True)
            batch_size = max(1, int(getattr(self, 'batch_size_var', 5))) if use_batch else 1
            if use_batch:
                self.append_log(f"⚡ Batch mode: {batch_size} parallel workers")

            # Pre-filter chunks to only those with untranslated keys
            pending_chunks = []
            for i, chunk in enumerate(chunks):
                new_keys = []
                new_texts = []
                for k, t in zip(chunk["keys"], chunk["texts"]):
                    if k not in progress or not progress[k].strip():
                        new_keys.append(k)
                        new_texts.append(t)
                if new_keys:
                    pending_chunks.append((i, {"keys": new_keys, "texts": new_texts}))

            self.append_log(f"📦 {len(pending_chunks)} chunks need translation (of {len(chunks)} total)")

            translated_count = 0
            progress_lock = __import__('threading').Lock()

            # Tell _send_core not to serialise requests behind the sequential lock
            _prev_batch = os.environ.get('BATCH_TRANSLATION', '')
            if batch_size > 1:
                os.environ['BATCH_TRANSLATION'] = '1'

            # Single shared client — same pattern as Chapter_Extractor / image mode.
            # BATCH_TRANSLATION=1 bypasses _sequential_send_lock in _send_core.
            shared_client = UnifiedClient(
                model=self.model_var,
                api_key=api_key,
                output_dir=gtool_out,
            )

            def _translate_chunk(chunk_info):
                """Worker function for translating a single chunk."""
                idx, sub_chunk = chunk_info
                source = rpgmaker_handler.format_chunk_for_translation(sub_chunk)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source},
                ]
                result = shared_client.send(
                    messages=messages,
                    temperature=float(self.trans_temp.text() or "0.3"),
                    max_tokens=self.max_output_tokens,
                )
                response_text = ""
                if result:
                    if isinstance(result, tuple):
                        response_text = result[0] if result[0] else ""
                    elif isinstance(result, str):
                        response_text = result
                    elif hasattr(result, 'content'):
                        response_text = result.content or ""
                if response_text:
                    parsed = rpgmaker_handler.parse_translated_chunk(
                        response_text, sub_chunk)
                    return idx, parsed, len(sub_chunk["keys"])
                return idx, {}, len(sub_chunk["keys"])

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {}
                for chunk_info in pending_chunks:
                    if self.stop_requested:
                        break
                    future = pool.submit(_translate_chunk, chunk_info)
                    futures[future] = chunk_info[0]  # map future -> chunk index

                for future in as_completed(futures):
                    if self.stop_requested:
                        self.append_log("⏹️ Translation stopped by user")
                        break
                    chunk_idx = futures[future]
                    try:
                        idx, parsed, total_keys = future.result()
                        if parsed:
                            with progress_lock:
                                progress.update(parsed)
                                translated_count += len(parsed)
                                rpgmaker_handler.save_progress(game_dir, progress)
                            self.append_log(
                                f"   ✅ Chunk {idx+1}/{len(chunks)}: "
                                f"{len(parsed)}/{total_keys} translated")
                        else:
                            self.append_log(
                                f"   ⚠️ Chunk {idx+1}/{len(chunks)}: "
                                f"no translations parsed")
                    except Exception as e:
                        self.append_log(f"   ⚠️ Chunk {chunk_idx+1} failed: {e}")

            self.append_log(f"\n📊 Translated {translated_count} strings total")

            # Restore batch mode env var
            if _prev_batch:
                os.environ['BATCH_TRANSLATION'] = _prev_batch
            elif 'BATCH_TRANSLATION' in os.environ:
                del os.environ['BATCH_TRANSLATION']

            # Update translation map
            try:
                with open(trans_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                for full_key, translated in progress.items():
                    parts = full_key.split("::", 1)
                    if len(parts) == 2:
                        fn, key = parts
                        if fn in trans_data and key in trans_data[fn]:
                            trans_data[fn][key]["translated"] = translated
                with open(trans_path, 'w', encoding='utf-8') as f:
                    json.dump(trans_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.append_log(f"⚠️ Failed to update translation map: {e}")

            # Apply translations to game files (always — also applies settings like font size)
            if progress:
                self.append_log("🔧 Applying translations to game files...")
                rpgmaker_handler.apply_translations(
                    data_dir, trans_path, self.append_log, version=version,
                    game_dir=game_dir)

            self.append_log(f"🎮 GTool: Translation complete!")
            self.append_log(f"💾 Translation data: {os.path.join(game_dir, 'GTool_Translation')}")
            self.append_log(f"📁 Backups: {os.path.join(game_dir, 'GTool_Translation', 'originals_backup')}")
            return translated_count > 0

        except Exception as e:
            self.append_log(f"❌ GTool error: {str(e)}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False

    def _register_rpgmaker_game_input(self, source, work_dir=None):
        """Mobile entry: ``prepare_rpgmaker_game`` + register the folder for ``run_translation_direct``.

        Returns the game folder to pass as the run's input (``_prepare_translation_run([it])``).
        """
        game_dir = prepare_rpgmaker_game(source, work_dir, log=self.append_log)
        registered = list(getattr(self, RPGMAKER_GAME_INPUTS_ATTR, None) or [])
        if game_dir not in registered:
            registered.append(game_dir)
        setattr(self, RPGMAKER_GAME_INPUTS_ATTR, registered)
        return game_dir
