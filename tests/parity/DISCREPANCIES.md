# Desktop discrepancies recorded during the mobile rewrite

Defaults and bugs noticed while extracting desktop code. Recorded, not fixed;
each needs a separate, user-approved change that also updates the goldens.

## U1 shared-core P1 (app_paths, config_store, prompt_defaults, metadata_defaults, ollama_settings, key_pools, output_naming)

Recorded while moving code at BASE_SHA 4c825d81; none of these were fixed (the
goldens and tests/test_shared_core_p1.py pin the existing behaviour). Line
numbers refer to `git show 4c825d81:src/<file>`.

### Desktop defaults / bugs found (not fixed)

1. **Plain-text config write at startup.** `TranslatorGUI.__init__` (translator_gui.py
   13745-13751): when `auto_update_check` is missing it calls
   `_atomic_json_write(CONFIG_FILE, self.config)` with the *decrypted* config, so API
   keys hit the disk unencrypted. The later auto-encryption pass (14880-14897) only
   re-saves when `api_key` / `replicate_api_key` lack `ENC:`, so other key fields
   (pools, provider keys) can stay plain until the next `save_config`.
2. **`book_title_prompt` has two defaults.** `MetadataBatchTranslatorUI._initialize_default_prompts`
   (metadata_batch_translator.py 216-219, now `metadata_defaults`) writes `""`; it runs
   inside the `__init__` config block (14270, again at 14538) before `_init_variables` (16503-16504),
   whose default is "Translate this book title to {target_lang} while retaining any
   acronyms:". On a fresh install the effective value is `""` (golden `fresh_install`).
3. **Stale `config['prompt_profiles']` after reconciliation.** `_init_variables`
   (16510-16555) assigns `config['prompt_profiles'] = self.prompt_profiles`, then
   replaces `self.prompt_profiles` with a reordered `new_profiles` when built-in profiles
   are missing; the config keeps the old dict until `save_config` copies
   `prompt_profiles` back (settings map 47470).
4. **Two hand-synced profile lists.** `_get_protected_prompt_profiles` (13350-13364) and
   `always_include_profiles` (16515-16530) are the same 13 names maintained twice
   ("keep in sync" comment). They and the `default_prompts` dict stay in translator_gui
   for now: tests/test_sdlxliff_support.py::test_sdlxliff_prompt_profile_is_bootstrapped_and_mirrored
   and tests/test_subtitle_processor.py::test_subtitle_prompt_profile_is_built_in_and_mirrored
   regex-match those literals in translator_gui.py.
5. **AI Hunter has two default sets.** `AIHunterConfigGUI.default_ai_hunter`
   (ai_hunter_enhanced.py 47-113, now `default_ai_hunter_config()`) vs
   `ImprovedAIHunterDetection.default_ai_hunter` (1007-1071): thresholds `text` 35 vs 85,
   `character` 90 vs 80; `retry_attempts` 6 vs 3; `detection_mode` weighted_average vs
   multi_method; `methods_required` 3 vs 2; only the GUI set has `ai_hunter_max_workers`,
   only the detection set has `lookback_chapters`. Which applies depends on whether the
   GUI class was constructed before a run (it merges its defaults into the config).
6. **AI Hunter defaults alias the config.** `AIHunterConfigGUI.__init__` stores
   `self.default_ai_hunter.copy()` (shallow) or a merge that reuses default sub-dicts, and
   the dialog edits nested values in place (`ai_config['thresholds'][method] = ...`, 939);
   "Reset to defaults" (989) can then restore the edited values within the same session.
   `ai_hunter_max_workers` is `cpu_count // 2` of the machine that first saved it.
7. **Library registry lookups ignore the library seam.** `_library_origins_raw_sources_for_stem`
   / `_library_raw_inputs_for_stem` (other_settings.py 152-225, now `output_naming`)
   hardcode `~/Documents/Glossarion/Library` instead of `epub_library.get_library_dir()`,
   so they do not honour `GLOSSARION_LIBRARY_DIR` (mobile HOME is `data/home`). Same
   result on desktop.
8. **`CONFIG_FILE` env is not honoured by the GUI path.** `app_paths.CONFIG_FILE` (was
   translator_gui 1314-1355) only follows `GLOSSARION_APP_DIR`; unified_api_client /
   GlossaryManager read `os.environ['CONFIG_FILE']`. Mobile sets both to the same file;
   if they ever diverge the GUI owner and the backend read different configs.
9. **`_get_app_dir()` is CWD on macOS/Linux** (1398-1417) while `CONFIG_FILE` follows
   `_APP_DIR`, so Payloads/Glossary/output roots follow the launch directory there.
10. **Other config writers skip `save_config`'s rules.** The system-prompt toggle
    (27283-27287) writes `encrypt_config(self.config)` directly (no backup, no
    `_config_restore_pending` guard, no google-credentials re-copy).
11. Cosmetic: config_backup looks for `halgakos.ico` (lowercase; the app ships
    `Halgakos.ico`, so no icon on case-sensitive filesystems); `_backup_config_file`
    computed an unused `base_dir`; the attribute `default_unified_auto_glosary_prompt3`
    is misspelled.

### Behaviour deltas introduced by P1 (intentional, golden-neutral)

- Config load: the sanitizer now runs after `load_config` closed the file. Before, its
  `_atomic_json_write` ran inside `with open(CONFIG_FILE)`; on Windows `os.replace` onto
  the open file fails (WinError 5, verified) and it fell back to a non-atomic direct
  write. Same bytes, now written atomically.
- `_get_environment_variables` evaluates `self.config` once for `apply_key_pools_to_runtime`
  outside the two `try` blocks; an owner without `config` would now raise there instead
  of a few lines later (every owner has `config`).
- `_process_text_file` (retain-extension rename) and `_ollama_settings_env_json` import
  `output_naming` / `ollama_settings` instead of the Qt modules (same functions); on a
  Qt-less runtime the rename now runs instead of being skipped by the `except`.
- `config_store.backup_config_file` / `restore_*` return the backup path (old functions
  returned None); `ensure_metadata_prompt_defaults` returns whether it added a key.
- `ai_hunter_enhanced.default_ai_hunter_config()` is a factory (not the planned
  `DEFAULT_AI_HUNTER_CONFIG` constant) so the cpu-count default and fresh nested dicts
  keep their per-instance behaviour; `merge_ai_hunter_config(existing, default=None)`.

## U2 shared-core P2 (owner_state, run_env, settings_persistence, headless_owner)

Moved at BASE_SHA 96af9adb (the merged U1 commit; its goldens are identical to the U0
goldens for all 12 scenarios). Line numbers refer to `git show 96af9adb:src/translator_gui.py`.
Nothing below was fixed; tests/test_headless_owner.py pins the behaviour.

### Desktop defaults / bugs found (not fixed)

1. **First run and later runs export different values.** The startup `save_config` (run by
   the glossary-mode shortcut handler) rewrites config.json from widget/var values, so a
   desktop restarted on that file differs from its own first run. `HeadlessOwner` built from
   `_collect_live_settings()` equals the restarted desktop exactly; the first-run differences
   are (KNOWN_ROUNDTRIP_DIVERGENCES):
   - number formatting: `SEND_INTERVAL_SECONDS` '5' -> '5.0' (delay_entry text vs
     `safe_float`), `CONNECT_TIMEOUT` '10' -> '10.0', `READ_TIMEOUT` '180' -> '180.0',
     `IMAGE_CHUNK_OVERLAP_PERCENT` '3' -> '3.0' (startup env from `*_var` strings, later
     from the floats save_config stored);
   - **built-in glossary prompts are lost after the first save**: `_glossary_env_mappings`
     (47059-47061) normalises `config['glossary_translation_prompt']` and
     `config['glossary_format_instructions']` to `''` when missing, the startup save persists
     the `''`, and `__init__` (14049-14073) then uses `''` instead of the built-in defaults, so
     `GLOSSARY_TRANSLATION_PROMPT` / `GLOSSARY_FORMAT_INSTRUCTIONS` are empty from the second
     launch on;
   - `EXTRACTION_MODE`: save_config derives `extraction_mode` from `file_filtering_level_var`
     (47770-47779) while the first-run startup env uses `_init_variables`' `extraction_mode_var`
     (e.g. 'smart' at first run, 'full' after a restart for `file_filtering_level='full'`).
2. **`_on_auto_glossary_shortcut_changed` writes config at startup**: the handler that turns a
   fresh install into `AUTO_GLOSSARY_MODE='off'` also runs `save_config(show_message=False)`, so
   every desktop start rewrites config.json even when nothing changed (HeadlessOwner runs the
   same handler with an in-memory `save_config`).
3. **`auto_load_glossary_for_file` is called unguarded** from the glossary-mode handler (inside
   a broad `try/except`), only when exactly one EPUB is selected; it is a desktop method the
   shared mixins cannot provide yet (HeadlessOwner gets it with the U3 glossary pipeline).
4. **Every start with an API key re-saves config.json, and the startup env depends on it.**
   `config_store.load_config` decrypts, so the post-startup auto-encryption check (14753-14768)
   finds a plain `api_key` / `replicate_api_key` on every launch with a key and runs
   `save_config(show_message=False)` after `initialize_environment_variables`. That save's
   env export (`_glossary_env_mappings`, default `'Loose'`) overrides the startup env
   (`initialize_environment_variables`, default `'none'`) for `GLOSSARY_ENTRY_TYPE_FILTER_MODE`
   when the config has no `glossary_entry_type_filter_mode`: a key-less config starts with
   `'none'`, a config with a key with `'Loose'`. The PDF compile backend inherits the process
   env, so its `GLOSSARY_ENTRY_TYPE_FILTER_MODE` follows the same split (translation and glossary
   runs set the key explicitly). Verified with a real offscreen TranslatorGUI
   (`tests/parity/real_gui_probe.py`). The message "API keys encrypted successfully!" is printed
   by HeadlessOwner too although its `save_config` is in memory (verbatim desktop code).
5. **`QLineEdit.setText` raises `TypeError` for a non-string value** (PySide6), so a config.json
   whose `chapter_range` is a number would make `_create_settings_section` fail; `None` reads
   back as `''`. The HeadlessOwner text shims mirror the `None` case and stringify other values
   (a desktop crash is not reproduced).

### Behaviour deltas introduced by U2 (intentional, golden-neutral)

- `_init_gui_backed_state()` holds the GUI-backed state assignments that the section builders
  made while creating widgets (vertex_location_var 17034, deep_scan_var 17059,
  default_model/model_var 22713-22714, context_mode_var 25335, translation_history_rolling
  25407-25408). `__init__` now calls it right before `_setup_gui()`; the statements read only
  config/vars and nothing in `_setup_gui`'s synchronous call closure reads or writes these
  attributes/keys earlier (checked when moving), and `_create_model_section` keeps its local
  as `default_model = self.model_var`.
- The glossary-mode shortcut handler was a nested function of `_create_settings_section`; it
  is now the method `ConfigStateMixin._on_auto_glossary_shortcut_changed(self, index)` (same
  body), connected and called at the same points. Its startup index is
  `_saved_auto_glossary_shortcut_index()`, the startup target language
  `_resolve_startup_target_language()`, the active-profile prompt fill
  `_init_active_profile_prompt()`; `_replay_gui_startup_handlers()` runs the startup handlers in
  desktop order for owners without Qt widgets.
- `__init__`'s second `MetadataBatchTranslatorUI(self)` try (identical to the config-block one)
  is now a call of the `_hook_metadata_defaults()` desktop override holding that code.
- `_InputOutputDialog` sets the Direct Text run attributes through
  `DirectTextRunOptions(...).apply_to(gui)`: same attributes, same order, but every value is
  evaluated before the first attribute is set (the expressions read only dialog state).
- save_config's local `safe_int` / `safe_float` are module functions of
  `settings_persistence` (translator_gui imports them); sections 2-3 and 4 are
  `_apply_live_settings_to_config()` / `_export_settings_env()` called at the same point.
- `_InputOutputDialog._FORCED_STREAM_ENV_KEYS` is an alias of `run_env.FORCED_STREAM_ENV_KEYS`;
  `_apply_forced_streaming_environment` reads the module constant.
- Split-outs called at the original positions: `_glossary_extraction_paths` +
  `_build_glossary_extraction_env` (`_extract_glossary_from_text_file`), `_build_epub_compile_env`
  (`run_epub_converter_direct`), `_build_pdf_compile_env` (`run_pdf_converter_direct`).
- Desktop-only side effects are hooks with GUI-free mixin defaults; TranslatorGUI overrides
  them with the original code: `_can_read_widgets`, `_hook_persist_sanitized_config`,
  `_hook_save_default_config`, `_hook_ensure_executor`, `_hook_metadata_defaults`,
  `_hook_context_mode_layout`. glm_proxy's `set_general_api_mode` needed no hook: its
  ImportError branch already sets the same `AUTHZA_USE_GENERAL_API` env value.
- U2 review: `_create_model_section`'s "Restore saved project selection on startup" block
  (23112-23122: `authgem_auth._cached_project_id[0]`, `_project_set_by_gui[0]`,
  `GOOGLE_CLOUD_PROJECT`) is `ConfigStateMixin._restore_authgem_project_selection()`, called at
  the same point; `_replay_gui_startup_handlers()` runs it first (the model section precedes the
  settings section). The block's local `import os` is the only `os` use in
  `_create_model_section`, so moving it changes no scoping there.
- U2 review: the last step of `__init__` (the auto-encryption check, 14753-14768) is
  `ConfigStateMixin._auto_encrypt_api_keys()`, called at the same point; HeadlessOwner calls it at
  the end of its `__init__` with its in-memory `save_config`.
- U2 review: `_find_raw_source_for_folder` and its closure moved verbatim from epub_library to the
  GUI-free `library_core` (epub_library.py @ 96af9adb: `_special_file_stem` 322, the registries /
  origins block `get_library_dir` 586 .. `_save_origins` 1012-1019, `_FILENAME_STRIP_CHARS` +
  `_norm_book_key` 2205-2422, the resolvers 2863-3064); epub_library re-imports every name (same
  objects) and the moved functions keep logging under the `epub_library` logger. The EPUB compile
  env (`_build_epub_compile_env`, was 38386) imports `library_core` instead of `epub_library`, so a
  desktop compile no longer imports the Qt/WebEngine epub_library module as a side effect (same
  resolver function, same result). Without this, mobile could never resolve the source EPUB
  (epub_library imports PySide6) and compiled with filename ordering. Tests that redirect the
  Library monkeypatch these names on `library_core` too (the moved functions resolve each other
  there).
- U2 review: HeadlessOwner's text shims read `None` back as `''` like Qt (`QLineEdit(None)`,
  `setText(None)`, `setPlainText(None)`); before, `chapter_range: null` exported `CHAPTER_RANGE='None'`.

### U2 review: parity oracle (freezer v2) and goldens

- The legacy boot replay now includes the post-startup auto-encryption `save_config`
  (`legacy_post_startup_block`, both freeze layouts). The 96af9adb oracle (U2 reference) and the
  4c825d81 oracle (U1 reference, tests/test_shared_core_p1.py) were re-frozen and their goldens
  re-captured (the two golden sets are still identical): 10 of 12 scenarios (those with a plain
  `api_key`) changed only in
  `GLOSSARY_ENTRY_TYPE_FILTER_MODE` ('none' -> 'Loose' in the boot env, the epub_compile env delta
  and the pdf_compile backend env) and the boot stdout. `real_gui_probe.py` (now also comparing
  the startup process env) reports MATCH for all 12 scenarios against the new goldens, and
  `real_gui_probe.py --headless` (real GUI vs HeadlessOwner from the same config.json) MATCH for
  all 12 scenarios and for ad-hoc configs with `authgem_project`, a plain key, null
  `chapter_range` / `vertex_ai_location` and two richer configs.
- From U2 on the oracle freezes the shared mixin modules too: `freeze()` saves owner_state /
  run_env / settings_persistence as they are at the frozen commit
  (`legacy_<sha12>__<module>.py`, manifest `frozen_mixins` with their sha256), the legacy owner
  uses those frozen classes (imports between them resolve to the frozen copies), and the
  staleness test checks their digests. Before, an oracle frozen at a commit with the mixins
  (U3 freezing U2) used the live modules, so tier D compared them against themselves.

### Desktop restart is not idempotent (found by the tier R helper, `tests/parity/roundtrip.py`)

`python tests/parity/roundtrip.py --restart` boots each golden scenario, reads the config.json
its startup `save_config` wrote, boots the desktop again from that file and compares the
process env after startup and the translation env. A restart differs from the first session on
these keys (every scenario unless noted). Recorded in `roundtrip.LEGACY_RESTART_DIVERGENCES`;
`test_restart_divergence_list_is_current` fails when one stops happening. HeadlessOwner is
built from saved settings, so on these keys mobile matches the desktop *after* a restart (tier
R vs a restart: 0 differences in all 12 scenarios), not its first session.

1. **Numeric text formatting.** `CONNECT_TIMEOUT` / `READ_TIMEOUT` / `IMAGE_CHUNK_OVERLAP_PERCENT`
   (startup env) and `IMAGE_CHUNK_OVERLAP_PERCENT` (translation env) are `'10'`, `'180'`, `'3'`
   in the first session and `'10.0'`, `'180.0'`, `'3.0'` after a restart: save_config stores
   `safe_float()` values and the init block exports `str(config value)`.
   `SEND_INTERVAL_SECONDS` goes `'5'` -> `'5.0'` the same way (`delay_entry` shows
   `str(config['delay'])`).
2. **Glossary prompts become empty after the first save.** `GLOSSARY_TRANSLATION_PROMPT` and
   `GLOSSARY_FORMAT_INSTRUCTIONS` carry the built-in default text in the first session (init
   block: `config.get(key, default)`). The startup save_config runs `_glossary_env_mappings`,
   which writes `config[key] = config.get(key, '') or ''` for the missing keys, so config.json
   stores `''` and every later session exports `''` (unless the Glossary Manager saved a text).
3. **Extraction mode rewrite** (`pdf_layout_custom_routes_ollama`): `EXTRACTION_MODE` is
   `'smart'` in the first session and `'full'` after a restart; save_config's extraction-mode
   compatibility block copies `file_filtering_level_var` into `config['extraction_mode']`.

## U3 shared-core P3, chain step 1 (stop_control, job_runner, text_jobs, input_preparation)

Moved at BASE_SHA 1719fb59 (the merged U2 commit; its goldens are identical to the 96af9adb
goldens for all 12 scenarios). Line numbers refer to `git show 1719fb59:src/translator_gui.py`.
Nothing below was fixed; tests/test_text_jobs.py, tests/test_job_runner.py and tier D pin the
behaviour.

### Desktop defaults / bugs found (not fixed)

1. **`_process_text_file` leaks two variables past its restore.** `EPUB_PATH` (EPUB inputs) and
   `GOOGLE_APPLICATION_CREDENTIALS` (Vertex models) are set before the argv/env snapshot
   (32220, 32269), so the `finally` restore keeps them for the rest of the session. Mobile's
   job scope (`job_runner.job_process_state`) restores them after the job.
2. **The glossary extractor's restore keeps `large_env`'s overflow store.**
   `_extract_glossary_from_text_file` restores argv + `os.environ` but never calls
   `large_env.clear_store()` (the translation runner does), so oversized glossary values stay
   in the store until the next translation run clears it.
3. **`stop_translation` computes `already_in_graceful_stop` and never uses it** (34844-34847);
   a double click forces the stop whatever the button shows.
4. **Two copies of the helper-process killer.** `stop_glossary_extraction` (35467-35533) has its
   own psutil block with a shorter helper list (no AuthND / Gemini-Free token helpers) and
   touches `GLOSSARY_STOP_FILE` from the background thread; `stop_translation`'s copy is now
   `stop_control.kill_helper_subprocesses`. The glossary stop was not rewired in this step.
5. **The PDF compile never looks at Stop.** `run_pdf_converter_direct` reports
   "PDF Compilation Success" whenever a file exists, even after Stop (the EPUB runner checks
   `stop_requested`). `CompileResult.stopped` is therefore only set for EPUB compiles.
6. **`run_epub_converter_direct` with an unloaded converter** (`fallback_compile_epub is None`)
   fails with "'NoneType' object is not callable" in a message box; only `epub_converter()`
   (the button handler) checks the lazy load first.
7. **Three copies of `is_traditional_translation_api`** (translator_gui, TransateKRtoEN 1169,
   GlossaryManager 470). The translator_gui one now lives in text_jobs; the backend copies
   were left alone.
8. The watchdog bar's display rule (`_update_api_watchdog`: cross-process file aggregation and
   `max(queued entries, scheduler_queued)`) is still GUI-interleaved; `ProgressWatcher` emits
   the raw `get_api_watchdog_state()` fields (mobile has no cross-process watchdog files).

### Behaviour deltas introduced by U3 step 1 (intentional, golden-neutral)

- `TranslatorGUI(TextJobsMixin, InputPreparationMixin, SettingsPersistenceMixin, RunEnvMixin,
  ConfigStateMixin, ...)`; HeadlessOwner gets the same two mixins first. Moved bodies were
  deleted from TranslatorGUI.
- `_process_text_file` / `_extract_glossary_from_text_file`: the inline snapshot is
  `job_runner.scoped_process_state(...).snapshot()` at the same line and `restore()` replaces
  the `finally` statements (same order: argv rebound, `os.environ.clear()` + `update`, then
  `large_env.clear_store()` for translation only). The snapshot also copies the argv list and
  the `large_env` store (unused by the desktop restore). Tier T compares the full sequence of
  environment operations with the frozen legacy code (tests/test_job_runner.py).
- The lazily loaded module globals `translation_main` / `glossary_main` / `fallback_compile_epub`
  are read through the `_backend_entry(name)` hook at the same point (desktop override: the
  translator_gui globals, so tests that patch them still work; mobile default: lazy import).
- The compile runners' message boxes are the `_notify_compile_result(kind, path, error)` hook
  (desktop override: the same `QTimer.singleShot(0, QMessageBox...)` calls) called where the
  `QTimer` lines were; `run_epub_converter_direct` / `run_pdf_converter_direct` keep their
  `finally` and call `_run_epub_compile()` / `_run_pdf_compile()`.
- `_resolve_zip_inputs_for_translation`'s `input_files_updated_signal.emit(resolved)` is
  `_ui_request('input_files_updated', resolved)` (desktop override: `<kind>_signal.emit(*args)`).
- `_convert_zip_input_to_epub_if_needed` passes `getattr(self, '_direct_text_archive_conversion_dir', '')`
  to `input_preparation.resolve_input_to_epub` before the archive-extension check (a read
  without side effects; the old body read it after the check).
- `stop_translation`: the click window is `stop_control.register_stop_click`, the force flags
  `apply_force_stop_flags()`, and the flag protocol `stop_control.request_stop(...)`; the latch
  (graceful_stop_active, stop timestamps, stop_requested) is a local callback.
  `getattr(self, 'wait_for_chunks_var', True)` is read once before the env flags (the old code
  read it twice, after them). The cleanup thread additionally checks
  `mobile_runtime.subprocesses_available()` (env reads of GLOSSARION_NO_PROCESSES /
  GLOSSARION_MOBILE / FLET_PLATFORM) before the psutil helper kill; on desktop it is always true.
- Run start: `run_translation_thread` / `run_glossary_extraction_thread` call
  `stop_control.reset_stop_env`, `make_run_id`, `clear_client_cancellation` and
  `prepare_glossary_stop_file` where the inline statements were (verbatim, checked by AST).
- `_reset_api_watchdog_progress` steps 1-2 (counters + watchdog files) are
  `stop_control.reset_api_watchdog(...)`; HeadlessOwner uses it without the progress bar.
- `epub_library._read_progress_summary` and its closure (special-file rules, gallery and
  sidecar filters) moved verbatim to `library_core` (epub_library re-imports the same objects)
  so `job_runner.ProgressWatcher` reads the Library card numbers without Qt.
- `is_traditional_translation_api` moved to text_jobs; translator_gui re-imports it.
- Mobile side of the same move: the env preview's `restore_mapping` / `isolated_key_pools` moved
  to job_runner (the preview imports them; `ENV_PREVIEW_LOCK` is `job_runner.JOB_LOCK`;
  `scoped_process_env()` is `scoped_process_state(keywise=True, isolate_key_pools=True,
  restore_cwd=True)`), and `job_runner.job_process_state()` is the scope a mobile job runs in.
- `src/mobile/tools/schema_extract.py` maps `library_core.py` to the `library` UI site (like
  `epub_library.py`), so moving the special-file resolvers leaves `settings_schema_data.py`
  byte-identical.
- Desktop tests repointed to where the code now lives: `_src_corpus` (text_jobs,
  input_preparation, job_runner, stop_control), test_ocagy_cli (text-error handler),
  test_pdf_workspace_compiler (compile runner source), test_partial_b2_batching (the
  publish-before-latch order is now asserted on `stop_control.request_stop` + the latch passed by
  `stop_translation`), test_zip_input_routing (binds the `_ui_request` hook), and the local
  test_unified_glossary_wiring (`library_core.py` holds the default special-file list).

### U3 parity harness changes

- freezer: SHARED_MIXIN_MODULES gains translation_pipeline / text_jobs / input_preparation
  (skipped while absent) and SHARED_HELPER_MODULES (job_runner, stop_control) are frozen whole
  without an owner class; frozen modules are compiled with `dont_inherit=True` (the freezer's
  own `from __future__ import annotations` turned their annotations into strings, which breaks
  `@dataclass` in a module that is not in sys.modules); `src/` is put on sys.path before
  loading frozen modules (`python tests/parity/freeze_legacy.py` failed for a U2+ SHA).
- goldens compare callables by their own name (`capture_golden.callable_name`): a recorded
  `__qualname__` names where the code lives (`LegacyMethods._process_text_file.<locals>.<lambda>`
  -> `TextJobsMixin._process_text_file.<locals>.<lambda>`); tier D already compared `__name__`.
  With this, the U1 (4c825d81) and U2 (96af9adb) worktree compositions reproduce their goldens
  unchanged after the U3 moves.
- tier D: `_parity_recorder` (FakeState's recorder) was part of every read set that reaches
  `append_log`, so ~30% of those states replaced the recorder and both sides failed with the same
  AttributeError (lower coverage, no false pass); it is now excluded. Archive / HTML / subtitle
  fixtures (`FUZZ_ARCHIVES`), the `archive_path` / `unique_files` argument profiles, per-entry
  `stubs` (recorders for backend work handed off, e.g. the metadata worker), fixed
  `time.gmtime/localtime` and `datetime.now` (generated EPUBs embed the time) and side `hooks`
  (tier T trace recorders) were added.

## U3 shared-core P4, chain step 2 (translation_pipeline)

Moved at BASE_SHA 1719fb59 on top of chain step 1 (line numbers: `git show
1719fb59:src/translator_gui.py`). `run_translation_thread`'s worker closure (29339-29798) is
`TranslationPipelineMixin._translation_worker(request)` and its run set-up (29081-29336)
`_prepare_translation_run(files=None) -> RunRequest`; `run_translation_direct`, the QA-failure /
multipass planning, `run_glossary_extraction_direct`, the image-folder glossary, glossary
auto-loading / auto-mapping and the input-selection helpers moved verbatim
(`GlossaryPipelineMixin`), plus `_glossary_editor_input_sources` from GlossaryManager_GUI.
Tier T (tests/parity/test_trace_parity.py: 17 scenarios, legacy oracle vs working-tree desktop
on the full record and vs HeadlessOwner + `stop_control` on the mobile projection) and
tests/test_translation_pipeline.py (every moved text = legacy text + the edits below) pin it.

### Desktop defaults / bugs found (not fixed)

1. **Dead Tk leftover in the run set-up.** `if hasattr(self, 'button_run'):
   self.button_run.config(text="⏹ Stop", state="normal")` (29307-29308): the Qt window's button
   is `run_button`, `button_run` never exists, so the branch never runs (`update_run_button()`
   after the thread start does the real update).
2. **`run_translation_direct`'s comment still names `simple_thread_target()`** as the owner of
   the end-of-run reset; that is now `_translation_worker` (comments were moved verbatim).
3. **Raw-input Library registry needs epub_library.** The set-up records inputs with
   `from epub_library import record_library_raw_input` inside `try/except: pass`; builds without
   epub_library (translator_lite / TurboLite specs, and mobile) silently skip it although the
   function lives in the GUI-free `library_core` (epub_library re-exports it). Kept verbatim:
   switching the import would start writing the registry in the lite builds. (U3 fix pass: the
   block is now TranslatorGUI's `_record_library_raw_inputs` hook override, still verbatim; the
   shared set-up no longer imports epub_library on mobile.)
4. **`_await_direct_text_glossary_approval` polls every 0.1 s** for Stop while the GUI decides;
   `stop_requested` is the only way out besides an answer (a closed dialog without an answer
   waits until Stop).
5. **Generative-mode sentinel on an empty selection.** With an image/video model and nothing
   selected the set-up runs on `["__generative_mode__"]` (desktop: prompt-only generation).
   Mobile reaches `_run_generative_prompt_mode`'s U7 placeholder (logged, run fails cleanly).

### Behaviour deltas introduced by U3 step 2 (intentional, golden-neutral)

- `TranslatorGUI(TranslationPipelineMixin, TextJobsMixin, InputPreparationMixin, ...)`;
  `TranslationPipelineMixin` inherits `GlossaryPipelineMixin` -> `PipelineHooksMixin` ->
  `job_runner.JobHooksMixin`; HeadlessOwner gets the same first base. Moved bodies were deleted
  from TranslatorGUI; GlossaryManagerMixin keeps `_glossary_editor_input_sources` as an alias of
  the shared function (the frozen oracle and other users of the GUI mixin still resolve it).
- `run_translation_thread` = its preflight (glossary-run guard, Parallel EPUB Pair notice,
  Run-as-Stop, stop clean-up wait, double click after a graceful stop, model check) +
  `request = self._prepare_translation_run()` (`None` = the old early returns) + the old thread
  launch with `target=self._translation_worker, args=(request,)` (same thread name). Statement
  order is unchanged (tier T full record: logs, Qt stub calls, signals, latches, backend env).
- The closure body is dedented one level; the set-up ends in `return RunRequest(...)` (files,
  run id, Resolve-QA request, multipass/refinement plan, metadata/single-chapter/Direct Text
  flags, stop settings: read-only facts for the caller; the worker reads the owner like the
  closure did, which captured no locals).
- Lazily loaded globals `translation_stop_flag` / `glossary_main` / `glossary_stop_flag` are read
  through `_backend_entry` at the same point (desktop override: translator_gui globals).
- Signals are `_ui_request(kind, ...)`: `trigger_qa_scan`, `thread_complete`,
  `input_files_updated`, and `direct_text_glossary_approval` (desktop override emits
  `<kind>_signal`; the GUI-free default turns the approval into a blocking `host.ask(kind,
  path=...)` and completes the `{event, accepted}` handshake with the answer: no host or a host
  error = declined). The "Please select file(s) to translate." box is `_ui_message('critical',
  ...)` (desktop override: `QMessageBox.critical(self, ...)`; default: log + `message` event).
- `TranslatorGUI._x(self, ...)` class-qualified calls in the QA helpers name the mixin that
  defines `_x` now (`TranslationPipelineMixin` / `RunEnvMixin`): same functions on desktop.
- `_InputOutputDialog._IMAGE_ATTACHMENT_EXTENSIONS` is `translation_pipeline.IMAGE_ATTACHMENT_EXTENSIONS`
  (equal set, now one object shared with `run_translation_direct`).
- GUI-free defaults (`PipelineHooksMixin`, overridden by TranslatorGUI's own methods):
  `_lazy_load_modules` (imports the backend entries through `_backend_entry`),
  `_attach_gui_logging_handlers`, `_create_watchdog_snapshot`, `_start_autoscroll_delay`,
  `_update_manual_glossary_status` (no-ops), and U7 placeholders `_process_image_file`,
  `_process_rpgmaker_game`, `_run_generative_prompt_mode` (log "not available in this build
  yet", return False; they must go when image_job / rpgmaker_job move:
  tests/test_translation_pipeline.py::test_u7_placeholders_shadow_no_real_runner trips then).
- Mobile divergences (by design): the model check (`_require_model_selection`, a message box)
  stays in the desktop preflight, mobile relies on `run_translation_direct`'s "no model is
  selected" stop; the antigravity proxy retry reset stays in the desktop thread launch
  (`KNOWN_TRACE_DIVERGENCES['mixins']`); the Library raw-input registry is skipped (item 3; the
  mobile FileBridge imports into Library/Raw itself).
- HeadlessOwner: `_modules_loaded` / `_modules_loading` (False, like `__init__`),
  `selected_files = []`, `current_file_index = 0` and an `entry_epub` TextShim ("No file
  selected", like `create_file_section`).
- Desktop tests repointed: `_src_corpus` (translation_pipeline classes), test_subtitle_processor
  (`_get_app_dir` patched where `_auto_load_glossary_after_extraction` now reads it; the
  `resolve_glossary_input_sources` wiring is asserted on the corpus + the GlossaryManager alias),
  test_text_jobs (HeadlessOwner base order with the pipeline mixin first; contract sample),
  test_headless_owner (`auto_load_glossary_for_file` is no longer deferred).

### U3 step 2 parity harness changes

- freezer: a named shared mixin includes its in-module base classes (`_in_module_lineage`), so an
  oracle frozen at a commit with translation_pipeline knows the GlossaryPipelineMixin /
  PipelineHooksMixin methods and seeds the closure with the TranslatorGUI methods they call.
- registry: `moved_functions` gains `translation_pipeline` (MIXINS, SHARED_MODULES), the pipeline
  hook names (HOOK_NAMES) and the 37 moved names (+ the two set-up/worker methods) with
  `fuzz=False`: the main oracle freezes most of them without their closure; tier T runs them.
- owner contract: both scanners skip classes nested in a method (a nested progress manager's
  `self`), and `headless_owner.compute_owner_contract` counts a `try` body with a broad handler as
  guarded like tests/parity/owner_contract.py (`_reset_api_watchdog_progress` and
  `auto_glossary_shortcut_combo` left OWNER_CONTRACT for that reason). `entry_epub` is a
  `RUNTIME_ATTRS` entry for the booted desktop fake (the parity boot installs only the widgets
  the golden closure reads; adding it would change the frozen widget checkpoint).
- tier D: printed tracebacks lose CPython's "Did you mean: 'x'?" suggestion (`fuzz_moved.mask`):
  it depends on how many attributes the owner class has, not on the moved code.
- test_parity_tiers' GUI-mixin list skips classes of shared modules (a shared mixin's in-module
  bases are not GUI mixins); test_trace_parity's call walker parses methods whose multi-line
  strings defeat `dedent`, skips nested classes and `hasattr(self, 'x')`-probed calls.

## U3 Direct Text core, chain step 3 (direct_text_store, direct_text_stream)

Moved at BASE_SHA 1719fb59 on top of chain steps 1-2 (line numbers: `git show
1719fb59:src/translator_gui.py`). `_InputOutputDialog`'s chat persistence (history v2,
externalised bodies, output folders, attachment persist / glossary sync, workspaces + Migrate,
response edits, media references, the markup sanitiser: 77 members) is
`direct_text_store.ChatStoreMixin`; its log-stream model (classifier, request segments, payload
markers, phases, token counting, commits, `_finish_translation`: 36 members) is
`direct_text_stream.DirectTextStreamMixin`. The dialog is
`_InputOutputDialog(DirectTextStreamMixin, ChatStoreMixin, QDialog)` and keeps every Qt member.
tests/test_direct_text_core.py pins it: tier V (every moved text = BASE_SHA text + the edits
below, the dialog rewiring = exactly these edits), tier P (the REAL dialog built offscreen from
BASE_SHA and from the working tree replays 41 scenarios: history load/render/save on a copy of
src/direct_text_chats.json and a synthetic history, 18 recorded log streams, 9 full sends through
`_start_translation` -> stream -> `_finish_translation`, 6 Migrate variants, response edits,
pure helpers; state, rendered transcript HTML, saved JSON and file trees are equal) and tier H
(`ChatStore` / `DirectTextStream` reproduce the dialog's data observations in a Qt-free child).

### Desktop defaults / bugs found (not fixed)

1. **Eager getattr default reads a widget.** `_effective_run_glossary_path` /
   `_sync_attachment_glossary` call `getattr(self.translator, '_direct_text_force_no_glossary',
   self.force_no_glossary_radio.isChecked())`: the default is evaluated first, so an owner without
   the radio raises (swallowed) and skips the No-Glossary check. The GUI-free hosts carry a
   `force_no_glossary_radio` shim (`isChecked()` = the run's policy) so the code stays verbatim.
2. **Chat folders ignore OUTPUT_DIRECTORY between runs.** `_ensure_conversation_output_folder_for_session`
   reads `_saved_env['OUTPUT_DIRECTORY']` (the env saved at send time; `{}` between runs), then
   config `output_directory`, then cwd / the app dir. A legacy history with inline bodies saved
   outside a run therefore externalises into `<cwd>/Direct Text/...` even when OUTPUT_DIRECTORY
   is set. `ChatStore(output_root=...)` (default: OUTPUT_DIRECTORY at construction) plays the
   saved-env role on mobile, so chat folders land in the Files-visible output root.
3. **Dispatch records alone do not schedule a repaint.** `_drain_log_queue` classifies `[spine-order:N] ... Direct
   Text dispatch` as "ignore" and requests no repaint, so a new card shows up with the next
   repaint. `DirectTextStream.drain()` reports a change from the card signature instead of the
   repaint request alone.

### Behaviour deltas introduced by U3 step 3 (intentional, parity-neutral)

- `_InputOutputDialog.<name>` class references inside moved members are
  `ChatStoreMixin.<name>` (same constants: the dialog keeps `_IMAGE_ATTACHMENT_EXTENSIONS` =
  the shared set; the other class constants moved with their methods).
- Hooks (implemented by the dialog with the original Qt code; GUI-free defaults on the hosts):
  `_direct_text_notice(level, parent, title, text)` = `QMessageBox.information/warning` and
  `_confirm_attachment_merge(parent, target)` = the "Attachment folder already exists" question
  in `_migrate_conversation_attachment`; the moved code also calls the dialog's
  `_refresh_chat_list`, `_render_output`, `_set_status`, `_schedule_stream_render`,
  `_update_history_window_after_append`, `_restore_run_context` (`CHAT_STORE_HOOKS`,
  `STREAM_HOOKS`; the mixins define none of them).
- New mixin methods called at the original points: `_init_chat_sessions()` (the chat-loading
  block of `__init__`; `_switching_chat = False` now follows it), `_prepare_direct_text_input()`
  (the temp-input block of `_start_translation`, dedented, imports `uuid`/`datetime` itself;
  `_start_translation` lost its now-unused local imports), `_attachment_card_actions()` (the
  "Attachment actions" link decision of `_render_output`: same calls, same order, same HTML).
- `apply_direct_text_run_environment(owner, output_root, is_attachment)` is the env block of
  `_start_translation` (OUTPUT_DIRECTORY/OUTPUT_DIR, DIRECT_TEXT_*, spine ordering, then the
  owner's forced streaming and `_apply_direct_text_runtime_environment`); the mobile
  `direct_text` job kind uses it after `DirectTextRunOptions(...).apply_to(owner)`.
- `_build_attachment_action_card` / `_build_attachment_extraction_summary` gather the same inputs
  in the same order and call data builders (`attachment_action_data`,
  `attachment_extraction_report`) + the persisted card formats (`attachment_action_card`,
  `extraction_report_card`); differential-fuzzed against the BASE_SHA methods.
- `_atomic_text_write` lives in direct_text_store (translator_gui re-imports the name).
- Tooling: `mobile/tools/schema_extract.py` scans both modules as Direct Text UI sites (generated
  settings data unchanged); `tests/_src_corpus.py` includes both modules.

### GUI-free host semantics (mobile)

- `ChatStore` session-scoped methods take the session dict (the mobile adapter owns the list),
  make it current for the call and run one history save afterwards when the moved code asked
  for one (saves inside the scope are deferred). The mobile adapter's
  `save_chat_history(sessions, current_chat_id)` adopts its list.
- `ChatStore.finish_run(session, run, stream_or_segments)` runs the dialog's
  `_finish_translation` on a store-bound `DirectTextStream`. On mobile it runs after the job
  restored the process environment, so the live `MANUAL_GLOSSARY` the desktop still sees at
  finish time is gone: the caller passes `force_no_glossary` and `glossary_path` (the owner's
  `manual_glossary_path` after the run) in *run*, otherwise the attachment tree keeps the
  pipeline's own glossary files unsynced. The temp root is removed like `_restore_run_context`
  (`cleanup=True`, kept when `_preserve_temp_root`).
- A standalone `DirectTextStream` (live cards) starts with the run and its assistant message
  active (the dialog sets both at send time); `feed()` never waits for a drain (desktop listener
  semantics).

## U3 Integrate (wiring, packaging, mobile on the shared Direct Text code)

Desktop-visible edits (both defaults are the dialog's code; tests/test_direct_text_core.py
tier V lists them as documented edits and tier P/H re-ran on the edited module):

- `ChatStoreMixin._prepare_direct_text_input` reads two class knobs: `tempfile.mkdtemp(prefix=...,
  dir=self._DIRECT_TEXT_TEMP_PARENT)` (class default None = the OS temp dir, exactly `mkdtemp`'s
  default) and `attached_extension in self._EXTRA_PASS_THROUGH_EXTENSIONS` (class default empty).
  Only `prepare_direct_text_input(..., temp_parent=, extra_pass_through_extensions=)` sets them,
  on its own throwaway host (the mobile chat: resumable run roots in app data; ZIP / SDLXLIFF /
  MP4 attachments handed to the pipeline as-is).
- New `ChatStore.commit_request_phase(session, stream)`: the dialog's `_commit_active_request_phase`
  on a store-bound `DirectTextStream` that adopted the live stream's state; the live stream then
  starts the next phase empty. GUI-free host only (the dialog keeps calling its own method).

Mobile divergences (by design, recorded):

1. **Run root kept after a stopped / failed chat run.** The dialog deletes its temp root at the end
   of every run (`_restore_run_context`). The chat calls `ChatStore.finish_run(..., cleanup=...)`
   with `cleanup` only for a run that finished without a stop, so Resume / Retry continue from the
   run's `translation_progress.json` (roots live in `<data>/direct_text_runs`, not the OS temp
   dir). Compile from the chat uses the run root while it exists, else the folder `finish_run`
   persisted the run into.
2. **Stop: any second tap forces.** The desktop forces on a second click within 1.0 s
   (`register_stop_click`); after that window a click is another graceful stop. UI_SPEC §2.4 makes
   every further tap on Finishing a force stop (the long-press "Force stop now" too). The sequence
   itself is the desktop's (`JobBackend.request_stop`: graceful state dropped on the owner,
   `apply_force_stop_flags`, the owner's watchdog reset, `stop_control.request_stop` with the
   `translation_stop_flag` hook and the watchdog-reset cleanup, converter flag for compile jobs,
   HTTP loggers silenced on a graceful stop, the desktop's stop log lines).
3. **No `save_config` on Stop and no 500 ms `_reset_stop_flags_if_idle`.** Mobile never persists
   config from a stop; the next job's `stop_control.reset_for_new_run` clears the flags.
4. **Glossary jobs stop through the translation protocol.** `stop_glossary_extraction` was not
   extracted (`stop_control.request_glossary_stop` does not exist; the fix pass removed
   JobBackend's dead lookup of it); the extractor still stops through `owner.stop_requested` +
   `make_glossary_stop_callback`, but `extract_glossary_from_epub.set_stop_flag(True)` is not
   called.
5. **The glossary gate freezes cards at question time.** The dialog commits the glossary-phase
   cards when the approval question arrives (`_commit_active_request_phase` on the GUI thread);
   the chat does the same on the UI loop when the job's `direct_text_glossary_approval` question
   reaches it (the job thread is blocked on the answer meanwhile).
6. **Windows dev runs share the desktop token store.** `authgpt_auth` resolves `~` at import with
   `os.path.expanduser`, which ignores the HOME override on Windows, so a `flet run` of the app on
   Windows reads (and Sign out would delete) the desktop's `%USERPROFILE%\.glossarion` ChatGPT
   tokens. Devices are unaffected (bootstrap sets HOME before any backend import). Host tests pin
   the store to the sandbox (`tests_host/test_bootstrap.app_env`).

Still mirrored in the mobile UI (rules inline in the dialog's Qt handlers; tests_host/test_chat.py
compares each with the dialog source): `_rename_chat`'s title rule, `_on_glossary_override_toggled`'s
config writes, the "Provide Manual Glossary" extension sniffing, `_schedule_stream_render`'s
cadence, the dialog `__init__` settings reads, `_reset_history_window`, and the first-run
`_show_glossary_mode_welcome` cards / glossary-mode writes. Everything else the chat does
(attachment size and filter, glossary policy, card window bounds, auto-title, timestamps, token
counting, input preparation, the run environment, request cards, the glossary gate, finishing)
calls the shared code.

## U3 fix pass (review findings)

Desktop-visible edits (behaviour-neutral; tests/test_translation_pipeline.py, test_job_runner.py
and test_direct_text_core.py list each one as a documented edit against BASE_SHA 1719fb59):

- `stop_control` gained the non-widget tail of `stop_translation` and the preflight wait of
  `run_translation_thread`, moved verbatim: `stop_epub_converter()` (the converter flag; the
  desktop still decides whether the converter runs), `announce_stop(graceful, log)` (HTTP logger
  silencing on a graceful stop via `silence_http_loggers` / `HTTP_LOGGER_NAMES`, then the stop-mode
  log line) and `wait_for_stop_cleanup(thread, log, timeout=3.0)` (join the previous immediate
  stop's `translation-stop-cleanup` thread; False = do not start). `stop_translation` and
  `run_translation_thread` call them at the old positions (save_config stays between the
  converter flag and `announce_stop`); the mobile `JobBackend` calls the same functions instead of
  its former copies.
- `_translation_worker` returns its outcome: `return False` on the closure's five early returns
  (modules not loaded, stopped while preparing archive inputs, stopped during the glossary pass,
  Direct Text approval declined, require-complete glossary gate) and after its caught exception,
  `return translation_completed` at the end of its `try`. The desktop thread ignores the value.
- The set-up's Library raw-input registry block is the `_record_library_raw_inputs(files)` hook:
  `TranslatorGUI` keeps the verbatim `epub_library` import in its override (lite builds keep
  skipping it); the GUI-free default does nothing, so the shared set-up never imports
  `epub_library` -> `dpi_setup` -> PySide6 (it did on hosts with PySide6: `flet run`, host tests).
- `PipelineHooksMixin._ui_request('trigger_qa_scan')` logs "The post-translation QA scan is not
  available in this build yet; skipped" before emitting the event (`UNSHARED_UI_REQUESTS`; goes when
  the QA scanner is shared in U6). TranslatorGUI's own `_ui_request` (the Qt signal) is unchanged.
- `ChatStoreMixin._effective_run_glossary_path` reads `MANUAL_GLOSSARY` from the `_RUN_ENVIRONMENT`
  knob: class default None = the live `os.environ` (the dialog's code). Only the GUI-free
  `DirectTextStream.load_run` sets it, to the run's recorded values (`run["run_env"]`, empty when
  none were recorded), because the mobile chat finishes on `gl-chat-finish`, possibly while the
  next queued job has exported its own `MANUAL_GLOSSARY`.
- `DirectTextStream.segments(drain=True)`: the default still drains everything; repaints pass
  `drain=False` and read what the budgeted drains classified.
- Tier T: the desktop projection (legacy oracle vs the working-tree TranslatorGUI) is unchanged
  and passes; the mobile projection leaves the Library registry tree out of its file delta
  (`trace_harness.MOBILE_IGNORED_TREES` / `MOBILE_IGNORED_PARENT_DIRS`: the hook's GUI-free default
  writes nothing there) and drops file-delta kinds that end up empty.

Mobile behaviour fixed (no desktop counterpart):

- A run whose worker returns False is Failed ("The translation did not complete (see the log).")
  or, after a Stop, Stopped; before, every caught failure ended Done and the chat deleted the run
  root of a failed run.
- A Stop that lands before the set-up's "Reset stop flags" block (the job was starting) survives in
  the job's own latch; the translate / direct_text adapter does not start the worker then
  ("⏹️ Translation stopped before it started"). On desktop the set-up runs inside the Run click.
- Stop on the UI loop latches the job and changes its state at once and runs the protocol on a
  `gl-job-stop` thread, in request order; a protocol whose job already ended is dropped. A job
  holds the same lock around its cleanup wait, run reset, owner build and pending-stop re-issue.
- The next job waits for the previous immediate stop's cleanup thread (desktop: the next Run
  click); still alive after 3 s = the job fails with "The previous translation is still stopping;
  tap Retry in a moment." (desktop: "try Start again shortly").
- JobService feeds each host log message whole to the request stream (blank messages included,
  the desktop listener's view), drains the stream on a `gl-job-stream` thread with the desktop's
  12 ms / 1200-record budget while the job runs and completely when it ends; `request_segments`
  and `RunStream.segments()` never drain.
- `JobService.resume` refuses an Interrupted job already resolved "resumed"; the chat's Resume of
  an Interrupted turn goes through `JobService.resume` (it resolves the entry).
- After a relaunch a chat JobCard reads the turn's real end from JobService (Interrupted / Stopped
  / Failed / Done, chapter counts) instead of guessing it from the committed cards.
- Background transitions are handled in order (a STARTING -> FAILED pair cannot leave the Android
  foreground service running); a new ChatGPT sign-in closes a session a failed paste left on the
  callback port; Export offers the attachment workspace's top-level EPUB / PDF
  (`_preferred_attachment_compiled_documents`), both when both exist.
- UI_SPEC gaps closed: "Sign-in required for ChatGPT" (notification + snackbar) when a job prints
  `authgpt_auth`'s browser-login fallback lines (`jobs.SIGN_IN_MARKERS`); "Glossary ready"
  notifications and a strip tap on a pending question open the owning chat (the approval card);
  the Queue snackbar has Undo; "Stop current & send" confirms first; the Plan card shows the iOS
  background notice and the effective glossary mode ("Glossary: Balanced (auto)", resolved with
  `RunEnvMixin._current_auto_glossary_mode`).

## U4 chain step 1 (model_catalog_core, settings_rules; translator_gui rewired)

Rewired at BASE_SHA a7aa4a75 (the U3 fix commit; parity oracle, trace oracle and goldens
re-frozen there). Line numbers refer to `git show a7aa4a75:src/translator_gui.py`. Nothing
below was fixed; tier D (`moved_functions.REWIRED`, 26 desktop handlers), tests/test_settings_rules.py
and tests/test_model_catalog_core.py pin the behaviour.

The U2-moved handlers the plan lists for settings_rules (`_update_auto_compression_factor`,
`_compression_chunk_budget`, `_enforce_context_batching_mode` / `_refresh_context_batching_controls`,
`_on_context_mode_changed`, `_on_auto_glossary_shortcut_changed`, `update_target_language`,
`_on_disable_temperature_toggle`) already live in owner_state / run_env, pinned byte-for-byte to
the U2 base by tests/test_headless_owner.py::test_moved_methods_are_verbatim. They were left
untouched: settings_rules *runs* them on a GUI-free owner (`_RuleOwner`, ConfigStateMixin +
RunEnvMixin, `*_var`s seeded from the config through `settings_schema` var names) with a
private `os.environ`, so there is still one copy of each rule.

### Desktop defaults / bugs found (not fixed)

1. **Two key-pool scans disagree.** `_iter_enabled_key_pool_models` (12610) scans ten pools
   (incl. metadata / QA / inpainter / TTS) and skips disabled non-dict entries; the login scans
   `_has_authgpt_in_key_pools` / `_has_authgem_in_key_pools` / `_has_authgem_vertex_in_key_pools`
   / `_has_authcd_in_key_pools` / `_collect_auth_account_ids_from_pools` use only the six chat
   pools and ignore `enabled` on non-dict entries. So an `authgpt3/` key in the TTS pool shows
   no ChatGPT login, while `authgpt0/` there does (through `_authgpt_pool_route_requested`), and
   `authgrok*/` in any of the ten pools does.
2. **Case handling differs per provider.** The AuthGrok / pool-route / Vertex scans lower-case
   and strip the pool model; `_has_authgpt/authgem/authcd_in_key_pools` match the raw text
   (`AUTHGPT2/x` in a pool is not detected).
3. **Paid Google Translate text is case-sensitive.** on_model_change hides the Vertex location
   for `model.lower() == 'google-translate'`, but the credential status uses
   `model == 'google-translate'`, so `Google-Translate` gets "✓ Credentials: <file> (Project: …)" /
   "⚠ No Google Cloud credentials selected" instead of the Translate API texts.
4. **Auto compression factor boundary.** `_update_auto_compression_factor` uses `< 16379` for 1.5
   (the other bounds are 32769 and 65536), so 16379-16383 output tokens already get 2.0.
5. `_apply_chunk_size` falls back to `max_output_tokens` 65536 when the attribute is missing; the
   owner's start-up default is 128000 (never reached on desktop: the attribute is always set).
6. **Chunk Size "inf" raises.** `_on_chunk_size_edited` converts the field with `int(float(text))`
   and catches only `TypeError` / `ValueError`, so `inf`, `-inf` or `1e400` raise `OverflowError`
   out of the Qt slot (after the Other Settings widget lookups; nothing else changes). The moved
   `settings_rules.parse_chunk_size_text` / `set_chunk_size_text` keep the bug (verbatim); a mobile
   caller must catch `OverflowError` itself (none exists yet; the mobile Chunk Size field is a
   schema tile).

### Behaviour deltas introduced by U4 step 1 (intentional, parity-neutral)

- `on_model_change`: the Google Cloud decision is `settings_rules.google_credentials_route` (the
  Multi-Key Manager creds hint is read once; it was read twice with the same result), the
  credential JSON is read by `google_creds_ready_text` and closed before the status label is
  set (it was set inside the `with`), and the status / prompt texts come from settings_rules.
  The AuthGPT / AuthGrok / AuthCD / AuthGem decisions are `authgpt_login_needed` /
  `authgrok_login_needed` / `authcd_login_needed` / `authgem_login_needed`, which take the desktop
  helpers as lazy callables (same evaluation order); the unused `_vx_match` regex is gone. The
  OCAGY / Z.AI / Arena / Antigravity blocks are unchanged (tests exec them in isolation; those
  routes are excluded on mobile).
- `_model_needs_google_creds`, `_iter_enabled_key_pool_models` (a lazy `yield from`),
  `_has_google_creds_model_in_key_pools`, `_has_vertex_model_in_key_pools`,
  `_has_authgpt/authgrok/authgem/authgem_vertex/authcd_in_key_pools`,
  `_authgpt/authgrok_pool_route_requested`, `_authgem_vertex_control_model` and
  `_collect_auth_account_ids_from_pools` are wrappers over settings_rules (moved bodies). The
  wrappers read the live editor hint attributes before the model test (side-effect-free
  `getattr`s; the bodies read them after) and pass `getattr(self, 'config', None)` where the
  body's own try block used to catch a missing config.
- Chunk Size: `_on_chunk_size_edited` parses the field with `settings_rules.parse_chunk_size_text`
  after reading the text and the Other Settings widgets, the original order (U4 review fix: it
  parsed first, so an `OverflowError` for `inf` skipped the stale-widget clean-up of the lookups;
  the conversion can raise, see bug 6); tier D's Chunk Size preset now includes `inf`;
  `_apply_chunk_size` uses `factor_for_chunk_size` + `record_chunk_size`;
  `_remember/_hold_manual_chunk_size` use `remember_manual_chunk_size` / `held_manual_chunk_size`.
- Model catalog: `_restore_removed_model_choices` is `model_catalog_core.restore_removed_models`
  (tombstones, add-to-saved, save, rollback, log) plus the picker refresh; it reads its
  `save_config` / `append_log` attributes up front. `_ensure_polled_model_marker_state`,
  `_apply_polled_model_icons` and `_apply_provider_model_catalog_refresh` take every data step
  from model_catalog_core in the old order. `_save_model_order` reads the "Lock mouse wheel"
  checkbox before snapshotting the config (it was read after) and rolls back through
  `ConfigSnapshot` (same per-key present/absent semantics). `_collect_custom_prefix_routes_from_table`
  validates each row with `custom_prefix_route_from_row` (same messages; the Base URL cell is
  still read only after the endpoint type passed).
- The rewired methods import the shared modules locally (several desktop tests exec these
  methods from source in a bare namespace).

### GUI-free semantics for mobile (settings_rules / model_catalog_core)

- `route_controls(model, config)` reproduces on_model_change's login buttons, credential row,
  Vertex location and GCP project picker for a config (checked against the frozen desktop
  handler for 21 models x 5 pool configs x 5 live hints x 3 credential states); `needs_api_key`
  comes from `UnifiedClient._model_needs_api_key` (best effort: its local-endpoint test reads
  environment variables a mobile UI thread does not set).
- `EXCLUDED_ROUTE_PREFIXES` / `excluded_route_reason` (platform rule, mobile only) use the
  mobile app's reason texts; services/model_catalog.py and ui/sheets/model_sheet_min.py still
  keep their own copies of the prefix table (Integrate: point them at settings_rules).
- `enforce_context_batching` seeds the batching mode from `config['batching_mode']` (default
  'aggressive'): a legacy config holding only `conservative_batching` is migrated by the desktop
  start-up (`_init_config_state`), not by the adapter. Desktop-saved configs always carry
  `batching_mode`.
- `fan_out_target_language` returns the two environment exports of `update_target_language`
  instead of writing them; a job exports them from its own config snapshot.
- `model_catalog_core.apply_provider_refresh` composes the desktop refresh steps for a config
  (tombstone clearing on explicit polls, picker list, marker merge, counts, auto-poll line);
  tests/test_model_catalog_core.py compares it with the desktop handler on random results.

### U4 step 1 parity harness changes

- freezer: `TG_ENTRY_METHODS_U4` (the rewired handlers plus the `_expire_polled_model_markers` /
  `_save_model_manager_state` callers) and their GUI-only callees as recorders (login-status
  updaters, token-store snapshots, account-slot combos, model combo / completer / poll border,
  the mouse-wheel guard, the consolidated poll log, the queued full refresh). Oracle, trace
  oracle and goldens re-frozen at a7aa4a75.
- tier T: an oracle frozen after the U3 moves holds the pipeline in its frozen mixin copies, so
  `test_trace_oracle_closes_over_the_pipeline` accepts mixin-provided names,
  `trace_harness.mutant_legacy_class` resolves the method through the legacy owner's MRO, and the
  stop-latch mutation is anchored before `stop_control.request_stop(` when the frozen
  `stop_translation` no longer publishes the flags itself.
- tier D: `moved_functions.REWIRED` (`mixin == 'TranslatorGUI'`, `shared` modules) with
  `fuzz_moved.check_available` support; `SETUP_PRESETS` (`setup["@preset"]`: widgets for every
  base plus base variants: route models x pool configs x live hints, Model Manager dialogs /
  tombstone configs, Chunk Size field texts); U4 argument profiles (catalog results, model
  values, list widgets + dialogs, prefix tables, polled keys, chunk sizes); generator results
  are materialised; `fakes.FakeStatefulWidget` / `FakeListWidget` / `FakeModelManager` /
  `FakeTable` (state in `describe()`, mutations printed so argument objects are compared too,
  `parity_copy()` deep copies); `QtStub.__instancecheck__` returns False (`isinstance(owner,
  QObject)` raised TypeError before, cutting every catalog-refresh state short); two Google
  credential fixtures. test_parity_tiers: tier D for every REWIRED entry with a 25% clean-state
  floor, and three injection self-tests (a rare-branch change in settings_rules /
  model_catalog_core must be caught through the desktop handler).

## U4 chain step 2 (prompt_profiles, settings_rules dialog rules; other_settings / GlossaryManager_GUI rewired)

Rewired at U4_BASE_SHA 9f06f8b3 (src identical to a7aa4a75 outside src/mobile, so the step 1
oracle, trace oracle and goldens still apply). Line numbers refer to
`git show 9f06f8b3:src/other_settings.py` / `src/GlossaryManager_GUI.py`. Nothing below was
fixed; tests/parity/test_u4_dialog_parity.py (legacy source vs working tree, real Qt widgets,
500 random states per function), tests/test_prompt_profiles.py and tests/test_settings_rules.py
pin the behaviour.

Moved: other_settings `on_profile_select` / `save_profile` / `delete_profile` / `save_profiles` /
`import_profiles` / `export_profiles` (15992-16397) -> `prompt_profiles` (state functions taking
`self`; the message boxes, combo box, editor and radio updates stay in other_settings and run as
hooks at their original points); `_sync_thoughts_lock_state` (896) -> `settings_rules.thoughts_lock_state`;
`_set_output_mode` (12311) and the `_update_output_mode_sub_settings` closure (12825) ->
`settings_rules.output_mode_flags` / `output_mode_sub_settings`; the GlossaryManager
`update_auto_glossary_state` closure (5229-5426: display -> mode map, prompt / Targeted Extraction
enabling, the Append Glossary / Auto-Mapping / Fuzzy Auto-Mapping locks) ->
`settings_rules.glossary_mode_from_display` / `glossary_mode_extracts` /
`glossary_mode_targeted_extraction` / `glossary_mode_toggle_steps`. The assistant prefill dialog
(translator_gui `show_assistant_prompt_dialog`) and `_quick_new_profile` were left for translator_gui's
owner in this step; the U4 review fixes rewire them onto `prompt_profiles.prefill_*` / `new_profile`
(section "U4 review fixes" below).

### Desktop defaults / bugs found (not fixed)

1. **Editable profile combo keeps index 0 at start-up.** `_create_profile_section` (translator_gui
   16925) fills the editable `profile_menu` and calls `setCurrentText(self.profile_var)`, which on
   an editable QComboBox only sets the line-edit text; the current index stays 0. Choosing the
   first profile from the list then emits no `currentIndexChanged`, so `on_profile_select` does not
   run (the autosave target stays the start-up profile). `_quick_new_profile` and `delete_profile`
   also select with `setCurrentText`.
2. **Two protected-profile lists.** `delete_profile`'s fallback when `_get_protected_prompt_profiles`
   raises has 8 names; `_get_protected_prompt_profiles` / `always_include_profiles` have 13
   (Refinement, the RPG Maker / NanoBanana / SDLXLIFF profiles are missing from the fallback).
3. **Rejected saves keep the staged edit.** The prompt editor's autosave writes every keystroke
   into `prompt_profiles[active]` and `config['prompt_profiles']` in memory; a save refused for an
   empty or duplicate name leaves that staged text in memory (and in the next config save), while
   the profile file on disk keeps the last saved text.
4. **Stream thinking off unchecks thoughts.** `_sync_thoughts_lock_state(False)` always unchecks
   "Enable thoughts" (and exports ENABLE_THOUGHTS=0), even when thoughts were on before stream
   thinking locked them; the lock does not remember the previous value.
5. **Unknown glossary modes.** The Glossary Manager combo maps an unknown saved mode to Balanced
   (index 5) while the main-window shortcut maps it to Off (index 0); start-up runs the shortcut
   handler first, so the stored mode becomes `off` before the Glossary Manager opens. An unknown
   *display* text in the Glossary Manager pass unlocks Auto-Mapping (neither forced on nor off).
6. **No Glossary differs per surface.** The main-window shortcut leaves `append_glossary` untouched
   for No Glossary; the Glossary Manager lock pass forces Append Glossary and Auto-Mapping off. A
   config therefore depends on whether the Glossary Manager was opened after the switch.
7. `import_profiles` merges anything `dict.update` accepts (a JSON list of `[name, text]` pairs
   imports too) and does not validate names or values.

### Behaviour deltas introduced by U4 step 2 (intentional, parity-neutral)

- Profile handlers: the state steps run inside `prompt_profiles`; the wrappers pass zero-argument
  callables for everything the old bodies read later (the editor text after the name checks, the
  profile combo text after config.json was read, `self.prompt_profiles` / `self.config` inside the
  file write, the export list after the file was opened) so evaluation order, exceptions and file
  side effects are unchanged. Profile files are still opened through other_settings' namespace
  (`_open_profile_file`; tests patch `other_settings.open`). Message-box texts and titles come from
  the outcome objects.
- `_set_output_mode` assigns the six vars, six config keys and six env vars in a loop over
  `output_mode_flags` (same order and values); the sub-settings closure reads one
  `output_mode_sub_settings` dict.
- Glossary Manager lock pass: one loop over `glossary_mode_toggle_steps(mode)` replaces the five
  branches (same order, same hasattr guards, same widget calls); the fuzzy hint label is looked up
  when the fuzzy step runs (the lookup has no side effects); unlocked hint labels get
  `lambda _, _attr=...: getattr(self, _attr).toggle()` (late-bound like before).
- `settings_schema_data.py` regenerated: `append_glossary_auto_load` / `fuzzy_auto_mapping` lose
  the `glossary.minimal` UI site (the lock pass writes `self.config[step.key]`, the toggles live
  in the General tab); every other generated value is unchanged by the desktop moves.

### GUI-free semantics for mobile (prompt_profiles / settings_rules)

- `profile_state_from_config(config)` runs the desktop start-up on a scratch owner
  (`_init_default_prompt_profiles`, `_init_variables` with a private environment,
  `initialize_extraction_variables`): built-ins added in priority order, the active profile
  validated, the extraction method resolved; it never touches the config.
- Without hooks, `delete_or_reset_profile` does what the desktop does through its widgets: a
  reset refreshes the last-saved copy and selects the reset profile; a delete selects the first
  remaining profile (`select_profile`, so an extraction profile also switches
  `text_extraction_method`).
- `evaluate_locks` now aggregates the stream-thinking lock of `enable_thoughts` and the Glossary
  Manager mode locks (`append_glossary`, `append_glossary_auto_load`, `fuzzy_auto_mapping`,
  `fuzzy_auto_mapping_threshold`) for the mode the Glossary Manager shows
  (`glossary_manager_mode`: the start-up shortcut normalisation). `apply_change` applies the
  stream-thinking lock, the output-mode flags and the main-window glossary shortcut; the Glossary
  Manager pass itself is `apply_glossary_mode_locks` (for the glossary pages).
- settings_schema `locked_if` = `lock:<key>` for the seven lock-rule keys; `visible_if` names the
  thinking / output-mode / glossary visibility rules; `evaluate_rule` / `settings_rules.evaluate`
  answer them (lock reason or '', visibility bool).

### Schema generator (src/mobile/tools/schema_extract.py, GENERATOR_VERSION 2)

- Typed defaults and index-coded combos settle the type before the secret / path name heuristics:
  `use_multi_api_keys` bool, `multi_api_keys` list, `number_spacing_token_fix` int (the combo stores
  the item index as '0'/'1'/'2'; desktop reads `str(value)`, so an int round-trips).
- Choices come from the desktop combo construction (`addItems` of literal / local / module
  constant lists, `addItem(label, data)` incl. loops over literal pairs, the value -> index dict of
  `setCurrentIndex({...}.get(v))`), tied to the key through the settings_map widget source or the
  combo's own `setCurrentIndex` / `setCurrentText` / `findText` / `findData` argument, resolving a
  local name through its reaching assignments only (functions reuse `idx` for many combos).
  Labels differing from values become `(value, label)` pairs; editable combos get the
  `editable_choices` flag and stay free text. 30 keys gained choices (REMOVE_AI_ARTIFACTS,
  auto_glossary_mode's 8 modes, GPT / Gemini / Anthropic / DeepSeek effort lists, PDF render modes
  and alignments, OpenRouter providers (editable), Azure API versions, output mode, image / video
  output options, manga options, ...). `CHOICES_OVERRIDES` keeps only the two keys without a combo.
- Known generator limitation left as is: the label tie (BIND methods) still resolves local names
  through every assignment, so a few labels / groups come from a neighbouring control (e.g.
  `openrouter_preferred_provider` 'Mode:' / 'Chapter Extraction Settings').

## U4 chain step 3 (key_pool_service; multi_api_key_manager / unified_api_client rewired)

Moved verbatim into the new GUI-free `src/key_pool_service.py`; the dialog methods are thin
wrappers that keep their widgets, message boxes, threads and Qt slots:

- `MultiAPIKeyDialog._dedicated_pool_specs` / `_dedicated_pool_spec` -> `dedicated_pool_specs()` /
  `dedicated_pool_spec()`; `POOL_SPECS` adds Translation (`main`), Fallback and Glossary (titles,
  labels, config / toggle keys, in-memory set / clear methods, env names, and the group
  descriptions the dialog now reads from here).
- `_export_pool_order` / `_pool_title` / `_pool_toggle_key` / `_sanitize_imported_keys` /
  `_collect_pools_for_export` -> `export_pool_order` / `pool_title` / `pool_toggle_key` /
  `sanitize_imported_keys` / `collect_pools_for_export`; `_import_keys` dispatch ->
  `classify_key_import`; `_import_pool_aware` planning, confirmation summary and result message ->
  `plan_pool_aware_import` / `pool_import_summary` / `pool_import_result_message`;
  `_import_legacy_list` entry conversion and message -> `legacy_key_entries` /
  `legacy_import_result_message`; `_export_keys` document -> `build_export_payload`
  (`glossarion-key-pools` version 1), `count_pool_keys` / `count_nonempty_pools`.
- Add-key buttons (`_add_key`, `_add_fallback_key`, `_add_glossary_key`, `_dedicated_add_key`):
  `missing_model_error`, `new_main_key_entry` (the `APIKeyEntry`), `new_key_entry` (the dict),
  `added_key_extra_info`.
- Key tests (`_submit_single_test`, `_test_single_fallback_key`, `_test_single_glossary_key`,
  `_dedicated_test_single_key`): `build_test_request(entry, pool)` (client arguments, per-key client
  attributes in the original order, debug-log wording, probe, send arguments, timeout),
  `send_test_request` / `configure_test_client` / `test_response_passed` (the `run_api_test`
  bodies), `cancel_test_client` / `reset_api_watchdog` (the timeout handlers); `_run_tests`'
  429 check -> `is_rate_limit_error`. Module level `_model_needs_api_key` /
  `_api_key_test_timeout_seconds` delegate to `model_needs_api_key` /
  `api_key_test_timeout_seconds` (the module names stay patchable).
- `RefusalPatternsDialog`: `DEFAULT_REFUSAL_PATTERNS` / `default_refusal_patterns()` (also the
  default of `unified_api_client.UnifiedClient._get_refusal_patterns`, previously a second copy),
  `load_refusal_patterns` / `load_disable_refusal_checks` / `load_refusal_length_limit`,
  `parse_refusal_length_limit` (save), `merge_refusal_pattern_lines` ("Load Patterns").

`APIKeyEntry` / `APIKeyPool` stay in multi_api_key_manager (`GLOSSARION_HEADLESS_KEY_MANAGER=1`).
Parity: `tests/test_key_pool_service.py` runs the module at `9f06f8b3` (`git show`, executed as a
separate module) and the working tree on identical fake-widget harnesses for 500 random states per
moved function (import / export, add key for all 11 pools, key tests for all 11 pools including
timeouts, the refusal dialog actions, `_get_refusal_patterns`); a mutation self-test (15 one-line
mutants of key_pool_service) is caught by those tests.

### Desktop defaults / bugs found (not fixed)

- **The export forgets the Glossary pool.** `_export_pool_order` is main, fallback, then the
  dedicated pools; `glossary_keys` / `use_glossary_keys` are never exported, and a file that
  contains a `glossary` pool is reported on import as "ignored unknown pool(s): glossary". Desktop
  export / import is unchanged; `key_pool_service.export_pools` / `import_pools` (mobile) include
  it.
- **TTS and Image gen / edit keys are tested with the chat probe.** The dedicated-pool test sends
  "Say 'API test successful'" to every pool, which speech-only and image-only models reject, so
  those keys show Failed. The desktop is unchanged (`build_test_request` still returns the chat
  probe, with `testable=False`); `run_key_test` (mobile) reports "Not testable" for `tts` and
  `inpainter`.
- **Two client-setup variants.** Translation / Fallback / Glossary tests set the Azure endpoint and
  API version only with the individual-endpoint toggle on, and also set `google_creds_path`; the
  dedicated-pool tests set `current_key_azure_endpoint` / `current_key_azure_api_version` whenever
  the key has them, toggle or not, and never `google_creds_path`. Kept as the `standard` /
  `dedicated` variants of `build_test_request`.
- Key tests ignore the per-key request parameters, output token limit and temperature (fixed
  temperature 0.7, max_tokens 1000). Kept.
- The Translation / Fallback debug lines can raise (`os.path.basename` / slicing of a non-string
  hand-edited `google_credentials` / `azure_endpoint`), which the dialog reports as a failed
  test; kept (the lines are emitted by `configure_test_client` at the same points).
- `_run_tests` / `_run_inline_tests` / `_create_progress_dialog` (the old progress-dialog test path,
  incl. its Gemini OpenAI-endpoint branch) have no callers; left in place.
- Two identical copies of the default refusal list remain outside this phase:
  `scan_html_folder.DEFAULT_REFUSAL_PATTERNS` (`tests/test_key_pool_service.py` asserts it still
  matches) and the `refusal_patterns` list of `TransateKRtoEN` (~24263, the response-validity
  check). Follow-up: import them from key_pool_service. (`manga_translator` ~8566 and
  `ocr_manager` ~727 keep their own, different lists on purpose.)

### Behaviour deltas introduced by U4 step 3 (intentional, parity-neutral)

- The Fallback / Glossary / dedicated closures build the request from the key dict when the
  probe runs but keep the `api_key` / `model` they read when the test was queued (as before);
  the Translation pool reads the `APIKeyEntry` at run time (as before).
- `_import_legacy_list` converts every key (`APIKeyEntry.from_dict`) before adding them to the
  pool instead of interleaving conversion and `add_key`; `add_key` cannot fail, so the pool,
  counts and log lines are the same.
- Each key test also evaluates `api_key_test_timeout_seconds(model)` inside the probe thread
  (`build_test_request`'s `timeout`, unused by the dialog, which keeps its own value).

### GUI-free semantics for mobile (key_pool_service)

- `export_pools(config, pools=None)`: all eleven pools in the order main, fallback, glossary,
  then the dedicated pools sorted (the desktop order plus Glossary), keys deep-copied; the
  `title` of Glossary is "Glossary Keys" (`pool_title('glossary')` keeps the desktop answer,
  'glossary').
- `import_pools(payload, config=None, *, apply=False, dry_run=False, known=None)`: returns
  `{kind, legacy, items: [(pool, keys, enabled)], skipped, unknown, error[, applied]}`; known pools
  default to all eleven; a pool-aware file replaces the listed pools (desktop), a legacy list or a
  single key object is appended to the Translation pool after `APIKeyEntry` normalisation (the
  desktop pool does the same); `apply_import_plan(config, plan)` writes it.
- `validate_entry(entry, pool)`: strips key and model, "Please enter a model name"; Translation keys
  become `APIKeyEntry.to_dict()`; other pools keep their dict shape with the per-key limit,
  temperature, delay, request parameters and disabled contexts normalised as the runtime pool does.
- `run_key_test(entry, pool, *, timeout=None, client_cls=None, log=None)`: one blocking probe on a
  daemon thread: `{ok, status, message, last_test_result}` with status passed / failed / error /
  rate_limited / timeout / untestable. A timed-out probe is cancelled like the dialog cancels it but
  not waited for (the dialog's `with ThreadPoolExecutor` waits for it before returning).

## U4 auth splits (oauth_session; authgem / authcd / authgrok begin/complete)

`tests/test_mobile_auth_splits.py` runs the three modules at `a7aa4a75` (`git show`) side by side
with the split ones under the same fake endpoints: pages served, requests sent, tokens, printed
output and errors are identical for Gemini `run_oauth_flow`, Claude `run_automatic_oauth_login`,
Grok `run_device_oauth_flow` / `poll_device_code_tokens` and the Grok PKCE flow.

### Desktop defaults / bugs found (not fixed)

- **Gemini waits out the timeout after an error callback.** After `?error=access_denied` (or any
  error redirect) `authgem_auth.run_oauth_flow` keeps waiting for a code until its 300 s timeout
  and only then raises. Kept for parity; the mobile split (`begin_oauth` / `complete_from_redirect`)
  reports the error at once.

### Behaviour deltas introduced by the splits (intentional)

- **authcd paste path fixed (plan §3).** `complete_oauth_exchange` splits a pasted `code#state`,
  checks the state and sends it to the token endpoint; `run_oauth_flow` passes it, and
  `build_auth_url` builds Claude Code's manual URL (`code=true`, `CLAUDE_CODE_LOGIN_SCOPES`). The
  automatic localhost flow (`run_automatic_oauth_login`) is unchanged (parity-tested).
- `authgem_auth.run_oauth_flow` now closes its listening socket when it returns.
- The token folder of all four auth modules is `oauth_session.default_token_dir()`:
  `GLOSSARION_TOKEN_DIR` when set, else `~/.glossarion` as before. Only the mobile bootstrap sets
  it (`<data>/home/.glossarion`; with `AUTH*_TOKEN_FILE`, `OPERA_ARIA_TOKEN_FILE` and
  `AUTHARENA_PROXY_DATA_DIR`), so a Windows dev run (`flet run`, where `expanduser` ignores `HOME`)
  no longer shares or deletes the desktop's tokens (U4 Integrate also moved
  `authgpt_auth._DEFAULT_TOKEN_DIR`, which numbered ChatGPT slots use, onto it).
- Mobile only (`oauth_session.is_mobile()`): the Claude Code CLI login raises, the CLI status /
  version / credential-import readers return None (no process, file or Credential Manager read),
  and the Grok browser / device flows that open `webbrowser` raise "sign in from Accounts".

## U4 Integrate (wiring, packaging, mobile on the shared U4 cores)

Desktop-visible edit:

- `model_options.due_provider_catalog_for_model` / `refresh_provider_model_catalogs`: the
  `from autharena_proxy import list_accounts` lines are guarded. When the module cannot be
  imported (the mobile bundle excludes the autharena/ route) the first returns None (no
  auto-poll) and the refresh reports `autharena: "unavailable in this build"` instead of raising.
  The desktop always ships autharena_proxy, so its branch is unchanged. This replaces the mobile
  catalog service's `sys.modules` stand-in.

### Desktop defaults / bugs found (not fixed)

- **Reset Settings to Defaults writes the preserved API keys in plain text.** The Other Settings
  reset builds `keys_to_preserve` from `self.config` (decrypted in memory) and `json.dump`s it
  over config.json without `encrypt_config` and without a backup; the keys stay unencrypted
  until the next save. Mobile Danger zone backs up first and writes through MobileConfigStore
  (encrypted); its preserved key set is the desktop block's (moved to
  `config_store.reset_preserved_keys` by the U4 review fixes; both callers use it).
- `save_profiles` is a non-atomic read-modify-write `json.dump` of config.json
  (`prompt_profiles.write_profiles_to_config_file` keeps it verbatim for desktop parity; the
  mobile app writes profiles through MobileConfigStore instead).

### Mobile divergences (by design, recorded)

1. **Model Manager edits save at once.** Reorder, remove (with Undo), restore, add, reset and the
   Poll-providers merge go through `model_catalog_core.save_model_order` /
   `restore_removed_models` immediately, with the desktop tombstone rules; the desktop dialog keeps
   a draft until Save.
2. **Poll credentials.** The configured `api_key` goes only to the configured model's provider
   (desktop rule). A per-provider group refresh uses the first enabled pool key whose model maps
   to that provider; the desktop also reads `OPENAI_API_KEY`-style environment variables, which a
   phone does not have.
3. **Key tests.** `run_key_test` reports TTS and Image gen/edit keys as "Not testable" (see U4
   step 3). Each key is tested in its own run environment of a HeadlessOwner (one `JOB_LOCK`
   hold per key, so a job started during "Test all" waits for one probe at most), built from the
   config with every key pool switched off: with the Translation pool on, the run environment
   makes a new UnifiedClient rotate through the pool (and Fallback retry with fallback keys)
   instead of sending the key under test. When a running job holds the lock and its environment
   has `USE_MULTI_API_KEYS` / `USE_FALLBACK_KEYS` on, the test reports "busy" instead of running.
   (The desktop dialog tests in the GUI process environment, which has the pools on only after a
   translation run or a live toggle exported them.) Results are stored on the key tested
   (api_key + model), not on the list index. Endpoints › Test connection keeps the full run
   environment, pools included, like a run.
4. **Thinking tiles in the ModelSheet write the global keys.** "Apply to this chat only" covers the
   model, profile and target language (the U3 chat override contract); per-chat thinking control
   is Chat settings › Disable all thinking.
5. **Reset to defaults / restore a backup reload the config store in place** instead of
   restarting the app (a backup is made first).
6. **LAN Ollama / LM Studio** use custom-prefix routes (`ollama-lan/` → `http://<host>:11434/v1`,
   `lmstudio-lan/` → `http://<host>:1234/v1`) because unified_api_client hard-codes localhost for
   `ollama/` and `lmstudio/`; a desktop that imports the config routes them the same way. The
   per-model Ollama options only reach the desktop-only `ollamapull/` route.

## U4 review fixes (desktop handlers onto the shared cores; mobile sign-in and key-test fixes)

Rewired at U4_BASE_SHA 9f06f8b3 (oracle: `git show` of that commit, real Qt widgets offscreen);
tests/parity/test_u4_dialog_parity.py section 5 compares legacy and working tree for at least
`PARITY_U4_STATES` (500) random states each, with harness self-tests (an injected rare-branch
change in each shared function is caught):

- `TranslatorGUI.show_assistant_prompt_dialog`: the dialog owns a `prompt_profiles.PrefillState`,
  filled by `prefill_load` from the three config values the dialog still reads itself (same
  `self.config.get` calls and defaults; mobile uses `prefill_state_from_config`, which makes the
  same reads); its closures `stage_prompt` / `select_profile` /
  `persist_profiles` / `new_profile` / `save_profile` / `delete_profile` call `prefill_stage` /
  `prefill_select` / `prefill_config_updates` / `prefill_new` / `prefill_save` /
  `prefill_delete(..., confirm=...)`; the combo box, editor, token count, message boxes (texts and
  titles from the outcomes), the Yes/No box (now the `confirm` hook, asked after the same checks),
  save / rollback and log lines stay in the dialog. Compared per step: combo items / index / text,
  editor text, token label, config, `assistant_prompt`, logs, save calls, every message box.
- `TranslatorGUI._quick_new_profile` -> `prompt_profiles.new_profile` with the new
  `update_widgets` hook (combo add + select, editor clear) at the original point, between the
  config writes and the active-profile switch (their Qt signals read the state there), and
  `persist` (`save_profiles`); the log line and delete-button label stay.
- `TranslatorGUI._fetch_authgem_projects`: the worker body (Resource Manager list, phase-1
  publish, parallel billing check) is the new `authgem_auth.list_gcp_projects(token, on_listed=,
  http=)`; the worker keeps the guard, the store / token, the attribute writes and the queued
  `_authgem_projects_loaded` per phase. The mobile GCP project picker calls the same function
  (it used `detect_gcp_project`, which returns the cached / selected project, as "billed").
- other_settings `_reset_config_to_defaults`: the preservation block is
  `config_store.reset_preserved_keys` and the preserved-items text `config_store.RESET_PRESERVED_TEXT`
  (the informative text is "This will restart the application.\n\n" + it, the same string); the
  mobile Danger zone imports both (its copy is gone).

### Behaviour deltas (intentional, parity-neutral)

- Assistant dialog: `import prompt_profiles` when the dialog opens; `save_profile` reads the
  editor text before the name checks (a side-effect-free read; it was read after them);
  `PrefillState.text` also tracks the selected text (unused by the dialog).
- `_quick_new_profile` reads the combo items before `self.prompt_profiles` (both side-effect-free
  reads; an owner without `prompt_profiles` raises the same AttributeError).
- `_fetch_authgem_projects`: `import authgem_auth` in the worker after the token (already imported
  by `_get_authgem_store_for_current_model`); the phase lists are built by the shared function.
- `_reset_config_to_defaults` imports config_store before reading `self.config`.
- Settings schema generator: `settings_schema_data.py` is unchanged. The dialog's profile combo
  now selects `state.active_name`, which the generator cannot follow to the config read, so
  `schema_extract.WIDGET_KEY_PINS` ties that combo to `active_assistant_prompt_profile` (the
  label and tooltip texts are still read from the dialog source).

### Desktop defaults / bugs found (not fixed)

- `_fetch_authgem_projects` / `list_gcp_projects`: a listing whose projects all lack a
  `projectId` publishes an empty phase 1 and then raises `ValueError` (`ThreadPoolExecutor`
  with `max_workers=0`), which the desktop worker logs at debug level; the mobile picker shows
  "Could not list projects".

### Mobile fixes (no desktop change)

- **Key pools never loaded inside a mobile job.** `job_runner.isolated_key_pools` (U3; used by
  `job_process_state`, i.e. every JobService job, the Env preview and the key-test run
  environment) detached every `UnifiedClient` class member whose name ends in `_key_pool`,
  including the methods `setup_multi_key_pool`, `get_key_pool`, `initialize_key_pool`,
  `_get_active_key_pool` and the dedicated-pool setups. Inside the scope they were None, so
  `apply_key_pools_to_runtime` failed silently and a run with the Translation pool on used the
  single main key (glossary / dedicated pools likewise). Only data attributes are scoped now
  (`_pool_state_items`); tests/test_job_runner.py covers the methods and a real multi-key client.
  The desktop never uses `isolate_key_pools`.
- Accounts: a LoginSheet opened without a slot signs in slot #0 (it took the bridge's last slot);
  the blocked-Send "Sign in with ChatGPT" opens the current model's ChatGPT slot; autostart does
  not begin a new loopback sign-in while one of that slot is saved (the paste of its redirect
  comes first, "Start over" begins a new one); another provider's sign-in in progress disables
  Start with a reason; refused starts / empty pastes show their error.
- The Send gate, drawer chip, ModelSheet and chat sign-in refresh use the model's slot
  (`services.oauth.sign_in_satisfied`): `authgptN/` needs slot #N; the pool routes
  `authgpt0/`, `authgrok0/` and `authgem-vertex0/` accept any signed-in slot of the provider.
- Welcome step 2: a Claude / Gemini / Grok sign-in opens the ModelSheet on that provider's models
  (the default model stays `authgpt/gpt-6-luna` until one is chosen).
- Key-pool export writes the plain-text file only after an export option is chosen (a dismissed
  sheet left it in `<data>/Exports`).
