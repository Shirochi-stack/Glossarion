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
   ~~Mobile reaches `_run_generative_prompt_mode`'s U7 placeholder (logged, run fails cleanly).~~
   U7: the placeholder is gone; mobile runs the moved desktop runner (`image_job`, see
   "U7 image / RPG Maker runners" below).

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
  `_update_manual_glossary_status` (no-ops). (Until U7 also the placeholders
  `_process_image_file`, `_process_rpgmaker_game`, `_run_generative_prompt_mode`, which logged
  "not available in this build yet" and returned False; U7 removed them with `U7_PLACEHOLDERS`
  when the real runners moved into image_job / rpgmaker_job, which `TranslationPipelineMixin`
  now inherits; tests/test_translation_pipeline.py::test_u7_runners_replaced_the_placeholders.)
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

## U5 Progress Manager / Glossary Progress core (progress_core, progress_actions, glossary_progress_core)

Moved at BASE_SHA 20b446b0 (the U4 commit; parity oracle and goldens re-frozen there, the
Progress Manager oracle in `tests/parity/legacy_progress/20b446b06b10/`). Line numbers refer to
`git show 20b446b0:src/Retranslation_GUI.py` (RG) and `src/TransateKRtoEN.py` (TK). The moved
bodies are byte-for-byte copies (pinned by tests/test_progress_core.py,
tests/test_progress_actions.py and tests/test_glossary_progress_core.py, `*_are_verbatim`);
`RetranslationMixin(ProgressViewMixin)` and module re-exports keep every desktop name resolving
to the same code. The deliberate behaviour change is the write path below; the other deltas
listed follow from the split.

### Writes switched to lock + re-read + three-way merge + atomic replace

Each former whole-file `open(..., 'w') + json.dump(...)` of `translation_progress.json` now
goes through `progress_core.mutate_progress(path, fn)` (actions: under the per-path lock,
re-read the newest file, apply the action to a copy, merge only the action's change, atomic
replace; nothing is written when nothing changed) or `progress_core._commit_view_progress(path,
baseline, prog)` (the view: merge the change since the view's last read/write into the newest
file via the existing `_merge_and_write_retranslation_progress`; an unparsable newest file is
replaced by the snapshot, as before). Before, a translator save that landed after the dialog had
loaded the JSON was overwritten by the dialog's stale copy.

| Legacy write (RG) | Where | Now |
|---|---|---|
| 20870 (atomic, not merged) | build: subtitle-ZIP member seeding | `_commit_view_progress` |
| 21000 | build: save after `cleanup_missing_files` | `_commit_view_progress` |
| 21009 / 21016 / 21024 | build: PDF outline seeding, metadata row, TOC/header rows | `_commit_view_progress` |
| 21153 / 21212 | build: auto-discovery from the output folder (no-OPF fallback / OPF-aware) | `_commit_view_progress` |
| 21605 | build: entries auto-discovered while matching the spine | `_commit_view_progress` |
| 31405 (atomic, not merged) | refresh: `_write_progress_json_safely`, used for subtitle-ZIP seeding, cleanup, TTS reconcile, PDF outline, metadata and TOC/header rows | `_commit_view_progress` against `data['_progress_view_baseline']` (the snapshot the refresh read) |
| 31405 via recreate | refresh: progress JSON recreated by auto-discovery after it was deleted | merge-write against an empty baseline |
| 32059 | `_rematch_spine_chapters` | `_commit_view_progress` |
| 28025 | Restore In Progress Status | `progress_actions.restore_in_progress` (`mutate_progress`) |
| 28155 | Remove QA Failed Mark (menu + button) | `progress_actions.remove_qa_marks` |
| 28246 | Remove refinement status | `progress_actions.remove_refinement_status` |
| 28482 | Retranslate Selected in audio mode (TTS reset) | `progress_actions.reset_tts` |
| 30564 | Delete Audio File (TTS reset) | `progress_actions.delete_row_audio` |
| 30723 (atomic, not merged) | Resolve QA issue (LLM token) | `progress_actions.resolve_llm_token_qa` |
| 31099 | Insert Missing Image (QA marker clean-up) | `progress_actions.insert_missing_images` |

Remove Pending Mark (RG 414) and Retranslate Selected (RG 1626 / 29240) were already merge
writes and are unchanged. Proof: `test_progress_core.py::test_mutate_progress_keeps_a_concurrent_translator_save`,
`::test_build_writes_keep_a_concurrent_translator_save`,
`::test_refresh_writes_keep_a_concurrent_translator_save`,
`::test_mutate_progress_serialises_writers`, and
`test_progress_actions.py::test_action_keeps_a_concurrent_translator_save[*]` (8 actions: a
translator save between the dialog's load and the action survives; the frozen desktop closure
is run on the same scenario and is asserted to lose it, except Remove Pending Mark).

Glossary progress (`Glossary/<book>/<book>_glossary_progress.json`): Mark as Completed (RG 25131)
and Remove from progress (RG 26235) re-read and write the file under
`glossary_refinement._progress_lock` + `locked_progress_file` (the lock the extractor's save and
`update_refinement_progress` take) and replace it atomically (temp file + `os.replace`); before,
they wrote it in place without the lock. Proof:
`test_glossary_progress_core.py::test_writes_wait_for_the_extractor_lock_and_keep_its_save`.

Not switched (left as they were): the two writes that create an empty progress file in a
freshly created output folder (router cache reuse RG 20695, build RG 20832; no other writer can
exist yet) and the image-folder view's retranslate / delete writes (RG 35406 / 35486; that view
stays in Retranslation_GUI and Retranslate Selected's plan/apply is U7).
(Superseded in U7: both image-folder writes now go through `mutate_progress`; see "U7 Retranslate
Selected, ... / Writes switched to `mutate_progress` (image-folder view)".)

Retry on a locked file (U5 review fix): the former refresh writer `_write_progress_json_safely`
(RG 31394-31421) retried 20 times with backoff on `PermissionError` / `OSError` (sleeps
`min(0.5, 0.03·2^min(n,5))` + 0-0.03 s jitter: 8.4-9.0 s before it gave up). The merge path that
replaced it reads the newest file once before writing, and `_write_progress_snapshot_atomic`
(verbatim-pinned) retries only its `os.replace`: 20 attempts, sleeps `min(0.4, 0.03·2^min(n,4))`
(6.45 s in all), on winerror 5 / 32 and, on Windows, on any `PermissionError`; any other error it
raises at once. So a Windows sharing violation while the translator replaced the file aborted the
whole refresh tick ("Progress file locked during refresh"). `_commit_view_progress` now retries
the read-merge-write on `OSError` with the former writer's 20-attempt backoff, checks a 10 s
deadline before every retry, and does not run the write again when the error comes out of the
atomic writer after its own retries (`_atomic_write_retried`: the traceback passes through
`_write_progress_snapshot_atomic` and the error is one it retries). Worst cases (virtual clock,
real atomic writer, longest jitter): a persistently refused `os.replace` blocks 6.45 s (20 replace
calls, one atomic write; it took 12.9 s and 40 calls before this fix, 8.4-9.0 s at HEAD); a
persistently refused read blocks 7.9-8.5 s (20 attempts). Only a read lock that lifts just before
the deadline, followed at once by a persistent replace lock, can add one atomic write to the read
retries (at most about 16.5 s). A `PermissionError` from creating the temporary file on Windows
also counts as retried by the writer and is not retried again (the temporary name is unique per
process, thread and nanosecond, so that error does not come from a sharing violation).
`_write_progress_json_safely` itself is unused but stays inside the verbatim-pinned refresh block
(`test_split_refresh_and_stats_hold_the_frozen_blocks`, RG 31373-31421). Proof:
`test_progress_core.py::test_view_commit_retries_a_sharing_violation_on_read`,
`::test_view_commit_gives_up_after_the_retry_budget`,
`::test_view_commit_on_a_persistent_lock_blocks_no_longer_than_the_former_writer[replace|read]`
(runs the frozen `_write_progress_json_safely` on the same clock as the bound).

Error handling is unchanged: the former in-place writes of Restore In Progress, Remove QA Failed
Mark, Remove refinement status and Delete Audio File had no `try`, so a write failure still
raises out of the action; the audio-mode TTS reset's `try` (print the failure, refresh, report
the counts) is kept through `reset_tts(...)['error']`
(`test_progress_actions.py::test_reset_tts_reports_a_failed_progress_write`), and the LLM-token
resolution's through `resolve_llm_token_qa(...)['error']`.

### Behaviour deltas introduced by the move (intentional, parity-neutral on the fixtures)

- **Merge caveats.** The three-way merge (verbatim `_merge_retranslation_progress_changes`) keeps
  every value the dialog did not change, and the dialog's value wins on a field both changed.
  A key the translator deleted after the dialog's read is re-added when it sits in a dict on the
  path of the dialog's change (e.g. another chapter under `chapters` when an action changed one
  chapter); deletions elsewhere stay. The per-path lock is in-process: the translator's
  `ProgressManager.save` does not take it, so a translator save landing between the re-read and
  the `os.replace` (one merge, milliseconds) can still be lost; before, the window was the whole
  time since the dialog last read the file.
- **Actions apply to the newest file**, not the dialog's cached copy: an entry the translator
  removed or re-keyed since the last refresh is not touched (counts and messages reflect what
  was found on disk). The audio-mode TTS reset deletes nothing when the newest file cannot be
  read (before, it deleted the audio and wrote the cached snapshot over the file).
- **Display-time state is no longer persisted by actions.** The frozen actions wrote the dialog's
  whole in-memory snapshot, which also carried what the view reconciled for display (chunk-ledger
  schema normalisation, `tts_status` synced to the audio files on disk, `last_updated` stamps).
  The shared actions write only their own change to the newest file; the next refresh reconciles
  the same way. The action parity tests compare both sides after that reconciliation
  (`test_progress_actions.py::_normalize_chunks`).
- **Cleanup no longer builds a TransateKRtoEN ProgressManager.** The view called
  `ProgressManager(dir).cleanup_missing_files(output_dir)` on a temporary manager whose `prog`
  was replaced by the view's; it now calls `progress_core.cleanup_missing_files(prog, output_dir)`
  (TK 6632-6778, verbatim; the TK method delegates to it). The temporary manager's constructor
  side effects are gone: `_init_or_load` re-reading the file and, for a file that had become
  unparsable since the view read it, its repair save / `translation_progress_backup_<t>.json`
  copy; the subtitle-mirror restore and `ENABLE_PROGRESS_DEDUP` pass (whose result was discarded).
  The "TransateKRtoEN still loading" deferral is kept (`_progress_cleanup_ready`).
- **Insert Missing Image** imports `TransateKRtoEN.ContentProcessor` when the restore runs
  (`progress_actions._default_restore_fn`) instead of before reading the chapter files; a failing
  import is still reported as "Failed to restore images: ...", but only after the source /
  output checks (whose own errors now come first).
- The output-folder lookup imports `_get_app_dir` from `app_paths` (U1 home of the function
  `translator_gui` re-exports) instead of `translator_gui`.
- **Free-variable `re` (side effect of the move, not a fix).** In
  `_force_retranslation_epub_or_text`, `import re` at RG 21734 (filename fallback of the row
  builder) made `re` a local of the whole method, bound only when some row needed that fallback.
  The Glossary Progress closures `_gp_context_target_label` (RG 24929) and
  `_gp_write_completed_summary` (RG 25793) used it as a free variable, so with no such row they
  raised `NameError` (context-menu target label; the completed-glossary summary). The build moved
  to `ProgressViewMixin._build_progress_view_data`, so both now use the module's `re`.
- The status icon/label/colour dicts of `_format_progress_list_display_text` /
  `_apply_progress_list_item_visuals` are module constants (`PM_STATUS_ICONS` / `_LABELS` /
  `_COLORS`), and the statistics numbers / label texts of `_update_statistics_display` and the
  initial stats bar are `_progress_statistics` / `progress_stats_labels`; same values (fuzzed
  against the frozen methods).
- Opening the Progress Manager still registers the Library workspace through
  `epub_library.record_library_raw_input` on desktop; the GUI-free default
  (`ProgressOwner`) records through `library_core.record_library_raw_input`.

### Desktop bugs found (recorded, not fixed)

1. **Refresh auto-discovery regex** (RG 31494, now in `ProgressViewMixin._reload_progress_view_data`):
   `re.findall(r"(\\d+)", base)` matches a literal backslash + `d`, so every output file
   rediscovered after the progress JSON was deleted gets `special_<base>` / no chapter number
   (the build at RG 20936 uses `r"(\d+)"`).
2. **Chapter 0 treated as missing** in `_update_chapter_status_info` (RG 32899, 32920, 32967):
   `actual_num or chapter_num` falls through for chapter 0. The method has no caller in the
   current tree (dead code), kept verbatim.
3. **Image-folder view reads a pre-2.1 progress shape** (RG 35047), so its progress-based hash
   removal never matches current files.
4. **Audio-mode label mismatch.** The context menu says "🔁 Retranslate Selected" while the button
   says "Reset TTS Selected" for the same TTS reset (RG 29496 vs 30900).
5. **Two "failed" rules.** The Library card (`library_core._read_progress_summary`) counts a
   completed parent with chunk QA failures as failed; the Progress Manager shows it Completed.
   `progress_core.compute_book_summary` exposes both (`chapters_failed` and
   `chunk_qa_failed_parents`).

(An earlier draft listed a "free-variable `sys`" bug in the Glossary Progress "Open glossary"
button. It does not exist: `_open_glossary_file` does `import subprocess, shutil, sys` itself
(RG 26485), and symtable shows only `re` as a free variable of the method's closures -- the
separately documented `re` item above. Removed in the U5 review fix.)

### GUI-free semantics for mobile

- `ProgressOwner(config)` is a widget-free `ProgressViewMixin` owner: special-file rules come from
  `translation_pipeline.GlossaryPipelineMixin`, the output mode from `RunEnvMixin._get_output_mode`
  over config-seeded `*_var`s (`settings_rules._config_var`). `build_book_progress` runs the same
  build (output folder, seeding, cleanup, spine matching, rows) and `present_row` / `compute_stats`
  give the row text pieces and the statistics bar the desktop shows (compared with the dialog on
  the four fixtures). `compute_book_summary` is read-only and cached by the progress snapshot
  signature (the desktop prefetch's `_progress_snapshot_listing`).
- Cleanup on mobile always runs (`_progress_cleanup_ready` is True; desktop defers while
  TransateKRtoEN is importing).
- The actions return counts / message data instead of showing dialogs (`*_message` helpers give
  the desktop texts); `row_actions(info)` lists the context-menu actions the desktop offers for a
  row (checked against the menu).
- `glossary_progress_core.open_glossary_progress` builds the same panel model the desktop panel
  binds; `mark_glossary_completed` / `remove_glossary_progress` use the locked writes above.

### U5 parity harness additions

- `tests/parity/progress_legacy.py`: freezes RG / the TK cleanup method at BASE_SHA into
  `legacy_progress/20b446b06b10/`, runs frozen statement ranges as functions (`block_function`),
  builds fixture workspaces (EPUB with chunk ledger, metadata and TOC/header rows; PDF outline
  sections; subtitle ZIP bundle; plain text; image folder; glossary progress), opens the real
  Progress Manager offscreen (legacy and current) with recorded, auto-answered dialogs and an
  auto-picking `QMenu`, and snapshots rows / colours / statistics plus the progress JSON and
  output tree before and after each action. Library registration is pointed at a temporary
  `GLOSSARION_LIBRARY_DIR` so no test touches `~/Documents/Glossarion/Library`. The autouse
  isolation fixtures of test_progress_core / test_progress_actions / test_glossary_progress_core
  also clear `OUTPUT_DIRECTORY` / `OUTPUT_DIR` (both cores resolve workspaces there before the
  config's output directory; with either set, 34 tests wrote their workspaces outside tmp_path
  and failed).
- Existing tests retargeted to the moved code: `_src_corpus.progress_manager_source()` (RG + the
  three modules) for source greps in test_glossary_progress_status_precedence,
  test_glossary_minimal_pass(_row), test_glossary_refinement_status_display,
  test_progress_model_metadata, test_sdlxliff_support and test_unified_glossary_wiring.

## U5 Library (library_core, library_covers, reader_doc, live_stream; epub_library rewired)

Moved at BASE_SHA 20b446b0. EL = `git show 20b446b0:src/epub_library.py`. Module functions
moved byte-for-byte into `library_core` (scans, resolvers, search / sort / format, card data,
Book Details helpers, Library actions), `library_covers` (cover chain) and `reader_doc`
(reader document, caches, TOC, overlay signature, themes); `epub_library` re-imports every
moved name, so desktop callers and monkeypatches of the module attribute keep resolving.
Qt classes now inherit GUI-free mixins listed first (plan §2): `_DualScannerThread`
(`DualScanMixin`), `_LibraryDeleteThread` (`LibraryDeleteMixin`), `_CoverLoader`
(`CoverLoaderMixin`), `_RawScanWorker` (`RawScanMixin`), `_ScanForRawDialog`
(`ScanForRawMixin`), `EpubLibraryDialog` (`LibraryShelfMixin`), `_BookDetailsLoader`
(`BookDetailsLoaderMixin`), `BookDetailsDialog` (`BookDetailsMixin`), the six reader threads
(`EpubCacheLoaderMixin`, `OverlayMergeMixin`, `ReaderImagePreloadMixin`,
`WorkspaceReaderLoaderMixin`, `EpubSearchMixin`, `EpubLoaderMixin`) and `EpubReaderDialog`
(`ReaderDocMixin`, `LiveStreamMixin`). Pinned by tests/test_library_core.py and
tests/test_reader_doc.py (`*_are_verbatim*`, `test_qt_reader_classes_changed_only_the_documented_methods`)
plus differential fuzz, file-system fixtures and offscreen dialog smoke against EL.

### Phase-1 splits inside desktop methods (behaviour unchanged, each pinned)

These desktop method bodies were rewritten into calls of helpers EXTRACTED from them into the
mixins (the helpers did not exist at 20b446b0; the epub_library placeholder comments say
"extracted from <method>"). The complete set of changed Qt-class methods is pinned:
`test_library_core.py::test_qt_library_classes_changed_only_the_documented_methods`
(`LIBRARY_CLASSES_CHANGED`) and `test_reader_doc.py::test_qt_reader_classes_changed_only_the_documented_methods`;
behaviour by the differential fuzz / file-system fixtures against EL.

- `_DualScannerThread.run` = `run` + `_merge_scan_rows` (statement-for-statement).
- `EpubReaderDialog._render_current` LAYOUT_ALL loop -> `_all_chapters_html`;
  `_drain_live_queue` loop -> `_drain_live_lines`; `_finish_live_translation` completion
  check / cleanup -> `_live_outcome` + `live_stream.live_outcome_text`; `_finalize_post_load`
  numbering -> `reader_doc._chapter_display_numbers`; `_open_google_translate` /
  `_open_web_define` URL building -> `_google_translate_url` / `_define_url` (AST and
  behaviour checks against EL).
- `_BookCard.__init__`: card progress / badge / size -> `_card_progress_view`,
  `_card_type_badge` (+ `_CARD_TYPE_BADGES`), `_card_size_text`.
- `_ScanForRawDialog`: `__init__` -> `_init_scan_state`; `_populate_tree` -> `_scan_status_text`;
  `_apply_matches` -> `_write_raw_pairings`.
- `EpubLibraryDialog`: `_library_page_bounds` / `_update_library_pagination_controls` ->
  `_page_bounds` / `_page_label`; `_update_organize_counts` -> `_organize_counts`,
  `_missing_raw_count`; `_import_paths_into_library` -> `_run_import`, `_import_toast_text`,
  `_import_summary`; `_ensure_output_override_matches` -> `_output_override_mismatch`,
  `_output_override_prompt_text`, `_apply_output_override_config`; `_organize_into_library` ->
  `_plan_organize`, `_organize_preview_lines`, `_organize_collisions`, `_execute_organize`,
  `_organize_summary` (and its nested `_unique_dest`, lifted to `library_core._unique_dest`);
  `_undo_organize_prompt` -> `_plan_undo`, `_undo_prompt_text`, `_undo_collisions`,
  `_execute_undo`, `_undo_summary`; `_on_auto_scan_done` -> `_scan_diff`;
  `_clear_saved_raw_link` -> `_plan_clear_raw_link`, `_clear_raw_link_prompt_text`,
  `_execute_clear_raw_link`; `_delete_books_prompt` -> `_plan_delete`, `_unregister_cards`,
  `_all_targets_not_started`; `_on_delete_finished` -> `_delete_result_summary`;
  `_confirm_delete_simple` -> `_delete_simple_prompt_text`.
- `_BookMetadataEditDialog.changed_values` -> `_metadata_changed_values`.
- `BookDetailsDialog`: `_update_progress_strip` -> `_progress_strip_text`;
  `_on_edit_metadata_clicked` -> `_save_metadata_edits` (raising `_MetadataEditError` with the
  dialog texts); `_update_toc_toggle_label` -> `_toc_toggle_state`; `_chapter_page_bounds` /
  `_update_chapter_pagination_controls` -> `_page_bounds` / `_page_label`; `_open_reader` ->
  `_plan_open_reader`. The wait cursor keeps its place: `_plan_open_reader(..., busy=)` calls the
  hook exactly where the pre-split method set `QApplication.setOverrideCursor(Qt.WaitCursor)` +
  `processEvents()` (entering the PDF-workspace branch / the EPUB branch, before the translated
  overlay is built; never on the system-viewer path). The first split set the cursor only after
  the plan, so a large in-progress book built its overlay with no busy cursor (U5 review fix;
  pinned by the cursor / overlay / reader call-order trace in
  `test_book_details_loader_and_reader_plan_match_legacy`).
- Renamed class references inside moved bodies: `_ScanForRawDialog.MATCH_EXACT` ->
  `ScanForRawMixin.MATCH_EXACT`; `EpubLibraryDialog._raw_is_in_library_raw` /
  `._library_raw_match_for_book` -> `LibraryShelfMixin.*` (same objects through inheritance).

### Qt replacements (pure computations only)

1. `QUrl(src).scheme().lower()` (EL 18226, 24968, 25048) -> `reader_doc._url_scheme(src)`,
   QUrl's own rule (text before the first `:` that precedes `?` / `#`, RFC 3986 scheme
   chars, no trimming, a bad authority keeps the scheme). Identical on a 40k-string corpus
   (`test_url_scheme_agrees_with_qurl`); `urllib.parse.urlsplit` would have differed on
   leading whitespace and malformed IPv6 hosts.
2. `QUrl.fromLocalFile(path).toString()` in `_process_html` (EL 24894, 24916) ->
   `self._reader_file_url(path)`. `EpubReaderDialog` overrides the hook with the Qt original;
   `ReaderDocument` uses `image_url_for` (mobile: the in-app server) or `Path.as_uri()`.
3. `_reader_image_is_sizeable` (EL 858): `QImageReader(buffer).size()` ->
   `library_covers._probe_image_size` (PNG / GIF / JPEG / BMP / WebP headers, SVG geometry,
   Pillow for the other formats Qt sniffs). Same sizes as QImageReader for complete files.
   Differences (only reachable for images <= 5120 bytes, the byte rule decides above that):
   an SVG with neither width/height nor viewBox has no size here (Qt measures the drawing's
   bounding box) so it is not full-page; truncated headers that QImageReader gives up on still
   report a size; TGA / PCX / multi-size ICO headers are not read (QImageReader misreports
   Pillow's TGA and reports the first ICO entry).
4. `_download_remote_cover_image` (EL 663): `QImage.fromData(data).isNull()` ->
   `library_covers._image_bytes_decodable(data)`: SVG root parse; Pillow decode limited to the
   formats Qt sniffs by content, with Qt's truncation rules (partial JPEG / GIF / BMP / XBM
   accepted once pixel data follows the header, PNG needs `IEND`); image signature without
   Pillow. Same answer as Qt on PNG / JPEG / GIF / BMP / WebP / TIFF / PPM / XBM / PCX / TGA
   samples and truncations (`test_image_probes_agree_with_qt`). Known gaps: ICO files cut
   inside the icon directory, a JPEG cut inside its SOS header (3 bytes).
5. `_animated_image_reader` and every pixmap / widget path stay in `epub_library` (Qt only).

### Seams (desktop never triggers them)

- `library_core._default_output_root` gained a 3-line prefix: when a `LibraryEnv` with output
  roots is installed (`install_library_env`, mobile start-up) its first root is the default
  output root; otherwise the EL body runs unchanged.
- `_cover_cache_dir` / `_epub_cache_dir` honour `set_cover_cache_dir` / `set_epub_cache_dir`
  (`_COVER_CACHE_DIR_OVERRIDE or ...`), set only by `install_library_env(cache_dir=...)`.
- `GLOSSARION_LIBRARY_DIR` (U3 seam of `get_library_dir`) is set by `install_library_env`.

### U1 gap fixed: output_naming honours GLOSSARION_LIBRARY_DIR

`output_naming._library_origins_raw_sources_for_stem` and `_library_raw_inputs_for_stem`
(U1, from other_settings) built `~/Documents/Glossarion/Library` by hand, so with
`GLOSSARION_LIBRARY_DIR` set (mobile, tests) output-folder naming read the real desktop
Library registries. Both now call `output_naming._library_dir()` ->
`library_core.library_root_path()` (lazy import): the same path on desktop when the variable
is unset (`test_output_naming_functions_are_verbatim_and_reexported` pins the one-token
change).

### U5 Library card counts (decided semantics)

`library_core.book_summary(progress_file, config)` (mobile cards) returns exactly what the
desktop card reads, `_read_progress_summary(progress_file, exclude_special=not
translate_special_files)`: sidecar entries, metadata / TOC / header rows
(`is_metadata_progress_entry`, `metadata_progress_key`), translation artifacts and gallery
pages never count; with special files excluded, configured special files drop from total and
tallies; a multi-chunk parent takes `effective_parent_status` and counts as **failed** when any
chunk failed QA even if the parent says completed; a `completed` row whose `output_file` is
missing counts as **in progress** (phantom completion); `pending` counts with in progress;
other statuses (e.g. `skipped`) count only in the total. `progress_core.compute_book_summary`
(the Progress Manager's rules, landed concurrently) reports QA-failed chunk parents separately
(`chunk_qa_failed_parents`); the integrator may route mobile cards through it only if it
reproduces these numbers. The desktop card keeps `_read_progress_summary`.

**Integration decision (U5 Integrate): cards stay on these numbers, desktop and mobile.**
`compute_book_summary` does not reproduce them. Measured read-only on the 217 real output
folders under `src/` (143 with a resolvable raw source; script
`scratchpad/u5_integ/card_compare.py`): the two agree on 1 book and differ on 142. The Progress
Manager total is the card total +1 on 78 books and +3 on 56 (the `__metadata__` row and the
`__translation_artifact__:toc` / header rows the PM lists, which the card excludes by design),
its completed count is higher on 62 (output files it auto-discovers and tracks count as
completed; the card counts only JSON-completed rows), and 8 books the desktop card shows as
"✨ Ready to compile" would have dropped back to "⏳ In progress" on mobile. So the mobile
Library card, the Book page Overview strip ("⏳ Translation in progress — d/t chapters") and
the shelf placement all come from `library_core.scan_library` rows (`_read_progress_summary`
plus the spine count, `card_progress_view`), exactly like the desktop card and Book Details;
the Book page's Chapters tab shows the Progress Manager's own statistics (`compute_stats` /
`BookProgress`), exactly like the desktop Progress Manager. The two numbers differ on mobile
where they differ on desktop. `compute_book_summary` remains the plain API for a
Progress-Manager-rules summary (cached, read-only). The `TODO(U5 integrator)` marker in
`library_core.book_summary` is replaced by this decision.

### Plain (mobile) API notes

- `load_book_details(phase="preview")` returns the preview payload as emitted (captured at
  emit, `on_preview` gets a deep copy). On desktop the queued `preview_ready` hands the same
  dict to the dialog, so for a non-EPUB source the dialog may see phase-2 fields the loader
  added meanwhile (harmless race; the full payload follows). Recorded only.
- `reader_doc.chapter_display_numbers(filenames)` lower-cases basenames first, as
  `_finalize_post_load` does before numbering.
- Mobile reader shell (`wrap_reader_html(..., mobile=True)` / `ReaderDocument.wrap(mobile=True)`):
  adds the viewport meta (`viewport-fit=cover`), safe-area padding, `100dvh` and `-webkit-`
  column-break fallbacks and a paging bridge; removing the three inserted pieces gives the
  desktop page byte-for-byte. Events: `console.log("GLRDR:" + json)` and a same-origin
  `fetch(POST /__ev)` with the same JSON (`{type, seq, chapter, ...}`: `ready`, `page`
  {page, count, reason}, `edge` {edge}, `tap` {zone}, `link` {href}, `selection` {text},
  `scroll` {fraction}, `scale` {scale}); `seq` de-duplicates the two transports and
  `GLRDR.setTransport('console'|'fetch'|'both')` narrows them. Checked in QtWebEngine
  offscreen (`test_mobile_bridge_pages_in_webengine`).

### Desktop bugs found (recorded, not fixed)

1. **Non-atomic progress writes** in `_mark_chapter_pending_for_retranslation` (EL 1159) and
   `_cleanup_incomplete_chapter_output` (EL 1264): `open(progress_file, "w") + json.dump`
   without lock, re-read or atomic replace, so a translator save between their read and write
   is lost and a crash mid-write truncates `translation_progress.json`. Candidates for
   `progress_core.mutate_progress` (U5 Progress) in a separate change.
2. **Book Details synthetic spine sort** (`_BookDetailsLoader.run`, EL 13916 `_sort_key`):
   `info.get(...)` runs before the `isinstance(info, dict)` check, so a non-dict chapter entry
   raises AttributeError (not caught by `except (TypeError, ValueError)`) and the details load
   fails; `actual_num` 0 falls through to `chapter_num` (`or`).
3. **Special-file cache signature** (`_special_file_settings_signature`, EL 310):
   `TRANSLATE_ALL_NUMBERED_HTML == '1' or config.get(..., True)` ignores
   `TRANSLATE_ALL_NUMBERED_HTML=0`, which `_resolve_translate_all_numbered` honours, so the
   reader / spine caches are not invalidated when only that variable flips to 0.
4. **Security: the desktop reader runs book scripts in a `file://` page** (`EpubReaderDialog`;
   already at HEAD, unchanged by U5, found in the U5 review). Chapter `<script>` elements, `on*`
   attributes and `javascript:` URLs survive `reader_doc._process_html` / `_wrap_html` (the only
   script handling in reader_doc is the search-text helper). `_set_html` (EL 14821-14837) writes
   the page to `_reader_<id>.html` in the cache and loads it with `QUrl.fromLocalFile`; the view
   keeps Qt's defaults `JavascriptEnabled` and `LocalContentCanAccessFileUrls`, and
   `_configure_epub_reader_web_settings` (EL 272-283) turns `LocalContentCanAccessRemoteUrls` on.
   A crafted EPUB can therefore read any local file the user can read (an OAuth token JSON, for
   example) and send it to a remote host, whatever `_reader_image_resource` (item 5) does.
   Verified in the review on HEAD and on the working tree: the real dialog, offscreen, on a
   one-chapter EPUB whose chapter `fetch()`es a scratch token file and beacons it with
   `new Image().src`; a 127.0.0.1 server received the token. This is a security bug, not a
   low-risk deferral. The hardening is a separate, labelled desktop change with its own tests:
   strip book `<script>`, `on*` handlers and `javascript:` URLs by extracting the mobile
   `sanitize_book_html` (mobile/app/glossarion_mobile/ui/reader/document.py) into shared code
   (reader_doc) for both readers instead of copying it; optionally add a per-page nonce CSP, and
   check whether book images still load with `LocalContentCanAccessFileUrls` off. Setting
   `JavascriptEnabled` to False is not an option: the reader's own pagination script
   (`_wrap_html`, reader_doc.py 2182) needs JavaScript. Mobile is not exposed: book HTML is
   sanitised and the page CSP allows only the page's own nonce scripts ("U5 review: mobile
   fixes").
5. **Reader image path traversal** (`reader_doc._reader_image_resource`, a verbatim move of EL
   `_reader_image_resource`; found in the U5 review): a chapter `<img src>` is resolved with
   `os.path.join(epub_dir, src)` (+ `normpath`), so `../..` and absolute paths reach any readable
   file, which `_process_html` then copies into the reader image cache next to the reader page.
   With item 4 a book script does not need this to read local files; on its own it puts a copy of
   any readable file in the cache. Hardening the resolver is a desktop-shared change and belongs
   in the same separate, labelled change as item 4, with its own tests. Mobile is protected
   without it: the ReaderServer serves only bytes that are an image by content and the page runs
   no book script.

### U5 parity harness additions

- tests/test_library_core.py: verbatim / re-export / import-hygiene checks, differential fuzz
  (`PARITY_U5_STATES`, default 500), file-system fixtures through the legacy and new dialog
  methods (`PARITY_U5_FS_STATES`, default 40: scans + merge, Organize / Undo per conflict
  policy, Delete, Clear raw link, Import, output override, Scan for Raw, Book Details loader,
  reader-open plan, metadata save, single-chapter helpers), offscreen smoke of
  `EpubLibraryDialog` / `BookDetailsDialog`, the plain API, image probes vs Qt. Comparisons that
  depend on the row order take the scan tabs in `scan_order` (mtime newest first, then path and
  output folder): the scans append output-folder rows in thread-completion order and then
  stable-sort by mtime, so two folders with the same mtime come out in either order, in legacy
  and new alike (EL 2848 / 2857, preserved; it made the metadata-save comparison flaky).
- tests/test_reader_doc.py: reader / live verbatim and Phase-1 extraction checks, `_url_scheme`
  vs QUrl, reader threads vs the plain API, `_process_html` / `_get_embedded_css` /
  `_wrap_html` through EL, the new desktop class and `ReaderDocument`
  (`PARITY_U5_READER_STATES`, default 40), live stream classify / drain / wrap / output folder /
  finish vs EL, the mobile shell and its WebEngine bridge, bilingual / native blocks,
  offscreen `EpubReaderDialog` smoke (every layout, legacy vs new).
- Retargeted monkeypatches (moved names live in the shared modules now):
  tests/test_epub_library_layout.py (reader_doc / library_covers / library_core targets;
  `test_large_raw_image_classification_does_not_decode_bitmap` now records calls of
  `reader_doc._probe_image_size`, the probe `_reader_image_is_sizeable` reaches since it moved,
  instead of patching the no-longer-used `epub_library.QImageReader`),
  tests/test_progress_model_metadata.py (numbering source), tests/test_headless_owner.py
  (library_core verbatim check limited to moved names), tests/test_shared_core_p1.py
  (output_naming `_library_dir()`), tests/test_glossary_usage.py (local, untracked:
  `test_ui_hooks_are_wired` greps `_src_corpus.progress_manager_source()` for the moved Glossary
  Progress strings; it still fails only on the pre-existing `GlossaryHideUnused`).
- Library isolation: every Library test sets `GLOSSARION_LIBRARY_DIR` / `OUTPUT_DIRECTORY` (and
  the default output root / cover cache) to tmp. tests/test_epub_library_layout.py did not
  (pre-existing at HEAD: 8 tests built a real `EpubLibraryDialog`, which mkdirs
  `~/Documents/Glossarion/Library` and reads its origins / shelves); the U5 review fix adds the
  autouse `isolated_library` fixture there (verified with a sandboxed USERPROFILE: nothing is
  created under it).

## U5 Integrate (wiring, packaging, Library / Reader / Progress on the shared U5 cores)

Desktop source is unchanged by the integration except for `headless_owner.py` (mobile-only
class) and the `library_core.book_summary` comment; the desktop smoke below and every parity
tier show legacy == working tree. Oracles were re-frozen at HEAD 410354ac (its `src/` is
byte-identical to U4 20b446b0): `freeze_legacy.py --sha HEAD`, `capture_golden.py`,
`trace_harness.py --freeze --sha HEAD`.

### Mobile run set-up records its raw inputs (trace parity)

- `HeadlessOwner._record_library_raw_inputs(files)` calls
  `library_core.record_library_raw_inputs(files)`, the function desktop's
  `TranslatorGUI._record_library_raw_inputs` reaches through `epub_library`'s re-export. A mobile
  translation now lists its input in `<Library>/library_raw_inputs.txt` like a desktop run, so the
  Library resolves the book's raw source through the registry. The
  `PipelineHooksMixin` default stays a no-op (other owners, tests).
- `trace_harness.MOBILE_IGNORED_TREES` / `MOBILE_IGNORED_PARENT_DIRS` are empty: the mobile
  projection's file delta now includes the Library tree and equals the desktop one (tier T passes).
- The offline E2E (`translate_glossary_off`) asserts the registry write; with the hook stubbed out
  it fails with "the translate job did not record e2e-glossary-off.epub in library_raw_inputs.txt"
  (negative control run during integration).

### Library card counts

Decided in the Library section above ("Integration decision"): cards (desktop and mobile), the
Book page Overview strip and shelf placement keep the desktop card numbers; the Chapters tab shows
the Progress Manager's statistics. `progress_core.compute_book_summary` disagreed with the card on
142 of 143 real workspaces measured (metadata / TOC-artifact rows, auto-discovered outputs) and
would have taken "Ready to compile" away from 8 books.

### Tier D for the U4 rewired handlers is pinned to the U4 parent oracle (harness fix)

After re-freezing at a post-U4 commit, `test_rewired_desktop_method_matches_legacy[on_model_change]`
(clean fraction 19% < 25%) and the three `test_tier_d_catches_a_difference_injected_into_shared_rules`
self-tests failed: the oracle then holds the rewired handlers, which call the LIVE
`settings_rules` / `model_catalog_core` exactly like the working tree, so an injected change in
those modules reaches both sides and the rewired code's fuzz coverage is measured against itself.
Verified independent of U5 and of `src/config.json`: a clean worktree at HEAD failed the same four
with the current config, with the U4-era config and with no config.json. The REWIRED tests now use
a `rewired_session` over the oracle frozen at `U4_BASE_SHA` (9f06f8b3, the U4 parent; frozen on
demand without moving `LATEST.txt`), as `test_u4_dialog_parity.py` already does. All 430 parity
tests pass (one documented desktop-only trace race skipped).

### Mobile wiring (no desktop change)

- `app.py` installs `LibraryFeature` then `ReaderFeature` after the U4 pages (each in
  try/except); the Library folders are pinned with `library_core.install_library_env` on the io pool
  right after install. `SHIPPED_MILESTONES` gains U5 (Tools hub: Progress manager, Glossary
  progress), `job_kinds.KIND_MODULES` gains `single_chapter` (`JobKind.SINGLE_CHAPTER`).
- `base.build_screen_view` lets a screen with its own `build_view(route)` build its View (the
  Reader: edge to edge, its chapters end drawer and back handling from the first frame).
- The chat job card's **Read** (finished) and **Open reader** (running) are enabled: the turn's
  output workspace (the run's pipeline folder, else the chat folder's per-attachment workspace)
  goes to `ReaderFeature.open_book` as a Library-scanner-shaped row over the attachment (else the
  shared raw-source resolvers); with no workspace yet an EPUB attachment opens on its own.
- Android: `[tool.flet.android.manifest_application] networkSecurityConfig` points at the
  extension's `res/xml/glossarion_network_security_config.xml`, which allows cleartext HTTP to
  `127.0.0.1` / `localhost` only (the Reader's page server); every other host keeps the platform
  default. `ci/verify_apk.py` fails a build whose merged `<application>` lacks a
  `manifest_application` attribute.
- `schema_extract` scans library_covers / reader_doc / live_stream (Library sites) and
  progress_core / progress_actions / glossary_progress_core (Progress sites); the regenerated
  `settings_schema_data.py` is byte-identical to the committed one.

### Self-test additions (recorded divergence)

- `smoke` suite `library_reader` (also run by `tools/host_smoke.py`): a partly translated
  workspace of the self-test EPUB in a scratch Library / Output, opened through
  `LibraryService.scan_blocking`, `load_details_blocking`, `progress_model.load_progress_view`
  and the Reader (`plan_open` + `ReaderSession` + `DocumentBuilder`, mobile page shell). It pins
  the shared Library env (`install_library_env`, `GLOSSARION_LIBRARY_DIR`, `OUTPUT_DIRECTORY`) to
  the scratch folders for a few seconds and holds `JOB_LOCK` meanwhile (a job started from the
  app waits; it can never resolve its output folder into the scratch Output). A Library screen
  refresh that happens during those seconds lists the scratch book once; the next refresh is
  normal (the E2E sandbox has the same property for its whole run).
- `e2e` `translate_glossary_off` opens its translated + compiled workspace in the Library
  (Completed shelf, 12/12), Book page, Chapters tab (12 completed rows) and the Reader (dual
  mode: 12 chapters carry the fake marker, the Original side the Korean source).

### Desktop offscreen smoke (integration)

`scratchpad/u5_integ/desktop_smoke_u5.py`: the REAL `TranslatorGUI` (offscreen, sandboxed app
dir / home / temp, fixture book from `tests/parity/progress_legacy`) for the working tree and
`git archive HEAD src`: Library dialog shelves + counters, Book Details rows / strip / metadata
editor, the Reader opened from Book Details (pages of the first chapters in all four layouts, TOC,
numbering), the Progress Manager rows + statistics and the Glossary Progress rows + labels are
identical in both trees.

## U5 review: mobile fixes (no desktop change)

- **Navigation.** `AppShell.show` pushes a full-screen route (Reader, metadata editor) and a child
  whose static parents are already on the stack (Scan for raw from the Book page) on top of the
  current stack; re-opening a screen already on it (or another book) returns to that depth. Back
  from the Reader returns to the Book page instead of the chat home (UI_SPEC §1.6 rule 5).
  Screens with a selection mode override `Screen.handle_back` (`base.build_screen_view` then sets
  `can_pop=False` + `on_confirm_pop`): Android back leaves selection mode first (rule 2).
- **Reader.** ▶ Continue / "Continue · Ch N · P%" open at the saved position (`resume`); a plain
  open that offers "Resume" does not overwrite the saved position with the untouched start until
  the reader moves. Open arguments (chapter file, raw-only, resume) are one-shot. The overlay of
  an in-progress book is polled only while a job for the book runs (one refresh when it ends).
  The Reader and the job service hold one reference-counted wakelock (`services/wakelock.py`).
- **Reader security.** Chapter HTML is sanitised (no `<script>` / frames / objects / `<base>` /
  meta refresh / `on*` / `javascript:`), the book CSS cannot close its `<style>`, the page's own
  scripts carry a per-page CSP nonce (`script-src 'nonce-…'`, no `'unsafe-inline'`), images are
  served only when they are images by content (with a sandboxing CSP), events only as JSON
  `POST`s (a book `<img src="/__ev?d=…">` carried the cookie), and a book's external link opens
  only after a confirmation (http / https / mailto).
- **Library / Book page.** "Retranslate this chapter" follows desktop Book Details: busy check,
  confirmation for a completed chapter, progress entry reset to pending before the job is queued.
  The Glossary tab notices a progress file written later and stops re-rendering every tick after
  one is deleted (one signature representation). Pull-to-refresh runs one full refresh at a time.
  The "⚙ COMPILING…" ribbon clears when a compile ends without changing the workspace. Quiet
  scans that change nothing push nothing. A TXT card / PDF without a workspace goes to the share
  sheet (else the Book page) instead of the Reader's error. The TranslateSheet reuses the sources
  it resolved on the io pool. A keyword delete view already dismissed by back is not popped again,
  and a failed delete re-enables it. The Chapters list mounts at most 1,500 rows (a window
  selector beyond that; 150-row steps with "Rows per page: All").

### Second review round (mobile, no desktop change)

- **Navigation.** Any route with static parents opened in the app over other screens (Files from
  the Book page / Library card / Output tab, the Overview "Last job" link, a settings page) is
  pushed on top like the Reader (`AppShell.show(in_app=True)` for `source="app"`); the static
  chain stays for an empty stack, top-level destinations without parents, drawer navigation
  (`navigate_to(..., reset=True)` from `_drawer_navigate`, the drawer status chip and Help items)
  and links from outside the app (deep links, notifications: on top only when full-screen or when
  their static parents are already open, as before). Screens kept below the new top are
  not shown again (`did_show` runs for new entries and the top only), so the Book page under the
  Reader no longer reloads its details and progress.
- **Back.** The keyword delete view has its own `View.route` (`/library/delete-confirm`; it was
  `/library`, so Flet resolved Android back to the Library's View and disposed the Library and
  the Book page under the still-visible overlay); `AppShell.push_overlay` gives any overlay whose
  route collides with another View an `/overlay-<n>` suffix (the Model manager sub-screens reused
  the current route too). On tablets the root View cannot pop while the main area shows a screen:
  system back runs the screen's `handle_back` (selection mode), else pops the main-area stack, and
  answers `confirm_pop(False)`; with the chat in the main area it leaves the app.
- **Reader.** Leaving while the first chapter still loads no longer acquires the wakelock or
  offers "Resume" for the disposed screen; the overlay is polled only for this book's running
  job, not a queued one. `SharedWakelock.acquire` counts a holder only after the platform call
  succeeded (a failed enable left a stale "jobs" holder that kept the screen on later).
- **Library.** The card ⋯ "Open in Reader" is disabled for a PDF without a workspace (the card tap
  shares it, as for TXT). Selection actions and leaving selection keep the Chapters window.
- **ReaderServer.** An oversized event body (up to 1 MiB) is read and dropped before the 413, so
  the close does not reset the connection (on Windows the client lost the 413).
- **Reader content isolation (verification round).** `sanitize_book_html` left a `<script>` that sits
  inside an SVG / MathML `<style>` untouched (html.parser keeps style contents as raw text; a
  browser parses markup there), and the page-wide nonce stamp then authorised it. Style text
  holding `<` is now made inert (`inert_css`, serialised verbatim as a bs4 `Stylesheet`), and
  `DocumentBuilder.build` defangs any `<script` / `</script` left in book-derived markup
  (`defang_book_scripts`) before the shell wraps it, so only the page's own scripts can carry
  the nonce. Mobile only; regression test
  `test_style_raw_text_in_svg_and_math_never_becomes_a_nonced_script`.

## U6 QA Scanner helpers (qa_scan_runtime additions; QA_Scanner_GUI and scan_html_folder rewired)

Moved at BASE_SHA e28e3a0f (the U5 commit; the parity oracle, goldens and trace oracle were
re-frozen there). Line numbers refer to `git show e28e3a0f:src/<file>`: QA_Scanner_GUI.py (QG),
qa_scan_runtime.py (QR), scan_html_folder.py (SH). tests/test_qa_runtime_additions.py pins the
moves: verbatim AST checks, a differential fuzz against the legacy source, and an end-to-end quick
scan (desktop environment against mobile environment).

### What moved
- QG 176-397 were copied byte for byte into qa_scan_runtime: `_qa_owner_output_mode`,
  `_qa_owner_uses_truncation_context`, `_qa_vision_ocr_source_path`, `_normalize_target_language`,
  `_normalize_source_language`, `check_epub_folder_match` and `normalize_name_for_comparison`.
  QA_Scanner_GUI imports them under the same names, so they are the same objects as before for
  `run_qa_scan`, the settings dialog and the OCR source rows.
- The search in `open_latest_qa_report` (QG 488-540) is now
  `qa_scan_runtime.find_latest_qa_report(override_dir, last_report_path)`. The method keeps its
  message boxes, the `last_qa_report_path` update, `openUrl` and its log lines.
- The AI-truncation default prompt (QG 4693-4703) is now `DEFAULT_AI_TRUNCATION_PROMPT`. The dialog
  keeps its local `_ai_trunc_default_prompt = DEFAULT_AI_TRUNCATION_PROMPT`, so the generated
  schema (`$expr: _ai_trunc_default_prompt`) is unchanged.
- The Custom-mode defaults (QG 1377-1388) are now `DEFAULT_CUSTOM_MODE_SETTINGS`. The dialog uses
  `custom_settings = dict(DEFAULT_CUSTOM_MODE_SETTINGS)`, a fresh copy each time, as the literal was.
- U4 carry-over: the `DEFAULT_REFUSAL_PATTERNS` list (SH 443-464) became
  `from key_pool_service import DEFAULT_REFUSAL_PATTERNS`. It has the same 30 strings in the same
  order. The function-local copy in TransateKRtoEN (~24120) is untouched: that file has a
  different owner (open item).

### Behaviour deltas (intentional, parity-neutral)
- `_qa_vision_ocr_source_path` builds its last candidate from `dirname(abspath(__file__))`.
  `__file__` is now qa_scan_runtime.py instead of QA_Scanner_GUI.py, which is the same directory
  in source runs and in every PyInstaller spec (both are flat `src` modules). On mobile it is the
  read-only backend directory, where that candidate never exists.
- `open_latest_qa_report` used to import `is_direct_text_qa_path` inside its `try`; the import is
  now module-level. QA_Scanner_GUI already imported qa_scan_runtime at module level, so no
  reachable behaviour changes.
- `scan_html_folder.DEFAULT_REFUSAL_PATTERNS` is now key_pool_service's tuple, so
  `_get_refusal_patterns_for_scan()` returns that tuple when config.json has no
  `refusal_patterns`. Its one caller only iterates the value. The literal check in
  tests/test_key_pool_service.py now finds no copy to compare in scan_html_folder;
  test_qa_runtime_additions asserts the identity instead.

### Mobile forcing (GUI-free; desktop unchanged)
- `qa_scan_runtime.mobile_qa_forcing_active()` is `not mobile_runtime.processes_available()`.
  That is true on Glossarion Mobile and whenever GLOSSARION_NO_PROCESSES is set. In that case
  `prepare_qa_scan_settings` sets `use_thread_executor = True`, and
  `apply_qa_scan_env_from_settings` adds two variables through `mobile_qa_env_overrides()`:
  `QA_USE_THREAD_EXECUTOR=1` and `AI_HUNTER_MAX_WORKERS`. The worker value is
  `MOBILE_QA_MAX_WORKERS` (2), or a smaller positive value already in the environment.
  `restore_env` restores both with the other QA_* variables.
- Every caller of `run_qa_scan_path` gets the forcing: the QA tool job, the post-translation scan
  and the Multipass "Failed" scan.
- On desktop, `mobile_qa_env_overrides()` returns `{}`, so the env mapping and the prepared
  settings are the legacy ones (fuzzed over 500 states against the legacy module).
- scan_html_folder already used threads when processes are unavailable (the U1 gates). The forcing
  also caps the worker count. Without it, a mobile scan used the `AI_HUNTER_MAX_WORKERS` that the
  HeadlessOwner exports (`cpu_count // 2`, for example 8 threads).
- End to end: the U3 offline E2E output workspace (`e2e --keep --only e2e_translate_glossary_off`)
  was quick-scanned twice. The first run used a real offscreen TranslatorGUI through
  `run_qa_scan(mode_override='quick-scan', non_interactive=True)` (auto-searched output folder,
  ProcessPoolExecutor). The second used the mobile environment with `HeadlessOwner` and
  `run_qa_scan_path` (2 threads, PySide6 never imported). With langdetect seeded, all 46
  workspace files are identical, including the 4 report files and translation_progress.json.

### Desktop defaults / bugs found (not fixed)
1. **Custom word-count multipliers never reach a scan.** `normalize_qa_scan_settings` (QR 198-206)
   always replaces `word_count_multipliers` with `CANONICAL_WORD_COUNT_MULTIPLIERS`. GUI scans,
   the post-translation scan and the Multipass "Failed" scan all pass through it
   (`run_qa_scan_path` -> `prepare_qa_scan_settings`). So the settings dialog's manual sliders
   (`use_auto_multipliers = False`, QG 5795-5817) are saved but have no effect.
2. **The QA default sets disagree.** The sets are: the GUI prewarm dict (QG 547-587), the widget
   defaults and save mirror (QG ~3634-5990), Reset (QG 6040-6320), `default_qa_scan_settings`
   (QR 129-195), save_config's `default_qa_settings` (settings_persistence 677), the `.get`
   defaults of `apply_qa_scan_env_from_settings` (QR 263-345) and run_env (2350-2374), and the
   scanner's `.get` defaults.
   - `check_missing_beautifulsoup_tags`: True in the GUI prewarm, checkbox, save mirror and Reset
     (QG 580, 3867, 5929, 6081, 6307). False in the runtime, save_config and scanner (QR 167,
     321; settings_persistence 677; SH 9858, 10655). Normalisation fills False when the key is
     unsaved, so on a fresh install the dialog shows the box ticked while scans run with the
     check off, until the dialog is saved once.
   - `check_missing_header_tags`: settings default True (QR 166; QG 4432, 6084, 6311). The env
     mirror and the scanner default to False (QR 320; SH 9745, 10375, 10990).
   - `truncation_embed_threshold`: the slider defaults to 45 (QG 4618). The runtime, Reset and
     scanner use 30 (QR 178; QG 6092; SH 11826).
   - `punctuation_loss_threshold`: 49 (QR 155; QG 3634, 6061). The env mirror defaults to 50
     (QR 338), and so does the scanner docstring (SH 2305).
   - `check_translation_artifacts` (QR 140) and `check_word_count_ratio` (QR 181) default to True,
     but the env mirror defaults them to False (QR 281, 326). This is latent on the
     `run_qa_scan_path` path, which normalises first; it applies only to callers that pass
     unnormalised dicts.
3. **The Custom mode's `min_duplicate_word_count` is dead.** It is a default (QG 1387) and is
   loaded from saved settings (QG 1406). But the dialog has no widget for it, and neither
   "Start Scan" (QG 1638-1650) nor "Save Settings" (QG 1658-1670) writes it.
4. **Custom "Save Settings" writes the wrong config.json in frozen builds.** It writes
   `os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')` (QG 1693), next to
   the module. In PyInstaller builds that is `_internal`, not the `CONFIG_FILE` beside the exe.
   The write also bypasses config_store and the config backups.
5. **A second Custom dialog has different key names.** The module-level
   `show_custom_detection_dialog` (QG 6445-, used by scan_html_folder's CLI `--interactive`,
   SH 12566) keeps its own defaults: `text_similarity` / `semantic_analysis` /
   `structural_patterns` / `minhash_similarity`. `run_qa_scan` uses `similarity` / `semantic` /
   `structural` / `minhash_threshold`. It was not moved, since it is a Qt dialog for the CLI.
6. **Duplicated constants**, identical today, so they can drift:
   - `DEFAULT_AI_ARTIFACT_PATTERNS` (QG 25-48, QR 32-55, SH 360 as a tuple).
   - `DEFAULT_AI_THINKING_PREAMBLE_PATTERNS` (QG 50-59, QR 633-642, SH 1491 as a tuple).
   - The AI-truncation prompt: now `DEFAULT_AI_TRUNCATION_PROMPT`, and again as the scanner's
     fallback (SH 7187-7197).
   - `_normalize_target_language` (moved) and the older `normalize_target_language` (QR 87-126)
     disagree. Whitespace-only input raises IndexError in the moved copy (`s.split()[0]`) and
     returns "english" in the runtime copy, which also `str()`s non-strings.
7. **`apply_qa_scan_env_from_settings` lacks two variables** that the dialog's save handler
   (QG 5896, 5921) and run_env (2355, 2374) export: `AI_HUNTER_MAX_WORKERS` and
   `QA_EXCLUDE_RUBY_TAGS` (shared-core design §4 item 9). A desktop scan therefore uses whatever
   the last save or translation run left in `os.environ`. On mobile, the forcing now sets
   `AI_HUNTER_MAX_WORKERS`.
8. **QA reports are not reproducible on short or ambiguous text.** scan_html_folder never seeds
   langdetect (SH 32, 1310; `DetectorFactory.seed` is unset). On the E2E workspace, the
   romanised-Korean `TOC.txt` / `translated_headers.txt` were flagged
   `Language_mismatch_detected_HR/SW_expected_English` at random: 3 of 4 desktop runs and 3 of 4
   mobile runs, with different files each time. The reports and `qa_failed` marks change between
   identical scans. Seeding (`DetectorFactory.seed = 0`) would make them deterministic, but it
   changes report output, so it needs approval. The E2E comparison seeds langdetect in every
   process through a `sitecustomize` (test-only).
9. **Executor log lines are misleading.** The scanner always logs "🔍 Starting scan with
   ProcessPoolExecutor" (SH 10319) and "⚡ ProcessPoolExecutor: ENABLED - Maximum performance
   achieved!" (SH 12264), even when it scans on a thread pool (the desktop thread toggle, and
   every mobile scan). Mobile shows the desktop text unchanged.

## U6 Glossary Editor document (glossary_document; GlossaryManager_GUI rewired)

Frozen source: GlossaryManager_GUI.py at e28e3a0f (U6 base; "GM n" below). The pure inner
functions of the Glossary Editor closure (`_setup_glossary_editor_tab`, GM 7563-12593), the
`save_edit` body of `_on_tree_double_click` (GM 12676-12750), `convert_glossary_format`
(GM 12792-12995), the editor's parse helpers (GM 1121-1555), five module-level gender helpers
(GM 91-177) and the Glossary Manager's prompt-profile config helpers (GM 457-568, 4144-4156)
moved to `src/glossary_document.py` with explicit parameters. The closures and methods stay in
GlossaryManager_GUI as thin wrappers with their dialogs, widgets and message boxes; the
module-level helpers are re-exported under their old names.

### What moved (and how the desktop calls it)

- Every moved body is the frozen text plus documented edits (tests/test_glossary_document.py
  `MOVED_BLOCKS`, 102 blocks): `self` state becomes explicit parameters, or a *document* object
  with the attribute names the desktop keeps on TranslatorGUI (`current_glossary_data`,
  `current_glossary_format`, `current_glossary_sections`, `current_gender_tracker_data/path`,
  `_pending_gender_decisions`, `_gender_variants_pending_save`, `glossary_column_fields`,
  `_original_translated_map`). The desktop passes `self`, so partial updates on an exception
  (for example `save_document` replacing the data with the gender-resolved list before a failed
  write) stay exactly as before.
- Tree items become a row protocol (`columnCount/text/setText/ref/set_ref`;
  `GlossaryManager_GUI._EditorTreeRow` over a QTreeWidgetItem, `glossary_document.EditorRow` for
  GUI-free callers). `update_row_highlight` runs through `on_replaced(col_key, after)` at the
  same point.
- GlossaryManager_GUI differs from the frozen file only in 91 rewired spans
  (`REWIRE_SPANS`, checked line by line). `parse_token_efficient_glossary` (GM 8567-8766, the
  undo/redo restore path's copy of `_parse_editor_token_glossary_async`) is now a call to the
  same `parse_token_glossary` (tier D: frozen copy vs shared function, equal on 500 states).
  `get_type_limit` (GM 9885-9896) and the unused `_format_tracker_location`,
  `_editor_translated_output_files`, `_read_translated_output_texts`,
  `_resolve_epub_output_dir`, `_trim_undo_stack` and `_entry_kind` closures went with their
  callers.
- Process-environment reads and writes are unchanged (OUTPUT_DIRECTORY / OUTPUT_DIR,
  GLOSSARY_SHARED_DIR, UNIFIED_GLOSSARY_RESOLVED_KEY, GLOSSARY_SKIP_GENDER_TRACKING, EPUB_PATH,
  EXTRACTION_WORKERS are read where they were; Remove Duplicates still sets
  GLOSSARY_DISABLE_HONORIFICS_FILTER).
- Evaluation order: a few wrappers read owner attributes eagerly that the frozen code read
  lazily (`get_baseline_translated`'s item ref, `update_html_files`' worker settings,
  `check_entry_matches`' checkbox states). These are attribute reads without side effects.

### Desktop bugs found (recorded, not fixed)

1. **Save As to a `.csv` file writes an empty file for token-CSV and JSON-object glossaries.**
   `save_as_glossary` (GM 10570-10591) writes rows only when the format is `list` (legacy CSV);
   a `token_csv` or `dict` glossary saved as `.csv` produces a 0-byte file, then the editor
   points at that file. Export Selection / Save As CSV also drop descriptions and custom fields
   and blank the gender of types without `has_gender` (GM 10067-10074, 10584-10591).
2. **BOM-prefixed glossaries.** The editor reads with `utf-8`, not `utf-8-sig`. A token CSV
   starting with a BOM loses its `Glossary Columns:` header (columns fall back to the defaults);
   a JSON glossary with a BOM fails to load ("Failed to load glossary"). Saving writes no BOM.
3. **The two load paths disagree.** The background load (`_parse_glossary_file_for_editor_async`,
   now `parse_glossary_file`) folds tracked gender variants, matches `.csv` case-insensitively
   and skips non-dict JSON items. The undo/redo restore path (GM 8768-8963, now
   `reparse_glossary_file`) keeps the variants, matches `.csv` case-sensitively (`Book.CSV` is
   parsed as JSON and fails) and raises on non-dict JSON items.
4. **Most actions are not undoable.** Delete Selected pushes an undo snapshot, then saves and
   reloads; the reload (`apply_loaded_glossary_result`, GM 8496-8497) clears the undo and redo
   history. The same holds for Clean Empty Fields, Remove Duplicates, Trim, Filter and a
   Convert onto the open file. Only cell edits, Replace and Resolve Gender stay undoable until
   the next save.
5. **Remove Duplicates backs up only when `glossary_auto_backup` is explicitly true**
   (`config.get('glossary_auto_backup', False)`, GM 9284). Every other action treats a missing
   setting as on (`create_glossary_backup`, Backup Settings dialog).
6. **Trim on a JSON-object glossary** computes `entries_to_remove` from `len(data)` (the number of
   top-level keys, normally 1) instead of the entry count (GM 9611), so the backup is skipped or
   misnamed. A negative "Keep first" value trims from the end (`data[:-n]`).
7. **Filter Entries Preview on a JSON-object glossary with search text raises** inside the slot:
   `check_entry_matches` calls `.values()` on the entry string (GM 9910, dict branch 9956).
8. **The external-change watcher restarts after every Save and Undo/Redo**
   (`_editor_auto_reload_timer.start()`, GM 10553 and 8537), even when it was stopped. Actions
   that write the file without updating `_editor_last_mtime` (Delete, Clean, Trim, Filter,
   Convert) therefore trigger a second reload on the next 500 ms tick.
9. **`_update_html_files_legacy`** (GM 10113-10287) is dead code; left in place.
10. **Different fallback defaults.** `update_html_files` falls back to
    `enable_parallel_extraction=False` when the owner lacks the attribute, while
    `ConfigStateMixin` defaults it to True. The custom-entry-type fallback dicts in the editor use
    `terms`; the owner default (owner_state 567) uses `term`. Neither differs on a real desktop
    (the attributes always exist).
11. **The Balanced/Full and Minimal profile combo boxes keep typed names.** They are editable
    without `setInsertPolicy(NoInsert)` (GM 773-774; the refinement combo sets it). Typing a name
    and pressing Enter adds it to the dropdown although no profile exists, and "+ New Profile"
    then skips that number ("New Profile #2"), since it also avoids names in the dropdown.
12. **Profile configs edited by hand.** A Balanced/Full or Minimal bucket holding a profile named
    "Default" (any case) or a non-text prompt is kept as is by the desktop actions. After a delete
    the desktop picks the first non-Default profile, while the shared Default-plus-named engine
    picks the first key. The UI cannot create such entries.

### GUI-free semantics for mobile (GlossaryDocument)

- `GlossaryDocument` runs each desktop action's shared steps in the same order, without the
  dialogs: confirmations are asked by the caller first, error boxes raise
  `GlossaryEditorError(text)`, information boxes come back as `(title, text)` with the desktop
  strings. Tier M replays every tier F operation sequence on a third copy of each real glossary:
  the glossary, tracker, backups, output files, data, sections, gender state, baseline and undo
  history match the frozen desktop editor after every step.
- `EditorOwner(config)` holds the TranslatorGUI state the editor reads (`custom_entry_types`, the
  gender-tracker settings, the parallel-update settings, the seeded `custom_glossary_fields`),
  computed like `ConfigStateMixin._init_config_state` (checked against HeadlessOwner).
- `editor_rows()` gives the rows the desktop tree shows, including the tracker label the tree
  puts in the gender column; Find/Replace on those rows changes the same cells as the desktop.
- Backups go through a caller-supplied `backup(doc, operation_name)` (the desktop's
  `create_glossary_backup`, which glossary_files owns). `list_editor_backups` lists what that
  function writes and `_clean_old_backups` prunes. `restore_backup` is mobile-only: the desktop
  has no restore button for editor backups (users copy files out of the Backups folder); it
  takes a backup of the current state and then saves the backup's entries like an edit.
- Remove Duplicates sets `GLOSSARY_DISABLE_HONORIFICS_FILTER` in `os.environ`, as the desktop
  does. A mobile caller should not run it while a job thread reads the environment.
- **Glossary Manager prompt profiles.** `GlossaryPromptProfiles` (Balanced/Full, Minimal) and
  `RefinementPromptProfiles` (system + user pairs) run prompt_profiles' Default-plus-named engine
  (`PrefillState` and `prefill_*`, as the "Asst. Prompt" dialog does). A refinement pair travels
  through the engine as one canonical JSON text. They write the same config keys as the desktop
  actions. The desktop action handlers (`_on_glossary_prompt_profile_selected`, `_auto_save_...`,
  `_new_...`, `_save_...`, `_delete_...` and the `_create_refinement_prompt_profile_controls`
  closures) are unchanged and still carry their own copy of the Default-plus-named steps.
  Tier P pins them to the shared engine on 60 random action sequences per bucket. Moving the
  desktop handlers onto `prefill_*` is left for a later milestone; item 12 is the only case that
  would change.
- Opening a profile row: the desktop stages the prompt shown in the editor into the selected
  profile (or Default) before applying the active profile. The classes take that prompt as
  `current_text` / `system` + `user`; the mobile settings layer supplies the value its prompt
  field shows.

### U6 parity harness additions (tests/test_glossary_document.py)

- Tier D runs the frozen methods, and closures extracted from the frozen file with their free
  variables as fakes, against the working-tree wrappers and the shared functions on 500 seeded
  states each. Covered: parsing, saving, Convert Format, the view helpers, Filter Entries,
  Find/Replace (Find Next / Replace / Replace All with the output-file fallback), Update output
  files, auto-selection and the hide-unused helpers and worker.
- Tier F builds the real editor tab twice offscreen: the frozen module and the working-tree
  module, both loaded with `__file__` in an empty temp folder so the user's src/Glossary is never
  listed. It drives both through their buttons, shortcuts and dialogs on copies of five real
  src/Glossary books plus JSON-list, JSON-object, comma-CSV and \x1F-CSV variants. Every step is
  compared on files, tree rows (texts, refs, hidden flags, highlight brushes), editor state,
  undo stacks, boxes and logs. The gender tracker's `updated_at` stamp is masked, backup names use
  a deterministic clock, and the 500 ms watcher only runs when the test triggers it.
- Tier G pins the frozen desktop's save and Convert Format output for 49 synthetic glossaries as
  hashes. glossary_document is checked against them without Qt (CI, Python 3.10); a Qt test
  re-derives them from the frozen closures.
- Tier P drives the frozen and working-tree prompt-profile rows (real widgets) and the
  GUI-free classes through random select / edit / rename / new / save / delete / failed-save
  sequences. Each step compares the config, owner attributes, editor text, boxes and logs. The
  Balanced/Full rows are selected through their handler, without Qt's insert-on-Enter (item 11).

## U6 Glossary file actions, Parallel EPUB pair core and the glossary Stop (glossary_files, parallel_epub_core, stop_control.request_glossary_stop)

Frozen sources: translator_gui.py and parallel_epub_glossary.py at e28e3a0f (U6 base; "TG n" /
"PEG n" below). HEAD's src tree is identical to e28e3a0f, so the e28e3a0f parity oracles (legacy,
legacy_trace, golden) are the HEAD freeze.

### What moved (and how the desktop calls it)

- `src/parallel_epub_core.py` (new, GUI-free, Python 3.10):
  - PEG 52-512 verbatim: the constants, `chapter_filename/text`, `compact_parallel_epub_selection`,
    `restore_parallel_epub_pairs`, the auto-mapping helpers and `auto_map_epub_chapters`,
    `apply_parallel_epub_wrapper` and `write_parallel_epub`. parallel_epub_glossary re-exports
    every name and keeps its top-level Qt import; parallel_epub_core imports without Qt.
  - The pure halves of `ParallelEpubPairDialog` methods: `load_parallel_epub_documents`
    (`_start_epub_load`'s loader), `prepare_persisted_parallel_epub_selection` /
    `parallel_epub_selection_matches` / `persisted_parallel_epub_rows`
    (`restore_persisted_selection`, `_apply_pending_persisted_mapping`),
    `translated_mapping_label`, `offset_parallel_epub_mapping` (`_apply_mapping_offset`),
    `valid_parallel_epub_rows`, `selected_parallel_epub_mapping`, `unpaired_file_counts`,
    `unpaired_warning_text`, `parallel_epub_mapping_status` (`_update_mapping_status`),
    `validate_parallel_epub_pair` / `build_parallel_epub_pairs` (`_accept_pair`),
    `parallel_epub_profiles` / `active_parallel_epub_profile` (`__init__`),
    `parallel_epub_prompt_settings` (`_persist_prompt_settings`). The dialog keeps the widgets.
  - TranslatorGUI pair helpers with `config` in place of `self.config`:
    `load_parallel_epub_chapters` (TG 28458), `build_parallel_epub_pair_artifact` (TG 28469),
    `resolve_parallel_epub_glossary_output_dir` (TG 20032; the selected-pair lookup for an empty
    raw path stays in TranslatorGUI), `parallel_epub_mapping_sidecar_path` /
    `write_parallel_epub_mapping_sidecar` / `read_parallel_epub_mapping_sidecar` (TG 20073-20128),
    `parallel_epub_pair_source_state` (the `_parallel_epub_pair_source` record of
    `_activate_parallel_epub_pair_source`) and `rebuild_parallel_epub_pair_result`
    (`_start_parallel_epub_pair_restore`'s worker up to the working EPUB).
- `src/glossary_files.py` (new, GUI-free, Python 3.10): `create_glossary_backup` /
  `clean_old_backups` (TG 10344, 10434; the editor path, data and "Continue anyway?" question
  are parameters), the 🗑️ / ↩️ closures of the auto-glossary row (TG 19463-19736) as
  `selected_glossary_epubs`, `collect_glossary_files_for_inputs`, `glossary_delete_display`,
  `delete_glossary_files`, `find_latest_glossary_backup`, `restore_glossary_backup` (the
  boxes, sounds, owner-state resets and auto-load re-trigger stay in the closures), the Map
  Glossaries to EPUBs data steps (TG 31572: `normalize_glossary_drop_path`,
  `is_allowed_glossary_file`, `mapped_glossary_for_input`, `build_glossary_mapping`,
  `copy_mapped_glossaries_to_outputs`) and `comprehensive_json_fix` / `analyze_json_errors`
  (TG 32288-32414). The U3 auto-mapping helpers stay in `translation_pipeline`
  (GlossaryPipelineMixin); `guess_glossary_for_input_file` / `copy_glossary_to_output_folders`
  run those methods on a bare owner (config / base_dir / append_log; no HeadlessOwner, no env
  writes) for GUI-free callers.
- `stop_control.request_glossary_stop` + `kill_glossary_helper_subprocesses`: the non-widget
  part of `stop_glossary_extraction` (TG 26528-26748) in desktop order: GRACEFUL_STOP ->
  graceful: silence the HTTP loggers -> `set_stop_requested()` -> immediate:
  TRANSLATION_CANCELLED=1 / GRACEFUL_STOP_COMPLETED=0, `glossary_stop_flag(True)`, the extractor
  and client stop flags, then the run-id-guarded cleanup thread (hard cancel, helper processes,
  GLOSSARY_STOP_FILE) -> the stop-mode log line. `stop_glossary_extraction` keeps the button /
  label code, the double-click rule (now through `register_stop_click`) and the idle-reset poll.
  This closes DISCREPANCIES U3 item 4's "the glossary stop was not rewired".
- translator_gui.py differs from the frozen file only in 40 rewired spans and
  parallel_epub_glossary.py in 39 (tests/test_glossary_files.py `TG_REWIRE_SPANS`,
  tests/test_parallel_epub_core.py `PEG_REWIRE_SPANS`). Every moved body is the frozen text plus
  documented edits (`MOVED_BODIES`, 16 + 18 bodies); the restructured dialog cores (offset, saved
  rows, selected mapping, status, validation, prompt settings) are pinned by the differential
  tiers instead.

### Behaviour deltas (intentional, parity-neutral)

- Evaluation order: `_accept_pair` reads the wrapper / system prompt and the selected mapping
  before its "still loading" check; `_apply_pending_persisted_mapping` writes each row once
  (the final state of the old clear-then-restore passes); `_apply_mapping_offset` computes every
  row before touching the table; `create_glossary_backup` reads the editor path once;
  `_delete_current_glossary` resolves `self._guess_glossary_for_input_file` before the loop and
  reads the auto / manual glossary attributes once. All are reads without side effects.
- `kill_glossary_helper_subprocesses` returns at once where subprocesses are unavailable
  (mobile); on desktop `subprocesses_available()` is always true. The graceful logger silencing
  goes through `silence_http_loggers` (the same six loggers).
- parallel_epub_glossary no longer imports `html`, `re`, `uuid`, `ebooklib.epub` and
  `special_file_flags` (their only users moved).
- GUI-free defaults: `create_glossary_backup` without `ask_continue` answers No after a failed
  backup; `copy_mapped_glossaries_to_outputs` returns (copied, already in place, failed);
  `resolve_parallel_epub_glossary_output_dir` returns "" for an empty raw path.
- `comprehensive_json_fix`: one line of its replacement table continues a triple-quoted key, so
  it keeps its absolute indentation (the key's text is unchanged; see bug 4).

### Desktop bugs found (recorded, not fixed)

1. **Delete Glossary leaves a migrated legacy glossary behind.** The collection lists a flat
   `Glossary/<book>_glossary.*`, then the auto-map guess (`_get_glossary_dir_candidates` ->
   `migrate_all_legacy_glossary_files`) moves that file into `Glossary/<book>/` before the
   delete step, which logs "⚠️ Failed to delete … [WinError 2]" and leaves the migrated file in
   place (seen in the offscreen real-GUI smoke on both trees).
2. **Restore Glossary restores one backup folder.** Delete moves every file into a
   `Backups/<timestamp>/` beside it (several folders, one timestamp). `_find_latest_backup`
   keeps the first folder with the greatest name, so Restore brings back only that folder's
   files.
3. **Some backups are never offered.** `_find_latest_backup` never looks in
   `<book>/Glossary/Backups` (where Delete puts the per-book `Glossary/<book>_glossary.*`); without
   an output override it also mixes cwd-relative `Glossary/` folders with `_get_app_dir()` ones,
   while Delete uses `_get_app_dir()` only.
4. **The JSON repair cannot fix curly quotes.** `_comprehensive_json_fix`'s table maps `"` to `"`
   twice and has one accidental key, the triple-quoted text `: "'",  # Left smart apostrophe` +
   newline + indentation; only dashes, the ellipsis, ZWSP and NBSP are normalised.
   `_analyze_json_errors`' smart-quote check `r'[''""…]'` is the string `[""…]`. Load Glossary's
   JSON auto-fix branch is dead code (a duplicated `elif file_extension == '.json'` passes
   first), so only `_convert_json_to_txt` runs the repair.
5. **Delete Glossary computes an unused mode.** `mode = config.get('auto_glossary_mode',
   'off').lower()` / `is_balanced_full` are never read; a `null` mode in config.json makes the
   whole delete fail with "⚠️ Error deleting glossary".
6. **Two Manual Glossary Only output resolvers.** Map Glossaries' Save copies into
   `<override>/<book>` (`_resolve_out_dir`, no subtitle-ZIP grouping) while Load Glossary uses
   `_copy_glossary_to_output_folders` -> `_resolve_translation_output_dir`; the log texts differ.
7. **The glossary Stop's helper list is shorter** than the translation Stop's (no AuthND /
   Gemini-Free token helpers), so a glossary run on those routes can leave a token helper
   running after an immediate Stop. Kept (`kill_glossary_helper_subprocesses`).

### GUI-free semantics for mobile

- Mobile glossary jobs should stop through `stop_control.request_glossary_stop(graceful=,
  set_stop_requested=, log=, glossary_stop_flag=, get_run_id=)`. The latch is
  `stop_requested` only (the desktop glossary Stop never sets `graceful_stop_active`). The
  desktop's force rule needs its "Finishing..." label, so a mobile force stop is the
  JobService's own second-tap path calling it with `graceful=False`. Switching
  `services/jobs.py` is the mobile-glossary agent's task; until then glossary jobs still use
  the translation protocol (U3 item 4).
- The mobile Parallel EPUB pair screen and job can drop their local copies:
  `load_parallel_epub_documents` (pair_load), `offset_parallel_epub_mapping` /
  `unpaired_file_counts` / `unpaired_warning_text` / `parallel_epub_mapping_status` /
  `valid_parallel_epub_rows` / `persisted_parallel_epub_rows` (MappingModel),
  `validate_parallel_epub_pair` / `build_parallel_epub_pairs` (Accept),
  `parallel_epub_profiles` / `active_parallel_epub_profile` / `parallel_epub_prompt_settings`
  (profiles), and for the job `rebuild_parallel_epub_pair_result` +
  `build_parallel_epub_pair_artifact` + `parallel_epub_pair_source_state` (the desktop restore
  path; tier F checks this route against the frozen desktop's working EPUB, record and sidecar).
- `collect_glossary_files_for_inputs` defaults its guess to `guess_glossary_for_input_file` for
  the given config, so a Library "Delete glossary files" needs no owner.

### Settings schema (for Integrate)

- `src/mobile/tools/schema_extract.py` reads config keys from its module lists only. With the
  moves, HEAD + the chain-2 files regenerates `settings_schema_data.py` with three changes:
  `glossary_max_backups` loses its `read` origin, `parallel_epub_glossary_profiles` and
  `parallel_epub_glossary_wrapper_prompt` disappear (their reads now live in glossary_files /
  parallel_epub_core), and `last_epub_path` becomes a read-origin key with `main.file` first.
  The last one is static analysis only: `_find_latest_backup` now passes
  `get_current_epub_path` to `selected_glossary_epubs` instead of calling it, so the
  generator no longer sees the startup call edge (`_update_restore_visibility`, 500 ms after
  start). Adding `glossary_files.py` and `parallel_epub_core.py` to `OWNER_MODULES` restores
  the first two and adds `parallel_epub_glossary_active_profile` (read by the dialog, which
  the generator never scanned). The freshness test already needs a regeneration for the
  Glossary Manager changes.

### U6 parity harness additions

- tests/test_parallel_epub_core.py: hygiene; verbatim block + 18 lifted bodies + rewire spans;
  the moved functions vs the frozen module (500 random chapter sets); the frozen dialog methods
  vs the working-tree ones on recording fakes (500 states of offset / unmap / saved mapping /
  accept / persist / restore sequences); the background loader closure and the profile set-up;
  the frozen TranslatorGUI pair helpers vs the wrappers (500 glossary-folder / sidecar states);
  fixture EPUB pairs through the frozen and working-tree desktop flow (load, map, accept,
  activate, sidecar, working EPUB, saved-pair restore) and through the GUI-free mobile route;
  the real dialog offscreen, frozen vs working tree.
- tests/test_glossary_files.py: hygiene; 16 moved bodies, the Stop protocol statements in order,
  the translator_gui rewire spans; editor backups (frozen method vs wrapper vs shared function),
  delete / latest backup / restore closures and the GUI-free sequence on random layouts, JSON
  repair, the owner-free adapters vs the mixin methods (500 states each, same fresh folder for
  every side, `_get_app_dir` always sandboxed); the glossary Stop trace (frozen vs wrapper vs a
  direct `request_glossary_stop`); the Map Glossaries dialog offscreen, frozen vs working tree.
- Tier T: two scenarios, `glossary_immediate_stop` and `glossary_graceful_stop`
  (tests/parity/trace_scenarios.py). For glossary entries the harness desktop run is now the
  Extract Glossary button (`run_glossary_extraction_thread`: run-start resets, then the direct
  run on the inline executor), a click is `stop_glossary_extraction`, and the mobile click is
  `stop_control.request_glossary_stop`. The mobile projection treats an absent
  `graceful_stop_active` as False (the pipelines read it with `getattr(..., False)`; a fresh mobile
  glossary job has none until a stop latches it). legacy / desktop / stop_control / mixins agree.
- Offscreen real-GUI smoke (scratch `desktop_smoke_u6c2.py`): the live TranslatorGUI of the
  working tree and of `git archive HEAD src` in sandboxes drive the 🗑️ / ↩️ buttons, editor
  backups, Load Glossary -> Map Glossaries (Auto-Fill, Save, Manual Glossary Only copy), the
  Parallel EPUB Pair dialog (load, offset, accept, activate, saved-pair restore) and immediate /
  graceful glossary Stop; both trees report identical files, logs, boxes and state.

## U6 Integrate (wiring, packaging, glossary / QA / compile on the shared U6 cores)

Oracles were re-frozen at HEAD e604e5a0 (its `src/` is byte-identical to U5 e28e3a0f, the U6 base):
`freeze_legacy.py --sha HEAD`, `capture_golden.py`, `trace_harness.py --freeze --sha HEAD`. All
parity tiers pass (448 tests; the two skips are the GLOSSARION_PY310 probe, run separately and
passing, and the documented desktop-only trace race).

### Desktop changes made by the integration (each pinned)

- **Unified glossary "Rebuild Now" helpers.** `glossary_document.unified_glossary_shared_dir(config)`
  and `unified_rebuild_settings(config, shared_dir)` are `GlossaryManagerMixin._unified_glossary_shared_dir`
  and the settings snapshot of `_rebuild_unified_glossary_now`, verbatim (frozen U6-base lines
  5800-5809 and 5854-5859; documented edits: `self.config` -> `config`, and
  `from translator_gui import _get_app_dir` -> `from app_paths import _get_app_dir`, the same
  function object since U1). GlossaryManager_GUI calls them (two more `REWIRE_SPANS`);
  tests/test_glossary_document.py pins both blocks and compares the frozen method / snapshot with the
  working-tree wrapper and the shared functions on 500 random configs. The mobile `unified_glossary`
  job uses them instead of its copied dict.
- **Refusal patterns (U4 carry-over, finished).** `TransateKRtoEN.is_qa_failed_response` imports
  `key_pool_service.DEFAULT_REFUSAL_PATTERNS` (same 30 strings, same order; the list is only
  iterated). 5,000 random responses (refusal phrases, mixed case, lengths around the 1,000-char
  window) give the same verdict from the HEAD function and the working tree.
  tests/test_key_pool_service.py now asserts that neither scan_html_folder nor TransateKRtoEN keeps
  a copy (`scan_html_folder.DEFAULT_REFUSAL_PATTERNS is key_pool_service.DEFAULT_REFUSAL_PATTERNS`).
- **Settings schema generator.** `schema_extract` scans `glossary_files.py` and
  `parallel_epub_core.py` as owner modules and `glossary_document.py` as a dialog module (UI-site
  roots: the editor functions -> `glossary.editor`, the prompt-profile helpers ->
  `glossary.balanced_full`, the refinement defaults -> `glossary.refinement`, the unified helpers ->
  `glossary.unified`). The new `SCAN_EXCLUDE` keeps glossary_document's mobile-only API
  (`EditorOwner`, `EditorRow`, `editor_rows`, `GlossaryPromptProfiles`, `RefinementPromptProfiles`,
  `GlossaryEditorError`, `GlossaryDocument`) out of the scan, so it adds no records of its own.
  The regenerated `settings_schema_data.py` differs from HEAD only in static-analysis detail:
  - every key's settings section is unchanged except `enabled` (a `.get('enabled')` on entry-type
    dicts that the generator reads as a setting): its first UI site is now `main.model` instead of
    `glossary.editor`, because the editor code it was reached from is no longer nested in the
    Glossary Manager tab function;
  - `parallel_epub_glossary_active_profile` is a new record (read by the Parallel EPUB dialog code
    now in a scanned module; section `glossary.parallel_epub`);
  - `last_epub_path` keeps its default `None`; `default_source` / `init_default` / `origins` change
    (chain 2 note above: the `get_current_epub_path` call edge moved);
  - `custom_glossary_fields`, `output_directory`, `unified_glossary_source_language` and
    `unified_glossary_combine_all_languages` gain or reorder UI sites (`glossary.editor`,
    `glossary.other`); none of them changes section;
  - `output_language` gains the `glossary.editor` UI site (after `other.response`), because
    `glossary_document.unified_glossary_folder_key` and `unified_rebuild_settings` read
    `config['output_language']`; its section stays `main.run` (the first mapped site).

### Mobile wiring (no desktop change)

- `app.py` installs `GlossaryFeature` then `ToolsFeature` after the Reader (each in try/except);
  `SHIPPED_MILESTONES` gains U6. The `tools.text` route (TextEditor, UI_SPEC §4.10) and the File
  browser's Open with / Rename / Delete move to U7 with the rest of §4.10: they were not part of
  the U6 build, and with U6 shipped they would have read "arrives in U6".
- `services/glossary.CONTRACT` names the real `glossary_files` / `parallel_epub_core` functions (one
  name per operation); the "Delete Glossary" text uses `glossary_files.glossary_delete_display`.
  A host test runs delete -> latest backup -> restore and the editor backup (pruned to
  `glossary_max_backups`) through the real `glossary_files`.
- Parallel EPUB pair: `ui/glossary/parallel_pair.PairMapping` keeps only the table cells; offset,
  unmap, restore, selection, unpaired counts / warning, status line, the Accept checks
  (`validate_parallel_epub_pair`), the pairs, the profile set-up and the persisted prompt settings
  are the dialog's `parallel_epub_core` functions (the earlier mobile re-implementation is gone).
  Accept is refused with "EPUB Still Loading" while an EPUB loads (the dialog's `_active_load`).
  The `parallel_pair` job is the desktop's saved-pair restore + activation:
  `rebuild_parallel_epub_pair_result` -> `build_parallel_epub_pair_artifact` ->
  `parallel_epub_pair_source_state`.
- Glossary jobs stop through `stop_control.request_glossary_stop` with the desktop's hooks: the
  loaded extractor's `set_stop_flag` (`glossary_stop_flag`) and the owner's `_glossary_run_id`
  (`get_run_id`); a forced stop is the desktop's double click (graceful False).
- Library / Book page "Compile EPUB" / "Compile PDF" use the desktop's compiler choice
  (`library_core._workspace_compile_kind`): a PDF workspace compiles with `compile_pdf`; an EPUB
  workspace's PDF is the EPUB compile with "Create PDF after EPUB" on (as the Converter does). Before,
  "Compile PDF" on an EPUB workspace ran the PDF workspace compiler.
- Hand-offs that said "arrives in U6": the Book page ⋯ "QA scan" opens Tools › QA Scanner for the
  book (`?out=<bid>`); the chat approval card's raw editor offers "Open in table editor" (the
  Glossary Manager on the same file); the Reader selection's and a chat response's "Add to
  glossary" open the book's (or the chat workspace's glossary.csv) editor with a new entry whose raw
  name is filled in (kept once the user Saves). The chat attachment card's "QA scan" stays disabled,
  now with the reason: the desktop never QA-scans Direct Text workspaces
  (`qa_scan_runtime.is_direct_text_qa_path`).

### Self-test and offline E2E additions

- `smoke` suite `glossary_qa` (also run by `tools/host_smoke.py`, android and ios simulations): a
  token-CSV glossary parsed, edited (one undo step), saved, re-parsed and saved again byte-stable
  through `glossary_document.GlossaryDocument`; a QA quick scan of a three-chapter workspace through
  `qa_scan_runtime.run_qa_scan_path`, which must run on threads on mobile (host_smoke's process
  tripwires stay silent), report every chapter and be found by `find_latest_qa_report`. The job lock
  is held meanwhile.
- `e2e` `glossary_edit_qa_pdf`: `extract_glossary` job -> the Glossary Manager document changes
  이서연's translation to "Seo-yeon Lumen" and saves (a `before_save` backup first) -> a Balanced
  `translate` job sends the edited entry in every chapter prompt that names 이서연 and never the old
  name -> `qa_scan` job (quick scan) reports the 12 chapters -> the Book page's "Compile PDF" spec
  writes the PDF through the PyMuPDF shim (12 pages, 12 outline entries, the translation marker in
  the text). The fake server records the injected glossary lines of each prompt (`glossary_lines`).
  Process hygiene now covers 11 jobs.

### Mobile divergences recorded from the U6 build reports (not fixed)

- **QA Scanner orchestration still dialog-only on desktop** (mobile rebuilds it over the shared
  helpers and desktop strings, pinned by tests_host/test_tools_ui.py): `QA_Scanner_GUI`'s bulk
  `run_scan` loop (the desktop file-name EPUB search is replaced by the Library's resolved raw
  source), the `stop_qa_scan` escalation (mobile: the translation stop protocol plus the scanner's
  own stop flag), `other_settings.delete_translated_headers_file` / `delete_toc_txt_file`,
  `validate_epub_structure_gui`'s result wording and the Load Font handler, and
  `metadata_batch_translator.configure_metadata_fields`' save / merge rules. The QA `mode_data`
  card texts are compared with the desktop source by AST. Each should become one shared function
  both front ends call.
- **Translate Headers Now** runs `translate_headers_standalone.run_translation` per EPUB and rebuilds
  the first EPUB (the desktop button's `run_translate_headers_gui` imports QMessageBox first). An
  existing translated_headers.txt is translated again instead of re-applied, PDF workspaces are
  skipped (Compile PDF translates bookmarks) and keyless models are allowed.
- **Metadata from the Library** passes `output_roots` as a list that can fall out of step with the
  inputs (the job then uses the default output root); the Library home and Book page do not ask
  the desktop "Metadata Already Exists" question (the Headers screen does).
- **QA "Reset to default"** is the Settings page's section reset (the keys are removed, the shared
  defaults apply); the desktop QA dialog writes hard-coded values that differ in places (see the U6
  QA section above).
- **Delete glossary files** clears `manual_glossary_path` only when that file was among the deleted
  ones; the desktop closure resets its owner state (manual / auto-loaded glossary) unconditionally.
- **Manual glossary refinement** ("✨ Refine this" / "✨ Refinement") stays disabled with its reason:
  `Retranslation_GUI._run_manual_glossary_refinement` and the plan step of the Glossary Progress
  confirm closure are not shared yet (`glossary_progress_core.plan_manual_glossary_refinement` /
  `run_manual_glossary_refinement` are the names the `glossary_refine` job binds to).

### Test isolation fix (integration)

- tests/test_glossary_files.py's Stop trace test set `TRANSLATION_CANCELLED=1` / `GRACEFUL_STOP=0` /
  `GRACEFUL_STOP_COMPLETED=0` directly while monkeypatch held earlier records of the same keys; the
  autouse `_isolated` fixture restored the environment and then monkeypatch's teardown re-applied
  those values, so a later test in the same pytest process (the QA desktop-vs-mobile quick scan's
  subprocesses) started already cancelled and wrote no report. The three U6 fixtures now call
  `monkeypatch.undo()` before restoring the environment captured at the start, and the QA E2E
  test drops the stop-signal keys from its subprocess environment.
- tests/test_glossary_document.py tier F (frozen vs working-tree editor tab) used only the user's
  src/Glossary books, which CI does not have (empty parameter set there). It now falls back to a
  synthetic book glossary in the same five layouts (token CSV + gender tracker with a tracked
  conflict, JSON list / dict, legacy CSV, \x1f CSV); `PARITY_U6_SYNTHETIC=1` forces it locally
  (5 / 5 pass).

### Desktop offscreen smoke (integration)

`scratchpad/u6_integ/desktop_smoke_u6.py`: the REAL `TranslatorGUI` (offscreen, sandboxed app dir /
home / temp / Library, the U6 E2E's translated workspace and book glossary as the fixture) for the
working tree and `git archive HEAD src`: the Glossary Manager (all five tabs built and visited:
every checkbox / combo / spin / line-edit / button state; the editor loads the book glossary, a
translated name is edited and Ctrl+S saves it), the QA Scanner's post-translation quick scan
(`run_qa_scan(mode_override='quick-scan', non_interactive=True)`: report files, progress statuses)
and `epub_converter(folder=...)` with "Create PDF after EPUB" (desktop WeasyPrint: the shim's
"PDF engine: mupdf-story" line never appears; EPUB members / chapter documents, PDF pages, outline
and text) give identical results in both trees (timestamps, the src path and thread completion
order normalised).

## U6 review: mobile fixes (no desktop change)

- **Glossary editor.** Swipe-to-delete's snackbar Undo works: `GlossaryService.delete(keep_snapshot=True)`
  takes the shared undo snapshot (`push_undo_snapshot`) before `GlossaryDocument.delete` (which, like
  the desktop, re-reads the file and clears the undo history), and Undo restores it with
  `undo_step` + save + load (`restore_snapshot`). The Undo is refused when the glossary changed
  after that delete (an edit, save, reload or other delete). The shared `delete()` is unchanged.
- **Unsaved state.** A tool box counts as saved only when the shared call wrote and re-read the file
  ("Success"); an "Info" box (Clean Empty Fields "No empty fields found", Remove Duplicates "No
  duplicates found") and Convert Format to another path leave unsaved edits unsaved (Back still asks,
  the Save dot stays, the auto-reload does not discard them).
- **View after a reload.** Every reload (Reload, the auto-reload, Delete, a tool, Undo of a glossary
  step, a backup restore, the swipe Undo) derives the view again like the desktop's load: column
  filters of vanished columns are dropped (`prune_column_filters`) and Hide unused is re-run
  (`_apply_hide_unused_entries_filter` runs after every desktop load), so Replace All targets the
  used rows again instead of shifted source indices.
- **Performance (UI_SPEC §7.3).** Tapping a row in selection mode changes that row in place (1.4 ms
  with 1,500 mounted rows, from ~0.8 s); only entering / leaving selection mode rebuilds the mounted
  rows. Search is debounced (0.25 s) and filtered on the io pool; Replace All runs on the io pool.
  The Parallel EPUB pair screen loads its prompt profiles on the io pool after it opens (the built-in
  default imports the glossary extractor); the Extract sheet resolves Library raw sources on the io
  pool.
- **Dialogs.** The glossary `ask()` / `prompt_text()` dialogs answer No / None when closed any other
  way (Android back): Save no longer stays stuck after a dismissed "Update output files" question.
  "Unsaved changes" › Cancel when switching files keeps the screen's title, gid and input.
- **Prompt profiles.** The Balanced/Full, Minimal and Refinement profile bars use the shared
  `glossary_document.GlossaryPromptProfiles` / `RefinementPromptProfiles` (the classes
  tests/test_glossary_document.py replays against the desktop controls) over a copy of config.json;
  the mobile-only `ProfileBucket` reimplementation is gone. Each action writes the changed keys
  (`GlossaryService.persist_prompt_profiles`; Balanced/Full and Minimal write only their own entry
  of the dicts they share). The stale-Default case (Default text older than a prompt edited in
  Settings) now keeps the edited prompt, and Default delete shows the desktop "glossary prompt
  profile" text.
- **Find / Replace.** UI_SPEC §4.1 / §5 listed scope (Raw / Translated / All), Match case and Whole
  word; the sheet is the desktop dialog (case-insensitive, every column, the shared `row_has_match`
  / `replace_in_row`) and the spec was amended to it rather than adding mobile-only matching rules.
- **Tools.** Compile PDF of an EPUB workspace (`compile_epub` + `pdf_after_epub`) lists the PDFs the
  run wrote as job outputs (Result card Share / Open; the offline E2E checks it). The QA report
  viewer falls back to the native view when the WebView page never posts "ready" (8 s) or a
  resource error is not followed by it (4 s), like the Reader. The Quick Scan sample size typed
  without leaving the field is saved at Start (the desktop persists it when a mode is picked; a
  tap on Start does not unfocus the field on phones). Rebuild Now reads `JobService.busy` as the
  property it is and stays disabled while its job is queued or running. Recorded divergence: while
  another run is active the desktop logs its warning and refuses ("try again when it finishes");
  mobile shows the same warning and queues the rebuild behind the run (jobs run one at a time).

### Second review round (U6; mobile and test-only, no desktop change)

- **Unsaved glossary edits.** The shell has a leave guard: before a navigation disposes screens
  (`AppShell.leaving_entries` for a drawer / sidebar destination, a chat row, a link from outside the
  app; `entries_above` for the main-area screens behind a popped full-screen View on a tablet) the app
  awaits each screen's optional `confirm_leave()`. `GlossaryScreen.confirm_leave` asks the same
  "Unsaved changes" question as Back; Keep editing cancels the navigation (an outside link puts the
  client route back). The client has already popped a full-screen View when its `view_pop` arrives, so
  there Keep editing drops only that View and the editor stays in the main area
  (`pop_view(keep_above=True)`). The desktop dialog only hides on close and keeps its edits; this is
  the mobile counterpart, not a desktop behaviour.
- **Add to glossary over the Reader (tablet).** The editor screen would open in the main area behind
  the full-screen Reader; with a full-screen View on top the new-entry sheet now opens over it
  (UI_SPEC §3.11: "EntrySheet prefilled with the raw term") and Add writes the entry off the UI loop
  (`open_document`, `add_entry`, `save_edits` with the "before_save" backup; a fresh document plus a
  new row has no translated-name change, so no output files are touched). Phones keep the editor.
- **Editor file switch.** Like the desktop (`_apply_hide_unused_entries_filter` runs after every
  load while the checkbox stays checked), Hide unused entries stays on across ◀ ▶ / file name ▾ /
  Save As and is re-run for the new file. Mobile-only: the search box's text keeps filtering the new
  file (it was shown but no longer applied); column filters, the selection and the sort start over.
  The app bar (phone) / main-area title (tablet) shows the new file name (`Screen.app_bar_title`).
- **UI loop.** The Glossaries list behind ◀ ▶ / file name ▾ is read on the io pool when the editor
  opens without it (Book page, chat card, Add to glossary, a deep link) and after Save As; the
  glossary's Library input (`LibraryService.raw_source`) is resolved once on the io pool, including a
  "not found". Parallel EPUB pair "From Library…" reads the books' EPUBs on the io pool and, for the
  translated side, now offers a book's compiled EPUB (`compiled_outputs_blocking`; the method it
  called did not exist), never the raw Library file that list also contains.
- **Jobs of an earlier visit.** A reopened Unified glossary, QA Scanner, Converter or Headers &
  metadata screen follows the queued or running job of its kinds (`JobWatch.adopt`; the Converter
  only a job of its output folder): Stop shows and Rebuild Now / Start / Compile / Translate Headers
  Now stay disabled, so a second tap no longer queues a duplicate job.
- **Use as manual glossary.** The "Use as manual glossary" sheet (with a chat open) answers "nothing"
  when it is closed by its Cancel row, Android back or an outside tap (`ActionSheet(on_cancel=)`), so
  the call returns instead of waiting forever.
- **Parity harness (tests/test_glossary_document.py, test-only).** Tier G hashes the JSON outputs
  with CRLF normalised to LF: `json.dump` in text mode writes CRLF on Windows and LF on Linux (the
  frozen desktop and glossary_document alike), so the pins only held on Windows; the 25 JSON-save
  pins were re-derived (the CSV writers pass `newline=''` and keep their exact pins). Without the
  user's src/Glossary corpus (CI) `real_trackers` falls back to the synthetic 루나 female/male
  tracker, so the file-parser tier still sees collapsed gender variants, and tier F starts the
  synthetic-token fixture with edit · edit · undo · redo · resolve · resolve (the rng draw is kept,
  so later steps are unchanged) so `test_editor_operations_were_exercised` holds on the synthetic
  fixtures. Real-corpus runs are unchanged.

## U7 image / RPG Maker runners (image_job, rpgmaker_job; translator_gui, translation_pipeline, direct_text_store, authgem_auth rewired)

Parent commit (oracles re-frozen there: `legacy/`, `legacy_trace/`, `golden/`): 41814faa.

### What moved (and how the desktop calls it)

- `image_job.ImageJobMixin`: `_process_image_file` (23705-24851, incl. the nested
  `ImageProgressManager` and `ImageClientWrapper`) and `_run_generative_prompt_mode`
  (23578-23703), verbatim except the two edits below.
- `rpgmaker_job.RpgMakerJobMixin`: `_process_rpgmaker_game` (24853-25279), verbatim except
  `game_dir = rpgmaker_game_dir(exe_path)` (the `.exe`'s folder exactly as before; a folder path
  is used as is).
- `translation_pipeline.TranslationPipelineMixin(GlossaryPipelineMixin, ImageJobMixin,
  RpgMakerJobMixin)`: the U3 placeholders (`PipelineHooksMixin._process_image_file` /
  `_process_rpgmaker_game` / `_run_generative_prompt_mode`, `U7_PLACEHOLDERS`) are deleted, so
  TranslatorGUI and HeadlessOwner both reach the real runners through the pipeline mixin (their
  base lists are unchanged; MRO: ... `JobHooksMixin`, `ImageJobMixin`, `RpgMakerJobMixin`,
  `SettingsPersistenceMixin` ...). The three bodies left TranslatorGUI.
- `direct_text_store` module functions the Direct Text dialog now calls (verbatim lines):
  `configured_glossary_override_mode` (`__init__`'s read of `direct_text_glossary_override_mode`),
  `glossary_override_config_updates` (`_on_glossary_override_toggled`'s three writes, applied with
  `config.update` in the same key order), `chat_rename_title` (`_rename_chat`),
  `manual_glossary_source_record` / `sniff_manual_glossary_extension` (the "Provide Manual
  Glossary" dialog's `_accept`; the info box for empty contents stays in the dialog) and
  `MANUAL_GLOSSARY_EXTENSIONS` (its `allowed_extensions`). The mobile mirrors in
  `ui/chat/direct_text_rules.py` can now call these (mobile-media-modes deletes the mirrors).
- `authgem_auth.authgem_project_items(billed, unbilled, unknown)` and
  `choose_authgem_project_index(saved, billed, unbilled, unknown)`: the GCP project picker's list
  (✅ billed, ❔ unknown, ⚠️ unbilled) and the selection rule of `_authgem_projects_loaded` (keep the
  saved project unless known unbilled, else the first billed, else the first unknown). The slot
  fills the combo from them; `combo.findData(saved)` became the list index of the same items
  (str data: identical; tests/test_mobile_auth_splits.py fuzzes 600 states against the parent).

### Behaviour deltas (intentional, parity-neutral on desktop)

- `run_translation_direct` (U7 edit): `elif ext == '.exe' or file_path in
  (getattr(self, RPGMAKER_GAME_INPUTS_ATTR, None) or ()):`. Only `_register_rpgmaker_game_input`
  (mobile) fills `_rpgmaker_game_inputs`; on the desktop a folder input still logs "Unsupported
  file type" (tests/test_rpgmaker_job.py::test_unregistered_folder_input_stays_unsupported).
- `_run_generative_prompt_mode` reads the prompt editor through `_generative_prompt_source()`:
  the moved lines unless `_generative_prompt_override` (`image_job.GENERATIVE_PROMPT_ATTR`) holds
  non-blank text. Desktop never sets it. Recorded in UI_SPEC §2.6 / Appendix C (the plan said a
  `run_env` hook; the runner lives in image_job, so the hook does too).
- `_run_generative_prompt_mode`'s `Generated_Media` is `mobile_runtime.data_dir(<module folder>)`:
  unchanged on desktop (no `GLOSSARION_DATA_DIR`); `__file__` is image_job.py, the same folder as
  translator_gui.py in source and frozen builds.

### Mobile entry points (GUI-free, no desktop change)

- Generate from prompt (UI_SPEC §2.6): set the output mode (`DirectTextRunOptions.output_mode`),
  `owner._generative_prompt_override = <composer text>`, then
  `_prepare_translation_run([image_job.GENERATIVE_MODE_SENTINEL])` + `_translation_worker`. The
  sentinel is needed for Audio: `_is_generative_output_mode()` only looks at the image/video
  toggles, so an empty selection in audio mode stops at "Please select file(s)" (desktop too).
- RPG Maker (UI_SPEC §4.10): `game_dir = owner._register_rpgmaker_game_input(source, work_dir)`
  (`prepare_rpgmaker_game`: a writable folder is used in place, a read-only one is copied once into
  `<work_dir>/folders/<name>-<hash of its real path>/<name>`, a ZIP is extracted once into
  `<work_dir>/zips/<zip name>-<hash of its member names, CRC-32s and sizes>` with a zip-slip guard;
  each records its source fingerprint in `.glossarion_rpg_source.json` (written last) and is reused
  only for that source (U7 review: two different games both named `game.zip` shared one extraction),
  rebuilt with `fresh=True` or when the marker is missing; calls for one folder are serialised
  (a scan and a job may prepare the same ZIP at once); the game root is the shallowest
  folder `rpgmaker_handler.detect_version` recognises, so a web/Android `www` deployment without
  `Game.exe` works), then the translate pair on `[game_dir]`. The result equals the desktop `.exe`
  dispatch on the same game (tests/test_rpgmaker_job.py::test_headless_owner_folder_entry_matches_the_exe_path).
- The progress manager of `_process_image_file` lives on the owner; a mobile job owner is new per
  job, so each job starts a fresh `ImageProgressManager` (desktop: see bug 1).

### Desktop bugs found (recorded, not fixed)

1. **Image progress follows the first image of the session.** `_process_image_file` creates
   `self.image_progress_manager` only `if not hasattr(self, 'image_progress_manager')` and nothing
   resets it, so every later image run of the same desktop session writes its
   `translation_progress.json` entries into the FIRST image's output folder (and checks resume
   state there).
2. **Generated video/audio from a prompt are saved as text.** `_run_generative_prompt_mode` only
   recognises `[GENERATED_IMAGE:...]`; the client returns `[GENERATED_VIDEO:...]` /
   `[GENERATED_AUDIO:...]` for video and speech, so those runs save
   `generated_<model>_<ts>.txt` holding the marker and never log "Media saved to". Likewise
   `_process_image_file` wraps a `[GENERATED_VIDEO:...]` / `[GENERATED_AUDIO:...]` response in the
   translated-page HTML instead of moving the media file into the output folder.
3. **Generative text results land in a temporary folder in one-file builds.** `Generated_Media` is
   next to `__file__`; in a PyInstaller one-file build that is the `_MEIPASS` extraction folder,
   deleted when the app exits.
4. **Docstring vs code.** `_run_generative_prompt_mode`'s docstring says the prompt comes from
   `translation_chunk_prompt` first; the code prefers the prompt editor / `system_prompt`, then a
   non-template `image_chunk_prompt`, then a non-template `translation_chunk_prompt`.
5. **"Image in skipped folder".** `check_image_status` skips any image whose name already exists in
   `<output>/images/`, which is also where covers and skipped images are copied, so such a name
   is never translated again in that folder.
6. **Title translation needs three "no"s.** `skip_img_title = getattr(self,
   'skip_image_title_translation_var', True) or config.get('skip_image_title_translation', True)
   or SKIP_IMAGE_TITLE_TRANSLATION == '1'`: with the config default True the title is translated
   only when the variable AND the config key are False.
7. **Early progress errors are silent.** `ImageProgressManager._init_or_load` logs a corrupt
   progress file only `if hasattr(self, 'append_log')`, but `append_log` is attached after the
   constructor ran, so the first load's "Creating new progress file due to error" never shows.
8. **RPG Maker: Vision mode translates images only.** `image_mode = ENABLE_IMAGE_TRANSLATION ==
   '1' or enable_image_translation_var`; Vision (and Video) output modes also set
   `enable_image_translation`, so a game run in Vision mode skips the text and runs the image
   pipeline.
9. **RPG Maker: dict-shaped image profile.** The image prompt is `prompt_profiles
   ["RPGMaker_GTool_Image"]` as is; a profile saved in the new `{"prompt": ...}` shape is handed to
   `translate_game_images` as a dict.
10. **RPG Maker: a stopped run still patches the game.** After Stop the runner still writes the map
    and calls `apply_translations` whenever any progress exists, so a partly translated game is
    patched (resumable: the next run restores the originals first).

### U7 parity harness additions

- `tests/parity/runner_parity.py` (CI-capable, no frozen oracle, no PySide6): the legacy method
  text at 41814faa (`git show`, compiled in a class body so multi-line strings keep their
  indentation) and the new mixin method run the same scenario on a minimal owner in two
  equal-length sandboxes with recording backends (`UnifiedClient` constructor / key pools /
  `send` / `send_image` with the env each call saw, `TransateKRtoEN.send_with_interrupt`,
  `rpgmaker_handler.translate_game_images`; real `rpgmaker_handler` extraction / apply with the
  character token estimate; fixed `time.time` / `time.strftime` / `datetime.now`); logs (traceback
  frames masked), calls, env delta, owner state and the whole tree (progress JSON, HTML, payloads,
  media, patched game data) must be equal. tests/test_image_job.py: 48 image + 14 generative
  scenarios; tests/test_rpgmaker_job.py: 20 game scenarios.
- Tier T (`trace_scenarios`): `image_vision_translate`, `image_batch_combined_folder`,
  `image_output_generated`, `video_input_generated`, `audio_mode_image_input`,
  `generative_image_prompt`, `generative_video_prompt`, `generative_audio_prompt`,
  `rpgmaker_exe_text`, `rpgmaker_exe_image_mode` (full desktop record and mobile projection both
  MATCH the frozen desktop). `trace_harness`: client `send` / `send_image`,
  `send_with_interrupt` (the real one waits on a thread with the frozen clock) and
  `translate_game_images` stubs; `time.strftime` follows the trace clock (payload names and
  timestamps); tiktoken off for `rpgmaker_handler`; `image_job` is a FILE_MODULE; the mobile
  driver passes the generative sentinel unchanged.
- `freeze_legacy.SHARED_HELPER_MODULES` += `image_job`, `rpgmaker_job` (oracles frozen at the U7
  commit or later freeze them whole, like `job_runner`); `moved_functions.SHARED_MODULES` +=
  both; the three placeholder names left `HOOK_NAMES`.
- tests/test_direct_text_core.py: `U7_DIALOG_EDITS` (the dialog rewiring is exactly these edits)
  and the rule functions replayed against the removed lines; tests/test_mobile_auth_splits.py:
  the project picker slot vs its parent (600 random billing results).

## U7 Async batch core (async_batch_core; async_api_processor's dialog rewired)

Frozen source: async_api_processor.py at 41814faa (the U6 commit; "AAP n" below).
tests/test_async_batch_core.py pins the move: verbatim AST tier, per-provider request and
submit goldens with mocked HTTP and a fake google-genai SDK (equal to the frozen code's), the job
file round trip (bytes equal to the frozen processor's), and an offscreen session of the frozen
dialog vs the rewired dialog (17 scripted steps over OpenAI / Anthropic / Gemini / Mistral / Groq:
estimate, submit, status, retrieve, retrieve again with "Create new" and with "Overwrite", cancel,
delete, clear, unsupported model, AuthGPT without an sk- key, no file) that HeadlessAsyncBatch then
reproduces (job files, provider and SDK calls, output workspaces).

### What moved
- AAP 48-107 (antigravity clamp fallback, optional tiktoken / txt_processor / google.generativeai /
  anthropic / openai imports) and AAP 110-1454 (`AsyncAPIStatus`, `AsyncJobInfo`,
  `AsyncAPIProcessor`) byte for byte. The only change: `AsyncAPIProcessor(gui_instance,
  jobs_file=None)`; `jobs_file or <the old default>` keeps `async_jobs.json` next to the module.
  The core keeps the logger name `async_api_processor` (the desktop log format prints `%(name)s`).
- 30 `AsyncProcessingDialog` methods (the workflow: prepare env, extract chapters, build messages,
  submit per provider, poll, status / retrieve / cancel / delete / clear handlers, estimate,
  `_handle_completed_job`, the OPF spine map, the API key lookup) moved to `AsyncBatchJobMixin`.
  Their bodies are the dialog's with 13 mechanical substitutions (QMessageBox / QTimer /
  QApplication calls and the five widget reads become `_async_*` hooks, `QMessageBox.Yes/No/Cancel`
  become `_MB_*`); the table is in the mixin docstring and `tests/test_async_batch_core.py`.
  The dialog defines every hook with the original Qt statement (`_MB_*` are properties returning
  the Qt enums), so the desktop runs the same calls in the same order.
- Dialog view logic shared with the mobile list: `job_display_row` (`_refresh_jobs_list`'s row
  texts), `selected_job_progress` (`_update_selected_job_progress`), `gui_model_name` /
  `async_support_status` (the "Current Model" row of `_create_info_section` /
  `_refresh_model_info`), `refresh_pending_job_statuses` (the `_start_auto_refresh` closure).
- async_api_processor re-exports every module-level name it had (the core classes are the same
  objects); its PySide6 import is guarded, so `import async_api_processor` works without Qt.

### Behaviour deltas (intentional, parity-neutral)
- Tracebacks that `_handle_completed_job` / the worker log name async_batch_core.py frames.
- The `__file__`-relative defaults (job list, `GLOSSARY_SHARED_DIR` fallback, the default output
  folder of `_handle_completed_job`) are now relative to async_batch_core.py: the same folder in
  source runs and in every PyInstaller spec (both are flat `src` modules). The frozen exe still
  switches the dialog's job list to the exe folder (dialog `__init__`, unchanged).
- Seven source-scan tests that read async_api_processor.py for env / prompt strings now read
  async_batch_core.py (test_epub_utils, test_glossary_match_toggle_wiring, test_translation_artifacts,
  test_unified_glossary_gender_exclusion, test_unified_glossary_wiring, test_sdlxliff_support;
  test_whole_term_scope checks both files).

### Desktop bugs found (recorded, not fixed)
1. **Estimate Cost Only never counts EPUB chapters.** The EPUB branch of `_estimate_cost` lost its
   sampling loop header (`env_vars`, `chapter_text` and `i` are undefined). With the output token
   limit enabled the NameError lands in "Failed to analyze EPUB" and the 50 x 15000-token fallback;
   with it disabled no chapter is counted ("Average content tokens per chapter: 0", $0.00). The
   automatic estimate after each submission writes that value into the job's cost column.
2. **Chunked chapters are never split.** `_extract_chapters_for_async` calls `ChapterSplitter`, which
   the module never imports: the NameError is logged as "Chunk splitting failed" and the chapter is
   kept whole with `needs_chunking=True`, so the worker skips it.
3. **Chapter-number regexes never match.** `r'chapter\\s*(\\d+)'` (AAP 3663, 4404) is a raw string with
   doubled backslashes (literal backslashes); `_extract_chapter_number` looks for `chapter_N` while
   the custom ids are `NNNN_<file stem>`, so it returns 0 (results are then ordered by spine only).
4. **OpenAI model mapping by prefix.** `_prepare_openai_batch` walks its table in insertion order and
   maps `gpt-4o-mini` to `gpt-4o` (a pricier model; also `gpt-4.1-mini` / `-nano` to `gpt-4.1`). The
   request golden pins it.
5. **Groq / Mistral jobs.** Groq batches are uploaded to OpenAI's Files / Batches endpoints with the
   Groq key; `check_job_status` / `retrieve_results` have no Mistral or Groq branch ("Unknown
   provider"), so those jobs stay pending; the Mistral request body is not the Mistral batch format.
6. **Gemini results cannot be retrieved.** `_retrieve_gemini_results` compares the SDK's state enum
   with the string `'JOB_STATE_SUCCEEDED'`, which is always unequal: "Batch job not completed".
7. **"Wait for completion" polls once.** `_start_polling` runs on the submission thread; its
   `QTimer.singleShot(interval, poll)` is created on a thread without an event loop and never fires.
   HeadlessAsyncBatch polls until the job ends (the intended behaviour), stoppable (see below).
8. **Cancel.** Anthropic jobs are marked cancelled locally without an API call; the
   `_cancel_openai/anthropic/gemini/mistral/groq_job` helpers are unused; `submit_batch` (async) calls
   coroutines that do not exist (dead code).
9. **Frozen exe job list.** The dialog loads `async_jobs.json` from the module folder (`_internal`)
   and again from the exe folder and keeps both sets.

### GUI-free semantics (HeadlessAsyncBatch, Glossarion Mobile)
- Questions (`QMessageBox.question` / a warning with buttons) go to
  `host.ask('async_batch_question', level=, title=, text=, buttons=['yes','no'(,'cancel')],
  default=)`; `answers={title: answer}` presets win (the mobile screen asks its own confirmation
  first); without either the answer is "no" ("cancel" never by default), so nothing destructive
  happens unasked. Notices are recorded in `messages` and emitted as `async_batch_message`; the cost
  label as `async_batch_cost`; the job tree as `async_batch_jobs` rows (`job_display_row`).
- `QTimer.singleShot(0, ...)` runs at once on the calling thread; a positive delay (polling) waits on
  that thread and is queued, so polling never deepens the stack, and stops when the host's stop
  latch is set ("⏹️ Async polling stopped"). The poll interval is `async_poll_interval` clamped to the
  spin box's 10-600 s; "Wait for completion" is `async_wait_for_completion`.
- The job list defaults to `<GLOSSARION_DATA_DIR>/async_jobs.json` (`default_jobs_file`; the module
  folder on desktop). `retrieve()` returns the output folders `_handle_completed_job` reports.

## U7 Review run orchestration (review_generator; review_dialog rewired)

Frozen source: review_dialog.py at 41814faa. tests/test_review_run_core.py runs the frozen and the
rewired dialog offscreen on 24 random GUI states per Start path (80 checked once) with recording
generators and compares every generator argument, the dialog log, the main-log replay, the
exported `ENABLE_STREAMING` and the restored `sys.stdout`; `run_review_session` (mobile) makes the
dialog's call for the same states.

### What moved (review_dialog keeps the widgets, boxes, queue, log redirection and poll timer)
- The parameter gathering of both Start paths (API key field, `model_var`, `ENDPOINT` / endpoint,
  temperature, the config copy with the live multi-key flag, input token limit, `{target_lang}` in
  both prompts) -> `review_run_params`; the four stop-flag resets -> `reset_review_stop_flags`; the
  streaming toggle -> `apply_review_streaming_env`; the `[Review]` log closure -> `review_log_fn`; the
  stdout redirect class -> `ReviewStdoutWriter`; the generate calls -> `run_review`; Generate All's
  worker (sequential / parallel, per-input output folder, nav / all_done messages) ->
  `run_all_reviews` + `review_all_output_dir`; the batch-size rules -> `single_review_batch_size` /
  `review_all_batch_size`; `_output_dir_for_file` / `_get_review_paths` / `_get_app_dir` ->
  `review_output_dir_for_file` / `review_paths_for` / `review_app_dir` (review_dialog keeps the old
  names as wrappers / alias).
- `run_review_session` is Start Review without Qt in the dialog's order (mobile Review job).

### Behaviour deltas (parity-neutral)
- `review_app_dir` resolves from review_generator.py (same folder as review_dialog.py).
- The final review prompt is read when the parameters are gathered (before the UI changes) instead of
  just before the thread starts; nothing changes `_final_review_prompt` in between.

### Desktop bugs found (recorded, not fixed)
1. Start Review's output folder expands `~` in the output override (`_output_dir_for_file`), Generate
   All's per-input folder does not: a `~/...` output directory yields two different folders.
2. Start Review passes the batch size only in chunk mode with batch translation on; Generate All also
   hands its parallel-review count to every chunked review as its chunk worker count.

## U7 Output-folder tools, QA bulk scan, Translate Headers Now, metadata field rules

Frozen sources at 41814faa: other_settings.py (OS), QA_Scanner_GUI.py (QG), translator_gui.py (TG),
translate_headers_standalone.py (THS), metadata_batch_translator.py (MBT). tests/test_u7_tool_cores.py
runs each frozen handler / closure body against the live module globals next to the rewired code
(40 random sandboxes each) and compares logs, message boxes (title, text, buttons), scanner calls,
the files left on disk and translation_progress.json.

### What moved (the desktop handlers keep their selection logic and dialogs, and call these)
- QG `run_qa_scan`'s `run_scan` worker body -> `qa_scan_runtime.run_bulk_qa_scan` (returns
  `(successful, failed)`; `on_report` replaces `self.last_qa_report_path =`); its
  `_load_current_qa_settings` closure -> `load_current_qa_settings(config)`; the run-start
  cancel-flag reset -> `reset_qa_cancel_flags`.
- The flag blocks of TG `stop_qa_scan` / `_do_qa_force_stop` / `_check_qa_stop_done` ->
  `next_qa_stop_phase`, `apply_qa_graceful_stop_flags`, `apply_qa_force_stop_flags`,
  `clear_qa_stop_flags` (same env / scanner / client flags, pinned against the frozen methods).
  **Open item:** translator_gui is owned by another U7 workstream; its three methods still carry
  their inline copies until that owner (or Integrate) replaces the flag blocks with these calls.
- OS "Delete Header Files" / "Delete TOC.txt" (two near-identical functions) -> one desktop helper
  `_delete_translation_cache_files(self, kind, error_label)` over `output_tools_core`
  (`plan_artifact_cache_delete`, `ArtifactDeletePlan.summary_text / question_text`,
  `delete_planned_artifact_caches`, `artifact_delete_result`, `ARTIFACT_DELETE_TEXTS`).
- OS `validate_epub_structure_gui`'s loop -> `validate_epub_outputs`; the "Load Font…" copy loop ->
  `import_custom_fonts`.
- THS `run_translate_headers_gui` body -> `translate_headers_now(gui, *, show_error,
  process_events, output_dir_for)` (returns `(successful, failed)`); the wrapper passes the message
  box and the Qt event pump. OS `run_standalone_translate_headers`' worker body (API client and
  multi-key set-up, the header run, the EPUB rebuild, client restore) ->
  `run_translate_headers_now(gui, model, api_key, *, headers_runner, rebuild_epub=True)`.
- MBT `configure_metadata_fields`' tables and rules -> module level: `METADATA_STANDARD_FIELDS`,
  `METADATA_DEFAULT_ENABLED_FIELDS`, `saved_metadata_fields_for_epub`, `metadata_field_checked`,
  `store_metadata_field_selection`, `final_metadata_fields_config`.

### Behaviour deltas (parity-neutral)
- Error tracebacks of the QA worker / header worker gain one frame (the shared function).
- `run_qa_scan` no longer imports `os` / `unified_api_client` locally (the module-level `os` is the
  same object; `unified_api_client` is imported where it is used).
- `METADATA_DEFAULT_ENABLED_FIELDS` is a frozenset (the dialog only tests membership).

### Mobile (these adapters now run the shared code; the rebuilt copies are gone)
- QA scan job: `run_bulk_qa_scan` with the Library sources as the "selected EPUBs" keyed by folder
  name (log: "✅ Matched from selected files"), the desktop file search next to / inside the folder
  for books without a known source, PDF source auto-detection, the desktop QA Stop escalation and
  the cleanup after a stopped scan.
- Translate Headers job: `run_translate_headers_now` with `translate_headers_now` as the runner
  (Library folders through `output_dir_for`, errors logged): an existing `translated_headers.txt` is
  now reconciled / repaired / re-applied like the desktop, and PDF workspaces are handled. Recorded
  divergence: the EPUB rebuild stays on the mobile compile path (`_run_epub_compile`, which lists
  the EPUB as a job output) instead of `fallback_compile_epub` on a folder found by name.
- Validate EPUB: `validate_epub_outputs` with each folder as its own book (`<folder>/<name>.epub`).
- Converter Load Font, Headers screen cache deletion and metadata field rules delegate to the cores.

### Desktop bugs found (recorded, not fixed)
1. The output-folder search is copied five times (delete headers, delete TOC, validate, the
   header run, the header rebuild) with small differences: only THS checks `<name>_PDF` folders,
   validate does not log the override, the rebuild ignores PDFs.
2. THS logs two mojibake lines for PDFs ("âœ… PDF headers complete", "âŒ PDF header translation
   failed": UTF-8 emoji saved as cp1252 text).
3. A stopped QA bulk scan still ends with "✅ QA scan completed successfully." / "✅ Bulk QA scan
   completed.".

## U7 Retranslate Selected, Resolve QA, image-folder view, manual glossary refinement and the SDLXLIFF reviewer core (progress_actions, progress_core, glossary_progress_core, sdlxliff_review_core; Retranslation_GUI rewired)

Moved at BASE_SHA 41814faa (the U6 commit; the Progress Manager oracle was re-frozen there with
`python tests/parity/progress_legacy.py --freeze --sha HEAD` into
`tests/parity/legacy_progress/41814faa95e2/`). Line numbers refer to
`git show 41814faa:src/Retranslation_GUI.py` (RG). Pinned by tests/test_retranslate_plan_apply.py and
tests/test_sdlxliff_review_core.py (`*_verbatim` plus file-system goldens against the frozen dialogs).

### What moved / split (and how the desktop calls it)

- **Retranslate Selected** (the `retranslate_selected` generator of `_add_retranslation_buttons_opf`,
  RG 21913-22959) is split into `progress_actions.plan_retranslation(book, rows, settings)` (selection
  normalisation, the metadata / artifact / mixed-audio guards and every confirmation text incl. the
  RECYCLED TOC/header pair, RG 21925-22253 verbatim blocks), `apply_retranslation(book, plan,
  linked_choice, sidecar_workers)` (RG 22273-22850 verbatim; `sidecar_workers` overrides the owner's
  extraction-worker count when given) and `retranslation_result_message(result)` (RG 22869-22959, the
  three dialogs as `(kind, title, message)`). `book` is the desktop data dict (+ `owner=`) or a
  `progress_core.BookProgress`; `rows` are list indices, `RowPresentation`s or display dicts. The
  desktop generator is now plan -> its dialogs -> `yield "run_background"` -> apply on the worker ->
  `yield "apply_ui"` -> refresh + message; the audio-mode TTS reset still runs `reset_tts` on the Qt
  thread. The progress write is unchanged (`_merge_and_write_retranslation_progress` with the
  authoritative chunk resets).
- **Resolve QA issue (raw foreign text)**: the preflight of `_start_single_progress_qa_resolution`
  (RG 16651-16713: target, refusals, `_single_qa_resolution_request` / `selected_files` /
  `current_file_index` / `_metadata_only_run` / `_single_chapter_filter` / `_force_stream_all`) is
  `progress_actions.prepare_single_qa_resolution(owner, data, info)`; the desktop method keeps the
  messages, `entry_epub`, the log line and `run_translation_thread`.
- **Image-folder Progress Manager**: output lookup (RG 26379-26415), the refresh scan (25240-25382),
  row text (25400-25423), Mark as Skipped (26718-26801), Delete Selected confirmation / deletes
  (26839-26850, 26857-26881) are `progress_core.image_folder_output_dir`, `scan_image_folder`,
  `image_folder_row_text`, `mark_image_folder_items_skipped`, `image_folder_delete_confirmation`,
  `delete_image_folder_items` (+ `image_folder_not_found_message`, `image_folder_mark_skipped_message`,
  `build_image_folder_progress` for mobile). The dialog keeps its list, selection, confirmations and
  timers. The initial list built at open (RG 26427-26600) stays in the dialog.
- **Manual glossary refinement**: the plan step of the Glossary Progress confirm closure (RG
  18239-18324) is `glossary_progress_core.prepare_manual_glossary_refinement` (returns a
  `ManualRefinementPreview`; refusals as `(kind, title, message)`), its post-dialog step (RG
  19046-19070) `finish_manual_glossary_refinement`, and `_run_manual_glossary_refinement` (RG
  15100-15196) `run_manual_glossary_refinement(owner, ...)`; the desktop closure keeps the busy / model
  checks and the preview dialog, the desktop method is a one-call wrapper.
  `plan_manual_glossary_refinement` (mobile) stands in for the dialog.
- **SDLXLIFF reviewer**: `sdlxliff_review_core` holds the module helpers (RG 342-668, 714-745 incl.
  `_get_app_dir`), `SdlxliffAutogenMixin` (RetranslationMixin's sidecar auto-generation: RG
  14979-14990 and 22 methods of 15199-16254) and `SdlxliffReviewCoreMixin` (the dialog constants RG
  846-925 / 9378-9397 and 214 widget-free dialog methods, verbatim except class-qualified calls now
  naming the mixin and three hook replacements: the Notepad page reload after a save
  (`_refresh_notepad_page_after_save`) and the row-editor insert of inject / Undo all edits
  (`_insert_into_review_editor`)). Split helpers the dialog now calls: `_init_review_state` (RG 939-958),
  `_translate_tooltip_work` (RG 10145-10157 / 10220-10232), `_store_tooltip_translations` (RG
  10316-10341), `_tooltip_translation_result_message` (RG 10361-10374), `_piece_header_text` (RG
  7385-7395), `_apply_machine_translation_threshold` (RG 4296-4301). `SDLXLIFFReviewDialog(
  SdlxliffReviewCoreMixin, QDialog)` overrides every GUI hook with its widget code;
  `RetranslationMixin(ProgressViewMixin, SdlxliffAutogenMixin)`; Retranslation_GUI re-exports the module
  helpers. The Notepad WebEngine JS, page cache, menus, prompts and threads stay in the dialog.

### Writes switched to `mutate_progress` (image-folder view)

Mark as Skipped (RG 26806) and Delete Selected (RG 26886) wrote the view's whole snapshot
(`json.dump(progress_data_current)`); they now remove the collected hash keys from the newest file
through `progress_core.mutate_progress`. Consequences: a translator save made after the view's last
refresh survives; nothing is written when nothing changed (Mark as Skipped used to rewrite the file
whenever one was loaded -- always for a v2.1 file, where no hash key ever matches); a progress file
deleted since the refresh is no longer recreated from the snapshot; Delete Selected's
"Removed <hash> from progress_data[...]" prints now follow the per-file "Deleted:" prints. The pre-2.1
shape bug (U5 desktop bug 3) is kept: against a v2.1 `chapters` file no hash key is found, so no
entry is removed (test_retranslate_plan_apply.py::test_image_folder_progress_hash_removal_follows_the_layout).

### Behaviour deltas (intentional, parity-neutral on the fixtures)

- Machine Translation preview: both worker closures call `_translate_tooltip_work`; when parsing /
  validating a response raises, the worker now reports no translations (before: the partially
  processed ones) -- the error is reported either way.
- `_apply_tooltip_translations` computes its status text through `_tooltip_translation_result_message`
  before the `try` that sets it (same text).
- Retranslate Selected reads the Manual editing checkbox before the guards (it read it after them).
- Moved code that named its class (`SDLXLIFFReviewDialog._normalize_review_text`,
  `RetranslationMixin._sdlxliff_is_extracted_epub_dir`, ...) now names the mixin: patching those names
  on the desktop classes, or patching moved module helpers on Retranslation_GUI, no longer reaches the
  moved callers (test_sdlxliff_support's manifest-flush test patches sdlxliff_review_core now).
- `Retranslation_GUI._get_app_dir` is the moved function (same code; its `__file__` is now
  sdlxliff_review_core in the same folder, so the result is identical).

### Desktop bugs found (recorded, not fixed)

1. The SDLXLIFF reviewer's progress writes (Mark as Completed / Undo, the manual-editing pending seed:
   `_write_review_progress_data`) are unlocked read-modify-writes of `translation_progress.json`, not
   `mutate_progress`: a translator save between the read and the replace is lost. Kept verbatim.
2. Image-folder view: the list built at open (RG 26427-26600, flat layout only, root images included)
   differs from the refresh that replaces it 0 ms later (RG 25240-25382: nested `images` or flat
   layout, root images only when tracked). Mobile shows the refresh's list.
3. Opening a Progress Manager prints "Failed to load icon: cannot access local variable 'sys'": an
   `import sys` inside one branch of `_force_retranslation_epub_or_text` makes `sys` local, so the
   dialog icon is never set when that branch is not taken.

### GUI-free semantics for mobile

- `plan_retranslation` / `apply_retranslation` on `progress_core.build_book_progress`: settings
  `{'manual_editing': owner._get_retranslation_manual_editing_state()}`; a RECYCLED single-artifact
  selection needs `linked_choice` (`plan.linked_choice_labels`); `retranslate_rows` plans and applies
  without confirmation and refuses a needed-but-missing choice. Audio mode plans `reset_tts`.
- `prepare_single_qa_resolution` sets the run state on the job's HeadlessOwner; the caller then
  starts the translation run (the pipeline turns the request into a Partial.b run).
- `SdlxliffReviewSession` / `open_sdlxliff_review`: `save_status_label` and `_edit_save_timer` are plain
  stand-ins (edits are saved by `flush_edits`); `refresh` reloads every piece when sidecars changed
  (the dialog rebuilds only changed pages); `_displayed_piece_row` is the selected piece; the Machine
  Translation preview runs on the caller's thread; provider credentials are stored with
  `set_machine_translation_credentials` (keys encrypted like the desktop's) and a provider without
  them is refused with the desktop's message instead of a prompt; the default auto-generation owner
  is `SdlxliffAutogenOwner(config)`; without a `context_parent` reviewer settings stay in `config`
  (the dialog without a parent writes `<app dir>/config.json`).
- `plan_manual_glossary_refinement`: the requested types / target chunk count are the preview's
  answers; the model is `owner.model_var`; "Nothing to Refine" returns None.
- `build_image_folder_progress`: the refresh's list and its "No translated files found" check.

### Parity harness additions

- tests/test_retranslate_plan_apply.py: verbatim tiers; 35 Retranslate Selected file-system goldens
  through the real offscreen dialogs (frozen generator vs working tree, scripted Yes / No / RECYCLED
  answers: chapters, merged children, refinement, manual editing with SDLXLIFF sidecars and Machine
  Translation previews, chunk segments / every chunk / parent absorbing children / segment not found,
  metadata phases, RECYCLED pairs, toggles off, audio TTS reset, >10 rows, PDF sections with compiled
  HTML/PDF and API chunks, subtitles, plain text); mobile plan/apply vs the desktop trees; Resolve QA
  preflight; image-folder dialogs (flat / nested / v2.1 / no progress x Mark as Skipped / Delete
  Selected); manual refinement plan vs the frozen closure and the runner vs the frozen method.
- tests/test_sdlxliff_review_core.py: verbatim tiers; reviewer goldens (open with sidecar
  regeneration, row / Notepad / Manual-editing edits, Mark as Completed / Undo, Machine Translation
  preview + Flag inaccurate + inject with a scripted translator, threshold / provider settings) on the
  frozen dialog, the working-tree dialog and the mobile session; import hygiene.
- `tests/_src_corpus.PROGRESS_MANAGER_MODULES` includes sdlxliff_review_core; retargeted source checks
  in test_sdlxliff_support (+ its manifest-flush monkeypatch), test_progress_model_metadata,
  test_progress_actions and test_glossary_refinement_status_display.

## U7 Integrate (wiring, packaging, desktop QA Stop rewiring, offline E2E)

The parity oracles stay frozen at 41814faa (the U6 commit; HEAD 5befbc80 differs from it only in
`src/mobile/pyproject.toml` and the mobile E2E). All parity tiers pass on the integrated tree (494
tests; the two skips are the GLOSSARION_PY310 probe, run separately with the 3.10 venv and passing,
and the documented desktop-only trace race). The full desktop suite (167 files, per file, isolated
Library) has no failure outside BASELINE_FAILURES.txt.

### Desktop changes made by the integration (each pinned)

- **QA Stop handlers call the shared stop helpers.** translator_gui's `stop_qa_scan` decides its
  branch with `qa_scan_runtime.next_qa_stop_phase(current_phase, graceful_stop_enabled)` and sets
  the graceful flags with `apply_qa_graceful_stop_flags()`; `_do_qa_force_stop` calls
  `apply_qa_force_stop_flags()`; `_check_qa_stop_done._delayed_flag_cleanup` keeps its "only if no
  new scan started" check and calls `clear_qa_stop_flags()`. The helpers hold the removed statements
  in their order (stop_scan, GRACEFUL_STOP, TRANSLATION_CANCELLED, the client's fast-path flags);
  button text / style, timers, logs and the background cleanup thread are unchanged. The mobile
  `qa_scan` job already used the helpers, so the desktop and mobile Stop share one code path.
  Pinned by tests/test_u7_tool_cores.py (the frozen 41814faa methods, the shared helpers and the
  rewired methods side by side: 1-3 clicks with graceful stop on and off, then the delayed cleanup
  with and without a new scan in between; env flags, client flags, stop_scan calls, phase and
  `stop_requested` compared) and by tests/test_image_job.py `TG_U7_SPANS` (six new spans).
- **Review generator Delete / Restore.** The file moves of `ReviewDialog._on_delete` (every existing
  review file to `<its folder>/backups/review_<timestamp>.md`) and `_on_restore` (newest `.md` backup
  of the primary review's `backups` folder over every review path, then the restored text) and
  `_get_backups_dir` moved into review_generator (`move_review_to_backups`, `review_backups`,
  `copy_review_backup`, `review_backups_dir`, `review_restore_question` for the overwrite box text);
  the dialog keeps its button feedback, the question box and the UI reload. The mobile Review
  screen's Delete / Restore (no longer disabled) call them plus `latest_review_backup` /
  `restore_review_backup`. Pinned by tests/test_review_run_core.py (the frozen 41814faa handlers and
  the rewired ones on five layouts: single review, volume reviews, no review, backup only, restore
  declined; files, widget calls, the question text and the restored text compared).
- **Settings schema generator** (src/mobile/tools/schema_extract.py). `image_job.py` and
  `rpgmaker_job.py` are owner modules (TranslatorGUI mixins); `sdlxliff_review_core.py` is a dialog
  module (UI site `progress`); `qa_scan_runtime.py` and `translate_headers_standalone.py` are scanned
  for the moved functions only (new `DIALOG_MODULE_FUNCTIONS`: `run_bulk_qa_scan` /
  `load_current_qa_settings` at QA_Scanner_GUI's `*` distance, with its nested-name hints;
  `run_translate_headers_now` -> `other.meta_data`); the Direct Text rule functions now in
  `direct_text_store` are `direct_text` UI roots, and the new `CONFIG_UPDATE_FUNCS` reads the dict
  literal `glossary_override_config_updates` returns as config writes. With these, the regenerated
  `settings_schema_data.py` equals HEAD except four keys whose records came from the manual glossary
  refinement confirm closure, whose plan step moved into `glossary_progress_core` (a split, not a
  verbatim move):
  - `custom_entry_types`: loses the label "Exact total chunk count:" and the tooltip "This one-run
    override never changes the saved Refinement settings.";
  - `glossary_refinement_chunking_mode`: loses that label and tooltip and its dialog default `'all'`
    (`default_source` dialog -> `env:translation`, origins lose `dialog`);
  - `glossary_refinement_system_prompt`: loses the tooltip;
  - `glossary_refinement_user_prompt`: loses the tooltip; its dialog default `''` becomes
    `{'$expr': 'default_user'}`.
  The label and tooltip belong to the closure's one-run chunk-count spin box, which edits none of
  these keys (label proximity in the old closure); no effective default, type or section changes.
- **Collector baseline.** `translate_headers_standalone->PySide6` allows 2 unguarded sites (was 1):
  `run_translate_headers_gui`'s QMessageBox import and its `processEvents` hook, both inside the
  desktop wrapper (the old in-loop QApplication imports were inside `try`). Mobile runs
  `translate_headers_now` / `run_translate_headers_now`, which import no Qt.
- **Packaging.** image_job, rpgmaker_job, async_batch_core, output_tools_core and
  sdlxliff_review_core are in both blocks of all 14 specs (module-level imports of
  translation_pipeline, async_api_processor, other_settings, Retranslation_GUI and progress_actions,
  so every tier; their src-local imports are all shipped); tests/test_mobile_runtime.py enforces it
  (`U7_SHARED_MODULES`, `U7_IMPORTERS`, no Qt / dialog / manga import). backend_manifest.toml lists
  the five; python-app.yml runs the seven new U7 test files on 3.10 and adds them plus
  review_generator to the PySide6-blocked import check; `moved_functions.SHARED_MODULES` gains
  async_batch_core, output_tools_core and review_generator (3.10 import probe).

### Mobile wiring (no desktop change)

- Job kinds `retranslate`, `resolve_qa`, `async_batch`, `review`, `rpgmaker`, `generate_media` and
  `translate_image` are registered (`job_kinds.KIND_MODULES` and `JobKind`); `SHIPPED_MILESTONES`
  gains U7, so the U7 router entries (Tools hub tiles, `chat.attachments`, `tools.text`) open.
- `ToolsFeature` builds `tools.async` / `tools.review` / `tools.sdlxliff` / `tools.rpgmaker`;
  `JobsFeature` builds `tools.text` and hands the File browser's "Open with › Reader" to the Reader
  feature; a JobStrip tap on a job waiting on an `async_batch_question` opens Tools › Async batch.
- app.py: the drawer's "New scratch chat" opens a scratch chat (`ChatView._on_new_scratch`), like
  the header button and Send as scratch.
- Dependencies: flet-audio / flet-video 1.0.3 (pure `py3-none-any` wheels) pass
  `check_mobile_wheels` for all five targets. `google-cloud-texttospeech` cannot be built for mobile:
  it needs grpcio >= 1.84 (through google-api-core / grpcio-status), which has no android_24 or
  ios_13 cp313 wheel (1.81 is the newest installable; the plan's grpcio trap), and its requests floor
  conflicts with the pinned requests 2.32.5. Google Cloud TTS voices therefore show a ReasonChip in
  the Audio options; the REST fallback stays a U9 tier-B item.

### Self-test and offline E2E additions

- Fake server: a request with an `image_url` part is a `vision` request answered with
  `FAKE_OCR_TEXT`; `POST /v1/images/generations` (the client's Images API route for the Image
  output mode) returns `FAKE_PNG` as `b64_json` and records the prompt; `leave_raw_once[chapter]`
  keeps a Korean passage in that chapter's next single-chapter answer.
- `e2e_retranslate_resolve_qa`: translate (the model leaves one Korean sentence in chapter 5), the
  Chapters tab plans Retranslate for chapters 3 and 7 (`plan_retranslation`, the desktop
  "Confirm Retranslation" copy), the `retranslate` job resets only those files / rows, the next
  translate run sends exactly chapters 3 and 7 (plus the book-metadata request every run sends), a QA
  quick scan flags chapter 5 (`korean_text_found_...`) and the row's Resolve QA runs the Partial.b
  `resolve_qa` job, which sends only chapter 5 and leaves no Korean. Fixture note: the quick scan
  skips the foreign-character check of chapters under 500 words except their headings (which the
  run rewrites from the translated headers) and reports a mostly Korean text as a language
  mismatch, not as raw Korean text, so the leftover is one sentence inside a ~600-word answer.
- `e2e_vision_and_generate`: a chat PNG attachment in the Vision mode sends one vision request and
  the OCR text reaches `response_001_<name>.html` and the chat; Generate from prompt (Image) sends
  the composer text to the Images API and the PNG becomes the response's `Direct Text 1.png`
  (`[GENERATED_IMAGE:...]`, `message_media`).
- The suite has 8 checks and 18 jobs; it passes on the repo `src/` and on the collected bundle.

### Desktop offscreen smoke (integration)

The real `TranslatorGUI` offscreen, HEAD vs the working tree (isolated src copies and sandboxes), on
the workspace the U7 E2E translated: Progress Manager › Retranslate Selected (chapters 3 and 7: the
same confirmation and result boxes, rows, statistics, output tree and progress JSON), the SDLXLIFF
reviewer (same pieces / rows / status; one row edit saved to the same sidecar bytes),
`_process_image_file` on a PNG with `send_image` stubbed (same request, logs, output folder and
progress file) and the Async Processing dialog with "Estimate Cost Only" (same widgets, labels and
boxes). Every section is identical.

### Desktop bug fixed (user-approved 2026-10-06; found by the U6 device E2E)

`scan_html_folder.update_new_format_progress` uses `hashlib` at line 5971 (the artifact branch) but
re-imports it locally at line 6023 (`import hashlib`), which makes `hashlib` a local name of the
whole function: whenever QA flags TOC.txt / translated_headers.txt the scan ends with
`UnboundLocalError: cannot access local variable 'hashlib'`. The scanner never seeds langdetect, so
this hit about one run in ten. Fixed in a separate commit with the owner's approval: the local
import is gone, so the module-level hashlib (line 25) is used; regression test
tests/test_qa_runtime_additions.py::test_update_new_format_progress_hashes_flagged_translation_artifacts. The mobile E2E pins the langdetect seed (5befbc80) so its QA checks are
deterministic.

## U7 review: second round (mobile and test-only, no desktop change)

- **Transcript cards stay live.** Flet 1.0.3 freezes a new control that the diff matches to an old
  one by key, and every re-render rebuilt every chat card under its `ScrollKey`: after any re-render
  the AudioCard could not show playback (and position events raised), Copy ✓, the jump highlight and
  the off-loop extras (a Vision run's OCR section, Refine's Compare with original) failed or were
  swallowed. Each card now sits in a `CardSlot` kept across renders (the slot carries the
  `ScrollKey`; a new card is swapped into its `content`), and a card whose inputs did not change is
  passed again as the same object; the running JobCard stays one object for the whole run. The
  AudioHub drops a card that cannot take an event instead of failing the playback.
- **Unsaved text-editor edits.** The `tools.text` editor's View cannot pop while its text is dirty:
  Back asks Save / Discard / Cancel (shared with the shell's leave guard) and leaves on Save or
  Discard.
- **SDLXLIFF reviewer.** Leaving it saves an edited Notepad document and typed row text (the
  dialog's `closeEvent` captures the Notepad HTML and flushes its queued edits); a re-render saves
  the document into its own piece first. The 2 s poll runs only while the reviewer is on top and the
  app is in the foreground.
- **Chat store.** Duplicate as scratch copies the original's `Chat Messages` body files into the
  scratch chat (Delete message in the original renames or removes them; the copy showed another
  response's text). Plan › Cancel's truncation remaps the sidecar like Delete message, so a
  cancelled Run again / Retranslate turn no longer groups a later send of the same file.
- **Attachments manager.** Migrate / Delete workspace are blocked for a workspace while any active or
  queued job's folder is inside it (the job card's Compile, ＋ › Retranslate chapters), not only while
  the chat's own run is live.
- **RPG Maker.** A scan no longer marks the prepared copy as the translated game: "Share game as ZIP"
  stays disabled after reopening the tool until a run applied the translation.
- **File browser.** Open with › Media viewer shows images, video and audio in the chat's MediaViewer
  (UI_SPEC §4.10).
- **tests/test_sdlxliff_review_core.py (test-only).** `test_mobile_session_api` found chapter 1 at
  piece 0, which holds only when `os.listdir` returns the SDLXLIFF folder sorted (NTFS). Without
  spine positions the shared core keeps the folder's listing order (the desktop rule,
  `Retranslation_GUI` `_load_pieces`), and ext4 lists in hash order, so the test now finds the piece
  by its output name and asserts only the file-name part of the header. Verified with the listing
  reversed. The core is unchanged.

## U8 Manga run env, batch runner, Files model and settings defaults (manga_env, manga_runner, manga_files_core, manga_settings_defaults, google_vision_rest; manga_integration / manga_settings_dialog / manga_translator rewired)

Frozen source: manga_integration.py, manga_settings_dialog.py and manga_translator.py at 9355bb5d
(main with the owner's safe_image upgrade; manga_integration and manga_settings_dialog are unchanged
since 9a46f869, where the move was first measured). The parity oracles (freeze_legacy, goldens,
trace oracle) were re-frozen at 9355bb5d. tests/test_manga_env.py pins the move: every moved method
equals its 9355bb5d text; `_start_translation_heavy` and `__init__` equal it once the split-outs are
inlined back; run-start parity (15 scenarios: custom-api / Google / Azure / Document Intelligence /
Qwen2-VL, key pools, batching, own-auth, aborts, existing translator, inpainting modes) and worker
parity (7 scenarios: sequential, failures, CBZ jobs and "CBZ at end", OUTPUT_DIRECTORY, parallel
panels, stop, model cleanup) run the frozen and the new code with recording stubs (logs, calls, env
delta, config, update queue); a per-image trace runs `MangaTranslator.process_image` of the frozen
module and of the working tree on a fixture page with recording BubbleDetector / OCRManager /
LocalInpainter / UnifiedClient stand-ins (4 scenarios: batched, full-page context, visual context,
skip inpainting) and compares the calls with their arguments, the result and the written pixels;
an offscreen smoke builds the desktop tab and MangaSettingsDialog over a HeadlessOwner.

### What moved
- 115 `MangaTranslationTab` methods, byte for byte, into five GUI-free mixins the tab now inherits
  (listed first, `QObject` last): `manga_files_core.MangaFilesMixin` (Files tab: source roots,
  process groups, image range, skip keys, drop / CBZ extraction, sort, selection persistence, CBZ
  packaging, output paths), `manga_files_core.MangaHooksMixin` (`_update_progress`,
  `_update_current_file`, `_stop_startup_heartbeat`, `_update_manga_preview_image_list_for_range`,
  plus GUI-free defaults of six hooks the tab overrides with its Qt code: `_log`, `_reset_ui_state`,
  `_monitor_translation_output`, `_update_manga_image_range_display`, `_add_manga_file_item`,
  `_rebuild_manga_file_listbox`), `manga_env.MangaEnvMixin` (settings state load / save / apply, the
  font-size presets, default prompts, custom image-edit env, glossary auto-load / paths / backups /
  debug files / glossary env), `manga_env.MangaOcrSessionMixin` (automatic OCR export, imported-OCR
  page map) and `manga_runner.MangaRunMixin` (cancellation flags, preflight, start, worker, stop,
  the manga glossary workflow).
- `_start_translation_heavy` stays in `MangaRunMixin` with three blocks split out verbatim into
  `MangaEnvMixin`: `_reset_manga_graceful_stop_env`, `_prepare_manga_run_env` (thread limits, OCR
  config and credential checks, API key / model / client, key pools, custom-api OCR env,
  `OCR_SYSTEM_PROMPT`, `MANGA_IMAGE_REQUEST_*`) and `_apply_manga_batch_env` (`BATCH_*`). The only
  edits: the six early `return`s of the prepare block became `return None`, and it returns
  `(ocr_config, api_key, model, needs_new_client)`.
- `__init__` blocks: `apply_manga_startup_thread_limits(main_gui)`, `_init_manga_run_state()`,
  `_init_manga_prompt_state()` (called from the same places).
- Module helpers (`_get_app_dir`, `_manga_cmd_debug_*`, `_translation_run_token_matches`,
  `_natural_sort_key`, `_MANGA_SKIP_PREFIX`, `_manga_filename_without_skip_prefix`) moved to
  manga_files_core and the Windows thread-priority block to manga_runner; manga_integration
  re-exports them. (`ImageStateManager` moved to manga_editor_core in the editor-core step.)
- `MangaSettingsDialog.default_settings` is built by `manga_settings_defaults.default_manga_settings()`
  (the literal moved with its comments); src/mobile/tools/schema_extract.py reads the
  `manga_settings.*` defaults there (settings_schema_data.py unchanged).
- New contract functions (no desktop caller): `manga_env.build_manga_run_env` (the env delta of a
  batch start), `apply_rendering_settings`, `build_ocr_config`, `prepare_manga_glossary_env` /
  `restore_manga_glossary_env`, `import_ocr_session`, `font_preset_updates` (the config writes of a
  font-size preset button, computed by running `_set_font_preset`), the default prompt getters;
  `manga_runner.HeadlessMangaRunner` / `run_manga_batch` (the MANGA job: the GUI-free lines of the
  Start click, then the moved start / worker / stop); `manga_settings_defaults.merge_manga_settings`
  / `MANGA_TOP_LEVEL_DEFAULTS`; `headless_owner.MANGA_OWNER_CONTRACT` (what the manga code reads
  unguarded from its `main_gui`: `config`, `_get_environment_variables`, `contextual_var`,
  `trans_history`; HeadlessOwner has all of them).

### Behaviour deltas (intentional)
- `_get_app_dir()`'s last fallback is `data_dir(os.getcwd())` instead of `os.getcwd()`
  (mobile_runtime: the same value on desktop, app storage on mobile).
- manga_translator: when `from google.cloud import vision` raises ImportError, the module import,
  `_ensure_google_client` and `_google_ocr_rois_batched` use `google_vision_rest.vision` (REST with
  the service-account JSON via google-auth, or an API key). With the SDK installed nothing changes;
  a desktop source run without the SDK now OCRs over REST instead of failing with "Google Cloud
  Vision required".
- `process_image` no longer replaces `builtins.print` (nor `unified_api_client.print`) on mobile
  (`_manga_print_hijack_enabled()`: `mobile_runtime.is_mobile()`); a mobile job captures its own
  output and a process-wide print left pointing at a finished job's log would leak other features'
  output into it. Desktop keeps the hijack.
- Tracebacks name the new modules. tests/test_manga_drag_drop.py patches `_get_app_dir` in the module
  that now defines `_manga_ocr_output_dir`.

### Desktop bugs / defaults found (recorded, not fixed)
1. **Azure Document Intelligence start check reads the Computer Vision fields.** For
   `azure-document-intelligence` the batch start (`_prepare_manga_run_env`, formerly
   manga_integration 15697-15725) requires and saves `azure_vision_key` / `azure_vision_endpoint`
   (the Azure Computer Vision entries) and puts them into `ocr_config`, while the translator loads
   the provider with `azure_document_intelligence_key` / `_endpoint` (manga_translator ~4922). A
   Document Intelligence user must fill the Computer Vision fields too, or the start aborts with
   "Azure credentials not configured". Pinned by the run-start scenario
   `document_intelligence_uses_cv_widgets`.
2. **The manga tab's top-level defaults differ from the startup env defaults.** With no saved value
   `_load_rendering_settings` runs with `manga_bg_opacity` 0 (startup env 130),
   `manga_free_text_only_bg_opacity` False (True), `manga_shadow_color` [255, 255, 255]
   ([204, 128, 128]) and `manga_full_page_context` True (False). `MANGA_TOP_LEVEL_DEFAULTS` holds
   what the tab runs with; mobile shows those.
3. **A failed inpainter preload stalls each page for up to an hour.** When
   `preload_local_inpainters_concurrent` creates no instance (e.g. the model file is missing), the
   pool keeps an empty entry for the key and `_get_thread_local_inpainter` polls it for
   `CHUNK_TIMEOUT` seconds (1800 with Retry Timeout off) twice before inpainting gives up. Mobile
   should only start a local-inpainting run once the model is on disk (manga_models).
4. **The default numeric sort orders by file name only.** Dropping a folder with chapter
   subfolders queues them chapter by chapter, then `_apply_manga_file_sort` (`('numeric', False)`)
   sorts all pages by `_natural_sort_key(basename)` (stable), interleaving chapters
   (`ch1/1.png, ch2/1.png, ch1/2.png, ch1/10.png`); the image range and an unsplit run follow that
   order. "Split first-level subfolders" process groups still split per chapter.

## U8 Integrate (editor core, model registry, mobile Tools › Manga, packaging, offline E2E)

The parity oracles stay frozen at 9355bb5d (the manga-env step's re-freeze; HEAD 28079156 adds three
owner commits that touch only the mobile app and `key_pool_service`, no manga source). This section
collects what the editor-core, model-registry and mobile-UI steps and the integration found; the
manga-env step's own section is above.

### Desktop packaging (pinned by tests/test_mobile_runtime.py)
- The eight U8 modules (`manga_settings_defaults`, `manga_env`, `manga_files_core`, `manga_runner`,
  `manga_editor_core`, `manga_models`, `google_vision_rest`, `azure_document_intelligence_rest`) follow
  the manga tiers: both blocks of translator_Heavy / translator_NoCuda / translator_linux_NoCuda,
  `app_files` only in the two Mac NoCuda specs, none of the nine Lite/standard specs.
  `test_u8_manga_modules_follow_the_manga_spec_tiers` checks each spec against the tier
  `manga_integration` itself ships in, that no module a Lite spec ships imports a U8 core at module
  level, and that translator_gui still gates manga with `importlib.util.find_spec("manga_integration")`.
- The desktop `ImageStateManager` worker process now unpickles
  `manga_editor_core._state_manager_worker_process` (the class moved there), so frozen builds need
  `manga_editor_core` in `app_modules` (done for the full manga tier).

### Editor core (manga_editor_core; ImageRenderer, manga_image_preview, manga_integration rewired)
Behaviour deltas (intentional):
- The editor's `google` OCR (`_run_ocr_on_regions`) falls back to `google_vision_rest` when
  `from google.cloud import vision` fails, like manga_translator. Desktop builds ship the SDK.
- The `ImageStateManager` worker child imports `manga_editor_core` instead of `manga_integration`
  (a lighter child process, no Qt). On mobile the class runs without the worker process
  (`mobile_runtime.processes_available()`).

Desktop bugs found (recorded, not fixed):
1. **Translate All leaves the preview on the cleaned file.** It queues `load_preview_image` with
   `<page>_translated/<page>_cleaned.png` and `preserve_rectangles=True`;
   `MangaImagePreviewWidget.load_image` then sets `current_image_path` (and the tab's
   `_current_image_path`) to that file, so the next `_persist_current_image_state` (e.g. on a page
   change) writes an `image_state.json` entry keyed by the cleaned image. The mobile
   `MangaEditorSession` keeps the original page open and reports the cleaned image as an output.
2. **Deleting a box does not re-index the page's texts.** `_handle_delete_rectangle` removes the
   rectangle but not its slot in `recognized_texts` / `translated_texts`; after a delete, the OCR
   export or a reload can attach the deleted box's text to a remaining box (reproduced with the
   real `MangaEditorSession`; mobile keeps the desktop behaviour).

### Model registry and OCR providers (manga_models, azure_document_intelligence_rest; bubble_detector, local_inpainter, ocr_manager, settings_schema)
- Mobile only: the default model caches are `<data>/models/{detector,inpainting,onnx}`
  (runtime_bootstrap exports BUBBLE_CACHE_DIR / MODEL_CACHE_DIR / ONNX_CACHE_DIR before the
  backend is importable); desktop keeps `models` and `~/.cache/inpainting`. ocr_manager's Azure
  Document Intelligence provider uses the REST client on mobile only (desktop: unchanged "SDK not
  installed" without the SDK). `settings_schema.UNAVAILABLE_RULES` marks the torch-only manga values
  (manga-ocr, Qwen2-VL, EasyOCR, DocTR, Paddle, RT-DETR torch, YOLO, Torch JIT / Hybrid inpainting,
  ONNX conversion) and `ollama` / `sd_local` unavailable on mobile, with a reason.

Desktop bugs found (recorded, not fixed):
3. **Azure Document Intelligence SDK name mismatch.** Every `requirements*.txt` pins
   `azure-ai-documentintelligence==1.0.2` (module `azure.ai.documentintelligence`), but
   `ocr_manager.AzureDocumentIntelligenceProvider` imports `azure.ai.formrecognizer`, so a fresh
   desktop install reports the provider as "SDK not installed".
4. **`ollama` and `sd_local` local inpainting have no backend.** The desktop combo offers them, but
   local_inpainter has no `LAMA_JIT_MODELS` entry or handler for either. Mobile shows them disabled
   ("Not functional on desktop either").

### Found by the offline E2E (recorded, not fixed)
5. **"Create CBZ at end" logs an error after a CBZ input was packaged.** With a CBZ in the Files list
   and the default `manga_create_cbz_at_end` on, the worker's end runs `_finalize_cbz_jobs` (which
   writes `<name>_translated.cbz` from `<name>_translated/`) and then
   `_create_cbz_from_isolated_folders`, which only looks for per-page `<page stem>_translated`
   folders next to the first page (or in OUTPUT_DIRECTORY) and logs "⚠️ No translated folders found
   for CBZ creation" / "❌ Error creating CBZ file: No translated images found" with a traceback. The
   run and the CBZ are fine; mobile runs the same code (moved verbatim) and shows the same log.

### Mobile (no desktop change)
- **Font-size presets.** `manga_settings_defaults.font_preset_updates` (manga-env step) measures a
  preset by running the moved `_set_font_preset` on scratch headless tabs whose set-up writes
  `os.environ` (and puts it back). The app therefore measures them on the io pool, under
  `job_runner.JOB_LOCK` (refused with "Presets can be applied once the running job has finished"
  while a job owns the process state, so a running batch never sees its env reverted), once per
  session (the desktop presets set constants); the Settings tab only checks
  `presets_available()` while it builds (the first measurement takes ~4 s on the host). Rendering
  Reset and the custom image-edit endpoint Test stay disabled with a ReasonChip (their desktop code
  is still bound to Qt dialogs).
- The manga box / model sheets use `components.sheet` (scrolling body, bottom inset) and close by
  identity (`components.dialogs.close_dialog`), following the owner's device fix 57f1835c.
- Tier-B SDKs stay out of the app: Google Cloud Vision runs through `google_vision_rest` and Azure
  Document Intelligence through `azure_document_intelligence_rest`. `check_mobile_wheels.py` on a
  scratch pyproject resolves grpcio==1.81.0 + grpcio-status==1.81.0 + google-cloud-vision 3.16.0
  and azure-ai-documentintelligence 1.0.2 for every Android/iOS target (no new errors besides the
  documented cryptography / pillow release blocker), so they can be pinned later.
- RapidOCR ships as a device-only dependency (`[tool.flet.android/ios].dependencies`:
  rapidocr-onnxruntime 1.2.3, pyclipper 1.4.0, shapely 2.1.2); `backend_manifest.toml` no longer
  lists it as unavailable. rapidocr-onnxruntime requires opencv-python, so the device build installs
  opencv-python and opencv-python-headless 5.0.0.93 (both provide `cv2`); watch the first APK/IPA.

### Self-test, offline E2E and host smoke additions
- `fake_llm_server`: OCR-response mode (`ocr_text`): vision requests answer with the given text
  (`FAKE_MANGA_OCR_TEXT`, Korean bubble text); an image request that already carries Hangul is the
  manga full-page-context translation, and a `[N] text` request gets the JSON object its prompt asks
  for (`manga_segments` / `manga_reply`). `fixtures.build_manga_cbz` writes a CBZ of Pillow pages.
- `e2e_manga_cbz`: a 3-page CBZ imported through FileBridge, added by the moved Files logic, run by
  the `manga` job (HeadlessMangaRunner, HeadlessOwner as `main_gui`): full-page custom-api OCR
  (bubble detection off, so no model download), translation through the same endpoint, inpainting
  skipped, text rendered on every page, the automatic OCR export and the translated CBZ.
- `tools/host_smoke.py` `manga_pipeline`: RT-DETR through the bundled bubble_detector's Python
  onnxruntime path on a tiny synthetic export planted in BUBBLE_CACHE_DIR (the bytes are pinned to
  their onnx.helper builder by tests/test_mobile_compat_patches_manga.py), then one fixture page
  through HeadlessMangaRunner with OCR and translation from the loopback fake server; env, cwd and
  config.json unchanged.

### Test hygiene fixed by the integration (test-only)
- tests/test_u7_tool_cores.py's stop-flag test left `GRACEFUL_STOP=1` in the process: it called
  `monkeypatch.delenv(raising=False)` on the absent key (which records nothing) and the code under
  test then set the variable directly, so teardown restored "1". In CI's single-process 3.10 run that
  broke `test_manga_env::test_headless_runner_stop_reaches_the_stop_protocol`. The test now records
  the original state first; the manga test clears the variable itself.
- tests/test_manga_env.py and tests/test_google_vision_rest.py no longer write into src/ (automatic
  OCR export, manga glossary backups, HTTP request logs): an autouse fixture with its own
  MonkeyPatch (several tests call `monkeypatch.undo()` between the legacy and new runs) points
  `_get_app_dir` at a temp dir and sets GLOSSARION_HTTP_LOG=0.
- Still pre-existing and not changed: tests/test_job_runner.py's `scoped_process_state` tests leave
  PYTHONPATH removed and `test_reset_for_new_run` / tests/test_glossary_files.py's stop trace leave
  `GRACEFUL_STOP=0` / `TRANSLATION_CANCELLED=1` behind in the process.

## U8 review fixes (mobile runs, model downloads, output paths, archives, editor gestures)

The parity oracles stay frozen at 9355bb5d: HEAD (1cd68178) contains U8 itself (committed as
9df4ebe4), so a re-freeze there would compare the moved code with itself. No moved method body
changed; the desktop calls none of the new code paths.

### Shared code (desktop behaviour unchanged)
- `manga_env.build_ocr_config(config)` (contract helper, no desktop caller) resolves the provider
  like the tab before any worker reads `ocr_provider_value` (`manga_ocr_provider`, then
  `ocr_provider`, then `custom-api`) instead of the method's never-used getattr fallback; a parity
  case compares it with a HeadlessMangaState's own `_build_manga_worker_ocr_config()`.
- `manga_models.apply_mobile_run_defaults(config)`: mobile only, fills every phone default the
  config does not store (absent / None / blank) into a run's config; the local inpainter, stored
  twice, follows the top-level `manga_local_inpaint_model` (what mobile Settings shows and
  `required_models` reads; the desktop calls it "more up-to-date than nested inpainting").
- `bubble_detector.hf_urllib_download` (mobile only: reached through `_hf_download_fn` /
  local_inpainter's mobile branch): a model the manga_models registry pins is downloaded by
  `manga_models.download` (pinned revision, `.partial` resume, size + sha256 check) and stops with
  the run's immediate Stop (MangaTranslator's global cancellation). A non-empty file already at the
  target is still reused; unregistered files keep the plain urllib path.
- `manga_files_core._get_app_dir`: second edit of the moved helper, the non-frozen Windows branch
  goes through `mobile_runtime.data_dir` too (desktop never sets GLOSSARION_DATA_DIR, so it is
  unchanged; the mobile app running from source on Windows no longer writes `OCR Text` /
  `MangaGlossary_Backup` into src/). tests/test_manga_env.py pins both edits (GET_APP_DIR_EDITS).

### Mobile (services.manga, job_kinds.manga, the Manga screens)
- A MANGA / MANGA_STEP job prepares its config snapshot (never written back): the phone defaults
  (above), no desktop `output_directory`, and the mitigation of desktop bug 1 in "U8 Manga run
  env" (the Document Intelligence start check): for an Azure
  Document Intelligence run the empty member of each key / endpoint pair (Computer Vision vs
  Document Intelligence; the CV endpoint placeholder counts as empty) is filled from the other,
  so the shared start check and the provider load both get the credentials Settings offers.
- `OUTPUT_DIRECTORY` is hidden from the manga code in mobile manga jobs (the job's process state
  puts it back) and in the Files tab's lookups (`manga_output_view`, under `job_runner.JOB_LOCK`;
  "busy" while a job owns the process state). The mobile env contract's OUTPUT_DIRECTORY is the
  platform's output root, which the manga code treats as a user's output override: every page went
  to `<root>/<page name>_translated/`, so chapter folders with the same page names overwrote each
  other and every series shared one folder and one `<root>_translated.cbz`. Pages now go next to
  their source copy in app storage (the desktop default); the automatic OCR export and the
  glossary backups go to the app folder (GLOSSARION_DATA_DIR).
- The job downloads the registered models the run loads that are not on the device yet before it
  starts (`ensure_run_models`: progress in the job, Stop cancels and keeps the partial file, a
  failed download fails the job instead of stalling every page: "U8 Manga run env" desktop bug 3);
  the editor passes the kinds
  a step loads (`params["model_kinds"]`). The model dialog's "Start anyway" became "Download in the run".
- Archives: a ZIP / CBZ is extracted under `<root>/<id of path + size + mtime>/<name>` (a different
  archive that reuses a name gets its own folder; the folder keeps the archive's name). The
  selection's CBZ jobs persist in mobile_state.json (`manga_cbz_jobs`) and are re-attached after a
  restart; Create CBZ packs pages of an imported CBZ back into `<name>_translated.cbz` with the
  moved `_finalize_cbz_jobs` (the others still go through `_create_cbz_from_isolated_folders`), and
  the archives a run or Create CBZ wrote are listed under Output (share / save).
- Editor: in Pan mode the gesture surface over the InteractiveViewer takes long-presses only (its pan
  recognizer won Flutter's gesture arena and swallowed one-finger panning); the source viewer is
  kept (same controls and keys, updated in place) while the page and the Pan / Edit mode stay the
  same, so refreshes keep the zoom. Not verified on a device yet.
- The offline E2E's manga scenario points GLOSSARION_DATA_DIR at its sandbox and finds the
  automatic OCR export through the Files tab's `ocr_dir()`.

### Desktop bug found (recorded, not fixed)
1. **An output override flattens pages with the same name.** With an output folder set, the worker
   (`_translation_worker`'s routing), `_get_manga_output_path_for_file` and the editor write every
   page to `<override>/<page name>_translated/<page>`, so `ch1/001.png` and `ch2/001.png` of one run
   overwrite each other, and Create CBZ packs the override folder's matching `_translated` folders
   into `<override name>_translated.cbz`. Without an override (the default) pages stay next to their
   source. Mobile no longer hits it (above).

### Second review round (U8; mobile and test-only, no desktop change)
References to the U8 desktop bugs name their section ("U8 Manga run env" desktop bug 3, "U8 review
fixes" desktop bug 1): each U8 section numbers its own list.
- **Lookups a running job refused.** While another job owns the process state
  (`job_runner.JOB_LOCK`), `MangaFileList.output_path_for` / `existing_outputs` / `existing_cbz`
  raise `MangaBusy` instead of answering "nothing" (and `ocr_dir()` without a folder computed
  before). The Files tab keeps its earlier-outputs lookup pending (Create CBZ / Download images say
  "Wait for the running job", not "Translate pages first"), the Editor its Translated view ("Wait
  for the running job…"), and both run it again when a job of any kind ends (the JobService reports
  the end after the job let go of the process state) and when the tab shows. Auto-saved OCR says
  "Wait for the running job to finish" instead of "No auto-saved OCR files yet".
- **Generated glossary in Settings.** The selection's glossary auto-load (the moved
  `_refresh_manga_selection_status`, run by every selection change) now runs in the job's view
  too (`services.manga._MobileFilesHost`, which calls the moved method unchanged): a glossary pass
  writes `<source>/Glossary/<name>_manga_glossary.*` and `<GLOSSARION_DATA_DIR>/MangaGlossary_Backup`,
  while the Files host looked in `<Output>/Glossary`, so `manga_generated_glossary_path` was never
  stored and Settings › Glossary kept "No glossary loaded". After a MANGA job the Files tab runs the
  auto-load again (`MangaFileList.refresh_glossary`, persisted like a selection change) and
  re-renders Settings; a selection change made while another job ran is marked stale and refreshed
  once a job ends.
- **CBZ archives inside folders and ZIPs.** The archive-id extraction folder of the first round
  covered a ZIP / CBZ added directly only; a CBZ found in a picked folder or an extracted ZIP still
  went to the shared `<cbz root>/<name>`, so two series' `vol1.cbz` overwrote each other's pages and
  a re-imported folder listed the old archive's pages too. `_MobileFilesHost._add_cbz_archive_images`
  now gives every CBZ its `<cbz root>/<archive id>/` temp root around the moved method (the
  per-batch temp-root juggling in `add_paths` is gone).
- **Model rows.** `ModelDownloadRow` calls its `on_change` only when the status changes; download
  progress re-renders the row alone. Settings rebuilt the whole tab on the UI loop for every
  progress report (inline inpainting row, model sheet), with file IO in the rebuild; it now keeps a
  row across its own rebuilds while it shows the same model (`SettingsTab._download_row`), so a
  download keeps reporting into the row on screen.
- **Back (UI_SPEC §1.6).** `MangaScreen.handle_back` follows the tab on screen: Files leaves
  selection mode first; the Editor drops the selected box, then the edit tool; otherwise the View
  pops. Before, every tab used the Editor's state.
- **A download cancelled from a model row.** A row's Cancel (`manga_models.cancel`) also stops a
  MANGA / MANGA_STEP job's download of that model; the job ended DONE with nothing done. Only the
  job's own Stop now ends it stopped; otherwise it fails with "The model download was cancelled;
  start again to resume it" (the partial file is kept).
- **A batch that ended while the screen was closed.** The Files tab applies the session's batch end
  when it shows again (run status, pages, archives, glossary; once per job). After an app restart
  `existing_cbz` also finds the run's "Create CBZ at end" archive of the run files' folder (the
  rule of `HeadlessMangaRunner.cbz_paths`), so it is listed under Output again.
- **tests/test_mobile_compat_patches_manga.py (test-only).**
  `test_hf_urllib_download_hands_registered_models_to_manga_models` builds no ONNX session, so it is
  marked `backend` (numpy + cv2, what importing bubble_detector needs) instead of `runtime`
  (onnxruntime): CI's python-app job, which has no onnxruntime, now runs the only test of the
  bubble_detector → manga_models hand-off.
