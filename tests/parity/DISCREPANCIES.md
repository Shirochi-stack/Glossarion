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
