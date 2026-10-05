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
