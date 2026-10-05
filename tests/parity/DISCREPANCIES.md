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
