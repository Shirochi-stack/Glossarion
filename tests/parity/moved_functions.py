"""Registry of desktop code that the mobile rewrite moves into shared GUI-free mixins.

One ``Moved`` entry per method (or class attribute) that leaves ``TranslatorGUI``
for a flat ``src/<module>.py`` mixin. ``tests/parity/test_parity_tiers.py`` is
parametrised over this list:

* tier D (``fuzz_moved``): the legacy frozen copy (``tests/parity/legacy``) and the
  new desktop method (``TranslatorGUI``'s MRO: shared mixin + desktop hook
  overrides) must agree on >= 500 seeded random owner states;
* MRO / duplicate checks: once the module exists, the mixin defines the name and
  ``TranslatorGUI``'s own body no longer does (moved bodies are deleted);
* tier I (``import_hygiene``) covers every module in ``SHARED_MODULES``;
* the owner contract (``owner_contract``) scans every module in ``MIXINS``.

Entries whose module does not exist yet are skipped with a clear reason, so the
same registry serves later milestones: add the U3+ moves (TextJobsMixin,
TranslationPipelineMixin, ...) here when they are planned.

Fields:

``name``      attribute on the mixin (the new side).
``module``    flat src module, ``mixin`` the class inside it.
``legacy``    frozen ``LegacyMethods`` attribute(s) that hold the old code
              (default: ``(name,)``). Synthetic ``__init__`` blocks are named
              ``legacy_*`` by ``freeze_legacy``.
``via``       fuzz through this caller on both sides instead (split-outs that
              have no legacy method of their own: the caller is frozen, and the
              working-tree caller now delegates to the split-out).
``args``      argument profile name (``fuzz_moved.ARG_PROFILES``); ``None``
              derives arguments from the signature and harvested call sites.
``setup``     attributes set on every base state before perturbation
              (``'<SANDBOX>'`` is replaced by the fuzz sandbox root).
``fuzz``      ``False`` disables tier D for the entry; ``reason`` says which
              tier covers it instead.
``optional``  the name may legitimately stay in ``TranslatorGUI`` (decided while
              moving); a missing mixin attribute then skips instead of failing.
``kind``      ``'method'`` or ``'attr'`` (class attributes: value equality only).
``stubs``     ``(module, attribute)`` pairs replaced by call recorders while the entry
              is fuzzed (backend work the moved code hands off, e.g. the metadata worker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

U1 = "U1"
U2 = "U2"
U3 = "U3"


@dataclass(frozen=True)
class Moved:
    name: str
    module: str
    mixin: str
    milestone: str = U2
    legacy: tuple = ()
    via: str | None = None
    args: str | None = None
    setup: dict = field(default_factory=dict)
    fuzz: bool = True
    reason: str = ""
    optional: bool = False
    kind: str = "method"
    stubs: tuple = ()

    @property
    def legacy_names(self) -> tuple:
        return tuple(self.legacy) or (self.via or self.name,)

    @property
    def call_name(self) -> str:
        """Attribute called on the new desktop owner."""
        return self.via or self.name

    @property
    def id(self) -> str:
        return f"{self.module}.{self.mixin}.{self.name}"


#: Shared mixin classes by module (tier D new side, owner contract scan, MRO checks).
MIXINS = {
    "owner_state": "ConfigStateMixin",
    "run_env": "RunEnvMixin",
    "settings_persistence": "SettingsPersistenceMixin",
    # U3
    "text_jobs": "TextJobsMixin",
    "input_preparation": "InputPreparationMixin",
    # (TranslationPipelineMixin inherits GlossaryPipelineMixin and PipelineHooksMixin)
    "translation_pipeline": "TranslationPipelineMixin",
}

#: Desktop hooks (shared-core §3.3): mixin default is GUI-free, TranslatorGUI overrides
#: with the original code. They are NOT moved names: TranslatorGUI may define them.
HOOK_NAMES = frozenset({
    "_can_read_widgets",
    "_backend_entry",
    "_hook_ensure_executor",
    "_hook_metadata_defaults",
    "_hook_authza_mode",
    "_hook_persist_sanitized_config",
    "_on_profile_reset",
    "_notify_compile_result",
    "_ui_request",
    "_subprocess_allowed",
    # added by the U2 move (startup save_config / context-mode widget layout)
    "_hook_save_default_config",
    "_hook_context_mode_layout",
    # U3 pipelines (translation_pipeline.PipelineHooksMixin): message box hook, and GUI-free
    # defaults of the TranslatorGUI GUI methods the moved pipelines call (desktop keeps its
    # own, which win), incl. the U7 placeholders (image / RPG Maker / generative runners)
    "_ui_message",
    "_lazy_load_modules",
    "_attach_gui_logging_handlers",
    "_create_watchdog_snapshot",
    "_start_autoscroll_delay",
    "_update_manual_glossary_status",
    # U3 fix pass: the set-up's Library raw-input registry write (desktop: epub_library)
    "_record_library_raw_inputs",
    "_process_image_file",
    "_process_rpgmaker_game",
    "_run_generative_prompt_mode",
})

#: Every flat GUI-free module of the shared core (tier I: import hygiene + Python 3.10 parse).
SHARED_MODULES = (
    # U1
    "mobile_runtime",
    "app_paths",
    "config_store",
    "prompt_defaults",
    "metadata_defaults",
    "ollama_settings",
    "key_pools",
    "output_naming",
    "pdf_mupdf_html",
    # U2
    "owner_state",
    "run_env",
    "settings_persistence",
    "headless_owner",
    "settings_schema",
    "settings_schema_data",
    "settings_rules",
    "library_core",
    # U3
    "job_runner",
    "stop_control",
    "text_jobs",
    "input_preparation",
    "translation_pipeline",
    # U3 step 3: the Direct Text dialog's mixins (tests/test_direct_text_core.py pins the moves)
    "direct_text_store",
    "direct_text_stream",
)

_RUN_ENV = ("run_env", "RunEnvMixin")
_STATE = ("owner_state", "ConfigStateMixin")
_PERSIST = ("settings_persistence", "SettingsPersistenceMixin")
_TEXT = ("text_jobs", "TextJobsMixin")
_INPUT = ("input_preparation", "InputPreparationMixin")


def _run_env(name: str, **kw) -> Moved:
    return Moved(name, *_RUN_ENV, **kw)


def _state(name: str, **kw) -> Moved:
    return Moved(name, *_STATE, **kw)


def _persist(name: str, **kw) -> Moved:
    return Moved(name, *_PERSIST, **kw)


def _text(name: str, **kw) -> Moved:
    return Moved(name, *_TEXT, milestone=U3, **kw)


def _input(name: str, **kw) -> Moved:
    return Moved(name, *_INPUT, milestone=U3, **kw)


_PIPELINE = ("translation_pipeline", "TranslationPipelineMixin")
#: why the pipeline moves are not fuzzed here (the main oracle freezes most of them without
#: their closure): tier T runs them end to end, legacy vs desktop vs mobile
_TIER_T = ("tier T (tests/parity/test_trace_parity.py) drives the whole pipeline, legacy oracle vs "
           "working-tree desktop vs HeadlessOwner; tests/test_translation_pipeline.py checks the "
           "body is the legacy body plus the documented edits")


def _pipeline(name: str, **kw) -> Moved:
    """TranslationPipelineMixin / GlossaryPipelineMixin (resolved through TranslationPipelineMixin)."""
    kw.setdefault("fuzz", False)
    kw.setdefault("reason", _TIER_T)
    return Moved(name, *_PIPELINE, milestone=U3, **kw)


_COMPILE_FOLDER = "<SANDBOX>/outputs/Fuzz Novel"
#: selected_files exercising every input-preparation branch (fuzz_moved.FUZZ_ARCHIVES)
_ARCHIVE_SELECTION = [
    "<SANDBOX>/inputs/Fuzz Chapters.zip",
    "<SANDBOX>/inputs/Fuzz Subs.zip",
    "<SANDBOX>/inputs/Fuzz Page.html",
    "<SANDBOX>/inputs/Fuzz Novel.epub",
    "<SANDBOX>/inputs/Fuzz Images.cbz",
]

#: shared-core design §3.2 (RunEnvMixin) + §1 P2 row (ConfigStateMixin) + §2 (SettingsPersistenceMixin).
MOVED = (
    # ---- RunEnvMixin: glossary resolution and main builder ---------------------------------
    _run_env("_resolve_glossary_for_env"),
    _run_env("_get_environment_variables"),
    # ---- metadata, streaming, Direct Text ------------------------------------------------------
    _run_env("_metadata_only_environment_for_file"),
    _run_env("_apply_forced_streaming_environment"),
    _run_env("_apply_direct_text_runtime_environment"),
    _run_env("_format_translation_anti_duplicate_settings"),
    _run_env("_log_translation_anti_duplicate_settings"),
    # ---- output-mode helpers ----------------------------------------------------------------
    _run_env("_model_is_image_gen"),
    _run_env("_model_is_video_gen"),
    _run_env("_is_generative_output_mode"),
    _run_env("_get_output_mode"),
    _run_env("_get_allowed_image_output_mode"),
    _run_env("_get_allowed_video_output_mode"),
    # ---- retry and chunk helpers ------------------------------------------------------------
    _run_env("_resolve_max_retry_tokens"),
    _run_env("_resolve_max_retries"),
    _run_env("_compression_chunk_budget"),
    # ---- multipass, range, scan helpers -------------------------------------------------------
    _run_env("_get_multipass_refinement_mode"),
    _run_env("_sync_multipass_refinement_mode_from_combo"),
    _run_env("_export_multipass_runtime_env"),
    _run_env("_live_chapter_range_settings"),
    _run_env("_export_chapter_range_runtime_env"),
    _run_env("_parse_chapter_range_text"),
    _run_env("_get_scan_phase_mode"),
    _run_env("_get_qa_scanner_settings_json"),
    # ---- custom prefix routes and Ollama --------------------------------------------------------
    _run_env("_normalize_custom_prefix_endpoint_type"),
    _run_env("_is_valid_custom_prefix_endpoint_type"),
    _run_env("_normalize_custom_prefix_routes"),
    _run_env("_custom_prefix_routes_env_json"),
    _run_env("_ollama_settings_env_json"),
    _run_env("_sync_custom_prefix_routes_env"),
    # ---- context, batching, glossary env values -------------------------------------------------
    _run_env("_context_mode_from_flags"),
    _run_env("_translation_batching_mode_for_env"),
    _run_env("_glossary_batching_mode_for_env"),
    _run_env("_live_bool_setting"),
    _run_env("_live_text_setting"),
    _run_env("_current_auto_glossary_mode"),
    _run_env("_current_glossary_request_env"),
    _run_env("_glossary_contextual_env_value"),
    _run_env("_glossary_skip_title_header_only_env_value"),
    _run_env("_glossary_add_minimal_pass_env_value"),
    _run_env("_glossary_match_engine_env_value"),
    _run_env("_strict_matching_env_dict"),
    _run_env("_unified_glossary_env_dict"),
    # ---- output paths -------------------------------------------------------------------------
    _run_env("_get_output_base_dir"),
    _run_env("_subtitle_zip_output_info"),
    _run_env("_resolve_translation_output_dir"),
    _run_env("_active_translation_output_mode"),
    _run_env("_output_side_glossary_backup_dir_for_source"),
    _run_env("_current_glossary_cjk_script_filter_enabled"),
    # ---- global env exports -------------------------------------------------------------------
    _run_env("_glossary_env_mappings"),
    _run_env("initialize_environment_variables"),
    _run_env("debug_environment_variables", optional=True,
             reason="called by initialize_environment_variables; may stay a TranslatorGUI method"),
    _run_env("_parallel_epub_system_prompt_for_file", optional=True,
             reason="called by the glossary env split-out; may stay a desktop hook"),
    # ---- split-outs (no legacy method of their own: fuzzed through the frozen caller) ---------
    _run_env("_build_glossary_extraction_env", via="_extract_glossary_from_text_file",
             reason="split out of _extract_glossary_from_text_file 38009-38221"),
    _run_env("_glossary_extraction_paths", via="_extract_glossary_from_text_file",
             reason="output/backup path prelude split out of _extract_glossary_from_text_file"),
    _run_env("_build_epub_compile_env", via="run_epub_converter_direct",
             setup={"epub_folder": _COMPILE_FOLDER},
             reason="split out of run_epub_converter_direct 38611-38791"),
    _run_env("_build_pdf_compile_env", via="run_pdf_converter_direct",
             setup={"pdf_folder": _COMPILE_FOLDER},
             reason="split out of run_pdf_converter_direct 38447-38482"),
    # ---- class attributes -----------------------------------------------------------------------
    _run_env("_IMAGE_MODEL_ALIASES", kind="attr"),
    _run_env("_VIDEO_MODEL_ALIASES", kind="attr"),
    _run_env("_CHUNK_BUDGET_SAFETY_MARGIN", kind="attr"),
    _run_env("_CUSTOM_PREFIX_ENDPOINT_TYPES", kind="attr"),

    # ---- ConfigStateMixin -------------------------------------------------------------------
    _state("_init_config_state", legacy=("legacy_init_block",),
           reason="__init__ config block 13738-14514 (frozen as legacy_init_block)"),
    _state("_init_variables"),
    _state("_init_default_prompts"),
    _state("_sanitize_config_prompts"),
    _state("_get_protected_prompt_profiles"),
    _state("_reset_prompt_profile_to_default"),
    _state("_migrate_strict_matching_config"),
    _state("_upgrade_special_file_exact"),
    _state("_coerce_live_bool"),
    _state("_init_gui_backed_state", fuzz=False,
           reason="new composite of the _setup_gui / initialize_extraction_variables side effects "
                  "(desktop runs them interleaved with widget creation); covered by the goldens' "
                  "boot phases (tier G) and the HeadlessOwner round trip (tier R)"),
    _state("_LEGACY_SPECIAL_FILE_EXACT_TOKENS", kind="attr"),
    # ---- startup handlers and helpers TranslatorGUI's builders run (moved verbatim; the
    # Qt lines are hooks: _hook_context_mode_layout, _can_read_widgets) -------------------------
    _state("_update_auto_compression_factor"),
    _state("_sync_chunk_size_entry"),
    _state("_set_batching_mode"),
    _state("_refresh_context_batching_controls"),
    _state("_enforce_context_batching_mode"),
    _state("_on_disable_temperature_toggle"),
    _state("_on_context_mode_changed"),
    _state("update_target_language"),
    # ---- new compositions / init splits without a legacy method of their own -------------------
    _state("_init_default_prompt_profiles", fuzz=False,
           reason="__init__ default-prompt block split out beside _init_config_state; covered by the "
                  "goldens' boot phase (tier G) and test_headless_owner's verbatim-move check"),
    _state("_init_watchdog_dir", fuzz=False,
           reason="__init__ watchdog block split out beside _init_config_state; covered by tier G boot "
                  "and test_headless_owner's verbatim-move check"),
    _state("_on_auto_glossary_shortcut_changed", fuzz=False,
           reason="lifted from a nested function of the legacy settings builder; fuzzed against that "
                  "nested function by test_headless_owner (400 seeded states) and covered by tier G boot"),
    _state("_saved_auto_glossary_shortcut_index", fuzz=False,
           reason="startup combo index the legacy builder computed inline; covered by tier G boot and "
                  "test_headless_owner's startup-handler replay test"),
    _state("_resolve_startup_target_language", fuzz=False,
           reason="startup target-language value the legacy builder computed inline; covered by tier G "
                  "boot and tier R"),
    _state("_init_active_profile_prompt", fuzz=False,
           reason="active-profile prompt init the legacy builder ran inline; covered by tier G boot"),
    _state("_replay_gui_startup_handlers", fuzz=False,
           reason="HeadlessOwner-only replay of the builders' startup handlers in desktop order; covered "
                  "by tier G (HeadlessOwner env) and test_headless_owner's replay-order test"),

    # ---- SettingsPersistenceMixin (save_config keeps its dialogs and delegates) ---------------
    _persist("_collect_live_settings", via="save_config", args="save_config",
             reason="extracted from save_config's settings_map loop 47447-48090"),
    _persist("_export_settings_env", via="save_config", args="save_config",
             reason="extracted from save_config 48092-48210"),
    _persist("_apply_live_settings_to_config", via="save_config", args="save_config",
             reason="save_config sections 2-3 in place (_collect_live_settings is its deep-copying view)"),

    # ---- TextJobsMixin (U3): per-file runners; backends are the stubbed entry points ----------
    _text("_process_text_file"),
    _text("_extract_glossary_from_text_file"),
    _text("_run_parallel_metadata_files", args="unique_files",
          stubs=(("metadata_translation_worker", "run_metadata_translation_job"),),
          reason="the metadata worker is a recorder; unique files keep the thread pool's log order fixed"),
    _text("_run_epub_compile", via="run_epub_converter_direct", setup={"epub_folder": _COMPILE_FOLDER},
          reason="the try/except of run_epub_converter_direct (the desktop runner keeps its finally)"),
    _text("_run_pdf_compile", via="run_pdf_converter_direct", setup={"pdf_folder": _COMPILE_FOLDER},
          reason="the try/except of run_pdf_converter_direct (the desktop runner keeps its finally)"),
    # ---- InputPreparationMixin (U3): ZIP / CBZ / HTML / subtitle-ZIP inputs ---------------------
    _input("_convert_zip_input_to_epub_if_needed", args="archive_path",
           reason="wrapper over input_preparation.resolve_input_to_epub (the moved body)"),
    _input("_extract_subtitle_zip_input_if_needed", args="archive_path"),
    _input("_has_epub_conversion_inputs", setup={"selected_files": _ARCHIVE_SELECTION}),
    _input("_resolve_zip_inputs_for_translation", setup={"selected_files": _ARCHIVE_SELECTION}),
    # ---- TranslationPipelineMixin (U3): run_translation_thread split at its worker thread -------
    _pipeline("_prepare_translation_run",
              reason="run_translation_thread's set-up between the Run-button preflight and the worker "
                     "thread (no legacy method of its own); " + _TIER_T),
    _pipeline("_translation_worker",
              reason="run_translation_thread's simple_thread_target closure body (no legacy method of "
                     "its own); " + _TIER_T),
    _pipeline("run_translation_direct"),
    _pipeline("_await_direct_text_glossary_approval"),
    _pipeline("_clear_automatic_glossary_for_non_epub_selection"),
    # ---- QA-failure collection and multipass refinement planning ---------------------------------
    _pipeline("_flatten_translation_qa_issue_text"),
    _pipeline("_is_foreign_character_translation_qa_issue"),
    _pipeline("_entry_has_foreign_character_qa_failure"),
    _pipeline("_collect_translation_qa_failures"),
    _pipeline("_chapter_scope_filename_key"),
    _pipeline("_filter_translation_qa_failures_to_current_range"),
    _pipeline("_translation_qa_failure_key"),
    _pipeline("_qa_failure_matches_resolution_request"),
    _pipeline("_prepare_multipass_qa_refinement_run"),
    _pipeline("_clear_translation_run_overrides"),
    _pipeline("_format_chapter_list"),
    _pipeline("_log_translation_qa_failure_summary"),
    # ---- GlossaryPipelineMixin: glossary extraction, image-folder glossary ----------------------
    _pipeline("run_glossary_extraction_direct"),
    _pipeline("_process_image_folder_for_glossary"),
    _pipeline("_init_image_glossary_progress_manager"),
    _pipeline("_save_intermediate_glossary_with_skip"),
    _pipeline("_call_api_with_interrupt"),
    # ---- glossary auto-loading / auto-mapping -----------------------------------------------------
    _pipeline("auto_load_glossary_for_file"),
    _pipeline("_auto_load_glossary_after_extraction"),
    _pipeline("_autofill_glossary_for_current_selection"),
    _pipeline("_glossary_dir_signature"),
    _pipeline("_get_glossary_dir_candidates"),
    _pipeline("_guess_glossary_for_input_file"),
    _pipeline("_copy_glossary_to_output_folders"),
    _pipeline("_sync_automapped_glossaries_to_output"),
    # ---- input-selection helpers both pipelines use ---------------------------------------------
    _pipeline("_is_special_file"),
    _pipeline("_should_skip_special_file"),
    _pipeline("_get_spine_filenames_for_preview"),
    _pipeline("_get_opf_file_order"),
    _pipeline("_windows_supported_input_path"),
    _pipeline("_windows_glossary_rename_dirs"),
    _pipeline("_path_lookup_key"),
    _pipeline("_remap_windows_renamed_epub_glossary"),
    _pipeline("_normalize_windows_input_filenames"),
)

MOVED_BY_NAME = {m.name: m for m in MOVED}
FUZZED = tuple(m for m in MOVED if m.kind == "method" and m.fuzz)
CLASS_ATTRS = tuple(m for m in MOVED if m.kind == "attr")
METHODS = tuple(m for m in MOVED if m.kind == "method")


def by_module(module: str) -> tuple:
    return tuple(m for m in MOVED if m.module == module)


def get(name: str) -> Moved:
    return MOVED_BY_NAME[name]


def _check_registry() -> None:
    names = [m.name for m in MOVED]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate moved names: {dup}")
    for m in MOVED:
        if m.module not in MIXINS or MIXINS[m.module] != m.mixin:
            raise ValueError(f"{m.name}: unknown mixin {m.module}.{m.mixin}")
        if m.name in HOOK_NAMES:
            raise ValueError(f"{m.name} is a hook, not a moved name")
        if m.kind not in ("method", "attr"):
            raise ValueError(f"{m.name}: bad kind {m.kind!r}")
        if not m.fuzz and not m.reason:
            raise ValueError(f"{m.name}: fuzz disabled without a reason")


_check_registry()

__all__ = [
    "CLASS_ATTRS",
    "FUZZED",
    "HOOK_NAMES",
    "METHODS",
    "MIXINS",
    "MOVED",
    "MOVED_BY_NAME",
    "Moved",
    "SHARED_MODULES",
    "by_module",
    "get",
]
