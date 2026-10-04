"""The 12 golden scenarios of plan section 9 plus desktop startup widget values.

A scenario is plain data:

``config``     dict written to the sandbox ``config.json`` (``None`` = no file:
               a true fresh install). Strings may contain ``<SANDBOX>``.
``files``      {path relative to the sandbox root: text} created before boot.
``env``        extra variables in the scrubbed baseline environment.
``widgets``    overrides of the desktop startup widget values
               ({attr: text | bool | combo data}).
``run_attrs``  attributes set on the owner after boot (run-time state such as
               Direct Text / metadata-only / single-chapter flags); a dict or a
               callable ``(sandbox) -> dict`` when keys depend on sandbox paths.
``input``      input file (relative to the sandbox root) for the run entries.
``entries``    capture entries (see capture_golden.ENTRY_FUNCS).

``startup_widgets(owner, scenario)`` reproduces how ``_setup_gui`` fills the
widgets the frozen code reads (source lines @ BASE_SHA noted per widget).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from parity.fakes import FakeCheck, FakeCombo, FakeLineEdit, FakeTextEdit, FakeWidget  # noqa: E402

# translator_gui.py @ BASE_SHA 25435-25438
CONTEXT_MODE_ITEMS = (
    ("Off", "off"),
    ("Contextual History", "contextual_history"),
    ("Rolling Summary (Replace)", "rolling_summary_replace"),
    ("Rolling Summary (Append)", "rolling_summary_append"),
)
# translator_gui.py @ BASE_SHA 25670-25675 (index map 25676-25683)
MULTIPASS_ITEMS = (
    ("Full", "full"),
    ("Full + raw", "full_with_raw"),
    ("Failed", "failed"),
    ("Partial", "partial"),
    ("Partial.b", "partial.b"),
    ("Partial.b2", "partial.b2"),
)
_MULTIPASS_INDEX = {data: i for i, (_t, data) in enumerate(MULTIPASS_ITEMS)}
# translator_gui.py @ BASE_SHA 28271-28274
REMOVE_ARTIFACTS_ITEMS = (("Off", "off"), ("Low", "low"), ("Medium", "medium"), ("High", "high"))
# translator_gui.py @ BASE_SHA 25975 (items) and 25986-25991 (initial index)
AUTO_GLOSSARY_SHORTCUT_ITEMS = (
    ("Off", None), ("Off (Fuzzy Mapping)", None), ("Manual Glossary Only", None), ("No Glossary", None),
    ("Minimal", None), ("Balanced", None), ("Full", None), ("Single Pass", None),
)
_AUTO_GLOSSARY_SHORTCUT_INDEX = {
    "off": 0, "off_fuzzy_automap": 1, "off_no_automap": 2, "no_glossary": 3,
    "minimal": 4, "balanced": 5, "full": 6, "single_pass": 7,
}


def _auto_glossary_shortcut_index(cfg) -> int:
    mode = cfg.get("auto_glossary_mode", None)
    if mode is None:
        mode = "minimal" if cfg.get("enable_auto_glossary", False) else "off"
    return _AUTO_GLOSSARY_SHORTCUT_INDEX.get(mode.lower(), 0)


def _target_language_items():
    try:
        from language_options import TARGET_LANGUAGES

        return tuple((lang, None) for lang in TARGET_LANGUAGES)
    except Exception:  # pragma: no cover
        return (("English", None),)


def startup_widgets(owner, scenario: dict) -> dict:
    """Fake widgets filled exactly as desktop ``_setup_gui`` fills them at startup."""
    cfg = owner.config
    profile = getattr(owner, "profile_var", None)
    profiles = getattr(owner, "prompt_profiles", {}) or {}
    widgets = {
        # 25239-25240
        "thread_delay_entry": FakeLineEdit(str(owner.thread_delay_var)),
        # 25270-25271
        "delay_entry": FakeLineEdit(str(cfg.get("delay", 5))),
        # 25276-25277
        "api_queue_entry": FakeLineEdit(str(owner.api_queue_var)),
        # 25322-25324
        "chapter_range_entry": FakeLineEdit(cfg.get("chapter_range", "")),
        # 25341-25349
        "use_spine_order_checkbox": FakeCheck(bool(cfg.get("use_spine_order", False)), "Spine Order"),
        # 25366-25367
        "token_limit_entry": FakeLineEdit(f"{cfg.get('token_limit') or 200000:,}"),
        # 25433-25479 (setCurrentIndex(idx if idx >= 0 else 0) of findData(context_mode_var))
        "context_mode_combo": FakeCombo.with_data(CONTEXT_MODE_ITEMS, owner.context_mode_var),
        # 17177 / 17202-17203
        "vertex_location_entry": FakeLineEdit(owner.vertex_location_var),
        "deep_scan_check": FakeCheck(bool(owner.deep_scan_var), "include subfolders"),
        # 25522-25523
        "trans_history": FakeLineEdit(str(cfg.get("translation_history_limit", 2))),
        "trans_history_label": FakeWidget(),
        # 25527-25540
        "rolling_summary_exchanges_label": FakeWidget(),
        "rolling_summary_exchanges_edit": FakeLineEdit(str(owner.rolling_summary_exchanges_var)),
        "rolling_summary_retain_label": FakeWidget(),
        "rolling_summary_retain_edit": FakeLineEdit(str(owner.rolling_summary_max_entries_var)),
        "rolling_summary_keys_btn": FakeWidget(),
        # 25587-25589 / 25634-25635
        "batch_checkbox": FakeCheck(bool(owner.batch_translation_var), "Batch Translation"),
        "batch_size_entry": FakeLineEdit(str(owner.batch_size_var)),
        # 28270-28297 (index 0 unless the saved level is found)
        "remove_artifacts_combo": FakeCombo.with_data(
            REMOVE_ARTIFACTS_ITEMS,
            owner.REMOVE_AI_ARTIFACTS_var if isinstance(owner.REMOVE_AI_ARTIFACTS_var, str) else "off",
        ),
        # 25974-25991: main-window glossary mode shortcut (its startup handler runs save_config)
        "auto_glossary_shortcut_combo": FakeCombo(
            AUTO_GLOSSARY_SHORTCUT_ITEMS, index=_auto_glossary_shortcut_index(cfg)
        ),
        # 25561-25562
        "trans_temp": FakeLineEdit(str(cfg.get("translation_temperature", 0.3))),
        # 25565-25573
        "disable_temperature_checkbox": FakeCheck(bool(owner.disable_temperature_var), "Disable"),
        # 25640-25652
        "multipass_checkbox": FakeCheck(bool(owner.multipass_mode_var), "Multipass mode"),
        # 25669-25683
        "multipass_refinement_mode_combo": FakeCombo(
            MULTIPASS_ITEMS, index=_MULTIPASS_INDEX.get(owner.multipass_refinement_mode_var, 0)
        ),
        # main-window Chunk Size field; text is driven by _sync_chunk_size_entry
        "chunk_size_entry": FakeLineEdit(""),
        # 28213-28217
        "api_key_entry": FakeLineEdit(cfg.get("api_key", "") or ""),
        # 28309 + 16819-16822 (active profile text)
        "prompt_text": FakeTextEdit(profiles[profile] if profile in profiles else ""),
        # 28438-28466: combo exists before update_target_language(final_lang) syncs it
        "target_lang_combo": FakeCombo(_target_language_items(), index=-1, text=""),
        # layout grid / glossary status row used by _on_context_mode_changed
        "frame": FakeWidget(),
        "_gloss_status_row": FakeWidget(),
    }
    for attr, value in (scenario.get("widgets") or {}).items():
        widget = widgets.get(attr)
        if isinstance(widget, FakeCheck):
            widget.setChecked(bool(value))
        elif isinstance(widget, FakeCombo):
            idx = widget.findData(value)
            widget.setCurrentIndex(idx) if idx >= 0 else widget.setCurrentText(str(value))
        elif isinstance(widget, FakeTextEdit):
            widget.setPlainText(str(value))
        elif isinstance(widget, FakeLineEdit):
            widget.setText(str(value))
        else:
            raise KeyError(f"unknown startup widget override {attr!r}")
    return widgets


# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

DEFAULT_ENTRIES = (
    "boot",
    "translation_env",
    "glossary_env_mappings",
    "run_helpers",
    "multipass_runtime_env",
    "chapter_range_runtime_env",
    "process_text_file",
    "extract_glossary",
    "epub_compile",
    "pdf_compile",
)

_GLOSSARY_CSV = "type,raw_name,translated_name,gender\ncharacter,김상현,Kim Sang-hyun,male\n"
_CREDS_JSON = '{"type": "service_account", "project_id": "parity-vertex-project"}'


def _key(i, prefix="sk-parity", model="gpt-4o-mini"):
    return {"api_key": f"{prefix}-{i}", "model": model, "cooldown": 60, "enabled": True}


def _metadata_run_attrs(sandbox):
    src = os.path.normcase(os.path.abspath(sandbox.path("inputs/Metadata Book.epub")))
    return {
        "_metadata_only_run": True,
        "_metadata_output_roots": {src: sandbox.path("outputs/metadata_root")},
    }


def _subtitle_run_attrs(sandbox):
    from subtitle_processor import plan_subtitle_archive_outputs

    archive = sandbox.path("inputs/Show S01.zip")
    members = [sandbox.path("work/Show S01/ep01.srt"), sandbox.path("work/Show S01/ep02.srt")]
    plan = plan_subtitle_archive_outputs(
        archive, members, sandbox.path("outputs"),
        work_base_dir=sandbox.path("work/.glossarion_subtitle_work"),
    )
    bundle_id = os.path.normcase(os.path.abspath(archive))
    bundle_files = [os.path.abspath(m) for m in members]
    for info in plan.values():  # shape of _extract_subtitle_zip_input_if_needed (42672-42681)
        info["bundle_id"] = bundle_id
        info["bundle_files"] = bundle_files
        info["bundle_work_dir"] = sandbox.path("work/.glossarion_subtitle_bundle")
    return {"_subtitle_zip_output_groups": plan}


SCENARIOS = {
    "fresh_install": {
        "description": "No config.json at all: desktop defaults (authgpt model, balanced glossary).",
        "config": None,
        "files": {"inputs/Parity Novel.epub": "PK"},
        "input": "inputs/Parity Novel.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "gemini_text_typical": {
        "description": "Typical Gemini text run: contextual history, conservative batching, range+spine.",
        "config": {
            "model": "gemini-2.5-flash",
            "api_key": "AIza-PARITY-TEST-0000",
            "active_profile": "Korean_BeautifulSoup",
            "output_language": "English",
            "delay": 2,
            "translation_temperature": 0.35,
            "translation_history_limit": 3,
            "contextual": True,
            "batching_mode": "conservative",
            "batch_translation": True,
            "batch_size": "5",
            "enable_gemini_thinking": True,
            "thinking_budget": "1024",
            "thinking_level": "medium",
            "auto_glossary_mode": "minimal",
            "token_limit": 150000,
            "token_limit_disabled": False,
            "chapter_range": "1-20",
            "use_spine_order": True,
            "glossary_max_sentences": 50,
            "max_output_tokens": 65536,
            "auto_update_check": False,
        },
        "files": {"inputs/Gemini Book.epub": "PK"},
        "input": "inputs/Gemini Book.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "vision_balanced": {
        "description": "Vision OCR output mode with balanced auto glossary and OCR batch size.",
        "config": {
            "model": "gemini-2.5-pro",
            "api_key": "AIza-PARITY-VISION",
            "output_mode": "vision",
            "enable_image_translation": True,
            "auto_glossary_mode": "balanced",
            "vision_ocr_batch_size": "3",
            "vision_ocr_keep_images": True,
            "batch_translation": True,
            "batch_size": "4",
            "text_extraction_method": "enhanced",
            "file_filtering_level": "comprehensive",
            "glossary_request_merging_enabled": False,
            "glossary_enable_chapter_split": True,
        },
        "files": {"inputs/Comic Scan.epub": "PK"},
        "input": "inputs/Comic Scan.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "refinement_multipass_partial_b": {
        "description": "Multipass refinement (partial.b) with QA scan phase and custom refinement prompts.",
        "config": {
            "model": "gpt-5-mini",
            "api_key": "sk-parity-refine",
            "auto_glossary_mode": "balanced",
            "multipass_mode": True,
            "multipass_refinement_mode": "Partial.B",
            "refinement_full_with_raw_raw_role": "SYSTEM",
            "refinement_system_prompt": "Refine to {target_lang}.",
            "refinement_partial_b_system_prompt": "",
            "refinement_partial_b_user_prompt": "Fix: {entries}",
            "scan_phase_enabled": True,
            "scan_phase_mode": "aggressive",
            "qa_scanner_settings": {"min_file_length": 10, "check_encoding_issues": True},
            "request_merging_enabled": True,
            "request_merge_count": 4,
            "retry_truncated": True,
            "max_retry_tokens": -1,
            "max_retries": "3",
        },
        "files": {"inputs/Refine Book.epub": "PK"},
        "input": "inputs/Refine Book.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "no_glossary": {
        "description": "No Glossary mode overrides a loaded manual glossary and compliance options.",
        "config": {
            "model": "deepseek-chat",
            "api_key": "sk-parity-noglossary",
            "auto_glossary_mode": "no_glossary",
            "append_glossary": True,
            "emergency_glossary_compliance": True,
            "manual_glossary_path": "<SANDBOX>/inputs/manual_glossary.csv",
        },
        "files": {
            "inputs/No Glossary Book.epub": "PK",
            "inputs/manual_glossary.csv": _GLOSSARY_CSV,
        },
        "run_attrs": {
            "manual_glossary_path": "<SANDBOX>/inputs/manual_glossary.csv",
            "manual_glossary_manually_loaded": True,
        },
        "input": "inputs/No Glossary Book.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "direct_text_attachment": {
        "description": "Direct Text attachment run (system-role instruction, manual glossary, no thinking).",
        "config": {
            "model": "claude-sonnet-4-5",
            "api_key": "sk-ant-parity",
            "system_prompt_to_user": True,
            "enable_streaming": False,
            "enable_anthropic_thinking": True,
            "anthropic_thinking_budget": "8000",
        },
        "files": {
            "inputs/attachment.txt": "첫 번째 문단.\n\n두 번째 문단.\n",
            "inputs/direct_glossary.csv": _GLOSSARY_CSV,
        },
        "run_attrs": {
            "_input_output_run_active": True,
            "_direct_text_attachment_prompt": "Translate the attached chapter faithfully.",
            "_direct_text_attachment_prompt_role": "SYSTEM",
            "_direct_text_skip_prompt_profile": False,
            "_direct_text_output_mode": "text",
            "_direct_text_force_multipass_off": True,
            "_direct_text_use_manual_glossary": True,
            "_direct_text_manual_glossary_path": "<SANDBOX>/inputs/direct_glossary.csv",
            "_direct_text_force_no_glossary": False,
            "_direct_text_skip_thinking": True,
        },
        "input": "inputs/attachment.txt",
        "entries": DEFAULT_ENTRIES + ("direct_text_env",),
    },
    "metadata_only_batch": {
        "description": "Metadata-only batch run with a per-source output root and all fields disabled.",
        "config": {
            "model": "gpt-4.1-mini",
            "api_key": "sk-parity-meta",
            "auto_glossary_mode": "full",
            "translate_metadata_fields": {"description": False, "subject": False, "_per_epub": {}},
            "metadata_translation_mode": "individual",
            "translate_book_title": False,
        },
        "files": {"inputs/Metadata Book.epub": "PK"},
        "run_attrs": _metadata_run_attrs,
        "input": "inputs/Metadata Book.epub",
        "entries": DEFAULT_ENTRIES + ("metadata_only_env",),
    },
    "single_chapter_stream": {
        "description": "Library/Reader single-chapter run with forced streaming and a stale range.",
        "config": {
            "model": "gemini-2.5-flash",
            "api_key": "AIza-PARITY-SINGLE",
            "chapter_range": "3-9",
            "enable_streaming": False,
            "stream_thinking_logs": False,
        },
        "files": {"inputs/Reader Book.epub": "PK"},
        "run_attrs": {
            "_single_chapter_filter": "OEBPS/Text/chapter0005.xhtml",
            "_force_stream_all": True,
        },
        "input": "inputs/Reader Book.epub",
        "entries": DEFAULT_ENTRIES + ("forced_streaming_env",),
    },
    "subtitle_zip_bundle": {
        "description": "Extracted subtitle ZIP member with bundle output/work mappings.",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": "sk-parity-subs",
            "active_profile": "Subtitle Translation",
            "auto_glossary_mode": "single_pass",
            "single_pass_glossary_header_prompt": "Also list new names.",
        },
        "files": {
            "inputs/Show S01.zip": "PK",
            "work/Show S01/ep01.srt": "1\n00:00:01,000 --> 00:00:02,000\n안녕\n",
            "work/Show S01/ep02.srt": "1\n00:00:01,000 --> 00:00:02,000\n잘가\n",
        },
        "run_attrs": _subtitle_run_attrs,
        "input": "work/Show S01/ep01.srt",
        "entries": DEFAULT_ENTRIES,
    },
    "vertex_with_creds": {
        "description": "Vertex AI model with a service-account file, custom location, empty API key.",
        "config": {
            "model": "vertex/gemini-2.5-pro",
            "google_cloud_credentials": "<SANDBOX>/inputs/vertex_creds.json",
            "vertex_ai_location": "europe-west4",
            "use_gemini_openai_endpoint": True,
            "gemini_openai_endpoint": "",
        },
        "files": {
            "inputs/Vertex Book.epub": "PK",
            "inputs/vertex_creds.json": _CREDS_JSON,
        },
        "input": "inputs/Vertex Book.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "all_key_pools_enabled": {
        "description": "All 11 key pools enabled with fake keys (in-memory pools + env JSON).",
        "config": {
            "model": "gpt-4o",
            "api_key": "sk-parity-main",
            "auto_glossary_mode": "off_fuzzy_automap",
            "use_multi_api_keys": True,
            "multi_api_keys": [_key(1), _key(2, model="gemini-2.5-flash")],
            "force_key_rotation": False,
            "rotation_frequency": 3,
            "use_fallback_keys": True,
            "fallback_keys": [_key(3)],
            "fallback_key_shuffle": True,
            "use_main_key_fallback": False,
            "use_glossary_keys": True,
            "glossary_keys": [_key(4)],
            "use_glossary_refinement_keys": True,
            "glossary_refinement_keys": [_key(5)],
            "use_metadata_keys": True,
            "metadata_keys": [_key(6)],
            "use_qa_scan_keys": True,
            "qa_scan_keys": [_key(7)],
            "use_ai_truncation_detection_keys": True,
            "ai_truncation_detection_keys": [_key(8)],
            "use_rolling_summary_keys": True,
            "rolling_summary_keys": [_key(9)],
            "use_truncation_retry_keys": True,
            "truncation_retry_keys": [_key(10)],
            "use_inpainter_keys": True,
            "inpainter_keys": [_key(11, model="gpt-image-1")],
            "use_tts_keys": True,
            "tts_keys": [_key(12, model="gpt-4o-mini-tts")],
        },
        "files": {"inputs/Pool Book.epub": "PK"},
        "input": "inputs/Pool Book.epub",
        "entries": DEFAULT_ENTRIES,
    },
    "pdf_layout_custom_routes_ollama": {
        "description": "PDF input with layout options, legacy XHTML render-mode migration, custom prefix routes, Ollama settings.",
        "config": {
            "model": "lan/qwen3:8b",
            "api_key": "sk-local-anything",
            "auto_glossary_mode": "off_no_automap",
            "use_custom_openai_endpoint": True,
            "openai_base_url": "http://192.168.1.20:11434/v1",
            "custom_prefix_routes": [
                {"prefix": "lan", "routing": "http://192.168.1.20:11434/v1/", "endpoint_type": "openai_chat"},
                {"prefix": "img/", "routing": "https://images.example.test", "endpoint_type": "openai-images"},
                {"prefix": "bad/", "routing": "ftp://not-http"},
                {"prefix": "LAN/", "routing": "http://duplicate.example.test"},
                {"prefix": "anth", "base_url": "https://anthropic.example.test", "type": "{base_url}/v1/messages"},
            ],
            "ollama_settings": {"auto_update": False, "models": {"qwen3:8b": {"num_ctx": 8192}}},
            "pdf_render_mode": "xhtml",
            "pdf_paragraph_alignment": "Centre",
            "pdf_header_alignment": "RIGHT",
            "pdf_paragraph_justification": "justified",
            "pdf_rtl_paragraph_layout": True,
            "pdf_extraction_workers": "2",
            "pdf_use_toc_sections": False,
            "enable_pdf_output": True,
            "pdf_generate_toc": True,
            "text_extraction_method": "standard",
            "file_filtering_level": "full",
            "use_toc_ncx": True,
            "batch_translate_headers": True,
        },
        "files": {"inputs/Layout Doc.pdf": "%PDF-1.4\n"},
        "input": "inputs/Layout Doc.pdf",
        "entries": DEFAULT_ENTRIES,
    },
}

for _name, _scenario in SCENARIOS.items():
    _scenario["name"] = _name

SCENARIO_NAMES = tuple(SCENARIOS)


def get(name: str) -> dict:
    return SCENARIOS[name]
