import ast
import copy
import inspect
import json
import os
from pathlib import Path
import textwrap
import threading
import types
import zipfile

import pytest
from bs4 import BeautifulSoup

from chapter_chunk_progress import (
    chunk_entry_needs_translation,
    effective_parent_status,
    ensure_chunk_entry_schema,
    extract_marked_chunks,
    remove_chunk_segments,
    remove_chunk_segments_from_file,
    reset_chunks_for_retranslation,
    wrap_chunk_html,
)
from chapter_display_numbering import (
    filename_chapter_number,
    nonreset_chapter_display_numbers,
)
from Retranslation_GUI import (
    RetranslationMixin,
    _merge_and_write_retranslation_progress,
    _clear_llm_token_qa_markers,
    _clear_missing_image_qa_markers,
    _clear_refinement_progress_fields,
    _combine_glossary_progress_legend_stats,
    _derive_glossary_refinement_aggregate_status,
    _filter_glossary_source_chapter_map,
    _find_matching_glossary_refinement_aggregate,
    _glossary_progress_filename_keys,
    _glossary_refinement_row_detail,
    _glossary_refinement_type_key,
    _index_epub_html_members,
    _match_epub_html_member_basename,
    _map_zero_based_glossary_progress_index,
    _merge_glossary_refinement_row_info,
    _normalize_glossary_refinement_selection,
    _normalize_progress_match_name,
    _parallel_glossary_progress_filename_aliases,
    _persist_progress_manager_source_link,
    _progress_entry_has_llm_token_qa,
    _progress_entry_has_missing_image_qa,
    _progress_entry_is_completed_image_only_for_display,
    _progress_entry_has_meaningful_tts_state,
    _progress_path_signature,
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _progress_status_hides_model_for_display,
    _progress_item_is_html,
    _repair_empty_attribute_qa_file,
    _resolve_dialog_window_parent,
    _snapshot_progress_output_dir,
    _select_progress_entry_for_display,
)
from TransateKRtoEN import (
    ContentProcessor,
    ProgressManager,
    TranslationConfig,
    _assign_translation_display_chapter_numbers,
    _chapter_log_number,
    _vision_ocr_header_markdown,
)
from image_translator import ImageTranslator
from scan_html_folder import update_progress_file
from unified_api_client import UnifiedClient, set_current_thread_actual_request_model
from extract_glossary_from_epub import (
    _confirmed_merged_child_indices,
    _glossary_chapter_display_number_map,
    _glossary_watchdog_request_label,
    _glossary_is_hard_stop_requested,
    _graceful_stop_should_drain_after_result,
    _is_graceful_stop_skip_error,
    main as extract_glossary_main,
    make_glossary_progress_context,
    _restore_glossary_in_progress_file,
    save_progress as save_glossary_progress,
)
import extract_glossary_from_epub as glossary_extractor
import Retranslation_GUI as retranslation_gui_module


@pytest.fixture(autouse=True)
def _clear_actual_request_metadata():
    set_current_thread_actual_request_model(None, None)
    yield
    set_current_thread_actual_request_model(None, None)


def test_chapter_display_numbers_do_not_reset_after_positive_sequence():
    assert nonreset_chapter_display_numbers([0, 1, 2, 0, 1, 0]) == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_chapter_display_numbers_keep_leading_zeroes_and_forward_jumps():
    assert nonreset_chapter_display_numbers([0, 0, 5, 6, 2, 3, 10]) == [
        0,
        0,
        5,
        6,
        7,
        8,
        10,
    ]


def test_progress_rematch_does_not_scan_duplicate_split_number_buckets(
    tmp_path,
    monkeypatch,
):
    gui = RetranslationMixin()
    gui.config = {}
    row_count = 600
    spine_chapters = []
    progress_chapters = {}
    for index in range(row_count):
        split_suffix = index % 2
        spine_name = f"part{index:04d}_split_{split_suffix:03d}.html"
        spine_chapters.append({
            "filename": spine_name,
            "file_chapter_num": split_suffix,
            "display_chapter_num": index,
            "position": index,
            "is_special": False,
            "status": "unknown",
            "output_file": None,
        })
        progress_chapters[f"orphan-{index}"] = {
            "actual_num": split_suffix,
            "original_basename": (
                f"unrelated{index:04d}_split_{split_suffix:03d}.html"
            ),
            "output_file": (
                f"response_unrelated{index:04d}_split_"
                f"{split_suffix:03d}.html"
            ),
            "status": "completed",
        }

    original_normalize = (
        retranslation_gui_module._normalize_progress_match_name
    )
    normalize_calls = 0

    def counted_normalize(value):
        nonlocal normalize_calls
        normalize_calls += 1
        return original_normalize(value)

    monkeypatch.setattr(
        retranslation_gui_module,
        "_normalize_progress_match_name",
        counted_normalize,
    )
    data = {
        "prog": {"chapters": progress_chapters},
        "output_dir": str(tmp_path),
        "progress_file": str(tmp_path / "translation_progress.json"),
        "file_path": str(tmp_path / "book.epub"),
        "spine_chapters": spine_chapters,
        "_refresh_read_only": True,
        "_prefetched_output_listing": set(),
    }

    gui._rematch_spine_chapters(data)

    # Linear index construction plus constant-time failed lookups. The former
    # raw-number fallback performed hundreds of thousands of comparisons here.
    assert normalize_calls < row_count * 20
    assert all(
        chapter["status"] == "not_translated"
        for chapter in spine_chapters
    )


def test_text_progress_ignores_inert_no_tts_placeholders(tmp_path):
    assert not _progress_entry_has_meaningful_tts_state({})
    assert not _progress_entry_has_meaningful_tts_state({"tts_status": "no_tts"})
    assert _progress_entry_has_meaningful_tts_state({"tts_status": "in_progress"})
    assert _progress_entry_has_meaningful_tts_state({"tts_file": "chapter.mp3"})

    gui = RetranslationMixin()
    gui.config = {}
    gui._existing_audio_for_entry = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("text-mode no_tts entries must not enter audio reconciliation")
    )
    data = {
        "output_dir": str(tmp_path),
        "prog": {
            "output_mode": "text",
            "chapters": {
                str(index): {
                    "output_file": f"chapter{index:04d}.html",
                    "tts_status": "no_tts",
                }
                for index in range(1_000)
            },
        },
    }

    assert gui._reconcile_tts_audio_files(data) is False


def test_sdlxliff_streamed_numbering_does_not_rewalk_prior_pieces():
    source = inspect.getsource(
        retranslation_gui_module.SDLXLIFFReviewDialog
        ._append_generated_sidecar_stream_piece
    )

    assert "nonreset_chapter_display_numbers" not in source


def test_filename_chapter_number_preserves_raw_identity_rules():
    assert filename_chapter_number("Text/part_0042.xhtml") == 42
    assert filename_chapter_number("Text/info.xhtml") == 0
    assert filename_chapter_number("Text/notice0042.xhtml", is_special=True) == 0


def test_glossary_log_numbers_use_same_nonreset_sequence_as_progress_views():
    filenames = {
        0: "part0000.html",
        1: "part0001.html",
        2: "part0002.html",
        3: "part0003_split_000.html",
        4: "part0003_split_001.html",
        5: "part0004_split_000.html",
    }
    positions = {idx: idx + 1 for idx in filenames}

    assert _glossary_chapter_display_number_map(filenames, positions) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    }


def test_translation_log_numbers_use_same_nonreset_sequence_as_progress_views():
    chapters = [
        {"original_basename": "part0000.html", "actual_chapter_num": 0},
        {"original_basename": "part0001.html", "actual_chapter_num": 1},
        {"original_basename": "part0002.html", "actual_chapter_num": 2},
        {"original_basename": "part0003_split_000.html", "actual_chapter_num": 0},
        {"original_basename": "part0003_split_001.html", "actual_chapter_num": 1},
        {"original_basename": "part0004_split_000.html", "actual_chapter_num": 0},
    ]

    assert _assign_translation_display_chapter_numbers(chapters) == [0, 1, 2, 3, 4, 5]
    assert [_chapter_log_number(chapter) for chapter in chapters] == [0, 1, 2, 3, 4, 5]
    assert [chapter["actual_chapter_num"] for chapter in chapters] == [0, 1, 2, 0, 1, 0]

    translation_source = (
        Path(__file__).resolve().parents[1] / "src" / "TransateKRtoEN.py"
    ).read_text(encoding="utf-8")
    assert 'f"💬 {_term} {log_num}: Chunk ' in translation_source
    assert 'current_request_label = f"Chapter {log_num} ' in translation_source
    assert '"📝 Queued image-only <title> translation for "' in translation_source
    assert 'f"📝 Image-only chapter {log_num}: queued its "' not in translation_source
    assert translation_source.count("'chapter': log_num") >= 2
    assert "'chapter': parent_log_num" in translation_source


def test_sequential_translation_assigns_log_number_before_first_use():
    source_file = inspect.getsourcefile(ProgressManager)
    translation_source = Path(source_file).read_text(encoding="utf-8")
    sequential_start = translation_source.index("# Second pass: process chapters")
    assignment = translation_source.index(
        "log_num = _chapter_log_number(c, actual_num)",
        sequential_start,
    )
    first_use = translation_source.index(
        "Output file missing for chapter {log_num}",
        sequential_start,
    )

    assert assignment < first_use


def test_all_requested_chapter_views_use_shared_nonreset_numbering():
    source_root = Path(__file__).resolve().parents[1] / "src"
    progress_source = (source_root / "Retranslation_GUI.py").read_text(
        encoding="utf-8"
    )
    reader_source = (source_root / "epub_library.py").read_text(
        encoding="utf-8"
    )

    assert "chapter['display_chapter_num'] = display_number" in progress_source
    assert "panel_state['_chapter_display_numbers']" in progress_source
    assert 'metadata["display_chapter_num"] = display_chapter_num' in progress_source
    assert "self._chapter_display_numbers = nonreset_chapter_display_numbers(" in reader_source


def test_translation_progress_displays_nonreset_number_without_mutating_raw_num():
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    row = {
        "num": 0,
        "display_num": 3,
        "status": "not_translated",
        "output_file": "response_part_b000.html",
        "original_filename": "part_b000.xhtml",
        "opf_position": 3,
        "info": {},
    }

    display, status = mixin._progress_list_display_text(
        row,
        {"show_model_info_state": False, "prog": {"chapters": {}}},
        20,
        25,
    )

    assert row["num"] == 0
    assert status == "not_translated"
    assert "[004] Ch.003" in display


def test_pending_progress_rows_hide_model_metadata_in_both_progress_views():
    assert _progress_status_hides_model_for_display("pending")
    assert _progress_status_hides_model_for_display("not_translated")
    assert _progress_status_hides_model_for_display("not completed")
    assert not _progress_status_hides_model_for_display("in_progress")
    assert not _progress_status_hides_model_for_display("completed")

    mixin = RetranslationMixin.__new__(RetranslationMixin)
    row = {
        "num": 3,
        "status": "not_translated",
        "output_file": "response_part0003.html",
        "original_filename": "part0003.xhtml",
        "opf_position": 3,
        "info": {
            "status": "pending",
            "previous_progress_entry": {
                "status": "completed",
                "model_name": "old-provider/model",
            },
        },
    }

    display, status = mixin._progress_list_display_text(
        row,
        {"show_model_info_state": True, "prog": {"chapters": {}}},
        20,
        25,
    )

    assert status == "not_translated"
    assert "part0003.xhtml" in display
    assert "old-provider/model" not in display
    assert "(model unknown)" not in display
    assert " -> " not in display

    glossary_source = Path(retranslation_gui_module.__file__).read_text(
        encoding="utf-8"
    )
    glossary_start = glossary_source.index("def _gp_display_for(")
    glossary_end = glossary_source.index(
        "def _gp_refinement_rows(", glossary_start
    )
    glossary_display_source = glossary_source[glossary_start:glossary_end]
    assert "hide_model = _progress_status_hides_model_for_display(status)" in glossary_display_source
    assert "if status in skipped_labels or hide_model:" in glossary_display_source


def test_glossary_progress_reactivates_a_manually_removed_chapter(tmp_path):
    progress_file = tmp_path / "book_glossary_progress.json"
    progress_file.write_text(
        json.dumps(
            {
                "chapters": {},
                "completed": [],
                "failed": [],
                "merged_indices": [],
                "in_progress": [],
                "manual_removed_indices": [0, 1],
                "manual_removed_session_id": glossary_extractor._GLOSSARY_PROGRESS_SESSION_ID,
                "progress_session_id": glossary_extractor._GLOSSARY_PROGRESS_SESSION_ID,
            }
        ),
        encoding="utf-8",
    )
    context = make_glossary_progress_context(
        progress_file=str(progress_file),
        output_file=str(tmp_path / "book_glossary.json"),
        chapter_positions={0: 1, 1: 2},
        chapter_numbers={0: 1, 1: 2},
        chapter_filenames={0: "pair_0001.xhtml", 1: "pair_0002.xhtml"},
        total_chapters=2,
    )

    save_glossary_progress([], [], [], failed=[], in_progress=[0], context=context)

    active = json.loads(progress_file.read_text(encoding="utf-8"))
    assert active["in_progress"] == [0]
    assert active["chapters"]["0"]["status"] == "in_progress"
    assert active["manual_removed_indices"] == [1]

    save_glossary_progress([0], [], [], failed=[], in_progress=[], context=context)

    completed = json.loads(progress_file.read_text(encoding="utf-8"))
    assert completed["completed"] == [0]
    assert completed["chapters"]["0"]["status"] == "completed"
    assert completed["manual_removed_indices"] == [1]


@pytest.mark.parametrize(
    ("configured_type", "section_type"),
    [
        ("concept", "concepts"),
        ("equipment", "equipments"),
        ("item", "items"),
        ("skill", "skills"),
        ("category", "categories"),
        ("class", "classes"),
    ],
)
def test_refinement_progress_matches_plural_section_headings(
    configured_type,
    section_type,
):
    assert _glossary_refinement_type_key(configured_type) == (
        _glossary_refinement_type_key(section_type)
    )


def test_live_refinement_entries_invalidate_saved_no_entries_state():
    merged = _merge_glossary_refinement_row_info(
        {
            "entry_type": "concept",
            "status": "not_refined",
            "entry_count_before": 181,
            "current_entry_count": 181,
            "entry_count_after": None,
            "reason": "",
        },
        {
            "entry_type": "concept",
            "status": "skipped",
            "entry_count_before": 0,
            "entry_count_after": 0,
            "reason": "no_entries",
        },
    )

    assert merged["status"] == "not_refined"
    assert merged["entry_count_before"] == 181
    assert merged["current_entry_count"] == 181
    assert merged["entry_count_after"] is None
    assert merged["reason"] == ""
    assert merged["_has_saved_progress"] is False


@pytest.mark.parametrize(
    ("info", "status", "expected"),
    [
        ({"current_entry_count": 181}, "not_refined", " | 181 entries"),
        ({"current_entry_count": 1}, "not_refined", " | 1 entry"),
        ({"current_entry_count": 0}, "skipped", " | 0 entries"),
        (
            {
                "current_entry_count": 26,
                "completed_chunks": 0,
                "total_chunks": 2,
            },
            "in_progress",
            " | 26 entries | chunks 0/2",
        ),
        (
            {
                "current_entry_count": 20,
                "entry_count_before": 26,
                "entry_count_after": 20,
            },
            "completed",
            " | 20 entries | refined 26 -> 20",
        ),
    ],
)
def test_refinement_rows_always_show_live_entry_counts(info, status, expected):
    assert _glossary_refinement_row_detail(info, status) == expected


def test_glossary_progress_persists_structural_skips_separately(tmp_path):
    progress_file = tmp_path / "book_glossary_progress.json"
    context = make_glossary_progress_context(
        progress_file=str(progress_file),
        output_file=str(tmp_path / "book_glossary.json"),
        chapter_positions={0: 1, 1: 2, 2: 3},
        chapter_numbers={0: 1, 1: 2, 2: 3},
        chapter_filenames={
            0: "image.xhtml",
            1: "heading.xhtml",
            2: "empty.xhtml",
        },
        total_chapters=3,
        chapter_status_overrides={
            0: "skipped_image_only",
            1: "skipped_title_header_only",
            2: "skipped_empty",
        },
    )

    progress_file.write_text(
        json.dumps(
            {
                "chapters": {},
                "completed": [],
                "skipped": [],
                "failed": [],
                "merged_indices": [],
                "in_progress": [],
                "manual_removed_indices": [0],
                "manual_removed_session_id": glossary_extractor._GLOSSARY_PROGRESS_SESSION_ID,
            }
        ),
        encoding="utf-8",
    )

    save_glossary_progress([0, 1, 2], [], [], context=context)

    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["completed"] == []
    assert progress["skipped"] == [0, 1, 2]
    assert progress.get("manual_removed_indices", []) == []
    assert {
        key: info["status"]
        for key, info in progress["chapters"].items()
    } == {
        "0": "skipped_image_only",
        "1": "skipped_title_header_only",
        "2": "skipped_empty",
    }
    assert {
        info["model_name"] for info in progress["chapters"].values()
    } == {"SKIPPED"}

    reloaded = glossary_extractor.load_progress(context=context)
    assert reloaded["completed"] == []
    assert reloaded["skipped"] == [0, 1, 2]


def test_glossary_structural_skip_ui_and_default_toggle_are_wired():
    source_root = Path(__file__).resolve().parents[1] / "src"
    glossary_gui = (source_root / "GlossaryManager_GUI.py").read_text(
        encoding="utf-8"
    )
    translator_gui = (source_root / "translator_gui.py").read_text(
        encoding="utf-8"
    )
    progress_gui = (source_root / "Retranslation_GUI.py").read_text(
        encoding="utf-8"
    )

    assert '"Skip title/header-only chapters"' in glossary_gui
    assert "self.config.get('glossary_skip_title_header_only', True)" in glossary_gui
    assert "settings_grid.addWidget(\n            self.glossary_skip_title_header_only_checkbox" in glossary_gui
    assert "self.glossary_skip_title_header_only_checkbox,\n            2,\n            2," in glossary_gui
    assert "'GLOSSARY_SKIP_TITLE_HEADER_ONLY'" in translator_gui
    assert "'glossary_skip_title_header_only', True" in translator_gui
    assert "'skipped_image_only': 'Image Only (Skipped)'" in progress_gui
    assert "'skipped_title_header_only': 'Title/Header Only (Skipped)'" in progress_gui
    assert 'lbl_gp_skipped = QLabel(f"⏭️ Skipped:' in progress_gui
    assert "lbl_gp_skipped.setVisible(True)" in progress_gui
    assert "width_ratio=0.39" in progress_gui
    assert "gp_stats_font = QFont('Arial', 9)" in progress_gui
    assert "'entries_by_ci': entries_by_ci" in progress_gui
    assert "cached_entries = cache.get('entries_by_ci', {}).get(ci, [])" in progress_gui
    assert "'skipped': len(_skip)" in progress_gui
    assert "if status in skipped_labels or hide_model:" in progress_gui


def test_glossary_progress_does_not_infer_cover_as_completed():
    progress_gui = (
        Path(__file__).resolve().parents[1] / "src" / "Retranslation_GUI.py"
    ).read_text(encoding="utf-8")

    assert "_gp_auto_completed_indices" not in progress_gui
    assert "panel_state['_auto_completed']" not in progress_gui


def test_glossary_refinement_watchdog_label_is_not_taken_from_prompt_content():
    prompt = "Refine this glossary entry whose description mentions Chapter 29."
    label = _glossary_watchdog_request_label(
        "glossary_refinement",
        chunk_idx=1,
        total_chunks=1,
    )

    assert "Chapter 29" in prompt
    assert label == "Glossary Refinement"
    assert _glossary_watchdog_request_label("glossary", 1, 1) is None
    assert _glossary_watchdog_request_label("glossary_refinement", 2, 4) == (
        "Glossary Refinement (chunk 2/4)"
    )


def test_remove_refinement_status_clears_current_and_restorable_state():
    entry = {
        "status": "completed",
        "refinement_status": "refined",
        "refined_at": 123.0,
        "refinement_error": "old failure",
        "unrefined_backup_file": "_unrefined/chapter.xhtml",
        "previous_progress_entry": {
            "status": "completed",
            "refinement_status": "refined",
            "refined_at": 122.0,
            "unrefined_backup_file": "_unrefined/older.xhtml",
        },
    }

    removed = _clear_refinement_progress_fields(entry)

    assert removed == 7
    for field in (
        "refinement_status",
        "refined_at",
        "refinement_error",
        "unrefined_backup_file",
    ):
        assert field not in entry
        assert field not in entry["previous_progress_entry"]


def test_progress_context_menu_places_remove_refinement_after_remove_qa():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "Retranslation_GUI.py"
    ).read_text(encoding="utf-8")
    start = source.index("        def show_context_menu(pos):")
    end = source.index(
        "        listbox.customContextMenuRequested.connect(show_context_menu)",
        start,
    )
    menu_block = source[start:end]

    assert menu_block.index("Remove QA Failed Mark") < menu_block.index(
        "Remove refinement status"
    )
    assert "remove_refinement_status()" in menu_block


def test_progress_context_menu_resolves_llm_token_qa_locally():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "Retranslation_GUI.py"
    ).read_text(encoding="utf-8")
    start = source.index("        def show_context_menu(pos):")
    end = source.index(
        "        listbox.customContextMenuRequested.connect(show_context_menu)",
        start,
    )
    menu_block = source[start:end]

    assert "has_raw_foreign_text_qa or has_llm_token_qa" in menu_block
    assert 'menu.addAction("⚠️ Resolve QA issue")' in menu_block
    assert "_resolve_llm_token_qa_issue(" in menu_block
    assert "_show_llm_token_repair_comparison(" in source
    assert "Before — malformed LLM token tag" in source
    assert "After — safe visible text in HTML" in source


def test_llm_token_qa_detection_and_marker_cleanup_preserve_other_issues():
    entry = {
        "status": "qa_failed",
        "qa_issues": True,
        "qa_issues_found": [
            'LLM_token_issue: \'<a and="" classes="">\'',
            "missing_images: 1",
        ],
        "qa_timestamp": 123.0,
    }

    assert _progress_entry_has_llm_token_qa(entry)
    changed, remaining = _clear_llm_token_qa_markers(entry)

    assert changed is True
    assert remaining is True
    assert entry["status"] == "qa_failed"
    assert entry["qa_issues_found"] == ["missing_images: 1"]

    entry["qa_issues_found"] = ["LLM_token_issue_1_found"]
    changed, remaining = _clear_llm_token_qa_markers(entry)
    assert changed is True
    assert remaining is False
    assert entry["status"] == "completed"
    assert "qa_issues_found" not in entry
    assert "qa_timestamp" not in entry


def test_missing_image_marker_cleanup_preserves_unrelated_qa_issues():
    entry = {
        "status": "qa_failed",
        "qa_issues": True,
        "qa_issues_found": [
            "missing_images_5_lost_(0/5)",
            "unwrapped_text_content_2_found",
            {"type": "Japanese_text_found_2_chars", "count": 2},
        ],
        "qa_issue_previews": {
            "missing_images_5_lost_(0/5)": ["cover.jpg"],
            "unwrapped_text_content_2_found": [
                "The prose discusses missing images but is a different issue."
            ],
        },
        "failure_reason": "Independent refinement failure",
        "error_message": "Independent API error",
        "qa_timestamp": 123.0,
    }

    assert _progress_entry_has_missing_image_qa(entry) is True
    changed, remaining = _clear_missing_image_qa_markers(entry)

    assert changed is True
    assert remaining is True
    assert entry["status"] == "qa_failed"
    assert entry["qa_issues"] is True
    assert entry["qa_issues_found"] == [
        "unwrapped_text_content_2_found",
        {"type": "Japanese_text_found_2_chars", "count": 2},
    ]
    assert entry["qa_issue_previews"] == {
        "unwrapped_text_content_2_found": [
            "The prose discusses missing images but is a different issue."
        ]
    }
    assert entry["failure_reason"] == "Independent refinement failure"
    assert entry["error_message"] == "Independent API error"
    assert entry["qa_timestamp"] == 123.0


def test_missing_image_marker_cleanup_completes_entry_when_it_was_only_issue():
    entry = {
        "status": "qa_failed",
        "qa_issues": True,
        "qa_issues_found": [
            {"type": "missing_images", "count": 5, "missing": ["cover.jpg"]}
        ],
        "qa_issue_previews": {
            "missing_images_5_lost_(0/5)": ["cover.jpg"]
        },
        "qa_timestamp": 123.0,
    }

    changed, remaining = _clear_missing_image_qa_markers(entry)

    assert changed is True
    assert remaining is False
    assert entry["status"] == "completed"
    assert "qa_issues" not in entry
    assert "qa_issues_found" not in entry
    assert "qa_issue_previews" not in entry
    assert "qa_timestamp" not in entry


def test_insert_missing_image_context_action_uses_targeted_qa_cleanup():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "Retranslation_GUI.py"
    ).read_text(encoding="utf-8")
    start = source.index("            elif act_insert_img and chosen == act_insert_img:")
    end = source.index(
        "            elif act_restore_in_progress and chosen == act_restore_in_progress:",
        start,
    )
    action_block = source[start:end]

    assert "_clear_missing_image_qa_markers(target)" in action_block
    assert "Other QA issues remain." in action_block
    assert "Images restored and QA flags cleared." not in action_block


def test_emergency_image_restore_corrects_nearby_header_estimation_error():
    long_translated_title = "Translated chapter title " * 8
    source = (
        '<html><body><h1>A</h1><p><img src="cover.jpg"/></p>'
        '<p>Source body.</p></body></html>'
    )
    translated = (
        f'<html><body><h1>{long_translated_title}</h1>'
        '<p>Translated body.</p></body></html>'
    )

    restored = ContentProcessor.emergency_restore_images(
        translated, source, verbose=False
    )

    assert restored.index('</h1>') < restored.index('<img')
    assert restored.index('<img') < restored.index('Translated body.')


def test_emergency_image_restore_preserves_author_image_above_first_header():
    source = (
        '<html><body><p><img src="cover.jpg"/></p>'
        '<h1>Chapter title</h1><p>Source body.</p></body></html>'
    )
    translated = (
        '<html><body><h1>Chapter title</h1>'
        '<p>Translated body.</p></body></html>'
    )

    restored = ContentProcessor.emergency_restore_images(
        translated, source, verbose=False
    )

    assert restored.index('<img') < restored.index('<h1>')


def test_emergency_image_restore_can_insert_before_second_header():
    source = (
        '<html><body><h1>Book title</h1><p>Introduction.</p>'
        '<p><img src="scene.jpg"/></p>'
        '<h2>Section title</h2><p>Source body.</p></body></html>'
    )
    translated = (
        '<html><body><h1>Book title</h1><p>Translated introduction.</p>'
        '<h2>Section title</h2><p>Translated body.</p></body></html>'
    )

    restored = ContentProcessor.emergency_restore_images(
        translated, source, verbose=False
    )

    assert restored.index('</h1>') < restored.index('<img')
    assert restored.index('<img') < restored.index('<h2>')


def test_emergency_image_restore_respects_source_position_when_estimate_is_far():
    preface = "A" * 240
    source = (
        '<html><body><h1>Chapter title</h1>'
        '<p><img src="cover.jpg"/></p><p>Source body.</p></body></html>'
    )
    translated = (
        f'<html><body><p>{preface}</p><h1>Chapter title</h1>'
        '<p>Translated body.</p></body></html>'
    )

    restored = ContentProcessor.emergency_restore_images(
        translated, source, verbose=False
    )

    assert restored.index('</h1>') < restored.index('<img')


def test_repair_empty_attribute_qa_file_uses_shared_llm_token_fix(tmp_path):
    output = tmp_path / "response_chapter.html"
    output.write_text(
        '<p>Before</p><a aesthetics="" attending="" can=""></a><p>After</p>',
        encoding="utf-8",
    )

    result = _repair_empty_attribute_qa_file(output)

    assert result == {
        "resolved": True,
        "changed": True,
        "repaired": 1,
        "remaining": 0,
        "repairs": [
            {
                "before": '<a aesthetics="" attending="" can="">',
                "after": "&lt;a aesthetics attending can&gt;",
            }
        ],
        "error": "",
    }
    repaired = output.read_text(encoding="utf-8")
    assert '&lt;a aesthetics attending can&gt;' in repaired
    assert 'aesthetics=""' not in repaired


def test_cleanup_missing_files_uses_one_directory_snapshot(tmp_path, monkeypatch):
    (tmp_path / "chapter_keep.html").write_text("keep", encoding="utf-8")
    (tmp_path / "chapter_renamed.xhtml").write_text("renamed", encoding="utf-8")

    progress = ProgressManager(str(tmp_path))
    progress.prog = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "output_file": "chapter_keep.html",
                "status": "completed",
            },
            "2": {
                "actual_num": 2,
                "output_file": "response_chapter_renamed.html",
                "status": "completed",
            },
            "3": {
                "actual_num": 3,
                "output_file": "chapter_missing.html",
                "status": "completed",
                "merged_chapters": [4],
            },
            "4": {
                "actual_num": 4,
                "output_file": "chapter_missing.html",
                "status": "merged",
                "merged_parent_chapter": 3,
            },
            "5": {
                "actual_num": 5,
                "output_file": "failed_missing.html",
                "status": "failed",
            },
            "6": {
                "actual_num": 6,
                "output_file": "pending_missing.html",
                "status": "pending_retry",
            },
        },
        "chapter_chunks": {"3": {"stale": True}},
        "version": "2.1",
    }

    real_listdir = os.listdir
    listdir_calls = []

    def counted_listdir(path):
        listdir_calls.append(path)
        return real_listdir(path)

    real_exists = os.path.exists
    exists_calls = []

    def counted_exists(path):
        exists_calls.append(path)
        return real_exists(path)

    monkeypatch.setattr(os, "listdir", counted_listdir)
    monkeypatch.setattr(os.path, "exists", counted_exists)

    progress.cleanup_missing_files(str(tmp_path))

    assert listdir_calls == [str(tmp_path)]
    assert exists_calls == []
    assert progress.prog["chapters"]["1"]["output_file"] == "chapter_keep.html"
    assert progress.prog["chapters"]["2"]["output_file"] == "chapter_renamed.xhtml"
    assert "3" not in progress.prog["chapters"]
    assert "4" not in progress.prog["chapters"]
    assert "3" not in progress.prog["chapter_chunks"]
    assert "5" in progress.prog["chapters"]
    assert "6" in progress.prog["chapters"]


def _epub_chunk_budget(initial=5750, cached=4250):
    return {
        "initial_output_token_limit": 12000,
        "cached_output_token_limit": 9000,
        "compression_factor": 2.0,
        "safety_margin": 500,
        "minimum_chunk_size": 1000,
        "initial_chunk_size": initial,
        "cached_chunk_size": cached,
    }


def test_chunk_progress_reuses_matching_budget_and_normalizes_indices(tmp_path):
    progress = ProgressManager(str(tmp_path))
    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )
    assert reason is None
    assert changed is True
    assert progress.record_chapter_chunk(
        "hash-1",
        2,
        3,
        "translated two",
        _epub_chunk_budget(),
        source_text="source two",
    )

    entry, reason, _changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )

    assert reason is None
    assert entry["schema_version"] == 2
    assert entry["completed"] == [2]
    assert entry["chunks"] == {"2": "translated two"}
    assert entry["entries"]["2"]["status"] == "completed"
    assert entry["entries"]["2"]["source"] == "source two"


def test_chunk_progress_migrates_v1_cache_to_selectable_v2_entries(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapter_chunks"]["hash-1"] = {
        "schema_version": 1,
        "total": 2,
        "completed": [1],
        "chunks": {"1": "cached one"},
        "chunk_metadata": {
            "1": {"model_name": "model-a", "key_identifier": "key-a"}
        },
        "chapter_status": "incomplete",
        **_epub_chunk_budget(),
    }

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )

    assert reason is None
    assert changed is True
    assert entry["schema_version"] == 2
    assert entry["chunks"] == {"1": "cached one"}
    assert entry["completed"] == [1]
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["1"]["model_name"] == "model-a"
    assert entry["entries"]["2"]["status"] == "pending"


def test_selective_chunk_reset_removes_only_its_html_segment_and_cache(
    tmp_path,
):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )
    for index in (1, 2, 3):
        progress.record_chapter_chunk(
            "hash-1",
            index,
            3,
            f"<p>translated {index}</p>",
            _epub_chunk_budget(),
            source_text=f"<p>source {index}</p>",
        )
    html = "\n".join(
        wrap_chunk_html(
            "hash-1", index, 3, f"<p>translated {index}</p>"
        )
        for index in (1, 2, 3)
    )
    entry = progress.prog["chapter_chunks"]["hash-1"]

    updated_html, removed = remove_chunk_segments(
        html, "hash-1", [2], entry
    )
    reset = progress.reset_chapter_chunks("hash-1", [2])

    assert removed == [2]
    assert reset == [2]
    assert sorted(extract_marked_chunks(updated_html, "hash-1")) == [1, 3]
    assert entry["completed"] == [1, 3]
    assert entry["entries"]["2"]["status"] == "pending"
    assert progress.get_reusable_chapter_chunks("hash-1") == {
        "1": "<p>translated 1</p>",
        "3": "<p>translated 3</p>",
    }


def test_chunk_removal_accepts_one_unambiguous_stale_pdf_marker_plan():
    entry = {
        "schema_version": 2,
        "total": 3,
        "chunks": {
            "1": "<p>translated 1</p>",
            "2": "<p>translated 2</p>",
            "3": "<p>translated 3</p>",
        },
    }
    html = "\n".join(
        wrap_chunk_html(
            "marker-key-before-pdf-normalization",
            index,
            3,
            f"<p>translated {index}</p>",
        )
        for index in (1, 2, 3)
    )

    updated_html, removed = remove_chunk_segments(
        html,
        "current-pdf-content-hash",
        [2],
        entry,
    )

    assert removed == [2]
    remaining = extract_marked_chunks(updated_html)
    assert sorted(remaining) == [1, 3]
    assert "translated 2" not in updated_html


def test_chunk_removal_strips_saved_code_fences_for_result_fallback():
    entry = {
        "schema_version": 2,
        "total": 2,
        "chunks": {
            "2": "```html\n<p>translated 2</p>\n```",
        },
    }

    updated_html, removed = remove_chunk_segments(
        "<p>translated 1</p>\n<p>translated 2</p>",
        "pdf-section-hash",
        [2],
        entry,
    )

    assert removed == [2]
    assert "translated 1" in updated_html
    assert "translated 2" not in updated_html


def test_stale_marker_fallback_rejects_multi_chapter_compiled_html():
    html = "\n".join((
        wrap_chunk_html("chapter-one", 1, 2, "<p>one-a</p>"),
        wrap_chunk_html("chapter-one", 2, 2, "<p>one-b</p>"),
        wrap_chunk_html("chapter-two", 1, 2, "<p>two-a</p>"),
        wrap_chunk_html("chapter-two", 2, 2, "<p>two-b</p>"),
    ))

    updated_html, removed = remove_chunk_segments(
        html,
        "unknown-chapter-key",
        [2],
        {"schema_version": 2, "total": 2, "chunks": {}},
    )

    assert removed == []
    assert updated_html == html


def test_selecting_every_chunk_deletes_the_response_html(tmp_path):
    output_path = tmp_path / "response_pdf_section_002.html"
    output_path.write_text(
        "\n".join(
            wrap_chunk_html(
                "pdf-section-hash",
                index,
                3,
                f"<p>translated {index}</p>",
            )
            for index in (1, 2, 3)
        ),
        encoding="utf-8",
    )
    entry = {
        "schema_version": 2,
        "total": 3,
        "chunks": {
            str(index): f"<p>translated {index}</p>"
            for index in (1, 2, 3)
        },
    }

    removed, file_deleted = remove_chunk_segments_from_file(
        str(output_path),
        "pdf-section-hash",
        [1, 2, 3],
        entry,
    )

    assert removed == [1, 2, 3]
    assert file_deleted is True
    assert not output_path.exists()


def test_selecting_some_chunks_rewrites_instead_of_deleting_response_html(
        tmp_path):
    output_path = tmp_path / "response_pdf_section_002.html"
    output_path.write_text(
        "\n".join(
            wrap_chunk_html(
                "pdf-section-hash",
                index,
                3,
                f"<p>translated {index}</p>",
            )
            for index in (1, 2, 3)
        ),
        encoding="utf-8",
    )
    entry = {
        "schema_version": 2,
        "total": 3,
        "chunks": {
            str(index): f"<p>translated {index}</p>"
            for index in (1, 2, 3)
        },
    }

    removed, file_deleted = remove_chunk_segments_from_file(
        str(output_path),
        "pdf-section-hash",
        [2],
        entry,
    )

    assert removed == [2]
    assert file_deleted is False
    assert output_path.exists()
    remaining = extract_marked_chunks(
        output_path.read_text(encoding="utf-8"),
        "pdf-section-hash",
    )
    assert sorted(remaining) == [1, 3]


def test_retranslating_epub_chunks_separately_deletes_file_after_last_chunk(
        tmp_path):
    output_path = tmp_path / "response_chapter_001.xhtml"
    output_path.write_text(
        "\n".join(
            wrap_chunk_html(
                "epub-chapter-hash",
                index,
                3,
                f"<p>translated {index}</p>",
            )
            for index in (1, 2, 3)
        ),
        encoding="utf-8",
    )
    progress = ProgressManager(str(tmp_path))
    budget = _epub_chunk_budget()
    progress.prepare_chapter_chunk_progress(
        "epub-chapter-hash", 3, budget, enabled=True
    )
    for index in (1, 2, 3):
        progress.record_chapter_chunk(
            "epub-chapter-hash",
            index,
            3,
            f"<p>translated {index}</p>",
            budget,
        )

    entry = progress.prog["chapter_chunks"]["epub-chapter-hash"]
    for index in (1, 2):
        removed, file_deleted = remove_chunk_segments_from_file(
            str(output_path),
            "epub-chapter-hash",
            [index],
            entry,
        )
        assert removed == [index]
        assert file_deleted is False
        assert output_path.exists()
        progress.reset_chapter_chunks("epub-chapter-hash", [index])

    removed, file_deleted = remove_chunk_segments_from_file(
        str(output_path),
        "epub-chapter-hash",
        [3],
        entry,
    )

    assert removed == [3]
    assert file_deleted is True
    assert not output_path.exists()


def test_last_physical_chunk_deletes_empty_html_shell_with_stale_cache(
        tmp_path):
    output_path = tmp_path / "response_pdf_section_003.html"
    output_path.write_text(
        """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body>
{chunk}
</body>
</html>""".format(
            chunk=wrap_chunk_html(
                "pdf-section-hash",
                2,
                2,
                "<p>translated 2</p>",
            )
        ),
        encoding="utf-8",
    )
    # Chunk 1 is stale in progress but is already absent from the physical
    # response, reproducing a prior progress-write interruption.
    entry = {
        "schema_version": 2,
        "total": 2,
        "chunks": {
            "1": "<p>translated 1</p>",
            "2": "<p>translated 2</p>",
        },
    }

    removed, file_deleted = remove_chunk_segments_from_file(
        str(output_path),
        "pdf-section-hash",
        [2],
        entry,
    )

    assert removed == [2]
    assert file_deleted is True
    assert not output_path.exists()


def test_empty_shell_detection_preserves_image_only_chunk_output(tmp_path):
    output_path = tmp_path / "response_image_only.html"
    output_path.write_text(
        """<!DOCTYPE html><html><head></head><body>
{chunk}<img src="cover.jpg"/>
</body></html>""".format(
            chunk=wrap_chunk_html("image-section", 2, 2, "<p>remove me</p>")
        ),
        encoding="utf-8",
    )
    entry = {
        "schema_version": 2,
        "total": 2,
        "chunks": {"1": "stale", "2": "<p>remove me</p>"},
    }

    removed, file_deleted = remove_chunk_segments_from_file(
        str(output_path), "image-section", [2], entry
    )

    assert removed == [2]
    assert file_deleted is False
    assert '<img src="cover.jpg"/>' in output_path.read_text(encoding="utf-8")


def test_chunk_qa_state_excludes_only_failed_chunk_from_resume_cache(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )
    for index in (1, 2):
        progress.record_chapter_chunk(
            "hash-1",
            index,
            2,
            f"translated {index}",
            _epub_chunk_budget(),
        )

    assert progress.set_chapter_chunk_qa(
        "hash-1", 2, ["llm_token_issue: 'BADTOKEN'"]
    )
    entry = progress.prog["chapter_chunks"]["hash-1"]
    assert entry["chapter_status"] == "qa_failed"
    assert entry["entries"]["2"]["status"] == "qa_failed"
    assert progress.get_reusable_chapter_chunks("hash-1") == {
        "1": "translated 1"
    }

    assert progress.set_chapter_chunk_qa("hash-1", 2, [])
    assert entry["chapter_status"] == "completed"
    assert progress.get_reusable_chapter_chunks("hash-1") == {
        "1": "translated 1",
        "2": "translated 2",
    }


def test_returned_truncated_chunk_is_retained_but_never_reused(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )

    assert progress.record_chapter_chunk(
        "hash-1",
        1,
        2,
        "truncated translation",
        _epub_chunk_budget(),
        qa_issues=["TRUNCATED"],
    )

    entry = progress.prog["chapter_chunks"]["hash-1"]
    assert entry["chunks"]["1"] == "truncated translation"
    assert entry["entries"]["1"]["status"] == "qa_failed"
    assert entry["entries"]["1"]["qa_issues_found"] == ["TRUNCATED"]
    assert entry["entries"]["2"]["status"] == "pending"
    assert progress.get_reusable_chapter_chunks("hash-1") == {}


def test_chunk_runtime_status_tracks_dispatch_and_cancel_without_regressing_cache(
    tmp_path,
):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-1", 1, 3, "translated one", _epub_chunk_budget()
    )

    assert progress.set_chapter_chunk_runtime_status(
        "hash-1",
        2,
        "in_progress",
        model_name="provider/model-a",
        key_identifier="key-a",
    )
    entry = progress.prog["chapter_chunks"]["hash-1"]
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "in_progress"
    assert entry["entries"]["2"]["model_name"] == "provider/model-a"
    assert entry["entries"]["2"]["key_identifier"] == "key-a"
    assert entry["chunk_metadata"]["2"] == {
        "model_name": "provider/model-a",
        "key_identifier": "key-a",
    }
    assert entry["entries"]["3"]["status"] == "pending"
    assert entry["chapter_status"] == "in_progress"
    assert progress.get_reusable_chapter_chunks("hash-1") == {
        "1": "translated one"
    }

    # A retry can rotate to a different key/model; the chunk owns the latest
    # actual route instead of inheriting a chapter-wide value.
    assert progress.set_chapter_chunk_runtime_status(
        "hash-1",
        2,
        "in_progress",
        model_name="provider/model-b",
        key_identifier="key-b",
    )
    assert entry["entries"]["2"]["model_name"] == "provider/model-b"
    assert entry["entries"]["2"]["key_identifier"] == "key-b"

    assert progress.reset_in_progress_chapter_chunks("hash-1") == [2]
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "pending"
    assert entry["chapter_status"] == "incomplete"


def test_saved_partial_output_cannot_restore_pending_chunks_as_completed(
    tmp_path,
):
    output_name = "p-003.xhtml"
    (tmp_path / output_name).write_text(
        "<p>saved truncated evidence</p>", encoding="utf-8"
    )
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-3", 5, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-3", 1, 5, "translated one", _epub_chunk_budget()
    )
    chapter = {
        "filename": output_name,
        "original_basename": output_name,
    }
    progress.update(
        2,
        3,
        "hash-3",
        output_name,
        status="in_progress",
        chapter_obj=chapter,
    )
    chapter_key = next(iter(progress.prog["chapters"]))

    should_translate, _message, existing_output = progress.check_chapter_status(
        2,
        3,
        "hash-3",
        str(tmp_path),
        chapter_obj=chapter,
    )

    assert should_translate is True
    assert existing_output == output_name
    assert progress.prog["chapters"][chapter_key]["status"] == "in_progress"
    assert not progress.prog["chapters"][chapter_key].get(
        "auto_restored_from_output"
    )

    # Repair a progress file that was already corrupted by the old
    # file-existence recovery logic (the exact state shown in the UI bug).
    progress.prog["chapters"][chapter_key]["status"] = "completed"
    should_translate, _message, existing_output = progress.check_chapter_status(
        2,
        3,
        "hash-3",
        str(tmp_path),
        chapter_obj=chapter,
    )
    assert should_translate is True
    assert existing_output == output_name
    assert progress.prog["chapters"][chapter_key]["status"] == "pending"


def test_parent_completion_is_rejected_while_chunk_work_remains(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-3", 5, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-3", 1, 5, "translated one", _epub_chunk_budget()
    )

    progress.update(
        2, 3, "hash-3", "p-003.xhtml", status="completed"
    )

    chapter_info = next(iter(progress.prog["chapters"].values()))
    entry = progress.prog["chapter_chunks"]["hash-3"]
    assert chapter_info["status"] == "pending"
    assert entry["chapter_status"] == "incomplete"
    assert chunk_entry_needs_translation(entry) is True
    assert effective_parent_status("completed", entry) == "pending"
    assert effective_parent_status("in_progress", entry) == "in_progress"


def test_progress_manager_expands_chapter_into_selectable_chunk_rows(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"]["1"] = {
        "actual_num": 1,
        "content_hash": "hash-1",
        "output_file": "chapter.html",
        "status": "completed",
    }
    progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-1", 1, 2, "translated 1", _epub_chunk_budget()
    )
    progress.set_chapter_chunk_qa("hash-1", 1, ["bad output"])

    rows = [
        {
            "key": "1",
            "num": 1,
            "info": progress.prog["chapters"]["1"],
            "output_file": "chapter.html",
            "status": "completed",
        }
    ]
    mixin = RetranslationMixin()
    mixin._append_chunk_progress_display_info(
        {"prog": progress.prog}, rows
    )

    assert len(rows) == 3
    assert rows[0]["key"] == "1"
    assert rows[0]["progress_key"] == "1"
    assert rows[0]["status"] == "pending"
    assert rows[1]["is_chunk_progress"] is True
    assert rows[1]["chunk_index"] == 1
    assert rows[1]["status"] == "qa_failed"
    assert rows[2]["chunk_index"] == 2
    assert rows[2]["status"] == "pending"
    assert rows[1]["parent_progress_key"] == "1"


def test_chunk_schema_and_progress_rows_sort_indexes_numerically():
    insertion_order = [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]
    entry = {
        "schema_version": 2,
        "total": 12,
        "entries": {
            str(index): {
                "index": index,
                "status": "completed",
                "model_name": f"model-{index}",
            }
            for index in insertion_order
        },
        "chunks": {
            str(index): f"translated {index}" for index in insertion_order
        },
        "chunk_metadata": {
            str(index): {"model_name": f"model-{index}"}
            for index in insertion_order
        },
        "completed": insertion_order,
    }

    assert ensure_chunk_entry_schema(entry) is True
    expected_keys = [str(index) for index in range(1, 13)]
    assert list(entry["entries"]) == expected_keys
    assert list(entry["chunks"]) == expected_keys
    assert list(entry["chunk_metadata"]) == expected_keys
    assert entry["completed"] == list(range(1, 13))

    for is_pdf in (False, True):
        parent = {
            "actual_num": 1,
            "content_hash": "hash-1",
            "output_file": "section.html" if is_pdf else "chapter.xhtml",
            "status": "completed",
        }
        if is_pdf:
            parent["pdf_toc_section"] = True
        rows = [{
            "key": "1",
            "num": 1,
            "info": parent,
            "output_file": parent["output_file"],
            "status": "completed",
        }]
        RetranslationMixin()._append_chunk_progress_display_info(
            {
                "prog": {
                    "chapters": {"1": parent},
                    "chapter_chunks": {"hash-1": entry},
                }
            },
            rows,
        )
        assert [row["chunk_index"] for row in rows[1:]] == list(range(1, 13))


@pytest.mark.parametrize(
    ("changed_budget", "expected_reason"),
    [
        (
            _epub_chunk_budget(initial=5000, cached=4250),
            "initial chunk size changed",
        ),
        (
            _epub_chunk_budget(initial=5750, cached=3500),
            "cached chunk size changed",
        ),
    ],
)
def test_incomplete_chunk_progress_is_invalidated_when_budget_changes(
    tmp_path,
    changed_budget,
    expected_reason,
):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-1", 1, 3, "stale", _epub_chunk_budget()
    )

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 3, changed_budget, enabled=True
    )

    assert changed is True
    assert expected_reason in reason
    assert entry["completed"] == []
    assert entry["chunks"] == {}


def test_completed_chunk_progress_survives_budget_and_split_changes(tmp_path):
    progress = ProgressManager(str(tmp_path))
    original_budget = _epub_chunk_budget()
    progress.prepare_chapter_chunk_progress(
        "hash-1", 2, original_budget, enabled=True
    )
    for index in (1, 2):
        progress.record_chapter_chunk(
            "hash-1",
            index,
            2,
            f"translated {index}",
            original_budget,
        )
    progress.mark_chapter_chunk_progress_status("hash-1", "completed")

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1",
        3,
        _epub_chunk_budget(initial=5000, cached=3500),
        enabled=True,
    )

    assert reason is None
    assert changed is False
    assert entry["total"] == 2
    assert entry["completed"] == [1, 2]
    assert set(entry["chunks"]) == {"1", "2"}
    assert entry["chapter_status"] == "completed"


def test_v2_missing_chunk_record_stays_pending_instead_of_being_reconstructed(
        tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapter_chunks"]["hash-1"] = {
        "schema_version": 2,
        "total": 2,
        "completed": [1, 2],
        "chunks": {
            "1": "translated 1",
            "2": "stale translated 2",
        },
        "chunk_metadata": {},
        "entries": {
            "1": {"index": 1, "status": "completed"},
        },
        "chapter_status": "completed",
        **_epub_chunk_budget(),
    }

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )

    assert reason is None
    assert changed is True
    assert entry["entries"]["1"]["status"] == "completed"
    assert entry["entries"]["2"]["status"] == "pending"
    assert entry["completed"] == [1]
    assert progress.get_reusable_chapter_chunks("hash-1") == {
        "1": "translated 1"
    }
    assert chunk_entry_needs_translation(entry) is True


def test_retranslate_selected_persistently_deletes_chunk_cache_despite_race(
        tmp_path):
    progress_path = tmp_path / "translation_progress.json"
    budget = _epub_chunk_budget()
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"]["pdf:section"] = {
        "actual_num": 3,
        "content_hash": "chunk-hash",
        "output_file": "response_pdf_section_003.html",
        "status": "completed",
    }
    for index in (1, 2):
        progress.record_chapter_chunk(
            "chunk-hash",
            index,
            2,
            f"translated {index}",
            budget,
            model_name=f"model-{index}",
        )
    progress.mark_chapter_chunk_progress_status("chunk-hash", "completed")
    baseline = json.loads(json.dumps(progress.prog))
    changed = json.loads(json.dumps(baseline))
    reset_chunks_for_retranslation(
        changed["chapter_chunks"]["chunk-hash"],
        [2],
    )
    changed["chapters"]["pdf:section"]["status"] = "pending"

    # Simulate a concurrent translator save that rewrote chunk 2 after the
    # Progress Manager took its baseline snapshot.
    latest = json.loads(json.dumps(baseline))
    latest_entry = latest["chapter_chunks"]["chunk-hash"]
    latest_entry["chunks"]["2"] = "new concurrent result"
    latest_entry["entries"]["2"]["result_sha256"] = "new-result"
    latest_entry["entries"]["2"]["model_name"] = "new-model"
    progress_path.write_text(json.dumps(latest), encoding="utf-8")

    saved = _merge_and_write_retranslation_progress(
        str(progress_path),
        baseline,
        changed,
        authoritative_chunk_resets=[{
            "chunk_key": "chunk-hash",
            "parent_key": "pdf:section",
            "indices": [2],
        }],
    )

    chunk_entry = saved["chapter_chunks"]["chunk-hash"]
    assert "2" not in chunk_entry["chunks"]
    assert "2" not in chunk_entry["chunk_metadata"]
    assert chunk_entry["completed"] == [1]
    assert chunk_entry["entries"]["2"]["status"] == "pending"
    assert "result_sha256" not in chunk_entry["entries"]["2"]
    assert "model_name" not in chunk_entry["entries"]["2"]
    assert saved["chapters"]["pdf:section"]["status"] == "pending"

    persisted = json.loads(progress_path.read_text(encoding="utf-8"))
    assert persisted == saved


def test_retranslation_progress_write_retries_windows_access_denied(
        tmp_path, monkeypatch):
    progress_path = tmp_path / "translation_progress.json"
    baseline = {"chapters": {}, "chapter_chunks": {}}
    progress_path.write_text(json.dumps(baseline), encoding="utf-8")
    changed = json.loads(json.dumps(baseline))
    changed["new_value"] = True

    real_replace = os.replace
    replace_attempts = []

    def transiently_locked_replace(source, destination):
        replace_attempts.append((source, destination))
        if len(replace_attempts) < 3:
            error = PermissionError(5, "Access is denied", destination)
            error.winerror = 5
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(
        retranslation_gui_module.os,
        "replace",
        transiently_locked_replace,
    )
    monkeypatch.setattr(retranslation_gui_module.time, "sleep", lambda _delay: None)

    saved = _merge_and_write_retranslation_progress(
        str(progress_path), baseline, changed
    )

    assert len(replace_attempts) == 3
    assert saved["new_value"] is True
    assert json.loads(progress_path.read_text(encoding="utf-8")) == saved
    assert not list(tmp_path.glob("translation_progress.json.*.tmp"))


def test_completed_parent_with_marked_output_and_missing_ledger_retranslates(
        tmp_path):
    output_name = "response_pdf_section_003.html"
    missing_key = "missing-active-ledger"
    (tmp_path / output_name).write_text(
        "\n".join(
            wrap_chunk_html(
                missing_key,
                index,
                2,
                f"<p>translated {index}</p>",
            )
            for index in (1, 2)
        ),
        encoding="utf-8",
    )
    progress = ProgressManager(str(tmp_path))
    progress_key = "pdf:section"
    progress.prog["chapters"][progress_key] = {
        "actual_num": 3,
        "content_hash": missing_key,
        "output_file": output_name,
        "status": "completed",
        "pdf_toc_section": True,
        "pdf_section_id": "section",
        "pdf_progress_key": progress_key,
    }

    should_translate, _message, existing_output = progress.check_chapter_status(
        2,
        3,
        missing_key,
        str(tmp_path),
        chapter_obj={
            "num": 3,
            "filename": "pdf_section_3.html",
            "content_hash": missing_key,
            "pdf_toc_section": True,
            "pdf_section_id": "section",
        },
    )

    assert should_translate is True
    assert existing_output == output_name
    assert progress.prog["chapters"][progress_key]["status"] == "pending"


def test_disabled_chunk_progress_removes_incomplete_cache(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=True
    )
    progress.record_chapter_chunk(
        "hash-1", 1, 2, "cached", _epub_chunk_budget()
    )

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 2, _epub_chunk_budget(), enabled=False
    )

    assert entry is None
    assert reason == "disabled"
    assert changed is True
    assert "hash-1" not in progress.prog["chapter_chunks"]


def test_single_chunk_chapter_has_no_chunk_cache_markers_or_child_row(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapter_chunks"]["hash-1"] = {
        "schema_version": 2,
        "total": 1,
        "completed": [1],
        "chunks": {"1": "<p>legacy cached chapter</p>"},
        "entries": {"1": {"index": 1, "status": "completed"}},
    }

    entry, reason, changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 1, _epub_chunk_budget(), enabled=True
    )

    assert entry is None
    assert reason is None
    assert changed is True
    assert "hash-1" not in progress.prog["chapter_chunks"]
    assert not progress.record_chapter_chunk(
        "hash-1", 1, 1, "translated", _epub_chunk_budget()
    )
    assert progress.get_reusable_chapter_chunks("hash-1") == {}
    assert wrap_chunk_html("hash-1", 1, 1, "<p>whole chapter</p>") == (
        "<p>whole chapter</p>"
    )

    progress.prog["chapters"]["1"] = {
        "actual_num": 1,
        "content_hash": "hash-1",
        "output_file": "chapter.html",
        "status": "completed",
    }
    # A stale record loaded by a GUI path is ignored defensively.
    progress.prog["chapter_chunks"]["hash-1"] = {
        "total": 1,
        "entries": {"1": {"index": 1, "status": "completed"}},
    }
    rows = [{
        "key": "1",
        "num": 1,
        "info": progress.prog["chapters"]["1"],
        "output_file": "chapter.html",
        "status": "completed",
    }]
    RetranslationMixin()._append_chunk_progress_display_info(
        {"prog": progress.prog}, rows
    )
    assert len(rows) == 1

    progress.save()
    saved = json.loads(Path(progress.PROGRESS_FILE).read_text(encoding="utf-8"))
    assert "hash-1" not in saved["chapter_chunks"]


def test_chunk_budget_uses_global_individual_and_cached_model_limits(
    monkeypatch,
):
    keys = [
        {
            "api_key": "one",
            "model": "model-a",
            "enabled": True,
            "individual_output_token_limit": 12000,
        },
        {
            "api_key": "two",
            "model": "model-b",
            "enabled": True,
            "individual_output_token_limit": None,
        },
        {
            "api_key": "disabled",
            "model": "model-c",
            "enabled": False,
            "individual_output_token_limit": 1000,
        },
    ]
    monkeypatch.setenv("MODEL", "model-main")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "20000")
    monkeypatch.setenv("COMPRESSION_FACTOR", "2")
    monkeypatch.setenv("USE_MULTI_API_KEYS", "1")
    monkeypatch.setenv("MULTI_API_KEYS", json.dumps(keys))
    monkeypatch.setenv("USE_FALLBACK_KEYS", "0")
    monkeypatch.setattr(
        UnifiedClient,
        "_model_token_limits",
        {"model-a": 9000, "model-b": 15000, "model-c": 500},
    )

    config = TranslationConfig()
    snapshot = config.get_chunk_budget_snapshot()

    assert config.get_initial_output_limit() == 12000
    assert config.get_effective_output_limit() == 9000
    assert snapshot["initial_chunk_size"] == 5750
    assert snapshot["cached_chunk_size"] == 4250


def test_chunk_progress_defaults_enabled_and_respects_explicit_disable(monkeypatch):
    monkeypatch.delenv("ENABLE_CHUNK_PROGRESS", raising=False)
    assert TranslationConfig().ENABLE_CHUNK_PROGRESS is True

    monkeypatch.setenv("ENABLE_CHUNK_PROGRESS", "0")
    assert TranslationConfig().ENABLE_CHUNK_PROGRESS is False


def test_unified_cached_limit_resolves_auth_route_aliases(monkeypatch):
    monkeypatch.setattr(
        UnifiedClient,
        "_model_token_limits",
        {
            "z-ai/glm-test": 7000,
            "authnd/z-ai/glm-test": 6500,
        },
    )
    monkeypatch.setattr(
        UnifiedClient,
        "_authcd_model_max_tokens",
        {"claude-test": 6000},
        raising=False,
    )

    assert UnifiedClient.get_cached_output_token_limit(
        "authnd3/z-ai/glm-test"
    ) == 6500
    assert UnifiedClient.get_cached_output_token_limit(
        "authcd2/claude-test"
    ) == 6000


def test_unified_cached_cap_wins_after_individual_key_override(monkeypatch):
    client = UnifiedClient.__new__(UnifiedClient)
    client.model = "model-a"
    client.client_type = "openai"
    client.current_key_output_token_limit = 12000
    client._get_active_request_model = lambda: "model-a"
    client._get_thread_local_client = lambda: types.SimpleNamespace(
        output_token_limit=None,
        per_key_max_output_tokens=None,
    )
    client._is_o_series_model = lambda: False
    monkeypatch.setattr(
        UnifiedClient,
        "_model_token_limits",
        {"model-a": 9000},
    )

    max_tokens, max_completion_tokens = client._normalize_token_params(
        20000,
        None,
    )

    assert max_tokens == 9000
    assert max_completion_tokens is None


@pytest.mark.parametrize("extension", [".srt", ".ass", ".lrc"])
def test_progress_manager_does_not_add_metadata_row_for_subtitles(
    tmp_path,
    extension,
):
    source = tmp_path / f"episode{extension}"
    gui = RetranslationMixin()
    rows = []

    gui._append_metadata_display_info(
        {
            "file_path": str(source),
            "output_dir": str(tmp_path / "output"),
            "prog": {"chapters": {}, "version": "2.1"},
        },
        rows,
    )

    assert rows == []


def test_progress_manager_does_not_add_metadata_row_for_empty_subtitle_zip(
    tmp_path,
):
    archive = tmp_path / "season.zip"
    extracted = tmp_path / "extract" / "episode.srt"
    gui = RetranslationMixin()
    gui._subtitle_zip_output_groups = {
        os.path.normcase(os.path.abspath(extracted)): {
            "archive_path": str(archive),
            "bundle_id": os.path.normcase(os.path.abspath(archive)),
            "bundle_files": [str(extracted)],
            "output_dir": str(tmp_path / "season"),
        }
    }
    rows = []

    gui._append_metadata_display_info(
        {
            "file_path": str(archive),
            "output_dir": str(tmp_path / "season"),
            "prog": {"chapters": {}, "version": "2.1"},
        },
        rows,
    )

    assert rows == []


def test_progress_manager_inspects_nested_subtitle_zip_without_session_mapping(
    tmp_path,
):
    archive = tmp_path / "season.zip"
    with zipfile.ZipFile(archive, "w") as subtitle_zip:
        subtitle_zip.writestr(
            "Season 01/episode.lrc",
            "[ti:Episode]\n[00:01.00]Hello\n",
        )
    gui = RetranslationMixin()
    rows = []
    data = {
        "file_path": str(archive),
        "output_dir": str(tmp_path / "season"),
        "prog": {"chapters": {}, "version": "2.1"},
    }

    assert gui._zip_is_subtitle_archive(str(archive)) is True
    assert gui._progress_view_is_subtitle(data) is True
    gui._append_metadata_display_info(data, rows)

    assert rows == []


def test_progress_manager_seeds_every_subtitle_zip_member_before_translation(
    tmp_path,
):
    archive = tmp_path / "season.zip"
    with zipfile.ZipFile(archive, "w") as subtitle_zip:
        subtitle_zip.writestr(
            "Season 01/episode.srt",
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        )
        subtitle_zip.writestr(
            "Season 02/episode.srt",
            "1\n00:00:00,000 --> 00:00:01,000\nAgain\n",
        )
        subtitle_zip.writestr(
            "Season 02/theme.ass",
            "[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,"
            "Default,,0,0,0,,Theme\n",
        )
    output_dir = tmp_path / "outputs" / "season"
    output_dir.mkdir(parents=True)
    gui = RetranslationMixin()
    prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}

    changed = gui._seed_subtitle_zip_progress_entries(
        str(archive),
        str(output_dir),
        prog,
    )

    assert changed is True
    assert len(prog["chapters"]) == 3
    assert {
        entry["status"] for entry in prog["chapters"].values()
    } == {"not_translated"}
    assert {
        Path(entry["output_file"]).name
        for entry in prog["chapters"].values()
    } == {"episode.srt", "episode_2.srt", "theme.ass"}
    assert set(prog["subtitle_files"]) == {
        "episode.srt",
        "episode_2.srt",
        "theme.ass",
    }
    assert {
        summary["status"]
        for summary in prog["subtitle_files"].values()
    } == {"not_translated"}

    first_key = "subtitle:episode.srt:1"
    prog["chapters"][first_key]["status"] = "completed"
    assert (
        gui._seed_subtitle_zip_progress_entries(
            str(archive),
            str(output_dir),
            prog,
        )
        is True
    )
    assert prog["chapters"][first_key]["status"] == "completed"
    assert prog["subtitle_files"]["episode.srt"]["status"] == "completed"


def test_progress_manager_does_not_invent_metadata_for_non_subtitle_zip(
    tmp_path,
):
    archive = tmp_path / "documents.zip"
    with zipfile.ZipFile(archive, "w") as document_zip:
        document_zip.writestr("notes/readme.txt", "Not subtitles")
    gui = RetranslationMixin()
    rows = []
    data = {
        "file_path": str(archive),
        "output_dir": str(tmp_path / "documents"),
        "prog": {"chapters": {}, "version": "2.1"},
    }

    assert gui._zip_is_subtitle_archive(str(archive)) is False
    assert gui._progress_view_is_subtitle(data) is False
    gui._append_metadata_display_info(data, rows)

    assert rows == []


def test_progress_manager_does_not_treat_epub_shaped_zip_as_subtitle_archive(
    tmp_path,
):
    archive = tmp_path / "book.zip"
    with zipfile.ZipFile(archive, "w") as epub_zip:
        epub_zip.writestr("mimetype", "application/epub+zip")
        epub_zip.writestr("META-INF/container.xml", "<container/>")
        epub_zip.writestr("OEBPS/content.opf", "<package/>")
        epub_zip.writestr("OEBPS/media/captions.srt", "incidental resource")

    gui = RetranslationMixin()

    assert gui._zip_is_subtitle_archive(str(archive)) is False


def test_disabled_epub_metadata_row_is_still_shown_as_skipped(tmp_path):
    gui = RetranslationMixin()
    gui.config = {"translate_book_title": False}
    rows = []

    gui._append_metadata_display_info(
        {
            "file_path": str(tmp_path / "book.epub"),
            "output_dir": str(tmp_path / "book"),
            "prog": {"chapters": {}, "version": "2.1"},
        },
        rows,
    )

    assert len(rows) == 1
    assert rows[0]["special_type"] == "metadata"
    assert rows[0]["status"] == "skipped"


def test_progress_manager_recovers_metadata_row_from_structural_snapshot(
    tmp_path,
):
    source = tmp_path / "book.epub"
    output_dir = tmp_path / "book"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "language": "en",
                "chapter_count": 12,
                "chapter_titles": {"1": "Chapter 1"},
            }
        ),
        encoding="utf-8",
    )
    gui = RetranslationMixin()
    gui.config = {
        "translate_book_title": True,
        "translate_metadata_fields": {
            "title": True,
            "description": True,
            "subject": True,
        },
    }
    prog = {"chapters": {}, "version": "2.1"}

    assert gui._ensure_metadata_progress_entry(
        prog,
        str(output_dir),
        str(source),
    ) is True

    entry = prog["chapters"]["__metadata__"]
    assert entry["status"] == "pending"
    assert entry["metadata_phase"] == "recovery"
    assert entry["metadata_regeneration_requested"] is True
    assert entry["metadata_fields"] == ["title", "description", "subject"]

    rows = []
    gui._append_metadata_display_info(
        {
            "file_path": str(source),
            "output_dir": str(output_dir),
            "prog": prog,
        },
        rows,
    )
    assert len(rows) == 1
    assert rows[0]["special_type"] == "metadata"
    assert rows[0]["status"] == "pending"


def test_structural_metadata_snapshot_preserves_existing_progress_history(
    tmp_path,
):
    source = tmp_path / "book.epub"
    output_dir = tmp_path / "book"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps({"language": "en", "chapter_count": 12}),
        encoding="utf-8",
    )
    existing = {
        "actual_num": -1,
        "output_file": "metadata.json",
        "status": "completed",
        "special_type": "metadata",
        "metadata_progress_key": "__metadata__",
        "metadata_fields": ["title", "description"],
        "model_name": "test-model",
    }
    prog = {
        "chapters": {"__metadata__": dict(existing)},
        "version": "2.1",
    }
    gui = RetranslationMixin()
    gui.config = {
        "translate_book_title": True,
        "translate_metadata_fields": {
            "title": True,
            "description": True,
        },
    }

    assert gui._ensure_metadata_progress_entry(
        prog,
        str(output_dir),
        str(source),
    ) is False
    assert prog["chapters"]["__metadata__"] == existing


def test_progress_backend_does_not_delete_metadata_history_for_empty_plan(
    tmp_path,
):
    progress = ProgressManager(str(tmp_path))
    existing = {
        "actual_num": -1,
        "output_file": "metadata.json",
        "status": "completed",
        "special_type": "metadata",
        "metadata_progress_key": "__metadata__",
        "metadata_fields": ["title"],
        "model_name": "test-model",
    }
    progress.prog["chapters"]["__metadata__"] = dict(existing)

    plan = progress.configure_metadata_progress(
        "together",
        {"language": "en", "chapter_count": 12},
        {"title": True},
        str(tmp_path / "metadata.json"),
        source_path=str(tmp_path / "book.epub"),
    )

    assert plan == []
    assert progress.prog["chapters"]["__metadata__"] == existing


def test_progress_update_captures_actual_request_model(tmp_path):
    progress = ProgressManager(str(tmp_path))
    set_current_thread_actual_request_model("deepseek-v4", "FALLBACK KEY (deepseek-v4)")

    progress.update(
        0,
        259,
        "hash-259",
        "ch259.xhtml",
        status="in_progress",
    )

    entry = progress.prog["chapters"]["259"]
    assert entry["model_name"] == "deepseek-v4"
    assert entry["key_identifier"] == "FALLBACK KEY (deepseek-v4)"


def test_progress_update_bookkeeping_can_ignore_stale_thread_model(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"]["260"] = {
        "actual_num": 260,
        "content_hash": "old-hash",
        "output_file": "ch260.xhtml",
        "status": "completed",
        "model_name": "gemini-3.1-flash-lite",
        "key_identifier": "MAIN KEY (gemini-3.1-flash-lite)",
    }
    set_current_thread_actual_request_model("deepseek-v4", "FALLBACK KEY (deepseek-v4)")

    progress.update(
        1,
        260,
        "new-hash",
        "ch260.xhtml",
        status="completed",
        prefer_thread_model=False,
    )

    entry = progress.prog["chapters"]["260"]
    assert entry["model_name"] == "gemini-3.1-flash-lite"
    assert entry["key_identifier"] == "MAIN KEY (gemini-3.1-flash-lite)"


def test_progress_update_accepts_copied_non_api_model_label(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"]["4"] = {
        "actual_num": 4,
        "content_hash": "old-hash",
        "output_file": "p-fmatter-004.xhtml",
        "status": "completed",
        "model_name": "old-api-model",
        "key_identifier": "OLD KEY",
    }
    set_current_thread_actual_request_model(
        "stale-thread-model",
        "STALE KEY",
    )

    progress.update(
        3,
        4,
        "new-hash",
        "p-fmatter-004.xhtml",
        status="completed",
        model_name="COPIED",
    )

    entry = progress.prog["chapters"]["4"]
    assert entry["status"] == "completed"
    assert entry["model_name"] == "COPIED"
    assert "key_identifier" not in entry
    progress.save()
    saved = json.loads((tmp_path / "translation_progress.json").read_text(
        encoding="utf-8"
    ))
    assert saved["chapters"]["4"]["model_name"] == "COPIED"
    assert "key_identifier" not in saved["chapters"]["4"]


def test_completed_progress_uses_configured_model_when_request_thread_metadata_is_missing(
    tmp_path,
    monkeypatch,
):
    progress = ProgressManager(str(tmp_path))
    monkeypatch.setenv("MODEL", "authgpt/gpt-5.6-luna")

    progress.update(
        0,
        1,
        "hash-1",
        "response_chapter_001.html",
        status="completed",
    )
    progress.save()

    saved = json.loads((tmp_path / "translation_progress.json").read_text(
        encoding="utf-8"
    ))
    assert saved["chapters"]["1"]["model_name"] == (
        "authgpt/gpt-5.6-luna"
    )


def test_completed_metadata_progress_uses_configured_model_fallback(
    tmp_path,
    monkeypatch,
):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"][progress.METADATA_PROGRESS_KEY] = {
        "status": "in_progress",
        "output_file": "metadata.json",
        "special_type": "metadata",
        "metadata_progress_key": progress.METADATA_PROGRESS_KEY,
    }
    monkeypatch.setenv("MODEL", "authgpt/gpt-5.6-luna")

    progress.update_metadata_status(
        "completed",
        str(tmp_path / "metadata.json"),
        key=progress.METADATA_PROGRESS_KEY,
    )

    assert progress.prog["chapters"][progress.METADATA_PROGRESS_KEY][
        "model_name"
    ] == "authgpt/gpt-5.6-luna"


def test_vision_ocr_progress_replaces_stale_model_with_active_request(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"]["251"] = {
        "actual_num": 251,
        "chapter_num": 251,
        "content_hash": "hash-251",
        "output_file": "No00251Chapter.xhtml",
        "status": "failed",
        "model_name": "deepseek-v4-flash",
        "key_identifier": "OLD KEY (deepseek-v4-flash)",
    }
    set_current_thread_actual_request_model(
        "gemini-3.1-flash-lite",
        "VISION KEY (gemini-3.1-flash-lite)",
    )

    progress.update_ocr_progress(
        251,
        0,
        8,
        output_file="No00251Chapter.xhtml",
        content_hash="hash-251",
    )

    entry = progress.prog["chapters"]["251"]
    assert entry["status"] == "in_progress"
    assert entry["ocr_progress"]["label"] == "0/8"
    assert entry["model_name"] == "gemini-3.1-flash-lite"
    assert entry["key_identifier"] == "VISION KEY (gemini-3.1-flash-lite)"


def test_single_image_chapter_header_is_prepended_to_combined_ocr():
    soup = BeautifulSoup(
        "<html><head><title>Fallback title</title></head>"
        "<body><h2>第四十一章 回城</h2><img src='chapter.gif'/></body></html>",
        "html.parser",
    )

    header = _vision_ocr_header_markdown(soup)
    combined = ImageTranslator._prepend_combined_ocr_prefix(
        "蜜儿娜躲在丛林里更换着衣服。",
        header,
    )

    assert header == "## 第四十一章 回城"
    assert combined == "## 第四十一章 回城\n\n蜜儿娜躲在丛林里更换着衣服。"


def test_combined_ocr_does_not_duplicate_header_already_seen_in_image():
    combined = ImageTranslator._prepend_combined_ocr_prefix(
        "第四十一章 回城\n\n蜜儿娜躲在丛林里更换着衣服。",
        "## 第四十一章 回城",
    )

    assert combined.count("第四十一章 回城") == 1


def test_title_is_used_when_epub_has_no_visible_body_heading():
    soup = BeautifulSoup(
        "<html><head><title>Chapter title</title></head><body><img src='chapter.gif'/></body></html>",
        "html.parser",
    )

    assert _vision_ocr_header_markdown(soup) == "## Chapter title"


def test_only_original_vision_heading_is_removed_after_translation_insert():
    soup = BeautifulSoup(
        "<html><body><h2>Source Header</h2><img src='chapter.gif'/></body></html>",
        "html.parser",
    )
    original_headers = list(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
    soup.find('img').replace_with(
        BeautifulSoup(
            "<div class='translated-text-only'><h2>Chapter Forty-Three: The Silver Holy Bow</h2><p>Inside the ruins.</p></div>",
            "html.parser",
        ).find('div')
    )

    for source_header in original_headers:
        if source_header.parent is not None:
            source_header.decompose()

    assert soup.find('h2').get_text(strip=True) == "Chapter Forty-Three: The Silver Holy Bow"


def test_image_translation_formatter_preserves_html_heading_without_nested_paragraph():
    translator = ImageTranslator.__new__(ImageTranslator)

    rendered = translator._format_translation_as_html(
        "<h2>Chapter Forty-Three: The Silver Holy Bow</h2>\n<p>Inside the ruins.</p>"
    )

    assert rendered.startswith("<h2>Chapter Forty-Three: The Silver Holy Bow</h2>")
    assert "<p><h2>" not in rendered


def test_image_translation_formatter_converts_markdown_heading():
    translator = ImageTranslator.__new__(ImageTranslator)

    rendered = translator._format_translation_as_html(
        "## Chapter Forty-Three: The Silver Holy Bow\n\nInside the ruins."
    )

    assert rendered.startswith("<h2>Chapter Forty-Three: The Silver Holy Bow</h2>")
    assert "<p>Inside the ruins.</p>" in rendered


def test_refinement_completion_preserves_refined_status_and_model(tmp_path):
    progress = ProgressManager(str(tmp_path))
    set_current_thread_actual_request_model("deepseek-refine", "FALLBACK KEY (deepseek-refine)")

    progress.update(
        2,
        261,
        "hash-261",
        "ch261.xhtml",
        status="completed",
    )
    progress.update_refinement_status(
        2,
        261,
        "hash-261-refined",
        "ch261.xhtml",
        "refined",
    )

    entry = progress.prog["chapters"]["261"]
    assert entry["model_name"] == "deepseek-refine"
    assert entry["refinement_status"] == "refined"
    assert "refined_at" in entry


def test_interrupted_pdf_update_recovers_model_and_refinement_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MODEL", raising=False)
    progress = ProgressManager(str(tmp_path))
    previous = {
        "actual_num": 8,
        "content_hash": "old-hash",
        "output_file": "response_pdf_section_008.html",
        "status": "completed",
        "model_name": "authnd/deepseek-ai/deepseek-v4-pro-0813",
        "refinement_status": "refined",
        "refined_at": 123.0,
        "unrefined_backup_file": "_unrefined/response_pdf_section_008.html",
        "pdf_toc_section": True,
        "pdf_section_id": "section-eight",
    }
    progress.prog["chapters"] = {
        "pdf:section-eight": {
            "actual_num": 8,
            "content_hash": "old-hash",
            "output_file": "response_pdf_section_008.html",
            "status": "in_progress",
            "previous_status": "completed",
            "previous_progress_entry": previous,
            "pdf_toc_section": True,
            "pdf_section_id": "section-eight",
        }
    }

    progress.update(
        7,
        8,
        "new-hash",
        "response_pdf_section_008.html",
        status="qa_failed",
        chapter_obj={
            "num": 8,
            "pdf_toc_section": True,
            "pdf_section_id": "section-eight",
        },
        qa_issues_found=["Chinese_text_found_1_chars_[神]"],
    )

    entry = progress.prog["chapters"]["pdf:section-eight"]
    assert entry["model_name"] == previous["model_name"]
    assert entry["refinement_status"] == "refined"
    assert entry["refined_at"] == 123.0
    assert entry["unrefined_backup_file"] == previous["unrefined_backup_file"]


def test_pdf_qa_scan_preserves_prior_refinement_and_model(tmp_path):
    progress_path = tmp_path / "translation_progress.json"
    output_file = "response_pdf_section_008.html"
    (tmp_path / output_file).write_text("<p>神</p>", encoding="utf-8")
    progress_path.write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "pdf:section-eight": {
                    "actual_num": 8,
                    "content_hash": "hash-eight",
                    "output_file": output_file,
                    "status": "completed",
                    "model_name": "authnd/deepseek-ai/deepseek-v4-pro-0813",
                    "refinement_status": "refined",
                    "refined_at": 123.0,
                    "pdf_toc_section": True,
                    "pdf_section_id": "section-eight",
                }
            },
            "chapter_chunks": {},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    logs = []

    update_progress_file(
        str(tmp_path),
        [{
            "filename": output_file,
            "filepath": str(tmp_path / output_file),
            "file_index": 7,
            "chapter_num": 8,
            "issues": ["Chinese_text_found_1_chars_[神]"],
            "qa_issue_previews": {},
            "duplicate_confidence": 0,
            "score": 1,
        }],
        logs.append,
        progress_path=str(progress_path),
    )

    saved = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = saved["chapters"]["pdf:section-eight"]
    assert entry["status"] == "qa_failed"
    assert entry["model_name"] == "authnd/deepseek-ai/deepseek-v4-pro-0813"
    assert entry["refinement_status"] == "refined"
    assert entry["refined_at"] == 123.0


def test_progress_load_promotes_nested_model_and_refinement_metadata(tmp_path):
    previous = {
        "status": "completed",
        "model_name": "provider/previous-model",
        "refinement_status": "refined",
        "refined_at": 77.0,
    }
    (tmp_path / "translation_progress.json").write_text(
        json.dumps({
            "version": "2.1",
            "chapters": {
                "pdf:nested": {
                    "actual_num": 11,
                    "output_file": "response_pdf_section_011.html",
                    "status": "completed",
                    "previous_progress_entry": previous,
                }
            },
            "chapter_chunks": {},
        }),
        encoding="utf-8",
    )

    progress = ProgressManager(str(tmp_path))
    entry = progress.prog["chapters"]["pdf:nested"]
    assert entry["model_name"] == "provider/previous-model"
    assert entry["refinement_status"] == "refined"
    assert entry["refined_at"] == 77.0


def test_refinement_status_returns_existing_chapter_zero_output_key(tmp_path):
    progress = ProgressManager(str(tmp_path))
    progress.prog["chapters"] = {
        "0": {
            "actual_num": 0,
            "output_file": "response_chapter_notice0000.html",
            "status": "completed",
        },
        "0_chapter_notice0001": {
            "actual_num": 0,
            "output_file": "response_chapter_notice0001.html",
            "status": "completed",
            "refinement_status": "not_refined",
        },
    }

    resolved_key = progress.update_refinement_status(
        1,
        0,
        "hash-notice-0001-refined",
        "response_chapter_notice0001.html",
        "refined",
        chapter_obj={
            "spine_order": 2,
            "original_basename": "chapter_notice0001.xhtml",
        },
    )

    assert resolved_key == "0_chapter_notice0001"
    assert "0@2" not in progress.prog["chapters"]
    progress.prog["chapters"][resolved_key]["unrefined_backup_file"] = (
        "_unrefined/response_chapter_notice0001.html"
    )
    assert progress.prog["chapters"][resolved_key]["refinement_status"] == "refined"
    assert progress.prog["chapters"][resolved_key]["unrefined_backup_file"] == (
        "_unrefined/response_chapter_notice0001.html"
    )


def test_fallback_temp_client_receives_pre_send_callback_context():
    source = UnifiedClient.__new__(UnifiedClient)
    source._thread_local = threading.local()
    temp = UnifiedClient.__new__(UnifiedClient)
    temp._thread_local = threading.local()

    callback_calls = []

    def callback():
        callback_calls.append("called")

    source.set_chapter_context(chapter=259, chunk=1, total_chunks=1)
    source_tls = source._get_thread_local_client()
    source_tls.current_request_id = "req-259"
    source_tls.current_request_context = "refinement"
    source_tls.pre_api_call_callback = None
    source_tls.last_pre_api_call_callback = callback
    source_tls.last_pre_api_call_callback_request_id = "req-259"

    source._copy_retry_request_context_to_temp_client(
        temp,
        context="refinement",
        request_id="req-259",
    )

    temp_tls = temp._get_thread_local_client()
    assert temp_tls.current_request_id == "req-259"
    assert temp_tls.current_request_context == "refinement"
    assert temp_tls.chapter_context["chapter"] == 259
    assert temp_tls.chapter_context["chunk"] == 1
    assert temp_tls.pre_api_call_callback is callback
    temp_tls.pre_api_call_callback()
    assert callback_calls == ["called"]


def test_progress_display_selector_prefers_active_and_refined_entries():
    previous = {
        "status": "completed",
        "model_name": "gemini-3.1-flash-lite",
        "refinement_status": "not_refined",
        "last_updated": 1,
    }
    active = {
        "status": "in_progress",
        "model_name": "deepseek-v4",
        "previous_progress_entry": previous,
        "last_updated": 2,
    }
    assert _select_progress_entry_for_display([previous, active], "in_progress") is active
    assert _progress_entry_model_for_display(active) == "deepseek-v4"

    plain_completed = {
        "status": "completed",
        "model_name": "gemini-3.1-flash-lite",
        "refinement_status": "not_refined",
        "last_updated": 10,
    }
    refined_completed = {
        "status": "completed",
        "model_name": "deepseek-v4",
        "refinement_status": "refined",
        "last_updated": 2,
    }
    selected = _select_progress_entry_for_display(
        [plain_completed, refined_completed],
        "completed",
    )
    assert selected is refined_completed
    assert _progress_entry_refined_for_display(selected)

    nested_history = {
        "status": "completed",
        "previous_progress_entry": {
            "status": "in_progress",
            "previous_progress_entry": refined_completed,
        },
    }
    assert _progress_entry_model_for_display(nested_history) == "deepseek-v4"
    assert _progress_entry_refined_for_display(nested_history)


def test_image_only_completed_progress_rows_show_copied_without_inheriting_badge():
    image_only_entry = {
        "status": "completed_image_only",
        "output_file": "part0003_split_000.html",
    }
    wrapped_row = {
        "status": "completed",
        "info": image_only_entry,
        "output_file": "part0003_split_000.html",
    }

    assert _progress_entry_is_completed_image_only_for_display(image_only_entry)
    assert _progress_entry_is_completed_image_only_for_display(wrapped_row)
    assert _progress_entry_model_for_display(image_only_entry) == "COPIED"
    progress_ui = RetranslationMixin()
    assert progress_ui._progress_entry_model_name(wrapped_row, {}) == "COPIED"

    display, display_status = progress_ui._progress_list_display_text(
        {
            "num": 3,
            "status": "completed",
            "info": image_only_entry,
            "output_file": "part0003_split_000.html",
        },
        {"show_model_info_state": True},
        20,
        25,
    )
    assert display_status == "completed"
    assert "📸 Image Only (Completed)" in display
    assert display.endswith("COPIED")

    active_retranslation = {
        "status": "in_progress",
        "previous_progress_entry": image_only_entry,
    }
    assert not _progress_entry_is_completed_image_only_for_display(active_retranslation)


def test_glossary_progress_legend_includes_refinement_rows():
    stats = _combine_glossary_progress_legend_stats(
        {
            "total": 674,
            "completed": 109,
            "skipped": 3,
            "in_progress": 0,
            "failed": 0,
            "merged": 565,
            "remaining": 0,
        },
        {
            "completed": 4,
            "in_progress": 6,
            "not_refined": 2,
            "refine_failed": 1,
            "skipped": 2,
            "partially_in_progress": 1,
        },
    )

    assert stats == {
        "total": 690,
        "completed": 113,
        "skipped": 5,
        "in_progress": 7,
        "failed": 0,
        "merged": 565,
        "remaining": 0,
        "not_refined": 2,
        "refine_failed": 1,
    }


def test_glossary_refinement_aggregate_status_precedence():
    assert _derive_glossary_refinement_aggregate_status(["completed", "failed"]) == "failed"
    assert _derive_glossary_refinement_aggregate_status(["completed", "in_progress"]) == "partially_in_progress"
    assert _derive_glossary_refinement_aggregate_status(["not_refined", "in_progress"]) == "partially_in_progress"
    assert _derive_glossary_refinement_aggregate_status(["in_progress", "in_progress"]) == "in_progress"
    assert _derive_glossary_refinement_aggregate_status(["in_progress", "skipped"]) == "in_progress"
    assert _derive_glossary_refinement_aggregate_status(["completed", "not_refined"]) == "not_refined"
    assert _derive_glossary_refinement_aggregate_status(["skipped", "skipped"]) == "skipped"
    assert _derive_glossary_refinement_aggregate_status(["completed", "skipped"]) == "completed"
    assert _derive_glossary_refinement_aggregate_status(
        ["completed", "skipped"],
        [True, False],
    ) == "not_refined"


def test_glossary_refinement_aggregate_selection_wins_over_specific_types():
    active = ["character", "term", "locations"]

    assert _normalize_glossary_refinement_selection(
        ["type::character", "all::character,term,locations", "type::locations"],
        active,
    ) == active
    assert _normalize_glossary_refinement_selection(
        ["type::terms", "type::locations"],
        active,
    ) == ["term", "locations"]


def test_glossary_refinement_aggregate_history_matches_scope_without_type_order():
    exact = {
        "status": "completed",
        "entry_count_before": 813,
        "entry_count_after": 653,
    }
    refinement = {
        "all::titles,character,terms": exact,
        "all::character,terms": {
            "status": "completed",
            "entry_count_before": 700,
            "entry_count_after": 600,
        },
    }

    assert _find_matching_glossary_refinement_aggregate(
        refinement,
        ["character", "terms", "titles"],
    ) is exact
    assert _find_matching_glossary_refinement_aggregate(
        refinement,
        ["character", "terms", "titles", "locations"],
    ) is None


def test_refinement_dialog_uses_triggering_panels_top_level_window():
    glossary_window = object()
    glossary_panel = types.SimpleNamespace(window=lambda: glossary_window)

    assert _resolve_dialog_window_parent(glossary_panel) is glossary_window
    assert _resolve_dialog_window_parent(None) is None


def test_glossary_progress_index_uses_filename_before_full_spine_row():
    progress_data = {
        "indexing": "chapter_index_zero_based",
        "chapter_filenames": {
            "0": "info.xhtml",
            "1": "chapter0001.xhtml",
        },
    }
    filename_key_to_index = {}
    for view_index, filename in enumerate(("cover.html", "info.xhtml", "chapter0001.xhtml")):
        for key in _glossary_progress_filename_keys(filename):
            filename_key_to_index[key] = view_index

    assert _map_zero_based_glossary_progress_index(0, progress_data, filename_key_to_index) == 1
    assert _map_zero_based_glossary_progress_index(1, progress_data, filename_key_to_index) == 2


def test_glossary_stop_reset_clears_unified_module_and_class_flags():
    import extract_glossary_from_epub as glossary
    import unified_api_client as unified
    import antigravity_proxy

    try:
        unified.set_stop_flag(True)
        assert unified.is_stop_requested()
        assert antigravity_proxy.is_cancelled()

        glossary.set_stop_flag(False)

        assert not glossary.is_stop_requested()
        assert not unified.is_stop_requested()
        assert not UnifiedClient.is_globally_cancelled()
        assert not antigravity_proxy.is_cancelled()
    finally:
        glossary.set_stop_flag(False)
        unified.set_stop_flag(False)


def test_glossary_explicit_user_cancel_is_not_retried_as_timeout(monkeypatch):
    from extract_glossary_from_epub import send_with_interrupt
    from unified_api_client import UnifiedClientError

    class CancelledClient:
        _multi_key_mode = False
        client_type = "openai"

        def __init__(self):
            self.calls = 0

        def send(self, *args, **kwargs):
            self.calls += 1
            raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")

    monkeypatch.setenv("TIMEOUT_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("RETRY_TIMEOUT", "0")
    client = CancelledClient()

    with pytest.raises(UnifiedClientError, match="Operation cancelled by user"):
        send_with_interrupt([], client, 0, 10, lambda: False)

    assert client.calls == 1


def test_graceful_stop_drains_only_after_a_result_is_committed(monkeypatch):
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "1")
    skipped = "Graceful stop active - not starting new API call"

    assert _is_graceful_stop_skip_error(skipped)
    assert not _graceful_stop_should_drain_after_result(False)
    assert _graceful_stop_should_drain_after_result(True)


def test_graceful_stop_distinguishes_queued_skips_from_force_stop(monkeypatch):
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")

    assert _is_graceful_stop_skip_error(
        "Glossary extraction stopped by user (skipped before API call)"
    )
    assert _is_graceful_stop_skip_error(
        "Glossary extraction stopped by user during threading delay"
    )
    assert not _glossary_is_hard_stop_requested(lambda: True)

    # A second click changes GRACEFUL_STOP to 0 before forcing cancellation.
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    assert _glossary_is_hard_stop_requested(lambda: False)


def test_graceful_result_branches_do_not_cancel_or_break_remaining_futures():
    source = textwrap.dedent(inspect.getsource(extract_glossary_main))
    tree = ast.parse(source)
    drain_branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_graceful_stop_should_drain_after_result"
            for child in ast.walk(node.test)
        ):
            drain_branches.append(node)

    assert len(drain_branches) == 2
    for branch in drain_branches:
        assert not any(isinstance(child, ast.Break) for child in ast.walk(branch))
        assert not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "cancel_all_futures"
            for child in ast.walk(branch)
        )


def test_merged_children_require_exact_successful_group_result():
    submitted = [12, 13, 14, 15, 16]
    successful = {
        "merged_indices": submitted[1:],
        "results": [
            {
                "idx": submitted[0],
                "data": [{"raw_name": "원문"}],
                "resp": "[result]",
                "finish_reason": "stop",
                "error": None,
            },
            *[
                {
                    "idx": idx,
                    "data": [],
                    "resp": "",
                    "error": None,
                    "merged_into": submitted[0],
                }
                for idx in submitted[1:]
            ],
        ],
    }

    assert _confirmed_merged_child_indices(successful, submitted) == submitted[1:]

    graceful_skip = {
        "merged_indices": submitted[1:],
        "results": [
            {
                "idx": idx,
                "data": [],
                "resp": "",
                "error": "Graceful stop active - not starting new API call",
            }
            for idx in submitted
        ],
    }
    assert _confirmed_merged_child_indices(graceful_skip, submitted) == []

    wrong_group = dict(successful, merged_indices=[18, 19, 20, 21])
    assert _confirmed_merged_child_indices(wrong_group, submitted) == []


def test_merged_children_rejected_when_parent_output_is_unusable():
    submitted = [7, 8, 9]
    empty_parent = {
        "merged_indices": submitted[1:],
        "results": [
            {"idx": 7, "data": [], "resp": "[]", "error": None},
            {"idx": 8, "data": [], "resp": "", "error": None, "merged_into": 7},
            {"idx": 9, "data": [], "resp": "", "error": None, "merged_into": 7},
        ],
    }

    assert _confirmed_merged_child_indices(empty_parent, submitted) == []


def test_glossary_stop_restores_previous_progress_entries_atomically(tmp_path):
    progress_file = tmp_path / "book_glossary_progress.json"
    failed_entry = {
        "chapter_index": 4,
        "actual_num": 4,
        "status": "failed",
        "output_file": "chapter0004.xhtml",
        "model_name": "previous-model",
    }
    completed_entry = {
        "chapter_index": 6,
        "actual_num": 6,
        "status": "completed",
        "output_file": "chapter0006.xhtml",
        "model_name": "previous-model",
    }
    progress_file.write_text(
        json.dumps(
            {
                "chapters": {
                    "4": {
                        **failed_entry,
                        "status": "in_progress",
                        "model_name": "current-model",
                        "previous_status": "failed",
                        "previous_progress_entry": failed_entry,
                    },
                    "6": {
                        **completed_entry,
                        "status": "in_progress",
                        "model_name": "current-model",
                        "previous_status": "completed",
                        "previous_progress_entry": completed_entry,
                    },
                    "7": {
                        "chapter_index": 7,
                        "actual_num": 7,
                        "status": "in_progress",
                        "output_file": "chapter0007.xhtml",
                        "previous_status": "not_completed",
                    },
                },
                "completed": [],
                "failed": [],
                "merged_indices": [],
                "in_progress": [4, 6, 7],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    context = make_glossary_progress_context(progress_file=str(progress_file))

    restored = _restore_glossary_in_progress_file(context)
    on_disk = json.loads(progress_file.read_text(encoding="utf-8"))

    assert restored == on_disk
    assert on_disk["chapters"]["4"] == failed_entry
    assert on_disk["chapters"]["6"] == completed_entry
    assert "7" not in on_disk["chapters"]
    assert on_disk["completed"] == [6]
    assert on_disk["failed"] == [4]
    assert on_disk["in_progress"] == []


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        (None, ""),
        ("", ""),
        ("chapter0001.xhtml", "chapter0001"),
        ("response_chapter0001.html", "chapter0001"),
        ("OEBPS/Text/response_chapter0001.htm.html.xhtml", "chapter0001"),
        (Path("Text") / "response_chapter_notice004.xhtml", "chapter_notice004"),
    ],
)
def test_normalize_progress_match_name_strips_response_prefix_and_all_extensions(
    raw_name,
    expected,
):
    assert _normalize_progress_match_name(raw_name) == expected


@pytest.mark.parametrize(
    ("display_info", "expected"),
    [
        ({"output_file": "response_chapter.html"}, True),
        ({"info": {"output_file": "Text/chapter.xhtml"}}, True),
        ({"output_file": "chapter.htm"}, True),
        ({"output_file": "chapter.txt"}, False),
        ({"info": {"output_file": "metadata.json"}}, False),
    ],
)
def test_progress_epub_reader_action_is_limited_to_html_entries(
    display_info,
    expected,
):
    assert _progress_item_is_html(display_info) is expected


def test_progress_epub_reader_matches_response_name_to_source_member():
    members = [
        "META-INF/container.xml",
        "OEBPS/Text/chapter0001.xhtml",
        "OEBPS/Text/chapter0002.xhtml",
    ]

    matched = _match_epub_html_member_basename(
        members,
        ["response_chapter0002.htm.html", "chapter0002"],
    )

    assert matched == "chapter0002.xhtml"


def test_progress_epub_reader_reuses_member_index_for_multiple_rows(monkeypatch):
    members = [
        "META-INF/container.xml",
        "OEBPS/Text/chapter0001.xhtml",
        "OEBPS/Text/chapter0002.xhtml",
    ]
    member_index = _index_epub_html_members(members)

    def unexpected_reindex(_members):
        raise AssertionError("prebuilt EPUB member index was not reused")

    monkeypatch.setattr(
        "Retranslation_GUI._index_epub_html_members",
        unexpected_reindex,
    )

    assert _match_epub_html_member_basename(
        members,
        ["response_chapter0001.html"],
        member_index=member_index,
    ) == "chapter0001.xhtml"
    assert _match_epub_html_member_basename(
        members,
        ["response_chapter0002.html"],
        member_index=member_index,
    ) == "chapter0002.xhtml"


def test_snapshot_progress_output_dir_scans_large_directory_once(tmp_path, monkeypatch):
    file_count = 2_500
    sample_index = 249
    sample_name = f"response_chapter_{sample_index:04d}.html.xhtml"

    for index in range(file_count):
        (tmp_path / f"response_chapter_{index:04d}.html.xhtml").write_bytes(b"x")

    # Directories must not leak into any of the file lookup structures.
    (tmp_path / "response_chapter_directory.html.xhtml").mkdir()

    sample_path = tmp_path / sample_name
    known_mtime_ns = 1_700_000_123_000_000_000
    os.utime(sample_path, ns=(known_mtime_ns, known_mtime_ns))
    expected_sample_mtime = sample_path.stat().st_mtime

    real_scandir = os.scandir
    scandir_calls = []
    listdir_calls = []

    def counted_scandir(path):
        scandir_calls.append(os.fspath(path))
        return real_scandir(path)

    def forbidden_listdir(path):
        listdir_calls.append(os.fspath(path))
        raise AssertionError("the output snapshot must not fall back to os.listdir")

    monkeypatch.setattr("Retranslation_GUI.os.scandir", counted_scandir)
    monkeypatch.setattr("Retranslation_GUI.os.listdir", forbidden_listdir)

    filenames, normalized, mtimes = _snapshot_progress_output_dir(tmp_path)

    assert scandir_calls == [os.fspath(tmp_path)]
    assert listdir_calls == []
    assert len(filenames) == file_count
    assert len(normalized) == file_count
    assert len(mtimes) == file_count
    assert sample_name in filenames
    assert normalized[f"chapter_{sample_index:04d}"] == sample_name
    assert mtimes[sample_name] == pytest.approx(expected_sample_mtime)
    assert "response_chapter_directory.html.xhtml" not in filenames
    assert "chapter_directory" not in normalized


def test_progress_path_signature_tracks_mtime_and_size_changes(tmp_path):
    progress_path = tmp_path / "translation_progress.json"
    assert _progress_path_signature(progress_path) is None

    progress_path.write_text("{}", encoding="utf-8")
    initial = _progress_path_signature(progress_path)
    initial_stat = progress_path.stat()
    assert initial == (initial_stat.st_mtime_ns, initial_stat.st_size)

    # Keep the same payload size but force a representable mtime change.
    changed_mtime_ns = initial_stat.st_mtime_ns + 2_000_000_000
    os.utime(progress_path, ns=(changed_mtime_ns, changed_mtime_ns))
    mtime_changed = _progress_path_signature(progress_path)
    changed_stat = progress_path.stat()
    assert mtime_changed == (changed_stat.st_mtime_ns, changed_stat.st_size)
    assert mtime_changed != initial
    assert mtime_changed[1] == initial[1]

    progress_path.write_text('{"chapters": {}}', encoding="utf-8")
    size_changed = _progress_path_signature(progress_path)
    assert size_changed != mtime_changed
    assert size_changed[1] != mtime_changed[1]

    progress_path.unlink()
    assert _progress_path_signature(progress_path) is None


def test_progress_manager_source_link_updates_epub_library_scan(
    tmp_path, monkeypatch
):
    import epub_library

    library_dir = tmp_path / "Library"
    output_root = tmp_path / "Output"
    source_epub = tmp_path / "source.epub"
    workspace = output_root / source_epub.stem
    library_dir.mkdir()
    workspace.mkdir(parents=True)

    with zipfile.ZipFile(source_epub, "w") as epub_zip:
        epub_zip.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf"/>
  </rootfiles>
</container>""",
        )
        epub_zip.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter1" href="Text/chapter0001.xhtml"
          media-type="application/xhtml+xml"/>
    <item id="chapter2" href="Text/chapter0002.xhtml"
          media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
  </spine>
</package>""",
        )

    (workspace / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "__metadata__": {
                        "status": "pending",
                        "output_file": "metadata.json",
                        "original_basename": "metadata.json",
                        "is_special": True,
                        "special_type": "metadata",
                        "metadata_progress_key": "__metadata__",
                    }
                },
                "chapter_chunks": {},
                "version": "2.1",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("TRANSLATE_SPECIAL_FILES", raising=False)
    monkeypatch.setattr(
        epub_library, "get_library_dir", lambda: str(library_dir)
    )
    monkeypatch.setattr(
        epub_library,
        "_resolve_output_roots",
        lambda _config=None: [str(output_root)],
    )

    assert _persist_progress_manager_source_link(source_epub, workspace)
    assert (workspace / "source_epub.txt").read_text(
        encoding="utf-8"
    ) == str(source_epub.resolve())
    assert epub_library.load_library_raw_inputs() == [
        str(source_epub.resolve())
    ]

    books = epub_library.scan_output_folders(
        {"translate_special_files": True}
    )

    assert len(books) == 1
    assert books[0]["raw_source_path"] == str(source_epub.resolve())
    assert books[0]["missing_raw_file"] is False
    assert books[0]["total_chapters"] == 2
    assert books[0]["completed_chapters"] == 0


def test_initial_spine_matching_has_no_directory_scan_inside_spine_loop():
    source = textwrap.dedent(
        inspect.getsource(RetranslationMixin._force_retranslation_epub_or_text)
    )
    tree = ast.parse(source)

    spine_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and any(
            isinstance(child, ast.Name) and child.id == "spine_chapters"
            for child in ast.walk(node.iter)
        )
    ]
    assert spine_loops, "expected to find the initial loops over spine_chapters"

    per_spine_listdir_calls = []
    for loop in spine_loops:
        for node in ast.walk(loop):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr == "listdir"
                and isinstance(owner, ast.Name)
                and owner.id == "os"
            ):
                per_spine_listdir_calls.append(node.lineno)

    assert per_spine_listdir_calls == []


def test_parallel_glossary_progress_filters_raw_spine_to_mapped_files():
    chapter_map = {
        0: "0000_Information.xhtml",
        1: "chapter0001.xhtml",
        2: "chapter0002.xhtml",
        3: "chapter0003.xhtml",
    }
    spine_map = {0: 1, 1: 2, 2: 3, 3: 4}

    filtered, filtered_spine = _filter_glossary_source_chapter_map(
        chapter_map,
        spine_map,
        ["chapter0001.xhtml", "chapter0003.xhtml"],
    )

    assert filtered == {0: "chapter0001.xhtml", 1: "chapter0003.xhtml"}
    assert filtered_spine == {0: 2, 1: 4}


def test_parallel_glossary_progress_maps_generated_pair_names_to_raw_rows():
    aliases = _parallel_glossary_progress_filename_aliases(
        ["chapter0001.xhtml", "chapter0003.xhtml"]
    )
    progress_data = {
        "indexing": "chapter_index_zero_based",
        "chapter_filenames": {
            "0": "pair_0001.xhtml",
            "1": "pair_0002.xhtml",
        },
    }

    assert aliases["pair_0001.xhtml"] == 0
    assert aliases["pair_0001"] == 0
    assert aliases["pair_0002.xhtml"] == 1
    assert (
        _map_zero_based_glossary_progress_index(0, progress_data, aliases) == 0
    )
    assert (
        _map_zero_based_glossary_progress_index(1, progress_data, aliases) == 1
    )


def test_progress_manager_routes_parallel_pair_through_raw_epub(tmp_path):
    raw_path = tmp_path / "raw.epub"
    generated_path = tmp_path / "raw_parallel_epub_pair.epub"
    raw_path.write_bytes(b"raw")
    generated_path.write_bytes(b"paired")
    cache_key = f"parallel::{raw_path}::{generated_path}"

    gui = RetranslationMixin()
    gui._parallel_epub_progress_manager_context = lambda: {
        "raw_path": str(raw_path),
        "generated_path": str(generated_path),
        "raw_filenames": ["chapter0001.xhtml", "chapter0002.xhtml"],
        "cache_key": cache_key,
    }
    opened = []
    gui._show_retranslation_shell_then_build = lambda *args, **kwargs: opened.append(
        (args, kwargs)
    )

    gui.force_retranslation()

    assert opened == [
        (
            (str(raw_path),),
            {
                "show_special_files_state": False,
                "cache_key": cache_key,
                "glossary_progress_source_path": str(generated_path),
                "glossary_progress_source_filenames": [
                    "chapter0001.xhtml",
                    "chapter0002.xhtml",
                ],
            },
        )
    ]


def test_progress_managers_use_event_driven_differential_refresh():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "Retranslation_GUI.py"
    ).read_text(encoding="utf-8")

    assert "QFileSystemWatcher" in source
    assert "progress_watch_debounce.setInterval(_PROGRESS_WATCH_DEBOUNCE_MS)" in source
    assert "gp_watch_debounce.setInterval(_PROGRESS_WATCH_DEBOUNCE_MS)" in source
    assert "_PROGRESS_WATCH_DEBOUNCE_MS = 500" in source
    assert "_PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS = 0.5" in source
    assert "prefetch_bridge.finished.emit(payload)" in source
    assert "_gp_timer.setInterval(2000)" in source
    assert "_auto_refresh_timer.setInterval(2000)" in source
    assert "_row_fingerprints" in source
    assert "_prefetch_ready" not in source
    assert "fallback_prog = copy.deepcopy(data.get('prog') or {})" not in source

    glossary_start = source.index("        def _show_glossary_progress():")
    glossary_end = source.index(
        "        glossary_progress_btn.clicked.connect(_show_glossary_progress)",
        glossary_start,
    )
    glossary_source = source[glossary_start:glossary_end]
    assert 'AnimatedRefreshButton("  Refresh")' in glossary_source
    assert "Full Refresh" not in glossary_source
    assert "on_complete=_finish_refresh_animation" in glossary_source
    assert "from PySide6.QtWidgets import QComboBox, QStackedWidget" not in glossary_source
    assert "dialog._show_glossary_progress = _show_glossary_progress" in source

    glossary_populate_start = source.index(
        "            def _populate_gp_listbox(_d, chunk_size=150):"
    )
    glossary_populate_end = source.index(
        "            _populate_gp_listbox(gp_data)",
        glossary_populate_start,
    )
    glossary_populate_source = source[glossary_populate_start:glossary_populate_end]
    assert "gp_listbox.clear()" not in glossary_populate_source
    assert "gp_listbox.insertItem(ci, item)" in glossary_populate_source


def test_progress_row_identity_is_stable_when_translation_starts():
    gui = RetranslationMixin()
    waiting = {
        "key": "chapter0001.xhtml",
        "num": 1,
        "output_file": "chapter0001.xhtml",
        "original_filename": "chapter0001.xhtml",
        "opf_position": 15,
    }
    active = {
        **waiting,
        "progress_key": "temporary-progress-hash",
        "status": "in_progress",
    }

    assert gui._progress_list_item_key(waiting) == gui._progress_list_item_key(active)


def test_progress_row_payload_revision_avoids_deep_prompt_comparison():
    gui = RetranslationMixin()
    original = {
        "num": 1,
        "status": "completed",
        "output_file": "chapter0001.xhtml",
        "info": {
            "status": "completed",
            "last_updated": 10,
            "saved_prompt": "a" * 100_000,
        },
    }
    same_visible_revision = copy.deepcopy(original)
    same_visible_revision["info"]["saved_prompt"] = "b" * 100_000
    changed_revision = copy.deepcopy(original)
    changed_revision["info"]["last_updated"] = 11

    assert (
        gui._progress_list_payload_revision(original)
        == gui._progress_list_payload_revision(same_visible_revision)
    )
    assert (
        gui._progress_list_payload_revision(original)
        != gui._progress_list_payload_revision(changed_revision)
    )


def test_unchanged_progress_rows_skip_display_formatting():
    gui = RetranslationMixin()
    info = {
        "key": "chapter0001.xhtml",
        "num": 1,
        "status": "in_progress",
        "output_file": "chapter0001.xhtml",
        "original_filename": "chapter0001.xhtml",
        "opf_position": 0,
        "info": {
            "status": "in_progress",
            "last_updated": 10,
        },
    }

    class _Item:
        def __init__(self):
            self.roles = {
                retranslation_gui_module.Qt.UserRole: {
                    "item_key": gui._progress_list_item_key(info),
                    "payload_revision": gui._progress_list_payload_revision(info),
                }
            }

        def data(self, role):
            return self.roles.get(role)

    class _Listbox:
        def __init__(self):
            self.row = _Item()

        def count(self):
            return 1

        def item(self, index):
            return self.row if index == 0 else None

    gui._progress_list_display_text = lambda *_args: (_ for _ in ()).throw(
        AssertionError("unchanged rows must not be formatted")
    )
    data = {
        "listbox": _Listbox(),
        "chapter_display_info": [info],
        "show_special_files_state": False,
        "show_model_info_state": False,
        "_progress_list_view_revision": (False, False),
    }

    gui._update_listbox_display(data)


def test_streamed_progress_reconcile_does_not_clear_or_queue_scroll_restores():
    stream_source = inspect.getsource(
        RetranslationMixin._populate_progress_listbox_streamed
    )
    refresh_source = inspect.getsource(RetranslationMixin._refresh_retranslation_data)

    assert "listbox.clear()" not in stream_source
    assert "listbox.takeItem" in stream_source
    assert "scrollToItem" not in refresh_source
    assert "_restore_scroll_again" not in refresh_source

    list_update_source = inspect.getsource(
        RetranslationMixin._update_listbox_display
    )
    assert "old_payload.get('info') != info" not in list_update_source
    assert (
        list_update_source.index("payload_revision =")
        < list_update_source.index("self._progress_list_display_text(")
    )
    assert "_PROGRESS_DIRECT_ROW_UPDATE_LIMIT" in list_update_source


def test_retranslate_selected_bulk_reset_runs_off_the_qt_thread():
    source = Path(retranslation_gui_module.__file__).read_text(encoding="utf-8")
    reset_start = source.index("def retranslate_selected():")
    reset_end = source.index("# Add buttons", reset_start)
    reset_source = source[reset_start:reset_end]

    assert 'yield "run_background"' in reset_source
    assert 'yield "apply_ui"' in reset_source
    assert 'name="progress-retranslate-selected"' in reset_source
    assert "working_progress = copy.deepcopy(progress_baseline)" in reset_source
    assert "merged_children_by_parent" in reset_source
    assert "for child_key, child_data in list(data['prog']" not in reset_source


def test_open_progress_managers_rebuild_when_input_signature_changes():
    refresh_source = inspect.getsource(
        RetranslationMixin._refresh_open_progress_managers_for_input_change
    )

    assert "_progress_manager_input_signature" in refresh_source
    assert "dialog.deleteLater()" in refresh_source
    assert "self.force_retranslation()" in refresh_source
    assert "self._reopen_glossary_progress_after_input_change()" in refresh_source


def test_live_epub_progress_matching_is_prepared_off_the_gui_thread():
    builder_source = inspect.getsource(
        RetranslationMixin._add_retranslation_buttons_opf
    )

    assert "prepared_data = {" in builder_source
    assert "append_auxiliary=False" in builder_source
    assert "prepared_chapter_display_info" in builder_source
    assert "_deferred_prefetched_progress_payload" in builder_source


def test_background_spine_rematch_can_skip_live_qt_auxiliary_rows(tmp_path):
    gui = RetranslationMixin()
    gui.config = {}
    auxiliary_calls = []
    gui._append_chunk_progress_display_info = lambda *_: auxiliary_calls.append(
        "chunks"
    )
    gui._append_metadata_display_info = lambda *_: auxiliary_calls.append(
        "metadata"
    )
    gui._append_translation_artifact_display_info = (
        lambda *_: auxiliary_calls.append("artifacts")
    )
    gui._append_pdf_ocr_display_info = lambda *_: auxiliary_calls.append("ocr")
    gui._append_image_gen_display_info = lambda *_: auxiliary_calls.append(
        "images"
    )
    data = {
        "prog": {"chapters": {}},
        "output_dir": str(tmp_path),
        "spine_chapters": [
            {
                "filename": "part0001.xhtml",
                "file_chapter_num": 1,
                "display_chapter_num": 1,
                "position": 0,
                "status": "unknown",
                "output_file": None,
                "is_special": False,
            }
        ],
        "_prefetched_output_listing": set(),
        "_refresh_read_only": True,
    }

    gui._rematch_spine_chapters(data, append_auxiliary=False)

    assert auxiliary_calls == []
    assert data["chapter_display_info"][0]["display_num"] == 1
    assert data["chapter_display_info"][0]["status"] == "not_translated"


def test_streamed_progress_reconcile_applies_only_latest_deferred_snapshot():
    stream_source = inspect.getsource(
        RetranslationMixin._populate_progress_listbox_streamed
    )

    assert "_deferred_prefetched_progress_payload" in stream_source
    assert "_apply_prefetched_progress_payload" in stream_source


def test_manual_glossary_refinement_has_no_hidden_model_fallback():
    source = inspect.getsource(RetranslationMixin)

    assert "or 'gemini-2.0-flash'" not in source
    assert "config.get('model') or os.getenv('MODEL')" not in source
    assert "_require_model_selection" in source


def test_refinement_override_checkbox_label_background_is_transparent():
    source = inspect.getsource(RetranslationMixin)
    selector = "QCheckBox#refinementOverrideCheckbox {"
    checkbox_style = source.split(selector, 1)[1].split("}", 1)[0]

    assert "background-color: transparent;" in checkbox_style
    assert "border: none;" in checkbox_style
