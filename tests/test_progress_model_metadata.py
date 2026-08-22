import ast
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

from Retranslation_GUI import (
    RetranslationMixin,
    _clear_llm_token_qa_markers,
    _clear_missing_image_qa_markers,
    _clear_refinement_progress_fields,
    _combine_glossary_progress_legend_stats,
    _filter_glossary_source_chapter_map,
    _glossary_progress_filename_keys,
    _index_epub_html_members,
    _match_epub_html_member_basename,
    _map_zero_based_glossary_progress_index,
    _normalize_progress_match_name,
    _parallel_glossary_progress_filename_aliases,
    _persist_progress_manager_source_link,
    _progress_entry_has_llm_token_qa,
    _progress_entry_has_missing_image_qa,
    _progress_path_signature,
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _progress_item_is_html,
    _repair_empty_attribute_qa_file,
    _snapshot_progress_output_dir,
    _select_progress_entry_for_display,
)
from TransateKRtoEN import (
    ContentProcessor,
    ProgressManager,
    TranslationConfig,
    _vision_ocr_header_markdown,
)
from image_translator import ImageTranslator
from unified_api_client import UnifiedClient, set_current_thread_actual_request_model
from extract_glossary_from_epub import (
    _confirmed_merged_child_indices,
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


@pytest.fixture(autouse=True)
def _clear_actual_request_metadata():
    set_current_thread_actual_request_model(None, None)
    yield
    set_current_thread_actual_request_model(None, None)


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
        "hash-1", 2, 3, "translated two", _epub_chunk_budget()
    )

    entry, reason, _changed = progress.prepare_chapter_chunk_progress(
        "hash-1", 3, _epub_chunk_budget(), enabled=True
    )

    assert reason is None
    assert entry["completed"] == [2]
    assert entry["chunks"] == {"2": "translated two"}


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


def test_glossary_progress_legend_includes_refinement_rows():
    stats = _combine_glossary_progress_legend_stats(
        {
            "total": 674,
            "completed": 109,
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
        },
    )

    assert stats == {
        "total": 687,
        "completed": 113,
        "in_progress": 6,
        "failed": 0,
        "merged": 565,
        "remaining": 0,
        "not_refined": 2,
        "refine_failed": 1,
    }


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
    assert "progress_watch_debounce.setInterval(100)" in source
    assert "gp_watch_debounce.setInterval(100)" in source
    assert "prefetch_bridge.finished.emit(payload)" in source
    assert "_gp_timer.setInterval(2000)" in source
    assert "_auto_refresh_timer.setInterval(2000)" in source
    assert "_row_fingerprints" in source
    assert "_prefetch_ready" not in source

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


def test_streamed_progress_reconcile_does_not_clear_or_queue_scroll_restores():
    stream_source = inspect.getsource(
        RetranslationMixin._populate_progress_listbox_streamed
    )
    refresh_source = inspect.getsource(RetranslationMixin._refresh_retranslation_data)

    assert "listbox.clear()" not in stream_source
    assert "listbox.takeItem" in stream_source
    assert "scrollToItem" not in refresh_source
    assert "_restore_scroll_again" not in refresh_source


def test_open_progress_managers_rebuild_when_input_signature_changes():
    refresh_source = inspect.getsource(
        RetranslationMixin._refresh_open_progress_managers_for_input_change
    )

    assert "_progress_manager_input_signature" in refresh_source
    assert "dialog.deleteLater()" in refresh_source
    assert "self.force_retranslation()" in refresh_source
    assert "self._reopen_glossary_progress_after_input_change()" in refresh_source
