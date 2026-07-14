import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
import threading

import pytest

from Retranslation_GUI import (
    RetranslationMixin,
    _glossary_progress_filename_keys,
    _map_zero_based_glossary_progress_index,
    _normalize_progress_match_name,
    _progress_path_signature,
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _snapshot_progress_output_dir,
    _select_progress_entry_for_display,
)
from TransateKRtoEN import ProgressManager
from unified_api_client import UnifiedClient, set_current_thread_actual_request_model
from extract_glossary_from_epub import (
    make_glossary_progress_context,
    _restore_glossary_in_progress_file,
)


@pytest.fixture(autouse=True)
def _clear_actual_request_metadata():
    set_current_thread_actual_request_model(None, None)
    yield
    set_current_thread_actual_request_model(None, None)


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

    try:
        unified.set_stop_flag(True)
        assert unified.is_stop_requested()

        glossary.set_stop_flag(False)

        assert not glossary.is_stop_requested()
        assert not unified.is_stop_requested()
        assert not UnifiedClient.is_globally_cancelled()
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
