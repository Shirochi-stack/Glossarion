import threading

import pytest

from Retranslation_GUI import (
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _select_progress_entry_for_display,
)
from TransateKRtoEN import ProgressManager
from unified_api_client import UnifiedClient, set_current_thread_actual_request_model


@pytest.fixture(autouse=True)
def _clear_actual_request_metadata():
    set_current_thread_actual_request_model(None, None)
    yield
    set_current_thread_actual_request_model(None, None)


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
