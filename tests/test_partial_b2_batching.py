import inspect
import threading
import time
from pathlib import Path

from TransateKRtoEN import (
    _multipass_graceful_stop_requested,
    _multipass_hard_stop_requested,
    _multipass_stop_requested,
    _partial_b2_batch_worker_count,
    _partial_b2_entries_per_request,
    _process_refinement_or_tts_mode,
    _restore_interrupted_refinement_snapshot,
    _run_partial_b2_request_batches,
)


def test_partial_b2_manual_entry_cap_is_preserved():
    class Config:
        PARTIAL_B2_ENTRIES_PER_REQUEST = 50

        @staticmethod
        def get_effective_output_limit():
            return 8192

        @staticmethod
        def get_effective_compression_factor():
            return 3.0

    entries, _max_tokens, _compression, _available, automatic = (
        _partial_b2_entries_per_request(Config())
    )
    assert entries == 50
    assert automatic is False


def test_partial_b2_capped_calls_use_batch_workers_and_keep_result_order():
    request_batches = [[index] for index in range(1, 7)]
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def send_batch(batch_index, batch_requests):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            # Releasing the GIL here gives all configured workers a chance to
            # overlap, just as real network requests do.
            time.sleep(0.05)
            return batch_index, batch_requests[0]
        finally:
            with state_lock:
                active -= 1

    results = _run_partial_b2_request_batches(
        request_batches,
        use_batch=True,
        batch_size=3,
        stop_requested=lambda: False,
        send_batch=send_batch,
    )

    assert results == [(index, index) for index in range(1, 7)]
    assert max_active == 3
    assert _partial_b2_batch_worker_count(True, 3, 6) == 3


def test_partial_b2_remains_sequential_when_batch_mode_is_disabled():
    request_batches = [[index] for index in range(1, 4)]
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def send_batch(batch_index, _batch_requests):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            return batch_index
        finally:
            with state_lock:
                active -= 1

    results = _run_partial_b2_request_batches(
        request_batches,
        use_batch=False,
        batch_size=8,
        stop_requested=lambda: False,
        send_batch=send_batch,
    )

    assert results == [1, 2, 3]
    assert max_active == 1


def test_multipass_stop_state_distinguishes_graceful_from_force(monkeypatch):
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    # A stale cancellation flag or the shared GUI callback must not turn the
    # first graceful click into a hard stop.
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")

    assert _multipass_graceful_stop_requested() is True
    assert _multipass_stop_requested(lambda: True) is True
    assert _multipass_hard_stop_requested(lambda: True) is False

    # The second click clears graceful mode before latching hard cancellation.
    monkeypatch.setenv("GRACEFUL_STOP", "0")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    assert _multipass_hard_stop_requested(lambda: False) is True


def test_interrupted_refinement_restores_prior_model_and_refined_state(tmp_path):
    from TransateKRtoEN import ProgressManager

    progress = ProgressManager(str(tmp_path))
    snapshot = {
        "actual_num": 8,
        "output_file": "response_pdf_section_008.html",
        "status": "completed",
        "model_name": "authnd/deepseek-ai/deepseek-v4-pro-0813",
        "refinement_status": "refined",
        "refined_at": 123.0,
    }
    current = {
        "actual_num": 8,
        "output_file": "response_pdf_section_008.html",
        "status": "in_progress",
        "refinement_status": "in_progress",
        "previous_status": "completed",
        "previous_progress_entry": snapshot,
    }
    progress.prog["chapters"] = {"pdf:section-eight": current}

    restored = _restore_interrupted_refinement_snapshot(
        progress,
        "pdf:section-eight",
        current,
        snapshot=snapshot,
    )

    assert restored == snapshot
    assert progress.prog["chapters"]["pdf:section-eight"] == snapshot


def test_partial_b2_graceful_stop_drains_running_and_cancels_queued_batches():
    request_batches = [[index] for index in range(1, 7)]
    state = {"stop": False, "hard": False}
    started = []
    started_lock = threading.Lock()
    two_started = threading.Event()
    release_running = threading.Event()
    result_box = {}

    def send_batch(batch_index, _batch_requests):
        with started_lock:
            started.append(batch_index)
            if len(started) >= 2:
                two_started.set()
        assert release_running.wait(2.0)
        return batch_index

    def run_batches():
        result_box["results"] = _run_partial_b2_request_batches(
            request_batches,
            use_batch=True,
            batch_size=2,
            stop_requested=lambda: state["stop"],
            hard_stop_requested=lambda: state["hard"],
            send_batch=send_batch,
        )

    runner = threading.Thread(target=run_batches)
    runner.start()
    assert two_started.wait(2.0)
    state["stop"] = True
    # Give the 50 ms polling loop time to cancel futures that have not started,
    # while the two active calls remain blocked as real HTTP calls would be.
    time.sleep(0.15)
    release_running.set()
    runner.join(2.0)

    assert not runner.is_alive()
    assert started == [1, 2]
    assert result_box["results"] == [1, 2]


def test_partial_b2_force_stop_abandons_the_running_wait_immediately():
    state = {"stop": False, "hard": False}
    started = threading.Event()
    release_running = threading.Event()
    result_box = {}

    def send_batch(_batch_index, _batch_requests):
        started.set()
        assert release_running.wait(2.0)
        return "late result"

    def run_batches():
        try:
            _run_partial_b2_request_batches(
                [[1], [2]],
                use_batch=True,
                batch_size=2,
                stop_requested=lambda: state["stop"],
                hard_stop_requested=lambda: state["hard"],
                send_batch=send_batch,
            )
        except Exception as exc:
            result_box["error"] = exc

    runner = threading.Thread(target=run_batches)
    runner.start()
    assert started.wait(2.0)
    state.update(stop=True, hard=True)
    runner.join(0.5)

    try:
        assert not runner.is_alive()
        assert "force-stopped" in str(result_box["error"])
    finally:
        # Let the abandoned fake transport worker unwind before the test exits.
        release_running.set()


def test_all_multipass_refinement_sends_preserve_in_flight_graceful_calls():
    source = inspect.getsource(_process_refinement_or_tts_mode)

    # send_with_interrupt protects an in-flight request during GRACEFUL_STOP by
    # default.  The old override converted the first click into cancellation in
    # full/failed, partial, partial.b, and partial.b2 alike.
    assert "bypass_graceful_stop=True" not in source


def test_graceful_stop_groups_queued_refinement_chapter_logs(
    monkeypatch, capsys
):
    class Config:
        OUTPUT_MODE = "refinement"
        MULTIPASS_MODE = True
        MULTIPASS_REFINEMENT_MODE = "failed"
        BATCH_TRANSLATION = True
        BATCH_SIZE = 1

    class ProgressStub:
        def __init__(self):
            self.prog = {}

        def save(self):
            return None

    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    monkeypatch.setenv("BATCH_SIZE", "1")
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "0")
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)

    _process_refinement_or_tts_mode(
        Config(),
        object(),
        [
            {"actual_chapter_num": chapter}
            for chapter in (9, 17, 2, 16, 3, 14)
        ],
        str(Path.cwd()),
        ProgressStub(),
        lambda: False,
        multipass_failed_mode=True,
    )

    output = capsys.readouterr().out
    summary = (
        "⏹️ Graceful stop skipped queued refinement for 6 chapters: "
        "2, 3, 9, 14, 16, 17"
    )
    assert output.count(summary) == 1
    assert "for Chapter 9" not in output
    assert "for Chapter 17" not in output


def test_gui_publishes_stop_mode_before_shared_stop_callback_latch():
    gui_source = (
        Path(__file__).resolve().parents[1] / "src" / "translator_gui.py"
    ).read_text(encoding="utf-8")
    stop_method = gui_source.split("    def stop_translation(self):", 1)[1].split(
        "    def preserve_file_path", 1
    )[0]

    mode_publish = stop_method.index(
        "os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'"
    )
    callback_latch = stop_method.index("self.stop_requested = True")
    assert mode_publish < callback_latch
