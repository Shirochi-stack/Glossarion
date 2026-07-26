import threading
import time

from TransateKRtoEN import (
    _partial_b2_batch_worker_count,
    _partial_b2_entries_per_request,
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
