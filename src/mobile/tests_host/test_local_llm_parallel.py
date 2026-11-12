"""Opt-in parallel requests retain independent contexts, cancellation and serial defaults."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_llm_fakes import FakePrefs
from glossarion_mobile.services.local_llm import engine as eng, service as svc
from glossarion_mobile.services.local_llm.parallel import PARALLEL_KEY
from glossarion_mobile.services.local_llm.server import LocalLLMError
from test_local_llm_server import make_service


def ready(tmp_path, enabled=True, **kwargs):
    service, _ = make_service(tmp_path, stems=("a", "b"), **kwargs)
    made = []
    def factory():
        engine = eng.FakeEngine()
        made.append(engine)
        return engine
    service._engine_factory = factory
    if enabled:
        service.save_parallel_requests(True)
    return service, made


def chat(service, text="hello", model="a", stop=lambda: False):
    return service.chat(model, {"messages": [{"role": "user", "content": text}]}, stop)


def wait_for(predicate):
    end = time.monotonic() + 3
    while not predicate() and time.monotonic() < end:
        time.sleep(.01)
    assert predicate()


def test_default_serial_and_saved_toggle(tmp_path):
    service, made = ready(tmp_path, enabled=False)
    assert not service.parallel_requests()
    first = chat(service, "first")
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service, "second")
        try:
            wait_for(lambda: service.gate.waiting() == 1)
            assert not second.done() and len(made) == 1
        finally:
            first.close()
        second.result(3).close()
    service.save_parallel_requests(True)
    fresh = svc.LocalEngineService(prefs=service.prefs, watch=False)
    assert fresh.parallel_requests()
    assert service.health()["request_slots"] == 2


def test_two_independent_contexts_and_fifo_overflow(tmp_path):
    service, made = ready(tmp_path)
    first = chat(service, "first answer")
    with ThreadPoolExecutor(2) as pool:
        second = pool.submit(chat, service, "second answer").result(3)
        third = pool.submit(chat, service, "third answer")
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            assert not third.done() and len(made) == 2
            assert made[0] is not made[1]
            assert service.health()["queue"] == 1
            assert not service.delete("a")
            assert "".join(first.deltas) == "Echo: first answer"
            assert "".join(second.deltas) == "Echo: second answer"
            assert first.usage["completion_tokens"] == second.usage["completion_tokens"] == 3
        finally:
            first.close()
            second.close()
        run = third.result(3)
        try:
            assert "".join(run.deltas) == "Echo: third answer"
        finally:
            run.close()
    assert not service._parallel.active and not made[1].loaded
    assert made[0].loaded


def test_disable_drains_both_answers_before_serial_admission(tmp_path):
    service, made = ready(tmp_path)
    first = chat(service)
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service).result(3)
        service.save_parallel_requests(False)
        third = pool.submit(chat, service)
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            first.close()
            assert not third.done()  # secondary still owns an admitted answer
            assert service.snapshot().state == "generating"
            assert not service.load_blocking("b")[0]
        finally:
            second.close()
        third.result(3).close()
    assert len(made) == 2 and not made[1].loaded
    assert service.prefs.get(PARALLEL_KEY) is False


@pytest.mark.parametrize("cancel", ["disconnect", "stop"])
def test_waiting_request_cancelled_and_stop_cannot_be_cleared(tmp_path, cancel):
    service, made = ready(tmp_path)
    first = chat(service)
    gone = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service).result(3)
        third = pool.submit(chat, service, stop=gone.is_set)
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            if cancel == "disconnect":
                gone.set()
            else:
                service.stop_generation()
            with pytest.raises(LocalLLMError) as error:
                third.result(3)
            assert error.value.status == 499
            assert not service._parallel.queue
        finally:
            first.close()
            second.close()
    assert not service._parallel.active and not made[1].loaded


def test_model_switch_waits_for_secondary_and_context_failure_releases_slot(tmp_path):
    service, made = ready(tmp_path)
    first = chat(service)
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service).result(3)
        first.close()
        other = pool.submit(chat, service, model="b")
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            assert not other.done()
        finally:
            second.close()
        other.result(3).close()
    first = chat(service)
    with ThreadPoolExecutor(1) as pool:
        failure = pool.submit(chat, service, "word " * 30000)
        try:
            with pytest.raises(LocalLLMError) as error:
                failure.result(3)
            assert error.value.code == "context_length_exceeded"
            assert not made[1].loaded
            pool.submit(chat, service, "recovered").result(3).close()
        finally:
            first.close()


def test_gpu_markers_are_independent_for_each_context(tmp_path):
    service, made = ready(tmp_path, gpu_backend="OpenCL", prefs=FakePrefs({svc.GPU_KEY: True}))
    first = chat(service)
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service).result(3)
        try:
            assert all(engine.on_gpu for engine in made)
            assert service.prefs.get(svc.GPU_ATTEMPT_KEY)
            assert service.prefs.get(svc.GPU_ATTEMPT_KEY + "_parallel")
            first.close()
            assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
            assert service.prefs.get(svc.GPU_ATTEMPT_KEY + "_parallel")
        finally:
            second.close()
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY + "_parallel") is None
    service.prefs.set(svc.GPU_ATTEMPT_KEY + "_parallel", {"stem": "a"})
    assert service._recover_gpu_attempt() == "a"
    assert service.gpu_fallback()["stem"] == "a"


def test_panel_toggle_starts_off_and_saves(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    service, _ = ready(tmp_path, enabled=False)
    panel = LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    try:
        assert not panel.parallel_switch.value
        panel.parallel_switch.value = True
        panel.parallel_switch.on_change(None)
        assert service.parallel_requests()
    finally:
        panel.dispose()


def test_device_parallel_probe_checks_overlap_and_cleanup(tmp_path):
    from glossarion_mobile.services.local_llm.selfcheck import parallel_generation_check
    service, made = ready(tmp_path, enabled=False)
    result = parallel_generation_check(service, "a")
    assert result["overlapping_requests"] == 2 and result["extra_context_unloaded"]
    assert len(result["answers"]) == 2 and len(made) == 2
    # Admitted fake requests must not be reported as measured native inference.
    assert result["native_decode_overlap_seconds"] == 0
    assert result["decode_calls_per_context"] == [0, 0]
    assert result["generation_threads"] == 0
    assert not service.parallel_requests()


def test_force_stop_reaches_both_generations_and_queue_limit(tmp_path):
    service, made = ready(tmp_path)
    service.gate.max_waiters = 1
    first = chat(service, "word " * 40)
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service, "other " * 40).result(3)
        waiting = pool.submit(chat, service)
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            with pytest.raises(LocalLLMError) as full:
                chat(service)
            assert full.value.status == 503
            service.stop_generation()
            with pytest.raises(LocalLLMError) as cancelled:
                waiting.result(3)
            assert cancelled.value.status == 499
            list(first.deltas)
            list(second.deltas)
            assert all(engine.generated <= eng.STOP_CHECK_EVERY for engine in made)
        finally:
            first.close()
            second.close()


def test_graceful_stop_finishes_both_active_answers(tmp_path, monkeypatch):
    service, made = ready(tmp_path)
    first = chat(service, "first answer")
    with ThreadPoolExecutor(1) as pool:
        second = pool.submit(chat, service, "second answer").result(3)
        waiting = pool.submit(chat, service)
        try:
            wait_for(lambda: len(service._parallel.queue) == 1)
            monkeypatch.setenv("GRACEFUL_STOP", "1")
            with pytest.raises(LocalLLMError):
                waiting.result(3)
            assert "".join(first.deltas) == "Echo: first answer"
            assert "".join(second.deltas) == "Echo: second answer"
        finally:
            first.close()
            second.close()
