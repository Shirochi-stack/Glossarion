"""Host tests: what one mobile instance does never reaches another (review findings and owner decisions,
2026-11-04).

* Owner: "we must not let instance 2 keys get used by instance 1": a plain job (chat, Book page, Chapters) that
  runs while a job of instance 2 runs uses instance 1's keys only, from its start or from the moment the
  instance 2 job starts; the request guard knows instance 2's keys. Without an instance 2 job nothing changes.
* The system time limit stops the one job left even when it is instance 2's.
* A Stop of an instance 2 job picked but not set up yet only latches it: instance 1's job is not touched.
* Instance 1's stopped job does not leave the on-device endpoint refusing instance 2's requests.
* The kept interpreter pool's one stop watcher asks each job's check as that job's instance.
* "Does a job write this folder?" guards see every instance's running job.
* A graceful stop of one instance does not silence the HTTP log of another instance's running job; the
  per-instance copies of the API client add no second 🌐 log filter.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_multi_instance_guards.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import instance_plan  # noqa: E402
from glossarion_mobile.services.background import BackgroundExecution  # noqa: E402
from glossarion_mobile.services.jobs import JobSpec, JobState, JobsView, pending_jobs  # noqa: E402
from glossarion_mobile.services.notifications import JobNotifications  # noqa: E402
from test_multi_instance_jobs import (  # noqa: E402,F401  (the fixtures run here too)
    KEYS,
    FakeStore,
    Native,
    Probe,
    _book,
    _mobile_env,
    _service,
    runtime,
    wait_for,
)
from test_multi_instance_keys import CONFIG, MAIN_KEY, KeyProbe  # noqa: E402


def _start(service, probe, spec, *, instance=None):
    job_id = service.submit(spec) if instance is None else service.submit_job(spec, instance=instance).job_id
    assert probe.entered.setdefault(job_id, threading.Event()).wait(60)
    return job_id


def _base_scope():
    """What instance 1's side of the app sees right now (this test thread belongs to no instance)."""
    import unified_api_client

    client = unified_api_client.UnifiedClient
    return {
        "scope": instance_plan.key_scope(),
        "pool": [r.get("api_key") for r in getattr(client, "_in_memory_multi_keys", None) or []],
        "fallback": [r.get("api_key") for r in getattr(client, "_in_memory_fallback_keys", None) or []],
        "tr-two foreign": instance_plan.foreign_key_owners("tr-two"),
        "main foreign": instance_plan.foreign_key_owners(MAIN_KEY),
    }


# ---------------------------------------------------------------------------
# Owner: a plain job while an instance 2 job runs
# ---------------------------------------------------------------------------

def test_a_plain_job_while_an_instance_2_job_runs_uses_only_instance_1_keys(tmp_path, runtime):
    """Instance 2 runs first, then a plain job starts: from its start it has instance 1's keys only (every pool,
    the main API key is instance 1's), and the request guard refuses instance 2's key. The other order: a plain
    job runs, then an instance 2 job starts: from then on instance 1's side is scoped too. Each ends undone."""
    probe = KeyProbe()
    service = _service(tmp_path, probe, FakeStore(CONFIG))
    try:
        two = _start(service, probe, JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2)
        plain = _start(service, probe, JobSpec("translate", "Plain", (_book(tmp_path, "a.epub"),)))
        seen = probe.keys[plain]
        assert seen["key_instance"] == "1" and seen["scope"] == 1
        assert seen["pool"] == ["tr-one"] and seen["glossary_pool"] == ["gl-one"]
        assert seen["api_key"] == MAIN_KEY  # the plain job's own config: the main API key is instance 1's
        now = _base_scope()
        assert now["scope"] == 1 and now["tr-two foreign"] == (2,) and now["main foreign"] == ()
        probe.gate(plain).set()
        assert wait_for(lambda: service.snapshot(plain).is_terminal)
        assert _base_scope()["scope"] is None  # undone with the plain job
        probe.gate(two).set()
        assert service.wait_idle(60)

        # a plain job first (no instance 2 job: every key), then instance 2 starts
        plain = _start(service, probe, JobSpec("translate", "Plain 2", (_book(tmp_path, "b.epub"),)))
        assert probe.keys[plain]["scope"] is None and probe.keys[plain]["translation"] == ["tr-one", "tr-two"]
        two = _start(service, probe, JobSpec("translate", "Two 2", (_book(tmp_path, "two2.epub"),)), instance=2)
        now = _base_scope()
        assert now["scope"] == 1 and now["pool"] == ["tr-one"] and now["tr-two foreign"] == (2,)
        assert probe.keys[two]["scope"] == 2 and probe.keys[two]["pool"] == ["tr-two"]
        probe.gate(plain).set()
        assert wait_for(lambda: service.snapshot(plain).is_terminal)
        probe.gate(two).set()
        assert service.wait_idle(60)
        assert service.snapshot(plain).state is JobState.DONE and service.snapshot(two).state is JobState.DONE
    finally:
        for gate in list(probe.release.values()):
            gate.set()
        service.wait_idle(60)
        service.close()
    after = _base_scope()
    assert after["scope"] is None and os.environ.get(instance_plan.ENV_KEY_INSTANCE) is None


# ---------------------------------------------------------------------------
# Review findings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform, event", [("android", "timeout"), ("ios", "continued_expired")])
def test_the_system_time_limit_stops_an_instance_2_job_left_alone(tmp_path, runtime, platform, event):
    """Instance 1's book is done first; instance 2's job is the only one running when the system time limit hits:
    it is stopped (by its id), and its "Paused" notification is true."""
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    try:
        one = _start(service, probe, JobSpec("translate", "One", (_book(tmp_path, "one.epub"),)), instance=1)
        two = _start(service, probe, JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2)
        probe.gate(one).set()
        assert wait_for(lambda: service.snapshot(one).is_terminal)
        native = Native()
        background = BackgroundExecution(native, platform=platform, jobs=service,
                                         notifications=JobNotifications(native, platform=platform))
        background.service_running = True
        handler = background.on_foreground_event if platform == "android" else background.on_background_task_event
        asyncio.run(handler({"type": event}))
        assert wait_for(lambda: service.snapshot(two).is_terminal)
        assert service.snapshot(two).state is JobState.CANCELLED
    finally:
        probe.gate(two).set()
        service.wait_idle(60)
        service.close()


def test_a_stop_before_the_instance_is_set_up_only_latches_it(tmp_path, runtime, monkeypatch):
    """The first instance 2 job is picked while its runtime is still being made: its Stop must not run the stop
    protocol on instance 1's modules (that cancelled instance 1's running job)."""
    import unified_api_client

    original = runtime.ensure
    monkeypatch.setattr(runtime, "ensure", lambda number: (time.sleep(1.0), original(number))[1])
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    try:
        one = _start(service, probe, JobSpec("translate", "One", (_book(tmp_path, "one.epub"),)), instance=1)
        two = service.submit_job(JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2).job_id
        assert wait_for(lambda: service.instance_busy(2))
        assert runtime.get(2) is None  # picked, not set up
        assert service.request_stop(two, force=True) == "force"
        assert service.wait_stops_idle(10)
        assert getattr(unified_api_client.UnifiedClient, "_global_cancelled", False) is False
        assert os.environ.get("TRANSLATION_CANCELLED") != "1"
        assert wait_for(lambda: service.snapshot(two).is_terminal)
        assert service.snapshot(two).state is JobState.CANCELLED
        probe.gate(one).set()
        assert service.wait_idle(60)
        assert service.snapshot(one).state is JobState.DONE and probe.seen[one]["saw_cancel"] is False
    finally:
        service.close()


def test_instance_1_stopped_leaves_the_on_device_endpoint_open_for_instance_2(tmp_path, runtime):
    """The on-device endpoint's request threads belong to no instance and read instance 1's stop flags: once
    instance 1's stopped job has ended, those flags no longer refuse instance 2's ``local/`` requests."""
    from glossarion_mobile.services.local_llm.service import process_stop_state

    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    try:
        one = _start(service, probe, JobSpec("translate", "One", (_book(tmp_path, "one.epub"),)), instance=1)
        two = _start(service, probe, JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2)
        service.request_stop(one, force=True)
        assert wait_for(lambda: service.snapshot(one).is_terminal)
        assert service.wait_stops_idle(10)
        seen = {}
        thread = threading.Thread(target=lambda: seen.update(state=process_stop_state()))
        thread.start()
        thread.join()
        assert seen["state"] is None and service.snapshot(two).state is JobState.RUNNING
    finally:
        probe.gate(two).set()
        service.wait_idle(60)
        service.close()


def test_the_interpreter_pool_watcher_asks_each_check_as_its_instance(runtime):
    """The kept pool's one ``glossarion-interpreter-stop`` thread starts on whichever caller came first: an
    instance 2 check it runs reads instance 2's environment, never instance 1's (stopped) one."""
    import job_runner  # noqa: F401  (the instance copies need the backend importable)
    from glossarion_mobile.services.interpreters import InterpreterPool

    instance = runtime.ensure(2)
    instance.prepare(env=dict(runtime.base_environ()))
    pool = InterpreterPool(1)
    seen = []
    try:
        first = pool._watch(lambda: False)  # instance 1 starts the watcher thread

        def check():
            seen.append((threading.current_thread().name, os.environ.get("TRANSLATION_CANCELLED")))
            return False

        def on_instance_two():
            os.environ["TRANSLATION_CANCELLED"] = "0"
            entry = pool._watch(check)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not any(name != "instance-2" for name, _v in seen):
                time.sleep(0.02)
            pool._unwatch(entry)

        runtime.base_environ()["TRANSLATION_CANCELLED"] = "1"  # instance 1 was stopped
        thread = instance.thread(on_instance_two, name="instance-2")
        thread.start()
        thread.join()
        pool._unwatch(first)
    finally:
        runtime.base_environ().pop("TRANSLATION_CANCELLED", None)
        pool.close()
    watcher = [value for name, value in seen if name == "glossarion-interpreter-stop"]
    assert watcher and set(watcher) == {"0"}, seen


def test_folder_guards_see_every_running_job():
    """``pending_jobs``: every instance's running job, then the queue (``active`` is only one of them)."""
    from types import SimpleNamespace

    one, two, queued = (SimpleNamespace(id=x) for x in ("one", "two", "queued"))
    view = JobsView(active=one, running=(one, two), queue=(queued,))
    assert [s.id for s in pending_jobs(view)] == ["one", "two", "queued"]
    assert [s.id for s in pending_jobs(SimpleNamespace(active=one, queue=[queued]))] == ["one", "queued"]
    assert pending_jobs(JobsView()) == []
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter

    adapter = JobsAdapter.__new__(JobsAdapter)
    adapter._signal = lambda name: (lambda: view) if name == "view" else None
    assert [s.id for s in adapter.pending()] == ["one", "two", "queued"]
    for path in ("app/glossarion_mobile/services/cloud_sync.py", "app/glossarion_mobile/services/share_links.py"):
        assert "pending_jobs(" in (MOBILE_DIR / path).read_text(encoding="utf-8"), path


def test_a_graceful_stop_of_one_instance_keeps_the_other_instances_http_log(tmp_path, runtime):
    """The HTTP client loggers are the process's: instance 2's graceful stop leaves their level as instance 1's
    running job needs it (a lone job's graceful stop still silences them, as on desktop)."""
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    httpx = logging.getLogger("httpx")
    level = httpx.level
    httpx.setLevel(logging.INFO)
    try:
        one = _start(service, probe, JobSpec("translate", "One", (_book(tmp_path, "one.epub"),)), instance=1)
        two = _start(service, probe, JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2)
        service._graceful_setting = lambda job: (True, True)  # the graceful stop setting on
        service.request_stop(two, escalate=False)
        assert service.wait_stops_idle(10)
        assert httpx.level == logging.INFO
        probe.gate(two).set()
        assert wait_for(lambda: service.snapshot(two).is_terminal)
        service.request_stop(one, escalate=False)  # the last job running: silenced as before
        assert service.wait_stops_idle(10)
        assert httpx.level == logging.CRITICAL
    finally:
        probe.gate(one).set()
        service.wait_idle(60)
        service.close()
        httpx.setLevel(level)


def test_instance_copies_add_no_second_http_emoji_filter(runtime):
    import unified_api_client  # noqa: F401  (instance 1's copy)

    before = [type(f).__name__ for f in logging.getLogger("httpx").filters]
    runtime.ensure(2).module("unified_api_client")
    after = [type(f).__name__ for f in logging.getLogger("httpx").filters]
    assert before.count("_HttpEmojiFilter") == after.count("_HttpEmojiFilter") == 1, (before, after)
