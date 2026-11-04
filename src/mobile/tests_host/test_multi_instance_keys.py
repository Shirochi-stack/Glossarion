"""Host tests: no API key of one instance reaches another instance's requests on mobile (design §6b).

Owner: "instance 2 must never use the fallback key of instance 1, we must not have instance contamination"; every
key pool is split by instance and the main API key belongs to instance 1.

* Every pool's keys store their instance like the desktop (``APIKeyEntry.to_dict``): the field only when it is not 1.
* JobService: a job of a multi-instance run gets ``instance_plan.config_for_instance`` (every pool narrowed, the main
  API key replaced for an instance >= 2) and runs under ``instance_worker.restrict_keys`` (``GLOSSARION_KEY_INSTANCE``,
  the client's pools rebuilt from that snapshot), instance 1 too; an instance >= 2 never inherits instance 1's key
  variables; a plain job (no multi-instance run) is left exactly as before.
* End to end (``multi_instance_keys_scenario``, the real pipeline against a recording fake server that blocks chosen
  keys, two books at once): instance 2's blocked Translation key is rescued by its own fallback key only, never by
  instance 1's nor the main API key; its glossary / metadata requests use its own key of that pool, or its own
  Translation key when it has none there.
* Resume withdraws the stopped job's "Stopped: … Tap to resume" notification (one instance or several).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_multi_instance_keys.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
os.environ.setdefault("GLOSSARION_HEADLESS_KEY_MANAGER", "1")

import instance_plan  # noqa: E402
from glossarion_mobile.services.jobs import RESUMED_EVENT, JobSpec, JobState  # noqa: E402
from test_multi_instance_e2e import DEPENDENCIES, SELFTEST_EPUB, _missing, run_scenario  # noqa: E402
from test_multi_instance_jobs import (  # noqa: E402,F401  (the fixtures run here too)
    TIMEOUT,
    FakeStore,
    Probe,
    _book,
    _mobile_env,
    _service,
    runtime,
    wait_for,
)

KEYS_SCENARIO = Path(__file__).resolve().parent / "multi_instance_keys_scenario.py"
MAIN_KEY = "sk-main-api-key"


def _key(api_key, instance=1):
    entry = {"api_key": api_key, "model": "gpt-4o-mini", "enabled": True}
    if instance != 1:
        entry["instance"] = instance
    return entry


#: two instances' keys in the Translation, Fallback and Glossary pools, and the main API key (instance 1's)
CONFIG = {
    "model": "gpt-4o-mini", "api_key": MAIN_KEY, "use_main_key_fallback": True,
    "use_multi_api_keys": True, "multi_api_keys": [_key("tr-one"), _key("tr-two", 2)],
    "use_fallback_keys": True, "fallback_keys": [_key("fb-one"), _key("fb-two", 2)],
    "use_glossary_keys": True, "glossary_keys": [_key("gl-one")],
}


# ==========================================================================
# Every pool's keys carry an instance, stored like the desktop
# ==========================================================================


@pytest.mark.skipif(bool(_missing("multi_api_key_manager")), reason="multi_api_key_manager not importable")
def test_every_pools_keys_store_their_instance_like_the_desktop(tmp_path):
    """Translation, Fallback, Glossary and every dedicated pool: the KeyEditor / bulk "🔢 Set instance…" save
    (``set_key_fields`` -> ``key_pool_service.validate_entry``) stores ``"instance": N`` as an integer only when
    it is not 1 (``APIKeyEntry.to_dict``, the desktop key lists), and a copy to another pool keeps it."""
    import key_pool_service
    from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController
    from test_multi_instance_ui import _store

    pools = {pid: spec["config_key"] for pid, spec in key_pool_service.POOL_SPECS.items()}
    assert set(pools.values()) == {spec[0] for spec in instance_plan.POOLS.values()}  # one list of pools
    store = _store(tmp_path, {key: [_key(f"{pid}-a"), _key(f"{pid}-b", 3)] for pid, key in pools.items()})
    keys = KeysController(store, KeyBackend())
    try:
        assert {spec.id for spec in keys.specs} == set(pools)
        for pid in pools:
            assert keys.set_key_fields(pid, [0], {"instance": "2"}) == (1, []), pid
            assert keys.set_key_fields(pid, [1], {"instance": 1}) == (1, []), pid
            stored = store.get(pools[pid])
            assert stored[0]["instance"] == 2 and type(stored[0]["instance"]) is int, (pid, stored[0])
            assert "instance" not in stored[1], (pid, stored[1])
            assert [instance_plan.key_instance(row) for row in stored] == [2, 1]
        # a key copied to another pool keeps its instance
        assert keys.copy_to("main", [0], "fallback") == 1
        assert store.get("fallback_keys")[-1]["instance"] == 2
    finally:
        store._saver.close()


# ==========================================================================
# JobService: each job's keys
# ==========================================================================


class KeyProbe(Probe):
    """``Probe`` that also records the job's key state: its key scope, the key fields of its config snapshot, the
    key variables of its environment and the API client's in-memory pools (its instance's own client)."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: dict = {}

    def run(self, ctx):
        import unified_api_client

        client = unified_api_client.UnifiedClient
        self.keys[ctx.job_id] = {
            "key_instance": os.environ.get(instance_plan.ENV_KEY_INSTANCE), "scope": instance_plan.key_scope(),
            "api_key": ctx.config.get("api_key"), "main_key_fallback": ctx.config.get("use_main_key_fallback"),
            "translation": [r.get("api_key") for r in ctx.config.get("multi_api_keys") or []],
            "fallback": [r.get("api_key") for r in ctx.config.get("fallback_keys") or []],
            "glossary_on": ctx.config.get("use_glossary_keys"),
            "env": {name: os.environ.get(name) for name in ("API_KEY", "FALLBACK_KEYS", "USE_MAIN_KEY_FALLBACK")},
            "pool": [r.get("api_key") for r in getattr(client, "_in_memory_multi_keys", None) or []],
            "glossary_pool": [r.get("api_key") for r in getattr(client, "_in_memory_glossary_keys", None) or []],
        }
        return super().run(ctx)


def _run_probe(service, probe, spec, *, instance=None):
    job_id = (service.submit(spec) if instance is None else service.submit_job(spec, instance=instance).job_id)
    assert probe.entered.setdefault(job_id, threading.Event()).wait(TIMEOUT)
    return job_id


def test_plain_job_is_unchanged_and_instance_1_of_a_run_is_scoped(tmp_path):
    """A plain job keeps every key and no key scope (single-instance users see no change); instance 1's job of a
    multi-instance run (``instance=1``) runs on ``config_for_instance(config, 1)`` under ``restrict_keys(1)``:
    instance 1's keys in every pool, the main API key (its own) kept; afterwards the scope and pools are back."""
    import job_runner
    import unified_api_client

    client = unified_api_client.UnifiedClient
    pools_before = job_runner._pool_state_items(client)
    probe = KeyProbe()
    service = _service(tmp_path, probe, FakeStore(CONFIG))
    try:
        plain = _run_probe(service, probe, JobSpec("translate", "Plain", (_book(tmp_path, "a.epub"),)))
        seen = probe.keys[plain]
        assert seen["key_instance"] is None and seen["scope"] is None
        assert seen["translation"] == ["tr-one", "tr-two"] and seen["fallback"] == ["fb-one", "fb-two"]
        assert seen["api_key"] == MAIN_KEY and seen["main_key_fallback"] is True and seen["glossary_on"] is True
        probe.gate(plain).set()
        assert service.wait_idle(TIMEOUT)

        one = _run_probe(service, probe, JobSpec("translate", "One", (_book(tmp_path, "b.epub"),)), instance=1)
        seen = probe.keys[one]
        assert seen["key_instance"] == "1" and seen["scope"] == 1
        assert seen["translation"] == ["tr-one"] and seen["fallback"] == ["fb-one"]
        assert seen["api_key"] == MAIN_KEY and seen["main_key_fallback"] is True  # the main key is instance 1's
        assert seen["pool"] == ["tr-one"] and seen["glossary_pool"] == ["gl-one"]  # rebuilt from its snapshot
        probe.gate(one).set()
        assert service.wait_idle(TIMEOUT)
        assert service.snapshot(plain).state is JobState.DONE and service.snapshot(one).state is JobState.DONE
    finally:
        service.close()
    assert os.environ.get(instance_plan.ENV_KEY_INSTANCE) is None and instance_plan.key_scope() is None
    after = job_runner._pool_state_items(client)
    assert after.keys() == pools_before.keys() and all(after[n] is v for n, v in pools_before.items())


def test_instance_2_job_gets_only_its_own_keys_and_resumes_on_them(tmp_path, runtime):
    """Instance 2: its snapshot holds only its keys (its own Translation key in place of the main API key, "Use Main
    GUI Key as Fallback #1" off, the Glossary pool off: no key of its own there); its key scope is 2 and its client's
    pools hold its keys; instance 1's key variables left in the process never reach its environment. Stopped and
    resumed, it carries on in a new job on instance 2 with the same keys (``RESUMED_EVENT`` names it)."""
    leftovers = {"API_KEY": MAIN_KEY, "FALLBACK_KEYS": json.dumps(CONFIG["fallback_keys"]),
                 "USE_MAIN_KEY_FALLBACK": "1", instance_plan.ENV_KEY_INSTANCE: "1"}
    saved = {name: os.environ.get(name) for name in leftovers}
    os.environ.update(leftovers)  # instance 1's process environment, as a run left it
    probe = KeyProbe()
    service = _service(tmp_path, probe, FakeStore(CONFIG))
    events: list = []
    service.on_event(lambda job_id, kind, data: events.append((job_id, kind, dict(data))))
    try:
        two = _run_probe(service, probe, JobSpec("translate", "Two", (_book(tmp_path, "two.epub"),)), instance=2)
        resumed = None
        for job_id in (two, None):
            if job_id is None:  # Stop, then Resume: the same instance, the same keys
                assert service.request_stop(two, force=True) == "force"
                assert wait_for(lambda: service.snapshot(two).is_terminal)
                assert service.snapshot(two).state is JobState.CANCELLED
                resumed = service.resume_job(two)
                assert resumed is not None and resumed.job_id != two and not resumed.reused
                job_id = resumed.job_id
                assert probe.entered.setdefault(job_id, threading.Event()).wait(TIMEOUT)
                assert service.snapshot(job_id).instance == 2
            seen = probe.keys[job_id]
            assert seen["key_instance"] == "2" and seen["scope"] == 2
            assert seen["translation"] == ["tr-two"] and seen["fallback"] == ["fb-two"]
            assert seen["api_key"] == "tr-two" and seen["main_key_fallback"] is False
            assert seen["glossary_on"] is False  # no Glossary key of instance 2: its Translation keys do that work
            assert seen["env"] == {"API_KEY": None, "FALLBACK_KEYS": None, "USE_MAIN_KEY_FALLBACK": None}
            assert seen["pool"] == ["tr-two"] and seen["glossary_pool"] == []
            assert MAIN_KEY not in json.dumps(seen) and "fb-one" not in json.dumps(seen)
        probe.gate(resumed.job_id).set()
        assert service.wait_idle(TIMEOUT)
        assert service.snapshot(resumed.job_id).state is JobState.DONE
        assert (two, RESUMED_EVENT, {"job": resumed.job_id}) in events
        assert {name: os.environ.get(name) for name in leftovers} == leftovers  # instance 1's own left alone
    finally:
        service.close()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ==========================================================================
# End to end: failing keys against a recording fake server
# ==========================================================================


@pytest.mark.skipif(not SELFTEST_EPUB.is_file(), reason="run tools/prepare_assets.py first")
@pytest.mark.skipif(bool(_missing(*DEPENDENCIES)), reason=f"backend dependencies missing: {_missing(*DEPENDENCIES)}")
@pytest.mark.parametrize("variant", ["glossary", "metadata"])
def test_failing_keys_never_cross_instances(tmp_path, variant):
    result, storage = run_scenario(tmp_path, KEYS_SCENARIO, "--variant", variant)
    assert list(result["checks"]) == ["setup", "concurrent_run", "instance_2_keys", "instance_1_keys",
                                      "dedicated_pools", "hygiene"]
    assert set(result["checks"].values()) == {"pass"}
    detail = result["detail"]
    keys_a, keys_b = set(detail["keys_A"]), set(detail["keys_B"])
    one = {"sk-mi-main-gui-key", "sk-mi-translation-one", "sk-mi-fallback-one", "sk-mi-glossary-one",
           "sk-mi-metadata-one"}
    assert keys_a <= one and not keys_b & one  # never a key of the other instance, never the main API key in 2
    assert {"sk-mi-translation-two", "sk-mi-fallback-two"} <= keys_b
    assert {"sk-mi-main-gui-key", "sk-mi-fallback-one"} <= keys_a  # instance 1's own main-key retry and fallback
    pools = detail["pools"]
    if variant == "glossary":
        assert pools["glossary_B"] == ["sk-mi-glossary-two"]
        assert pools["other_B"] == ["sk-mi-fallback-two", "sk-mi-translation-two"]
    else:
        assert pools["glossary_B"] == ["sk-mi-translation-two"] and pools["other_B"] == ["sk-mi-metadata-two"]
    sandboxes = storage["temp"] / "glossarion-e2e"
    assert not sandboxes.exists() or not any(sandboxes.iterdir())


# ==========================================================================
# Resume withdraws the "Stopped: … Tap to resume" notification
# ==========================================================================


def _has_flet() -> bool:
    return not _missing("flet", "msgpack")


@pytest.mark.skipif(not _has_flet(), reason="flet / msgpack not installed")
def test_resuming_a_stopped_job_withdraws_its_stopped_notification(tmp_path):
    """Owner report (mobile UI agent's device check): after a stopped job was resumed from the Jobs page, its
    "Stopped: <book> (n/m) Tap to resume" notification stayed. Any Resume now withdraws it (and the job's
    "Paused" one); a stopped job nobody resumed keeps its notification. One instance here; the event is the same
    for every instance (``test_instance_2_job_gets_only_its_own_keys_and_resumes_on_them``)."""
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.services.notifications import action_notification_id, done_notification_id
    from glossarion_mobile.state.store import LoopGuard
    from test_jobs import FakeNative, Gate, _HiddenPage, _jobs_feature, _until, epub, quick_worker

    def posted(native, notification_id):
        return [c[1][1] for c in native.calls if c[0] == "show_notification" and c[1][0] == notification_id]

    async def scenario():
        page = _HiddenPage()
        dispatcher = UiDispatcher(page, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        native = FakeNative()
        feature, _app, backend, _notes, _routes = _jobs_feature(tmp_path, dispatcher=dispatcher, native=native)
        service = feature.service
        try:
            stopped = []
            for name in ("Kept.epub", "Book.epub"):
                gate = Gate()
                backend.behavior = gate
                job_id = service.submit(JobSpec("translate", name, (epub(tmp_path, name),)))
                assert await asyncio.to_thread(gate.entered.wait, TIMEOUT)
                service.request_stop(job_id, force=True)
                assert await _until(lambda: service.snapshot(job_id).is_terminal, TIMEOUT)
                assert service.snapshot(job_id).state is JobState.CANCELLED
                assert await _until(lambda: posted(native, done_notification_id(job_id)), TIMEOUT)
                assert posted(native, done_notification_id(job_id))[-1].startswith(f"Stopped: {name}")
                stopped.append(job_id)
            kept, job_id = stopped
            backend.behavior = quick_worker
            withdrawn = ("cancel_notification", (done_notification_id(job_id),), {})
            assert withdrawn not in native.calls
            new_id = service.resume(job_id)  # the Jobs page's Resume
            assert new_id and new_id != job_id
            assert await _until(lambda: withdrawn in native.calls, TIMEOUT)
            assert ("cancel_notification", (action_notification_id(job_id),), {}) in native.calls
            assert await asyncio.to_thread(service.wait_idle, TIMEOUT)
            assert service.snapshot(new_id).state is JobState.DONE
            # the job nobody resumed keeps its "Stopped" notification
            assert ("cancel_notification", (done_notification_id(kept),), {}) not in native.calls
        finally:
            feature.close()
            await dispatcher.stop()

    asyncio.run(scenario())
