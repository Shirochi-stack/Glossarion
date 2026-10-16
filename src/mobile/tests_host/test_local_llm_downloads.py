"""GGUF downloads and the model registry (``services/local_llm/downloads.py``) against a local fake Hugging
Face that honours ``Range``: fresh download, pause + resume (206, re-hash), a server that ignores Range
(200 restarts), a dropped connection (retry resumes), checksum mismatch, free-space refusal, cancel,
progress throttling on the loop, atomic registry and stem collisions. No network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from types import SimpleNamespace

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import hf

REPO = "tester/Tiny-GGUF"
NAME = "Tiny-Q4_K_M.gguf"
PAYLOAD = os.urandom(3 * 1024 * 1024 + 12345)
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture()
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    server.add(REPO, {NAME: PAYLOAD, "Tiny-Q8_0.gguf": PAYLOAD[:1000]})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    monkeypatch.setattr(dl, "BACKOFF", (0, 0, 0, 0, 0))
    yield server
    server.close()


def _spec(registry, size=len(PAYLOAD), sha=SHA, repo=REPO, file=NAME):
    return dl.DownloadSpec(repo, file, fakes.FakeHF.COMMIT, size, sha, registry.stem_for(repo, file))


def _manager(tmp_path, **kwargs):
    registry = dl.Registry(str(tmp_path / "models"))
    return registry, dl.DownloadManager(registry=registry, throttle=kwargs.pop("throttle", 0), **kwargs)


def test_fresh_download_verifies_registers_and_reports_on_the_loop(fake_hf, tmp_path):
    registry, manager = _manager(tmp_path)
    seen = []

    async def scenario():
        loop_thread = threading.get_ident()
        manager.subscribe(lambda s: seen.append((s.phase, s.bytes_done, threading.get_ident() == loop_thread)))
        return await manager.download(_spec(registry))

    state = asyncio.run(scenario())
    assert state.done and not state.error and state.status == "Downloaded ✓" and state.fraction == 1.0
    final = registry.file_path("Tiny-Q4_K_M")
    with open(final, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == SHA
    assert not os.path.exists(final + ".part") and not os.path.exists(final + ".part.json")
    entry = registry.load()["Tiny-Q4_K_M"]
    assert entry["sha256"] == SHA and entry["size"] == len(PAYLOAD) and entry["revision"] == fakes.FakeHF.COMMIT
    assert entry["repo"] == REPO and entry["file"] == NAME and entry["downloaded_at"] > 0
    assert all(on_loop for _p, _b, on_loop in seen)  # listeners always run on the event loop
    assert any(p == "downloading" and 0 < b < len(PAYLOAD) for p, b, _l in seen)
    assert registry.resolve("local/tiny-q4_k_m") == "Tiny-Q4_K_M" and registry.find(REPO, NAME) == "Tiny-Q4_K_M"
    with open(registry.path, encoding="utf-8") as handle:
        assert json.load(handle)["Tiny-Q4_K_M"]["sha256"] == SHA
    assert not [n for n in os.listdir(registry.root) if n.endswith(".tmp")]  # atomic write left nothing


def test_pause_then_resume_rehashes_and_uses_a_range(fake_hf, tmp_path):
    fake_hf.chunk_delay = 0.01
    registry, manager = _manager(tmp_path)
    phases = []

    async def scenario():
        spec = _spec(registry)

        def on_state(state):
            phases.append(state.phase)
            if state.phase == "downloading" and state.bytes_done > 1024 * 1024 and state.running:
                manager.pause(spec.stem)

        unsub = manager.subscribe(on_state)
        first = await manager.download(spec)
        unsub()
        assert first.paused and not first.done and 0 < first.bytes_done < len(PAYLOAD)
        assert first.status.startswith("Paused")
        part = registry.file_path(spec.stem) + ".part"
        assert os.path.getsize(part) == first.bytes_done and os.path.exists(part + ".json")
        # a new manager (app restart) lists the part as paused; nothing resumes on its own
        _r, again = _manager(tmp_path)
        paused = again.scan()
        assert [s.stem for s in paused] == [spec.stem] and paused[0].bytes_done == first.bytes_done
        phases.clear()
        fake_hf.chunk_delay = 0
        again.subscribe(lambda s: phases.append(s.phase))
        resumed = await again.download(again.pending_spec(spec.stem))
        return first, resumed

    first, resumed = asyncio.run(scenario())
    assert resumed.done and "checking" in phases and "verifying" in phases
    ranges = [r for _p, r in fake_hf.requests if r]
    assert ranges and ranges[-1] == f"bytes={first.bytes_done}-"
    assert registry.load()["Tiny-Q4_K_M"]["sha256"] == SHA


def test_a_server_that_ignores_range_restarts_cleanly(fake_hf, tmp_path):
    registry = dl.Registry(str(tmp_path / "models"))
    spec = _spec(registry)
    os.makedirs(registry.root)
    part = registry.file_path(spec.stem) + ".part"
    with open(part, "wb") as handle:
        handle.write(PAYLOAD[:500000])
    with open(part + ".json", "w", encoding="utf-8") as handle:
        json.dump({"repo": REPO, "file": NAME, "revision": spec.revision, "size": spec.size, "sha256": SHA}, handle)
    fake_hf.ignore_range = True
    entry = dl.download_blocking(spec, registry, progress=lambda u: None, should_stop=lambda: None)
    assert entry["sha256"] == SHA and entry["size"] == len(PAYLOAD)
    assert [r for _p, r in fake_hf.requests if r] == ["bytes=500000-"]


def test_a_dropped_connection_is_retried_and_resumed(fake_hf, tmp_path):
    registry = dl.Registry(str(tmp_path / "models"))
    fake_hf.cut_after = 2_500_000  # after two 1 MiB blocks reached the disk
    updates = []
    entry = dl.download_blocking(_spec(registry), registry, progress=updates.append, should_stop=lambda: None)
    assert entry["sha256"] == SHA
    ranges = [r for _p, r in fake_hf.requests if r]
    assert len(ranges) == 1 and ranges[0].startswith("bytes=")
    assert any("retrying" in str(u.get("status", "")) for u in updates)


def test_checksum_mismatch_deletes_the_part(fake_hf, tmp_path):
    registry, manager = _manager(tmp_path)
    state = asyncio.run(manager.download(_spec(registry, sha="0" * 64)))
    assert state.error == "checksum mismatch" and state.status == "Failed: checksum mismatch"
    assert os.listdir(registry.root) == []  # no part, no sidecar, no registry entry
    assert registry.load() == {}


def test_not_enough_storage_is_refused_before_any_byte(fake_hf, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "disk_usage", lambda path: SimpleNamespace(total=10**9, used=0, free=100 * 1024 * 1024))
    registry, manager = _manager(tmp_path)
    state = asyncio.run(manager.download(_spec(registry)))
    assert state.error.startswith("Not enough storage: ") and "needed" in state.error and "free" in state.error
    assert not any("resolve" in p for p, _r in fake_hf.requests)


def test_cancel_deletes_the_part_and_sidecar(fake_hf, tmp_path):
    fake_hf.chunk_delay = 0.01
    registry, manager = _manager(tmp_path)

    async def scenario():
        spec = _spec(registry)

        def on_state(state):
            if state.phase == "downloading" and state.bytes_done > 500_000 and state.running:
                manager.cancel(spec.stem)

        manager.subscribe(on_state)
        return await manager.download(spec)

    state = asyncio.run(scenario())
    assert state.cancelled and state.status == "Cancelled" and not state.done
    assert os.listdir(registry.root) == []
    assert "Tiny-Q4_K_M" not in manager.states


def test_progress_is_throttled_but_phase_changes_and_the_end_always_arrive(fake_hf, tmp_path):
    registry, manager = _manager(tmp_path, throttle=60)
    seen = []
    manager.subscribe(lambda s: seen.append((s.phase, s.running)))
    state = asyncio.run(manager.download(_spec(registry)))
    assert state.done
    assert len(seen) <= 5, seen  # start, the first downloading tick, verifying, end
    assert seen[-1] == ("done", False)


def test_stems_collide_by_owner_and_delete_removes_the_file(fake_hf, tmp_path):
    registry = dl.Registry(str(tmp_path / "models"))
    dl.download_blocking(_spec(registry), registry, progress=lambda u: None, should_stop=lambda: None)
    fake_hf.add("other/Tiny-GGUF", {NAME: PAYLOAD})
    spec = _spec(registry, repo="other/Tiny-GGUF")
    assert spec.stem == "Tiny-Q4_K_M-other"
    dl.download_blocking(spec, registry, progress=lambda u: None, should_stop=lambda: None)
    assert sorted(registry.load()) == ["Tiny-Q4_K_M", "Tiny-Q4_K_M-other"]
    assert registry.stem_for(REPO, NAME) == "Tiny-Q4_K_M"  # an existing download keeps its stem
    assert registry.remove("Tiny-Q4_K_M") and not os.path.exists(registry.file_path("Tiny-Q4_K_M"))
    assert sorted(registry.load()) == ["Tiny-Q4_K_M-other"]


def test_format_bytes():
    assert dl.format_bytes(0) == "0 B" and dl.format_bytes(532517120) == "533 MB"
    assert dl.format_bytes(5680522464) == "5.7 GB" and dl.format_bytes(1280835840) == "1.3 GB"
