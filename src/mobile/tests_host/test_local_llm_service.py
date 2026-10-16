"""``LocalEngineService`` (``services/local_llm/service.py``): one model loaded at a time, the unload
policies (fake clock and lifecycle names), the RAM gate, thread / context defaults, the catalog refresh
after a download, the job-config rule for ``local/`` models and the engine pin. FakeEngine only.
"""

from __future__ import annotations

import asyncio
import os
import re
import tomllib
from pathlib import Path

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf
from glossarion_mobile.services.local_llm import service as svc
from test_local_llm_server import make_service

GB = 1024 ** 3


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _chat(service, model, text="hi"):
    run = service.chat(model, {"messages": [{"role": "user", "content": text}]}, lambda: False)
    with run:
        return "".join(run.deltas)


def test_one_model_loaded_at_a_time(tmp_path):
    service, engine = make_service(tmp_path, stems=("a", "b"))
    assert _chat(service, "local/a") == "Echo: hi" and service.loaded_stem == "a"
    assert _chat(service, "a", "again") == "Echo: again" and len(engine.loads) == 1  # no reload
    assert _chat(service, "local/B") == "Echo: hi" and service.loaded_stem == "b"
    assert [os.path.basename(p) for p, _c, _t in engine.loads] == ["a.gguf", "b.gguf"] and engine.unloads == 1
    assert service.load_blocking("a") == (True, None) and service.loaded_stem == "a"
    assert service.unload() and service.loaded_stem is None and service.snapshot().state == "idle"


def test_unload_policies(tmp_path):
    clock = Clock()
    job = {"model": None}
    service, engine = make_service(tmp_path, clock=clock, active_job_model=lambda: job["model"])
    service.platform = "android"
    _chat(service, "fake-echo")
    clock.now += svc.IDLE_UNLOAD - 1
    assert service.tick() is None and service.loaded_stem == "fake-echo"
    clock.now += 2
    job["model"] = "local/fake-echo"
    assert service.tick() is None  # a running job keeps its model
    job["model"] = None
    assert service.tick() == "idle" and service.loaded_stem is None
    # Android: 60 s in the background unloads, unless a job runs; resume forgets the background time
    _chat(service, "fake-echo")
    assert service.on_lifecycle("hide") is None and service.loaded_stem == "fake-echo"
    clock.now += svc.BACKGROUND_UNLOAD - 1
    assert service.tick() is None
    service.on_lifecycle("resume")
    clock.now += 30
    assert service.tick() is None and service.background_since is None
    service.on_lifecycle("pause")
    clock.now += svc.BACKGROUND_UNLOAD
    job["model"] = "gpt-6"
    assert service.tick() is None  # any running job keeps the process (and the model)
    job["model"] = None
    assert service.tick() == "background" and service.loaded_stem is None
    # iOS: hiding unloads at once (jetsam), unless a job runs
    service.on_lifecycle("resume")
    service.platform = "ios"
    _chat(service, "fake-echo")
    job["model"] = "local/fake-echo"
    assert service.on_lifecycle("hide") is None and service.loaded_stem == "fake-echo"
    job["model"] = None
    service.on_lifecycle("resume")
    assert service.on_lifecycle("hide") == "unloaded" and service.loaded_stem is None
    # detach: unload and stop the endpoint
    service.start()
    assert service.server.running and os.environ.get(svc.URL_ENV) == service.server.base_url
    _chat(service, "fake-echo")
    assert service.on_lifecycle("detach") == "stopped"
    assert not service.server.running and service.loaded_stem is None and svc.URL_ENV not in os.environ
    assert svc.default_service() is None


def test_ram_gate(tmp_path):
    memory = {"value": (8 * GB, 6 * GB)}
    service, engine = make_service(tmp_path, memory=lambda: memory["value"])
    service.registry.add("fake-echo", {"repo": "t/fake-echo", "file": "fake-echo.gguf", "size": 2 * GB, "sha256": "0" * 64})
    service.platform = "android"
    need = svc.ram_need(2 * GB, 8192)
    assert need == int(2 * GB * 1.15 + 8192 * 64 * 1024 + 400 * 1024 * 1024)
    assert service.ram_check("fake-echo") == ("ok", "")
    memory["value"] = (8 * GB, 2 * GB)
    verdict, message = service.ram_check("fake-echo")
    assert verdict == "confirm" and message.endswith("Android may close Glossarion while this model runs. Load anyway?")
    ok, error = service.load_blocking("fake-echo")
    assert not ok and "Load anyway?" in error  # the screen asks
    with pytest.raises(svc.LocalLLMError) as info:  # a job / chat request never asks: it fails with the reason
        _chat(service, "fake-echo")
    assert info.value.status == 507 and "Load it in Model manager › On-device to run it anyway." in info.value.message
    assert service.load_blocking("fake-echo", force=True) == (True, None)
    service.unload()
    memory["value"] = (2 * GB, 1 * GB)
    verdict, message = service.ram_check("fake-echo")
    assert verdict == "refuse" and message.startswith("This phone has 2.1 GB RAM;")
    assert service.load_blocking("fake-echo", force=True)[0] is False
    service.platform = "ios"
    memory["value"] = (8 * GB, None)  # need ~3.2 GB: under half of 8 GB
    assert service.ram_check("fake-echo")[0] == "ok"
    memory["value"] = (4 * GB, None)
    verdict, message = service.ram_check("fake-echo")
    assert verdict == "refuse" and "about half" in message


def test_thread_and_context_defaults(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    for cpus, threads in ((8, 4), (6, 4), (4, 2), (2, 2), (16, 4)):
        monkeypatch.setattr(os, "cpu_count", lambda c=cpus: c)
        assert eng.default_threads() == threads
    prefs = fakes.FakePrefs()
    service, engine = make_service(tmp_path, prefs=prefs)
    service.registry.add("fake-echo", {"repo": "t/fake-echo", "file": "fake-echo.gguf", "size": 1 * GB})
    assert service.context_for("fake-echo") == 8192
    service.registry.add("fake-echo", {"repo": "t/fake-echo", "file": "fake-echo.gguf", "size": 5 * GB})
    assert service.context_for("fake-echo") == 4096
    assert service.save_settings("4096x", "") == (None, "Context must be a whole number")
    assert service.save_settings("6000", "3") == ({"n_ctx": 6000, "n_threads": 3}, None)
    assert service.context_for("fake-echo") == 6000 and service.threads() == 3
    prefs.data["local_model_presets"] = [{"name": "Mine", "repo": "t/fake-echo", "file": "fake-echo.gguf", "n_ctx": 2048}]
    assert service.context_for("fake-echo") == 2048  # the preset's context wins
    _chat(service, "fake-echo")
    assert engine.loads[-1][1:] == (2048, 3)


def test_download_refreshes_the_catalog_and_delete_unloads(tmp_path, monkeypatch):
    server = fakes.FakeHF()
    payload = os.urandom(200_000)
    server.add("tester/Tiny-GGUF", {"Tiny-Q4_K_M.gguf": payload, "mmproj-F16.gguf": b"x"})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    refreshed = []
    try:
        service, engine = make_service(tmp_path, stems=(), refresh_catalog=lambda: refreshed.append(1))

        async def scenario():
            commit, files = await service.resolve(hf.HfRef("tester/Tiny-GGUF"))
            spec = service.spec_for("tester/Tiny-GGUF", commit, hf.pick_file(files))
            return await service.download(spec)

        state = asyncio.run(scenario())
    finally:
        server.close()
    assert state.done and refreshed == [1]
    assert [m["id"] for m in service.list_models()] == ["Tiny-Q4_K_M"]
    _chat(service, "local/Tiny-Q4_K_M")
    assert service.delete("Tiny-Q4_K_M") and service.loaded_stem is None and refreshed == [1, 1]
    assert service.list_models() == [] and service.registry.resolve("local/Tiny-Q4_K_M") is None


def test_model_block_only_while_the_app_service_runs(tmp_path, monkeypatch):
    from glossarion_mobile.services import model_catalog as mc

    monkeypatch.delenv(svc.URL_ENV, raising=False)
    service, _engine = make_service(tmp_path)
    assert mc.model_block("local/anything") is None  # no service: local/ is a user's own llama-server
    service.start()
    try:
        assert mc.model_block("local/fake-echo") is None
        reason, detail = mc.model_block("local/Qwen3.5-2B-Q4_K_M")
        assert reason == "local/Qwen3.5-2B-Q4_K_M is not downloaded"
        assert detail.endswith("Download it in Model manager › On-device.")
        assert mc.chip_of(mc.provider_of("local/fake-echo")) == "local"
        assert mc.provider_label("local") == "On this device"
    finally:
        service.stop()
    assert mc.model_block("local/Qwen3.5-2B-Q4_K_M") is None


def test_local_jobs_run_serially_whatever_the_settings_say():
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

    config = with_mobile_streaming_defaults({"model": "local/Qwen3.5-2B-Q4_K_M", "batch_translation": True,
                                             "enable_parallel_extraction": True, "extraction_workers": 4})
    assert (config["batch_translation"], config["enable_parallel_extraction"], config["extraction_workers"]) == (False, False, 1)
    cloud = with_mobile_streaming_defaults({"model": "gpt-6", "batch_translation": True})
    assert cloud["batch_translation"] is True


def test_engine_status_without_llama_cpp(monkeypatch, tmp_path):
    monkeypatch.setattr(eng.importlib.util, "find_spec", lambda name: None)
    available, reason = eng.engine_status()
    assert not available and "llama-cpp-python is not installed" in reason
    service = svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=dl.Registry(str(tmp_path / "m")), watch=False)
    assert service.snapshot().available is False and service.snapshot().unavailable_reason == reason
    with pytest.raises(svc.LocalLLMError) as info:
        service._ensure_loaded("x")
    assert info.value.status == 503 and "not available" in info.value.message


def test_engine_version_matches_the_pyproject_pin():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    flet = pyproject["tool"]["flet"]
    pins = {platform: [d for d in flet[platform].get("dependencies", []) if d.lower().startswith("llama-cpp-python")]
            for platform in ("android", "ios")}
    if not any(pins.values()):
        pytest.skip("llama-cpp-python is not pinned in pyproject.toml yet (packaging lands separately)")
    for platform, found in pins.items():
        assert found, f"no llama-cpp-python pin for {platform}"
        assert re.fullmatch(r"llama-cpp-python==(.+)", found[0]).group(1) == eng.ENGINE_VERSION
