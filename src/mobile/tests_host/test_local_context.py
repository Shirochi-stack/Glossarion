"""On-device Context (owner 2026-10-18): the ``ContextControl`` slider + field (Model manager › On-device and
the Welcome Request size step), choosing a model resets it to that model's own (4,096 for Heavy), a changed
Context reloads the model, and a job on ``local/`` sends batch size 1 by default and half the context as
its output limit.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_context.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")

import local_llm_fakes as fakes  # noqa: E402
from glossarion_mobile.services.local_llm import downloads as dl  # noqa: E402
from glossarion_mobile.services.local_llm import engine as eng  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from glossarion_mobile.state import setting_writes as sw  # noqa: E402

LIGHT, HEAVY = "Qwen3.5-2B-UD-Q4_K_XL", "Qwen3.5-9B-Q4_K_M"


def _service(tmp_path):
    engine = eng.FakeEngine()
    service = svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, memory=lambda: (0, None))
    for stem, size in ((LIGHT, 1339752704), (HEAVY, 5680522464)):
        service.registry.add(stem, {"repo": "unsloth/x", "file": f"{stem}.gguf", "size": size})
        Path(service.registry.file_path(stem)).write_bytes(b"gguf")
    return service, engine


def _event(value):
    return type("E", (), {"control": type("C", (), {"value": value})()})()


def test_context_control_slider_steps_and_the_field_goes_past_16384():
    from glossarion_mobile.ui.screens.local_models_panel import ContextControl

    seen = []
    control = ContextControl(8192, seen.append)
    assert (control.slider.min, control.slider.max, control.slider.divisions) == (2048, 16384, 14)
    assert control.slider.value == 8192 and control.field.value == "8192"
    control._from_slider(_event(12100.0))  # snaps to 1,024 steps
    assert seen[-1] == 12288 and control.field.value == "12288"
    control._from_field(_event("32768"))  # the field goes beyond the slider, which stays at its end
    assert seen[-1] == 32768 and control.slider.value == 16384
    count = len(seen)
    for bad in ("", "abc", "100", "999999"):  # not a number, below 512, above presets.MAX_CTX
        control._from_field(_event(bad))
    assert len(seen) == count
    control.set_value(4096)
    assert control.slider.value == 4096 and control.field.value == "4096" and control.value == 4096


def test_choosing_heavy_resets_the_context_and_the_slider_shows_4096(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    service, engine = _service(tmp_path)
    model = {"id": f"local/{LIGHT}"}
    panel = LocalModelsPanel(service=service, prefs=service.prefs, use_model=lambda m: model.update(id=m),
                             current_model=lambda: model["id"])
    panel.build()
    assert panel.context.value == 8192  # Light's own context (a file under 4 GB)
    assert panel.save_context(16384) is None and service.settings()["n_ctx"] == 16384
    assert service.context_for(LIGHT) == 16384

    assert asyncio.run(panel.activate(HEAVY, "Heavy"))
    assert model["id"] == f"local/{HEAVY}" and service.settings()["n_ctx"] == 0  # back to the model's own
    assert panel.context.value == 4096 and panel.context.slider.value == 4096 and panel.ctx_field.value == "4096"
    assert engine.loads[-1][1] == 4096

    assert asyncio.run(panel.activate(LIGHT, "Light"))
    assert panel.context.value == 8192 and engine.loads[-1][1] == 8192
    panel.dispose()


def test_a_changed_context_reloads_the_loaded_model(tmp_path):
    service, engine = _service(tmp_path)
    service.load_blocking(LIGHT)
    service.load_blocking(LIGHT)
    assert [load[1] for load in engine.loads] == [8192]  # same context: no reload
    service.save_context(12288)
    service.load_blocking(LIGHT)
    assert [load[1] for load in engine.loads] == [8192, 12288]


def test_local_jobs_use_batch_size_1_and_half_the_context_as_output(tmp_path):
    service, _engine = _service(tmp_path)
    assert sw.default_batch_size(f"local/{LIGHT}") == 1
    assert sw.default_batch_size("gpt-6") == 5 and sw.default_batch_size("authds/deepseek") == 1
    svc.set_default_service(service)
    try:
        config = sw.with_mobile_streaming_defaults({"model": f"local/{LIGHT}"})
        assert config["batch_size"] == 1 and config["max_output_tokens"] == 4096
        service.save_context(20000)
        config = sw.with_mobile_streaming_defaults({"model": f"local/{LIGHT}", "max_output_tokens": 16384,
                                                    "batch_size": 3})
        assert config["max_output_tokens"] == 10000 and config["batch_size"] == 3  # a saved batch size wins
        service.save_context(0)
        assert sw.with_mobile_streaming_defaults({"model": f"local/{HEAVY}"})["max_output_tokens"] == 2048
        # a cloud model keeps the saved budget
        assert sw.with_mobile_streaming_defaults({"model": "gpt-6", "max_output_tokens": 16384})["max_output_tokens"] == 16384
    finally:
        svc.set_default_service(None)
    assert sw.local_output_limit(f"local/{LIGHT}") is None  # no engine service: the saved budget stays


def test_a_job_exports_each_on_device_models_context_for_the_glossary_chunk_budget(tmp_path, monkeypatch):
    """owner 2026-10-18: the Minimal glossary sent ~20,000 tokens to an 8,192-token model; a job now exports
    every local/ model's context (CONTEXT_ENV) which the shared code reads to size its chunks."""
    import os

    from glossarion_mobile.services import jobs
    from unified_api_client import model_context_window

    monkeypatch.delenv(svc.CONTEXT_ENV, raising=False)
    service, _engine = _service(tmp_path)
    with jobs.local_context_scope():  # no engine service: nothing is exported
        assert svc.CONTEXT_ENV not in os.environ
    svc.set_default_service(service)
    try:
        assert sw.local_context_env() == f"local/{LIGHT}=8192;local/{HEAVY}=4096"
        service.save_context(12288)
        with jobs.local_context_scope():
            assert os.environ[svc.CONTEXT_ENV] == f"local/{LIGHT}=12288;local/{HEAVY}=12288"
            assert model_context_window(f"local/{LIGHT}") == 12288
        assert svc.CONTEXT_ENV not in os.environ  # restored when the job ends
    finally:
        svc.set_default_service(None)


def _welcome(service, model):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.welcome import WelcomeScreen

    s = WelcomeScreen(parse_route("/welcome"), oauth=object(), navigate=lambda r: None,
                      local_engine=lambda: service, prefs=service.prefs, current_model=lambda: model)
    s.get_body()
    return s


def _by_key(root) -> dict:
    out, stack = {}, [root]
    while stack:
        c = stack.pop()
        key = getattr(c, "key", None)
        if key:
            out[str(key)] = c
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)
    return out


def test_welcome_request_size_shows_the_context_for_an_on_device_model(tmp_path):
    service, _engine = _service(tmp_path)
    s = _welcome(service, f"local/{HEAVY}")
    rows = _by_key(s._run_budget())
    assert rows["welcome-ctx"].value == "4096" and rows["welcome-ctx-slider"].value == 4096
    assert rows["welcome-max-tokens"].value == "2048" and rows["welcome-max-tokens"].disabled
    assert rows["welcome-max-tokens-slider"].disabled
    assert s.default_batch_size() == 1
    s.context_control._from_slider(_event(10240.0))
    assert service.settings()["n_ctx"] == 10240 and rows["welcome-max-tokens"].value == "5120"
    assert s.flow.max_output_tokens is None  # the saved budget stays for cloud models

    cloud = _by_key(_welcome(service, "gpt-6")._run_budget())
    assert "welcome-ctx" not in cloud and not cloud["welcome-max-tokens"].disabled
    assert cloud["welcome-max-tokens"].value == "16384"

def test_welcome_finish_saves_the_request_size_it_showed(tmp_path):
    """owner 2026-10-18: Welcome showed batch size 1 and half the context for an on-device model, but saved
    nothing untouched, so Run options kept 5 and 128,000."""
    service, _engine = _service(tmp_path)
    s = _welcome(service, f"local/{HEAVY}")
    updates = s.finish()
    assert updates["batch_size"] == 1 and updates["max_output_tokens"] == 2048  # Heavy: 4,096 context

    s = _welcome(service, "gpt-6")
    updates = s.finish()
    assert updates["batch_size"] == 5 and updates["max_output_tokens"] == 16384
    # values the user set still win
    s = _welcome(service, "gpt-6")
    s.flow.batch_size, s.flow.max_output_tokens = 3, 8192
    updates = s.finish()
    assert updates["batch_size"] == 3 and updates["max_output_tokens"] == 8192

def test_on_device_sampling_follows_qwen_and_replaces_the_request(tmp_path):
    """owner 2026-10-18: on-device requests use their own sampling (Qwen's recommendation: presence penalty
    1.5 ...) instead of the global anti-duplicate values; the temperature stays the request's."""
    service, _engine = _service(tmp_path)
    assert service.sampling() == svc.DEFAULT_ONDEVICE_SAMPLING and "temperature" not in service.sampling()
    assert svc.DEFAULT_ONDEVICE_SAMPLING["presence_penalty"] == 1.0
    params = service._params({"temperature": 0.7, "presence_penalty": 0.0, "frequency_penalty": 0.4, "seed": 7,
                              "max_tokens": 100}, 4096)
    assert params["temperature"] == 0.7 and params["presence_penalty"] == 1.0 and params["repeat_penalty"] == 1.0
    assert params["top_p"] == 0.8 and params["top_k"] == 20 and params["min_p"] == 0.0
    assert "frequency_penalty" not in params and params["seed"] == 7 and params["max_tokens"] == 100

    saved, error = service.save_sampling({"top_k": "40", "presence_penalty": "", "repeat_penalty": "1.1"})
    assert error is None and saved["top_k"] == 40
    assert saved["presence_penalty"] == 1.0 and saved["repeat_penalty"] == 1.1  # blank -> Qwen's value
    assert service._params({}, 512)["top_k"] == 40
    assert service.save_sampling({"min_p": "3"})[1] == "Min P must be between 0 and 1"
    assert service.save_sampling({"top_k": "2.5"})[1] == "Top K must be a whole number"
    assert service.save_sampling({"top_p": "abc"})[1] == "Top P must be a number"
    assert service.sampling()["top_k"] == 40  # a rejected save changes nothing


def test_on_device_engine_card_saves_and_resets_sampling(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    service, _engine = _service(tmp_path)
    panel = LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: f"local/{LIGHT}")
    panel.build()
    assert "temperature" not in panel.sampling_fields and panel.sampling_fields["presence_penalty"].value == "1"
    panel.sampling_fields["presence_penalty"].value = "1.2"
    assert panel.save_sampling() is None and service.sampling()["presence_penalty"] == 1.2
    panel.sampling_fields["min_p"].value = "9"
    assert panel.save_sampling() and panel.sampling_error.visible
    assert panel.reset_sampling() is None and service.sampling() == svc.DEFAULT_ONDEVICE_SAMPLING
    assert panel.sampling_fields["presence_penalty"].value == "1"
    panel.dispose()

def test_welcome_temperature_is_0_7_only_for_an_on_device_model(tmp_path):
    """owner 2026-10-18: an on-device model gets temperature 0.7; cloud models keep the saved one / 0.3."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.welcome import WelcomeScreen

    def welcome(saved, model):
        s = WelcomeScreen(parse_route("/welcome"), oauth=object(), navigate=lambda r: None,
                          current_model=lambda: model, config_get=lambda key, default=None: saved.get(key, default))
        s.get_body()
        return s

    for saved in ({}, {"translation_temperature": 0.3}, {"translation_temperature": 1.1}):
        s = welcome(saved, f"local/{LIGHT}")
        assert s.shown_temperature() == 0.7 and s.finish()["translation_temperature"] == 0.7
    s = welcome({}, "gpt-6")
    assert s.shown_temperature() == 0.3 and s.finish()["translation_temperature"] == 0.3
    s = welcome({"translation_temperature": 1.1}, "gpt-6")
    assert s.shown_temperature() == 1.1 and s.finish()["translation_temperature"] == 1.1
    s = welcome({}, f"local/{LIGHT}")
    s.flow.temperature = 0.2  # a value set in Welcome wins
    assert s.finish()["translation_temperature"] == 0.2
