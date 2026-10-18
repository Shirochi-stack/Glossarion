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
