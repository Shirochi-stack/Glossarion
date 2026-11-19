"""The "Run on" dropdown and the presets list. Hotfix 3 and 6 made the former "Use GPU (experimental)" switch add the
GPU-friendly Q4_0 presets (``presets.GPU_PRESETS``); owner 2026-10-31: the Custom rows replace them, so choosing
where the model runs only changes that, on Android and iOS alike, and no build, choice or offer adds a preset. What the
former rows left behind on a phone is in test_devfix7_custom_models.py. FakeEngine / FakePrefs, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")
pytest.importorskip("msgpack")

import local_llm_fakes as fakes  # noqa: E402
from glossarion_mobile.services.local_llm import presets  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from test_local_llm_server import make_service  # noqa: E402

DEFAULT_NAMES = [p["name"] for p in presets.DEFAULT_PRESETS]


def _panel(tmp_path, backend="OpenCL", prefs=None):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    prefs = prefs if prefs is not None else fakes.FakePrefs()
    service, _engine = make_service(tmp_path, prefs=prefs, gpu_backend=backend)
    said: list = []
    panel = LocalModelsPanel(service=service, prefs=prefs, notify=said.append)
    panel.build()
    return panel, prefs, service, said


def _names(prefs) -> list:
    return [p["name"] for p in presets.load_presets(prefs)]


def _pick(dropdown, mode: str) -> None:
    """The user picks a mode: Flet sets the value, then runs on_select."""
    dropdown.value = mode
    dropdown.on_select(None)


@pytest.mark.parametrize("backend", ["OpenCL", "Metal"])
def test_choosing_a_mode_changes_no_preset_and_says_nothing(tmp_path, backend):
    panel, prefs, service, said = _panel(tmp_path, backend=backend)
    assert panel.run_mode.visible and panel.run_mode.value == "auto" and _names(prefs) == DEFAULT_NAMES
    for mode, gpu in (("cpu_gpu", True), ("gpu", True), ("cpu", False), ("auto", False)):
        _pick(panel.run_mode, mode)
        assert prefs.get(svc.RUN_MODE_KEY) == mode and service.use_gpu() is gpu and _names(prefs) == DEFAULT_NAMES
    assert said == [] and service.registry.resolve("fake-echo")


def test_a_build_without_a_gpu_backend_hides_the_dropdown(tmp_path):
    panel, prefs, service, said = _panel(tmp_path, backend=None)
    assert not panel.run_mode.visible and panel.run_mode.options == []
    _pick(panel.run_mode, "cpu_gpu")
    assert service.run_mode() == "auto" and not service.use_gpu() and panel.run_mode.value is None
    assert _names(prefs) == DEFAULT_NAMES and said == []


def test_the_welcome_page_dropdown_keeps_its_cells(tmp_path):
    """The Welcome page ("Run on this phone") shows the same dropdown (``OnDeviceFlow._run_section``)."""
    from test_welcome_drawer_compact import _engine_kwargs, by_key, screen

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    count = len(DEFAULT_NAMES) + len(presets.CUSTOM_FAMILIES)
    assert rows["welcome-run-mode"].value == "auto" and _names(kwargs["prefs"]) == DEFAULT_NAMES
    assert len(rows["welcome-ondevice"].controls) == count and "welcome-gpu-presets" not in rows
    _pick(rows["welcome-run-mode"], "cpu_gpu")
    assert kwargs["prefs"].get(svc.RUN_MODE_KEY) == "cpu_gpu" and _names(kwargs["prefs"]) == DEFAULT_NAMES
    rows = by_key(s.page_area.content)
    assert rows["welcome-run-mode"].value == "cpu_gpu" and len(rows["welcome-ondevice"].controls) == count
    assert "welcome-model-local" not in rows  # no note
