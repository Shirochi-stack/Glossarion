"""The "Use GPU (experimental)" switch and the presets list. Hotfix 3 and 6 made the switch add the GPU-friendly
Q4_0 presets (``presets.GPU_PRESETS``); owner 2026-10-31: the Custom rows replace them, so the switch only turns
the GPU on and off on Android and iOS alike, and no build, tap or offer adds a preset. What the former rows left
behind on a phone is in test_devfix7_custom_models.py. FakeEngine / FakePrefs, no network.
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


def _tap_switch(panel, value: bool) -> None:
    """The user flips the switch: Flet sets the value, then runs on_change."""
    panel.gpu_switch.value = value
    panel.gpu_switch.on_change(None)


@pytest.mark.parametrize("backend", ["OpenCL", "Metal"])
def test_the_switch_changes_no_preset_and_says_nothing(tmp_path, backend):
    panel, prefs, service, said = _panel(tmp_path, backend=backend)
    assert panel.gpu_switch.visible and panel.gpu_switch.value is False and _names(prefs) == DEFAULT_NAMES
    _tap_switch(panel, True)
    assert prefs.get(svc.GPU_KEY) is True and service.use_gpu() and _names(prefs) == DEFAULT_NAMES
    _tap_switch(panel, False)
    assert prefs.get(svc.GPU_KEY) is False and not service.use_gpu() and _names(prefs) == DEFAULT_NAMES
    assert said == [] and service.registry.resolve("fake-echo")


def test_a_build_without_a_gpu_backend_hides_the_switch(tmp_path):
    panel, prefs, _service, said = _panel(tmp_path, backend=None)
    assert not panel.gpu_switch.visible
    panel.gpu_switch.value = True
    assert panel.save_gpu() is False and panel.gpu_switch.value is False
    assert _names(prefs) == DEFAULT_NAMES and said == []


def test_the_welcome_page_switch_keeps_its_cells(tmp_path):
    """The Welcome page ("Run on this phone") shows the same switch (``OnDeviceFlow._gpu_section``)."""
    from test_welcome_drawer_compact import _engine_kwargs, by_key, screen

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    count = len(DEFAULT_NAMES) + len(presets.CUSTOM_FAMILIES)
    assert rows["welcome-gpu"].value is False and _names(kwargs["prefs"]) == DEFAULT_NAMES
    assert len(rows["welcome-ondevice"].controls) == count and "welcome-gpu-presets" not in rows
    switch = rows["welcome-gpu"]
    switch.value = True
    switch.on_change(None)
    assert kwargs["prefs"].get(svc.GPU_KEY) is True and _names(kwargs["prefs"]) == DEFAULT_NAMES
    rows = by_key(s.page_area.content)
    assert rows["welcome-gpu"].value is True and len(rows["welcome-ondevice"].controls) == count
    assert "welcome-model-local" not in rows  # no note
