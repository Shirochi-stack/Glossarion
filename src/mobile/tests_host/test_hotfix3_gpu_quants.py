"""Hotfix 3, issue 5 (owner 2026-10-26: "Use gpu should automatically show the gpu friendly quants, not
require an extra tap"): switching "Use GPU (experimental)" on with Android's OpenCL backend adds the
GPU-friendly Q4_0 presets (``presets.GPU_PRESETS``) at once, through the same ``add_gpu_presets`` the
"Add GPU-friendly presets (Q4_0)" button calls: idempotent, the list re-renders, a short note says so.
Switching it off keeps every preset and downloaded file; a build of the panel never changes the list (a
removed Q4_0 preset stays removed, the button is the fallback); iOS (Metal) and builds without a GPU
backend add nothing. FakeEngine / FakePrefs, no network.

Hotfix 6 (owner 2026-10-27: "gpu friendly presets are always shown regardless of toggle state"): on
Android (OpenCL) the first time the presets show (Model manager, Welcome page, model picker) they are
added quietly whatever the switch says (``offer_gpu_presets_once``), and the button shows while one is
missing, switch on or off.
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

GPU_NAMES = [p["name"] for p in presets.GPU_PRESETS]
GPU_FILES = [p["file"] for p in presets.GPU_PRESETS]
DEFAULT_NAMES = [p["name"] for p in presets.DEFAULT_PRESETS]
ADDED_FOUR = "Added 4 GPU-friendly presets (Q4_0)"


def _walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for name in ("controls", "content"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                stack.extend(reversed([c for c in child if isinstance(c, ft.Control)]))
            elif isinstance(child, ft.Control):
                stack.append(child)


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


def _shown(panel) -> list:
    """The preset names the Models card shows (each row's first text is its name)."""
    return [next(c.value for c in _walk(row) if isinstance(c, ft.Text)) for row in panel.preset_column.controls]


def _tap_switch(panel, value: bool) -> None:
    """The user flips the switch: Flet sets the value, then runs on_change."""
    panel.gpu_switch.value = value
    panel.gpu_switch.on_change(None)


def test_switching_the_gpu_on_adds_the_q4_0_presets_at_once(tmp_path):
    panel, prefs, service, said = _panel(tmp_path)
    assert panel.gpu_switch.visible and panel.gpu_switch.value is False
    assert _names(prefs) == DEFAULT_NAMES and _shown(panel) == DEFAULT_NAMES
    assert panel.gpu_presets_button.visible  # hotfix 6: offered while missing, whatever the switch says
    # on: the Q4_0 presets join the list, the card shows them at once, one short note, no button
    _tap_switch(panel, True)
    assert prefs.get(svc.GPU_KEY) is True and service.use_gpu()
    assert _names(prefs) == DEFAULT_NAMES + GPU_NAMES and _shown(panel) == DEFAULT_NAMES + GPU_NAMES
    assert not panel.gpu_presets_button.visible
    assert said == [ADDED_FOUR]
    # off: every preset and the downloaded model stay; nothing is said
    _tap_switch(panel, False)
    assert prefs.get(svc.GPU_KEY) is False and not service.use_gpu()
    assert _names(prefs) == DEFAULT_NAMES + GPU_NAMES and service.registry.resolve("fake-echo")
    assert not panel.gpu_presets_button.visible and said == [ADDED_FOUR]
    # on again: idempotent (no duplicate row, no second note, never "already in the list")
    _tap_switch(panel, True)
    files = [p.get("file") for p in presets.load_presets(prefs)]
    assert all(files.count(f) == 1 for f in GPU_FILES) and len(files) == 8
    assert said == [ADDED_FOUR]


def test_a_build_never_changes_the_list_and_the_button_stays_the_fallback(tmp_path):
    # switched on before this build (or the Q4_0 presets removed / Restore defaults while it is on): the
    # panel opens with the list as saved and offers the one-tap button
    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    panel, prefs, _service, said = _panel(tmp_path / "older", prefs=prefs)
    assert panel.gpu_switch.value is True and _names(prefs) == DEFAULT_NAMES and said == []
    assert panel.gpu_presets_button.visible
    panel.gpu_presets_button.on_click(None)
    assert _names(prefs) == DEFAULT_NAMES + GPU_NAMES and said == [ADDED_FOUR]
    assert not panel.gpu_presets_button.visible
    # a Q4_0 preset the user removes stays removed when the panel opens again
    light_gpu = len(DEFAULT_NAMES) + GPU_NAMES.index("Light · GPU (Q4_0)")
    assert panel.remove_preset(light_gpu)
    panel, prefs, _service, said = _panel(tmp_path / "again", prefs=prefs)
    assert "Light · GPU (Q4_0)" not in _names(prefs) and said == []
    assert panel.gpu_presets_button.visible  # one is missing: the button adds it back
    assert panel.add_gpu_presets() == ["Light · GPU (Q4_0)"] and said == ["Added 1 GPU-friendly preset (Q4_0)"]
    assert not panel.gpu_presets_button.visible
    # the button with nothing missing (a second tap while it hides) says so
    assert panel.add_gpu_presets() == [] and said[-1] == "The GPU-friendly presets are already in the list"


def test_metal_and_builds_without_a_gpu_backend_add_nothing(tmp_path):
    # iOS (Metal) runs the default K-quants: the switch saves, the list stays, no button
    panel, prefs, service, said = _panel(tmp_path / "ios", backend="Metal")
    _tap_switch(panel, True)
    assert prefs.get(svc.GPU_KEY) is True and service.use_gpu()
    assert _names(prefs) == DEFAULT_NAMES and not panel.gpu_presets_button.visible and said == []
    # no backend (x86_64 Android, desktop): the switch is hidden and cannot go on
    panel, prefs, service, said = _panel(tmp_path / "none", backend=None)
    assert not panel.gpu_switch.visible
    panel.gpu_switch.value = True
    assert panel.save_gpu() is False and panel.gpu_switch.value is False
    assert _names(prefs) == DEFAULT_NAMES and said == []


def test_a_full_list_keeps_what_fits(tmp_path):
    user = [{"name": f"u{i}", "repo": "me/x", "file": f"f{i}.gguf"} for i in range(presets.MAX_PRESETS - 2)]
    prefs = fakes.FakePrefs({presets.PREF_KEY: user})
    rows, added = presets.add_gpu_presets(prefs)
    assert added == GPU_NAMES[:2] and len(rows) == presets.MAX_PRESETS and _names(prefs)[-2:] == GPU_NAMES[:2]
    assert [p["name"] for p in presets.missing_gpu_presets(rows)] == GPU_NAMES[2:]
    # full: nothing is added and the saved list is untouched
    before = presets.load_presets(prefs)
    assert presets.add_gpu_presets(prefs) == (before, []) and presets.load_presets(prefs) == before
    # the switch says nothing; the button (two still missing) says why it cannot add them
    panel, prefs, _service, said = _panel(tmp_path, prefs=prefs)
    _tap_switch(panel, True)
    assert presets.load_presets(prefs) == before and said == [] and panel.gpu_presets_button.visible
    assert panel.add_gpu_presets() == []
    assert said == [f"Keep it to {presets.MAX_PRESETS} presets: remove one to add the GPU-friendly presets"]


def test_the_welcome_page_shows_them_whatever_the_switch_says(tmp_path):
    """The Welcome page ("Run on this phone") shows the same switch (``OnDeviceFlow._gpu_section``); hotfix 6:
    its cells include the Q4_0 presets with the switch off, added without a note."""
    from test_welcome_drawer_compact import _engine_kwargs, by_key, screen

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    assert rows["welcome-gpu"].value is False and _names(kwargs["prefs"]) == DEFAULT_NAMES + GPU_NAMES
    assert len(rows["welcome-ondevice"].controls) == len(DEFAULT_NAMES) + len(GPU_NAMES)
    assert not rows["welcome-gpu-presets"].visible and "welcome-model-local" not in rows  # no note
    switch = rows["welcome-gpu"]
    switch.value = True
    switch.on_change(None)
    assert kwargs["prefs"].get(svc.GPU_KEY) is True and _names(kwargs["prefs"]) == DEFAULT_NAMES + GPU_NAMES
    # the page re-rendered: the same cells, the switch on, no button, still no note
    rows = by_key(s.page_area.content)
    assert rows["welcome-gpu"].value is True and not rows["welcome-gpu-presets"].visible
    assert len(rows["welcome-ondevice"].controls) == len(DEFAULT_NAMES) + len(GPU_NAMES)
    assert "welcome-model-local" not in rows


def test_a_gpu_switched_on_before_this_build_gets_the_presets_once_when_the_model_manager_opens(tmp_path):
    """Hotfix 2 upgrade: 'Use GPU' was already on and the button was never tapped. Opening the Model
    manager adds the Q4_0 presets once (no tap); a preset removed afterwards stays removed."""
    from glossarion_mobile.ui.screens import local_models_panel as lmp

    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    panel, prefs, _service, said = _panel(tmp_path / "upgrade", prefs=prefs)
    assert _names(prefs) == DEFAULT_NAMES  # building the panel alone still changes nothing
    assert panel.offer_gpu_presets_once() == GPU_NAMES
    assert _names(prefs) == DEFAULT_NAMES + GPU_NAMES and said == []  # quiet (hotfix 6)
    assert prefs.get(lmp.GPU_PRESETS_OFFERED_KEY) is True and not panel.gpu_presets_button.visible
    light_gpu = len(DEFAULT_NAMES) + GPU_NAMES.index("Light · GPU (Q4_0)")
    assert panel.remove_preset(light_gpu)
    panel, prefs, _service, said = _panel(tmp_path / "reopen", prefs=prefs)
    assert panel.offer_gpu_presets_once() == [] and "Light · GPU (Q4_0)" not in _names(prefs) and said == []


def test_the_one_time_offer_ignores_the_switch_and_skips_metal_and_an_earlier_add(tmp_path):
    from glossarion_mobile.ui.screens import local_models_panel as lmp

    panel, prefs, _service, said = _panel(tmp_path / "off")  # GPU off: added all the same (hotfix 6)
    assert panel.offer_gpu_presets_once() == GPU_NAMES and _names(prefs) == DEFAULT_NAMES + GPU_NAMES
    assert prefs.get(lmp.GPU_PRESETS_OFFERED_KEY) is True and said == []
    panel, prefs, _service, said = _panel(tmp_path / "ios", backend="Metal", prefs=fakes.FakePrefs({svc.GPU_KEY: True}))
    assert panel.offer_gpu_presets_once() == [] and _names(prefs) == DEFAULT_NAMES
    # switched on in this build: the switch adds them, and the open afterwards does not repeat it
    panel, prefs, _service, said = _panel(tmp_path / "switched")
    _tap_switch(panel, True)
    assert prefs.get(lmp.GPU_PRESETS_OFFERED_KEY) is True
    assert panel.remove_preset(len(DEFAULT_NAMES))
    assert panel.offer_gpu_presets_once() == [] and len(_names(prefs)) == len(DEFAULT_NAMES) + len(GPU_NAMES) - 1
