"""Owner 2026-11-19 (Galaxy Z Fold 8): "The tap to use button has no animated loading bar when tapped, so it looks
unresponsive while the model loads, or NPU downloads", and a failed NPU files download needed a restart.

From the tap on a downloaded on-device model until its load ends (success or failure), every place that offers it
(the Welcome cell, Model manager › On-device's preset and other-model rows, the model picker's preset and file rows)
shows one loading state (``OnDeviceFlow.loading_state`` / ``row_state`` "loading"): "Loading…" or the NPU's own line
over a thin bar, indeterminate while the model loads, the NPU files' fraction while they download. It shows before
the first await, goes on failure too, a second tap starts nothing, and a load started elsewhere shows as well. The
Welcome page re-renders for it (and not for nothing), gives the "Run on" dropdown room above, and after a failed NPU
files download says why with "Try the NPU again", which downloads at once.

Fakes only (FakeEngine, FakePrefs, the fake NPU stages and release host of test_local_npu); no network, no sleeps.
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
pytest.importorskip("msgpack")

from glossarion_mobile.services.local_llm import presets  # noqa: E402
from glossarion_mobile.ui.screens import local_models_panel as lmp  # noqa: E402
from test_local_npu import Q40, FakeHTTP, chat, npu_service, set_backend  # noqa: E402
from test_local_npu import _no_abort_note  # noqa: E402,F401 - the autouse fixture
from test_welcome_drawer_compact import by_key, screen  # noqa: E402

CELL = "welcome-ondevice-row-m-q4-0"


def _setup(tmp_path, **kwargs):
    """A phone with the NPU, ``m-Q4_0`` downloaded and a preset row for it (the last row)."""
    service, engine, probe = npu_service(tmp_path, memory=lambda: (8 * 1024 ** 3, 6 * 1024 ** 3), **kwargs)
    service.run_npu_check()
    set_backend(service, Q40, "auto")
    index = len(presets.load_presets(service.prefs)) - 1
    return service, engine, index


def _cell(welcome) -> tuple:
    """(detail line, bar value or "none") of the Welcome cell."""
    cell = by_key(welcome.page_area.content)[CELL]
    lines = cell.content.controls
    bar = lines[-1] if isinstance(lines[-1], ft.ProgressBar) else None
    return lines[1].value, ("none" if bar is None else bar.value)


class SlowLoad:
    """``service.load`` held until ``release``: records what ``probe()`` shows at the moment the load starts (before
    any await of the tap), then runs the real load or returns ``result``."""

    def __init__(self, service, probe, result=None):
        self.service = service
        self.probe = probe
        self.result = result
        self.calls: list = []
        self.gate = asyncio.Event()

    async def __call__(self, stem):
        self.calls.append(self.probe())
        await self.gate.wait()
        return self.result if self.result is not None else self.service.load_blocking(stem)


def test_welcome_cell_shows_loading_at_once_follows_the_npu_files_and_clears(tmp_path):
    service, engine, index = _setup(tmp_path)
    chosen: list = []
    welcome = screen(local_engine=lambda: service, prefs=service.prefs, on_set_model=chosen.append,
                     current_model=lambda: chosen[-1] if chosen else "")
    assert _cell(welcome) == ("Tap to use", "none")
    loader = SlowLoad(service, lambda: _cell(welcome))
    service.load = loader

    async def scenario():
        tap = asyncio.ensure_future(welcome.use_ondevice(index))
        await asyncio.sleep(0)
        # it showed before the tap's first await: "Loading…" over an indeterminate bar
        assert loader.calls == [("Loading…", None)] and _cell(welcome) == ("Loading…", None)
        assert welcome.ondevice.row_state(presets.load_presets(service.prefs)[index], Q40)[0] == "loading"
        # a second tap while it loads starts nothing
        assert await welcome.use_ondevice(index) is None and len(loader.calls) == 1
        # an engine notice that changes nothing on screen does not rebuild the row (nor its indeterminate bar)
        shown = welcome.ondevice_box.content
        service._notify()
        assert welcome.ondevice_box.content is shown
        # the NPU files download: its own line and the real fraction reach the cell
        service._npu_phase, service._npu_bytes = "downloading", (13_000_000, 26_000_000)
        service._notify()
        assert _cell(welcome) == ("Getting NPU files · 13 / 26 MB", 0.5)
        service._npu_bytes = (19_500_000, 26_000_000)
        service._notify()
        assert _cell(welcome) == ("Getting NPU files · 20 / 26 MB", 0.75)
        service._npu_retry_text = "Connection problem (ConnectionError); retrying in 4 s…"
        service._notify()
        assert _cell(welcome)[0] == "Connection problem (ConnectionError); retrying in 4 s…"
        service._npu_phase, service._npu_retry_text = "starting", None
        service._notify()
        assert _cell(welcome) == ("Starting the NPU…", None)
        service._npu_phase = None
        service._notify()
        assert _cell(welcome) == ("Loading…", None)
        loader.gate.set()
        return await tap

    assert asyncio.run(scenario()) == f"local/{Q40}"
    assert engine.backends == ["npu"] and _cell(welcome) == ("✓ In use", "none")
    welcome.dispose()


def test_welcome_cell_loading_clears_when_the_load_fails(tmp_path):
    service, engine, index = _setup(tmp_path)
    welcome = screen(local_engine=lambda: service, prefs=service.prefs)
    loader = SlowLoad(service, lambda: _cell(welcome), result=(False, "not enough memory"))
    service.load = loader

    async def scenario():
        tap = asyncio.ensure_future(welcome.use_ondevice(index))
        await asyncio.sleep(0)
        assert _cell(welcome) == ("Loading…", None)
        loader.gate.set()
        return await tap

    assert asyncio.run(scenario()) is None
    assert _cell(welcome) == ("Tap to use", "none") and engine.backends == []
    assert welcome.ondevice.loading_stems == set()
    assert by_key(welcome.page_area.content)["welcome-model-local"].value == "not enough memory"
    welcome.dispose()


def test_a_load_started_elsewhere_shows_on_welcome_and_in_the_manager(tmp_path):
    service, _engine, index = _setup(tmp_path)
    welcome = screen(local_engine=lambda: service, prefs=service.prefs)
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    service._set("loading", Q40)                                                  # a request's load
    assert service.loading_model() == Q40
    assert _cell(welcome) == ("Loading…", None)                                   # the notice re-rendered it
    keys = by_key(panel.preset_column)
    assert keys[f"local-preset-action-{index}"].content == "Loading…" and keys[f"local-preset-action-{index}"].disabled
    assert keys[f"local-preset-status-{index}"].value == "Loading…"
    assert keys[f"local-preset-progress-{index}"].value is None
    service._set("idle", None)
    assert _cell(welcome) == ("Tap to use", "none") and service.loading_model() is None
    panel.dispose()
    welcome.dispose()


def test_model_manager_rows_show_the_loading_state_from_the_tap(tmp_path):
    service, engine, index = _setup(tmp_path, stems=(Q40, "extra"))
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()

    def preset_row():
        keys = by_key(panel.preset_column)
        return (keys[f"local-preset-action-{index}"].content, keys[f"local-preset-status-{index}"].value,
                getattr(keys.get(f"local-preset-progress-{index}"), "value", "none"))

    def other_row():
        keys = by_key(panel.other_card)
        return (keys["local-other-use-extra"].content, keys["local-other-use-extra"].disabled,
                getattr(keys.get("local-other-progress-extra"), "value", "none"))

    assert preset_row()[0] == "Use" and other_row() == ("Use", False, "none")
    loader = SlowLoad(service, lambda: (preset_row(), other_row()))
    service.load = loader

    async def scenario():
        tap = asyncio.ensure_future(panel.get_preset(index))
        await asyncio.sleep(0)
        assert loader.calls[0][0] == ("Loading…", "Loading…", None)
        loader.gate.set()
        assert await tap == f"local/{Q40}"
        loader.gate = asyncio.Event()
        other = asyncio.ensure_future(panel._use_other("extra"))
        await asyncio.sleep(0)
        assert loader.calls[1][1] == ("Loading…", True, None)
        assert await panel._use_other("extra") is False and len(loader.calls) == 2  # a second tap: nothing
        loader.gate.set()
        return await other

    assert asyncio.run(scenario()) is True
    assert other_row() == ("Use", False, "none") and engine.backends[-1] in ("npu", "cpu")
    panel.dispose()


def test_model_picker_rows_show_the_loading_state_and_clear_on_failure(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    service, _engine, index = _setup(tmp_path, stems=(Q40, "extra"))
    sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"],
                          env=ms.SheetEnv(prefs=service.prefs, local_engine=service), tab="ondevice")
    key = f"ondevice-preset-{index}"

    def rows():
        keys = by_key(ft.Column(sheet.ondevice_list.controls))
        file_bar = keys.get("ondevice-file-extra-progress")
        return ((keys[f"{key}-action"].content, keys[f"{key}-status"].value,
                 getattr(keys.get(f"{key}-progress"), "value", "none")),
                (keys["ondevice-file-extra-action"].content, keys["ondevice-file-extra-status"].value,
                 "none" if file_bar is None else file_bar.value))

    assert rows()[0][0] == "Use" and rows()[1][0] == "Use"
    loader = SlowLoad(service, rows, result=(False, "boom"))
    service.load = loader
    flow = sheet.ondevice_flow()

    async def scenario():
        tap = asyncio.ensure_future(flow.get_preset(index))
        await asyncio.sleep(0)
        assert loader.calls[0][0] == ("Loading…", "Loading…", None)
        service._npu_phase, service._npu_bytes = "downloading", (6_500_000, 26_000_000)
        service._notify()
        assert rows()[0] == ("Loading…", "Getting NPU files · 6 / 26 MB", 0.25)
        service._npu_phase = None
        loader.gate.set()
        assert await tap is None
        assert rows()[0][0] == "Use" and rows()[0][2] == "none"                # cleared on the failure
        loader.gate = asyncio.Event()
        use = asyncio.ensure_future(sheet.use_downloaded("extra"))
        await asyncio.sleep(0)
        assert loader.calls[1][1] == ("Loading…", "Loading…", None)
        assert await sheet.use_downloaded("extra") is False and len(loader.calls) == 2
        loader.gate.set()
        return await use

    assert asyncio.run(scenario()) is False
    assert rows()[1] == ("Use", "On this device", "none")


def test_welcome_try_the_npu_again_after_a_failed_files_download(tmp_path):
    import requests

    chosen = [f"local/{Q40}"]
    service, engine, _index = _setup(tmp_path, files=False, http=FakeHTTP(error=requests.ConnectionError("offline")))
    welcome = screen(local_engine=lambda: service, prefs=service.prefs, on_set_model=chosen.append,
                     current_model=lambda: chosen[-1])
    assert "welcome-npu-files" not in by_key(welcome.page_area.content)
    chat(service)                                                                 # four tries, then the CPU
    assert engine.backends == ["cpu"] and service.slept == [2, 4, 8]
    keys = by_key(welcome.page_area.content)                                     # the failure's notice re-rendered
    line = "NPU files not downloaded: Download failed after 4 tries (ConnectionError)"
    assert keys["welcome-npu-files-note"].value == line
    retry = keys["welcome-npu-retry"]
    assert retry.content == "Try the NPU again"
    # a cloud model chosen while the local one stays loaded: no row (its reload would switch the model)
    chosen.append("gpt-6")
    welcome.render_ondevice()
    assert service.loaded_stem == Q40 and "welcome-npu-files" not in by_key(welcome.page_area.content)
    chosen.append(f"local/{Q40}")
    welcome.render_ondevice()
    assert "welcome-npu-retry" in by_key(welcome.page_area.content)
    service._npu_http = FakeHTTP()
    assert asyncio.run(welcome.retry_npu(Q40)) is True                           # downloads at once and loads
    assert engine.backends == ["cpu", "npu"] and service.npu_download_error is None
    assert "welcome-npu-files" not in by_key(welcome.page_area.content)
    assert not any("not available" in str(getattr(c, "value", "")) for c in by_key(welcome.page_area.content).values())
    welcome.dispose()


def test_welcome_run_on_dropdown_has_room_above_it(tmp_path):
    from glossarion_mobile.ui.screens import welcome as welcome_mod

    service, _engine, _index = _setup(tmp_path)
    welcome = screen(local_engine=lambda: service, prefs=service.prefs)
    keys = by_key(welcome.page_area.content)
    box = keys["welcome-run-mode-box"]
    assert box.content is keys["welcome-run-mode"] and box.padding.top == welcome_mod.RUN_MODE_GAP >= 8
    welcome.dispose()
    # the Model manager's own dropdown is unchanged (no box around it)
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    assert "local-run-mode-box" not in by_key(panel.control)
    panel.dispose()
