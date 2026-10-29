"""The model picker's On-device tab (``ui/sheets/model_sheet.py`` over Model manager › On-device's
``OnDeviceFlow``): the tab next to Model (also in field mode), the preset rows and every other downloaded
``local/`` model with size, "Reads images", RAM hint and one state button; Download asks first, shows
Cancel while it downloads and then loads and picks ``local/<stem>``; Use on a downloaded Hugging Face
file picks it; the link to Model manager › On-device; the 360-412 dp layout.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_model_sheet_ondevice.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.services.local_llm import service as svc


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

PAYLOAD = os.urandom(2 * 1024 * 1024 + 333)
REPO = "tester/Tiny-GGUF"
FILE = "Tiny-Q4_K_M.gguf"
GIB = 1024 ** 3


def _walk(control):
    stack = [control]
    while stack:
        c = stack.pop()
        yield c
        for attr in ("content", "controls", "title", "subtitle", "trailing", "leading"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)


def _find(rows, key):
    for row in rows:
        for c in _walk(row):
            if getattr(c, "key", None) == key:
                return c
    return None


def _texts(rows) -> list:
    out = []
    for row in rows:
        for c in _walk(row):
            if type(c).__name__ == "Text" and isinstance(getattr(c, "value", None), str):
                out.append(c.value)
    return out


@pytest.fixture()
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    server.add(REPO, {FILE: PAYLOAD, "mmproj-F16.gguf": b"p" * 100})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    yield server
    server.close()


def _presets():
    return [{"name": "Tiny", "repo": REPO, "file": FILE, "note": "test model", "n_ctx": 0, "size": len(PAYLOAD),
             "sha256": hashlib.sha256(PAYLOAD).hexdigest(), "mmproj": "mmproj-F16.gguf"},
            {"name": "Huge", "repo": "big/Huge-GGUF", "file": "Huge-Q4_K_M.gguf", "note": "", "n_ctx": 0,
             "size": 40 * GIB}]


def _service(tmp_path, prefs):
    engine = eng.FakeEngine()
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, memory=lambda: (8 * GIB, 6 * GIB))
    return service, engine


def _add_file(service, stem, repo, file, *, payload=PAYLOAD, mmproj=False):
    registry = service.registry
    os.makedirs(registry.root, exist_ok=True)
    with open(registry.file_path(stem), "wb") as handle:
        handle.write(payload)
    entry = {"repo": repo, "file": file, "revision": "r", "size": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest(), "downloaded_at": 1}
    registry.add(stem, entry)
    if mmproj:  # its vision projector (<stem>.mmproj.gguf), recorded on the model's entry
        with open(registry.file_path(registry.projector_key(stem)), "wb") as handle:
            handle.write(b"p" * 100)
        registry.attach_projector(stem, {"file": "mmproj-F16.gguf", "revision": "r", "size": 100,
                                         "sha256": "0" * 64, "downloaded_at": 1})


def _sheet(service, prefs, *, current="gpt-6", opened=None, **kwargs):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    selected = []
    env = ms.SheetEnv(prefs=prefs, local_engine=service,
                      open_local_models=(lambda: opened.append(1)) if opened is not None else None)
    sheet = ms.ModelSheet(current_model=current, models=["gpt-6"], env=env,
                          on_select=lambda *a: selected.append(a), **kwargs)
    return sheet, selected


def test_the_on_device_tab_sits_next_to_model_also_in_field_mode(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
    service, _engine = _service(tmp_path, prefs)
    sheet, _sel = _sheet(service, prefs)
    assert [s.value for s in sheet.tabs.segments] == ["model", "ondevice", "profile", "language"]
    assert sheet.tabs.segments[1].label == "On-device" and sheet.tabs.visible and sheet.tabs_row.visible
    # four segments fit 360 dp without the selected tick; a narrower phone scrolls them sideways
    assert sheet.tabs.show_selected_icon is False and sheet.tabs_row.scroll == ms.ft.ScrollMode.AUTO
    field, _sel = _sheet(service, prefs, field_mode=True)
    assert [s.value for s in field.tabs.segments] == ["model", "ondevice"] and field.tabs.visible
    field_ondevice, _sel = _sheet(service, prefs, field_mode=True, tab="ondevice")
    assert field_ondevice.tab == "ondevice"
    # a callable (the chat integration's lazy getter) works too
    lazy = ms.ModelSheet(current_model="gpt-6", env=ms.SheetEnv(prefs=prefs, local_engine=lambda: service))
    assert "ondevice" in lazy.tab_ids
    # without the engine: the U3 / desktop-less sheet is unchanged
    plain = ms.ModelSheet(current_model="gpt-6", env=ms.SheetEnv())
    assert [s.value for s in plain.tabs.segments] == ["model", "profile", "language"]
    assert plain.tabs.show_selected_icon is True
    plain_field = ms.ModelSheet(current_model="gpt-6", env=ms.SheetEnv(), field_mode=True, tab="ondevice")
    assert not plain_field.tabs.visible and not plain_field.tabs_row.visible and plain_field.tab == "model"


def test_on_device_rows_list_presets_and_added_files_with_size_vision_ram_and_state(tmp_path):
    prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
    service, _engine = _service(tmp_path, prefs)
    # a model added from Hugging Face in Model manager › On-device (no preset names it), with image support
    _add_file(service, "Extra-Q4_K_M", "someone/Extra-GGUF", "Extra-Q4_K_M.gguf", payload=b"e" * 4096, mmproj=True)
    opened = []
    sheet, selected = _sheet(service, prefs, opened=opened)
    sheet.tab = "ondevice"
    sheet.refresh()
    rows = sheet.ondevice_list.controls
    # the model tab's search, chips and route rows are not shown on this tab
    assert not sheet.search.visible and not sheet.chip_row.visible and not sheet.route_row.visible
    assert not sheet.chat_switch.visible
    texts = _texts(rows)
    assert "Presets · 2" in texts and "Downloaded · 1" in texts
    assert _find(rows, "ondevice-preset-0-facts").value == f"{dl.format_bytes(len(PAYLOAD))} · ✓ fits this phone"
    assert _find(rows, "ondevice-preset-0-vision") is not None  # the preset names its projector
    assert _find(rows, "ondevice-preset-0-action").content == "Download"
    huge = _find(rows, "ondevice-preset-1-facts")
    assert huge.value.endswith("✕ too big for this phone") and huge.color == "error"
    assert _find(rows, "ondevice-preset-1-vision") is None
    assert "local/Extra-Q4_K_M" in texts and _find(rows, "ondevice-file-Extra-Q4_K_M-vision") is not None
    assert "someone/Extra-GGUF" in _find(rows, "ondevice-file-Extra-Q4_K_M-facts").value
    assert _find(rows, "ondevice-file-Extra-Q4_K_M-action").content == "Use"
    # the link closes the picker and opens Model manager › On-device
    link = _find(rows, "ondevice-manager")
    assert link.title.value == "Model manager › On-device"
    link.on_click(None)
    assert opened == [1] and sheet.closed and selected == []


def test_use_on_a_downloaded_hugging_face_file_picks_local_stem(tmp_path):
    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
        service, engine = _service(tmp_path, prefs)
        _add_file(service, "Extra-Q4_K_M", "someone/Extra-GGUF", "Extra-Q4_K_M.gguf", payload=b"e" * 4096)
        sheet, selected = _sheet(service, prefs, tab="ondevice")
        task = _find(sheet.ondevice_list.controls, "ondevice-file-Extra-Q4_K_M-action").on_click(None)
        assert await task is True
        assert selected == [("model", "local/Extra-Q4_K_M", False)] and sheet.closed
        assert service.loaded_stem == "Extra-Q4_K_M" and engine.loads
        assert prefs.get("model_recents")[0] == "local/Extra-Q4_K_M"
        # opened again with it as the model: In use
        again, _sel = _sheet(service, prefs, current="local/Extra-Q4_K_M", tab="ondevice")
        action = _find(again.ondevice_list.controls, "ondevice-file-Extra-Q4_K_M-action")
        assert action.content == "In use" and action.disabled
        assert _find(again.ondevice_list.controls, "ondevice-file-Extra-Q4_K_M-status").value == "Loaded · in use"
        service.stop()

    asyncio.run(scenario())


def test_download_asks_first_shows_cancel_then_loads_and_selects(tmp_path, fake_hf):
    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
        service, engine = _service(tmp_path, prefs)
        service.downloads.throttle = 0
        sheet, selected = _sheet(service, prefs, tab="ondevice")
        flow = sheet.ondevice_flow()
        answer = {"value": False}
        asked = []
        flow.confirm_fn = lambda title, body, label: asked.append((title, label)) or answer["value"]
        # No: nothing downloads, nothing is picked, the sheet stays open
        task = _find(sheet.ondevice_list.controls, "ondevice-preset-0-action").on_click(None)
        assert await task is None and asked == [("Download Tiny?", "Download")]
        assert selected == [] and not sheet.closed and not any("/resolve/" in p for p, _r in fake_hf.requests)
        assert _find(sheet.ondevice_list.controls, "ondevice-preset-0-action").content == "Download"
        # Yes: while it downloads the button is Cancel (the sheet repaints from the download events)
        answer["value"] = True
        seen = []
        unsub = service.downloads.subscribe(
            lambda s: seen.append(_find(sheet.ondevice_list.controls, "ondevice-preset-0-action").content))
        task = _find(sheet.ondevice_list.controls, "ondevice-preset-0-action").on_click(None)
        assert await task == "local/Tiny-Q4_K_M"
        unsub()
        assert "Cancel" in seen
        assert selected == [("model", "local/Tiny-Q4_K_M", False)] and sheet.closed
        assert service.loaded_stem == "Tiny-Q4_K_M" and engine.loads and not service.is_vision("Tiny-Q4_K_M")
        # closed: the sheet no longer listens to downloads or the engine
        assert not sheet._ondevice_unsubs
        reopened, _sel = _sheet(service, prefs, current="local/Tiny-Q4_K_M", tab="ondevice")
        rows = reopened.ondevice_list.controls
        assert _find(rows, "ondevice-preset-0-action").content == "In use"
        assert _find(rows, "ondevice-preset-0-status").value == "Loaded · in use"
        assert _find(rows, "ondevice-preset-0-vision") is not None
        assert "Downloaded · 1" not in _texts(rows)  # the preset's own file is not listed twice
        service.stop()

    asyncio.run(scenario())


def test_cancel_while_downloading_stops_it_and_picks_nothing(tmp_path, fake_hf):
    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
        service, _engine = _service(tmp_path, prefs)
        service.downloads.throttle = 0
        fake_hf.chunk_delay = 0.01
        sheet, selected = _sheet(service, prefs, tab="ondevice")
        sheet.ondevice_flow().confirm_fn = lambda *a: True
        tapped = []

        def tap_cancel(state):
            if state.running and state.phase == "downloading" and state.bytes_done > 0 and not tapped:
                button = _find(sheet.ondevice_list.controls, "ondevice-preset-0-action")
                assert button.content == "Cancel"
                assert _find(sheet.ondevice_list.controls, "ondevice-preset-0-progress") is not None
                tapped.append(1)
                button.on_click(None)

        service.downloads.subscribe(tap_cancel)
        task = _find(sheet.ondevice_list.controls, "ondevice-preset-0-action").on_click(None)
        assert await task is None and tapped
        assert selected == [] and not sheet.closed
        assert not [n for n in os.listdir(service.registry.root) if ".part" in n]
        assert _find(sheet.ondevice_list.controls, "ondevice-preset-0-action").content == "Download"
        sheet.close()
        service.stop()

    asyncio.run(scenario())


def test_no_engine_message_and_render_only_on_the_open_on_device_tab(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
    service, _engine = _service(tmp_path, prefs)
    sheet, _sel = _sheet(service, prefs)
    assert sheet.ondevice is None  # nothing on-device is built until the tab shows
    sheet.tab = "ondevice"
    sheet.refresh()
    assert sheet.ondevice is not None and sheet._ondevice_unsubs
    rows = sheet.ondevice_list.controls
    # owner 2026-10-27: the On-device rows at full height (a Column, no scrolling inside the list)
    assert rows and sheet.ondevice_list.visible and not sheet.list_view.visible and not sheet.list_view.controls
    sheet.tab = "model"
    sheet.refresh()
    model_rows = sheet.list_view.controls
    assert sheet.list_view.visible and not sheet.ondevice_list.visible and not sheet.ondevice_list.controls
    sheet.render_ondevice()  # a download event while the Model tab shows leaves its rows alone
    assert sheet.list_view.controls is model_rows and not sheet.ondevice_list.controls
    sheet._on_dismiss()
    assert not sheet._ondevice_unsubs and not service.downloads.listeners
    # the sheet without an engine (env.local_engine is None) never offers the tab; asked anyway it says why
    bare = ms.ModelSheet(current_model="gpt-6", env=ms.SheetEnv())
    assert _find(bare.ondevice_rows(), "ondevice-none") is not None


@pytest.mark.parametrize("width", [360, 412])
def test_on_device_rows_fit_phone_widths(tmp_path, width):
    import flet as ft

    prefs = fakes.FakePrefs({presets.PREF_KEY: _presets()})
    service, _engine = _service(tmp_path, prefs)
    _add_file(service, "A-very-long-model-file-name-from-hugging-face-Q4_K_M", "someone/Long-GGUF",
              "A-very-long-model-file-name-from-hugging-face-Q4_K_M.gguf", payload=b"x" * 2048, mmproj=True)
    sheet, _sel = _sheet(service, prefs, tab="ondevice")
    for row in sheet.ondevice_list.controls:
        for c in _walk(row):
            fixed = getattr(c, "width", None)
            assert fixed is None or fixed <= width - 40, (type(c).__name__, fixed)
            if isinstance(c, ft.Text) and c.key and str(c.key).startswith("ondevice-"):
                assert c.max_lines and c.overflow == ft.TextOverflow.ELLIPSIS, c.key
        if str(getattr(row, "key", "")).startswith(("ondevice-preset-", "ondevice-file-")):
            texts, buttons = row.content.controls
            assert isinstance(texts, ft.Column) and texts.expand  # the name column takes what the button leaves
            assert len(buttons.controls) <= 2
            title = texts.controls[0]
            assert title.max_lines == 1 and title.overflow == ft.TextOverflow.ELLIPSIS


def test_hotfix6_q4_0_presets_show_with_the_gpu_off_and_use_loads_when_ram_is_tight(tmp_path):
    """Owner 2026-10-27: on Android (OpenCL) the picker lists the GPU-friendly presets whatever "Use GPU" says,
    and Use on a model tight on free RAM loads it: the warning is the sheet's note, never a question."""
    async def scenario():
        prefs = fakes.FakePrefs()  # the default presets
        engine = eng.FakeEngine()
        memory = {"value": (8 * GIB, 6 * GIB)}
        service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                         engine_factory=lambda: engine, watch=False, platform="android",
                                         memory=lambda: memory["value"])
        service.gpu_backend = "OpenCL"
        _add_file(service, "Extra-Q4_K_M", "someone/Extra-GGUF", "Extra-Q4_K_M.gguf", payload=b"e" * 4096)
        sheet, selected = _sheet(service, prefs, tab="ondevice")
        assert not service.use_gpu()
        names = [p["name"] for p in presets.load_presets(prefs)]
        assert names == [p["name"] for p in presets.DEFAULT_PRESETS + presets.GPU_PRESETS]
        for index in range(len(names)):
            assert _find(sheet.ondevice_list.controls, f"ondevice-preset-{index}-action") is not None
        memory["value"] = (8 * GIB, 256 * 1024 ** 2)  # 256 MB free: tight
        assert service.ram_check("Extra-Q4_K_M")[0] == "confirm"
        task = _find(sheet.ondevice_list.controls, "ondevice-file-Extra-Q4_K_M-action").on_click(None)
        assert await task is True
        assert selected == [("model", "local/Extra-Q4_K_M", False)] and service.loaded_stem == "Extra-Q4_K_M"
        assert "Android may close Glossarion while this model runs." in sheet.refresh_note
        assert "Load anyway" not in sheet.refresh_note
        service.stop()

    asyncio.run(scenario())
