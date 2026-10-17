"""Model manager › On-device (``ui/screens/local_models_panel.py``, the ``local`` tab of
``ModelManagerScreen``): the tab, the one-tap Download (fake Hugging Face + FakeEngine: download, load,
``use_model``), progress rows with Pause / Resume / Cancel / Delete, Add from Hugging Face, the preset
editor, the big-download question, the engine-unavailable state and the 360 / 320 dp layout.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_models_panel.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
from pathlib import Path

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

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_local", Path(__file__).with_name("test_bootstrap.py"))
_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_local", Path(__file__).with_name("test_models_keys.py"))

PAYLOAD = os.urandom(2 * 1024 * 1024 + 777)
REPO = "tester/Tiny-GGUF"
FILE = "Tiny-Q4_K_M.gguf"


def _helpers():
    tb = importlib.util.module_from_spec(_TB_SPEC)
    _TB_SPEC.loader.exec_module(tb)
    mk = importlib.util.module_from_spec(_MK_SPEC)
    _MK_SPEC.loader.exec_module(mk)
    return tb, mk


def _texts(control) -> list:
    out = []

    def walk(c):
        if c is None:
            return
        value = getattr(c, "value", None)
        if isinstance(value, str) and type(c).__name__ == "Text":
            out.append(value)
        content = getattr(c, "content", None)
        if isinstance(content, str):
            out.append(content)
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                for item in child:
                    walk(item)
            elif child is not None and not isinstance(child, str):
                walk(child)

    walk(control)
    return out


def _find(control, key):
    stack = [control]
    while stack:
        c = stack.pop()
        if getattr(c, "key", None) == key:
            return c
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)
    return None


@pytest.fixture()
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    server.add(REPO, {FILE: PAYLOAD, "Tiny-BF16.gguf": PAYLOAD[:5000], "mmproj-F16.gguf": b"p"})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    yield server
    server.close()


def _tiny_presets():
    return [{"name": "Tiny", "repo": REPO, "file": FILE, "note": "test model", "n_ctx": 0, "size": len(PAYLOAD),
             "sha256": hashlib.sha256(PAYLOAD).hexdigest()},
            {"name": "Auto", "repo": REPO, "file": "", "quant": "", "note": "", "n_ctx": 0}]


def _engine_service(tmp_path, prefs, engine=None, **kwargs):
    engine = engine or eng.FakeEngine()
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, memory=lambda: (0, None), **kwargs)
    return service, engine


def test_on_device_tab_one_tap_download_load_and_use(tmp_path, fake_hf):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel, LocalPresetEditor
    from glossarion_mobile.ui.screens.model_manager import TABS, ModelManagerScreen

    tb, mk = _helpers()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        store = mk._store(tmp_path, {"model": "gpt-6"})
        catalog = mk._service(store, mk.FakeOptions(catalog=["gpt-6"]))
        catalog.load_blocking()
        prefs = fakes.FakePrefs({presets.PREF_KEY: _tiny_presets()})
        engine_service, engine = _engine_service(tmp_path, prefs)
        notes, used = [], []

        def use(model):
            used.append(model)
            store.set("model", model)

        screen = ModelManagerScreen(parse_route("/settings/models"), catalog=catalog, page=page,
                                    notify=lambda m, a=None, cb=None: notes.append(m),
                                    spawn=lambda c: asyncio.ensure_future(c), prefs=prefs, use_model=use,
                                    local_engine=engine_service, current_model=lambda: store.get("model", ""))
        page.views[0].controls.append(screen.get_body())
        screen.did_show()
        page.update()
        assert dict(TABS)["local"] == "On-device" and [s.value for s in screen.tabs.segments][-1] == "local"
        assert screen.local is None  # built when first shown
        screen.set_tab("local")
        panel = screen.local
        assert isinstance(panel, LocalModelsPanel) and screen.local_box.visible and screen.add_fab.content == "Add preset"
        assert not screen.rows.control.visible and not screen.prefix_list.visible
        texts = _texts(panel.preset_column)
        assert "Tiny" in texts and f"Not downloaded · {dl.format_bytes(len(PAYLOAD))}" in texts
        assert _find(panel.preset_column, "local-preset-action-0").content == "Download"
        asked = []
        panel.confirm_fn = lambda title, body, label: asked.append(title) or True  # every download asks first

        # ONE tap: download (progress), load, use as the model
        states = []
        panel.downloads.subscribe(lambda s: states.append((s.phase, _find(panel.preset_column, "local-preset-action-0").content)))
        model = await panel.get_preset(0)
        assert model == "local/Tiny-Q4_K_M" and used == ["local/Tiny-Q4_K_M"]
        assert ("downloading", "Cancel download") in states  # the undo sits where Download was tapped
        assert asked == ["Download Tiny?"]
        assert engine_service.loaded_stem == "Tiny-Q4_K_M" and engine.loads
        assert "local/Tiny-Q4_K_M" in catalog.snapshot.models
        assert _find(panel.preset_column, "local-preset-action-0").content == "In use"
        assert _find(panel.preset_column, "local-preset-status-0").value == "Loaded · in use"
        # the repository has a vision projector: it came in the same tap and the model reads images
        assert engine_service.is_vision("Tiny-Q4_K_M") and engine.mmproj.endswith("Tiny-Q4_K_M.mmproj.gguf")
        assert _find(panel.preset_column, "local-preset-vision-0") is not None
        assert panel.engine_state.value == "Ready · Tiny-Q4_K_M · reads images" and not panel.unload_button.disabled
        assert any(n.startswith("Model: local/Tiny-Q4_K_M (Tiny)") for n in notes)

        # the file-less preset of the same repo auto-picks Q4_K_M: already there, so it is just used
        assert panel.stem_of_preset(panel.presets()[1]) == "Tiny-Q4_K_M"

        # unload from the Engine card
        assert await panel.unload() and engine_service.loaded_stem is None
        assert panel.engine_state.value == "Not loaded"
        assert _find(panel.preset_column, "local-preset-status-0").value == "In use · loads on the first request"

        # editor: rename keeps the verified file; a bad repo is refused; the FAB opens a new preset
        editor = panel.open_editor(0)
        assert isinstance(editor, LocalPresetEditor) and editor.repo.value == REPO and editor.file.value == FILE
        editor.name.value = "Renamed"
        assert editor.save() is None and presets.load_presets(prefs)[0]["name"] == "Renamed"
        assert presets.load_presets(prefs)[0]["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
        new = screen.open_add()
        assert isinstance(new, LocalPresetEditor)
        new.name.value, new.repo.value = "Bad", "not a repo"
        assert new.save() and new.error.visible
        new.repo.value = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF"
        assert new.save() is None and presets.load_presets(prefs)[-1]["repo"] == "unsloth/Qwen3.5-2B-GGUF"

        # delete the file: the row goes back to Download
        dialog = panel.confirm_delete("Tiny-Q4_K_M")
        await dialog._on_confirm()
        assert engine_service.registry.load() == {}
        assert _find(panel.preset_column, "local-preset-action-0").content == "Download"

        # restore defaults
        dialog = panel.confirm_restore()
        await dialog._on_confirm()
        assert [p["name"] for p in presets.load_presets(prefs)] == ["Garbage", "Light", "Medium", "Heavy"]
        screen.set_tab("models")
        assert not screen.local_box.visible and screen.rows.control.visible
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_pause_resume_cancel_rows(tmp_path, fake_hf):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: _tiny_presets()[:1]})
        service, _engine = _engine_service(tmp_path, prefs)
        service.downloads.throttle = 0
        used = []
        panel = LocalModelsPanel(service=service, prefs=prefs, use_model=used.append, confirm=lambda *a: True)
        panel.build()
        fake_hf.chunk_delay = 0.01
        seen = []

        def on_state(state):
            seen.append(state.status)
            if state.running and state.phase == "downloading" and state.bytes_done > 0:
                assert _find(panel.preset_column, "local-preset-pause-0") is not None
                assert _find(panel.preset_column, "local-preset-progress-0").value == state.fraction
                panel.pause(state.stem)

        unsub = service.downloads.subscribe(on_state)
        assert await panel.get_preset(0) is None and used == []
        unsub()
        assert _find(panel.preset_column, "local-preset-action-0").content == "Resume"
        assert _find(panel.preset_column, "local-preset-status-0").value.startswith("Paused · ")
        assert _find(panel.preset_column, "local-preset-cancel-0") is not None
        assert any(s.endswith("left") or "MB" in s for s in seen)
        fake_hf.chunk_delay = 0
        assert await panel.get_preset_resume(0) == "local/Tiny-Q4_K_M" and used == ["local/Tiny-Q4_K_M"]
        assert service.registry.find(REPO, FILE) == "Tiny-Q4_K_M"
        # a second download (other repo name) cancelled mid-way leaves nothing
        fake_hf.add("tester/Other-GGUF", {FILE: PAYLOAD})
        fake_hf.chunk_delay = 0.01

        def cancel_soon(state):
            if state.running and state.phase == "downloading" and state.bytes_done > 0:
                panel.cancel(state.stem)

        service.downloads.subscribe(cancel_soon)
        panel.hf_field.value = "tester/Other-GGUF"
        assert await panel.download_typed() is None
        assert not [n for n in os.listdir(service.registry.root) if ".part" in n]
        panel.dispose()

    asyncio.run(scenario())


def test_add_from_hugging_face_saves_a_preset_and_asks_for_big_files(tmp_path, fake_hf):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: []})
        service, _engine = _engine_service(tmp_path, prefs)
        asked, used = [], []
        answer = {"value": False}

        def confirm(title, body, label):
            asked.append((title, body, label))
            return answer["value"]

        panel = LocalModelsPanel(service=service, prefs=prefs, use_model=used.append, confirm=confirm,
                                 choose=lambda repo, files: files[0])
        panel.build()
        panel.hf_field.value = "not a ref"
        assert await panel.download_typed() is None and "is not a Hugging Face repo id" in panel.hf_field.error
        panel.hf_field.value = f"https://huggingface.co/{REPO}"
        # every download asks first, also a small one (a mis-tap never starts one); No downloads nothing
        assert await panel.download_typed() is None and len(asked) == 1
        assert asked[0][0] == "Download model?" and asked[0][1].startswith("Download ") and "pause or cancel" in asked[0][1]
        assert "Wi-Fi" not in asked[0][1]  # the mobile-data note is for large downloads only
        assert not any(f"/{REPO}/resolve/" in p for p, _r in fake_hf.requests)
        answer["value"] = True
        assert await panel.download_typed() == "Tiny-Q4_K_M" and len(asked) == 2
        answer["value"] = False
        assert used == ["local/Tiny-Q4_K_M"] and panel.hf_status.value == "Downloaded local/Tiny-Q4_K_M ✓"
        assert presets.load_presets(prefs)[0]["name"] == "Tiny-Q4_K_M" and presets.load_presets(prefs)[0]["file"] == FILE
        # a 2 GB listing asks first; No downloads nothing
        fake_hf.trees["big/Big-GGUF"] = [("Big-Q4_K_M.gguf", 2_000_000_000, "a" * 64)]
        panel.hf_field.value = "big/Big-GGUF"
        assert await panel.download_typed() is None
        assert asked and asked[-1][1].startswith("Download 2.0 GB from Hugging Face?") and "big/Big-GGUF" in asked[-1][1]
        assert "Wi-Fi" in asked[-1][1]
        assert not any("/big/Big-GGUF/resolve/" in p for p, _r in fake_hf.requests)
        # no Q4_K_M and several files: the chooser picks
        fake_hf.add("odd/Odd-GGUF", {"Odd-F16.gguf": b"x" * 10, "Odd-Q3_K_S.gguf": b"y" * 12})
        panel.hf_field.value = "odd/Odd-GGUF"
        answer["value"] = True
        assert await panel.download_typed() == "Odd-F16"
        # a gated / missing repo says why
        panel.hf_field.value = "nobody/Missing-GGUF"
        assert await panel.download_typed() is None and "gated or private" in panel.hf_status.value

    asyncio.run(scenario())


def test_engine_unavailable_is_said_not_crashed(tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    monkeypatch.setattr(eng.importlib.util, "find_spec", lambda name: None)
    prefs = fakes.FakePrefs()
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "m")), watch=False)
    panel = LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert panel.unavailable_text.visible and "llama-cpp-python is not installed" in panel.unavailable_text.value
    assert panel.engine_state.value == "Not loaded" and panel.unload_button.disabled
    assert [r.key for r in panel.preset_column.controls] == [f"local-preset-{i}" for i in range(4)]


@pytest.mark.parametrize("width", [360, 320])
def test_layout_fits_small_phones(tmp_path, width):
    """Preset rows: the text column expands; the actions wrap onto their own row; no fixed width wider
    than the phone; the four tabs scroll instead of overflowing."""
    import flet as ft

    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel
    from glossarion_mobile.ui.screens import model_manager as mm

    prefs = fakes.FakePrefs()
    service, _engine = _engine_service(tmp_path, prefs)
    panel = LocalModelsPanel(service=service, prefs=prefs)
    control = panel.build()
    row = panel.preset_column.controls[0].content
    texts = row.controls[0].controls[0]
    assert isinstance(texts, ft.Column) and texts.expand
    actions = row.controls[-1]
    assert isinstance(actions, ft.Row) and actions.wrap
    for text in texts.controls:
        if isinstance(text, ft.Row):  # the "Reads images" mark: an icon and one ellipsized line
            assert text.key.startswith("local-preset-vision-")
            text = text.controls[-1]
        assert text.max_lines and text.overflow == ft.TextOverflow.ELLIPSIS
    stack = [control]
    while stack:
        c = stack.pop()
        fixed = getattr(c, "width", None)
        assert fixed is None or fixed <= width - 40, (type(c).__name__, fixed)
        for attr in ("content", "controls"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif child is not None and not isinstance(child, str) and hasattr(child, "__dict__"):
                stack.append(child)
    source = Path(mm.__file__).read_text(encoding="utf-8")
    assert "ft.Row([self.tabs], scroll=ft.ScrollMode.AUTO)" in source and "show_selected_icon=False" in source


def test_model_sheet_offers_on_device_models(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    opened = []
    env = ms.SheetEnv(open_local_models=lambda: opened.append(1))
    assert env.open_local_models is not None
    source = Path(ms.__file__).read_text(encoding="utf-8")
    assert '"Get on-device models…"' in source and 'key="model-get-local"' in source


def test_a_text_only_download_gets_image_support_from_the_menu(tmp_path, fake_hf):
    """A model downloaded before the projector existed (or whose projector failed) reads text only; the
    row's ⋮ menu offers "Add image support", which downloads the repository's mmproj and marks the row."""
    import flet as ft

    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: _tiny_presets()[:1]})
        service, engine = _engine_service(tmp_path, prefs)
        registry = service.registry
        os.makedirs(registry.root, exist_ok=True)
        with open(registry.file_path("Tiny-Q4_K_M"), "wb") as handle:
            handle.write(PAYLOAD)
        registry.add("Tiny-Q4_K_M", {"repo": REPO, "file": FILE, "revision": "r", "size": len(PAYLOAD),
                                     "sha256": hashlib.sha256(PAYLOAD).hexdigest(), "downloaded_at": 1})
        notes = []
        panel = LocalModelsPanel(service=service, prefs=prefs, notify=notes.append, confirm=lambda *a: True)
        panel.build()
        menu = _find(panel.preset_column, "local-preset-menu-0")
        assert "Add image support" in [i.content for i in menu.items]
        assert _find(panel.preset_column, "local-preset-vision-0") is None
        assert await panel.add_vision(0)
        assert service.is_vision("Tiny-Q4_K_M") and notes[-1] == "local/Tiny-Q4_K_M reads images now"
        assert _find(panel.preset_column, "local-preset-vision-0") is not None
        menu = _find(panel.preset_column, "local-preset-menu-0")
        assert "Add image support" not in [i.content for i in menu.items]
        assert isinstance(_find(panel.preset_column, "local-preset-vision-0"), ft.Row)
        panel.dispose()

    asyncio.run(scenario())
