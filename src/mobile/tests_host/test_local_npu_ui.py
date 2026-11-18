"""NPU rows and settings in the three on-device places (Model manager › On-device, the Welcome page, the model
picker): the built-in NPU presets and "Custom · NPU" only where the phone's NPU check passed (nothing at all
elsewhere), a tap saving the row and running the one-tap flow, the Custom · NPU sheet's quant marks and order, the
preset editor's Backend and NPU layers, the engine card's "Use NPU" and power mode, the row's backend line, its
fallback line and "Try the NPU again". Fakes only (FakeEngine, FakePrefs, a fake NPU check, the fake Hugging Face).
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

import local_llm_fakes as fakes  # noqa: E402
from glossarion_mobile.services.local_llm import engine as eng  # noqa: E402
from glossarion_mobile.services.local_llm import hf, presets  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET  # noqa: E402
from glossarion_mobile.ui.screens import local_models_panel as lmp  # noqa: E402
from test_devfix7_custom_models import _by_key, _Page, _texts  # noqa: E402
from test_local_llm_server import make_service  # noqa: E402
from test_local_npu import Probe, chat, failing_state, npu_service, set_backend  # noqa: E402

GIB = 1024 ** 3
REPO = "unsloth/Qwen3.5-2B-GGUF"
TREE = [("Qwen3.5-2B-Q4_K_M.gguf", 3000, "a" * 64), ("Qwen3.5-2B-Q4_0.gguf", 2000, "b" * 64),
        ("Qwen3.5-2B-IQ4_XS.gguf", 1500, "c" * 64), ("Qwen3.5-2B-Q8_0.gguf", 5000, "d" * 64),
        ("mmproj-F16.gguf", 500, "f" * 64)]


@pytest.fixture
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    server.trees[REPO] = TREE
    yield server
    server.close()


def _services(tmp_path):
    """(a phone whose NPU check passed, one whose check failed, a build without the NPU)."""
    memory = lambda: (8 * GIB, 6 * GIB)  # noqa: E731
    passed, _e, _p = npu_service(tmp_path / "a", stems=(), memory=memory)
    failed, _e, _p = npu_service(tmp_path / "b", stems=(), memory=memory,
                                 probe=Probe(failing_state("libcdsprpc", "dlopen failed")))
    plain, _e = make_service(tmp_path / "c", stems=(), memory=memory)
    for service in (passed, failed):
        service.run_npu_check()
    return passed, failed, plain


NPU_NAMES = ["Garbage · NPU", "Light · NPU", "Medium · NPU", "Custom · NPU"]


# ---- the three places ----------------------------------------------------------------------------------------------

def test_model_manager_shows_the_npu_rows_only_with_the_npu(tmp_path):
    passed, failed, plain = _services(tmp_path)
    for service, shown in ((passed, True), (failed, False), (plain, False)):
        panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
        panel.build()
        texts = _texts(panel.preset_column)
        keys = _by_key(panel.preset_column)
        assert all((name in texts) is shown for name in NPU_NAMES)
        assert ("local-npu-section" in keys) is shown and ("local-custom-npu" in keys) is shown
        assert (lmp.NPU_SECTION_TEXT in texts) is shown
        assert not any("not available" in t for t in texts)                    # no "not available" rows
        card = _by_key(panel.control)
        assert card["local-npu-box"].visible is shown
        # 4 defaults + 4 Custom families; with the NPU: + the section line, 3 NPU rows, Custom · NPU
        assert len(panel.preset_column.controls) == (13 if shown else 8)


def test_model_picker_shows_the_npu_rows_only_with_the_npu(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    passed, failed, plain = _services(tmp_path)
    for service, shown in ((passed, True), (failed, False), (plain, False)):
        sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"],
                              env=ms.SheetEnv(prefs=service.prefs, local_engine=service), tab="ondevice")
        texts = [t for row in sheet.ondevice_list.controls for t in _texts(row)]
        assert all((name in texts) is shown for name in NPU_NAMES)
        assert (f"Presets · {12 if shown else 8}" in texts)


def test_welcome_shows_the_npu_cells_only_with_the_npu(tmp_path):
    from test_welcome_drawer_compact import by_key, screen

    passed, failed, plain = _services(tmp_path)
    for service, shown in ((passed, True), (failed, False), (plain, False)):
        s = screen(local_engine=lambda svc_=service: svc_, prefs=service.prefs)
        rows = by_key(s.page_area.content)
        keys = set(rows)
        assert ("welcome-ondevice-npu-0" in keys) is shown and ("welcome-ondevice-custom-npu" in keys) is shown
        assert len(rows["welcome-ondevice"].controls) == (12 if shown else 8)
        if shown:
            assert _texts(rows["welcome-ondevice-custom-npu"]) == ["NPU", "Custom"]
            assert _texts(rows["welcome-ondevice-npu-1"])[0] == "Light · NPU"


def test_the_rows_appear_when_the_background_check_finishes(tmp_path):
    from test_welcome_drawer_compact import by_key, screen

    service, _engine, _probe = npu_service(tmp_path, stems=())
    s = screen(local_engine=lambda: service, prefs=service.prefs)
    assert "welcome-ondevice-npu-0" not in by_key(s.page_area.content)
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    assert not _by_key(panel.control)["local-npu-box"].visible
    service.run_npu_check()                                                    # notifies the listeners
    assert "welcome-ondevice-npu-0" in by_key(s.page_area.content)
    assert _by_key(panel.control)["local-npu-box"].visible and "Light · NPU" in _texts(panel.preset_column)


def test_a_saved_row_of_the_same_file_stands_for_the_npu_row(tmp_path):
    passed, _failed, _plain = _services(tmp_path)
    flow = lmp.OnDeviceFlow(service=passed, prefs=passed.prefs)
    assert [i for i, _row in flow.npu_presets()] == [0, 1, 2]
    presets.add_row(passed.prefs, dict(presets.NPU_PRESETS[1]))
    assert [i for i, _row in flow.npu_presets()] == [0, 2]                    # shown once, as the saved row


# ---- a tap: saved, downloaded, loaded on the NPU ------------------------------------------------------------------

def test_tapping_an_npu_preset_saves_it_and_runs_it_on_the_npu(tmp_path, fake_hf):
    fake_hf.trees.pop(REPO)
    row = presets.NPU_PRESETS[1]
    fake_hf.add(REPO, {row["file"]: b"Q" * 4096, "mmproj-F16.gguf": b"M" * 64})
    passed, _failed, _plain = _services(tmp_path)
    used = []
    panel = lmp.LocalModelsPanel(service=passed, prefs=passed.prefs, use_model=used.append, confirm=lambda *a: True)
    panel.build()

    async def scenario():
        tasks = []
        panel.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        _by_key(panel.preset_column)["local-npu-action-1"].on_click(None)
        await tasks[0]

    asyncio.run(scenario())
    saved = presets.load_presets(passed.prefs)[-1]
    assert saved["name"] == "Light · NPU" and saved["backend"] == "npu" and saved["file"] == row["file"]
    assert used and used[-1].startswith("local/") and passed.engine.backends == ["npu"]
    assert passed.snapshot().backend == "NPU"
    texts = _texts(panel.preset_column)
    assert "Runs on NPU · Hexagon v81" in texts                               # the model row says what runs it
    assert texts.count("Light · NPU") == 1                                    # the saved row, not the built-in too


# ---- Custom · NPU ---------------------------------------------------------------------------------------------------

def test_custom_npu_sheet_marks_and_orders_the_quants(tmp_path, fake_hf):
    passed, _failed, _plain = _services(tmp_path)
    page = _Page()
    panel = lmp.LocalModelsPanel(service=passed, prefs=passed.prefs, page=page)
    panel.build()
    _by_key(panel.preset_column)["local-custom-action-npu"].on_click(None)
    sheet = panel.last_dialog
    assert sheet.backend == "npu" and sheet.family is presets.NPU_FAMILY
    assert "Custom · NPU" in _texts(sheet.dialog.content)
    two_b = next(s for s in presets.NPU_FAMILY["sizes"] if s["repo"] == REPO)
    sheet.size = two_b
    asyncio.run(sheet.load())
    keys = _by_key(sheet.quant_box)
    labels = [_texts(keys[f"custom-quant-{i}"])[0] for i in range(4)]
    assert labels == ["Q4_0", "Q4_K_M", "Q8_0", "IQ4_XS"]                     # accepted first, then the rest
    assert _texts(keys["custom-quant-npu-0"]) == [lmp.NPU_FRIENDLY_TEXT]
    assert _texts(keys["custom-quant-npu-1"]) == ["NPU-friendly"]
    assert _texts(keys["custom-quant-npu-3"]) == ["Partly CPU on NPU"]
    assert keys["custom-quant-art-0"].src == HALGAKOS_ASSET and "custom-quant-art-1" not in keys
    assert keys["custom-quant-0"].border is not None                          # Q4_0 highlighted
    nine = next(s for s in presets.NPU_FAMILY["sizes"] if s["repo"] == "unsloth/Qwen3.5-9B-GGUF")
    fake_hf.trees[nine["repo"]] = [("Qwen3.5-9B-Q4_0.gguf", 5000, "e" * 64)]
    sheet.size = nine
    asyncio.run(sheet.load())
    assert "Qwen 3.5 9B: over the NPU's 3.5 GB: slower" in _texts(sheet.quant_box)


def test_custom_npu_pick_becomes_an_npu_row(tmp_path, fake_hf):
    fake_hf.trees.pop(REPO)
    fake_hf.add(REPO, {"Qwen3.5-2B-Q4_0.gguf": b"Q" * 4096})
    passed, _failed, _plain = _services(tmp_path)
    flow = lmp.OnDeviceFlow(service=passed, prefs=passed.prefs, confirm=lambda *a: True)

    async def scenario():
        sheet = flow.open_npu_custom()
        sheet.size = next(s for s in presets.NPU_FAMILY["sizes"] if s["repo"] == REPO)
        await sheet.load()
        tasks = []
        flow.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        sheet.pick(sheet.files()[0][0])
        await tasks[-1]

    asyncio.run(scenario())
    row = presets.load_presets(passed.prefs)[-1]
    assert row["backend"] == "npu" and row["name"] == "Qwen 3.5 2B · Q4_0 · NPU"
    assert passed.engine.backends == ["npu"]


def test_ordinary_custom_sheet_is_unchanged_with_the_npu(tmp_path, fake_hf):
    passed, _failed, _plain = _services(tmp_path)
    flow = lmp.OnDeviceFlow(service=passed, prefs=passed.prefs)
    sheet = lmp.CustomModelSheet(flow, presets.custom_family("qwen35"),
                                 size=next(s for s in presets.CUSTOM_FAMILIES[0]["sizes"] if s["repo"] == REPO))
    asyncio.run(sheet.load())
    keys = _by_key(sheet.quant_box)
    assert sheet.backend == "auto" and not any(k.startswith("custom-quant-npu") for k in keys)


# ---- the editor -----------------------------------------------------------------------------------------------------

def test_the_editor_has_backend_and_npu_layers_only_with_the_npu(tmp_path):
    passed, _failed, plain = _services(tmp_path)
    presets.save_presets(passed.prefs, [{"name": "Mine", "repo": "r/m", "file": "m-Q4_0.gguf", "backend": "gpu",
                                         "npu_layers": 7}])
    panel = lmp.LocalModelsPanel(service=passed, prefs=passed.prefs)
    panel.build()
    editor = panel.open_editor(0)
    assert editor.backend.visible and editor.npu_layers.visible and editor.backend.value == "gpu"
    assert [o.key for o in editor.backend.options] == ["auto", "npu", "gpu", "cpu"]
    assert [o.text for o in editor.backend.options] == ["Auto", "NPU", "GPU", "CPU"]
    assert editor.npu_layers.value == "7"
    editor.backend.value, editor.npu_layers.value = "npu", "24"
    assert editor.save() is None
    row = presets.load_presets(passed.prefs)[0]
    assert row["backend"] == "npu" and row["npu_layers"] == 24
    editor = panel.open_editor(0)
    editor.npu_layers.value = "abc"
    assert "NPU layers" in editor.save()
    # without the NPU the fields are hidden and the row keeps its own backend
    presets.save_presets(plain.prefs, [{"name": "Mine", "repo": "r/m", "file": "m-Q4_0.gguf", "backend": "npu"}])
    panel = lmp.LocalModelsPanel(service=plain, prefs=plain.prefs)
    panel.build()
    editor = panel.open_editor(0)
    assert not editor.backend.visible and not editor.npu_layers.visible and "backend" not in editor.row()
    editor.note.value = "renamed note"
    assert editor.save() is None
    assert presets.load_presets(plain.prefs)[0]["backend"] == "npu"


# ---- engine card, backend line, Try the NPU again -------------------------------------------------------------------

def test_engine_card_use_npu_and_power_mode(tmp_path):
    passed, _failed, _plain = _services(tmp_path)
    panel = lmp.LocalModelsPanel(service=passed, prefs=passed.prefs)
    panel.build()
    keys = _by_key(panel.control)
    assert keys["local-npu"].value is True and keys["local-npu-power"].value == "burst"
    assert [o.key for o in keys["local-npu-power"].options] == list(eng.NPU_POWER_MODES)
    panel.npu_switch.value = False
    assert panel.save_npu() is False and passed.use_npu() is False
    panel.npu_power.value = "balanced"
    assert panel.save_npu_power() == "balanced" and passed.npu_power() == "balanced"
    assert panel.npu_power_note.value == "Applies after restarting Glossarion."


def test_row_backend_line_fallback_and_try_again(tmp_path):
    engine = eng.FakeEngine(npu_error="ggml-hex: failed to open session")
    service, engine, _probe = npu_service(tmp_path, engine=engine)
    set_backend(service, "m-Q4_0", "auto")
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/m-Q4_0")
    panel.build()
    chat(service)
    panel.render(push=False)
    index = len(presets.load_presets(service.prefs)) - 1
    keys = _by_key(panel.preset_column)
    line = keys[f"local-preset-backend-{index}"]
    assert line.value == "NPU unavailable for m-Q4_0: ggml-hex: failed to open session. Using CPU."
    menu = keys[f"local-preset-menu-{index}"]
    retry = next(item for item in menu.items if item.content == "Try the NPU again")
    engine.npu_error = None
    retry.on_click(None)
    assert service.npu_fallback("m-Q4_0") is None
    chat(service)
    panel.render(push=False)
    keys = _by_key(panel.preset_column)
    assert keys[f"local-preset-backend-{index}"].value == "Runs on NPU · Hexagon v81"
    assert not any(item.content == "Try the NPU again" for item in keys[f"local-preset-menu-{index}"].items)
    panel.render_engine(service.snapshot(), push=False)
    assert "Runs on NPU: Hexagon v81 (HTP0)" in panel.engine_detail.value
