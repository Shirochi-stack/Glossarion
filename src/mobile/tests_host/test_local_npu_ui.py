"""NPU in the three on-device places (Model manager › On-device, the Welcome page, the model picker): no extra NPU
rows anywhere (owner 2026-11-18: the existing slots use the NPU under Auto), the existing Custom sheets' and the
Sigma quant sheet's NPU marks only where the phone's NPU check passed (a pick stays an ordinary Auto row), the
preset editor's Backend and NPU layers, the engine card's "Run on" dropdown (its modes per phone, its line, the GPU
layers slider under CPU + GPU only, the GPU fallback line, the Q4_0 hint) and power mode, the row's backend line, its
fallback line and "Try the NPU again"; the three screens never wait for a slow NPU check or files download and show
its progress bar instead, then follow it; the picker's "Model manager › On-device" row leads with the Halgakos avatar;
the device checks' NPU card times every mode on a worker thread, shows the progress and stops on Stop.
Fakes only (FakeEngine, FakePrefs, a fake NPU check, the fake Hugging Face, a fake release host).
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
from glossarion_mobile.ui.screens import local_models_panel as lmp  # noqa: E402
from test_devfix7_custom_models import _by_key, _Page, _texts  # noqa: E402
from test_local_llm_server import make_service  # noqa: E402
from test_local_npu import Probe, chat, failing_state, npu_service, set_backend  # noqa: E402
from test_local_npu import _no_abort_note  # noqa: E402,F401 - the autouse fixture
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from test_bootstrap import _fake_session, _shutdown, _spike_main, app_env, storage  # noqa: E402,F401 - fixtures

GIB = 1024 ** 3
REPO = "unsloth/Qwen3.5-2B-GGUF"
NINE = "unsloth/Qwen3.5-9B-GGUF"
TREE = [("Qwen3.5-2B-Q4_K_M.gguf", 3000, "a" * 64), ("Qwen3.5-2B-Q4_0.gguf", 2000, "b" * 64),
        ("Qwen3.5-2B-IQ4_XS.gguf", 1500, "c" * 64), ("Qwen3.5-2B-Q8_0.gguf", 5000, "d" * 64),
        ("mmproj-F16.gguf", 500, "f" * 64)]


@pytest.fixture
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    server.trees[REPO] = TREE
    server.trees[NINE] = [("Qwen3.5-9B-Q4_0.gguf", 5000, "e" * 64)]
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


def _no_npu_rows(texts: list, keys) -> None:
    """None of the removed extra rows: no "· NPU" preset, no "Custom · NPU", no NPU section line or keys."""
    assert not any("· NPU" in str(t) for t in texts) and "Custom · NPU" not in texts
    assert not any("npu-preset" in k or "custom-npu" in k or "ondevice-npu" in k or k == "local-npu-section"
                   for k in keys)
    assert not any("not available" in str(t) for t in texts)                 # no "not available" rows


# ---- the three places: the same rows with and without the NPU ------------------------------------------------------

def test_model_manager_has_no_extra_npu_rows(tmp_path):
    passed, failed, plain = _services(tmp_path)
    for service, offered in ((passed, True), (failed, False), (plain, False)):
        panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
        panel.build()
        _no_npu_rows(_texts(panel.preset_column), _by_key(panel.preset_column))
        assert lmp.NPU_SECTION_TEXT not in _texts(panel.preset_column)
        assert len(panel.preset_column.controls) == 8                          # 4 defaults + 4 Custom families
        assert _by_key(panel.control)["local-npu-box"].visible is offered     # the engine card's NPU settings stay


def test_model_picker_has_no_extra_npu_rows(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    passed, failed, plain = _services(tmp_path)
    for service in (passed, failed, plain):
        sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"],
                              env=ms.SheetEnv(prefs=service.prefs, local_engine=service), tab="ondevice")
        texts = [t for row in sheet.ondevice_list.controls for t in _texts(row)]
        keys = {k for row in sheet.ondevice_list.controls for k in _by_key(row)}
        _no_npu_rows(texts, keys)
        assert "Presets · 8" in texts


def test_welcome_has_no_extra_npu_cells(tmp_path):
    from test_welcome_drawer_compact import by_key, screen

    passed, failed, plain = _services(tmp_path)
    for service in (passed, failed, plain):
        s = screen(local_engine=lambda svc_=service: svc_, prefs=service.prefs)
        rows = by_key(s.page_area.content)
        _no_npu_rows(_texts(rows["welcome-ondevice"]), set(rows))
        assert len(rows["welcome-ondevice"].controls) == 8


def test_the_engine_card_follows_a_background_check_and_the_rows_stay(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, stems=())
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    before = _texts(panel.preset_column)
    assert not _by_key(panel.control)["local-npu-box"].visible
    service.run_npu_check()                                                    # notifies the listeners
    assert _by_key(panel.control)["local-npu-box"].visible
    panel.render(push=False)
    assert _texts(panel.preset_column) == before                               # no rows appear


# ---- the existing Custom sheets: NPU marks only with the NPU, same order, ordinary rows ---------------------------

def _open_qwen(service, size_repo=REPO):
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, page=_Page())
    panel.build()
    _by_key(panel.preset_column)["local-custom-action-qwen35"].on_click(None)
    sheet = panel.last_dialog
    sheet.size = next(s for s in presets.custom_family("qwen35")["sizes"] if s["repo"] == size_repo)
    asyncio.run(sheet.load())
    return sheet


def test_custom_sheet_marks_the_quants_only_with_the_npu(tmp_path, fake_hf):
    passed, failed, plain = _services(tmp_path)
    sheet = _open_qwen(passed)
    assert sheet.family is presets.custom_family("qwen35") and "Custom · Qwen 3.5" in _texts(sheet.dialog.content)
    keys = _by_key(sheet.quant_box)
    labels = [_texts(keys[f"custom-quant-{i}"])[0] for i in range(4)]
    assert labels == ["IQ4_XS", "Q4_0", "Q4_K_M", "Q8_0"]                     # main's order: smallest first
    assert _texts(keys["custom-quant-npu-0"]) == [lmp.NPU_PARTLY_CPU_TEXT]
    assert _texts(keys["custom-quant-npu-1"]) == [lmp.NPU_FRIENDLY_TEXT]
    assert _texts(keys["custom-quant-npu-2"]) == ["NPU-friendly"] and _texts(keys["custom-quant-npu-3"]) == ["NPU-friendly"]
    assert "custom-npu-size-note" not in keys
    sheet.size = next(s for s in presets.custom_family("qwen35")["sizes"] if s["repo"] == NINE)
    asyncio.run(sheet.load())
    assert "Qwen 3.5 9B: over the NPU's 3.5 GB: slower" in _texts(sheet.quant_box)
    for service in (failed, plain):                                            # exactly as on main
        sheet = _open_qwen(service)
        keys = _by_key(sheet.quant_box)
        assert [_texts(keys[f"custom-quant-{i}"])[0] for i in range(4)] == labels
        assert not any(k.startswith("custom-quant-npu") for k in keys)
        texts = _texts(sheet.quant_box)
        assert lmp.NPU_FRIENDLY_TEXT not in texts and lmp.NPU_PARTLY_CPU_TEXT not in texts
        sheet = _open_qwen(service, NINE)
        assert "custom-npu-size-note" not in _by_key(sheet.quant_box)


def test_a_custom_pick_with_the_npu_is_an_ordinary_auto_row_on_the_npu(tmp_path, fake_hf):
    fake_hf.trees.pop(REPO)
    fake_hf.add(REPO, {"Qwen3.5-2B-Q4_0.gguf": b"Q" * 4096})
    passed, _failed, _plain = _services(tmp_path)
    flow = lmp.OnDeviceFlow(service=passed, prefs=passed.prefs, confirm=lambda *a: True)

    async def scenario():
        sheet = flow.open_custom("qwen35", size=next(s for s in presets.CUSTOM_FAMILIES[0]["sizes"]
                                                     if s["repo"] == REPO))
        await sheet.load()
        tasks = []
        flow.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        sheet.pick(sheet.files()[0][0])
        await tasks[-1]

    asyncio.run(scenario())
    row = presets.load_presets(passed.prefs)[-1]
    assert row["name"] == "Qwen 3.5 2B · Q4_0" and row["note"] == "Custom · Qwen 3.5 2B · Q4_0"
    assert "backend" not in row and passed.engine.backends == ["npu"]          # Auto: on the NPU


# ---- the Sigma quant sheet ------------------------------------------------------------------------------------------

def test_sigma_quant_sheet_marks_the_quants_only_with_the_npu(tmp_path):
    files = [hf.HfFile("gguf/sft3-IQ4_XS.gguf", 3000), hf.HfFile(presets.SIGMA_Q8_FILE, 8000),
             hf.HfFile("gguf/sft3-Q4_K_M.gguf", 5000)]

    def subtitles(chooser):
        return [tile.subtitle.value for tile in chooser.group.content.controls]

    async def scenario():
        marked = lmp.FileChooser(presets.SIGMA_REPO, files, files[1], npu=True)
        plain = lmp.FileChooser(presets.SIGMA_REPO, files, files[1])
        other = lmp.FileChooser("u/m-GGUF", files, files[1], npu=True)          # only Sigma's sheet is marked
        return subtitles(marked), subtitles(plain), subtitles(other)

    marked, plain, other = asyncio.run(scenario())
    assert [s.rsplit(" · ", 1)[-1] for s in marked] == [lmp.NPU_PARTLY_CPU_TEXT, lmp.NPU_FRIENDLY_TEXT,
                                                         lmp.NPU_FRIENDLY_TEXT]
    assert marked[1].startswith(plain[1]) and "Recommended" in plain[1]       # the order and labels stay
    for texts in (plain, other):
        assert not any("NPU" in s for s in texts)

    # the flow passes the phone's NPU check to the sheet
    passed, failed, _plain = _services(tmp_path)
    for service, offered in ((passed, True), (failed, False)):
        flow = lmp.OnDeviceFlow(service=service, prefs=service.prefs, page=_Page())

        async def choose(flow=flow):
            task = asyncio.ensure_future(flow.choose(presets.SIGMA_REPO, files, files[1]))
            await asyncio.sleep(0)
            chooser = flow.last_dialog
            chooser.pick(None)
            await task
            return chooser

        assert asyncio.run(choose()).npu is offered


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

def _tensor() -> str:
    """The internal value of "NPU + GPU, tensor split" (whatever the service names it)."""
    return next(mode for mode in svc.RUN_MODES if svc.RUN_MODE_LABELS[mode] == "NPU + GPU, tensor split")


def _pick(panel, mode) -> str:
    """The user picks ``mode`` in "Run on": Flet sets the value, then runs on_select."""
    panel.run_mode.value = mode
    panel.run_mode.on_select(None)
    return panel.run_mode.value


ALL_LABELS = ["Auto (NPU only)", "NPU only", "NPU + GPU", "NPU + GPU, tensor split", "CPU + GPU", "GPU only", "CPU only"]


def test_engine_card_run_on_and_power_mode(tmp_path):
    """One "Run on" dropdown, the app's existing dropdown built like the NPU power mode: every mode this phone has
    (an NPU phone with the GPU backend: all seven, tensor split right after NPU + GPU), Auto naming what it runs,
    one line on the chosen mode. Choosing saves at once, says nothing and runs no native work."""
    passed, _failed, _plain = _services(tmp_path)
    said: list = []
    panel = lmp.LocalModelsPanel(service=passed, prefs=passed.prefs, notify=said.append)
    panel.build()
    keys = _by_key(panel.control)
    assert "local-npu" not in keys and "local-gpu" not in keys and "local-npu-switch-note" not in keys
    run_on, power = keys["local-run-mode"], keys["local-npu-power"]
    assert type(run_on) is type(power) is ft.Dropdown and run_on.dense is power.dense is True
    assert run_on.label == "Run on" and run_on.visible and run_on.value == "auto"
    assert [o.key for o in run_on.options] == ["auto", "npu", "npu_gpu", _tensor(), "cpu_gpu", "gpu", "cpu"]
    assert [o.text for o in run_on.options] == ALL_LABELS
    assert keys["local-run-mode-note"].value == "Uses the NPU on this phone."
    assert not keys["local-gpu-layers-box"].visible                            # the slider is CPU + GPU's alone
    loads = list(getattr(passed.engine, "loads", None) or [])                 # no engine made yet, or its loads
    for mode in ("npu", "npu_gpu", _tensor(), "gpu", "cpu", "cpu_gpu", "auto"):
        assert _pick(panel, mode) == mode and passed.prefs.get(svc.RUN_MODE_KEY) == mode
        assert passed.run_mode() == mode and panel.run_mode_note.value == passed.run_mode_note(mode)
        assert panel.gpu_layers_box.visible is (mode == "cpu_gpu")
    assert list(getattr(passed.engine, "loads", None) or []) == loads and said == []  # no native work, no snackbar
    assert passed.loaded_stem is None
    assert [o.text for o in run_on.options] == ALL_LABELS                      # the labels never change
    # a value this phone does not have is refused: the dropdown shows the saved one again
    assert _pick(panel, "bogus") == "auto" and passed.run_mode() == "auto"
    assert keys["local-npu-power"].value == "burst"
    assert [o.key for o in keys["local-npu-power"].options] == list(eng.NPU_POWER_MODES)
    panel.npu_power.value = "balanced"
    assert panel.save_npu_power() == "balanced" and passed.npu_power() == "balanced"
    # the NPU has not started in this process: the first NPU load uses the new mode, no restart needed
    assert panel.npu_power_note.value == "Burst is the fastest and may slow down when the phone gets hot."
    passed._npu_session(passed.npu_skel_dir())                                 # stage 2 ran (a first NPU load)
    panel.npu_power.value = "power_saver"
    panel.save_npu_power()
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


@pytest.mark.parametrize("label,split", [("NPU + GPU", "layer"), ("NPU + GPU, tensor split", "tensor")])
def test_row_and_engine_card_name_npu_and_gpu_together(tmp_path, label, split):
    # "Run on" NPU + GPU (either split): the existing row line and engine detail say so; no new rows, no slider
    service, engine, _probe = npu_service(tmp_path)
    set_backend(service, "m-Q4_0", "auto")
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/m-Q4_0")
    panel.build()
    service.run_npu_check()                                                    # the card follows the answer
    mode = next(o.key for o in panel.run_mode.options if o.text == label)
    assert _pick(panel, mode) == mode and not panel.gpu_layers_box.visible
    chat(service)
    assert engine.backends == [f"npu+gpu:{split}"]
    panel.render(push=False)
    index = len(presets.load_presets(service.prefs)) - 1
    keys = _by_key(panel.preset_column)
    assert keys[f"local-preset-backend-{index}"].value == "Runs on NPU + GPU"
    assert not any(item.content == "Try the NPU again" for item in keys[f"local-preset-menu-{index}"].items)
    panel.render_engine(service.snapshot(), push=False)
    assert f"Runs on NPU + GPU ({split} split): Hexagon v81 (HTP0) + Fake GPU (Test)" in panel.engine_detail.value
    assert not panel.npu_status.visible and panel.run_mode.value == mode


@pytest.mark.parametrize("label,split,other", [("NPU + GPU, tensor split", "tensor", "NPU + GPU"),
                                               ("NPU + GPU", "layer", "NPU + GPU, tensor split")])
def test_a_refused_npu_and_gpu_runs_on_the_npu_and_keeps_its_label(tmp_path, label, split, other):
    """Owner 2026-11-19: the tensor split is a "Run on" option of its own. When the load is refused (it aborts at load
    on the owner's phone until the engine fix), the model runs on the NPU alone; the option keeps its label (no "not
    available" wording) and the existing fallback line says why."""
    engine = eng.FakeEngine(split_error={split: "llama.cpp refused the split"})
    service, engine, _probe = npu_service(tmp_path, engine=engine)
    set_backend(service, "m-Q4_0", "auto")
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/m-Q4_0")
    panel.build()
    service.run_npu_check()                                                    # the card follows the answer
    mode = next(o.key for o in panel.run_mode.options if o.text == label)
    _pick(panel, mode)
    chat(service)
    assert engine.backends == [f"npu+gpu:{split}", "npu"]
    panel.render_engine(service.snapshot(), push=False)
    panel.render(push=False)
    assert panel.run_mode.value == mode and [o.text for o in panel.run_mode.options] == ALL_LABELS
    assert not any("not available" in o.text for o in panel.run_mode.options)
    assert panel.npu_status.visible
    assert "unavailable for m-Q4_0" in panel.npu_status.value and "llama.cpp refused the split" in panel.npu_status.value
    assert panel.npu_status.value.endswith("Using the NPU.")
    index = len(presets.load_presets(service.prefs)) - 1
    assert _by_key(panel.preset_column)[f"local-preset-backend-{index}"].value == "Runs on NPU · Hexagon v81"
    assert "Runs on NPU: Hexagon v81 (HTP0)" in panel.engine_detail.value
    # picking the other split loads it
    engine.split_error = {}
    other_mode = next(o.key for o in panel.run_mode.options if o.text == other)
    _pick(panel, other_mode)
    chat(service)
    assert engine.backends[-1] == f"npu+gpu:{'layer' if split == 'tensor' else 'tensor'}"


def test_auto_runs_on_the_npu_alone_and_an_old_npu_and_gpu_record_changes_nothing(tmp_path):
    # Auto on an NPU phone with the GPU backend: the NPU alone, no slider, and an NPU + GPU record left by the build
    # that crashed adds no line to the engine card or the row
    service, engine, _probe = npu_service(tmp_path)
    set_backend(service, "m-Q4_0", "auto")
    service._split_failed("m-Q4_0", svc.SPLIT_CRASH_REASON)
    service.run_npu_check()
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/m-Q4_0")
    panel.build()
    assert panel.run_mode.value == "auto" and panel.run_mode.options[0].text == "Auto (NPU only)"
    chat(service)
    assert engine.backends == ["npu"]
    panel.render(push=False)
    panel.render_engine(service.snapshot(), push=False)
    index = len(presets.load_presets(service.prefs)) - 1
    keys = _by_key(panel.preset_column)
    assert keys[f"local-preset-backend-{index}"].value == "Runs on NPU · Hexagon v81"
    assert not any(item.content == "Try the NPU again" for item in keys[f"local-preset-menu-{index}"].items)
    assert not panel.gpu_layers_box.visible and not panel.npu_status.visible


# ---- "Run on" per phone, the GPU line, the slider and the Q4_0 hint (the engine card's former GPU switch tests) ------

def test_run_on_is_hidden_where_only_auto_and_cpu_exist(tmp_path):
    """x86_64 and iOS 13 / 14 (no GPU backend, no NPU): no dropdown, no line, no slider on either screen; a saved
    GPU mode is refused."""
    from test_welcome_drawer_compact import by_key, screen

    service, _engine = make_service(tmp_path)
    assert service.gpu_backend is None and service.run_mode_options() == []
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    keys = _by_key(panel.control)
    assert not keys["local-run-mode"].visible and not keys["local-run-mode-note"].visible
    assert not keys["local-gpu-layers-box"].visible and not keys["local-gpu-swap"].visible
    assert service.save_run_mode("cpu_gpu") == "auto" and not service.use_gpu()
    assert "welcome-run-mode" not in by_key(screen(local_engine=lambda: service, prefs=service.prefs).page_area.content)


@pytest.mark.parametrize("backend", ["OpenCL", "Metal"])
def test_a_gpu_phone_without_the_npu_offers_the_gpu_modes(tmp_path, backend):
    """Android arm64 without the NPU, iOS 15+: Auto (CPU only), CPU + GPU, GPU only, CPU only."""
    service, _engine = make_service(tmp_path, gpu_backend=backend)
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    assert [o.key for o in panel.run_mode.options] == ["auto", "cpu_gpu", "gpu", "cpu"]
    assert [o.text for o in panel.run_mode.options] == ["Auto (CPU only)", "CPU + GPU", "GPU only", "CPU only"]
    assert panel.run_mode.value == "auto" and panel.run_mode_note.value == "Uses the CPU on this phone."


def test_engine_card_run_on_gpu_line_and_device(tmp_path):
    prefs = fakes.FakePrefs()
    engine = eng.FakeEngine(gpu_error="no supported GPU: llama.cpp's OpenCL backend runs on Qualcomm Adreno GPUs only")
    service, engine = make_service(tmp_path / "android", prefs=prefs, engine=engine, gpu_backend="OpenCL")
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert panel.run_mode.key == "local-run-mode" and not panel.gpu_status.visible
    assert _pick(panel, "cpu_gpu") == "cpu_gpu" and service.use_gpu()
    assert "Q4_0" in panel.run_mode_note.value
    assert presets.load_presets(prefs) == list(presets.DEFAULT_PRESETS)       # a mode adds no presets
    # a GPU failure shows on the card, the device line says CPU
    chat(service, stem="fake-echo")
    panel.render_engine(service.snapshot(), push=False)
    assert panel.gpu_status.visible
    assert panel.gpu_status.value.startswith("GPU unavailable for fake-echo: no supported GPU")
    assert "Runs on CPU" in panel.engine_detail.value
    # a new choice gives the GPU a new try: the line goes
    _pick(panel, "gpu")
    panel.render_engine(service.snapshot(), push=False)
    assert not panel.gpu_status.visible
    # iOS (Metal), GPU only saved before
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "gpu"})
    service, _engine = make_service(tmp_path / "ios", prefs=prefs, gpu_backend="Metal")
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert panel.run_mode.visible and panel.run_mode.value == "gpu" and not panel.gpu_layers_box.visible


def test_gpu_slider_maps_cpu_partial_and_all(tmp_path, monkeypatch):
    service, _engine = make_service(tmp_path, prefs=fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu_gpu"}),
                                    gpu_backend="OpenCL")
    monkeypatch.setattr(service, "gpu_layer_limit", lambda stem=None: 24)
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    try:
        assert panel.gpu_layers_box.visible and panel.gpu_layers_slider.max == 24
        assert panel.gpu_layers_slider.value == 8 and not panel.gpu_layers_slider.disabled
        for position, expected in ((16, 16), (0, 0), (24, -1)):
            panel.gpu_layers_slider.value = position
            panel.gpu_layers_slider.on_change(None)
            panel.gpu_layers_slider.on_change_end(None)
            assert service.gpu_layers() == expected
        assert panel.gpu_layers_label.value == "GPU layers: 24 / 24"
        assert panel.gpu_layers_note.value.endswith("Changes apply to the next model load.")
        for mode in ("gpu", "auto", "cpu"):                                    # the slider is CPU + GPU's alone
            _pick(panel, mode)
            assert not panel.gpu_layers_box.visible
        _pick(panel, "cpu_gpu")
        assert panel.gpu_layers_box.visible and service.gpu_layers() == -1     # the setting is kept
    finally:
        panel.dispose()


def test_gpu_hint_offers_the_q4_0_file_of_a_k_quant_model(tmp_path):
    """owner 2026-10-22: a K-quant model in use on the Adreno GPU may be slower than the CPU: the engine card (and the
    Welcome page) say so while "Run on" puts layers on the GPU. Owner 2026-10-31: the tap opens the Custom menu at
    that family and size with Q4_0 highlighted."""
    light = next(p for p in presets.DEFAULT_PRESETS if p["name"] == "Light")
    q4_file = "Qwen3.5-2B-Q4_0.gguf"
    assert presets.is_gpu_quant(q4_file)
    assert presets.file_quant(light["file"]) == "Q4_K_XL" and not presets.is_gpu_quant(light["file"])

    def downloaded(service, stem, file):
        with open(service.registry.file_path(stem), "wb") as handle:  # the registry lists files on disk
            handle.write(b"GGUF")
        service.registry.add(stem, {"repo": light["repo"], "file": file, "revision": "r", "sha256": "0" * 64,
                                    "size": 4, "downloaded_at": 1})

    def panel_for(sub, backend="OpenCL", mode="cpu_gpu", file=light["file"]):
        prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: mode} if mode else {})
        service, _engine = make_service(tmp_path / sub, stems=(), prefs=prefs, gpu_backend=backend)
        downloaded(service, "light", file)
        current = {"model": "local/light"}
        panel = lmp.LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: current["model"])
        panel.build()
        return panel, prefs, service, current

    panel, prefs, service, current = panel_for("android")
    assert panel.gpu_swap_box.visible
    assert panel.gpu_swap_text.value.startswith("Light is Q4_K_XL, which can be slow on this GPU")
    assert panel.gpu_swap_button.content == "Choose Q4_0 · Qwen 3.5 2B"
    # the tap: the Custom menu at that family and size, Q4_0 highlighted; nothing joins the list yet
    spawned = []
    panel.spawn_fn = lambda coro: (spawned.append(coro), coro.close())
    sheet = panel.open_gpu_swap()
    assert sheet.family["id"] == "qwen35" and sheet.size["label"] == "2B" and sheet.highlight_gpu and len(spawned) == 1
    assert presets.load_presets(prefs) == list(presets.DEFAULT_PRESETS)
    # the Q4_0 model in use: the hint goes
    downloaded(service, "light-q4", q4_file)
    current["model"] = "local/light-q4"
    panel._render_gpu_swap(push=False)
    assert not panel.gpu_swap_box.visible
    # GPU only shows it too; choosing CPU only hides it at once
    panel, _prefs, _service, _current = panel_for("gpu-only", mode="gpu")
    assert panel.gpu_swap_box.visible
    _pick(panel, "cpu")
    assert not panel.gpu_swap_box.visible
    # no hint: Auto (CPU only here), a Q4_0 model, iOS (Metal runs the K-quants), a non-local model
    assert not panel_for("auto", mode=None)[0].gpu_swap_box.visible
    assert not panel_for("q4", file=q4_file)[0].gpu_swap_box.visible
    assert not panel_for("ios", backend="Metal")[0].gpu_swap_box.visible
    panel, _prefs, _service, current = panel_for("remote")
    current["model"] = "authgpt/gpt-6-luna"
    panel._render_gpu_swap(push=False)
    assert not panel.gpu_swap_box.visible


# ---- no delay, a progress bar instead (B1-B3) ------------------------------------------------------------------------

def _wait(condition, seconds=8.0):
    import time

    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert condition()


def _three_screens(service):
    """Build Model manager › On-device, the Welcome page and the picker's On-device tab: (seconds, panel keys,
    Welcome keys, picker texts)."""
    import time

    from glossarion_mobile.ui.sheets import model_sheet as ms
    from test_welcome_drawer_compact import by_key, screen

    started = time.monotonic()
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs)
    panel.build()
    welcome = screen(local_engine=lambda: service, prefs=service.prefs)
    sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"],
                          env=ms.SheetEnv(prefs=service.prefs, local_engine=service), tab="ondevice")
    seconds = time.monotonic() - started
    return seconds, panel, welcome, sheet, lambda: by_key(welcome.page_area.content), \
        lambda: [t for row in sheet.ondevice_list.controls for t in _texts(row)]


def test_the_screens_never_wait_for_a_slow_check_and_follow_it(tmp_path):
    from test_local_npu import Pre

    service, _engine, probe = npu_service(tmp_path, stems=(), precheck=Pre(delay=3.0))
    service.start()
    try:
        _wait(lambda: service.npu_progress() is not None)                        # stage 1 is running (and slow)
        seconds, panel, welcome, sheet, welcome_keys, picker_texts = _three_screens(service)
        assert seconds < 1.5 and service._npu_state is None                    # well under the 3 s check
        keys = _by_key(panel.control)
        assert keys["local-npu-box"].visible and keys["local-npu-progress"].visible
        assert keys["local-npu-progress"].value is None                       # indeterminate
        assert keys["local-npu-line"].value == "Checking the NPU…" and not keys["local-npu-power"].visible
        assert [o.text for o in keys["local-run-mode"].options] == ["Auto (CPU only)", "CPU + GPU", "GPU only",
                                                                    "CPU only"]
        assert _texts(welcome_keys()["welcome-npu-busy"]) == ["Checking the NPU…"]
        assert "Checking the NPU…" in picker_texts()
        # the check's end re-renders the screens after npu_progress() turns None (``_notify`` in its finally): wait for
        # the screens themselves, not only the service
        _wait(lambda: service._npu_state is not None and service.npu_progress() is None
              and _by_key(panel.control)["local-npu-line"].value == "Hexagon v81 (HTP) ready"
              and "welcome-npu-busy" not in welcome_keys() and "Checking the NPU…" not in picker_texts())
        keys = _by_key(panel.control)
        assert keys["local-npu-box"].visible and not keys["local-npu-progress"].visible
        assert keys["local-npu-power"].visible
        assert keys["local-npu-line"].value == "Hexagon v81 (HTP) ready" and probe.calls == []
        # the check's answer added the NPU modes to "Run on" on both screens, and Auto names the NPU
        assert [o.text for o in keys["local-run-mode"].options] == ALL_LABELS
        assert keys["local-run-mode"].value == "auto"
        _wait(lambda: [o.text for o in welcome_keys()["welcome-run-mode"].options] == ALL_LABELS)
        assert "welcome-npu-busy" not in welcome_keys() and "Checking the NPU…" not in picker_texts()
    finally:
        service.stop()
        from glossarion_mobile.services.local_llm import service as svc

        svc.set_default_service(None)


def test_a_failing_slow_check_hides_the_section_without_a_line(tmp_path):
    from test_local_npu import Pre

    service, _engine, _probe = npu_service(tmp_path, stems=(), precheck=Pre(failing_state("libcdsprpc", "nope"),
                                                                             delay=1.5))
    service.start()
    try:
        _wait(lambda: service.npu_progress() is not None)
        _seconds, panel, _welcome, _sheet, _wk, _pt = _three_screens(service)
        assert _by_key(panel.control)["local-npu-box"].visible
        _wait(lambda: service._npu_state is not None and service.npu_progress() is None
              and not _by_key(panel.control)["local-npu-box"].visible)
        _no_npu_rows(_texts(panel.control), set())                             # no "not available" text
    finally:
        service.stop()
        from glossarion_mobile.services.local_llm import service as svc

        svc.set_default_service(None)


def test_the_screens_show_a_slow_download_and_the_start(tmp_path):
    import threading

    from test_local_npu import FakeHTTP, Probe

    service, engine, _probe = npu_service(tmp_path, files=False, http=FakeHTTP(chunk_delay=0.35),
                                          probe=Probe(delay=1.0))
    service.run_npu_check()
    worker = threading.Thread(target=lambda: chat(service), daemon=True)
    worker.start()
    _wait(lambda: (service.npu_progress() or {}).get("phase") == "downloading")
    seconds, panel, welcome, sheet, welcome_keys, picker_texts = _three_screens(service)
    assert seconds < 1.5 and worker.is_alive()                                 # well under the ~3 s download
    keys = _by_key(panel.control)
    assert keys["local-npu-progress"].visible and keys["local-npu-progress"].value is not None  # determinate
    assert keys["local-npu-line"].value.startswith("Getting NPU files · ") and keys["local-npu-line"].value.endswith(" MB")
    assert panel.engine_state.value.startswith("Loading… · m-Q4_0 · Getting NPU files · ")  # the model's own state
    assert _texts(welcome_keys()["welcome-npu-busy"])[0].startswith("Getting NPU files · ")
    assert any(str(t).startswith("Getting NPU files · ") for t in picker_texts())
    _wait(lambda: service.npu_progress() is not None and service.npu_progress()["phase"] == "starting")
    keys = _by_key(panel.control)
    assert keys["local-npu-line"].value == "Starting the NPU…" and keys["local-npu-progress"].value is None
    assert panel.engine_state.value == "Loading… · m-Q4_0 · Starting the NPU…"
    worker.join(10)
    assert engine.backends == ["npu"]
    keys = _by_key(panel.control)
    assert not keys["local-npu-progress"].visible and keys["local-npu-line"].value == "Hexagon v81 (HTP0) ready"
    assert "welcome-npu-busy" not in welcome_keys()


def test_a_failed_download_line_and_try_again_on_the_row(tmp_path):
    import requests

    from test_local_npu import FakeHTTP

    service, engine, _probe = npu_service(tmp_path, files=False, http=FakeHTTP(error=requests.Timeout("slow")))
    set_backend(service, "m-Q4_0", "auto")
    panel = lmp.LocalModelsPanel(service=service, prefs=service.prefs, current_model=lambda: "local/m-Q4_0")
    panel.build()
    chat(service)
    panel.render(push=False)
    line = "NPU files not downloaded: Download failed after 4 tries (Timeout)"
    assert _by_key(panel.control)["local-npu-line"].value == line
    index = len(presets.load_presets(service.prefs)) - 1
    keys = _by_key(panel.preset_column)
    assert keys[f"local-preset-backend-{index}"].value == line
    service._npu_http = FakeHTTP()
    retry = next(i for i in keys[f"local-preset-menu-{index}"].items if i.content == "Try the NPU again")
    retry.on_click(None)
    chat(service)
    assert engine.backends == ["cpu", "npu"]


# ---- the picker's "Model manager › On-device" row: the Halgakos avatar (C) --------------------------------------------

def test_the_ondevice_manager_row_leads_with_the_halgakos_avatar(tmp_path):
    from glossarion_mobile.ui.components.empty_state import HALGAKOS_AVATAR
    from glossarion_mobile.ui.sheets import model_sheet as ms

    for service in _services(tmp_path)[::2]:
        sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"],
                              env=ms.SheetEnv(prefs=service.prefs, local_engine=service), tab="ondevice")
        row = next(r for r in sheet.ondevice_list.controls if getattr(r, "key", None) == "ondevice-manager")
        assert isinstance(row.leading, ft.CircleAvatar) and row.leading.foreground_image_src == HALGAKOS_AVATAR
        assert row.leading.radius * 2 == ms.ONDEVICE_MANAGER_AVATAR == 24          # a ListTile icon's size
        assert row.title.value == ms.ONDEVICE_MANAGER_LABEL
    sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"], env=ms.SheetEnv(prefs=fakes.FakePrefs()))
    assert isinstance(sheet._ondevice_manager_row().leading, ft.CircleAvatar)  # the no-engine row too


# ---- Logs & diagnostics › device checks › NPU (Snapdragon): timing every mode -----------------------------------------

def test_device_checks_npu_card_times_every_mode_follows_and_stops(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.services.local_llm import selfcheck
    from glossarion_mobile.ui import spike

    service, engine, _probe = npu_service(tmp_path / "svc", engine=eng.FakeEngine(delay=0.02))
    service.run_npu_check()
    monkeypatch.setattr(service, "gpu_layer_limit", lambda stem=None: 24)     # NPU + CPU's layer split needs it
    monkeypatch.setattr(spike.SpikeApp, "_engine_service", staticmethod(lambda: service))
    labels =[selfcheck.SPEED_LABELS[mode] for mode in selfcheck.speed_modes(service)]
    count = len(labels)
    assert count == len(selfcheck.SPEED_MODES)                                 # this phone has every mode

    async def until(predicate, timeout=10.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            assert asyncio.get_running_loop().time() < deadline, "timed out"
            await asyncio.sleep(0.01)

    async def scenario():
        conn, session = _fake_session("windows")
        page = session.page
        await _spike_main()(page)
        await session.after_event(page)
        app = page.data
        try:
            card = app.cards["npu"]
            buttons = card.control.content.content.controls[2].controls
            assert [b.content for b in buttons] == ["Refresh", "Time every mode", "Stop"]
            await app.npu_stop()                                               # nothing runs: nothing happens
            seen: list = []
            original = card.set
            monkeypatch.setattr(card, "set", lambda status, text=None: (seen.append((status, text)),
                                                                         original(status, text)))
            await app.npu_time()
            text = card.result.value
            assert card.status == "pass", text
            for label in labels:
                assert f"  {label}: prompt " in text, text
            running = [t for s, t in seen if s == "running" and t]
            assert running[0] == "Timing local/m-Q4_0…"
            assert [t.splitlines()[0] for t in running[1:]] == [
                f"Timing local/m-Q4_0: {label} ({i} of {count})…" for i, label in enumerate(labels, 1)]
            # Stop during a slow answer: the card says so, the timing ends at its next stop check
            engine.delay = 0.05
            engine.reply = lambda messages: " ".join(["word"] * 60)
            task = asyncio.ensure_future(app.npu_time())
            await until(lambda: f"(1 of {count})" in (card.result.value or ""))
            await app.npu_time()                                               # a second press while it runs: ignored
            await app.npu_stop()
            assert card.result.value.endswith("Stopping after the current step…")
            await task
            assert card.status == "pass" and f"  {labels[1]}: stopped" in card.result.value
            assert f"  {labels[-1]}: stopped" in card.result.value and not app._npu_timing
            assert service.prefs.get(svc.SPEED_KEY)["running"] is None
        finally:
            await _shutdown(app)

    asyncio.run(scenario())
