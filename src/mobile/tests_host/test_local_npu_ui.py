"""NPU in the three on-device places (Model manager › On-device, the Welcome page, the model picker): no extra NPU
rows anywhere (owner 2026-11-18: the existing slots use the NPU under Auto), the existing Custom sheets' and the
Sigma quant sheet's NPU marks only where the phone's NPU check passed (a pick stays an ordinary Auto row), the
preset editor's Backend and NPU layers, the engine card's "Use NPU" and power mode, the row's backend line, its
fallback line and "Try the NPU again"; the three screens never wait for a slow NPU check or files download and show
its progress bar instead, then follow it; the picker's "Model manager › On-device" row leads with the Halgakos avatar.
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
        assert keys["local-npu-line"].value == "Checking the NPU…" and not keys["local-npu"].visible
        assert _texts(welcome_keys()["welcome-npu-busy"]) == ["Checking the NPU…"]
        assert "Checking the NPU…" in picker_texts()
        # the check's end re-renders the screens after npu_progress() turns None (``_notify`` in its finally): wait for
        # the screens themselves, not only the service
        _wait(lambda: service._npu_state is not None and service.npu_progress() is None
              and _by_key(panel.control)["local-npu-line"].value == "Hexagon v81 (HTP) ready"
              and "welcome-npu-busy" not in welcome_keys() and "Checking the NPU…" not in picker_texts())
        keys = _by_key(panel.control)
        assert keys["local-npu-box"].visible and not keys["local-npu-progress"].visible and keys["local-npu"].visible
        assert keys["local-npu-line"].value == "Hexagon v81 (HTP) ready" and probe.calls == []
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
    line = "NPU files not downloaded: Download failed (Timeout)"
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
