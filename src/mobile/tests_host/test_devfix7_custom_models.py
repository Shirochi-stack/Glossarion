"""Owner 2026-10-31: "We need a custom quant and parameter area for our on device qwen 3.5 model, so replace the 4
gpu presets and do the same in the welcome screen ... clicking on custom should open a menu letting me select
parameter size + quant ... highlight gpu friendly quants with an uncropped halgakos icon for android users", and
"Allow the slider to hit 262,144 ... default it to 16,384 for all on device models in the welcome screen."

The Custom rows (``presets.CUSTOM_FAMILIES``) in the Model manager, the model picker and on the Welcome page, the
one ``CustomModelSheet`` they open, what a chosen quant becomes, what the former GPU preset rows leave behind, and
the Context slider's new end. A fake Hugging Face on loopback (``local_llm_fakes.FakeHF``), FakeEngine, FakePrefs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
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
from glossarion_mobile.services.local_llm import downloads as dl  # noqa: E402
from glossarion_mobile.services.local_llm import hf, presets  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET  # noqa: E402
from glossarion_mobile.ui.screens import local_models_panel as lmp  # noqa: E402
from test_local_llm_server import make_service  # noqa: E402

GIB = 1024 ** 3
REPO = "unsloth/Qwen3.5-2B-GGUF"
QWEN = presets.custom_family("qwen35")
TWO_B = next(size for size in QWEN["sizes"] if size["repo"] == REPO)
Q4_0 = "Qwen3.5-2B-Q4_0.gguf"
#: a repository's files as the fake lists them (path, size, sha256)
TREE = [
    ("Qwen3.5-2B-Q4_K_M.gguf", 3000, "a" * 64), (Q4_0, 2000, "b" * 64), ("Qwen3.5-2B-UD-Q4_K_XL.gguf", 4000, None),
    ("Qwen3.5-2B-BF16.gguf", 9000, "c" * 64), ("Qwen3.5-2B-Q8_0-00001-of-00002.gguf", 8000, "d" * 64),
    ("Qwen3.5-2B-Q8_0-00002-of-00002.gguf", 8000, "e" * 64), ("mmproj-F16.gguf", 500, "f" * 64),
    ("mmproj-F32.gguf", 900, "9" * 64),
]


@pytest.fixture
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    server.trees[REPO] = TREE
    yield server
    server.close()


def _flow(tmp_path, *, backend="OpenCL", memory=(8 * GIB, 6 * GIB), prefs=None, stems=(), confirm=True, current=None):
    prefs = prefs if prefs is not None else fakes.FakePrefs()
    service, _engine = make_service(tmp_path, stems=stems, prefs=prefs, gpu_backend=backend, memory=lambda: memory)
    used: list = []

    def use(model):
        used.append(model)

    answers = confirm if callable(confirm) else (lambda *_a: confirm)
    flow = lmp.OnDeviceFlow(service=service, prefs=prefs, use_model=use, confirm=answers,
                            current_model=current or (lambda: ""))
    return flow, prefs, service, used


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


def _by_key(root) -> dict:
    return {str(node.key): node for node in _walk(root) if getattr(node, "key", None) is not None}


def _texts(root) -> list:
    return [node.value for node in _walk(root) if isinstance(node, ft.Text)]


async def _listed(flow, *, size=TWO_B, **kwargs):
    sheet = lmp.CustomModelSheet(flow, QWEN, size=size, **kwargs)
    await sheet.load()
    return sheet


# ---- the table ----------------------------------------------------------------------------------------------


def test_the_table_has_the_four_families_and_real_looking_refs():
    assert [family["name"] for family in presets.CUSTOM_FAMILIES] == ["Qwen 3.5", "Llama", "Gemma", "LFM2"]
    assert [size["label"] for size in QWEN["sizes"]] == ["0.8B", "2B", "4B", "9B"]
    repos = [size["repo"] for family in presets.CUSTOM_FAMILIES for size in family["sizes"]]
    assert len(set(repos)) == len(repos)
    for repo in repos:
        assert hf.parse_ref(repo).repo == repo and repo.startswith("unsloth/")
    assert presets.custom_size_for("UNSLOTH/qwen3.5-2b-gguf") == (QWEN, TWO_B)
    assert presets.custom_size_for("me/other") is None and presets.custom_family("nope") is None
    # the defaults stay the first-run data
    assert [p["name"] for p in presets.DEFAULT_PRESETS] == ["Garbage", "Light", "Medium", "Heavy"]


@pytest.mark.parametrize("file,label,no_quant", [
    ("Qwen3.5-2B-Q4_0.gguf", "Q4_0", False), ("Qwen3.5-2B-UD-Q4_K_XL.gguf", "Q4_K_XL", False),
    ("Qwen3.5-2B-BF16.gguf", "No quant (BF16)", True), ("LFM2-350M-F16.gguf", "No quant (F16)", True),
    ("x-f32.gguf", "No quant (F32)", True), ("sub/odd-name.gguf", "odd-name.gguf", False),
])
def test_quant_labels(file, label, no_quant):
    assert presets.quant_label(file) == label and presets.is_no_quant(file) is no_quant


def test_gemma_4_assistant_heads_are_not_models(fake_hf):
    fake_hf.trees["unsloth/gemma-4-E2B-it-GGUF"] = [
        ("mtp-gemma-4-E2B-it.gguf", 98, None), ("MTP/mtp-gemma-4-E2B-it-F16.gguf", 170, None),
        ("gemma-4-E2B-it-Q4_0.gguf", 3000, None), ("mmproj-F16.gguf", 900, None)]
    listing = hf.list_repo(hf.HfRef("unsloth/gemma-4-E2B-it-GGUF"))
    assert [f.path for f in listing.files] == ["gemma-4-E2B-it-Q4_0.gguf"]
    assert [f.path for f in listing.projectors] == ["mmproj-F16.gguf"]


# ---- the menu ----------------------------------------------------------------------------------------------


def test_sizes_first_then_the_real_files_of_that_repository(tmp_path, fake_hf):
    flow, _prefs, _service, _used = _flow(tmp_path)

    async def scenario():
        spawned = []
        flow.spawn_fn = lambda coro: spawned.append(asyncio.ensure_future(coro))
        sheet = lmp.CustomModelSheet(flow, QWEN)
        keys = _by_key(sheet.dialog.content)
        assert "custom-pick-size" in keys and "custom-quant-0" not in keys and fake_hf.requests == []  # no request yet
        assert [keys[f"custom-size-{i}"].content for i in range(4)] == ["0.8B", "2B", "4B", "9B"]
        assert keys["custom-sizes"].wrap is True  # a 320 dp phone wraps them
        keys["custom-size-1"].on_click(None)
        assert isinstance(sheet.size_row.controls[1], ft.FilledButton)
        assert "custom-loading" in _by_key(sheet.dialog.content)
        await spawned[-1]
        texts = _texts(sheet.quant_box)
        # smallest first; the split set and the projectors are not offered; "No quant" in plain words
        labels = [t for t in texts if t in ("Q4_0", "Q4_K_M", "Q4_K_XL", "No quant (BF16)")]
        assert labels == ["Q4_0", "Q4_K_M", "Q4_K_XL", "No quant (BF16)"]
        assert not any("00001" in t or "mmproj" in t or "Q8_0" in t for t in texts)
        assert "Reads images" in texts and "Not quantized: full precision, the largest file" in texts
        requests = len(fake_hf.requests)
        # another size and back: the second visit asks nothing
        sheet.select_size(QWEN["sizes"][0])  # not on the fake: the failure shows, then back to the listed one
        await spawned[-1]
        assert "custom-error" in _by_key(sheet.quant_box)
        sheet.select_size(TWO_B)
        assert "custom-quant-0" in _by_key(sheet.quant_box)
        assert len(fake_hf.requests) == requests + 1

    asyncio.run(scenario())


def test_no_unquantized_file_means_no_no_quant_row(tmp_path, fake_hf):
    fake_hf.trees[REPO] = [row for row in TREE if "BF16" not in row[0]]
    flow, *_ = _flow(tmp_path)
    sheet = asyncio.run(_listed(flow))
    assert not any(str(t).startswith("No quant") for t in _texts(sheet.quant_box))


def test_rows_carry_the_size_and_the_ram_hint_and_refuse_what_cannot_load(tmp_path, fake_hf):
    fake_hf.trees[REPO] = [("Qwen3.5-2B-Q4_0.gguf", 1 * GIB, None), ("Qwen3.5-2B-Q8_0.gguf", 4 * GIB, None),
                           ("Qwen3.5-2B-BF16.gguf", 20 * GIB, None)]
    flow, *_ = _flow(tmp_path, memory=(8 * GIB, 3 * GIB))
    flow.service.platform = "android"
    sheet = asyncio.run(_listed(flow))
    keys = _by_key(sheet.quant_box)
    assert keys["custom-quant-facts-0"].value == f"{dl.format_bytes(GIB)} · ✓ fits this phone"
    assert keys["custom-quant-facts-1"].value == f"{dl.format_bytes(4 * GIB)} · ⚠ tight on free RAM"
    assert keys["custom-quant-facts-2"].value == f"{dl.format_bytes(20 * GIB)} · ✕ too big for this phone"
    assert keys["custom-quant-0"].on_click is not None and keys["custom-quant-1"].on_click is not None
    assert keys["custom-quant-2"].on_click is None  # the RAM gate would refuse it after a 20 GB download
    assert keys["custom-quant-facts-2"].color == ft.Colors.ERROR


def test_a_failed_listing_says_so_and_retries_only_on_the_tap(tmp_path, fake_hf):
    flow, *_ = _flow(tmp_path)
    fake_hf.status = 500

    async def scenario():
        spawned = []
        flow.spawn_fn = lambda coro: spawned.append(asyncio.ensure_future(coro))
        sheet = await _listed(flow)
        keys = _by_key(sheet.quant_box)
        assert "Could not list the files of Qwen 3.5 2B" in keys["custom-error"].value and not sheet.loading
        count = len(fake_hf.requests)
        await asyncio.sleep(0)
        assert len(fake_hf.requests) == count and spawned == []  # no retry by itself
        fake_hf.status = None
        keys["custom-retry"].on_click(None)
        await spawned[-1]
        assert "custom-quant-0" in _by_key(sheet.quant_box) and "custom-error" not in _by_key(sheet.quant_box)

    asyncio.run(scenario())


def test_unexpected_listing_errors_end_the_wait_too(tmp_path):
    flow, _prefs, service, _used = _flow(tmp_path)

    async def boom(_ref):
        raise OSError("no route")

    service.resolve = boom
    sheet = asyncio.run(_listed(flow))
    assert not sheet.loading and "Could not reach Hugging Face (OSError)" in _by_key(sheet.quant_box)["custom-error"].value


def test_an_old_answer_does_not_replace_the_newer_size_or_touch_a_closed_sheet(tmp_path):
    flow, _prefs, service, _used = _flow(tmp_path)
    gates: dict = {}

    async def resolve(ref):
        gate = gates[ref.repo] = asyncio.Event()
        await gate.wait()
        return hf.RepoFiles("c", (hf.HfFile(f"{ref.repo.split('/')[1]}-Q4_0.gguf", 10),), ())

    service.resolve = resolve

    async def scenario():
        sheet = lmp.CustomModelSheet(flow, QWEN, size=QWEN["sizes"][0])
        old = asyncio.ensure_future(sheet.load())
        await asyncio.sleep(0)
        sheet.size = TWO_B
        new = asyncio.ensure_future(sheet.load())
        await asyncio.sleep(0)
        gates[REPO].set()
        await new
        assert not sheet.loading and "custom-quant-0" in _by_key(sheet.quant_box)
        gates[QWEN["sizes"][0]["repo"]].set()
        await old
        assert sheet.size is TWO_B and not sheet.loading and len(sheet.files()) == 1
        # closed while listing: the answer changes nothing on it
        closed = lmp.CustomModelSheet(flow, QWEN, size=TWO_B)
        task = asyncio.ensure_future(closed.load())
        await asyncio.sleep(0)
        closed.dialog.on_dismiss(None)
        gates[REPO].set()
        await task
        assert closed.loading  # dropped; nobody is looking

    asyncio.run(scenario())


# ---- the GPU mark -------------------------------------------------------------------------------------------


def _art(sheet) -> list:
    return [node for node in _walk(sheet.quant_box) if isinstance(node, ft.Image)]


def test_android_marks_q4_0_with_the_whole_halgakos_and_words(tmp_path, fake_hf):
    flow, *_ = _flow(tmp_path, backend="OpenCL")
    sheet = asyncio.run(_listed(flow))
    art = _art(sheet)
    assert len(art) == 1 and art[0].src == HALGAKOS_ASSET == "halgakos_chibi.png"  # never the head crop or the icon
    assert art[0].fit == ft.BoxFit.CONTAIN and art[0].height >= 32 and art[0].width >= 32
    assert _texts(sheet.quant_box).count("GPU-friendly") == 1
    row = next(node for node in sheet.quant_box.controls if str(getattr(node, "key", "")).startswith("custom-quant-")
               and art[0] in list(_walk(node)))
    assert "Q4_0" in _texts(row) and "GPU-friendly" in _texts(row)
    assert row.border is None  # only the hint highlights the row


@pytest.mark.parametrize("backend", ["Metal", None])
def test_ios_and_builds_without_opencl_show_no_mark(tmp_path, fake_hf, backend):
    flow, *_ = _flow(tmp_path, backend=backend)
    sheet = asyncio.run(_listed(flow, highlight_gpu=True))
    assert _art(sheet) == [] and "GPU-friendly" not in _texts(sheet.quant_box)
    assert all(getattr(node, "border", None) is None for node in sheet.quant_box.controls)


def test_a_row_fits_a_narrow_phone_by_construction(tmp_path, fake_hf):
    """The mascot has a fixed 40 dp, the text takes the rest and ellipsizes: no fixed width over a 320 dp sheet."""
    flow, *_ = _flow(tmp_path)
    sheet = asyncio.run(_listed(flow))
    for node in _walk(sheet.dialog.content):
        assert (getattr(node, "width", None) or 0) <= 40
    for row in (c for c in sheet.quant_box.controls if str(getattr(c, "key", "")).startswith("custom-quant-")):
        assert row.content.controls[-1].expand is True


# ---- a chosen quant -----------------------------------------------------------------------------------------


def _real_files(fake_hf):
    fake_hf.trees.pop(REPO)
    fake_hf.add(REPO, {Q4_0: b"Q" * 4096, "Qwen3.5-2B-BF16.gguf": b"B" * 8192, "mmproj-F16.gguf": b"M" * 1024})


def _tap_quant(sheet, label):
    for row in sheet.quant_box.controls:
        if label in _texts(row) and str(getattr(row, "key", "")).startswith("custom-quant-"):
            row.on_click(None)
            return
    raise AssertionError(label)


def test_choosing_a_quant_downloads_it_as_an_ordinary_row_with_image_support(tmp_path, fake_hf):
    _real_files(fake_hf)
    flow, prefs, service, used = _flow(tmp_path)

    async def scenario():
        tasks = []
        flow.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        after = []
        sheet = lmp.CustomModelSheet(flow, QWEN, size=TWO_B, on_used=after.append)
        await sheet.load()
        _tap_quant(sheet, "Q4_0")
        assert sheet.closed
        await tasks[-1]
        row = presets.load_presets(prefs)[-1]
        assert row["name"] == "Qwen 3.5 2B · Q4_0" and row["repo"] == REPO and row["file"] == Q4_0
        assert row["mmproj"] == "mmproj-F16.gguf" and row["size"] == 4096 and row["mmproj_size"] == 1024
        assert [p["name"] for p in presets.load_presets(prefs)][:4] == ["Garbage", "Light", "Medium", "Heavy"]
        stem = flow.stem_of_preset(row)
        assert service.registry.resolve(stem) and used == [f"local/{stem}"] and after == used
        assert flow.row_state(row, stem)[0] == "use" and flow.is_vision_preset(row, stem)
        assert flow.other_models()[0] == []  # the row names it: not listed twice
        # the menu then says it is on the phone; choosing it again adds nothing and downloads nothing
        again = await _listed(flow)
        assert "On this device" in _texts(again.quant_box)
        count = len(presets.load_presets(prefs))
        requests = [r for r in fake_hf.requests if "/resolve/" in r[0]]
        assert await flow.get_custom(presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4096))) == f"local/{stem}"
        assert len(presets.load_presets(prefs)) == count
        assert [r for r in fake_hf.requests if "/resolve/" in r[0]] == requests

    asyncio.run(scenario())


def test_no_quant_downloads_too(tmp_path, fake_hf):
    _real_files(fake_hf)
    flow, prefs, service, used = _flow(tmp_path)
    row = presets.custom_row(TWO_B, hf.HfFile("Qwen3.5-2B-BF16.gguf", 8192))
    assert asyncio.run(flow.get_custom(row)) and used
    saved = presets.load_presets(prefs)[-1]
    assert saved["name"] == "Qwen 3.5 2B · No quant (BF16)" and saved["note"].startswith("Custom · ")


def test_declining_the_download_leaves_no_row(tmp_path, fake_hf):
    _real_files(fake_hf)
    flow, prefs, _service, used = _flow(tmp_path, confirm=False)
    row = presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4096))
    assert asyncio.run(flow.get_custom(row)) is None
    assert presets.load_presets(prefs) == list(presets.DEFAULT_PRESETS) and used == []


def test_a_failed_download_keeps_its_row_with_retry(tmp_path, fake_hf):
    flow, prefs, _service, _used = _flow(tmp_path)
    fake_hf.status = 500
    row = presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4096))
    assert asyncio.run(flow.get_custom(row)) is None
    saved = presets.load_presets(prefs)[-1]
    assert saved["file"] == Q4_0 and flow.row_state(saved, flow.stem_of_preset(saved))[0] == "error"


def test_a_full_list_says_why(tmp_path):
    user = [{"name": f"u{i}", "repo": "me/x", "file": f"f{i}.gguf"} for i in range(presets.MAX_PRESETS)]
    prefs = fakes.FakePrefs({presets.PREF_KEY: user})
    flow, prefs, _service, _used = _flow(tmp_path, prefs=prefs)
    said: list = []
    flow.notify = said.append
    assert asyncio.run(flow.get_custom(presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4)))) is None
    assert len(presets.load_presets(prefs)) == presets.MAX_PRESETS and "remove one" in said[0]


def test_a_taken_name_gets_a_number(tmp_path):
    prefs = fakes.FakePrefs({presets.PREF_KEY: [{"name": "Qwen 3.5 2B · Q4_0", "repo": "me/x", "file": "x.gguf"}]})
    rows, index, added = presets.add_row(prefs, presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4)))
    assert added and rows[index]["name"] == "Qwen 3.5 2B · Q4_0 2"


# ---- the former GPU presets ---------------------------------------------------------------------------------


def _old_rows():
    return [dict(row) for row in presets.GPU_PRESETS]


def _download(service, stem, row, *, n_ctx=None):
    os.makedirs(service.registry.root, exist_ok=True)
    with open(service.registry.file_path(stem), "wb") as handle:
        handle.write(b"GGUF")
    service.registry.add(stem, {"repo": row["repo"], "file": row["file"], "revision": "r", "sha256": "0" * 64,
                                "size": 4, "downloaded_at": 1})


def test_downloaded_rows_stay_and_the_others_go(tmp_path):
    old = _old_rows()
    old[0]["n_ctx"] = 12288  # its stored settings
    mine = {"name": "Mine", "repo": "me/x", "file": "x.gguf", "note": "", "quant": "", "n_ctx": 0}
    renamed = {**old[3], "name": "My Heavy"}  # a Q4_0 row the user renamed is their own
    prefs = fakes.FakePrefs({presets.PREF_KEY: [*presets.DEFAULT_PRESETS, old[0], old[1], mine, renamed]})
    service, _engine = make_service(tmp_path, stems=(), prefs=prefs, gpu_backend="OpenCL")
    _download(service, "garbage-gpu", old[0])
    flow = lmp.OnDeviceFlow(service=service, prefs=prefs)
    rows = flow.presets()
    assert [r["name"] for r in rows] == ["Garbage", "Light", "Medium", "Heavy", "Garbage · GPU (Q4_0)", "Mine", "My Heavy"]
    kept = rows[4]
    assert kept["n_ctx"] == 12288 and flow.row_state(kept, flow.stem_of_preset(kept))[0] == "use"
    assert service.context_for("garbage-gpu") == 12288
    assert os.path.isfile(service.registry.file_path("garbage-gpu"))
    # the next flow changes nothing more
    lmp.OnDeviceFlow(service=service, prefs=prefs)
    assert [r["name"] for r in presets.load_presets(prefs)] == [r["name"] for r in rows]


def test_a_user_with_only_defaults_saves_nothing(tmp_path):
    prefs = fakes.FakePrefs()
    flow, prefs, *_ = _flow(tmp_path, prefs=prefs)
    assert presets.PREF_KEY not in prefs.data


def test_a_custom_pick_of_the_same_file_is_not_a_former_row(tmp_path):
    row = presets.custom_row(TWO_B, hf.HfFile(Q4_0, 4))
    assert not presets.is_old_gpu_row(row) and presets.is_old_gpu_row(_old_rows()[1])
    assert presets.is_old_gpu_row({**_old_rows()[1], "name": "Light · GPU (Q4_0) 2"})
    prefs = fakes.FakePrefs({presets.PREF_KEY: [row]})
    _flow(tmp_path, prefs=prefs)
    assert [r["file"] for r in presets.load_presets(prefs)] == [Q4_0]


def test_nothing_adds_them_any_more(tmp_path):
    for gone in ("add_gpu_presets", "add_gpu_preset", "missing_gpu_presets", "gpu_preset_for"):
        assert not hasattr(presets, gone)
    assert not hasattr(lmp.OnDeviceFlow, "offer_gpu_presets_once") and not hasattr(lmp, "GPU_PRESETS_OFFERED_KEY")
    prefs = fakes.FakePrefs()
    service, _engine = make_service(tmp_path, prefs=prefs, gpu_backend="OpenCL")
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    panel.did_show()
    assert "local-gpu-presets" not in _by_key(panel.control)
    assert presets.PREF_KEY not in prefs.data


# ---- the Q4_0 hint ------------------------------------------------------------------------------------------


def test_the_hint_opens_the_menu_at_that_size_with_q4_0_highlighted(tmp_path, fake_hf):
    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    service, _engine = make_service(tmp_path, stems=(), prefs=prefs, gpu_backend="OpenCL")
    _download(service, "light", {"repo": REPO, "file": "Qwen3.5-2B-UD-Q4_K_XL.gguf"})
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: "local/light")
    panel.build()
    assert panel.gpu_swap_box.visible and panel.gpu_swap_button.content == "Choose Q4_0 · Qwen 3.5 2B"

    async def scenario():
        tasks = []
        panel.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        panel.gpu_swap_button.on_click(None)
        await tasks[-1]
        return panel.last_dialog

    sheet = asyncio.run(scenario())
    assert sheet.size is TWO_B and sheet.family is QWEN
    marked = [row for row in sheet.quant_box.controls if getattr(row, "border", None) is not None]
    assert len(marked) == 1 and "Q4_0" in _texts(marked[0]) and marked[0].bgcolor == ft.Colors.SECONDARY_CONTAINER


def test_a_model_outside_the_table_gets_no_hint(tmp_path):
    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    service, _engine = make_service(tmp_path, stems=(), prefs=prefs, gpu_backend="OpenCL")
    _download(service, "own", {"repo": "me/x", "file": "x-Q4_K_M.gguf"})
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: "local/own")
    panel.build()
    assert panel.gpu_swap() is None and not panel.gpu_swap_box.visible


# ---- the three screens --------------------------------------------------------------------------------------


class _Page:
    width, height = 320, 640

    def __init__(self):
        self.shown = []

    def show_dialog(self, dialog):
        dialog.open = True
        self.shown.append(dialog)

    def update(self, *_a):
        pass


def _names():
    return [f"Custom · {family['name']}" for family in presets.CUSTOM_FAMILIES]


def test_model_manager_rows_open_the_sheet_and_it_closes_by_identity(tmp_path):
    prefs = fakes.FakePrefs()
    service, _engine = make_service(tmp_path, prefs=prefs)
    page = _Page()
    panel = lmp.LocalModelsPanel(service=service, prefs=prefs, page=page)
    panel.build()
    keys = _by_key(panel.preset_column)
    assert len(panel.preset_column.controls) == 8
    for family in presets.CUSTOM_FAMILIES:
        assert keys[f"local-custom-action-{family['id']}"].content == "Custom"
    assert set(_names()) <= set(_texts(panel.preset_column))
    keys["local-custom-action-llama"].on_click(None)
    sheet = panel.last_dialog
    assert page.shown == [sheet.dialog] and sheet.family["name"] == "Llama" and sheet.dialog.open
    assert sheet.dialog.key == "custom-sheet"
    sheet.close()
    assert not sheet.dialog.open and sheet.closed


def test_a_tap_on_a_replaced_control_does_not_raise():
    class Detached:
        @property
        def page(self):
            raise RuntimeError("event control is not attached to a page")

    assert lmp.tapped_page(Detached()) is None and lmp.tapped_page(None) is None


def test_model_picker_rows(tmp_path):
    from glossarion_mobile.ui.sheets import model_sheet as ms

    prefs = fakes.FakePrefs()
    service, _engine = make_service(tmp_path, prefs=prefs)
    env = ms.SheetEnv(prefs=prefs, local_engine=service)
    sheet = ms.ModelSheet(current_model="gpt-6", models=["gpt-6"], env=env, tab="ondevice")
    texts = [t for row in sheet.ondevice_list.controls for t in _texts(row)]
    assert "Presets · 8" in texts and set(_names()) <= set(texts)
    page = _Page()
    flow = sheet.ondevice_flow()
    flow.page = page
    keys = {k: v for row in sheet.ondevice_list.controls for k, v in _by_key(row).items()}
    button = keys["ondevice-custom-gemma-action"]
    assert button.content == "Custom" and isinstance(button, ft.FilledTonalButton)
    button.on_click(None)
    assert flow.last_dialog.family["name"] == "Gemma" and page.shown == [flow.last_dialog.dialog]


def test_welcome_cells_open_the_same_sheet_and_a_pick_counts_like_a_sign_in(tmp_path, fake_hf):
    from test_welcome_drawer_compact import _engine_kwargs, by_key, screen

    _real_files(fake_hf)
    kwargs, service, _engine = _engine_kwargs(tmp_path)
    service.gpu_backend = "OpenCL"
    s = screen(**kwargs)
    rows = by_key(s.page_area.content)
    assert len(rows["welcome-ondevice"].controls) == 8
    cell = rows["welcome-ondevice-custom-qwen35"]
    assert _texts(cell) == ["Qwen 3.5", "Custom"] and cell.col == {"xs": 3}
    page = _Page()
    flow = s.ondevice_flow()
    flow.page = page
    flow.confirm_fn = lambda *_a: True

    async def scenario():
        tasks = []
        flow.spawn_fn = lambda coro: tasks.append(asyncio.ensure_future(coro))
        cell.on_click(None)
        sheet = flow.last_dialog
        assert isinstance(sheet, lmp.CustomModelSheet) and page.shown == [sheet.dialog]
        sheet.select_size(TWO_B)
        await tasks[-1]
        _tap_quant(sheet, "Q4_0")
        await tasks[-1]

    asyncio.run(scenario())
    assert s.flow.signed_in is True and "runs on this phone" in s.model_notes["local"]
    rows = by_key(s.page_area.content)
    assert s.ondevice_key(presets.load_presets(kwargs["prefs"])[-1]) in rows  # its own cell, like any preset
    assert len(rows["welcome-ondevice"].controls) == 9


# ---- Context: 262,144 ---------------------------------------------------------------------------------------


def test_the_slider_ends_at_262144_with_the_same_default():
    assert lmp.CONTEXT_SLIDER == (2048, 262144, 1024)
    control = lmp.ContextControl(4096, lambda _v: None)
    assert control.slider.max == 262144 and control.slider.divisions == 254 and control.slider.value == 4096
    assert svc.DEFAULT_CONTEXT_SMALL == 4096 and svc.DEFAULT_CONTEXT_LARGE == 2048


def test_the_ram_warning_counts_the_new_end(tmp_path):
    """A 1 GB model needs 1.15 GB + 64 KiB per context token + 0.4 GB: about 2.5 GB at 16,384, 3.5 GB at 32,768."""
    memory = {"value": (8 * GIB, 3 * GIB)}
    prefs = fakes.FakePrefs()
    service, _engine = make_service(tmp_path, stems=("light",), prefs=prefs, memory=lambda: memory["value"])
    service.platform = "android"
    service.registry.add("light", {**service.registry.load()["light"], "size": GIB})
    assert service.save_context(16384)[1] is None and service.ram_check("light")[0] == "ok"
    assert service.save_context(32768)[1] is None and service.settings()["n_ctx"] == 32768
    assert service.ram_check("light")[0] == "confirm" and service.context_for("light") == 32768
    assert service.preset_ram_check({"name": "x", "size": GIB})[0] == "confirm"
    memory["value"] = (3 * GIB, 3 * GIB)  # more than the whole phone
    assert service.ram_check("light")[0] == "refuse"


def test_welcome_uses_the_same_slider(tmp_path):
    from test_welcome_drawer_compact import _engine_kwargs, screen

    kwargs, _service, _engine = _engine_kwargs(tmp_path)
    s = screen(**kwargs, current_model=lambda: "local/whatever")
    s.flow.step = 0
    budget = s._run_budget()
    keys = _by_key(budget)
    assert keys["welcome-ctx-slider"].max == 262144 and keys["welcome-ctx-slider"].divisions == 254
    assert keys["welcome-ctx"].value == "16384" and _service.settings()["n_ctx"] == 16384


def test_welcome_keeps_a_saved_context_and_applies_the_default_when_skipped_past(tmp_path):
    from test_welcome_drawer_compact import _engine_kwargs, screen

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    s = screen(**kwargs, current_model=lambda: "local/whatever")
    s.finish()  # never reached Request size
    assert service.settings()["n_ctx"] == 16384
    service.save_context(8192)
    s = screen(**kwargs, current_model=lambda: "local/whatever")
    assert _by_key(s._run_budget())["welcome-ctx"].value == "8192"
    assert service.settings()["n_ctx"] == 8192


def test_a_cloud_model_in_welcome_saves_no_context(tmp_path):
    from test_welcome_drawer_compact import _engine_kwargs, screen

    kwargs, service, _engine = _engine_kwargs(tmp_path)
    screen(**kwargs, current_model=lambda: "authgpt/gpt-6-luna").finish()
    assert service.settings()["n_ctx"] == 0


def test_the_slider_is_yellow_when_tight_and_red_over_the_ram():
    levels = {4096: "ok", 32768: "confirm", 131072: "refuse"}
    control = lmp.ContextControl(4096, lambda _v: None, ram_level=lambda tokens: levels[tokens])
    assert control.slider.active_color is None and control.slider.thumb_color is None
    control.set_value(32768)
    assert control.slider.active_color == lmp.ft.Colors.AMBER and control.slider.thumb_color == lmp.ft.Colors.AMBER
    control.set_value(131072)
    assert control.slider.active_color == lmp.ft.Colors.ERROR
    control.set_value(4096)
    assert control.slider.active_color is None
    control.ram_level = None  # no verdict to show
    control.set_value(131072)
    assert control.slider.active_color is None


def test_the_flow_asks_the_ram_gate_for_the_current_model(tmp_path):
    service, _engine = make_service(tmp_path, stems=("light",), memory=lambda: (3 * GIB, 2 * GIB))
    service.platform = "android"
    service.registry.add("light", {**service.registry.load()["light"], "size": GIB})
    flow = lmp.OnDeviceFlow.__new__(lmp.OnDeviceFlow)
    flow.service, flow.current_model = service, lambda: "local/light"
    assert flow.context_ram_level(2048) == "ok"
    assert flow.context_ram_level(16384) == "confirm"
    assert flow.context_ram_level(262144) == "refuse"
    flow.current_model = lambda: "authgpt/gpt-6-luna"
    assert flow.context_ram_level(262144) == "ok"
