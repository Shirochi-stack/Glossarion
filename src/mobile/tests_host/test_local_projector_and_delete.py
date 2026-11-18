"""On-device models: a repository's vision projector is never the model, and Delete covers every model row.

Owner's phone: Add from Hugging Face of ``prism-ml/Ternary-Bonsai-2-9B-gguf`` with "Save as a preset" downloaded
``Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf`` as the model ("Could not load local/Ternary-Bonsai-2-9B-mmproj-Q8_0:
Failed to load model from file"): the listing only knew projectors named ``mmproj-*``, so that one counted as a
model and won the Q8_0 auto-pick. Now every ``*mmproj*.gguf`` is a projector (never offered, picked or saved as the
model), drafters are not listed, unusual quants (PQ2_0, PTQ1_0, TQ1_0) open the file chooser with nothing
preselected, and earlier bad state is repaired: such presets become the repository's file-less row with that
projector as image support, such downloads become a text-only model's image support or are deleted.

Delete: one confirmation (model name, space freed, an optional preset removal for the user's own rows), refused
with the reason under the row while a running job or an answer uses the model, unloading an idle one first,
for downloaded, failed and Other-downloads rows; a projector belongs to its own model only.

No network: the fake Hugging Face is a local server (``local_llm_fakes.FakeHF``).

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_projector_and_delete.py
"""

from __future__ import annotations

import asyncio
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


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

BONSAI = "prism-ml/Ternary-Bonsai-2-9B-gguf"
#: the real file names of that repository (Hugging Face API, 2026-11-18); tiny stand-in bytes
BONSAI_FILES = {
    "Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf": b"d" * 11,
    "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf": b"m" * 26,
    "Ternary-Bonsai-2-9B-PQ2_0.gguf": b"p" * 24,
    "Ternary-Bonsai-2-9B-PTQ1_0.gguf": b"t" * 20,
    "Ternary-Bonsai-2-9B-Q2_0.gguf": b"q" * 25,
    "Ternary-Bonsai-2-9B-mmproj-BF16.gguf": b"B" * 9,
    "Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf": b"8" * 6,
    "README.md": b"readme",
}
MMPROJ_Q8 = "Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf"


@pytest.fixture()
def fake_hf(monkeypatch):
    server = fakes.FakeHF()
    server.add(BONSAI, BONSAI_FILES)
    server.add("t/Tern-GGUF", {"Tern-TQ1_0.gguf": b"x" * 30, "Tern-mmproj-F16.gguf": b"y" * 5})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    yield server
    server.close()


def _service(tmp_path, prefs, engine=None, **kwargs):
    engine = engine or eng.FakeEngine()
    kwargs.setdefault("stop_state", lambda: None)
    service = svc.LocalEngineService(prefs=prefs, registry=dl.Registry(str(tmp_path / "models")),
                                     engine_factory=lambda: engine, watch=False, memory=lambda: (0, None), **kwargs)
    return service, engine


def _put(registry, stem, repo, file, data=b"model", projector=None):
    os.makedirs(registry.root, exist_ok=True)
    with open(registry.file_path(stem), "wb") as handle:
        handle.write(data)
    registry.add(stem, {"repo": repo, "file": file, "revision": "r", "size": len(data), "sha256": None,
                        "downloaded_at": 1})
    if projector is not None:
        with open(registry.file_path(registry.projector_key(stem)), "wb") as handle:
            handle.write(projector)
        registry.attach_projector(stem, {"file": "mmproj-F16.gguf", "revision": "r", "size": len(projector)})


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


# ---- the projector is never the model -------------------------------------------------------------


def test_listing_never_offers_or_picks_a_projector(fake_hf):
    listing = hf.list_repo(hf.parse_ref(BONSAI))
    names = [f.name for f in listing.files]
    assert not any("mmproj" in n.lower() for n in names)
    assert "Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf" not in names  # a drafter cannot answer on its own
    assert sorted(names) == ["Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf", "Ternary-Bonsai-2-9B-PQ2_0.gguf",
                             "Ternary-Bonsai-2-9B-PTQ1_0.gguf", "Ternary-Bonsai-2-9B-Q2_0.gguf"]
    assert [p.name for p in listing.projectors] == ["Ternary-Bonsai-2-9B-mmproj-BF16.gguf", MMPROJ_Q8]
    # none of the usual quants (the MTP file's Q8_0 is its draft head's): nothing is guessed, the user chooses
    assert hf.pick_file(listing.files) is None
    with pytest.raises(hf.HfError, match="vision projector"):
        hf.pick_file(listing.files, file=MMPROJ_Q8)
    # a projector a caller lists anyway is still never the pick
    assert hf.pick_file([hf.HfFile(MMPROJ_Q8, 6), hf.HfFile("Ternary-Bonsai-2-9B-Q2_0.gguf", 25)]).name \
        == "Ternary-Bonsai-2-9B-Q2_0.gguf"
    assert hf.pick_projector(listing.projectors).name == "Ternary-Bonsai-2-9B-mmproj-BF16.gguf"
    # one ternary model beside its projector: that model, whatever its quant tag
    tern = hf.list_repo(hf.parse_ref("t/Tern-GGUF"))
    assert hf.pick_file(tern.files).name == "Tern-TQ1_0.gguf" and [p.name for p in tern.projectors] == \
        ["Tern-mmproj-F16.gguf"]
    assert hf.is_projector_name("x/MMPROJ-f16.gguf") and not hf.is_projector_name("Qwen3.5-2B-Q4_0.gguf")


@needs_flet
def test_add_from_hugging_face_asks_among_the_models_and_saves_one(tmp_path, fake_hf):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: []})
        service, engine = _service(tmp_path, prefs)
        offered, asked = [], []

        def choose(repo, files, selected):
            offered.append(([f.name for f in files], selected))
            return next(f for f in files if f.name == "Ternary-Bonsai-2-9B-Q2_0.gguf")

        panel = LocalModelsPanel(service=service, prefs=prefs, choose=choose,
                                 confirm=lambda title, body, label: asked.append(body) or True)
        panel.build()
        panel.hf_field.value = BONSAI
        assert await panel.download_typed() == "Ternary-Bonsai-2-9B-Q2_0"
        files, selected = offered[0]
        assert selected is None and not any("mmproj" in f.lower() for f in files)
        assert "mmproj" not in asked[-1]
        row = presets.load_presets(prefs)[0]
        assert row["file"] == "Ternary-Bonsai-2-9B-Q2_0.gguf" and row["repo"] == BONSAI
        assert engine.loads and engine.loads[-1][0].endswith("Ternary-Bonsai-2-9B-Q2_0.gguf")
        # the projector waits for the first image, then comes as this model's companion
        assert not service.is_vision("Ternary-Bonsai-2-9B-Q2_0")
        assert await panel.add_vision(0)
        assert service.is_vision("Ternary-Bonsai-2-9B-Q2_0")
        assert service.registry.load()["Ternary-Bonsai-2-9B-Q2_0"]["mmproj"]["file"] == \
            "Ternary-Bonsai-2-9B-mmproj-BF16.gguf"
        assert service.registry.resolve("Ternary-Bonsai-2-9B-mmproj-Q8_0") is None
        panel.dispose()

    asyncio.run(scenario())


def test_a_saved_preset_naming_a_projector_is_repaired():
    prefs = fakes.FakePrefs({presets.PREF_KEY: [
        {"name": "Ternary-Bonsai-2-9B-mmproj-Q8_0", "repo": BONSAI, "file": MMPROJ_Q8, "quant": "", "note": BONSAI,
         "n_ctx": 0, "size": 6, "sha256": "a" * 64},
        {"name": "Tern-mmproj-F16", "repo": "t/Tern-GGUF", "file": "Tern-mmproj-F16.gguf", "note": "", "n_ctx": 0},
        {"name": "Tern", "repo": "t/Tern-GGUF", "file": "", "note": "", "n_ctx": 0},
    ]})
    rows = presets.load_presets(prefs)
    assert [r["name"] for r in rows] == ["Ternary-Bonsai-2-9B-gguf", "Tern"]  # the duplicate of Tern goes
    repaired = rows[0]
    assert repaired["file"] == "" and "size" not in repaired and "sha256" not in repaired
    assert repaired["mmproj"] == MMPROJ_Q8  # the projector stays as its image support
    assert prefs.data[presets.PREF_KEY] == rows  # saved, so the repair runs once
    assert presets.validate_preset({"name": "X", "repo": BONSAI, "file": MMPROJ_Q8}, rows)[1].startswith(
        f"{MMPROJ_Q8} is a vision projector")


def test_a_projector_downloaded_as_a_model_is_repaired_at_start(tmp_path):
    registry = dl.Registry(str(tmp_path / "models"))
    _put(registry, "Ternary-Bonsai-2-9B-mmproj-Q8_0", BONSAI, MMPROJ_Q8, b"proj")
    _put(registry, "Other-mmproj-F16", "o/Other-GGUF", "Other-mmproj-F16.gguf", b"pp")
    _put(registry, "Ternary-Bonsai-2-9B-Q2_0", BONSAI, "Ternary-Bonsai-2-9B-Q2_0.gguf", b"model")
    with open(registry.file_path("Lost-mmproj-BF16") + dl.PART_SUFFIX, "wb") as handle:
        handle.write(b"part")
    import json

    with open(registry.file_path("Lost-mmproj-BF16") + dl.PART_SUFFIX + ".json", "w", encoding="utf-8") as handle:
        json.dump({"repo": "l/Lost", "file": "Lost-mmproj-BF16.gguf", "revision": "r", "size": 9, "kind": "model"},
                  handle)
    service = svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=registry, engine_factory=eng.FakeEngine,
                                     watch=False, memory=lambda: (0, None))
    assert sorted(service.repaired) == [("Lost-mmproj-BF16", "deleted"), ("Other-mmproj-F16", "deleted"),
                                        ("Ternary-Bonsai-2-9B-mmproj-Q8_0", "attached to Ternary-Bonsai-2-9B-Q2_0")]
    assert sorted(registry.load()) == ["Ternary-Bonsai-2-9B-Q2_0"]
    assert service.is_vision("Ternary-Bonsai-2-9B-Q2_0")
    with open(registry.projector_path("Ternary-Bonsai-2-9B-Q2_0"), "rb") as handle:
        assert handle.read() == b"proj"
    assert sorted(os.listdir(registry.root)) == ["Ternary-Bonsai-2-9B-Q2_0.gguf",
                                                 "Ternary-Bonsai-2-9B-Q2_0.mmproj.gguf", "models.json"]
    assert svc.LocalEngineService(prefs=fakes.FakePrefs(), registry=registry, watch=False).repaired == []


def test_a_real_load_failure_keeps_the_engine_reason(tmp_path):
    class Failing(eng.FakeEngine):
        def load(self, path, *args, **kwargs):
            raise ValueError(f"Failed to load model from file: {path}")

    service, _engine = _service(tmp_path, fakes.FakePrefs(), Failing())
    _put(service.registry, "M-Q2_0", "u/M", "M-Q2_0.gguf")
    ok, message = service.load_blocking("M-Q2_0")
    assert not ok and message == ("Could not load local/M-Q2_0: Failed to load model from file: "
                                  f"{service.registry.file_path('M-Q2_0')}")


# ---- Delete ---------------------------------------------------------------------------------------


@needs_flet
def test_delete_frees_the_files_and_offers_the_preset(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        own = {"name": "Mine", "repo": "u/A-GGUF", "file": "A-Q4_0.gguf", "quant": "", "note": "", "n_ctx": 0}
        garbage = dict(presets.DEFAULT_PRESETS[0])
        prefs = fakes.FakePrefs({presets.PREF_KEY: [own, garbage]})
        service, engine = _service(tmp_path, prefs)
        registry = service.registry
        _put(registry, "A-Q4_0", own["repo"], own["file"], b"a" * 4000, projector=b"p" * 1000)
        _put(registry, "B-Q4_0", "u/B-GGUF", "B-Q4_0.gguf", b"b" * 300, projector=b"q" * 200)
        _put(registry, "Qwen3.5-0.8B-UD-Q4_K_XL", garbage["repo"], garbage["file"], b"g" * 50)
        used = ["local/A-Q4_0"]
        panel = LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: used[-1], use_model=used.append)
        panel.build()
        assert (await service.load("A-Q4_0"))[0] and service.loaded_stem == "A-Q4_0"  # loaded, idle
        menu = _find(panel.preset_column, "local-preset-menu-0")
        assert "Delete file" in [i.content for i in menu.items]

        dialog = panel.confirm_delete("A-Q4_0", 0)
        assert dialog.body == f"Delete Mine (local/A-Q4_0) from this phone? This frees {dl.format_bytes(5000)}."
        assert dialog.items == ["Model file", "Image support"]
        option = panel.delete_dialog_option
        assert option is not None and option.label == "Also remove the preset “Mine”" and option.value is False
        option.value = True
        await dialog._on_confirm()
        assert service.loaded_stem is None and engine.unloads  # unloaded first
        assert not os.path.exists(registry.file_path("A-Q4_0"))
        assert not os.path.exists(registry.file_path(registry.projector_key("A-Q4_0")))
        assert registry.projector_path("B-Q4_0") and os.path.exists(registry.projector_path("B-Q4_0"))  # B's own
        assert [r["name"] for r in presets.load_presets(prefs)] == ["Garbage"]
        assert used[-1] == "local/A-Q4_0"  # the selection stays, shown as not downloaded like any missing file

        # a default row: its file goes, the preset stays and shows Download again; no preset option
        dialog = panel.confirm_delete("Qwen3.5-0.8B-UD-Q4_K_XL", 0)
        assert panel.delete_dialog_option is None and "Garbage (local/Qwen3.5-0.8B-UD-Q4_K_XL)" in dialog.body
        await dialog._on_confirm()
        assert _find(panel.preset_column, "local-preset-action-0").content == "Download"
        assert [r["name"] for r in presets.load_presets(prefs)] == ["Garbage"]

        # Other downloads: its Delete removes the row
        assert _find(panel.other_column, "local-other-B-Q4_0") is not None
        _find(panel.other_column, "local-other-delete-B-Q4_0").on_click(None)
        await panel.last_dialog._on_confirm()
        assert registry.load() == {} and not panel.other_card.visible
        assert os.listdir(registry.root) == ["models.json"]
        panel.dispose()

    asyncio.run(scenario())


@needs_flet
def test_delete_is_refused_while_a_job_or_an_answer_uses_the_model(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: []})
        job = {"model": "local/A-Q4_0"}
        service, _engine = _service(tmp_path, prefs, active_job_model=lambda: job["model"])
        _put(service.registry, "A-Q4_0", "u/A-GGUF", "A-Q4_0.gguf")
        panel = LocalModelsPanel(service=service, prefs=prefs)
        panel.build()
        assert panel.confirm_delete("A-Q4_0") is None
        note = _find(panel.other_column, "local-other-delete-note-A-Q4_0")
        assert note.value == "local/A-Q4_0 is in use by a running job: stop the job, then delete it"
        assert not panel.delete_file("A-Q4_0") and service.registry.resolve("A-Q4_0")
        # an answer holding the engine
        job["model"] = None
        assert service.gate.try_acquire()
        service._stem = "A-Q4_0"
        assert panel.confirm_delete("A-Q4_0") is None
        assert "is loading or answering a request" in _find(panel.other_column, "local-other-delete-note-A-Q4_0").value
        service.gate.release()
        service._stem = None
        dialog = panel.confirm_delete("A-Q4_0")  # the reason is gone: the note too
        assert dialog is not None and _find(panel.other_column, "local-other-delete-note-A-Q4_0") is None
        await dialog._on_confirm()
        assert service.registry.resolve("A-Q4_0") is None
        panel.dispose()

    asyncio.run(scenario())


@needs_flet
def test_a_failed_download_can_be_deleted(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    async def scenario():
        prefs = fakes.FakePrefs({presets.PREF_KEY: []})
        service, _engine = _service(tmp_path, prefs)
        registry = service.registry
        os.makedirs(registry.root, exist_ok=True)
        part = registry.file_path("C-Q4_0") + dl.PART_SUFFIX
        with open(part, "wb") as handle:
            handle.write(b"c" * 700)
        service.downloads.states["C-Q4_0"] = dl.DownloadState("C-Q4_0", "u/C-GGUF", "C-Q4_0.gguf", running=False,
                                                              error="Download failed after 6 tries (HTTP 503)")
        panel = LocalModelsPanel(service=service, prefs=prefs)
        panel.build()
        assert _find(panel.other_column, "local-dl-resume-C-Q4_0").content == "Retry"
        _find(panel.other_column, "local-dl-delete-C-Q4_0").on_click(None)
        dialog = panel.last_dialog
        assert dialog.body.endswith(f"This frees {dl.format_bytes(700)}.") and dialog.items == ["Unfinished model download"]
        await dialog._on_confirm()
        assert not os.path.exists(part) and not panel.other_card.visible
        panel.dispose()

    asyncio.run(scenario())
