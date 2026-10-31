"""Optional GPU for the on-device engine (``services/local_llm``): "Use GPU (experimental)" (off by default,
offered only with a GPU backend), the automatic CPU fallback (GPU load failure, a GPU answer failing, a
crash while the GPU started), the device line, the RAM gate, ``LlamaEngine``'s layer placement and its
first-token check, the Q4_0 GPU presets and the engine card's controls. FakeEngine / a fake llama_cpp.
"""

from __future__ import annotations

import importlib.util
import math
import types

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import presets
from glossarion_mobile.services.local_llm import service as svc
from test_local_llm_server import make_service

GB = 1024 ** 3


def _chat(service, model="local/fake-echo", text="hi"):
    run = service.chat(model, {"messages": [{"role": "user", "content": text}]}, lambda: False)
    with run:
        return "".join(run.deltas)


def _gpu_service(tmp_path, engine=None, gpu_on=True, backend="OpenCL", **kwargs):
    prefs = kwargs.pop("prefs", None) or fakes.FakePrefs({svc.GPU_KEY: True} if gpu_on else {})
    return make_service(tmp_path, engine=engine, prefs=prefs, gpu_backend=backend, **kwargs)


# ---- which devices get the option -----------------------------------------------------------------------

def test_gpu_is_offered_on_ios_and_android_arm64_only(monkeypatch):
    monkeypatch.setattr(eng, "ios_major", lambda: 17)
    assert eng.gpu_offer("ios") == "Metal"
    monkeypatch.setattr(eng, "ios_major", lambda: 14)       # ggml-metal needs an iOS 15 API
    assert eng.gpu_offer("ios") is None
    assert eng.gpu_offer("android", "aarch64") == "OpenCL"
    assert eng.gpu_offer("android", "x86_64") is None        # the emulator / Chromebooks: no GPU backend
    assert eng.gpu_offer("desktop", "AMD64") is None and eng.gpu_offer("", "aarch64") is None


def test_gpu_is_off_by_default_and_hidden_without_a_backend(tmp_path):
    service, engine = make_service(tmp_path, gpu_backend="OpenCL")
    assert not service.use_gpu() and service.gpu_note() is None
    _chat(service)
    assert engine.gpu_loads == [False] and service.snapshot().device == "CPU"
    # no backend (desktop, x86_64): the saved setting is ignored and cannot be switched on
    service, engine = make_service(tmp_path / "x", prefs=fakes.FakePrefs({svc.GPU_KEY: True}))
    assert service.gpu_backend is None and not service.use_gpu() and service.save_gpu(True) is False
    _chat(service)
    assert engine.gpu_loads == [False]


# ---- GPU load, device line, setting changes ---------------------------------------------------------------

def test_gpu_on_loads_on_the_gpu_and_says_which_device(tmp_path):
    service, engine = _gpu_service(tmp_path, eng.FakeEngine(gpu_device="QUALCOMM Adreno(TM) 830 (OpenCL)"))
    assert _chat(service) == "Echo: hi"
    assert engine.gpu_loads == [True] and engine.calls[-1]["gpu"] is True
    snap = service.snapshot()
    assert snap.device == "GPU: QUALCOMM Adreno(TM) 830 (OpenCL)" and snap.gpu_note is None
    # the first answer ended: no crash marker left behind
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    _chat(service, text="again")
    assert engine.gpu_loads == [True]                         # no reload
    # switching the GPU off frees the idle model; the next request loads it on the CPU
    assert service.save_gpu(False) is False and service.loaded_stem is None
    _chat(service)
    assert engine.gpu_loads == [True, False] and service.snapshot().device == "CPU"
    assert service.prefs.get(svc.GPU_KEY) is False


def test_gpu_failure_at_load_falls_back_to_the_cpu_and_stays_there(tmp_path):
    reason = "no supported GPU: llama.cpp's OpenCL backend runs on Qualcomm Adreno GPUs only"
    service, engine = _gpu_service(tmp_path, eng.FakeEngine(gpu_error=reason), stems=("a", "b"))
    assert _chat(service, "local/a") == "Echo: hi"          # the request still gets its answer
    assert engine.gpu_loads == [True, False] and not engine.on_gpu
    snap = service.snapshot()
    assert snap.device == "CPU" and snap.gpu_note == f"GPU unavailable for a: {reason} — using CPU"
    assert service.gpu_fallback() == {"stem": "a", "reason": reason}
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    _chat(service, "local/a", "again")
    assert engine.gpu_loads == [True, False]                  # stays on the CPU: no reload loop
    # another model gets a new GPU try
    _chat(service, "local/b")
    assert engine.gpu_loads == [True, False, True, False] and service.gpu_fallback()["stem"] == "b"
    # switching the setting off and on forgets the fallback
    service.save_gpu(False)
    assert service.gpu_fallback() is None and service.gpu_note() is None
    service.save_gpu(True)
    engine.gpu_error = None
    _chat(service, "local/b")
    assert engine.gpu_loads[-1] is True and engine.on_gpu and service.snapshot().gpu_note is None


def test_gpu_failure_while_answering_retries_the_request_on_the_cpu(tmp_path):
    engine = eng.FakeEngine(gpu_chat_error="llama_decode returned -3")
    service, engine = _gpu_service(tmp_path, engine)
    assert _chat(service) == "Echo: hi"
    assert engine.gpu_loads == [True, False] and [c["gpu"] for c in engine.calls] == [True, False]
    fallback = service.gpu_fallback()
    assert fallback["stem"] == "fake-echo" and "the GPU failed while answering (llama_decode returned -3)" in \
        fallback["reason"]
    assert service.snapshot().state == "ready" and not service.gate.busy


def test_a_crash_while_the_gpu_started_keeps_that_model_on_the_cpu(tmp_path):
    prefs = fakes.FakePrefs({svc.GPU_KEY: True, svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo", "at": 1}})
    service, engine = _gpu_service(tmp_path, prefs=prefs)
    service.start()
    try:
        assert prefs.get(svc.GPU_ATTEMPT_KEY) is None
        assert service.gpu_fallback() == {"stem": "fake-echo", "reason": svc.CRASH_REASON}
        _chat(service)
        assert engine.gpu_loads == [False]
        assert svc.CRASH_REASON in service.snapshot().gpu_note
    finally:
        service.stop()
        svc.set_default_service(None)


def test_the_crash_marker_lives_from_the_gpu_load_to_the_first_answer(tmp_path):
    service, engine = _gpu_service(tmp_path)
    assert service.load_blocking("fake-echo") == (True, None) and engine.on_gpu
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY)["stem"] == "fake-echo"   # a crash now is caught at next start
    _chat(service)
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    # an unload (normal end) clears it too
    service.unload()
    service.load_blocking("fake-echo")
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY)
    service.unload()
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    # a marker from a run with the GPU since switched off is dropped silently
    prefs = fakes.FakePrefs({svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo"}})
    service, _engine = make_service(tmp_path / "y", prefs=prefs, gpu_backend="Metal")
    assert service._recover_gpu_attempt() == "fake-echo" and service.gpu_fallback() is None


def test_ram_gate_counts_the_gpu(tmp_path):
    memory = lambda: (4 * GB, None)  # an iPhone with 4 GB: iOS refuses above half
    service, _engine = _gpu_service(tmp_path, backend="Metal", memory=memory)
    service.platform = "ios"
    size = GB
    service.registry.add("fake-echo", {**service.registry.load()["fake-echo"], "size": size})
    n_ctx = service.context_for("fake-echo")
    cpu = svc.ram_need(size, n_ctx)
    assert cpu < 2 * GB < cpu + svc.gpu_extra(size)
    assert service.ram_check("fake-echo")[0] == "ok"
    assert service.ram_check("fake-echo", gpu=True)[0] == "refuse"
    assert svc.gpu_extra(GB) == int(0.1 * GB + 256 * svc.MIB)


# ---- LlamaEngine: placement and the first-token check ------------------------------------------------------

class _FakeLlama:
    made: list = []
    fail = None
    logits = (0.5, -1.0, 2.0, 0.0)

    def __init__(self, **kwargs):
        if _FakeLlama.fail:
            raise ValueError(_FakeLlama.fail)
        self.kwargs = kwargs
        self.evals, self.resets, self.closed = [], 0, False
        self._ctx = types.SimpleNamespace(get_logits=lambda: list(_FakeLlama.logits))
        self.metadata = {}
        _FakeLlama.made.append(self)

    def tokenize(self, text, add_bos=True, special=False):
        return [1, 2, 3, 4]

    def reset(self):
        self.resets += 1

    def eval(self, tokens):
        self.evals.append(list(tokens))

    def n_vocab(self):
        return len(_FakeLlama.logits)

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_llama(monkeypatch):
    _FakeLlama.made, _FakeLlama.fail, _FakeLlama.logits = [], None, (0.5, -1.0, 2.0, 0.0)
    module = types.SimpleNamespace(Llama=_FakeLlama)
    monkeypatch.setattr(eng.LlamaEngine, "_module", lambda self: module)
    monkeypatch.setattr(eng, "_make_formatter", lambda llm: None)
    monkeypatch.setattr(eng, "cpu_variant", lambda: "android_armv8.6_1")
    state = {"backend": "OpenCL", "device": "QUALCOMM Adreno(TM) 830", "error": None}
    monkeypatch.setattr(eng, "gpu_state", lambda: dict(state))
    return state


def test_llama_engine_places_layers_by_the_gpu_flag(fake_llama):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4)
    cpu = _FakeLlama.made[-1].kwargs
    assert (cpu["n_gpu_layers"], cpu["split_mode"], cpu["main_gpu"]) == (0, eng.SPLIT_MODE_NONE, -1)
    assert not engine.on_gpu and engine.device == "CPU (android_armv8.6_1)" and not _FakeLlama.made[-1].evals
    engine.load("m.gguf", 2048, 4, gpu=True)
    gpu = _FakeLlama.made[-1]
    assert (gpu.kwargs["n_gpu_layers"], gpu.kwargs["split_mode"], gpu.kwargs["main_gpu"]) == (-1, 0, 0)
    assert engine.on_gpu and engine.device == "GPU: QUALCOMM Adreno(TM) 830 (OpenCL)"
    assert gpu.evals == [[1, 2, 3, 4], [4]] and gpu.resets == 2        # the check ran and left it reset
    assert _FakeLlama.made[0].closed                                    # the CPU copy was closed first
    engine.unload()
    assert engine.device is None and not engine.on_gpu


def test_llama_engine_gpu_failures_raise_gpu_unavailable(fake_llama):
    engine = eng.LlamaEngine()
    fake_llama.update(device=None, error="no OpenCL driver (dlopen failed: library \"libOpenCL.so\" not found)")
    with pytest.raises(eng.GpuUnavailable, match="no OpenCL driver"):
        engine.load("m.gguf", 2048, 4, gpu=True)
    assert not _FakeLlama.made and not engine.loaded              # nothing was loaded
    fake_llama.update(device="Apple A17 Pro GPU", backend="Metal", error=None)
    _FakeLlama.fail = "failed to load model"
    with pytest.raises(eng.GpuUnavailable, match="did not load on the GPU"):
        engine.load("m.gguf", 2048, 4, gpu=True)
    _FakeLlama.fail = None
    _FakeLlama.logits = (math.nan, 1.0, 2.0, 3.0)
    with pytest.raises(eng.GpuUnavailable, match="first token on the GPU failed .*NaN or infinite"):
        engine.load("m.gguf", 2048, 4, gpu=True)
    assert _FakeLlama.made[-1].closed and not engine.loaded
    _FakeLlama.logits = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(eng.GpuUnavailable, match="all equal"):
        engine.load("m.gguf", 2048, 4, gpu=True)
    # a CPU load error stays a plain error (the service reports it as before)
    _FakeLlama.fail = "bad file"
    with pytest.raises(ValueError, match="bad file"):
        engine.load("m.gguf", 2048, 4)


def test_gpu_state_without_our_loader(monkeypatch):
    monkeypatch.setattr(eng.importlib, "import_module", lambda name: types.SimpleNamespace())
    assert eng.gpu_state() == {"backend": None, "device": None, "error": "this engine build has no GPU support"}

    def missing(name):
        raise ImportError("No module named 'llama_cpp'")

    monkeypatch.setattr(eng.importlib, "import_module", missing)
    assert eng.gpu_state()["error"].startswith(eng.UNAVAILABLE)


# ---- presets ----------------------------------------------------------------------------------------------

def test_former_gpu_presets_are_q4_0_of_the_default_repos():
    defaults = {p["repo"]: p for p in presets.DEFAULT_PRESETS}
    assert [p["file"] for p in presets.GPU_PRESETS] == [
        "Qwen3.5-0.8B-Q4_0.gguf", "Qwen3.5-2B-Q4_0.gguf", "Qwen3.5-4B-Q4_0.gguf", "Qwen3.5-9B-Q4_0.gguf"]
    assert [p["size"] for p in presets.GPU_PRESETS] == [507154688, 1214873856, 2583221408, 5379417312]
    for row in presets.GPU_PRESETS:
        assert presets._clean(row) == row                        # a valid preset row, nothing dropped
        assert row["repo"] in defaults and row["mmproj"] == defaults[row["repo"]]["mmproj"]
        assert row["mmproj_sha256"] == defaults[row["repo"]]["mmproj_sha256"] and "Q4_0" in row["name"]
    # the defaults (CPU, iOS) are unchanged and hold no Q4_0 file
    assert not any("Q4_0" in p["file"] for p in presets.DEFAULT_PRESETS)


# ---- the engine card ---------------------------------------------------------------------------------------

def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")


@needs_flet
def test_engine_card_gpu_switch_note_and_status(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    # no GPU backend: nothing to switch
    prefs = fakes.FakePrefs()
    service, _engine = make_service(tmp_path / "none", prefs=prefs)
    panel = LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert not panel.gpu_switch.visible and not panel.gpu_note.visible
    # Android arm64 (OpenCL): off by default
    prefs = fakes.FakePrefs()
    engine = eng.FakeEngine(gpu_error="no supported GPU: llama.cpp's OpenCL backend runs on Qualcomm Adreno GPUs only")
    service, engine = make_service(tmp_path / "android", prefs=prefs, engine=engine, gpu_backend="OpenCL")
    panel = LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert panel.gpu_switch.visible and panel.gpu_switch.value is False and panel.gpu_switch.key == "local-gpu"
    assert panel.gpu_switch.label == "Use GPU (experimental)" and "Adreno" in panel.gpu_note.value
    assert "Q4_0" in panel.gpu_note.value
    panel.gpu_switch.value = True
    assert panel.save_gpu() is True and prefs.get(svc.GPU_KEY) is True
    # owner 2026-10-31: the switch adds no presets any more (the Custom rows replace them)
    assert presets.load_presets(prefs) == list(presets.DEFAULT_PRESETS)
    # a GPU failure shows on the card, the device line says CPU
    _chat(service)
    panel.render_engine(service.snapshot(), push=False)
    assert panel.gpu_status.visible and panel.gpu_status.value.startswith("GPU unavailable for fake-echo: no supported GPU")
    assert "Runs on CPU" in panel.engine_detail.value
    # iOS (Metal): the switch
    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    service, _engine = make_service(tmp_path / "ios", prefs=prefs, gpu_backend="Metal")
    panel = LocalModelsPanel(service=service, prefs=prefs)
    panel.build()
    assert panel.gpu_switch.visible and panel.gpu_switch.value is True and "Metal" in panel.gpu_note.value


@needs_flet
def test_gpu_hint_offers_the_q4_0_file_of_a_k_quant_model(tmp_path):
    """owner 2026-10-22: a model downloaded before "Use GPU" was switched on (the defaults are K-quants)
    still runs on the Adreno GPU, maybe slower than the CPU, and nothing failed to make it fall back: the
    engine card (and the Welcome page) say so. Owner 2026-10-31: the tap opens the Custom menu at that family
    and size with Q4_0 highlighted."""
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    light = next(p for p in presets.DEFAULT_PRESETS if p["name"] == "Light")
    q4_file = "Qwen3.5-2B-Q4_0.gguf"
    assert presets.is_gpu_quant(q4_file)
    assert presets.file_quant(light["file"]) == "Q4_K_XL" and not presets.is_gpu_quant(light["file"])

    def downloaded(service, stem, file):
        with open(service.registry.file_path(stem), "wb") as handle:  # the registry lists files on disk
            handle.write(b"GGUF")
        service.registry.add(stem, {"repo": light["repo"], "file": file, "revision": "r", "sha256": "0" * 64,
                                    "size": 4, "downloaded_at": 1})

    def panel_for(sub, backend="OpenCL", gpu_on=True, file=light["file"]):
        prefs = fakes.FakePrefs({svc.GPU_KEY: True} if gpu_on else {})
        service, _engine = make_service(tmp_path / sub, stems=(), prefs=prefs, gpu_backend=backend)
        downloaded(service, "light", file)
        current = {"model": "local/light"}
        panel = LocalModelsPanel(service=service, prefs=prefs, current_model=lambda: current["model"])
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
    # no hint: GPU off, a Q4_0 model, iOS (Metal runs the K-quants), a non-local model
    assert not panel_for("off", gpu_on=False)[0].gpu_swap_box.visible
    assert not panel_for("q4", file=q4_file)[0].gpu_swap_box.visible
    assert not panel_for("ios", backend="Metal")[0].gpu_swap_box.visible
    panel, _prefs, _service, current = panel_for("remote")
    current["model"] = "authgpt/gpt-6-luna"
    panel._render_gpu_swap(push=False)
    assert not panel.gpu_swap_box.visible


# ---- the smoke check: inert while off -------------------------------------------------------------------------

def test_smoke_check_fails_when_a_gpu_backend_registered_by_itself(monkeypatch):
    from glossarion_mobile.services.local_llm import selfcheck

    class _Fn:
        def __init__(self, fn):
            self.fn, self.restype, self.argtypes = fn, None, None

        def __call__(self, *a):
            return self.fn(*a)

    registered = {b"Metal"}
    libggml = types.SimpleNamespace(ggml_backend_reg_by_name=_Fn(lambda name: 1 if name in registered else 0))
    ext = types.SimpleNamespace(gpu_backend_name=lambda: "Metal", GPU_STATE={}, _LIBGGML=libggml)
    monkeypatch.setattr(selfcheck.importlib, "import_module", lambda name: ext)
    monkeypatch.setenv("GLOSSARION_PLATFORM", "ios")
    with pytest.raises(AssertionError, match=r"GPU backend \['Metal'\] registered although the GPU was never"):
        selfcheck.gpu_backend_check()
    registered.clear()
    assert selfcheck.gpu_backend_check() == {"backend": "Metal", "registered": [], "requested": False, "state": {}}
    # the user switched it on: registered is expected
    registered.add(b"Metal")
    ext.GPU_STATE = {"backend": "Metal", "device": "Apple A17 Pro GPU", "error": None}
    assert selfcheck.gpu_backend_check()["registered"] == ["Metal"]
    # the app and the build disagree (an Android arm64 build without libggml-opencl)
    monkeypatch.setenv("GLOSSARION_PLATFORM", "android")
    ext.gpu_backend_name = lambda: None
    monkeypatch.setattr(eng, "gpu_offer", lambda platform=None, machine=None: "OpenCL")
    with pytest.raises(AssertionError, match="offers GPU backend 'OpenCL', the engine build ships None"):
        selfcheck.gpu_backend_check()
