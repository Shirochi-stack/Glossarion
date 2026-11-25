"""The GPU for the on-device engine (``services/local_llm``): the "Run on" GPU modes (CPU + GPU with the GPU layers
setting, GPU only with every layer; offered only with a GPU backend), the automatic CPU fallback (GPU load failure, a GPU
answer failing, a crash while the GPU started), the device line, the RAM gate, ``LlamaEngine``'s layer placement and
its first-token check and the Q4_0 GPU presets. FakeEngine / a fake llama_cpp. The engine card's controls are tested in
test_local_npu_ui.py.
"""

from __future__ import annotations

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


def _gpu_service(tmp_path, engine=None, mode="cpu_gpu", backend="OpenCL", **kwargs):
    prefs = kwargs.pop("prefs", None) or fakes.FakePrefs({svc.RUN_MODE_KEY: mode} if mode else {})
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


def test_auto_runs_on_the_cpu_and_the_gpu_modes_need_a_backend(tmp_path):
    service, engine = make_service(tmp_path, gpu_backend="OpenCL")
    assert service.run_mode() == "auto" and not service.use_gpu() and service.gpu_note() is None
    assert service.run_mode_options() == ["auto", "cpu_gpu", "gpu", "cpu"]   # iOS 15+ / Android without the NPU
    assert service.run_mode_label("auto") == "Auto (CPU only)" and not service.gpu_layers_apply()
    _chat(service)
    assert engine.gpu_loads == [False] and service.snapshot().device == "CPU"
    # no backend (desktop, x86_64, iOS 13 / 14): no dropdown, a saved GPU mode runs on the CPU and cannot be chosen
    service, engine = make_service(tmp_path / "x", prefs=fakes.FakePrefs({svc.RUN_MODE_KEY: "gpu"}))
    assert service.gpu_backend is None and service.run_mode_options() == [] and not service.use_gpu()
    assert service.run_mode_shown() == "cpu" and service.save_run_mode("cpu_gpu") == "cpu"
    assert service.prefs.get(svc.RUN_MODE_KEY) == "gpu"                        # the saved value is kept
    _chat(service)
    assert engine.gpu_loads == [False]


# ---- GPU load, device line, setting changes ---------------------------------------------------------------

def test_cpu_gpu_loads_on_the_gpu_and_says_which_device(tmp_path):
    service, engine = _gpu_service(tmp_path, eng.FakeEngine(gpu_device="QUALCOMM Adreno(TM) 830 (OpenCL)"))
    assert service.use_gpu() and service.gpu_layers_apply()
    assert _chat(service) == "Echo: hi"
    assert engine.gpu_loads == [True] and engine.calls[-1]["gpu"] is True and engine.n_gpu_layers == 8
    snap = service.snapshot()
    assert snap.device == "GPU: QUALCOMM Adreno(TM) 830 (OpenCL)" and snap.gpu_note is None
    # the first answer ended: no crash marker left behind
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    _chat(service, text="again")
    assert engine.gpu_loads == [True]                         # no reload
    # CPU only: nothing is freed from the screen; the next request loads it on the CPU
    assert service.save_run_mode("cpu") == "cpu" and service.loaded_stem == "fake-echo"
    _chat(service)
    assert engine.gpu_loads == [True, False] and service.snapshot().device == "CPU"
    assert service.prefs.get(svc.RUN_MODE_KEY) == "cpu" and not service.use_gpu()


def test_gpu_only_offloads_every_layer_whatever_the_setting(tmp_path):
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "gpu", svc.GPU_LAYERS_KEY: 0})
    service, engine = _gpu_service(tmp_path, prefs=prefs)
    assert service.use_gpu() and not service.gpu_layers_apply()               # the slider is for CPU + GPU
    _chat(service)
    assert engine.gpu_loads == [True] and engine.n_gpu_layers == -1
    service.save_gpu_layers(4)
    _chat(service)
    assert engine.gpu_loads == [True, True] and engine.n_gpu_layers == -1     # the setting change reloaded it
    # a failure: the CPU, kept for that model
    engine.unload()
    failing, engine2 = _gpu_service(tmp_path / "f", eng.FakeEngine(gpu_error="out of GPU memory"), mode="gpu")
    _chat(failing)
    assert engine2.gpu_loads == [True, False] and failing.gpu_fallback()["reason"] == "out of GPU memory"
    assert failing.gpu_note() == "GPU unavailable for fake-echo: out of GPU memory — using CPU"


def test_gpu_failure_at_load_falls_back_to_the_cpu_and_stays_there(tmp_path):
    reason = "no supported GPU: llama.cpp's OpenCL backend runs on Qualcomm Adreno GPUs only"
    service, engine = _gpu_service(tmp_path, eng.FakeEngine(gpu_error=reason), stems=("a", "b"))
    assert _chat(service, "local/a") == "Echo: hi"          # the request still gets its answer
    assert engine.gpu_loads == [True, False] and not engine.on_gpu
    snap = service.snapshot()
    assert snap.device == "CPU" and snap.gpu_note == f"GPU unavailable for a: {reason} — using CPU"
    assert service.gpu_fallback() == {"stem": "a", "reason": reason, "layers": eng.DEFAULT_GPU_LAYERS}
    assert service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    _chat(service, "local/a", "again")
    assert engine.gpu_loads == [True, False]                  # stays on the CPU: no reload loop
    # another model gets a new GPU try
    _chat(service, "local/b")
    assert engine.gpu_loads == [True, False, True, False] and service.gpu_fallback()["stem"] == "b"
    # saving a run mode forgets the fallback
    service.save_run_mode("cpu")
    assert service.gpu_fallback() is None and service.gpu_note() is None
    service.save_run_mode("cpu_gpu")
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
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu_gpu", svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo", "at": 1}})
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
    # a marker left while the run mode shows Auto still counts where the build has a GPU (a model's "GPU" row loads
    # on the GPU under Auto): recorded, so the next load does not crash the same way
    prefs = fakes.FakePrefs({svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo"}})
    service, _engine = make_service(tmp_path / "y", prefs=prefs, gpu_backend="Metal")
    assert service._recover_gpu_attempt() == "fake-echo" and service.gpu_fallback()["stem"] == "fake-echo"
    # without a GPU backend it is dropped silently
    prefs = fakes.FakePrefs({svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo"}})
    service, _engine = make_service(tmp_path / "z", prefs=prefs)
    assert service._recover_gpu_attempt() == "fake-echo" and service.gpu_fallback() is None


def test_a_gpu_preset_row_is_cpu_gpu_under_any_run_mode(tmp_path):
    service, engine = _gpu_service(tmp_path, mode="auto")
    presets.save_presets(service.prefs, [{"name": "row", "repo": "t/fake-echo", "file": "fake-echo.gguf", "quant": "",
                                          "note": "", "n_ctx": 0, "backend": "gpu"}])
    _chat(service)
    assert engine.gpu_loads == [True] and engine.n_gpu_layers == 8
    assert service._placement("fake-echo") == ("gpu", 8, None)
    service.save_run_mode("cpu")
    assert service._placement("fake-echo") == ("gpu", 8, None)               # the row wins over the dropdown


def test_a_gpu_preset_row_shows_the_slider_the_gpu_line_and_the_hint_state(tmp_path):
    # Auto on a GPU phone, the model's row says "GPU": the screens follow the model, not the dropdown
    service, engine = _gpu_service(tmp_path, eng.FakeEngine(gpu_error="out of GPU memory"), mode="auto")
    presets.save_presets(service.prefs, [{"name": "row", "repo": "t/fake-echo", "file": "fake-echo.gguf", "quant": "",
                                          "note": "", "n_ctx": 0, "backend": "gpu"}])
    assert not service.use_gpu() and not service.gpu_layers_apply()            # "Run on" alone: Auto
    assert service.model_run_mode("fake-echo") == "cpu_gpu"
    assert service.use_gpu("fake-echo") and service.gpu_layers_apply("fake-echo")
    _chat(service)
    assert engine.gpu_loads == [True, False]
    assert service.gpu_note() == "GPU unavailable for fake-echo: out of GPU memory — using CPU"
    # the other way round: CPU + GPU in the dropdown, a row that says "CPU" uses no slider
    presets.save_presets(service.prefs, [{"name": "row", "repo": "t/fake-echo", "file": "fake-echo.gguf", "quant": "",
                                          "note": "", "n_ctx": 0, "backend": "cpu"}])
    service.save_run_mode("cpu_gpu")
    assert service.gpu_layers_apply() and not service.gpu_layers_apply("fake-echo")
    assert not service.use_gpu("fake-echo")


def test_gpu_only_failing_leaves_cpu_and_gpu_on_the_gpu(tmp_path):
    # GPU only (every layer) failed for the model: CPU + GPU's 8 layers still run on the GPU, with no GPU line
    record = {"stem": "fake-echo", "reason": "out of device memory", "layers": -1}
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu_gpu", svc.GPU_FALLBACK_KEY: dict(record)})
    service, engine = _gpu_service(tmp_path, prefs=prefs)
    assert service._placement("fake-echo") == ("gpu", 8, None) and service.gpu_note() is None
    _chat(service)
    assert engine.gpu_loads == [True] and engine.n_gpu_layers == 8
    # GPU only itself stays on the CPU and says why
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "gpu", svc.GPU_FALLBACK_KEY: dict(record)})
    service, _engine = _gpu_service(tmp_path / "g", prefs=prefs)
    assert service._placement("fake-echo") == ("cpu", 0, None)
    assert service.gpu_note() == "GPU unavailable for fake-echo: out of device memory — using CPU"
    # 8 layers failing keeps 8 and more off; an older record without a count keeps every GPU load off
    for stored, mode, placed in (({**record, "layers": 8}, "cpu_gpu", ("cpu", 0, None)),
                                 ({**record, "layers": 8}, "gpu", ("cpu", 0, None)),
                                 ({"stem": "fake-echo", "reason": "x"}, "cpu_gpu", ("cpu", 0, None))):
        prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: mode, svc.GPU_FALLBACK_KEY: stored})
        service, _engine = _gpu_service(tmp_path / f"{mode}-{stored.get('layers')}", prefs=prefs)
        assert service._placement("fake-echo") == placed
    # a crash in GPU only is recorded with its count: CPU + GPU is left alone
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu_gpu",
                             svc.GPU_ATTEMPT_KEY: {"stem": "fake-echo", "at": 1, "layers": -1}})
    service, _engine = _gpu_service(tmp_path / "c", prefs=prefs)
    assert service._recover_gpu_attempt() == "fake-echo"
    assert service.gpu_fallback() == {"stem": "fake-echo", "reason": svc.CRASH_REASON, "layers": -1}
    assert service._placement("fake-echo") == ("gpu", 8, None)


def test_the_cpu_and_gpu_note_names_q4_0_only_for_opencl(tmp_path):
    android, _engine = _gpu_service(tmp_path)
    assert "Fastest with Q4_0." in android.run_mode_note("cpu_gpu")
    ios, _engine = _gpu_service(tmp_path / "ios", backend="Metal")
    assert ios.run_mode_note("cpu_gpu") == ("The GPU layers below run on the GPU, the rest on the CPU. A failure uses "
                                            "the CPU.")


def test_ram_gate_counts_the_gpu(tmp_path):
    memory = lambda: (4 * GB, None)  # an iPhone with 4 GB: iOS refuses above half
    service, _engine = _gpu_service(tmp_path, backend="Metal", memory=memory)
    service.platform = "ios"
    service.save_context(4096)  # isolate GPU overhead at a saved lower context
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
    assert (gpu.kwargs["n_gpu_layers"], gpu.kwargs["split_mode"], gpu.kwargs["main_gpu"]) == (eng.DEFAULT_GPU_LAYERS, 0, 0)
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
    # the stock (Unsloth) repos of the earlier defaults: the uncensored defaults (owner 2026-11-25) moved away
    old_repos = {repo for _name, repo, _file in presets.OLD_STOCK_PRESETS}
    assert [p["file"] for p in presets.GPU_PRESETS] == [
        "Qwen3.5-0.8B-Q4_0.gguf", "Qwen3.5-2B-Q4_0.gguf", "Qwen3.5-4B-Q4_0.gguf", "Qwen3.5-9B-Q4_0.gguf"]
    assert [p["size"] for p in presets.GPU_PRESETS] == [507154688, 1214873856, 2583221408, 5379417312]
    for row in presets.GPU_PRESETS:
        assert presets._clean(row) == row                        # a valid preset row, nothing dropped
        assert presets.custom_size_for(row["repo"]) is None and row["mmproj"] == "mmproj-F16.gguf"
        assert "mmproj_repo" not in row and presets.projector_repo(row) == row["repo"]  # their own projector
        if row["repo"] == "unsloth/Qwen3.5-9B-GGUF":
            assert row["mmproj_sha256"] == "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"
            continue
        assert row["repo"] in old_repos and "Q4_0" in row["name"]
    # Unsloth's 0.8B / 2B projectors are the ones the uncensored Garbage / Light use from there
    for row, default in zip(presets.GPU_PRESETS[:2], presets.DEFAULT_PRESETS[:2]):
        assert default["mmproj_repo"] == row["repo"] and default["mmproj_sha256"] == row["mmproj_sha256"]
    # the defaults (CPU, iOS) are unchanged and hold no Q4_0 file
    assert not any("Q4_0" in p["file"] for p in presets.DEFAULT_PRESETS)


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


def test_gpu_layer_count_persists_reloads_and_all_is_explicit(tmp_path):
    service, engine = _gpu_service(tmp_path)
    assert service.gpu_layers() == 8
    _chat(service)
    assert engine.n_gpu_layers == 8
    assert service.save_gpu_layers(20) == 20 and service.loaded_stem is None
    _chat(service)
    assert engine.n_gpu_layers == 20
    loads = len(engine.loads)
    service.save_gpu_layers(20)
    _chat(service)
    assert len(engine.loads) == loads
    assert svc.LocalEngineService(prefs=service.prefs, registry=service.registry, watch=False).gpu_layers() == 20
    service.save_gpu_layers(-1)
    _chat(service)
    assert engine.n_gpu_layers == -1
    service.save_gpu_layers(0)
    _chat(service)
    assert not engine.on_gpu and engine.n_gpu_layers == 0
    for invalid in (-2, 999, "bad", None):
        assert service.save_gpu_layers(invalid) == 0
    service.save_gpu_layers(4)
    _chat(service)
    assert engine.on_gpu and engine.n_gpu_layers == 4


def test_gpu_layer_change_during_generation_waits_for_next_request(tmp_path):
    service, engine = _gpu_service(tmp_path)
    run = service.chat("local/fake-echo", {"messages": [{"role": "user", "content": "hi"}]}, lambda: False)
    with run:
        assert engine.n_gpu_layers == 8
        service.save_gpu_layers(12)
        assert engine.loaded and engine.n_gpu_layers == 8
        assert "".join(run.deltas) == "Echo: hi"
    _chat(service)
    assert engine.n_gpu_layers == 12


@pytest.mark.parametrize("count", [0, 5, 24, -1])
def test_native_gpu_placement_uses_selected_count(fake_llama, count):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, gpu=True, n_gpu_layers=count)
    assert _FakeLlama.made[-1].kwargs["n_gpu_layers"] == count
    assert engine.on_gpu == (count != 0)
    assert bool(_FakeLlama.made[-1].evals) == (count != 0)


def test_zero_selected_during_gpu_load_is_applied_on_next_request(tmp_path):
    service, engine = _gpu_service(tmp_path)
    original = engine.load
    def load(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs.get("gpu"):
            service.save_gpu_layers(0)
    engine.load = load
    assert service.load_blocking("fake-echo") == (True, None)
    assert engine.on_gpu
    _chat(service)
    assert not engine.on_gpu and engine.gpu_loads == [True, False]
