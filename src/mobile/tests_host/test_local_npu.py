"""NPU (Qualcomm Hexagon) for the on-device engine, app side (``services/local_llm``): the backend choice per model
(auto / npu / gpu / cpu x the phone's NPU check x "Use NPU" x the quant x a saved fallback), every row of the
fallback matrix, the two crash markers (the attempt marker from an NPU load to its first answer's end, the in-flight
marker around every NPU answer) and their recovery at start, "Try the NPU again", the watchdog, ``device_index`` for
the GPU and the NPU, the wheel's NPU functions through ``engine.npu_state`` (and a wheel without them), the presets'
backend fields, a model added from Hugging Face running on the NPU under Auto, the diagnostics block and the
self-check's shared markers. Fakes only: FakeEngine, FakePrefs, a fake ``register_npu_backend``; no native NPU.
"""

from __future__ import annotations

import itertools
import threading
import time
import types

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import presets
from glossarion_mobile.services.local_llm import selfcheck
from glossarion_mobile.services.local_llm import service as svc
from test_local_llm_gpu import _FakeLlama, fake_llama  # noqa: F401 - the fixture
from test_local_llm_server import make_service

Q40 = "m-Q4_0"  # "best"
QK = "m-Q4_K_M"  # "ok"
IQ = "m-IQ4_XS"  # "cpu"
VERSION = "9.16.0+abc/llama-0.3.37"


def passing_state(**changes) -> dict:
    state = {"backend": "HTP", "device": "HTP0", "arch": 81, "error": "",
             "steps": [{"name": name, "ok": True, "detail": f"{name} ok"} for name in eng.NPU_STEPS]}
    state.update(changes)
    return state


def failing_state(step: str, detail: str = "nope") -> dict:
    state = passing_state(error=detail)
    for entry in state["steps"]:
        if entry["name"] == step:
            entry.update(ok=False, detail=detail)
    return state


class Probe:
    """A fake ``register_npu_backend`` (``engine.npu_state``'s place): records its calls."""

    def __init__(self, state=None, delay=0.0, error=None):
        self.state = state if state is not None else passing_state()
        self.delay = delay
        self.error = error
        self.calls: list = []

    def __call__(self, skel_dir, options):
        self.calls.append((skel_dir, dict(options)))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.state)


def npu_service(tmp_path, *, stems=(Q40,), probe=None, prefs=None, gpu=False, engine=None, version=VERSION,
                **kwargs):
    prefs = prefs if prefs is not None else fakes.FakePrefs({svc.GPU_KEY: True} if gpu else {})
    probe = probe if probe is not None else Probe()
    service, engine = make_service(tmp_path, stems=stems, prefs=prefs, engine=engine, npu_probe=probe,
                                   app_version=version, gpu_backend="OpenCL", **kwargs)
    return service, engine, probe


def chat(service, stem=Q40, text="hi"):
    run = service.chat(f"local/{stem}", {"messages": [{"role": "user", "content": text}]}, lambda: False)
    with run:
        return "".join(run.deltas)


def set_backend(service, stem, backend, **extra):
    """A preset row for ``stem`` (make_service's repo / file) with this backend."""
    rows = [r for r in presets.load_presets(service.prefs) if r["repo"] != f"t/{stem}"]
    row = {"name": f"row {stem}", "repo": f"t/{stem}", "file": f"{stem}.gguf", "quant": "", "note": "", "n_ctx": 0,
           **extra}
    if backend != "auto":
        row["backend"] = backend
    presets.save_presets(service.prefs, [*rows, row])


# ---- the quant kinds and the preset rows ----------------------------------------------------------------------

@pytest.mark.parametrize("file,kind", [
    ("Qwen3.5-2B-Q4_0.gguf", "best"), ("x-q4_0.gguf", "best"),
    ("Qwen3.5-2B-Q4_K_M.gguf", "ok"), ("Qwen3.5-2B-UD-Q4_K_XL.gguf", "ok"), ("x-Q2_K.gguf", "ok"),
    ("x-Q3_K_S.gguf", "ok"), ("x-Q5_K_M.gguf", "ok"), ("x-Q6_K.gguf", "ok"), ("x-Q8_0.gguf", "ok"),
    ("x-Q4_1.gguf", "ok"), ("x-Q5_0.gguf", "ok"), ("x-IQ4_NL.gguf", "ok"), ("x-IQ2_XXS.gguf", "ok"),
    ("x-BF16.gguf", "ok"), ("x-F16.gguf", "ok"), ("gpt-oss-20b-MXFP4.gguf", "ok"),
    ("x-IQ4_XS.gguf", "cpu"), ("x-IQ3_XXS.gguf", "cpu"), ("x-IQ3_M.gguf", "cpu"), ("x-IQ2_M.gguf", "cpu"),
    ("x-IQ1_S.gguf", "cpu"), ("odd-name.gguf", "cpu"), (None, "cpu"),
])
def test_npu_quant_kind(file, kind):
    assert presets.npu_quant_kind(file) == kind


def test_npu_presets_reuse_the_verified_q4_0_rows():
    assert [row["name"] for row in presets.NPU_PRESETS] == ["Garbage · NPU", "Light · NPU", "Medium · NPU"]
    for npu_row, gpu_row in zip(presets.NPU_PRESETS, presets.GPU_PRESETS):
        for key in ("repo", "file", "size", "sha256", "mmproj", "mmproj_size", "mmproj_sha256"):
            assert npu_row[key] == gpu_row[key]
        assert npu_row["backend"] == "npu" and presets.npu_quant_kind(npu_row["file"]) == "best"
        assert presets._clean(npu_row)["backend"] == "npu"
    assert all(row["size"] < presets.NPU_SESSION_BYTES for row in presets.NPU_PRESETS)  # 9B is Custom only
    family = presets.custom_family("npu")
    assert family is presets.NPU_FAMILY and family["backend"] == "npu"
    repos = [s["repo"] for f in presets.CUSTOM_FAMILIES for s in f["sizes"]]
    assert [s["repo"] for s in family["sizes"]] == repos
    nine = next(s for s in family["sizes"] if s["repo"] == "unsloth/Qwen3.5-9B-GGUF")
    assert nine["npu_note"] == "over the NPU's 3.5 GB: slower" and nine["group"] == "Qwen 3.5"
    assert all("npu_note" not in s for f in presets.CUSTOM_FAMILIES for s in f["sizes"])  # the table is unchanged


def test_rows_carry_backend_and_npu_layers_and_auto_is_not_stored():
    base = {"name": "A", "repo": "r/a", "file": "a-Q4_0.gguf"}
    assert "backend" not in presets._clean(base) and "npu_layers" not in presets._clean(base)
    assert "backend" not in presets._clean({**base, "backend": "auto"})
    assert "backend" not in presets._clean({**base, "backend": "tpu"})
    clean = presets._clean({**base, "backend": "NPU", "npu_layers": "12"})
    assert clean["backend"] == "npu" and clean["npu_layers"] == 12
    assert presets._clean({**base, "npu_layers": 0}).get("npu_layers") is None
    hf_file = types.SimpleNamespace(path="Qwen3.5-2B-Q4_0.gguf", size=10, sha256=None)
    size = presets.CUSTOM_FAMILIES[0]["sizes"][1]
    row = presets.custom_row(size, hf_file, backend="npu")
    assert row["backend"] == "npu" and row["name"] == "Qwen 3.5 2B · Q4_0 · NPU" and "NPU" in row["note"]
    assert "backend" not in presets.custom_row(size, hf_file)


def test_validate_preset_backend_and_layers():
    rows = [{"name": "Old", "repo": "r/a", "file": "a-Q4_0.gguf", "backend": "gpu", "npu_layers": 9}]
    clean, error = presets.validate_preset({"name": "Old", "repo": "r/a", "file": "a-Q4_0.gguf", "backend": "npu",
                                            "npu_layers": "20"}, rows, 0)
    assert error is None and clean["backend"] == "npu" and clean["npu_layers"] == 20
    clean, _ = presets.validate_preset({"name": "Old", "repo": "r/a", "backend": "auto", "npu_layers": ""}, rows, 0)
    assert "backend" not in clean and "npu_layers" not in clean          # Auto, all layers
    clean, _ = presets.validate_preset({"name": "Old", "repo": "r/a"}, rows, 0)  # the fields were hidden: kept
    assert clean["backend"] == "gpu" and clean["npu_layers"] == 9
    assert presets.validate_preset({"name": "N", "repo": "r/a", "backend": "tpu"}, rows)[1].startswith("Backend")
    assert "NPU layers" in presets.validate_preset({"name": "N", "repo": "r/a", "npu_layers": "x"}, rows)[1]
    assert "NPU layers" in presets.validate_preset({"name": "N", "repo": "r/a", "npu_layers": "0.5"}, rows)[1]
    assert presets.validate_preset({"name": "N", "repo": "r/a", "npu_layers": "all"}, rows)[0].get("npu_layers") is None


# ---- the engine side: the wheel's functions ---------------------------------------------------------------------

def _ext(monkeypatch, **functions):
    ext = types.SimpleNamespace(**functions)
    monkeypatch.setattr(eng, "_extensions", lambda: ext)
    monkeypatch.setattr(eng, "capture_hex_log", lambda: False)
    return ext


def test_a_wheel_without_the_npu_functions_offers_nothing(monkeypatch):
    monkeypatch.setattr(eng, "_extensions", lambda: None)
    state = eng.npu_state("/skel")
    assert not eng.npu_passed(state) and eng.npu_failure(state)[0] == "shipped"
    _ext(monkeypatch, register_gpu_backend=lambda: {})                         # an older wheel
    state = eng.npu_state("/skel")
    assert eng.npu_failure(state) == ("shipped", eng.NO_NPU)
    assert eng.device_index("HTP") == -1 and eng.device_index("OpenCL", missing=0) == 0
    _ext(monkeypatch, npu_backend_shipped=lambda: False, register_npu_backend=lambda *a: pytest.fail("loaded"))
    assert eng.npu_failure(eng.npu_state("/skel"))[0] == "shipped"           # iOS / x86_64 wheel


def test_npu_state_calls_register_with_the_skel_dir_and_options(monkeypatch):
    calls = []

    def register(skel_dir, options=None):
        calls.append((skel_dir, options))
        return passing_state(arch="81")

    _ext(monkeypatch, npu_backend_shipped=lambda: True, register_npu_backend=register,
         device_index=lambda name: {"HTP": 1, "OpenCL": 0}.get(name, -1))
    state = eng.npu_state("/data/htp/x", {eng.NPU_POWER_ENV: "balanced"})
    assert calls == [("/data/htp/x", {eng.NPU_POWER_ENV: "balanced"})]
    assert state["arch"] == 81 and eng.npu_passed(state) and eng.npu_label(state) == "Hexagon v81 (HTP0)"
    assert eng.device_index("HTP") == 1 and eng.device_index("OpenCL") == 0 and eng.device_index("Metal") == -1


@pytest.mark.parametrize("step", eng.NPU_STEPS)
def test_every_failing_step_is_named(step):
    assert eng.npu_failure(failing_state(step, "why")) == (step, "why")


def test_arch_registry_and_missing_steps():
    assert eng.npu_failure(passing_state(arch=None)) == ("arch", "the HTP arch is unknown")
    assert eng.npu_failure(passing_state(arch=69))[0] == "arch"              # never upstream's silent v73 guess
    assert eng.npu_failure(passing_state(device=""))[0] == "registry"
    state = passing_state()
    state["steps"] = state["steps"][:3]
    assert eng.npu_failure(state)[0] == "skeletons"
    assert not eng.npu_passed(None) and not eng.npu_passed({})


def test_npu_platform_is_android_arm64_only():
    assert eng.npu_platform("android", "aarch64") and eng.npu_platform("android", "arm64-v8a")
    assert not eng.npu_platform("android", "x86_64") and not eng.npu_platform("ios", "arm64")
    assert not eng.npu_platform("desktop", "aarch64")


def test_device_index_places_gpu_and_npu_loads(fake_llama, monkeypatch):  # noqa: F811
    _ext(monkeypatch, device_index=lambda name: {"HTP": 0, "OpenCL": 1}.get(name, -1))
    eng._NPU_STATE.update(passing_state())
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, gpu=True, n_gpu_layers=8)
    assert _FakeLlama.made[-1].kwargs["main_gpu"] == 1 and engine.on_gpu and not engine.on_npu
    engine.load("m.gguf", 2048, 4, npu=True, n_gpu_layers=-1)
    kwargs = _FakeLlama.made[-1].kwargs
    assert (kwargs["main_gpu"], kwargs["n_gpu_layers"], kwargs["split_mode"]) == (0, -1, eng.SPLIT_MODE_NONE)
    assert engine.on_npu and not engine.on_gpu and engine.backend == "NPU"
    assert engine.device == "NPU: Hexagon v81 (HTP0)" and _FakeLlama.made[-1].evals  # the first-token check ran
    engine.load("m.gguf", 2048, 4, npu=True, n_gpu_layers=12)
    assert _FakeLlama.made[-1].kwargs["n_gpu_layers"] == 12
    engine.load("m.gguf", 2048, 4)
    assert _FakeLlama.made[-1].kwargs["main_gpu"] == -1 and engine.backend == "CPU"


def test_npu_load_failures_raise_npu_unavailable(fake_llama, monkeypatch):  # noqa: F811
    _ext(monkeypatch)                                                          # no device_index: no NPU device
    engine = eng.LlamaEngine()
    with pytest.raises(eng.NpuUnavailable):
        engine.load("m.gguf", 2048, 4, npu=True)
    engine.load("m.gguf", 2048, 4, gpu=True)                                   # the GPU keeps main_gpu 0
    assert _FakeLlama.made[-1].kwargs["main_gpu"] == 0
    _ext(monkeypatch, device_index=lambda name: {"HTP": 1}.get(name, -1))
    with pytest.raises(eng.GpuUnavailable, match="device list"):              # never lands on the NPU
        engine.load("m.gguf", 2048, 4, gpu=True)
    _FakeLlama.logits = (float("nan"),) * 4
    with pytest.raises(eng.NpuUnavailable, match="first token on the NPU"):
        engine.load("m.gguf", 2048, 4, npu=True)
    assert not engine.loaded and _FakeLlama.made[-1].closed
    _FakeLlama.fail = "the session failed"
    with pytest.raises(eng.NpuUnavailable, match="did not load on the NPU"):
        engine.load("m.gguf", 2048, 4, npu=True)


def test_the_hex_log_hook_keeps_ggml_hex_lines(monkeypatch):
    import ctypes

    printed = []
    original = lambda level, text, user: printed.append(text)  # noqa: E731
    installed = []
    llama_cpp = types.SimpleNamespace(llama_log_callback=lambda fn: fn,
                                      llama_log_set=lambda cb, user: installed.append(cb))
    modules = {"llama_cpp": llama_cpp, "llama_cpp._logger": types.SimpleNamespace(llama_log_callback=original)}
    monkeypatch.setattr(eng.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(eng, "_HEX_HOOK", [])
    monkeypatch.setattr(eng, "_HEX_PENDING", [""])
    eng._HEX_LOG.clear()
    assert eng.capture_hex_log() and len(installed) == 1 and eng.capture_hex_log() and len(installed) == 1
    hook = installed[0]
    hook(1, b"ggml-hex: Hexagon Arch version v81, DMA64 enabled\n", ctypes.c_void_p(0))
    hook(1, b"llama_model_load: something else\nggml-hex: HTP0 new ses", None)
    hook(5, b"sion\n", None)
    assert eng.hex_log_lines() == ["ggml-hex: Hexagon Arch version v81, DMA64 enabled", "ggml-hex: HTP0 new session"]
    assert len(printed) == 3                                                   # everything still goes where it went
    for i in range(30):
        hook(1, f"ggml-hex: line {i}\n".encode(), None)
    assert len(eng.hex_log_lines()) == eng.HEX_LOG_LINES and eng.hex_log_lines()[-1] == "ggml-hex: line 29"


# ---- the backend choice -------------------------------------------------------------------------------------------

MATRIX = list(itertools.product(("auto", "npu", "gpu", "cpu"), (True, False), (True, False), ("best", "ok", "cpu"),
                                (False, True), (True, False)))


def _expected(choice, passed, use_npu, kind, fallback, use_gpu):
    if choice == "cpu":
        return "cpu"
    if passed and not fallback and (choice == "npu" or (choice == "auto" and use_npu and kind != "cpu")):
        return "npu"
    if choice == "gpu":
        return "gpu"
    return "gpu" if use_gpu else "cpu"


@pytest.mark.parametrize("choice,passed,use_npu,kind,fallback,use_gpu", MATRIX)
def test_backend_choice_matrix(tmp_path, choice, passed, use_npu, kind, fallback, use_gpu):
    stem = {"best": Q40, "ok": QK, "cpu": IQ}[kind]
    prefs = fakes.FakePrefs({svc.GPU_KEY: use_gpu, svc.NPU_KEY: use_npu})
    probe = Probe(passing_state() if passed else failing_state("session", "unsigned PD denied"))
    service, _engine, _probe = npu_service(tmp_path, stems=(stem,), probe=probe, prefs=prefs)
    set_backend(service, stem, choice)
    if fallback:
        service._npu_failed(stem, "an earlier failure")
    service.run_npu_check()
    assert service._placement(stem)[0] == _expected(choice, passed, use_npu, kind, fallback, use_gpu)


def test_the_choice_reaches_the_engine_and_the_device_line(tmp_path):
    service, engine, probe = npu_service(tmp_path, stems=(Q40, IQ), gpu=True)
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["npu"] and engine.npu_layers == [-1] and engine.calls[-1]["npu"] is True
    snap = service.snapshot()
    assert snap.backend == "NPU" and snap.device == "NPU: Hexagon v81 (HTP0)" and snap.npu_note is None
    assert service.last_load["backend"] == "NPU" and service.last_load["stem"] == Q40
    assert service.health(f"local/{Q40}")["backend"] == "NPU"
    assert len(probe.calls) == 1 and probe.calls[0][1] == {eng.NPU_POWER_ENV: "burst"}
    assert probe.calls[0][0].replace("\\", "/").endswith("htp/9.16.0_abc_llama-0.3.37")
    chat(service, text="again")
    assert engine.backends == ["npu"]                                          # no reload
    chat(service, IQ)                                                          # IQ4_XS: Auto skips the NPU
    assert engine.backends == ["npu", "gpu"] and service.snapshot().backend == "GPU"
    set_backend(service, Q40, "npu", npu_layers=10)
    chat(service)
    assert engine.backends[-1] == "npu" and engine.npu_layers[-1] == 10
    set_backend(service, Q40, "cpu")
    chat(service)
    assert engine.backends[-1] == "cpu" and len(probe.calls) == 1             # the check runs once per process


def test_use_npu_switch_and_power_mode(tmp_path):
    service, engine, probe = npu_service(tmp_path)
    assert service.use_npu() and service.npu_power() == "burst"
    chat(service)
    assert service.save_use_npu(False) is False and service.loaded_stem is None  # the idle NPU model is freed
    chat(service)
    assert engine.backends == ["npu", "cpu"]
    service.save_use_npu(True)
    assert service.save_npu_power("balanced") == "balanced" and service.npu_power_pending()
    assert service.save_npu_power("turbo") == "balanced"
    assert service.npu_options() == {eng.NPU_POWER_ENV: "balanced"}
    assert set(eng.NPU_POWER_MODES) >= {"burst", "balanced", "power_saver"} and eng.NPU_DEFAULT_POWER == "burst"


def test_the_parallel_child_never_uses_the_npu(tmp_path):
    service, _engine, _probe = npu_service(tmp_path)
    child = service._parallel._secondary()
    assert child._npu_probe is None and not child.npu_offered()
    service.run_npu_check()
    assert child._placement(Q40)[0] == "cpu" and service._placement(Q40)[0] == "npu"


def test_a_model_added_from_hugging_face_runs_on_the_npu_under_auto(tmp_path):
    service, engine, _probe = npu_service(tmp_path, stems=())
    entry_stem = "Qwen3.5-2B-Q4_0"
    import os

    os.makedirs(service.registry.root, exist_ok=True)
    with open(service.registry.file_path(entry_stem), "wb") as handle:
        handle.write(b"GGUF")
    service.registry.add(entry_stem, {"repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_0.gguf",
                                      "revision": "r", "sha256": "0" * 64, "size": 4, "downloaded_at": 1})
    # "Save as a preset" of Add from Hugging Face writes this row: no backend, so Auto
    presets.save_presets(service.prefs, [{"name": entry_stem, "repo": "unsloth/Qwen3.5-2B-GGUF",
                                          "file": "Qwen3.5-2B-Q4_0.gguf", "quant": "", "note": "x", "n_ctx": 0}])
    assert service.backend_choice(entry_stem) == "auto"
    chat(service, entry_stem)
    assert engine.backends == ["npu"]
    presets.save_presets(service.prefs, [])                                    # without a preset row: Auto too
    service.unload()
    chat(service, entry_stem)
    assert engine.backends == ["npu", "npu"]


# ---- the fallback matrix ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("step", ["libcdsprpc", "arch", "skeletons", "registry", "session"])
def test_a_failing_check_hides_the_npu(tmp_path, step):
    service, engine, _probe = npu_service(tmp_path, probe=Probe(failing_state(step, f"{step} failed")), gpu=True)
    service.run_npu_check()
    assert not service.npu_offered() and service.npu_failure() == (step, f"{step} failed")
    chat(service)
    assert engine.backends == ["gpu"] and service.snapshot().npu_note is None


def test_no_npu_build_offers_nothing_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARION_PLATFORM", raising=False)
    service, engine = make_service(tmp_path, stems=(Q40,), prefs=fakes.FakePrefs({svc.GPU_KEY: True}),
                                   gpu_backend="OpenCL")
    assert service._npu_probe is None and not service.npu_offered() and not service.npu_possible()
    assert service.run_npu_check()["steps"][0]["name"] == "shipped"
    set_backend(service, Q40, "npu")                                           # even a saved NPU row
    chat(service)
    assert engine.backends == ["gpu"] and service.snapshot().npu_note is None
    assert "backend" not in service.health(f"local/{Q40}")
    assert not any(k.startswith("local_llm_npu") for k in service.prefs.data)  # no NPU state written


def test_a_failing_load_falls_back_and_keeps_the_reason(tmp_path):
    engine = eng.FakeEngine(npu_error="the first token on the NPU failed (12 of 248320 logits are NaN)")
    service, engine, _probe = npu_service(tmp_path, engine=engine, gpu=True)
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["npu", "gpu"] and service.snapshot().backend == "GPU"
    fallback = service.npu_fallback(Q40)
    assert fallback["reason"].startswith("the first token on the NPU failed")
    note = service.snapshot().npu_note
    assert note == f"NPU unavailable for {Q40}: {fallback['reason']}. Using GPU." and "—" not in note
    assert service.prefs.get(svc.NPU_ATTEMPT_KEY) is None
    chat(service, text="again")
    assert engine.backends == ["npu", "gpu"]                                   # no retry loop
    service.save_gpu(False)
    service.unload()
    chat(service)
    assert engine.backends[-1] == "cpu" and service.npu_note().endswith("Using CPU.")


def test_a_failing_npu_answer_is_answered_on_the_gpu(tmp_path):
    engine = eng.FakeEngine(npu_chat_error="ggml-hex: graph-compute failed")
    service, engine, _probe = npu_service(tmp_path, engine=engine, gpu=True)
    assert chat(service) == "Echo: hi"
    assert [c["npu"] for c in engine.calls] == [True, False] and engine.backends == ["npu", "gpu"]
    assert "the NPU failed while answering (ggml-hex: graph-compute failed)" in service.npu_fallback(Q40)["reason"]
    assert service.prefs.get(svc.NPU_INFLIGHT_KEY) is None and not service.gate.busy


def test_the_fallback_lasts_until_file_backend_or_version_change_or_try_again(tmp_path):
    engine = eng.FakeEngine(npu_error="boom")
    prefs = fakes.FakePrefs()
    service, engine, _probe = npu_service(tmp_path, engine=engine, prefs=prefs)
    chat(service)
    assert service.npu_fallback(Q40)
    engine.npu_error = None
    set_backend(service, Q40, "npu")                                           # the backend choice changed
    assert service.npu_fallback(Q40) is None
    set_backend(service, Q40, "auto")
    assert service.npu_fallback(Q40)                                           # back to the choice it failed under
    entry = service.registry.load()[Q40]
    service.registry.add(Q40, {**entry, "file": "m-Q4_0-v2.gguf"})             # another file
    assert service.npu_fallback(Q40) is None
    service.registry.add(Q40, entry)
    assert service.npu_fallback(Q40)
    newer, _e, _p = npu_service(tmp_path / "n", prefs=prefs, version="9.16.1/llama-0.3.37")
    newer.registry = service.registry
    assert newer.npu_fallback(Q40) is None                                     # a new app version tries again
    assert service.retry_npu(Q40) and service.npu_fallback(Q40) is None and not service.retry_npu(Q40)
    service.unload()
    chat(service)
    assert engine.backends[-1] == "npu"


# ---- the markers --------------------------------------------------------------------------------------------------

def test_attempt_marker_lives_while_the_npu_load_runs(tmp_path):
    service, engine, _probe = npu_service(tmp_path)
    seen = []
    real_load = engine.load

    def watched_load(*args, **kwargs):
        seen.append(service.prefs.get(svc.NPU_ATTEMPT_KEY))
        return real_load(*args, **kwargs)

    engine.load = watched_load
    assert service.load_blocking(Q40) == (True, None) and engine.on_npu
    assert seen[0]["stem"] == Q40                                              # set before the native load
    # the load ran its first token on the DSP: an idle loaded model leaves no marker (answers have their own)
    assert service.prefs.get(svc.NPU_ATTEMPT_KEY) is None
    chat(service)
    assert service.prefs.get(svc.NPU_ATTEMPT_KEY) is None


def test_npu_markers_are_on_disk_before_the_native_call(tmp_path):
    """``Prefs.set`` only schedules a debounced save: a DSP abort right after it would lose the marker."""
    import json

    from glossarion_mobile.state.prefs import Prefs

    path = tmp_path / "mobile_state.json"
    prefs = Prefs(path, debounce=60.0)
    prefs.load()

    def on_disk(key):
        return json.loads(path.read_text(encoding="utf-8")).get(key) if path.exists() else None

    disk: dict = {}

    class Watching(Probe):
        def __call__(self, skel_dir, options):
            disk["check"] = on_disk(svc.NPU_CHECK_ATTEMPT_KEY)
            return super().__call__(skel_dir, options)

    def reply(messages):
        disk["inflight"] = on_disk(svc.NPU_INFLIGHT_KEY)
        return "fine"

    engine = eng.FakeEngine(reply=reply)
    service, engine, _probe = npu_service(tmp_path / "s", prefs=prefs, probe=Watching(), engine=engine)
    real_load = engine.load

    def watched_load(*args, **kwargs):
        disk["attempt"] = on_disk(svc.NPU_ATTEMPT_KEY)
        return real_load(*args, **kwargs)

    engine.load = watched_load
    try:
        chat(service)
        assert disk["check"]["version"] == VERSION
        assert disk["attempt"]["stem"] == Q40 and disk["inflight"]["stem"] == Q40
        for key in (svc.NPU_CHECK_ATTEMPT_KEY, svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY):
            assert on_disk(key) is None                                        # the clears are saved too
        # a fallback is saved at once (the next start must not try the NPU again)
        service._npu_failed(Q40, "boom")
        assert on_disk(svc.NPU_FALLBACK_KEY)[Q40]["reason"] == "boom"
        # the self-check's prefs pass the save on to the app's Prefs
        shared = selfcheck._DictPrefs(shared=prefs)
        shared.set(svc.NPU_ATTEMPT_KEY, {"stem": "s"})
        shared.flush()
        assert on_disk(svc.NPU_ATTEMPT_KEY) == {"stem": "s"}
    finally:
        prefs.close()


def test_an_answer_cut_short_in_the_background_records_no_fallback(tmp_path):
    marks = []
    holder = {}

    def reply(messages):
        service = holder["service"]
        marks.append(service.prefs.get(svc.NPU_INFLIGHT_KEY))
        service.on_lifecycle("hide")
        marks.append(service.prefs.get(svc.NPU_INFLIGHT_KEY))
        service.on_lifecycle("show")
        marks.append(service.prefs.get(svc.NPU_INFLIGHT_KEY))
        return "fine"

    service, _engine, _probe = npu_service(tmp_path, engine=eng.FakeEngine(reply=reply))
    holder["service"] = service
    chat(service)
    assert [m["background"] for m in marks] == [False, True, False] and {m["stem"] for m in marks} == {Q40}
    assert service.prefs.get(svc.NPU_INFLIGHT_KEY) is None
    service.on_lifecycle("hide")
    assert service.prefs.get(svc.NPU_INFLIGHT_KEY) is None                     # no answer: nothing written
    # the process died in the background (memory killer, swiped away): nothing recorded at the next start
    prefs = fakes.FakePrefs({svc.NPU_INFLIGHT_KEY: {"stem": Q40, "at": 1, "background": True}})
    later, engine, _p = npu_service(tmp_path / "l", prefs=prefs)
    later.start()
    try:
        assert prefs.get(svc.NPU_INFLIGHT_KEY) is None and later.npu_fallback(Q40) is None
        chat(later)
        assert engine.backends == ["npu"]
    finally:
        later.stop()
        svc.set_default_service(None)


def test_only_the_service_that_set_the_attempt_marker_clears_it(tmp_path):
    # the parallel secondary and the self-check services share the key with the app's own service
    app_prefs = fakes.FakePrefs({svc.NPU_ATTEMPT_KEY: {"stem": Q40, "at": 1}})
    child, _e = make_service(tmp_path / "c", stems=(Q40,), prefs=app_prefs, _parallel_child=True)
    child.load_blocking(Q40)
    child.unload()
    assert app_prefs.get(svc.NPU_ATTEMPT_KEY) == {"stem": Q40, "at": 1}
    check, _e2, _p = npu_service(tmp_path / "s", prefs=selfcheck._DictPrefs(shared=app_prefs), recover_npu=False)
    check._npu_failed(QK, "the first token on the NPU failed")
    check.unload()
    assert app_prefs.get(svc.NPU_ATTEMPT_KEY) == {"stem": Q40, "at": 1}


def test_inflight_marker_is_held_during_every_npu_answer(tmp_path):
    seen = []
    holder = {}

    def reply(messages):
        seen.append(holder["service"].prefs.get(svc.NPU_INFLIGHT_KEY))
        return "fine"

    service, engine, _probe = npu_service(tmp_path, engine=eng.FakeEngine(reply=reply))
    holder["service"] = service
    chat(service)
    chat(service)
    assert [m["stem"] for m in seen] == [Q40, Q40]                             # set before each answer
    assert service.prefs.get(svc.NPU_INFLIGHT_KEY) is None                     # cleared after
    set_backend(service, Q40, "cpu")
    chat(service)
    assert seen[-1] is None                                                    # never around a CPU answer

    def broken(messages):
        raise ValueError("bad template")

    service2, _e, _p = npu_service(tmp_path / "b", engine=eng.FakeEngine(reply=broken))
    with pytest.raises(Exception):
        chat(service2)
    assert service2.prefs.get(svc.NPU_INFLIGHT_KEY) is None and not service2.gate.busy


@pytest.mark.parametrize("key", [svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY])
def test_a_marker_left_by_a_dead_process_records_the_fallback_at_start(tmp_path, key):
    prefs = fakes.FakePrefs({key: {"stem": Q40, "at": 1}})
    service, engine, _probe = npu_service(tmp_path, prefs=prefs)
    service.start()
    try:
        assert prefs.get(svc.NPU_ATTEMPT_KEY) is None and prefs.get(svc.NPU_INFLIGHT_KEY) is None
        assert service.npu_fallback(Q40) == {"stem": Q40, "reason": svc.NPU_CRASH_REASON}
        assert svc.NPU_CRASH_REASON == "Glossarion closed while the NPU was running this model"
        service.run_npu_check()
        chat(service)
        assert engine.backends == ["cpu"]
        assert svc.NPU_CRASH_REASON in service.snapshot().npu_note
        assert service.retry_npu(Q40)                                          # "Try the NPU again"
        chat(service)
        assert engine.backends == ["cpu", "npu"]
    finally:
        service.stop()
        svc.set_default_service(None)


def test_a_check_the_app_died_in_keeps_the_npu_off_until_the_version_changes(tmp_path):
    prefs = fakes.FakePrefs({svc.NPU_CHECK_ATTEMPT_KEY: {"version": VERSION, "at": 1}})
    service, _engine, probe = npu_service(tmp_path, prefs=prefs)
    service.start()
    try:
        state = service.run_npu_check()
        assert probe.calls == [] and not service.npu_offered()                 # not tried again
        assert service.npu_failure() == ("session", svc.NPU_CHECK_CRASH_REASON) and state["error"]
        assert prefs.get(svc.NPU_CHECK_ATTEMPT_KEY) is None
    finally:
        service.stop()
        svc.set_default_service(None)
    newer, _e, probe2 = npu_service(tmp_path / "n", prefs=prefs, version="9.16.1/llama-0.3.37")
    assert newer.npu_offered() is False and newer.run_npu_check() and probe2.calls and newer.npu_offered()


def test_the_check_runs_in_the_background_at_start_and_marks_itself(tmp_path):
    marks = []
    holder = {}

    class Watching(Probe):
        def __call__(self, skel_dir, options):
            marks.append(holder["prefs"].get(svc.NPU_CHECK_ATTEMPT_KEY))
            return super().__call__(skel_dir, options)

    prefs = fakes.FakePrefs()
    holder["prefs"] = prefs
    service, _engine, probe = npu_service(tmp_path, prefs=prefs, probe=Watching())
    service.start()
    try:
        deadline = time.monotonic() + 5
        while service._npu_state is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.npu_offered() and marks[0]["version"] == VERSION
        assert prefs.get(svc.NPU_CHECK_ATTEMPT_KEY) is None
        assert prefs.get(svc.NPU_CHECK_KEY)["blocked"] is False
    finally:
        service.stop()
        svc.set_default_service(None)
    # the next start shows the rows from the saved check before its own has run
    later, _e, _p = npu_service(tmp_path / "l", prefs=prefs, probe=Probe(delay=5))
    assert later._npu_state is None and later.npu_offered()


def test_the_watchdog_on_a_hanging_check(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, probe=Probe(delay=0.5))
    service.npu_watchdog_seconds = 0.05
    state = service.run_npu_check()
    assert not eng.npu_passed(state) and service.npu_hung and not service.npu_offered()
    assert service.prefs.get(svc.NPU_CHECK_KEY)["blocked"] is True
    # a restart does not bring the NPU back (the check stays blocked until the app version changes); models that
    # never used the NPU get no note
    assert service.npu_note() is None and service.npu_note(Q40) is None
    assert service.npu_hang_note() == svc.NPU_CHECK_HANG_NOTE
    assert svc.NPU_CHECK_HANG_NOTE == "The NPU check did not answer. The NPU stays off until the next app update."
    assert svc.NPU_CHECK_HANG_NOTE in selfcheck.npu_lines(service, soc="")


def test_the_watchdog_on_a_hanging_first_load(tmp_path):
    engine = eng.FakeEngine(npu_load_delay=0.4)
    service, engine, _probe = npu_service(tmp_path, engine=engine, gpu=True)
    service.npu_watchdog_seconds = 0.05
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["npu", "gpu"] and service.snapshot().backend == "GPU"
    assert service.npu_fallback(Q40)["reason"] == svc.NPU_HANG_REASON and service.npu_hung
    assert service.snapshot().npu_note == "The NPU did not answer. Restart Glossarion; this model will use the GPU."
    assert service.npu_hang_note() == service.npu_note(Q40)
    assert service.npu_note(QK) is None                                        # only the model whose load hung
    # only the first NPU load of a process is watched
    other, engine2, _p = npu_service(tmp_path / "o", engine=eng.FakeEngine())
    other.npu_watchdog_seconds = 0.05
    chat(other)
    other.unload()
    engine2.npu_load_delay = 0.2
    chat(other)
    assert engine2.backends == ["npu", "npu"] and other.npu_fallback(Q40) is None


# ---- diagnostics and the self-check ------------------------------------------------------------------------------

def test_diagnostics_block_when_the_npu_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(eng, "hex_log_lines", lambda: ["ggml-hex: HTP0 new session : session-id 0"])
    service, engine, _probe = npu_service(tmp_path, stems=(Q40, QK), gpu=True)
    service.run_npu_check()
    chat(service)
    service._npu_fallbacks()
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert set(speeds) == {"npu", "gpu", "cpu"} and speeds["npu"]["backend"] == "NPU"
    assert speeds["gpu"]["backend"] == "GPU" and speeds["cpu"]["backend"] == "CPU"
    assert all(speeds[b]["tokens"] > 0 for b in speeds) and service.backend_override is None
    assert service.loaded_stem is None                                         # reloads on its usual backend
    lines = selfcheck.npu_lines(service, speeds, soc="QTI SM8850")
    text = "\n".join(lines)
    assert lines[0] == "NPU: available" and "SoC: QTI SM8850" in lines
    for name in eng.NPU_STEPS:
        assert f"✓ {name}: {name} ok" in lines
    assert "Arch: v81" in lines and "Device: HTP0" in lines
    assert "Use NPU: on · power mode: burst" in lines
    assert "Last load: m-Q4_0 on CPU" in text                                 # the CPU timing run was the last
    assert text.count("tok/s") == 3 and "Speed (16 tokens): NPU " in text
    assert "Fallback: none" in lines
    assert "ggml-hex log:" in lines and "  ggml-hex: HTP0 new session : session-id 0" in lines
    service._npu_failed(QK, "the first token on the NPU failed")
    assert f"Fallback: {QK}: the first token on the NPU failed" in selfcheck.npu_lines(service, soc="")


@pytest.mark.parametrize("step", eng.NPU_STEPS)
def test_diagnostics_line_for_each_failing_step(tmp_path, step):
    service, _engine, _probe = npu_service(tmp_path, probe=Probe(failing_state(step, "dlopen failed")))
    service.run_npu_check()
    assert selfcheck.npu_lines(service, soc="x") == [f"NPU: not available ({step}: dlopen failed)"]


def test_diagnostics_line_without_the_npu(tmp_path):
    service, _engine = make_service(tmp_path)
    assert selfcheck.npu_lines(service) == ["NPU: not available (shipped: this build has no NPU (Android arm64 only))"]
    assert selfcheck.npu_lines(None)[0].startswith("NPU: not available (engine:")
    service2, _e, _p = npu_service(tmp_path / "b", probe=Probe(passing_state(arch=69)))
    service2.run_npu_check()
    assert selfcheck.npu_lines(service2, soc="") == ["NPU: not available (arch: unsupported HTP arch 69)"]
    speeds = selfcheck.npu_speed_check(service2, Q40)
    assert "NPU not available (arch" in speeds["npu"]["skipped"] and speeds["cpu"]["tokens"] > 0


def test_speed_check_refuses_during_a_job(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, active_job_model=lambda: f"local/{Q40}")
    assert "error" in selfcheck.npu_speed_check(service, Q40)


def test_self_check_services_share_the_app_markers(tmp_path):
    app_prefs = fakes.FakePrefs({"other": 1})
    prefs = selfcheck._DictPrefs({svc.SETTINGS_KEY: {"n_ctx": 2048}}, shared=app_prefs)
    for key in (svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY, svc.NPU_FALLBACK_KEY, svc.NPU_CHECK_KEY,
                svc.NPU_CHECK_ATTEMPT_KEY):
        prefs.set(key, {"stem": "s"})
        assert app_prefs.get(key) == {"stem": "s"} and prefs.get(key) == {"stem": "s"}
    prefs.set(svc.SETTINGS_KEY, {"n_ctx": 1})
    assert app_prefs.get(svc.SETTINGS_KEY) is None and prefs.get("other") is None
    # a self-check's NPU answer that dies is recorded for the app: its in-flight marker is in the app's Prefs
    seen = []
    holder = {}

    def reply(messages):
        seen.append(app_prefs.get(svc.NPU_INFLIGHT_KEY))
        return "ok"

    plain = selfcheck._DictPrefs(shared=app_prefs)
    service, _engine, _probe = npu_service(tmp_path, prefs=plain, engine=eng.FakeEngine(reply=reply),
                                           recover_npu=False)
    holder["s"] = service
    chat(service)
    assert seen[0]["stem"] == Q40 and app_prefs.get(svc.NPU_INFLIGHT_KEY) is None
    # its own start does not consume the app's markers
    app_prefs.set(svc.NPU_INFLIGHT_KEY, {"stem": "app-model"})
    service.start()
    try:
        assert app_prefs.get(svc.NPU_INFLIGHT_KEY) == {"stem": "app-model"}
    finally:
        service.stop()
        svc.set_default_service(None)


def test_npu_backend_check_reads_the_wheel(monkeypatch):
    ext = types.SimpleNamespace(npu_backend_shipped=lambda: True, register_npu_backend=lambda *a: {},
                                device_index=lambda n: 0)
    monkeypatch.setattr(selfcheck.importlib, "import_module", lambda name: ext)
    assert selfcheck.npu_backend_check()["functions"] is True and selfcheck.npu_backend_check()["shipped"] is True
    monkeypatch.setattr(selfcheck.importlib, "import_module", lambda name: types.SimpleNamespace())
    result = selfcheck.npu_backend_check()
    assert result["functions"] is False and result["shipped"] is None and result["device_index"] is False


def test_the_job_log_names_the_backend(tmp_path, monkeypatch):
    import unified_api_client as uac

    printed: list = []
    monkeypatch.setattr(uac, "print", lambda *a, **k: printed.append(" ".join(map(str, a))), raising=False)
    service, _engine, _probe = npu_service(tmp_path)
    chat(service)
    health = service.health(f"local/{Q40}")
    assert health["backend"] == "NPU" and "npu_note" not in health
    client = uac.UnifiedClient.__new__(uac.UnifiedClient)
    client._local_engine_health = lambda base, model: health
    client._local_engine_loaded_note("http://x/v1", f"local/{Q40}", {"before": 1})
    service._npu_failed(Q40, "boom")
    client._local_engine_health = lambda base, model: {**health, "npu_note": service.npu_note(Q40)}
    client._local_engine_loaded_note("http://x/v1", f"local/{Q40}", {"before": 1})
    lines = [line for line in printed if line.strip()]
    assert lines[0].startswith(f"✓ On-device model {Q40} loaded in ") and lines[0].endswith(" · runs on NPU")
    assert lines[-1] == f"⚠ NPU unavailable for {Q40}: boom. Using CPU."
    # a phone without the NPU: the line stays as it was
    plain, _e = make_service(tmp_path / "p")
    chat(plain, "fake-echo")
    assert "backend" not in plain.health("local/fake-echo")
