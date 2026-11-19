"""NPU on iOS (the Apple Neural Engine, llama.cpp draft PR 31969), app side: the engine's ANE states (stage 1 on a
simulator, below iOS 18, without the framework; stage 2's registry and test compile), the compile pass of an NPU
load (one KV bucket, the collect pass of token ids only, the storage gate, the real pass, the freeze), Metal's
registry name "MTL" for the GPU, and the service on an iPhone: no files to download and no power mode, Auto's quant
and size rules, the engine card's "Preparing this model for the NPU · n / m (first time only)" and the cache hit,
the watchdog that counts from the last progress, the fallback to Metal on every failure (stage 2, the load, the
storage gate, a full phone, a hang), the attempt marker's background flag, the ledger of NPU copies, the
diagnostics block and the smoke report's precheck, the engine card (no power mode, no quant marks, the copies line)
and a screen that never waits for a slow check; and Android left as it was. Fakes only: FakeEngine, FakePrefs, fake
loader functions and a fake Llama; no device, no network.
"""

from __future__ import annotations

import math
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
from test_local_npu import FakeHTTP, Pre, Probe, chat, npu_service, set_backend

STEM = "m-Q4_K_M"
IQ = "m-IQ4_XS"
VERSION = "9.16.0+abc/llama-0.3.37"
GB = 1_000_000_000


def ane_state(**changes) -> dict:
    """``register_npu_backend``'s answer on an iPhone whose Neural Engine passed (the loader's shape)."""
    state = {"backend": "ANE", "device": "ANE", "arch": None, "chip": "h15", "error": "",
             "steps": [{"name": "shipped", "ok": True, "detail": "libggml-ane.dylib"},
                       {"name": "device", "ok": True, "detail": "iPhone15,2, iOS 18.4"},
                       {"name": "framework", "ok": True, "detail": "Apple Neural Engine h15"},
                       {"name": "registry", "ok": True, "detail": "ANE"},
                       {"name": "compile", "ok": True, "detail": "test program compiled and ran"}]}
    state.update(changes)
    return state


def ane_failing(step: str, detail: str) -> dict:
    state = ane_state(error=detail)
    for entry in state["steps"]:
        if entry["name"] == step:
            entry.update(ok=False, detail=detail)
    return state


def ane_service(tmp_path, *, stems=(STEM,), probe=None, precheck=None, engine=None, prefs=None, gpu=True,
                free=100 * GB, **kwargs):
    """A service on an iPhone (platform "ios", Metal as its GPU) with fakes of the ANE loader's two stages."""
    prefs = prefs if prefs is not None else fakes.FakePrefs({svc.GPU_KEY: gpu})
    probe = probe if probe is not None else Probe(ane_state())
    precheck = precheck if precheck is not None else Pre(probe.state)
    http = FakeHTTP()
    service, engine = make_service(tmp_path, stems=stems, prefs=prefs, engine=engine, npu_probe=probe,
                                   npu_precheck=precheck, npu_source=lambda: pytest.fail("no source on the ANE"),
                                   npu_root=str(tmp_path / "htp"), npu_session=http, app_version=VERSION,
                                   gpu_backend="Metal", **kwargs)
    service.platform = "ios"
    service.free_storage = lambda path: free
    service.npu_progress_poll = 0.01
    service.http = http
    return service, engine, probe


class AnePre(Pre):
    """``npu_precheck`` of the ANE loader: the first three steps, its backend and chip."""

    def __call__(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        steps = []
        for step in self.state["steps"]:
            if step["name"] in eng.NPU_STEPS_BY_BACKEND["ANE"][:3]:
                steps.append(dict(step))
                if not step["ok"]:
                    break  # the loader stops at the first failing step (a simulator never reaches "framework")
        failing = next((s for s in steps if not s["ok"]), None)
        return {"backend": "ANE", "chip": self.state.get("chip"), "arch": None,
                "error": failing["detail"] if failing else None, "steps": steps}


class AneProbe(Probe):
    """``register_npu_backend`` of the ANE loader: no files, the state as given; records its calls."""

    def __call__(self, skel_dir, options):
        self.calls.append((skel_dir, dict(options)))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        return {**self.state, "steps": [dict(s) for s in self.state["steps"]]}


def talk(service, stem=STEM, text="hi"):
    return chat(service, stem, text)


def ane(tmp_path, state=None, **kwargs):
    state = state if state is not None else ane_state()
    probe = AneProbe(state)
    return ane_service(tmp_path, probe=probe, precheck=AnePre(state), **kwargs)


#: an ANE compile pass the first time: 4 program families collected, then compiled one by one
FRESH = [{"families_pending": 4, "bytes_pending": 400_000_000},
         {"families_pending": 3, "compiles_fresh": 1, "bytes_fresh": 100_000_000, "bytes_pending": 300_000_000},
         {"families_pending": 2, "compiles_fresh": 2, "bytes_fresh": 200_000_000, "bytes_pending": 200_000_000},
         {"families_pending": 1, "compiles_fresh": 3, "bytes_fresh": 300_000_000, "bytes_pending": 100_000_000},
         {"families_pending": 0, "compiles_fresh": 4, "bytes_fresh": 400_000_000}]
#: the same model at a later launch: every family from the iOS cache
CACHED = [{"families_pending": 4, "bytes_pending": 400_000_000},
          {"families_pending": 2, "compiles_cached": 2, "bytes_pending": 200_000_000},
          {"families_pending": 0, "compiles_cached": 4}]


# ---- the engine: ANE states ---------------------------------------------------------------------------------------

def test_the_loader_steps_by_backend():
    assert eng.NPU_STEPS_BY_BACKEND == {"HTP": eng.NPU_STEPS,
                                        "ANE": ("shipped", "device", "framework", "registry", "compile")}
    assert eng.NPU_STEPS == ("shipped", "libcdsprpc", "arch", "skeletons", "registry", "session")  # Android as it was
    assert eng.NPU_BACKEND == "HTP" and eng.ANE_BACKEND == "ANE"


@pytest.mark.parametrize("step,detail", [
    ("shipped", "this engine build does not carry the NPU backend"),
    ("device", "iPhone simulator: no Neural Engine"),
    ("device", "needs iOS 18 or later (this phone: iOS 17.5)"),
    ("framework", "AppleNeuralEngine.framework did not load"),
])
def test_ane_precheck_failures_name_their_step(step, detail):
    raw = AnePre(ane_failing(step, detail))()
    state = eng.npu_precheck_state(raw)
    assert state["backend"] == "ANE" and state["arch"] is None and state["chip"] == "h15"
    assert eng.npu_precheck_failure(state) == (step, detail)


def test_ane_precheck_passes_without_any_arch():
    state = eng.npu_precheck_state(AnePre(ane_state())())
    assert eng.npu_precheck_failure(state) is None                       # the HTP arch rule never applies
    state["steps"] = state["steps"][:2]
    assert eng.npu_precheck_failure(state)[0] == "framework"


@pytest.mark.parametrize("step", eng.NPU_STEPS_BY_BACKEND["ANE"])
def test_every_failing_ane_step_is_named(step):
    assert eng.npu_failure(ane_failing(step, "why")) == (step, "why")


def test_ane_state_label_and_missing_steps():
    assert eng.npu_passed(ane_state()) and eng.npu_label(ane_state()) == "Apple Neural Engine (h15)"
    assert eng.npu_label(ane_state(chip="")) == "Apple Neural Engine"
    assert eng.npu_failure(ane_state(device=""))[0] == "registry"
    state = ane_state()
    state["steps"] = state["steps"][:4]
    assert eng.npu_failure(state)[0] == "compile"
    assert eng.npu_failure(ane_state(error="boom")) == ("compile", "boom")


def _ext(monkeypatch, **functions):
    ext = types.SimpleNamespace(**functions)
    monkeypatch.setattr(eng, "_extensions", lambda: ext)
    monkeypatch.setattr(eng, "capture_hex_log", lambda: False)
    return ext


def test_engine_reads_the_ane_loader(monkeypatch):
    calls = []
    _ext(monkeypatch, npu_precheck=lambda: AnePre(ane_state())(), npu_backend_shipped=lambda: True,
         register_npu_backend=lambda skel, options: calls.append((skel, options)) or ane_state(),
         npu_skeleton_source=lambda: {"release": "b12387", "commit": "a956a852", "url": "", "size": 0,
                                      "sha256": "", "files": {}},
         device_index=lambda name: {"ANE": 0, "MTL": 1}.get(name, -1))
    assert eng.npu_precheck_failure(eng.npu_precheck()) is None
    assert eng.npu_skeleton_source()["files"] == {}                           # nothing to download
    state = eng.npu_state("", {})
    assert calls == [("", {})] and state["chip"] == "h15" and state["arch"] is None and eng.npu_passed(state)
    assert eng.npu_backend() == "ANE" and eng.npu_label() == "Apple Neural Engine (h15)"
    eng._NPU_STATE.clear()
    assert eng.npu_backend() == "HTP"


def test_ane_log_lines_are_kept_with_the_hexagon_ones(monkeypatch):
    monkeypatch.setattr(eng, "_HEX_PENDING", [""])
    eng._HEX_LOG.clear()
    eng._remember_hex("ggml_ane_program_create: compiled family 3\nllama: other\nggml-ane: ANE h15\n")
    assert eng.hex_log_lines() == ["ggml_ane_program_create: compiled family 3", "ggml-ane: ANE h15"]
    eng._HEX_LOG.clear()


def test_ane_primitives_are_no_ops_without_the_loader(monkeypatch):
    monkeypatch.setattr(eng, "_extensions", lambda: None)
    assert eng.npu_stats() == {} and eng.npu_model_begin(2048, 4) is None
    assert eng.npu_collect(True) is None and eng.npu_freeze(True) is None
    _ext(monkeypatch, npu_stats=lambda: {"compiles_fresh": "2", "storage_refused": 1, "other": 9})
    stats = eng.npu_stats()
    assert stats["compiles_fresh"] == 2 and stats["storage_refused"] == 1 and set(stats) == set(eng.ANE_STATS)


# ---- the engine: the compile pass of an ANE load --------------------------------------------------------------

class _AneLoader:
    """The ANE loader's primitives, recording their calls in order (``stats`` changes as the pass runs)."""

    def __init__(self, collected=None, compiled=None):
        self.calls: list = []
        self.collected = collected or {"families_pending": 3, "bytes_pending": 300_000_000}
        self.compiled = compiled or {"compiles_fresh": 3, "bytes_fresh": 300_000_000}
        self.stats = {}
        self.collecting = False

    def install(self, monkeypatch):
        _ext(monkeypatch, device_index=lambda name: {"ANE": 0, "MTL": 1}.get(name, -1),
             npu_model_begin=self.begin,
             npu_collect=self.collect, npu_freeze=lambda on: self.calls.append(("freeze", on)),
             npu_stats=lambda: dict(self.stats))
        eng._NPU_STATE.clear()
        eng._NPU_STATE.update(ane_state())

    def begin(self, n_kv, threads):
        # like glossarion_ane_reset_stats: the counters reset, the storage refusal is the process's state and stays
        self.calls.append(("begin", n_kv, threads))
        self.stats = {k: v for k, v in self.stats.items() if k == "storage_refused"}

    def collect(self, on):
        self.calls.append(("collect", on))
        self.collecting = on
        if not on:
            self.stats = dict(self.collected)


class _AneLlama(_FakeLlama):
    loader: "_AneLoader" = None
    bos = 7
    engine = None  # when set: what its npu_stats() says before the compile pass begins
    before: list = []

    def n_ctx(self):
        if _AneLlama.engine is not None:
            _AneLlama.before.append(_AneLlama.engine.npu_stats())
        return self.kwargs["n_ctx"]

    def token_bos(self):
        return _AneLlama.bos

    def eval(self, tokens):
        super().eval(tokens)
        loader = _AneLlama.loader
        loader.calls.append(("eval", len(tokens), set(tokens), loader.collecting))
        if not loader.collecting and len(tokens) > 1:                   # the real pass compiles the families
            loader.stats = dict(loader.compiled)


@pytest.fixture()
def ane_llama(fake_llama, monkeypatch):  # noqa: F811
    loader = _AneLoader()
    loader.install(monkeypatch)
    _AneLlama.loader, _AneLlama.bos, _AneLlama.engine, _AneLlama.before = loader, 7, None, []
    monkeypatch.setattr(eng.LlamaEngine, "_module", lambda self: types.SimpleNamespace(Llama=_AneLlama))
    yield loader
    _AneLlama.engine = None
    eng._NPU_STATE.clear()


def test_an_ane_load_runs_the_compile_pass_in_order(ane_llama):
    gates = []
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 4096, 3, npu=True, n_gpu_layers=-1, npu_gate=lambda stats: gates.append(stats))
    llm = _FakeLlama.made[-1]
    assert (llm.kwargs["main_gpu"], llm.kwargs["split_mode"], llm.kwargs["n_batch"]) == (0, eng.SPLIT_MODE_NONE, 512)
    assert llm.kwargs["flash_attn"] is True and engine.on_npu and engine.device == "NPU: Apple Neural Engine (h15)"
    assert ane_llama.calls == [
        ("begin", 4096, 3),                                   # one KV bucket: the context size
        ("collect", True), ("eval", 512, {7}, True), ("eval", 1, {7}, True), ("collect", False),
        ("eval", 512, {7}, False), ("eval", 1, {7}, False),   # every family compiles once
        ("freeze", True)]                                     # from here odd shapes run on the CPU
    (gate,) = gates                                           # the storage gate sees the collected families
    assert (gate["families_pending"], gate["bytes_pending"], gate["compiles_fresh"]) == (3, 300_000_000, 0)
    assert not llm.closed and engine.npu_stats()["compiles_fresh"] == 3
    # token ids only, never text: BOS, else 0
    _AneLlama.bos = -1
    engine.load("m.gguf", 4096, 3, npu=True)
    assert {t for call in _FakeLlama.made[-1].evals for t in call} == {0}


def test_ane_compile_pass_failures_leave_no_model(ane_llama):
    engine = eng.LlamaEngine()

    def refuse(stats):
        raise eng.NpuUnavailable("not enough free storage for the NPU copy of this model (needs about 3.0 GB)")

    with pytest.raises(eng.NpuUnavailable, match="not enough free storage"):
        engine.load("m.gguf", 2048, 4, npu=True, npu_gate=refuse)
    assert _FakeLlama.made[-1].closed and not engine.loaded and ("freeze", False) in ane_llama.calls
    assert not [c for c in ane_llama.calls if c[0] == "eval" and not c[3]]  # nothing compiled after the refusal
    _FakeLlama.logits = (math.nan,) * 4
    with pytest.raises(eng.NpuUnavailable, match="preparing the model for the NPU failed .*NaN"):
        engine.load("m.gguf", 2048, 4, npu=True)
    _FakeLlama.logits = (0.5, -1.0, 2.0, 0.0)
    ane_llama.compiled = {"compiles_fresh": 1, "storage_refused": 1}
    with pytest.raises(eng.NpuStorageFull, match=eng.ANE_STORAGE_TEXT) as refused:
        engine.load("m.gguf", 2048, 4, npu=True)
    assert not engine.loaded and refused.value.before_pass is False          # this model's own pass ran out
    # the backend refuses every compile for the rest of the process: the next load stops before its pass,
    # and says so (the service does not blame that model)
    ane_llama.calls.clear()
    with pytest.raises(eng.NpuStorageFull) as refused:
        engine.load("other.gguf", 2048, 4, npu=True)
    assert refused.value.before_pass is True and not engine.loaded and _FakeLlama.made[-1].closed
    assert not [c for c in ane_llama.calls if c[0] in ("eval", "collect")]


def test_the_kv_bucket_and_the_batch_fit_any_context(ane_llama):
    engine = eng.LlamaEngine()
    # the app's default context (24576): one bucket of 8192 rows (the backend would follow the KV length above
    # 16384, and an iPhone keeps attention over more than 8192 rows on the CPU anyway)
    engine.load("m.gguf", 24576, 3, npu=True)
    assert ane_llama.calls[0] == ("begin", 8192, 3)
    assert eng.ane_kv_bucket(24576) == eng.ane_kv_bucket(8192) == 8192 and eng.ane_kv_bucket(4096) == 4096
    # a 512-token context: 511 tokens and one more fit (512 + 1 would not)
    ane_llama.calls.clear()
    engine.load("m.gguf", 512, 3, npu=True)
    evals = [c[1] for c in ane_llama.calls if c[0] == "eval"]
    assert ane_llama.calls[0] == ("begin", 512, 3) and evals == [511, 1, 511, 1] and engine.on_npu


def test_the_counters_are_the_loading_models_own(ane_llama):
    engine = eng.LlamaEngine()
    _AneLlama.engine = engine
    engine.load("a.gguf", 2048, 3, npu=True)
    assert engine.npu_stats()["compiles_fresh"] == 3                         # model A's pass, done
    engine.load("b.gguf", 2048, 3, npu=True)
    # while B's file loaded (before its pass reset the process-wide counters), A's counts were not shown as B's
    assert _AneLlama.before == [{}, {}]


def test_the_hexagon_load_keeps_its_first_token_check(ane_llama):
    eng._NPU_STATE.clear()
    eng._NPU_STATE.update({"backend": "HTP", "device": "HTP0", "arch": 81})
    eng._extensions().device_index = lambda name: {"HTP": 0}.get(name, -1)
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, npu=True)
    assert not [c for c in ane_llama.calls if c[0] in ("begin", "collect", "freeze")]
    assert _FakeLlama.made[-1].evals == [[1, 2, 3, 4], [4]]


def test_metal_is_found_by_its_registry_name(fake_llama, monkeypatch):  # noqa: F811
    fake_llama.update(device="Apple A17 Pro GPU", backend="Metal", error=None)
    _ext(monkeypatch, device_index=lambda name: {"ANE": 0, "MTL": 1}.get(name, -1))
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, gpu=True)
    assert _FakeLlama.made[-1].kwargs["main_gpu"] == 1                      # never the ANE that registered first
    _ext(monkeypatch, device_index=lambda name: {"ANE": 0}.get(name, -1))
    with pytest.raises(eng.GpuUnavailable, match="Metal backend is not in llama.cpp's device list"):
        engine.load("m.gguf", 2048, 4, gpu=True)
    _ext(monkeypatch)                                                          # a build without device_index
    engine.load("m.gguf", 2048, 4, gpu=True)
    assert _FakeLlama.made[-1].kwargs["main_gpu"] == 0


# ---- the service on an iPhone -------------------------------------------------------------------------------------

def test_a_simulator_is_never_offered_the_npu_and_nothing_downloads(tmp_path):
    service, engine, probe = ane(tmp_path, ane_failing("device", "iPhone simulator: no Neural Engine"))
    service.run_npu_check()
    assert not service.npu_offered() and service.npu_backend() == "ANE"
    assert service.npu_failure() == ("device", "iPhone simulator: no Neural Engine")
    assert service.npu_status_line() is None and service.npu_copies_line() is None
    assert talk(service) == "Echo: hi"
    assert engine.backends == ["gpu"] and probe.calls == [] and service.http.calls == []
    assert selfcheck.npu_lines(service, soc="")[0] == "NPU: not available (device: iPhone simulator: no Neural Engine)"


def test_below_ios_18_or_without_the_framework(tmp_path):
    for step, detail in (("device", "needs iOS 18 or later (this phone: iOS 17.5)"),
                         ("framework", "AppleNeuralEngine.framework did not load")):
        service, engine, _p = ane(tmp_path / step, ane_failing(step, detail))
        service.run_npu_check()
        assert service.npu_failure() == (step, detail) and not service.npu_offered()


def test_the_iphone_registers_without_files_or_power_mode(tmp_path):
    service, engine, probe = ane(tmp_path, engine=eng.FakeEngine(npu_device="Apple Neural Engine (h15)"))
    service.run_npu_check()
    assert service.npu_offered() and service.npu_files_ready() and service.npu_options() == {}
    assert service.prefs.get(svc.NPU_CHECK_KEY)["source"] == {"files": {}}
    assert service.npu_status_line() == svc.NPU_ANE_READY_TEXT
    assert talk(service) == "Echo: hi"
    assert probe.calls == [("", {})] and service.http.calls == []           # no download, no GGML_HEXAGON_*
    assert engine.backends == ["npu"] and service.snapshot().backend == "NPU"
    assert not service.npu_power_pending()
    service.save_npu_power("balanced")
    assert not service.npu_power_pending()                                   # no power mode on the ANE


def test_a_failed_test_compile_keeps_the_npu_off_and_loads_on_metal(tmp_path):
    service, engine, probe = ane(tmp_path, ane_failing("compile", "the test program did not compile"))
    assert talk(service) == "Echo: hi"
    assert engine.backends == ["gpu"] and len(probe.calls) == 1               # stage 2 ran once, before the load
    assert service.npu_failure() == ("compile", "the test program did not compile") and not service.npu_offered()
    assert service.npu_fallback(STEM) is None                                # the NPU, not the model
    talk(service, text="again")
    assert len(probe.calls) == 1


def test_auto_on_the_ane_takes_every_quant_but_only_small_files(tmp_path):
    service, _engine, _p = ane(tmp_path, stems=(STEM, IQ))
    service.run_npu_check()
    assert service._placement(IQ)[0] == "npu"                                # IQ4_XS: requantised, not "cpu"
    assert presets.npu_quant_kind("x-IQ4_XS.gguf", "ANE") == "ok" and presets.npu_quant_kind("odd.gguf", "ANE") == "cpu"
    entry = dict(service.registry.load()[STEM], size=svc.NPU_ANE_MAX_FILE_BYTES + 1)
    service.registry.add(STEM, entry)
    assert service._placement(STEM)[0] == "gpu"                              # over 1.5 GB: Auto uses Metal
    set_backend(service, STEM, "npu")
    # chosen by hand, or the diagnostics' timing run: still not over 1.5 GB (a copy iOS keeps for good)
    assert service._placement(STEM)[0] == "gpu"
    service.backend_override = "npu"
    assert service._placement(STEM)[0] == "gpu"
    service.backend_override = None
    # a registry entry without a size: the file on disk says
    service.registry.add(STEM, dict(entry, size=0))
    assert service._placement(STEM)[0] == "npu"
    import os

    os.remove(service.registry.file_path(STEM))
    assert service._placement(STEM)[0] == "gpu"                              # unknown size: not on the ANE
    service.save_use_npu(False)
    assert service._placement(IQ)[0] == "gpu"


def test_first_run_compile_progress_and_the_ledger(tmp_path):
    engine = eng.FakeEngine(npu_compile=FRESH, npu_step_delay=0.06, npu_device="Apple Neural Engine (h15)")
    service, engine, _p = ane(tmp_path, engine=engine)
    seen: list = []
    service.subscribe(lambda snap: seen.append((service.npu_progress() or {}).get("text")))
    phases: list = []
    service.subscribe(lambda snap: phases.append(snap.phase))
    assert talk(service) == "Echo: hi" and engine.backends == ["npu"]
    texts = [t for t in seen if t and t.startswith("Preparing")]
    assert texts[0] == "Preparing this model for the NPU · 0 / 4"            # before the first program compiled
    assert "Preparing this model for the NPU · 1 / 4 (first time only)" in texts
    assert "Preparing this model for the NPU · 3 / 4 (first time only)" in texts
    assert "Starting the NPU…" in seen                                         # stage 2, then the compile pass
    assert any(p and p.startswith("Preparing") for p in phases)              # the engine card's loading line
    assert service.npu_progress() is None and service._npu_phase is None
    assert engine.npu_gates == [FRESH[0]]
    copies = service.prefs.get(svc.NPU_COPIES_KEY)
    (key, entry), = copies.items()
    # keyed by the KV bucket: the default context (24576) and every other one from 8192 up share one copy
    assert service.context_for(STEM) >= eng.ANE_MAX_KV
    assert key == f"{STEM}.gguf|4|{eng.ANE_MAX_KV}|{eng.ENGINE_VERSION}"
    assert service._npu_copy_key(STEM, 8192)[0] == key != service._npu_copy_key(STEM, 4096)[0]
    assert entry["stem"] == STEM and entry["bytes_estimate"] == 1_000_000_000 and "partial" not in entry
    assert service.npu_copies_line() == ("NPU copies of models: about 1.0 GB of phone storage. iOS keeps them; "
                                         "Glossarion cannot remove them.")
    assert service.npu_status_line() == "Apple Neural Engine (h15) ready"
    lines = selfcheck.npu_lines(service, soc="")
    assert "NPU copies: 1 model, about 1.0 GB of phone storage (estimate)" in lines
    assert service.prefs.get(svc.NPU_ATTEMPT_KEY) is None


def test_a_cache_hit_skips_the_gate_and_adds_no_copy(tmp_path):
    engine = eng.FakeEngine(npu_compile=FRESH, npu_device="Apple Neural Engine (h15)")
    service, engine, _p = ane(tmp_path, engine=engine)
    talk(service)
    before = service.prefs.get(svc.NPU_COPIES_KEY)
    service.unload()
    engine.npu_compile = [dict(s) for s in CACHED]
    engine.npu_step_delay = 0.05
    service.free_storage = lambda path: 1                                    # full: but the copy is in the cache
    seen: list = []
    service.subscribe(lambda snap: seen.append((service.npu_progress() or {}).get("text")))
    assert talk(service) == "Echo: hi" and engine.backends == ["npu", "npu"]
    assert service.prefs.get(svc.NPU_COPIES_KEY) == before
    assert not [t for t in seen if t and "first time only" in t]
    assert any(t and t.startswith("Preparing this model for the NPU ·") for t in seen)


def test_the_storage_gate_falls_back_to_metal(tmp_path):
    engine = eng.FakeEngine(npu_compile=FRESH)
    # needs 0.4 x (2.5 cache + 2 temporary files while loaded) + 2 GB = 3.8 GB
    service, engine, _p = ane(tmp_path, engine=engine, free=3 * GB)
    assert talk(service) == "Echo: hi"
    assert engine.backends == ["npu", "gpu"] and service.snapshot().backend == "GPU"
    reason = service.npu_fallback(STEM)["reason"]
    assert reason == "not enough free storage for the NPU copy of this model (needs about 3.8 GB)"
    assert service.prefs.get(svc.NPU_COPIES_KEY) is None and service.prefs.get(svc.NPU_ATTEMPT_KEY) is None
    assert service.npu_note(STEM) == f"NPU unavailable for {STEM}: {reason}. Using GPU."


@pytest.mark.parametrize("failure", ["load", "storage_refused", "chat"])
def test_every_other_failure_falls_back_to_metal(tmp_path, failure):
    kwargs = {"load": {"npu_error": "the model did not load on the NPU (bad graph)"},
              "storage_refused": {"npu_compile": [FRESH[0], {"compiles_fresh": 1, "storage_refused": 1}]},
              "chat": {"npu_chat_error": "llama_decode returned 1"}}[failure]
    engine = eng.FakeEngine(**kwargs)
    service, engine, _p = ane(tmp_path, engine=engine)
    assert talk(service) == "Echo: hi"
    assert engine.backends[-1] == "gpu" and service.npu_fallback(STEM) is not None
    if failure == "storage_refused":
        assert service.npu_fallback(STEM)["reason"] == eng.ANE_STORAGE_TEXT
    talk(service, text="again")
    assert engine.backends.count("npu") == 1                                  # no retry loop


def test_a_full_phone_blames_only_the_model_whose_pass_ran_out(tmp_path):
    # model A's pass runs out of storage: A's fallback; the backend refuses every compile from here on
    engine = eng.FakeEngine(npu_compile=[FRESH[0], dict(FRESH[1], storage_refused=1)])
    service, engine, _p = ane(tmp_path, engine=engine, stems=(STEM, IQ))
    assert talk(service) == "Echo: hi" and engine.backends == ["npu", "gpu"]
    assert service.npu_fallback(STEM)["reason"] == eng.ANE_STORAGE_TEXT and service._npu_storage_full
    # model B in the same process: never tried on the NPU, and no fallback of its own (a restart can load it)
    assert talk(service, IQ) == "Echo: hi" and engine.backends == ["npu", "gpu", "gpu"]
    assert service.npu_fallback(IQ) is None
    # a load that meets the refusal before its pass (another service of the process) records nothing either
    other, refusing, _p2 = ane(tmp_path / "b", engine=eng.FakeEngine(npu_compile=[{"storage_refused": 1}]))
    assert talk(other) == "Echo: hi" and refusing.backends == ["npu", "gpu"]
    assert other.npu_fallback(STEM) is None and other._npu_storage_full
    assert other.prefs.get(svc.NPU_ATTEMPT_KEY) is None


def test_the_ledger_keeps_what_a_failed_pass_compiled(tmp_path):
    # two families compile fresh, then the logits check fails: the copies stay in the iOS cache
    engine = eng.FakeEngine(npu_compile=FRESH[:3], npu_error="preparing the model for the NPU failed (NaN)")
    service, engine, _p = ane(tmp_path, engine=engine)
    assert talk(service) == "Echo: hi" and engine.backends == ["npu", "gpu"]
    (entry,) = service.prefs.get(svc.NPU_COPIES_KEY).values()
    assert entry["partial"] is True and entry["bytes_estimate"] == 500_000_000   # 200 MB fresh x 2.5
    assert service.npu_copies_line().startswith("NPU copies of models: about 0.5 GB")
    # "Try the NPU again": the partial copy does not skip the gate, and a pass that works completes the entry
    engine.npu_error = None
    engine.npu_compile = [dict(s) for s in FRESH]
    service.retry_npu(STEM)
    service.unload()
    assert talk(service, text="again") == "Echo: again" and engine.backends[-1] == "npu"
    assert len(engine.npu_gates) == 2
    (entry,) = service.prefs.get(svc.NPU_COPIES_KEY).values()
    assert "partial" not in entry and entry["bytes_estimate"] == 500_000_000 + 1_000_000_000


def test_a_pass_killed_half_way_leaves_its_estimate(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Killed(eng.FakeEngine):
        def load(self, *args, **kwargs):
            gate = kwargs.get("npu_gate")
            if gate is not None:
                gate(dict(FRESH[0]))       # the collect pass passed the gate; the app dies during the compiles
                started.set()
                release.wait(5)
            return super().load(*args, **kwargs)

    service, engine, _p = ane(tmp_path, engine=Killed())
    worker = threading.Thread(target=talk, args=(service,))
    worker.start()
    assert started.wait(5)
    (entry,) = service.prefs.get(svc.NPU_COPIES_KEY).values()
    assert entry["partial"] is True and entry["bytes_estimate"] == 1_000_000_000  # 400 MB pending x 2.5
    release.set()
    worker.join(5)


def test_the_copies_line_stays_when_the_npu_is_no_longer_offered(tmp_path):
    copies = {f"{STEM}.gguf|4|8192|{eng.ENGINE_VERSION}": {"stem": STEM, "bytes_estimate": 3_400_000_000}}
    prefs = fakes.FakePrefs({svc.GPU_KEY: True, svc.NPU_COPIES_KEY: copies})
    service, _engine, _p = ane(tmp_path, ane_failing("compile", "the test program did not compile"), prefs=prefs)
    talk(service)
    assert not service.npu_offered()
    assert service.npu_copies_line() == ("NPU copies of models: about 3.4 GB of phone storage. iOS keeps them; "
                                         "Glossarion cannot remove them.")
    assert "NPU copies: 1 model, about 3.4 GB of phone storage (estimate)" in selfcheck.npu_lines(service, soc="")


def test_time_in_the_background_is_no_hang(tmp_path, monkeypatch):
    # in the background (iOS suspends the app and its compile): the watchdog does not run down
    stuck = eng.FakeEngine(npu_compile=FRESH[:2], npu_step_delay=0.01, npu_load_delay=0.5)
    service, stuck, _p = ane(tmp_path, engine=stuck)
    service.npu_watchdog_seconds = 0.15
    service.background_since = 1.0
    assert talk(service) == "Echo: hi"
    assert stuck.backends == ["npu"] and not service.npu_hung and service.npu_fallback(STEM) is None
    # back in the foreground after a suspension: the clock jumped while nothing ran; that gap is no hang either
    real = time.monotonic
    jump = [0.0]
    monkeypatch.setattr(svc.time, "monotonic", lambda: real() + jump[0])
    stuck2 = eng.FakeEngine(npu_compile=FRESH[:2], npu_step_delay=0.01, npu_load_delay=0.4)
    other, stuck2, _p2 = ane(tmp_path / "s", engine=stuck2)
    other.npu_watchdog_seconds = 0.6
    threading.Timer(0.1, lambda: jump.__setitem__(0, 1000.0)).start()
    assert talk(other) == "Echo: hi"
    assert stuck2.backends == ["npu"] and not other.npu_hung and other.npu_fallback(STEM) is None


def test_the_watchdog_counts_from_the_last_progress(tmp_path):
    # a compile that keeps making progress runs longer than the watchdog and is not a hang
    engine = eng.FakeEngine(npu_compile=FRESH, npu_step_delay=0.15)          # 0.6 s in all, progress every 0.15 s
    service, engine, _p = ane(tmp_path, engine=engine)
    service.npu_watchdog_seconds = 0.45
    assert talk(service) == "Echo: hi"
    assert engine.backends == ["npu"] and not service.npu_hung and service.npu_fallback(STEM) is None
    # one that stops moving is: the restart message, Metal for this model
    stuck = eng.FakeEngine(npu_compile=FRESH[:2], npu_step_delay=0.01, npu_load_delay=0.6)
    other, stuck, _p = ane(tmp_path / "s", engine=stuck)
    other.npu_watchdog_seconds = 0.15
    assert talk(other) == "Echo: hi"
    assert stuck.backends == ["npu", "gpu"] and other.npu_hung
    assert other.npu_fallback(STEM)["reason"] == svc.NPU_HANG_REASON
    assert other.snapshot().npu_note == "The NPU did not answer. Restart Glossarion; this model will use the GPU."


def test_the_attempt_marker_carries_the_background_flag_on_ios(tmp_path):
    started, release = threading.Event(), threading.Event()
    markers: list = []

    class Blocking(eng.FakeEngine):
        def load(self, *args, **kwargs):
            if kwargs.get("npu"):
                started.set()
                release.wait(5)
            return super().load(*args, **kwargs)

    service, engine, _p = ane(tmp_path, engine=Blocking())
    worker = threading.Thread(target=talk, args=(service,))
    worker.start()
    assert started.wait(5)
    markers.append(dict(service.prefs.get(svc.NPU_ATTEMPT_KEY)))
    service.on_lifecycle("hide")
    markers.append(dict(service.prefs.get(svc.NPU_ATTEMPT_KEY)))
    # the app dies here (in the background): the next start records nothing
    after, _e, _p2 = ane(tmp_path / "next", prefs=fakes.FakePrefs({svc.NPU_ATTEMPT_KEY: markers[-1]}))
    assert after._recover_npu() is None and after.npu_fallback(STEM) is None
    assert after.prefs.get(svc.NPU_ATTEMPT_KEY) is None
    release.set()
    worker.join(5)
    assert markers[0]["stem"] == STEM and markers[0]["background"] is False and markers[1]["background"] is True
    # killed in the foreground during the compile pass: that model's NPU fallback
    crashed, _e, _p3 = ane(tmp_path / "fg", prefs=fakes.FakePrefs({svc.NPU_ATTEMPT_KEY: markers[0]}))
    assert crashed._recover_npu() == STEM
    assert crashed.npu_fallback(STEM)["reason"] == svc.NPU_CRASH_REASON


def test_the_smoke_report_carries_stage_1(monkeypatch):
    ext = types.SimpleNamespace(npu_backend_shipped=lambda: True, register_npu_backend=lambda *a: {},
                                npu_precheck=lambda: AnePre(ane_failing("device", "iPhone simulator: no Neural Engine"))(),
                                npu_skeleton_source=lambda: {}, device_index=lambda n: 0,
                                npu_backend_name=lambda: "ANE")
    monkeypatch.setattr(selfcheck.importlib, "import_module", lambda name: ext)
    result = selfcheck.npu_backend_check()
    assert result["functions"] is True and result["shipped"] is True and result["backend"] == "ANE"
    steps = {s["name"]: s["ok"] for s in result["precheck"]["steps"]}
    assert steps == {"shipped": True, "device": False}                     # the framework is never touched
    ext.npu_precheck = lambda: (_ for _ in ()).throw(OSError("boom"))
    assert selfcheck.npu_backend_check()["precheck"] == {"error": "OSError: boom", "steps": []}
    assert result["registered"] == [] and result["requested"] is False         # no libggml handle: nothing seen


def test_the_smoke_check_fails_when_an_npu_backend_registered_by_itself(monkeypatch):
    class _Fn:
        def __init__(self, fn):
            self.fn, self.restype, self.argtypes = fn, None, None

        def __call__(self, *a):
            return self.fn(*a)

    registered = {b"ANE"}
    ext = types.SimpleNamespace(npu_backend_shipped=lambda: True, register_npu_backend=lambda *a: {},
                                npu_precheck=lambda: AnePre(ane_failing("device", "iPhone simulator: no Neural Engine"))(),
                                npu_skeleton_source=lambda: {}, device_index=lambda n: 0, npu_backend_name=lambda: "ANE",
                                NPU_STATE={}, GPU_STATE={}, gpu_backend_name=lambda: "Metal",
                                _LIBGGML=types.SimpleNamespace(
                                    ggml_backend_reg_by_name=_Fn(lambda name: 1 if name in registered else 0)))
    monkeypatch.setattr(selfcheck.importlib, "import_module", lambda name: ext)
    with pytest.raises(AssertionError, match=r"NPU backend \['ANE'\] registered although the app never started"):
        selfcheck.npu_backend_check()
    ext.NPU_STATE = ane_state()                                                # the app started it: expected
    assert selfcheck.npu_backend_check()["registered"] == ["ANE"]
    registered.clear()
    ext.NPU_STATE = {}
    assert selfcheck.npu_backend_check()["registered"] == []
    # Metal by its registry name "MTL": a Metal registered by itself is caught too
    registered.add(b"MTL")
    monkeypatch.setenv("GLOSSARION_PLATFORM", "ios")
    with pytest.raises(AssertionError, match=r"GPU backend \['Metal'\] registered although"):
        selfcheck.gpu_backend_check()


def test_the_diagnostics_block_on_the_ane(tmp_path, monkeypatch):
    monkeypatch.setattr(eng, "hex_log_lines", lambda: ["ggml-ane: ANE h15"])
    state = ane_state()
    state["steps"][-1]["detail"] = "test program ran (from the iOS cache)"
    service, engine, _p = ane(tmp_path, state, engine=eng.FakeEngine(npu_device="Apple Neural Engine (h15)"))
    talk(service)
    lines = selfcheck.npu_lines(service, soc="")
    assert lines[0] == "NPU: available"
    import re

    assert re.fullmatch(r"Apple Neural Engine \(h15\) · check \d+\.\d s · start \d+\.\d s "
                        r"\(test program from the iOS cache\)", lines[1])
    assert "✓ framework: Apple Neural Engine h15" in lines and "Use NPU: on" in lines
    assert not any(line.startswith(("Arch:", "NPU files")) or "power mode" in line for line in lines)
    assert "ggml-ane log:" in lines and "  ggml-ane: ANE h15" in lines
    assert svc.NPU_COPIES_KEY in selfcheck.SHARED_PREF_KEYS


# ---- Android stays as it was --------------------------------------------------------------------------------------

def test_android_keeps_its_files_power_mode_and_marker(tmp_path):
    seen: list = []
    holder: dict = {}

    class Recording(eng.FakeEngine):
        def load(self, *args, **kwargs):
            if kwargs.get("npu"):
                service = holder["service"]
                seen.append((service.prefs.get(svc.NPU_ATTEMPT_KEY), service._npu_phase, sorted(kwargs)))
            return super().load(*args, **kwargs)

    service, engine, probe = npu_service(tmp_path, stems=("m-Q4_0",), engine=Recording())
    holder["service"] = service
    service.run_npu_check()
    assert service.npu_backend() == "HTP" and service.npu_needs_files()
    assert service.npu_options() == {eng.NPU_POWER_ENV: "burst"} and service.npu_copies_line() is None
    talk(service, "m-Q4_0")
    (marker, phase, kwargs), = seen
    assert set(marker) == {"stem", "at"} and marker["stem"] == "m-Q4_0"      # no background flag on Android
    assert phase is None and "npu_gate" not in kwargs                        # no compile pass, no storage gate
    assert service.prefs.get(svc.NPU_COPIES_KEY) is None
    assert presets.npu_quant_kind("x-IQ4_XS.gguf") == "cpu"                  # the Hexagon quant rule
