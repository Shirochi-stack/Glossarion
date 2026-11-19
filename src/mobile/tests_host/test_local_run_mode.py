""""Run on" for the on-device engine (``services/local_llm``, owner 2026-11-19): the run modes and where each is
offered, Auto, the move of every older install to Auto (whatever "Use GPU" / "Use NPU" were), ``save_run_mode``, the
fallback chain of every mode step by step, the leftovers of the automatic NPU + GPU build, NPU reads / CPU writes
(``engine.load(handoff=True)``: the service's timing-only load, and ``LlamaEngine``'s two models, its sequence-state
hand-off, the per-call ``generate`` wrapper, its self-test at load and its failures: a refused state, too little memory,
a Stop during the NPU's read), NPU + CPU with a layer split, and the timing's stored format with the older entries'
renames. Fakes only: FakeEngine, FakePrefs, a fake ``llama_cpp`` module.
"""

from __future__ import annotations

import itertools
import logging
import types

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import selfcheck
from glossarion_mobile.services.local_llm import service as svc
from test_local_llm_server import make_service
from test_local_npu import IQ, Q40, Probe, chat, failing_state, npu_service, set_backend, timed_service


@pytest.fixture(autouse=True)
def _no_abort_note(tmp_path, monkeypatch):
    """No test reads or writes a real native abort note."""
    monkeypatch.setattr(eng, "abort_note_path", lambda: str(tmp_path / "no-abort-note.txt"))
    monkeypatch.setattr(eng, "_ABORT_NOTES", {})


def _phone(tmp_path, kind, **kwargs):
    """A service on ``kind``: "npu+gpu" (Snapdragon with Adreno), "npu" (NPU, no GPU backend), "gpu" (iOS 15+ /
    another Android arm64), "none" (x86_64, iOS 13 / 14)."""
    if kind in ("npu+gpu", "npu"):
        service, engine, _probe = npu_service(tmp_path, gpu_backend="OpenCL" if kind == "npu+gpu" else "", **kwargs)
        service.run_npu_check()
        return service, engine
    return make_service(tmp_path, stems=(Q40,), gpu_backend="OpenCL" if kind == "gpu" else None, **kwargs)


# ---- the modes, where they are offered, Auto ------------------------------------------------------------------------

def test_the_mode_list_labels_and_order():
    assert svc.RUN_MODES == ("auto", "npu", "npu_gpu", "npu_gpu_tensor", "cpu_gpu", "gpu", "cpu")
    assert [svc.RUN_MODE_LABELS[m] for m in svc.RUN_MODES] == [
        "Auto", "NPU only", "NPU + GPU", "NPU + GPU, tensor split", "CPU + GPU", "GPU only", "CPU only"]
    assert svc.RUN_MODE_KEY == "local_llm_run_mode" and svc.RUN_MODE_SPLITS == {"npu_gpu": "layer",
                                                                               "npu_gpu_tensor": "tensor"}
    assert not hasattr(svc, "AUTO_SPLIT") and not hasattr(svc, "AUTO_SPLIT_ENABLED")
    for name in ("save_gpu", "use_npu", "save_use_npu", "backend_override", "split_override", "_split_wanted"):
        assert not hasattr(svc.LocalEngineService, name)


@pytest.mark.parametrize("kind,options,auto", [
    ("npu+gpu", ["auto", "npu", "npu_gpu", "npu_gpu_tensor", "cpu_gpu", "gpu", "cpu"], "npu"),
    ("npu", ["auto", "npu", "cpu"], "npu"),
    ("gpu", ["auto", "cpu_gpu", "gpu", "cpu"], "cpu"),
    ("none", [], "cpu"),
])
def test_options_per_platform_and_what_auto_runs_as(tmp_path, kind, options, auto):
    service, _engine = _phone(tmp_path, kind)
    assert service.run_mode_options() == options and service.auto_mode() == auto
    assert service.run_mode() == "auto" and service.run_mode_shown() == "auto"
    assert service.run_mode_label("auto") == {"npu": "Auto (NPU only)", "cpu": "Auto (CPU only)"}[auto]
    assert service.run_mode_note() == {"npu": "Uses the NPU on this phone.", "cpu": "Uses the CPU on this phone."}[auto]


def test_the_npu_options_wait_for_the_check_and_go_with_a_failed_start(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, probe=Probe(failing_state("session", "unsigned PD denied")))
    assert service.run_mode_options() == ["auto", "cpu_gpu", "gpu", "cpu"]  # stage 1 has not answered yet
    service.run_npu_check()
    assert "npu_gpu_tensor" in service.run_mode_options()
    assert service.save_run_mode("npu_gpu_tensor") == "npu_gpu_tensor"
    chat(service)                                                              # stage 2 fails at the first NPU load
    assert service.run_mode_options() == ["auto", "cpu_gpu", "gpu", "cpu"]
    assert service.run_mode_shown() == "cpu_gpu" and service.run_mode() == "npu_gpu_tensor"  # the saved value stays
    assert service.save_run_mode("npu") == "cpu_gpu"                           # refused: not offered now


@pytest.mark.parametrize("saved,kind,shown", [
    ("npu", "gpu", "cpu"), ("npu_gpu", "gpu", "cpu_gpu"), ("npu_gpu_tensor", "gpu", "cpu_gpu"),
    ("npu_gpu", "npu", "cpu"), ("cpu_gpu", "npu", "cpu"), ("gpu", "none", "cpu"), ("bogus", "npu+gpu", "auto"),
    ("cpu", "none", "cpu"),
])
def test_a_saved_mode_the_phone_does_not_offer_shows_what_runs_instead(tmp_path, saved, kind, shown):
    service, _engine = _phone(tmp_path, kind, prefs=fakes.FakePrefs({svc.RUN_MODE_KEY: saved}))
    assert service.run_mode_shown() == shown


def test_notes_labels_and_the_gpu_flags_per_mode(tmp_path):
    service, _engine = _phone(tmp_path, "npu+gpu")
    expected = {
        "npu": "The NPU runs the model. A failure, or a quant it cannot run, uses the CPU.",
        "npu_gpu": "Half the layers on the NPU, half on the GPU. A failure uses the NPU alone.",
        "npu_gpu_tensor": "Every layer split between the NPU and the GPU. A failure uses the NPU alone.",
        "cpu_gpu": "The GPU layers below run on the GPU, the rest on the CPU. Fastest with Q4_0. A failure uses the CPU.",
        "gpu": "Every layer on the GPU. A failure uses the CPU.",
        "cpu": "The CPU runs the model.",
    }
    for mode, note in expected.items():
        assert service.run_mode_note(mode) == note and service.run_mode_label(mode) == svc.RUN_MODE_LABELS[mode]
        service.save_run_mode(mode)
        assert service.run_mode_note() == note
        assert service.use_gpu() is (mode in ("npu_gpu", "npu_gpu_tensor", "cpu_gpu", "gpu"))
        assert service.gpu_layers_apply() is (mode == "cpu_gpu")


def test_save_run_mode_forgets_the_fallbacks_and_never_unloads(tmp_path):
    service, engine = _phone(tmp_path, "npu+gpu")
    seen: list = []
    service.subscribe(lambda snap: seen.append(snap))
    chat(service)
    unloads = engine.unloads
    service._gpu_failed(Q40, "gpu boom")
    service._split_failed(Q40, "layer boom", "layer")
    service._split_failed(Q40, "tensor boom", "tensor")
    service._npu_failed(Q40, "npu boom")
    service._mark_gpu_attempt(Q40)
    count = len(seen)
    assert service.save_run_mode("cpu_gpu") == "cpu_gpu" and len(seen) > count
    assert service.prefs.get(svc.RUN_MODE_KEY) == "cpu_gpu"
    assert service.gpu_fallback() is None and service.prefs.get(svc.GPU_ATTEMPT_KEY) is None
    assert service.split_fallback(Q40, "layer") is None and service.split_fallback(Q40, "tensor") is None
    assert service.npu_fallback(Q40)                                           # kept: "Try the NPU again" forgets it
    assert engine.unloads == unloads and service.loaded_stem == Q40           # no native work from the screen
    for refused in ("", None, "tpu", "AUTO"):
        assert service.save_run_mode(refused) == "cpu_gpu"
    assert service.prefs.get(svc.RUN_MODE_KEY) == "cpu_gpu"


# ---- everyone starts on Auto (owner 2026-11-19) ------------------------------------------------------------------

LEGACY = list(itertools.product((None, True, False), (None, True, False), (None, 0, 8, -1)))


@pytest.mark.parametrize("kind", ["npu+gpu", "gpu", "none"])
@pytest.mark.parametrize("use_gpu,use_npu,layers", LEGACY)
def test_every_older_install_starts_on_auto(tmp_path, caplog, kind, use_gpu, use_npu, layers):
    data = {key: value for key, value in ((svc.GPU_KEY, use_gpu), (svc.NPU_KEY, use_npu),
                                          (svc.GPU_LAYERS_KEY, layers)) if value is not None}
    data[svc.SPLIT_FALLBACK_KEY] = {Q40: {"reason": "the automatic build's tensor split", "backend": "auto"}}
    prefs = fakes.FakePrefs(data)
    with caplog.at_level(logging.INFO, logger="glossarion.local_llm"):
        service, _engine = _phone(tmp_path, kind, prefs=prefs)
    assert prefs.get(svc.RUN_MODE_KEY) == "auto" and service.run_mode() == "auto"
    assert prefs.get(svc.GPU_KEY) is None and prefs.get(svc.NPU_KEY) is None and prefs.get(svc.SPLIT_FALLBACK_KEY) is None
    assert prefs.get(svc.GPU_LAYERS_KEY) == layers                             # the GPU layers setting is kept
    moved = [r.getMessage() for r in caplog.records if r.getMessage().startswith("on-device run mode")]
    assert len(moved) == (1 if use_gpu is not None or use_npu is not None else 0)
    if moved:
        assert moved[0] == (f"on-device run mode: Auto (from Use GPU {'on' if use_gpu else 'off'}, "
                            f"Use NPU {'off' if use_npu is False else 'on'})")


def test_the_move_runs_once_and_never_in_the_parallel_child(tmp_path):
    prefs = fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu_gpu", svc.GPU_KEY: True})
    service, _engine = make_service(tmp_path, prefs=prefs, gpu_backend="OpenCL")
    assert prefs.get(svc.RUN_MODE_KEY) == "cpu_gpu" and prefs.get(svc.GPU_KEY) is True  # already moved: untouched
    fresh = fakes.FakePrefs()
    child, _e = make_service(tmp_path / "c", prefs=fresh, _parallel_child=True)
    assert fresh.get(svc.RUN_MODE_KEY) is None and child.run_mode() == "auto"
    owner, _e2 = make_service(tmp_path / "o", prefs=fresh)
    assert fresh.get(svc.RUN_MODE_KEY) == "auto"                               # a fresh install: Auto
    assert svc.GPU_KEY not in fresh.data and svc.NPU_KEY not in fresh.data and svc.SPLIT_FALLBACK_KEY not in fresh.data


def test_self_check_services_share_the_run_mode(tmp_path):
    app = fakes.FakePrefs({svc.RUN_MODE_KEY: "cpu"})
    prefs = selfcheck._DictPrefs({svc.SETTINGS_KEY: {"n_ctx": 2048}}, shared=app)
    service, _engine = make_service(tmp_path, prefs=prefs, gpu_backend="OpenCL")
    assert service.run_mode() == "cpu" and svc.RUN_MODE_KEY in selfcheck.SHARED_PREF_KEYS
    assert svc.NPU_KEY not in selfcheck.SHARED_PREF_KEYS
    assert selfcheck.npu_summary(service)["run_mode"] == "cpu"


# ---- every chain, step by step --------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,split", [("npu_gpu", "layer"), ("npu_gpu_tensor", "tensor")])
def test_the_npu_gpu_chain_ends_on_the_cpu_one_step_at_a_time(tmp_path, mode, split):
    engine = eng.FakeEngine(split_error={split: "refused"}, npu_error="npu boom", gpu_error="gpu boom")
    service, engine, _probe = npu_service(tmp_path, engine=engine, mode=mode)
    assert chat(service) == "Echo: hi"
    assert engine.backends == [f"npu+gpu:{split}", "npu", "gpu", "cpu"]
    assert service.split_fallback(Q40, split)["reason"] == "refused"
    assert service.npu_fallback(Q40)["reason"] == "npu boom" and service.gpu_fallback()["reason"] == "gpu boom"
    for key in (svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY, svc.GPU_ATTEMPT_KEY):
        assert service.prefs.get(key) is None
    assert service.snapshot().npu_note == f"NPU unavailable for {Q40}: npu boom. Using CPU."   # CPU + GPU dropped too
    chat(service, text="again")
    service.unload()
    chat(service)
    assert engine.backends == [f"npu+gpu:{split}", "npu", "gpu", "cpu", "cpu"]  # no retry loop


@pytest.mark.parametrize("mode,first", [("npu", "npu"), ("auto", "npu")])
def test_npu_only_and_auto_fall_back_to_the_cpu(tmp_path, mode, first):
    service, engine, _probe = npu_service(tmp_path, engine=eng.FakeEngine(npu_error="boom"), mode=mode)
    chat(service)
    assert engine.backends == [first, "cpu"] and service.npu_note() == f"NPU unavailable for {Q40}: boom. Using CPU."
    assert service.gpu_fallback() is None


@pytest.mark.parametrize("mode,layers", [("cpu_gpu", 8), ("gpu", -1)])
def test_the_gpu_modes_fall_back_to_the_cpu(tmp_path, mode, layers):
    service, engine, _probe = npu_service(tmp_path, engine=eng.FakeEngine(gpu_error="boom"), mode=mode)
    chat(service)
    assert engine.backends == ["gpu", "cpu"] and engine.npu_layers == [] and service.gpu_fallback()["reason"] == "boom"
    service.save_run_mode(mode)
    engine.gpu_error = None
    service.unload()
    chat(service)
    assert engine.backends[-1] == "gpu" and engine.n_gpu_layers == layers


def test_cpu_only_has_no_fallback(tmp_path):
    class Broken(eng.FakeEngine):
        def load(self, *args, **kwargs):
            raise ValueError("bad file")

    service, _engine, _probe = npu_service(tmp_path, engine=Broken(), mode="cpu")
    with pytest.raises(svc.LocalLLMError) as caught:
        chat(service)
    assert caught.value.status == 500 and "bad file" in caught.value.message


def test_auto_runs_on_the_cpu_for_a_quant_the_npu_cannot_take_or_a_failed_files_download(tmp_path):
    service, engine, _probe = npu_service(tmp_path, stems=(Q40, IQ))
    chat(service, IQ)
    assert engine.backends == ["cpu"] and service.npu_fallback(IQ) is None
    service.npu_download_error, service._npu_fetch_failed_at = "offline", service.clock()  # a download just failed
    chat(service, Q40)
    assert engine.backends == ["cpu", "cpu"] and service.npu_fallback(Q40) is None  # no record: a later load retries


def test_a_preset_row_npu_skips_the_quant_check_and_falls_back_to_the_cpu(tmp_path):
    service, engine, _probe = npu_service(tmp_path, stems=(IQ,), engine=eng.FakeEngine(npu_error="boom"),
                                          mode="npu_gpu")
    set_backend(service, IQ, "npu")
    chat(service, IQ)
    assert engine.backends == ["npu", "cpu"]                                  # NPU only's chain, not NPU + GPU's


# ---- markers of every mode and the leftovers ------------------------------------------------------------------------

@pytest.mark.parametrize("marker,recorded", [
    ({"split": "layer", "mode": "npu_gpu"}, "layer"),
    ({"split": "tensor", "mode": "npu_gpu_tensor"}, "tensor"),
    ({"split": True}, None),                                                   # the automatic build's in-flight marker
    ({"split": "tensor"}, None),                                               # its attempt marker
    ({"split": "layer", "mode": "npu_gpu_tensor"}, None),                      # not a pair this build writes
])
def test_split_markers_are_recorded_only_when_this_build_wrote_them(tmp_path, marker, recorded):
    prefs = fakes.FakePrefs({svc.NPU_INFLIGHT_KEY: {"stem": Q40, "at": 1, **marker}})
    service, _engine, _probe = npu_service(tmp_path, prefs=prefs)
    assert service._recover_npu() == (Q40 if recorded else None)
    assert prefs.get(svc.NPU_INFLIGHT_KEY) is None and service.npu_fallback(Q40) is None
    for split in ("layer", "tensor"):
        assert bool(service.split_fallback(Q40, split)) is (split == recorded)


def test_split_records_say_which_option_and_need_their_own_split(tmp_path):
    service, _engine, _probe = npu_service(tmp_path)
    service._split_failed(Q40, "why", "tensor")
    assert service.prefs.get(svc.SPLIT_TENSOR_FALLBACK_KEY)[Q40]["split"] == "tensor"
    assert service.prefs.get(svc.SPLIT_FALLBACK_KEY) is None
    service.prefs.set(svc.SPLIT_FALLBACK_KEY, {Q40: {**service.prefs.get(svc.SPLIT_TENSOR_FALLBACK_KEY)[Q40]}})
    assert service.split_fallback(Q40, "layer") is None                        # a tensor entry in the layer key: no


# ---- NPU reads, CPU writes and NPU + CPU in the service (the timing's exact loads) --------------------------------

def test_handoff_loads_both_copies_without_markers_and_reports_the_hand_off(tmp_path):
    seen = {}
    holder = {}

    def reply(messages):
        seen["inflight"] = holder["service"].prefs.get(svc.NPU_INFLIGHT_KEY)
        return "fine"

    service, engine, _probe = npu_service(tmp_path, engine=eng.FakeEngine(reply=reply))
    holder["service"] = service
    service.run_override = "handoff"
    assert service.load_blocking(Q40) == (True, None)
    assert engine.backends == ["npu+cpu:handoff"] and service.snapshot().backend == "NPU + CPU"
    assert service.snapshot().device == "NPU + CPU: Hexagon v81 (HTP0) reads, CPU writes"
    assert service.prefs.get(svc.NPU_ATTEMPT_KEY) is None and not engine.mmproj
    assert chat(service) == "fine" and seen["inflight"] is None               # the timing marks it, not the NPU marker
    assert engine.last_timing["handoff_bytes"] == 3_200_000 and engine.last_timing["handoff_s"] > 0
    assert service.load_blocking(Q40) == (True, None) and engine.backends == ["npu+cpu:handoff"]  # kept
    service.run_override = None
    chat(service)                                                              # a normal request: Auto again
    assert engine.backends == ["npu+cpu:handoff", "npu"]


def test_a_refused_handoff_is_an_error_with_no_fallback(tmp_path):
    engine = eng.FakeEngine(handoff_error="the CPU could not continue from the NPU (incompatible V transposition)")
    service, engine, _probe = npu_service(tmp_path, engine=engine)
    service.run_override = "handoff"
    ok, error = service.load_blocking(Q40)
    assert not ok and error.endswith("incompatible V transposition)") and engine.backends == ["npu+cpu:handoff"]
    assert service.loaded_stem is None and service.npu_fallback(Q40) is None and service.split_fallback(Q40) is None


def test_handoff_needs_memory_for_two_copies(tmp_path):
    service, engine, _probe = npu_service(tmp_path)
    one = svc.ram_need(4, service.context_for(Q40))
    assert service.handoff_need(Q40) == one + int(4 * 1.15) + service.context_for(Q40) * 64 * 1024
    service.memory = lambda: (one + 64 * svc.MIB, None)                        # one copy fits, two do not
    verdict, message = service.ram_check(Q40, handoff=True)
    assert verdict == "refuse" and message.startswith("not enough memory for two copies of the model (needs ")
    assert ", phone has " in message and message.endswith(" GB)")
    service.run_override = "handoff"
    ok, error = service.load_blocking(Q40)
    assert not ok and error == message and engine.backends == []
    service.run_override = None
    timed, _e = timed_service(tmp_path / "t")
    timed.memory = lambda: (svc.ram_need(4, timed.context_for(Q40)) + 64 * svc.MIB, None)
    speeds = selfcheck.npu_speed_check(timed, Q40, backends=("handoff", "cpu"))
    assert speeds["handoff"]["skipped"].startswith("not enough memory for two copies") and speeds["cpu"]["tokens"]


def test_a_stop_during_the_npus_read_ends_the_answer(tmp_path):
    service, engine, _probe = npu_service(tmp_path)
    service.run_override = "handoff"
    service.load_blocking(Q40)
    stopped = {"now": False}
    real = engine.chat

    def chat_then_stop(messages, params, should_stop):
        stopped["now"] = True                                                  # the Stop arrives as the NPU reads
        return real(messages, params, should_stop)

    engine.chat = chat_then_stop

    def should_stop():
        return stopped["now"]

    with pytest.raises(svc.LocalLLMError) as caught:
        service.chat(f"local/{Q40}", {"messages": [{"role": "user", "content": "hi"}]}, should_stop)
    assert caught.value.status == 499 and not service.gate.busy and service.npu_fallback(Q40) is None


def test_npu_cpu_puts_half_the_layers_on_the_npu(tmp_path):
    service, engine, _probe = npu_service(tmp_path)
    service.run_override = "npu_cpu"
    ok, error = service.load_blocking(Q40)
    assert not ok and error == "NPU + CPU: the model's layer count is unknown" and engine.backends == []
    service.gpu_layer_limit = lambda stem=None: 43
    assert service.npu_cpu_layers(Q40) == 21
    assert service.load_blocking(Q40) == (True, None)
    assert engine.backends == ["npu"] and engine.npu_layers == [21] and service.prefs.get(svc.NPU_ATTEMPT_KEY) is None
    engine.npu_chat_error = "graph-compute failed"
    with pytest.raises(RuntimeError):                                          # the timing's load: never answered elsewhere
        chat(service)
    assert service.npu_fallback(Q40) is None and engine.backends == ["npu"]


def test_strict_timing_skips(tmp_path):
    service, _engine = timed_service(tmp_path, prefs=fakes.FakePrefs({svc.GPU_LAYERS_KEY: 0}))
    service.gpu_layer_limit = lambda stem=None: None
    speeds = selfcheck.npu_speed_check(service, Q40, backends=("cpu_gpu", "npu_cpu", "gpu"))
    assert speeds["cpu_gpu"] == {"skipped": "GPU layers is 0"} and speeds["npu_cpu"] == {"skipped": "layer count unknown"}
    assert speeds["gpu"]["backend"] == "GPU" and speeds["gpu"]["layers"] == -1


# ---- the timing's stored format ------------------------------------------------------------------------------------

def test_an_older_timing_entry_is_read_with_the_new_names(tmp_path):
    old = {"version": "x", "stem": Q40, "at": 1, "running": None, "crashed": ["layer"], "notes": {"layer": "abort"},
           "modes": {"npu": {"skipped": "a"}, "gpu": {"skipped": "b"}, "cpu": {"skipped": "c"},
                     "layer": {"skipped": "d"}, "tensor": {"skipped": "e"}}}
    service, _engine = timed_service(tmp_path, prefs=fakes.FakePrefs({svc.SPEED_KEY: old}))
    saved = selfcheck.saved_speeds(service)
    assert saved == {"npu": {"skipped": "a"}, "npu_gpu": {"skipped": "d"}, "cpu_gpu": {"skipped": "b"},
                     "cpu": {"skipped": "c"}, "npu_gpu_tensor": {"skipped": "e"}}
    speeds = selfcheck.npu_speed_check(service, Q40, backends=("npu_gpu",))
    assert speeds["npu_gpu"]["skipped"] == f"{selfcheck.SPEED_CRASHED_SKIPPED}: abort"
    stored = service.prefs.get(svc.SPEED_KEY)
    assert stored["format"] == 2 and "gpu" not in stored["modes"] and stored["modes"]["cpu_gpu"] == {"skipped": "b"}
    # a format-2 entry is read as it is: its "gpu" is GPU only
    service.prefs.set(svc.SPEED_KEY, {**stored, "modes": {"gpu": {"skipped": "f"}}})
    assert selfcheck.saved_speeds(service) == {"gpu": {"skipped": "f"}}


def test_the_card_lines_auto_and_fastest(tmp_path):
    service, _engine = timed_service(tmp_path, mode="cpu_gpu")
    speeds = {"cpu": {"backend": "CPU", "prompt_tok_s": 10.0, "gen_tok_s": 5.0, "tok_s": 4.0},
              "npu": {"backend": "NPU", "prompt_tok_s": 90.0, "gen_tok_s": 4.0, "tok_s": 3.0}}
    lines = selfcheck.speed_lines(speeds, "npu")
    assert lines[0] == "  Auto runs as: NPU only" and lines[-1] == "  Fastest: prompt NPU only · generation CPU only"
    assert selfcheck._fastest({"cpu": speeds["cpu"]}) is None                  # one mode: nothing to compare
    text = selfcheck.npu_lines(service, speeds, soc="")
    assert "Run on: CPU + GPU (Auto: NPU only) · power mode: burst" in text


def test_the_fastest_line_leaves_out_a_mode_that_fell_back():
    speeds = {"cpu": {"backend": "CPU", "prompt_tok_s": 10.0, "gen_tok_s": 5.0, "tok_s": 4.0},
              "gpu": {"backend": "GPU", "prompt_tok_s": 20.0, "gen_tok_s": 3.0, "tok_s": 3.0},
              "npu": {"backend": "CPU", "prompt_tok_s": 40.0, "gen_tok_s": 12.0, "tok_s": 9.0}}   # ran on CPU
    lines = selfcheck.speed_lines(speeds)
    assert "  NPU only: prompt 40.0 tok/s · generation 12.0 tok/s (ran on CPU)" in lines
    assert lines[-1] == "  Fastest: prompt GPU only · generation CPU only"
    assert selfcheck._fastest({"cpu": speeds["cpu"], "npu": speeds["npu"]}) is None


def test_gpu_only_failing_in_the_timing_keeps_cpu_gpu_and_the_tensor_split(tmp_path):
    # GPU only takes no layers on this phone, 8 layers work: the timing's GPU only falls back with its count, the
    # tensor split after it is still timed, and the saved CPU + GPU still loads on the GPU afterwards
    service, engine = timed_service(tmp_path, mode="cpu_gpu")
    service.gpu_layer_limit = lambda stem=None: 24
    real_load = engine.load

    def load(*args, **kwargs):
        if kwargs.get("gpu") and kwargs.get("n_gpu_layers") == -1:
            raise eng.GpuUnavailable("out of device memory (all layers)")
        return real_load(*args, **kwargs)

    engine.load = load
    speeds = selfcheck.npu_speed_check(service, Q40, backends=("cpu_gpu", "gpu", "npu_gpu_tensor"))
    assert speeds["cpu_gpu"]["backend"] == "GPU" and speeds["gpu"]["backend"] == "CPU"
    assert "skipped" not in speeds["npu_gpu_tensor"]
    assert service.gpu_fallback()["layers"] == -1 and service._placement(Q40) == ("gpu", 8, None)
    assert service.gpu_note() is None
    # a CPU + GPU failure (8 layers) keeps the NPU + GPU modes' GPU share off too: they are skipped
    service._gpu_failed(Q40, "the first token on the GPU failed", 8)
    assert selfcheck._speed_skip(service, Q40, "npu_gpu_tensor") == \
        f"GPU unavailable for {Q40}: the first token on the GPU failed"


def test_npu_lines_list_each_dropped_option(tmp_path):
    service, _engine, _probe = npu_service(tmp_path)
    service.run_npu_check()
    service._split_failed(Q40, "refused", "layer")
    service._split_failed(Q40, "aborted", "tensor")
    lines = selfcheck.npu_lines(service, soc="")
    assert f"NPU + GPU dropped: {Q40}: refused" in lines and f"NPU + GPU, tensor split dropped: {Q40}: aborted" in lines
    summary = selfcheck.npu_summary(service)
    assert summary["split_fallbacks"] == {"npu_gpu": {Q40: "refused"}, "npu_gpu_tensor": {Q40: "aborted"}}


# ---- LlamaEngine: NPU reads, CPU writes ------------------------------------------------------------------------------

class _Llama:
    """A fake ``llama_cpp.Llama``: the NPU copy when ``main_gpu`` >= 0, else the CPU copy; ``eval`` decodes one batch
    through ``_ctx.decode`` (where a Stop is seen), ``create_chat_completion`` calls ``self.generate`` like
    ``_create_completion`` does with the prompt tokens."""

    made: list = []
    closed: list = []
    fail: dict = {}
    logits = (0.5, -1.0, 2.0, 0.0)
    prompt = [1, 2, 3, 4, 5, 6]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.role = "npu" if kwargs.get("main_gpu", -1) >= 0 else "cpu"
        if _Llama.fail.get(self.role):
            raise ValueError(_Llama.fail[self.role])
        self.evals: list = []
        self.resets = 0
        self.cleared = 0
        self.input_ids = [0] * 64
        self.n_tokens = 0
        self._requires_eval = True
        self.ctx = f"ctx-{self.role}"
        self.metadata = {}
        self.generated: list = []
        owner = self

        class Ctx:
            def decode(self, batch):
                return 0

            def kv_cache_clear(self):
                owner.cleared += 1

            def get_logits(self):
                return list(_Llama.logits)

        self._ctx = Ctx()
        _Llama.made.append(self)

    def tokenize(self, text, add_bos=True, special=False):
        return [1, 2, 3, 4, 5]

    def reset(self):
        self.resets += 1
        self.n_tokens = 0
        self._requires_eval = True

    def eval(self, tokens):
        tokens = list(tokens)
        self._ctx.decode(types.SimpleNamespace(n_tokens=len(tokens)))
        self.evals.append(tokens)
        self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        self._requires_eval = False

    def n_vocab(self):
        return len(_Llama.logits)

    def close(self):
        _Llama.closed.append(self.role)

    def generate(self, tokens, *args, **kwargs):
        self.generated.append({"tokens": list(tokens), "n_tokens": self.n_tokens,
                               "ids": list(self.input_ids[:self.n_tokens]), "requires_eval": self._requires_eval})
        yield 7

    def create_chat_completion(self, **kwargs):
        def chunks():
            for _token in self.generate(list(_Llama.prompt)):
                yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return chunks()


@pytest.fixture()
def handoff_llama(monkeypatch):
    _Llama.made, _Llama.closed, _Llama.fail, _Llama.logits = [], [], {}, (0.5, -1.0, 2.0, 0.0)
    state = {"size": 3200, "set": 3200, "calls": []}

    def get_size(ctx, seq):
        state["calls"].append(("size", ctx, seq))
        return state["size"]

    def get_data(ctx, buffer, size, seq):
        state["calls"].append(("get", ctx, size, seq))
        return size

    def set_data(ctx, buffer, size, seq):
        state["calls"].append(("set", ctx, size, seq))
        return state["set"]

    module = types.SimpleNamespace(Llama=_Llama, llama_state_seq_get_size=get_size,
                                   llama_state_seq_get_data=get_data, llama_state_seq_set_data=set_data)
    monkeypatch.setattr(eng.LlamaEngine, "_module", lambda self: module)
    monkeypatch.setattr(eng, "_make_formatter", lambda llm: None)
    monkeypatch.setattr(eng, "device_index", lambda name, missing=-1: {"HTP": 2}.get(name, missing))
    monkeypatch.setattr(eng, "capture_aborts", lambda path=None: False)
    monkeypatch.setitem(eng._NPU_STATE, "device", "HTP0")
    monkeypatch.setitem(eng._NPU_STATE, "arch", 81)
    return state


def test_llama_engine_handoff_load_builds_both_and_tests_the_hand_off(handoff_llama):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    reader, llm = _Llama.made
    assert (reader.kwargs["n_gpu_layers"], reader.kwargs["split_mode"], reader.kwargs["main_gpu"]) == (-1, 0, 2)
    assert (llm.kwargs["n_gpu_layers"], llm.kwargs["split_mode"], llm.kwargs["main_gpu"]) == (0, 0, -1)
    for key in ("n_ctx", "n_batch", "n_ubatch", "flash_attn", "swa_full"):     # the same context on both
        assert reader.kwargs[key] == llm.kwargs[key]
    assert "type_k" not in reader.kwargs and "tensor_split" not in reader.kwargs
    # the NPU's first-token check, then the hand-off self-test: N-1 tokens on the NPU, the last on the CPU
    assert reader.evals == [[1, 2, 3, 4, 5], [5], [1, 2, 3, 4]] and llm.evals == [[5]]
    assert handoff_llama["calls"] == [("size", "ctx-npu", 0), ("get", "ctx-npu", 3200, 0), ("set", "ctx-cpu", 3200, 0)]
    assert reader.n_tokens == 0 and llm.n_tokens == 0                          # both left reset
    assert engine.handoff and engine.reader is reader and engine.llm is llm and engine.backend == "NPU + CPU"
    assert not engine.on_npu and not engine.on_gpu and engine.split is None and not engine.vision
    assert engine.device == "NPU + CPU: Hexagon v81 (HTP0) reads, CPU writes"
    engine.unload()
    assert _Llama.closed == ["cpu", "npu"] and engine.reader is None and not engine.handoff


def test_llama_engine_handoff_answers_through_the_per_call_wrapper(handoff_llama):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    reader, llm = _Llama.made
    handoff_llama["calls"].clear()
    text = "".join(engine.chat([{"role": "user", "content": "hi"}], {"max_tokens": 4}, lambda: False))
    assert text == "ok"
    assert reader.evals[-1] == [1, 2, 3, 4, 5]                                 # the exact prompt but its last token
    call = llm.generated[-1]
    # the CPU's own generate gets every token with the restored record: it replays only the last one
    assert call == {"tokens": [1, 2, 3, 4, 5, 6], "n_tokens": 5, "ids": [1, 2, 3, 4, 5], "requires_eval": True}
    assert [c[0] for c in handoff_llama["calls"]] == ["size", "get", "set"]
    assert "generate" not in vars(llm)                                         # the wrapper is gone after the call
    assert engine.last_timing["handoff_bytes"] == 3200 and engine.last_timing["handoff_s"] >= 0
    assert engine.last_timing["prompt_tokens"] == 5                            # the NPU's batch counted as progress


def test_llama_engine_handoff_failures(handoff_llama, monkeypatch):
    engine = eng.LlamaEngine()
    with pytest.raises(ValueError, match="without a split"):
        engine.load("m.gguf", 2048, 4, handoff=True)
    with pytest.raises(ValueError, match="an NPU load"):
        engine.load("m.gguf", 2048, 4, npu=False, split="layer", handoff=True)
    # the CPU refuses the NPU's state at load: both copies freed, the reason says so
    handoff_llama["set"] = 0
    with pytest.raises(eng.SplitUnavailable, match=r"could not continue from the NPU \(the CPU could not read"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    assert _Llama.closed == ["cpu", "npu"] and not engine.loaded and engine.reader is None
    handoff_llama["set"] = 3200
    # an empty state, the NPU's first token, a model that does not load on either side
    handoff_llama["size"] = 0
    with pytest.raises(eng.SplitUnavailable, match="state is empty"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    handoff_llama["size"] = 3200
    _Llama.logits = (float("nan"),) * 4
    with pytest.raises(eng.SplitUnavailable, match="first token on the NPU failed"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    _Llama.logits = (0.5, -1.0, 2.0, 0.0)
    _Llama.fail = {"cpu": "no memory"}
    with pytest.raises(eng.SplitUnavailable, match="did not load on the CPU"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    _Llama.fail = {"npu": "session failed"}
    with pytest.raises(eng.SplitUnavailable, match="did not load on the NPU"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    _Llama.fail = {}
    monkeypatch.setattr(eng, "device_index", lambda name, missing=-1: -1)
    monkeypatch.setitem(eng._NPU_STATE, "error", "")
    with pytest.raises(eng.SplitUnavailable, match="NPU backend is not registered"):
        engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    assert not engine.loaded


def test_llama_engine_handoff_refused_during_an_answer_and_a_stop_in_the_npus_read(handoff_llama):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, npu=True, handoff=True)
    reader, llm = _Llama.made
    handoff_llama["set"] = 0
    with pytest.raises(RuntimeError, match="could not read the NPU's state"):
        "".join(engine.chat([{"role": "user", "content": "hi"}], {"max_tokens": 4}, lambda: False))
    assert "generate" not in vars(llm) and llm.generated == []
    handoff_llama["set"] = 3200
    evals = len(reader.evals)
    with pytest.raises(eng.GenerationStopped):
        "".join(engine.chat([{"role": "user", "content": "hi"}], {"max_tokens": 4}, lambda: True))
    assert len(reader.evals) == evals and llm.generated == []                  # stopped before the NPU's decode
    assert "generate" not in vars(llm) and "decode" not in vars(reader._ctx)   # every wrapper removed
    assert engine.loaded and engine.handoff                                    # the models stay for the next answer
