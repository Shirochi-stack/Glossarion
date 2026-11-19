"""NPU (Qualcomm Hexagon) for the on-device engine, app side (``services/local_llm``): the backend choice per model
(auto / npu / gpu / cpu x the phone's NPU check x "Use NPU" x the quant x a saved fallback), every row of the
fallback matrix, the two crash markers (the attempt marker from an NPU load to its first answer's end, the in-flight
marker around every NPU answer) and their recovery at start, "Try the NPU again", the watchdog, ``device_index`` for
the GPU and the NPU, the wheel's NPU functions through ``engine.npu_state`` (and a wheel without them), the presets'
backend fields, the default slots and a model added from Hugging Face running on the NPU under Auto (no separate
NPU rows), an old saved "· NPU" row still loading on the NPU, the diagnostics block and the
self-check's shared markers; the two stages (``npu_precheck`` once per app version, ``register_npu_backend`` once per
process before the first NPU load) and the first-use download of the HTP skeletons from llama.cpp's release (one GET,
https redirects only, only the pinned members, checksums, a symlink member, an offline phone, cancel and resume, the
release-keyed folder); NPU + GPU together (the engine's layer / tensor placement, its device list and refusals) and
the device checks' five-way timing (stored results, the per-mode crash marker, refusals, Stop). Fakes only: FakeEngine, FakePrefs, fakes of the three wheel functions, tiny tar.gz archives
and a fake HTTP client; no native NPU, no network.
"""

from __future__ import annotations

import contextlib
import itertools
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


# ---- the fake skeleton release (``npu_skeleton_source``) and its archive --------------------------------------------

RELEASE = "b12387"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/llama-{RELEASE}-bin-linux-arm64-snapdragon.tar.gz"
ASSET = "https://release-assets.githubusercontent.com/github-production-release-asset/1/llama.tar.gz"
ARCHS = ("v73", "v75", "v79", "v81", "v85")
SKELETONS = {arch: (f"htp-{arch} ".encode() * 64) for arch in ARCHS}
MEMBER = "llama-{release}/lib/libggml-htp-{arch}.so"


def build_archive(contents=None, *, links=(("llama-b12387/lib/libggml.so", "libggml.so.0"),),
                  extra=(("llama-b12387/bin/llama-cli", b"\x7fELF"),)) -> bytes:
    """A tiny tar.gz like the release's: the five skeletons (``contents``), other files, and symlink members
    ``links`` = ((name, target), ...)."""
    import io
    import tarfile

    contents = SKELETONS if contents is None else contents
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in extra:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
        for arch, data in contents.items():
            info = tarfile.TarInfo(MEMBER.format(release=RELEASE, arch=arch))
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def make_source(archive: bytes, *, skeletons=None, url=URL) -> dict:
    """``npu_skeleton_source()`` of the INTERFACE for ``archive`` (``skeletons``: the pinned contents)."""
    import hashlib

    skeletons = SKELETONS if skeletons is None else skeletons
    return {"release": RELEASE, "commit": "a956a852c9c8bc1fb07206364838ddef299178c0", "url": url,
            "size": len(archive), "sha256": hashlib.sha256(archive).hexdigest(),
            "files": {arch: {"name": f"libggml-htp-{arch}.so", "member": MEMBER.format(release=RELEASE, arch=arch),
                             "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for arch, data in skeletons.items()}}


ARCHIVE = build_archive()
SOURCE = make_source(ARCHIVE)


class FakeResponse:
    def __init__(self, status, body=b"", headers=None, chunk_delay=0.0):
        self.status_code = status
        self.body = body
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Length", str(len(body)))
        self.chunk_delay = chunk_delay
        self.closed = False

    def iter_content(self, chunk_size=1):
        step = max(1, len(self.body) // 8)
        for start in range(0, len(self.body), step):
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
            yield self.body[start:start + step]

    def close(self):
        self.closed = True


class FakeHTTP:
    """The download's HTTP client: github.com answers 302 to the asset host, which serves ``body``. Records every
    request (URL, headers, redirects, timeout)."""

    def __init__(self, body=ARCHIVE, *, redirect=ASSET, error=None, chunk_delay=0.0):
        self.body = body
        self.redirect = redirect
        self.error = error
        self.chunk_delay = chunk_delay
        self.calls: list = []

    def get(self, url, headers=None, stream=False, timeout=None, allow_redirects=True):
        self.calls.append({"url": url, "headers": dict(headers or {}), "stream": stream, "timeout": timeout,
                           "allow_redirects": allow_redirects})
        if self.error is not None:
            raise self.error
        if url == URL and self.redirect:
            return FakeResponse(302, headers={"Location": self.redirect})
        start = int(str((headers or {}).get("Range", "bytes=0-"))[6:].split("-")[0] or 0)
        if start:
            return FakeResponse(206, self.body[start:], {"Content-Range": f"bytes {start}-{len(self.body) - 1}/{len(self.body)}"},
                                chunk_delay=self.chunk_delay)
        return FakeResponse(200, self.body, chunk_delay=self.chunk_delay)


class Pre:
    """A fake ``npu_precheck`` (stage 1): the first three steps of ``state`` (a ``Probe``'s), arch as "v81"."""

    def __init__(self, state=None, delay=0.0):
        self.state = state if state is not None else passing_state()
        self.delay = delay
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        steps = [dict(s) for s in self.state["steps"] if s["name"] in eng.NPU_PRECHECK_STEPS]
        arch = self.state.get("arch")
        failing = next((s for s in steps if not s["ok"]), None)
        return {"arch": f"v{arch}" if arch is not None else None, "error": failing["detail"] if failing else None,
                "steps": steps}


class Probe:
    """A fake ``register_npu_backend`` (``engine.npu_state``'s place), as the INTERFACE says: its "skeletons" step
    passes only when ``skel_dir`` holds the phone's file of the pinned size (else "NPU files not downloaded" and
    nothing registered); records its calls."""

    def __init__(self, state=None, delay=0.0, error=None, source=None):
        self.state = state if state is not None else passing_state()
        self.delay = delay
        self.error = error
        self.source = source or SOURCE
        self.calls: list = []

    def __call__(self, skel_dir, options):
        import os

        self.calls.append((skel_dir, dict(options)))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        state = {**self.state, "steps": [dict(s) for s in self.state["steps"]]}
        spec = self.source["files"].get(f"v{state.get('arch')}")
        path = os.path.join(skel_dir, spec["name"]) if spec else ""
        if not spec or not os.path.isfile(path) or os.path.getsize(path) != spec["size"]:
            for step in state["steps"]:
                if step["name"] == "skeletons":
                    step.update(ok=False, detail=eng.NPU_FILES_MISSING)
            state.update(error=eng.NPU_FILES_MISSING, device="")
        return state


def place_skeleton(root, arch="v81") -> str:
    """The phone's skeleton already on disk (an earlier start downloaded it)."""
    import os

    folder = svc.npu_skel_dir(RELEASE, str(root))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"libggml-htp-{arch}.so")
    with open(path, "wb") as handle:
        handle.write(SKELETONS[arch])
    return path


def npu_service(tmp_path, *, stems=(Q40,), probe=None, prefs=None, gpu=False, engine=None, version=VERSION,
                precheck=None, source=None, http=None, files=True, **kwargs):
    """A service on a phone with the NPU (fakes for both stages, the skeleton release and its HTTP host);
    ``files``: the phone's skeleton is already on disk (else the first NPU load downloads it from ``http``)."""
    prefs = prefs if prefs is not None else fakes.FakePrefs({svc.GPU_KEY: True} if gpu else {})
    probe = probe if probe is not None else Probe()
    precheck = precheck if precheck is not None else Pre(probe.state)
    root = tmp_path / "htp"
    if files and probe.state.get("arch") in (73, 75, 79, 81, 85):
        place_skeleton(root, f"v{probe.state['arch']}")
    service, engine = make_service(tmp_path, stems=stems, prefs=prefs, engine=engine, npu_probe=probe,
                                   npu_precheck=precheck, npu_source=lambda: source or SOURCE,
                                   npu_root=str(root), npu_session=http if http is not None else FakeHTTP(),
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


def test_no_separate_npu_rows_the_existing_slots_are_auto():
    # owner 2026-11-18: no "· NPU" presets and no "Custom · NPU" family; the existing slots use the NPU under Auto
    for name in ("NPU_PRESETS", "NPU_FAMILY", "_npu_row", "_npu_family"):
        assert not hasattr(presets, name)
    assert presets.custom_family("npu") is None
    assert all(presets.backend_of(row) == "auto" for row in presets.DEFAULT_PRESETS)
    for row in presets.DEFAULT_PRESETS[:3]:                                    # Garbage, Light, Medium: UD-Q4_K_XL
        assert presets.npu_quant_kind(row["file"]) == "ok"
    sizes = [s for f in presets.CUSTOM_FAMILIES for s in f["sizes"]]
    assert all("npu_note" not in s for s in sizes)                             # the table is unchanged
    assert [s["name"] for s in sizes if presets.npu_size_note(s)] == ["Qwen 3.5 9B"]
    assert presets.npu_size_note(sizes[-1]) == "" and presets.npu_size_note(None) == ""
    nine = presets.custom_size_for("unsloth/Qwen3.5-9B-GGUF")[1]
    assert presets.npu_size_note(nine) == "over the NPU's 3.5 GB: slower"


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
    row = presets.custom_row(size, hf_file)                                    # a Custom pick is an Auto row
    assert "backend" not in row and row["name"] == "Qwen 3.5 2B · Q4_0" and "NPU" not in row["note"]


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
    probe = Probe(passing_state() if passed else failing_state("libcdsprpc", "unsigned PD denied"))
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
    assert probe.calls[0][0].replace("\\", "/").endswith(f"htp/{RELEASE}")           # keyed on the release, not the app
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


def _register(service, repo, file):
    """A downloaded model (as the one-tap flow leaves it): its file and registry entry; the stem."""
    import os

    stem = file[:-len(".gguf")]
    os.makedirs(service.registry.root, exist_ok=True)
    with open(service.registry.file_path(stem), "wb") as handle:
        handle.write(b"GGUF")
    service.registry.add(stem, {"repo": repo, "file": file, "revision": "r", "sha256": "0" * 64, "size": 4,
                                "downloaded_at": 1})
    return stem


@pytest.mark.parametrize("passed,use_npu,use_gpu,expected", [
    (True, True, False, "npu"), (True, True, True, "npu"),
    (True, False, True, "gpu"), (True, False, False, "cpu"),
    (False, True, True, "gpu"), (False, True, False, "cpu"),
])
def test_the_default_slots_use_the_npu_under_auto(tmp_path, passed, use_npu, use_gpu, expected):
    # Garbage / Light / Medium (UD-Q4_K_XL, saved list untouched: the defaults) and a Hugging Face ID download of a
    # K-quant: no backend on the row, so Auto puts them on the NPU where the check passed and "Use NPU" is on
    prefs = fakes.FakePrefs({svc.GPU_KEY: use_gpu, svc.NPU_KEY: use_npu})
    probe = Probe(passing_state() if passed else failing_state("libcdsprpc", "unsigned PD denied"))
    service, engine, _probe = npu_service(tmp_path, stems=(), probe=probe, prefs=prefs)
    service.run_npu_check()
    stems = [_register(service, row["repo"], row["file"]) for row in presets.DEFAULT_PRESETS[:3]]
    stems.append(_register(service, "someone/Other-GGUF", "Other-7B-Q4_K_M.gguf"))  # no preset row at all
    assert [row["name"] for row in presets.load_presets(service.prefs)][:3] == ["Garbage", "Light", "Medium"]
    for stem in stems:
        assert service.backend_choice(stem) == "auto"
        assert service._placement(stem)[0] == expected
    chat(service, stems[1])
    assert engine.backends == [expected]


def test_an_old_saved_npu_row_still_loads_on_the_npu(tmp_path):
    # a row saved from the earlier test build's "Light · NPU" / Custom · NPU: an ordinary editable row, kept as is
    service, engine, _probe = npu_service(tmp_path, stems=(), prefs=fakes.FakePrefs({svc.NPU_KEY: False}))
    stem = _register(service, "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_0.gguf")
    old = [{"name": "Light · NPU", "repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_0.gguf", "quant": "",
            "note": "Qwen3.5 2B · Q4_0 on the NPU · ~2 GB RAM", "n_ctx": 0, "backend": "npu"},
           {"name": "Qwen 3.5 4B · Q8_0 · NPU", "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q8_0.gguf",
            "quant": "", "note": "Custom · NPU · Qwen 3.5 4B · Q8_0", "n_ctx": 0, "backend": "npu"}]
    presets.save_presets(service.prefs, old)
    rows = presets.load_presets(service.prefs)
    assert [(r["name"], r["backend"]) for r in rows] == [("Light · NPU", "npu"), ("Qwen 3.5 4B · Q8_0 · NPU", "npu")]
    assert service.backend_choice(stem) == "npu"
    chat(service, stem)                                                        # "npu" even with "Use NPU" off
    assert engine.backends == ["npu"]
    clean, error = presets.validate_preset({**rows[0], "name": "Light (mine)"}, rows, 0)  # still editable
    assert error is None and clean["backend"] == "npu" and clean["name"] == "Light (mine)"


# ---- the fallback matrix ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("step", ["libcdsprpc", "arch", "skeletons", "registry", "session"])
def test_a_failing_check_hides_the_npu(tmp_path, step):
    # stage 1 (libcdsprpc, arch) hides it at once; stage 2 (skeletons, registry, session) runs before the first NPU
    # load: that load goes to the GPU and the NPU is not offered for this app version afterwards (A6)
    service, engine, _probe = npu_service(tmp_path, probe=Probe(failing_state(step, f"{step} failed")), gpu=True)
    service.run_npu_check()
    assert service.npu_offered() is (step not in eng.NPU_PRECHECK_STEPS)
    chat(service)
    assert not service.npu_offered() and service.npu_failure() == (step, f"{step} failed")
    assert engine.backends == ["gpu"] and service.snapshot().npu_note is None and service.npu_fallback(Q40) is None
    if step not in eng.NPU_PRECHECK_STEPS:
        assert service.prefs.get(svc.NPU_SESSION_KEY)["state"]["error"] == f"{step} failed"
        again, _e, probe2 = npu_service(tmp_path / "again", prefs=service.prefs)  # the next start: still off
        assert not again.npu_offered() and again._placement(Q40)[0] == "gpu" and probe2.calls == []


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
    pre = Pre()
    newer, _e, probe2 = npu_service(tmp_path / "n", prefs=prefs, version="9.16.1/llama-0.3.37", precheck=pre)
    assert newer.npu_offered() is False and newer.run_npu_check() and pre.calls == 1 and newer.npu_offered()
    assert probe2.calls == []                                                  # stage 2 waits for an NPU load


def test_a_start_the_app_died_in_keeps_the_npu_off_until_the_version_changes(tmp_path):
    prefs = fakes.FakePrefs({svc.NPU_CHECK_ATTEMPT_KEY: {"version": VERSION, "at": 1, "stage": 2}, svc.GPU_KEY: True})
    service, engine, probe = npu_service(tmp_path, prefs=prefs)
    service.start()
    try:
        service.run_npu_check()
        assert not service.npu_offered() and service.npu_failure() == ("session", svc.NPU_START_CRASH_REASON)
        chat(service)
        assert engine.backends == ["gpu"] and probe.calls == []
    finally:
        service.stop()
        svc.set_default_service(None)


def test_the_check_runs_in_the_background_at_start_and_marks_itself(tmp_path):
    marks = []
    holder = {}

    class Watching(Pre):
        def __call__(self):
            marks.append(holder["prefs"].get(svc.NPU_CHECK_ATTEMPT_KEY))
            return super().__call__()

    prefs = fakes.FakePrefs()
    holder["prefs"] = prefs
    service, _engine, probe = npu_service(tmp_path, prefs=prefs, precheck=Watching())
    service.start()
    try:
        deadline = time.monotonic() + 5
        while service._npu_state is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.npu_offered() and marks[0]["version"] == VERSION and marks[0]["stage"] == 1
        assert prefs.get(svc.NPU_CHECK_ATTEMPT_KEY) is None and probe.calls == []  # stage 2 waits for a load
        saved = prefs.get(svc.NPU_CHECK_KEY)
        assert saved["blocked"] is False and saved["source"]["release"] == RELEASE
    finally:
        service.stop()
        svc.set_default_service(None)
    # the next start shows the rows from the saved check, and never runs stage 1 again for this version
    pre = Pre(delay=5)
    later, _e, _p = npu_service(tmp_path / "l", prefs=prefs, precheck=pre)
    later.start()
    try:
        assert later._npu_state is None and later.npu_offered() and later.npu_progress() is None
        assert later.run_npu_check() and pre.calls == 0
    finally:
        later.stop()
        svc.set_default_service(None)


@pytest.mark.parametrize("pre", ["no backend", "no dsp", "passing"])
def test_a_quick_check_never_shows_checking(tmp_path, pre):
    # a build without the backend and a phone without a Hexagon DSP answer at once: the screens never see a bar
    state = {"no backend": failing_state("shipped", eng.NO_NPU), "no dsp": failing_state("libcdsprpc", "dlopen failed"),
             "passing": passing_state()}[pre]
    service, _engine, _probe = npu_service(tmp_path, stems=(), precheck=Pre(state))
    seen: list = []
    service.subscribe(lambda snap: seen.append(service.npu_progress()))
    service.start()
    try:
        deadline = time.monotonic() + 5
        while service._npu_state is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service._npu_state is not None and service.npu_progress() is None
        assert not [p for p in seen if p is not None]
    finally:
        service.stop()
        svc.set_default_service(None)
    # a slow one shows it once it has run npu_quiet_seconds
    slow, _e, _p = npu_service(tmp_path / "s", stems=(), precheck=Pre(delay=0.4))
    slow.npu_quiet_seconds = 0.05
    shown: list = []
    slow.subscribe(lambda snap: shown.append((slow.npu_progress() or {}).get("phase")))
    slow.run_npu_check()
    assert "checking" in shown and slow.npu_progress() is None


def test_the_watchdog_on_a_hanging_check(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, precheck=Pre(delay=0.5))
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
    service, engine, _probe = npu_service(tmp_path, stems=(Q40, QK), gpu=True, engine=eng.FakeEngine(delay=0.02))
    service.run_npu_check()
    chat(service)
    service._npu_fallbacks()
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert set(speeds) == set(selfcheck.SPEED_MODES) and speeds["npu"]["backend"] == "NPU"
    assert speeds["gpu"]["backend"] == "GPU" and speeds["cpu"]["backend"] == "CPU"
    assert speeds["layer"]["backend"] == speeds["tensor"]["backend"] == "NPU + GPU"
    assert all(speeds[b]["tokens"] > 0 for b in speeds) and service.backend_override is None
    assert service.loaded_stem is None                                         # reloads on its usual backend
    lines = selfcheck.npu_lines(service, speeds, soc="QTI SM8850")
    text = "\n".join(lines)
    assert lines[0] == "NPU: available" and "SoC: QTI SM8850" in lines
    import re

    assert re.fullmatch(r"Stage 1 \(check\): \d+\.\d s · Stage 2 \(start\): \d+\.\d s", lines[2])
    assert lines[3] == f"NPU files: llama.cpp {RELEASE} · libggml-htp-v81.so on the phone"
    assert not any(line.startswith("Last download error") for line in lines)
    for name in eng.NPU_STEPS:
        assert f"✓ {name}: {name} ok" in lines
    assert "Arch: v81" in lines and "Device: HTP0" in lines
    assert "Use NPU: on · power mode: burst" in lines
    assert "Last load: m-Q4_0 on NPU + GPU" in text                           # the tensor split run was the last
    assert text.count("tok/s") == 10 and "Speed (16 tokens):" in lines
    assert "  NPU + GPU, tensor split: prompt " in text and " (ran on" not in text
    assert "Fallback: none" in lines
    assert "ggml-hex log:" in lines and "  ggml-hex: HTP0 new session : session-id 0" in lines
    service._npu_failed(QK, "the first token on the NPU failed")
    assert f"Fallback: {QK}: the first token on the NPU failed" in selfcheck.npu_lines(service, soc="")


@pytest.mark.parametrize("step", eng.NPU_STEPS)
def test_diagnostics_line_for_each_failing_step(tmp_path, step):
    service, _engine, _probe = npu_service(tmp_path, probe=Probe(failing_state(step, "dlopen failed")))
    service.run_npu_check()
    chat(service)                                                              # stage 2 runs before the NPU load
    lines = selfcheck.npu_lines(service, soc="x")
    assert lines[0] == f"NPU: not available ({step}: dlopen failed)"
    assert lines[1].startswith("Stage 1 (check): 0.") and lines[1].endswith(
        "Stage 2 (start): not run" if step in eng.NPU_PRECHECK_STEPS else " s")
    if step not in eng.NPU_PRECHECK_STEPS:
        assert lines[2] == f"NPU files: llama.cpp {RELEASE} · libggml-htp-v81.so on the phone"


def test_diagnostics_line_without_the_npu(tmp_path):
    service, _engine = make_service(tmp_path)
    assert selfcheck.npu_lines(service) == ["NPU: not available (shipped: this build has no NPU (Android arm64 only))"]
    assert selfcheck.npu_lines(None)[0].startswith("NPU: not available (engine:")
    service2, _e, _p = npu_service(tmp_path / "b", probe=Probe(passing_state(arch=69)))
    service2.run_npu_check()
    lines = selfcheck.npu_lines(service2, soc="")
    assert lines[0] == "NPU: not available (arch: unsupported HTP arch 69)" and len(lines) == 2
    assert lines[1].startswith("Stage 1 (check): ") and lines[1].endswith("Stage 2 (start): not run")
    speeds = selfcheck.npu_speed_check(service2, Q40)
    assert "NPU not available (arch" in speeds["npu"]["skipped"] and speeds["cpu"]["tokens"] > 0


def test_speed_check_refuses_during_a_job(tmp_path):
    service, _engine, _probe = npu_service(tmp_path, active_job_model=lambda: f"local/{Q40}")
    assert "error" in selfcheck.npu_speed_check(service, Q40)


# ---- NPU + GPU together: the engine's placement and the five-way timing ---------------------------------------------

@pytest.fixture()
def split_engine(fake_llama, monkeypatch):
    """LlamaEngine over the fake Llama, with the NPU registered (``device_index``), two device handles and a recorded
    device list (``_model_devices``)."""
    indexes = {"HTP": 0, "OpenCL": 1}
    lists: list = []
    monkeypatch.setattr(eng, "device_index", lambda name, missing=-1: indexes.get(name, -1))
    monkeypatch.setattr(eng, "_device_handles", lambda names: [{"HTP": 0x1110, "OpenCL": 0x2220}[n] for n in names])

    import contextlib

    @contextlib.contextmanager
    def devices(handles):
        lists.append(list(handles))
        yield

    monkeypatch.setattr(eng, "_model_devices", devices)
    monkeypatch.setitem(eng._NPU_STATE, "device", "HTP0")
    monkeypatch.setitem(eng._NPU_STATE, "arch", 81)
    return types.SimpleNamespace(gpu=fake_llama, indexes=indexes, lists=lists)


def test_engine_split_placement_and_nothing_leaks_into_other_loads(split_engine):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4, npu=True, split="layer")
    layer = _FakeLlama.made[-1]
    assert (layer.kwargs["n_gpu_layers"], layer.kwargs["split_mode"], layer.kwargs["main_gpu"]) == (-1, 1, 0)
    assert layer.kwargs["tensor_split"] == [1.0, 1.0] and layer.kwargs["flash_attn"] is True
    assert "type_k" not in layer.kwargs and "type_v" not in layer.kwargs
    assert split_engine.lists == [[0x1110, 0x2220]]                            # the NPU first, then the GPU
    assert layer.evals == [[1, 2, 3, 4], [4]]                                  # the first-token check ran
    assert engine.split == "layer" and not engine.on_npu and not engine.on_gpu and engine.backend == "NPU + GPU"
    assert engine.device == "NPU + GPU (layer split): Hexagon v81 (HTP0) + QUALCOMM Adreno(TM) 830 (OpenCL)"
    engine.load("m.gguf", 2048, 4, npu=True, split="tensor")
    tensor = _FakeLlama.made[-1]
    assert layer.closed and (tensor.kwargs["split_mode"], tensor.kwargs["n_gpu_layers"]) == (3, -1)
    assert (tensor.kwargs["type_k"], tensor.kwargs["type_v"]) == (eng.GGML_TYPE_F16, eng.GGML_TYPE_F16) == (1, 1)
    assert tensor.kwargs["flash_attn"] is True and tensor.kwargs["tensor_split"] == [1.0, 1.0]
    # the loads after it: today's placement, no device list, no KV type, no tensor split
    for kwargs, expected in ((dict(npu=True, n_gpu_layers=-1), (-1, 0, 0)), (dict(gpu=True), (8, 0, 1)),
                             ({}, (0, 0, -1))):
        engine.load("m.gguf", 2048, 4, **kwargs)
        made = _FakeLlama.made[-1].kwargs
        assert (made["n_gpu_layers"], made["split_mode"], made["main_gpu"]) == expected
        assert not {"type_k", "type_v", "tensor_split"} & set(made) and engine.split is None
    assert len(split_engine.lists) == 2
    engine.unload()
    assert engine.split is None and engine.backend is None


def test_engine_split_refusals_raise_split_unavailable(split_engine, monkeypatch):
    engine = eng.LlamaEngine()
    with pytest.raises(ValueError, match="is an NPU load"):                           # a split is an NPU load
        engine.load("m.gguf", 2048, 4, split="layer")
    with pytest.raises(ValueError, match="unknown"):
        engine.load("m.gguf", 2048, 4, npu=True, split="row")
    split_engine.indexes.pop("HTP")
    monkeypatch.setitem(eng._NPU_STATE, "error", "")
    with pytest.raises(eng.SplitUnavailable, match="NPU backend is not registered"):
        engine.load("m.gguf", 2048, 4, npu=True, split="layer")
    split_engine.indexes["HTP"] = 0
    split_engine.gpu.update(device=None, error="no OpenCL driver")
    with pytest.raises(eng.SplitUnavailable, match="no OpenCL driver"):
        engine.load("m.gguf", 2048, 4, npu=True, split="tensor")
    split_engine.gpu.update(device="QUALCOMM Adreno(TM) 830", error=None)

    def missing(names):
        raise eng.SplitUnavailable("the OpenCL backend has no device")

    monkeypatch.setattr(eng, "_device_handles", missing)
    with pytest.raises(eng.SplitUnavailable, match="OpenCL backend has no device"):
        engine.load("m.gguf", 2048, 4, npu=True, split="layer")
    monkeypatch.setattr(eng, "_device_handles", lambda names: [1, 2])
    assert not _FakeLlama.made                                                 # nothing was built so far
    # llama.cpp refuses the model: its own error line is the reason
    _FakeLlama.fail = "Failed to load model from file: m.gguf"
    monkeypatch.setattr(eng, "_model_devices", _refuse_with_native_line)
    with pytest.raises(eng.SplitUnavailable, match=r"did not load on the NPU \+ GPU \(Failed to load model from file: "
                                                   r"m\.gguf\): .*not implemented for architecture 'gemma3n'$"):
        engine.load("m.gguf", 2048, 4, npu=True, split="tensor")
    _FakeLlama.fail = None
    _FakeLlama.logits = (math.nan, 1.0, 2.0, 3.0)
    with pytest.raises(eng.SplitUnavailable, match=r"first token on the NPU \+ GPU failed"):
        engine.load("m.gguf", 2048, 4, npu=True, split="layer")
    assert _FakeLlama.made[-1].closed and not engine.loaded and engine.split is None


@contextlib.contextmanager
def _refuse_with_native_line(handles):
    """``_model_devices`` while llama.cpp logs why it refuses (the error log is cleared when the load starts), then
    the generic line it logs after every failed load (llama.cpp's llama_model_load_from_file_impl)."""
    eng._remember_hex("llama_model_load: error loading model: LLAMA_SPLIT_MODE_TENSOR not implemented for "
                      "architecture 'gemma3n'\n", eng.NATIVE_ERROR_LEVEL)
    eng._remember_hex("llama_model_load_from_file_impl: failed to load model\n", eng.NATIVE_ERROR_LEVEL)
    yield


def test_native_errors_keep_only_error_lines():
    eng._NATIVE_ERRORS.clear()
    eng._remember_hex("llama_model_loader: loaded meta data\n", 2)
    eng._remember_hex("llama_init_from_model: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n",
                      eng.NATIVE_ERROR_LEVEL)
    assert eng.native_errors() == ["llama_init_from_model: SPLIT_MODE_TENSOR requires flash_attn to be enabled"]
    assert eng.native_reason() == eng.native_errors()[0]
    eng._remember_hex("llama_model_load_from_file_impl: failed to load model\n", eng.NATIVE_ERROR_LEVEL)
    assert eng.native_reason() == "llama_init_from_model: SPLIT_MODE_TENSOR requires flash_attn to be enabled"
    eng._NATIVE_ERRORS.clear()
    eng._remember_hex("llama_model_load_from_file_impl: failed to load model\n", eng.NATIVE_ERROR_LEVEL)
    assert eng.native_reason() is None                                         # the generic line is no reason
    eng._NATIVE_ERRORS.clear()


def test_model_devices_fills_the_device_list_for_the_loading_thread_only(monkeypatch):
    import ctypes
    import sys

    class Params(ctypes.Structure):  # llama_model_params' first fields (llama_cpp.py at the pin)
        _fields_ = [("devices", ctypes.c_void_p), ("tensor_buft_overrides", ctypes.c_void_p),
                    ("n_gpu_layers", ctypes.c_int32), ("split_mode", ctypes.c_int)]

    module = types.SimpleNamespace(llama_model_default_params=lambda: Params())
    original = module.llama_model_default_params
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", module)
    seen = {}
    with eng._model_devices([0x1110, 0x2220]):
        params = module.llama_model_default_params()
        listed = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
        seen["list"] = [listed[0], listed[1], listed[2]]
        other = threading.Thread(target=lambda: seen.update(other=module.llama_model_default_params().devices))
        other.start()
        other.join()
    assert seen == {"list": [0x1110, 0x2220, None], "other": None}             # NULL-terminated, this thread only
    assert module.llama_model_default_params is original and module.llama_model_default_params().devices is None


def test_device_handles_in_the_asked_order(monkeypatch):
    class Fn:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    regs = {b"HTP": 7, b"OpenCL": 9}
    lib = types.SimpleNamespace(ggml_backend_reg_by_name=Fn(lambda name: regs.get(name, 0)),
                                ggml_backend_reg_dev_count=Fn(lambda reg: 1 if reg == 7 else 0),
                                ggml_backend_reg_dev_get=Fn(lambda reg, i: reg * 100 + i))
    monkeypatch.setattr(eng, "_extensions", lambda: types.SimpleNamespace(_LIBGGML=lib))
    assert eng._device_handles(("HTP",)) == [700]
    with pytest.raises(eng.SplitUnavailable, match="the OpenCL backend has no device"):
        eng._device_handles(("HTP", "OpenCL"))
    with pytest.raises(eng.SplitUnavailable, match="the Metal backend is not registered"):
        eng._device_handles(("Metal",))
    monkeypatch.setattr(eng, "_extensions", lambda: None)
    with pytest.raises(eng.SplitUnavailable, match="cannot choose the devices"):
        eng._device_handles(("HTP",))


def timed_service(tmp_path, **kwargs):
    kwargs.setdefault("engine", eng.FakeEngine(delay=0.02))
    service, engine, probe = npu_service(tmp_path, gpu=True, **kwargs)
    service.run_npu_check()
    return service, engine


def test_speed_check_times_five_modes_and_stores_them(tmp_path):
    service, engine = timed_service(tmp_path)
    texts: list = []
    speeds = selfcheck.npu_speed_check(service, Q40, progress=texts.append)
    assert list(speeds) == list(selfcheck.SPEED_MODES)
    assert engine.backends == ["npu", "gpu", "cpu", "npu+gpu:layer", "npu+gpu:tensor"]
    for mode, result in speeds.items():
        assert result["backend"] == selfcheck.SPEED_BACKENDS[mode]
        assert result["prompt_tok_s"] > 0 and result["gen_tok_s"] > 0 and result["prompt_tokens"] > 0
    assert speeds["tensor"]["device"] == "NPU + GPU (tensor split): Hexagon v81 (HTP0) + Fake GPU (Test)"
    assert [t.splitlines()[0] for t in texts] == [f"Timing local/{Q40}: {selfcheck.SPEED_LABELS[m]} ({i} of 5)…"
                                                  for i, m in enumerate(selfcheck.SPEED_MODES, 1)]
    assert texts[-1].splitlines()[1].startswith("  NPU: prompt ")              # the modes done so far follow
    stored = service.prefs.get(svc.SPEED_KEY)
    assert stored["running"] is None and stored["stem"] == Q40 and stored["version"] == VERSION
    assert stored["modes"] == speeds and selfcheck.saved_speeds(service) == speeds
    # the timing changed nothing: no override, no marker, no fallback; "auto" still loads on the NPU alone
    assert service.split_override is None and service.backend_override is None and service.loaded_stem is None
    for key in (svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY, svc.GPU_ATTEMPT_KEY, svc.GPU_FALLBACK_KEY):
        assert not service.prefs.get(key)
    assert service.npu_fallback(Q40) is None
    assert chat(service) == "Echo: hi" and engine.backends[-1] == "npu" and engine.calls[-1]["split"] is None
    text = "\n".join(selfcheck.npu_lines(service, soc=""))                     # Refresh shows the stored ones
    assert "Speed (16 tokens, last timing):" in text and text.count("tok/s") == 10


def test_without_the_npu_the_combined_modes_are_not_listed(tmp_path):
    service, engine = timed_service(tmp_path, probe=Probe(passing_state(arch=69)))
    assert selfcheck.speed_modes(service) == ("npu", "gpu", "cpu")
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert list(speeds) == ["npu", "gpu", "cpu"] and "npu+gpu:layer" not in engine.backends
    assert "NPU not available (arch" in speeds["npu"]["skipped"]
    plain, _engine = make_service(tmp_path / "plain", stems=(Q40,), gpu_backend="OpenCL")
    assert selfcheck.speed_modes(plain) == ("npu", "gpu", "cpu")
    assert "NPU + GPU" not in "\n".join(selfcheck.speed_lines(speeds))


def test_a_refused_split_reports_why_and_the_other_modes_run(tmp_path):
    why = ("the model did not load on the NPU + GPU (Failed to load model from file): llama_model_load: error loading "
           "model: LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'gemma3n'")
    service, engine = timed_service(tmp_path, engine=eng.FakeEngine(delay=0.02, split_error={"tensor": why}))
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert speeds["tensor"]["error"] == f"Could not load local/{Q40}: {why}"
    assert speeds["layer"]["gen_tok_s"] > 0 and speeds["cpu"]["gen_tok_s"] > 0
    assert engine.backends == ["npu", "gpu", "cpu", "npu+gpu:layer", "npu+gpu:tensor"]  # no CPU reload after it
    assert service.npu_fallback(Q40) is None and service.gpu_fallback() is None
    lines = selfcheck.speed_lines(speeds)
    assert lines[-1] == f"  NPU + GPU, tensor split: Could not load local/{Q40}: {why}"
    assert service.prefs.get(svc.SPEED_KEY)["modes"]["tensor"] == speeds["tensor"]


def test_a_failing_split_answer_is_reported_without_a_fallback(tmp_path):
    service, engine = timed_service(tmp_path, engine=eng.FakeEngine(delay=0.02, split_chat_error="ggml abort"))
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert speeds["layer"] == speeds["tensor"] == {"backend": "NPU + GPU", "error": "RuntimeError: ggml abort"}
    assert speeds["npu"]["gen_tok_s"] > 0
    assert service.npu_fallback(Q40) is None and not service.prefs.get(svc.NPU_INFLIGHT_KEY)
    assert engine.backends.count("cpu") == 1                                   # nothing answered it elsewhere


def test_a_model_with_an_npu_fallback_skips_the_combined_modes(tmp_path):
    service, engine = timed_service(tmp_path)
    service._npu_failed(Q40, "the first token on the NPU failed")
    speeds = selfcheck.npu_speed_check(service, Q40)
    reason = f"NPU unavailable for {Q40}: the first token on the NPU failed"
    assert speeds["layer"] == {"skipped": reason} == speeds["tensor"]
    assert speeds["npu"]["backend"] == "GPU"                                   # today's rule: it ran elsewhere
    assert "(ran on GPU)" in selfcheck.speed_lines(speeds)[0]


def test_split_override_without_the_npu_does_not_load(tmp_path):
    service, engine = npu_service(tmp_path, gpu=True, probe=Probe(failing_state("session", "no session")))[:2]
    service.run_npu_check()
    service.split_override = "layer"
    ok, error = service.load_blocking(Q40)
    assert not ok and error == "NPU not available (session: no session)" and not engine.backends
    service.split_override = None
    assert service.load_blocking(Q40) == (True, None) and engine.backends == ["gpu"]


def test_a_mode_the_app_died_in_is_reported_and_skipped_once(tmp_path):
    seen: list = []

    class Watching(eng.FakeEngine):
        def load(self, *args, **kwargs):
            seen.append((kwargs.get("split") or ("npu" if kwargs.get("npu") else "x"),
                         (service.prefs.get(svc.SPEED_KEY) or {}).get("running"), service.prefs.flushes))
            super().load(*args, **kwargs)

    prefs = fakes.FakePrefs({svc.GPU_KEY: True})
    prefs.flushes = 0

    def flush():
        prefs.flushes += 1

    prefs.flush = flush
    service, engine = timed_service(tmp_path, engine=Watching(delay=0.02), prefs=prefs)
    selfcheck.npu_speed_check(service, Q40)
    marked = {mode: (running, flushes) for mode, running, flushes in seen}
    assert marked["layer"][0] == "layer" and marked["tensor"][0] == "tensor"   # on disk before the native load
    assert marked["npu"][0] == "npu" and all(flushes > 0 for _r, flushes in marked.values())
    # the app died while the layer split ran: the marker is still there at the next start
    stored = prefs.get(svc.SPEED_KEY)
    prefs.set(svc.SPEED_KEY, {**stored, "running": "layer"})
    assert selfcheck.saved_speeds(service)["layer"] == {"crashed": "crashed last time"}
    assert "  NPU + GPU, layer split: crashed last time" in selfcheck.npu_lines(service, soc="")
    engine.backends.clear()
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert speeds["layer"]["skipped"] == selfcheck.SPEED_CRASHED_SKIPPED and "npu+gpu:layer" not in engine.backends
    assert speeds["tensor"]["gen_tok_s"] > 0                                   # the other modes still ran
    assert "  NPU + GPU, layer split: crashed last time, skipped (time again to try it)" in \
        selfcheck.speed_lines(speeds)
    assert prefs.get(svc.SPEED_KEY)["running"] is None
    engine.backends.clear()
    speeds = selfcheck.npu_speed_check(service, Q40)                          # asked again: it runs
    assert speeds["layer"]["gen_tok_s"] > 0 and "npu+gpu:layer" in engine.backends


def test_stop_ends_the_timing_and_a_second_timing_is_refused(tmp_path):
    service, engine = timed_service(tmp_path)
    stop = threading.Event()

    def progress(text):
        if "(2 of 5)" in text:
            stop.set()

    speeds = selfcheck.npu_speed_check(service, Q40, should_stop=stop.is_set, progress=progress)
    assert speeds["npu"]["gen_tok_s"] > 0 and engine.backends == ["npu"]
    assert all(speeds[m] == {"skipped": "stopped"} for m in ("gpu", "cpu", "layer", "tensor"))
    assert service.prefs.get(svc.SPEED_KEY)["running"] is None and service.loaded_stem is None
    # a Stop during an answer ends it at the next stop check
    stop.clear()
    slow = eng.FakeEngine(delay=0.05, reply=lambda messages: " ".join(["word"] * 40))
    service2, _e = timed_service(tmp_path / "b", engine=slow)
    threading.Timer(0.3, stop.set).start()
    speeds = selfcheck.npu_speed_check(service2, Q40, should_stop=stop.is_set)
    assert speeds["npu"] == {"skipped": "stopped"} and slow.generated < 40
    with selfcheck._SPEED_LOCK:
        assert selfcheck.npu_speed_check(service, Q40) == {"error": selfcheck.SPEED_RUNNING}


def test_a_stopped_timing_keeps_the_earlier_results_of_the_modes_it_did_not_reach(tmp_path):
    service, engine = timed_service(tmp_path)
    first = selfcheck.npu_speed_check(service, Q40)
    stop = threading.Event()
    speeds = selfcheck.npu_speed_check(service, Q40, should_stop=stop.is_set,
                                       progress=lambda text: stop.set() if "(2 of 5)" in text else None)
    assert speeds["gpu"] == {"skipped": "stopped"}                             # the card shows this run
    stored = service.prefs.get(svc.SPEED_KEY)["modes"]
    assert stored["npu"] == speeds["npu"]
    assert all(stored[m] == first[m] for m in ("gpu", "cpu", "layer", "tensor"))
    # another model's timing does not mix in the first one's numbers
    service.prefs.set(svc.SPEED_KEY, {**service.prefs.get(svc.SPEED_KEY), "stem": "other"})
    selfcheck.npu_speed_check(service, Q40, should_stop=lambda: True)
    assert service.prefs.get(svc.SPEED_KEY)["modes"] == {m: {"skipped": "stopped"} for m in selfcheck.SPEED_MODES}


def test_each_mode_keeps_its_marker_until_its_model_is_freed(tmp_path):
    freed: list = []

    class Watching(eng.FakeEngine):
        def unload(self):
            if self.path is not None:
                freed.append((self.backends[-1], (service.prefs.get(svc.SPEED_KEY) or {}).get("running")))
            super().unload()

    service, engine = timed_service(tmp_path, engine=Watching(delay=0.02))
    assert chat(service) == "Echo: hi" and engine.backends == ["npu"]          # a model loaded before the timing
    selfcheck.npu_speed_check(service, Q40)
    assert freed == [("npu", None), ("npu", "npu"), ("gpu", "gpu"), ("cpu", "cpu"), ("npu+gpu:layer", "layer"),
                     ("npu+gpu:tensor", "tensor")]


def test_every_crashed_mode_stays_skipped_until_a_timing_has_reported_it(tmp_path):
    snapshots: dict = {}

    class Watching(eng.FakeEngine):
        def load(self, *args, **kwargs):
            if kwargs.get("split"):
                snapshots[kwargs["split"]] = dict(service.prefs.get(svc.SPEED_KEY) or {})
            super().load(*args, **kwargs)

    service, engine = timed_service(tmp_path, engine=Watching(delay=0.02))
    selfcheck.npu_speed_check(service, Q40)
    service.prefs.set(svc.SPEED_KEY, snapshots["layer"])                      # the app died in the layer split
    snapshots.clear()
    engine.backends.clear()
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert speeds["layer"]["skipped"] == selfcheck.SPEED_CRASHED_SKIPPED and "npu+gpu:layer" not in engine.backends
    died = snapshots["tensor"]                                                 # on disk while the tensor split ran
    assert died["running"] == "tensor" and died["crashed"] == ["layer"]
    service.prefs.set(svc.SPEED_KEY, died)                                     # ... and the app died in it too
    shown = selfcheck.saved_speeds(service)
    assert shown["layer"]["crashed"] == shown["tensor"]["crashed"] == selfcheck.SPEED_CRASHED
    engine.backends.clear()
    speeds = selfcheck.npu_speed_check(service, Q40)                          # neither runs: both are reported
    assert speeds["layer"]["skipped"] == speeds["tensor"]["skipped"] == selfcheck.SPEED_CRASHED_SKIPPED
    assert engine.backends == ["npu", "gpu", "cpu"] and service.prefs.get(svc.SPEED_KEY)["crashed"] == []
    engine.backends.clear()
    speeds = selfcheck.npu_speed_check(service, Q40)                          # asked again: both run
    assert speeds["layer"]["gen_tok_s"] > 0 and speeds["tensor"]["gen_tok_s"] > 0


def test_a_model_with_a_gpu_fallback_skips_the_combined_modes(tmp_path):
    service, engine = timed_service(tmp_path)
    service._gpu_failed(Q40, "the first token on the GPU failed")
    speeds = selfcheck.npu_speed_check(service, Q40)
    reason = f"GPU unavailable for {Q40}: the first token on the GPU failed"
    assert speeds["layer"] == {"skipped": reason} == speeds["tensor"]
    assert speeds["gpu"]["backend"] == "CPU" and not any(b.startswith("npu+gpu") for b in engine.backends)
    other, engine2 = timed_service(tmp_path / "b")
    other._gpu_failed("another-model", "no")                                   # another model's fallback: they run
    assert selfcheck.npu_speed_check(other, Q40)["layer"]["gen_tok_s"] > 0


def test_the_timing_warms_a_cpu_load_and_says_how_many_layers_a_partial_offload_ran(tmp_path):
    service, engine = timed_service(tmp_path)
    assert chat(service) == "Echo: hi" and engine.warms == []                  # a normal load is never warmed
    speeds = selfcheck.npu_speed_check(service, Q40)
    assert engine.warms == ["cpu"]                                             # the offload loads ran their check
    assert speeds["gpu"]["layers"] == eng.DEFAULT_GPU_LAYERS and speeds["layer"]["layers"] == -1
    assert speeds["cpu"]["layers"] == 0
    gpu = {**speeds["gpu"], "layer_limit": 33}
    assert selfcheck._speed_text(gpu, "gpu").endswith(" (8 of 33 layers on the GPU, the rest on the CPU)")
    assert not selfcheck._speed_text({**gpu, "layers": 33}, "gpu").endswith(")")
    assert selfcheck._speed_text({**gpu, "layer_limit": None}, "gpu").endswith(" (8 layers on the GPU)")
    assert not selfcheck._speed_text({**speeds["layer"], "layer_limit": 33}, "layer").endswith(")")
    assert selfcheck._speed_text({**speeds["npu"], "layers": 20, "layer_limit": 33}, "npu").endswith(
        " (20 of 33 layers on the NPU, the rest on the CPU)")


def test_llama_engine_warm_runs_the_first_token_check_once_on_a_cpu_load(fake_llama):
    engine = eng.LlamaEngine()
    engine.load("m.gguf", 2048, 4)
    model = _FakeLlama.made[-1]
    assert model.evals == []
    engine.warm()
    engine.warm()
    assert model.evals == [[1, 2, 3, 4], [4]]                                  # once, the GPU check's tokens
    engine.load("m.gguf", 2048, 4, gpu=True)
    gpu = _FakeLlama.made[-1]
    engine.warm()
    assert gpu.evals == [[1, 2, 3, 4], [4]]                                    # its load already checked it
    engine.unload()
    engine.warm()                                                              # nothing loaded: nothing to do


def test_self_check_services_share_the_app_markers(tmp_path):
    app_prefs = fakes.FakePrefs({"other": 1})
    prefs = selfcheck._DictPrefs({svc.SETTINGS_KEY: {"n_ctx": 2048}}, shared=app_prefs)
    for key in (svc.NPU_ATTEMPT_KEY, svc.NPU_INFLIGHT_KEY, svc.NPU_FALLBACK_KEY, svc.NPU_CHECK_KEY,
                svc.NPU_SESSION_KEY, svc.NPU_CHECK_ATTEMPT_KEY):
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
                                npu_precheck=lambda: {}, npu_skeleton_source=lambda: {}, device_index=lambda n: 0)
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


# ---- the two stages and the NPU files' first-use download (A1-A6) -------------------------------------------------

def test_engine_precheck_and_skeleton_source(monkeypatch):
    _ext(monkeypatch, npu_precheck=lambda: {"arch": "v81", "error": None,
                                            "steps": [{"name": n, "ok": True, "detail": ""} for n in eng.NPU_PRECHECK_STEPS]},
         npu_skeleton_source=lambda: SOURCE)
    state = eng.npu_precheck()
    assert state["arch"] == 81 and eng.npu_precheck_failure(state) is None
    assert eng.npu_skeleton_source() == SOURCE and eng.npu_skeleton_file(SOURCE, 81)["name"] == "libggml-htp-v81.so"
    assert eng.npu_skeleton_file(SOURCE, 69) is None
    _ext(monkeypatch, npu_skeleton_source=lambda: {**SOURCE, "url": URL.replace("https:", "http:")})
    assert eng.npu_skeleton_source() is None                                   # never a plain-http archive
    _ext(monkeypatch, npu_backend_shipped=lambda: True)                       # a wheel without npu_precheck
    assert eng.npu_precheck_failure(eng.npu_precheck()) == ("shipped", eng.NO_NPU)
    assert eng.npu_skeleton_source() is None
    assert eng.npu_precheck_failure({"arch": 81, "steps": [{"name": "shipped", "ok": True}]})[0] == "libcdsprpc"


def _skel_files(service) -> dict:
    import os

    folder = service.npu_skel_dir()
    return {name: open(os.path.join(folder, name), "rb").read() for name in sorted(os.listdir(folder))}


def test_the_first_npu_load_downloads_the_files_once(tmp_path):
    import os

    http = FakeHTTP(chunk_delay=0.01)
    service, engine, probe = npu_service(tmp_path, files=False, http=http)
    root = tmp_path / "htp"
    for old in ("9.16.0_abc_llama-0.3.37", "b9999"):                         # the test build's folders, an old release
        os.makedirs(root / old, exist_ok=True)
        (root / old / "libggml-htp-v81.so").write_bytes(b"old")
    service.run_npu_check()
    assert service.npu_offered() and not service.npu_files_ready() and http.calls == []  # stage 1 downloads nothing
    phases: list = []
    service.subscribe(lambda snap: phases.append((snap.state, snap.phase, (service.npu_progress() or {}).get("phase"))))
    assert chat(service) == "Echo: hi" and engine.backends == ["npu"]
    # one plain GET of the fixed URL, its redirect followed by hand (https only), a generic User-Agent, a short
    # connect timeout; no device or user data
    assert [c["url"] for c in http.calls] == [URL, ASSET]
    assert all(c["allow_redirects"] is False and c["timeout"] == svc.NPU_FETCH_TIMEOUT for c in http.calls)
    assert http.calls[0]["headers"] == {"Accept-Encoding": "identity", "User-Agent": "Glossarion-Mobile"}
    # only the five pinned members, each verified; the archive is gone; the other folders too
    assert _skel_files(service) == {f"libggml-htp-{arch}.so": data for arch, data in SKELETONS.items()}
    assert sorted(os.listdir(root)) == [RELEASE] and probe.calls[0][0] == service.npu_skel_dir()
    # the load showed the download, then the NPU's start, then the model (A3, B2)
    loading = [phase for state, phase, _p in phases if state == "loading" and phase]
    assert any(p.startswith("Getting NPU files · ") and p.endswith(" MB") for p in loading)
    assert svc.NPU_STARTING_TEXT in loading and loading.index(svc.NPU_STARTING_TEXT) > 0
    assert {p for _s, _ph, p in phases} >= {"downloading", "starting"} and service.npu_progress() is None
    assert service.npu_status_line() == "Hexagon v81 (HTP0) ready" and service.npu_files_ready()
    # an app update with the same engine keeps the files (keyed on the release, A5)
    newer, engine2, probe2 = npu_service(tmp_path, files=False, http=FakeHTTP(), version="9.16.1/llama-0.3.37")
    chat(newer)
    assert engine2.backends == ["npu"] and newer._npu_http.calls == [] and len(probe2.calls) == 1


def test_the_status_line_before_and_after_the_files(tmp_path):
    big = {**SOURCE, "size": 26027402}                                          # the real archive's size
    service, _engine, _probe = npu_service(tmp_path, files=False, source=big)
    service.run_npu_check()
    assert service.npu_status_line() == "NPU files (26 MB) download from llama.cpp's release at the first model load."
    place_skeleton(tmp_path / "htp")
    assert service.npu_files_ready(refresh=True) and service.npu_status_line() == "Hexagon v81 (HTP) ready"


def test_an_offline_phone_falls_through_and_tries_again_only_on_request(tmp_path):
    import requests

    http = FakeHTTP(error=requests.ConnectionError("no network"))
    service, engine, probe = npu_service(tmp_path, files=False, http=http)
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["cpu"] and len(http.calls) == 1 and probe.calls == []  # one try, no retry loop
    assert service.npu_fallback(Q40) is None and service.npu_offered()          # not a model fallback (A4)
    line = "NPU files not downloaded: Download failed (ConnectionError)"
    assert service.npu_status_line() == line and service.npu_files_note(Q40) == line
    assert f"Last download error: Download failed (ConnectionError)" in selfcheck.npu_lines(service, soc="")
    assert f"NPU files: llama.cpp {RELEASE} · libggml-htp-v81.so not downloaded" in selfcheck.npu_lines(service, soc="")
    chat(service, text="again")
    assert engine.backends == ["cpu"] and len(http.calls) == 1                 # no reload, no second try
    # the next start tries again
    later, engine2, _p = npu_service(tmp_path, files=False, http=FakeHTTP(), prefs=service.prefs)
    chat(later)
    assert engine2.backends == ["npu"]
    # "Try the NPU again" too
    service._npu_http = FakeHTTP()
    assert service.retry_npu(Q40) and service.npu_download_error is None
    chat(service)
    assert engine.backends == ["cpu", "npu"] and service._npu_http.calls == []  # the other start had fetched them


def test_a_stop_during_the_download_ends_the_load_and_the_next_load_resumes(tmp_path):
    import os

    from glossarion_mobile.services.local_llm.server import LocalLLMError

    http = FakeHTTP(chunk_delay=0.05)
    service, engine, probe = npu_service(tmp_path, files=False, http=http)
    service.run_npu_check()
    stopper = threading.Timer(0.12, service.stop_generation)
    stopper.start()
    with pytest.raises(LocalLLMError) as stopped:
        chat(service)
    stopper.join()
    # the Stop ended the load: nothing loaded on the GPU / CPU, and it is not a failed download
    assert stopped.value.status == 499 and engine.backends == [] and probe.calls == []
    assert service.npu_download_error is None and not service._npu_fetch_tried
    assert service.snapshot().state == "idle" and service.npu_progress() is None
    assert service.npu_status_line().startswith("NPU files (")              # "… download … at the first model load."
    part = os.path.join(service.npu_skel_dir(), "skeletons.tar.gz.part")
    assert 0 < os.path.getsize(part) < len(ARCHIVE)                           # kept for the next try
    # the next job's load resumes it, no "Try the NPU again" needed
    service._npu_http = http2 = FakeHTTP()
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["npu"] and http2.calls[-1]["headers"]["Range"].startswith("bytes=")
    assert not os.path.exists(part) and len(_skel_files(service)) == 5


def test_an_archive_left_by_a_dead_app_is_extracted_without_a_download(tmp_path):
    import os

    service, engine, _probe = npu_service(tmp_path, files=False, http=FakeHTTP(error=AssertionError("no download")))
    service.run_npu_check()
    folder = service.npu_skel_dir()
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "skeletons.tar.gz"), "wb") as handle:  # verified, renamed, never extracted
        handle.write(ARCHIVE)
    with open(os.path.join(folder, "libggml-htp-v73.so.tmp"), "wb") as handle:  # an extraction cut short
        handle.write(b"half")
    assert chat(service) == "Echo: hi" and engine.backends == ["npu"] and service._npu_http.calls == []
    assert sorted(os.listdir(folder)) == sorted(f"libggml-htp-{arch}.so" for arch in ARCHS)


def test_leftovers_next_to_ready_files_go_at_the_first_npu_load(tmp_path):
    import os

    service, engine, _probe = npu_service(tmp_path)                         # the phone's file is on disk
    service.run_npu_check()
    folder = service.npu_skel_dir()
    for name in ("skeletons.tar.gz", "skeletons.tar.gz.part", "skeletons.tar.gz.part.json",
                 "libggml-htp-v79.so.tmp"):
        with open(os.path.join(folder, name), "wb") as handle:
            handle.write(b"left over")
    assert chat(service) == "Echo: hi" and engine.backends == ["npu"]
    assert os.listdir(folder) == ["libggml-htp-v81.so"]


@pytest.mark.parametrize("case", ["archive checksum", "member checksum", "symlink member", "http redirect",
                                  "missing member", "oversized member"])
def test_a_bad_download_writes_nothing_and_falls_through(tmp_path, case):
    import hashlib
    import os

    http, source = FakeHTTP(), None
    if case == "archive checksum":
        source = {**SOURCE, "sha256": "0" * 64}
    elif case == "member checksum":
        source = make_source(ARCHIVE, skeletons={**SKELETONS, "v85": b"other bytes" * 50})
        source["files"]["v85"]["size"] = len(SKELETONS["v85"])
    elif case == "symlink member":
        # the phone's own skeleton is a link in this archive: never followed, nothing extracted
        archive = build_archive({a: d for a, d in SKELETONS.items() if a != "v81"},
                                links=(("llama-b12387/lib/libggml-htp-v81.so", "/etc/passwd"),))
        http, source = FakeHTTP(archive), make_source(archive)
    elif case == "http redirect":
        http = FakeHTTP(redirect="http://example.invalid/llama.tar.gz")
    elif case == "missing member":
        archive = build_archive({a: d for a, d in SKELETONS.items() if a != "v73"})
        http, source = FakeHTTP(archive), make_source(archive)
    else:
        archive = build_archive({**SKELETONS, "v79": SKELETONS["v79"] * 3})
        http, source = FakeHTTP(archive), make_source(archive)
        source["files"]["v79"].update(size=len(SKELETONS["v79"]),
                                      sha256=hashlib.sha256(SKELETONS["v79"]).hexdigest())
    service, engine, probe = npu_service(tmp_path, files=False, http=http, source=source)
    assert chat(service) == "Echo: hi"
    assert engine.backends == ["cpu"] and probe.calls == [] and service.npu_fallback(Q40) is None
    folder = service.npu_skel_dir()
    assert not [n for n in os.listdir(folder) if n.endswith(".so") or n.endswith(".tmp")]  # nothing kept
    assert not os.path.exists(os.path.join(folder, "skeletons.tar.gz"))
    assert service.npu_status_line().startswith("NPU files not downloaded: ")
    if case == "http redirect":
        assert service.npu_download_error == "the download was redirected to an address that is not https"
        assert [c["url"] for c in http.calls] == [URL]
    if case in ("archive checksum", "member checksum"):
        assert "checksum mismatch" in service.npu_download_error


def test_the_start_failing_after_a_download_keeps_the_npu_off(tmp_path):
    service, engine, probe = npu_service(tmp_path, files=False, gpu=True,
                                         probe=Probe(failing_state("session", "unsigned PD denied")))
    chat(service)
    assert len(probe.calls) == 1 and engine.backends == ["gpu"] and not service.npu_offered()
    assert service.npu_failure() == ("session", "unsigned PD denied") and service.npu_fallback(Q40) is None
    assert selfcheck.npu_lines(service, soc="")[0] == "NPU: not available (session: unsigned PD denied)"
    chat(service, text="again")
    assert engine.backends == ["gpu"] and len(probe.calls) == 1                # stage 2 at most once


def test_register_without_the_files_registers_nothing_and_is_not_a_failure(tmp_path):
    # the INTERFACE's "NPU files not downloaded" (a file that went away after the check): this load is not on the
    # NPU, the NPU stays offered, nothing is saved as a failed start
    service, engine, probe = npu_service(tmp_path, files=True, http=FakeHTTP(error=OSError("offline")))
    import os

    service.run_npu_check()
    original = probe.__call__

    def vanished(skel_dir, options):
        os.remove(os.path.join(skel_dir, "libggml-htp-v81.so"))
        return original(skel_dir, options)

    service._npu_probe = vanished
    chat(service)
    assert engine.backends == ["cpu"] and service.npu_offered() and service.prefs.get(svc.NPU_SESSION_KEY) is None
    assert service.npu_download_error == eng.NPU_FILES_MISSING
