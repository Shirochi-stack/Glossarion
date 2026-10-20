#!/usr/bin/env python3
"""The engine's optional OpenCL backend loads only on demand and falls back cleanly (mobile-wheels.yml,
job ``arm64-engine-check`` with ``gpu_check``).

The Android arm64 wheel ships libggml-opencl.so (GGML_OPENCL, Adreno kernels) next to the CPU variants;
it alone needs the phone's libOpenCL.so, and the patched loader registers it only when the app's
"Use GPU" setting is on (recipes/llama-cpp-python, mobile.patch 8.). The Android libraries need bionic,
so the job builds the same sdist + patch with the same flags on Linux arm64 and runs this script there,
once per OpenCL situation a phone can be in:

* ``--expect no-driver``: no libOpenCL at all (the ICD loader package removed after the build).
  Importing llama_cpp and generating on the CPU must work; ``register_gpu_backend()`` must report
  "no OpenCL driver (...)", and the app's ``LlamaEngine.load(gpu=True)`` must raise ``GpuUnavailable``.
* ``--expect no-adreno``: an OpenCL driver with no Adreno GPU (pocl's CPU device, like a Mali / Xclipse
  phone): the backend loads, drops the device, and reports "no supported GPU".

Both also check statically that libggml-opencl is the only library that needs libOpenCL and that no
library needs libggml-opencl, then load the model on the CPU through the app's engine after the failed
GPU attempt (the service's fallback path) and generate a few tokens.

  gpu_backend_check.py --expect no-driver|no-adreno --model M [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOBILE = HERE.parent.parent
EXPECT = {"no-driver": "no OpenCL driver", "no-adreno": "no supported GPU"}


def static_check(lib_dir: Path) -> dict:
    sys.path.insert(0, str(HERE))
    import wheels  # noqa: E402  (parse_elf)

    needs = {}
    for path in sorted(lib_dir.glob("*.so")):
        needs[path.name] = wheels.parse_elf(path.read_bytes())["needed"]
    opencl = {name for name, needed in needs.items() if any(n.startswith("libOpenCL.so") for n in needed)}
    if opencl != {"libggml-opencl.so"}:
        raise AssertionError(f"libraries needing libOpenCL: {sorted(opencl)} (expected only libggml-opencl.so)")
    users = sorted(name for name, needed in needs.items() if "libggml-opencl.so" in needed)
    if users:
        raise AssertionError(f"{users} need libggml-opencl.so: it must only be loaded on demand")
    return {"libggml-opencl.so": needs["libggml-opencl.so"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--expect", required=True, choices=sorted(EXPECT))
    ap.add_argument("--model", required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    report: dict = {"expect": args.expect}
    ok = False
    try:
        started = time.perf_counter()
        import llama_cpp
        from llama_cpp import _ctypes_extensions as ext

        report["import_s"] = round(time.perf_counter() - started, 2)
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        report["static"] = static_check(lib_dir)
        report["cpu_variant"] = ext.CPU_BACKEND_VARIANT
        if not str(ext.CPU_BACKEND_VARIANT or "").startswith("android_"):
            raise AssertionError(f"no CPU variant registered: {ext.CPU_BACKEND_VARIANT}")
        if ext.gpu_backend_name() != "OpenCL":
            raise AssertionError(f"gpu_backend_name() = {ext.gpu_backend_name()!r}, expected 'OpenCL'")
        if ext.GPU_STATE:
            raise AssertionError(f"a GPU backend was registered before anyone asked: {ext.GPU_STATE}")

        sys.path.insert(0, str(MOBILE / "app"))
        from glossarion_mobile.services.local_llm import engine as eng

        engine = eng.LlamaEngine()
        try:
            engine.load(args.model, 512, 2, gpu=True)
            raise AssertionError(f"the GPU load succeeded without an Adreno GPU (device {engine.device})")
        except eng.GpuUnavailable as exc:
            report["gpu_error"] = str(exc)
        state = dict(ext.GPU_STATE)
        report["gpu_state"] = state
        if state.get("device") or EXPECT[args.expect] not in str(state.get("error") or ""):
            raise AssertionError(f"register_gpu_backend() = {state}, expected the error {EXPECT[args.expect]!r}")
        if engine.loaded:
            raise AssertionError("the engine kept a model after GpuUnavailable")
        # the service's fallback: the same model on the CPU
        started = time.perf_counter()
        engine.load(args.model, 512, 2)
        report["cpu_load_s"] = round(time.perf_counter() - started, 2)
        report["device"] = engine.device
        if engine.on_gpu or not str(engine.device or "").startswith("CPU"):
            raise AssertionError(f"the CPU load reports device {engine.device!r}")
        text = "".join(engine.chat([{"role": "user", "content": "Say hello in English."}],
                                   {"max_tokens": 8, "temperature": 0.0}, lambda: False))
        report["text"] = text[:80]
        report["tokens"] = engine.last_usage.get("completion_tokens")
        if not text.strip():
            raise AssertionError("the CPU fallback generated nothing")
        engine.unload()
        ok = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["ok"] = ok
    print("GPU_BACKEND_CHECK " + json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
