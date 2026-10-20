#!/usr/bin/env python3
"""Runtime CPU-variant check of the on-device engine on Linux arm64 (mobile-wheels.yml, job
``arm64-cpu-variants``, dispatched with ``cpu_check``).

The Android arm64 wheel registers one of four CPU backends at import (recipes/llama-cpp-python,
mobile.patch 7.), but its libraries need Android's bionic and cannot run on a CI runner. The job
therefore builds the same sdist with the same patch on Linux arm64 with the same four variants
(``-DGLOSSARION_ANDROID_CPU_VARIANTS=ON``) and drives the same patched loader:

* natively (GitHub's arm64 runner: Neoverse N2, which has dotprod, fp16 and i8mm like a recent phone
  core): the best variant registers, the model generates, and tokens/s per variant
  (``LLAMA_CPP_CPU_VARIANT`` forces one) show what dotprod / i8mm buy;
* under ``qemu-aarch64 -cpu <model>`` (user mode enforces the model's ISA): a Cortex-A53 (armv8.0,
  the oldest arm64 phones) must pick the baseline and generate - probing the dotprod / i8mm
  variants must not SIGILL - and a Cortex-A76 must pick dotprod + fp16. A tiny program with one SDOT
  first proves that qemu really raises SIGILL for a missing feature.

  cpu_variants_check.py probe [--model M] [--tokens N] [--prompt-tokens P] [--threads T]
  cpu_variants_check.py run --model M [--bench-model M2 ...] --json report.json [--summary summary.md]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARKER = "CPU_VARIANT_PROBE"
VARIANTS = ("android_armv8.6_1", "android_armv8.2_2", "android_armv8.2_1", "android_armv8.0_1")
# qemu CPU model -> the variant the loader must register on it
QEMU_EXPECT = {"cortex-a53": "android_armv8.0_1", "cortex-a76": "android_armv8.2_2", "max": "android_armv8.6_1"}
PROMPT = ("Translate the following paragraph into English and keep the names unchanged. "
          "The old lighthouse keeper climbed the spiral stairs every evening, counting each step, "
          "and when the lamp was lit he wrote the date, the weather and the ships he saw into a book "
          "that nobody had read for forty years. ")
SDOT_C = r"""
#include <stdio.h>
int main(void) {
    __asm__ volatile("sdot v0.4s, v1.16b, v2.16b" ::: "v0");
    puts("sdot ran");
    return 0;
}
"""


def probe(args) -> int:
    t0 = time.perf_counter()
    import llama_cpp
    from llama_cpp import _ctypes_extensions as ext

    info = llama_cpp.llama_print_system_info()
    out = {"variant": getattr(ext, "CPU_BACKEND_VARIANT", None),
           "system_info": (info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)).strip(),
           "import_s": round(time.perf_counter() - t0, 2), "machine": platform.machine()}
    if args.model:
        llm = llama_cpp.Llama(model_path=args.model, n_ctx=1024, n_batch=512, n_threads=args.threads,
                              verbose=False)
        tokens = llm.tokenize(PROMPT.encode("utf-8"), add_bos=True)
        while len(tokens) < args.prompt_tokens:
            tokens += llm.tokenize(PROMPT.encode("utf-8"), add_bos=False)
        tokens = tokens[:args.prompt_tokens]
        llm.reset()
        t = time.perf_counter()
        llm.eval(tokens)
        pp = time.perf_counter() - t
        generated = []
        t = time.perf_counter()
        for _ in range(args.tokens):
            tok = llm.sample(temp=0.0)
            generated.append(tok)
            llm.eval([tok])
        tg = time.perf_counter() - t
        out.update({"model": Path(args.model).name, "threads": args.threads, "prompt_tokens": len(tokens),
                    "pp_tps": round(len(tokens) / pp, 2), "gen_tokens": len(generated),
                    "tg_tps": round(len(generated) / tg, 2),
                    "text": llm.detokenize(generated).decode("utf-8", "replace")})
    print(MARKER, json.dumps(out), flush=True)
    return 0


def _run_probe(prefix: list, env_extra: dict, model: str | None, tokens: int, prompt_tokens: int, threads: int,
               timeout: int) -> dict:
    cmd = prefix + [sys.executable, str(Path(__file__).resolve()), "probe", "--tokens", str(tokens),
                    "--prompt-tokens", str(prompt_tokens), "--threads", str(threads)]
    if model:
        cmd += ["--model", model]
    env = {**os.environ, **env_extra}
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    result = {"cmd": " ".join(prefix) or "native", "env": env_extra, "rc": proc.returncode,
              "wall_s": round(time.perf_counter() - t, 1)}
    if proc.returncode < 0:
        result["signal"] = signal.Signals(-proc.returncode).name
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER + " "):
            result.update(json.loads(line[len(MARKER) + 1:]))
    if proc.returncode != 0 or "variant" not in result:
        result["stderr"] = proc.stderr[-3000:]
    return result


def _sigill_control(qemu: str) -> dict:
    """One SDOT under qemu: native (has dotprod) runs it, a Cortex-A53 must die of SIGILL."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return {"error": "no C compiler"}
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "sdot.c", Path(tmp) / "sdot"
        src.write_text(SDOT_C, encoding="utf-8")
        subprocess.run([cc, "-O1", "-march=armv8.2-a+dotprod", str(src), "-o", str(exe)], check=True)
        out = {}
        for label, cmd in (("native", [str(exe)]), ("cortex-a53", [qemu, "-cpu", "cortex-a53", str(exe)]),
                           ("cortex-a76", [qemu, "-cpu", "cortex-a76", str(exe)])):
            rc = subprocess.run(cmd, capture_output=True, text=True).returncode
            out[label] = signal.Signals(-rc).name if rc < 0 else rc
    return out


def run(args) -> int:
    report = {"machine": platform.machine(), "cpu_features": _cpu_features(), "native": [], "qemu": [],
              "errors": []}
    errors = report["errors"]
    # 1. native: automatic choice, then each variant forced (the loader still refuses unsupported ones)
    auto = _run_probe([], {}, args.model, args.tokens, args.prompt_tokens, args.threads, 1800)
    report["native"].append(auto)
    if auto.get("rc") != 0 or not auto.get("gen_tokens"):
        errors.append(f"native: the engine did not load and generate: {auto}")
    expected = expected_variant(report["cpu_features"])
    report["native_expected"] = expected
    if auto.get("variant") != expected:
        errors.append(f"native: registered {auto.get('variant')!r}, but this CPU's features "
                      f"({report['cpu_features']}) select {expected!r}")
    for model in args.bench_model or [args.model]:
        for variant in VARIANTS:
            r = _run_probe([], {"LLAMA_CPP_CPU_VARIANT": variant}, model, args.tokens, args.prompt_tokens,
                           args.threads, 1800)
            r["forced"] = variant
            report["native"].append(r)
            if r.get("rc") != 0:
                errors.append(f"native, {variant} forced: rc {r.get('rc')} {r.get('signal', '')}")
    # 2. qemu: older CPUs must get an older variant and never SIGILL
    qemu = shutil.which("qemu-aarch64") or shutil.which("qemu-aarch64-static")
    if not qemu:
        errors.append("qemu-aarch64 is not installed")
    else:
        control = _sigill_control(qemu)
        report["sigill_control"] = control
        if control.get("cortex-a53") != "SIGILL" or control.get("cortex-a76") != 0:
            errors.append(f"qemu does not enforce the CPU model's features (SDOT control: {control})")
        for model_name, expect in QEMU_EXPECT.items():
            r = _run_probe([qemu, "-cpu", model_name], {}, args.qemu_model or args.model, args.qemu_tokens,
                           args.qemu_prompt_tokens, args.threads, 3600)
            r["cpu_model"], r["expected"] = model_name, expect
            report["qemu"].append(r)
            if r.get("rc") != 0:
                errors.append(f"qemu -cpu {model_name}: rc {r.get('rc')} {r.get('signal', '')}: "
                              f"{r.get('stderr', '')[-500:]}")
            elif r.get("variant") != expect:
                errors.append(f"qemu -cpu {model_name}: registered {r.get('variant')!r}, expected {expect!r}")
            elif not r.get("gen_tokens"):
                errors.append(f"qemu -cpu {model_name}: generated nothing")
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary:
        Path(args.summary).write_text(_summary(report), encoding="utf-8")
    print(_summary(report))
    for e in errors:
        print(f"ERROR   {e}")
    print(f"cpu-variants check: {len(errors)} errors")
    return 1 if errors else 0


def expected_variant(features: str) -> str:
    """The variant ggml_backend_score() must pick for /proc/cpuinfo ``features`` (HWCAP names)."""
    have = set(features.split())
    dotprod, fp16, i8mm = "asimddp" in have, "fphp" in have, "i8mm" in have   # ggml reads HWCAP_FPHP
    if dotprod and fp16 and i8mm:
        return "android_armv8.6_1"
    if dotprod and fp16:
        return "android_armv8.2_2"
    return "android_armv8.2_1" if dotprod else "android_armv8.0_1"


def _cpu_features() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("features"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _summary(report: dict) -> str:
    lines = ["## On-device engine: runtime CPU variants (Linux arm64)", "",
             f"Runner CPU features: `{report['cpu_features']}`", ""]
    lines += ["| run | variant | prompt tok/s | gen tok/s | model | rc |", "|---|---|---|---|---|---|"]
    for r in report["native"]:
        label = f"native, forced {r['forced']}" if r.get("forced") else "native, automatic"
        lines.append(f"| {label} | {r.get('variant')} | {r.get('pp_tps', '-')} | {r.get('tg_tps', '-')} | "
                     f"{r.get('model', '-')} | {r.get('rc')} {r.get('signal', '')} |")
    for r in report["qemu"]:
        lines.append(f"| qemu -cpu {r['cpu_model']} (expect {r['expected']}) | {r.get('variant')} | "
                     f"{r.get('pp_tps', '-')} | {r.get('tg_tps', '-')} | {r.get('model', '-')} | "
                     f"{r.get('rc')} {r.get('signal', '')} |")
    if "sigill_control" in report:
        lines += ["", f"SDOT control (qemu must kill the Cortex-A53 run): `{report['sigill_control']}`"]
    auto = next((r for r in report["native"] if not r.get("forced")), {})
    if auto.get("system_info"):
        lines += ["", f"System info (automatic): `{auto['system_info']}`"]
    if report["errors"]:
        lines += ["", "**Errors**", ""] + [f"- {e}" for e in report["errors"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="import llama_cpp, report the registered variant, optionally generate")
    pr.add_argument("--model")
    pr.add_argument("--tokens", type=int, default=32)
    pr.add_argument("--prompt-tokens", type=int, default=128)
    pr.add_argument("--threads", type=int, default=4)
    rn = sub.add_parser("run", help="native + qemu checks and the per-variant benchmark")
    rn.add_argument("--model", required=True, help="GGUF for the automatic run and the qemu runs")
    rn.add_argument("--bench-model", action="append", help="GGUF(s) to benchmark per variant (default --model)")
    rn.add_argument("--qemu-model", help="GGUF for the qemu runs (default --model)")
    rn.add_argument("--tokens", type=int, default=64)
    rn.add_argument("--prompt-tokens", type=int, default=256)
    rn.add_argument("--qemu-tokens", type=int, default=4)
    rn.add_argument("--qemu-prompt-tokens", type=int, default=16)
    rn.add_argument("--threads", type=int, default=4)
    rn.add_argument("--json", required=True)
    rn.add_argument("--summary")
    args = p.parse_args(argv)
    return probe(args) if args.cmd == "probe" else run(args)


if __name__ == "__main__":
    sys.exit(main())
