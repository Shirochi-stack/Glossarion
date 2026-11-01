"""What a QA scan does to the Flet event loop, measured on the host (owner 2026-11-01: "the mobile GUI was
lagging a lot during the QA scanning").

Each run is one ``qa_scan`` job through the real ``JobService`` on a synthetic book, in its own process, with
the phone's gates (``GLOSSARION_MOBILE`` / ``GLOSSARION_NO_PROCESSES``: thread pools in the interpreter that runs
the loop). The loop runs the real ``UiDispatcher`` pump and a real ``LogConsole`` on the fake Flet session of
``tests_host/test_bootstrap.py``, so every repaint is diffed and encoded. Reported per run:

* ``late``: how late ``await asyncio.sleep(0.016)`` wakes, p50 / p95 / max ms (the loop waiting for the GIL);
* ``tap``: how long a fixed piece of pure Python (``--tap`` ms when idle; a phone's tap handler or diff is tens of
  ms) takes on the loop, p50 / p95 ms next to its idle time, while the workers hold the GIL;
* ``total``: the scan's time; ``repaint``: the loop's time in the pump (log lines, repaints), with the line count;
* ``findings``: a hash of the report's files, issues and scores, to compare two runs of the same book.

Every path (Library, output, config, home) is a temp folder. Run from src/mobile with the mobile venv, with
bytecode on so the runs after the first do not measure the compiler (the phone's bundle is compiled):

    PYTHONPYCACHEPREFIX=<scratch> .venv/Scripts/python tools/qa_scan_bench.py --workers 1 2 4 --mode quick-scan --sample 1000
    ... --src <another checkout's src>   to measure an older tree
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]

_COMMON = ("the knight walked into hall and greeted everyone warmly while rain fell over old castle walls she said "
           "nothing but her eyes followed sword across table where merchants argued about silver dragon mountain "
           "village river forest shadow captain soldier whisper letter promise ancient magic emperor palace servant "
           "morning evening horse road market blood honor secret").split()
_SYLLABLES = "ka ri mo ten sa lu vi dor an el ith ber gal ron mi su ta ne ko ha yu wen fa dra li osh".split()


def _paragraph(rng: random.Random, words: int, topic: list) -> str:
    sentences, sentence = [], []
    for _ in range(words):
        sentence.append(rng.choice(topic) if rng.random() < 0.45 else rng.choice(_COMMON))
        if len(sentence) >= rng.randint(8, 18):
            text = " ".join(sentence)
            sentences.append(text[0].upper() + text[1:] + ".")
            sentence = []
    return " ".join(sentences) or "Silence."


def make_book(folder: Any, chapters: int = 300, words: int = 3500, seed: int = 9151) -> str:
    """A translated book: ``response_*.html`` chapters of their own vocabulary, a translation_progress.json and
    planted findings (a duplicate every 40th, untranslated text every 41st, no <html> every 43rd, a tag without
    its bracket every 53rd)."""
    folder = os.fspath(folder)
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(seed)
    vocabulary = sorted({"".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4))) for _ in range(6000)})
    progress, last = {}, ""
    for index in range(1, chapters + 1):
        topic = rng.sample(vocabulary, 300)
        paragraphs, left = [], words
        while left > 0:
            count = min(left, rng.randint(40, 120))
            paragraphs.append(f"<p>{_paragraph(rng, count, topic)}</p>")
            left -= count
        text = (f"<html><head><title>Chapter {index}</title></head><body><h1>Chapter {index}</h1>"
                + "".join(paragraphs) + "</body></html>")
        if index % 40 == 0:
            text = last.replace(f"Chapter {index - 1}", f"Chapter {index}")
        elif index % 41 == 0:
            text = text.replace("</body>", "<p>" + "기사는 홀로 걸어 들어가 모두에게 인사했다. " * 8 + "</p></body>")
        elif index % 43 == 0:
            text = text.replace("<html>", "").replace("</html>", "")
        elif index % 53 == 0:
            text = text.replace("</p><p>", "</p<p>", 1)
        name = f"response_{index:04d}_ch{index:04d}.html"
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        last = text
        progress[str(index)] = {"status": "completed", "output_file": name, "actual_num": index}
    with open(os.path.join(folder, "translation_progress.json"), "w", encoding="utf-8") as handle:
        json.dump({"chapters": progress}, handle)
    return folder


def report_findings(folder: Any) -> list:
    """``[[file, sorted issues, score, duplicate confidence]]`` of the folder's scan report (paths left out)."""
    folder = os.fspath(folder)
    path = os.path.join(folder, os.path.basename(folder.rstrip("/\\")) + "_Scan Report", "validation_results.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return sorted([row.get("filename"), sorted(str(i) for i in row.get("issues") or ()), row.get("score"),
                   row.get("duplicate_confidence")] for row in rows)


def percentile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))] if ordered else 0.0


def _spin(units: int) -> None:
    for _ in range(units):
        total = 0
        for i in range(1000):
            total += i * i % 7


def _calibrate(target: float) -> int:
    units = 1
    while True:
        started = time.perf_counter()
        _spin(units)
        took = time.perf_counter() - started
        if took >= target / 2:
            return max(1, int(units * target / took))
        units *= 2


async def _measure(args: argparse.Namespace, folder: str, scratch: str, config: dict) -> dict:
    import importlib.util

    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.services.jobs import JobService, JobSpec
    from glossarion_mobile.ui.components.log_console import LogConsole
    import qa_scan_runtime

    if hasattr(qa_scan_runtime, "MOBILE_QA_MAX_WORKERS"):  # a tree from before the setting
        qa_scan_runtime.MOBILE_QA_MAX_WORKERS = args.workers[0]
    spec = importlib.util.spec_from_file_location("_bench_tb", Path(args.src) / "mobile" / "tests_host" / "test_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    _conn, session = bootstrap._fake_session("android")
    page = session.page

    class Store:
        def flush(self) -> bool:
            return False

        def snapshot(self) -> dict:
            return json.loads(json.dumps(config))

        def set_job_running(self, running: bool) -> None:
            pass

        def get(self, key: str, default=None):
            return config.get(key, default)

    dispatcher = UiDispatcher(page)
    dispatcher.bind()
    repaint = [0.0, 0]
    real_flush = dispatcher.flush

    def timed_flush() -> int:
        started = time.perf_counter()
        try:
            return real_flush()
        finally:
            repaint[0] += time.perf_counter() - started
            repaint[1] += 1

    dispatcher.flush = timed_flush
    dispatcher.start()
    service = JobService(jobs_dir=os.path.join(scratch, "jobs"), logs_dir=os.path.join(scratch, "logs"),
                         data_dir=os.path.join(scratch, "data"), config_store=Store(), dispatcher=dispatcher)
    units = _calibrate(args.tap / 1000)
    idle = []
    for _ in range(20):
        started = time.perf_counter()
        _spin(units)
        idle.append((time.perf_counter() - started) * 1000)
    late, tap = [], []
    done = asyncio.Event()

    async def sleeper() -> None:
        while not done.is_set():
            started = time.perf_counter()
            await asyncio.sleep(0.016)
            late.append((time.perf_counter() - started - 0.016) * 1000)

    async def tapper() -> None:
        while not done.is_set():
            await asyncio.sleep(0.15)
            started = time.perf_counter()
            _spin(units)
            tap.append((time.perf_counter() - started) * 1000)

    if args.switch:
        sys.setswitchinterval(args.switch)
    began = time.perf_counter()
    job_id = service.submit(JobSpec(kind="qa_scan", title=os.path.basename(folder), inputs=(folder,),
                                    params={"mode": args.mode, "targets": [{"folder": folder, "source": None}],
                                            "disable_word_count": True}))
    probes = [asyncio.ensure_future(sleeper()), asyncio.ensure_future(tapper())]
    console = None
    while True:
        snap = service.snapshot(job_id)
        if console is None and service.log_buffer(job_id) is not None:
            console = LogConsole(buffer=service.log_buffer(job_id), dispatcher=dispatcher, backlog=2000)
            page.controls.append(console)
            page.update()
        if snap is not None and snap.is_terminal:
            break
        await asyncio.sleep(0.05)
    total = time.perf_counter() - began
    done.set()
    for probe in probes:
        await probe
    buffer = service.log_buffer(job_id)
    findings = report_findings(folder)
    await dispatcher.stop()
    return {"workers": args.workers[0], "mode": args.mode, "sample": args.sample, "state": str(snap.state),
            "total_s": round(total, 2), "late_p50": round(percentile(late, .5), 1), "late_p95": round(percentile(late, .95), 1),
            "late_max": round(max(late or [0]), 1), "tap_idle": round(statistics.median(idle), 1),
            "tap_p50": round(percentile(tap, .5), 1), "tap_p95": round(percentile(tap, .95), 1),
            "repaint_s": round(repaint[0], 3), "repaints": repaint[1], "lines": buffer.last_seq if buffer else 0,
            "files": len(findings), "with_issues": sum(1 for row in findings if row[1]),
            "findings": hashlib.sha256(json.dumps(findings, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]}


def _run_one(args: argparse.Namespace) -> None:
    scratch = tempfile.mkdtemp(prefix="run", dir=os.path.dirname(args.book))
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "local"),
                      ("GLOSSARION_LIBRARY_DIR", "library"), ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        os.makedirs(os.path.join(scratch, sub), exist_ok=True)
        os.environ[name] = os.path.join(scratch, sub)
    os.environ.update({"GLOSSARION_MOBILE": "1", "GLOSSARION_NO_PROCESSES": "1", "GLOSSARION_HTTP_LOG": "0",
                       "CONFIG_FILE": os.path.join(scratch, "data", "config.json"),
                       "AI_HUNTER_MAX_WORKERS": str(args.workers[0])})
    config = {"output_language": "English", "ai_hunter_config": {"ai_hunter_max_workers": args.workers[0]},
              "qa_scanner_settings": {"quick_scan_sample_size": args.sample}}
    with open(os.environ["CONFIG_FILE"], "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    # the phone has none of these; the host's user site would add their import costs to the run
    for hidden in ("PySide6", "shiboken6", "torch", "ultralytics", "transformers", "sentence_transformers"):
        sys.modules[hidden] = None
    sys.path[:0] = [args.src, os.path.join(args.src, "mobile", "app")]
    if sys.platform == "win32":  # this process only: 15.6 ms timer ticks would hide everything below them
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    folder = os.path.join(scratch, "out", "SyntheticBook")
    shutil.copytree(args.book, folder)
    print("RESULT " + json.dumps(asyncio.run(_measure(args, folder, scratch, config))), flush=True)
    os._exit(0)  # the scanner's daemon threads and temp files do not matter any more


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--mode", default="quick-scan")
    parser.add_argument("--sample", type=int, default=1000, help="quick_scan_sample_size (0 = duplicate check off)")
    parser.add_argument("--chapters", type=int, default=300)
    parser.add_argument("--words", type=int, default=3500)
    parser.add_argument("--src", default=str(SRC_DIR), help="the src folder to measure")
    parser.add_argument("--switch", type=float, default=0.0, help="sys.setswitchinterval before the job (explore)")
    parser.add_argument("--tap", type=float, default=40.0, help="the loop probe's work, ms when idle")
    parser.add_argument("--book", default="", help=argparse.SUPPRESS)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        _run_one(args)
        return 0
    with tempfile.TemporaryDirectory(prefix="glqabench", ignore_cleanup_errors=True) as root:
        book = make_book(os.path.join(root, "book"), args.chapters, args.words)
        print(f"{'workers':>7} {'total s':>8} {'late p50/p95/max ms':>22} {'tap idle p50/p95 ms':>22} "
              f"{'repaint s':>10} {'lines':>6} {'issues':>7} findings")
        for workers in args.workers:
            command = [sys.executable, str(HERE), "--one", "--book", book, "--workers", str(workers), "--mode", args.mode,
                       "--sample", str(args.sample), "--src", args.src, "--switch", str(args.switch), "--tap", str(args.tap)]
            done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            found = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
            if not found:
                print(f"{workers:>7} failed: {done.stdout[-400:]} {done.stderr[-400:]}")
                continue
            row = json.loads(found[-1][7:])
            print(f"{workers:>7} {row['total_s']:>8.2f} {row['late_p50']:>8.1f}/{row['late_p95']:>6.1f}/{row['late_max']:>6.1f} "
                  f"{row['tap_idle']:>8.1f}/{row['tap_p50']:>6.1f}/{row['tap_p95']:>6.1f} {row['repaint_s']:>10.3f} "
                  f"{row['lines']:>6} {row['with_issues']:>7} {row['findings']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
