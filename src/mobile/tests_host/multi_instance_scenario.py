"""Multi-instance end to end (design §6): two books translated at once, book A in instance 1, book B in instance 2.

The real JobService, HeadlessOwner and shared pipeline, offline: ``diagnostics.e2e.E2ESession`` gives the sandbox, the
fake OpenAI server, the config store and the process / network / write guards. The Translation key pool holds one
key per instance (``KEY_ONE`` for instance 1, ``KEY_TWO`` for instance 2), the glossary is Balanced and the Unified
Glossary is on, so both books' glossaries are merged into the one shared file while both run. Book B (5 chapters,
names of its own) is stopped after its second chapter and resumed on its instance. What is checked:

* the two jobs ran at the same time (their requests overlap) on their own instance;
* every request of a book carried its instance's key only (the fake server reads the Authorization header);
* each instance saw its own environment, argv (``TransateKRtoEN`` reads the book from ``sys.argv[1]``), key pool
  and stdout (each job's log names its own book only);
* stopping book B left book A running to the end; outputs and progress files are separate;
* the unified glossary holds both books' entries (nothing lost to the concurrent read -> merge -> write);
* no process was spawned, nothing was written outside the sandbox, the process state was restored.

``run()`` returns the result as a dict (``ok``, ``checks``, ``detail``). Host: ``tests_host/test_multi_instance_e2e.py``
runs ``main()`` in a subprocess (bootstrap into ``FLET_APP_STORAGE_*`` dirs); device: a probe calls ``run()`` after the
app's own bootstrap. Python 3.10 compatible; no pytest.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if APP_DIR.is_dir() and str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

KEY_ONE = "sk-glossarion-instance-one"
KEY_TWO = "sk-glossarion-instance-two"
BOOK_B_CHAPTERS = 5
STOP_AFTER = 2  # book B's chapter answers before its Stop
#: Book B's recurring names: none of them is in the self-test EPUB (book A)
BOOK_B_NAMES = (
    ("character", "윤하늘", "Ha-neul Yoon", "Female", "Lantern keeper of the valley"),
    ("character", "정우진", "Woo-jin Jeong", "Male", "Ferryman of the black river"),
    ("term", "흑룡 계곡", "Black Dragon Valley", "", "Valley east of the river"),
    ("term", "청동 등불", "Bronze Lantern", "", "A lantern that never goes out"),
)


def build_book_b(path: Path, chapters: int = BOOK_B_CHAPTERS, title_format: str = "제{index}화") -> Path:
    """A small Korean EPUB whose chapters ("제N화" headings, like the self-test book) name only ``BOOK_B_NAMES``.
    ``title_format``: the chapters' table-of-contents titles (the nav page is translated as a chapter too)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("glossarion-multi-instance-book-b")
    book.set_title("흑룡 계곡의 등불")
    book.set_language("ko")
    book.add_author("Glossarion")
    items = []
    for index in range(1, chapters + 1):
        chapter = epub.EpubHtml(title=title_format.format(index=index), file_name=f"text/chapter{index:04d}.xhtml",
                                lang="ko")
        chapter.content = (
            f"<html><head><title>제{index}화</title></head><body><h1>제{index}화 흑룡 계곡의 밤</h1>"
            f"<p>윤하늘은 청동 등불을 들고 흑룡 계곡으로 내려갔다. 정우진이 그 뒤를 따랐다.</p>"
            f"<p>정우진은 윤하늘에게 말했다. 흑룡 계곡의 바람은 차갑고, 청동 등불은 꺼지지 않았다.</p>"
            f"<p>{index}번째 밤이 지나자 윤하늘과 정우진은 다시 길을 떠났다.</p></body></html>"
        )
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


class Recorder:
    """Every chat request the fake server answers: which book it is for (its last user message names book A's or
    book B's characters), the key it carried, when it ran; plus each instance's state while book B's requests run.
    Book A's chapter requests wait (``release_a``, at most ``PARK_SECONDS``) until book B was stopped, so book A is
    mid-run, a request in flight, while book B works and when it stops."""

    PARK_SECONDS = 180.0

    def __init__(self, server: Any, names_a: tuple, names_b: tuple) -> None:
        self.server = server
        self.names_a, self.names_b = names_a, names_b
        self.rows: list = []
        self.samples: list = []
        self.lock = threading.Lock()
        self.on_b_answer: Optional[Callable[[int], Any]] = None
        self.release_a = threading.Event()
        self.a_parked = threading.Event()  # a chapter request of book A is in flight (waiting for release_a)
        self._b_answers = 0
        self._serve = server._serve_chat
        server._serve_chat = self._serve_chat

    def _book_of(self, payload: dict) -> str:
        from glossarion_mobile.diagnostics.fake_llm_server import _content_text

        messages = [m for m in payload.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
        text = _content_text(messages[-1].get("content")) if messages else ""
        in_a = any(name in text for name in self.names_a)
        in_b = any(name in text for name in self.names_b)
        return "A" if in_a and not in_b else "B" if in_b and not in_a else ("both" if in_a else "none")

    def _serve_chat(self, handler: Any, payload: dict, path: str) -> None:
        auth = str(handler.headers.get("Authorization") or "")
        book = self._book_of(payload)
        row = {"book": book, "key": auth.replace("Bearer ", ""), "started": time.time(), "finished": None,
               "kind": self.server._classify(payload)}
        with self.lock:
            self.rows.append(row)
        if book == "B":
            self._sample()
        elif book == "A" and row["kind"] == "translation":
            row["parked"] = not self.release_a.is_set()
            if row["parked"]:
                self.a_parked.set()
            self.release_a.wait(self.PARK_SECONDS)
        try:
            self._serve(handler, payload, path)
        finally:
            row["finished"] = time.time()
            if book == "B" and row["kind"] == "translation":
                with self.lock:
                    self._b_answers += 1
                    count = self._b_answers
                hook = self.on_b_answer
                if hook is not None:
                    hook(count)

    def _sample(self) -> None:
        """Instance 1's and instance 2's environment, argv and key pool while book B's request runs."""
        from glossarion_mobile.services import instance_runtime

        runtime = instance_runtime.get(2)
        if runtime is None:
            return

        import api_key_encryption

        handler = api_key_encryption.get_handler()

        def pool(module: Any) -> list:
            """The keys of a client module's Translation pool, decrypted (the glossary extractor loads config.json's
            ``ENC:`` values)."""
            cls = getattr(module, "UnifiedClient", None)
            keys = getattr(cls, "_in_memory_multi_keys", None) if cls is not None else None
            return sorted({str(handler.decrypt_value(str(k.get("api_key") or ""))) for k in keys or ()
                           if isinstance(k, dict)})

        sample = {
            "base_instance": instance_runtime.base_environ().get("GLOSSARION_INSTANCE"),
            "instance_env": runtime.env.get("GLOSSARION_INSTANCE"),
            "base_argv": list(instance_runtime.base_argv()),
            "instance_argv": list(runtime.argv),
            "base_pool": pool(sys.modules.get("unified_api_client")),
            "instance_pool": pool(runtime.loaded("unified_api_client")),
        }
        with self.lock:
            self.samples.append(sample)


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _unified_csv(roots: list) -> Optional[Path]:
    for root in roots:
        for path in sorted(Path(root).rglob("glossary_unified.csv")):
            return path
    return None


def run(paths: Any = None, *, root: Optional[str] = None, keep: bool = False,
        log: Optional[Callable[[str], Any]] = None) -> dict:
    from glossarion_mobile.diagnostics import e2e
    from glossarion_mobile.diagnostics.fake_llm_server import DEFAULT_GLOSSARY, FAKE_MODEL, GlossaryEntry
    from glossarion_mobile.services import instance_runtime
    from glossarion_mobile.services.jobs import JobState

    session = e2e.E2ESession(paths, root=root, keep=keep, log=log)
    result: dict = {"ok": False, "checks": {}, "detail": {}}
    detail, checks = result["detail"], result["checks"]

    def begin(name: str) -> None:
        """The checks so far passed; ``name`` is the one running now (an exception fails it)."""
        for key, value in checks.items():
            if value == "running":
                checks[key] = "pass"
        checks[name] = "running"
        session.log(f"multi-instance check: {name}")

    try:
        begin("setup")
        session.setup()
        server = session.server
        server.glossary = tuple(DEFAULT_GLOSSARY) + tuple(GlossaryEntry(*entry) for entry in BOOK_B_NAMES)
        server.delay = 0.05
        names_a = tuple(entry.raw_name for entry in DEFAULT_GLOSSARY if entry.type == "character")
        names_b = tuple(entry[1] for entry in BOOK_B_NAMES if entry[0] == "character")
        recorder = Recorder(server, names_a, names_b)
        session.configure(
            "balanced",
            use_multi_api_keys=True,
            multi_api_keys=[
                {"api_key": KEY_ONE, "model": FAKE_MODEL, "enabled": True, "cooldown": 60},
                {"api_key": KEY_TWO, "model": FAKE_MODEL, "enabled": True, "cooldown": 60, "instance": 2},
            ],
            enable_unified_glossary=True,
        )
        book_a = session.import_epub("instance-one-book.epub")
        book_b_source = build_book_b(session.root / "fixtures" / "instance-two-book.epub")
        book_b = session.files.import_paths([str(book_b_source)], names=["instance-two-book.epub"])[0].path
        service = session.service
        before = e2e.process_state()
        installed_before = instance_runtime.installed()
        stopped: dict = {}

        def stop_b_after(count: int) -> None:
            if count == STOP_AFTER and "mode" not in stopped and stopped.get("job"):
                recorder.a_parked.wait(Recorder.PARK_SECONDS)  # book A is mid-run, a chapter request in flight
                stopped["at"] = time.time()
                try:
                    stopped["mode"] = service.request_stop(stopped["job"], reason="multi-instance scenario: stop B")
                finally:
                    recorder.release_a.set()  # book A goes on while book B stops

        recorder.on_b_answer = stop_b_after

        begin("concurrent_run")
        t0, t_start = time.monotonic(), time.time()
        job_a = service.submit_job(session.translate_spec(book_a, "instance-one-book.epub"), instance=1).job_id
        detail["runtime_installed_by_instance_1"] = instance_runtime.installed() and not installed_before
        stopped["job"] = job_b = service.submit_job(
            session.translate_spec(book_b, "instance-two-book.epub"), instance=2).job_id
        both_running = False
        deadline = time.monotonic() + session.job_timeout
        while time.monotonic() < deadline:
            running = {s.id for s in service.running_snapshots() if s.state is JobState.RUNNING}
            both_running = both_running or {job_a, job_b} <= running
            snaps = [service.snapshot(job_a), service.snapshot(job_b)]
            if all(s is not None and s.is_terminal for s in snaps):
                break
            time.sleep(0.05)
        recorder.release_a.set()
        snap_a, snap_b = service.snapshot(job_a), service.snapshot(job_b)
        _check(snap_a.is_terminal and snap_b.is_terminal, f"jobs did not end: {snap_a.state} / {snap_b.state}")
        _check(not detail["runtime_installed_by_instance_1"], "an instance-1 job installed the instance runtime")
        _check(both_running, "the two jobs never ran at the same time")
        _check(snap_a.instance == 1 and snap_b.instance == 2, f"instances {snap_a.instance} / {snap_b.instance}")
        a_rows = [r for r in recorder.rows if r["book"] == "A"]
        b_rows = [r for r in recorder.rows if r["book"] == "B"]
        overlap = any(ra["started"] < (rb["finished"] or 1e18) and rb["started"] < (ra["finished"] or 1e18)
                      for ra in a_rows for rb in b_rows)
        _check(overlap, "no request of book A overlapped a request of book B")

        begin("stop_one_instance")
        _check(snap_b.state is JobState.CANCELLED and stopped.get("mode") == "graceful",
               f"book B (instance 2) ended {snap_b.state} after stop {stopped.get('mode')}: {snap_b.error}")
        _check(snap_a.state is JobState.DONE, f"book A (instance 1) ended {snap_a.state}: {snap_a.error}")
        detail["parked_a"] = sum(1 for r in a_rows if r.get("parked"))
        _check(detail["parked_a"] >= 1, "book A had no chapter request in flight while book B ran")
        # Resume book B: it runs again on instance 2 with instance 2's key and finishes the book
        resumed = service.resume_job(job_b)
        _check(resumed is not None and not resumed.reused, f"Resume of book B: {resumed}")
        snap_r = session.wait(service, resumed.job_id)
        _check(snap_r.state is JobState.DONE and snap_r.instance == 2,
               f"resumed book B ended {snap_r.state} on instance {snap_r.instance}: {snap_r.error}")
        _check(service.wait_idle(60), "the service did not go idle")
        detail["secs"] = round(time.monotonic() - t0, 1)

        begin("keys_per_instance")
        a_rows = [r for r in recorder.rows if r["book"] == "A"]
        b_rows = [r for r in recorder.rows if r["book"] == "B"]
        keys_a = sorted({r["key"] for r in a_rows})
        keys_b = sorted({r["key"] for r in b_rows})
        detail["requests"] = {"A": len(a_rows), "B": len(b_rows), "keys_A": keys_a, "keys_B": keys_b,
                              "other": len(recorder.rows) - len(a_rows) - len(b_rows),
                              "rows": [(r["book"], r["kind"], r["key"][-3:], round(r["started"] - t_start, 2))
                                       for r in recorder.rows]}
        _check(keys_a == [KEY_ONE], f"book A's requests carried {keys_a}")
        _check(keys_b == [KEY_TWO], f"book B's requests carried {keys_b}")
        _check(any(r["kind"] == "glossary" for r in a_rows) and any(r["kind"] == "glossary" for r in b_rows),
               "both books should have sent glossary requests (Balanced)")

        begin("process_state_per_instance")
        samples = [s for s in recorder.samples if s["instance_pool"]]
        _check(samples, "no sample of instance 2's state was taken")
        detail["samples"] = samples[:2]
        for sample in samples:
            _check(sample["base_instance"] is None and sample["instance_env"] == "2", f"GLOSSARION_INSTANCE: {sample}")
            _check(sample["instance_pool"] == [KEY_TWO], f"instance 2's key pool: {sample['instance_pool']}")
            _check(KEY_TWO not in sample["base_pool"], f"instance 1's key pool holds instance 2's key: {sample}")
        # the glossary extractor and TransateKRtoEN read their book from sys.argv: each instance its own
        argv_b = [" ".join(s["instance_argv"]) for s in samples if len(s["instance_argv"]) > 1]
        _check(argv_b and all("instance-two-book" in a and "instance-one-book" not in a for a in argv_b),
               f"instance 2 argv: {argv_b}")
        argv_a = [" ".join(s["base_argv"]) for s in samples if len(s["base_argv"]) > 1]
        _check(all("instance-two-book" not in a for a in argv_a), f"instance 1 argv: {argv_a}")
        detail["argv_seen"] = {"instance_1": len([a for a in argv_a if "instance-one-book" in a]), "instance_2": len(argv_b)}

        begin("logs_per_instance")
        log_a = "\n".join(service.read_log_tail(job_a, 100000))
        log_b = "\n".join(service.read_log_tail(job_b, 100000))
        _check("instance-one-book" in log_a and "instance-two-book" not in log_a, "book A's log names book B")
        _check("instance-two-book" in log_b and "instance-one-book" not in log_b, "book B's log names book A")

        begin("outputs_separate")
        folder_a, folder_b = snap_a.output_dir, snap_r.output_dir
        _check(folder_a and folder_b and os.path.normcase(folder_a) != os.path.normcase(folder_b),
               f"output folders {folder_a!r} / {folder_b!r}")
        # the Library card counts of each workspace (``library_core._read_progress_summary``, the E2E's own reader)
        progress_a, progress_b = e2e._progress_summary(folder_a), e2e._progress_summary(folder_b)
        detail["progress"] = {book: {key: counts.get(key) for key in ("total", "completed", "failed")}
                              for book, counts in (("A", progress_a), ("B", progress_b))}
        _check(progress_a.get("total") == progress_a.get("completed") == e2e.CHAPTERS, f"book A progress {progress_a}")
        _check(progress_b.get("total") == progress_b.get("completed") == BOOK_B_CHAPTERS, f"book B progress {progress_b}")
        _check(any(p.lower().endswith(".epub") for p in snap_a.outputs) and any(
            p.lower().endswith(".epub") for p in snap_r.outputs), f"outputs {snap_a.outputs} / {snap_r.outputs}")

        begin("unified_glossary_merged")
        roots = [session.root, os.getcwd(), os.environ.get("GLOSSARION_DATA_DIR") or session.root]
        unified = _unified_csv(roots)
        _check(unified is not None, f"no glossary_unified.csv under {roots}")
        text = unified.read_text(encoding="utf-8-sig")
        # its terms (characters, which carry a gender, stay out of it by default: exclude gender entries)
        terms = [entry.raw_name for entry in DEFAULT_GLOSSARY if entry.type == "term"] + [
            entry[1] for entry in BOOK_B_NAMES if entry[0] == "term"]
        missing = [name for name in terms if name not in text]
        detail["unified"] = {"path": str(unified), "entries": max(0, len(text.splitlines()) - 1), "missing": missing}
        _check(not missing, f"the unified glossary lost {missing}")

        begin("hygiene")
        diff = e2e.diff_process_state(before, e2e.process_state())
        _check(not diff, f"the process state was not restored: {diff}")
        spawns = [e for e in (session.guards.spawns if session.guards else []) if not e.get("stdlib")]
        writes = list(session.audit.violations) if session.audit else []
        _check(not spawns, f"process spawn attempts: {spawns[:2]}")
        _check(not writes, f"writes outside the sandbox: {writes[:3]}")
        runtime = instance_runtime.get(2)
        client = runtime.loaded("unified_api_client") if runtime is not None else None
        payloads = str(client._payloads_dir()) if client is not None else ""
        detail["instance_2"] = {"module_copies": len(runtime.modules) if runtime is not None else 0,
                                "payloads": payloads}
        _check(payloads.rstrip("\\/").endswith("Instance 2"), f"instance 2's payloads go to {payloads!r}")
        begin("done")
        checks.pop("done", None)
        result["ok"] = True
        return result
    except BaseException as exc:  # noqa: BLE001 - the result says what failed
        session.failed = True
        for key, value in checks.items():
            if value == "running":
                checks[key] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        session.close()


def main(argv: Optional[list] = None, *, scenario: Optional[Callable[..., dict]] = None) -> int:
    """Host entry: bootstrap into the ``FLET_APP_STORAGE_*`` dirs (like ``diagnostics.e2e.main``), run, write JSON.
    ``scenario``: another scenario's ``run(paths, keep=..., variant=...)`` (``multi_instance_keys_scenario``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Glossarion mobile multi-instance end to end (fake OpenAI server)")
    parser.add_argument("--json", help="write the result JSON to this file")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox after a passing run")
    parser.add_argument("--variant", help="the scenario's variant (multi_instance_keys_scenario.VARIANTS)")
    args = parser.parse_args(argv)
    if args.json:
        args.json = os.path.abspath(args.json)  # bootstrap() changes the working directory
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.diagnostics import e2e

    os.environ[e2e.ISOLATED_ENV] = "1"
    paths = rb.bootstrap()
    rb.warm_import(emit=False)  # the app's own start-up imports (the base modules every instance copies)
    if scenario is None:
        result = run(paths, keep=args.keep)
    else:
        result = scenario(paths, keep=args.keep, **({"variant": args.variant} if args.variant else {}))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("ok", "checks")}, ensure_ascii=False), file=sys.stderr, flush=True)
    if not result["ok"]:
        print(result.get("traceback", ""), file=sys.stderr, flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
