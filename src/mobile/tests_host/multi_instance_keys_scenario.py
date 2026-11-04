"""Multi-instance key isolation end to end (design §6b): no API key of one instance reaches the other's requests.

Owner: "instance 2 must never use the fallback key of instance 1, we must not have instance contamination"; the main
API key is instance 1's. The real JobService, instance runtime, HeadlessOwner and shared pipeline, offline
(``diagnostics.e2e.E2ESession``, like ``multi_instance_scenario``): book A runs in instance 1 and book B in instance
2 at the same time. The Translation, Fallback, Glossary and Metadata pools hold keys of both instances, and the main
API key (instance 1's) has "Use Main GUI Key as Fallback #1" on. The fake server records the key of every request
(``KeyRecorder``) and answers the chapter requests of the ``BLOCKED`` keys with an empty ``content_filter`` answer
(prohibited content: the client's main-key retry, then the Fallback pool):

* instance 2's Translation key is blocked: its chapters are rescued by instance 2's own fallback key only - never
  instance 1's fallback key, never the main API key, never a key of instance 1's pools;
* instance 1's Translation key and the main API key are blocked: its chapters are rescued by its own fallback key and
  it never uses a key of instance 2;
* ``variant`` "glossary": instance 2 has a Glossary key of its own (its glossary requests use it) and no Metadata
  key (that pool is off for it: its book-title requests use its own Translation key, blocked, then its fallback
  key); "metadata": the other way round (its glossary requests use its own Translation key);
* each job ran under its instance's key scope (``GLOSSARION_KEY_INSTANCE``); afterwards nothing of it is left.

``run()`` returns the result as a dict (``ok``, ``checks``, ``detail``). Host: ``tests_host/test_multi_instance_keys.py``
runs ``main()`` in a subprocess (``multi_instance_scenario.main``'s bootstrap). Python 3.10 compatible; no pytest.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_instance_scenario import BOOK_B_NAMES, Recorder, _check, build_book_b  # noqa: E402

MAIN_KEY = "sk-mi-main-gui-key"  # config api_key: instance 1's
TR_ONE, TR_TWO = "sk-mi-translation-one", "sk-mi-translation-two"
FB_ONE, FB_TWO = "sk-mi-fallback-one", "sk-mi-fallback-two"
GL_ONE, GL_TWO = "sk-mi-glossary-one", "sk-mi-glossary-two"
MD_ONE, MD_TWO = "sk-mi-metadata-one", "sk-mi-metadata-two"
ONE = frozenset({MAIN_KEY, TR_ONE, FB_ONE, GL_ONE, MD_ONE})
TWO = frozenset({TR_TWO, FB_TWO, GL_TWO, MD_TWO})
#: their translation requests (chapters, book title) get an empty content_filter answer
BLOCKED = frozenset({TR_ONE, TR_TWO, MAIN_KEY})
#: which dedicated pool instance 2 has a key of (the other one is off for it)
VARIANTS = ("glossary", "metadata")
#: a name only book A's requests carry / only book B's (its characters, a place in its title)
TITLE_A, TITLE_B = "은빛 숲", "흑룡 계곡"


def _key(api_key: str, instance: int = 1) -> dict:
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    entry = {"api_key": api_key, "model": FAKE_MODEL, "enabled": True, "cooldown": 60}
    if instance != 1:
        entry["instance"] = instance  # stored only when it is not 1 (APIKeyEntry.to_dict)
    return entry


def _send_blocked(handler: Any, payload: dict) -> None:
    """An empty answer stopped by the provider's content filter (``finish_reason: content_filter``)."""
    model = str(payload.get("model") or "")
    if payload.get("stream"):
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}]
        body = "".join("data: " + json.dumps({"id": "chatcmpl-blocked", "object": "chat.completion.chunk",
                                              "created": 1, "model": model, **c}) + "\n\n" for c in chunks)
        raw = (body + "data: [DONE]\n\n").encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
        handler.wfile.flush()
        return
    handler._send_json(200, {"id": "chatcmpl-blocked", "object": "chat.completion", "created": 1, "model": model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                          "finish_reason": "content_filter", "logprobs": None}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}})


class KeyRecorder(Recorder):
    """``Recorder`` (the book, key and kind of every request) that answers the ``blocked`` keys' translation requests
    with an empty ``content_filter`` answer and notes each instance's key scope while the request runs. No request
    waits (book A is not parked)."""

    def __init__(self, server: Any, names_a: tuple, names_b: tuple, blocked: frozenset) -> None:
        super().__init__(server, names_a, names_b)
        self.release_a.set()
        self.blocked = blocked

    def _serve_chat(self, handler: Any, payload: dict, path: str) -> None:
        from glossarion_mobile.diagnostics.fake_llm_server import _content_text, chapter_numbers
        from glossarion_mobile.services import instance_runtime

        key = str(handler.headers.get("Authorization") or "").replace("Bearer ", "")
        kind = self.server._classify(payload)
        users = [m for m in payload.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
        runtime = instance_runtime.get(2)
        row = {"book": self._book_of(payload), "key": key, "kind": kind, "started": time.time(), "finished": None,
               "chapter": bool(chapter_numbers(_content_text(users[-1].get("content")) if users else "")),
               "blocked": key in self.blocked and kind == "translation",
               # the key scope of instance 1's process environment and of instance 2's own one, right now
               "scope_1": instance_runtime.base_environ().get("GLOSSARION_KEY_INSTANCE"),
               "scope_2": runtime.env.get("GLOSSARION_KEY_INSTANCE") if runtime is not None else None}
        with self.lock:
            self.rows.append(row)
        try:
            if row["blocked"]:
                _send_blocked(handler, payload)
            else:
                self._serve(handler, payload, path)
        finally:
            row["finished"] = time.time()


def run(paths: Any = None, *, root: Optional[str] = None, keep: bool = False, variant: str = "glossary",
        log: Optional[Callable[[str], Any]] = None) -> dict:
    from glossarion_mobile.diagnostics import e2e
    from glossarion_mobile.diagnostics.fake_llm_server import DEFAULT_GLOSSARY, GlossaryEntry
    from glossarion_mobile.services.jobs import JobState

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    session = e2e.E2ESession(paths, root=root, keep=keep, log=log)
    result: dict = {"ok": False, "checks": {}, "detail": {"variant": variant}}
    detail, checks = result["detail"], result["checks"]

    def begin(name: str) -> None:
        for key, value in checks.items():
            if value == "running":
                checks[key] = "pass"
        checks[name] = "running"
        session.log(f"multi-instance keys check: {name}")

    try:
        begin("setup")
        session.setup()
        server = session.server
        server.glossary = tuple(DEFAULT_GLOSSARY) + tuple(GlossaryEntry(*entry) for entry in BOOK_B_NAMES)
        server.delay = 0.02
        names_a = tuple(e.raw_name for e in DEFAULT_GLOSSARY if e.type == "character") + (TITLE_A,)
        names_b = tuple(e[1] for e in BOOK_B_NAMES if e[0] == "character") + (TITLE_B,)
        recorder = KeyRecorder(server, names_a, names_b, BLOCKED)
        own_glossary = variant == "glossary"
        session.configure(
            "balanced",
            api_key=MAIN_KEY,
            use_main_key_fallback=True,
            use_multi_api_keys=True,
            multi_api_keys=[_key(TR_ONE), _key(TR_TWO, 2)],
            use_fallback_keys=True,
            fallback_keys=[_key(FB_ONE), _key(FB_TWO, 2)],
            use_glossary_keys=True,
            glossary_keys=[_key(GL_ONE)] + ([_key(GL_TWO, 2)] if own_glossary else []),
            use_metadata_keys=True,
            metadata_keys=[_key(MD_ONE)] + ([] if own_glossary else [_key(MD_TWO, 2)]),
        )
        book_a = session.import_epub("keys-instance-one-book.epub")
        # its table of contents names its place too: every request of book B (its nav page's too) is told apart
        book_b_source = build_book_b(session.root / "fixtures" / "keys-instance-two-book.epub",
                                     title_format="제{index}화 " + TITLE_B)
        book_b = session.files.import_paths([str(book_b_source)], names=["keys-instance-two-book.epub"])[0].path
        service = session.service
        before = e2e.process_state()

        begin("concurrent_run")
        job_a = service.submit_job(session.translate_spec(book_a, "keys-instance-one-book.epub"), instance=1).job_id
        job_b = service.submit_job(session.translate_spec(book_b, "keys-instance-two-book.epub"), instance=2).job_id
        both_running = False
        deadline = time.monotonic() + session.job_timeout
        while time.monotonic() < deadline:
            running = {s.id for s in service.running_snapshots() if s.state is JobState.RUNNING}
            both_running = both_running or {job_a, job_b} <= running
            snaps = [service.snapshot(job_a), service.snapshot(job_b)]
            if all(s is not None and s.is_terminal for s in snaps):
                break
            time.sleep(0.05)
        _check(service.wait_idle(60), "the service did not go idle")
        snap_a, snap_b = service.snapshot(job_a), service.snapshot(job_b)
        rows = list(recorder.rows)
        detail["rows"] = [(r["book"], r["kind"], "chapter" if r["chapter"] else "other", r["key"][6:],
                           "blocked" if r["blocked"] else "ok", r["scope_1"], r["scope_2"]) for r in rows]
        detail["states"] = {"A": snap_a.state.value, "B": snap_b.state.value, "errors": [snap_a.error, snap_b.error]}
        _check(both_running, "the two jobs never ran at the same time")
        _check(snap_a.state is JobState.DONE, f"book A (instance 1) ended {snap_a.state}: {snap_a.error}")
        _check(snap_b.state is JobState.DONE, f"book B (instance 2) ended {snap_b.state}: {snap_b.error}")
        a_rows = [r for r in rows if r["book"] == "A"]
        b_rows = [r for r in rows if r["book"] == "B"]
        overlap = any(ra["started"] < (rb["finished"] or 1e18) and rb["started"] < (ra["finished"] or 1e18)
                      for ra in a_rows for rb in b_rows)
        _check(overlap, "no request of book A overlapped a request of book B")
        others = [r for r in rows if r["book"] not in ("A", "B")]
        _check(not others, f"requests of neither book: {[(r['kind'], r['key']) for r in others]}")

        begin("instance_2_keys")
        keys_b = sorted({r["key"] for r in b_rows})
        detail["keys_B"] = keys_b
        _check(set(keys_b) <= TWO, f"book B (instance 2) used a key of instance 1: {sorted(set(keys_b) - TWO)}")
        _check(MAIN_KEY not in keys_b and FB_ONE not in keys_b, f"book B reached the main / fallback key: {keys_b}")
        b_chapters = [r for r in b_rows if r["kind"] == "translation" and r["chapter"]]
        rescued_b = [r for r in b_chapters if r["key"] == FB_TWO]
        _check(any(r["key"] == TR_TWO and r["blocked"] for r in b_chapters), "book B's Translation key was not tried")
        _check(len(rescued_b) >= 5, f"instance 2's own fallback key rescued {len(rescued_b)} of book B's chapters")
        _check(all(r["scope_2"] == "2" for r in b_rows), "book B ran without instance 2's key scope")

        begin("instance_1_keys")
        keys_a = sorted({r["key"] for r in a_rows})
        detail["keys_A"] = keys_a
        _check(set(keys_a) <= ONE, f"book A (instance 1) used a key of instance 2: {sorted(set(keys_a) - ONE)}")
        a_chapters = [r for r in a_rows if r["kind"] == "translation" and r["chapter"]]
        _check(any(r["key"] == MAIN_KEY for r in a_chapters), "instance 1's main-key retry did not run")
        _check(len([r for r in a_chapters if r["key"] == FB_ONE]) >= e2e.CHAPTERS,
               "instance 1's own fallback key did not rescue every chapter of book A")
        _check(all(r["scope_1"] == "1" for r in a_rows), "book A ran without instance 1's key scope")

        begin("dedicated_pools")
        glossary_b = sorted({r["key"] for r in b_rows if r["kind"] == "glossary"})
        glossary_a = sorted({r["key"] for r in a_rows if r["kind"] == "glossary"})
        title_b = [r for r in b_rows if r["kind"] == "translation" and not r["chapter"]]
        title_a = [r for r in a_rows if r["kind"] == "translation" and not r["chapter"]]
        detail["pools"] = {"glossary_A": glossary_a, "glossary_B": glossary_b,
                           "other_A": sorted({r["key"] for r in title_a}), "other_B": sorted({r["key"] for r in title_b})}
        _check(glossary_a == [GL_ONE], f"book A's glossary requests used {glossary_a}")
        _check(title_a and {r["key"] for r in title_a} == {MD_ONE}, f"book A's title requests used {title_a}")
        if own_glossary:
            # its own Glossary key; no Metadata key of its own: its book title goes to its Translation key (blocked),
            # then its own fallback key
            _check(glossary_b == [GL_TWO], f"book B's glossary requests used {glossary_b}")
            _check(title_b and {r["key"] for r in title_b} == {TR_TWO, FB_TWO} and title_b[0]["key"] == TR_TWO,
                   f"book B's title requests used {[r['key'] for r in title_b]}")
        else:
            # no Glossary key of its own: its glossary requests use its Translation key; its own Metadata key
            _check(glossary_b == [TR_TWO], f"book B's glossary requests used {glossary_b}")
            _check(title_b and {r["key"] for r in title_b} == {MD_TWO}, f"book B's title requests used {title_b}")
        session.log(f"multi-instance keys: {json.dumps(detail['pools'])}")

        begin("hygiene")
        import instance_plan

        diff = e2e.diff_process_state(before, e2e.process_state())
        _check(not diff, f"the process state was not restored: {diff}")
        _check(os.environ.get("GLOSSARION_KEY_INSTANCE") is None and instance_plan.key_scope() is None,
               "instance 1's key scope was left in the process")
        spawns = [e for e in (session.guards.spawns if session.guards else []) if not e.get("stdlib")]
        writes = list(session.audit.violations) if session.audit else []
        _check(not spawns, f"process spawn attempts: {spawns[:2]}")
        _check(not writes, f"writes outside the sandbox: {writes[:3]}")
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


if __name__ == "__main__":  # pragma: no cover
    import multi_instance_scenario

    raise SystemExit(multi_instance_scenario.main(scenario=run))
