"""The self-test's ``scan_interpreters`` check: does a QA scan batch run in a child interpreter on this device?

Starts a child through ``services.interpreters``, imports the scanner in it and runs one real batch of a small
translated folder (broken tag, untranslated Korean), then runs the same batch on this thread: the findings must match.

The glossary's smart filter and Generate MD / TXT go the same way through their real wiring (``"filter"``, ``"sidecars"``
in the detail): the same result as on this thread, or why not. ``"kept"`` is how many children stayed for the next job.

What CI reads is the ``GLOSSARION_INTERPRETERS {json}`` line ``check`` prints: ``"result": "pass"`` or
``"result": "fallback"`` with the reason, on Android and iOS alike. A device where a child cannot start scans on threads
(``job_kinds.qa._ScanChildren``), so a fallback is not a failure there. Android is the exception: the emulator proved
it works there, so a fallback on Android means a build lost it. Different findings fail on every platform. A native
crash in the child would take the app with it: CI then shows a smoke that died, not a fallback.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["MARKER", "check_scan_in_child"]

MARKER = "GLOSSARION_INTERPRETERS"

_TEXT = ("The knight walked into the hall and greeted everyone warmly while the rain fell over the old castle walls. "
         "She said nothing, but her eyes followed the sword across the table where the merchants argued. ")


def _write_chapters(folder: Path) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    long = _TEXT * 30  # below about 500 words the small-file rules leave some checks out
    bodies = (f"<p>{long}</p>",
              f"<p>{long}</p><p>{'기사는 홀로 걸어 들어가 모두에게 인사했다. ' * 8}</p>",
              f'<p>{long}</p><p"{long}</p>')
    names = []
    for number, body in enumerate(bodies, start=1):
        name = f"response_{number:03d}_chapter{number}.html"
        (folder / name).write_text(f"<html><head><title>Chapter {number}</title></head><body><h1>Chapter {number}</h1>"
                                   f"{body}</body></html>", encoding="utf-8")
        names.append(name)
    return names


def _findings(rows: Any) -> list:
    return sorted((row.get("filename"), sorted(str(issue) for issue in row.get("issues") or ())) for row in rows)


def _lost_on_android(platform: str, what: str, lines: list) -> str:
    if platform == "android":
        raise AssertionError(f"{what} does not run in a child interpreter on Android any more: {lines}")
    return "fallback"


def _check_filter(work: Path, platform: str) -> str:
    """The glossary's smart filter on a text just long enough to get a child."""
    try:
        import GlossaryManager as backend
    except ImportError as exc:
        return f"not in this build: {exc}"[:120]
    from glossarion_mobile.job_kinds.glossary import FILTER_CHILD_MIN_CHARS, filter_in_child

    sentence = "김서연님은 왕궁에서 박지훈에게 공작 이야기를 하며 걸어갔다. 이수아씨가 숲속에서 최민준을 바라보았다. "
    text = sentence * (FILTER_CHILD_MIN_CHARS // len(sentence) + 2)
    here = backend._filter_text_for_glossary(text, 2, 200)
    lines: list = []
    with filter_in_child(lines.append):
        there = backend._filter_text_for_glossary(text, 2, 200)
    if any("on threads" in str(line) for line in lines):
        return _lost_on_android(platform, "the glossary filter", lines)
    if here != there or not there[0] or not any("separate Python interpreter" in str(line) for line in lines):
        raise AssertionError(f"the glossary filter differs in a child interpreter: {lines}, {there[0][:80]!r}")
    return "pass"


def _check_sidecars(work: Path, platform: str) -> str:
    """Generate MD / TXT: an English chapter in a child and a Korean one back on the thread."""
    import md_txt_sidecar_writer as writer

    from glossarion_mobile.job_kinds.compile import SIDECARS_CHILD_MIN_FILES, sidecars_in_children

    results = []
    lines: list = []
    for name, children in (("here", False), ("there", True)):
        folder = work / f"sidecars_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "response_1.html").write_text(f"<html><body><h1>One</h1><p>{_TEXT}</p></body></html>", encoding="utf-8")
        (folder / "response_2.html").write_text(f"<p>{'기사는 홀로 걸어 들어가 모두에게 인사했다. ' * 8}</p>", encoding="utf-8")
        with sidecars_in_children(lines.append if children else (lambda _line: None),
                                  SIDECARS_CHILD_MIN_FILES if children else 0):
            done = writer.generate_md_txt_for_output_dir(str(folder), do_md=True, do_txt=False, max_workers=2, log=lines.append)
        results.append((done, {path.name: path.read_bytes() for path in (folder / "MD").glob("*.md")}))
    if any("on threads" in str(line) for line in lines):
        return _lost_on_android(platform, "Generate MD / TXT", lines)
    if results[0] != results[1] or results[0][0] != (2, 0, 2) or not any("separate Python interpreter" in str(line) for line in lines):
        raise AssertionError(f"Generate MD / TXT differs in a child interpreter: {lines}")
    return "pass"


def check_scan_in_child(work: Path, platform: str) -> dict:
    """The detail of the check. Raises only for what must fail the smoke (see the module's docstring)."""
    import scan_html_folder  # here first: see ``services.interpreters`` on a module a child loads first

    from glossarion_mobile.services import interpreters

    folder = Path(work) / "Book"
    names = _write_chapters(folder)
    settings = {"target_language": "english", "foreign_char_threshold": 0, "check_translation_artifacts": True,
                "check_missing_tag_bracket": True, "min_file_length": 0}
    # ``process_html_file_batch``'s shorter argument tuple, the one that needs no source book
    args = (list(enumerate(names)), str(folder), settings, "quick-scan", {}, {}, False, {}, {})
    detail: dict = {"result": "fallback", "python": sys.version.split()[0], "platform": platform,
                    "sp_bootstrap": importlib.util.find_spec("_sp_bootstrap") is not None}
    started = time.monotonic()
    pool = interpreters.InterpreterPool(1, preload=("scan_html_folder",), work_dir=str(work))
    try:
        rows = pool.call("scan_html_folder", "process_html_file_batch", args, should_stop=lambda: False)
        detail["secs"] = round(time.monotonic() - started, 2)
    except interpreters.InterpreterUnavailable as exc:
        detail["reason"] = " ".join(str(exc).split())[-300:]
        rows = None
    finally:
        pool.close()
        detail["released"] = pool.live() == 0
    if rows is None:
        if platform == "android":
            raise AssertionError(f"a child interpreter does not work on Android any more: {detail['reason']}")
        return detail
    here, there = _findings(scan_html_folder.process_html_file_batch(args)), _findings(rows)
    if here != there:
        raise AssertionError(f"a batch finds other things in a child interpreter: {there} against {here}")
    if len(there) != len(names) or not any(issues for _name, issues in there):
        raise AssertionError(f"the batch looked at nothing: {there}")
    detail.update(result="pass", files=len(there), issues=sum(len(issues) for _name, issues in there))
    detail["filter"] = _check_filter(work, platform)
    detail["sidecars"] = _check_sidecars(work, platform)
    kept = interpreters.kept_pool()
    _total, free = interpreters.memory_now()
    detail.update(kept=kept.live(), kept_started=kept.started, kept_destroyed=kept.destroyed,
                  free_mb=int(free / (1024 * 1024)) if free else None)
    return detail
