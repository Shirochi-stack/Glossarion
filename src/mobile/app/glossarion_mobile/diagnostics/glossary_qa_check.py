"""Glossary document + QA quick scan check (U6 self-test; ``tools/host_smoke.py`` runs it too).

``check_glossary_document`` drives the Glossary Manager's document the way the mobile
editor does (``glossary_document.GlossaryDocument``, the desktop Glossary Editor's steps):
a token-CSV glossary is parsed, one translated name is edited (one undo step), saved,
parsed again (the edit is there, the other entries are unchanged) and saved once more
without edits (byte-stable). ``check_qa_quick_scan`` runs the QA Scanner's quick scan on a
small translated workspace through ``qa_scan_runtime.run_qa_scan_path`` (the path the
``qa_scan`` job takes): on Glossarion Mobile the scan is forced onto threads
(``mobile_qa_forcing_active``; host_smoke's process tripwires catch any spawn), every
chapter is in ``validation_results.json`` and ``find_latest_qa_report`` finds the HTML
report.

Both run on scratch files only (nothing of the user's Library, Output or settings is
touched). Python 3.10 compatible; no Flet import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["GlossaryQaCheckFailure", "check_glossary_document", "check_qa_quick_scan", "SAMPLE_ENTRIES"]

#: The sample glossary (token CSV): two characters and a term.
SAMPLE_ENTRIES = (
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": "a witch"},
    {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "male", "description": ""},
    {"type": "term", "raw_name": "마나", "translated_name": "Mana", "gender": "", "description": ""},
)
EDITED_NAME = "Lunaria"
_CHAPTERS = (
    "Luna walked into the old library. The shelves were tall and dusty, and the lamps flickered softly.",
    "Kai answered the door. He had been waiting for hours, and the rain had not stopped all evening.",
    "The mana stones glowed when Luna touched them, and the whole room filled with a pale blue light.",
)


class GlossaryQaCheckFailure(AssertionError):
    """A glossary document / QA scan expectation failed."""


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise GlossaryQaCheckFailure(message)


def _names(doc: Any) -> dict:
    return {str(e.get("raw_name")): str(e.get("translated_name")) for e in doc.current_glossary_data or []
            if isinstance(e, dict)}


def check_glossary_document(work: Path) -> dict:
    """Parse -> edit -> save -> re-parse -> save again (byte-stable) a token-CSV glossary."""
    import glossary_document as gd

    folder = Path(work) / "Glossary" / "Selftest"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Selftest_glossary.csv"
    gd.write_token_csv([dict(e) for e in SAMPLE_ENTRIES], str(path), [], ["description"])
    logs: list = []
    config = {"glossary_auto_backup": False, "update_html_on_save": False}
    doc = gd.GlossaryDocument.open(str(path), dict(config), log=logs.append)
    _check(doc.current_glossary_format == "token_csv", f"parsed as {doc.current_glossary_format!r}, not token_csv")
    _check(_names(doc) == {e["raw_name"]: e["translated_name"] for e in SAMPLE_ENTRIES},
           f"parsed entries {_names(doc)}")
    rows = doc.rows()
    _check(len(rows) == len(SAMPLE_ENTRIES), f"{len(rows)} editor rows for {len(SAMPLE_ENTRIES)} entries")
    target = next(r for r in rows if "루나" in r.texts)
    stored, _ref = doc.edit_cell(target.source_ref, "translated_name", EDITED_NAME)
    _check(stored == EDITED_NAME and doc.can_undo(), "the cell edit was not applied with an undo step")
    _check(doc.translated_changes() == [("Luna", EDITED_NAME)], f"changes {doc.translated_changes()}")
    report = doc.save_edits(update_output_files=False)
    _check(report.get("saved"), f"save_edits did not save: {report} {logs[-3:]}")
    saved = path.read_bytes()
    again = gd.GlossaryDocument.open(str(path), dict(config))
    expected = {e["raw_name"]: e["translated_name"] for e in SAMPLE_ENTRIES}
    expected["루나"] = EDITED_NAME
    _check(_names(again) == expected, f"re-parsed entries {_names(again)}")
    _check(again.save(), "saving the unchanged glossary failed")
    _check(path.read_bytes() == saved, "saving an unchanged glossary changed its bytes")
    return {"format": doc.current_glossary_format, "entries": len(rows), "bytes": len(saved),
            "columns": list(doc.glossary_column_fields or [])}


def check_qa_quick_scan(work: Path) -> dict:
    """A QA quick scan of a three-chapter workspace through the ``qa_scan`` job's shared path."""
    import qa_scan_runtime

    output = Path(work) / "Output"
    folder = output / "Selftest"
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for number, text in enumerate(_CHAPTERS, start=1):
        name = f"response_{number:03d}_chapter{number}.html"
        (folder / name).write_text(
            f"<html><head><title>Chapter {number}</title></head><body><h1>Chapter {number}</h1>"
            f"<p>{text}</p><p>{text}</p></body></html>", encoding="utf-8")
        names.append(name)
    logs: list = []
    forced = bool(qa_scan_runtime.mobile_qa_forcing_active())
    result = qa_scan_runtime.run_qa_scan_path(str(folder), log=logs.append, mode="quick-scan", config={})
    reports = sorted(p for p in folder.rglob("validation_results.json"))
    _check(reports, f"no validation_results.json was written (last log lines: {logs[-4:]})")
    rows = json.loads(reports[0].read_text(encoding="utf-8"))
    _check(sorted(r.get("filename") for r in rows) == names, f"report rows {[r.get('filename') for r in rows]}")
    latest = qa_scan_runtime.find_latest_qa_report(str(output))
    _check(latest and os.path.basename(latest) == "validation_results.html" and os.path.isfile(latest),
           f"find_latest_qa_report found {latest!r}")
    threads = any("ThreadPoolExecutor" in str(line) for line in logs)
    if forced:
        _check(threads, "the mobile scan did not run on threads (no ThreadPoolExecutor line in the log)")
    return {"files": len(rows), "issues": sum(len(r.get("issues") or []) for r in rows), "forced_threads": forced,
            "thread_pool": threads, "report": os.path.relpath(latest, str(folder)), "result": type(result).__name__}
