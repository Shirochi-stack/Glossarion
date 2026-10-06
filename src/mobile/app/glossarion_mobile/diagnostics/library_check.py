"""Library / Book page / Progress / Reader check on a translated workspace (U5 self-test).

``verify_book`` opens one translated workspace the way the app does, through the
Library's own Flet-free code paths over the shared cores:

1. **Library scan**: ``services.library.LibraryService.scan_blocking`` (``library_core.scan_library``
   + ``card_progress_view``): the book is on the expected shelf with the expected card
   counts, pill and raw source;
2. **Book page**: ``load_details_blocking`` (preview + full, ``library_core.load_book_details``)
   and ``details_model`` (``BookDetailsModel``): title and chapter list;
3. **Chapters tab**: ``ui.library.progress_model.load_progress_view`` (``progress_core``):
   the chapter rows and their Progress Manager statuses;
4. **Reader**: ``ui.reader.session.plan_open`` (``library_core.plan_open_reader``) +
   ``ReaderSession`` + ``ui.reader.document.DocumentBuilder`` (``reader_doc.ReaderDocument``
   with the mobile page shell): the translated chapters carry the expected text, the
   original side still has the Korean source, and the built page has the mobile shell
   and its event bridge.

Everything runs inside :func:`isolated_library`: the shared Library environment
(``library_core.install_library_env``: Library root, Output root, cover / EPUB caches)
and ``GLOSSARION_LIBRARY_DIR`` / ``OUTPUT_DIRECTORY`` point at a sandbox for the length
of the check, and the app's own environment is put back afterwards, so a device
self-test never reads or writes the user's Library or Output.

Used by the ``smoke`` self-test (``check_library_reader``: a synthetic, partly
translated workspace of the self-test EPUB from :func:`selftest_workspace`; host:
``tools/host_smoke.py``) and by the ``e2e`` suite (the workspace the real translate +
compile jobs produced). Python 3.10 compatible; no Flet import.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import types
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

__all__ = ["SELFTEST_MARKER", "isolated_library", "selftest_workspace", "spine_documents", "verify_book"]

#: Text every translated chapter of the synthetic workspace carries.
SELFTEST_MARKER = "GLSELFTEST"
_HANGUL = re.compile("[가-힣]")
_TAG = re.compile(r"<[^>]+>")
_ENV_KEYS = ("GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY")


class LibraryCheckFailure(AssertionError):
    """A Library / Book page / Progress / Reader expectation failed."""


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise LibraryCheckFailure(message)


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _text(html: str) -> str:
    return _TAG.sub(" ", str(html or "").split("<body", 1)[-1])


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def isolated_library(root: Path, config: Optional[Mapping[str, Any]] = None) -> Iterator[Any]:
    """A ``LibraryService`` on ``root/Library``, ``root/Output`` and ``root/cache`` (shared env pinned).

    Restores ``GLOSSARION_LIBRARY_DIR`` / ``OUTPUT_DIRECTORY`` and the previously installed
    ``library_core.LibraryEnv`` (the app's own; none in a bare host process) on exit.
    """
    import library_core

    from glossarion_mobile.services.library import LibraryService

    root = Path(root)
    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    current = getattr(library_core, "current_library_env", None)
    previous = current() if callable(current) else None
    paths = types.SimpleNamespace(library=root / "Library", output=root / "Output", cache=root / "cache")
    for directory in (paths.library, paths.output, paths.cache):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["GLOSSARION_LIBRARY_DIR"] = str(paths.library)
    os.environ["OUTPUT_DIRECTORY"] = str(paths.output)
    service = LibraryService(paths=paths, config=dict(config or {}))
    try:
        _check(service.ensure_env() is not None, "library_core.install_library_env is missing from this build")
        yield service
    finally:
        try:
            if previous is not None:
                library_core.install_library_env(previous)
            else:
                library_core.uninstall_library_env()
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def spine_documents(epub: Path) -> list:
    """The spine's XHTML documents of ``epub`` as ``(href, basename)`` in reading order (nav excluded)."""
    with zipfile.ZipFile(epub) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
        opf_path = rootfile.get("full-path") or ""
        opf = ET.fromstring(archive.read(opf_path))
    base = os.path.dirname(opf_path)
    items = {}
    for el in opf.iter():
        if el.tag.endswith("}item") or el.tag == "item":
            items[el.get("id")] = (el.get("href") or "", el.get("media-type") or "", el.get("properties") or "")
    out = []
    for el in opf.iter():
        if el.tag.endswith("}itemref") or el.tag == "itemref":
            href, media, props = items.get(el.get("idref"), ("", "", ""))
            if "nav" in props.split() or "html" not in media:
                continue
            out.append((f"{base}/{href}" if base else href, os.path.basename(href)))
    return out


def selftest_workspace(root: Path, epub: Path, *, name: str = "selftest-library") -> dict:
    """A partly translated workspace of ``epub`` under ``root`` (the ``smoke`` check's book).

    ``Library/Raw/<name>.epub`` is the raw source (found by the shared resolver's
    Library/Raw name rule); ``Output/<name>/`` holds a desktop-format
    ``translation_progress.json`` and response files: the first chapters completed (their
    text carries :data:`SELFTEST_MARKER`), then one ``qa_failed`` chapter (its response
    file kept), one ``in_progress`` chapter (no file yet) and the rest not started.
    """
    root = Path(root)
    raw_dir = root / "Library" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{name}.epub"
    shutil.copyfile(epub, raw)
    output = root / "Output" / name
    output.mkdir(parents=True, exist_ok=True)
    documents = spine_documents(raw)
    _check(len(documents) >= 4, f"{Path(epub).name}: only {len(documents)} spine documents")
    completed = len(documents) - 4 if len(documents) > 6 else len(documents) - 2
    chapters: dict = {}
    statuses: dict = {}
    for number, (_href, basename) in enumerate(documents, start=1):
        stem = os.path.splitext(basename)[0]
        output_file = f"response_{stem}.html"
        if number <= completed:
            status = "completed"
        elif number == completed + 1:
            status = "qa_failed"
        elif number == completed + 2:
            status = "in_progress"
        else:
            continue
        if status in ("completed", "qa_failed"):
            (output / output_file).write_text(
                "<html><head><title>Chapter {0}</title></head><body><h1>Chapter {0}</h1>"
                "<p>{1} Translated chapter {0} of the self-test book.</p></body></html>".format(number, SELFTEST_MARKER),
                encoding="utf-8")
        entry = {"actual_num": number, "content_hash": f"selftest-{number:04d}", "output_file": output_file,
                 "status": status, "last_updated": 1_000_000.0 + number, "original_basename": basename,
                 "model_name": "selftest"}
        if status == "qa_failed":
            entry["qa_issues"] = True
            entry["qa_issues_found"] = ["korean_text_found_12_chars_"]
        chapters[str(number)] = entry
        statuses[number] = status
    progress = {"version": "2.1", "chapters": chapters, "chapter_chunks": {}}
    (output / "translation_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    counts = Counter(statuses.values())
    return {
        "name": name,
        "raw": raw,
        "output": output,
        "documents": len(documents),
        "completed": counts["completed"],
        # the Reader overlay shows every chapter with a response file (the QA-failed one too)
        "translated_files": counts["completed"] + counts["qa_failed"],
        "statuses": {
            "completed": counts["completed"],
            "qa_failed": counts["qa_failed"],
            "in_progress": counts["in_progress"],
            "not_translated": len(documents) - len(statuses),
        },
    }


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _find_book(snapshot: Any, output_dir: Path) -> tuple:
    target = _norm(output_dir)
    for shelf in ("in_progress", "completed"):
        for book in getattr(snapshot, shelf):
            folders = [book.get("output_folder"), book.get("path")]
            if book.get("path") and str(book.get("path")).lower().endswith(".epub"):
                folders.append(os.path.dirname(str(book.get("path"))))
            if any(f and _norm(f) == target for f in folders):
                return shelf, dict(book)
    return None, None


def verify_book(
    service: Any,
    *,
    output_dir: Path,
    raw_path: Optional[Path],
    shelf: str,
    completed: int,
    total: Optional[int] = None,
    statuses: Optional[Mapping[str, int]] = None,
    marker: str = SELFTEST_MARKER,
    translated_chapters: Optional[int] = None,
    title: Optional[str] = None,
) -> dict:
    """Open the workspace ``output_dir`` in the Library, Book page, Chapters tab and Reader.

    ``service`` comes from :func:`isolated_library`. ``completed`` / ``total`` are the card
    counts (``total`` None: at least ``completed``), ``statuses`` the Chapters tab's
    chapter-row status counts, ``translated_chapters`` how many Reader chapters carry
    ``marker`` (default ``completed``).
    """
    from glossarion_mobile.services.library import book_key
    from glossarion_mobile.ui.library import progress_model as pm
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader import session as rs
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    detail: dict = {}
    # 1. Library scan (both shelves, card views) ------------------------------------------------
    snapshot = service.scan_blocking()
    _check(snapshot.ok, f"Library scan failed: {snapshot.error}")
    found_shelf, book = _find_book(snapshot, Path(output_dir))
    names = [b.get("name") for b in snapshot.all_books()]
    _check(book is not None, f"the Library did not list {Path(output_dir).name} (books: {names})")
    _check(found_shelf == shelf, f"{book.get('name')} is on the {found_shelf} shelf, expected {shelf}")
    done = int(book.get("completed_chapters", 0) or 0)
    card_total = int(book.get("total_chapters", 0) or 0)
    _check(done == completed, f"card counts {done}/{card_total}, expected {completed} completed")
    if total is not None:
        _check(card_total == total, f"card counts {done}/{card_total}, expected {total} chapters")
    else:
        _check(card_total >= completed, f"card total {card_total} < {completed} completed")
    if raw_path is not None:
        found_raw = book.get("raw_source_path") or service.raw_source(book)
        _check(found_raw and _norm(found_raw) == _norm(raw_path),
               f"raw source {found_raw!r}, expected {str(raw_path)!r}")
    view = snapshot.views.get(book_key(book))
    if shelf == "in_progress":
        expected_pill = (f"✨ Ready to compile ({done}/{card_total})" if done >= card_total > 0
                         else f"⏳ {done}/{card_total}")
        _check(view is not None and view.get("pill_text") == expected_pill,
               f"card pill {view and view.get('pill_text')!r}, expected {expected_pill!r}")
    else:
        _check(view is None, f"a completed card shows a pill: {view}")
    detail["library"] = {"shelf": found_shelf, "card": f"{done}/{card_total}", "state": book.get("translation_state"),
                         "pill": (view or {}).get("pill_text"), "books": len(names)}

    # 2. Book page (Book Details loader, metadata model) ------------------------------------------
    preview = service.load_details_blocking(book, "preview")
    full = service.load_details_blocking(book, "full")
    chapters_info = list(full.get("chapters_info") or [])
    _check(len(chapters_info) >= completed, f"Book page lists {len(chapters_info)} chapters (< {completed})")
    model = service.details_model(book, full)
    book_title = model.title()
    if title is not None:
        _check(book_title == title, f"Book page title {book_title!r}, expected {title!r}")
    detail["book_page"] = {"title": book_title, "chapters": len(chapters_info),
                           "preview_title": (preview.get("details") or {}).get("title")}

    # 3. Chapters tab (Progress Manager rows) -------------------------------------------------------
    progress = pm.load_progress_view(service, book, show_special=False, show_model_info=True)
    _check(progress.error is None, f"Chapters tab failed: {progress.error}")
    rows = [r for r in progress.rows if r.kind == "chapter"]
    counted = dict(Counter(r.status for r in rows))
    if statuses is not None:
        # Spine documents the Progress Manager lists as special files (nav, cover) show as skipped.
        wanted = {k: v for k, v in statuses.items() if v}
        compared = {k: v for k, v in counted.items() if k != "skipped" or "skipped" in wanted}
        _check(compared == wanted, f"Chapters tab statuses {counted}, expected {wanted}")
    detail["chapters_tab"] = {"rows": len(rows), "statuses": counted, "total_text": progress.total_text}

    # 4. Reader (shared open decision, session, mobile page) ------------------------------------------
    engine = rs.DocEngine(config=service.config_snapshot())
    plan = rs.plan_open(book, engine=engine)
    _check(not plan.error, f"Reader cannot open the book: {plan.error}")
    session = rs.ReaderSession(plan, engine=engine, cache_dir=str(Path(service.cache_dir()) / "reader"))
    builder = DocumentBuilder(session, lambda *args, **kwargs: "/selftest/img")
    try:
        session.load()
        _check(session.count >= completed, f"Reader has {session.count} chapters (< {completed})")
        translated = [i for i in range(session.count) if marker in _text(session.chapter_html(i, rm.TRANSLATED))]
        expected_marked = completed if translated_chapters is None else translated_chapters
        _check(len(translated) == expected_marked,
               f"{len(translated)} Reader chapters carry {marker!r}, expected {expected_marked}")
        leftover = [i for i in translated if _HANGUL.search(_text(session.chapter_html(i, rm.TRANSLATED)))]
        _check(not leftover, f"translated Reader chapters with Korean text left: {leftover}")
        first = translated[0]
        _check(session.has_alternate, f"the Reader has no Original side for {plan.mode} mode")
        # The segmented Original / Translated control (a dual book reloads the raw EPUB).
        session.set_flavor(rm.ORIGINAL)
        original = _text(session.chapter_html(first))
        session.set_flavor(rm.TRANSLATED)
        _check(_HANGUL.search(original) and marker not in original,
               "the Reader's Original side does not show the Korean source")
        themes = engine.themes() or [{}]
        built = builder.build(first, settings=rm.ReaderSettings(), layout=rm.LAYOUT_SINGLE,
                              theme=rm.theme_for(themes, rm.ReaderSettings()), doc_id="selftest",
                              event_url="/selftest/__ev")
        _check(marker in built.html, "the built Reader page lost the chapter text")
        _check(built.paged and built.has_page_bridge and "viewport-fit=cover" in built.html,
               "the built Reader page has no mobile shell / page bridge")
        detail["reader"] = {"mode": plan.mode, "chapters": session.count, "translated": len(translated),
                            "page_bytes": len(built.html.encode("utf-8"))}
    finally:
        builder.close()
        session.close()
    return detail


def check_selftest_library(work: Path, epub: Path, *, expect_title: Optional[str] = None) -> dict:
    """The ``smoke`` self-test: :func:`selftest_workspace` of ``epub`` checked by :func:`verify_book`."""
    work = Path(work)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    workspace = selftest_workspace(work, Path(epub))
    try:
        with isolated_library(work) as service:
            return verify_book(
                service,
                output_dir=workspace["output"],
                raw_path=workspace["raw"],
                shelf="in_progress",
                completed=workspace["completed"],
                total=workspace["documents"],
                statuses=workspace["statuses"],
                translated_chapters=workspace["translated_files"],
                title=expect_title,
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
