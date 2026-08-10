"""Compile translated response files in a PDF workspace into one PDF."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


LogCallback = Callable[[str], None]


_SOURCE_ALIGNMENT_CORRECTION_CACHE: dict[tuple, dict[int, dict[int, str]]] = {}
_SOURCE_ALIGNMENT_CORRECTION_LOCK = threading.Lock()


class PDFCompilationCancelled(RuntimeError):
    """Raised when compilation is cancelled between renderer phases."""


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)
    else:
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def _rapid_render_worker_count(job_count: int, requested: int | None = None) -> int:
    """Use the dedicated PDF worker setting for rapid rendering."""
    if job_count <= 1:
        return 1
    from pdf_fast_extractor import resolve_pdf_extraction_workers

    resolved = resolve_pdf_extraction_workers(requested)
    return max(1, min(job_count, resolved))


def build_bookmark_render_jobs(
    chapter_parts: list[str],
    chapter_orders: list[tuple[str, int, str]],
    worker_count: int,
) -> list[tuple[int, str, list[tuple[str, int, str]]]]:
    """Create exactly one independently scheduled render job per bookmark."""
    if not chapter_parts:
        return []
    if len(chapter_parts) != len(chapter_orders):
        raise ValueError("Bookmark render parts and order records do not match")
    del worker_count  # Concurrency limits execution, not bookmark boundaries.
    return [
        (job_index, chapter_part, [chapter_orders[job_index]])
        for job_index, chapter_part in enumerate(chapter_parts)
    ]


def _render_workspace_pdf_shard(job: tuple) -> dict:
    """Process-pool entry point for one bookmark-aware PDF render job."""
    (
        job_index,
        total_jobs,
        source,
        base_url,
        output_path,
        chapter_order,
        write_kwargs,
    ) = job
    started = time.perf_counter()
    from weasyprint import HTML as WeasyHTML

    document = WeasyHTML(string=source, base_url=base_url).render()
    anchor_pages = {}
    for source_filename, chapter_number, _title in chapter_order:
        local_page = 0
        anchor_name = f"chapter-{chapter_number}"
        for page_index, page in enumerate(document.pages):
            if anchor_name in page.anchors:
                local_page = page_index
                break
        anchor_pages[str(chapter_number)] = local_page
        anchor_pages[str(source_filename)] = local_page
    document.write_pdf(output_path, **dict(write_kwargs or {}))
    return {
        "index": job_index,
        "total": total_jobs,
        "path": output_path,
        "pages": len(document.pages),
        "chapters": len(chapter_order),
        "first_chapter": chapter_order[0][1] if chapter_order else None,
        "last_chapter": chapter_order[-1][1] if chapter_order else None,
        "anchor_pages": anchor_pages,
        "elapsed": time.perf_counter() - started,
        "bytes": os.path.getsize(output_path),
    }


def render_workspace_bookmarks_rapid(
    batch_jobs: list[tuple[int, str, list[tuple[str, int, str]]]],
    base_url: str,
    log_callback: LogCallback | None = None,
    *,
    max_workers: int | None = None,
    write_kwargs: dict | None = None,
) -> dict:
    """Render bookmark-aware shards in isolated processes and return paths."""
    jobs = list(batch_jobs or [])
    if not jobs:
        return {"results": [], "temp_dir": "", "workers": 0}

    worker_count = _rapid_render_worker_count(len(jobs), max_workers)
    total = len(jobs)
    started = time.perf_counter()
    completed = 0
    heartbeat_done = threading.Event()
    configured = os.environ.get("PDF_EXTRACTION_WORKERS", "auto")
    temp_dir = tempfile.mkdtemp(prefix="rapid_pdf_", dir=base_url)
    _log(
        log_callback,
        f"⚡ Rapid workspace renderer: {total} bookmark-aware job(s) on "
        f"{worker_count} process worker(s) "
        f"(PDF extraction setting={configured}, CPU cores={os.cpu_count() or 1})",
    )

    def heartbeat() -> None:
        try:
            interval = max(
                0.05,
                float(os.environ.get("PDF_COMPILE_HEARTBEAT_SECONDS", "3")),
            )
        except (TypeError, ValueError):
            interval = 3.0
        while not heartbeat_done.wait(interval):
            elapsed = time.perf_counter() - started
            _log(
                log_callback,
                f"⏳ Rapid workspace renderer heartbeat: {completed}/{total} "
                f"job(s) complete, {elapsed:.0f}s elapsed",
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="rapid-pdf-render-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    results: dict[int, Any] = {}
    try:
        process_jobs = []
        for job_index, source, chapter_order in jobs:
            output_path = os.path.join(
                temp_dir,
                f"bookmark_job_{job_index + 1:04d}.pdf",
            )
            first = chapter_order[0][1] if chapter_order else "?"
            last = chapter_order[-1][1] if chapter_order else "?"
            _log(
                log_callback,
                f"  ▶ Queued render job {job_index + 1}/{total}: "
                f"bookmark sections {first}-{last}, "
                f"{len(chapter_order)} section(s), {len(source):,} HTML characters",
            )
            process_jobs.append((
                job_index,
                total,
                source,
                base_url,
                output_path,
                chapter_order,
                dict(write_kwargs or {}),
            ))

        with ProcessPoolExecutor(
            max_workers=worker_count,
        ) as executor:
            futures = {
                executor.submit(_render_workspace_pdf_shard, job): job[0]
                for job in process_jobs
            }
            for future in as_completed(futures):
                job_index = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Rapid render job {job_index + 1}/{total} failed: {exc}"
                    ) from exc
                results[result["index"]] = result
                completed += 1
                _log(
                    log_callback,
                    f"  ✅ Render job {result['index'] + 1}/{total}: "
                    f"sections {result['first_chapter']}-{result['last_chapter']}, "
                    f"{result['pages']} page(s), "
                    f"{result['bytes'] / 1024 / 1024:.2f} MiB in "
                    f"{result['elapsed']:.1f}s "
                    f"({completed}/{total} complete)",
                )
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        heartbeat_done.set()
        heartbeat_thread.join(timeout=0.5)

    elapsed = time.perf_counter() - started
    _log(
        log_callback,
        f"⚡ Rapid workspace renderer finished {total} job(s) in "
        f"{elapsed:.1f}s; bookmark order preserved",
    )
    return {
        "results": [results[job_index] for job_index, _source, _order in jobs],
        "temp_dir": temp_dir,
        "workers": worker_count,
    }


def _natural_key(value: str) -> list:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value or ""))
    ]


def _numeric_order(value, fallback: int) -> tuple:
    try:
        return (0, float(value), fallback)
    except (TypeError, ValueError):
        return (1, fallback, fallback)


def _workspace_response_entries(folder: str) -> list[tuple[str, str]]:
    """Return existing translated response files as ``(path, title)``."""
    progress_path = os.path.join(folder, "translation_progress.json")
    entries: list[tuple[tuple, str, str]] = []
    seen: set[str] = set()
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        chapters = progress.get("chapters", {}) if isinstance(progress, dict) else {}
    except (OSError, ValueError, TypeError):
        chapters = {}

    if isinstance(chapters, dict):
        for index, (key, info) in enumerate(chapters.items()):
            if not isinstance(info, dict):
                continue
            output_name = str(info.get("output_file") or "").strip()
            if not output_name.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            output_path = (
                output_name
                if os.path.isabs(output_name)
                else os.path.join(folder, output_name)
            )
            if not os.path.isfile(output_path):
                continue
            normalized = os.path.normcase(os.path.normpath(output_path))
            if normalized in seen:
                continue
            seen.add(normalized)
            title = str(
                info.get("pdf_toc_title_translated")
                or info.get("translated_title")
                or info.get("pdf_section_title_translated")
                or info.get("pdf_section_title")
                or info.get("pdf_toc_title")
                or info.get("title")
                or f"Section {info.get('actual_num') or key}"
            ).strip()
            entries.append((
                _numeric_order(info.get("actual_num", key), index),
                output_path,
                title,
            ))

    if entries:
        entries.sort(key=lambda item: item[0])
        return [(path, title) for _order, path, title in entries]

    fallback = []
    try:
        for entry in os.scandir(folder):
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name.lower()
            if name.startswith("response_") and name.endswith(
                (".html", ".xhtml", ".htm")
            ):
                fallback.append(entry.path)
    except (OSError, PermissionError):
        pass
    fallback.sort(key=lambda path: _natural_key(os.path.basename(path)))
    return [
        (path, f"Section {index}")
        for index, path in enumerate(fallback, 1)
    ]


def _load_translation_cache_by_source(path: str) -> dict[str, str]:
    """Load successful cache entries keyed by their exact source text."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        from translate_headers_standalone import load_translations_from_file

        originals, translated, _outputs = load_translations_from_file(
            path, log_callback=lambda _message: None
        )
    except Exception:
        return {}
    result = {}
    for number, original in originals.items():
        source = str(original or "").strip()
        value = str(translated.get(number) or "").strip()
        if source and value:
            result[source] = value
    return result


def load_pdf_workspace_artifact_chapters(output_dir: str) -> list[dict]:
    """Reconstruct lightweight PDF chapters for standalone/compile phases."""
    progress_path = os.path.join(output_dir, "translation_progress.json")
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []
    items = list((progress.get("chapters") or {}).items())
    items.sort(
        key=lambda item: _numeric_order(
            item[1].get("actual_num") if isinstance(item[1], dict) else None,
            10 ** 9,
        )
    )
    chapters = []
    for _key, entry in items:
        if not isinstance(entry, dict) or not entry.get("pdf_toc_section"):
            continue
        number = entry.get("actual_num")
        try:
            number_token = str(int(float(number)))
        except (TypeError, ValueError):
            number_token = str(number)
        raw_path = os.path.join(
            output_dir, "word_count", f"pdf_section_{number_token}.html"
        )
        try:
            with open(raw_path, "r", encoding="utf-8", errors="replace") as handle:
                raw_body = handle.read()
        except OSError:
            raw_body = ""
        chapters.append({
            "num": number,
            "title": entry.get("pdf_toc_title") or entry.get("title"),
            "body": raw_body,
            "pdf_toc_section": True,
            "pdf_toc_title": entry.get("pdf_toc_title"),
            "pdf_section_title": entry.get("pdf_section_title"),
            "pdf_section_id": entry.get("pdf_section_id"),
        })
    return chapters


def _pdf_artifact_records(
    chapters: list[dict],
    output_dir: str,
    progress: dict,
) -> tuple[list[dict], list[dict]]:
    """Build bookmark and h1-h6 records in stable PDF reading order."""
    progress_chapters = (
        progress.get("chapters", {}) if isinstance(progress, dict) else {}
    )
    progress_by_section = {}
    progress_by_number = {}
    if isinstance(progress_chapters, dict):
        for progress_key, entry in progress_chapters.items():
            if not isinstance(entry, dict):
                continue
            section_id = str(entry.get("pdf_section_id") or "").strip()
            if section_id:
                progress_by_section[section_id] = (progress_key, entry)
            number = entry.get("actual_num", entry.get("chapter_num"))
            if number is not None:
                progress_by_number[str(number)] = (progress_key, entry)

    bookmark_records = []
    header_records = []
    seen_sections = set()
    ordered = sorted(
        enumerate(chapters or []),
        key=lambda item: _numeric_order(item[1].get("num"), item[0]),
    )
    for fallback_index, chapter in ordered:
        if not isinstance(chapter, dict):
            continue
        section_id = str(chapter.get("pdf_section_id") or "").strip()
        section_key = section_id or str(chapter.get("num", fallback_index + 1))
        # Token splitting can produce several chunks for one bookmark. TOC and
        # header artifacts remain one workload per original bookmark section.
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)

        progress_pair = progress_by_section.get(section_id)
        if not progress_pair:
            progress_pair = progress_by_number.get(str(chapter.get("num")))
        progress_key, progress_entry = progress_pair or (None, {})
        output_file = str(progress_entry.get("output_file") or "").strip()
        output_path = (
            output_file
            if os.path.isabs(output_file)
            else os.path.join(output_dir, output_file)
        ) if output_file else ""

        source_title = str(
            chapter.get("pdf_section_title")
            or chapter.get("pdf_toc_title")
            or chapter.get("title")
            or progress_entry.get("pdf_section_title")
            or progress_entry.get("pdf_toc_title")
            or progress_entry.get("title")
            or ""
        ).strip()
        is_bookmark = bool(
            chapter.get("pdf_toc_section")
            or progress_entry.get("pdf_toc_section")
            or section_id
        )
        if is_bookmark and source_title:
            bookmark_records.append({
                "source": source_title,
                "chapter": chapter,
                "progress_key": progress_key,
                "progress_entry": progress_entry,
                "output_file": output_file,
            })

        # Prefer the immutable extraction copy. The in-memory chapter body can
        # be altered by image cleanup, request merging, or multipass handling
        # before this post-translation phase runs.
        number = chapter.get("num", fallback_index + 1)
        try:
            number_token = str(int(float(number)))
        except (TypeError, ValueError):
            number_token = str(number)
        raw_path = os.path.join(
            output_dir, "word_count", f"pdf_section_{number_token}.html"
        )
        try:
            with open(raw_path, "r", encoding="utf-8", errors="replace") as handle:
                raw_html = handle.read()
        except OSError:
            raw_html = str(chapter.get("body") or chapter.get("content") or "")
        if not raw_html:
            continue
        raw_soup = BeautifulSoup(raw_html, "html.parser")
        for tag_index, heading in enumerate(
            raw_soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ):
            source = heading.get_text(" ", strip=True)
            if not source:
                continue
            header_records.append({
                "source": source,
                "tag_index": tag_index,
                "tag_name": heading.name,
                "output_file": output_file,
                "output_path": output_path,
                "section_key": section_key,
            })
    return bookmark_records, header_records


def _translate_pdf_artifact_workload(
    *,
    kind: str,
    records: list[dict],
    output_dir: str,
    translator,
    batch_size: int,
    log_callback: LogCallback | None,
    save_cache: bool = True,
) -> dict[int, str]:
    """Translate one PDF TOC/header workload with cache and cross-cache reuse."""
    if not records:
        return {}
    filename = "TOC.txt" if kind == "toc" else "translated_headers.txt"
    other_filename = (
        "translated_headers.txt" if kind == "toc" else "TOC.txt"
    )
    cache_path = os.path.join(output_dir, filename)
    own_cache = _load_translation_cache_by_source(cache_path)
    other_cache = _load_translation_cache_by_source(
        os.path.join(output_dir, other_filename)
    )
    originals = {
        index: str(record.get("source") or "").strip()
        for index, record in enumerate(records, 1)
    }
    translated = {}
    remaining = {}
    for number, source in originals.items():
        cached = own_cache.get(source) or other_cache.get(source)
        if cached:
            translated[number] = cached
        else:
            remaining[number] = source

    reused = len(translated)
    if reused:
        _log(
            log_callback,
            f"♻️ PDF {kind}: reused {reused}/{len(originals)} cached translation(s)",
        )

    # The TOC duplicate option is a request-level optimization. Preserve every
    # bookmark record in TOC.txt while sending repeated source labels only once.
    duplicate_groups = {}
    request_entries = dict(remaining)
    if kind == "toc" and os.getenv(
        "SKIP_DUPLICATE_TOC_TRANSLATION", "0"
    ) == "1":
        request_entries = {}
        first_by_source = {}
        for number, source in remaining.items():
            if source in first_by_source:
                duplicate_groups.setdefault(first_by_source[source], []).append(number)
            else:
                first_by_source[source] = number
                request_entries[number] = source

    if request_entries:
        api_translated = translator.translate_headers_batch(
            request_entries,
            batch_size=batch_size,
            translation_type=kind,
        ) or {}
        translated.update(api_translated)
        for first_number, duplicate_numbers in duplicate_groups.items():
            value = translated.get(first_number)
            if value:
                for duplicate_number in duplicate_numbers:
                    translated[duplicate_number] = value

    current_titles = {
        index: {"filename": record.get("output_file") or ""}
        for index, record in enumerate(records, 1)
    }
    if save_cache:
        translator._save_translations_to_file(
            originals, translated, cache_path, current_titles
        )
    try:
        from translation_artifacts import update_translation_artifact_progress

        actual_model = getattr(translator, "_last_batch_actual_model", None)
        if not actual_model:
            getter = getattr(
                translator.client, "get_last_actual_request_model", None
            )
            if callable(getter):
                try:
                    actual_model = getter()
                except Exception:
                    actual_model = None
        if save_cache:
            update_translation_artifact_progress(
                output_dir,
                kind,
                "completed" if len(translated) == len(originals) else "failed",
                model_name=(
                    actual_model or getattr(translator.client, "model", None)
                ),
                error_message=(
                    None
                    if len(translated) == len(originals)
                    else f"{len(originals) - len(translated)} entry translation(s) failed"
                ),
            )
    except Exception:
        pass
    return translated


def translate_pdf_workspace_artifacts(
    chapters: list[dict],
    output_dir: str,
    api_client,
    *,
    progress_manager=None,
    log_callback: LogCallback | None = None,
    stop_callback: Callable[[], bool] | None = None,
    config: dict[str, Any] | None = None,
    use_toc: bool | None = None,
    use_headers: bool | None = None,
    update_html_headers: bool | None = None,
    save_header_translations: bool | None = None,
) -> dict[str, int]:
    """Batch-translate PDF bookmarks and extracted HTML h1-h6 elements.

    This is the PDF counterpart of the EPUB ``toc.ncx`` and chapter-header
    phases. It uses the same BatchHeaderTranslator, prompt settings, batch
    limits, TOC.txt/translated_headers.txt cache formats, and progress rows.
    """
    if use_toc is None:
        use_toc = os.getenv("USE_TOC_NCX", "1") == "1"
    if use_headers is None:
        use_headers = os.getenv("BATCH_TRANSLATE_HEADERS", "0") == "1"
    if update_html_headers is None:
        update_html_headers = os.getenv("UPDATE_HTML_HEADERS", "1") == "1"
    if save_header_translations is None:
        save_header_translations = (
            os.getenv("SAVE_HEADER_TRANSLATIONS", "1") == "1"
        )
    if not use_toc and not use_headers:
        return {"toc": 0, "headers": 0}
    if stop_callback and stop_callback():
        raise PDFCompilationCancelled("PDF artifact translation stopped by user")

    if progress_manager is not None:
        progress = progress_manager.prog
        try:
            progress_manager.save()
        except Exception:
            pass
    else:
        try:
            with open(
                os.path.join(output_dir, "translation_progress.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                progress = json.load(handle)
        except (OSError, ValueError, TypeError):
            progress = {"chapters": {}}

    bookmark_records, header_records = _pdf_artifact_records(
        chapters, output_dir, progress
    )
    if not bookmark_records and not header_records:
        _log(log_callback, "ℹ️ PDF contains no bookmark/header artifacts to translate")
        return {"toc": 0, "headers": 0}

    from metadata_batch_translator import BatchHeaderTranslator

    translator_config = dict(config or {})
    translator_config["output_dir"] = os.path.abspath(output_dir)
    translator = BatchHeaderTranslator(
        api_client,
        translator_config,
        stop_check_fn=stop_callback,
    )
    result = {"toc": 0, "headers": 0}

    toc_translated = {}
    if use_toc and bookmark_records:
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF bookmark translation stopped by user")
        _log(
            log_callback,
            f"📑 Translating {len(bookmark_records)} PDF bookmark title(s) in batches…",
        )
        try:
            toc_batch_size = int(os.getenv("TOC_NCX_PER_BATCH", "-1"))
        except (TypeError, ValueError):
            toc_batch_size = -1
        toc_translated = _translate_pdf_artifact_workload(
            kind="toc",
            records=bookmark_records,
            output_dir=output_dir,
            translator=translator,
            batch_size=toc_batch_size,
            log_callback=log_callback,
        )
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF bookmark translation stopped by user")
        for index, record in enumerate(bookmark_records, 1):
            translated_title = toc_translated.get(index)
            if not translated_title:
                continue
            chapter = record.get("chapter")
            if isinstance(chapter, dict):
                chapter["pdf_toc_title_original"] = record["source"]
                chapter["pdf_toc_title_translated"] = translated_title
                chapter["translated_title"] = translated_title
                chapter["title"] = translated_title
            entry = record.get("progress_entry")
            if isinstance(entry, dict):
                entry["pdf_toc_title_original"] = record["source"]
                entry["pdf_toc_title_translated"] = translated_title
                entry["translated_title"] = translated_title
        result["toc"] = len(toc_translated)

    header_translated = {}
    if use_headers and header_records:
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF header translation stopped by user")
        _log(
            log_callback,
            f"🔤 Translating {len(header_records)} PDF HTML header(s) in batches…",
        )
        try:
            header_batch_size = int(os.getenv("HEADERS_PER_BATCH", "-1"))
        except (TypeError, ValueError):
            header_batch_size = -1
        header_translated = _translate_pdf_artifact_workload(
            kind="headers",
            records=header_records,
            output_dir=output_dir,
            translator=translator,
            batch_size=header_batch_size,
            log_callback=log_callback,
            save_cache=bool(save_header_translations),
        )
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF header translation stopped by user")
        records_by_file = {}
        if update_html_headers:
            for index, record in enumerate(header_records, 1):
                if index in header_translated and record.get("output_path"):
                    records_by_file.setdefault(record["output_path"], []).append(
                        (record, header_translated[index])
                    )
        updated_files = 0
        for path, replacements in records_by_file.items():
            if stop_callback and stop_callback():
                raise PDFCompilationCancelled("PDF header translation stopped by user")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    soup = BeautifulSoup(handle.read(), "html.parser")
            except OSError:
                continue
            headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            changed = False
            for record, translated_title in replacements:
                tag_index = int(record.get("tag_index") or 0)
                if tag_index < len(headings):
                    headings[tag_index].clear()
                    headings[tag_index].append(translated_title)
                    changed = True
                elif tag_index == 0:
                    container = soup.body if soup.body is not None else soup
                    new_heading = soup.new_tag(record.get("tag_name") or "h1")
                    new_heading.string = translated_title
                    container.insert(0, new_heading)
                    changed = True
            if changed:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(str(soup))
                updated_files += 1
        _log(
            log_callback,
            f"✅ Applied translated PDF headers to {updated_files} HTML file(s)",
        )
        result["headers"] = len(header_translated)

    # BatchHeaderTranslator updates artifact rows atomically on disk. Merge
    # those rows back into the live ProgressManager before its next save.
    if progress_manager is not None:
        try:
            with open(
                os.path.join(output_dir, "translation_progress.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                disk_progress = json.load(handle)
            disk_chapters = disk_progress.get("chapters", {})
            for key, entry in disk_chapters.items():
                if str(key).startswith("__translation_artifact__"):
                    progress_manager.prog.setdefault("chapters", {})[key] = entry
        except (OSError, ValueError, TypeError):
            pass
        progress_manager.save()
    else:
        # Standalone header/TOC runs do not own a live ProgressManager. Merge
        # atomically-written artifact rows into the document whose PDF entries
        # received translated bookmark fields, then persist the combined view.
        progress_path = os.path.join(output_dir, "translation_progress.json")
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                disk_progress = json.load(handle)
            disk_chapters = disk_progress.get("chapters", {})
            progress_chapters = progress.setdefault("chapters", {})
            for key, entry in disk_chapters.items():
                if str(key).startswith("__translation_artifact__"):
                    progress_chapters[key] = entry
            temp_path = f"{progress_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(progress, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, progress_path)
        except (OSError, ValueError, TypeError):
            pass
    return result


def _fragment_body(content: str) -> str:
    soup = BeautifulSoup(content or "", "html.parser")
    container = soup.body if soup.body is not None else soup
    return "".join(str(child) for child in container.contents)


def _paragraph_alignment_value(paragraph, fallback="left") -> str:
    alignment = str(
        paragraph.get("data-pdf-source-alignment") or ""
    ).strip().lower()
    if alignment in {"left", "center", "right", "justify"}:
        return alignment
    for css_class in paragraph.get("class", []):
        css_class = str(css_class)
        if css_class.startswith("pdf-align-"):
            candidate = css_class[len("pdf-align-"):].strip().lower()
            if candidate in {"left", "center", "right", "justify"}:
                return candidate
    for declaration in str(paragraph.get("style") or "").split(";"):
        name, separator, value = declaration.partition(":")
        if separator and name.strip().casefold() == "text-align":
            candidate = value.strip().lower().replace("!important", "").strip()
            if candidate in {"left", "center", "right", "justify"}:
                return candidate
    return fallback


def _set_paragraph_source_alignment(paragraph, alignment: str) -> None:
    alignment = (
        alignment if alignment in {"left", "center", "right", "justify"}
        else "left"
    )
    paragraph["data-pdf-source-alignment"] = alignment
    classes = [
        css_class
        for css_class in paragraph.get("class", [])
        if not str(css_class).startswith("pdf-align-")
    ]
    classes.append(f"pdf-align-{alignment}")
    paragraph["class"] = classes
    declarations = []
    for declaration in str(paragraph.get("style") or "").split(";"):
        declaration = declaration.strip()
        name = declaration.partition(":")[0].strip().casefold()
        if declaration and name != "text-align":
            declarations.append(declaration)
    declarations.append(f"text-align:{alignment}")
    paragraph["style"] = ";".join(declarations)


def _source_alignment_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split())


def _cached_source_alignment_corrections(folder: str) -> dict[int, dict[int, str]]:
    """Recheck legacy cached ``center`` values against source line geometry.

    Fast Semantic version 5 could mistake a normal wrapped paragraph for
    centered text when its lines happened to have similar centers.  Opening
    every page again would defeat the extraction cache, so inspect only cache
    pages containing a centered paragraph and only read those source pages.
    """
    manifest_path = os.path.join(
        folder,
        ".pdf_extraction_cache",
        "manifest_fast_semantic.json",
    )
    cache_root = os.path.join(
        folder,
        ".pdf_extraction_cache",
        "pages",
        "fast_semantic",
    )
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(manifest, dict) or not os.path.isdir(cache_root):
        return {}

    source_info = manifest.get("source")
    source_info = source_info if isinstance(source_info, dict) else {}
    source_pdf = str(source_info.get("path") or "").strip()
    if not os.path.isfile(source_pdf):
        try:
            from output_workspace import read_workspace_source_path

            source_pdf = read_workspace_source_path(folder)
        except Exception:
            source_pdf = ""
    if not source_pdf.lower().endswith(".pdf") or not os.path.isfile(source_pdf):
        return {}

    try:
        manifest_mtime = os.stat(manifest_path).st_mtime_ns
        source_mtime = os.stat(source_pdf).st_mtime_ns
    except OSError:
        return {}
    cache_key = (
        os.path.normcase(os.path.abspath(folder)),
        os.path.normcase(os.path.abspath(source_pdf)),
        source_mtime,
        str(source_info.get("sha256") or ""),
        manifest_mtime,
    )
    with _SOURCE_ALIGNMENT_CORRECTION_LOCK:
        cached = _SOURCE_ALIGNMENT_CORRECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    affected_pages: dict[int, list[tuple[int, str, int]]] = {}
    try:
        cache_names = sorted(
            name
            for name in os.listdir(cache_root)
            if name.lower().endswith(".json")
        )
    except OSError:
        return {}
    for cache_name in cache_names:
        try:
            with open(
                os.path.join(cache_root, cache_name),
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            page_number = int(payload.get("page_number"))
            source_soup = BeautifulSoup(
                str(payload.get("html") or ""),
                "html.parser",
            )
        except (OSError, ValueError, TypeError):
            continue
        occurrences: dict[str, int] = {}
        affected = []
        for index, paragraph in enumerate(
            source_soup.select(".pdf-fast-semantic-page p")
        ):
            text_key = _source_alignment_text_key(
                paragraph.get_text(" ", strip=True)
            )
            occurrence = occurrences.get(text_key, 0)
            occurrences[text_key] = occurrence + 1
            if _paragraph_alignment_value(paragraph) == "center" and text_key:
                affected.append((index, text_key, occurrence))
        if affected:
            affected_pages[page_number] = affected

    corrections: dict[int, dict[int, str]] = {}
    if affected_pages:
        try:
            import fitz
            from pdf_fast_extractor import (
                _layout_paragraph_alignment,
                _semantic_layout_geometry,
            )

            block_flags = int(getattr(fitz, "TEXTFLAGS_BLOCKS", 195)) & ~int(
                getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
            )
            dict_flags = int(getattr(fitz, "TEXTFLAGS_DICT", 199)) & ~int(
                getattr(fitz, "TEXT_PRESERVE_IMAGES", 4)
            )
            document = fitz.open(source_pdf)
            try:
                for page_number, affected in affected_pages.items():
                    if page_number < 1 or page_number > document.page_count:
                        continue
                    page = document[page_number - 1]
                    try:
                        blocks = page.get_text(
                            "blocks",
                            sort=True,
                            flags=block_flags,
                        ) or []
                    except TypeError:
                        blocks = page.get_text("blocks", sort=True) or []
                    layout_by_number, column_bounds = _semantic_layout_geometry(
                        page,
                        dict_flags,
                    )
                    candidates: dict[str, list[str]] = {}
                    for block in blocks:
                        if len(block) < 7 or int(block[6]) != 0:
                            continue
                        text_value = " ".join(str(block[4] or "").split())
                        text_key = _source_alignment_text_key(text_value)
                        if not text_key:
                            continue
                        alignment = _layout_paragraph_alignment(
                            layout_by_number.get(int(block[5])),
                            float(page.rect.width),
                            column_bounds,
                            text_value,
                        )
                        candidates.setdefault(text_key, []).append(alignment)
                    for index, text_key, occurrence in affected:
                        matches = candidates.get(text_key) or []
                        if occurrence >= len(matches):
                            continue
                        alignment = matches[occurrence]
                        if alignment in {"left", "center", "right", "justify"}:
                            corrections.setdefault(page_number, {})[index] = alignment
            finally:
                document.close()
        except Exception:
            # Source-format restoration remains best effort.  A missing PDF
            # dependency or unreadable source must not block compilation.
            corrections = {}

    with _SOURCE_ALIGNMENT_CORRECTION_LOCK:
        stale_keys = [
            key
            for key in _SOURCE_ALIGNMENT_CORRECTION_CACHE
            if key[0] == cache_key[0] and key != cache_key
        ]
        for stale_key in stale_keys:
            _SOURCE_ALIGNMENT_CORRECTION_CACHE.pop(stale_key, None)
        _SOURCE_ALIGNMENT_CORRECTION_CACHE[cache_key] = corrections
    return corrections


def restore_pdf_source_paragraph_alignment(content: str, folder: str) -> str:
    """Restore source alignment from cached PDF pages after model translation.

    Translation responses are not authoritative formatting sources: a model can
    remove the extraction classes or rewrite many ``text-align`` declarations.
    Each fast-semantic article retains its PDF page number, so copy the original
    alignment sequence from that page's extraction artifact before compilation.
    """
    soup = BeautifulSoup(content or "", "html.parser")
    cache_root = os.path.join(
        folder,
        ".pdf_extraction_cache",
        "pages",
        "fast_semantic",
    )
    if not os.path.isdir(cache_root):
        return str(content or "")

    page_cache = {}
    alignment_corrections = None
    changed = False
    for container in soup.select(".pdf-fast-semantic-page[data-pdf-page]"):
        try:
            page_number = int(container.get("data-pdf-page"))
        except (TypeError, ValueError):
            continue
        if page_number not in page_cache:
            cache_path = os.path.join(
                cache_root,
                f"page_{page_number:06d}.json",
            )
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    page_payload = json.load(handle)
                source_soup = BeautifulSoup(
                    str(page_payload.get("html") or ""),
                    "html.parser",
                )
                page_cache[page_number] = [
                    _paragraph_alignment_value(paragraph)
                    for paragraph in source_soup.select(
                        ".pdf-fast-semantic-page p"
                    )
                ]
            except (OSError, ValueError, TypeError):
                page_cache[page_number] = None
        source_alignments = page_cache.get(page_number)
        if source_alignments is None:
            continue
        if "center" in source_alignments:
            if alignment_corrections is None:
                alignment_corrections = _cached_source_alignment_corrections(folder)
            for index, corrected in (
                alignment_corrections.get(page_number, {}).items()
            ):
                if index < len(source_alignments):
                    source_alignments[index] = corrected
        translated_paragraphs = container.find_all("p")
        for index, paragraph in enumerate(translated_paragraphs):
            # If the model inserted an extra paragraph, default it to the
            # ordinary source flow instead of trusting a hallucinated center.
            alignment = (
                source_alignments[index]
                if index < len(source_alignments)
                else "left"
            )
            _set_paragraph_source_alignment(paragraph, alignment)
            changed = True
    return str(soup) if changed else str(content or "")


def normalize_fast_semantic_paragraph_alignment(content: str) -> str:
    """Preserve detected source formatting or apply the configured override."""
    from pdf_fast_extractor import (
        pdf_rtl_paragraph_layout_enabled,
        resolve_pdf_paragraph_alignment,
    )

    soup = BeautifulSoup(content or "", "html.parser")
    changed = False
    for paragraph in soup.select(".pdf-fast-semantic-page p"):
        style = str(paragraph.get("style") or "")
        source_alignment = _paragraph_alignment_value(paragraph)
        alignment = resolve_pdf_paragraph_alignment(
            source_alignment,
            paragraph.get_text(" ", strip=True),
        )
        classes = [
            css_class
            for css_class in paragraph.get("class", [])
            if not str(css_class).startswith("pdf-align-")
        ]
        classes.append(f"pdf-align-{alignment}")
        paragraph["class"] = classes

        declarations = []
        for declaration in style.split(";"):
            declaration = declaration.strip()
            name = declaration.partition(":")[0].strip().casefold()
            if not declaration or name == "text-align":
                continue
            declarations.append(declaration)
        declarations.append(f"text-align:{alignment}")
        paragraph["style"] = ";".join(declarations)
        changed = True

    rtl_enabled = pdf_rtl_paragraph_layout_enabled()
    for container in soup.select(
        ".pdf-fast-semantic-page, .pdf-fast-layout-page"
    ):
        classes = [str(css_class) for css_class in container.get("class", [])]
        marker_enabled = (
            str(container.get("data-pdf-rtl-layout") or "").lower() == "true"
        )
        if rtl_enabled:
            if "pdf-rtl-layout" not in classes:
                classes.append("pdf-rtl-layout")
            container["class"] = classes
            container["dir"] = "rtl"
            container["data-pdf-rtl-layout"] = "true"
            changed = True
        elif marker_enabled:
            container["class"] = [
                css_class for css_class in classes
                if css_class != "pdf-rtl-layout"
            ]
            container.attrs.pop("dir", None)
            container.attrs.pop("data-pdf-rtl-layout", None)
            changed = True
    return str(soup) if changed else str(content or "")


def normalize_fast_semantic_heading_alignment(
    content: str,
    alignment: str | None,
) -> str:
    """Restore or override the alignment of a translated section heading."""
    from pdf_fast_extractor import (
        normalize_pdf_header_alignment,
        resolve_pdf_header_alignment,
    )

    soup = BeautifulSoup(content or "", "html.parser")
    heading = soup.find("h1")
    if heading is None:
        return str(content or "")

    source_alignment = str(alignment or "").strip().lower()
    if source_alignment not in {"left", "center", "right"}:
        source_alignment = str(
            heading.get("data-pdf-source-alignment") or ""
        ).strip().lower()
    if source_alignment not in {"left", "center", "right"}:
        style_match = re.search(
            r"(?:^|;)\s*text-align\s*:\s*(left|center|right)\b",
            str(heading.get("style") or ""),
            re.IGNORECASE,
        )
        source_alignment = style_match.group(1).lower() if style_match else "left"

    override = normalize_pdf_header_alignment()
    resolved = resolve_pdf_header_alignment(
        source_alignment,
        alignment_override=override,
    )

    declarations = []
    for declaration in str(heading.get("style") or "").split(";"):
        declaration = declaration.strip()
        name = declaration.partition(":")[0].strip().casefold()
        if declaration and name != "text-align":
            declarations.append(declaration)
    declarations.append(f"text-align:{resolved}")
    heading["style"] = ";".join(declarations)
    heading["data-pdf-source-alignment"] = source_alignment
    return str(soup)


def _workspace_source_heading_alignments(folder: str) -> dict[str, str]:
    """Map response basenames to heading alignment measured from the raw PDF."""
    progress_path = os.path.join(folder, "translation_progress.json")
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        chapters = progress.get("chapters", {})
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(chapters, dict):
        return {}

    try:
        from output_workspace import read_workspace_source_path

        source_pdf = read_workspace_source_path(folder)
        if not source_pdf.lower().endswith(".pdf") or not os.path.isfile(source_pdf):
            return {}
        import fitz
        from pdf_fast_extractor import _semantic_title_key, _text_alignment

        document = fitz.open(source_pdf)
    except Exception:
        return {}

    alignments: dict[str, str] = {}
    try:
        for info in chapters.values():
            if not isinstance(info, dict) or not info.get("pdf_toc_section"):
                continue
            output_name = os.path.basename(str(info.get("output_file") or ""))
            title = str(
                info.get("pdf_toc_title")
                or info.get("pdf_section_title")
                or info.get("title")
                or ""
            ).strip()
            try:
                page_number = int(info.get("pdf_start_page") or 0)
            except (TypeError, ValueError):
                page_number = 0
            if not output_name or not title or not (1 <= page_number <= len(document)):
                continue

            page = document[page_number - 1]
            title_key = _semantic_title_key(title)
            for block in page.get_text("blocks", sort=True) or []:
                if len(block) < 5:
                    continue
                block_key = _semantic_title_key(block[4])
                if block_key != title_key and title_key not in block_key:
                    continue
                alignments[output_name.casefold()] = _text_alignment(
                    block[:4],
                    float(page.rect.width),
                    is_heading=True,
                )
                break
    finally:
        document.close()
    return alignments


def normalize_pdf_workspace_translated_html(
    content: str,
    folder: str,
    heading_alignment: str | None = None,
) -> str:
    """Apply source-authoritative PDF paragraph and heading formatting."""
    restored = restore_pdf_source_paragraph_alignment(content, folder)
    normalized = normalize_fast_semantic_paragraph_alignment(restored)
    return normalize_fast_semantic_heading_alignment(
        normalized,
        heading_alignment,
    )


def _write_response_html_if_changed(path: str, original: str, content: str) -> bool:
    """Atomically persist repaired response formatting without touching text."""
    if content == original:
        return False
    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass
    return True


def _image_reference_basename(src: str) -> str:
    try:
        path = unquote(urlsplit(str(src or "")).path)
    except Exception:
        path = str(src or "")
    return os.path.basename(path.replace("\\", "/"))


def _legacy_page_image_reference(filename: str):
    match = re.fullmatch(
        r"page_(\d+)_img_(\d+)\.[a-z0-9]+",
        str(filename or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _valid_cached_page_images(folder: str, page_number: int) -> list[dict]:
    cache_root = os.path.join(folder, ".pdf_extraction_cache")
    candidates = [
        os.path.join(cache_root, "pages", mode, f"page_{page_number:06d}.json")
        for mode in ("fast_semantic", "fast_layout")
    ]
    targeted_dir = os.path.join(cache_root, "targeted_images")
    if os.path.isdir(targeted_dir):
        candidates.extend(
            os.path.join(targeted_dir, name)
            for name in os.listdir(targeted_dir)
            if name.endswith(f"_page_{page_number:06d}.json")
        )
    images_dir = os.path.join(folder, "images")
    for cache_path in candidates:
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, list):
            continue
        valid = []
        for item in images:
            if not isinstance(item, dict):
                continue
            filename = os.path.basename(str(item.get("filename") or ""))
            if filename and os.path.isfile(os.path.join(images_dir, filename)):
                normalized = dict(item)
                normalized["filename"] = filename
                valid.append(normalized)
        if valid:
            return valid
    return []


def _legacy_image_slots(images: list[dict]) -> list[dict]:
    """Collapse repeated placements to legacy one-slot-per-image-resource order."""
    slots = []
    seen = set()
    for position, image in enumerate(images or []):
        resource_index = image.get("resource_index")
        xref = int(image.get("xref") or 0)
        key = (
            "resource",
            int(resource_index),
        ) if resource_index is not None else (
            "xref",
            xref,
        ) if xref else (
            "placement",
            position,
        )
        if key in seen:
            continue
        seen.add(key)
        slots.append(image)
    return slots


def _repair_pdf_image_references(
    folder: str,
    contents: list[str],
    *,
    log_callback: LogCallback | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> tuple[list[str], dict]:
    """Resolve legacy page image names to content-addressed fast assets."""

    soups = [BeautifulSoup(content or "", "html.parser") for content in contents]
    referenced_nodes = []
    for soup in soups:
        for node in soup.find_all("img"):
            src = str(node.get("src") or "").strip()
            if not src or src.startswith(("data:", "http://", "https://")):
                continue
            referenced_nodes.append((node, src, _image_reference_basename(src)))

    unique_names = {name for _node, _src, name in referenced_nodes if name}
    _log(
        log_callback,
        f"🖼️ PDF image preflight: {len(referenced_nodes)} reference(s), "
        f"{len(unique_names)} unique file(s)",
    )
    images_dir = os.path.join(folder, "images")
    os.makedirs(images_dir, exist_ok=True)
    alias_map: dict[str, str] = {}
    required_page_slots = {}
    already_present = set()
    rename_map = {}
    try:
        with open(os.path.join(folder, "image_rename_map.json"), "r", encoding="utf-8") as handle:
            loaded_rename_map = json.load(handle)
        if isinstance(loaded_rename_map, dict):
            rename_map = {
                os.path.basename(str(old_name or "")): os.path.basename(str(new_name or ""))
                for old_name, new_name in loaded_rename_map.items()
                if old_name and new_name
            }
    except (OSError, ValueError, TypeError):
        rename_map = {}
    renamed_from = {new_name: old_name for old_name, new_name in rename_map.items()}
    for name in unique_names:
        if os.path.isfile(os.path.join(images_dir, name)):
            already_present.add(name)
            continue
        page_slot = _legacy_page_image_reference(name)
        if not page_slot and name in renamed_from:
            page_slot = _legacy_page_image_reference(renamed_from[name])
        if page_slot:
            required_page_slots[name] = page_slot

    required_pages = sorted({page for page, _slot in required_page_slots.values()})
    page_images = {
        page: _valid_cached_page_images(folder, page)
        for page in required_pages
    }
    missing_pages = [page for page in required_pages if not page_images.get(page)]
    recovered_pages = {}
    if missing_pages:
        try:
            from output_workspace import read_workspace_source_path
            from pdf_fast_extractor import ensure_pdf_page_images

            source_pdf = read_workspace_source_path(folder)
            if source_pdf.lower().endswith(".pdf") and os.path.isfile(source_pdf):
                _log(
                    log_callback,
                    f"🖼️ Recovering images from {len(missing_pages)} specifically "
                    "referenced PDF page(s); unrelated pages will not be scanned",
                )
                recovered_pages = ensure_pdf_page_images(
                    source_pdf,
                    folder,
                    missing_pages,
                    stop_callback=stop_callback,
                    progress_callback=lambda message: _log(log_callback, message),
                )
        except Exception as exc:
            if exc.__class__.__name__ == "PDFExtractionCancelled":
                raise PDFCompilationCancelled(
                    "PDF compilation stopped by user"
                ) from exc
            _log(log_callback, f"⚠️ Targeted PDF image recovery failed: {exc}")
    page_images.update(recovered_pages)

    for old_name, (page_number, image_number) in required_page_slots.items():
        images = _legacy_image_slots(page_images.get(page_number) or [])
        if 1 <= image_number <= len(images):
            alias_map[old_name] = str(images[image_number - 1].get("filename") or "")

    # Section-level renames were written as old page name -> new section name.
    # Apply that same alias to the content-addressed cache target.
    for old_base, new_base in rename_map.items():
        if old_base in alias_map and new_base:
            alias_map[new_base] = alias_map[old_base]

    repaired = 0
    unresolved = set()
    for node, original_src, basename in referenced_nodes:
        if basename in already_present:
            continue
        target = alias_map.get(basename)
        if target and os.path.isfile(os.path.join(images_dir, target)):
            node["src"] = re.sub(
                re.escape(basename) + r"(?=([?#].*)?$)",
                target,
                original_src,
                count=1,
            )
            repaired += 1
        else:
            unresolved.add(basename)

    _log(
        log_callback,
        f"🖼️ PDF image preflight complete: {len(already_present)} already valid, "
        f"{repaired} reference(s) repaired, {len(unresolved)} unresolved",
    )
    if unresolved:
        preview = ", ".join(sorted(unresolved)[:5])
        suffix = "…" if len(unresolved) > 5 else ""
        _log(log_callback, f"⚠️ Missing PDF images: {preview}{suffix}")
    return [str(soup) for soup in soups], {
        "references": len(referenced_nodes),
        "repaired": repaired,
        "unresolved": len(unresolved),
    }


def _renderer_heartbeat(log_callback, done_event: threading.Event, section_count: int) -> None:
    started = time.perf_counter()
    try:
        heartbeat_seconds = max(
            0.05,
            float(os.environ.get("PDF_COMPILE_HEARTBEAT_SECONDS", "3")),
        )
    except (TypeError, ValueError):
        heartbeat_seconds = 3.0
    while not done_event.wait(heartbeat_seconds):
        elapsed = int(time.perf_counter() - started)
        _log(
            log_callback,
            f"⏳ PDF renderer heartbeat: laying out {section_count} section(s), "
            f"{elapsed}s elapsed",
        )


def _source_pdf_stem(folder: str) -> str:
    try:
        from output_workspace import read_workspace_source_path

        source = read_workspace_source_path(folder)
    except Exception:
        source = ""
    if source.lower().endswith(".pdf"):
        return os.path.splitext(os.path.basename(source))[0]
    leaf = os.path.basename(os.path.normpath(folder)) or "translated"
    return re.sub(r"_PDF(?:_\d+)?$", "", leaf, flags=re.IGNORECASE)


def _workspace_book_metadata(folder: str) -> dict:
    metadata_path = os.path.join(folder, "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _compiled_book_identity(folder: str) -> tuple[str, str, dict]:
    """Return the display title, safe output stem, and workspace metadata."""
    from pdf_output_naming import safe_pdf_book_filename_stem

    metadata = _workspace_book_metadata(folder)
    fallback = _source_pdf_stem(folder)
    display_title = str(
        metadata.get("title")
        or metadata.get("original_title")
        or fallback
    ).strip() or fallback
    folder_units = len(os.path.abspath(folder).encode("utf-16-le")) // 2
    suffix_units = len("\\_translated.pdf".encode("utf-16-le")) // 2
    path_budget = max(24, 240 - folder_units - suffix_units)
    stem = safe_pdf_book_filename_stem(
        display_title,
        fallback,
        max_units=min(180, path_budget),
    )
    return display_title, stem, metadata


def _remember_compiled_output_names(
    folder: str,
    metadata: dict,
    html_name: str,
    pdf_name: str,
) -> None:
    """Persist current compiled names so a future title change can retire them."""
    if not metadata:
        return
    metadata = dict(metadata)
    metadata["compiled_html_file"] = os.path.basename(html_name)
    metadata["compiled_pdf_file"] = os.path.basename(pdf_name)
    metadata_path = os.path.join(folder, "metadata.json")
    temporary_path = f"{metadata_path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, metadata_path)
    finally:
        if os.path.isfile(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _retire_previous_compiled_names(
    folder: str,
    current_paths: tuple[str, str],
    metadata: dict,
) -> None:
    """Remove only known app-generated predecessors after a successful compile."""
    current = {
        os.path.normcase(os.path.abspath(path))
        for path in current_paths
    }
    source_stem = _source_pdf_stem(folder)
    candidates = {
        str(metadata.get("compiled_html_file") or ""),
        str(metadata.get("compiled_pdf_file") or ""),
        f"{source_stem}_translated.html",
        f"{source_stem}_translated.pdf",
    }
    for name in candidates:
        basename = os.path.basename(name)
        lowered = basename.casefold()
        if not lowered.endswith(("_translated.html", "_translated.pdf")):
            continue
        path = os.path.abspath(os.path.join(folder, basename))
        if os.path.normcase(path) in current:
            continue
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _normalize_workspace_pdf_section_filenames(folder: str) -> int:
    """Move title-bearing section files to short names and update mappings."""
    from pdf_output_naming import (
        move_pdf_output_to_readable_name,
        readable_pdf_section_filename,
    )

    progress_path = os.path.join(folder, "translation_progress.json")
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
    except (OSError, ValueError, TypeError):
        return 0
    if not isinstance(progress, dict):
        return 0
    chapters = progress.get("chapters")
    if not isinstance(chapters, dict):
        return 0

    occupied = {
        os.path.basename(str(info.get("output_file") or "")).casefold()
        for info in chapters.values()
        if isinstance(info, dict) and info.get("output_file")
    }
    renamed = 0
    changed = False
    file_mapping = {}
    for info in chapters.values():
        if not isinstance(info, dict) or not info.get("pdf_toc_section"):
            continue
        actual_num = info.get("actual_num", info.get("chapter_num", 0))
        current = os.path.basename(str(info.get("output_file") or ""))
        preferred = readable_pdf_section_filename(
            {"pdf_toc_section": True, "num": actual_num},
            actual_num=actual_num,
            retain=not current.casefold().startswith("response_"),
        )
        if current.casefold() == preferred.casefold():
            continue
        try:
            mapped, moved = move_pdf_output_to_readable_name(
                folder,
                current,
                preferred,
                occupied=occupied,
            )
        except OSError:
            continue
        if not mapped:
            continue
        occupied.discard(current.casefold())
        occupied.add(mapped.casefold())
        info["output_file"] = mapped
        file_mapping[current.casefold()] = mapped
        changed = True
        if moved:
            renamed += 1

    completed_list = progress.get("completed_list")
    if isinstance(completed_list, list) and file_mapping:
        for item in completed_list:
            if not isinstance(item, dict):
                continue
            current = os.path.basename(str(item.get("file") or ""))
            mapped = file_mapping.get(current.casefold())
            if mapped:
                item["file"] = mapped

    if changed:
        temporary_path = f"{progress_path}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(progress, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, progress_path)
        finally:
            if os.path.isfile(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
    return renamed


def _outline_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split()).casefold()


def _keep_only_section_bookmarks(pdf_path: str, titles: list[str]) -> None:
    """Normalize the generated outline to exactly one entry per response."""
    import fitz

    temp_path = f"{pdf_path}.outline.tmp"
    try:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
    except OSError:
        pass
    document = fitz.open(pdf_path)
    try:
        generated = document.get_toc(simple=True) or []
        if (
            len(generated) == len(titles)
            and all(
                len(row) >= 3
                and _outline_title_key(row[1]) == _outline_title_key(title)
                for row, title in zip(generated, titles)
            )
        ):
            # WeasyPrint already produced the exact requested outline. Avoid
            # rewriting the PDF, which also lets compilation finish while the
            # newly rendered document is open in an external viewer.
            return
        cursor = 0
        cleaned = []
        for title in titles:
            wanted = _outline_title_key(title)
            matched_page = None
            for position in range(cursor, len(generated)):
                row = generated[position]
                if len(row) >= 3 and _outline_title_key(row[1]) == wanted:
                    matched_page = max(1, int(row[2]))
                    cursor = position + 1
                    break
            if matched_page is None:
                # WeasyPrint exposes the explicit bookmark anchors in its
                # outline. This fallback keeps the result valid if another
                # renderer dropped them, without reintroducing sentence-level
                # heading bookmarks.
                matched_page = cleaned[-1][2] if cleaned else 1
            cleaned.append([1, title, matched_page])
        document.set_toc(cleaned)
        document.save(temp_path, garbage=4, deflate=True)
    finally:
        document.close()
    os.replace(temp_path, pdf_path)


def compile_pdf_workspace(
    folder: str,
    log_callback: LogCallback | None = None,
    stop_callback: Callable[[], bool] | None = None,
    api_client=None,
) -> str:
    """Build a translated PDF from the current response HTML files."""
    if not folder or not os.path.isdir(folder):
        raise ValueError("PDF output workspace does not exist.")
    compile_started = time.perf_counter()
    folder = os.path.abspath(folder)
    _log(log_callback, "🚀 Rapid PDF workspace compiler started")
    _log(log_callback, f"📂 Workspace: {folder}")
    if api_client is not None and (
        os.getenv("USE_TOC_NCX", "1") == "1"
        or os.getenv("BATCH_TRANSLATE_HEADERS", "0") == "1"
    ):
        artifact_started = time.perf_counter()
        _log(
            log_callback,
            "📝 Phase 1/6: checking translated PDF title, bookmark, and "
            "header artifacts",
        )
        artifact_chapters = load_pdf_workspace_artifact_chapters(folder)
        if artifact_chapters:
            _log(
                log_callback,
                f"📝 Artifact plan: {len(artifact_chapters)} translated section(s)",
            )
            translate_pdf_workspace_artifacts(
                artifact_chapters,
                folder,
                api_client,
                log_callback=log_callback,
                stop_callback=stop_callback,
            )
        _log(
            log_callback,
            f"✅ Artifact phase finished in "
            f"{time.perf_counter() - artifact_started:.1f}s",
        )
    else:
        _log(
            log_callback,
            "📝 Phase 1/6: artifact translation already complete or not requested",
        )
    renamed_sections = _normalize_workspace_pdf_section_filenames(folder)
    if renamed_sections:
        _log(
            log_callback,
            f"Normalized {renamed_sections} PDF section filename(s) to short numbered names",
        )
    entries = _workspace_response_entries(folder)
    if not entries:
        raise ValueError("No translated response HTML files were found.")

    _log(
        log_callback,
        f"🔎 Located {len(entries)} translated response HTML section(s)",
    )
    preparation_started = time.perf_counter()
    _log(
        log_callback,
        "🎨 Phase 2/6: restoring source PDF formatting and normalizing HTML",
    )
    source_heading_alignments = _workspace_source_heading_alignments(folder)
    _log(
        log_callback,
        f"🎨 Source heading alignment map: "
        f"{len(source_heading_alignments)} section(s)",
    )
    source_contents = []
    titles = []
    last_decile = -1
    for index, (path, title) in enumerate(entries, 1):
        if stop_callback and stop_callback():
            raise PDFCompilationCancelled("PDF compilation stopped by user")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            original_content = handle.read()
        content = normalize_pdf_workspace_translated_html(
            original_content,
            folder,
            source_heading_alignments.get(os.path.basename(path).casefold()),
        )
        _write_response_html_if_changed(path, original_content, content)
        title = title or f"Section {index}"
        titles.append(title)
        source_contents.append(content)
        percent = int(index * 100 / len(entries))
        decile = percent // 10
        if index == 1 or index == len(entries) or decile > last_decile:
            last_decile = decile
            _log(
                log_callback,
                f"📄 PDF preparation: {index}/{len(entries)} sections ({percent}%)",
            )

    source_characters = sum(len(content) for content in source_contents)
    _log(
        log_callback,
        f"✅ HTML normalization complete: {len(source_contents)} section(s), "
        f"{source_characters:,} characters in "
        f"{time.perf_counter() - preparation_started:.1f}s",
    )

    image_started = time.perf_counter()
    _log(log_callback, "🖼️ Phase 3/6: validating and repairing image references")
    source_contents, image_stats = _repair_pdf_image_references(
        folder,
        source_contents,
        log_callback=log_callback,
        stop_callback=stop_callback,
    )
    _log(
        log_callback,
        f"✅ Image phase complete: {image_stats['references']} reference(s), "
        f"{image_stats['repaired']} repaired, "
        f"{image_stats['unresolved']} unresolved in "
        f"{time.perf_counter() - image_started:.1f}s",
    )
    from pdf_fast_extractor import pdf_rtl_paragraph_layout_enabled

    rtl_layout = pdf_rtl_paragraph_layout_enabled()
    section_class = "compiled-pdf-section"
    section_attributes = ""
    if rtl_layout:
        section_class += " pdf-rtl-layout"
        section_attributes = ' dir="rtl" data-pdf-rtl-layout="true"'
    assembly_started = time.perf_counter()
    _log(log_callback, "🧩 Phase 4/6: assembling bookmark-delimited PDF document")
    sections = []
    for index, (content, title) in enumerate(zip(source_contents, titles), 1):
        sections.append(
            f'<section class="{section_class}" data-section="{index}"'
            f'{section_attributes}>'
            f'<div id="pdf-section-{index}" class="pdf-bookmark-anchor">'
            f"{html.escape(title)}</div>"
            f"{_fragment_body(content)}</section>"
        )

    book_title, stem, book_metadata = _compiled_book_identity(folder)
    html_path = os.path.join(folder, f"{stem}_translated.html")
    pdf_path = os.path.join(folder, f"{stem}_translated.pdf")
    document_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(book_title)} - Translated</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    h1, h2, h3, h4, h5, h6 {{ bookmark-level: none !important; }}
    .pdf-bookmark-anchor {{
      bookmark-level: 1 !important;
      bookmark-label: content(text);
      height: 0; margin: 0; padding: 0; overflow: hidden;
      color: transparent; font-size: 0; line-height: 0;
    }}
    .compiled-pdf-section {{ margin: 0; padding: 0; }}
    .compiled-pdf-section + .compiled-pdf-section {{
      break-before: page;
      page-break-before: always;
    }}
    .pdf-fast-semantic-page p.pdf-align-left {{ text-align: left !important; }}
    .pdf-fast-semantic-page p.pdf-align-center {{ text-align: center !important; }}
    .pdf-fast-semantic-page p.pdf-align-right {{ text-align: right !important; }}
    .pdf-fast-semantic-page p.pdf-align-justify {{
      text-align: justify !important;
      text-justify: auto;
    }}
    .pdf-rtl-layout {{ direction: rtl; }}
    .pdf-rtl-layout p, .pdf-rtl-layout li,
    .pdf-rtl-layout td, .pdf-rtl-layout th {{
      direction: rtl;
      unicode-bidi: plaintext;
    }}
    .pdf-rtl-layout p.pdf-align-justify {{
      text-align-last: right;
    }}
    .pdf-fast-semantic-page a {{
      text-decoration: underline;
    }}
    .pdf-table {{
      width: 100%; border-collapse: collapse; margin: 1em 0;
      break-inside: avoid; page-break-inside: avoid;
    }}
    .pdf-table th, .pdf-table td {{
      border: 1px solid #777; padding: 0.35em 0.5em;
      vertical-align: top; text-align: left;
    }}
    .pdf-image, .pdf-vector-graphic {{
      margin: 1em 0; text-align: center;
      break-inside: avoid; page-break-inside: avoid;
    }}
    .pdf-image img, .pdf-vector-graphic img {{
      max-width: 100%; height: auto;
    }}
  </style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(document_html)
    html_bytes = os.path.getsize(html_path)
    _log(log_callback, f"📕 Book title: {book_title}")
    _log(log_callback, f"🧾 Combined HTML: {html_path}")
    _log(
        log_callback,
        f"✅ Assembly complete: {len(sections)} section(s), "
        f"{html_bytes / 1024 / 1024:.2f} MiB in "
        f"{time.perf_counter() - assembly_started:.1f}s",
    )

    from pdf_extractor import create_pdf_from_html

    css_path = os.path.join(folder, "styles.css")
    images_dir = os.path.join(folder, "images")
    if stop_callback and stop_callback():
        raise PDFCompilationCancelled("PDF compilation stopped by user")
    _log(
        log_callback,
        f"⚡ Phase 5/6: rendering with WeasyPrint "
        f"({image_stats['references']} image reference(s), "
        f"{image_stats['unresolved']} unresolved)",
    )
    _log(log_callback, f"📄 PDF target: {pdf_path}")
    render_started = time.perf_counter()
    renderer_done = threading.Event()
    heartbeat = threading.Thread(
        target=_renderer_heartbeat,
        args=(log_callback, renderer_done, len(entries)),
        name="pdf-compiler-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        success = create_pdf_from_html(
            document_html,
            pdf_path,
            css_path=css_path if os.path.isfile(css_path) else None,
            images_dir=images_dir if os.path.isdir(images_dir) else None,
        )
    finally:
        renderer_done.set()
        heartbeat.join(timeout=0.5)
    if not success or not os.path.isfile(pdf_path):
        raise RuntimeError("The PDF renderer did not create an output file.")
    render_elapsed = time.perf_counter() - render_started
    output_size = os.path.getsize(pdf_path)
    output_pages = 0
    try:
        import fitz

        with fitz.open(pdf_path) as document:
            output_pages = document.page_count
    except Exception:
        output_pages = 0
    page_summary = f", {output_pages} page(s)" if output_pages else ""
    _log(
        log_callback,
        f"✅ Rendering complete in {render_elapsed:.1f}s{page_summary}, "
        f"{output_size / 1024 / 1024:.2f} MiB",
    )
    bookmark_started = time.perf_counter()
    _log(
        log_callback,
        "📑 Phase 6/6: normalizing one bookmark per translated section",
    )
    _keep_only_section_bookmarks(pdf_path, titles)
    _retire_previous_compiled_names(
        folder,
        (html_path, pdf_path),
        book_metadata,
    )
    _remember_compiled_output_names(
        folder,
        book_metadata,
        os.path.basename(html_path),
        os.path.basename(pdf_path),
    )
    _log(
        log_callback,
        f"✅ Bookmark and metadata finalization complete in "
        f"{time.perf_counter() - bookmark_started:.1f}s",
    )
    _log(
        log_callback,
        f"✅ PDF compilation complete in "
        f"{time.perf_counter() - compile_started:.1f}s: {pdf_path}",
    )
    return pdf_path
