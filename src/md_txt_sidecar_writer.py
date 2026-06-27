"""Shared HTML -> MD / TXT sidecar writer.

Mirrors ``sdlxliff_sidecar_writer.py``: both the translation pipeline (live
output, written next to where the .sdlxliff sidecar is written) and the
retroactive "Generate" buttons in Other Settings use the *same* extraction so
the MD/TXT always reflect the valid-HTML-tag list via the html2text extractor.

The MD and TXT files contain identical content (the html2text/markdown
extraction of the translated HTML); only the file extension differs. This
matches the user-selected behaviour ("Identical, just extensions").
"""

import os
import threading

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:  # pragma: no cover - stdlib always present
    ThreadPoolExecutor = None
    as_completed = None


_HTML_EXTS = (".html", ".htm", ".xhtml")


def _md_enabled():
    return str(os.getenv("OUTPUT_MD", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _txt_enabled():
    return str(os.getenv("OUTPUT_TXT", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _output_stem(output_filename):
    """Return the base stem for an HTML output name, or None if not HTML."""
    output_name = os.path.basename(str(output_filename).replace("\\", "/"))
    stem, ext = os.path.splitext(output_name)
    if ext.lower() not in _HTML_EXTS:
        return None
    return stem


def extract_markdown_from_html(target_html):
    """Run the translated HTML back through the html2text extractor.

    Uses EnhancedTextExtractor so the result honours the valid HTML tag list
    (``html_tag_entities.VALID_ENTITY_TAGS``) exactly like enhanced extraction
    does. Returns the extracted markdown/plain text (a str).
    """
    if not isinstance(target_html, str) or not target_html.strip():
        return ""
    from enhanced_text_extractor import EnhancedTextExtractor

    # "comprehensive" extracts the body content (avoids the 'full' XML
    # declaration artifact warning) while keeping all translated text.
    extractor = EnhancedTextExtractor(
        filtering_mode="comprehensive",
        preserve_structure=True,
    )
    display_text, translation_text, _title = extractor.extract_chapter_content(
        target_html, extraction_mode="comprehensive"
    )
    return display_text or translation_text or ""


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text if isinstance(text, str) else str(text or ""))


def _write_html_md_txt_sidecars(
    output_dir,
    output_filename,
    target_html,
    *,
    force_md=None,
    force_txt=None,
    extracted_text=None,
    raise_errors=False,
):
    """Write MD and/or TXT sidecars for one translated HTML output.

    By default the ``OUTPUT_MD`` / ``OUTPUT_TXT`` environment toggles decide
    which files are written. ``force_md`` / ``force_txt`` override them (used by
    the retroactive buttons). Returns a list of written paths, or None.
    """
    do_md = _md_enabled() if force_md is None else bool(force_md)
    do_txt = _txt_enabled() if force_txt is None else bool(force_txt)
    if not (do_md or do_txt):
        return None
    if not output_dir or not output_filename:
        return None
    if not isinstance(target_html, str) and extracted_text is None:
        return None

    stem = _output_stem(output_filename)
    if not stem:
        return None

    try:
        text = extracted_text if extracted_text is not None else extract_markdown_from_html(target_html)
    except Exception as exc:
        if raise_errors:
            raise
        print(f"WARNING: Failed to extract MD/TXT text for {output_filename}: {exc}")
        return None

    written = []
    try:
        if do_md:
            md_dir = os.path.join(output_dir, "MD")
            os.makedirs(md_dir, exist_ok=True)
            md_path = os.path.join(md_dir, f"{stem}.md")
            _write_text(md_path, text)
            written.append(md_path)
        if do_txt:
            txt_dir = os.path.join(output_dir, "TXT")
            os.makedirs(txt_dir, exist_ok=True)
            txt_path = os.path.join(txt_dir, f"{stem}.txt")
            _write_text(txt_path, text)
            written.append(txt_path)
    except Exception as exc:
        if raise_errors:
            raise
        print(f"WARNING: Failed to write MD/TXT sidecar for {output_filename}: {exc}")
        return None

    return written or None


def _list_root_html_files(output_dir):
    """Return absolute paths of translated HTML files at the output-dir root.

    Subfolders (SDLXLIFF/, MD/, TXT/, images/, css/, ...) are skipped because
    ``os.scandir`` only yields top-level entries and we filter to files.
    """
    files = []
    try:
        with os.scandir(output_dir) as scan:
            for entry in scan:
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                if entry.name.lower().endswith(_HTML_EXTS):
                    files.append(entry.path)
    except Exception:
        return []
    files.sort()
    return files


def generate_md_txt_for_output_dir(
    output_dir,
    *,
    do_md=True,
    do_txt=True,
    max_workers=4,
    log=None,
    should_stop=None,
):
    """Retroactively (re)generate MD/TXT for every root HTML file in a folder.

    Runs the html2text extraction in a thread pool so it never blocks the GUI.
    Existing MD/TXT are overwritten so they always reflect the current valid
    HTML tag list. Returns ``(ok_count, fail_count, total)``.
    """
    def _log(msg):
        if callable(log):
            try:
                log(msg)
            except Exception:
                pass

    if not (do_md or do_txt):
        return (0, 0, 0)
    if not output_dir or not os.path.isdir(output_dir):
        _log(f"  ⚠️ Output folder not found: {output_dir}")
        return (0, 0, 0)

    html_files = _list_root_html_files(output_dir)
    total = len(html_files)
    if total == 0:
        _log(f"  📄 No HTML files found in: {output_dir}")
        return (0, 0, 0)

    kinds = ("MD" if do_md else "") + ("/" if do_md and do_txt else "") + ("TXT" if do_txt else "")
    _log(f"  🔁 Generating {kinds} for {total} HTML file(s)…")

    ok = 0
    fail = 0
    done = 0
    lock = threading.Lock()

    def _process(path):
        if callable(should_stop) and should_stop():
            return ("stop", path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                html = fh.read()
        except Exception as exc:
            return ("fail", f"{os.path.basename(path)}: read error: {exc}")
        try:
            written = _write_html_md_txt_sidecars(
                output_dir,
                path,
                html,
                force_md=do_md,
                force_txt=do_txt,
                raise_errors=True,
            )
            return ("ok", written)
        except Exception as exc:
            return ("fail", f"{os.path.basename(path)}: {exc}")

    workers = max(1, int(max_workers or 1))
    use_pool = ThreadPoolExecutor is not None and workers > 1 and total > 1

    if use_pool:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, p): p for p in html_files}
            for fut in as_completed(futures):
                status, payload = fut.result()
                with lock:
                    done += 1
                    if status == "ok":
                        ok += 1
                    elif status == "fail":
                        fail += 1
                        _log(f"  ⚠️ {payload}")
                    if done % 25 == 0 or done == total:
                        _log(f"  … {done}/{total} processed")
                if status == "stop":
                    _log("  ⏹️ Stop requested; aborting MD/TXT generation.")
                    break
    else:
        for p in html_files:
            status, payload = _process(p)
            done += 1
            if status == "stop":
                _log("  ⏹️ Stop requested; aborting MD/TXT generation.")
                break
            if status == "ok":
                ok += 1
            elif status == "fail":
                fail += 1
                _log(f"  ⚠️ {payload}")
            if done % 25 == 0 or done == total:
                _log(f"  … {done}/{total} processed")

    return (ok, fail, total)
