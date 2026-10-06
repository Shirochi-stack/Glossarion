"""TRANSLATE_HEADERS: "Translate Headers Now" for one or more translated EPUB workspaces.

Desktop: Other Settings › "Translate Headers Now" (``other_settings.run_standalone_translate_headers``)
creates a ``UnifiedClient`` for the main model/key, runs
``translate_headers_standalone.run_translate_headers_gui(gui)`` (OPF-spine source titles ->
``BatchHeaderTranslator`` -> ``translated_headers.txt`` + the chapter ``<h1>``/``<title>``
updates) and then rebuilds the current EPUB with ``fallback_compile_epub``.

``run_translate_headers_gui`` imports ``PySide6.QtWidgets.QMessageBox`` before anything
else and shows message boxes on errors, so it cannot run in a mobile job (no PySide6 on the
device; without a QApplication a message box aborts the process on a desktop dev run). The
job therefore runs the module's own GUI-free entry for every EPUB,
``translate_headers_standalone.run_translation(source, output_dir, log_callback)`` - the same
``translate_headers_standalone`` engine the EPUB compile pipeline calls, configured from the
compile environment (``HEADERS_PER_BATCH``, ``FAILED_TRANSLATION_RETRY_ATTEMPTS``,
``BATCH_HEADER_*`` prompts, ``UPDATE_HTML_HEADERS``, ``SAVE_HEADER_TRANSLATIONS``, model / key,
temperature, max tokens). Before each book the job exports that environment with the owner's
``_build_epub_compile_env(folder)`` (desktop: the live GUI env) and configures the key pools
with ``key_pools.apply_key_pools_to_runtime`` (the desktop's multi-key client set-up).

Recorded divergences (until the desktop orchestration is split from its dialogs): an existing
``translated_headers.txt`` is translated again (TOC.txt entries are still reused) instead of
being re-applied with ``apply_existing_translations``; PDF workspaces are not handled here (Compile
PDF translates the bookmarks/headers); keyless models (``authgpt/``…) work, where the desktop
button refuses an empty API key field.

Afterwards the first EPUB target is rebuilt with ``owner._run_epub_compile(folder)`` (desktop:
the current EPUB only) unless ``rebuild_epub`` is False or the job was stopped.

params: ``targets`` (``[{"source", "folder"}]``; else the inputs are the source EPUBs and the
owner resolves their output folders), ``rebuild_epub`` (default True).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from glossarion_mobile.job_kinds import owner_method, result_fields

__all__ = ["KINDS", "normalize_targets", "run"]


def normalize_targets(params: Mapping[str, Any], inputs: Sequence[str], resolve: Any = None) -> list:
    """``[(source, folder)]`` - EPUB sources with their output folders (folder may be '')."""
    raw = params.get("targets")
    items: list = []
    if isinstance(raw, (list, tuple)) and raw:
        for item in raw:
            if isinstance(item, Mapping):
                items.append((item.get("source"), item.get("folder")))
            elif isinstance(item, (list, tuple)) and item:
                items.append((item[0], item[1] if len(item) > 1 else None))
    else:
        items = [(path, None) for path in inputs or ()]
    out: list = []
    seen: set = set()
    for source, folder in items:
        if not source:
            continue
        source = os.path.abspath(os.fspath(source))
        key = os.path.normcase(source)
        if key in seen:
            continue
        seen.add(key)
        if not folder and callable(resolve):
            try:
                folder = resolve(source)
            except Exception:
                folder = None
        out.append((source, os.path.abspath(os.fspath(folder)) if folder else ""))
    return out


def _has_html(folder: str) -> bool:
    try:
        return any(name.lower().endswith((".html", ".xhtml", ".htm")) for name in os.listdir(folder))
    except OSError:
        return False


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    owner = ctx.owner
    params = ctx.params or {}
    targets = normalize_targets(params, ctx.inputs, getattr(owner, "_resolve_translation_output_dir", None))
    if not targets:
        raise JobError("No EPUB or PDF file selected, or the file does not exist.")
    build_env = owner_method(owner, "_build_epub_compile_env")
    try:
        from translate_headers_standalone import run_translation as translate_book_headers
    except Exception as exc:  # pragma: no cover - bundle without the module
        raise JobError(f"The shared header translator is not in this build ({exc}).") from exc
    log = ctx.log
    total = len(targets)
    log(f"📊 Will process {total} EPUB/PDF file(s)")
    successful = failed = 0
    done_folders: list = []
    for index, (source, folder) in enumerate(targets, 1):
        if ctx.stop_requested():
            log("\n⛔ Translation stopped by user")
            log(f"📊 Stopped after processing {successful + failed}/{total} file(s)")
            break
        log(f"\n{'=' * 60}")
        kind = "PDF" if source.lower().endswith(".pdf") else "EPUB"
        log(f"📄 Processing {kind} {index}/{total}: {os.path.basename(source)}")
        log(f"{'=' * 60}")
        if total > 1:
            ctx.phase(f"Headers {index}/{total}")
        else:
            ctx.phase("Translating headers")
        base = os.path.splitext(os.path.basename(source))[0]
        if kind == "PDF":
            log("⏭️ PDF bookmark/header translation runs with Compile PDF on mobile; skipping.")
            failed += 1
            continue
        if not os.path.isfile(source):
            log(f"⚠️ Source EPUB not found: {source}")
            failed += 1
            continue
        if not folder or not os.path.isdir(folder) or not _has_html(folder):
            log(f"⚠️ Output directory not found for: {base}")
            failed += 1
            log(f"⏭️ Skipping to next EPUB... ({successful + failed}/{total} processed)\n")
            continue
        log(f"✓ Found output directory: {folder}")
        build_env(folder)
        os.environ["EPUB_PATH"] = source
        try:
            import key_pools

            key_pools.apply_key_pools_to_runtime(getattr(owner, "config", {}) or {})
        except Exception as exc:
            log(f"⚠️ Key pools not configured: {exc}")
        if total == 1:
            log("🌐 Starting standalone header translation...")
        result = translate_book_headers(source, folder, log_callback=log)
        if ctx.stop_requested():
            log(f"⛔ Translation stopped for: {base}")
            failed += 1
        elif getattr(result, "successful_noop", False):
            log("✅ Chapter headers complete: no source header tags found")
            successful += 1
            done_folders.append((source, folder))
        elif result:
            log(f"✅ Successfully translated {len(result)} chapter headers!")
            if os.environ.get("SAVE_HEADER_TRANSLATIONS", "1") == "1":
                log(f"📄 Translations saved to: {os.path.join(folder, 'translated_headers.txt')}")
            if os.environ.get("UPDATE_HTML_HEADERS", "1") == "1":
                log(f"🗂️ HTML files updated in: {folder}")
            successful += 1
            done_folders.append((source, folder))
        else:
            log(f"⚠️ No chapters were translated for: {base}")
            failed += 1
    if total > 1:
        log(f"\n{'=' * 60}")
        log("📊 Translation Summary:")
        log(f"  ✅ Successful: {successful}/{total}")
        if failed > 0:
            log(f"  ❌ Failed: {failed}/{total}")
        log(f"{'=' * 60}")
    ctx.set_result(headers_done=[f for _s, f in done_folders], headers_failed=failed)
    outputs: list = []
    if ctx.stop_requested():
        return {"ok": None, "outputs": outputs}
    epub_targets = [(s, f) for s, f in done_folders if s.lower().endswith(".epub")]
    if params.get("rebuild_epub", True) and epub_targets:
        source, folder = epub_targets[0]
        log("\n📦 Rebuilding EPUB with translated headers...")
        log(f"📂 Output directory: {folder}")
        ctx.set_output_dir(folder)
        ctx.phase("Rebuilding EPUB")
        compile_epub = owner_method(owner, "_run_epub_compile")
        ok, compiled, error = result_fields(compile_epub(folder))
        if ok is False:
            log(f"⚠️ Failed to rebuild EPUB: {error}")
        else:
            outputs.extend(compiled)
            if not ctx.stop_requested():
                log("✅ EPUB rebuilt successfully with translated headers!")
    if successful == 0:
        return {"ok": False, "outputs": outputs, "error": "No chapter headers were translated (see the log)."}
    return {"ok": True, "outputs": outputs}


KINDS = {
    "translate_headers": {"verb": "Translating headers", "icon": "TITLE", "stop_kind": "translation", "run": run},
}
