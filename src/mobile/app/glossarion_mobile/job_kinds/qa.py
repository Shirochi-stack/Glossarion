"""QA_SCAN: the desktop QA Scanner run over one or more translation output folders.

Every folder goes through ``qa_scan_runtime.run_qa_scan_path`` - the shared path the
desktop QA Scanner (``QA_Scanner_GUI.run_qa_scan``), the post-translation scan phase and
the worker-side Multipass "Failed" scan all use: it normalises the settings, mirrors them
into the ``QA_*`` env, configures the scanner caches and calls
``scan_html_folder.scan_html_folder`` (reports in ``<folder>/<folder>_Scan Report/``, the
``qa_failed`` marks in ``translation_progress.json``). The owner is the job's
``HeadlessOwner`` (``prepare_qa_scan_settings(owner=...)`` reads its config, API key field
and model exactly as it reads TranslatorGUI's).

What the desktop's ``run_scan`` closure does around that call, this adapter does with the
same rules and log lines (UI_SPEC §4.4; the interactive parts - mode dialog, source
pickers, mismatch question - happen on the QA screen before the job is queued):

* settings: ``normalize_qa_scan_settings(config['qa_scanner_settings'],
  target_language=config['output_language'])`` (desktop ``_load_current_qa_settings``);
  the Custom mode thresholds are the saved ``custom_mode_settings`` (the Custom sheet
  saves them first, like the desktop "Start Scan" button);
* ``disable_word_count`` (the user answered "continue without word count"): the run's
  ``check_word_count_ratio`` is off;
* a bulk scan (2+ folders) whose folder has no matching source disables every
  source-dependent check (word count, AI truncation, silent truncation) - the desktop's
  "NO FALLBACK TO GLOBAL EPUB FOR BULK SCANS" rule; on mobile the source comes from the
  Library (or the picker), never from a file-name search;
* a source/folder name mismatch is logged with ``check_epub_folder_match`` when
  ``qa_scan_runtime`` provides it (the shared copy of the QA_Scanner_GUI helper);
* Direct Text workspaces are skipped (``is_direct_text_qa_path``), a missing folder is
  skipped, a failing folder fails the whole job only when it is the only one;
* bulk summary, cache statistics (``cache_show_stats``) and the closing line.

Stop: ``run_qa_scan_path`` polls the job latch; the first poll after a Stop also raises
``scan_html_folder.stop_scan()`` (the desktop QA stop sets both). The job's stop protocol
is the translation one (JobService); the desktop's QA-specific escalation
(``stop_qa_scan``) is not extracted (recorded divergence).

params: ``mode`` (``quick-scan`` | ``aggressive`` | ``ai-hunter`` | ``custom``),
``targets`` (``[{"folder", "source"}]``; else every input is a folder), ``disable_word_count``.
outputs: the ``validation_results.html`` reports that exist after the scan.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

__all__ = ["KINDS", "MODES", "REPORT_FILE", "normalize_targets", "report_path_for", "run"]

MODES = ("quick-scan", "aggressive", "ai-hunter", "custom")
REPORT_FILE = "validation_results.html"
#: settings whose checks need the source file (desktop ``_needs_epub``)
SOURCE_DEPENDENT_CHECKS = ("check_word_count_ratio", "check_ai_truncation_detection", "check_silent_truncation")


def report_path_for(folder: str) -> str:
    """``<folder>/<basename>_Scan Report/validation_results.html`` (``scan_html_folder.generate_reports``)."""
    folder = os.fspath(folder)
    return os.path.join(folder, os.path.basename(folder.rstrip("/\\")) + "_Scan Report", REPORT_FILE)


def normalize_targets(params: Mapping[str, Any], inputs: Sequence[str]) -> list:
    """``[(folder, source or None)]`` from ``params["targets"]`` (else the inputs as folders)."""
    out: list = []
    seen: set = set()
    raw = params.get("targets")
    items: list = []
    if isinstance(raw, (list, tuple)) and raw:
        for item in raw:
            if isinstance(item, Mapping):
                items.append((item.get("folder"), item.get("source")))
            elif isinstance(item, (list, tuple)) and item:
                items.append((item[0], item[1] if len(item) > 1 else None))
            elif item:
                items.append((item, None))
    else:
        items = [(path, None) for path in inputs or ()]
    for folder, source in items:
        if not folder:
            continue
        folder = os.path.abspath(os.fspath(folder))
        key = os.path.normcase(folder)
        if key in seen:
            continue
        seen.add(key)
        source = os.path.abspath(os.fspath(source)) if source else None
        out.append((folder, source))
    return out


def _qa_runtime() -> Any:
    from glossarion_mobile.services.jobs import JobError

    try:
        import qa_scan_runtime
    except Exception as exc:  # pragma: no cover - bundle without the scanner
        raise JobError(f"The shared QA scanner (qa_scan_runtime) is not in this build ({exc}).") from exc
    return qa_scan_runtime


def _load_settings(qa_scan_runtime: Any, config: Mapping[str, Any]) -> dict:
    """Desktop ``_load_current_qa_settings``: the shared normaliser over the saved settings."""
    try:
        main_lang = config.get("output_language") or os.getenv("OUTPUT_LANGUAGE", "")
        return qa_scan_runtime.normalize_qa_scan_settings(config.get("qa_scanner_settings", {}),
                                                          target_language=main_lang)
    except Exception:
        return dict(config.get("qa_scanner_settings", {}) or {})


def _stop_flag(ctx: Any) -> Callable[[], bool]:
    raised: list = []

    def stop() -> bool:
        if not ctx.stop_requested():
            return False
        if not raised:
            raised.append(True)
            try:
                from scan_html_folder import stop_scan

                stop_scan()
            except Exception:
                pass
        return True

    return stop


def _log_cache_stats(log: Callable[[str], Any]) -> None:
    try:
        from scan_html_folder import get_cache_info

        cache_stats = get_cache_info()
    except Exception:
        return
    log("\n📊 Cache Performance Statistics:")
    for name, info in cache_stats.items():
        if info:
            hit_rate = info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
            log(f"  {name}: {info.hits} hits, {info.misses} misses ({hit_rate:.1%} hit rate)")


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    params = ctx.params or {}
    mode = str(params.get("mode") or "quick-scan")
    if mode not in MODES:
        raise JobError(f"Unknown QA scan mode: {mode}")
    targets = normalize_targets(params, ctx.inputs)
    if not targets:
        raise JobError("Nothing to scan: no output folder.")
    qa = _qa_runtime()
    is_direct_text = getattr(qa, "is_direct_text_qa_path", lambda _path: False)
    name_match = getattr(qa, "check_epub_folder_match", None)
    config = ctx.config or getattr(ctx.owner, "config", {}) or {}
    log = ctx.log
    allowed: list = []
    for folder, source in targets:
        if not os.path.isdir(folder):
            log(f"⚠️ Ignoring missing output folder: {folder}")
            continue
        if is_direct_text(folder):
            log(f"⏭️ Skipping Direct Text folder during QA scan: {folder}")
            continue
        if source and (is_direct_text(source) or not os.path.isfile(source)):
            if not os.path.isfile(source):
                log(f"⚠️ Source file not found, scanning without it: {os.path.basename(source)}")
            source = None
        allowed.append((folder, source))
    if not allowed:
        log("⏭️ QA scan skipped: no non-Direct-Text output folders were selected.")
        return {"ok": False, "outputs": [], "error": "No output folder to scan."}
    # No ``set_output_dir``: ProgressWatcher would show the folder's translation progress
    # ("Ch 12/80") as the scan's progress; the phase line counts the folders instead.
    total = len(allowed)
    if total == 1:
        log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {allowed[0][0]}")
    else:
        log(f"🔍 Starting bulk QA scan in {mode.upper()} mode for {total} folders")
    ctx.phase("Scanning")
    stop_flag = _stop_flag(ctx)
    disable_word_count = bool(params.get("disable_word_count"))
    successful = failed = 0
    reports: list = []
    last_settings: dict = {}
    for index, (folder, source) in enumerate(allowed):
        if stop_flag():
            log(f"⚠️ Bulk scan stopped by user at folder {index + 1}/{total}")
            break
        folder_name = os.path.basename(folder)
        if total > 1:
            log(f"\n📁 [{index + 1}/{total}] Scanning folder: {folder_name}")
            ctx.phase(f"Scanning {index + 1}/{total}")
        settings = dict(_load_settings(qa, config))
        last_settings = settings
        if disable_word_count:
            settings["check_word_count_ratio"] = False
        needs_source = any(settings.get(key, False) for key in SOURCE_DEPENDENT_CHECKS)
        if total > 1 and needs_source:
            if source:
                log(f"  📖 Using EPUB: {os.path.basename(source)}")
            else:
                log(f"  ⚠️ No matching EPUB found for folder '{folder_name}' - disabling EPUB-dependent checks")
                for key in SOURCE_DEPENDENT_CHECKS:
                    settings[key] = False
        if (source and callable(name_match) and settings.get("check_word_count_ratio", False)
                and settings.get("warn_name_mismatch", True)):
            epub_name = os.path.splitext(os.path.basename(source))[0]
            folder_check = os.path.basename(folder.rstrip("/\\"))
            try:
                matches = name_match(epub_name, folder_check, settings.get("custom_output_suffixes", ""))
            except Exception:
                matches = True
            if not matches:
                log(f"  ⚠️ Warning: source/folder name mismatch - {epub_name} vs {folder_check}")
        try:
            qa.run_qa_scan_path(
                folder,
                log=log,
                stop_flag=stop_flag,
                mode=mode,
                qa_settings=settings,
                epub_path=source,
                selected_files=None,
                text_file_mode=None,
                owner=ctx.owner,
            )
            successful += 1
            report = report_path_for(folder)
            if os.path.exists(report):
                reports.append(report)
            if total > 1:
                log(f"✅ Folder '{folder_name}' scan completed successfully")
        except Exception as folder_error:
            failed += 1
            log(f"❌ Folder '{folder_name}' scan failed: {folder_error}")
            if total == 1:
                raise
    if total > 1:
        log(f"\n📋 Bulk scan summary: {successful} successful, {failed} failed")
    if last_settings.get("cache_show_stats", False):
        _log_cache_stats(log)
    if reports:
        ctx.add_outputs(reports)
        ctx.set_result(qa_reports=list(reports), qa_mode=mode, qa_folders=[f for f, _s in allowed])
    if ctx.stop_requested():
        return {"ok": None, "outputs": reports}
    log("✅ QA scan completed successfully." if total == 1 else "✅ Bulk QA scan completed.")
    if successful == 0:
        return {"ok": False, "outputs": reports, "error": "No folder could be scanned (see the log)."}
    return {"ok": True, "outputs": reports}


KINDS = {
    "qa_scan": {"verb": "QA scanning", "icon": "FACT_CHECK", "stop_kind": "translation", "run": run},
}
