"""QA_SCAN: the desktop QA Scanner run over one or more translation output folders.

The scan itself is the desktop's: ``qa_scan_runtime.run_bulk_qa_scan`` is the body of the
desktop QA Scanner's ``run_scan`` worker (``QA_Scanner_GUI.run_qa_scan``, moved in U7). Per
folder it reloads the saved settings (``load_current_qa_settings``), matches the source EPUB
of a bulk scan by folder name (no fallback to another book's EPUB: the source-dependent
checks are disabled instead), logs a source/folder name mismatch, auto-detects a PDF source
and calls ``run_qa_scan_path`` (normalised settings, ``QA_*`` env, scanner caches,
``scan_html_folder.scan_html_folder``: reports in ``<folder>/<folder>_Scan Report/``, the
``qa_failed`` marks in ``translation_progress.json``), then logs the bulk summary and the cache
statistics. The owner is the job's ``HeadlessOwner``.

What happens around it here (UI_SPEC §4.4; the interactive parts - mode dialog, source
pickers, mismatch question - happen on the QA screen before the job is queued):

* the run start resets the global cancel flags (``reset_qa_cancel_flags``, desktop
  ``run_qa_scan``);
* the targets' sources come from the Library (or the picker): a bulk scan hands them to the
  shared loop as its "selected EPUBs" keyed by folder name, a single folder scan as its EPUB;
* Direct Text workspaces and missing folders are skipped (desktop: before the worker starts);
* ``disable_word_count`` (the user answered "continue without word count") switches the word
  count check off for the run;
* Stop follows the desktop QA escalation (``next_qa_stop_phase``): the first Stop sets the
  graceful flags (``apply_qa_graceful_stop_flags``: scan loop stop, ``GRACEFUL_STOP``,
  ``TRANSLATION_CANCELLED``), a force stop (graceful stop off, or the second Stop) the force
  flags (``apply_qa_force_stop_flags``: + the client cancellation); after a stopped scan the
  heavy flags are cleared (``clear_qa_stop_flags``).

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
#: settings whose checks need the source file (desktop ``_needs_epub``; the QA screen's source question)
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
    if not hasattr(qa_scan_runtime, "run_bulk_qa_scan"):  # pragma: no cover - older bundle
        raise JobError("This build's qa_scan_runtime has no run_bulk_qa_scan (U7).")
    return qa_scan_runtime


def _stop_flag(ctx: Any, qa: Any) -> tuple:
    """``(stop_flag, phase)``: the scan's stop poll with the desktop QA Stop escalation."""
    state = {"phase": "idle"}

    def graceful() -> bool:
        check = getattr(getattr(ctx, "host", None), "is_graceful_stop", None)
        try:
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def stop() -> bool:
        if not ctx.stop_requested():
            return False
        phase = state["phase"]
        if phase == "idle" or (phase == "graceful" and not graceful()):
            new = qa.next_qa_stop_phase(phase, graceful())
            if new == "graceful":
                qa.apply_qa_graceful_stop_flags()
            elif new == "force":
                qa.apply_qa_force_stop_flags()
            if new:
                state["phase"] = new
        return True

    return stop, state


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
    # Reset global cancel flags in case of a previous stop (desktop run_qa_scan)
    qa.reset_qa_cancel_flags()
    qa_settings = qa.load_current_qa_settings(config)
    # No ``set_output_dir``: ProgressWatcher would show the folder's translation progress
    # ("Ch 12/80") as the scan's progress; the phase line counts the folders instead.
    folders = [folder for folder, _source in allowed]
    total = len(folders)
    if total == 1:
        log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folders[0]}")
    else:
        log(f"🔍 Starting bulk QA scan in {mode.upper()} mode for {total} folders")
    ctx.phase("Scanning")
    stop_flag, stop_state = _stop_flag(ctx, qa)
    # The Library knows each folder's source: the shared loop's "selected EPUBs", keyed by folder name
    epub_basename_map = {os.path.basename(folder.rstrip("/\\")): source
                         for folder, source in allowed if source} if total > 1 else {}
    reports: list = []

    def scan_log(message: Any) -> None:
        text = str(message)
        if total > 1 and text.startswith("\n📁 ["):
            ctx.phase("Scanning " + text.split("[", 1)[1].split("]", 1)[0])
        log(message)

    def on_report(path: str) -> None:
        if path not in reports:
            reports.append(path)

    try:
        successful, _failed = qa.run_bulk_qa_scan(
            folders,
            mode=mode,
            epub_path=allowed[0][1] if total == 1 else None,
            qa_settings=qa_settings,
            load_settings=lambda: qa.load_current_qa_settings(config),
            selected_mode_value=mode,
            disable_word_count_for_run=bool(params.get("disable_word_count")),
            epub_basename_map=epub_basename_map,
            global_selected_files=None,
            log=scan_log,
            stop_flag=stop_flag,
            owner=ctx.owner,
            on_report=on_report,
        )
    finally:
        if stop_state["phase"] != "idle":
            qa.clear_qa_stop_flags()
    if reports:
        ctx.add_outputs(reports)
        ctx.set_result(qa_reports=list(reports), qa_mode=mode, qa_folders=folders)
    if ctx.stop_requested():
        return {"ok": None, "outputs": reports}
    if successful == 0:
        return {"ok": False, "outputs": reports, "error": "No folder could be scanned (see the log)."}
    return {"ok": True, "outputs": reports}


KINDS = {
    "qa_scan": {"verb": "QA scanning", "icon": "FACT_CHECK", "stop_kind": "translation", "run": run},
}
