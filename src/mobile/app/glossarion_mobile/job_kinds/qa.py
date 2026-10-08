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
* Direct Text workspaces and missing folders are skipped (desktop: before the worker starts),
  except a target the chat itself submitted with ``"direct_text": True`` (owner 2026-10-08:
  QA scan from the chat, ``qa_model.chat_qa_job``): that one keeps its source and the shared
  loop runs with ``allow_direct_text=True``, the opt-in of ``run_qa_scan_path``'s Direct Text
  guard. Unflagged Direct Text folders (and the inputs fallback) keep the desktop skip;
* ``disable_word_count`` (the user answered "continue without word count") switches the word
  count check off for the run;
* the Quick Scan duplicate-check sample size: when config.json has none, Glossarion Mobile
  scans with ``MOBILE_QUICK_SAMPLE_SIZE`` (0 = duplicate check off, owner 2026-10-08; the
  desktop default stays ``qa_scan_runtime``'s 1000). It is applied inside the per-folder
  settings loader, so every folder of a bulk scan gets it; a saved value always wins and the
  config is never written here (``qa_model.migrate_quick_sample_size`` turns a saved desktop
  1000 into 0 once);
* Stop follows the desktop QA escalation (``next_qa_stop_phase``): the first Stop sets the
  graceful flags (``apply_qa_graceful_stop_flags``: scan loop stop, ``GRACEFUL_STOP``,
  ``TRANSLATION_CANCELLED``), a force stop (graceful stop off, or the second Stop) the force
  flags (``apply_qa_force_stop_flags``: + the client cancellation); after a stopped scan the
  heavy flags are cleared (``clear_qa_stop_flags``).

params: ``mode`` (``quick-scan`` | ``aggressive`` | ``ai-hunter`` | ``custom``),
``targets`` (``[{"folder", "source"}]``, + ``"direct_text": True`` for a chat workspace the
chat submitted; else every input is a folder), ``disable_word_count``.
outputs: the ``validation_results.html`` reports that exist after the scan.

Why the mobile sample size is not the generic ``params["config_overrides"]``: that is a shallow
top-level ``config.update`` at job start (``services.jobs._config_snapshot``), so a nested
``qa_scanner_settings`` override would replace the whole saved dict; the default here only
fills the one key the saved settings lack.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = ["KINDS", "MOBILE_QUICK_SAMPLE_SIZE", "MODES", "QUICK_SAMPLE_KEY", "REPORT_FILE",
           "normalize_targets", "report_path_for", "run", "saved_quick_sample_size", "with_mobile_qa_defaults"]

MODES = ("quick-scan", "aggressive", "ai-hunter", "custom")
REPORT_FILE = "validation_results.html"
#: settings whose checks need the source file (desktop ``_needs_epub``; the QA screen's source question)
SOURCE_DEPENDENT_CHECKS = ("check_word_count_ratio", "check_ai_truncation_detection", "check_silent_truncation")
#: config.json path of the Quick Scan duplicate-check sample size (desktop "Quick Scan duplicate
#: check sample size (characters)": -1 = all text, 0 = disable check)
QUICK_SAMPLE_KEY = ("qa_scanner_settings", "quick_scan_sample_size")
#: Glossarion Mobile's value when config.json has none (owner 2026-10-08: the duplicate check is
#: off by default on phones; the desktop default stays ``qa_scan_runtime``'s 1000)
MOBILE_QUICK_SAMPLE_SIZE = 0


def report_path_for(folder: str) -> str:
    """``<folder>/<basename>_Scan Report/validation_results.html`` (``scan_html_folder.generate_reports``)."""
    folder = os.fspath(folder)
    return os.path.join(folder, os.path.basename(folder.rstrip("/\\")) + "_Scan Report", REPORT_FILE)


def saved_quick_sample_size(config: Optional[Mapping[str, Any]]) -> Any:
    """The Quick Scan sample size saved in config.json (None when there is none)."""
    settings = (config or {}).get(QUICK_SAMPLE_KEY[0]) if isinstance(config, Mapping) else None
    if not isinstance(settings, Mapping):
        return None
    return settings.get(QUICK_SAMPLE_KEY[1])


def with_mobile_qa_defaults(settings: Mapping[str, Any], config: Optional[Mapping[str, Any]]) -> dict:
    """``settings`` (``load_current_qa_settings``) with the mobile Quick Scan sample size when the
    saved config has none (the shared normaliser fills in the desktop 1000 otherwise)."""
    out = dict(settings or {})
    if saved_quick_sample_size(config) is None:
        out[QUICK_SAMPLE_KEY[1]] = MOBILE_QUICK_SAMPLE_SIZE
    return out


def normalize_targets(params: Mapping[str, Any], inputs: Sequence[str], *, with_flags: bool = False) -> list:
    """``[(folder, source or None)]`` from ``params["targets"]`` (else the inputs as folders).

    ``with_flags``: ``[(folder, source or None, direct_text)]``, where ``direct_text`` is True
    only for a target mapping that carries ``"direct_text": True`` (a chat QA scan).
    """
    out: list = []
    seen: set = set()
    raw = params.get("targets")
    items: list = []
    if isinstance(raw, (list, tuple)) and raw:
        for item in raw:
            if isinstance(item, Mapping):
                items.append((item.get("folder"), item.get("source"), item.get("direct_text") is True))
            elif isinstance(item, (list, tuple)) and item:
                items.append((item[0], item[1] if len(item) > 1 else None, False))
            elif item:
                items.append((item, None, False))
    else:
        items = [(path, None, False) for path in inputs or ()]
    for folder, source, flagged in items:
        if not folder:
            continue
        folder = os.path.abspath(os.fspath(folder))
        key = os.path.normcase(folder)
        if key in seen:
            continue
        seen.add(key)
        source = os.path.abspath(os.fspath(source)) if source else None
        out.append((folder, source, flagged) if with_flags else (folder, source))
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
    targets = normalize_targets(params, ctx.inputs, with_flags=True)
    if not targets:
        raise JobError("Nothing to scan: no output folder.")
    qa = _qa_runtime()
    is_direct_text = getattr(qa, "is_direct_text_qa_path", lambda _path: False)
    config = ctx.config or getattr(ctx.owner, "config", {}) or {}
    log = ctx.log
    allowed: list = []
    chat_scan = False  # a chat-submitted Direct Text workspace is among the folders
    for folder, source, flagged in targets:
        if not os.path.isdir(folder):
            log(f"⚠️ Ignoring missing output folder: {folder}")
            continue
        if is_direct_text(folder) and not flagged:
            log(f"⏭️ Skipping Direct Text folder during QA scan: {folder}")
            continue
        if source and ((is_direct_text(source) and not flagged) or not os.path.isfile(source)):
            if not os.path.isfile(source):
                log(f"⚠️ Source file not found, scanning without it: {os.path.basename(source)}")
            source = None
        chat_scan = chat_scan or (flagged and (is_direct_text(folder) or is_direct_text(source)))
        allowed.append((folder, source))
    if not allowed:
        log("⏭️ QA scan skipped: no non-Direct-Text output folders were selected.")
        return {"ok": False, "outputs": [], "error": "No output folder to scan."}
    # Reset global cancel flags in case of a previous stop (desktop run_qa_scan)
    qa.reset_qa_cancel_flags()

    def load_settings() -> dict:
        # the saved settings, reloaded per folder like the desktop loop, + the mobile sample size default
        return with_mobile_qa_defaults(qa.load_current_qa_settings(config), config)

    qa_settings = load_settings()
    # No ``set_output_dir``: ProgressWatcher would show the folder's translation progress
    # ("Ch 12/80") as the scan's progress; the phase line counts the folders instead.
    folders = [folder for folder, _source in allowed]
    total = len(folders)
    if total == 1:
        log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folders[0]}")
    else:
        log(f"🔍 Starting bulk QA scan in {mode.upper()} mode for {total} folders")
    if mode == "quick-scan":  # the value this run uses (the Tools / Settings field), in plain words
        size = qa_settings.get(QUICK_SAMPLE_KEY[1])
        note = " (duplicate check off)" if str(size).strip() == "0" else ""
        default = " · Glossarion Mobile default" if saved_quick_sample_size(config) is None else ""
        log(f"⚡ Quick Scan duplicate check sample size: {size}{note}{default}")
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

    # only a chat-submitted Direct Text workspace opts out of run_qa_scan_path's Direct Text guard
    opt_in: dict = {"allow_direct_text": True} if chat_scan else {}
    try:
        successful, _failed = qa.run_bulk_qa_scan(
            folders,
            mode=mode,
            epub_path=allowed[0][1] if total == 1 else None,
            qa_settings=qa_settings,
            load_settings=load_settings,
            selected_mode_value=mode,
            disable_word_count_for_run=bool(params.get("disable_word_count")),
            epub_basename_map=epub_basename_map,
            global_selected_files=None,
            log=scan_log,
            stop_flag=stop_flag,
            owner=ctx.owner,
            on_report=on_report,
            **opt_in,
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
