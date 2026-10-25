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
  config is never written here;
* Stop follows the desktop QA escalation (``next_qa_stop_phase``): the first Stop sets the
  graceful flags (``apply_qa_graceful_stop_flags``: scan loop stop, ``GRACEFUL_STOP``,
  ``TRANSLATION_CANCELLED``), a force stop (graceful stop off, or the second Stop) the force
  flags (``apply_qa_force_stop_flags``: + the client cancellation); after a stopped scan the
  heavy flags are cleared (``clear_qa_stop_flags``).

params: ``mode`` (``quick-scan`` | ``aggressive`` | ``ai-hunter`` | ``custom``),
``targets`` (``[{"folder", "source"}]``, + ``"direct_text": True`` for a chat workspace the
chat submitted; else every input is a folder), ``disable_word_count``.
outputs: the ``validation_results.html`` reports that exist after the scan.

The post-translation scan (desktop "scanning phase", hotfix 1 issue 6): ``run_scan_phase`` is the
desktop handler of the shared worker's ``_ui_request('trigger_qa_scan')``
(``TranslatorGUI._trigger_qa_scan_on_main_thread`` -> ``run_qa_scan(mode_override=
_get_scan_phase_mode(), non_interactive=True)``) for the translation's own job
(``translate.post_translation_qa``): the translated sources' output folders (the desktop
auto-search over the run's selection, here the folders the job recorded), then ``scan`` - the
same scan as a ``qa_scan`` job, so its reports, ``qa_failed`` marks and result fields are the ones
a QA job leaves.

Why the mobile sample size is not the generic ``params["config_overrides"]``: that is a shallow
top-level ``config.update`` at job start (``services.jobs._config_snapshot``), so a nested
``qa_scanner_settings`` override would replace the whole saved dict; the default here only
fills the one key the saved settings lack.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = ["KINDS", "MOBILE_QUICK_SAMPLE_SIZE", "MODES", "QUICK_SAMPLE_KEY", "REPORT_FILE",
           "SCAN_PHASE_SOURCE_EXTENSIONS", "normalize_targets", "report_path_for", "run", "run_scan_phase",
           "saved_quick_sample_size", "scan", "scan_phase_mode", "with_mobile_qa_defaults"]

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
    return scan(ctx, targets, mode, disable_word_count=bool(params.get("disable_word_count")))


def scan(ctx: Any, targets: Sequence[tuple], mode: str, *, disable_word_count: bool = False,
         qa: Any = None) -> dict:
    """The scan of a ``qa_scan`` job over ``[(folder, source or None, direct_text)]`` (``run``; the
    post-translation scan ``run_scan_phase`` too); returns ``{"ok", "outputs", "error"}``."""
    qa = qa if qa is not None else _qa_runtime()
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
            disable_word_count_for_run=bool(disable_word_count),
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


# ---- the post-translation scan (desktop "scanning phase") --------------------------------------------

#: The sources whose output folders the scanning phase looks for: the run's selection filtered like
#: ``QA_Scanner_GUI._qa_auto_search_sources`` (EPUB / TXT / PDF / MD / HTML).
SCAN_PHASE_SOURCE_EXTENSIONS = (".epub", ".txt", ".pdf", ".md", ".html", ".htm", ".xhtml")


def scan_phase_mode(owner: Any, config: Optional[Mapping[str, Any]] = None) -> str:
    """The scanning phase's QA mode: the owner's ``_get_scan_phase_mode`` (shared ``run_env.RunEnvMixin``,
    the desktop handler's call), else config.json ``scan_phase_mode``; ``quick-scan`` when unknown."""
    get = getattr(owner, "_get_scan_phase_mode", None)
    mode: Any = None
    if callable(get):
        try:
            mode = get()
        except Exception:
            mode = None
    if not mode:
        mode = (config or {}).get("scan_phase_mode") if isinstance(config, Mapping) else None
    mode = str(mode or "quick-scan").strip().lower()
    return mode if mode in MODES else "quick-scan"


def _scan_file_count(folder: str, text_mode: bool, is_html: Callable[[str], bool]) -> int:
    """The auto-search folder check (``QA_Scanner_GUI._auto_search_output_folders``): HTML/XHTML files,
    in text mode also ``.txt``."""
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    return sum(1 for name in names if (text_mode and name.lower().endswith(".txt")) or is_html(name))


def run_scan_phase(ctx: Any, sources: Sequence[str], folder_for: Callable[[str], Optional[str]], *,
                   mode: Optional[str] = None) -> list:
    """The desktop post-translation QA scan, in the translation's own job; returns the reports it wrote.

    Desktop: the worker's ``_ui_request('trigger_qa_scan')`` reaches ``_trigger_qa_scan_on_main_thread``,
    i.e. ``run_qa_scan(mode_override=self._get_scan_phase_mode(), non_interactive=True)``: the scanner
    starts, the selection's sources (Direct Text ones left out) are auto-searched for their output folders
    (none found, or Auto-search output off: "Skipping scan"), and the folders are scanned in that mode
    without a dialog; a scan error is logged and the translation stays done. Here ``folder_for(source)``
    gives the output folder the job translated into, and ``scan`` runs exactly what a ``qa_scan`` job runs
    (the mobile Quick Scan sample size, the QA Stop escalation, the reports added to the job's outputs and
    ``qa_reports`` / ``qa_mode`` / ``qa_folders`` on its result). A Stop before the scan skips it.
    """
    log = ctx.log
    owner = ctx.owner
    config = ctx.config or getattr(owner, "config", {}) or {}
    if not isinstance(config, Mapping):
        config = {}
    mode = mode if mode in MODES else scan_phase_mode(owner, config)
    if ctx.stop_requested():
        log("⏹️ Post-translation QA scan skipped: the job was stopped")
        return []
    try:
        qa = _qa_runtime()
    except Exception as exc:
        log(f"❌ Failed to start QA scan: {exc}")
        return []
    log("✅ QA scanner initialized successfully")
    is_direct_text = getattr(qa, "is_direct_text_qa_path", lambda _path: False)
    candidates: list = []
    seen: set = set()
    for path in sources or ():
        if not path or not str(path).lower().endswith(SCAN_PHASE_SOURCE_EXTENSIONS):
            continue
        path = os.path.abspath(os.fspath(path))
        if os.path.normcase(path) not in seen:
            seen.add(os.path.normcase(path))
            candidates.append(path)
    direct = [path for path in candidates if is_direct_text(path)]
    candidates = [path for path in candidates if not is_direct_text(path)]
    for path in direct:
        log(f"⏭️ Excluding Direct Text source from QA scan: {path}")
    if direct and not candidates:
        log("⏭️ QA scan skipped: Direct Text cannot trigger automatic QA scanning.")
        return []
    if not candidates:
        log("⚠️ Scanning phase: No EPUB files available for folder detection")
        log("⚠️ Skipping scan")
        return []
    if not config.get("qa_auto_search_output", True):
        # the desktop's non-interactive skip lines (its Expected / Exists / Available-folders debug lines name
        # '<EPUB folder>/<stem>', which is not where a mobile job writes: DISCREPANCIES "Post-translation QA scan")
        log(f"⚠️ Scanning phase: No matching output folders found for {len(candidates)} EPUB file(s)")
        log("⚠️ Skipping scan")
        return []
    is_html = getattr(qa, "is_html_like_path", None)
    if not callable(is_html):
        def is_html(name: str) -> bool:
            return str(name).lower().endswith((".html", ".xhtml", ".htm"))
    text_setting = bool(config.get("qa_text_file_mode", False))
    targets: list = []
    folders_seen: set = set()
    for source in candidates:
        base = os.path.splitext(os.path.basename(source))[0]
        try:
            folder = folder_for(source)
        except Exception:
            folder = None
        folder = os.path.abspath(os.fspath(folder)) if folder else None
        text_mode = text_setting or source.lower().endswith((".txt", ".pdf"))
        file_type = "TXT/HTML" if text_mode else "HTML/XHTML"
        count = _scan_file_count(folder, text_mode, is_html) if folder and os.path.isdir(folder) else 0
        if count:
            log(f"📁 Auto-selected output folder for {base}: {folder}")
            log(f"   Found {count} {file_type} files to scan")
            if os.path.normcase(folder) not in folders_seen:
                folders_seen.add(os.path.normcase(folder))
                targets.append((folder, source, False))
            continue
        if folder and os.path.isdir(folder):
            log(f"  [{base}] Folder exists but contains no {file_type} files: {folder}")
        log(f"  ⚠️ No output folder found for {base}")
    if not targets:
        log(f"⚠️ Scanning phase: No matching output folders found for {len(candidates)} EPUB file(s)")
        log("⚠️ Skipping scan")
        return []
    if len(candidates) > 1:
        log(f"📚 {len(candidates)} source files selected (primary: {os.path.basename(candidates[0])})")
    else:
        log(f"📚 Using EPUB from scan list: {os.path.basename(candidates[0])}")
    try:
        result = scan(ctx, targets, mode, qa=qa)
    except Exception as exc:
        log(f"❌ QA scan error: {exc}")
        log(f"Traceback: {traceback.format_exc()}")
        return []
    return [path for path in (result.get("outputs") or ()) if path]


KINDS = {
    "qa_scan": {"verb": "QA scanning", "icon": "FACT_CHECK", "stop_kind": "translation", "run": run},
}
