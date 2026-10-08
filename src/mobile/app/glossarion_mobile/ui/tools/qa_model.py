"""QA Scanner screen model (UI_SPEC §4.4): modes, Custom thresholds, pre-run checks, reports.

Pure Python (no Flet). Everything that decides a scan comes from the shared
``qa_scan_runtime`` (settings normalisation, the source/folder name matcher, the
Custom-mode defaults, the latest-report search) or ``job_kinds.qa`` (the job); this
module only shapes it for the screen:

* ``MODE_CARDS``: the desktop mode dialog's cards (value, emoji, title, subtitle,
  features, recommendation - the ``mode_data`` literal of ``QA_Scanner_GUI.run_qa_scan``;
  ``tests_host/test_tools_ui.py`` keeps them equal), shown in the UI_SPEC order;
* Custom mode: ``custom_values`` / ``custom_saved`` convert between the saved
  ``qa_scanner_settings.custom_mode_settings`` (fractions) and the sheet's percents
  exactly like the desktop Custom dialog's load and "Start Scan" save;
* pre-run checks with the desktop dialog texts (no source for word count, name mismatch);
* the Quick Scan duplicate-check sample size on mobile (owner 2026-10-08): 0 (duplicate check
  off) when config.json has none (``job_kinds.qa.MOBILE_QUICK_SAMPLE_SIZE``; the desktop keeps
  1000), a one-time migration of a saved desktop 1000 to 0 (``migrate_quick_sample_size``, flag
  in Prefs), still editable in Tools › QA Scanner and Settings › QA Scanner Settings;
* chat QA (owner 2026-10-08): ``chat_qa_job`` is the ``qa_scan`` job a chat submits for its own
  workspace - Quick Scan with the same sample size as Tools › QA Scanner, the Direct Text
  workspace flagged so the shared scan path's opt-in lets it through;
* reports: ``<folder>/<folder>_Scan Report/validation_results.html`` (+ ``.json``), the
  summary the report viewer shows, and the report HTML prepared for the WebView
  (file links post an "open" event instead of navigating; a CSP nonce lets only that
  script run).
"""

from __future__ import annotations

import html as html_lib
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

from glossarion_mobile.job_kinds.qa import (
    MOBILE_QUICK_SAMPLE_SIZE,
    QUICK_SAMPLE_KEY,
    REPORT_FILE,
    SOURCE_DEPENDENT_CHECKS,
    report_path_for,
    saved_quick_sample_size,
    with_mobile_qa_defaults,
)

__all__ = [
    "CHAT_QA_MODE",
    "CHAT_QA_SAMPLE_SIZE",
    "CHAT_QA_UNAVAILABLE",
    "CUSTOM_LIMITS",
    "CUSTOM_THRESHOLDS",
    "DESKTOP_QUICK_SAMPLE_SIZE",
    "DISPLAY_ORDER",
    "EVENT_PREFIX",
    "MISMATCH_TITLE",
    "MOBILE_QUICK_SAMPLE_SIZE",
    "MODE_CARDS",
    "ModeCard",
    "NO_SOURCE_TITLE",
    "QA_FAILED_FILTER",
    "QUICK_SAMPLE_HINT",
    "QUICK_SAMPLE_KEY",
    "QUICK_SAMPLE_LABEL",
    "QUICK_SAMPLE_MIGRATION_PREF",
    "ReportEntry",
    "ReportRow",
    "ReportSummary",
    "annotate_report_html",
    "chat_qa_job",
    "chat_qa_summary_line",
    "custom_defaults",
    "custom_saved",
    "custom_values",
    "effective_settings",
    "list_reports",
    "load_report_summary",
    "migrate_quick_sample_size",
    "mismatch_text",
    "names_match",
    "needs_source",
    "no_source_text",
    "parse_report_event",
    "qa_spec",
    "quick_sample_size",
    "report_folder",
    "report_path_for",
    "sample_size_text",
]

#: The Book page Chapters filter a QA report's "Open in Chapters" selects: the Failed chip
#: (``progress_core.STATUS_GROUPS['failed']`` = failed + qa_failed + refine_failed).
QA_FAILED_FILTER = "failed"
EVENT_PREFIX = "GLQA:"


@dataclass(frozen=True)
class ModeCard:
    value: str
    emoji: str
    title: str
    subtitle: str
    features: tuple
    recommendation: Optional[str] = None


#: ``QA_Scanner_GUI.run_qa_scan`` ``mode_data`` (desktop order).
MODE_CARDS = (
    ModeCard("ai-hunter", "🤖", "AI HUNTER", "30% threshold", (
        "✓ Catches AI retranslations", "✓ Different translation styles", "⚠ MANY false positives",
        "✓ Same chapter, different words", "✓ Detects paraphrasing", "✓ Ultimate duplicate finder"),
        "⚡ Best for finding ALL similar content"),
    ModeCard("aggressive", "🔥", "AGGRESSIVE", "75% threshold", (
        "✓ Catches most duplicates", "✓ Good for similar chapters", "⚠ Some false positives",
        "✓ Finds edited duplicates", "✓ Moderate detection", "✓ Balanced approach"), None),
    ModeCard("quick-scan", "⚡", "QUICK SCAN", "85% threshold, Speed optimized", (
        "✓ 3-5x faster scanning", "✓ Checks consecutive chapters only", "✓ Simplified analysis",
        "✓ Skips AI Hunter", "✓ Good for large libraries", "✓ Minimal resource usage"),
        "✅ Recommended for average use"),
    ModeCard("custom", "⚙️", "CUSTOM", "Configurable", (
        "✓ Fully customizable", "✓ Set your own thresholds", "✓ Advanced controls",
        "✓ Fine-tune detection", "✓ Expert mode", "✓ Maximum flexibility"), None),
)
#: UI_SPEC §4.4 order: Quick Scan (Recommended) · Aggressive · AI Hunter · Custom.
DISPLAY_ORDER = ("quick-scan", "aggressive", "ai-hunter", "custom")
MODES_BY_VALUE = {card.value: card for card in MODE_CARDS}

QUICK_SAMPLE_LABEL = "Quick Scan duplicate check sample size (characters):"
QUICK_SAMPLE_HINT = "Used only for duplicate detection; -1 = all text, 0 = disable check"

#: Custom dialog "Detection Thresholds (%)" rows: key, label, description (desktop wording).
CUSTOM_THRESHOLDS = (
    ("similarity", "Text Similarity", "Character-by-character comparison"),
    ("semantic", "Semantic Analysis", "Meaning and context matching"),
    ("structural", "Structural Patterns", "Document structure similarity"),
    ("word_overlap", "Word Overlap", "Common words between texts"),
    ("minhash_threshold", "MinHash Similarity", "Fast approximate matching"),
)
#: Custom dialog ranges: (minimum, maximum, step) per field (desktop spin boxes / sliders).
CUSTOM_LIMITS = {
    "threshold": (10, 100, 1),
    "consecutive_chapters": (1, 10, 1),
    "sample_size": (-1, 2000000000, 500),
    "min_text_length": (100, 5000, 100),
}


# ---- settings ---------------------------------------------------------------------------------


def _runtime() -> Any:
    try:
        import qa_scan_runtime

        return qa_scan_runtime
    except Exception:
        return None


def effective_settings(config: Mapping[str, Any]) -> dict:
    """The settings a scan starts from: ``normalize_qa_scan_settings`` over the saved ones (+ the
    mobile Quick Scan sample size when none is saved, as the job applies it)."""
    saved = dict((config or {}).get("qa_scanner_settings") or {})
    runtime = _runtime()
    if runtime is None:
        return with_mobile_qa_defaults(saved, config)
    try:
        language = (config or {}).get("output_language") or os.getenv("OUTPUT_LANGUAGE", "")
        return with_mobile_qa_defaults(runtime.normalize_qa_scan_settings(saved, target_language=language), config)
    except Exception:
        return with_mobile_qa_defaults(saved, config)


#: Prefs flag (``mobile_state.json``) of the one-time migration below.
QUICK_SAMPLE_MIGRATION_PREF = "qa_quick_sample_size_mobile_default"
#: The desktop default (``qa_scan_runtime.default_qa_scan_settings``) the U6-U9 QA screen saved
#: from its field on every Start / blur, so a phone's config.json very likely holds it.
DESKTOP_QUICK_SAMPLE_SIZE = 1000


def quick_sample_size(config: Optional[Mapping[str, Any]]) -> Any:
    """The Quick Scan sample size a mobile scan uses: the saved one, else ``MOBILE_QUICK_SAMPLE_SIZE``."""
    saved = saved_quick_sample_size(config)
    return MOBILE_QUICK_SAMPLE_SIZE if saved is None else saved


def migrate_quick_sample_size(get_cfg: Callable[..., Any], set_cfg: Callable[[Any, Any], Any],
                              prefs: Any) -> bool:
    """One-time mobile migration (owner 2026-10-08): a saved sample size of exactly 1000 becomes 0.

    ``get_cfg(key, default)`` / ``set_cfg(key, value)`` take the config path tuple
    (``MobileConfigStore.get``/``set``, ``ToolsContext.cfg``/``set_cfg``). The flag goes into Prefs
    (never config.json) whatever was saved, so a 1000 the owner types later stays 1000. Without
    Prefs nothing happens (the run could not be recorded). True when the value was changed.
    """
    if prefs is None:
        return False
    try:
        if prefs.get(QUICK_SAMPLE_MIGRATION_PREF):
            return False
        value = get_cfg(QUICK_SAMPLE_KEY, None)
        changed = False
        if type(value) is int and value == DESKTOP_QUICK_SAMPLE_SIZE:
            set_cfg(QUICK_SAMPLE_KEY, MOBILE_QUICK_SAMPLE_SIZE)
            changed = True
        prefs.set(QUICK_SAMPLE_MIGRATION_PREF, True)
        return changed
    except Exception:
        return False


def sample_size_text(value: Any) -> str:
    """The Quick Scan duplicate-check setting in plain words (desktop hint: -1 = all text, 0 = off)."""
    text = str(value).strip()
    if text == "0":
        return "duplicate check off (sample size 0)"
    if text == "-1":
        return "duplicate check on the full text (sample size -1)"
    return f"duplicate check sample size {text}"


def needs_source(settings: Mapping[str, Any]) -> bool:
    """Any source-dependent check is on (word count, AI truncation, silent truncation)."""
    return any(bool(settings.get(key, False)) for key in SOURCE_DEPENDENT_CHECKS)


def custom_defaults() -> Optional[dict]:
    """``qa_scan_runtime.DEFAULT_CUSTOM_MODE_SETTINGS`` (percents); None when the build lacks it."""
    runtime = _runtime()
    defaults = getattr(runtime, "DEFAULT_CUSTOM_MODE_SETTINGS", None) if runtime is not None else None
    return dict(defaults) if isinstance(defaults, Mapping) else None


def custom_values(saved: Optional[Mapping[str, Any]], defaults: Mapping[str, Any]) -> dict:
    """The Custom sheet's values: defaults overridden by the saved settings (desktop load).

    Saved thresholds are fractions (0.85); the sheet shows percents (``int(x * 100)``).
    """
    values = dict(defaults)
    if not saved:
        return values
    thresholds = saved.get("thresholds") or {}
    if thresholds:
        for key, _label, _desc in CUSTOM_THRESHOLDS:
            fallback = defaults.get(key, 0) / 100
            try:
                values[key] = int(float(thresholds.get(key, fallback)) * 100)
            except (TypeError, ValueError):
                values[key] = defaults.get(key)
    for key in ("consecutive_chapters", "check_all_pairs", "sample_size", "min_text_length",
                "min_duplicate_word_count"):
        if key in defaults:
            values[key] = saved.get(key, defaults[key])
    return values


def custom_saved(values: Mapping[str, Any]) -> dict:
    """``qa_scanner_settings.custom_mode_settings`` as the desktop Custom dialog saves it."""
    return {
        "thresholds": {key: int(values[key]) / 100 for key, _label, _desc in CUSTOM_THRESHOLDS},
        "consecutive_chapters": int(values["consecutive_chapters"]),
        "check_all_pairs": bool(values["check_all_pairs"]),
        "sample_size": int(values["sample_size"]),
        "min_text_length": int(values["min_text_length"]),
    }


# ---- pre-run checks (desktop dialogs) ------------------------------------------------------------

NO_SOURCE_TITLE = "No Source EPUB/HTML Selected"
MISMATCH_TITLE = "Source/Folder Name Mismatch"


def no_source_text() -> str:
    return "Word count cross-reference is enabled but no source EPUB/HTML file is selected."


def names_match(source: str, folder: str, settings: Mapping[str, Any]) -> bool:
    """``qa_scan_runtime.check_epub_folder_match`` (True when the helper is unavailable)."""
    runtime = _runtime()
    match = getattr(runtime, "check_epub_folder_match", None) if runtime is not None else None
    if not callable(match) or not source or not folder:
        return True
    epub_name = os.path.splitext(os.path.basename(source))[0]
    folder_name = os.path.basename(folder.rstrip("/\\"))
    try:
        return bool(match(epub_name, folder_name, settings.get("custom_output_suffixes", "")))
    except Exception:
        return True


def mismatch_text(source: str, folder: str) -> str:
    kind = "HTML" if source.lower().endswith((".html", ".htm", ".xhtml")) else "EPUB"
    epub_name = os.path.splitext(os.path.basename(source))[0]
    folder_name = os.path.basename(folder.rstrip("/\\"))
    return ("The source file and output folder names don't match:\n\n"
            f"📖 {kind}: {epub_name}\n"
            f"📁 Folder: {folder_name}\n\n"
            "This might mean you're comparing the wrong files.")


def qa_spec(targets: Sequence[Any], mode: str, *, disable_word_count: bool = False,
            origin: Optional[Mapping[str, Any]] = None) -> Any:
    """The ``qa_scan`` JobSpec for ToolTargets (folders + sources)."""
    from glossarion_mobile.services.jobs import JobSpec

    folders = [t.folder for t in targets if getattr(t, "folder", "")]
    if not folders:
        raise ValueError("Choose an output folder to scan")
    first = targets[0]
    title = str(getattr(first, "title", "") or os.path.basename(folders[0]))
    if len(folders) > 1:
        title = f"{title} +{len(folders) - 1}"
    params = {"mode": mode, "targets": [_target_param(t) for t in targets if getattr(t, "folder", "")]}
    if disable_word_count:
        params["disable_word_count"] = True
    return JobSpec(kind="qa_scan", title=title, inputs=tuple(folders), params=params,
                   origin=dict(origin or {"type": "tools", "tool": "qa", "label": "Tools · QA Scanner"}))


def _target_param(target: Any) -> dict:
    """``ToolTarget.to_param()``, + ``"direct_text": True`` for a Direct Text workspace (only the chat
    submits one: Tools › QA Scanner's ``qa_eligibility`` keeps them out of its picks)."""
    param = target.to_param()
    if getattr(target, "direct_text", False):
        param["direct_text"] = True
    return param


# ---- chat QA (owner 2026-10-08) ------------------------------------------------------------------

#: A scan started from the chat runs Quick Scan ...
CHAT_QA_MODE = "quick-scan"
#: ... with the mobile sample size (0 = duplicate check off) unless the owner saved another one in
#: Tools › QA Scanner / Settings › QA Scanner Settings (the job reads the same value).
CHAT_QA_SAMPLE_SIZE = MOBILE_QUICK_SAMPLE_SIZE
#: Why a chat workspace cannot be scanned by a build whose shared scanner lacks the opt-in.
CHAT_QA_UNAVAILABLE = "Available once this book is in the Library"


def _allows_direct_text(runtime: Any) -> bool:
    """The shared scan loop has the ``allow_direct_text`` opt-in (``qa_scan_runtime.run_bulk_qa_scan``;
    a wrapper that forwards ``**kwargs`` - instrumentation, a decorator without ``functools.wraps`` -
    passes it on too)."""
    loop = getattr(runtime, "run_bulk_qa_scan", None)
    if not callable(loop):
        return False
    try:
        parameters = inspect.signature(loop).parameters
    except (TypeError, ValueError):
        return False
    return "allow_direct_text" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def chat_qa_job(folder: str, source: Optional[str], *, cid: str,
                chat_title: str) -> Union[tuple, str]:
    """The ``qa_scan`` job a chat submits for one of its workspaces, as the chat's
    ``env.jobs.submit(kind, title, inputs, params, origin)`` parts; else the reason it cannot run.

    Quick Scan (``CHAT_QA_MODE``); the sample size is not fixed here: the job reads the saved
    value, else the mobile default, so the chat scans like Tools › QA Scanner does. A Direct Text
    workspace (``Output/Direct Text/<chat>/Attachments/<book>``) is flagged ``direct_text`` so the
    job opts out of the shared scan path's Direct Text guard; a workspace the Library already
    holds scans like any book. Without a source the word-count check is off for the run (one tap
    asks no question). The origin has no ``params["chat_id"]``: the chat's JobStrip shows the job.
    Pure: no disk access (the job logs a missing source and scans without it).
    """
    from glossarion_mobile.ui.tools.targets import ToolTarget

    if not folder:
        return "No output folder to scan"
    runtime = _runtime()
    if runtime is None:
        return "The QA scanner is not in this build"
    is_direct_text = getattr(runtime, "is_direct_text_qa_path", None)
    folder = os.path.abspath(os.fspath(folder))
    source = os.path.abspath(os.fspath(source)) if source else ""
    direct_text = bool(callable(is_direct_text) and (is_direct_text(folder) or (source and is_direct_text(source))))
    if direct_text and not _allows_direct_text(runtime):
        return CHAT_QA_UNAVAILABLE
    target = ToolTarget(title=os.path.basename(folder.rstrip("/\\")) or folder, folder=folder, source=source,
                        origin="chat", direct_text=direct_text)
    title = str(chat_title or "").strip()
    spec = qa_spec([target], CHAT_QA_MODE, disable_word_count=not source,
                   origin={"type": "chat", "cid": cid, "label": f"Chat · {title}" if title else "Chat"})
    return spec.kind, spec.title, tuple(spec.inputs), dict(spec.params), dict(spec.origin)


def chat_qa_summary_line(config: Optional[Mapping[str, Any]] = None) -> str:
    """What a chat scan runs, in plain words: ``Quick Scan · duplicate check off (sample size 0)``
    (``config``: the saved settings; without it, the mobile default)."""
    size = quick_sample_size(config) if config is not None else CHAT_QA_SAMPLE_SIZE
    return f"{MODES_BY_VALUE[CHAT_QA_MODE].title.title()} · {sample_size_text(size)}"


# ---- reports -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportEntry:
    path: str  # validation_results.html
    folder: str  # the scanned output folder
    mtime: float

    @property
    def title(self) -> str:
        return os.path.basename(self.folder.rstrip("/\\"))


@dataclass(frozen=True)
class ReportRow:
    index: Any
    filename: str
    score: int
    issues: tuple
    preview: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ReportSummary:
    total: int
    with_issues: int
    clean: int
    rows: tuple = ()  # files with issues, report order
    issue_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        return f"{self.total} files · {self.with_issues} with issues · {self.clean} clean"


def report_folder(report_path: str) -> str:
    """The scanned folder of ``<folder>/<name>_Scan Report/validation_results.html``."""
    return os.path.dirname(os.path.dirname(os.path.abspath(report_path)))


def list_reports(folders: Iterable[str], roots: Iterable[str] = (), *, limit: int = 100) -> list:
    """Reports of the given output folders and of every folder directly under the roots, newest first."""
    candidates: list = []
    seen: set = set()

    def add(folder: str) -> None:
        if not folder:
            return
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            return
        seen.add(key)
        candidates.append(os.path.abspath(folder))

    for folder in folders or ():
        add(folder)
    for root in roots or ():
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                add(path)
    out: list = []
    for folder in candidates:
        report = report_path_for(folder)
        try:
            mtime = os.path.getmtime(report)
        except OSError:
            continue
        out.append(ReportEntry(path=report, folder=folder, mtime=mtime))
    out.sort(key=lambda e: e.mtime, reverse=True)
    return out[:limit]


def _issue_type(issue: str) -> str:
    # scan_html_folder.generate_html_report's grouping
    return issue.split(":")[0] if ":" in issue else issue.split("_")[0]


def load_report_summary(report_path: str) -> Optional[ReportSummary]:
    """Summary of a report from its ``validation_results.json`` (None when it is missing/invalid)."""
    json_path = os.path.join(os.path.dirname(report_path), "validation_results.json")
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(results, list):
        return None
    rows: list = []
    counts: dict = {}
    with_issues = 0
    for item in results:
        if not isinstance(item, Mapping):
            continue
        issues = tuple(str(i) for i in (item.get("issues") or ()))
        for issue in issues:
            kind = _issue_type(issue)
            counts[kind] = counts.get(kind, 0) + 1
        if issues:
            with_issues += 1
            try:
                score = int(item.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            try:
                confidence = float(item.get("duplicate_confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            rows.append(ReportRow(index=item.get("file_index"), filename=str(item.get("filename") or ""),
                                  score=score, issues=issues, preview=str(item.get("preview") or "")[:300],
                                  confidence=confidence))
    total = sum(1 for item in results if isinstance(item, Mapping))
    return ReportSummary(total=total, with_issues=with_issues, clean=total - with_issues, rows=tuple(rows),
                         issue_counts=dict(sorted(counts.items())))


_FILE_LINK = re.compile(r"""<a href=(['"])\.\./(?P<name>[^'"]+)\1(?P<rest>[^>]*)>""", re.IGNORECASE)

_REPORT_CSS = (
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<style>body{margin:12px;-webkit-text-size-adjust:100%;}"
    "table{display:block;overflow-x:auto;}"
    "a[data-glqa-file]{color:#1a73e8;text-decoration:underline;cursor:pointer;}"
    ".glqa-open{display:inline-block;margin-top:4px;padding:4px 8px;border-radius:6px;"
    "border:1px solid #1a73e8;color:#1a73e8;font-size:0.85em;text-decoration:none;}</style>"
)


def annotate_report_html(text: str, *, nonce: str, event_path: str) -> str:
    """The report for the WebView: file links become "open" events, plus an "Open in Chapters" link.

    The scanner links every file as ``<a href='../<file>' target='_blank'>`` (a path the
    WebView cannot open); here each one carries ``data-glqa-file`` and a small "Open in
    Chapters" link follows it. One nonce'd script posts ``{type, file, seq}`` through
    ``console.log("GLQA:" + json)`` and the server's ``fetch`` event fallback (deduplicated
    by ``seq``).
    """

    def link(match: "re.Match[str]") -> str:
        name = match.group("name")
        attr = html_lib.escape(html_lib.unescape(name), quote=True)
        return f"<a href='#' data-glqa-file=\"{attr}\">"

    body = _FILE_LINK.sub(link, text)
    body = re.sub(r"(<a href='#' data-glqa-file=\"(?P<f>[^\"]*)\">.*?</a>)",
                  lambda m: m.group(1) + f"<br><a href='#' class='glqa-open' data-glqa-file=\"{m.group('f')}\" "
                  "data-glqa-action='chapters'>Open in Chapters</a>", body, flags=re.DOTALL)
    script = (
        f"<script nonce=\"{nonce}\">(function(){{var seq=0;var EV={json.dumps(event_path)};"
        "function post(o){o.seq=++seq;var s=JSON.stringify(o);"
        f"try{{console.log({json.dumps(EVENT_PREFIX)}+s);}}catch(e){{}}"
        "try{fetch(EV,{method:'POST',headers:{'Content-Type':'application/json'},body:s,"
        "credentials:'same-origin'});}catch(e){}}"
        "document.addEventListener('click',function(ev){var el=ev.target;"
        "while(el&&el!==document&&!(el.getAttribute&&el.getAttribute('data-glqa-file'))){el=el.parentNode;}"
        "if(!el||el===document){return;}ev.preventDefault();"
        "post({type:'open',file:el.getAttribute('data-glqa-file'),"
        "action:el.getAttribute('data-glqa-action')||'file'});},true);"
        "post({type:'ready'});})();</script>"
    )
    if re.search(r"<head[^>]*>", body, re.IGNORECASE):
        body = re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + _REPORT_CSS, body, count=1, flags=re.IGNORECASE)
    else:
        body = _REPORT_CSS + body
    if re.search(r"</body>", body, re.IGNORECASE):
        body = re.sub(r"</body>", lambda m: script + m.group(0), body, count=1, flags=re.IGNORECASE)
    else:
        body += script
    return body


def parse_report_event(message: Any) -> Optional[dict]:
    """A page event: a console line ``GLQA:{json}`` or an already-decoded server payload."""
    if isinstance(message, Mapping):
        payload = dict(message)
    else:
        text = str(message or "")
        index = text.find(EVENT_PREFIX)
        if index < 0:
            return None
        try:
            payload = json.loads(text[index + len(EVENT_PREFIX):])
        except ValueError:
            return None
    if not isinstance(payload, dict) or payload.get("type") not in ("open", "ready"):
        return None
    if payload.get("type") == "open":
        name = os.path.basename(str(payload.get("file") or "").replace("\\", "/"))
        if not name or name in (".", ".."):
            return None
        payload["file"] = name
        payload["action"] = "chapters" if payload.get("action") == "chapters" else "file"
    return payload


def report_exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.basename(path) == REPORT_FILE
