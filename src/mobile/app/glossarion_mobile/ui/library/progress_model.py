"""Book page Chapters / Glossary models over the shared progress cores (UI_SPEC §3.7, §3.8). No Flet.

Everything here runs on the io pool and calls the shared code the desktop Progress
Manager and Glossary Progress panels run:

* Chapters: ``progress_core.build_book_progress`` (opens a source exactly like the
  desktop Progress Manager: output folder, source link, seeding, cleanup, spine
  match - writes through the three-way merge), ``refresh_book_progress`` (the 2 s
  read-only tick; ``read_only=False`` = Refresh), ``set_view_toggles``; its
  ``RowPresentation`` rows and ``ProgressStats`` become ``RowVM`` / ``StatChip``.
* Actions: ``progress_actions`` (``plan_remove_qa_marks`` / ``remove_qa_marks``,
  ``remove_pending_marks``, ``refinement_status_keys`` / ``remove_refinement_status``,
  ``restore_in_progress``, ``reset_tts``, ``find_row_audio`` / ``delete_row_audio``,
  ``resolve_llm_token_qa``, ``insert_missing_images``, ``build_partial_b_request``,
  ``special_keyword_for`` / ``remove_special_keyword``, ``row_actions``) and their
  desktop summary texts. Every progress write is ``mutate_progress`` inside them.
* Glossary: ``glossary_progress_core`` (``open_glossary_progress``,
  ``reload_glossary_progress``, ``glossary_rows``, ``glossary_stats``,
  ``mark_glossary_completed``, ``remove_glossary_progress``, ``glossary_footnotes``,
  ``write_glossary_summary``; writes under the extractor's lock + atomic replace).

The owner of a book's view is a ``progress_core.ProgressOwner`` over the app config
snapshot; settings it changes (Show model info, Do not skip) are saved back sparsely
through ``LibraryService.save_owner_config``. Nothing here decides a status, a count
or an eligibility rule.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.services.library import CoreMissing

log = logging.getLogger("glossarion.library")

__all__ = [
    "ACTION_LABELS",
    "ActionPlan",
    "GP_GROUPS",
    "GlossaryRowVM",
    "GlossaryView",
    "PM_GROUP_ORDER",
    "ProgressView",
    "RowVM",
    "StatChip",
    "apply_action",
    "glossary_signature",
    "load_glossary_view",
    "load_progress_view",
    "mode_label",
    "plan_action",
    "progress_signature",
    "row_action_ids",
    "row_matches",
    "run_glossary_action",
]

PM_GROUP_ORDER = ("completed", "merged", "in_progress", "pending", "missing", "failed", "skipped")
#: Chips shown even at 0 (UI_SPEC §3.7: Completed and the missing/failed groups).
_PINNED_GROUPS = ("completed", "missing", "failed")
_DEFAULT_GROUPS = {
    "completed": ("completed",), "merged": ("merged",), "in_progress": ("in_progress",), "pending": ("pending",),
    "missing": ("not_translated", "not_refined", "no_tts"), "failed": ("failed", "qa_failed", "refine_failed"),
    "skipped": ("skipped",),
}

#: Glossary chip groups (``glossary_progress_core.GLOSSARY_STATUS_GROUPS`` wins).
GP_GROUPS = {
    "completed": ("completed",),
    "skipped": ("skipped", "skipped_empty", "skipped_image_only", "skipped_title_header_only"),
    "in_progress": ("in_progress", "partially_in_progress"),
    "failed": ("failed", "qa_failed"),
    "merged": ("merged",),
    "remaining": ("not_completed", "not_translated", "no_tts"),
    "not_refined": ("not_refined",),
    "refine_failed": ("refine_failed",),
}
GP_GROUP_ORDER = ("completed", "skipped", "in_progress", "failed", "merged", "remaining", "not_refined",
                  "refine_failed")
_GP_HIDE_AT_ZERO = ("merged", "not_refined", "refine_failed")
_GP_CHIP = {
    "completed": ("✅", "Completed", "completed"),
    "skipped": ("⏭️", "Skipped", "skipped"),
    "in_progress": ("\U0001f504", "In Progress", "in_progress"),
    "failed": ("❌", "Failed", "failed"),
    "merged": ("\U0001f517", "Merged", "merged"),
    "remaining": ("⬜", "Not Translated", "not_translated"),
    "not_refined": ("✨", "Not Refined", "not_refined"),
    "refine_failed": ("\U0001f480", "Refine Failed", "refine_failed"),
}
_GP_LABELS = {
    "completed": "Completed", "skipped": "Skipped", "skipped_empty": "Empty (Skipped)",
    "skipped_image_only": "Image Only (Skipped)", "skipped_title_header_only": "Title/Header Only (Skipped)",
    "failed": "Failed", "qa_failed": "Qa Failed", "merged": "Merged", "in_progress": "In Progress",
    "partially_in_progress": "In Progress", "not_completed": "Not Completed", "not_refined": "Not Refined",
    "refine_failed": "Refine Failed",
}

_MODES = {"text": "Text", "vision": "Vision", "image": "Image", "video": "Video", "audio": "Audio",
          "refinement": "Refinement"}

#: Chapters actions: id -> label (desktop context-menu wording).
ACTION_LABELS: dict = {
    "remove_qa": "\U0001f9f9 Remove QA Failed Mark",
    "remove_pending": "\U0001f9fd Remove Pending Mark",
    "remove_refinement": "⭐ Remove refinement status",
    "restore_in_progress": "Restore In Progress Status",
    "reset_tts": "Reset TTS",
    "delete_audio": "\U0001f5d1️ Delete Audio File",
    "resolve_qa": "⚠️ Resolve QA issue",
    "insert_image": "\U0001f5bc️ Insert Missing Image",
    "do_not_skip": "⏭️ Do not skip",
}
#: ``progress_actions.row_actions`` ids -> our action ids.
_ROW_ACTION_IDS = {
    "remove_qa": "remove_qa", "remove_pending": "remove_pending", "remove_refinement": "remove_refinement",
    "restore_in_progress": "restore_in_progress", "delete_audio": "delete_audio", "resolve_qa": "resolve_qa",
    "insert_missing_image": "insert_image", "do_not_skip": "do_not_skip", "open_file": "open_file",
    "open_audio": "open_audio", "copy_qa": "copy_qa", "open_reader": "open_reader", "retranslate": "retranslate",
}


def mode_label(mode: str) -> str:
    return f"Mode: {_MODES.get(str(mode or 'text').lower(), str(mode or 'Text').title())}"


@dataclass(frozen=True)
class StatChip:
    group: str
    emoji: str
    label: str
    count: int
    status: str  # palette key
    pinned: bool = False

    @property
    def text(self) -> str:
        return f"{self.emoji} {self.label} {self.count}"

    @property
    def visible(self) -> bool:
        return self.pinned or self.count > 0


@dataclass(frozen=True)
class RowVM:
    key: str
    kind: str
    status: str
    icon: str
    label: str
    title: str
    subtitle: str = ""
    badges: tuple = ()
    qa_lines: tuple = ()
    chunk_text: str = ""
    children: tuple = ()
    parent_key: Optional[str] = None
    progress_key: Optional[str] = None
    output_file: str = ""
    filename: str = ""
    is_special: bool = False
    skipped_special: bool = False
    hidden_unless_special: bool = False
    search_text: str = ""
    model: str = ""
    opf_position: Optional[int] = None
    raw: Any = None  # progress_core.RowPresentation

    @property
    def info(self) -> dict:
        info = getattr(self.raw, "info", None)
        return info if isinstance(info, dict) else {}

    @property
    def entry(self) -> dict:
        entry = self.info.get("info")
        return entry if isinstance(entry, dict) else {}


@dataclass(frozen=True)
class ProgressView:
    rows: tuple = ()
    chips: tuple = ()
    total_text: str = ""
    completed: int = 0
    total: int = 0  # countable rows (skipped excluded): the summary strip bar
    mode: str = "text"
    output_dir: str = ""
    progress_file: str = ""
    created: Optional[str] = None
    signature: Any = None
    error: Optional[str] = None
    unreadable: bool = False
    missing: tuple = ()
    notices: tuple = ()
    groups: Mapping[str, tuple] = field(default_factory=lambda: dict(_DEFAULT_GROUPS))
    show_special: bool = False
    show_model_info: bool = True
    state: Any = None  # progress_core.BookProgress

    @property
    def fraction(self) -> Optional[float]:
        return (self.completed / self.total) if self.total else None


@dataclass(frozen=True)
class GlossaryRowVM:
    key: str
    kind: str  # "minimal" | "refinement" | "chapter"
    status: str
    icon: str
    label: str
    title: str
    subtitle: str = ""
    badges: tuple = ()
    qa_lines: tuple = ()
    pinned: bool = False
    index: Optional[int] = None
    raw: Any = None  # glossary_progress_core.GlossaryRow


@dataclass(frozen=True)
class GlossaryView:
    path: Optional[str] = None
    book_title: str = ""
    rows: tuple = ()
    chips: tuple = ()
    total_text: str = ""
    deleted: bool = False
    empty: bool = False
    glossary_file: Optional[str] = None
    signature: Any = None
    error: Optional[str] = None
    missing: tuple = ()
    groups: Mapping[str, tuple] = field(default_factory=lambda: dict(GP_GROUPS))
    state: Any = None  # the glossary progress model


def row_matches(row: Any, query: str) -> bool:
    """Chapters search: titles, file names, chunk text and QA issue strings."""
    if not query:
        return True
    needle = query.casefold()
    hay = " ".join([getattr(row, "search_text", ""), row.title, row.subtitle, getattr(row, "filename", ""),
                    getattr(row, "output_file", ""), getattr(row, "chunk_text", ""), " ".join(row.qa_lines)])
    return needle in hay.casefold()


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------


def _source_for(service: Any, book: Mapping[str, Any]) -> str:
    """The raw source, else a path named like the workspace (never created, never linked)."""
    source = service.raw_source(book)
    if source:
        return source
    folder = str(book.get("output_folder") or "")
    kind = str(book.get("workspace_kind") or "epub").lower()
    ext = {"txt": ".txt", "pdf": ".pdf", "html": ".html"}.get(kind, ".epub")
    return os.path.join(folder, os.path.basename(os.path.normpath(folder)) + ext) if folder else ""


def make_owner(service: Any) -> Any:
    """A ``ProgressOwner`` over the config snapshot; its config writes are saved back sparsely."""
    progress_core = service.core.module("progress_core")
    if progress_core is None or not hasattr(progress_core, "ProgressOwner"):
        raise CoreMissing("progress_core.ProgressOwner")
    return progress_core.ProgressOwner(service.config_snapshot(), save_config=service.save_owner_config)


def _row_vm(pres: Any) -> RowVM:
    info = getattr(pres, "info", {}) or {}
    status = str(getattr(pres, "status", "unknown") or "unknown")
    qa = [str(issue) for issue in (getattr(pres, "qa_issues", ()) or ())]
    more = int(getattr(pres, "qa_more", 0) or 0)
    if more:
        qa.append(f"(+{more} more)")
    filename = str(info.get("original_filename") or "")
    entry = info.get("info") if isinstance(info.get("info"), dict) else {}
    search = [filename, str(getattr(pres, "output_file", "") or ""), str(getattr(pres, "text", "") or ""),
              str(entry.get("translated_title") or "")]
    search.extend(str(issue) for issue in (entry.get("qa_issues_found") or ()))
    try:
        opf_position = int(info["opf_position"]) if info.get("opf_position") is not None else None
    except (TypeError, ValueError):
        opf_position = None
    is_special = bool(getattr(pres, "is_special", False))
    return RowVM(
        key=str(getattr(pres, "row_id", "") or f"row:{getattr(pres, 'progress_key', '') or filename}"),
        kind=str(getattr(pres, "kind", "chapter") or "chapter"),
        status=status,
        icon=str(getattr(pres, "icon", "") or "❓"),
        label=str(getattr(pres, "label", status) or status),
        title=str(getattr(pres, "title", "") or filename),
        subtitle=str(getattr(pres, "subtitle", "") or ""),
        badges=tuple(str(b) for b in (getattr(pres, "badges", ()) or ()) if b),
        qa_lines=tuple(qa),
        chunk_text=str(getattr(pres, "chunk_summary", "") or ""),
        parent_key=getattr(pres, "parent_key", None),
        progress_key=getattr(pres, "progress_key", None),
        output_file=str(getattr(pres, "output_file", "") or ""),
        filename=filename,
        is_special=is_special,
        skipped_special=status == "skipped" and is_special,
        hidden_unless_special=bool(getattr(pres, "hidden", False)),
        search_text=" ".join(s for s in search if s),
        model=str(getattr(pres, "model", "") or ""),
        opf_position=opf_position,
        raw=pres,
    )


def _nest_chunks(rows: Sequence[RowVM]) -> tuple:
    """Chunk rows become ``children`` of their parent row (UI_SPEC §3.7 "↳ Chunk i/T")."""
    out: list = []
    by_progress: dict = {}
    children: dict = {}
    for row in rows:
        if row.kind == "chunk":
            parent = by_progress.get(row.parent_key) if row.parent_key else None
            if parent is None and out:
                parent = len(out) - 1
            if parent is not None:
                children.setdefault(parent, []).append(row)
                continue
        out.append(row)
        if row.progress_key:
            by_progress[row.progress_key] = len(out) - 1
    for index, kids in children.items():
        out[index] = replace(out[index], children=tuple(kids))
    return tuple(out)


def _chips(stats: Any, groups: Mapping[str, tuple]) -> tuple:
    def count(name: str) -> int:
        try:
            return int(getattr(stats, name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    missing_label = str(getattr(stats, "missing_label", "") or "⬜ Not Translated")
    missing_emoji, _, missing_text = missing_label.partition(" ")
    mode = str(getattr(stats, "mode", "text") or "text")
    failed_icon = str(getattr(stats, "failed_icon", "") or "❌")
    failed_label = str(getattr(stats, "failed_label", "") or "Failed")
    spec = {
        "completed": ("✅", "Completed", count("completed"), "completed"),
        "merged": ("\U0001f517", "Merged", count("merged"), "merged"),
        "in_progress": ("\U0001f504", "In Progress", count("in_progress"), "in_progress"),
        "pending": ("❓", "Pending", count("pending"), "pending"),
        "missing": (missing_emoji, missing_text, count("missing"),
                    {"refinement": "not_refined", "audio": "no_tts"}.get(mode, "not_translated")),
        "failed": (failed_icon, failed_label, count("failed"), "refine_failed" if mode == "refinement" else "failed"),
        "skipped": ("⏭️", "Skipped", count("skipped"), "skipped"),
    }
    return tuple(StatChip(group, *spec[group], pinned=group in _PINNED_GROUPS)
                 for group in PM_GROUP_ORDER if group in spec and (group in groups or group in _DEFAULT_GROUPS))


def _view(book_progress: Any, *, groups: Mapping[str, tuple], show_special: bool, show_model: bool,
          created: Optional[str] = None) -> ProgressView:
    stats = book_progress.stats
    rows = _nest_chunks([_row_vm(p) for p in book_progress.rows])
    total = int(getattr(stats, "total", 0) or 0)
    skipped = int(getattr(stats, "skipped", 0) or 0)
    return ProgressView(
        rows=rows,
        chips=_chips(stats, groups),
        total_text=str(getattr(stats, "total_label", "") or f"Total: {total}"),
        completed=int(getattr(stats, "completed", 0) or 0),
        total=max(0, total - skipped),
        mode=str(getattr(stats, "mode", "text") or "text"),
        output_dir=str(book_progress.output_dir or ""),
        progress_file=str(book_progress.progress_file or ""),
        created=created,
        signature=book_progress.signature,
        notices=tuple(getattr(book_progress, "notices", ()) or ()),
        groups=groups,
        show_special=show_special,
        show_model_info=show_model,
        state=book_progress,
    )


def load_progress_view(service: Any, book: Mapping[str, Any], *, show_special: bool = False,
                       show_model_info: bool = True, full: bool = False,
                       previous: Optional[ProgressView] = None) -> ProgressView:
    """Blocking: open (first time) or refresh (``full`` = Refresh, else the read-only tick) a book's view."""
    progress_core = service.core.module("progress_core")
    if progress_core is None or not hasattr(progress_core, "build_book_progress"):
        return ProgressView(error="The Progress Manager core is not available in this build",
                            missing=("progress_core.build_book_progress",))
    groups = {str(k): tuple(v) for k, v in (getattr(progress_core, "STATUS_GROUPS", None) or _DEFAULT_GROUPS).items()}
    output_dir = str(book.get("output_folder") or "")
    previous_state = previous.state if previous is not None else None
    try:
        if previous_state is not None and hasattr(progress_core, "refresh_book_progress"):
            toggles = (previous.show_special != show_special) or (previous.show_model_info != show_model_info)
            if toggles and hasattr(progress_core, "set_view_toggles"):
                progress_core.set_view_toggles(previous_state, show_special_files=show_special,
                                               show_model_info=show_model_info, persist=False)
            book_progress = progress_core.refresh_book_progress(previous_state, read_only=not full,
                                                                force=full or toggles)
            created = None
        else:
            source = _source_for(service, book)
            if not source:
                return ProgressView(error="This book has no output workspace yet")
            owner = make_owner(service)
            book_progress = progress_core.build_book_progress(
                source, owner.config, owner=owner, fixed_output_dir=output_dir or None,
                show_special_files=show_special, show_model_info=show_model_info)
            if book_progress is None:
                notices = owner.take_notices() if hasattr(owner, "take_notices") else []
                message = notices[-1][2] if notices else "The output folder could not be created"
                return ProgressView(output_dir=output_dir, error=str(message), notices=tuple(notices))
            created = getattr(book_progress, "created_folder", None)
    except CoreMissing as exc:
        return ProgressView(output_dir=output_dir, error=str(exc), missing=(exc.name,))
    except (OSError, ValueError) as exc:
        if previous is not None:
            return replace(previous, error="Progress file could not be read — showing last snapshot",
                           unreadable=True)
        return ProgressView(output_dir=output_dir, error=f"Progress file could not be read ({exc})", unreadable=True)
    return _view(book_progress, groups=groups, show_special=show_special, show_model=show_model_info,
                 created=created)


def progress_signature(service: Any, book: Mapping[str, Any], view: Optional[ProgressView] = None) -> Any:
    """Blocking: ``progress_core.snapshot_signature`` of the book's workspace (the 2 s poll)."""
    output_dir = (view.output_dir if view is not None and view.output_dir else str(book.get("output_folder") or ""))
    if not output_dir:
        return None
    progress_file = (view.progress_file if view is not None and view.progress_file
                     else os.path.join(output_dir, "translation_progress.json"))
    fn = service.core.fn("progress_core", "snapshot_signature")
    if fn is None:
        try:
            stat = os.stat(progress_file)
            return ("stat", stat.st_mtime_ns, stat.st_size, len(os.listdir(output_dir)))
        except OSError:
            return ("missing",)
    include_tts = bool(view is not None and str(view.mode).lower() == "audio")
    return fn(progress_file, output_dir, include_tts=include_tts)


def row_action_ids(service: Any, view: ProgressView, row: RowVM, selected: Sequence[RowVM] = ()) -> Optional[set]:
    """Blocking: the actions ``progress_actions.row_actions`` allows on a row (None: unknown)."""
    fn = service.core.fn("progress_actions", "row_actions")
    state = view.state
    if fn is None or state is None:
        return None
    try:
        ids = fn(state.owner, state.data, row.info, [r.info for r in selected] or None)
    except Exception:
        return None
    return {_ROW_ACTION_IDS.get(i, i) for i in ids}


@dataclass
class ActionPlan:
    """What an action will touch, decided by the shared plan helpers before the confirmation."""

    action: str
    targets: list = field(default_factory=list)  # display infos / keys the apply step receives
    count: int = 0
    refusal: Optional[str] = None  # desktop "None of the selected chapters …" text
    extra: dict = field(default_factory=dict)


def plan_action(service: Any, view: ProgressView, action: str, rows: Sequence[RowVM]) -> ActionPlan:
    """Blocking: the targets of an action (desktop selection filters + refusal texts)."""
    pa = service.core.module("progress_actions")
    if pa is None:
        raise CoreMissing("progress_actions")
    state = view.state
    if state is None:
        return ActionPlan(action, refusal="The progress could not be read")
    infos = [r.info for r in rows if r.info]
    prog = state.data.get("prog") or {}
    if not infos:
        return ActionPlan(action, refusal="Please select at least one chapter.")
    if action == "remove_qa":
        targets = pa.plan_remove_qa_marks(prog, infos)
        if not targets:
            return ActionPlan(action, refusal="None of the selected chapters have 'qa_failed' or 'failed' status.")
        return ActionPlan(action, targets, len(targets))
    if action == "remove_refinement":
        keys = pa.refinement_status_keys(prog, infos)
        if not keys:
            return ActionPlan(action, refusal="None of the selected chapters have refinement status.")
        return ActionPlan(action, list(keys), len(keys))
    if action == "restore_in_progress":
        targets = [i for i in infos if i.get("status") == "in_progress"]
        if not targets:
            return ActionPlan(action, refusal="None of the selected chapters have 'in_progress' status.")
        return ActionPlan(action, targets, len(targets))
    if action == "delete_audio":
        info = infos[0]
        path = pa.find_row_audio(state.owner, state.data, info)
        if not path:
            return ActionPlan(action, refusal="No audio file was found for this chapter.")
        return ActionPlan(action, [info], 1, extra={"audio_path": path})
    if action == "resolve_qa":
        info = infos[0]
        entry = info.get("info") if isinstance(info.get("info"), dict) else {}
        request = pa.build_partial_b_request(state.data, info)
        if request is not None:
            return ActionPlan(action, [info], 1, extra={"partial_b": request})
        output = os.path.join(state.data.get("output_dir") or "", str(info.get("output_file") or ""))
        if not entry.get("qa_issues_found") or not os.path.isfile(output):
            return ActionPlan(action, refusal="This chapter has no resolvable QA issue.")
        return ActionPlan(action, [info], 1, extra={"output_path": output})
    if action == "do_not_skip":
        keyword = pa.special_keyword_for(state.owner, infos[0])
        if not keyword:
            return ActionPlan(action, refusal="No special-file keyword matches this file.")
        return ActionPlan(action, [infos[0]], 1, extra={"keyword": keyword})
    # remove_pending / reset_tts / insert_image: the shared action filters the rows itself
    return ActionPlan(action, infos, len(infos))


def apply_action(service: Any, view: ProgressView, plan: ActionPlan, *, restore_fn: Any = None) -> str:
    """Blocking: run the planned action; returns the desktop result text."""
    pa = service.core.module("progress_actions")
    if pa is None:
        raise CoreMissing("progress_actions")
    state = view.state
    data = state.data
    progress_file, output_dir = data["progress_file"], data["output_dir"]
    action = plan.action
    try:
        if action == "remove_qa":
            result = pa.remove_qa_marks(progress_file, output_dir, plan.targets)
            return f"Removed failed mark from {int((result or {}).get('cleared', 0))} chapters."
        if action == "remove_pending":
            return pa.remove_pending_message(pa.remove_pending_marks(progress_file, output_dir, plan.targets))
        if action == "remove_refinement":
            cleared = pa.remove_refinement_status(progress_file, plan.targets)
            return f"Removed refinement status from {int(cleared or 0)} chapter(s)."
        if action == "restore_in_progress":
            return pa.restore_in_progress_message(pa.restore_in_progress(progress_file, output_dir, plan.targets))
        if action == "reset_tts":
            return pa.reset_tts_message(pa.reset_tts(state.owner, progress_file, output_dir, plan.targets))
        if action == "delete_audio":
            pa.delete_row_audio(state.owner, data, plan.targets[0], plan.extra.get("audio_path"))
            return f"Deleted {os.path.basename(str(plan.extra.get('audio_path') or 'the audio file'))}."
        if action == "resolve_qa":
            if plan.extra.get("partial_b") is not None:
                return ("This QA issue needs the translation engine (a Partial.b request on this chapter); "
                        "single-entry QA resolution jobs arrive in U7.")
            output_path = plan.extra["output_path"]
            outcome = pa.resolve_llm_token_qa(progress_file, plan.targets[0], output_path)
            if outcome.get("error"):
                return f"Could not update progress: {outcome['error']}"
            if not (outcome.get("repair") or {}).get("resolved"):
                return str((outcome.get("repair") or {}).get("message") or "No LLM-token issue could be repaired.")
            return pa.llm_token_repair_summary(outcome, output_path)
        if action == "insert_image":
            kind, _title, message, _refreshed = pa.insert_missing_images(data, plan.targets[0], restore_fn)
            return str(message)
        if action == "do_not_skip":
            changed = pa.remove_special_keyword(state.owner, plan.extra["keyword"])
            if changed is False:
                return "Nothing changed"
            return f"⏭️ Do not skip: removed keyword '{plan.extra['keyword']}'"
    finally:
        service.mark_dirty()
    raise ValueError(action)


def action_message(result: Any, fallback: str = "Done") -> str:
    if result is None:
        return fallback
    if isinstance(result, str):
        return result
    for name in ("message", "summary", "text"):
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
        if value:
            return str(value)
    return fallback


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def _gp_row_vm(row: Any, index: int) -> GlossaryRowVM:
    kind_raw = str(getattr(row, "kind", "chapter") or "chapter")
    kind = {"minimal_pass": "minimal"}.get(kind_raw, kind_raw)
    status = str(getattr(row, "status", "not_completed") or "not_completed")
    display = str(getattr(row, "display", "") or "")
    parts = [p.strip() for p in display.split(" | ")] if display else []
    title = parts[0] if parts else str(getattr(row, "filename", "") or "")
    rest = [p for p in parts[2:] if p]
    if kind == "refinement" and rest:
        # "Refinement | ✨ Not Refined | character -> model | 1,234 entries | …" -> "Refinement · character -> model"
        title, rest = f"{title} · {rest[0]}", rest[1:]
    subtitle = " · ".join(rest).replace(" -> ", " → ")
    label = _GP_LABELS.get(status, status.replace("_", " ").title())
    if len(parts) > 1:
        status_part = parts[1]
        icon = str(getattr(row, "icon", "") or "")
        text = status_part[len(icon):].strip() if icon and status_part.startswith(icon) else status_part
        label = text or label
    key = getattr(row, "key", index)
    try:
        chapter_index = int(getattr(row, "chapter_index")) if getattr(row, "chapter_index", None) is not None else None
    except (TypeError, ValueError):
        chapter_index = None
    return GlossaryRowVM(
        key=f"{kind}:{key}",
        kind=kind,
        status=status,
        icon=str(getattr(row, "icon", "") or "⬜"),
        label=label,
        title=title or f"Row {index + 1}",
        subtitle=subtitle,
        qa_lines=tuple(str(i) for i in (getattr(row, "issues", ()) or ())),
        pinned=kind in ("minimal", "refinement"),
        index=chapter_index,
        raw=row,
    )


def _gp_chips(stats: Mapping[str, Any]) -> tuple:
    chips = []
    for group in GP_GROUP_ORDER:
        emoji, label, status = _GP_CHIP[group]
        try:
            count = int((stats or {}).get(group, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        chips.append(StatChip(group, emoji, label, count, status, pinned=group not in _GP_HIDE_AT_ZERO))
    return tuple(chips)


def _find_glossary_progress_path(service: Any, book: Mapping[str, Any],
                                  progress: Optional[ProgressView] = None) -> Optional[str]:
    """Blocking: where the book's glossary progress file is now (None when there is none)."""
    gpc = service.core.module("glossary_progress_core")
    if gpc is None or not hasattr(gpc, "find_glossary_progress"):
        return None
    try:
        state = progress.state if progress is not None else None
        owner = getattr(state, "owner", None) or make_owner(service)
        found = gpc.find_glossary_progress(owner, _source_for(service, book))
    except Exception:
        log.debug("looking up the glossary progress failed", exc_info=True)
        return None
    return str(found) if found else None


def glossary_signature(view: Optional[GlossaryView], service: Any = None, book: Optional[Mapping[str, Any]] = None,
                       progress: Optional[ProgressView] = None) -> Any:
    """Blocking (the Book page's 2 s poll): the glossary progress file's signature in the form
    ``load_glossary_view`` stores (``glossary_progress_core.glossary_progress_signature``:
    ``(mtime_ns, ctime_ns, size)``, None when there is no file).

    A deleted file therefore compares equal to the deleted view (no reload every tick), and the
    empty state (no file yet) looks the file up again, so one written later (Extract glossary)
    is noticed."""
    if view is None:
        return None
    path = view.path
    if not path and view.empty and service is not None and book is not None:
        path = _find_glossary_progress_path(service, book, progress)
    if not path:
        return None
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, getattr(stat, "st_ctime_ns", 0), stat.st_size)
    except OSError:
        return None


def load_glossary_view(service: Any, book: Mapping[str, Any], *, previous: Optional[GlossaryView] = None,
                       progress: Optional[ProgressView] = None) -> GlossaryView:
    """Blocking: the Glossary tab model (``glossary_progress_core``)."""
    gpc = service.core.module("glossary_progress_core")
    title = str(book.get("name") or "")
    if gpc is None or not hasattr(gpc, "open_glossary_progress"):
        return GlossaryView(book_title=title, error="The Glossary Progress core is not available in this build",
                            missing=("glossary_progress_core",))
    groups = {str(k): tuple(v) for k, v in (getattr(gpc, "GLOSSARY_STATUS_GROUPS", None) or GP_GROUPS).items()}
    state = progress.state if progress is not None else None
    owner = getattr(state, "owner", None) or make_owner(service)
    output_dir = str(book.get("output_folder") or "") or None
    source = _source_for(service, book)
    model = previous.state if previous is not None else None
    if model is not None:
        path = model._find_gp_for_file(model.fp) or getattr(model, "gp_path", None)
        if not path or not os.path.isfile(path):
            return GlossaryView(path=previous.path, book_title=previous.book_title or title, rows=previous.rows,
                                chips=previous.chips, total_text=previous.total_text, deleted=True,
                                glossary_file=previous.glossary_file, groups=groups, state=model)
        data = gpc.reload_glossary_progress(model)
    else:
        prog = getattr(state, "data", {}).get("prog") if state is not None else None
        model = gpc.open_glossary_progress(owner, source, output_dir=output_dir, prog=prog)
        if model is None:
            return GlossaryView(book_title=title, empty=True, groups=groups)
        path = model._find_gp_for_file(model.fp) or getattr(model, "gp_path", None)
        data = None
    rows = [_gp_row_vm(row, i) for i, row in enumerate(gpc.glossary_rows(model, data))]
    pinned = [r for r in rows if r.pinned]
    stats = gpc.glossary_stats(model, data) or {}
    glossary_file = None
    try:
        glossary_file = gpc.find_glossary_file(model)
    except Exception:
        glossary_file = None
    book_title = str(getattr(model, "book_title", "") or title)
    return GlossaryView(
        path=str(path) if path else None,
        book_title=book_title,
        rows=tuple(pinned + [r for r in rows if not r.pinned]),
        chips=_gp_chips(stats),
        total_text=f"Total: {int(stats.get('total', 0) or 0)}",
        glossary_file=str(glossary_file) if glossary_file else None,
        signature=gpc.glossary_progress_signature(str(path)) if path else None,
        groups=groups,
        state=model,
    )


def run_glossary_action(service: Any, view: GlossaryView, action: str, rows: Sequence[GlossaryRowVM]) -> Any:
    """Blocking: Mark as completed / Remove from progress / footnotes / completed summary."""
    gpc = service.core.module("glossary_progress_core")
    if gpc is None:
        raise CoreMissing("glossary_progress_core")
    model = view.state
    if model is None:
        raise ValueError("No glossary progress file")
    targets = [r.raw for r in rows]
    if action == "mark_completed":
        result = gpc.mark_glossary_completed(model, targets)
        service.mark_dirty()
        return result
    if action == "remove":
        result = gpc.remove_glossary_progress(model, targets)
        service.mark_dirty()
        return result
    if action == "footnote":
        return gpc.glossary_footnotes(model, [r.index for r in rows if r.index is not None])
    if action == "summary":
        return gpc.write_glossary_summary(model)
    raise ValueError(action)


def filter_rows(rows: Iterable[Any], status_filter: Optional[str], groups: Mapping[str, Sequence[str]]) -> list:
    if not status_filter:
        return list(rows)
    members = tuple(groups.get(status_filter, (status_filter,)))
    return [r for r in rows if r.status in members]
