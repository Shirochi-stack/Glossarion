"""Progress Manager core: the GUI-free half of Retranslation_GUI (mobile milestone U5).

Moved verbatim from ``git show 20b446b0:src/Retranslation_GUI.py`` (line numbers below)
and ``TransateKRtoEN.py``:

* module helpers RG 81-105, 124-228, 292-326, 463-540, 621-647, 947-975, 1275,
  1310-1400, 1494-1672, 1743-1872, 2172-2281: QA classifiers, progress-name
  normalisation, the atomic snapshot writer, the per-path lock, the recursive
  three-way merge and merge-and-write, row display helpers;
* ``ProgressViewMixin``: the RetranslationMixin methods that build, reconcile and
  render the Progress Manager data (RG 17931-18645, 19069-19889, 19891-20013,
  31719-33044, 33066-33338) plus the data halves split out of
  ``_force_retranslation_epub_or_text`` (RG 20801-21779), ``_refresh_retranslation_data``
  (RG 31333-31639) and ``_update_statistics_display`` (RG 33727-33779);
* ``cleanup_missing_files`` = ``TransateKRtoEN.ProgressManager.cleanup_missing_files``
  (TK 6632-6778); the TransateKRtoEN method now delegates here.

Desktop: ``RetranslationMixin(ProgressViewMixin)`` (the mixin sits right after
RetranslationMixin in TranslatorGUI's MRO, so every name resolves to the same code) and
Retranslation_GUI re-exports each moved name.  Mobile: ``ProgressOwner`` (config-backed,
no widgets) + ``build_book_progress`` / ``refresh_book_progress`` / ``compute_book_summary``.

Writes: every progress write goes through the per-path lock, a re-read of the newest
file, the three-way merge and an atomic replace (``mutate_progress`` /
``commit_progress``).  The view's former whole-file writes are listed in
tests/parity/DISCREPANCIES.md (U5).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from chapter_chunk_progress import (
    chunk_failure_summary,
    chunk_status_summary_text,
    effective_parent_status,
    ensure_chunk_entry_schema,
    extract_marked_chunks_for_entry,
    is_multi_chunk_entry,
    reset_chunks_for_retranslation,
    sorted_chunk_items,
)
from chapter_display_numbering import nonreset_chapter_display_numbers
from epub_package import find_epub_opf_member
from metadata_progress import (
    METADATA_PROGRESS_KEY,
    build_metadata_progress_plan,
    is_metadata_progress_entry,
    metadata_field_complete,
    resolve_metadata_field_settings,
)
from pdf_output_naming import (
    move_pdf_output_to_readable_name,
    readable_pdf_section_filename,
)
from translation_artifacts import (
    TRANSLATION_ARTIFACT_SPECS,
    is_translation_artifact_progress_entry,
    translation_artifact_spec_for_kind,
)

_IS_MACOS = (sys.platform == 'darwin')
_PROGRESS_SIDECAR_FILENAMES = frozenset({"source_epub.txt"})
# Files that live in an output folder but are never chapters. The glossary
# extension and the unified glossary copy are written beside glossary.csv, so
# they need the same exclusion or the Progress Manager would list them.
_NON_CHAPTER_OUTPUT_FILENAMES = frozenset({
    "glossary.csv", "glossary.json",
    "glossary_extension.csv", "glossary_extension.md",
    "glossary_extension.txt", "glossary_extension.json",
    "glossary_unified.csv", "glossary_unified.md",
    "glossary_unified.txt", "glossary_unified.json",
    "metadata.json", "styles.css", "rolling_summary.txt", "source_epub.txt",
})
# Filesystem watchers can emit several notifications for one atomic JSON save
# (temporary file creation, replace, directory update). Coalesce those bursts
# before parsing and reconciling a large progress file.
_PROGRESS_WATCH_DEBOUNCE_MS = 500
_PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS = 0.5
_PROGRESS_DIRECT_ROW_UPDATE_LIMIT = 96
_RAW_FOREIGN_TEXT_QA_RE = re.compile(
    r"(?:^|[^a-z])(?:korean|japanese|chinese|hebrew|arabic|syriac|thai|cyrillic)"
    r"_text_found_\d+_chars_",
    re.IGNORECASE,
)

_LLM_TOKEN_QA_RE = re.compile(
    r"(?:^|[^a-z0-9])llm[_\s-]*token[_\s-]*issue",
    re.IGNORECASE,
)
_MISSING_IMAGE_QA_RE = re.compile(
    r"(?:^|[^a-z0-9])missing[_\s-]*images?(?=$|[^a-z0-9])",
    re.IGNORECASE,
)


def _progress_total_label(total, chunks):
    if chunks:
        return f"Total: {total} ({total - chunks} chapters + {chunks} chunks)"
    return f"Total: {total}"


def _sync_parent_chunk_qa_summary(prog, parent_key, chunk_key, output_dir=None):
    """Mirror aggregate child QA state without failing the parent chapter."""
    if not isinstance(prog, dict):
        return None
    chunk_entry = prog.get("chapter_chunks", {}).get(str(chunk_key or ""))
    parent = prog.get("chapters", {}).get(parent_key)
    if not isinstance(chunk_entry, dict) or not isinstance(parent, dict):
        return None
    ensure_chunk_entry_schema(chunk_entry)
    summary = chunk_failure_summary(chunk_entry)
    failed_indices = sorted(
        int(index)
        for index, record in chunk_entry.get("entries", {}).items()
        if isinstance(record, dict)
        and str(record.get("status") or "").lower() in {"qa_failed", "failed"}
    )
    parent["chunk_qa_summary"] = {
        **summary,
        "failed_indices": failed_indices,
        "last_updated": time.time(),
    }
    parent["has_chunk_qa_failures"] = bool(failed_indices)
    parent["chunk_qa_issues_found"] = {
        index: list(record.get("qa_issues_found") or [])
        for index, record in chunk_entry.get("entries", {}).items()
        if isinstance(record, dict) and record.get("qa_issues_found")
    }
    if not failed_indices:
        parent.pop("chunk_qa_issues_found", None)
    status = str(parent.get("status") or "").lower()
    if summary["pending"] and status == "completed":
        parent["status"] = "pending"
        parent["last_updated"] = time.time()
    elif (
        summary["total"] and not summary["pending"] and output_dir
        and (status == "pending" or parent.get("manual_editing_pending"))
        and status != "in_progress"
    ):
        output_path = _pending_mark_output_path(
            {"status": "pending", "info": parent}, output_dir
        )
        blocks = _pending_mark_chunk_blocks(output_path, chunk_key, chunk_entry)
        expected = set(range(1, summary["total"] + 1))
        if set(blocks) == expected:
            _restore_pending_mark_record(parent)
    return summary


def _pending_mark_output_path(info, output_dir):
    """Resolve only this pending row's exact translated HTML output."""
    if not isinstance(info, dict) or str(info.get("status") or "").lower() != "pending":
        return None
    record = info.get("info") or {}
    output = info.get("output_file") or record.get("output_file")
    if not isinstance(output, str) or not output.lower().endswith((".html", ".xhtml", ".htm")):
        return None
    path = os.path.normpath(os.path.join(output_dir, output.replace("\\", "/")))
    return path if os.path.isfile(path) else None


_QA_MARK_FIELDS = (
    "qa_issues", "qa_timestamp", "qa_issues_found", "qa_issue_previews",
    "duplicate_confidence", "failure_reason", "error_message",
)
_CHUNK_QA_MIRROR_FIELDS = (
    "has_chunk_qa_failures", "chunk_qa_summary", "chunk_qa_issues_found",
)


def _chunk_ledger_for_progress_entry(prog, progress_key, entry):
    """Return ``(chunk_key, ledger)`` for a chapter's multi-chunk plan, else ``(chunk_key, None)``."""
    chunk_key = str((entry or {}).get("content_hash") or progress_key or "")
    chunk_entry = prog.get("chapter_chunks", {}).get(chunk_key) if chunk_key else None
    return chunk_key, (chunk_entry if is_multi_chunk_entry(chunk_entry) else None)


def progress_entry_has_qa_mark(prog, progress_key, entry):
    """Whether a chapter row, or any chunk under it, still carries a failed mark."""
    if not isinstance(entry, dict):
        return False
    if str(entry.get("status") or "").lower() in ("qa_failed", "failed"):
        return True
    if entry.get("has_chunk_qa_failures"):
        return True
    _chunk_key, chunk_entry = _chunk_ledger_for_progress_entry(
        prog, progress_key, entry
    )
    return bool(chunk_entry and chunk_failure_summary(chunk_entry)["failed"])


def _pending_mark_chunk_blocks(path, chunk_key, entry):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as source:
            blocks = extract_marked_chunks_for_entry(source.read(), chunk_key, entry)
        total = int(entry.get("total") or 0)
        return {
            index: block for index, block in blocks.items()
            if 1 <= index <= total and block.get("total") == total
            and block.get("content", "").strip()
        }
    except (OSError, ValueError, UnicodeError):
        return {}


def _restore_pending_mark_record(record):
    previous = record.get("previous_progress_entry")
    if isinstance(previous, dict):
        for field in (
            "qa_issues", "qa_issues_found", "qa_issue_previews", "qa_timestamp",
            "duplicate_confidence", "model_name", "key_identifier",
            "refinement_status", "refined_at", "unrefined_backup_file",
        ):
            if field not in record and field in previous:
                record[field] = copy.deepcopy(previous[field])
    has_qa = bool(record.get("qa_issues_found") or record.get("qa_issues"))
    record["status"] = "qa_failed" if has_qa else "completed"
    for field in (
        "manual_editing_pending", "previous_status", "previous_status_unknown",
        "previous_progress_entry", "previous_chunk_metadata",
    ):
        record.pop(field, None)
    record["last_updated"] = time.time()



def _is_progress_sidecar_entry(entry=None, output_file=None):
    """Return whether a progress row points at an internal workspace sidecar."""
    entry = entry if isinstance(entry, dict) else {}
    candidates = (
        output_file,
        entry.get("output_file"),
        entry.get("original_basename"),
        entry.get("filename"),
    )
    return any(
        os.path.basename(str(candidate or "")).casefold()
        in _PROGRESS_SIDECAR_FILENAMES
        for candidate in candidates
        if candidate
    )


def _progress_entry_has_raw_foreign_text_qa(entry):
    """Return whether a progress entry carries a raw foreign-text QA issue."""
    seen = set()
    current = entry
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        values = []
        # chunk_qa_issues_found mirrors chunk-level failures on a completed
        # parent as {chunk index: [issues]}; the dict branch below flattens it.
        for key in (
            'qa_issues_found', 'qa_issues', 'failure_reason', 'error_message',
            'chunk_qa_issues_found',
        ):
            value = current.get(key)
            if isinstance(value, dict):
                values.extend(value.keys())
                values.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                values.extend(value)
            elif value is not None:
                values.append(value)
        for value in values:
            text = str(value or '').strip()
            normalized = text.lower().replace('-', '_').replace(' ', '_')
            if (
                _RAW_FOREIGN_TEXT_QA_RE.search(normalized)
                or ('_text_found_' in normalized and '_chars_' in normalized)
            ):
                return True
        current = current.get('previous_progress_entry')
    return False


def _qa_value_has_llm_token_issue(value):
    """Return whether a progress QA value identifies an LLM-token issue."""
    if isinstance(value, dict):
        return any(
            _qa_value_has_llm_token_issue(key)
            or _qa_value_has_llm_token_issue(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_qa_value_has_llm_token_issue(item) for item in value)
    return bool(_LLM_TOKEN_QA_RE.search(str(value or "")))


def _progress_entry_has_llm_token_qa(entry):
    """Return whether an entry or its restorable snapshot has LLM-token QA."""
    seen = set()
    current = entry
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        for key in (
            'qa_issues_found', 'qa_issues', 'qa_issue_previews',
            'failure_reason', 'error_message',
        ):
            if _qa_value_has_llm_token_issue(current.get(key)):
                return True
        current = current.get('previous_progress_entry')
    return False


def _qa_value_has_missing_image_issue(value):
    """Return whether a structured QA value identifies missing images."""
    if isinstance(value, dict):
        return any(
            _qa_value_has_missing_image_issue(key)
            or _qa_value_has_missing_image_issue(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_qa_value_has_missing_image_issue(item) for item in value)
    return bool(_MISSING_IMAGE_QA_RE.search(str(value or "")))


def _progress_entry_has_missing_image_qa(entry):
    """Return whether an entry or restorable snapshot has missing-image QA."""
    seen = set()
    current = entry
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        for key in (
            'qa_issues_found', 'qa_issues', 'qa_issue_previews',
            'failure_reason', 'error_message',
        ):
            if _qa_value_has_missing_image_issue(current.get(key)):
                return True
        current = current.get('previous_progress_entry')
    return False


@lru_cache(maxsize=32768)
def _normalize_progress_match_text(name):
    """Cached string-only implementation for stable chapter filenames."""
    base = os.path.basename(name)
    if base.startswith("response_"):
        base = base[len("response_"):]
    # This is the hot path for every OPF/progress comparison.  Once basename
    # has removed the directory, rfind preserves splitext's dotfile behavior
    # without repeatedly invoking the much heavier path parser.
    while (dot_index := base.rfind(".")) > 0:
        base = base[:dot_index]
    return base


def _normalize_progress_match_name(name):
    """Normalize source/output names used by Progress Manager matching."""
    if not name:
        return ""
    return _normalize_progress_match_text(str(name))


def _progress_entry_has_meaningful_tts_state(entry):
    """Return whether a non-audio progress entry still needs TTS reconciliation."""
    if not isinstance(entry, dict):
        return False
    if entry.get('tts_file'):
        return True
    tts_status = str(entry.get('tts_status') or '').lower().strip()
    return tts_status not in ('', 'none', 'no_tts')


_PROGRESS_READER_HTML_EXTENSIONS = (".html", ".htm", ".xhtml")


_ARTIFACT_ROW_STATUS_RANK = {
    'completed': 5,
    'in_progress': 4,
    'qa_failed': 3,
    'failed': 2,
    'error': 2,
    'pending': 1,
    'skipped': 0,
}


def _artifact_row_sort_rank(key, entry, canonical_key):
    """Rank a duplicate artifact progress row so the authoritative one wins.

    Ordering (highest wins): status priority (a ``completed`` row beats a
    ``failed`` one), then recency by ``last_updated`` (a newer row supersedes an
    older one whose provenance changed, e.g. claude -> RECYCLED), then the
    canonical progress key as a final tie-breaker.
    """
    entry = entry if isinstance(entry, dict) else {}
    status = str(entry.get('status') or '').strip().lower()
    status_rank = _ARTIFACT_ROW_STATUS_RANK.get(status, 1)
    try:
        last_updated = float(entry.get('last_updated') or 0)
    except (TypeError, ValueError):
        last_updated = 0.0
    is_canonical = 1 if key == canonical_key else 0
    return (status_rank, last_updated, is_canonical)


def _progress_item_is_html(display_info) -> bool:
    """Return whether a Progress Manager row points to an HTML document."""
    display_info = display_info if isinstance(display_info, dict) else {}
    progress_entry = display_info.get("info", {}) or {}
    output_file = (
        display_info.get("output_file")
        or progress_entry.get("output_file")
        or ""
    )
    return str(output_file).lower().endswith(_PROGRESS_READER_HTML_EXTENSIONS)


def _index_epub_html_members(member_names):
    """Build the reusable lookup used to match progress rows to EPUB members."""
    html_members = [
        str(name).replace("\\", "/")
        for name in (member_names or [])
        if str(name).lower().endswith(_PROGRESS_READER_HTML_EXTENSIONS)
    ]
    by_path = {}
    by_basename = {}
    by_stem = {}
    for member in html_members:
        normalized = member.lower().strip("/")
        basename = os.path.basename(normalized)
        stem = _normalize_progress_match_name(basename).lower()
        by_path.setdefault(normalized, member)
        by_basename.setdefault(basename, member)
        if stem:
            by_stem.setdefault(stem, member)
    return by_path, by_basename, by_stem


def _match_epub_html_member_basename(
    member_names,
    candidates,
    member_index=None,
):
    """Match progress filenames to an EPUB HTML member and return its basename.

    ``member_index`` lets callers matching many rows reuse the same lookup.
    Building it for every chapter made the Progress Manager launch path scale
    quadratically with the EPUB chapter count.
    """
    if member_index is None:
        member_index = _index_epub_html_members(member_names)
    by_path, by_basename, by_stem = member_index

    candidate_values = [str(value).replace("\\", "/") for value in (candidates or []) if value]
    for candidate in candidate_values:
        normalized = candidate.lower().strip("/")
        basename = os.path.basename(normalized)
        match = by_path.get(normalized) or by_basename.get(basename)
        if match:
            return os.path.basename(match)
    for candidate in candidate_values:
        stem = _normalize_progress_match_name(candidate).lower()
        match = by_stem.get(stem)
        if match:
            return os.path.basename(match)
    return None


def _snapshot_progress_output_dir(output_dir):
    """Return one reusable directory snapshot for large EPUB matching.

    The old loader repeatedly called ``listdir``/``isfile`` for every spine
    entry.  Keeping names, normalized-name matches, and mtimes together makes
    the initial scan linear in the number of files.
    """
    filenames = set()
    normalized = {}
    mtimes = {}
    try:
        with os.scandir(output_dir) as scan:
            for entry in scan:
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                filenames.add(entry.name)
                normalized.setdefault(_normalize_progress_match_name(entry.name), entry.name)
                try:
                    mtimes[entry.name] = entry.stat().st_mtime
                except OSError:
                    pass
    except OSError:
        pass
    return filenames, normalized, mtimes


def _write_progress_snapshot_atomic(path, payload):
    """Atomically persist a Progress Manager snapshot.

    Windows can briefly deny replacement while the translator, a file
    watcher, or antivirus software has the destination open.  The main
    translation progress writer already tolerates that sharing window; the
    Progress Manager must do the same because its HTML edits happen before
    this commit.
    """
    progress_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(progress_dir, exist_ok=True)
    temp_path = (
        f"{path}.{os.getpid()}.{threading.get_ident()}."
        f"{time.time_ns()}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.flush()
            try:
                os.fsync(target.fileno())
            except OSError:
                pass
        last_error = None
        for attempt in range(20):
            try:
                os.replace(temp_path, path)
                return
            except OSError as exc:
                last_error = exc
                winerror = getattr(exc, "winerror", None)
                retryable = winerror in {5, 32} or (
                    os.name == "nt" and isinstance(exc, PermissionError)
                )
                if not retryable or attempt >= 19:
                    raise
                # Keep this bounded while covering the transient sharing
                # locks commonly produced by Windows scanners/watchers.
                time.sleep(min(0.4, 0.03 * (2 ** min(attempt, 4))))
        if last_error is not None:  # pragma: no cover - loop always returns/raises
            raise last_error
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


_RETRANSLATION_PROGRESS_LOCKS_GUARD = threading.Lock()
_RETRANSLATION_PROGRESS_LOCKS = {}


def _retranslation_progress_lock(path):
    """Return one in-process lock for a Progress Manager JSON file."""
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _RETRANSLATION_PROGRESS_LOCKS_GUARD:
        lock = _RETRANSLATION_PROGRESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _RETRANSLATION_PROGRESS_LOCKS[key] = lock
        return lock


def _merge_retranslation_progress_changes(baseline, changed, latest):
    """Apply only this retranslation action's JSON changes to ``latest``.

    Progress Manager windows can start from different snapshots.  A normal
    full-file write would let the last window overwrite changes made by the
    others.  This recursive three-way merge preserves fields that this action
    did not change while applying its selected chapter resets.
    """
    if baseline == changed:
        return copy.deepcopy(latest)
    if isinstance(baseline, dict) and isinstance(changed, dict):
        merged = copy.deepcopy(latest) if isinstance(latest, dict) else {}
        ordered_keys = list(baseline)
        ordered_keys.extend(key for key in changed if key not in baseline)
        for key in ordered_keys:
            in_baseline = key in baseline
            in_changed = key in changed
            in_latest = isinstance(latest, dict) and key in latest
            if in_baseline and not in_changed:
                # Do not remove a value another action changed after our
                # snapshot; this matters for merged-child chapter entries.
                if not in_latest or latest.get(key) == baseline.get(key):
                    merged.pop(key, None)
                continue
            if not in_baseline:
                merged[key] = copy.deepcopy(changed[key])
                continue
            if not in_latest:
                merged[key] = copy.deepcopy(changed[key])
                continue
            merged[key] = _merge_retranslation_progress_changes(
                baseline[key],
                changed[key],
                latest[key],
            )
        return merged
    return copy.deepcopy(changed)


def _merge_and_write_retranslation_progress(
    path,
    baseline,
    changed,
    *,
    authoritative_chunk_resets=None,
):
    """Merge one selection reset into the newest progress file atomically."""
    with _retranslation_progress_lock(path):
        latest = copy.deepcopy(baseline)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as progress_file:
                disk_progress = json.load(progress_file)
            if isinstance(disk_progress, dict):
                latest = disk_progress
        merged = _merge_retranslation_progress_changes(
            baseline,
            changed,
            latest,
        )
        # Explicit Retranslate Selected chunk resets are authoritative. A
        # translator save racing this dialog must not resurrect the selected
        # result/metadata after the HTML segment has already been removed.
        for reset_spec in authoritative_chunk_resets or []:
            if not isinstance(reset_spec, dict):
                continue
            chunk_key = str(reset_spec.get("chunk_key") or "")
            parent_key = reset_spec.get("parent_key")
            indices = reset_spec.get("indices") or []
            chunk_entry = merged.get("chapter_chunks", {}).get(chunk_key)
            if not isinstance(chunk_entry, dict):
                continue
            reset_chunks_for_retranslation(chunk_entry, indices)
            parent_entry = merged.get("chapters", {}).get(parent_key)
            if isinstance(parent_entry, dict):
                parent_entry["status"] = "pending"
                parent_entry["failure_reason"] = ""
                parent_entry["error_message"] = ""
                parent_entry["last_updated"] = time.time()
                _clear_refinement_progress_fields(parent_entry)
                _sync_parent_chunk_qa_summary(
                    merged,
                    parent_key,
                    chunk_key,
                )
        _write_progress_snapshot_atomic(path, merged)
        return merged


def _persist_progress_manager_source_link(file_path, output_dir, registry_cb=None):
    """Persist the raw input used by a Progress Manager workspace.

    Progress Manager can scaffold an output folder before translation starts.
    The Library scans that folder independently, so it needs the same durable
    source association that a real translation run writes.  Keep both lookup
    routes in sync:

      * ``source_epub.txt`` ties this exact workspace to its input.
      * The Library raw-input registry provides a fallback if the sidecar is
        later removed or the workspace is relocated.

    Returns True when the workspace has a valid sidecar after this call.
    Registry failures are intentionally non-fatal: the exact workspace
    sidecar is sufficient for the Library to recover the EPUB spine.
    """
    if not file_path or not output_dir:
        return False
    try:
        source_path = os.path.abspath(str(file_path))
        workspace = os.path.abspath(str(output_dir))
    except (TypeError, ValueError, OSError):
        return False
    if not os.path.isfile(source_path) or not os.path.isdir(workspace):
        return False

    sidecar = os.path.join(workspace, "source_epub.txt")
    linked = False
    try:
        current = ""
        if os.path.isfile(sidecar):
            with open(sidecar, "r", encoding="utf-8", errors="ignore") as f:
                current = f.read().strip()
        if current:
            current_path = (
                current
                if os.path.isabs(current)
                else os.path.join(workspace, current)
            )
            try:
                linked = (
                    os.path.normcase(os.path.abspath(current_path))
                    == os.path.normcase(source_path)
                )
            except (TypeError, ValueError, OSError):
                linked = False
        if not linked:
            temp_path = (
                f"{sidecar}.{os.getpid()}.{threading.get_ident()}."
                f"{time.time_ns()}.tmp"
            )
            try:
                with open(temp_path, "w", encoding="utf-8") as target:
                    target.write(source_path)
                os.replace(temp_path, sidecar)
                linked = True
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
    except OSError:
        linked = False

    try:
        if callable(registry_cb):
            registry_cb(source_path)
        else:
            from epub_library import record_library_raw_input
            record_library_raw_input(source_path)
    except Exception:
        # Progress Manager must still open if the optional Library registry
        # cannot be loaded or written.  The sidecar remains authoritative.
        pass
    return linked


def _progress_path_signature(path):
    """Cheap signature used to avoid parsing unchanged progress files."""
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _clear_refinement_progress_fields(entry):
    """Remove refinement state from an entry and its restorable snapshot."""
    if not isinstance(entry, dict):
        return 0
    removed = 0
    fields = (
        "refinement_status",
        "refined_at",
        "refinement_error",
        "unrefined_backup_file",
    )
    for field in fields:
        if field in entry:
            entry.pop(field, None)
            removed += 1
    previous_entry = entry.get("previous_progress_entry")
    if isinstance(previous_entry, dict):
        for field in fields:
            if field in previous_entry:
                previous_entry.pop(field, None)
                removed += 1
    return removed


def _progress_entry_refined_for_display(entry):
    current = entry if isinstance(entry, dict) else None
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        status = str(current.get('refinement_status') or '').lower().strip()
        if status:
            return status in ('refined', 'completed')
        current = current.get('previous_progress_entry')
    return False


def _progress_entry_refinement_failed_for_display(entry):
    current = entry if isinstance(entry, dict) else None
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        status = str(current.get('refinement_status') or '').lower().strip()
        if status:
            return status in ('failed', 'error')
        current = current.get('previous_progress_entry')
    return False


def _progress_entry_model_for_display(entry):
    if _progress_entry_is_completed_image_only_for_display(entry):
        return 'COPIED'

    current = entry if isinstance(entry, dict) else None
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        model_name = str(
            current.get('model_name') or current.get('model') or ''
        ).strip()
        if model_name:
            return model_name
        current = current.get('previous_progress_entry')
    return ''


def _progress_status_hides_model_for_display(status):
    """Return whether a row has not started and therefore has no current model."""
    normalized = str(status or '').strip().lower().replace(' ', '_')
    return normalized in {'pending', 'not_translated', 'not_completed'}


def _progress_entry_is_completed_image_only_for_display(entry):
    """Return whether the row's current result is an image-only copy."""
    if not isinstance(entry, dict):
        return False

    # Progress Manager rows wrap the current persisted entry in either ``info``
    # or ``progress_entry``. Do not walk ``previous_progress_entry`` here: a
    # new translation in progress must not inherit the old image-only badge.
    candidates = [entry]
    for key in ('info', 'progress_entry'):
        candidate = entry.get(key)
        if isinstance(candidate, dict):
            candidates.append(candidate)
    return any(
        str(candidate.get('status') or '').lower().strip()
        == 'completed_image_only'
        for candidate in candidates
    )


def _format_qa_issue_for_progress_display(issue, qa_issue_previews=None):
    """Render a QA issue with its saved preview context."""
    issue_code = str(issue)
    previews = qa_issue_previews if isinstance(qa_issue_previews, dict) else {}
    preview = re.sub(
        r'\s+',
        ' ',
        str(previews.get(issue_code, '') or ''),
    ).strip()
    preview_limit = 420 if issue_code.startswith('ai_truncation_detected') else 160
    if len(preview) > preview_limit:
        preview = preview[:preview_limit - 3].rstrip() + '...'
    return (
        f'{issue_code} — Preview: {preview}'
        if preview
        else issue_code
    )


def _select_progress_entry_for_display(entries, display_status=None):
    """Pick the single progress entry that should drive a visible row."""
    candidates = [entry for entry in (entries or []) if isinstance(entry, dict)]
    if not candidates:
        return {}

    desired = str(display_status or '').lower().strip()
    completed_statuses = (
        'completed',
        'completed_empty',
        'completed_image_only',
    )

    def _score(entry):
        status = str(entry.get('status') or '').lower().strip()
        score = 0
        if desired == 'in_progress':
            if status == 'in_progress':
                score += 100
            elif status in completed_statuses:
                score += 20
        elif desired == 'completed':
            if status in completed_statuses:
                score += 100
            if _progress_entry_refined_for_display(entry):
                score += 50
        elif desired in ('failed', 'qa_failed'):
            if status == desired:
                score += 100
            elif status in ('failed', 'qa_failed', 'error'):
                score += 80
        elif desired == 'merged' and status == 'merged':
            score += 100
        elif desired and status == desired:
            score += 100

        if str(entry.get('model_name') or entry.get('model') or '').strip():
            score += 10
        elif _progress_entry_model_for_display(entry):
            score += 3

        try:
            score += min(float(entry.get('last_updated') or 0), 9999999999) / 9999999999
        except Exception:
            pass
        return score

    return max(candidates, key=_score)


# ---------------------------------------------------------------------------
# Status vocabulary (RG 33129-33156 and _apply_progress_list_item_visuals 33340-33358)
# ---------------------------------------------------------------------------

#: Icon per Progress Manager display status (RG _progress_list_display_text).
PM_STATUS_ICONS = {
    'completed': '✅',
    'merged': '🔗',
    'failed': '❌',
    'qa_failed': '❌',
    'refine_failed': '💀',
    'in_progress': '🔄',
    'pending': '❓',
    'not_translated': '⬜',
    'not_refined': '✨',
    'no_tts': '🔊',
    'skipped': '⏭️',
    'unknown': '❓'
}
#: Label per Progress Manager display status (RG _progress_list_display_text).
PM_STATUS_LABELS = {
    'completed': 'Completed',
    'merged': 'Merged',
    'failed': 'Failed',
    'qa_failed': 'QA Failed',
    'refine_failed': 'Refine Failed',
    'in_progress': 'In Progress',
    'pending': 'Pending',
    'not_translated': 'Not Translated',
    'not_refined': 'Not Refined',
    'no_tts': 'No TTS',
    'skipped': 'Skipped',
    'unknown': 'Unknown'
}
#: Row colour per display status (RG _apply_progress_list_item_visuals); 'white' otherwise.
PM_STATUS_COLORS = {
    'completed': 'green',
    'merged': '#17a2b8',
    'failed': 'red',
    'qa_failed': 'red',
    'refine_failed': '#7f5f00',
    'not_translated': '#2b6cb0',
    'not_refined': '#8a63d2',
    'no_tts': '#8a63d2',
    'skipped': '#9aa0a6',
    'in_progress': 'orange',
}
#: (icon, label, colour) per display status: the vocabulary every surface binds to.
STATUS_VOCAB = {
    status: (PM_STATUS_ICONS[status], PM_STATUS_LABELS[status], PM_STATUS_COLORS.get(status, 'white'))
    for status in PM_STATUS_LABELS
}
STATUS_VOCAB['file_missing'] = ('❓', 'file_missing', 'white')
#: Stats-chip groups (the clickable legend of the desktop statistics row).
STATUS_GROUPS = {
    'completed': ('completed',),
    'merged': ('merged',),
    'in_progress': ('in_progress',),
    'pending': ('pending',),
    'missing': ('not_translated', 'not_refined', 'no_tts'),
    'failed': ('failed', 'qa_failed', 'refine_failed'),
    'skipped': ('skipped',),
}


def progress_status_color(status):
    """Desktop row colour of a Progress Manager display status."""
    return PM_STATUS_COLORS.get(status, 'white')


def progress_stats_labels(mode):
    """Legend texts that depend on the output mode (RG 27664-27674 / 33808-33814).

    Returns ``(missing_label, failed_icon, failed_label, failed_color)``.
    """
    missing_label = "✨ Not Refined" if mode == 'refinement' else ("🔊 No TTS" if mode == 'audio' else "⬜ Not Translated")
    failed_icon = "💀" if mode == 'refinement' else "❌"
    failed_label = "Refine Failed" if mode == 'refinement' else "Failed"
    failed_color = "#7f5f00" if mode == 'refinement' else "red"
    return missing_label, failed_icon, failed_label, failed_color


# ---------------------------------------------------------------------------
# Writes: lock -> re-read -> three-way merge -> atomic replace
# ---------------------------------------------------------------------------

#: Public names of the moved I/O helpers.
write_progress_atomic = _write_progress_snapshot_atomic
progress_lock = _retranslation_progress_lock
merge_progress_changes = _merge_retranslation_progress_changes
commit_progress = _merge_and_write_retranslation_progress
snapshot_output_dir = _snapshot_progress_output_dir


def _read_progress_file(path):
    """The newest progress JSON on disk ({} when missing; a non-dict counts as missing)."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as source:
        loaded = json.load(source)
    return loaded if isinstance(loaded, dict) else {}


def mutate_progress(path, fn, *, authoritative_chunk_resets=None):
    """Apply ``fn`` to the newest progress snapshot and persist only its changes.

    Under the per-path lock: re-read ``path``, deep-copy it, call ``fn(prog)`` (which
    mutates ``prog`` in place and returns any value), then three-way merge the change
    into the file as it is at write time and replace it atomically.  Fields ``fn`` did
    not touch keep their newest value, so a translator save that lands while the
    action runs is preserved.  Returns ``fn``'s return value.
    """
    with _retranslation_progress_lock(path):
        baseline = _read_progress_file(path)
        changed = copy.deepcopy(baseline)
        result = fn(changed)
        if changed != baseline:
            _merge_and_write_retranslation_progress(
                path,
                baseline,
                changed,
                authoritative_chunk_resets=authoritative_chunk_resets,
            )
        return result


#: Retry budget of a view commit: the 20 attempts / backoff of the former
#: ``_write_progress_json_safely`` (RG 31394-31421), also bounded in wall time (checked
#: before every retry).  An error the atomic writer already retried is not retried again
#: (``_atomic_write_retried``), so a persistently locked file blocks no longer than the
#: former writer's 20 attempts (8.4-9.0 s).
_VIEW_COMMIT_ATTEMPTS = 20
_VIEW_COMMIT_BUDGET_S = 10.0


def _atomic_write_retried(exc):
    """Whether ``exc`` left ``_write_progress_snapshot_atomic`` after its own retries.

    The atomic writer retries ``os.replace`` 20 times (sleeps totalling about 6.5 s) on
    the errors it treats as transient -- winerror 5 / 32, and any ``PermissionError`` on
    Windows -- and then re-raises the last one; running the whole commit again would
    repeat that wait.  Any other error it raises at once.
    """
    if not isinstance(exc, OSError):
        return False
    winerror = getattr(exc, "winerror", None)
    if not (winerror in {5, 32} or (os.name == "nt" and isinstance(exc, PermissionError))):
        return False
    code = getattr(_write_progress_snapshot_atomic, "__code__", None)
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is code:
            return True
        tb = tb.tb_next
    return False


def _commit_view_progress(path, baseline, prog):
    """Persist the Progress Manager's in-memory snapshot through the three-way merge.

    Replaces the view's whole-file writes (DISCREPANCIES U5): only what changed since
    ``baseline`` (the snapshot as last read or written) reaches the file, so translator
    updates saved meanwhile survive.  A newest file that cannot be parsed is replaced
    by the snapshot (the former behaviour).  Returns the next baseline.

    A sharing violation (PermissionError / OSError) while reading the newest file -- the
    translator replacing the JSON at that moment on Windows -- is retried with the former
    writer's backoff instead of aborting the refresh tick.  The atomic write retries its
    own ``os.replace``; once it gives up, the commit does too.
    """
    import random

    deadline = time.monotonic() + _VIEW_COMMIT_BUDGET_S
    last_error = None
    for attempt in range(_VIEW_COMMIT_ATTEMPTS):
        if last_error is not None and time.monotonic() >= deadline:
            raise last_error
        try:
            try:
                _merge_and_write_retranslation_progress(path, baseline, prog)
            except ValueError:
                _write_progress_snapshot_atomic(path, prog)
            return copy.deepcopy(prog)
        except OSError as exc:
            if attempt >= _VIEW_COMMIT_ATTEMPTS - 1 or _atomic_write_retried(exc):
                raise
            last_error = exc
            time.sleep(min(0.5, 0.03 * (2 ** min(attempt, 5))) + random.uniform(0, 0.03))
    return copy.deepcopy(prog)  # pragma: no cover - the loop returns or raises


def _no_pump(msg=None):
    return None


def cleanup_missing_files(prog, output_dir):
    """Remove missing files and clear merged children of missing parents"""
    cleaned_count = 0
    deleted_parents = set()  # Track which parent chapters were deleted
    parents_with_missing_files = set()  # Track parents with missing files (for merged children clearing)

    # Snapshot the output directory once for this cleanup pass.  The old
    # implementation called ``os.path.exists`` for every progress row and,
    # for every missing row, scanned the whole directory again looking for
    # a response_/HTML-extension rename.  Large EPUBs could therefore turn
    # a single cleanup into thousands of directory scans on the GUI thread.
    #
    # Keep the original top-level lookup semantics (including choosing the
    # first matching renamed file returned by the directory listing).  Only
    # nested/absolute output paths fall back to ``os.path.exists`` because a
    # non-recursive directory snapshot cannot represent those safely.
    _html_exts = {'.html', '.xhtml', '.htm', '.xml'}

    def _norm_cleanup(fn):
        b = os.path.basename(os.fspath(fn))
        if b.startswith('response_'):
            b = b[len('response_'):]
        while True:
            b2, e2 = os.path.splitext(b)
            if e2.lower() in _html_exts:
                b = b2
            else:
                break
        return b.lower()

    try:
        output_names = os.listdir(output_dir)
        output_snapshot_available = True
    except Exception:
        output_names = []
        output_snapshot_available = False

    existing_top_level = {
        os.path.normcase(name)
        for name in output_names
    }
    renamed_html_by_normalized_name = {}
    for name in output_names:
        if name.lower().endswith(('.html', '.xhtml', '.htm')):
            renamed_html_by_normalized_name.setdefault(_norm_cleanup(name), name)

    def _output_exists(output_file):
        output_name = os.fspath(output_file)
        if output_snapshot_available and not os.path.isabs(output_name):
            normalized_name = os.path.normpath(output_name)
            if os.path.dirname(normalized_name) in ('', '.'):
                return os.path.normcase(os.path.basename(normalized_name)) in existing_top_level
        return os.path.exists(os.path.join(output_dir, output_name))
    
    # First pass: Remove entries for missing files (except merged children and certain non-final states)
    for chapter_key, chapter_info in list(prog["chapters"].items()):
        output_file = chapter_info.get("output_file")
        status = chapter_info.get("status")
        status_l = status.lower().strip() if isinstance(status, str) else (str(status).lower().strip() if status is not None else "")
        if (
            is_metadata_progress_entry(chapter_key, chapter_info)
            and output_file
            and not _output_exists(output_file)
        ):
            # metadata.json is a regeneratable shared output. Preserve all
            # of its mode-specific phase rows when the file is deleted.
            if (
                status_l != "pending"
                or not chapter_info.get("metadata_regeneration_requested")
            ):
                chapter_info["status"] = "pending"
                chapter_info["metadata_regeneration_requested"] = True
                chapter_info["last_updated"] = time.time()
            continue
        # MERGED CHAPTERS FIX: Don't delete merged children in first pass
        # They will be handled in second pass if their parent was deleted
        if status == "merged":
            continue

        # Subtitle batch rows share one final SRT/ASS/LRC output. That file is
        # intentionally absent until every batch for the source subtitle
        # has completed, so its absence cannot be used to delete an
        # individual completed batch while Progress Manager is open.
        if (
            chapter_info.get("subtitle_progress_key")
            or chapter_info.get("subtitle_source_file")
            or chapter_info.get("subtitle_output_file")
        ):
            continue

        # PDF outline rows are deliberately created before page content is
        # extracted, so their expected output does not exist yet.
        if chapter_info.get("pdf_outline_seed"):
            continue
        
        # QA_FAILED / FAILED / IN_PROGRESS / PENDING FIX:
        # Don't delete entries that are meant to be visible in the retranslation UI
        # even when their output file is missing.
        # - qa_failed/failed: should remain visible for investigation/retry
        # - in_progress: file doesn't exist yet because translation is ongoing
        # - pending: user explicitly marked for retranslation; file may have been deleted on purpose
        if status_l.startswith("pending") or status_l in ["qa_failed", "failed", "in_progress"]:
            continue
        
        if output_file:
            if not _output_exists(output_file):
                # Before deleting, check if the file was renamed (response_/extension toggle)
                expected_norm = _norm_cleanup(output_file)
                renamed_match = renamed_html_by_normalized_name.get(expected_norm)

                if renamed_match:
                    # File was renamed (retain toggle) – update the stored filename
                    chapter_info['output_file'] = renamed_match
                    continue

                actual_num = chapter_info.get("actual_num")
                if actual_num is not None:
                    # Track if this was a parent of merged chapters
                    deleted_parents.add(actual_num)
                    
                    # Also track if this chapter has merged children (for later clearing)
                    if chapter_info.get("merged_chapters"):
                        parents_with_missing_files.add(actual_num)
                
                # Delete the entry
                del prog["chapters"][chapter_key]
                
                # Remove chunk data
                if chapter_key in prog.get("chapter_chunks", {}):
                    del prog["chapter_chunks"][chapter_key]
                
                cleaned_count += 1
    
    # Second pass: Clear merged children whose parents were deleted OR have missing files
    if deleted_parents or parents_with_missing_files:
        all_affected_parents = deleted_parents | parents_with_missing_files
        for chapter_key, chapter_info in list(prog["chapters"].items()):
            if chapter_info.get("status") == "merged":
                parent_num = chapter_info.get("merged_parent_chapter")
                if parent_num in all_affected_parents:
                    actual_num = chapter_info.get("actual_num")
                    print(f"🔓 Clearing merged child chapter {actual_num} (parent {parent_num} file is missing)")
                    del prog["chapters"][chapter_key]
                    cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"🔄 Removed {cleaned_count} missing file entries")
    return cleaned_count


# ---------------------------------------------------------------------------
# ProgressViewMixin: the Progress Manager's data methods (moved from RetranslationMixin)
# ---------------------------------------------------------------------------


class ProgressViewMixin:
    """GUI-free Progress Manager methods shared by desktop and mobile.

    Hooks (``_show_message``, ``_flash_pm_button_green``, ``_progress_reload_error``,
    ``_progress_cleanup_ready``) have GUI-free defaults here; RetranslationMixin
    overrides each with its original desktop code.
    """

    _RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY = "retranslation_show_model_info"
    _RETRANSLATION_MANUAL_EDITING_CONFIG_KEY = "retranslation_manual_editing"

    # -- hooks ---------------------------------------------------------------

    def _show_message(self, msg_type, title, message, parent=None):
        """GUI-free default: record the notice for the caller (desktop: QMessageBox)."""
        notices = self.__dict__.setdefault('_progress_notices', [])
        notices.append((msg_type, title, message))
        return msg_type != 'question'

    def _flash_pm_button_green(self, folder_path=None):
        """GUI-free default: remember the created folder (desktop also flashes a button)."""
        if folder_path:
            self._pm_created_folder = folder_path

    def _progress_reload_error(self, data, title, message):
        """A refresh could not recreate the progress file (desktop: warning box)."""
        self._show_message('warning', title, message)

    def _progress_cleanup_ready(self):
        """Whether missing-file cleanup may run now (desktop: TransateKRtoEN finished importing)."""
        return True

    def _progress_raw_input_recorder(self):
        """The Library raw-input registry writer used when a view links its source.

        GUI-free default: ``library_core.record_library_raw_input``.  Desktop returns
        None, which keeps the original ``epub_library`` import (lite builds skip it).
        """
        try:
            from library_core import record_library_raw_input
        except Exception:
            return None
        return record_library_raw_input

    # -- settings, special files, seeding (RG 17931-18645) --------------------

    def _get_retranslation_show_model_info_state(self, file_path=None):
        """Return the persisted Show Model Info preference, with live dialog cache first."""
        try:
            if file_path:
                file_key = os.path.abspath(file_path)
                cached = getattr(self, '_retranslation_dialog_cache', {}).get(file_key, {})
                if isinstance(cached, dict) and 'show_model_info_state' in cached:
                    return bool(cached.get('show_model_info_state'))
        except Exception:
            pass
        try:
            return bool(getattr(self, 'config', {}).get(self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY, True))
        except Exception:
            return True

    def _persist_retranslation_show_model_info_state(self, enabled):
        """Persist the Show Model Info preference across app sessions."""
        try:
            if not hasattr(self, 'config') or not isinstance(self.config, dict):
                self.config = {}
            self.config[self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY] = bool(enabled)
            if hasattr(self, 'save_config') and callable(self.save_config):
                self.save_config(show_message=False)
        except Exception as exc:
            try:
                print(f"⚠️ Could not persist Show Model Info state: {exc}")
            except Exception:
                pass

    def _get_retranslation_manual_editing_state(self):
        """Return the persisted Manual editing preference."""
        try:
            return bool(
                getattr(self, 'config', {}).get(
                    self._RETRANSLATION_MANUAL_EDITING_CONFIG_KEY,
                    False,
                )
            )
        except Exception:
            return False

    def _persist_retranslation_manual_editing_state(self, enabled):
        """Persist Manual editing across Progress Manager sessions."""
        try:
            if not hasattr(self, 'config') or not isinstance(self.config, dict):
                self.config = {}
            self.config[self._RETRANSLATION_MANUAL_EDITING_CONFIG_KEY] = bool(enabled)
            if hasattr(self, 'save_config') and callable(self.save_config):
                self.save_config(show_message=False)
        except Exception as exc:
            try:
                print(f"⚠️ Could not persist Manual editing state: {exc}")
            except Exception:
                pass

    def _progress_file_is_skipped_special(self, filename, fallback_is_special=False):
        """Return True only for special files that translation would skip."""
        translate_special = bool(
            getattr(self, 'translate_special_files_var', False)
            or getattr(self, 'config', {}).get('translate_special_files', False)
        )
        if hasattr(self, '_should_skip_special_file'):
            return self._should_skip_special_file(filename, translate_special)
        if translate_special:
            return False
        is_special = fallback_is_special
        if filename and hasattr(self, '_is_special_file'):
            is_special = self._is_special_file(filename)
        if not is_special:
            return False
        translate_all_numbered = bool(
            getattr(self, 'translate_all_numbered_html_var', True)
            or getattr(self, 'config', {}).get('translate_all_numbered_html', True)
        )
        if translate_all_numbered:
            stem = os.path.splitext(os.path.basename(str(filename or '')))[0]
            if stem.lower().startswith('response_'):
                stem = stem[len('response_'):]
            if re.search(r'\d', stem):
                return False
        return True

    def _progress_entry_is_skipped_special(self, ch):
        """True when a Progress Manager entry is a skipped special file —
        i.e. one of the rows the "Show skipped files" toggle reveals."""
        if not isinstance(ch, dict):
            return False
        if self._is_translation_artifact_progress_info(ch):
            nested = ch.get('info') if isinstance(ch.get('info'), dict) else {}
            enabled = ch.get(
                'artifact_translation_enabled',
                nested.get('artifact_translation_enabled', True),
            )
            return not bool(enabled)
        info = ch.get('info') if isinstance(ch.get('info'), dict) else {}
        fname = (
            ch.get('original_filename') or ch.get('filename')
            or ch.get('output_file') or ch.get('href')
            or info.get('original_filename') or info.get('output_file')
            or info.get('key') or ''
        )
        return self._progress_file_is_skipped_special(
            fname, bool(ch.get('is_special') or info.get('is_special')))

    def _metadata_progress_tracking_enabled(self, file_path=None):
        """Whether metadata.json should participate in translation progress."""
        source_path = str(file_path or '').lower()
        if file_path and not source_path.endswith(('.epub', '.pdf')):
            return False
        if source_path.endswith('.pdf'):
            if hasattr(self, 'skip_pdf_title_translation_var'):
                if bool(getattr(self, 'skip_pdf_title_translation_var')):
                    return False
            config = getattr(self, 'config', {})
            if isinstance(config, dict) and bool(
                config.get('skip_pdf_title_translation', False)
            ):
                return False
        if hasattr(self, 'translate_book_title_var'):
            return bool(getattr(self, 'translate_book_title_var'))
        config = getattr(self, 'config', {})
        return bool(config.get('translate_book_title', True)) if isinstance(config, dict) else True

    def _translation_artifact_progress_tracking_enabled(
        self, kind, file_path=None
    ):
        """Whether one generated translation artifact participates in progress."""
        if file_path and not str(file_path).lower().endswith(('.epub', '.pdf')):
            return False
        spec = translation_artifact_spec_for_kind(kind)
        if not spec:
            return False
        attr_name = spec['toggle_attr']
        if hasattr(self, attr_name):
            return bool(getattr(self, attr_name))
        config = getattr(self, 'config', {})
        if not isinstance(config, dict):
            return bool(spec['default_enabled'])
        if spec['toggle_config'] in config:
            return bool(config[spec['toggle_config']])
        fallback_key = spec.get('toggle_fallback_config')
        if fallback_key and fallback_key in config:
            return bool(config[fallback_key])
        return bool(spec['default_enabled'])

    def _zip_is_subtitle_archive(self, file_path):
        """Inspect and cache whether a ZIP is a non-EPUB subtitle archive."""
        path = os.path.abspath(str(file_path or ''))
        if not path.lower().endswith('.zip') or not os.path.isfile(path):
            return False
        try:
            stat_result = os.stat(path)
            signature = (
                int(getattr(stat_result, 'st_mtime_ns', 0)),
                int(stat_result.st_size),
            )
        except OSError:
            return False

        cache = getattr(self, '_subtitle_zip_classification_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._subtitle_zip_classification_cache = cache
        cache_key = os.path.normcase(path)
        cached = cache.get(cache_key)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == signature
        ):
            return bool(cached[1])

        is_subtitle_archive = False
        try:
            from subtitle_processor import SUBTITLE_EXTENSIONS

            with zipfile.ZipFile(path, 'r') as archive:
                infos = archive.infolist()
                # The subtitle extractor rejects archives beyond this limit,
                # so Progress Manager must not classify one it cannot process.
                if len(infos) <= 5000:
                    normalized_names = {
                        str(info.filename or '').replace('\\', '/').lstrip('/').casefold()
                        for info in infos
                    }
                    is_epub_container = (
                        'meta-inf/container.xml' in normalized_names
                        and any(name.endswith('.opf') for name in normalized_names)
                    )
                    if not is_epub_container and 'mimetype' in normalized_names:
                        mimetype_info = next(
                            (
                                info
                                for info in infos
                                if str(info.filename or '').replace('\\', '/').lstrip('/').casefold()
                                == 'mimetype'
                            ),
                            None,
                        )
                        if mimetype_info is not None:
                            is_epub_container = (
                                archive.read(mimetype_info)
                                .decode('ascii', errors='ignore')
                                .strip()
                                == 'application/epub+zip'
                            )
                    if not is_epub_container:
                        is_subtitle_archive = any(
                            not info.is_dir()
                            and os.path.splitext(str(info.filename or ''))[1].lower()
                            in SUBTITLE_EXTENSIONS
                            for info in infos
                        )
        except Exception:
            is_subtitle_archive = False

        # Keep the cache bounded because this mixin can inspect many selections
        # over a long-running GUI session.
        if len(cache) >= 64 and cache_key not in cache:
            cache.clear()
        cache[cache_key] = (signature, is_subtitle_archive)
        return is_subtitle_archive

    def _path_is_subtitle_progress_source(self, file_path):
        """Return whether a source path should use subtitle progress semantics."""
        extension = os.path.splitext(str(file_path or ''))[1].lower()
        if extension in ('.srt', '.ass', '.lrc'):
            return True
        return extension == '.zip' and self._zip_is_subtitle_archive(file_path)

    def _pdf_outline_progress_plan(self, file_path):
        """Read and cache PDF bookmark ranges without extracting page content."""
        pdf_path = os.path.abspath(str(file_path or ''))
        if not pdf_path.lower().endswith('.pdf') or not os.path.isfile(pdf_path):
            return []

        setting = getattr(self, 'pdf_use_toc_sections_var', None)
        if setting is None and isinstance(getattr(self, 'config', None), dict):
            setting = self.config.get('pdf_use_toc_sections', True)
        if isinstance(setting, str):
            setting = setting.strip().lower() not in ('0', 'false', 'no', 'off')
        if setting is False:
            return []

        try:
            stat = os.stat(pdf_path)
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return []

        cache = getattr(self, '_pdf_outline_progress_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._pdf_outline_progress_cache = cache
        cached = cache.get(pdf_path)
        if cached and cached[0] == signature:
            return [dict(section) for section in cached[1]]

        try:
            from pdf_extractor import extract_pdf_toc_section_plan
            plan = extract_pdf_toc_section_plan(pdf_path)
        except Exception as exc:
            print(f"Warning: Could not read PDF bookmarks for Progress Manager: {exc}")
            plan = []

        normalized = [dict(section) for section in plan if isinstance(section, dict)]
        if len(cache) >= 32 and pdf_path not in cache:
            cache.clear()
        cache[pdf_path] = (signature, normalized)
        return [dict(section) for section in normalized]

    def _seed_pdf_outline_progress_entries(self, file_path, output_dir, prog):
        """Seed one Progress Manager row per PDF bookmark section."""
        if not output_dir or not isinstance(prog, dict):
            return False
        sections = self._pdf_outline_progress_plan(file_path)
        if not sections:
            return False

        chapters = prog.setdefault('chapters', {})
        if not isinstance(chapters, dict):
            chapters = {}
            prog['chapters'] = chapters

        retain = (
            os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1'
            or bool((getattr(self, 'config', {}) or {}).get('retain_source_extension', False))
        )
        output_dir = os.path.abspath(str(output_dir))
        source_file = os.path.abspath(str(file_path))
        now = time.time()
        changed = False

        def _norm(value):
            return _normalize_progress_match_name(value).casefold()

        def _set(entry, key, value):
            nonlocal changed
            if entry.get(key) != value:
                entry[key] = value
                changed = True

        def _entry_section_id(progress_key, entry):
            section_id = str(entry.get('pdf_section_id') or '').strip()
            if section_id:
                return section_id
            stored_key = str(entry.get('pdf_progress_key') or progress_key or '')
            if stored_key.startswith('pdf:') and not stored_key.startswith('pdf:outline:'):
                return stored_key.split(':', 2)[1].strip()
            return ''

        for section in sections:
            try:
                section_num = int(section.get('num'))
                start_page = int(section.get('start_page'))
                end_page = int(section.get('end_page'))
            except (TypeError, ValueError):
                continue
            title = str(section.get('title') or f'Section {section_num}').strip()
            level = int(section.get('level') or 0)
            section_id = str(section.get('section_id') or '').strip()
            section_stem = f'pdf_section_{section_num}'
            expected_output = readable_pdf_section_filename(
                {
                    'num': section_num,
                    'title': title,
                    'pdf_toc_title': title,
                    'pdf_section_title': title,
                    'pdf_toc_section': True,
                },
                actual_num=section_num,
                retain=retain,
            )

            exact = []
            chunks = []
            stable_exact = []
            stable_chunks = []
            stale_seeds = []
            for key, entry in list(chapters.items()):
                if not isinstance(entry, dict):
                    continue
                normalized_output = _norm(entry.get('output_file'))
                entry_num = entry.get('actual_num', entry.get('chapter_num'))
                if (
                    entry.get('pdf_outline_seed')
                    and str(entry_num) == str(section_num)
                ):
                    stale_seeds.append((key, entry))
                if section_id and _entry_section_id(key, entry) == section_id:
                    if (
                        str(key).startswith(f'pdf:{section_id}:')
                        or (
                            normalized_output.startswith(
                                f'pdf_section_{section_id}_'
                            )
                            and normalized_output != f'pdf_section_{section_id}'
                        )
                    ):
                        stable_chunks.append((key, entry))
                    else:
                        stable_exact.append((key, entry))
                    continue
                if normalized_output == section_stem:
                    exact.append((key, entry))
                elif normalized_output.startswith(f'{section_stem}_'):
                    chunks.append((key, entry))

            # Once a large bookmark section has real chunk rows, retire its
            # provisional one-row outline seed instead of displaying both.
            if stable_chunks or stable_exact:
                matching = stable_chunks or stable_exact
                matching_keys = {key for key, _entry in matching}
                for key, entry in stale_seeds:
                    if key not in matching_keys and entry.get('pdf_outline_seed'):
                        chapters.pop(key, None)
                        changed = True
            elif chunks:
                for key, entry in exact:
                    if entry.get('pdf_outline_seed'):
                        chapters.pop(key, None)
                        changed = True
                matching = chunks
            else:
                matching = exact

            if not matching:
                for key, entry in stale_seeds:
                    if entry.get('pdf_outline_seed'):
                        chapters.pop(key, None)
                        changed = True
                progress_key = (
                    f'pdf:{section_id}'
                    if section_id
                    else f'pdf:outline:{section_num}'
                )
                output_path = os.path.join(output_dir, expected_output)
                seeded_entry = {
                    'actual_num': section_num,
                    'chapter_num': section_num,
                    'content_hash': '',
                    'output_file': expected_output,
                    'status': 'completed' if os.path.isfile(output_path) else 'not_translated',
                    'last_updated': now,
                    'original_basename': f'{section_stem}.html',
                    'source_file': source_file,
                    'title': title,
                    'pdf_toc_title': title,
                    'pdf_toc_section': True,
                    'pdf_toc_level': level,
                    'pdf_start_page': start_page,
                    'pdf_end_page': end_page,
                    'pdf_outline_seed': True,
                }
                if section_id:
                    seeded_entry['pdf_section_id'] = section_id
                    seeded_entry['pdf_progress_key'] = progress_key
                chapters[progress_key] = seeded_entry
                changed = True
                continue

            for progress_key, entry in matching:
                is_split_entry = bool(
                    section_id
                    and str(progress_key).startswith(f'pdf:{section_id}:')
                )
                if not is_split_entry:
                    occupied_outputs = [
                        other.get('output_file')
                        for other_key, other in chapters.items()
                        if other_key != progress_key and isinstance(other, dict)
                    ]
                    try:
                        mapped_output, _moved = move_pdf_output_to_readable_name(
                            output_dir,
                            entry.get('output_file'),
                            expected_output,
                            occupied=occupied_outputs,
                        )
                    except OSError as exc:
                        mapped_output = entry.get('output_file') or expected_output
                        print(
                            'Warning: Could not rename PDF section output '
                            f"{entry.get('output_file')!r}: {exc}"
                        )
                    _set(entry, 'output_file', mapped_output)
                _set(entry, 'pdf_toc_section', True)
                _set(entry, 'pdf_toc_title', title)
                if not str(entry.get('title') or '').strip():
                    _set(entry, 'title', title)
                _set(entry, 'pdf_toc_level', level)
                _set(entry, 'pdf_start_page', start_page)
                _set(entry, 'pdf_end_page', end_page)
                if section_id:
                    had_stable_metadata = bool(
                        entry.get('pdf_section_id')
                        and entry.get('pdf_progress_key')
                    )
                    _set(entry, 'pdf_section_id', section_id)
                    stable_progress_key = f'pdf:{section_id}'
                    existing_progress_key = str(
                        entry.get('pdf_progress_key') or progress_key or ''
                    )
                    desired_progress_key = (
                        existing_progress_key
                        if existing_progress_key.startswith(
                            f'{stable_progress_key}:'
                        )
                        else stable_progress_key
                    )
                    _set(entry, 'pdf_progress_key', desired_progress_key)
                    # Existing completed rows from the old cache schema have
                    # hashes from before PDF image-reference canonicalization.
                    # Let runtime reconciliation migrate that hash once rather
                    # than treating all bookmarks as changed source content.
                    if (
                        not had_stable_metadata
                        and not entry.get('pdf_outline_seed')
                    ):
                        _set(entry, 'pdf_hash_migration_pending', True)
                if entry.get('pdf_outline_seed'):
                    output_value = entry.get('output_file') or expected_output
                    output_path = (
                        output_value
                        if os.path.isabs(str(output_value))
                        else os.path.join(output_dir, str(output_value))
                    )
                    desired_status = (
                        'completed' if os.path.isfile(output_path) else 'not_translated'
                    )
                    _set(entry, 'status', desired_status)

        return changed

    def _seed_subtitle_zip_progress_entries(self, file_path, output_dir, prog):
        """Seed one Not Translated row per subtitle ZIP member for the PM."""
        archive_path = os.path.abspath(str(file_path or ''))
        if (
            not archive_path.lower().endswith('.zip')
            or not self._zip_is_subtitle_archive(archive_path)
            or not output_dir
            or not isinstance(prog, dict)
        ):
            return False

        try:
            from pathlib import PurePosixPath
            from subtitle_processor import (
                SUBTITLE_EXTENSIONS,
                _safe_archive_component,
            )

            planned = []
            used_output_names = set()
            with zipfile.ZipFile(archive_path, 'r') as archive:
                infos = archive.infolist()
                if len(infos) > 5000:
                    return False
                for info in infos:
                    if info.is_dir():
                        continue
                    archive_name = str(info.filename or '').replace('\\', '/')
                    pure_path = PurePosixPath(archive_name)
                    parts = list(pure_path.parts)
                    extension = os.path.splitext(parts[-1] if parts else '')[1]
                    if extension.lower() not in SUBTITLE_EXTENSIONS:
                        continue
                    if (
                        not parts
                        or pure_path.is_absolute()
                        or re.match(r'^[A-Za-z]:', archive_name)
                        or any(part in ('', '.', '..') for part in parts)
                    ):
                        return False

                    safe_name = _safe_archive_component(
                        parts[-1],
                        f"subtitle_{len(planned) + 1}{extension.lower()}",
                    )
                    safe_stem, safe_extension = os.path.splitext(safe_name)
                    safe_extension = (
                        safe_extension
                        if safe_extension.lower() in SUBTITLE_EXTENSIONS
                        else extension
                    )
                    candidate = f"{safe_stem}{safe_extension}"
                    suffix = 2
                    while candidate.casefold() in used_output_names:
                        candidate = f"{safe_stem}_{suffix}{safe_extension}"
                        suffix += 1
                    used_output_names.add(candidate.casefold())
                    planned.append(
                        {
                            'member_name': archive_name,
                            'source_basename': parts[-1],
                            'output_name': candidate,
                            'crc': int(getattr(info, 'CRC', 0) or 0),
                            'size': int(getattr(info, 'file_size', 0) or 0),
                        }
                    )
        except Exception as exc:
            print(f"Warning: Could not seed subtitle ZIP progress: {exc}")
            return False

        if not planned:
            return False

        output_dir = os.path.abspath(str(output_dir))
        chapters = prog.setdefault('chapters', {})
        if not isinstance(chapters, dict):
            chapters = {}
            prog['chapters'] = chapters
        changed = False
        now = time.time()

        def _normalized(path):
            value = str(path or '').strip()
            if not value:
                return ''
            if not os.path.isabs(value):
                value = os.path.join(output_dir, value)
            return os.path.normcase(os.path.abspath(value))

        for source_index, item in enumerate(planned, start=1):
            final_output = os.path.abspath(
                os.path.join(output_dir, item['output_name'])
            )
            final_norm = _normalized(final_output)
            matching = [
                entry
                for entry in chapters.values()
                if isinstance(entry, dict)
                and self._is_subtitle_progress_entry(entry)
                and _normalized(
                    entry.get('subtitle_output_file')
                    or entry.get('output_file')
                ) == final_norm
            ]
            progress_key = f"subtitle:{item['output_name']}:1"
            if not matching and progress_key not in chapters:
                status = (
                    'completed'
                    if os.path.isfile(final_output)
                    else 'not_translated'
                )
                chapters[progress_key] = {
                    'actual_num': source_index,
                    'content_hash': (
                        f"zip:{item['crc']:08x}:{item['size']}"
                    ),
                    'output_file': final_output,
                    'status': status,
                    'last_updated': now,
                    'original_basename': item['source_basename'],
                    'subtitle_progress_key': progress_key,
                    'subtitle_source_file': (
                        f"{archive_path}!/{item['member_name']}"
                    ),
                    'subtitle_source_batch_num': 1,
                    'subtitle_source_batch_count': 1,
                    'subtitle_bundle_source_index': source_index,
                    'subtitle_output_file': final_output,
                    'subtitle_archive_member': item['member_name'],
                    'subtitle_archive_seed': True,
                    **(
                        {'auto_discovered': True}
                        if status == 'completed'
                        else {}
                    ),
                }
                changed = True

        summaries = prog.setdefault('subtitle_files', {})
        if not isinstance(summaries, dict):
            summaries = {}
            prog['subtitle_files'] = summaries
            changed = True
        for item in planned:
            final_output = os.path.abspath(
                os.path.join(output_dir, item['output_name'])
            )
            final_norm = _normalized(final_output)
            file_entries = [
                (key, entry)
                for key, entry in chapters.items()
                if isinstance(entry, dict)
                and self._is_subtitle_progress_entry(entry)
                and _normalized(
                    entry.get('subtitle_output_file')
                    or entry.get('output_file')
                ) == final_norm
            ]
            if not file_entries:
                continue
            statuses = [
                str(entry.get('status') or 'not_translated').lower()
                for _, entry in file_entries
            ]
            total_batches = max(
                [len(file_entries)]
                + [
                    int(entry.get('subtitle_source_batch_count') or 0)
                    for _, entry in file_entries
                ]
            )
            completed_batches = sum(
                status in ('completed', 'completed_empty', 'completed_image_only')
                for status in statuses
            )
            failed_batches = sum(
                status in ('qa_failed', 'failed', 'error', 'file_missing')
                for status in statuses
            )
            in_progress_batches = sum(
                status in ('in_progress', 'queued')
                for status in statuses
            )
            not_translated_batches = sum(
                status in ('not_translated', 'not translated', 'not_completed')
                for status in statuses
            )
            if failed_batches:
                summary_status = 'failed'
            elif total_batches and completed_batches >= total_batches:
                summary_status = 'completed'
            elif in_progress_batches or completed_batches:
                summary_status = 'in_progress'
            elif total_batches and not_translated_batches >= total_batches:
                summary_status = 'not_translated'
            else:
                summary_status = 'pending'
            first_entry = file_entries[0][1]
            summary = {
                'source_file': first_entry.get('subtitle_source_file', ''),
                'output_file': final_output,
                'status': summary_status,
                'total_batches': total_batches,
                'completed_batches': completed_batches,
                'failed_batches': failed_batches,
                'in_progress_batches': in_progress_batches,
                'not_translated_batches': not_translated_batches,
                'pending_batches': max(
                    0,
                    total_batches
                    - completed_batches
                    - failed_batches
                    - in_progress_batches
                    - not_translated_batches,
                ),
                'batch_keys': [key for key, _ in file_entries],
            }
            if summaries.get(item['output_name']) != summary:
                summaries[item['output_name']] = summary
                changed = True
        return changed

    # -- row kinds, metadata/artifact rows (RG 19069-19889) ------------------

    @staticmethod
    def _is_metadata_progress_info(info):
        if not isinstance(info, dict):
            return False
        nested = info.get('info') if isinstance(info.get('info'), dict) else {}
        entry = info.get('progress_entry') if isinstance(info.get('progress_entry'), dict) else nested
        special_type = info.get('special_type') or entry.get('special_type')
        output_file = info.get('output_file') or entry.get('output_file')
        return special_type == 'metadata' or os.path.basename(str(output_file or '')).lower() == 'metadata.json'

    @staticmethod
    def _is_translation_artifact_progress_info(info):
        if not isinstance(info, dict):
            return False
        nested = info.get('info') if isinstance(info.get('info'), dict) else {}
        entry = (
            info.get('progress_entry')
            if isinstance(info.get('progress_entry'), dict)
            else nested
        )
        key = (
            info.get('progress_key')
            or entry.get('translation_artifact_progress_key')
        )
        output_file = info.get('output_file') or entry.get('output_file')
        return is_translation_artifact_progress_entry(
            key,
            entry or {'output_file': output_file},
        )

    def _progress_entry_needs_special_visibility(self, info):
        """Return whether a row depends on the special-files visibility toggle."""
        if isinstance(info, dict) and info.get('is_subtitle'):
            return False
        if self._is_translation_artifact_progress_info(info):
            nested = info.get('info') if isinstance(info.get('info'), dict) else {}
            enabled = info.get(
                'artifact_translation_enabled',
                nested.get('artifact_translation_enabled', True),
            )
            return not bool(enabled)
        if self._is_metadata_progress_info(info):
            nested = info.get('info') if isinstance(info.get('info'), dict) else {}
            enabled = info.get(
                'metadata_translation_enabled',
                nested.get('metadata_translation_enabled', True),
            )
            # Active metadata is always visible. Disabled metadata behaves like
            # every other skipped special file and follows the visibility toggle.
            return not bool(enabled)
        return self._progress_entry_is_skipped_special(info)

    @staticmethod
    def _is_subtitle_progress_entry(entry, output_file=''):
        """Return True when a progress entry represents an SRT/ASS/LRC batch."""
        if not isinstance(entry, dict):
            return False
        if (
            entry.get('subtitle_progress_key')
            or entry.get('subtitle_source_file')
            or entry.get('subtitle_output_file')
            or entry.get('subtitle_bundle_source_index') is not None
        ):
            return True
        for candidate in (
            output_file,
            entry.get('output_file'),
            entry.get('original_basename'),
        ):
            if os.path.splitext(str(candidate or ''))[1].lower() in ('.srt', '.ass', '.lrc'):
                return True
        return False

    def _build_subtitle_progress_row(self, prog, entries, output_file=''):
        """Collapse subtitle batches into one stable per-file progress row."""
        if not entries:
            return None
        subtitle_entries = [
            (key, entry)
            for key, entry in entries
            if self._is_subtitle_progress_entry(entry, output_file)
        ]
        if not subtitle_entries:
            return None

        first_key, first_entry = subtitle_entries[0]
        final_output = next(
            (
                str(entry.get('subtitle_output_file') or '').strip()
                for _, entry in subtitle_entries
                if str(entry.get('subtitle_output_file') or '').strip()
            ),
            str(output_file or first_entry.get('output_file') or '').strip(),
        )
        summary = {}
        subtitle_files = prog.get('subtitle_files', {}) if isinstance(prog, dict) else {}
        if isinstance(subtitle_files, dict):
            summary = subtitle_files.get(os.path.basename(final_output), {})
            if not isinstance(summary, dict):
                summary = {}
        if summary.get('output_file'):
            final_output = str(summary['output_file'])

        source_file = str(summary.get('source_file') or '').strip()
        if not source_file:
            source_file = next(
                (
                    str(entry.get('subtitle_source_file') or '').strip()
                    for _, entry in subtitle_entries
                    if str(entry.get('subtitle_source_file') or '').strip()
                ),
                '',
            )
        original_filename = os.path.basename(
            source_file
            or str(first_entry.get('original_basename') or '')
            or final_output
        )

        source_indices = []
        for _, entry in subtitle_entries:
            try:
                source_index = int(entry.get('subtitle_bundle_source_index'))
            except (TypeError, ValueError):
                continue
            if source_index > 0:
                source_indices.append(source_index)
        subtitle_index = min(source_indices) if source_indices else 1

        if summary:
            status = str(summary.get('status') or 'pending').lower()
            total_batches = int(summary.get('total_batches') or len(subtitle_entries))
            completed_batches = int(summary.get('completed_batches') or 0)
        else:
            statuses = [
                str(entry.get('status') or 'pending').lower()
                for _, entry in subtitle_entries
            ]
            total_batches = max(
                [len(subtitle_entries)]
                + [
                    int(entry.get('subtitle_source_batch_count') or 0)
                    for _, entry in subtitle_entries
                ]
            )
            completed_batches = sum(
                status in ('completed', 'completed_empty', 'completed_image_only')
                for status in statuses
            )
            if any(status in ('qa_failed', 'failed', 'error', 'file_missing') for status in statuses):
                status = 'failed'
            elif total_batches > 0 and completed_batches >= total_batches:
                status = 'completed'
            elif any(status in ('in_progress', 'queued') for status in statuses) or completed_batches:
                status = 'in_progress'
            else:
                status = 'pending'

        no_translation_required = all(
            bool(entry.get('subtitle_no_translatable_text'))
            for _, entry in subtitle_entries
        )
        if no_translation_required:
            status = 'not_translated'
            completed_batches = 0

        progress_keys = [str(key) for key, _ in subtitle_entries]
        return {
            'key': first_key,
            'num': subtitle_index,
            'info': first_entry,
            'output_file': final_output,
            'status': status,
            'duplicate_count': 1,
            'entries': subtitle_entries,
            'is_special': False,
            'is_subtitle': True,
            'original_filename': original_filename,
            'subtitle_summary': summary,
            'subtitle_total_batches': total_batches,
            'subtitle_completed_batches': completed_batches,
            'progress_key': (
                first_entry.get('subtitle_progress_key') or first_key
            ),
            'progress_keys': progress_keys,
        }

    def _ensure_metadata_progress_entry(self, prog, output_dir, file_path=None):
        """Synchronize special metadata rows with the configured API-call mode."""
        if not isinstance(prog, dict):
            return False
        chapters = prog.setdefault('chapters', {})
        old_entries = {
            key: dict(entry)
            for key, entry in chapters.items()
            if isinstance(entry, dict) and is_metadata_progress_entry(key, entry)
        }
        if not self._metadata_progress_tracking_enabled(file_path):
            for key in old_entries:
                chapters.pop(key, None)
            return bool(old_entries)

        metadata_path = os.path.join(output_dir, 'metadata.json')
        metadata_exists = os.path.isfile(metadata_path)
        if not metadata_exists:
            if old_entries:
                changed = False
                for key, entry in old_entries.items():
                    entry_changed = False
                    if str(entry.get('status', '')).lower() != 'pending':
                        entry['status'] = 'pending'
                        changed = True
                        entry_changed = True
                    if not entry.get('metadata_regeneration_requested'):
                        entry['metadata_regeneration_requested'] = True
                        changed = True
                        entry_changed = True
                    if entry_changed:
                        entry['last_updated'] = time.time()
                    chapters[key] = entry
                return changed
            chapters[METADATA_PROGRESS_KEY] = {
                'actual_num': -1,
                'content_hash': '',
                'output_file': 'metadata.json',
                'original_basename': 'metadata.json',
                'status': 'pending',
                'last_updated': time.time(),
                'is_special': True,
                'special_type': 'metadata',
                'metadata_progress_key': METADATA_PROGRESS_KEY,
                'metadata_mode': self.config.get('metadata_translation_mode', 'together'),
                'metadata_phase': 'combined',
                'metadata_fields': [],
                'metadata_label': 'Metadata',
                'metadata_index': 0,
                'metadata_regeneration_requested': True,
            }
            return True

        try:
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                metadata = json.load(metadata_file)
        except Exception:
            metadata = {}

        field_settings = getattr(self, 'translate_metadata_fields', None)
        if not isinstance(field_settings, dict):
            field_settings = self.config.get('translate_metadata_fields', {})
        mode = self.config.get('metadata_translation_mode', 'together')
        plan = build_metadata_progress_plan(
            mode,
            metadata,
            field_settings,
            source_path=file_path,
        )

        # A compiler-created metadata.json may temporarily contain only
        # structural bookkeeping (language/chapter_count/chapter_titles).  In
        # that state there is no source field from which to build an API phase,
        # but removing the existing phase rows makes Metadata disappear from
        # Progress Manager and loses its retry history.  Preserve existing rows
        # until source metadata is available again.  For an already affected
        # workspace, seed one pending recovery row so the user can select it and
        # regenerate metadata from the source EPUB/PDF.
        if not plan:
            if old_entries:
                return False
            resolved_fields = resolve_metadata_field_settings(
                field_settings,
                file_path,
            )
            requested_fields = [
                str(field)
                for field, enabled in resolved_fields.items()
                if str(field) != '_per_epub' and bool(enabled)
            ]
            chapters[METADATA_PROGRESS_KEY] = {
                'actual_num': -1,
                'content_hash': '',
                'output_file': 'metadata.json',
                'original_basename': 'metadata.json',
                'status': 'pending',
                'last_updated': time.time(),
                'is_special': True,
                'special_type': 'metadata',
                'metadata_progress_key': METADATA_PROGRESS_KEY,
                'metadata_mode': mode,
                'metadata_phase': 'recovery',
                'metadata_fields': requested_fields,
                'metadata_label': 'Metadata',
                'metadata_index': 0,
                'metadata_regeneration_requested': True,
            }
            return True

        legacy_entry = old_entries.get(METADATA_PROGRESS_KEY)
        for key in old_entries:
            chapters.pop(key, None)
        changed = False
        if not plan:
            return bool(old_entries)

        for phase in plan:
            key = phase['key']
            fields = list(phase['fields'])
            existing = old_entries.get(key)
            fallback = legacy_entry if existing is None and len(old_entries) == 1 else None
            previous = existing or fallback or {}
            fields_complete = bool(fields) and all(
                metadata_field_complete(metadata, field) for field in fields
            )
            existing_matches_phase = bool(
                existing is not None
                and list(existing.get('metadata_fields') or []) == fields
                and existing.get('metadata_mode', phase['mode']) == phase['mode']
            )
            if existing_matches_phase:
                status = str(existing.get('status') or 'pending').lower()
            elif fields_complete:
                status = 'completed'
            elif fallback is not None:
                fallback_status = str(fallback.get('status') or 'pending').lower()
                status = 'pending' if fallback_status == 'completed' else fallback_status
            else:
                status = 'pending'
            entry = {
                'actual_num': -1,
                'content_hash': previous.get('content_hash', ''),
                'output_file': 'metadata.json',
                'original_basename': 'metadata.json',
                'status': status,
                'last_updated': previous.get('last_updated', os.path.getmtime(metadata_path)),
                'is_special': True,
                'special_type': 'metadata',
                'metadata_progress_key': key,
                'metadata_mode': phase['mode'],
                'metadata_phase': phase['phase'],
                'metadata_fields': fields,
                'metadata_label': phase['label'],
                'metadata_index': phase['index'],
            }
            for preserved_key in (
                'model_name', 'model', 'key_identifier', 'failure_reason',
                'error_message', 'metadata_regeneration_requested',
                'qa_issues', 'qa_timestamp', 'qa_issues_found',
                'qa_issue_previews', 'duplicate_confidence',
                'refinement_status', 'refined_at', 'refinement_error',
                'unrefined_backup_file',
            ):
                if preserved_key in previous:
                    entry[preserved_key] = previous[preserved_key]
            chapters[key] = entry
            if old_entries.get(key) != entry:
                changed = True
        return changed or set(old_entries) != {phase['key'] for phase in plan}

    def _ensure_translation_artifact_progress_entries(
        self, prog, output_dir, file_path=None
    ):
        """Synchronize TOC/header cache rows with their generating toggles."""
        if not isinstance(prog, dict):
            return False
        chapters = prog.setdefault('chapters', {})
        if not isinstance(chapters, dict):
            chapters = {}
            prog['chapters'] = chapters

        changed = False
        for spec in TRANSLATION_ARTIFACT_SPECS:
            old_entries = {
                key: dict(entry)
                for key, entry in chapters.items()
                if isinstance(entry, dict)
                and (
                    entry.get('special_type') == spec['kind']
                    or os.path.basename(
                        str(entry.get('output_file') or '')
                    ).casefold() == spec['filename'].casefold()
                )
            }
            enabled = self._translation_artifact_progress_tracking_enabled(
                spec['kind'], file_path
            )
            if not enabled:
                for key in old_entries:
                    chapters.pop(key, None)
                changed = changed or bool(old_entries)
                continue

            artifact_path = os.path.join(output_dir, spec['filename'])
            artifact_exists = os.path.isfile(artifact_path)
            previous = (
                old_entries.get(spec['progress_key'])
                or next(iter(old_entries.values()), {})
            )
            previous_status = str(
                previous.get('status') or 'pending'
            ).lower()
            if previous_status in {
                'qa_failed', 'failed', 'error', 'in_progress'
            }:
                status = previous_status
            elif not artifact_exists:
                status = 'pending'
            else:
                status = 'completed'

            content_hash = previous.get('content_hash', '')
            last_updated = previous.get('last_updated', time.time())
            if artifact_exists:
                try:
                    with open(artifact_path, 'rb') as artifact_file:
                        content_hash = hashlib.sha256(
                            artifact_file.read()
                        ).hexdigest()
                    last_updated = os.path.getmtime(artifact_path)
                except OSError:
                    pass

            entry = {
                'actual_num': spec['actual_num'],
                'content_hash': content_hash,
                'output_file': spec['filename'],
                'original_basename': spec['filename'],
                'status': status,
                'last_updated': last_updated,
                'is_special': True,
                'special_type': spec['kind'],
                'translation_artifact_progress_key': spec['progress_key'],
                'translation_artifact_label': spec['label'],
                'artifact_translation_enabled': True,
            }
            for preserved_key in (
                'model_name', 'model', 'key_identifier', 'failure_reason',
                'error_message', 'qa_issues', 'qa_timestamp',
                'qa_issues_found', 'qa_issue_previews',
                'duplicate_confidence', 'refinement_status', 'refined_at',
                'refinement_error', 'unrefined_backup_file',
            ):
                if preserved_key in previous:
                    entry[preserved_key] = previous[preserved_key]

            for key in old_entries:
                chapters.pop(key, None)
            chapters[spec['progress_key']] = entry
            if (
                len(old_entries) != 1
                or old_entries.get(spec['progress_key']) != entry
            ):
                changed = True
        return changed

    def _progress_view_is_subtitle(self, data):
        """Return whether a Progress Manager dataset belongs to subtitles."""
        data = data if isinstance(data, dict) else {}
        if data.get('progress_source_is_subtitle') is True:
            return True
        file_path = str(data.get('file_path') or '')
        extension = os.path.splitext(file_path)[1].lower()
        if self._path_is_subtitle_progress_source(file_path):
            return True

        prog = data.get('prog') if isinstance(data.get('prog'), dict) else {}
        chapters = prog.get('chapters', {})
        cached = data.get('_progress_view_is_subtitle_cache')
        if (
            isinstance(cached, tuple)
            and len(cached) == 4
            and cached[0] is prog
            and cached[1] is chapters
            and cached[2] == file_path
        ):
            return bool(cached[3])

        # EPUB/PDF progress cannot be a subtitle bundle. Avoid probing every
        # chapter entry on each newly loaded JSON object for those large books.
        if extension in ('.epub', '.pdf'):
            data['_progress_view_is_subtitle_cache'] = (
                prog,
                chapters,
                file_path,
                False,
            )
            return False

        is_subtitle = isinstance(prog.get('subtitle_files'), dict)
        if not is_subtitle and isinstance(chapters, dict):
            is_subtitle = any(
                self._is_subtitle_progress_entry(entry)
                for entry in chapters.values()
                if isinstance(entry, dict)
            )
        if is_subtitle:
            data['_progress_view_is_subtitle_cache'] = (
                prog,
                chapters,
                file_path,
                True,
            )
            return True

        # A newly opened subtitle ZIP can precede its first progress entry.
        # In that window, use the extraction session's archive mapping rather
        # than showing an unrelated synthetic metadata row.
        if extension == '.zip' and file_path:
            archive_key = os.path.normcase(os.path.abspath(file_path))
            mappings = getattr(self, '_subtitle_zip_output_groups', None)
            if isinstance(mappings, dict):
                for info in mappings.values():
                    if not isinstance(info, dict):
                        continue
                    mapped_archive = (
                        info.get('archive_path') or info.get('bundle_id')
                    )
                    if (
                        mapped_archive
                        and os.path.normcase(os.path.abspath(str(mapped_archive)))
                        == archive_key
                    ):
                        data['_progress_view_is_subtitle_cache'] = (
                            prog,
                            chapters,
                            file_path,
                            True,
                        )
                        return True
        data['_progress_view_is_subtitle_cache'] = (
            prog,
            chapters,
            file_path,
            False,
        )
        return False

    def _progress_managed_special_entries(self, data):
        """Index metadata/header artifact rows once per progress snapshot."""
        data = data if isinstance(data, dict) else {}
        prog = data.get('prog') if isinstance(data.get('prog'), dict) else {}
        chapters = prog.get('chapters', {})
        cached = data.get('_progress_managed_special_entries_cache')
        if (
            isinstance(cached, tuple)
            and len(cached) == 4
            and cached[0] is prog
            and cached[1] is chapters
        ):
            return cached[2], cached[3]

        metadata_entries = []
        artifact_entries = {
            spec['kind']: [] for spec in TRANSLATION_ARTIFACT_SPECS
        }
        artifact_kind_by_key = {
            spec['progress_key']: spec['kind']
            for spec in TRANSLATION_ARTIFACT_SPECS
        }
        artifact_kind_by_filename = {
            spec['filename'].casefold(): spec['kind']
            for spec in TRANSLATION_ARTIFACT_SPECS
        }
        if isinstance(chapters, dict):
            for key, entry in chapters.items():
                if not isinstance(entry, dict):
                    continue
                key_text = str(key or '')
                if (
                    key_text.startswith('__metadata__')
                    or entry.get('special_type') == 'metadata'
                ):
                    metadata_entries.append((key, entry))
                output_basename = (
                    str(entry.get('output_file') or '')
                    .replace('\\', '/')
                    .rsplit('/', 1)[-1]
                    .casefold()
                )
                special_type = entry.get('special_type')
                artifact_kind = (
                    special_type
                    if special_type in artifact_entries
                    else artifact_kind_by_key.get(key_text)
                    or artifact_kind_by_key.get(
                        entry.get('translation_artifact_progress_key')
                    )
                    or artifact_kind_by_filename.get(output_basename)
                )
                if artifact_kind:
                    artifact_entries[artifact_kind].append((key, entry))

        data['_progress_managed_special_entries_cache'] = (
            prog,
            chapters,
            metadata_entries,
            artifact_entries,
        )
        return metadata_entries, artifact_entries

    def _append_metadata_display_info(self, data, chapter_display_info):
        """Add each tracked metadata API phase as a selectable row."""
        if self._progress_view_is_subtitle(data):
            return
        file_path = str(data.get('file_path') or '')
        # EPUB metadata and the PDF book-title phase share metadata.json.
        if not file_path.lower().endswith(('.epub', '.pdf')):
            return
        metadata_enabled = self._metadata_progress_tracking_enabled(file_path)
        rows = []
        if metadata_enabled:
            entries, _artifact_entries = self._progress_managed_special_entries(
                data
            )
            entries = list(entries)
            if not entries:
                # The managed-entry cache is intentionally keyed by progress
                # snapshot identity.  A non-read-only refresh can add the
                # recovery row to that same dictionary, so fall back to a tiny
                # direct lookup rather than hiding Metadata until the dialog is
                # reopened.
                chapters = (
                    data.get('prog', {}).get('chapters', {})
                    if isinstance(data.get('prog'), dict)
                    else {}
                )
                entries = [
                    (key, entry)
                    for key, entry in chapters.items()
                    if isinstance(entry, dict)
                    and is_metadata_progress_entry(key, entry)
                ]
            entries.sort(key=lambda item: (item[1].get('metadata_index', 999), str(item[0])))
            if not entries:
                # Read-only background snapshots do not mutate progress files.
                # Still render a pending row so an enabled metadata phase never
                # vanishes from the Progress Manager between refreshes.
                entry = {
                    'actual_num': -1,
                    'output_file': 'metadata.json',
                    'original_basename': 'metadata.json',
                    'status': 'pending',
                    'is_special': True,
                    'special_type': 'metadata',
                    'metadata_progress_key': METADATA_PROGRESS_KEY,
                    'metadata_translation_enabled': True,
                    'metadata_label': 'Metadata',
                    'metadata_regeneration_requested': True,
                }
                entries = [(METADATA_PROGRESS_KEY, entry)]
        else:
            # Disabled metadata is a display-only skipped special file. It does
            # not belong in translation_progress.json until translation is on.
            entry = {
                'actual_num': -1,
                'output_file': 'metadata.json',
                'original_basename': 'metadata.json',
                'status': 'skipped',
                'is_special': True,
                'special_type': 'metadata',
                'metadata_translation_enabled': False,
                'metadata_label': 'Metadata',
            }
            entries = [(METADATA_PROGRESS_KEY, entry)]
        metadata_path = os.path.join(data.get('output_dir') or '', 'metadata.json')
        for key, entry in entries:
            status = str(entry.get('status') or ('pending' if metadata_enabled else 'skipped')).lower()
            if not metadata_enabled:
                status = 'skipped'
            elif status == 'completed' and not os.path.isfile(metadata_path):
                status = 'pending'
            rows.append({
                'key': key,
                'num': -1,
                'info': entry,
                'output_file': 'metadata.json',
                'status': status,
                'duplicate_count': 1,
                'entries': [(key, entry)],
                'original_filename': 'metadata.json',
                'is_special': True,
                'special_type': 'metadata',
                'metadata_translation_enabled': metadata_enabled,
                'metadata_label': entry.get('metadata_label', 'Metadata'),
                'metadata_fields': list(entry.get('metadata_fields') or []),
                'progress_key': key if metadata_enabled else None,
            })
        chapter_display_info[0:0] = rows

    def _append_translation_artifact_display_info(
        self, data, chapter_display_info
    ):
        """Add TOC.txt and translated_headers.txt rows after metadata rows."""
        if self._progress_view_is_subtitle(data):
            return
        file_path = str(data.get('file_path') or '')
        if not file_path.lower().endswith(('.epub', '.pdf')):
            return

        _prog = data.get('prog') if isinstance(data.get('prog'), dict) else {}
        prog_chapters = _prog.get('chapters') if isinstance(_prog, dict) else None

        _metadata_entries, artifact_entries = (
            self._progress_managed_special_entries(data)
        )
        # The fallback chapter builder iterates every progress entry, including
        # TOC/header artifacts. Remove those generic rows before inserting the
        # canonical rows. Use the snapshot's tiny set of artifact keys instead
        # of running the general artifact classifier across every chapter.
        artifact_progress_keys = {
            key
            for entries in artifact_entries.values()
            for key, _entry in entries
        }
        artifact_kinds = set(artifact_entries)
        artifact_filenames = {
            spec['filename'].casefold()
            for spec in TRANSLATION_ARTIFACT_SPECS
        }

        def _is_managed_artifact_row(info):
            if not isinstance(info, dict):
                return False
            nested = info.get('info')
            if not isinstance(nested, dict):
                nested = {}
            if (
                info.get('progress_key') in artifact_progress_keys
                or info.get('key') in artifact_progress_keys
                or info.get('special_type') in artifact_kinds
                or nested.get('special_type') in artifact_kinds
            ):
                return True
            output_basename = (
                str(info.get('output_file') or nested.get('output_file') or '')
                .replace('\\', '/')
                .rsplit('/', 1)[-1]
                .casefold()
            )
            return output_basename in artifact_filenames

        chapter_display_info[:] = [
            info for info in chapter_display_info
            if not _is_managed_artifact_row(info)
        ]
        rows = []
        for spec in TRANSLATION_ARTIFACT_SPECS:
            enabled = self._translation_artifact_progress_tracking_enabled(
                spec['kind'], file_path
            )
            entries = artifact_entries.get(spec['kind'], [])
            if enabled:
                if not entries:
                    continue
                # An artifact can carry duplicate progress rows under two key
                # schemes (the canonical ``__translation_artifact__:*`` key and
                # a legacy ``actual_num`` key like ``-2``). One writer updates
                # one row, the QA scanner the other, so the file can hold
                # disagreeing rows — a stale ``failed`` twin outliving a real
                # ``completed`` one, or an outdated ``completed`` row whose
                # provenance has since changed (e.g. claude -> RECYCLED). Keep
                # exactly one authoritative row and drop the rest: rank by
                # status (completed wins), then recency, then the canonical key.
                key, entry = max(
                    entries, key=lambda item: _artifact_row_sort_rank(
                        item[0], item[1], spec['progress_key']
                    )
                )
                if isinstance(prog_chapters, dict):
                    for dup_key, _dup_entry in entries:
                        if dup_key != key:
                            prog_chapters.pop(dup_key, None)
            else:
                key = spec['progress_key']
                entry = {
                    'actual_num': spec['actual_num'],
                    'output_file': spec['filename'],
                    'original_basename': spec['filename'],
                    'status': 'skipped',
                    'is_special': True,
                    'special_type': spec['kind'],
                    'translation_artifact_progress_key': key,
                    'translation_artifact_label': spec['label'],
                    'artifact_translation_enabled': False,
                }

            status = str(
                entry.get('status')
                or ('pending' if enabled else 'skipped')
            ).lower()
            artifact_path = os.path.join(
                data.get('output_dir') or '', spec['filename']
            )
            if not enabled:
                status = 'skipped'
            elif status == 'completed' and not os.path.isfile(artifact_path):
                status = 'pending'
            rows.append({
                'key': key,
                'num': spec['actual_num'],
                'info': entry,
                'output_file': spec['filename'],
                'status': status,
                'duplicate_count': 1,
                'entries': [(key, entry)],
                'original_filename': spec['filename'],
                'is_special': True,
                'special_type': spec['kind'],
                'translation_artifact': True,
                'translation_artifact_label': spec['label'],
                'artifact_translation_enabled': enabled,
                'progress_key': key if enabled else None,
            })

        insert_at = 0
        while (
            insert_at < len(chapter_display_info)
            and self._is_metadata_progress_info(
                chapter_display_info[insert_at]
            )
        ):
            insert_at += 1
        chapter_display_info[insert_at:insert_at] = rows

    _SPECIAL_KEYWORDS_DEFAULT = ('title, toc, copyright, preface, nav, message, '
                                 'notice, colophon, dedication, epigraph, foreword, '
                                 'acknowledgment, author, appendix, bibliography')
    _SPECIAL_EXACT_DEFAULT = 'index, glossary, glossary_extension, glossary_unified'

    def _special_skip_keyword_lists(self):
        """Active special-file keyword lists (mirrors Other Settings)."""
        cfg = getattr(self, 'config', None)
        cfg = cfg if isinstance(cfg, dict) else {}
        kw_raw = (os.environ.get('SPECIAL_FILE_KEYWORDS')
                  or cfg.get('special_file_keywords')
                  or self._SPECIAL_KEYWORDS_DEFAULT)
        exact_raw = (os.environ.get('SPECIAL_FILE_EXACT')
                     or cfg.get('special_file_exact')
                     or self._SPECIAL_EXACT_DEFAULT)
        keywords = [t.strip().lower() for t in str(kw_raw).split(',') if t.strip()]
        exact = [t.strip().lower() for t in str(exact_raw).split(',') if t.strip()]
        return keywords, exact

    @staticmethod
    def _special_skip_stem(filename):
        """Normalize a filename the way the special-file matcher does."""
        base = os.path.basename(str(filename or '')).lower()
        if base.startswith('response_'):
            base = base[len('response_'):]
        html_exts = {'.html', '.xhtml', '.htm', '.txt', '.xml'}
        while True:
            stem, ext = os.path.splitext(base)
            if ext not in html_exts:
                break
            base = stem
        return base

    def _special_skip_keyword_for_filename(self, filename):
        """Return the keyword that makes *filename* a skipped special file."""
        stem = self._special_skip_stem(filename)
        if not stem:
            return None
        keywords, exact = self._special_skip_keyword_lists()
        if stem in exact:
            return stem
        for kw in keywords:
            if kw and kw in stem:
                return kw
        return None

    def _special_skip_keyword_for_progress_info(self, info):
        """Return a skip keyword only for ordinary chapter/file rows."""
        if (
            self._is_metadata_progress_info(info)
            or self._is_translation_artifact_progress_info(info)
        ):
            return None
        filename = (
            info.get('original_filename', '')
            or info.get('output_file', '')
            or info.get('key', '')
        )
        if not self._progress_file_is_skipped_special(
            filename, info.get('is_special', False)
        ):
            return None
        return self._special_skip_keyword_for_filename(filename)

    def _remove_special_skip_keyword(self, keyword):
        """Drop *keyword* from the Other Settings special-file lists.

        Updates the config dict, the live Other Settings vars, and the
        environment so the translation pipeline and every skip check in
        this dialog agree immediately. Returns True when anything changed.
        """
        keyword = str(keyword or '').strip().lower()
        if not keyword:
            return False
        cfg = getattr(self, 'config', None)
        cfg = cfg if isinstance(cfg, dict) else None
        removed = False
        for cfg_key, var_attr, editor_attr, env_key, default in (
            ('special_file_keywords', 'special_file_keywords_var',
             '_special_file_keywords_edit', 'SPECIAL_FILE_KEYWORDS',
             self._SPECIAL_KEYWORDS_DEFAULT),
            ('special_file_exact', 'special_file_exact_var',
             '_special_file_exact_edit', 'SPECIAL_FILE_EXACT',
             self._SPECIAL_EXACT_DEFAULT),
        ):
            raw = (os.environ.get(env_key)
                   or (cfg.get(cfg_key) if cfg else None)
                   or getattr(self, var_attr, None)
                   or default)
            tokens = [t.strip() for t in str(raw).split(',') if t.strip()]
            kept = [t for t in tokens if t.lower() != keyword]
            if len(kept) == len(tokens):
                continue
            new_text = ', '.join(kept)
            if cfg is not None:
                cfg[cfg_key] = new_text
            try:
                setattr(self, var_attr, new_text)
            except Exception:
                pass
            os.environ[env_key] = new_text
            editor = getattr(self, editor_attr, None)
            try:
                if (
                    editor is not None
                    and editor.toPlainText().strip() != new_text
                ):
                    previously_blocked = editor.blockSignals(True)
                    try:
                        editor.setPlainText(new_text)
                    finally:
                        editor.blockSignals(previously_blocked)
            except (AttributeError, RuntimeError):
                # The dialog may not have been built yet, or Qt may already
                # have destroyed an old editor during application shutdown.
                pass
            removed = True
        if removed:
            try:
                self.save_config(show_message=False)
            except Exception:
                pass
        return removed

    # -- the Progress Manager build (data half of _force_retranslation_epub_or_text) --

    def _progress_view_output_dir(self, file_path, resolved_output_dir=None):
        """Return ``(epub_base, override_dir, output_dir)`` for a source (RG 20809-20826)."""
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check for output directory override
        override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
        if not override_dir and hasattr(self, 'config'):
            override_dir = self.config.get('output_directory')
            
        if resolved_output_dir:
            output_dir = os.path.abspath(str(resolved_output_dir))
        elif override_dir:
            output_dir = os.path.join(override_dir, epub_base)
        else:
            output_dir = epub_base
        # On macOS .app bundles, cwd can be '/' (read-only root).
        # Resolve relative output paths against the input file's directory.
        # Only on macOS — on Windows this would change the output dir and break progress tracking.
        if _IS_MACOS and not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), output_dir)
        return epub_base, override_dir, output_dir

    def _ensure_progress_view_workspace(self, output_dir, parent_dialog=None):
        """Create a missing output folder with an empty v2.1 progress file (RG 20828-20842).

        Returns False when the folder could not be created (the build then stops).
        """
        if not os.path.exists(output_dir):
            # Output folder doesn't exist - create it with an empty progress file
            try:
                os.makedirs(output_dir, exist_ok=True)
                empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                progress_file_path = os.path.join(output_dir, "translation_progress.json")
                with open(progress_file_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_prog, f, ensure_ascii=False, indent=2)
                print(f"📁 Created output folder: {output_dir}")
                # Flash the PM button green to signal folder creation
                self._flash_pm_button_green(output_dir)
            except Exception as e:
                if not parent_dialog:
                    self._show_message('error', "Error", f"Could not create output folder: {e}")
                return False
        return True

    def _read_progress_view_spine(self, file_path):
        """Read the source EPUB's OPF spine (RG 21041-21146).

        Returns ``(spine_chapters, opf_chapter_order, is_epub, opf_parsed)``.
        """
        spine_chapters = []
        opf_chapter_order = {}
        is_epub = file_path.lower().endswith('.epub')
        opf_parsed = False

        if is_epub and os.path.exists(file_path):
            try:
                import xml.etree.ElementTree as ET
                import zipfile
                
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Find the package document declared by container.xml.
                    opf_path = None
                    opf_content = None
                    opf_path = find_epub_opf_member(zf)
                    
                    if opf_path:
                        opf_content = zf.read(opf_path)
                        
                        # Parse OPF
                        root = ET.fromstring(opf_content)
                        
                        # Handle namespaces
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        if root.tag.startswith('{'):
                            default_ns = root.tag[1:root.tag.index('}')]
                            ns = {'opf': default_ns}
                        
                        # Get manifest - all chapter files
                        manifest_chapters = {}
                        
                        for item in root.findall('.//opf:manifest/opf:item', ns):
                            item_id = item.get('id')
                            href = item.get('href')
                            media_type = item.get('media-type', '')
                            
                            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                                filename = os.path.basename(href)
                                
                                # Detect special files using configured keyword lists
                                # (mirrors TransateKRtoEN._is_configured_special_file)
                                is_special = self._is_special_file(filename) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename)))
                                
                                # Add all files - UI will handle filtering based on toggle
                                manifest_chapters[item_id] = {
                                    'filename': filename,
                                    'href': href,
                                    'media_type': media_type,
                                    'is_special': is_special
                                }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    is_special = chapter_info.get('is_special', False)
                                    
                                    # Extract chapter number from filename
                                    import re
                                    matches = re.findall(r'(\d+)', filename)
                                    if matches:
                                        file_chapter_num = 0 if is_special else int(matches[-1])
                                    elif is_special:
                                        # Special files without numbers should be chapter 0
                                        file_chapter_num = 0
                                    else:
                                        # Non-numbered OPF files like info.xhtml are
                                        # not real chapter numbers in the progress UI.
                                        file_chapter_num = 0
                                    
                                    # Add all files - UI will handle filtering based on toggle
                                    spine_chapters.append({
                                        'id': idref,
                                        'filename': filename,
                                        'href': chapter_info.get('href'),
                                        'position': len(spine_chapters),
                                        'file_chapter_num': file_chapter_num,
                                        'status': 'unknown',  # Will be updated
                                        'output_file': None,    # Will be updated
                                        'is_special': is_special
                                    })
                                    
                                    # Store the order for later use
                                    opf_chapter_order[filename] = len(spine_chapters) - 1
                                    
                                    # Also store without extension for matching
                                    filename_noext = os.path.splitext(filename)[0]
                                    opf_chapter_order[filename_noext] = len(spine_chapters) - 1
                                    opf_parsed = True
                        
            except Exception as e:
                print(f"Warning: Could not parse OPF: {e}")

        if spine_chapters:
            display_numbers = nonreset_chapter_display_numbers(
                chapter.get('file_chapter_num') for chapter in spine_chapters
            )
            for chapter, display_number in zip(
                spine_chapters, display_numbers
            ):
                chapter['display_chapter_num'] = display_number
        return spine_chapters, opf_chapter_order, is_epub, opf_parsed

    def _build_progress_view_data(
        self,
        file_path,
        parent_dialog=None,
        resolved_output_dir=None,
        _pump_loading=None,
    ):
        """Load, seed, reconcile and match the progress of one source file.

        The data half of ``RetranslationMixin._force_retranslation_epub_or_text``
        (RG 20801-21779, moved verbatim; the writes go through ``_commit_view_progress``).
        Returns the dict the desktop dialog and the mobile Book page are built from, or
        None when the output folder could not be created.
        """
        if _pump_loading is None:
            _pump_loading = _no_pump

        # Classify ZIPs before creating/loading their progress state. This
        # prevents raw ZIP inputs from being assumed to be subtitle archives,
        # while allowing a real subtitle ZIP to suppress EPUB-only artifacts
        # even before extraction has populated its first batch entry.
        progress_source_is_subtitle = self._path_is_subtitle_progress_source(
            file_path
        )

        epub_base, override_dir, output_dir = self._progress_view_output_dir(
            file_path, resolved_output_dir
        )

        if not self._ensure_progress_view_workspace(output_dir, parent_dialog):
            return None

        # Opening Progress Manager is enough to create a Library workspace.
        # Persist its exact raw input now, before OPF parsing, so a Library
        # scan can recover this EPUB's cover and full spine even when no
        # translation run has started yet.
        _persist_progress_manager_source_link(
            file_path, output_dir, registry_cb=self._progress_raw_input_recorder()
        )
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            # No progress file - create empty progress structure
            # This allows fuzzy matching to discover existing files
            print("⚠️ No progress file found - will attempt to discover existing translations")
            prog = {
                "chapters": {},
                "chapter_chunks": {},
                "version": "2.1"
            }
        else:
            with open(progress_file, 'r', encoding='utf-8') as f:
                prog = json.load(f)

        _view_baseline = copy.deepcopy(prog)

        if self._seed_subtitle_zip_progress_entries(
            file_path,
            output_dir,
            prog,
        ):
            try:
                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
                print(
                    "Seeded subtitle ZIP members in Progress Manager before "
                    "translation"
                )
            except Exception as exc:
                print(f"Warning: Could not save seeded subtitle progress: {exc}")

        # Snapshot the output directory once.  Large EPUBs used to rescan and
        # restat this directory for every spine row, turning a 2,000-file book
        # into millions of filesystem operations on the GUI thread.
        (_existing_output_files,
         _normalized_output_files,
         _output_file_mtimes) = _snapshot_progress_output_dir(output_dir)

        # Helper: auto-discover completed files when no OPF is available
        def _auto_discover_from_output_dir(output_dir, prog):
            updated = False
            try:
                # Only exclude _translated.* combined output files when the source
                # file itself does NOT contain "_translated" in its name
                source_has_translated = "_translated" in os.path.basename(file_path).lower()
                files = [
                    f for f in _existing_output_files
                    # accept any extension except known non-chapter files
                    if (source_has_translated or not f.lower().endswith("_translated.txt"))
                    and (source_has_translated or not f.lower().endswith("_translated.pdf"))
                    and (source_has_translated or not f.lower().endswith("_translated.html"))
                    and f != "translation_progress.json"
                    and f.lower() not in _NON_CHAPTER_OUTPUT_FILENAMES
                    and f.casefold() not in _PROGRESS_SIDECAR_FILENAMES
                    and not f.lower().endswith(".epub")
                    and not f.lower().endswith(".cache")
                ]
                tracked_output_files = {
                    os.path.normcase(os.path.abspath(
                        ch.get("output_file")
                        if os.path.isabs(str(ch.get("output_file") or ""))
                        else os.path.join(output_dir, str(ch.get("output_file") or ""))
                    ))
                    for ch in prog.get("chapters", {}).values()
                    if isinstance(ch, dict) and ch.get("output_file")
                }
                subtitle_auto_index = 0
                subtitle_source = progress_source_is_subtitle
                for fname in sorted(files):
                    fname_path_key = os.path.normcase(
                        os.path.abspath(os.path.join(output_dir, fname))
                    )
                    is_subtitle_output = (
                        subtitle_source
                        and os.path.splitext(fname)[1].lower() in ('.srt', '.ass', '.lrc')
                    )
                    if is_subtitle_output:
                        subtitle_auto_index += 1
                    base = os.path.basename(fname)
                    # Normalize by stripping response_ and all extensions
                    if base.startswith("response_"):
                        base = base[len("response_"):]
                    while True:
                        new_base, ext = os.path.splitext(base)
                        if not ext:
                            break
                        base = new_base

                    import re
                    m = re.findall(r"(\d+)", base)
                    chapter_num = (
                        subtitle_auto_index
                        if is_subtitle_output
                        else (int(m[-1]) if m else None)
                    )
                    key = (
                        f"subtitle:auto:{fname}"
                        if is_subtitle_output
                        else (
                            str(chapter_num)
                            if chapter_num is not None
                            else f"special_{base}"
                        )
                    )
                    actual_num = chapter_num if chapter_num is not None else 0

                    if key in prog.get("chapters", {}):
                        continue
                    
                    # Also check if any existing entry already references this output file
                    if fname_path_key in tracked_output_files:
                        continue

                    discovered_entry = {
                        "actual_num": actual_num,
                        "content_hash": "",
                        "output_file": fname,
                        "status": "completed",
                        "last_updated": _output_file_mtimes.get(fname, time.time()),
                        "auto_discovered": True,
                        "original_basename": fname
                    }
                    if is_subtitle_output:
                        discovered_entry.update({
                            "subtitle_progress_key": key,
                            "subtitle_output_file": os.path.abspath(
                                os.path.join(output_dir, fname)
                            ),
                            "subtitle_bundle_source_index": subtitle_auto_index,
                            "subtitle_source_batch_num": 1,
                            "subtitle_source_batch_count": 1,
                        })
                    prog.setdefault("chapters", {})[key] = discovered_entry
                    tracked_output_files.add(fname_path_key)
                    updated = True
            except Exception as e:
                print(f"⚠️ Auto-discovery (no OPF) failed: {e}")
            return updated
        
        # Clean up missing files and merged children when opening the GUI
        # This handles the case where parent files were manually deleted
        # NOTE: resolved without the import lock so opening this GUI can never
        # freeze behind a translation worker that is mid-import of the heavy
        # TransateKRtoEN module (10-20s in frozen builds).
        if self._progress_cleanup_ready():
            cleanup_missing_files(prog, output_dir)

            # Save the cleaned progress back to file
            _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
        else:
            # Module still importing elsewhere — skip cleanup now; the 2s
            # auto-refresh will run it once the module is ready.
            print("⏳ TransateKRtoEN still loading — deferring progress cleanup to next refresh")

        if self._seed_pdf_outline_progress_entries(file_path, output_dir, prog):
            try:
                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
            except Exception as e:
                print(f"Warning: Could not save PDF bookmark progress entries: {e}")

        if self._ensure_metadata_progress_entry(prog, output_dir, file_path):
            try:
                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
            except Exception as e:
                print(f"⚠️ Could not update metadata progress entry: {e}")
        if self._ensure_translation_artifact_progress_entries(
            prog, output_dir, file_path
        ):
            try:
                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
            except Exception as e:
                print(
                    f"⚠️ Could not update TOC/header progress entries: {e}"
                )
        
        _pump_loading("Reading progress data...")
        
        spine_chapters, opf_chapter_order, is_epub, opf_parsed = (
            self._read_progress_view_spine(file_path)
        )

        # If no OPF/spine, fall back to auto-discovery from output_dir
        if not opf_parsed or len(spine_chapters) == 0:
            if _auto_discover_from_output_dir(output_dir, prog):
                try:
                    _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
                    print("💾 Saved auto-discovered progress (no OPF available)")
                except Exception as e:
                    print(f"⚠️ Failed to save auto-discovered progress: {e}")
        else:
            # OPF-AWARE AUTO-DISCOVERY: Use OPF filenames as original_basename
            # This ensures correct mapping between OPF entries and response files
            progress_updated = False
            _progress_chapters = prog.setdefault("chapters", {})
            _progress_by_original = {
                ch.get("original_basename"): ch
                for ch in _progress_chapters.values()
                if isinstance(ch, dict) and ch.get("original_basename")
            }
            _progress_by_output = {
                ch.get("output_file"): ch
                for ch in _progress_chapters.values()
                if isinstance(ch, dict) and ch.get("output_file")
            }
            for spine_ch in spine_chapters:
                opf_filename = spine_ch['filename']  # e.g., "0009_10_.xhtml"
                base_name = os.path.splitext(opf_filename)[0]  # e.g., "0009_10_"
                
                # Look for corresponding response file on disk
                response_file = f"response_{base_name}.html"
                if response_file in _existing_output_files:
                    tracked_info = _progress_by_original.get(opf_filename)
                    if tracked_info is None:
                        tracked_info = _progress_by_output.get(response_file)
                        if tracked_info is not None and tracked_info.get("original_basename") != opf_filename:
                            tracked_info["original_basename"] = opf_filename
                            _progress_by_original[opf_filename] = tracked_info
                            progress_updated = True

                    if tracked_info is None:
                        # Create new progress entry with correct original_basename
                        chapter_num = spine_ch['file_chapter_num']
                        key = str(chapter_num) if chapter_num else f"special_{base_name}"
                        
                        # Avoid duplicate keys
                        if key not in _progress_chapters:
                            new_entry = {
                                "actual_num": chapter_num,
                                "content_hash": "",
                                "output_file": response_file,
                                "status": "completed",
                                "last_updated": _output_file_mtimes.get(response_file, time.time()),
                                "auto_discovered": True,
                                "original_basename": opf_filename  # CORRECT: OPF filename
                            }
                            _progress_chapters[key] = new_entry
                            _progress_by_original[opf_filename] = new_entry
                            _progress_by_output[response_file] = new_entry
                            progress_updated = True
                            print(f"✅ OPF-aware discovery: {opf_filename} -> {response_file}")
            
            if progress_updated:
                try:
                    _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
                    #print("💾 Saved OPF-aware auto-discovered progress")
                except Exception as e:
                    print(f"⚠️ Failed to save progress: {e}")
        
        _pump_loading("Parsing EPUB structure...")
        
        # =====================================================
        # MATCH OPF CHAPTERS WITH TRANSLATION PROGRESS
        # =====================================================
        
        # Helper: normalize filenames for OPF / progress matching
        # We intentionally strip a leading "response_" prefix so that
        # files like "chapter001.xhtml" and "response_chapter001.xhtml"
        # are treated as referring to the same logical entry.
        def _normalize_opf_match_name(name: str) -> str:
            return _normalize_progress_match_name(name)

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build all matching indexes once.  Several of these used to be full
        # progress-dictionary scans inside the spine loop.
        basename_to_progress = {}
        response_file_to_progress = {}
        actualnum_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            progress_ref = (chapter_key, chapter_info)
            original_basename = chapter_info.get("original_basename", "")
            if original_basename:
                norm_key = _normalize_opf_match_name(original_basename)
                basename_to_progress.setdefault(norm_key, []).append(progress_ref)
            output_file = chapter_info.get("output_file", "")
            if output_file:
                response_file_to_progress.setdefault(output_file, []).append(progress_ref)
                norm_key = _normalize_opf_match_name(output_file)
                if norm_key != output_file:
                    response_file_to_progress.setdefault(norm_key, []).append(progress_ref)
            actual_num = chapter_info.get('actual_num')
            if actual_num is None:
                actual_num = chapter_info.get('chapter_num')
            if actual_num is not None:
                actualnum_to_progress.setdefault(actual_num, []).append(progress_ref)

        retain_source_extension = (
            os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1'
            or self.config.get('retain_source_extension', False)
        )

        _n_spine = len(spine_chapters)
        for idx, spine_ch in enumerate(spine_chapters):
            if idx % 80 == 0:
                _pump_loading(f"Matching chapters ({idx + 1}/{_n_spine})...")
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Special files need to check what actually exists on disk
            if is_special:
                # Check for response_ prefix version
                response_with_prefix = f"response_{base_name}.html"
                if retain_source_extension:
                    expected_response = filename
                elif response_with_prefix in _existing_output_files:
                    expected_response = response_with_prefix
                else:
                    # Fallback to original filename
                    expected_response = filename
            else:
                # Use OPF filename directly to avoid mismatching
                if retain_source_extension:
                    expected_response = filename
                else:
                    # Handle .htm.html -> .html conversion
                    stripped_base_name = base_name
                    if base_name.endswith('.htm'):
                        stripped_base_name = base_name[:-4]  # Remove .htm suffix
                    expected_response = filename  # Use exact OPF filename
                    
                    # Also check for response_ prefix version (used by the translator
                    # when TRANSLATE_ALL_NUMBERED_HTML overrides the special-file skip)
                    response_with_prefix = f"response_{base_name}.html"
                    if expected_response not in _existing_output_files and \
                       response_with_prefix in _existing_output_files:
                        expected_response = response_with_prefix
            
            # Check various ways to find the translation progress info
            matched_info = None
            matched_key = None

            def _set_matched_progress(chapter_key, chapter_info):
                nonlocal matched_info, matched_key
                matched_info = chapter_info
                matched_key = chapter_key
            
            # Method 1: Check by original basename (ignoring response_ prefix)
            basename_key = _normalize_opf_match_name(filename)
            if basename_key in basename_to_progress:
                entries = basename_to_progress[basename_key]
                if entries:
                    chapter_key, chapter_info = entries[0]
                    # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                    status = chapter_info.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num:
                            _set_matched_progress(chapter_key, chapter_info)
                    else:
                        _set_matched_progress(chapter_key, chapter_info)
            
            # Method 2: Check by response file (with corrected extension)
            if not matched_info:
                entries = (
                    response_file_to_progress.get(expected_response)
                    or response_file_to_progress.get(_normalize_opf_match_name(expected_response))
                )
                if entries:
                    for chapter_key, chapter_info in entries:
                        # For transient states, also verify actual_num matches.
                        status = chapter_info.get('status', '')
                        if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                            if chapter_info.get('actual_num') != chapter_num:
                                continue
                            _set_matched_progress(chapter_key, chapter_info)
                        else:
                            _set_matched_progress(chapter_key, chapter_info)
                        break
            
            # Method 4: CRUCIAL - Match by chapter number (actual_num vs file_chapter_num)
            # Also check composite keys for special files (e.g., "0_message", "0_TOC")
            if not matched_info:
                # First try simple chapter number key
                simple_key = str(chapter_num)
                if simple_key in prog.get("chapters", {}):
                    chapter_info = prog["chapters"][simple_key]
                    out_file = chapter_info.get('output_file')
                    status = chapter_info.get('status', '')
                    orig_base = chapter_info.get('original_basename', '')
                    if orig_base:
                        orig_base = os.path.basename(orig_base)
                    
                    # Merged chapters: check if parent exists AND original_basename matches
                    if status == 'merged':
                        parent_num = chapter_info.get('merged_parent_chapter')
                        # For merged chapters, match by original_basename (not output_file)
                        # because output_file points to parent's file, not this chapter's source file
                        # Strip extension for comparison since orig_base may not have it
                        filename_noext = os.path.splitext(filename)[0]
                        if parent_num is not None and (
                            _opf_names_equal(orig_base, filename)
                            or _opf_names_equal(orig_base, filename_noext)
                            or not orig_base
                        ):
                            parent_key = str(parent_num)
                            if parent_key in prog.get("chapters", {}):
                                # Just verify parent exists, don't enforce 'completed' status
                                # This ensures we show 'merged' even if parent is completed_empty or other states
                                _set_matched_progress(simple_key, chapter_info)
                    # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                    # to match to avoid cross-matching files.
                    elif status in ['in_progress', 'failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num and (
                            out_file == expected_response or _opf_names_equal(out_file, expected_response)
                        ):
                            _set_matched_progress(simple_key, chapter_info)
                    # qa_failed chapters: match by chapter number only so they are always visible
                    elif status == 'qa_failed':
                        if chapter_info.get('actual_num') == chapter_num:
                            _set_matched_progress(simple_key, chapter_info)
                    # Normal match: output file matches expected (ignoring response_ prefix)
                    elif out_file == expected_response or _opf_names_equal(out_file, expected_response):
                        _set_matched_progress(simple_key, chapter_info)
                
                # If not found, check for composite key (chapter_num + filename)
                if not matched_info and is_special:
                    # For special files, try composite key format: "{chapter_num}_{filename_without_extension}"
                    base_name = os.path.splitext(filename)[0]
                    # Remove "response_" prefix if present in the filename
                    if base_name.startswith("response_"):
                        base_name = base_name[9:]
                    composite_key = f"{chapter_num}_{base_name}"
                    
                    if composite_key in prog.get("chapters", {}):
                        _set_matched_progress(composite_key, prog["chapters"][composite_key])
                
                # Fallback: iterate through all entries matching chapter number,
                # but only accept when it clearly refers to the same source file.
                # This prevents files like "000_information.xhtml" and "0153_0.xhtml"
                # (both parsed as chapter 0) from being conflated.
                if not matched_info:
                    for chapter_key, chapter_info in actualnum_to_progress.get(chapter_num, ()):
                        actual_num = chapter_info.get('actual_num')
                        # Also check 'chapter_num' as fallback
                        if actual_num is None:
                            actual_num = chapter_info.get('chapter_num')
                        
                        if actual_num is not None and actual_num == chapter_num:
                            orig_base = chapter_info.get('original_basename', '')
                            if orig_base:
                                orig_base = os.path.basename(orig_base)
                            out_file = chapter_info.get('output_file')
                            status = chapter_info.get('status', '')
                            qa_issues = chapter_info.get('qa_issues_found', [])
                            
                            # Merged chapters: match by actual_num AND original_basename
                            # For merged, output_file points to parent so we must match by source filename
                            if status == 'merged':
                                parent_num = chapter_info.get('merged_parent_chapter')
                                # Match by original_basename (the source file), not output_file (parent's file)
                                # Strip extension for comparison since orig_base may not have it
                                filename_noext = os.path.splitext(filename)[0]
                                if parent_num is not None and (
                                    _opf_names_equal(orig_base, filename)
                                    or _opf_names_equal(orig_base, filename_noext)
                                    or not orig_base
                                ):
                                    # Check if parent chapter exists
                                    parent_key = str(parent_num)
                                    if parent_key in prog.get("chapters", {}):
                                        # Just verify parent exists, don't enforce 'completed' status
                                        _set_matched_progress(chapter_key, chapter_info)
                                        break
                            
                            # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                            # to match to avoid cross-matching files.
                            if status in ['in_progress', 'failed', 'pending']:
                                if actual_num == chapter_num and (
                                    out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                ):
                                    _set_matched_progress(chapter_key, chapter_info)
                                    break
                            # qa_failed chapters: match by chapter number only so they are always visible,
                            # even when filenames don't line up perfectly.
                            elif status == 'qa_failed':
                                if actual_num == chapter_num:
                                    _set_matched_progress(chapter_key, chapter_info)
                                    break
                            
                            # Only treat as a match for other statuses if the original basename matches
                            # this filename, or, when original_basename is missing, the output_file matches
                            # what we expect.
                            if status not in ['in_progress', 'failed', 'qa_failed', 'pending']:
                                if (
                                    orig_base and _opf_names_equal(orig_base, filename)
                                ) or (
                                    not orig_base and out_file and (
                                        out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                    )
                                ):
                                    _set_matched_progress(chapter_key, chapter_info)
                                    break
            
            # Determine if translation file exists
            file_exists = expected_response in _existing_output_files
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                status = (
                    'pending'
                    if matched_info.get('manual_editing_pending')
                    else matched_info.get('status', 'unknown')
                )
                spine_ch['progress_key'] = matched_key
                
                # CRITICAL: For failed/in_progress/qa_failed/pending, ALWAYS use progress status
                # Never let file existence override these statuses
                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    # Skip all other logic - don't check file existence
                    continue
                
                # For other statuses (completed, merged, etc.)
                spine_ch['status'] = status
                
                # For special files, always use the original filename (ignore what's in progress JSON)
                if is_special:
                    spine_ch['output_file'] = expected_response
                else:
                    spine_ch['output_file'] = matched_info.get('output_file', expected_response)
                
                spine_ch['progress_entry'] = matched_info
                
                # Handle null output_file
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response
                
                # Verify file actually exists for completed status
                if status == 'completed':
                    if spine_ch['output_file'] not in _existing_output_files:
                        # If the expected_response file exists, prefer that and
                        # transparently update the progress entry.
                        if file_exists and expected_response:
                            if expected_response in _existing_output_files:
                                spine_ch['output_file'] = expected_response

                                # If this spine chapter is tied to a concrete
                                # progress entry, keep it consistent.
                                if 'progress_entry' in spine_ch and spine_ch['progress_entry'] is not None:
                                    spine_ch['progress_entry']['output_file'] = expected_response

                                    # matched_key already identifies the master entry.
                                    if matched_key in prog.get('chapters', {}):
                                        prog['chapters'][matched_key]['output_file'] = expected_response
                            else:
                                # No matching file anywhere – mark as missing.
                                spine_ch['status'] = 'not_translated'
                        else:
                            # Legacy behaviour: nothing on disk for this entry.
                            spine_ch['status'] = 'not_translated'
            
            elif file_exists:
                # File exists but no progress tracking - mark as completed
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response
            
            else:
                # No file and no progress tracking - LAST RESORT: Try exact filename matching
                # This handles the case where progress file was deleted but files exist
                # Match by filename only (ignore response_ prefix and all extensions)
                
                # Normalize the OPF filename
                normalized_opf = _normalize_opf_match_name(filename)

                # O(1) lookup in the one directory snapshot instead of one
                # listdir/isfile pass per unmatched spine entry.
                matched_file = _normalized_output_files.get(normalized_opf)
                
                if matched_file:
                    # Found an exact matching file by normalized name - mark as completed
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                    print(f"📁 Matched: {filename} -> {matched_file}")
                else:
                    # No file and no progress tracking - not translated
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS
        # =====================================================
        
        # Check if we discovered any new completed files (exact matched by normalized filename)
        # and add them to the progress file
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']
                
                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                # If the key exists but points to a DIFFERENT file, use a composite
                # key to avoid overwriting (e.g. chapter0003 vs chapter_notice0003).
                existing = prog.get("chapters", {}).get(chapter_key)
                if existing:
                    existing_out = existing.get('output_file', '')
                    existing_base = existing.get('original_basename', '')
                    # If same output file, this is already tracked
                    if existing_out == output_file:
                        continue
                    # Different file occupies this key — use composite key
                    base_noext = os.path.splitext(filename)[0]
                    chapter_key = f"{chapter_num}_{base_noext}"
                    # Also skip if composite key already exists
                    if chapter_key in prog.get("chapters", {}):
                        continue
                
                prog.setdefault("chapters", {})[chapter_key] = {
                    "actual_num": chapter_num,
                    "content_hash": "",  # Unknown since we don't have the source
                    "output_file": output_file,
                    "status": "completed",
                    "last_updated": _output_file_mtimes.get(output_file, time.time()),
                    "auto_discovered": True,
                    "original_basename": filename
                }
                progress_updated = True
                print(f"✅ Auto-discovered and tracked: {filename} -> {output_file} (key: {chapter_key})")
        
        # Save progress file if we added new entries
        if progress_updated:
            try:
                _view_baseline = _commit_view_progress(progress_file, _view_baseline, prog)
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file: {e}")
        
        _pump_loading(f"Building chapter list ({len(prog.get('chapters', {}))} entries)...")
        
        # =====================================================
        # BUILD DISPLAY INFO
        # =====================================================
        
        chapter_display_info = []
        
        if spine_chapters:
            # Use OPF order
            for spine_ch in spine_chapters:
                display_info = {
                    'key': spine_ch.get('filename', ''),
                    'num': spine_ch['file_chapter_num'],
                    'display_num': spine_ch.get(
                        'display_chapter_num', spine_ch['file_chapter_num']
                    ),
                    'info': spine_ch.get('progress_entry', {}),
                    'output_file': spine_ch['output_file'],
                    'status': spine_ch['status'],
                    'duplicate_count': 1,
                    'entries': [],
                    'opf_position': spine_ch['position'],
                    'original_filename': spine_ch['filename'],
                    'is_special': spine_ch.get('is_special', False),
                    'progress_key': spine_ch.get('progress_key')
                }
                chapter_display_info.append(display_info)
        else:
            # Fallback to original logic if no OPF
            # Known non-chapter files that should never appear in the progress list
            _non_chapter_files = _NON_CHAPTER_OUTPUT_FILENAMES
            _source_has_translated = "_translated" in os.path.basename(file_path).lower()
            files_to_entries = {}
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                status = chapter_info.get("status", "")
                
                # Skip known non-chapter files
                if (
                    (output_file and output_file.lower() in _non_chapter_files)
                    or _is_progress_sidecar_entry(chapter_info, output_file)
                ):
                    continue
                # Skip combined _translated output files (unless source itself has _translated)
                if output_file and not _source_has_translated and any(
                    output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
                ):
                    continue
                
                # Include chapters with output files OR transient statuses with null output file (legacy)
                # (composite keys like "0_TOC" should still be represented in the UI)
                if output_file or status in ["in_progress", "pending", "failed", "qa_failed"]:
                    # For merged chapters, use a unique key (chapter_key) instead of output_file
                    # This ensures merged chapters appear as separate entries in the list
                    if status == "merged":
                        file_key = f"_merged_{chapter_key}"
                    elif output_file:
                        file_key = output_file
                    elif status == "in_progress":
                        file_key = f"_in_progress_{chapter_key}"
                    elif status == "pending":
                        file_key = f"_pending_{chapter_key}"
                    elif status == "qa_failed":
                        file_key = f"_qa_failed_{chapter_key}"
                    else:  # failed
                        file_key = f"_failed_{chapter_key}"
                    
                    if file_key not in files_to_entries:
                        files_to_entries[file_key] = []
                    files_to_entries[file_key].append((chapter_key, chapter_info))
            
            for output_file, entries in files_to_entries.items():
                chapter_key, chapter_info = entries[0]

                subtitle_row = self._build_subtitle_progress_row(
                    prog,
                    entries,
                    output_file,
                )
                if subtitle_row is not None:
                    if subtitle_row['status'] == 'completed':
                        subtitle_path = subtitle_row['output_file']
                        if not os.path.isabs(subtitle_path):
                            subtitle_path = os.path.join(output_dir, subtitle_path)
                        if not os.path.exists(subtitle_path):
                            subtitle_row['status'] = 'not_translated'
                    chapter_display_info.append(subtitle_row)
                    continue
                
                # Get the actual output file (strip placeholder prefix if present)
                actual_output_file = output_file
                if (
                    output_file.startswith("_merged_")
                    or output_file.startswith("_in_progress_")
                    or output_file.startswith("_pending_")
                    or output_file.startswith("_failed_")
                    or output_file.startswith("_qa_failed_")
                ):
                    # For merged/in_progress/pending/failed/qa_failed, get the actual output_file from chapter_info
                    actual_output_file = chapter_info.get("output_file", "")
                    if not actual_output_file:
                        # Generate expected filename based on actual_num
                        actual_num = chapter_info.get("actual_num")
                        if actual_num is not None:
                            # Use .txt extension for text files, .html for EPUB
                            ext = ".txt" if file_path.endswith(".txt") else ".html"
                            actual_output_file = f"response_section_{actual_num}{ext}"
                
                # Check if this is a special file (files without numbers)
                original_basename = chapter_info.get("original_basename", "")
                filename_to_check = original_basename if original_basename else actual_output_file
                
                is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
                
                # Extract chapter number - prioritize stored values
                chapter_num = None
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                
                # Fallback: extract from filename
                if chapter_num is None:
                    import re
                    matches = re.findall(r'(\d+)', actual_output_file)
                    if matches:
                        chapter_num = int(matches[-1])
                    else:
                        chapter_num = 999999
                
                status = chapter_info.get("status", "unknown")
                if status in ("completed_empty", "completed_image_only"):
                    status = "completed"
                
                # Check file existence
                if status == "completed":
                    output_path = os.path.join(output_dir, actual_output_file)
                    if not os.path.exists(output_path):
                        status = "file_missing"
                
                chapter_display_info.append({
                    'key': chapter_key,
                    'num': chapter_num,
                    'info': chapter_info,
                    'output_file': actual_output_file,  # Use actual output file, not placeholder
                    'status': status,
                    'duplicate_count': len(entries),
                    'entries': entries,
                    'is_special': is_special
                })
            
            # Sort by chapter number
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)

        _display_data = {
            'prog': prog,
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_source_is_subtitle': progress_source_is_subtitle,
        }
        self._append_chunk_progress_display_info(
            _display_data, chapter_display_info
        )
        self._append_metadata_display_info(_display_data, chapter_display_info)
        self._append_translation_artifact_display_info(
            _display_data, chapter_display_info
        )
        self._append_pdf_ocr_display_info(_display_data, chapter_display_info)
        self._append_image_gen_display_info(_display_data, chapter_display_info)

        return {
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': prog,
            'progress_source_is_subtitle': progress_source_is_subtitle,
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'existing_output_files': _existing_output_files,
            'fixed_output_dir': (
                os.path.abspath(str(resolved_output_dir))
                if resolved_output_dir
                else None
            ),
        }

    # -- refresh (data half of _refresh_retranslation_data) ------------------

    def _resolve_progress_view_output_dir(self, data):
        """Follow an output-folder override changed while the view is open (RG 31335-31371)."""
        try:
            file_path = data.get('file_path')
            if file_path and not data.get('fixed_output_dir'):
                epub_base = os.path.splitext(os.path.basename(file_path))[0]
                override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
                if not override_dir and hasattr(self, 'config'):
                    try:
                        override_dir = self.config.get('output_directory')
                    except Exception:
                        override_dir = None

                expected_output_dir = os.path.join(override_dir, epub_base) if override_dir else epub_base
                # On macOS .app bundles, cwd can be '/' (read-only root).
                # Resolve relative output paths against the input file's directory.
                # Only on macOS — on Windows this would change the output dir and break progress tracking.
                if _IS_MACOS and not os.path.isabs(expected_output_dir):
                    expected_output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), expected_output_dir)
                expected_progress_file = os.path.join(expected_output_dir, "translation_progress.json")

                # Update in-place if changed
                if expected_output_dir and data.get('output_dir') != expected_output_dir:
                    data['output_dir'] = expected_output_dir
                if expected_progress_file and data.get('progress_file') != expected_progress_file:
                    data['progress_file'] = expected_progress_file

                # Keep cache consistent too (if present)
                try:
                    file_key = os.path.abspath(file_path)
                    if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                        cached = self._retranslation_dialog_cache[file_key]
                        if isinstance(cached, dict):
                            cached['output_dir'] = data.get('output_dir')
                            cached['progress_file'] = data.get('progress_file')
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Could not re-resolve output override on refresh: {e}")

    def _reload_progress_view_data(self, data):
        """Reload the progress JSON and rebuild the rows of a view (RG 31373-31639).

        Read-only ticks (``data['_refresh_read_only']``) never write.  Returns False
        when the refresh stopped early (the desktop then leaves the list unchanged).
        """
        def _read_progress_json_safely(path):
            import random
            import time as _time
            last_error = None
            for _attempt in range(20):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                    data['_last_good_prog'] = copy.deepcopy(loaded)
                    return loaded
                except (PermissionError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    last_error = e
                    _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
            snapshot = data.get('_last_good_prog') or data.get('prog')
            if isinstance(snapshot, dict):
                print(f"⚠️ Progress file locked during refresh; using last good snapshot this tick: {last_error}")
                return copy.deepcopy(snapshot)
            raise last_error

        def _write_progress_json_safely(path, payload):
            import random
            import tempfile
            import time as _time
            progress_dir = os.path.dirname(path) or '.'
            if progress_dir:
                os.makedirs(progress_dir, exist_ok=True)
            last_error = None
            for _attempt in range(20):
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as tmp:
                        temp_path = tmp.name
                        json.dump(payload, tmp, ensure_ascii=False, indent=2)
                        tmp.flush()
                        try:
                            os.fsync(tmp.fileno())
                        except Exception:
                            pass
                    os.replace(temp_path, path)
                    return True
                except (PermissionError, OSError) as e:
                    last_error = e
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
            raise last_error

        # Reload progress file. Prefer the snapshot prefetched off-thread by
        # the silent auto-refresh — zero disk I/O on the GUI thread.
        _prefetched_prog = data.pop('_prefetched_prog', None)
        _prefetched_prog_path = data.pop('_prefetched_prog_path', None)
        if _prefetched_prog is not None and _prefetched_prog_path == data.get('progress_file'):
            data['prog'] = _prefetched_prog
            # The background loader created this object exclusively for
            # this snapshot. Retaining it is sufficient for fallback and
            # avoids another full progress-tree clone on the GUI thread.
            data['_last_good_prog'] = data['prog']
        # Check if the progress file exists first
        elif not os.path.exists(data['progress_file']):
            print(f"⚠️ Progress file not found: {data['progress_file']}")
            # Recreate a minimal progress file and auto-discover completed files from output_dir
            prog = {
                "chapters": {},
                "chapter_chunks": {},
                "version": "2.1"
            }

            def _auto_discover_from_output_dir(output_dir, prog):
                updated = False
                try:
                    files = [
                        f for f in os.listdir(output_dir)
                        if os.path.isfile(os.path.join(output_dir, f))
                        # accept any extension except .epub
                        and not f.lower().endswith("_translated.txt")
                        and f != "translation_progress.json"
                        and f.lower() not in _NON_CHAPTER_OUTPUT_FILENAMES
                        and f.casefold() not in _PROGRESS_SIDECAR_FILENAMES
                        and not f.lower().endswith(".epub")
                        and not f.lower().endswith(".cache")
                    ]
                    subtitle_auto_index = 0
                    subtitle_source = str(data.get('file_path') or '').lower().endswith(
                        ('.srt', '.ass', '.lrc', '.zip')
                    )
                    for fname in sorted(files):
                        is_subtitle_output = (
                            subtitle_source
                            and os.path.splitext(fname)[1].lower() in ('.srt', '.ass', '.lrc')
                        )
                        if is_subtitle_output:
                            subtitle_auto_index += 1
                        base = os.path.basename(fname)
                        if base.startswith("response_"):
                            base = base[len("response_"):]
                        while True:
                            new_base, ext = os.path.splitext(base)
                            if not ext:
                                break
                            base = new_base
                        import re
                        m = re.findall(r"(\\d+)", base)
                        chapter_num = (
                            subtitle_auto_index
                            if is_subtitle_output
                            else (int(m[-1]) if m else None)
                        )
                        key = (
                            f"subtitle:auto:{fname}"
                            if is_subtitle_output
                            else (
                                str(chapter_num)
                                if chapter_num is not None
                                else f"special_{base}"
                            )
                        )
                        actual_num = chapter_num if chapter_num is not None else 0
                        if key in prog.get("chapters", {}):
                            continue
                        discovered_entry = {
                            "actual_num": actual_num,
                            "content_hash": "",
                            "output_file": fname,
                            "status": "completed",
                            "last_updated": os.path.getmtime(os.path.join(output_dir, fname)),
                            "auto_discovered": True,
                            "original_basename": fname
                        }
                        if is_subtitle_output:
                            discovered_entry.update({
                                "subtitle_progress_key": key,
                                "subtitle_output_file": os.path.abspath(
                                    os.path.join(output_dir, fname)
                                ),
                                "subtitle_bundle_source_index": subtitle_auto_index,
                                "subtitle_source_batch_num": 1,
                                "subtitle_source_batch_count": 1,
                            })
                        prog.setdefault("chapters", {})[key] = discovered_entry
                        updated = True
                except Exception as e:
                    print(f"⚠️ Auto-discovery (refresh no OPF) failed: {e}")
                return updated

            if _auto_discover_from_output_dir(data['output_dir'], prog):
                print("💾 Recreated progress file via auto-discovery (refresh)")
            try:
                _merge_and_write_retranslation_progress(
                    data['progress_file'],
                    {"chapters": {}, "chapter_chunks": {}, "version": "2.1"},
                    prog,
                )
            except PermissionError as e:
                print(f"⚠️ Progress file locked during refresh recreate; will retry on next refresh tick: {e}")
                return False
            except Exception as e:
                self._progress_reload_error(data, "Progress File Error",
                                            f"Could not recreate progress file:\n{e}")
                return False
        
        # The translator may briefly lock/replace the JSON; retry and skip this tick if it stays locked.
        # Skipped when a prefetched snapshot was already applied above.
        if _prefetched_prog is None or _prefetched_prog_path != data.get('progress_file'):
            data['prog'] = _read_progress_json_safely(data['progress_file'])
            data['_last_good_prog'] = copy.deepcopy(data['prog'])
        data['_progress_view_baseline'] = (
            None if bool(data.get('_refresh_read_only')) else copy.deepcopy(data['prog'])
        )

        if (
            not bool(data.get('_refresh_read_only'))
            and self._seed_subtitle_zip_progress_entries(
                data.get('file_path'),
                data.get('output_dir'),
                data['prog'],
            )
        ):
            data['_progress_view_baseline'] = _commit_view_progress(
                data['progress_file'], data['_progress_view_baseline'], data['prog']
            )
            data['_last_good_prog'] = copy.deepcopy(data['prog'])

        def _progress_has_active_entries(prog):
            try:
                return any(
                    isinstance(info, dict)
                    and str(info.get('status', '')).lower() == 'in_progress'
                    for info in (prog or {}).get('chapters', {}).values()
                )
            except Exception:
                return False
        
        # Clean up missing files and merged children before display unless disabled
        # NOTE: this runs on the GUI thread from a 2s QTimer. It must NEVER
        # use a plain `from TransateKRtoEN import ...` — if the translation
        # worker is importing that module right now (first Run Translation
        # click in a frozen build), the import lock would freeze the whole
        # GUI for the entire 10-20s import. Skip cleanup for this tick instead.
        # Read-only ticks (silent auto-refresh on a prefetched snapshot) must
        # not run disk-scanning cleanup or write the progress file: the
        # snapshot may be ~2s stale, and cleanup itself stats the output dir.
        _read_only_tick = bool(data.get('_refresh_read_only'))
        if (not _read_only_tick
                and not data.get('skip_cleanup', False)
                and not _progress_has_active_entries(data['prog'])):
            if self._progress_cleanup_ready():
                before_cleanup = copy.deepcopy(data['prog'])
                cleanup_missing_files(data['prog'], data['output_dir'])

                # Save only if cleanup really changed the file. During active translation
                # refresh should be a reader, not another progress writer.
                if data['prog'] != before_cleanup:
                    data['_progress_view_baseline'] = _commit_view_progress(
                        data['progress_file'], data['_progress_view_baseline'], data['prog']
                    )

        if self._reconcile_tts_audio_files(data) and not _read_only_tick:
            data['_progress_view_baseline'] = _commit_view_progress(
                data['progress_file'], data['_progress_view_baseline'], data['prog']
            )

        pdf_outline_changed = self._seed_pdf_outline_progress_entries(
            data.get('file_path'), data.get('output_dir'), data['prog']
        )
        if pdf_outline_changed and not _read_only_tick:
            data['_progress_view_baseline'] = _commit_view_progress(
                data['progress_file'], data['_progress_view_baseline'], data['prog']
            )

        if (
            not _read_only_tick
            and self._ensure_metadata_progress_entry(
                data['prog'], data['output_dir'], data.get('file_path')
            )
        ):
            data['_progress_view_baseline'] = _commit_view_progress(
                data['progress_file'], data['_progress_view_baseline'], data['prog']
            )
        if (
            not _read_only_tick
            and self._ensure_translation_artifact_progress_entries(
                data['prog'], data['output_dir'], data.get('file_path')
            )
        ):
            data['_progress_view_baseline'] = _commit_view_progress(
                data['progress_file'], data['_progress_view_baseline'], data['prog']
            )
        
        # Check if we're using OPF-based display or fallback
        if data.get('spine_chapters'):
            # OPF-based: Re-run full matching logic to update merged status correctly
            # We need to re-match spine chapters against the updated progress JSON
            self._rematch_spine_chapters(data)
        else:
            # Fallback mode: REBUILD chapter_display_info from scratch to pick up new entries
            # This is necessary for text files or EPUBs without OPF
            self._rebuild_chapter_display_info(data)
        data.pop('_progress_view_baseline', None)
        return True

    # -- statistics (RG 33727-33779 and the initial legend 27600-27621) ---------

    def _progress_statistics(self, data):
        """Statistics row values of a view (RG _update_statistics_display 33728-33767).

        Returns ``(total_chapters, chunk_count, completed, merged, in_progress, pending,
        missing, failed, skipped, mode)``.
        """
        chapter_display_info = data.get('chapter_display_info', [])
        chunk_count = sum(bool(info.get("is_chunk_progress")) for info in chapter_display_info)
        pdf_rows = [info for info in chapter_display_info if info.get('pdf_ocr')]
        if pdf_rows and len(pdf_rows) == len(chapter_display_info):
            pdf_info = pdf_rows[0].get('info') or {}
            try:
                total_chapters = int(pdf_info.get('total') or 0)
                completed = min(int(pdf_info.get('done') or 0), total_chapters)
                failed = int(pdf_info.get('failed') or 0)
            except (TypeError, ValueError):
                total_chapters = len(chapter_display_info)
                completed = 0
                failed = 0
            merged = 0
            pending = 0
            skipped = 0
            status = self._progress_display_status(pdf_rows[0], data)
            in_progress = 1 if status == 'in_progress' else 0
            missing = max(0, total_chapters - completed - failed)
        else:
            total_chapters = len(chapter_display_info)
            # Skipped special files get their own count and are
            # excluded from the regular statuses (mirrors the
            # initial legend build).
            non_skipped = []
            skipped = 0
            for info in chapter_display_info:
                if self._progress_entry_is_skipped_special(info):
                    skipped += 1
                else:
                    non_skipped.append(info)
            display_statuses = [self._progress_display_status(info, data) for info in non_skipped]
            completed = sum(1 for status in display_statuses if status == 'completed')
            merged = sum(1 for status in display_statuses if status == 'merged')
            in_progress = sum(1 for status in display_statuses if status == 'in_progress')
            pending = sum(1 for status in display_statuses if status == 'pending')
            missing = sum(1 for status in display_statuses if status in ['not_translated', 'not_refined', 'no_tts'])
            failed = sum(1 for status in display_statuses if status in ['failed', 'qa_failed', 'refine_failed'])

        mode = self._current_progress_output_mode(data)
        return (
            total_chapters,
            chunk_count,
            completed,
            merged,
            in_progress,
            pending,
            missing,
            failed,
            skipped,
            mode,
        )

    def _progress_initial_statistics(self, prog, chapter_display_info, spine_chapters):
        """Legend values when the dialog is built (RG 27600-27621).

        Returns ``(total_chapters, chunk_count, completed, merged, in_progress, pending,
        missing, failed, skipped)``.
        """
        _stats_data = {'prog': prog}
        _stats_entries = chapter_display_info or spine_chapters or []
        chunk_count = sum(bool(info.get("is_chunk_progress")) for info in _stats_entries)
        total_chapters = len(_stats_entries)
        completed = merged = in_progress = pending = missing = failed = skipped = 0
        for ch in _stats_entries:
            if self._progress_entry_is_skipped_special(ch):
                skipped += 1
                continue
            _st = self._progress_display_status(ch, _stats_data)
            if _st == 'completed':
                completed += 1
            elif _st == 'merged':
                merged += 1
            elif _st == 'in_progress':
                in_progress += 1
            elif _st == 'pending':
                pending += 1
            elif _st in ('not_translated', 'not_refined', 'no_tts'):
                missing += 1
            elif _st in ('failed', 'qa_failed', 'refine_failed'):
                failed += 1
        return (
            total_chapters,
            chunk_count,
            completed,
            merged,
            in_progress,
            pending,
            missing,
            failed,
            skipped,
        )

    # -- matching, rows, status, model column (RG 31719-33044) ----------------

    def _rematch_spine_chapters(self, data, append_auxiliary=True):
        """Re-run the full spine chapter matching logic against updated progress JSON"""
        prog = data['prog']
        output_dir = data['output_dir']
        spine_chapters = data['spine_chapters']

        def _normalize_opf_match_name(name: str) -> str:
            return _normalize_progress_match_name(name)

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build indexes once (O(n))
        basename_to_progress = {}
        response_to_progress = {}
        actualnum_to_progress = {}
        actualnum_orig_to_progress = {}
        actualnum_out_to_progress = {}
        actualnum_merged_without_orig = {}
        composite_to_progress = {}

        chapters_dict = prog.get("chapters", {})
        for chapter_key, ch in chapters_dict.items():
            progress_ref = (chapter_key, ch)
            orig = ch.get("original_basename", "")
            out = ch.get("output_file", "")
            actual_num = ch.get("actual_num")
            norm_orig = _normalize_opf_match_name(orig) if orig else ""
            norm_out = _normalize_opf_match_name(out) if out else ""

            if orig:
                basename_to_progress.setdefault(norm_orig, []).append(progress_ref)
            if out:
                response_to_progress.setdefault(out, []).append(progress_ref)
                if norm_out != out:
                    response_to_progress.setdefault(norm_out, []).append(progress_ref)
            if actual_num is not None:
                actualnum_to_progress.setdefault(actual_num, []).append(progress_ref)
                if norm_orig:
                    actualnum_orig_to_progress.setdefault(
                        (actual_num, norm_orig), []
                    ).append(progress_ref)
                if norm_out:
                    actualnum_out_to_progress.setdefault(
                        (actual_num, norm_out), []
                    ).append(progress_ref)
                if ch.get('status') == 'merged' and not norm_orig:
                    actualnum_merged_without_orig.setdefault(
                        actual_num, []
                    ).append(progress_ref)

            fname_for_comp = orig or out
            if fname_for_comp and actual_num is not None:
                filename_noext = os.path.splitext(_normalize_opf_match_name(fname_for_comp))[0]
                composite_to_progress[f"{actual_num}_{filename_noext}"] = progress_ref

        # Cache directory listing to avoid thousands of exists calls.
        # Prefer the snapshot prefetched off-thread by the silent auto-refresh;
        # even a single scandir on the GUI thread can stall for seconds while
        # EPUB compile saturates the disk (verified via the freeze watchdog).
        existing_files = data.pop('_prefetched_output_listing', None)
        if existing_files is None:
            try:
                with os.scandir(output_dir) as _scan:
                    existing_files = {e.name for e in _scan if e.is_file()}
            except Exception:
                existing_files = set()
        normalized_existing_files = {}
        for existing_file in existing_files:
            normalized_existing_files.setdefault(
                _normalize_opf_match_name(existing_file),
                existing_file,
            )

        def file_exists_fast(fname: str) -> bool:
            return fname in existing_files

        retain_source_extension = (
            os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1'
            or self.config.get('retain_source_extension', False)
        )

        for spine_ch in spine_chapters:
            # Do not retain a progress object removed by a newer JSON snapshot.
            spine_ch.pop('progress_key', None)
            spine_ch.pop('progress_entry', None)
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)

            base_name = os.path.splitext(filename)[0]
            retain = retain_source_extension

            if is_special:
                response_with_prefix = f"response_{base_name}.html"
                if retain:
                    expected_response = filename
                elif file_exists_fast(response_with_prefix):
                    expected_response = response_with_prefix
                else:
                    expected_response = filename
            else:
                response_with_prefix = f"response_{base_name}.html"
                if retain:
                    expected_response = filename
                elif not file_exists_fast(filename) and file_exists_fast(response_with_prefix):
                    expected_response = response_with_prefix
                else:
                    expected_response = filename

            matched_info = None
            matched_key = None
            basename_key = _normalize_opf_match_name(filename)

            def _set_matched_progress(chapter_key, chapter_info):
                nonlocal matched_info, matched_key
                matched_info = chapter_info
                matched_key = chapter_key

            # 1) original basename map
            lst = basename_to_progress.get(basename_key)
            if lst:
                for chapter_key, ch in lst:
                    status = ch.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if ch.get('actual_num') == chapter_num:
                            _set_matched_progress(chapter_key, ch)
                            break
                    else:
                        _set_matched_progress(chapter_key, ch)
                        break

            # 2) response map (choose highest severity, prefer matching chapter_num)
            if not matched_info:
                lookup_keys = [
                    expected_response,
                    _normalize_opf_match_name(expected_response),
                    f"response_{expected_response}" if not expected_response.startswith("response_") else expected_response,
                    basename_key
                ]
                lst = None
                for k in lookup_keys:
                    if k in response_to_progress:
                        lst = response_to_progress[k]
                        break
                if lst:
                    has_qa = any(ch.get('status') == 'qa_failed' for _, ch in lst)
                    if has_qa:
                        lst = [(chapter_key, ch) for chapter_key, ch in lst if ch.get('status') != 'pending']
                    severity = {'qa_failed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1, 'completed': 0}
                    best_key = None
                    best = None
                    best_score = -1
                    for chapter_key, ch in lst:
                        status = ch.get('status', '')
                        score = severity.get(status, -1)
                        matches_num = ch.get('actual_num') == chapter_num
                        if score > best_score or (score == best_score and matches_num):
                            best_key = chapter_key
                            best = ch
                            best_score = score
                            # If exact chapter match and highest severity, keep going in case of even higher severity
                    if best:
                        _set_matched_progress(best_key, best)

            # 3) composite key
            if not matched_info:
                filename_noext = base_name
                if filename_noext.startswith("response_"):
                    filename_noext = filename_noext[len("response_"):]
                comp_key = f"{chapter_num}_{filename_noext}"
                comp_ref = composite_to_progress.get(comp_key)
                if comp_ref:
                    _set_matched_progress(comp_ref[0], comp_ref[1])

            # 4) actual_num map fallback (avoid mis-matching special files)
            if not matched_info and chapter_num in actualnum_to_progress:
                # Split filenames end in _000/_001, so hundreds of unrelated
                # rows can share the same raw number. The former fallback
                # rescanned that entire bucket for every unmatched spine row,
                # creating an O(n²) GUI-thread stall on large split EPUBs.
                # Exact composite indexes preserve the legacy fallback without
                # considering candidates whose filenames cannot possibly match.
                normalized_expected = _normalize_opf_match_name(
                    expected_response
                )
                candidate_refs = []
                seen_candidate_keys = set()
                for refs in (
                    actualnum_orig_to_progress.get(
                        (chapter_num, basename_key), ()
                    ),
                    actualnum_out_to_progress.get(
                        (chapter_num, normalized_expected), ()
                    ),
                    actualnum_out_to_progress.get(
                        (chapter_num, basename_key), ()
                    ),
                    actualnum_merged_without_orig.get(chapter_num, ()),
                ):
                    for candidate_key, candidate_info in refs:
                        if candidate_key in seen_candidate_keys:
                            continue
                        seen_candidate_keys.add(candidate_key)
                        candidate_refs.append(
                            (candidate_key, candidate_info)
                        )

                for chapter_key, ch in candidate_refs:
                    status = ch.get('status', '')
                    out_file = ch.get('output_file')
                    orig_base = ch.get('original_basename', '') or ''
                    normalized_orig = _normalize_opf_match_name(orig_base)
                    normalized_out = _normalize_opf_match_name(out_file)

                    # If this spine entry is a special file (no digits), require filename match to avoid hijacking by other chapter 0 entries
                    if is_special:
                        fname_matches = (
                            (normalized_orig and normalized_orig == basename_key)
                            or (
                                out_file
                                and (
                                    normalized_out == normalized_expected
                                    or out_file == expected_response
                                )
                            )
                        )
                        if not fname_matches:
                            continue

                    if status == 'merged':
                        if normalized_orig == basename_key or not orig_base:
                            _set_matched_progress(chapter_key, ch)
                            break
                    elif status in ['in_progress', 'failed', 'pending', 'qa_failed']:
                        if out_file and (
                            normalized_out == normalized_expected
                            or out_file == expected_response
                        ):
                            _set_matched_progress(chapter_key, ch)
                            break
                    else:
                        if (
                            (normalized_orig and normalized_orig == basename_key)
                            or (
                                out_file
                                and (
                                    normalized_out == normalized_expected
                                    or out_file == expected_response
                                )
                            )
                        ):
                            _set_matched_progress(chapter_key, ch)
                            break

            file_exists = file_exists_fast(expected_response)

            if matched_info:
                status = (
                    'pending'
                    if matched_info.get('manual_editing_pending')
                    else matched_info.get('status', 'unknown')
                )
                spine_ch['progress_key'] = matched_key

                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    continue

                spine_ch['status'] = status
                spine_ch['output_file'] = expected_response if is_special else matched_info.get('output_file', expected_response)
                spine_ch['progress_entry'] = matched_info
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response

                if status == 'completed':
                    output_file = spine_ch['output_file']
                    if not file_exists_fast(output_file):
                        if file_exists and expected_response:
                            spine_ch['output_file'] = expected_response
                            matched_info['output_file'] = expected_response
                        else:
                            spine_ch['status'] = 'not_translated'

            elif file_exists:
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response

            else:
                norm_target = _normalize_opf_match_name(filename)
                matched_file = normalized_existing_files.get(norm_target)
                if matched_file:
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                else:
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS (refresh path)
        # =====================================================
        
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']

                # Require normalized filename match between spine file and output file, and the file must exist
                norm_spine = _normalize_opf_match_name(filename)
                norm_out = _normalize_opf_match_name(output_file)
                file_exists = file_exists_fast(output_file)
                if norm_spine != norm_out or not file_exists:
                    continue

                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                if chapter_key not in prog.get("chapters", {}):
                    prog.setdefault("chapters", {})[chapter_key] = {
                        "actual_num": chapter_num,
                        "content_hash": "",  # Unknown since we don't have the source
                        "output_file": output_file,
                        "status": "completed",
                        "last_updated": time.time(),
                        "auto_discovered": True,
                        "original_basename": filename
                    }
                    progress_updated = True
                    print(f"✅ Auto-discovered and tracked (refresh): {filename} -> {output_file}")
        
        # Save progress file if we added new entries
        if progress_updated and not data.get('_refresh_read_only'):
            try:
                data['_progress_view_baseline'] = _commit_view_progress(
                    data['progress_file'],
                    data.get('_progress_view_baseline') or {},
                    prog,
                )
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file (refresh)")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file during refresh: {e}")
        
        # Rebuild chapter_display_info from updated spine_chapters
        chapter_display_info = []
        for spine_ch in spine_chapters:
            display_info = {
                'key': spine_ch.get('filename', ''),
                'num': spine_ch['file_chapter_num'],
                'display_num': spine_ch.get(
                    'display_chapter_num', spine_ch['file_chapter_num']
                ),
                'info': spine_ch.get('progress_entry', {}),
                'output_file': spine_ch['output_file'],
                'status': spine_ch['status'],
                'duplicate_count': 1,
                'entries': [],
                'opf_position': spine_ch['position'],
                'original_filename': spine_ch['filename'],
                'is_special': spine_ch.get('is_special', False),
                'progress_key': spine_ch.get('progress_key')
            }
            chapter_display_info.append(display_info)
        
        if append_auxiliary:
            self._append_chunk_progress_display_info(data, chapter_display_info)
            self._append_metadata_display_info(data, chapter_display_info)
            self._append_translation_artifact_display_info(
                data, chapter_display_info
            )
            self._append_pdf_ocr_display_info(data, chapter_display_info)
            self._append_image_gen_display_info(data, chapter_display_info)
        data['chapter_display_info'] = chapter_display_info
    
    def _rebuild_chapter_display_info(self, data):
        """Rebuild chapter_display_info from scratch (for fallback mode without OPF)"""
        # This is the same logic as the initial build in _force_retranslation_epub_or_text
        # but extracted here so refresh can use it
        
        prog = data['prog']
        output_dir = data['output_dir']
        file_path = data.get('file_path', '')
        show_special = data.get('show_special_files_state', False)
        
        # Known non-chapter files that should never appear in the progress list
        _non_chapter_files = _NON_CHAPTER_OUTPUT_FILENAMES
        _source_has_translated = "_translated" in os.path.basename(file_path).lower()
        files_to_entries = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            status = chapter_info.get("status", "")
            
            # Skip known non-chapter files
            if (
                (output_file and output_file.lower() in _non_chapter_files)
                or _is_progress_sidecar_entry(chapter_info, output_file)
            ):
                continue
            # Skip combined _translated output files (unless source itself has _translated)
            if output_file and not _source_has_translated and any(
                output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
            ):
                continue
            
            # Include chapters with output files OR transient statuses with null output file (legacy)
            if output_file or status in ["in_progress", "pending", "failed", "qa_failed"]:
                # For merged chapters, use a unique key (chapter_key) instead of output_file
                # This ensures merged chapters appear as separate entries in the list
                if status == "merged":
                    file_key = f"_merged_{chapter_key}"
                elif output_file:
                    file_key = output_file
                elif status == "in_progress":
                    file_key = f"_in_progress_{chapter_key}"
                elif status == "pending":
                    file_key = f"_pending_{chapter_key}"
                elif status == "qa_failed":
                    file_key = f"_qa_failed_{chapter_key}"
                else:  # failed
                    file_key = f"_failed_{chapter_key}"
                
                if file_key not in files_to_entries:
                    files_to_entries[file_key] = []
                files_to_entries[file_key].append((chapter_key, chapter_info))
        
        chapter_display_info = []
        
        for output_file, entries in files_to_entries.items():
            chapter_key, chapter_info = entries[0]

            subtitle_row = self._build_subtitle_progress_row(
                prog,
                entries,
                output_file,
            )
            if subtitle_row is not None:
                if subtitle_row['status'] == 'completed':
                    subtitle_path = subtitle_row['output_file']
                    if not os.path.isabs(subtitle_path):
                        subtitle_path = os.path.join(output_dir, subtitle_path)
                    if not os.path.exists(subtitle_path):
                        subtitle_row['status'] = 'not_translated'
                chapter_display_info.append(subtitle_row)
                continue
            
            # Get the actual output file (strip placeholder prefix if present)
            actual_output_file = output_file
            if (
                output_file.startswith("_merged_")
                or output_file.startswith("_in_progress_")
                or output_file.startswith("_pending_")
                or output_file.startswith("_failed_")
                or output_file.startswith("_qa_failed_")
            ):
                # For merged/in_progress/failed/qa_failed, get the actual output_file from chapter_info
                actual_output_file = chapter_info.get("output_file", "")
                if not actual_output_file:
                    # Generate expected filename based on actual_num
                    actual_num = chapter_info.get("actual_num")
                    if actual_num is not None:
                        # Use .txt extension for text files, .html for EPUB
                        ext = ".txt" if file_path.endswith(".txt") else ".html"
                        actual_output_file = f"response_section_{actual_num}{ext}"
            
            # Check if this is a special file using configured keyword lists
            original_basename = chapter_info.get("original_basename", "")
            filename_to_check = original_basename if original_basename else actual_output_file
            
            is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
            
            # Don't skip special files here - let the display logic handle hiding them
            # This ensures chapter_display_info contains all items, and the listbox
            # will properly hide/show items based on the toggle state
            
            # Extract chapter number - prioritize stored values
            chapter_num = None
            if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                chapter_num = chapter_info['actual_num']
            elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                chapter_num = chapter_info['chapter_num']
            
            # Fallback: extract from filename
            if chapter_num is None:
                import re
                matches = re.findall(r'(\d+)', actual_output_file)
                if matches:
                    chapter_num = int(matches[-1])
                else:
                    chapter_num = 999999
            
            status = chapter_info.get("status", "unknown")
            if status in ("completed_empty", "completed_image_only"):
                status = "completed"
            
            # Check file existence
            if status == "completed":
                output_path = os.path.join(output_dir, actual_output_file)
                if not os.path.exists(output_path):
                    status = "not_translated"
            
            chapter_display_info.append({
                'key': chapter_key,
                'num': chapter_num,
                'info': chapter_info,
                'output_file': actual_output_file,  # Use actual output file, not placeholder
                'status': status,
                'duplicate_count': len(entries),
                'entries': entries,
                'is_special': is_special,
                'progress_key': chapter_key
            })
        
        # Sort by chapter number
        chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
        
        self._append_chunk_progress_display_info(data, chapter_display_info)
        self._append_metadata_display_info(data, chapter_display_info)
        self._append_translation_artifact_display_info(
            data, chapter_display_info
        )
        self._append_pdf_ocr_display_info(data, chapter_display_info)
        self._append_image_gen_display_info(data, chapter_display_info)

        # Update data with rebuilt list
        data['chapter_display_info'] = chapter_display_info

    def _append_chunk_progress_display_info(self, data, chapter_display_info):
        """Insert selectable per-chunk rows after their EPUB/PDF chapter."""
        prog = data.get("prog") if isinstance(data, dict) else None
        if not isinstance(prog, dict):
            return
        chapter_chunks = prog.get("chapter_chunks", {})
        chapters = prog.get("chapters", {})
        if not isinstance(chapter_chunks, dict) or not chapter_chunks:
            return

        expanded = []
        for parent in list(chapter_display_info or []):
            expanded.append(parent)
            if (
                parent.get("is_subtitle")
                or parent.get("pdf_ocr")
                or self._is_metadata_progress_info(parent)
                or self._is_translation_artifact_progress_info(parent)
            ):
                continue
            parent_key = parent.get("progress_key")
            if not parent_key:
                fallback_key = parent.get("key")
                if fallback_key in chapters and isinstance(
                    chapters.get(fallback_key), dict
                ):
                    parent_key = fallback_key
            if parent_key and not parent.get("progress_key"):
                parent["progress_key"] = parent_key
            parent_entry = (
                chapters.get(parent_key)
                if parent_key and isinstance(chapters.get(parent_key), dict)
                else parent.get("info") or parent.get("progress_entry") or {}
            )
            chunk_key = str(
                (parent_entry or {}).get("content_hash")
                or parent_key
                or ""
            )
            chunk_entry = chapter_chunks.get(chunk_key)
            if not is_multi_chunk_entry(chunk_entry):
                continue
            ensure_chunk_entry_schema(chunk_entry)
            parent["status"] = effective_parent_status(
                parent.get("status"),
                chunk_entry,
            )
            total = int(chunk_entry.get("total") or 0)
            for raw_index, record in sorted_chunk_items(
                chunk_entry.get("entries", {})
            ):
                if not isinstance(record, dict):
                    continue
                try:
                    chunk_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                expanded.append({
                    "key": f"chunk:{chunk_key}:{chunk_index}",
                    "num": parent.get("num"),
                    "display_num": parent.get("display_num", parent.get("num")),
                    "info": record,
                    "output_file": parent.get("output_file", ""),
                    "status": record.get("status", "pending"),
                    "duplicate_count": 1,
                    "entries": [],
                    "original_filename": parent.get("original_filename", ""),
                    "is_special": parent.get("is_special", False),
                    "is_chunk_progress": True,
                    "chunk_index": chunk_index,
                    "total_chunks": total,
                    "chunk_progress_key": chunk_key,
                    "parent_progress_key": parent_key,
                    "parent_info": parent_entry,
                    "pdf_toc_section": bool(
                        isinstance(parent_entry, dict)
                        and parent_entry.get("pdf_toc_section")
                    ),
                })
        chapter_display_info[:] = expanded

    def _append_pdf_ocr_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for PDF Vision OCR progress."""
        try:
            prog = data.get('prog') or {}
            pdf_ocr = prog.get('pdf_ocr')
            progress_output_mode = str(prog.get('output_mode') or data.get('output_mode') or '').lower().strip()
            ui_output_mode = ""
            try:
                if hasattr(self, '_get_output_mode'):
                    ui_output_mode = str(self._get_output_mode() or '').lower().strip()
            except Exception:
                ui_output_mode = ""
            if ui_output_mode and ui_output_mode != 'vision':
                return
            if progress_output_mode and progress_output_mode != 'vision':
                return
            current_file = str(data.get('file_path') or '')
            if current_file and not current_file.lower().endswith('.pdf'):
                return
            if not current_file:
                pdf_source = ""
                if isinstance(pdf_ocr, dict):
                    pdf_source = str(pdf_ocr.get('source_file') or '')
                if not pdf_source or not pdf_source.lower().endswith('.pdf'):
                    return
            if not isinstance(pdf_ocr, dict):
                output_dir = data.get('output_dir') or ''
                image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')
                image_dir = os.path.join(output_dir, 'images')
                single_dir = os.path.join(output_dir, 'OCR', 'single')
                image_count = 0
                cached_count = 0
                try:
                    if os.path.isdir(image_dir):
                        image_count = sum(
                            1 for name in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, name)) and name.lower().endswith(image_exts)
                        )
                except Exception:
                    image_count = 0
                try:
                    if os.path.isdir(single_dir):
                        cached_count = sum(
                            1 for name in os.listdir(single_dir)
                            if os.path.isfile(os.path.join(single_dir, name)) and name.lower().endswith('.txt')
                        )
                except Exception:
                    cached_count = 0
                total_guess = max(image_count, cached_count)
                if total_guess <= 0:
                    return
                pdf_ocr = {
                    'source_file': data.get('file_path'),
                    'ocr_source_file': '',
                    'status': 'completed' if cached_count >= total_guess else 'in_progress',
                    'total': total_guess,
                    'done': cached_count,
                    'cached': cached_count,
                    'no_text': 0,
                    'failed': 0,
                    'cache_inferred': True,
                }
            total = int(pdf_ocr.get('total') or 0)
            pages = pdf_ocr.get('pages') if isinstance(pdf_ocr.get('pages'), dict) else {}
            if total <= 0 and pages:
                total = len(pages)
            if total <= 0:
                return
            done = int(pdf_ocr.get('done') or 0)
            cached = int(pdf_ocr.get('cached') or 0)
            failed = int(pdf_ocr.get('failed') or 0)
            no_text = int(pdf_ocr.get('no_text') or 0)
            status = str(pdf_ocr.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'
            source_file = os.path.basename(str(pdf_ocr.get('source_file') or data.get('file_path') or 'PDF'))
            ocr_source_file = os.path.basename(str(pdf_ocr.get('ocr_source_file') or ''))
            label_bits = [f"{min(done, total)}/{total} pages"]
            if cached:
                label_bits.append(f"{cached} cached")
            if no_text:
                label_bits.append(f"{no_text} no-text")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"{source_file} -> {ocr_source_file or '_OCR.pdf'} ({', '.join(label_bits)})"
            info = dict(pdf_ocr)
            info['status'] = status
            info['ocr_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__pdf_ocr__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__pdf_ocr__',
                'pdf_ocr': True,
            })
        except Exception as e:
            print(f"Warning: could not read PDF OCR progress: {e}")

    def _append_image_gen_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for image generation progress (image output mode)."""
        try:
            prog = data.get('prog') or {}
            image_gen = prog.get('image_gen')
            if not isinstance(image_gen, dict):
                return
            # Hide the row when the user switches output mode away from 'image'.
            # Check the live GUI combo directly — it's the most reliable source.
            combo = getattr(self, '_output_mode_combo', None)
            if combo is not None:
                try:
                    live_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex(), 'text')
                    if live_mode != 'image':
                        return
                except Exception:
                    pass
            # Verify the source is epub/pdf using the stored source_file or the current file.
            source_file = str(image_gen.get('source_file') or data.get('file_path') or '')
            if source_file and not source_file.lower().endswith(('.epub', '.pdf')):
                return

            total = int(image_gen.get('total') or 0)
            if total <= 0:
                return
            done = int(image_gen.get('done') or 0)
            success = int(image_gen.get('success') or 0)
            skipped = int(image_gen.get('skipped') or 0)
            failed = int(image_gen.get('failed') or 0)
            status = str(image_gen.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'

            source_file = os.path.basename(str(image_gen.get('source_file') or data.get('file_path') or 'EPUB'))
            label_bits = [f"{min(done, total)}/{total} images"]
            if success:
                label_bits.append(f"{success} generated")
            if skipped:
                label_bits.append(f"{skipped} skipped")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"🎨 Image Generation: {source_file} ({', '.join(label_bits)})"

            info = dict(image_gen)
            info['status'] = status
            info['image_gen_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__image_gen__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__image_gen__',
                'image_gen': True,
            })
        except Exception as e:
            print(f"Warning: could not read image gen progress: {e}")
    
    def _current_progress_output_mode(self, data=None, entry=None):
        """Prefer the live GUI output mode over stale mode values saved in progress JSON."""
        candidates = []

        combo = getattr(self, '_output_mode_combo', None)
        if combo is not None:
            try:
                idx_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex())
                if idx_mode:
                    candidates.append(idx_mode)
            except RuntimeError:
                pass
            except Exception:
                pass
            try:
                candidates.append(combo.currentText())
            except RuntimeError:
                pass
            except Exception:
                pass

        try:
            if hasattr(self, '_get_output_mode'):
                candidates.append(self._get_output_mode())
        except Exception:
            pass

        candidates.append(getattr(self, 'output_mode_var', None))

        config = getattr(self, 'config', None)
        if isinstance(config, dict):
            candidates.append(config.get('output_mode'))
            if config.get('enable_audio_output_mode'):
                candidates.append('audio')
            if config.get('enable_refinement_output_mode'):
                candidates.append('refinement')

        prog = (data or {}).get('prog') or {}
        candidates.append(prog.get('output_mode'))
        if isinstance(entry, dict):
            candidates.append(entry.get('output_mode'))

        for candidate in candidates:
            mode = str(candidate or '').lower().strip()
            if 'audio' in mode:
                return 'audio'
            if 'refine' in mode or 'refinement' in mode:
                return 'refinement'
            if mode in ('text', 'vision', 'image', 'video'):
                return mode
        return 'text'

    def _audio_stem_variants(self, output_file):
        stem = os.path.splitext(os.path.basename(output_file or ""))[0]
        if not stem:
            return []
        variants = [stem]
        if stem.startswith("response_"):
            variants.append(stem[len("response_"):])
        else:
            variants.append(f"response_{stem}")
        return list(dict.fromkeys(variants))

    def _normalize_progress_output_name(self, name: str) -> str:
        """Normalize translated/source output names for response_/extension-tolerant matching."""
        if not name:
            return ""
        base = os.path.basename(str(name).replace("\\", "/"))
        if base.lower().startswith("response_"):
            base = base[len("response_"):]
        while True:
            stem, ext = os.path.splitext(base)
            if not ext:
                break
            base = stem
        return base.lower()

    def _resolve_existing_output_path(self, output_dir, output_file=None, display_info=None, prog=None):
        """Resolve an output file while tolerating stale OCR rows and filename mode changes."""
        display_info = display_info or {}
        progress_entry = display_info.get("info") or display_info.get("progress_entry") or {}
        prog = prog or {}
        chapters = prog.get("chapters", {}) if isinstance(prog, dict) else {}
        candidates = []

        def add_candidate(value):
            if value:
                text = str(value).replace("\\", "/")
                if text not in candidates:
                    candidates.append(text)

        add_candidate(output_file)
        add_candidate(display_info.get("output_file"))
        add_candidate(progress_entry.get("output_file") if isinstance(progress_entry, dict) else None)
        if isinstance(progress_entry, dict):
            previous = progress_entry.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        progress_key = display_info.get("progress_key")
        if progress_key and isinstance(chapters.get(progress_key), dict):
            tracked = chapters[progress_key]
            add_candidate(tracked.get("output_file"))
            previous = tracked.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        target_num = display_info.get("num")
        original_names = {
            self._normalize_progress_output_name(display_info.get("original_filename")),
            self._normalize_progress_output_name(display_info.get("original_basename")),
        }
        original_names.discard("")

        for tracked in chapters.values():
            if not isinstance(tracked, dict):
                continue
            tracked_num = tracked.get("actual_num", tracked.get("chapter_num"))
            tracked_names = {
                self._normalize_progress_output_name(tracked.get("output_file")),
                self._normalize_progress_output_name(tracked.get("original_basename")),
                self._normalize_progress_output_name(tracked.get("original_filename")),
            }
            if str(tracked_num) == str(target_num) or (original_names and tracked_names & original_names):
                add_candidate(tracked.get("output_file"))
                previous = tracked.get("previous_progress_entry")
                if isinstance(previous, dict):
                    add_candidate(previous.get("output_file"))

        for candidate in candidates:
            path = candidate if os.path.isabs(candidate) else os.path.join(output_dir, candidate)
            if os.path.isfile(path):
                rel = os.path.relpath(path, output_dir).replace("\\", "/") if os.path.isabs(candidate) else candidate
                return rel, path

        target_norms = {self._normalize_progress_output_name(value) for value in candidates if value}
        target_norms |= original_names
        target_norms.discard("")
        if not target_norms:
            return None, None
        try:
            with os.scandir(output_dir) as _scan:
                for _e in _scan:
                    if _e.is_file() and self._normalize_progress_output_name(_e.name) in target_norms:
                        return _e.name, _e.path
        except Exception:
            pass
        return None, None

    def _audio_candidates_for_entry(self, output_dir, entry):
        """Return possible audio files for a progress entry as (relative, absolute) pairs."""
        candidates = []
        if not isinstance(entry, dict):
            return candidates

        stored = entry.get('tts_file')
        if stored:
            candidates.append(stored)

        output_file = entry.get('output_file')
        if output_file:
            for stem in self._audio_stem_variants(output_file):
                for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                    candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

        seen = set()
        resolved = []
        for candidate in candidates:
            if not candidate:
                continue
            normalized = str(candidate).replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            abs_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
            rel_path = os.path.relpath(abs_path, output_dir).replace("\\", "/") if os.path.isabs(normalized) else normalized
            resolved.append((rel_path, abs_path))
        return resolved

    def _existing_audio_for_entry(self, output_dir, entry, tts_dir_listing=None):
        """Find an existing audio file for a progress entry.

        When ``tts_dir_listing`` (a lowercase set of filenames inside
        ``output_dir/text_to_speech``) is provided, candidates under that
        folder are matched in memory instead of via os.path.exists. This
        matters because the periodic GUI refresh otherwise issues thousands
        of stat calls per tick (chapters x stem variants x extensions), which
        stalls the GUI thread for seconds while translation workers and AV
        scanning saturate the disk in frozen builds.
        """
        for rel_path, abs_path in self._audio_candidates_for_entry(output_dir, entry):
            if tts_dir_listing is not None:
                rel_norm = str(rel_path).replace("\\", "/")
                parent, _, base = rel_norm.rpartition("/")
                if parent == "text_to_speech":
                    if base.lower() in tts_dir_listing:
                        return rel_path, abs_path
                    continue
            if os.path.exists(abs_path):
                return rel_path, abs_path
        return None, None

    def _reconcile_tts_audio_files(self, data):
        """Keep progress TTS status aligned with generated audio files on disk."""
        prog = data.get('prog') or {}
        output_dir = data.get('output_dir')
        if not output_dir:
            return False
        chapters = prog.get('chapters', {})
        if (
            self._current_progress_output_mode(data) != 'audio'
            and not any(
                _progress_entry_has_meaningful_tts_state(entry)
                for entry in chapters.values()
            )
        ):
            return False

        # One directory listing per reconcile pass instead of per-candidate
        # os.path.exists probes (~chapters x ~18 stats each tick on the GUI
        # thread — the main "Not Responding" source during translation start).
        # Prefer the snapshot prefetched off-thread by the silent auto-refresh.
        tts_dir_listing = data.pop('_prefetched_tts_listing', None)
        if tts_dir_listing is None:
            try:
                tts_dir_listing = {
                    name.lower() for name in os.listdir(os.path.join(output_dir, "text_to_speech"))
                }
            except OSError:
                tts_dir_listing = set()

        changed = False
        now = time.time()
        for _key, entry in chapters.items():
            if not isinstance(entry, dict):
                continue
            output_file = entry.get('output_file')
            if not output_file:
                continue
            rel_audio, _abs_audio = self._existing_audio_for_entry(output_dir, entry, tts_dir_listing)
            tts_status = str(entry.get('tts_status') or 'no_tts').lower().strip()

            if rel_audio:
                if tts_status not in ('tts_completed', 'completed') or entry.get('tts_file') != rel_audio:
                    entry['tts_status'] = 'tts_completed'
                    entry['tts_file'] = rel_audio
                    entry.pop('tts_error', None)
                    entry.setdefault('tts_at', now)
                    entry['last_updated'] = now
                    changed = True
                continue

            had_audio_state = (
                entry.get('tts_file')
                or tts_status in ('tts_completed', 'completed', 'in_progress')
            )
            if had_audio_state:
                entry['tts_status'] = 'no_tts'
                entry.pop('tts_file', None)
                entry.pop('tts_at', None)
                entry.pop('tts_error', None)
                entry['last_updated'] = now
                changed = True
        return changed

    def _progress_display_status(self, info, data=None):
        """Derive the status shown in Progress Manager for post-processing modes."""
        # Generated translation artifacts are translation phases, not chapter
        # post-processing items. Keep their own status in TTS/refine views.
        if (
            self._is_metadata_progress_info(info)
            or self._is_translation_artifact_progress_info(info)
        ):
            status = str(info.get('status') or 'pending').lower()
            if status in ('completed_empty', 'completed_image_only'):
                return 'completed'
            return status
        # Skipped special files (the rows the "Show skipped files" toggle
        # reveals) get their own display status instead of masquerading
        # as Not Translated.
        try:
            if self._progress_entry_is_skipped_special(info):
                return 'skipped'
        except Exception:
            pass
        status = info.get('status', 'unknown')
        entry = info.get('progress_entry') or info.get('info') or {}
        if entry.get('manual_editing_pending'):
            status = 'pending'
        mode = self._current_progress_output_mode(data, entry)

        if status in ('completed_empty', 'completed_image_only'):
            status = 'completed'

        ref_status = str(entry.get('refinement_status') or '').lower().strip()
        tts_status = str(entry.get('tts_status') or '').lower().strip()
        if status == 'in_progress' and (ref_status == 'in_progress' or tts_status == 'in_progress'):
            return 'in_progress'

        if status == 'in_progress' and data and data.get('output_dir'):
            previous_status = str(entry.get('previous_status') or '').lower().strip()
            previous_entry = entry.get('previous_progress_entry')
            if previous_status in ('completed', 'completed_empty', 'completed_image_only') or (
                isinstance(previous_entry, dict)
                and str(previous_entry.get('status') or '').lower().strip() in ('completed', 'completed_empty', 'completed_image_only')
            ):
                _resolved_file, resolved_path = self._resolve_existing_output_path(
                    data.get('output_dir'),
                    info.get('output_file') or entry.get('output_file'),
                    info,
                    data.get('prog'),
                )
                if resolved_path and os.path.exists(resolved_path):
                    return 'completed'

        if status in ('failed', 'qa_failed', 'in_progress', 'pending', 'merged', 'not_translated'):
            return status
        if mode == 'refinement':
            ref_status = ref_status or 'not_refined'
            if ref_status in ('failed', 'error'):
                return 'refine_failed'
            if ref_status == 'in_progress':
                return 'in_progress'
            if ref_status not in ('refined', 'completed'):
                return 'not_refined'
        if mode == 'audio':
            tts_status = tts_status or 'no_tts'
            if tts_status in ('failed', 'error'):
                return 'failed'
            if tts_status == 'in_progress':
                return 'in_progress'
            if tts_status not in ('tts_completed', 'completed'):
                return 'no_tts'
        return status

    def _progress_entry_is_refined(self, info):
        """Return True when a completed progress entry also has refined output."""
        try:
            entry = info.get('progress_entry') or info.get('info') or info
            if not isinstance(entry, dict):
                return False
            return str(entry.get('refinement_status') or '').lower().strip() in ('refined', 'completed')
        except Exception:
            return False

    def _progress_entry_refinement_failed(self, info):
        """Return True when translation completed but refinement failed."""
        try:
            entry = info.get('progress_entry') or info.get('info') or info
            if not isinstance(entry, dict):
                return False
            return str(entry.get('refinement_status') or '').lower().strip() in ('failed', 'error')
        except Exception:
            return False

    def _update_chapter_status_info(self, data):
        """Update chapter status information after refresh"""
        # Re-check file existence and update status for each chapter
        for info in data['chapter_display_info']:
            if info.get("is_chunk_progress"):
                chunk_entry = data.get("prog", {}).get(
                    "chapter_chunks", {}
                ).get(str(info.get("chunk_progress_key") or ""))
                if isinstance(chunk_entry, dict):
                    ensure_chunk_entry_schema(chunk_entry)
                    record = chunk_entry.get("entries", {}).get(
                        str(info.get("chunk_index"))
                    )
                    if isinstance(record, dict):
                        info["info"] = record
                        info["status"] = record.get("status", "pending")
                continue
            output_file = info['output_file']
            resolved_output_file, resolved_output_path = self._resolve_existing_output_path(
                data['output_dir'],
                output_file,
                info,
                data.get('prog'),
            )
            output_path = resolved_output_path or os.path.join(data['output_dir'], output_file)
            
            # Find matching progress entry
            matched_info = None
            matched_key = None
            chapters = data['prog'].get("chapters", {})

            progress_key = info.get('progress_key')
            if progress_key and isinstance(chapters.get(progress_key), dict):
                matched_key = progress_key
                matched_info = chapters[progress_key]
            
            # PRIORITY 1: Match by BOTH actual_num AND output_file
            # This prevents cross-matching between files with same chapter number but different filenames
            if not matched_info:
                for chapter_key, chapter_info in chapters.items():
                    actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                    ch_output = chapter_info.get('output_file')
                    
                    # BOTH must match - no fallback
                    if actual_num is not None and actual_num == info['num'] and ch_output == output_file:
                        matched_key = chapter_key
                        matched_info = chapter_info
                        break
            
            # PRIORITY 2: Fall back to output_file matching if no actual_num match
            if not matched_info:
                # Prefer completed over failed/pending/in_progress; keep qa_failed highest
                severity = {'qa_failed': 5, 'completed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1}
                best_key = None
                best = None
                best_score = -1
                for chapter_key, chapter_info in chapters.items():
                    if chapter_info.get('output_file') == output_file:
                        status = chapter_info.get('status', 'unknown')
                        score = severity.get(status, -1)
                        # Prefer higher severity; tie-breaker: matching actual_num if present
                        matches_num = (chapter_info.get('actual_num') or chapter_info.get('chapter_num')) == info['num']
                        if score > best_score or (score == best_score and matches_num):
                            best_score = score
                            best_key = chapter_key
                            best = chapter_info
                if best:
                    matched_key = best_key
                    matched_info = best
            
            # Update status based on current state from progress file
            if matched_info:
                new_status = (
                    'pending'
                    if matched_info.get('manual_editing_pending')
                    else matched_info.get('status', 'unknown')
                )
                # Handle legacy completed variants as completed for display
                if new_status in ('completed_empty', 'completed_image_only'):
                    new_status = 'completed'
                chunk_key = str(
                    matched_info.get("content_hash")
                    or matched_key
                    or ""
                )
                chunk_entry = data.get("prog", {}).get(
                    "chapter_chunks", {}
                ).get(chunk_key)
                if isinstance(chunk_entry, dict):
                    new_status = effective_parent_status(
                        new_status,
                        chunk_entry,
                    )
                # Verify file actually exists for completed status (but NOT for merged - merged chapters
                # don't have their own output files, they point to parent's file)
                if new_status == 'completed' and not os.path.exists(output_path):
                    new_status = 'not_translated'
                elif new_status == 'completed' and resolved_output_file:
                    info['output_file'] = resolved_output_file
                info['status'] = new_status
                info['info'] = matched_info
                info['progress_key'] = matched_key
            elif os.path.exists(output_path):
                # Before marking as completed based on file existence, check if this chapter
                # is actually marked as merged in the progress file (by actual_num lookup)
                # This handles the case where old output files exist from before merging was enabled
                is_merged_chapter = False
                for chapter_key, chapter_info in chapters.items():
                    actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                    if actual_num is not None and actual_num == info['num']:
                        if chapter_info.get('status') == 'merged':
                            is_merged_chapter = True
                            info['status'] = 'merged'
                            info['info'] = chapter_info
                            info['progress_key'] = chapter_key
                            break
                
                if not is_merged_chapter:
                    info['status'] = 'completed'
                    info.pop('info', None)
                    info.pop('progress_entry', None)
                    info.pop('progress_key', None)
            else:
                info['status'] = 'not_translated'
                info.pop('info', None)
                info.pop('progress_entry', None)
                info.pop('progress_key', None)

    def _progress_entry_model_name(self, info, data=None):
        """Return the model name attached to a progress row, with old-file fallbacks."""
        if _progress_entry_is_completed_image_only_for_display(info):
            return "COPIED"

        candidates = []
        if isinstance(info, dict):
            candidates.append(info)
            for key in ('info', 'progress_entry'):
                value = info.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
        if isinstance(data, dict):
            prog = data.get('prog')
            if isinstance(prog, dict):
                progress_key = info.get('progress_key') if isinstance(info, dict) else None
                chapters = prog.get('chapters', {})
                if progress_key and isinstance(chapters, dict) and isinstance(chapters.get(progress_key), dict):
                    candidates.append(chapters[progress_key])

        seen = set()
        for root in candidates:
            candidate = root
            while isinstance(candidate, dict) and id(candidate) not in seen:
                seen.add(id(candidate))
                model_name = str(
                    candidate.get('model_name')
                    or candidate.get('model')
                    or ''
                ).strip()
                if not (
                    candidate.get('subtitle_no_translatable_text')
                    and model_name.lower() == 'no api needed'
                ) and model_name:
                    return model_name
                candidate = candidate.get('previous_progress_entry')
        return "(model unknown)"

    def _progress_model_column_text(self, info, data, fallback_output):
        if isinstance(data, dict) and data.get('show_model_info_state'):
            if _progress_status_hides_model_for_display(
                info.get('status') if isinstance(info, dict) else None
            ):
                return ''
            return self._progress_entry_model_name(info, data)
        return fallback_output

    def _progress_list_column_widths(self, chapter_display_info, data):
        max_original_len = 0
        max_output_len = 0
        for info in chapter_display_info or []:
            if 'opf_position' not in info:
                continue
            original_file = info.get('original_filename', '')
            output_file = self._progress_model_column_text(info, data, info.get('output_file', ''))
            max_original_len = max(max_original_len, len(original_file))
            max_output_len = max(max_output_len, len(output_file))
        return max(max_original_len, 20), max(max_output_len, 25)

    def _progress_list_item_key(self, info):
        if not isinstance(info, dict):
            return None
        if info.get("is_chunk_progress"):
            return (
                f"chunk:{info.get('chunk_progress_key')}:"
                f"{info.get('chunk_index')}"
            )
        # A progress_key is transient for OPF rows: it appears when a chapter
        # enters progress and may disappear when its progress entry is removed.
        # Base identity on the source row so a status change never looks like a
        # remove/insert operation to the streamed list reconciler.
        opf_position = info.get('opf_position')
        if opf_position is not None:
            return f"opf:{opf_position}:{info.get('original_filename', '')}"
        source_key = info.get('key')
        if source_key not in (None, ''):
            return f"source:{source_key}"
        progress_key = info.get('progress_key')
        if progress_key:
            return f"progress:{progress_key}"
        output_file = info.get('output_file')
        if output_file:
            return f"output:{output_file}"
        return f"row:{info.get('num')}:{info.get('original_filename', '')}:{info.get('key', '')}"

    def _progress_list_payload_revision(self, info):
        """Return a cheap revision marker for context-menu row metadata."""
        if not isinstance(info, dict):
            return ()
        entry = info.get('info') or info.get('progress_entry') or {}
        if not isinstance(entry, dict):
            entry = {}
        previous = entry.get('previous_progress_entry')
        if not isinstance(previous, dict):
            previous = {}
        qa_issues = entry.get('qa_issues_found')
        if not isinstance(qa_issues, (list, tuple)):
            qa_issues = ()
        return (
            info.get('progress_key'),
            info.get('status'),
            info.get('output_file'),
            info.get('original_filename'),
            info.get('num'),
            info.get('display_num'),
            info.get('duplicate_count'),
            entry.get('last_updated'),
            entry.get('status'),
            entry.get('output_file'),
            entry.get('model_name') or entry.get('model'),
            entry.get('manual_editing_pending'),
            entry.get('refinement_status'),
            entry.get('tts_status'),
            entry.get('tts_file'),
            entry.get('merged_parent_chapter'),
            tuple(str(issue) for issue in qa_issues),
            previous.get('last_updated'),
            previous.get('status'),
            previous.get('model_name') or previous.get('model'),
        )


    def _progress_row_parts(self, info, data):
        """Status, icon, label and model column of one row (RG 33158-33193).

        The first half of ``_progress_list_display_text``; the desktop formats the
        parts into its monospace line, the mobile ``present_row`` into a list tile.
        """
        status_icons = PM_STATUS_ICONS
        status_labels = PM_STATUS_LABELS

        chapter_num = info.get('display_num', info['num'])
        status = self._progress_display_status(info, data)
        output_file = info['output_file']
        output_display = self._progress_model_column_text(info, data, output_file)
        show_model_info = bool(
            isinstance(data, dict) and data.get('show_model_info_state')
        )
        hide_model_info = (
            show_model_info
            and _progress_status_hides_model_for_display(status)
        )
        pipe_column_suffix = '' if hide_model_info else f" | {output_display}"
        arrow_column_suffix = '' if hide_model_info else f" -> {output_display}"
        icon = status_icons.get(status, '❓')
        status_label = status_labels.get(status, status)
        if (
            status == 'completed'
            and _progress_entry_is_completed_image_only_for_display(info)
        ):
            icon = '📸'
            status_label = 'Image Only (Completed)'
        elif status == 'completed' and self._progress_entry_refinement_failed(info):
            status_label = f"{status_label} 💀"
        elif status == 'completed' and self._progress_entry_is_refined(info):
            status_label = f"{status_label} ⭐"
        chapter_info = info.get('info') or info.get('progress_entry') or {}
        ocr_progress = chapter_info.get('ocr_progress') if isinstance(chapter_info, dict) else None
        if status == 'in_progress' and isinstance(ocr_progress, dict):
            try:
                ocr_done = int(ocr_progress.get('done', 0))
                ocr_total = int(ocr_progress.get('total', 0))
            except (TypeError, ValueError):
                ocr_done = 0
                ocr_total = 0
            if ocr_total > 0:
                status_label = f"{status_label} ({min(ocr_done, ocr_total)}/{ocr_total})"
        return {
            'chapter_num': chapter_num,
            'status': status,
            'output_file': output_file,
            'output_display': output_display,
            'show_model_info': show_model_info,
            'hide_model_info': hide_model_info,
            'pipe_column_suffix': pipe_column_suffix,
            'arrow_column_suffix': arrow_column_suffix,
            'icon': icon,
            'status_label': status_label,
            'chapter_info': chapter_info,
            'ocr_progress': ocr_progress,
        }

    def _progress_list_display_text(self, info, data, max_original_len, max_output_len):
        _parts = self._progress_row_parts(info, data)
        chapter_num = _parts['chapter_num']
        status = _parts['status']
        output_file = _parts['output_file']
        hide_model_info = _parts['hide_model_info']
        pipe_column_suffix = _parts['pipe_column_suffix']
        arrow_column_suffix = _parts['arrow_column_suffix']
        icon = _parts['icon']
        status_label = _parts['status_label']
        chapter_info = _parts['chapter_info']

        if info.get("is_chunk_progress"):
            try:
                if info.get("pdf_toc_section"):
                    pdf_parent = float(chapter_num)
                    chapter_label = (
                        f"Section {int(pdf_parent):03d}"
                        if pdf_parent.is_integer()
                        else f"Section {chapter_num}"
                    )
                else:
                    chapter_label = (
                        f"Ch.{int(chapter_num):03d}"
                        if not isinstance(chapter_num, float)
                        or chapter_num.is_integer()
                        else f"Ch.{chapter_num:06.1f}"
                    )
            except (TypeError, ValueError):
                chapter_label = f"Ch.{chapter_num}"
            display = (
                f"   ↳ {chapter_label} Chunk {info.get('chunk_index')}/"
                f"{info.get('total_chunks')} | {icon} {status_label:11s}"
                f"{pipe_column_suffix}"
            )
        elif self._is_metadata_progress_info(info):
            metadata_label = (
                info.get('metadata_label')
                or (info.get('info') or {}).get('metadata_label')
                or 'Metadata'
            )
            display = (
                f"{metadata_label} | {icon} {status_label:11s} | "
                f"metadata.json{arrow_column_suffix}"
            )
        elif self._is_translation_artifact_progress_info(info):
            artifact_label = (
                info.get('translation_artifact_label')
                or (info.get('info') or {}).get(
                    'translation_artifact_label'
                )
                or output_file
            )
            display = (
                f"{artifact_label} | {icon} {status_label:11s} | "
                f"{output_file}{arrow_column_suffix}"
            )
        elif info.get('pdf_toc_section') or chapter_info.get('pdf_toc_section'):
            start_page = (
                info.get('pdf_start_page')
                if info.get('pdf_start_page') is not None
                else chapter_info.get('pdf_start_page')
            )
            end_page = (
                info.get('pdf_end_page')
                if info.get('pdf_end_page') is not None
                else chapter_info.get('pdf_end_page')
            )
            if start_page is not None and end_page is not None:
                page_label = (
                    f"Page {start_page}"
                    if str(start_page) == str(end_page)
                    else f"Pages {start_page}-{end_page}"
                )
            else:
                page_label = "Pages unknown"
            try:
                numeric_num = float(chapter_num)
                section_label = (
                    f"{int(numeric_num):03d}"
                    if numeric_num.is_integer()
                    else f"{numeric_num:05.1f}"
                )
            except (TypeError, ValueError):
                section_label = str(chapter_num)
            display = (
                f"Section {section_label} | {icon} {status_label:14s} | "
                f"{page_label}{pipe_column_suffix}"
            )
        elif info.get('pdf_ocr'):
            display = (
                f"PDF OCR | {icon} {status_label:18s}"
                f"{pipe_column_suffix}"
            )
        elif info.get('is_subtitle'):
            original_file = info.get('original_filename') or os.path.basename(output_file)
            try:
                completed_batches = int(info.get('subtitle_completed_batches') or 0)
                total_batches = int(info.get('subtitle_total_batches') or 0)
            except (TypeError, ValueError):
                completed_batches = 0
                total_batches = 0
            batch_label = (
                f" | Batches {completed_batches}/{total_batches}"
                if total_batches > 1
                else ""
            )
            display = (
                f"Subtitle {int(chapter_num):03d} | {icon} {status_label:11s} | "
                f"{original_file}{arrow_column_suffix}{batch_label}"
            )
        elif 'opf_position' in info:
            original_file = info.get('original_filename', '')
            opf_pos = info['opf_position'] + 1
            if isinstance(chapter_num, float):
                if chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}}{arrow_column_suffix}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:06.1f} | {icon} {status_label:11s} | {original_file:<{max_original_len}}{arrow_column_suffix}"
            else:
                display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}}{arrow_column_suffix}"
        else:
            if isinstance(chapter_num, float) and chapter_num.is_integer():
                display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:11s}{pipe_column_suffix}"
            elif isinstance(chapter_num, float):
                display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:11s}{pipe_column_suffix}"
            else:
                display = f"Chapter {chapter_num:03d} | {icon} {status_label:11s}{pipe_column_suffix}"

        if hide_model_info:
            display = display.rstrip()

        if status == 'qa_failed':
            qa_issues = chapter_info.get('qa_issues_found', []) if isinstance(chapter_info, dict) else []
            if qa_issues:
                qa_issue_previews = chapter_info.get('qa_issue_previews', {})
                if not isinstance(qa_issue_previews, dict):
                    qa_issue_previews = {}

                issues_display = ', '.join(
                    _format_qa_issue_for_progress_display(issue, qa_issue_previews)
                    for issue in qa_issues[:2]
                )
                if len(qa_issues) > 2:
                    issues_display += f' (+{len(qa_issues)-2} more)'
                display += f" | {issues_display}"

        if status == 'merged':
            parent_chapter = chapter_info.get('merged_parent_chapter') if isinstance(chapter_info, dict) else None
            if parent_chapter:
                display += f" | → Ch.{parent_chapter}"

        if not info.get('is_subtitle') and info.get('duplicate_count', 1) > 1:
            display += f" | ({info['duplicate_count']} entries)"

        return display, status



# ---------------------------------------------------------------------------
# ProgressOwner: a config-backed owner for the shared methods (mobile, tests)
# ---------------------------------------------------------------------------

#: Owner attributes the special-file rules read with a default that differs from the
#: config default; seeded from the config exactly as the desktop start-up seeds them.
_PROGRESS_OWNER_VARS = (
    'translate_special_files_var',
    'special_file_keywords_var',
    'special_file_exact_var',
    'translate_all_numbered_html_var',
)


class ProgressOwner(ProgressViewMixin):
    """GUI-free owner of the Progress Manager methods.

    ``config`` is the app config dict (read live, so a setting changed in Settings
    applies on the next refresh, like desktop).  The special-file rules are the
    translation pipeline's (``GlossaryPipelineMixin._is_special_file``) on attributes
    seeded through ``settings_rules`` (the values desktop start-up gives them);
    ``save_config`` persists through the caller's callback.
    """

    def __init__(self, config, *, save_config=None, **attrs):
        self.config = config if isinstance(config, dict) else {}
        self._save_config_cb = save_config
        self._progress_notices = []
        self.reseed()
        for name, value in attrs.items():
            setattr(self, name, value)

    def reseed(self):
        """Re-read the seeded attributes from ``config`` (after a settings change)."""
        try:
            from settings_rules import _config_var
        except Exception:
            _config_var = None
        for name in _PROGRESS_OWNER_VARS:
            try:
                value = _config_var(self.config, name) if _config_var else None
            except KeyError:
                continue
            if name == 'special_file_exact_var':
                from owner_state import ConfigStateMixin
                if not hasattr(type(self), '_LEGACY_SPECIAL_FILE_EXACT_TOKENS'):
                    type(self)._LEGACY_SPECIAL_FILE_EXACT_TOKENS = (
                        ConfigStateMixin._LEGACY_SPECIAL_FILE_EXACT_TOKENS
                    )
                value = ConfigStateMixin._upgrade_special_file_exact(self, value)
            setattr(self, name, value)

    def _is_special_file(self, filename):
        from translation_pipeline import GlossaryPipelineMixin
        return GlossaryPipelineMixin._is_special_file(self, filename)

    def _should_skip_special_file(self, filename, translate_special=False):
        from translation_pipeline import GlossaryPipelineMixin
        return GlossaryPipelineMixin._should_skip_special_file(self, filename, translate_special)

    def _get_output_mode(self):
        from run_env import RunEnvMixin
        return RunEnvMixin._get_output_mode(self)

    def save_config(self, show_message=False):
        if callable(self._save_config_cb):
            self._save_config_cb(self.config)
        return True

    def take_notices(self):
        """Messages the shared code raised since the last call: ``[(kind, title, text)]``."""
        notices = list(self._progress_notices)
        self._progress_notices.clear()
        return notices


# ---------------------------------------------------------------------------
# Thin public API over the moved methods (names of the U5 shared contract)
# ---------------------------------------------------------------------------


def resolve_output_dir(owner, source_path, *, fixed_output_dir=None):
    """Output folder of a source: fixed folder > OUTPUT_DIRECTORY/OUTPUT_DIR > config > stem."""
    return owner._progress_view_output_dir(source_path, fixed_output_dir)[2]


def ensure_workspace(owner, output_dir):
    """Create a missing output folder with an empty v2.1 progress file; False on failure."""
    return owner._ensure_progress_view_workspace(output_dir)


def load_progress(path, *, fallback=None):
    """The progress JSON at ``path`` ({} or ``fallback`` when missing or unreadable)."""
    try:
        return _read_progress_file(path)
    except (OSError, ValueError):
        return copy.deepcopy(fallback) if isinstance(fallback, dict) else {}


def read_spine(owner, epub_path):
    """``(spine_chapters, opf_chapter_order)`` of an EPUB (empty for other sources)."""
    spine_chapters, opf_chapter_order, _is_epub, _opf_parsed = owner._read_progress_view_spine(epub_path)
    return spine_chapters, opf_chapter_order


def reconcile_workspace(owner, data, *, read_only=True):
    """Reload ``data['prog']`` and rebuild ``data['chapter_display_info']`` (desktop refresh).

    ``read_only`` ticks never write; a full reconcile seeds/cleans and commits through
    the three-way merge.  Returns False when the refresh stopped early.
    """
    data['_refresh_read_only'] = bool(read_only)
    try:
        owner._resolve_progress_view_output_dir(data)
        return owner._reload_progress_view_data(data)
    finally:
        data['_refresh_read_only'] = False


def match_spine(owner, data, append_auxiliary=True):
    """Re-match ``data['spine_chapters']`` against ``data['prog']`` (fills chapter_display_info)."""
    owner._rematch_spine_chapters(data, append_auxiliary=append_auxiliary)
    return data['chapter_display_info']


def build_fallback_rows(owner, data):
    """Rows of a source without an OPF spine (text, PDF, subtitles)."""
    owner._rebuild_chapter_display_info(data)
    return data['chapter_display_info']


def append_aux_rows(owner, data, rows):
    """Chunk children, metadata, TOC/header artifacts, PDF OCR and image-generation rows."""
    owner._append_chunk_progress_display_info(data, rows)
    owner._append_metadata_display_info(data, rows)
    owner._append_translation_artifact_display_info(data, rows)
    owner._append_pdf_ocr_display_info(data, rows)
    owner._append_image_gen_display_info(data, rows)
    return rows


def display_status(owner, row, data):
    """The output-mode-aware status a row shows (desktop ``_progress_display_status``)."""
    return owner._progress_display_status(row, data)


# ---------------------------------------------------------------------------
# Presentation: one structured row per desktop list row
# ---------------------------------------------------------------------------


@dataclass
class RowPresentation:
    """One Progress Manager row as data (the desktop line plus its pieces)."""

    row_id: str
    kind: str
    status: str
    icon: str
    label: str
    color: str
    title: str
    subtitle: str
    model: str
    badges: List[str]
    qa_issues: List[str]
    qa_more: int
    hidden: bool
    is_special: bool
    progress_key: Optional[str]
    output_file: str
    chunk_index: Optional[int]
    total_chunks: Optional[int]
    parent_key: Optional[str]
    chunk_summary: str
    text: str
    info: Dict[str, Any] = dataclass_field(repr=False, default_factory=dict)


def _row_kind(owner, info):
    if info.get('is_chunk_progress'):
        return 'chunk'
    if owner._is_metadata_progress_info(info):
        return 'metadata'
    if owner._is_translation_artifact_progress_info(info):
        return 'artifact'
    chapter_info = info.get('info') or info.get('progress_entry') or {}
    if info.get('pdf_toc_section') or (isinstance(chapter_info, dict) and chapter_info.get('pdf_toc_section')):
        return 'pdf_section'
    if info.get('pdf_ocr'):
        return 'pdf_ocr'
    if info.get('image_gen'):
        return 'image_gen'
    if info.get('is_subtitle'):
        return 'subtitle'
    if 'opf_position' in info:
        return 'chapter'
    return 'fallback'


def _chapter_label(num, prefix='Ch.'):
    try:
        if isinstance(num, float) and not num.is_integer():
            return f"{prefix}{num:06.1f}"
        return f"{prefix}{int(num):03d}"
    except (TypeError, ValueError):
        return f"{prefix}{num}"


def present_row(owner, info, data, *, show_special_files=None, widths=None):
    """Present one row of ``data['chapter_display_info']`` for a list UI.

    ``text`` is the desktop line (``_progress_list_display_text``); ``title`` /
    ``subtitle`` / ``badges`` are its pieces as UI_SPEC §3.7 lays them out.
    """
    if widths is None:
        widths = data.get('_progress_list_column_widths') or owner._progress_list_column_widths(
            data.get('chapter_display_info') or [], data
        )
    text, status = owner._progress_list_display_text(info, data, widths[0], widths[1])
    parts = owner._progress_row_parts(info, data)
    kind = _row_kind(owner, info)
    chapter_info = parts['chapter_info'] if isinstance(parts['chapter_info'], dict) else {}
    num = parts['chapter_num']
    if show_special_files is None:
        show_special_files = bool(data.get('show_special_files_state'))
    hidden = bool(owner._progress_entry_needs_special_visibility(info) and not show_special_files)

    if kind == 'chunk':
        base = (
            _chapter_label(num, 'Section ') if info.get('pdf_toc_section')
            else _chapter_label(num)
        )
        title = f"↳ {base} · Chunk {info.get('chunk_index')}/{info.get('total_chunks')}"
    elif kind == 'metadata':
        title = f"Metadata: {info.get('metadata_label') or chapter_info.get('metadata_label') or 'Metadata'}"
    elif kind == 'artifact':
        title = str(
            info.get('translation_artifact_label')
            or chapter_info.get('translation_artifact_label')
            or parts['output_file']
        )
    elif kind == 'pdf_section':
        start_page = info.get('pdf_start_page') if info.get('pdf_start_page') is not None else chapter_info.get('pdf_start_page')
        end_page = info.get('pdf_end_page') if info.get('pdf_end_page') is not None else chapter_info.get('pdf_end_page')
        if start_page is not None and end_page is not None:
            pages = f"Page {start_page}" if str(start_page) == str(end_page) else f"Pages {start_page}-{end_page}"
        else:
            pages = "Pages unknown"
        title = f"{_chapter_label(num, 'Section ')} · {pages}"
    elif kind == 'pdf_ocr':
        title = f"PDF OCR · {parts['output_file']}"
    elif kind == 'image_gen':
        title = str(parts['output_file'])
    elif kind == 'subtitle':
        original = info.get('original_filename') or os.path.basename(str(parts['output_file']))
        title = f"Subtitle {_chapter_label(num, '')} · {original}"
        try:
            completed_batches = int(info.get('subtitle_completed_batches') or 0)
            total_batches = int(info.get('subtitle_total_batches') or 0)
        except (TypeError, ValueError):
            completed_batches = total_batches = 0
        if total_batches > 1:
            title += f" · Batches {completed_batches}/{total_batches}"
    elif kind == 'chapter':
        title = f"{_chapter_label(num)} · {info.get('original_filename', '')}"
    else:
        title = f"{_chapter_label(num, 'Chapter ')}"

    model = ''
    if not _progress_status_hides_model_for_display(status):
        model = owner._progress_entry_model_name(info, data)
    subtitle = '' if parts['hide_model_info'] else str(parts['output_display'] or '')
    if kind in ('pdf_ocr', 'image_gen'):
        subtitle = ''

    badges = []
    label = PM_STATUS_LABELS.get(status, status)
    if status == 'completed' and _progress_entry_is_completed_image_only_for_display(info):
        badges.append('📸')
    elif status == 'completed' and owner._progress_entry_refinement_failed(info):
        badges.append('💀')
    elif status == 'completed' and owner._progress_entry_is_refined(info):
        badges.append('⭐')
    ocr_progress = parts['ocr_progress']
    if status == 'in_progress' and isinstance(ocr_progress, dict):
        try:
            ocr_done = int(ocr_progress.get('done', 0))
            ocr_total = int(ocr_progress.get('total', 0))
        except (TypeError, ValueError):
            ocr_done = ocr_total = 0
        if ocr_total > 0:
            badges.append(f"OCR {min(ocr_done, ocr_total)}/{ocr_total}")
    if status == 'merged' and chapter_info.get('merged_parent_chapter'):
        badges.append(f"→ Ch.{chapter_info.get('merged_parent_chapter')}")
    if not info.get('is_subtitle') and info.get('duplicate_count', 1) > 1:
        badges.append(f"({info['duplicate_count']} entries)")

    qa_issues = []
    qa_more = 0
    if status == 'qa_failed':
        found = chapter_info.get('qa_issues_found', []) or []
        if isinstance(found, list) and found:
            previews = chapter_info.get('qa_issue_previews', {})
            if not isinstance(previews, dict):
                previews = {}
            qa_issues = [_format_qa_issue_for_progress_display(issue, previews) for issue in found[:2]]
            qa_more = max(0, len(found) - 2)

    chunk_summary = ''
    if kind == 'chapter' or kind == 'fallback' or kind == 'pdf_section':
        chunk_key = str(chapter_info.get('content_hash') or info.get('progress_key') or '')
        chunk_entry = ((data.get('prog') or {}).get('chapter_chunks') or {}).get(chunk_key)
        if is_multi_chunk_entry(chunk_entry):
            try:
                chunk_summary = chunk_status_summary_text(chunk_entry)
            except Exception:
                chunk_summary = ''

    color = progress_status_color(status)
    return RowPresentation(
        row_id=str(owner._progress_list_item_key(info) or ''),
        kind=kind,
        status=status,
        icon=parts['icon'],
        label=label,
        color=color,
        title=title,
        subtitle=subtitle,
        model=model,
        badges=badges,
        qa_issues=qa_issues,
        qa_more=qa_more,
        hidden=hidden,
        is_special=bool(info.get('is_special')),
        progress_key=info.get('progress_key'),
        output_file=str(parts['output_file'] or ''),
        chunk_index=info.get('chunk_index'),
        total_chunks=info.get('total_chunks'),
        parent_key=info.get('parent_progress_key'),
        chunk_summary=chunk_summary,
        text=text,
        info=info,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class ProgressStats:
    """The statistics row of a Progress Manager view."""

    total: int
    chunks: int
    completed: int
    merged: int
    in_progress: int
    pending: int
    missing: int
    failed: int
    skipped: int
    mode: str
    total_label: str
    missing_label: str
    failed_icon: str
    failed_label: str
    failed_color: str

    @property
    def done_fraction(self):
        """Completed share of the non-skipped rows (0.0 when there are none)."""
        countable = self.total - self.skipped
        return (self.completed / countable) if countable > 0 else 0.0


def compute_stats(owner, data):
    """``ProgressStats`` of a view (desktop ``_update_statistics_display`` numbers)."""
    (total_chapters, chunk_count, completed, merged, in_progress, pending,
     missing, failed, skipped, mode) = owner._progress_statistics(data)
    missing_label, failed_icon, failed_label, failed_color = progress_stats_labels(mode)
    return ProgressStats(
        total=total_chapters,
        chunks=chunk_count,
        completed=completed,
        merged=merged,
        in_progress=in_progress,
        pending=pending,
        missing=missing,
        failed=failed,
        skipped=skipped,
        mode=mode,
        total_label=_progress_total_label(total_chapters, chunk_count),
        missing_label=missing_label,
        failed_icon=failed_icon,
        failed_label=failed_label,
        failed_color=failed_color,
    )


# ---------------------------------------------------------------------------
# Change detection (RG _silent_refresh 29917-29944)
# ---------------------------------------------------------------------------


def _progress_snapshot_listing(progress_file, output_dir, prefetch_tts):
    """``(signatures, listing, tts_listing)`` of a view (RG 29921-29944).

    ``signatures`` = (progress (mtime_ns, size), (#files, hash) of the output folder,
    the same for text_to_speech/ or None).
    """
    progress_signature = _progress_path_signature(progress_file)
    try:
        with os.scandir(output_dir) as scan:
            listing = {entry.name for entry in scan if entry.is_file()}
    except Exception:
        listing = set()
    tts_listing = None
    if prefetch_tts:
        try:
            tts_listing = {
                name.lower()
                for name in os.listdir(os.path.join(output_dir, "text_to_speech"))
            }
        except OSError:
            tts_listing = set()

    signatures = (
        progress_signature,
        (len(listing), hash(frozenset(listing))),
        (
            (len(tts_listing), hash(frozenset(tts_listing)))
            if tts_listing is not None else None
        ),
    )
    return signatures, listing, tts_listing


def snapshot_signature(progress_file, output_dir, *, include_tts=False):
    """Cheap change signature of a view: equal signatures need no rebuild."""
    return _progress_snapshot_listing(progress_file, output_dir, include_tts)[0]


class ProgressPoller:
    """Poll a view's ``snapshot_signature`` on a daemon thread (desktop: 2 s fallback timer).

    ``on_change(signature)`` runs on the poller thread when the signature differs from
    the last one seen; the UI marshals it to its own loop.  ``pause()`` while the page
    is hidden, ``resume()`` (which polls at once) when it shows again.
    """

    def __init__(self, progress_file, output_dir, on_change, *, interval=2.0,
                 include_tts=False, initial_signature=None):
        self.progress_file = progress_file
        self.output_dir = output_dir
        self.on_change = on_change
        self.interval = float(interval)
        self.include_tts = include_tts
        self.last_signature = initial_signature
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._paused = False
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="progress-poller", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        self._wake.set()

    def poll_now(self):
        """Check once on the calling thread; returns the new signature or None if unchanged."""
        signature = snapshot_signature(self.progress_file, self.output_dir, include_tts=self.include_tts)
        if signature != self.last_signature:
            self.last_signature = signature
            return signature
        return None

    def _run(self):
        while not self._stop.is_set():
            if not self._paused:
                try:
                    changed = self.poll_now()
                except Exception:
                    changed = None
                if changed is not None:
                    try:
                        self.on_change(changed)
                    except Exception:
                        pass
            self._wake.wait(self.interval)
            self._wake.clear()


# ---------------------------------------------------------------------------
# Book progress: the mobile Book page's Chapters tab
# ---------------------------------------------------------------------------


@dataclass
class BookProgress:
    """One source file's Progress Manager state (``data`` is the desktop data dict)."""

    owner: Any
    data: Dict[str, Any]
    rows: List[RowPresentation]
    stats: ProgressStats
    signature: Tuple
    created_folder: Optional[str] = None
    notices: List[Tuple[str, str, str]] = dataclass_field(default_factory=list)

    @property
    def output_dir(self):
        return self.data.get('output_dir')

    @property
    def progress_file(self):
        return self.data.get('progress_file')

    @property
    def visible_rows(self):
        return [row for row in self.rows if not row.hidden]


def _present_book(owner, data, *, show_special_files=None):
    rows = data.get('chapter_display_info') or []
    widths = owner._progress_list_column_widths(rows, data)
    data['_progress_list_column_widths'] = widths
    return [
        present_row(owner, info, data, show_special_files=show_special_files, widths=widths)
        for info in rows
    ]


def _include_tts_listing(owner, data):
    prog = data.get('prog') or {}
    return owner._current_progress_output_mode(data) == 'audio' or any(
        _progress_entry_has_meaningful_tts_state(entry)
        for entry in (prog.get('chapters') or {}).values()
    )


def build_book_progress(source_path, config=None, *, owner=None, fixed_output_dir=None,
                        show_special_files=None, show_model_info=None, pump=None):
    """Open a source's progress exactly like the desktop Progress Manager does.

    Creates a missing output folder, links the source (``source_epub.txt`` + Library
    registry), seeds subtitle/PDF/metadata/artifact rows, cleans up missing outputs and
    matches the spine; every write goes through the three-way merge.  Returns None
    when the output folder could not be created (``owner.take_notices()`` says why).
    """
    owner = owner if owner is not None else ProgressOwner(config or {})
    owner._pm_created_folder = None
    data = owner._build_progress_view_data(
        source_path,
        resolved_output_dir=fixed_output_dir,
        _pump_loading=pump,
    )
    if data is None:
        return None
    if show_special_files is None:
        show_special_files = str(source_path).lower().endswith(
            ('.txt', '.pdf', '.csv', '.json', '.srt', '.ass', '.lrc', '.zip')
        )
    if show_model_info is None:
        show_model_info = owner._get_retranslation_show_model_info_state()
    data['show_special_files_state'] = bool(show_special_files)
    data['show_model_info_state'] = bool(show_model_info)
    signature = snapshot_signature(
        data['progress_file'], data['output_dir'], include_tts=_include_tts_listing(owner, data)
    )
    data['_last_applied_snapshot_signatures'] = signature
    created = getattr(owner, '_pm_created_folder', None)
    owner._pm_created_folder = None
    return BookProgress(
        owner=owner,
        data=data,
        rows=_present_book(owner, data),
        stats=compute_stats(owner, data),
        signature=signature,
        created_folder=created,
        notices=owner.take_notices() if hasattr(owner, 'take_notices') else [],
    )


def refresh_book_progress(book, *, read_only=True, force=False):
    """Re-read a book's progress; returns ``book`` unchanged when nothing moved on disk.

    ``read_only`` is the desktop's silent tick (no writes); ``read_only=False`` is the
    explicit Refresh (seed, clean up, commit through the merge).
    """
    owner = book.owner
    data = book.data
    signature = snapshot_signature(
        data['progress_file'], data['output_dir'], include_tts=_include_tts_listing(owner, data)
    )
    if not force and read_only and signature == book.signature:
        return book
    if not reconcile_workspace(owner, data, read_only=read_only):
        return book
    signature = snapshot_signature(
        data['progress_file'], data['output_dir'], include_tts=_include_tts_listing(owner, data)
    )
    data['_last_applied_snapshot_signatures'] = signature
    return BookProgress(
        owner=owner,
        data=data,
        rows=_present_book(owner, data),
        stats=compute_stats(owner, data),
        signature=signature,
        created_folder=None,
        notices=owner.take_notices() if hasattr(owner, 'take_notices') else [],
    )


def set_view_toggles(book, *, show_special_files=None, show_model_info=None, persist=True):
    """Change the Show special files / Show model info toggles of a book view."""
    data = book.data
    if show_special_files is not None:
        data['show_special_files_state'] = bool(show_special_files)
    if show_model_info is not None:
        data['show_model_info_state'] = bool(show_model_info)
        if persist:
            book.owner._persist_retranslation_show_model_info_state(bool(show_model_info))
    book.rows = _present_book(book.owner, data)
    return book


_SUMMARY_CACHE: Dict[Tuple, Any] = {}
_SUMMARY_CACHE_LOCK = threading.Lock()


@dataclass
class BookSummary:
    """Library-card numbers from the same rules as the Book page.

    ``stats`` is the Chapters tab's statistics row (chunk rows included); the card
    fields count chapter-level rows only (chunk rows and skipped special files
    excluded), each with the Progress Manager's display status.
    """

    stats: Optional[ProgressStats]
    output_dir: str
    progress_file: str
    has_progress: bool
    chapters_total: int
    chapters_completed: int
    chapters_failed: int
    chapters_in_progress: int
    chunk_qa_failed_parents: int
    signature: Tuple

    @property
    def completed(self):
        return self.chapters_completed

    @property
    def total(self):
        return self.chapters_total

    @property
    def percent(self):
        return int(round(100 * self.chapters_completed / self.chapters_total)) if self.chapters_total else 0


def compute_book_summary(source_path, config=None, *, owner=None, output_dir=None):
    """Card progress of a source, read-only (never writes, never creates folders).

    Uses the Progress Manager's own matching and status rules on a read-only refresh,
    so a Library card and its Book page never disagree.  ``chunk_qa_failed_parents``
    is the other "failed" rule (a completed chapter with QA-failed chunks), which the
    desktop Library card counts as failed (``library_core._read_progress_summary``).
    Results are cached per (source, progress file, output folder) signature.
    """
    owner = owner if owner is not None else ProgressOwner(config or {})
    if output_dir is None:
        output_dir = resolve_output_dir(owner, source_path)
    progress_file = os.path.join(output_dir, "translation_progress.json")
    try:
        source_stat = os.stat(source_path)
        source_signature = (source_stat.st_mtime_ns, source_stat.st_size)
    except OSError:
        source_signature = None
    signature = (
        os.path.normcase(os.path.abspath(str(source_path))),
        source_signature,
        snapshot_signature(progress_file, output_dir, include_tts=True),
    )
    with _SUMMARY_CACHE_LOCK:
        cached = _SUMMARY_CACHE.get(signature[0])
    if cached is not None and cached.signature == signature:
        return cached
    if not os.path.isdir(output_dir):
        summary = BookSummary(None, output_dir, progress_file, False, 0, 0, 0, 0, 0, signature)
    else:
        spine_chapters, opf_chapter_order = read_spine(owner, source_path)
        data = {
            'file_path': source_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': load_progress(progress_file),
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': [],
            'fixed_output_dir': output_dir,
            'progress_source_is_subtitle': owner._path_is_subtitle_progress_source(source_path),
            '_prefetched_prog': None,
        }
        data['_prefetched_prog'] = data['prog']
        data['_prefetched_prog_path'] = progress_file
        data['_refresh_read_only'] = True
        try:
            owner._reload_progress_view_data(data)
        finally:
            data['_refresh_read_only'] = False
        stats = compute_stats(owner, data)
        chunk_failed = 0
        chapters_total = chapters_completed = chapters_failed = chapters_in_progress = 0
        chunks = (data['prog'].get('chapter_chunks') or {}) if isinstance(data['prog'], dict) else {}
        for info in data.get('chapter_display_info') or []:
            if info.get('is_chunk_progress') or owner._progress_entry_is_skipped_special(info):
                continue
            status = owner._progress_display_status(info, data)
            chapters_total += 1
            chapters_completed += status == 'completed'
            chapters_failed += status in STATUS_GROUPS['failed']
            chapters_in_progress += status == 'in_progress'
            entry = info.get('info') or {}
            if not isinstance(entry, dict):
                continue
            key = str(entry.get('content_hash') or info.get('progress_key') or '')
            chunk_entry = chunks.get(key)
            if (
                status == 'completed'
                and is_multi_chunk_entry(chunk_entry)
                and chunk_failure_summary(chunk_entry).get('failed')
            ):
                chunk_failed += 1
        summary = BookSummary(
            stats, output_dir, progress_file, os.path.isfile(progress_file),
            chapters_total, chapters_completed, chapters_failed, chapters_in_progress,
            chunk_failed, signature,
        )
    with _SUMMARY_CACHE_LOCK:
        if len(_SUMMARY_CACHE) > 512:
            _SUMMARY_CACHE.clear()
        _SUMMARY_CACHE[signature[0]] = summary
    return summary


# ---------------------------------------------------------------------------
# Image-folder Progress Manager (mobile milestone U7)
#
# The data halves of RetranslationMixin._force_retranslation_images_folder and
# _refresh_image_folder_data (``git show 41814faa:src/Retranslation_GUI.py``: output
# lookup RG 26377-26415, the refresh scan 25236-25386, row text 25399-25423, Mark as
# Skipped 26718-26801 and Delete Selected 26857-26881).  The dialog keeps its list,
# selection, confirmations and messages and calls these.  The view still reads the
# pre-2.1 flat / ``images`` progress layout (DISCREPANCIES U5 desktop bug 3: against a
# v2.1 ``chapters`` file no hash key is ever found, so nothing is removed from it);
# its two progress writes now go through ``mutate_progress`` (DISCREPANCIES U7).
# ---------------------------------------------------------------------------


def image_folder_output_dir(self, folder_name, script_dir):
    """The translation output folder of an image folder, or None (RG 26377-26415).

    ``self``: the owner (its ``config['output_directory']`` is used when
    OUTPUT_DIRECTORY is unset); ``script_dir``: the application directory.
    Returns ``(output_dir, possible_output_dirs)``.
    """
    # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
    possible_output_dirs = [
        os.path.join(script_dir, folder_name),  # Script dir + folder name (without extension)
        os.path.join(script_dir, f"{folder_name}_translated"),  # Script dir + folder_translated
        folder_name,  # Just the folder name in current directory
        f"{folder_name}_translated",  # folder_translated in current directory
    ]

    # Check for output directory override
    override_dir = os.environ.get('OUTPUT_DIRECTORY')
    if not override_dir and hasattr(self, 'config'):
        override_dir = self.config.get('output_directory')

    if override_dir:
        # If override is set, check inside it for the folder name
        possible_output_dirs.insert(0, os.path.join(override_dir, folder_name))
        possible_output_dirs.insert(1, os.path.join(override_dir, f"{folder_name}_translated"))

    output_dir = None
    for possible_dir in possible_output_dirs:
        print(f"Checking: {possible_dir}")
        if os.path.exists(possible_dir):
            # Check if it has translation_progress.json or HTML files
            if os.path.exists(os.path.join(possible_dir, "translation_progress.json")):
                output_dir = possible_dir
                print(f"Found output directory with progress tracker: {output_dir}")
                break
            # Check if it has any HTML files
            elif os.path.isdir(possible_dir):
                try:
                    files = os.listdir(possible_dir)
                    if any(f.lower().endswith(('.html', '.xhtml', '.htm')) for f in files):
                        output_dir = possible_dir
                        print(f"Found output directory with HTML files: {output_dir}")
                        break
                except:
                    pass
    return output_dir, possible_output_dirs


def image_folder_not_found_message(folder_name, folder_path, script_dir, possible_output_dirs):
    """The desktop's "Info" text when no output folder exists (RG 26418-26422)."""
    return (f"No translation output found for '{folder_name}'.\n\n"
        f"Selected folder: {folder_path}\n"
        f"Script directory: {script_dir}\n\n"
        f"Checked locations:\n" + "\n".join(f"- {d}" for d in possible_output_dirs))


def scan_image_folder(output_dir, progress_file):
    """Rescan an image folder's output (RG 25236-25386): ``{'file_info', 'progress_data',
    'html_files', 'image_files'}``; ``file_info`` rows are ``{'type': 'translated' |
    'cover', 'file', 'path', 'hash_key', 'output_dir'}`` in display order."""
    def _normalize_output_file(output_file, output_dir):
        if not output_file:
            return None
        # Normalize separators
        normalized = str(output_file).replace('\\', '/')
        # If absolute, try to store relative to output_dir when possible
        if os.path.isabs(normalized):
            try:
                rel = os.path.relpath(normalized, output_dir)
                if not rel.startswith('..'):
                    return rel.replace('\\', '/')
            except Exception:
                pass
            return normalized
        # If it's a relative path, keep as-is (preserve subfolders)
        return normalized

    # ALWAYS reload progress data from file to catch deletions
    progress_data = None
    html_files = []
    has_progress_tracking = os.path.exists(progress_file)

    if has_progress_tracking:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            print(f"🔄 Reloaded progress file from disk")

            # Extract files from progress data (primary source)
            # Check if this is the newer nested structure with 'images' key
            images_dict = progress_data.get('images', {})
            if images_dict:
                # Newer structure: progress_data['images'][hash] = {entry}
                for key, value in images_dict.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = _normalize_output_file(value['output_file'], output_dir)

                        # Only include if file actually exists on disk
                        if output_file and output_file not in html_files:
                            full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                            if os.path.exists(full_path):
                                html_files.append(output_file)
                            else:
                                #print(f"⚠️ File in progress but not on disk: {output_file}")
                                pass
            else:
                # Older structure: progress_data[hash] = {entry}
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = _normalize_output_file(value['output_file'], output_dir)

                        # Only include if file actually exists on disk
                        if output_file and output_file not in html_files:
                            full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                            if os.path.exists(full_path):
                                html_files.append(output_file)
                            else:
                                #print(f"⚠️ File in progress but not on disk: {output_file}")
                                pass
        except Exception as e:
            print(f"Failed to load progress file: {e}")
            has_progress_tracking = False

    # Also scan directory for any HTML files not in progress (fallback)
    if os.path.exists(output_dir):
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if (os.path.isfile(file_path) and 
                    file.lower().endswith(('.html', '.xhtml', '.htm')) and 
                    file not in html_files):
                    html_files.append(file)
        except Exception as e:
            print(f"Error scanning directory: {e}")

    # Rescan cover images
    image_files = []
    images_dir = os.path.join(output_dir, "images")
    if os.path.exists(images_dir):
        try:
            for file in os.listdir(images_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_files.append(file)
        except Exception as e:
            print(f"Error scanning images directory: {e}")

    # Rebuild file_info list
    file_info = []

    # Add translated files (both HTML and generated images)
    for html_file in sorted(set(html_files)):
        # Determine file type and extract info
        file_name = os.path.basename(html_file)
        is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
        is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))

        if is_html:
            match = re.match(r'response_(\d+)_(.+)\.html', file_name)
            if match:
                index = match.group(1)
                base_name = match.group(2)
        elif is_image:
            # For generated images, just use the filename
            base_name = os.path.splitext(file_name)[0]

        # Find hash key if progress tracking exists
        hash_key = None
        if progress_data:
            # Check nested structure first
            images_dict = progress_data.get('images', {})
            if images_dict:
                for key, value in images_dict.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = _normalize_output_file(value['output_file'], output_dir)
                        if output_file and output_file == html_file:
                            hash_key = key
                            break
            else:
                # Check flat structure
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = _normalize_output_file(value['output_file'], output_dir)
                        if output_file and output_file == html_file:
                            hash_key = key
                            break

        file_info.append({
            'type': 'translated',
            'file': html_file,
            'path': html_file if os.path.isabs(html_file) else os.path.join(output_dir, html_file),
            'hash_key': hash_key,
            'output_dir': output_dir
        })

    # Add cover images
    for img_file in sorted(image_files):
        file_info.append({
            'type': 'cover',
            'file': img_file,
            'path': os.path.join(images_dir, img_file),
            'hash_key': None,
            'output_dir': output_dir
        })
    return {
        'file_info': file_info,
        'progress_data': progress_data,
        'html_files': html_files,
        'image_files': image_files,
    }


def image_folder_row_text(info):
    """The list text of one image-folder row (RG 25400-25423)."""
    if info['type'] == 'translated':
        file_name = os.path.basename(info['file'])
        # Check if it's an HTML file or a generated image
        is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
        is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))

        if is_html:
            match = re.match(r'response_(\d+)_(.+)\.html', file_name)
            if match:
                index = match.group(1)
                base_name = match.group(2)
                display = f"📄 Image {index} | {base_name} | ✅ Completed"
            else:
                display = f"📄 {file_name} | ✅ Completed"
        elif is_image:
            # Generated image file (e.g., Test1.png from imagen)
            base_name = os.path.splitext(file_name)[0]
            display = f"🖼️ {base_name} | ✅ Completed"
        else:
            display = f"📄 {file_name} | ✅ Completed"
    elif info['type'] == 'cover':
        display = f"🖼️ Cover | {info['file']} | ⏭️ Skipped (cover)"
    else:
        display = f"📄 {info['file']}"
    return display


def _drop_image_folder_progress_keys(hash_keys, verbose=False):
    """``mutate_progress`` callback removing image entries by hash key from the
    pre-2.1 nested (``images``) or flat layout (RG 26774-26781 / 26869-26878)."""
    def _apply(progress_data_current):
        removed = 0
        for hash_key in hash_keys:
            # Check nested structure first
            if 'images' in progress_data_current and hash_key in progress_data_current['images']:
                del progress_data_current['images'][hash_key]
                removed += 1
                if verbose:
                    print(f"Removed {hash_key} from progress_data['images']")
            # Check flat structure
            elif hash_key in progress_data_current:
                del progress_data_current[hash_key]
                removed += 1
                if verbose:
                    print(f"Removed {hash_key} from progress_data")
        return removed
    return _apply


def mark_image_folder_items_skipped(folder_path, output_dir, progress_file, progress_data_current,
                                   items_to_move, info_list):
    """Mark as Skipped (RG 26718-26810): copy each item's source image into
    ``<output>/images`` (searched in the folder, its parent and the working
    directory), delete its translated HTML and drop its progress entry.

    ``items_to_move``: ``[(row index, file_info row), ...]`` (covers excluded);
    ``info_list`` is updated in place.  Returns ``{'moved', 'failed', 'displays':
    {row index: new list text}, 'info_list'}``.
    """
    displays = {}
    removed_hash_keys = []
    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    moved_count = 0
    failed_count = 0

    for idx, item in items_to_move:
        try:
            # Extract the original image name from the HTML filename
            # Expected format: response_001_imagename.html (also accept compound extensions)
            html_file = item['file']
            html_base = os.path.basename(html_file)
            match = re.match(r'^response_\d+_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', html_base, re.IGNORECASE)

            if match:
                base_name = match.group(1)
                # Try to find the original image with common extensions
                original_found = False

                # Look for the source image in multiple locations
                search_paths = [
                    folder_path,  # Original folder path
                    os.path.dirname(folder_path),  # Parent of folder path
                    os.getcwd(),  # Script directory
                ]

                for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    for search_path in search_paths:
                        if not search_path or not os.path.exists(search_path):
                            continue

                        # Check in the search path
                        possible_source = os.path.join(search_path, base_name + ext)
                        if os.path.exists(possible_source) and os.path.isfile(possible_source):
                            # Copy to images folder
                            dest_path = os.path.join(images_dir, base_name + ext)
                            if not os.path.exists(dest_path):
                                import shutil
                                shutil.copy2(possible_source, dest_path)
                                print(f"Copied {base_name + ext} from {possible_source} to images folder")
                            original_found = True
                            break
                    if original_found:
                        break

                if not original_found:
                    print(f"Warning: Could not find original image for {html_file} in: {search_paths}")
                    # Even if source not found, we can still delete the HTML and mark it

            # Delete the HTML translation file
            if os.path.exists(item['path']):
                os.remove(item['path'])
                print(f"Deleted translation: {item['path']}")

                # Remove from progress tracking if applicable
                if progress_data_current and item.get('hash_key'):
                    hash_key = item['hash_key']
                    removed_hash_keys.append(hash_key)

            # Update the listbox display
            display = f"🖼️ Skipped | {base_name if match else html_base} | ⏭️ Moved to images folder"
            displays[idx] = display

            # Update file_info
            info_list[idx] = {
                'type': 'cover',  # Treat as cover type since it's in images folder
                'file': base_name + ext if match and original_found else html_base,
                'path': os.path.join(images_dir, base_name + ext if match and original_found else html_base),
                'hash_key': None,
                'output_dir': output_dir
            }

            moved_count += 1

        except Exception as e:
            print(f"Failed to process {item['file']}: {e}")
            failed_count += 1
    # Save updated progress if modified
    if progress_data_current:
        try:
            mutate_progress(progress_file, _drop_image_folder_progress_keys(removed_hash_keys))
            print(f"Updated progress tracking file")
        except Exception as e:
            print(f"Failed to update progress file: {e}")
    return {'moved': moved_count, 'failed': failed_count, 'displays': displays, 'info_list': info_list}


def image_folder_mark_skipped_message(result):
    """``(kind, title, message)`` of Mark as Skipped (RG 26820-26827)."""
    moved_count = result['moved']
    failed_count = result['failed']
    if failed_count > 0:
        return ('warning', "Partial Success",
            f"Moved {moved_count} image(s) to be skipped.\n"
            f"Failed to process {failed_count} item(s).")
    return ('info', "Success",
        f"Moved {moved_count} image(s) to the images folder.\n"
        "They will be skipped in future translations.")


def image_folder_delete_confirmation(info_list, selected_indices):
    """Delete Selected's confirmation text (RG 26839-26850)."""
    # Count types
    translated_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'translated')
    cover_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'cover')

    # Build confirmation message
    msg_parts = []
    if translated_count > 0:
        msg_parts.append(f"{translated_count} translated image(s)")
    if cover_count > 0:
        msg_parts.append(f"{cover_count} cover image(s)")

    confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
    return confirm_msg


def delete_image_folder_items(progress_file, progress_data_current, info_list, selected_indices):
    """Delete Selected (RG 26857-26890): delete the selected outputs / covers and drop
    their progress entries.  Returns the number of files deleted."""
    removed_hash_keys = []
    # Delete selected files
    deleted_count = 0

    for idx in selected_indices:
        info = info_list[idx]
        try:
            if os.path.exists(info['path']):
                os.remove(info['path'])
                deleted_count += 1
                print(f"Deleted: {info['path']}")

                # Remove from progress tracking if applicable
                if progress_data_current and info.get('hash_key'):
                    hash_key = info['hash_key']
                    removed_hash_keys.append(hash_key)

        except Exception as e:
            print(f"Failed to delete {info['path']}: {e}")
    # ALWAYS save progress file after any deletions
    if deleted_count > 0 and progress_data_current:
        try:
            mutate_progress(progress_file, _drop_image_folder_progress_keys(removed_hash_keys, verbose=True))
            print(f"Updated progress tracking file")
        except Exception as e:
            print(f"Failed to update progress file: {e}")
    return deleted_count


def image_folder_delete_message(deleted_count):
    """``(kind, title, message)`` of Delete Selected (RG 26896-26898)."""
    return ('info', "Success",
        f"Deleted {deleted_count} file(s).\n\n"
        "They will be retranslated on the next run.")


def build_image_folder_progress(folder_path, config=None, *, owner=None, script_dir=None):
    """Open an image folder's Progress Manager data without widgets.

    Returns ``(data, None)`` with the desktop's refresh data (``type``, ``file_info``,
    ``progress_file``, ``progress_data``, ``output_dir``, ``folder_path``, ``rows``: the
    list texts) or ``(None, (kind, title, message))`` with the desktop's "Info" text
    when there is no output (folder) yet.  The rows are what the desktop list shows
    after the refresh that runs as soon as it opens.
    """
    if owner is None:
        owner = ProgressOwner(dict(config or {}))
    if script_dir is None:
        from app_paths import _get_app_dir
        script_dir = _get_app_dir()
    if os.path.isfile(folder_path):
        folder_name = os.path.splitext(os.path.basename(folder_path))[0]
    else:
        folder_name = os.path.basename(folder_path)
    output_dir, possible_output_dirs = image_folder_output_dir(owner, folder_name, script_dir)
    if not output_dir:
        return None, ('info', "Info", image_folder_not_found_message(
            folder_name, folder_path, script_dir, possible_output_dirs))
    progress_file = os.path.join(output_dir, "translation_progress.json")
    scanned = scan_image_folder(output_dir, progress_file)
    if not scanned['html_files'] and not scanned['image_files']:
        return None, ('info', "Info",
            f"No translated files found in: {output_dir}\n\n"
            f"Progress tracking: {'Yes' if os.path.exists(progress_file) else 'No'}")
    data = {
        'type': 'image_folder',
        'file_info': scanned['file_info'],
        'progress_file': progress_file,
        'progress_data': scanned['progress_data'],
        'output_dir': output_dir,
        'folder_path': folder_path,
        'rows': [image_folder_row_text(info) for info in scanned['file_info']],
    }
    return data, None


__all__ = [
    'BookProgress',
    'BookSummary',
    'METADATA_PROGRESS_KEY',
    'PM_STATUS_COLORS',
    'PM_STATUS_ICONS',
    'PM_STATUS_LABELS',
    'ProgressOwner',
    'ProgressPoller',
    'ProgressStats',
    'ProgressViewMixin',
    'RowPresentation',
    'STATUS_GROUPS',
    'STATUS_VOCAB',
    'append_aux_rows',
    'build_book_progress',
    'build_fallback_rows',
    'build_image_folder_progress',
    'cleanup_missing_files',
    'commit_progress',
    'compute_book_summary',
    'compute_stats',
    'delete_image_folder_items',
    'display_status',
    'ensure_workspace',
    'image_folder_delete_confirmation',
    'image_folder_delete_message',
    'image_folder_mark_skipped_message',
    'image_folder_not_found_message',
    'image_folder_output_dir',
    'image_folder_row_text',
    'load_progress',
    'mark_image_folder_items_skipped',
    'match_spine',
    'merge_progress_changes',
    'mutate_progress',
    'present_row',
    'progress_lock',
    'progress_stats_labels',
    'progress_status_color',
    'read_spine',
    'reconcile_workspace',
    'refresh_book_progress',
    'resolve_output_dir',
    'scan_image_folder',
    'set_view_toggles',
    'snapshot_output_dir',
    'snapshot_signature',
    'write_progress_atomic',
]
