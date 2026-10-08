"""Progress Manager row actions (non-retranslate), GUI-free (mobile milestone U5).

Moved verbatim from ``git show 20b446b0:src/Retranslation_GUI.py`` (RG line numbers):

* module helpers RG 230-289 (QA-mark clearing), 328-423 (pending-mark recovery),
  542-618 / 650-759 (LLM-token / missing-image QA clearing), 762-842 (empty-attribute
  file repair), 1675-1740 (bulk SDLXLIFF sidecar reset/delete);
* the data halves of the action closures of ``_add_retranslation_buttons_opf``:
  ``_normalize_filename`` / ``_find_progress_entry`` (27900-27921), restore in-progress
  (27934-28021), remove QA failed mark (28072-28150), remove refinement status
  (28180-28242), the audio-mode TTS reset of Retranslate Selected (28413-28478), the
  audio file lookup / TTS reset of Delete Audio File (30508-30565), the LLM-token QA
  resolution (30679-30712), Insert Missing Image (30985-31127) and the Partial.b request
  of ``_start_single_progress_qa_resolution`` (18953-18983, 19023-19036).

U7 (``git show 41814faa:src/Retranslation_GUI.py``): Retranslate Selected split into
``plan_retranslation`` / ``apply_retranslation`` / ``retranslation_result_message`` (the
``retranslate_selected`` generator, RG 21921-22959) and the Resolve QA (Partial.b)
preflight ``prepare_single_qa_resolution`` (RG 16647-16719).

The desktop closures keep their selection, confirmation dialogs, refresh and messages
and call these functions; every progress write goes through
``progress_core.mutate_progress`` (lock, re-read, three-way merge, atomic replace),
applying the action to the newest snapshot instead of the dialog's cached copy
(the former whole-file writes are listed in tests/parity/DISCREPANCIES.md, U5), or --
Retranslate Selected, as before -- ``_merge_and_write_retranslation_progress``.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Set, Tuple

from chapter_chunk_progress import (
    chunk_failure_summary,
    ensure_chunk_entry_schema,
    is_multi_chunk_entry,
    remove_chunk_segments_from_file,
    reset_chunks_for_retranslation,
    set_chunk_qa,
    sorted_chunk_items,
)
from metadata_progress import (
    METADATA_PROGRESS_KEY,
    is_metadata_progress_entry,
)
from progress_core import (
    _chunk_ledger_for_progress_entry,
    _clear_refinement_progress_fields,
    _merge_and_write_retranslation_progress,
    _pending_mark_chunk_blocks,
    _pending_mark_output_path,
    _progress_entry_has_llm_token_qa,
    _progress_entry_has_missing_image_qa,
    _progress_entry_has_raw_foreign_text_qa,
    _progress_item_is_html,
    _QA_MARK_FIELDS,
    _CHUNK_QA_MIRROR_FIELDS,
    _MISSING_IMAGE_QA_RE,
    _qa_value_has_llm_token_issue,
    _restore_pending_mark_record,
    _retranslation_progress_lock,
    _sync_parent_chunk_qa_summary,
    mutate_progress,
    progress_entry_has_qa_mark,
)
from sdlxliff_review_core import _sdlxliff_machine_translation_path
from sdlxliff_sidecar_writer import _reset_sdlxliff_target_for_manual_retranslation
from translation_artifacts import (
    reset_translation_artifact_progress_entries,
    translation_artifact_path,
    translation_artifact_spec_for_filename,
    translation_artifact_spec_for_kind,
    translation_artifacts_are_recycled_linked,
)


def clear_progress_entry_qa_mark(prog, progress_key, entry, output_dir=None):
    """Remove the failed mark from one chapter row and from every chunk under it.

    Clearing only the chapter used to leave its chunk records qa_failed (and
    clearing only the chunks left the chapter qa_failed), so the rows came
    back red on the next refresh. Returns True when anything changed.
    """
    if not isinstance(entry, dict):
        return False
    changed = False
    if str(entry.get("status") or "").lower() in ("qa_failed", "failed"):
        entry["status"] = "completed"
        changed = True
    for field in _QA_MARK_FIELDS + _CHUNK_QA_MIRROR_FIELDS:
        if field in entry:
            entry.pop(field, None)
            changed = True
    chunk_key, chunk_entry = _chunk_ledger_for_progress_entry(
        prog, progress_key, entry
    )
    if chunk_entry is not None:
        ensure_chunk_entry_schema(chunk_entry)
        for index, record in sorted_chunk_items(chunk_entry.get("entries", {})):
            if not isinstance(record, dict):
                continue
            marked = (
                str(record.get("status") or "").lower() in ("qa_failed", "failed")
                or record.get("qa_issues_found")
            )
            if marked and set_chunk_qa(chunk_entry, index, []):
                changed = True
        _sync_parent_chunk_qa_summary(prog, progress_key, chunk_key, output_dir)
    if changed:
        entry["last_updated"] = time.time()
    return changed


def clear_chunk_row_qa_mark(prog, info, output_dir=None):
    """Remove the failed mark from one chunk row.

    Once no chunk of the chapter is failed any more, a chapter-level
    qa_failed status has nothing left to point at and is cleared as well.
    """
    chunk_key = str(info.get("chunk_progress_key") or "")
    chunk_entry = prog.get("chapter_chunks", {}).get(chunk_key)
    if not isinstance(chunk_entry, dict):
        return False
    if not set_chunk_qa(chunk_entry, info.get("chunk_index"), []):
        return False
    parent_key = info.get("parent_progress_key")
    summary = _sync_parent_chunk_qa_summary(prog, parent_key, chunk_key, output_dir)
    parent = prog.get("chapters", {}).get(parent_key)
    if (
        isinstance(parent, dict)
        and summary is not None
        and not summary["failed"]
        and str(parent.get("status") or "").lower() in ("qa_failed", "failed")
    ):
        clear_progress_entry_qa_mark(prog, parent_key, parent, output_dir)
    return True

def _recover_pending_marks(prog, output_dir, selected_infos):
    """Apply explicit recovery to fresh state, never to a cached UI snapshot."""
    recovered = skipped = 0
    chapters = prog.get("chapters", {})
    seen = set()
    for info in selected_infos:
        is_chunk = bool(info.get("is_chunk_progress"))
        parent_key = info.get("parent_progress_key") if is_chunk else (
            info.get("progress_key") or info.get("key")
        )
        parent = chapters.get(parent_key)
        if not isinstance(parent, dict):
            # Older display rows may omit their key. Only an unambiguous,
            # exact output filename match may identify their progress entry.
            matches = [
                (key, row) for key, row in chapters.items()
                if isinstance(row, dict) and info.get("output_file")
                and row.get("output_file") == info.get("output_file")
            ]
            if is_chunk or len(matches) != 1:
                skipped += 1
                continue
            parent_key, parent = matches[0]
        chunk_key = str(parent.get("content_hash") or parent_key)
        chunk_entry = prog.get("chapter_chunks", {}).get(chunk_key)
        if is_chunk and str(info.get("chunk_progress_key") or "") != chunk_key:
            skipped += 1
            continue
        identity = (parent_key, str(info.get("chunk_index")) if is_chunk else None)
        if identity in seen:
            continue
        seen.add(identity)
        status = str(parent.get("status") or "").lower()
        if status == "in_progress":
            skipped += 1
            continue
        if not is_chunk and status != "pending" and not parent.get("manual_editing_pending"):
            skipped += 1
            continue
        path = _pending_mark_output_path({"status": "pending", "info": parent}, output_dir)
        if not path:
            skipped += 1
            continue
        if is_multi_chunk_entry(chunk_entry):
            ensure_chunk_entry_schema(chunk_entry)
            blocks = _pending_mark_chunk_blocks(path, chunk_key, chunk_entry)
            records = chunk_entry.get("entries", {})
            indices = [str(info.get("chunk_index"))] if is_chunk else list(records)
            changed = 0
            for key in indices:
                record = records.get(key)
                if not isinstance(record, dict) or record.get("status") != "pending":
                    continue
                block = blocks.get(record.get("index"))
                if block is None:
                    continue
                # The saved HTML is authoritative for this explicit recovery;
                # never overwrite it with an older cached response.
                result = block["content"]
                chunk_entry.setdefault("chunks", {})[key] = result
                record["result_sha256"] = hashlib.sha256(result.encode("utf-8")).hexdigest()
                _restore_pending_mark_record(record)
                changed += 1
            if changed:
                ensure_chunk_entry_schema(chunk_entry)
                summary = chunk_failure_summary(chunk_entry)
                chunk_entry["chapter_status"] = (
                    "qa_failed" if summary["failed"] else
                    "incomplete" if summary["pending"] else "completed"
                )
                chunk_entry["last_updated"] = time.time()
            previous_state = (parent.get("status"), parent.get("manual_editing_pending"))
            if changed or not is_chunk:
                _sync_parent_chunk_qa_summary(prog, parent_key, chunk_key, output_dir)
            if changed or (parent.get("status"), parent.get("manual_editing_pending")) != previous_state:
                recovered += changed or 1
            else:
                skipped += 1
        elif is_chunk:
            skipped += 1
        else:
            _restore_pending_mark_record(parent)
            recovered += 1
    return {"recovered": recovered, "skipped": skipped}


def _remove_pending_marks(progress_file, output_dir, selected_infos):
    """Persist a user-requested recovery using the latest progress snapshot."""
    with _retranslation_progress_lock(progress_file):
        with open(progress_file, "r", encoding="utf-8") as source:
            baseline = json.load(source)
        changed = copy.deepcopy(baseline)
        result = _recover_pending_marks(changed, output_dir, selected_infos)
        if result["recovered"]:
            _merge_and_write_retranslation_progress(progress_file, baseline, changed)
        return result

def _without_llm_token_qa(value):
    """Remove only LLM-token issue values while preserving other QA details."""
    if isinstance(value, list):
        filtered = []
        for item in value:
            cleaned = _without_llm_token_qa(item)
            if cleaned not in (None, "", [], (), {}, set()):
                filtered.append(cleaned)
        return filtered
    if isinstance(value, tuple):
        return tuple(_without_llm_token_qa(list(value)))
    if isinstance(value, set):
        return set(_without_llm_token_qa(list(value)))
    if isinstance(value, dict):
        filtered = {}
        for key, item in value.items():
            if _qa_value_has_llm_token_issue(key):
                continue
            cleaned = _without_llm_token_qa(item)
            if cleaned not in (None, "", [], (), {}, set()):
                filtered[key] = cleaned
        return filtered
    if _qa_value_has_llm_token_issue(value):
        return None
    return value


def _clear_llm_token_qa_markers(entry):
    """Clear resolved LLM-token QA markers and return ``(changed, remaining)``."""
    if not isinstance(entry, dict):
        return False, False

    changed = False
    for key in (
        'qa_issues_found', 'qa_issue_previews', 'failure_reason',
        'error_message', 'qa_issues',
    ):
        if key not in entry:
            continue
        original = entry.get(key)
        filtered = _without_llm_token_qa(original)
        if filtered != original:
            changed = True
        if filtered in (None, "", [], (), {}, set()):
            entry.pop(key, None)
        else:
            entry[key] = filtered

    previous = entry.get('previous_progress_entry')
    previous_remaining = False
    if isinstance(previous, dict):
        previous_changed, previous_remaining = _clear_llm_token_qa_markers(
            previous
        )
        changed = changed or previous_changed

    remaining = previous_remaining or any(
        bool(entry.get(key))
        for key in (
            'qa_issues_found', 'qa_issue_previews', 'failure_reason',
            'error_message',
        )
    )
    qa_flag = entry.get('qa_issues')
    if not isinstance(qa_flag, bool):
        remaining = remaining or bool(qa_flag)

    if not remaining:
        if entry.pop('qa_issues', None) is not None:
            changed = True
        if entry.pop('qa_timestamp', None) is not None:
            changed = True
        if str(entry.get('status') or '').lower() in {'qa_failed', 'failed'}:
            entry['status'] = 'completed'
            entry['last_updated'] = time.time()
            changed = True
    return changed, remaining

def _qa_scalar_is_missing_image_issue(value):
    """Return whether one scalar starts with a missing-image issue marker."""
    if isinstance(value, (dict, list, tuple, set)):
        return False
    return bool(_MISSING_IMAGE_QA_RE.match(str(value or "").strip()))


def _qa_mapping_is_missing_image_issue(value):
    """Identify a structured issue by its code/type fields, not its preview."""
    if not isinstance(value, dict):
        return False
    identity_keys = {'type', 'issue', 'issue_type', 'code', 'name'}
    return any(
        str(key).strip().lower() in identity_keys
        and _qa_scalar_is_missing_image_issue(item)
        for key, item in value.items()
    )


def _without_missing_image_qa(value):
    """Remove complete missing-image issue values while preserving others."""
    if isinstance(value, list):
        filtered = []
        for item in value:
            if (
                _qa_scalar_is_missing_image_issue(item)
                or _qa_mapping_is_missing_image_issue(item)
            ):
                continue
            cleaned = (
                _without_missing_image_qa(item)
                if isinstance(item, (dict, list, tuple, set))
                else item
            )
            if cleaned not in (None, "", [], (), {}, set()):
                filtered.append(cleaned)
        return filtered
    if isinstance(value, tuple):
        return tuple(_without_missing_image_qa(list(value)))
    if isinstance(value, set):
        return set(_without_missing_image_qa(list(value)))
    if isinstance(value, dict):
        if _qa_mapping_is_missing_image_issue(value):
            return None
        filtered = {}
        for key, item in value.items():
            if _qa_scalar_is_missing_image_issue(key):
                continue
            cleaned = (
                _without_missing_image_qa(item)
                if isinstance(item, (dict, list, tuple, set))
                else item
            )
            if cleaned not in (None, "", [], (), {}, set()):
                filtered[key] = cleaned
        return filtered
    if _qa_scalar_is_missing_image_issue(value):
        return None
    return value


def _clear_missing_image_qa_markers(entry):
    """Clear only missing-image QA markers; return ``(changed, remaining)``."""
    if not isinstance(entry, dict):
        return False, False

    changed = False
    for key in ('qa_issues_found', 'qa_issue_previews', 'qa_issues'):
        if key not in entry:
            continue
        original = entry.get(key)
        filtered = _without_missing_image_qa(original)
        if filtered != original:
            changed = True
        if filtered in (None, "", [], (), {}, set()):
            entry.pop(key, None)
        else:
            entry[key] = filtered

    # Error fields can represent independent translation failures. Never
    # remove them merely because a missing-image marker was also present.
    previous = entry.get('previous_progress_entry')
    previous_remaining = False
    if isinstance(previous, dict):
        previous_changed, previous_remaining = _clear_missing_image_qa_markers(
            previous
        )
        changed = changed or previous_changed

    remaining = previous_remaining or any(
        bool(entry.get(key))
        for key in (
            'qa_issues_found', 'qa_issue_previews', 'failure_reason',
            'error_message',
        )
    )
    qa_flag = entry.get('qa_issues')
    if not isinstance(qa_flag, bool):
        remaining = remaining or bool(qa_flag)

    if not remaining:
        if entry.pop('qa_issues', None) is not None:
            changed = True
        if entry.pop('qa_timestamp', None) is not None:
            changed = True
        if str(entry.get('status') or '').lower() in {'qa_failed', 'failed'}:
            entry['status'] = 'completed'
            entry['last_updated'] = time.time()
            changed = True
    return changed, remaining

def _repair_empty_attribute_qa_file(file_path):
    """Apply the shared empty-attribute fixer and verify the issue is gone."""
    from _empty_attr_fix import count_empty_attr_tags, fix_empty_attr_tags

    raw_path = str(file_path or "").strip()
    path = (
        os.path.abspath(os.path.normpath(raw_path))
        if raw_path else ""
    )
    if not path or not os.path.isfile(path):
        return {
            'resolved': False,
            'changed': False,
            'repaired': 0,
            'remaining': 0,
            'repairs': [],
            'error': f"Output file not found: {path or file_path}",
        }
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            original = handle.read()
        before = count_empty_attr_tags(original)
        if before == 0:
            return {
                'resolved': True,
                'changed': False,
                'repaired': 0,
                'remaining': 0,
                'repairs': [],
                'error': '',
            }

        repairs = []
        repaired_content = fix_empty_attr_tags(
            original,
            repair_log=repairs,
            repair_log_limit=20,
        )
        remaining = count_empty_attr_tags(repaired_content)
        if remaining:
            return {
                'resolved': False,
                'changed': False,
                'repaired': before - remaining,
                'remaining': remaining,
                'repairs': repairs,
                'error': (
                    f"{remaining} empty-attribute tag(s) remain after repair."
                ),
            }

        temporary = (
            f"{path}.{os.getpid()}.{threading.get_ident()}.llm-token-fix.tmp"
        )
        try:
            with open(temporary, 'w', encoding='utf-8', newline='') as handle:
                handle.write(repaired_content)
            os.replace(temporary, path)
        finally:
            if os.path.isfile(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        return {
            'resolved': True,
            'changed': repaired_content != original,
            'repaired': before,
            'remaining': 0,
            'repairs': repairs,
            'error': '',
        }
    except Exception as exc:
        return {
            'resolved': False,
            'changed': False,
            'repaired': 0,
            'remaining': 0,
            'repairs': [],
            'error': str(exc),
        }


def _bulk_retranslation_sidecar_updates(
    paths,
    *,
    manual_editing,
    max_workers,
):
    """Reset or delete independent SDLXLIFF sidecars in parallel."""
    unique_paths = []
    seen = set()
    for path in paths or []:
        if not path:
            continue
        normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(os.fspath(path))

    result = {
        "cleared": 0,
        "deleted": 0,
        "failed": [],
    }
    if not unique_paths:
        return result

    try:
        worker_count = min(
            len(unique_paths),
            max(1, int(max_workers or 1)),
        )
    except (TypeError, ValueError):
        worker_count = 1

    def _update_one(path):
        try:
            if manual_editing:
                changed = _reset_sdlxliff_target_for_manual_retranslation(path)
                return "cleared" if changed else "missing", path, None
            try:
                os.remove(path)
            except FileNotFoundError:
                return "missing", path, None
            return "deleted", path, None
        except Exception as exc:
            return "failed", path, str(exc)

    if worker_count == 1:
        outcomes = map(_update_one, unique_paths)
        for status, path, error in outcomes:
            if status in {"cleared", "deleted"}:
                result[status] += 1
            elif status == "failed":
                result["failed"].append((path, error))
        return result

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="RetranslationSidecar",
    ) as executor:
        for status, path, error in executor.map(_update_one, unique_paths):
            if status in {"cleared", "deleted"}:
                result[status] += 1
            elif status == "failed":
                result["failed"].append((path, error))
    return result


# ---------------------------------------------------------------------------
# Closure bodies of _add_retranslation_buttons_opf (RG 27900-31127)
# ---------------------------------------------------------------------------

def _normalize_filename(name: str) -> str:
    if not name:
        return ""
    base = os.path.basename(name)
    if base.startswith("response_"):
        base = base[len("response_"):]
    while True:
        new_base, ext = os.path.splitext(base)
        if not ext:
            break
        base = new_base
    return base

def _find_progress_entry(chapter_info, prog):
    """Strict: match only identical output_file string."""
    target_out = chapter_info.get('output_file')
    if not target_out:
        return None
    for key, ch in prog.get("chapters", {}).items():
        if ch.get('output_file') == target_out:
            return key, ch
    return None


def _restore_regular_in_progress_entry(info, output_dir):
    if not isinstance(info, dict):
        return None
    previous_status = str(info.get('previous_status', '') or '').lower()
    previous_entry = info.get('previous_progress_entry')
    transient_statuses = {'in_progress', 'not_translated', 'not translated', 'not_completed'}
    if isinstance(previous_entry, dict):
        restored = dict(previous_entry)
        restored_status = str(restored.get('status', previous_status) or previous_status).lower()
        if restored_status and restored_status not in transient_statuses:
            restored.pop('previous_status', None)
            restored.pop('previous_progress_entry', None)
            return restored
    if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
        restored = dict(info)
        restored['status'] = 'failed' if previous_status == 'error' else previous_status
        restored.pop('previous_status', None)
        restored.pop('previous_progress_entry', None)
        restored.pop('previous_status_unknown', None)
        return restored
    if info.get('previous_status_unknown'):
        restored = dict(info)
        restored['status'] = 'failed'
        restored.pop('previous_status', None)
        restored.pop('previous_progress_entry', None)
        restored.pop('previous_status_unknown', None)
        return restored
    output_file = info.get('output_file')
    output_exists = bool(output_file and os.path.exists(os.path.join(output_dir, output_file)))
    if previous_status in ('not_translated', 'not translated', 'not_completed', ''):
        if previous_status and not output_exists:
            return None
        if output_exists:
            restored = dict(info)
            restored['status'] = 'failed'
            restored.pop('previous_status', None)
            restored.pop('previous_progress_entry', None)
            restored.pop('previous_status_unknown', None)
            return restored
    return None


def _apply_restore_in_progress(prog, output_dir, in_progress_chapters):
    """Restore In Progress Status on ``prog`` (RG 27992-28021); returns the counts."""
    restored_count = 0
    deleted_count = 0
    failed_count = 0
    progress_updated = False

    for info in in_progress_chapters:
        match = None
        progress_key = info.get('progress_key')
        if progress_key and progress_key in prog.get("chapters", {}):
            match = (progress_key, prog["chapters"][progress_key])
        else:
            match = _find_progress_entry(info, prog)

        if not match:
            print(f"WARNING: Could not find in-progress entry for {info.get('num')} ({info.get('output_file')})")
            continue

        key, entry = match
        restored = _restore_regular_in_progress_entry(entry, output_dir)
        if restored:
            prog["chapters"][key] = restored
            progress_updated = True
            if restored.get('status') == 'failed':
                failed_count += 1
            else:
                restored_count += 1
        else:
            del prog["chapters"][key]
            progress_updated = True
            deleted_count += 1
    return {
        'restored': restored_count,
        'deleted': deleted_count,
        'failed': failed_count,
        'progress_updated': progress_updated,
    }


def plan_remove_qa_marks(prog, selected_chapters):
    """Rows of a selection that carry a failed mark (RG 28072-28093)."""
    data = {'prog': prog}
    def _progress_entry_for_row(info):
        progress_key = info.get('progress_key')
        chapters = data['prog'].get("chapters", {})
        if progress_key and isinstance(chapters.get(progress_key), dict):
            return progress_key, chapters[progress_key]
        match = _find_progress_entry(info, data['prog'])
        if isinstance(match, tuple) and len(match) == 2:
            return match
        return None, None

    failed_chapters = []
    for ch in selected_chapters:
        if ch['status'] in ['qa_failed', 'failed']:
            failed_chapters.append(ch)
            continue
        if ch.get("is_chunk_progress"):
            continue
        # A completed parent whose chunk rows are QA failed carries
        # the mark too; selecting it must clear those chunks.
        row_key, row_entry = _progress_entry_for_row(ch)
        if progress_entry_has_qa_mark(data['prog'], row_key, row_entry):
            failed_chapters.append(ch)
    return failed_chapters


def _apply_remove_qa_failed_mark(prog, output_dir, failed_chapters):
    """Remove QA Failed Mark on ``prog`` (RG 28108-28150); returns the counts."""
    cleared_count = 0
    progress_updated = False
    for info in failed_chapters:
        if info.get("is_chunk_progress"):
            # Clears the chunk record and, once no chunk of the
            # chapter is failed any more, the chapter's own
            # qa_failed status as well.
            if clear_chunk_row_qa_mark(
                prog, info, output_dir
            ):
                cleared_count += 1
                progress_updated = True
            continue
        match = None
        progress_key = info.get('progress_key')
        if progress_key and progress_key in prog.get("chapters", {}):
            match = (progress_key, prog["chapters"][progress_key])
        else:
            match = _find_progress_entry(info, prog)

        # Normalize target output for multi-entry cleanup
        target_out = info.get('output_file')
        target_norm = _normalize_filename(target_out)
        if match:
            # Clear failed/qa_failed on ALL entries sharing this
            # output file (normalized), including the chunk ledger
            # under each of them.
            for key, entry in prog.get("chapters", {}).items():
                if not isinstance(entry, dict):
                    continue
                entry_out = entry.get('output_file')
                if not entry_out:
                    continue
                if _normalize_filename(entry_out) == target_norm:
                    if progress_entry_has_qa_mark(
                        prog, key, entry
                    ) and clear_progress_entry_qa_mark(
                        prog, key, entry, output_dir
                    ):
                        cleared_count += 1
                        progress_updated = True
        else:
            print(f"WARNING: Could not find chapter entry for {info.get('num')} ({info.get('output_file')})")
    return {'cleared': cleared_count, 'progress_updated': progress_updated}


def refinement_status_keys(prog, selected_chapters):
    """Progress keys of a selection that carry refinement state (RG 28180-28215)."""
    chapters = prog.get('chapters', {})
    matching_keys = set()
    for info in selected_chapters:
        progress_key = info.get('progress_key')
        if progress_key and isinstance(chapters.get(progress_key), dict):
            matching_keys.add(progress_key)

        target_norm = _normalize_filename(info.get('output_file'))
        if not target_norm:
            continue
        for key, entry in chapters.items():
            if not isinstance(entry, dict):
                continue
            if _normalize_filename(entry.get('output_file')) == target_norm:
                matching_keys.add(key)

    refinement_fields = {
        'refinement_status',
        'refined_at',
        'refinement_error',
        'unrefined_backup_file',
    }
    keys_with_refinement = []
    for key in matching_keys:
        entry = chapters.get(key)
        if not isinstance(entry, dict):
            continue
        previous_entry = entry.get('previous_progress_entry')
        if (
            refinement_fields.intersection(entry)
            or (
                isinstance(previous_entry, dict)
                and refinement_fields.intersection(previous_entry)
            )
        ):
            keys_with_refinement.append(key)
    return keys_with_refinement


def _apply_remove_refinement_status(prog, keys_with_refinement):
    """Remove refinement status on ``prog`` (RG 28239-28242); returns the cleared count."""
    chapters = prog.get('chapters', {})
    cleared_count = 0
    for key in keys_with_refinement:
        if _clear_refinement_progress_fields(chapters.get(key)):
            cleared_count += 1
    return cleared_count


def _apply_reset_tts(self, prog, output_dir, selected_chapters):
    """Audio-mode Reset TTS on ``prog`` and disk (RG 28413-28478); returns the counts.

    ``self`` is the Progress Manager owner (``_audio_stem_variants``).
    """
    deleted_count = 0
    status_reset_count = 0
    missing_audio_count = 0
    progress_updated = False

    def _audio_candidates(ch_entry, ch_info):
        candidates = []
        stored_tts_file = ch_entry.get('tts_file') if isinstance(ch_entry, dict) else None
        if stored_tts_file:
            candidates.append(stored_tts_file)

        output_file = (ch_entry or {}).get('output_file') or ch_info.get('output_file')
        if output_file:
            configured_ext = str(os.environ.get('TTS_AUDIO_FORMAT') or 'mp3').lower().strip().lstrip('.')
            for stem in self._audio_stem_variants(output_file):
                for ext in [configured_ext, 'mp3', 'wav']:
                    if ext:
                        candidates.append(os.path.join('text_to_speech', f"{stem}.{ext}"))

        seen = set()
        paths = []
        for candidate in candidates:
            normalized = str(candidate).replace('\\', '/')
            if normalized in seen:
                continue
            seen.add(normalized)
            full_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
            paths.append(full_path)
        return paths

    for ch_info in selected_chapters:
        match = None
        progress_key = ch_info.get('progress_key')
        if progress_key and progress_key in prog.get("chapters", {}):
            match = (progress_key, prog["chapters"][progress_key])
        else:
            match = _find_progress_entry(ch_info, prog)

        ch_entry = match[1] if match else {}
        deleted_for_chapter = False
        for audio_path in _audio_candidates(ch_entry, ch_info):
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    deleted_count += 1
                    deleted_for_chapter = True
                    print(f"Deleted TTS audio: {audio_path}")
            except Exception as e:
                print(f"Failed to delete TTS audio {audio_path}: {e}")

        if not deleted_for_chapter:
            missing_audio_count += 1

        if match:
            chapter_key, ch_entry = match
            ch_entry["tts_status"] = "no_tts"
            ch_entry.pop("tts_file", None)
            ch_entry.pop("tts_at", None)
            ch_entry.pop("tts_error", None)
            ch_entry["last_updated"] = time.time()
            progress_updated = True
            status_reset_count += 1
            print(f"Reset TTS status to no_tts for chapter {ch_info.get('num')} (key: {chapter_key})")
        else:
            print(f"WARNING: Could not find exact progress entry for {ch_info.get('output_file')}; skipped TTS status reset")

    return {
        'deleted': deleted_count,
        'status_reset': status_reset_count,
        'missing_audio': missing_audio_count,
        'progress_updated': progress_updated,
    }


def _find_audio_file_for_item(self, data, display_info):
    """Return the generated TTS file path associated with an HTML row, if one exists."""
    progress_entry = display_info.get('info', {}) or {}
    output_file = display_info.get('output_file') or progress_entry.get('output_file')
    candidates = []

    stored_tts_file = progress_entry.get('tts_file')
    if stored_tts_file:
        candidates.append(stored_tts_file)

    progress_key = display_info.get('progress_key')
    if progress_key and progress_key in data.get('prog', {}).get('chapters', {}):
        tracked_tts_file = data['prog']['chapters'][progress_key].get('tts_file')
        if tracked_tts_file:
            candidates.append(tracked_tts_file)

    if output_file:
        for _key, tracked in data.get('prog', {}).get('chapters', {}).items():
            if isinstance(tracked, dict) and tracked.get('output_file') == output_file and tracked.get('tts_file'):
                candidates.append(tracked.get('tts_file'))

        for stem in self._audio_stem_variants(output_file):
            for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(candidate).replace("\\", "/")
        if normalized in seen:
            continue
        seen.add(normalized)
        path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
        if os.path.exists(path):
            return path
    return None


def _reset_tts_progress_for_output(prog, output_file):
    """Reset TTS state of every entry sharing ``output_file`` (RG 30546-30561)."""
    if not output_file:
        return 0
    updated = 0
    now = time.time()
    for _key, tracked in prog.get('chapters', {}).items():
        if not isinstance(tracked, dict):
            continue
        if tracked.get('output_file') != output_file:
            continue
        tracked['tts_status'] = 'no_tts'
        tracked.pop('tts_file', None)
        tracked.pop('tts_at', None)
        tracked.pop('tts_error', None)
        tracked['last_updated'] = now
        updated += 1
    return updated


def _llm_token_qa_targets(prog, display_info):
    """Entries whose LLM-token QA markers a repair clears (RG 30679-30705)."""
    chapters = prog.get('chapters', {})
    target_output = (
        display_info.get('output_file')
        or (display_info.get('info', {}) or {}).get('output_file')
    )
    target_norm = _normalize_filename(target_output)
    progress_key = display_info.get('progress_key')
    targets = []
    seen_targets = set()

    def _add_target(entry):
        if isinstance(entry, dict) and id(entry) not in seen_targets:
            seen_targets.add(id(entry))
            targets.append(entry)

    if progress_key and isinstance(chapters.get(progress_key), dict):
        _add_target(chapters[progress_key])
    if target_norm:
        for entry in chapters.values():
            if (
                isinstance(entry, dict)
                and _normalize_filename(entry.get('output_file'))
                == target_norm
            ):
                _add_target(entry)
    if not targets:
        _add_target(display_info.get('info', {}))
    return targets


def _clear_llm_token_targets(targets):
    """Clear LLM-token QA markers of ``targets`` (RG 30707-30712)."""
    remaining_other_qa = False
    progress_changed = False
    for target in targets:
        changed, remaining = _clear_llm_token_qa_markers(target)
        progress_changed = progress_changed or changed
        remaining_other_qa = remaining_other_qa or remaining
    return progress_changed, remaining_other_qa


def _missing_image_qa_targets(prog, display_info, progress_entry):
    """Entries whose missing-image QA markers an insertion clears (RG 31048-31084)."""
    chapters = prog.get('chapters', {})
    target_out = (
        display_info.get('output_file')
        or progress_entry.get('output_file')
    )
    target_identity = str(target_out or '').replace(
        '\\', '/'
    ).casefold()
    progress_key = display_info.get('progress_key')
    targets = []
    seen_targets = set()

    def _add_image_qa_target(entry):
        if (
            isinstance(entry, dict)
            and id(entry) not in seen_targets
        ):
            seen_targets.add(id(entry))
            targets.append(entry)

    if (
        progress_key
        and isinstance(chapters.get(progress_key), dict)
    ):
        _add_image_qa_target(chapters[progress_key])
    if target_identity:
        for entry in chapters.values():
            if (
                isinstance(entry, dict)
                and str(
                    entry.get('output_file') or ''
                ).replace('\\', '/').casefold()
                == target_identity
            ):
                _add_image_qa_target(entry)
    if not targets:
        _add_image_qa_target(progress_entry)
    return targets


def _clear_missing_image_targets(targets):
    """Clear missing-image QA markers of ``targets`` (RG 31086-31095)."""
    progress_changed = False
    remaining_other_qa = False
    for target in targets:
        changed, remaining = (
            _clear_missing_image_qa_markers(target)
        )
        progress_changed = progress_changed or changed
        remaining_other_qa = (
            remaining_other_qa or remaining
        )
    return progress_changed, remaining_other_qa


def _partial_b_target(data, display_info):
    """The progress entry a Partial.b QA resolution targets (RG 18953-18983)."""
    chapters = data.get('prog', {}).get('chapters', {})
    # A chunk row resolves through its parent chapter: the parent carries
    # the output file and the chunk-level QA mirror Partial.b targets.
    progress_key = (
        display_info.get('progress_key')
        or display_info.get('parent_progress_key')
    )
    progress_entry = (
        chapters.get(progress_key)
        if progress_key and isinstance(chapters, dict)
        else None
    )
    output_file = display_info.get('output_file')
    if not isinstance(progress_entry, dict) and isinstance(chapters, dict):
        output_folded = os.path.basename(
            str(output_file or '')
        ).casefold()
        match = next(
            (
                (key, entry)
                for key, entry in chapters.items()
                if isinstance(entry, dict)
                and os.path.basename(
                    str(entry.get('output_file') or '')
                ).casefold() == output_folded
            ),
            None,
        )
        if match:
            progress_key, progress_entry = match

    return progress_key, progress_entry, output_file


def _partial_b_request(data, progress_key, progress_entry, output_file, source_path):
    """The single-entry Partial.b request (RG 19023-19036)."""
    return {
        'source_path': source_path,
        'progress_path': (
            os.path.abspath(str(data.get('progress_file')).strip())
            if str(data.get('progress_file') or '').strip() else ''
        ),
        'progress_key': str(progress_key or ''),
        'output_file': str(
            output_file or progress_entry.get('output_file') or ''
        ),
        'actual_num': progress_entry.get(
            'actual_num', progress_entry.get('chapter_num')
        ),
    }


# ---------------------------------------------------------------------------
# Public actions (shared by the desktop closures and the mobile Book page)
# ---------------------------------------------------------------------------


def _resolve_progress_path(data_or_path):
    if isinstance(data_or_path, dict):
        return data_or_path.get('progress_file')
    return data_or_path


def restore_in_progress(progress_file, output_dir, in_progress_chapters):
    """Restore In Progress Status on the newest progress file.

    Each selected in_progress row goes back to its previous_progress_entry, its
    previous_status, ``failed`` (unknown previous state / output exists), or is
    removed (a not-translated placeholder).  Returns the counts.
    """
    return mutate_progress(
        progress_file,
        lambda prog: _apply_restore_in_progress(prog, output_dir, in_progress_chapters),
    )


def restore_in_progress_message(result):
    """The desktop summary text of ``restore_in_progress`` (RG 28028-28035)."""
    restored_count = result.get('restored', 0)
    deleted_count = result.get('deleted', 0)
    failed_count = result.get('failed', 0)
    message_parts = []
    if restored_count:
        message_parts.append(f"restored {restored_count}")
    if deleted_count:
        message_parts.append(f"removed {deleted_count} not-translated placeholder(s)")
    if failed_count:
        message_parts.append(f"marked {failed_count} as failed")
    message = "Successfully " + ", ".join(message_parts) + "." if message_parts else "No in-progress marks were changed."
    return message


def remove_qa_marks(progress_file, output_dir, failed_chapters):
    """Remove the failed / qa_failed mark from rows (and every chunk under them)."""
    return mutate_progress(
        progress_file,
        lambda prog: _apply_remove_qa_failed_mark(prog, output_dir, failed_chapters),
    )


def remove_pending_marks(progress_file, output_dir, selected_infos):
    """Restore pending rows whose saved HTML output exists (HTML is authoritative)."""
    return _remove_pending_marks(progress_file, output_dir, selected_infos)


def remove_pending_message(result):
    """The desktop summary text of ``remove_pending_marks`` (RG 28053-28055)."""
    return (
        f"Restored {result['recovered']} pending entries. "
        f"Skipped {result['skipped']} ineligible selections. "
        "Existing QA findings were preserved."
    )


def remove_refinement_status(progress_file, keys_with_refinement):
    """Forget the refinement state of rows (the refined HTML itself is not reverted)."""
    return mutate_progress(
        progress_file,
        lambda prog: _apply_remove_refinement_status(prog, keys_with_refinement),
    )


def reset_tts(owner, progress_file, output_dir, selected_chapters):
    """Audio mode's Retranslate: delete generated TTS audio and mark rows No TTS.

    Returns the counts (``deleted``, ``status_reset``, ``missing_audio``,
    ``progress_updated``) plus ``error``: the text of a progress-file failure (the
    desktop prints it and still reports the counts, as the former in-place write's
    ``try`` did; when the newest file cannot be read nothing is deleted).
    """
    result = {'error': ''}

    def _apply(prog):
        result.update(_apply_reset_tts(owner, prog, output_dir, selected_chapters))
        return result

    try:
        mutate_progress(progress_file, _apply)
    except Exception as exc:
        result['error'] = str(exc)
    return result


def reset_tts_message(result):
    """The desktop summary text of ``reset_tts`` (RG 28490-28497)."""
    deleted_count = result.get('deleted', 0)
    status_reset_count = result.get('status_reset', 0)
    missing_audio_count = result.get('missing_audio', 0)
    success_parts = []
    if deleted_count > 0:
        success_parts.append(f"deleted {deleted_count} TTS file(s)")
    if status_reset_count > 0:
        success_parts.append(f"marked {status_reset_count} chapter(s) as No TTS")
    if missing_audio_count > 0:
        success_parts.append(f"{missing_audio_count} chapter(s) had no audio file on disk")
    message = "Successfully " + ", ".join(success_parts) + "." if success_parts else "No TTS changes made."
    return message


def find_row_audio(owner, data, display_info):
    """The generated TTS file of a row, or None."""
    return _find_audio_file_for_item(owner, data, display_info)


def delete_row_audio(owner, data, display_info, audio_path=None):
    """Delete a row's generated audio and reset TTS on every entry sharing its output.

    Returns the number of progress entries reset.  Raises OSError when the file
    cannot be deleted (the progress is then left unchanged).
    """
    if audio_path is None:
        audio_path = _find_audio_file_for_item(owner, data, display_info)
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    output_file = display_info.get('output_file')
    if not output_file:
        return 0
    return mutate_progress(
        data['progress_file'],
        lambda prog: _reset_tts_progress_for_output(prog, output_file),
    )


def resolve_llm_token_qa(progress_file, display_info, output_path):
    """Repair empty-attribute LLM-token tags in one output and clear only those markers.

    Returns ``{'repair': <_repair_empty_attribute_qa_file result>, 'progress_changed',
    'remaining_other_qa', 'error'}``; ``error`` is set when the progress file could not
    be updated after a successful repair.
    """
    result = _repair_empty_attribute_qa_file(output_path)
    outcome = {
        'repair': result,
        'progress_changed': False,
        'remaining_other_qa': False,
        'error': '',
    }
    if not result.get('resolved'):
        return outcome

    def _apply(prog):
        targets = _llm_token_qa_targets(prog, display_info)
        return _clear_llm_token_targets(targets)

    try:
        progress_changed, remaining_other_qa = mutate_progress(progress_file, _apply)
    except Exception as exc:
        outcome['error'] = str(exc)
        return outcome
    outcome['progress_changed'] = progress_changed
    outcome['remaining_other_qa'] = remaining_other_qa
    return outcome


def llm_token_repair_summary(outcome, output_path):
    """The desktop result text of a resolved LLM-token issue (RG 30751-30765)."""
    repaired_count = int(outcome['repair'].get('repaired') or 0)
    if repaired_count:
        summary = (
            f"Repaired {repaired_count} empty-attribute LLM token "
            f"tag(s) in {os.path.basename(output_path)}."
        )
    else:
        summary = (
            "No malformed empty-attribute tags remain; the stale "
            "LLM token QA mark was cleared."
        )
    if outcome['remaining_other_qa']:
        summary += "\n\nOther QA issues remain on this entry."
    else:
        summary += "\n\nThe QA issue has been resolved."
    return summary


def _default_restore_fn(translated_html, source_html, verbose=True, rename_map=None):
    from TransateKRtoEN import ContentProcessor
    return ContentProcessor.emergency_restore_images(
        translated_html, source_html, verbose=verbose, rename_map=rename_map
    )


def insert_missing_images(data, display_info, restore_fn=None):
    """Insert Missing Image: restore images from the source chapter (RG 30987-31123).

    ``restore_fn(translated, source, verbose=True, rename_map=...)`` is
    ``ContentProcessor.emergency_restore_images`` by default (it needs the translation
    engine).  Returns ``(kind, title, message, refreshed)``: the message the desktop
    shows with ``_show_message`` and whether the output / progress changed.
    """
    if restore_fn is None:
        restore_fn = _default_restore_fn
    progress_entry = display_info.get('info', {})
    try:
        import zipfile

        # Load rename map from output directory
        rename_map = None
        rename_map_path = os.path.join(data['output_dir'], 'image_rename_map.json')
        if os.path.exists(rename_map_path):
            try:
                with open(rename_map_path, 'r', encoding='utf-8') as f:
                    rename_map = json.load(f) or {}
            except Exception:
                pass

        # 1. Get Source Content from EPUB
        epub_path = data['file_path']
        original_filename = display_info.get('original_filename')
        source_html = None

        if original_filename:
            try:
                def normalize_name(n):
                    base = os.path.basename(n)
                    if base.startswith('response_'):
                        base = base[9:]
                    return os.path.splitext(base)[0].lower()

                target_base = normalize_name(original_filename)

                with zipfile.ZipFile(epub_path, 'r') as zf:
                    for fname in zf.namelist():
                        if normalize_name(fname) == target_base:
                            source_html = zf.read(fname).decode('utf-8', errors='ignore')
                            break
            except Exception as ex:
                print(f"Extraction error: {ex}")

        if not source_html:
            return ('error', "Error", "Could not extract source HTML for this chapter.", False)

        # 2. Get Translated Content
        output_file = display_info.get('output_file')
        output_path = os.path.join(data['output_dir'], output_file)

        if not os.path.exists(output_path):
            return ('error', "Error", "Output file not found.", False)
        with open(output_path, 'r', encoding='utf-8') as f:
            translated_html = f.read()

        # 3. Restore using ContentProcessor (supports all image formats + rename map)
        restored_html = restore_fn(
            translated_html, source_html, verbose=True, rename_map=rename_map
        )

        if restored_html == translated_html:
            return ('info', "Info", "No missing images could be automatically restored.", False)

        # 4. Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(restored_html)

        # 5. Clear only the resolved missing-image QA marker. Other QA failures
        # must remain intact.
        def _apply(prog):
            targets = _missing_image_qa_targets(prog, display_info, progress_entry)
            return _clear_missing_image_targets(targets)

        progress_changed, remaining_other_qa = mutate_progress(data['progress_file'], _apply)

        if progress_changed and remaining_other_qa:
            message = (
                "Images restored and the missing-image QA issue "
                "was cleared. Other QA issues remain."
            )
        elif progress_changed:
            message = (
                "Images restored and the missing-image QA issue "
                "was cleared."
            )
        else:
            message = (
                "Images restored. No stored missing-image QA "
                "marker was found to clear."
            )
        return ('info', "Success", message, True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ('error', "Error", f"Failed to restore images: {e}", False)


def build_partial_b_request(data, display_info):
    """The Partial.b request resolving one raw-foreign-text QA entry, or None.

    ``None`` when the entry no longer carries such an issue or the source is missing;
    the engine runs the request (desktop: ``_single_qa_resolution_request``).
    """
    data = data if isinstance(data, dict) else {}
    display_info = display_info if isinstance(display_info, dict) else {}
    progress_key, progress_entry, output_file = _partial_b_target(data, display_info)
    if not _progress_entry_has_raw_foreign_text_qa(progress_entry):
        return None
    source_value = str(data.get('file_path') or '').strip()
    source_path = os.path.abspath(source_value) if source_value else ''
    if not source_path or not os.path.isfile(source_path):
        return None
    return _partial_b_request(data, progress_key, progress_entry, output_file, source_path)


def special_keyword_for(owner, info):
    """The Other Settings keyword that makes a row a skipped special file, or None."""
    return owner._special_skip_keyword_for_progress_info(info)


def remove_special_keyword(owner, keyword):
    """Do not skip: drop ``keyword`` from the special-file lists (config + env + vars)."""
    return owner._remove_special_skip_keyword(keyword)


def row_actions(owner, data, info, selected_infos=None):
    """Row actions of the desktop context menu (RG 30878-30928) that apply to a row.

    Returns action ids in menu order: ``do_not_skip`` alone for a skipped special
    file, otherwise ``open_file``, ``open_audio`` / ``delete_audio``, ``copy_qa``,
    ``open_reader``, ``retranslate``, ``resolve_qa``, ``insert_missing_image``,
    ``remove_qa``, ``remove_pending``, ``remove_refinement``, ``restore_in_progress``.
    """
    info = info if isinstance(info, dict) else {}
    try:
        if owner._special_skip_keyword_for_progress_info(info):
            return ['do_not_skip']
    except Exception:
        pass
    progress_entry = info.get('info', {}) or {}
    qa_issues = progress_entry.get('qa_issues_found', [])
    if not isinstance(qa_issues, list):
        qa_issues = []
    actions = ['open_file']
    if _find_audio_file_for_item(owner, data, info):
        actions.extend(['open_audio', 'delete_audio'])
    if qa_issues:
        actions.append('copy_qa')
    if _progress_item_is_html(info):
        actions.append('open_reader')
    actions.append('retranslate')
    if (
        _progress_entry_has_raw_foreign_text_qa(progress_entry)
        or _progress_entry_has_llm_token_qa(progress_entry)
    ):
        actions.append('resolve_qa')
    if _progress_entry_has_missing_image_qa(progress_entry):
        actions.append('insert_missing_image')
    actions.append('remove_qa')
    if _pending_mark_output_path(info, data.get('output_dir')):
        actions.append('remove_pending')
    actions.append('remove_refinement')
    selected = selected_infos if selected_infos is not None else [info]
    if any((selected_info or {}).get('status') == 'in_progress' for selected_info in selected):
        actions.append('restore_in_progress')
    return actions


# ---------------------------------------------------------------------------
# Retranslate Selected: plan / apply (mobile milestone U7)
#
# Split of the ``retranslate_selected`` generator inside
# ``RetranslationMixin._add_retranslation_buttons_opf`` (frozen at 41814faa:
# selection normalisation and guards RG 21921-22093, confirmation copy
# 22095-22262, the worker half 22273-22850 and the result message 22869-22959).
# The desktop generator keeps its dialogs and its worker thread and calls
# ``plan_retranslation`` -> (dialogs) -> ``apply_retranslation`` on the worker ->
# ``retranslation_result_message`` on the Qt thread.  Every progress write is the
# former ``_merge_and_write_retranslation_progress`` (lock, re-read, three-way
# merge, authoritative chunk resets, atomic replace).
# ---------------------------------------------------------------------------


@dataclass
class RetranslationPlan:
    """What Retranslate Selected will do, decided before any confirmation.

    ``mode``: ``'retranslate'`` (ask ``confirm_title`` / ``confirm_message`` with
    Yes / No, or -- ``needs_linked_choice`` -- the three-button RECYCLED dialog
    "Delete Both Linked Files" / "Keep <counterpart_filename>" / "Cancel"),
    ``'reset_tts'`` (audio output: ask, then ``reset_tts``) or ``'refused'``
    (show ``refusal`` = ``(kind, title, message)`` and stop).
    """

    mode: str = 'retranslate'
    refusal: Optional[Tuple[str, str, str]] = None
    confirm_title: str = ''
    confirm_message: str = ''
    needs_linked_choice: bool = False
    selected_filename: str = ''
    counterpart_filename: str = ''
    selected_chapters: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    chunk_selected: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    selected_chunk_indices: Dict[str, Set[int]] = dataclass_field(default_factory=dict)
    fully_selected_chunk_keys: Set[str] = dataclass_field(default_factory=set)
    metadata_selected: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    artifact_selected: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    selected_artifact_kinds: Set[str] = dataclass_field(default_factory=set)
    recycled_artifact_pair_selected: bool = False
    reset_all_metadata: bool = False
    manual_editing: bool = False
    missing_count: int = 0
    existing_count: int = 0

    @property
    def count(self):
        return len(self.selected_chapters)

    @property
    def linked_choice_labels(self):
        """The RECYCLED dialog's buttons: (choice, label) in desktop order."""
        if not self.needs_linked_choice:
            return ()
        return (
            ("both", "Delete Both Linked Files"),
            ("selected_only", f"Keep {self.counterpart_filename}"),
            ("cancel", "Cancel"),
        )


@dataclass
class RetranslationResult:
    """Counts and the merged progress of ``apply_retranslation``."""

    selected_count: int = 0
    manual_editing: bool = False
    deleted_count: int = 0
    marked_count: int = 0
    status_reset_count: int = 0
    refinement_cleared_count: int = 0
    merged_cleared_count: int = 0
    sidecar_deleted_count: int = 0
    sidecar_cleared_count: int = 0
    sidecar_failed_count: int = 0
    machine_translation_deleted_count: int = 0
    machine_translation_failed_count: int = 0
    chunk_reset_count: int = 0
    full_chapter_chunk_reset_count: int = 0
    chunk_segment_deleted_count: int = 0
    chunk_segment_removal_failures: List[Tuple[Any, str, str, str]] = dataclass_field(default_factory=list)
    compiled_pdf_section_removed_count: int = 0
    compiled_pdf_chunk_segment_removed_count: int = 0
    compiled_pdf_deleted_count: int = 0
    progress_updated: bool = False
    merged_progress: Optional[Dict[str, Any]] = dataclass_field(default=None, repr=False)

    def as_dict(self):
        out = dict(vars(self))
        out.pop('merged_progress', None)
        return out


def _retranslation_book(book, owner=None):
    """``(owner, data)`` of a desktop data dict (+ owner) or a ``progress_core.BookProgress``."""
    if isinstance(book, dict):
        if owner is None:
            raise TypeError("a Progress Manager data dict needs its owner")
        return owner, book
    return (owner if owner is not None else book.owner), book.data


def _retranslation_rows(data, rows):
    """Selected rows as ``chapter_display_info`` dicts, in the given order.

    ``rows``: indices into ``data['chapter_display_info']`` (the desktop list rows),
    ``progress_core.RowPresentation`` objects (``.info``) or the display dicts.
    """
    infos = data.get('chapter_display_info') or []
    out = []
    for row in rows or []:
        if isinstance(row, bool):
            continue
        if isinstance(row, int):
            out.append(infos[row])
        elif isinstance(row, dict):
            out.append(row)
        elif isinstance(getattr(row, 'info', None), dict):
            out.append(row.info)
    return out


def plan_retranslation(book, rows, settings=None, *, owner=None):
    """Plan Retranslate Selected for ``rows`` without touching disk.

    ``book``: the desktop Progress Manager data dict (pass ``owner``) or a
    ``progress_core.BookProgress``.  ``settings``: ``{'manual_editing': bool}`` (the
    Progress Manager's "Manual editing" toggle).  Returns a ``RetranslationPlan``
    carrying the guard refusals and the confirmation copy exactly as the desktop
    shows them.
    """
    self, data = _retranslation_book(book, owner)
    settings = settings if isinstance(settings, dict) else {}

    def _manual_editing_enabled():
        return bool(settings.get('manual_editing'))

    plan = RetranslationPlan()
    selected_chapters = _retranslation_rows(data, rows)
    if not selected_chapters:
        plan.mode = 'refused'
        plan.refusal = ('warning', "No Selection", "Please select at least one chapter.")
        return plan
    selected_parent_keys = {
        chapter.get("progress_key")
        for chapter in selected_chapters
        if not chapter.get("is_chunk_progress")
        and chapter.get("progress_key")
    }
    # A selected parent requests a full chapter reset. Do not also
    # process its child chunk rows, which would otherwise edit the
    # same output file twice and inflate the confirmation counts.
    selected_chapters = [
        chapter
        for chapter in selected_chapters
        if not chapter.get("is_chunk_progress")
        or chapter.get("parent_progress_key") not in selected_parent_keys
    ]
    chunk_selected = [
        chapter
        for chapter in selected_chapters
        if chapter.get("is_chunk_progress")
    ]
    selected_chunk_indices = {}
    for chapter in chunk_selected:
        chunk_key = str(chapter.get("chunk_progress_key") or "")
        try:
            chunk_index = int(chapter.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if chunk_key and chunk_index > 0:
            selected_chunk_indices.setdefault(chunk_key, set()).add(
                chunk_index
            )
    fully_selected_chunk_keys = set()
    for chunk_key, indices in selected_chunk_indices.items():
        chunk_entry = data['prog'].get("chapter_chunks", {}).get(
            chunk_key
        )
        try:
            total_chunks = int((chunk_entry or {}).get("total") or 0)
        except (TypeError, ValueError):
            total_chunks = 0
        if (
            total_chunks > 1
            and indices == set(range(1, total_chunks + 1))
        ):
            fully_selected_chunk_keys.add(chunk_key)

    metadata_selected = [ch for ch in selected_chapters if self._is_metadata_progress_info(ch)]
    artifact_selected = [
        ch for ch in selected_chapters
        if self._is_translation_artifact_progress_info(ch)
    ]
    recycled_artifact_pair_selected = bool(
        artifact_selected
        and translation_artifacts_are_recycled_linked(data.get('prog'))
    )
    selected_artifact_kinds = set()
    for artifact_info in artifact_selected:
        raw_kind = (
            artifact_info.get('special_type')
            or (artifact_info.get('info') or {}).get('special_type')
        )
        artifact_spec = (
            translation_artifact_spec_for_kind(raw_kind)
            or translation_artifact_spec_for_filename(
                artifact_info.get('output_file')
            )
        )
        if artifact_spec:
            selected_artifact_kinds.add(artifact_spec['kind'])
    translated_file_selected = metadata_selected + artifact_selected
    tracked_metadata_keys = {
        key for key, entry in data.get('prog', {}).get('chapters', {}).items()
        if isinstance(entry, dict) and is_metadata_progress_entry(key, entry)
    }
    selected_metadata_keys = {
        ch.get('progress_key') for ch in metadata_selected if ch.get('progress_key')
    }
    reset_all_metadata = bool(
        metadata_selected
        and tracked_metadata_keys
        and tracked_metadata_keys.issubset(selected_metadata_keys)
    )
    plan.selected_chapters = selected_chapters
    plan.chunk_selected = chunk_selected
    plan.selected_chunk_indices = selected_chunk_indices
    plan.fully_selected_chunk_keys = fully_selected_chunk_keys
    plan.metadata_selected = metadata_selected
    plan.artifact_selected = artifact_selected
    plan.selected_artifact_kinds = selected_artifact_kinds
    plan.recycled_artifact_pair_selected = recycled_artifact_pair_selected
    plan.reset_all_metadata = reset_all_metadata
    if metadata_selected and not self._metadata_progress_tracking_enabled(data.get('file_path')):
        plan.mode = 'refused'
        plan.refusal = (
            'info',
            "Metadata Translation Disabled",
            "Enable 'Translate Book Title / Metadata' before requesting metadata regeneration.",
        )
        return plan
    disabled_artifacts = [
        ch for ch in artifact_selected
        if not self._translation_artifact_progress_tracking_enabled(
            ch.get('special_type')
            or (ch.get('info') or {}).get('special_type'),
            data.get('file_path'),
        )
    ]
    if disabled_artifacts:
        disabled_labels = [
            ch.get('translation_artifact_label')
            or (ch.get('info') or {}).get(
                'translation_artifact_label'
            )
            or ch.get('output_file')
            for ch in disabled_artifacts
        ]
        plan.mode = 'refused'
        plan.refusal = (
            'info',
            "Translation Feature Disabled",
            "Enable the matching TOC/header translation toggle before "
            "requesting regeneration for: "
            + ", ".join(disabled_labels),
        )
        return plan
    if (
        self._current_progress_output_mode(data) == 'audio'
        and translated_file_selected
        and len(translated_file_selected) != len(selected_chapters)
    ):
        plan.mode = 'refused'
        plan.refusal = (
            'warning',
            "Mixed Selection",
            "Select metadata.json, TOC.txt, and translated_headers.txt "
            "separately from chapter rows when resetting Audio output.",
        )
        return plan

    if (
        self._current_progress_output_mode(data) == 'audio'
        and not translated_file_selected
    ):
        count = len(selected_chapters)
        plan.mode = 'reset_tts'
        plan.confirm_title = "Confirm TTS Reset"
        plan.confirm_message = f"This will delete only generated TTS audio for {count} selected chapter(s), mark them as No TTS, and leave translated HTML files untouched.\n\nContinue?"
        return plan

    # Count different types
    missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
    existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
    manual_editing_retranslation = _manual_editing_enabled()

    count = len(selected_chapters)
    if metadata_selected and len(metadata_selected) == count:
        if reset_all_metadata:
            confirm_msg = (
                "This will delete metadata.json, mark all selected metadata phases pending, "
                "and regenerate them on the next translation run.\n\nContinue?"
            )
        else:
            selected_labels = [
                ch.get('metadata_label')
                or (ch.get('info') or {}).get('metadata_label')
                or 'Metadata'
                for ch in metadata_selected
            ]
            confirm_msg = (
                "This will keep metadata.json, mark only the selected metadata phase(s) "
                "pending, and regenerate them on the next translation run:\n\n"
                + ", ".join(selected_labels)
                + "\n\nContinue?"
            )
    elif chunk_selected and len(chunk_selected) == count:
        chunk_labels = [
            f"{'Section' if chapter.get('pdf_toc_section') else 'Ch.'}"
            f"{chapter.get('num')} chunk "
            f"{chapter.get('chunk_index')}/{chapter.get('total_chunks')}"
            for chapter in chunk_selected
        ]
        if fully_selected_chunk_keys:
            complete_count = len(fully_selected_chunk_keys)
            partial_count = sum(
                1
                for chapter in chunk_selected
                if str(chapter.get("chunk_progress_key") or "")
                not in fully_selected_chunk_keys
            )
            action_text = (
                f"delete the translated HTML file for {complete_count} "
                "chapter/section(s) because every chunk is selected"
            )
            if partial_count:
                action_text += (
                    f", remove {partial_count} individually selected "
                    "chunk segment(s) from other HTML files"
                )
            confirm_msg = (
                "This will " + action_text + ", reset the selected chunk "
                "progress, and preserve unselected sibling chunks:\n\n"
                + ", ".join(chunk_labels[:20])
                + (f" (+{len(chunk_labels) - 20} more)" if len(chunk_labels) > 20 else "")
                + "\n\nContinue?"
            )
        else:
            confirm_msg = (
                "This will remove only the selected translated chunk "
                "segment(s) from their HTML files, preserve every other "
                "cached chunk, and mark the selected chunks pending:\n\n"
                + ", ".join(chunk_labels[:20])
                + (f" (+{len(chunk_labels) - 20} more)" if len(chunk_labels) > 20 else "")
                + "\n\nContinue?"
            )
    elif count > 10:
        if missing_count > 0 and existing_count > 0:
            existing_action = (
                f"Delete {existing_count} existing chapter output file(s), retain their "
                "SDLXLIFF sidecars, and clear the translated targets for manual editing"
                if manual_editing_retranslation
                else f"Delete and retranslate {existing_count} existing chapters and their SDLXLIFF sidecars"
            )
            confirm_msg = f"This will:\n• Mark {missing_count} missing chapters for translation\n• {existing_action}\n\nTotal: {count} chapters\n\nContinue?"
        elif missing_count > 0:
            confirm_msg = f"This will mark {missing_count} missing chapters for translation.\n\nContinue?"
        else:
            if manual_editing_retranslation:
                confirm_msg = (
                    f"This will delete {existing_count} translated chapter output file(s), "
                    "retain their SDLXLIFF sidecars, clear the translated targets, and mark "
                    "them pending for manual editing.\n\nContinue?"
                )
            else:
                confirm_msg = f"This will delete {existing_count} translated chapters and their SDLXLIFF sidecars, then mark them for retranslation.\n\nContinue?"
    else:
        chapters = [
            (
                ch.get('metadata_label')
                or (ch.get('info') or {}).get('metadata_label')
                or "Metadata"
            ) if self._is_metadata_progress_info(ch) else (
                ch.get('translation_artifact_label')
                or (ch.get('info') or {}).get(
                    'translation_artifact_label'
                )
                or ch.get('output_file')
            ) if self._is_translation_artifact_progress_info(ch) else f"Ch.{ch['num']}"
            for ch in selected_chapters
        ]
        confirm_msg = f"This will process:\n\n{', '.join(chapters)}\n\n"
        if missing_count > 0:
            confirm_msg += f"• {missing_count} missing chapters will be marked for translation\n"
        if existing_count > 0:
            if manual_editing_retranslation:
                confirm_msg += (
                    f"• {existing_count} existing chapter output file(s) will be deleted; "
                    "their SDLXLIFF sidecars will be retained with translated targets "
                    "cleared for manual editing\n"
                )
            else:
                confirm_msg += f"• {existing_count} existing chapters and SDLXLIFF sidecars will be deleted and retranslated\n"
        confirm_msg += "\nContinue?"

    plan.missing_count = missing_count
    plan.existing_count = existing_count
    plan.manual_editing = manual_editing_retranslation
    if (
        recycled_artifact_pair_selected
        and len(selected_artifact_kinds) == 1
    ):
        selected_kind = next(iter(selected_artifact_kinds))
        selected_spec = translation_artifact_spec_for_kind(selected_kind)
        counterpart_kind = (
            "headers" if selected_kind == "toc" else "toc"
        )
        counterpart_spec = translation_artifact_spec_for_kind(
            counterpart_kind
        )
        selected_filename = selected_spec['filename']
        counterpart_filename = counterpart_spec['filename']
        confirm_msg = confirm_msg.rstrip()
        if confirm_msg.endswith("Continue?"):
            confirm_msg = confirm_msg[:-len("Continue?")].rstrip()
        confirm_msg += (
            f"\n\n{selected_filename} and {counterpart_filename} are linked "
            "because one was RECYCLED from the other. Keeping "
            f"{counterpart_filename} will delete and reset only the selected "
            f"{selected_filename}, but it may be rebuilt by reusing "
            f"{counterpart_filename} without a new API translation.\n\n"
            "Delete both linked files to force both translations to be "
            "generated again, or keep the unselected counterpart?"
        )
        plan.needs_linked_choice = True
        plan.selected_filename = selected_filename
        plan.counterpart_filename = counterpart_filename
        plan.confirm_title = "Confirm Linked Retranslation"
    else:
        if recycled_artifact_pair_selected:
            confirm_msg = confirm_msg.rstrip()
            if confirm_msg.endswith("Continue?"):
                confirm_msg = confirm_msg[:-len("Continue?")].rstrip()
            confirm_msg += (
                "\n\nTOC.txt and translated_headers.txt are RECYCLED-linked. "
                "Both linked files are already selected, so both will be "
                "deleted and reset.\n\nContinue?"
            )
        plan.confirm_title = "Confirm Retranslation"
    plan.confirm_message = confirm_msg
    return plan


def apply_retranslation(book, plan, linked_choice=None, sidecar_workers=None, *, owner=None):
    """Run a confirmed ``RetranslationPlan`` (blocking; the desktop runs it on a worker).

    ``linked_choice``: the RECYCLED dialog's answer (``'both'`` deletes and resets TOC.txt
    and translated_headers.txt together; anything else keeps the counterpart).
    ``sidecar_workers``: SDLXLIFF sidecar reset/delete threads (default: the owner's
    extraction workers when parallel extraction is on, else 1).

    Deletes the selected outputs (whole files, or only the selected chunk segments),
    invalidates compiled PDF output for PDF sections, resets metadata / subtitle /
    artifact / chapter rows to pending (clearing refinement state and the chapter's
    cached chunks, dropping its merged children), resets or deletes the SDLXLIFF
    sidecars and deletes Machine Translation previews, then merge-writes the progress
    (``_merge_and_write_retranslation_progress``).  Returns a ``RetranslationResult``;
    ``retranslation_result_message`` gives the desktop's summary.
    """
    self, data = _retranslation_book(book, owner)
    if plan is None or plan.mode != 'retranslate':
        raise ValueError("apply_retranslation needs a confirmed 'retranslate' plan")
    selected_chapters = plan.selected_chapters
    selected_chunk_indices = plan.selected_chunk_indices
    fully_selected_chunk_keys = plan.fully_selected_chunk_keys
    reset_all_metadata = plan.reset_all_metadata
    manual_editing_retranslation = plan.manual_editing
    delete_both_linked_artifacts = bool(plan.needs_linked_choice and linked_choice == "both")
    sidecar_workers_override = sidecar_workers

    def _sdlxliff_sidecar_path_for_output_file(output_file):
        if not output_file:
            return None
        output_name = os.path.basename(str(output_file).replace("\\", "/"))
        if not output_name:
            return None
        return os.path.join(data['output_dir'], "SDLXLIFF", f"{output_name}.sdlxliff")

    def _machine_translation_path_for_output_file(output_file):
        return _sdlxliff_machine_translation_path(data['output_dir'], output_file)

    # Capture this window's starting snapshot.  The final save merges
    # only our changes into the newest on-disk progress so concurrent
    # Retranslate Selected actions do not clobber one another.
    progress_baseline = copy.deepcopy(data['prog'])
    working_progress = copy.deepcopy(progress_baseline)
    merged_progress = None

    # Process chapters - DELETE FILES AND UPDATE PROGRESS
    deleted_count = 0
    marked_count = 0
    status_reset_count = 0
    refinement_cleared_count = 0
    merged_cleared_count = 0
    sidecar_deleted_count = 0
    sidecar_cleared_count = 0
    sidecar_failed_count = 0
    sidecar_paths_to_update = {}
    machine_translation_deleted_count = 0
    machine_translation_failed_count = 0
    chunk_reset_count = 0
    authoritative_chunk_resets = {}
    full_chapter_chunk_reset_count = 0
    chunk_segment_deleted_count = 0
    chunk_segment_removal_failures = []
    processed_full_chunk_keys = set()
    compiled_pdf_section_removed_count = 0
    compiled_pdf_chunk_segment_removed_count = 0
    compiled_pdf_deleted_count = 0
    progress_updated = False
    metadata_file_deleted = False

    if delete_both_linked_artifacts:
        for artifact_kind in ("toc", "headers"):
            artifact_path = translation_artifact_path(
                data['output_dir'], artifact_kind, existing_only=True
            )
            if not artifact_path:
                continue
            try:
                os.remove(artifact_path)
                deleted_count += 1
                print(
                    "Deleted linked RECYCLED translation artifact: "
                    f"{artifact_path}"
                )
            except Exception as e:
                print(
                    "Failed to delete linked RECYCLED translation artifact "
                    f"{artifact_path}: {e}"
                )
        linked_reset_count = reset_translation_artifact_progress_entries(
            working_progress, ("toc", "headers")
        )
        if linked_reset_count:
            status_reset_count += linked_reset_count
            progress_updated = True

    merged_children_by_parent = {}
    for child_key, child_entry in working_progress.get(
        "chapters", {}
    ).items():
        if (
            isinstance(child_entry, dict)
            and child_entry.get("status") == "merged"
        ):
            merged_children_by_parent.setdefault(
                child_entry.get("merged_parent_chapter"), []
            ).append(child_key)

    for ch_info in selected_chapters:
        output_file = ch_info['output_file']
        actual_num = ch_info['num']
        progress_key = ch_info.get('progress_key')

        chapter_entry = (
            working_progress.get("chapters", {}).get(progress_key)
            if progress_key
            else None
        )
        if not isinstance(chapter_entry, dict):
            chapter_entry = ch_info.get("info") or {}
        if (
            not ch_info.get("is_chunk_progress")
            and isinstance(chapter_entry, dict)
            and chapter_entry.get("pdf_toc_section")
        ):
            try:
                from pdf_workspace_compiler import (
                    invalidate_compiled_pdf_source_output,
                )

                invalidated = invalidate_compiled_pdf_source_output(
                    data['output_dir'],
                    output_file,
                )
                compiled_pdf_section_removed_count += int(
                    invalidated.get("sections_removed") or 0
                )
                compiled_pdf_deleted_count += int(
                    invalidated.get("pdf_files_deleted") or 0
                )
            except Exception as exc:
                print(
                    "Failed to invalidate compiled PDF output for "
                    f"{output_file}: {exc}"
                )

        if ch_info.get("is_chunk_progress"):
            chunk_key = str(ch_info.get("chunk_progress_key") or "")
            parent_key = ch_info.get("parent_progress_key")
            chunk_index = ch_info.get("chunk_index")
            delete_complete_chunk_file = (
                chunk_key in fully_selected_chunk_keys
            )
            if (
                delete_complete_chunk_file
                and chunk_key in processed_full_chunk_keys
            ):
                continue
            chunk_indices = (
                sorted(selected_chunk_indices.get(chunk_key, set()))
                if delete_complete_chunk_file
                else [chunk_index]
            )
            if delete_complete_chunk_file:
                processed_full_chunk_keys.add(chunk_key)
            chunk_entry = working_progress.get("chapter_chunks", {}).get(
                chunk_key
            )
            parent_entry = working_progress.get("chapters", {}).get(
                parent_key
            )
            if not isinstance(chunk_entry, dict) or not isinstance(
                parent_entry, dict
            ):
                print(
                    f"WARNING: Missing chunk/parent progress entry for "
                    f"chapter {actual_num}, chunk {chunk_index}"
                )
                continue

            resolved_output_file, resolved_output_path = (
                self._resolve_existing_output_path(
                    data['output_dir'],
                    output_file,
                    {
                        "info": parent_entry,
                        "progress_entry": parent_entry,
                        "progress_key": parent_key,
                        "num": actual_num,
                        "output_file": output_file,
                        "original_filename": ch_info.get(
                            "original_filename", ""
                        ),
                    },
                    working_progress,
                )
            )
            if resolved_output_path:
                output_file = resolved_output_file or output_file
                output_path = resolved_output_path
            else:
                output_path = (
                    output_file
                    if os.path.isabs(str(output_file or ""))
                    else os.path.join(
                        data['output_dir'], str(output_file or "")
                    )
                )

            output_exists = bool(
                output_file and os.path.isfile(output_path)
            )
            response_segment_removed = not output_exists
            if output_exists:
                try:
                    removed, file_deleted = (
                        remove_chunk_segments_from_file(
                            output_path,
                            chunk_key,
                            chunk_indices,
                            chunk_entry,
                        )
                    )
                    if removed:
                        if file_deleted:
                            deleted_count += 1
                        else:
                            chunk_segment_deleted_count += len(removed)
                        response_segment_removed = True
                    else:
                        chunk_segment_removal_failures.append(
                            (
                                actual_num,
                                ",".join(str(index) for index in chunk_indices),
                                output_path,
                                "chunk boundary/result was not found",
                            )
                        )
                except Exception as exc:
                    chunk_segment_removal_failures.append(
                        (
                            actual_num,
                            ",".join(str(index) for index in chunk_indices),
                            output_path,
                            str(exc),
                        )
                    )
                    print(
                        f"Failed to remove chapter {actual_num} chunk(s) "
                        f"{chunk_indices} from {output_path}: {exc}"
                    )

            if not response_segment_removed:
                print(
                    "ERROR: Chunk progress was not reset because its "
                    f"HTML segment remained in {output_path}"
                )
                continue

            if parent_entry.get("pdf_toc_section"):
                try:
                    if delete_complete_chunk_file:
                        from pdf_workspace_compiler import (
                            invalidate_compiled_pdf_source_output,
                        )
                        invalidated = invalidate_compiled_pdf_source_output(
                            data['output_dir'],
                            output_file,
                        )
                        compiled_pdf_section_removed_count += int(
                            invalidated.get("sections_removed") or 0
                        )
                    else:
                        from pdf_workspace_compiler import (
                            invalidate_compiled_pdf_api_chunks,
                        )
                        invalidated = invalidate_compiled_pdf_api_chunks(
                            data['output_dir'],
                            chunk_key,
                            chunk_indices,
                            entry=chunk_entry,
                        )
                        compiled_pdf_chunk_segment_removed_count += int(
                            invalidated.get("chunk_segments_removed") or 0
                        )
                    compiled_pdf_deleted_count += int(
                        invalidated.get("pdf_files_deleted") or 0
                    )
                except Exception as exc:
                    print(
                        "Failed to remove PDF API chunk from compiled HTML: "
                        f"{exc}"
                    )

            reset = reset_chunks_for_retranslation(
                chunk_entry,
                chunk_indices,
            )
            if reset:
                reset_key = (chunk_key, parent_key)
                authoritative_chunk_resets.setdefault(
                    reset_key, set()
                ).update(int(index) for index in chunk_indices)
                parent_entry["status"] = "pending"
                parent_entry["failure_reason"] = ""
                parent_entry["error_message"] = ""
                parent_entry["last_updated"] = time.time()
                _clear_refinement_progress_fields(parent_entry)
                _sync_parent_chunk_qa_summary(
                    working_progress, parent_key, chunk_key
                )
                chunk_reset_count += len(reset)
                status_reset_count += 1
                progress_updated = True
            continue

        if (
            delete_both_linked_artifacts
            and self._is_translation_artifact_progress_info(ch_info)
        ):
            continue

        if self._is_metadata_progress_info(ch_info):
            metadata_path = os.path.join(data['output_dir'], 'metadata.json')
            if reset_all_metadata and not metadata_file_deleted:
                try:
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                        deleted_count += 1
                        print(f"Deleted metadata for regeneration: {metadata_path}")
                    metadata_file_deleted = True
                except Exception as e:
                    print(f"Failed to delete metadata.json: {e}")

            entry_key = progress_key or METADATA_PROGRESS_KEY
            entry = working_progress.setdefault('chapters', {}).setdefault(
                entry_key,
                {
                    'actual_num': -1,
                    'content_hash': '',
                    'output_file': 'metadata.json',
                    'original_basename': 'metadata.json',
                    'is_special': True,
                    'special_type': 'metadata',
                    'metadata_progress_key': entry_key,
                    'metadata_fields': list(ch_info.get('metadata_fields') or []),
                    'metadata_label': ch_info.get('metadata_label', 'Metadata'),
                },
            )
            entry['status'] = 'pending'
            entry['metadata_regeneration_requested'] = True
            entry['failure_reason'] = ''
            entry['error_message'] = ''
            entry['last_updated'] = time.time()
            progress_updated = True
            status_reset_count += 1
            print(f"Reset metadata translation phase to pending: {entry_key}")
            continue

        if ch_info.get('is_subtitle'):
            if ch_info['status'] == 'not_translated':
                marked_count += 1
                continue

            subtitle_matches = []
            seen_subtitle_keys = set()
            for subtitle_key in ch_info.get('progress_keys', []):
                subtitle_entry = working_progress.get('chapters', {}).get(subtitle_key)
                if (
                    subtitle_key not in seen_subtitle_keys
                    and isinstance(subtitle_entry, dict)
                ):
                    seen_subtitle_keys.add(subtitle_key)
                    subtitle_matches.append((subtitle_key, subtitle_entry))
            for subtitle_key, subtitle_entry in ch_info.get('entries', []):
                if (
                    subtitle_key not in seen_subtitle_keys
                    and isinstance(subtitle_entry, dict)
                ):
                    seen_subtitle_keys.add(subtitle_key)
                    subtitle_matches.append((subtitle_key, subtitle_entry))

            if not subtitle_matches:
                print(
                    f"WARNING: Could not find subtitle progress entries for "
                    f"{output_file}; skipped deletion and status reset"
                )
                continue

            output_path = (
                output_file
                if os.path.isabs(output_file)
                else os.path.join(data['output_dir'], output_file)
            )
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    deleted_count += 1
                    print(f"Deleted subtitle: {output_path}")
            except Exception as e:
                print(f"Failed to delete subtitle {output_path}: {e}")

            for subtitle_key, subtitle_entry in subtitle_matches:
                subtitle_entry['status'] = 'pending'
                subtitle_entry['failure_reason'] = ''
                subtitle_entry['error_message'] = ''
                if _clear_refinement_progress_fields(subtitle_entry):
                    refinement_cleared_count += 1
                status_reset_count += 1
                print(
                    f"Reset subtitle batch to pending "
                    f"(file index: {actual_num}, key: {subtitle_key})"
                )
            working_progress.pop('subtitle_files', None)
            progress_updated = True
            continue

        if ch_info['status'] != 'not_translated':
            # Reset status to pending for ALL non-not_translated chapters, but only if we can match the exact progress entry
            match = None
            if progress_key and progress_key in working_progress["chapters"]:
                match = (progress_key, working_progress["chapters"][progress_key])
            else:
                match = _find_progress_entry(ch_info, working_progress)
            if match:
                chapter_key, ch_entry = match
                target_output_file = ch_entry.get('output_file') or ch_info['output_file']
                # Delete existing file only after we know which entry to update
                if output_file:
                    output_path = os.path.join(data['output_dir'], output_file)
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            deleted_count += 1
                    except Exception as e:
                        print(f"Failed to delete {output_path}: {e}")

                sidecar_paths = []
                seen_sidecars = set()
                machine_translation_paths = []
                seen_machine_translation = set()
                for candidate_output in (output_file, target_output_file):
                    sidecar_path = _sdlxliff_sidecar_path_for_output_file(candidate_output)
                    if not sidecar_path:
                        continue
                    sidecar_key = os.path.normcase(os.path.abspath(sidecar_path))
                    if sidecar_key in seen_sidecars:
                        continue
                    seen_sidecars.add(sidecar_key)
                    sidecar_paths.append(sidecar_path)
                    machine_translation_path = _machine_translation_path_for_output_file(candidate_output)
                    if machine_translation_path:
                        machine_translation_key = os.path.normcase(os.path.abspath(machine_translation_path))
                        if machine_translation_key not in seen_machine_translation:
                            seen_machine_translation.add(machine_translation_key)
                            machine_translation_paths.append(machine_translation_path)
                for sidecar_path in sidecar_paths:
                    sidecar_key = os.path.normcase(
                        os.path.abspath(sidecar_path)
                    )
                    sidecar_paths_to_update.setdefault(
                        sidecar_key,
                        sidecar_path,
                    )
                for machine_translation_path in machine_translation_paths:
                    try:
                        if os.path.exists(machine_translation_path):
                            os.remove(machine_translation_path)
                            machine_translation_deleted_count += 1
                    except Exception as e:
                        machine_translation_failed_count += 1
                        print(f"Failed to delete Machine Translation preview {machine_translation_path}: {e}")

                ch_entry["status"] = "pending"
                ch_entry["failure_reason"] = ""
                ch_entry["error_message"] = ""
                if manual_editing_retranslation:
                    ch_entry["manual_editing_pending"] = True
                else:
                    ch_entry.pop("manual_editing_pending", None)
                if self._is_translation_artifact_progress_info(ch_info):
                    ch_entry["content_hash"] = ""
                    ch_entry.pop("model_name", None)
                    ch_entry.pop("model", None)
                if _clear_refinement_progress_fields(ch_entry):
                    refinement_cleared_count += 1
                # A parent-row retranslation is a full chapter reset.
                # Clear every cached child result so it cannot be
                # silently rebuilt from the chunk resume cache.
                parent_chunk_key = str(
                    ch_entry.get("content_hash") or chapter_key
                )
                parent_chunk_entry = working_progress.get(
                    "chapter_chunks", {}
                ).get(parent_chunk_key)
                if isinstance(parent_chunk_entry, dict):
                    ensure_chunk_entry_schema(parent_chunk_entry)
                    reset = reset_chunks_for_retranslation(
                        parent_chunk_entry,
                        list(parent_chunk_entry.get("chunks", {})),
                    )
                    if reset:
                        full_chapter_chunk_reset_count += 1
                        reset_key = (parent_chunk_key, chapter_key)
                        authoritative_chunk_resets.setdefault(
                            reset_key, set()
                        ).update(int(index) for index in reset)
                    _sync_parent_chunk_qa_summary(
                        working_progress, chapter_key, parent_chunk_key
                    )
                progress_updated = True
                status_reset_count += 1
            else:
                print(f"WARNING: Could not find exact progress entry for {output_file}; skipped deletion and status reset")

            # MERGED CHILDREN FIX: Clear any merged children of this chapter
            # ONLY clear children that still have "merged" status
            # If split-the-merge succeeded, children will have their own status (completed/qa_failed)
            # and should NOT be deleted when parent is retranslated
            for child_key in merged_children_by_parent.pop(actual_num, []):
                child_data = working_progress["chapters"].get(child_key)
                if isinstance(child_data, dict) and child_data.get("status") == "merged":
                    child_actual_num = child_data.get("actual_num")
                    print(f"🔓 Clearing merged status for child chapter {child_actual_num} (parent {actual_num} being retranslated)")
                    del working_progress["chapters"][child_key]
                    merged_cleared_count += 1
                    progress_updated = True
        else:
            # Just marking for translation (no file to delete)
            marked_count += 1

    if sidecar_paths_to_update:
        retranslation_config = getattr(self, "config", {})
        if not isinstance(retranslation_config, dict):
            retranslation_config = {}
        parallel_enabled = bool(
            getattr(
                self,
                "enable_parallel_extraction_var",
                retranslation_config.get(
                    "enable_parallel_extraction",
                    True,
                ),
            )
        )
        raw_sidecar_workers = getattr(
            self,
            "extraction_workers_var",
            retranslation_config.get(
                "extraction_workers",
                os.environ.get("EXTRACTION_WORKERS", "1"),
            ),
        )
        sidecar_workers = raw_sidecar_workers if parallel_enabled else 1
        if sidecar_workers_override is not None:
            sidecar_workers = sidecar_workers_override
        sidecar_result = _bulk_retranslation_sidecar_updates(
            sidecar_paths_to_update.values(),
            manual_editing=manual_editing_retranslation,
            max_workers=sidecar_workers,
        )
        sidecar_cleared_count += int(sidecar_result["cleared"])
        sidecar_deleted_count += int(sidecar_result["deleted"])
        sidecar_failures = list(sidecar_result["failed"])
        sidecar_failed_count += len(sidecar_failures)
        failed_action = (
            "clear translated target in"
            if manual_editing_retranslation
            else "delete"
        )
        for failed_path, error in sidecar_failures[:10]:
            print(
                f"Failed to {failed_action} SDLXLIFF sidecar "
                f"{failed_path}: {error}"
            )
        if len(sidecar_failures) > 10:
            print(
                "Additional SDLXLIFF sidecar failures omitted: "
                f"{len(sidecar_failures) - 10}"
            )

    cleanup_summary = []
    if deleted_count:
        cleanup_summary.append(
            f"deleted {deleted_count} translated output file(s)"
        )
    if sidecar_cleared_count:
        cleanup_summary.append(
            f"reset {sidecar_cleared_count} retained SDLXLIFF sidecar(s)"
        )
    if sidecar_deleted_count:
        cleanup_summary.append(
            f"deleted {sidecar_deleted_count} SDLXLIFF sidecar(s)"
        )
    if machine_translation_deleted_count:
        cleanup_summary.append(
            "deleted "
            f"{machine_translation_deleted_count} Machine Translation preview(s)"
        )
    if cleanup_summary:
        print("Bulk retranslation cleanup: " + ", ".join(cleanup_summary))

    # Save the updated progress if we made changes
    if progress_updated:
        try:
            merged_progress = _merge_and_write_retranslation_progress(
                data['progress_file'],
                progress_baseline,
                working_progress,
                authoritative_chunk_resets=[
                    {
                        "chunk_key": chunk_key,
                        "parent_key": parent_key,
                        "indices": sorted(indices),
                    }
                    for (chunk_key, parent_key), indices
                    in authoritative_chunk_resets.items()
                ],
            )
            print(f"Updated progress tracking file - reset {status_reset_count} chapter statuses to pending")
        except Exception as e:
            print(f"Failed to update progress file: {e}")

    return RetranslationResult(
        selected_count=len(selected_chapters),
        manual_editing=manual_editing_retranslation,
        deleted_count=deleted_count,
        marked_count=marked_count,
        status_reset_count=status_reset_count,
        refinement_cleared_count=refinement_cleared_count,
        merged_cleared_count=merged_cleared_count,
        sidecar_deleted_count=sidecar_deleted_count,
        sidecar_cleared_count=sidecar_cleared_count,
        sidecar_failed_count=sidecar_failed_count,
        machine_translation_deleted_count=machine_translation_deleted_count,
        machine_translation_failed_count=machine_translation_failed_count,
        chunk_reset_count=chunk_reset_count,
        full_chapter_chunk_reset_count=full_chapter_chunk_reset_count,
        chunk_segment_deleted_count=chunk_segment_deleted_count,
        chunk_segment_removal_failures=list(chunk_segment_removal_failures),
        compiled_pdf_section_removed_count=compiled_pdf_section_removed_count,
        compiled_pdf_chunk_segment_removed_count=compiled_pdf_chunk_segment_removed_count,
        compiled_pdf_deleted_count=compiled_pdf_deleted_count,
        progress_updated=progress_updated,
        merged_progress=merged_progress,
    )


def retranslation_result_message(result):
    """The desktop result dialog of Retranslate Selected (RG 22869-22959).

    Returns ``(kind, title, message)``: ``('warning', "Chunk HTML Not Updated", ...)``
    when a chunk segment could not be removed, ``('info', "Success", ...)`` or
    ``('info', "Info", "No changes made.")``.
    """
    deleted_count = result.deleted_count
    sidecar_deleted_count = result.sidecar_deleted_count
    sidecar_cleared_count = result.sidecar_cleared_count
    machine_translation_deleted_count = result.machine_translation_deleted_count
    chunk_segment_deleted_count = result.chunk_segment_deleted_count
    compiled_pdf_section_removed_count = result.compiled_pdf_section_removed_count
    compiled_pdf_chunk_segment_removed_count = result.compiled_pdf_chunk_segment_removed_count
    compiled_pdf_deleted_count = result.compiled_pdf_deleted_count
    chunk_reset_count = result.chunk_reset_count
    full_chapter_chunk_reset_count = result.full_chapter_chunk_reset_count
    marked_count = result.marked_count
    status_reset_count = result.status_reset_count
    refinement_cleared_count = result.refinement_cleared_count
    merged_cleared_count = result.merged_cleared_count
    sidecar_failed_count = result.sidecar_failed_count
    machine_translation_failed_count = result.machine_translation_failed_count
    manual_editing_retranslation = result.manual_editing
    chunk_segment_removal_failures = result.chunk_segment_removal_failures
    selected_count = result.selected_count

    # Build success message
    success_parts = []
    if deleted_count > 0:
        success_parts.append(f"Deleted {deleted_count} files")
    if sidecar_deleted_count > 0:
        success_parts.append(f"deleted {sidecar_deleted_count} SDLXLIFF sidecar(s)")
    if sidecar_cleared_count > 0:
        success_parts.append(
            f"retained {sidecar_cleared_count} SDLXLIFF sidecar(s) and cleared their translated targets"
        )
    if machine_translation_deleted_count > 0:
        success_parts.append(f"deleted {machine_translation_deleted_count} Machine Translation preview file(s)")
    if chunk_segment_deleted_count > 0:
        success_parts.append(
            f"removed {chunk_segment_deleted_count} selected chunk segment(s) from HTML"
        )
    if compiled_pdf_section_removed_count > 0:
        success_parts.append(
            "removed "
            f"{compiled_pdf_section_removed_count} selected PDF section(s) "
            "from compiled HTML"
        )
    if compiled_pdf_chunk_segment_removed_count > 0:
        success_parts.append(
            "removed "
            f"{compiled_pdf_chunk_segment_removed_count} selected API chunk "
            "segment(s) from compiled PDF HTML"
        )
    if compiled_pdf_deleted_count > 0:
        success_parts.append(
            f"deleted {compiled_pdf_deleted_count} stale compiled PDF file(s)"
        )
    if chunk_reset_count > 0:
        success_parts.append(
            f"reset {chunk_reset_count} chunk(s) while preserving their siblings"
        )
    if full_chapter_chunk_reset_count > 0:
        success_parts.append(
            "cleared cached chunks for "
            f"{full_chapter_chunk_reset_count} full chapter reset(s)"
        )
    if marked_count > 0:
        success_parts.append(f"marked {marked_count} missing chapters for translation")
    if status_reset_count > 0:
        success_parts.append(f"reset {status_reset_count} chapter statuses to pending")
    if refinement_cleared_count > 0:
        success_parts.append(f"cleared refinement state for {refinement_cleared_count} chapter(s)")
    if merged_cleared_count > 0:
        success_parts.append(f"cleared {merged_cleared_count} merged child chapters")
    if sidecar_failed_count > 0:
        failed_action = "clear" if manual_editing_retranslation else "delete"
        success_parts.append(f"failed to {failed_action} {sidecar_failed_count} SDLXLIFF sidecar(s)")
    if machine_translation_failed_count > 0:
        success_parts.append(f"failed to delete {machine_translation_failed_count} Machine Translation preview file(s)")

    if chunk_segment_removal_failures:
        failure_lines = [
            f"Section {chapter_num} chunk {chunk_index}: "
            f"{os.path.basename(path)} ({reason})"
            for chapter_num, chunk_index, path, reason
            in chunk_segment_removal_failures[:10]
        ]
        if len(chunk_segment_removal_failures) > 10:
            failure_lines.append(
                f"(+{len(chunk_segment_removal_failures) - 10} more)"
            )
        warning_message = (
            "The following chunk(s) were not reset because their HTML "
            "segments could not be removed. Their progress entries were "
            "left unchanged:\n\n" + "\n".join(failure_lines)
        )
        if success_parts:
            warning_message += (
                "\n\nOther selected changes succeeded: "
                + ", ".join(success_parts)
                + "."
            )
        return ('warning', "Chunk HTML Not Updated", warning_message)
    elif success_parts:
        success_msg = "Successfully " + ", ".join(success_parts) + "."
        if deleted_count > 0 or marked_count > 0 or merged_cleared_count > 0:
            total_to_translate = selected_count + merged_cleared_count
            success_msg += f"\n\nTotal {total_to_translate} chapters ready for translation."
        return ('info', "Success", success_msg)
    else:
        return ('info', "Info", "No changes made.")


def retranslate_rows(book, rows, settings=None, *, linked_choice=None, sidecar_workers=None, owner=None):
    """Plan and apply in one call (no confirmation): ``(plan, result_or_None)``.

    ``result`` is None when the plan is refused or is an audio-mode TTS reset (run
    ``reset_tts`` for those), or when a RECYCLED pair needs ``linked_choice`` and it
    is ``'cancel'`` / missing.
    """
    plan = plan_retranslation(book, rows, settings, owner=owner)
    if plan.mode != 'retranslate':
        return plan, None
    if plan.needs_linked_choice and linked_choice not in ('both', 'selected_only'):
        return plan, None
    return plan, apply_retranslation(book, plan, linked_choice, sidecar_workers, owner=owner)


# ---------------------------------------------------------------------------
# Resolve QA issue (raw foreign text): the single-entry Partial.b preflight
# (RG _start_single_progress_qa_resolution 16647-16719, frozen at 41814faa)
# ---------------------------------------------------------------------------


def prepare_single_qa_resolution(owner, data, display_info):
    """Preflight of a single-entry Partial.b QA resolution run.

    Refuses (nothing changes) when the entry no longer has a raw foreign-text QA
    issue (``refresh`` is True: the desktop re-reads the view), when the source file
    is missing, or while a translation or glossary thread runs.  Otherwise sets the
    owner's run state exactly like the desktop (``_single_qa_resolution_request``,
    ``selected_files = [source]``, ``current_file_index = 0``, and clears
    ``_metadata_only_run`` / ``_single_chapter_filter`` / ``_force_stream_all``);
    the caller then starts the translation run (desktop ``run_translation_thread``;
    the translation pipeline turns the request into a Partial.b run on that entry).

    Returns ``{'ok', 'refusal': (kind, title, message) | None, 'refresh', 'request',
    'source_path', 'label', 'log'}``.
    """
    data = data if isinstance(data, dict) else {}
    display_info = display_info if isinstance(display_info, dict) else {}
    outcome = {
        'ok': False, 'refusal': None, 'refresh': False, 'request': None,
        'source_path': '', 'label': '', 'log': '',
    }
    progress_key, progress_entry, output_file = _partial_b_target(
        data, display_info
    )

    if not _progress_entry_has_raw_foreign_text_qa(progress_entry):
        outcome['refusal'] = (
            'info',
            "QA Issue Already Resolved",
            "This entry no longer has a raw foreign-text QA issue.",
        )
        outcome['refresh'] = True
        return outcome

    source_value = str(data.get('file_path') or '').strip()
    source_path = os.path.abspath(source_value) if source_value else ''
    if not source_path or not os.path.isfile(source_path):
        outcome['refusal'] = (
            'error',
            "Source File Missing",
            f"Could not start Partial.b because the source file is missing:\n"
            f"{source_path or '[unknown]'}",
        )
        return outcome
    self = owner
    if (
        getattr(self, 'translation_thread', None)
        and self.translation_thread.is_alive()
    ) or (
        getattr(self, 'glossary_thread', None)
        and self.glossary_thread.is_alive()
    ):
        outcome['refusal'] = (
            'info',
            "Process Running",
            "Wait for the current translation or glossary process to finish first.",
        )
        return outcome

    self._single_qa_resolution_request = _partial_b_request(
        data, progress_key, progress_entry, output_file, source_path
    )
    self.selected_files = [source_path]
    self.current_file_index = 0
    self._metadata_only_run = False
    self._single_chapter_filter = None
    self._force_stream_all = False

    label = (
        display_info.get('translation_artifact_label')
        or display_info.get('metadata_label')
        or output_file
        or f"entry {progress_key}"
    )
    outcome.update(
        ok=True,
        request=self._single_qa_resolution_request,
        source_path=source_path,
        label=label,
        log=f"⚠️ Queued Partial.b QA resolution for {label} only",
    )
    return outcome


__all__ = [
    'RetranslationPlan',
    'RetranslationResult',
    'apply_retranslation',
    'build_partial_b_request',
    'clear_chunk_row_qa_mark',
    'clear_progress_entry_qa_mark',
    'delete_row_audio',
    'find_row_audio',
    'insert_missing_images',
    'llm_token_repair_summary',
    'qa_issue_search_target',
    'plan_remove_qa_marks',
    'plan_retranslation',
    'prepare_single_qa_resolution',
    'refinement_status_keys',
    'remove_pending_marks',
    'remove_pending_message',
    'remove_qa_marks',
    'remove_refinement_status',
    'remove_special_keyword',
    'reset_tts',
    'reset_tts_message',
    'resolve_llm_token_qa',
    'restore_in_progress',
    'restore_in_progress_message',
    'retranslate_rows',
    'retranslation_result_message',
    'row_actions',
    'special_keyword_for',
]


def qa_issue_search_target(qa_file_path, qa_issues):
    """Progress Manager ✏️ Edit File (find QA issue): ``(search_term, line_number)`` for a QA-flagged
    output (moved verbatim from Retranslation_GUI): the first quoted / bracketed text of the QA issue
    lines, else the file's first non-ASCII run; the line is where the term (or its longest prefix)
    first appears (1 when not found). The desktop copies the term and opens an editor at the line;
    Glossarion Mobile's text editor opens at the term."""
    search_term = None
    _line_num = 1
    # Extract a meaningful search term from the QA issue strings
    # Try all common delimiter styles in order
    _QUOTE_PATTERNS = [
        r"'([^']+)'",                    # single quotes: 'text'
        r'"([^"]+)"',                   # double quotes: "text"
        r"\u201c([^\u201d]+)\u201d",    # curly double quotes: “text”
        r"\u2018([^\u2019]+)\u2019",    # curly single quotes: ‘text’
        r"\u300c([^\u300d]+)\u300d",    # Japanese corner brackets: 「text」
        r"\u300e([^\u300f]+)\u300f",    # Japanese white corner brackets: 『text』
        r"\uff62([^\uff63]+)\uff63",    # Halfwidth corner brackets
        r"\[([^\]]+)\]",              # square brackets: [text]
        r"\(([^)]+)\)",               # parentheses: (text)
    ]
    for _issue in qa_issues:
        _s = str(_issue)
        for _pat in _QUOTE_PATTERNS:
            _m = re.search(_pat, _s)
            if _m and _m.group(1).strip():
                search_term = _m.group(1)
                break
        if search_term:
            break
    # Fallback: scan file for any non-ASCII sequence
    if not search_term:
        try:
            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                _content = _f.read()
            _m = re.search(r'[^\x00-\x7f]{1,30}', _content)
            if _m:
                search_term = _m.group(0)
        except Exception:
            pass
    # Find line number of search term in file
    # Try progressively shorter prefixes in case the QA term is truncated
    if search_term and os.path.exists(qa_file_path):
        try:
            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                _lines = _f.readlines()
            # Strip surrounding quote/bracket chars so we search raw content
            _STRIP_QUOTES = '\'"「」『』“”‘’｢｣《》〈〉（）'
            _bare = search_term.strip(_STRIP_QUOTES)
            _base = _bare if _bare else search_term
            # Build candidates: full bare term, then shrinking prefixes (min 1 char)
            _candidates = [_base[:_l] for _l in range(len(_base), 0, -1)]
            for _cand in _candidates:
                for _i, _ln in enumerate(_lines, 1):
                    if _cand in _ln:
                        _line_num = _i
                        break
                if _line_num > 1:
                    break
        except Exception:
            pass
    return search_term, _line_num
