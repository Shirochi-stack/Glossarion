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

The desktop closures keep their selection, confirmation dialogs, refresh and messages
and call these functions; every progress write goes through
``progress_core.mutate_progress`` (lock, re-read, three-way merge, atomic replace),
applying the action to the newest snapshot instead of the dialog's cached copy
(the former whole-file writes are listed in tests/parity/DISCREPANCIES.md, U5).
Retranslate Selected's plan/apply arrives with milestone U7.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from chapter_chunk_progress import (
    chunk_failure_summary,
    ensure_chunk_entry_schema,
    is_multi_chunk_entry,
    set_chunk_qa,
    sorted_chunk_items,
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
from sdlxliff_sidecar_writer import _reset_sdlxliff_target_for_manual_retranslation


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


__all__ = [
    'build_partial_b_request',
    'clear_chunk_row_qa_mark',
    'clear_progress_entry_qa_mark',
    'delete_row_audio',
    'find_row_audio',
    'insert_missing_images',
    'llm_token_repair_summary',
    'plan_remove_qa_marks',
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
    'row_actions',
    'special_keyword_for',
]
