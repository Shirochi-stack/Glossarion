"""Shared EPUB translation-chunk progress and HTML marker helpers.

This module is intentionally dependency-light so the translator, Progress
Manager GUI, QA scanner, and EPUB Library can share one chunk contract without
importing the full translation pipeline.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Iterable


CHUNK_PROGRESS_SCHEMA_VERSION = 2

_CHUNK_BLOCK_RE = re.compile(
    r"<!--\s*GLOSSARION_CHUNK_START\s+key=(?P<key>[0-9a-f]+)\s+"
    r"idx=(?P<idx>\d+)\s+total=(?P<total>\d+)\s*-->"
    r"(?P<content>.*?)"
    r"<!--\s*GLOSSARION_CHUNK_END\s+key=(?P=key)\s+idx=(?P=idx)\s*-->",
    re.IGNORECASE | re.DOTALL,
)


def _positive_int(value, default=0):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def chunk_marker_key(chapter_key) -> str:
    return hashlib.sha256(str(chapter_key or "").encode("utf-8")).hexdigest()[:16]


def chunk_html_markers(chapter_key, chunk_index, total_chunks):
    marker_key = chunk_marker_key(chapter_key)
    chunk_index = _positive_int(chunk_index)
    total_chunks = _positive_int(total_chunks)
    return (
        f"<!-- GLOSSARION_CHUNK_START key={marker_key} "
        f"idx={chunk_index} total={total_chunks} -->",
        f"<!-- GLOSSARION_CHUNK_END key={marker_key} idx={chunk_index} -->",
    )


def wrap_chunk_html(chapter_key, chunk_index, total_chunks, content) -> str:
    # A 1/1 "chunk" is the complete document. Markers only add bookkeeping
    # noise when there is nothing smaller to resume or retranslate.
    if _positive_int(total_chunks) <= 1:
        return str(content or "")
    start, end = chunk_html_markers(chapter_key, chunk_index, total_chunks)
    return f"{start}\n{str(content or '')}\n{end}"


def is_multi_chunk_entry(entry) -> bool:
    """Return True only for a persisted, genuinely split chapter plan."""
    return bool(
        isinstance(entry, dict) and _positive_int(entry.get("total")) > 1
    )


def prune_single_chunk_entries(chapter_chunks) -> list[str]:
    """Remove obsolete 1/1 plans from a ``chapter_chunks`` mapping."""
    if not isinstance(chapter_chunks, dict):
        return []
    removed = []
    for chapter_key, entry in list(chapter_chunks.items()):
        if isinstance(entry, dict) and _positive_int(entry.get("total")) == 1:
            chapter_chunks.pop(chapter_key, None)
            removed.append(str(chapter_key))
    return removed


def extract_marked_chunks(html_text, chapter_key=None):
    """Return marker-delimited chunks keyed by one-based chunk index."""
    text = str(html_text or "")
    expected_key = chunk_marker_key(chapter_key) if chapter_key is not None else None
    chunks = {}
    for match in _CHUNK_BLOCK_RE.finditer(text):
        if expected_key and match.group("key").lower() != expected_key.lower():
            continue
        index = _positive_int(match.group("idx"))
        if not index:
            continue
        chunks[index] = {
            "index": index,
            "total": _positive_int(match.group("total")),
            "marker_key": match.group("key"),
            "content": match.group("content").strip("\r\n"),
            "start": match.start(),
            "end": match.end(),
            "full": match.group(0),
        }
    return chunks


def _chunk_record(entry, chunk_index, create=False):
    if not isinstance(entry, dict):
        return None
    records = entry.setdefault("entries", {}) if create else entry.get("entries", {})
    if not isinstance(records, dict):
        if not create:
            return None
        records = {}
        entry["entries"] = records
    key = str(_positive_int(chunk_index))
    if key == "0":
        return None
    record = records.get(key)
    if not isinstance(record, dict) and create:
        record = {"index": int(key), "status": "pending"}
        records[key] = record
    return record if isinstance(record, dict) else None


def ensure_chunk_entry_schema(entry, total_chunks=None):
    """Upgrade one chapter chunk entry in place and synchronize mirrors."""
    if not isinstance(entry, dict):
        return False
    changed = False
    total = _positive_int(total_chunks, _positive_int(entry.get("total")))
    chunks = entry.get("chunks")
    if not isinstance(chunks, dict):
        chunks = {}
        entry["chunks"] = chunks
        changed = True
    metadata = entry.get("chunk_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        entry["chunk_metadata"] = metadata
        changed = True
    records = entry.get("entries")
    if not isinstance(records, dict):
        records = {}
        entry["entries"] = records
        changed = True

    completed = set()
    for value in entry.get("completed", []) if isinstance(entry.get("completed"), list) else []:
        index = _positive_int(value)
        if index:
            completed.add(index)

    indexes = set(range(1, total + 1))
    for mapping in (chunks, records):
        for raw_index in mapping:
            index = _positive_int(raw_index)
            if index:
                indexes.add(index)

    normalized_records = {}
    normalized_chunks = {}
    for index in sorted(indexes):
        key = str(index)
        old_record = records.get(key)
        record = dict(old_record) if isinstance(old_record, dict) else {}
        result = chunks.get(key)
        if isinstance(result, str):
            normalized_chunks[key] = result
            record["result_sha256"] = hashlib.sha256(
                result.encode("utf-8", errors="ignore")
            ).hexdigest()
        record["index"] = index
        status = str(record.get("status") or "").strip().lower()
        if status not in {"completed", "pending", "in_progress", "qa_failed", "failed"}:
            status = "completed" if index in completed and isinstance(result, str) else "pending"
        if status == "completed" and not isinstance(result, str):
            status = "pending"
        record["status"] = status
        meta = metadata.get(key)
        if isinstance(meta, dict):
            for field in ("model_name", "key_identifier"):
                if meta.get(field) and not record.get(field):
                    record[field] = meta[field]
        record.setdefault("qa_issues_found", [])
        record.setdefault("qa_issue_previews", {})
        normalized_records[key] = record

    normalized_completed = sorted(
        index
        for index, record in (
            (int(key), value) for key, value in normalized_records.items()
        )
        if record.get("status") == "completed" and str(index) in normalized_chunks
    )
    if entry.get("schema_version") != CHUNK_PROGRESS_SCHEMA_VERSION:
        entry["schema_version"] = CHUNK_PROGRESS_SCHEMA_VERSION
        changed = True
    if entry.get("total") != total:
        entry["total"] = total
        changed = True
    if entry.get("entries") != normalized_records:
        entry["entries"] = normalized_records
        changed = True
    if entry.get("chunks") != normalized_chunks:
        entry["chunks"] = normalized_chunks
        changed = True
    if entry.get("completed") != normalized_completed:
        entry["completed"] = normalized_completed
        changed = True
    return changed


def reusable_chunk_results(entry):
    if not is_multi_chunk_entry(entry):
        return {}
    ensure_chunk_entry_schema(entry)
    chunks = entry.get("chunks", {})
    records = entry.get("entries", {})
    return {
        key: result
        for key, result in chunks.items()
        if isinstance(result, str)
        and isinstance(records.get(str(key)), dict)
        and records[str(key)].get("status") == "completed"
    }


def record_chunk_result(
    entry,
    chunk_index,
    result,
    *,
    source_text=None,
    model_name=None,
    key_identifier=None,
):
    if not isinstance(entry, dict) or not isinstance(result, str):
        return False
    ensure_chunk_entry_schema(entry)
    index = _positive_int(chunk_index)
    if not index or (entry.get("total") and index > int(entry["total"])):
        return False
    key = str(index)
    entry.setdefault("chunks", {})[key] = result
    record = _chunk_record(entry, index, create=True)
    record.update({
        "index": index,
        "status": "completed",
        "result_sha256": hashlib.sha256(
            result.encode("utf-8", errors="ignore")
        ).hexdigest(),
        "qa_issues_found": [],
        "qa_issue_previews": {},
        "qa_timestamp": None,
        "last_updated": time.time(),
    })
    if isinstance(source_text, str):
        record["source"] = source_text
        record["source_sha256"] = hashlib.sha256(
            source_text.encode("utf-8", errors="ignore")
        ).hexdigest()
    if model_name:
        record["model_name"] = str(model_name)
    if key_identifier:
        record["key_identifier"] = str(key_identifier)
    metadata = entry.setdefault("chunk_metadata", {})
    if model_name or key_identifier:
        metadata[key] = {
            "model_name": str(model_name or "").strip() or None,
            "key_identifier": str(key_identifier or "").strip() or None,
        }
    ensure_chunk_entry_schema(entry)
    entry["last_updated"] = time.time()
    return True


def set_chunk_runtime_status(
    entry,
    chunk_index,
    status,
    *,
    model_name=None,
    key_identifier=None,
):
    """Set runtime status and the actual route used by one planned chunk."""
    if not isinstance(entry, dict):
        return False
    status = str(status or "").strip().lower()
    if status not in {"pending", "in_progress"}:
        return False
    ensure_chunk_entry_schema(entry)
    record = _chunk_record(entry, chunk_index, create=True)
    if record is None:
        return False
    key = str(_positive_int(chunk_index))
    record["status"] = status
    if model_name:
        record["model_name"] = str(model_name).strip()
    if key_identifier:
        record["key_identifier"] = str(key_identifier).strip()
    if model_name or key_identifier:
        metadata = entry.setdefault("chunk_metadata", {})
        current_metadata = metadata.get(key)
        current_metadata = (
            dict(current_metadata) if isinstance(current_metadata, dict) else {}
        )
        if model_name:
            current_metadata["model_name"] = str(model_name).strip()
        if key_identifier:
            current_metadata["key_identifier"] = str(key_identifier).strip()
        metadata[key] = current_metadata
    record["last_updated"] = time.time()
    ensure_chunk_entry_schema(entry)
    statuses = {
        str(value.get("status") or "pending").lower()
        for value in entry.get("entries", {}).values()
        if isinstance(value, dict)
    }
    if statuses.intersection({"qa_failed", "failed"}):
        entry["chapter_status"] = "qa_failed"
    elif "in_progress" in statuses:
        entry["chapter_status"] = "in_progress"
    elif "pending" in statuses:
        entry["chapter_status"] = "incomplete"
    else:
        entry["chapter_status"] = "completed"
    entry["last_updated"] = time.time()
    return True


def reset_in_progress_chunks(entry):
    """Return all interrupted in-flight chunks to resumable pending state."""
    if not isinstance(entry, dict):
        return []
    ensure_chunk_entry_schema(entry)
    reset = []
    for raw_index, record in entry.get("entries", {}).items():
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "").lower() != "in_progress":
            continue
        record["status"] = "pending"
        record["last_updated"] = time.time()
        try:
            reset.append(int(raw_index))
        except (TypeError, ValueError):
            pass
    if reset:
        statuses = {
            str(value.get("status") or "pending").lower()
            for value in entry.get("entries", {}).values()
            if isinstance(value, dict)
        }
        entry["chapter_status"] = (
            "qa_failed"
            if statuses.intersection({"qa_failed", "failed"})
            else "incomplete"
        )
        entry["last_updated"] = time.time()
        ensure_chunk_entry_schema(entry)
    return sorted(reset)


def set_chunk_qa(entry, chunk_index, issues, previews=None, confidence=0):
    """Set or clear QA state for one persisted chunk result."""
    if not isinstance(entry, dict):
        return False
    ensure_chunk_entry_schema(entry)
    record = _chunk_record(entry, chunk_index, create=True)
    if record is None:
        return False
    issues = list(issues or [])
    record["qa_issues_found"] = issues
    record["qa_issue_previews"] = dict(previews or {})
    record["qa_timestamp"] = time.time()
    record["duplicate_confidence"] = confidence or 0
    if issues:
        record["status"] = "qa_failed"
    elif str(chunk_index) in entry.get("chunks", {}):
        record["status"] = "completed"
    else:
        record["status"] = "pending"
    record["last_updated"] = time.time()
    ensure_chunk_entry_schema(entry)
    summary = chunk_failure_summary(entry)
    entry["chapter_status"] = (
        "qa_failed"
        if summary["failed"]
        else "incomplete"
        if summary["pending"]
        else "completed"
    )
    entry["last_updated"] = time.time()
    return True


def reset_chunks_for_retranslation(entry, chunk_indices: Iterable[int]):
    """Drop selected cached results and mark only those chunks pending."""
    if not isinstance(entry, dict):
        return []
    ensure_chunk_entry_schema(entry)
    reset = []
    for raw_index in chunk_indices or []:
        index = _positive_int(raw_index)
        record = _chunk_record(entry, index, create=False)
        if not index or record is None:
            continue
        key = str(index)
        entry.get("chunks", {}).pop(key, None)
        entry.get("chunk_metadata", {}).pop(key, None)
        for field in (
            "result_sha256", "model_name", "key_identifier", "qa_timestamp",
            "duplicate_confidence",
        ):
            record.pop(field, None)
        record["status"] = "pending"
        record["qa_issues_found"] = []
        record["qa_issue_previews"] = {}
        record["last_updated"] = time.time()
        reset.append(index)
    ensure_chunk_entry_schema(entry)
    if reset:
        entry["chapter_status"] = "incomplete"
        entry["last_updated"] = time.time()
    return reset


def remove_chunk_segments(html_text, chapter_key, chunk_indices, entry=None):
    """Remove selected marker blocks from one chapter output document.

    The progress content hash can change independently of an already-written
    marker key (for example after PDF source normalization). Because one
    response HTML file represents exactly one chapter/section, an unambiguous
    single marker plan in that file is a safe secondary identity.
    """
    text = str(html_text or "")
    wanted = {_positive_int(index) for index in chunk_indices or []}
    wanted.discard(0)
    removed = []
    marked = extract_marked_chunks(text, chapter_key)
    spans = []
    for index in sorted(wanted):
        block = marked.get(index)
        if block:
            spans.append((block["start"], block["end"], index))
    for start, end, index in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
        removed.append(index)

    remaining = wanted.difference(removed)
    if remaining:
        all_marked = []
        for match in _CHUNK_BLOCK_RE.finditer(text):
            all_marked.append({
                "index": _positive_int(match.group("idx")),
                "total": _positive_int(match.group("total")),
                "marker_key": match.group("key"),
                "start": match.start(),
                "end": match.end(),
            })
        marker_keys = {
            str(block.get("marker_key") or "").casefold()
            for block in all_marked
            if isinstance(block, dict) and block.get("marker_key")
        }
        entry_total = _positive_int(
            entry.get("total") if isinstance(entry, dict) else 0
        )
        marker_totals = {
            _positive_int(block.get("total"))
            for block in all_marked
            if isinstance(block, dict)
        }
        marker_totals.discard(0)
        plan_is_unambiguous = bool(
            len(marker_keys) == 1
            and len(marker_totals) == 1
            and (
                not entry_total
                or marker_totals == {entry_total}
            )
        )
        if plan_is_unambiguous:
            blocks_by_index = {
                block["index"]: block
                for block in all_marked
                if block.get("index")
            }
            fallback_spans = []
            for index in sorted(remaining):
                block = blocks_by_index.get(index)
                if block:
                    fallback_spans.append(
                        (block["start"], block["end"], index)
                    )
            for start, end, index in sorted(fallback_spans, reverse=True):
                text = text[:start] + text[end:]
                removed.append(index)

    remaining = wanted.difference(removed)
    if remaining and isinstance(entry, dict):
        chunks = entry.get("chunks", {}) if isinstance(entry.get("chunks"), dict) else {}
        for index in sorted(remaining):
            result = chunks.get(str(index))
            if not isinstance(result, str) or not result:
                continue
            candidates = [result]
            without_fence = re.sub(
                r"\n?```\s*$",
                "",
                re.sub(
                    r"^```(?:html)?\s*\n?",
                    "",
                    result,
                    count=1,
                    flags=re.MULTILINE | re.IGNORECASE,
                ),
                count=1,
                flags=re.MULTILINE,
            )
            if without_fence != result:
                candidates.append(without_fence)
            for candidate in candidates:
                if candidate and candidate in text:
                    text = text.replace(candidate, "", 1)
                    removed.append(index)
                    break
    return text, sorted(removed)


def chunk_failure_summary(entry):
    if not is_multi_chunk_entry(entry):
        return {"total": 0, "completed": 0, "failed": 0, "pending": 0}
    ensure_chunk_entry_schema(entry)
    records = entry.get("entries", {})
    statuses = [
        str(record.get("status") or "pending").lower()
        for record in records.values()
        if isinstance(record, dict)
    ]
    return {
        "total": _positive_int(entry.get("total"), len(statuses)),
        "completed": statuses.count("completed"),
        "failed": sum(status in {"qa_failed", "failed"} for status in statuses),
        "pending": sum(status in {"pending", "in_progress"} for status in statuses),
    }


def chunk_entry_needs_translation(entry):
    """Return whether a persisted chunk plan still has work to translate."""
    summary = chunk_failure_summary(entry)
    return bool(summary["total"] and (summary["failed"] or summary["pending"]))


def effective_parent_status(status, entry):
    """Prevent pending chunks from being hidden by a completed parent row.

    Individual chunk QA failures intentionally do not fail an otherwise
    completed parent chapter: the scanner and UIs expose those failures on the
    child rows. Pending/in-progress chunks are different; they mean the chapter
    output is structurally incomplete and therefore cannot be called complete.
    """
    base_status = str(status or "").strip().lower()
    summary = chunk_failure_summary(entry)
    if not summary["total"] or not summary["pending"]:
        return status
    if base_status in {
        "in_progress", "qa_failed", "failed", "error", "file_missing"
    }:
        return status
    return "pending"


def chunk_status_summary_text(entry, limit=8):
    if not is_multi_chunk_entry(entry):
        return ""
    ensure_chunk_entry_schema(entry)
    icons = {
        "completed": "✓",
        "qa_failed": "⚠",
        "failed": "✗",
        "in_progress": "…",
        "pending": "○",
    }
    records = entry.get("entries", {})
    labels = []
    for key in sorted(records, key=lambda value: _positive_int(value)):
        record = records[key]
        status = str(record.get("status") or "pending").lower()
        labels.append(f"{key}{icons.get(status, '?')}")
    if len(labels) > limit:
        labels = labels[:limit] + [f"+{len(labels) - limit}"]
    return "Chunks " + " ".join(labels) if labels else ""
