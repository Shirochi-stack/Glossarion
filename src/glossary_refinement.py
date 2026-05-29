# -*- coding: utf-8 -*-
"""Shared optional glossary refinement step.

This module owns the refinement prompt/config/progress behavior. Callers keep
their own glossary loading/saving formats and pass in their parser, deduper, and
API sender so the balanced/full and minimal paths stay in sync without becoming
coupled to each other's implementation details.
"""

import hashlib
import csv
import io
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional

DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT = """You are refining an already extracted translation glossary.

Your job is cleanup, not broad re-extraction. Preserve useful entries and return only the refined glossary entries for the provided entry type or entry types.

Glossary schema:
{fields}

Active refinement entry types:
{entries}

Critical refinement rules:
- Keep the existing glossary schema and fields. Return refined glossary CSV data rows only, using the columns and delimiter shown in the glossary schema above. Do not include a header row.
- Remove duplicate entries, near-duplicates, and entries that only differ by trivial spacing, casing, honorifics, or punctuation.
- Remove generic or unnecessary entries that are not useful for translation consistency.
- For character entries, ensure there are no full-name character entries. If a character appears as a full name, split it into separate entries for the given name/first name and surname/family name. Do not combine first names, surnames, titles, nicknames, or aliases into one entry. Keep raw_name focused on the exact source form and translated_name focused on the target form.
- Reject useless entries where raw_name and translated_name are essentially the same word or duplicate text.
- Do not invent entries, translations, genders, descriptions, aliases, or facts that are not present in the provided glossary content.
- If two entries conflict, keep the more specific and translation-useful one.
- Keep active custom entry types separate; do not move entries into another type unless the current entry type is plainly wrong.

Return only the refined glossary content. Do not include markdown, explanations, comments, or surrounding prose."""

DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT = ""

_progress_lock = threading.Lock()
_SCHEMA_PLACEHOLDERS = ("{fields1}", "{{fields1}}", "{fields}", "{{fields}}", "{columns}", "{{columns}}")


def refinement_enabled() -> bool:
    return os.getenv("GLOSSARY_REFINEMENT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def selected_refinement_types(active_types: Iterable[str]) -> List[str]:
    active = [str(t).strip() for t in active_types if str(t).strip()]
    mode = os.getenv("GLOSSARY_REFINEMENT_TYPE_MODE", "all").strip().lower()
    if mode != "selected":
        return active
    raw = os.getenv("GLOSSARY_REFINEMENT_SELECTED_TYPES", "")
    selected = [t.strip() for t in raw.split(",") if t.strip()]
    selected_lc = {t.lower() for t in selected}
    return [t for t in active if t.lower() in selected_lc]


def _batch_translation_enabled() -> bool:
    return os.getenv("BATCH_TRANSLATION", "0").strip().lower() in ("1", "true", "yes", "on")


def _batch_size() -> int:
    try:
        return max(1, int(os.getenv("BATCH_SIZE", os.getenv("GLOSSARY_BATCH_SIZE", "1"))))
    except Exception:
        return 1


def _has_schema_placeholder(prompt_text: str) -> bool:
    text = str(prompt_text or "")
    return any(token in text for token in _SCHEMA_PLACEHOLDERS)


def _is_legacy_default_refinement_prompt(prompt_text: str) -> bool:
    text = str(prompt_text or "")
    return (
        "using the same columns and delimiter shown in the provided glossary content" in text
        and not _has_schema_placeholder(text)
    )


def _active_custom_fields() -> List[str]:
    try:
        custom_fields = json.loads(os.getenv("GLOSSARY_CUSTOM_FIELDS", "[]"))
        if not isinstance(custom_fields, list):
            return []
        return [str(field).strip() for field in custom_fields if str(field).strip()]
    except Exception:
        return []


def _description_active(custom_fields: Optional[List[str]] = None) -> bool:
    fields = custom_fields if custom_fields is not None else _active_custom_fields()
    return any(str(field).strip().lower() == "description" for field in fields or [])


def _strip_inactive_description(entries: List[Dict]) -> List[Dict]:
    if _description_active():
        return [dict(entry) for entry in entries or [] if isinstance(entry, dict)]
    cleaned = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        cleaned.append({
            key: value
            for key, value in dict(entry).items()
            if str(key).strip().lower() != "description"
        })
    return cleaned


def _refinement_budget_label(available_tokens: int, mtoks: int) -> str:
    try:
        budget = int(available_tokens)
    except Exception:
        budget = available_tokens
    try:
        output_limit = int(mtoks)
    except Exception:
        output_limit = None
    try:
        raw_factor = os.getenv(
            "GLOSSARY_REFINEMENT_COMPRESSION_FACTOR",
            os.getenv("COMPRESSION_FACTOR", os.getenv("GLOSSARY_COMPRESSION_FACTOR", "")),
        )
        compression_factor = float(raw_factor)
    except Exception:
        raw_factor = ""
        compression_factor = None

    parts = []
    if output_limit:
        parts.append(f"output limit {output_limit:,}")
    if compression_factor and compression_factor > 0:
        margin = None
        try:
            if int(max(1000, int((output_limit - 500) / compression_factor))) == int(budget):
                margin = 500
        except Exception:
            margin = None
        if margin is not None:
            parts.append(f"margin {margin:,}")
        parts.append(f"compression {raw_factor or compression_factor}")
    return f"budget {int(budget):,}" + (f" ({', '.join(parts)})" if parts else "")


def _entry_columns(entries: List[Dict]) -> List[str]:
    columns = ["type", "raw_name", "translated_name"]
    custom_fields = _active_custom_fields()
    description_active = _description_active(custom_fields)
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for key in entry.keys():
            if str(key).strip().lower() == "description" and not description_active:
                continue
            if key not in columns:
                columns.append(key)
    for field in custom_fields:
        field = str(field or "").strip()
        if field and field not in columns:
            columns.append(field)
    return columns


def _prompt_requests_unit_separator(system_prompt: str, user_prompt: str) -> bool:
    prompt_text = f"{system_prompt or ''}\n{user_prompt or ''}"
    return (
        "{fields1}" in prompt_text
        or "{{fields1}}" in prompt_text
        or "Unit Separator" in prompt_text
        or "\\x1F" in prompt_text
        or "\\x1f" in prompt_text
        or "\x1F" in prompt_text
    )


def _join_payload_row(values: List[str], delimiter: str) -> str:
    if delimiter == "\x1F":
        return delimiter.join(values)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="")
    writer.writerow(values)
    return out.getvalue()


def _entry_payload(entries: List[Dict], columns: Optional[List[str]] = None, delimiter: str = ",") -> str:
    columns = columns or _entry_columns(entries)
    lines = []
    for entry in entries:
        lines.append(_join_payload_row([str(entry.get(col, "")) for col in columns], delimiter))
    return "\n".join(lines)


def _entry_hash(entry_type: str, entries: List[Dict], chunking_mode: str) -> str:
    payload = {
        "entry_type": entry_type,
        "chunking_mode": chunking_mode,
        "entries": entries,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_replace_file(src: str, dst: str, atomic_replace_fn: Optional[Callable[[str, str], None]] = None) -> None:
    if atomic_replace_fn:
        atomic_replace_fn(src, dst)
    else:
        os.replace(src, dst)


@contextmanager
def locked_progress_file(progress_file: Optional[str]):
    """Serialize progress JSON mutations across processes."""
    if not progress_file:
        yield
        return

    progress_dir = os.path.dirname(progress_file) or "."
    os.makedirs(progress_dir, exist_ok=True)
    lock_path = f"{progress_file}.lock"
    lock_f = open(lock_path, "a+b")
    locked = False
    try:
        if lock_f.seek(0, os.SEEK_END) == 0:
            lock_f.write(b"\0")
            lock_f.flush()
            os.fsync(lock_f.fileno())
        lock_f.seek(0)

        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    lock_f.seek(0)
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            locked = True

        yield
    finally:
        try:
            if locked:
                lock_f.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


def load_refinement_progress(progress_file: Optional[str]) -> Dict:
    if not progress_file or not os.path.exists(progress_file):
        return {}
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        refinement = data.get("refinement", {}) if isinstance(data, dict) else {}
        return refinement if isinstance(refinement, dict) else {}
    except Exception:
        return {}


def update_refinement_progress(
    progress_file: Optional[str],
    key: str,
    entry: Dict,
    *,
    atomic_replace_fn: Optional[Callable[[str, str], None]] = None,
) -> None:
    if not progress_file:
        return
    with _progress_lock:
        with locked_progress_file(progress_file):
            data = {}
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}
            refinement = data.setdefault("refinement", {})
            if not isinstance(refinement, dict):
                refinement = {}
                data["refinement"] = refinement
            existing = refinement.get(key, {}) if isinstance(refinement.get(key), dict) else {}
            merged = dict(existing)
            merged.update(entry or {})
            merged["last_updated"] = time.time()
            refinement[key] = merged

            progress_dir = os.path.dirname(progress_file) or "."
            os.makedirs(progress_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=progress_dir,
                delete=False,
                suffix=".tmp",
            ) as temp_f:
                temp_path = temp_f.name
                json.dump(data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())
            _atomic_replace_file(temp_path, progress_file, atomic_replace_fn)


def remove_refinement_progress(
    progress_file: Optional[str],
    key: str,
    *,
    atomic_replace_fn: Optional[Callable[[str, str], None]] = None,
) -> None:
    if not progress_file or not key or not os.path.exists(progress_file):
        return
    with _progress_lock:
        with locked_progress_file(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return
            if not isinstance(data, dict):
                return
            refinement = data.get("refinement")
            if not isinstance(refinement, dict) or key not in refinement:
                return
            refinement.pop(key, None)

            progress_dir = os.path.dirname(progress_file) or "."
            os.makedirs(progress_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=progress_dir,
                delete=False,
                suffix=".tmp",
            ) as temp_f:
                temp_path = temp_f.name
                json.dump(data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())
            _atomic_replace_file(temp_path, progress_file, atomic_replace_fn)


def _render_prompt_placeholders(prompt_text: str, columns: List[str], entry_type: str, chunk_idx=None, total_chunks=None, active_entry_types: Optional[List[str]] = None) -> str:
    if not prompt_text:
        return ""
    sep = "\x1F"
    fields1 = f"Columns (separated by Unit Separator character \\x1F):\n{sep.join(columns)}"
    fields = f"Columns:\n{', '.join(columns)}"
    entries = ", ".join(str(t).strip() for t in (active_entry_types or []) if str(t).strip())
    replacements = {
        "{fields1}": fields1,
        "{{fields1}}": fields1,
        "{fields}": fields,
        "{{fields}}": fields,
        "{columns}": fields,
        "{{columns}}": fields,
        "{entries}": entries,
        "{{entries}}": entries,
        "{entry_type}": str(entry_type or ""),
        "{{entry_type}}": str(entry_type or ""),
        "{chunk_index}": str(chunk_idx or ""),
        "{{chunk_index}}": str(chunk_idx or ""),
        "{total_chunks}": str(total_chunks or ""),
        "{{total_chunks}}": str(total_chunks or ""),
    }
    rendered = str(prompt_text)
    for needle, replacement in replacements.items():
        rendered = rendered.replace(needle, replacement)
    return rendered


def _build_messages(system_prompt: str, user_prompt: str, entry_type: str, chunk_text: str, columns: List[str], chunk_idx=None, total_chunks=None, active_entry_types: Optional[List[str]] = None) -> List[Dict]:
    messages = []
    system_prompt = _render_prompt_placeholders(system_prompt, columns, entry_type, chunk_idx, total_chunks, active_entry_types)
    user_prompt = _render_prompt_placeholders(user_prompt, columns, entry_type, chunk_idx, total_chunks, active_entry_types)
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    user_parts = []
    if user_prompt:
        user_parts.append(user_prompt.strip())
    user_parts.append(chunk_text)
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def _sanitize_messages_for_api(msgs: List[Dict], fallback_text: str) -> List[Dict]:
    if not any(m.get("role") == "user" for m in msgs):
        msgs = msgs + [{"role": "user", "content": fallback_text or ""}]
    sanitized = []
    for msg in msgs:
        msg2 = dict(msg)
        msg2.pop("_raw_content_object", None)
        if msg2.get("content") is None:
            msg2["content"] = ""
        sanitized.append(msg2)
    return sanitized


def _issue_from_finish_reason(finish_reason, default_issue=None):
    finish_text = str(finish_reason or "").strip().lower()
    if finish_text in ("length", "max_tokens") or "max_tokens" in finish_text:
        return "TRUNCATED"
    return default_issue


def _actual_request_model_name(client=None) -> str:
    try:
        from unified_api_client import get_current_thread_actual_request_model
        model_name = str(get_current_thread_actual_request_model() or "").strip()
        if model_name:
            return model_name
    except Exception:
        pass
    try:
        if client is not None and hasattr(client, "get_last_actual_request_model"):
            model_name = str(client.get_last_actual_request_model() or "").strip()
            if model_name:
                return model_name
    except Exception:
        pass
    try:
        tls = client._get_thread_local_client() if client is not None and hasattr(client, "_get_thread_local_client") else None
        model_name = str(getattr(tls, "model", "") or "").strip()
        if model_name:
            return model_name
    except Exception:
        pass
    try:
        return str(getattr(client, "model", "") or "").strip()
    except Exception:
        return ""


def _key_pool_from_identifier(key_identifier: str) -> str:
    key_identifier = str(key_identifier or "").strip()
    pool_prefixes = (
        ("GlossaryRefinementKey#", "glossary_refinement"),
        ("GlossaryKey#", "glossary"),
        ("MetadataKey#", "metadata"),
        ("VisionKey#", "vision"),
        ("TruncationRetryKey#", "truncation_retry"),
        ("AITruncationDetectionKey#", "ai_truncation_detection"),
        ("ImageGenEditKey#", "inpainter"),
        ("Key#", "multi"),
        ("FALLBACK KEY", "fallback"),
        ("Main Key", "main"),
        ("Single Key", "single"),
    )
    for prefix, pool_name in pool_prefixes:
        if key_identifier.startswith(prefix):
            return pool_name
    return ""


def _actual_request_key_identifier(client=None) -> str:
    try:
        from unified_api_client import get_current_thread_actual_request_key_identifier
        key_identifier = str(get_current_thread_actual_request_key_identifier() or "").strip()
        if key_identifier:
            return key_identifier
    except Exception:
        pass
    try:
        if client is not None and hasattr(client, "get_last_actual_request_key_identifier"):
            key_identifier = str(client.get_last_actual_request_key_identifier() or "").strip()
            if key_identifier:
                return key_identifier
    except Exception:
        pass
    try:
        tls = client._get_thread_local_client() if client is not None and hasattr(client, "_get_thread_local_client") else None
        key_identifier = str(getattr(tls, "last_actual_key_identifier", "") or getattr(tls, "key_identifier", "") or "").strip()
        if key_identifier:
            return key_identifier
    except Exception:
        pass
    try:
        return str(getattr(client, "last_actual_key_identifier", "") or getattr(client, "key_identifier", "") or "").strip()
    except Exception:
        return ""


def _actual_request_key_context(client=None) -> Dict:
    key_identifier = _actual_request_key_identifier(client)
    if not key_identifier:
        return {}
    context = {"key_identifier": key_identifier}
    key_pool = _key_pool_from_identifier(key_identifier)
    if key_pool:
        context["key_pool"] = key_pool
    return context


def _call_send(send_fn, messages, client, temp, mtoks, check_stop, chunk_timeout, chunk_idx, total_chunks, context_label):
    try:
        client.context = context_label
        if hasattr(client, "_get_thread_local_client"):
            tls = client._get_thread_local_client()
            tls.current_request_context = context_label
    except Exception:
        pass
    try:
        return send_fn(
            messages,
            client,
            temp,
            mtoks,
            check_stop,
            chunk_timeout=chunk_timeout,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            context=context_label,
        )
    except TypeError:
        return send_fn(
            messages=messages,
            client=client,
            temperature=temp,
            max_tokens=mtoks,
            stop_check_fn=check_stop,
            chunk_timeout=chunk_timeout,
            context=context_label,
        )


def refine_glossary_entries(
    glossary: List[Dict],
    *,
    client,
    temp: float,
    mtoks: int,
    check_stop: Callable[[], bool],
    chapter_splitter,
    available_tokens: int,
    chunk_timeout,
    parse_response_fn: Callable[[str], List[Dict]],
    dedupe_fn: Callable[[List[Dict]], List[Dict]],
    custom_entry_types_fn: Callable[[], Dict],
    send_fn: Callable,
    progress_file: Optional[str] = None,
    output_path: Optional[str] = None,
    atomic_replace_fn: Optional[Callable[[str, str], None]] = None,
    log: Callable[[str], None] = print,
) -> List[Dict]:
    if not refinement_enabled():
        return glossary
    if not glossary:
        log("Glossary refinement enabled, but glossary is empty; skipping.")
        return glossary
    glossary = _strip_inactive_description(glossary)

    custom_types = custom_entry_types_fn()
    active_types = [t for t, cfg in custom_types.items() if not isinstance(cfg, dict) or cfg.get("enabled", True)]
    selected_types = selected_refinement_types(active_types)
    if not selected_types:
        log("Glossary refinement enabled, but no active/selected entry types matched; skipping.")
        return glossary

    system_prompt = os.getenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT)
    if not str(system_prompt or "").strip() or _is_legacy_default_refinement_prompt(system_prompt):
        system_prompt = DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
    user_prompt = os.getenv("GLOSSARY_REFINEMENT_USER_PROMPT", DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT)
    raw_chunking_mode = os.getenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "separate").strip().lower()
    send_all_types = raw_chunking_mode in ("all", "all_types", "all_in_one", "all_entries", "combined")
    canonical_mode = "all" if send_all_types else "separate"
    skip_dedupe = os.getenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "0").strip().lower() in ("1", "true", "yes", "on")
    payload_delimiter = "\x1F" if _prompt_requests_unit_separator(system_prompt, user_prompt) else ","
    payload_delimiter_name = "unit_separator" if payload_delimiter == "\x1F" else "comma"
    hash_mode = f"{canonical_mode}:{payload_delimiter_name}"

    log(f"\n🧹 Glossary refinement enabled for: {', '.join(selected_types)}")
    refined_by_type = {}
    progress = load_refinement_progress(progress_file)
    selected_lc = {t.lower() for t in selected_types}

    def _count_payload_tokens(text: str) -> int:
        try:
            return chapter_splitter.count_tokens(text)
        except Exception:
            return len(text) // 3
    budget_label = _refinement_budget_label(available_tokens, mtoks)

    all_selected_entries = [
        dict(e) for e in glossary
        if str(e.get("type", "")).strip().lower() in selected_lc
    ]

    broad_type_key = f"all::{','.join(selected_types)}"
    for old_key, old_info in list(progress.items()):
        old_entry_type = ""
        if isinstance(old_info, dict):
            old_entry_type = str(old_info.get("entry_type") or "").strip().lower()
        if (
            (str(old_key).startswith("all::") and str(old_key) != broad_type_key)
            or (old_entry_type in ("selected glossary entries", "all selected entry types") and str(old_key) != broad_type_key)
        ):
            remove_refinement_progress(progress_file, old_key, atomic_replace_fn=atomic_replace_fn)
            progress.pop(old_key, None)

    type_keys = {entry_type: f"type::{entry_type}" for entry_type in selected_types}
    entries_by_type = {
        entry_type: [
            e for e in all_selected_entries
            if str(e.get("type", "")).strip().lower() == entry_type.lower()
        ]
        for entry_type in selected_types
    }
    type_hashes = {
        entry_type: _entry_hash(entry_type, entries, hash_mode)
        for entry_type, entries in entries_by_type.items()
        if entries
    }
    pending_types = []

    for entry_type in selected_types:
        entries = entries_by_type.get(entry_type) or []
        type_key = type_keys[entry_type]
        type_hash = type_hashes.get(entry_type) or _entry_hash(entry_type, entries, hash_mode)
        type_progress = progress.get(type_key, {})
        if isinstance(type_progress, dict) and type_progress.get("status") == "completed":
            # A completed type can appear in either shape on the next run:
            # the original pre-refinement input or the refined output loaded
            # back from glossary.csv/json. Accept both hashes so completed
            # refinement work is not resent just because the persisted file is
            # already refined.
            completed_hashes = {
                str(type_progress.get("input_hash") or ""),
                str(type_progress.get("output_hash") or ""),
            }
            if type_hash in completed_hashes:
                continue
        if not entries:
            no_entries_update = {
                "entry_type": entry_type,
                "status": "completed",
                "input_hash": type_hash,
                "output_hash": type_hash,
                "chunking_mode": canonical_mode,
                "payload_delimiter": payload_delimiter_name,
                "entry_count_before": 0,
                "entry_count_after": 0,
                "completed_chunks": 0,
                "total_chunks": 0,
                "output_file": os.path.basename(output_path or ""),
                "reason": "no_entries",
            }
            update_refinement_progress(progress_file, type_key, no_entries_update, atomic_replace_fn=atomic_replace_fn)
            progress[type_key] = dict(progress.get(type_key, {}), **no_entries_update)
            continue
        placeholder = {
            "entry_type": entry_type,
            "status": "not_refined",
            "input_hash": type_hash,
            "chunking_mode": canonical_mode,
            "payload_delimiter": payload_delimiter_name,
            "entry_count_before": len(entries),
            "output_file": os.path.basename(output_path or ""),
        }
        update_refinement_progress(progress_file, type_key, placeholder, atomic_replace_fn=atomic_replace_fn)
        existing = dict(type_progress) if isinstance(type_progress, dict) else {}
        existing.update(placeholder)
        progress[type_key] = existing
        pending_types.append(entry_type)

    if not pending_types:
        log("Glossary refinement already completed for selected entry types, or no entries were present; skipping.")
        return glossary

    selected_types = pending_types
    selected_lc = {t.lower() for t in selected_types}
    all_selected_entries = [
        e for entry_type in selected_types for e in entries_by_type.get(entry_type, [])
    ]
    broad_input_hash = _entry_hash("all selected entry types", all_selected_entries, hash_mode)
    if send_all_types:
        broad_placeholder = {
            "entry_type": "all selected entry types",
            "status": "not_refined",
            "input_hash": broad_input_hash,
            "chunking_mode": canonical_mode,
            "payload_delimiter": payload_delimiter_name,
            "entry_count_before": len(all_selected_entries),
            "output_file": os.path.basename(output_path or ""),
        }
        update_refinement_progress(progress_file, broad_type_key, broad_placeholder, atomic_replace_fn=atomic_replace_fn)
        existing_broad = dict(progress.get(broad_type_key, {})) if isinstance(progress.get(broad_type_key), dict) else {}
        existing_broad.update(broad_placeholder)
        progress[broad_type_key] = existing_broad
    if send_all_types:
        groups = [("selected glossary entries", all_selected_entries, broad_type_key, selected_lc)]
    else:
        groups = [
            (entry_type, entries_by_type.get(entry_type, []), type_keys[entry_type], {entry_type.lower()})
            for entry_type in selected_types
        ]

    def _original_mapping_for_group(entry_type, entries):
        if send_all_types:
            return {
                selected_type: [
                    e for e in entries
                    if str(e.get("type", "")).strip().lower() == selected_type.lower()
                ]
                for selected_type in selected_types
            }
        return {entry_type: entries}

    def _process_group(group):
        entry_type, entries, type_key, allowed_types_lc = group
        if check_stop():
            return "stopped", entry_type, {}
        if not entries:
            return "empty", entry_type, {}

        payload_columns = _entry_columns(entries)
        planned_chunks = []
        group_selected_types = selected_types if send_all_types else [entry_type]

        if send_all_types:
            combined_payload = _entry_payload(entries, payload_columns, payload_delimiter)
            token_count = _count_payload_tokens(combined_payload)
            if token_count <= available_tokens:
                log(f"Glossary refinement keeping all selected entry types as one fitted chunk ({token_count:,} tokens, {budget_label}).")
                planned_chunks.append((combined_payload, entry_type, True))
            else:
                split_chunks = [
                    chunk_text
                    for chunk_html, _local_idx, _local_total in chapter_splitter.split_chapter(
                        combined_payload,
                        available_tokens,
                        filename="glossary_refinement_selected_types.txt",
                    )
                    for chunk_text in [str(chunk_html or "").strip()]
                    if chunk_text
                ]
                if not split_chunks:
                    split_chunks = [combined_payload]
                log(f"🪓 Glossary refinement split all selected entry types into {len(split_chunks)} token-budgeted chunk(s) ({token_count:,} tokens, {budget_label}).")
                planned_chunks.extend((chunk_text, entry_type, False) for chunk_text in split_chunks)
        else:
            for selected_type in group_selected_types:
                type_entries = [
                    e for e in entries
                    if str(e.get("type", "")).strip().lower() == selected_type.lower()
                ]
                if not type_entries:
                    continue
                type_payload = _entry_payload(type_entries, payload_columns, payload_delimiter)
                token_count = _count_payload_tokens(type_payload)
                if token_count <= available_tokens:
                    log(f"Glossary refinement keeping {selected_type} entries as one fitted type chunk ({token_count:,} tokens, {budget_label}).")
                    planned_chunks.append((type_payload, selected_type, True))
                    continue

                split_chunks = [
                    chunk_text
                    for chunk_html, _local_idx, _local_total in chapter_splitter.split_chapter(
                        type_payload,
                        available_tokens,
                        filename=f"glossary_refinement_{selected_type}.txt",
                    )
                    for chunk_text in [str(chunk_html or "").strip()]
                    if chunk_text
                ]
                if not split_chunks:
                    split_chunks = [type_payload]
                log(f"🪓 Glossary refinement split oversized {selected_type} entries into {len(split_chunks)} token-budgeted chunk(s) ({token_count:,} tokens, {budget_label}).")
                planned_chunks.extend((chunk_text, selected_type, False) for chunk_text in split_chunks)

        if not planned_chunks:
            payload = _entry_payload(entries, payload_columns, payload_delimiter)
            planned_chunks = [(payload, entry_type, True)]
        total_chunks = len(planned_chunks)
        chunks = [
            (chunk_text, chunk_idx, total_chunks, chunk_entry_type, whole_type_chunk)
            for chunk_idx, (chunk_text, chunk_entry_type, whole_type_chunk) in enumerate(planned_chunks, 1)
        ]
        if send_all_types:
            per_type_total_chunks = {
                selected_type: total_chunks
                for selected_type in group_selected_types
                if entries_by_type.get(selected_type)
            }
        else:
            per_type_total_chunks = {}
            for _chunk_text, _chunk_idx, _total_chunks, chunk_entry_type, _whole_type_chunk in chunks:
                per_type_total_chunks[chunk_entry_type] = per_type_total_chunks.get(chunk_entry_type, 0) + 1
        if total_chunks > 1:
            log(f"🧮 Glossary refinement will process {total_chunks} total chunk(s) across selected entry types.")

        if send_all_types:
            update_refinement_progress(progress_file, broad_type_key, {
                "entry_type": "all selected entry types",
                "status": "in_progress",
                "input_hash": broad_input_hash,
                "chunking_mode": canonical_mode,
                "payload_delimiter": payload_delimiter_name,
                "entry_count_before": len(entries),
                "completed_chunks": 0,
                "total_chunks": total_chunks,
                "output_file": os.path.basename(output_path or ""),
            }, atomic_replace_fn=atomic_replace_fn)

        for selected_type in group_selected_types:
            type_entries = entries_by_type.get(selected_type) or []
            if not type_entries:
                continue
            update_refinement_progress(progress_file, type_keys[selected_type], {
                "entry_type": selected_type,
                "status": "in_progress",
                "input_hash": type_hashes.get(selected_type) or _entry_hash(selected_type, type_entries, hash_mode),
                "chunking_mode": canonical_mode,
                "payload_delimiter": payload_delimiter_name,
                "entry_count_before": len(type_entries),
                "completed_chunks": 0,
                "total_chunks": per_type_total_chunks.get(selected_type, 0),
                "output_file": os.path.basename(output_path or ""),
            }, atomic_replace_fn=atomic_replace_fn)

        def _process_chunk(chunk_info):
            chunk_text, chunk_idx, total_chunks, chunk_entry_type, whole_type_chunk = chunk_info
            if check_stop():
                return {"status": "stopped", "chunk_idx": chunk_idx, "total_chunks": total_chunks, "entry_type": chunk_entry_type}

            if send_all_types:
                log(f"✨ Refining selected glossary entries ({chunk_idx}/{total_chunks})...")
            elif whole_type_chunk:
                log(f"✨ Refining {chunk_entry_type} entries ({chunk_idx}/{total_chunks})...")
            else:
                log(f"✨ Refining glossary chunks ({chunk_idx}/{total_chunks})...")
            msgs = _build_messages(system_prompt, user_prompt, chunk_entry_type, chunk_text, payload_columns, chunk_idx, total_chunks, selected_types)
            msgs = _sanitize_messages_for_api(msgs, chunk_text)
            context_label = "glossary_refinement"
            try:
                raw, finish_reason, _raw_obj = _call_send(
                    send_fn,
                    msgs,
                    client,
                    temp,
                    mtoks,
                    check_stop,
                    chunk_timeout,
                    chunk_idx,
                    total_chunks,
                    context_label,
                )
            except Exception as e:
                return {
                    "status": "failed",
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "entry_type": chunk_entry_type,
                    "error": str(e),
                    "model_name": _actual_request_model_name(client),
                    "request_context": _actual_request_key_context(client),
                }

            model_name = _actual_request_model_name(client)
            request_context = _actual_request_key_context(client)
            response_text = raw[0] if isinstance(raw, tuple) else raw
            response_text = response_text if isinstance(response_text, str) else str(response_text or "")
            parsed = parse_response_fn(response_text)
            parsed = [
                entry for entry in parsed
                if isinstance(entry, dict) and str(entry.get("type", "")).strip().lower() in allowed_types_lc
            ]
            parsed = _strip_inactive_description(parsed)
            if not parsed:
                return {
                    "status": "failed",
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "entry_type": chunk_entry_type,
                    "error": "empty_or_invalid_response",
                    "model_name": model_name,
                    "request_context": request_context,
                }

            if _issue_from_finish_reason(finish_reason, None) == "TRUNCATED":
                return {
                    "status": "failed",
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "entry_type": chunk_entry_type,
                    "error": "TRUNCATED",
                    "model_name": model_name,
                    "request_context": request_context,
                }

            return {
                "status": "ok",
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "entry_type": chunk_entry_type,
                "entries": parsed,
                "model_name": model_name,
                "request_context": request_context,
            }

        def _result_model_update(result):
            model_update = dict(result.get("request_context") or {})
            model_name = result.get("model_name")
            if model_name:
                model_update["model_name"] = model_name
            return model_update

        def _record_failed_chunk(result):
            chunk_idx = result.get("chunk_idx", "?")
            result_entry_type = result.get("entry_type") or entry_type
            error = result.get("error") or "unknown_error"
            if error == "empty_or_invalid_response":
                log(f"⚠️ Refinement returned no valid entries for chunk {chunk_idx}; keeping original selected entries.")
            elif error == "TRUNCATED":
                log(f"Refinement chunk {chunk_idx} was truncated; keeping original selected entries.")
            else:
                log(f"Refinement failed for chunk {chunk_idx}: {error}")
            if send_all_types:
                broad_failed_update = {
                    "entry_type": "all selected entry types",
                    "status": "failed",
                    "error": error,
                    "completed_chunks": max(completed_by_type.values(), default=0),
                    "total_chunks": len(chunks),
                }
                broad_failed_update.update(_result_model_update(result))
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    broad_failed_update,
                    atomic_replace_fn=atomic_replace_fn,
                )
                failed_update = {
                    "status": "failed",
                    "error": error,
                }
                failed_update.update(_result_model_update(result))
                for selected_type in group_selected_types:
                    typed_failed_update = dict(failed_update)
                    typed_failed_update.update({
                        "entry_type": selected_type,
                        "completed_chunks": completed_by_type.get(selected_type, 0),
                        "total_chunks": per_type_total_chunks.get(selected_type, result.get("total_chunks")),
                    })
                    update_refinement_progress(
                        progress_file,
                        type_keys[selected_type],
                        typed_failed_update,
                        atomic_replace_fn=atomic_replace_fn,
                    )
                return
            failed_update = {
                "entry_type": result_entry_type,
                "status": "failed",
                "error": error,
                "completed_chunks": 0,
                "total_chunks": per_type_total_chunks.get(result_entry_type, result.get("total_chunks")),
            }
            failed_update.update(_result_model_update(result))
            update_refinement_progress(
                progress_file,
                type_keys.get(result_entry_type, f"type::{result_entry_type}"),
                failed_update,
                atomic_replace_fn=atomic_replace_fn,
            )

        refined_entries = []
        last_model_name = ""
        last_request_context = {}

        def _remember_success(result):
            nonlocal last_model_name, last_request_context
            if result.get("model_name"):
                last_model_name = result.get("model_name")
            if result.get("request_context"):
                last_request_context = dict(result.get("request_context") or {})

        completed_by_type = {selected_type: 0 for selected_type in group_selected_types}

        def _mark_type_chunk_success(result):
            if send_all_types:
                broad_completed_chunks = max(completed_by_type.values(), default=0) + 1
                broad_update = {
                    "entry_type": "all selected entry types",
                    "status": "in_progress",
                    "completed_chunks": broad_completed_chunks,
                    "total_chunks": len(chunks),
                }
                broad_update.update(_result_model_update(result))
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    broad_update,
                    atomic_replace_fn=atomic_replace_fn,
                )
                for selected_type in group_selected_types:
                    completed_by_type[selected_type] = completed_by_type.get(selected_type, 0) + 1
                    chunk_update = {
                        "entry_type": selected_type,
                        "status": "in_progress",
                        "completed_chunks": completed_by_type[selected_type],
                        "total_chunks": per_type_total_chunks.get(selected_type, result.get("total_chunks")),
                    }
                    chunk_update.update(_result_model_update(result))
                    update_refinement_progress(
                        progress_file,
                        type_keys[selected_type],
                        chunk_update,
                        atomic_replace_fn=atomic_replace_fn,
                    )
                return
            result_entry_type = result.get("entry_type") or entry_type
            completed_by_type[result_entry_type] = completed_by_type.get(result_entry_type, 0) + 1
            chunk_update = {
                "entry_type": result_entry_type,
                "status": "in_progress",
                "completed_chunks": completed_by_type[result_entry_type],
                "total_chunks": per_type_total_chunks.get(result_entry_type, result.get("total_chunks")),
            }
            chunk_update.update(_result_model_update(result))
            update_refinement_progress(
                progress_file,
                type_keys.get(result_entry_type, f"type::{result_entry_type}"),
                chunk_update,
                atomic_replace_fn=atomic_replace_fn,
            )

        def _mark_all_pending_stopped():
            if send_all_types:
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    {
                        "entry_type": "all selected entry types",
                        "status": "in_progress",
                        "completed_chunks": max(completed_by_type.values(), default=0),
                        "total_chunks": len(chunks),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in group_selected_types:
                update_refinement_progress(
                    progress_file,
                    type_keys[selected_type],
                    {
                        "entry_type": selected_type,
                        "status": "in_progress",
                        "completed_chunks": completed_by_type.get(selected_type, 0),
                        "total_chunks": per_type_total_chunks.get(selected_type, 0),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )

        def _mark_remaining_pending_failed(error, skip_entry_type=None):
            if send_all_types:
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    {
                        "entry_type": "all selected entry types",
                        "status": "failed",
                        "error": error,
                        "completed_chunks": max(completed_by_type.values(), default=0),
                        "total_chunks": len(chunks),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in group_selected_types:
                if skip_entry_type and selected_type == skip_entry_type:
                    continue
                update_refinement_progress(
                    progress_file,
                    type_keys[selected_type],
                    {
                        "entry_type": selected_type,
                        "status": "failed",
                        "error": error,
                        "completed_chunks": completed_by_type.get(selected_type, 0),
                        "total_chunks": per_type_total_chunks.get(selected_type, 0),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )

        chunk_batch_enabled = _batch_translation_enabled() and len(chunks) > 1
        if chunk_batch_enabled:
            max_workers = min(_batch_size(), len(chunks))
            log(f"Glossary refinement batch mode: {max_workers} parallel chunk request(s)")
            completed_chunks = 0
            chunk_results = {}
            failed_result = None
            stopped_result = None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_chunk, chunk): chunk for chunk in chunks}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        chunk = futures.get(future)
                        result = {
                            "status": "failed",
                            "chunk_idx": chunk[1] if chunk else "?",
                            "total_chunks": chunk[2] if chunk else len(chunks),
                            "entry_type": chunk[3] if chunk else entry_type,
                            "error": str(e),
                        }
                    status = result.get("status")
                    if status == "stopped":
                        stopped_result = result
                        for pending in futures:
                            pending.cancel()
                        break
                    if status != "ok":
                        failed_result = result
                        for pending in futures:
                            pending.cancel()
                        break

                    chunk_results[result["chunk_idx"]] = result
                    completed_chunks += 1
                    _remember_success(result)
                    _mark_type_chunk_success(result)

            if stopped_result:
                completed_chunks = len(chunk_results)
                log(f"Glossary refinement stopped during chunk {stopped_result.get('chunk_idx')}/{stopped_result.get('total_chunks')}")
                _mark_all_pending_stopped()
                return "stopped", entry_type, {}

            if failed_result:
                _record_failed_chunk(failed_result)
                if not send_all_types:
                    _mark_remaining_pending_failed(
                        "refinement_aborted_after_chunk_failure",
                        failed_result.get("entry_type"),
                    )
                refined_entries = []
            else:
                for chunk_idx in sorted(chunk_results):
                    refined_entries.extend(chunk_results[chunk_idx].get("entries") or [])
        else:
            completed_chunks = 0
            for chunk in chunks:
                result = _process_chunk(chunk)
                status = result.get("status")
                if status == "stopped":
                    log(f"Glossary refinement stopped during chunk {result.get('chunk_idx')}/{result.get('total_chunks')}")
                    _mark_all_pending_stopped()
                    return "stopped", entry_type, {}
                if status != "ok":
                    _record_failed_chunk(result)
                    if not send_all_types:
                        _mark_remaining_pending_failed(
                            "refinement_aborted_after_chunk_failure",
                            result.get("entry_type"),
                        )
                    refined_entries = []
                    break

                refined_entries.extend(result.get("entries") or [])
                completed_chunks += 1
                _remember_success(result)
                _mark_type_chunk_success(result)

        if refined_entries:
            if not skip_dedupe:
                refined_entries = dedupe_fn(refined_entries)
            if send_all_types:
                result_mapping = {}
                for selected_type in group_selected_types:
                    typed_refined = [
                        e for e in refined_entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
                    result_mapping[selected_type] = typed_refined or [
                        e for e in entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
            else:
                result_mapping = {entry_type: refined_entries}
            model_name = last_model_name or _actual_request_model_name(client)
            request_update = dict(last_request_context or _actual_request_key_context(client))
            if model_name:
                request_update["model_name"] = model_name
            if send_all_types:
                broad_completed_update = {
                    "entry_type": "all selected entry types",
                    "status": "completed",
                    "input_hash": broad_input_hash,
                    "output_hash": _entry_hash("all selected entry types", refined_entries, hash_mode),
                    "entry_count_before": len(entries),
                    "entry_count_after": len(refined_entries),
                    "completed_chunks": total_chunks,
                    "total_chunks": total_chunks,
                    "chunking_mode": canonical_mode,
                    "payload_delimiter": payload_delimiter_name,
                    "output_file": os.path.basename(output_path or ""),
                }
                broad_completed_update.update(request_update)
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    broad_completed_update,
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in group_selected_types:
                original_type_entries = entries_by_type.get(selected_type) or []
                refined_type_entries = result_mapping.get(selected_type, [])
                completed_update = {
                    "entry_type": selected_type,
                    "status": "completed",
                    "input_hash": type_hashes.get(selected_type) or _entry_hash(selected_type, original_type_entries, hash_mode),
                    "output_hash": _entry_hash(selected_type, refined_type_entries, hash_mode),
                    "entry_count_before": len(original_type_entries),
                    "entry_count_after": len(refined_type_entries),
                    "completed_chunks": per_type_total_chunks.get(selected_type, 0),
                    "total_chunks": per_type_total_chunks.get(selected_type, 0),
                    "chunking_mode": canonical_mode,
                    "payload_delimiter": payload_delimiter_name,
                    "output_file": os.path.basename(output_path or ""),
                }
                completed_update.update(request_update)
                update_refinement_progress(
                    progress_file,
                    type_keys[selected_type],
                    completed_update,
                    atomic_replace_fn=atomic_replace_fn,
                )
            log(f"✅ Refined selected entries: {len(entries)} -> {len(refined_entries)} entries")
            return "ok", entry_type, result_mapping

        return "failed", entry_type, _original_mapping_for_group(entry_type, entries)

    work_groups = [group for group in groups if group[1]]
    stopped = False
    batch_enabled = _batch_translation_enabled() and not send_all_types and len(work_groups) > 1
    if batch_enabled:
        max_workers = min(_batch_size(), len(work_groups))
        log(f"🚀 Glossary refinement batch mode: {max_workers} parallel request(s)")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_group, group): group[0] for group in work_groups}
            for future in as_completed(futures):
                try:
                    status, entry_type, result_mapping = future.result()
                except Exception as e:
                    entry_type = futures.get(future, "entry type")
                    log(f"Refinement failed for {entry_type}: {e}")
                    status, result_mapping = "failed", {}
                if result_mapping:
                    refined_by_type.update(result_mapping)
                if status == "stopped":
                    stopped = True
                    for pending in futures:
                        pending.cancel()
                    break
    else:
        for group in work_groups:
            status, _entry_type, result_mapping = _process_group(group)
            if result_mapping:
                refined_by_type.update(result_mapping)
            if status == "stopped":
                stopped = True
                break

    if stopped:
        return glossary

    if not refined_by_type:
        return glossary

    selected_lc = {t.lower() for t in refined_by_type}
    rebuilt = [entry for entry in glossary if str(entry.get("type", "")).strip().lower() not in selected_lc]
    for entry_type in selected_types:
        rebuilt.extend(refined_by_type.get(entry_type, []))
    rebuilt = _strip_inactive_description(rebuilt)
    if not skip_dedupe:
        rebuilt = dedupe_fn(rebuilt)
        rebuilt = _strip_inactive_description(rebuilt)
    return rebuilt
