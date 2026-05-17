# -*- coding: utf-8 -*-
"""Shared optional glossary refinement step.

This module owns the refinement prompt/config/progress behavior. Callers keep
their own glossary loading/saving formats and pass in their parser, deduper, and
API sender so the balanced/full and minimal paths stay in sync without becoming
coupled to each other's implementation details.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup


_progress_lock = threading.Lock()


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


def _entry_columns(entries: List[Dict]) -> List[str]:
    columns = ["type", "raw_name", "translated_name"]
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for key in entry.keys():
            if key not in columns:
                columns.append(key)
    return columns


def _entry_payload(entries: List[Dict]) -> str:
    sep = "\x1F"
    columns = _entry_columns(entries)
    lines = [sep.join(columns)]
    for entry in entries:
        lines.append(sep.join(str(entry.get(col, "")) for col in columns))
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


def _build_messages(system_prompt: str, user_prompt: str, entry_type: str, chunk_text: str, chunk_idx=None, total_chunks=None) -> List[Dict]:
    messages = []
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


def _call_send(send_fn, messages, client, temp, mtoks, check_stop, chunk_timeout, chunk_idx, total_chunks, context_label):
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

    custom_types = custom_entry_types_fn()
    active_types = [t for t, cfg in custom_types.items() if not isinstance(cfg, dict) or cfg.get("enabled", True)]
    selected_types = selected_refinement_types(active_types)
    if not selected_types:
        log("Glossary refinement enabled, but no active/selected entry types matched; skipping.")
        return glossary

    system_prompt = os.getenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", "")
    user_prompt = os.getenv("GLOSSARY_REFINEMENT_USER_PROMPT", "")
    chunking_mode = os.getenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "separate").strip().lower()
    send_all_types = chunking_mode in ("all", "all_types", "all_in_one")
    canonical_mode = "all" if send_all_types else "separate"
    skip_dedupe = os.getenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "0").strip().lower() in ("1", "true", "yes", "on")

    log(f"\nGlossary refinement enabled for: {', '.join(selected_types)}")
    refined_by_type = {}
    progress = load_refinement_progress(progress_file)
    selected_lc = {t.lower() for t in selected_types}

    if send_all_types:
        all_selected_entries = [
            dict(e) for e in glossary
            if str(e.get("type", "")).strip().lower() in selected_lc
        ]
        groups = [("all selected entry types", all_selected_entries, f"all::{','.join(selected_types)}", selected_lc)]
    else:
        groups = []
        for entry_type in selected_types:
            entries = [dict(e) for e in glossary if str(e.get("type", "")).strip().lower() == entry_type.lower()]
            groups.append((entry_type, entries, f"type::{entry_type}", {entry_type.lower()}))

    for entry_type, entries, type_key, allowed_types_lc in groups:
        if check_stop():
            log("Glossary refinement stopped before completion")
            break
        if not entries:
            continue

        type_hash = _entry_hash(entry_type, entries, canonical_mode)
        type_progress = progress.get(type_key, {})
        if (
            isinstance(type_progress, dict)
            and type_progress.get("status") == "completed"
            and type_hash in (type_progress.get("input_hash"), type_progress.get("output_hash"))
        ):
            log(f"Refinement already completed for {entry_type}; skipping.")
            if send_all_types:
                for selected_type in selected_types:
                    refined_by_type[selected_type] = [
                        e for e in entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
            else:
                refined_by_type[entry_type] = entries
            continue

        payload = _entry_payload(entries)
        chunks = [(payload, 1, 1)]
        try:
            token_count = chapter_splitter.count_tokens(payload)
        except Exception:
            token_count = len(payload) // 3
        if not send_all_types and token_count > available_tokens:
            wrapped = f"<html><body><p>{payload.replace(chr(10), '</p><p>')}</p></body></html>"
            chunks = [
                (BeautifulSoup(chunk_html, "html.parser").get_text("\n", strip=True), chunk_idx, total_chunks)
                for chunk_html, chunk_idx, total_chunks in chapter_splitter.split_chapter(
                    wrapped,
                    available_tokens,
                    filename="glossary_refinement.txt",
                )
            ]

        update_refinement_progress(progress_file, type_key, {
            "entry_type": entry_type,
            "status": "in_progress",
            "input_hash": type_hash,
            "chunking_mode": canonical_mode,
            "total_chunks": len(chunks),
            "output_file": os.path.basename(output_path or ""),
        }, atomic_replace_fn=atomic_replace_fn)

        refined_entries = []
        for chunk_text, chunk_idx, total_chunks in chunks:
            if check_stop():
                log(f"Glossary refinement stopped during {entry_type} chunk {chunk_idx}/{total_chunks}")
                update_refinement_progress(
                    progress_file,
                    type_key,
                    {"status": "in_progress", "completed_chunks": chunk_idx - 1},
                    atomic_replace_fn=atomic_replace_fn,
                )
                return glossary

            log(f"Refining {entry_type} entries ({chunk_idx}/{total_chunks})...")
            msgs = _build_messages(system_prompt, user_prompt, entry_type, chunk_text, chunk_idx, total_chunks)
            msgs = _sanitize_messages_for_api(msgs, chunk_text)
            context_label = f"glossary refinement ({entry_type} {chunk_idx}/{total_chunks})"
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
                log(f"Refinement failed for {entry_type} chunk {chunk_idx}: {e}")
                update_refinement_progress(
                    progress_file,
                    type_key,
                    {"status": "failed", "error": str(e)},
                    atomic_replace_fn=atomic_replace_fn,
                )
                refined_entries = []
                break

            response_text = raw[0] if isinstance(raw, tuple) else raw
            response_text = response_text if isinstance(response_text, str) else str(response_text or "")
            parsed = parse_response_fn(response_text)
            parsed = [
                entry for entry in parsed
                if isinstance(entry, dict) and str(entry.get("type", "")).strip().lower() in allowed_types_lc
            ]
            if not parsed:
                log(f"Refinement returned no valid {entry_type} entries for chunk {chunk_idx}; keeping original entries for this type.")
                update_refinement_progress(
                    progress_file,
                    type_key,
                    {"status": "failed", "error": "empty_or_invalid_response"},
                    atomic_replace_fn=atomic_replace_fn,
                )
                refined_entries = []
                break

            refined_entries.extend(parsed)
            update_refinement_progress(
                progress_file,
                type_key,
                {"status": "in_progress", "completed_chunks": chunk_idx},
                atomic_replace_fn=atomic_replace_fn,
            )
            if _issue_from_finish_reason(finish_reason, None) == "TRUNCATED":
                log(f"Refinement for {entry_type} chunk {chunk_idx} was truncated; keeping original entries for this type.")
                update_refinement_progress(
                    progress_file,
                    type_key,
                    {"status": "failed", "error": "TRUNCATED"},
                    atomic_replace_fn=atomic_replace_fn,
                )
                refined_entries = []
                break

        if refined_entries:
            if not skip_dedupe:
                refined_entries = dedupe_fn(refined_entries)
            if send_all_types:
                for selected_type in selected_types:
                    typed_refined = [
                        e for e in refined_entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
                    refined_by_type[selected_type] = typed_refined or [
                        e for e in entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
            else:
                refined_by_type[entry_type] = refined_entries
            update_refinement_progress(progress_file, type_key, {
                "status": "completed",
                "input_hash": type_hash,
                "output_hash": _entry_hash(entry_type, refined_entries, canonical_mode),
                "entry_count_before": len(entries),
                "entry_count_after": len(refined_entries),
                "completed_chunks": len(chunks),
            }, atomic_replace_fn=atomic_replace_fn)
            log(f"Refined {entry_type}: {len(entries)} -> {len(refined_entries)} entries")
        else:
            if send_all_types:
                for selected_type in selected_types:
                    refined_by_type[selected_type] = [
                        e for e in entries
                        if str(e.get("type", "")).strip().lower() == selected_type.lower()
                    ]
            else:
                refined_by_type[entry_type] = entries

    if not refined_by_type:
        return glossary

    selected_lc = {t.lower() for t in refined_by_type}
    rebuilt = [entry for entry in glossary if str(entry.get("type", "")).strip().lower() not in selected_lc]
    for entry_type in selected_types:
        rebuilt.extend(refined_by_type.get(entry_type, []))
    if not skip_dedupe:
        rebuilt = dedupe_fn(rebuilt)
    return rebuilt
