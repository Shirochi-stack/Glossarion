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
import unicodedata
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Dict, Iterable, List, Optional

DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT = """You are refining an already extracted translation glossary.

Your job is cleanup, not broad re-extraction. Preserve useful entries and return only the refined glossary entries for the provided entry type or entry types.

Glossary schema:
{fields}

Active refinement entry types:
{entries}

Critical refinement rules:
- Keep the existing glossary schema and fields. Return refined glossary CSV data rows only, using the columns and delimiter shown in the glossary schema above. Do not include a header row.
- Keep the exact same column order, do not rearrange it.
- Remove duplicate entries, near-duplicates, and entries that only differ by trivial spacing, casing, honorifics, or punctuation.
- Remove generic or unnecessary entries that are not useful for translation consistency.
- Strictly follow a Character & Surname Tracking and Resolution process. If a Surnames entry type exists, check EVERY character against the surname entries. Actively cross-reference raw_name, translated_name, descriptions, family membership, and relationship clues to resolve each character's surname; a pre-existing full name is NOT required when the provided glossary's evidence establishes the connection. For EVERY resolved match, you MUST combine surname and given name into ONE canonical full-name character entry, replacing the given-name-only entry and merging duplicate character entries for that person. Leaving a resolved character with only a given name is an error. Preserve shared surname entries; they do NOT replace the required full-name character entry. Never append a surname twice or include titles, nicknames, or aliases. Use the source-language full name in raw_name and its target-language equivalent in translated_name, respecting each language's name order. Keep unresolved names unchanged; never guess a surname. Splitting a character's full name into separate given-name and surname entries is allowed ONLY when no Surnames type exists. Before returning the glossary, verify that no resolved character remains in a given-name-only entry.
- Reject useless entries where raw_name and translated_name are essentially the same word or duplicate text.
- Do not invent entries, translations, genders, descriptions, aliases, or facts that are not present in the provided glossary content.
- If two entries conflict, keep the more specific and translation-useful one.
- Keep active custom entry types separate; do not move entries into another type unless the current entry type is plainly wrong.

Return only the refined glossary content. Do not include markdown, explanations, comments, or surrounding prose."""

DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT = ""
DEFAULT_GLOSSARY_REFINEMENT_CHUNKING_MODE = "all"

_progress_lock = threading.Lock()
_SCHEMA_PLACEHOLDERS = ("{fields1}", "{{fields1}}", "{fields}", "{{fields}}", "{columns}", "{{columns}}")


def refinement_enabled() -> bool:
    return os.getenv("GLOSSARY_REFINEMENT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def refinement_waits_for_completion() -> bool:
    """Whether automatic refinement must wait for complete extraction progress."""
    return os.getenv("GLOSSARY_REFINEMENT_WAIT_FOR_COMPLETION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def refinement_reopens_on_source_change() -> bool:
    """Whether source-name changes can reopen completed refinement types."""
    return os.getenv("GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def selected_refinement_types(active_types: Iterable[str]) -> List[str]:
    active = [str(t).strip() for t in active_types if str(t).strip()]
    mode = os.getenv("GLOSSARY_REFINEMENT_TYPE_MODE", "all").strip().lower()
    if mode != "selected":
        return active
    raw = os.getenv("GLOSSARY_REFINEMENT_SELECTED_TYPES", "")
    selected = [t.strip() for t in raw.split(",") if t.strip()]
    selected_lc = {_refinement_type_key(t) for t in selected}
    return [t for t in active if _refinement_type_key(t) in selected_lc]


def refinement_chunking_mode() -> str:
    """Return the canonical runtime request mode without mutating settings.

    This fallback is worker-only.  GUI controls must be initialized from an
    explicit saved value instead of treating this fallback as a user choice.
    """
    raw_mode = os.getenv(
        "GLOSSARY_REFINEMENT_CHUNKING_MODE",
        DEFAULT_GLOSSARY_REFINEMENT_CHUNKING_MODE,
    ).strip().lower()
    if raw_mode in ("all", "all_types", "all_in_one", "all_entries", "combined"):
        return "all"
    return "separate"


@dataclass
class RefinementRunOptions:
    """One-run overrides for automatic or user-triggered refinement."""

    selected_types: Optional[List[str]] = None
    chunking_mode: Optional[str] = None
    force: bool = False
    run_when_disabled: bool = False
    target_chunk_count: Optional[int] = None


@dataclass
class RefinementPlannedChunk:
    """A serialized request payload whose boundaries are whole glossary rows."""

    payload: str
    entry_type: str
    selected_types: List[str]
    columns: List[str]
    token_count: int
    whole_type_chunk: bool = False


@dataclass
class RefinementPlan:
    """Pure preview/execution contract for a glossary refinement run."""

    selected_types: List[str]
    chunking_mode: str
    chunks: List[RefinementPlannedChunk] = dataclass_field(default_factory=list)
    per_type_counts: Dict[str, int] = dataclass_field(default_factory=dict)
    per_type_tokens: Dict[str, int] = dataclass_field(default_factory=dict)
    total_payload_tokens: int = 0
    per_chunk_token_estimates: List[int] = dataclass_field(default_factory=list)
    available_tokens: int = 0
    requested_chunk_count: Optional[int] = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


def _canonical_refinement_mode(value: Optional[str]) -> str:
    raw_mode = str(value or "").strip().lower()
    if raw_mode in ("all", "all_types", "all_in_one", "all_entries", "combined"):
        return "all"
    return "separate"


def _refinement_type_key(value: str) -> str:
    """Normalize configured types and plural section-heading names."""
    key = str(value or "").strip().casefold()
    if key in ("term", "terms"):
        return "terms"
    if len(key) > 3 and key.endswith("ies"):
        return key[:-3] + "y"
    if key.endswith(("sses", "xes", "ches", "shes", "zes")):
        return key[:-2]
    if len(key) > 1 and key.endswith("s") and not key.endswith(("ss", "us", "is")):
        return key[:-1]
    return key


def _count_with_splitter(chapter_splitter, text: str) -> int:
    try:
        return max(0, int(chapter_splitter.count_tokens(text)))
    except Exception:
        return max(0, len(str(text or "")) // 3)


def _partition_entries_for_budget(
    entries: List[Dict],
    columns: List[str],
    delimiter: str,
    available_tokens: int,
    chapter_splitter,
) -> List[List[Dict]]:
    """Greedily split on row boundaries while respecting the token budget."""
    if not entries:
        return []
    budget = max(1, int(available_tokens or 1))
    partitions: List[List[Dict]] = []
    current: List[Dict] = []
    current_tokens = 0
    for entry in entries:
        row_payload = _entry_payload([entry], columns, delimiter)
        row_tokens = max(1, _count_with_splitter(chapter_splitter, row_payload))
        separator_tokens = 1 if current else 0
        if current and current_tokens + separator_tokens + row_tokens > budget:
            partitions.append(current)
            current = [entry]
            current_tokens = row_tokens
        else:
            current.append(entry)
            current_tokens += separator_tokens + row_tokens
    if current:
        partitions.append(current)
    return partitions


def _partition_prioritized_types(
    ordered_types, entries_by_type, columns, delimiter, available_tokens,
    chapter_splitter, custom_entry_types,
):
    """Pack whole types by priority; split a type only when it cannot fit alone."""
    budget = max(1, int(available_tokens or 1))
    if custom_entry_types is None:
        try:
            custom_entry_types = json.loads(os.getenv("GLOSSARY_CUSTOM_ENTRY_TYPES", "{}"))
        except (TypeError, ValueError):
            custom_entry_types = {}
    if not isinstance(custom_entry_types, dict) or not custom_entry_types:
        custom_entry_types = {
            name: {'enabled': True, 'has_gender': True}
            for name in ('character', 'titles', 'nicknames')
        }
    metadata = {
        _refinement_type_key(name): cfg
        for name, cfg in custom_entry_types.items() if isinstance(cfg, dict)
    }

    def priority(entry_type):
        key = _refinement_type_key(entry_type)
        if key == 'character':
            return 0
        if key == 'surname':
            return 1
        cfg = metadata.get(key, {})
        return 2 if cfg.get('enabled', True) and cfg.get('has_gender', False) else 3

    def tokens(entries):
        return _count_with_splitter(chapter_splitter, _entry_payload(entries, columns, delimiter))

    def fit_rows(entries):
        # Validate serialized payloads too: row token estimates are not additive
        # for every tokenizer, especially around CSV quoting and delimiters.
        if tokens(entries) <= budget:
            return [entries]
        if len(entries) == 1:
            raise ValueError(
                f"A {entries[0].get('type', 'glossary')} entry exceeds the refinement "
                f"chunk budget of {budget:,} tokens. Increase the budget or shorten the entry."
            )
        middle = len(entries) // 2
        return fit_rows(entries[:middle]) + fit_rows(entries[middle:])

    partitions = []
    for entry_type in sorted(ordered_types, key=priority):
        entries = entries_by_type[entry_type]
        if not entries:
            continue
        if tokens(entries) <= budget:
            groups = [entries]
        else:
            groups = [
                group
                for part in _partition_entries_for_budget(
                    entries, columns, delimiter, budget, chapter_splitter
                )
                for group in fit_rows(part)
            ]
        for group in groups:
            # First fit lets a smaller gendered type join characters/surnames
            # even when an earlier, larger gendered type needs its own request.
            for partition in partitions:
                if tokens(partition + group) <= budget:
                    partition.extend(group)
                    break
            else:
                partitions.append(list(group))
    return partitions


def _partition_entries_exact(
    entries: List[Dict],
    chunk_count: int,
    columns: List[str],
    delimiter: str,
    chapter_splitter,
) -> List[List[Dict]]:
    """Token-balance ordered entries into exactly ``chunk_count`` non-empty parts."""
    if not entries:
        return []
    requested = max(1, min(int(chunk_count or 1), len(entries)))
    if requested == 1:
        return [list(entries)]
    if requested == len(entries):
        return [[entry] for entry in entries]

    row_tokens = [
        max(1, _count_with_splitter(
            chapter_splitter,
            _entry_payload([entry], columns, delimiter),
        ))
        for entry in entries
    ]
    result: List[List[Dict]] = []
    cursor = 0
    remaining_tokens = sum(row_tokens)
    for part_index in range(requested):
        remaining_parts = requested - part_index
        remaining_entries = len(entries) - cursor
        if remaining_parts == 1:
            result.append(list(entries[cursor:]))
            break
        max_take = remaining_entries - (remaining_parts - 1)
        target = remaining_tokens / float(remaining_parts)
        taken = 0
        part_tokens = 0
        while taken < max_take:
            next_tokens = row_tokens[cursor + taken]
            if taken and abs(part_tokens - target) <= abs((part_tokens + next_tokens) - target):
                break
            part_tokens += next_tokens
            taken += 1
        if taken <= 0:
            taken = 1
            part_tokens = row_tokens[cursor]
        result.append(list(entries[cursor:cursor + taken]))
        cursor += taken
        remaining_tokens -= part_tokens
    return result


def _separate_exact_allocations(
    selected_types: List[str],
    entries_by_type: Dict[str, List[Dict]],
    per_type_tokens: Dict[str, int],
    target_chunk_count: int,
) -> Dict[str, int]:
    """Allocate an exact total proportionally, with one chunk per non-empty type."""
    non_empty = [entry_type for entry_type in selected_types if entries_by_type.get(entry_type)]
    if not non_empty:
        return {}
    minimum = len(non_empty)
    maximum = sum(len(entries_by_type[entry_type]) for entry_type in non_empty)
    target = max(minimum, min(int(target_chunk_count or minimum), maximum))
    allocations = {entry_type: 1 for entry_type in non_empty}
    remaining = target - minimum
    capacities = {
        entry_type: max(0, len(entries_by_type[entry_type]) - 1)
        for entry_type in non_empty
    }
    while remaining > 0:
        candidates = [entry_type for entry_type in non_empty if capacities[entry_type] > 0]
        if not candidates:
            break
        total_weight = sum(max(1, per_type_tokens.get(entry_type, 0)) for entry_type in candidates)
        ideal = {
            entry_type: remaining * max(1, per_type_tokens.get(entry_type, 0)) / float(total_weight or 1)
            for entry_type in candidates
        }
        # Assign at least one at a time so capacity caps are naturally respected.
        chosen = max(
            candidates,
            key=lambda entry_type: (
                ideal[entry_type] / max(1, allocations[entry_type]),
                per_type_tokens.get(entry_type, 0),
                -selected_types.index(entry_type),
            ),
        )
        allocations[chosen] += 1
        capacities[chosen] -= 1
        remaining -= 1
    return allocations


def plan_refinement(
    glossary: List[Dict],
    *,
    selected_types: Iterable[str],
    chunking_mode: str,
    chapter_splitter,
    available_tokens: int,
    target_chunk_count: Optional[int] = None,
    system_prompt: str = "",
    user_prompt: str = "",
    custom_entry_types: Optional[Dict] = None,
) -> RefinementPlan:
    """Build the exact row-safe payload plan used by preview and execution."""
    ordered_types: List[str] = []
    seen = set()
    for entry_type in selected_types or []:
        clean = str(entry_type or "").strip()
        if clean and _refinement_type_key(clean) not in seen:
            ordered_types.append(clean)
            seen.add(_refinement_type_key(clean))
    canonical_mode = _canonical_refinement_mode(chunking_mode)
    delimiter = "\x1F" if _prompt_requests_unit_separator(system_prompt, user_prompt) else ","
    entries_by_type = {
        entry_type: [
            dict(entry) for entry in glossary or []
            if isinstance(entry, dict)
            and _refinement_type_key(entry.get("type", "")) == _refinement_type_key(entry_type)
        ]
        for entry_type in ordered_types
    }
    per_type_counts = {
        entry_type: len(entries_by_type[entry_type]) for entry_type in ordered_types
    }
    per_type_tokens: Dict[str, int] = {}
    for entry_type in ordered_types:
        typed_entries = entries_by_type[entry_type]
        columns = _entry_columns(typed_entries)
        payload = _entry_payload(typed_entries, columns, delimiter)
        per_type_tokens[entry_type] = _count_with_splitter(chapter_splitter, payload) if typed_entries else 0

    chunks: List[RefinementPlannedChunk] = []
    total_payload_tokens = sum(per_type_tokens.values())
    if canonical_mode == "all":
        combined_entries = [
            entry
            for entry_type in ordered_types
            for entry in entries_by_type[entry_type]
        ]
        if combined_entries:
            columns = _entry_columns(combined_entries)
            combined_payload = _entry_payload(combined_entries, columns, delimiter)
            total_payload_tokens = _count_with_splitter(chapter_splitter, combined_payload)
            if target_chunk_count is None:
                if total_payload_tokens <= max(1, int(available_tokens or 1)):
                    partitions = [combined_entries]
                else:
                    partitions = _partition_prioritized_types(
                        ordered_types, entries_by_type, columns, delimiter,
                        available_tokens, chapter_splitter, custom_entry_types,
                    )
            else:
                partitions = _partition_entries_exact(
                    combined_entries, target_chunk_count, columns, delimiter, chapter_splitter
                )
            for partition in partitions:
                payload = _entry_payload(partition, columns, delimiter)
                chunks.append(RefinementPlannedChunk(
                    payload=payload,
                    entry_type="selected glossary entries",
                    selected_types=list(ordered_types),
                    columns=list(columns),
                    token_count=_count_with_splitter(chapter_splitter, payload),
                    whole_type_chunk=len(partitions) == 1,
                ))
    else:
        allocations = None
        if target_chunk_count is not None:
            allocations = _separate_exact_allocations(
                ordered_types, entries_by_type, per_type_tokens, target_chunk_count
            )
        for entry_type in ordered_types:
            typed_entries = entries_by_type[entry_type]
            if not typed_entries:
                continue
            columns = _entry_columns(typed_entries)
            if allocations is None:
                partitions = _partition_entries_for_budget(
                    typed_entries, columns, delimiter, available_tokens, chapter_splitter
                )
            else:
                partitions = _partition_entries_exact(
                    typed_entries,
                    allocations.get(entry_type, 1),
                    columns,
                    delimiter,
                    chapter_splitter,
                )
            for partition in partitions:
                payload = _entry_payload(partition, columns, delimiter)
                chunks.append(RefinementPlannedChunk(
                    payload=payload,
                    entry_type=entry_type,
                    selected_types=[entry_type],
                    columns=list(columns),
                    token_count=_count_with_splitter(chapter_splitter, payload),
                    whole_type_chunk=len(partitions) == 1,
                ))

    return RefinementPlan(
        selected_types=ordered_types,
        chunking_mode=canonical_mode,
        chunks=chunks,
        per_type_counts=per_type_counts,
        per_type_tokens=per_type_tokens,
        total_payload_tokens=total_payload_tokens,
        per_chunk_token_estimates=[chunk.token_count for chunk in chunks],
        available_tokens=max(0, int(available_tokens or 0)),
        requested_chunk_count=(int(target_chunk_count) if target_chunk_count is not None else None),
    )


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
    discovered = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for key in entry.keys():
            key = str(key or "").strip()
            if key and key not in discovered:
                discovered.append(key)

    def _find_column(name: str) -> Optional[str]:
        target = str(name or "").strip().lower()
        for key in discovered + custom_fields:
            if str(key or "").strip().lower() == target:
                return str(key).strip()
        return None

    # Core glossary fields always use the canonical schema order, regardless
    # of the insertion order of keys in individual entry dictionaries.
    gender_column = _find_column("gender")
    if gender_column:
        columns.append(gender_column)
    description_column = _find_column("description")
    if description_active and description_column:
        columns.append(description_column)

    standard_fields = {"type", "raw_name", "translated_name", "gender", "description"}
    # Configured custom fields define their own order after the core schema.
    for field in custom_fields:
        field = str(field or "").strip()
        if field and field.lower() not in standard_fields and field not in columns:
            columns.append(field)
    # Preserve any unexpected entry fields after the configured schema.
    for key in discovered:
        key_lower = key.lower()
        if key_lower == "description" and not description_active:
            continue
        if key_lower not in standard_fields and key not in columns:
            columns.append(key)
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


def _payload_type_counts(payload_text: str, columns: List[str], delimiter: str = ",") -> Dict[str, int]:
    type_idx = 0
    try:
        lowered = [str(col).strip().lower() for col in columns or []]
        if "type" in lowered:
            type_idx = lowered.index("type")
    except Exception:
        type_idx = 0

    counts: Dict[str, int] = {}
    if not str(payload_text or "").strip():
        return counts
    try:
        if delimiter == "\x1F":
            rows = [
                line.split(delimiter)
                for line in str(payload_text).splitlines()
                if str(line or "").strip()
            ]
        else:
            rows = csv.reader(io.StringIO(str(payload_text or "")))
        for row in rows:
            if not row or len(row) <= type_idx:
                continue
            entry_type = str(row[type_idx] or "").strip()
            if not entry_type:
                continue
            counts[entry_type] = counts.get(entry_type, 0) + 1
    except Exception:
        return counts
    return counts


def _format_type_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "no recognizable entry rows"
    return ", ".join(f"{entry_type} x{count:,}" for entry_type, count in counts.items())


def _entry_hash(entry_type: str, entries: List[Dict], chunking_mode: str) -> str:
    payload = {
        "entry_type": entry_type,
        "chunking_mode": chunking_mode,
        "entries": entries,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_IDENTITY_HASH_VERSION = "raw-name-v2"


def _entry_identity_hash(
    entry_type: str, entries: List[Dict], chunking_mode: str,
    *, version: str = _IDENTITY_HASH_VERSION,
) -> str:
    """Hash the stable identities of a refinement category.

    Full entry dictionaries are deliberately unsuitable for deciding whether a
    completed refinement must run again.  The glossary writers sort rows, drop
    internal fields, and may harmonize translated aliases after refinement.
    Those persistence-only changes must not turn completed API work back into
    pending work.  Raw/source names are the durable identities: a newly added or
    removed source term changes this hash, while a manual translation or
    description edit does not get overwritten by another refinement run.
    Request mode, delimiter, and singular/plural type aliases do not change
    those identities. The v1 form is retained only to verify older progress.
    """

    identities = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw_name = unicodedata.normalize(
            "NFC", str(entry.get("raw_name", "") or "")
        ).strip().casefold()
        if not raw_name:
            # Malformed legacy rows are uncommon, but keep them distinguishable
            # so adding/removing one still invalidates the completed category.
            raw_name = unicodedata.normalize(
                "NFC", str(entry.get("translated_name", "") or "")
            ).strip().casefold()
        identities.add(raw_name)

    payload = {
        "version": version,
        "entry_type": _refinement_type_key(unicodedata.normalize("NFC", str(entry_type or ""))),
        "raw_names": sorted(identities),
    }
    if version == "raw-name-v1":
        payload["entry_type"] = unicodedata.normalize("NFC", str(entry_type or "")).strip().casefold()
        payload["chunking_mode"] = str(chunking_mode or "").strip().casefold()
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _legacy_completed_count_matches(type_progress: Dict, entries: List[Dict], output_path: Optional[str]) -> bool:
    """Safely migrate completed rows written before identity hashes existed."""

    if not isinstance(type_progress, dict) or type_progress.get("status") != "completed":
        return False
    if type_progress.get("input_identity_hash") or type_progress.get("output_identity_hash"):
        return False
    try:
        if int(type_progress.get("entry_count_after")) != len(entries or []):
            return False
    except (TypeError, ValueError):
        return False

    recorded_output = os.path.basename(str(type_progress.get("output_file") or ""))
    requested_output = os.path.basename(str(output_path or ""))
    return not recorded_output or not requested_output or recorded_output == requested_output


def _completed_refinement_migration(entry_type, entries, type_progress, hash_mode, output_path):
    """Return metadata updates for reusable completion, or None for pending work."""
    if not isinstance(type_progress, dict) or type_progress.get("status") != "completed":
        return None
    identity_hash = _entry_identity_hash(entry_type, entries, hash_mode)
    identity_hashes = {
        str(type_progress.get(field) or "")
        for field in ("input_identity_hash", "output_identity_hash")
    } - {""}
    if type_progress.get("identity_hash_version") == _IDENTITY_HASH_VERSION:
        if identity_hash in identity_hashes:
            return {}
        # Migration may have seen only the input or only the refined output.
        # Retain the other accepted v1 shape so loading it later does not resend
        # the type. A fresh run/manual acceptance clears these legacy hashes.
        legacy_hashes = type_progress.get("legacy_identity_hashes") or []
        if legacy_hashes:
            legacy_hash = _entry_identity_hash(
                type_progress.get("legacy_identity_entry_type") or entry_type,
                entries,
                type_progress.get("legacy_identity_hash_mode") or hash_mode,
                version="raw-name-v1",
            )
            if legacy_hash in legacy_hashes:
                return {}
        return None

    migration = {
        "identity_hash_version": _IDENTITY_HASH_VERSION,
        "input_identity_hash": identity_hash,
        "output_identity_hash": identity_hash,
    }
    # The old progress manager alone wrote completed_at. Its explicit manual
    # acceptance must survive stale hashes left over from an earlier request.
    # Bind that acceptance once; subsequent source changes use the v2 check.
    if type_progress.get("completed_at") or type_progress.get("manually_marked_completed"):
        migration.update({
            "manually_marked_completed": True,
            "entry_count_before": len(entries),
            "entry_count_after": len(entries),
        })
        return migration

    saved_type = str(type_progress.get("entry_type") or entry_type)
    current_mode, current_delimiter = hash_mode.split(":", 1)
    saved_mode = _canonical_refinement_mode(type_progress.get("chunking_mode") or current_mode)
    saved_delimiter = str(type_progress.get("payload_delimiter") or current_delimiter)
    saved_hash_mode = f"{saved_mode}:{saved_delimiter}"
    if identity_hashes:
        legacy_hash = _entry_identity_hash(saved_type, entries, saved_hash_mode, version="raw-name-v1")
        if legacy_hash in identity_hashes:
            migration.update({
                "legacy_identity_hashes": sorted(identity_hashes),
                "legacy_identity_entry_type": saved_type,
                "legacy_identity_hash_mode": saved_hash_mode,
            })
            return migration
        return None

    completed_hashes = {
        str(type_progress.get(field) or "")
        for field in ("input_hash", "output_hash")
    }
    if (
        _entry_hash(saved_type, entries, saved_hash_mode) in completed_hashes
        or _legacy_completed_count_matches(type_progress, entries, output_path)
    ):
        return migration
    return None


def _find_type_refinement_progress(progress, entry_type):
    """Use the latest type state across spelling aliases; exact spelling wins ties."""
    key = f"type::{entry_type}"
    aliases = [
        (saved_key, info) for saved_key, info in progress.items()
        if str(saved_key).startswith("type::") and isinstance(info, dict)
        and _refinement_type_key(str(saved_key).split("::", 1)[1]) == _refinement_type_key(entry_type)
    ]
    if aliases:
        def updated_at(item):
            timestamps = [0]
            for field in ("last_updated", "completed_at"):
                try:
                    timestamps.append(float(item[1].get(field) or 0))
                except (TypeError, ValueError):
                    pass
            return max(timestamps), item[0] == key
        return max(aliases, key=updated_at)
    return key, {}


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
    options: Optional[RefinementRunOptions] = None,
    plan: Optional[RefinementPlan] = None,
) -> List[Dict]:
    options = options or RefinementRunOptions()
    if not refinement_enabled() and not options.run_when_disabled:
        return glossary
    if not glossary:
        log("Glossary refinement enabled, but glossary is empty; skipping.")
        return glossary
    original_glossary = glossary
    glossary = _strip_inactive_description(glossary)

    custom_types = custom_entry_types_fn()
    active_types = [t for t, cfg in custom_types.items() if not isinstance(cfg, dict) or cfg.get("enabled", True)]
    if options.selected_types is None:
        selected_types = selected_refinement_types(active_types)
    else:
        requested_lc = {
            _refinement_type_key(entry_type)
            for entry_type in options.selected_types
            if str(entry_type or "").strip()
        }
        selected_types = [
            entry_type for entry_type in active_types
            if _refinement_type_key(entry_type) in requested_lc
        ]
    if not selected_types:
        log("Glossary refinement enabled, but no active/selected entry types matched; skipping.")
        return glossary

    system_prompt = os.getenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT)
    if not str(system_prompt or "").strip() or _is_legacy_default_refinement_prompt(system_prompt):
        system_prompt = DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
    user_prompt = os.getenv("GLOSSARY_REFINEMENT_USER_PROMPT", DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT)
    canonical_mode = _canonical_refinement_mode(
        options.chunking_mode if options.chunking_mode is not None else refinement_chunking_mode()
    )
    send_all_types = canonical_mode == "all"
    skip_dedupe = os.getenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "0").strip().lower() in ("1", "true", "yes", "on")
    payload_delimiter = "\x1F" if _prompt_requests_unit_separator(system_prompt, user_prompt) else ","
    payload_delimiter_name = "unit_separator" if payload_delimiter == "\x1F" else "comma"
    hash_mode = f"{canonical_mode}:{payload_delimiter_name}"

    refined_by_type = {}
    progress = load_refinement_progress(progress_file)
    selected_lc = {_refinement_type_key(t) for t in selected_types}

    # Progress updates are merged with the previous run.  Stamp the newly
    # requested client model before the first API response so an old model_name
    # cannot remain visible while this run is still at chunks 0/N.  Successful
    # and failed request results may replace this with the provider-resolved
    # model later.
    try:
        requested_model_name = str(getattr(client, "model", "") or "").strip()
    except Exception:
        requested_model_name = ""
    if not requested_model_name:
        try:
            thread_client = (
                client._get_thread_local_client()
                if client is not None and hasattr(client, "_get_thread_local_client")
                else None
            )
            requested_model_name = str(
                getattr(thread_client, "model", "") or ""
            ).strip()
        except Exception:
            requested_model_name = ""

    def _count_payload_tokens(text: str) -> int:
        try:
            return chapter_splitter.count_tokens(text)
        except Exception:
            return len(text) // 3
    budget_label = _refinement_budget_label(available_tokens, mtoks)

    all_selected_entries = [
        dict(e) for e in glossary
        if _refinement_type_key(e.get("type", "")) in selected_lc
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
            if _refinement_type_key(e.get("type", "")) == _refinement_type_key(entry_type)
        ]
        for entry_type in selected_types
    }
    type_hashes = {
        entry_type: _entry_hash(entry_type, entries, hash_mode)
        for entry_type, entries in entries_by_type.items()
        if entries
    }
    type_identity_hashes = {
        entry_type: _entry_identity_hash(entry_type, entries, hash_mode)
        for entry_type, entries in entries_by_type.items()
    }
    pending_types = []
    completed_types = []
    empty_types = []
    reopen_on_source_change = refinement_reopens_on_source_change()

    for entry_type in selected_types:
        entries = entries_by_type.get(entry_type) or []
        type_key = type_keys[entry_type]
        type_hash = type_hashes.get(entry_type) or _entry_hash(entry_type, entries, hash_mode)
        type_identity_hash = type_identity_hashes[entry_type]
        saved_type_key, type_progress = _find_type_refinement_progress(progress, entry_type)
        if not options.force:
            if type_progress.get("status") == "completed" and not reopen_on_source_change:
                # Preserve the completion hashes so enabling this option later
                # still detects changes since the last actual refinement.
                completed_types.append(entry_type)
                continue
            migration_update = _completed_refinement_migration(
                entry_type, entries, type_progress, hash_mode, output_path,
            )
            if migration_update is not None:
                if migration_update:
                    update_refinement_progress(progress_file, saved_type_key, migration_update, atomic_replace_fn=atomic_replace_fn)
                    progress[saved_type_key] = dict(type_progress, **migration_update)
                completed_types.append(entry_type)
                continue
            if type_progress.get("status") == "completed" and entries:
                log(f"🔄 Glossary refinement reopening {entry_type}: source entries changed or saved completion could not be verified.")
        if not entries:
            no_entries_update = {
                "entry_type": entry_type,
                "status": "skipped",
                "input_hash": type_hash,
                "output_hash": type_hash,
                "identity_hash_version": _IDENTITY_HASH_VERSION,
                "input_identity_hash": type_identity_hash,
                "output_identity_hash": type_identity_hash,
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
            empty_types.append(entry_type)
            continue
        placeholder = {
            "entry_type": entry_type,
            "status": "not_refined",
            "manually_marked_completed": False,
            "completed_at": None,
            "error": None,
            "legacy_identity_hashes": None,
            "legacy_identity_entry_type": None,
            "legacy_identity_hash_mode": None,
            "model_name": requested_model_name,
            "input_hash": type_hash,
            "identity_hash_version": _IDENTITY_HASH_VERSION,
            "input_identity_hash": type_identity_hash,
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

    if completed_types:
        completed_count = sum(len(entries_by_type[t]) for t in completed_types)
        log(f"⏭️ Glossary refinement skipping completed types: {', '.join(completed_types)} ({completed_count:,} entries).")
    if empty_types:
        log(f"⏭️ Glossary refinement skipping empty types: {', '.join(empty_types)}.")
    if not pending_types:
        log("Glossary refinement already completed for selected entry types, or no entries were present; skipping.")
        return glossary

    selected_types = pending_types
    selected_lc = {_refinement_type_key(t) for t in selected_types}
    all_selected_entries = [
        e for entry_type in selected_types for e in entries_by_type.get(entry_type, [])
    ]
    scope_label = "requested types (forced rerun)" if options.force else "pending types"
    log(f"\n🧹 Glossary refinement {scope_label}: {', '.join(selected_types)} ({len(all_selected_entries):,} entries).")
    broad_input_hash = _entry_hash("all selected entry types", all_selected_entries, hash_mode)
    broad_input_identity_hash = _entry_identity_hash("all selected entry types", all_selected_entries, hash_mode)
    if send_all_types:
        broad_placeholder = {
            "entry_type": "all selected entry types",
            "status": "not_refined",
            "model_name": requested_model_name,
            "input_hash": broad_input_hash,
            "identity_hash_version": _IDENTITY_HASH_VERSION,
            "input_identity_hash": broad_input_identity_hash,
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
            (entry_type, entries_by_type.get(entry_type, []), type_keys[entry_type], {_refinement_type_key(entry_type)})
            for entry_type in selected_types
        ]

    effective_plan = plan
    plan_types_lc = {
        _refinement_type_key(entry_type)
        for entry_type in getattr(effective_plan, "selected_types", []) or []
    }
    if (
        effective_plan is None
        or plan_types_lc != {_refinement_type_key(entry_type) for entry_type in selected_types}
        or _canonical_refinement_mode(getattr(effective_plan, "chunking_mode", "")) != canonical_mode
    ):
        effective_plan = plan_refinement(
            glossary,
            selected_types=selected_types,
            chunking_mode=canonical_mode,
            chapter_splitter=chapter_splitter,
            available_tokens=available_tokens,
            target_chunk_count=options.target_chunk_count,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            custom_entry_types=custom_types,
        )

    def _original_mapping_for_group(entry_type, entries):
        if send_all_types:
            return {
                selected_type: [
                    e for e in entries
                    if _refinement_type_key(e.get("type", "")) == _refinement_type_key(selected_type)
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

        matching_plan_chunks = [
            chunk for chunk in effective_plan.chunks
            if send_all_types or _refinement_type_key(chunk.entry_type) == _refinement_type_key(entry_type)
        ]
        if matching_plan_chunks:
            payload_columns = list(matching_plan_chunks[0].columns)
            planned_chunks.extend(
                (chunk.payload, chunk.entry_type, chunk.whole_type_chunk)
                for chunk in matching_plan_chunks
            )
            token_count = sum(chunk.token_count for chunk in matching_plan_chunks)
            scope_label = "all selected entry types" if send_all_types else entry_type
            split_kind = "exact user-selected" if effective_plan.requested_chunk_count is not None else "token-budgeted"
            log(
                f"🪓 Glossary refinement planned {scope_label} as "
                f"{len(matching_plan_chunks)} {split_kind} chunk(s) "
                f"({token_count:,} tokens, {budget_label})."
            )

        if not planned_chunks:
            payload = _entry_payload(entries, payload_columns, payload_delimiter)
            planned_chunks = [(payload, entry_type, True)]
        total_chunks = len(planned_chunks)
        chunks = [
            (chunk_text, chunk_idx, total_chunks, chunk_entry_type, whole_type_chunk)
            for chunk_idx, (chunk_text, chunk_entry_type, whole_type_chunk) in enumerate(planned_chunks, 1)
        ]
        if send_all_types:
            for chunk_text, chunk_idx, _total_chunks, _chunk_entry_type, _whole_type_chunk in chunks:
                chunk_counts = _payload_type_counts(chunk_text, payload_columns, payload_delimiter)
                try:
                    chunk_tokens = _count_payload_tokens(chunk_text)
                except Exception:
                    chunk_tokens = len(str(chunk_text or "")) // 3
                log(
                    f"🧩 Glossary refinement chunk {chunk_idx}/{total_chunks} includes "
                    f"{_format_type_counts(chunk_counts)} ({chunk_tokens:,} tokens)."
                )
        # Combined requests can contain disjoint sets of types. A type only
        # depends on the chunks containing its input rows, not every request.
        chunk_types = {}
        for chunk_text, chunk_idx, _total, chunk_entry_type, _whole in chunks:
            payload_types = {
                _refinement_type_key(value)
                for value in _payload_type_counts(chunk_text, payload_columns, payload_delimiter)
            }
            chunk_types[chunk_idx] = [
                selected_type for selected_type in group_selected_types
                if _refinement_type_key(selected_type) in payload_types
            ] if send_all_types else [entry_type]
        per_type_total_chunks = {
            selected_type: sum(selected_type in members for members in chunk_types.values())
            for selected_type in group_selected_types
        }
        if total_chunks > 1:
            log(f"🧮 Glossary refinement will process {total_chunks} total chunk(s) across selected entry types.")

        if send_all_types:
            update_refinement_progress(progress_file, broad_type_key, {
                "entry_type": "all selected entry types",
                "status": "in_progress",
                "model_name": requested_model_name,
                "input_hash": broad_input_hash,
                "identity_hash_version": _IDENTITY_HASH_VERSION,
                "input_identity_hash": broad_input_identity_hash,
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
                "model_name": requested_model_name,
                "input_hash": type_hashes.get(selected_type) or _entry_hash(selected_type, type_entries, hash_mode),
                "identity_hash_version": _IDENTITY_HASH_VERSION,
                "input_identity_hash": type_identity_hashes[selected_type],
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
                cancelled = (
                    check_stop()
                    or str(getattr(e, "error_type", "")).lower() == "cancelled"
                    or any(marker in str(e).lower() for marker in (
                        "stopped by user", "cancelled by user", "canceled by user",
                    ))
                )
                return {
                    "status": "stopped" if cancelled else "failed",
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
                if isinstance(entry, dict) and _refinement_type_key(entry.get("type", "")) in allowed_types_lc
            ]
            parsed = _strip_inactive_description(parsed)
            if not parsed:
                return {
                    "status": "stopped" if check_stop() else "failed",
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "entry_type": chunk_entry_type,
                    "error": "empty_or_invalid_response",
                    "model_name": model_name,
                    "request_context": request_context,
                }

            if _issue_from_finish_reason(finish_reason, None) == "TRUNCATED":
                return {
                    "status": "stopped" if check_stop() else "failed",
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
            result_entry_type = entry_type
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
                    "completed_chunks": len(successful_results),
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
                    if selected_type in completed_type_results:
                        continue
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
                "completed_chunks": completed_by_type.get(result_entry_type, 0),
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
        successful_results = {}
        completed_type_results = {}

        def _mark_type_chunk_success(result):
            successful_results[result["chunk_idx"]] = result
            if send_all_types:
                broad_update = {
                    "entry_type": "all selected entry types",
                    "status": "in_progress",
                    "completed_chunks": len(successful_results),
                    "total_chunks": len(chunks),
                }
                broad_update.update(_result_model_update(result))
                update_refinement_progress(
                    progress_file,
                    broad_type_key,
                    broad_update,
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in chunk_types[result["chunk_idx"]]:
                completed_by_type[selected_type] += 1
                chunk_update = {
                    "entry_type": selected_type,
                    "status": "in_progress",
                    "completed_chunks": completed_by_type[selected_type],
                    "total_chunks": per_type_total_chunks[selected_type],
                }
                chunk_update.update(_result_model_update(result))
                if completed_by_type[selected_type] == per_type_total_chunks[selected_type]:
                    # Commit a type only when every chunk containing its input
                    # has succeeded. Other types may still be running.
                    typed_entries = [
                        row for index in sorted(successful_results)
                        if selected_type in chunk_types[index]
                        for row in successful_results[index].get("entries", [])
                        if _refinement_type_key(row.get("type", "")) == _refinement_type_key(selected_type)
                    ]
                    original_entries = entries_by_type[selected_type]
                    if not skip_dedupe:
                        typed_entries = dedupe_fn(typed_entries)
                    typed_entries = typed_entries or original_entries
                    completed_type_results[selected_type] = typed_entries
                    chunk_update.update({
                        "status": "completed",
                        "error": None,
                        "input_hash": type_hashes[selected_type],
                        "output_hash": _entry_hash(selected_type, typed_entries, hash_mode),
                        "identity_hash_version": _IDENTITY_HASH_VERSION,
                        "input_identity_hash": type_identity_hashes[selected_type],
                        "output_identity_hash": _entry_identity_hash(selected_type, typed_entries, hash_mode),
                        "entry_count_before": len(original_entries),
                        "entry_count_after": len(typed_entries),
                    })
                update_refinement_progress(
                    progress_file,
                    type_keys[selected_type],
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
                        "error": None,
                        "completed_chunks": len(successful_results),
                        "total_chunks": len(chunks),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in group_selected_types:
                if selected_type in completed_type_results:
                    continue
                update_refinement_progress(
                    progress_file,
                    type_keys[selected_type],
                    {
                        "entry_type": selected_type,
                        "status": "in_progress",
                        "error": None,
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
                        "completed_chunks": len(successful_results),
                        "total_chunks": len(chunks),
                    },
                    atomic_replace_fn=atomic_replace_fn,
                )
            for selected_type in group_selected_types:
                if selected_type in completed_type_results or (
                    skip_entry_type
                    and _refinement_type_key(selected_type) == _refinement_type_key(skip_entry_type)
                ):
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
                    if future.cancelled():
                        continue
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
                        continue
                    if status != "ok":
                        failed_result = result
                        for pending in futures:
                            pending.cancel()
                        continue

                    chunk_results[result["chunk_idx"]] = result
                    completed_chunks += 1
                    _remember_success(result)
                    _mark_type_chunk_success(result)

            if failed_result:
                _record_failed_chunk(failed_result)
                if not send_all_types:
                    _mark_remaining_pending_failed(
                        "refinement_aborted_after_chunk_failure",
                        failed_result.get("entry_type"),
                    )
                return "stopped" if stopped_result else "failed", entry_type, completed_type_results

            if stopped_result:
                completed_chunks = len(chunk_results)
                log(f"Glossary refinement stopped during chunk {stopped_result.get('chunk_idx')}/{stopped_result.get('total_chunks')}")
                _mark_all_pending_stopped()
                return "stopped", entry_type, completed_type_results

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
                    return "stopped", entry_type, completed_type_results
                if status != "ok":
                    _record_failed_chunk(result)
                    if not send_all_types:
                        _mark_remaining_pending_failed(
                            "refinement_aborted_after_chunk_failure",
                            result.get("entry_type"),
                        )
                    return "failed", entry_type, completed_type_results

                refined_entries.extend(result.get("entries") or [])
                completed_chunks += 1
                _remember_success(result)
                _mark_type_chunk_success(result)

        if refined_entries:
            result_mapping = completed_type_results
            refined_entries = [
                row for selected_type in group_selected_types
                for row in result_mapping.get(selected_type, [])
            ]
            model_name = last_model_name or _actual_request_model_name(client)
            request_update = dict(last_request_context or _actual_request_key_context(client))
            if model_name:
                request_update["model_name"] = model_name
            if send_all_types:
                broad_completed_update = {
                    "entry_type": "all selected entry types",
                    "status": "completed",
                    "error": None,
                    "input_hash": broad_input_hash,
                    "output_hash": _entry_hash("all selected entry types", refined_entries, hash_mode),
                    "identity_hash_version": _IDENTITY_HASH_VERSION,
                    "input_identity_hash": broad_input_identity_hash,
                    "output_identity_hash": _entry_identity_hash("all selected entry types", refined_entries, hash_mode),
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
                if future.cancelled():
                    continue
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
                    # Drain running groups: their completed results must be
                    # saved even when another group observes Stop first.
    else:
        for group in work_groups:
            status, _entry_type, result_mapping = _process_group(group)
            if result_mapping:
                refined_by_type.update(result_mapping)
            if status == "stopped":
                stopped = True
                break

    if not refined_by_type:
        return original_glossary if stopped else glossary

    selected_lc = {_refinement_type_key(t) for t in refined_by_type}
    rebuilt = [entry for entry in glossary if _refinement_type_key(entry.get("type", "")) not in selected_lc]
    for entry_type in selected_types:
        rebuilt.extend(refined_by_type.get(entry_type, []))
    rebuilt = _strip_inactive_description(rebuilt)
    if not skip_dedupe:
        rebuilt = dedupe_fn(rebuilt)
        rebuilt = _strip_inactive_description(rebuilt)
    return rebuilt
