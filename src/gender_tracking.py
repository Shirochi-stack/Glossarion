"""Shared gender-tracker resolution helpers.

The extractor owns occurrence collection and persistence.  This module keeps
the read-only decision rules in one place so glossary saving, compression,
and the editor cannot drift apart.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


BINARY_GENDERS = ("male", "female")
UNKNOWN_GENDERS = {"", "unknown", "n/a", "na", "none", "-"}
VALID_DECISIONS = {"auto", *BINARY_GENDERS}


def normalize_gender(value: Any) -> str:
    gender = str(value or "").strip().lower()
    aliases = {
        "m": "male",
        "man": "male",
        "boy": "male",
        "masc": "male",
        "masculine": "male",
        "f": "female",
        "woman": "female",
        "girl": "female",
        "fem": "female",
        "feminine": "female",
    }
    return aliases.get(gender, gender)


def tracker_key(raw_name: Any) -> str:
    return str(raw_name or "").strip().casefold()


def tracker_path_for_glossary(glossary_path: str) -> str:
    if not glossary_path:
        return ""
    stem, _ext = os.path.splitext(glossary_path)
    if stem.endswith("_glossary"):
        stem = stem[:-len("_glossary")]
    elif os.path.basename(stem).lower() == "glossary":
        stem = os.path.join(os.path.dirname(stem), "gender")
    return f"{stem}_gender_tracker.json"


def tracker_entry_for_raw(tracker: Optional[Mapping[str, Any]], raw_name: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tracker, Mapping):
        return None
    entries = tracker.get("entries")
    if not isinstance(entries, Mapping):
        return None
    entry = entries.get(tracker_key(raw_name))
    return entry if isinstance(entry, dict) else None


def normalized_decision(entry: Optional[Mapping[str, Any]]) -> str:
    decision = normalize_gender(entry.get("decision", "auto")) if isinstance(entry, Mapping) else "auto"
    return decision if decision in VALID_DECISIONS else "auto"


def normalized_occurrences(entry: Optional[Mapping[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    if not isinstance(entry, Mapping):
        return []
    result: List[Tuple[Dict[str, Any], str]] = []
    seen = set()
    for occurrence in entry.get("occurrences", []):
        if not isinstance(occurrence, dict):
            continue
        gender = normalize_gender(occurrence.get("gender"))
        if gender in UNKNOWN_GENDERS:
            continue
        signature = (
            gender,
            str(occurrence.get("chapter_num", "")),
            os.path.basename(str(occurrence.get("chapter_file", "") or "")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        result.append((occurrence, gender))
    return result


def gender_counts(entry: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(gender for _occurrence, gender in normalized_occurrences(entry))
    return dict(counts)


def has_binary_conflict(entry: Optional[Mapping[str, Any]]) -> bool:
    counts = gender_counts(entry)
    return counts.get("male", 0) > 0 and counts.get("female", 0) > 0


def normalize_threshold(value: Any = None) -> float:
    percentage_input = value is None or isinstance(value, (str, int))
    if value is None:
        value = os.getenv("GLOSSARY_GENDER_NOISE_THRESHOLD", "10")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 10.0
    # Callers may provide either the GUI percentage or the internal fraction.
    if percentage_input or threshold > 1.0:
        threshold /= 100.0
    return max(0.0, min(1.0, threshold))


def normalize_bias(value: Any = None) -> str:
    if value is None:
        value = os.getenv("GLOSSARY_GENDER_TRACKING_BIAS", "none")
    bias = normalize_gender(value)
    return bias if bias in BINARY_GENDERS else "none"


def rarity_stats(entry: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    counts = gender_counts(entry)
    total = sum(counts.values())
    if not total:
        return {}
    return {
        gender: {"count": count, "total": total, "ratio": count / total}
        for gender, count in counts.items()
    }


def rare_genders(
    entry: Optional[Mapping[str, Any]],
    threshold: Any = None,
    bias: Any = None,
) -> set[str]:
    threshold_value = normalize_threshold(threshold)
    if threshold_value <= 0:
        return set()
    bias_value = normalize_bias(bias)
    return {
        gender
        for gender, values in rarity_stats(entry).items()
        if gender != bias_value and values["ratio"] <= threshold_value
    }


def viable_binary_genders(
    entry: Optional[Mapping[str, Any]],
    threshold: Any = None,
    bias: Any = None,
) -> set[str]:
    observed = {gender for gender in gender_counts(entry) if gender in BINARY_GENDERS}
    if not observed:
        return set()
    threshold_value = normalize_threshold(threshold)
    bias_value = normalize_bias(bias)
    # This is the compressor's existing explicit "keep both" setting.
    if threshold_value >= 1.0 and bias_value == "none":
        return observed
    viable = observed - rare_genders(entry, threshold_value, bias_value)
    return viable or observed


def automatic_storage_gender(entry: Optional[Mapping[str, Any]], existing_gender: Any = "") -> str:
    """Choose the persisted Auto gender using observation frequency.

    A tie is deliberately stable: retain the existing valid value, otherwise
    use the earliest binary observation.
    """
    counts = gender_counts(entry)
    binary_counts = {gender: counts.get(gender, 0) for gender in BINARY_GENDERS}
    highest = max(binary_counts.values(), default=0)
    if highest <= 0:
        return normalize_gender(existing_gender)
    winners = [gender for gender in BINARY_GENDERS if binary_counts[gender] == highest]
    if len(winners) == 1:
        return winners[0]
    existing = normalize_gender(existing_gender)
    if existing in winners:
        return existing
    for _occurrence, gender in normalized_occurrences(entry):
        if gender in winners:
            return gender
    return winners[0]


def resolved_storage_gender(entry: Optional[Mapping[str, Any]], existing_gender: Any = "") -> str:
    decision = normalized_decision(entry)
    if decision in BINARY_GENDERS:
        return decision
    return automatic_storage_gender(entry, existing_gender)


def _chapter_ref_parts(chapter_ref: Any) -> Tuple[Optional[float], str]:
    if isinstance(chapter_ref, Mapping):
        chapter_num = chapter_ref.get("chapter_num")
        chapter_file = chapter_ref.get("chapter_file")
    else:
        chapter_num = chapter_ref
        chapter_file = None
    if chapter_num is None:
        chapter_num = os.getenv("CURRENT_CHAPTER_NUM")
    if not chapter_file:
        chapter_file = os.getenv("CURRENT_CHAPTER_FILE")
    try:
        chapter_num_value = float(chapter_num)
    except (TypeError, ValueError):
        chapter_num_value = None
    return chapter_num_value, os.path.basename(str(chapter_file or ""))


def automatic_chapter_gender(
    entry: Optional[Mapping[str, Any]],
    chapter_ref: Any = None,
    threshold: Any = None,
    bias: Any = None,
) -> Optional[str]:
    occurrences = normalized_occurrences(entry)
    rare = rare_genders(entry, threshold, bias)
    if rare:
        filtered = [(occurrence, gender) for occurrence, gender in occurrences if gender not in rare]
        if filtered:
            occurrences = filtered
    if not occurrences:
        return None

    chapter_num, chapter_file = _chapter_ref_parts(chapter_ref)
    if chapter_file:
        for occurrence, gender in reversed(occurrences):
            if os.path.basename(str(occurrence.get("chapter_file", ""))) == chapter_file:
                return gender
    if chapter_num is not None:
        best_gender = None
        best_num = None
        for occurrence, gender in occurrences:
            try:
                occurrence_num = float(occurrence.get("chapter_num"))
            except (TypeError, ValueError):
                continue
            if occurrence_num <= chapter_num and (best_num is None or occurrence_num >= best_num):
                best_gender = gender
                best_num = occurrence_num
        if best_gender:
            return best_gender
    return occurrences[-1][1]


def effective_gender(
    entry: Optional[Mapping[str, Any]],
    stored_gender: Any = "",
    chapter_ref: Any = None,
    threshold: Any = None,
    bias: Any = None,
) -> str:
    decision = normalized_decision(entry)
    if decision in BINARY_GENDERS:
        return decision
    if isinstance(chapter_ref, Mapping) and chapter_ref.get("use_storage_gender"):
        return (
            resolved_storage_gender(entry, stored_gender)
            or normalize_gender(stored_gender)
        )
    return (
        automatic_chapter_gender(entry, chapter_ref, threshold, bias)
        or resolved_storage_gender(entry, stored_gender)
        or normalize_gender(stored_gender)
    )


def editor_gender_status(
    entry: Optional[Mapping[str, Any]],
    stored_gender: Any = "",
    threshold: Any = None,
    bias: Any = None,
) -> Dict[str, Any]:
    decision = normalized_decision(entry)
    conflict = has_binary_conflict(entry)
    storage_gender = resolved_storage_gender(entry, stored_gender)
    viable = viable_binary_genders(entry, threshold, bias)
    unresolved = conflict and decision == "auto"
    if decision in BINARY_GENDERS:
        label = decision.title()
    elif unresolved and len(viable) > 1:
        label = "Male / Female"
    elif unresolved and len(viable) == 1:
        label = f"{next(iter(viable)).title()}*"
    else:
        label = (storage_gender or normalize_gender(stored_gender)).title()
    return {
        "decision": decision,
        "conflict": conflict,
        "unresolved": unresolved,
        "storage_gender": storage_gender,
        "viable_genders": viable,
        "label": label,
        "stats": rarity_stats(entry),
    }


def collapse_tracked_gender_variants(
    entries: Sequence[Dict[str, Any]],
    tracker: Optional[Mapping[str, Any]],
    *,
    has_gender: Optional[Callable[[Dict[str, Any]], bool]] = None,
    score_entry: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Collapse tracked binary variants and apply the persisted winner.

    Rows are collapsed only when the associated tracker has evidence for both
    male and female.  This keeps ordinary/manual same-name rows unchanged when
    there is no tracker evidence.
    """
    copied = [dict(entry) if isinstance(entry, dict) else entry for entry in entries or []]
    groups: Dict[str, List[int]] = {}
    tracker_items: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(copied):
        if not isinstance(entry, dict):
            continue
        if has_gender is not None and not has_gender(entry):
            continue
        gender = normalize_gender(entry.get("gender", ""))
        if gender not in BINARY_GENDERS:
            continue
        key = tracker_key(entry.get("raw_name", ""))
        tracker_entry = tracker_entry_for_raw(tracker, entry.get("raw_name", ""))
        if not key or not has_binary_conflict(tracker_entry):
            continue
        groups.setdefault(key, []).append(index)
        tracker_items[key] = tracker_entry

    replacements: Dict[int, Dict[str, Any]] = {}
    skipped: set[int] = set()
    collapsed_count = 0
    for key, indices in groups.items():
        first_index = indices[0]
        first_entry = copied[first_index]
        tracker_entry = tracker_items[key]
        winner = resolved_storage_gender(tracker_entry, first_entry.get("gender", ""))
        candidates = [copied[index] for index in indices]
        if score_entry is not None:
            chosen = max(candidates, key=score_entry)
        else:
            chosen = candidates[0]
        replacement = dict(chosen)
        for field in ("raw_name", "translated_name"):
            if field in first_entry:
                replacement[field] = first_entry.get(field, "")
        # Preserve useful values that only existed on the other variant.
        for index in indices:
            for field, value in copied[index].items():
                if field not in replacement or replacement[field] in (None, "", [], {}):
                    replacement[field] = value
        replacement["gender"] = winner
        replacements[first_index] = replacement
        skipped.update(indices[1:])
        collapsed_count += max(0, len(indices) - 1)

    result: List[Dict[str, Any]] = []
    for index, entry in enumerate(copied):
        if index in skipped:
            continue
        result.append(replacements.get(index, entry))
    return result, collapsed_count


def occurrence_bounds(entry: Optional[Mapping[str, Any]], gender: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    wanted = normalize_gender(gender)
    matches = [occurrence for occurrence, actual in normalized_occurrences(entry) if actual == wanted]
    return (matches[0], matches[-1]) if matches else (None, None)
