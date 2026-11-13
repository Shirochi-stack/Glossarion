"""Shared gender-tracker resolution helpers.

The extractor owns occurrence collection.  This module keeps the decision
rules in one place so glossary saving, compression, and the editor cannot
drift apart, and it holds the one writer of the tracker sidecars: the
cross-process guard and the atomic JSON write used by the gender tracker and
the relations tracker (``relations_tracking``), plus the gender tracker's
load / order / write and the one function that edits a gender entry.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


BINARY_GENDERS = ("male", "female")
# "empty": what a model writes when it is told to leave the gender empty
UNKNOWN_GENDERS = {"", "unknown", "n/a", "na", "none", "-", "empty"}
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


# What a gender column can legitimately hold, beyond male / female and their
# aliases. Lifted from parse_api_response, which already used it to notice a
# description sitting in the gender position.
_GENDER_WORDS = frozenset({
    "male", "female", "unknown", "nonbinary", "non-binary", "ambiguous", "mixed",
    "various", "n/a", "na", "none", "-", "other", "neutral", "both", "genderless",
    "androgynous", "?",
})
_GLUED_GENDER_RE = re.compile(r"^\s*([A-Za-z/\-?]{1,12}),(?=\S)")


def looks_like_gender_value(value: Any) -> bool:
    return normalize_gender(value) in _GENDER_WORDS


def _reads_like_a_sentence(text: str) -> bool:
    # Deliberately conservative: a short label this module does not know
    # (여성, a custom value) is left alone; only prose is moved.
    return len(text) > 30 or len(text.split()) >= 5


def repair_gender_description(gender: Any, description: Any) -> Tuple[str, str]:
    """Put a gender and a description back in their own columns.

    Two shapes older glossaries contain, both from rows whose column count
    did not match the header:

    * the description sitting in the gender slot -- `[A nervous recruit ...]`;
    * the gender glued onto the description -- `male,A temporary ...`
      (no space after the comma: it is a column join, not prose).
    """
    gender = str(gender or "").strip()
    description = str(description or "").strip()
    if gender and not looks_like_gender_value(gender) and _reads_like_a_sentence(gender):
        description = gender if not description else f"{gender} {description}"
        gender = ""
    if not gender:
        glued = _GLUED_GENDER_RE.match(description)
        if glued and looks_like_gender_value(glued.group(1)):
            gender = glued.group(1)
            description = description[glued.end():].lstrip()
    return gender, description


# Header names a name-only row can echo; such a row is never an entry.
_NAME_ONLY_HEADER_WORDS = frozenset({"type", "raw_name", "translated_name", "gender", "description"})


def is_name_only_header(cells: Sequence[Any]) -> bool:
    """A ``raw_name,translated_name[,gender]`` header: the names without a type column."""
    names = [str(cell or "").strip().lower() for cell in cells or ()]
    return "raw_name" in names and "translated_name" in names and "type" not in names


def name_only_row(cells: Sequence[Any], known_types: Any = ()) -> Optional[List[str]]:
    """``[raw_name, translated_name, gender]`` of a row written without its type column, else None.

    The On-device prompts ask small models for ``raw_name,translated_name,gender`` lines,
    and some drop the gender. Such a row is a character: two or three fields, the first
    is not an entry type (``known_types`` plus character/term, plurals included) and a
    third field, when present, is empty or reads as a gender. An echoed header or prose
    (a sentence, a label ending in ``:``) is not a row.
    """
    fields = [str(cell or "").strip() for cell in cells or ()]
    if len(fields) not in (2, 3):
        return None
    raw_name, translated_name = fields[0], fields[1]
    gender = fields[2] if len(fields) == 3 else ""
    if not raw_name or not translated_name:
        return None
    if gender and not looks_like_gender_value(gender):
        return None
    known = {str(name or "").strip().lower() for name in known_types or ()} | {"character", "term"}
    first = raw_name.lower()
    forms = {first, first + "s", first[:-1] if first.endswith("s") else first,
             first[:-3] + "y" if first.endswith("ies") else first}
    if forms & known:
        return None
    for name in (raw_name, translated_name):
        if (name.lower() in _NAME_ONLY_HEADER_WORDS or len(name) > 60 or len(name.split()) >= 6
                or name.endswith((":", "：", ".", "。"))):
            return None
    return [raw_name, translated_name, gender]


def display_gender(value: Any) -> str:
    """The form a gender is stored and shown in: first letter capital.

    Comparisons everywhere use normalize_gender() (lower case), so the
    tracker's consolidated `male` / `female` used to be written back next
    to the `Male` / `Female` the AI returns, and one glossary column ended
    up with both spellings. Aliases are resolved too (`m` -> `Male`).
    Anything else keeps its own spelling with the first letter raised
    (`unknown` -> `Unknown`, `non-binary` -> `Non-binary`, `N/A` stays).
    """
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = normalize_gender(text)
    if normalized in BINARY_GENDERS:
        return normalized.capitalize()   # male, MALE, m, boy -> Male
    return text[:1].upper() + text[1:]


def normalize_entries_gender(entries: Any) -> Any:
    """Apply display_gender() in place to a list of entry dicts, or to the
    values of a name -> entry mapping. Returns what it was given."""
    if isinstance(entries, Mapping):
        rows = entries.values()
    elif isinstance(entries, (list, tuple)):
        rows = entries
    else:
        return entries
    for row in rows:
        if not isinstance(row, dict):
            continue
        # The description key keeps whatever casing the user configured.
        desc_key = next((k for k in row if isinstance(k, str) and k.strip().lower() == "description"), None)
        old_gender = str(row.get("gender") or "")
        old_desc = str(row.get(desc_key) or "") if desc_key else ""
        gender, description = repair_gender_description(old_gender, old_desc)
        if description != old_desc.strip():
            row[desc_key or "description"] = description
        if gender or old_gender:
            row["gender"] = display_gender(gender)
    return entries


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


# ---------------------------------------------------------------------------
# Tracker sidecar files (the gender tracker and the relations tracker)
# ---------------------------------------------------------------------------

#: The gender tracker's in-process lock (extract_glossary_from_epub._gender_tracker_lock is this object).
TRACKER_LOCK = threading.RLock()
_guard_depths = threading.local()


def _lock_file(lock_base: str):
    """The cross-process lock file (glossary_refinement.locked_progress_file)."""
    from glossary_refinement import locked_progress_file
    return locked_progress_file(lock_base)


@contextlib.contextmanager
def tracker_file_guard(tracker_path: str, thread_lock: Any = None):
    """Hold a tracker sidecar for one load -> modify -> write.

    ``thread_lock`` (default ``TRACKER_LOCK``) serialises this process's threads; a lock
    file in the temp folder (one per tracker path, so the glossary folder stays clean)
    serialises processes, so the GUI and an extraction subprocess cannot overwrite each
    other's update. Re-entrant: a nested guard on the same path takes the file lock once.
    """
    lock = thread_lock if thread_lock is not None else TRACKER_LOCK
    key = os.path.normcase(os.path.abspath(tracker_path)) if tracker_path else ""
    with lock:
        depths = getattr(_guard_depths, "depths", None)
        if depths is None:
            depths = _guard_depths.depths = {}
        guard = None
        if tracker_path and not depths.get(key):
            digest = hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()
            lock_base = os.path.join(tempfile.gettempdir(), "glossarion_tracker_locks", digest)
            try:
                guard = _lock_file(lock_base)
                guard.__enter__()
            except Exception:
                guard = None  # no lock file possible: the in-process lock still holds
        depths[key] = depths.get(key, 0) + 1
        try:
            yield
        finally:
            depths[key] -= 1
            if not depths[key]:
                depths.pop(key, None)
            if guard is not None:
                guard.__exit__(None, None, None)


def atomic_replace_file(temp_path: str, target_path: str, max_retries: int = 3, delay: float = 0.5):
    """Atomically replace target_path with temp_path, retrying on Windows lock errors.

    Uses os.replace() which is atomic on Windows (unlike os.remove+os.rename).
    Retries with a short delay to handle transient file locks from antivirus,
    editors, or other processes that briefly hold the file open.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return  # Success
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    # All retries exhausted — clean up temp file and raise
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
    raise last_err


def atomic_write_json(path: str, data: Any) -> None:
    """Write a tracker sidecar: temp file in the same folder, fsync, atomic replace."""
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False, suffix=".tmp") as temp_f:
        temp_path = temp_f.name
        json.dump(data, temp_f, ensure_ascii=False, indent=2)
        temp_f.flush()
        os.fsync(temp_f.fileno())
    atomic_replace_file(temp_path, path)


def file_stamp(path: Optional[str]) -> Optional[Tuple[int, int]]:
    """``(mtime_ns, size)`` of ``path``, or None when it is missing (one os.stat)."""
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def read_json_file(path: Optional[str]) -> Optional[Any]:
    """The parsed file, ``{}`` when it is missing, None when it cannot be read or parsed
    (a file being replaced right now): a reader skips it instead of showing it empty."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gender tracker file
# ---------------------------------------------------------------------------

def load_tracker(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"version": 2, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("version", 1)
            data.setdefault("entries", {})
            return data
    except Exception:
        pass
    return {"version": 2, "entries": {}}


def tracker_int(value, fallback=10**9):
    try:
        if value is None or value == "":
            return fallback
        return int(value)
    except Exception:
        return fallback


def tracker_occurrence_sort_key(occurrence: Dict):
    chapter_file = str(occurrence.get("chapter_file", "") or "")
    file_num_match = re.search(r"(\d+)", chapter_file)
    file_num = int(file_num_match.group(1)) if file_num_match else 10**9
    return (
        tracker_int(occurrence.get("chapter_index")),
        tracker_int(occurrence.get("chapter_num")),
        file_num,
        chapter_file,
        str(occurrence.get("gender", "") or ""),
    )


def normalize_tracker_order(tracker: Dict):
    if not isinstance(tracker, dict):
        return tracker
    entries = tracker.get("entries", {})
    if not isinstance(entries, dict):
        return tracker

    def first_occurrence_key(item):
        occurrences = item.get("occurrences", []) if isinstance(item, dict) else []
        occurrences = [o for o in occurrences if isinstance(o, dict)]
        return tracker_occurrence_sort_key(occurrences[0]) if occurrences else (10**9, 10**9, 10**9, "", "")

    normalized_entries = []
    for key, item in entries.items():
        if not isinstance(item, dict):
            normalized_entries.append((key, item))
            continue
        occurrences = [o for o in item.get("occurrences", []) if isinstance(o, dict)]
        occurrences.sort(key=tracker_occurrence_sort_key)
        item["occurrences"] = occurrences

        changes = []
        previous = None
        for occurrence in occurrences:
            if previous and previous.get("gender") != occurrence.get("gender"):
                changes.append({
                    "from": previous.get("gender"),
                    "to": occurrence.get("gender"),
                    "chapter_index": occurrence.get("chapter_index"),
                    "chapter_num": occurrence.get("chapter_num"),
                    "chapter_file": occurrence.get("chapter_file", ""),
                })
            previous = occurrence
        item["changes"] = changes

        genders = item.setdefault("genders", {})
        if isinstance(genders, dict):
            for gender, meta in genders.items():
                if not isinstance(meta, dict):
                    continue
                first = next((o for o in occurrences if o.get("gender") == gender), None)
                if first:
                    meta["first_seen_chapter"] = first.get("chapter_num")
                    meta["first_seen_file"] = first.get("chapter_file", "")

        normalized_entries.append((key, item))

    normalized_entries.sort(key=lambda pair: (first_occurrence_key(pair[1]), str(pair[0])))
    tracker["entries"] = {key: item for key, item in normalized_entries}
    return tracker


def write_tracker(path: str, tracker: Dict) -> None:
    """Stamp, order and atomically write the gender tracker (raises on failure)."""
    if not path or not isinstance(tracker, dict):
        return
    tracker["version"] = max(2, tracker_int(tracker.get("version"), fallback=2))
    tracker["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    normalize_tracker_order(tracker)
    atomic_write_json(path, tracker)


def _comparable(tracker: Any) -> str:
    data = dict(tracker) if isinstance(tracker, dict) else {}
    data.pop("updated_at", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def update_tracker_file(path: str, mutate: Callable[[Dict], Any]) -> Dict[str, Any]:
    """Load -> ``mutate(tracker)`` -> write (only when something changed) under the guard.

    Returns ``{"changed", "result" (mutate's return), "before" (the tracker as loaded),
    "tracker" (as written), "stamp" (file_stamp after)}``. A file that exists but cannot
    be parsed is never overwritten: ``ValueError``.
    """
    with tracker_file_guard(path):
        raw = read_json_file(path)
        if raw is None or not isinstance(raw, dict):
            raise ValueError(f"The gender tracker cannot be read: {path}")
        tracker = load_tracker(path) if raw else {"version": 2, "entries": {}}
        before = copy.deepcopy(tracker)
        result = mutate(tracker)
        normalize_tracker_order(tracker)
        changed = _comparable(normalize_tracker_order(copy.deepcopy(before))) != _comparable(tracker)
        if changed:
            write_tracker(path, tracker)
        return {"changed": changed, "result": result, "before": before, "tracker": tracker,
                "stamp": file_stamp(path)}


def _same_chapter_num(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def set_tracker_gender(
    tracker: Dict,
    raw_name: Any,
    gender: Any,
    chapter: Any = None,
    *,
    chapter_file: str = "",
    chapter_index: Any = None,
    translated_name: str = "",
) -> bool:
    """Set a gender entry in a gender tracker dict (in place); True when it changed.

    ``chapter`` None: the gender for every chapter (the entry's ``decision``, as the
    Glossary Editor sets it; ``"auto"`` or ``""`` clears it). With a chapter number: that
    chapter's observations are replaced by one hand-set occurrence (``"manual": True``,
    which a later extraction of the chapter does not override) and a decision is reset to
    ``auto`` so the chapter's gender shows; ``""`` / ``"auto"`` removes the chapter's
    observations. The entry is created when missing.
    """
    raw = str(raw_name or "").strip()
    if not raw or not isinstance(tracker, dict):
        return False
    wanted = normalize_gender(gender)
    if wanted in UNKNOWN_GENDERS:
        wanted = "auto"
    if wanted not in VALID_DECISIONS:
        return False
    entries = tracker.setdefault("entries", {})
    key = tracker_key(raw)
    item = entries.get(key)
    if not isinstance(item, dict):
        if wanted == "auto":
            return False
        item = {"raw_name": raw, "translated_name": translated_name or "", "decision": "auto",
                "genders": {}, "occurrences": [], "changes": []}
        entries[key] = item
    before = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    if translated_name and not item.get("translated_name"):
        item["translated_name"] = translated_name
    if chapter is None:
        item["decision"] = wanted
    else:
        chapter_file = os.path.basename(str(chapter_file or ""))
        kept = []
        for occurrence in item.get("occurrences", []):
            if isinstance(occurrence, dict) and _same_chapter_num(occurrence.get("chapter_num"), chapter) and (
                    not chapter_file or not occurrence.get("chapter_file")
                    or os.path.basename(str(occurrence.get("chapter_file"))) == chapter_file):
                if chapter_index is None and occurrence.get("chapter_index") is not None:
                    chapter_index = occurrence.get("chapter_index")
                if not chapter_file and occurrence.get("chapter_file"):
                    chapter_file = os.path.basename(str(occurrence.get("chapter_file")))
                continue
            kept.append(occurrence)
        if wanted != "auto":
            kept.append({
                "raw_name": item.get("raw_name") or raw,
                "gender": wanted,
                "chapter_index": chapter_index,
                "chapter_num": chapter,
                "chapter_file": chapter_file,
                "source_path": tracker.get("source_path", ""),
                "translated_name": item.get("translated_name", ""),
                "manual": True,
            })
            item.setdefault("genders", {}).setdefault(wanted, {
                "first_seen_chapter": chapter, "first_seen_file": chapter_file})
            if normalize_gender(item.get("decision", "auto")) in BINARY_GENDERS:
                item["decision"] = "auto"
        item["occurrences"] = kept
        genders = item.get("genders")
        if isinstance(genders, dict):
            seen = {normalize_gender(o.get("gender")) for o in kept if isinstance(o, dict)}
            for name in [g for g in genders if normalize_gender(g) not in seen]:
                genders.pop(name, None)
    return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) != before


def chapter_has_manual_gender(item: Optional[Mapping[str, Any]], chapter_num: Any, chapter_file: Any = "") -> bool:
    """A hand-set gender at this chapter (extraction leaves that chapter alone)."""
    if not isinstance(item, Mapping):
        return False
    chapter_file = os.path.basename(str(chapter_file or ""))
    for occurrence in item.get("occurrences", []) or []:
        if (isinstance(occurrence, Mapping) and occurrence.get("manual")
                and _same_chapter_num(occurrence.get("chapter_num"), chapter_num)
                and (not chapter_file or not occurrence.get("chapter_file")
                     or os.path.basename(str(occurrence.get("chapter_file"))) == chapter_file)):
            return True
    return False
