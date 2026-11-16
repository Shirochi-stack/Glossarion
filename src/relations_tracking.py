"""Shared relations-tracker rules (GUI-free).

The glossary AI returns a ``relations`` column on character rows
(``other raw_name: what that character is to this one; ...``).  The column is
never written to the glossary file; every observation is stored in a
``*_relations_tracker.json`` sidecar together with the chapter it came from.
At translation time the tracker adds a relations column that only shows what
is known up to the chapter being translated.

This module holds the pure rules shared by the extractor, the translator, the
desktop GUI and mobile: settings, file names, parsing, recording, reading at a
chapter, injection into glossary text, the prompt placeholder, the custom-field
rules and a dependency-free graph layout.  It also holds the one writer of the
sidecar (``tracker_guard`` + ``write_tracker`` / ``edit_tracker``: the extractor,
the desktop diagram and the mobile screen all write through it), the hand edits
of the Relations & Gender Tracker (corrections and ignores that a later
extraction honours), the per-book view choices and the gender reading the
diagram shows (from the book's gender tracker, ``gender_tracking``).

Nothing heavy is imported at import time (no Qt, no Flet, no app modules).
"""

from __future__ import annotations

import bisect
import copy
import csv
import difflib
import io
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from typing import Any, Callable, Container, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gender_tracking as _gender

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATIONS_PLACEHOLDER = "{relations_instruction}"
RELATIONS_PLACEHOLDER_LINE = "- {relations_instruction}"
DEFAULT_FIELD_NAME = "relations"
#: Glossary columns the relations field can never be named after.
RESERVED_FIELD_NAMES = ("type", "raw_name", "translated_name", "gender", "description")
HISTORY_MODES = ("previous", "full", "current")
DEFAULT_HISTORY_MODE = "previous"

ENV_ENABLE = "GLOSSARY_ENABLE_RELATIONS_TRACKER"
ENV_HISTORY_MODE = "GLOSSARY_RELATIONS_HISTORY_MODE"
ENV_FIELD = "GLOSSARY_RELATIONS_FIELD"

CONFIG_ENABLE = "glossary_enable_relations_tracker"
CONFIG_HISTORY_MODE = "glossary_relations_history_mode"
CONFIG_FIELD = "glossary_relations_field_name"
#: Minimal Glossary Generation asks for (and records) relations only with this on as well.
ENV_MINIMAL = "GLOSSARY_MINIMAL_INCLUDE_RELATIONS"
CONFIG_MINIMAL = "glossary_minimal_include_relations"
#: A fresh config (no config.json yet, or one "Reset Settings to Defaults" wrote) starts with both
#: toggles on (owner 2026-11-13: its prompts are the built-in ones, so no prompt reset is needed).
#: The key lists the toggles that still take that default: desktop sets them on once at startup
#: (``apply_fresh_defaults``); Glossarion Mobile keeps them following the selected model until the
#: user sets one (``state.setting_writes``). An existing config never has it.
FRESH_DEFAULTS_KEY = "glossary_relations_fresh_defaults"
FRESH_DEFAULT_KEYS = (CONFIG_ENABLE, CONFIG_MINIMAL)

TRACKER_SUFFIX = "_relations_tracker.json"
TRACKER_BASENAME = "relations_tracker.json"
TRACKER_VERSION = 1

_TRUE_VALUES = ("1", "true", "yes", "on")
_BIG = float("1e18")
_LABEL_SIMILARITY = 0.85
_MAX_LABEL_CHARS = 60
_MAX_LABEL_WORDS = 8
_MAX_NAME_CHARS = 80
_QUOTES = "\"'`“”‘’「」『』«»"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_CONFIG_FOR_ENV = {
    ENV_ENABLE: CONFIG_ENABLE,
    ENV_HISTORY_MODE: CONFIG_HISTORY_MODE,
    ENV_FIELD: CONFIG_FIELD,
    ENV_MINIMAL: CONFIG_MINIMAL,
}


def _setting(settings: Optional[Mapping[str, Any]], env_name: str) -> Any:
    """Value from a request snapshot (ENV names) or a config dict (config keys), else os.environ."""
    if isinstance(settings, Mapping):
        if env_name in settings and settings.get(env_name) is not None:
            return settings.get(env_name)
        config_name = _CONFIG_FOR_ENV.get(env_name)
        if config_name and config_name in settings and settings.get(config_name) is not None:
            return settings.get(config_name)
    return os.environ.get(env_name)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def tracking_enabled(settings: Optional[Mapping[str, Any]] = None) -> bool:
    return _truthy(_setting(settings, ENV_ENABLE))


def minimal_tracking_enabled(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Minimal glossary relations: the tracker AND "Include Relations Column" (Minimal tab) are on."""
    return tracking_enabled(settings) and _truthy(_setting(settings, ENV_MINIMAL))


def field_name(settings: Optional[Mapping[str, Any]] = None) -> str:
    value = str(_setting(settings, ENV_FIELD) or "").strip()
    if value.casefold() in RESERVED_FIELD_NAMES:
        # Never let the tracker claim a real glossary column (a hand-edited config).
        return DEFAULT_FIELD_NAME
    return value or DEFAULT_FIELD_NAME


def field_name_problem(name: Any, other_fields: Iterable[Any] = ()) -> str:
    """Why ``name`` cannot be the relations field's name ('' when it can).

    Shared by the desktop rename (Custom Fields double-click) and the mobile list editor.
    """
    name = str(name or "").strip()
    if not name:
        return "The relations field needs a name."
    folded = name.casefold()
    if folded in RESERVED_FIELD_NAMES:
        return f"'{name}' is a built-in glossary column."
    if folded in {str(f or "").strip().casefold() for f in other_fields or ()}:
        return f"'{name}' is already a custom field."
    return ""


def history_mode(settings: Optional[Mapping[str, Any]] = None) -> str:
    value = str(_setting(settings, ENV_HISTORY_MODE) or "").strip().lower()
    return value if value in HISTORY_MODES else DEFAULT_HISTORY_MODE


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def tracker_key(raw_name: Any) -> str:
    """Same key rule as the gender tracker."""
    from gender_tracking import tracker_key as _gender_tracker_key
    return _gender_tracker_key(raw_name)


def tracker_path_for_glossary(glossary_path: str) -> str:
    """``X_glossary.csv`` -> ``X_relations_tracker.json``; ``glossary.*`` -> ``relations_tracker.json``.

    Uses gender_tracking.tracker_path_for_glossary for the stem rule and swaps
    the suffix, so both sidecars always sit next to each other.
    """
    if not glossary_path:
        return ""
    from gender_tracking import tracker_path_for_glossary as _gender_path
    stem, _ext = os.path.splitext(glossary_path)
    if os.path.basename(stem).lower() == "glossary":
        return os.path.join(os.path.dirname(stem), TRACKER_BASENAME)
    gender_path = _gender_path(glossary_path)
    gender_suffix = "_gender_tracker.json"
    return gender_path[: -len(gender_suffix)] + TRACKER_SUFFIX


def is_tracker_file(name: Any) -> bool:
    base = os.path.basename(str(name or "")).lower()
    return base == TRACKER_BASENAME or base.endswith(TRACKER_SUFFIX)


def empty_tracker() -> Dict[str, Any]:
    return {"version": TRACKER_VERSION, "entries": {}}


def load_tracker(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return empty_tracker()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return empty_tracker()
    if not isinstance(data, dict) or not isinstance(data.get("entries", {}), dict):
        return empty_tracker()
    data.setdefault("version", TRACKER_VERSION)
    data.setdefault("entries", {})
    return data


#: This process's lock for relations trackers (the extractor's _relations_tracker_lock is this object).
TRACKER_LOCK = threading.RLock()
#: Optional tracker keys, written only once the user edits or picks a view (absent otherwise).
EDITS_KEY = "edits"
VIEW_KEY = "view"


def tracker_guard(tracker_path: Optional[str]):
    """Context manager holding the tracker for one load -> modify -> write: ``TRACKER_LOCK``
    plus a lock file per tracker in the temp folder (other processes). Re-entrant."""
    return _gender.tracker_file_guard(tracker_path or "", TRACKER_LOCK)


def file_stamp(path: Optional[str]) -> Optional[Tuple[int, int]]:
    """``(mtime_ns, size)`` of a file, None when missing: the diagram's once-a-second check."""
    return _gender.file_stamp(path)


def read_tracker(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Like ``load_tracker`` (an empty tracker when the file is missing), but None when the file
    exists and cannot be read or parsed right now: a reader keeps what it shows and retries."""
    data = _gender.read_json_file(path)
    if data is None:
        return None
    if data == {}:
        return empty_tracker()
    if not isinstance(data, dict) or not isinstance(data.get("entries", {}), dict):
        return None
    data.setdefault("version", TRACKER_VERSION)
    data.setdefault("entries", {})
    return data


def write_tracker(path: str, tracker: Dict[str, Any]) -> None:
    """Stamp, normalise and atomically write the tracker (call it inside ``tracker_guard``).
    Raises on failure (the extractor's wrapper prints a warning instead)."""
    if not path or not isinstance(tracker, dict):
        return
    tracker["version"] = TRACKER_VERSION
    tracker["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    normalize_tracker(tracker)
    _prune_optional_keys(tracker)
    _gender.atomic_write_json(path, tracker)


def _comparable(tracker: Any) -> str:
    data = dict(tracker) if isinstance(tracker, Mapping) else {}
    data.pop("updated_at", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def edit_tracker(path: str, mutate: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
    """Load -> ``mutate(tracker)`` -> write, under ``tracker_guard``; written only when it changed.

    ``mutate`` is any of the editing functions below (or a lambda calling several).
    Returns ``{"changed": bool, "result": what mutate returned, "before": the tracker as loaded
    (keep it for ``undo_edit``), "tracker": the tracker now (as written), "stamp": file_stamp
    after the write}``. A missing file starts empty; a file that exists but cannot be parsed is
    never overwritten (``ValueError``).
    """
    with tracker_guard(path):
        tracker = read_tracker(path)
        if tracker is None:
            raise ValueError(f"The relations tracker cannot be read: {path}")
        before = copy.deepcopy(tracker)
        result = mutate(tracker)
        normalize_tracker(tracker)
        _prune_optional_keys(tracker)
        reference = normalize_tracker(copy.deepcopy(before))
        _prune_optional_keys(reference)
        changed = _comparable(reference) != _comparable(tracker)
        if changed:
            write_tracker(path, tracker)
        return {"changed": changed, "result": result, "before": before, "tracker": tracker,
                "stamp": file_stamp(path)}


def _undo_parts(tracker: Any) -> Dict[Tuple[str, ...], Any]:
    parts: Dict[Tuple[str, ...], Any] = {}
    if not isinstance(tracker, Mapping):
        return parts
    entries = tracker.get("entries")
    if isinstance(entries, Mapping):
        for key, item in entries.items():
            parts[("entry", str(key))] = item
    for name in (EDITS_KEY, VIEW_KEY):
        if name in tracker:
            parts[(name,)] = tracker[name]
    return parts


def _undo_compare(value: Any) -> str:
    """An entry as edited, without what normalisation derives from other entries."""
    value = copy.deepcopy(value)
    if isinstance(value, dict) and isinstance(value.get("relations"), dict):
        for rel in value["relations"].values():
            if isinstance(rel, dict):
                rel.pop("changes", None)
                for occurrence in rel.get("occurrences") or []:
                    if isinstance(occurrence, dict):
                        occurrence.pop("flip", None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _restore_parts(path: str, edit: Mapping[str, Any], reader: Callable[[str], Any],
                   writer: Callable[[str, Dict[str, Any]], None], guard: Callable[[str], Any]) -> Dict[str, Any]:
    before, after = edit.get("before"), edit.get("tracker")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {"undone": False, "reason": "nothing"}
    with guard(path):
        current = reader(path)
        if not isinstance(current, dict):
            return {"undone": False, "reason": "unreadable"}
        old, new, now = _undo_parts(before), _undo_parts(after), _undo_parts(current)
        touched = [part for part in list(old) + [p for p in new if p not in old]
                   if _undo_compare(old.get(part)) != _undo_compare(new.get(part))]
        if not touched:
            return {"undone": False, "reason": "nothing"}
        if any(_undo_compare(now.get(part)) != _undo_compare(new.get(part)) for part in touched):
            return {"undone": False, "reason": "changed"}
        entries = current.setdefault("entries", {})
        for part in touched:
            value = copy.deepcopy(old.get(part))
            if part[0] == "entry":
                if part in old:
                    entries[part[1]] = value
                else:
                    entries.pop(part[1], None)
            elif part in old:
                current[part[0]] = value
            else:
                current.pop(part[0], None)
        writer(path, current)
        return {"undone": True, "tracker": current, "stamp": _gender.file_stamp(path)}


def undo_edit(path: str, edit: Mapping[str, Any]) -> Dict[str, Any]:
    """Undo one ``edit_tracker`` / ``save_view`` result: the characters (and the edit or view
    records) it changed get their earlier state back, under the lock.

    Returns ``{"undone": True, "tracker", "stamp"}``, or ``{"undone": False, "reason"}`` with
    ``"changed"`` (another writer changed one of those characters since: nothing is undone),
    ``"unreadable"`` or ``"nothing"``. Other characters' later updates are kept.
    """
    return _restore_parts(path, edit, read_tracker, write_tracker, tracker_guard)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _strip_quotes(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    return text.strip(_QUOTES).strip()


def parse_relations_cell(text: Any) -> List[Tuple[str, str]]:
    """``"도현: Older Brother; 하린: Best Friend"`` -> ``[("도현", "Older Brother"), ("하린", "Best Friend")]``."""
    if not isinstance(text, str):
        return []
    cell = _strip_quotes(text.replace("；", ";").replace("：", ":"))
    if not cell:
        return []
    pieces: List[str] = []
    for chunk in re.split(r"[;\n\r]+", cell):
        if chunk.count(":") > 1:
            # "A: Father, B: Mother" -- commas used as the item separator.
            pieces.extend(re.split(r"[,，、]", chunk))
        else:
            pieces.append(chunk)
    result: List[Tuple[str, str]] = []
    seen = set()
    for piece in pieces:
        piece = _strip_quotes(piece)
        if ":" not in piece:
            continue
        target, _sep, label = piece.partition(":")
        target = _strip_quotes(target)
        label = _strip_quotes(label).rstrip(".。").strip()
        if not target or not label or ":" in label:
            continue
        if len(target) > _MAX_NAME_CHARS or len(label) > _MAX_LABEL_CHARS:
            continue
        if len(label.split()) > _MAX_LABEL_WORDS:
            continue
        key = tracker_key(target)
        if key in seen:
            continue
        seen.add(key)
        result.append((target, label))
    return result


def _label_key(label: Any) -> str:
    text = unicodedata.normalize("NFKC", str(label or "")).casefold()
    key = "".join(ch for ch in text if ch.isalnum())
    return key or text.strip()


# Filler that names no real relationship; dropped when a cell is recorded (any case).
FILLER_LABELS = ("acquaintance", "stranger", "unknown", "none", "n/a", "surname", "same person", "self")
_FILLER_KEYS = frozenset(_label_key(label) for label in FILLER_LABELS)


def is_filler_label(label: Any) -> bool:
    return _label_key(label) in _FILLER_KEYS


# A label that only adds one of these to another (Ex-husband / Husband, Stepfather /
# Father) names a different relationship, however similar the letters are.
_QUALIFIERS = ("ex", "former", "late", "step", "half", "future", "fake", "adoptive", "adopted",
               "foster", "sworn", "co", "vice")


def _label_words(label: Any) -> List[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(label or "")).casefold())


def _differs_by_qualifier(a: Any, b: Any, ka: str, kb: str) -> bool:
    short_words, long_words = sorted((_label_words(a), _label_words(b)), key=len)
    if len(short_words) < len(long_words):
        extra = list(long_words)
        for word in short_words:
            if word in extra:
                extra.remove(word)
        if len(extra) == len(long_words) - len(short_words) and all(w in _QUALIFIERS for w in extra):
            return True
    short_key, long_key = sorted((ka, kb), key=len)
    return long_key.endswith(short_key) and long_key[: len(long_key) - len(short_key)] in _QUALIFIERS


def labels_equivalent(a: Any, b: Any) -> bool:
    ka, kb = _label_key(a), _label_key(b)
    if ka == kb:
        return True
    if not ka or not kb:
        return False
    if _differs_by_qualifier(a, b, ka, kb):
        return False
    return difflib.SequenceMatcher(None, ka, kb).ratio() >= _LABEL_SIMILARITY


# ---------------------------------------------------------------------------
# Ordering / normalisation
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _file_number(chapter_file: Any) -> Optional[int]:
    match = re.search(r"(\d+)", str(chapter_file or ""))
    return int(match.group(1)) if match else None


def _chapter_sort_key(occurrence: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Same order as the gender tracker: chapter_index, chapter_num, number in file name, file."""
    chapter_file = str(occurrence.get("chapter_file", "") or "")
    index = _num(occurrence.get("chapter_index"))
    number = _num(occurrence.get("chapter_num"))
    file_num = _file_number(chapter_file)
    return (
        _BIG if index is None else index,
        _BIG if number is None else number,
        _BIG if file_num is None else file_num,
        chapter_file,
    )


def _occurrence_sort_key(occurrence: Mapping[str, Any]) -> Tuple[Any, ...]:
    wording = str(occurrence.get("wording") or occurrence.get("label") or "")
    return _chapter_sort_key(occurrence) + (wording,)


def _chapter_signature(occurrence: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(occurrence.get("chapter_num")), str(occurrence.get("chapter_file", "") or ""))


def _occurrence_signature(occurrence: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (_label_key(occurrence.get("label")),) + _chapter_signature(occurrence)


def _normalize_relation(rel: Dict[str, Any]) -> None:
    occurrences = [dict(o) for o in rel.get("occurrences", []) if isinstance(o, dict) and str(o.get("label") or o.get("wording") or "").strip()]
    occurrences.sort(key=_occurrence_sort_key)

    # First wording wins, in chapter order (independent of arrival order).
    canonical: List[str] = []
    kept: List[Dict[str, Any]] = []
    seen = set()
    for occurrence in occurrences:
        wording = str(occurrence.get("wording") or occurrence.get("label") or "").strip()
        label = next((c for c in canonical if labels_equivalent(wording, c)), None)
        if label is None:
            label = wording
            canonical.append(label)
        occurrence["label"] = label
        if wording != label:
            occurrence["wording"] = wording
        else:
            occurrence.pop("wording", None)
        occurrence.pop("blip", None)
        occurrence.pop("flip", None)
        signature = _occurrence_signature(occurrence)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(occurrence)

    # Blip filter: one chapter of label X between two runs of the same label Y.
    chapters: List[Tuple[Tuple[str, str], str]] = []
    for occurrence in kept:
        signature = _chapter_signature(occurrence)
        if chapters and chapters[-1][0] == signature:
            chapters[-1] = (signature, occurrence["label"])
        else:
            chapters.append((signature, occurrence["label"]))
    runs: List[List[Any]] = []  # [label, [chapter signatures]]
    for signature, label in chapters:
        if runs and runs[-1][0] == label:
            runs[-1][1].append(signature)
        else:
            runs.append([label, [signature]])
    blips = set()
    for i in range(1, len(runs) - 1):
        label, sigs = runs[i]
        if len(sigs) == 1 and runs[i - 1][0] == runs[i + 1][0] and runs[i - 1][0] != label:
            blips.add((sigs[0], label))
    for occurrence in kept:
        if (_chapter_signature(occurrence), occurrence["label"]) in blips:
            occurrence["blip"] = True
    rel["occurrences"] = kept


def _main_labels(rel: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """(usual, first) label of an edge, blips left out: usual = seen in the most chapters
    (ties: the earlier one), first = the earliest one."""
    chapters: Dict[str, set] = {}
    first: Dict[str, int] = {}
    for position, occurrence in enumerate(rel.get("occurrences", []) or []):
        if not isinstance(occurrence, Mapping) or occurrence.get("blip") or not occurrence.get("label"):
            continue
        label = occurrence["label"]
        chapters.setdefault(label, set()).add(_chapter_signature(occurrence))
        first.setdefault(label, position)
    if not chapters:
        return None
    usual = min(chapters, key=lambda label: (-len(chapters[label]), first[label]))
    return usual, min(first, key=first.get)


def _mark_flips(entries: Mapping[str, Any]) -> None:
    """Reciprocal-flip filter: on edge A→B, a label equal to the reverse edge B→A's usual
    label is the model writing the wrong direction, not a change. It never fires when
    the two edges share a main label (Friend / Friend, even if one later turns Enemy).
    The main labels come from the sorted, blip-filtered occurrences before any flip is
    marked, so the result does not depend on arrival order."""
    mains: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}
    for key, item in entries.items():
        if isinstance(item, dict) and isinstance(item.get("relations"), dict):
            for tkey, rel in item["relations"].items():
                if isinstance(rel, dict):
                    mains[(key, tkey)] = _main_labels(rel)
    for (key, tkey), own in mains.items():
        reverse = mains.get((tkey, key))
        if not own or not reverse or any(labels_equivalent(a, b) for a in own for b in reverse):
            continue
        for occurrence in entries[key]["relations"][tkey].get("occurrences", []):
            label = occurrence.get("label")
            if (label and not occurrence.get("blip") and not labels_equivalent(label, own[0])
                    and labels_equivalent(label, reverse[0])):
                occurrence["flip"] = True


def _ignored(occurrence: Mapping[str, Any]) -> bool:
    return bool(occurrence.get("blip") or occurrence.get("flip"))


def _compute_changes(rel: Dict[str, Any]) -> None:
    changes: List[Dict[str, Any]] = []
    previous = None
    for occurrence in rel.get("occurrences", []):
        if _ignored(occurrence):
            continue
        if previous is not None and previous != occurrence["label"]:
            changes.append({
                "from": previous,
                "to": occurrence["label"],
                "chapter_index": occurrence.get("chapter_index"),
                "chapter_num": occurrence.get("chapter_num"),
                "chapter_file": occurrence.get("chapter_file", ""),
            })
        previous = occurrence["label"]
    rel["changes"] = changes


def _first_key_of(item: Mapping[str, Any]) -> Tuple[Any, ...]:
    best = None
    relations = item.get("relations", {}) if isinstance(item, Mapping) else {}
    if isinstance(relations, Mapping):
        for rel in relations.values():
            if not isinstance(rel, Mapping):
                continue
            for occurrence in rel.get("occurrences", []) or []:
                if isinstance(occurrence, Mapping):
                    key = _chapter_sort_key(occurrence)
                    if best is None or key < best:
                        best = key
    return best if best is not None else (_BIG, _BIG, _BIG, "")


def _normalize_descriptions(item: Dict[str, Any]) -> None:
    """One description per chapter, in chapter order (the key is dropped when there are none)."""
    descriptions = item.get("descriptions")
    if not isinstance(descriptions, list):
        item.pop("descriptions", None)
        return
    kept: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for description in descriptions:
        if isinstance(description, Mapping) and str(description.get("text") or "").strip():
            kept[_chapter_signature(description)] = dict(description)  # a later record of a chapter wins
    if kept:
        item["descriptions"] = sorted(kept.values(), key=_chapter_sort_key)
    else:
        item.pop("descriptions", None)


def normalize_tracker(tracker: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tracker, dict):
        return tracker
    entries = tracker.get("entries")
    if not isinstance(entries, dict):
        tracker["entries"] = {}
        return tracker
    ordered = []
    for key, item in entries.items():
        if isinstance(item, dict):
            item.setdefault("aliases", [])
            _normalize_descriptions(item)
            relations = item.get("relations")
            if not isinstance(relations, dict):
                relations = {}
            for rel in relations.values():
                if isinstance(rel, dict):
                    _normalize_relation(rel)
            item["relations"] = dict(sorted(
                relations.items(),
                key=lambda pair: (_first_key_of({"relations": {pair[0]: pair[1]}}), str(pair[0])),
            ))
        ordered.append((key, item))
    _mark_flips(entries)
    for _key, item in ordered:
        if isinstance(item, dict):
            for rel in item["relations"].values():
                if isinstance(rel, dict):
                    _compute_changes(rel)
    ordered.sort(key=lambda pair: (_first_key_of(pair[1]) if isinstance(pair[1], Mapping) else (_BIG, _BIG, _BIG, ""), str(pair[0])))
    tracker["entries"] = {key: item for key, item in ordered}
    tracker.setdefault("version", TRACKER_VERSION)
    return tracker


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _entry_field_keys(entry: Mapping[str, Any], field: str) -> List[str]:
    wanted = field.strip().casefold()
    return [k for k in entry.keys() if isinstance(k, str) and k.strip().casefold() == wanted]


def _change_signatures(tracker: Mapping[str, Any]) -> set:
    result = set()
    for key, item in (tracker.get("entries") or {}).items():
        if not isinstance(item, Mapping):
            continue
        for tkey, rel in (item.get("relations") or {}).items():
            if not isinstance(rel, Mapping):
                continue
            for change in rel.get("changes", []) or []:
                result.add((key, tkey, change.get("from"), change.get("to"),
                            str(change.get("chapter_num")), change.get("chapter_file", "")))
    return result


#: A description kept per chapter is cut here; the glossary file keeps the full text.
MAX_DESCRIPTION_CHARS = 600


def _entry_description(entry: Mapping[str, Any]) -> str:
    """The row's description cell, as one line."""
    for key in _entry_field_keys(entry, "description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text if len(text) <= MAX_DESCRIPTION_CHARS else text[:MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return ""


def _new_entry(raw_name: str, translated_name: str = "") -> Dict[str, Any]:
    return {"raw_name": raw_name, "translated_name": translated_name, "aliases": [], "relations": {}}


def _new_relation(target_raw: str, target_translated: str = "") -> Dict[str, Any]:
    return {"target_raw": target_raw, "target_translated": target_translated, "occurrences": [], "changes": []}


def record_relations(
    tracker: Dict[str, Any],
    entries: Iterable[Any],
    *,
    field: Optional[str] = None,
    chapter_index: Any = None,
    chapter_num: Any = None,
    chapter_file: str = "",
    source_path: Optional[str] = None,
    glossary_path: Optional[str] = None,
    pop_field: bool = True,
) -> Dict[str, Any]:
    """Record one chapter's relations cells into ``tracker`` (in place).

    A cell item ``Y: label`` on X's row says what X is to Y; it is stored on the
    reverse edge, ``entries[Y].relations[X] = label``. Filler labels are dropped.
    """
    stats: Dict[str, Any] = {"new": 0, "changed": 0, "changes": [], "pairs": 0, "descriptions": 0}
    if not isinstance(tracker, dict):
        return stats
    field = (field or "").strip() or field_name()
    tracker_entries = tracker.get("entries")
    if not isinstance(tracker_entries, dict):
        tracker_entries = {}
        tracker["entries"] = tracker_entries
    if source_path:
        tracker["source_path"] = source_path
    if glossary_path:
        tracker["glossary_path"] = glossary_path
    entry_list = [e for e in (entries or []) if isinstance(e, dict)]
    before = _change_signatures(tracker)
    chapter_file = os.path.basename(str(chapter_file or ""))
    edits = _edits(tracker)  # the user's corrections and ignores (None for an unedited tracker)

    batch_translated: Dict[str, str] = {}
    for entry in entry_list:
        raw = str(entry.get("raw_name") or "").strip()
        translated = str(entry.get("translated_name") or "").strip()
        if raw and translated:
            batch_translated.setdefault(tracker_key(raw), translated)

    for entry in entry_list:
        keys = _entry_field_keys(entry, field)
        cell = ""
        for k in keys:
            value = entry.get(k)
            if isinstance(value, str) and value.strip():
                cell = value
                break
        if pop_field:
            for k in keys:
                entry.pop(k, None)
        row_raw = str(entry.get("raw_name") or "").strip()
        if not row_raw or not cell:
            continue
        row_key = tracker_key(row_raw)
        for other_raw, label in parse_relations_cell(cell):
            if is_filler_label(label):
                continue
            other_key = tracker_key(other_raw)
            if not other_key or other_key == row_key:
                continue
            # The model writes, on X's row, "Y: what X is to Y". The tracker stores what the
            # target is to the entry, so the item is filed on Y's entry with X as the target.
            # Y may have no row in this response: its entry is still created (the glossary
            # sync fills in its translated name later).
            key, raw, tkey, target_raw = other_key, other_raw, row_key, row_raw
            if edits:
                label = _edit_verdict(edits, key, tkey, label, chapter_num, chapter_file)
                if label is None:
                    continue
            translated = batch_translated.get(key, "")
            item = tracker_entries.get(key)
            if not isinstance(item, dict):
                item = _new_entry(raw, translated)
                tracker_entries[key] = item
            item.setdefault("aliases", [])
            if translated and not item.get("translated_name"):
                item["translated_name"] = translated
            relations = item.setdefault("relations", {})
            rel = relations.get(tkey)
            if not isinstance(rel, dict):
                target_item = tracker_entries.get(tkey)
                target_translated = batch_translated.get(tkey) or (
                    str(target_item.get("translated_name") or "") if isinstance(target_item, Mapping) else "")
                rel = _new_relation(target_raw, target_translated)
                relations[tkey] = rel
                stats["new"] += 1
            elif not rel.get("target_translated") and batch_translated.get(tkey):
                rel["target_translated"] = batch_translated[tkey]
            rel.setdefault("occurrences", []).append({
                "label": label,
                "chapter_index": chapter_index,
                "chapter_num": chapter_num,
                "chapter_file": chapter_file,
            })
            stats["pairs"] += 1

    # The description each character was given in this chapter (the diagram shows the one
    # written at the chapter it is looking at). Kept for character rows and for any name
    # that has a relation; the glossary file still holds its own single description.
    involved = set(tracker_entries)
    for item in tracker_entries.values():
        if isinstance(item, Mapping):
            involved.update((item.get("relations") or {}).keys())
    for entry in entry_list:
        raw = str(entry.get("raw_name") or "").strip()
        text = _entry_description(entry)
        if not raw or not text:
            continue
        key = tracker_key(raw)
        if key not in involved and str(entry.get("type") or "").strip().casefold() != "character":
            continue
        if _description_blocked(edits, tracker_entries.get(key), key, chapter_num, chapter_file):
            continue
        item = tracker_entries.get(key)
        if not isinstance(item, dict):
            item = _new_entry(raw, batch_translated.get(key, ""))
            tracker_entries[key] = item
        elif batch_translated.get(key) and not item.get("translated_name"):
            item["translated_name"] = batch_translated[key]
        signature = (str(chapter_num), chapter_file)
        descriptions = item.setdefault("descriptions", [])
        if any(isinstance(d, Mapping) and _chapter_signature(d) == signature and d.get("text") == text
               for d in descriptions):
            continue
        descriptions.append({"text": text, "chapter_index": chapter_index, "chapter_num": chapter_num,
                             "chapter_file": chapter_file})
        stats["descriptions"] += 1

    normalize_tracker(tracker)
    for key, item in tracker_entries.items():
        if not isinstance(item, Mapping):
            continue
        for tkey, rel in (item.get("relations") or {}).items():
            if not isinstance(rel, Mapping):
                continue
            for change in rel.get("changes", []) or []:
                signature = (key, tkey, change.get("from"), change.get("to"),
                             str(change.get("chapter_num")), change.get("chapter_file", ""))
                if signature in before:
                    continue
                stats["changes"].append({
                    "raw_name": item.get("raw_name", key),
                    "target_raw": rel.get("target_raw", tkey),
                    "from": change.get("from"),
                    "to": change.get("to"),
                    "chapter_num": change.get("chapter_num"),
                    "chapter_file": change.get("chapter_file", ""),
                })
    stats["changed"] = len(stats["changes"])
    return stats


# ---------------------------------------------------------------------------
# Glossary sync (after dedup)
# ---------------------------------------------------------------------------

def _nfc_fold(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text or "")).strip().casefold()


def _name_words(name: Any) -> List[str]:
    return [w for w in re.split(r"[\s·・]+", unicodedata.normalize("NFC", str(name or "")).strip()) if w]


def _merge_occurrences(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    existing = {_occurrence_signature(o) for o in target.get("occurrences", []) if isinstance(o, Mapping)}
    for occurrence in source.get("occurrences", []) or []:
        if not isinstance(occurrence, Mapping):
            continue
        signature = _occurrence_signature(occurrence)
        if signature in existing:
            continue
        existing.add(signature)
        target.setdefault("occurrences", []).append(dict(occurrence))


def _add_aliases(target: Dict[str, Any], names: Iterable[Any], own_raw: Any) -> None:
    aliases = target.setdefault("aliases", [])
    own = tracker_key(own_raw)
    for name in names:
        name = str(name or "").strip()
        if name and tracker_key(name) != own and name not in aliases:
            aliases.append(name)


def sync_tracker_with_glossary(
    tracker: Dict[str, Any],
    glossary_entries: Iterable[Any],
    strip_honorifics: Optional[Callable[[str], str]] = None,
) -> Dict[str, int]:
    """Align tracker names with the saved glossary and re-key deduplicated names (in place)."""
    stats = {"rekeyed": 0, "merged": 0, "unlisted": 0}
    if not isinstance(tracker, dict) or not isinstance(tracker.get("entries"), dict):
        return stats
    canonical: Dict[str, Tuple[str, str]] = {}
    by_stripped: Dict[str, str] = {}
    by_translated: Dict[str, str] = {}

    def _stripped(raw: str) -> str:
        if strip_honorifics is None:
            return ""
        try:
            return _nfc_fold(strip_honorifics(raw))
        except Exception:
            return ""

    # Whole word of a multi-word character name -> keys (refinement turns 바알 into 바알 아르젠트).
    by_word_raw: Dict[str, set] = {}
    by_word_translated: Dict[str, set] = {}

    for entry in glossary_entries or []:
        if not isinstance(entry, Mapping):
            continue
        raw = str(entry.get("raw_name") or "").strip()
        if not raw:
            continue
        translated = str(entry.get("translated_name") or "").strip()
        key = tracker_key(raw)
        if key in canonical:
            continue
        canonical[key] = (raw, translated)
        stripped = _stripped(raw)
        if stripped:
            by_stripped.setdefault(stripped, key)
        if translated:
            by_translated.setdefault(_nfc_fold(translated), key)
        entry_type = str(entry.get("type") or "character").strip().casefold()
        if entry_type == "character":
            for name, index in ((raw, by_word_raw), (translated, by_word_translated)):
                words = _name_words(name)
                if len(words) > 1:
                    for word in words:
                        index.setdefault(_nfc_fold(word), set()).add(key)
    if not canonical:
        return stats

    def _by_word(name: str, index: Dict[str, set]) -> Optional[str]:
        if len(_name_words(name)) != 1:
            return None
        keys = index.get(_nfc_fold(name))
        return next(iter(keys)) if keys and len(keys) == 1 else None

    def _resolve(key: str, raw: str, translated: str) -> Optional[str]:
        if key in canonical:
            return key
        stripped = _stripped(raw)
        if stripped and stripped in by_stripped:
            return by_stripped[stripped]
        if translated and _nfc_fold(translated) in by_translated:
            return by_translated[_nfc_fold(translated)]
        # A one-word name that is a word of exactly one longer character name.
        for name in (raw, stripped):
            found = _by_word(name, by_word_raw) if name else None
            if found:
                return found
        return _by_word(translated, by_word_translated) if translated else None

    entries: Dict[str, Any] = tracker["entries"]
    unlisted_names = set()
    rekeyed: Dict[str, str] = {}  # old key -> glossary key (the user's edits follow the name)

    # 1. Entries.
    for key in list(entries.keys()):
        item = entries.get(key)
        if not isinstance(item, dict) or key in canonical:
            continue
        new_key = _resolve(key, str(item.get("raw_name") or key), str(item.get("translated_name") or ""))
        if new_key is None:
            if item.get("relations"):  # a name kept only for its descriptions is not an unlisted relation
                unlisted_names.add(key)
            continue
        entries.pop(key, None)
        rekeyed[key] = new_key
        survivor = entries.get(new_key)
        if not isinstance(survivor, dict):
            survivor = _new_entry(canonical[new_key][0], canonical[new_key][1])
            entries[new_key] = survivor
            stats["rekeyed"] += 1
        else:
            stats["merged"] += 1
        _add_aliases(survivor, [item.get("raw_name") or key] + list(item.get("aliases") or []), canonical[new_key][0])
        if item.get("descriptions"):
            # The survivor's own description of a chapter wins (it is appended last).
            survivor["descriptions"] = [dict(d) for d in item["descriptions"] if isinstance(d, Mapping)] + [
                d for d in survivor.get("descriptions") or [] if isinstance(d, Mapping)]
        relations = survivor.setdefault("relations", {})
        for tkey, rel in (item.get("relations") or {}).items():
            if not isinstance(rel, Mapping) or tkey == new_key:
                continue
            if isinstance(relations.get(tkey), dict):
                _merge_occurrences(relations[tkey], rel)
            else:
                relations[tkey] = {k: (list(v) if isinstance(v, list) else v) for k, v in rel.items()}

    # 2. Targets, and refresh names.
    for key, item in entries.items():
        if not isinstance(item, dict):
            continue
        if key in canonical:
            item["raw_name"], translated = canonical[key]
            if translated:
                item["translated_name"] = translated
        relations = item.get("relations")
        if not isinstance(relations, dict):
            continue
        for tkey in list(relations.keys()):
            rel = relations.get(tkey)
            if not isinstance(rel, dict):
                continue
            if tkey in canonical:
                continue
            target_item = entries.get(tkey)
            translated = str(rel.get("target_translated") or "") or (
                str(target_item.get("translated_name") or "") if isinstance(target_item, Mapping) else "")
            new_tkey = _resolve(tkey, str(rel.get("target_raw") or tkey), translated)
            if new_tkey is None:
                unlisted_names.add(tkey)
                continue
            relations.pop(tkey, None)
            rekeyed[tkey] = new_tkey
            if new_tkey == key:
                continue
            if isinstance(relations.get(new_tkey), dict):
                _merge_occurrences(relations[new_tkey], rel)
                _add_aliases(relations[new_tkey], [rel.get("target_raw") or tkey], canonical[new_tkey][0])
                stats["merged"] += 1
            else:
                moved = {k: (list(v) if isinstance(v, list) else v) for k, v in rel.items()}
                _add_aliases(moved, [rel.get("target_raw") or tkey], canonical[new_tkey][0])
                relations[new_tkey] = moved
                stats["rekeyed"] += 1
        for tkey, rel in relations.items():
            if isinstance(rel, dict) and tkey in canonical:
                rel["target_raw"], translated = canonical[tkey]
                if translated:
                    rel["target_translated"] = translated

    stats["unlisted"] = len(unlisted_names)
    if _edits(tracker) or VIEW_KEY in tracker:
        _rekey_edits(tracker, rekeyed)
        _apply_edits(tracker)  # names merged from an alias carry the AI's uncorrected labels
    normalize_tracker(tracker)
    return stats


# ---------------------------------------------------------------------------
# Hand edits (Relations & Gender Tracker)
# ---------------------------------------------------------------------------
#
# Every function edits a tracker dict in place and returns ``{"changed": bool, ...}``; none
# writes a file (wrap them in ``edit_tracker`` for that). Names are raw names or tracker keys
# (``tracker_key`` of either is the key). "What B is to A" is ``entries[A].relations[B]``.
#
# What the user corrected is remembered under the tracker's optional "edits" key, so a later
# extraction that repeats the AI's mistake does not bring it back:
#   corrections           [{entry, target, from, to}]  label ``from`` on entry->target is recorded as ``to``
#   ignored               [{entry, target, label[, chapter_num, chapter_file]}]  never recorded
#                         (with a chapter: only at that chapter)
#   ignored_pairs         [[a, b]]  nothing is recorded between the two, either way
#   removed               [key]     the character is not recorded again (entry or relation)
#   ignored_descriptions  [{entry, chapter_num, chapter_file}]
# Adding or editing by hand lifts the matching ignore. An unedited tracker has no "edits" key.

_EDIT_LISTS = ("corrections", "ignored", "ignored_pairs", "removed", "ignored_descriptions")


def _edits(tracker: Any, create: bool = False) -> Optional[Dict[str, Any]]:
    edits = tracker.get(EDITS_KEY) if isinstance(tracker, Mapping) else None
    if isinstance(edits, dict):
        return edits
    if not create or not isinstance(tracker, dict):
        return None
    edits = {}
    tracker[EDITS_KEY] = edits
    return edits


def _edit_list(edits: Dict[str, Any], name: str) -> List[Any]:
    value = edits.get(name)
    if not isinstance(value, list):
        value = []
        edits[name] = value
    return value


def _records(edits: Optional[Mapping[str, Any]], name: str) -> List[Any]:
    value = edits.get(name) if isinstance(edits, Mapping) else None
    return value if isinstance(value, list) else []


def _prune_optional_keys(tracker: Any) -> None:
    """Drop empty edit lists, an empty "edits" key and a view equal to the defaults."""
    if not isinstance(tracker, dict):
        return
    if EDITS_KEY in tracker:
        edits = tracker.get(EDITS_KEY)
        if isinstance(edits, dict):
            for name in [n for n, v in edits.items() if not v]:
                edits.pop(name, None)
        if not edits or not isinstance(edits, dict):
            tracker.pop(EDITS_KEY, None)
    if VIEW_KEY in tracker and view_settings(tracker) == _default_view():
        tracker.pop(VIEW_KEY, None)


def _clean_label(label: Any) -> str:
    return " ".join(str(label or "").split())


def _base(chapter_file: Any) -> str:
    return os.path.basename(str(chapter_file or ""))


def _same_chapter(num_a: Any, file_a: Any, num_b: Any, file_b: Any) -> bool:
    a, b = _num(num_a), _num(num_b)
    same = (a == b) if a is not None and b is not None else str(num_a) == str(num_b)
    fa, fb = _base(file_a), _base(file_b)
    return same and (not fa or not fb or fa == fb)


def _record_matches_chapter(record: Mapping[str, Any], chapter_num: Any, chapter_file: Any) -> bool:
    if "chapter_num" not in record and "chapter_file" not in record:
        return True  # an ignore for every chapter
    return _same_chapter(record.get("chapter_num"), record.get("chapter_file"), chapter_num, chapter_file)


def _pair(a: str, b: str) -> List[str]:
    return sorted([a, b])


def _is_removed(edits: Optional[Mapping[str, Any]], key: str) -> bool:
    return key in _records(edits, "removed")


def _pair_ignored(edits: Optional[Mapping[str, Any]], a: str, b: str) -> bool:
    wanted = _pair(a, b)
    return any(isinstance(p, list) and sorted(str(x) for x in p) == wanted for p in _records(edits, "ignored_pairs"))


def _on_edge(record: Any, key: str, tkey: str) -> bool:
    return isinstance(record, Mapping) and record.get("entry") == key and record.get("target") == tkey


def _corrected(edits: Optional[Mapping[str, Any]], key: str, tkey: str, label: str) -> str:
    current, seen = label, set()
    for _step in range(8):
        wanted = _label_key(current)
        if wanted in seen:
            break
        seen.add(wanted)
        hit = next((c for c in _records(edits, "corrections")
                    if _on_edge(c, key, tkey) and c.get("to") and labels_equivalent(c.get("from"), current)), None)
        if hit is None:
            break
        current = str(hit["to"])
    return current


def _label_ignored(edits: Optional[Mapping[str, Any]], key: str, tkey: str, label: Any,
                   chapter_num: Any, chapter_file: Any) -> bool:
    return any(_on_edge(r, key, tkey) and labels_equivalent(r.get("label"), label)
               and _record_matches_chapter(r, chapter_num, chapter_file)
               for r in _records(edits, "ignored"))


def _edit_verdict(edits: Optional[Mapping[str, Any]], key: str, tkey: str, label: Any,
                  chapter_num: Any = None, chapter_file: Any = "") -> Optional[str]:
    """The label recorded for ``label`` on entry ``key`` -> target ``tkey``, or None when dropped."""
    label = str(label or "")
    if not edits:
        return label
    if _is_removed(edits, key) or _is_removed(edits, tkey) or _pair_ignored(edits, key, tkey):
        return None
    corrected = _corrected(edits, key, tkey, label)
    for candidate in (label, corrected):
        if _label_ignored(edits, key, tkey, candidate, chapter_num, chapter_file):
            return None
    return corrected


def recorded_label(tracker: Optional[Mapping[str, Any]], raw_name: Any, target: Any, label: Any,
                   chapter_num: Any = None, chapter_file: Any = "") -> Optional[str]:
    """What ``record_relations`` keeps of "``target`` is ``label`` to ``raw_name``" given the
    user's edits: the corrected label, or None when an ignore drops it."""
    return _edit_verdict(_edits(tracker), tracker_key(raw_name), tracker_key(target), label,
                         chapter_num, _base(chapter_file))


def _description_blocked(edits: Optional[Mapping[str, Any]], item: Any, key: str,
                         chapter_num: Any, chapter_file: Any) -> bool:
    if _is_removed(edits, key):
        return True
    if any(isinstance(r, Mapping) and r.get("entry") == key
           and _same_chapter(r.get("chapter_num"), r.get("chapter_file"), chapter_num, chapter_file)
           for r in _records(edits, "ignored_descriptions")):
        return True
    if isinstance(item, Mapping):  # a description written by hand for this chapter stays
        for description in item.get("descriptions") or []:
            if (isinstance(description, Mapping) and description.get("edited")
                    and _same_chapter(description.get("chapter_num"), description.get("chapter_file"),
                                      chapter_num, chapter_file)):
                return True
    return False


def _rekey_edits(tracker: Dict[str, Any], rekeyed: Mapping[str, str]) -> None:
    """The glossary sync merged ``old`` into ``new``: the edits about ``old`` now name ``new``."""
    if not rekeyed:
        return
    edits = _edits(tracker)
    if edits:
        for name in ("corrections", "ignored"):
            for record in _records(edits, name):
                if isinstance(record, dict):
                    for field in ("entry", "target"):
                        if record.get(field) in rekeyed:
                            record[field] = rekeyed[record[field]]
        for record in _records(edits, "ignored_descriptions"):
            if isinstance(record, dict) and record.get("entry") in rekeyed:
                record["entry"] = rekeyed[record["entry"]]
        pairs = []
        for pair in _records(edits, "ignored_pairs"):
            if isinstance(pair, list) and len(pair) == 2:
                a, b = (rekeyed.get(str(x), str(x)) for x in pair)
                if a != b and _pair(a, b) not in pairs:
                    pairs.append(_pair(a, b))
        if "ignored_pairs" in edits:
            edits["ignored_pairs"] = pairs
        # "removed" is left alone: a removed name has no entry left to merge.
    view = tracker.get(VIEW_KEY)
    if isinstance(view, dict) and isinstance(view.get("hidden"), list):
        hidden = []
        for key in view["hidden"]:
            key = rekeyed.get(str(key), str(key))
            if key not in hidden:
                hidden.append(key)
        view["hidden"] = hidden


def _apply_edits(tracker: Dict[str, Any]) -> None:
    """Drop what the edits ignore and correct what they correct, everywhere in the tracker."""
    edits = _edits(tracker)
    entries = tracker.get("entries")
    if not edits or not isinstance(entries, dict):
        return
    for key in [k for k in entries if _is_removed(edits, k)]:
        entries.pop(key, None)
    for key, item in entries.items():
        if not isinstance(item, dict):
            continue
        relations = item.get("relations")
        if isinstance(relations, dict):
            for tkey in list(relations):
                rel = relations.get(tkey)
                if not isinstance(rel, dict):
                    continue
                if _is_removed(edits, tkey) or _pair_ignored(edits, key, tkey):
                    relations.pop(tkey, None)
                    continue
                kept, touched = [], False
                for occurrence in rel.get("occurrences", []) or []:
                    if not isinstance(occurrence, dict):
                        kept.append(occurrence)
                        continue
                    original = str(occurrence.get("wording") or occurrence.get("label") or "")
                    num, file = occurrence.get("chapter_num"), occurrence.get("chapter_file", "")
                    verdict = _edit_verdict(edits, key, tkey, original, num, file)
                    if verdict is not None and _label_key(verdict) == _label_key(original):
                        verdict = _edit_verdict(edits, key, tkey, occurrence.get("label"), num, file)
                    if verdict is None:
                        touched = True
                        continue
                    if _label_key(verdict) != _label_key(original):
                        occurrence = dict(occurrence, label=verdict)
                        occurrence.pop("wording", None)
                        touched = True
                    kept.append(occurrence)
                if touched:
                    rel["occurrences"] = kept
                    if not kept:
                        relations.pop(tkey, None)
        descriptions = item.get("descriptions")
        if isinstance(descriptions, list):
            item["descriptions"] = [
                d for d in descriptions
                if not (isinstance(d, Mapping) and not d.get("edited") and any(
                    isinstance(r, Mapping) and r.get("entry") == key
                    and _same_chapter(r.get("chapter_num"), r.get("chapter_file"),
                                      d.get("chapter_num"), d.get("chapter_file"))
                    for r in _records(edits, "ignored_descriptions")))]


def _entries_of(tracker: Dict[str, Any]) -> Dict[str, Any]:
    entries = tracker.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        tracker["entries"] = entries
    return entries


def _ensure_entry(tracker: Dict[str, Any], raw_name: Any, translated_name: Any = "") -> Tuple[str, Dict[str, Any]]:
    key = tracker_key(raw_name)
    entries = _entries_of(tracker)
    item = entries.get(key)
    translated = str(translated_name or "").strip()
    if not isinstance(item, dict):
        item = _new_entry(str(raw_name or "").strip(), translated)
        entries[key] = item
    elif translated and not item.get("translated_name"):
        item["translated_name"] = translated
    item.setdefault("relations", {})
    return key, item


def _relation_of(tracker: Mapping[str, Any], key: str, tkey: str) -> Optional[Dict[str, Any]]:
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    item = entries.get(key) if isinstance(entries, Mapping) else None
    relations = item.get("relations") if isinstance(item, Mapping) else None
    rel = relations.get(tkey) if isinstance(relations, Mapping) else None
    return rel if isinstance(rel, dict) else None


def _known_chapters(tracker: Mapping[str, Any]) -> List[Tuple[Any, Any, str]]:
    """``(chapter_index, chapter_num, chapter_file)`` of every recorded relation and description."""
    found = []
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    if not isinstance(entries, Mapping):
        return found
    for item in entries.values():
        if not isinstance(item, Mapping):
            continue
        records = list(_iter_occurrences({"entries": {"x": item}}))
        records += [d for d in item.get("descriptions") or [] if isinstance(d, Mapping)]
        for record in records:
            found.append((record.get("chapter_index"), record.get("chapter_num"), _base(record.get("chapter_file"))))
    return found


def _estimate_index(known: Sequence[Tuple[Any, Any, str]], number: Optional[float], step: float) -> Any:
    """A chapter index for a chapter number nothing was recorded at: just after the chapters
    before it (``step`` past the highest of their indexes), else just before the first one."""
    if number is None:
        return None
    lower = [_num(i) for i, n, _f in known if _num(i) is not None and _num(n) is not None and _num(n) < number]
    if lower:
        value = max(lower) + step
    else:
        higher = [_num(i) for i, n, _f in known if _num(i) is not None and _num(n) is not None and _num(n) > number]
        if not higher:
            return None
        value = min(higher) - step
    return int(value) if value == int(value) else value


def chapter_slot(tracker: Optional[Mapping[str, Any]], chapter: Any = None,
                 chapter_file: Any = "") -> Tuple[Any, Any, str]:
    """``(chapter_index, chapter_num, chapter_file)`` for a hand edit at ``chapter`` (None: the
    last chapter recorded). Index and file come from what was recorded at that chapter, so the
    edit sorts with the AI's records; a chapter with nothing recorded gets an index in between."""
    if chapter is None:
        chapter = chapter_bounds(tracker)[1]
    chapter_file = _base(chapter_file)
    number = _num(chapter)
    known = _known_chapters(tracker or {})
    same = [k for k in known if number is not None and _num(k[1]) == number
            and (not chapter_file or not k[2] or k[2] == chapter_file)]
    if same:
        same.sort(key=lambda k: (_BIG if _num(k[0]) is None else _num(k[0]), k[2]))
        index, _n, recorded_file = same[0]
        return index, _display_num(chapter), chapter_file or recorded_file
    return _estimate_index(known, number, 0.5), _display_num(chapter), chapter_file


def _lift_label_ignores(edits: Optional[Dict[str, Any]], key: str, tkey: str, label: Any,
                        chapter_num: Any = None, chapter_file: Any = "", every_chapter: bool = True) -> None:
    if not edits or "ignored" not in edits:
        return
    edits["ignored"] = [
        r for r in _records(edits, "ignored")
        if not (_on_edge(r, key, tkey) and labels_equivalent(r.get("label"), label)
                and (every_chapter or _record_matches_chapter(r, chapter_num, chapter_file)))]


def _lift_name(edits: Optional[Dict[str, Any]], *keys: str) -> None:
    if edits and "removed" in edits:
        edits["removed"] = [k for k in _records(edits, "removed") if k not in keys]


def _lift_pair(edits: Optional[Dict[str, Any]], a: str, b: str) -> None:
    if edits and "ignored_pairs" in edits:
        wanted = _pair(a, b)
        edits["ignored_pairs"] = [p for p in _records(edits, "ignored_pairs")
                                  if not (isinstance(p, list) and sorted(str(x) for x in p) == wanted)]


def _drop_correction_from(edits: Optional[Dict[str, Any]], key: str, tkey: str, label: Any) -> None:
    """A label the user wrote is never corrected into something else."""
    if edits and "corrections" in edits:
        wanted = _label_key(label)
        edits["corrections"] = [c for c in _records(edits, "corrections")
                                if not (_on_edge(c, key, tkey) and _label_key(c.get("from")) == wanted)]


def _add_ignore(edits: Dict[str, Any], key: str, tkey: str, label: Any,
                chapter_num: Any = None, chapter_file: Any = None, every_chapter: bool = True) -> None:
    label = _clean_label(label)
    if not label:
        return
    record: Dict[str, Any] = {"entry": key, "target": tkey, "label": label}
    if not every_chapter:
        record["chapter_num"] = chapter_num
        record["chapter_file"] = _base(chapter_file)
    ignored = _edit_list(edits, "ignored")
    for existing in ignored:
        if (_on_edge(existing, key, tkey) and _label_key(existing.get("label")) == _label_key(label)
                and ("chapter_num" in existing) == ("chapter_num" in record)
                and (every_chapter or _same_chapter(existing.get("chapter_num"), existing.get("chapter_file"),
                                                    chapter_num, chapter_file))):
            return
    ignored.append(record)


def _relation_label_now(tracker: Mapping[str, Any], rel: Optional[Mapping[str, Any]], chapter: Any,
                        fallback: bool = True) -> Optional[str]:
    """The label shown at ``chapter``; with ``fallback``, the latest one when none is shown yet."""
    if not isinstance(rel, Mapping):
        return None
    state = None
    if chapter is not None:
        state = _relation_state(rel, _visible_predicate(tracker, chapter_cutoff({"chapter_num": chapter})), "current")
    if state is None and (fallback or chapter is None):
        state = _relation_state(rel, lambda occurrence: True, "current")
    return state["label"] if state else None


def relation_labels(tracker: Optional[Mapping[str, Any]], a: Any, b: Any,
                    up_to_chapter: Any = None) -> Tuple[str, str]:
    """``(label_ab, label_ba)`` of a pair at ``up_to_chapter`` (what b is to a, what a is to b;
    "" for an unknown side): what an edit dialog starts from."""
    ka, kb = tracker_key(a), tracker_key(b)
    if not isinstance(tracker, Mapping):
        return "", ""
    return (_relation_label_now(tracker, _relation_of(tracker, ka, kb), up_to_chapter, False) or "",
            _relation_label_now(tracker, _relation_of(tracker, kb, ka), up_to_chapter, False) or "")


def _finish(tracker: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    normalize_tracker(tracker)
    _prune_optional_keys(tracker)
    return result


def add_relation(tracker: Dict[str, Any], raw_name: Any, target: Any, label: Any = None,
                 reverse_label: Any = None, *, chapter: Any = None, chapter_file: Any = "",
                 translated_name: Any = "", target_translated: Any = "") -> Dict[str, Any]:
    """Add "``target`` is ``label`` to ``raw_name``" and/or "``raw_name`` is ``reverse_label`` to
    ``target``" at ``chapter`` (None: the last chapter recorded), by hand.

    Either name may be new (its entry is created). Lifts the ignores this contradicts: a
    removed name, an ignored pair, an ignored label. Returns ``{"changed", "added": n}``.
    """
    result: Dict[str, Any] = {"changed": False, "added": 0}
    if not isinstance(tracker, dict):
        return result
    key, tkey = tracker_key(raw_name), tracker_key(target)
    sides = [(key, tkey, _clean_label(label), raw_name, translated_name, target, target_translated),
             (tkey, key, _clean_label(reverse_label), target, target_translated, raw_name, translated_name)]
    if not key or not tkey or key == tkey or not any(side[2] for side in sides):
        return result
    index, number, file = chapter_slot(tracker, chapter, chapter_file)
    edits = _edits(tracker)
    _lift_name(edits, key, tkey)
    _lift_pair(edits, key, tkey)
    entries = _entries_of(tracker)
    for own, other, text, own_raw, own_translated, other_raw, other_translated in sides:
        if not text:
            continue
        _own, item = _ensure_entry(tracker, own_raw, own_translated)
        relations = item.setdefault("relations", {})
        rel = relations.get(other)
        if not isinstance(rel, dict):
            other_item = entries.get(other)
            translated = str(other_translated or "").strip() or (
                str(other_item.get("translated_name") or "") if isinstance(other_item, Mapping) else "")
            rel = _new_relation(str(other_raw or "").strip(), translated)
            relations[other] = rel
        _lift_label_ignores(edits, own, other, text)
        _drop_correction_from(edits, own, other, text)
        occurrences = rel.setdefault("occurrences", [])
        if any(isinstance(o, Mapping) and _label_key(o.get("label")) == _label_key(text)
               and _same_chapter(o.get("chapter_num"), o.get("chapter_file"), number, file)
               for o in occurrences):
            continue
        occurrences.append({"label": text, "chapter_index": index, "chapter_num": number,
                            "chapter_file": file, "manual": True})
        result["added"] += 1
    result["changed"] = True
    return _finish(tracker, result)


def set_relation_label(tracker: Dict[str, Any], raw_name: Any, target: Any, label: Any, *,
                       old_label: Any = None, chapter: Any = None, chapter_file: Any = "",
                       translated_name: Any = "", target_translated: Any = "") -> Dict[str, Any]:
    """Set what ``target`` is to ``raw_name`` (one direction of a relation).

    ``old_label`` is the label being replaced (default: the one shown at ``chapter``); every
    record of it on this direction becomes ``label`` and the AI's wordings of it are
    remembered as corrections. ``label`` "" clears the side: those records are deleted and
    remembered as ignored. A direction with nothing recorded gets ``label`` added at
    ``chapter`` (``add_relation``). Returns ``{"changed", "old": replaced label, "count"}``.
    """
    result: Dict[str, Any] = {"changed": False, "old": None, "count": 0}
    if not isinstance(tracker, dict):
        return result
    key, tkey = tracker_key(raw_name), tracker_key(target)
    if not key or not tkey or key == tkey:
        return result
    new = _clean_label(label)
    rel = _relation_of(tracker, key, tkey)
    old = _clean_label(old_label) if old_label is not None else _relation_label_now(tracker, rel, chapter)
    old_key = _label_key(old) if old else ""
    occurrences = [o for o in (rel or {}).get("occurrences", []) if isinstance(o, dict)]
    matching = [o for o in occurrences if old_key and old_key in (_label_key(o.get("label")), _label_key(o.get("wording")))]
    if not matching:
        if not new:
            return result
        added = add_relation(tracker, raw_name, target, new, None, chapter=chapter, chapter_file=chapter_file,
                             translated_name=translated_name, target_translated=target_translated)
        return dict(result, changed=added["changed"], count=added["added"])
    result.update(old=old, count=len(matching))
    originals = []
    for occurrence in matching:
        for wording in (occurrence.get("wording"), occurrence.get("label")):
            wording = _clean_label(wording)
            if wording and wording not in originals:
                originals.append(wording)
    if old not in originals:
        originals.append(old)
    edits = _edits(tracker, create=True)
    if not new:
        rel["occurrences"] = [o for o in occurrences if not any(o is m for m in matching)]
        for wording in originals:
            _add_ignore(edits, key, tkey, wording)
        if not rel["occurrences"]:
            tracker["entries"][key]["relations"].pop(tkey, None)
        result["changed"] = True
        return _finish(tracker, result)
    if all(o.get("label") == new and "wording" not in o for o in matching) and old == new:
        return result
    for occurrence in matching:
        occurrence["label"] = new
        occurrence.pop("wording", None)
    corrections = _edit_list(edits, "corrections")
    for correction in corrections:  # an earlier correction to the old label now leads to the new one
        if _on_edge(correction, key, tkey) and _label_key(correction.get("to")) == old_key:
            correction["to"] = new
    for wording in originals:
        if _label_key(wording) == _label_key(new):
            continue
        hit = next((c for c in corrections if _on_edge(c, key, tkey) and _label_key(c.get("from")) == _label_key(wording)), None)
        if hit is None:
            corrections.append({"entry": key, "target": tkey, "from": wording, "to": new})
        else:
            hit["to"] = new
    edits["corrections"] = [c for c in corrections
                            if not (_on_edge(c, key, tkey) and _label_key(c.get("from")) == _label_key(c.get("to")))]
    _drop_correction_from(edits, key, tkey, new)
    _lift_label_ignores(edits, key, tkey, new)
    _lift_pair(edits, key, tkey)
    result["changed"] = True
    return _finish(tracker, result)


def edit_relation(tracker: Dict[str, Any], a: Any, b: Any, label_ab: Any = None, label_ba: Any = None, *,
                  old_ab: Any = None, old_ba: Any = None, chapter: Any = None, chapter_file: Any = "",
                  translated_a: Any = "", translated_b: Any = "") -> Dict[str, Any]:
    """Edit both directions of a pair: ``label_ab`` = what b is to a, ``label_ba`` = what a is to
    b (None: leave that side, "": clear it). See ``set_relation_label``.
    Returns ``{"changed", "ab": result, "ba": result}``."""
    result: Dict[str, Any] = {"changed": False, "ab": None, "ba": None}
    if label_ab is not None:
        result["ab"] = set_relation_label(tracker, a, b, label_ab, old_label=old_ab, chapter=chapter,
                                          chapter_file=chapter_file, translated_name=translated_a,
                                          target_translated=translated_b)
    if label_ba is not None:
        result["ba"] = set_relation_label(tracker, b, a, label_ba, old_label=old_ba, chapter=chapter,
                                          chapter_file=chapter_file, translated_name=translated_b,
                                          target_translated=translated_a)
    result["changed"] = bool((result["ab"] or {}).get("changed") or (result["ba"] or {}).get("changed"))
    return result


def delete_relation(tracker: Dict[str, Any], a: Any, b: Any) -> Dict[str, Any]:
    """Delete a pair, both directions, and ignore the pair from now on.
    Returns ``{"changed", "removed": records deleted}``."""
    result: Dict[str, Any] = {"changed": False, "removed": 0}
    if not isinstance(tracker, dict):
        return result
    ka, kb = tracker_key(a), tracker_key(b)
    if not ka or not kb or ka == kb:
        return result
    for own, other in ((ka, kb), (kb, ka)):
        rel = _relation_of(tracker, own, other)
        if rel is not None:
            result["removed"] += len([o for o in rel.get("occurrences", []) if isinstance(o, Mapping)])
            tracker["entries"][own]["relations"].pop(other, None)
            result["changed"] = True
    if not result["changed"]:
        return result
    edits = _edits(tracker, create=True)
    if not _pair_ignored(edits, ka, kb):
        _edit_list(edits, "ignored_pairs").append(_pair(ka, kb))
    for name in ("corrections", "ignored"):  # the pair's ignore covers them
        if name in edits:
            edits[name] = [r for r in _records(edits, name) if not (_on_edge(r, ka, kb) or _on_edge(r, kb, ka))]
    return _finish(tracker, result)


def delete_history_entry(tracker: Dict[str, Any], raw_name: Any, target: Any, label: Any,
                         from_chapter: Any = None, to_chapter: Any = None, chapter_file: Any = "") -> Dict[str, Any]:
    """Delete the records of ``label`` on "``target`` is ... to ``raw_name``" from ``from_chapter``
    to ``to_chapter`` (inclusive; None: unbounded; one chapter: both the same, ``chapter_file``
    narrows it) -- one history row of ``character_history`` (a timeline run or a blip). Each
    deleted record is ignored at its chapter, so re-extracting that chapter does not bring it
    back; later chapters may still record the label. Returns ``{"changed", "removed"}``."""
    result: Dict[str, Any] = {"changed": False, "removed": 0}
    if not isinstance(tracker, dict):
        return result
    key, tkey = tracker_key(raw_name), tracker_key(target)
    rel = _relation_of(tracker, key, tkey)
    wanted = _label_key(label)
    if rel is None or not wanted:
        return result
    low, high = _num(from_chapter), _num(to_chapter)
    chapter_file = _base(chapter_file)

    def _hit(occurrence: Mapping[str, Any]) -> bool:
        if wanted not in (_label_key(occurrence.get("label")), _label_key(occurrence.get("wording"))):
            return False
        number = _num(occurrence.get("chapter_num"))
        if low is not None and (number is None or number < low):
            return False
        if high is not None and (number is None or number > high):
            return False
        return not chapter_file or not occurrence.get("chapter_file") or _base(occurrence.get("chapter_file")) == chapter_file

    occurrences = [o for o in rel.get("occurrences", []) if isinstance(o, dict)]
    victims = [o for o in occurrences if _hit(o)]
    if not victims:
        return result
    edits = _edits(tracker, create=True)
    for occurrence in victims:
        for wording in {_clean_label(occurrence.get("label")), _clean_label(occurrence.get("wording"))} - {""}:
            _add_ignore(edits, key, tkey, wording, occurrence.get("chapter_num"),
                        occurrence.get("chapter_file"), every_chapter=False)
    rel["occurrences"] = [o for o in occurrences if not any(o is v for v in victims)]
    if not rel["occurrences"]:
        tracker["entries"][key]["relations"].pop(tkey, None)
    result.update(changed=True, removed=len(victims))
    return _finish(tracker, result)


def remove_character(tracker: Dict[str, Any], raw_name: Any) -> Dict[str, Any]:
    """Remove a character (a junk entry such as "Older Sibling"): its entry and every relation
    that points at it; it is not recorded again. Returns ``{"changed", "relations": removed}``."""
    result: Dict[str, Any] = {"changed": False, "relations": 0}
    if not isinstance(tracker, dict):
        return result
    key = tracker_key(raw_name)
    entries = _entries_of(tracker)
    if not key:
        return result
    item = entries.pop(key, None)
    found = isinstance(item, Mapping)
    if found:
        result["relations"] += len([r for r in (item.get("relations") or {}).values() if isinstance(r, Mapping)])
    for other in entries.values():
        relations = other.get("relations") if isinstance(other, dict) else None
        if isinstance(relations, dict) and key in relations:
            relations.pop(key, None)
            result["relations"] += 1
            found = True
    edits = _edits(tracker)
    if not found and _is_removed(edits, key):
        return result
    # A name with no entry (one only the gender tracker knows) is still remembered as removed:
    # it is no longer drawn and later extractions do not add it.
    edits = _edits(tracker, create=True)
    if key not in _records(edits, "removed"):
        _edit_list(edits, "removed").append(key)
    for name in ("corrections", "ignored"):  # covered by the removal
        if name in edits:
            edits[name] = [r for r in _records(edits, name)
                           if not (isinstance(r, Mapping) and key in (r.get("entry"), r.get("target")))]
    if "ignored_pairs" in edits:
        edits["ignored_pairs"] = [p for p in _records(edits, "ignored_pairs") if not (isinstance(p, list) and key in p)]
    if "ignored_descriptions" in edits:
        edits["ignored_descriptions"] = [r for r in _records(edits, "ignored_descriptions")
                                         if not (isinstance(r, Mapping) and r.get("entry") == key)]
    view = tracker.get(VIEW_KEY)
    if isinstance(view, dict) and isinstance(view.get("hidden"), list):
        view["hidden"] = [h for h in view["hidden"] if h != key]
    result["changed"] = True
    return _finish(tracker, result)


def set_description(tracker: Dict[str, Any], raw_name: Any, text: Any, chapter: Any = None, *,
                    chapter_file: Any = "", translated_name: Any = "") -> Dict[str, Any]:
    """Write the description of ``raw_name`` at ``chapter`` (None: the last chapter recorded) by
    hand; a later extraction of that chapter keeps it. "" deletes it (``delete_description``).
    Returns ``{"changed"}``."""
    result: Dict[str, Any] = {"changed": False}
    if not isinstance(tracker, dict) or not tracker_key(raw_name):
        return result
    text = " ".join(str(text or "").split())
    if chapter is None:
        chapter = chapter_bounds(tracker)[1]
    if not text:
        return delete_description(tracker, raw_name, chapter, chapter_file=chapter_file)
    edits = _edits(tracker)
    key = tracker_key(raw_name)
    _lift_name(edits, key)
    _key, item = _ensure_entry(tracker, raw_name, translated_name)
    descriptions = item.setdefault("descriptions", [])
    found = next((d for d in descriptions if isinstance(d, dict)
                  and _same_chapter(d.get("chapter_num"), d.get("chapter_file"), chapter, chapter_file)), None)
    if found is not None:
        if found.get("text") == text and found.get("edited"):
            return result
        found["text"] = text
        found["edited"] = True
    else:
        index, number, file = chapter_slot(tracker, chapter, chapter_file)
        descriptions.append({"text": text, "chapter_index": index, "chapter_num": number,
                             "chapter_file": file, "edited": True})
    if edits and "ignored_descriptions" in edits:
        edits["ignored_descriptions"] = [
            r for r in _records(edits, "ignored_descriptions")
            if not (isinstance(r, Mapping) and r.get("entry") == key
                    and _same_chapter(r.get("chapter_num"), r.get("chapter_file"), chapter, chapter_file))]
    result["changed"] = True
    return _finish(tracker, result)


def delete_description(tracker: Dict[str, Any], raw_name: Any, chapter: Any, *,
                       chapter_file: Any = "") -> Dict[str, Any]:
    """Delete the description of ``raw_name`` written at ``chapter``; that chapter's description
    is not recorded again. Returns ``{"changed", "removed"}``."""
    result: Dict[str, Any] = {"changed": False, "removed": 0}
    if not isinstance(tracker, dict):
        return result
    key = tracker_key(raw_name)
    item = _entries_of(tracker).get(key)
    descriptions = item.get("descriptions") if isinstance(item, dict) else None
    if not isinstance(descriptions, list):
        return result
    victims = [d for d in descriptions if isinstance(d, Mapping)
               and _same_chapter(d.get("chapter_num"), d.get("chapter_file"), chapter, chapter_file)]
    if not victims:
        return result
    item["descriptions"] = [d for d in descriptions if not any(d is v for v in victims)]
    edits = _edits(tracker, create=True)
    ignored = _edit_list(edits, "ignored_descriptions")
    for description in victims:
        record = {"entry": key, "chapter_num": description.get("chapter_num"),
                  "chapter_file": _base(description.get("chapter_file"))}
        if record not in ignored:
            ignored.append(record)
    result.update(changed=True, removed=len(victims))
    return _finish(tracker, result)


# ---------------------------------------------------------------------------
# Reading at a chapter
# ---------------------------------------------------------------------------

def chapter_cutoff(chapter_ref: Any, tracker: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The chapter bound of a request, or None when there is no usable chapter.

    ``{"chapter_num": float|None, "chapter_file": basename, "start": float|None,
    "position": int|None, "label": shown in the log}``.

    Same order as ``gender_tracking.automatic_chapter_gender``: when ``tracker`` has
    the request's own chapter file, that file decides (extraction and translation may
    number chapters differently). A single chapter then sees everything up to and
    including that file in the tracker's chapter order (``position``); a merged group
    shifts its last chapter number by the file's numbering offset. Otherwise the
    chapter number decides, and a file-only reference is placed by the number in its
    name. ``chapter_num`` / ``start`` are in the tracker's numbering.
    """
    if chapter_ref is None or chapter_ref == "" or chapter_ref == {}:
        return None
    if isinstance(chapter_ref, Mapping):
        if chapter_ref.get("use_storage_gender"):
            return None
        start = _num(chapter_ref.get("chapter_num"))
        end = _num(chapter_ref.get("chapter_num_end"))
        chapter_file = os.path.basename(str(chapter_ref.get("chapter_file") or ""))
    else:
        start = _num(chapter_ref)
        end = None
        chapter_file = ""
    bound = end if end is not None else start
    if bound is None and not chapter_file:
        return None
    cutoff: Dict[str, Any] = {"chapter_num": bound, "chapter_file": chapter_file, "start": start,
                              "position": None, "label": _display_num(bound) if bound is not None else None}
    matched = _exact_file_chapter(tracker, chapter_file) if tracker is not None else None
    if matched is not None:
        position, file_num = matched
        if end is None:
            cutoff.update({"chapter_num": None, "start": None, "position": position})
            if cutoff["label"] is None:
                cutoff["label"] = _display_num(file_num) if file_num is not None else chapter_file
            return cutoff
        if start is not None and file_num is not None:
            offset = file_num - start
            cutoff.update({"chapter_num": end + offset, "start": file_num})
        return cutoff
    if bound is None and tracker is not None:
        position = _file_position(tracker, chapter_file)
        if position is None:
            return None
        cutoff["position"] = position
    return cutoff


def _iter_occurrences(tracker: Optional[Mapping[str, Any]]):
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    if not isinstance(entries, Mapping):
        return
    for item in entries.values():
        if not isinstance(item, Mapping):
            continue
        for rel in (item.get("relations") or {}).values():
            if not isinstance(rel, Mapping):
                continue
            for occurrence in rel.get("occurrences", []) or []:
                if isinstance(occurrence, Mapping):
                    yield occurrence


def _chapter_list(tracker: Optional[Mapping[str, Any]]) -> List[Tuple[Tuple[Any, ...], Tuple[str, str], str]]:
    chapters = {}
    for occurrence in _iter_occurrences(tracker):
        signature = _chapter_signature(occurrence)
        if signature not in chapters:
            chapters[signature] = (_chapter_sort_key(occurrence), signature, str(occurrence.get("chapter_file", "") or ""))
    return sorted(chapters.values(), key=lambda c: (c[0], c[1]))


def _exact_file_chapter(tracker: Optional[Mapping[str, Any]], chapter_file: str) -> Optional[Tuple[int, Optional[float]]]:
    """``(position, chapter_num)`` of the last tracker chapter recorded from exactly ``chapter_file``."""
    if not chapter_file:
        return None
    chapters = _chapter_list(tracker)
    found = [i for i, c in enumerate(chapters) if os.path.basename(c[2]) == chapter_file]
    if not found:
        return None
    position = found[-1]
    return position, _num(chapters[position][1][0])


def _file_position(tracker: Optional[Mapping[str, Any]], chapter_file: str) -> Optional[int]:
    """Index (in the tracker's chapter order) of the last chapter at or before ``chapter_file``."""
    if not chapter_file:
        return None
    chapters = _chapter_list(tracker)
    found = [i for i, c in enumerate(chapters) if os.path.basename(c[2]) == chapter_file]
    if found:
        return found[-1]
    wanted = _file_number(chapter_file)
    if wanted is None:
        return None
    position = -1
    for i, chapter in enumerate(chapters):
        number = _file_number(chapter[2])
        if number is not None and number <= wanted:
            position = i
    return position


def _visible_predicate(tracker: Optional[Mapping[str, Any]], cutoff: Optional[Mapping[str, Any]]) -> Callable[[Mapping[str, Any]], bool]:
    if cutoff is None:
        return lambda occurrence: True
    bound = cutoff.get("chapter_num")
    if bound is not None:
        def _by_num(occurrence: Mapping[str, Any]) -> bool:
            number = _num(occurrence.get("chapter_num"))
            return number is not None and number <= bound
        return _by_num
    position = cutoff.get("position")
    if position is None:
        position = _file_position(tracker, str(cutoff.get("chapter_file") or ""))
    if position is None:
        return lambda occurrence: False
    allowed = {c[1] for c in _chapter_list(tracker)[: position + 1]}
    return lambda occurrence: _chapter_signature(occurrence) in allowed


def _relation_state(rel: Mapping[str, Any], visible: Callable[[Mapping[str, Any]], bool], mode: str) -> Optional[Dict[str, Any]]:
    occurrences = [o for o in rel.get("occurrences", []) or []
                   if isinstance(o, Mapping) and not _ignored(o) and o.get("label") and visible(o)]
    if not occurrences:
        return None
    runs: List[List[Any]] = []  # [label, first occurrence]
    for occurrence in occurrences:
        if runs and runs[-1][0] == occurrence["label"]:
            continue
        runs.append([occurrence["label"], occurrence])
    current = runs[-1][0]
    if mode == "current":
        former: List[str] = []
    elif mode == "full":
        former = []
        for label, _occ in reversed(runs[:-1]):
            if label != current and label not in former:
                former.append(label)
    else:
        former = [runs[-2][0]] if len(runs) > 1 else []
    return {
        "label": current,
        "former": former,
        "changed_at": runs[-1][1].get("chapter_num") if len(runs) > 1 else None,
        "since": runs[-1][1].get("chapter_num"),
        "changed_file": runs[-1][1].get("chapter_file", "") if len(runs) > 1 else "",
    }


def _relations_visible(tracker: Optional[Mapping[str, Any]], raw_name: Any,
                       visible: Callable[[Mapping[str, Any]], bool], mode: str) -> List[Dict[str, Any]]:
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    if not isinstance(entries, Mapping):
        return []
    item = entries.get(tracker_key(raw_name))
    if not isinstance(item, Mapping):
        return []
    mode = mode if mode in HISTORY_MODES else DEFAULT_HISTORY_MODE
    result = []
    for tkey, rel in (item.get("relations") or {}).items():
        if not isinstance(rel, Mapping):
            continue
        state = _relation_state(rel, visible, mode)
        if state is None:
            continue
        target_item = entries.get(tkey)
        target_translated = str(rel.get("target_translated") or "") or (
            str(target_item.get("translated_name") or "") if isinstance(target_item, Mapping) else "")
        state.update({
            "target_key": tkey,
            "target_raw": rel.get("target_raw") or tkey,
            "target_translated": target_translated,
        })
        result.append(state)
    return result


def relations_at(tracker: Optional[Mapping[str, Any]], raw_name: Any, chapter_ref: Any,
                 mode: str = DEFAULT_HISTORY_MODE) -> List[Dict[str, Any]]:
    """What ``raw_name``'s relations are at ``chapter_ref`` (never uses a later chapter)."""
    cutoff = chapter_cutoff(chapter_ref, tracker)
    if cutoff is None:
        return []
    return _relations_visible(tracker, raw_name, _visible_predicate(tracker, cutoff), mode)


def _clean_for_render(text: str) -> str:
    return str(text or "").replace(";", ",").strip()


def render_relations(relations: Sequence[Mapping[str, Any]],
                     name_for: Optional[Callable[[str], Optional[str]]] = None) -> str:
    parts = []
    for rel in relations or []:
        if not isinstance(rel, Mapping) or not rel.get("label"):
            continue
        target_raw = str(rel.get("target_raw") or "")
        if name_for is not None:
            name = name_for(target_raw)
        else:
            name = rel.get("target_translated") or target_raw
        if not name:
            continue
        text = f"{_clean_for_render(name)}: {_clean_for_render(rel['label'])}"
        former = [str(f) for f in rel.get("former") or [] if f]
        if former:
            text += f" [formerly {', '.join(former)}]"
        parts.append(text)
    return "; ".join(parts)


def character_history(tracker: Optional[Mapping[str, Any]], raw_name: Any,
                      up_to_chapter: Any = None) -> List[Dict[str, Any]]:
    """Per-target timeline: runs of labels with their chapter span, plus blips.

    With ``up_to_chapter`` it stops there, like the diagram it is shown beside: nothing
    from a later chapter, and a relation first seen later is left out.
    """
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    if not isinstance(entries, Mapping):
        return []
    item = entries.get(tracker_key(raw_name))
    if not isinstance(item, Mapping):
        return []
    if up_to_chapter is None:
        visible: Callable[[Mapping[str, Any]], bool] = lambda occurrence: True  # noqa: E731
    else:
        visible = _visible_predicate(tracker, chapter_cutoff({"chapter_num": up_to_chapter}))
    result = []
    for tkey, rel in (item.get("relations") or {}).items():
        if not isinstance(rel, Mapping):
            continue
        timeline: List[Dict[str, Any]] = []
        blips = []
        for occurrence in rel.get("occurrences", []) or []:
            if not isinstance(occurrence, Mapping) or not occurrence.get("label") or not visible(occurrence):
                continue
            if _ignored(occurrence):  # blips and direction flips
                blips.append({"label": occurrence["label"], "chapter_num": occurrence.get("chapter_num"),
                              "chapter_file": occurrence.get("chapter_file", ""),
                              **({"flip": True} if occurrence.get("flip") else {})})
                continue
            if timeline and timeline[-1]["label"] == occurrence["label"]:
                timeline[-1]["to_chapter"] = occurrence.get("chapter_num")
                timeline[-1]["to_file"] = occurrence.get("chapter_file", "")
                timeline[-1]["count"] += 1
                continue
            timeline.append({
                "label": occurrence["label"],
                "from_chapter": occurrence.get("chapter_num"),
                "from_file": occurrence.get("chapter_file", ""),
                "to_chapter": occurrence.get("chapter_num"),
                "to_file": occurrence.get("chapter_file", ""),
                "count": 1,
            })
        if not timeline and not blips and up_to_chapter is not None:
            continue  # nothing of this relation had been seen by then
        target_item = entries.get(tkey)
        result.append({
            "target_key": tkey,
            "target_raw": rel.get("target_raw") or tkey,
            "target_translated": str(rel.get("target_translated") or "") or (
                str(target_item.get("translated_name") or "") if isinstance(target_item, Mapping) else ""),
            "current": timeline[-1]["label"] if timeline else "",
            "timeline": timeline,
            "changes": [dict(c) for c in rel.get("changes", []) or [] if isinstance(c, Mapping) and visible(c)],
            "blips": blips,
        })
    return result


def description_history(tracker: Optional[Mapping[str, Any]], raw_name: Any) -> List[Dict[str, Any]]:
    """Every description recorded for ``raw_name``, in chapter order: ``{text, chapter_num, chapter_file}``."""
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    item = entries.get(tracker_key(raw_name)) if isinstance(entries, Mapping) else None
    if not isinstance(item, Mapping):
        return []
    found = [d for d in item.get("descriptions") or [] if isinstance(d, Mapping) and str(d.get("text") or "").strip()]
    return [{"text": str(d["text"]), "chapter_num": d.get("chapter_num"), "chapter_file": d.get("chapter_file", "")}
            for d in sorted(found, key=_chapter_sort_key)]


def description_at(tracker: Optional[Mapping[str, Any]], raw_name: Any,
                   up_to_chapter: Any = None) -> Optional[Dict[str, Any]]:
    """The description written for ``raw_name`` at ``up_to_chapter`` or, failing that, the last
    one before it (never a later chapter's). ``None`` chapter: the latest one recorded."""
    history = description_history(tracker, raw_name)
    if up_to_chapter is not None:
        limit = _num(up_to_chapter)
        if limit is None:
            return None
        history = [d for d in history if _num(d.get("chapter_num")) is not None and _num(d["chapter_num"]) <= limit]
    return history[-1] if history else None


def descriptions_shown(tracker: Optional[Mapping[str, Any]], raw_name: Any, up_to_chapter: Any = None,
                       glossary_entries: Optional[Iterable[Any]] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str]:
    """What a diagram shows for a character: ``(written, earlier, source)``.

    ``written`` is the description written at ``up_to_chapter`` (or the last one before it),
    ``earlier`` the ones before that, newest first. ``source`` is ``"chapter"`` for those. A
    tracker made before descriptions were recorded has none: then ``written`` is the
    glossary's present description (no chapter) and ``source`` is ``"glossary"``. Nothing
    at all gives ``(None, [], "")``; nothing yet at that chapter ``(None, [], "chapter")``.
    """
    recorded = description_history(tracker, raw_name)
    if not recorded:
        key = tracker_key(raw_name)
        for entry in glossary_entries or []:
            if isinstance(entry, Mapping) and tracker_key(entry.get("raw_name")) == key:
                text = " ".join(str(entry.get("description") or "").split())
                if text:
                    return {"text": text, "chapter_num": None, "chapter_file": ""}, [], "glossary"
        return None, [], ""
    written = description_at(tracker, raw_name, up_to_chapter)
    if written is None:
        return None, [], "chapter"
    cut = recorded.index(written)
    return written, list(reversed(recorded[:cut])), "chapter"


def chapter_bounds(tracker: Optional[Mapping[str, Any]]) -> Tuple[Any, Any]:
    numbers = [_num(o.get("chapter_num")) for o in _iter_occurrences(tracker)]
    numbers = [n for n in numbers if n is not None]
    if not numbers:
        return (None, None)
    return (_display_num(min(numbers)), _display_num(max(numbers)))


def diagram_chapter_bounds(tracker: Optional[Mapping[str, Any]],
                           gender_tracker: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[int], Optional[int]]:
    """The diagram slider's whole-chapter range: the relations tracker's chapters and the gender
    tracker's (desktop and mobile use the same range)."""
    numbers = [_num(n) for n in chapter_bounds(tracker)]
    entries = gender_tracker.get("entries") if isinstance(gender_tracker, Mapping) else None
    for item in (entries.values() if isinstance(entries, Mapping) else ()):
        for occurrence in (item.get("occurrences") or []) if isinstance(item, Mapping) else []:
            if isinstance(occurrence, Mapping):
                numbers.append(_num(occurrence.get("chapter_num")))
    numbers = [n for n in numbers if n is not None]
    if not numbers:
        return None, None
    return int(math.floor(min(numbers))), int(math.ceil(max(numbers)))


# ---------------------------------------------------------------------------
# Translation-time injection
# ---------------------------------------------------------------------------

def _display_num(value: Any) -> Any:
    number = _num(value)
    if number is None:
        return value
    return int(number) if number == int(number) else number


def _split_line_ending(line: str) -> Tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def _token_identity(line: str) -> Tuple[str, str]:
    """(raw, translated) of a token-efficient ``* raw = translated ...`` line (glossary_usage's parser)."""
    from glossary_usage import _parse_token_entry_line
    parsed = _parse_token_entry_line(line)
    if not parsed:
        return "", ""
    return str(parsed.get("raw_name") or ""), str(parsed.get("translated_name") or "")


def _detect_format(text: str) -> str:
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if any(l.startswith("===") for l in lines) or any(l.lower().startswith("glossary columns:") for l in lines):
        return "token"
    if lines and "raw_name" in lines[0].lower():
        return "csv"
    return ""


def _json_raw(value: Mapping[str, Any], key: Any = "") -> str:
    return str(value.get("raw_name") or value.get("original_name") or value.get("original") or key or "").strip()


def _json_translated(value: Mapping[str, Any]) -> str:
    return str(value.get("translated_name") or value.get("translation") or value.get("translated")
               or value.get("name") or "").strip()


def _json_items(data: Any) -> List[Tuple[Dict[str, Any], str]]:
    items: List[Tuple[Dict[str, Any], str]] = []
    if isinstance(data, list):
        for value in data:
            if isinstance(value, dict):
                items.append((value, _json_raw(value)))
    elif isinstance(data, dict):
        entries = data.get("entries")
        if isinstance(entries, list):  # {"metadata": {...}, "entries": [...]}
            return _json_items(entries)
        container = entries if isinstance(entries, dict) else data
        for key, value in container.items():
            if key == "metadata":
                continue
            if isinstance(value, dict):
                items.append((value, _json_raw(value, key)))
            elif isinstance(value, list):  # {"characters": [...], "terms": [...]}
                for item in value:
                    if isinstance(item, dict):
                        items.append((item, _json_raw(item)))
    return items


def _csv_split(line: str, sep: str) -> List[str]:
    if sep != ",":
        return line.split(sep)
    try:
        return next(csv.reader([line]))
    except Exception:
        return line.split(",")


def _csv_join(cells: List[str], sep: str) -> str:
    if sep != ",":
        return sep.join(cells)
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(cells)
    return buffer.getvalue()


def inject_relations_column(
    glossary_text: Any,
    glossary_path: Optional[str],
    chapter_ref: Any = None,
    settings: Optional[Mapping[str, Any]] = None,
    tracker: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Add the relations column (known up to the request's chapter) to the entries in ``glossary_text``."""
    disabled = {"enabled": False}
    if not tracking_enabled(settings) or not isinstance(glossary_text, str) or not glossary_text.strip():
        return glossary_text, disabled
    if tracker is None:
        tracker = load_tracker(tracker_path_for_glossary(glossary_path or ""))
    if not isinstance(tracker, Mapping) or not tracker.get("entries"):
        return glossary_text, disabled
    cutoff = chapter_cutoff(chapter_ref, tracker)
    if cutoff is None:
        return glossary_text, disabled

    field = field_name(settings)
    mode = history_mode(settings)
    visible = _visible_predicate(tracker, cutoff)
    if cutoff.get("label") is not None:
        cutoff_label = cutoff["label"]
    elif cutoff.get("chapter_num") is not None:
        cutoff_label = _display_num(cutoff["chapter_num"])
    else:
        chapters = _chapter_list(tracker)[: (cutoff.get("position") or 0) + 1]
        nums = [_num(c[1][0]) for c in chapters if _num(c[1][0]) is not None]
        cutoff_label = _display_num(max(nums)) if nums else cutoff.get("chapter_file")
    stats: Dict[str, Any] = {"enabled": True, "entries": 0, "relations": 0, "cutoff": cutoff_label, "changed_here": []}

    start = cutoff.get("start")
    end = cutoff.get("chapter_num")

    def _changed_here(state: Mapping[str, Any]) -> bool:
        if state.get("changed_at") is None:
            return False
        if start is not None and end is not None:
            number = _num(state.get("changed_at"))
            return number is not None and start <= number <= end
        chapter_file = cutoff.get("chapter_file")
        return bool(chapter_file) and os.path.basename(str(state.get("changed_file") or "")) == chapter_file

    fmt = _detect_format(glossary_text)
    if not fmt:
        return glossary_text, stats

    # Collect the entries present in the text: (raw, translated).
    names: Dict[str, str] = {}
    data: Any = None
    lines: List[str] = []
    csv_info: Dict[str, Any] = {}
    if fmt == "json":
        data = json.loads(glossary_text.strip())
        for value, raw in _json_items(data):
            if raw:
                names.setdefault(tracker_key(raw), _json_translated(value) or raw)
    elif fmt == "token":
        lines = glossary_text.splitlines(True)
        for line in lines:
            body, _end = _split_line_ending(line)
            if body.strip().startswith("* "):
                raw, translated = _token_identity(body)
                if raw:
                    names.setdefault(tracker_key(raw), translated or raw)
    else:
        lines = glossary_text.splitlines(True)
        header_index = next(i for i, l in enumerate(lines) if l.strip())
        header_body, _end = _split_line_ending(lines[header_index])
        sep = "\x1f" if "\x1f" in glossary_text else ","
        header = [c.strip() for c in _csv_split(header_body, sep)]
        lowered = [c.lower() for c in header]
        raw_idx = lowered.index("raw_name") if "raw_name" in lowered else 1
        trans_idx = lowered.index("translated_name") if "translated_name" in lowered else 2
        csv_info = {"header_index": header_index, "sep": sep, "header": header,
                    "raw_idx": raw_idx, "trans_idx": trans_idx}
        for line in lines[header_index + 1:]:
            body, _end = _split_line_ending(line)
            if not body.strip():
                continue
            cells = _csv_split(body, sep)
            raw = cells[raw_idx].strip() if len(cells) > raw_idx else ""
            translated = cells[trans_idx].strip() if len(cells) > trans_idx else ""
            if raw:
                names.setdefault(tracker_key(raw), translated or raw)

    def _name_for(target_raw: str) -> Optional[str]:
        return names.get(tracker_key(target_raw))

    rendered_cache: Dict[str, str] = {}

    def _rendered(raw: str, translated: str, paren_safe: bool = False) -> str:
        key = tracker_key(raw)
        if key in rendered_cache:
            return rendered_cache[key]
        states = [s for s in _relations_visible(tracker, raw, visible, mode) if _name_for(s["target_raw"])]
        text = ""
        if states:
            if paren_safe:
                def _safe(value: Optional[str]) -> Optional[str]:
                    return value.replace("(", "[").replace(")", "]") if value else value
                safe_states = [dict(s, label=_safe(s["label"]), former=[_safe(f) for f in s["former"]]) for s in states]
                text = render_relations(safe_states, lambda t: _safe(_name_for(t)))
            else:
                text = render_relations(states, _name_for)
            if text:
                stats["entries"] += 1
                stats["relations"] += len(states)
                source_name = translated or raw
                for state in states:
                    if _changed_here(state):
                        stats["changed_here"].append(
                            f"{source_name} → {render_relations([state], _name_for)}")
        rendered_cache[key] = text
        return text

    changed = False
    if fmt == "json":
        for value, raw in _json_items(data):
            if not raw:
                continue
            text = _rendered(raw, _json_translated(value))
            if text:
                value[field] = text
                changed = True
        if not changed:
            return glossary_text, stats
        return json.dumps(data, ensure_ascii=False, indent=2), stats

    if fmt == "token":
        out = []
        header_at = None
        for line in lines:
            body, ending = _split_line_ending(line)
            stripped = body.strip()
            if header_at is None and stripped.lower().startswith("glossary columns:"):
                header_at = len(out)
            if stripped.startswith("* "):
                raw, translated = _token_identity(body)
                text = _rendered(raw, translated, paren_safe=True) if raw else ""
                if text:
                    body = f"{body.rstrip()} ({field}: {text})"
                    changed = True
            out.append(body + ending)
        if not changed:
            return glossary_text, stats
        if header_at is not None:
            body, ending = _split_line_ending(out[header_at])
            columns = [c.strip().casefold() for c in body.split(":", 1)[1].split(",")]
            if field.casefold() not in columns:
                out[header_at] = f"{body.rstrip()}, {field}{ending}"
        return "".join(out), stats

    # Legacy CSV.
    sep = csv_info["sep"]
    header = csv_info["header"]
    header_index = csv_info["header_index"]
    field_idx = next((i for i, c in enumerate(header) if c.casefold() == field.casefold()), None)
    width = len(header) if field_idx is not None else len(header) + 1
    target_idx = field_idx if field_idx is not None else len(header)
    out = list(lines[: header_index + 1])
    for line in lines[header_index + 1:]:
        body, ending = _split_line_ending(line)
        if body.strip():
            cells = _csv_split(body, sep)
            raw = cells[csv_info["raw_idx"]].strip() if len(cells) > csv_info["raw_idx"] else ""
            translated = cells[csv_info["trans_idx"]].strip() if len(cells) > csv_info["trans_idx"] else ""
            text = _rendered(raw, translated) if raw else ""
            if text:
                cells = list(cells) + [""] * max(0, width - len(cells))
                cells[target_idx] = text
                body = _csv_join(cells, sep)
                changed = True
        out.append(body + ending)
    if not changed:
        return glossary_text, stats
    if field_idx is None:
        body, ending = _split_line_ending(out[header_index])
        out[header_index] = f"{body}{sep}{field}{ending}"
    return "".join(out), stats


def relations_log_line(stats: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(stats, Mapping) or not stats.get("enabled"):
        return None
    cutoff = stats.get("cutoff")
    count = int(stats.get("relations") or 0)
    if count <= 0:
        return f"🔗 Relations: none recorded up to Ch. {cutoff}"
    characters = int(stats.get("entries") or 0)
    noun = "character" if characters == 1 else "characters"
    line = f"🔗 Relations: {count} shown for {characters} {noun} (up to Ch. {cutoff})"
    changed = [str(c) for c in stats.get("changed_here") or [] if c]
    if changed:
        line += " | changed here: " + "; ".join(changed)
    return line


# ---------------------------------------------------------------------------
# Prompt placeholder
# ---------------------------------------------------------------------------

_INSTRUCTION_TEMPLATE = (
    'The `{field}` column: for {gender_types} entries only, list this character\'s relationships to OTHER named '
    'characters that are shown or stated in THIS text. Write each as `raw_name: role`, separated by "; ", '
    'where raw_name is the other character\'s raw_name exactly as in their own entry and role is what THIS '
    'row\'s character is to that other character, in {language}, 1–3 words (Father, Mother, Daughter, Older '
    'Sister, Wife, Best Friend, Master, Disciple, Subordinate, Rival, Enemy). Both directions: a '
    'father\'s row has `<his daughter\'s raw_name>: Father` and the daughter\'s row has `<her father\'s '
    'raw_name>: Daughter`. List only real, specific relationships; never filler such as Acquaintance, Stranger '
    'or Unknown, and never a note such as Surname. Keep the same wording for the same relationship; use a '
    'different word only when the relationship itself changed in this text. Do not guess, do not describe events '
    'or feelings, never use commas inside this column. Leave it empty for all other entry types, surname-only '
    'entries, and when no relationship is shown.\n'
    '{column_rule}\n'
    'Example rows that mirror each other: `{example_row_1}` and `{example_row_2}`'
)
_DECLARE_TEMPLATE = 'Add a `{field}` column as one extra LAST column of every row. '
_COLUMN_RULE = '`{field}` is its own column: never write relationships in another column.'
_COLUMN_RULE_POSITION = (
    '`{field}` is its own column, column {index} of {total} ({columns}): never write relationships in '
    'another column.'
)

_BASE_COLUMNS = ("type", "raw_name", "translated_name", "gender")
#: The two made-up characters of the glossary prompt's own example rows. They are not
#: real names, so a model cannot take them for characters of the book it is reading.
#: The long form (column rule + full descriptions) is the owner's choice (2026-11-14): the
#: short one of 2026-11-13 was followed worse. "Fiancé" is left out of the label examples
#: on purpose: small models copied it into books that have no fiancé.
_EXAMPLE_ROWS = (
    {"type": "character", "raw_name": "이히리ᐐ 나애", "translated_name": "Dihirit Ade", "gender": "female",
     "description": '"The enigmatic guild leader of the Shadow Lotus who operates from the concealed '
                    'backrooms of the capital"',
     "relations": "뢤사난: Older Sister"},
    {"type": "character", "raw_name": "뢤사난", "translated_name": "Kim Sang-hyu", "gender": "male",
     "description": '"A master swordsman from the Northern Sect known for his icy demeanor and '
                    'unparalleled skill with the Frost Blade technique"',
     "relations": "이히리ᐐ 나애: Younger Brother"},
)


def instruction_columns(columns: Optional[Iterable[Any]], field: str) -> List[str]:
    """The row's columns with the relations field in them (added last when missing)."""
    names = [str(c).strip() for c in (columns or []) if str(c).strip()]
    if not any(name.casefold() == field.casefold() for name in names):
        names.append(field)
    return names


def _example_row(example: Mapping[str, str], columns: Sequence[str], field: str) -> str:
    cells = []
    for name in columns:
        if name.casefold() == field.casefold():
            cells.append(example["relations"])
        else:
            cells.append(example.get(name.casefold(), ""))
    return ",".join(cells)


def relations_gender_types(gender_types: Any = None) -> str:
    """What ``{gender_types}`` reads in the relations text: the comma list of the enabled entry types
    with gender (a list or an already joined string), or "character" when there is none."""
    if isinstance(gender_types, str):
        text = gender_types.strip()
    else:
        text = ", ".join(str(t).strip() for t in (gender_types or ()) if str(t).strip())
    return text or "character"


def instruction_text(field: Optional[str] = None, language: Optional[str] = "English",
                     declare_column: bool = False, columns: Optional[Iterable[Any]] = None,
                     template: Optional[str] = None, gender_types: Any = None) -> str:
    """The ``{relations_instruction}`` expansion.

    ``columns`` is the row layout the prompt asks for. With it the two example rows have
    every column in its place and the text says which column the relations go in; a prompt
    whose other examples end at ``description`` otherwise teaches the model to write them
    there.
    ``template`` replaces the built-in text (a user-edited one); it may use the tokens
    ``{field}``, ``{language}``, ``{gender_types}``, ``{column_rule}``, ``{example_row_1}`` and
    ``{example_row_2}``, or none.
    ``gender_types``: the enabled entry types with gender (``relations_gender_types``).
    """
    field = (field or "").strip() or field_name()
    language = (language or "").strip() or "English"
    if columns:
        layout = instruction_columns(columns, field)
        index = next(i for i, name in enumerate(layout) if name.casefold() == field.casefold()) + 1
        rule = (_COLUMN_RULE_POSITION.replace("{index}", str(index)).replace("{total}", str(len(layout)))
                .replace("{columns}", ", ".join(layout)))
    else:
        layout = instruction_columns(_BASE_COLUMNS, field)
        rule = _COLUMN_RULE
    text = _INSTRUCTION_TEMPLATE if template is None else str(template)
    text = text.replace("{column_rule}", rule)
    text = text.replace("{example_row_1}", _example_row(_EXAMPLE_ROWS[0], layout, field))
    text = text.replace("{example_row_2}", _example_row(_EXAMPLE_ROWS[1], layout, field))
    text = text.replace("{field}", field).replace("{language}", language)
    text = text.replace("{gender_types}", relations_gender_types(gender_types))
    if declare_column:
        text = _DECLARE_TEMPLATE.replace("{field}", field) + text
    return text


_PLACEHOLDER_LINE_RE = re.compile(
    r"^[ \t]*-?[ \t]*" + re.escape(RELATIONS_PLACEHOLDER) + r"[ \t]*\r?\n?",
    flags=re.MULTILINE,
)


def apply_relations_placeholder(prompt_text: Any, enabled: Optional[bool] = None, field: Optional[str] = None,
                                language: Optional[str] = None, declare_column: bool = False,
                                columns: Optional[Iterable[Any]] = None, template: Optional[str] = None,
                                gender_types: Any = None) -> Any:
    """Expand ``{relations_instruction}`` when tracking is on; remove its whole line when off.

    ``template``: a user-edited instruction text (``instruction_text``); one that expands to
    nothing removes the line as when tracking is off. ``gender_types``: see ``instruction_text``.
    """
    if not isinstance(prompt_text, str) or RELATIONS_PLACEHOLDER not in prompt_text:
        return prompt_text
    if enabled is None:
        enabled = tracking_enabled()
    if enabled:
        if language is None:
            language = os.environ.get("GLOSSARY_TARGET_LANGUAGE", "English")
        text = instruction_text(field, language, declare_column=declare_column, columns=columns,
                                template=template, gender_types=gender_types)
        if not text:
            return _PLACEHOLDER_LINE_RE.sub("", prompt_text).replace(RELATIONS_PLACEHOLDER, "")
        return prompt_text.replace(RELATIONS_PLACEHOLDER, text)
    return _PLACEHOLDER_LINE_RE.sub("", prompt_text)


def fresh_config_fields(config: Mapping[str, Any]) -> List[Any]:
    """The custom fields of ``config`` with the relations column added, as turning the tracker on adds it
    (``enable_field_name`` + ``apply_relations_field``). An absent list is the desktop's own start
    (owner_state: ["description"] unless the user removed it)."""
    fields = config.get("custom_glossary_fields")
    fields = list(fields) if isinstance(fields, (list, tuple)) else []
    if not fields and not config.get("custom_field_description_removed", False):
        fields = ["description"]
    name = str(config.get(CONFIG_FIELD) or "").strip()
    if not name or name.casefold() in RESERVED_FIELD_NAMES:
        name = DEFAULT_FIELD_NAME
    return apply_relations_field(fields, True, enable_field_name(fields, name))


def apply_fresh_defaults(config: Any) -> bool:
    """Desktop startup: a config carrying ``FRESH_DEFAULTS_KEY`` gets the listed toggles on (each one it
    does not hold yet) and, with the tracker, the relations column; the key is consumed. Returns True
    when it applied. A config without the key (every existing one) is not touched."""
    if not isinstance(config, dict) or FRESH_DEFAULTS_KEY not in config:
        return False
    keys = config.pop(FRESH_DEFAULTS_KEY)
    keys = [key for key in (keys if isinstance(keys, (list, tuple)) else ()) if key in FRESH_DEFAULT_KEYS]
    for key in keys:
        if key in config:
            continue
        if key == CONFIG_ENABLE:
            config["custom_glossary_fields"] = fresh_config_fields(config)
        config[key] = True
    return True


def minimal_relations_requested(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Minimal's own "Include Relations Column" toggle (``settings``: a config dict or an env snapshot)."""
    if isinstance(settings, Mapping):
        return _truthy(settings.get(ENV_MINIMAL, settings.get(CONFIG_MINIMAL)))
    return _truthy(os.environ.get(ENV_MINIMAL))


def prompts_to_check(prompts: Optional[Mapping[str, Any]],
                     settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The prompts that must hold the placeholder: Balanced/Full always, Minimal only with its toggle on."""
    if not isinstance(prompts, Mapping):
        return {}
    minimal_on = minimal_relations_requested(settings)
    return {key: text for key, text in prompts.items() if key != "minimal" or minimal_on}


def prompts_missing_placeholder(prompts: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(prompts, Mapping):
        return []
    return [key for key, text in prompts.items()
            if not isinstance(text, str) or RELATIONS_PLACEHOLDER not in text]


def reset_question(names: Iterable[Any]) -> str:
    """The one question asked when the tracker is turned on and prompts lack the placeholder.

    ``names`` are the display names of those prompts ("Balanced/Full", "Minimal").
    """
    names = [str(name) for name in names or [] if str(name or "").strip()]
    if len(names) <= 1:
        name = names[0] if names else "glossary"
        return (f"The {name} prompt has no {RELATIONS_PLACEHOLDER} line, so the AI will not return relations.\n\n"
                f"Reset the {name} prompt to default?")
    listed = " and ".join([", ".join(names[:-1]), names[-1]])
    every = "both" if len(names) == 2 else "these"
    return (f"The {listed} prompts have no {RELATIONS_PLACEHOLDER} line, so the AI will not return relations.\n\n"
            f"Reset {every} prompts to default?")


# ---------------------------------------------------------------------------
# Custom-field rules
# ---------------------------------------------------------------------------

def _fold(name: Any) -> str:
    return str(name or "").strip().casefold()


def apply_relations_field(fields: Optional[Iterable[Any]], enabled: bool, name: Optional[str] = None,
                          previous_name: Optional[str] = None) -> List[Any]:
    """Add (once, at the end), keep, rename in place, or remove the relations custom field."""
    current = list(fields or [])
    name = (name or "").strip() or DEFAULT_FIELD_NAME
    wanted = _fold(name)
    previous = _fold(previous_name) if previous_name else ""
    if not enabled:
        drop = {wanted} | ({previous} if previous else set())
        return [f for f in current if _fold(f) not in drop]
    result: List[Any] = []
    placed = False
    for item in current:
        folded = _fold(item)
        if folded == wanted or (previous and folded == previous):
            if not placed:
                result.append(name)
                placed = True
            continue
        result.append(item)
    if not placed:
        result.append(name)
    return result


def enable_field_name(fields: Optional[Iterable[Any]], name: Optional[str] = None) -> str:
    """The relations field's name when the tracker is turned ON (off -> on only).

    Turning the tracker off removes its field, so a field that already has the name at
    that moment is the user's own column: keep it and use the first free
    ``<name>_tracker``, ``<name>_tracker_2``, ... instead.
    """
    name = (name or "").strip() or DEFAULT_FIELD_NAME
    taken = {_fold(f) for f in (fields or [])}
    if _fold(name) not in taken:
        return name
    base = f"{name}_tracker"
    candidate, number = base, 2
    while _fold(candidate) in taken:
        candidate = f"{base}_{number}"
        number += 1
    return candidate


def is_relations_field(name: Any, settings_or_config: Optional[Mapping[str, Any]] = None) -> bool:
    folded = _fold(name)
    return bool(folded) and folded == _fold(field_name(settings_or_config))


def persisted_custom_fields(fields: Optional[Iterable[Any]], settings: Optional[Mapping[str, Any]] = None) -> List[Any]:
    current = list(fields or [])
    if not tracking_enabled(settings):
        return current
    wanted = _fold(field_name(settings))
    return [f for f in current if _fold(f) != wanted]


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------

def edge_label_text(edge: Mapping[str, Any], show_former: bool = False) -> str:
    """``"Father / Daughter"``: what b is to a, then what a is to b (from each side's own row).

    With ``show_former`` the earlier labels follow in brackets. A relation that reads the
    same from both sides is written once (``"Teammate"``, not ``"Teammate / Teammate"``), and
    so is one with only one side recorded (``"Club Member"``); none recorded gives "" (draw
    no label). ``edge_description_lines`` says who is what to whom.
    Shared by the desktop and mobile diagrams.
    """
    def _side(label: Any, former: Any) -> str:
        label = str(label or '').strip()
        if not label:
            return ''
        if show_former and former:
            return f"{label} [{', '.join(str(f) for f in former)}]"
        return label
    first = _side(edge.get('label_ab'), edge.get('former_ab'))
    second = _side(edge.get('label_ba'), edge.get('former_ba'))
    if not first or not second or first.casefold() == second.casefold():
        return first or second
    return f"{first} / {second}"


def wrap_label(text: str, width_of: Callable[[str], float], limit: float) -> str:
    """``text`` broken between words into lines no wider than ``limit`` (as ``width_of`` measures;
    a single word wider stays whole on its own line), joined by newlines. A label as long as a
    sentence on one line needs a line as long as itself clear of every other line: in the middle
    of a picture there is no such stretch, so those lines ran out past everything else (the
    owner's book: two labels over 1,000 px wide set the width of the whole picture)."""
    lines: List[str] = []
    for word in str(text).split(" "):
        if lines and width_of(lines[-1] + " " + word) <= limit:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "\n".join(lines)


def edge_description_lines(edge: Mapping[str, Any], name_for: Callable[[str], Any]) -> List[str]:
    """``["B is Father to A", "A is Daughter to B"]`` for an edge (unknown sides left out);
    ``name_for(node_id)`` gives the shown name. For tooltips and the mobile sheet."""
    lines = []
    a, b = str(edge.get('a')), str(edge.get('b'))
    for label, subject, other in ((edge.get('label_ab'), b, a), (edge.get('label_ba'), a, b)):
        label = str(label or '').strip()
        if label:
            lines.append(f"{name_for(subject)} is {label} to {name_for(other)}")
    return lines


def filter_graph(graph: Optional[Mapping[str, Any]], hidden: Iterable[Any] = (), show_relations: bool = True,
                 show_gender: bool = True) -> Dict[str, Any]:
    """A copy of ``graph`` as drawn with the view choices: hidden names and their lines left out,
    no lines with ``show_relations`` off, no gender-only names and no gender with
    ``show_gender`` off. Lay out the unfiltered graph and draw this one, so the names that stay
    keep their positions."""
    hidden_keys = {tracker_key(h) for h in hidden or ()}
    graph = dict(graph or {})
    nodes = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, Mapping) or tracker_key(node.get("id")) in hidden_keys:
            continue
        if not show_gender and node.get("gender_only"):
            continue
        node = dict(node)
        if not show_gender:
            node["gender"] = ""
        nodes.append(node)
    ids = {node.get("id") for node in nodes}
    edges = [dict(edge) for edge in graph.get("edges") or []
             if show_relations and isinstance(edge, Mapping) and edge.get("a") in ids and edge.get("b") in ids]
    graph["nodes"] = nodes
    graph["edges"] = edges
    return graph


def build_graph(tracker: Optional[Mapping[str, Any]], glossary_entries: Optional[Iterable[Any]] = None,
                up_to_chapter: Any = None, show_unlisted: bool = False,
                mode: str = DEFAULT_HISTORY_MODE, *, gender_tracker: Optional[Mapping[str, Any]] = None,
                include_gender_only: bool = False, hidden: Iterable[Any] = (), show_relations: bool = True,
                show_gender: bool = True) -> Dict[str, Any]:
    """Nodes and one edge per unordered pair of characters, as known up to ``up_to_chapter``.

    With ``gender_tracker`` (the book's gender tracker dict, ``read_gender_tracker``) each node's
    ``gender`` is the one known at ``up_to_chapter`` (``gender_at``: the tracker first, the
    glossary's gender as fallback) and carries ``"gender_only": False``; ``include_gender_only``
    adds the characters with gender tracking but no relation (by then) as nodes with
    ``degree`` 0 and ``"gender_only": True``. ``hidden`` / ``show_relations`` /
    ``show_gender`` apply ``filter_graph``. With the defaults the result is what it always was.
    """
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    low, high = chapter_bounds(tracker)
    graph: Dict[str, Any] = {"nodes": [], "edges": [], "chapters": [low, high]}
    if not isinstance(entries, Mapping):
        return graph
    glossary: Optional[Dict[str, Mapping[str, Any]]] = None
    if glossary_entries is not None:
        glossary = {}
        for entry in glossary_entries:
            if isinstance(entry, Mapping) and str(entry.get("raw_name") or "").strip():
                glossary.setdefault(tracker_key(entry.get("raw_name")), entry)
    if up_to_chapter is None:
        visible = lambda occurrence: True  # noqa: E731
    else:
        visible = _visible_predicate(tracker, chapter_cutoff({"chapter_num": up_to_chapter}))

    def _listed(key: str) -> bool:
        return glossary is None or key in glossary

    raw_names: Dict[str, str] = {}
    translated: Dict[str, str] = {}
    for key, item in entries.items():
        if isinstance(item, Mapping):
            raw_names.setdefault(key, str(item.get("raw_name") or key))
            if item.get("translated_name"):
                translated.setdefault(key, str(item.get("translated_name")))

    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, item in entries.items():
        if not isinstance(item, Mapping):
            continue
        for state in _relations_visible(tracker, item.get("raw_name") or key, visible, mode):
            tkey = state["target_key"]
            if tkey == key:
                continue
            raw_names.setdefault(tkey, str(state["target_raw"]))
            if state.get("target_translated"):
                translated.setdefault(tkey, str(state["target_translated"]))
            if not show_unlisted and (not _listed(key) or not _listed(tkey)):
                continue
            a, b = (key, tkey) if key < tkey else (tkey, key)
            edge = pairs.setdefault((a, b), {"a": a, "b": b, "label_ab": "", "label_ba": "",
                                             "former_ab": [], "former_ba": [], "changed": False, "since": None,
                                             "_since": []})
            suffix = "ab" if key == a else "ba"  # label_ab = what b is to a (from a's row)
            edge[f"label_{suffix}"] = state["label"]
            edge[f"former_{suffix}"] = list(state["former"])
            if state.get("changed_at") is not None:
                edge["changed"] = True
            if state.get("since") is not None:
                edge["_since"].append(state["since"])

    degree: Dict[str, int] = {}
    edges = []
    for (a, b) in sorted(pairs):
        edge = pairs[(a, b)]
        since_values = edge.pop("_since")
        edge["since"] = max(since_values, key=lambda v: (_num(v) if _num(v) is not None else -_BIG)) if since_values else None
        edges.append(edge)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    def _node_order(key: str) -> Tuple[Any, ...]:
        item = entries.get(key)
        return (_first_key_of(item) if isinstance(item, Mapping) else (_BIG, _BIG, _BIG, ""), key)

    nodes = []
    for key in sorted(degree, key=_node_order):
        entry = glossary.get(key) if glossary is not None else None
        label = (str(entry.get("translated_name") or "") if entry else "") or translated.get(key) or raw_names.get(key, key)
        nodes.append({
            "id": key,
            "raw_name": (str(entry.get("raw_name")) if entry else "") or raw_names.get(key, key),
            "label": label,
            "gender": str(entry.get("gender") or "") if entry else "",
            "degree": degree[key],
            "listed": _listed(key),
        })
    if gender_tracker is not None:
        for node in nodes:
            entry = glossary.get(node["id"]) if glossary is not None else None
            node["gender"] = gender_at(gender_tracker, node["raw_name"], up_to_chapter,
                                       glossary_gender=entry.get("gender") if entry else "")
            node["gender_only"] = False
        if include_gender_only:
            removed = set(_records(_edits(tracker), "removed"))
            gender_entries = gender_tracker.get("entries") if isinstance(gender_tracker, Mapping) else None
            for gkey, gitem in (gender_entries.items() if isinstance(gender_entries, Mapping) else ()):
                if not isinstance(gitem, Mapping):
                    continue
                raw = str(gitem.get("raw_name") or gkey).strip()
                key = tracker_key(raw)
                if not key or key in degree or key in removed or (not show_unlisted and not _listed(key)):
                    continue
                entry = glossary.get(key) if glossary is not None else None
                history = gender_history(gender_tracker, raw, up_to_chapter,
                                         glossary_gender=entry.get("gender") if entry else "")
                if history["source"] not in ("tracker", "decision") or not history["current"]:
                    continue
                degree[key] = 0
                nodes.append({
                    "id": key,
                    "raw_name": (str(entry.get("raw_name")) if entry else "") or raw,
                    "label": (str(entry.get("translated_name") or "") if entry else "")
                    or str(gitem.get("translated_name") or "") or translated.get(key) or raw,
                    "gender": history["current"],
                    "degree": 0,
                    "listed": _listed(key),
                    "gender_only": True,
                })
    graph["nodes"] = nodes
    graph["edges"] = edges
    if hidden or not show_relations or not show_gender:
        return filter_graph(graph, hidden, show_relations, show_gender)
    return graph


# ---------------------------------------------------------------------------
# Gender (the book's gender tracker, read for the diagram)
# ---------------------------------------------------------------------------

def gender_tracker_path_for_glossary(glossary_path: Optional[str]) -> str:
    """The gender tracker next to the glossary (``X_glossary.csv`` -> ``X_gender_tracker.json``)."""
    return _gender.tracker_path_for_glossary(glossary_path or "")


def read_gender_tracker(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The gender tracker at ``path`` (``{"version": 2, "entries": {}}`` when missing), or None
    when it exists and cannot be read right now (skip it until the next check)."""
    data = _gender.read_json_file(path)
    if data is None:
        return None
    if data == {}:
        return {"version": 2, "entries": {}}
    if not isinstance(data, dict) or not isinstance(data.get("entries", {}), dict):
        return None
    data.setdefault("entries", {})
    return data


def _gender_points(entry: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One gender per chapter, in chapter order (a hand-set one wins at its chapter)."""
    if not isinstance(entry, Mapping):
        return []
    occurrences = [o for o in entry.get("occurrences", []) or [] if isinstance(o, Mapping)
                   and _gender.normalize_gender(o.get("gender")) not in _gender.UNKNOWN_GENDERS]
    occurrences.sort(key=lambda o: _gender.tracker_occurrence_sort_key(dict(o)))
    points: List[Dict[str, Any]] = []
    for occurrence in occurrences:
        point = {"gender": _gender.normalize_gender(occurrence.get("gender")),
                 "chapter_num": occurrence.get("chapter_num"),
                 "chapter_file": _base(occurrence.get("chapter_file")),
                 "manual": bool(occurrence.get("manual"))}
        if points and _same_chapter(points[-1]["chapter_num"], points[-1]["chapter_file"],
                                    point["chapter_num"], point["chapter_file"]):
            if points[-1]["manual"] and not point["manual"]:
                continue
            points[-1] = point
        else:
            points.append(point)
    return points


def gender_history(gender_tracker: Optional[Mapping[str, Any]], raw_name: Any, up_to_chapter: Any = None,
                   glossary_gender: Any = "") -> Dict[str, Any]:
    """A character's gender as the diagram shows it, up to ``up_to_chapter`` (None: all).

    ``{"raw_name", "decision": "auto"|"male"|"female" (set for every chapter), "current": the
    gender known at that chapter ("" unknown), "source": "decision" | "tracker" | "glossary" | "",
    "timeline": [{"gender", "from_chapter", "to_chapter", "from_file", "to_file", "count",
    "manual"}]}`` -- runs like ``character_history``'s. The decision wins, then the tracker's
    last gender at or before the chapter, then ``glossary_gender``.
    """
    entry = _gender.tracker_entry_for_raw(gender_tracker, raw_name)
    decision = _gender.normalized_decision(entry) if entry else "auto"
    points = _gender_points(entry)
    if up_to_chapter is not None:
        limit = _num(up_to_chapter)
        points = [p for p in points if limit is not None and _num(p["chapter_num"]) is not None
                  and _num(p["chapter_num"]) <= limit]
    timeline: List[Dict[str, Any]] = []
    for point in points:
        if timeline and timeline[-1]["gender"] == point["gender"]:
            run = timeline[-1]
            run.update(to_chapter=point["chapter_num"], to_file=point["chapter_file"], count=run["count"] + 1)
            run["manual"] = run["manual"] or point["manual"]
            continue
        timeline.append({"gender": point["gender"], "from_chapter": point["chapter_num"],
                         "from_file": point["chapter_file"], "to_chapter": point["chapter_num"],
                         "to_file": point["chapter_file"], "count": 1, "manual": point["manual"]})
    if decision in _gender.BINARY_GENDERS:
        current, source = decision, "decision"
    elif timeline:
        current, source = timeline[-1]["gender"], "tracker"
    else:
        fallback = _gender.normalize_gender(glossary_gender)
        current = "" if fallback in _gender.UNKNOWN_GENDERS else fallback
        source = "glossary" if current else ""
    return {"raw_name": str((entry or {}).get("raw_name") or raw_name or ""), "decision": decision,
            "current": current, "source": source, "timeline": timeline}


def gender_at(gender_tracker: Optional[Mapping[str, Any]], raw_name: Any, up_to_chapter: Any = None,
              glossary_gender: Any = "") -> str:
    """The gender known at ``up_to_chapter`` (lower case, "" unknown); see ``gender_history``."""
    return gender_history(gender_tracker, raw_name, up_to_chapter, glossary_gender)["current"]


def gender_history_text(history: Optional[Mapping[str, Any]]) -> str:
    """``"Male (Ch. 1), Female (Ch. 2-4)"``; ``"Female (all chapters)"`` for a decision;
    ``"Female (glossary)"``; "" when unknown."""
    if not isinstance(history, Mapping):
        return ""
    if history.get("source") == "decision":
        return f"{_gender.display_gender(history.get('current'))} (all chapters)"
    runs = []
    for run in history.get("timeline") or []:
        start, end = _display_num(run.get("from_chapter")), _display_num(run.get("to_chapter"))
        span = f"Ch. {start}" if start == end else f"Ch. {start}-{end}"
        runs.append(f"{_gender.display_gender(run.get('gender'))} ({span})")
    if runs:
        return ", ".join(runs)
    if history.get("source") == "glossary" and history.get("current"):
        return f"{_gender.display_gender(history.get('current'))} (glossary)"
    return ""


def set_gender(glossary_path: str, raw_name: Any, gender: Any, chapter: Any = None, *,
               chapter_file: Any = "", translated_name: Any = "") -> Dict[str, Any]:
    """Set a gender entry in the book's gender tracker FILE (locked load -> modify -> write
    through the gender tracker's own writer, ``gender_tracking.update_tracker_file``).

    ``chapter`` None: for every chapter (the decision the Glossary Editor sets; "auto" or ""
    clears it). A chapter number: that chapter only (``gender_tracking.set_tracker_gender``).
    Returns ``update_tracker_file``'s ``{"changed", "result", "before", "tracker", "stamp"}``
    (keep it for ``undo_gender_edit``). Raises ``ValueError`` for an unreadable file.
    """
    path = gender_tracker_path_for_glossary(glossary_path)
    if not path:
        raise ValueError("No glossary path for the gender tracker.")
    relations = load_tracker(tracker_path_for_glossary(glossary_path)) if chapter is not None else None

    def _apply(gender_tracker: Dict[str, Any]) -> bool:
        translated = str(translated_name or "")
        if chapter is None:
            return _gender.set_tracker_gender(gender_tracker, raw_name, gender, None, translated_name=translated)
        file = _base(chapter_file)
        known = []
        for item in (gender_tracker.get("entries") or {}).values():
            for occurrence in (item.get("occurrences") or []) if isinstance(item, Mapping) else []:
                if isinstance(occurrence, Mapping):
                    known.append((occurrence.get("chapter_index"), occurrence.get("chapter_num"),
                                  _base(occurrence.get("chapter_file"))))
        same = [k for k in known if _same_chapter(k[1], k[2], chapter, file) and _num(k[0]) is not None]
        if same:
            index, file = same[0][0], file or same[0][2]
        else:
            index, _number, recorded_file = chapter_slot(relations, chapter, file)
            file = file or recorded_file
            if index is None:
                index = _estimate_index(known, _num(chapter), 0)
            elif _num(index) != int(_num(index)):
                index = int(_num(index))  # the gender tracker sorts by whole indexes, then numbers
        return _gender.set_tracker_gender(gender_tracker, raw_name, gender, _display_num(chapter),
                                          chapter_file=file, chapter_index=index, translated_name=translated)

    return _gender.update_tracker_file(path, _apply)


def undo_gender_edit(glossary_path: str, edit: Mapping[str, Any]) -> Dict[str, Any]:
    """Undo one ``set_gender`` result (same rules and return value as ``undo_edit``)."""
    path = gender_tracker_path_for_glossary(glossary_path)
    return _restore_parts(path, edit, read_gender_tracker, _gender.write_tracker,
                          lambda p: _gender.tracker_file_guard(p))


# ---------------------------------------------------------------------------
# View choices (per book, inside the tracker file; shared by desktop and mobile)
# ---------------------------------------------------------------------------

def _default_view() -> Dict[str, Any]:
    return {"hidden": [], "show_relations": True, "show_gender": True}


def view_settings(tracker: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``{"hidden": [node ids], "show_relations": bool, "show_gender": bool}`` (defaults when unset)."""
    view = tracker.get(VIEW_KEY) if isinstance(tracker, Mapping) else None
    result = _default_view()
    if isinstance(view, Mapping):
        hidden = []
        for name in view.get("hidden") or []:
            key = tracker_key(name)
            if key and key not in hidden:
                hidden.append(key)
        result["hidden"] = hidden
        result["show_relations"] = view.get("show_relations", True) is not False
        result["show_gender"] = view.get("show_gender", True) is not False
    return result


def set_view(tracker: Dict[str, Any], *, hidden: Optional[Iterable[Any]] = None,
             show_relations: Optional[bool] = None, show_gender: Optional[bool] = None) -> Dict[str, Any]:
    """Change the view choices in a tracker dict (None: keep); ``hidden`` replaces the list.
    Relations data is not touched. Returns ``{"changed", "view"}``."""
    current = view_settings(tracker)
    wanted = dict(current)
    if hidden is not None:
        keys = []
        for name in hidden:
            key = tracker_key(name)
            if key and key not in keys:
                keys.append(key)
        wanted["hidden"] = keys
    if show_relations is not None:
        wanted["show_relations"] = bool(show_relations)
    if show_gender is not None:
        wanted["show_gender"] = bool(show_gender)
    if wanted == current or not isinstance(tracker, dict):
        return {"changed": False, "view": current}
    if wanted == _default_view():
        tracker.pop(VIEW_KEY, None)
    else:
        tracker[VIEW_KEY] = wanted
    return {"changed": True, "view": wanted}


def hide_names(tracker: Dict[str, Any], names: Iterable[Any], hide: bool = True) -> Dict[str, Any]:
    """Add names to (or with ``hide`` False take them out of) the hidden list."""
    hidden = view_settings(tracker)["hidden"]
    keys = [tracker_key(n) for n in names or () if tracker_key(n)]
    if hide:
        hidden += [k for k in keys if k not in hidden]
    else:
        hidden = [k for k in hidden if k not in keys]
    return set_view(tracker, hidden=hidden)


def save_view(path: str, *, hidden: Optional[Iterable[Any]] = None, show_relations: Optional[bool] = None,
              show_gender: Optional[bool] = None) -> Dict[str, Any]:
    """``set_view`` on the tracker FILE (locked write; ``edit_tracker``'s return value)."""
    return edit_tracker(path, lambda tracker: set_view(tracker, hidden=hidden, show_relations=show_relations,
                                                       show_gender=show_gender))


def known_names(tracker: Optional[Mapping[str, Any]], glossary_entries: Optional[Iterable[Any]] = None,
                gender_tracker: Optional[Mapping[str, Any]] = None) -> List[Dict[str, str]]:
    """Every character name the pickers offer: ``[{"id", "raw_name", "label"}]`` sorted by label
    (glossary character rows, tracker names and relation targets, gender tracker names; removed
    names left out). ``label`` is the translated name when there is one."""
    removed = set(_records(_edits(tracker), "removed"))
    found: Dict[str, Dict[str, str]] = {}

    def _add(raw: Any, translated: Any = "") -> None:
        raw = str(raw or "").strip()
        key = tracker_key(raw)
        if not key or key in removed:
            return
        name = found.setdefault(key, {"id": key, "raw_name": raw, "label": ""})
        if not name["label"] and str(translated or "").strip():
            name["label"] = str(translated).strip()

    for entry in glossary_entries or []:
        if isinstance(entry, Mapping) and str(entry.get("type") or "character").strip().casefold() == "character":
            _add(entry.get("raw_name"), entry.get("translated_name"))
    entries = tracker.get("entries") if isinstance(tracker, Mapping) else None
    for key, item in (entries.items() if isinstance(entries, Mapping) else ()):
        if not isinstance(item, Mapping):
            continue
        _add(item.get("raw_name") or key, item.get("translated_name"))
        for tkey, rel in (item.get("relations") or {}).items():
            if isinstance(rel, Mapping):
                _add(rel.get("target_raw") or tkey, rel.get("target_translated"))
    gender_entries = gender_tracker.get("entries") if isinstance(gender_tracker, Mapping) else None
    for key, item in (gender_entries.items() if isinstance(gender_entries, Mapping) else ()):
        if isinstance(item, Mapping):
            _add(item.get("raw_name") or key, item.get("translated_name"))
    names = list(found.values())
    for name in names:
        name["label"] = name["label"] or name["raw_name"]
    return sorted(names, key=lambda n: (n["label"].casefold(), n["id"]))


def _fit_positions(positions: Dict[str, List[float]], width: float, height: float) -> Dict[str, Tuple[float, float]]:
    if not positions:
        return {}
    margin = min(width, height) * 0.08
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = max_x - min_x, max_y - min_y
    avail_x, avail_y = max(width - 2 * margin, 1e-9), max(height - 2 * margin, 1e-9)
    scales = []
    if span_x > 1e-9:
        scales.append(avail_x / span_x)
    if span_y > 1e-9:
        scales.append(avail_y / span_y)
    scale = min(scales) if scales else 0.0
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    result = {}
    for key, (x, y) in positions.items():
        nx = width / 2.0 + (x - cx) * scale
        ny = height / 2.0 + (y - cy) * scale
        result[key] = (min(max(nx, 0.0), width), min(max(ny, 0.0), height))
    return result


#: Rings with more names keep their order (ordering costs the square of the count).
_RING_ORDER_MAX = 200
#: When names on the first ring are related to each other they sit at 1, 1 + this and 1 + twice
#: this times the ring's distance, in turn.
_RING_STAGGER = 0.3
#: The widest wedge (radians) a name on the first ring shares out among the names hanging from it:
#: round a focus with one relation that name's wedge would be the full turn, and its lines to the
#: names on the far side of the focus would run through the focus.
_WEDGE_MAX = 1.5 * math.pi


def _ring_order(members: List[str], adjacency: Dict[str, List[str]]) -> List[str]:
    """``members`` in an order round the ring where names related to each other sit side by side:
    the lines between them stay short and out at the ring, not across the middle where every line
    of the focus runs. Each next name is the one with most relations to the last few placed."""
    if len(members) > _RING_ORDER_MAX:
        return members
    inside = set(members)
    links = {node: [other for other in adjacency.get(node, []) if other in inside and other != node]
             for node in members}
    left = list(members)
    order: List[str] = []
    while left:
        if not order or not any(other in left for other in links[order[-1]]):
            # a new run: the name with most relations among those left
            pick = max(left, key=lambda node: (sum(1 for other in links[node] if other in left), -left.index(node)))
        else:
            recent = order[-3:]
            pick = max(left, key=lambda node: (sum(2 if other == recent[-1] else 1
                                                   for other in links[node] if other in recent),
                                               sum(1 for other in links[node] if other in left), -left.index(node)))
        order.append(pick)
        left.remove(pick)
    return order


def _radial_layout(ids: List[str], adjacency: Dict[str, List[str]], focus: str,
                   width: float, height: float, spacing: float = 0.0,
                   spoke_size: Optional[Mapping[str, Tuple[float, float]]] = None,
                   wedges: bool = True) -> Dict[str, Tuple[float, float]]:
    """The focus in the centre, the others on rings by their distance from it. With ``spacing``
    a crowded ring moves out until its members sit that far apart along it. ``spoke_size``: the
    label size of the focus's line to each name on the first ring (see ``_spread_spokes``).
    ``wedges``: names past the first ring sit in the wedge of the name they hang from (their lines
    stay short and apart; a crowded wedge moves its ring out), else evenly round their ring (for a
    diagram that labels and clears the focus's lines only: the rings stay as small as they can)."""
    depth = {focus: 0}
    parent: Dict[str, Optional[str]] = {focus: None}
    order = [focus]
    for node in order:
        for neighbour in adjacency.get(node, []):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                parent[neighbour] = node
                order.append(neighbour)
    max_depth = max(depth.values()) if depth else 0
    unreachable = [i for i in ids if i not in depth]
    rings: Dict[int, List[str]] = {}
    children: Dict[str, List[str]] = {}
    for node in order[1:]:
        rings.setdefault(depth[node], []).append(node)
        children.setdefault(parent[node], []).append(node)
    # how many names hang below each (itself included): each gets a wedge of its parent's that wide
    below: Dict[str, int] = {}
    for node in reversed(order):
        below[node] = 1 + sum(below[child] for child in children.get(node, ()))
    if unreachable:
        max_depth += 1
        rings[max_depth] = unreachable
    radius = min(width, height) / 2.0 * 0.88
    step = radius / max_depth if max_depth else 0.0
    cx, cy = width / 2.0, height / 2.0
    wedge: Dict[str, Tuple[float, float]] = {}
    angle: Dict[str, float] = {focus: 0.0}
    positions = {focus: (cx, cy)}
    previous = 0.0
    for ring in sorted(rings):
        members = rings[ring]
        count = len(members)
        thetas: List[float] = []
        outside = bool(unreachable) and ring == max_depth
        if ring == 1 and not outside:
            members[:] = _ring_order(sorted(members), adjacency)
            thetas, gaps = _spread_spokes([2.0 * math.pi * index / count for index in range(count)],
                                          [(spoke_size or {}).get(node) for node in members],
                                          [below[node] for node in members])
            for index, node in enumerate(members):
                low, high = thetas[index] - gaps[index - 1] / 2.0, thetas[index] + gaps[index] / 2.0
                if high - low > _WEDGE_MAX:
                    # (a focus with one or two relations) not round the far side of the focus: the
                    # lines from this name to its own would run through the focus there
                    low, high = thetas[index] - _WEDGE_MAX / 2.0, thetas[index] + _WEDGE_MAX / 2.0
                wedge[node] = (low, high)
        elif not wedges and not outside:
            # evenly round the ring, in the order of the names they hang from; past a focus with one
            # relation not round its far side (see above)
            members.sort(key=lambda n: (angle.get(parent[n], 10.0), n))
            if ring == 2 and len(rings[1]) == 1:
                middle = angle[rings[1][0]]
                thetas = [middle - _WEDGE_MAX / 2.0 + _WEDGE_MAX * (index + 0.5) / count for index in range(count)]
            else:
                offset = (math.pi / count) * (ring % 2)
                thetas = [offset + 2.0 * math.pi * index / count for index in range(count)]
        elif not outside:
            # each name's children share its wedge, as wide as what hangs below each of them
            members.sort(key=lambda n: (wedge[parent[n]][0], n))
            placed = []
            for owner in dict.fromkeys(parent[n] for n in members):
                low, high = wedge[owner]
                kids = sorted(children[owner], key=lambda n: n)
                total = sum(below[kid] for kid in kids)
                start = low
                for kid in kids:
                    end = start + (high - low) * below[kid] / total
                    wedge[kid] = (start, end)
                    placed.append(kid)
                    start = end
            members[:] = placed
            thetas = [(wedge[node][0] + wedge[node][1]) / 2.0 for node in members]
        else:  # names not related to the focus at all: evenly round the outside
            offset = (math.pi / count) * (ring % 2)
            thetas = [offset + 2.0 * math.pi * index / count for index in range(count)]
        distance = step * ring
        stagger = 0.0
        if spacing > 0.0:
            even = count * spacing / (2.0 * math.pi)
            if ring > 1 and members and all(node in wedge for node in members):
                # names share their parent's wedge: far enough out that each wedge holds its names
                crowded = max(spacing / max(wedge[node][1] - wedge[node][0], 1e-9) for node in members)
                even = max(even, min(crowded, 4.0 * even))
            distance = max(distance, previous + step, even)
            inside = set(members)
            if ring == 1 and any(other in inside for node in members for other in adjacency.get(node, [])):
                # names related to each other side by side on one circle: the lines from one of them
                # to the next two would run almost together, so they sit at three distances
                stagger = _RING_STAGGER
        previous = distance * (1.0 + 2.0 * stagger)
        for index, node in enumerate(members):
            theta = thetas[index]
            angle[node] = theta
            reach = distance * (1.0 + stagger * (index % 3))
            positions[node] = (cx + reach * math.cos(theta), cy + reach * math.sin(theta))
    return positions


#: A name's children fan out over at most this share of the full turn, centred away from its parent.
_FAN_SHARE = 0.8
#: Children related to other children sit at least this many times as far out as the furthest
#: child without (the lines between them stay outside the labels round the parent).
_TREE_OUTER = 1.25

def _tree_layout(ids: Sequence[str], adjacency: Mapping[str, Sequence[str]], root: str,
                 node_size: Mapping[str, Tuple[float, float]],
                 label_of: Callable[[str, str], Optional[Tuple[float, float]]]) -> Dict[str, List[float]]:
    """Every name round the one it hangs from (a tree from ``root``; a name hangs from its related
    name nearer the root with most relations), ``root`` at ``(0, 0)``: the children of a name fan
    out round it (all round for the root, away from its parent for the others), in an order where
    children related to each other sit side by side, each as far out as the label of its line
    needs to clear the lines beside it (``label_of(a, b)``: that label's size) and as its own
    children need room. Children with relations to other children sit further out than those
    without, so the lines between them run outside the labels round the parent. Names that do not
    reach ``root`` go round the outside, each group the same way round its busiest name. The
    result lists the names out from the root, each name's children in their order round it."""
    clear = LINE_CLEARANCE + _ROOM_GAP
    depth = {root: 0}
    queue = [root]
    for node in queue:
        for other in adjacency.get(node, ()):
            if other not in depth:
                depth[other] = depth[node] + 1
                queue.append(other)
    parent: Dict[str, Optional[str]] = {root: None}
    children: Dict[str, List[str]] = {node: [] for node in queue}
    for node in queue[1:]:
        ups = [other for other in adjacency.get(node, ()) if depth.get(other) == depth[node] - 1]
        up = max(ups, key=lambda other: (len(adjacency.get(other, ())), other))
        parent[node] = up
        children[up].append(node)
    below: Dict[str, int] = {}
    for node in reversed(queue):
        below[node] = 1 + sum(below[kid] for kid in children[node])

    def extent(node: str, theta: float) -> float:
        """How far a name's box reaches from its centre in the direction ``theta``."""
        w, h = node_size.get(node, (0.0, 0.0))
        return abs(math.cos(theta)) * w / 2.0 + abs(math.sin(theta)) * h / 2.0

    # bottom up: where each child sits relative to its parent, and how far each subtree reaches
    reach: Dict[str, float] = {}
    offset: Dict[str, Tuple[float, float]] = {}  # child: (angle from the fan's middle, distance)
    for node in reversed(queue):
        kids = children[node]
        w, h = node_size.get(node, (0.0, 0.0))
        own = math.hypot(w, h) / 2.0 + clear
        if not kids:
            reach[node] = own
            continue
        kids[:] = _ring_order(sorted(kids), {k: list(adjacency.get(k, ())) for k in kids})
        count = len(kids)
        full = node == root
        span =2.0 * math.pi if full else min(2.0 * math.pi * _FAN_SHARE, max(math.pi / 2.0, 0.5 * math.pi * count))
        sizes = [label_of(node, kid) or (0.0, 0.0) for kid in kids]
        if full:
            # all round: label-aware gaps (see ``_spread_spokes``), wider for names with more below
            angles, _gaps = _spread_spokes([2.0 * math.pi * k / count for k in range(count)], sizes,
                                           [below[kid] for kid in kids])
        else:
            # a fan whose middle points at angle 0 (turned away from the parent later); a big
            # subtree gets a share as wide as it needs
            weights = [max(lh / 2.0 + clear, lw / 4.0 + clear) + reach[kid] / 2.0 for kid, (lw, lh) in zip(kids, sizes)]
            total = sum(weights)
            angles = []
            run = 0.0
            for value in weights:
                share = value / total * span
                angles.append(run + share / 2.0 - span / 2.0)
                run += share
        gaps = []
        for k in range(count):
            if count == 1:
                gaps.append(span / 2.0 if not full else math.pi)
                continue
            # (the end children of a fan: the gap to the parent's line beyond the fan's edge)
            outside = math.pi * (1 - _FAN_SHARE)
            wrap = angles[0] + 2.0 * math.pi - angles[-1]
            before = angles[k] - angles[k - 1] if k > 0 else (wrap if full else angles[0] + span / 2.0 + outside)
            after = angles[k + 1] - angles[k] if k + 1 < count else (wrap if full else span / 2.0 - angles[k] + outside)
            gaps.append(min(before, after))
        linked = {kid for kid in kids if any(other in kids for other in adjacency.get(kid, ()))}
        distances = []
        for kid, theta, gap, (lw, lh) in zip(kids, angles, gaps, sizes):
            if full:
                along = abs(math.cos(theta)) * lw / 2.0 + abs(math.sin(theta)) * lh / 2.0
                sideways = abs(math.sin(theta)) * lw / 2.0 + abs(math.cos(theta)) * lh / 2.0 + clear
            else:
                # (a fan is turned later: the label's own angle is not known yet, so a middling case)
                along = max(lw, lh) / 2.0
                sideways = max(lw, lh) / 2.0 * 0.7 + clear
            pill = along + sideways / max(math.sin(min(gap, math.pi / 2.0)), 0.05)
            pill = max(pill, own + along + clear)
            distance = pill + along + clear + math.hypot(*node_size.get(kid, (0.0, 0.0))) / 2.0
            if len(kids) > 1 and reach[kid] > own:
                # a child with names below it sits out far enough that their room clears the lines
                # beside it, but not further than its own reach past the label (a narrow gap is the
                # placing's to settle, not a reason to carry a whole branch far out)
                wide = (reach[kid] + clear) / max(math.sin(min(gap, math.pi / 2.0)), 0.05)
                distance = max(distance, min(wide, distance + reach[kid]))
            distances.append(distance)
        plain = [d for kid, d in zip(kids, distances) if kid not in linked and not children[kid]]
        if plain and linked:
            outer = max(plain) * _TREE_OUTER
            distances = [max(d, outer) if kid in linked else d for kid, d in zip(kids, distances)]
        for kid, theta, distance in zip(kids, angles, distances):
            offset[kid] = (theta, distance)
        reach[node] = max(own, max(d + reach[kid] for kid, d in zip(kids, distances)))
    # top down: positions, each fan turned to point away from the parent
    positions: Dict[str, List[float]] = {root: [0.0, 0.0]}
    heading: Dict[str, float] = {root: 0.0}
    for node in queue[1:]:
        up = parent[node]
        theta, distance = offset[node]
        angle = theta + (heading[up] if parent[up] is not None else 0.0)
        heading[node] = angle
        px, py = positions[up]
        positions[node] = [px + distance * math.cos(angle), py + distance * math.sin(angle)]
    # the order to place them in (see ``_place_one_by_one``): out from the root, the names with most
    # relations first (they need the most room), else in their order round their parents
    order = [root]
    for node in order:
        order.extend(children[node])
    rank = {node: index for index, node in enumerate(order)}
    order.sort(key=lambda node: (depth[node], -len(adjacency.get(node, ())), rank[node]))
    positions = {node: positions[node] for node in order}
    rest = [node for node in ids if node not in depth]
    if rest:
        # the names that do not reach the root: each group round its busiest name, under the picture
        first = max(rest, key=lambda node: (len(adjacency.get(node, ())), node))
        group = _tree_layout(rest, adjacency, first, node_size, label_of)
        low_x = min(p[0] for p in group.values())
        low_y = min(p[1] for p in group.values())
        for node, (gx, gy) in group.items():
            positions[node] = [gx - low_x - reach[root], gy - low_y + reach[root] + 4.0 * clear]
    return positions


def _spread_spokes(thetas: List[float], sizes: Sequence[Optional[Tuple[float, float]]],
                   shares: Sequence[int]) -> Tuple[List[float], List[float]]:
    """``(directions, gaps)``: directions round a name for its lines, in the same order, and the
    angle from each to the next. The gap between two lines is as wide as their labels need (a
    label of width w on a line at angle a reaches ``|sin a| w / 2 + |cos a| h / 2`` sideways, so
    near-vertical lines with their wide, flat labels get wide gaps and near-horizontal ones narrow
    gaps), and at least as wide as the names hanging below the two (``shares``) need."""
    count = len(thetas)
    if count < 2:
        return thetas, [2.0 * math.pi] * count
    start = thetas[0]
    gaps = [2.0 * math.pi / count] * count
    for _round in range(6):
        reach = []
        for theta, size in zip(thetas, sizes):
            w, h = size or (0.0, 0.0)
            reach.append(abs(math.sin(theta)) * w / 2.0 + abs(math.cos(theta)) * h / 2.0 + LINE_CLEARANCE)
        unit = sum(reach) / count
        need = [max(reach[i], reach[(i + 1) % count], unit * (shares[i] + shares[(i + 1) % count]) / 2.0)
                for i in range(count)]
        total = sum(need)
        gaps = [2.0 * math.pi * value / total for value in need]
        thetas = [start]
        for gap in gaps[:-1]:
            thetas.append(thetas[-1] + gap)
    return thetas, gaps


#: Up to this many characters the layout is tried several times and the tidiest one kept.
_REFINED_LAYOUT_MAX_NODES = 40
_REFINED_LAYOUT_ITERATIONS = 200
#: Pair-repulsion evaluations spent on one small graph (keeps a phone under about a second).
_REFINED_LAYOUT_BUDGET = 300_000
_REFINED_LAYOUT_MAX_TRIES = 8  # measured: as tidy as 16 on a real book's shape, in half the time
#: Past this many lines no move untangles much, and trying costs seconds.
_UNTANGLE_MAX_EDGES = 60
#: How many of the tried layouts (the tidiest ones) get the untangling pass.
_UNTANGLE_TRIES = 2


def _force_positions(ids: List[str], edge_list: List[Tuple[str, str]], width: float, height: float,
                     rng: random.Random, iterations: int) -> Dict[str, List[float]]:
    """Force layout where every pair of nodes repels (small graphs only: quadratic)."""
    n = len(ids)
    index = {node: i for i, node in enumerate(ids)}
    links = [(index[a], index[b]) for a, b in edge_list]
    # Which connected group each node is in: separate groups only push each other away
    # at close range, so they end up side by side and not in opposite corners.
    group = list(range(n))

    def _root(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
    for i, j in links:
        group[_root(i)] = _root(j)
    group = [_root(i) for i in range(n)]
    degree = [0] * n
    for i, j in links:
        degree[i] += 1
        degree[j] += 1
    k = math.sqrt(width * height / n)
    k2 = k * k
    apart2 = (1.3 * k) ** 2
    cx, cy = width / 2.0, height / 2.0
    radius = min(width, height) * 0.4
    order = list(range(n))
    rng.shuffle(order)  # each try starts from a different arrangement around the circle
    xs = [0.0] * n
    ys = [0.0] * n
    for slot, i in enumerate(order):
        theta = 2.0 * math.pi * slot / n
        jitter = 0.85 + 0.3 * rng.random()
        xs[i] = cx + radius * jitter * math.cos(theta)
        ys[i] = cy + radius * jitter * math.sin(theta)
    temperature = min(width, height) / 8.0
    cooling = temperature / (iterations + 1)
    # A pull to the centre as strong as the distance: groups with no line between them
    # settle beside each other instead of drifting to opposite corners.
    gravity = 1.0
    for _iteration in range(iterations):
        dxs = [0.0] * n
        dys = [0.0] * n
        for i in range(n):
            xi, yi = xs[i], ys[i]
            for j in range(i + 1, n):
                ddx, ddy = xi - xs[j], yi - ys[j]
                dist2 = ddx * ddx + ddy * ddy
                if dist2 > apart2 and group[i] != group[j]:
                    continue
                if dist2 < 1e-6:
                    ddx, ddy = rng.random() - 0.5, rng.random() - 0.5
                    dist2 = ddx * ddx + ddy * ddy + 1e-6
                factor = k2 / dist2
                fx, fy = ddx * factor, ddy * factor
                dxs[i] += fx
                dys[i] += fy
                dxs[j] -= fx
                dys[j] -= fy
        for i, j in links:
            ddx, ddy = xs[i] - xs[j], ys[i] - ys[j]
            # Weaker between two well-connected names: a tight group keeps lines long
            # enough to carry their labels instead of collapsing into a knot.
            force = (math.sqrt(ddx * ddx + ddy * ddy) or 1e-3) / (k * math.sqrt(min(degree[i], degree[j])))
            fx, fy = ddx * force, ddy * force
            dxs[i] -= fx
            dys[i] -= fy
            dxs[j] += fx
            dys[j] += fy
        for i in range(n):
            dx = dxs[i] + (cx - xs[i]) * gravity
            dy = dys[i] + (cy - ys[i]) * gravity
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-9:
                limited = min(length, temperature)
                xs[i] += dx / length * limited
                ys[i] += dy / length * limited
        temperature = max(temperature - cooling, 1.0)
    # Names are drawn as wide boxes: keep every pair a box apart (more so side by side).
    near = 0.6 * k
    for _pass in range(40):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                ddx, ddy = (xs[i] - xs[j]) / 1.6, ys[i] - ys[j]
                dist = math.sqrt(ddx * ddx + ddy * ddy)
                if dist >= near:
                    continue
                if dist < 1e-6:
                    ddx, ddy, dist = rng.random() - 0.5, rng.random() - 0.5, 1.0
                push = (near - dist) / dist / 2.0
                xs[i] += ddx * push * 1.6
                ys[i] += ddy * push
                xs[j] -= ddx * push * 1.6
                ys[j] -= ddy * push
                moved = True
        if not moved:
            break
    return {node: [xs[i], ys[i]] for node, i in index.items()}


def _segments_cross(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float], p4: Sequence[float]) -> bool:
    def _side(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = _side(p3, p4, p1), _side(p3, p4, p2)
    d3, d4 = _side(p1, p2, p3), _side(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = min(1.0, max(0.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def layout_untidiness(positions: Mapping[str, Sequence[float]], edge_list: Sequence[Tuple[str, str]]) -> float:
    """How hard a layout is to read: crossing lines, lines running through a name, names too close.

    Distances are judged against the average line length, so the number does not depend on
    the size the layout is drawn at.
    """
    lengths = [math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
               for a, b in edge_list]
    unit = (sum(lengths) / len(lengths)) if lengths else 0.0
    if unit <= 1e-9:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        unit = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) / max(1.0, math.sqrt(len(positions)))
    score = 0.0
    for i, (a, b) in enumerate(edge_list):
        pa, pb = positions[a], positions[b]
        for c, d in edge_list[i + 1:]:
            if a in (c, d) or b in (c, d):
                continue
            if _segments_cross(pa, pb, positions[c], positions[d]):
                score += 1.0
        for node, point in positions.items():
            if node != a and node != b and _point_segment_distance(point, pa, pb) < 0.22 * unit:
                score += 2.0
    nodes = list(positions)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]) < 0.45 * unit:
                score += 1.5
    return score


def _untangle(positions: Dict[str, List[float]], edge_list: List[Tuple[str, str]]) -> None:
    """Move one name at a time to a nearby spot where its lines cross fewer others.

    A force layout draws five people who are all related to each other as a star with five
    crossings; putting one of them inside the other four leaves a single crossing. Only
    what the moved name changes is counted, so a pass stays cheap.
    """
    if len(positions) < 4 or len(edge_list) < 2 or len(edge_list) > _UNTANGLE_MAX_EDGES:
        return
    lengths = [math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
               for a, b in edge_list]
    unit = sum(lengths) / len(lengths)
    if unit <= 1e-9:
        return
    own: Dict[str, List[Tuple[str, str]]] = {node: [] for node in positions}
    for edge in edge_list:
        own[edge[0]].append(edge)
        own[edge[1]].append(edge)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    # Stay inside the area the layout already uses, so a move never stretches the picture.
    low_x, high_x, low_y, high_y = min(xs), max(xs), min(ys), max(ys)

    def _cost(node: str, point: Sequence[float]) -> float:
        cost = 0.0
        mine = own[node]
        for a, b in mine:
            other = b if a == node else a
            end = positions[other]
            length = math.hypot(point[0] - end[0], point[1] - end[1])
            if length < 0.6 * unit:
                cost += 3.0  # too short to carry its label
            for c, d in edge_list:
                if node in (c, d) or other in (c, d):
                    continue
                if _segments_cross(point, end, positions[c], positions[d]):
                    cost += 1.0
            for third, spot in positions.items():
                if third != node and third != other and _point_segment_distance(spot, point, end) < 0.22 * unit:
                    cost += 2.0
        for a, b in edge_list:
            if node not in (a, b) and _point_segment_distance(point, positions[a], positions[b]) < 0.22 * unit:
                cost += 2.0
        for third, spot in positions.items():
            if third != node and math.hypot((point[0] - spot[0]) / 1.6, point[1] - spot[1]) < 0.45 * unit:
                cost += 3.0
        return cost

    ring = [(math.cos(2.0 * math.pi * step / 12), math.sin(2.0 * math.pi * step / 12)) for step in range(12)]
    for _round in range(3):
        improved = False
        for node in sorted(positions, key=lambda item: -len(own[item])):
            if not own[node]:
                continue
            here = positions[node]
            current = _cost(node, here)
            if current <= 0.0:
                continue
            ends = [positions[b if a == node else a] for a, b in own[node]]
            centre = (sum(p[0] for p in ends) / len(ends), sum(p[1] for p in ends) / len(ends))
            spots = [centre]
            for radius in (0.35, 0.7, 1.1):
                spots += [(centre[0] + ux * radius * unit, centre[1] + uy * radius * unit) for ux, uy in ring]
            spots += [(here[0] + ux * 0.5 * unit, here[1] + uy * 0.5 * unit) for ux, uy in ring]
            best_spot, best_cost = None, current
            for spot in spots:
                if not (low_x <= spot[0] <= high_x and low_y <= spot[1] <= high_y):
                    continue
                cost = _cost(node, spot)
                if cost < best_cost:
                    best_spot, best_cost = spot, cost
            if best_spot is not None:
                positions[node] = [best_spot[0], best_spot[1]]
                improved = True
        if not improved:
            break


def _refined_layout(ids: List[str], edge_list: List[Tuple[str, str]], width: float, height: float,
                    seed: int) -> Dict[str, List[float]]:
    """The tidiest of several force layouts (see ``layout_untidiness``); same result for the same seed."""
    n = len(ids)
    pairs = max(1, n * (n - 1) // 2)
    tries = max(1, min(_REFINED_LAYOUT_MAX_TRIES,
                       _REFINED_LAYOUT_BUDGET // (_REFINED_LAYOUT_ITERATIONS * pairs)))
    # The force passes are cheap (8 of them on 15 names: 0.05 s); untangling is not (0.15 s
    # each), so only the most promising few are untangled. Untangling all of them made a
    # 15-name diagram take 1.3 s to open.
    scored = []
    for attempt in range(tries):
        rng = random.Random(int(seed) * 1_000_003 + attempt)
        positions = _force_positions(ids, edge_list, width, height, rng, _REFINED_LAYOUT_ITERATIONS)
        score = layout_untidiness(positions, edge_list)
        scored.append((score, attempt, positions))
        if score == 0.0:
            return positions
    scored.sort(key=lambda item: (item[0], item[1]))
    best_score, _attempt, best = scored[0]
    for score, _attempt, positions in scored[:_UNTANGLE_TRIES]:
        _untangle(positions, edge_list)
        score = layout_untidiness(positions, edge_list)
        if score < best_score:
            best, best_score = positions, score
    return best


#: Where along its line a label sits when the layout made no room for it (or it does not fit there),
#: as fractions of the way from one end (middle first): the first spot clear of the names and the
#: other labels wins. The layout's own spots (``layout_spot``) may be anywhere on the line.
LABEL_SPOTS = (0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.2, 0.8)
#: Above this many lines the layout makes room for the name boxes and the focus character's labels
#: and lines only (both diagrams then label only that character's lines).
ROOM_FOR_LABELS_MAX_EDGES = 120
#: Space kept between two boxes, so text drawn a little wider than estimated still fits.
_ROOM_GAP = 4.0
#: How far every line stays from a label that is not its own and from a name it does not end at
#: (scale 1): 6 px that show at a glance, half the widest line (2.2 px), and a pixel for text drawn
#: a little wider than estimated.
LINE_CLEARANCE = 8.0
#: A problem growing the picture cannot fix: a name (or label) nearer a line than this share of the
#: room it needs, two lines from one name less than this sine apart.
_STUCK = 0.05
#: How far such an end turns round the shared name (radians, about 12 degrees).
_STUCK_TURN = 0.21
#: Two lines from one name that only growing the whole picture this many times over would part
#: turn apart too.
_TURN_NEED = 4.0
#: One growth step: every name at least this many times as far from the middle, at most ``_ROOM_GROW_MAX``.
_ROOM_GROW = 1.08
_ROOM_GROW_MAX = 1.5
#: Growth steps at most (1.08 ** 90: 1,000 times the size): the guarantee for what is left.
_GROW_ROUNDS = 90
#: Single names are moved out of the way while at most this many problems are left ...
_RELOCATE_MAX = 40
#: ... and until this many tries in a row at one size left as many problems.
_RELOCATE_TRIES = 3
#: The worst problems tried for room around them alone before the whole picture grows.
_LOCAL_TRIES = 1
#: While more problems are left, only those needing this many times the growth most others need
#: (and at least this much) are tried.
_HARD_NEED = 3.0
#: Spots tried for single names in one layout, in all (keeps a big book's layout under a second).
_RELOCATE_BUDGET = 800
#: Names tried in one go.
_RELOCATE_NAMES = 12
#: Names with more lines than this are not tried (each spot tried costs a look at all their lines).
_RELOCATE_MAX_LINES = 12
#: A name with at least this many relations that have no other relation fans them out (see ``_fan_out``).
_FAN_MIN = 4
#: Halvings when the grown picture is shrunk back to the smallest size that stays clear.
_SHRINK_STEPS = 5

Box = Tuple[float, float, float, float]


class _BoxGrid:
    """Boxes ``(x0, y0, x1, y1)`` filed in a coarse grid, to find what a box overlaps quickly."""

    def __init__(self, cell: float) -> None:
        self.cell = max(float(cell), 1.0)
        self.cells: Dict[Tuple[int, int], List[int]] = {}
        self.boxes: List[Box] = []

    def _cells(self, box: Box) -> List[Tuple[int, int]]:
        c = self.cell
        ys = range(int(box[1] // c), int(box[3] // c) + 1)
        return [(gx, gy) for gx in range(int(box[0] // c), int(box[2] // c) + 1) for gy in ys]

    def add(self, box: Box) -> int:
        index = len(self.boxes)
        self.boxes.append(box)
        cells = self.cells
        for key in self._cells(box):
            if key in cells:
                cells[key].append(index)
            else:
                cells[key] = [index]
        return index

    def hits(self, box: Box) -> List[Tuple[int, float]]:
        """``[(index, shared area)]`` of the boxes ``box`` overlaps (touching edges do not count)."""
        keys = self._cells(box)
        cells = self.cells
        if len(keys) == 1:
            candidates = cells.get(keys[0], ())
        else:
            candidates = set()
            for key in keys:
                candidates.update(cells.get(key, ()))
            candidates = sorted(candidates)
        x0, y0, x1, y1 = box
        found = []
        boxes = self.boxes
        for index in candidates:
            ox0, oy0, ox1, oy1 = boxes[index]
            w = (x1 if x1 < ox1 else ox1) - (x0 if x0 > ox0 else ox0)
            if w <= 0.0:
                continue
            h = (y1 if y1 < oy1 else oy1) - (y0 if y0 > oy0 else oy0)
            if h > 0.0:
                found.append((index, w * h))
        return found

    def overlap(self, box: Box) -> float:
        return sum(area for _index, area in self.hits(box))


def label_spot(pa: Sequence[float], pb: Sequence[float], width: float, height: float,
               overlap: Callable[[Box], float], spots: Sequence[float] = LABEL_SPOTS) -> Tuple[float, float, float]:
    """``(cx, cy, t)``: the centre of a ``width`` x ``height`` label ON the line ``pa``-``pb``.

    The first of ``spots`` (fractions of the way from ``pa``) where ``overlap(box)`` is 0 wins,
    else the spot with the least overlap: the label slides along its line, never off it.
    """
    best = None
    for t in spots:
        cx = pa[0] + (pb[0] - pa[0]) * t
        cy = pa[1] + (pb[1] - pa[1]) * t
        area = overlap((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0))
        if best is None or area < best[0]:
            best = (area, cx, cy, t)
        if area <= 0.0:
            break
    return best[1], best[2], best[3]


def _placed_boxes(positions: Mapping[str, Sequence[float]], edges: Sequence[Tuple[Any, str, str]],
                  node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                  gap: float = 0.0, prefer: Optional[Mapping[Any, float]] = None
                  ) -> Tuple[_BoxGrid, List[Tuple[Any, ...]]]:
    """Every name box, then every label placed by ``label_spot`` (shorter lines choose first; a
    label tries its spot in ``prefer`` before ``LABEL_SPOTS``), each box grown by ``gap``. Returns
    the grid and what each box is: ``("n", id)`` or ``("l", key, a, b, t)``."""
    prefer = prefer or {}
    sizes = list(node_size.values()) + list(label_size.values())
    grid = _BoxGrid(max(max(w for w, _h in sizes), max(h for _w, h in sizes)) if sizes else 100.0)
    owners: List[Tuple[Any, ...]] = []
    half = gap / 2.0
    for node, (w, h) in node_size.items():
        point = positions.get(node)
        if point is None:
            continue
        grid.add((point[0] - w / 2.0 - half, point[1] - h / 2.0 - half,
                  point[0] + w / 2.0 + half, point[1] + h / 2.0 + half))
        owners.append(("n", node))
    todo = []
    for key, a, b in edges:
        size = label_size.get(key)
        pa, pb = positions.get(a), positions.get(b)
        if size is None or pa is None or pb is None:
            continue
        todo.append((math.hypot(pb[0] - pa[0], pb[1] - pa[1]), len(todo), key, a, b, pa, pb, size))
    for _length, _order, key, a, b, pa, pb, (w, h) in sorted(todo, key=lambda item: (item[0], item[1])):
        w, h = w + gap, h + gap
        spot = prefer.get(key)
        cx, cy, t = label_spot(pa, pb, w, h, grid.overlap, LABEL_SPOTS if spot is None else (spot,) + LABEL_SPOTS)
        grid.add((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0))
        owners.append(("l", key, a, b, t))
    return grid, owners


def place_labels(positions: Mapping[str, Sequence[float]], edges: Sequence[Tuple[Any, str, str]],
                 node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                 prefer: Optional[Mapping[Any, float]] = None) -> Dict[Any, Tuple[float, float]]:
    """``{key: (cx, cy)}`` for the labels of ``edges`` (``[(key, a, b)]``) of the sizes in
    ``label_size``, clear of the name boxes of ``node_size`` and of each other where the line allows.
    ``prefer`` (``{key: t}``, e.g. from ``layout_spots``) is tried first: the spot the layout made room for."""
    grid, owners = _placed_boxes(positions, edges, node_size, label_size, prefer=prefer)
    placed = {}
    for owner, box in zip(owners, grid.boxes):
        if owner[0] == "l":
            placed[owner[1]] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return placed


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-9 else (1.0, 0.0)


def _cells_along(p: Sequence[float], q: Sequence[float], cell: float) -> List[Tuple[int, int]]:
    """The grid cells the segment ``p``-``q`` passes through, in order (a grid walk, not its bounding box)."""
    x, y = p[0], p[1]
    gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
    tx, ty = int(math.floor(q[0] / cell)), int(math.floor(q[1] / cell))
    cells = [(gx, gy)]
    dx, dy = q[0] - x, q[1] - y
    step_x, step_y = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    t_max_x = (((gx + (1 if dx > 0 else 0)) * cell - x) / dx) if dx else _BIG
    t_max_y = (((gy + (1 if dy > 0 else 0)) * cell - y) / dy) if dy else _BIG
    t_dx = cell / abs(dx) if dx else _BIG
    t_dy = cell / abs(dy) if dy else _BIG
    for _step in range(abs(tx - gx) + abs(ty - gy)):
        if t_max_x < t_max_y:
            gx += step_x
            t_max_x += t_dx
        else:
            gy += step_y
            t_max_y += t_dy
        cells.append((gx, gy))
    return cells


def _hit_interval(px: float, py: float, dx: float, dy: float,
                  planes: Sequence[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
    """``(lo, hi)``: where along ``p + t * d`` (t in 0..1) a point is inside every half-plane
    ``mx * x + my * y <= s`` of ``planes``; None when nowhere."""
    lo, hi = 0.0, 1.0
    for mx, my, s in planes:
        k = mx * dx + my * dy
        r = s - (mx * px + my * py)
        if -1e-12 < k < 1e-12:
            if r < 0.0:
                return None
            continue
        v = r / k
        if k > 0.0:
            if v < hi:
                hi = v
        elif v > lo:
            lo = v
        if lo >= hi:
            return None
    return lo, hi


def _box_planes(box: Box, half_w: float, half_h: float) -> Tuple[Tuple[float, float, float], ...]:
    """Where the centre of a ``2 * half_w`` x ``2 * half_h`` box overlaps ``box``."""
    return ((1.0, 0.0, box[2] + half_w), (-1.0, 0.0, half_w - box[0]),
            (0.0, 1.0, box[3] + half_h), (0.0, -1.0, half_h - box[1]))


LineGeo = Tuple[float, float, float, float, float, float, float]


def _line_geo(p: Sequence[float], q: Sequence[float]) -> LineGeo:
    """What the clearance tests need of the line ``p``-``q``, worked out once: its bounding box,
    unit normal and offset along that normal.

    A box of half sizes ``hw`` x ``hh`` comes within ``LINE_CLEARANCE`` of the line when its centre
    is inside the line widened by the box grown by the clearance: a six-sided shape bounded by the
    line's bounding box grown by ``hw + c`` / ``hh + c`` and by two lines along it, ``|nx| (hw + c) +
    |ny| (hh + c)`` to either side (square corners: a little more room than a round clearance,
    never less)."""
    nx, ny = _unit(-(q[1] - p[1]), q[0] - p[0])
    return (min(p[0], q[0]), max(p[0], q[0]), min(p[1], q[1]), max(p[1], q[1]), nx, ny, nx * p[0] + ny * p[1])


def _near_line(x: float, y: float, geo: LineGeo, half_w: float, half_h: float) -> bool:
    """A ``2 * half_w`` x ``2 * half_h`` box centred on ``(x, y)`` comes within the clearance of the
    line (see ``_line_geo``)."""
    big_w, big_h = half_w + LINE_CLEARANCE, half_h + LINE_CLEARANCE
    x0, x1, y0, y1, nx, ny, across = geo
    if x <= x0 - big_w or x >= x1 + big_w or y <= y0 - big_h or y >= y1 + big_h:
        return False
    return abs(nx * x + ny * y - across) < abs(nx) * big_w + abs(ny) * big_h


def _line_interval(px: float, py: float, dx: float, dy: float, geo: LineGeo, half_w: float, half_h: float
                   ) -> Optional[Tuple[float, float]]:
    """Where along ``p + t * d`` a box centred there comes within the clearance of the line."""
    big_w, big_h = half_w + LINE_CLEARANCE, half_h + LINE_CLEARANCE
    x0, x1, y0, y1, nx, ny, across = geo
    ex, ey = px + dx, py + dy
    if ((px if px > ex else ex) <= x0 - big_w or (px if px < ex else ex) >= x1 + big_w
            or (py if py > ey else ey) <= y0 - big_h or (py if py < ey else ey) >= y1 + big_h):
        return None
    reach = abs(nx) * big_w + abs(ny) * big_h
    side_start = nx * px + ny * py - across
    side_end = side_start + nx * dx + ny * dy
    if (side_start >= reach and side_end >= reach) or (side_start <= -reach and side_end <= -reach):
        return None  # the whole way stays to one side of the line, further than the clearance
    return _hit_interval(px, py, dx, dy, ((1.0, 0.0, x1 + big_w), (-1.0, 0.0, big_w - x0),
                                          (0.0, 1.0, y1 + big_h), (0.0, -1.0, big_h - y0),
                                          (nx, ny, across + reach), (-nx, -ny, reach - across)))


def _box_interval(px: float, py: float, dx: float, dy: float, box: Box, half_w: float, half_h: float
                  ) -> Optional[Tuple[float, float]]:
    """Where along ``p + t * d`` a box centred there overlaps ``box``."""
    ex, ey = px + dx, py + dy
    if ((px if px > ex else ex) <= box[0] - half_w or (px if px < ex else ex) >= box[2] + half_w
            or (py if py > ey else ey) <= box[1] - half_h or (py if py < ey else ey) >= box[3] + half_h):
        return None
    return _hit_interval(px, py, dx, dy, _box_planes(box, half_w, half_h))


def _free_stretches(intervals: Sequence[Tuple[float, ...]], margin: float) -> List[Tuple[float, float]]:
    """The stretches of 0..1 outside every ``(lo, hi, ...)`` of ``intervals``, each ``margin``
    narrower at both ends (stretches narrower than that are left out)."""
    free = []
    start = 0.0
    for lo, hi in sorted((item[0], item[1]) for item in intervals):
        if lo > start:
            free.append((start, lo))
        if hi > start:
            start = hi
    if start < 1.0:
        free.append((start, 1.0))
    return [(lo + margin, hi - margin) for lo, hi in free if hi - lo > 2.0 * margin]


def _choose_spot(intervals: Sequence[Tuple[float, float, Any]], margin: float, prefer: Optional[float]
                 ) -> Tuple[float, List[Any]]:
    """``(t, what is still in the way there)``: a spot in 0..1 outside every ``(lo, hi, what)``,
    ``margin`` inside the free stretch: ``prefer`` while it is free, else the free spot nearest the
    middle. With no free spot, the one fewest things cover."""
    free = _free_stretches(intervals, margin)
    if prefer is not None and any(lo <= prefer <= hi for lo, hi in free):
        return prefer, []
    if free:
        best = min((min(max(0.5, lo), hi) for lo, hi in free), key=lambda t: (abs(t - 0.5), t))
        return best, []
    # no free spot: where the fewest things overlap (a sweep over the ends of the stretches)
    events = sorted([(lo, 1) for lo, _hi, _what in intervals] + [(hi, -1) for _lo, hi, _what in intervals])
    depth = sum(1 for lo, _hi, _what in intervals if lo <= 0.0)
    best_key, best_t = None, 0.5
    previous = 0.0
    for value, change in events + [(1.0, 0)]:
        value = min(max(value, 0.0), 1.0)
        if value > previous:
            t = min(max(0.5, previous), value)
            key = (depth, abs(t - 0.5))
            if best_key is None or key < best_key:
                best_key, best_t = key, t
            previous = value
        if value > 0.0:
            depth += change
    return best_t, [what for lo, hi, what in intervals if lo < best_t < hi]


def _scale_up(positions: Dict[str, List[float]], factor: float) -> None:
    """Every name ``factor`` times as far from the middle: boxes keep their size, so every gap
    between a label and a line that is not its own grows (unless they meet in a point)."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    for point in positions.values():
        point[0] = cx + (point[0] - cx) * factor
        point[1] = cy + (point[1] - cy) * factor


def _room_round(positions: Dict[str, List[float]], edges: Sequence[Tuple[Any, str, str]],
                node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                name_lines: Sequence[int], spots: Dict[Any, float], cell: float,
                report: Optional[List[Tuple[float, Tuple[str, ...], Any]]] = None,
                pinned: Optional[str] = None, turn_need: float = _BIG,
                lines_seen: Optional[Dict[int, Tuple[Any, ...]]] = None
                ) -> Tuple[int, Dict[str, List[float]], Dict[Any, float]]:
    """One look at the picture: ``(problems, moves, label spots)``. ``lines_seen``: as for
    ``_relocate`` (shared with it).

    Problems are: two boxes (names, labels) overlapping, a line of ``name_lines`` within
    ``LINE_CLEARANCE`` of a name it does not end at, and a label with no spot on its line that is
    clear of the other boxes and of every other line by ``LINE_CLEARANCE``. Every label first looks
    for such a spot along its line (the cheapest fix). ``moves`` fix the problems growing the
    picture cannot fix (a name on a line, two lines from one name in the same direction, two boxes
    on the same spot). ``report`` gets ``(grow, names, moves)`` of every problem: about how many
    times bigger the picture must get to clear it, the names involved and the moves that would fix it.
    Two lines from one name that growing would part only at more than ``turn_need`` times the size
    turn apart too; ``pinned`` (the focus) is never the end that turns.
    """
    clear = LINE_CLEARANCE
    half = _ROOM_GAP / 2.0
    moves: Dict[str, List[float]] = {}

    lines_of: Dict[str, int] = {}
    for _key, a, b in edges:
        lines_of[a] = lines_of.get(a, 0) + 1
        lines_of[b] = lines_of.get(b, 0) + 1

    def settle(need: float, names: Tuple[str, ...], suggested: List[Tuple[str, float, float]], stuck: bool) -> None:
        """One problem: reported, and its moves done when growing cannot fix it."""
        if report is not None:
            report.append((need, names, suggested))
        if not stuck:
            return
        for node, dx, dy in suggested:
            move = moves.get(node)
            if move is None:
                moves[node] = [dx, dy]
            else:
                move[0] += dx
                move[1] += dy

    lines = [(positions[a], positions[b]) for _key, a, b in edges]
    if lines_seen is None:
        lines_seen = {}
    geos = []
    line_cells: List[List[Tuple[int, int]]] = []
    line_grid: Dict[Tuple[int, int], List[int]] = {}
    for index, (p, q) in enumerate(lines):
        ends = (p[0], p[1], q[0], q[1], cell)
        seen = lines_seen.get(index)
        if seen is None or seen[0] != ends:
            seen = lines_seen[index] = (ends, _line_geo(p, q), _cells_along(p, q, cell))
        geos.append(seen[1])
        cells = seen[2]
        line_cells.append(cells)
        for key in cells:
            if key in line_grid:
                line_grid[key].append(index)
            else:
                line_grid[key] = [index]
    grid = _BoxGrid(cell)
    owners: List[Tuple[Any, ...]] = []
    for node, (w, h) in node_size.items():
        point = positions.get(node)
        if point is None:
            continue
        grid.add((point[0] - w / 2.0 - half, point[1] - h / 2.0 - half,
                  point[0] + w / 2.0 + half, point[1] + h / 2.0 + half))
        owners.append(("n", node))
    found = 0

    def box_push(first: Tuple[Any, ...], box: Box, second: Tuple[Any, ...], other: Box) -> None:
        """Move names so the boxes of ``first`` and ``second`` (a name or a label) come apart."""
        dx = (other[0] + other[2] - box[0] - box[2]) / 2.0
        dy = (other[1] + other[3] - box[1] - box[3]) / 2.0
        pen_x = (box[2] - box[0] + other[2] - other[0]) / 2.0 - abs(dx)
        pen_y = (box[3] - box[1] + other[3] - other[1]) / 2.0 - abs(dy)
        # vx, vy: how far second must move away from first to clear it (a pixel more, so pushes
        # that partly cancel still end with the two apart instead of touching)
        if pen_x < pen_y:
            pen = pen_x + 1.0
            vx, vy = (pen if dx >= 0 else -pen), 0.0
        else:
            pen = pen_y + 1.0
            vx, vy = 0.0, (pen if dy >= 0 else -pen)
        if first[0] == "n" and second[0] == "l":
            first, second, vx, vy = second, first, -vx, -vy
        # growing by ``need`` parts them: the centres move apart, the boxes keep their size
        need = min((pen_x + abs(dx)) / max(abs(dx), 1e-9), (pen_y + abs(dy)) / max(abs(dy), 1e-9))
        names = (tuple(first[1:2] if first[0] == "n" else first[2:4])
                 + tuple(second[1:2] if second[0] == "n" else second[2:4]))
        suggested: List[Tuple[str, float, float]] = []
        if first[0] == "n":  # two names
            suggested += [(first[1], -vx / 2.0, -vy / 2.0), (second[1], vx / 2.0, vy / 2.0)]
        elif second[0] == "n":  # a label and a name
            name = second[1]
            if name in (first[2], first[3]):  # its own end: lengthen the line
                far = first[3] if first[2] == name else first[2]
                ux, uy = _unit(positions[name][0] - positions[far][0], positions[name][1] - positions[far][1])
                suggested += [(far, -ux * pen / 2.0, -uy * pen / 2.0), (name, ux * pen / 2.0, uy * pen / 2.0)]
            else:
                suggested += [(name, vx / 2.0, vy / 2.0), (first[2], -vx / 2.0, -vy / 2.0),
                              (first[3], -vx / 2.0, -vy / 2.0)]
        else:  # two labels
            shared = {first[2], first[3]} & {second[2], second[3]}
            if len(shared) == 1:
                hub = shared.pop()
                for edge, sign in ((first, -1.0), (second, 1.0)):
                    far = edge[3] if edge[2] == hub else edge[2]
                    ux, uy = _unit(positions[far][0] - positions[hub][0], positions[far][1] - positions[hub][1])
                    suggested.append((far, ux * pen / 2.0 + sign * vx / 2.0, uy * pen / 2.0 + sign * vy / 2.0))
            else:
                suggested += [(node, -vx / 2.0, -vy / 2.0) for node in (first[2], first[3])]
                suggested += [(node, vx / 2.0, vy / 2.0) for node in (second[2], second[3])]
        # growing the picture parts two boxes whose centres differ
        settle(need, names, suggested, abs(dx) <= 0.5 and abs(dy) <= 0.5)

    def away_from_line(cx: float, cy: float, p: Sequence[float], q: Sequence[float],
                       half_w: float, half_h: float) -> Tuple[float, float, float, float, float]:
        """``(ux, uy, short, s, distance)``: the way from the line to the point, how much nearer
        than the clearance a box there is, where along the line the nearest point is, and how far."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        length2 = dx * dx + dy * dy
        s = 0.0 if length2 < 1e-12 else min(1.0, max(0.0, ((cx - p[0]) * dx + (cy - p[1]) * dy) / length2))
        ox, oy = cx - (p[0] + s * dx), cy - (p[1] + s * dy)
        distance = math.hypot(ox, oy)
        if distance > 1e-6:
            ux, uy = ox / distance, oy / distance
        else:
            ux, uy = _unit(-dy, dx)
        need = abs(ux) * half_w + abs(uy) * half_h + clear
        return ux, uy, max(need - distance, 0.0) + 1.0, s, distance

    # two names
    name_count = len(owners)
    for p in range(name_count):
        box = grid.boxes[p]
        for q, _area in grid.hits(box):
            if q > p:
                found += 1
                box_push(owners[p], box, owners[q], grid.boxes[q])
    # a line through a name it does not end at
    if name_lines:
        wanted = set(name_lines)
        for p in range(name_count):
            node = owners[p][1]
            w, h = node_size[node]
            cx, cy = positions[node]
            reach = max(w, h) / 2.0 + clear
            seen = set()
            for gx in range(int(math.floor((cx - reach) / cell)), int(math.floor((cx + reach) / cell)) + 1):
                for gy in range(int(math.floor((cy - reach) / cell)), int(math.floor((cy + reach) / cell)) + 1):
                    seen.update(line_grid.get((gx, gy), ()))
            for index in sorted(seen):
                _key, a, b = edges[index]
                if node == a or node == b or index not in wanted:
                    continue
                pa, pb = lines[index]
                if not _near_line(cx, cy, geos[index], w / 2.0, h / 2.0):
                    continue
                found += 1
                ux, uy, short, s, distance = away_from_line(cx, cy, pa, pb, w / 2.0, h / 2.0)
                # the name steps off the line, the line a little the other way (the name all the
                # way when it sits on the line: growing never takes it off)
                stuck = distance <= _STUCK * (short + distance)
                if stuck:
                    suggested = [(node, ux * short, uy * short)]
                else:
                    suggested = [(node, ux * short * 0.6, uy * short * 0.6),
                                 (a, -ux * short * 0.4 * (1.0 - s), -uy * short * 0.4 * (1.0 - s)),
                                 (b, -ux * short * 0.4 * s, -uy * short * 0.4 * s)]
                settle((short + distance) / max(distance, 1e-9), (node, a, b), suggested, stuck)
    # every label: a free spot on its line, else what is in the way gets moves
    todo = []
    for index, (key, a, b) in enumerate(edges):
        size = label_size.get(key)
        if size is not None:
            pa, pb = lines[index]
            todo.append((math.hypot(pb[0] - pa[0], pb[1] - pa[1]), index, key, a, b, size))
    placed: Dict[Any, float] = {}
    for length, index, key, a, b, (w, h) in sorted(todo, key=lambda item: (item[0], item[1])):
        pa, pb = lines[index]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        near = set()
        for gx, gy in line_cells[index]:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    near.add((gx + ox, gy + oy))
        near_lines = set()
        near_boxes = set()
        for cell_key in near:
            near_lines.update(line_grid.get(cell_key, ()))
            near_boxes.update(grid.cells.get(cell_key, ()))
        intervals = []
        hw, hh = w / 2.0 + half, h / 2.0 + half
        # the stretch a label on this line can cover, grown by the clearance: anything outside it
        # is skipped without a closer look
        low_x, high_x = min(pa[0], pb[0]) - hw - clear, max(pa[0], pb[0]) + hw + clear
        low_y, high_y = min(pa[1], pb[1]) - hh - clear, max(pa[1], pb[1]) + hh + clear
        for other in sorted(near_lines):
            geo = geos[other]
            if other == index or geo[1] < low_x or geo[0] > high_x or geo[3] < low_y or geo[2] > high_y:
                continue
            span = _line_interval(pa[0], pa[1], dx, dy, geo, w / 2.0, h / 2.0)
            if span is not None:
                intervals.append((span[0], span[1], ("line", other)))
        boxes = grid.boxes
        for other in sorted(near_boxes):
            box = boxes[other]
            if box[2] < low_x or box[0] > high_x or box[3] < low_y or box[1] > high_y:
                continue
            span = _box_interval(pa[0], pa[1], dx, dy, box, hw, hh)
            if span is not None:
                intervals.append((span[0], span[1], ("box", other)))
        t, in_way = _choose_spot(intervals, 1.0 / max(length, 1.0), spots.get(key))
        cx, cy = pa[0] + dx * t, pa[1] + dy * t
        mine = ("l", key, a, b, t)
        box = (cx - hw, cy - hh, cx + hw, cy + hh)
        grid.add(box)
        owners.append(mine)
        placed[key] = t
        for kind, other in in_way:
            found += 1
            if kind == "box":
                box_push(mine, box, owners[other], grid.boxes[other])
                continue
            _okey, c, d = edges[other]
            pc, pd = lines[other]
            ux, uy, short, _s, distance = away_from_line(cx, cy, pc, pd, w / 2.0, h / 2.0)
            shared = {a, b} & {c, d}
            need = (short + distance) / max(distance, 1e-9)
            if len(shared) == 1:
                # two lines from one name: turn them apart and lengthen the label's line, so the
                # label moves out to where the two lines are far enough apart
                hub = shared.pop()
                far, far_other = (b if a == hub else a), (d if c == hub else c)
                ph, pf, po = positions[hub], positions[far], positions[far_other]
                ex, ey = _unit(pf[0] - ph[0], pf[1] - ph[1])
                fx, fy = _unit(po[0] - ph[0], po[1] - ph[1])
                sine = abs(ex * fy - ey * fx)
                from_hub = math.hypot(cx - ph[0], cy - ph[1])
                along = max(from_hub / max(math.hypot(pf[0] - ph[0], pf[1] - ph[1]), 1e-9), 0.2)
                if sine <= _STUCK or need > turn_need:
                    # (almost) the same direction: growing never parts them (or only by growing the
                    # whole picture many times over), so the end with fewer lines turns away round
                    # the shared name (never the pinned focus: its moves are dropped)
                    mover = far if lines_of.get(far, 0) <= lines_of.get(far_other, 0) else far_other
                    if mover == pinned:
                        mover = far_other if mover == far else far
                    pm = positions[mover]
                    side = ex * fy - ey * fx
                    turn = _STUCK_TURN if (side >= 0.0) == (mover == far_other) else -_STUCK_TURN
                    vx, vy = pm[0] - ph[0], pm[1] - ph[1]
                    cos_t, sin_t = math.cos(turn), math.sin(turn)
                    settle(need, (far, far_other, hub),
                           [(mover, ph[0] + vx * cos_t - vy * sin_t - pm[0], ph[1] + vx * sin_t + vy * cos_t - pm[1])],
                           True)
                    continue
                grow = short / max(sine, 0.25) / (2.0 * along)
                settle(need, (far, far_other, hub),
                       [(far, ux * short / (2.0 * along) + ex * grow, uy * short / (2.0 * along) + ey * grow),
                        (far_other, -ux * short / 2.0, -uy * short / 2.0)], False)
            else:
                # the label's line and the other line step apart (the label's line all the way when
                # the label sits on the other line: growing never parts them)
                stuck = distance <= _STUCK * (short + distance)
                if stuck:
                    suggested = [(node, ux * short, uy * short) for node in (a, b)]
                else:
                    suggested = ([(node, ux * short / 2.0, uy * short / 2.0) for node in (a, b)]
                                 + [(node, -ux * short / 2.0, -uy * short / 2.0) for node in (c, d)])
                settle(need, (a, b, c, d), suggested, stuck)
    return found, moves, placed


def _relocate(positions: Dict[str, List[float]], edges: Sequence[Tuple[Any, str, str]],
              node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
              name_lines: Sequence[int], spots: Mapping[Any, float], cell: float,
              report: Sequence[Tuple[str, ...]], budget: List[int], pinned: Optional[str] = None,
              lines_seen: Optional[Dict[int, Tuple[Any, ...]]] = None) -> bool:
    """For each problem in ``report``, try its names (fewest lines first) on spots around where they
    are and keep a spot where fewer things are in the way of that name, its box, its lines and their
    labels (the others stay where they are). ``budget[0]``: spots that may still be tried (it goes
    down; at 0 nothing is tried, so the time stays bounded). ``lines_seen``: what each line's grid
    cells were worked out from, kept between calls (most lines do not move between them; with
    1,000 names they cost more than everything else). True when a name moved."""
    if lines_seen is None:
        lines_seen = {}
    if budget[0] <= 0:
        return False
    clear = LINE_CLEARANCE
    half = _ROOM_GAP / 2.0
    wanted = set(name_lines)
    incident: Dict[str, List[int]] = {node: [] for node in positions}
    for index, (_key, a, b) in enumerate(edges):
        incident[a].append(index)
        incident[b].append(index)
    labelled = {index: label_size[key] for index, (key, _a, _b) in enumerate(edges) if key in label_size}

    def build(before=None, moved_node: Optional[str] = None):
        """The grids of what is placed; after ``moved_node`` moved, ``before`` updated for its lines."""
        if before is not None:
            line_grid, geos = before[0], before[1]
            todo = incident[moved_node]
        else:
            line_grid, geos = {}, [None] * len(edges)
            todo = range(len(edges))
        for index in todo:
            _key, a, b = edges[index]
            pa, pb = positions[a], positions[b]
            ends = (pa[0], pa[1], pb[0], pb[1], cell)
            seen = lines_seen.get(index)
            if before is not None and seen is not None:
                for cell_key in seen[2]:
                    line_grid[cell_key].remove(index)
            if seen is None or seen[0] != ends:
                seen = lines_seen[index] = (ends, _line_geo(pa, pb), _cells_along(pa, pb, cell))
            geos[index] = seen[1]
            for cell_key in seen[2]:
                line_grid.setdefault(cell_key, []).append(index)
        names = _BoxGrid(cell)
        name_of: List[str] = []
        for node, (w, h) in node_size.items():
            x, y = positions[node]
            names.add((x - w / 2.0 - half, y - h / 2.0 - half, x + w / 2.0 + half, y + h / 2.0 + half))
            name_of.append(node)
        labels = _BoxGrid(cell)
        label_of: List[int] = []
        for index, (w, h) in labelled.items():
            key, a, b = edges[index]
            t = spots.get(key)
            if t is None:
                continue
            pa, pb = positions[a], positions[b]
            x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
            labels.add((x - w / 2.0 - half, y - h / 2.0 - half, x + w / 2.0 + half, y + h / 2.0 + half))
            label_of.append(index)
        return line_grid, geos, names, name_of, labels, label_of

    def cost(node: str, x: float, y: float, state, limit: int = 1 << 30) -> int:
        """What is in the way of ``node`` at ``(x, y)``; counting stops at ``limit``."""
        line_grid, geos, names, name_of, labels, label_of = state
        own = set(incident[node])
        w, h = node_size[node]
        box = (x - w / 2.0 - half, y - h / 2.0 - half, x + w / 2.0 + half, y + h / 2.0 + half)
        total = sum(1 for index, _area in names.hits(box) if name_of[index] != node)
        total += sum(1 for index, _area in labels.hits(box) if label_of[index] not in own)
        if total >= limit:
            return total
        reach = max(w, h) / 2.0 + clear
        near = set()
        for gx in range(int(math.floor((x - reach) / cell)), int(math.floor((x + reach) / cell)) + 1):
            for gy in range(int(math.floor((y - reach) / cell)), int(math.floor((y + reach) / cell)) + 1):
                near.update(line_grid.get((gx, gy), ()))
        for index in near:
            if index not in own and index in wanted and _near_line(x, y, geos[index], w / 2.0, h / 2.0):
                total += 1
        mine = []
        for index in own:
            _key, a, b = edges[index]
            other = b if a == node else a
            q = positions[other]
            mine.append((math.hypot(q[0] - x, q[1] - y), index, other, q))
        mine.sort(key=lambda item: (item[0], item[1]))
        new_boxes: List[Box] = [box]
        new_geos = {index: _line_geo((x, y), q) for _length, index, _other, q in mine}
        for length, index, other, q in mine:
            if total >= limit:
                return total
            geo = new_geos[index]
            cells = set()
            for gx, gy in _cells_along((x, y), q, cell):
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        cells.add((gx + ox, gy + oy))
            near_names, near_labels, near_lines = set(), set(), set()
            for cell_key in cells:
                near_names.update(names.cells.get(cell_key, ()))
                near_labels.update(labels.cells.get(cell_key, ()))
                near_lines.update(line_grid.get(cell_key, ()))
            if index in wanted:
                for found in near_names:
                    third = name_of[found]
                    if third != node and third != other:
                        tw, th = node_size[third]
                        tx, ty = positions[third]
                        if _near_line(tx, ty, geo, tw / 2.0, th / 2.0):
                            total += 1
            for found in near_labels:
                if label_of[found] in own:
                    continue
                lb = labels.boxes[found]
                if _near_line((lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0, geo,
                              (lb[2] - lb[0]) / 2.0 - half, (lb[3] - lb[1]) / 2.0 - half):
                    total += 1
            size = labelled.get(index)
            if size is None:
                continue
            lw, lh = size
            hw, hh = lw / 2.0 + half, lh / 2.0 + half
            dx, dy = q[0] - x, q[1] - y
            intervals = []
            for other_line in near_lines:
                if other_line not in own:
                    span = _line_interval(x, y, dx, dy, geos[other_line], lw / 2.0, lh / 2.0)
                    if span is not None:
                        intervals.append((span[0], span[1], 0))
            for other_line in own:
                if other_line != index:
                    span = _line_interval(x, y, dx, dy, new_geos[other_line], lw / 2.0, lh / 2.0)
                    if span is not None:
                        intervals.append((span[0], span[1], 0))
            boxes = [names.boxes[i] for i in near_names if name_of[i] != node]
            boxes += [labels.boxes[i] for i in near_labels if label_of[i] not in own]
            for other_box in boxes + new_boxes:
                span = _box_interval(x, y, dx, dy, other_box, hw, hh)
                if span is not None:
                    intervals.append((span[0], span[1], 0))
            t, in_way = _choose_spot(intervals, 1.0 / max(length, 1.0), None)
            total += len(in_way)
            cx, cy = x + dx * t, y + dy * t
            new_boxes.append((cx - hw, cy - hh, cx + hw, cy + hh))
        return total

    moved = False
    state = build()
    tried = {pinned} if pinned is not None else set()
    low_x = min(p[0] for p in positions.values())
    high_x = max(p[0] for p in positions.values())
    low_y = min(p[1] for p in positions.values())
    high_y = max(p[1] for p in positions.values())
    ring = [(math.cos(2.0 * math.pi * step / 8), math.sin(2.0 * math.pi * step / 8)) for step in range(8)]
    turns = [(math.cos(angle), math.sin(angle)) for degrees in (8, -8, 20, -20)
             for angle in (math.radians(degrees),)]
    for _need, problem, suggested in report:
        for node in sorted(set(problem), key=lambda item: (len(incident[item]), problem.index(item))):
            if node in tried or len(incident[node]) > _RELOCATE_MAX_LINES or len(tried) >= _RELOCATE_NAMES:
                continue
            tried.add(node)
            x, y = positions[node]
            current = cost(node, x, y, state)
            if current <= 0:
                continue
            # the way the problem pushes it (once, and further), nearby spots, and (for a name with
            # few lines) its line to each other end turned a little or made longer: a crowded name's
            # lines spread and lengthen
            spots_to_try = [[(x + dx, y + dy) for name, dx, dy in suggested if name == node and (dx or dy)]]
            # turned around the other names of the problem (two lines from one name in almost the
            # same direction come apart), or further from them
            for other in dict.fromkeys(problem):
                if other != node:
                    ox, oy = positions[other]
                    vx, vy = x - ox, y - oy
                    spots_to_try.append([(ox + vx * c - vy * s, oy + vx * s + vy * c) for c, s in turns]
                                        + [(ox + vx * 1.35, oy + vy * 1.35)])
            spots_to_try += [[(x + ux * radius * cell, y + uy * radius * cell) for ux, uy in ring]
                             for radius in (1.0, 2.5)]
            best = None
            for group in spots_to_try:
                for spot in group:
                    if not (low_x <= spot[0] <= high_x and low_y <= spot[1] <= high_y):
                        continue  # inside the picture as it is: a move never stretches it
                    if budget[0] <= 0:
                        break
                    budget[0] -= 1
                    value = cost(node, spot[0], spot[1], state, current if best is None else best[0])
                    if value < current and (best is None or value < best[0]):
                        best = (value, spot)
                if best is not None and best[0] == 0:
                    break
            if best is not None:
                positions[node][0], positions[node][1] = best[1]
                moved = True
                state = build(state, node)
                break
    return moved


def _room_cell(node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]]) -> float:
    """Grid cells at least half the widest box plus the clearance: whatever a label on a line can
    touch is filed in a cell next to one the line passes through."""
    sizes = list(node_size.values()) + list(label_size.values())
    biggest = max(max(w, h) for w, h in sizes) if sizes else 100.0
    return biggest / 2.0 + LINE_CLEARANCE + _ROOM_GAP


def _make_room(positions: Dict[str, List[float]], edges: Sequence[Tuple[Any, str, str]],
               node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
               name_lines: Optional[Sequence[int]] = None, pinned: Optional[str] = None,
               spots: Optional[Mapping[Any, float]] = None
               ) -> Tuple[Dict[str, List[float]], Dict[Any, float], float]:
    """Move names until no name box or label covers another, no label is touched by a line that
    is not its own and no line of ``name_lines`` (default: all) runs through a name it does not end at.
    ``pinned`` (the focus) is never moved by itself and ends where it started.

    Cheapest first: a label takes another free spot on its line (``_room_round``); a name in the
    way of a problem that would need the picture to grow a lot moves to a nearby spot where less
    is in the way (``_relocate``: turned round a related name, off a line, further out); room is
    made round one problem; only then the whole picture grows: names spread, boxes keep their size,
    so every label gets further from the lines that are not its own (``_grow_and_shrink``, which
    then shrinks it back as far as it stays clear). ``spots``: where labels sit already (kept while
    they stay free). Returns the positions, the spot (``t``) of every label and how many times over
    growing the whole picture made it.
    """
    if name_lines is None:
        name_lines = range(len(edges))
    name_lines = list(name_lines)
    cell = _room_cell(node_size, label_size)
    start ={key: list(point) for key, point in positions.items()}
    lines_seen: Dict[int, Tuple[Any, ...]] = {}  # each line's grid cells (see ``_relocate``)

    def look(where: Dict[str, List[float]], spots: Dict[Any, float],
             report: Optional[List[Tuple[float, Tuple[str, ...], Any]]] = None, turn: bool = False):
        found, moves, spots = _room_round(where, edges, node_size, label_size, name_lines, spots, cell, report,
                                          pinned, _TURN_NEED if turn else _BIG, lines_seen)
        if pinned is not None:
            moves.pop(pinned, None)
            if report:
                report[:] = [(need, names, [item for item in suggested if item[0] != pinned])
                             for need, names, suggested in report]
        return found, moves, spots

    def relocator() -> Callable[..., bool]:
        budget = [_RELOCATE_BUDGET]  # spots tried, shared by every relocation of one attempt

        def relocate(where: Dict[str, List[float]], spots: Dict[Any, float], report: List[Tuple[str, ...]],
                     early: bool = False) -> bool:
            if early and budget[0] < _RELOCATE_BUDGET * 2 // 3:
                return False  # the rest is kept for the last few problems, where it helps most
            return _relocate(where, edges, node_size, label_size, name_lines, spots, cell, report, budget, pinned,
                             lines_seen)
        return relocate

    found, positions, spots, growth = _grow_and_shrink(positions, dict(spots or {}), start, look, relocator())
    if pinned is not None and pinned in positions:
        # growing moved everything round the middle of the picture: the focus goes back where it was
        dx, dy = start[pinned][0] - positions[pinned][0], start[pinned][1] - positions[pinned][1]
        for point in positions.values():
            point[0] += dx
            point[1] += dy
    return positions, spots, growth


def _span(positions: Mapping[str, Sequence[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return max(xs) - min(xs), max(ys) - min(ys)


def _area(positions: Mapping[str, Sequence[float]]) -> float:
    width, height = _span(positions)
    return width * height


def _grow_locally(positions: Dict[str, List[float]], spots: Dict[Any, float], found: int,
                  report: Sequence[Tuple[float, Tuple[str, ...], Any]], look: Callable[..., Any]
                  ) -> Optional[Tuple[Dict[str, List[float]], Dict[Any, float]]]:
    """Room around one problem instead of the whole picture: inside a circle round its names
    everything grows ``need`` times, outside it everything moves out by as much as the circle grew
    (no two names get nearer). The first of the worst problems for which that leaves fewer problems
    wins; None when none does."""
    tried = 0
    for need, names, _suggested in sorted(report, key=lambda item: -item[0]):
        if tried >= _LOCAL_TRIES:
            break
        points = [positions[name] for name in dict.fromkeys(names) if name in positions]
        if not points:
            continue
        tried += 1
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        # local: at most a fifth of the picture, so the picture grows far less than by growing all of it
        radius = min(max(math.hypot(p[0] - cx, p[1] - cy) for p in points) * 1.25, 0.2 * max(_span(positions))) + 1.0
        factor = min(max(need, _ROOM_GROW), _ROOM_GROW_MAX)
        trial = {}
        for key, (x, y) in positions.items():
            r = math.hypot(x - cx, y - cy)
            grown = r * factor if r <= radius else r + (factor - 1.0) * radius
            scale = grown / r if r > 1e-9 else 1.0
            trial[key] = [cx + (x - cx) * scale, cy + (y - cy) * scale]
        trial_found, _moves, trial_spots = look(trial, spots)
        if trial_found < found:
            return trial, trial_spots
    return None


def _nudge(positions: Dict[str, List[float]], spots: Dict[Any, float], found: int,
           report: Sequence[Tuple[float, Tuple[str, ...], Any]], look: Callable[..., Any]
           ) -> Optional[Tuple[Dict[str, List[float]], Dict[Any, float]]]:
    """The moves the problems ask for (a name off a line, two lines apart), all at once, once and
    twice as far; kept when fewer problems are left, else None."""
    for factor in (1.0, 2.0):
        trial = {key: list(point) for key, point in positions.items()}
        for _need, _names, suggested in report:
            for node, dx, dy in suggested:
                trial[node][0] += dx * factor
                trial[node][1] += dy * factor
        trial_found, _moves, trial_spots = look(trial, spots)
        if trial_found < found:
            return trial, trial_spots
    return None


def _grow_and_shrink(positions: Dict[str, List[float]], spots: Dict[Any, float],
                     start: Mapping[str, Sequence[float]], look: Callable[..., Any],
                     relocate: Callable[..., bool]
                     ) -> Tuple[int, Dict[str, List[float]], Dict[Any, float], float]:
    """Until nothing is in the way: move single names out of the way (``relocate``), make room round
    a problem (``_grow_locally``) or do the moves the problems ask for (``_nudge``) while that
    leaves fewer problems, else grow the picture a step (with the moves for what growing cannot
    fix, from ``_room_round``). Then shrink it back as far as it stays clear, but not
    below the size it started at. Returns ``(problems left, positions, spots, growth)``: growth is
    how many times over growing the whole picture made it."""
    growth = 1.0
    grown = 0
    tries = 0
    before = None  # (problems, positions, moves, spots, report) before names were moved one by one
    for _round in range(3 * _GROW_ROUNDS):
        report: List[Tuple[float, Tuple[str, ...]]] = []
        found, moves, spots = look(positions, spots, report)
        if not found or grown >= _GROW_ROUNDS:
            break
        if before is not None:
            if found < before[0]:
                tries = 0
            else:  # moving them did not help: back to before, and grow
                found, positions, moves, spots, report = before
                tries = _RELOCATE_TRIES
            before = None
        # problems that growing would take long to clear (or a few left): move one of their names to
        # a nearby spot where less is in the way, worst first (cheaper than growing the whole picture
        # for them); kept only when fewer problems are left
        needs = sorted(item[0] for item in report)
        outlier = max(_HARD_NEED, _HARD_NEED * needs[len(needs) // 2])
        hard = sorted((item for item in report if item[0] > outlier or found <= _RELOCATE_MAX),
                      key=lambda item: -item[0])
        if tries < _RELOCATE_TRIES and hard:
            tries += 1
            before = (found, {key: list(point) for key, point in positions.items()}, moves, spots, report)
            if relocate(positions, spots, hard, found > _RELOCATE_MAX):
                continue
            before = None
        # then more room where the problems are: the names around each grow apart, everything
        # further out moves out by as much (kept only when fewer problems are left)
        if found <= _RELOCATE_MAX:
            local = _grow_locally(positions, spots, found, report, look)
            if local is None:
                local = _nudge(positions, spots, found, report, look)
            if local is not None:
                positions, spots = local
                tries = 0
                continue
            # the few problems left would grow the whole picture: two lines from one name that only
            # many times the size would part turn apart as well (else one such pair next to two names
            # nothing can move grew the picture a hundred times over)
            report = []
            found, moves, spots = look(positions, spots, report, turn=True)
        for node, (dx, dy) in moves.items():
            positions[node][0] += dx
            positions[node][1] += dy
        # as much as clears about half of the problems (the shrink afterwards takes back too much)
        needs = sorted(item[0] for item in report)
        factor = min(max(needs[len(needs) // 2], _ROOM_GROW), _ROOM_GROW_MAX)
        _scale_up(positions, factor)
        growth *= factor
        grown += 1
        tries = 0
    for _round in range(_GROW_ROUNDS if found else 0):
        # the guarantee: growing alone (with the moves for what growing cannot fix)
        for node, (dx, dy) in moves.items():
            positions[node][0] += dx
            positions[node][1] += dy
        _scale_up(positions, _ROOM_GROW_MAX)
        growth *= _ROOM_GROW_MAX
        found, moves, spots = look(positions, spots, turn=True)
        if not found:
            break
    if found:
        return found, positions, spots, growth
    size, smallest = max(_span(positions)), max(_span(start))
    if size <= smallest * 1.001:
        return found, positions, spots, growth
    low, high = smallest / size, 1.0
    best = (positions, spots, 1.0)
    for _step in range(_SHRINK_STEPS):
        factor = (low + high) / 2.0
        trial = {key: list(point) for key, point in positions.items()}
        _scale_up(trial, factor)
        problems, _moves, trial_spots = look(trial, spots)
        if problems:
            low = factor
        else:
            high = factor
            best = (trial, trial_spots, factor)
    return 0, best[0], best[1], growth * best[2]


class _Placed:
    """What is placed so far while names are placed one at a time (see ``_place_one_by_one``): name
    boxes and labels in grids, lines and planned lines ("ghosts": from a placed name to where a
    related name not yet placed is expected) filed by the grid cells they pass through."""

    def __init__(self, cell: float, node_size: Mapping[str, Tuple[float, float]],
                 edges: Sequence[Tuple[Any, str, str]], label_size: Mapping[Any, Tuple[float, float]],
                 name_lines: Sequence[int]) -> None:
        self.cell = cell
        self.node_size = node_size
        self.edges = edges
        self.label_size = label_size
        self.name_lines = set(name_lines)
        self.positions: Dict[str, List[float]] = {}
        self.name_box: Dict[str, Box] = {}
        self.name_cells: Dict[Tuple[int, int], set] = {}
        self.label_box: Dict[int, Box] = {}
        self.label_cells: Dict[Tuple[int, int], set] = {}
        self.line_cells: Dict[Tuple[int, int], set] = {}
        self.cells_of_line: Dict[int, List[Tuple[int, int]]] = {}
        self.geos: Dict[int, LineGeo] = {}
        self.ghost_cells: Dict[Tuple[int, int], List[int]] = {}
        self.ghosts: Dict[int, Tuple[LineGeo, str, str]] = {}
        self.ghosts_to: Dict[str, List[int]] = {}
        self.ghost_count = 0
        self.last_failed = -1
        self.checks = getattr(self, "checks", 0)  # spots checked so far (kept by ``spread``)
        self.spots: Dict[Any, float] = {}
        self.label_size_of = {index: label_size[key] for index, (key, _a, _b) in enumerate(edges) if key in label_size}
        self.incident: Dict[str, List[Tuple[int, str]]] = {}
        for index, (_key, a, b) in enumerate(edges):
            self.incident.setdefault(a, []).append((index, b))
            self.incident.setdefault(b, []).append((index, a))

    def _around(self, cells: Iterable[Tuple[int, int]]) -> set:
        """``cells`` and the ring of cells round them."""
        near = set()
        for gx, gy in cells:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    near.add((gx + ox, gy + oy))
        return near

    def _cells_of(self, box: Box, ring: int = 0) -> List[Tuple[int, int]]:
        """The cells ``box`` covers, with ``ring`` more all round."""
        c = self.cell
        return [(gx, gy) for gx in range(int(box[0] // c) - ring, int(box[2] // c) + 1 + ring)
                for gy in range(int(box[1] // c) - ring, int(box[3] // c) + 1 + ring)]

    # Everything is filed in the cells it covers and the ring round them, so whatever comes near a
    # line or a box is found in the cells that line or box itself covers.

    def _add_label(self, index: int, box: Box) -> None:
        self.label_box[index] = box
        for cell_key in self._cells_of(box, 1):
            self.label_cells.setdefault(cell_key, set()).add(index)

    def _drop_label(self, index: int) -> None:
        box = self.label_box.pop(index)
        for cell_key in self._cells_of(box, 1):
            self.label_cells[cell_key].discard(index)

    def _names_on(self, box: Box) -> List[str]:
        """The names whose boxes overlap ``box`` (touching edges do not count)."""
        found = set()
        for cell_key in self._cells_of(box):
            found.update(self.name_cells.get(cell_key, ()))
        return [node for node in found
                if min(box[2], self.name_box[node][2]) > max(box[0], self.name_box[node][0])
                and min(box[3], self.name_box[node][3]) > max(box[1], self.name_box[node][1])]

    def remove(self, node: str) -> Tuple[float, float, List[Tuple[int, Box, float]]]:
        """Take ``node``, its lines and their labels out again; returns what ``place`` needs to put
        them back as they were."""
        x, y = self.positions.pop(node)
        box = self.name_box.pop(node)
        for cell_key in self._cells_of(box, 1):
            self.name_cells[cell_key].discard(node)
        labels = []
        for index, other in self.incident.get(node, ()):
            geo = self.geos.pop(index, None)
            if geo is None:
                continue
            for cell_key in self.cells_of_line.pop(index):
                self.line_cells[cell_key].discard(index)
            if index in self.label_box:
                labels.append((index, self.label_box[index], self.spots[self.edges[index][0]]))
                self._drop_label(index)
                del self.spots[self.edges[index][0]]
        return x, y, labels

    def _near(self, cells: Iterable[Tuple[int, int]], ghosts: bool) -> Tuple[set, set, set, set]:
        """``(names, labels, lines, planned lines)`` filed in ``cells`` (planned ones with ``ghosts``)."""
        near_names, near_labels, near_lines, near_ghosts = set(), set(), set(), set()
        name_cells, label_cells, line_cells = self.name_cells, self.label_cells, self.line_cells
        ghost_cells = self.ghost_cells
        for cell_key in cells:
            if cell_key in name_cells:
                near_names.update(name_cells[cell_key])
            if cell_key in label_cells:
                near_labels.update(label_cells[cell_key])
            if cell_key in line_cells:
                near_lines.update(line_cells[cell_key])
            if ghosts and cell_key in ghost_cells:
                near_ghosts.update(ghost_cells[cell_key])
        return near_names, near_labels, near_lines, near_ghosts

    def _free_spot(self, p: Sequence[float], q: Sequence[float], size: Tuple[float, float],
                   near: Tuple[set, set, set, set], node: str, skip_line: Optional[int],
                   more_lines: Iterable[LineGeo], more_boxes: Sequence[Box], skip_labels: Container[int],
                   from_p: bool) -> Optional[Tuple[float, Box]]:
        """``(t, box)``: the spot on ``p``-``q`` (fraction from ``p``) for a label of ``size`` clear of
        the names, labels, lines and planned lines of ``near`` (see ``_near``; planned lines to
        ``node``, ``skip_line`` and labels of ``skip_labels`` left out) and of ``more_lines`` /
        ``more_boxes``, nearest ``p`` (``from_p``) or ``q``; None when there is none."""
        free = self._stretches(p, q, size, near, node, skip_line, more_lines, more_boxes, skip_labels)
        if not free:
            return None
        half = _ROOM_GAP / 2.0
        hw, hh = size[0] / 2.0 + half, size[1] / 2.0 + half
        # half a pixel inside the free stretch: the room pass, measuring again, finds it free too
        inside = 0.5 / max(math.hypot(q[0] - p[0], q[1] - p[1]), 1.0)
        lo, hi = free[0] if from_p else free[-1]
        t = min(lo + inside, (lo + hi) / 2.0) if from_p else max(hi - inside, (lo + hi) / 2.0)
        cx, cy = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        return t, (cx - hw, cy - hh, cx + hw, cy + hh)

    def ray_room(self, p: Sequence[float], q: Sequence[float], size: Tuple[float, float], node: str,
                 ghosts: bool) -> List[Tuple[float, float]]:
        """Where along ``p``-``q`` (as distances from ``p``) a label of ``size`` is clear of what is
        placed (and, with ``ghosts``, of the planned lines not to ``node``)."""
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        free = self._stretches(p, q, size, self._near(_cells_along(p, q, self.cell), ghosts), node, None,
                               (), (), ())
        return [(lo * length, hi * length) for lo, hi in free]

    def _stretches(self, p: Sequence[float], q: Sequence[float], size: Tuple[float, float],
                   near: Tuple[set, set, set, set], node: str, skip_line: Optional[int],
                   more_lines: Iterable[LineGeo], more_boxes: Sequence[Box], skip_labels: Container[int]
                   ) -> List[Tuple[float, float]]:
        """The free stretches of ``p``-``q`` for a label of ``size`` (see ``_free_spot``)."""
        half = _ROOM_GAP / 2.0
        lw, lh = size
        hw, hh = lw / 2.0 + half, lh / 2.0 + half
        px, py = p[0], p[1]
        dx, dy = q[0] - px, q[1] - py
        near_names, near_labels, near_lines, near_ghosts = near
        intervals = []
        geos = [self.geos[index] for index in near_lines if index != skip_line]
        geos += list(more_lines)
        # the stretch a label on this line can cover, grown by the clearance: what is outside it is
        # skipped without a closer look
        reach_x, reach_y = hw + LINE_CLEARANCE, hh + LINE_CLEARANCE
        low_x, high_x = min(px, q[0]) - reach_x, max(px, q[0]) + reach_x
        low_y, high_y = min(py, q[1]) - reach_y, max(py, q[1]) + reach_y

        big_w, big_h = lw / 2.0 + LINE_CLEARANCE, lh / 2.0 + LINE_CLEARANCE

        def add_lines(lines: Iterable[LineGeo]) -> None:
            # ``_line_interval`` written out here (the hottest loop of placing names one at a time)
            for geo in lines:
                x0, x1, y0, y1, nx, ny, across = geo
                if x1 < low_x or x0 > high_x or y1 < low_y or y0 > high_y:
                    continue
                reach = (nx if nx > 0.0 else -nx) * big_w + (ny if ny > 0.0 else -ny) * big_h
                side_start = nx * px + ny * py - across
                side_end = side_start + nx * dx + ny * dy
                if (side_start >= reach and side_end >= reach) or (side_start <= -reach and side_end <= -reach):
                    continue
                # ``_hit_interval`` written out too: the line's box grown, then the band along it
                lo, hi = 0.0, 1.0
                for k, r in ((dx, x1 + big_w - px), (-dx, big_w - x0 + px), (dy, y1 + big_h - py),
                             (-dy, big_h - y0 + py), (side_end - side_start, reach - side_start),
                             (side_start - side_end, reach + side_start)):
                    if -1e-12 < k < 1e-12:
                        if r < 0.0:
                            break
                        continue
                    v = r / k
                    if k > 0.0:
                        if v < hi:
                            hi = v
                    elif v > lo:
                        lo = v
                    if lo >= hi:
                        break
                else:
                    intervals.append((lo, hi))
        add_lines(geos)
        boxes = [self.name_box[found] for found in near_names]
        boxes += [self.label_box[index] for index in near_labels if index not in skip_labels]
        for other_box in boxes + list(more_boxes):
            if other_box[2] < low_x or other_box[0] > high_x or other_box[3] < low_y or other_box[1] > high_y:
                continue
            span = _box_interval(px, py, dx, dy, other_box, hw, hh)
            if span is not None:
                intervals.append(span)
        margin = 1.0 / max(math.hypot(dx, dy), 1.0)
        free = _free_stretches(intervals, margin)
        if free and near_ghosts:
            # the planned lines only when the rest leaves room (most spots fail before)
            add_lines(item[0] for item in (self.ghosts.get(ghost) for ghost in near_ghosts)
                      if item is not None and item[2] != node)
            free = _free_stretches(intervals, margin)
        return free

    def check(self, node: str, x: float, y: float, ghosts: bool = True, limit: int = 1
              ) -> Tuple[int, List[Tuple[int, Box, float]]]:
        """``(problems, labels)`` of ``node`` placed at ``(x, y)`` with its lines to the names already
        placed: its box on a name or within the clearance of a line, a new line within the clearance
        of a name, a new label with no spot on its line clear of every other box and line (and, with
        ``ghosts``, of the planned lines). A label already placed that the new box covers or a new
        line comes near slides along its own line to another free spot (none: a problem).
        ``labels``: ``(edge, box, t)`` of each new or moved label (``t`` from the edge's first name).
        Counting stops at ``limit``."""
        self.checks += 1
        half = _ROOM_GAP / 2.0
        w, h = self.node_size[node]
        box = (x - w / 2.0 - half, y - h / 2.0 - half, x + w / 2.0 + half, y + h / 2.0 + half)
        bad = len(self._names_on(box))
        if bad >= limit:
            return bad, []
        moving = set()  # labels in the way that must slide along their lines
        _names, near_labels, near_lines, near_ghosts = self._near(self._cells_of(box), ghosts)
        for index in near_labels:
            lb = self.label_box[index]
            if min(box[2], lb[2]) > max(box[0], lb[0]) and min(box[3], lb[3]) > max(box[1], lb[1]):
                moving.add(index)
        for index in near_lines:
            if index in self.name_lines and _near_line(x, y, self.geos[index], w / 2.0, h / 2.0):
                bad += 1
                if bad >= limit:
                    return bad, []
        for ghost in near_ghosts:
            item = self.ghosts.get(ghost)
            if item is not None and item[2] != node and _near_line(x, y, item[0], w / 2.0, h / 2.0):
                bad += 1
                if bad >= limit:
                    return bad, []
        positions = self.positions
        new = []
        for index, other in self.incident.get(node, ()):
            q = positions.get(other)
            if q is not None:
                new.append((math.hypot(q[0] - x, q[1] - y), index, other, q))
        new.sort(key=lambda item: (item[0], item[1]))
        new_geos = {index: _line_geo((x, y), q) for _length, index, _other, q in new}
        new_near = {}
        lines_of = self.incident
        for _length, index, other, q in new:
            geo = new_geos[index]
            near = new_near[index] = self._near(_cells_along((x, y), q, self.cell), ghosts)
            near_names, near_labels = near[0], near[1]
            if index in self.name_lines:
                for third in near_names:
                    if third != node and third != other:
                        tw, th = self.node_size[third]
                        tx, ty = positions[third]
                        if _near_line(tx, ty, geo, tw / 2.0, th / 2.0):
                            bad += 1
                            if bad >= limit:
                                return bad, []
            for found in near_labels:
                lb = self.label_box[found]
                if found not in moving and _near_line((lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0, geo,
                                                      (lb[2] - lb[0]) / 2.0 - half, (lb[3] - lb[1]) / 2.0 - half):
                    moving.add(found)
        if len(moving) > _PLACE_SLIDES:
            return bad + len(moving), []
        new_labels: List[Tuple[int, Box, float]] = []
        # the label that found no room at the last spot tried first: most spots fail on the same one
        new.sort(key=lambda item: item[1] != self.last_failed)
        for _length, index, other, q in new:
            size = self.label_size.get(self.edges[index][0])
            if size is None:
                continue
            spot = self._free_spot((x, y), q, size, new_near[index], node, None,
                                   [g for i, g in new_geos.items() if i != index],
                                   [box] + [item[1] for item in new_labels], moving,
                                   len(lines_of.get(node, ())) >= len(lines_of.get(other, ())))
            if spot is None:
                self.last_failed = index
                bad += 1
                if bad >= limit:
                    return bad, []
                continue
            t, label = spot
            # new labels never sit on the new lines: those are among the lines kept clear
            new_labels.append((index, label, t if self.edges[index][1] == node else 1.0 - t))
        for index in sorted(moving):
            _key, a, b = self.edges[index]
            pa, pb = positions[a], positions[b]
            spot = self._free_spot(pa, pb, self.label_size[self.edges[index][0]],
                                   self._near(_cells_along(pa, pb, self.cell), ghosts), node, index,
                                   list(new_geos.values()), [box] + [item[1] for item in new_labels], moving,
                                   len(lines_of.get(a, ())) >= len(lines_of.get(b, ())))
            if spot is None:
                bad += 1
                if bad >= limit:
                    return bad, []
                continue
            new_labels.append((index, spot[1], spot[0]))
        if bad:
            return bad, []
        # a label that slid must clear the new labels too, and the new lines the labels that slid
        for k, (index, label, _t) in enumerate(new_labels):
            for other_index, other_label, _u in new_labels[:k]:
                if (min(label[2], other_label[2]) > max(label[0], other_label[0])
                        and min(label[3], other_label[3]) > max(label[1], other_label[1])):
                    return 1, []
        return 0, new_labels

    def place(self, node: str, x: float, y: float, labels: Sequence[Tuple[int, Box, float]]) -> None:
        """Put ``node`` at ``(x, y)`` with its lines to the names placed and ``labels`` (from ``check``)."""
        half = _ROOM_GAP / 2.0
        w, h = self.node_size[node]
        self.positions[node] = [x, y]
        box = (x - w / 2.0 - half, y - h / 2.0 - half, x + w / 2.0 + half, y + h / 2.0 + half)
        self.name_box[node] = box
        for cell_key in self._cells_of(box, 1):
            self.name_cells.setdefault(cell_key, set()).add(node)
        for index, other in self.incident.get(node, ()):
            q = self.positions.get(other)
            if q is None or other == node or index in self.geos:
                continue
            self.geos[index] = _line_geo((x, y), q)
            cells = self._around(_cells_along((x, y), q, self.cell))
            self.cells_of_line[index] = cells
            for cell_key in cells:
                self.line_cells.setdefault(cell_key, set()).add(index)
        for index, box, t in labels:
            if index in self.label_box:
                self._drop_label(index)
            self._add_label(index, box)
            self.spots[self.edges[index][0]] = t
        for ghost in self.ghosts_to.pop(node, ()):
            self.ghosts.pop(ghost, None)

    def extent(self, aspect: float, around: Optional[str] = None) -> float:
        """``_drawn_extent`` of what is placed: the larger of its width and ``aspect`` times its height
        (``around``: of the area centred on that name, see ``_centred_size``)."""
        boxes = list(self.name_box.values()) + list(self.label_box.values())
        if not boxes:
            return 0.0
        width, height = _centred_size(boxes, self.positions.get(around) if around is not None else None)
        return max(width, aspect * height) - _ROOM_GAP

    def spread(self, factor: float, centre: Sequence[float], expected: Callable[[str], Sequence[float]]) -> None:
        """Everything placed ``factor`` times as far from ``centre`` (boxes keep their size, labels
        their spot on their lines), the planned lines drawn again to where ``expected`` now says."""
        cx, cy = centre[0], centre[1]
        positions = {node: [cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor]
                     for node, p in self.positions.items()}
        plans = [(node, target) for _geo, node, target in self.ghosts.values()]
        spots = dict(self.spots)
        order = list(self.name_box)
        self.__init__(self.cell, self.node_size, self.edges, self.label_size, self.name_lines)
        for node in order:
            self.place(node, positions[node][0], positions[node][1], ())
        half = _ROOM_GAP / 2.0
        for index, (key, a, b) in enumerate(self.edges):
            t = spots.get(key)
            if t is None or a not in positions or b not in positions:
                continue
            pa, pb = positions[a], positions[b]
            lw, lh = self.label_size_of[index]
            x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
            self._add_label(index, (x - lw / 2.0 - half, y - lh / 2.0 - half, x + lw / 2.0 + half, y + lh / 2.0 + half))
            self.spots[key] = t
        for node, target in plans:
            self.plan(node, target, expected(target))

    def plan(self, node: str, target: str, where: Sequence[float]) -> None:
        """A planned line from the placed ``node`` to where ``target`` is expected."""
        p = self.positions[node]
        if math.hypot(where[0] - p[0], where[1] - p[1]) < 1.0:
            return
        self.ghost_count += 1
        ghost = self.ghost_count
        self.ghosts[ghost] =(_line_geo(p, where), node, target)
        self.ghosts_to.setdefault(target, []).append(ghost)
        for cell_key in self._around(_cells_along(p, where, self.cell)):
            self.ghost_cells.setdefault(cell_key, []).append(ghost)


#: Candidate spots for a name tried round where the guide puts it: rings this far apart (scale 1) ...
_PLACE_STEP = 36.0
#: ... first evenly, then each this many times as far, out to ``_PLACE_REACH`` times the guide's size.
_PLACE_GROW = 1.12
_PLACE_REACH = 1.5
#: A name with placed relations is tried out from the busiest of them, turned this far (radians)
#: from where the guide expects it, at distances each this many times the last ...
_PLACE_TURNS = tuple(sorted({sign * 0.06 * k for k in range(11) for sign in (1, -1)}
                            | {sign * value for value in (0.8, 1.05, 1.4, 1.9, 2.4) for sign in (1, -1)} | {math.pi},
                            key=lambda value: (abs(value), value)))
#: (1.1: the owner's book 3,892 x 2,907 px in 0.74 s, against 5,532 x 2,607 in 0.97 s at 1.05;
#: also no larger round a main character with 40 to 45 relations, measured)
_PLACE_GROW_RAY = 1.1
#: The picture is made again with this spacing (the smaller kept) while placing names took at most
#: this many checks of a spot in all (neither spacing wins on every shape: 100 names with 110
#: random relations, 1,300 checks in 0.14 s, 2,461 x 1,959 px against 2,882 x 2,241; 12 names all
#: related to each other at phone sizes, 3,900 checks; the owner's book takes 8,000, in 0.75 s).
_PLACE_GROW_RAY_TOO = 1.05
_TRY_TOO_CHECKS = 4500
#: ... and (with more than one) at the nearest of the spots round the middle of them.
_PLACE_MIDDLE = 200
#: How much the length of a name's lines counts against a spot, next to its distance from the guide
#: spot (a little over 1: between the anchor and the guide spot, the nearer the anchor the better).
_PLACE_SHORTER = 1.05
#: Spots tried again without keeping the planned lines clear, when none kept them clear.
_PLACE_UNPLANNED = 400
#: Spots counted in full for a name no spot suits (the least bad wins; the room pass fixes the rest).
_PLACE_FALLBACK = 40
#: Labels already placed that may slide along their lines to make room for one name.
_PLACE_SLIDES = 8
#: Spots first tried at least this far apart (then the better ones near the one found).
_PLACE_COARSE = 90.0
#: A name with no room: what is placed spreads out this many times over, at most this often.
_PLACE_SPREAD = 1.2
_PLACE_REGROW = 6


def _ring_of(placed: "_Placed", centre: str, own: Iterable[str]) -> Optional[Tuple[float, float, List[Tuple[float, float]]]]:
    """The first ring round ``centre`` (see ``focus_ring``) from what is placed; None before it has any."""
    if centre not in placed.positions:
        return None
    return focus_ring(placed.positions, centre, [other for other in own if other in placed.positions])


def focus_ring(positions: Mapping[str, Sequence[float]], centre: str, own: Iterable[str]
               ) -> Optional[Tuple[float, float, List[Tuple[float, float]]]]:
    """``(x, y, [(angle, distance), ...])``: the ring through ``own`` (the focus's relations) round
    ``centre``, by angle; between two of them its distance from the centre goes linearly from one's
    to the other's (one alone: a circle). None without any. See ``_inside_ring``."""
    cx, cy = positions[centre][0], positions[centre][1]
    points = sorted((math.atan2(positions[other][1] - cy, positions[other][0] - cx) % (2.0 * math.pi),
                     math.hypot(positions[other][0] - cx, positions[other][1] - cy)) for other in own)
    return (cx, cy, points) if points else None


def _ring_holds(positions: Mapping[str, Sequence[float]], centre: Optional[str], own: Iterable[str]) -> bool:
    """No name but the focus's own relations (``own``) lies inside their ring round the focus
    ``centre`` (see ``focus_ring``); True without a focus."""
    own = set(own)
    if centre is None or centre not in positions or not own:
        return True
    ring = focus_ring(positions, centre, [other for other in own if other in positions])
    return ring is None or not any(_inside_ring(ring, point[0], point[1]) for key, point in positions.items()
                                   if key != centre and key not in own)


def _inside_ring(ring: Tuple[float, float, List[Tuple[float, float]]], x: float, y: float) -> bool:
    """``(x, y)`` is not further from the ring's centre than the ring is in that direction."""
    cx, cy, points = ring
    distance = math.hypot(x - cx, y - cy)
    if len(points) == 1:
        return distance <= points[0][1]
    theta = math.atan2(y - cy, x - cx) % (2.0 * math.pi)
    after = bisect.bisect_left(points, (theta, -1.0)) % len(points)
    (t0, r0), (t1, r1) = points[after - 1], points[after]
    span = (t1 - t0) % (2.0 * math.pi) or 2.0 * math.pi
    share = ((theta - t0) % (2.0 * math.pi)) / span
    return distance <= r0 + (r1 - r0) * share


def _place_one_by_one(guide: Mapping[str, Sequence[float]], edges: Sequence[Tuple[Any, str, str]],
                      node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                      name_lines: Sequence[int], cell: float, root: Optional[str] = None,
                      order: Optional[Sequence[str]] = None, aspect: float = 1.0,
                      give_up: Optional[float] = None, grow_ray: float = _PLACE_GROW_RAY,
                      centred: bool = False
                      ) -> Optional[Tuple[Dict[str, List[float]], Dict[Any, float], int, float, int]]:
    """Names placed one at a time near where ``guide`` puts them, each at the nearest spot where it
    and its lines to the names already placed clear everything placed: its box, the lines (no line
    within the clearance of a name it does not end at or of a label not its own), and a free spot on
    each new line for its label, nearest the end with more relations. What is placed never moves,
    so the picture grows only where a name finds no room. Lines from a placed name to a related
    name not placed yet are planned (towards that name's guide spot, moved as its placed relations
    moved) and kept clear while there is a choice. ``root`` (default: the name with most lines) is
    placed first, where the guide puts it. A name with no room anywhere near spreads out what is
    placed so far (everything placed further from the first name; early on, while little is placed,
    that costs little room) and tries again. Returns ``(positions, label spots, names placed with
    a problem left, how many times over spreading made what was placed, how many spots were
    checked)``; None as soon as what is placed reaches past ``give_up`` (see ``_drawn_extent``:
    another layout already did better). ``grow_ray``: see ``_PLACE_GROW_RAY``. ``centred``: the
    picture is built round ``root`` (the focus): it is shown centred on it (see ``_pull_in``), and
    no other name goes nearer it than the furthest of its own relations placed (place those first)."""
    placed = _Placed(cell, node_size, edges, label_size, name_lines)
    lines_of = {node: len(placed.incident.get(node, ())) for node in guide}
    left = [node for node in guide]
    if root is None or root not in guide:
        root = max(left, key=lambda node: (lines_of[node], node))
    own = {other for _index, other in placed.incident.get(root, ())} if centred else set()
    xs = [p[0] for p in guide.values()]
    ys = [p[1] for p in guide.values()]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    rings = []
    radius = 0.0
    while radius <= _PLACE_REACH * size:
        rings.append(radius)
        radius = radius + _PLACE_STEP if radius < 10 * _PLACE_STEP else radius * _PLACE_GROW
    pattern = [(0.0, 0.0, 0.0)]
    for radius in rings[1:]:
        count = min(32, max(8, int(2.0 * math.pi * radius / _PLACE_STEP)))
        offset = (math.pi / count) * (len(pattern) % 2)
        pattern += [(radius * math.cos(offset + 2.0 * math.pi * k / count),
                     radius * math.sin(offset + 2.0 * math.pi * k / count), radius) for k in range(count)]
    trouble = 0
    spread = [1.0]
    count_placed = {node: 0 for node in guide}

    def expected(node: str) -> Tuple[float, float]:
        """The guide spot of ``node``, moved as its placed relations (else the nearest placed names) moved."""
        gx, gy = guide[node]
        moved = [other for _index, other in placed.incident.get(node, ()) if other in placed.positions]
        if not moved:
            near = sorted(placed.positions, key=lambda other: (math.hypot(guide[other][0] - gx, guide[other][1] - gy),
                                                               other))[:3]
            moved = near
        if not moved:
            return gx, gy
        dx = sum(placed.positions[o][0] - guide[o][0] for o in moved) / len(moved)
        dy = sum(placed.positions[o][1] - guide[o][1] for o in moved) / len(moved)
        return gx + dx, gy + dy

    def candidates_for(node: str, target: Optional[Tuple[float, float]] = None):
        """``(candidates, rays, anchor, anchor line)``: spots to try, ``(rank, x, y, ray)`` best
        first, ``ray`` naming the direction from the anchor (the busiest placed relation) it lies on.
        ``target``: where to look first instead of where the guide expects ``node``."""
        if target is not None:
            ex, ey = target
        else:
            ex, ey = (guide[node][0], guide[node][1]) if node == root else expected(node)
        candidates = [(r, ex + px, ey + py, None) for px, py, r in pattern]
        anchors = [placed.positions[other] for _index, other in placed.incident.get(node, ())
                   if other in placed.positions]
        rays: Dict[int, Tuple[float, float, float, float]] = {}  # ray: (ux, uy, need behind the label, length)
        anchor = anchor_line = None
        if node != root and anchors:
            # out from the busiest placed relation, turned a little or more: the name moves out along
            # its line, the label gets room nearer that relation
            anchor = max((other for _index, other in placed.incident.get(node, ()) if other in placed.positions),
                         key=lambda other: (lines_of[other], other))
            anchor_line = next(index for index, other in placed.incident[node] if other == anchor)
            ax, ay = placed.positions[anchor]
            vx, vy = ex - ax, ey - ay
            base = math.hypot(vx, vy)
            heading = math.atan2(vy, vx) if base > 1.0 else 0.0
            far = max(2.5 * base, base + 1500.0)
            aw, ah = node_size[anchor]
            nw, nh = node_size[node]
            lw, lh = label_size.get(edges[anchor_line][0], (0.0, 0.0))
            candidates = [] if len(anchors) == 1 else candidates
            add = candidates.append
            hypot = math.hypot
            for turn in _PLACE_TURNS:
                theta = heading + turn
                ux, uy = math.cos(theta), math.sin(theta)
                ext_a = abs(ux) * aw / 2.0 + abs(uy) * ah / 2.0
                ext_n = abs(ux) * nw / 2.0 + abs(uy) * nh / 2.0
                along = abs(ux) * lw / 2.0 + abs(uy) * lh / 2.0 + _ROOM_GAP
                ray = len(rays)
                rays[ray] = (ux, uy, ext_n + along + _ROOM_GAP, far)
                distance = ext_a + ext_n + 2.0 * LINE_CLEARANCE + (2.0 * along if lw else 0.0)
                penalty = abs(turn) * base * 0.5
                while distance <= far:
                    x, y = ax + ux * distance, ay + uy * distance
                    add((hypot(x - ex, y - ey) + penalty, x, y, ray))
                    distance *= grow_ray
            if len(anchors) > 1:
                # round the middle of its placed relations too: its lines short
                bx = sum(p[0] for p in anchors) / len(anchors)
                by = sum(p[1] for p in anchors) / len(anchors)
                candidates += [(math.hypot(bx + px - ex, by + py - ey), bx + px, by + py, None)
                               for px, py, _r in pattern[:_PLACE_MIDDLE]]
            # nearest the guide spot first, then by the length of its lines; a name with relations
            # still to place keeps nearer its guide spot (the room round it is planned for them)
            weight = 1.0 + 0.5 * sum(1 for _index, other in placed.incident.get(node, ())
                                     if other not in placed.positions)
            count = len(anchors)
            hypot = math.hypot
            ends = [(p[0], p[1]) for p in anchors]
            candidates = [(rank * weight + _PLACE_SHORTER * sum([hypot(x - px, y - py) for px, py in ends]) / count,
                           x, y, ray) for rank, x, y, ray in candidates]
        candidates.sort(key=lambda item: item[0])
        return candidates, rays, anchor, anchor_line

    def search(node: str, candidates, rays, anchor, anchor_line, use_ghosts: bool, tries: int = 1 << 30):
        """The best of ``candidates`` where ``node`` fits: first among spots at least
        ``_PLACE_COARSE`` apart (at most ``tries`` of them), then among the better ones near the spot found."""
        room: Dict[int, List[Tuple[float, float]]] = {}
        # built round a focus: a name not related to it stays outside the ring of the focus's relations
        ring = _ring_of(placed, root, own) if own and node != root and node not in own else None

        def fits(x: float, y: float, ray: Optional[int]):
            if ring is not None and _inside_ring(ring, x, y):
                return None
            if ray is not None and anchor_line in placed.label_size_of:
                # a spot on a ray only when the anchor's label finds room on it before the name
                ax, ay = placed.positions[anchor]
                ux, uy, behind, length = rays[ray]
                free = room.get(ray)
                if free is None:
                    free = room[ray] = placed.ray_room((ax, ay), (ax + ux * length, ay + uy * length),
                                                       placed.label_size_of[anchor_line], node, use_ghosts)
                reach = math.hypot(x - ax, y - ay) - behind
                if not any(lo <= reach for lo, _hi in free):
                    return None
            bad, labels = placed.check(node, x, y, use_ghosts)
            return None if bad else labels

        kept: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        step = _PLACE_COARSE
        step2 = step * step
        empty = ()
        for rank, x, y, ray in candidates:
            gx, gy = int(x // step), int(y // step)
            near = False
            for cell_key in ((gx - 1, gy - 1), (gx - 1, gy), (gx - 1, gy + 1), (gx, gy - 1), (gx, gy),
                             (gx, gy + 1), (gx + 1, gy - 1), (gx + 1, gy), (gx + 1, gy + 1)):
                for kx, ky in kept.get(cell_key, empty):
                    if (x - kx) * (x - kx) + (y - ky) * (y - ky) < step2:
                        near = True
                        break
                if near:
                    break
            if near:
                continue
            kept.setdefault((gx, gy), []).append((x, y))
            tries -= 1
            if tries < 0:
                return None
            labels = fits(x, y, ray)
            if labels is None:
                continue
            for finer, fx, fy, fray in candidates:
                if finer >= rank:
                    break
                if math.hypot(fx - x, fy - y) <= 1.5 * step:
                    finer_labels = fits(fx, fy, fray)
                    if finer_labels is not None:
                        return fx, fy, finer_labels
            return x, y, labels
        return None

    order_of = {node: index for index, node in enumerate(sorted(guide))}
    current = root
    while current is not None:
        node = current
        left.remove(node)
        chosen = None
        for _attempt in range(_PLACE_REGROW + 1):
            candidates, rays, anchor, anchor_line = candidates_for(node)
            # the nearest spot keeping the planned lines clear, else the nearest at all
            for use_ghosts, high in ((True, len(candidates)), (False, _PLACE_UNPLANNED)):
                chosen = search(node, candidates[:high], rays, anchor, anchor_line, use_ghosts)
                if chosen is not None:
                    break
            if chosen is not None or len(placed.positions) < 2:
                break
            # no room anywhere near: what is placed so far spreads out round the first name (every
            # gap between a label and a line not its own grows) and the name tries again
            placed.spread(_PLACE_SPREAD, placed.positions[root], expected)
            spread[0] *= _PLACE_SPREAD
        if chosen is None:
            trouble += 1
            best = None
            for _rank, x, y, _ray in candidates[:_PLACE_FALLBACK]:
                bad, labels = placed.check(node, x, y, False, best[0] if best else 1 << 30)
                if best is None or bad < best[0]:
                    best = (bad, x, y, labels)
            chosen = best[1:]
        placed.place(node, *chosen)
        if give_up is not None and placed.extent(aspect, root if centred else None) > give_up:
            return None
        for _index, other in placed.incident.get(node, ()):
            count_placed[other] = count_placed.get(other, 0) + 1
            if other not in placed.positions:
                placed.plan(node, other, expected(other))
        current = None
        if order is not None:
            current = next((other for other in order if other in left), None)
        if left and current is None:
            current = max(left, key=lambda other: (count_placed.get(other, 0) > 0, lines_of[other],
                                                   count_placed.get(other, 0), -order_of[other]))
    if not trouble:
        _pull_in(placed, root, candidates_for, search, aspect, centred)
    # labels of lines whose spot was never checked (a name placed with problems): the room pass finds one
    return placed.positions, placed.spots, trouble, spread[0], placed.checks


#: Names at the edge of the picture tried nearer their relations, at most this many times in all,
#: when they reach this far (scale 1) past everything else there.
_PULL_TRIES = 40
_PULL_GAIN = 40.0
#: Spots tried for one such name (the nearest first; it stays where it is when none of them suits).
_PULL_SPOTS = 60


def _pull_in(placed: _Placed, root: str, candidates_for: Callable[..., Any], search: Callable[..., Any],
             aspect: float = 1.0, centred: bool = False) -> None:
    """The picture made smaller: the name (with the names related to it alone) whose box or label
    reaches furthest past the rest on one side moves to the best spot nearer its relations and the
    middle where it and its lines still clear everything, while that makes the picture smaller
    as it is shown in a window ``aspect`` times as wide as high (the sides that decide how far it
    must be zoomed out are pulled in; ``centred``: the window is centred on ``root``, so the side
    further from it decides). Labels that slid for a move slide back when it is undone."""
    lines_of = {node: len(placed.incident.get(node, ())) for node in placed.positions}

    def size() -> Tuple[float, float]:
        boxes = list(placed.name_box.values()) + list(placed.label_box.values())
        return _centred_size(boxes, placed.positions[root] if centred else None)

    def extent() -> float:
        """How far the picture must be zoomed out to fit the window (larger: further)."""
        width, height = size()
        return max(width, aspect * height)

    def group_of(node: str) -> List[str]:
        """``node`` and the names related to it alone (they move with it)."""
        return [node] + sorted(other for _index, other in placed.incident.get(node, ())
                               if lines_of.get(other) == 1 and other != root)

    def outermost(side: int) -> Optional[Tuple[str, float]]:
        """The name to move for ``side`` (0 left, 1 top, 2 right, 3 bottom) and how far what moves
        with it reaches past everything else there."""
        sign = 1.0 if side < 2 else -1.0
        items = [(sign * box[side], ("n", node)) for node, box in placed.name_box.items()]
        items += [(sign * box[side], ("l", index)) for index, box in placed.label_box.items()]
        items.sort()
        kind, what = items[0][1]
        if kind == "n":
            node = what
        else:  # a label: the end of its line with fewer relations moves
            _key, a, b = placed.edges[what]
            node = a if (lines_of[a], a) < (lines_of[b], b) else b
        if lines_of.get(node) == 1:
            node = next(other for _index, other in placed.incident[node])  # a name alone with its relation
        if node == root:
            return None
        moving = set(group_of(node))
        lines = {index for member in moving for index, _other in placed.incident.get(member, ())}
        rest = [value for value, (kind, what) in items
                if not ((kind == "n" and what in moving) or (kind == "l" and what in lines))]
        return (node, (rest[0] - items[0][0]) if rest else 0.0)

    def target_of(node: str, x: float, y: float) -> Tuple[float, float]:
        rx, ry = placed.positions[root]
        ends = [placed.positions[other] for _index, other in placed.incident.get(node, ()) if other in placed.positions]
        mx = sum(p[0] for p in ends) / len(ends)
        my = sum(p[1] for p in ends) / len(ends)
        # half way from where it is to (near) the middle of its relations, a step towards the root
        mx, my = mx + (rx - mx) * 0.3, my + (ry - my) * 0.3
        return x + (mx - x) * 0.5, y + (my - y) * 0.5

    tried = set()
    budget = _PULL_TRIES
    while budget > 0:
        todo = []
        width, height = size()
        for side in ((0, 2) if width >= aspect * height else (1, 3)):
            pick = outermost(side)
            if pick is not None and pick[0] not in tried and pick[1] > _PULL_GAIN and pick[0] not in todo:
                todo.append(pick[0])
        if not todo:
            break
        for node in todo:
            budget -= 1
            tried.add(node)
            before = extent()
            group = group_of(node)
            old = {member: placed.remove(member) for member in reversed(group)}
            done: List[Tuple[str, List[Tuple[int, Box, float]]]] = []  # (name, labels that slid for it)
            ox, oy = old[node][0], old[node][1]
            for member in group:
                if not any(other in placed.positions for _index, other in placed.incident.get(member, ())):
                    break  # nothing placed to hang from (a group on its own): it stays
                mx, my = old[member][0], old[member][1]
                if member == node:
                    target = target_of(node, mx, my)
                else:  # where it was next to the name it hangs from
                    px, py = placed.positions[node]
                    target = (px + mx - ox, py + my - oy)
                candidates, rays, anchor, anchor_line = candidates_for(member, target)
                found = search(member, candidates, rays, anchor, anchor_line, False, _PULL_SPOTS)
                if found is None:
                    break
                own = {index for index, _other in placed.incident.get(member, ())}
                slid = [(index, placed.label_box[index], placed.spots[placed.edges[index][0]])
                        for index, _box, _t in found[2] if index not in own and index in placed.label_box]
                placed.place(member, *found)
                done.append((member, slid))
            if len(done) == len(group) and extent() < before * 0.999:
                tried.clear()
                tried.add(node)
                continue
            # undo: what moved goes back, the labels that slid for it slide back
            for member, slid in reversed(done):
                placed.remove(member)
                for index, box, t in slid:
                    placed._drop_label(index)
                    placed._add_label(index, box)
                    placed.spots[placed.edges[index][0]] = t
            for member in group:
                placed.place(member, *old[member])


class Layout(dict):
    """``{id: (x, y)}`` from ``layout_graph``; ``label_spots`` ``{(a, b): t}``: where along each
    line (the fraction of the way from ``a``) the layout made room for its label."""

    label_spots: Dict[Tuple[str, str], float] = {}
    #: More lines than ``ROOM_FOR_LABELS_MAX_EDGES``: room was made for the focus character's labels only.
    labels_capped: bool = False
    #: How many times over the last resort of the room pass, growing the whole picture, made it.
    growth: float = 1.0
    #: How many times over names placed early were spread out to make room for one placed later.
    spread: float = 1.0


def layout_spot(layout: Mapping[str, Any], a: Any, b: Any) -> Optional[float]:
    """Where the layout made room for the label of the line ``a``-``b`` (fraction from ``a``), else None."""
    spots = getattr(layout, "label_spots", None) or {}
    if (a, b) in spots:
        return spots[(a, b)]
    if (b, a) in spots:
        return 1.0 - spots[(b, a)]
    return None


def layout_graph(graph: Optional[Mapping[str, Any]], width: float = 1000.0, height: float = 700.0,
                 focus: Optional[str] = None, seed: int = 0,
                 node_size: Optional[Callable[[Mapping[str, Any]], Any]] = None,
                 label_size: Optional[Callable[[Mapping[str, Any]], Any]] = None) -> Dict[str, Tuple[float, float]]:
    """Deterministic positions inside ``width`` x ``height``: seeded force layout, or radial around ``focus``.

    With a ``focus`` the picture is built round it, whatever the shape of the book: it is the centre,
    its relations are the nearest names round it (no other name inside their ring, see
    ``focus_ring``), the others further out by their distance from it, the names with no path to it
    outside; a focus with no relations sits at the free spot nearest the middle of the picture the
    others make without a focus. The caller centres its view on the focus.

    ``node_size(node)`` and ``label_size(edge)`` give the drawn ``(width, height)`` of a name box
    and of a line's label (None: no label), in the units of ``width`` and ``height``. With them the
    names are placed one at a time where they, their lines and labels clear everything placed
    (``_place_one_by_one``, round a main character as a tree out from it, see ``_tree_root``), and
    a last pass checks that no box or label covers another and no line comes near a label or a
    name it does not end at, moving names (growing the whole picture only as the last resort) until
    none does (see ``_make_room``); positions may then reach past ``width`` x ``height`` (the
    caller sizes its canvas to them), and the result is a ``Layout`` whose ``label_spots`` say
    where each label has its room (see ``layout_spot``).
    """
    if node_size is None:
        return _plain_layout(graph, width, height, focus, seed)
    nodes: Dict[str, Mapping[str, Any]] = {}
    for node in (graph or {}).get("nodes") or []:
        if isinstance(node, Mapping) and node.get("id") is not None:
            nodes.setdefault(str(node.get("id")), node)
    edges = []
    for edge in (graph or {}).get("edges") or []:
        if isinstance(edge, Mapping):
            a, b = str(edge.get("a")), str(edge.get("b"))
            if a in nodes and b in nodes and a != b:
                edges.append((a, b, edge))
    names = {key: tuple(node_size(node)) for key, node in nodes.items()}
    labels: Dict[Any, Tuple[float, float]] = {}
    if label_size is not None:
        for index, (_a, _b, edge) in enumerate(edges):
            size = label_size(edge)
            if size:
                labels[index] = tuple(size)
    capped = len(labels) > ROOM_FOR_LABELS_MAX_EDGES or len(edges) > ROOM_FOR_LABELS_MAX_EDGES
    # Every line stays clear of the names it does not end at; above ``_NAME_LINES_MAX_EDGES``
    # lines only the focus character's lines do (checking them all against every name costs seconds).
    name_lines = list(range(len(edges)))
    if capped:
        # too many to make room for all: the focus character's only (the diagrams then label
        # only that character's lines)
        mine = [index for index, (a, b, _edge) in enumerate(edges) if focus is not None and str(focus) in (a, b)]
        labels = {index: labels[index] for index in mine if index in labels}
        if len(labels) > ROOM_FOR_LABELS_MAX_EDGES:
            labels = {}
        if len(edges) > _NAME_LINES_MAX_EDGES:
            name_lines = mine
    # How far apart names on a ring around a focus should start: about a name box, or a label
    # (labels sit between two rings, where the ring is shorter). The room pass fixes the rest.
    spacing = 0.0
    if names:
        spacing = 0.8 * sum(w for w, _h in names.values()) / len(names)
    if labels:
        spacing = max(spacing, 0.6 * sum(w + h for w, h in labels.values()) / len(labels))
    # The picture is built round the focus: it is the centre, its relations the nearest names round
    # it, the others further out by their distance from it (rings, a tree out from it, or the picture
    # without a focus with the focus drawn in among its relations: the smallest view centred on it).
    centre = str(focus) if focus is not None and str(focus) in nodes else None
    if centre is not None and edges and not any(centre in (a, b) for a, b, _edge in edges):
        # a focus with no relations: the picture without it, and it at the free spot nearest the middle
        return _round_a_loner(graph, width, height, centre, seed, node_size, label_size, nodes, edges, names)
    spokes = {}
    if centre is not None:
        for index, (a, b, _edge) in enumerate(edges):
            if index in labels and centre in (a, b):
                spokes[b if a == centre else a] = labels[index]
    positions = _plain_layout(graph, width, height, centre, seed, spacing, spokes, wedges=not capped)
    if len(positions) < 2:
        return positions
    # Names without a line (gender tracked only) take no part in making room: growing the picture
    # would carry them off with it. They go in rows under it.
    linked = {a for a, _b, _edge in edges} | {b for _a, b, _edge in edges}
    loose = {key for key in positions if key not in linked} if linked else set()
    start = {key: [point[0], point[1]] for key, point in positions.items() if key not in loose}
    keyed = [(index, a, b) for index, (a, b, _edge) in enumerate(edges)]
    boxes = {key: size for key, size in names.items() if key not in loose}
    pinned = centre if centre is not None and centre in positions else None
    if focus is None or str(focus) not in positions:
        _fan_out(start, [(a, b) for a, b, _edge in edges], labels, names)
    guide = {key: list(point) for key, point in start.items()}
    spots: Dict[Any, float] = {}
    spread = 1.0
    cell = _room_cell(boxes, labels)
    aspect = float(width) / float(height) if width and height else 1.0
    # Names are placed one at a time (see ``_place_one_by_one``) unless the rings or the force
    # layout already leave room for everything, or every line runs from one name (a plain star:
    # the rings and the room pass make it smaller, measured).
    # Round a focus they are placed one by one also above the label cap (room is made for its labels
    # only) and round a leaf of a star: there the rings and the room pass let other names in among
    # the focus's own relations.
    shared = set.intersection(*({a, b} for a, b, _edge in edges)) if edges else set()
    star = len(edges) > 2 and bool(shared) and (pinned is None or pinned in shared)
    own = {b if a == pinned else a for a, b, _edge in edges if pinned in (a, b)} if pinned is not None else set()
    placed = ((not capped or pinned is not None) and not star
              and bool(_room_round(start, keyed, boxes, labels, name_lines, {}, cell)[0]))
    if not placed:
        moved, spots, growth = _make_room(start, keyed, boxes, labels, name_lines, pinned=pinned, spots=spots)
    else:
        # Guided by the force layout (or the rings round the focus) and, round a main character,
        # also by a tree out from it; round a focus also by the picture without one (see
        # ``_kept_guide``); each placed with candidate spots on its rays this far apart: the smallest
        # picture is kept (no guide and no spacing wins on every shape, measured; the second spacing
        # only while placing took few checks, and not round a focus, which already tries more
        # guides), each try given up once it is larger.
        attempts = [(start, pinned, _out_from(pinned, edges, start) if pinned is not None else None)]
        adjacency: Dict[str, List[str]] = {}
        between: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for index, (a, b, _edge) in enumerate(edges):
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            if index in labels:
                between[(a, b)] = between[(b, a)] = labels[index]
        hub = _tree_root(edges, pinned)
        if hub is not None:
            tree = _tree_layout(sorted(start), adjacency, hub, boxes, lambda a, b: between.get((a, b)))
            ox, oy = positions[hub]
            tree_start = {key: [ox + x, oy + y] for key, (x, y) in tree.items()}
            attempts.append((tree_start, hub, list(tree_start)))
        if pinned is not None:
            # also the picture the book has without a focus, the focus drawn in among its relations
            # and the others kept out of their ring as they are placed round it: round a name at the
            # edge of the book the rings and the tree out from it put everything on one side of it,
            # and the view centred on it grew to over twice the picture without a focus (the owner's
            # book, measured over every name: 22 of 62 names; with this guide too, 1)
            kept, kept_order = _kept_guide(graph, width, height, seed, spacing, capped, edges, labels, names, start,
                                           adjacency, between, boxes, pinned, own)
            attempts.append((kept, pinned, kept_order))
        best = None
        work = 0
        for grow_ray in (_PLACE_GROW_RAY, _PLACE_GROW_RAY_TOO):
            if best is not None and (work > _TRY_TOO_CHECKS or pinned is not None):
                break
            for guide_start, root, order in attempts:
                done = _place_one_by_one(guide_start, keyed, boxes, labels, name_lines, cell, root, order, aspect,
                                         None if best is None or best[0][0] else
                                         (_GIVE_UP if pinned is None else _GIVE_UP_FOCUS) * best[0][1], grow_ray,
                                         centred=pinned is not None and root == pinned)
                if done is None:
                    continue
                work += done[4]
                other, other_spots, other_growth = _make_room(done[0], keyed, boxes, labels, name_lines,
                                                              pinned=pinned, spots=done[1])
                # (round a focus: a picture whose first ring holds only the focus's relations first)
                size = (not _ring_holds(other, pinned, own),
                        _drawn_extent(other, keyed, boxes, labels, other_spots, aspect, pinned))
                if best is None or size < best[0]:
                    best = (size, other, other_spots, other_growth, done[3])
        _size, moved, spots, growth, spread = best
    if placed and len(boxes) <= _COMPARE_MAX_NAMES:
        # a few names: the room pass alone is cheap here, and keeps the picture nearest the guide;
        # kept unless placing one by one ends clearly smaller
        other, other_spots, other_growth = _make_room(guide, keyed, boxes, labels, name_lines, pinned=pinned)
        if (_ring_holds(other, pinned, own) >= _ring_holds(moved, pinned, own)
                and _drawn_extent(moved, keyed, boxes, labels, spots, aspect, pinned)
                > _COMPARE_GAIN * _drawn_extent(other, keyed, boxes, labels, other_spots, aspect, pinned)):
            moved, spots, growth, spread = other, other_spots, other_growth, 1.0
    if pinned is not None and pinned in moved:
        # the focus where the rings have it, whichever guide won (the picture moved as a whole)
        dx, dy = positions[pinned][0] - moved[pinned][0], positions[pinned][1] - moved[pinned][1]
        moved = {key: [point[0] + dx, point[1] + dy] for key, point in moved.items()}
    if spots and labels:
        spots = _slide_labels(moved, keyed, boxes, labels, name_lines, spots, cell)
    if loose:
        _place_in_rows(moved, [key for key in positions if key in loose], names)
    result = Layout((key, (point[0], point[1])) for key, point in moved.items())
    result.label_spots = {(edges[index][0], edges[index][1]): t for index, t in spots.items()}
    result.labels_capped = capped
    result.growth = growth
    result.spread = spread
    return result


def _slide_labels(positions: Mapping[str, Sequence[float]], edges: Sequence[Tuple[Any, str, str]],
                  node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                  name_lines: Sequence[int], spots: Mapping[Any, float], cell: float) -> Dict[Any, float]:
    """``spots`` with each label slid along its line to the free spot nearest the end with more
    relations, where that is nearer than its spot now (free: clear of every name, every other label
    and every other line, as when names are placed). A label was placed when its line was drawn,
    nearest that end then; what was placed later, and the room pass, may have left room nearer
    (the owner's book: 60 of 115 labels were past 60 % of their line from that end)."""
    placed = _Placed(cell, node_size, edges, label_size, name_lines)
    for node in sorted(positions):
        if node in node_size:
            placed.place(node, positions[node][0], positions[node][1], ())
    half = _ROOM_GAP / 2.0
    lines_of = {node: len(placed.incident.get(node, ())) for node in positions}
    result = dict(spots)
    for index, (key, a, b) in enumerate(edges):
        t, size = spots.get(key), label_size.get(key)
        if t is None or size is None or a not in placed.positions or b not in placed.positions:
            continue
        pa, pb = placed.positions[a], placed.positions[b]
        x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
        placed._add_label(index, (x - size[0] / 2.0 - half, y - size[1] / 2.0 - half,
                                  x + size[0] / 2.0 + half, y + size[1] / 2.0 + half))
    # twice: a label that slid may leave room nearer for one looked at before it
    for index, (key, a, b) in list(enumerate(edges)) * 2:
        if index not in placed.label_box:
            continue
        pa, pb = placed.positions[a], placed.positions[b]
        size = label_size[key]
        old = placed.label_box[index]
        placed._drop_label(index)
        free = placed._stretches(pa, pb, size, placed._near(_cells_along(pa, pb, cell), False), a, index,
                                 (), (), ())
        t = result[key]
        from_a = lines_of.get(a, 0) >= lines_of.get(b, 0)
        if free:
            inside = 0.5 / max(math.hypot(pb[0] - pa[0], pb[1] - pa[1]), 1.0)
            lo, hi = free[0] if from_a else free[-1]
            new = min(lo + inside, (lo + hi) / 2.0) if from_a else max(hi - inside, (lo + hi) / 2.0)
            here = any(low <= t <= high for low, high in free)
            if not here or (new < t if from_a else new > t):
                cx, cy = pa[0] + (pb[0] - pa[0]) * new, pa[1] + (pb[1] - pa[1]) * new
                placed._add_label(index, (cx - size[0] / 2.0 - half, cy - size[1] / 2.0 - half,
                                          cx + size[0] / 2.0 + half, cy + size[1] / 2.0 + half))
                result[key] = new
                continue
        placed._add_label(index, old)
    return result


#: Above the label cap, up to this many lines all stay clear of the names they do not end at (above
#: it the picture that needs grows fast: a main character with 30 relations among 70 names and 160
#: lines in all, 5,800 px; with 300 lines, 17,600 against 5,200 when only the focus's lines do).
_NAME_LINES_MAX_EDGES = 160


#: Up to this many names with relations the room pass alone is tried too, and kept unless placing
#: one by one ends smaller than this share of it.
_COMPARE_MAX_NAMES = 40
_COMPARE_GAIN = 0.85
#: The tree out from a main character is given up once what it has placed reaches this many times
#: past the picture the force layout gave (the room pass may still make it smaller: by up to a
#: quarter, measured, round a main character with 45 relations).
_GIVE_UP = 1.35
#: ... and round a focus (three guides are tried, see ``layout_graph``) as soon as it is larger than
#: the smallest so far, centred on the focus (the owner's book over every name: the time the median
#: name takes from 2.2 to 1.9 times the picture without a focus, the sizes as good).
_GIVE_UP_FOCUS = 1.0


def _drawn_extent(positions: Mapping[str, Sequence[float]], edges: Sequence[Tuple[Any, str, str]],
                  node_size: Mapping[str, Tuple[float, float]], label_size: Mapping[Any, Tuple[float, float]],
                  spots: Mapping[Any, float], aspect: float, around: Optional[str] = None) -> float:
    """How far the names and labels (at ``spots``) must be zoomed out to fit a window ``aspect``
    times as wide as high: the larger of the width and ``aspect`` times the height. ``around``: a
    name the window centres on (the focus): the width and height reach as far on its other side."""
    boxes = []
    for node, (w, h) in node_size.items():
        x, y = positions[node]
        boxes.append((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))
    for key, a, b in edges:
        t, size = spots.get(key), label_size.get(key)
        if t is None or size is None:
            continue
        pa, pb = positions[a], positions[b]
        x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
        boxes.append((x - size[0] / 2.0, y - size[1] / 2.0, x + size[0] / 2.0, y + size[1] / 2.0))
    if not boxes:
        return 0.0
    width, height = _centred_size(boxes, positions[around] if around in positions else None)
    return max(width, aspect * height)


def _centred_size(boxes: Sequence[Box], centre: Optional[Sequence[float]]) -> Tuple[float, float]:
    """``(width, height)`` of what ``boxes`` cover; with a ``centre``, of the area centred on it that
    holds them all (as far on each side of it as they reach on the further side)."""
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if centre is None:
        return x1 - x0, y1 - y0
    cx, cy = centre[0], centre[1]
    return 2.0 * max(cx - x0, x1 - cx), 2.0 * max(cy - y0, y1 - cy)


def _tree_root(edges: Sequence[Tuple[str, str, Any]], pinned: Optional[str]) -> Optional[str]:
    """The name a labelled layout hangs everything from (see ``_tree_layout``) when one name has
    at least twice the relations of any other and is related to at least a third of all names with
    relations (a book round one main character): the focus, else that name. None: the rings round
    the focus or the force layout suit better (a tree out from the root spreads a sparse picture
    with long branches far out), or every line runs from one name (a plain star: the rings and the
    room pass, measured smaller there)."""
    if pinned is not None and all(pinned in (a, b) for a, b, _edge in edges):
        return None
    count: Dict[str, int] = {}
    for a, b, _edge in edges:
        count[a] = count.get(a, 0) + 1
        count[b] = count.get(b, 0) + 1
    if len(count) < 3:
        return None
    ranked = sorted(count, key=lambda node: (-count[node], node))
    top, second = count[ranked[0]], count[ranked[1]]
    if top >= 2 * second and 3 * top >= len(count) and top < len(edges):
        return pinned if pinned is not None else ranked[0]
    return None


def _out_from(centre: str, edges: Sequence[Tuple[str, str, Any]], names: Iterable[str]) -> List[str]:
    """``names`` in the order to place them round ``centre``: out from it by the number of steps
    from it, the names with most relations first at each step (those not reaching it last)."""
    links: Dict[str, List[str]] = {}
    for a, b, _edge in edges:
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    depth = {centre: 0}
    queue = [centre]
    for node in queue:
        for other in links.get(node, ()):
            if other not in depth:
                depth[other] = depth[node] + 1
                queue.append(other)
    far = len(depth) + 1
    return sorted(names, key=lambda node: (depth.get(node, far), -len(links.get(node, ())), node))


#: In the guide from the picture without a focus (``_kept_guide``) the focus moves this share of the
#: way from where that picture has it to the middle of the picture.
_KEPT_PULL = 0.5


def _kept_guide(graph: Optional[Mapping[str, Any]], width: float, height: float, seed: int, spacing: float,
                capped: bool, edges: Sequence[Tuple[str, str, Any]], labels: Mapping[Any, Tuple[float, float]],
                names: Mapping[str, Tuple[float, float]], start: Mapping[str, Sequence[float]],
                adjacency: Mapping[str, Sequence[str]], between: Mapping[Tuple[str, str], Tuple[float, float]],
                boxes: Mapping[str, Tuple[float, float]], centre: str, own: Iterable[str]
                ) -> Tuple[Dict[str, List[float]], List[str]]:
    """``(guide, order)`` for the names ``start`` holds, round ``centre``: where the picture without
    a focus would have them (the tree out from the main character, else the force layout), with
    ``centre`` moved part of the way to the middle of that picture (see ``_KEPT_PULL``) and its
    relations (``own``) drawn in close round it; the order places the focus, then its relations, then
    the rest as that picture does. Placed round the focus (``_place_one_by_one``) the book keeps its
    shape, and the focus sits inside it rather than at its edge, where the rings or a tree out from
    a name at the edge of the book put it."""
    hub = _tree_root(edges, None)
    first = [centre] + sorted((other for other in own if other in start),
                              key=lambda other: (-len(adjacency.get(other, ())), other))
    if hub is not None:
        tree = _tree_layout(sorted(start), adjacency, hub, boxes, lambda a, b: between.get((a, b)))
        guide = {key: [point[0], point[1]] for key, point in tree.items() if key in start}
        order = first + [key for key in guide if key not in first]
    else:
        plain = _plain_layout(graph, width, height, None, seed, spacing, {}, wedges=not capped)
        guide = {key: [plain[key][0], plain[key][1]] for key in start}
        _fan_out(guide, [(a, b) for a, b, _edge in edges], labels, names)
        order = first
    if centre not in guide:
        return guide, order
    # the focus part of the way to the middle (the view is centred on it), its relations moved in
    # along their directions from it to no further than their lines and labels need
    xs = [point[0] for point in guide.values()]
    ys = [point[1] for point in guide.values()]
    mx, my = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    gx, gy = guide[centre]
    px, py = gx + (mx - gx) * _KEPT_PULL, gy + (my - gy) * _KEPT_PULL
    guide[centre] = [px, py]
    fw, fh = names[centre]
    clear = 2.0 * LINE_CLEARANCE + _ROOM_GAP
    for index, (a, b, _edge) in enumerate(edges):
        if centre not in (a, b):
            continue
        other = b if a == centre else a
        if other not in guide:
            continue
        ox, oy = guide[other]
        dx, dy = ox - gx, oy - gy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            dx, dy, length = 1.0, 0.0, 1.0
        ux, uy = dx / length, dy / length
        ow, oh = names[other]
        lw, lh = labels.get(index, (0.0, 0.0))
        short = (abs(ux) * (fw + ow) / 2.0 + abs(uy) * (fh + oh) / 2.0 + 2.0 * (abs(ux) * lw + abs(uy) * lh)
                 + 2.0 * clear)
        reach = min(math.hypot(ox - px, oy - py), short)
        guide[other] = [px + ux * reach, py + uy * reach]
    return guide, order


def _round_a_loner(graph: Optional[Mapping[str, Any]], width: float, height: float, centre: str, seed: int,
                   node_size: Callable[[Mapping[str, Any]], Any],
                   label_size: Optional[Callable[[Mapping[str, Any]], Any]],
                   nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Tuple[str, str, Any]],
                   names: Mapping[str, Tuple[float, float]]) -> "Layout":
    """A focus with no relations (gender tracked only): the picture without it, as without a focus,
    and the focus at the free spot nearest the middle of that picture, clear of every name, label and
    line: the centre of the diagram."""
    rest = dict(graph or {})
    rest["nodes"] = [node for key, node in nodes.items() if key != centre]
    base = layout_graph(rest, width, height, None, seed, node_size, label_size)
    keyed = [(index, a, b) for index, (a, b, _edge) in enumerate(edges)]
    spots: Dict[int, float] = {}
    sizes: Dict[int, Tuple[float, float]] = {}
    for index, a, b in keyed:
        t = layout_spot(base, a, b)
        size = label_size(edges[index][2]) if label_size is not None and t is not None else None
        if size:
            spots[index] = t if (a, b) in base.label_spots else 1.0 - t
            sizes[index] = tuple(size)
    boxes = {key: names[key] for key in base}
    boxes[centre] = names[centre]
    cell = _room_cell(boxes, sizes)
    placed = _Placed(cell, boxes, keyed, sizes, range(len(keyed)))
    for key in sorted(base):
        placed.place(key, base[key][0], base[key][1], ())
    half = _ROOM_GAP / 2.0
    for index, t in spots.items():
        _key, a, b = keyed[index]
        pa, pb = base[a], base[b]
        x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
        lw, lh = sizes[index]
        placed._add_label(index, (x - lw / 2.0 - half, y - lh / 2.0 - half, x + lw / 2.0 + half, y + lh / 2.0 + half))
    linked = {a for a, _b, _edge in edges} | {b for _a, b, _edge in edges}
    middle = [base[key] for key in base if key in linked] or list(base.values())
    mx = (min(p[0] for p in middle) + max(p[0] for p in middle)) / 2.0
    my = (min(p[1] for p in middle) + max(p[1] for p in middle)) / 2.0
    span = max(_span({key: base[key] for key in base}) + (1.0,))
    spot = None
    radius = 0.0
    while spot is None and radius <= span:
        count = 1 if radius == 0.0 else max(8, int(2.0 * math.pi * radius / (_PLACE_STEP / 2.0)))
        tries = sorted(((radius * math.cos(2.0 * math.pi * k / count), radius * math.sin(2.0 * math.pi * k / count))
                        for k in range(count)), key=lambda d: (abs(d[1]), d[0], d[1]))
        for dx, dy in tries:
            bad, moved = placed.check(centre, mx + dx, my + dy, False)
            if not bad and not moved:
                spot = (mx + dx, my + dy)
                break
        radius += _PLACE_STEP / 2.0
    if spot is None:  # (nothing free inside the picture) under it
        bottom = max(b[3] for b in placed.name_box.values())
        spot = (mx, bottom + names[centre][1] + _ROOM_GAP + 2.0 * LINE_CLEARANCE)
    result = Layout((key, (point[0], point[1])) for key, point in base.items())
    result[centre] = spot
    for attribute in ("label_spots", "labels_capped", "growth", "spread"):
        setattr(result, attribute, getattr(base, attribute))
    return result


def _fan_out(positions: Dict[str, List[float]], pairs: Sequence[Tuple[str, str]],
             label_size: Mapping[int, Tuple[float, float]], node_size: Mapping[str, Tuple[float, float]]) -> None:
    """Around a name with several relations that have no other relation (leaves), those names
    spread evenly over the free directions between its other lines, far enough out that the label
    on each line clears the lines beside it: a label of width w on a line at angle a reaches
    ``|sin a| w / 2 + |cos a| h / 2`` sideways, which needs a distance of that (plus the clearance)
    over the sine of the angle to the next line."""
    links: Dict[str, List[Tuple[str, int]]] = {}
    for index, (a, b) in enumerate(pairs):
        links.setdefault(a, []).append((b, index))
        links.setdefault(b, []).append((a, index))
    done = set()
    for hub in sorted(links, key=lambda node: (-len(links[node]), node)):
        leaves = [(other, index) for other, index in links[hub] if len(links[other]) == 1 and other not in done]
        if len(leaves) < _FAN_MIN or hub in done or 2 * len(leaves) < len(links[hub]):
            continue  # mostly lines to names with other relations: the gaps between those decide
        hx, hy = positions[hub]

        def angle_of(node: str) -> float:
            return math.atan2(positions[node][1] - hy, positions[node][0] - hx) % (2.0 * math.pi)
        fixed = sorted(angle_of(other) for other, _index in links[hub] if len(links[other]) != 1)
        leaves.sort(key=lambda item: (angle_of(item[0]), item[0]))
        every = sorted(angle_of(other) for other, _index in links[hub])

        def roomy(other: str, index: int) -> bool:
            """The label in the middle of the line already clears the lines on both sides."""
            theta = angle_of(other)
            position = every.index(theta)
            gap = min((theta - every[position - 1]) % (2.0 * math.pi) or 2.0 * math.pi,
                      (every[(position + 1) % len(every)] - theta) % (2.0 * math.pi) or 2.0 * math.pi)
            w, h = label_size.get(index, (0.0, 0.0))
            sideways = abs(math.sin(theta)) * w / 2.0 + abs(math.cos(theta)) * h / 2.0 + LINE_CLEARANCE
            distance = math.hypot(positions[other][0] - hx, positions[other][1] - hy)
            return distance / 2.0 * math.sin(min(gap, math.pi / 2.0)) >= sideways
        if all(roomy(other, index) for other, index in leaves):
            done.add(hub)
            continue  # nothing to spread: the picture stays as it was
        lengths = sorted(math.hypot(positions[other][0] - hx, positions[other][1] - hy) for other, _i in links[hub])
        longest = 2.5 * lengths[len(lengths) // 2]
        if not fixed:
            # all around: gaps as wide as the labels need (see ``_spread_spokes``)
            start = angle_of(leaves[0][0])
            thetas, gaps = _spread_spokes([start + 2.0 * math.pi * k / len(leaves) for k in range(len(leaves))],
                                          [label_size.get(index) for _other, index in leaves], [1] * len(leaves))
            slots = [(theta, min(gaps[k - 1], gaps[k])) for k, theta in enumerate(thetas)]
            longest = _BIG
        else:
            # each leaf stays in the gap between two other lines it is in now, spread evenly there
            groups: Dict[int, List[int]] = {}
            for order, (other, _index) in enumerate(leaves):
                theta = angle_of(other)
                gap = sum(1 for value in fixed if value <= theta) % len(fixed)
                groups.setdefault(gap, []).append(order)
            slots = [(0.0, 0.0)] * len(leaves)
            for gap, members in groups.items():
                low = fixed[gap - 1]
                high = fixed[gap] if gap > 0 else fixed[0] + 2.0 * math.pi
                width = (high - low) % (2.0 * math.pi) or 2.0 * math.pi
                for k, order in enumerate(members):
                    slots[order] = (low + width * (k + 1) / (len(members) + 1), width / (len(members) + 1))
        for (other, index), (theta, spacing) in zip(leaves, slots):
            w, h = label_size.get(index, (0.0, 0.0))
            sideways = abs(math.sin(theta)) * w / 2.0 + abs(math.cos(theta)) * h / 2.0 + LINE_CLEARANCE
            # the label in the middle of the line clears the lines beside it
            need = 2.0 * sideways / max(math.sin(min(spacing, math.pi / 2.0)), 0.05) + max(node_size[other]) / 2.0
            # (between other lines not further out than the hub's lines reach: a narrow gap is the
            # growing's to fix, not one long line's)
            distance = max(math.hypot(positions[other][0] - hx, positions[other][1] - hy), min(need, longest))
            positions[other] = [hx + distance * math.cos(theta), hy + distance * math.sin(theta)]
            done.add(other)
        done.add(hub)


def _place_in_rows(positions: Dict[str, List[float]], loose: Sequence[str],
                   node_size: Mapping[str, Tuple[float, float]]) -> None:
    """``loose`` names in rows under the names of ``positions``, as wide as the picture, clear of
    every line and label (all of those are above the rows)."""
    gap = _ROOM_GAP + 2.0 * LINE_CLEARANCE
    lefts = [p[0] - node_size[key][0] / 2.0 for key, p in positions.items()]
    rights = [p[0] + node_size[key][0] / 2.0 for key, p in positions.items()]
    bottom = max(p[1] + node_size[key][1] / 2.0 for key, p in positions.items())
    left, right = min(lefts), max(rights)
    right = max(right, left + max(node_size[key][0] for key in loose))
    row_height = max(node_size[key][1] for key in loose) + gap
    x, y = left, bottom + gap + row_height / 2.0
    for key in loose:
        w = node_size[key][0]
        if x > left and x + w > right:
            x, y = left, y + row_height
        positions[key] = [x + w / 2.0, y]
        x += w + gap


def _plain_layout(graph: Optional[Mapping[str, Any]], width: float, height: float,
                  focus: Optional[str], seed: int, spacing: float = 0.0,
                  spoke_size: Optional[Mapping[str, Tuple[float, float]]] = None,
                  wedges: bool = True) -> Dict[str, Tuple[float, float]]:
    width = float(width) if width and width > 0 else 1000.0
    height = float(height) if height and height > 0 else 700.0
    nodes = list((graph or {}).get("nodes") or [])
    ids = [str(n.get("id")) for n in nodes if isinstance(n, Mapping) and n.get("id") is not None]
    seen = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: (width / 2.0, height / 2.0)}
    id_set = set(ids)
    adjacency: Dict[str, List[str]] = {i: [] for i in ids}
    edge_list = []
    for edge in (graph or {}).get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        a, b = str(edge.get("a")), str(edge.get("b"))
        if a in id_set and b in id_set and a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)
            edge_list.append((a, b))
    if focus is not None and str(focus) in id_set:
        return _radial_layout(ids, adjacency, str(focus), width, height, spacing, spoke_size, wedges)

    n = len(ids)
    if n <= _REFINED_LAYOUT_MAX_NODES:
        return _fit_positions(_refined_layout(ids, edge_list, width, height, seed), width, height)

    rng = random.Random(seed)
    area = width * height
    k = math.sqrt(area / n)
    positions = {}
    radius = min(width, height) * 0.4
    for index, node in enumerate(ids):
        theta = 2.0 * math.pi * index / n
        jitter = 0.85 + 0.3 * rng.random()
        positions[node] = [width / 2.0 + radius * jitter * math.cos(theta),
                           height / 2.0 + radius * jitter * math.sin(theta)]
    iterations = 60 if n <= 100 else (40 if n <= 400 else 25)
    temperature = min(width, height) / 10.0
    cooling = temperature / (iterations + 1)
    cell = 2.0 * k
    k2 = k * k
    cutoff2 = cell * cell
    gravity = 0.02
    cx, cy = width / 2.0, height / 2.0
    # For a diagram that makes room for labels (``spacing``): two well-connected names pull less
    # and push each other more, so a name with many relations gets long lines and two of them do
    # not sit side by side with their lines to the same names running almost together.
    degree = {node: 0 for node in ids}
    for a, b in edge_list:
        degree[a] += 1
        degree[b] += 1
    weight = {node: (degree[node] + 1.0) ** 0.75 if spacing > 0.0 else 1.0 for node in ids}
    for _iteration in range(iterations):
        disp = {node: [0.0, 0.0] for node in ids}
        grid: Dict[Tuple[int, int], List[str]] = {}
        for node in ids:
            x, y = positions[node]
            grid.setdefault((int(x // cell), int(y // cell)), []).append(node)
        for (gx, gy), members in grid.items():
            neighbours = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbours.extend(grid.get((gx + dx, gy + dy), ()))
            for v in members:
                vx, vy = positions[v]
                dv = disp[v]
                for u in neighbours:
                    if u is v:
                        continue
                    ux, uy = positions[u]
                    ddx, ddy = vx - ux, vy - uy
                    dist2 = ddx * ddx + ddy * ddy
                    if dist2 > cutoff2:
                        continue
                    if dist2 < 1e-6:
                        ddx, ddy = rng.random() - 0.5, rng.random() - 0.5
                        dist2 = ddx * ddx + ddy * ddy + 1e-6
                    factor = k2 / dist2 * weight[u]
                    dv[0] += ddx * factor
                    dv[1] += ddy * factor
        for a, b in edge_list:
            ax, ay = positions[a]
            bx, by = positions[b]
            ddx, ddy = ax - bx, ay - by
            dist = math.sqrt(ddx * ddx + ddy * ddy) or 1e-3
            force = dist / (k * min(weight[a], weight[b]))
            fx, fy = ddx * force, ddy * force
            disp[a][0] -= fx
            disp[a][1] -= fy
            disp[b][0] += fx
            disp[b][1] += fy
        for node in ids:
            x, y = positions[node]
            dx, dy = disp[node]
            dx += (cx - x) * gravity * k / 10.0
            dy += (cy - y) * gravity * k / 10.0
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-9:
                limited = min(length, temperature)
                x += dx / length * limited
                y += dy / length * limited
            positions[node] = [min(max(x, 0.0), width), min(max(y, 0.0), height)]
        temperature = max(temperature - cooling, 1.0)
    return _fit_positions(positions, width, height)
