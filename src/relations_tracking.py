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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    wanted = _label_key(label)
    return any(_on_edge(r, key, tkey) and _label_key(r.get("label")) == wanted
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
    wanted = _label_key(label)
    edits["ignored"] = [
        r for r in _records(edits, "ignored")
        if not (_on_edge(r, key, tkey) and _label_key(r.get("label")) == wanted
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
    if not found:
        return result
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
    'The `{field}` column: for character entries only, list this character\'s relationships to OTHER named '
    'characters that are shown or stated in THIS text. Write each as `raw_name: role`, separated by "; ", '
    'where raw_name is the other character\'s raw_name exactly as in their own entry and role is what THIS '
    'row\'s character is to that other character, in {language}, 1–3 words (Father, Mother, Daughter, Older '
    'Sister, Wife, Fiancé, Best Friend, Master, Disciple, Subordinate, Rival, Enemy). Both directions: a '
    'father\'s row has `<his daughter\'s raw_name>: Father` and the daughter\'s row has `<her father\'s '
    'raw_name>: Daughter`. List only real, specific relationships; never filler such as Acquaintance, Stranger '
    'or Unknown, and never a note such as Surname. Keep the same wording for the same relationship; use a '
    'different word only when the relationship itself changed in this text. Do not guess, do not describe events '
    'or feelings, never use commas inside this column. Leave it empty for non-character entries, surname-only '
    'entries, and when no relationship is shown.\n'
    'Example rows that mirror each other: `{example_row_1}` and `{example_row_2}`'
)
_DECLARE_TEMPLATE = 'Add a `{field}` column as one extra LAST column of every row. '

_BASE_COLUMNS = ("type", "raw_name", "translated_name", "gender")
#: The two made-up characters of the glossary prompt's own example rows. They are not
#: real names, so a model cannot take them for characters of the book it is reading.
#: Kept SHORT: a long instruction made a 7B model that followed the short one ignore it
#: (owner, 2026-11-13). The description cell is only there to hold the column's place.
_EXAMPLE_ROWS = (
    {"type": "character", "raw_name": "이히리ᐐ 나애", "translated_name": "Dihirit Ade", "gender": "female",
     "description": '"Guild leader"', "relations": "뢤사난: Older Sister"},
    {"type": "character", "raw_name": "뢤사난", "translated_name": "Kim Sang-hyu", "gender": "male",
     "description": '"Swordsman"', "relations": "이히리ᐐ 나애: Younger Brother"},
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


def instruction_text(field: Optional[str] = None, language: Optional[str] = "English",
                     declare_column: bool = False, columns: Optional[Iterable[Any]] = None,
                     template: Optional[str] = None) -> str:
    """The ``{relations_instruction}`` expansion.

    ``columns`` is the row layout the prompt asks for. With it the two example rows have
    every column in its place; a prompt whose other examples end at ``description``
    otherwise teaches the model to write the relations there. Nothing else is added for
    it: the text stays as short as it was with the five-column examples.
    ``template`` replaces the built-in text (a user-edited one); it may use the tokens
    ``{field}``, ``{language}``, ``{example_row_1}`` and ``{example_row_2}``, or none.
    """
    field = (field or "").strip() or field_name()
    language = (language or "").strip() or "English"
    layout = instruction_columns(columns if columns else _BASE_COLUMNS, field)
    text = _INSTRUCTION_TEMPLATE if template is None else str(template)
    text = text.replace("{example_row_1}", _example_row(_EXAMPLE_ROWS[0], layout, field))
    text = text.replace("{example_row_2}", _example_row(_EXAMPLE_ROWS[1], layout, field))
    text = text.replace("{field}", field).replace("{language}", language)
    if declare_column:
        text = _DECLARE_TEMPLATE.replace("{field}", field) + text
    return text


_PLACEHOLDER_LINE_RE = re.compile(
    r"^[ \t]*-?[ \t]*" + re.escape(RELATIONS_PLACEHOLDER) + r"[ \t]*\r?\n?",
    flags=re.MULTILINE,
)


def apply_relations_placeholder(prompt_text: Any, enabled: Optional[bool] = None, field: Optional[str] = None,
                                language: Optional[str] = None, declare_column: bool = False,
                                columns: Optional[Iterable[Any]] = None, template: Optional[str] = None) -> Any:
    """Expand ``{relations_instruction}`` when tracking is on; remove its whole line when off.

    ``template``: a user-edited instruction text (``instruction_text``); one that expands to
    nothing removes the line as when tracking is off.
    """
    if not isinstance(prompt_text, str) or RELATIONS_PLACEHOLDER not in prompt_text:
        return prompt_text
    if enabled is None:
        enabled = tracking_enabled()
    if enabled:
        if language is None:
            language = os.environ.get("GLOSSARY_TARGET_LANGUAGE", "English")
        text = instruction_text(field, language, declare_column=declare_column, columns=columns,
                                template=template)
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


def _radial_layout(ids: List[str], adjacency: Dict[str, List[str]], focus: str,
                   width: float, height: float) -> Dict[str, Tuple[float, float]]:
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
    for node in order[1:]:
        rings.setdefault(depth[node], []).append(node)
    if unreachable:
        max_depth += 1
        rings[max_depth] = unreachable
    radius = min(width, height) / 2.0 * 0.88
    step = radius / max_depth if max_depth else 0.0
    cx, cy = width / 2.0, height / 2.0
    angle: Dict[str, float] = {focus: 0.0}
    positions = {focus: (cx, cy)}
    for ring in sorted(rings):
        members = rings[ring]
        members.sort(key=lambda n: (angle.get(parent.get(n) or "", 10.0), n))
        count = len(members)
        offset = (math.pi / count) * (ring % 2)
        for index, node in enumerate(members):
            theta = offset + 2.0 * math.pi * index / count
            angle[node] = theta
            positions[node] = (cx + step * ring * math.cos(theta), cy + step * ring * math.sin(theta))
    return positions


#: Up to this many characters the layout is tried several times and the tidiest one kept.
_REFINED_LAYOUT_MAX_NODES = 40
_REFINED_LAYOUT_ITERATIONS = 200
#: Pair-repulsion evaluations spent on one small graph (keeps a phone under about a second).
_REFINED_LAYOUT_BUDGET = 300_000
_REFINED_LAYOUT_MAX_TRIES = 8  # measured: as tidy as 16 on a real book's shape, in half the time
#: Past this many lines no move untangles much, and trying costs seconds.
_UNTANGLE_MAX_EDGES = 60


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
    best: Optional[Dict[str, List[float]]] = None
    best_score = 0.0
    for attempt in range(tries):
        rng = random.Random(int(seed) * 1_000_003 + attempt)
        positions = _force_positions(ids, edge_list, width, height, rng, _REFINED_LAYOUT_ITERATIONS)
        _untangle(positions, edge_list)
        score = layout_untidiness(positions, edge_list)
        if best is None or score < best_score:
            best, best_score = positions, score
            if score == 0.0:
                break
    return best or {}


def layout_graph(graph: Optional[Mapping[str, Any]], width: float = 1000.0, height: float = 700.0,
                 focus: Optional[str] = None, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    """Deterministic positions inside ``width`` x ``height``: seeded force layout, or radial around ``focus``."""
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
        return _radial_layout(ids, adjacency, str(focus), width, height)

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
                    factor = k2 / dist2
                    dv[0] += ddx * factor
                    dv[1] += ddy * factor
        for a, b in edge_list:
            ax, ay = positions[a]
            bx, by = positions[b]
            ddx, ddy = ax - bx, ay - by
            dist = math.sqrt(ddx * ddx + ddy * ddy) or 1e-3
            force = dist / k
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
