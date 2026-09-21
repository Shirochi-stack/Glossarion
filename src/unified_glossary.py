# -*- coding: utf-8 -*-
"""Unified glossary: one deduplicated glossary shared by every novel.

Each book keeps its own ``Glossary/<book>/<book>_glossary.csv``. This module
maintains a merged copy of all of them, split by source and target language:

    Glossary/Unified Glossary/<source>-<target>/glossary_unified.csv

Deduplication (``skip_duplicate_entries``) is a serial fuzzy pass, so it only
runs at the start and end of a glossary generation phase, never per request.
Two things keep that cheap:

* ``unified_glossary_state.json`` beside each unified file records the
  ``(size, mtime_ns)`` of every input and the dedup settings, the same way the
  EPUB text cache validates itself. A full rebuild (Generate Unified Glossary)
  is skipped when nothing changed; an incremental merge is skipped when the
  current book's file is unchanged.
* Reading and language-detecting the per-book files fans out over
  ``EXTRACTION_WORKERS`` threads, the same setting Other Settings uses for
  every other parallel job (it is ``1`` when Parallel Processing is off).

The heavy glossary machinery lives in ``extract_glossary_from_epub``; it is
imported lazily so this module stays importable from the GUI and from
TransateKRtoEN without a cycle. Nothing here touches Qt.
"""

import json
import os
import tempfile
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from glossary_paths import resolve_shared_glossary_dir, sanitize_glossary_folder_name

UNIFIED_FOLDER_NAME = "Unified Glossary"
UNIFIED_BASENAME = "glossary_unified"
STATE_FILENAME = "unified_glossary_state.json"
STATE_VERSION = 1

_TRUTHY = ("1", "true", "yes", "on")
_SKIP_DIR_NAMES = frozenset({"unified glossary", "backups"})
_SKIP_FILE_PREFIXES = ("glossary_extension", "glossary_unified")
_SKIP_FILE_SUFFIXES = (
    "_glossary_progress.json",
    "_gender_tracker.json",
    "_glossary_history.json",
)
# Output of extract_glossary_from_epub._detect_dominant_script_glossary -> language.
_SCRIPT_TO_LANGUAGE = {
    "korean": "korean",
    "japanese": "japanese",
    "cjk": "chinese",
    "cyrillic": "russian",
    "arabic": "arabic",
    "latin": "english",
    "other": "other",
}
_LANGUAGE_ALIASES = {
    "ko": "korean", "kor": "korean",
    "ja": "japanese", "jpn": "japanese",
    "en": "english", "eng": "english",
    "zh": "chinese", "zho": "chinese", "chi": "chinese",
    "mandarin": "chinese", "cantonese": "chinese",
}
# save_glossary_csv is reused for the unified files. These two switches stop
# it from injecting the *current* book's title into a cross-book file and from
# writing a gender tracker beside it.
_WRITE_OVERRIDES = {
    "GLOSSARY_INCLUDE_BOOK_TITLE": "0",
    "GLOSSARY_SKIP_GENDER_TRACKING": "1",
}

_LOCK = threading.RLock()


def _extractor():
    import extract_glossary_from_epub
    return extract_glossary_from_epub


# ─── Settings ────────────────────────────────────────────────────────────────

def _setting(settings, name, default=None):
    """Read from an explicit request snapshot first, then the environment."""
    if isinstance(settings, dict) and name in settings:
        value = settings.get(name)
        return default if value is None else value
    return os.getenv(name, default)


def _truthy(value):
    return str(value or "").strip().lower() in _TRUTHY


def enabled(settings=None):
    return _truthy(_setting(settings, "ENABLE_UNIFIED_GLOSSARY", "0"))


def generate_enabled(settings=None):
    return _truthy(_setting(settings, "GENERATE_UNIFIED_GLOSSARY", "0"))


def combine_all_enabled(settings=None):
    return _truthy(_setting(settings, "UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES", "0"))


def configured_source_language(settings=None):
    value = _setting(settings, "UNIFIED_GLOSSARY_SOURCE_LANGUAGE", "auto")
    return _normalize_language(value) or "auto"


def target_language(settings=None):
    return _normalize_language(_setting(settings, "OUTPUT_LANGUAGE", "English")) or "english"


# ─── Languages and folder keys ───────────────────────────────────────────────

def _normalize_language(value):
    lang = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    lang = " ".join(lang.split())
    if lang in ("", "unknown", "und", "none", "null"):
        return None
    if "chinese" in lang:
        # Detection cannot tell Simplified from Traditional, so neither can
        # the folder key.
        return "chinese"
    return _LANGUAGE_ALIASES.get(lang, lang)


def folder_key(source, target):
    src = _normalize_language(source) or "other"
    tgt = _normalize_language(target) or "english"
    return sanitize_glossary_folder_name(f"{src}-{tgt}".replace(" ", "_"))


def describe_folder_key(source_setting, combine_all, target):
    """Folder key for display before a run has resolved 'auto'."""
    if combine_all:
        return folder_key("all", target)
    return folder_key(_normalize_language(source_setting) or "auto", target)


def _language_from_script(script):
    return _SCRIPT_TO_LANGUAGE.get(str(script or "").strip().lower(), "other")


def detect_text_language(text):
    if not text:
        return "other"
    script = _extractor()._detect_dominant_script_glossary(text, max_chars=20000)
    return _language_from_script(script)


def detect_entries_language(entries):
    """Language of a glossary from the script of its raw names."""
    names = []
    for entry in entries or []:
        if isinstance(entry, dict):
            raw = str(entry.get("raw_name") or "").strip()
            if raw:
                names.append(raw)
        if len(names) >= 500:
            break
    return detect_text_language(" ".join(names))


def resolve_source_language(entries=None, chapters=None, settings=None):
    """The source language this run's unified glossary belongs to.

    Order: Combine all languages -> configured language -> the language the
    extraction already detected (SOURCE_LANGUAGE env or metadata.json) -> the
    script of the book's own glossary entries -> the script of the chapter
    text -> 'other'.
    """
    if combine_all_enabled(settings):
        return "all"
    configured = configured_source_language(settings)
    if configured and configured != "auto":
        return configured

    extractor = _extractor()
    env_lang = extractor._normalize_detected_language(
        _setting(settings, "GLOSSARY_SOURCE_LANGUAGE", "")
        or _setting(settings, "SOURCE_LANGUAGE", "")
    )
    if env_lang:
        return _normalize_language(env_lang) or "other"

    # Pure read (no module cache): the GUI process may have detected a
    # different book earlier in the same session.
    detected, _meta_path = extractor._read_detected_language_from_metadata(
        _setting(settings, "OUTPUT_PATH", "") or None,
        _setting(settings, "EPUB_PATH", "") or None,
    )
    if detected:
        return _normalize_language(detected) or "other"

    if entries:
        lang = detect_entries_language(entries)
        if lang != "other":
            return lang

    if chapters:
        parts = []
        for chapter in list(chapters)[:10]:
            text = chapter.get("body") if isinstance(chapter, dict) else chapter
            text = str(text or "").strip()
            if text:
                parts.append(text[:5000])
        lang = detect_text_language("\n".join(parts))
        if lang != "other":
            return lang
    return "other"


# ─── Paths ───────────────────────────────────────────────────────────────────

def shared_glossary_dir(settings=None, fallback_base=None):
    raw = _setting(settings, "GLOSSARY_SHARED_DIR", "") or None
    return resolve_shared_glossary_dir(raw, fallback_base=fallback_base)


def unified_root(shared_dir):
    return os.path.join(os.path.abspath(shared_dir), UNIFIED_FOLDER_NAME)


def unified_paths(shared_dir, key):
    """(folder, json_path, csv_path, state_path) for one language key."""
    folder = os.path.join(unified_root(shared_dir), key)
    return (
        folder,
        os.path.join(folder, UNIFIED_BASENAME + ".json"),
        os.path.join(folder, UNIFIED_BASENAME + ".csv"),
        os.path.join(folder, STATE_FILENAME),
    )


def is_unified_glossary_filename(name):
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    return stem.lower() == UNIFIED_BASENAME


def is_book_glossary_filename(name):
    lower = os.path.basename(str(name or "")).lower()
    if lower.startswith(_SKIP_FILE_PREFIXES) or lower.endswith(_SKIP_FILE_SUFFIXES):
        return False
    return lower.endswith(("_glossary.csv", "_glossary.json")) or lower in (
        "glossary.csv", "glossary.json",
    )


def _pick_book_glossary_file(book_dir, folder_name):
    try:
        names = os.listdir(book_dir)
    except OSError:
        return None
    files = {}
    for name in names:
        if is_book_glossary_filename(name) and os.path.isfile(os.path.join(book_dir, name)):
            files[name.lower()] = name
    if not files:
        return None
    for ext in (".csv", ".json"):
        preferred = f"{folder_name}_glossary{ext}".lower()
        if preferred in files:
            return os.path.join(book_dir, files[preferred])
        suffixed = sorted(n for n in files if n.endswith(f"_glossary{ext}"))
        if suffixed:
            return os.path.join(book_dir, files[suffixed[0]])
        if f"glossary{ext}" in files:
            return os.path.join(book_dir, files[f"glossary{ext}"])
    return None


def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def iter_book_glossary_files(shared_dir):
    """Every per-book glossary under ``shared_dir``, newest first.

    Only direct children are book folders. The unified folder and the legacy
    shared Backups folder are never inputs, and sidecars (progress, gender
    tracker, history, extension, unified copies) are never picked.
    """
    root = os.path.abspath(shared_dir or "")
    if not root or not os.path.isdir(root):
        return []
    found = []
    for name in os.listdir(root):
        if name.strip().casefold() in _SKIP_DIR_NAMES:
            continue
        book_dir = os.path.join(root, name)
        if not os.path.isdir(book_dir):
            continue
        path = _pick_book_glossary_file(book_dir, name)
        if path:
            found.append(path)
    found.sort(key=_safe_mtime, reverse=True)
    return found


def book_glossary_file(book_path):
    """The on-disk file for a book glossary given either its .json or .csv path."""
    if not book_path:
        return None
    stem, _ext = os.path.splitext(str(book_path))
    for candidate in (str(book_path), stem + ".csv", stem + ".json"):
        if os.path.isfile(candidate):
            return candidate
    return None


def worker_count(n_items):
    """Bounded by EXTRACTION_WORKERS, which Other Settings sets to 1 when
    Parallel Processing is off."""
    try:
        workers = int(str(os.getenv("EXTRACTION_WORKERS", "1")).strip() or "1")
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(max(1, int(n_items or 0)), workers))


# ─── Fingerprints and state ──────────────────────────────────────────────────

def _path_key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _stat_fingerprint(path):
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _signature(key, combine_all):
    """Dedup settings that change the result; a change forces a rebuild."""
    return {
        "key": key,
        "combine_all": bool(combine_all),
        "fuzzy_threshold": str(os.getenv("GLOSSARY_FUZZY_THRESHOLD", "0.9")),
        "algorithm": str(os.getenv("GLOSSARY_DUPLICATE_ALGORITHM", "auto")).lower(),
        "dedupe_translations": str(os.getenv("GLOSSARY_DEDUPE_TRANSLATIONS", "1")),
        "legacy_csv": str(os.getenv("GLOSSARY_USE_LEGACY_CSV", "0")),
    }


def _load_state(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return {}
    return data


def _save_state(state_path, state):
    state = dict(state)
    state["version"] = STATE_VERSION
    folder = os.path.dirname(state_path) or "."
    os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=folder, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        temp_path = handle.name
    _extractor()._atomic_replace_file(temp_path, state_path)


# ─── Entries ─────────────────────────────────────────────────────────────────

def load_entries(path):
    if not path or not os.path.exists(path):
        return []
    entries = _extractor()._load_glossary_file(path, quiet=True)
    return [entry for entry in entries if isinstance(entry, dict)]


def _raw_key(entry):
    raw = str(entry.get("raw_name") or "")
    return unicodedata.normalize("NFC", raw).strip().casefold()


def _strip_non_shareable(entries):
    """Drop per-novel title rows and rows with no raw name."""
    kept = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type") or "").strip().lower() in ("book", "books"):
            continue
        if not _raw_key(entry):
            continue
        kept.append(entry)
    return kept


def _dedupe(entries):
    if not entries:
        return []
    return _extractor().skip_duplicate_entries(list(entries), glossary_path=None)


def _write(entries, json_path):
    extractor = _extractor()
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with extractor._environment_overrides(_WRITE_OVERRIDES):
        extractor.save_glossary_json(entries, json_path)
        extractor.save_glossary_csv(entries, json_path)


def _remove_quiet(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
            return True
    except OSError:
        pass
    return False


# ─── Rebuild / merge / per-book copy ─────────────────────────────────────────

def rebuild(shared_dir, key, current_entries=None, current_path=None, log=print):
    """Rebuild every unified glossary from all book folders.

    Returns False when the fingerprint state says nothing changed. The
    current book's in-memory entries are used in place of its file and are
    placed first so they win raw-name conflicts; other books follow newest
    file first.
    """
    combine_all = combine_all_enabled()
    target = target_language()
    files = iter_book_glossary_files(shared_dir)
    inputs = {_path_key(path): _stat_fingerprint(path) for path in files}
    _folder, _json_path, csv_path, state_path = unified_paths(shared_dir, key)
    state = _load_state(state_path)
    signature = _signature(key, combine_all)
    if (
        state.get("signature") == signature
        and state.get("inputs") == inputs
        and state.get("output") == _stat_fingerprint(csv_path)
    ):
        log(f"📚 Unified glossary up to date ({len(files)} book glossaries) — skipping rebuild")
        return False

    workers = worker_count(len(files))
    log(
        f"📚 Unified glossary: rebuilding from {len(files)} book glossaries "
        f"under {os.path.basename(os.path.abspath(shared_dir))}/ (workers={workers})…"
    )
    current_keys = set()
    if current_path:
        stem = os.path.splitext(str(current_path))[0]
        current_keys = {_path_key(current_path), _path_key(stem + ".csv"), _path_key(stem + ".json")}

    def _load(path):
        entries = _strip_non_shareable(load_entries(path))
        return path, entries, detect_entries_language(entries)

    loaded = []
    if files:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(_load, files))

    groups = {}
    book_counts = {}
    current = _strip_non_shareable(current_entries or [])
    if current:
        groups.setdefault(key, []).extend(current)
        book_counts[key] = book_counts.get(key, 0) + 1
    for path, entries, language in loaded:
        if _path_key(path) in current_keys:
            continue  # the in-memory copy above is authoritative
        if not entries:
            continue
        group_key = folder_key("all", target) if combine_all else folder_key(language, target)
        groups.setdefault(group_key, []).extend(entries)
        book_counts[group_key] = book_counts.get(group_key, 0) + 1

    written = {}
    for group_key, entries in groups.items():
        # The dedup pass prints its own "[Dedup] …" lines; say whose they are
        # first, or a 30k-entry pass looks like it belongs to the book.
        log(
            f"📚 Unified glossary [{group_key}]: deduplicating {len(entries):,} entries "
            f"combined from {book_counts.get(group_key, 0)} book glossaries…"
        )
        deduped = _dedupe(entries)
        if not deduped:
            continue
        _g_folder, g_json, g_csv, g_state = unified_paths(shared_dir, group_key)
        with _LOCK:
            _write(deduped, g_json)
        _save_state(g_state, {
            "signature": _signature(group_key, combine_all),
            "inputs": inputs,
            "output": _stat_fingerprint(g_csv),
            "mirrors": {},
        })
        written[group_key] = len(deduped)
        log(
            f"📚 Unified glossary [{group_key}] rebuilt: {book_counts.get(group_key, 0)} book glossaries, "
            f"{len(entries):,} → {len(deduped):,} entries → {g_csv}"
        )
    if key not in written:
        # Nothing in this language yet; record the inputs so the next run
        # does not re-read every file just to learn that again.
        _save_state(state_path, {
            "signature": signature, "inputs": inputs,
            "output": _stat_fingerprint(csv_path), "mirrors": state.get("mirrors") or {},
        })
    return True


def merge_book(shared_dir, key, book_entries, book_path=None, log=print):
    """Merge one book into its unified glossary; skipped when unchanged."""
    _folder, json_path, csv_path, state_path = unified_paths(shared_dir, key)
    state = _load_state(state_path)
    on_disk = book_glossary_file(book_path)
    book_key = _path_key(on_disk) if on_disk else None
    book_fp = _stat_fingerprint(on_disk) if on_disk else None
    signature = _signature(key, combine_all_enabled())
    inputs = dict(state.get("inputs") or {})
    if (
        book_key
        and book_fp is not None
        and inputs.get(book_key) == book_fp
        and state.get("signature") == signature
        and state.get("output") == _stat_fingerprint(csv_path)
    ):
        log("📚 Unified glossary: this book is already merged — skipping")
        return False

    book = _strip_non_shareable(book_entries or [])
    if not book:
        return False
    existing = _strip_non_shareable(load_entries(csv_path))
    log(
        f"📚 Unified glossary [{key}]: merging {len(book):,} entries from this book into "
        f"{len(existing):,} existing unified entries (deduplicating {len(book) + len(existing):,})…"
    )
    deduped = _dedupe(book + existing)
    if not deduped:
        return False
    with _LOCK:
        _write(deduped, json_path)
    if book_key:
        inputs[book_key] = book_fp
    state.update({
        "signature": signature,
        "inputs": inputs,
        "output": _stat_fingerprint(csv_path),
        "mirrors": state.get("mirrors") or {},
    })
    _save_state(state_path, state)
    log(
        f"📚 Unified glossary [{key}] updated: {len(book):,} entries from this book → "
        f"{len(deduped):,} total → {csv_path}"
    )
    return True


def write_book_copy(shared_dir, key, book_entries, target_dir, book_path=None, log=print):
    """Write ``<target_dir>/glossary_unified.csv`` = unified minus this book.

    The copy is what gets appended to the prompt beside the book glossary, so
    entries the book glossary already carries are left out. When nothing
    remains the stale copy is removed so an empty file is never appended.
    """
    if not target_dir:
        return False
    folder, _json_path, csv_path, state_path = unified_paths(shared_dir, key)
    target_json = os.path.join(target_dir, UNIFIED_BASENAME + ".json")
    target_csv = os.path.join(target_dir, UNIFIED_BASENAME + ".csv")
    if _path_key(target_dir) == _path_key(folder):
        return False
    output_fp = _stat_fingerprint(csv_path)
    if output_fp is None:
        _remove_quiet(target_csv)
        _remove_quiet(target_json)
        return False

    book = _strip_non_shareable(book_entries or [])
    on_disk = book_glossary_file(book_path)
    marker = {
        "output": output_fp,
        "book": _stat_fingerprint(on_disk) if on_disk else None,
        "book_count": len(book),
    }
    state = _load_state(state_path)
    mirrors = dict(state.get("mirrors") or {})
    target_key = _path_key(target_dir)
    previous = mirrors.get(target_key) or {}
    if previous.get("marker") == marker and bool(previous.get("written")) == os.path.isfile(target_csv):
        return False

    own = {_raw_key(entry) for entry in book}
    remaining = [
        entry for entry in _strip_non_shareable(load_entries(csv_path))
        if _raw_key(entry) not in own
    ]
    if remaining:
        os.makedirs(target_dir, exist_ok=True)
        with _LOCK:
            _write(remaining, target_json)
        written = True
        log(f"📚 Unified glossary copy: {len(remaining)} cross-novel entries → {target_csv}")
    else:
        _remove_quiet(target_csv)
        _remove_quiet(target_json)
        written = False
    mirrors[target_key] = {"marker": marker, "written": written}
    state["mirrors"] = mirrors
    _save_state(state_path, state)
    return written


def sync_phase(
    stage, book_entries, book_glossary_path, target_dirs,
    chapters=None, log=print, settings=None, merge=True,
):
    """Entry point for the start/end-of-phase hooks. Never raises.

    ``stage`` is ``"start"`` or ``"end"``. With Generate Unified Glossary on,
    the start hook rebuilds from every book folder (fingerprint-guarded);
    otherwise the book is merged incrementally. Every ``target_dir`` then
    gets the per-book copy. ``merge=False`` only refreshes the copies, for
    callers whose merge already happened elsewhere.
    """
    if not enabled(settings):
        return None
    try:
        shared_dir = shared_glossary_dir(settings, fallback_base=book_glossary_path)
        source = resolve_source_language(
            entries=book_entries, chapters=chapters, settings=settings,
        )
        key = folder_key(source, target_language(settings))
        os.environ["UNIFIED_GLOSSARY_RESOLVED_KEY"] = key
        if merge:
            if stage == "start" and generate_enabled(settings):
                rebuild(
                    shared_dir, key,
                    current_entries=book_entries, current_path=book_glossary_path,
                    log=log,
                )
            else:
                merge_book(shared_dir, key, book_entries, book_path=book_glossary_path, log=log)
        for target_dir in target_dirs or []:
            if target_dir:
                write_book_copy(
                    shared_dir, key, book_entries, target_dir,
                    book_path=book_glossary_path, log=log,
                )
        return key
    except Exception as exc:
        log(f"⚠️ Unified glossary ({stage}): {exc} — continuing")
        return None


def resolve_prompt_glossary_path(actual_glossary_path=None, settings=None):
    """The unified file to append to a request, or None.

    Prefers the per-book copy beside the glossary being sent (unified minus
    that book's own entries); falls back to the canonical file.
    """
    if actual_glossary_path:
        beside = os.path.join(
            os.path.dirname(os.path.abspath(actual_glossary_path)),
            UNIFIED_BASENAME + ".csv",
        )
        if os.path.isfile(beside):
            return beside
    try:
        shared_dir = shared_glossary_dir(settings, fallback_base=actual_glossary_path)
        key = str(_setting(settings, "UNIFIED_GLOSSARY_RESOLVED_KEY", "") or "").strip()
        if not key:
            key = folder_key(resolve_source_language(settings=settings), target_language(settings))
        csv_path = unified_paths(shared_dir, key)[2]
    except Exception:
        return None
    return csv_path if os.path.isfile(csv_path) else None


def count_entries(path):
    """Entry count for log summaries; 0 when unreadable."""
    try:
        return len(load_entries(path))
    except Exception:
        return 0
