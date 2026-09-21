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

import hashlib
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
    # The writer re-runs alias harmonisation, a pure-Python all-pairs loop:
    # fine for a book, minutes of frozen GUI for a 30k-entry unified list.
    "GLOSSARY_ALIAS_AWARE_NAME_MATCHING": "0",
}
# Alias-aware matching aligns names that contain one another *within a book*.
# Across novels it is both meaningless and an all-pairs Python loop.
_DEDUPE_OVERRIDES = {
    "GLOSSARY_ALIAS_AWARE_NAME_MATCHING": "0",
}
# Lists at least this large are deduplicated in a child process.
_SUBPROCESS_MIN_ENTRIES_DEFAULT = 2000

_LOCK = threading.RLock()
_THREAD_STATE = threading.local()
# One manual rebuild at a time (the Rebuild Now button).
_REBUILD_NOW_LOCK = threading.Lock()


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


def _entries_digest(entries):
    """Order-independent digest of what a glossary contributes.

    Book-title rows, empty values and private keys are left out, so the
    digest only moves when something that would reach the unified glossary
    moves.
    """
    rows = []
    for entry in _strip_non_shareable(entries):
        rows.append(sorted(
            (str(name), str(value)) for name, value in entry.items()
            if value not in (None, "") and not str(name).startswith("_")
        ))
    rows.sort()
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_fingerprint(path, previous=None, parsed=None):
    """``{size, mtime_ns, digest}`` for a book glossary file.

    The extractor re-saves a book's glossary at the end of every run whether
    or not anything changed, so size/mtime alone would report a change each
    time. They are only the fast path: while they match the recorded values
    the recorded digest is reused without opening the file; once they differ
    the file is parsed and the *content* digest decides. Parsed entries are
    left in ``parsed`` so a caller that goes on to merge does not read twice.
    """
    stat = _stat_fingerprint(path)
    if stat is None:
        return None
    if (
        isinstance(previous, dict)
        and previous.get("digest")
        and previous.get("size") == stat["size"]
        and previous.get("mtime_ns") == stat["mtime_ns"]
    ):
        return {**stat, "digest": previous["digest"]}
    entries = _strip_non_shareable(load_entries(path))
    if parsed is not None:
        parsed[_path_key(path)] = entries
    return {**stat, "digest": _entries_digest(entries)}


def _same_input(recorded, current):
    """Same content? Falls back to size/mtime for states written before digests."""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    if recorded.get("digest") and current.get("digest"):
        return recorded["digest"] == current["digest"]
    return (
        recorded.get("size") == current.get("size")
        and recorded.get("mtime_ns") == current.get("mtime_ns")
    )


def _inputs_match(recorded, current):
    recorded = recorded or {}
    current = current or {}
    return set(recorded) == set(current) and all(
        _same_input(recorded[name], current[name]) for name in current
    )


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


# ─── Deduplication ───────────────────────────────────────────────────────────
#
# A book glossary is a few hundred entries and dedups in well under a second.
# A unified glossary is tens of thousands: every entry is fuzzy-searched
# against everything kept so far, which is minutes of work that holds
# Python's interpreter lock. Run on a worker *thread* that starves the GUI
# thread and the whole window stutters. So:
#
#   * large jobs run in a child process (same pattern as the Minimal glossary
#     worker), leaving the GUI process idle;
#   * the per-run merge never re-deduplicates the unified list. It is already
#     deduplicated, so only the book's entries are matched against it.

class _DedupeStopped(Exception):
    """The user pressed Stop while a background dedup was running."""


def _subprocess_min_entries():
    try:
        return max(0, int(os.getenv(
            "UNIFIED_GLOSSARY_SUBPROCESS_MIN_ENTRIES", str(_SUBPROCESS_MIN_ENTRIES_DEFAULT),
        )))
    except (TypeError, ValueError):
        return _SUBPROCESS_MIN_ENTRIES_DEFAULT


def _full_dedupe(entries):
    """The regular two-pass dedup over a whole list."""
    if not entries:
        return []
    return _extractor().skip_duplicate_entries(list(entries), glossary_path=None)


def _incremental_merge(book, existing):
    """Merge ``book`` into ``existing`` without re-comparing ``existing``.

    ``existing`` came out of a previous dedup, so its entries are already
    distinct from one another. Only the book's entries need a fuzzy search,
    which turns a (book + unified)² job into a book × unified one. A book
    entry that matches replaces the unified row (the book being worked on
    wins); one that matches nothing is appended.
    """
    extractor = _extractor()
    book = [entry for entry in (book or []) if isinstance(entry, dict)]
    existing = [entry for entry in (existing or []) if isinstance(entry, dict)]
    if not book:
        return list(existing)
    book = extractor.skip_duplicate_entries(list(book), glossary_path=None)
    if not existing:
        return book

    threshold = float(os.getenv("GLOSSARY_FUZZY_THRESHOLD", "0.9"))
    try:
        import rapidfuzz  # noqa: F401
        use_rapidfuzz = True
    except Exception:
        use_rapidfuzz = False
    config = config_no_partial = partial_gender_only = None
    try:
        from duplicate_detection_config import get_duplicate_detection_config
        config = get_duplicate_detection_config()
        config_no_partial = dict(config)
        config_no_partial["algorithms"] = [
            name for name in config.get("algorithms", []) if name != "partial"
        ]
        partial_gender_only = extractor._partial_ratio_gender_only()
    except Exception:
        pass

    seen, seen_lower, index_by_raw, index_by_key = [], [], {}, {}
    for idx, entry in enumerate(existing):
        raw = str(entry.get("raw_name") or "")
        cleaned = unicodedata.normalize("NFC", extractor.remove_honorifics(raw) or "")
        seen.append((cleaned, raw, entry))
        seen_lower.append(cleaned.lower())
        index_by_raw.setdefault(raw, idx)
        index_by_key.setdefault(_raw_key(entry), idx)

    merged = list(existing)
    added = []
    replaced = 0
    for entry in book:
        raw = str(entry.get("raw_name") or "")
        idx = index_by_key.get(_raw_key(entry))
        if idx is None:
            cleaned = unicodedata.normalize("NFC", extractor.remove_honorifics(raw) or "")
            is_duplicate, _score, best_raw = extractor._find_best_duplicate_match(
                cleaned, seen, threshold, use_rapidfuzz, entry,
                _config=config, _partial_gender_only=partial_gender_only,
                _config_no_partial=config_no_partial, _seen_lower_names=seen_lower,
            )
            if is_duplicate:
                idx = index_by_raw.get(best_raw)
        if idx is None:
            added.append(entry)
        else:
            merged[idx] = entry
            replaced += 1
    print(
        f"[Dedup] Unified merge: {len(book):,} book entries → {replaced:,} replaced an existing "
        f"row, {len(added):,} added ({len(existing):,} unified entries were not re-compared)"
    )
    return merged + added


def _dedupe_job(kind, first, second, env_vars):
    """Child-process entry point. Returns (entries, captured log lines)."""
    import contextlib
    import io

    os.environ.update(env_vars or {})
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if kind == "merge":
            result = _incremental_merge(first, second)
        else:
            result = _full_dedupe(first)
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    return result, lines


def _stop_requested():
    # "Rebuild Now" runs outside any translation/extraction run, where the
    # run-level stop flags may still be set from an earlier cancelled run.
    if getattr(_THREAD_STATE, "ignore_stop", False):
        return False
    try:
        if os.environ.get("TRANSLATION_CANCELLED") == "1":
            return True
        return bool(_extractor().is_stop_requested())
    except Exception:
        return False


def _run_dedupe_in_subprocess(kind, first, second, total, log):
    import concurrent.futures
    import multiprocessing
    import time

    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    env_vars = {name: value for name, value in os.environ.items() if name.startswith("GLOSSARY_")}
    env_vars.update(_DEDUPE_OVERRIDES)
    log(
        f"📚 Unified glossary: deduplicating {total:,} entries in a separate process "
        "so the GUI stays responsive…"
    )
    started = last_beat = time.time()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_dedupe_job, kind, first, second, env_vars)
        while True:
            try:
                result, lines = future.result(timeout=0.5)
                break
            except concurrent.futures.TimeoutError:
                if _stop_requested():
                    for process in list((getattr(executor, "_processes", None) or {}).values()):
                        try:
                            process.terminate()
                        except Exception:
                            pass
                    raise _DedupeStopped("stopped by user")
                now = time.time()
                if now - last_beat >= 20:
                    last_beat = now
                    log(f"📚 Unified glossary: still deduplicating {total:,} entries… ({int(now - started)}s)")
        for line in lines[:40]:
            log(line)
        log(f"📚 Unified glossary: dedup finished in {time.time() - started:.0f}s")
        return result
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_dedupe(kind, first, second=None, log=print):
    """Dedup ``first`` ("full") or merge ``first`` into ``second`` ("merge")."""
    first = list(first or [])
    second = list(second or [])
    total = len(first) + len(second)
    if not total:
        return []
    if total >= _subprocess_min_entries():
        try:
            return _run_dedupe_in_subprocess(kind, first, second, total, log)
        except _DedupeStopped:
            raise
        except Exception as exc:  # incl. BrokenProcessPool: fall back in-process
            log(
                f"⚠️ Unified glossary: background dedup process failed ({exc}); "
                "running in-process instead (the GUI may lag until it finishes)"
            )
    with _extractor()._environment_overrides(_DEDUPE_OVERRIDES):
        if kind == "merge":
            return _incremental_merge(first, second)
        return _full_dedupe(first)


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

def rebuild(shared_dir, key, current_entries=None, current_path=None, log=print,
            force=False, settings=None):
    """Bring every unified glossary up to date with all book folders.

    ``force`` skips the fingerprint shortcuts and always rebuilds from
    scratch. ``key`` may be None when there is no current book (the Rebuild
    Now button): every language group is still written, there is just no
    book whose entries get to go first.


    Three outcomes, cheapest first:
      * nothing changed since the last build -> returns False, stat calls only;
      * a valid unified file and only a few changed books -> just those books
        are merged in (``_incremental_merge``), seconds even at 30k entries;
      * otherwise a full rebuild: every book is read (in parallel) and each
        language group is deduplicated from scratch, in a child process when
        large. The current book's in-memory entries go first so they win
        raw-name conflicts; other books follow newest file first.
    """
    combine_all = combine_all_enabled(settings)
    target = target_language(settings)
    files = iter_book_glossary_files(shared_dir)
    if key:
        _folder, _json_path, csv_path, state_path = unified_paths(shared_dir, key)
        state = _load_state(state_path)
        signature = _signature(key, combine_all)
    else:
        csv_path = state_path = signature = None
        state = {}
        force = True
    previous_inputs = state.get("inputs") or {}

    # Content fingerprints. Files whose size/mtime still match the recorded
    # values are not opened; the rest are parsed once and kept in ``parsed``.
    parsed = {}

    def _fingerprint(path):
        return _path_key(path), _input_fingerprint(path, previous_inputs.get(_path_key(path)), parsed)

    if len(files) > 1:
        with ThreadPoolExecutor(max_workers=worker_count(len(files))) as pool:
            inputs = dict(pool.map(_fingerprint, files))
    else:
        inputs = dict(_fingerprint(path) for path in files)

    if (
        not force
        and state.get("signature") == signature
        and _inputs_match(previous_inputs, inputs)
        and state.get("output") == _stat_fingerprint(csv_path)
    ):
        if previous_inputs != inputs:
            # Same content, newer timestamps: remember them so the next run
            # takes the no-read fast path again.
            state["inputs"] = inputs
            _save_state(state_path, state)
        log(
            f"📚 Unified glossary [{key}] is up to date: none of the {len(files)} book glossaries "
            "changed since the last build — skipping rebuild"
        )
        return False

    current_keys = set()
    if current_path:
        stem = os.path.splitext(str(current_path))[0]
        current_keys = {_path_key(current_path), _path_key(stem + ".csv"), _path_key(stem + ".json")}
    current = _strip_non_shareable(current_entries or [])

    def _load(path):
        if _path_key(path) in current_keys and current:
            entries = current  # the in-memory copy is authoritative
        elif _path_key(path) in parsed:
            entries = parsed[_path_key(path)]
        else:
            entries = _strip_non_shareable(load_entries(path))
        return path, entries, detect_entries_language(entries)

    # A unified file that is still valid only needs the books that changed
    # since it was written merged in, which is cheap. A full rebuild is for
    # the first build, a changed dedup setting, or a missing/edited output.
    output_fp = _stat_fingerprint(csv_path)
    changed = [
        path for path in files
        if not _same_input(previous_inputs.get(_path_key(path)), inputs[_path_key(path)])
    ]
    if (
        not force
        and previous_inputs
        and output_fp is not None
        and state.get("signature") == signature
        and state.get("output") == output_fp
        and len(changed) <= max(3, len(files) // 4)
    ):
        workers = worker_count(len(changed))
        log(
            f"📚 Unified glossary: {len(changed)} of {len(files)} book glossaries changed "
            f"since the last build — merging only those (workers={workers})"
        )
        loaded_changed = []
        if changed:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                loaded_changed = list(pool.map(_load, changed))
        by_group = {}
        for path, entries, language in loaded_changed:
            if not entries:
                continue
            if _path_key(path) in current_keys:
                group_key = key
            else:
                group_key = folder_key("all", target) if combine_all else folder_key(language, target)
            by_group.setdefault(group_key, []).extend(entries)
        for group_key, entries in by_group.items():
            _g_folder, g_json, g_csv, g_state = unified_paths(shared_dir, group_key)
            existing = _strip_non_shareable(load_entries(g_csv))
            log(
                f"📚 Unified glossary [{group_key}]: merging {len(entries):,} entries from changed "
                f"books into {len(existing):,} existing unified entries…"
            )
            merged = _run_dedupe("merge", entries, existing, log=log)
            if not merged:
                continue
            with _LOCK:
                _write(merged, g_json)
            group_state = _load_state(g_state)
            group_state.update({
                "signature": _signature(group_key, combine_all),
                "inputs": inputs,
                "output": _stat_fingerprint(g_csv),
                "mirrors": group_state.get("mirrors") or {},
            })
            _save_state(g_state, group_state)
            log(f"📚 Unified glossary [{group_key}] updated: {len(merged):,} entries → {g_csv}")
        if key not in by_group:
            state.update({"inputs": inputs, "output": _stat_fingerprint(csv_path)})
            _save_state(state_path, state)
        return True

    workers = worker_count(len(files))
    log(
        f"📚 Unified glossary: rebuilding from {len(files)} book glossaries "
        f"under {os.path.basename(os.path.abspath(shared_dir))}/ (workers={workers})…"
    )
    loaded = []
    if files:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(_load, files))

    groups = {}
    book_counts = {}
    if current:
        groups.setdefault(key, []).extend(current)
        book_counts[key] = book_counts.get(key, 0) + 1
    for path, entries, language in loaded:
        if current and _path_key(path) in current_keys:
            continue  # already added above, first, so it wins conflicts
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
        deduped = _run_dedupe("full", entries, log=log)
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
    if key and key not in written:
        # Nothing in this language yet; record the inputs so the next run
        # does not re-read every file just to learn that again.
        _save_state(state_path, {
            "signature": signature, "inputs": inputs,
            "output": _stat_fingerprint(csv_path), "mirrors": state.get("mirrors") or {},
        })
    if not written:
        log(f"📚 Unified glossary: nothing to build — no entries found in {len(files)} book glossaries")
    return True


def rebuild_now(shared_dir=None, settings=None, log=print):
    """Forced full rebuild with no current book (the Rebuild Now button).

    Returns True when it ran, False when another manual rebuild is already
    running. Never raises. Run-level stop flags are ignored: this is not part
    of a run, and a flag left over from a cancelled one must not abort it.
    """
    if not _REBUILD_NOW_LOCK.acquire(blocking=False):
        log("📚 Unified glossary: a rebuild is already running")
        return False
    _THREAD_STATE.ignore_stop = True
    try:
        root = os.path.abspath(shared_dir) if shared_dir else shared_glossary_dir(settings)
        if not os.path.isdir(root):
            log(f"📚 Unified glossary: no Glossary folder at {root} — nothing to rebuild")
            return True
        rebuild(root, None, log=log, force=True, settings=settings)
        log("📚 Unified glossary: rebuild finished")
        return True
    except Exception as exc:
        log(f"⚠️ Unified glossary rebuild failed: {exc}")
        return True
    finally:
        _THREAD_STATE.ignore_stop = False
        _REBUILD_NOW_LOCK.release()


def merge_book(shared_dir, key, book_entries, book_path=None, log=print):
    """Merge one book into its unified glossary; skipped when unchanged."""
    _folder, json_path, csv_path, state_path = unified_paths(shared_dir, key)
    state = _load_state(state_path)
    on_disk = book_glossary_file(book_path)
    book_key = _path_key(on_disk) if on_disk else None
    signature = _signature(key, combine_all_enabled())
    inputs = dict(state.get("inputs") or {})
    book_fp = _input_fingerprint(on_disk, inputs.get(book_key)) if on_disk else None
    if (
        book_key
        and book_fp is not None
        and _same_input(inputs.get(book_key), book_fp)
        and state.get("signature") == signature
        and state.get("output") == _stat_fingerprint(csv_path)
    ):
        if inputs.get(book_key) != book_fp:
            # Re-saved with identical content: keep the new timestamps so the
            # next check does not have to open the file.
            inputs[book_key] = book_fp
            state["inputs"] = inputs
            _save_state(state_path, state)
        log(
            f"📚 Unified glossary [{key}]: this book's glossary is unchanged since it was "
            "last merged (fingerprint match) — skipping merge"
        )
        return False

    book = _strip_non_shareable(book_entries or [])
    if not book:
        return False
    existing = _strip_non_shareable(load_entries(csv_path))
    log(
        f"📚 Unified glossary [{key}]: merging {len(book):,} entries from this book into "
        f"{len(existing):,} existing unified entries…"
    )
    deduped = _run_dedupe("merge", book, existing, log=log)
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
    state = _load_state(state_path)
    # Content digest, not mtime: the book file is re-saved every run, and an
    # identical re-save must not make this rewrite a 30k-entry copy.
    book_digest = None
    if on_disk:
        recorded = (state.get("inputs") or {}).get(_path_key(on_disk))
        book_digest = (_input_fingerprint(on_disk, recorded) or {}).get("digest")
    marker = {
        "output": output_fp,
        "book": book_digest or _entries_digest(book),
        "book_count": len(book),
    }
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
                    log=log, settings=settings,
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
