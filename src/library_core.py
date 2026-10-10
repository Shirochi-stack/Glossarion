"""library_core: GUI-free Library registry and source-EPUB resolver (moved verbatim from epub_library).

Shared GUI-free core (Glossarion mobile rewrite). ``epub_library`` imports PySide6 at
module scope, so code that runs without Qt (``run_env._build_epub_compile_env`` on
Glossarion Mobile) could not resolve a workspace's source EPUB and compiled with
filename ordering. These functions moved here byte-for-byte; ``epub_library``
re-imports every name, so desktop callers get the same function objects:

* the Library folders (``get_library_dir`` with the ``GLOSSARION_LIBRARY_DIR`` seam,
  ``get_library_raw_dir``, ``get_library_translated_dir``) and the one-time legacy
  layout migration;
* the raw / translated input registries and the ``library_origins.txt`` mapping;
* the source-EPUB resolver ``_find_raw_source_for_folder`` with
  ``_read_source_epub_pointer``, ``_origins_raw_sources_for_stem``,
  ``_validate_source_epub_for_workspace`` and the helpers ``_special_file_stem`` /
  ``_norm_book_key``.
* (U3) ``_read_progress_summary`` (the Library card's translation_progress.json
  counts, also read by ``job_runner.ProgressWatcher``) with the special-file rules
  (``_is_configured_special_file`` and its config/env resolvers), ``_is_gallery_filename``
  and ``_is_progress_sidecar_entry``.

* (U5) the rest of the Library: output roots and workspace resolvers, both scans
  (``scan_output_folders`` / ``scan_library_completed``) and their merge
  (``DualScanMixin``), search / sort / format / paging, card data
  (``_card_progress_view``, badges), Book Details (``BookDetailsLoaderMixin``,
  ``BookDetailsMixin``, ``_parse_epub_details``, chapter row specs, metadata edits),
  the Library actions (``LibraryShelfMixin``: import, Organize / Undo, Delete, Clear
  raw link, output-root check), Scan for Raw (``RawScanMixin`` /
  ``ScanForRawMixin``) and the single-chapter progress helpers. Mixins hold
  methods moved byte-for-byte out of the Qt dialogs / threads, which now inherit
  them (listed first in their bases); plain objects (``LibraryShelf``,
  ``BookDetailsModel``, ``RawScanSession``) and functions below serve callers
  without Qt. ``LibraryEnv`` / ``install_library_env`` pin the Library, output root
  and caches for Glossarion Mobile (desktop never installs one).

Log records keep the ``epub_library`` logger name so desktop log routing is
unchanged. Tests that redirect the Library patch these names here (the moved
functions resolve each other through this module).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import copy
import hashlib
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from chapter_chunk_progress import (
    chunk_failure_summary,
    chunk_status_summary_text,
    effective_parent_status,
    ensure_chunk_entry_schema,
    is_multi_chunk_entry,
    reset_chunks_for_retranslation,
    sorted_chunk_items,
)
from epub_package import find_epub_opf_member
from library_covers import _extract_cover, _find_cover_in_dir
from metadata_progress import is_metadata_progress_entry
from translation_artifacts import is_translation_artifact_progress_entry

# Same logger as before the move (records keep the "epub_library" name).
logger = logging.getLogger("epub_library")

#: The process's installed ``LibraryEnv`` (U5; None on desktop). Read by the
#: ``_default_output_root`` seam; see :func:`install_library_env`.
_LIBRARY_ENV = None


def _special_file_stem(name: str) -> str:
    """Normalize a source/response filename to a comparable lowercase stem."""
    if not name:
        return ""
    base = os.path.basename(str(name)).lower()
    if base.startswith("response_"):
        base = base[len("response_"):]
    html_like_exts = {".html", ".xhtml", ".htm", ".txt", ".xml"}
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() not in html_like_exts:
            break
        base = stem
    return base


def get_library_dir() -> str:
    """Return the dedicated Glossarion Library folder (root).

    The library is organized into two subfolders:
      * ``Raw/``        — curated raw source EPUBs the user has imported.
      * ``Translated/`` — curated compiled EPUBs (finished translations).

    ``GLOSSARION_LIBRARY_DIR`` overrides the location (set only by Glossarion
    Mobile, whose HOME is redirected into app storage); desktop never sets it.
    """
    _library_override = os.environ.get("GLOSSARION_LIBRARY_DIR", "").strip()
    if _library_override:
        docs = Path(_library_override)
    else:
        docs = Path.home() / "Documents" / "Glossarion" / "Library"
    try:
        docs.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(docs)


def get_library_raw_dir() -> str:
    """Return ``Library/Raw`` — home for imported raw source EPUBs."""
    raw = Path(get_library_dir()) / "Raw"
    try:
        raw.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(raw)


def get_library_translated_dir() -> str:
    """Return ``Library/Translated`` — home for curated compiled EPUBs."""
    trans = Path(get_library_dir()) / "Translated"
    try:
        trans.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(trans)


def _migrate_legacy_library_layout() -> None:
    """One-time migration: move any EPUBs sitting in ``Library/`` root into
    ``Library/Translated/``.

    Older builds stored organized EPUBs directly under ``Library/``. The
    new layout reserves that folder for the ``Raw/`` and ``Translated/``
    subfolders plus the origins registry. Running this on every library
    scan is idempotent — once there are no root-level EPUBs left, it's a
    no-op. Each moved file is recorded in ``origins['translated']`` so the
    user can reverse it via the Undo Move button.
    """
    lib = get_library_dir()
    trans = get_library_translated_dir()
    try:
        entries = list(os.scandir(lib))
    except (PermissionError, OSError):
        return
    origins = _load_origins()
    trans_origins = dict(origins.get("translated", {}) or {})
    moved_any = False
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        if not entry.name.lower().endswith(".epub"):
            continue  # keep origins file + other stray text files intact
        src = entry.path
        dst = os.path.join(trans, entry.name)
        if os.path.normcase(os.path.normpath(os.path.abspath(src))) == \
                os.path.normcase(os.path.normpath(os.path.abspath(dst))):
            continue
        if os.path.isfile(dst):
            # Don't clobber an existing Translated copy.
            logger.debug("Legacy migration skipped (dst exists): %s", dst)
            continue
        try:
            shutil.move(src, dst)
            trans_origins[entry.name] = os.path.abspath(src)
            moved_any = True
            logger.info("Legacy library migration: %s -> %s", src, dst)
        except OSError as exc:
            logger.debug("Legacy migration move failed for %s: %s", src, exc)
    if moved_any:
        origins["translated"] = trans_origins
        _save_origins(origins)


# Names of tracking / registry files that live alongside the library
# shelves and must NEVER be mistaken for content by the scanners,
# the Undo orphan pass, or the count helpers. Keeping these as plain
# filenames (not paths) lets the same blocklist catch both the new
# ``Library/*.txt`` location and any legacy copy still inside
# ``Library/Raw`` from an earlier build.
_LIBRARY_TRACKING_FILENAMES = frozenset({
    "library_raw_inputs.txt",
    "library_translated_inputs.txt",
    "library_origins.txt",
})


def get_library_raw_inputs_file() -> str:
    """Return ``Library/library_raw_inputs.txt`` — one path per line.

    Tracks every raw input file that has been run through the translator.
    The list is append-only; duplicates are collapsed on read.

    Sits at the Library root (next to ``library_translated_inputs.txt``
    and ``library_origins.txt``) rather than inside ``Library/Raw`` so
    the content-scanning code doesn't mistake the registry for a raw
    source file. A legacy build wrote it into ``Library/Raw``; the
    migration below moves any surviving copy up to the root on first
    access so existing users don't lose their registry state.
    """
    new_path = os.path.join(get_library_dir(), "library_raw_inputs.txt")
    legacy_path = os.path.join(
        get_library_raw_dir(), "library_raw_inputs.txt")
    try:
        legacy_exists = os.path.isfile(legacy_path)
        new_exists = os.path.isfile(new_path)
        if legacy_exists and not new_exists:
            # Promote the legacy copy to the new location.
            try:
                shutil.move(legacy_path, new_path)
            except OSError:
                # Fallback: best-effort copy + remove so the legacy
                # path doesn't keep tripping the content scanners.
                try:
                    shutil.copyfile(legacy_path, new_path)
                    os.remove(legacy_path)
                except OSError:
                    logger.debug(
                        "library_raw_inputs migration failed: %s",
                        traceback.format_exc())
        elif legacy_exists and new_exists:
            # Both exist — merge the legacy entries into the new file
            # then drop the legacy copy so future scans only hit one.
            try:
                merged: list[str] = []
                seen: set[str] = set()
                for src in (new_path, legacy_path):
                    try:
                        with open(src, "r", encoding="utf-8") as f:
                            for line in f:
                                p = line.strip()
                                if not p:
                                    continue
                                key = os.path.normcase(os.path.normpath(
                                    os.path.abspath(p)))
                                if key in seen:
                                    continue
                                seen.add(key)
                                merged.append(p)
                    except OSError:
                        continue
                with open(new_path, "w", encoding="utf-8") as f:
                    for p in merged:
                        f.write(p + "\n")
                try:
                    os.remove(legacy_path)
                except OSError:
                    pass
            except Exception:
                logger.debug(
                    "library_raw_inputs merge failed: %s",
                    traceback.format_exc())
    except Exception:
        logger.debug(
            "library_raw_inputs path resolve failed: %s",
            traceback.format_exc())
    return new_path


def load_library_raw_inputs() -> list[str]:
    """Return the deduplicated list of raw input paths recorded so far."""
    path = get_library_raw_inputs_file()
    if not os.path.isfile(path):
        return []
    seen: set[str] = set()
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    except OSError:
        return []
    return out


def record_library_raw_input(path: str) -> None:
    """Append *path* to the raw-inputs registry (no-op on failure / dup)."""
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    existing = {
        os.path.normcase(os.path.normpath(os.path.abspath(p)))
        for p in load_library_raw_inputs()
    }
    if os.path.normcase(os.path.normpath(abs_path)) in existing:
        return
    reg = get_library_raw_inputs_file()
    try:
        with open(reg, "a", encoding="utf-8") as f:
            f.write(abs_path + "\n")
    except OSError:
        logger.debug("Could not append to %s", reg)


def remove_library_raw_input(path: str) -> None:
    """Drop *path* from the raw-inputs registry.

    Mirrors :func:`remove_library_translated_input` for the raw side
    so the Delete handler can unregister a card whose backing file
    lives outside the Library / output roots: the physical file stays
    where the user put it, but the flash card disappears from the
    scanner output because its registration is gone.
    """
    if not path:
        return
    try:
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except (TypeError, ValueError):
        return
    remaining = [
        p for p in load_library_raw_inputs()
        if os.path.normcase(os.path.normpath(os.path.abspath(p))) != key
    ]
    reg = get_library_raw_inputs_file()
    try:
        with open(reg, "w", encoding="utf-8") as f:
            for p in remaining:
                f.write(p + "\n")
    except OSError:
        logger.debug("Could not rewrite %s", reg)


def get_library_translated_inputs_file() -> str:
    """Return ``Library/library_translated_inputs.txt`` — one path per line.

    Tracks every *compiled* EPUB that has been registered with the
    Library via drag-drop / Import but **not yet physically moved**
    into ``Library/Translated``. Mirrors :func:`get_library_raw_inputs_file`
    for the raw side. The scanner reads this file to surface those
    files on the Completed tab even though they live outside the
    library folder; Organize later moves them and prunes their entry.
    """
    return os.path.join(get_library_dir(), "library_translated_inputs.txt")


def load_library_translated_inputs() -> list[str]:
    """Return the deduplicated list of registered translated paths."""
    path = get_library_translated_inputs_file()
    if not os.path.isfile(path):
        return []
    seen: set[str] = set()
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    except OSError:
        return []
    return out


def record_library_translated_input(path: str) -> None:
    """Append *path* to the translated-inputs registry.

    No-op on failure / duplicate. Mirrors
    :func:`record_library_raw_input` but targets the translated-side
    registry.
    """
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    existing = {
        os.path.normcase(os.path.normpath(os.path.abspath(p)))
        for p in load_library_translated_inputs()
    }
    if os.path.normcase(os.path.normpath(abs_path)) in existing:
        return
    reg = get_library_translated_inputs_file()
    try:
        with open(reg, "a", encoding="utf-8") as f:
            f.write(abs_path + "\n")
    except OSError:
        logger.debug("Could not append to %s", reg)


def remove_library_translated_input(path: str) -> None:
    """Drop *path* from the translated-inputs registry.

    Called when Organize moves the file into ``Library/Translated`` —
    the in-place registration is no longer meaningful once the
    compiled EPUB has been promoted to the curated shelf.
    """
    if not path:
        return
    try:
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except (TypeError, ValueError):
        return
    remaining = [
        p for p in load_library_translated_inputs()
        if os.path.normcase(os.path.normpath(os.path.abspath(p))) != key
    ]
    reg = get_library_translated_inputs_file()
    try:
        with open(reg, "w", encoding="utf-8") as f:
            for p in remaining:
                f.write(p + "\n")
    except OSError:
        logger.debug("Could not rewrite %s", reg)


def _origins_file() -> str:
    """Path to the library origins mapping file."""
    return os.path.join(get_library_dir(), "library_origins.txt")


# Structured origins format (v3):
#   { "version": 3,
#     "raw":        { "<basename in Library/Raw>":        "<absolute original path>" },
#     "translated": { "<basename in Library/Translated>": "<absolute original path>" },
#     "pairs":      { "<basename in Library/Translated>": "<basename in Library/Raw>" } }
#
# The ``pairs`` bucket is a direct translated↔raw link for books whose
# compiled and source EPUBs were organized together. It's consulted by
# :func:`_find_raw_source_for_library_epub` *first* because filename-stem
# matching fails when the raw is named in one language and the compiled
# in another (common for translations) — and the output-folder fallback
# breaks as soon as that folder is deleted or its ``source_epub.txt``
# sidecar drifts out of sync.
#
# Legacy v1 format was a flat ``{basename: original_path}`` dict and is
# promoted into the ``translated`` bucket on read so the user's existing
# undo history is preserved.


def _load_origins() -> dict:
    """Load the structured origins mapping (auto-upgrades the legacy format).

    Also sanitizes the loaded mapping by dropping any entry whose key
    is a library registry filename (``library_raw_inputs.txt``,
    ``library_translated_inputs.txt``, ``library_origins.txt``).
    Legacy builds placed ``library_raw_inputs.txt`` inside ``Library/Raw``
    where the organize scan happily treated it as a raw source and wrote
    a ghost entry into ``origins['raw']``. Undo then reported
    ``raw:library_raw_inputs.txt: not found in Library/Raw`` on every
    restore because the registry file had since been promoted to the
    library root. Filtering here means any pre-existing garbage entry
    is forgotten on first read, and the sanitized dict is flushed back
    to disk so the next load stays clean.
    """
    import json
    try:
        with open(_origins_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": 3, "raw": {}, "translated": {}, "pairs": {}}
    if not isinstance(data, dict):
        return {"version": 3, "raw": {}, "translated": {}, "pairs": {}}
    # Legacy flat mapping → promote to ``translated``.
    if "version" not in data and "raw" not in data and "translated" not in data:
        data = {"version": 3, "raw": {}, "translated": dict(data), "pairs": {}}
    data.setdefault("version", 3)
    data.setdefault("raw", {})
    data.setdefault("translated", {})
    data.setdefault("pairs", {})
    if not isinstance(data["raw"], dict):
        data["raw"] = {}
    if not isinstance(data["translated"], dict):
        data["translated"] = {}
    if not isinstance(data["pairs"], dict):
        data["pairs"] = {}

    # Drop tracking-file ghosts from every bucket. ``pairs`` holds raw
    # basenames as values too, so a legacy entry pointing at the
    # registry file must also be purged there.
    dirty = False
    for bucket in ("raw", "translated"):
        sanitized = {
            k: v for k, v in data[bucket].items()
            if k and k.lower() not in _LIBRARY_TRACKING_FILENAMES
        }
        if len(sanitized) != len(data[bucket]):
            data[bucket] = sanitized
            dirty = True
    sanitized_pairs = {
        k: v for k, v in data["pairs"].items()
        if k and k.lower() not in _LIBRARY_TRACKING_FILENAMES
        and (not isinstance(v, str)
             or v.lower() not in _LIBRARY_TRACKING_FILENAMES)
    }
    if len(sanitized_pairs) != len(data["pairs"]):
        data["pairs"] = sanitized_pairs
        dirty = True
    if dirty:
        try:
            _save_origins(data)
        except Exception:
            logger.debug("origins sanitize-flush failed: %s",
                         traceback.format_exc())
    return data


def _save_origins(origins: dict):
    """Persist the structured origins mapping."""
    import json
    try:
        with open(_origins_file(), "w", encoding="utf-8") as f:
            json.dump(origins, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


# Characters Windows strips / mangles in filenames that the translator
# happily leaves in a book's ``metadata.json`` title. We remove them
# from both sides before comparing so a title like ``"Foo: Bar."`` in
# metadata still matches the on-disk filename ``"Foo Bar.epub"``.
_FILENAME_STRIP_CHARS = '<>:"/\\|?*'


def _norm_book_key(value: str) -> str:
    """Normalize a book title/filename for cross-source comparison.

    The scanner links a Library/Translated EPUB to its originating
    workspace through several potentially-misaligned sources:

      * the workspace folder name (source-language raw stem)
      * the resolved raw source stem (also source-language)
      * the workspace's ``metadata.json`` title fields
        (``title`` is usually the *translated* title after compile)
      * the library EPUB filename stem (translated title, post
        Windows filename sanitization)

    Different call-sites produce slightly different strings for the
    same book — e.g. metadata might hold
    ``"… Romance Fantasy."`` (trailing period) while the on-disk
    EPUB is ``"… Romance Fantasy.epub"`` (period stripped because
    NTFS forbids trailing dots). Normalizing through this helper
    makes those two strings equal so the pairing succeeds.

    Rules:
      * strip characters disallowed in Windows filenames
      * collapse whitespace runs into single spaces
      * strip leading/trailing whitespace AND trailing periods
        (NTFS silently drops trailing ``.``)
      * casefold for Unicode-aware lowercase comparison

    Returns ``""`` when the input doesn't contain anything meaningful
    so callers can skip the key instead of collapsing onto the empty
    string (which would spuriously match unrelated unnamed entries).
    """
    if not value:
        return ""
    import re as _re
    s = str(value)
    # Drop Windows-illegal chars so on-disk filenames and in-memory
    # titles collapse to the same key.
    s = s.translate({ord(c): " " for c in _FILENAME_STRIP_CHARS})
    # Collapse whitespace runs.
    s = _re.sub(r"\s+", " ", s)
    # Drop trailing periods + whitespace (NTFS / FAT strip trailing
    # dots from filenames, so a metadata title ending in ``.`` won't
    # match the on-disk basename unless we normalize it away).
    s = s.strip().rstrip(". \t")
    return s.casefold()


def _read_source_epub_pointer(folder: str) -> str | None:
    """Return the source EPUB path recorded in ``source_epub.txt`` if present.

    The translator drops this sidecar file inside the output folder pointing at
    the original input EPUB — a more reliable hint than matching folder names.
    Forward and backward slashes are normalized; relative paths are resolved
    against the output folder itself.
    """
    sidecar = os.path.join(folder, "source_epub.txt")
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    candidate = raw.replace("/", os.sep).replace("\\", os.sep)
    if not os.path.isabs(candidate):
        candidate = os.path.join(folder, candidate)
    return os.path.normpath(candidate) if os.path.isfile(candidate) else None


def _origins_raw_sources_for_stem(folder_stem: str) -> list[str]:
    """Raw source paths in ``library_origins['raw']`` matching *folder_stem*.

    Matches the stem against both the Library/Raw basename (the key)
    and the recorded original path (the value), preferring the
    Library/Raw copy when it still exists on disk. Returns EVERY hit so
    callers can detect ambiguity (len > 1 == duplicate origins for one
    workspace name).
    """
    if not folder_stem:
        return []
    try:
        raw_map = _load_origins().get("raw", {}) or {}
    except Exception:
        return []
    if not isinstance(raw_map, dict):
        return []
    raw_dir = get_library_raw_dir()
    stem_key = os.path.normcase(folder_stem)
    matches: list[str] = []
    for lib_basename, original_path in raw_map.items():
        lb = os.path.basename(str(lib_basename or ""))
        lb_stem = os.path.splitext(lb)[0]
        op = str(original_path or "")
        op_stem = os.path.splitext(os.path.basename(op))[0]
        if (os.path.normcase(lb_stem) != stem_key
                and os.path.normcase(op_stem) != stem_key):
            continue
        lib_path = os.path.join(raw_dir, lb)
        if lb and os.path.isfile(lib_path):
            matches.append(os.path.abspath(lib_path))
        elif op and os.path.isfile(op):
            matches.append(os.path.abspath(op))
    return matches


def _validate_source_epub_for_workspace(folder: str, source_path: str) -> bool:
    """Content check: does *source_path* look like the EPUB that actually
    fed this workspace?

    An exact normalized match between the workspace folder name and the
    source filename is accepted first. This is the identity rule used by
    Scan for Raw's Exact mode, and it matters for rolling / partial EPUBs:
    a workspace can contain chapter history from several source batches
    while ``source_epub.txt`` deliberately points at only the latest batch.

    Otherwise, compares the workspace's ``translation_progress.json``
    ``original_basename`` stems against the EPUB's internal file names. A
    poisoned mapping (e.g. a ``source_epub.txt`` overwritten during a
    multi-EPUB run) normally points at a differently named book whose
    internals barely overlap, so it fails here and the caller can invalidate
    the link.

    Returns True when there is no evidence either way (non-EPUB source,
    no progress file, unreadable data) — only a demonstrated mismatch
    invalidates a candidate.
    """
    try:
        if not source_path or not str(source_path).lower().endswith(".epub"):
            return True
        if not os.path.isfile(source_path):
            return False
        folder_name = os.path.basename(os.path.normpath(folder))
        source_stem = os.path.splitext(os.path.basename(source_path))[0]
        folder_key = _norm_book_key(folder_name)
        source_key = _norm_book_key(source_stem)
        if folder_key and folder_key == source_key:
            return True
        progress_file = os.path.join(folder, "translation_progress.json")
        if not os.path.isfile(progress_file):
            return True
        import json as _json
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
        chapters = prog.get("chapters", {}) if isinstance(prog, dict) else {}
        stems: set[str] = set()
        for progress_key, ch in chapters.items():
            if isinstance(ch, dict):
                # Progress Manager scaffolds synthetic metadata, TOC, and
                # chapter-header translation rows before any source chapter
                # is processed. These are generated workspace artifacts, not
                # EPUB manifest items. Including them in the content-overlap
                # check makes every valid source_epub.txt pointer fail while
                # those are the only rows present, so cards incorrectly show
                # "missing raw" and never obtain the EPUB's spine count.
                # Use the shared classifiers so current and legacy forms of
                # both kinds of synthetic progress row are ignored here.
                if (
                    is_metadata_progress_entry(progress_key, ch)
                    or bool(ch.get("metadata_progress_key"))
                    or is_translation_artifact_progress_entry(progress_key, ch)
                ):
                    continue
                ob = ch.get("original_basename")
                if ob:
                    stems.add(_special_file_stem(str(ob)))
        stems.discard("")
        if not stems:
            return True
        import zipfile as _zipfile
        with _zipfile.ZipFile(source_path, "r") as zf:
            epub_stems = {_special_file_stem(n) for n in zf.namelist()}
        epub_stems.discard("")
        if not epub_stems:
            return False
        overlap = len(stems & epub_stems)
        # Majority overlap required: the true source contains (nearly)
        # every tracked chapter; a wrong book matches at most a few
        # generic names (cover, toc, …).
        return overlap >= max(1, (len(stems) + 1) // 2)
    except Exception:
        logger.debug("Source-EPUB validation failed open-ended: %s",
                     traceback.format_exc())
        return True


def _find_raw_source_for_folder(folder: str) -> str | None:
    """Try to locate the raw source file that fed this output folder.

    Candidates are gathered in priority order and the FIRST one that
    passes :func:`_validate_source_epub_for_workspace` wins:

      1. ``library_origins['raw']`` stem matches — maintained by
         organize/undo, survives sidecar corruption.
      2. ``library_raw_inputs.txt`` stem matches — every raw the user
         ever loaded, the broadest reliable registry.
      3. ``Library/Raw/<folder_basename>.<ext>`` — direct name match.
      4. ``<folder>/source_epub.txt`` — LAST RESORT: the sidecar can be
         stale or overwritten by multi-EPUB runs.

    A candidate that exists but fails content validation is skipped —
    a poisoned pointer must never win just because the file exists.
    Returns ``None`` when nothing validates (the link is invalidated
    rather than mapped to the wrong book).
    """
    folder_name = os.path.basename(os.path.normpath(folder))
    if not folder_name:
        return None
    fn_key = os.path.normcase(folder_name)
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(p):
        if not p:
            return
        try:
            if not os.path.isfile(p):
                return
            key = os.path.normcase(os.path.normpath(p))
        except Exception:
            return
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    # 1. Origins registry stem matches
    for p in _origins_raw_sources_for_stem(folder_name):
        _add(p)
    # 2. Raw-inputs registry stem matches
    for p in load_library_raw_inputs():
        try:
            stem = os.path.splitext(os.path.basename(p))[0]
        except Exception:
            continue
        if os.path.normcase(stem) == fn_key:
            _add(p)
    # 3. Library/Raw direct name match
    raw_dir = get_library_raw_dir()
    for ext in (".epub", ".txt", ".pdf", ".html"):
        _add(os.path.join(raw_dir, folder_name + ext))
    # 4. source_epub.txt pointer — last resort
    _add(_read_source_epub_pointer(folder))

    for cand in candidates:
        if _validate_source_epub_for_workspace(folder, cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# translation_progress.json summary (U3: the Library card numbers, also the mobile
# job_runner.ProgressWatcher) and the special-file rules it applies; moved verbatim
# from epub_library.
# ---------------------------------------------------------------------------

_DEFAULT_SPECIAL_FILE_KEYWORDS = (
    "title, toc, copyright, preface, nav, message, notice, colophon, "
    "dedication, epigraph, foreword, acknowledgment, author, appendix, "
    "bibliography"
)


_DEFAULT_SPECIAL_FILE_EXACT = "index, glossary, glossary_extension, glossary_unified"


def _parse_special_file_list(value: object) -> list[str]:
    """Parse a comma-separated special-file setting into lowercase tokens."""
    if value is None:
        return []
    return [p.strip().lower() for p in str(value).split(",") if p.strip()]


def _resolve_special_file_lists(config: dict | None = None) -> tuple[list[str], list[str]]:
    """Return configured (substring_keywords, exact_keywords).

    The Other Settings dialog exposes the same values as
    ``special_file_keywords`` / ``special_file_exact`` in config and mirrors
    edits into ``SPECIAL_FILE_KEYWORDS`` / ``SPECIAL_FILE_EXACT``. Environment
    variables win because translator worker processes use them as runtime
    overrides; otherwise config wins; otherwise use the UI defaults.
    """
    cfg = config or {}

    if "SPECIAL_FILE_KEYWORDS" in os.environ:
        kw_value = os.environ.get("SPECIAL_FILE_KEYWORDS", "")
    elif "special_file_keywords" in cfg:
        kw_value = cfg.get("special_file_keywords", "")
    else:
        kw_value = _DEFAULT_SPECIAL_FILE_KEYWORDS

    if "SPECIAL_FILE_EXACT" in os.environ:
        exact_value = os.environ.get("SPECIAL_FILE_EXACT", "")
    elif "special_file_exact" in cfg:
        exact_value = cfg.get("special_file_exact", "")
    else:
        exact_value = _DEFAULT_SPECIAL_FILE_EXACT

    return _parse_special_file_list(kw_value), _parse_special_file_list(exact_value)


def _has_number_in_filename(name: str) -> bool:
    """Return True if the filename (without extension) contains a digit.

    Mirrors the translator's ``_has_number_in_filename`` so the library
    classifies files the same way the translation pipeline does.
    """
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    return bool(re.search(r"\d", stem))


def _resolve_translate_all_numbered(config: dict | None = None) -> bool:
    """Return the effective ``translate_all_numbered_html`` setting.

    Environment variable ``TRANSLATE_ALL_NUMBERED_HTML`` wins, then the
    config dict value (default True).
    """
    env = os.environ.get("TRANSLATE_ALL_NUMBERED_HTML", "").strip()
    if env == "1":
        return True
    if env == "0":
        return False
    return bool((config or {}).get("translate_all_numbered_html", True))


def _is_configured_special_file(name: str, config: dict | None = None) -> bool:
    """Return True when *name* matches the configured special-file lists.

    When ``translate_all_numbered_html`` is enabled in *config*, files whose
    stem contains a digit are NOT considered special — they will be translated
    despite matching a skip keyword.
    """
    stem = _special_file_stem(name)
    if not stem:
        return False
    keywords, exact = _resolve_special_file_lists(config)
    is_match = stem in exact or any(kw in stem for kw in keywords)
    if not is_match:
        return False
    # When the "Translate All Numbered HTML Files" toggle is ON, files
    # with a digit in their filename are force-translated by the
    # translator even if they match a skip keyword.  The library must
    # mirror that decision so the displayed chapter list, progress
    # fractions, and "Show skipped files" filtering reflect the real
    # translation scope.
    if _resolve_translate_all_numbered(config) and _has_number_in_filename(name):
        return False
    return True


def _is_gallery_filename(name: str) -> bool:
    """Return True if *name* refers to the auto-generated gallery page.

    Matches ``gallery.xhtml``, ``gallery.html``, ``Gallery.xhtml``,
    ``response_gallery.*`` and similar variants (case-insensitive,
    extension-agnostic). The gallery is injected by the translator's
    compile step, never a real source chapter, so it must never count
    toward the translation progress fraction nor render a status badge.
    """
    if not name:
        return False
    base = os.path.basename(str(name)).lower()
    if base.startswith("response_"):
        base = base[len("response_"):]
    stem = os.path.splitext(base)[0]
    return stem == "gallery"


_PROGRESS_SIDECAR_FILENAMES = frozenset({
    "source_epub.txt",
    "image_rename_map.json",
    "image_reference_map.json",
})


def _is_progress_sidecar_entry(key, entry: dict | None = None) -> bool:
    """Return True for workspace bookkeeping rows that are never chapters."""
    entry = entry if isinstance(entry, dict) else {}
    for value in (
        key,
        entry.get("original_basename"),
        entry.get("output_file"),
    ):
        if (value and os.path.basename(str(value)).casefold()
                in _PROGRESS_SIDECAR_FILENAMES):
            return True
    return False


def _read_progress_summary(progress_file: str, exclude_special: bool = False,
                           config: dict | None = None) -> dict | None:
    """Return a lightweight summary of translation_progress.json or None on failure.

    The summary counts chapter statuses so the library card can render a
    fraction/percentage without paying for the full OPF-aware match.

    When *exclude_special* is True, entries matching the configured
    special-file substring/exact lists are dropped from both the total and
    the status tallies. This matches :func:`_count_epub_spine_items` and
    :func:`_count_translated_response_files` so the toggle shifts numerator
    and denominator together.

    **File-existence verification**: entries whose ``status`` is
    ``"completed"`` are only counted as completed when the
    corresponding ``output_file`` still exists on disk (relative to
    the progress file's folder). The translator writes
    ``response_*.html`` files during translation and marks the
    progress entry ``completed`` — but if the user later manually
    deletes one of those files, the JSON still carries the stale
    ``completed`` status. Without this verification the card would
    keep reading e.g. 60/60 even though only 55 of the response
    files remain on disk. Any demoted entries are counted as
    ``in_progress`` instead so the fraction reflects reality.
    """
    import json as _json
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return None
    chapters = prog.get("chapters", {}) or {}
    total = 0
    completed = 0
    in_progress = 0
    failed = 0
    output_folder = os.path.dirname(progress_file) if progress_file else ""
    output_folder_ok = bool(output_folder) and os.path.isdir(output_folder)
    for key, ch in chapters.items():
        if not isinstance(ch, dict):
            continue
        if _is_progress_sidecar_entry(key, ch):
            continue
        # Metadata / TOC / chapter-header translation rows are workspace
        # phases, not book chapters. Counting them inflated the total past
        # the raw spine and let two extra completed rows hide a missing
        # chapter, so a book could land on Completed with work left.
        if (
            is_metadata_progress_entry(key, ch)
            or bool(ch.get("metadata_progress_key"))
            or is_translation_artifact_progress_entry(key, ch)
        ):
            continue
        name = (ch.get("original_basename")
                or ch.get("output_file")
                or str(key)
                or "")
        # Auto-generated gallery entries never count toward progress,
        # regardless of the translate-special-files toggle.
        if _is_gallery_filename(name):
            continue
        if exclude_special:
            # Prefer original_basename (source file) over output_file
            # (response_*.html) so a run started with translate_special=ON
            # and flipped OFF still filters correctly. Fall back to the
            # progress-file key as a last resort.
            stem = _special_file_stem(name)
            # Only skip when we HAVE a filename to judge by — entries
            # with no recognizable name stay in the count rather than
            # silently disappearing.
            if stem and _is_configured_special_file(name, config):
                continue
        total += 1
        status = ch.get("status", "")
        chunk_key = str(ch.get("content_hash") or key)
        chunk_entry = prog.get("chapter_chunks", {}).get(chunk_key)
        if is_multi_chunk_entry(chunk_entry):
            chunk_summary = chunk_failure_summary(chunk_entry)
            status = effective_parent_status(status, chunk_entry)
            # Keep the parent chapter row individually complete for scanner
            # compatibility, but count a child-only QA failure as failed in
            # the book-level aggregate rather than claiming the book is done.
            if chunk_summary["failed"] and status == "completed":
                status = "qa_failed"
        # Phantom-completion check: the progress JSON may say
        # "completed" but the actual output file could have been
        # deleted by the user. Verify it's still on disk before
        # counting it as done — otherwise the card fraction lies.
        if status == "completed" and output_folder_ok:
            of = ch.get("output_file") or ""
            if of:
                candidate = of if os.path.isabs(of) else os.path.join(
                    output_folder, of)
                if not os.path.isfile(candidate):
                    # Demote locally for counting. The JSON itself
                    # isn't rewritten — the translator's own progress
                    # manager is the source of truth for that.
                    status = "in_progress"
        if status == "completed":
            completed += 1
        elif status in ("in_progress", "pending"):
            in_progress += 1
        elif status in ("failed", "qa_failed"):
            failed += 1
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "prog": prog,
    }


# ---------------------------------------------------------------------------
# U5: moved verbatim from epub_library (scans, resolvers, query, card data,
# Book Details helpers); epub_library re-imports every name.
# ---------------------------------------------------------------------------


def _special_file_settings_signature(config: dict | None = None) -> str:
    """Stable signature used by caches that filter configured special files."""
    keywords, exact = _resolve_special_file_lists(config)
    cfg = config or {}
    _all_numbered = (
        os.environ.get('TRANSLATE_ALL_NUMBERED_HTML', '0') == '1'
        or cfg.get('translate_all_numbered_html', True)
    )
    return hashlib.md5(
        ("kw=" + ",".join(keywords) + "|exact=" + ",".join(exact)
         + "|numbered=" + str(int(bool(_all_numbered)))).encode("utf-8")
    ).hexdigest()[:10]


# Bump this salt whenever the loader's output schema changes so old
# pickled caches miss cleanly instead of being served back forever.
#   v2  — spine-first chapter resolution + text/html fallback.
#   v3  — authoritative items (cover, nav, TOC pages) no longer dropped
#          by the text-length filter.
#   v4  — reader respects the Show-special-files toggle (cache key now
#          embeds its state so on/off entries don't collide).
#   v6  - image cache includes manifest-declared image assets even when
#          ebooklib classifies them as ITEM_UNKNOWN (notably image/webp).
#   v7  - image entries are lightweight EPUB-member descriptors; bytes are
#          extracted only when a displayed/preloaded chapter references them.
_EPUB_CACHE_SCHEMA = "v7"


def _epub_cache_key(epub_path: str, show_special_files: bool = True,
                    config: dict | None = None) -> str:
    """Generate a cache key from path + file modification time + schema.

    *show_special_files* is baked into the key so switching the
    Show-special-files toggle forces a fresh parse instead of serving a
    cache produced under the opposite toggle state.
    """
    try:
        mtime = os.path.getmtime(epub_path)
    except OSError:
        mtime = 0
    special_sig = (
        _special_file_settings_signature(config)
        if not show_special_files else ""
    )
    salt = (
        f"{_EPUB_CACHE_SCHEMA}|special={int(bool(show_special_files))}"
        f"|special_sig={special_sig}"
    )
    raw = f"{epub_path}|{mtime}|{salt}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def _workspace_compile_kind(book: dict | None, folder: str) -> str:
    """Return the compiled format appropriate for an output workspace."""
    try:
        from output_workspace import workspace_source_format

        recorded = workspace_source_format(folder)
    except Exception:
        recorded = ""
    if recorded == "PDF":
        return "pdf"
    if recorded == "EPUB":
        return "epub"
    book = book or {}
    for value in (
        book.get("workspace_kind"),
        book.get("compiled_output_kind"),
        book.get("type"),
    ):
        if str(value or "").lower() == "pdf":
            return "pdf"
    return "epub"


def _mark_chapter_pending_for_retranslation(output_folder: str,
                                            chapter_filename: str) -> bool:
    """Reset a chapter's translation_progress.json entry to ``pending``.

    Mirrors what Retranslation_GUI's "force retranslation" does for a single
    chapter: the matching entry's status is reset and its on-disk
    ``response_*`` output file is deleted so the pipeline re-translates it.
    Matching is by source-file stem (``original_basename``), falling back to
    the recorded ``output_file``. Returns True when anything changed.
    """
    import json as _json
    try:
        progress_file = os.path.join(output_folder, "translation_progress.json")
        if not os.path.isfile(progress_file):
            return False
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
        chapters = prog.get("chapters", {}) or {}
        target_stem = os.path.splitext(
            os.path.basename(str(chapter_filename)))[0].strip().lower()
        if not target_stem:
            return False
        changed = False
        for key, ch in chapters.items():
            if not isinstance(ch, dict):
                continue
            stems = set()
            ob = str(ch.get("original_basename") or "")
            if ob:
                stems.add(os.path.splitext(os.path.basename(ob))[0].lower())
            of = str(ch.get("output_file") or "")
            if of:
                stems.add(os.path.splitext(os.path.basename(of))[0].lower())
            if target_stem not in stems:
                continue
            status = str(ch.get("status") or "")
            if of:
                candidate = of if os.path.isabs(of) else os.path.join(
                    output_folder, of)
                try:
                    if os.path.isfile(candidate):
                        os.remove(candidate)
                        logger.info("Deleted %s for retranslation", candidate)
                        changed = True
                except OSError as e:
                    logger.warning("Could not delete %s: %s", candidate, e)
            if status != "pending":
                ch["status"] = "pending"
                ch["failure_reason"] = ""
                ch["error_message"] = ""
                changed = True
            chunk_key = str(ch.get("content_hash") or key)
            chunk_entry = prog.get("chapter_chunks", {}).get(chunk_key)
            if is_multi_chunk_entry(chunk_entry):
                ensure_chunk_entry_schema(chunk_entry)
                reset = reset_chunks_for_retranslation(
                    chunk_entry,
                    list(chunk_entry.get("chunks", {})),
                )
                if reset:
                    changed = True
        if changed:
            with open(progress_file, "w", encoding="utf-8") as f:
                _json.dump(prog, f, ensure_ascii=False, indent=2)
        return changed
    except Exception:
        logger.warning("Failed to mark chapter pending: %s",
                       traceback.format_exc())
        return False


def _chapter_completed_in_progress(output_folder: str,
                                   chapter_filename: str) -> bool:
    """True when the chapter has a ``completed`` progress entry whose
    response file actually exists on disk."""
    import json as _json
    try:
        progress_file = os.path.join(output_folder, "translation_progress.json")
        if not os.path.isfile(progress_file):
            return False
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
        target_stem = os.path.splitext(
            os.path.basename(str(chapter_filename)))[0].strip().lower()
        if not target_stem:
            return False
        for ch in (prog.get("chapters", {}) or {}).values():
            if not isinstance(ch, dict):
                continue
            stems = set()
            for field in ("original_basename", "output_file"):
                val = str(ch.get(field) or "")
                if val:
                    stems.add(os.path.splitext(os.path.basename(val))[0].lower())
            if target_stem not in stems:
                continue
            if str(ch.get("status") or "") != "completed":
                continue
            of = str(ch.get("output_file") or "")
            if not of:
                continue
            candidate = of if os.path.isabs(of) else os.path.join(
                output_folder, of)
            if os.path.isfile(candidate):
                return True
        return False
    except Exception:
        logger.debug("Completion check failed: %s", traceback.format_exc())
        return False


def _cleanup_incomplete_chapter_output(output_folder: str,
                                       chapter_filename: str) -> bool:
    """Erase whatever an interrupted single-chapter run left behind.

    A stopped/failed run can leave a partially-written ``response_*`` file
    and an ``in_progress`` progress entry — the reader overlay would then
    keep rendering the half-translated chapter. This deletes the partial
    output file and resets the matching progress entry to ``pending``
    (entries that genuinely completed are left untouched). Returns True
    when anything changed.
    """
    import json as _json
    try:
        progress_file = os.path.join(output_folder, "translation_progress.json")
        if not os.path.isfile(progress_file):
            return False
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = _json.load(f)
        chapters = prog.get("chapters", {}) or {}
        target_stem = os.path.splitext(
            os.path.basename(str(chapter_filename)))[0].strip().lower()
        if not target_stem:
            return False
        changed = False
        for key, ch in chapters.items():
            if not isinstance(ch, dict):
                continue
            stems = set()
            for field in ("original_basename", "output_file"):
                val = str(ch.get(field) or "")
                if val:
                    stems.add(os.path.splitext(os.path.basename(val))[0].lower())
            if target_stem not in stems:
                continue
            status = str(ch.get("status") or "")
            of = str(ch.get("output_file") or "")
            candidate = ""
            if of:
                candidate = of if os.path.isabs(of) else os.path.join(
                    output_folder, of)
            # A genuine completion (status + file on disk) is left alone.
            if status == "completed" and candidate and os.path.isfile(candidate):
                continue
            if candidate and os.path.isfile(candidate):
                try:
                    os.remove(candidate)
                    logger.info("Removed partial translation %s", candidate)
                    changed = True
                except OSError as e:
                    logger.warning("Could not delete %s: %s", candidate, e)
            if status not in ("", "pending"):
                ch["status"] = "pending"
                ch["failure_reason"] = ""
                ch["error_message"] = ""
                changed = True
        if changed:
            with open(progress_file, "w", encoding="utf-8") as f:
                _json.dump(prog, f, ensure_ascii=False, indent=2)
        return changed
    except Exception:
        logger.warning("Incomplete-output cleanup failed: %s",
                       traceback.format_exc())
        return False


def _library_io_worker_count(total: int, cap: int = 8) -> int:
    """Return a conservative thread count for independent library disk reads."""
    if total <= 1:
        return 1
    try:
        cpu_count = os.cpu_count() or 2
    except Exception:
        cpu_count = 2
    return min(total, cap, max(2, cpu_count * 2))


def _reader_worker_count(total: int, config: dict | None = None) -> int:
    """Return the user-configured worker count for EPUB reader tasks."""
    if total <= 1:
        return 1
    cfg = config or {}
    try:
        if cfg and not bool(cfg.get("enable_parallel_extraction", True)):
            return 1
    except Exception:
        pass

    raw_workers = None
    if cfg and "extraction_workers" in cfg:
        raw_workers = cfg.get("extraction_workers")
    if raw_workers is None:
        raw_workers = os.environ.get("EXTRACTION_WORKERS", "2")
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError):
        workers = 1
    return min(total, max(1, workers))


def _resolve_output_roots(config: dict | None = None) -> list[str]:
    """Return every directory the translator may have written output folders into.

    Both the configured override (``OUTPUT_DIRECTORY`` env var or
    ``config['output_directory']``) AND the default fallback location
    (app dir on Windows, CWD elsewhere) are returned when they exist
    on disk, so the In Progress + Completed tabs surface flash cards
    from either path in the same scan. Results are de-duplicated by
    normalized absolute path so an override pointing at the same dir
    as the fallback doesn't produce two scan passes.

    Previously this helper treated the override as *strict* — when
    set, the fallback was excluded. Users reported losing access to
    older workspaces still sitting in the fallback location after
    setting an override, so the helper now unions the two.
    """
    config = config or {}
    roots: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if not candidate:
            return
        try:
            abs_p = os.path.abspath(candidate)
        except (TypeError, ValueError):
            return
        if not os.path.isdir(abs_p):
            return
        key = os.path.normcase(os.path.normpath(abs_p))
        if key in seen:
            return
        seen.add(key)
        roots.append(abs_p)

    # Override first so it wins priority when both locations hold the
    # same workspace basename (the scanner dedups workspaces by
    # folder-level key, first-seen wins in :func:`scan_output_folders`).
    override = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory")
    _add(override)

    default_dir = _default_output_root()
    _add(default_dir)

    return roots


def _default_output_root() -> str:
    """Return the implicit default output root used when no override is set.

    Mirrors the fallback rule inside :func:`_resolve_output_roots`: on
    Windows, the frozen app's own directory (or the source file's dir
    in dev runs); on other platforms, the current working directory.
    Surfacing it as a standalone helper lets the "Load for translation"
    flow compare a flash card's backing output folder against the
    active override even when the override is empty (i.e. "use the
    default") — the two cases are indistinguishable without this value.
    """
    if _LIBRARY_ENV is not None and _LIBRARY_ENV.output_roots:
        # U5 seam: Glossarion Mobile pins its Output folder (desktop never does).
        return _LIBRARY_ENV.output_roots[0]
    if platform.system() == "Windows":
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def _expected_output_root_for_book(book: dict) -> str:
    """Return the output root directory a flash card's translation lives
    under, or ``""`` when the book doesn't carry a resolvable
    ``output_folder``.

    A "output root" here is the PARENT of the workspace folder —
    i.e. whichever of ``OUTPUT_DIRECTORY`` / the default fallback root
    produced this card during :func:`scan_output_folders`. Library-
    organized cards (``in_library=True``) typically have no active
    ``output_folder`` so this helper returns ``""`` for them — the
    caller should treat that as "no mismatch to check" rather than as
    a real root.
    """
    if not isinstance(book, dict):
        return ""
    out = book.get("output_folder") or ""
    if not out:
        return ""
    try:
        out_abs = os.path.abspath(out)
    except Exception:
        return ""
    if not os.path.isdir(out_abs):
        return ""
    parent = os.path.dirname(out_abs)
    return parent if parent and os.path.isdir(parent) else ""


def _output_paths_equal(a: str, b: str) -> bool:
    """Case-insensitive, normalized equality for two filesystem paths.

    Treats two empty strings as equal so the "no override configured"
    state compares cleanly against itself.
    """
    if not a and not b:
        return True
    if not a or not b:
        return False
    try:
        return (os.path.normcase(os.path.normpath(os.path.abspath(a)))
                == os.path.normcase(os.path.normpath(os.path.abspath(b))))
    except Exception:
        return False


# Per-process cache of OPF search metadata keyed by ``path|mtime`` so a
# repeated scan (auto-refresh, undo, organize, …) doesn't re-open the
# zip for every library EPUB every time. Each value is ``(titles, subjects)``.
_EPUB_SEARCH_METADATA_CACHE: dict[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {}


def _extract_epub_search_metadata(
    epub_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return title-like strings and subjects from an EPUB's OPF metadata.

    Reads only the OPF (no content decode), pulling ``dc:title``,
    ``dc:alternative``, and the ``calibre:original_title`` meta element
    the compiler writes at build time (see ``_create_book`` in
    ``epub_converter.py``), plus every repeatable ``dc:subject`` value.
    Returned entries deliberately stay raw: title callers normalize their
    values for matching, while the library search uses subjects as displayed.

    Cached per ``(path, mtime)`` so repeat scans are cheap. Returns an
    empty pair on any failure — missing metadata is never fatal.
    """
    if not epub_path or not os.path.isfile(epub_path):
        return (), ()
    try:
        mtime = os.path.getmtime(epub_path)
    except OSError:
        mtime = 0
    cache_key = f"{epub_path}|{mtime}"
    cached = _EPUB_SEARCH_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    titles: list[str] = []
    subjects: list[str] = []
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            opf_path = find_epub_opf_member(zf)
            if opf_path and opf_path in names_set:
                opf_xml = zf.read(opf_path).decode(
                    "utf-8", errors="replace")
                otree = ET.fromstring(opf_xml)
                DC = "http://purl.org/dc/elements/1.1/"
                OPF = "http://www.idpf.org/2007/opf"
                for tag in ("title", "alternative"):
                    for el in otree.findall(f".//{{{DC}}}{tag}"):
                        txt = (el.text or "").strip()
                        if txt:
                            titles.append(txt)
                for el in otree.findall(f".//{{{DC}}}subject"):
                    txt = (el.text or "").strip()
                    if txt:
                        subjects.append(txt)
                # Calibre-style ``<meta name="calibre:original_title"
                # content="…"/>`` — this is how the translator stores
                # the raw source title when compiling, so it's the
                # single most useful signal for pairing a
                # translated-name library EPUB back to its
                # raw-name workspace.
                for meta_el in otree.findall(f".//{{{OPF}}}meta"):
                    name = meta_el.get("name") or ""
                    if name.lower() in ("calibre:original_title",
                                        "original_title"):
                        content = meta_el.get("content") or ""
                        content = content.strip()
                        if content:
                            titles.append(content)
    except Exception:
        logger.debug("OPF search metadata extraction failed for %s: %s",
                     epub_path, traceback.format_exc())
    result = (
        tuple(dict.fromkeys(titles)),
        tuple(dict.fromkeys(subjects)),
    )
    _EPUB_SEARCH_METADATA_CACHE[cache_key] = result
    return result


def _extract_epub_titles(epub_path: str) -> tuple[str, ...]:
    """Return title-like strings embedded in *epub_path*'s OPF metadata."""
    return _extract_epub_search_metadata(epub_path)[0]


def _extract_epub_subjects(epub_path: str) -> tuple[str, ...]:
    """Return every ordered, de-duplicated ``dc:subject`` value."""
    return _extract_epub_search_metadata(epub_path)[1]


_LIBRARY_TAG_SEARCH_KEYS = (
    "subject",
    "subjects",
    "original_subject",
    "tags",
    "genres",
)


def _iter_library_search_values(value):
    """Yield scalar strings from list-like metadata search values."""
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_library_search_values(item)
        return
    if value is None:
        return
    text = str(value).strip()
    if text:
        yield text


def _book_library_tag_values(book: dict) -> tuple[str, ...]:
    """Return normalized searchable tag strings carried by a book row."""
    tag_sources = [
        book.get("subjects"),
        book.get("raw_subjects"),
        book.get("tags"),
        book.get("genres"),
    ]
    metadata = book.get("metadata_json") or {}
    if isinstance(metadata, dict):
        tag_sources.extend(metadata.get(key) for key in _LIBRARY_TAG_SEARCH_KEYS)

    return tuple(dict.fromkeys(
        value.casefold()
        for source in tag_sources
        for value in _iter_library_search_values(source)
    ))


def _book_library_title_values(book: dict) -> tuple[str, ...]:
    """Every title a user may search a card by, translated and raw.

    Raw (source-language) titles are always searchable, whether or not
    the "Raw titles" toggle currently shows them on the card.
    """
    titles = [book.get("name"), _card_raw_title(book)]
    for path_key in ("raw_source_path", "original_path"):
        path = book.get(path_key) or ""
        if path:
            titles.append(os.path.splitext(os.path.basename(str(path)))[0])
    titles.append(book.get("folder_name"))
    metadata = book.get("metadata_json") or {}
    if isinstance(metadata, dict):
        for key in ("title", "translated_title", "original_title", "raw_title", "source_title"):
            titles.append(metadata.get(key))
    return tuple(dict.fromkeys(
        value.casefold()
        for title in titles
        for value in _iter_library_search_values(title)
    ))


def _book_matches_library_query(book: dict, query: str) -> bool:
    """Match a library query against translated/raw titles and tags/subjects."""
    needle = str(query or "").strip().casefold()
    if not needle:
        return True
    if any(needle in value for value in _book_library_title_values(book)):
        return True
    return any(needle in value for value in _book_library_tag_values(book))


# Process-level cache for EPUB spine-item counts, keyed by ``path|mtime``.
_SPINE_COUNT_CACHE: dict[str, int] = {}


def _count_epub_spine_items(epub_path: str, exclude_special: bool = False,
                            config: dict | None = None) -> int:
    """Return the number of itemrefs in the EPUB's spine (0 on failure).

    Cheap: reads only ``META-INF/container.xml`` + the OPF, never the
    chapter HTML. Results are memoised per ``(path, mtime, exclude_special)``
    so repeated scans don't re-open the zip. This is the authoritative
    chapter count for EPUB workspaces — ``translation_progress.json`` only
    tracks chapters the translator has actually processed, so it
    under-reports the real length for freshly imported / early-progress
    novels.

    When *exclude_special* is True, spine items matching the configured
    special-file substring/exact lists are skipped, matching the translator's
    default behavior of not translating special files unless
    ``translate_special_files`` is explicitly enabled.
    """
    if not epub_path or not os.path.isfile(epub_path):
        return 0
    try:
        base_key = _epub_cache_key(epub_path)
    except Exception:
        base_key = ""
    special_sig = _special_file_settings_signature(config) if exclude_special else ""
    cache_key = (
        f"{base_key}|excl={int(bool(exclude_special))}|special={special_sig}"
        if base_key else ""
    )
    if cache_key and cache_key in _SPINE_COUNT_CACHE:
        return _SPINE_COUNT_CACHE[cache_key]
    count = 0
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)
            opf_path = find_epub_opf_member(zf)
            if opf_path and opf_path in names_set:
                try:
                    opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
                    tree = ET.fromstring(opf_xml)
                    OPF = "http://www.idpf.org/2007/opf"
                    manifest: dict[str, str] = {}
                    for item in tree.findall(f".//{{{OPF}}}item"):
                        item_id = item.get("id", "")
                        href = item.get("href", "")
                        if item_id and href:
                            manifest[item_id] = href
                    spine = tree.find(f".//{{{OPF}}}spine")
                    if spine is not None:
                        for itemref in spine.findall(f"{{{OPF}}}itemref"):
                            idref = itemref.get("idref") or ""
                            href = manifest.get(idref, "")
                            basename = os.path.basename(href)
                            # Auto-generated gallery page never counts
                            # toward the spine total, no matter the
                            # translate-special-files toggle.
                            if _is_gallery_filename(basename):
                                continue
                            if exclude_special:
                                if _is_configured_special_file(basename, config):
                                    continue
                            count += 1
                except Exception:
                    pass
    except Exception:
        logger.debug("Spine count failed for %s: %s",
                     epub_path, traceback.format_exc())
    if cache_key:
        _SPINE_COUNT_CACHE[cache_key] = count
    return count


def _resolve_translate_special_files(config: dict | None) -> bool:
    """Return the effective ``translate_special_files`` setting.

    Environment variable ``TRANSLATE_SPECIAL_FILES`` takes precedence over
    the config dict so runtime overrides used by the translator work for
    the library scanner too.
    """
    env = os.environ.get("TRANSLATE_SPECIAL_FILES", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool((config or {}).get("translate_special_files", False))


def _resolve_show_special_files(config: dict | None) -> bool:
    """Return the effective "show special files" flag for reader / details.

    Mirrors the logic baked into :class:`BookDetailsDialog.__init__` so
    callers that don't go through Book Details (e.g. the library card's
    "Open in Reader" context-menu action) still pick up the same resolved
    state: the explicit ``epub_details_show_special_files`` preference
    when set, else the global ``translate_special_files`` flag. An
    explicit False is only honoured when the global is also False —
    turning ON "translate special" always propagates through.
    """
    cfg = config or {}
    translate_special = _resolve_translate_special_files(cfg)
    stored = cfg.get("epub_details_show_special_files", None)
    if stored is None:
        return translate_special
    return bool(stored) or translate_special


def _is_special_spine_item(name: str, config: dict | None = None) -> bool:
    """Return True if *name* matches configured special-file keywords.

    This intentionally does not use the old "no digits means special"
    heuristic: files like ``cover.html`` or ``info.html`` are ordinary spine
    entries unless the user puts matching tokens in Other Settings.
    """
    return _is_configured_special_file(name, config)


def _count_translated_response_files(folder: str, exclude_special: bool = False,
                                     config: dict | None = None) -> int:
    """Count ``response_*.{html,xhtml,htm,txt}`` files inside *folder*.

    Acts as a filesystem-based "done" count fallback for cards whose
    ``translation_progress.json`` is missing / empty (e.g. a crashed run
    that never flushed ``status=completed`` to the sidecar).

    When *exclude_special* is True, response files matching the configured
    special-file substring/exact lists are skipped. This keeps denominator
    and numerator consistent when the "translate special files" toggle is
    off.
    """
    if not folder or not os.path.isdir(folder):
        return 0
    count = 0
    try:
        for entry in os.scandir(folder):
            if not entry.is_file(follow_symlinks=False):
                continue
            lower = entry.name.lower()
            if not lower.startswith("response_"):
                continue
            if not lower.endswith((".html", ".xhtml", ".htm", ".txt")):
                continue
            # Gallery is auto-generated — never count it toward done.
            if _is_gallery_filename(entry.name):
                continue
            if exclude_special:
                if _is_configured_special_file(lower, config):
                    continue
            count += 1
    except (PermissionError, OSError):
        return 0
    return count


def _folder_has_output_epub(folder: str) -> str | None:
    """Return the path to the first .epub in *folder* or None."""
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                return entry.path
    except (PermissionError, OSError):
        pass
    return None


def _folder_has_compiled_output(folder: str) -> tuple[str, str] | None:
    """Return ``(path, kind)`` for any compiled translation output, or None.

    Recognized kinds (in priority order):
      * ``"epub"`` — any ``*.epub`` at the folder root.
      * ``"pdf"``  — a ``*_translated.pdf`` alongside the progress file.
      * ``"txt"``  — a ``*_translated.txt`` alongside the progress file.
      * ``"html"`` — a ``*_translated.html`` alongside the progress file.
    """
    outputs = _list_compiled_outputs(folder)
    return outputs[0] if outputs else None


def _list_compiled_outputs(folder: str) -> list[tuple[str, str]]:
    """Return every compiled output in *folder* as ``[(path, kind), …]``.

    Same detection rules as :func:`_folder_has_compiled_output`, but
    returns the FULL list in priority order instead of just the first.
    Used by :func:`scan_output_folders` to flag folders that contain
    more than one compiled artefact (e.g. two ``.epub`` files from
    successive recompiles, or a ``.epub`` paired with a leftover
    ``*_translated.html``) so the card can render a warning badge and
    the user can investigate / clean up.

    ``*_translated.html`` files are silently excluded when a
    ``*_translated.pdf`` with the same stem exists in the folder,
    because PDF translation always emits a companion HTML for
    debugging — it is not a separate output.
    """
    results: list[tuple[str, str]] = []
    try:
        entries = list(os.scandir(folder))
    except (PermissionError, OSError):
        return results
    # EPUBs win when present — that's the typical compiled shelf artifact.
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        if entry.name.lower().endswith(".epub"):
            results.append((entry.path, "epub"))
    # Fall-back compiled outputs for TXT/PDF/HTML translations.
    priority = (("_translated.pdf", "pdf"),
                ("_translated.txt", "txt"),
                ("_translated.html", "html"))
    # Collect PDF stems so we can suppress their companion debug HTML.
    pdf_stems: set[str] = set()
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        nl = entry.name.lower()
        if nl.endswith("_translated.pdf"):
            pdf_stems.add(nl[: -len("_translated.pdf")])
    for suffix, kind in priority:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            nl = entry.name.lower()
            if nl.endswith(suffix):
                # Skip _translated.html when a _translated.pdf with the
                # same stem exists — the HTML is a PDF debug artifact.
                if kind == "html" and nl[: -len(suffix)] in pdf_stems:
                    continue
                results.append((entry.path, kind))
    return results


def _detect_workspace_kind(folder: str, source_epub_path: str = "") -> str:
    """Best-effort classification of a translation workspace.

    Returns one of ``"epub"``, ``"txt"``, ``"pdf"``, ``"image"`` or ``"other"``.
    The heuristic is:
      1. If *source_epub_path* is provided and has a known extension, use it.
      2. Folder contains ``content.opf`` / ``*.epub`` → EPUB.
      3. Folder contains a compiled ``*_translated.{txt,pdf,html}`` → that kind.
      4. Folder contains ``word_count/`` and no OPF → TXT (the Glossarion
         text translator writes word_count per chunk, EPUBs don't).
      5. Otherwise → ``"other"``.
    """
    if source_epub_path:
        low = source_epub_path.lower()
        for ext, kind in ((".epub", "epub"), (".txt", "txt"), (".pdf", "pdf")):
            if low.endswith(ext):
                return kind
        if low.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            return "image"
    try:
        has_opf = False
        has_epub = False
        has_word_count_dir = False
        translated_ext: str | None = None
        for entry in os.scandir(folder):
            nm = entry.name.lower()
            if entry.is_file(follow_symlinks=False):
                if nm.endswith(".opf"):
                    has_opf = True
                elif nm.endswith(".epub"):
                    has_epub = True
                elif nm.endswith("_translated.txt"):
                    translated_ext = translated_ext or "txt"
                elif nm.endswith("_translated.pdf"):
                    translated_ext = translated_ext or "pdf"
                elif nm.endswith("_translated.html"):
                    translated_ext = translated_ext or "html"
            elif entry.is_dir(follow_symlinks=False):
                if nm == "word_count":
                    has_word_count_dir = True
    except (PermissionError, OSError):
        return "other"
    if has_opf or has_epub:
        return "epub"
    if translated_ext:
        return translated_ext
    if has_word_count_dir:
        return "txt"
    return "other"


def _resolve_book_output_folder(book: dict) -> str:
    """Return the output-folder path that belongs to *book*, or ``""``.

    A "real" output folder is one the translator actually wrote into —
    i.e. the workspace holding ``translation_progress.json`` /
    ``response_*.html`` artefacts. This helper is deliberately strict
    about not dressing up ``Library/Translated`` (where a compiled EPUB
    just happens to live) as an output folder, because that's never
    what the user means when they ask for the "output folder".

    Resolution order:

      1. ``book['output_folder']`` — set by :func:`scan_output_folders`
         for in-progress + promoted-compiled cards.
      2. ``library_origins['translated']`` — for ``Library/Translated``
         entries the stored "pre-organize" path's *parent* was the
         original output folder; usually still on disk because organize
         only moves the compiled EPUB out of it.

    Returns ``""`` when neither resolves to an existing directory so the
    caller can disable the button / take a different fallback instead
    of opening the book's own containing folder (which, for library
    entries, is ``Library/Translated`` — not what the user wants).
    """
    out = (book.get("output_folder") or "") if isinstance(book, dict) else ""
    if out and os.path.isdir(out):
        return out
    if isinstance(book, dict) and book.get("in_library"):
        try:
            origins = _load_origins()
            trans_map = origins.get("translated", {}) or {}
            orig_path = trans_map.get(os.path.basename(book.get("path", "")))
            if orig_path:
                orig_folder = os.path.dirname(str(orig_path))
                if orig_folder and os.path.isdir(orig_folder):
                    return orig_folder
        except Exception:
            logger.debug("Output-folder origins lookup failed: %s",
                         traceback.format_exc())
    return ""


def _resolve_book_source_file(book: dict) -> str:
    """Return the raw source file path the 🔗 button should reveal, or ``""``.

    Mirrors :class:`BookDetailsDialog._resolve_source_file_target`
    (and is used by it) so the Book Details dialog and the card
    context menu both resolve sources through one code path:

      1. ``book['raw_source_path']`` — populated by the scanner for
         every resolvable card.
      2. For ``Library/Translated`` entries, re-run
         :func:`_find_raw_source_for_library_epub` so cards whose
         scan result predates the origins ``pairs`` entry can still
         find the matching raw.

    Returns ``""`` when no RAW source can be resolved — the caller
    (context menu / 🔗 button) omits / disables the action in that
    case instead of revealing a misleading target. Previously this
    fell back to ``book['path']``, which for ``Library/Translated``
    entries is the *compiled* EPUB sitting inside the translated
    shelf — so the action appeared to "work" but just opened the
    translated subfolder (identical to the "Reveal Translated File"
    action, which is the opposite of what the user asked for).

    Caches any fresh resolution back onto the book dict so repeat
    calls (and the reader's Raw toggle) skip the lookup.
    """
    if not isinstance(book, dict):
        return ""
    path = book.get("raw_source_path", "") or ""
    if path and os.path.isfile(path):
        return path
    if book.get("in_library"):
        lib_path = book.get("path", "") or ""
        try:
            resolved = _find_raw_source_for_library_epub(lib_path)
        except Exception:
            resolved = ""
            logger.debug("Source-file library lookup failed: %s",
                         traceback.format_exc())
        if resolved and os.path.isfile(resolved):
            book["raw_source_path"] = resolved
            return resolved
    return ""


def _resolve_book_metadata_source(book: dict) -> str:
    """Return the original EPUB that can feed metadata translation."""
    source = _resolve_book_source_file(book)
    if (
        not source
        and isinstance(book, dict)
        and book.get("type") == "epub"
        and not book.get("in_library")
        and os.path.isfile(book.get("path", ""))
    ):
        source = book["path"]
    if (
        source
        and source.lower().endswith(".epub")
        and os.path.isfile(source)
    ):
        return source
    return ""


def _resolve_book_translated_file(book: dict) -> str:
    """Return the compiled / translated EPUB path to reveal, or ``""``.

    Shared by :class:`EpubLibraryDialog._show_context_menu` (the
    "Reveal Translated File" action) and :class:`BookDetailsDialog`
    (the 📕 icon button) so the two entry points stay in lockstep.

    Resolution order:

      1. Library entries (``in_library=True``) whose ``path`` ends in
         ``.epub`` — the EPUB sitting inside ``Library/Translated``
         IS the translated artefact.
      2. ``compiled_output_path`` set by the scanner when a workspace
         row was either promoted-to-compiled or state-upgraded via the
         origins/title match pass (``_DualScannerThread``).
      3. ``output_epub_path`` legacy field — same semantics.
      4. ``book['path']`` when it itself is an ``.epub`` on disk
         (covers completed workspace rows whose ``path`` already
         points at the compiled artefact).

    Returns ``""`` when no resolvable translated file exists on disk
    so the caller can omit / disable the action rather than pointing
    at a missing target.
    """
    if not isinstance(book, dict):
        return ""
    lib_path = book.get("path", "") or ""
    if (book.get("in_library")
            and isinstance(lib_path, str)
            and lib_path.lower().endswith(".epub")
            and os.path.isfile(lib_path)):
        return lib_path
    for key in ("compiled_output_path", "output_epub_path"):
        cand = book.get(key) or ""
        if cand and os.path.isfile(cand):
            return cand
    if (isinstance(lib_path, str)
            and lib_path.lower().endswith(".epub")
            and os.path.isfile(lib_path)):
        return lib_path
    return ""


def _find_raw_source_for_library_epub(library_epub_path: str) -> str:
    """Best-effort lookup for the raw source of a Library/Translated EPUB.

    Unlike :func:`_find_raw_source_for_folder` (which starts from an
    output-folder layout complete with ``source_epub.txt``), this helper
    works purely from a compiled EPUB sitting inside ``Library/Translated``
    and has to recover the original source by name matching + the origins
    registry. Search order:

      0. ``library_origins['pairs']`` — an explicit translated↔raw
         mapping written at organize time. This is the ONLY step that
         survives a raw whose filename stem doesn't match the compiled
         EPUB's stem (e.g. a Korean raw paired with an English
         translation), which is the common case for real translations.
      1. ``Library/Raw/<same-stem>.epub`` — direct basename match, the
         common case for files organized into the library together.
      2. ``load_library_raw_inputs()`` — any registered raw input whose
         filename stem matches.
      3. ``library_origins['translated']`` → source output folder →
         ``source_epub.txt`` pointer. Works until the output folder is
         deleted or the sidecar is stale.

    Returns an absolute path string, or ``""`` when nothing matched.
    """
    if not library_epub_path or not os.path.isfile(library_epub_path):
        return ""
    stem = os.path.splitext(os.path.basename(library_epub_path))[0]
    if not stem:
        return ""
    raw_dir = get_library_raw_dir()
    # 0. Explicit translated→raw pairing persisted at organize time.
    try:
        origins = _load_origins()
        pairs = origins.get("pairs", {}) or {}
        pair_raw_basename = pairs.get(os.path.basename(library_epub_path))
        if pair_raw_basename:
            pair_path = os.path.join(raw_dir, pair_raw_basename)
            if os.path.isfile(pair_path):
                return os.path.abspath(pair_path)
    except Exception:
        logger.debug("Library-raw pairs lookup failed: %s",
                     traceback.format_exc())
        origins = {}
    # 1. Library/Raw/<stem>.epub (case-insensitive extension match)
    try:
        with os.scandir(raw_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                nm = entry.name
                nl = nm.lower()
                if not nl.endswith(".epub"):
                    continue
                if os.path.splitext(nm)[0] == stem:
                    return os.path.abspath(entry.path)
    except (PermissionError, OSError, FileNotFoundError):
        pass
    # 2. Raw-inputs registry
    for p in load_library_raw_inputs():
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith(".epub"):
            continue
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return os.path.abspath(p)
    # 3. origins["translated"] → source output folder → raw resolver
    try:
        trans_map = (origins or _load_origins()).get("translated", {}) or {}
        orig_path = trans_map.get(os.path.basename(library_epub_path))
        if orig_path:
            orig_folder = os.path.dirname(orig_path)
            if orig_folder and os.path.isdir(orig_folder):
                resolved = _find_raw_source_for_folder(orig_folder)
                if resolved and os.path.isfile(resolved):
                    return os.path.abspath(resolved)
    except Exception:
        logger.debug("Library-raw origins lookup failed: %s",
                     traceback.format_exc())
    return ""


def scan_library_completed(config: dict | None = None) -> list[dict]:
    """Scan ``Library/Translated`` (and registered-in-place EPUBs) for completed books.

    Two sources are merged:

      1. **Physically in ``Library/Translated``** — the curated shelf.
         Every ``.epub`` found via a recursive walk is surfaced with
         ``in_library=True``.
      2. **Registered in place** — paths written to
         ``Library/library_translated_inputs.txt`` by the
         drag-drop / Import pipeline. The file lives wherever the user
         dropped it (Downloads, a cloud-synced folder, etc.); the
         scanner surfaces it with ``in_library=False`` so Organize
         can pick it up later, while the Completed tab still shows
         the card immediately after registration. Missing entries are
         silently dropped.

    Before scanning we run the legacy migration that moves any EPUBs
    still sitting in the legacy Library root into ``Translated/``.
    """
    # Legacy layout support — idempotent, safe to call every scan.
    _migrate_legacy_library_layout()
    library_dir = os.path.normpath(os.path.abspath(get_library_translated_dir()))
    results: list[dict] = []
    seen: set[str] = set()

    def _walk(root: str, max_depth: int = 4, depth: int = 0):
        if depth > max_depth:
            return
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            lower = entry.name.lower()
                            if not lower.endswith(".epub"):
                                continue
                            norm = os.path.normpath(os.path.abspath(entry.path))
                            if norm in seen:
                                continue
                            seen.add(norm)
                            try:
                                stat = os.stat(entry.path)
                            except OSError:
                                continue
                            results.append({
                                "name": os.path.splitext(entry.name)[0],
                                "path": entry.path,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "in_library": True,
                                "type": "epub",
                                "subjects": list(
                                    _extract_epub_subjects(entry.path)
                                ),
                                "raw_source_path": "",
                                "_needs_raw_source_lookup": True,
                                # Library-filed cards need the same
                                # ``missing_raw_file`` flag as
                                # workspace cards so the \u26a0
                                # \"missing raw\" badge renders when
                                # the compiled EPUB's original raw
                                # can't be resolved via origins /
                                # Library/Raw / the raw-inputs
                                # registry. Without this, a library
                                # card that inherited \"in_progress\"
                                # state via :class:`_DualScannerThread`
                                # silently dropped the badge even
                                # though the raw was genuinely gone
                                # \u2014 and Reveal source file hid
                                # itself (since the raw file was
                                # unresolvable) without a
                                # corresponding warning.
                                "missing_raw_file": True,
                            })
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            _walk(entry.path, max_depth, depth + 1)
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

    if os.path.isdir(library_dir):
        _walk(library_dir)

    pending_raw_lookup = [
        row for row in results
        if row.pop("_needs_raw_source_lookup", False)
    ]
    if pending_raw_lookup:
        try:
            _load_origins()
        except Exception:
            pass

        def _resolve_library_raw(
            row: dict,
        ) -> tuple[dict, str, tuple[str, ...]]:
            raw_path = _find_raw_source_for_library_epub(row.get("path", ""))
            raw_path = raw_path or ""
            raw_subjects = _extract_epub_subjects(raw_path) if raw_path else ()
            return row, raw_path, raw_subjects

        workers = _library_io_worker_count(len(pending_raw_lookup), cap=8)
        if workers <= 1:
            resolved_iter = [
                _resolve_library_raw(row)
                for row in pending_raw_lookup
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_resolve_library_raw, row)
                    for row in pending_raw_lookup
                ]
                resolved_iter = []
                for future in as_completed(futures):
                    try:
                        resolved_iter.append(future.result())
                    except Exception:
                        logger.debug("Library raw lookup failed: %s",
                                     traceback.format_exc())
        for row, raw_counterpart, raw_subjects in resolved_iter:
            row["raw_source_path"] = raw_counterpart
            row["missing_raw_file"] = not bool(raw_counterpart)
            row["raw_subjects"] = list(raw_subjects)

    # Source 2: registered-in-place translated EPUBs. These were
    # dropped / imported onto the Completed tab but deliberately left
    # where they are on disk so the user can reverse the import
    # without fishing the file back out of Library/Translated. They
    # carry ``in_library=False`` + ``registered_translated=True`` so
    # :meth:`_organize_into_library` picks them up as candidates to
    # move into the curated shelf.
    for p in load_library_translated_inputs():
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith(".epub"):
            continue
        norm = os.path.normpath(os.path.abspath(p))
        if norm in seen:
            continue
        seen.add(norm)
        try:
            stat = os.stat(p)
        except OSError:
            continue
        results.append({
            "name": os.path.splitext(os.path.basename(p))[0],
            "path": os.path.abspath(p),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "in_library": False,
            "registered_translated": True,
            "type": "epub",
            "subjects": list(_extract_epub_subjects(p)),
            "raw_source_path": "",
            # Registered-in-place translated imports have no raw
            # link by construction \u2014 they're drag-dropped
            # compiled EPUBs with no associated workspace. The
            # badge would be misleading here because the
            # concept of a \"raw source\" doesn't apply to these
            # cards at all; default the flag to False so the
            # missing-raw warning stays scoped to workspace /
            # organized library entries.
            "missing_raw_file": False,
        })

    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def scan_output_folders(config: dict | None = None) -> list[dict]:
    """Scan the output root(s) for translation folders.

    Honors the OUTPUT_DIRECTORY override strictly via :func:`_resolve_output_roots`.
    A folder qualifies when it has either a compiled ``.epub`` *or* a
    translation_progress.json with at least one recorded chapter.

    Each result carries ``metadata_json`` (already loaded) so the card/details
    dialog can render without a second filesystem hit.

    Folders with a compiled ``.epub`` are emitted as ``type="epub"`` so they
    render identically to Library entries in the Completed tab (``path`` is
    the .epub itself). Folders without a compiled EPUB are emitted as
    ``type="in_progress"`` for the In Progress tab. Callers use
    :func:`split_output_folders_by_status` to partition the two lists.
    """
    import json as _json
    config = config or {}
    roots = _resolve_output_roots(config)
    if not roots:
        return []
    # Honor the "translate special files" toggle: when OFF (default), special
    # configured special files are never translated, so they shouldn't
    # count toward the card's total. Otherwise a fully-translated EPUB sits
    # at 98/100 forever because two special files were skipped by design.
    exclude_special = not _resolve_translate_special_files(config)

    folder_items: list[tuple[str, str]] = []
    seen_folders: set[str] = set()
    for root in roots:
        try:
            it = os.scandir(root)
        except (PermissionError, OSError):
            continue
        with it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                folder = entry.path
                key = os.path.normcase(os.path.normpath(folder))
                if key in seen_folders:
                    continue
                seen_folders.add(key)
                folder_items.append((entry.name, folder))

    if not folder_items:
        return []

    raw_abs = os.path.normcase(os.path.normpath(
        os.path.abspath(get_library_raw_dir())))

    def _scan_output_folder(entry_name: str, folder: str) -> dict | None:
        progress_file = os.path.join(folder, "translation_progress.json")
        metadata_file = os.path.join(folder, "metadata.json")
        compiled_outputs = _list_compiled_outputs(folder)
        compiled = compiled_outputs[0] if compiled_outputs else None
        output_epub = compiled[0] if (compiled and compiled[1] == "epub") else None
        compiled_path = compiled[0] if compiled else None
        compiled_kind = compiled[1] if compiled else None
        compiled_conflicts = [
            (os.path.basename(p), k)
            for p, k in compiled_outputs[1:]
        ]
        has_progress = os.path.isfile(progress_file)
        has_metadata = os.path.isfile(metadata_file)

        summary = _read_progress_summary(
            progress_file, exclude_special=exclude_special, config=config,
        ) if has_progress else None
        progress_unparseable = has_progress and summary is None
        progress_total = summary["total"] if summary else 0
        progress_done = summary["completed"] if summary else 0
        failed = summary["failed"] if summary else 0

        raw_source_path = _find_raw_source_for_folder(folder)
        raw_in_library = bool(raw_source_path) and (
            os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(raw_source_path)))) == raw_abs
        )

        fs_done = _count_translated_response_files(
            folder, exclude_special=exclude_special, config=config)
        spine_total = 0
        if raw_source_path and raw_source_path.lower().endswith(".epub"):
            spine_total = _count_epub_spine_items(
                raw_source_path, exclude_special=exclude_special, config=config)
        total = max(progress_total, spine_total)
        done = progress_done if progress_total > 0 else fs_done

        if (not compiled and not has_progress
                and not raw_source_path
                and fs_done == 0 and not raw_in_library):
            return None

        missing_raw_file = bool(not raw_source_path and (
            compiled or has_progress or fs_done > 0))

        fully_translated = done >= total and total > 0
        if progress_unparseable:
            translation_state = "outdated_progress"
        elif fully_translated:
            translation_state = "completed" if compiled else "ready_to_compile"
        elif progress_total <= 0 and fs_done == 0:
            translation_state = "not_started"
        else:
            translation_state = "in_progress"

        if translation_state == "not_started" and missing_raw_file:
            return None

        workspace_kind = _detect_workspace_kind(folder, raw_source_path or "")
        metadata_json: dict = {}
        if has_metadata:
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    loaded = _json.load(f)
                if isinstance(loaded, dict):
                    metadata_json = loaded
            except (OSError, _json.JSONDecodeError):
                metadata_json = {}

        raw_title = (
            metadata_json.get("title")
            or metadata_json.get("original_title")
            or entry_name
        )

        promote_to_compiled = bool(compiled) and translation_state == "completed"
        if promote_to_compiled:
            card_path = compiled_path
            card_type = compiled_kind
            is_in_progress = False
            try:
                stat = os.stat(compiled_path)
            except OSError:
                return None
        else:
            card_path = folder
            card_type = "in_progress"
            is_in_progress = (
                translation_state != "completed"
                or translation_state == "outdated_progress"
            )
            try:
                stat = os.stat(folder)
            except OSError:
                return None

        return {
            "name": raw_title,
            "folder_name": entry_name,
            "path": card_path,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "in_library": False,
            "type": card_type,
            "raw_source_path": raw_source_path or "",
            "workspace_kind": workspace_kind,
            "translation_state": translation_state,
            "is_in_progress": is_in_progress,
            "output_folder": folder,
            "progress_file": progress_file if has_progress else "",
            "metadata_json_path": metadata_file if has_metadata else "",
            "metadata_json": metadata_json,
            "total_chapters": total,
            "completed_chapters": done,
            "failed_chapters": failed,
            "pending_chapters": max(0, total - done),
            "has_output_epub": bool(output_epub),
            "output_epub_path": output_epub or "",
            "has_compiled_output": bool(compiled),
            "compiled_output_path": compiled_path or "",
            "compiled_output_kind": compiled_kind or "",
            "compiled_conflicts": compiled_conflicts,
            "missing_raw_file": missing_raw_file,
        }

    results: list[dict] = []
    workers = _library_io_worker_count(len(folder_items), cap=8)
    if workers <= 1:
        for entry_name, folder in folder_items:
            row = _scan_output_folder(entry_name, folder)
            if row:
                results.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_scan_output_folder, entry_name, folder)
                for entry_name, folder in folder_items
            ]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    logger.debug("Output folder scan failed: %s",
                                 traceback.format_exc())
                    continue
                if row:
                    results.append(row)
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def split_output_folders_by_status(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ``scan_output_folders`` results into (completed, in_progress).

    Routing is driven by ``translation_state`` — ``"completed"`` means
    done >= total (the scanner already treats that as authoritative, so
    a compiled ``.epub`` alone doesn't force the card to Completed when
    only 58/60 chapters are actually translated). Missing / empty state
    falls back to ``has_compiled_output`` / ``has_output_epub`` for
    backwards compatibility with pre-v2 rows.
    """
    completed: list[dict] = []
    in_progress: list[dict] = []
    for r in rows:
        state = r.get("translation_state")
        if state == "outdated_progress":
            # Pinned to In Progress regardless of compile state so
            # the "Outdated Progress file" warning is never hidden
            # under a Completed-tab row.
            in_progress.append(r)
        elif state == "completed":
            completed.append(r)
        elif state in ("in_progress", "not_started", "ready_to_compile"):
            in_progress.append(r)
        elif r.get("has_compiled_output") or r.get("has_output_epub"):
            completed.append(r)
        else:
            in_progress.append(r)
    return completed, in_progress


def _find_in_progress_novels(config: dict | None = None) -> list[dict]:
    """Locate novels whose translation is in progress (no output EPUB yet).

    Strictly only inspects the output roots returned by :func:`_resolve_output_roots`.
    For each first-level subfolder that contains a `translation_progress.json`
    and has NO output `.epub`, try to find the source EPUB via (in priority order):
      1. ``<folder>/source_epub.txt`` — authoritative pointer written by the
         translator.
      2. A basename match (folder name == EPUB basename) in the allowed roots.

    EPUBs already organized into the Library folder are *never* reported as
    in-progress: the Library is the curated/read-only shelf and the progress
    view is deliberately limited to active translations.

    Returns a list of dicts compatible with the rest of the scanner (with
    extra in-progress metadata).
    """
    config = config or {}
    roots = _resolve_output_roots(config)
    if not roots:
        return []
    exclude_special = not _resolve_translate_special_files(config)

    library_dir_norm = os.path.normcase(os.path.normpath(os.path.abspath(get_library_dir())))

    def _is_in_library(path: str) -> bool:
        """True when *path* resolves inside the Glossarion Library folder."""
        try:
            norm = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        except (TypeError, ValueError):
            return False
        return (norm == library_dir_norm
                or norm.startswith(library_dir_norm + os.sep)
                or os.path.normcase(os.path.dirname(norm)) == library_dir_norm)

    # Index every EPUB in the roots by basename (without extension) so we can
    # match folders to source files without a second filesystem walk per folder.
    epub_by_base: dict[str, str] = {}
    for root in roots:
        try:
            for entry in os.scandir(root):
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".epub"):
                    base = os.path.splitext(entry.name)[0]
                    # Prefer first match; tolerate case-insensitive dupes
                    epub_by_base.setdefault(base, entry.path)
                    epub_by_base.setdefault(base.lower(), entry.path)
        except (PermissionError, OSError):
            continue

    results: list[dict] = []
    seen_folders: set[str] = set()
    for root in roots:
        try:
            it = os.scandir(root)
        except (PermissionError, OSError):
            continue
        with it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                folder = entry.path
                folder_key = os.path.normcase(os.path.normpath(folder))
                if folder_key in seen_folders:
                    continue
                seen_folders.add(folder_key)
                progress_file = os.path.join(folder, "translation_progress.json")
                if not os.path.isfile(progress_file):
                    continue
                if _folder_has_output_epub(folder):
                    # Already done — normal scan will pick up the .epub.
                    continue
                summary = _read_progress_summary(
                    progress_file, exclude_special=exclude_special,
                    config=config)
                if summary is None:
                    continue
                folder_name = entry.name
                # Shared validating resolver: origins registry →
                # raw-inputs registry → Library/Raw → source_epub.txt
                # (last resort). Every candidate is content-validated
                # against translation_progress.json, so a sidecar
                # poisoned by a multi-EPUB run is invalidated instead
                # of mapping the card to the wrong book.
                source_path = _find_raw_source_for_folder(folder)
                # Fall back to folder-name↔basename match within the
                # allowed roots — validated the same way.
                if not source_path:
                    candidate = epub_by_base.get(folder_name) or epub_by_base.get(folder_name.lower())
                    if candidate and _validate_source_epub_for_workspace(folder, candidate):
                        source_path = candidate
                if not source_path or not os.path.isfile(source_path):
                    # No source EPUB visible — skip so we don't surface a ghost entry.
                    continue
                # Never mark Library-organized EPUBs as in-progress: the user
                # wants those to look clean (no progress badge) in the library
                # unless a progress file is explicitly present inside the
                # configured output root — which this path is not.
                if _is_in_library(source_path):
                    continue
                try:
                    stat = os.stat(source_path)
                except OSError:
                    continue
                results.append({
                    "name": folder_name,
                    "path": source_path,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "in_library": False,
                    "type": "epub",
                    "is_in_progress": True,
                    "output_folder": folder,
                    "progress_file": progress_file,
                    "total_chapters": summary["total"],
                    "completed_chapters": summary["completed"],
                    "failed_chapters": summary["failed"],
                    "pending_chapters": max(0, summary["total"] - summary["completed"]),
                })
    return results


def scan_for_epubs(config: dict | None = None) -> list[dict]:
    config = config or {}
    results: list[dict] = []
    seen: set[str] = set()

    library_dir = os.path.normpath(os.path.abspath(get_library_dir()))

    def _add(path: str, file_type: str = "epub"):
        norm = os.path.normpath(os.path.abspath(path))
        if norm in seen:
            return
        seen.add(norm)
        try:
            stat = os.stat(path)
            in_lib = norm.startswith(library_dir + os.sep) or os.path.dirname(norm) == library_dir
            results.append({
                "name": os.path.splitext(os.path.basename(path))[0],
                "path": path,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "in_library": in_lib,
                "type": file_type,
            })
        except OSError:
            pass

    def _walk(root: str, max_depth: int = 3, depth: int = 0):
        if depth > max_depth:
            return
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            lower = entry.name.lower()
                            if lower.endswith(".epub"):
                                _add(entry.path, "epub")
                            elif lower.endswith(".pdf"):
                                _add(entry.path, "pdf")
                            elif lower.endswith(".txt") and "_translated" in lower:
                                _add(entry.path, "txt")
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            _walk(entry.path, max_depth, depth + 1)
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass

    _walk(library_dir, max_depth=4)

    # Strictly honor the OUTPUT_DIRECTORY override: when set, we walk only
    # that directory for translation outputs / in-progress sources. Without
    # an override, we walk the default (app dir / CWD). This mirrors the
    # same rule applied by :func:`_find_in_progress_novels`.
    for root in _resolve_output_roots(config):
        _walk(root)

    # In-progress novels: source EPUBs with a translation_progress.json but no
    # compiled .epub yet. Strictly limited to the default / override output dirs.
    try:
        in_progress = _find_in_progress_novels(config)
    except Exception:
        logger.debug("In-progress scan failed: %s", traceback.format_exc())
        in_progress = []
    for ip in in_progress:
        norm = os.path.normpath(os.path.abspath(ip["path"]))
        if norm in seen:
            # Annotate the existing result with in-progress data — but never
            # for library-organized EPUBs. The Library shelf is supposed to
            # stay status-free unless a matching progress file was found in
            # the configured output root; _find_in_progress_novels already
            # guards against that, but be defensive here as well.
            for r in results:
                if os.path.normpath(os.path.abspath(r["path"])) == norm:
                    if r.get("in_library"):
                        break
                    r.update({
                        "is_in_progress": True,
                        "output_folder": ip["output_folder"],
                        "progress_file": ip["progress_file"],
                        "total_chapters": ip["total_chapters"],
                        "completed_chapters": ip["completed_chapters"],
                        "failed_chapters": ip["failed_chapters"],
                        "pending_chapters": ip["pending_chapters"],
                    })
                    break
            continue
        seen.add(norm)
        results.append(ip)

    results.sort(key=lambda r: r["mtime"], reverse=True)

    # Attach original source paths for files that were moved to Library.
    # v2 origins are split into raw/translated buckets; flatten them for
    # this lookup since we just want "original path for this basename".
    origins = _load_origins()
    flat_origins: dict[str, str] = {}
    for bucket in ("raw", "translated"):
        flat_origins.update(origins.get(bucket, {}) or {})
    for r in results:
        basename = os.path.basename(r["path"])
        if basename in flat_origins:
            r["original_path"] = flat_origins[basename]

    return results


SORT_DATE = "date"


SORT_NAME = "name"


SORT_SIZE = "size"


# File-format filter chips on the shared toolbar. ``FORMAT_ALL`` is the
# default (no filter); every other value maps to either the scanned
# row's ``type`` field (for library / compiled cards) or its
# ``workspace_kind`` field (for in-progress folder cards). See
# :meth:`EpubLibraryDialog._format_of_book` for the per-row mapping.
FORMAT_ALL = "all"


FORMAT_EPUB = "epub"


FORMAT_TXT = "txt"


FORMAT_PDF = "pdf"


FORMAT_HTML = "html"


FORMAT_IMAGE = "image"


SIZE_2XS = "2xs"


SIZE_XS = "xs"


SIZE_COMPACT = "compact"


SIZE_NORMAL = "normal"


SIZE_LARGE = "large"


SIZE_XL = "xl"


SIZE_2XL = "2xl"


SIZE_3XL = "3xl"


SIZE_4XL = "4xl"


SIZE_5XL = "5xl"


SIZE_6XL = "6xl"


_ALL_SIZES = [
    SIZE_2XS, SIZE_XS, SIZE_COMPACT, SIZE_NORMAL, SIZE_LARGE,
    SIZE_XL, SIZE_2XL, SIZE_3XL, SIZE_4XL, SIZE_5XL, SIZE_6XL,
]


# Title rendering: there's no hard character cap anymore — the title is
# rendered at ``title_size`` first; if it overflows ``title_max_h`` vertically
# we dynamically shrink the font down to ``title_min_size`` and only truncate
# with an ellipsis if even that's not enough. See :func:`_fit_title_text`.
# ``cover_h`` is ~15.5% taller than the raw aspect-match height of
# ``card_w`` (a 10% bump followed by another 5%) so the thumbnail
# area claims more of the card's footprint without the width
# changing — the image renders with more vertical room to fill
# (still via ``KeepAspectRatio``, so wider covers letterbox a
# little less) while leaving the text rows below it untouched.
_SIZE_PRESETS = {
    SIZE_2XS:     {"card_w": 78,  "cover_h": 115, "title_size": "7.5pt",  "title_min_size": "5.5pt", "title_max_h": 42,  "spacing": 2},
    SIZE_XS:      {"card_w": 92,  "cover_h": 136, "title_size": "8pt",    "title_min_size": "6pt",   "title_max_h": 46,  "spacing": 2},
    SIZE_COMPACT: {"card_w": 110, "cover_h": 162, "title_size": "8.5pt",  "title_min_size": "6.5pt", "title_max_h": 50,  "spacing": 3},
    SIZE_NORMAL:  {"card_w": 140, "cover_h": 203, "title_size": "9pt",    "title_min_size": "7pt",   "title_max_h": 55,  "spacing": 4},
    SIZE_LARGE:   {"card_w": 180, "cover_h": 260, "title_size": "9.5pt",  "title_min_size": "7.5pt", "title_max_h": 61,  "spacing": 5},
    SIZE_XL:      {"card_w": 230, "cover_h": 335, "title_size": "10pt",   "title_min_size": "8pt",   "title_max_h": 67,  "spacing": 6},
    SIZE_2XL:     {"card_w": 290, "cover_h": 422, "title_size": "10.5pt", "title_min_size": "8pt",   "title_max_h": 76,  "spacing": 8},
    SIZE_3XL:     {"card_w": 360, "cover_h": 520, "title_size": "11pt",   "title_min_size": "8.5pt", "title_max_h": 84,  "spacing": 10},
    SIZE_4XL:     {"card_w": 440, "cover_h": 635, "title_size": "11.5pt", "title_min_size": "9pt",   "title_max_h": 92,  "spacing": 12},
    SIZE_5XL:     {"card_w": 530, "cover_h": 762, "title_size": "12pt",   "title_min_size": "9.5pt", "title_max_h": 101, "spacing": 14},
    SIZE_6XL:     {"card_w": 630, "cover_h": 912, "title_size": "12.5pt", "title_min_size": "10pt",  "title_max_h": 109, "spacing": 16},
}


def _attach_cross_location_duplicates(completed: list[dict],
                                      output_rows: list[dict]) -> None:
    """Flag completed cards whose EPUB basename exists in two places.

    The ⚠ ``compiled_conflicts`` badge is normally populated by
    :func:`scan_output_folders` for *intra-folder* duplicates (two
    compiled artefacts in the same output workspace). It never fired
    when the duplication was *cross-location* — e.g. one EPUB in
    ``Library/Translated/Foo.epub`` and a second in
    ``<output_root>/Foo/Foo.epub`` — because those two paths come from
    different scans. This helper walks the merged completed list and
    the raw ``output_rows`` (pre-ghost-filter) and appends a conflict
    entry on every card that shares a basename with a compiled output
    at a different path.

    Modifies *completed* in place; does not return anything.
    """
    if not completed:
        return

    # Map lowercased basename → list of compiled abs paths that exist
    # on disk in any of the scanned output folders. ``output_rows``
    # carries ``compiled_output_path`` for every folder the output
    # scanner saw, including the ones that got ghost-filtered from
    # the merged completed list.
    by_basename: dict[str, list[str]] = {}
    for r in output_rows or []:
        cpath = r.get("compiled_output_path") or ""
        if not cpath or not os.path.isfile(cpath):
            continue
        key = os.path.basename(cpath).lower()
        by_basename.setdefault(key, []).append(cpath)
    # Also index the compiled basenames that actually survived into
    # the merged completed list so two non-ghost output-folder cards
    # with the same filename still flag each other.
    for r in completed:
        p = r.get("path", "") or ""
        if not p or not p.lower().endswith(".epub"):
            continue
        key = os.path.basename(p).lower()
        by_basename.setdefault(key, [])
        if p not in by_basename[key]:
            by_basename[key].append(p)
    if not by_basename:
        return

    for r in completed:
        p = r.get("path", "") or ""
        if not p or not p.lower().endswith(".epub"):
            continue
        key = os.path.basename(p).lower()
        siblings = by_basename.get(key, [])
        if len(siblings) < 2:
            continue
        self_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(p)))
        existing = list(r.get("compiled_conflicts") or [])
        # Seed with the basenames already recorded so we don't double
        # up when intra-folder conflicts ALSO exist on the same card.
        seen_labels = {lbl for lbl, _kind in existing}
        for other in siblings:
            other_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(other)))
            if other_abs == self_abs:
                continue
            # Tag the extra copy with its *parent directory* name so
            # the tooltip makes it obvious where the duplicate lives
            # (``Foo.epub (Library/Translated)`` vs. ``Foo.epub (Foo)``).
            parent = os.path.basename(os.path.dirname(other)) or "…"
            label = f"{os.path.basename(other)} ({parent})"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            existing.append((label, "epub"))
        if existing:
            r["compiled_conflicts"] = existing


def _unique_dest(directory: str, base_name: str) -> str:
    """``directory/base_name``, or ``base_name (2)``, ``(3)``... when that file exists.

    The Library's Keep Both rule (Organize; also Glossarion Mobile's FileBridge
    imports): only existing *files* count as taken, and the counter goes between
    the stem and the extension (``Book (2).epub``).
    """
    cand = os.path.join(directory, base_name)
    if not os.path.isfile(cand):
        return cand
    stem, ext = os.path.splitext(base_name)
    counter = 2
    while True:
        cand = os.path.join(directory, f"{stem} ({counter}){ext}")
        if not os.path.isfile(cand):
            return cand
        counter += 1


def _page_bounds(total, page, page_size) -> tuple:
    """``(start, end, page_count, page)`` of a 0-based *page* (``page_size <= 0`` = all).

    Shared by the Library pager and the Book Details chapter pager: an empty
    list or the All size gives one page; *page* is clamped into range.
    """
    total = max(0, int(total or 0))
    if total <= 0:
        return 0, 0, 1, 0
    if page_size <= 0:
        return 0, total, 1, 0
    page_count = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(int(page or 0), page_count - 1))
    start = page * page_size
    return start, min(total, start + page_size), page_count, page


def _page_label(page, page_count, start, end, total, page_size) -> str:
    """Library pager label: ``Page X / Y · a-b of N`` (``All N`` / ``0 of 0``)."""
    if total <= 0:
        detail = "0 of 0"
    elif page_size <= 0:
        detail = f"All {total}"
    else:
        detail = f"{start + 1}-{end} of {total}"
    return f"Page {page + 1} / {page_count} \u00b7 {detail}"


def _card_raw_title(book: dict) -> str:
    """Best-guess raw / source-language label for a library flash card.

    The "Raw titles" toolbar toggle maps to *this* function, and users
    expect it to reveal the original source *filename* on disk — the
    one they'd use to hunt the file down in Explorer — not a
    translated metadata title that happens to sit in ``metadata.json``.
    Resolution order (filename-first):

      1. Stem of ``raw_source_path`` — the resolved raw EPUB / PDF /
         TXT the scanner matched to this card. This is the authoritative
         source filename whenever it's available.
      2. Stem of ``original_path`` recorded in the origins registry
         (Library-organized files that were moved from elsewhere still
         know where they came from).
      3. ``folder_name`` — for in-progress workspaces this equals the
         raw EPUB's basename because output folders are scaffolded from
         the source filename.
      4. ``metadata.json`` ``original_title`` / ``raw_title`` /
         ``source_title`` when the translator stored one explicitly.
         (Kept as a fallback so books without a resolvable raw source
         still surface *something* source-language-y rather than
         reverting to the translated name.)
      5. Fall back to the card's default ``name``.
    """
    for path_key in ("raw_source_path", "original_path"):
        p = book.get(path_key) or ""
        if p:
            return os.path.splitext(os.path.basename(p))[0]
    fn = book.get("folder_name")
    if fn:
        return str(fn)
    md = book.get("metadata_json") or {}
    for key in ("original_title", "raw_title", "source_title"):
        val = md.get(key)
        if val:
            return str(val)
    return str(book.get("name", ""))


#: Card type badge (text, colour) per scanner ``type`` / workspace kind.
_CARD_TYPE_BADGES = {
    "epub":  ("\U0001f4d5EPUB",  "#6c63ff"),
    "pdf":   ("\U0001f4c4PDF",   "#e74c3c"),
    "txt":   ("\U0001f4d7TXT",   "#2ecc71"),
    "html":  ("\U0001f310HTML",  "#3498db"),
    "image": ("\U0001f5bc\ufe0fIMG", "#f39c12"),
    "in_progress": ("\U0001f4c1FOLDER", "#ffd166"),
}


def _card_type_badge(book: dict) -> tuple:
    """``(text, colour)`` of a card's type badge; folders show their workspace kind."""
    file_type = book.get("type", "epub")
    type_info = _CARD_TYPE_BADGES
    if file_type == "in_progress":
        kind = (book.get("workspace_kind") or "other").lower()
        badge_text, badge_color = type_info.get(kind, type_info["in_progress"])
    else:
        badge_text, badge_color = type_info.get(file_type, type_info["epub"])
    return badge_text, badge_color


def _card_size_text(size) -> str:
    """A card's size label: ``x.x MB`` from 1 MB up, else ``N KB``."""
    size_mb = size / (1024 * 1024)
    return f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size / 1024:.0f} KB"


def _card_progress_view(book: dict) -> dict | None:
    """A Library card's progress pill and cover ribbon as data (None = no pill).

    Cards that are not in progress, or whose state is ``completed``, show
    neither. The percentage is floored (216/217 never reads 100%). Keys:
    ``state``, ``done``, ``total``, ``pct``, ``pill_text``, ``pill_tooltip``,
    ``pill_color``, ``pill_background``, ``pill_border``, ``show_pct``,
    ``pct_text``, ``ribbon_text``, ``ribbon_background``.
    """
    if not book.get("is_in_progress"):
        return None
    total = int(book.get("total_chapters", 0) or 0)
    done = int(book.get("completed_chapters", 0) or 0)
    state = book.get("translation_state") or (
        "in_progress" if total else "not_started"
    )
    # 100% translated + compiled EPUB = completed \u2014 no pill;
    # those cards render plain on the Completed tab.
    if state == "completed":
        return None
    # Floor, not round: 216/217 must not read as 100%.
    pct = int((done * 100) // total) if total else 0
    show_pct = False
    if state == "outdated_progress":
        pill_text = "\u26a0 Outdated Progress file"
        pill_tooltip = (
            "The ``translation_progress.json`` in this "
            "workspace was written by an older version "
            "of Glossarion and can't be parsed \u2014 the "
            "card is pinned here so you can re-run the "
            "translation or remove the folder."
        )
        colors = ("#ffb347", "rgba(255, 179, 71, 0.18)", "#ffb347")
        ribbon_text = "OUTDATED PROGRESS"
        ribbon_bg = "rgba(255, 179, 71, 0.92)"
    elif state == "not_started":
        pill_text = "\U0001f195 Not started"
        pill_tooltip = "Imported into Library/Raw, translation not started yet."
        colors = ("#8ab4d0", "rgba(138, 180, 208, 0.15)", "#8ab4d0")
        ribbon_text = "NOT STARTED"
        ribbon_bg = "rgba(138, 180, 208, 0.92)"
    elif state == "ready_to_compile":
        pill_text = (
            f"\u2728 Ready to compile "
            f"({done}/{total})" if total
            else "\u2728 Ready to compile"
        )
        pill_tooltip = (
            "All chapters translated \u2014 compile the "
            "final EPUB to graduate this card to the "
            "Completed tab."
        )
        colors = ("#6ee8a0", "rgba(110, 232, 160, 0.16)", "#6ee8a0")
        ribbon_text = "READY TO COMPILE"
        # Darker mint so the ribbon reads crisply on a
        # light / pale cover without blowing out the
        # white label text next to it.
        ribbon_bg = "rgba(60, 170, 110, 0.95)"
    else:
        pill_text = f"\u23f3 {done}/{total}" if total else "\u23f3 In progress"
        pill_tooltip = (
            f"Translation in progress \u2014 {pct}% ({done}/{total} chapters)"
        )
        colors = ("#ffd166", "rgba(108, 99, 255, 0.18)", "#6c63ff")
        show_pct = bool(total)
        ribbon_text = "IN PROGRESS"
        ribbon_bg = "rgba(108, 99, 255, 0.92)"
    return {
        "state": state,
        "done": done,
        "total": total,
        "pct": pct,
        "pill_text": pill_text,
        "pill_tooltip": pill_tooltip,
        "pill_color": colors[0],
        "pill_background": colors[1],
        "pill_border": colors[2],
        "show_pct": show_pct,
        "pct_text": f"{pct}%",
        "ribbon_text": ribbon_text,
        "ribbon_background": ribbon_bg,
    }


# Compiled once: used by :func:`_extract_html_title_fast` to yank a chapter
# title out of a 32 KB HTML preview without spinning up BeautifulSoup for
# every file. ``re`` is a C extension that releases the GIL on each search,
# so this also lets a ThreadPoolExecutor actually get work done in parallel.
_RE_HTML_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


_RE_HTML_HEADING = re.compile(rb"<(h[1-6])[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


_RE_HTML_STRIP_TAGS = re.compile(rb"<[^>]+>")


_RE_HTML_WS = re.compile(rb"\s+")


def _extract_html_title_fast(raw: bytes) -> str:
    """Return the first ``<title>`` (or h1–h6) text from an HTML chunk.

    Drop-in replacement for a BeautifulSoup title extraction when you just
    need the document title. Roughly an order of magnitude faster on the
    chapter-list hot path for 400-entry spines, and because the underlying
    ``re`` calls release the GIL it's safe to call concurrently from a
    :class:`~concurrent.futures.ThreadPoolExecutor`.
    """
    if not raw:
        return ""
    try:
        from html import unescape
        for regex, group in ((_RE_HTML_TITLE, 1), (_RE_HTML_HEADING, 2)):
            m = regex.search(raw)
            if not m:
                continue
            inner = _RE_HTML_STRIP_TAGS.sub(b" ", m.group(group))
            inner = _RE_HTML_WS.sub(b" ", inner).strip()
            if inner:
                return unescape(inner.decode("utf-8", errors="replace")).strip()
    except Exception:
        logger.debug("Fast title parse failed: %s", traceback.format_exc())
    return ""


def _parse_epub_details(epub_path: str, parse_chapter_titles: bool = True) -> dict:
    """Extract OPF metadata, spine order and per-chapter raw titles.

    Returns a dict shaped roughly like the OPF DC schema plus a ``chapters``
    list of ``{'href', 'filename', 'title'}``. All fields are best-effort and
    may be empty strings/lists on failure.

    When *parse_chapter_titles* is False, per-chapter HTML is not opened
    and chapter titles fall back to filename-derived labels. This is the
    fast path :class:`_BookDetailsLoader` uses for its preview pass so
    the cover + metadata render immediately while the real titles are
    parsed in the background.
    """
    import zipfile
    import posixpath
    from xml.etree import ElementTree as ET
    from html import unescape

    details = {
        "title": "",
        "authors": [],
        "publisher": "",
        "language": "",
        "date": "",
        "description": "",
        "subjects": [],
        "identifier": "",
        "chapters": [],
    }

    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            names_set = set(names)

            opf_path = find_epub_opf_member(zf)
            if not opf_path or opf_path not in names_set:
                return details

            opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
            tree = ET.fromstring(opf_xml)
            opf_dir = posixpath.dirname(opf_path)

            DC = "http://purl.org/dc/elements/1.1/"
            OPF = "http://www.idpf.org/2007/opf"

            def _dc(tag: str) -> list[str]:
                return [
                    (el.text or "").strip()
                    for el in tree.findall(f".//{{{DC}}}{tag}")
                    if (el.text or "").strip()
                ]

            titles = _dc("title")
            details["title"] = titles[0] if titles else ""
            details["authors"] = _dc("creator")
            publishers = _dc("publisher")
            details["publisher"] = publishers[0] if publishers else ""
            languages = _dc("language")
            details["language"] = languages[0] if languages else ""
            dates = _dc("date")
            details["date"] = dates[0] if dates else ""
            descriptions = _dc("description")
            details["description"] = unescape(descriptions[0]) if descriptions else ""
            details["subjects"] = _dc("subject")
            identifiers = _dc("identifier")
            details["identifier"] = identifiers[0] if identifiers else ""

            # Build a manifest id -> (href, media_type) lookup so we can
            # resolve spine itemrefs into concrete chapter files.
            manifest: dict[str, tuple[str, str]] = {}
            for item_el in tree.findall(f".//{{{OPF}}}item"):
                item_id = item_el.get("id", "")
                href = item_el.get("href", "")
                media = item_el.get("media-type", "")
                if not item_id or not href:
                    continue
                full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
                manifest[item_id] = (full_href, media)

            spine_el = tree.find(f".//{{{OPF}}}spine")
            ordered_hrefs: list[tuple[str, str]] = []  # [(id, href)]
            if spine_el is not None:
                for itemref in spine_el.findall(f"{{{OPF}}}itemref"):
                    idref = itemref.get("idref")
                    if idref and idref in manifest:
                        ordered_hrefs.append((idref, manifest[idref][0]))

            # Parse each chapter file for a title. Keep it cheap: only peek at
            # the first ~32KB which is more than enough for <title>/<h1>. The
            # per-chapter HTML scan is the dominant cost for large spines
            # (~399 chapters), so callers that only need metadata skip it
            # via ``parse_chapter_titles=False`` and use filename fallbacks.
            #
            # When we DO want chapter titles we do two optimizations:
            #   1. Read every chapter's first 32 KB serially from the zip
            #      (``zipfile.ZipFile`` is not thread-safe for concurrent
            #      reads) — this is fast since it's just stream decompression.
            #   2. Dispatch the actual title extraction to a thread pool
            #      using :func:`_extract_html_title_fast`. That function is
            #      built on the C-backed ``re`` module, so the GIL is
            #      released and we actually get parallel speedup on the
            #      CPU-bound parse step.
            chap_raw: dict[str, bytes] = {}
            if parse_chapter_titles:
                for idref, href in ordered_hrefs:
                    media_type = manifest.get(idref, ("", ""))[1]
                    if media_type and ("html" not in media_type.lower()
                                       and "xhtml" not in media_type.lower()):
                        continue
                    if href in names_set and href not in chap_raw:
                        try:
                            chap_raw[href] = zf.read(href)[:32_768]
                        except Exception:
                            chap_raw[href] = b""

            titles_by_href: dict[str, str] = {}
            if chap_raw:
                try:
                    max_workers = _reader_worker_count(len(chap_raw))
                    if max_workers <= 1:
                        for h, data in chap_raw.items():
                            t = _extract_html_title_fast(data)
                            if t:
                                titles_by_href[h] = t
                    else:
                        hrefs = list(chap_raw.keys())
                        datas = [chap_raw[h] for h in hrefs]
                        with ThreadPoolExecutor(max_workers=max_workers) as pool:
                            for h, t in zip(
                                    hrefs,
                                    pool.map(_extract_html_title_fast, datas)):
                                if t:
                                    titles_by_href[h] = t
                except Exception:
                    logger.debug("Parallel title parse failed, falling back: %s",
                                 traceback.format_exc())
                    for h, data in chap_raw.items():
                        t = _extract_html_title_fast(data)
                        if t:
                            titles_by_href[h] = t

            chapters = []
            for idref, href in ordered_hrefs:
                media_type = manifest.get(idref, ("", ""))[1]
                if media_type and ("html" not in media_type.lower() and "xhtml" not in media_type.lower()):
                    # Non-text spine item — still include so index matches.
                    chapters.append({"href": href, "filename": os.path.basename(href),
                                     "title": os.path.splitext(os.path.basename(href))[0]})
                    continue
                title = titles_by_href.get(href, "")
                if not title:
                    title = os.path.splitext(os.path.basename(href))[0]
                    title = title.replace("_", " ").replace("-", " ").strip() or title
                chapters.append({"href": href, "filename": os.path.basename(href), "title": title})
            details["chapters"] = chapters
    except Exception:
        logger.debug("EPUB details parse failed: %s", traceback.format_exc())

    return details


def _read_translated_chapter_title(path: str) -> str:
    """Extract a translated-chapter title from a response_*.html file.

    Uses the fast regex-based title extractor instead of BeautifulSoup so
    that batched calls from :class:`_BookDetailsLoader` finish in tens of
    milliseconds rather than seconds on a 400-chapter output folder.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(32_768)
        return _extract_html_title_fast(raw)
    except Exception:
        logger.debug("Translated title parse failed for %s: %s", path, traceback.format_exc())
    return ""


_CHAPTER_PRIMARY_STYLES = {
    "raw": "color: #c8cbe0; font-size: 10pt; font-weight: bold;",
    "translated": "color: #e0e0e0; font-size: 10pt; font-weight: bold;",
}


_CHAPTER_BADGE_STYLES = {
    "completed": (
        "color: #7ec87e; background: rgba(126, 200, 126, 0.12);"
        " border: 1px solid #7ec87e; border-radius: 10px;"
        " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
    ),
    "failed": (
        "color: #ff9e6d; background: rgba(255, 158, 109, 0.12);"
        " border: 1px solid #ff9e6d; border-radius: 10px;"
        " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
    ),
    "in_progress": (
        "color: #ffd166; background: rgba(255, 209, 102, 0.12);"
        " border: 1px solid #ffd166; border-radius: 10px;"
        " padding: 2px 10px; font-size: 8pt; font-weight: bold;"
    ),
    "pending": (
        "color: #7a8599; background: #2a2a3e;"
        " border: 1px solid #3a3a5e; border-radius: 10px;"
        " padding: 2px 10px; font-size: 8pt;"
    ),
}


_CHAPTER_BADGE_TEXT = {
    "completed": "\u2714 Translated",
    "failed": "\u26a0 Failed",
    "qa_failed": "\u26a0 QA failed",
    "in_progress": "\u23f3 Working",
    "pending": "Pending",
}


def _prepare_chapter_row_spec(info: dict, show_raw_title: bool = False) -> dict:
    """Build the pure-Python display model for a chapter row."""
    info = dict(info or {})
    status = info.get("status", "") or ""
    translated = info.get("translated_title") or ""
    raw = info.get("raw_title") or ""
    filename = info.get("filename", "") or ""
    chunk_status_text = str(info.get("chunk_status_text") or "").strip()
    filename_display = (
        f"{filename} · {chunk_status_text}" if chunk_status_text else filename
    )

    if show_raw_title:
        primary_text = raw or filename
        primary_class = "raw"
    elif translated and status == "completed":
        primary_text = translated
        primary_class = "translated"
    else:
        primary_text = raw or filename
        primary_class = "raw"

    primary_tooltip = ""
    if show_raw_title and translated and translated != raw:
        primary_tooltip = f"Translated: {translated}"
    elif (
        not show_raw_title
        and translated
        and status == "completed"
        and raw
        and raw != translated
    ):
        primary_tooltip = f"Raw: {raw}"

    badge_text = ""
    badge_style = ""
    if not bool(info.get("is_gallery")):
        badge_key = status
        if status == "qa_failed":
            badge_key = "failed"
        badge_text = _CHAPTER_BADGE_TEXT.get(status, "")
        badge_style = _CHAPTER_BADGE_STYLES.get(badge_key, "")
        chunk_summary = info.get("chunk_summary")
        if isinstance(chunk_summary, dict) and chunk_summary.get("failed"):
            failed = int(chunk_summary.get("failed") or 0)
            total = int(chunk_summary.get("total") or 0)
            badge_text = f"⚠ {failed}/{total} chunks"
            badge_style = _CHAPTER_BADGE_STYLES.get("failed", "")

    return {
        "info": info,
        "primary_text": primary_text,
        "primary_class": primary_class,
        "primary_style": _CHAPTER_PRIMARY_STYLES.get(primary_class, ""),
        "primary_tooltip": primary_tooltip,
        "filename": filename_display,
        "badge_text": badge_text,
        "badge_style": badge_style,
    }


_EDITABLE_BOOK_METADATA_FIELDS = (
    "title",
    "creator",
    "publisher",
    "language",
    "date",
    "description",
    "subject",
)


def _metadata_subject_values(value) -> list[str]:
    """Normalize a metadata subject value into ordered, unique tags."""
    values: list[str] = []
    seen: set[str] = set()

    def add(item) -> None:
        if isinstance(item, (list, tuple, set, frozenset)):
            for child in item:
                add(child)
            return
        text = str(item or "").strip()
        if not text:
            return
        if "#" in text:
            parts = [
                match.group(1).strip().strip(",;")
                for match in re.finditer(r"#([^#]+)", text)
            ]
        else:
            parts = [
                part.strip()
                for part in re.split(r"[,;\n]+", text)
            ]
        for part in parts:
            if not part:
                continue
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(part)

    add(value)
    return values


def _merge_manual_metadata_edits(
    existing: dict,
    edits: dict,
    source_values: dict | None = None,
) -> tuple[dict, set[str]]:
    """Merge user-edited display values without discarding metadata fields."""
    merged = dict(existing or {})
    source_values = source_values or {}
    changed: set[str] = set()

    def normalized(field: str, value):
        if field == "subject":
            return _metadata_subject_values(value)
        return str(value or "").strip()

    def stored_value(field: str, value):
        if field != "subject":
            return normalized(field, value)
        subjects = normalized(field, value)
        if len(subjects) == 1:
            return subjects[0]
        return subjects

    for field in _EDITABLE_BOOK_METADATA_FIELDS:
        if field not in edits:
            continue
        new_value = stored_value(field, edits[field])
        if normalized(field, merged.get(field)) == normalized(field, new_value):
            continue

        original_key = (
            "original_title" if field == "title" else f"original_{field}"
        )
        original_value = source_values.get(field)
        if not normalized(field, original_value):
            original_value = merged.get(field)
        if (
            original_key not in merged
            and normalized(field, original_value)
            and normalized(field, original_value) != normalized(field, new_value)
        ):
            merged[original_key] = stored_value(field, original_value)

        merged[field] = new_value
        translated_key = (
            "title_translated"
            if field == "title"
            else f"{field}_translated"
        )
        # A manual value is authoritative output metadata. Marking it complete
        # prevents a later chapter compile from silently translating over it.
        merged[translated_key] = True
        changed.add(field)

    return merged, changed


class _MetadataEditError(Exception):
    """metadata.json could not be read or saved (message = the dialog text)."""


def _metadata_changed_values(initial_values: dict, current: dict) -> dict:
    """Fields of *current* that differ from *initial_values* (tags compared as tag sets)."""
    initial_values = dict(initial_values or {})
    changed = {}
    for field, value in current.items():
        if field == "subject":
            old_value = _metadata_subject_values(
                initial_values.get(field)
            )
            new_value = _metadata_subject_values(value)
        else:
            old_value = str(
                initial_values.get(field) or ""
            ).strip()
            new_value = str(value or "").strip()
        if old_value != new_value:
            changed[field] = value
    return changed


class DualScanMixin:
    """``_DualScannerThread``: both scans, organized-folder dedupe, origins pairing, state
    inheritance and cross-location conflicts. Needs ``self._config`` and
    ``self.scan_finished.emit(in_progress, completed)``.
    """

    def run(self):
        output_rows = []
        library_rows = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            scan_jobs = {
                pool.submit(scan_output_folders, self._config): "output",
                pool.submit(scan_library_completed, self._config): "library",
            }
            for future in as_completed(scan_jobs):
                scan_kind = scan_jobs[future]
                try:
                    rows = future.result()
                except Exception:
                    logger.debug("%s scan failed: %s",
                                 scan_kind.title(), traceback.format_exc())
                    rows = []
                if scan_kind == "output":
                    output_rows = rows
                else:
                    library_rows = rows
        self._merge_scan_rows(output_rows, library_rows)

    def _merge_scan_rows(self, output_rows, library_rows):
        """Merge the two scans (dedupe, origins pairing, state inheritance)."""
        completed_from_output, in_progress = split_output_folders_by_status(output_rows)

        # Post-organize dedup: any output folder whose compiled EPUB was
        # moved into Library/Translated shows up in TWO scans:
        #   * ``scan_library_completed`` — the new Library/Translated file.
        #   * ``scan_output_folders``    — the owning folder (no compiled
        #     EPUB anymore but progress=100% still classifies it as
        #     ``completed``).
        # The origins registry tells us which output folders were the
        # source of a library-filed EPUB; we skip those ghost
        # folder-only rows so the user sees a single card per book.
        try:
            origins = _load_origins()
            trans_map = origins.get("translated", {}) or {}
        except Exception:
            trans_map = {}
        organized_folders: set[str] = set()
        for orig_path in trans_map.values():
            if not orig_path:
                continue
            folder = os.path.dirname(str(orig_path))
            if folder:
                organized_folders.add(
                    os.path.normcase(os.path.normpath(
                        os.path.abspath(folder)))
                )

        # Title-based workspace index as a fallback when the origins
        # registry doesn't link a Library/Translated EPUB to its
        # originating workspace. Pulls candidate keys from the
        # workspace folder name, the raw source stem, AND the
        # workspace's ``metadata.json`` title fields — so a workspace
        # whose folder / raw source is named in the *source* language
        # (e.g. "… RoFan Who Is …") still matches its compiled EPUB
        # named in the *translated* language (e.g. "… Romance Fantasy")
        # through the ``metadata_json['title']`` the translator wrote
        # at compile time. Without this fallback the duplicate card
        # appears on BOTH the In Progress tab (workspace row) and
        # the Completed tab (library row) because neither the ghost
        # filter nor the state-inheritance block below can link them.
        #
        # All keys pass through :func:`_norm_book_key` so Windows
        # filename mangling (stripped trailing ``.``, case drift,
        # whitespace collapse) can't desync the two sides of the
        # comparison.
        workspace_by_key: dict[str, dict] = {}
        for ws in output_rows:
            ws_folder = ws.get("output_folder") or ""
            if not ws_folder:
                continue
            candidates: set[str] = set()
            fn = ws.get("folder_name") or os.path.basename(ws_folder)
            if fn:
                candidates.add(_norm_book_key(
                    os.path.splitext(fn)[0]))
            raw = ws.get("raw_source_path") or ""
            if raw:
                candidates.add(_norm_book_key(
                    os.path.splitext(os.path.basename(raw))[0]))
            md = ws.get("metadata_json") or {}
            if isinstance(md, dict):
                for md_key in ("title", "original_title",
                               "translated_title", "raw_title",
                               "source_title", "english_title"):
                    val = md.get(md_key)
                    if isinstance(val, str) and val.strip():
                        candidates.add(_norm_book_key(val))
            for key in candidates:
                if key:
                    workspace_by_key.setdefault(key, ws)

        # Library-side index keyed by the same normalization so the
        # ghost filter + inheritance loop can consult it without
        # recomputing per-row. Each library EPUB contributes MULTIPLE
        # normalized keys so the pairing can land when ANY of them
        # intersects a workspace key:
        #   • filename stem (translated title, post NTFS sanitize)
        #   • ``dc:title`` / ``dc:alternative`` from the EPUB OPF
        #   • ``calibre:original_title`` meta element — the single
        #     most useful signal for raws whose folder is named in
        #     the source language, because the translator writes
        #     the raw title here at compile time.
        library_by_key: dict[str, dict] = {}
        for lr in library_rows:
            if not lr.get("in_library"):
                continue
            lib_path = lr.get("path", "") or ""
            if not lib_path:
                continue
            keys: set[str] = set()
            stem = os.path.splitext(os.path.basename(lib_path))[0]
            k = _norm_book_key(stem)
            if k:
                keys.add(k)
            for opf_title in _extract_epub_titles(lib_path):
                k = _norm_book_key(opf_title)
                if k:
                    keys.add(k)
            for k in keys:
                library_by_key.setdefault(k, lr)

        # Extend the ghost set with workspaces whose title-key
        # matches a Library/Translated entry. The library row will
        # represent the book (with inherited state via the block
        # below) so the workspace-side row must drop off the In
        # Progress tab to avoid the duplicate card.
        #
        # Also persist an authoritative link into ``library_origins.txt``
        # for every pair we discover: ``translated[lib_basename] =
        # <workspace>/<lib_basename>`` (restore target for Undo Move)
        # and ``pairs[lib_basename] = <raw_basename>`` when the raw
        # sits inside ``Library/Raw`` (so
        # :func:`_find_raw_source_for_library_epub` takes the fast
        # origins path on the next call). The user explicitly asked
        # for origins to be updated as we discover pairings — without
        # this, ``_undo_organize_prompt`` has nothing to undo for
        # cards that landed in Library/Translated via any route
        # other than the Organize button.
        origins_dirty = False
        if workspace_by_key and library_by_key:
            trans_map_mut = dict(trans_map) if isinstance(trans_map, dict) else {}
            pair_map_mut = dict(origins.get("pairs", {}) or {}) if isinstance(origins, dict) else {}
            raw_dir_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_raw_dir())))
            for key, ws in workspace_by_key.items():
                lr = library_by_key.get(key)
                if not lr:
                    continue
                ws_folder = ws.get("output_folder") or ""
                if ws_folder:
                    organized_folders.add(
                        os.path.normcase(os.path.normpath(
                            os.path.abspath(ws_folder)))
                    )
                lib_basename = os.path.basename(lr.get("path", "") or "")
                if not lib_basename or not ws_folder:
                    continue
                existing_entry = trans_map_mut.get(lib_basename) or ""
                desired_entry = os.path.join(ws_folder, lib_basename)
                try:
                    same = bool(existing_entry) and (
                        os.path.normcase(os.path.normpath(
                            os.path.abspath(existing_entry)))
                        == os.path.normcase(os.path.normpath(
                            os.path.abspath(desired_entry)))
                    )
                except Exception:
                    same = False
                if not same:
                    trans_map_mut[lib_basename] = desired_entry
                    origins_dirty = True
                # A ``ready_to_compile`` workspace whose compiled EPUB
                # now lives in ``Library/Translated`` is genuinely
                # completed — the compile step already ran, the
                # artefact just got organized out. Upgrade the
                # in-memory state so:
                #   * the library card stays on the Completed tab
                #     (inheritance block below only rewrites the
                #     library state when the workspace is
                #     non-completed, so a "completed" ws state
                #     keeps the library row in place)
                #   * the workspace row doesn't show a stale
                #     "Ready to compile" pill on the In Progress tab
                #     (it'll be ghost-filtered out anyway, but the
                #     state has to be right for the brief window
                #     where both tabs still contain it).
                #   * ``has_compiled_output`` / ``compiled_output_path``
                #     now point at the library-filed EPUB so callers
                #     that consult those fields resolve to the real
                #     compiled artefact.
                if ws.get("translation_state") == "ready_to_compile":
                    lib_path = lr.get("path", "") or ""
                    ws["translation_state"] = "completed"
                    ws["is_in_progress"] = False
                    if lib_path:
                        ws["has_compiled_output"] = True
                        ws["compiled_output_path"] = lib_path
                        ws["compiled_output_kind"] = "epub"
                        ws["has_output_epub"] = True
                        ws["output_epub_path"] = lib_path
                raw_src = ws.get("raw_source_path") or ""
                if raw_src and os.path.isfile(raw_src):
                    try:
                        raw_parent = os.path.normcase(os.path.normpath(
                            os.path.abspath(os.path.dirname(raw_src))))
                    except Exception:
                        raw_parent = ""
                    if raw_parent == raw_dir_abs:
                        raw_basename = os.path.basename(raw_src)
                        if (raw_basename
                                and pair_map_mut.get(lib_basename) != raw_basename):
                            pair_map_mut[lib_basename] = raw_basename
                            origins_dirty = True
            if origins_dirty:
                try:
                    origins["translated"] = trans_map_mut
                    origins["pairs"] = pair_map_mut
                    _save_origins(origins)
                    trans_map = trans_map_mut
                except Exception:
                    logger.debug(
                        "origins auto-update failed: %s",
                        traceback.format_exc())

        # Unified state upgrade: every ``ready_to_compile`` workspace
        # whose folder is in ``organized_folders`` is actually
        # COMPLETED — its compiled EPUB just lives in
        # ``Library/Translated`` instead of alongside the progress
        # file. Runs regardless of whether the link came from
        # origins.txt, the title-key fallback, or a mix, so the
        # "ready to compile" check now consults the library shelf
        # uniformly via the combined ghost set.
        #
        # Without this upgrade, the inheritance block below would
        # overwrite the library card's state with ``ready_to_compile``
        # and yank it onto the In Progress tab — exactly the
        # duplicate-card / wrong-state behaviour the user hit when
        # organizing an EPUB in then out of Library/Translated.
        if organized_folders:
            trans_dir_abs = get_library_translated_dir()
            # Build a reverse lookup from workspace folder → library
            # path so we can populate ``compiled_output_path`` on the
            # upgraded workspace without another disk scan.
            folder_to_lib_path: dict[str, str] = {}
            for lib_basename, orig_path in (trans_map or {}).items():
                if not orig_path:
                    continue
                try:
                    parent = os.path.normcase(os.path.normpath(
                        os.path.abspath(os.path.dirname(str(orig_path)))))
                except Exception:
                    continue
                candidate = os.path.join(trans_dir_abs, lib_basename)
                if os.path.isfile(candidate):
                    folder_to_lib_path.setdefault(parent, candidate)
            for ws in output_rows:
                if ws.get("translation_state") != "ready_to_compile":
                    continue
                ws_folder = ws.get("output_folder") or ""
                if not ws_folder:
                    continue
                fk = os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
                if fk not in organized_folders:
                    continue
                ws["translation_state"] = "completed"
                ws["is_in_progress"] = False
                lib_path = folder_to_lib_path.get(fk, "")
                if lib_path:
                    ws["has_compiled_output"] = True
                    ws["compiled_output_path"] = lib_path
                    ws["compiled_output_kind"] = "epub"
                    ws["has_output_epub"] = True
                    ws["output_epub_path"] = lib_path

        def _is_organized_ghost(row: dict) -> bool:
            """True when *row* is an output-folder scan result whose
            compiled EPUB has been organized into Library/Translated."""
            folder = row.get("output_folder") or ""
            if not folder:
                return False
            fk = os.path.normcase(os.path.normpath(
                os.path.abspath(folder)))
            return fk in organized_folders

        # Merge: library entries first (they're the curated shelf), then
        # output-folder completions that aren't already represented by path.
        seen_paths: set[str] = set()
        completed: list[dict] = []
        for r in library_rows:
            key = os.path.normcase(os.path.normpath(os.path.abspath(r["path"])))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            completed.append(r)
        for r in completed_from_output:
            key = os.path.normcase(os.path.normpath(os.path.abspath(r["path"])))
            if key in seen_paths:
                continue
            if _is_organized_ghost(r):
                continue
            seen_paths.add(key)
            completed.append(r)
        completed.sort(key=lambda r: r["mtime"], reverse=True)

        # Apply the same ghost-filter to the In Progress tab so a
        # post-organize folder that somehow slips through the "completed"
        # classification (partial undo, mismatched progress file, etc.)
        # doesn't linger as a phantom in-progress card either.
        in_progress = [r for r in in_progress if not _is_organized_ghost(r)]

        # Library-vs-workspace state inheritance via ``origins.txt``
        # (primary) + filename fallback (secondary). NO dedupe —
        # every card stays visible exactly once. Each
        # ``Library/Translated`` entry inherits the translation_state
        # + progress numbers of its owning workspace when they can
        # be linked, regardless of whether that workspace was
        # ghost-filtered above. That fixes:
        #   * the "99%% but shown as Completed" regression after
        #     organize: the ghost filter takes the workspace *row*
        #     off the In Progress list, and this block re-routes the
        #     library row (which replaces it) onto the In Progress
        #     tab while the underlying translation is still at 99 %%.
        #   * the "ready_to_compile shows up on BOTH tabs" bug: when
        #     origins doesn't link the pair, the filename fallback
        #     below still matches them so the library row inherits
        #     ``ready_to_compile`` and moves to In Progress (with
        #     the workspace row ghost-filtered above).
        #
        # ``trans_map`` maps ``library_basename → original_workspace_path``;
        # the parent dir of that path is the workspace folder. Rows
        # without a ``trans_map`` record fall through to the
        # title-based ``workspace_by_key`` lookup built above.
        workspace_by_folder: dict[str, dict] = {}
        for ws in output_rows:
            ws_folder = ws.get("output_folder") or ""
            if not ws_folder:
                continue
            workspace_by_folder[
                os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
            ] = ws

        if workspace_by_folder and (trans_map or workspace_by_key):
            for r in completed:
                if not r.get("in_library"):
                    continue
                lib_basename = os.path.basename(r.get("path", "") or "")
                ws_row = None
                # 1. Origins-based link (authoritative).
                if trans_map:
                    orig_path = trans_map.get(lib_basename)
                    if orig_path:
                        orig_folder = os.path.dirname(str(orig_path))
                        if orig_folder:
                            origin_key = os.path.normcase(os.path.normpath(
                                os.path.abspath(orig_folder)))
                            ws_row = workspace_by_folder.get(origin_key)
                # 2. Title-key fallback — mirrors the ghost-set
                #    extension above so both sides of the dedup
                #    (workspace row off In Progress, library row
                #    moved from Completed to In Progress) kick in
                #    for the same pair of cards. The key covers
                #    folder name, raw source stem, and the
                #    metadata.json title fields so raws named in the
                #    source language still match their English-
                #    titled compiled EPUBs.
                if not ws_row:
                    lib_key = _norm_book_key(
                        os.path.splitext(lib_basename)[0])
                    if lib_key:
                        ws_row = workspace_by_key.get(lib_key)
                if not ws_row:
                    continue
                ws_state = ws_row.get("translation_state") or ""
                # Only inherit when the workspace is actually NOT
                # completed — otherwise a finished book would get
                # yanked onto the In Progress tab with stale
                # progress numbers. A completed workspace + library
                # file is the normal post-organize state; leave the
                # library card alone there.
                if ws_state == "completed" or not ws_state:
                    continue
                r["translation_state"] = ws_state
                r["is_in_progress"] = True
                r["total_chapters"] = ws_row.get("total_chapters", 0)
                r["completed_chapters"] = ws_row.get(
                    "completed_chapters", 0)
                r["failed_chapters"] = ws_row.get("failed_chapters", 0)
                r["pending_chapters"] = ws_row.get(
                    "pending_chapters", 0)
                r["output_folder"] = ws_row.get("output_folder", "")
                r["progress_file"] = ws_row.get("progress_file", "")
                # Track the raw source so the In Progress card can
                # render a cover / resolve the raw-open actions
                # even though the row itself lives in Library/Translated.
                raw_src_from_ws = ws_row.get("raw_source_path") or ""
                if raw_src_from_ws:
                    r["raw_source_path"] = raw_src_from_ws
                # Keep ``missing_raw_file`` honest for inherited
                # library rows. If after inheritance the card
                # still has no resolvable raw source (neither its
                # own library-side lookup nor the workspace
                # produced one), flip the flag on so the
                # \u26a0 \"missing raw\" badge surfaces. Otherwise
                # (raw path present on either side) keep the flag
                # off so the badge doesn't render spuriously.
                r["missing_raw_file"] = not bool(
                    r.get("raw_source_path") or "")

            # Move any library rows that inherited a non-completed
            # state over to the In Progress tab, and make sure we
            # don't end up with a duplicate workspace row for the
            # SAME output folder in the in_progress bucket (can
            # happen when the ghost filter didn't kick in).
            moved_in_progress_folders: set[str] = set()
            still_completed: list[dict] = []
            for r in completed:
                state = r.get("translation_state")
                if state and state != "completed" and r.get("in_library"):
                    in_progress.append(r)
                    of = r.get("output_folder") or ""
                    if of:
                        moved_in_progress_folders.add(
                            os.path.normcase(os.path.normpath(
                                os.path.abspath(of)))
                        )
                else:
                    still_completed.append(r)
            completed = still_completed

            if moved_in_progress_folders:
                in_progress = [
                    r for r in in_progress
                    if not r.get("in_library")
                    and os.path.normcase(os.path.normpath(
                        os.path.abspath(r.get("output_folder", ""))))
                    not in moved_in_progress_folders
                    or r.get("in_library")
                ]

        # Cross-location duplicate detection: if a ``Library/Translated``
        # entry has the same basename as a compiled EPUB still sitting in
        # an output folder, surface it on the library card's ⚠ badge.
        # This catches the case where the user organized the EPUB into
        # the library but the original (or a re-compiled copy) still
        # lives in the output folder — without this check, the ghost
        # filter above silently dropped the output row and the user
        # never saw any indication that two physical copies exist.
        _attach_cross_location_duplicates(completed, output_rows)

        self.scan_finished.emit(in_progress, completed)


class LibraryDeleteMixin:
    """``_LibraryDeleteThread``: parallel delete of ``(label, path, is_folder)`` targets.
    Needs ``self._targets``, ``self.progress.emit(done, total, label)`` and
    ``self.delete_finished.emit(results)``.
    """

    @staticmethod
    def _delete_one(label: str, pth: str, is_folder: bool) -> tuple:
        try:
            if is_folder:
                shutil.rmtree(pth)
            else:
                os.remove(pth)
            return (label, pth, is_folder, True, "")
        except Exception as exc:
            return (label, pth, is_folder, False, str(exc))

    def run(self):
        total = len(self._targets)
        if total <= 0:
            self.delete_finished.emit([])
            return

        results: list[tuple] = []
        done = 0
        # Disk deletion is I/O-heavy. A small cap gives real parallelism
        # without turning a spinning disk or network share into a traffic jam.
        max_workers = min(4, total)
        try:
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="LibraryDelete",
            ) as pool:
                future_map = {
                    pool.submit(self._delete_one, label, pth, is_folder):
                    (label, pth, is_folder)
                    for label, pth, is_folder in self._targets
                }
                for future in as_completed(future_map):
                    label, pth, is_folder = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = (label, pth, is_folder, False, str(exc))
                    results.append(result)
                    done += 1
                    self.progress.emit(done, total, label)
        except Exception:
            logger.error("Parallel library delete failed: %s",
                         traceback.format_exc())
            seen = {
                os.path.normcase(os.path.normpath(os.path.abspath(r[1])))
                for r in results if len(r) > 1
            }
            for label, pth, is_folder in self._targets:
                try:
                    key = os.path.normcase(os.path.normpath(
                        os.path.abspath(pth)))
                except Exception:
                    key = pth
                if key not in seen:
                    results.append(
                        (label, pth, is_folder, False,
                         "Delete worker stopped before this item finished.")
                    )
        self.delete_finished.emit(results)


class RawScanMixin:
    """``_RawScanWorker``: walk a folder for raw candidates and pair workspaces (exact /
    fuzzy). Needs ``_folder``, ``_suffixes``, ``_tracking``, ``_books``, ``_mode``,
    ``_threshold``, ``_prewalked``, ``_cancelled`` and ``self.results.emit(...)``.
    """

    def _classify(self, root_dir: str,
                  files: list[str]) -> list[tuple[str, str]]:
        """Filter + normalize a single directory's files (executor task).

        Pure function on the inputs plus ``self._suffixes`` /
        ``self._tracking`` — no shared mutable state, so it's safe
        to run across multiple pool workers concurrently.
        """
        out: list[tuple[str, str]] = []
        suffixes = self._suffixes
        tracking = self._tracking
        for name in files:
            if self._cancelled:
                break
            lower = name.lower()
            if not lower.endswith(suffixes):
                continue
            if lower in tracking:
                continue
            if name.startswith("."):
                continue
            stem = os.path.splitext(name)[0]
            key = _norm_book_key(stem)
            if not key:
                continue
            out.append((key, os.path.join(root_dir, name)))
        return out

    def _walk(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        folder = self._folder
        if not folder or not os.path.isdir(folder):
            return candidates
        try:
            futures: list = []
            with ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="ScanForRaw",
            ) as executor:
                for root_dir, _dirs, files in os.walk(folder):
                    if self._cancelled:
                        break
                    if not files:
                        continue
                    futures.append(executor.submit(
                        self._classify, root_dir, list(files)))
                for fut in futures:
                    if self._cancelled:
                        break
                    try:
                        candidates.extend(fut.result())
                    except Exception:
                        logger.debug(
                            "ScanForRaw classify task failed: %s",
                            traceback.format_exc())
        except (PermissionError, OSError) as exc:
            logger.debug(
                "ScanForRaw folder walk failed for %s: %s", folder, exc)
        except Exception:
            logger.debug(
                "ScanForRaw worker crashed: %s", traceback.format_exc())
        return candidates

    @staticmethod
    def _book_keys(book: dict) -> list[str]:
        keys: set[str] = set()
        fn = book.get("folder_name") or os.path.basename(
            book.get("output_folder") or book.get("path") or "")
        if fn:
            keys.add(_norm_book_key(os.path.splitext(fn)[0]))
        md = book.get("metadata_json") or {}
        if isinstance(md, dict):
            for md_key in ("title", "original_title",
                           "translated_title", "raw_title",
                           "source_title", "english_title"):
                val = md.get(md_key)
                if isinstance(val, str) and val.strip():
                    keys.add(_norm_book_key(val))
        return [k for k in keys if k]

    # Maps a workspace's ``workspace_kind`` to the raw file
    # extensions that are legitimately pairable with it. EPUB
    # workspaces only ever want ``.epub`` sources; a ``.pdf``
    # candidate with the same filename stem must NOT win just
    # because the normalized title collides. Kinds that aren't
    # in the map (``""``, ``"other"``, ``"in_progress"``,
    # ``"image"``) fall back to "accept any" because we can't
    # predict the right extension without more signal.
    _KIND_ALLOWED_EXTS = {
        "epub": (".epub",),
        "txt":  (".txt",),
        "pdf":  (".pdf",),
        "html": (".html", ".htm"),
    }

    def _compute_matches(self,
                         candidates: list[tuple[str, str]]) -> dict:
        """Return ``{output_folder: {book, path, ratio, accepted}}``.

        All heavy lifting (including Fuzzy's ``SequenceMatcher``
        calls) runs here on the worker thread. Main thread only
        receives the final dict and paints the tree.
        """
        import difflib
        matches: dict[str, dict] = {}
        mode = self._mode
        threshold = self._threshold / 100.0
        # Pool the SequenceMatcher calls across workers too \u2014 for a
        # big candidate set Fuzzy matching is the real hotspot.
        kind_allowed = self._KIND_ALLOWED_EXTS

        def _best_for(book: dict) -> tuple[str, float]:
            book_keys = self._book_keys(book)
            if not book_keys or not candidates:
                return "", 0.0
            # Per-book extension gate: a workspace that advertises
            # its own kind (EPUB / TXT / PDF / HTML) must only be
            # paired with candidates whose extension matches. This
            # closes the hole where a ``.pdf`` raw whose filename
            # stem collides with an EPUB workspace's title would
            # be auto-accepted at ratio 1.0 just because the
            # normalized-key matched.
            book_kind = (book.get("workspace_kind") or "").lower()
            allowed_exts = kind_allowed.get(book_kind)
            if allowed_exts:
                usable = [
                    (ck, cp) for ck, cp in candidates
                    if cp.lower().endswith(allowed_exts)
                ]
            else:
                usable = candidates
            if not usable:
                return "", 0.0
            if mode == ScanForRawMixin.MATCH_EXACT:
                book_key_set = set(book_keys)
                for cand_key, cand_path in usable:
                    if self._cancelled:
                        break
                    if cand_key in book_key_set:
                        return cand_path, 1.0
                return "", 0.0
            best_path = ""
            best_ratio = 0.0
            sm = difflib.SequenceMatcher()
            for cand_key, cand_path in usable:
                if self._cancelled:
                    break
                sm.set_seq2(cand_key)
                for wk in book_keys:
                    sm.set_seq1(wk)
                    # Cheap length-ratio prefilter \u2014 skip candidates
                    # that can't possibly reach the threshold so we
                    # don't pay for a full ratio() call on obvious
                    # non-matches. real_quick_ratio is O(1).
                    if sm.real_quick_ratio() < threshold:
                        continue
                    ratio = sm.ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_path = cand_path
                        if best_ratio >= 0.999:
                            return best_path, best_ratio
            if best_ratio >= threshold:
                return best_path, best_ratio
            return "", best_ratio

        try:
            with ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="ScanForRawMatch",
            ) as executor:
                book_futures = []
                for book in self._books:
                    if self._cancelled:
                        break
                    ws_folder = (book.get("output_folder")
                                 or book.get("path") or "")
                    if not ws_folder:
                        continue
                    book_futures.append(
                        (ws_folder, book,
                         executor.submit(_best_for, book)))
                for ws_folder, book, fut in book_futures:
                    if self._cancelled:
                        break
                    try:
                        matched_path, ratio = fut.result()
                    except Exception:
                        matched_path, ratio = "", 0.0
                        logger.debug(
                            "ScanForRaw match task failed: %s",
                            traceback.format_exc())
                    matches[ws_folder] = {
                        "book": book,
                        "path": matched_path,
                        "ratio": ratio,
                        "accepted": bool(matched_path),
                    }
        except Exception:
            logger.debug(
                "ScanForRaw match pass crashed: %s",
                traceback.format_exc())
        return matches

    def run(self) -> None:
        candidates = (list(self._prewalked)
                      if self._prewalked is not None
                      else self._walk())
        if self._cancelled:
            return
        matches = self._compute_matches(candidates)
        if self._cancelled:
            return
        self.results.emit(self._folder, candidates, matches)


class ScanForRawMixin:
    """``_ScanForRawDialog`` state and decisions (settings, extensions, status line,
    writing the accepted pairings) without the dialog.
    """

    MATCH_EXACT = "exact"

    MATCH_FUZZY = "fuzzy"

    _SUPPORTED_EXTS = (".epub", ".txt", ".pdf", ".html", ".htm")

    def _init_scan_state(self, in_progress_books: list[dict],
                         config: dict | None = None) -> None:
        """Books to pair plus the persisted mode / threshold / folder / extensions."""
        self._config = config or {}
        # Copy so we don't hold live pointers into the parent dialog's
        # state (the scanner thread is free to replace the list).
        # We only care about workspace-backed cards — the pairing
        # writes ``source_epub.txt`` into the output folder, so
        # library-filed cards (which don't own a workspace) can't be
        # paired this way and are skipped.
        self._books: list[dict] = [
            dict(b) for b in (in_progress_books or [])
            if bool(b.get("output_folder"))
            and (b.get("missing_raw_file")
                 or not b.get("raw_source_path"))
        ]
        self._mode = self._config.get(
            "epub_library_scan_raw_mode", self.MATCH_EXACT)
        if self._mode not in (self.MATCH_EXACT, self.MATCH_FUZZY):
            self._mode = self.MATCH_EXACT
        try:
            self._threshold = int(
                self._config.get("epub_library_scan_raw_threshold", 70))
        except (TypeError, ValueError):
            self._threshold = 70
        self._threshold = max(40, min(95, self._threshold))
        self._scan_folder = self._config.get(
            "epub_library_scan_raw_folder", "") or ""
        # Extensions the user wants to scan for. Default is Auto:
        # derive the set from each in-progress workspace's known
        # ``workspace_kind`` (the scanner already classifies each
        # folder as epub / txt / pdf / image / other via
        # :func:`_detect_workspace_kind`, which is what drives the
        # per-card extension badge). Auto mode means the scan
        # defaults to exactly the extensions the visible missing-raw
        # cards need — so an EPUB-only library doesn't bother
        # hashing every TXT / PDF file in the chosen folder.
        #
        # The user can flip to manual mode by unchecking the Auto
        # toggle and then toggling individual extension checkboxes.
        # Persisted state:
        #   * ``epub_library_scan_raw_auto``   (bool, default True)
        #   * ``epub_library_scan_raw_exts``   (manual-mode selection)
        valid_exts = {"epub", "txt", "pdf", "html"}
        self._valid_exts = valid_exts
        auto_default = True
        try:
            self._auto_mode = bool(self._config.get(
                "epub_library_scan_raw_auto", auto_default))
        except Exception:
            self._auto_mode = auto_default
        stored_exts = self._config.get(
            "epub_library_scan_raw_exts", None)
        if isinstance(stored_exts, (list, tuple, set)) and stored_exts:
            self._manual_exts: set[str] = {
                str(e).strip().lower().lstrip(".")
                for e in stored_exts
                if str(e).strip().lower().lstrip(".") in valid_exts
            }
        else:
            self._manual_exts = set(valid_exts)
        if not self._manual_exts:
            self._manual_exts = set(valid_exts)
        # Live selection used by :meth:`_rescan_folder`. Recomputed
        # from ``_books`` when Auto is on; copied from
        # ``_manual_exts`` otherwise. Seeded now so the first scan
        # (triggered from ``__init__``) has a value to read.
        self._selected_exts: set[str] = (
            self._derive_auto_exts() if self._auto_mode
            else set(self._manual_exts))
        if not self._selected_exts:
            # Defensive: never leave the set empty — an empty set
            # would walk the folder but match zero files, reading
            # to the user as a silent "no matches" even though the
            # folder is full of candidates.
            self._selected_exts = set(valid_exts)
        # Candidate file index — populated by :meth:`_rescan_folder`,
        # a list of ``(normalized_stem, absolute_path)`` tuples so the
        # fuzzy matcher can iterate without re-scanning the folder on
        # every slider tick.
        self._candidates: list[tuple[str, str]] = []
        # Current matches keyed by workspace output_folder. Each value
        # is ``(matched_path, ratio, accepted)``. ``accepted=False``
        # means the user un-ticked the checkbox in the preview.
        self._matches: dict[str, dict] = {}

    def _derive_auto_exts(self) -> set[str]:
        """Infer the extension set from the missing-raw workspaces.

        Each in-progress workspace already carries a
        ``workspace_kind`` (``epub`` / ``txt`` / ``pdf`` / ``image``
        / ``other``) that drives the per-card badge. "Auto" mode
        reuses that classification so the scan only walks the
        extensions the visible cards actually need. An ``image``
        kind is tolerated but doesn't map to a searchable text
        extension, so it's ignored; ``other`` (unknown) expands to
        every extension since we can't predict what the user will
        bring. An empty selection falls back to the full set so the
        scan doesn't no-op.
        """
        valid = self._valid_exts
        out: set[str] = set()
        for b in self._books:
            kind = (b.get("workspace_kind")
                    or b.get("type") or "").lower()
            if kind in valid:
                out.add(kind)
            elif kind in ("", "other", "in_progress"):
                # Unknown / folder-only — fall back to the full set
                # so the scan still has a chance to pair the card.
                out.update(valid)
        if not out:
            out = set(valid)
        return out

    def _ext_suffixes(self) -> tuple[str, ...]:
        """Build the suffix tuple from the current extension selection.

        The HTML checkbox covers both ``.html`` and ``.htm`` because
        they're interchangeable on disk.
        """
        ext_suffixes: tuple[str, ...] = tuple(
            f".{e}" if e != "html" else ".html"
            for e in self._selected_exts
        )
        if "html" in self._selected_exts:
            ext_suffixes = ext_suffixes + (".htm",)
        return ext_suffixes

    def _scan_status_text(self, hits: int) -> str:
        """Scan for Raw status line for the current matches (*hits* = rows with a match)."""
        mode_label = (
            "exact" if self._mode == self.MATCH_EXACT
            else f"fuzzy \u2265 {self._threshold}%")
        workspace_count = len(self._matches)
        candidate_count = len(self._candidates)
        if workspace_count == 0:
            # No missing-raw cards fed into the dialog in the first
            # place — usually means the button was opened before
            # the scanner populated any workspaces.
            return (
                "\u24d8 No missing-raw workspaces to pair.")
        elif candidate_count == 0:
            return (
                "\u26a0 No candidate files found in this folder "
                f"({mode_label}). Pick a different folder or "
                "enable more extensions."
            )
        elif hits == 0:
            hint = (
                "try lowering the Similarity slider"
                if self._mode == self.MATCH_FUZZY
                else "switch to Fuzzy match or rename the raw "
                     "files to match the workspace folder names")
            return (
                f"\u26a0 0 of {workspace_count} workspace"
                f"{'s' if workspace_count != 1 else ''} matched "
                f"({candidate_count} candidate file"
                f"{'s' if candidate_count != 1 else ''} scanned, "
                f"{mode_label}). Try {hint}."
            )
        else:
            return (
                f"\u2714 {hits} of {workspace_count} workspace"
                f"{'s' if workspace_count != 1 else ''} matched "
                f"({candidate_count} candidate file"
                f"{'s' if candidate_count != 1 else ''} scanned, "
                f"{mode_label})."
            )

    def _write_raw_pairings(self) -> int:
        """Write ``source_epub.txt`` + register the raw for every accepted match."""
        written = 0
        for ws_folder, info in self._matches.items():
            if not info.get("accepted"):
                continue
            raw_path = info.get("path") or ""
            if not raw_path or not os.path.isfile(raw_path):
                continue
            if not ws_folder or not os.path.isdir(ws_folder):
                continue
            try:
                sidecar = os.path.join(ws_folder, "source_epub.txt")
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(os.path.abspath(raw_path))
                try:
                    record_library_raw_input(raw_path)
                except Exception:
                    logger.debug(
                        "record_library_raw_input failed for %s: %s",
                        raw_path, traceback.format_exc())
                written += 1
            except OSError as exc:
                logger.debug(
                    "Scan-for-raw sidecar write failed for %s: %s",
                    ws_folder, exc)
        return written


class LibraryShelfMixin:
    """``EpubLibraryDialog`` decisions without widgets: filters / sort / card signatures,
    the auto-refresh diff, Organize / Undo / Delete / Clear-raw-link plans and
    executions, imports and the output-root check. State: ``_in_progress_books``,
    ``_completed_books``, ``_config``, ``_sort_mode``, ``_format_filter`` and a
    ``_search`` object with ``.text()`` (desktop: the toolbar search box).
    """

    @staticmethod
    def _format_of_book(book: dict) -> str:
        """Return the FORMAT_* key that describes *book* for filtering.

        Rules:
          * In-progress folder cards are classified by their
            ``workspace_kind`` (the scanner already resolves this from
            the raw source extension or the folder's on-disk contents).
          * Other cards (library entries, promoted compiled workspaces,
            registered-in-place translated imports) are classified by
            their ``type`` field.
          * Unknown / other values collapse to ``FORMAT_ALL`` so they
            never accidentally match a specific chip — the All chip
            always shows them.
        """
        file_type = (book.get("type") or "").lower()
        kind: str
        if file_type == "in_progress":
            kind = (book.get("workspace_kind") or "").lower()
        else:
            kind = file_type
        mapping = {
            "epub":  FORMAT_EPUB,
            "txt":   FORMAT_TXT,
            "pdf":   FORMAT_PDF,
            "html":  FORMAT_HTML,
            "image": FORMAT_IMAGE,
        }
        return mapping.get(kind, FORMAT_ALL)

    def _sorted_books(self, books):
        if self._sort_mode == SORT_NAME:
            return sorted(books, key=lambda b: b["name"].lower())
        elif self._sort_mode == SORT_SIZE:
            return sorted(books, key=lambda b: b["size"], reverse=True)
        return sorted(books, key=lambda b: b["mtime"], reverse=True)

    def _count_raw_movable(self) -> int:
        """Return how many raw sources aren't already in Library/Raw.

        Walks BOTH the In Progress and Completed book lists — a book at
        100 %% progress lives in ``_completed_books`` yet its raw source
        may still be sitting outside Library/Raw (e.g. the original EPUB
        that fed the translation). Library-tagged entries are skipped
        because their raw, if any, is already filed. Raw paths are
        deduplicated so one file counted against two scan rows doesn't
        double the counter.
        """
        raw_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_raw_dir())))
        count = 0
        seen: set[str] = set()

        def _bump(book: dict) -> int:
            p = book.get("raw_source_path") or ""
            if not p or not os.path.isfile(p):
                return 0
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == raw_abs:
                return 0
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen:
                return 0
            seen.add(key)
            return 1

        for book in self._in_progress_books:
            count += _bump(book)
        for book in self._completed_books:
            if book.get("in_library"):
                continue
            count += _bump(book)
        return count

    def _count_trans_movable(self) -> int:
        """Return how many Completed compiled EPUBs aren't already in Library/Translated."""
        trans_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_translated_dir())))
        count = 0
        for book in self._completed_books:
            if book.get("in_library"):
                continue
            p = book.get("path") or ""
            if not p or not os.path.isfile(p):
                continue
            if not p.lower().endswith(".epub"):
                continue
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent != trans_abs:
                count += 1
        return count

    def _build_workspace_title_index(self) -> dict:
        """Return a ``{normalized_title_key: workspace_book_dict}`` map.

        Walks both the current in-progress and completed book lists
        (as seen by the dialog) and indexes each workspace-backed
        entry by every candidate title we can derive — folder name,
        raw source stem, and every ``metadata.json`` title field.

        Used by :meth:`_undo_organize_prompt` to pick a restore target
        for Library/Translated files that have no ``origins['translated']``
        entry. Keys run through :func:`_norm_book_key` so comparisons
        survive the NTFS filename mangling that desynchronizes the
        on-disk stem from the metadata title (``"… Fantasy."`` vs
        ``"… Fantasy"``).
        """
        index: dict[str, dict] = {}

        def _ingest(book: dict) -> None:
            ws_folder = book.get("output_folder") or ""
            if not ws_folder:
                return
            keys: set[str] = set()
            fn = book.get("folder_name") or os.path.basename(ws_folder)
            if fn:
                keys.add(_norm_book_key(os.path.splitext(fn)[0]))
            raw = book.get("raw_source_path") or ""
            if raw:
                keys.add(_norm_book_key(
                    os.path.splitext(os.path.basename(raw))[0]))
            md = book.get("metadata_json") or {}
            if isinstance(md, dict):
                for md_key in ("title", "original_title",
                               "translated_title", "raw_title",
                               "source_title", "english_title"):
                    val = md.get(md_key)
                    if isinstance(val, str) and val.strip():
                        keys.add(_norm_book_key(val))
            for key in keys:
                if key:
                    index.setdefault(key, book)

        for book in self._in_progress_books:
            _ingest(book)
        for book in self._completed_books:
            # Library entries don't own a workspace — skip so a library
            # row matching itself doesn't produce a nonsense restore
            # target pointing back at Library/Translated.
            if book.get("in_library"):
                continue
            _ingest(book)
        return index

    @staticmethod
    def _count_library_files(folder: str,
                             epub_only: bool = False) -> int:
        """Count candidate restorable files sitting directly in *folder*.

        ``epub_only`` restricts the count to ``.epub`` files (used for
        Library/Translated where only compiled EPUBs are tracked);
        other shelves include ``.txt``, ``.pdf``, and ``.html`` too.
        Registry / tracking files (``library_*_inputs.txt``,
        ``library_origins.txt``) are always excluded so a legacy
        copy sitting inside ``Library/Raw`` doesn't inflate the
        Undo counter / enable state.
        """
        if not folder or not os.path.isdir(folder):
            return 0
        exts = (".epub",) if epub_only else (
            ".epub", ".txt", ".pdf", ".html")
        count = 0
        try:
            for entry in os.scandir(folder):
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                    continue
                if entry.name.lower().endswith(exts):
                    count += 1
        except (PermissionError, OSError):
            return 0
        return count

    def _organize_counts(self) -> dict:
        """Counters behind the Organize / Undo buttons (``_update_organize_counts``).

        ``raw_count`` / ``trans_count``: files Organize would move into
        ``Library/Raw`` / ``Library/Translated``. ``raw_undo`` / ``trans_undo``:
        the larger of the origins entries and the files physically in each shelf.
        """
        raw_count = self._count_raw_movable()
        trans_count = self._count_trans_movable()
        try:
            origins = _load_origins()
            raw_orig = len(origins.get("raw", {}) or {})
            trans_orig = len(origins.get("translated", {}) or {})
        except Exception:
            raw_orig = 0
            trans_orig = 0
        # Also count files physically sitting in the library shelves.
        # Undo now covers orphan files (no origins entry) too, so the
        # button must stay enabled / labelled as long as SOMETHING is
        # restorable — not only when the origins registry has rows.
        raw_disk = self._count_library_files(get_library_raw_dir())
        trans_disk = self._count_library_files(
            get_library_translated_dir(), epub_only=True)
        raw_undo = max(raw_orig, raw_disk)
        trans_undo = max(trans_orig, trans_disk)
        return {
            "raw_count": raw_count,
            "trans_count": trans_count,
            "raw_undo": raw_undo,
            "trans_undo": trans_undo,
        }

    def _missing_raw_count(self) -> int:
        """Cards (both tabs) carrying the ``missing_raw_file`` warning."""
        return sum(
            1 for b in (
                list(self._in_progress_books)
                + list(self._completed_books))
            if b.get("missing_raw_file"))

    def _run_import(self, paths, target: str = "raw"):
        """Validate + register *paths* (``_import_paths_into_library``'s core).

        Returns ``(imported, skipped, errors)``: ``imported`` holds the absolute
        paths registered through :meth:`_import_single_file`, ``skipped`` and
        ``errors`` the per-file diagnostics shown in the summary.
        """
        if target == "translated":
            supported_exts = (".epub",)
            dest_label = "Library (registered in place)"
            dest_dir = get_library_translated_dir()
        else:
            supported_exts = (".epub", ".txt", ".pdf", ".html", ".htm")
            dest_label = "Library (registered in place)"
            dest_dir = get_library_raw_dir()

        # First pass: validate each path. Neither target relocates
        # files anymore, so there's no collision bucket — everything
        # accepted lands in ``fresh`` and runs through
        # :meth:`_import_single_file`, which just appends to the
        # appropriate input registry (+ scaffolds an output folder
        # for the raw side).
        imported: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        fresh: list[str] = []
        for raw_path in paths:
            if not raw_path:
                continue
            try:
                path = os.path.abspath(raw_path)
            except (TypeError, ValueError):
                continue
            if not os.path.isfile(path):
                skipped.append(f"{os.path.basename(raw_path)} (not a file)")
                continue
            if not path.lower().endswith(supported_exts):
                # On the Completed tab only .epub makes sense; non-EPUBs
                # are called out explicitly so the user understands why
                # a mixed drop didn't land.
                if target == "translated":
                    skipped.append(
                        f"{os.path.basename(path)} "
                        f"(only EPUBs go to Library/Translated)"
                    )
                else:
                    skipped.append(
                        f"{os.path.basename(path)} (unsupported type)"
                    )
                continue
            fresh.append(path)

        collisions: list[tuple[str, str]] = []  # never populated now
        collision_policy = "keep_both"

        def _process(path: str, policy: str) -> None:
            try:
                dest = self._import_single_file(
                    path, target=target, collision_policy=policy)
                if dest:
                    imported.append(dest)
                elif policy == "skip":
                    skipped.append(
                        f"{os.path.basename(path)} (duplicate \u2014 skipped)")
            except Exception as exc:
                logger.error("Import failed for %s: %s\n%s",
                             path, exc, traceback.format_exc())
                errors.append(f"{os.path.basename(path)}: {exc}")

        for p in fresh:
            _process(p, "keep_both")  # policy doesn't matter, no collision
        for src, _existing in collisions:
            _process(src, collision_policy)
        return imported, skipped, errors

    @staticmethod
    def _import_toast_text(imported, skipped, errors, target: str = "raw") -> str:
        """Drag-drop import feedback: one short, non-modal status line."""
        if imported and not errors:
            return (
                f"\u2705  Registered {len(imported)} file"
                f"{'s' if len(imported) != 1 else ''} "
                f"with the Library"
            )
        elif imported and errors:
            return (
                f"\u26a0\ufe0f  Registered {len(imported)} with the "
                f"Library, failed {len(errors)}"
            )
        elif errors:
            return (
                f"\u26a0\ufe0f  Import failed ({len(errors)} error"
                f"{'s' if len(errors) != 1 else ''})"
            )
        elif skipped:
            # Translated-target skips are already self-describing
            # ("only EPUBs go to Library/Translated"); use a matching
            # short toast so users understand why nothing landed.
            if target == "translated":
                return (
                    f"\u2139\ufe0f  Only EPUBs can be dropped onto the "
                    f"Completed tab ({len(skipped)} skipped)"
                )
            else:
                return (
                    f"\u2139\ufe0f  Skipped {len(skipped)} unsupported "
                    f"file{'s' if len(skipped) != 1 else ''}"
                )
        return ""

    @staticmethod
    def _import_summary(imported, skipped, errors, target: str = "raw"):
        """File-picker import summary as ``(title, body)``."""
        # Summary dialog — one message per batch, not per file.
        title = "Import"
        parts: list[str] = []
        if imported:
            target_tab = (
                "Library/Translated" if target == "translated"
                else "Library/Raw"
            )
            parts.append(
                f"Registered {len(imported)} file"
                f"{'s' if len(imported) != 1 else ''} with the "
                f"Library \u2014 no files were moved. Click "
                f"\u201cOrganize\u201d when you're ready to move "
                f"them into {target_tab}."
            )
            if len(imported) <= 10:
                parts.append("\n".join(
                    "  \u2022 " + os.path.basename(p) for p in imported))
            else:
                parts.append("\n".join(
                    "  \u2022 " + os.path.basename(p) for p in imported[:10]))
                parts.append(f"  \u2026 and {len(imported) - 10} more.")
        if skipped:
            parts.append(
                f"\nSkipped {len(skipped)} file{'s' if len(skipped) != 1 else ''}:"
            )
            parts.append("\n".join("  \u2022 " + s for s in skipped[:8]))
        if errors:
            parts.append(
                f"\n{len(errors)} error{'s' if len(errors) != 1 else ''}:"
            )
            parts.append("\n".join("  \u2022 " + e for e in errors[:5]))
        if imported and target != "translated":
            parts.append(
                "\nRight-click any card and choose \u201cLoad for translation\u201d "
                "when you're ready to translate."
            )
        body = "\n".join(parts)
        return title, body

    def _import_single_file(self, path: str, target: str = "raw",
                             collision_policy: str = "keep_both"
                             ) -> str | None:
        """Register *path* with the library (raw or translated side).

        Both branches now leave the source file **exactly where it is**
        on disk. Nothing is copied or moved — the import just wires
        the path into the library's tracking files so a card can
        surface on the appropriate tab. Moving into ``Library/Raw`` /
        ``Library/Translated`` is a separate, deliberate step
        triggered by the Organize button. Both the raw and translated
        registrations are fully reversible: Organize writes an entry
        into ``library_origins.txt`` before relocating, and Undo Move
        restores the file and re-adds it to the appropriate input
        registry so the card reappears on its tab.

        Per-target wiring:

          * ``"raw"`` — appends the absolute path to
            ``Library/Raw/library_raw_inputs.txt`` and scaffolds an
            output folder under the configured output root with a
            ``source_epub.txt`` sidecar pointing at the *original*
            location. The In Progress tab's scanner uses these to
            surface a Not Started card for the file.
          * ``"translated"`` — appends the absolute path to
            ``Library/library_translated_inputs.txt``. The Completed
            tab's scanner includes these as ``in_library=False`` +
            ``registered_translated=True`` cards so the user can
            read them and the Organize button can promote them into
            ``Library/Translated``.

        *collision_policy* is accepted for backwards compatibility
        but no longer matters — nothing is being relocated here, so
        there's nothing to collide with. It remains in the signature
        so future callers can still pass it without a TypeError.

        Returns the *original* absolute path on success, or ``None``
        on failure. Raises no exceptions — failures are logged and
        surfaced via the caller's aggregated error list.
        """
        path_abs = os.path.abspath(path)
        if target == "translated":
            # ---- Translated branch: register in place.
            try:
                record_library_translated_input(path_abs)
            except Exception as exc:
                logger.error(
                    "Translated registration failed for %s: %s\n%s",
                    path_abs, exc, traceback.format_exc())
                raise
            return path_abs

        # ---- Raw branch: register in place + scaffold output folder.
        try:
            record_library_raw_input(path_abs)
            roots = _resolve_output_roots(self._config)
            if roots:
                output_root = roots[0]
                base = os.path.splitext(os.path.basename(path_abs))[0]
                output_folder = os.path.join(output_root, base)
                os.makedirs(output_folder, exist_ok=True)
                # ``source_epub.txt`` points at the *real* location of
                # the raw source so the translator (and the In Progress
                # scanner) can find it without going through
                # Library/Raw. When the user later runs Organize, it
                # moves the file and rewrites this sidecar to the new
                # ``Library/Raw\…`` path.
                sidecar = os.path.join(output_folder, "source_epub.txt")
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(path_abs)
                progress_file_path = os.path.join(
                    output_folder, "translation_progress.json")
                if not os.path.isfile(progress_file_path):
                    import json as _json
                    with open(progress_file_path, "w", encoding="utf-8") as pf:
                        _json.dump(
                            {"chapters": {}, "chapter_chunks": {}, "version": "2.1"},
                            pf, ensure_ascii=False, indent=2,
                        )
        except Exception as exc:
            logger.error("Raw registration failed for %s: %s\n%s",
                         path_abs, exc, traceback.format_exc())
            raise
        return path_abs

    def _output_override_mismatch(self, books: list):
        """Decide whether loading *books* needs an output-root switch (``None`` = no).

        A card's workspace lives under one output root; when that root differs
        from the current override (or the default root when none is set), returns
        ``{expected_roots, mismatched, current_override, default_root,
        new_override, current_label, new_label, preview, multi_root_warning}``.
        """
        # Collect the distinct expected roots across every book. Each
        # root corresponds to a different card-producing output
        # location — normally one, but mixed selections can span more.
        expected_roots: list[str] = []
        seen_keys: set[str] = set()
        for b in books or []:
            root = _expected_output_root_for_book(b)
            if not root:
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(root)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expected_roots.append(root)
        if not expected_roots:
            return None

        current_override = (
            os.environ.get("OUTPUT_DIRECTORY")
            or (self._config.get("output_directory") if self._config else "")
            or ""
        ).strip()
        default_root = _default_output_root()
        current_effective = current_override or default_root

        mismatched: list[tuple[str, str]] = []
        for b in books or []:
            root = _expected_output_root_for_book(b)
            if not root:
                continue
            if not _output_paths_equal(root, current_effective):
                mismatched.append((b.get("name") or "", root))
        if not mismatched:
            return None

        # Pick the target root. If the first mismatched root IS the
        # implicit default, clear the override (empty string) so the
        # translator falls back to the default root the same way an
        # unset field would. Otherwise set the override to that root.
        target_root = mismatched[0][1]
        new_override = ("" if _output_paths_equal(target_root, default_root)
                        else target_root)

        current_label = current_override or f"{default_root}  (default)"
        new_label = new_override or f"{default_root}  (default)"

        preview: list[str] = []
        for name, root in mismatched[:5]:
            label = name or os.path.basename(os.path.normpath(root))
            preview.append(f"  \u2022 {label}\n      {root}")
        if len(mismatched) > 5:
            preview.append(f"  \u2026 and {len(mismatched) - 5} more")

        multi_root_warning = ""
        if len(expected_roots) > 1:
            multi_root_warning = (
                "\n\n\u26a0 The selection spans multiple output folders; "
                "only the first mismatched root will be applied. Load "
                "cards from one folder at a time to avoid this."
            )

        return {
            "expected_roots": expected_roots,
            "mismatched": mismatched,
            "current_override": current_override,
            "default_root": default_root,
            "new_override": new_override,
            "current_label": current_label,
            "new_label": new_label,
            "preview": preview,
            "multi_root_warning": multi_root_warning,
        }

    @staticmethod
    def _output_override_prompt_text(info: dict) -> str:
        """The "Output Folder Mismatch" question."""
        current_label = info["current_label"]
        new_label = info["new_label"]
        preview = info["preview"]
        multi_root_warning = info["multi_root_warning"]
        return (
            "The selected translation is saved under a different folder "
            "than the current output-folder override.\n\n"
            f"Current override:\n  {current_label}\n\n"
            f"Will switch to:\n  {new_label}\n\n"
            "Mismatched books:\n"
            + "\n".join(preview)
            + multi_root_warning
            + "\n\nUpdate the override so a new translation run writes "
            "into the same folder as the existing progress?"
        )

    def _apply_output_override_config(self, new_override: str) -> None:
        """Switch the output-directory override in the config and the process env."""
        try:
            if self._config is not None:
                self._config["output_directory"] = new_override
            if new_override:
                os.environ["OUTPUT_DIRECTORY"] = new_override
            else:
                os.environ.pop("OUTPUT_DIRECTORY", None)
        except Exception:
            logger.debug(
                "Failed to apply output_directory override: %s",
                traceback.format_exc(),
            )

    def _plan_organize(self) -> dict:
        """Collect what Organize would move (``_organize_into_library``'s first pass).

        Raw sources outside ``Library/Raw`` (both tabs, de-duplicated) and compiled
        EPUBs outside ``Library/Translated`` (Completed cards plus the translated-
        inputs registry). Returns the shelf paths and ``raw_moves`` /
        ``translated_moves`` as ``[(book, path), ...]``.
        """
        raw_dir = get_library_raw_dir()
        trans_dir = get_library_translated_dir()
        raw_abs = os.path.normcase(os.path.normpath(os.path.abspath(raw_dir)))
        trans_abs = os.path.normcase(os.path.normpath(os.path.abspath(trans_dir)))

        # Collect raw sources that aren't already in Library/Raw. Walk
        # BOTH tabs — a book at 100 %% progress lives in
        # ``_completed_books`` but its raw source may still be sitting
        # outside Library/Raw (the common case for a freshly-compiled
        # translation). Previously only ``_in_progress_books`` was
        # inspected, so finishing a translation effectively hid the raw
        # from "Organize" forever. Raw paths are deduped so a book that
        # appears in both scans doesn't get scheduled twice.
        raw_moves: list[tuple[dict, str]] = []
        seen_raw_keys: set[str] = set()

        def _queue_raw(book: dict) -> None:
            p = book.get("raw_source_path") or ""
            if not p or not os.path.isfile(p):
                return
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == raw_abs:
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen_raw_keys:
                return
            seen_raw_keys.add(key)
            raw_moves.append((book, p))

        for book in self._in_progress_books:
            _queue_raw(book)
        for book in self._completed_books:
            # Library entries are already filed — nothing to organize.
            if book.get("in_library"):
                continue
            _queue_raw(book)

        # Collect compiled EPUBs that aren't already in Library/Translated.
        # Two sources are pulled together:
        #   * Completed-tab cards whose ``path`` points at a compiled
        #     EPUB sitting somewhere other than ``Library/Translated``
        #     — includes both promoted-output-folder cards and
        #     registered-in-place translated imports.
        #   * ``library_translated_inputs.txt`` directly, as a safety
        #     net in case a registered entry hasn't made it into the
        #     scan result yet (first auto-refresh still pending, etc.).
        translated_moves: list[tuple[dict, str]] = []
        seen_trans_keys: set[str] = set()

        def _queue_trans(book: dict, p: str) -> None:
            if not p or not os.path.isfile(p):
                return
            if not p.lower().endswith(".epub"):
                return  # only compiled .epub files get organized for now
            parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(p))))
            if parent == trans_abs:
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen_trans_keys:
                return
            seen_trans_keys.add(key)
            translated_moves.append((book, p))

        for book in self._completed_books:
            # Skip Library entries (they already live in Library/Translated).
            if book.get("in_library"):
                continue
            _queue_trans(book, book.get("path") or "")
        # Belt-and-suspenders: also walk the registry directly so a
        # just-dropped file still gets organized even if the card
        # list hasn't refreshed yet.
        for p in load_library_translated_inputs():
            _queue_trans({"registered_translated": True}, p)
        return {
            "raw_dir": raw_dir,
            "trans_dir": trans_dir,
            "raw_abs": raw_abs,
            "trans_abs": trans_abs,
            "raw_moves": raw_moves,
            "translated_moves": translated_moves,
        }

    @staticmethod
    def _organize_preview_lines(plan: dict) -> list:
        """Organize confirmation lines, one per shelf with something to move."""
        raw_moves = plan["raw_moves"]
        translated_moves = plan["translated_moves"]
        preview = []
        if raw_moves:
            preview.append(
                f"Raw \u2192 Library/Raw: {len(raw_moves)} file"
                f"{'s' if len(raw_moves) != 1 else ''}")
        if translated_moves:
            preview.append(
                f"Translated \u2192 Library/Translated: {len(translated_moves)} file"
                f"{'s' if len(translated_moves) != 1 else ''}")
        return preview

    @staticmethod
    def _organize_collisions(plan: dict):
        """Planned moves whose target name exists: ``(raw_collisions, trans_collisions)``."""
        raw_dir = plan["raw_dir"]
        trans_dir = plan["trans_dir"]
        raw_moves = plan["raw_moves"]
        translated_moves = plan["translated_moves"]
        # Pre-scan for name collisions in Library/Raw and
        # Library/Translated. If any exist, ask the user once for a
        # single policy that applies to every duplicate in this run —
        # mirrors the drag-drop import prompt so a name collision can
        # no longer silently auto-rename with ``(2)`` / ``(3)``.
        raw_collisions: list[tuple[str, str]] = []
        for _b, src in raw_moves:
            candidate = os.path.join(raw_dir, os.path.basename(src))
            if (os.path.isfile(candidate)
                    and os.path.abspath(src) != os.path.abspath(candidate)):
                raw_collisions.append((src, candidate))
        trans_collisions: list[tuple[str, str]] = []
        for _b, src in translated_moves:
            candidate = os.path.join(trans_dir, os.path.basename(src))
            if (os.path.isfile(candidate)
                    and os.path.abspath(src) != os.path.abspath(candidate)):
                trans_collisions.append((src, candidate))
        return raw_collisions, trans_collisions

    def _execute_organize(self, plan: dict, collision_policy: str = "keep_both",
                          all_collisions=()) -> dict:
        """Perform Organize's moves (``_organize_into_library``'s second pass).

        *collision_policy* (``replace`` / ``keep_both`` / ``skip``) applies to the
        sources listed in *all_collisions* (``_organize_collisions``). Moves raw
        sources into ``Library/Raw`` (rewriting ``source_epub.txt``) and compiled
        EPUBs into ``Library/Translated``, records origins + pairs and returns the
        counters, the errors and the ``(old, new)`` path moves.
        """
        raw_dir = plan["raw_dir"]
        trans_dir = plan["trans_dir"]
        raw_abs = plan["raw_abs"]
        raw_moves = plan["raw_moves"]
        translated_moves = plan["translated_moves"]
        # Fast-lookup set of source paths whose dest already exists,
        # so each move loop can apply *collision_policy* while still
        # letting non-colliding files fall through unchanged.
        colliding_srcs = {
            os.path.normcase(os.path.normpath(os.path.abspath(s)))
            for s, _d in all_collisions
        }

        def _src_has_collision(src: str) -> bool:
            try:
                key = os.path.normcase(os.path.normpath(
                    os.path.abspath(src)))
            except Exception:
                return False
            return key in colliding_srcs

        origins = _load_origins()
        raw_origins = dict(origins.get("raw", {}) or {})
        trans_origins = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})

        moved_raw = 0
        moved_trans = 0
        skipped_raw = 0
        skipped_trans = 0
        errors: list[str] = []
        # Per-book dest-basename trackers (keyed by ``id(book)``) so we can
        # pair translated↔raw after both moves finish. Books that only
        # have one side moved in this run contribute a partial entry and
        # we fall back to ``raw_source_path`` on the book dict below.
        raw_dest_by_book: dict[int, str] = {}
        trans_dest_by_book: dict[int, str] = {}
        # Collected (old_abs_path, new_abs_path) pairs for every move in
        # this run — emitted via :attr:`files_reorganized` at the end so
        # the translator GUI can update any stale paths it's holding
        # onto (e.g. the "Input file" line edit still pointing at
        # ``Downloads/novel.epub`` after the raw moved into
        # ``Library/Raw/novel.epub``).
        path_moves: list[tuple[str, str]] = []

        def _resolve_dest(directory: str, base_name: str, src: str
                          ) -> str | None:
            """Pick the destination path for *src* under *collision_policy*.

            Returns ``None`` when the file should be skipped (policy =
            "skip" on a collision). For ``"replace"`` the pre-existing
            file is removed so :func:`shutil.move` lands atomically; for
            ``"keep_both"`` we fall through to the counter-suffix path.
            """
            dest = os.path.join(directory, base_name)
            if not _src_has_collision(src):
                return dest
            if collision_policy == "skip":
                return None
            if collision_policy == "replace":
                try:
                    if os.path.isfile(dest):
                        os.remove(dest)
                except OSError as rm_exc:
                    # Log and fall back to keep_both so the move doesn't
                    # hard-fail just because the replace couldn't happen.
                    logger.debug("Replace-on-organize remove failed: %s",
                                 rm_exc)
                    return _unique_dest(directory, base_name)
                return dest
            return _unique_dest(directory, base_name)

        # Raw sources: MOVE + update source_epub.txt pointer.
        for book, src in raw_moves:
            try:
                dest = _resolve_dest(raw_dir, os.path.basename(src), src)
                if dest is None:  # user chose Skip All
                    skipped_raw += 1
                    continue
                shutil.move(src, dest)
                raw_origins[os.path.basename(dest)] = os.path.abspath(src)
                record_library_raw_input(dest)
                out_folder = book.get("output_folder") or ""
                if out_folder and os.path.isdir(out_folder):
                    try:
                        with open(os.path.join(out_folder, "source_epub.txt"),
                                  "w", encoding="utf-8") as f:
                            f.write(dest)
                    except OSError as pe:
                        logger.debug("Update source_epub.txt failed: %s", pe)
                raw_dest_by_book[id(book)] = os.path.basename(dest)
                path_moves.append(
                    (os.path.abspath(src), os.path.abspath(dest)))
                moved_raw += 1
            except Exception as exc:
                errors.append(f"raw:{os.path.basename(src)}: {exc}")

        # Translated compiled EPUBs: MOVE into Library/Translated.
        for book, src in translated_moves:
            try:
                dest = _resolve_dest(trans_dir, os.path.basename(src), src)
                if dest is None:
                    skipped_trans += 1
                    continue
                shutil.move(src, dest)
                trans_origins[os.path.basename(dest)] = os.path.abspath(src)
                trans_dest_by_book[id(book)] = os.path.basename(dest)
                path_moves.append(
                    (os.path.abspath(src), os.path.abspath(dest)))
                # If this file was in the registered-in-place
                # translated registry, drop it now — the in-place
                # registration is superseded by the origins entry
                # above, which is what Undo keys on.
                try:
                    remove_library_translated_input(src)
                except Exception:
                    logger.debug(
                        "Failed to prune translated-inputs entry for %s",
                        src,
                    )
                moved_trans += 1
            except Exception as exc:
                errors.append(f"translated:{os.path.basename(src)}: {exc}")

        # Pair up translated↔raw so later lookups don't have to rely on
        # filename-stem matching (which fails when raw and translated are
        # in different languages) or the output-folder sidecar (which
        # fails if that folder is later deleted). For each book whose
        # translated was moved, its raw is either:
        #   * in ``raw_dest_by_book`` (we just organized it), or
        #   * already filed under ``Library/Raw`` from a previous import
        #     (pick it up via the book's ``raw_source_path``).
        for book_id, trans_basename in trans_dest_by_book.items():
            raw_basename = raw_dest_by_book.get(book_id)
            if not raw_basename:
                # Fall back to the book's pre-existing raw if it already
                # lives in Library/Raw.
                paired_book = None
                for source_list in (self._completed_books,
                                    self._in_progress_books):
                    for b in source_list:
                        if id(b) == book_id:
                            paired_book = b
                            break
                    if paired_book is not None:
                        break
                if paired_book is not None:
                    rp = paired_book.get("raw_source_path") or ""
                    if rp and os.path.isfile(rp):
                        rp_parent = os.path.normcase(os.path.normpath(
                            os.path.abspath(os.path.dirname(rp))))
                        if rp_parent == raw_abs:
                            raw_basename = os.path.basename(rp)
            if raw_basename:
                pair_map[trans_basename] = raw_basename

        origins["raw"] = raw_origins
        origins["translated"] = trans_origins
        origins["pairs"] = pair_map
        _save_origins(origins)
        return {
            "moved_raw": moved_raw,
            "moved_trans": moved_trans,
            "skipped_raw": skipped_raw,
            "skipped_trans": skipped_trans,
            "errors": errors,
            "path_moves": path_moves,
        }

    @staticmethod
    def _organize_summary(result: dict) -> str:
        """The Organize summary message."""
        moved_raw = result["moved_raw"]
        moved_trans = result["moved_trans"]
        skipped_raw = result["skipped_raw"]
        skipped_trans = result["skipped_trans"]
        errors = result["errors"]
        summary_parts = []
        if moved_raw:
            summary_parts.append(
                f"Moved {moved_raw} raw source"
                f"{'s' if moved_raw != 1 else ''} into Library/Raw.")
        if moved_trans:
            summary_parts.append(
                f"Moved {moved_trans} compiled EPUB"
                f"{'s' if moved_trans != 1 else ''} into Library/Translated.")
        if skipped_raw or skipped_trans:
            total_skipped = skipped_raw + skipped_trans
            summary_parts.append(
                f"Skipped {total_skipped} duplicate"
                f"{'s' if total_skipped != 1 else ''}."
            )
        summary = "\n".join(summary_parts) or "Nothing was moved."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        return summary

    def _plan_undo(self) -> dict:
        """Collect what Undo Move can restore (``_undo_organize_prompt``'s first pass).

        ``raw_map`` / ``trans_map`` map shelf basenames to restore targets: the
        origins registry plus orphan files found on disk (best-guess workspace by
        title key, else the Library's parent folder; listed again in
        ``raw_orphans`` / ``trans_orphans``).
        """
        origins = _load_origins()
        raw_map = dict(origins.get("raw", {}) or {})
        trans_map = dict(origins.get("translated", {}) or {})
        pair_map = dict(origins.get("pairs", {}) or {})

        raw_dir = get_library_raw_dir()
        trans_dir = get_library_translated_dir()

        # Extend raw_map / trans_map with every EPUB actually on disk
        # in the library shelves so "Undo Move" covers orphan files
        # too. For orphans we compute a best-guess restore target via
        # title matching against the scanned in-progress / completed
        # workspaces; if no match lands we fall back to the library
        # parent dir so the file lands somewhere reachable rather
        # than just vanishing. User feedback drove this: clicking
        # Undo on a file with no origins record used to silently do
        # nothing, leaving the file sitting in Library/Translated
        # forever.
        workspace_by_key = self._build_workspace_title_index()

        def _best_guess_restore(lib_path: str,
                                default_parent: str) -> str:
            """Pick a restore destination for *lib_path* when no origin
            entry exists. Tries title-matched workspace first, falls
            back to *default_parent* (the parent of the library dir,
            i.e. where the user is likely to find the file).
            """
            stem = os.path.splitext(os.path.basename(lib_path))[0]
            candidate_keys: set[str] = set()
            k = _norm_book_key(stem)
            if k:
                candidate_keys.add(k)
            try:
                for t in _extract_epub_titles(lib_path):
                    k = _norm_book_key(t)
                    if k:
                        candidate_keys.add(k)
            except Exception:
                logger.debug("Undo title extraction failed: %s",
                             traceback.format_exc())
            for k in candidate_keys:
                ws = workspace_by_key.get(k)
                if not ws:
                    continue
                ws_folder = ws.get("output_folder") or ""
                if ws_folder and os.path.isdir(ws_folder):
                    return os.path.join(
                        ws_folder, os.path.basename(lib_path))
            return os.path.join(
                default_parent, os.path.basename(lib_path))

        raw_orphans: dict[str, str] = {}
        trans_orphans: dict[str, str] = {}
        if os.path.isdir(raw_dir):
            raw_default_parent = os.path.dirname(
                os.path.normpath(raw_dir)) or os.path.expanduser("~")
            try:
                for entry in os.scandir(raw_dir):
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    # Never treat library registry files as content —
                    # a legacy copy of ``library_raw_inputs.txt`` used to
                    # live in ``Library/Raw`` and would trip the
                    # collision prompt if surfaced as an orphan.
                    if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                        continue
                    nl = entry.name.lower()
                    if not (nl.endswith(".epub") or nl.endswith(".txt")
                            or nl.endswith(".pdf") or nl.endswith(".html")):
                        continue
                    if entry.name in raw_map:
                        continue
                    raw_orphans[entry.name] = _best_guess_restore(
                        entry.path, raw_default_parent)
            except (PermissionError, OSError):
                pass
        if os.path.isdir(trans_dir):
            trans_default_parent = os.path.dirname(
                os.path.normpath(trans_dir)) or os.path.expanduser("~")
            try:
                for entry in os.scandir(trans_dir):
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name.lower() in _LIBRARY_TRACKING_FILENAMES:
                        continue
                    if not entry.name.lower().endswith(".epub"):
                        continue
                    if entry.name in trans_map:
                        continue
                    trans_orphans[entry.name] = _best_guess_restore(
                        entry.path, trans_default_parent)
            except (PermissionError, OSError):
                pass

        # Merge orphans into the restore maps so the existing loops
        # below process them alongside registry-backed entries.
        raw_map.update(raw_orphans)
        trans_map.update(trans_orphans)
        return {
            "origins": origins,
            "raw_map": raw_map,
            "trans_map": trans_map,
            "pair_map": pair_map,
            "raw_orphans": raw_orphans,
            "trans_orphans": trans_orphans,
        }

    @staticmethod
    def _undo_prompt_text(plan: dict) -> str:
        """The Undo Move question (counts per shelf plus the orphan note)."""
        raw_map = plan["raw_map"]
        trans_map = plan["trans_map"]
        raw_orphans = plan["raw_orphans"]
        trans_orphans = plan["trans_orphans"]
        raw_count = len(raw_map)
        trans_count = len(trans_map)
        orphan_note = ""
        if raw_orphans or trans_orphans:
            pieces = []
            if trans_orphans:
                pieces.append(f"{len(trans_orphans)} translated")
            if raw_orphans:
                pieces.append(f"{len(raw_orphans)} raw")
            orphan_note = (
                f"\n\n({' + '.join(pieces)} orphan file"
                f"{'s' if (len(raw_orphans) + len(trans_orphans)) != 1 else ''} "
                "had no origins record — these will be moved to the "
                "best-guess matching workspace or to the Library's "
                "parent folder.)"
            )
        return (
            f"Which category do you want to restore to the original location?\n\n"
            f"  \u2022 Raw sources in Library/Raw: {raw_count}\n"
            f"  \u2022 Translated EPUBs in Library/Translated: {trans_count}"
            f"{orphan_note}"
        )

    @staticmethod
    def _undo_collisions(plan: dict, restore_raw: bool, restore_trans: bool) -> list:
        """Restores whose original location is occupied: ``[(library_file, original), ...]``."""
        raw_map = plan["raw_map"]
        trans_map = plan["trans_map"]
        # Pre-scan for restore collisions: any library file whose
        # original location already has a file (different contents or
        # a replacement). Prompt once for a policy applied across the
        # whole Undo batch so Windows' ``shutil.move`` doesn't hard-fail
        # silently on name conflicts.
        undo_collisions: list[tuple[str, str]] = []
        if restore_raw:
            raw_dir_pre = get_library_raw_dir()
            for lib_name, orig_path in raw_map.items():
                lib_file = os.path.join(raw_dir_pre, lib_name)
                if (os.path.isfile(lib_file) and os.path.isfile(orig_path)
                        and os.path.abspath(lib_file) != os.path.abspath(orig_path)):
                    undo_collisions.append((lib_file, orig_path))
        if restore_trans:
            trans_dir_pre = get_library_translated_dir()
            for lib_name, orig_path in trans_map.items():
                lib_file = os.path.join(trans_dir_pre, lib_name)
                if (os.path.isfile(lib_file) and os.path.isfile(orig_path)
                        and os.path.abspath(lib_file) != os.path.abspath(orig_path)):
                    undo_collisions.append((lib_file, orig_path))
        return undo_collisions

    def _execute_undo(self, plan: dict, restore_raw: bool, restore_trans: bool,
                      undo_policy: str = "keep_both", undo_collisions=()) -> dict:
        """Move shelf files back (``_undo_organize_prompt``'s second pass).

        Rewrites ``source_epub.txt`` pointers that pointed into the Library,
        re-registers restored translations that do not land in a workspace,
        prunes origins / pairs and returns counters, errors and path moves.
        """
        origins = plan["origins"]
        raw_map = plan["raw_map"]
        trans_map = plan["trans_map"]
        pair_map = plan["pair_map"]
        restored_raw = 0
        restored_trans = 0
        skipped_undo = 0
        errors: list[str] = []
        # Undo also relocates files — track ``(old_lib_path, orig_path)``
        # pairs so we can emit :attr:`files_reorganized` at the end,
        # mirroring the Organize path. The translator GUI uses this to
        # rewrite any stale ``Library/Raw\x.epub`` path it's still
        # holding back to the restored original location.
        path_moves: list[tuple[str, str]] = []

        colliding_orig = {
            os.path.normcase(os.path.normpath(os.path.abspath(o)))
            for _l, o in undo_collisions
        }

        def _resolve_undo_dest(orig_path: str) -> str | None:
            """Pick a destination under *undo_policy* for a restored file.

            Returns ``None`` when the restore should be skipped entirely.
            """
            key = os.path.normcase(os.path.normpath(
                os.path.abspath(orig_path)))
            if key not in colliding_orig:
                return orig_path
            if undo_policy == "skip":
                return None
            if undo_policy == "replace":
                try:
                    if os.path.isfile(orig_path):
                        os.remove(orig_path)
                except OSError as rm_exc:
                    logger.debug("Replace-on-undo remove failed: %s", rm_exc)
                    # Fall through to keep_both.
                else:
                    return orig_path
            # keep_both (or replace fallback): counter-suffix in the
            # original's parent directory like Explorer does.
            parent = os.path.dirname(orig_path) or "."
            base = os.path.basename(orig_path)
            stem, ext = os.path.splitext(base)
            counter = 2
            while True:
                cand = os.path.join(parent, f"{stem} ({counter}){ext}")
                if not os.path.isfile(cand):
                    return cand
                counter += 1

        if restore_raw and raw_map:
            raw_dir = get_library_raw_dir()
            remaining = {}
            for lib_name, orig_path in raw_map.items():
                lib_file = os.path.join(raw_dir, lib_name)
                if not os.path.isfile(lib_file):
                    errors.append(f"raw:{lib_name}: not found in Library/Raw")
                    continue
                dest_path = _resolve_undo_dest(orig_path)
                if dest_path is None:  # policy = skip
                    remaining[lib_name] = orig_path
                    skipped_undo += 1
                    continue
                try:
                    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, dest_path)
                    restored_raw += 1
                    path_moves.append(
                        (os.path.abspath(lib_file),
                         os.path.abspath(dest_path)))
                    # Any output folders whose source_epub.txt still points
                    # at the library copy get rewritten to where the raw
                    # actually landed (``dest_path`` — may be the original
                    # location or a ``(2)``-suffixed sibling when Keep Both
                    # was chosen on a collision).
                    try:
                        for root in _resolve_output_roots(self._config):
                            try:
                                for sub in os.scandir(root):
                                    if not sub.is_dir(follow_symlinks=False):
                                        continue
                                    sidecar = os.path.join(sub.path, "source_epub.txt")
                                    if not os.path.isfile(sidecar):
                                        continue
                                    try:
                                        with open(sidecar, "r", encoding="utf-8") as f:
                                            raw_text = f.read().strip()
                                    except OSError:
                                        continue
                                    if (os.path.normcase(os.path.normpath(raw_text)) ==
                                            os.path.normcase(os.path.normpath(lib_file))):
                                        try:
                                            with open(sidecar, "w", encoding="utf-8") as f:
                                                f.write(dest_path)
                                        except OSError:
                                            pass
                            except (PermissionError, OSError):
                                continue
                    except Exception:
                        pass
                except Exception as exc:
                    remaining[lib_name] = orig_path
                    errors.append(f"raw:{lib_name}: {exc}")
            origins["raw"] = remaining
            # Drop any pair entries that reference a raw basename we
            # just restored — the raw no longer lives in Library/Raw
            # so a future ``_find_raw_source_for_library_epub`` lookup
            # would otherwise return a stale path.
            restored_basenames = set(raw_map.keys()) - set(remaining.keys())
            if restored_basenames and pair_map:
                pair_map = {
                    tb: rb for tb, rb in pair_map.items()
                    if rb not in restored_basenames
                }

        if restore_trans and trans_map:
            trans_dir = get_library_translated_dir()
            remaining = {}
            for lib_name, orig_path in trans_map.items():
                lib_file = os.path.join(trans_dir, lib_name)
                if not os.path.isfile(lib_file):
                    errors.append(f"translated:{lib_name}: not found in Library/Translated")
                    continue
                dest_path = _resolve_undo_dest(orig_path)
                if dest_path is None:  # policy = skip
                    remaining[lib_name] = orig_path
                    skipped_undo += 1
                    continue
                try:
                    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                except OSError:
                    pass
                try:
                    shutil.move(lib_file, dest_path)
                    restored_trans += 1
                    path_moves.append(
                        (os.path.abspath(lib_file),
                         os.path.abspath(dest_path)))
                    # Re-add the restored file to the translated-
                    # inputs registry ONLY when the restore target
                    # is NOT inside an active output folder.
                    #
                    # Rationale: the translated-inputs registry is
                    # for orphan compiled EPUBs the user dropped
                    # onto the Completed tab from outside any
                    # workspace. When Undo restores a file BACK
                    # INTO an output folder (the typical
                    # post-Organize case), the workspace's own
                    # scan already surfaces the book on the
                    # appropriate tab via the output-folder row —
                    # re-registering would add a SECOND card on
                    # the Completed tab alongside the workspace's
                    # In Progress card, producing the duplicate
                    # the user just hit.
                    dest_parent = os.path.dirname(dest_path)
                    is_workspace_restore = bool(
                        dest_parent
                        and os.path.isfile(os.path.join(
                            dest_parent, "translation_progress.json"))
                    )
                    if not is_workspace_restore:
                        try:
                            record_library_translated_input(dest_path)
                        except Exception:
                            logger.debug(
                                "Failed to re-register restored translated path %s",
                                dest_path,
                            )
                except Exception as exc:
                    remaining[lib_name] = orig_path
                    errors.append(f"translated:{lib_name}: {exc}")
            origins["translated"] = remaining
            # A translated file that's been restored out of Library/
            # Translated can no longer be looked up as a pair key.
            restored_trans_basenames = set(trans_map.keys()) - set(remaining.keys())
            if restored_trans_basenames and pair_map:
                pair_map = {
                    tb: rb for tb, rb in pair_map.items()
                    if tb not in restored_trans_basenames
                }

        origins["pairs"] = pair_map
        _save_origins(origins)
        return {
            "restored_raw": restored_raw,
            "restored_trans": restored_trans,
            "skipped_undo": skipped_undo,
            "errors": errors,
            "path_moves": path_moves,
        }

    @staticmethod
    def _undo_summary(plan: dict, restore_raw: bool, restore_trans: bool,
                      result: dict) -> str:
        """The Undo Move summary message."""
        raw_map = plan["raw_map"]
        trans_map = plan["trans_map"]
        restored_raw = result["restored_raw"]
        restored_trans = result["restored_trans"]
        skipped_undo = result["skipped_undo"]
        errors = result["errors"]
        summary_parts = []
        if restore_raw:
            summary_parts.append(
                f"Raw restored: {restored_raw}/{len(raw_map) if raw_map else 0}")
        if restore_trans:
            summary_parts.append(
                f"Translated restored: {restored_trans}/{len(trans_map) if trans_map else 0}")
        if skipped_undo:
            summary_parts.append(
                f"Skipped {skipped_undo} duplicate"
                f"{'s' if skipped_undo != 1 else ''} at the original location."
            )
        summary = "\n".join(summary_parts) or "Nothing was restored."
        if errors:
            summary += (f"\n\n{len(errors)} error"
                        f"{'s' if len(errors) != 1 else ''}:\n"
                        + "\n".join(errors[:5]))
        return summary

    def _filtered(self, books: list[dict]) -> list[dict]:
        query = self._search.text().strip()
        if query:
            books = [
                book for book in books
                if _book_matches_library_query(book, query)
            ]
        if self._format_filter != FORMAT_ALL:
            books = [
                b for b in books
                if self._format_of_book(b) == self._format_filter
            ]
        return self._sorted_books(books)

    def _scan_diff(self, in_progress: list[dict], completed: list[dict]):
        """Compare a fresh scan with the shelf (``_on_auto_scan_done``'s decision).

        Returns ``(structure_changed, changed_by_tab, new_by_tab)``: paths added or
        removed on a tab, a card entering / leaving the filtered result or a name
        change under A-Z sort need a rebuild; otherwise only the cards whose
        :meth:`_card_signature` changed (``changed_by_tab[tab]``) are replaced.
        """
        old_by_tab = {
            "ip": self._books_by_path(self._in_progress_books),
            "comp": self._books_by_path(self._completed_books),
        }
        new_by_tab = {
            "ip": self._books_by_path(in_progress),
            "comp": self._books_by_path(completed),
        }

        structure_changed = any(
            set(old_by_tab[key]) != set(new_by_tab[key])
            for key in ("ip", "comp")
        )
        changed_by_tab: dict[str, set[str]] = {"ip": set(), "comp": set()}
        if not structure_changed:
            query = self._search.text().strip()
            for tab_key in ("ip", "comp"):
                for path, new_book in new_by_tab[tab_key].items():
                    old_book = old_by_tab[tab_key][path]
                    if self._card_signature(old_book) == self._card_signature(new_book):
                        continue
                    changed_by_tab[tab_key].add(path)

                    # A changed title/tag/format can make a card enter or
                    # leave the current filtered result. That changes page
                    # membership, so targeted replacement is no longer safe.
                    old_matches = self._book_matches_current_filters(
                        old_book, query=query)
                    new_matches = self._book_matches_current_filters(
                        new_book, query=query)
                    if old_matches != new_matches:
                        structure_changed = True
                        break
                    if (
                        self._sort_mode == SORT_NAME
                        and str(old_book.get("name", "")).casefold()
                        != str(new_book.get("name", "")).casefold()
                    ):
                        structure_changed = True
                        break
                if structure_changed:
                    break
        return structure_changed, changed_by_tab, new_by_tab

    @staticmethod
    def _books_by_path(books: list[dict]) -> dict[str, dict]:
        """Return scan rows keyed by their stable Library path."""
        return {
            str(book.get("path", "") or ""): book
            for book in books
            if book.get("path", "")
        }

    def _book_matches_current_filters(
        self, book: dict, *, query: str | None = None
    ) -> bool:
        """Return whether *book* belongs to the currently filtered shelf."""
        if query is None:
            query = self._search.text().strip()
        if query and not _book_matches_library_query(book, query):
            return False
        return (
            self._format_filter == FORMAT_ALL
            or self._format_of_book(book) == self._format_filter
        )

    @staticmethod
    def _card_signature(book: dict) -> tuple:
        """Return a hashable signature of the card-rendering inputs.

        Used by :meth:`_populate_grid_common` to decide whether a
        cached :class:`_BookCard` for a given path is still valid,
        or needs to be rebuilt because the underlying book changed
        (progress advanced, missing-raw badge appeared, etc.).

        Includes fields rendered by :class:`_BookCard` plus searchable tag
        metadata that must keep the cached card's ``book`` payload current.
        A stable signature means filter toggles can reuse the existing widget
        without a :func:`_fit_title_text` shrink loop or a fresh
        :class:`_CoverLoader` thread per card.
        """
        return (
            book.get("path", "") or "",
            book.get("name", "") or "",
            _card_raw_title(book),
            int(book.get("completed_chapters", 0) or 0),
            int(book.get("total_chapters", 0) or 0),
            int(book.get("failed_chapters", 0) or 0),
            int(book.get("pending_chapters", 0) or 0),
            str(book.get("translation_state", "") or ""),
            bool(book.get("is_in_progress", False)),
            bool(book.get("missing_raw_file", False)),
            bool(book.get("has_compiled_output", False)),
            str(book.get("workspace_kind", "") or ""),
            str(book.get("type", "") or ""),
            str(book.get("raw_source_path", "") or ""),
            str(book.get("original_path", "") or ""),
            float(book.get("mtime", 0) or 0),
            int(book.get("size", 0) or 0),
            len(book.get("compiled_conflicts") or []),
            _book_library_tag_values(book),
        )

    @staticmethod
    def _raw_is_in_library_raw(raw_src: str) -> bool:
        """True when *raw_src* lives directly inside ``Library/Raw``.

        A raw sitting in ``Library/Raw`` is resolved by the scanner
        via the implicit ``Library/Raw/<folder_name>.<ext>`` filename
        pattern (route 2 of :func:`_find_raw_source_for_folder`),
        which is NOT a \"saved link\" the user can clear from the
        UI \u2014 the file itself is the match. Detecting that case
        here lets the Clear action hide itself and refuse to run
        for those cards, so the user isn't shown a no-op.
        """
        if not raw_src:
            return False
        try:
            raw_parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(raw_src))))
            lib_raw = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_raw_dir())))
        except (TypeError, ValueError, OSError):
            return False
        return bool(lib_raw) and raw_parent == lib_raw

    @staticmethod
    def _library_raw_match_for_book(book: dict) -> str:
        """Return a ``Library/Raw/<folder_name>.<ext>`` file path if
        one exists on disk for this workspace, else ``""``.

        Mirrors route 2 of :func:`_find_raw_source_for_folder` so
        the Clear gate can detect when the card is effectively
        Library/Raw-backed EVEN IF its cached ``raw_source_path``
        points elsewhere (e.g. the sidecar was written before the
        raw was moved into Library/Raw). When this returns a hit,
        the scanner's next pass WILL resolve the raw via route 2
        regardless of sidecar / registry state, so clearing would
        be a no-op.
        """
        if not isinstance(book, dict):
            return ""
        folder_name = book.get("folder_name") or ""
        if not folder_name:
            ws_folder = _resolve_book_output_folder(book)
            if ws_folder:
                folder_name = os.path.basename(
                    os.path.normpath(ws_folder))
        if not folder_name:
            return ""
        raw_dir = get_library_raw_dir()
        for ext in (".epub", ".txt", ".pdf", ".html"):
            candidate = os.path.join(raw_dir, folder_name + ext)
            if os.path.isfile(candidate):
                return candidate
        return ""

    @staticmethod
    def _card_has_saved_raw_link(book: dict) -> bool:
        """Return True when the card has something to clear.

        A card has a \"saved raw link\" when ANY of the following
        surface the raw source for it:

          1. ``<workspace>/source_epub.txt`` exists on disk
             (written by the translator / Scan-for-Raw).
          2. ``book['raw_source_path']`` is listed in
             ``library_raw_inputs.txt`` (registered through the
             translator run, Import, or Scan-for-Raw's Apply pass).

        Route 2 of :func:`_find_raw_source_for_folder` \u2014 the
        implicit ``Library/Raw/<folder_name>.ext`` filename-pattern
        match \u2014 is NOT considered here because there's
        nothing to \"clear\" (the file itself is the match
        source; removing it would require moving / renaming it).
        Cards whose raw is resolvable via that pattern are
        skipped entirely \u2014 including cards whose cached
        ``raw_source_path`` still points at the pre-move location
        but whose workspace folder name would NOW match a file in
        ``Library/Raw``. The scanner will re-resolve via route 2
        on the next pass regardless of what a sidecar / registry
        entry says, so clearing would read as broken.

        Workspace resolution goes through
        :func:`_resolve_book_output_folder` so library-filed cards
        whose compiled EPUB was organized into ``Library/Translated``
        still resolve to their originating workspace via the
        origins registry.
        """
        if not isinstance(book, dict):
            return False
        raw_src = book.get("raw_source_path") or ""
        # Library/Raw-backed raws don't expose a clearable link
        # \u2014 whether the card's cached ``raw_source_path``
        # points there directly, or a matching file sitting in
        # ``Library/Raw`` is waiting to be picked up by route 2
        # on the next scan.
        if LibraryShelfMixin._raw_is_in_library_raw(raw_src):
            return False
        if LibraryShelfMixin._library_raw_match_for_book(book):
            return False
        # 1. Sidecar on disk.
        ws_folder = _resolve_book_output_folder(book)
        if ws_folder and os.path.isdir(ws_folder):
            sidecar = os.path.join(ws_folder, "source_epub.txt")
            if os.path.isfile(sidecar):
                return True
        # 2. Registry-backed link.
        if not raw_src:
            return False
        try:
            raw_key = os.path.normcase(os.path.normpath(
                os.path.abspath(raw_src)))
        except (TypeError, ValueError):
            return False
        try:
            for p in load_library_raw_inputs():
                if not p:
                    continue
                try:
                    reg_key = os.path.normcase(os.path.normpath(
                        os.path.abspath(p)))
                except Exception:
                    continue
                if reg_key == raw_key:
                    return True
        except Exception:
            pass
        return False

    def _plan_clear_raw_link(self, books: list) -> list:
        """Cards with a clearable raw link: ``[(workspace, old_raw, book, had_sidecar), ...]``.

        A sidecar-backed link deletes ``<workspace>/source_epub.txt``; a
        registry-only link unregisters the raw from ``library_raw_inputs.txt``.
        Cards whose raw sits in ``Library/Raw`` (or would match there by folder
        name) have nothing to clear and are skipped.
        """
        # De-dup by workspace folder and snapshot the current raw
        # pointer so we can optionally unregister it afterwards.
        # ``_resolve_book_output_folder`` is used here (not a raw
        # ``book['output_folder']`` lookup) so library-filed cards
        # whose compiled EPUB was organized into ``Library/Translated``
        # still resolve to their originating workspace via the
        # origins registry \u2014 otherwise the Clear action silently
        # skipped them, which the user observed as an inconsistency.
        #
        # Each target carries the workspace folder (if any), the raw
        # pointer recorded on disk (sidecar content OR
        # ``raw_source_path`` as a fallback for registry-only
        # entries), and a flag telling the deletion pass whether a
        # sidecar actually existed on disk. Cards whose raw was
        # resolved only via the registry have no sidecar to delete
        # but still benefit from unregistering the raw so the next
        # scan can re-derive the match cleanly.
        targets: list[tuple[str, str, dict, bool]] = []
        seen_ws: set[str] = set()
        seen_registry: set[str] = set()
        for b in books:
            # Skip cards whose raw lives in ``Library/Raw`` \u2014 the
            # filename-pattern route resolves those regardless of
            # any sidecar / registry state, so \"clearing\" would be
            # a no-op (the scanner would just re-resolve the raw
            # via route 2 on the next scan). The second check
            # covers cards whose cached ``raw_source_path`` is
            # stale but whose workspace folder name now matches a
            # file in ``Library/Raw`` (post-Organize or manual
            # copy). Without it a leftover sidecar would keep the
            # action visible even though clearing it would have
            # no user-visible effect.
            raw_src_full = b.get("raw_source_path") or ""
            if self._raw_is_in_library_raw(raw_src_full):
                continue
            if self._library_raw_match_for_book(b):
                continue
            ws_folder = _resolve_book_output_folder(b)
            sidecar = ""
            sidecar_raw = ""
            if ws_folder and os.path.isdir(ws_folder):
                cand = os.path.join(ws_folder, "source_epub.txt")
                if os.path.isfile(cand):
                    sidecar = cand
                    try:
                        with open(sidecar, "r", encoding="utf-8") as fh:
                            sidecar_raw = fh.read().strip()
                    except OSError:
                        sidecar_raw = ""
            if sidecar:
                ws_key = os.path.normcase(os.path.normpath(
                    os.path.abspath(ws_folder)))
                if ws_key in seen_ws:
                    continue
                seen_ws.add(ws_key)
                targets.append((ws_folder, sidecar_raw, b, True))
                continue
            # No sidecar \u2014 registry-only link?
            raw_src = raw_src_full
            if not raw_src:
                continue
            try:
                raw_key = os.path.normcase(os.path.normpath(
                    os.path.abspath(raw_src)))
            except (TypeError, ValueError):
                continue
            in_registry = False
            try:
                for p in load_library_raw_inputs():
                    if not p:
                        continue
                    try:
                        reg_key = os.path.normcase(os.path.normpath(
                            os.path.abspath(p)))
                    except Exception:
                        continue
                    if reg_key == raw_key:
                        in_registry = True
                        break
            except Exception:
                in_registry = False
            if not in_registry:
                continue
            if raw_key in seen_registry:
                continue
            seen_registry.add(raw_key)
            # ``ws_folder`` may be empty here \u2014 that's fine, we
            # just skip sidecar deletion and only unregister.
            targets.append((ws_folder or "", raw_src, b, False))
        return targets

    @staticmethod
    def _clear_raw_link_prompt_text(targets: list) -> str:
        """The Clear-saved-raw-link confirmation text for *targets*."""
        # Confirmation prompt: list the affected workspace(s) and the
        # raw path each is currently pointing at. The explanatory
        # footer varies based on whether we're deleting sidecars,
        # unregistering raws, or both \u2014 the user shouldn't see
        # \"only source_epub.txt is deleted\" when a registry-only
        # entry is being cleared.
        preview_lines = []
        for ws_folder, old_raw, b, _had_sidecar in targets[:6]:
            label = (b.get("folder_name")
                     or os.path.basename(ws_folder)
                     or b.get("name") or "")
            if old_raw:
                preview_lines.append(
                    f"  \u2022 {label}\n      \u2192 {old_raw}")
            else:
                preview_lines.append(f"  \u2022 {label}")
        if len(targets) > 6:
            preview_lines.append(f"  \u2026 and {len(targets) - 6} more.")
        has_sidecars = any(t[3] for t in targets)
        has_registry_only = any(not t[3] for t in targets)
        if has_sidecars and has_registry_only:
            footer = (
                "The workspace folders are left untouched \u2014 "
                "``source_epub.txt`` is deleted where present, and "
                "the raw path is unregistered from "
                "``library_raw_inputs.txt``. The next library scan "
                "will re-detect the workspace kind from the folder "
                "contents.")
        elif has_sidecars:
            footer = (
                "The workspace folders are left untouched \u2014 "
                "only ``source_epub.txt`` is deleted. The next "
                "library scan will re-detect the workspace kind "
                "from the folder contents.")
        else:
            footer = (
                "No ``source_epub.txt`` sidecar exists for these "
                "cards \u2014 the raw path will be unregistered from "
                "``library_raw_inputs.txt`` instead. The next "
                "library scan will re-detect the workspace kind "
                "from the folder contents.")
        if len(targets) == 1:
            return (
                "Remove the saved raw-source pointer for this "
                "workspace?\n\n"
                + "\n".join(preview_lines)
                + "\n\n" + footer
            )
        else:
            return (
                f"Remove the saved raw-source pointer for "
                f"{len(targets)} workspace"
                f"{'s' if len(targets) != 1 else ''}?\n\n"
                + "\n".join(preview_lines)
                + "\n\n" + footer
            )

    def _execute_clear_raw_link(self, targets: list) -> int:
        """Delete the sidecars / unregister the raws; returns how many links were cleared."""
        cleared = 0
        for ws_folder, old_raw, _b, had_sidecar in targets:
            if had_sidecar and ws_folder:
                sidecar = os.path.join(ws_folder, "source_epub.txt")
                try:
                    os.remove(sidecar)
                    cleared += 1
                except OSError as exc:
                    logger.debug(
                        "Clear saved raw link failed for %s: %s",
                        sidecar, exc)
                    continue
            else:
                # Registry-only link \u2014 no sidecar to delete, but
                # unregistering the raw still counts as \"cleared\".
                cleared += 1
            if not old_raw:
                continue
            if not had_sidecar:
                # Registry-only clear: the registry entry IS the
                # link for this card. Remove it unconditionally,
                # even if other workspaces reference the same raw
                # via their own ``source_epub.txt`` sidecars \u2014
                # those keep resolving through route 1 and don't
                # depend on the registry. The previous code ran
                # the same \"still referenced\" sweep used for
                # sidecar-based clears, which would see those
                # sidecars and refuse to unregister, leaving this
                # card's link stubbornly in place.
                try:
                    remove_library_raw_input(old_raw)
                except Exception:
                    logger.debug(
                        "Registry-only unregister failed: %s",
                        traceback.format_exc())
                continue
            # Sidecar-based clear: the sweep protects against
            # orphaning a raw that another workspace still needs
            # as a registry fallback. Only drop the registry
            # entry when no OTHER workspace points at it.
            try:
                still_referenced = False
                roots_checked: set[str] = set()
                for root in _resolve_output_roots(self._config):
                    root_key = os.path.normcase(os.path.normpath(
                        os.path.abspath(root)))
                    if root_key in roots_checked:
                        continue
                    roots_checked.add(root_key)
                    try:
                        for entry in os.scandir(root):
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            other_sidecar = os.path.join(
                                entry.path, "source_epub.txt")
                            if not os.path.isfile(other_sidecar):
                                continue
                            try:
                                with open(other_sidecar, "r",
                                          encoding="utf-8") as fh:
                                    val = fh.read().strip()
                            except OSError:
                                continue
                            if not val:
                                continue
                            try:
                                if (os.path.normcase(
                                        os.path.normpath(
                                            os.path.abspath(val)))
                                        == os.path.normcase(
                                            os.path.normpath(
                                                os.path.abspath(old_raw)))):
                                    still_referenced = True
                                    break
                            except Exception:
                                continue
                    except (PermissionError, OSError):
                        continue
                    if still_referenced:
                        break
                if not still_referenced:
                    remove_library_raw_input(old_raw)
            except Exception:
                logger.debug(
                    "Raw-input unregister sweep failed: %s",
                    traceback.format_exc())
        return cleared

    def _plan_delete(self, books: list):
        """Resolve Delete targets: ``(targets, unregister_cards)``.

        ``targets`` are ``(label, path, is_folder, book)`` inside the Library or an
        output root (workspace folders recursively, library files, plus the paired
        raw when it sits in ``Library/Raw``); ``unregister_cards`` are
        ``(book, path)`` outside those safe roots, which are only unregistered.
        """
        # Resolve targets: (label, path, is_folder, book_dict). We keep
        # the book dict so the confirmation dialog can classify each
        # target (Not Started vs. In Progress vs. Completed) and
        # enumerate what's inside a folder target.
        targets: list[tuple[str, str, bool, dict]] = []
        seen_targets: set[str] = set()
        # (book_dict, path) pairs for cards whose backing file lives
        # outside the Library + output-root safe zones. For those we
        # only unregister the tracking-file entry so the flash card
        # disappears — the physical file stays exactly where the
        # user put it. Handled silently: no message box, no summary
        # entry. This is the "Add Translation from Downloads" flow:
        # the Library pointed at the file, the user clicks Delete
        # on the card, and they just want the card gone, not the
        # source EPUB wiped off their drive.
        unregister_cards: list[tuple[dict, str]] = []

        # Safe roots for delete: anything under the Library folder
        # (covers Raw / Translated / registry files) or any configured
        # output root. A path outside ALL of these is considered
        # off-limits — a user-owned file from Downloads or wherever,
        # which ``Delete`` must never touch because Glossarion didn't
        # put it there. Computed once per prompt to keep the queue
        # loop cheap.
        safe_roots: list[str] = []
        try:
            lib_abs = os.path.normcase(os.path.normpath(
                os.path.abspath(get_library_dir())))
            if lib_abs:
                safe_roots.append(lib_abs)
        except Exception:
            logger.debug("Library dir resolve failed: %s",
                         traceback.format_exc())
        try:
            for root in _resolve_output_roots(self._config):
                r = os.path.normcase(os.path.normpath(
                    os.path.abspath(root)))
                if r and r not in safe_roots:
                    safe_roots.append(r)
        except Exception:
            logger.debug("Output roots resolve failed: %s",
                         traceback.format_exc())

        def _is_inside_safe_root(pth: str) -> bool:
            """True when *pth* is inside Library/ or an output root."""
            if not pth or not safe_roots:
                return False
            try:
                key = os.path.normcase(os.path.normpath(
                    os.path.abspath(pth)))
            except Exception:
                return False
            for root in safe_roots:
                if key == root or key.startswith(root + os.sep):
                    return True
            return False

        def _queue(label: str, pth: str, is_folder: bool, book: dict) -> None:
            # Hard safety gate: Delete must never reach outside the
            # Library folder or the configured output roots. A raw
            # EPUB registered in place from Downloads, a stray compiled
            # EPUB dropped onto the Completed tab without Organize,
            # etc. all land here — we route them to the
            # unregister-only path so the flash card disappears but
            # the on-disk file stays intact.
            if not _is_inside_safe_root(pth):
                unregister_cards.append((book, pth))
                return
            key = os.path.normcase(os.path.normpath(os.path.abspath(pth)))
            if key in seen_targets:
                return
            seen_targets.add(key)
            targets.append((label, pth, is_folder, book))

        # Library/Raw absolute path prefix — used to decide whether a
        # card's raw source qualifies for auto-cleanup alongside the
        # workspace. Only raws that actually live inside ``Library/Raw``
        # are deletable; raws anywhere else (Downloads, a user's own
        # folder) are explicitly left untouched so deleting a card
        # never removes files the library didn't put there itself.
        raw_dir_abs = os.path.normcase(os.path.normpath(
            os.path.abspath(get_library_raw_dir())))

        def _queue_library_raw_copy(b: dict) -> None:
            rp = b.get("raw_source_path") or ""
            if not rp or not os.path.isfile(rp):
                return
            rp_parent = os.path.normcase(os.path.normpath(
                os.path.abspath(os.path.dirname(rp))))
            if rp_parent != raw_dir_abs:
                # Raw lives outside Library/Raw — never touch it.
                return
            _queue(os.path.basename(rp), rp, False, b)

        for b in books:
            file_type = b.get("type", "epub")
            in_library = bool(b.get("in_library"))
            output_folder = b.get("output_folder") or ""
            # In-progress cards always delete the folder + (when present)
            # the matching raw in Library/Raw. The raw is only queued
            # when it actually sits inside ``Library/Raw`` so we never
            # wipe a source file that originated from Downloads or any
            # other user directory.
            if file_type == "in_progress":
                folder = output_folder or b.get("path", "") or ""
                if folder and os.path.isdir(folder):
                    _queue(b.get("name") or os.path.basename(folder),
                           folder, True, b)
                _queue_library_raw_copy(b)
                continue
            # Completed but NOT library-filed: the card represents the
            # entire output-folder workspace (compiled file plus any
            # ``response_*``, ``_translated.*``, ``images/``, etc.).
            if (not in_library and output_folder
                    and os.path.isdir(output_folder)):
                _queue(b.get("name") or os.path.basename(output_folder),
                       output_folder, True, b)
                _queue_library_raw_copy(b)
                continue
            # Library entry (or any other loose file-backed card).
            # ``Library/Translated`` cards only own the compiled .epub
            # themselves, but if a paired raw still sits in
            # ``Library/Raw`` we queue it too so one Delete click
            # removes both halves of the library pair. Raws living
            # outside ``Library/Raw`` (e.g. the user's Downloads
            # folder) are left alone by :func:`_queue_library_raw_copy`.
            p = b.get("path", "") or ""
            if p and os.path.isfile(p):
                _queue(b.get("name") or os.path.basename(p), p, False, b)
            _queue_library_raw_copy(b)
        return targets, unregister_cards

    def _unregister_cards(self, unregister_cards) -> int:
        """Silently unregister cards whose files live outside the safe roots."""
        removed = 0
        for bk, pth in unregister_cards:
            try:
                if bk.get("in_library"):
                    # A library-filed EPUB inside Library/Translated
                    # can never reach this branch (it's inside the
                    # safe root), so any ``in_library=True`` card
                    # here is a raw-inputs entry — prune it.
                    remove_library_raw_input(pth)
                elif bk.get("registered_translated"):
                    remove_library_translated_input(pth)
                elif bk.get("type") == "in_progress":
                    # Not-started / in-progress card whose raw
                    # source sits outside Library/Raw: drop it
                    # from the raw-inputs registry so the card
                    # disappears on the next scan.
                    remove_library_raw_input(pth)
                else:
                    # Fallback: try both registries. No-op when
                    # the path isn't in either.
                    remove_library_raw_input(pth)
                    remove_library_translated_input(pth)
                removed += 1
                try:
                    self._selected_paths_ip.discard(pth)
                    self._selected_paths_comp.discard(pth)
                except Exception:
                    pass
            except Exception:
                logger.debug(
                    "Silent unregister failed for %s: %s",
                    pth, traceback.format_exc())
        return removed

    @staticmethod
    def _all_targets_not_started(targets) -> bool:
        """True when every target is a Not Started card (simple confirmation)."""
        # Classify the batch. Simple prompt is only allowed when EVERY
        # target came from a "Not Started" card — any in-progress or
        # completed workspace (including a Library/Translated compiled
        # EPUB) in the selection bumps the whole batch into the
        # typed-keyword prompt.
        def _is_not_started(b: dict) -> bool:
            state = (b.get("translation_state") or "").lower()
            if state:
                return state == "not_started"
            # Fallback for rows without an explicit ``translation_state``
            # field: only *in-progress workspace* cards can be
            # "not_started". Library/Translated entries and compiled
            # output cards have ``type`` set to the file kind
            # (``"epub"`` / ``"pdf"`` / …) and represent finished work —
            # they must ALWAYS take the typed-keyword prompt path,
            # otherwise the simple Yes/Cancel dialog would let a
            # finished compiled EPUB (or worse, a shelf-filed one) be
            # deleted with a single click. Library entries don't
            # populate ``completed_chapters`` / ``has_compiled_output``
            # so the old progress-based heuristic fell through to
            # ``True`` for them — hence the explicit type gate here.
            if b.get("type") != "in_progress":
                return False
            if b.get("in_library"):
                return False
            return (int(b.get("completed_chapters", 0) or 0) == 0
                    and not b.get("has_compiled_output", False))
        return all(_is_not_started(b) for _l, _p, _f, b in targets)

    def _delete_result_summary(self, results, target_count: int):
        """Count a delete batch's results: ``(deleted, errors, summary_text)``."""
        deleted = 0
        errors: list[str] = []
        for result in results or []:
            label, pth, _is_folder, ok, error = result
            if ok:
                deleted += 1
                try:
                    self._selected_paths_ip.discard(pth)
                    self._selected_paths_comp.discard(pth)
                except Exception:
                    pass
            else:
                logger.error("Delete failed for %s: %s", pth, error)
                errors.append(f"{label}: {error}")

        summary = f"Deleted {deleted} of {target_count} item" \
                  f"{'s' if target_count != 1 else ''}."
        if errors:
            summary += (
                f"\n\n{len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}:\n"
                + "\n".join("  - " + e for e in errors[:5])
            )
        return deleted, errors, summary

    # Either keyword unlocks the Delete button in the typed-confirmation
    # dialog. ``"halgakos"`` is the thematic brand-name safeguard;
    # ``"delete"`` is the mundane escape hatch for users who don't want
    # to hunt down the Glossarion mascot. Both are matched after
    # ``.strip().lower()`` so case / surrounding whitespace is forgiven.
    _DELETE_KEYWORDS = ("halgakos", "delete")

    @staticmethod
    def _summarize_folder_contents(folder: str) -> list[str]:
        """Return a bulleted breakdown of artefacts inside *folder*.

        Used to build a detailed "exactly what gets deleted" warning
        for output-folder workspaces. Unknown / unclassified files
        fall into "other files" so the counts always add up.
        """
        counts = {
            "translated chapter HTML files": 0,
            "compiled EPUB files": 0,
            "compiled PDF files": 0,
            "translated text files": 0,
            "translated HTML pages": 0,
            "images": 0,
            "glossary files": 0,
            "progress / history files": 0,
            "other files": 0,
        }
        total_bytes = 0
        try:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    ln = name.lower()
                    fpath = os.path.join(root, name)
                    try:
                        total_bytes += os.path.getsize(fpath)
                    except OSError:
                        pass
                    if ln.startswith("response_") and ln.endswith(
                            (".html", ".htm", ".xhtml")):
                        counts["translated chapter HTML files"] += 1
                    elif ln.endswith(".epub"):
                        counts["compiled EPUB files"] += 1
                    elif "_translated" in ln and ln.endswith(".pdf"):
                        counts["compiled PDF files"] += 1
                    elif "_translated" in ln and ln.endswith(".txt"):
                        counts["translated text files"] += 1
                    elif "_translated" in ln and ln.endswith(
                            (".html", ".htm", ".xhtml")):
                        counts["translated HTML pages"] += 1
                    elif ln.endswith(
                            (".jpg", ".jpeg", ".png", ".webp",
                             ".gif", ".bmp")):
                        counts["images"] += 1
                    elif "glossary" in ln:
                        counts["glossary files"] += 1
                    elif ln in (
                            "translation_progress.json",
                            "translation_history.json",
                            "metadata.json",
                            "source_epub.txt",
                    ):
                        counts["progress / history files"] += 1
                    else:
                        counts["other files"] += 1
        except OSError:
            return []

        lines = [
            f"    \u00b7 {v} {k}"
            for k, v in counts.items() if v > 0
        ]
        if total_bytes:
            if total_bytes >= 1024 * 1024:
                size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_bytes / 1024:.0f} KB"
            lines.append(f"    \u00b7 total on disk: {size_str}")
        return lines

    def _format_delete_detail(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> str:
        """Build a rich "what will be deleted" block for the dialog."""
        lines: list[str] = []
        for label, pth, is_folder, _book in targets[:10]:
            if is_folder:
                lines.append(f"\u25be  {label}  —  output folder")
                lines.append(f"       {pth}")
                contents = self._summarize_folder_contents(pth)
                if contents:
                    lines.extend(contents)
                else:
                    lines.append("    \u00b7 (folder is empty)")
            else:
                try:
                    size = os.path.getsize(pth)
                    if size >= 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    else:
                        size_str = f"{size / 1024:.0f} KB"
                except OSError:
                    size_str = "?"
                lines.append(f"\u25be  {label}  —  file ({size_str})")
                lines.append(f"       {pth}")
            lines.append("")
        if len(targets) > 10:
            lines.append(f"\u2026 and {len(targets) - 10} more item(s).")
        return "\n".join(lines).rstrip()

    def _delete_simple_prompt_text(self, targets) -> str:
        """The Yes / Cancel text for a Not-Started-only delete batch."""
        preview = self._format_delete_detail(targets)
        return (
            f"Permanently delete {len(targets)} "
            f"Not Started item{'s' if len(targets) != 1 else ''}?\n\n"
            f"{preview}\n\nThis cannot be undone."
        )


class BookDetailsLoaderMixin:
    """``_BookDetailsLoader``: OPF metadata + cover + metadata.json (``preview_ready``),
    then per-chapter titles and translation status (``done``). Needs ``_book``,
    ``_config``, ``_should_stop()`` and the three emit hooks.
    """

    def run(self):
        if self._should_stop():
            return
        try:
            book_path = self._book.get("path", "") or ""
            book_type = self._book.get("type", "epub")
            progress_file = self._book.get("progress_file")
            output_folder = self._book.get("output_folder")

            # Tab-driven dispatch:
            #   * in_progress card: ``path`` is an OUTPUT FOLDER. We look for
            #     the source EPUB via ``source_epub.txt`` or any .epub in the
            #     folder; if none is found we still build the details page
            #     from metadata.json + translation_progress.json alone.
            #   * epub card: ``path`` points at a real .epub. We also probe
            #     the sibling translation_progress.json so completed EPUBs
            #     inside an output folder still get per-chapter status.
            source_epub = ""
            if book_type == "in_progress":
                output_folder = output_folder or book_path
                # Prefer the raw_source_path already resolved by the scanner
                # (validated via source_epub.txt, Library/Raw lookup, and the
                # raw-inputs registry). Freshly imported Not Started cards
                # whose output folder only has the sidecar can still surface
                # a cover + full spine this way.
                raw_source = self._book.get("raw_source_path") or ""
                if raw_source and os.path.isfile(raw_source):
                    source_epub = raw_source
                if output_folder and os.path.isdir(output_folder):
                    progress_file = progress_file or os.path.join(
                        output_folder, "translation_progress.json")
                    # Authoritative pointer file second, then any .epub in
                    # the folder (which may be the compiled output).
                    if not source_epub:
                        pointed = _read_source_epub_pointer(output_folder)
                        if pointed:
                            source_epub = pointed
                    if not source_epub:
                        for entry in os.scandir(output_folder):
                            if (entry.is_file(follow_symlinks=False)
                                    and entry.name.lower().endswith(".epub")):
                                source_epub = entry.path
                                break
            else:
                source_epub = book_path if os.path.isfile(book_path) else ""
                if not progress_file or not output_folder:
                    parent_dir = os.path.dirname(book_path)
                    if parent_dir and os.path.isdir(parent_dir):
                        candidate_pf = os.path.join(parent_dir, "translation_progress.json")
                        if os.path.isfile(candidate_pf):
                            progress_file = progress_file or candidate_pf
                            output_folder = output_folder or parent_dir

            # Only treat the resolved source as an EPUB when its extension
            # actually matches — TXT/PDF raw sources should skip the
            # zip-based parsing so they don't produce empty details silently.
            source_is_epub = (bool(source_epub)
                              and source_epub.lower().endswith(".epub")
                              and os.path.isfile(source_epub))
            source_is_pdf = bool(source_epub) and source_epub.lower().endswith(".pdf")

            # ---- Phase 1: Fast metadata + cover (no per-chapter HTML) ----
            # Skips the BeautifulSoup pass over every spine chapter so the
            # details hero paints instantly; the full chapter titles are
            # re-parsed in Phase 2 below and emitted via ``done``.
            details = _parse_epub_details(source_epub, parse_chapter_titles=False) if source_is_epub else {
                "title": "", "authors": [], "publisher": "", "language": "",
                "date": "", "description": "", "subjects": [], "identifier": "",
                "chapters": [],
            }
            cover = _extract_cover(source_epub) if source_is_epub else None
            # Broaden cover search so TXT/PDF workspaces that keep a cover
            # next to their compiled output still get a real thumbnail, and
            # so EPUBs whose embedded cover extraction somehow fails fall
            # back to any image the output folder has on disk.
            if not cover and output_folder and os.path.isdir(output_folder):
                cover = _find_cover_in_dir(output_folder)

            # Load metadata.json eagerly so the preview already has the
            # translator-overriden title / authors / description.
            metadata_json = None
            if output_folder:
                meta_path = os.path.join(output_folder, "metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        import json as _json
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata_json = _json.load(f)
                    except Exception:
                        metadata_json = None

            # Emit the preview so the dialog can paint the hero row now.
            if not self._should_stop():
                self.preview_ready.emit({
                    "details": details,
                    "cover": cover or "",
                    "metadata_json": metadata_json or {},
                })
            else:
                return

            # ---- Phase 2: Slow per-chapter title parsing ----
            # Re-parse the spine WITH per-chapter HTML title extraction so
            # the chapter list can show "Prologue", "Chapter 1: ..." etc.
            # rather than just filename stubs.
            if source_is_epub:
                details = _parse_epub_details(source_epub, parse_chapter_titles=True)
                if self._should_stop():
                    return

            prog = None
            if progress_file and os.path.isfile(progress_file):
                summary = _read_progress_summary(progress_file)
                if summary is not None:
                    prog = summary["prog"]

            # Resolve each spine chapter to a translation status.
            chapters_info = []
            all_prog_chapters = (prog or {}).get("chapters", {}) or {}
            # Workspace sidecars are bookkeeping, not readable content. They
            # must not become synthesized Book Details rows when a TXT/PDF
            # source has no EPUB spine to supply the chapter list.
            prog_chapters = {
                key: info
                for key, info in all_prog_chapters.items()
                if (isinstance(info, dict)
                    and not _is_progress_sidecar_entry(key, info))
            }
            prog_key_by_id = {id(info): str(key) for key, info in prog_chapters.items()}
            # Build lookup by normalized basename (no extension, no response_ prefix).
            def _norm(name: str) -> str:
                base = os.path.basename(name or "")
                if base.lower().startswith("response_"):
                    base = base[len("response_"):]
                while True:
                    stem, ext = os.path.splitext(base)
                    if not ext:
                        break
                    base = stem
                return base.lower()

            # --- Paired raw-EPUB title harvest (library entries only) ---
            #
            # For a Completed-tab library entry the EPUB we just parsed IS
            # the compiled translation — every ``ch['title']`` is already
            # translated, so there's no distinction between raw and
            # translated titles. When a paired raw EPUB exists
            # (``raw_source_path`` from the scanner, or resolved via
            # :func:`_find_raw_source_for_library_epub`), parse its spine
            # too and remember a filename-normalized + index-based lookup
            # of source-language titles. :func:`_resolve_chapter` below
            # then swaps the title semantics so the BookDetails
            # "Show raw titles" toggle has something to flip to.
            library_raw_title_by_norm: dict[str, str] = {}
            library_raw_titles_by_index: list[str] = []
            if (self._book.get("in_library")
                    and not self._book.get("is_in_progress")):
                raw_path = self._book.get("raw_source_path", "") or ""
                if not raw_path:
                    try:
                        raw_path = _find_raw_source_for_library_epub(book_path) or ""
                    except Exception:
                        raw_path = ""
                        logger.debug("Library-raw title resolve failed: %s",
                                     traceback.format_exc())
                if (raw_path and os.path.isfile(raw_path)
                        and raw_path.lower().endswith(".epub")):
                    try:
                        raw_details = _parse_epub_details(
                            raw_path, parse_chapter_titles=True)
                    except Exception:
                        raw_details = None
                        logger.debug("Raw EPUB parse failed for %s: %s",
                                     raw_path, traceback.format_exc())
                    for rc in (raw_details or {}).get("chapters", []) or []:
                        rt = (rc.get("title") or "").strip()
                        library_raw_titles_by_index.append(rt)
                        fn = rc.get("filename") or ""
                        if fn and rt:
                            library_raw_title_by_norm.setdefault(_norm(fn), rt)

            prog_by_basename: dict[str, dict] = {}
            prog_by_output: dict[str, dict] = {}
            for key, info in prog_chapters.items():
                if not isinstance(info, dict):
                    continue
                ob = info.get("original_basename") or ""
                of = info.get("output_file") or ""
                if ob:
                    prog_by_basename.setdefault(_norm(ob), info)
                if of:
                    prog_by_output.setdefault(_norm(of), info)

            # If we could neither load a progress file nor see an output
            # folder on disk, there is no translation context for this book
            # at all — we leave ``status`` empty so the UI renders no badge
            # (instead of misleadingly labeling every chapter "Pending").
            has_progress_context = bool(prog is not None
                                         or (output_folder and os.path.isdir(output_folder)))

            # Source EPUB absent: synthesize a spine from the progress file's
            # ``original_basename`` entries so the Chapters list still works.
            if not details.get("chapters") and prog_chapters:
                def _sort_key(item):
                    info = item[1]
                    try:
                        return int(info.get("actual_num") or info.get("chapter_num") or 0)
                    except (TypeError, ValueError):
                        return 0
                synth = []
                for key, info in sorted(prog_chapters.items(), key=_sort_key):
                    if not isinstance(info, dict):
                        continue
                    ob = info.get("original_basename") or info.get("output_file") or key
                    if source_is_pdf and not info.get("pdf_toc_section"):
                        output_ext = os.path.splitext(str(
                            info.get("output_file") or ""
                        ))[1].lower()
                        original_ext = os.path.splitext(str(
                            info.get("original_basename") or ""
                        ))[1].lower()
                        if not ({output_ext, original_ext}
                                & {".html", ".htm", ".xhtml"}):
                            continue
                    title = (
                        info.get("pdf_toc_title_translated")
                        or info.get("translated_title")
                        or info.get("pdf_section_title_translated")
                        or info.get("pdf_toc_title")
                        or info.get("pdf_section_title")
                        or info.get("title")
                        or ob
                    )
                    if title == ob:
                        title = os.path.splitext(os.path.basename(title))[0]
                        title = title.replace("_", " ").replace("-", " ").strip() or title
                    synth.append({
                        "href": ob,
                        "filename": os.path.basename(ob),
                        "title": title,
                    })
                if synth:
                    details["chapters"] = synth

            # Resolve every chapter's on-disk translation state in parallel.
            # Each per-chapter task is file-I/O bound (a few ``os.path.isfile``
            # probes + a 32 KB read of the matching translated HTML), so a
            # small thread pool turns a serial ~N × latency walk over a
            # 400-chapter output folder into something that finishes in the
            # time of a couple of sequential disk hits.
            chapters = details.get("chapters", []) or []
            output_dir_ok = bool(output_folder and os.path.isdir(output_folder))
            def _resolve_chapter(item):
                idx, ch = item
                filename = ch["filename"]
                raw_title = ch["title"]
                is_special = _is_configured_special_file(filename, self._config)
                is_gallery = _is_gallery_filename(filename)
                norm_key = _norm(filename)
                match = prog_by_basename.get(norm_key) or prog_by_output.get(norm_key)
                status = (match or {}).get("status", "")
                output_file = (match or {}).get("output_file", "")
                translated_title = ""
                translated_path = ""
                if output_dir_ok:
                    candidate_names = []
                    if output_file:
                        candidate_names.append(output_file)
                    base = os.path.splitext(filename)[0]
                    candidate_names.append(f"response_{base}.html")
                    candidate_names.append(f"response_{base}.xhtml")
                    candidate_names.append(f"{base}.html")
                    candidate_names.append(f"{base}.xhtml")
                    for candidate in candidate_names:
                        p = os.path.join(output_folder, candidate)
                        if os.path.isfile(p):
                            translated_path = p
                            if not status:
                                status = "completed"
                            break
                    if translated_path:
                        translated_title = _read_translated_chapter_title(translated_path)
                # Library-paired case: the loader parsed the COMPILED
                # EPUB, so ``raw_title`` right now is already translated.
                # Swap it with the paired raw EPUB's title (filename
                # normalized first, then index as a fallback) and promote
                # the compiled title to ``translated_title`` so
                # _ChapterRow + the Show-raw-titles toggle behave the
                # same way they do for in-progress books.
                if library_raw_title_by_norm or library_raw_titles_by_index:
                    paired_raw = library_raw_title_by_norm.get(norm_key, "")
                    if (not paired_raw
                            and idx < len(library_raw_titles_by_index)):
                        paired_raw = library_raw_titles_by_index[idx]
                    if paired_raw and paired_raw != raw_title:
                        translated_title = raw_title  # compiled title
                        raw_title = paired_raw
                        if not status:
                            # The book IS a completed translation — mark
                            # the row as completed so the default title
                            # policy in _ChapterRow picks the translated
                            # version (not the raw).
                            status = "completed"
                if not status:
                    status = "pending" if has_progress_context else ""
                if is_gallery:
                    status = ""
                resolved = {
                    "index": idx,
                    "filename": filename,
                    "raw_title": raw_title,
                    "translated_title": translated_title,
                    "translated_path": translated_path,
                    "status": status,
                    "is_special": is_special,
                    "is_gallery": is_gallery,
                }
                if match:
                    resolved["progress_key"] = prog_key_by_id.get(id(match), "")
                    resolved["output_file"] = output_file
                    chunk_key = str(
                        match.get("content_hash")
                        or resolved["progress_key"]
                        or ""
                    )
                    chunk_entry = (prog or {}).get("chapter_chunks", {}).get(
                        chunk_key
                    )
                    if is_multi_chunk_entry(chunk_entry):
                        ensure_chunk_entry_schema(chunk_entry)
                        summary = chunk_failure_summary(chunk_entry)
                        resolved["status"] = effective_parent_status(
                            resolved.get("status"),
                            chunk_entry,
                        )
                        resolved["chunk_progress_key"] = chunk_key
                        resolved["chunk_summary"] = summary
                        resolved["chunk_status_text"] = (
                            chunk_status_summary_text(chunk_entry, limit=50)
                        )
                        resolved["chunks"] = [
                            {
                                "index": int(chunk_index),
                                "status": record.get("status", "pending"),
                                "qa_issues_found": list(
                                    record.get("qa_issues_found") or []
                                ),
                                "model_name": record.get("model_name"),
                                "key_identifier": record.get("key_identifier"),
                            }
                            for chunk_index, record in sorted_chunk_items(
                                chunk_entry.get("entries", {})
                            )
                            if isinstance(record, dict)
                        ]
                    for pdf_key in (
                        "pdf_toc_section",
                        "pdf_toc_title",
                        "pdf_toc_title_original",
                        "pdf_toc_title_translated",
                        "pdf_section_title_translated",
                        "pdf_section_id",
                        "pdf_start_page",
                        "pdf_end_page",
                    ):
                        if match.get(pdf_key) is not None:
                            resolved[pdf_key] = match.get(pdf_key)
                return resolved

            chapters_info = []
            if chapters:
                items = list(enumerate(chapters))
                try:
                    max_workers = _reader_worker_count(
                        len(items), config=self._config)
                    if max_workers <= 1:
                        chapters_info = []
                        for it in items:
                            if self._should_stop():
                                return
                            chapters_info.append(_resolve_chapter(it))
                    else:
                        with ThreadPoolExecutor(max_workers=max_workers) as pool:
                            chapters_info = list(pool.map(_resolve_chapter, items))
                except Exception:
                    logger.debug("Parallel chapter resolve failed, falling back: %s",
                                 traceback.format_exc())
                    chapters_info = []
                    for it in items:
                        if self._should_stop():
                            return
                        chapters_info.append(_resolve_chapter(it))

            # metadata_json was loaded in Phase 1 above.

            if not self._should_stop():
                self.done.emit({
                    "details": details,
                    "cover": cover or "",
                    "chapters_info": chapters_info,
                    "metadata_json": metadata_json or {},
                    "progress": prog or {},
                })
        except Exception as exc:
            if not self._should_stop():
                logger.error("Book details load error: %s\n%s", exc, traceback.format_exc())
                self.error.emit(f"{exc}")


class BookDetailsMixin:
    """``BookDetailsDialog`` decisions without widgets: chapter filters and counts, the
    progress strip, tags / authors, metadata editor values and save, the reader
    overlay and the reader-open plan. State: ``_book``, ``_config``, ``_details``,
    ``_metadata_json``, ``_chapters_info``, ``_show_special_files``,
    ``_show_qa_failures_only`` and ``_toc_search`` (``.text()``).
    """

    def _progress_strip_text(self):
        """The Book Details progress strip text, or None when it is hidden."""
        if not self._book.get("is_in_progress"):
            return None
        # Gallery pages are unconditionally excluded (translator-generated,
        # not real source chapters). ``is_special`` is additionally used
        # to honor the user's "Show special files" checkbox.
        if self._show_special_files:
            progress_items = [c for c in self._chapters_info
                              if not c.get("is_gallery")]
        else:
            progress_items = [c for c in self._chapters_info
                              if not c.get("is_special")
                              and not c.get("is_gallery")]
        done = sum(
            1
            for c in progress_items
            if c.get("status") == "completed"
            and not (c.get("chunk_summary") or {}).get("failed")
            and not (c.get("chunk_summary") or {}).get("pending")
        )
        total = len(progress_items) or int(self._book.get("total_chapters", 0) or 0)
        # When the book has reached 100% translation, the card already
        # renders on the Completed tab without an "in progress" ribbon
        # (see :func:`split_output_folders_by_status`). Hide the details
        # strip too so the dialog doesn't contradict the card.
        translation_done = bool(total) and done >= total
        state = self._book.get("translation_state") or ""
        if translation_done or state == "completed":
            return None
        if total:
            pct = int((done * 100) // total)
            return (
                f"\u23f3  Translation in progress \u2014 {done}/{total} chapters ({pct}%)"
            )
        return "\u23f3  Translation in progress"

    def _collect_tag_values(self, *sources) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            tag = str(value or "").strip().strip(",;")
            if not tag:
                return
            key = tag.casefold()
            if key in seen:
                return
            seen.add(key)
            tags.append(tag)

        for source in sources:
            if isinstance(source, str):
                values = [source]
            else:
                try:
                    values = list(source or [])
                except TypeError:
                    values = [source]
            for raw in values:
                text = str(raw or "").strip()
                if not text:
                    continue
                if "#" in text:
                    parts = [
                        m.group(1).strip().strip(",;")
                        for m in re.finditer(r"#([^#]+)", text)
                    ]
                else:
                    parts = [p.strip() for p in re.split(r"[,;]", text)]
                for part in parts:
                    add(part)

        return tags

    def _metadata_author_values(self) -> list[str]:
        """Return output-metadata creators, then source EPUB authors."""
        authors = self._metadata_json.get("creator")
        if not authors:
            authors = self._metadata_json.get("authors")
        if not authors:
            authors = self._details.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        try:
            values = list(authors or [])
        except TypeError:
            values = [authors]
        return [str(author).strip() for author in values if str(author).strip()]

    def _display_tag_values(self) -> list[str]:
        """Return translated output tags, falling back to source EPUB tags."""
        metadata_tag_keys = ("subject", "subjects", "genres", "tags")
        if any(key in self._metadata_json for key in metadata_tag_keys):
            return self._collect_tag_values(
                *(self._metadata_json.get(key) or []
                  for key in metadata_tag_keys)
            )
        return self._collect_tag_values(self._details.get("subjects") or [])

    def _metadata_editor_values(self) -> dict:
        """Return the effective values currently shown on the details page."""
        title = (
            self._metadata_json.get("title")
            or self._details.get("title")
            or self._book.get("name", "")
        )
        publisher = (
            self._metadata_json.get("publisher")
            or self._details.get("publisher")
            or ""
        )
        language = (
            self._metadata_json.get("language")
            or self._details.get("language")
            or ""
        )
        date = (
            self._metadata_json.get("date")
            or self._details.get("date")
            or ""
        )
        description = (
            self._metadata_json.get("description")
            or self._details.get("description")
            or ""
        )
        return {
            "title": str(title or ""),
            "creator": ", ".join(self._metadata_author_values()),
            "publisher": str(publisher or ""),
            "language": str(language or ""),
            "date": str(date or ""),
            "subject": ", ".join(self._display_tag_values()),
            "description": str(description or ""),
        }

    def _source_metadata_values(self) -> dict:
        """Return source-EPUB values used to preserve original_* fields."""
        raw_authors = self._details.get("authors") or []
        if isinstance(raw_authors, str):
            raw_authors = [raw_authors]
        return {
            "title": self._details.get("title") or "",
            "creator": ", ".join(
                str(author).strip()
                for author in raw_authors
                if str(author).strip()
            ),
            "publisher": self._details.get("publisher") or "",
            "language": self._details.get("language") or "",
            "date": self._details.get("date") or "",
            "description": self._details.get("description") or "",
            "subject": self._details.get("subjects") or [],
        }

    def _save_metadata_edits(self, output_folder: str, edits: dict):
        """Merge *edits* into ``<output_folder>/metadata.json`` and save it atomically.

        Keeps ``original_<field>`` and sets ``<field>_translated`` (see
        :func:`_merge_manual_metadata_edits`). Returns the saved dict, None when
        nothing changed; raises :class:`_MetadataEditError` with the dialog text.
        """
        metadata_path = os.path.join(output_folder, "metadata.json")
        current_metadata = dict(self._metadata_json or {})
        if os.path.isfile(metadata_path):
            try:
                import json as _json
                with open(metadata_path, "r", encoding="utf-8") as stream:
                    loaded = _json.load(stream)
                if isinstance(loaded, dict):
                    current_metadata = loaded
            except Exception as exc:
                raise _MetadataEditError(
                    f"Could not read metadata.json:\n{exc}")

        updated, changed_fields = _merge_manual_metadata_edits(
            current_metadata,
            edits,
            self._source_metadata_values(),
        )
        if not changed_fields:
            return None

        temp_path = ""
        try:
            import json as _json
            file_descriptor, temp_path = tempfile.mkstemp(
                prefix=".metadata-",
                suffix=".json.tmp",
                dir=output_folder,
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                _json.dump(updated, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temp_path, metadata_path)
            temp_path = ""
        except Exception as exc:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise _MetadataEditError(
                f"Could not save metadata.json:\n{exc}")
        return updated

    def _visible_counts(self) -> tuple[int, int]:
        """Return (done, total) considering the special-files toggle.

        Gallery pages are unconditionally excluded — they're
        translator-generated artefacts, not real source chapters.
        """
        items = self._chapter_base_infos()
        total = len(items)
        done = sum(
            1
            for c in items
            if c.get("status") == "completed"
            and not (c.get("chunk_summary") or {}).get("failed")
            and not (c.get("chunk_summary") or {}).get("pending")
        )
        return done, total

    def _has_progress_context(self) -> bool:
        """True when at least one chapter has a non-empty translation status."""
        return any((c.get("status") or "") for c in self._chapters_info)

    def _toc_toggle_state(self):
        """``(text, tooltip)`` of the Chapters / Failures toggle button."""
        if self._show_qa_failures_only:
            failure_count = sum(
                1 for chapter in self._chapter_base_infos()
                if (
                    str(chapter.get("status") or "").strip().lower()
                    == "qa_failed"
                    or bool((chapter.get("chunk_summary") or {}).get("failed"))
                )
            )
            return f"Failures  ({failure_count})", "Show all chapters"

        done, total = self._visible_counts()
        prefix = "Chapters"
        if not total:
            suffix = "  (\u2014)"
        elif self._has_progress_context():
            suffix = f"  ({done}/{total})"
        else:
            # No progress file anywhere — just show the total count without a
            # misleading completed/total fraction.
            suffix = f"  ({total})"
        return prefix + suffix, "Show QA failures only"

    def _chapter_base_infos(self) -> list[dict]:
        if self._show_special_files:
            return [
                c for c in self._chapters_info
                if not c.get("is_gallery")
            ]
        return [
            c for c in self._chapters_info
            if not c.get("is_special") and not c.get("is_gallery")
        ]

    def _filtered_chapter_infos(self) -> list[dict]:
        items = self._chapter_base_infos()
        if self._show_qa_failures_only:
            items = [
                info for info in items
                if (
                    str(info.get("status") or "").strip().lower()
                    == "qa_failed"
                    or bool((info.get("chunk_summary") or {}).get("failed"))
                )
            ]
        search = getattr(self, "_toc_search", None)
        needle = (search.text() if search is not None else "")
        needle = (needle or "").strip().lower()
        if not needle:
            return items
        filtered = []
        for info in items:
            hay = " ".join(str(x) for x in (
                info.get("raw_title", ""),
                info.get("translated_title", ""),
                info.get("filename", ""),
                info.get("chunk_status_text", ""),
                " ".join(
                    str(issue)
                    for chunk in info.get("chunks", [])
                    for issue in chunk.get("qa_issues_found", [])
                ),
            )).lower()
            if needle in hay:
                filtered.append(info)
        return filtered

    def _build_translated_overlay(self) -> tuple[dict[str, dict], list[str]]:
        """Return (overlay, extra_image_dirs) for a translated reader view.

        The overlay maps the source chapter's filename (lowercased basename)
        → translated HTML path + title so the reader can swap source content
        with translated content in place. Keying by filename (rather than
        index) avoids ordering/skip mismatches between the reader's loader
        (manifest order, filters short chapters) and our own spine-based
        parser. Image directories let the reader resolve assets that only
        exist in the translator's output.
        """
        overlay: dict[str, dict] = {}
        for ci in self._chapters_info:
            path = ci.get("translated_path") or ""
            if not path or not os.path.isfile(path):
                continue
            filename = ci.get("filename") or ""
            if not filename:
                continue
            key = os.path.basename(filename).lower()
            if not key:
                continue
            overlay[key] = {
                "path": path,
                "title": ci.get("translated_title") or "",
                # A response file can exist even though the translation failed
                # QA.  Preserve the progress status so the reader does not
                # mistake every on-disk response for a completed chapter.
                "status": str(ci.get("status") or "").strip().lower(),
            }
        extra_dirs: list[str] = []
        output_folder = self._book.get("output_folder")
        if output_folder and os.path.isdir(output_folder):
            for sub in ("images", "translated_images"):
                candidate = os.path.join(output_folder, sub)
                if os.path.isdir(candidate):
                    extra_dirs.append(candidate)
        return overlay, extra_dirs

    def _translated_css_dirs(self) -> list[str]:
        """Return CSS directories that belong to the translated output folder."""
        output_folder = self._book.get("output_folder")
        if not output_folder:
            for ci in self._chapters_info:
                path = ci.get("translated_path") or ""
                if path and os.path.isfile(path):
                    output_folder = os.path.dirname(path)
                    break
        if not output_folder or not os.path.isdir(output_folder):
            return []
        dirs: list[str] = []
        css_dir = os.path.join(output_folder, "css")
        dirs.append(css_dir)
        # Some HTML outputs keep styles directly beside the responses.
        try:
            if any(name.lower().endswith(".css")
                   for name in os.listdir(output_folder)):
                dirs.append(output_folder)
        except OSError:
            pass
        return dirs

    def _plan_open_reader(self, initial_chapter: int | None = None,
                          raw_only: bool = False, busy=None) -> dict:
        """Decide how Book Details opens a book (``_open_reader`` minus the Qt dialog).

        ``{"mode": "workspace" | "epub", "source", "kwargs"}`` for the integrated
        reader (``EpubReaderDialog(source, config=..., **kwargs)``) or
        ``{"mode": "system", "target"}`` for the OS viewer (``target`` may be "").
        ``busy`` (optional, no arguments) is called exactly where the desktop
        method set its wait cursor: on entering the workspace / EPUB branch,
        before the overlay is built. The system-viewer path never calls it.
        """
        book_path = self._book.get("path", "") or ""
        raw_source = self._book.get("raw_source_path", "") or ""
        compiled = self._book.get("compiled_output_path", "") or ""
        output_folder = (
            self._book.get("output_folder")
            or _resolve_book_output_folder(self._book)
            or ""
        )
        if output_folder and not raw_source:
            pointed_source = _read_source_epub_pointer(output_folder) or ""
            if pointed_source and os.path.isfile(pointed_source):
                raw_source = pointed_source

        # A translated PDF workspace already has the same ordered HTML
        # chapter model the reader needs.  Use it directly instead of
        # falling through to the system PDF viewer.  Raw mode is supplied
        # by the reader's lazy bookmark-range cache, so opening this dialog
        # never extracts the whole PDF again.
        if (output_folder and os.path.isdir(output_folder)
                and raw_source.lower().endswith(".pdf")
                and os.path.isfile(raw_source)
                and os.path.isfile(os.path.join(
                    output_folder, "translation_progress.json"))):
            if busy is not None:
                busy()
            initial_filename = None
            if (isinstance(initial_chapter, int)
                    and 0 <= initial_chapter < len(self._chapters_info)):
                info = self._chapters_info[initial_chapter] or {}
                initial_filename = (
                    info.get("output_file")
                    or info.get("filename")
                    or None
                )
            has_translated = any(
                c.get("translated_path") for c in self._chapters_info
            )
            title = (
                self._metadata_json.get("title")
                or self._details.get("title")
                or self._book.get("name")
            )
            return {
                "mode": "workspace",
                "source": raw_source,
                "kwargs": {
                    "initial_chapter": initial_chapter,
                    "initial_chapter_filename": initial_filename,
                    "window_title": (f"{title} (Translated)" if title else None),
                    "workspace_dir": output_folder,
                    "initial_show_raw": bool(raw_only or not has_translated),
                },
            }

        def _is_epub_file(p: str) -> bool:
            return bool(p) and p.lower().endswith(".epub") and os.path.isfile(p)

        # Resolve the EPUB to hand to EpubReaderDialog. For in-progress
        # cards ``book['path']`` is the OUTPUT FOLDER (not a file), so we
        # MUST consult ``raw_source_path`` / ``compiled_output_path`` too.
        # Priority:
        #   * raw_only or is_in_progress → raw_source first (that's the
        #     reader base the translated overlay sits on top of).
        #   * completed / library        → book_path first (compiled or
        #     library EPUB is what the user wants to read).
        if raw_only or self._book.get("is_in_progress"):
            epub_candidates = [raw_source, book_path, compiled]
        else:
            epub_candidates = [book_path, raw_source, compiled]
        epub_for_reader = next(
            (p for p in epub_candidates if _is_epub_file(p)), ""
        )

        if epub_for_reader:
            if busy is not None:
                busy()
            overlay: dict[str, dict] = {}
            extra_dirs: list[str] = []
            translated_css_dirs: list[str] = []
            overlay_provider = None
            window_title = None
            # Start polling even before the first translated chapter lands.
            if not raw_only and self._book.get("is_in_progress"):
                overlay, extra_dirs = self._build_translated_overlay()
                if output_folder:
                    from reader_overlay import make_epub_overlay_provider

                    overlay_provider = make_epub_overlay_provider(
                        output_folder,
                        [ci.get("filename") for ci in self._chapters_info],
                        initial_overlay=overlay,
                    )
                    refreshed_overlay = overlay_provider()
                    if refreshed_overlay is not None:
                        overlay, extra_dirs = refreshed_overlay
                translated_css_dirs = self._translated_css_dirs()
                if overlay:
                    # Derive the displayed title from the metadata.json /
                    # OPF title, falling back to the book's name.
                    window_title = (self._metadata_json.get("title")
                                    or self._details.get("title")
                                    or self._book.get("name"))
                    if window_title:
                        window_title = f"{window_title} (Translated)"
            # Translate the spine-index initial_chapter into a filename so the
            # reader resolves it against its own (manifest-ordered) chapter
            # list. This also prevents off-by-one jumps when the source EPUB
            # has nav/toc items that are skipped by the reader's loader.
            initial_filename = None
            if isinstance(initial_chapter, int) and 0 <= initial_chapter < len(self._chapters_info):
                initial_filename = self._chapters_info[initial_chapter].get("filename") or None
            # Completed-tab mode (no overlay, has raw source): let the
            # reader flip between the compiled EPUB (book_path) and
            # the resolved raw source. Skipped when an overlay is
            # active — overlay mode handles the Raw toggle by
            # swapping in-memory chapter lists instead of reloading.
            alt_for_reader = ""
            if (not overlay and not raw_only
                    and not self._book.get("is_in_progress")
                    and raw_source
                    and os.path.isfile(raw_source)
                    and raw_source.lower().endswith(".epub")):
                try:
                    if os.path.normcase(os.path.abspath(raw_source)) != \
                            os.path.normcase(os.path.abspath(epub_for_reader)):
                        alt_for_reader = raw_source
                except Exception:
                    alt_for_reader = ""
            # The provider owns only workspace paths and source filenames;
            # hidden Book Details rows cannot leave the reader stale.
            return {
                "mode": "epub",
                "source": epub_for_reader,
                "kwargs": {
                    "initial_chapter": initial_chapter,
                    "initial_chapter_filename": initial_filename,
                    "translated_overlay": overlay or None,
                    "extra_image_dirs": extra_dirs or None,
                    "translated_css_dirs": translated_css_dirs or None,
                    "window_title": window_title,
                    # Propagate the dialog's current toggle so the reader's
                    # TOC matches what the Book Details chapter list
                    # shows (configured special files hidden when this is off).
                    "show_special_files": self._show_special_files,
                    "alt_epub_path": alt_for_reader or None,
                    "overlay_provider": overlay_provider,
                    "toc_output_dir": output_folder or None,
                },
            }

        # No EPUB base resolvable — the workspace is TXT / PDF / HTML /
        # image. Hand off to the OS default viewer with a concrete file.
        chapter_translated = ""
        if (not raw_only
                and isinstance(initial_chapter, int)
                and 0 <= initial_chapter < len(self._chapters_info)):
            tp = self._chapters_info[initial_chapter].get("translated_path", "") or ""
            if tp and os.path.isfile(tp):
                chapter_translated = tp

        book_path_is_file = bool(book_path) and os.path.isfile(book_path)
        target = ""
        if raw_only:
            if raw_source and os.path.isfile(raw_source):
                target = raw_source
            elif book_path_is_file:
                target = book_path
        else:
            if chapter_translated:
                target = chapter_translated
            elif compiled and os.path.isfile(compiled):
                target = compiled
            elif book_path_is_file:
                target = book_path
            elif raw_source and os.path.isfile(raw_source):
                target = raw_source
        return {"mode": "system", "target": target}

    def _resolve_output_folder_target(self) -> str:
        """Return the output-folder path the 📁 button should open, or "".

        Thin wrapper around :func:`_resolve_book_output_folder` so the
        enable / tooltip state and the click handler share one
        resolver with the card context menu (the two previously drifted
        out of sync: Book Details consulted the origins registry but
        the context menu fell back to the book's containing folder,
        which for library-organized entries was ``Library/Translated``
        instead of the original output folder).
        """
        return _resolve_book_output_folder(self._book)

    def _resolve_source_file_target(self) -> str:
        """Return the raw source file path the 🔗 button should reveal.

        Thin wrapper around :func:`_resolve_book_source_file` so the
        Book Details source button and the card context menu's "Reveal
        source file" action share one resolution path.
        """
        return _resolve_book_source_file(self._book)

    def _resolve_translated_file_target(self) -> str:
        """Return the compiled translated EPUB path the 📕 button opens.

        Thin wrapper around :func:`_resolve_book_translated_file` so
        the Book Details translated button and the card context menu's
        "Reveal Translated File" action share one resolution path.
        """
        return _resolve_book_translated_file(self._book)


# ---------------------------------------------------------------------------
# U5 public API: the Library for callers without Qt (Glossarion Mobile, tools).
# Plain objects carry the attributes the desktop dialogs keep on ``self``, so the
# mixins above run unchanged; functions return what the desktop shows or emits.
# ---------------------------------------------------------------------------

#: Library / Book Details "per page" choices (``"all"`` = no paging).
PAGE_SIZE_OPTIONS = (20, 50, 100, 250, 500, "all")
#: Words that unlock the typed delete confirmation (case-insensitive).
DELETE_KEYWORDS = LibraryShelfMixin._DELETE_KEYWORDS
RAW_MATCH_EXACT = ScanForRawMixin.MATCH_EXACT
RAW_MATCH_FUZZY = ScanForRawMixin.MATCH_FUZZY
#: Extensions the Library imports as raw books / as translations.
RAW_IMPORT_EXTENSIONS = (".epub", ".txt", ".pdf", ".html", ".htm")
TRANSLATED_IMPORT_EXTENSIONS = (".epub",)

norm_book_key = _norm_book_key
book_matches_query = _book_matches_library_query
card_raw_title = _card_raw_title
card_progress_view = _card_progress_view
card_type_badge = _card_type_badge
card_size_text = _card_size_text
split_by_status = split_output_folders_by_status
attach_cross_location_duplicates = _attach_cross_location_duplicates
prepare_chapter_row_spec = _prepare_chapter_row_spec
merge_manual_metadata_edits = _merge_manual_metadata_edits
metadata_changed_values = _metadata_changed_values
metadata_subject_values = _metadata_subject_values
parse_epub_details = _parse_epub_details
mark_chapter_pending_for_retranslation = _mark_chapter_pending_for_retranslation
chapter_completed_in_progress = _chapter_completed_in_progress
cleanup_incomplete_chapter_output = _cleanup_incomplete_chapter_output
resolve_output_roots = _resolve_output_roots
find_raw_source_for_folder = _find_raw_source_for_folder
find_raw_source_for_library_epub = _find_raw_source_for_library_epub
resolve_book_output_folder = _resolve_book_output_folder
resolve_book_source_file = _resolve_book_source_file
resolve_book_metadata_source = _resolve_book_metadata_source
resolve_book_translated_file = _resolve_book_translated_file
list_compiled_outputs = _list_compiled_outputs
detect_workspace_kind = _detect_workspace_kind
workspace_compile_kind = _workspace_compile_kind
page_bounds = _page_bounds
page_label = _page_label
format_of_book = LibraryShelfMixin._format_of_book
card_signature = LibraryShelfMixin._card_signature
books_by_path = LibraryShelfMixin._books_by_path
summarize_folder_contents = LibraryShelfMixin._summarize_folder_contents
count_library_files = LibraryShelfMixin._count_library_files
MetadataEditError = _MetadataEditError


def unique_destination(directory: str, name: str, include_dirs: bool = False) -> str:
    """``directory/name`` or the next free ``name (N)`` (the Library's Keep Both rule).

    The rule is :func:`_unique_dest` (Organize): only existing files count as taken
    and the counter goes before the extension. *include_dirs* also treats existing
    folders as taken (a picked folder copied into the Inbox); for those the whole
    name gets the counter.
    """
    if not include_dirs:
        return _unique_dest(directory, name)
    candidate = os.path.join(directory, name)
    if not os.path.exists(candidate):
        return candidate
    if os.path.isdir(candidate):
        stem, ext = name, ""
    else:
        stem, ext = os.path.splitext(name)
    counter = 2
    while True:
        candidate = os.path.join(directory, f"{stem} ({counter}){ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def record_library_raw_inputs(files) -> None:
    """Record every existing raw input in the Library's raw-inputs registry.

    The run set-up hook ``_record_library_raw_inputs`` (desktop: TranslatorGUI, mobile:
    HeadlessOwner) so a translated book can always be found by the Library later.
    """
    try:
        for _p in (files or []):
            if _p and os.path.isfile(_p):
                record_library_raw_input(_p)
    except Exception:
        pass


def library_root_path() -> str:
    """The Library folder :func:`get_library_dir` uses, without creating it."""
    _library_override = os.environ.get("GLOSSARION_LIBRARY_DIR", "").strip()
    if _library_override:
        return str(Path(_library_override))
    return str(Path.home() / "Documents" / "Glossarion" / "Library")


# -- process environment ---------------------------------------------------------------

class LibraryEnv:
    """Where this process keeps its Library: ``library_root``, ``output_roots``,
    ``cache_dir`` and the ``config`` dict the scans read.

    Desktop never installs one (the Library lives in ``~/Documents/Glossarion/Library``,
    output roots come from ``OUTPUT_DIRECTORY`` + the app folder, caches in ``%TEMP%``).
    Glossarion Mobile installs one at start-up (:func:`install_library_env`).
    """

    __slots__ = ("library_root", "output_roots", "cache_dir", "config")

    def __init__(self, library_root=None, output_roots=(), cache_dir=None, config=None):
        self.library_root = os.path.abspath(str(library_root)) if library_root else None
        self.output_roots = tuple(
            os.path.abspath(str(root)) for root in (output_roots or ()) if root)
        self.cache_dir = os.path.abspath(str(cache_dir)) if cache_dir else None
        self.config = config if config is not None else {}

    @classmethod
    def from_runtime(cls, config=None) -> "LibraryEnv":
        """The environment the scans use right now (no override)."""
        return cls(get_library_dir(), tuple(_resolve_output_roots(config)), None, config)

    def __repr__(self) -> str:
        return (f"LibraryEnv(library_root={self.library_root!r}, output_roots={self.output_roots!r}, "
                f"cache_dir={self.cache_dir!r})")


def install_library_env(env: LibraryEnv) -> LibraryEnv:
    """Pin the Library to *env* for this process (Glossarion Mobile start-up).

    * ``library_root`` -> ``GLOSSARION_LIBRARY_DIR`` (the :func:`get_library_dir` seam);
    * ``output_roots[0]`` -> the default output root (``_default_output_root``), so
      the scans see exactly the app's Output folder instead of the process cwd;
    * ``cache_dir`` -> cover cache ``<cache>/covers`` and EPUB cache ``<cache>/epub``.
    """
    global _LIBRARY_ENV
    if env.library_root:
        os.environ["GLOSSARION_LIBRARY_DIR"] = env.library_root
        try:
            os.makedirs(env.library_root, exist_ok=True)
        except OSError:
            pass
    for root in env.output_roots:
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            pass
    if env.cache_dir:
        import library_covers
        import reader_doc
        library_covers.set_cover_cache_dir(os.path.join(env.cache_dir, "covers"))
        reader_doc.set_epub_cache_dir(os.path.join(env.cache_dir, "epub"))
    _LIBRARY_ENV = env
    return env


def current_library_env():
    """The installed :class:`LibraryEnv` (None on desktop)."""
    return _LIBRARY_ENV


def uninstall_library_env() -> None:
    """Undo :func:`install_library_env` (tests); ``GLOSSARION_LIBRARY_DIR`` is left as is."""
    global _LIBRARY_ENV
    _LIBRARY_ENV = None
    import library_covers
    import reader_doc
    library_covers.set_cover_cache_dir(None)
    reader_doc.set_epub_cache_dir(None)


# -- scanning ----------------------------------------------------------------------------

class _Signal:
    """Minimal ``Signal`` stand-in for plain jobs: ``emit`` records every call."""

    __slots__ = ("calls", "_callback")

    def __init__(self, callback=None):
        self.calls = []
        self._callback = callback

    def emit(self, *args):
        self.calls.append(args)
        if self._callback is not None:
            self._callback(*args)

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


class _DualScanJob(DualScanMixin):
    def __init__(self, config=None):
        self._config = config or {}
        self.scan_finished = _Signal()


def scan_library(config: dict | None = None):
    """Scan the Library shelves and every output root like the desktop Library.

    Returns ``(in_progress, completed)`` card rows (``_DualScannerThread``): both scans,
    the organized-folder ghost filter, origins / pairs discovered by title key (written
    to ``library_origins.txt`` exactly like desktop), state inheritance onto Library
    rows and the cross-location ``⚠ +N`` conflicts.
    """
    job = _DualScanJob(config)
    job.run()
    in_progress, completed = job.scan_finished.last
    return in_progress, completed


def merge_scans(output_rows: list, library_rows: list, config: dict | None = None):
    """Merge pre-computed ``scan_output_folders`` / ``scan_library_completed`` rows.

    Returns ``(in_progress, completed, origins_updates)``. The merge records the
    translated↔workspace links it discovers in ``library_origins.txt`` (as desktop
    does); ``origins_updates`` lists what changed: ``{"translated": {...},
    "pairs": {...}}`` (new or changed entries only).
    """
    before = _load_origins()
    job = _DualScanJob(config)
    job._merge_scan_rows(output_rows, library_rows)
    in_progress, completed = job.scan_finished.last
    after = _load_origins()
    updates = {}
    for bucket in ("translated", "pairs"):
        old = before.get(bucket, {}) or {}
        new = after.get(bucket, {}) or {}
        changed = {k: v for k, v in new.items() if old.get(k) != v}
        if changed:
            updates[bucket] = changed
    return in_progress, completed, updates


def book_summary(progress_file: str, config: dict | None = None,
                 exclude_special: bool | None = None) -> dict | None:
    """Card counts for one workspace: ``{total, completed, in_progress, failed, prog}``.

    The Library card numbers (``scan_output_folders``). Semantics are those of
    ``_read_progress_summary`` (see DISCREPANCIES.md "U5 Library card counts"):
    sidecar / metadata / artifact / gallery rows never count, a chunked parent whose
    chunks failed QA counts as failed, and a ``completed`` row whose response file is
    gone counts as in progress.
    """
    # Decided at U5 integration (DISCREPANCIES.md "U5 Library card counts"): Library cards on
    # desktop AND mobile keep these numbers. progress_core.compute_book_summary (the Progress
    # Manager's rules) also counts the metadata / TOC-artifact rows and output files it
    # discovers, so on the 143 real workspaces measured it disagreed with the card for 142
    # and would have taken "Ready to compile" away from 8; it stays the Chapters tab's number.
    if exclude_special is None:
        exclude_special = not _resolve_translate_special_files(config)
    return _read_progress_summary(progress_file, exclude_special=exclude_special, config=config)


class _SearchText:
    """``.text()`` shim standing in for the desktop search box."""

    __slots__ = ("value",)

    def __init__(self, value=""):
        self.value = str(value or "")

    def text(self) -> str:
        return self.value


class LibraryShelf(LibraryShelfMixin):
    """The Library home's state (``EpubLibraryDialog`` without widgets).

    Holds the two shelves plus the query / format / sort the desktop toolbar holds,
    and exposes the desktop decisions: filtering, card signatures and scan diffs,
    Organize / Undo / Delete / Clear raw link plans and executions, imports and the
    output-root check. Confirmations are the caller's: every ``plan_*`` returns what
    the desktop shows before acting.
    """

    def __init__(self, in_progress=(), completed=(), config=None, *, sort_mode=SORT_DATE,
                 format_filter=FORMAT_ALL, query=""):
        self._in_progress_books = list(in_progress or [])
        self._completed_books = list(completed or [])
        self._config = config if config is not None else {}
        self._sort_mode = sort_mode
        self._format_filter = format_filter
        self._search = _SearchText(query)
        self._selected_paths_ip = set()
        self._selected_paths_comp = set()

    # -- state ------------------------------------------------------------------------
    @property
    def in_progress(self) -> list:
        return self._in_progress_books

    @property
    def completed(self) -> list:
        return self._completed_books

    @property
    def query(self) -> str:
        return self._search.text()

    def set_query(self, query: str) -> None:
        self._search = _SearchText(query)

    def set_sort(self, sort_mode: str) -> None:
        self._sort_mode = sort_mode

    def set_format_filter(self, format_filter: str) -> None:
        self._format_filter = format_filter

    def rescan(self):
        """Full rescan (``scan_library``); returns ``(structure_changed, changed_by_tab)``."""
        in_progress, completed = scan_library(self._config)
        return self.apply_scan(in_progress, completed)

    def apply_scan(self, in_progress, completed):
        """Adopt a scan; returns ``(structure_changed, changed_by_tab)`` (2 s refresh rule)."""
        structure_changed, changed_by_tab, _new = self._scan_diff(in_progress, completed)
        self._in_progress_books = list(in_progress)
        self._completed_books = list(completed)
        return structure_changed, changed_by_tab

    def visible(self, tab: str = "ip") -> list:
        """The filtered + sorted shelf (``"ip"`` In progress, ``"comp"`` Completed)."""
        books = self._completed_books if tab == "comp" else self._in_progress_books
        return self._filtered(list(books))

    def counts(self) -> dict:
        """Organize / Undo counters plus the Scan-for-Raw badge count."""
        counts = dict(self._organize_counts())
        counts["missing_raw"] = self._missing_raw_count()
        return counts

    # -- actions ----------------------------------------------------------------------
    def import_paths(self, paths, target: str = "raw"):
        """Register *paths* in place (desktop Import): ``(imported, skipped, errors)``."""
        return self._run_import(list(paths or []), target)

    def plan_organize(self) -> dict:
        plan = self._plan_organize()
        plan["preview"] = self._organize_preview_lines(plan)
        raw_collisions, trans_collisions = self._organize_collisions(plan)
        plan["collisions"] = raw_collisions + trans_collisions
        plan["raw_collisions"] = raw_collisions
        plan["trans_collisions"] = trans_collisions
        return plan

    def execute_organize(self, plan: dict, collision_policy: str = "keep_both") -> dict:
        result = self._execute_organize(plan, collision_policy, plan.get("collisions", ()))
        result["summary"] = self._organize_summary(result)
        return result

    def plan_undo(self) -> dict:
        plan = self._plan_undo()
        plan["prompt"] = self._undo_prompt_text(plan)
        return plan

    def undo_collisions(self, plan: dict, restore_raw: bool, restore_trans: bool) -> list:
        return self._undo_collisions(plan, restore_raw, restore_trans)

    def execute_undo(self, plan: dict, restore_raw: bool, restore_trans: bool,
                     undo_policy: str = "keep_both", undo_collisions=None) -> dict:
        if undo_collisions is None:
            undo_collisions = self._undo_collisions(plan, restore_raw, restore_trans)
        result = self._execute_undo(plan, restore_raw, restore_trans, undo_policy, undo_collisions)
        result["summary"] = self._undo_summary(plan, restore_raw, restore_trans, result)
        return result

    def plan_delete(self, books) -> dict:
        targets, unregister_cards = self._plan_delete(list(books or []))
        return {
            "targets": targets,
            "unregister": unregister_cards,
            "needs_keyword": bool(targets) and not self._all_targets_not_started(targets),
            "detail": self._format_delete_detail(targets) if targets else "",
            "simple_prompt": self._delete_simple_prompt_text(targets) if targets else "",
        }

    def execute_delete(self, plan: dict, targets=None, on_progress=None) -> dict:
        """Delete the confirmed *targets* (default: every planned target) in parallel.

        Unsafe cards (outside the Library / output roots) are only unregistered.
        Returns ``{results, deleted, errors, summary, unregistered}``.
        """
        chosen = list(plan.get("targets") or []) if targets is None else list(targets)
        results = delete_targets([(label, pth, is_folder) for label, pth, is_folder, _b in chosen],
                                 on_progress=on_progress)
        deleted, errors, summary = self._delete_result_summary(results, len(chosen))
        unregistered = self._unregister_cards(plan.get("unregister") or [])
        return {"results": results, "deleted": deleted, "errors": errors, "summary": summary,
                "unregistered": unregistered}

    def plan_clear_raw_link(self, books) -> dict:
        targets = self._plan_clear_raw_link(list(books or []))
        return {"targets": targets,
                "prompt": self._clear_raw_link_prompt_text(targets) if targets else ""}

    def execute_clear_raw_link(self, plan: dict) -> int:
        return self._execute_clear_raw_link(list(plan.get("targets") or []))

    def output_root_mismatch(self, books):
        """None when *books* can load as-is, else the switch the desktop proposes."""
        info = self._output_override_mismatch(list(books or []))
        if info is not None:
            info["prompt"] = self._output_override_prompt_text(info)
        return info

    def apply_output_override(self, new_override: str) -> None:
        self._apply_output_override_config(new_override)


def sort_books(books, sort_mode: str = SORT_DATE) -> list:
    """Date (newest) / A-Z / Size (largest) order like the desktop toolbar."""
    return LibraryShelf(sort_mode=sort_mode)._sorted_books(list(books or []))


def filter_books(books, query: str = "", format_filter: str = FORMAT_ALL,
                 sort_mode: str = SORT_DATE) -> list:
    """Search (titles, raw titles, tags) + format filter + sort (``_filtered``)."""
    return LibraryShelf(sort_mode=sort_mode, format_filter=format_filter,
                        query=query)._filtered(list(books or []))


def diff_scans(old_in_progress, old_completed, new_in_progress, new_completed, *,
               query: str = "", format_filter: str = FORMAT_ALL, sort_mode: str = SORT_DATE):
    """The 2 s auto-refresh decision: ``(structure_changed, changed_by_tab)``."""
    shelf = LibraryShelf(old_in_progress, old_completed, sort_mode=sort_mode,
                         format_filter=format_filter, query=query)
    structure_changed, changed_by_tab, _new = shelf._scan_diff(
        list(new_in_progress or []), list(new_completed or []))
    return structure_changed, changed_by_tab


def import_paths(paths, target: str = "raw", config: dict | None = None, *,
                 copy_into_library: bool = False, record_origins: bool = False) -> dict:
    """Add files to the Library: ``{imported, skipped, errors, copied}``.

    Desktop registers files in place (``copy_into_library=False``): a raw file is
    recorded in ``library_raw_inputs.txt`` and gets a workspace (``<root>/<stem>/`` with
    ``source_epub.txt`` and an empty v2.1 ``translation_progress.json``); a translation
    is recorded in ``library_translated_inputs.txt``.

    Glossarion Mobile cannot reference picker paths later, so ``copy_into_library``
    first copies each file into ``Library/Raw`` (or ``Library/Translated``) with the
    Keep Both rule (identical content reuses the existing copy) and registers the copy.
    ``record_origins`` also records ``origins[raw|translated][copy] = original`` so Undo
    can move it back. Translations copied into ``Library/Translated`` need no
    registration: the shelf scan lists them.
    """
    shelf = LibraryShelf(config=config)
    paths = [p for p in (paths or []) if p]
    copied = []
    if copy_into_library:
        exts = TRANSLATED_IMPORT_EXTENSIONS if target == "translated" else RAW_IMPORT_EXTENSIONS
        dest_dir = get_library_translated_dir() if target == "translated" else get_library_raw_dir()
        prepared = []
        skipped_early = []
        for path in paths:
            src = os.path.abspath(path)
            if not os.path.isfile(src) or not src.lower().endswith(exts):
                prepared.append(src)  # validation reports it
                continue
            try:
                if os.path.normcase(os.path.dirname(src)) == os.path.normcase(os.path.abspath(dest_dir)):
                    prepared.append(src)
                    continue
                dest = os.path.join(dest_dir, os.path.basename(src))
                if os.path.isfile(dest) and _same_file_content(src, dest):
                    prepared.append(dest)
                    copied.append((src, dest, True))
                    continue
                dest = _unique_dest(dest_dir, os.path.basename(src))
                shutil.copy2(src, dest)
                prepared.append(dest)
                copied.append((src, dest, False))
            except OSError as exc:
                skipped_early.append(f"{os.path.basename(src)}: {exc}")
        if record_origins and copied:
            origins = _load_origins()
            bucket = "translated" if target == "translated" else "raw"
            mapping = dict(origins.get(bucket, {}) or {})
            for src, dest, _reused in copied:
                mapping[os.path.basename(dest)] = src
            origins[bucket] = mapping
            _save_origins(origins)
        if target == "translated":
            in_library = [d for _s, d, _r in copied]
            others = [p for p in prepared if p not in in_library]
            imported, skipped, errors = shelf._run_import(others, target) if others else ([], [], [])
            imported = in_library + imported
        else:
            imported, skipped, errors = shelf._run_import(prepared, target)
        errors = skipped_early + errors
    else:
        imported, skipped, errors = shelf._run_import(paths, target)
    return {"imported": imported, "skipped": skipped, "errors": errors,
            "copied": [{"source": s, "path": d, "reused": r} for s, d, r in copied]}


def _same_file_content(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        digest_a = hashlib.sha1()
        digest_b = hashlib.sha1()
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                chunk_a = fa.read(1024 * 1024)
                chunk_b = fb.read(1024 * 1024)
                if not chunk_a and not chunk_b:
                    break
                digest_a.update(chunk_a)
                digest_b.update(chunk_b)
        return digest_a.digest() == digest_b.digest()
    except OSError:
        return False


class _DeleteJob(LibraryDeleteMixin):
    def __init__(self, targets, on_progress=None):
        self._targets = list(targets or [])
        self.progress = _Signal(on_progress)
        self.delete_finished = _Signal()


def delete_targets(targets, on_progress=None) -> list:
    """Delete ``(label, path, is_folder)`` targets in parallel (``_LibraryDeleteThread``).

    *on_progress(done, total, label)* follows each item; returns
    ``[(label, path, is_folder, ok, error), ...]``.
    """
    job = _DeleteJob(targets, on_progress)
    job.run()
    return list(job.delete_finished.last[0]) if job.delete_finished.calls else []


def is_delete_keyword(text: str) -> bool:
    """True when *text* unlocks the typed delete confirmation."""
    return str(text or "").strip().lower() in DELETE_KEYWORDS


# -- Scan for Raw --------------------------------------------------------------------------

class _RawScanJob(RawScanMixin):
    def __init__(self, scan_folder, ext_suffixes, books, mode, threshold, prewalked=None):
        self._folder = scan_folder
        self._suffixes = ext_suffixes
        self._tracking = _LIBRARY_TRACKING_FILENAMES
        self._books = list(books or [])
        self._mode = mode
        self._threshold = int(threshold)
        self._prewalked = prewalked
        self._cancelled = False
        self.results = _Signal()


class RawScanSession(ScanForRawMixin):
    """Scan for Raw (``_ScanForRawDialog`` without widgets).

    Built from the shelf rows like the desktop dialog (only workspace cards missing
    their raw), it keeps the persisted ``epub_library_scan_raw_*`` settings, walks a
    folder and pairs workspaces exactly or fuzzily, then writes the accepted pairings.
    """

    def __init__(self, books, config=None):
        self._init_scan_state(list(books or []), config)

    @property
    def books(self) -> list:
        return self._books

    @property
    def matches(self) -> dict:
        return self._matches

    def configure(self, *, folder=None, mode=None, threshold=None, auto=None, exts=None) -> None:
        """Change the folder / mode / threshold / extension selection (and the config)."""
        if folder is not None:
            self._scan_folder = str(folder or "")
            self._config["epub_library_scan_raw_folder"] = self._scan_folder
        if mode is not None and mode in (self.MATCH_EXACT, self.MATCH_FUZZY):
            self._mode = mode
            self._config["epub_library_scan_raw_mode"] = mode
        if threshold is not None:
            self._threshold = max(40, min(95, int(threshold)))
            self._config["epub_library_scan_raw_threshold"] = self._threshold
        if exts is not None:
            chosen = {str(e).strip().lower().lstrip(".") for e in exts}
            chosen &= set(self._valid_exts)
            if chosen:
                self._manual_exts = chosen
                self._config["epub_library_scan_raw_exts"] = sorted(chosen)
        if auto is not None:
            self._auto_mode = bool(auto)
            self._config["epub_library_scan_raw_auto"] = self._auto_mode
        self._selected_exts = (self._derive_auto_exts() if self._auto_mode
                               else set(self._manual_exts)) or set(self._valid_exts)

    def cancel(self) -> None:
        """Stop a running :meth:`scan` at its next directory / workspace boundary."""
        job = getattr(self, "_job", None)
        if job is not None:
            job._cancelled = True

    def scan(self, rematch: bool = False) -> dict:
        """Walk the folder (or re-match the cached candidates) and pair the workspaces.

        Returns ``{candidates, matches, hits, status}`` with the desktop status line.
        """
        if not self._scan_folder or not os.path.isdir(self._scan_folder):
            self._matches = {}
            self._candidates = []
            return {"candidates": [], "matches": {}, "hits": 0,
                    "status": "⚠ Pick a folder that exists on disk."}
        prewalked = list(self._candidates) if (rematch and self._candidates) else None
        job = _RawScanJob(self._scan_folder, self._ext_suffixes(), self._books, self._mode,
                          self._threshold, prewalked=prewalked)
        self._job = job
        try:
            job.run()
        finally:
            self._job = None
        if not job.results.calls:
            return {"candidates": list(self._candidates), "matches": dict(self._matches),
                    "hits": 0, "status": ""}
        _folder, candidates, matches = job.results.last
        self._candidates = list(candidates)
        self._matches = dict(matches)
        hits = sum(1 for info in self._matches.values() if info.get("path"))
        return {"candidates": list(self._candidates), "matches": dict(self._matches),
                "hits": hits, "status": self._scan_status_text(hits)}

    def set_accepted(self, workspace_folder: str, accepted: bool) -> None:
        if workspace_folder in self._matches:
            self._matches[workspace_folder]["accepted"] = bool(accepted)

    def apply(self) -> int:
        """Write ``source_epub.txt`` + register the raw for every accepted match."""
        return self._write_raw_pairings()


# -- Book details ---------------------------------------------------------------------------

class _DetailsJob(BookDetailsLoaderMixin):
    def __init__(self, book, config=None, should_stop=None, on_preview=None):
        self._book = book
        self._config = config or {}
        self._cancelled = False
        self._stop_callback = should_stop if callable(should_stop) else None
        # The loader keeps filling the preview's ``details`` after emitting it; a Qt
        # signal hands the dialog a copy, so the plain job snapshots it as well.
        self.preview_ready = _Signal(
            (lambda payload: on_preview(copy.deepcopy(payload))) if on_preview else None)
        self.done = _Signal()
        self.error = _Signal()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        if self._stop_callback is not None:
            try:
                return bool(self._stop_callback())
            except Exception:
                return False
        return False


class BookDetailsError(RuntimeError):
    """``load_book_details`` failed (message = the desktop loader's error text)."""


def load_book_details(book: dict, config: dict | None = None, phase: str = "full",
                      should_stop=None, on_preview=None):
    """Book page data (``_BookDetailsLoader``): metadata, cover, metadata.json, chapters.

    ``phase="preview"`` returns after the fast pass ``{details, cover, metadata_json}``
    (OPF metadata without per-chapter titles); ``"full"`` returns
    ``{details, cover, chapters_info, metadata_json, progress}`` and calls
    *on_preview(payload)* with the preview first. None when *should_stop* fired.
    """
    if phase == "preview":
        captured = []

        def _stop():
            if captured:
                return True
            return bool(should_stop()) if callable(should_stop) else False

        def _preview(payload):
            captured.append(payload)
            if on_preview is not None:
                on_preview(copy.deepcopy(payload))

        job = _DetailsJob(book, config, _stop, _preview)
        job.run()
        if job.error.calls:
            raise BookDetailsError(str(job.error.last[0]))
        return captured[0] if captured else None
    job = _DetailsJob(book, config, should_stop, on_preview)
    job.run()
    if job.error.calls:
        raise BookDetailsError(str(job.error.last[0]))
    return job.done.last[0] if job.done.calls else None


class BookDetailsModel(BookDetailsMixin):
    """Book page state (``BookDetailsDialog`` without widgets) over a details payload.

    Exposes the chapter filters (special files, QA failures, search), the progress strip,
    the Chapters toggle label, tags / authors, the metadata editor values and save, the
    translated overlay for the reader and the reader-open decision.
    """

    def __init__(self, book: dict, payload: dict | None = None, config: dict | None = None, *,
                 show_special_files=None, search: str = "", qa_failures_only: bool = False):
        payload = payload or {}
        self._book = book
        self._config = config if config is not None else {}
        self._details = payload.get("details") or {
            "title": "", "authors": [], "publisher": "", "language": "", "date": "",
            "description": "", "subjects": [], "identifier": "", "chapters": [],
        }
        self._metadata_json = dict(payload.get("metadata_json") or book.get("metadata_json") or {})
        self._chapters_info = list(payload.get("chapters_info") or [])
        self._progress = payload.get("progress") or {}
        self._cover = payload.get("cover") or ""
        if show_special_files is None:
            show_special_files = _resolve_show_special_files(self._config)
        self._show_special_files = bool(show_special_files)
        self._show_qa_failures_only = bool(qa_failures_only)
        self._toc_search = _SearchText(search)

    def set_search(self, text: str) -> None:
        self._toc_search = _SearchText(text)

    def set_show_special_files(self, value: bool) -> None:
        self._show_special_files = bool(value)

    def set_qa_failures_only(self, value: bool) -> None:
        self._show_qa_failures_only = bool(value)

    @property
    def chapters_info(self) -> list:
        return self._chapters_info

    def visible_chapters(self) -> list:
        return self._filtered_chapter_infos()

    def counts(self):
        """``(done, total)`` over the visible base rows (gallery / special rules)."""
        return self._visible_counts()

    def progress_strip_text(self):
        return self._progress_strip_text()

    def toggle_label(self):
        return self._toc_toggle_state()

    def tags(self) -> list:
        return self._display_tag_values()

    def authors(self) -> list:
        return self._metadata_author_values()

    def title(self) -> str:
        return (self._metadata_json.get("title") or self._details.get("title")
                or self._book.get("name", ""))

    def synopsis(self) -> str:
        return (self._metadata_json.get("description")
                or self._details.get("description") or "").strip()

    def editor_values(self) -> dict:
        return self._metadata_editor_values()

    def save_metadata(self, edits: dict):
        """Save the changed editor *edits*; returns the new metadata (None = no change).

        Raises :class:`MetadataEditError` (dialog text) when metadata.json can't be read
        or written, or when the book has no output workspace.
        """
        output_folder = self._resolve_output_folder_target()
        if not output_folder or not os.path.isdir(output_folder):
            raise _MetadataEditError("Could not resolve this book's output workspace.")
        updated = self._save_metadata_edits(output_folder, edits)
        if updated is not None:
            self._metadata_json = updated
            self._book["metadata_json"] = dict(updated)
        return updated

    def translated_overlay(self):
        return self._build_translated_overlay()

    def plan_open_reader(self, initial_chapter=None, raw_only: bool = False) -> dict:
        return self._plan_open_reader(initial_chapter, raw_only)

    def row_specs(self, show_raw_title: bool = False) -> list:
        return [_prepare_chapter_row_spec(info, show_raw_title)
                for info in self._filtered_chapter_infos()]


def save_metadata_json_atomic(book: dict, edits: dict, payload: dict | None = None,
                              config: dict | None = None):
    """Merge *edits* into the book workspace's metadata.json (Book page editor)."""
    return BookDetailsModel(book, payload, config).save_metadata(edits)


def plan_open_reader(book: dict, payload: dict | None = None, initial_chapter=None,
                     raw_only: bool = False, config: dict | None = None,
                     show_special_files=None) -> dict:
    """How the Book page opens *book* (workspace / epub reader / system viewer)."""
    model = BookDetailsModel(book, payload, config, show_special_files=show_special_files)
    return model.plan_open_reader(initial_chapter, raw_only)


__all__ = [
    "BookDetailsError",
    "BookDetailsLoaderMixin",
    "BookDetailsMixin",
    "BookDetailsModel",
    "DELETE_KEYWORDS",
    "DualScanMixin",
    "FORMAT_ALL",
    "FORMAT_EPUB",
    "FORMAT_HTML",
    "FORMAT_IMAGE",
    "FORMAT_PDF",
    "FORMAT_TXT",
    "LibraryDeleteMixin",
    "LibraryEnv",
    "LibraryShelf",
    "LibraryShelfMixin",
    "MetadataEditError",
    "PAGE_SIZE_OPTIONS",
    "RAW_IMPORT_EXTENSIONS",
    "RAW_MATCH_EXACT",
    "RAW_MATCH_FUZZY",
    "RawScanMixin",
    "RawScanSession",
    "SIZE_2XL",
    "SIZE_2XS",
    "SIZE_3XL",
    "SIZE_4XL",
    "SIZE_5XL",
    "SIZE_6XL",
    "SIZE_COMPACT",
    "SIZE_LARGE",
    "SIZE_NORMAL",
    "SIZE_XL",
    "SIZE_XS",
    "SORT_DATE",
    "SORT_NAME",
    "SORT_SIZE",
    "ScanForRawMixin",
    "TRANSLATED_IMPORT_EXTENSIONS",
    "_ALL_SIZES",
    "_CARD_TYPE_BADGES",
    "_CHAPTER_BADGE_STYLES",
    "_CHAPTER_BADGE_TEXT",
    "_CHAPTER_PRIMARY_STYLES",
    "_DEFAULT_SPECIAL_FILE_EXACT",
    "_DEFAULT_SPECIAL_FILE_KEYWORDS",
    "_EDITABLE_BOOK_METADATA_FIELDS",
    "_EPUB_CACHE_SCHEMA",
    "_FILENAME_STRIP_CHARS",
    "_LIBRARY_TAG_SEARCH_KEYS",
    "_LIBRARY_TRACKING_FILENAMES",
    "_MetadataEditError",
    "_PROGRESS_SIDECAR_FILENAMES",
    "_RE_HTML_HEADING",
    "_RE_HTML_STRIP_TAGS",
    "_RE_HTML_TITLE",
    "_RE_HTML_WS",
    "_SIZE_PRESETS",
    "_attach_cross_location_duplicates",
    "_book_library_tag_values",
    "_book_library_title_values",
    "_book_matches_library_query",
    "_card_progress_view",
    "_card_raw_title",
    "_card_size_text",
    "_card_type_badge",
    "_chapter_completed_in_progress",
    "_cleanup_incomplete_chapter_output",
    "_count_epub_spine_items",
    "_count_translated_response_files",
    "_default_output_root",
    "_detect_workspace_kind",
    "_epub_cache_key",
    "_expected_output_root_for_book",
    "_extract_epub_search_metadata",
    "_extract_epub_subjects",
    "_extract_epub_titles",
    "_extract_html_title_fast",
    "_find_in_progress_novels",
    "_find_raw_source_for_folder",
    "_find_raw_source_for_library_epub",
    "_folder_has_compiled_output",
    "_folder_has_output_epub",
    "_has_number_in_filename",
    "_is_configured_special_file",
    "_is_gallery_filename",
    "_is_progress_sidecar_entry",
    "_is_special_spine_item",
    "_iter_library_search_values",
    "_library_io_worker_count",
    "_list_compiled_outputs",
    "_load_origins",
    "_mark_chapter_pending_for_retranslation",
    "_merge_manual_metadata_edits",
    "_metadata_changed_values",
    "_metadata_subject_values",
    "_migrate_legacy_library_layout",
    "_norm_book_key",
    "_origins_file",
    "_origins_raw_sources_for_stem",
    "_output_paths_equal",
    "_page_bounds",
    "_page_label",
    "_parse_epub_details",
    "_parse_special_file_list",
    "_prepare_chapter_row_spec",
    "_read_progress_summary",
    "_read_source_epub_pointer",
    "_read_translated_chapter_title",
    "_reader_worker_count",
    "_resolve_book_metadata_source",
    "_resolve_book_output_folder",
    "_resolve_book_source_file",
    "_resolve_book_translated_file",
    "_resolve_output_roots",
    "_resolve_show_special_files",
    "_resolve_special_file_lists",
    "_resolve_translate_all_numbered",
    "_resolve_translate_special_files",
    "_save_origins",
    "_special_file_settings_signature",
    "_special_file_stem",
    "_unique_dest",
    "_validate_source_epub_for_workspace",
    "_workspace_compile_kind",
    "attach_cross_location_duplicates",
    "book_matches_query",
    "book_summary",
    "books_by_path",
    "card_progress_view",
    "card_raw_title",
    "card_signature",
    "card_size_text",
    "card_type_badge",
    "chapter_completed_in_progress",
    "cleanup_incomplete_chapter_output",
    "count_library_files",
    "current_library_env",
    "delete_targets",
    "detect_workspace_kind",
    "diff_scans",
    "filter_books",
    "find_raw_source_for_folder",
    "find_raw_source_for_library_epub",
    "format_of_book",
    "get_library_dir",
    "get_library_raw_dir",
    "get_library_raw_inputs_file",
    "get_library_translated_dir",
    "get_library_translated_inputs_file",
    "import_paths",
    "install_library_env",
    "is_delete_keyword",
    "library_root_path",
    "list_compiled_outputs",
    "load_book_details",
    "load_library_raw_inputs",
    "load_library_translated_inputs",
    "mark_chapter_pending_for_retranslation",
    "merge_manual_metadata_edits",
    "merge_scans",
    "metadata_changed_values",
    "metadata_subject_values",
    "norm_book_key",
    "page_bounds",
    "page_label",
    "parse_epub_details",
    "plan_open_reader",
    "prepare_chapter_row_spec",
    "record_library_raw_input",
    "record_library_raw_inputs",
    "record_library_translated_input",
    "remove_library_raw_input",
    "remove_library_translated_input",
    "resolve_book_metadata_source",
    "resolve_book_output_folder",
    "resolve_book_source_file",
    "resolve_book_translated_file",
    "resolve_output_roots",
    "save_metadata_json_atomic",
    "scan_for_epubs",
    "scan_library",
    "scan_library_completed",
    "scan_output_folders",
    "sort_books",
    "split_by_status",
    "split_output_folders_by_status",
    "summarize_folder_contents",
    "uninstall_library_env",
    "unique_destination",
    "workspace_compile_kind",
]
