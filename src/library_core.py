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

The rest of the Library logic (scans, covers, book details, actions) follows in U5
(plan section 2). Log records keep the ``epub_library`` logger name so desktop log
routing is unchanged. Tests that redirect the Library patch these names here (the
moved functions resolve each other through this module).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import logging
import os
import shutil
import traceback
from pathlib import Path

from metadata_progress import is_metadata_progress_entry
from translation_artifacts import is_translation_artifact_progress_entry

# Same logger as before the move (records keep the "epub_library" name).
logger = logging.getLogger("epub_library")


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


__all__ = [
    "_FILENAME_STRIP_CHARS",
    "_LIBRARY_TRACKING_FILENAMES",
    "_find_raw_source_for_folder",
    "_load_origins",
    "_migrate_legacy_library_layout",
    "_norm_book_key",
    "_origins_file",
    "_origins_raw_sources_for_stem",
    "_read_source_epub_pointer",
    "_save_origins",
    "_special_file_stem",
    "_validate_source_epub_for_workspace",
    "get_library_dir",
    "get_library_raw_dir",
    "get_library_raw_inputs_file",
    "get_library_translated_dir",
    "get_library_translated_inputs_file",
    "load_library_raw_inputs",
    "load_library_translated_inputs",
    "record_library_raw_input",
    "record_library_translated_input",
    "remove_library_raw_input",
    "remove_library_translated_input",
]
