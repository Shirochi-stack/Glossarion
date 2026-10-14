"""Read current EPUB translation overlays without depending on a GUI snapshot."""

import json
import os

from chapter_chunk_progress import (
    effective_parent_status,
    ensure_chunk_entry_schema,
    is_multi_chunk_entry,
)


_HTML_EXTENSIONS = (".html", ".xhtml", ".htm")


def _basename(value):
    return os.path.basename(str(value or "").replace("\\", "/"))


def _chapter_name(value):
    name = _basename(value).lower()
    if name.startswith("response_"):
        name = name[len("response_"):]
    # Response names can retain the original extension (chapter.xhtml.html).
    while name.endswith(_HTML_EXTENSIONS):
        name = os.path.splitext(name)[0]
    return name


def make_epub_overlay_provider(output_dir, source_filenames, *, initial_overlay=None):
    """Return a reader poll callback using source filenames and fresh disk state.

    Missing or invalid progress snapshots return None after a progress file has
    been seen, so an interrupted/atomic rewrite cannot clear a loaded overlay.
    Workspaces without a progress file can still discover response files. Titles
    are deliberately omitted: the merge worker derives them from the bytes it
    actually loads, including rewritten responses.
    """
    output_dir = os.path.abspath(os.fspath(output_dir))
    progress_path = os.path.join(output_dir, "translation_progress.json")
    progress_seen = os.path.isfile(progress_path)
    filenames = list(dict.fromkeys(
        _basename(name) for name in source_filenames
        if str(name or "").lower().endswith(_HTML_EXTENSIONS)
    ))
    by_basename = {name.lower(): name for name in filenames}
    by_stem = {}
    for name in filenames:
        by_stem.setdefault(_chapter_name(name), name)
    initial_paths = {}
    initial_statuses = {}
    for name, entry in (initial_overlay or {}).items():
        path = entry if isinstance(entry, str) else (entry or {}).get("path")
        if path:
            initial_paths[_basename(name).lower()] = str(path)
            if isinstance(entry, dict) and entry.get("status"):
                initial_statuses[_basename(name).lower()] = entry["status"]

    def _match_source(candidates):
        for candidate in candidates:
            match = by_basename.get(_basename(candidate).lower())
            if match:
                return match
        for candidate in candidates:
            match = by_stem.get(_chapter_name(candidate))
            if match:
                return match
        return None

    def _provider():
        nonlocal progress_seen
        if not os.path.isdir(output_dir):
            return None
        try:
            with open(progress_path, "r", encoding="utf-8") as stream:
                progress = json.load(stream)
            if not isinstance(progress, dict) or not isinstance(
                    progress.get("chapters"), dict):
                return None
            progress_seen = True
        except FileNotFoundError:
            if progress_seen:
                return None
            progress = {"chapters": {}}
        except (OSError, ValueError):
            return None

        matched = {}
        for key, entry in progress["chapters"].items():
            if not isinstance(entry, dict):
                continue
            candidates = [
                entry.get("original_basename"),
                entry.get("original_filename"),
                entry.get("chapter_file"),
                entry.get("source_filename"),
                entry.get("filename"),
                key,
                entry.get("output_file"),
            ]
            filename = _match_source(candidates)
            if filename:
                matched.setdefault(filename.lower(), (key, entry))

        overlay = {}
        chunks = progress.get("chapter_chunks") or {}
        for filename in filenames:
            source_key = filename.lower()
            progress_key, entry = matched.get(source_key, ("", {}))
            candidates = []
            if entry.get("output_file"):
                candidates.append(str(entry["output_file"]))
            stem = os.path.splitext(filename)[0]
            candidates.extend((
                f"response_{stem}.html", f"response_{stem}.xhtml",
                f"{stem}.html", f"{stem}.xhtml",
            ))
            if source_key in initial_paths:
                candidates.append(initial_paths[source_key])
            for candidate in candidates:
                if not candidate.lower().endswith(_HTML_EXTENSIONS):
                    continue
                path = os.path.normpath(os.path.join(
                    output_dir, candidate.replace("\\", "/")))
                if not os.path.isfile(path):
                    continue
                status = str(entry.get("status") or initial_statuses.get(source_key)
                             or "completed").strip().lower()
                chunk_key = str(entry.get("content_hash") or progress_key)
                chunk_entry = chunks.get(chunk_key) if isinstance(chunks, dict) else None
                if isinstance(chunk_entry, dict) and is_multi_chunk_entry(chunk_entry):
                    ensure_chunk_entry_schema(chunk_entry)
                    status = effective_parent_status(status, chunk_entry)
                overlay[source_key] = {"path": path, "status": status}
                break
        extra_dirs = [
            path for path in (
                os.path.join(output_dir, "images"),
                os.path.join(output_dir, "translated_images"),
            ) if os.path.isdir(path)
        ]
        return overlay, extra_dirs

    return _provider
