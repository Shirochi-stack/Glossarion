"""Completion check for the optional translation glossary gate."""

import json
import os


def progress_path_for_source(source_path, root):
    """Find the progress file written by Balanced/Full extraction."""
    from glossary_paths import get_book_glossary_dir

    base = os.path.splitext(os.path.basename(source_path))[0]
    shared = os.path.join(root, "Glossary")
    candidates = (
        os.path.join(get_book_glossary_dir(shared, base, create=False), f"{base}_glossary_progress.json"),
        os.path.join(shared, f"{base}_glossary_progress.json"),
        os.path.join(root, base, "Glossary", f"{base}_glossary_progress.json"),
        os.path.join(root, base, "glossary_progress.json"),
    )
    return next((path for path in candidates if os.path.isfile(path)), None)


def refinement_complete(progress, glossary_path, config):
    """Check the selected refinement types that have glossary entries."""
    if not config.get("glossary_refinement_enabled", False):
        return True, "complete"
    if not glossary_path or not os.path.isfile(glossary_path):
        return False, "glossary file is unavailable for refinement"

    from glossary_usage import parse_glossary_file

    def key(value):
        value = str(value or "").strip().casefold()
        return "terms" if value == "term" else value

    try:
        entries = parse_glossary_file(glossary_path)
    except (OSError, ValueError) as exc:
        return False, f"glossary file cannot be read: {exc}"
    types_with_entries = {key(entry.get("type")) for entry in entries if isinstance(entry, dict)}
    types_with_entries.discard("")
    if config.get("glossary_refinement_type_mode") == "selected":
        selected = {key(value) for value in config.get("glossary_refinement_selected_types", [])}
        types_with_entries &= selected
    if not types_with_entries:
        return True, "complete"

    refinement = progress.get("refinement", {})
    if not isinstance(refinement, dict):
        refinement = {}
    completed = {
        key(info.get("entry_type") or str(name).split("::", 1)[-1])
        for name, info in refinement.items()
        if str(name).startswith("type::")
        and isinstance(info, dict)
        and str(info.get("status", "")).lower() == "completed"
    }
    missing = types_with_entries - completed
    if missing:
        return False, f"refinement is incomplete for {', '.join(sorted(missing))}"
    return True, "complete"


def glossary_complete(progress_path, glossary_path, config, *, require_minimal_pass=False, is_epub=False):
    """Return (ready, reason) using the saved chapter, Minimal and refinement rows."""
    if not progress_path or not os.path.isfile(progress_path):
        return False, "glossary progress is unavailable"
    try:
        with open(progress_path, "r", encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError) as exc:
        return False, f"glossary progress cannot be read: {exc}"

    from extract_glossary_from_epub import glossary_progress_completion

    complete, represented, total, reason = glossary_progress_completion(progress, is_epub=is_epub)
    if not complete:
        return False, f"extraction is incomplete ({represented}/{total}: {reason})"
    if require_minimal_pass:
        minimal = progress.get("minimal_pass", {})
        status = str(minimal.get("status", "")).lower() if isinstance(minimal, dict) else ""
        if status != "completed" and not (status == "skipped" and minimal.get("reason") == "no_entries"):
            return False, f"Minimal pass is {status or 'not completed'}"
    return refinement_complete(progress, glossary_path, config)
