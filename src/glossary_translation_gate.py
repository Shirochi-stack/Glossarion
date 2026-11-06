"""Completion check for the optional translation glossary gate."""

import hashlib
import json
import os
import re

#: Image inputs; the image-folder glossary keeps one progress file per folder.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
#: Finish reasons of a reply that was cut off: never an answer, even when it reads as empty.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "truncated"})
#: JSON wrappers parse_api_response unwraps (``{"entries": [...]}``).
_JSON_WRAPPER_KEYS = frozenset({"entries", "glossary", "characters", "terms", "data"})
_THINKING_BLOCK = re.compile(
    r"<(?:thinking|think|thought)>.*?</(?:thinking|think|thought)>", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)\s*```", re.DOTALL)
_COLUMN_NAME = re.compile(r"[a-z_][a-z0-9_ ]*")
_TABLE_RULE = re.compile(r"[|\s:\-]*-[|\s:\-]*")
_FENCE_LANGUAGES = frozenset({"csv", "tsv", "text", "txt", "plaintext", "json"})


def is_image_source(source_path):
    return os.path.splitext(str(source_path or ""))[1].lower() in IMAGE_EXTENSIONS


def _is_glossary_header_line(line):
    """``type,raw_name,translated_name[,...]`` (or the name-only header) in any column separator."""
    line = line.replace("\\x1F", "\x1f").replace("\\x1f", "\x1f")
    for separator in ("\x1f", "\t", "|", ","):
        if separator in line:
            cells = line.split(separator)
            break
    else:
        cells = line.split()
    names = [cell.strip().strip("\"'`*").strip().lower() for cell in cells]
    names = [name for name in names if name]
    if not names or not all(_COLUMN_NAME.fullmatch(name) for name in names):
        return False
    return "raw_name" in names and ("type" in names or "translated_name" in names)


def is_empty_glossary_answer(response_text, finish_reason=None):
    """True when a glossary reply means "0 entries": ``[]`` or only the CSV header row.

    An image or chapter without names legitimately has nothing to extract, and the model
    then answers with the empty list or just the header the prompt asked for. A reply that
    was cut off (``finish_reason`` "length") is not an answer, and any other text (a
    refusal, garbage, a partial row) is not empty: those stay failures.
    """
    if str(finish_reason or "").strip().lower() in _TRUNCATED_FINISH_REASONS:
        return False
    text = _THINKING_BLOCK.sub("", str(response_text or "")).strip()
    fenced = _CODE_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip() for line in text.splitlines()]
    if lines and lines[0].lower() in _FENCE_LANGUAGES:  # "```csv" with the fence already cut off
        lines = lines[1:]
    text = "\n".join(lines).strip()
    if not text:
        return False
    try:
        data = json.loads(text)
    except ValueError:
        pass
    else:
        if isinstance(data, dict) and data and set(data) <= _JSON_WRAPPER_KEYS:
            return all(value == [] for value in data.values())
        return data == []
    lines = [line for line in lines if line and not line.startswith("#") and not _TABLE_RULE.fullmatch(line)]
    return bool(lines) and all(_is_glossary_header_line(line) for line in lines)


def image_glossary_progress_path(image_path, root):
    """The ``<folder>_glossary_progress.json`` the image-folder glossary writes for this image.

    Image extraction keeps one progress file per image folder in ``<output>/Glossary`` (an
    output override) or ``Glossary`` beside the images.
    """
    folder = os.path.dirname(os.path.abspath(image_path))
    name = f"{os.path.basename(folder) or 'images'}_glossary_progress.json"
    candidates = (
        os.path.join(root, "Glossary", name),
        os.path.join(folder, "Glossary", name),
    )
    return next((path for path in candidates if os.path.isfile(path)), None)


def _image_content_hash(image_path):
    hasher = hashlib.sha256()
    with open(image_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_glossary_complete(progress_path, image_paths):
    """Return (ready, reason): every image's row in the image-folder progress is done.

    Done is ``completed`` (with any number of entries, 0 included: an image without text
    is a finished extraction) or a skipped cover. Rows are keyed by the image's content
    hash, like the extractor's own resume check.
    """
    if not progress_path or not os.path.isfile(progress_path):
        return False, "glossary progress is unavailable"
    try:
        with open(progress_path, "r", encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError) as exc:
        return False, f"glossary progress cannot be read: {exc}"
    images = progress.get("images") if isinstance(progress, dict) else None
    if not isinstance(images, dict):
        images = {}
    pending = []
    for image_path in image_paths:
        name = os.path.basename(image_path)
        marker = progress.get(f"skip_{name}") if isinstance(progress, dict) else None
        if isinstance(marker, dict) and marker.get("status") == "skipped":
            continue
        try:
            row = images.get(_image_content_hash(image_path))
        except OSError:
            row = None
        status = str((row or {}).get("status") or "not extracted").lower()
        if status not in ("completed", "skipped_cover"):
            pending.append(f"{name}: {status}")
    if pending:
        return False, f"image extraction is incomplete ({'; '.join(pending)})"
    return True, "complete"


def progress_path_for_source(source_path, root):
    """Find the progress file written by Balanced/Full extraction."""
    from glossary_paths import get_book_glossary_dir

    if is_image_source(source_path):
        return image_glossary_progress_path(source_path, root)
    base = os.path.splitext(os.path.basename(source_path))[0]
    shared = os.path.join(root, "Glossary")
    candidates = (
        os.path.join(get_book_glossary_dir(shared, base, create=False), f"{base}_glossary_progress.json"),
        os.path.join(shared, f"{base}_glossary_progress.json"),
        os.path.join(root, base, "Glossary", f"{base}_glossary_progress.json"),
        os.path.join(root, base, "glossary_progress.json"),
    )
    return next((path for path in candidates if os.path.isfile(path)), None)


def retryable_glossary_qa_failures(progress_path, *, skip_api_errors=False):
    """Return failed chapter indices; optionally omit API_ERROR-only failures."""
    if not progress_path or not os.path.isfile(progress_path):
        return set()
    try:
        with open(progress_path, "r", encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return set()
    if not isinstance(progress, dict):
        return set()

    def index(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    failed = {index(value) for value in (progress.get("failed") or [])}
    failed.discard(None)
    chapters = progress.get("chapters", {})
    qa_failed = set()
    chapter_issues = {}
    if isinstance(chapters, dict):
        for key, info in chapters.items():
            if isinstance(info, dict) and str(info.get("status", "")).lower() == "qa_failed":
                chapter_index = index(info.get("chapter_index", key))
                if chapter_index is not None:
                    failed.add(chapter_index)
                    qa_failed.add(chapter_index)
                    chapter_issues[chapter_index] = info.get("qa_issues_found") or []

    issues_by_index = progress.get("qa_issues_found", {})
    if not isinstance(issues_by_index, dict):
        issues_by_index = {}
    retryable = set()
    for chapter_index in failed:
        issues = issues_by_index.get(
            str(chapter_index), issues_by_index.get(chapter_index, chapter_issues.get(chapter_index, []))
        )
        if isinstance(issues, str):
            issues = [issues]
        normalized = {str(issue).strip().upper() for issue in (issues or [])}
        if (normalized or chapter_index in qa_failed) and not (
            skip_api_errors and normalized == {"API_ERROR"}
        ):
            retryable.add(chapter_index)
    return retryable


def refinement_complete(progress, glossary_path, config):
    """Check the active, selected refinement types that have glossary entries."""
    if not config.get("glossary_refinement_enabled", False):
        return True, "complete"
    if not glossary_path or not os.path.isfile(glossary_path):
        return False, "glossary file is unavailable for refinement"

    from glossary_usage import parse_glossary_file
    from glossary_refinement import _refinement_type_key as key

    try:
        entries = parse_glossary_file(glossary_path)
    except (OSError, ValueError) as exc:
        return False, f"glossary file cannot be read: {exc}"
    types_with_entries = {key(entry.get("type")) for entry in entries if isinstance(entry, dict)}
    types_with_entries.discard("")
    custom_types = config.get("custom_entry_types")
    if not isinstance(custom_types, dict) or not custom_types:
        from extract_glossary_from_epub import get_custom_entry_types
        custom_types = get_custom_entry_types()
    active_types = {
        key(name) for name, settings in custom_types.items()
        if not isinstance(settings, dict) or settings.get("enabled", True)
    }
    types_with_entries &= active_types
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


def glossary_complete(progress_path, glossary_path, config, *, require_minimal_pass=False, is_epub=False,
                      source_path=None):
    """Return (ready, reason) using the saved chapter, Minimal and refinement rows.

    An image source is checked against its folder's image progress: the image glossary
    has no chapters, Minimal pass or refinement rows.
    """
    if source_path and is_image_source(source_path):
        return image_glossary_complete(progress_path, [source_path])
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
