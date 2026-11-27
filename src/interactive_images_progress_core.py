"""Interactive Images progress: the per-book progress file, its rows, stats and actions (GUI-free).

``<output>/Interactive_Images/interactive_images_progress.json`` (inside the subfolder, so the
translation Progress Manager never lists it as a chapter). Mirrors ``glossary_progress_core``'s public
shape (find / open / reload / signature / rows / stats / actions) and reuses its locked, atomic
writer: every change goes through :func:`mutate` (cross-process lock + re-read + atomic replace).

Each chapter entry records the hash of the translated chapter's visible text it was built from; a
chapter whose text changed since (retranslated) shows as ``stale``.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import interactive_images_core as core

PROGRESS_VERSION = 1

#: Stored statuses; ``stale`` and ``not_built`` are derived.
STORED_STATUSES = ('pending', 'in_progress', 'completed', 'css_only', 'partial', 'failed', 'rejected', 'skipped')

STATUS_ICONS = {
    'completed': '✅',
    'css_only': '🎨',
    'partial': '⚠️',
    'failed': '❌',
    'rejected': '🚫',
    'skipped': '⏭️',
    'in_progress': '🔄',
    'pending': '❓',
    'stale': '🕒',
    'not_built': '⬜',
}
STATUS_LABELS = {
    'completed': 'Completed',
    'css_only': 'Styles only',
    'partial': 'Partial',
    'failed': 'Failed',
    'rejected': 'Rejected (text changed)',
    'skipped': 'Skipped',
    'in_progress': 'In Progress',
    'pending': 'Pending',
    'stale': 'Stale (chapter changed)',
    'not_built': 'Not built',
}
#: Row colours (the Progress Manager vocabulary where the status exists there).
STATUS_COLORS = {
    'completed': 'green',
    'css_only': '#17a2b8',
    'partial': 'orange',
    'failed': 'red',
    'rejected': 'red',
    'skipped': '#9aa0a6',
    'in_progress': 'orange',
    'pending': 'white',
    'stale': '#d69e2e',
    'not_built': '#2b6cb0',
}
#: Legend groups (the clickable stats chips).
STATUS_GROUPS = {
    'built': ('completed', 'css_only'),
    'partial': ('partial',),
    'stale': ('stale',),
    'failed': ('failed',),
    'rejected': ('rejected',),
    'not_built': ('not_built', 'pending', 'skipped', 'in_progress'),
}
#: Statuses "Build pending" leaves alone (when not stale).
DONE_STATUSES = ('completed', 'css_only')

_HTML_EXTENSIONS = ('.html', '.htm', '.xhtml')
_RESPONSE_NUM_RE = re.compile(r'^response_(\d+)', re.I)


# ---------------------------------------------------------------------------------------------
# File I/O (glossary_progress_core's lock + atomic writer)
# ---------------------------------------------------------------------------------------------

def write_json_atomic(path, data):
    from glossary_progress_core import _write_glossary_progress_atomic
    _write_glossary_progress_atomic(path, data)


def empty_progress():
    return {'version': PROGRESS_VERSION, 'profile': '', 'stylesheet': {}, 'chapters': {}}


def read_progress_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return empty_progress()
    if not isinstance(data, dict):
        return empty_progress()
    data.setdefault('version', PROGRESS_VERSION)
    if not isinstance(data.get('chapters'), dict):
        data['chapters'] = {}
    return data


def mutate(path, fn):
    """Apply ``fn(data)`` to the newest progress file under the cross-process lock and write it atomically.
    Returns ``fn``'s result."""
    from glossary_progress_core import _locked_glossary_progress_write

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _locked_glossary_progress_write(path):
        data = read_progress_file(path)
        result = fn(data)
        write_json_atomic(path, data)
        return result


def find_progress(output_dir):
    path = core.progress_path(output_dir)
    return path if os.path.isfile(path) else None


def progress_signature(path):
    """(mtime_ns, ctime_ns, size) of the progress file, None when missing."""
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    except OSError:
        return None


def book_signature(output_dir):
    """Changes when the images progress file or the translation progress file changes (row staleness)."""
    return (progress_signature(core.progress_path(output_dir)),
            progress_signature(os.path.join(output_dir, 'translation_progress.json')))


# ---------------------------------------------------------------------------------------------
# Translated chapters of a book
# ---------------------------------------------------------------------------------------------

@dataclass
class TranslatedChapter:
    key: str                 # output file name (the progress key)
    number: int              # chapter number used in image names
    output_file: str
    path: str
    status: str              # translation status
    title: str = ''


def _chapter_number(entry, output_file, fallback):
    match = _RESPONSE_NUM_RE.match(os.path.basename(output_file))
    if match:
        return int(match.group(1))
    value = entry.get('actual_num')
    try:
        number = float(value)
        if number.is_integer():
            return int(number)
    except (TypeError, ValueError):
        pass
    return fallback


def translated_chapters(output_dir, *, completed_only=False):
    """The book's chapter outputs (HTML) from ``translation_progress.json``, in chapter order. ``status`` is
    the translation status; merged chapters (their parent's file) are left out."""
    path = os.path.join(output_dir, 'translation_progress.json')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            prog = json.load(handle)
    except (OSError, ValueError):
        prog = {}
    entries = (prog.get('chapters') or {}) if isinstance(prog, dict) else {}
    seen, out = set(), []
    for index, entry in enumerate(entries.values() if isinstance(entries, dict) else ()):
        if not isinstance(entry, dict):
            continue
        output_file = os.path.basename(str(entry.get('output_file') or ''))
        status = str(entry.get('status') or '')
        if not output_file or status == 'merged' or output_file in seen:
            continue
        if not output_file.lower().endswith(_HTML_EXTENSIONS):
            continue
        full = os.path.join(output_dir, output_file)
        if not os.path.isfile(full):
            continue
        if completed_only and not status.startswith('completed'):
            continue
        seen.add(output_file)
        out.append(TranslatedChapter(output_file, _chapter_number(entry, output_file, index + 1), output_file,
                                     full, status, str(entry.get('title') or '')))
    out.sort(key=lambda chapter: (chapter.number, chapter.key))
    return out


def translation_completed(chapter):
    return str(chapter.status or '').startswith('completed')


# ---------------------------------------------------------------------------------------------
# Model, rows, stats
# ---------------------------------------------------------------------------------------------

@dataclass
class ImagesModel:
    output_dir: str
    path: str
    data: dict
    chapters: List[TranslatedChapter] = field(default_factory=list)
    _hash_cache: dict = field(default_factory=dict)

    def current_hash(self, chapter):
        """Visible-text hash of a chapter file now (cached per file signature)."""
        sig = progress_signature(chapter.path)
        cached = self._hash_cache.get(chapter.path)
        if cached and cached[0] == sig:
            return cached[1]
        try:
            with open(chapter.path, 'r', encoding='utf-8') as handle:
                value = core.visible_text_hash(handle.read())
        except OSError:
            value = ''
        self._hash_cache[chapter.path] = (sig, value)
        return value


def open_progress(output_dir):
    """The book's images progress (an empty one when no file exists yet)."""
    path = core.progress_path(output_dir)
    return ImagesModel(output_dir, path, read_progress_file(path), translated_chapters(output_dir))


def reload_progress(model):
    model.data = read_progress_file(model.path)
    model.chapters = translated_chapters(model.output_dir)
    return model.data


@dataclass
class ImagesRow:
    kind: str                  # 'chapter'
    key: str                   # output file name
    display: str
    status: str                # stored or derived status
    color: str
    icon: str
    chapter_index: Optional[int] = None
    filename: str = ''
    images: int = 0
    error: str = ''
    stale: bool = False
    stored_status: str = ''
    entry: Any = None


def entry_status(model, chapter, entry=None):
    """``(status, stale)`` of one chapter: the stored status, ``stale`` when the chapter's text changed since
    it was built, ``not_built`` without an entry."""
    entry = entry if entry is not None else (model.data.get('chapters') or {}).get(chapter.key)
    if not isinstance(entry, dict):
        return 'not_built', False
    status = str(entry.get('status') or 'pending')
    if status in ('completed', 'css_only', 'partial') and entry.get('source_hash'):
        if model.current_hash(chapter) != entry.get('source_hash'):
            return 'stale', True
    return status, False


def row_display(chapter, status, images):
    count = f"{images} image{'s' if images != 1 else ''}" if images else "no images"
    label = STATUS_LABELS.get(status, status)
    return f"{STATUS_ICONS.get(status, '⬜')} Ch.{chapter.number:>4} │ {chapter.key} │ {count} │ {label}"


def image_rows(model, chapters=None):
    """One row per translated chapter, in chapter order."""
    rows = []
    entries = model.data.get('chapters') or {}
    for index, chapter in enumerate(chapters if chapters is not None else model.chapters):
        entry = entries.get(chapter.key)
        status, stale = entry_status(model, chapter, entry)
        images = 0
        error = ''
        if isinstance(entry, dict):
            images = sum(1 for image in entry.get('images') or () if image.get('status') == 'completed')
            error = str(entry.get('error') or '')
        rows.append(ImagesRow('chapter', chapter.key, row_display(chapter, status, images), status,
                              STATUS_COLORS.get(status, 'white'), STATUS_ICONS.get(status, '⬜'), index,
                              chapter.key, images, error, stale,
                              str(entry.get('status') or '') if isinstance(entry, dict) else '', entry))
    return rows


def image_stats(model, rows=None):
    """``{group: count}`` for :data:`STATUS_GROUPS` plus ``total`` and ``images``."""
    rows = rows if rows is not None else image_rows(model)
    stats = {group: 0 for group in STATUS_GROUPS}
    for row in rows:
        for group, statuses in STATUS_GROUPS.items():
            if row.status in statuses:
                stats[group] += 1
    stats['total'] = len(rows)
    stats['images'] = sum(row.images for row in rows)
    return stats


def needs_build(row):
    return row.status not in DONE_STATUSES


def pending_keys(model):
    """Chapters "Build pending" works on: not built, stale, failed, partial, rejected, pending."""
    return [row.key for row in image_rows(model) if needs_build(row)]


def tab_count_text(stats):
    """The Progress Manager tab label count: ``built/total`` (and ``k stale``)."""
    built = stats.get('built', 0) + stats.get('partial', 0)
    text = f"{built}/{stats.get('total', 0)}"
    if stats.get('stale'):
        text += f", {stats['stale']} stale"
    return text


# ---------------------------------------------------------------------------------------------
# Entry writes (the runner)
# ---------------------------------------------------------------------------------------------

def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def set_entry(path, key, entry):
    def apply(data):
        data.setdefault('chapters', {})[key] = entry
    mutate(path, apply)


def set_status(path, key, status, error=''):
    def apply(data):
        entry = data.setdefault('chapters', {}).setdefault(key, {})
        entry['status'] = status
        entry['error'] = error
    mutate(path, apply)


def set_book_info(path, profile, stylesheet):
    def apply(data):
        data['profile'] = profile
        data['stylesheet'] = stylesheet
    mutate(path, apply)


# ---------------------------------------------------------------------------------------------
# Actions (Progress Manager tab / mobile Images tab)
# ---------------------------------------------------------------------------------------------

def _keys(targets):
    out = []
    for target in targets or ():
        key = target.key if isinstance(target, ImagesRow) else str(target)
        if key and key not in out:
            out.append(key)
    return out


def mark_for_rerun(path, targets):
    """Set the chapters to ``pending`` (Build pending rebuilds them). Returns the count."""
    keys = _keys(targets)

    def apply(data):
        chapters = data.setdefault('chapters', {})
        for key in keys:
            entry = chapters.setdefault(key, {})
            entry['status'] = 'pending'
            entry['error'] = ''
        return len(keys)
    return mutate(path, apply) if keys else 0


def strip_chapter_file(chapter_path):
    """Remove Glossarion's insertions from a chapter file (text-checked). Returns True when it changed."""
    try:
        with open(chapter_path, 'r', encoding='utf-8') as handle:
            html = handle.read()
    except OSError:
        return False
    if not core.has_injections(html):
        return False
    stripped = core.strip_injections(html)
    ok, _hint = core.check_text_unchanged(html, stripped)
    if not ok:
        return False
    write_text_atomic(chapter_path, stripped)
    return True


def write_text_atomic(path, text):
    """Atomic text write (temp file + ``os.replace``; retried briefly on Windows sharing errors)."""
    import tempfile

    folder = os.path.dirname(path) or '.'
    fd, temp = tempfile.mkstemp(prefix='.glx-', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
        for attempt in range(10):
            try:
                os.replace(temp, path)
                temp = None
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        if temp and os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError:
                pass


def delete_entries(output_dir, targets, strip_html=True):
    """Delete chapters' images: the image files, the injected markup (text-checked) and the entries.
    Returns the number of chapters changed."""
    keys = _keys(targets)
    if not keys:
        return 0
    chapters = {chapter.key: chapter for chapter in translated_chapters(output_dir)}
    path = core.progress_path(output_dir)
    data = read_progress_file(path)
    changed = 0
    for key in keys:
        chapter = chapters.get(key)
        entry = (data.get('chapters') or {}).get(key) or {}
        files = [image.get('file') for image in entry.get('images') or () if image.get('file')]
        if chapter is not None:
            files.extend(core.chapter_images_on_disk(output_dir, chapter.number))
            if strip_html:
                strip_chapter_file(chapter.path)
        for name in set(files):
            try:
                os.remove(os.path.join(core.interactive_dir(output_dir), os.path.basename(name)))
            except OSError:
                pass
        changed += 1

    def apply(latest):
        for key in keys:
            (latest.get('chapters') or {}).pop(key, None)
    if os.path.isfile(path):
        mutate(path, apply)
    return changed


def open_targets(output_dir, row):
    """``{'images': [paths], 'chapter': path, 'folder': path}`` for a row's Open actions."""
    folder = core.interactive_dir(output_dir)
    entry = row.entry if isinstance(row, ImagesRow) and isinstance(row.entry, dict) else {}
    images = []
    for image in entry.get('images') or ():
        name = image.get('file')
        if name and image.get('status') == 'completed':
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                images.append(full)
    key = row.key if isinstance(row, ImagesRow) else str(row)
    return {'images': images, 'chapter': os.path.join(output_dir, key), 'folder': folder}


def plan_text(row):
    """A readable summary of a chapter's plan (Show plan)."""
    entry = row.entry if isinstance(row, ImagesRow) and isinstance(row.entry, dict) else {}
    if not entry:
        return "No plan yet for this chapter."
    lines = [f"Status: {STATUS_LABELS.get(row.status, row.status)}"]
    for key in ('text_model', 'image_model', 'built_at'):
        if entry.get(key):
            lines.append(f"{key.replace('_', ' ').capitalize()}: {entry[key]}")
    if entry.get('error'):
        lines.append(f"Error: {entry['error']}")
    for index, image in enumerate(entry.get('images') or (), 1):
        lines.append('')
        lines.append(f"Image {index} after paragraph {image.get('after_paragraph')} ({image.get('status')})")
        if image.get('file'):
            lines.append(f"  File: {image['file']}")
        if image.get('characters'):
            lines.append(f"  Characters: {', '.join(image['characters'])}")
        if image.get('refs'):
            lines.append(f"  References: {', '.join(image['refs'])}")
        if image.get('quote'):
            lines.append(f"  Quote: {image['quote']}")
        lines.append(f"  Prompt: {image.get('prompt', '')}")
        if image.get('error'):
            lines.append(f"  Error: {image['error']}")
    classes = entry.get('classes') or ()
    if classes:
        lines.append('')
        lines.append("Classes: " + ', '.join(f"{c.get('class')} (paragraph {c.get('paragraph')})" for c in classes))
    return '\n'.join(lines)


__all__ = [
    'DONE_STATUSES', 'ImagesModel', 'ImagesRow', 'STATUS_COLORS', 'STATUS_GROUPS', 'STATUS_ICONS', 'STATUS_LABELS',
    'STORED_STATUSES', 'TranslatedChapter', 'book_signature', 'delete_entries', 'empty_progress', 'entry_status',
    'find_progress', 'image_rows', 'image_stats', 'mark_for_rerun', 'mutate', 'needs_build', 'now_iso',
    'open_progress', 'open_targets', 'pending_keys', 'plan_text', 'progress_signature', 'read_progress_file',
    'reload_progress', 'set_book_info', 'set_entry', 'set_status', 'strip_chapter_file', 'tab_count_text',
    'translated_chapters', 'translation_completed', 'write_json_atomic', 'write_text_atomic',
]
