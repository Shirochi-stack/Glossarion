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
#: The stats chips, in order: (group, label, colour) (desktop tab and mobile Images tab).
STAT_CHIPS = (
    ('built', "✅ Built", '#27ae60'),
    ('partial', "⚠️ Partial", 'orange'),
    ('stale', "🕒 Stale", '#d69e2e'),
    ('failed', "❌ Failed", '#e74c3c'),
    ('rejected', "🚫 Rejected", '#e74c3c'),
    ('not_built', "⬜ Not built", '#5a9fd4'),
)
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
    number: int              # the chapter number (the gender tracker's key; not unique: every notice is 0)
    output_file: str
    path: str
    status: str              # translation status
    title: str = ''
    position: int = 0        # 1-based position in the book (OPF spine): unique, names the images
    source_name: str = ''    # the source file the chapter came from (``original_basename``)
    skip_reason: str = ''    # why glossary extraction would skip it too ('' = not skipped)


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


def _stem(name):
    return os.path.splitext(os.path.basename(str(name or '')))[0].casefold()


def glossary_skip_rules(config=None, owner=None):
    """The glossary's skip rules as ``{'skip_title_header_only': bool, 'special': settings}``: what
    glossary extraction skips, read from the user's settings the way the glossary run reads them.

    ``owner`` (the desktop window or a HeadlessOwner): its own run-environment values; otherwise the
    ``config`` keys (the mobile app's snapshot). Neither: the defaults (title/header-only skipped, special
    files skipped with the default keyword lists)."""
    from epub_special_files import glossary_special_file_settings, glossary_special_file_settings_for_owner

    config = config if isinstance(config, dict) else (getattr(owner, 'config', None) or {})
    skip_title = config.get('glossary_skip_title_header_only', True)
    value = getattr(owner, '_glossary_skip_title_header_only_env_value', None)
    if callable(value):
        try:
            skip_title = value() == '1'
        except Exception:
            pass
    if owner is not None and hasattr(owner, 'translate_special_files_var'):
        special = glossary_special_file_settings_for_owner(owner)
    else:
        special = glossary_special_file_settings({
            'TRANSLATE_SPECIAL_FILES': '1' if config.get('translate_special_files') else '0',
            'SPECIAL_FILE_KEYWORDS': str(config.get('special_file_keywords') or ''),
            'SPECIAL_FILE_EXACT': str(config.get('special_file_exact') or ''),
            'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL': '1' if config.get(
                'never_consider_in_between_files_as_special', True) else '0',
        })
    if isinstance(skip_title, str):
        skip_title = skip_title.strip().lower() in ('1', 'true', 'yes', 'on')
    return {'skip_title_header_only': bool(skip_title), 'special': special}


def book_spine(output_dir):
    """The book's OPF spine (HTML file names in reading order); [] when the book has no OPF (text, PDF, ...).
    The translation's own spine reader (workspace ``content.opf``, then the source EPUB)."""
    try:
        from TransateKRtoEN import _single_pass_spine_files
        return list(_single_pass_spine_files(output_dir) or [])
    except Exception:
        return []


_STRUCTURE_CACHE = {}


def _structural_kind(path):
    """The glossary's structural kind of a chapter file (``image_only`` / ``title_header_only`` / ``empty`` / ''),
    cached per file signature."""
    sig = progress_signature(path)
    cached = _STRUCTURE_CACHE.get(path)
    if cached and cached[0] == sig:
        return cached[1]
    try:
        from epub_special_files import classify_glossary_html_document
        with open(path, 'rb') as handle:
            kind = classify_glossary_html_document(handle.read())['structural_kind']
    except Exception:
        kind = ''
    _STRUCTURE_CACHE[path] = (sig, kind)
    return kind


_SKIP_LABELS = {'image_only': 'image-only', 'empty': 'empty', 'title_header_only': 'title/header-only'}


def chapter_skip_reason(path, rules):
    """Why glossary extraction would skip the chapter at ``path`` for its structure ('' = it would not):
    the glossary's own classification (``epub_special_files.classify_glossary_html_document``) of the
    translated HTML; title/header-only only while the glossary's skip setting is on."""
    kind = _structural_kind(path)
    if kind == 'title_header_only' and not (rules or {}).get('skip_title_header_only', True):
        return ''
    return _SKIP_LABELS.get(kind, '')


def translated_chapters(output_dir, *, completed_only=False, rules=None):
    """The book's chapter outputs (HTML) from ``translation_progress.json``, in the book's reading order (the OPF
    spine; chapters the spine does not name follow, by chapter number). ``status`` is the translation status;
    merged chapters (their parent's file) are left out. ``position`` is the unique 1-based place in the book;
    ``skip_reason`` says why the glossary skips a chapter (``rules``: :func:`glossary_skip_rules`)."""
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
                                     full, status, str(entry.get('title') or ''),
                                     source_name=os.path.basename(str(entry.get('original_basename') or ''))))
    _place_in_book(out, output_dir, rules if rules is not None else glossary_skip_rules())
    return out


def _place_in_book(chapters, output_dir, rules):
    """Order ``chapters`` by the OPF spine, set each one's ``position``, and mark the ones the glossary skips."""
    from epub_special_files import glossary_special_file_flags

    spine = book_spine(output_dir)
    index_of = {}
    for index, name in enumerate(spine):
        index_of.setdefault(_stem(name), index)
    flags = glossary_special_file_flags(spine, rules['special']) if spine else []
    order = {}
    for chapter in chapters:
        stem = _stem(chapter.source_name)
        hit = index_of.get(stem) if stem else None
        if hit is None:
            hit = index_of.get(re.sub(r'^response_', '', _stem(chapter.output_file)))
        if hit is not None:
            order[chapter.key] = hit
            chapter.source_name = os.path.basename(spine[hit])
    chapters.sort(key=lambda c: (order[c.key] if c.key in order else len(spine), c.number, c.key))
    unmatched = [c for c in chapters if c.key not in order]
    for chapter in chapters:
        if chapter.key in order:
            chapter.position = order[chapter.key] + 1
    if unmatched:
        # no spine place: keep the chapter number as the position when every number is distinct (the old
        # names), else count on after the spine so that no two chapters share a name
        numbers = [c.number for c in unmatched]
        distinct = not spine and len(set(numbers)) == len(numbers) and min(numbers) > 0
        for offset, chapter in enumerate(unmatched, 1):
            chapter.position = chapter.number if distinct else len(spine) + offset
    for chapter in chapters:
        if chapter.key in order and flags[order[chapter.key]]:
            chapter.skip_reason = 'special file'
        elif translation_completed(chapter):   # only chapters the step can build are classified
            chapter.skip_reason = chapter_skip_reason(chapter.path, rules)


def translation_completed(chapter):
    return str(chapter.status or '').startswith('completed')


#: Completed translations with nothing to illustrate: an image-only copy and an empty chapter.
UNLISTED_TRANSLATION_STATUSES = ('completed_image_only', 'completed_empty')


def chapter_listed(chapter):
    """Whether the Interactive Images list shows a chapter: its translation completed (the chapters the
    runner can build), is not an image-only or empty copy, and is not one the glossary skips
    (``skip_reason``: a special file, an image-only / empty / title-header-only document). A chapter listed
    without an images entry is a "Not built" row; skipped, failed, QA-failed and in-progress translations
    are not listed."""
    return (translation_completed(chapter) and str(chapter.status or '') not in UNLISTED_TRANSLATION_STATUSES
            and not chapter.skip_reason)


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
    rules: Any = None          # glossary_skip_rules(); None = the defaults

    def _read(self, chapter):
        """``(visible-text hash, has Glossarion markup)`` of a chapter file now (cached per file signature)."""
        sig = progress_signature(chapter.path)
        cached = self._hash_cache.get(chapter.path)
        if cached and cached[0] == sig:
            return cached[1], cached[2]
        try:
            with open(chapter.path, 'r', encoding='utf-8') as handle:
                html = handle.read()
            value, injected = core.visible_text_hash(html), core.has_injections(html)
        except OSError:
            value, injected = '', False
        self._hash_cache[chapter.path] = (sig, value, injected)
        return value, injected

    def current_hash(self, chapter):
        """Visible-text hash of a chapter file now (cached per file signature)."""
        return self._read(chapter)[0]

    def has_markup(self, chapter):
        """Whether the chapter file still holds Glossarion's figures / classes / book wrapper."""
        return self._read(chapter)[1]


def open_progress(output_dir, rules=None):
    """The book's images progress (an empty one when no file exists yet). ``rules``: the glossary's skip rules
    (:func:`glossary_skip_rules` of the user's settings; None = the defaults)."""
    path = core.progress_path(output_dir)
    rules = rules if rules is not None else glossary_skip_rules()
    return ImagesModel(output_dir, path, read_progress_file(path), translated_chapters(output_dir, rules=rules),
                       rules=rules)


def reload_progress(model):
    model.data = read_progress_file(model.path)
    model.chapters = translated_chapters(model.output_dir, rules=model.rules)
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
        # rewritten with the same text (a retranslation that came out identical): the insertions are gone
        built = any(isinstance(image, dict) and image.get('status') == 'completed'
                    for image in entry.get('images') or ()) or bool(entry.get('classes'))
        if built and not model.has_markup(chapter):
            return 'stale', True
    return status, False


def row_display(chapter, status, images):
    count = f"{images} image{'s' if images != 1 else ''}" if images else "no images"
    label = STATUS_LABELS.get(status, status)
    return f"{STATUS_ICONS.get(status, '⬜')} [{chapter.position:03d}] Ch.{chapter.number:>4} │ {chapter.key} │ {count} │ {label}"


def image_rows(model, chapters=None):
    """One row per listed chapter (:func:`chapter_listed`), in chapter order; a chapter without an images entry
    is a "Not built" row."""
    rows = []
    entries = model.data.get('chapters') or {}
    if chapters is None:
        chapters = [chapter for chapter in model.chapters if chapter_listed(chapter)]
    for index, chapter in enumerate(chapters):
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
    """The Progress Manager tab label count: ``built/total`` over the listed chapters (and ``k stale``).
    Built is the "Built" chip: completed and styles-only; stale, partial, failed and rejected are not."""
    text = f"{stats.get('built', 0)}/{stats.get('total', 0)}"
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
            files.extend(core.chapter_images_on_disk(output_dir, chapter.position))
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


# ---------------------------------------------------------------------------------------------
# Gallery: the book's AI images, one card each (desktop dialog and mobile sheet share this model)
# ---------------------------------------------------------------------------------------------

@dataclass
class GalleryImage:
    id: str                    # '<chapter key>|<file>': unique within a book
    key: str                   # the chapter's output file name
    chapter_number: int        # the chapter's position in the book (image names; shown as "Ch. n")
    chapter_title: str
    index: int                 # 1-based position among the chapter's images
    file: str                  # file name inside Interactive_Images/
    path: str                  # full path
    characters: List[str] = field(default_factory=list)   # the characters present in the scene
    prompt: str = ''
    aspect_ratio: str = ''

    @property
    def caption(self):
        """``Ch. 12 · A, B`` (the card text)."""
        names = ', '.join(self.characters)
        return f"Ch. {self.chapter_number}" + (f" · {names}" if names else "")


def gallery_images(model):
    """The completed images of the book (their file exists), in chapter then image order."""
    entries = model.data.get('chapters') or {}
    folder = core.interactive_dir(model.output_dir)
    out = []
    for chapter in model.chapters:
        entry = entries.get(chapter.key)
        if not isinstance(entry, dict):
            continue
        for index, image in enumerate(entry.get('images') or (), 1):
            if not isinstance(image, dict) or image.get('status') != 'completed' or not image.get('file'):
                continue
            name = os.path.basename(str(image['file']))
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            names = image.get('scene') or image.get('characters') or []
            out.append(GalleryImage(f"{chapter.key}|{name}", chapter.key, chapter.position, chapter.title, index, name,
                                    full, [str(n) for n in names if n], str(image.get('prompt') or ''),
                                    str(image.get('aspect_ratio') or '')))
    out.sort(key=lambda item: (item.chapter_number, item.key, item.index))
    return out


def gallery_characters(items, glossary_names=()):
    """The names the character filter offers: the glossary's characters plus any name an image has, sorted
    case-insensitively, each once."""
    seen, names = set(), []
    for name in list(glossary_names or ()) + [n for item in items or () for n in item.characters]:
        name = ' '.join(str(name or '').split())
        folded = name.casefold()
        if name and folded not in seen:
            seen.add(folded)
            names.append(name)
    return sorted(names, key=str.casefold)


def gallery_chapters(items):
    """``[(chapter key, label)]`` of the chapters that have images (the chapter filter)."""
    seen, out = set(), []
    for item in items or ():
        if item.key not in seen:
            seen.add(item.key)
            title = f" {item.chapter_title}" if item.chapter_title else ''
            out.append((item.key, f"Ch. {item.chapter_number}{title}"))
    return out


def gallery_filter(items, character='', chapter_key=''):
    """The images with ``character`` in the scene (case-insensitive; '' = any) in chapter ``chapter_key``
    ('' = any)."""
    wanted = ' '.join(str(character or '').split()).casefold()
    out = []
    for item in items or ():
        if chapter_key and item.key != chapter_key:
            continue
        if wanted and wanted not in {' '.join(n.split()).casefold() for n in item.characters}:
            continue
        out.append(item)
    return out


def gallery_thumbnail(path, size=256):
    """JPEG bytes of an image scaled to fit ``size`` x ``size`` (the mobile gallery's cards; ``safe_image`` decodes
    it); None when the file cannot be read."""
    try:
        from io import BytesIO

        from safe_image import open_image

        with open_image(path) as image:
            picture = image.convert('RGB')
        picture.thumbnail((int(size), int(size)))
        buffer = BytesIO()
        picture.save(buffer, format='JPEG', quality=82)
        return buffer.getvalue()
    except Exception:
        return None


def gallery_delete_confirmation(count):
    """The gallery's Delete question (desktop dialog and mobile)."""
    return (f"Delete {count} image(s)?\n"
            "The image files are deleted and their figures are removed from the chapter files "
            "(the text stays as it is).")


@dataclass
class GalleryDeleteResult:
    deleted: int = 0
    chapters: int = 0
    skipped: list = field(default_factory=list)    # [(GalleryImage, reason)]


def delete_gallery_images(output_dir, items):
    """Delete single images: per chapter the figures are removed from the chapter file (text-checked: a chapter
    whose text would change is left as it is and its images are kept), then the image files, then the
    image records of the progress file (an entry left with no image and no class is removed, so the chapter
    shows as not built). Returns a :class:`GalleryDeleteResult`."""
    result = GalleryDeleteResult()
    by_chapter = {}
    for item in items or ():
        by_chapter.setdefault(item.key, []).append(item)
    if not by_chapter:
        return result
    chapters = {chapter.key: chapter for chapter in translated_chapters(output_dir)}
    folder = core.interactive_dir(output_dir)
    removed = {}                                      # chapter key -> set of file names
    for key, group in by_chapter.items():
        chapter = chapters.get(key)
        if chapter is not None:
            try:
                with open(chapter.path, 'r', encoding='utf-8') as handle:
                    original = handle.read()
            except OSError as exc:
                result.skipped.extend((item, f"chapter not readable: {exc}") for item in group)
                continue
            html = original
            for item in group:
                html = core.remove_figure(html, core.image_src(item.file))
            if html != original:
                ok, hint = core.check_text_unchanged(original, html)
                if not ok:
                    result.skipped.extend((item, f"the chapter text would change ({hint})") for item in group)
                    continue
                write_text_atomic(chapter.path, html)
        for item in group:
            try:
                os.remove(os.path.join(folder, os.path.basename(item.file)))
            except FileNotFoundError:
                pass
            except OSError as exc:
                result.skipped.append((item, f"could not delete the file: {exc}"))
                continue
            removed.setdefault(key, set()).add(item.file)
            result.deleted += 1
        if key in removed:
            result.chapters += 1
    path = core.progress_path(output_dir)
    if removed and os.path.isfile(path):
        def apply(data):
            entries = data.get('chapters') or {}
            for key, names in removed.items():
                entry = entries.get(key)
                if not isinstance(entry, dict):
                    continue
                entry['images'] = [image for image in entry.get('images') or ()
                                   if not (isinstance(image, dict) and os.path.basename(str(image.get('file') or ''))
                                           in names)]
                if not entry['images'] and not entry.get('classes'):
                    entries.pop(key, None)
        mutate(path, apply)
    return result


#: Statuses "Retry failed" rebuilds.
RETRY_STATUSES = ('failed', 'partial', 'rejected')

#: Row actions of the Progress Manager tab (desktop context menu) and the mobile Images tab (row sheet /
#: selection bar), in menu order: id -> label.
ROW_ACTIONS = (
    ('build', "🖼 Build / Rebuild"),
    ('retry', "🔁 Retry failed"),
    ('mark', "❓ Mark for rebuild"),
    ('delete', "🗑 Delete images"),
    ('open_images', "🖼 Open image(s)"),
    ('open_chapter', "📄 Open chapter"),
    ('show_plan', "📋 Show plan"),
    ('open_folder', "📂 Open folder"),
)


def retry_rows(rows):
    return [row for row in rows or () if row is not None and row.status in RETRY_STATUSES]


def deletable_rows(rows):
    """Rows with something to delete (a stored entry)."""
    return [row for row in rows or () if row is not None and row.stored_status]


def action_targets(action, rows):
    """The rows an action works on (``retry``: the failed ones; ``delete``: those with an entry)."""
    if action == 'retry':
        return retry_rows(rows)
    if action == 'delete':
        return deletable_rows(rows)
    return [row for row in rows or () if row is not None]


def action_reason(action, rows, *, running=False):
    """Why an action is unavailable for ``rows`` (None = available); the same rules on desktop and mobile."""
    targets = action_targets(action, rows)
    if action in ('build', 'retry', 'delete') and running:
        return "Interactive Images is running"
    if action == 'retry' and not targets:
        return "No failed, partial or rejected chapter selected"
    if action == 'delete' and not targets:
        return "Nothing built for the selected chapters"
    if action == 'open_images' and not any(row.images for row in targets):
        return "No images yet"
    if action not in ('open_folder',) and not targets:
        return "No chapter selected"
    return None


def action_label(action, rows):
    label = dict(ROW_ACTIONS)[action]
    if action in ('build', 'retry', 'delete'):
        label += f" ({len(action_targets(action, rows))})"
    return label


def delete_confirmation(count):
    """The Delete images question (desktop dialog and mobile)."""
    return (f"Delete the interactive images of {count} chapter(s)?\n"
            "The image files are deleted and the figures and classes Glossarion inserted are removed "
            "from the chapter files (the text stays as it is).")


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
    'DONE_STATUSES', 'ImagesModel', 'ImagesRow', 'RETRY_STATUSES', 'ROW_ACTIONS', 'action_label', 'action_reason',
    'action_targets', 'delete_confirmation', 'deletable_rows', 'retry_rows', 'STAT_CHIPS', 'STATUS_COLORS', 'STATUS_GROUPS', 'STATUS_ICONS', 'STATUS_LABELS',
    'STORED_STATUSES', 'TranslatedChapter', 'book_signature', 'delete_entries', 'empty_progress', 'entry_status',
    'chapter_listed', 'find_progress', 'image_rows', 'image_stats', 'mark_for_rerun', 'mutate', 'needs_build', 'now_iso',
    'open_progress', 'open_targets', 'pending_keys', 'plan_text', 'progress_signature', 'read_progress_file',
    'reload_progress', 'set_book_info', 'set_entry', 'set_status', 'strip_chapter_file', 'tab_count_text',
    'translated_chapters', 'translation_completed', 'write_json_atomic', 'write_text_atomic',
    'book_spine', 'chapter_skip_reason', 'glossary_skip_rules',
    'GalleryDeleteResult', 'GalleryImage', 'delete_gallery_images', 'gallery_chapters', 'gallery_characters',
    'gallery_delete_confirmation', 'gallery_filter', 'gallery_images', 'gallery_thumbnail',
]
