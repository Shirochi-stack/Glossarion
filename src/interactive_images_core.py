"""Interactive Images: the GUI-free core (desktop and mobile).

A separate step after translation, never inside it. For each translated chapter a text model gets
the chapter as numbered paragraphs and returns a JSON plan (where images go, which ``glx-`` style
classes go on which paragraphs, which characters are in each scene, what each image shows); an image
model draws each planned image from its prompt plus reference images (it never sees HTML), and
Glossarion inserts the ``<figure>`` after the located paragraph.

This module holds the pieces that need no model: block numbering, plan parsing and validation,
re-locating a paragraph by its quote, injection and strip (string level, so the rest of the chapter
file stays byte for byte as it was), the visible-text check, file naming, reference images and
their slot selection, chunking, placeholder filling, the settings and the prompt profiles.
Python 3.10 compatible; no Qt / Flet imports.
"""

import copy
import difflib
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from interactive_css_presets import INTERACTIVE_FOLDER, STYLESHEET_FILENAME  # noqa: F401 (re-exported)

PROGRESS_FILENAME = 'interactive_images_progress.json'
REFERENCES_FILENAME = 'references.json'
REFERENCES_SUBFOLDER = 'references'
IMAGE_PREFIX = 'Glossarion_'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
#: Glossarion's own image names: ``Glossarion_ch012_1.png`` (chapter 12, first image)
GENERATED_IMAGE_RE = re.compile(r'^Glossarion_ch(\d+)_(\d+)\.(png|jpe?g|webp)$', re.I)

#: Block elements that get a paragraph number. ``table`` / ``pre`` are numbered whole.
BLOCK_TAGS = ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'pre', 'table', 'div')
_ATOMIC_TAGS = ('table', 'pre')
_LIST_TAGS = ('ul', 'ol', 'dl')

MAX_IMAGES_PER_CHAPTER = 10
MAX_REFERENCES = 14


# ---------------------------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------------------------

def interactive_dir(output_dir):
    return os.path.join(output_dir, INTERACTIVE_FOLDER)


def progress_path(output_dir):
    return os.path.join(interactive_dir(output_dir), PROGRESS_FILENAME)


def references_path(output_dir):
    return os.path.join(interactive_dir(output_dir), REFERENCES_FILENAME)


def image_filename(chapter_num, k, ext='png'):
    """``Glossarion_ch012_1.png``: fixed names, so a rerun replaces the earlier images."""
    try:
        number = int(chapter_num)
    except (TypeError, ValueError):
        number = 0
    ext = str(ext or 'png').lower().lstrip('.')
    if ext == 'jpeg':
        ext = 'jpg'
    return f"{IMAGE_PREFIX}ch{number:03d}_{int(k)}.{ext}"


def image_src(filename):
    """The ``src`` a chapter file (in the output folder) uses for one of Glossarion's images."""
    return f"{INTERACTIVE_FOLDER}/{filename}"


def is_interactive_image_src(src):
    """Whether an ``<img src>`` names one of the Interactive Images (``Interactive_Images/`` folder or a
    ``Glossarion_chNNN_k`` file). Image maps, QA counts and the EPUB rename map leave these alone."""
    value = str(src or '').split('?', 1)[0].split('#', 1)[0].replace('\\', '/')
    if not value or value.startswith('data:'):
        return False
    parts = [part for part in value.split('/') if part and part != '.']
    if any(part.lower() == INTERACTIVE_FOLDER.lower() for part in parts[:-1]):
        return True
    return bool(parts) and bool(GENERATED_IMAGE_RE.match(parts[-1]))


def chapter_images_on_disk(output_dir, chapter_num):
    """``Glossarion_ch<num>_*`` files of one chapter (sorted by k)."""
    folder = interactive_dir(output_dir)
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for name in names:
        match = GENERATED_IMAGE_RE.match(name)
        if match and int(match.group(1)) == int(chapter_num):
            found.append((int(match.group(2)), name))
    return [name for _k, name in sorted(found)]


# ---------------------------------------------------------------------------------------------
# Visible text (the normalization of TransateKRtoEN.ContentProcessor.get_content_hash)
# ---------------------------------------------------------------------------------------------

def visible_text(html):
    """The chapter's visible text: scripts, styles, meta and link removed, whitespace collapsed (exactly
    ``ContentProcessor.get_content_hash``'s normalization)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(str(html or ''), 'html.parser')
    for tag in soup(['script', 'style', 'meta', 'link']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return ' '.join(text.split())


def visible_text_hash(html):
    """sha256 of :func:`visible_text` (equal to ``ContentProcessor.get_content_hash`` of the same HTML)."""
    return hashlib.sha256(visible_text(html).encode('utf-8')).hexdigest()


def check_text_unchanged(original_html, new_html):
    """``(ok, diff_hint)``: the visible text of both must be the same string. ``diff_hint`` shows the first
    80 differing characters of each side when it is not."""
    before = visible_text(original_html)
    after = visible_text(new_html)
    if before == after:
        return True, ''
    i = 0
    limit = min(len(before), len(after))
    while i < limit and before[i] == after[i]:
        i += 1
    return False, f"text differs at {i}: {before[i:i + 80]!r} -> {after[i:i + 80]!r}"


# ---------------------------------------------------------------------------------------------
# Blocks: numbered paragraphs with their offsets in the original string
# ---------------------------------------------------------------------------------------------

@dataclass
class Block:
    n: int                 # 1-based paragraph number
    tag: str
    text: str
    start: int             # offset of '<tag' in the chapter string
    end: int               # offset just after '</tag>'
    insert_at: int         # where a figure after this block goes (after the enclosing list for an <li>)
    open_end: int          # offset just after the start tag's '>'


def _line_starts(text):
    starts = [0]
    for match in re.finditer('\n', text):
        starts.append(match.end())
    return starts


def _tag_offset(starts, tag):
    line = getattr(tag, 'sourceline', None)
    pos = getattr(tag, 'sourcepos', None)
    if line is None or pos is None or line < 1 or line > len(starts):
        return None
    return starts[line - 1] + pos


def _start_tag_end(html, start):
    """Offset just after the start tag beginning at ``start`` (quotes respected)."""
    i, quote, n = start + 1, None, len(html)
    while i < n:
        ch = html[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch == '>':
            return i + 1
        i += 1
    return None


def _element_end(html, start, name):
    """Offset just after the matching ``</name>`` of the element starting at ``start`` (None if unclosed)."""
    open_end = _start_tag_end(html, start)
    if open_end is None:
        return None, None
    if html[open_end - 2:open_end] == '/>':
        return open_end, open_end
    pattern = re.compile(r'<(/?)' + re.escape(name) + r'(?=[\s/>])[^>]*>', re.I)
    depth, pos = 1, open_end
    while depth:
        match = pattern.search(html, pos)
        if not match:
            return open_end, None
        if match.group(1):
            depth -= 1
        elif not match.group(0).endswith('/>'):
            depth += 1
        pos = match.end()
    return open_end, pos


def _norm_space(text):
    return ' '.join(str(text or '').split())


def number_blocks(html):
    """The chapter's text blocks in document order, numbered 1..n.

    A block is a ``p`` / ``h1-h6`` / ``li`` / ``blockquote`` / ``pre`` / ``table`` / ``div`` with text that
    holds no other such block (a ``blockquote`` of paragraphs numbers its paragraphs); tables and ``pre`` are
    numbered whole. Glossarion's own figures are skipped, so the numbers stay the same across a strip and
    re-injection."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body or soup
    starts = _line_starts(html)
    blocks = []

    def candidate(tag):
        return getattr(tag, 'name', None) in BLOCK_TAGS

    for tag in body.find_all(BLOCK_TAGS):
        if any(getattr(parent, 'name', None) in _ATOMIC_TAGS for parent in tag.parents):
            continue
        if any(parent.get('data-glx') == 'img' for parent in tag.parents if hasattr(parent, 'get')):
            continue
        if tag.name not in _ATOMIC_TAGS and tag.find(candidate):
            continue
        text = _norm_space(tag.get_text(' ', strip=True))
        if not text:
            continue
        start = _tag_offset(starts, tag)
        if start is None or html[start:start + len(tag.name) + 1].lower() != '<' + tag.name:
            continue
        open_end, end = _element_end(html, start, tag.name)
        if end is None:
            continue
        insert_at = end
        if tag.name == 'li':
            outer = None
            for parent in tag.parents:
                if getattr(parent, 'name', None) in _LIST_TAGS:
                    outer = parent
            if outer is not None:
                list_start = _tag_offset(starts, outer)
                if list_start is not None:
                    _o, list_end = _element_end(html, list_start, outer.name)
                    if list_end is not None:
                        insert_at = list_end
        blocks.append(Block(len(blocks) + 1, tag.name, text, start, end, insert_at, open_end))
    return blocks


def format_paragraphs(blocks):
    """``[n] text`` lines (the ``{paragraphs}`` placeholder)."""
    return '\n'.join(f"[{block.n}] {block.text}" for block in blocks)


def chapter_title(html, fallback=''):
    match = re.search(r'<title[^>]*>(.*?)</title>', html or '', re.S | re.I)
    title = _norm_space(re.sub(r'<[^>]+>', ' ', match.group(1))) if match else ''
    if not title:
        heading = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', html or '', re.S | re.I)
        title = _norm_space(re.sub(r'<[^>]+>', ' ', heading.group(1))) if heading else ''
    return title or str(fallback or '')


# ---------------------------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------------------------

class PlanError(ValueError):
    pass


@dataclass
class PlannedImage:
    after_paragraph: int
    quote: str
    characters: List[str]
    prompt: str
    matched: Dict[str, str] = field(default_factory=dict)   # name in plan -> glossary name
    status: str = 'planned'                                  # 'planned' | 'unplaced'


@dataclass
class PlannedClass:
    paragraph: int
    quote: str
    css_class: str


@dataclass
class Plan:
    images: List[PlannedImage] = field(default_factory=list)
    classes: List[PlannedClass] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def extract_json_object(text):
    """The first JSON object in a model reply (code fences and leading prose tolerated)."""
    value = str(text or '')
    fence = re.search(r'```(?:json)?\s*\n?(.*?)```', value, re.S | re.I)
    candidates = [fence.group(1)] if fence else []
    candidates.append(value)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        pos = candidate.find('{')
        while pos >= 0:
            try:
                obj, _end = decoder.raw_decode(candidate, pos)
            except ValueError:
                pos = candidate.find('{', pos + 1)
                continue
            if isinstance(obj, dict):
                return obj
            pos = candidate.find('{', pos + 1)
    raise PlanError("plan was not JSON")


def _int(value):
    try:
        return int(str(value).strip().strip('[]'))
    except (TypeError, ValueError):
        return None


def _norm_match(text):
    return ' '.join(re.sub(r'[\W_]+', ' ', str(text or '').casefold()).split())


def match_name(name, glossary_names):
    """The glossary name for a planned character: exact, then case-insensitive, then a raw / alias name
    (``glossary_names`` maps every known spelling to the display name). None when nothing matches."""
    name = _norm_space(name)
    if not name or not glossary_names:
        return None
    if name in glossary_names:
        return glossary_names[name]
    folded = name.casefold()
    for key, display in glossary_names.items():
        if key.casefold() == folded:
            return display
    return None


def glossary_name_map(entries):
    """``{spelling: display name}`` from glossary entries (``glossary_usage`` dicts): the translated name is the
    display name; the raw name and any aliases point at it. Character entries first."""
    names = {}
    ordered = sorted(entries or (), key=lambda e: 0 if str(e.get('type', '')).lower().startswith('char') else 1)
    for entry in ordered:
        display = _norm_space(entry.get('translated_name') or entry.get('raw_name'))
        if not display:
            continue
        for spelling in (entry.get('translated_name'), entry.get('raw_name'), *(entry.get('aliases') or ())):
            spelling = _norm_space(spelling)
            if spelling and spelling not in names:
                names[spelling] = display
    return names


def character_names(entries):
    """Display names of the glossary's characters (``{character_names}``), in glossary order."""
    out = []
    for entry in entries or ():
        kind = str(entry.get('type', '') or '').lower()
        if kind and not kind.startswith('char'):
            continue
        name = _norm_space(entry.get('translated_name') or entry.get('raw_name'))
        if name and name not in out:
            out.append(name)
    return out


def locate_block(blocks, number, quote, threshold=0.8):
    """The block a plan item means: block ``number`` when its text starts with ``quote`` (or nearly, ratio >=
    ``threshold``), else the block whose start best matches ``quote`` anywhere in the chapter; None if none
    reaches ``threshold``."""
    if not blocks:
        return None
    by_number = {block.n: block for block in blocks}
    wanted = _norm_match(quote)
    if not wanted:
        return by_number.get(number)

    def score(block):
        text = _norm_match(block.text)
        if text.startswith(wanted):
            return 1.0
        head = text[:len(wanted)]
        ratio = difflib.SequenceMatcher(None, head, wanted).ratio() if head else 0.0
        if ratio < threshold and wanted in text:
            ratio = max(ratio, threshold)
        return ratio

    block = by_number.get(number)
    if block is not None and score(block) >= threshold:
        return block
    best, best_score = None, 0.0
    for candidate in blocks:
        value = score(candidate)
        distance = abs(candidate.n - (number or 0))
        if value > best_score or (value == best_score and best is not None
                                  and distance < abs(best.n - (number or 0))):
            best, best_score = candidate, value
    return best if best_score >= threshold else None


def parse_plan(text, n_blocks, allowed_classes, max_images, *, glossary_names=None):
    """Validate a planner reply into a :class:`Plan` (paragraph numbers checked later against the blocks).

    ``images`` is capped at ``max_images`` and entries without a prompt are dropped; ``classes`` keeps only
    names in ``allowed_classes``. Character names are matched against ``glossary_names``
    (:func:`glossary_name_map`); unmatched names stay for the prompt text but get no reference images."""
    data = extract_json_object(text)
    plan = Plan()
    allowed = set(allowed_classes or ())
    raw_images = data.get('images') or []
    if not isinstance(raw_images, list):
        raw_images = []
    for item in raw_images:
        if not isinstance(item, dict):
            continue
        prompt = _norm_space(item.get('prompt'))
        if not prompt:
            plan.warnings.append("dropped an image without a prompt")
            continue
        number = _int(item.get('after_paragraph'))
        raw_chars = item.get('characters') or []
        if isinstance(raw_chars, str):
            raw_chars = [part for part in re.split(r'[,;]', raw_chars)]
        chars = []
        for name in raw_chars if isinstance(raw_chars, list) else ():
            name = _norm_space(name)
            if name and name not in chars:
                chars.append(name)
        matched = {}
        for name in chars:
            hit = match_name(name, glossary_names)
            if hit:
                matched[name] = hit
        plan.images.append(PlannedImage(number if number is not None else 0, _norm_space(item.get('quote')),
                                        chars, prompt, matched))
    if max_images is not None and len(plan.images) > max(0, int(max_images)):
        plan.warnings.append(f"kept {max(0, int(max_images))} of {len(plan.images)} images")
        plan.images = plan.images[:max(0, int(max_images))]
    raw_classes = data.get('classes') or []
    for item in raw_classes if isinstance(raw_classes, list) else ():
        if not isinstance(item, dict):
            continue
        name = str(item.get('class') or '').strip().lstrip('.')
        number = _int(item.get('paragraph'))
        if not name or number is None:
            continue
        if name not in allowed:
            plan.warnings.append(f"dropped unknown class '{name}'")
            continue
        plan.classes.append(PlannedClass(number, _norm_space(item.get('quote')), name))
    return plan


def merge_plans(plans, max_images):
    """One plan from per-chunk plans: images and classes sorted by paragraph, images capped."""
    merged = Plan()
    for plan in plans:
        merged.images.extend(plan.images)
        merged.classes.extend(plan.classes)
        merged.warnings.extend(plan.warnings)
    merged.images.sort(key=lambda item: item.after_paragraph)
    merged.classes.sort(key=lambda item: item.paragraph)
    if len(merged.images) > max_images:
        merged.images = merged.images[:max_images]
    return merged


# ---------------------------------------------------------------------------------------------
# Chunking (whole chapter when it fits, else block-aligned chunks with global numbers)
# ---------------------------------------------------------------------------------------------

def chunk_blocks(blocks, budget_tokens, count_tokens, overhead_tokens=0):
    """Split ``blocks`` into consecutive runs whose ``[n] text`` lines fit ``budget_tokens`` (minus
    ``overhead_tokens``, the prompt without its paragraphs). Never splits inside a block; a block that alone is
    too large gets its own chunk. ``budget_tokens <= 0`` = one chunk."""
    if not blocks:
        return []
    if not budget_tokens or budget_tokens <= 0:
        return [list(blocks)]
    room = max(1, int(budget_tokens) - int(overhead_tokens or 0))
    chunks, current, used = [], [], 0
    for block in blocks:
        cost = max(1, int(count_tokens(f"[{block.n}] {block.text}\n") or 0))
        if current and used + cost > room:
            chunks.append(current)
            current, used = [], 0
        current.append(block)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def chunk_image_share(max_images, chunk_len, total_len):
    """``{images_per_chapter}`` of one chunk: its share of the chapter's images (rounded up)."""
    if not total_len or max_images <= 0:
        return 0
    return min(max_images, int(math.ceil(max_images * chunk_len / float(total_len))))


# ---------------------------------------------------------------------------------------------
# Placeholders (by name only; JSON braces in a prompt stay literal)
# ---------------------------------------------------------------------------------------------

PLANNER_PLACEHOLDERS = ('images_per_chapter', 'character_names', 'image_subjects', 'css_classes', 'chapter_title',
                        'paragraphs')
IMAGE_PLACEHOLDERS = ('scene_prompt', 'characters', 'reference_list')
CSS_PLACEHOLDERS = ('user_request', 'css_classes')
REFERENCE_LINE_PLACEHOLDERS = ('n', 'name')
#: Placeholders whose template line is removed when they expand to nothing (the glossary placeholders'
#: rule: an empty text removes the line), so the default "Characters" mode and a scene without characters
#: leave no empty label behind.
LINE_DROP_PLACEHOLDERS = ('image_subjects', 'characters')


def fill_placeholders(template, values, drop_empty=()):
    """``template`` with each ``{name}`` whose name is a key of ``values`` replaced by its value; every other
    brace (JSON in the prompt) is left as it is. Nothing is added outside the placeholders. A line holding a
    placeholder named in ``drop_empty`` whose value is empty is removed with its line break."""
    from prompt_defaults import PLACEHOLDER_RE

    text = str(template or '')
    for name in drop_empty or ():
        if name in values and str(values[name]) == '' and ('{' + name + '}') in text:
            lines = text.split('\n')
            kept = [line for line in lines if ('{' + name + '}') not in line]
            text = '\n'.join(kept)

    def replace(match):
        name = match.group(1)
        if name in values:
            return str(values[name])
        return match.group(0)

    return PLACEHOLDER_RE.sub(replace, text)


# ---------------------------------------------------------------------------------------------
# Injection and strip (string level)
# ---------------------------------------------------------------------------------------------

FIGURE_TEMPLATE = '<figure class="glx-figure" data-glx="img"><img class="glx-illustration" src="{src}" alt=""/></figure>'
_FIGURE_RE = re.compile(r'<figure\b[^>]*\bdata-glx\s*=\s*["\']img["\'][^>]*>.*?</figure>', re.S | re.I)
_ATTR_RE_T = r'(\s{name}\s*=\s*)(["\'])(.*?)\2'


def figure_html(src):
    return FIGURE_TEMPLATE.format(src=str(src).replace('"', '%22'))


def _get_attr(tag_text, name):
    match = re.search(_ATTR_RE_T.format(name=re.escape(name)), tag_text, re.S | re.I)
    return match.group(3) if match else None


def _add_classes_to_start_tag(tag_text, classes):
    """The start tag with ``classes`` added to ``class`` and listed in ``data-glx-cls``."""
    existing = _get_attr(tag_text, 'class')
    current = (existing or '').split()
    marked = (_get_attr(tag_text, 'data-glx-cls') or '').split()
    new = [name for name in classes if name not in current]
    if not new:
        return tag_text
    close = '/>' if tag_text.endswith('/>') else '>'
    head = tag_text[:-len(close)]
    if existing is None:
        head += f' class="{" ".join(new)}"'
    else:
        head = re.sub(_ATTR_RE_T.format(name='class'),
                      lambda m: f"{m.group(1)}{m.group(2)}{(m.group(3) + ' ' if m.group(3) else '') + ' '.join(new)}{m.group(2)}",
                      head, count=1, flags=re.S | re.I)
    if marked:
        head = re.sub(_ATTR_RE_T.format(name='data-glx-cls'),
                      lambda m: f"{m.group(1)}{m.group(2)}{' '.join(marked + new)}{m.group(2)}",
                      head, count=1, flags=re.S | re.I)
    else:
        head += f' data-glx-cls="{" ".join(new)}"'
    return head + close


def _strip_start_tag(tag_text):
    marked = (_get_attr(tag_text, 'data-glx-cls') or '').split()
    text = re.sub(r'\sdata-glx-cls\s*=\s*(["\']).*?\1', '', tag_text, count=1, flags=re.S | re.I)
    if not marked:
        return text

    def fix(m):
        kept = [name for name in m.group(3).split() if name not in marked]
        if not kept:
            return ''
        return f"{m.group(1)}{m.group(2)}{' '.join(kept)}{m.group(2)}"

    return re.sub(_ATTR_RE_T.format(name='class'), fix, text, count=1, flags=re.S | re.I)


_MARKED_TAG_RE = re.compile(r'<[A-Za-z][A-Za-z0-9]*\b[^>]*\sdata-glx-cls\s*=[^>]*>', re.S)


#: The ``glx-book`` wrapper around the chapter body (EPUB builders rebuild ``<body>`` without its attributes,
#: so the book class sits on a wrapper div of its own).
BOOK_WRAPPER_OPEN = '<div class="glx-book" data-glx="book">'
BOOK_WRAPPER_CLOSE = '</div>'
_BODY_CLOSE_RE = re.compile(r'</body\s*>', re.I)


def _strip_book_wrapper(text):
    start = text.find(BOOK_WRAPPER_OPEN)
    if start < 0:
        return text
    closes = list(_BODY_CLOSE_RE.finditer(text, start))
    if closes:
        end = closes[-1].start()
        if text[end - len(BOOK_WRAPPER_CLOSE):end] == BOOK_WRAPPER_CLOSE:
            text = text[:end - len(BOOK_WRAPPER_CLOSE)] + text[end:]
    return text[:start] + text[start + len(BOOK_WRAPPER_OPEN):]


def strip_injections(html):
    """``html`` without anything Glossarion inserted: every ``[data-glx="img"]`` figure, the ``glx-book``
    wrapper, and each class named in a ``data-glx-cls`` attribute (plus the attribute). The user's own
    classes stay."""
    text = _FIGURE_RE.sub('', str(html or ''))
    text = _strip_book_wrapper(text)
    return _MARKED_TAG_RE.sub(lambda m: _strip_start_tag(m.group(0)), text)


def has_injections(html):
    value = str(html or '')
    return bool(_FIGURE_RE.search(value) or 'data-glx-cls' in value or BOOK_WRAPPER_OPEN in value)


def apply_plan(html, blocks, images, classes, *, book_class=False):
    """Insert figures and classes into ``html`` (already stripped). ``images``: ``[(block, src)]``;
    ``classes``: ``[(block, class)]``; ``book_class``: wrap the body content in the ``glx-book`` div."""
    edits = []  # (offset, kind, payload): insertions at an offset, or start-tag rewrites
    class_map = {}
    for block, name in classes:
        class_map.setdefault(block.start, (block, []))[1].append(name)
    for start, (block, names) in class_map.items():
        edits.append((start, block.open_end, 'tag', names))
    by_offset = {}
    for block, src in images:
        by_offset.setdefault(block.insert_at, []).append(figure_html(src))
    for offset, figures in by_offset.items():
        edits.append((offset, offset, 'insert', ''.join(figures)))
    out = html
    for start, end, kind, payload in sorted(edits, key=lambda e: (e[0], 0 if e[2] == 'insert' else 1), reverse=True):
        if kind == 'insert':
            out = out[:start] + payload + out[start:]
        else:
            out = out[:start] + _add_classes_to_start_tag(out[start:end], payload) + out[end:]
    if book_class:
        body = re.search(r'<body\b[^>]*>', out, re.I)
        closes = list(_BODY_CLOSE_RE.finditer(out))
        if body and closes and closes[-1].start() >= body.end():
            end = closes[-1].start()
            out = out[:body.end()] + BOOK_WRAPPER_OPEN + out[body.end():end] + BOOK_WRAPPER_CLOSE + out[end:]
    return out


def replace_image_src(html, old_src, new_src):
    return html.replace(f'src="{old_src}"', f'src="{new_src}"')


def remove_figure(html, src):
    pattern = re.compile(r'<figure\b[^>]*\bdata-glx\s*=\s*["\']img["\'][^>]*>\s*<img\b[^>]*src="'
                         + re.escape(src) + r'"[^>]*>\s*</figure>', re.S | re.I)
    return pattern.sub('', html, count=1)


# ---------------------------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------------------------

#: (pattern, max reference images) for image models; first match wins. None = no limit known.
REFERENCE_IMAGE_LIMITS = (
    (re.compile(r'dall-?e-?3', re.I), 0),
    (re.compile(r'dall-?e-?2', re.I), 1),
    (re.compile(r'gpt-image', re.I), 1),
    (re.compile(r'imagen', re.I), 0),
    (re.compile(r'gemini.*image|nano-?banana', re.I), 14),
    (re.compile(r'^(?:nan|nanogpt|authnan)/', re.I), None),
)


def reference_limit(model):
    """How many reference images ``model`` takes (None = no known limit)."""
    value = str(model or '')
    for pattern, limit in REFERENCE_IMAGE_LIMITS:
        if pattern.search(value):
            return limit
    return None


def effective_reference_count(model, max_refs):
    limit = reference_limit(model)
    max_refs = max(0, int(max_refs or 0))
    return max_refs if limit is None else min(max_refs, limit)


NO_REFERENCE_WARNING = "This image model takes no reference images; characters are described by name only"


def reference_note(model):
    """``(level, text)`` the settings screens show next to the image model (desktop dialog and mobile):
    ``('warning', NO_REFERENCE_WARNING)`` for a model that takes none, ``('info', ...)`` with its limit,
    ``('', '')`` for an unknown or empty model."""
    model = str(model or '').strip()
    limit = reference_limit(model) if model else None
    if model and limit == 0:
        return 'warning', NO_REFERENCE_WARNING
    if model and limit is not None:
        return 'info', (f"This image model takes up to {limit} reference image"
                        f"{'s' if limit != 1 else ''}; more are not sent")
    return '', ''


def reference_rows(refs):
    """``[(name, [relative paths])]`` of the characters, sorted case-insensitively (the references editors)."""
    chars = (refs or {}).get('characters') or {}
    return sorted(((name, list(paths)) for name, paths in chars.items()), key=lambda kv: kv[0].casefold())


def add_reference_names(refs, names):
    """List ``names`` (typed or from the glossary) as characters without images; returns how many were new."""
    chars = refs.setdefault('characters', {})
    added = 0
    for name in names or ():
        name = ' '.join(str(name or '').split())
        if name and name not in chars:
            chars[name] = []
            added += 1
    return added


def clear_reference_character(output_dir, refs, name):
    """Delete a character's reference images and its row."""
    for rel in list((refs.get('characters') or {}).get(name, [])):
        remove_reference_image(output_dir, refs, rel)
    (refs.get('characters') or {}).pop(name, None)


def empty_references():
    return {'version': 1, 'characters': {}, 'style': []}


def load_references(output_dir):
    try:
        with open(references_path(output_dir), 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return empty_references()
    if not isinstance(data, dict):
        return empty_references()
    chars = data.get('characters') if isinstance(data.get('characters'), dict) else {}
    style = data.get('style') if isinstance(data.get('style'), list) else []
    return {
        'version': 1,
        'characters': {str(k): [str(p) for p in v if isinstance(p, str)]
                       for k, v in chars.items() if isinstance(v, list)},
        'style': [str(p) for p in style if isinstance(p, str)],
    }


def save_references(output_dir, refs):
    from interactive_images_progress_core import write_json_atomic

    os.makedirs(interactive_dir(output_dir), exist_ok=True)
    write_json_atomic(references_path(output_dir), refs)


def _safe_stem(text):
    stem = re.sub(r'[^\w-]+', '_', str(text or '').strip().lower(), flags=re.U).strip('_')
    return stem[:40] or 'ref'


def is_reference_image(path):
    """True when ``path`` decodes as a PNG / JPEG / WebP image (``safe_image.open_image``, the rule for
    images Glossarion did not make itself); the references editors refuse anything else."""
    if os.path.splitext(str(path or ''))[1].lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        from safe_image import open_image
        with open_image(path) as image:
            return bool(image.size)
    except Exception:
        return False


def add_reference_image(output_dir, refs, source, character=None, *, data=None, ext=None):
    """Copy an image into ``Interactive_Images/references/`` and list it under ``character`` (None = an art
    style reference). ``data`` (bytes) replaces reading ``source`` (mobile content URIs). Returns the relative
    path stored in ``refs`` (mutated)."""
    folder = os.path.join(interactive_dir(output_dir), REFERENCES_SUBFOLDER)
    os.makedirs(folder, exist_ok=True)
    ext = (ext or os.path.splitext(str(source or ''))[1] or '.png').lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = '.png'
    stem = _safe_stem(character) if character else 'style'
    k = 1
    while os.path.exists(os.path.join(folder, f"{stem}_{k}{ext}")):
        k += 1
    target = os.path.join(folder, f"{stem}_{k}{ext}")
    if data is not None:
        with open(target, 'wb') as handle:
            handle.write(data)
    else:
        shutil.copyfile(source, target)
    rel = f"{REFERENCES_SUBFOLDER}/{os.path.basename(target)}"
    if character:
        refs.setdefault('characters', {}).setdefault(character, []).append(rel)
    else:
        refs.setdefault('style', []).append(rel)
    return rel


def remove_reference_image(output_dir, refs, rel, *, delete_file=True):
    changed = False
    for name in list(refs.get('characters', {})):
        paths = refs['characters'][name]
        if rel in paths:
            paths.remove(rel)
            changed = True
    if rel in refs.get('style', []):
        refs['style'].remove(rel)
        changed = True
    if delete_file and changed:
        try:
            os.remove(os.path.join(interactive_dir(output_dir), rel))
        except OSError:
            pass
    return changed


def select_references(refs, scene_characters, max_refs):
    """``[(name, relative path)]`` for one image: the first image of each scene character (plan order), then
    each one's second image, and so on up to ``max_refs``; the remaining slots take the style references in
    order (name ``None``)."""
    max_refs = max(0, int(max_refs or 0))
    chars = (refs or {}).get('characters') or {}
    lists = []
    for name in scene_characters or ():
        paths = chars.get(name)
        if paths is None:
            for key, value in chars.items():
                if key.casefold() == str(name).casefold():
                    paths = value
                    break
        if paths:
            lists.append((name, list(paths)))
    picked = []
    depth = 0
    while len(picked) < max_refs and any(depth < len(paths) for _n, paths in lists):
        for name, paths in lists:
            if depth < len(paths) and len(picked) < max_refs:
                picked.append((name, paths[depth]))
        depth += 1
    for path in (refs or {}).get('style') or ():
        if len(picked) >= max_refs:
            break
        picked.append((None, path))
    return picked


def reference_list_text(picked, line_template, style_label=''):
    """``{reference_list}``: one line per attached reference from the editable line template
    (``{n}``, ``{name}``; ``{name}`` is the character, or ``style_label`` -- the editable ``style_reference``
    placeholder text -- for an art-style image). No fixed word is added here."""
    lines = []
    for index, (name, _path) in enumerate(picked, 1):
        lines.append(fill_placeholders(line_template, {'n': index, 'name': name or style_label}))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

#: config key -> default (the step reads these from the owner's config in-process)
DEFAULTS = {
    'interactive_images_auto_after_translation': False,
    'interactive_images_recompile_after': True,
    'interactive_images_text_model': '',
    'interactive_images_image_model': '',
    'interactive_images_max_references': 5,
    'interactive_images_resolution': '1K',
    'interactive_images_aspect_ratio': '3:4',
    'interactive_images_planner_max_input_tokens': 0,
    'interactive_images_reference_line': 'Image {n}: {name}',
    'interactive_images_per_chapter': '1',
    'interactive_images_include_gender': True,
    'interactive_images_subjects': 'characters',
    'interactive_images_aspect_choices': ['1:1', '3:4', '4:3', '16:9'],   # in ASPECT_CHOICES order
    'interactive_images_placeholder_texts': {},
    'interactive_css_enabled': False,
    'interactive_css_preset': 'Classic Book',
}

RESOLUTIONS = ('1K', '2K', '4K')
#: The ratios an image can have; ``auto`` = pick one at random from the ticked ``ASPECT_CHOICES``.
ASPECT_CHOICES = ('1:1', '3:4', '4:3', '2:3', '3:2', '9:16', '16:9', '21:9')
ASPECT_RATIOS = ('auto',) + ASPECT_CHOICES
DEFAULT_ASPECT_CHOICES = tuple(DEFAULTS['interactive_images_aspect_choices'])
#: What the planner may illustrate: characters (the default), backgrounds / scenery, or both.
SUBJECTS = ('characters', 'backgrounds', 'both')
SUBJECT_LABELS = {'characters': 'Characters', 'backgrounds': 'Backgrounds', 'both': 'Both'}
#: The images-per-chapter dropdown (0 = styles only)
PER_CHAPTER_CHOICES = tuple(str(n) for n in range(0, MAX_IMAGES_PER_CHAPTER + 1))


def settings_values(config):
    """The plain (non-prompt) settings as ``{config key: value}`` with their defaults (the settings screens'
    starting values; the same keys :func:`settings_from_config` reads)."""
    s = settings_from_config(config)
    return {
        'interactive_images_text_model': s.text_model,
        'interactive_images_image_model': s.image_model,
        'interactive_images_max_references': s.max_references,
        'interactive_images_resolution': s.resolution,
        'interactive_images_aspect_ratio': s.aspect_ratio,
        'interactive_images_auto_after_translation': s.auto_after_translation,
        'interactive_images_recompile_after': s.recompile_after,
        'interactive_images_include_gender': s.include_gender,
        'interactive_images_subjects': s.subjects,
        'interactive_images_aspect_choices': list(s.aspect_choices),
    }


def coerce_aspect_choices(value):
    """The ticked ratios as a list in :data:`ASPECT_CHOICES` order (unknown values dropped; a bad value = the
    default selection; an empty list stays empty: Auto then leaves the ratio to the model)."""
    if isinstance(value, str):
        value = [part.strip() for part in re.split(r'[,;\s]+', value) if part.strip()]
    if not isinstance(value, (list, tuple, set)):
        return list(DEFAULT_ASPECT_CHOICES)
    wanted = {str(item).strip() for item in value}
    return [ratio for ratio in ASPECT_CHOICES if ratio in wanted]


def coerce_setting(key, value):
    """A settings screen's value for ``key`` as the step reads it (clamped / typed)."""
    if key == 'interactive_images_max_references':
        try:
            return max(0, min(MAX_REFERENCES, int(value)))
        except (TypeError, ValueError):
            return DEFAULTS[key]
    if key in ('interactive_images_auto_after_translation', 'interactive_images_recompile_after',
               'interactive_images_include_gender', 'interactive_css_enabled'):
        return _bool(value)
    if key == 'interactive_images_resolution':
        return value if value in RESOLUTIONS else DEFAULTS[key]
    if key == 'interactive_images_aspect_ratio':
        return value if value in ASPECT_RATIOS else DEFAULTS[key]
    if key == 'interactive_images_subjects':
        return value if value in SUBJECTS else DEFAULTS[key]
    if key == 'interactive_images_aspect_choices':
        return coerce_aspect_choices(value)
    return str(value or '').strip()


def parse_per_chapter(value, default=1):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        number = default
    return max(0, min(MAX_IMAGES_PER_CHAPTER, number))


@dataclass
class InteractiveImagesSettings:
    text_model: str
    image_model: str
    per_chapter: int
    max_references: int
    resolution: str
    aspect_ratio: str
    planner_max_input_tokens: int
    reference_line: str
    planner_prompt: str
    image_system_prompt: str
    image_prompt: str
    css_prompt: str
    auto_after_translation: bool
    recompile_after: bool
    css_enabled: bool
    css_preset: str
    include_gender: bool = True
    subjects: str = 'characters'
    aspect_choices: tuple = DEFAULT_ASPECT_CHOICES
    placeholder_texts: dict = field(default_factory=dict)   # the stored expansion texts ({name: text})


def _cfg(config, key):
    value = (config or {}).get(key)
    return DEFAULTS.get(key) if value is None else value


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def settings_from_config(config):
    """The step's settings from an owner config (the active profile's prompts are its runtime keys)."""
    import prompt_defaults as pd

    def prompt(key, default):
        value = (config or {}).get(key)
        return default if value is None else str(value)

    try:
        max_refs = int(_cfg(config, 'interactive_images_max_references'))
    except (TypeError, ValueError):
        max_refs = 5
    try:
        budget = int(_cfg(config, 'interactive_images_planner_max_input_tokens') or 0)
    except (TypeError, ValueError):
        budget = 0
    return InteractiveImagesSettings(
        text_model=str(_cfg(config, 'interactive_images_text_model') or '').strip(),
        image_model=str(_cfg(config, 'interactive_images_image_model') or '').strip(),
        per_chapter=parse_per_chapter(_cfg(config, 'interactive_images_per_chapter')),
        max_references=max(0, min(MAX_REFERENCES, max_refs)),
        resolution=str(_cfg(config, 'interactive_images_resolution') or '1K'),
        aspect_ratio=str(_cfg(config, 'interactive_images_aspect_ratio') or 'auto'),
        planner_max_input_tokens=max(0, budget),
        reference_line=str(_cfg(config, 'interactive_images_reference_line') or ''),
        planner_prompt=prompt('interactive_images_planner_prompt', pd.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT),
        image_system_prompt=prompt('interactive_images_image_system_prompt',
                                   pd.DEFAULT_INTERACTIVE_IMAGES_IMAGE_SYSTEM_PROMPT),
        image_prompt=prompt('interactive_images_image_prompt', pd.DEFAULT_INTERACTIVE_IMAGES_IMAGE_PROMPT),
        css_prompt=prompt('interactive_images_css_prompt', pd.DEFAULT_INTERACTIVE_IMAGES_CSS_PROMPT),
        auto_after_translation=_bool(_cfg(config, 'interactive_images_auto_after_translation')),
        recompile_after=_bool(_cfg(config, 'interactive_images_recompile_after')),
        css_enabled=_bool(_cfg(config, 'interactive_css_enabled')),
        css_preset=str(_cfg(config, 'interactive_css_preset') or 'Classic Book'),
        include_gender=_bool(_cfg(config, 'interactive_images_include_gender')),
        subjects=coerce_setting('interactive_images_subjects', _cfg(config, 'interactive_images_subjects')),
        aspect_choices=tuple(coerce_aspect_choices(_cfg(config, 'interactive_images_aspect_choices'))),
        placeholder_texts=stored_placeholder_texts(config),
    )


# ---------------------------------------------------------------------------------------------
# Placeholder texts (the user-editable expansions of the placeholders the step fills)
# ---------------------------------------------------------------------------------------------
#
# Nothing the step sends to a model is written in code. Every sentence, label or word that reaches a request
# comes from (a) the profile's own prompts or (b) one of the texts below, each of which the user can edit
# (desktop: right-click a prompt editor > "Edit placeholder text...", mobile: Profile > Placeholder texts).
# The stored overrides are ``{name: text}`` in ``interactive_images_placeholder_texts``; ``reference_line``
# keeps its own older key. An empty text expands to nothing.

PLACEHOLDER_TEXTS_KEY = 'interactive_images_placeholder_texts'
REFERENCE_LINE_KEY = 'interactive_images_reference_line'

_CHARACTER_GENDER_TEXT = '{name} ({gender})'
_REFERENCE_LINE_TEXT = DEFAULTS[REFERENCE_LINE_KEY]
_SUBJECT_TEXTS = {
    'characters': '',
    'backgrounds': ("Illustrate scenery and places only, with no characters in the picture: leave the "
                    "characters list empty."),
    'both': ("You may also pick scenery or establishing moments with no characters in them: leave the "
             "characters list empty for those."),
}


def placeholder_text_defs():
    """``[{'name', 'default', 'note'}]``: the editable placeholder texts, in editor order."""
    return [
        {'name': 'character_gender', 'default': _CHARACTER_GENDER_TEXT,
         'note': ("How a character is written in {character_names} (planner) and {characters} (image prompt) "
                  "when a gender is confidently known (Include gender on; the glossary and the Relations & "
                  "Gender Tracker agree). Tokens: {name}, {gender}. Any other character is just the name. "
                  "An empty text sends no gender.")},
        {'name': 'gender_male', 'default': 'male',
         'note': "The word {gender} stands for when the character is male."},
        {'name': 'gender_female', 'default': 'female',
         'note': "The word {gender} stands for when the character is female."},
        {'name': 'image_subjects_characters', 'default': _SUBJECT_TEXTS['characters'],
         'note': ("{image_subjects} in the planner prompt when Image subjects is Characters (the line is "
                  "removed when this text is empty).")},
        {'name': 'image_subjects_backgrounds', 'default': _SUBJECT_TEXTS['backgrounds'],
         'note': "{image_subjects} in the planner prompt when Image subjects is Backgrounds."},
        {'name': 'image_subjects_both', 'default': _SUBJECT_TEXTS['both'],
         'note': "{image_subjects} in the planner prompt when Image subjects is Both."},
        {'name': 'style_reference', 'default': 'art style',
         'note': "{name} in a reference line for an art-style image (a character image uses the character's name)."},
        {'name': 'reference_line', 'default': _REFERENCE_LINE_TEXT,
         'note': ("One line of {reference_list} in the image prompt per attached reference image. "
                  "Tokens: {n} (1, 2, ...), {name}.")},
    ]


#: The ``{tokens}`` a placeholder text may contain (the editors list them next to the text).
PLACEHOLDER_TEXT_TOKENS = {'character_gender': ('name', 'gender'), 'reference_line': ('n', 'name')}


def placeholder_default(name):
    for item in placeholder_text_defs():
        if item['name'] == name:
            return item['default']
    return ''


def stored_placeholder_texts(config):
    """The stored ``{name: text}`` overrides of an owner config (known names only; ``reference_line`` excluded)."""
    stored = (config or {}).get(PLACEHOLDER_TEXTS_KEY)
    if not isinstance(stored, dict):
        return {}
    known = {item['name'] for item in placeholder_text_defs()} - {'reference_line'}
    return {str(k): v for k, v in stored.items() if str(k) in known and isinstance(v, str)}


def placeholder_text(source, name):
    """The text placeholder ``name`` expands to. ``source``: an :class:`InteractiveImagesSettings` or an owner
    config (the stored text, else the built-in default)."""
    settings = source if isinstance(source, InteractiveImagesSettings) else settings_from_config(source)
    if name == 'reference_line':
        return settings.reference_line
    stored = settings.placeholder_texts.get(name)
    return placeholder_default(name) if stored is None else stored


def set_placeholder_text(config, name, text):
    """Store ``text`` as placeholder ``name``'s text in ``config`` (a text equal to the default removes the
    override). Returns True when the config changed."""
    text = str(text if text is not None else '')
    default = placeholder_default(name)
    if name == 'reference_line':
        before = config.get(REFERENCE_LINE_KEY)
        config[REFERENCE_LINE_KEY] = text
        return before != text
    stored = dict(stored_placeholder_texts(config))
    before = dict(stored)
    if text == default:
        stored.pop(name, None)
    else:
        stored[name] = text
    config[PLACEHOLDER_TEXTS_KEY] = stored
    return before != stored


def subjects_text(settings):
    """``{image_subjects}``: the editable text of the chosen Image subjects value."""
    return placeholder_text(settings, 'image_subjects_' + settings.subjects)


# ---------------------------------------------------------------------------------------------
# Gender (glossary column + Relations & Gender Tracker), only when it is confident
# ---------------------------------------------------------------------------------------------

#: A gender change that the tracker records counts as dated only when each side has this many observations.
DATED_CHANGE_MIN_OBSERVATIONS = 2


def _gender_class(value):
    """``'male'`` / ``'female'`` for a binary value (aliases resolved), ``'other'`` for any other stated value
    (neutral, nonbinary, mixed, ...), ``''`` for blank / unknown."""
    import gender_tracking as gt

    gender = gt.normalize_gender(value)
    if gender in gt.UNKNOWN_GENDERS:
        return ''
    return gender if gender in gt.BINARY_GENDERS else 'other'


def _dated_gender(entry, chapter_number):
    """The tracker's gender at ``chapter_number`` when the entry records ONE dated change (``male`` then
    ``female`` or the reverse, each side seen at least :data:`DATED_CHANGE_MIN_OBSERVATIONS` times) and the
    chapter is clearly on one side of it; '' otherwise (a flip-flop, a lone outlier, a chapter in the gap or
    no chapter)."""
    import gender_tracking as gt

    try:
        chapter = float(chapter_number)
    except (TypeError, ValueError):
        return ''
    occurrences = sorted((pair for pair in gt.normalized_occurrences(entry)),
                         key=lambda pair: gt.tracker_occurrence_sort_key(pair[0]))
    runs = []   # [gender, count, first chapter number, last chapter number]
    for occurrence, gender in occurrences:
        try:
            number = float(occurrence.get('chapter_num'))
        except (TypeError, ValueError):
            return ''
        if runs and runs[-1][0] == gender:
            runs[-1][1] += 1
            runs[-1][3] = number
        else:
            runs.append([gender, 1, number, number])
    if len(runs) != 2 or any(run[1] < DATED_CHANGE_MIN_OBSERVATIONS for run in runs):
        return ''
    first, second = runs
    if chapter >= second[2]:
        return second[0]
    if chapter <= first[3]:
        return first[0]
    return ''


def tracker_gender(entry, chapter_number=None):
    """``(gender, conflict)`` of one gender-tracker entry: the user's decision when it is male / female; the
    one gender seen when only one was; the gender at ``chapter_number`` when exactly one dated change is
    recorded and the data is unambiguous; otherwise ``('', True)`` (both seen without a clear dated change,
    or a non-binary value mixed with a binary one). No entry / nothing seen = ``('', False)``."""
    import gender_tracking as gt

    if not isinstance(entry, dict):
        return '', False
    decision = gt.normalized_decision(entry)
    if decision in gt.BINARY_GENDERS:
        return decision, False
    counts = gt.gender_counts(entry)
    male, female = counts.get('male', 0), counts.get('female', 0)
    other = sum(count for gender, count in counts.items() if gender not in gt.BINARY_GENDERS)
    if other and (male or female):
        return '', True
    if male and female:
        dated = _dated_gender(entry, chapter_number)
        return (dated, False) if dated else ('', True)
    if male:
        return 'male', False
    if female:
        return 'female', False
    return '', False


def confident_gender(glossary_genders, tracker_entries=(), chapter_number=None):
    """``'male'`` / ``'female'`` when the gender of ONE character is confidently known, else ``''``.

    The rule (the point is to avoid a wrong gender: a missing one only costs a hint):

    * ``glossary_genders``: the gender cells of every glossary row of this character. A stated non-binary value
      (neutral, nonbinary, mixed ...) or rows that disagree = no gender. Blank / unknown = no claim.
    * ``tracker_entries``: the gender-tracker entries of the same character (matched by exact raw name only).
      A conflict (both genders seen without one clear dated change at ``chapter_number``, see
      :func:`tracker_gender`) = no gender, whatever the glossary says.
    * glossary and tracker both claim: they must agree, else no gender. One claims and the other has nothing
      against it: that claim. Neither: no gender.
    """
    classes = {_gender_class(value) for value in glossary_genders or ()} - {''}
    if 'other' in classes or len(classes) > 1:
        return ''
    glossary_claim = next(iter(classes)) if classes else ''
    claims = set()
    for entry in tracker_entries or ():
        claim, conflict = tracker_gender(entry, chapter_number)
        if conflict:
            return ''
        if claim:
            claims.add(claim)
    if len(claims) > 1:
        return ''
    tracker_claim = next(iter(claims)) if claims else ''
    if glossary_claim and tracker_claim and glossary_claim != tracker_claim:
        return ''
    return glossary_claim or tracker_claim


def load_gender_tracker(output_dir, glossary_path=None):
    """The book's gender tracker (``<glossary stem>_gender_tracker.json``; its entries are keyed by the raw
    name), ``{}`` when there is none."""
    path = glossary_path or find_book_glossary(output_dir)
    if not path:
        return {}
    try:
        import gender_tracking as gt

        data = gt.read_json_file(gt.tracker_path_for_glossary(path))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def character_genders(entries, tracker, chapter_number=None):
    """``{display name: 'male' | 'female'}`` for the glossary's characters whose gender is confident
    (:func:`confident_gender`). Rows sharing a display name are one character (their cells must agree); a
    tracker entry is matched by the row's raw name, never by a similar name."""
    import gender_tracking as gt

    rows = {}
    for entry in entries or ():
        kind = str(entry.get('type', '') or '').lower()
        if kind and not kind.startswith('char'):
            continue
        name = _norm_space(entry.get('translated_name') or entry.get('raw_name'))
        if not name:
            continue
        genders, trackers = rows.setdefault(name, ([], []))
        genders.append(entry.get('gender'))
        found = gt.tracker_entry_for_raw(tracker, entry.get('raw_name'))
        if found is not None and all(found is not other for other in trackers):
            trackers.append(found)
    out = {}
    for name, (genders, trackers) in rows.items():
        gender = confident_gender(genders, trackers, chapter_number)
        if gender:
            out[name] = gender
    return out


def format_character(name, gender, settings):
    """The character as the planner / image prompt lists it: the ``character_gender`` placeholder text with
    ``{name}`` and ``{gender}`` (the ``gender_male`` / ``gender_female`` text) when ``gender`` is known and
    Include gender is on, else just the name."""
    if not gender or not settings.include_gender:
        return name
    fmt = placeholder_text(settings, 'character_gender')
    label = placeholder_text(settings, 'gender_' + gender)
    if not fmt.strip() or not label.strip():
        return name
    return fill_placeholders(fmt, {'name': name, 'gender': label})


def pick_aspect_ratio(settings, chapter_number, index):
    """The ratio of image ``index`` (1-based) of chapter ``chapter_number``: the fixed ratio, or -- with Auto --
    one of the ticked ratios chosen pseudo-randomly from a hash of ``chapter:index`` (stable across reruns of
    the same chapter, different between images). '' = send no ratio (Auto with nothing ticked)."""
    ratio = str(settings.aspect_ratio or '')
    if ratio != 'auto':
        return ratio if ratio in ASPECT_CHOICES else ''
    choices = list(settings.aspect_choices or ())
    if not choices:
        return ''
    try:
        chapter = int(chapter_number)
    except (TypeError, ValueError):
        chapter = 0
    digest = hashlib.sha256(f"{chapter}:{int(index)}".encode('ascii')).digest()
    return choices[int.from_bytes(digest[:4], 'big') % len(choices)]


# ---------------------------------------------------------------------------------------------
# Prompt profiles (the review / glossary refinement profile engine)
# ---------------------------------------------------------------------------------------------

def _profiles_base():
    from prompt_profiles import PromptSetProfiles
    return PromptSetProfiles


def builtin_profile_texts(per_chapter):
    import prompt_defaults as pd
    return {
        'planner': pd.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT,
        'image_system': pd.DEFAULT_INTERACTIVE_IMAGES_IMAGE_SYSTEM_PROMPT,
        'image': pd.DEFAULT_INTERACTIVE_IMAGES_IMAGE_PROMPT,
        'css': pd.DEFAULT_INTERACTIVE_IMAGES_CSS_PROMPT,
        'per_chapter': str(per_chapter),
    }


class InteractiveImagesProfiles(_profiles_base()):
    """Interactive Images prompt profiles, without widgets: each holds the planner prompt, the image system
    prompt, the image prompt template, the CSS generation prompt and the images-per-chapter count (as text).

    Built-ins: Default (1 image per chapter), "Illustrated" (2) and "Styles only" (0: classes only, no image
    model needed). Every action writes the three profile keys and the active profile's runtime keys
    (``interactive_images_*_prompt``, ``interactive_images_per_chapter``) into ``owner.config``."""

    FIELDS = ('planner', 'image_system', 'image', 'css', 'per_chapter')
    PROFILES_KEY = 'interactive_images_prompt_profiles'
    DEFAULT_KEY = 'interactive_images_prompt_profile_default'
    ACTIVE_KEY = 'active_interactive_images_prompt_profile'
    RUNTIME_KEYS = (
        'interactive_images_planner_prompt',
        'interactive_images_image_system_prompt',
        'interactive_images_image_prompt',
        'interactive_images_css_prompt',
        'interactive_images_per_chapter',
    )
    LABEL = 'interactive images prompt profile'
    DEFAULT_BUILTIN = builtin_profile_texts(1)
    BUILTINS = {
        'Illustrated': builtin_profile_texts(2),
        'Styles only': builtin_profile_texts(0),
    }

    def __init__(self, owner, *, persist=None, log=None, on_runtime=None):
        self.on_runtime = on_runtime
        super().__init__(owner, dict(self.DEFAULT_BUILTIN), persist=persist, log=log)

    def _write_runtime(self, texts):
        values = dict(texts)
        values['per_chapter'] = str(parse_per_chapter(values.get('per_chapter', '1')))
        if self.on_runtime is not None:
            self.on_runtime(values)
            return
        config = self.owner.config
        for field_name, key in zip(self.FIELDS, self.RUNTIME_KEYS):
            config[key] = values.get(field_name, '')


def profile_texts_for_run(config):
    """The active profile's texts without building widgets (the runner; falls back to the runtime keys)."""
    settings = settings_from_config(config)
    return {
        'planner': settings.planner_prompt,
        'image_system': settings.image_system_prompt,
        'image': settings.image_prompt,
        'css': settings.css_prompt,
        'per_chapter': str(settings.per_chapter),
    }


# ---------------------------------------------------------------------------------------------
# Glossary of a book
# ---------------------------------------------------------------------------------------------

def find_book_glossary(output_dir):
    """The book's glossary file (the translator's own lookup when importable; else the output folder)."""
    try:
        from TransateKRtoEN import find_glossary_file
        found = find_glossary_file(output_dir, settings={})
        if found:
            return found
    except Exception:
        pass
    for name in ('glossary.csv', 'glossary.md', 'glossary.txt', 'glossary.json'):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_glossary_entries(output_dir, path=None):
    path = path or find_book_glossary(output_dir)
    if not path:
        return []
    try:
        from glossary_usage import parse_glossary_file
        return list(parse_glossary_file(path) or [])
    except Exception:
        return []


__all__ = [name for name in dir() if not name.startswith('_') and name not in (
    'copy', 'difflib', 'hashlib', 'json', 'math', 'os', 're', 'shutil', 'dataclass', 'field',
    'Dict', 'List', 'Optional', 'Tuple')]
