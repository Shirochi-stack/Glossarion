"""WeasyPrint-subset HTML-to-PDF engine built on PyMuPDF ``fitz.Story``.

Glossarion Mobile cannot ship WeasyPrint (it needs the Pango/GObject native
libraries), so the PDF call sites (_pdf_worker, pdf_extractor,
epub_converter._generate_pdf and pdf_workspace_compiler) import ``HTML``,
``CSS`` and ``FontConfiguration`` from this module instead when
:func:`is_selected` is true: inside Glossarion Mobile, or when
``GLOSSARION_PDF_ENGINE=mupdf`` forces it (host testing). Desktop keeps
WeasyPrint otherwise.

Only the API those call sites use is provided::

    HTML(string=..., filename=..., base_url=...).render(stylesheets=None) -> Document
    HTML(...).write_pdf(target=None, stylesheets=None, **options)
    CSS(string=..., filename=..., font_config=...)
    FontConfiguration()                                     # no-op
    Document.pages, Document.copy(pages), Document.write_pdf(target=None, **options)
    Page.anchors   {id: (x0, y0, x1, y1)}
    Page.bookmarks [(level, label, (x, y), state)]
    Page.width, Page.height                                 # PDF points

Supported subset:
- ``@page`` size and margins (plain ``@page`` rules; ``:first``/named pages
  are ignored), and ``@top-*`` / ``@bottom-*`` margin boxes whose content is
  ``counter(page)``. Page numbers are stamped per rendered document, the way
  WeasyPrint lays out its page counter.
- Bookmarks for h1-h3, suppressed by a ``bookmark-level: none`` rule on
  ``*``, ``body *``, ``html *`` or ``hN``.
- ``id`` anchors, and internal ``#id`` and external links that resolve
  across documents merged with ``Document.copy()``.
- CSS3 ``break-before/break-after: page`` mapped to MuPDF's
  ``page-break-*`` properties.
- MuPDF's built-in CJK fallback fonts; fonts are subset on write.

Not supported (documented loss versus WeasyPrint): paged-media counters
other than the page number, floats/flex/grid layout, per-element
``bookmark-level`` values, and CSS outside MuPDF's subset (MuPDF ignores it).

PyMuPDF is not thread-safe, so rendering and writing hold a module lock.
fitz is imported lazily; this module stays importable on Python 3.10.
"""

import io
import os
import re
import threading
from urllib.parse import unquote, urlsplit

import mobile_runtime

ENGINE_NAME = "mupdf-story"

# A4 portrait and a 75 CSS px margin: WeasyPrint's user-agent @page defaults.
_DEFAULT_PAGE_SIZE = (595.2756, 841.8898)
_DEFAULT_MARGIN = 56.25
_BOOKMARK_LEVELS = (1, 2, 3)
_MAX_PAGES = 100000
_STALL_LIMIT = 500
# Placed before the first box so a page break on the document's first element
# cannot make fitz.Story re-break on every page (it never makes progress).
_LEADING_SPACER = '<div style="height:0;margin:0;padding:0;border:0"></div>'

_LOCK = threading.RLock()

_PAPER_SIZES = {
    "a3": (841.8898, 1190.5512),
    "a4": (595.2756, 841.8898),
    "a5": (419.5276, 595.2756),
    "a6": (297.6378, 419.5276),
    "b4": (708.6614, 1000.6299),
    "b5": (498.8976, 708.6614),
    "jis-b4": (728.5039, 1031.8110),
    "jis-b5": (515.9055, 728.5039),
    "letter": (612.0, 792.0),
    "legal": (612.0, 1008.0),
    "ledger": (1224.0, 792.0),
}
_UNIT_PT = {
    "pt": 1.0,
    "px": 0.75,
    "in": 72.0,
    "cm": 72.0 / 2.54,
    "mm": 72.0 / 25.4,
    "q": 72.0 / 101.6,
    "pc": 12.0,
    "em": 12.0,
    "rem": 12.0,
    "ex": 6.0,
}
_NAMED_GRAYS = {
    "black": 0.0, "dimgray": 0.41, "dimgrey": 0.41, "gray": 0.5, "grey": 0.5,
    "darkgray": 0.66, "darkgrey": 0.66, "silver": 0.75, "lightgray": 0.83,
    "lightgrey": 0.83, "white": 1.0,
}
_LINK_SCHEMES = ("http", "https", "mailto", "tel", "ftp")

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STYLE_BLOCK_RE = re.compile(r"(<style\b[^>]*>)(.*?)(</style\s*>)", re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RES = (
    re.compile(r'(\bstyle\s*=\s*")([^"]*)(")', re.IGNORECASE),
    re.compile(r"(\bstyle\s*=\s*')([^']*)(')", re.IGNORECASE),
)
_CSS3_BREAK_RE = re.compile(
    r"(?<![\w-])break-(before|after)\s*:\s*(?:page|always|left|right|recto|verso)\b",
    re.IGNORECASE,
)
_MARGIN_BOX_RE = re.compile(
    r"@(top|bottom)-(left|center|right)(?:-corner)?\b[^{]*\{([^{}]*)\}",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_BODY_OPEN_RE = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
_HEAD_CLOSE_RE = re.compile(r"</head\s*>", re.IGNORECASE)
_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_BOOKMARK_NONE_RE = re.compile(r"bookmark-level\s*:\s*none", re.IGNORECASE)
_HEADING_SELECTOR_RE = re.compile(r"^h([1-6])$", re.IGNORECASE)
_ALL_ELEMENTS_SELECTORS = ("*", "body *", "html *", "* *", "html body *")


def _fitz():
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz
    return fitz


def is_available():
    """True when PyMuPDF with ``Story`` support can be imported."""
    try:
        fitz = _fitz()
    except Exception:
        return False
    return hasattr(fitz, "Story") and hasattr(fitz, "DocumentWriter")


def is_selected():
    """True when the shim replaces WeasyPrint at the PDF call sites.

    Glossarion Mobile always uses it; ``GLOSSARION_PDF_ENGINE=mupdf`` forces
    it elsewhere. Desktop never sets either, so it keeps WeasyPrint.
    """
    if os.environ.get("GLOSSARION_PDF_ENGINE", "").strip().lower() == "mupdf":
        return True
    return mobile_runtime.is_mobile()


# --------------------------------------------------------------------------
# WeasyPrint-compatible classes
# --------------------------------------------------------------------------

class FontConfiguration:
    """No-op stand-in: MuPDF resolves fonts itself (built-in + CJK fallback)."""

    def __init__(self, *args, **kwargs):
        pass

    def add_font_face(self, *args, **kwargs):
        return {}


class CSS:
    """A stylesheet given as ``string``, ``filename``/``url`` or ``file_obj``."""

    def __init__(self, guess=None, filename=None, url=None, file_obj=None,
                 string=None, encoding=None, base_url=None, font_config=None,
                 **_kwargs):
        text = string
        if text is None:
            if file_obj is None and hasattr(guess, "read"):
                file_obj = guess
            if file_obj is not None:
                text = _decode(file_obj.read(), encoding)
            else:
                source = filename or url or guess
                if source:
                    with open(_local_path(source), "rb") as handle:
                        text = _decode(handle.read(), encoding)
        self.text = text or ""
        self.base_url = base_url


class HTML:
    """An HTML document given as ``string``, ``filename``/``url`` or ``file_obj``."""

    def __init__(self, guess=None, filename=None, url=None, file_obj=None,
                 string=None, encoding=None, base_url=None, **_kwargs):
        if string is None:
            if file_obj is None and hasattr(guess, "read"):
                file_obj = guess
            if file_obj is not None:
                string = _decode(file_obj.read(), encoding)
            else:
                source = filename or url or guess
                if not source:
                    raise TypeError("HTML() needs string, filename, url or file_obj")
                path = _local_path(source)
                with open(path, "rb") as handle:
                    string = _decode(handle.read(), encoding)
                if base_url is None:
                    base_url = os.path.dirname(os.path.abspath(path))
        self.string = string
        self.base_url = base_url

    def render(self, stylesheets=None, font_config=None, **_options):
        return _render(self.string, self.base_url, stylesheets)

    def write_pdf(self, target=None, stylesheets=None, zoom=1, **options):
        return self.render(stylesheets=stylesheets).write_pdf(target, **options)


class Page:
    """One rendered page. Coordinates are PDF points from the top-left corner."""

    def __init__(self, width, height, source, index):
        self.width = width
        self.height = height
        self.anchors = {}
        self.bookmarks = []
        self.links = []
        self._source = source
        self._index = index


class _Metadata:
    def __init__(self, title=None):
        self.title = title


class Document:
    """Rendered pages, possibly drawn from several ``HTML.render()`` calls."""

    def __init__(self, pages, metadata=None):
        self.pages = list(pages)
        self.metadata = metadata if metadata is not None else _Metadata()

    def copy(self, pages="all"):
        if pages is None or (isinstance(pages, str) and pages == "all"):
            pages = self.pages
        return Document(list(pages), _Metadata(self.metadata.title))

    def write_pdf(self, target=None, zoom=1, attachments=None, finisher=None, **options):
        with _LOCK:
            data = _assemble(self.pages, options, self.metadata.title)
        if target is None:
            return data
        if hasattr(target, "write"):
            target.write(data)
            return None
        with open(_local_path(target), "wb") as handle:
            handle.write(data)
        return None


class _Source:
    """Serialized PDF of one ``HTML.render()`` call (shared by its pages)."""

    def __init__(self, data):
        self.data = data


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _decode(data, encoding=None):
    if isinstance(data, str):
        return data
    return data.decode(encoding or "utf-8", errors="replace")


def _local_path(value):
    """Turn a ``file:`` URL or a plain path into a filesystem path."""
    value = os.fspath(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not value.lower().startswith("file:"):
        return value
    path = unquote(urlsplit(value).path)
    if re.match(r"^/[A-Za-z]:[/\\]", path):
        path = path[1:]
    return path


def _base_dir(base_url):
    """Directory that relative image/CSS references resolve against."""
    if not base_url:
        return None
    path = _local_path(base_url)
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    if path.endswith(("/", "\\")):
        return None
    parent = os.path.dirname(os.path.abspath(path))
    return parent if os.path.isdir(parent) else None


def _length_pt(value, reference=None):
    """CSS length to points; None when it cannot be parsed (e.g. ``auto``)."""
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?|-?\.\d+)\s*([a-zA-Z%]*)\s*$", value or "")
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "":
        return number if number == 0 else number * 0.75
    if unit == "%":
        return number * reference / 100.0 if reference else None
    factor = _UNIT_PT.get(unit)
    return number * factor if factor is not None else None


def _parse_color(value):
    """``(rgb tuple, opacity)`` for the simple colour forms used in Glossarion CSS."""
    text = (value or "").strip().lower()
    match = re.match(r"^rgba?\(([^)]*)\)$", text)
    if match:
        parts = [p.strip() for p in match.group(1).replace("/", ",").split(",") if p.strip()]
        try:
            rgb = []
            for part in parts[:3]:
                if part.endswith("%"):
                    rgb.append(float(part[:-1]) / 100.0)
                else:
                    rgb.append(float(part) / 255.0)
            alpha = 1.0
            if len(parts) > 3:
                alpha_text = parts[3]
                alpha = float(alpha_text[:-1]) / 100.0 if alpha_text.endswith("%") else float(alpha_text)
            if len(rgb) == 3:
                return tuple(max(0.0, min(1.0, c)) for c in rgb), max(0.0, min(1.0, alpha))
        except ValueError:
            pass
    match = re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", text)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4)), 1.0
    if text in _NAMED_GRAYS:
        gray = _NAMED_GRAYS[text]
        return (gray, gray, gray), 1.0
    return (0.0, 0.0, 0.0), 1.0


def _declarations(block):
    """``{property: value}`` from a declaration block (later wins, !important dropped)."""
    result = {}
    for part in block.split(";"):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        name = name.strip().lower()
        value = re.sub(r"\s*!\s*important\s*$", "", value.strip(), flags=re.IGNORECASE)
        if name:
            result[name] = value
    return result


def _translate_breaks(css_text):
    return _CSS3_BREAK_RE.sub(lambda m: "page-break-%s: always" % m.group(1).lower(), css_text)


def _split_page_rules(css_text):
    """Remove ``@page`` blocks (nested braces included) from comment-free CSS.

    Returns ``(css_without_page_rules, [(prelude, body), ...])``. MuPDF's CSS
    parser rejects the nested margin-box rules and then drops the rule after.
    """
    rules = []
    out = []
    pos = 0
    lowered = css_text.lower()
    while True:
        start = lowered.find("@page", pos)
        if start < 0:
            out.append(css_text[pos:])
            break
        brace = css_text.find("{", start)
        if brace < 0:
            out.append(css_text[pos:])
            break
        depth = 0
        end = brace
        while end < len(css_text):
            char = css_text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        out.append(css_text[pos:start])
        rules.append((css_text[start + 5:brace].strip(), css_text[brace + 1:end]))
        pos = end + 1
    return "".join(out), rules


def _clean_css(css_text):
    """Comment-free CSS ready for MuPDF, plus its ``@page`` rules."""
    css_text = _COMMENT_RE.sub("", css_text or "")
    css_text, page_rules = _split_page_rules(css_text)
    return _translate_breaks(css_text), page_rules


def _disabled_bookmark_levels(css_texts):
    disabled = set()
    for css_text in css_texts:
        for selectors, block in _RULE_RE.findall(css_text):
            if not _BOOKMARK_NONE_RE.search(block):
                continue
            for selector in selectors.split(","):
                selector = " ".join(selector.split()).lower()
                if selector in _ALL_ELEMENTS_SELECTORS:
                    disabled.update(_BOOKMARK_LEVELS)
                    continue
                heading = _HEADING_SELECTOR_RE.match(selector)
                if heading:
                    disabled.add(int(heading.group(1)))
    return disabled


class _PageLayout:
    """Page size, margins and page-number boxes from the cascaded ``@page`` rules."""

    def __init__(self):
        self.width, self.height = _DEFAULT_PAGE_SIZE
        self.margins = [_DEFAULT_MARGIN] * 4  # top, right, bottom, left
        self.number_boxes = {}  # (vertical, horizontal) -> style dict or None

    def apply(self, prelude, body):
        if prelude.strip():
            # Named pages and :first/:left/:right/:blank are not modelled.
            return
        for vertical, horizontal, block in _MARGIN_BOX_RE.findall(body):
            decls = _declarations(block)
            content = decls.get("content", "")
            key = (vertical.lower(), horizontal.lower())
            if "counter(page)" in content.replace(" ", "").lower():
                self.number_boxes[key] = decls
            elif content:
                self.number_boxes[key] = None
        decls = _declarations(_MARGIN_BOX_RE.sub("", body))
        if "size" in decls:
            self._apply_size(decls["size"])
        if "margin" in decls:
            values = [_length_pt(v, self.width) for v in decls["margin"].split()]
            values = [v for v in values if v is not None]
            if values:
                if len(values) == 1:
                    values = values * 4
                elif len(values) == 2:
                    values = [values[0], values[1], values[0], values[1]]
                elif len(values) == 3:
                    values = [values[0], values[1], values[2], values[1]]
                self.margins = [max(0.0, v) for v in values[:4]]
        for index, side in enumerate(("top", "right", "bottom", "left")):
            value = _length_pt(decls.get("margin-" + side), self.width)
            if value is not None:
                self.margins[index] = max(0.0, value)

    def _apply_size(self, value):
        tokens = value.lower().split()
        orientation = None
        width = height = None
        lengths = []
        for token in tokens:
            if token in ("landscape", "portrait"):
                orientation = token
            elif token in _PAPER_SIZES:
                width, height = _PAPER_SIZES[token]
            else:
                length = _length_pt(token)
                if length is not None and length > 0:
                    lengths.append(length)
        if lengths:
            width = lengths[0]
            height = lengths[1] if len(lengths) > 1 else lengths[0]
        if width is None:
            width, height = self.width, self.height
        if orientation == "landscape" and width < height:
            width, height = height, width
        elif orientation == "portrait" and width > height:
            width, height = height, width
        self.width, self.height = float(width), float(height)

    def content_rect(self, fitz):
        top, right, bottom, left = self.margins
        rect = fitz.Rect(left, top, self.width - right, self.height - bottom)
        if rect.is_empty or rect.width < 36 or rect.height < 36:
            return fitz.Rect(0, 0, self.width, self.height)
        return rect


def _prepare_html(html_text, user_css_texts):
    """Sanitise markup and stylesheets for MuPDF and collect the page layout."""
    page_rules = []
    cleaned_css = []

    def _style_block(match):
        css_text, rules = _clean_css(match.group(2))
        page_rules.extend(rules)
        cleaned_css.append(css_text)
        return match.group(1) + css_text + match.group(3)

    html_text = _STYLE_BLOCK_RE.sub(_style_block, html_text or "")
    for pattern in _STYLE_ATTR_RES:
        html_text = pattern.sub(
            lambda m: m.group(1) + _translate_breaks(m.group(2)) + m.group(3),
            html_text,
        )

    user_css = []
    for css_text in user_css_texts:
        css_text, rules = _clean_css(css_text)
        page_rules.extend(rules)
        cleaned_css.append(css_text)
        user_css.append(css_text)

    layout = _PageLayout()
    for prelude, body in page_rules:
        layout.apply(prelude, body)

    body_match = _BODY_OPEN_RE.search(html_text)
    if body_match:
        html_text = html_text[:body_match.end()] + _LEADING_SPACER + html_text[body_match.end():]
    else:
        head_match = _HEAD_CLOSE_RE.search(html_text)
        if head_match:
            html_text = html_text[:head_match.end()] + _LEADING_SPACER + html_text[head_match.end():]
        else:
            html_text = _LEADING_SPACER + html_text

    title_match = _TITLE_RE.search(html_text)
    title = " ".join(re.sub(r"<[^>]+>", "", title_match.group(1)).split()) if title_match else None
    return html_text, "\n".join(user_css), layout, _disabled_bookmark_levels(cleaned_css), title or None


def _stylesheet_texts(stylesheets):
    texts = []
    for sheet in stylesheets or ():
        if isinstance(sheet, CSS):
            texts.append(sheet.text)
        elif isinstance(sheet, (str, bytes, os.PathLike)) or hasattr(sheet, "read"):
            texts.append(CSS(sheet).text)
        elif hasattr(sheet, "text"):
            texts.append(str(sheet.text))
    return texts


def _number_box_rect(fitz, layout, vertical, horizontal, fontsize):
    top, right, bottom, left = layout.margins
    # insert_textbox needs about 1.7 x fontsize per line or it draws nothing.
    height = fontsize * 2.2
    if vertical == "bottom":
        band_top, band_height = layout.height - bottom, bottom
    else:
        band_top, band_height = 0.0, top
    if band_height >= height:
        y0 = band_top + (band_height - height) / 2.0
    elif vertical == "bottom":
        y0 = layout.height - height - 2.0
    else:
        y0 = 2.0
    x0 = left if left + 36 < layout.width - right else 0.0
    x1 = layout.width - right if left + 36 < layout.width - right else layout.width
    align = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
    }[horizontal]
    return fitz.Rect(x0, y0, x1, y0 + height), align


def _stamp_page_numbers(fitz, document, layout):
    boxes = [(key, style) for key, style in sorted(layout.number_boxes.items()) if style is not None]
    if not boxes:
        return
    for page_index in range(document.page_count):
        page = document[page_index]
        for (vertical, horizontal), style in boxes:
            fontsize = _length_pt(style.get("font-size", "")) or 10.0
            color, opacity = _parse_color(style.get("color", ""))
            rect, align = _number_box_rect(fitz, layout, vertical, horizontal, fontsize)
            page.insert_textbox(
                rect,
                str(page_index + 1),
                fontsize=fontsize,
                fontname="helv",
                color=color,
                fill_opacity=opacity,
                align=align,
                overlay=True,
            )


def _is_blank(page):
    return (
        not page.get_text("text").strip()
        and not page.get_images(full=False)
        and not page.get_drawings()
    )


def _position_collector(meta, disabled_levels):
    """One-argument ``Story.element_positions`` callback filling ``meta``."""

    def _collect(position):
        meta["positions"] += 1
        opening = bool(int(getattr(position, "open_close", 3) or 0) & 1)
        rect = tuple(float(v) for v in position.rect)
        element_id = getattr(position, "id", None)
        if element_id and opening and element_id not in meta["anchors"]:
            meta["anchors"][element_id] = rect
        level = int(getattr(position, "heading", 0) or 0)
        if opening and level in _BOOKMARK_LEVELS and level not in disabled_levels:
            label = " ".join(str(getattr(position, "text", "") or "").split())
            if label:
                meta["bookmarks"].append((level, label, (rect[0], rect[1]), "open"))
        href = getattr(position, "href", None)
        if href:
            meta["links"].append((rect, str(href)))

    return _collect


def _render(html_text, base_url, stylesheets):
    fitz = _fitz()
    html_text, user_css, layout, disabled_levels, title = _prepare_html(
        html_text, _stylesheet_texts(stylesheets))
    base_dir = _base_dir(base_url)
    with _LOCK:
        archive = fitz.Archive(base_dir) if base_dir else None
        story = fitz.Story(html=html_text, user_css=user_css or None, archive=archive)
        mediabox = fitz.Rect(0, 0, layout.width, layout.height)
        where = layout.content_rect(fitz)
        buffer = io.BytesIO()
        writer = fitz.DocumentWriter(buffer)
        page_meta = []
        previous_fill = None
        stalled = 0
        more = 1
        try:
            while more:
                meta = {"anchors": {}, "bookmarks": [], "links": [], "positions": 0}
                device = writer.begin_page(mediabox)
                more, filled = story.place(where)
                story.element_positions(_position_collector(meta, disabled_levels))
                story.draw(device)
                writer.end_page()
                page_meta.append(meta)

                # A stalled Story re-places the same partial box on every page
                # without reporting any element; a full page is progress.
                fill = tuple(round(float(v), 2) for v in filled)
                partial = fill[3] - fill[1] < where.height - 1
                if more and partial and fill == previous_fill and not meta["positions"]:
                    stalled += 1
                else:
                    stalled = 0
                previous_fill = fill
                if stalled >= _STALL_LIMIT or len(page_meta) >= _MAX_PAGES:
                    raise RuntimeError(
                        "MuPDF Story stopped making progress after %d page(s)" % len(page_meta))
        finally:
            writer.close()

        document = fitz.open("pdf", buffer.getvalue())
        try:
            if document.page_count > 1 and _is_blank(document[0]):
                # The leading spacer sat alone on page 1 because the first real
                # box forces a page break; WeasyPrint would not emit that page.
                document.delete_page(0)
                dropped = page_meta.pop(0)
                top = (where.x0, where.y0, where.x1, where.y0)
                anchors = dict((name, top) for name in dropped["anchors"])
                anchors.update(page_meta[0]["anchors"])
                page_meta[0]["anchors"] = anchors
                page_meta[0]["bookmarks"] = [
                    (level, label, (where.x0, where.y0), state)
                    for level, label, _point, state in dropped["bookmarks"]
                ] + page_meta[0]["bookmarks"]
            _stamp_page_numbers(fitz, document, layout)
            try:
                document.subset_fonts()
            except Exception:
                pass
            source = _Source(document.tobytes(garbage=4, deflate=True))
        finally:
            document.close()

    pages = []
    for index, meta in enumerate(page_meta):
        page = Page(layout.width, layout.height, source, index)
        page.anchors.update(meta["anchors"])
        page.bookmarks.extend(meta["bookmarks"])
        page.links.extend(meta["links"])
        pages.append(page)
    return Document(pages, _Metadata(title))


def _link_for(fitz, href, rect, anchor_targets):
    if href.startswith("#"):
        target = anchor_targets.get(unquote(href[1:]))
        if target is None:
            return None
        page_index, anchor_rect = target
        return {
            "kind": fitz.LINK_GOTO,
            "from": fitz.Rect(rect),
            "page": page_index,
            "to": fitz.Point(anchor_rect[0], anchor_rect[1]),
            "zoom": 0,
        }
    scheme = urlsplit(href).scheme.lower()
    if scheme in _LINK_SCHEMES:
        return {"kind": fitz.LINK_URI, "from": fitz.Rect(rect), "uri": href}
    return None


def _assemble(pages, options, title):
    fitz = _fitz()
    output = fitz.open()
    opened = {}
    try:
        run = None  # [source, first_index, last_index]
        runs = []
        for page in pages:
            if run is not None and run[0] is page._source and page._index == run[2] + 1:
                run[2] = page._index
                continue
            run = [page._source, page._index, page._index]
            runs.append(run)
        for source, first, last in runs:
            document = opened.get(id(source))
            if document is None:
                document = fitz.open("pdf", source.data)
                opened[id(source)] = document
            output.insert_pdf(document, from_page=first, to_page=last, links=False, annots=False)

        anchor_targets = {}
        for page_index, page in enumerate(pages):
            for name, rect in page.anchors.items():
                anchor_targets.setdefault(name, (page_index, rect))

        for page_index, page in enumerate(pages):
            pdf_page = output[page_index]
            for rect, href in page.links:
                link = _link_for(fitz, href, rect, anchor_targets)
                if link is not None:
                    pdf_page.insert_link(link)

        toc = []
        previous_level = 0
        for page_index, page in enumerate(pages):
            for bookmark in page.bookmarks:
                level, label, point = bookmark[0], bookmark[1], bookmark[2]
                label = str(label or "").strip()
                if not label:
                    continue
                level = max(1, min(int(level), previous_level + 1))
                x, y = (tuple(point) + (0.0, 0.0))[:2]
                toc.append([level, label, page_index + 1, {
                    "kind": fitz.LINK_GOTO,
                    "page": page_index,
                    "to": fitz.Point(x, y),
                    "zoom": 0,
                }])
                previous_level = level
        if toc:
            output.set_toc(toc)
        if title:
            output.set_metadata({"title": title})
        try:
            output.subset_fonts()
        except Exception:
            pass
        return output.tobytes(garbage=4, deflate=not options.get("uncompressed_pdf"))
    finally:
        output.close()
        for document in opened.values():
            document.close()
