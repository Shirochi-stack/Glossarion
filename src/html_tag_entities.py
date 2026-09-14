"""Helpers for preserving HTML markup and escaping angle-bracket prose."""

from __future__ import annotations

import html
import re


VALID_ENTITY_TAGS = frozenset({
    'html', 'head', 'body', 'title', 'meta', 'link', 'style', 'noscript',
    'p', 'div', 'span', 'br', 'hr', 'img', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'pre', 'code', 'em', 'strong', 'b', 'i', 'u', 's', 'strike', 'del', 'ins', 'mark',
    'small', 'sub', 'sup',
    'table', 'thead', 'tbody', 'tr', 'td', 'th', 'caption', 'col', 'colgroup',
    'blockquote', 'q', 'cite',
    'section', 'article', 'header', 'footer', 'nav', 'main', 'aside', 'details', 'summary',
    'figure', 'figcaption',
    'form', 'input', 'button', 'select', 'option', 'textarea', 'label', 'fieldset', 'legend',
    'iframe', 'canvas', 'svg', 'image', 'math',
    'video', 'audio', 'source', 'track', 'embed', 'object', 'param',
    'map', 'area',
    'ruby', 'rt', 'rp', 'rb', 'rtc',
    'center', 'font', 'base',
})

_LT_ENTITY = r'&(?:lt|LT|#0*60|#[xX]0*3[cC]);'
_GT_ENTITY = r'&(?:gt|GT|#0*62|#[xX]0*3[eE]);'
_DOUBLE_QUOTE_ENTITY = r'&(?:quot|QUOT|#0*34|#[xX]0*22);'
_SINGLE_QUOTE_ENTITY = r'&(?:apos|#0*39|#[xX]0*27);'
_ANGLE_OPEN_RE = re.compile(rf'<|{_LT_ENTITY}')
# Consume whole quoted attribute values before looking for a tag boundary.
# Quotes only become delimiters after '=', so apostrophes in prose such as
# <A hero's journey> cannot swallow the following markup.
_TAG_BOUNDARY_RE = re.compile(
    rf'''(?P<attribute>=\s*(?:"[^"]*"|'[^']*'|'''
    rf'{_DOUBLE_QUOTE_ENTITY}(?:(?!{_DOUBLE_QUOTE_ENTITY}).)*{_DOUBLE_QUOTE_ENTITY}|'
    rf'{_SINGLE_QUOTE_ENTITY}(?:(?!{_SINGLE_QUOTE_ENTITY}).)*{_SINGLE_QUOTE_ENTITY}))'
    rf'|(?P<opening><|{_LT_ENTITY})|(?P<closing>>|{_GT_ENTITY})',
    re.DOTALL,
)
_COMMENT_END_RE = re.compile(rf'--(?P<closing>>|{_GT_ENTITY})')
_STRAY_P_GT_USER_RE = re.compile(
    r"([.,#!$%\^&\*;:{}=\-_`~()?\"'\u2019\u201c\u201d\u00bb\u00ab\]]|<p>)"
    r"[\u200b\s]*p(?:&gt;|>)[\r\n]*",
    re.IGNORECASE,
)
_START_NAME_RE = re.compile(r'^([A-Za-z][A-Za-z0-9:_\-.]*)')
_ATTR_ASSIGN_RE = re.compile(
    r'''(?:^|\s)[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+)'''
)


def looks_like_valid_html_tag(inner: str, valid_tags=None) -> bool:
    """Return True only for real HTML-like markup, not prose in angle brackets.

    ``inner`` is the text between the outer angle brackets.  HTML comments do
    not have a tag name, but they still need to pass through every caller that
    uses this helper to distinguish markup from narrative text.
    """
    if not isinstance(inner, str):
        return False
    valid_tags = VALID_ENTITY_TAGS if valid_tags is None else valid_tags
    stripped = inner.strip()
    if not stripped:
        return False

    # A complete ``<!-- ... -->`` comment appears here without its outer
    # angle brackets: ``!-- ... --``.  Keep the minimum length check so an
    # unterminated ``<!-->`` fragment is not accepted as markup.
    if stripped.startswith('!--'):
        return len(stripped) >= 5 and stripped.endswith('--')

    if stripped.startswith(('!', '?')):
        return False

    closing = stripped.startswith('/')
    if closing:
        tag_bits = stripped[1:].strip()
        match = _START_NAME_RE.match(tag_bits)
        if not match:
            return False
        tag_name = match.group(1).lower()
        remainder = tag_bits[match.end():].strip()
        return tag_name.rsplit(':', 1)[-1] in valid_tags and not remainder

    self_closing = stripped.endswith('/')
    tag_bits = stripped[:-1].rstrip() if self_closing else stripped
    match = _START_NAME_RE.match(tag_bits)
    if not match:
        return False
    tag_name = match.group(1).lower()
    if tag_name.endswith('/'):
        tag_name = tag_name[:-1]
    if tag_name.rsplit(':', 1)[-1] not in valid_tags:
        return False

    remainder = tag_bits[match.end():].strip()
    if not remainder:
        return True
    return bool(_ATTR_ASSIGN_RE.search(remainder))


def _iter_tag_spans(text: str):
    """Yield outer/inner boundaries, ignoring brackets in quoted attributes."""
    position = 0
    while opening := _ANGLE_OPEN_RE.search(text, position):
        start, inner_start = opening.span()
        position = inner_start
        entity_depth = 0
        while True:
            # Comments may contain both angle brackets and quotation marks.
            if text.startswith('!--', inner_start):
                comment_end = _COMMENT_END_RE.search(text, inner_start + 3)
                if comment_end:
                    yield start, inner_start, comment_end.start('closing'), comment_end.end()
                    position = comment_end.end()
                    break

            boundary = _TAG_BOUNDARY_RE.search(text, position)
            if boundary is None:
                return
            position = boundary.end()
            if boundary.lastgroup == 'opening':
                if text[start] == '<' and text[boundary.start()] == '&':
                    # Encoded angle prose inside a raw candidate is text,
                    # e.g. <A hero &lt;Prison Detective&gt; arrives>.
                    entity_depth += 1
                    continue
                # An unquoted nested '<' starts a new candidate rather than
                # letting malformed prose consume the next real element.
                start, inner_start = boundary.span()
                entity_depth = 0
            elif boundary.lastgroup == 'closing':
                if entity_depth and text[boundary.start()] == '&':
                    entity_depth -= 1
                    continue
                yield start, inner_start, boundary.start(), boundary.end()
                break


def unescape_valid_html_tag_entities(text: str) -> str:
    """Rehydrate known tags and complete comments while preserving angle-bracket prose."""
    if not isinstance(text, str) or '&' not in text:
        return text

    pieces = []
    previous = 0
    for start, inner_start, inner_end, end in _iter_tag_spans(text):
        # Raw tags are traversed too, keeping encoded markup inside their
        # attributes opaque (e.g. title="&lt;em&gt;").
        if text[start] != '&' or text[inner_end] != '&':
            continue
        inner = html.unescape(text[inner_start:inner_end])
        if looks_like_valid_html_tag(inner):
            pieces.extend((text[previous:start], f'<{inner}>'))
            previous = end

    pieces.append(text[previous:])
    return ''.join(pieces)


def escape_invalid_html_tags(text: str) -> str:
    """Escape angle-bracket prose without splitting real tags at URL brackets."""
    if not isinstance(text, str) or '<' not in text:
        return text

    pieces = []
    previous = 0
    for start, inner_start, inner_end, end in _iter_tag_spans(text):
        # A nested candidate can leave an unmatched '<' in the preceding
        # prose. Escape it too, so parsing cannot discard that text.
        pieces.append(text[previous:start].replace('<', '&lt;'))
        inner = text[inner_start:inner_end]
        if text[start] == '<' and not looks_like_valid_html_tag(inner):
            inner = inner.replace('<', '&lt;').replace('>', '&gt;')
            pieces.append(f'&lt;{inner}&gt;')
        else:
            pieces.append(text[start:end])
        previous = end

    pieces.append(text[previous:].replace('<', '&lt;'))
    return ''.join(pieces)


def fix_stray_p_gt_artifacts(text: str) -> str:
    """Remove stray paragraph-tag crumbs like p&gt; emitted as visible text."""
    if not isinstance(text, str):
        return text
    lowered = text.lower()
    if 'p' not in lowered or ('&gt;' not in lowered and '&#' not in lowered and '>' not in text):
        return text

    return _STRAY_P_GT_USER_RE.sub(r'\1', text)
