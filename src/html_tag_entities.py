"""Helpers for safely rehydrating escaped HTML tag entities."""

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
    'iframe', 'canvas', 'svg', 'math',
    'video', 'audio', 'source', 'track', 'embed', 'object', 'param',
    'map', 'area',
    'ruby', 'rt', 'rp', 'rb', 'rtc',
    'center', 'font', 'base',
})

_TAG_ENTITY_RE = re.compile(
    r'(&lt;|&LT;|&#0*60;|&#x0*3[cC];)(.*?)(&gt;|&GT;|&#0*62;|&#x0*3[eE];)',
    re.DOTALL,
)
_STRAY_P_GT_USER_RE = re.compile(
    r"([.,#!$%\^&\*;:{}=\-_`~()?\"'\u2019\u201c\u201d\u00bb\u00ab\]]|<p>)"
    r"[\u200b\s]*p(?:&gt;|>)[\r\n]*",
    re.IGNORECASE,
)
_START_NAME_RE = re.compile(r'^([A-Za-z][A-Za-z0-9:_\-.]*)')
_ATTR_ASSIGN_RE = re.compile(
    r'''(?:^|\s)[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+)'''
)


def looks_like_valid_html_tag(inner: str) -> bool:
    """Return True only for real HTML-like tag syntax, not prose in angle brackets."""
    if not isinstance(inner, str):
        return False
    stripped = inner.strip()
    if not stripped or stripped.startswith(('!', '?')):
        return False

    closing = stripped.startswith('/')
    if closing:
        tag_bits = stripped[1:].strip()
        match = _START_NAME_RE.match(tag_bits)
        if not match:
            return False
        tag_name = match.group(1).lower()
        remainder = tag_bits[match.end():].strip()
        return tag_name in VALID_ENTITY_TAGS and not remainder

    self_closing = stripped.endswith('/')
    tag_bits = stripped[:-1].rstrip() if self_closing else stripped
    match = _START_NAME_RE.match(tag_bits)
    if not match:
        return False
    tag_name = match.group(1).lower()
    if tag_name.endswith('/'):
        tag_name = tag_name[:-1]
    if tag_name not in VALID_ENTITY_TAGS:
        return False

    remainder = tag_bits[match.end():].strip()
    if not remainder:
        return True
    return bool(_ATTR_ASSIGN_RE.search(remainder))


def unescape_valid_html_tag_entities(text: str) -> str:
    """Turn encoded known HTML tags into real tags while preserving narrative angle text."""
    if not isinstance(text, str) or '&' not in text:
        return text

    def repl(match: re.Match) -> str:
        inner = html.unescape(match.group(2))
        stripped = inner.strip()
        if not stripped:
            return match.group(0)
        if looks_like_valid_html_tag(stripped):
            return f'<{inner}>'
        return match.group(0)

    return _TAG_ENTITY_RE.sub(repl, text)


def fix_stray_p_gt_artifacts(text: str) -> str:
    """Remove stray paragraph-tag crumbs like p&gt; emitted as visible text."""
    if not isinstance(text, str):
        return text
    lowered = text.lower()
    if 'p' not in lowered or ('&gt;' not in lowered and '&#' not in lowered and '>' not in text):
        return text

    return _STRAY_P_GT_USER_RE.sub(r'\1', text)
