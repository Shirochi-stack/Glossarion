"""Shared helper for the 'LLM Token Fix' (Fix Empty Attribute Tags) feature.

Some local LLMs hallucinate tokens as tags whose attributes are all empty,
e.g.::

    <status effects="" high="" temperature="" unconscious=""></status>
    <a a="" and="" be="" cause="" provided...="" the="" thorough=""></a>
    <unique ability=""></unique>

A legitimate HTML tag never has *every* attribute empty, so this is almost
always a tokenizer artifact. This module rewrites those patterns into
visible text::

    &lt;status effects high temperature unconscious&gt;
    &lt;a a and be cause provided... the thorough&gt;
    &lt;unique ability&gt;

so the content survives into the rendered output instead of being silently
dropped by the browser / EPUB reader / markdown post-processor.

The function is intentionally aggressive: it does NOT special-case known
HTML tag names (``<a>``, ``<p>``, ``<i>``, ``<span>``, …). A real ``<a>``
anchor always has at least one non-empty attribute or zero attributes; a
real ``<span>`` never has twelve blank attributes in a row.

This helper is the single source of truth for both:
  * repair (BeautifulSoup / html2text / EPUB converter post-processes)
  * detection (QA scanner).
"""
from __future__ import annotations

import re

# A single empty-attribute token: ``foo=""`` or ``foo = ''``.
# Attribute names allow any char that isn't whitespace, ``=``, ``>`` or ``/``
# so we accept LLM-noise like ``provided...``, ``effects:``, ``data-x``.
_EMPTY_ATTR_TOKEN = r'[^\s=>/]+\s*=\s*(?:""|\'\')'

# Tag name: must start with a letter, then any of ``A-Za-z0-9_:.-``.
# Accepts namespaced / dotted names like ``epub:type`` or ``ns.foo``.
_TAG_NAME = r'[A-Za-z][A-Za-z0-9_:\-.]*'

# Paired form: ``<Tag attr="" attr="">...</Tag>``
EMPTY_ATTR_TAG_PAIR_RE = re.compile(
    r'<(' + _TAG_NAME + r')'
    r'((?:\s+' + _EMPTY_ATTR_TOKEN + r')+)'
    r'\s*>'
    r'(.*?)'
    r'</\1\s*>',
    re.DOTALL,
)

# Self-closing form: ``<Tag attr="" attr="" />``
EMPTY_ATTR_TAG_SELF_RE = re.compile(
    r'<(' + _TAG_NAME + r')'
    r'((?:\s+' + _EMPTY_ATTR_TOKEN + r')+)'
    r'\s*/>',
    re.DOTALL,
)

# Detector-only: matches the *presence* of an empty-attribute tag (paired
# or self-closing). Used by the QA scanner to surface how many showed up.
EMPTY_ATTR_TAG_ANY_RE = re.compile(
    r'<' + _TAG_NAME +
    r'(?:\s+' + _EMPTY_ATTR_TOKEN + r')+'
    r'\s*/?>',
    re.DOTALL,
)

_ATTR_NAME_RE = re.compile(r'\s+([^\s=>/]+)\s*=\s*(?:""|\'\')')

# Standard HTML / SVG / MathML / common legacy tag names. A *single* empty
# attribute on one of these is plausibly intentional (e.g. ``<p class="">``
# as a CSS hook); two or more empty attributes in a row is practically
# always an LLM hallucination regardless of tag name.
_STANDARD_HTML_TAGS = frozenset([
    'html', 'head', 'body', 'title', 'meta', 'link', 'style', 'script',
    'noscript',
    'p', 'div', 'span', 'br', 'hr', 'img', 'a',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'pre', 'code', 'em', 'strong', 'b', 'i', 'u', 's', 'strike', 'del',
    'ins', 'mark', 'small', 'sub', 'sup',
    'table', 'thead', 'tbody', 'tr', 'td', 'th', 'caption', 'col',
    'colgroup',
    'blockquote', 'q', 'cite',
    'section', 'article', 'header', 'footer', 'nav', 'main', 'aside',
    'details', 'summary', 'figure', 'figcaption',
    'form', 'input', 'button', 'select', 'option', 'textarea', 'label',
    'fieldset', 'legend',
    'iframe', 'canvas', 'svg', 'math',
    'video', 'audio', 'source', 'track', 'embed', 'object', 'param',
    'map', 'area', 'center', 'font', 'base',
])


def _attr_names(attrs_block: str) -> list[str]:
    return _ATTR_NAME_RE.findall(attrs_block)


def _should_rewrite(tag: str, names: list[str]) -> bool:
    """Decide whether a matched empty-attribute tag should be rewritten.

    Two-plus empty attributes is always treated as a hallucination; a
    single empty attribute on a standard HTML tag is left alone so that
    legitimate CSS-hook markup like ``<p class="">`` survives.
    """
    if len(names) >= 2:
        return True
    return tag.lower() not in _STANDARD_HTML_TAGS



# ---------------------------------------------------------------------------
# Mangled closing-tag pattern
# ---------------------------------------------------------------------------
# Some LLMs tokenize ``</p>`` poorly, merging the closing tag into the
# previous tag's attribute space:
#
#     <br <="" p=""/>    ← should be  <br></p>
#     <br <="" p="">     ← should be  <br></p>
#
# The ``<=""`` is the remnant of ``</`` and the next attr name (``p``) is
# the closing tag name.  No real HTML attribute is ever named ``<``, so
# this signature is unambiguous.
_MANGLED_CLOSE_RE = re.compile(
    r'<(' + _TAG_NAME + r')'           # (1) opening tag name
    r'([^>]*?)'                         # (2) any real attrs before the mangled part
    r'\s+<=\s*(?:""|\'\')\s+'           # mangled < attr  (the broken "</" )
    r'(' + _TAG_NAME + r')\s*=\s*(?:""|\'\')'  # (3) closing tag name
    r'\s*/?>',                          # close /> or >
    re.DOTALL | re.IGNORECASE,
)


def fix_empty_attr_tags(text: str) -> str:
    """Rewrite empty-attribute tags into visible ``&lt;tag …&gt;`` text.

    Both paired (``<t a=""></t>``) and self-closing (``<t a=""/>``) forms
    are handled, in either double- or single-quoted flavour, with any
    amount of whitespace around ``=``.

    Also reconstructs mangled closing tags where the LLM tokenizer merged
    ``</tag>`` into the previous element's attribute list, e.g.
    ``<br <="" p=""/>`` → ``<br></p>``.

    Leaves unrelated HTML alone.
    """
    if not isinstance(text, str) or not text:
        return text

    # --- Pre-pass: reconstruct mangled closing tags ----------------------
    # Must run BEFORE the normal empty-attr rewrite, otherwise the normal
    # pass would escape the tag to ``&lt;br < p/&gt;`` (garbage text).
    def _repl_mangled_close(m: re.Match) -> str:
        tag = m.group(1)
        real_attrs = (m.group(2) or '').strip()
        close_tag = m.group(3)
        opener = f'<{tag} {real_attrs}>' if real_attrs else f'<{tag}>'
        return f'{opener}</{close_tag}>'

    text = _MANGLED_CLOSE_RE.sub(_repl_mangled_close, text)

    # --- Normal empty-attr rewrite ---------------------------------------
    def _repl_pair(m: re.Match) -> str:
        tag = m.group(1)
        names = _attr_names(m.group(2))
        if not _should_rewrite(tag, names):
            return m.group(0)
        body = m.group(3)
        return f"&lt;{tag} {' '.join(names)}&gt;{body}"

    def _repl_self(m: re.Match) -> str:
        tag = m.group(1)
        names = _attr_names(m.group(2))
        if not _should_rewrite(tag, names):
            return m.group(0)
        return f"&lt;{tag} {' '.join(names)}/&gt;"

    # Paired form first so the inner content of a paired tag isn't partially
    # consumed by the self-closing pattern.  Loop because nested empty-attr
    # tags (e.g. <breeding lv.2=""><monsterization grant="" lv.1="">
    # </monsterization></breeding>) leave inner tags unprocessed after the
    # outer match is rewritten — the body of the outer match contained the
    # inner tag, and a single pass only touches the outermost pair.
    for _ in range(20):  # safety cap to prevent infinite loops
        text, n = EMPTY_ATTR_TAG_PAIR_RE.subn(_repl_pair, text)
        if n == 0:
            break
    text = EMPTY_ATTR_TAG_SELF_RE.sub(_repl_self, text)
    return text


_ATTR_BLOCK_RE = re.compile(
    r'<(' + _TAG_NAME + r')'
    r'((?:\s+' + _EMPTY_ATTR_TOKEN + r')+)'
    r'\s*/?>',
    re.DOTALL,
)


def _iter_rewritable(text: str):
    """Yield every match that :func:`fix_empty_attr_tags` would rewrite.

    Mirrors the rewrite rules: 2+ empty attributes, OR a single empty
    attribute on a non-standard tag.
    """
    for m in _ATTR_BLOCK_RE.finditer(text):
        tag = m.group(1)
        names = _attr_names(m.group(2))
        if _should_rewrite(tag, names):
            yield m


def count_empty_attr_tags(text: str) -> int:
    """Return the number of rewrite-eligible empty-attribute tag occurrences.

    Counts both paired and self-closing forms but skips cases the fixer
    would deliberately preserve (e.g. ``<p class="">`` CSS hooks), so the
    QA scanner report never flags issues the repair won't touch.
    """
    if not isinstance(text, str) or not text:
        return 0
    return sum(1 for _ in _iter_rewritable(text))


def find_empty_attr_tags(text: str, limit: int = 20) -> list[str]:
    """Return up to ``limit`` literal matches of rewrite-eligible tags.

    Useful for the QA report preview so users can see *which* garbage tags
    were found, not just how many. Mirrors :func:`count_empty_attr_tags`.
    """
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    for m in _iter_rewritable(text):
        out.append(m.group(0))
        if len(out) >= limit:
            break
    return out
