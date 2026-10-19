"""Helpers for removing duplicate heading/paragraph title pairs in HTML, and for collapsing the
runs of empty paragraphs / line breaks some books repeat after every line (chapter text sent to
the model only; the source files are never changed)."""

import os
import re
import unicodedata

from bs4 import Comment, NavigableString, Tag


HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_MEANINGFUL_EMPTY_TAGS = frozenset(
    {
        "audio",
        "canvas",
        "embed",
        "hr",
        "iframe",
        "img",
        "input",
        "math",
        "object",
        "picture",
        "select",
        "svg",
        "textarea",
        "video",
    }
)


def _tag_text(tag):
    return tag.get_text(strip=True).replace("\xa0", " ").strip()


def _comparison_text(tag):
    text = unicodedata.normalize("NFKC", _tag_text(tag))
    text = " ".join(text.split())
    return text.casefold()


def _is_empty_html_tag(node):
    if not isinstance(node, Tag):
        return False
    if _tag_text(node):
        return False
    if node.name in _MEANINGFUL_EMPTY_TAGS:
        return False
    return node.find(_MEANINGFUL_EMPTY_TAGS) is None


def _is_ignorable_sibling(node):
    if isinstance(node, Comment):
        return True
    if isinstance(node, NavigableString):
        return not str(node).replace("\xa0", " ").strip()
    return _is_empty_html_tag(node)


def _first_non_empty_sibling(tag, direction):
    siblings = tag.next_siblings if direction == "next" else tag.previous_siblings
    for sibling in siblings:
        if _is_ignorable_sibling(sibling):
            continue
        return sibling
    return None


def remove_duplicate_heading_paragraph_pairs(
    soup,
    heading_tags=HEADING_TAGS,
    check_next=True,
    check_previous=True,
):
    """Remove duplicate <p> siblings around h1-h6 tags.

    Empty tags between the heading and paragraph are ignored, so markup like
    ``<h1>Title</h1><p> </p><p>Title</p>`` still removes the duplicate
    paragraph.
    """
    removed_any = False

    for heading in soup.find_all(list(heading_tags)):
        heading_id = heading.get("id", "")
        if heading_id and heading_id.startswith("split-"):
            continue

        heading_text = _tag_text(heading)
        if not heading_text or "SPLIT MARKER" in heading_text:
            continue

        if check_next:
            next_sibling = _first_non_empty_sibling(heading, "next")
            if isinstance(next_sibling, Tag) and next_sibling.name == "p":
                if _comparison_text(heading) == _comparison_text(next_sibling):
                    next_sibling.decompose()
                    removed_any = True
                    continue

        if check_previous:
            prev_sibling = _first_non_empty_sibling(heading, "previous")
            if isinstance(prev_sibling, Tag) and prev_sibling.name == "p":
                if _comparison_text(heading) == _comparison_text(prev_sibling):
                    prev_sibling.decompose()
                    removed_any = True

    return removed_any


# --- Excessive empty <p> / <br> runs ---------------------------------------------------------
#: "Remove excessive empty p and br tags (On-device)": on by default, only for requests that go to
#: Glossarion Mobile's on-device model (a ``local/`` model on the published GLOSSARION_LOCAL_LLM_URL).
EMPTY_TAGS_ON_DEVICE_ENV = "REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE"
#: "Remove excessive empty p and br tags (all models)": off by default, every model when on.
EMPTY_TAGS_ALL_MODELS_ENV = "REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS"
#: At most this many line breaks stand for one run of blank lines.
MAX_BLANK_LINE_BREAKS = 2

_BLANK = r"(?:\s|&nbsp;|&#160;|&#x0*a0;)"
_BR = r"(?:<br\b[^>]*>|</br\s*>)"
# A <p> holding only whitespace / &nbsp; and <br> tags (or a self-closing <p/>).
_EMPTY_P = rf"<p\b[^>]*?(?:/>|>(?:{_BLANK}|{_BR})*</p\s*>)"
_RUN_ITEM = rf"(?:{_EMPTY_P}|{_BR})"
_RUN_ITEM_RE = re.compile(_RUN_ITEM, re.IGNORECASE)
_EMPTY_P_RE = re.compile(rf"\A{_EMPTY_P}\Z", re.IGNORECASE)
_BR_RE = re.compile(_BR, re.IGNORECASE)
_RUN_RE = re.compile(rf"{_RUN_ITEM}(?:{_BLANK}*{_RUN_ITEM})*", re.IGNORECASE)


def _collapsed_run(match):
    run = match.group(0)
    breaks = 0
    has_paragraph = False
    for item in _RUN_ITEM_RE.finditer(run):
        text = item.group(0)
        if _EMPTY_P_RE.match(text):
            has_paragraph = True
            # an empty paragraph without a <br> is still one blank line
            breaks += max(1, len(_BR_RE.findall(text)))
        else:
            breaks += 1
    if not has_paragraph and breaks <= MAX_BLANK_LINE_BREAKS:
        return run  # already at most two line breaks: keep the source markup as it is
    return "<br/>" * min(breaks, MAX_BLANK_LINE_BREAKS)


def collapse_empty_paragraphs(html):
    """Collapse runs of empty paragraphs and line breaks for the text sent to the model.

    An empty paragraph is a ``<p>`` whose content is only whitespace, ``&nbsp;`` and ``<br>``
    tags. A run of consecutive empty paragraphs and standalone ``<br>`` tags (whitespace between
    them ignored) becomes its line-break count capped at two (an empty ``<p></p>`` counts as one),
    so ``<p><br/></p><p><br/></p><p><br/></p>`` -> ``<br/><br/>``; ``text<br/><br/><br/>`` inside a
    paragraph keeps two. Paragraphs holding text or any other element (images, spans, ...) are
    never touched, and HTML without such runs comes back unchanged.
    """
    if not isinstance(html, str) or not html or "<" not in html:
        return html
    return _RUN_RE.sub(_collapsed_run, html)


def _env_on(name, default):
    return str(os.environ.get(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def is_on_device_model(model):
    """True when ``model`` is a ``local/`` model served by Glossarion Mobile's on-device engine."""
    if not str(model or "").strip().lower().startswith("local/"):
        return False
    try:
        from unified_api_client import UnifiedClient

        return UnifiedClient._is_on_device_endpoint(UnifiedClient._local_openai_base_url("local"))
    except Exception:
        return False


def should_collapse_empty_paragraphs(model=None):
    """Whether chapter text for ``model`` (default: ``MODEL``) gets ``collapse_empty_paragraphs``."""
    if _env_on(EMPTY_TAGS_ALL_MODELS_ENV, "0"):
        return True
    if not _env_on(EMPTY_TAGS_ON_DEVICE_ENV, "1"):
        return False
    return is_on_device_model(model if model is not None else os.environ.get("MODEL", ""))


def prepare_html_for_model(html, model=None):
    """``html`` as the model should see it: excessive empty paragraphs / line breaks collapsed when
    one of the two toggles applies to ``model``; otherwise unchanged."""
    if not should_collapse_empty_paragraphs(model):
        return html
    return collapse_empty_paragraphs(html)
