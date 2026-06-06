"""Helpers for removing duplicate heading/paragraph title pairs in HTML."""

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
