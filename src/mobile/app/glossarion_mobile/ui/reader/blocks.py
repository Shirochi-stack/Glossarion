"""Chapter HTML -> native blocks (UI_SPEC §3.11 "Native fallback", LivePanel content).

The native fallback renderer (Windows/Linux dev, where flet-webview has no
platform view; the "Lightweight reader" switch; WebView failures) and the live
"Translate this chapter" panel show chapter HTML with Flet ``Text`` /
``Markdown`` / ``Image`` controls instead of a browser. This module turns HTML
(also a half-received stream: unclosed tags and a cut-off trailing tag are
tolerated) into a flat list of ``Block`` s and into Markdown.

``reader_doc.html_to_blocks`` wins when the shared core provides one
(``html_to_blocks(html)`` below checks); otherwise this presentational
conversion is used. It is display-only: nothing here decides chapter content.

Pure Python (stdlib ``html.parser``), no Flet import; Python 3.10 compatible.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Optional

__all__ = ["Block", "blocks_to_markdown", "html_to_blocks", "html_to_markdown", "plain_text"]

_BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "li", "tr", "figure", "figcaption", "pre",
               "dd", "dt", "td", "th", "header", "footer", "aside", "nav", "main", "body", "center"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_SKIP = {"script", "style", "noscript", "head", "title", "template", "svg"}
_BREAK = {"br"}
_RULE = {"hr"}
_IMAGE = {"img", "image"}
_INLINE_STYLES = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic"}
_RT = {"rt", "rp"}  # ruby annotations: dropped from the reading text (the base text stays)


@dataclass
class Block:
    """One displayable unit: ``kind`` is ``heading`` / ``para`` / ``image`` / ``rule`` / ``quote`` / ``item``."""

    kind: str
    text: str = ""
    level: int = 0  # heading level
    src: str = ""  # image source (as written in the HTML)
    alt: str = ""
    spans: list = field(default_factory=list)  # [(text, {"bold", "italic"})] for rich paragraphs

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "level": self.level, "src": self.src, "alt": self.alt}


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._spans: list[tuple[str, frozenset]] = []
        self._kind = "para"
        self._level = 0
        self._skip = 0
        self._styles: list[str] = []
        self._quote = 0
        self._item = 0

    # ---- helpers ---------------------------------------------------------------------------

    def _flush(self) -> None:
        text = "".join(t for t, _ in self._spans)
        if text.strip():
            spans = _merge_spans([(re.sub(r"[ \t\r\f\v]+", " ", t), s) for t, s in self._spans])
            joined = "".join(t for t, _ in spans)
            lead = len(joined) - len(joined.lstrip())
            if lead and spans:
                first_text, first_style = spans[0]
                spans[0] = (first_text.lstrip(), first_style)
            if spans:
                last_text, last_style = spans[-1]
                spans[-1] = (last_text.rstrip(), last_style)
            clean = "".join(t for t, _ in spans).strip()
            kind = self._kind
            if kind == "para" and self._quote:
                kind = "quote"
            elif kind == "para" and self._item:
                kind = "item"
            self.blocks.append(Block(kind=kind, text=clean, level=self._level,
                                     spans=[s for s in spans if s[0]]))
        self._spans = []
        self._kind = "para"
        self._level = 0

    # ---- parser callbacks -----------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP or tag in _RT:
            self._skip += 1
            return
        if self._skip:
            return
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag in _HEADINGS:
            self._flush()
            self._kind = "heading"
            self._level = _HEADINGS[tag]
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag == "blockquote":
                self._quote += 1
            elif tag == "li":
                self._item += 1
        elif tag in _BREAK:
            self._spans.append(("\n", frozenset(self._styles)))
        elif tag in _RULE:
            self._flush()
            self.blocks.append(Block(kind="rule"))
        elif tag in _IMAGE:
            src = attributes.get("src") or attributes.get("href") or attributes.get("xlink:href") or ""
            if src:
                self._flush()
                self.blocks.append(Block(kind="image", src=src, alt=attributes.get("alt", "")))
        elif tag in _INLINE_STYLES:
            self._styles.append(_INLINE_STYLES[tag])

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in _INLINE_STYLES:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP or tag in _RT:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in _HEADINGS or tag in _BLOCK_TAGS:
            self._flush()
            if tag == "blockquote":
                self._quote = max(0, self._quote - 1)
            elif tag == "li":
                self._item = max(0, self._item - 1)
        elif tag in _INLINE_STYLES:
            style = _INLINE_STYLES[tag]
            for index in range(len(self._styles) - 1, -1, -1):
                if self._styles[index] == style:
                    del self._styles[index]
                    break

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        self._spans.append((data.replace("\n", " "), frozenset(self._styles)))

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._flush()


def _merge_spans(spans: list) -> list:
    out: list = []
    for text, style in spans:
        if not text:
            continue
        if out and out[-1][1] == style:
            out[-1] = (out[-1][0] + text, style)
        else:
            out.append((text, style))
    return out


_TRAILING_TAG = re.compile(r"<[^<>]*$")
_HAS_MARKUP = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")


def _local_html_to_blocks(source: str) -> list[Block]:
    text = str(source or "")
    text = _TRAILING_TAG.sub("", text)  # a tag cut off mid-stream
    if not _HAS_MARKUP.search(text):
        # Plain-text stream: one paragraph per non-empty line (the desktop live view
        # turns newlines into <br> for text without block tags).
        return [Block(kind="para", text=line.strip(), spans=[(line.strip(), frozenset())])
                for line in html_lib.unescape(text).splitlines() if line.strip()]
    parser = _Collector()
    try:
        parser.feed(text)
        parser.close()
    except Exception:  # never let a malformed chapter break the fallback reader
        stripped = re.sub(r"<[^>]+>", " ", text)
        return [Block(kind="para", text=" ".join(html_lib.unescape(stripped).split()))]
    return parser.blocks


def html_to_blocks(source: str, *, shared: Optional[Callable[[str], Any]] = None) -> list[Block]:
    """Blocks of a chapter (``shared`` = ``reader_doc.html_to_blocks`` when the core has it)."""
    if shared is not None:
        try:
            result = shared(source)
        except Exception:
            result = None
        if result:
            blocks: list[Block] = []
            for item in result:
                if isinstance(item, Block):
                    blocks.append(item)
                elif isinstance(item, dict):
                    blocks.append(Block(kind=str(item.get("kind") or "para"), text=str(item.get("text") or ""),
                                        level=int(item.get("level") or 0), src=str(item.get("src") or ""),
                                        alt=str(item.get("alt") or "")))
            if blocks:
                return blocks
    return _local_html_to_blocks(source)


def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_\[\]#<>|])", r"\\\1", text)


def _md_spans(block: Block) -> str:
    if not block.spans:
        return _md_escape(block.text)
    parts = []
    for text, style in block.spans:
        piece = _md_escape(text).replace("\n", "  \n")
        if not piece.strip():
            parts.append(piece)
            continue
        lead = piece[: len(piece) - len(piece.lstrip())]
        trail = piece[len(piece.rstrip()):]
        core = piece.strip()
        if "bold" in style and "italic" in style:
            core = f"***{core}***"
        elif "bold" in style:
            core = f"**{core}**"
        elif "italic" in style:
            core = f"*{core}*"
        parts.append(lead + core + trail)
    return "".join(parts).strip()


def blocks_to_markdown(blocks: list[Block], *, image_text: str = "🖼") -> str:
    out = []
    for block in blocks:
        if block.kind == "heading":
            out.append("#" * max(1, min(6, block.level or 2)) + " " + _md_escape(block.text))
        elif block.kind == "rule":
            out.append("---")
        elif block.kind == "image":
            out.append(f"{image_text} {_md_escape(block.alt)}".strip())
        elif block.kind == "quote":
            out.append("> " + _md_spans(block))
        elif block.kind == "item":
            out.append("- " + _md_spans(block))
        else:
            out.append(_md_spans(block))
    return "\n\n".join(item for item in out if item.strip())


def html_to_markdown(source: str) -> str:
    """Markdown for a (possibly half-received) HTML or plain-text fragment."""
    return blocks_to_markdown(_local_html_to_blocks(source))


def plain_text(source: str) -> str:
    return "\n".join(block.text for block in _local_html_to_blocks(source) if block.text)
