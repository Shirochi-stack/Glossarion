"""Reader page assembly (UI_SPEC §3.11 "Document") over the shared ``reader_doc``, no Flet.

``reader_doc.ReaderDocument`` is the desktop reader's document state without Qt
(``EpubReaderDialog._process_html`` / ``_get_embedded_css`` / ``_wrap_html``
byte-for-byte). ``DocumentBuilder`` keeps one per open source and flavour so its
processed-HTML, image and embedded-CSS caches survive chapter turns, and builds a
page in these steps:

1. the chapter body: ``ReaderSession.chapter_html`` (Original / Translated /
   Bilingual via ``reader_doc.build_bilingual_chapter``), or for **Scroll all**
   ``ReaderDocument.all_chapters_body`` (the desktop LAYOUT_ALL page) with each
   chapter heading tagged ``data-glrdr-ch`` so the page can report the chapter on
   screen;
2. ``process_html``: images are materialised by the shared code and handed to
   ``image_url_for`` (the ``_reader_file_url`` hook), which registers them with the
   ``ReaderServer`` (``/<token>/img/<id>``) instead of ``file://`` URLs;
3. ``wrap(..., mobile=True, event_url=<server event path>, chapter=, initial_page=)``:
   the shared mobile shell and its paging bridge;
4. ``bridge.inject_extras``: the live style (``model.override_css``) and the Reader's
   extras with this page's ``window.__GLRDR_CFG``.

Book content never runs code in the page (U5 review): chapter HTML goes through
``sanitize_book_html`` (no ``<script>`` / frames / objects / ``<base>`` / meta refresh,
no ``on*`` handlers, no ``javascript:`` URLs) before ``process_html``; the book's
embedded CSS cannot close its ``<style>`` element (``inert_css``); Scroll-all chapter
titles are escaped. The page's own scripts (the shared shell, the desktop pager and
the extras) then carry a fresh nonce (``BuiltDocument.nonce``) that the
``ReaderServer`` CSP allows exclusively (``publish(html, script_nonce=)``).

``build_native_blocks`` gives the chapter as blocks for the native fallback
(``reader_doc.html_to_blocks``).
"""

from __future__ import annotations

import hashlib
import html as html_lib
import logging
import os
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.services.library import CoreMissing
from glossarion_mobile.ui.reader import bridge
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.reader.blocks import Block, html_to_blocks
from glossarion_mobile.ui.reader.session import MODE_DUAL, MODE_OVERLAY, MODE_WORKSPACE, ReaderSession

__all__ = ["BuiltDocument", "DocumentBuilder", "ImageRegistrar", "build_native_blocks", "defang_book_scripts",
           "inert_css",
           "sanitize_book_html", "stamp_script_nonce", "tag_chapter_headings"]

log = logging.getLogger("glossarion.reader")


class ImageRegistrar:
    """``image_url_for`` for ``ReaderDocument``: a materialised image file -> a server URL.

    The shared ``_process_html`` writes each chapter image to the reader image cache
    and asks the hook for its URL; bytes and resource dicts are accepted as well.
    """

    def __init__(self, register: Callable[..., str], session: Optional[ReaderSession] = None) -> None:
        self.register = register
        self.session = session
        self.count = 0

    def __call__(self, target: Any = None, *args: Any, **kwargs: Any) -> str:
        self.count += 1
        if isinstance(target, (bytes, bytearray)):
            data = bytes(target)
            return self.register(hashlib.sha1(data).hexdigest() + _ext(kwargs.get("src") or ""), data)
        if isinstance(target, Mapping):
            resource = dict(target)
            key = str(resource.get("identity") or resource.get("path") or resource.get("member") or id(resource))
            session = self.session
            return self.register(key, lambda r=resource: session._resource_bytes(r) if session else None)
        text = os.fspath(target) if target is not None else ""
        if text.startswith("file://"):
            from urllib.parse import unquote, urlsplit

            text = unquote(urlsplit(text).path)
            if os.name == "nt" and text.startswith("/") and len(text) > 2 and text[2] == ":":
                text = text[1:]
        if text and os.path.isfile(text):
            path = os.path.abspath(text)
            return self.register(path, path)
        session = self.session
        source = session.active_path if session is not None else ""
        return self.register(f"{source}|{text}", lambda s=text: session.image_bytes(s) if session else None)


#: Elements that run code, load other documents or re-point relative URLs.
_ACTIVE_TAGS = ("script", "iframe", "frame", "frameset", "object", "embed", "applet", "base", "portal")
#: Attributes holding a URL a link / load could follow.
_URL_ATTRS = frozenset({"href", "src", "xlink:href", "action", "formaction", "data", "poster", "background",
                        "srcset", "lowsrc", "dynsrc", "ping"})
_BAD_URL = re.compile(r"^(?:javascript|vbscript|livescript):|^data:\s*(?:text/html|application/xhtml)", re.I)
_SUSPECT = re.compile(
    r"<\s*(?:script|iframe|frame|object|embed|applet|base|portal)\b"
    r"|<\s*meta\b[^>]*http-equiv\s*=\s*[\"']?\s*refresh"
    r"|[\s\"'/]on[a-z]+\s*="
    r"|(?:java|vb|live)script\s*:"
    r"|data:\s*(?:text/html|application/xhtml)",
    re.I)
_SCRIPT_OPEN = re.compile(r"<script(?=[\s>])", re.I)
#: ``<style>`` contents: html.parser keeps them as raw text, but inside SVG / MathML (foreign
#: content) a browser parses markup there ("<svg><style><script>"), so any "<" in a style
#: element must be made inert before the page is built.
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)(?:</style\s*>|\Z)", re.I | re.S)
#: A ``<script`` / ``</script`` tag opening left in book-derived markup (raw text the sanitiser
#: does not parse as a tag): defanged so the page-wide nonce stamp can never authorise it.
_BOOK_SCRIPT_TAG = re.compile(r"<(?=/?script[\s/>])", re.I)
_SCRIPT_URL = re.compile(r"(?:java|vb|live)script:|data:(?:text/html|application/xhtml)", re.I)
_CONTROL_SPACE = re.compile(r"[\x00-\x20]+")


def sanitize_book_html(markup: str) -> str:
    """Chapter HTML without anything that runs: ``<script>``, frames, objects, ``<base>``, meta
    refresh, ``on*`` handler attributes and ``javascript:`` / ``data:text/html`` URLs.

    Markup with none of those (nearly every book) is returned unchanged, byte for byte, so the
    shared ``process_html`` cache and output stay those of the desktop reader."""
    text = str(markup or "")
    # Browsers drop tabs / newlines inside URLs ("java\tscript:"), so look at a squeezed copy too.
    if not text or not (_SUSPECT.search(text) or _SCRIPT_URL.search(_CONTROL_SPACE.sub("", text))
                        or _style_has_markup(text)):
        return text
    try:
        from bs4 import BeautifulSoup
    except Exception:  # pragma: no cover - bs4 ships with the reader core
        return html_lib.escape(text)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(list(_ACTIVE_TAGS)):
        tag.decompose()
    for tag in soup.find_all("meta"):
        equiv = next((str(v) for k, v in tag.attrs.items() if str(k).lower() == "http-equiv"), "")
        if equiv.strip().lower() == "refresh":
            tag.decompose()
    for tag in soup.find_all(True):
        for name in list(tag.attrs):
            lower = str(name).lower()
            if lower.startswith("on"):
                del tag.attrs[name]
            elif lower in _URL_ATTRS:
                value = tag.attrs[name]
                value = " ".join(value) if isinstance(value, (list, tuple)) else str(value)
                if _BAD_URL.search(_CONTROL_SPACE.sub("", value)):
                    del tag.attrs[name]
    for tag in soup.find_all("style"):
        css = tag.get_text()
        if "<" in css:
            tag.string = _stylesheet(inert_css(css))
    return str(soup)


def _style_has_markup(text: str) -> bool:
    """True when a ``<style>`` element's raw text holds a ``<`` (markup inside SVG / MathML)."""
    return any("<" in m.group(1) for m in _STYLE_BLOCK.finditer(text))


def _stylesheet(css: str) -> Any:
    """``css`` as bs4 style text (serialised verbatim, not entity-escaped)."""
    try:
        from bs4.element import Stylesheet

        return Stylesheet(css)
    except Exception:  # pragma: no cover - bs4 < 4.10
        return css


def defang_book_scripts(markup: str) -> str:
    """``<script`` / ``</script`` left in book-derived markup -> ``&lt;script``: never a tag."""
    return _BOOK_SCRIPT_TAG.sub("&lt;", markup) if markup and "script" in markup.lower() else markup


def inert_css(css: str) -> str:
    """Book CSS placed inside the page's ``<style>``: ``<`` (CSS escape ``\\3C``) can never
    close the element and start markup of its own."""
    return css.replace("<", "\\3C ") if css and "<" in css else css


def stamp_script_nonce(page: str, nonce: str) -> str:
    """Every ``<script>`` of a page built from sanitised content -> ``<script nonce="...">``."""
    return _SCRIPT_OPEN.sub(f'<script nonce="{nonce}"', page)


def _ext(name: str) -> str:
    ext = os.path.splitext(str(name or ""))[1].lower()
    return ext if 1 < len(ext) <= 6 else ""


def tag_chapter_headings(body: str, theme: Mapping[str, Any]) -> str:
    """Tag the k-th Scroll-all chapter heading (``_all_chapters_html``'s exact markup) ``data-glrdr-ch='k'``."""
    opening = (f"<h2 style='color: {theme.get('heading')}; border-bottom: 1px solid {theme.get('border')}; "
               f"padding-bottom: 6px; margin-top: 30px;'>")
    counter = iter(range(1 << 30))
    return re.sub(re.escape(opening), lambda m: f"<h2 data-glrdr-ch='{next(counter)}' " + m.group(0)[4:], body)


@dataclass
class BuiltDocument:
    html: str
    doc_id: str
    chapter: int
    layout: str
    paged: bool
    spread: int
    flavor: str
    generation: int
    fragment: str  # URL fragment for the start position ("f=0.5" or "")
    has_page_bridge: bool
    nonce: str = ""  # the page's own scripts carry it; ReaderServer.publish(script_nonce=) allows only them


class DocumentBuilder:
    """Builds reader pages for one ``ReaderSession`` (blocking; call on the io pool)."""

    def __init__(self, session: ReaderSession, register_image: Callable[..., str]) -> None:
        self.session = session
        self.registrar = ImageRegistrar(register_image, session)
        self._docs: dict = {}
        self._lock = threading.Lock()

    def _module(self) -> Any:
        module = self.session.engine.core.module("reader_doc")
        if module is None or not hasattr(module, "ReaderDocument"):
            raise CoreMissing("reader_doc.ReaderDocument")
        return module

    def document(self, flavor: str) -> Any:
        """The ``ReaderDocument`` for the active source and ``flavor`` (cached)."""
        session = self.session
        plan = session.plan
        workspace = plan.mode == MODE_WORKSPACE
        css_dirs = list(plan.css_dirs or session.manifest.get("css_dirs") or [])
        key = (os.path.normcase(os.path.abspath(session.active_path or ".")), flavor == rm.ORIGINAL, workspace,
               tuple(css_dirs))
        with self._lock:
            doc = self._docs.get(key)
            if doc is None:
                module = self._module()
                doc = module.ReaderDocument(
                    session.active_path,
                    images=session.images,
                    extra_image_dirs=list(session.extra_image_dirs),
                    config=session.engine.config,
                    translated_overlay=session.overlay if plan.mode == MODE_OVERLAY else None,
                    raw_epub_alt_path=plan.raw_path if plan.mode == MODE_DUAL else "",
                    translated_css_dirs=css_dirs,
                    workspace_mode=workspace,
                    image_url_for=self.registrar,
                    chapter_filenames=list(session.filenames),
                    show_raw=flavor == rm.ORIGINAL,
                )
                # The book's stylesheet sits inside the page's <style>: keep it from closing it.
                embedded = doc._get_embedded_css
                doc._get_embedded_css = lambda embedded=embedded: inert_css(embedded())
                self._docs[key] = doc
        # State that changes while the session is open (the overlay refresh swaps images /
        # image folders; the desktop dialog updates the same attributes on itself).
        doc.set_images(session.images)
        doc._extra_image_dirs = list(session.extra_image_dirs)
        doc._translated_overlay = (session.overlay or None) if plan.mode == MODE_OVERLAY else None
        doc._show_raw = flavor == rm.ORIGINAL
        return doc

    def close(self) -> None:
        with self._lock:
            docs, self._docs = list(self._docs.values()), {}
        for doc in docs:
            try:
                doc.close()
            except Exception:
                pass

    def build(
        self,
        index: int,
        *,
        settings: rm.ReaderSettings,
        layout: str,
        theme: Mapping[str, Any],
        doc_id: str,
        event_url: str,
        hint: Optional[Mapping[str, Any]] = None,
        find: Optional[Mapping[str, Any]] = None,
        anchor: Optional[str] = None,
    ) -> BuiltDocument:
        session = self.session
        flavor = session.flavor
        if session.plan.mode == MODE_WORKSPACE and flavor in (rm.ORIGINAL, rm.BILINGUAL):
            for row in (range(session.count) if layout == rm.LAYOUT_ALL else (index,)):
                session.ensure_workspace_raw(row)
        doc = self.document(flavor)
        doc.set_theme(dict(theme))
        doc._font_family = settings.font_family
        doc._font_size = settings.font_size
        doc._line_spacing = settings.line_spacing
        paged = rm.is_paged(layout)
        spread = rm.spread_for(layout)
        if layout == rm.LAYOUT_ALL:
            chapters = [(html_lib.escape(str(session.chapter_title(i) or ""), quote=False),
                         sanitize_book_html(session.chapter_html(i, flavor))) for i in range(session.count)]
            body = tag_chapter_headings(doc.all_chapters_body(chapters), doc._get_theme())
        else:
            body = doc.process_html(sanitize_book_html(session.chapter_html(index, flavor)))
        # Belt and braces: nothing from the book can carry the nonce stamped below, even if the
        # sanitiser and the browser ever parse a chapter differently.
        body = defang_book_scripts(body)
        initial_page, fragment = bridge.position_fragment(hint)
        page = doc.wrap(body, paged, spread, mobile=True, event_url=event_url,
                        chapter=None if layout == rm.LAYOUT_ALL else index, initial_page=initial_page)
        cfg = bridge.bridge_config(doc=doc_id, layout=layout, tap_zones=settings.tap_zones,
                                   chapter=index if layout == rm.LAYOUT_ALL else None,
                                   scroll_all=layout == rm.LAYOUT_ALL, hint=hint, find=find, anchor=anchor)
        has_shell = bridge.has_shell_bridge(page)
        html = bridge.inject_extras(page, cfg, live_css=rm.override_css(theme, settings, layout=layout))
        # Book scripts are gone (sanitize_book_html, inert_css, defang_book_scripts): every <script>
        # left is the page's own.
        nonce = secrets.token_urlsafe(18)
        html = stamp_script_nonce(html, nonce)
        return BuiltDocument(html=html, doc_id=doc_id, chapter=index, layout=layout, paged=paged, spread=spread,
                             flavor=flavor, generation=session.generation,
                             fragment="" if layout == rm.LAYOUT_ALL else fragment, has_page_bridge=has_shell,
                             nonce=nonce)


def build_native_blocks(session: ReaderSession, index: int) -> list[Block]:
    """The chapter as native blocks (fallback renderer; ``reader_doc.html_to_blocks``)."""
    if session.plan.mode == MODE_WORKSPACE and session.flavor in (rm.ORIGINAL, rm.BILINGUAL):
        session.ensure_workspace_raw(index)
    shared = session.engine.fn("reader_doc", "html_to_blocks")
    return html_to_blocks(session.chapter_html(index), shared=shared)
