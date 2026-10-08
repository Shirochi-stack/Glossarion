"""The Reader page bridge (UI_SPEC §3.11 "Document"): Python side + page extras.

The page shell and its touch bridge come from the shared core:
``reader_doc.wrap_reader_html(..., mobile=True)`` injects the viewport / safe-area
CSS and ``window.GLRDR`` (tap zones, swipe paging, ``ready`` / ``page`` / ``edge`` /
``tap`` / ``scale`` / ``link`` / ``selection`` / ``scroll`` events with a sequence
number, ``console.log("GLRDR:" + json)`` plus a same-origin ``fetch`` fallback to the
in-app ``ReaderServer``; commands ``goTo``, ``next``, ``prev``, ``goToFraction``,
``measure``, ``setTransport``, ``post``). The Reader never ships a second paging
bridge. ``READER_EXTRAS_JS`` only adds what the Reader needs on top of it, through
the shell's own ``GLRDR.post`` (so sequence numbers and transport stay shared):

* command aliases the Reader calls (``go``, ``goFraction``, ``goLast``, ``httpOff`` =
  ``setTransport('console')`` once a console event proved that channel works);
* ``find(text, occurrence)`` (search hit highlight + jump), ``anchor(id)``,
  ``applyStyle(css)`` (Aa changes without a reload; the page keeps its proportional
  position), ``clearSelection()``;
* a ``selection`` event with empty text when the selection is cleared (the shell only
  reports non-empty selections) and a ``selrect`` event with the selection's rect, so
  the chip row can sit next to it;
* "tap edges to turn pages" off: side taps become centre taps (capture phase, before
  the shell's handler);
* Scroll all: ``scroll`` events with the chapter on screen (``ch``/``chf``, from the
  ``data-glrdr-ch`` headings) and the restore of a chapter position on load;
* ``img`` on a double-tapped image.

``normalize_event`` accepts the shell's schema (``type``, ``edge: end|start``,
``zone``, ``scale``, ``reason``) and the extras'; ``EventDeduper`` drops the second
copy of an event delivered by both channels and events of a replaced page.

Pure Python; no Flet import.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

__all__ = [
    "CONSOLE_PREFIX",
    "EVENT_TYPES",
    "EventDeduper",
    "READER_EXTRAS_JS",
    "ReaderEvent",
    "bridge_config",
    "has_shell_bridge",
    "inject_extras",
    "js_call",
    "normalize_event",
    "parse_console_message",
    "position_fragment",
    "scale_steps",
]

CONSOLE_PREFIX = "GLRDR:"  # reader_doc.MOBILE_EVENT_PREFIX
EVENT_TYPES = frozenset({"ready", "page", "edge", "tap", "sel", "selrect", "pinch", "scroll", "link", "img", "found",
                         "err"})
_MAX_TEXT = 4000
PINCH_STEP = 0.12  # one font-size point per 12% of pinch scale
MAX_PINCH_STEPS = 6

READER_EXTRAS_JS = r"""
(function () {
  var G = window.GLRDR;
  if (!G || G.__extras) { return; }
  G.__extras = 1;
  var C = window.__GLRDR_CFG || {};
  function post(o) { o.doc = C.doc || ''; G.post(o); }
  function cols() { return document.getElementById('columns'); }
  function pyRound(x) {
    var r = Math.round(x);
    if (Math.abs(x % 1) === 0.5 && r % 2 !== 0) { r -= 1; }
    return r;
  }
  function span(c) {
    var gap = (typeof _PAGE_GAP !== 'undefined') ? _PAGE_GAP : 0;
    var w = (typeof _pageWidthFor === 'function') ? _pageWidthFor(c)
      : Math.max(1, Math.floor(c.clientWidth || window.innerWidth || 1));
    return Math.max(1, w + gap);
  }
  function docHeight() {
    return Math.max(document.documentElement.scrollHeight || 0, document.body ? document.body.scrollHeight : 0);
  }
  G.go = function (p) { G.goTo(p); };
  G.goFraction = function (f) { G.goToFraction(f); };
  G.goLast = function () {
    if (cols()) { G.goTo(1e9); } else { window.scrollTo(0, docHeight()); }
  };
  G.httpOff = function () { G.setTransport('console'); };

  function reveal(el) {
    var c = cols();
    if (c) {
      var r = el.getBoundingClientRect(), cr = c.getBoundingClientRect();
      G.goTo(Math.floor((r.left - cr.left + c.scrollLeft) / span(c)));
    } else {
      try { el.scrollIntoView({block: 'center'}); } catch (e) { el.scrollIntoView(); }
    }
  }
  function clearMarks() {
    var ms = document.querySelectorAll('mark.glrdr-hit');
    for (var i = 0; i < ms.length; i++) {
      var m = ms[i], p = m.parentNode;
      while (m.firstChild) { p.insertBefore(m.firstChild, m); }
      p.removeChild(m);
      try { p.normalize(); } catch (e) {}
    }
  }
  G.find = function (text, occ) {
    clearMarks();
    if (!text) { return; }
    var root = cols() || document.body, low = String(text).toLowerCase(), n = 0, want = occ || 0;
    var walker = document.createTreeWalker(root, 4, null, false), node;
    while ((node = walker.nextNode())) {
      var lv = (node.nodeValue || '').toLowerCase(), i = lv.indexOf(low);
      while (i >= 0) {
        if (n === want) {
          try {
            var range = document.createRange();
            range.setStart(node, i); range.setEnd(node, i + low.length);
            var mark = document.createElement('mark');
            mark.className = 'glrdr-hit';
            range.surroundContents(mark);
            reveal(mark);
            post({type: 'found', ok: true});
          } catch (e) { post({type: 'found', ok: false}); }
          return;
        }
        n++;
        i = lv.indexOf(low, i + Math.max(1, low.length));
      }
    }
    post({type: 'found', ok: false, matches: n});
  };
  G.anchor = function (id) {
    var el = document.getElementById(id) || (document.getElementsByName(id) || [])[0];
    if (el) { reveal(el); }
  };
  G.applyStyle = function (css) {
    var s = document.getElementById('glrdr-live');
    if (!s) {
      s = document.createElement('style');
      s.id = 'glrdr-live';
      (document.head || document.documentElement).appendChild(s);
    }
    s.textContent = css || '';
    setTimeout(function () {
      if (!cols()) { return; }
      var old = G.count(), p = G.page();
      var n = G.measure();
      var t = (old > 0 && p >= old - 1) ? n - 1
        : pyRound((old > 1 ? Math.max(0, Math.min(1, p / (old - 1))) : 0) * (n - 1));
      G.goTo(t);
    }, 30);
  };
  G.clearSelection = function () {
    try { window.getSelection().removeAllRanges(); } catch (e) {}
  };

  function selText() {
    try { return String(window.getSelection ? window.getSelection() : '').trim(); } catch (e) { return ''; }
  }
  function linkOf(el) {
    while (el && el !== document.body && el.nodeType === 1) {
      if ((el.tagName === 'A' || el.tagName === 'a') && el.getAttribute('href')) { return el; }
      el = el.parentNode;
    }
    return null;
  }
  if (C.zones === false) {
    window.addEventListener('click', function (e) {
      if (!cols() || linkOf(e.target) || selText()) { return; }
      var w = window.innerWidth || document.documentElement.clientWidth || 1, x = e.clientX / w;
      if (x < 1 / 3 || x > 2 / 3) {
        e.stopPropagation();
        post({type: 'tap', zone: 'center'});
      }
    }, true);
  }
  var selTimer = 0, hadSel = false;
  document.addEventListener('selectionchange', function () {
    clearTimeout(selTimer);
    selTimer = setTimeout(function () {
      var s = window.getSelection ? window.getSelection() : null, text = selText();
      if (!text) {
        if (hadSel) { hadSel = false; post({type: 'selection', text: ''}); }
        return;
      }
      hadSel = true;
      try {
        if (s && s.rangeCount) {
          var b = s.getRangeAt(0).getBoundingClientRect(), w = window.innerWidth || 1, h = window.innerHeight || 1;
          post({type: 'selrect', x: b.left / w, y: b.top / h, w: b.width / w, h: b.height / h});
        }
      } catch (e) {}
    }, 360);
  });
  function chapterTops() {
    var hs = document.querySelectorAll('[data-glrdr-ch]'), out = [];
    for (var i = 0; i < hs.length; i++) {
      out.push({ch: +hs[i].getAttribute('data-glrdr-ch'),
                top: hs[i].getBoundingClientRect().top + (window.scrollY || 0)});
    }
    return out;
  }
  function chInfo() {
    var tops = chapterTops(), y = (window.scrollY || 0) + 4, ch = C.ch || 0, top = 0, next = docHeight();
    for (var i = 0; i < tops.length; i++) {
      if (tops[i].top <= y) {
        ch = tops[i].ch; top = tops[i].top;
        next = (i + 1 < tops.length) ? tops[i + 1].top : docHeight();
      } else { break; }
    }
    return {ch: ch, f: next > top ? Math.max(0, Math.min(1, (y - top) / (next - top))) : 0};
  }
  if (C.all) {
    var scrollTimer = 0;
    window.addEventListener('scroll', function () {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(function () {
        var max = Math.max(1, docHeight() - window.innerHeight), info = chInfo();
        post({type: 'scroll', fraction: Math.min(1, Math.max(0, (window.scrollY || 0) / max)),
              ch: info.ch, chf: info.f});
      }, 260);
    }, {passive: true});
    var restored = false;
    var restore = function () {
      if (restored) { return; }
      restored = true;
      var tops = chapterTops(), h = C.hint || {}, top = null, next = docHeight();
      for (var i = 0; i < tops.length; i++) {
        if (tops[i].ch === (C.ch || 0)) {
          top = tops[i].top;
          next = (i + 1 < tops.length) ? tops[i + 1].top : docHeight();
          break;
        }
      }
      if (top !== null) {
        var y = top + (h.last ? Math.max(0, next - top - window.innerHeight) : (+h.fraction || 0) * (next - top));
        window.scrollTo(0, Math.max(0, Math.round(y)));
      }
    };
    if (document.readyState === 'complete') { setTimeout(restore, 0); }
    else { window.addEventListener('load', function () { setTimeout(restore, 0); }); }
  }
  document.addEventListener('dblclick', function (e) {
    var el = e.target;
    if (el && (el.tagName === 'IMG' || el.tagName === 'img' || el.tagName === 'image')) {
      post({type: 'img', src: el.getAttribute('src') || el.getAttribute('href') || el.getAttribute('xlink:href') || ''});
    }
  }, true);
  var pending = function () {
    if (C.anchor) { G.anchor(C.anchor); }
    if (C.find && C.find.text) { G.find(C.find.text, C.find.occurrence || 0); }
  };
  if (C.anchor || (C.find && C.find.text)) {
    if (document.readyState === 'complete') { setTimeout(pending, 60); }
    else { window.addEventListener('load', function () { setTimeout(pending, 60); }); }
  }
})();
"""


def bridge_config(
    *,
    doc: str,
    layout: str,
    tap_zones: bool = True,
    chapter: Optional[int] = None,
    scroll_all: bool = False,
    hint: Optional[Mapping[str, Any]] = None,
    find: Optional[Mapping[str, Any]] = None,
    anchor: Optional[str] = None,
) -> dict:
    """``window.__GLRDR_CFG`` for one page (read by ``READER_EXTRAS_JS``)."""
    cfg: dict[str, Any] = {"doc": str(doc), "layout": str(layout), "zones": bool(tap_zones)}
    if chapter is not None:
        cfg["ch"] = int(chapter)
    if scroll_all:
        cfg["all"] = True
        cfg["hint"] = {k: v for k, v in dict(hint or {}).items() if k in ("fraction", "last")}
    if find and find.get("text"):
        cfg["find"] = {"text": str(find["text"])[:500], "occurrence": int(find.get("occurrence") or 0)}
    if anchor:
        cfg["anchor"] = str(anchor)[:200]
    return cfg


def position_fragment(hint: Optional[Mapping[str, Any]]) -> tuple:
    """``(initial_page, url fragment)`` for the shell's start position.

    The shell starts at ``initial_page`` or at ``#f=<fraction>`` (page
    ``round((count - 1) * f)``); "the last page" is a page beyond the end (clamped).
    """
    hint = dict(hint or {})
    if hint.get("last"):
        return 10 ** 6, ""
    fraction = hint.get("fraction")
    if fraction is not None:
        try:
            value = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return None, f"f={value:.6f}".rstrip("0").rstrip(".")
        return 0, ""
    page = hint.get("page")
    try:
        return max(0, int(page)), ""
    except (TypeError, ValueError):
        return 0, ""


def _script_json(data: Any) -> str:
    """JSON safe inside a ``<script>`` element (no ``</script>``, no U+2028/2029 breakage)."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026") \
        .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


_BODY_CLOSE = re.compile(r"</body\s*>", re.I)


def has_shell_bridge(html: str) -> bool:
    """True when the page carries the shared mobile shell (``wrap_reader_html(mobile=True)``)."""
    return "window.GLRDR" in html and CONSOLE_PREFIX in html


def inject_extras(html: str, cfg: Mapping[str, Any], *, live_css: str = "") -> str:
    """Add the live style element, this page's config and the extras before ``</body>``."""
    tail = (f'<style id="glrdr-live">{live_css}</style>'
            f"<script>window.__GLRDR_CFG={_script_json(dict(cfg))};</script>"
            f"<script>{READER_EXTRAS_JS}</script>")
    matches = list(_BODY_CLOSE.finditer(html))
    if matches:
        at = matches[-1].start()
        return html[:at] + tail + html[at:]
    return html + tail


def js_call(function: str, *args: Any) -> str:
    """``window.GLRDR && GLRDR.<function>(<json args>)`` for ``WebView.run_javascript``."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function):
        raise ValueError(f"invalid bridge function {function!r}")
    rendered = ",".join(_script_json(a) for a in args)
    return f"window.GLRDR&&GLRDR.{function}&&GLRDR.{function}({rendered});"


# ---- events -------------------------------------------------------------------------------------


def parse_console_message(message: Any, prefix: str = CONSOLE_PREFIX) -> Optional[dict]:
    """The event dict of a ``GLRDR:`` console line (``None`` for other console output).

    ``prefix`` lets another page protocol share this console channel parser: the
    WebViewBridge's hidden pages answer with ``GLWVB:`` lines (services/webview_bridge.py)."""
    text = str(message or "")
    at = text.find(prefix)
    if at < 0 or at > 16:  # some WebViews prefix the source location
        return None
    try:
        payload = json.loads(text[at + len(prefix):])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _unit(value: Any) -> Optional[float]:
    number = _float(value)
    return None if number is None else max(0.0, min(1.0, number))


def scale_steps(scale: Any) -> int:
    """Font-size steps for a finished pinch (one point per 12%, at most ±6)."""
    value = _float(scale)
    if value is None or value <= 0:
        return 0
    steps = int(round(math.log(value) / math.log(1 + PINCH_STEP)))
    return max(-MAX_PINCH_STEPS, min(MAX_PINCH_STEPS, steps))


@dataclass(frozen=True)
class ReaderEvent:
    type: str
    doc: str = ""
    seq: Optional[int] = None
    page: Optional[int] = None
    count: Optional[int] = None
    fraction: Optional[float] = None
    direction: int = 0
    text: str = ""
    href: str = ""
    external: bool = False
    step: int = 0
    ended: bool = False
    rect: Optional[tuple] = None  # (x, y, w, h) as viewport fractions
    chapter: Optional[int] = None
    chapter_fraction: Optional[float] = None
    zone: str = ""
    ok: Optional[bool] = None
    why: str = ""
    channel: str = ""
    key: str = ""  # de-duplication key (identical for the console and HTTP copies)
    raw: Mapping[str, Any] = field(default_factory=dict)


_TYPE_ALIASES = {
    "selection": "sel", "select": "sel", "scale": "pinch", "zoom": "pinch", "image": "img", "error": "err",
    "boundary": "edge", "loaded": "ready",
}
_EXTERNAL = re.compile(r"^(?:https?:|mailto:|tel:)", re.I)


def normalize_event(payload: Optional[Mapping[str, Any]]) -> Optional[ReaderEvent]:
    """A validated ``ReaderEvent`` from a page payload (shell schema ``type`` or extras ``t``)."""
    if not isinstance(payload, Mapping):
        return None
    kind = str(payload.get("type") or payload.get("t") or "").strip().lower()
    kind = _TYPE_ALIASES.get(kind, kind)
    if kind not in EVENT_TYPES:
        return None
    count = _int(payload.get("count"), None)
    page = _int(payload.get("page"), None)
    if count is not None:
        count = max(1, min(count, 100000))
    if page is not None:
        page = max(0, min(page, (count - 1) if count else 100000))
    edge = str(payload.get("edge") or "").lower()
    direction = _int(payload.get("dir"), 0) or 0
    if edge == "end":
        direction = 1
    elif edge == "start":
        direction = -1
    direction = 1 if direction > 0 else (-1 if direction < 0 else 0)
    rect = None
    if kind == "selrect":
        values = [_unit(payload.get(k)) for k in ("x", "y", "w", "h")]
        if all(v is not None for v in values):
            rect = tuple(values)
    step = 0
    ended = False
    if kind == "pinch":
        if payload.get("scale") is not None:
            step, ended = scale_steps(payload.get("scale")), True
        else:
            step = _int(payload.get("step"), 0) or 0
            step = 1 if step > 0 else (-1 if step < 0 else 0)
            ended = bool(payload.get("end"))
    text = payload.get("text")
    if text is None and kind == "err":
        text = payload.get("msg") or payload.get("message")
    href = str(payload.get("href") or payload.get("src") or "")[:2048]
    external = payload.get("external")
    if external is None and kind == "link":
        external = bool(_EXTERNAL.match(href))
    ok = payload.get("ok")
    chapter = _int(payload.get("ch"), None)
    if chapter is None:
        chapter = _int(payload.get("chapter"), None)
    clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    try:
        key = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        key = repr(sorted(clean.items(), key=lambda kv: str(kv[0])))
    return ReaderEvent(
        type=kind,
        doc=str(payload.get("doc") or ""),
        seq=_int(payload.get("seq"), None),
        page=page,
        count=count,
        fraction=_unit(payload.get("fraction")),
        direction=direction,
        text=str(text or "")[:_MAX_TEXT],
        href=href,
        external=bool(external),
        step=step,
        ended=ended,
        rect=rect,
        chapter=chapter,
        chapter_fraction=_unit(payload.get("chf")),
        zone=str(payload.get("zone") or "")[:16],
        ok=None if ok is None else bool(ok),
        why=str(payload.get("reason") or payload.get("why") or "")[:32],
        channel=str(payload.get("_channel") or "")[:16],
        key=key,
        raw=dict(payload),
    )


class EventDeduper:
    """Accepts each event once, and only for the page on screen.

    Both channels (console and HTTP) may deliver the same event: the copies are
    identical payloads (same sequence number). Events of a replaced page are stale:
    the extras' events carry the page id (``doc``), the shell's carry the chapter
    the page was built for.
    """

    def __init__(self, window: int = 512) -> None:
        self.doc = ""
        self.chapter: Optional[int] = None
        self.window = window
        self._seen: set = set()
        self._order: list = []
        self.dropped = 0

    def set_document(self, doc: str, chapter: Optional[int] = None) -> None:
        self.doc = str(doc)
        self.chapter = chapter
        self._seen.clear()
        self._order.clear()

    def accept(self, event: ReaderEvent) -> bool:
        if self.doc and event.doc and event.doc != self.doc:
            self.dropped += 1
            return False
        if (self.chapter is not None and event.chapter is not None and event.chapter != self.chapter
                and not event.doc):
            self.dropped += 1
            return False
        if event.seq is None:
            return True
        key = (event.seq, event.key)
        if key in self._seen:
            self.dropped += 1
            return False
        self._seen.add(key)
        self._order.append(key)
        if len(self._order) > self.window:
            old = self._order.pop(0)
            self._seen.discard(old)
        return True
