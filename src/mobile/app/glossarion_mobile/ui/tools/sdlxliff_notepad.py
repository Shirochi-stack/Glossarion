"""The SDLXLIFF reviewer's Notepad layout on mobile: the desktop's rendered, live HTML editor.

Desktop (``Retranslation_GUI.SDLXLIFFReviewDialog._add_notepad_document_editor``): one QWebEngineView shows
the piece's whole output chapter (``notepad_document``) with its images; the shared in-page script
(``sdlxliff_review_core.NOTEPAD_EDITING_SCRIPT``) turns every text run into an editable island over an
immutable HTML tree (source text on hover, B / I / U, Enter adds a break or a translator note, purple /
empty markers) and sets ``window.__sdlNotepadDirty`` on every edit; a 350 ms poll captures the DOM and
the core turns it back into a saveable document (``notepad_document_from_browser_html``) that goes
through the normal save path 500 ms later (``edit_document`` + ``flush_edits``).

Mobile does the same in a ``flet_webview.WebView`` (owner 2026-10-30: the reviewer "looks nothing like
the desktop counterpart ... The desktop version has a notepad mode as well, which is not a text editor,
but a real time html editor"):

* the page is published on the Reader's local server (``ReaderServer.publish``, its CSP runs only
  scripts with the page's nonce) with the workspace images served by it
  (``_notepad_renderable_document_html(url_for=register_image)``); the editing script and the capture
  poll are one nonce'd ``<script>`` at the end of the body that removes itself first, so it never
  reaches the captured HTML;
* the poll posts the dirty DOM as ``GLSDL:`` console lines (split into ``{id, part, parts, chunk}``
  like the WebView bridge's answers) and :meth:`NotepadView.on_console` reassembles the newest one;
* 500 ms after the last capture the document is saved on the io pool; leaving the reviewer saves a
  capture that is still waiting (``flush``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any, Callable, Optional

import flet as ft

log = logging.getLogger("glossarion.tools.sdlxliff_notepad")

#: console prefix of the Notepad page's captures
CONSOLE_PREFIX = "GLSDL:"
#: the desktop's capture poll (``_sdl_edit_poll_timer``) and save delay (``_edit_save_timer``)
POLL_MS = 350
SAVE_DELAY = 0.5
#: characters per console line (a long chapter goes out in parts)
CHUNK_CHARS = 8000
MAX_PARTS = 4096
#: published document name on the Reader's server
DOCUMENT_NAME = "sdlxliff-notepad.html"
NO_WEBVIEW = ("The Notepad layout needs the in-app browser (Android / iOS); "
              "the Compact layout edits the rows here")

#: the capture poll: posts the DOM after an edit (``window.__sdlNotepadDirty``, set by the editing script)
CAPTURE_SCRIPT = (
    "(function () {\n"
    "  if (window.__glSdlCapture) return;\n"
    "  var seq = 0, size = %d, prefix = %s;\n"
    "  function send() {\n"
    "    var html = document.documentElement.outerHTML; seq += 1;\n"
    "    var parts = Math.max(1, Math.ceil(html.length / size));\n"
    "    for (var i = 0; i < parts; i++) {\n"
    "      console.log(prefix + JSON.stringify({id: seq, part: i, parts: parts,\n"
    "                                           chunk: html.slice(i * size, (i + 1) * size)}));\n"
    "    }\n"
    "  }\n"
    "  window.__glSdlCapture = send;\n"
    "  setInterval(function () {\n"
    "    if (!window.__sdlNotepadDirty) return;\n"
    "    window.__sdlNotepadDirty = false;\n"
    "    send();\n"
    "  }, %d);\n"
    "})();\n"
) % (CHUNK_CHARS, json.dumps(CONSOLE_PREFIX), POLL_MS)


def page_html(renderable: str, editing_script: str, nonce: str) -> str:
    """The published page: the renderable document plus one nonce'd script (the editor, then the
    capture poll) that removes its own element first."""
    script = ("<script nonce=\"%s\">(function(){var s=document.currentScript;if(s&&s.parentNode)"
              "s.parentNode.removeChild(s);})();\n%s\n%s</script>"
              % (nonce, editing_script.replace("</script", "<\\/script"), CAPTURE_SCRIPT))
    lower = renderable.lower()
    at = lower.rfind("</body>")
    if at < 0:
        return renderable + script
    return renderable[:at] + script + renderable[at:]


class NotepadView:
    """One piece's Notepad editor (a WebView over the Reader's server); ``control`` is what the reviewer
    shows. ``save(piece_index, document, user_added_target_indexes, user_added_break_positions)`` runs on
    the io pool and returns the status text; ``on_saved(status)`` runs on the loop afterwards."""

    def __init__(self, *, core: Any, session: Any, piece_index: int, server: Any,
                 io: Callable[..., Any], spawn: Callable[[Any], Any],
                 save: Callable[..., Any], on_saved: Optional[Callable[[str], Any]] = None,
                 webview_factory: Optional[Callable[..., Any]] = None) -> None:
        self.core = core
        self.session = session
        self.piece_index = piece_index
        self.server = server
        self.io = io
        self.spawn = spawn
        self.save = save
        self.on_saved = on_saved
        self.webview_factory = webview_factory
        self.prefix = ""
        self.url = ""
        self.control: Optional[ft.Control] = None
        self._chunks: dict = {}
        self._latest = 0
        self._pending: Optional[str] = None  # the newest captured DOM not saved yet
        self._save_task: Any = None
        self.saves = 0

    # ---- page --------------------------------------------------------------------------------------

    def build_page(self) -> str:
        """Blocking (io pool): publish the piece's Notepad page; returns its URL."""
        session, core = self.session, self.core
        document = str(session.notepad_document(self.piece_index) or "")
        self.prefix = core.SdlxliffReviewCoreMixin._notepad_document_prefix(document)
        piece = session.pieces[self.piece_index]

        def url_for(path: str) -> str:
            return self.server.register_image(path, path)

        renderable = session._notepad_renderable_document_html(piece, document, url_for=url_for)
        nonce = secrets.token_urlsafe(18)
        self.url = self.server.publish(page_html(renderable, core.NOTEPAD_EDITING_SCRIPT, nonce),
                                       name=DOCUMENT_NAME, script_nonce=nonce)
        return self.url

    def make_control(self) -> ft.Control:
        factory = self.webview_factory
        if factory is None:
            import flet_webview as fwv

            factory = fwv.WebView
        self.control = factory(
            url=self.url, expand=True, key=f"sdl-notepad-{secrets.token_hex(4)}",
            on_console_message=self.on_console,
            # the page is the Reader server's (http://127.0.0.1); a book's own links never navigate it
            prevent_links=["https:", "intent:", "javascript:", "file:", "content:", "mailto:", "tel:", "data:"],
        )
        return self.control

    # ---- captures ----------------------------------------------------------------------------------

    def on_console(self, e: Any) -> None:
        text = str(getattr(e, "message", e) or "")
        at = text.find(CONSOLE_PREFIX)
        if at < 0:
            return
        try:
            payload = json.loads(text[at + len(CONSOLE_PREFIX):])
            seq, part, parts = int(payload["id"]), int(payload["part"]), int(payload["parts"])
            chunk = payload["chunk"]
        except (ValueError, KeyError, TypeError):
            return
        if seq < self._latest or not (1 <= parts <= MAX_PARTS and 0 <= part < parts) or not isinstance(chunk, str):
            return
        if seq > self._latest:
            self._latest, self._chunks = seq, {}
        self._chunks[part] = chunk
        if len(self._chunks) < parts:
            return
        html = "".join(self._chunks[i] for i in range(parts))
        self._chunks = {}
        self.captured(html)

    def captured(self, html: str) -> None:
        """A complete capture: saved ``SAVE_DELAY`` after the last one (the desktop's debounce)."""
        self._pending = html
        task, self._save_task = self._save_task, None
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
        self._save_task = self.spawn(self._save_later())

    async def _save_later(self) -> None:
        try:
            await asyncio.sleep(SAVE_DELAY)
        except asyncio.CancelledError:
            return
        await self.flush()

    async def flush(self) -> str:
        """Save the newest capture now (the reviewer's re-render / leave); '' when nothing waits."""
        html, self._pending = self._pending, None
        if html is None:
            return ""
        document, indexes, breaks = self.core.SdlxliffReviewCoreMixin.notepad_document_from_browser_html(
            html, self.prefix)
        try:
            status = await self.io(self.save, self.piece_index, document, indexes, breaks)
        except Exception as exc:
            log.exception("saving the Notepad document failed")
            status = f"Save failed: {exc}"
        self.saves += 1
        if self.on_saved is not None:
            self.on_saved(str(status or "Saved"))
        return str(status or "Saved")

    @property
    def dirty(self) -> bool:
        return self._pending is not None
