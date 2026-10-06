"""QA report viewer (``/tools/qa/report/<rid>``, UI_SPEC §4.4, §5.5 HtmlView).

``rid`` is the opaque ``Prefs.file_ref`` of a ``validation_results.html`` (routes never carry
paths). The header shows the report's summary (``validation_results.json``: files scanned,
files with issues, clean files) with **Open in Chapters** - the book's Chapters tab with the
QA-failed filter (``/library/book/<bid>?tab=chapters&filter=failed``) - and Share.

* Android / iOS: the scanner's own HTML report in ``flet_webview.WebView``, served by an
  in-app ``ReaderServer`` (loopback, token path, nonce CSP: only the viewer's script runs).
  Every file link becomes an "open" event (console ``GLQA:`` + the server's fetch fallback,
  deduplicated); each file row also gets an "Open in Chapters" link. When the page never
  reports in (no "ready" event within ``WEB_READY_GRACE`` seconds, or ``WEB_ERROR_GRACE``
  after a WebView resource error: cleartext to 127.0.0.1 blocked, a broken system WebView)
  the screen switches to the native view, like the Reader.
* Windows / Linux dev (no WebView): the same report rendered natively from the JSON (files
  with issues, their issues and preview, per-file Open in Chapters) - or html2text Markdown
  when the JSON is missing - plus "Open in browser".
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.tools import qa_model as qm
from glossarion_mobile.ui.tools.common import card, hint_text

__all__ = ["QaReportScreen"]

log = logging.getLogger("glossarion.tools.qa")

MAX_NATIVE_ROWS = 400
WEB_READY_GRACE = 8.0  # seconds the served page has to post its "ready" event
WEB_ERROR_GRACE = 4.0  # seconds after a WebView resource error (the Reader's grace)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _markdown(html_text: str) -> str:
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_images = True
        return converter.handle(html_text)
    except Exception:
        import re

        return re.sub(r"<[^>]+>", " ", html_text)


class QaReportScreen(Screen):
    title = "QA report"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        rid = match.params.get("rid", "") if match is not None else ""
        prefs = ctx.prefs
        path = None
        if prefs is not None and rid:
            try:
                path = prefs.resolve_file_ref(rid)
            except Exception:
                path = None
        self.path: str = str(path or "")
        self.folder: str = qm.report_folder(self.path) if self.path else ""
        self.summary: Optional[qm.ReportSummary] = None
        self.renderer: str = ""
        self.server: Any = None
        self.webview: Any = None
        self.seen_seqs: set = set()
        self.events: list = []
        self.opened: list = []
        self.raw: str = ""
        self.page_ready = False
        self.disposed = False
        self._web_check: Any = None

    # ---- layout -------------------------------------------------------------------------------

    def actions(self) -> list:
        return [ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip="Share report", on_click=self._on_share,
                              key="qa-report-share")]

    def build_body(self) -> ft.Control:
        if not self.path or not os.path.isfile(self.path):
            return EmptyState(icon="DESCRIPTION", title="Report not found",
                              body="This QA report no longer exists. Run the scan again.",
                              primary=("QA Scanner", lambda e: self.ctx.go("tools.qa")), key="qa-report-missing")
        self.headline = ft.Text("Loading…", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="qa-report-headline")
        self.issue_chips = ft.Row(wrap=True, spacing=6, run_spacing=6, key="qa-report-issues")
        self.chapters_button = ft.FilledTonalButton(content="Open in Chapters", icon=ft.Icons.FILTER_ALT,
                                                    on_click=lambda e: self.open_in_chapters(), key="qa-report-chapters")
        self.browser_button = ft.TextButton(content="Open in browser", icon=ft.Icons.OPEN_IN_NEW,
                                            on_click=self._on_browser, visible=False, key="qa-report-browser")
        header = card(os.path.basename(self.folder.rstrip("/\\")) or "QA report",
                      [self.headline, self.issue_chips, ft.Row([self.chapters_button, self.browser_button], wrap=True)],
                      icon="FACT_CHECK", key="qa-report-header")
        self.body_slot = ft.Container(expand=True, key="qa-report-body",
                                      content=ft.ProgressRing(width=24, height=24))
        return ft.Column([ft.Container(content=header, padding=ft.Padding.symmetric(horizontal=12, vertical=8)),
                          self.body_slot], expand=True, spacing=0)

    def did_show(self) -> None:
        if self.path and os.path.isfile(self.path):
            self.ctx.spawn(self.load())

    def dispose(self) -> None:
        self.disposed = True
        check, self._web_check = self._web_check, None
        if check is not None and not check.done():
            check.cancel()
        server, self.server = self.server, None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

    # ---- loading -----------------------------------------------------------------------------------

    async def load(self) -> str:
        path = self.path
        self.summary, raw = await self.ctx.io(lambda: (qm.load_report_summary(path), _read_text(path)))
        self.raw = raw
        self._render_summary()
        use_webview = False
        try:
            use_webview = bool(self.ctx.webview_ok())
        except Exception:
            use_webview = False
        if use_webview:
            try:
                self._show_webview(raw)
                return self.renderer
            except Exception:
                log.exception("the QA report WebView failed; using the native view")
        self._show_native(raw)
        return self.renderer

    def _render_summary(self) -> None:
        summary = self.summary
        if summary is None:
            self.headline.value = "Translation QA Report"
            self.issue_chips.controls = []
        else:
            self.headline.value = (f"Total Files Scanned: {summary.total} · Files with Issues: {summary.with_issues}"
                                   f" · Clean Files: {summary.clean}")
            self.issue_chips.controls = [
                ft.Chip(label=ft.Text(f"{kind}: {count}"), key=f"qa-issue-{kind}")
                for kind, count in list(summary.issue_counts.items())[:24]
            ]
        self.ctx.push(self.headline, self.issue_chips)

    def _show_webview(self, raw: str) -> None:
        import flet_webview as fwv

        from glossarion_mobile.services.reader_server import ReaderServer

        server = ReaderServer(on_event=self._on_server_event)
        server.start()
        self.server = server
        nonce = secrets.token_urlsafe(16)
        page_html = qm.annotate_report_html(raw, nonce=nonce, event_path=server.event_path)
        url = server.publish(page_html, name="report.html", script_nonce=nonce)
        # The page lives on http://127.0.0.1 (the in-app server); every other navigation is blocked
        # (file links are turned into events by the page script).
        self.webview = fwv.WebView(
            url=url, expand=True, on_console_message=self._on_console, on_web_resource_error=self._on_web_error,
            prevent_links=["https:", "intent:", "javascript:", "file:", "content:", "mailto:", "tel:"],
        )
        self.body_slot.content = self.webview
        self.renderer = "webview"
        self.ctx.push(self.body_slot)
        self._schedule_web_check(WEB_READY_GRACE)

    # ---- WebView health (the Reader's fallback rule) ---------------------------------------------

    def _heard_from_page(self) -> bool:
        server = self.server
        return self.page_ready or bool(self.events) or (server is not None and server.events_received > 0)

    def _on_web_error(self, e: Any) -> None:
        """A resource failed: when the page itself does not report in soon, use the native view."""
        log.warning("QA report WebView resource error: %s", getattr(e, "data", e))
        if self.renderer != "webview" or self._heard_from_page():
            return
        self._schedule_web_check(WEB_ERROR_GRACE)

    def _schedule_web_check(self, grace: float) -> None:
        previous = self._web_check
        if previous is not None and not previous.done():
            previous.cancel()
        self._web_check = self.ctx.spawn(self.check_webview(grace))

    async def check_webview(self, grace: float = 0.0) -> bool:
        """After ``grace`` seconds: switch to the native view when the page never reported in.
        True when it switched."""
        try:
            if grace:
                await asyncio.sleep(grace)
        except asyncio.CancelledError:
            return False
        if self.disposed or self.renderer != "webview" or self._heard_from_page():
            return False
        log.warning("the QA report page never reported in; using the native view")
        server, self.server = self.server, None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass
        self.webview = None
        self._show_native(self.raw)
        self.ctx.say("The report page view is unavailable here; showing the summary")
        return True

    def _show_native(self, raw: str) -> None:
        summary = self.summary
        rows: list[ft.Control] = []
        if summary is not None:
            if not summary.rows:
                rows.append(hint_text("No issues found.", key="qa-report-clean"))
            for row in summary.rows[:MAX_NATIVE_ROWS]:
                issues = [ft.Text(f"• {issue}", theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True)
                          for issue in row.issues]
                if row.preview:
                    issues.append(ft.Text(row.preview, theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True,
                                          color=ft.Colors.ON_SURFACE_VARIANT, max_lines=4))
                issues.append(ft.TextButton(content="Open in Chapters", icon=ft.Icons.FILTER_ALT,
                                            on_click=lambda e, f=row.filename: self.open_in_chapters(f),
                                            key=f"qa-row-open-{row.filename}"))
                confidence = f" · {int(row.confidence * 100)}%" if row.confidence else ""
                rows.append(ft.ExpansionTile(
                    title=ft.Text(row.filename, max_lines=2),
                    subtitle=ft.Text(f"#{row.index} · score {row.score} · {len(row.issues)} issue"
                                     f"{'s' if len(row.issues) != 1 else ''}{confidence}",
                                     theme_style=ft.TextThemeStyle.BODY_SMALL),
                    controls=issues, controls_padding=ft.Padding.only(left=16, right=12, bottom=8),
                    key=f"qa-row-{row.filename}"))
            if len(summary.rows) > MAX_NATIVE_ROWS:
                rows.append(hint_text(f"… and {len(summary.rows) - MAX_NATIVE_ROWS} more files with issues "
                                      "(open the report in a browser)"))
        else:
            rows.append(ft.Markdown(_markdown(raw), selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                    key="qa-report-markdown"))
        self.body_slot.content = ft.ListView(controls=rows, expand=True, spacing=2,
                                             padding=ft.Padding.symmetric(horizontal=12, vertical=4),
                                             key="qa-report-native")
        self.renderer = "native"
        self.browser_button.visible = self.ctx.open_url is not None
        self.ctx.push(self.body_slot, self.browser_button)

    # ---- events --------------------------------------------------------------------------------------

    def _on_console(self, e: Any) -> None:
        payload = qm.parse_report_event(getattr(e, "message", ""))
        if payload is not None:
            self.handle_event(payload)

    def _on_server_event(self, payload: dict) -> None:
        dispatcher = self.ctx.dispatcher
        event = qm.parse_report_event(payload)
        if event is None:
            return
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(self.handle_event, event)
        else:
            self.handle_event(event)

    def handle_event(self, payload: dict) -> None:
        seq = payload.get("seq")
        if seq is not None:
            if seq in self.seen_seqs:
                return
            self.seen_seqs.add(seq)
        self.events.append(payload)
        if payload.get("type") == "ready":
            self.page_ready = True
        if payload.get("type") == "open":
            self.open_in_chapters(payload.get("file"))

    # ---- actions ------------------------------------------------------------------------------------------

    def open_in_chapters(self, filename: Optional[str] = None) -> Optional[str]:
        """The book's Chapters tab with the QA-failed filter chip selected."""
        bid = self.ctx.bid_for_folder(self.folder)
        if not bid:
            self.ctx.say("This folder is not in the Library")
            return None
        self.opened.append(filename)
        self.ctx.go("library.book", {"bid": bid}, {"tab": "chapters", "filter": qm.QA_FAILED_FILTER})
        return bid

    async def _on_share(self, e: Any = None) -> bool:
        files = self.ctx.files
        if files is None or not self.path:
            self.ctx.say("Sharing is not available in this session")
            return False
        return bool(await files.share([self.path], title="QA report"))

    async def _on_browser(self, e: Any = None) -> Any:
        opener = self.ctx.open_url
        if opener is None or not self.path:
            return None
        from pathlib import Path

        result = opener(Path(self.path).as_uri())
        if hasattr(result, "__await__"):
            result = await result
        return result
