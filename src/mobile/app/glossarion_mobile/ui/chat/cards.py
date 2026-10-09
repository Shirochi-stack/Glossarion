"""Glossary approval card, JobCard and RequestSheet (UI_SPEC §2.11, §2.12, §5.4).

``GlossaryApprovalCard`` answers the running job's blocking glossary question
(desktop ``_await_direct_text_glossary_approval``): ✏️ Edit opens the generated
glossary in a full-screen raw editor (atomic save that keeps the UTF-8 BOM, like
``_edit_direct_text_generated_glossary``), ✓ Yes continues, ■ No rejects and stops.

``JobCard`` is the assistant-position card of an attachment turn:
Plan -> Queued -> Running -> Result. Running shows the ProgressWatcher progress
("Chapter 12/48 · 3 in flight · ETA …"), the current request and the turn's request
cards (``Requests (N)``, open while it runs: the cards the run already committed - the
glossary gate's - then its live ones); every list shows its newest rows (the chat's
rendered-card limit) and "↑ Show … earlier requests" pages back. Result groups the
persisted request cards, the desktop "Extraction report", the output chips of the turn's
own workspace (compiled EPUB / PDF,
``*_translated.txt``, subtitles, SDLXLIFF, glossary; ``set_outputs``) and the "Attachment
actions" (Read · Share/Export · Compile ▾ EPUB / PDF · QA scan · Open output · Retry failed ·
Open in Library: a finished book's workspace moves into the Library by itself, so the card links
to its Library book); an action that cannot run is shown disabled with a ReasonChip, nothing
hidden (``set_action_reason`` sets a reason the chat works out on the io pool).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.services.jobs import GLOSSARY_QUESTION_KINDS, is_glossary_question
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.chat_ops import OUTPUT_KINDS
from glossarion_mobile.ui.chat.direct_text_rules import (
    attachment_icon,
    attachment_kind_label,
    display_markdown,
    format_attachment_size,
)
from glossarion_mobile.ui.chat.job_binding import CardPhase
from glossarion_mobile.ui.chat.stream_bridge import segment_processing_label
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, semantic

__all__ = [
    "ATTACHMENT_ACTIONS",
    "ATTACHMENT_ACTION_REASONS",
    "EARLIER_REQUESTS_TEMPLATE",
    "GLOSSARY_QUESTION_KINDS",
    "LIBRARY_WAIT_REASON",
    "GlossaryApprovalCard",
    "GlossaryEditorView",
    "GlossaryReviewSheet",
    "JobCard",
    "NO_GLOSSARY_FILE_TEXT",
    "RequestSheet",
    "glossary_preview",
    "glossary_question",
    "material_surface",
    "save_text_keep_bom",
]

NO_GLOSSARY_FILE_TEXT = "No editable glossary file was found. You can continue or stop this run."
_MUTED = ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)

#: (id, label, icon, milestone when it ships or None)
ATTACHMENT_ACTIONS = (
    ("read", "Read", "AUTO_STORIES", None),
    ("export", "Share / Export", "IOS_SHARE", None),
    ("compile", "Compile ▾", "MENU_BOOK", None),
    ("qa", "QA scan", "FACT_CHECK", None),
    ("open_output", "Open output", "FOLDER_OPEN", None),
    ("progress", "Progress", "TIMELINE", None),
    ("retry", "Retry failed", "REPLAY", None),
    ("library", "Open in Library", "LOCAL_LIBRARY", None),
)
#: "Open in Library" while the turn's workspace is still in the chat's Attachments folder: the
#: chat moves a finished book into the Library by itself (no Migrate step on mobile).
LIBRARY_WAIT_REASON = "Added to the Library when the translation finishes"
#: Attachment actions that start disabled for a reason other than a later milestone (the chat view
#: replaces a default through ``JobCard.set_action_reason`` once it knows better).
ATTACHMENT_ACTION_REASONS = {
    "library": LIBRARY_WAIT_REASON,
}


# ---------------------------------------------------------------------------
# Glossary file helpers (blocking: call off the UI loop)
# ---------------------------------------------------------------------------


def glossary_preview(path: str, limit: int = 5) -> dict:
    """``{exists, name, entries, preview: [(raw, translated)], text, bom}`` for the approval card."""
    info = {"exists": False, "name": os.path.basename(str(path or "")), "entries": 0, "preview": [], "text": "", "bom": False}
    if not path or not os.path.isfile(path):
        return info
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return info
    info["exists"] = True
    info["bom"] = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig", errors="replace")
    info["text"] = text
    preview: list = []
    entries = 0
    ext = os.path.splitext(path)[1].lower()
    try:
        # The shared glossary parser (desktop editor filters / footnotes): plain CSV, JSON and the
        # token-efficient "Glossary Columns: ... / === CHARACTERS === / * raw = translated" format
        # the glossary extractor writes by default.
        from glossary_usage import parse_glossary_content

        items = parse_glossary_content(text, ".json" if ext == ".json" else ".csv")
        entries = len(items)
        preview = [(str(item.get("raw_name") or ""), str(item.get("translated_name") or "")) for item in items[:limit]]
    except Exception:
        lines = [line for line in text.splitlines() if line.strip()]
        entries = len(lines)
        preview = [(line[:80], "") for line in lines[:limit]]
    info["entries"] = entries
    info["preview"] = preview
    return info


def save_text_keep_bom(path: str, text: str, has_bom: bool) -> None:
    """Atomic replace that keeps a UTF-8 BOM (desktop generated-glossary editor save)."""
    temp_path = f"{path}.direct-text-edit-{os.getpid()}.tmp"
    try:
        encoding = "utf-8-sig" if has_bom else "utf-8"
        with open(temp_path, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Glossary approval
# ---------------------------------------------------------------------------


class GlossaryApprovalCard(ft.Container):
    def __init__(
        self,
        *,
        path: str,
        info: Optional[dict] = None,
        on_answer: Optional[Callable[[bool], Any]] = None,
        on_edit: Optional[Callable[[str], Any]] = None,
        on_always: Optional[Callable[[], Any]] = None,
        dark: bool = False,
        key: Any = "approval-card",
    ) -> None:
        super().__init__(key=key)
        self.path = str(path or "")
        self.info = info or {"exists": bool(self.path and os.path.isfile(self.path)), "name": os.path.basename(self.path),
                             "entries": 0, "preview": []}
        self.on_answer = on_answer
        self.on_edit = on_edit
        # "Always accept" (chat cards only): remember the choice (the chat's Prefs key), then answer Yes;
        # the owner of ``on_always`` does both. The Library review gate's sheet has no such button.
        self.on_always = on_always
        self.answered: Optional[bool] = None
        self.always_accepted = False
        success = semantic("success", dark)
        exists = bool(self.info.get("exists"))
        body: list[ft.Control] = [
            ft.Text("GLOSSARION · ACTION REQUIRED", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.TERTIARY),
            ft.Text("Glossary generation complete", theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
            ft.Text("Accept this glossary and start translation?", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
        ]
        if exists:
            entries = int(self.info.get("entries") or 0)
            body.append(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.DESCRIPTION, size=18),
                        ft.Text(f"{self.info.get('name')} · {entries:,} entries", theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                                expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                    ],
                    spacing=6,
                )
            )
            for raw, translated in list(self.info.get("preview") or [])[:5]:
                body.append(
                    ft.Text(f"{raw} → {translated}" if translated else raw, theme_style=ft.TextThemeStyle.BODY_SMALL,
                            max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
                )
        else:
            body.append(ft.Text(NO_GLOSSARY_FILE_TEXT, theme_style=ft.TextThemeStyle.BODY_SMALL, color=_MUTED))
        self.edit_button = ft.FilledTonalButton(content="✏️ Edit", disabled=not exists, on_click=self._edit)
        self.yes_button = ft.FilledButton(
            content="✓ Yes", on_click=lambda e: self.answer(True),
            style=ft.ButtonStyle(bgcolor=success, color=ft.Colors.WHITE),
        )
        self.no_button = ft.OutlinedButton(
            content="■ No", on_click=lambda e: self.answer(False),
            style=ft.ButtonStyle(color=ft.Colors.ERROR, side=ft.BorderSide(width=1, color=ft.Colors.ERROR)),
        )
        body.append(ft.Row([self.edit_button, self.yes_button, self.no_button], wrap=True, spacing=8, run_spacing=8))
        self.always_button = ft.TextButton(
            content="Always accept", icon=ft.Icons.DONE_ALL, on_click=lambda e: self.always(),
            tooltip="Accept this glossary and every generated glossary from now on (Chat settings to change)",
            visible=on_always is not None, key="approval-always",
        )
        body.append(self.always_button)
        self.content = ft.Column(body, spacing=6, tight=True)
        self.bgcolor = ft.Colors.TERTIARY_CONTAINER
        self.border_radius = tokens.RADII["plan_card"]
        self.padding = ft.Padding.all(14)

    def _disable(self) -> None:
        for button in (self.edit_button, self.yes_button, self.no_button, self.always_button):
            button.disabled = True
        try:
            self.update()
        except Exception:
            pass

    def answer(self, accepted: bool) -> bool:
        if self.answered is not None:
            return False
        self.answered = bool(accepted)
        self._disable()
        if self.on_answer is not None:
            self.on_answer(bool(accepted))
        return True

    def always(self) -> bool:
        """"Always accept": ``on_always`` stores the choice and answers Yes (one answer, never two)."""
        if self.answered is not None or self.on_always is None:
            return False
        self.answered = True
        self.always_accepted = True
        self._disable()
        self.on_always()
        return True

    def _edit(self, e: Any = None) -> None:
        if self.on_edit is not None and self.path:
            self.on_edit(self.path)


def glossary_question(snapshot: Any) -> Optional[dict]:
    """The pending glossary-approval question of a job snapshot (``{id, kind, data}``), else None.

    Blocking job questions the approval card answers (the chat run's and the Library review gate's):
    ``services.jobs.is_glossary_question`` (``GLOSSARY_QUESTION_KINDS``), the one predicate the job
    service, the chat runs and the notifications use."""
    question = getattr(snapshot, "question", None) if snapshot is not None else None
    if not isinstance(question, dict) or not is_glossary_question(question.get("kind")):
        return None
    return question


class GlossaryReviewSheet:
    """A job's glossary question answered outside a chat (UI_SPEC §3.10 / Appendix C "Book-origin glossary
    gate": the Book page, Jobs › job): the approval card's content (✏️ Edit · ✓ Yes · ■ No) in a sheet.
    ``on_answer(accepted)`` answers the job (``JobService.answer``); closing the sheet leaves it waiting."""

    def __init__(self, *, path: str, info: Optional[dict] = None, on_answer: Callable[[bool], Any],
                 on_edit: Optional[Callable[[str], Any]] = None, title: str = "") -> None:
        from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame

        self.on_answer = on_answer
        self.answered: Optional[bool] = None
        self._page: Any = None
        self.card = GlossaryApprovalCard(path=path, info=info, on_answer=self._answered, on_edit=on_edit,
                                         key="review-approval-card")
        controls: list[ft.Control] = []
        if title:
            controls.append(ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS, key="review-approval-title"))
        controls.append(self.card)
        self.dialog = bottom_sheet(sheet_frame(scroll_column(controls)), key="review-approval-sheet")

    def _answered(self, accepted: bool) -> None:
        self.answered = bool(accepted)
        self.close()
        self.on_answer(bool(accepted))

    def show(self, page: Any) -> "GlossaryReviewSheet":
        self._page = page
        if page is not None:
            page.show_dialog(self.dialog)
        return self

    def close(self) -> None:
        from glossarion_mobile.ui.components.dialogs import close_dialog

        close_dialog(self._page, self.dialog)


class GlossaryEditorView:
    """Full-screen "Edit Generated Glossary — <file>" (Raw mode). ``on_table`` (U6) opens the file in
    the Glossary Manager's table editor instead (the chat's GlossaryFeature hook)."""

    def __init__(
        self,
        path: str,
        text: str,
        *,
        has_bom: bool,
        on_saved: Optional[Callable[[str], Any]] = None,
        on_close: Optional[Callable[[], Any]] = None,
        save: Callable[[str, str, bool], Any] = save_text_keep_bom,
        mono: str = "monospace",
        on_table: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.path = path
        self.has_bom = has_bom
        self.on_table = on_table
        self.on_saved = on_saved
        self.on_close = on_close
        self._save = save
        self.saved = False
        self.error_text = ft.Text("", color=ft.Colors.ERROR, visible=False)
        self.editor = ft.TextField(
            value=text,
            multiline=True,
            min_lines=20,
            expand=True,
            text_style=ft.TextStyle(font_family=mono, size=13),
            border=ft.OutlineInputBorder(),
        )
        self.view = ft.View(
            route="/chat/glossary-edit",
            appbar=ft.AppBar(
                title=ft.Text(f"Edit Generated Glossary — {os.path.basename(path)}", max_lines=1,
                              overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel", on_click=lambda e: self.close(),
                                      size_constraints=HIT_TARGET),
                actions=[ft.TextButton(content="Save", on_click=lambda e: self.save())],
            ),
            controls=[
                ft.SafeArea(
                    expand=True,
                    content=ft.Column(
                        [
                            ft.Text(path, theme_style=ft.TextThemeStyle.LABEL_SMALL, selectable=True, color=_MUTED),
                            ft.Row([ft.TextButton(content="Open in table editor", icon=ft.Icons.TABLE_ROWS,
                                                  on_click=lambda e: self.open_table(), key="gloss-edit-table")],
                                   visible=on_table is not None),
                            self.editor,
                            self.error_text,
                        ],
                        expand=True,
                        spacing=8,
                    ),
                )
            ],
            padding=12,
        )

    def open_table(self) -> None:
        """Leave the raw editor for the Glossary Manager's table editor on the same file."""
        if self.on_table is None:
            return
        self.close()
        self.on_table()

    def save(self) -> bool:
        try:
            self._save(self.path, self.editor.value or "", self.has_bom)
        except Exception as exc:
            self.error_text.value = f"Could not save the generated glossary:\n\n{exc}"
            self.error_text.visible = True
            try:
                self.error_text.update()
            except Exception:
                pass
            return False
        self.saved = True
        if self.on_saved is not None:
            self.on_saved(self.path)
        self.close()
        return True

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()


# ---------------------------------------------------------------------------
# Request sheet
# ---------------------------------------------------------------------------


class RequestSheet:
    """Full streaming content + thinking of one request (UI_SPEC §2.12.3)."""

    def __init__(self, segment: dict) -> None:
        self.segment = dict(segment)
        content = display_markdown(str(segment.get("content", "") or "")) or "*No translated output was emitted for this request.*"
        thinking = str(segment.get("thinking", "") or "")
        controls: list[ft.Control] = [
            ft.Text(str(segment.get("label", "") or "Request"), theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
            ft.Text(segment_processing_label(segment), theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED),
        ]
        if thinking.strip():
            controls.append(
                ft.ExpansionTile(
                    title="Thinking",
                    controls=[ft.Container(content=ft.Markdown(thinking[-50000:], selectable=True), padding=8,
                                           bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, border_radius=8)],
                )
            )
        controls.append(ft.Markdown(content, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB))
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=24),
                                 content=ft.Column(controls, tight=True, spacing=8, scroll=ft.ScrollMode.AUTO)),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def show(self, page: Any) -> None:
        page.show_dialog(self.dialog)


# ---------------------------------------------------------------------------
# JobCard
# ---------------------------------------------------------------------------


#: the JobCard's page of older request rows (the chat's rendered-card limit; UI_SPEC §2.12.3)
EARLIER_REQUESTS_TEMPLATE = "↑ Show {n} earlier requests ({hidden} hidden)"


def _row_signature(segment: dict) -> tuple:
    """What a request row shows (an unchanged segment keeps its row control across live repaints)."""
    content = str(segment.get("content", "") or "")
    return (str(segment.get("label") or ""), str(segment.get("phase") or ""), bool(segment.get("complete")),
            len(content), content[:400], len(str(segment.get("thinking", "") or "")),
            int(segment.get("thinking_tokens") or 0), int(segment.get("text_tokens") or 0),
            segment.get("index"), str(segment.get("status_label") or ""))


def _request_row(segment: dict, on_open: Optional[Callable[[dict], Any]]) -> ft.Control:
    phase = str(segment.get("phase") or "processing")
    phase_label = {"thinking": "Thinking", "text": "Generating"}.get(phase, "Processing")
    if segment.get("complete"):
        phase_label = "Done"
    preview = str(segment.get("content", "") or "").strip().replace("\n", " ")
    tokens_text = f"T {int(segment.get('thinking_tokens') or 0):,} · O {int(segment.get('text_tokens') or 0):,}"
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(str(segment.get("label") or "Request"), theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                                expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Container(content=ft.Text(phase_label, theme_style=ft.TextThemeStyle.LABEL_SMALL),
                                     bgcolor=ft.Colors.SECONDARY_CONTAINER, border_radius=6,
                                     padding=ft.Padding.symmetric(horizontal=6)),
                        ft.Text(tokens_text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED),
                    ],
                    spacing=6,
                ),
                ft.Text(preview[:400], theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=3,
                        overflow=ft.TextOverflow.ELLIPSIS, visible=bool(preview)),
            ],
            spacing=2,
            tight=True,
        ),
        padding=ft.Padding.symmetric(horizontal=8, vertical=6),
        border_radius=8,
        on_click=(lambda e, s=segment: on_open(s)) if on_open else None,
    )


def material_surface(content: ft.Control, *, bgcolor: Any, radius: float, padding: Any) -> ft.Card:
    """A card's coloured, rounded surface painted by a Material (a flat Flutter ``Card``), not by a decorated
    ``Container``. A ``ListTile`` (every ``ExpansionTile`` header: Requests, Run options, the report) paints
    its background and ink on the nearest Material; a coloured DecoratedBox in between hides them, and
    Flutter reports "ListTile background color or ink splashes may be invisible", which fails the Android UI
    tests (Build Mobile run 37940790686). Same look: no elevation, no margin, the radius clips like the
    Container's did."""
    return ft.Card(
        content=ft.Container(content=content, padding=padding),
        bgcolor=bgcolor,
        elevation=0,
        margin=ft.Margin.all(0),
        shape=ft.RoundedRectangleBorder(radius=radius),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )


class JobCard(ft.Container):
    """Plan / Queued / Running / Result card for one attachment turn."""

    def __init__(
        self,
        *,
        attachment: Optional[dict] = None,
        phase: CardPhase = CardPhase("done"),
        on_action: Optional[Callable[[str], Any]] = None,
        on_open_request: Optional[Callable[[dict], Any]] = None,
        on_open_output: Optional[Callable[[str, str], Any]] = None,
        key: Any = None,
        dark: bool = False,
        icon: Optional[str] = None,
        meta: Optional[str] = None,
        row_page: int = 20,
        requests_expanded: bool = False,
    ) -> None:
        super().__init__(key=key)
        self.attachment = dict(attachment or {})
        # Requests: the newest ``row_page`` rows of the turn first, a page more per "Show earlier"
        self.row_page = max(1, int(row_page or 20))
        self.shown_rows = self.row_page
        self.request_segments: list = []  # every request of the turn (``set_requests``)
        # The running card: the turn's committed request cards (the glossary gate's, an earlier run's)
        # listed before the run's live ones (``ChatView._update_live_job_card``)
        self.saved_requests: list = []
        self._row_cache: dict = {}
        self.phase = phase
        self.on_action = on_action
        self.on_open_request = on_open_request
        self.on_open_output = on_open_output
        self.outputs: list = []
        self.dark = dark
        self.status = ""
        # action id -> disabled reason (None: enabled) set by the chat (``set_action_reason``); it wins
        # over the static ATTACHMENT_ACTION_REASONS default
        self.action_reasons: dict = {}
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_SMALL, expand=True, max_lines=2,
                                  overflow=ft.TextOverflow.ELLIPSIS)
        self.state_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.ring = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)
        self.progress = ft.ProgressBar(value=None, visible=False)
        self.line_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, visible=False)
        self.current_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False, max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS)
        # U9 (UI_SPEC §2.12.3 / §2.12.4): the running card's issue chip ("Rate limited · retrying in 30 s",
        # "Key cooling", "Waiting for network…") and the Result's "N QA failed" chip (-> Progress)
        self.issue_chip = ft.Chip(label=ft.Text(""), leading=ft.Icon(ft.Icons.HOURGLASS_TOP, size=16),
                                  visible=False, key="job-issue")
        self.failed_chip = ft.Chip(label=ft.Text(""), leading=ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ERROR, size=16),
                                   visible=False, key="job-qa-failed",
                                   on_click=lambda e: self.on_action("progress") if self.on_action else None)
        self.requests_column = ft.Column([], spacing=2, tight=True)
        self.earlier_requests_button = ft.TextButton(content="", visible=False,
                                                     on_click=lambda e: self.show_earlier_requests())
        # the running card opens its list (like Jobs › job while the job is active), a Result's stays shut
        self.requests_tile = ft.ExpansionTile(title="Requests (0)",
                                              controls=[self.earlier_requests_button, self.requests_column],
                                              visible=False, maintain_state=True, dense=True,
                                              expanded=bool(requests_expanded), on_change=self._on_requests_toggle)
        self.report_md = ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
        self.report_tile = ft.ExpansionTile(title="Extraction report", controls=[ft.Container(content=self.report_md, padding=8)],
                                            visible=False, dense=True)
        self.ocr_column = ft.Column([], spacing=6, tight=True)
        self.ocr_tile = ft.ExpansionTile(title="OCR (0)", controls=[ft.Container(content=self.ocr_column, padding=8)],
                                         visible=False, dense=True, key="job-ocr")
        self.plan_box = ft.Column([], spacing=8, tight=True, visible=False)
        # UI_SPEC §2.12.4 Result: the turn's output files (tap: open / share)
        self.outputs_row = ft.Row([], wrap=True, spacing=6, run_spacing=4, visible=False, key="job-outputs")
        self.buttons = ft.Row([], wrap=True, spacing=8, run_spacing=4)
        self.action_buttons: dict = {}
        # ``icon`` / ``meta``: a tool job's card (a QA scan) names its own icon and summary line
        icon_name = icon or attachment_icon(str(self.attachment.get("extension") or ""))
        ext = str(self.attachment.get("extension") or "")
        if meta is None:
            meta = f"{attachment_kind_label(ext)} · {format_attachment_size(self.attachment.get('size'))}" if self.attachment else ""
        self.body = ft.Column(
            [
                ft.Row(
                    [ft.Icon(icon_data(icon_name), color=ft.Colors.PRIMARY), self.title_text, self.ring],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(meta, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, visible=bool(meta)),
                self.state_text,
                self.progress,
                self.line_text,
                self.current_text,
                self.issue_chip,
                self.failed_chip,
                self.plan_box,
                self.requests_tile,
                self.ocr_tile,
                self.report_tile,
                self.outputs_row,
                self.buttons,
            ],
            spacing=6,
            tight=True,
        )
        # the ExpansionTiles' headers need a Material right above them (``material_surface``)
        self.content = material_surface(self.body, bgcolor=ft.Colors.SURFACE_CONTAINER,
                                        radius=tokens.RADII["plan_card"], padding=ft.Padding.all(12))
        self.title_text.value = str(self.attachment.get("name") or "Attachment")
        self.set_phase(phase)

    # ---- state ----------------------------------------------------------------------------

    def _button(self, action_id: str, label: str, kind: str = "tonal", disabled_reason: Optional[str] = None) -> ft.Control:
        handler = (lambda e, a=action_id: self.on_action(a)) if self.on_action else None
        if disabled_reason:
            button: ft.Control = ft.Row(
                [ft.OutlinedButton(content=label, disabled=True), ReasonChip(reason=disabled_reason)], spacing=4, tight=True
            )
        elif kind == "filled":
            button = ft.FilledButton(content=label, on_click=handler)
        elif kind == "text":
            button = ft.TextButton(content=label, on_click=handler)
        elif kind == "error":
            button = ft.FilledButton(content=label, on_click=handler,
                                     style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR))
        else:
            button = ft.FilledTonalButton(content=label, on_click=handler)
        self.action_buttons[action_id] = button
        return button

    def set_action_reason(self, action_id: str, reason: Optional[str]) -> None:
        """Enable a Result action (``reason`` None) or disable it with a ReasonChip; the buttons are
        rebuilt for the current phase."""
        if action_id in self.action_reasons and self.action_reasons[action_id] == reason:
            return
        self.action_reasons[action_id] = reason
        self.set_phase(self.phase, status=self.status)

    def set_phase(self, phase: CardPhase, *, status: str = "") -> None:
        self.phase = phase
        self.status = status
        name = phase.name
        live = phase.live
        self.ring.visible = live
        self.progress.visible = name in ("running", "stopping", "force_stopping")
        self.plan_box.visible = name == "plan"
        if hasattr(self, "outputs_row"):
            self.outputs_row.visible = bool(self.outputs) and not live and name not in ("plan", "queued")
        self.action_buttons = {}
        buttons: list[ft.Control] = []
        if name == "plan":
            self.state_text.value = "Ready to translate"
            buttons = [self._button("start", "Start", "filled"), self._button("cancel_plan", "Cancel", "text")]
        elif name == "queued":
            self.state_text.value = status or "Queued · starts after the current job"
            buttons = [self._button("cancel_queued", "Cancel", "text")]
        elif name in ("running", "stopping", "force_stopping"):
            self.state_text.value = status or {"stopping": "Stopping after current request…",
                                               "force_stopping": "Force stopping…"}.get(name, "Translating")
            buttons = [
                self._button("stop", "Stop", "error") if name == "running" else self._button("force_stop", "Force stop", "error"),
                self._button("open_reader", "Open reader"),
                self._button("progress", "Progress", "text"),
                self._button("log", "Log", "text"),
            ]
        else:
            self.state_text.value = status or {"done": "Done", "stopped": "Stopped", "failed": "Failed",
                                               "interrupted": "Interrupted"}.get(name, "Done")
            for action_id, label, _icon, milestone in ATTACHMENT_ACTIONS:
                if action_id == "retry" and name == "done" and "failed" not in (status or ""):
                    continue
                if action_id in self.action_reasons:
                    reason = self.action_reasons[action_id]
                else:
                    reason = ATTACHMENT_ACTION_REASONS.get(action_id) or (f"Arrives in {milestone}" if milestone else None)
                buttons.append(self._button(action_id, label, disabled_reason=reason))
            if name in ("stopped", "interrupted"):
                buttons.insert(0, self._button("resume", "Resume", "filled"))
        self.buttons.controls = buttons

    def set_issue(self, label: str) -> None:
        """The running card's issue chip (empty: hidden)."""
        self.issue_chip.label = ft.Text(label)
        self.issue_chip.visible = bool(label) and self.phase.live

    def set_failed(self, count: int) -> None:
        """Result: "N QA failed" (failed + QA-failed chapters) opens the Progress manager's Chapters."""
        count = max(0, int(count or 0))
        self.failed_chip.label = ft.Text(f"{count} QA failed")
        self.failed_chip.visible = count > 0 and not self.phase.live and self.phase.name != "plan"

    def set_progress(self, fraction: Optional[float], line: str = "", current: str = "") -> None:
        self.progress.value = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        self.line_text.value = line
        self.line_text.visible = bool(line)
        self.current_text.value = current
        self.current_text.visible = bool(current)

    def set_requests(self, segments: Sequence[dict]) -> None:
        """Every request of the turn; the newest ``shown_rows`` are rows ("Requests (N)" counts them all,
        "↑ Show … earlier requests" pages back). A row whose segment did not change stays the same
        control, so a live repaint re-sends only the requests that moved."""
        self.request_segments = list(segments)
        total = len(self.request_segments)
        shown = self.request_segments[-self.shown_rows:] if total > self.shown_rows else self.request_segments
        cache: dict = {}
        rows: list = []
        for segment in shown:
            signature = _row_signature(segment)
            row = self._row_cache.get(signature) if signature not in cache else None
            if row is None:
                row = _request_row(segment, self.on_open_request)
            cache.setdefault(signature, row)
            rows.append(row)
        self._row_cache = cache
        self.requests_column.controls = rows
        hidden = total - len(shown)
        self.earlier_requests_button.content = EARLIER_REQUESTS_TEMPLATE.format(n=min(hidden, self.row_page),
                                                                                hidden=hidden)
        self.earlier_requests_button.visible = hidden > 0
        self.requests_tile.title = f"Requests ({total})"
        self.requests_tile.visible = bool(total)

    def show_earlier_requests(self) -> None:
        """"↑ Show … earlier requests": one page more of the older rows."""
        self.shown_rows += self.row_page
        self.set_requests(self.request_segments)
        self.push()

    def reveal_request(self, index: int) -> bool:
        """Jump-to / search: open the Requests list with the row of message ``index`` shown (paging back
        to it); False when the card has no such row."""
        for position, segment in enumerate(self.request_segments):
            if segment.get("index") == index:
                needed = len(self.request_segments) - position
                if needed > self.shown_rows:
                    self.shown_rows = -(-needed // self.row_page) * self.row_page
                    self.set_requests(self.request_segments)
                self.requests_tile.expanded = True
                return True
        return False

    def _on_requests_toggle(self, e: Any) -> None:
        value = getattr(e, "data", None)
        if isinstance(value, str):
            value = value.strip().lower() == "true"
        if isinstance(value, bool):
            self.requests_tile.expanded = value  # the next reveal / repaint starts from what the user sees

    def set_ocr(self, entries: Sequence[tuple]) -> None:
        """Vision: the run's cached OCR text per image (UI_SPEC §2.6 "a collapsible OCR section")."""
        self.ocr_column.controls = [
            ft.Column([ft.Text(str(name), theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED),
                       ft.Text(str(text)[:4000], theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True)],
                      spacing=2, tight=True)
            for name, text in entries
        ]
        self.ocr_tile.title = f"OCR ({len(entries)})"
        self.ocr_tile.visible = bool(entries)

    def set_outputs(self, outputs: Sequence[tuple]) -> None:
        """Result: one chip per output file ``(path, kind)`` (``chat_ops.workspace_outputs``)."""
        self.outputs = [(str(path), str(kind)) for path, kind in outputs or ()]
        chips = []
        for index, (path, kind) in enumerate(self.outputs):
            label, icon = OUTPUT_KINDS.get(kind, (kind.upper() or "File", "INSERT_DRIVE_FILE"))
            name = os.path.basename(path)
            chips.append(ft.Chip(
                label=ft.Text(f"{label} · {name}", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.Icon(icon_data(icon), size=16), tooltip=name,
                on_click=(lambda e, p=path, k=kind: self.on_open_output(p, k)) if self.on_open_output else None,
                key=f"job-output-{index}"))
        self.outputs_row.controls = chips
        self.outputs_row.visible = bool(chips) and not self.phase.live and self.phase.name not in ("plan", "queued")

    def set_report(self, markdown: str) -> None:
        self.report_md.value = display_markdown(markdown)
        self.report_tile.visible = bool(str(markdown or "").strip())

    def set_plan(self, controls: Sequence[ft.Control], start_menu: Sequence[tuple] = ()) -> None:
        """The Plan body; ``start_menu`` [(action id, label)] adds the Start ▾ split (UI_SPEC §2.12.1
        "Run as async batch")."""
        self.plan_box.controls = list(controls)
        if start_menu and self.phase.name == "plan":
            menu = ft.PopupMenuButton(
                icon=ft.Icons.ARROW_DROP_DOWN, tooltip="More ways to start", key="plan-start-menu",
                items=[ft.PopupMenuItem(content=label, on_click=(lambda e, a=action_id: self.on_action(a))
                                        if self.on_action else None)
                       for action_id, label in start_menu],
            )
            controls_row = list(self.buttons.controls)
            controls_row.insert(1 if controls_row else 0, menu)
            self.buttons.controls = controls_row
            self.action_buttons["start_menu"] = menu

    def push(self) -> None:
        try:
            self.update()
        except Exception:
            pass
