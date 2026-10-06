"""Glossary approval card, JobCard and RequestSheet (UI_SPEC §2.11, §2.12, §5.4).

``GlossaryApprovalCard`` answers the running job's blocking glossary question
(desktop ``_await_direct_text_glossary_approval``): ✏️ Edit opens the generated
glossary in a full-screen raw editor (atomic save that keeps the UTF-8 BOM, like
``_edit_direct_text_generated_glossary``), ✓ Yes continues, ■ No rejects and stops.

``JobCard`` is the assistant-position card of an attachment turn:
Plan -> Queued -> Running -> Result. Running shows the ProgressWatcher progress
("Chapter 12/48 · 3 in flight · ETA …"), the current request and the live request
cards (``Requests (N)``); Result groups the persisted request cards, the desktop
"Extraction report" and the "Attachment actions" (Read · Share/Export · Compile ·
QA scan · Open output · Retry failed · Migrate; surfaces that ship later are shown
disabled with a ReasonChip, nothing hidden).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
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
    "GlossaryApprovalCard",
    "GlossaryEditorView",
    "JobCard",
    "NO_GLOSSARY_FILE_TEXT",
    "RequestSheet",
    "glossary_preview",
    "save_text_keep_bom",
]

NO_GLOSSARY_FILE_TEXT = "No editable glossary file was found. You can continue or stop this run."
_MUTED = ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)

#: (id, label, icon, milestone when it ships or None)
ATTACHMENT_ACTIONS = (
    ("read", "Read", "AUTO_STORIES", None),
    ("export", "Share / Export", "IOS_SHARE", None),
    ("compile", "Compile", "MENU_BOOK", None),
    ("qa", "QA scan", "FACT_CHECK", None),
    ("open_output", "Open output", "FOLDER_OPEN", None),
    ("retry", "Retry failed", "REPLAY", None),
    ("migrate", "Migrate", "DRIVE_FILE_MOVE", "U7"),
)
#: Attachment actions that stay disabled for a reason other than a later milestone.
ATTACHMENT_ACTION_REASONS = {
    # The desktop QA scan skips Direct Text workspaces ("⏭️ Excluding Direct Text source from QA
    # scan"; qa_scan_runtime.is_direct_text_qa_path), and a chat attachment workspace is one.
    "qa": "Chat workspaces are not QA-scanned (desktop: Direct Text is excluded); "
          "use Save to Library, then Tools › QA Scanner",
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
        dark: bool = False,
        key: Any = "approval-card",
    ) -> None:
        super().__init__(key=key)
        self.path = str(path or "")
        self.info = info or {"exists": bool(self.path and os.path.isfile(self.path)), "name": os.path.basename(self.path),
                             "entries": 0, "preview": []}
        self.on_answer = on_answer
        self.on_edit = on_edit
        self.answered: Optional[bool] = None
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
        self.content = ft.Column(body, spacing=6, tight=True)
        self.bgcolor = ft.Colors.TERTIARY_CONTAINER
        self.border_radius = tokens.RADII["plan_card"]
        self.padding = ft.Padding.all(14)

    def answer(self, accepted: bool) -> bool:
        if self.answered is not None:
            return False
        self.answered = bool(accepted)
        for button in (self.edit_button, self.yes_button, self.no_button):
            button.disabled = True
        try:
            self.update()
        except Exception:
            pass
        if self.on_answer is not None:
            self.on_answer(bool(accepted))
        return True

    def _edit(self, e: Any = None) -> None:
        if self.on_edit is not None and self.path:
            self.on_edit(self.path)


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


class JobCard(ft.Container):
    """Plan / Queued / Running / Result card for one attachment turn."""

    def __init__(
        self,
        *,
        attachment: Optional[dict] = None,
        phase: CardPhase = CardPhase("done"),
        on_action: Optional[Callable[[str], Any]] = None,
        on_open_request: Optional[Callable[[dict], Any]] = None,
        key: Any = None,
        dark: bool = False,
    ) -> None:
        super().__init__(key=key)
        self.attachment = dict(attachment or {})
        self.phase = phase
        self.on_action = on_action
        self.on_open_request = on_open_request
        self.dark = dark
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_SMALL, expand=True, max_lines=2,
                                  overflow=ft.TextOverflow.ELLIPSIS)
        self.state_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.ring = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)
        self.progress = ft.ProgressBar(value=None, visible=False)
        self.line_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, visible=False)
        self.current_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False, max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS)
        self.requests_column = ft.Column([], spacing=2, tight=True)
        self.requests_tile = ft.ExpansionTile(title="Requests (0)", controls=[self.requests_column], visible=False,
                                              maintain_state=True, dense=True)
        self.report_md = ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
        self.report_tile = ft.ExpansionTile(title="Extraction report", controls=[ft.Container(content=self.report_md, padding=8)],
                                            visible=False, dense=True)
        self.plan_box = ft.Column([], spacing=8, tight=True, visible=False)
        self.buttons = ft.Row([], wrap=True, spacing=8, run_spacing=4)
        self.action_buttons: dict = {}
        icon_name = attachment_icon(str(self.attachment.get("extension") or ""))
        ext = str(self.attachment.get("extension") or "")
        meta = f"{attachment_kind_label(ext)} · {format_attachment_size(self.attachment.get('size'))}" if self.attachment else ""
        self.content = ft.Column(
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
                self.plan_box,
                self.requests_tile,
                self.report_tile,
                self.buttons,
            ],
            spacing=6,
            tight=True,
        )
        self.bgcolor = ft.Colors.SURFACE_CONTAINER
        self.border_radius = tokens.RADII["plan_card"]
        self.padding = ft.Padding.all(12)
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

    def set_phase(self, phase: CardPhase, *, status: str = "") -> None:
        self.phase = phase
        name = phase.name
        live = phase.live
        self.ring.visible = live
        self.progress.visible = name in ("running", "stopping", "force_stopping")
        self.plan_box.visible = name == "plan"
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
                self._button("log", "Log", "text"),
            ]
        else:
            self.state_text.value = status or {"done": "Done", "stopped": "Stopped", "failed": "Failed",
                                               "interrupted": "Interrupted"}.get(name, "Done")
            for action_id, label, _icon, milestone in ATTACHMENT_ACTIONS:
                if action_id == "retry" and name == "done" and "failed" not in (status or ""):
                    continue
                reason = ATTACHMENT_ACTION_REASONS.get(action_id) or (f"Arrives in {milestone}" if milestone else None)
                buttons.append(self._button(action_id, label, disabled_reason=reason))
            if name in ("stopped", "interrupted"):
                buttons.insert(0, self._button("resume", "Resume", "filled"))
        self.buttons.controls = buttons

    def set_progress(self, fraction: Optional[float], line: str = "", current: str = "") -> None:
        self.progress.value = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        self.line_text.value = line
        self.line_text.visible = bool(line)
        self.current_text.value = current
        self.current_text.visible = bool(current)

    def set_requests(self, segments: Sequence[dict]) -> None:
        rows = [_request_row(s, self.on_open_request) for s in segments]
        self.requests_column.controls = rows
        self.requests_tile.title = f"Requests ({len(rows)})"
        self.requests_tile.visible = bool(rows)

    def set_report(self, markdown: str) -> None:
        self.report_md.value = display_markdown(markdown)
        self.report_tile.visible = bool(str(markdown or "").strip())

    def set_plan(self, controls: Sequence[ft.Control]) -> None:
        self.plan_box.controls = list(controls)

    def push(self) -> None:
        try:
            self.update()
        except Exception:
            pass
