"""Message cards of the transcript (UI_SPEC §2.9, §2.10, §5.4).

* ``UserBubble`` - ``["user", text]``: right-aligned bubble (<= 75% width), collapses
  past 12 lines with "Show more"; long-press -> ActionSheet (Copy · Select text).
* ``UserFileCard`` - ``["user_file", name, path, size, prompt, role]``: document card
  with type icon, file name, "EXT · 1.2 MB" and the role label + instruction.
* ``AssistantMessage`` - ``["assistant", content, thinking, processing_label,
  output_folder, request_label, storage]``: full width, header
  ``GLOSSARION · <request label> · <time>``, the collapsible Thinking disclosure
  (expanded indices persisted in the v2 ``expanded`` set), Markdown content
  (HTML sanitised and converted, long outputs truncated with "Show full
  translation (N chars)") and the action row Copy · Retranslate · ⋯.
  The same control renders a live request card (``set_live(segment)``) while the
  run streams; text is pushed at the coalesced cadence the transcript decides.

Bodies are passed as callables so saved cards read ``Chat Messages/`` files lazily.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.direct_text_rules import (
    attachment_icon,
    attachment_kind_label,
    display_markdown,
    format_attachment_size,
    split_long_output,
    timestamp_label,
)
from glossarion_mobile.ui.chat.stream_bridge import segment_processing_label
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = [
    "AssistantMessage",
    "ROLE_LABELS",
    "THINKING_TAIL_CHARS",
    "UserBubble",
    "UserFileCard",
    "thinking_display",
]

ROLE_LABELS = {"user": "User instruction", "system": "System instruction", "assistant": "Assistant instruction"}
THINKING_TAIL_CHARS = 50000
COLLAPSE_LINES = 12
COPIED_SECONDS = 1.6
_MUTED = ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)


def thinking_display(text: str, live: bool) -> str:
    """The expanded Thinking body: the last 50,000 characters (desktop rule) or the fallback text."""
    value = str(text or "")
    if not value.strip():
        return "Waiting for the thinking stream…" if live else "No thinking stream was emitted for this response."
    if len(value) > THINKING_TAIL_CHARS:
        return "… earlier thinking output omitted …\n\n" + value[-THINKING_TAIL_CHARS:]
    return value


def _bubble_width(available: Optional[float]) -> Optional[float]:
    if not available:
        return None
    return min(640.0, available * 0.75)


class UserBubble(ft.Row):
    def __init__(
        self,
        text: str,
        *,
        index: int = -1,
        available_width: Optional[float] = None,
        on_long_press: Optional[Callable[["UserBubble"], Any]] = None,
        key: Any = None,
    ) -> None:
        super().__init__(alignment=ft.MainAxisAlignment.END, key=key)
        self.text = str(text or "")
        self.index = index
        self.expanded = False
        self.collapsible = self.text.count("\n") + 1 > COLLAPSE_LINES
        self.body = ft.Text(
            self.text,
            theme_style=ft.TextThemeStyle.BODY_LARGE,
            max_lines=COLLAPSE_LINES if self.collapsible else None,
            overflow=ft.TextOverflow.ELLIPSIS if self.collapsible else None,
        )
        self.more = ft.TextButton(content="Show more", visible=self.collapsible, on_click=self._toggle)
        bubble = ft.Container(
            content=ft.Column([self.body, self.more], spacing=2, tight=True),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=ft.BorderRadius.only(top_left=18, top_right=18, bottom_left=18, bottom_right=6),
            padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            width=_bubble_width(available_width),
            on_long_press=(lambda e: on_long_press(self)) if on_long_press else None,
        )
        self.controls = [bubble]
        self.bubble = bubble

    def _toggle(self, e: Any = None) -> None:
        self.expanded = not self.expanded
        self.body.max_lines = None if self.expanded else COLLAPSE_LINES
        self.more.content = "Show less" if self.expanded else "Show more"
        try:
            self.update()
        except Exception:
            pass


class UserFileCard(ft.Row):
    def __init__(
        self,
        name: str,
        path: str,
        size: Any,
        prompt: str = "",
        role: str = "user",
        *,
        index: int = -1,
        missing: bool = False,
        on_tap: Optional[Callable[["UserFileCard"], Any]] = None,
        on_long_press: Optional[Callable[["UserFileCard"], Any]] = None,
        key: Any = None,
    ) -> None:
        import os

        super().__init__(alignment=ft.MainAxisAlignment.END, key=key)
        self.name = str(name or os.path.basename(str(path or "")) or "file")
        self.path = str(path or "")
        self.prompt = str(prompt or "")
        self.role = role if role in ROLE_LABELS else "user"
        self.index = index
        extension = os.path.splitext(self.name)[1].lower()
        self.meta = f"{attachment_kind_label(extension)} · {format_attachment_size(size)}"
        rows: list[ft.Control] = [
            ft.Row(
                [
                    ft.Icon(icon_data(attachment_icon(extension)), size=32, color=ft.Colors.PRIMARY),
                    ft.Column(
                        [
                            ft.Text(self.name, theme_style=ft.TextThemeStyle.TITLE_SMALL, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Text(self.meta + (" · missing" if missing else ""), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                    color=ft.Colors.ERROR if missing else _MUTED),
                        ],
                        spacing=0,
                        tight=True,
                        expand=True,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        ]
        if self.prompt:
            rows.append(ft.Container(height=1, bgcolor=ft.Colors.OUTLINE_VARIANT, margin=ft.Margin.symmetric(vertical=6)))
            rows.append(ft.Text(ROLE_LABELS[self.role], theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED))
            rows.append(ft.Text(self.prompt, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=False))
        card = ft.Container(
            content=ft.Column(rows, spacing=2, tight=True),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=tokens.RADII["user_file_card"],
            padding=ft.Padding.all(12),
            width=320,
            on_click=(lambda e: on_tap(self)) if on_tap else None,
            on_long_press=(lambda e: on_long_press(self)) if on_long_press else None,
        )
        self.controls = [card]
        self.card = card


class AssistantMessage(ft.Column):
    """One assistant card (saved message or live request segment)."""

    def __init__(
        self,
        *,
        index: int = -1,
        request_label: str = "",
        created_at: str = "",
        processing_label: str = "Processing",
        content: Callable[[], str] | str = "",
        thinking: Callable[[], str] | str = "",
        expanded: bool = False,
        live: bool = False,
        actions: bool = True,
        on_toggle_thinking: Optional[Callable[["AssistantMessage", bool], Any]] = None,
        on_copy: Optional[Callable[["AssistantMessage"], Any]] = None,
        on_retranslate: Optional[Callable[["AssistantMessage"], Any]] = None,
        on_more: Optional[Callable[["AssistantMessage"], Any]] = None,
        on_show_full: Optional[Callable[["AssistantMessage"], Any]] = None,
        key: Any = None,
    ) -> None:
        super().__init__(spacing=2, key=key)
        self.index = index
        self.live = live
        self.thinking_expanded = bool(expanded)
        self._content_source = content
        self._thinking_source = thinking
        self.on_toggle_thinking = on_toggle_thinking
        self.on_copy = on_copy
        self.on_show_full = on_show_full
        self.request_label = str(request_label or "")
        self.created_at = str(created_at or "")
        self.header_text = ft.Text(
            self._header(), theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS, expand=True,
        )
        header = ft.Row(
            [ft.CircleAvatar(foreground_image_src=HALGAKOS_ASSET, radius=10), self.header_text],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.thinking_label = ft.Text(
            "", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=_MUTED, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS
        )
        self.thinking_toggle = ft.Container(
            content=self.thinking_label,
            on_click=self._toggle_thinking,
            padding=ft.Padding.symmetric(vertical=6),
            key=f"thinking-{index}",
        )
        self.thinking_body = ft.Container(
            content=ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=ft.Padding.all(10),
            visible=False,
        )
        self.content_md = ft.Markdown(
            "", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB, auto_follow_links=True
        )
        self.pending_text = ft.Text(
            "Working on your translation …", italic=True, theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=_MUTED,
            visible=False,
        )
        self.status_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, visible=False,
                                   max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.full_button = ft.TextButton(content="", visible=False, on_click=lambda e: self._show_full())
        self.copy_button = ft.IconButton(
            icon=ft.Icons.CONTENT_COPY, icon_size=18, tooltip="Copy output", size_constraints=HIT_TARGET,
            on_click=lambda e: on_copy(self) if on_copy else None,
        )
        self.retranslate_button = ft.IconButton(
            icon=ft.Icons.REFRESH, icon_size=18, tooltip="Retranslate", size_constraints=HIT_TARGET,
            on_click=lambda e: on_retranslate(self) if on_retranslate else None,
        )
        self.more_button = ft.IconButton(
            icon=ft.Icons.MORE_HORIZ, icon_size=18, tooltip="More", size_constraints=HIT_TARGET,
            on_click=lambda e: on_more(self) if on_more else None,
        )
        self.actions_row = ft.Row(
            [self.copy_button, self.retranslate_button, self.more_button], spacing=0, visible=actions and not live
        )
        self.controls = [header, self.thinking_toggle, self.thinking_body, self.pending_text, self.status_text,
                         self.content_md, self.full_button, self.actions_row]
        self.processing_label = processing_label
        self.rendered_content = ""
        self.refresh(processing_label=processing_label)

    # ---- data ------------------------------------------------------------------------------

    @staticmethod
    def _read(source: Callable[[], str] | str) -> str:
        if callable(source):
            try:
                return str(source() or "")
            except Exception:
                return "*The saved response file is missing or unreadable.*"
        return str(source or "")

    @property
    def content_text(self) -> str:
        return self._read(self._content_source)

    @property
    def thinking_text(self) -> str:
        return self._read(self._thinking_source)

    def _header(self) -> str:
        parts = ["GLOSSARION"]
        if self.request_label:
            parts.append(self.request_label)
        stamp = timestamp_label(self.created_at)
        if stamp:
            parts.append(stamp)
        return " · ".join(parts)

    # ---- rendering ----------------------------------------------------------------------------

    def refresh(self, *, processing_label: Optional[str] = None) -> None:
        if processing_label is not None:
            self.processing_label = processing_label
        arrow = "▼" if self.thinking_expanded else "▶"
        suffix = "  …" if self.live else ""
        self.thinking_label.value = f"{arrow}  {self.processing_label}{suffix}"
        self.thinking_body.visible = self.thinking_expanded
        if self.thinking_expanded:
            self.thinking_body.content.value = thinking_display(self.thinking_text, self.live)
        content = self.content_text
        preview, truncated = split_long_output(content)
        markdown = display_markdown(preview)
        self.rendered_content = markdown
        self.content_md.value = markdown
        self.content_md.visible = bool(markdown.strip())
        self.pending_text.visible = self.live and not markdown.strip()
        self.full_button.visible = truncated
        if truncated:
            self.full_button.content = f"Show full translation ({len(content):,} chars)"
        self.header_text.value = self._header()

    def set_live(self, segment: dict) -> None:
        """Update from a live request segment (``RunStream.segments()``)."""
        self.live = not bool(segment.get("complete")) or self.live
        self.request_label = str(segment.get("label", "") or self.request_label)
        self.created_at = str(segment.get("created_at", "") or self.created_at)
        self._content_source = str(segment.get("content", "") or "")
        self._thinking_source = str(segment.get("thinking", "") or "")
        status = str(segment.get("status_label", "") or "")
        self.refresh(processing_label=segment_processing_label(segment))
        # the latest pipeline line while nothing has streamed yet (the job log's last line)
        self.status_text.value = status
        self.status_text.visible = bool(status) and self.pending_text.visible

    def finish_live(self) -> None:
        self.live = False
        self.status_text.visible = False
        self.actions_row.visible = True
        self.refresh()

    def _toggle_thinking(self, e: Any = None) -> None:
        self.thinking_expanded = not self.thinking_expanded
        self.refresh()
        if self.on_toggle_thinking is not None:
            self.on_toggle_thinking(self, self.thinking_expanded)
        self._push()

    def _show_full(self) -> None:
        if self.on_show_full is not None:
            self.on_show_full(self)

    def show_copied(self) -> None:
        """Copy -> ✓ for 1.6 s (desktop "✓ Copied")."""
        self.copy_button.icon = ft.Icons.CHECK
        self.copy_button.tooltip = "Copied"
        self._push()

    def reset_copied(self) -> None:
        self.copy_button.icon = ft.Icons.CONTENT_COPY
        self.copy_button.tooltip = "Copy output"
        self._push()

    def _push(self) -> None:
        try:
            self.update()
        except Exception:
            pass
