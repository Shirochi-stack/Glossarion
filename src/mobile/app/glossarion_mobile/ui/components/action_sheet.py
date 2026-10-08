"""ActionSheet (UI_SPEC §5.2): the app's long-press / overflow action menu.

Flet 1.0.3 has no Material action sheet (only ``CupertinoActionSheet``) and no
anchored popover, and ``PopupMenuButton`` cannot be opened from code, so long
action lists are a custom ``BottomSheet(show_drag_handle=True, scrollable=True)``
of ``ListTile`` rows plus a Cancel row. On tablets the same rows sit in a
centred ``AlertDialog`` (<= 560 dp). Tapping a row closes the sheet first,
then runs the action. Destructive rows use the error colour; unavailable rows
stay visible, disabled, with a ReasonChip (nothing hidden, §0 item 6).

The rows scroll when they may not fit (a 14-row chapter menu on a phone, any
menu at large text); a scrolling body makes a Flet 1.0.3 sheet full height, so
a menu that surely fits at 200 % text keeps a compact body (``show``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.sheet import fits_compact, page_width, sheet_frame, text_height
from glossarion_mobile.ui.theme import icon_data

__all__ = ["ActionItem", "ActionSheet"]


@dataclass
class ActionItem:
    label: str
    on_select: Optional[Callable[[], Any]] = None  # sync or async; runs after the sheet closes
    icon: Any = None  # ft.Icons member or token icon name
    destructive: bool = False
    disabled_reason: Optional[str] = None  # shown as a ReasonChip; the row is disabled
    key: Optional[str] = None


class ActionSheet:
    """Builds the sheet (phone) or dialog (tablet); ``show(page)`` opens it."""

    def __init__(
        self,
        items: Sequence[ActionItem],
        *,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        cancel_label: str = "Cancel",
        tablet: bool = False,
        on_cancel: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.items = list(items)
        self.title = title
        self.subtitle = subtitle
        self.tablet = tablet
        # Called once when the sheet closes without a row: the Cancel row, Android back or an
        # outside tap (an awaited choice then never waits forever).
        self.on_cancel = on_cancel
        self._cancelled = False
        self.selected: Optional[ActionItem] = None
        self.explained: Optional[ActionItem] = None  # the last unavailable item tapped (its reason shown)
        self._page: Any = None
        self.tiles: list[ft.ListTile] = [self._tile(item) for item in self.items]
        self.cancel_tile = ft.ListTile(
            title=ft.Text(cancel_label),
            leading=ft.Icon(ft.Icons.CLOSE),
            on_click=self._on_cancel,
            min_height=tokens.SIZES["hit_target"],
            key="cancel",
        )
        header: list[ft.Control] = []
        if title:
            header.append(ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600))
        if subtitle:
            header.append(ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        self.body = body = ft.Column(
            tight=True,
            spacing=0,
            scroll=ft.ScrollMode.AUTO,  # show() keeps a short menu compact
            controls=(
                [ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=8), content=ft.Column(header, tight=True, spacing=2))]
                if header
                else []
            )
            + list(self.tiles)
            + [self.cancel_tile],
        )
        if tablet:
            self.dialog: Any = ft.AlertDialog(
                content=ft.Container(width=tokens.SIZES["dialog_max"], content=body),
                content_padding=ft.Padding.symmetric(vertical=12),
            )
        else:
            self.dialog = ft.BottomSheet(
                content=sheet_frame(body, padding=ft.Padding.only(bottom=8)),
                show_drag_handle=True,
                scrollable=True,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            )
        if on_cancel is not None:
            # set before show_dialog (Flet wraps the handler present when the dialog opens)
            self.dialog.on_dismiss = self._on_dismiss

    def _tile(self, item: ActionItem) -> ft.ListTile:
        # A disabled item keeps an enabled row (Flet disables every child of a disabled ListTile, so its
        # ReasonChip would never open): muted label, and a tap shows the reason instead of selecting.
        unavailable = item.disabled_reason is not None
        color = ft.Colors.ON_SURFACE_VARIANT if unavailable else (ft.Colors.ERROR if item.destructive else None)
        return ft.ListTile(
            title=ft.Text(item.label, color=color),
            leading=ft.Icon(icon_data(item.icon), color=color) if item.icon is not None else None,
            trailing=ReasonChip(reason=item.disabled_reason) if item.disabled_reason else None,
            on_click=lambda e, it=item: self._on_select(e, it),
            min_height=tokens.SIZES["hit_target"],
            key=item.key or item.label,
        )

    def estimated_height(self, width: float) -> float:
        """Upper estimate (dp, at 200 % text) of the header, rows and Cancel at ``width``."""
        hit = float(tokens.SIZES["hit_target"])
        inner = max(120.0, width - 88)  # ListTile padding and the leading icon
        total = 8.0
        for label, reason in [(i.label, i.disabled_reason) for i in self.items] + [("Cancel", None)]:
            total += max(hit, text_height(label, inner - (140 if reason else 0), 16) + 16)
        if self.title:
            total += text_height(self.title, width - 32, 16)
        if self.subtitle:
            total += text_height(self.subtitle, width - 32, 12)
        if self.title or self.subtitle:
            total += 8
        return total

    def show(self, page: Any) -> None:
        self._page = page
        width = min(page_width(page), float(tokens.SIZES["dialog_max"])) if self.tablet else page_width(page)
        self.body.scroll = None if fits_compact(page, self.estimated_height(width)) else ft.ScrollMode.AUTO
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)

    def _on_cancel(self, e: Any = None) -> None:
        self.close()
        self._fire_cancel()

    def _on_dismiss(self, e: Any = None) -> None:
        if self.selected is None:  # closed without a row (back gesture, outside tap)
            self._fire_cancel()

    def _fire_cancel(self) -> None:
        if self._cancelled or self.on_cancel is None:
            return
        self._cancelled = True
        call_handler(self.on_cancel)

    def _on_select(self, e: Any, item: ActionItem) -> None:
        if item.disabled_reason is not None:
            self.explain(item)
            return
        self.selected = item
        self.close()
        call_handler(item.on_select)

    def explain(self, item: ActionItem) -> Any:
        """Tap on an unavailable item: its reason (the InfoSheet the ReasonChip opens)."""
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        self.explained = item
        sheet = InfoSheet(title=item.label, body=str(item.disabled_reason or ""))
        if self._page is not None:
            sheet.show(self._page)
        return sheet

    def item(self, label: str) -> ActionItem:
        for candidate in self.items:
            if candidate.label == label:
                return candidate
        raise KeyError(label)
