"""SelectionTopBar + BulkActionBar (UI_SPEC §3.3, §3.7, §5.6), shared by the Library, Chapters and Glossary.

Top bar: ✕ · "N selected" · Select all · optional "Select ▾" (Completed / QA
Failed / Failed group). Bottom bar (64 dp): at most 4 icon+label actions and
**More** (an ``ActionSheet`` with the rest). Actions that do not apply are shown
disabled with their reason (never hidden). Labels collapse to icons (tooltip +
semantics) at >= 160% text scale (§7.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["BulkAction", "BulkActionBar", "SelectionTopBar"]

MAX_VISIBLE_ACTIONS = 4


@dataclass
class BulkAction:
    id: str
    label: str
    icon: Any
    on_select: Optional[Callable[[], Any]] = None
    disabled_reason: Optional[str] = None
    destructive: bool = False

    def as_item(self) -> ActionItem:
        return ActionItem(self.label, self.on_select, icon=self.icon, destructive=self.destructive,
                          disabled_reason=self.disabled_reason, key=f"bulk-{self.id}")


class SelectionTopBar:
    def __init__(self, *, on_close: Callable[[], Any], on_select_all: Callable[[], Any],
                 select_menu: Sequence[tuple] = (), key: str = "selection-top") -> None:
        self.count_text = ft.Text("0 selected", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, expand=True,
                                  key="count")
        controls: list[ft.Control] = [
            ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close selection", size_constraints=HIT_TARGET,
                          on_click=lambda e: call_handler(on_close), key="close"),
            self.count_text,
            ft.TextButton(content="Select all", on_click=lambda e: call_handler(on_select_all), key="select-all"),
        ]
        if select_menu:
            controls.append(ft.PopupMenuButton(
                content=ft.Row([ft.Text("Select"), ft.Icon(ft.Icons.ARROW_DROP_DOWN)], tight=True, spacing=0),
                items=[ft.PopupMenuItem(content=label, on_click=lambda e, fn=fn: call_handler(fn))
                       for label, fn in select_menu],
                tooltip="Select by status",
                size_constraints=HIT_TARGET,
                key="select-menu",
            ))
        self.control = ft.Container(
            content=ft.Row(controls, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            border_radius=tokens.RADII["card"],
            padding=ft.Padding.symmetric(horizontal=4),
            height=tokens.SIZES["app_bar"],
            visible=False,
            key=key,
        )

    def set_count(self, count: int) -> None:
        self.count_text.value = f"{count} selected"
        self.control.visible = count > 0 or self.control.visible

    def show(self, visible: bool) -> None:
        self.control.visible = visible


class BulkActionBar:
    def __init__(self, *, page: Any = None, tablet: bool = False, compact: bool = False,
                 key: str = "bulk-bar") -> None:
        self.page = page
        self.tablet = tablet
        self.compact = compact
        self.actions: list[BulkAction] = []
        self.more_sheet: Optional[ActionSheet] = None
        self.row = ft.Row(spacing=0, alignment=ft.MainAxisAlignment.SPACE_AROUND, expand=True)
        self.control = ft.Container(
            content=self.row,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            height=tokens.SIZES["bottom_bar"],
            padding=ft.Padding.symmetric(horizontal=4),
            visible=False,
            key=key,
        )
        self.buttons: dict[str, ft.Control] = {}

    def _button(self, action: BulkAction) -> ft.Control:
        disabled = action.disabled_reason is not None
        color = ft.Colors.ERROR if action.destructive and not disabled else None
        parts: list[ft.Control] = [ft.Icon(icon_data(action.icon), color=color, size=22)]
        if not self.compact:
            parts.append(ft.Text(action.label, size=11, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, color=color,
                                 text_align=ft.TextAlign.CENTER))
        button = ft.Container(
            content=ft.Column(parts, spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
            on_click=(lambda e, a=action: self._run(a)),
            padding=ft.Padding.symmetric(horizontal=6, vertical=4),
            border_radius=tokens.RADII["card"],
            tooltip=action.disabled_reason or action.label,
            opacity=0.38 if disabled else 1.0,
            expand=True,
            alignment=ft.Alignment.CENTER,
            ink=True,
            key=f"bulk-{action.id}",
        )
        return ft.Semantics(content=button, label=action.label, button=True)

    def _run(self, action: BulkAction) -> Any:
        if action.disabled_reason is not None:
            if self.page is not None:
                from glossarion_mobile.ui.components.dialogs import show_snackbar

                try:
                    show_snackbar(self.page, action.disabled_reason)
                except Exception:
                    pass
            return None
        return call_handler(action.on_select)

    def set_actions(self, primary: Sequence[BulkAction], more: Sequence[BulkAction] = ()) -> None:
        primary = list(primary)
        overflow = list(more)
        if len(primary) > MAX_VISIBLE_ACTIONS:
            overflow = primary[MAX_VISIBLE_ACTIONS:] + overflow
            primary = primary[:MAX_VISIBLE_ACTIONS]
        self.actions = primary + overflow
        self.overflow = overflow
        self.buttons = {a.id: self._button(a) for a in primary}
        controls = list(self.buttons.values())
        if overflow:
            more_action = BulkAction("more", "More", "MORE_HORIZ", on_select=self.open_more)
            self.buttons["more"] = self._button(more_action)
            controls.append(self.buttons["more"])
        self.row.controls = controls

    def open_more(self) -> Optional[ActionSheet]:
        items = [a.as_item() for a in getattr(self, "overflow", [])]
        if not items:
            return None
        self.more_sheet = ActionSheet(items, title="More", tablet=self.tablet)
        if self.page is not None:
            self.more_sheet.show(self.page)
        return self.more_sheet

    def action(self, action_id: str) -> BulkAction:
        for action in self.actions:
            if action.id == action_id:
                return action
        raise KeyError(action_id)

    def show(self, visible: bool) -> None:
        self.control.visible = visible
