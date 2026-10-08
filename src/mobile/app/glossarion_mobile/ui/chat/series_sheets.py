"""Series sheets (UI_SPEC §2.15): the series picker and the series editor.

* ``SeriesPickerSheet``: "Move to Series…" (a chat: chat ⋯ menu, drawer long-press sheet, Chat
  settings) and "Add to Series" (Library books): every series with its colour dot and name (✓ the
  current one), "New series…", and "Remove from series" for a chat that is in one. A bottom sheet
  on phones, a dialog on tablets (like ``ActionSheet``).
* ``SeriesEditorDialog``: name (≤ 80 characters), colour (8 tonal swatches) and cover (one of the
  linked Library books, or none); in edit mode also "Delete series" (confirmed; the chats stay and
  only leave the series, linked books stay in the Library).

Pure presentation: the SeriesFeature applies the choices to ``state.series``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.state.series import MAX_NAME_CHARS, SERIES_COLORS, clean_name, color_hex
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.sheet import sheet_frame

__all__ = ["DELETE_BODY", "SeriesEditorDialog", "SeriesPickerSheet", "color_dot"]

DELETE_BODY = ("The chats stay and only leave the series; linked books stay in the Library. "
               "The series defaults are removed.")


def color_dot(color: str, size: int = 12, key: Optional[str] = None) -> ft.Container:
    """The series colour as a dot (``color``: a swatch name or a hex value)."""
    value = color if str(color).startswith("#") else color_hex(color)
    return ft.Container(width=size, height=size, border_radius=size / 2, bgcolor=value, key=key)


class SeriesPickerSheet:
    """Pick a series (or a new one / none). ``on_pick(sid)`` runs after the sheet closed: a series
    id, or None for "Remove from series"; ``on_new()`` for "New series…"."""

    def __init__(
        self,
        *,
        series: Sequence[Any],  # state.series.Series rows
        current: Optional[str] = None,
        title: str = "Move to Series",
        subtitle: Optional[str] = None,
        on_pick: Optional[Callable[[Optional[str]], Any]] = None,
        on_new: Optional[Callable[[], Any]] = None,
        allow_remove: bool = False,
        tablet: bool = False,
    ) -> None:
        self.series = list(series)
        self.current = current
        self.on_pick = on_pick
        self.on_new = on_new
        self.tablet = tablet
        self.choice: Any = None  # sid, None (remove), "new" or "cancel" once closed
        self._page: Any = None
        self.tiles: dict = {}
        rows: list[ft.Control] = []
        for item in self.series:
            selected = item.id == current
            tile = ft.ListTile(
                leading=color_dot(item.color_hex, 16),
                title=ft.Text(item.name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                trailing=ft.Icon(ft.Icons.CHECK, color=ft.Colors.PRIMARY) if selected else None,
                selected=selected,
                min_height=tokens.SIZES["hit_target"],
                on_click=lambda e, s=item.id: self._pick(s),
                key=f"pick-series-{item.id}",
            )
            self.tiles[item.id] = tile
            rows.append(tile)
        if not self.series:
            rows.append(ft.Container(
                padding=ft.Padding.symmetric(horizontal=16, vertical=8),
                content=ft.Text("No series yet. A series groups chats and books and gives its chats shared "
                                "defaults (model, profile, language, glossary).",
                                theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ))
        self.new_tile = ft.ListTile(leading=ft.Icon(ft.Icons.ADD), title=ft.Text("New series…"),
                                    min_height=tokens.SIZES["hit_target"], on_click=lambda e: self._new(),
                                    key="pick-series-new")
        rows.append(self.new_tile)
        self.remove_tile: Optional[ft.ListTile] = None
        if allow_remove and current:
            self.remove_tile = ft.ListTile(leading=ft.Icon(ft.Icons.REMOVE_CIRCLE_OUTLINE),
                                           title=ft.Text("Remove from series"), min_height=tokens.SIZES["hit_target"],
                                           on_click=lambda e: self._pick(None), key="pick-series-remove")
            rows.append(self.remove_tile)
        rows.append(ft.ListTile(leading=ft.Icon(ft.Icons.CLOSE), title=ft.Text("Cancel"),
                                min_height=tokens.SIZES["hit_target"], on_click=lambda e: self._cancel(),
                                key="cancel"))
        header: list[ft.Control] = [ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                            weight=ft.FontWeight.W_600)]
        if subtitle:
            header.append(ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                  color=ft.Colors.ON_SURFACE_VARIANT))
        self.body = ft.Column(
            [ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=8),
                          content=ft.Column(header, tight=True, spacing=2)), *rows],
            tight=True, spacing=0, scroll=ft.ScrollMode.AUTO,
        )
        if tablet:
            self.dialog: Any = ft.AlertDialog(content=ft.Container(width=tokens.SIZES["dialog_max"], content=self.body),
                                              content_padding=ft.Padding.symmetric(vertical=12))
        else:
            self.dialog = ft.BottomSheet(content=sheet_frame(self.body, padding=ft.Padding.only(bottom=8)),
                                         show_drag_handle=True, scrollable=True,
                                         bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)

    def _pick(self, sid: Optional[str]) -> None:
        self.choice = sid
        self.close()
        call_handler(self.on_pick, sid)

    def _new(self) -> None:
        self.choice = "new"
        self.close()
        call_handler(self.on_new)

    def _cancel(self) -> None:
        self.choice = "cancel"
        self.close()


class SeriesEditorDialog:
    """New / edit series: name, colour, cover. ``on_save({"name", "color", "cover_bid"})``."""

    def __init__(
        self,
        *,
        title: str = "New series",
        name: str = "",
        color: Optional[str] = None,
        books: Sequence[tuple] = (),  # (bid, title) of the linked books (cover choices)
        cover_bid: str = "",
        on_save: Optional[Callable[[dict], Any]] = None,
        on_delete: Optional[Callable[[], Any]] = None,
        save_label: str = "Save",
    ) -> None:
        self.on_save = on_save
        self.on_delete = on_delete
        self.color = color if color in dict(SERIES_COLORS) else SERIES_COLORS[0][0]
        self._page: Any = None
        self.saved = False
        self.confirm: Optional[ConfirmDialog] = None
        self.name_field = ft.TextField(label="Series name", value=name, autofocus=True, max_length=MAX_NAME_CHARS,
                                       on_submit=lambda e: self._save(), key="series-name")
        self.swatches: dict = {}
        for swatch, hex_value in SERIES_COLORS:
            self.swatches[swatch] = ft.Container(
                width=tokens.SIZES["hit_target"],
                height=tokens.SIZES["hit_target"],
                alignment=ft.Alignment.CENTER,
                tooltip=swatch.title(),
                on_click=lambda e, s=swatch: self.set_color(s),
                content=self._swatch_face(hex_value, swatch == self.color),
                key=f"swatch-{swatch}",
            )
        options = [ft.DropdownOption(key="", text="No cover")]
        options += [ft.DropdownOption(key=bid, text=str(book_title or bid)) for bid, book_title in books]
        self.cover = ft.Dropdown(label="Cover (a linked book)", value=cover_bid if cover_bid in {b for b, _t in books}
                                 else "", options=options, dense=True, disabled=not books, key="series-cover")
        content: list[ft.Control] = [
            self.name_field,
            ft.Text("Colour", theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
            ft.Row(list(self.swatches.values()), wrap=True, spacing=0, run_spacing=0),
            self.cover,
        ]
        if not books:
            content.append(ft.Text("Link books from the Library (⋯ › Add to Series) to use one as the cover.",
                                   theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        actions: list[ft.Control] = []
        if on_delete is not None:
            actions.append(ft.TextButton(content="Delete series", on_click=lambda e: self._ask_delete(),
                                         style=ft.ButtonStyle(color=ft.Colors.ERROR), key="series-delete"))
        actions += [ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="series-cancel"),
                    ft.FilledButton(content=save_label, on_click=lambda e: self._save(), key="series-save")]
        self.dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Container(width=tokens.SIZES["dialog_max"],
                                 content=ft.Column(content, tight=True, spacing=8, scroll=ft.ScrollMode.AUTO)),
            actions=actions,
            actions_alignment=ft.MainAxisAlignment.END,
        )

    @staticmethod
    def _swatch_face(hex_value: str, selected: bool) -> ft.Control:
        return ft.Container(
            width=32, height=32, border_radius=16, bgcolor=hex_value, alignment=ft.Alignment.CENTER,
            border=ft.Border.all(3, ft.Colors.ON_SURFACE) if selected else None,
            content=ft.Icon(ft.Icons.CHECK, size=18, color=ft.Colors.WHITE) if selected else None,
        )

    def set_color(self, swatch: str) -> None:
        if swatch not in self.swatches:
            return
        self.color = swatch
        for name, holder in self.swatches.items():
            holder.content = self._swatch_face(dict(SERIES_COLORS)[name], name == swatch)
            try:
                holder.update()
            except Exception:
                pass

    def values(self) -> dict:
        return {"name": clean_name(self.name_field.value), "color": self.color,
                "cover_bid": str(self.cover.value or "")}

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)

    def _save(self) -> None:
        if self.saved:  # a second tap while the dialog closes
            return
        values = self.values()
        if not values["name"]:
            self.name_field.error = "Enter a name"
            try:
                self.name_field.update()
            except Exception:
                pass
            return
        self.saved = True
        self.close()
        call_handler(self.on_save, values)

    def _ask_delete(self) -> None:
        name = clean_name(self.name_field.value) or "this series"

        def confirmed() -> None:
            self.close()
            call_handler(self.on_delete)

        self.confirm = ConfirmDialog(title=f"Delete {name}?", body=DELETE_BODY, confirm_label="Delete",
                                     destructive=True, on_confirm=confirmed)
        if self._page is not None:
            self.confirm.show(self._page)
