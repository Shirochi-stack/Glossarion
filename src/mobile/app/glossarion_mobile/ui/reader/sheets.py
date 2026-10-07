"""Small Reader sheets: Bookmarks (⋯ › Bookmarks), the glossary entry sheet stub, the image viewer.

* ``BookmarksSheet`` lists Prefs ``reader_bookmarks`` for the book with
  "Add bookmark here"; tap jumps, the trailing icon removes.
* ``GlossaryEntryStub`` is what the selection's "Add to glossary" opens until
  the glossary editor ships (U6): the selected raw term prefilled, a
  translation field, Copy, and a disabled Save with its reason (nothing hidden).
* ``ImageViewer``: a double-tapped page image, zoomable (``InteractiveViewer``).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["BookmarksSheet", "GlossaryEntryStub", "ImageViewer"]


class _Sheet:
    sheet: Any

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        page = getattr(self, "_page", None)
        close_dialog(page, self.sheet)


class BookmarksSheet(_Sheet):
    def __init__(
        self,
        bookmarks: Sequence[dict],
        *,
        label_for: Callable[[dict], str],
        on_open: Callable[[dict], Any],
        on_remove: Callable[[int], Any],
        on_add: Callable[[], Any],
    ) -> None:
        self.on_open = on_open
        self.on_remove = on_remove
        self.on_add = on_add
        self.label_for = label_for
        self.list_view = ft.ListView(controls=[], spacing=0, expand=True, key="bookmarks-list")
        self.empty = ft.Text("No bookmarks yet.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                             color=ft.Colors.ON_SURFACE_VARIANT)
        self.set_items(bookmarks)
        body = ft.Container(
            padding=ft.Padding.only(left=16, right=16, bottom=8),
            content=ft.Column(
                [
                    ft.Row([ft.Text("Bookmarks", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, expand=True),
                            ft.FilledTonalButton(content="Add bookmark here", icon=ft.Icons.BOOKMARK_ADD,
                                                 on_click=lambda e: call_handler(self.on_add), key="bookmark-add")]),
                    self.empty,
                    self.list_view,
                ],
                spacing=8,
                expand=True,
            ),
        )
        self.sheet = ft.BottomSheet(content=ft.Container(content=body, height=420), show_drag_handle=True,
                                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def set_items(self, bookmarks: Sequence[dict]) -> None:
        self.items = [dict(b) for b in bookmarks]
        self.empty.visible = not self.items
        self.list_view.controls = [
            ft.ListTile(
                leading=ft.Icon(ft.Icons.BOOKMARK),
                title=ft.Text(self.label_for(mark), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text(time.strftime("%Y-%m-%d %H:%M", time.localtime(float(mark.get("created") or 0))),
                                 theme_style=ft.TextThemeStyle.BODY_SMALL),
                trailing=ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Remove bookmark",
                                       size_constraints=HIT_TARGET,
                                       on_click=lambda e, i=index: call_handler(self.on_remove, i)),
                on_click=lambda e, m=mark: self._open(m),
                min_height=tokens.SIZES["row_two_line"],
                key=f"bookmark-{index}",
            )
            for index, mark in enumerate(self.items)
        ]
        for control in (self.list_view, self.empty):
            try:
                control.update()
            except Exception:
                pass

    def _open(self, mark: dict) -> None:
        self.close()
        call_handler(self.on_open, mark)


class GlossaryEntryStub(_Sheet):
    """Fallback when the Glossary Manager (GlossaryFeature) is not installed in this session."""

    REASON = "Glossary Manager unavailable"
    DETAIL = ("Adding terms from the Reader opens the Glossary Manager's entry editor, which is not available in "
              "this session. Copy the term for now.")

    def __init__(self, term: str, *, on_copy: Callable[[str], Any], book_title: str = "") -> None:
        self.term = ft.TextField(label="Raw term", value=term, dense=True, key="gloss-raw")
        self.translation = ft.TextField(label="Translation", dense=True, autofocus=True, key="gloss-translation")
        self.save_button = ft.FilledButton(content="Save", disabled=True, key="gloss-save")
        body = ft.Container(
            padding=ft.Padding.only(left=16, right=16, bottom=16),
            content=ft.Column(
                [
                    ft.Text("Add to glossary", theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.Text(book_title, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            visible=bool(book_title)),
                    self.term,
                    self.translation,
                    ft.Row([ReasonChip(reason=self.REASON, detail=self.DETAIL),
                            ft.TextButton(content="Copy term", icon=ft.Icons.CONTENT_COPY,
                                          on_click=lambda e: call_handler(on_copy, self.term.value or "")),
                            self.save_button],
                           alignment=ft.MainAxisAlignment.END, wrap=True),
                ],
                spacing=10,
                tight=True,
            ),
        )
        self.sheet = ft.BottomSheet(content=body, show_drag_handle=True, scrollable=True,
                                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)


class ImageViewer(_Sheet):
    def __init__(self, data: bytes, *, height: Optional[float] = None) -> None:
        viewer = ft.InteractiveViewer(content=ft.Image(src=data, fit=ft.BoxFit.CONTAIN), min_scale=0.5, max_scale=6,
                                      expand=True)
        self.sheet = ft.BottomSheet(content=ft.Container(content=viewer, height=min(720.0, (height or 800) * 0.85)),
                                    show_drag_handle=True, bgcolor=ft.Colors.BLACK)
