"""Quick-action chips above the composer (UI_SPEC §2.7).

With an attachment in the composer: **Translate** · **Extract glossary first** (books) ·
**Translate as manga** (images / CBZ) · **Open in Reader** (EPUB / TXT). At most five, in a scrolling
row, dismissible (× hides them until the attachment changes). The chat decides what each chip
does (``ChatView._on_quick_chip``): the ＋ sheet tools for glossary and manga (the manga tool
receives the attachment), Send for Translate, the Reader for an EPUB or TXT.

``chips_for_attachment`` is pure (host-tested); ``QuickChips`` is the row.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["MAX_CHIPS", "QUICK_CHIPS", "QuickChips", "chips_for_attachment"]

MAX_CHIPS = 5
#: (chip id, label, icon)
QUICK_CHIPS = (
    ("translate", "Translate", "TRANSLATE"),
    ("extract_glossary", "Extract glossary first", "SPELLCHECK"),
    ("manga", "Translate as manga", "AUTO_STORIES"),
    ("open_reader", "Open in Reader", "MENU_BOOK"),
)
_GLOSSARY_SOURCES = (".epub", ".pdf", ".txt", ".md", ".html", ".htm", ".xhtml", ".zip", ".csv", ".json")
_MANGA_SOURCES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".cbz")
_READER_SOURCES = (".epub", ".txt")


def chips_for_attachment(path: Optional[str]) -> list:
    """Chip ids for the composer attachment ``path`` (empty without one)."""
    if not path:
        return []
    ext = os.path.splitext(str(path))[1].lower()
    ids = ["translate"]
    if ext in _GLOSSARY_SOURCES:
        ids.append("extract_glossary")
    if ext in _MANGA_SOURCES:
        ids.append("manga")
    if ext in _READER_SOURCES:
        ids.append("open_reader")
    return ids[:MAX_CHIPS]


class QuickChips:
    """The chip row; ``set_attachment(path)`` rebuilds it only when the chips change."""

    def __init__(self, on_select: Callable[[str], Any]) -> None:
        self.on_select = on_select
        self.signature: tuple = ()
        self.dismissed: Optional[str] = None
        self._gen = 0
        self.ids: list = []
        self.row = ft.Row([], scroll=ft.ScrollMode.AUTO, spacing=6, visible=False, key="quick-chips")
        self.control = ft.Container(content=self.row, padding=ft.Padding.symmetric(horizontal=12, vertical=2))
        self.control.visible = False

    def set_attachment(self, path: Optional[str], *, hidden: bool = False) -> bool:
        """Show the chips for ``path``; returns True when the row changed."""
        ids = [] if hidden or (path and path == self.dismissed) else chips_for_attachment(path)
        signature = (path, tuple(ids))
        if signature == self.signature:
            return False
        self.signature = signature
        self.ids = ids
        self._gen += 1  # per-build keys: Flet 1.0.3 freezes a control re-rendered under the same key
        labels = {cid: (label, icon) for cid, label, icon in QUICK_CHIPS}
        chips: list = [
            ft.Chip(label=ft.Text(labels[cid][0]), leading=ft.Icon(icon_data(labels[cid][1]), size=16),
                    on_click=lambda e, c=cid: self.on_select(c), key=f"quick-{cid}-{self._gen}")
            for cid in ids
        ]
        if chips:
            chips.append(ft.IconButton(icon=ft.Icons.CLOSE, icon_size=16, tooltip="Hide suggestions",
                                       on_click=lambda e, p=path: self.dismiss(p), key=f"quick-dismiss-{self._gen}", size_constraints=HIT_TARGET))
        self.row.controls = chips
        self.row.visible = bool(chips)
        self.control.visible = bool(chips)
        return True

    def dismiss(self, path: Optional[str]) -> None:
        self.dismissed = path
        self.set_attachment(path)
        try:
            self.control.update()
        except Exception:
            pass

    @property
    def visible_ids(self) -> Sequence[str]:
        return tuple(self.ids)
