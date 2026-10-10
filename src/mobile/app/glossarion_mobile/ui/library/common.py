"""Shared plumbing for the Library screens: ``LibraryContext`` and small status widgets (UI_SPEC §5.3, §5.6).

``LibraryContext`` is what every Library / Book page screen receives instead of the
whole app (host tests pass a fake page and service): the ``LibraryService``, the
page, navigation, snackbars, the io pool, haptics and the platform flags.

Widgets: ``status_avatar`` (32 dp circle, Material icon on a 16% tint, the
emoji as the semantics label), ``stat_chip`` (emoji + label + count chip; tap
filters, long-press jumps - wrapped in a ``GestureDetector`` because ``Chip`` has
no long-press), ``pill`` and ``mode_badge``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, resolve_color, status_color

__all__ = [
    "LibraryContext",
    "OPENS_ELSEWHERE_REASON",
    "icon_button",
    "mode_badge",
    "opens_in_another_app",
    "pill",
    "section_title",
    "stat_chip",
    "status_avatar",
    "tinted",
]

log = logging.getLogger("glossarion.library.ui")


@dataclass
class LibraryContext:
    service: Any  # services.library.LibraryService
    page: Any = None
    dispatcher: Any = None
    navigate: Optional[Callable[..., Any]] = None  # (name, params=None, query=None)
    notify: Optional[Callable[..., Any]] = None  # (message, action_label=None, on_action=None)
    files: Any = None  # FileBridge
    jobs: Any = None  # JobsFeature (submit, subscribe, has_kind, request_stop)
    prefs: Any = None
    haptics: Any = None
    shell: Any = None
    intents: Any = None
    copy_text: Optional[Callable[[str], Any]] = None
    reader: Optional[Callable[[], Any]] = None  # -> the ReaderFeature (``open_book``), when installed
    glossary: Optional[Callable[[], Any]] = None  # -> the GlossaryFeature (TranslateSheet glossary chip, U9)
    push_overlay: Optional[Callable[[Any], Any]] = None
    pop_overlay: Optional[Callable[[], Any]] = None
    foreground: Callable[[], bool] = lambda: True
    platform: str = "desktop"
    tablet: bool = False
    dark: bool = False
    text_scale: float = 1.0
    extras: dict = field(default_factory=dict)

    # ---- threading -------------------------------------------------------------------------

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await self.service.io(fn, *args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    @staticmethod
    def push(*controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:  # not mounted (tests) or detached
                pass

    # ---- feedback / navigation ---------------------------------------------------------------

    def say(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> Any:
        notify = self.notify
        if notify is None:
            log.info("library: %s", message)
            return None
        try:
            return notify(message, action_label, on_action)
        except TypeError:
            return notify(message)

    async def translate_metadata(self, books: Any) -> Optional[str]:
        """Translate Metadata for Library books (bulk "Metadata", card ⋯, Book ⋯, Book › Overview): the
        desktop Library's ``_translate_metadata_for_books`` asks "Metadata Already Exists" (Cancel / Yes)
        when an output folder already has a metadata.json (a regeneration replaces edited or translated
        fields; ``headers_model.existing_metadata_warning``, the copy Tools › Headers & metadata uses),
        then submits the ``metadata`` job. Returns the job id, or None."""
        from glossarion_mobile.ui.tools import headers_model as hm
        from glossarion_mobile.ui.tools.common import ask

        service = self.service
        books = [dict(book) for book in books or ()]
        if not books:
            return None
        if not service.has_job_kind("metadata"):
            self.say("Metadata translation is not available in this session")
            return None
        workspace_for = getattr(service, "workspace_for", None)
        folders = [(workspace_for(book) if callable(workspace_for) else "") or str(book.get("output_folder") or "")
                   for book in books]
        try:
            warning = await self.io(hm.existing_metadata_warning, folders)
        except Exception:
            log.debug("metadata.json check failed", exc_info=True)
            warning = None
        if warning:
            answer = await ask(self, "Metadata Already Exists", warning,
                               [("cancel", "Cancel", "text"), ("yes", "Yes", "filled")], key="lib-meta-exists")
            if answer != "yes":
                return None
        try:
            spec = await self.io(service.metadata_spec, books)
            return await service.submit(spec)
        except Exception as exc:
            self.say(f"Could not start: {exc}")
            return None

    async def share_books(self, books: Any) -> bool:
        """↗ Share (Library card ⋯, selection › More, Book page ⋯): each book's first compiled output
        (``compiled_outputs_blocking``), else its own file, else its raw source, through the FileBridge
        share sheet (the desktop "Open File" for a phone). True when the sheet was shown."""
        files = self.files
        if files is None:
            self.say("Sharing is not available in this session")
            return False
        service = self.service
        books = [dict(book) for book in books or ()]

        def targets() -> list:
            out = []
            for book in books:
                outputs = service.compiled_outputs_blocking(book)
                if outputs:
                    out.append(outputs[0][0])
                elif book.get("path") and os.path.isfile(str(book.get("path"))):
                    out.append(str(book.get("path")))
                elif service.raw_source(book):
                    out.append(service.raw_source(book))
            return out

        paths = await self.io(targets)
        if not paths:
            self.say("Nothing to share for the selection")
            return False
        return bool(await files.share(paths))

    def go(self, name: str, params: Optional[dict] = None, query: Optional[dict] = None) -> None:
        navigate = self.navigate
        if navigate is None:
            log.info("library: navigate %s %s %s (no navigator)", name, params, query)
            return
        try:
            navigate(name, params, query)
        except TypeError:
            navigate(name, params)

    def open_reader(self, book: Optional[Mapping[str, Any]] = None, *, bid: Optional[str] = None,
                    path: Optional[str] = None, chapter: Optional[int] = None,
                    chapter_filename: Optional[str] = None, mode: Optional[str] = None,
                    raw_only: bool = False, resume: bool = False) -> Optional[str]:
        """Open the Reader: ``ReaderFeature.open_book`` (the row / file passed in-process), else the
        ``/reader/<bid>`` route (the Reader resolves the id with ``LibraryService.book_for_bid``).
        ``resume`` opens at the saved reading position (▶ Continue)."""
        reader = self.reader() if callable(self.reader) else None
        opener = getattr(reader, "open_book", None)
        if callable(opener):
            kwargs = {"resume": True} if resume else {}
            try:
                return opener(dict(book) if book is not None else None, path=path, chapter=chapter,
                              chapter_filename=chapter_filename, mode=mode, raw_only=raw_only, **kwargs)
            except Exception:
                log.exception("ReaderFeature.open_book failed")
        if bid is None and book is not None:
            bid = self.service.bid_for(book)
        if bid is None and path:
            kind = "txt" if str(path).lower().endswith(".txt") else "epub"  # the Reader opens EPUB and TXT
            bid = self.service.bid_for({"path": path, "type": kind,
                                        "name": os.path.splitext(os.path.basename(path))[0]})
        if not bid:
            return None
        query: dict = {}
        if chapter is not None and chapter >= 0:
            query["ch"] = int(chapter)
        if mode in ("translated", "original", "bilingual"):
            query["mode"] = mode
        self.go("reader", {"bid": bid}, query or None)
        return bid

    def show(self, dialog: Any) -> Any:
        if self.page is not None and dialog is not None:
            dialog.show(self.page)
        return dialog

    def haptic(self, kind: str) -> None:
        haptics = self.haptics
        if haptics is None:
            return
        try:
            haptics.fire(kind)
        except Exception:
            pass

    def is_top(self, screen: Any) -> bool:
        shell = self.shell
        if shell is None:
            return True
        top = getattr(shell, "top_screen", None)
        if top is screen:
            return True
        return bool(getattr(screen, "embedded", False))

    def color(self, status: str) -> str:
        return status_color(status, self.dark)


#: Why a PDF without a translation workspace has no Reader action (card menus, the Book page).
OPENS_ELSEWHERE_REASON = "A PDF without a translation workspace opens in another app (↗ Share)"


def opens_in_another_app(book: Mapping[str, Any]) -> bool:
    """A PDF without a translation workspace: library_core's "system" open decision (the desktop
    opens it in the system viewer; the Reader cannot show it). The card menus offer no Reader item
    and the Book page's read button is disabled (``OPENS_ELSEWHERE_REASON`` names ↗ Share): a tap
    opens the Book page for every card. TXT books open in the Reader (its text mode)."""
    kind = str(book.get("type") or "")
    return kind == "pdf" and not book.get("output_folder")


def tinted(color: str, opacity: float = 0.16) -> str:
    return ft.Colors.with_opacity(opacity, resolve_color(color))


def icon_button(icon: Any, tooltip: str, on_click: Any = None, *, key: Optional[str] = None,
                disabled: bool = False, icon_color: Any = None, selected: Optional[bool] = None) -> ft.IconButton:
    """48 dp target, compact visual (UI_SPEC §5.0)."""
    return ft.IconButton(icon=icon_data(icon), tooltip=tooltip, on_click=on_click, size_constraints=HIT_TARGET,
                         key=key, disabled=disabled, icon_color=icon_color, selected=selected)


def status_avatar(status: str, *, icon: Optional[str] = None, emoji: str = "", label: str = "",
                  dark: bool = False, size: int = 32) -> ft.Control:
    """``StatusAvatar``: Material icon on a 16% tint of the status colour; emoji + label for screen readers."""
    color = status_color(status, dark)
    style = tokens.status_style(status)
    circle = ft.Container(
        width=size,
        height=size,
        border_radius=size / 2,
        bgcolor=tinted(color, 0.16),
        alignment=ft.Alignment.CENTER,
        content=ft.Icon(icon_data(icon or style.icon), size=size * 0.56, color=color),
    )
    return ft.Semantics(content=circle, label=f"{emoji} {label or style.label}".strip())


def stat_chip(text: str, status: str, *, selected: bool = False, on_select: Any = None, on_long_press: Any = None,
              dark: bool = False, key: Optional[str] = None, visible: bool = True) -> ft.Control:
    """Stats-row chip: tap toggles the filter, long-press jumps to the next matching row."""
    color = status_color(status, dark)
    chip = ft.Chip(
        label=ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=color),
        selected=selected,
        on_select=on_select,
        show_checkmark=False,
        visual_density=ft.VisualDensity.COMPACT,
        shape=ft.RoundedRectangleBorder(radius=tokens.RADII["chip"]),
        border_side=ft.BorderSide(1, color),
        selected_color=tinted(color, 0.18),
    )
    if on_long_press is None:
        # A GestureDetector without any handler is a Flutter error box on device ("GestureDetector should have
        # at least one event handler defined", Book page › At a glance, U12 item 7).
        chip.key = key
        chip.visible = visible
        return chip
    return ft.GestureDetector(content=chip, on_long_press_start=on_long_press, key=key, visible=visible)


def pill(text: str, *, color: str, background: str, border: Optional[str] = None, size: int = 11,
         key: Optional[str] = None, tooltip: Optional[str] = None) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=size, weight=ft.FontWeight.W_700, color=resolve_color(color), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
        bgcolor=background,
        border=ft.Border.all(1, resolve_color(border or color)),
        border_radius=tokens.RADII["badge"] / 2,
        padding=ft.Padding.symmetric(horizontal=5, vertical=1),
        key=key,
        tooltip=tooltip,
    )


def mode_badge(text: str, *, key: Optional[str] = None) -> ft.Container:
    """``ModeBadge`` ("Mode: Text")."""
    return ft.Container(
        content=ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SECONDARY_CONTAINER),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
        border_radius=tokens.RADII["badge"],
        padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        key=key,
    )


def section_title(text: str) -> ft.Text:
    return ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY, weight=ft.FontWeight.W_600)


def run_handler(handler: Any, *args: Any) -> Any:
    return call_handler(handler, *args)
