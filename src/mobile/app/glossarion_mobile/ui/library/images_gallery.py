"""Interactive Images gallery (Book page › Images › ⋯ › Gallery, Tools › Interactive Images › Gallery; desktop
``interactive_images_gui.InteractiveImagesGalleryDialog``).

A bottom sheet with a grid of the book's AI images: each card shows the image, its chapter and the characters
present. Filter by character (the glossary's names) and chapter. Long-press a card to start selecting, tap to
add or remove cards, then **Delete selected**; a tap outside a selection opens the image full size (Delete ·
Go to chapter). Everything is the shared ``interactive_images_progress_core`` (``gallery_images`` /
``gallery_filter`` / ``delete_gallery_images``: the same text-checked delete the desktop gallery uses).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ALL_CHAPTERS", "ALL_CHARACTERS", "ImagesGallery", "THUMB_SIZE"]

log = logging.getLogger("glossarion.library.ui")

THUMB_SIZE = 256
ALL_CHARACTERS = "All characters"
ALL_CHAPTERS = "All chapters"
_ANY = "*"   # the dropdowns' "no filter" key (a dropdown option needs a non-empty key)


def _ipc() -> Any:
    import interactive_images_progress_core

    return interactive_images_progress_core


def _core() -> Any:
    import interactive_images_core

    return interactive_images_core


class ImagesGallery:
    """The gallery of one book's output ``folder``. ``open_chapter(image)`` opens the image's chapter (the
    caller's Reader / Book page); ``on_changed()`` runs after images were deleted."""

    def __init__(self, ctx: Any, folder: str, *, open_chapter: Optional[Callable[[Any], Any]] = None,
                 on_changed: Optional[Callable[[], Any]] = None, key: str = "ixg") -> None:
        self.ctx = ctx
        self.folder = str(folder or "")
        self.open_chapter = open_chapter
        self.on_changed = on_changed
        self.key = key
        self.images: list = []
        self.glossary_names: list = []
        self.thumbs: dict = {}
        self.filter_character = ""
        self.filter_chapter = ""
        self.selected: set = set()
        self.selecting = False
        self.sheet: Any = None
        self.viewer: Any = None
        self._page: Any = None
        self.character_dd = ft.Dropdown(label="Character", dense=True, expand=True, key=f"{key}-character",
                                        on_select=lambda e: self.set_character(e.control.value))
        self.chapter_dd = ft.Dropdown(label="Chapter", dense=True, expand=True, key=f"{key}-chapter",
                                      on_select=lambda e: self.set_chapter(e.control.value))
        self.count_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key=f"{key}-count")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.TERTIARY, visible=False,
                            key=f"{key}-note")
        self.hint = ft.Text("Long-press a card to select; then tap to add or remove.",
                            theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            key=f"{key}-hint")
        self.select_text = ft.Text("", weight=ft.FontWeight.W_600, key=f"{key}-selected")
        self.select_all_button = ft.TextButton(content="Select all", key=f"{key}-select-all",
                                               on_click=lambda e: self.select_all())
        self.cancel_button = ft.TextButton(content="Cancel", key=f"{key}-cancel",
                                           on_click=lambda e: self.exit_selection())
        self.delete_button = ft.FilledButton(content="🗑 Delete selected", key=f"{key}-delete",
                                             style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR,
                                                                  color=ft.Colors.ON_ERROR),
                                             on_click=lambda e: self.ctx.spawn(self.delete_selected()))
        self.selection_bar = ft.Row([self.select_text, self.select_all_button, self.cancel_button, self.delete_button],
                                    spacing=4, wrap=True, visible=False, key=f"{key}-selection")
        self.grid = ft.GridView(max_extent=170, child_aspect_ratio=0.78, spacing=8, run_spacing=8, expand=True,
                                key=f"{key}-grid")
        self.empty_text = ft.Text("No images yet. Build the chapters first.", color=ft.Colors.ON_SURFACE_VARIANT,
                                  visible=False, key=f"{key}-empty")

    # ---- data -------------------------------------------------------------------------------------

    def _read(self) -> tuple:
        ipc, core = _ipc(), _core()
        model = ipc.open_progress(self.folder)
        images = ipc.gallery_images(model)
        names = core.character_names(core.load_glossary_entries(
            self.folder, owner=core.glossary_resolver(self.ctx.config_snapshot())))
        thumbs = {}
        for image in images:
            data = self.thumbs.get((image.id, os.path.getmtime(image.path)))
            if data is None:
                data = ipc.gallery_thumbnail(image.path, THUMB_SIZE)
            thumbs[(image.id, os.path.getmtime(image.path))] = data
        return images, names, thumbs

    async def load(self) -> list:
        """Read the book's images, glossary names and thumbnails (off the UI thread) and draw the grid."""
        images, names, thumbs = await self.ctx.io(self._read)
        self.images, self.glossary_names, self.thumbs = images, names, thumbs
        ids = {image.id for image in images}
        self.selected &= ids
        if not self.selected:
            self.selecting = False
        self._fill_filters()
        self.render()
        return images

    def _fill_filters(self) -> None:
        ipc = _ipc()
        characters = ipc.gallery_characters(self.images, self.glossary_names)
        chapters = ipc.gallery_chapters(self.images)
        self.character_dd.options = [ft.DropdownOption(key=_ANY, text=ALL_CHARACTERS)] + [
            ft.DropdownOption(key=name, text=name) for name in characters]
        self.chapter_dd.options = [ft.DropdownOption(key=_ANY, text=ALL_CHAPTERS)] + [
            ft.DropdownOption(key=key, text=label) for key, label in chapters]
        if self.filter_character and self.filter_character not in characters:
            self.filter_character = ""
        if self.filter_chapter and self.filter_chapter not in {key for key, _label in chapters}:
            self.filter_chapter = ""
        self.character_dd.value = self.filter_character or _ANY
        self.chapter_dd.value = self.filter_chapter or _ANY

    def visible(self) -> list:
        return _ipc().gallery_filter(self.images, self.filter_character, self.filter_chapter)

    def set_character(self, value: Any) -> None:
        self.filter_character = "" if value in (None, _ANY) else str(value)
        self.render()

    def set_chapter(self, value: Any) -> None:
        self.filter_chapter = "" if value in (None, _ANY) else str(value)
        self.render()

    # ---- the grid ---------------------------------------------------------------------------------

    def _thumb(self, image: Any) -> Optional[bytes]:
        try:
            return self.thumbs.get((image.id, os.path.getmtime(image.path)))
        except OSError:
            return None

    def _card(self, image: Any) -> ft.Control:
        selected = image.id in self.selected
        data = self._thumb(image)
        picture: ft.Control = (ft.Image(src=data, fit=ft.BoxFit.COVER, expand=True,
                                        border_radius=tokens.RADII["card"]) if data
                               else ft.Container(content=ft.Icon(ft.Icons.BROKEN_IMAGE), expand=True,
                                                 alignment=ft.Alignment.CENTER,
                                                 bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH))
        layers: list = [picture]
        if self.selecting:
            layers.append(ft.Container(
                content=ft.Icon(ft.Icons.CHECK_CIRCLE if selected else ft.Icons.RADIO_BUTTON_UNCHECKED, size=22,
                                color=ft.Colors.PRIMARY if selected else ft.Colors.ON_SURFACE_VARIANT),
                alignment=ft.Alignment.TOP_RIGHT, padding=4, key=f"{self.key}-mark-{image.id}"))
        return ft.Container(
            content=ft.Column([
                ft.Stack(layers, expand=True),
                ft.Text(image.caption, size=11, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            ], spacing=2),
            key=f"{self.key}-card-{image.id}", padding=4, border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, i=image: self.tap(i.id), on_long_press=lambda e, i=image: self.long_press(i.id),
            ink=True)

    def render(self) -> None:
        shown = self.visible()
        self.grid.controls = [self._card(image) for image in shown]
        self.empty_text.visible = not shown
        self.empty_text.value = ("No images yet. Build the chapters first." if not self.images
                                 else "No image matches the filters.")
        self.count_text.value = f"{len(shown)} of {len(self.images)} images"
        self._sync_selection_bar()
        self.ctx.push(self.grid, self.empty_text, self.count_text, self.character_dd, self.chapter_dd)

    def _sync_selection_bar(self) -> None:
        count = len(self.selected)
        self.selection_bar.visible = self.selecting
        self.hint.visible = not self.selecting and bool(self.images)
        self.select_text.value = f"{count} selected"
        self.delete_button.content = f"🗑 Delete selected ({count})" if count else "🗑 Delete selected"
        self.delete_button.disabled = count == 0
        self.ctx.push(self.selection_bar, self.hint, self.select_text, self.delete_button)

    # ---- selection --------------------------------------------------------------------------------

    def tap(self, image_id: str) -> Any:
        """A tap: toggles the card while selecting, else opens the image full size."""
        if self.selecting:
            return self.toggle(image_id)
        image = next((i for i in self.images if i.id == image_id), None)
        return self.open_viewer(image) if image is not None else None

    def long_press(self, image_id: str) -> None:
        """A long press starts selecting (and selects that card)."""
        haptic = getattr(self.ctx, "haptic", None)
        if not self.selecting and callable(haptic):
            haptic("medium_impact")
        self.selecting = True
        self.toggle(image_id, force=True)

    def toggle(self, image_id: str, force: Optional[bool] = None) -> None:
        on = (image_id not in self.selected) if force is None else force
        if on:
            self.selected.add(image_id)
        else:
            self.selected.discard(image_id)
        if not self.selected:
            self.selecting = False
        self.render()

    def select_all(self) -> None:
        self.selecting = True
        self.selected = {image.id for image in self.visible()}
        self.render()

    def exit_selection(self) -> None:
        self.selecting = False
        self.selected = set()
        self.render()

    def selected_images(self) -> list:
        return [image for image in self.images if image.id in self.selected]

    # ---- delete / viewer ---------------------------------------------------------------------------

    async def delete_images(self, images: Sequence[Any]) -> int:
        """Delete ``images`` after the confirmation (shared text-checked delete); returns how many went."""
        from glossarion_mobile.ui.glossary.common import ask

        images = list(images or ())
        if not images:
            return 0
        ipc = _ipc()
        if not await ask(self.ctx, title="Delete Images", body=ipc.gallery_delete_confirmation(len(images)),
                         confirm="Delete", cancel="Cancel", destructive=True):
            return 0
        try:
            result = await self.ctx.io(ipc.delete_gallery_images, self.folder, images)
        except Exception as exc:
            log.exception("gallery delete failed")
            self.ctx.say(f"Could not delete the images: {exc}")
            return 0
        if result.skipped:
            self.note.value = f"{len(result.skipped)} image(s) were kept: " + "; ".join(
                f"{image.file}: {reason}" for image, reason in result.skipped[:2])
            self.note.visible = True
        else:
            self.note.visible = False
        self.ctx.push(self.note)
        self.selected = set()
        self.selecting = False
        await self.load()
        if result.deleted and self.on_changed is not None:
            try:
                self.on_changed()
            except Exception:
                log.debug("gallery on_changed failed", exc_info=True)
        return result.deleted

    async def delete_selected(self) -> int:
        return await self.delete_images(self.selected_images())

    def go_to_chapter(self, image: Any) -> Any:
        if self.open_chapter is None or image is None:
            return None
        self.close_viewer()
        self.close()
        return self.open_chapter(image)

    def open_viewer(self, image: Any) -> Optional["_Viewer"]:
        if image is None:
            return None
        viewer = _Viewer(self, image)
        self.viewer = viewer
        page = self._page or getattr(self.ctx, "page", None)
        if page is not None:
            viewer.show(page)
        return viewer

    def close_viewer(self) -> None:
        viewer, self.viewer = self.viewer, None
        if viewer is not None:
            viewer.close()

    # ---- sheet ---------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        page = getattr(self.ctx, "page", None)
        height = float(getattr(page, "height", 0) or 760) * 0.92
        return ft.Container(content=ft.Column([
            ft.Row([ft.Text("🖼 Gallery", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600,
                            expand=True),
                    self.count_text, ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", on_click=lambda e: self.close(), size_constraints=HIT_TARGET,
                                                   key=f"{self.key}-close")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.character_dd, self.chapter_dd], spacing=8),
            self.hint, self.selection_bar, self.note,
            self.empty_text, self.grid,
        ], spacing=6, expand=True), padding=ft.Padding.symmetric(horizontal=12, vertical=8), height=height,
            key=f"{self.key}-sheet")

    def show(self, page: Any = None) -> "ImagesGallery":
        self._page = page or getattr(self.ctx, "page", None)
        self.sheet = ft.BottomSheet(content=self.build(), scrollable=True, show_drag_handle=True,
                                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        if self._page is not None:
            self._page.show_dialog(self.sheet)
        self.ctx.spawn(self.load())
        return self

    def close(self) -> None:
        sheet, self.sheet = self.sheet, None
        if sheet is not None and self._page is not None:
            close_dialog(self._page, sheet)


class _Viewer:
    """One image at full size with its details (Delete · Go to chapter · Close)."""

    def __init__(self, gallery: ImagesGallery, image: Any) -> None:
        self.gallery = gallery
        self.image = image
        self._page: Any = None
        data = None
        try:
            with open(image.path, "rb") as handle:
                data = handle.read()
        except OSError:
            data = None
        details = [image.caption]
        if image.aspect_ratio:
            details.append(f"Ratio {image.aspect_ratio}")
        picture = (ft.Image(src=data, fit=ft.BoxFit.CONTAIN, border_radius=tokens.RADII["card"],
                            key=f"{gallery.key}-viewer-image") if data
                   else ft.Text("The image file could not be read."))
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=ft.Column([
                picture,
                ft.Text(" | ".join(details), weight=ft.FontWeight.W_600),
                ft.Text(image.prompt, theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True,
                        color=ft.Colors.ON_SURFACE_VARIANT, visible=bool(image.prompt)),
                ft.Row([
                    ft.TextButton(content="📄 Go to chapter", key=f"{gallery.key}-viewer-chapter",
                                  visible=gallery.open_chapter is not None,
                                  on_click=lambda e: gallery.go_to_chapter(image)),
                    ft.TextButton(content="🗑 Delete", key=f"{gallery.key}-viewer-delete",
                                  style=ft.ButtonStyle(color=ft.Colors.ERROR),
                                  on_click=lambda e: gallery.ctx.spawn(self.delete())),
                    ft.TextButton(content="Close", key=f"{gallery.key}-viewer-close", on_click=lambda e: self.close()),
                ], alignment=ft.MainAxisAlignment.END, wrap=True),
            ], tight=True, scroll=ft.ScrollMode.AUTO, spacing=8), padding=16),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    async def delete(self) -> int:
        count = await self.gallery.delete_images([self.image])
        if count:
            self.close()
        return count

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        if self._page is not None:
            close_dialog(self._page, self.sheet)
