"""Tools › Interactive Images (``/tools/interactive-images``; desktop ``interactive_images_gui``
``InteractiveImagesDialog`` + ``InteractiveStylesDialog``).

Every rule, text and file is the shared modules' (``interactive_images_core``,
``interactive_css_presets``, ``interactive_images_runner``); this screen only lays them out:

* **Book**: the book the references and "Apply to book" are for (SourcePicker; ``?out=<bid>`` from the
  Book page), ▶ Build pending (an ``interactive_images`` job) and the book's Images tab.
* **Profile**: the profile row (the Glossary settings' ``ProfileBar`` over
  ``InteractiveImagesProfiles``: ＋ New · 💾 Save · 🗑 Delete / ↺ Reset for Default, Illustrated and
  Styles only), images per chapter 0..10 (default 1; 0 = styles only) and the profile's four prompts as
  prompt cards (planner, image, image system, CSS) with their placeholders.
* **Models**: the text model (empty = the main model) and the image model (``ModelPicker``), the
  reference-image note of the image model (``reference_note``: a warning when it takes none), the
  Image Gen / Edit key pool (its switch and the pool tile that opens Keys › pool, the desktop
  "🔑 Image Keys" preview), max reference images (default 5), resolution, aspect ratio, run options.
* **References**: the book's ``Interactive_Images/references.json``: characters (typed or from the
  glossary) with their images, and art-style images; picked images are copied into
  ``Interactive_Images/references/``.
* **Styles**: the 12 built-in CSS presets and the user's own: Duplicate · ✨ New from prompt (an
  ``interactive_images`` job with ``css_request``) · ✏ Edit CSS · 💾 Save · 🗑 Delete · 📌 Apply to book,
  and the PREVIEW: ``preview_document(css)`` (the shared sample chapter) in a flet-webview over the
  in-app ``ReaderServer`` on Android / iOS; where no WebView runs (desktop host runs, a page that never
  loads) a native note with the classes the style defines and its CSS.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.switch_row import switch_row
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, toggle
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import JobWatch, card, hint_text, schema_tiles
from glossarion_mobile.ui.tools.review import ReviewProfileSource

__all__ = ["InteractiveImagesProfileSource", "InteractiveImagesScreen", "PROMPT_FIELDS", "STYLE_PREVIEW_GRACE",
           "StylePreview", "TABS", "interactive_styles_controls"]

log = logging.getLogger("glossarion.tools.interactive_images")

KIND = "interactive_images"
TABS = ("profile", "models", "references", "styles")
_TAB_LABELS = {"profile": "Profile", "models": "Models", "references": "References", "styles": "Styles"}
#: (profile field, heading, placeholder names attribute of interactive_images_core, key)
PROMPT_FIELDS = (
    ("planner", "Planner prompt", "PLANNER_PLACEHOLDERS", "ix-planner-card"),
    ("image", "Image prompt", "IMAGE_PLACEHOLDERS", "ix-image-card"),
    ("image_system", "Image system prompt (optional)", None, "ix-image-system-card"),
    ("css", "CSS prompt (New from prompt)", "CSS_PLACEHOLDERS", "ix-css-card"),
)
#: seconds the preview WebView has to report a loaded page before the native view replaces it
STYLE_PREVIEW_GRACE = 8.0
PREVIEW_DEBOUNCE = 0.3
NATIVE_NOTE = ("The live preview needs the in-app web view (Android / iOS). Here: the classes this style "
               "defines and its CSS.")
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]


def profiles_text(profiles: Any) -> dict:
    """The selected profile's staged texts ({} without profiles)."""
    return dict(profiles.texts) if profiles is not None else {}


def _core() -> Any:
    import interactive_images_core

    return interactive_images_core


def _presets() -> Any:
    import interactive_css_presets

    return interactive_css_presets


class InteractiveImagesProfileSource(ReviewProfileSource):
    """``ReviewProfileSource`` (tools/review.py; Manga's image edit profiles do the same) over
    ``interactive_images_core.InteractiveImagesProfiles``: a copy of config.json, and the keys an action changed
    written back (the three profile keys and the active profile's runtime keys)."""

    noun = "interactive images prompt profiles"

    def profiles_class(self) -> Any:
        return _core().InteractiveImagesProfiles


def interactive_styles_controls(ctx: Any, *, key: str = "ix-styles-row") -> list:
    """The "Interactive styles" rows the EPUB options show under the CSS override (desktop Other Settings ›
    EPUB › Chapter styles): the switch (``interactive_css_enabled``), the style dropdown, Styles & preview…
    and Clear, which only turns interactive styles off (the CSS override file is never touched)."""
    presets = _presets()
    config = ctx.config_snapshot()
    switch = toggle(value=bool(config.get(presets.ENABLED_KEY, False)), key=f"{key}-enabled",
                    on_change=lambda e: ctx.set_cfg(presets.ENABLED_KEY, bool(e.control.value)))
    dropdown = ft.Dropdown(label="Interactive style", dense=True, expand=True, key=f"{key}-preset",
                           options=[ft.DropdownOption(key=n, text=n) for n in presets.preset_names(config)],
                           value=presets.active_preset_name(config),
                           on_select=lambda e: ctx.set_cfg(presets.PRESET_KEY, e.control.value))

    def clear(_e: Any = None) -> None:
        switch.value = False
        ctx.set_cfg(presets.ENABLED_KEY, False)
        ctx.push(switch)

    return [
        switch_row(switch, "Interactive styles (glossarion_interactive.css, linked after the CSS override)"),
        ft.Row([dropdown], key=f"{key}-preset-row"),
        ft.Row([
            ft.FilledTonalButton(content="🎨 Styles & Preview…", key=f"{key}-open",
                                 on_click=lambda e: ctx.go("tools.interactive-images", None, {"tab": "styles"})),
            ft.TextButton(content="Clear", key=f"{key}-clear", on_click=clear,
                          tooltip="Turn interactive styles off (the CSS override is not touched)"),
            ft.TextButton(content="🖼 Interactive Images…", key=f"{key}-images",
                          on_click=lambda e: ctx.go("tools.interactive-images")),
        ], wrap=True),
    ]


class StylePreview:
    """The preset preview: the shared ``preview_document(css)`` in a WebView (served by the in-app
    ``ReaderServer``, like the QA report), else the native view (note + defined classes + CSS)."""

    def __init__(self, ctx: Any, *, key: str = "ix-preview") -> None:
        self.ctx = ctx
        self.key = key
        self.css = ""
        self.renderer = ""  # "webview" | "native"
        self.server: Any = None
        self.webview: Any = None
        self.page_ready = False
        self.version = 0
        self.disposed = False
        #: the WebView is built when the Styles tab is on screen (a TabBarView builds only the visible page, so a
        #: WebView on a hidden tab never loads and the grace check would give up on it)
        self.active = False
        self.html = ""
        self._debounce: Any = None
        self._check: Any = None
        self.slot = ft.Container(height=520, border_radius=tokens.RADII["card"],
                                 bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST, key=key,
                                 clip_behavior=ft.ClipBehavior.ANTI_ALIAS)

    @property
    def control(self) -> ft.Control:
        return self.slot

    def _webview_ok(self) -> bool:
        try:
            return bool(self.ctx.webview_ok())
        except Exception:
            return False

    def show(self, css: str) -> str:
        """Render ``css`` at once (the first show, a preset switch)."""
        self.css = str(css or "")
        self.html = _presets().preview_document(self.css)
        if not self.active and self.renderer != "webview":
            self.renderer = "pending"
            return self.renderer
        if self.renderer != "native" and self._webview_ok():
            try:
                self._show_webview()
                return self.renderer
            except Exception:
                log.exception("the style preview WebView failed; using the native view")
        self._show_native()
        return self.renderer

    def activate(self) -> str:
        """The Styles tab is on screen: render (the WebView now, so its page can load)."""
        if self.active and self.renderer in ("webview", "native"):
            return self.renderer
        self.active = True
        return self.show(self.css)

    def update_css(self, css: str) -> None:
        """CSS edits: re-rendered after a 300 ms pause in typing."""
        self.css = str(css or "")
        previous = self._debounce
        if previous is not None and not previous.done():
            previous.cancel()
        self._debounce = self.ctx.spawn(self._debounced())

    async def _debounced(self) -> None:
        try:
            await asyncio.sleep(PREVIEW_DEBOUNCE)
        except asyncio.CancelledError:
            return
        if not self.disposed:
            self.show(self.css)

    def _show_webview(self) -> None:
        import flet_webview as fwv

        from glossarion_mobile.services.reader_server import ReaderServer

        if self.server is None:
            self.server = ReaderServer()
            self.server.start()
        self.version += 1
        url = self.server.publish(self.html, name=f"style-preview-{self.version}.html")
        if self.webview is None or self.renderer != "webview":
            self.page_ready = False
            self.webview = fwv.WebView(
                url=url, expand=True, on_page_ended=self._on_loaded,
                prevent_links=["https:", "intent:", "javascript:", "file:", "content:", "mailto:", "tel:"])
            self.slot.content = self.webview
            self.renderer = "webview"
            self.ctx.push(self.slot)
            self._schedule_check(STYLE_PREVIEW_GRACE)
        else:
            # flet-webview reads ``url`` only when built: the Reader's navigate_webview (load_request) loads it
            self.ctx.spawn(self._navigate(url))

    async def _navigate(self, url: str) -> None:
        from glossarion_mobile.ui.reader.reader_view import navigate_webview

        webview = self.webview
        if webview is None or self.disposed:
            return
        try:
            await navigate_webview(webview, url)
        except Exception:
            log.debug("style preview navigation failed", exc_info=True)

    def _on_loaded(self, e: Any = None) -> None:
        self.page_ready = True

    def _schedule_check(self, grace: float) -> None:
        previous = self._check
        if previous is not None and not previous.done():
            previous.cancel()
        self._check = self.ctx.spawn(self.check_webview(grace))

    async def check_webview(self, grace: float = 0.0) -> bool:
        """After ``grace`` seconds: the native view when the page never loaded (the QA report's rule)."""
        try:
            if grace:
                await asyncio.sleep(grace)
        except asyncio.CancelledError:
            return False
        if self.disposed or self.renderer != "webview" or self.page_ready:
            return False
        log.warning("the style preview page never loaded; using the native view")
        self._stop_server()
        self.webview = None
        self._show_native()
        return True

    def _show_native(self) -> None:
        presets = _presets()
        classes = presets.defined_classes(self.css)
        self.slot.content = ft.Column([
            ft.Text(NATIVE_NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                    key=f"{self.key}-note"),
            ft.Text("Classes: " + (", ".join(classes) if classes else "(none)"),
                    theme_style=ft.TextThemeStyle.BODY_SMALL, key=f"{self.key}-classes"),
            ft.Text(self.css or "(empty)", selectable=True, size=11, font_family=getattr(self.ctx, "mono", None),
                    key=f"{self.key}-css"),
        ], spacing=6, scroll=ft.ScrollMode.AUTO, tight=True)
        self.slot.padding = 10
        self.renderer = "native"
        self.ctx.push(self.slot)

    def _stop_server(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

    def dispose(self) -> None:
        self.disposed = True
        for task in (self._debounce, self._check):
            if task is not None and not task.done():
                task.cancel()
        self._stop_server()


class InteractiveImagesScreen(Screen):
    title = "Interactive Images"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("interactive_images", {})
        self.target: Optional[tg.ToolTarget] = self.state.get("target")
        tab = match.get("tab") if match is not None else None
        self.initial_tab = tab if tab in TABS else "profile"
        self.refs: dict = _core().empty_references()
        self.watch = JobWatch(ctx, self._on_job_end, None)
        self.picker: Any = None
        self.editor: Any = None
        self.bar: Any = None
        self.preview = StylePreview(ctx)
        self.css_editor_open = False
        self.last_css_job: Optional[str] = None
        self.tab_built: dict = {}

    # ---- layout -----------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.book_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, expand=True, key="ix-book")
        self.build_button = ft.FilledButton(content="▶ Build pending", key="ix-tool-build",
                                            on_click=lambda e: self.ctx.spawn(self.build_pending()))
        self.progress_button = ft.TextButton(content="Images tab", icon=ft.Icons.LIST_ALT, key="ix-tool-progress",
                                             on_click=lambda e: self.open_progress())
        self.gallery_button = ft.TextButton(content="🖼 Gallery", key="ix-tool-gallery",
                                            on_click=lambda e: self.open_gallery())
        book = ft.Container(content=ft.Column([
            ft.Row([ft.Icon(ft.Icons.MENU_BOOK, size=18), self.book_text,
                    ft.TextButton(content="Choose…", on_click=lambda e: self.open_picker(), key="ix-choose")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.build_button, self.progress_button, self.gallery_button], wrap=True, spacing=8),
        ], spacing=4, tight=True), padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, key="ix-book-strip")
        bodies = {"profile": self._profile_tab(), "models": self._models_tab(),
                  "references": self._references_tab(), "styles": self._styles_tab()}
        self.tabs = ft.Tabs(
            length=len(TABS), selected_index=TABS.index(self.initial_tab), expand=True, on_change=self._on_tab,
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label=_TAB_LABELS[t]) for t in TABS], scrollable=True, key="ix-tabbar"),
                ft.TabBarView(controls=[bodies[t] for t in TABS], expand=True, key="ix-tabview"),
            ], expand=True, spacing=0),
            key="ix-tabs")
        self._render_book()
        return ft.Column([book, self.tabs], spacing=0, expand=True, key="ix-screen")

    @staticmethod
    def _page(controls: Sequence[ft.Control], key: str) -> ft.Control:
        return ft.ListView(controls=list(controls), expand=True, spacing=tokens.SPACING["md"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key=key)

    @property
    def current_tab(self) -> str:
        index = getattr(getattr(self, "tabs", None), "selected_index", 0) or 0
        return TABS[index] if 0 <= index < len(TABS) else TABS[0]

    def set_tab(self, name: str) -> None:
        if name in TABS and getattr(self, "tabs", None) is not None:
            self.tabs.selected_index = TABS.index(name)
            self.ctx.push(self.tabs)
            self._on_tab()

    def _on_tab(self, e: Any = None) -> None:
        if self.current_tab == "styles":
            self.preview.activate()

    # ---- book ---------------------------------------------------------------------------------------

    @property
    def folder(self) -> str:
        return str(getattr(self.target, "folder", "") or "")

    def _render_book(self) -> None:
        target = self.target
        if target is None:
            self.book_text.value = "No book chosen (references and Apply to book need one)"
        else:
            self.book_text.value = f"{target.title} · 📁 {os.path.basename(os.path.normpath(target.folder))}" \
                if target.folder else f"{target.title} · not translated yet"
        reason = self.build_reason()
        self.build_button.disabled = reason is not None
        self.build_button.tooltip = reason
        self.progress_button.disabled = not (target is not None and target.bid)
        self.gallery_button.disabled = not self.folder
        self.ctx.push(self.book_text, self.build_button, self.progress_button, self.gallery_button)

    def build_reason(self) -> Optional[str]:
        if not self.folder:
            return "Choose a translated book first"
        if not self.ctx.has_kind(KIND):
            return "The job service is not running"
        if self.watch.active() is not None:
            return "Interactive Images is running"
        return None

    def did_show(self) -> None:
        for snap in self.watch.adopt((KIND,)):
            pass
        out = self.match.get("out") if self.match is not None else None
        if out:
            self.ctx.spawn(self.preselect_book(out))
        elif self.target is not None:
            self.ctx.spawn(self.load_references())
        if self.bar is not None:
            self.bar.load()
        self._render_book()
        if self.preview.renderer == "":
            self.preview.show(self._css_for(self.style_name))
        self._on_tab()

    def dispose(self) -> None:
        self.watch.stop()
        self.preview.dispose()

    async def preselect_book(self, bid: str) -> Optional[tg.ToolTarget]:
        service = getattr(self.ctx, "service", None)
        book = service.book_for_bid(bid) if service is not None else None
        if not book:
            return None
        target = await self.ctx.io(tg.target_for_book, service, book)
        if target is not None:
            self.set_target(target)
        return target

    def open_picker(self) -> Any:
        from glossarion_mobile.ui.tools.source_picker import SourcePicker

        def chosen(targets: list) -> None:
            if targets:
                self.set_target(targets[0])

        picker = SourcePicker(self.ctx, title="Interactive Images for…", multi=False,
                              eligible=lambda t: None if t.folder else "Not translated yet",
                              on_done=chosen, segment="library", find_folder=True)
        self.picker = picker
        if self.ctx.page is not None:
            picker.show(self.ctx.page)
        return picker

    def set_target(self, target: tg.ToolTarget) -> None:
        self.target = target
        self.state["target"] = target
        self.ctx.remember_source("tools.interactive-images", target.title)
        self._render_book()
        self.ctx.spawn(self.load_references())

    def open_progress(self) -> None:
        if self.target is not None and self.target.bid:
            self.ctx.go("library.book", {"bid": self.target.bid}, {"tab": "images"})

    def open_gallery(self) -> Any:
        """🖼 Gallery: the book's images as cards with multi-select delete (``ui.library.images_gallery``)."""
        from glossarion_mobile.ui.library.images_gallery import ImagesGallery

        if not self._need_book():
            return None
        bid = getattr(self.target, "bid", "")
        gallery = ImagesGallery(
            self.ctx, self.folder,
            open_chapter=(lambda image: self.ctx.go("library.book", {"bid": bid}, {"tab": "images"})) if bid else None)
        self.gallery = gallery
        gallery.show(self.ctx.page)
        return gallery

    async def build_pending(self) -> Optional[str]:
        reason = self.build_reason()
        if reason is not None:
            self.ctx.say(reason)
            return None
        from glossarion_mobile.services.jobs import JobSpec

        spec = JobSpec(kind=KIND, title=self.target.title, inputs=(self.folder,), params={},
                       origin={"type": "tool", "label": "Tools · Interactive Images", "bid": self.target.bid})
        job_id = await self.ctx.submit(spec)
        self.watch.watch(job_id)
        self._render_book()
        return job_id

    def _on_job_end(self, snap: Any) -> None:
        result = dict(getattr(snap, "result", None) or {})
        css = result.get("interactive_css")
        if isinstance(css, dict) and getattr(snap, "id", None) == self.last_css_job:
            self.store_generated(css)
        elif css is None and getattr(snap, "error", None):
            self.ctx.say(str(snap.error))
        self._render_book()

    # ---- Profile ------------------------------------------------------------------------------------

    def _profile_tab(self) -> ft.Control:
        from glossarion_mobile.ui.glossary.settings_tabs import ProfileBar

        core = _core()
        self.prompt_cards = ft.Column(spacing=8, tight=True, key="ix-prompt-cards")
        self.per_chapter = ft.Dropdown(label="Images per chapter", dense=True, width=180, key="ix-per-chapter",
                                       options=[ft.DropdownOption(key=v, text=v) for v in core.PER_CHAPTER_CHOICES],
                                       on_select=lambda e: self.set_per_chapter(e.control.value))
        self.art_style = ft.TextField(label="Art style", dense=True, key="ix-art-style", expand=True,
                                      hint_text=core.DEFAULTS["interactive_images_art_style"],
                                      on_blur=lambda e: self.set_art_style(e.control.value),
                                      on_submit=lambda e: self.set_art_style(e.control.value))
        self.bar = ProfileBar(self.ctx, "interactive_images", prompt_keys=core.InteractiveImagesProfiles.RUNTIME_KEYS,
                              source=InteractiveImagesProfileSource(self.ctx), label="Interactive Images profile",
                              fields=core.InteractiveImagesProfiles.FIELDS, noun="interactive images",
                              on_change=lambda: self.render_prompts() if self.bar is not None else None)
        self.render_prompts()
        values = core.settings_values(self.ctx.config_snapshot())
        self.gender_switch = toggle(value=bool(values["interactive_images_include_gender"]),
                                    key="ix-interactive_images_include_gender",
                                    on_change=lambda e: self.set_value("interactive_images_include_gender",
                                                                       bool(e.control.value)))
        self.subjects = ft.Dropdown(label="Image subjects", dense=True, width=200, key="ix-interactive_images_subjects",
                                    options=[ft.DropdownOption(key=s, text=core.SUBJECT_LABELS[s])
                                             for s in core.SUBJECTS],
                                    value=values["interactive_images_subjects"],
                                    on_select=lambda e: self.set_value("interactive_images_subjects", e.control.value))
        self.placeholder_rows = ft.Column(spacing=6, tight=True, key="ix-placeholder-rows")
        self.render_placeholders()
        return self._page([
            card("Characters & scenes", [
                switch_row(self.gender_switch, "Include gender with character names"),
                hint_text("A gender is sent only when it is confident: the glossary and the Relations & Gender "
                          "Tracker agree, or one has it and the other has nothing against it. The wording is the "
                          "{character_gender} placeholder text."),
                ft.Row([self.subjects], wrap=True),
                hint_text("Characters: scenes with the book's characters. Backgrounds: scenery and establishing "
                          "moments without characters. Both: the planner may pick either ({image_subjects})."),
            ], icon="FACE", key="ix-subjects-card"),
            card("Profile", [
                self.bar.control,
                ft.Row([self.per_chapter,
                        hint_text("0 = styles only: classes from the book's interactive stylesheet, no image model")],
                       wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.art_style,
                hint_text("The look of every picture, sent through {art_style} in the image prompt. An empty text "
                          "removes that line. Stored per profile."),
                hint_text("Each profile holds the planner, image and CSS prompts, the images per chapter and the "
                          "art style. "
                          "The step runs after translation: a chapter whose text would change is rejected."),
            ], icon="TUNE", key="ix-profile-card"),
            card("Prompts", [self.prompt_cards], icon="EDIT_NOTE", key="ix-prompts-card"),
            card("Placeholder texts", [
                hint_text("Every word the step puts into a request comes from your prompts or from these texts; an "
                          "empty text expands to nothing. Shared with the desktop (Edit placeholder text…)."),
                self.placeholder_rows,
            ], icon="TEXT_FIELDS", key="ix-placeholders-card"),
        ], "ix-profile")

    def render_placeholders(self) -> None:
        """One row per editable placeholder text: its current text, Edit and (when edited) Reset."""
        core = _core()
        config = self.ctx.config_snapshot()
        rows = []
        for item in core.placeholder_text_defs():
            name = item["name"]
            text = core.placeholder_text(config, name)
            edited = text != item["default"]
            rows.append(ft.Container(content=ft.Column([
                ft.Text("{" + name + "}", weight=ft.FontWeight.W_600),
                ft.Text(text if text else "(empty: expands to nothing)", theme_style=ft.TextThemeStyle.BODY_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS,
                        key=f"ix-ph-text-{name}"),
                ft.Row([ft.TextButton(content="✏ Edit", key=f"ix-ph-edit-{name}",
                                      on_click=lambda e, n=name: self.edit_placeholder(n)),
                        ft.TextButton(content="↺ Reset", key=f"ix-ph-reset-{name}", visible=edited,
                                      on_click=lambda e, n=name: self.set_placeholder(n, core.placeholder_default(n)))],
                       spacing=0),
            ], spacing=1, tight=True), padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                bgcolor=ft.Colors.SURFACE_CONTAINER, border_radius=tokens.RADII["card"], key=f"ix-ph-{name}"))
        self.placeholder_rows.controls = rows
        self.ctx.push(self.placeholder_rows)

    def set_placeholder(self, name: str, text: str) -> Optional[str]:
        """Store a placeholder text through the shared setter (the desktop's Placeholder Text window does the same)."""
        core = _core()
        config = self.ctx.config_snapshot()
        core.set_placeholder_text(config, name, text)
        key = core.REFERENCE_LINE_KEY if name == "reference_line" else core.PLACEHOLDER_TEXTS_KEY
        self.ctx.set_cfg(key, config.get(key))
        self.render_placeholders()
        return None

    def _links(self, names: Any, field: Optional[str]) -> Any:
        """The placeholders of an editor like links (``services.prompt_placeholders``): the shared
        ``describe_placeholder`` is what a hold shows (tooltip text only, never sent to a model)."""
        from glossarion_mobile.services.prompt_placeholders import PlaceholderLinks

        core = _core()
        return PlaceholderLinks(
            frozenset(str(n).strip("{}") for n in names),
            lambda name, _text: core.describe_placeholder(name, self.ctx.config_snapshot(), None, field))

    def edit_placeholder(self, name: str) -> Any:
        from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

        settings = getattr(self.ctx, "settings", None)
        if settings is None:
            self.ctx.say("The prompt editor needs the Settings context")
            return None
        core = _core()
        names = tuple("{" + n + "}" for n in core.PLACEHOLDER_TEXT_TOKENS.get(name, ()))
        editor = PromptEditorSheet(
            settings, title="{" + name + "}", subtitle="Placeholder text · Interactive Images · shared with the desktop",
            value=core.placeholder_text(self.ctx.config_snapshot(), name), default=core.placeholder_default(name),
            placeholders=names, on_save=lambda text, n=name: self.set_placeholder(n, text),
            links=self._links(core.PLACEHOLDER_TEXT_TOKENS.get(name, ()), None))
        self.editor = editor
        editor.show()
        return editor

    @property
    def profiles(self) -> Any:
        return self.bar.profiles if self.bar is not None else None

    def render_prompts(self) -> None:
        from glossarion_mobile.ui.screens.profiles import ProfileList, profile_prompt_card
        from glossarion_mobile.ui.theme import mono_family

        profiles = self.profiles
        name = self.bar.selected if self.bar is not None else ""
        texts = profiles.texts if profiles is not None else {}
        builtin = profiles.builtin_texts(name) if profiles is not None else None
        self.per_chapter.value = str(_core().parse_per_chapter(texts.get("per_chapter", "1")))
        self.per_chapter.disabled = profiles is None
        self.art_style.value = str(texts.get("art_style", "") or "")
        self.art_style.disabled = profiles is None
        controls = []
        for field_name, heading, _names, key in PROMPT_FIELDS:
            listing = ProfileList(names=[name], texts={name: str(texts.get(field_name, "") or "")},
                                  defaults={name: builtin[field_name]} if builtin else {},
                                  protected=frozenset([name]) if builtin else frozenset())
            controls.append(profile_prompt_card(
                listing, name, heading=heading, missing=profiles is None,
                on_edit=lambda f=field_name: self.edit_prompt(f), on_new=self.bar.new if self.bar else None,
                disabled_reason=self.bar.error if self.bar else None,
                mono=mono_family(getattr(self.ctx, "page", None)), key=key))
        self.prompt_cards.controls = controls
        self.ctx.push(self.prompt_cards, self.per_chapter, self.art_style)

    def _stage(self, field_name: str, value: str) -> Optional[str]:
        profiles = self.profiles
        if profiles is None:
            return (self.bar.error if self.bar else None) or "Interactive Images profiles are unavailable"
        texts = dict(profiles.texts)
        texts[field_name] = str(value or "")
        profiles.stage(self.bar.selected, texts)
        if not self.bar._persist():
            return "Could not save the Interactive Images profile. Please try again."
        self.render_prompts()
        return None

    def set_per_chapter(self, value: Any) -> Optional[str]:
        return self._stage("per_chapter", str(_core().parse_per_chapter(value)))

    def set_art_style(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if profiles_text(self.profiles).get("art_style", "") == text:
            return None
        return self._stage("art_style", text)

    def edit_prompt(self, field_name: str) -> Any:
        from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

        profiles = self.profiles
        settings = getattr(self.ctx, "settings", None)
        if profiles is None or settings is None:
            self.ctx.say((self.bar.error if self.bar else None) or "The prompt editor needs the Settings context")
            return None
        core = _core()
        row = next(r for r in PROMPT_FIELDS if r[0] == field_name)
        names = tuple("{" + n + "}" for n in getattr(core, row[2], ())) if row[2] else ()
        name = self.bar.selected
        builtin = profiles.builtin_texts(name) or {}
        editor = PromptEditorSheet(
            settings, title=name, subtitle=f"{row[1]} · Interactive Images profile · shared with the desktop",
            value=str(profiles.texts.get(field_name, "") or ""), default=builtin.get(field_name),
            placeholders=names, on_save=lambda text, f=field_name: self._stage(f, text),
            links=self._links(getattr(core, row[2], ()) if row[2] else (), field_name))
        self.editor = editor
        editor.show()
        return editor

    # ---- Models -------------------------------------------------------------------------------------

    def _models_tab(self) -> ft.Control:
        from glossarion_mobile.ui.sheets.model_sheet import ModelPicker, sheet_env

        core = _core()
        values = core.settings_values(self.ctx.config_snapshot())
        env = sheet_env()
        self.text_model = ModelPicker(value=values["interactive_images_text_model"], label="Text model (empty = main)",
                                      env=env, on_change=lambda v: self.set_value("interactive_images_text_model", v),
                                      key="ix-text-model")
        self.image_model = ModelPicker(value=values["interactive_images_image_model"], label="Image model (empty: pool model, else the text model)", env=env,
                                       on_change=self._on_image_model, key="ix-image-model")
        self.ref_note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="ix-ref-note")
        self.pool_note = hint_text("", key="ix-pool-note")
        pool_tiles, _pool_map = schema_tiles(self.ctx, ("use_inpainter_keys", "inpainter_keys"))

        def dropdown(key: str, label: str, choices: Sequence[str], value: Any, width: int = 150) -> ft.Dropdown:
            return ft.Dropdown(label=label, dense=True, width=width, value=str(value), key=f"ix-{key}",
                               options=[ft.DropdownOption(key=str(c), text=str(c)) for c in choices],
                               on_select=lambda e, k=key: self.set_value(k, e.control.value))

        self.max_refs = dropdown("interactive_images_max_references", "Max reference images",
                                 [str(n) for n in range(0, core.MAX_REFERENCES + 1)],
                                 values["interactive_images_max_references"], 190)
        self.resolution = dropdown("interactive_images_resolution", "Resolution", core.RESOLUTIONS,
                                   values["interactive_images_resolution"])
        self.aspect = dropdown("interactive_images_aspect_ratio", "Aspect ratio", core.ASPECT_RATIOS,
                               values["interactive_images_aspect_ratio"])
        # Auto: the ratios an image may get, ticked as chips (shown while the ratio is Auto)
        self.ratio_row = ft.Row(wrap=True, spacing=6, run_spacing=4, key="ix-ratio-chips")
        self.ratio_hint = hint_text("Auto: each image gets one of the ticked ratios, picked at random (the same "
                                    "ratio for a chapter's image number on every rerun).", key="ix-ratio-hint")
        self._render_ratios()
        switches = []
        for key, label in (("interactive_images_auto_after_translation", "Run after each translation"),
                           ("interactive_images_recompile_after",
                            "Recompile the EPUB/PDF afterwards (if it was compiled)")):
            switch = toggle(value=bool(values[key]), key=f"ix-{key}",
                            on_change=lambda e, k=key: self.set_value(k, bool(e.control.value)))
            switches.append(switch_row(switch, label))
        self._render_model_notes()
        return self._page([
            card("Models", [
                self.text_model,
                ft.Row([ft.TextButton(content="Use the main model", icon=ft.Icons.RESTART_ALT, key="ix-text-main",
                                      on_click=lambda e: self.text_model.set_value(""))], wrap=True),
                self.image_model, self.ref_note,
            ], icon="SMART_TOY", key="ix-models-card"),
            card("Image keys", [
                hint_text("The image requests use the Image Gen / Edit key pool when it is on (desktop 🔑 Image Keys). "
                      "With no image model set, the pool's model draws; with no pool, the text model does "
                      "(when it is not known to be text-only)."),
                *pool_tiles, self.pool_note,
                ft.Row([ft.TextButton(content="🔑 Image keys", icon=ft.Icons.KEY, key="ix-image-keys",
                                      on_click=lambda e: self.ctx.go("settings.keys.pool", {"pool": "inpainter"}))]),
            ], icon="KEY", key="ix-keys-card"),
            card("Images", [ft.Row([self.max_refs, self.resolution, self.aspect], wrap=True, spacing=8),
                            self.ratio_hint, self.ratio_row],
                 icon="IMAGE", key="ix-image-card"),
            card("Run", switches, icon="PLAY_CIRCLE", key="ix-run-card"),
        ], "ix-models")

    def set_value(self, key: str, value: Any) -> Any:
        value = _core().coerce_setting(key, value)
        self.ctx.set_cfg(key, value)
        if key == "interactive_images_image_model":
            self._render_model_notes()
        elif key in ("interactive_images_aspect_ratio", "interactive_images_aspect_choices"):
            self._render_ratios()
        return value

    def _render_ratios(self) -> None:
        """The Auto ratio chips: ticked = a ratio an image may get; hidden unless the aspect ratio is Auto."""
        core = _core()
        config = self.ctx.config_snapshot()
        values = core.settings_values(config)
        ticked = values["interactive_images_aspect_choices"]
        auto = values["interactive_images_aspect_ratio"] == "auto"
        self.ratio_row.controls = [
            ft.Chip(label=ft.Text(ratio), selected=ratio in ticked, key=f"ix-ratio-{ratio}",
                    on_click=lambda e, r=ratio: self.toggle_ratio(r)) for ratio in core.ASPECT_CHOICES]
        self.ratio_row.visible = auto
        self.ratio_hint.visible = auto
        self.ctx.push(self.ratio_row, self.ratio_hint)

    def toggle_ratio(self, ratio: str) -> list:
        """Tick / untick one ratio of the Auto pool (stored as ``interactive_images_aspect_choices``)."""
        core = _core()
        ticked = list(core.settings_values(self.ctx.config_snapshot())["interactive_images_aspect_choices"])
        if ratio in ticked:
            ticked.remove(ratio)
        else:
            ticked.append(ratio)
        return self.set_value("interactive_images_aspect_choices", ticked)

    def _on_image_model(self, value: str) -> None:
        self.set_value("interactive_images_image_model", value)

    def _render_model_notes(self) -> None:
        import interactive_images_runner as runner

        core = _core()
        config = self.ctx.config_snapshot()
        model = str(config.get("interactive_images_image_model") or "") or runner.image_pool_model(config)
        level, text = core.reference_note(model)
        self.ref_note.value = (("⚠ " if level == "warning" else "ℹ ") + text) if text else ""
        self.ref_note.color = ft.Colors.ERROR if level == "warning" else ft.Colors.ON_SURFACE_VARIANT
        self.ref_note.visible = bool(text)
        keys = [k for k in (config.get("inpainter_keys") or ()) if isinstance(k, dict)]
        on = bool(config.get("use_inpainter_keys"))
        pool_model = runner.image_pool_model(config)
        self.pool_note.value = (f"{len(keys)} key(s) in the pool · {'on' if on else 'off'}"
                                + (f" · first model: {pool_model}" if pool_model else ""))
        self.ctx.push(self.ref_note, self.pool_note)

    # ---- References ---------------------------------------------------------------------------------

    def _references_tab(self) -> ft.Control:
        self.refs_list = ft.Column(spacing=6, tight=True, key="ix-refs")
        self.style_list = ft.Column(spacing=4, tight=True, key="ix-style-refs")
        self.name_field = ft.TextField(label="Character name", dense=True, expand=True, key="ix-ref-name")
        self.refs_note = hint_text("", key="ix-refs-note")
        self.glossary_note = hint_text("", key="ix-refs-glossary")
        self.glossary_names: tuple = ()
        self.glossary_file: Optional[str] = None
        controls = [
            card("Characters", [
                hint_text("Images are copied into Interactive_Images/references/ of the book. The scene's characters "
                          "fill the reference slots first, then the art-style images."),
                self.glossary_note,
                self.refs_note,
                ft.Row([self.name_field,
                        ft.IconButton(icon=ft.Icons.ADD, tooltip="+ Character", size_constraints=HIT_TARGET,
                                      on_click=lambda e: self.ctx.spawn(self.add_character()), key="ix-ref-add")],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.refs_list,
            ], icon="FACE", key="ix-refs-card"),
            card("Art style", [
                self.style_list,
                ft.Row([ft.FilledTonalButton(content="Add art-style image…", icon=ft.Icons.ADD_PHOTO_ALTERNATE,
                                             on_click=lambda e: self.ctx.spawn(self.add_images(None)),
                                             key="ix-style-add")], wrap=True),
            ], icon="PALETTE", key="ix-style-card"),
        ]
        self._render_refs()
        return self._page(controls, "ix-references")

    async def load_references(self) -> dict:
        folder = self.folder
        core = _core()
        self.refs = await self.ctx.io(core.load_references, folder) if folder else core.empty_references()

        def glossary_characters():
            path = self.glossary_in_use()
            return path, tuple(core.character_names(core.load_glossary_entries(folder, path))) if path else ()

        self.glossary_file, self.glossary_names = await self.ctx.io(glossary_characters) if folder else (None, ())
        self._render_refs()
        return self.refs

    def _render_refs(self) -> None:
        core = _core()
        folder = self.folder
        self.refs_note.value = "" if folder else "Choose a translated book first."
        self.refs_note.visible = not folder
        rows = []
        if folder:
            self.glossary_note.value = (
                f"Characters from the glossary translation uses: {self.glossary_file}" if self.glossary_file
                else "No glossary in use for this book (translation uses none).")
        self.glossary_note.visible = bool(folder)
        for name, paths, in_glossary in core.reference_rows_with_glossary(self.refs, self.glossary_names):
            files = ", ".join(os.path.basename(p) for p in paths) or "no images yet"
            if not in_glossary:
                files += " · not in glossary"
            rows.append(ft.Container(content=ft.Row([
                ft.Column([ft.Text(name, weight=ft.FontWeight.W_600),
                           ft.Text(f"{len(paths)}: {files}", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                   color=ft.Colors.ON_SURFACE_VARIANT, max_lines=2,
                                   overflow=ft.TextOverflow.ELLIPSIS)], spacing=1, tight=True, expand=True),
                ft.IconButton(icon=ft.Icons.ADD_PHOTO_ALTERNATE, tooltip="Add…", size_constraints=HIT_TARGET,
                              on_click=lambda e, n=name: self.ctx.spawn(self.add_images(n)), key=f"ix-ref-img-{name}"),
                ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Clear", size_constraints=HIT_TARGET,
                              on_click=lambda e, n=name: self.ctx.spawn(self.clear_character(n)),
                              key=f"ix-ref-clear-{name}"),
            ], spacing=4), padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                bgcolor=ft.Colors.SURFACE_CONTAINER, border_radius=tokens.RADII["card"], key=f"ix-ref-{name}"))
        self.refs_list.controls = rows or [hint_text("No characters yet.", key="ix-refs-empty")]
        styles = [ft.Row([ft.Text(os.path.basename(p), expand=True),
                          ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Remove", size_constraints=HIT_TARGET,
                                        on_click=lambda e, r=p: self.ctx.spawn(self.remove_style(r)),
                                        key=f"ix-style-rm-{os.path.basename(p)}")])
                  for p in (self.refs.get("style") or ())]
        self.style_list.controls = styles or [hint_text("No art-style images.", key="ix-style-empty")]
        self.ctx.push(self.refs_list, self.style_list, self.refs_note, self.glossary_note)

    def _need_book(self) -> bool:
        if self.folder:
            return True
        self.ctx.say("Choose a translated book first")
        return False

    async def _save_refs(self) -> None:
        refs = copy.deepcopy(self.refs)
        await self.ctx.io(_core().save_references, self.folder, refs)
        self._render_refs()

    async def add_character(self) -> int:
        name = str(self.name_field.value or "")
        if not name.strip() or not self._need_book():
            return 0
        added = _core().add_reference_names(self.refs, [name])
        self.name_field.value = ""
        self.ctx.push(self.name_field)
        await self._save_refs()
        return added

    def glossary_in_use(self) -> Optional[str]:
        """Blocking: the glossary translation uses for this book (the Glossary Mode / auto-glossary output,
        resolved by the shared ``RunEnvMixin._resolve_book_glossary``), or None."""
        core = _core()
        return core.find_book_glossary(self.folder, core.glossary_resolver(self.ctx.config_snapshot())) \
            if self.folder else None

    async def add_images(self, name: Optional[str]) -> list:
        if not self._need_book():
            return []
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return []
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=IMAGE_EXTENSIONS,
                                            dialog_title="Select reference images")
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return []
        paths = [getattr(f, "path", None) for f in picked or () if getattr(f, "path", None)]
        return await self.add_image_paths(name, paths)

    async def add_image_paths(self, name: Optional[str], paths: Sequence[str]) -> list:
        core = _core()
        added, refused = [], []
        for path in paths:
            ok = await self.ctx.io(core.is_reference_image, path)
            if not ok:
                refused.append(os.path.basename(path))
                continue
            try:
                added.append(await self.ctx.io(core.add_reference_image, self.folder, self.refs, path, name))
            except OSError as exc:
                self.ctx.say(f"Could not copy {os.path.basename(path)}: {exc}")
        if refused:
            self.ctx.say("Not a PNG, JPEG or WebP image: " + ", ".join(refused[:3]))
        await self._save_refs()
        return added

    async def clear_character(self, name: str) -> None:
        await self.ctx.io(_core().clear_reference_character, self.folder, self.refs, name)
        await self._save_refs()

    async def remove_style(self, rel: str) -> None:
        await self.ctx.io(_core().remove_reference_image, self.folder, self.refs, rel)
        await self._save_refs()

    # ---- Styles -------------------------------------------------------------------------------------

    @property
    def style_name(self) -> str:
        return _presets().active_preset_name(self.ctx.config_snapshot())

    def _css_for(self, name: str) -> str:
        return _presets().preset_css(self.ctx.config_snapshot(), name) or ""

    def _styles_tab(self) -> ft.Control:
        presets = _presets()
        self.style_dropdown = ft.Dropdown(label="Style", dense=True, expand=True, key="ix-style",
                                          on_select=lambda e: self.select_style(e.control.value))
        self.style_warnings = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.TERTIARY,
                                      visible=False, key="ix-style-warnings")
        self.css_field = ft.TextField(multiline=True, min_lines=8, max_lines=18, dense=True, visible=False,
                                      text_size=12, text_style=ft.TextStyle(font_family=getattr(self.ctx, "mono", None)),
                                      on_change=lambda e: self.preview.update_css(e.control.value or ""),
                                      key="ix-css-editor")
        self.save_style_button = ft.TextButton(content="💾 Save", key="ix-style-save",
                                               on_click=lambda e: self.ctx.spawn(self.save_style()))
        self.delete_style_button = ft.TextButton(content="🗑 Delete", key="ix-style-delete",
                                                 on_click=lambda e: self.ctx.spawn(self.delete_style()))
        self.prompt_button = ft.TextButton(content="✨ New from prompt…", key="ix-style-prompt",
                                           on_click=lambda e: self.ctx.spawn(self.new_from_prompt()))
        actions = ft.Row([
            ft.TextButton(content="⧉ Duplicate", key="ix-style-duplicate", on_click=lambda e: self.duplicate_style()),
            self.prompt_button,
            ft.TextButton(content="✏ Edit CSS", key="ix-style-edit", on_click=lambda e: self.toggle_editor()),
            self.save_style_button, self.delete_style_button,
            ft.TextButton(content="📌 Apply to book", key="ix-style-apply",
                          on_click=lambda e: self.ctx.spawn(self.apply_to_book())),
        ], wrap=True, spacing=0)
        enabled = toggle(value=bool(self.ctx.config_snapshot().get(presets.ENABLED_KEY, False)),
                         key="ix-css-enabled",
                         on_change=lambda e: self.ctx.set_cfg(presets.ENABLED_KEY, bool(e.control.value)))
        self._fill_styles(self.style_name)
        return self._page([
            card("Interactive styles", [
                hint_text("CSS-only styles for the translated chapters (EPUB readers strip scripts). Every rule is "
                          "scoped under a .glx- class; the book's stylesheet is glossarion_interactive.css, linked "
                          "after the CSS override."),
                switch_row(enabled, "Interactive styles (link the stylesheet, mark chapters with glx-book)"),
                ft.Row([self.style_dropdown], key="ix-style-row"), actions, self.style_warnings, self.css_field,
            ], icon="PALETTE", key="ix-styles-card"),
            card("Preview", [self.preview.control], icon="PREVIEW", key="ix-preview-card"),
        ], "ix-styles")

    def _fill_styles(self, selected: str) -> None:
        presets = _presets()
        config = self.ctx.config_snapshot()
        names = presets.preset_names(config)
        self.style_dropdown.options = [ft.DropdownOption(key=n, text=n) for n in names]
        self.style_dropdown.value = selected if selected in names else presets.DEFAULT_PRESET
        builtin = presets.is_builtin_preset(self.style_dropdown.value)
        self.css_field.value = presets.preset_css(config, self.style_dropdown.value) or ""
        self.css_field.read_only = builtin
        self.save_style_button.disabled = builtin
        self.delete_style_button.disabled = builtin
        self.ctx.push(self.style_dropdown, self.css_field, self.save_style_button, self.delete_style_button)

    def _show_warnings(self, warnings: Sequence[str]) -> None:
        warnings = list(warnings or ())
        self.style_warnings.value = ("Adjusted: " + "; ".join(warnings[:6]) + (" …" if len(warnings) > 6 else "")
                                     if warnings else "")
        self.style_warnings.visible = bool(warnings)
        self.ctx.push(self.style_warnings)

    def select_style(self, name: Any) -> str:
        presets = _presets()
        name = str(name or presets.DEFAULT_PRESET)
        if self.ctx.config_snapshot().get(presets.PRESET_KEY) != name:
            self.ctx.set_cfg(presets.PRESET_KEY, name)
        self._fill_styles(name)
        self._show_warnings(())
        self.preview.show(self._css_for(name))
        return name

    def toggle_editor(self) -> bool:
        self.css_field.visible = not self.css_field.visible
        self.ctx.push(self.css_field)
        return self.css_field.visible

    def _store(self, name: str, css: str) -> str:
        """A user preset saved (scoped) and selected: ``save_user_preset`` on a copy of the config."""
        presets = _presets()
        config = self.ctx.config_snapshot()
        try:
            name, warnings = presets.save_user_preset(config, name, css)
        except ValueError as exc:
            self.ctx.say(str(exc))
            return ""
        self.ctx.set_cfg(presets.USER_PRESETS_KEY, config[presets.USER_PRESETS_KEY])
        self.ctx.set_cfg(presets.PRESET_KEY, name)
        self._fill_styles(name)
        self._show_warnings(warnings)
        self.preview.show(self._css_for(name))
        return name

    def duplicate_style(self) -> str:
        presets = _presets()
        base = str(self.style_dropdown.value or presets.DEFAULT_PRESET)
        name = presets.duplicate_name(self.ctx.config_snapshot(), base)
        stored = self._store(name, str(self.css_field.value or "") or self._css_for(base))
        if stored and not self.css_field.visible:
            self.toggle_editor()
        return stored

    async def save_style(self) -> str:
        presets = _presets()
        name = str(self.style_dropdown.value or "")
        if presets.is_builtin_preset(name):
            return ""
        return self._store(name, str(self.css_field.value or ""))

    async def delete_style(self) -> bool:
        from glossarion_mobile.ui.glossary.common import ask

        presets = _presets()
        name = str(self.style_dropdown.value or "")
        if presets.is_builtin_preset(name):
            return False
        if not await ask(self.ctx, title="Delete Style", body=f"Delete the style '{name}'?", confirm="Delete",
                         cancel="Cancel", destructive=True):
            return False
        config = self.ctx.config_snapshot()
        presets.delete_user_preset(config, name)
        self.ctx.set_cfg(presets.USER_PRESETS_KEY, config.get(presets.USER_PRESETS_KEY, {}))
        self.ctx.set_cfg(presets.PRESET_KEY, config.get(presets.PRESET_KEY, presets.DEFAULT_PRESET))
        self.select_style(presets.active_preset_name(self.ctx.config_snapshot()))
        return True

    async def new_from_prompt(self) -> Optional[str]:
        from glossarion_mobile.ui.glossary.common import prompt_text

        if not self.ctx.has_kind(KIND):
            self.ctx.say("The job service is not running")
            return None
        request = await prompt_text(self.ctx, title="New Style from Prompt",
                                    label="Describe the style ({user_request} in the CSS prompt)", ok="Write")
        if not request:
            return None
        from glossarion_mobile.services.jobs import JobSpec

        spec = JobSpec(kind=KIND, title="New interactive style", inputs=(), params={"css_request": request},
                       origin={"type": "tool", "label": "Tools · Interactive Images"})
        job_id = await self.ctx.submit(spec)
        self.last_css_job = job_id
        self.watch.watch(job_id)
        self.ctx.say("Asking the text model for a stylesheet…")
        return job_id

    def store_generated(self, result: dict) -> str:
        """The finished New-from-prompt job's stylesheet as a user preset (named from the request)."""
        presets = _presets()
        name = presets.preset_name_for_request(self.ctx.config_snapshot(), result.get("request") or "")
        stored = self._store(name, str(result.get("css") or ""))
        if stored:
            self._show_warnings(result.get("warnings") or ())
            if not self.css_field.visible:
                self.toggle_editor()
        return stored

    async def apply_to_book(self) -> Optional[str]:
        presets = _presets()
        if not self._need_book():
            return None
        name = str(self.style_dropdown.value or presets.DEFAULT_PRESET)
        css = self._css_for(name) if presets.is_builtin_preset(name) else str(self.css_field.value or "")
        path, warnings = await self.ctx.io(presets.write_book_stylesheet, self.folder, css)
        self._show_warnings(warnings)
        self.ctx.say(f"Wrote {os.path.basename(path)} for this book; it is linked when the EPUB is compiled.")
        return path
