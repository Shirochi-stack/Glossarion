"""ToolsFeature: wires the U6 Tools hub, QA Scanner, QA report viewer, Converter and Headers &
metadata screens into GlossarionApp.

``await ToolsFeature.install(app)`` - call it in ``GlossarionApp.start`` after the Library
(``_install_library``: the tools pick Library books and open the Book page) and the Reader:

1. builds the ``ToolsContext`` per screen from the app (LibraryService, FileBridge,
   JobsFeature, Prefs, the Settings feature's ``SettingsContext`` and config store, the URL
   launcher, the WebView probe, the app folders) and exposes the feature as ``app.tools``;
2. wraps the shell's screen factory for ``tools``, ``tools.qa``, ``tools.qa.report``,
   ``tools.convert`` and ``tools.headers`` (every other route falls through, so the U5
   Progress manager screens and the later tools keep their own factories).

The job kinds behind the screens (``qa_scan``, ``validate_epub``, ``rename_outputs``,
``translate_headers``, ``metadata``) are registered in ``job_kinds``; registering ``metadata``
also enables the Library's "Translate Metadata" actions (they check
``JobService.has_kind("metadata")``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from glossarion_mobile.ui.router import RouteMatch

__all__ = ["IMPLEMENTED_ROUTES", "SCREEN_ROUTES", "ToolsFeature"]

log = logging.getLogger("glossarion.tools")

SCREEN_ROUTES = ("tools", "tools.qa", "tools.qa.report", "tools.convert", "tools.headers", "tools.async",
                 "tools.review", "tools.sdlxliff", "tools.rpgmaker")
#: Routes this feature ships (Integrate merges them into the hub / drawer "implemented" sets).
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES)


class ToolsFeature:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self.tool_state: dict = {}  # per-session screen state (chosen books, mode), kept across visits
        self.screens_built: list = []

    @classmethod
    async def install(cls, app: Any) -> "ToolsFeature":
        feature = cls(app)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.tools = self
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory

    # ---- context ---------------------------------------------------------------------------------

    def _platform(self) -> str:
        platform = getattr(getattr(self.page, "platform", None), "value", None)
        return str(platform or "desktop")

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def _webview_ok(self) -> bool:
        try:
            from glossarion_mobile.ui.reader.reader_view import webview_supported

            return bool(webview_supported(self.page))
        except Exception:
            return False

    def _open_url(self, url: str) -> Any:
        launcher = getattr(self.app, "url_launcher", None)
        if launcher is None:
            return None
        return launcher.launch_url(url)

    def _push_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is not None:
            shell.push_overlay(view)
            try:
                self.page.update()
            except Exception:
                pass

    def _pop_overlay(self) -> None:
        back = getattr(self.app, "back", None)
        if callable(back):
            back()

    def context(self) -> Any:
        from glossarion_mobile.ui import tokens
        from glossarion_mobile.ui.tools.common import ToolsContext

        app = self.app
        shell = getattr(app, "shell", None)
        state = getattr(app, "state", None)
        try:
            scale = float(state.text_scale.value) if state is not None else 1.0
        except Exception:
            scale = 1.0
        paths = getattr(app, "paths", None)
        data_dir = str(getattr(paths, "data", "") or "") if paths is not None else ""
        output_root = str(getattr(paths, "output", "") or "") if paths is not None else ""
        output_root = output_root or os.environ.get("OUTPUT_DIRECTORY", "")
        settings_feature = getattr(app, "settings", None)
        settings_ctx = getattr(settings_feature, "ctx", None)
        import_dir = ""
        if settings_ctx is not None:
            import_dir = str((getattr(settings_ctx, "extras", {}) or {}).get("import_dir") or "")
        library_feature = getattr(app, "library_feature", None)
        ctx = ToolsContext(
            service=getattr(app, "library", None),
            page=self.page,
            dispatcher=self.dispatcher,
            navigate=getattr(app, "navigate_to", None),
            notify=getattr(app, "notify", None),
            files=getattr(app, "files", None),
            jobs=getattr(app, "jobs", None),
            prefs=getattr(app, "prefs", None),
            haptics=getattr(app, "haptics", None),
            shell=shell,
            intents=getattr(app, "intents", None),
            copy_text=getattr(app, "_copy_text", None),
            reader=lambda: getattr(app, "reader", None),
            push_overlay=self._push_overlay,
            pop_overlay=self._pop_overlay,
            foreground=getattr(library_feature, "foreground", lambda: True),
            platform=self._platform(),
            tablet=bool(getattr(shell, "tablet", False)),
            dark=self._dark(),
            text_scale=scale,
            settings=settings_ctx,
            store=getattr(app, "config_store", None),
            open_url=self._open_url,
            webview_ok=self._webview_ok,
            data_dir=data_dir,
            output_root=output_root,
            chats_root=os.path.join(output_root, "Direct Text") if output_root else "",
            import_dir=import_dir or (os.path.join(data_dir, "imports") if data_dir else ""),
            tool_state=self.tool_state,
        )
        try:
            from glossarion_mobile.ui.theme import mono_family

            ctx.mono = mono_family(self.page)  # type: ignore[attr-defined]
        except Exception:
            ctx.mono = tokens.MONO_FAMILIES.get(self._platform(), "monospace")  # type: ignore[attr-defined]
        return ctx

    # ---- screens ---------------------------------------------------------------------------------

    def implemented_routes(self) -> frozenset:
        """Routes with a real screen in this session (this feature + the Library's progress tools)."""
        routes = set(IMPLEMENTED_ROUTES)
        try:
            from glossarion_mobile.ui.library.feature import IMPLEMENTED_ROUTES as LIBRARY_ROUTES

            if getattr(self.app, "library_feature", None) is not None:
                routes |= set(LIBRARY_ROUTES)
        except Exception:
            pass
        if getattr(self.app, "jobs", None) is not None:
            routes |= {"tools.files", "tools.files.folder"}
        if getattr(self.app, "manga", None) is not None:  # U8 MangaFeature
            try:
                from glossarion_mobile.ui.tools.manga.feature import IMPLEMENTED_ROUTES as MANGA_ROUTES

                routes |= set(MANGA_ROUTES)
            except Exception:
                pass
        return frozenset(routes)

    def make_screen(self, match: RouteMatch) -> Any:
        name = match.name
        ctx = self.context()
        if name == "tools":
            from glossarion_mobile.ui.tools.hub import ToolsHubScreen

            return ToolsHubScreen(match, ctx, implemented=self.implemented_routes())
        if name == "tools.qa":
            from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

            return QaScannerScreen(match, ctx)
        if name == "tools.qa.report":
            from glossarion_mobile.ui.tools.qa_report import QaReportScreen

            return QaReportScreen(match, ctx)
        if name == "tools.convert":
            from glossarion_mobile.ui.tools.converter import ConverterScreen

            return ConverterScreen(match, ctx)
        if name == "tools.headers":
            from glossarion_mobile.ui.tools.headers_screen import HeadersScreen

            return HeadersScreen(match, ctx)
        if name == "tools.async":
            from glossarion_mobile.ui.tools.async_batch import AsyncBatchScreen

            return AsyncBatchScreen(match, ctx)
        if name == "tools.review":
            from glossarion_mobile.ui.tools.review import ReviewScreen

            return ReviewScreen(match, ctx)
        if name == "tools.sdlxliff":
            from glossarion_mobile.ui.tools.sdlxliff import SdlxliffScreen

            return SdlxliffScreen(match, ctx)
        if name == "tools.rpgmaker":
            from glossarion_mobile.ui.tools.rpgmaker import RpgMakerScreen

            return RpgMakerScreen(match, ctx)
        return None

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        if match.name in SCREEN_ROUTES:
            try:
                screen = self.make_screen(match)
            except Exception:
                log.exception("building the %s screen failed", match.name)
                screen = None
            if screen is not None:
                self.screens_built.append(match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen

    # ---- plumbing ----------------------------------------------------------------------------------

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None
