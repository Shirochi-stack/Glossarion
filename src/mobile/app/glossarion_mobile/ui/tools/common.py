"""Shared plumbing for the Tools screens: ``ToolsContext`` and a few small widgets.

``ToolsContext`` is the Library's ``LibraryContext`` (navigation, snackbars, FileBridge,
JobsFeature, Prefs, haptics, Reader access - the Tools rows are mostly Library books) plus
what only the tools need: the schema ``SettingsContext`` (option tiles render with the
Settings tiles), the config store, ``open_url`` (desktop dev "Open in browser"), the
WebView probe and the app folders. Host tests build it directly with fakes.

``JobWatch`` follows the jobs a tool screen started: it subscribes to
``JobService.on_transition`` (delivered on the UI loop) and calls back once per job when
it ends, so a screen can show its result card.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library.common import LibraryContext
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["ChoiceDialog", "JobWatch", "ToolsContext", "action_button", "ask", "card", "hint_text", "kv_line",
           "schema_tiles", "section_header"]

log = logging.getLogger("glossarion.tools")


@dataclass
class ToolsContext(LibraryContext):
    settings: Any = None  # ui.settings.context.SettingsContext (schema tiles)
    store: Any = None  # MobileConfigStore (or a dict in tests)
    open_url: Optional[Callable[[str], Any]] = None
    webview_ok: Callable[[], bool] = lambda: False
    data_dir: str = ""
    output_root: str = ""
    chats_root: str = ""
    import_dir: str = ""
    tool_state: dict = field(default_factory=dict)  # per-session screen state (selected targets…)

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        service = self.service
        if service is not None and hasattr(service, "io"):
            return await service.io(fn, *args)
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-tools")
        return await asyncio.to_thread(fn, *args)

    # ---- config --------------------------------------------------------------------------------

    def cfg(self, key: Any, default: Any = None) -> Any:
        store = self.store
        if store is None:
            service = self.service
            getter = getattr(service, "cfg", None) if service is not None else None
            return getter(key, default) if callable(getter) and isinstance(key, str) else default
        try:
            if isinstance(store, dict) and isinstance(key, tuple):
                value: Any = store
                for part in key:
                    if not isinstance(value, dict) or part not in value:
                        return default
                    value = value[part]
            else:
                value = store.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def set_cfg(self, key: Any, value: Any) -> None:
        store = self.store
        if store is None:
            setter = getattr(self.service, "set_cfg", None) if self.service is not None else None
            if callable(setter) and isinstance(key, str):
                setter(key, value)
            return
        try:
            if isinstance(store, dict):
                if isinstance(key, tuple):
                    node = store
                    for part in key[:-1]:
                        node = node.setdefault(part, {})
                    node[key[-1]] = value
                else:
                    store[key] = value
            else:
                store.set(key, value)
        except Exception:
            log.exception("saving %s failed", key)

    def config_snapshot(self) -> dict:
        store = self.store
        if store is None:
            snap = getattr(self.service, "config_snapshot", None) if self.service is not None else None
            return dict(snap() or {}) if callable(snap) else {}
        if isinstance(store, dict):
            return dict(store)
        try:
            return dict(store.snapshot() or {})
        except Exception:
            return {}

    # ---- jobs ------------------------------------------------------------------------------

    def has_kind(self, kind: str) -> bool:
        jobs = self.jobs
        checker = getattr(jobs, "has_kind", None) if jobs is not None else None
        if not callable(checker):
            return False
        try:
            return bool(checker(kind))
        except Exception:
            return False

    async def submit(self, spec: Any) -> Optional[str]:
        jobs = self.jobs
        if jobs is None:
            self.say("The job service is not running")
            return None
        result = jobs.submit(spec)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        if spec.kind in ("compile_epub", "compile_pdf"):
            compiling = getattr(self.service, "compiling", None)
            if isinstance(compiling, set):
                for path in spec.inputs:
                    compiling.add(os.path.normcase(os.path.abspath(path)))
        return result

    def remember_source(self, tool: str, label: str) -> None:
        """Hub tile subtitle: the last source a tool ran on (``mobile_state.json``)."""
        prefs = self.prefs
        if prefs is None or not label:
            return
        try:
            current = dict(prefs.get("tools_last_sources", {}) or {})
            current[tool] = str(label)[:120]
            prefs.set("tools_last_sources", current)
        except Exception:
            log.debug("saving the last tool source failed", exc_info=True)

    def last_source(self, tool: str) -> str:
        prefs = self.prefs
        if prefs is None:
            return ""
        try:
            return str((prefs.get("tools_last_sources", {}) or {}).get(tool) or "")
        except Exception:
            return ""

    def bid_for_folder(self, folder: str) -> Optional[str]:
        """The Library route id of the book whose output folder is ``folder`` (synthesised if unscanned)."""
        service = self.service
        if service is None or not folder:
            return None
        key = os.path.normcase(os.path.abspath(folder))
        try:
            books = service.snapshot.all_books()
        except Exception:
            books = ()
        for book in books or ():
            out = str(book.get("output_folder") or "")
            if out and os.path.normcase(os.path.abspath(out)) == key:
                return service.bid_for(book)
        synthesise = getattr(type(service), "_synthesise", None)
        row = synthesise(folder) if callable(synthesise) else None
        if row is None:
            return None
        return service.remember(row) if hasattr(service, "remember") else service.bid_for(row)


class JobWatch:
    """Calls ``on_end(snapshot)`` once for each watched job when it reaches a terminal state."""

    def __init__(self, ctx: ToolsContext, on_end: Callable[[Any], Any],
                 on_change: Optional[Callable[[Any], Any]] = None) -> None:
        self.ctx = ctx
        self.on_end = on_end
        self.on_change = on_change
        self.job_ids: set = set()
        self.ended: dict = {}
        self._unsubs: list = []

    def watch(self, job_id: Optional[str]) -> None:
        if job_id:
            self.job_ids.add(job_id)
            self.start()

    def start(self) -> None:
        if self._unsubs:
            return
        jobs = self.ctx.jobs
        on_transition = getattr(jobs, "on_transition", None) if jobs is not None else None
        if callable(on_transition):
            try:
                self._unsubs.append(on_transition(self._on_transition))
            except Exception:
                log.debug("subscribing to job transitions failed", exc_info=True)

    def stop(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    def _on_transition(self, snap: Any, previous: Any) -> None:
        job_id = getattr(snap, "id", None)
        if job_id not in self.job_ids:
            return
        if self.on_change is not None:
            try:
                self.on_change(snap)
            except Exception:
                log.exception("tool job change handler failed")
        if getattr(snap, "is_terminal", False) and job_id not in self.ended:
            self.ended[job_id] = snap
            try:
                self.on_end(snap)
            except Exception:
                log.exception("tool job end handler failed")

    def adopt(self, kinds: Sequence[str], match: Optional[Callable[[Any], bool]] = None) -> list:
        """Watch the queued / running jobs of ``kinds`` (``match(snapshot)`` narrows them) that another
        instance of the screen started: a reopened tool screen keeps its Stop and disabled actions.
        Returns their snapshots, the running job first."""
        jobs = self.ctx.jobs
        view_of = getattr(jobs, "view", None) if jobs is not None else None
        if not callable(view_of):
            return []
        try:
            view = view_of()
        except Exception:
            log.debug("reading the job queue failed", exc_info=True)
            return []
        found = []
        for snap in (getattr(view, "active", None), *tuple(getattr(view, "queue", ()) or ())):
            if snap is None or getattr(snap, "is_terminal", True) or getattr(snap, "kind", "") not in kinds:
                continue
            if match is not None:
                try:
                    if not match(snap):
                        continue
                except Exception:
                    continue
            self.watch(getattr(snap, "id", None))
            found.append(snap)
        return found

    def active(self) -> Optional[Any]:
        """The watched job still queued or running (from the JobService)."""
        jobs = self.ctx.jobs
        snapshot = getattr(jobs, "snapshot", None) if jobs is not None else None
        if not callable(snapshot):
            return None
        for job_id in list(self.job_ids):
            if job_id in self.ended:
                continue
            try:
                snap = snapshot(job_id)
            except Exception:
                snap = None
            if snap is not None and not getattr(snap, "is_terminal", True):
                return snap
        return None


# ---- widgets -------------------------------------------------------------------------------------


def section_header(text: str, *, key: Optional[str] = None) -> ft.Text:
    return ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                   weight=ft.FontWeight.W_600, key=key)


def hint_text(text: str, *, key: Optional[str] = None, color: Any = None) -> ft.Text:
    return ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=color or ft.Colors.ON_SURFACE_VARIANT,
                   key=key)


def kv_line(label: str, value: str, *, key: Optional[str] = None) -> ft.Text:
    return ft.Text(spans=[ft.TextSpan(f"{label}: ", style=ft.TextStyle(weight=ft.FontWeight.W_600)),
                          ft.TextSpan(value)], theme_style=ft.TextThemeStyle.BODY_SMALL, key=key,
                   max_lines=3, overflow=ft.TextOverflow.ELLIPSIS)


def card(title: str, controls: Sequence[ft.Control], *, icon: Any = None, key: Optional[str] = None,
         trailing: Optional[ft.Control] = None, subtitle: Optional[str] = None) -> ft.Container:
    """Tonal group card (the SectionCard look, built inline so it can be updated in place)."""
    header: list[ft.Control] = []
    if icon is not None:
        header.append(ft.Icon(icon_data(icon), color=ft.Colors.PRIMARY, size=20))
    texts: list[ft.Control] = [section_header(title)]
    if subtitle:
        texts.append(hint_text(subtitle))
    header.append(ft.Column(texts, spacing=2, tight=True, expand=True))
    if trailing is not None:
        header.append(trailing)
    return ft.Container(
        content=ft.Column([ft.Row(header, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                           *controls], spacing=tokens.SPACING["sm"], tight=True),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=tokens.RADII["card"],
        padding=tokens.SPACING["card_padding"],
        key=key,
    )


def action_button(label: str, icon: Any, on_click: Any, *, key: Optional[str] = None,
                  reason: Optional[str] = None, filled: bool = False, destructive: bool = False) -> ft.Control:
    """A tonal (or filled) button; unavailable actions stay visible, disabled, with a ReasonChip."""
    style = ft.ButtonStyle(color=ft.Colors.ERROR) if destructive else None
    cls = ft.FilledButton if filled else ft.FilledTonalButton
    button = cls(content=label, icon=icon_data(icon), on_click=on_click, disabled=reason is not None,
                 key=key, style=style, tooltip=reason)
    if reason is None:
        return button
    return ft.Row([button, ReasonChip(reason=_short(reason), detail=reason)], spacing=4, wrap=True,
                  key=f"{key}-row" if key else None)


def _short(reason: str) -> str:
    return reason if len(reason) <= 32 else reason[:30].rstrip() + "…"


def icon_tile_button(icon: Any, tooltip: str, on_click: Any, *, key: Optional[str] = None) -> ft.IconButton:
    return ft.IconButton(icon=icon_data(icon), tooltip=tooltip, on_click=on_click, size_constraints=HIT_TARGET,
                         key=key)


def schema_tiles(ctx: Any, keys: Sequence[str]) -> tuple:
    """Settings tiles for schema keys (the Settings pages' own tiles): ``(controls, {key: tile})``.

    Keys the schema does not know are skipped; without a SettingsContext the list is empty.
    """
    settings = getattr(ctx, "settings", None)
    schema = getattr(settings, "schema", None) if settings is not None else None
    if schema is None or not getattr(schema, "available", False):
        return [], {}
    from glossarion_mobile.ui.settings.tiles import EffectiveConfig, make_tile

    config = EffectiveConfig(settings.store)
    controls: list = []
    tiles: dict = {}
    for key in keys:
        try:
            spec = schema.spec(key)
        except Exception:
            spec = None
        if spec is None:
            continue
        try:
            tile = make_tile(spec, settings, config=config)
            tile.refresh(push=False)
        except Exception:
            log.exception("building the %s tile failed", key)
            continue
        tiles[key] = tile
        controls.append(tile.control)
    return controls, tiles


class ChoiceDialog:
    """A desktop-style question with up to three answers (Yes / No / Cancel dialogs).

    ``options``: ``(value, label, kind)`` with kind ``filled`` / ``text`` / ``destructive``.
    ``await dialog.wait()`` returns the chosen value (None when dismissed or cancelled).
    """

    def __init__(self, title: str, body: str, options: Sequence[tuple], *, key: str = "choice") -> None:
        self.title = title
        self.body = body
        self.options = list(options)
        self.choice: Optional[str] = None
        self._future: Optional[asyncio.Future] = None
        self._page: Any = None
        self.buttons: dict = {}
        actions: list[ft.Control] = []
        for value, label, kind in self.options:
            if kind == "filled":
                button: ft.Control = ft.FilledButton(content=label, on_click=lambda e, v=value: self.choose(v),
                                                     key=f"{key}-{value}")
            elif kind == "destructive":
                button = ft.FilledButton(content=label, on_click=lambda e, v=value: self.choose(v),
                                         style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR),
                                         key=f"{key}-{value}")
            else:
                button = ft.TextButton(content=label, on_click=lambda e, v=value: self.choose(v),
                                       key=f"{key}-{value}")
            self.buttons[value] = button
            actions.append(button)
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(width=tokens.SIZES["dialog_max"],
                                 content=ft.Column([ft.Text(body, selectable=True)], tight=True,
                                                   scroll=ft.ScrollMode.AUTO)),
            actions=actions,
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: self._resolve(None),
            key=key,
        )

    def show(self, page: Any) -> "ChoiceDialog":
        self._page = page
        try:
            self._future = asyncio.get_running_loop().create_future()
        except RuntimeError:
            self._future = None
        if page is not None:
            page.show_dialog(self.dialog)
        return self

    def _resolve(self, value: Optional[str]) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(value)

    def choose(self, value: Optional[str]) -> None:
        self.choice = value
        if self._page is not None and getattr(self.dialog, "open", False):
            try:
                self._page.pop_dialog()
            except Exception:
                pass
        self._resolve(value)

    async def wait(self) -> Optional[str]:
        if self._future is None:
            return self.choice
        return await self._future


async def ask(ctx: Any, title: str, body: str, options: Sequence[tuple], *, key: str = "choice") -> Optional[str]:
    """Show a ChoiceDialog and wait for the answer (tests set ``ctx.extras['answers']`` to script it)."""
    scripted = getattr(ctx, "extras", {}).get("answers") if hasattr(ctx, "extras") else None
    if isinstance(scripted, list):
        ctx.extras.setdefault("asked", []).append((title, body))
        return scripted.pop(0) if scripted else None
    dialog = ChoiceDialog(title, body, options, key=key)
    ctx.extras["last_dialog"] = dialog
    dialog.show(ctx.page)
    return await dialog.wait()
