"""Model Manager (``/settings/models``; UI_SPEC §4.11) and the Models & keys feature install.

**Models** (desktop "Manage Models" › Models): search; chips Polled only (``hide_unpolled``, the
desktop "🧪 Hide unpolled models" toggle that also filters every picker), Custom (entries that are
not in the built-in / polled catalog) and Removed (the tombstones in
``model_manager_removed_models``); "🌐 Poll providers" with the desktop status text and a
per-provider status chip row; the list in saved order (``ReorderableListView``: drag to reorder;
swipe to remove with Undo; in Removed, swipe to restore); FAB "Add model"; ⋯ Reset to defaults.
Every edit is saved at once through ``ModelCatalogService`` → ``model_catalog_core`` (the desktop
saves on "Save"; tombstones are computed the same way, so removed models never come back from a
passive catalog merge).

**Custom prefixes**: the ``custom_prefix_routes`` table (prefix → base URL → endpoint type) with an
add/edit sheet validated like the desktop table.

``ModelsKeysFeature.install(app)`` wires the catalog service, the ModelSheet environment and the
screens for ``settings.models``, ``settings.keys``, ``settings.keys.pool`` and
``settings.endpoints`` into the app (Refusal patterns and Local AI open as full-screen views).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services import model_catalog as mc
from glossarion_mobile.services.model_catalog import CatalogSnapshot, ModelCatalogService
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen, build_screen_view
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["IMPLEMENTED_ROUTES", "ModelManagerScreen", "ModelsKeysFeature", "PrefixEditor", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.models")

SCREEN_ROUTES = ("settings.models", "settings.keys", "settings.keys.pool", "settings.endpoints")
#: Settings home pages this feature ships (merged into ``SettingsHome.implemented`` once installed).
IMPLEMENTED_ROUTES = frozenset(("settings.models", "settings.keys", "settings.endpoints"))
READY_TEXT = "Ready to poll online catalogs\n✓ marks models confirmed within 7 days"
RESET_TEXT = ("Replace the current list with the built-in default model catalog?\n\n"
              "Any custom models you added will be lost.")
WHEEL_REASON = "Desktop mouse-wheel guard; touch screens have no wheel. The setting is kept."
ENDPOINT_TYPES_FALLBACK = ("/chat/completions", "/images/generations", "/v1/messages", "/v1/ocr", "/{model_id}")


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


def endpoint_type_options() -> tuple:
    try:
        from run_env import RunEnvMixin

        return tuple(RunEnvMixin._CUSTOM_PREFIX_ENDPOINT_TYPES)
    except Exception:
        return ENDPOINT_TYPES_FALLBACK


class ModelManagerScreen(Screen):
    title = "Model manager"

    def __init__(self, match: Optional[RouteMatch], *, catalog: ModelCatalogService, page: Any = None,
                 notify: Optional[Callable[..., Any]] = None, spawn: Optional[Callable[[Any], Any]] = None,
                 run_io: Optional[Callable[..., Any]] = None, tablet: bool = False) -> None:
        super().__init__(match)
        self.catalog = catalog
        self.page = page
        self.notify = notify
        self.spawn_fn = spawn
        self.run_io = run_io
        self.tablet = tablet
        self.tab = "models"
        self.query = ""
        self.filter = "all"  # all | custom | removed
        self.poll_status = READY_TEXT
        self.polling = False
        self.snapshot: CatalogSnapshot = catalog.snapshot
        self.builtin_keys: Optional[set] = None
        self._unsub: Optional[Callable[[], None]] = None
        self.last_dialog: Any = None
        self.prefix_editor: Optional[PrefixEditor] = None

    # ---- helpers --------------------------------------------------------------------------------

    def say(self, message: str, action: Optional[str] = None, on_action: Any = None) -> None:
        if self.notify is None:
            log.info("models: %s", message)
            return
        try:
            self.notify(message, action, on_action)
        except TypeError:
            self.notify(message)

    def spawn(self, coro: Any) -> Any:
        return self.spawn_fn(coro) if self.spawn_fn is not None else asyncio.ensure_future(coro)

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _edit(self, ok_error: tuple, success: Optional[str] = None) -> bool:
        ok, error = ok_error
        if not ok:
            if error:
                self.say(error if isinstance(error, str) else error[1])
            return False
        self.snapshot = self.catalog.snapshot
        if success:
            self.say(success)
        self.render()
        return True

    # ---- body -----------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.tabs = ft.SegmentedButton(
            segments=[ft.Segment(value="models", label="Models"), ft.Segment(value="prefixes", label="Custom prefixes")],
            selected=[self.tab], on_change=self._on_tab,
        )
        self.search = ft.TextField(hint_text="Search models", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self.set_query(e.control.value or ""),
                                   border_radius=tokens.RADII["field"])
        self.polled_chip = ft.Chip(label=ft.Text("Polled only"), show_checkmark=True,
                                   selected=self.snapshot.hide_unpolled, on_click=lambda e: self.toggle_polled_only(),
                                   key="mm-polled-only")
        self.custom_chip = ft.Chip(label=ft.Text("Custom"), selected=False, on_click=lambda e: self.set_filter("custom"),
                                   key="mm-custom")
        self.removed_chip = ft.Chip(label=ft.Text("Removed"), selected=False, on_click=lambda e: self.set_filter("removed"),
                                    key="mm-removed")
        self.poll_button = ft.FilledTonalButton(content="🌐 Poll providers", on_click=lambda e: self.spawn(self.poll()),
                                                key="mm-poll")
        self.poll_text = ft.Text(self.poll_status, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT)
        self.status_row = ft.Row([], scroll=ft.ScrollMode.AUTO, spacing=6)
        self.hint = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.models_header = ft.Column([
            self.search,
            ft.Row([self.polled_chip, self.custom_chip, self.removed_chip], wrap=True, spacing=6),
            ft.Row([self.poll_button], wrap=True),
            self.poll_text,
            self.status_row,
            self.hint,
        ], spacing=8, tight=True)
        self.list_view = ft.ReorderableListView(controls=[], expand=True, on_reorder=self._on_reorder,
                                                show_default_drag_handles=False, build_controls_on_demand=True,
                                                padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"]))
        self.prefix_list = ft.ListView(controls=[], expand=True, spacing=6,
                                       padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"]))
        self.add_fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content="Add model", on_click=lambda e: self.open_add(),
                                               key="mm-add")
        self.core_notice = ft.Container(visible=not self.catalog.core.available, content=ReasonChip(
            reason="Editing needs the model catalog core", detail=mc.CORE_MISSING), padding=ft.Padding.only(left=12))
        self.body_column = ft.Column([
            ft.Container(content=self.tabs, padding=ft.Padding.symmetric(horizontal=12, vertical=6)),
            self.core_notice,
            ft.Container(content=self.models_header, padding=ft.Padding.symmetric(horizontal=12)),
            self.list_view,
            self.prefix_list,
            ft.Container(content=ft.Row([self.add_fab], alignment=ft.MainAxisAlignment.END),
                         padding=ft.Padding.only(right=16, bottom=16)),
        ], expand=True, spacing=4)
        self.render()
        return self.body_column

    def actions(self) -> list:
        return [ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", items=[
            ft.PopupMenuItem(content="Reset to defaults", icon=ft.Icons.RESTART_ALT, on_click=lambda e: self.confirm_reset()),
            ft.PopupMenuItem(content="Lock mouse wheel (desktop only)", icon=ft.Icons.MOUSE, disabled=True,
                             on_click=lambda e: self.say(WHEEL_REASON)),
        ])]

    def did_show(self) -> None:
        if self._unsub is None:
            self._unsub = self.catalog.subscribe(self._on_snapshot)
        if not self.catalog.snapshot.loaded:
            self.spawn(self._load())
        if self.builtin_keys is None:
            self.spawn(self._load_builtin())

    def dispose(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _load(self) -> None:
        try:
            await self.catalog.load()
        except Exception:
            log.exception("loading the model list failed")

    async def _load_builtin(self) -> None:
        try:
            models = await self.io(lambda: list(self.catalog.options.get_model_options()))
        except Exception:
            models = []
        self.builtin_keys = {str(m).casefold() for m in models}
        if self.filter == "custom":
            self.render()

    def _on_snapshot(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot
        self.render()

    # ---- rendering ----------------------------------------------------------------------------------

    def visible_models(self) -> list:
        snap = self.snapshot
        if self.filter == "removed":
            models = list(snap.removed)
        else:
            models = snap.visible_models()
            if self.filter == "custom":
                builtin = self.builtin_keys or set()
                models = [m for m in models if m.casefold() not in builtin]
        if self.query.strip():
            models = mc.rank_models(models, self.query, limit=len(models) or 1)
        return models

    @property
    def reorderable(self) -> bool:
        return self.filter == "all" and not self.query.strip() and not self.snapshot.hide_unpolled

    def render(self, push: bool = True) -> None:
        snap = self.snapshot
        models_tab = self.tab == "models"
        self.models_header.visible = models_tab
        self.list_view.visible = models_tab
        self.prefix_list.visible = not models_tab
        self.add_fab.content = "Add model" if models_tab else "Add prefix"
        self.polled_chip.selected = snap.hide_unpolled
        self.custom_chip.selected = self.filter == "custom"
        self.removed_chip.selected = self.filter == "removed"
        self.poll_button.disabled = self.polling or snap.is_polling()
        self.poll_button.content = "⏳ Polling…" if (self.polling or "*" in snap.polling) else "🌐 Poll providers"
        self.poll_text.value = self.poll_status
        self.status_row.controls = self._status_chips(snap)
        if models_tab:
            models = self.visible_models()
            if self.filter == "removed":
                self.hint.value = f"{len(models)} removed · swipe to restore"
            elif self.reorderable:
                self.hint.value = f"{len(models)} models · drag ☰ to reorder · swipe to remove"
            else:
                self.hint.value = f"{len(models)} shown · clear the search and filters to reorder"
            self.list_view.controls = [self._row(i, m) for i, m in enumerate(models)]
            if not models:
                self.list_view.controls = [ft.Container(
                    key="mm-empty", padding=ft.Padding.all(16),
                    content=ft.Text("No removed models." if self.filter == "removed" else
                                    ("Loading…" if not snap.loaded else "No models match."),
                                    color=ft.Colors.ON_SURFACE_VARIANT))]
        else:
            self.prefix_list.controls = self._prefix_rows()
        if push:
            _push(self.body_column if hasattr(self, "body_column") else None)

    def _status_chips(self, snap: CatalogSnapshot) -> list:
        chips = []
        for provider, status in sorted(snap.statuses.items()):
            text = str(status)
            ok = text.startswith("online")
            color = semantic("success") if ok else (ft.Colors.OUTLINE if "credential" in text else semantic("warning"))
            short = text.replace("online ", "").replace("static fallback ", "")
            chips.append(ft.Chip(label=ft.Text(f"{mc.provider_label(provider)} · {short}",
                                               theme_style=ft.TextThemeStyle.LABEL_SMALL),
                                 leading=ft.Icon(ft.Icons.CIRCLE, size=10, color=color), tooltip=text,
                                 key=f"mm-status-{provider}"))
        return chips

    def _row(self, index: int, model: str) -> ft.Control:
        removed_view = self.filter == "removed"
        polled = self.snapshot.is_polled(model)
        provider = mc.provider_of(model, list(self.snapshot.custom_routes))
        subtitle = mc.provider_label(provider) + (" · ✓ polled" if polled else "")
        reason = mc.excluded_route(model)
        tile = ft.ListTile(
            title=ft.Text(model, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            leading=ft.Icon(ft.Icons.CHECK if polled else ft.Icons.SMART_TOY_OUTLINED,
                            color=semantic("success") if polled else ft.Colors.ON_SURFACE_VARIANT),
            trailing=ft.Row([
                *([ReasonChip(reason="Not on mobile", detail=mc.excluded_detail(model))] if reason else []),
                *([ft.ReorderableDragHandle(content=ft.Icon(ft.Icons.DRAG_HANDLE))] if self.reorderable else []),
                ft.IconButton(icon=ft.Icons.RESTORE if removed_view else ft.Icons.DELETE_OUTLINE,
                              tooltip="Restore" if removed_view else "Remove", size_constraints=HIT_TARGET,
                              on_click=lambda e, m=model: self.restore(m) if removed_view else self.remove(m)),
            ], spacing=0, tight=True),
            dense=True, min_height=52,
        )
        background = ft.Container(bgcolor=semantic("success") if removed_view else ft.Colors.ERROR,
                                  padding=ft.Padding.symmetric(horizontal=16), alignment=ft.Alignment.CENTER_LEFT,
                                  content=ft.Icon(ft.Icons.RESTORE if removed_view else ft.Icons.DELETE_OUTLINE,
                                                  color=ft.Colors.ON_ERROR))
        return ft.Dismissible(
            key=f"mm-{model}",
            content=tile,
            background=background,
            secondary_background=background,
            on_dismiss=lambda e, m=model: self.restore(m) if removed_view else self.remove(m),
        )

    # ---- model edits -----------------------------------------------------------------------------------

    def _on_tab(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", []) or []) if e is not None else []
        self.set_tab(selected[0] if selected else "models")

    def set_tab(self, tab: str) -> None:
        self.tab = tab if tab in ("models", "prefixes") else "models"
        self.tabs.selected = [self.tab]
        self.render()

    def set_query(self, query: str) -> None:
        self.query = query
        self.render()

    def set_filter(self, value: str) -> None:
        self.filter = "all" if self.filter == value else value
        self.render()

    def toggle_polled_only(self) -> bool:
        value = not self.snapshot.hide_unpolled
        self.catalog.set_hide_unpolled(value)
        self.snapshot = self.catalog.snapshot
        self.render()
        return value

    def _on_reorder(self, e: Any) -> None:
        if not self.reorderable:
            self.render()
            return
        old, new = int(getattr(e, "old_index", -1)), int(getattr(e, "new_index", -1))
        self._edit(self.catalog.move_model(old, new))

    def remove(self, model: str) -> bool:
        before = list(self.snapshot.models)
        if not self._edit(self.catalog.remove_models([model])):
            self.render()
            return False
        self.say(f"Removed {model}", "Undo", lambda: self._undo(before))
        return True

    def _undo(self, before: list) -> None:
        self._edit(self.catalog.save_order(before, list(self.catalog.snapshot.models)))

    def restore(self, model: str) -> bool:
        ok = self._edit(self.catalog.restore_removed([model], add_to_saved=True), f"Restored {model}")
        if not ok:
            self.render()
        return ok

    def add(self, model: str) -> bool:
        return self._edit(self.catalog.add_model(model), f"Added {model.strip()}")

    def open_add(self) -> Any:
        if self.tab == "prefixes":
            return self.open_prefix_editor(None)
        field = ft.TextField(hint_text="Type a model ID to add…", autofocus=True, dense=True,
                             on_submit=lambda e: submit())

        def submit() -> None:
            text = (field.value or "").strip()
            if not text:
                return
            if not close_dialog(self.page, dialog) and self.page is not None:
                return  # already closed: a second tap / Enter never adds twice
            self.add(text)

        dialog = ft.AlertDialog(title=ft.Text("Add model"), content=field, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
            ft.FilledButton(content="➕ Add", on_click=lambda e: submit()),
        ])
        self.add_field, self.add_submit = field, submit
        self.last_dialog = dialog
        if self.page is not None:
            self.page.show_dialog(dialog)
        return dialog

    def confirm_reset(self) -> ConfirmDialog:
        dialog = ConfirmDialog(title="Reset Model List", body=RESET_TEXT, confirm_label="Reset", destructive=True,
                               on_confirm=lambda: self._edit(self.catalog.reset_to_defaults(), "Model list reset"))
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    async def poll(self) -> Any:
        """Desktop "🌐 Poll Providers": a full explicit poll, then the manager list becomes the polled
        catalog minus tombstones plus the genuinely custom entries (``manager_poll_models``)."""
        self.polling = True
        self.poll_status = "Contacting provider catalogs in the background…"
        self.render()
        previous = list(self.catalog.snapshot.models)
        if self.builtin_keys is None:
            await self._load_builtin()
        try:
            outcome = await self.catalog.refresh(None, explicit=True)
        finally:
            self.polling = False
        if not outcome.skipped:
            refreshed = self.catalog.manager_poll_models(previous, self.builtin_keys or set())
            if refreshed:
                ok, error = await self.io(lambda: self.catalog.save_order(refreshed, previous))
                if not ok and error:
                    self.say(error)
                known = self.catalog.core.fn("known_catalog_keys")
                result = self.catalog.last_result
                if known is not None and result is not None:
                    self.builtin_keys = set(known(list(getattr(result, "models", []) or [])))
        self.poll_status = outcome.message or READY_TEXT
        self.snapshot = self.catalog.snapshot
        self.render()
        return outcome

    # ---- custom prefixes ---------------------------------------------------------------------------------

    def routes(self) -> list:
        value = self.catalog.config.get("custom_prefix_routes", [])
        return [dict(r) for r in value if isinstance(r, dict)] if isinstance(value, list) else []

    def _prefix_rows(self) -> list:
        routes = self.routes()
        rows: list = [ft.Text("Use models like myprefix/model-name. Base URL is the provider root; Endpoint Type is "
                              "the path shape.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                              color=ft.Colors.ON_SURFACE_VARIANT)]
        if not routes:
            rows.append(ft.Container(content=ft.Text("No custom prefixes yet.", color=ft.Colors.ON_SURFACE_VARIANT),
                                     padding=ft.Padding.all(12), key="prefix-empty"))
        for index, route in enumerate(routes):
            rows.append(ft.ListTile(
                title=ft.Text(str(route.get("prefix", ""))),
                subtitle=ft.Text(f"{route.get('routing', route.get('base_url', ''))} · "
                                 f"{route.get('endpoint_type', '/chat/completions')}", max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.Icon(ft.Icons.ALT_ROUTE),
                trailing=ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete", size_constraints=HIT_TARGET,
                                       on_click=lambda e, i=index: self.delete_prefix(i)),
                on_click=lambda e, i=index: self.open_prefix_editor(i),
                key=f"prefix-{index}",
            ))
        return rows

    def open_prefix_editor(self, index: Optional[int]) -> "PrefixEditor":
        routes = self.routes()
        current = routes[index] if index is not None and 0 <= index < len(routes) else None
        editor = PrefixEditor(route=current, on_save=lambda row: self.save_prefix(index, row))
        self.prefix_editor = editor
        if self.page is not None:
            editor.show(self.page)
        return editor

    def save_prefix(self, index: Optional[int], row: dict) -> Optional[str]:
        routes = self.routes()
        if index is not None and 0 <= index < len(routes):
            merged = dict(routes[index])
            merged.update(row)
            routes[index] = merged
        else:
            routes.append(row)
        ok, error = self.catalog.save_prefix_routes(routes)
        if not ok:
            return error[1] if isinstance(error, tuple) else str(error)
        self.render()
        return None

    def delete_prefix(self, index: int) -> ConfirmDialog:
        routes = self.routes()
        route = routes[index] if 0 <= index < len(routes) else {}

        def delete() -> None:
            remaining = [r for i, r in enumerate(self.routes()) if i != index]
            self.catalog.save_prefix_routes(remaining)
            self.render()

        dialog = ConfirmDialog(title="Delete custom prefix",
                               body=f"Delete 1 custom prefix route?\n\n{route.get('prefix', '')}",
                               confirm_label="Delete", destructive=True, on_confirm=delete)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog


class PrefixEditor:
    """Add / edit one custom prefix route (prefix · base URL · endpoint type)."""

    def __init__(self, *, route: Optional[dict] = None, on_save: Callable[[dict], Optional[str]]) -> None:
        route = dict(route or {})
        self.on_save = on_save
        self.saved = False
        self._page: Any = None
        self.prefix = ft.TextField(value=str(route.get("prefix", "")), label="Prefix", hint_text="myprefix/", dense=True)
        self.routing = ft.TextField(value=str(route.get("routing", route.get("base_url", ""))), label="Base URL",
                                    hint_text="https://api.example.com/v1", dense=True, keyboard_type=ft.KeyboardType.URL)
        types = endpoint_type_options()
        current = str(route.get("endpoint_type", "/chat/completions") or "/chat/completions")
        self.endpoint_type = ft.Dropdown(value=current, label="Endpoint Type", editable=True, dense=True,
                                         options=[ft.DropdownOption(key=t, text=t) for t in dict.fromkeys([current, *types])])
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False)
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=16), content=ft.Column([
                ft.Text("Custom prefix", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
                self.prefix, self.routing, self.endpoint_type, self.error,
                ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close()),
                        ft.FilledButton(content="Save", on_click=lambda e: self.save())],
                       alignment=ft.MainAxisAlignment.END),
            ], tight=True, spacing=10)),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def row(self) -> dict:
        return {"prefix": (self.prefix.value or "").strip(), "routing": (self.routing.value or "").strip(),
                "endpoint_type": (self.endpoint_type.value or "/chat/completions").strip()}

    def save(self) -> Optional[str]:
        if self.saved:  # a second tap while the sheet closes
            return None
        row = self.row()
        if not row["prefix"] and not row["routing"]:
            error: Optional[str] = "Row 1 needs both a prefix and Base URL."
        else:
            error = self.on_save(row)
        if error:
            self.error.value = error
            self.error.visible = True
            _push(self.error)
            return error
        self.saved = True
        self.close()
        return None

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)


# ---- feature install ------------------------------------------------------------------------------------


class ModelsKeysFeature:
    """Catalog service + ModelSheet env + Models/Keys/Endpoints screens (call after the chat install)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        settings = getattr(app, "settings", None)
        self.ctx = getattr(settings, "ctx", None)
        self.store = getattr(app, "config_store", None)
        self.prefs = getattr(app, "prefs", None)
        self.catalog = ModelCatalogService(self.store, run_io=self.run_io, post=self._post)
        from glossarion_mobile.ui.screens.endpoints import run_in_run_env
        from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController

        self.backend = KeyBackend()
        self.keys = KeysController(self.store, self.backend, test_runner=run_in_run_env)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list = []
        self.screens_built: list = []

    @classmethod
    async def install(cls, app: Any) -> "ModelsKeysFeature":
        feature = cls(app)
        feature.attach(app)
        return feature

    def attach(self, app: Any) -> None:
        from glossarion_mobile.ui.sheets.model_sheet import SheetEnv, install_sheet_env

        app.models_keys = self
        mc.set_default_service(self.catalog)
        install_sheet_env(SheetEnv(
            catalog=self.catalog, store=self.store, prefs=self.prefs, ctx=self.ctx,
            navigate=getattr(app, "navigate_to", None), notify=getattr(app, "notify", None),
            copy_text=getattr(app, "_copy_text", None),
            read_clipboard=getattr(getattr(app, "clipboard", None), "get", None),
            run_io=self.run_io, spawn=self.spawn, sign_in=self.sign_in, signed_in_keys=self.signed_in_keys,
            test_key=self.test_key, tablet=self._tablet,
        ))
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        if self.store is not None and not self._unsubs:
            observe = getattr(self.store, "observe_keys", None)
            if observe is not None:
                keys = ("custom_model_list", "model_manager_removed_models", "custom_prefix_routes",
                        "model_manager_hide_unpolled_models")
                self._unsubs.append(observe(keys, lambda key, value: self._post(self._schedule_reload)))
                # desktop: a model change auto-polls that provider once its 24 h TTL is due (debounced)
                self._unsubs.append(observe(("model", "api_key"), lambda key, value: self._post(self._schedule_auto_poll)))
        self.spawn(self.catalog.load())

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    # ---- plumbing ---------------------------------------------------------------------------------------

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(fn, *args)
        else:
            fn(*args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        return asyncio.ensure_future(coro)

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-models-io")
        return await asyncio.to_thread(fn, *args)

    def _schedule_reload(self) -> None:
        if getattr(self, "_reload_pending", False):
            return
        self._reload_pending = True

        async def reload() -> None:
            await asyncio.sleep(0.3)
            self._reload_pending = False
            await self.catalog.load()

        try:
            self.spawn(reload())
        except RuntimeError:  # no running loop (worker thread without a dispatcher)
            self._reload_pending = False

    def _schedule_auto_poll(self, delay: float = 1.2) -> None:
        """Desktop ``_schedule_current_provider_catalog_refresh``: poll after typing settles."""
        if getattr(self, "_auto_poll_pending", False):
            return
        self._auto_poll_pending = True

        async def poll() -> None:
            await asyncio.sleep(delay)
            self._auto_poll_pending = False
            try:
                await self.catalog.maybe_auto_poll()
            except Exception:
                log.debug("auto-poll failed", exc_info=True)

        try:
            self.spawn(poll())
        except RuntimeError:  # no running loop (a worker thread without a dispatcher)
            self._auto_poll_pending = False

    def _tablet(self) -> bool:
        return bool(getattr(getattr(self.app, "shell", None), "tablet", False))

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def _oauth(self) -> Any:
        return getattr(getattr(self.app, "chat_feature", None), "oauth", None)

    def signed_in_keys(self) -> frozenset:
        """``AppState.signed_in`` plus the OAuthBridge slot keys (``authgpt``, ``authgpt2``, ``authgem``…)."""
        state = getattr(self.app, "state", None)
        signed = set(getattr(getattr(state, "signed_in", None), "value", None) or ())
        oauth = self._oauth()
        try:
            signed.update(getattr(oauth, "signed_in", None) or ())
        except Exception:
            pass
        return frozenset(signed)

    def sign_in(self, route: str, account: int = 0) -> Any:
        """A sign-in chip / row: the Accounts LoginSheet for that provider and slot (OAuthBridge);
        the U3 ChatGPT sheet or the Accounts screen when the provider flow is not available."""
        oauth = self._oauth()
        if oauth is not None:
            try:
                from glossarion_mobile.ui.screens.accounts import LoginSheet

                sheet = LoginSheet(oauth, provider=route, account_id=int(account or 0),
                                   copy_text=getattr(self.app, "_copy_text", None))
                sheet.show(self.page)
                return sheet
            except (TypeError, KeyError, ImportError):
                log.debug("provider sign-in sheet unavailable for %s", route, exc_info=True)
        chat_view = getattr(self.app, "chat_view", None)
        opener = getattr(chat_view, "open_login_sheet", None)
        if route == "authgpt" and not account and opener is not None:
            return opener()
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("settings.accounts")
        return None

    async def test_key(self, api_key: str, model: str) -> dict:
        entry = {"api_key": api_key, "model": model}
        return await self.run_io(lambda: self.keys.test_entry(entry, "main"))

    # ---- screens -----------------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        notify = getattr(self.app, "notify", None)
        if match.name == "settings.models":
            screen: Any = ModelManagerScreen(match, catalog=self.catalog, page=self.page, notify=notify,
                                             spawn=self.spawn, run_io=self.run_io, tablet=self._tablet())
        elif match.name in ("settings.keys", "settings.keys.pool"):
            from glossarion_mobile.ui.screens.keys import KeysScreen
            from glossarion_mobile.ui.sheets.model_sheet import sheet_env

            paths = getattr(self.app, "paths", None)
            export_dir = os.path.join(str(paths.data), "Exports") if paths is not None else None
            screen = KeysScreen(match, controller=self.keys, ctx=self.ctx, sheet_env=sheet_env(),
                                files=getattr(self.app, "files", None), page=self.page, notify=notify,
                                copy_text=getattr(self.app, "_copy_text", None),
                                read_clipboard=getattr(getattr(self.app, "clipboard", None), "get", None),
                                run_io=self.run_io, spawn=self.spawn, open_refusal_patterns=self.open_refusal_patterns,
                                export_dir=export_dir, tablet=self._tablet(), dark=self._dark())
        elif match.name == "settings.endpoints" and self.ctx is not None:
            from glossarion_mobile.ui.screens.endpoints import EndpointsScreen

            screen = EndpointsScreen(match, self.ctx, run_io=self.run_io, open_local_ai=self.open_local_ai,
                                     copy_text=getattr(self.app, "_copy_text", None),
                                     read_clipboard=getattr(getattr(self.app, "clipboard", None), "get", None))
        else:
            return None
        self.screens_built.append(match.name)
        return screen

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        try:
            screen = self.make_screen(match)
        except Exception:
            log.exception("building the %s screen failed", match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
            implemented = getattr(screen, "implemented", None)
            if match.name == "settings" and isinstance(implemented, frozenset):
                screen.implemented = implemented | IMPLEMENTED_ROUTES  # Settings home: no "Arrives in U4" chips
        return screen

    def _push_screen(self, screen: Screen) -> Optional[ft.View]:
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return None
        view = build_screen_view(screen, shell.current_route)
        shell.push_overlay(view)
        try:
            self.page.update()
        except Exception:
            pass
        screen.did_show()
        return view

    def open_refusal_patterns(self) -> Any:
        from glossarion_mobile.ui.screens.refusal_patterns import RefusalModel, RefusalPatternsScreen

        screen = RefusalPatternsScreen(None, model=RefusalModel(self.store),
                                       page=self.page, files=getattr(self.app, "files", None),
                                       notify=getattr(self.app, "notify", None), run_io=self.run_io, spawn=self.spawn)
        self._push_screen(screen)
        return screen

    def open_local_ai(self) -> Any:
        from glossarion_mobile.ui.screens.local_ai import LocalAiScreen

        screen = LocalAiScreen(None, store=self.store, catalog=self.catalog, notify=getattr(self.app, "notify", None),
                               spawn=self.spawn, navigate=getattr(self.app, "navigate_to", None))
        self._push_screen(screen)
        return screen
