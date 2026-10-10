"""ModelSheet / ModelPicker / PoeSetupSheet (UI_SPEC §2.2, §4.11, §5.4).

Replaces the U3 ``model_sheet_min`` (that module is now an alias of this one, so the chat
header keeps calling ``ModelSheetMin(...)`` / ``load_model_catalog`` unchanged).

**Structure** (phone: draggable ``BottomSheet``, full screen under 640 dp; tablet: the same
content in the sheet, capped at 420 dp wide):

* title row "Model" (or "Use once" in one-shot mode) with ⋯: Refresh online models (the
  selected model's provider, ignoring the 24 h TTL = desktop "🌐 Refresh Online Models"),
  Manage models, Hide unpolled models;
* Model · Profile · Language segments (field mode: Model only);
* Model tab: search (exact > prefix > path-segment > contains; numbered account aliases such as
  ``authgpt2/…`` while typing one), provider chips, then ★ Favorites (Prefs) · Recent (last 8) ·
  provider groups (polled models first, "✓ polled", a 🌐 refresh per group with a Shimmer while
  that provider polls) · custom routes. Rows: status dot (green ready · amber needs sign-in or a
  key, with an inline "Sign in" / "Add key" · grey + ReasonChip for routes excluded on mobile,
  which stay visible and keep a desktop value), provider badge, ✓ polled; long-press →
  ActionSheet (★ Favorite · Copy id · Provider info · Refresh this provider);
* route row for the selected model (``model_catalog.route_info`` = shared ``route_controls``):
  KeyField for the API key + Test, sign-in chip, Google credentials / Vertex location / GCP
  project (schema tiles), Poe setup, the excluded-route reason;
* Thinking & effort (``ExpansionTile``): the schema tiles of the model's family (the same keys
  as Settings › Thinking & reasoning, rendered by the shared settings tiles);
* footer: Manage models · Keys · Accounts · ⓘ Provider information.

Selecting writes the global ``model`` key, or the chat override when "Apply to this chat only"
is on (``on_select(field, value, this_chat_only)``, the U3 contract). One-shot mode only returns
the choice. ``ModelPicker`` is the read-only field (▾) that opens this sheet in field mode
(KeyEditor, Manga, Glossary, QA).

App services come from ``SheetEnv`` (installed by ``ModelsKeysFeature``); without it the sheet
still works on the plain model list the caller passes (U3 behaviour).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import re

import flet as ft

from glossarion_mobile.state.languages import remember_language, target_languages
from glossarion_mobile.services import model_catalog as mc
from glossarion_mobile.services.model_catalog import (
    EXCLUDED_ROUTE_PREFIXES,
    PROVIDER_CHIPS,
    THINKING_FIELDS,
    GENERAL_THINKING_FIELDS,
    FAMILY_TITLES,
    CatalogSnapshot,
    RouteInfo,
    excluded_detail,
    excluded_route,
)
from glossarion_mobile.services.oauth import sign_in_satisfied, slot_key
from glossarion_mobile.ui import motion, tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.sheet import scroll_column, sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = [
    "EXCLUDED_ROUTE_PREFIXES",
    "MAX_RECENTS",
    "ModelPicker",
    "ModelSheet",
    "ModelSheetMin",
    "PoeSetupSheet",
    "SheetEnv",
    "excluded_route",
    "filter_models",
    "install_sheet_env",
    "load_model_catalog",
    "sheet_env",
]

log = logging.getLogger("glossarion.models")

MAX_ROWS = mc.MAX_RESULTS
MAX_RECENTS = 8
FAVORITES_PREF = "model_favorites"
RECENTS_PREF = "model_recents"
LANG_RECENTS_PREF = "language_recents"
_SIGN_IN_ROUTES = ("authgpt", "authgem", "authgem-vertex", "authcd", "authgrok")

#: Mobile footnote under the desktop provider text (routes the phone cannot run).
PROVIDER_INFO_MOBILE_NOTE = ("On mobile, routes that need a desktop program (antigravity/, ocagy/, ocz/, authza/, "
                             "autharena/, search/opera, ollamapull/) are listed but disabled.")
_PROVIDER_INFO_CACHE: dict = {}


def provider_info_markdown() -> str:
    """The desktop "Model Provider Information" text (``model_options.provider_info_html``, moved out of
    TranslatorGUI._show_model_info_dialog) as Markdown for the ⓘ sheet, plus the mobile note."""
    cached = _PROVIDER_INFO_CACHE.get("md")
    if cached is not None:
        return cached
    try:
        import model_options

        html = model_options.provider_info_html()
    except Exception as exc:  # backend not importable (host tests without src/)
        log.debug("provider info unavailable: %s", exc)
        return PROVIDER_INFO_MOBILE_NOTE
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_images = True
        text = converter.handle(html).strip()
    except Exception:
        import re as _re

        text = _re.sub(r"<[^>]+>", "", html).strip()
    text = f"{text}\n\n*{PROVIDER_INFO_MOBILE_NOTE}*"
    _PROVIDER_INFO_CACHE["md"] = text
    return text


#: Routes whose client needs a package this build does not ship (plan dependency rule; the stored
#: key / cookie is kept): route prefix -> (import name, ReasonChip, detail)
DEPENDENCY_ROUTES = {
    "poe/": ("poe_api_wrapper", "Needs poe-api-wrapper · not in this build",
             "The Poe client (poe-api-wrapper) has no Android / iOS build (backend_manifest: desktop-only Poe "
             "client), so poe/ models cannot run on mobile. The saved p-b cookie is kept for the desktop."),
}
#: Browser-backed keyless routes (they run in the hidden in-app browser, services.webview_bridge).
WEBVIEW_ROUTES = ("authnd/", "search/gemini")


def dependency_reason(model: str) -> Optional[tuple]:
    """(ReasonChip, detail) when the model's route cannot run in this build, else None."""
    lowered = str(model or "").strip().lower()
    for prefix, (module, chip, detail) in DEPENDENCY_ROUTES.items():
        if lowered.startswith(prefix):
            import importlib.util

            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            return None if found else (chip, detail)
    if lowered.startswith(WEBVIEW_ROUTES):
        try:
            from glossarion_mobile.services import webview_bridge

            available, reason = webview_bridge.availability()
        except Exception:
            available, reason = False, ""
        if not available:
            from glossarion_mobile.ui.screens.accounts import NEEDS_WEBVIEW_CHIP

            return NEEDS_WEBVIEW_CHIP, reason or "The in-app browser is not available in this session."
    return None


# ---- U3 compatibility helpers (model_sheet_min) ------------------------------------------------------


def load_model_catalog(config_get: Callable[[str, Any], Any]) -> list:
    """Blocking: the desktop picker list (saved custom list + catalog - removed); U3 contract."""
    service = mc.default_service()
    if service is not None:
        return list(service.load_blocking().models)
    import model_options  # backend module

    custom = config_get("custom_model_list", None)
    removed = config_get("model_manager_removed_models", []) or []
    return list(
        model_options.merge_saved_model_options(
            custom if isinstance(custom, list) else None, model_options.get_model_options(), removed
        )
    )


def filter_models(models: Sequence[str], query: str, current: str = "", limit: int = MAX_ROWS) -> list:
    """Current model first when nothing is typed; ranked search otherwise (U3 name kept)."""
    try:
        return mc.rank_models(models, query, current=current, limit=limit)
    except Exception:
        needle = str(query or "").strip().casefold()
        return [m for m in models if not needle or needle in str(m).casefold()][:limit]


# ---- app services ------------------------------------------------------------------------------------


@dataclass
class SheetEnv:
    """What the sheet uses from the app; every field is optional."""

    catalog: Any = None  # ModelCatalogService
    store: Any = None  # MobileConfigStore
    prefs: Any = None  # Prefs (favourites / recents)
    ctx: Any = None  # SettingsContext (schema tiles for thinking / Google credentials)
    navigate: Optional[Callable[..., Any]] = None  # (route name, params=None)
    notify: Optional[Callable[..., Any]] = None
    copy_text: Optional[Callable[[str], Any]] = None
    read_clipboard: Optional[Callable[[], Any]] = None
    run_io: Optional[Callable[..., Any]] = None
    spawn: Optional[Callable[[Any], Any]] = None
    sign_in: Optional[Callable[[str, int], Any]] = None  # (route, account) -> opens the LoginSheet
    signed_in_keys: Optional[Callable[[], Any]] = None  # -> set of "authgpt", "authgpt2", ...
    test_key: Optional[Callable[..., Any]] = None  # async (api_key, model) -> result dict
    tablet: Callable[[], bool] = field(default=lambda: False)
    # U9: (route) -> the next free account slot ("+ Add account": OAuthBridge.next_slot)
    next_slot: Optional[Callable[[str], int]] = None
    # U9: (account id) -> shows the 📊 Gemini status sheet (OAuthBridge.gemini_status; desktop authgem
    # status button next to the Gemini login)
    gemini_status: Optional[Callable[[int], Any]] = None
    # U9: (model) -> an ``accounts.GcpProjectPicker`` for the route row of authgem-vertex/ models (the Accounts
    # project dropdown: billing marks, desktop selection rule, authgem_auth project cache)
    gcp_project_picker: Optional[Callable[[str], Any]] = None


_SHEET_ENV: Optional[SheetEnv] = None


def install_sheet_env(env: Optional[SheetEnv]) -> None:
    global _SHEET_ENV
    _SHEET_ENV = env


def sheet_env() -> SheetEnv:
    return _SHEET_ENV if _SHEET_ENV is not None else SheetEnv()


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


def _dot(color: str, tooltip: str) -> ft.Control:
    return ft.Container(width=10, height=10, border_radius=5, bgcolor=color, tooltip=tooltip)


# ---- the sheet ------------------------------------------------------------------------------------


class ModelSheet:
    TABS = ("model", "profile", "language")

    def __init__(
        self,
        *,
        current_model: str = "",
        current_profile: str = "",
        current_language: str = "",
        models: Sequence[str] = (),
        profiles: Sequence[str] = (),
        languages: Sequence[str] = (),
        signed_in: Optional[Callable[[str], bool]] = None,
        tab: str = "model",
        chat_scope: bool = False,
        on_select: Optional[Callable[[str, str, bool], Any]] = None,  # (field, value, this_chat_only)
        on_sign_in: Optional[Callable[[], Any]] = None,
        on_accounts: Optional[Callable[[], Any]] = None,
        env: Optional[SheetEnv] = None,
        one_shot: bool = False,
        field_mode: bool = False,
        title: Optional[str] = None,
    ) -> None:
        self.env = env or sheet_env()
        self.current = {"model": current_model, "profile": current_profile, "language": current_language}
        self.profiles = list(profiles)
        self.languages = list(languages)
        self._signed_in = signed_in
        self.on_select = on_select
        self.on_sign_in = on_sign_in
        self.on_accounts = on_accounts
        self.one_shot = one_shot
        self.field_mode = field_mode
        self.heading = "Use once" if one_shot else (title or "Model")
        self.tab = tab if tab in self.TABS and not field_mode else "model"
        self.query = ""
        self.chip = "all"
        self.selected_value: Optional[str] = None
        self._page: Any = None
        self._unsub: Optional[Callable[[], None]] = None
        self.closed = False
        self.rows: dict = {}
        self.group_headers: dict = {}
        self.route: Optional[RouteInfo] = None
        self.thinking_tiles: dict = {}
        self._tiles: dict = {}
        self._route_signature: Any = None
        self._key_cache: Optional[tuple] = None
        self._favorites_cache: Optional[list] = None
        catalog = self.env.catalog
        snap = catalog.snapshot if catalog is not None else None
        if snap is not None and snap.loaded:
            self.snapshot = snap
        else:
            self.snapshot = CatalogSnapshot(models=tuple(models), loaded=bool(models))
        self._keyless: set = set()
        self._build()
        self.refresh()

    # ---- building ----------------------------------------------------------------------------------

    def _build(self) -> None:
        self.title_text = ft.Text(self.heading, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600,
                                  expand=True)
        self.hide_item = ft.PopupMenuItem(content="Hide unpolled models", icon=ft.Icons.FILTER_ALT_OUTLINED,
                                          checked=self.snapshot.hide_unpolled, on_click=self._toggle_hide_unpolled)
        self.menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            tooltip="More",
            items=[
                ft.PopupMenuItem(content="Refresh online models", icon=ft.Icons.PUBLIC,
                                 on_click=lambda e: self.refresh_selected_provider()),
                ft.PopupMenuItem(content="Manage models", icon=ft.Icons.TUNE, on_click=lambda e: self._nav("settings.models")),
                self.hide_item,
            ],
        )
        self.tabs = ft.SegmentedButton(
            segments=[ft.Segment(value="model", label="Model"), ft.Segment(value="profile", label="Profile"),
                      ft.Segment(value="language", label="Language")],
            selected=[self.tab],
            on_change=self._on_tab,
            visible=not self.field_mode and not self.one_shot,
        )
        self.search = ft.TextField(hint_text="Search models", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self.set_query(e.control.value or ""),
                                   on_submit=self.submit_query,
                                   border_radius=tokens.RADII["field"])
        self.chips: dict = {}
        for chip_id, label in PROVIDER_CHIPS:
            self.chips[chip_id] = ft.Chip(label=ft.Text(label), selected=chip_id == self.chip,
                                          on_click=lambda e, c=chip_id: self.set_chip(c), show_checkmark=False,
                                          key=f"model-chip-{chip_id}")
        self.chip_row = ft.Row(list(self.chips.values()), scroll=ft.ScrollMode.AUTO, spacing=6)
        self.chat_switch = ft.Switch(label="Apply to this chat only", value=False,
                                     visible=not self.one_shot and not self.field_mode)
        self._chat_scope_initial = False
        self.list_view = ft.ListView(controls=[], height=400, spacing=0, build_controls_on_demand=True)
        self.status_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                   visible=False)
        self.route_row = ft.Column([], spacing=6, tight=True, visible=False)
        self.thinking = ft.ExpansionTile(title=ft.Text("Thinking & effort"), controls=[], expanded=False,
                                         visible=False, controls_padding=ft.Padding.only(left=4, right=4, bottom=8))
        # Owner (U14): the other families' thinking sections and the shared ones, like desktop's Other Settings
        # (the model's own family stays first, in ``self.thinking``)
        self.thinking_others = ft.Column([], spacing=0, tight=True, visible=False)
        self.other_thinking_tiles: dict = {}
        self.footer = ft.Row(
            [
                ft.TextButton(content="Manage models", on_click=lambda e: self._nav("settings.models")),
                ft.TextButton(content="Keys", on_click=lambda e: self._nav("settings.keys")),
                ft.TextButton(content="Accounts", on_click=lambda e: self._accounts()),
                ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Provider information", on_click=self.open_provider_info,
                              size_constraints=HIT_TARGET),
            ],
            spacing=4,
            wrap=True,
        )
        # The sheet scrolls as a whole (the list keeps its own height and scrolls inside it): the
        # route rows, "Thinking & effort" and the footer sit below the list, past a phone's height.
        self.dialog = ft.BottomSheet(
            content=sheet_frame(
                padding=ft.Padding.only(left=12, right=12, bottom=16),
                content=scroll_column(
                    [
                        ft.Row([self.title_text, self.menu], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        self.tabs,
                        self.search,
                        self.chip_row,
                        self.chat_switch,
                        self.status_text,
                        self.list_view,
                        self.route_row,
                        self.thinking,
                        self.thinking_others,
                        self.footer,
                    ],
                    spacing=8,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    # ---- state helpers -----------------------------------------------------------------------------

    def _pref_list(self, key: str) -> list:
        prefs = self.env.prefs
        if prefs is None:
            return []
        value = prefs.get(key, [])
        return [str(v) for v in value if str(v).strip()] if isinstance(value, list) else []

    def favorites(self) -> list:
        if self._favorites_cache is None:
            self._favorites_cache = self._pref_list(FAVORITES_PREF)
        return list(self._favorites_cache)

    def recents(self) -> list:
        return self._pref_list(RECENTS_PREF)[:MAX_RECENTS]

    def toggle_favorite(self, model: str) -> bool:
        favs = self.favorites()
        if model in favs:
            favs.remove(model)
            on = False
        else:
            favs.insert(0, model)
            on = True
        if self.env.prefs is not None:
            self.env.prefs.set(FAVORITES_PREF, favs)
        self._favorites_cache = None
        self.refresh()
        self._push_all()
        return on

    def _remember(self, key: str, value: str) -> None:
        prefs = self.env.prefs
        if prefs is None or not value:
            return
        items = [v for v in self._pref_list(key) if v != value]
        prefs.set(key, [value] + items[: MAX_RECENTS * 2 - 1])

    def is_polled(self, model: str) -> bool:
        return self.snapshot.is_polled(model)

    def signed_in(self, model: str, route: Optional[str] = None, account: Optional[int] = None) -> bool:
        """Signed in for the model's sign-in route (``route``/``account`` override the model's own)."""
        own_route, own_account = mc.login_route(model)
        route = route or own_route
        account = own_account if account is None else account
        if route is None:
            return False
        getter = self.env.signed_in_keys
        if getter is None:  # U3 caller without the app services: its own (ChatGPT) check
            if self._signed_in is not None and route == own_route:
                try:
                    return bool(self._signed_in(model))
                except Exception:
                    return False
            return False
        try:
            keys = set(getter() or ())  # AppState.signed_in: "authgpt", "authgpt2", "authgem", ...
        except Exception:
            return False
        if route == own_route and int(account or 0) == int(own_account or 0):
            # the model's own sign-in: slot #N, or any slot for the pool routes authgpt0/,
            # authgrok0/ and authgem-vertex0/ (the Send gate uses the same rule)
            return sign_in_satisfied(model, keys)
        return slot_key(route, account) in keys

    def _key_state(self) -> tuple:
        """(configured API key present, providers with an enabled pool key); once per render."""
        if self._key_cache is None:
            store = self.env.store
            api_key = str(store.get("api_key", "") or "").strip() if store is not None else ""
            providers: frozenset = frozenset()
            catalog = self.env.catalog
            if catalog is not None:
                try:
                    providers = frozenset(catalog.provider_keys(list(self.snapshot.custom_routes)))
                except Exception:
                    providers = frozenset()
            self._key_cache = (bool(api_key) and not api_key.startswith("ENC:"), providers)
        return self._key_cache

    def has_key_for(self, model: str) -> bool:
        if self.env.store is None:
            return True
        has_main, providers = self._key_state()
        if has_main:
            return True
        try:
            owner = self.env.catalog.options.catalog_provider_for_model(model, list(self.snapshot.custom_routes)) \
                if self.env.catalog is not None else None
        except Exception:
            owner = None
        return bool(owner) and owner in providers

    def row_state(self, model: str) -> tuple:
        """(state, action label or None): ready · sign_in · add_key · excluded."""
        if excluded_route(model):
            return "excluded", None
        if dependency_reason(model) is not None:
            return "unavailable", None
        route, _account = mc.login_route(model)
        if route in _SIGN_IN_ROUTES:
            if not self.signed_in(model):
                label = "Sign in with ChatGPT" if route == "authgpt" else "Sign in"
                return "sign_in", label
            return "ready", None
        if self._keyless and model in self._keyless:
            return "ready", None
        if self.env.store is not None and not self.has_key_for(model) and self._needs_key(model):
            return "add_key", "Add key"
        return "ready", None

    def _needs_key(self, model: str) -> bool:
        if self._keyless:
            return model not in self._keyless
        lowered = str(model or "").lower()
        return not lowered.startswith(("ollama/", "lmstudio/", "google-translate-free", "search/"))

    # ---- rows --------------------------------------------------------------------------------------

    def _model_row(self, model: str, section: str = "group") -> ft.Control:
        state, action = self.row_state(model)
        polled = self.is_polled(model)
        provider = mc.provider_of(model)
        if state == "excluded":
            dot = _dot(ft.Colors.OUTLINE, "Not available on mobile")
            trailing: Any = ReasonChip(reason="Not available on mobile", detail=excluded_detail(model) or excluded_route(model))
        elif state == "unavailable":
            chip, detail = dependency_reason(model) or ("Not available", "")
            dot = _dot(ft.Colors.OUTLINE, chip)
            trailing = ReasonChip(reason=chip, detail=detail)
        elif state == "sign_in":
            dot = _dot(semantic("warning"), "Needs sign-in")
            trailing = ft.TextButton(content=action, on_click=lambda e, m=model: self._sign_in(m))
        elif state == "add_key":
            dot = _dot(semantic("warning"), "Needs an API key")
            trailing = ft.TextButton(content=action, on_click=lambda e: self._nav("settings.keys"))
        else:
            dot = _dot(semantic("success"), "Ready")
            trailing = None
        subtitle_parts = [mc.provider_label(provider)]
        if polled:
            subtitle_parts.append("✓ polled")
        if model in self.favorites():
            subtitle_parts.append("★")
        selected = model == self.current["model"]
        row = ft.ListTile(
            leading=dot,
            title=ft.Text(model, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(" · ".join(subtitle_parts), theme_style=ft.TextThemeStyle.BODY_SMALL,
                             color=ft.Colors.ON_SURFACE_VARIANT),
            trailing=trailing,
            selected=selected,
            dense=True,
            min_height=56,
            on_click=None if state in ("excluded", "unavailable") else (lambda e, m=model: self.select("model", m)),
            on_long_press=lambda e, m=model: self.open_row_actions(m),
            key=f"model-{section}-{model}",  # unique per section (a model can be listed twice)
        )
        self.rows[model] = row
        return row

    def _section_header(self, title: str, *, provider: Optional[str] = None, count: Optional[int] = None) -> ft.Control:
        label = f"{title} · {count}" if count is not None else title
        text = ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY, expand=True)
        controls: list = [text]
        polling = provider is not None and self.snapshot.is_polling(provider)
        if provider is not None:
            controls.append(ft.IconButton(icon=ft.Icons.PUBLIC, tooltip=f"Refresh online models for {title}",
                                          on_click=lambda e, p=provider: self.refresh_provider(p),
                                          disabled=polling or mc.provider_excluded(provider),
                                          size_constraints=HIT_TARGET, icon_size=18, key=f"refresh-{provider}"))
        status = self.snapshot.statuses.get(provider) if provider else None
        if status and provider is not None and mc.provider_excluded(provider):
            status = mc.EXCLUDED_STATUS  # never a poll error for a route that cannot run here
        body: ft.Control = ft.Row(controls, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        if status and provider is not None:
            body = ft.Column([body, ft.Text(str(status), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                            color=ft.Colors.ON_SURFACE_VARIANT)], spacing=0, tight=True)
        header: ft.Control = ft.Container(content=body, padding=ft.Padding.only(left=12, right=4, top=6),
                                          key=f"group-{provider or title}")
        if polling:
            # a static tint under reduce motion (UI_SPEC §6.3)
            header = motion.shimmer(header, base_color=ft.Colors.PRIMARY_CONTAINER, highlight_color=ft.Colors.SURFACE)
        if provider is not None:
            self.group_headers[provider] = header
        return header

    def _choice_row(self, field_name: str, value: str, subtitle: Optional[str] = None) -> ft.Control:
        row = ft.ListTile(
            title=ft.Text(value),
            subtitle=ft.Text(subtitle, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS,
                             theme_style=ft.TextThemeStyle.BODY_SMALL) if subtitle else None,
            leading=ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if value == self.current[field_name]
                            else ft.Icons.RADIO_BUTTON_UNCHECKED),
            selected=value == self.current[field_name],
            dense=True,
            min_height=48,
            on_click=lambda e, v=value: self.select(field_name, v),
            key=f"{field_name}-{value}",
        )
        self.rows[value] = row
        return row

    # ---- model list ------------------------------------------------------------------------------

    def visible_models(self) -> list:
        models = self.snapshot.visible_models()
        if self.chip != "all":
            routes = list(self.snapshot.custom_routes)
            models = [m for m in models if mc.chip_of(mc.provider_of(m, routes)) == self.chip]
        return models

    def model_rows(self) -> list:
        """Controls for the Model tab (headers interleaved with rows unless searching)."""
        self.group_headers = {}
        models = self.visible_models()
        if self.query.strip():
            ranked = filter_models(models, self.query, limit=MAX_ROWS)
            rows = [self._model_row(m, "search") for m in ranked]
            typed = self.free_text_row()
            if typed is not None:
                rows.insert(0, typed)
            return rows
        out: list = []
        current = self.current["model"]
        if current and current not in models and (self.chip == "all"):
            out.append(self._section_header("Selected"))
            out.append(self._model_row(current, "selected"))
        known = set(models)
        favorites = [m for m in self.favorites() if m in known or m == current]
        if favorites:
            out.append(self._section_header("★ Favorites"))
            out.extend(self._model_row(m, "favorites") for m in favorites)
        recents = [m for m in self.recents() if m in known and m not in favorites]
        if recents:
            out.append(self._section_header("Recent"))
            out.extend(self._model_row(m, "recent") for m in recents)
        aliases = self.account_aliases()
        if aliases:
            out.append(self._section_header("Account aliases"))
            out.extend(self._model_row(m, "aliases") for m in aliases)
        routes = list(self.snapshot.custom_routes)
        groups = mc.group_models(models, routes, polled=self.is_polled)
        budget = MAX_ROWS
        for group in groups:
            if budget <= 0:
                break
            out.append(self._section_header(group.label, provider=group.provider, count=len(group.models)))
            for model in group.models[:budget]:
                out.append(self._model_row(model))
            budget -= len(group.models)
        if not models and not out:
            message = "No models match this filter." if (self.chip != "all" or self.snapshot.hide_unpolled) else \
                "Loading the model list…"
            out.append(ft.Container(content=ft.Text(message, color=ft.Colors.ON_SURFACE_VARIANT),
                                    padding=ft.Padding.all(12), key="models-empty"))
        return out

    def typed_model(self) -> str:
        """The search text as a model id when no listed model is exactly that (the desktop model box is an
        editable combo: any typed or pasted id is used as is), else ''."""
        needle = self.query.strip()
        if not needle:
            return ""
        listed = set(self.snapshot.visible_models()) | {str(self.current.get("model") or "")}
        return "" if needle in listed else needle

    def free_text_row(self) -> Optional[ft.Control]:
        """``Use “<typed id>”`` at the top of the search results (like the Language tab's free-text row);
        an excluded route is shown disabled with its reason."""
        needle = self.typed_model()
        if not needle:
            return None
        reason = excluded_route(needle)
        return ft.ListTile(
            title=ft.Text(f"Use “{needle}”", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(reason or "Not in the model list · used exactly as typed",
                             theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            leading=ft.Icon(ft.Icons.EDIT), dense=True, disabled=bool(reason),
            on_click=None if reason else (lambda e, v=needle: self.select("model", v)),
            key="model-free-text",
        )

    def submit_query(self, e: Any = None) -> None:
        """Search field Enter: the typed id (Model tab) / language (Language tab) like its free-text row,
        else the single remaining match."""
        needle = self.query.strip()
        if not needle:
            return
        if self.tab == "model":
            typed = self.typed_model()
            if typed and not excluded_route(typed):
                self.select("model", typed)
                return
            ranked = filter_models(self.visible_models(), needle, limit=2)
            if len(ranked) == 1 or (ranked and ranked[0] == needle):
                self.select("model", ranked[0])
        elif self.tab == "language":
            names = list(self.languages)
            exact = next((v for v in names if v.casefold() == needle.casefold()), None)
            self.select("language", exact or needle)

    def signed_in_slots(self, route: str) -> list:
        """Signed-in account slots of a sign-in route (from ``signed_in_keys``: authgpt, authgpt2, …)."""
        getter = self.env.signed_in_keys
        if getter is None or not route:
            return []
        try:
            keys = set(getter() or ())
        except Exception:
            return []
        slots: set = set()
        for key in keys:
            text = str(key)
            if text == route:
                slots.add(0)
            elif text.startswith(route) and text[len(route):].isdigit():
                slots.add(int(text[len(route):]))
        return sorted(slots)

    def account_aliases(self) -> list:
        """UI_SPEC §2.2 "Account aliases": the current sign-in model under every signed-in slot
        (``model_options.numbered_model_completion_values`` renders ``authgpt2/…`` from the canonical
        route), plus the rotating pool alias (``authgpt0/…``) when more than one slot is signed in."""
        current = str(self.current.get("model") or "")
        route, account = mc.login_route(current)
        if not route:
            return []
        slots = self.signed_in_slots(route)
        if not slots:
            return []
        rest = current.split("/", 1)[1] if "/" in current else ""
        canonical = f"{route}/{rest}"
        try:
            import model_options

            render = model_options.numbered_model_completion_values
        except Exception:
            render = None
        out: list = []
        wanted = [n for n in slots if n] + ([0] if len(slots) > 1 and route in ("authgpt", "authgrok", "authgem-vertex")
                                           else [])
        for slot in wanted:
            if render is not None:
                values = render([canonical], f"{route}{slot}/")
                alias = values[0] if values else f"{route}{slot}/{rest}"
            else:
                alias = f"{route}{slot}/{rest}"
            if alias != current and alias not in out:
                out.append(alias)
        if account and canonical != current and 0 in slots:
            out.insert(0, canonical)
        return out

    def refresh(self) -> None:
        self.rows = {}
        self._key_cache = None
        self._favorites_cache = None
        model_tab = self.tab == "model"
        self.search.visible = model_tab or self.tab == "language"
        self.search.hint_text = "Search models" if model_tab else "Search or type a language"
        self.chip_row.visible = model_tab
        # the model is always this chat's own (U13); profile and language may still apply to every chat
        self.chat_switch.visible = not self.one_shot and not self.field_mode and self.tab != "model"
        for chip_id, chip in self.chips.items():
            chip.selected = chip_id == self.chip
        self.hide_item.checked = bool(self.snapshot.hide_unpolled)
        if model_tab:
            rows = self.model_rows()
        elif self.tab == "profile":
            rows = self.profile_rows()
        else:
            rows = self.language_rows()
        self.list_view.controls = rows
        self._refresh_route_row()

    def _push_all(self) -> None:
        _push(self.dialog)

    def set_query(self, query: str) -> None:
        self.query = query
        if (self.search.value or "") != query:  # set from code (Welcome: a provider's models)
            self.search.value = query
        self.refresh()
        self._push_all()

    def set_chip(self, chip_id: str) -> None:
        self.chip = chip_id if chip_id in self.chips else "all"
        self.refresh()
        self._push_all()

    def set_models(self, models: Sequence[str]) -> None:
        """U3 contract: the caller loaded the list itself."""
        if self.env.catalog is not None and self.snapshot.loaded:
            return  # the catalog service owns the list (and its markers)
        self.snapshot = CatalogSnapshot(models=tuple(models), loaded=True)
        self.refresh()
        self._push_all()

    def apply_snapshot(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot
        if self.closed:
            return
        self.refresh()
        self._push_all()

    def set_keyless(self, keyless: set) -> None:
        self._keyless = set(keyless or ())
        if self.closed:
            return
        self.refresh()
        self._push_all()

    # ---- profile / language tabs ---------------------------------------------------------------------

    def _profile_texts(self) -> dict:
        """Every profile's text as Settings › Profiles & prompts lists it (built-ins included on a fresh config); once per render."""
        from glossarion_mobile.ui.screens.profiles import ProfileService

        store = self.env.store
        try:
            return dict(ProfileService(store).listing().texts) if store is not None else {}
        except Exception:
            stored = store.get("prompt_profiles", None) if store is not None else None
            return dict(stored) if isinstance(stored, dict) else {}

    def _profile_preview(self, name: str, texts: Optional[dict] = None) -> Optional[str]:
        text = (self._profile_texts() if texts is None else texts).get(name)
        if isinstance(text, str):
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
            return "\n".join(lines) or None
        return None

    def profile_rows(self) -> list:
        from glossarion_mobile.ui.screens.profiles import SPECIALISED_GROUP, grouped_profile_names

        names = self.profiles or ([self.current["profile"]] if self.current["profile"] else [])
        texts = self._profile_texts()
        rows: list = []
        for title, group in grouped_profile_names(names):
            if title == SPECIALISED_GROUP:  # chat order: the task-specific built-ins after the translation ones
                rows.append(ft.Container(content=ft.Text(title, theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                                                         color=ft.Colors.PRIMARY),
                                         padding=ft.Padding.only(left=16, top=8), key="profile-group-specialised"))
            rows.extend(self._choice_row("profile", p, self._profile_preview(p, texts)) for p in group)
        store = self.env.store
        role = bool(store.get("system_prompt_to_user", False)) if store is not None else False
        self.role_toggle = ft.SegmentedButton(
            segments=[ft.Segment(value="system", label="System"), ft.Segment(value="user", label="User")],
            selected=["user" if role else "system"],
            on_change=self._on_role,
            disabled=store is None,
        )
        prefill = str(store.get("active_assistant_prompt_profile", "") or "") if store is not None else ""
        rows.append(ft.Container(
            padding=ft.Padding.only(left=12, right=12, top=8),
            content=ft.Column([
                ft.Row([ft.Text("Prompt role", expand=True), self.role_toggle],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.ListTile(title=ft.Text("Assistant prefill"), subtitle=ft.Text(prefill or "None"),
                            leading=ft.Icon(ft.Icons.SHORT_TEXT), on_click=lambda e: self._nav("settings.prefill"),
                            dense=True, key="profile-prefill"),
                ft.Row([ft.TextButton(content="Edit / New / Manage", icon=ft.Icons.EDIT_NOTE,
                                      on_click=lambda e: self._nav("settings.profiles"))]),
            ], spacing=4, tight=True),
            key="profile-extras",
        ))
        return rows

    def _on_role(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", []) or []) if e is not None else []
        to_user = bool(selected and selected[0] == "user")
        if self.env.store is not None:
            self.env.store.set("system_prompt_to_user", to_user)

    def language_rows(self) -> list:
        needle = self.query.strip()
        names = list(target_languages(self.env.prefs, self.languages))
        if self.current["language"] and self.current["language"] not in names:
            names.append(self.current["language"])
        rows: list = []
        recent = [v for v in self._pref_list(LANG_RECENTS_PREF) if v in names][:5]
        if recent and not needle:
            rows.append(self._section_header("Recent"))
            rows.extend(self._choice_row("language", v) for v in recent)
            rows.append(self._section_header("All languages"))
        matches = [v for v in names if not needle or needle.casefold() in v.casefold()]
        rows.extend(self._choice_row("language", v) for v in matches)
        if needle and needle.casefold() not in {v.casefold() for v in names}:
            rows.insert(0, ft.ListTile(title=ft.Text(f"Use “{needle}”"), leading=ft.Icon(ft.Icons.EDIT),
                                       on_click=lambda e, v=needle: self.select("language", v), dense=True,
                                       key="language-free-text"))
        if not needle:  # U11 item 9: any language a model can write, not only the built-in list
            rows.append(ft.ListTile(title=ft.Text("＋ Add a language"),
                                    subtitle=ft.Text("Type its name in the search field above"),
                                    leading=ft.Icon(ft.Icons.ADD), dense=True, key="language-add",
                                    on_click=lambda e: self._focus_language_search()))
        return rows

    def _focus_language_search(self) -> None:
        self.search.hint_text = "Type a language, then pick “Use …”"
        try:
            self.search.update()
            run = getattr(self._page, "run_task", None)
            if callable(run):
                run(self.search.focus)
        except Exception:
            pass

    # ---- route row + thinking -----------------------------------------------------------------------

    def _refresh_route_row(self) -> None:
        model_tab = self.tab == "model"
        model = self.current["model"]
        info = self.route
        if info is None or info.model != model:
            info = self._quick_route(model)
        if not model_tab:
            self.route_row.controls = []
            self._route_signature = None
        else:
            # Flet 1.0.3 freezes a keyed control that replaces an equal-keyed one, so the
            # (mutable) route controls are rebuilt only when what they show changes.
            signature = (info, self.signed_in(info.model, info.login, info.account) if info.login else None)
            if signature != self._route_signature:
                self._route_signature = signature
                self.route_row.controls = self._route_controls(info)
        self.route_row.visible = bool(self.route_row.controls)
        self._refresh_thinking(info if model_tab else None)

    def _quick_route(self, model: str) -> RouteInfo:
        """Route info without backend imports (the full one is computed off the loop)."""
        route, account = mc.login_route(model)
        lowered = str(model or "").lower()
        return RouteInfo(model=model, provider=mc.provider_of(model), excluded=excluded_route(model),
                         excluded_detail=excluded_detail(model), logins=(route,) if route else (), account=account,
                         needs_key=self._needs_key(model), poe=lowered.startswith("poe/"), family=mc.family_of(model))

    def set_route(self, info: RouteInfo) -> None:
        self.route = info
        if not self.closed:
            self._refresh_route_row()
            self._push_all()

    def _route_controls(self, info: RouteInfo) -> list:
        controls: list = []
        if not info.model:
            return controls
        if info.excluded:
            controls.append(ft.Row([ft.Text(info.excluded, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
                                    ReasonChip(reason="Not available on mobile", detail=info.excluded_detail)],
                                   key="route-excluded"))
            return controls
        missing = dependency_reason(info.model)
        if missing is not None:
            chip, detail = missing
            controls.append(ft.Row([ft.Text(detail, color=ft.Colors.ON_SURFACE_VARIANT, expand=True,
                                            theme_style=ft.TextThemeStyle.BODY_SMALL),
                                    ReasonChip(reason=chip, detail=detail)], key="route-dependency"))
            if info.poe:
                controls.append(ft.Row([ft.FilledTonalButton(content="Poe setup", icon=ft.Icons.COOKIE_OUTLINED,
                                                             disabled=True, key="route-poe"),
                                        ReasonChip(reason=chip, detail=detail)], wrap=True))
            return controls
        if info.logins:
            own_route, _own_account = mc.login_route(info.model)
            chips: list = []
            for login in info.logins:
                account = info.account if login == own_route else 0
                signed = self.signed_in(info.model, login, account)
                title = mc.LOGIN_TITLES.get(login, login)
                slot = f" #{account}" if account else ""
                label = f"{title}{slot} ✓" if signed else (
                    "Sign in with ChatGPT" if login == "authgpt" else f"Sign in with {title}")
                chips.append(ft.Chip(label=ft.Text(label), leading=ft.Icon(ft.Icons.ACCOUNT_CIRCLE_OUTLINED, size=18),
                                     on_click=lambda e, r=login, a=account: self.open_auth_menu(r, a),
                                     key=f"route-login-{login}"))
                chips.append(self._slot_menu(login, account))
                if login == "authgem" and signed and self.env.gemini_status is not None:
                    chips.append(ft.IconButton(icon=ft.Icons.INSIGHTS, tooltip="📊 Gemini status (quota, verification)",
                                               on_click=lambda e, a=account: self._gemini_status(a),
                                               size_constraints=HIT_TARGET, key="route-authgem-status"))
            controls.append(ft.Row([*chips, ft.TextButton(content="Accounts", on_click=lambda e: self._accounts())],
                                   wrap=True, spacing=6))
        if info.needs_key and self.env.store is not None:
            from glossarion_mobile.ui.screens.key_editor import KeyField

            store = self.env.store
            self.key_field = KeyField(
                value=str(store.get("api_key", "") or ""),
                label="API key",
                on_change=lambda value: store.set("api_key", value),
                on_test=(lambda value: self.env.test_key(value, info.model)) if self.env.test_key is not None else None,
                read_clipboard=self.env.read_clipboard,
                copy_text=self.env.copy_text,
            )  # no explicit key: a rebuilt KeyField must not be reconciled (and frozen) into the old one
            controls.append(self.key_field)
        if info.poe:
            controls.append(ft.FilledTonalButton(content="Poe setup", icon=ft.Icons.COOKIE_OUTLINED,
                                                 on_click=lambda e: self.open_poe_setup(), key="route-poe"))
        ctx = self.env.ctx
        tile_keys = []
        if info.google_creds and info.google_creds_text:
            color = semantic("success") if info.google_creds_level == "ready" else (
                ft.Colors.ERROR if info.google_creds_level == "error" else semantic("warning"))
            controls.append(ft.Text(info.google_creds_text, color=color, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    key="route-gcreds-status"))
        if info.google_creds:
            tile_keys.append("google_cloud_credentials")
        if info.vertex_location:
            tile_keys.append("vertex_ai_location")
        if info.gcp_project:
            picker = self._gcp_picker(info.model)
            if picker is not None:
                controls.append(picker)
            else:  # no OAuthBridge in this session: the plain project id tile
                tile_keys.append("authgem_project")
        for key in tile_keys:
            tile = self._schema_tile(key)
            if tile is not None:
                controls.append(tile.control)
            elif ctx is None:
                controls.append(ft.TextButton(content=f"Set {key} in Settings",
                                              on_click=lambda e: self._nav("settings")))
        return controls

    def _gcp_picker(self, model: str) -> Any:
        """The GCP project picker control for ``model`` (built once per sheet and model), else None."""
        factory = self.env.gcp_project_picker
        if factory is None:
            return None
        cached = getattr(self, "_gcp_picker_cache", None)
        if cached is not None and cached[0] == model:
            return cached[1]
        try:
            picker = factory(model)
            control = picker.build() if picker is not None else None
        except Exception:
            log.exception("building the GCP project picker failed")
            return None
        self.gcp_picker = picker
        self._gcp_picker_cache = (model, control)
        return control

    def _schema_tile(self, key: str) -> Any:
        """The shared settings tile for ``key``, built once per sheet (tiles mutate in place)."""
        cached = self._tiles.get(key)
        if cached is not None:
            cached.refresh(push=False)
            return cached
        ctx = self.env.ctx
        if ctx is None:
            return None
        try:
            spec = ctx.schema.spec(key)
        except Exception:
            spec = None
        if spec is None:
            return None
        from glossarion_mobile.ui.settings.tiles import make_tile

        try:
            tile = make_tile(spec, ctx)
        except Exception:
            log.exception("building the %s tile failed", key)
            return None
        self._tiles[key] = tile
        return tile

    def _refresh_thinking(self, info: Optional[RouteInfo]) -> None:
        family = info.family if info is not None and not info.excluded else None
        self._refresh_other_thinking(family)
        keys = THINKING_FIELDS.get(family or "", ())
        if not keys:
            self.thinking.visible = False
            self.thinking.controls = []
            return
        self.thinking.visible = True
        self.thinking.title = ft.Text(f"Thinking & effort · {FAMILY_TITLES.get(family, family)}")
        controls: list = []
        self.thinking_tiles = {}
        for key in keys:
            tile = self._schema_tile(key)
            if tile is not None:
                self.thinking_tiles[key] = tile
                controls.append(tile.control)
        if not controls:
            controls.append(ft.TextButton(content="Open Settings › Thinking & reasoning",
                                          on_click=lambda e: self._nav("settings.section", {"section": "thinking"})))
        controls.append(ft.Text("Thinking values are global settings (Chat settings › Disable all thinking turns "
                                "them off for one chat).", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        self.thinking.controls = controls

    def _refresh_other_thinking(self, family: Optional[str]) -> None:
        """One collapsed section per other thinking family (Gemini, Anthropic, DeepSeek, GPT…) plus the shared
        thinking settings, so every desktop thinking section is reachable from the model sheet."""
        sections = [(name, f"Thinking · {FAMILY_TITLES.get(name, name)}", keys)
                    for name, keys in THINKING_FIELDS.items() if name != family]
        sections.append(("general", "Thinking · All models", GENERAL_THINKING_FIELDS))
        tiles_out: list = []
        self.other_thinking_tiles = {}
        for name, title, keys in sections:
            controls = []
            for key in keys:
                tile = self._schema_tile(key)
                if tile is not None:
                    self.other_thinking_tiles[key] = tile
                    controls.append(tile.control)
            if controls:
                tiles_out.append(ft.ExpansionTile(title=ft.Text(title), controls=controls, expanded=False,
                                                  key=f"thinking-{name}",
                                                  controls_padding=ft.Padding.only(left=4, right=4, bottom=8)))
        self.thinking_others.controls = tiles_out
        self.thinking_others.visible = bool(tiles_out)

    # ---- actions --------------------------------------------------------------------------------------

    def _on_tab(self, e: Any = None) -> None:
        selected = list(getattr(e.control, "selected", []) or []) if e is not None else []
        self.tab = selected[0] if selected else "model"
        self.query = ""
        self.search.value = ""
        self.refresh()
        self._push_all()

    def select(self, field_name: str, value: str) -> None:
        if field_name == "model" and excluded_route(value):
            return
        self.current[field_name] = value
        self.selected_value = value
        if field_name == "model":
            self._remember(RECENTS_PREF, value)
        elif field_name == "language":
            self._remember(LANG_RECENTS_PREF, value)
            remember_language(self.env.prefs, value, self.languages)
        chat_scope = bool(self.chat_switch.value) and not self.one_shot
        # Close first: on_select may open the next sheet or a snackbar (a one-shot send opens the
        # Manual glossary sheet), which a later close must not take down instead of this one.
        self.close()
        if self.on_select is not None:
            self.on_select(field_name, value, chat_scope)

    def _slot_menu(self, route: str, account: int) -> ft.Control:
        """Appendix A LoginChip slot menu "#1 ▾": the signed-in slots (use that account's model alias)
        and "+ Add account" (the next free slot's LoginSheet)."""
        items: list = []
        current = str(self.current.get("model") or "")
        rest = current.split("/", 1)[1] if "/" in current else ""
        for slot in self.signed_in_slots(route):
            alias = f"{route}{slot}/{rest}" if slot else f"{route}/{rest}"
            items.append(ft.PopupMenuItem(content=f"#{slot or 1} · {alias}" + (" ✓" if slot == account else ""),
                                          on_click=lambda e, m=alias: self.select("model", m)))
        items.append(ft.PopupMenuItem(content="+ Add account", icon=ft.Icons.PERSON_ADD_ALT,
                                      on_click=lambda e, r=route: self._add_account(r)))
        return ft.PopupMenuButton(icon=ft.Icons.ARROW_DROP_DOWN, tooltip="Account slots", items=items,
                                  key=f"route-slots-{route}")

    def _add_account(self, route: str) -> None:
        slot = 1
        next_slot = self.env.next_slot
        if next_slot is not None:
            try:
                slot = int(next_slot(route))
            except Exception:
                log.debug("next account slot failed for %s", route, exc_info=True)
        else:
            known = self.signed_in_slots(route)
            slot = (max(known) + 1) if known else 1
        self._sign_in_route(route, slot)

    def _sign_in(self, model: str) -> None:
        route, account = mc.login_route(model)
        self._sign_in_route(route or "authgpt", account)

    def open_auth_menu(self, current: str, account: int = 0) -> ActionSheet:
        """The route row's sign-in chip (U11 item 7): switch between sign-ins without the login page. Each
        sign-in that is signed in switches the model to its most cost-efficient one
        (``recommended_model``); one that is not opens its sign-in. Signing in again (a refresh) and the
        Accounts page are rows of the same menu."""
        try:
            signed_keys = set(self.env.signed_in_keys() or ()) if self.env.signed_in_keys is not None else set()
        except Exception:
            signed_keys = set()
        models = getattr(self.snapshot, "models", ()) or ()
        items: list = []
        for route in mc.AUTH_ROUTES:
            title = mc.LOGIN_TITLES.get(route, route)
            signed = any(re.fullmatch(route + r"\d*", str(k)) for k in signed_keys) or (
                route == current and self.signed_in(self.current.get("model", ""), route, account))
            best = mc.recommended_model(route, models)
            mark = " (current)" if route == current else ""
            if signed and best:
                items.append(ActionItem(f"{title} ✓ · {best}{mark}", lambda m=best: self.select("model", m),
                                        icon="ACCOUNT_CIRCLE", key=f"auth-menu-{route}"))
            elif signed:
                items.append(ActionItem(f"{title} ✓{mark}", None, icon="ACCOUNT_CIRCLE", key=f"auth-menu-{route}",
                                        disabled_reason="Poll providers to list its models"))
            else:
                items.append(ActionItem(f"{title} · Sign in{mark}", lambda r=route: self._sign_in_route(r, 0),
                                        icon="LOGIN", key=f"auth-menu-{route}"))
        title = mc.LOGIN_TITLES.get(current, current)
        slot = f" #{account}" if account else ""
        items.append(ActionItem(f"Sign in to {title}{slot} again", lambda: self._sign_in_route(current, account),
                                icon="REFRESH", key="auth-menu-refresh"))
        items.append(ActionItem("Accounts…", self._accounts, icon="MANAGE_ACCOUNTS", key="auth-menu-accounts"))
        sheet = ActionSheet(items, title="Switch sign-in", subtitle="Uses each sign-in's most cost-efficient model",
                            tablet=bool(self.env.tablet()))
        if self._page is not None:
            sheet.show(self._page)
        self.action_sheet = sheet
        return sheet

    def _sign_in_route(self, route: str, account: int = 0) -> None:
        self.close()
        if self.env.sign_in is not None:
            call_handler(self.env.sign_in, route, account)
        elif self.on_sign_in is not None:
            call_handler(self.on_sign_in)
        else:
            self._nav("settings.accounts")

    def _accounts(self) -> None:
        self.close()
        if self.on_accounts is not None:
            call_handler(self.on_accounts)
        else:
            self._nav("settings.accounts")

    def _gemini_status(self, account_id: int) -> Any:
        """The route row's 📊: the same Gemini status sheet as Accounts › slot ⋯ › 📊 Status."""
        handler = self.env.gemini_status
        if handler is None:
            return None
        result = handler(int(account_id or 0))
        if asyncio.iscoroutine(result):
            spawn = self.env.spawn
            return spawn(result) if spawn is not None else asyncio.ensure_future(result)
        return result

    def _nav(self, route_name: str, params: Optional[dict] = None) -> None:
        self.close()
        navigate = self.env.navigate
        if navigate is not None:
            if params:
                navigate(route_name, params)
            else:
                navigate(route_name)

    def _say(self, message: str) -> None:
        if self.env.notify is not None:
            try:
                self.env.notify(message)
            except Exception:
                pass

    def _toggle_hide_unpolled(self, e: Any = None) -> None:
        value = not bool(self.snapshot.hide_unpolled)
        catalog = self.env.catalog
        if catalog is not None:
            catalog.set_hide_unpolled(value)
            self.snapshot = catalog.snapshot
        elif self.env.store is not None:
            self.env.store.set("model_manager_hide_unpolled_models", value)
            self.snapshot = dataclasses.replace(self.snapshot, hide_unpolled=value)
        self.refresh()
        self._push_all()

    def refresh_selected_provider(self) -> Any:
        provider = mc.provider_of(self.current["model"], list(self.snapshot.custom_routes))
        catalog = self.env.catalog
        if catalog is not None:
            try:
                provider = catalog.options.catalog_provider_for_model(self.current["model"], list(self.snapshot.custom_routes)) or provider
            except Exception:
                pass
        return self.refresh_provider(provider)

    def refresh_provider(self, provider: str) -> Any:
        catalog = self.env.catalog
        if catalog is None:
            self._say("Online model refresh needs the model catalog service")
            return None
        if mc.provider_excluded(provider):
            self._say(f"{mc.provider_label(provider)} isn't available on mobile")
            return None

        async def run() -> Any:
            outcome = await catalog.refresh(provider, explicit=True)
            if outcome.message:
                self._say(outcome.message)
            return outcome

        spawn = self.env.spawn
        coro = run()
        if spawn is not None:
            return spawn(coro)
        import asyncio

        return asyncio.ensure_future(coro)

    def open_row_actions(self, model: str) -> ActionSheet:
        favorite = model in self.favorites()
        provider = mc.provider_of(model, list(self.snapshot.custom_routes))
        catalog_provider = None
        if self.env.catalog is not None:
            try:
                catalog_provider = self.env.catalog.options.catalog_provider_for_model(model, list(self.snapshot.custom_routes))
            except Exception:
                catalog_provider = None
        items = [
            ActionItem("Remove from favorites" if favorite else "★ Favorite", lambda: self.toggle_favorite(model),
                       icon="STAR_OUTLINE"),
            ActionItem("Copy id", lambda: self._copy(model), icon="CONTENT_COPY"),
            ActionItem("Provider info", lambda: self.open_provider_info(provider=provider), icon="INFO_OUTLINE"),
            ActionItem("Refresh this provider", (lambda: self.refresh_provider(catalog_provider)) if catalog_provider else None,
                       icon="PUBLIC", disabled_reason=None if catalog_provider else "No online catalog for this route"),
        ]
        sheet = ActionSheet(items, title=model, subtitle=mc.provider_label(provider), tablet=bool(self.env.tablet()))
        if self._page is not None:
            sheet.show(self._page)
        self.action_sheet = sheet
        return sheet

    def _copy(self, text: str) -> Any:
        if self.env.copy_text is not None:
            return call_handler(self.env.copy_text, text)
        return None

    def open_provider_info(self, e: Any = None, *, provider: Optional[str] = None) -> InfoSheet:
        """ⓘ: the desktop "Model Provider Information" (shared text) with the provider's catalog status."""
        title = "Model Provider Information" if provider is None else mc.provider_label(provider)
        body = provider_info_markdown()
        if provider is not None:
            status = self.snapshot.statuses.get(provider)
            body = (f"**Catalog status:** {status}\n\n" if status else "") + body
        sheet = InfoSheet(title=title, body=body, markdown=True)
        if self._page is not None:
            sheet.show(self._page)
        return sheet

    def open_poe_setup(self) -> "PoeSetupSheet":
        sheet = PoeSetupSheet(env=self.env, model=self.current["model"])
        if self._page is not None:
            sheet.show(self._page)
        self.poe_sheet = sheet
        return sheet

    # ---- presentation ---------------------------------------------------------------------------------

    def show(self, page: Any) -> None:
        self._page = page
        height = float(getattr(page, "height", 0) or 0)
        if height and height < 640:
            self.dialog.fullscreen = True
        try:
            tablet = bool(self.env.tablet())
        except Exception:
            tablet = False
        if tablet:  # UI_SPEC §2.2: a 420 dp panel on tablets (Flet has no anchored popover)
            self.dialog.size_constraints = ft.BoxConstraints(max_width=420)
        if height:
            self.list_view.height = max(240, min(520, height * 0.48))
        catalog = self.env.catalog
        if catalog is not None:
            self._unsub = catalog.subscribe(self.apply_snapshot)
        page.show_dialog(self.dialog)
        self._start_background()

    def _start_background(self) -> None:
        """Load the catalog / route info off the loop and auto-poll the selected provider (24 h)."""
        catalog = self.env.catalog
        spawn = self.env.spawn
        if spawn is None:
            return

        async def work() -> None:
            run_io = self.env.run_io
            if catalog is not None and not catalog.snapshot.loaded:
                try:
                    await catalog.load()
                except Exception:
                    log.exception("loading the model catalog failed")
            model = self.current["model"]
            if run_io is not None:
                try:
                    config = self.env.store.snapshot() if self.env.store is not None else {}
                    info = await run_io(mc.route_info, model, config)
                    self.set_route(info)
                except Exception:
                    log.debug("route info failed", exc_info=True)
                try:
                    models = list(self.snapshot.models)
                    routes = list(self.snapshot.custom_routes)
                    keyless = await run_io(lambda: {m for m in models if not mc.model_needs_api_key(m, routes)})
                    self.set_keyless(keyless)
                except Exception:
                    log.debug("key classification failed", exc_info=True)
            if catalog is not None and model:
                try:
                    outcome = await catalog.maybe_auto_poll(model)
                    if outcome is not None and outcome.message and outcome.provider:
                        self._say(outcome.message)  # model_catalog_core.auto_poll_message
                except Exception:
                    log.debug("auto-poll failed", exc_info=True)

        spawn(work())

    def close(self) -> None:
        self.closed = True
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        close_dialog(self._page, self.dialog)


#: U3 name (``ui.sheets.model_sheet_min.ModelSheetMin``); the full sheet now.
ModelSheetMin = ModelSheet


# ---- field mode -----------------------------------------------------------------------------------


class ModelPicker(ft.Container):
    """Read-only model field with ▾ that opens the ModelSheet in field mode (UI_SPEC §2.2)."""

    def __init__(self, *, value: str = "", label: str = "Model", on_change: Optional[Callable[[str], Any]] = None,
                 env: Optional[SheetEnv] = None, page: Any = None, key: Optional[str] = None,
                 disabled: bool = False) -> None:
        super().__init__(key=key)
        self.value = value
        self.label = label
        self.on_change_value = on_change
        self.picker_env = env
        self._picker_page = page
        self.sheet: Optional[ModelSheet] = None
        self.field = ft.TextField(value=value, label=label, read_only=True, dense=True,
                                  suffix=ft.Icon(ft.Icons.ARROW_DROP_DOWN), on_click=self.open,
                                  border_radius=tokens.RADII["field"], disabled=disabled)
        self.reason = ReasonChip(reason="Not available on mobile", detail=excluded_detail(value)) \
            if excluded_route(value) else None
        self.content = ft.Column([c for c in (self.field, self.reason) if c is not None], spacing=4, tight=True)

    def open(self, e: Any = None) -> ModelSheet:
        sheet = ModelSheet(current_model=self.value, env=self.picker_env, field_mode=True,
                           on_select=lambda _f, value, _chat: self.set_value(value))
        self.sheet = sheet
        page = self._picker_page
        if page is None:
            try:
                page = self.page  # raises until the field is on a page
            except Exception:
                page = None
        if page is not None:
            sheet.show(page)
        return sheet

    def set_value(self, value: str) -> None:
        self.value = value
        self.field.value = value
        _push(self.field)
        if self.on_change_value is not None:
            call_handler(self.on_change_value, value)


# ---- Poe -------------------------------------------------------------------------------------------

POE_STEPS = (
    "1. Go to poe.com and LOG IN to your account",
    "2. Open the browser's developer tools (a desktop browser is easiest)",
    "3. Navigate to the cookies of https://poe.com (Chrome/Edge: Application → Cookies; "
    "Firefox: Storage → Cookies; Safari: Storage → Cookies)",
    "4. Find the cookie named 'p-b'",
    "5. Copy its value",
    "6. Paste it below: Glossarion stores it as p-b:<value> in the API key",
)
POE_NOTE = ("Note: The cookie value is usually a long string ending with %3D%3D\n"
            "If you see multiple p-b cookies, use the one with the longest value.")
POE_URL = "https://poe.com"


class PoeSetupSheet:
    """POE Authentication Required (desktop ``_show_poe_setup_dialog``): the p-b cookie goes into
    the API key as ``p-b:<value>``. The route is deprecated; the guide link and Test stay."""

    def __init__(self, *, env: Optional[SheetEnv] = None, model: str = "poe/") -> None:
        self.env = env or sheet_env()
        self.model = model
        self._page: Any = None
        store = self.env.store
        current = str(store.get("api_key", "") or "") if store is not None else ""
        cookie = current[4:] if current.startswith("p-b:") else ""
        from glossarion_mobile.ui.screens.key_editor import KeyField

        self.field = KeyField(value=cookie, label="p-b cookie value", on_change=self.save,
                              on_test=(lambda value: self.env.test_key(self.api_key_for(value), self.model))
                              if self.env.test_key is not None else None,
                              read_clipboard=self.env.read_clipboard, copy_text=self.env.copy_text, key="poe-cookie")
        missing = dependency_reason("poe/")
        self.missing = missing
        if missing is not None:  # dependency rule: shown, disabled, the cookie kept
            self.field.disabled = True
        controls: list = [
            ft.Text("POE Cookie Authentication", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            *([ft.Row([ReasonChip(reason=missing[0], detail=missing[1])], key="poe-missing")] if missing else []),
            ft.Row([ft.Icon(ft.Icons.WARNING_AMBER, color=semantic("warning")),
                    ft.Text("The poe/ route is deprecated and may stop working.", expand=True)]),
            ft.Text("⚠️ POE uses HttpOnly cookies that cannot be accessed by JavaScript", color=ft.Colors.ERROR,
                    weight=ft.FontWeight.W_600),
            ft.Text("You must manually copy the cookie from Developer Tools", color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text("How to Get Your POE Cookie", weight=ft.FontWeight.W_600),
            *[ft.Text(step, theme_style=ft.TextThemeStyle.BODY_SMALL) for step in POE_STEPS],
            ft.Text(POE_NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            self.field,
            ft.Row([ft.TextButton(content="Open poe.com", icon=ft.Icons.OPEN_IN_NEW, url=POE_URL),
                    ft.TextButton(content="Close", on_click=lambda e: self.close())], wrap=True),
        ]
        self.dialog = ft.BottomSheet(
            content=sheet_frame(scroll_column(controls, spacing=8), padding=ft.Padding.only(left=16, right=16, bottom=24)),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    @staticmethod
    def api_key_for(cookie: str) -> str:
        value = str(cookie or "").strip()
        if not value:
            return ""
        return value if value.startswith("p-b:") else f"p-b:{value}"

    def save(self, cookie: str) -> None:
        store = self.env.store
        if store is None:
            return
        value = self.api_key_for(cookie)
        if value:
            store.set("api_key", value)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
