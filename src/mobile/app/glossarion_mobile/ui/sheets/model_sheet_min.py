"""Minimal ModelSheet (U3; the full sheet of UI_SPEC §2.2 arrives in U4).

Tabs Model · Profile · Language (a SegmentedButton), opened from the header subtitle
spans. The Model tab searches the catalog the desktop picker shows -
``model_options.merge_saved_model_options(config['custom_model_list'],
model_options.get_model_options(), config['model_manager_removed_models'])`` - with
the current model first, a status dot per row (green: ready · amber: needs ChatGPT
sign-in, with an inline "Sign in with ChatGPT" button · grey + ReasonChip: route
excluded on mobile; the row stays visible and a desktop value is preserved). The
"Apply to this chat only" switch writes the chat override instead of the global
``model`` key. Profile lists the prompt profiles (``active_profile``); Language the
shared ``language_options.TARGET_LANGUAGES`` (``output_language``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.chat.send_state import requires_chatgpt_sign_in
from glossarion_mobile.ui.components.reason_chip import ReasonChip


__all__ = ["EXCLUDED_ROUTE_PREFIXES", "ModelSheetMin", "excluded_route", "filter_models", "load_model_catalog"]

#: Plan "Excluded on mobile" routes (npm/bun/desktop-binary).
EXCLUDED_ROUTE_PREFIXES = ("antigravity/", "ocagy", "ocz/", "authza", "autharena", "search/opera", "ollamapull/")
MAX_ROWS = 300


def excluded_route(model: Optional[str]) -> Optional[str]:
    value = str(model or "").strip().lower()
    for prefix in EXCLUDED_ROUTE_PREFIXES:
        if value.startswith(prefix):
            route = value.split("/", 1)[0] + "/" if "/" in value else value
            return f"{route} isn't available on mobile"
    return None


def load_model_catalog(config_get: Callable[[str, Any], Any]) -> list:
    """Blocking: the desktop picker list (saved custom list + catalog - removed)."""
    import model_options  # backend module

    custom = config_get("custom_model_list", None)
    removed = config_get("model_manager_removed_models", []) or []
    return list(
        model_options.merge_saved_model_options(
            custom if isinstance(custom, list) else None, model_options.get_model_options(), removed
        )
    )


def filter_models(models: Sequence[str], query: str, current: str = "", limit: int = MAX_ROWS) -> list:
    """Current model first, then case-insensitive substring matches in catalog order."""
    needle = str(query or "").strip().casefold()
    out: list = []
    seen: set = set()
    if current and (not needle or needle in current.casefold()):
        out.append(current)
        seen.add(current.casefold())
    for model in models:
        key = str(model).casefold()
        if key in seen or (needle and needle not in key):
            continue
        out.append(model)
        seen.add(key)
        if len(out) >= limit:
            break
    return out


class ModelSheetMin:
    def __init__(
        self,
        *,
        current_model: str,
        current_profile: str,
        current_language: str,
        models: Sequence[str] = (),
        profiles: Sequence[str] = (),
        languages: Sequence[str] = (),
        signed_in: Callable[[str], bool] = lambda model: False,
        tab: str = "model",
        chat_scope: bool = False,
        on_select: Optional[Callable[[str, str, bool], Any]] = None,  # (field, value, this_chat_only)
        on_sign_in: Optional[Callable[[], Any]] = None,
        on_accounts: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.current = {"model": current_model, "profile": current_profile, "language": current_language}
        self.models = list(models)
        self.profiles = list(profiles)
        self.languages = list(languages)
        self.signed_in = signed_in
        self.on_select = on_select
        self.on_sign_in = on_sign_in
        self.on_accounts = on_accounts
        self.tab = tab if tab in ("model", "profile", "language") else "model"
        self.query = ""
        self._page: Any = None
        self.chat_switch = ft.Switch(label="Apply to this chat only", value=chat_scope)
        self.tabs = ft.SegmentedButton(
            segments=[ft.Segment(value="model", label="Model"), ft.Segment(value="profile", label="Profile"),
                      ft.Segment(value="language", label="Language")],
            selected=[self.tab],
            on_change=self._on_tab,
        )
        self.search = ft.TextField(hint_text="Search models", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self.set_query(e.control.value or ""))
        self.list_view = ft.ListView(controls=[], height=420, build_controls_on_demand=True)
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=12, right=12, bottom=16),
                content=ft.Column(
                    [
                        ft.Row([ft.Text("Model", theme_style=ft.TextThemeStyle.TITLE_LARGE, expand=True),
                                ReasonChip(reason="Full model manager arrives in U4")]),
                        self.tabs,
                        self.search,
                        self.chat_switch,
                        self.list_view,
                        ft.Row(
                            [
                                ft.TextButton(content="Accounts", on_click=lambda e: self._accounts()),
                                ft.TextButton(content="Manage models", disabled=True),
                            ],
                            spacing=8,
                        ),
                    ],
                    tight=True,
                    spacing=8,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )
        self.rows: dict = {}
        self.refresh()

    # ---- rows -------------------------------------------------------------------------------

    def _model_row(self, model: str) -> ft.Control:
        reason = excluded_route(model)
        needs_sign_in = requires_chatgpt_sign_in(model) and not self.signed_in(model)
        if reason:
            dot, trailing = ft.Colors.OUTLINE, ReasonChip(reason="Not available on mobile", detail=reason)
        elif needs_sign_in:
            dot = ft.Colors.AMBER
            trailing = ft.TextButton(content="Sign in with ChatGPT", on_click=lambda e: self._sign_in())
        else:
            dot, trailing = ft.Colors.GREEN, None
        selected = model == self.current["model"]
        row = ft.ListTile(
            leading=ft.Container(width=10, height=10, border_radius=5, bgcolor=dot),
            title=ft.Text(model, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            trailing=trailing,
            selected=selected,
            dense=True,
            min_height=48,
            on_click=None if reason else (lambda e, m=model: self.select("model", m)),
            key=f"model-{model}",
        )
        self.rows[model] = row
        return row

    def _choice_row(self, field_name: str, value: str) -> ft.Control:
        row = ft.ListTile(
            title=ft.Text(value),
            leading=ft.Icon(ft.Icons.CHECK) if value == self.current[field_name] else None,
            selected=value == self.current[field_name],
            dense=True,
            min_height=48,
            on_click=lambda e, v=value: self.select(field_name, v),
            key=f"{field_name}-{value}",
        )
        self.rows[value] = row
        return row

    def refresh(self) -> None:
        self.rows = {}
        self.search.visible = self.tab == "model"
        if self.tab == "model":
            rows = [self._model_row(m) for m in filter_models(self.models, self.query, self.current["model"])]
        elif self.tab == "profile":
            names = self.profiles or [self.current["profile"]]
            rows = [self._choice_row("profile", p) for p in names]
        else:
            names = self.languages or [self.current["language"]]
            rows = [self._choice_row("language", lang) for lang in names]
        self.list_view.controls = rows

    def set_query(self, query: str) -> None:
        self.query = query
        self.refresh()
        self._push()

    def set_models(self, models: Sequence[str]) -> None:
        self.models = list(models)
        self.refresh()
        self._push()

    def _on_tab(self, e: Any = None) -> None:
        selected = list(getattr(e.control, "selected", []) or []) if e is not None else []
        self.tab = selected[0] if selected else "model"
        self.refresh()
        self._push()

    def select(self, field_name: str, value: str) -> None:
        self.current[field_name] = value
        if self.on_select is not None:
            self.on_select(field_name, value, bool(self.chat_switch.value))
        self.close()

    def _sign_in(self) -> None:
        self.close()
        if self.on_sign_in is not None:
            self.on_sign_in()

    def _accounts(self) -> None:
        self.close()
        if self.on_accounts is not None:
            self.on_accounts()

    # ---- presentation -----------------------------------------------------------------------

    def _push(self) -> None:
        try:
            self.dialog.update()
        except Exception:
            pass

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()
