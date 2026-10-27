"""MiniModelSheet: the composer's mini model picker (hotfix 5).

Owner: "next to output mode. Add a mini model selector. It basically only shows models from the current
model provider and behaves like our current claude model selector ..., where sliding up gracefully expands
it, and a button at the bottom to redirect to the full model picker page"; then: "Include the thinking
efforts associated with that model. Synced to the model picker settings menu ... Keep this compact.
Effort dropdown + pass thinking to unknown routes toggle".

The composer's model chip (the model's short name + ▾, ``Composer.model_chip``) opens it:

* the open chat's model provider only (``model_catalog.provider_of``; numbered account aliases such as
  ``authgpt2/…`` belong to their route), polled models first like that provider's group in the full
  sheet, the chat's model ticked, names without the provider's own route prefix;
* pinned at the bottom: the model's Effort (``services.thinking_effort``: Off up to what the model supports,
  the same global keys as the full sheet's "Thinking & effort" and Settings › Thinking & reasoning), for an
  unknown route the "Pass thinking to unknown routes" switch (``pass_thinking_all_openai``) above it, and
  "All models", which closes this sheet and opens the full ModelSheet;
* on the phone app the rows sit in the extension's ``DraggableSheet`` (Flutter's DraggableScrollableSheet):
  the sheet opens at about half the screen, a swipe up grows it to full height before the rows scroll, a
  swipe down shrinks it and closes it. Without the extension (desktop ``flet run``) the same rows sit in an
  ordinary scrolling sheet.

Everything else is the full sheet's own (``ModelSheet`` is the base class): the rows (``_model_row``:
status dot, Sign in / Add key, long-press actions), the catalog subscription, the background load and the
24 h auto-poll of the selected provider, ``select`` (closes, then ``on_select(field, value,
this_chat_only)``) and ``close``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services import model_catalog as mc
from glossarion_mobile.services.thinking_effort import PASS_UNKNOWN_KEY, PASS_UNKNOWN_LABEL, EffortControl, effort_control
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.drag_sheet import draggable_sheet_class, height_fraction
from glossarion_mobile.ui.components.sheet import scroll_column, sheet_frame
from glossarion_mobile.ui.sheets.model_sheet import MAX_ROWS, ModelSheet, _push
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["MiniModelSheet", "short_model_name"]

log = logging.getLogger("glossarion.models")


def short_model_name(model: str, provider: str) -> str:
    """``model`` without its provider's own route prefix (``authgpt/gpt-6-luna`` -> ``gpt-6-luna``); an
    account alias (``authgpt2/…``) or a bare id keeps its name."""
    text = str(model or "")
    prefix = provider[7:] if provider.startswith("custom:") else (f"{provider}/" if provider else "")
    if prefix and text.lower().startswith(prefix.lower()) and len(text) > len(prefix):
        return text[len(prefix):]
    return text


#: the mini sheet's title per tab (hotfix 5: the Plan card's profile / language chips open it too, owner: "should
#: use a preview like the + button does. Only showing the configured profiles, not a full menu. Sames goes for the
#: language button ..., but both allowing custom options when scrolling to the bottom")
MINI_TITLES = {"model": "Select model", "profile": "Prompt profile", "language": "Target language"}
#: estimated row heights (dp) per tab: a profile row carries a 3-line prompt preview
MINI_ROW_HEIGHTS = {"model": 64.0, "profile": 96.0, "language": 52.0}


class MiniModelSheet(ModelSheet):
    #: rows the half-height sheet is sized for (a longer provider list grows with a swipe up)
    COLLAPSED_ROWS = 5
    ROW_HEIGHT = 64.0
    HEADER_HEIGHT = 64.0
    FOOTER_HEIGHT = 76.0
    SWITCH_HEIGHT = 48.0

    def __init__(self, *, on_all_models: Optional[Callable[[], Any]] = None, **kwargs: Any) -> None:
        self.on_all_models = on_all_models
        self.provider = ""
        self.row_controls: list = []
        self.mini_body: Any = None
        self.native = False
        self._providers: dict = {}  # model -> provider, across refreshes (the catalog can hold thousands)
        self.effort: Optional[EffortControl] = None
        self._prepared = False
        super().__init__(**kwargs)

    # ---- building ----------------------------------------------------------------------------------

    def _build(self) -> None:
        super()._build()  # the full sheet's controls: the catalog updates and the base methods use them
        self.mini_title = ft.Text(MINI_TITLES.get(self.tab, "Select model"), theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                  weight=ft.FontWeight.W_600)
        self.mini_subtitle = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        header: list = [self.mini_title, self.mini_subtitle]
        if self.tab in ("profile", "language"):
            # the full sheet's "Apply to this chat only" (a profile / language may still apply to every chat)
            self.chat_switch.visible = True
            header.append(self.chat_switch)
        self.mini_header = ft.Container(content=ft.Column(header, spacing=0, tight=True),
                                        padding=ft.Padding.only(left=16, right=16, top=4, bottom=8),
                                        key="mini-model-header")
        self.custom_language: Optional[ft.TextField] = None
        self.effort_dropdown = ft.Dropdown(label="Effort", dense=True, expand=True, options=[], visible=False,
                                           on_select=self._on_effort, key="mini-model-effort")
        self.pass_switch = ft.Switch(label=PASS_UNKNOWN_LABEL, value=False, visible=False,
                                     on_change=self._on_pass_unknown, key="mini-model-pass-thinking")
        self.all_button = ft.FilledTonalButton(content="All models", icon=ft.Icons.APPS, on_click=self.open_all_models,
                                               key="mini-model-all")
        self.mini_footer = ft.SafeArea(
            content=ft.Container(
                content=ft.Column([self.pass_switch,
                                   ft.Row([self.effort_dropdown, self.all_button], spacing=8,
                                          alignment=ft.MainAxisAlignment.END,
                                          vertical_alignment=ft.CrossAxisAlignment.CENTER)],
                                  spacing=4, tight=True),
                padding=ft.Padding.only(left=16, right=16, top=8, bottom=8),
            ),
            avoid_intrusions_top=False,
            key="mini-model-footer",
        )
        self._sync_effort()

    def _provider_of(self, model: str, routes: list) -> str:
        provider = self._providers.get(model)
        if provider is None:
            provider = self._providers[model] = mc.provider_of(model, routes)
        return provider

    def refresh(self) -> None:
        self.rows = {}
        self._key_cache = None
        self._favorites_cache = None
        if getattr(self.snapshot, "custom_routes", ()) != getattr(self, "_routes_seen", None):
            self._providers = {}  # a custom route change can move models between providers
            self._routes_seen = self.snapshot.custom_routes
        if self.tab == "profile":
            # the configured profiles; the role, the prefill and Edit / New / Manage at the end of the list
            self.mini_subtitle.value = f"Now: {self.current['profile']}" if self.current["profile"] else ""
            self.row_controls = self.profile_rows()
        elif self.tab == "language":
            self.mini_subtitle.value = f"Now: {self.current['language']}" if self.current["language"] else ""
            self.row_controls = self.mini_language_rows()
        else:
            model = self.current["model"]
            routes = list(self.snapshot.custom_routes)
            self.provider = self._provider_of(model, routes) if model else ""
            self._sync_poll_state()
            self.row_controls = self.model_rows()
        if self.mini_body is not None:
            self.mini_body.controls = self.row_controls

    def mini_language_rows(self) -> list:
        """The target languages (recent first), and at the end any other language typed in (a model writes it)."""
        rows = [row for row in self.language_rows() if getattr(row, "key", None) != "language-add"]
        self.custom_language = ft.TextField(hint_text="Another language", dense=True, expand=True,
                                            on_submit=lambda e: self.use_custom_language(), key="language-custom")
        rows.append(ft.Container(
            content=ft.Row([self.custom_language,
                            ft.IconButton(icon=ft.Icons.CHECK, tooltip="Use this language", size_constraints=HIT_TARGET,
                                          on_click=lambda e: self.use_custom_language(), key="language-custom-use")],
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.only(left=16, right=8, top=4, bottom=8), key="language-custom-row"))
        return rows

    def use_custom_language(self) -> Optional[str]:
        value = str(getattr(self.custom_language, "value", "") or "").strip()
        if not value:
            return None
        self.select("language", value)
        return value

    def _sync_poll_state(self) -> None:
        """The provider under the title, with "checking online models…" while it polls (the full sheet's
        animated search border is not on this sheet)."""
        if not hasattr(self, "mini_subtitle") or self.tab != "model":
            return
        label = mc.provider_label(self.provider) if self.provider else "No model selected"
        busy = bool(self.provider) and (self.provider in self._refreshing or self.snapshot.is_polling(self.provider))
        self.mini_subtitle.value = f"{label} · checking online models…" if busy else label

    def model_rows(self) -> list:
        """The current provider's models: polled first (its group in the full sheet), the chat's model ticked."""
        self.group_headers = {}
        routes = list(self.snapshot.custom_routes)
        current = self.current["model"]
        models = [m for m in self.snapshot.visible_models() if self.provider and self._provider_of(m, routes) == self.provider]
        if current and current not in models:
            models.insert(0, current)
        groups = mc.group_models(models, routes, polled=self.is_polled)
        ordered = [m for group in groups for m in group.models]
        rows = [self._mini_row(m) for m in ordered[:MAX_ROWS]]
        if not rows:
            message = "Loading the model list…" if not self.snapshot.loaded else "No other models from this provider."
            rows.append(ft.Container(content=ft.Text(message, color=ft.Colors.ON_SURFACE_VARIANT),
                                     padding=ft.Padding.all(16), key="mini-models-empty"))
        return rows

    def _mini_row(self, model: str) -> ft.Control:
        row = self._model_row(model, "mini")
        name = short_model_name(model, self.provider)
        if name != model:
            row.title = ft.Text(name, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
        if model == self.current["model"] and row.trailing is None:
            row.trailing = ft.Icon(ft.Icons.CHECK, color=ft.Colors.PRIMARY)
        return row

    def set_route(self, info: Any) -> None:
        self.route = info  # no route row here; the rows' own Sign in / Add key cover it

    # ---- effort ------------------------------------------------------------------------------------

    def _effective(self, key: str) -> Any:
        store = self.env.store
        try:
            return store.effective(key)
        except Exception:
            return store.get(key) if store is not None else None

    def _sync_effort(self) -> None:
        store = self.env.store
        control = None
        if store is not None and self.tab == "model":  # the profile / language sheets have no Effort
            try:
                control = effort_control(self.current["model"], self._effective,
                                         custom_routes=list(self.snapshot.custom_routes))
            except Exception:
                log.exception("effort for %s failed", self.current["model"])
        self.effort = control
        self.effort_dropdown.visible = control is not None
        self.effort_dropdown.options = [ft.DropdownOption(key=c.value, text=c.label) for c in control.choices] \
            if control is not None else []
        self.effort_dropdown.value = control.value if control is not None else None
        self.pass_switch.visible = bool(control is not None and control.unknown)
        self.pass_switch.value = bool(control is not None and control.pass_unknown)

    def _write(self, values: dict) -> None:
        from glossarion_mobile.state.setting_writes import write_settings

        write_settings(self.env.store, values)

    def set_effort(self, value: str) -> bool:
        """Store the pick (the same keys as the full sheet's "Thinking & effort"); True when it was one."""
        choice = self.effort.choice(value) if self.effort is not None else None
        if choice is None:
            return False
        self._write(dict(choice.writes))
        self._sync_effort()
        _push(self.mini_footer)
        return True

    def set_pass_unknown(self, value: bool) -> None:
        self._write({PASS_UNKNOWN_KEY: bool(value)})
        self._sync_effort()
        _push(self.mini_footer)

    def _on_effort(self, e: Any = None) -> None:
        self.set_effort(str(getattr(getattr(e, "control", None), "value", "") or ""))

    def _on_pass_unknown(self, e: Any = None) -> None:
        self.set_pass_unknown(bool(getattr(getattr(e, "control", None), "value", False)))

    # ---- showing -----------------------------------------------------------------------------------

    def sheet_sizes(self, page: Any) -> tuple:
        """(initial, max) fractions of the sheet's height: about half the screen, sized for
        ``COLLAPSED_ROWS`` rows; full height when more rows wait (or the list is still loading)."""
        try:
            height = float(getattr(page, "height", 0) or 0)
        except (TypeError, ValueError):
            height = 0.0
        if height <= 0:
            return 0.5, 1.0
        model = self.tab == "model"
        footer = (self.FOOTER_HEIGHT + (self.SWITCH_HEIGHT if self.pass_switch.visible else 0.0)) if model else 0.0
        header = self.HEADER_HEIGHT + (0.0 if model else self.SWITCH_HEIGHT)  # "Apply to this chat only"
        row = MINI_ROW_HEIGHTS.get(self.tab, self.ROW_HEIGHT)
        count = len(self.row_controls)
        shown = max(1, min(count, self.COLLAPSED_ROWS))
        initial = height_fraction(page, header + shown * row + footer, low=0.3, high=0.6)
        everything = height_fraction(page, header + count * row + footer, low=0.0, high=100.0)
        loading = model and not self.snapshot.loaded
        maximum = 1.0 if (loading or everything > initial) else initial
        return initial, maximum

    def _make_dialog(self, page: Any) -> ft.BottomSheet:
        sheet_cls = draggable_sheet_class(page)
        self.native = sheet_cls is not None
        if sheet_cls is not None:
            initial, maximum = self.sheet_sizes(page)
            footer = self.mini_footer if self.tab == "model" else None  # Effort + All models
            self.mini_body = sheet_cls(controls=self.row_controls, header=self.mini_header, footer=footer,
                                       initial_size=initial, min_size=round(initial / 2, 3), max_size=maximum,
                                       key="mini-model-sheet")
            content: ft.Control = self.mini_body
        else:  # desktop flet run: an ordinary scrolling sheet
            self.mini_body = ft.Column(self.row_controls, spacing=0, tight=True, key="mini-model-sheet")
            footer_rows = [self.mini_footer] if self.tab == "model" else None
            content = sheet_frame(scroll_column([self.mini_header, self.mini_body], footer=footer_rows, spacing=0),
                                  padding=ft.Padding.only(bottom=8))
        return ft.BottomSheet(content=content, show_drag_handle=True, scrollable=True, draggable=True,
                              bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def prepare(self, page: Any) -> ft.BottomSheet:
        """Build the sheet for ``page`` (``show`` does it when nobody did), so a caller can chain its own
        ``on_dismiss`` before it opens (Flet keeps the handler present when a dialog opens)."""
        self._page = page
        self.dialog = self._make_dialog(page)
        # swiped away / back: catalog updates stop repainting a closed sheet
        self.dialog.on_dismiss = self._on_dismiss
        self._prepared = True
        return self.dialog

    def show(self, page: Any) -> None:
        if not getattr(self, "_prepared", False) or self._page is not page:
            self.prepare(page)
        try:
            tablet = bool(self.env.tablet())
        except Exception:
            tablet = False
        if tablet:  # the full sheet's 420 dp panel
            self.dialog.size_constraints = ft.BoxConstraints(max_width=420)
        if self.tab != "model":  # a profile / language: no catalog to follow or load
            page.show_dialog(self.dialog)
            return
        catalog = self.env.catalog
        if catalog is not None:
            self._unsub = catalog.subscribe(self.apply_snapshot)
        page.show_dialog(self.dialog)
        self._start_background()

    def open_all_models(self, e: Any = None) -> None:
        """"All models": this sheet closes and the full ModelSheet opens."""
        self.close()
        call_handler(self.on_all_models)
