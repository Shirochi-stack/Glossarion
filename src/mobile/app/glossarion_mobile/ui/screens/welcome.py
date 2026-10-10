"""First-run Welcome (``/welcome``, UI_SPEC §4.17): 5 steps, re-runnable from About.

1. **Welcome · Sign in** - ChatGPT, Claude, Gemini and Grok side by side (U11 item 1: no
   provider is pushed; LoginSheet through the OAuthBridge, Grok shows its device code). After a
   sign-in the provider's models are polled (``ModelCatalogService.refresh``): ChatGPT keeps the
   default ``authgpt/gpt-6-luna`` while the provider still lists it, otherwise (and for the other
   providers) the model becomes that provider's most cost-efficient one
   (``model_catalog.recommended_model``: Sonnet, the newest Flash, the newest Grok); "Change
   model" opens the ModelSheet on that provider's models. "Use an API key or a local model" goes
   to step 2; "Skip for now" moves on.
2. **Other providers** - choose another model (ModelSheet), paste an API key (the desktop main
   key field, ``api_key``), or set up a local Ollama / LM Studio host (Settings › Endpoints).
3. **Target language and glossary mode** - the desktop first-run "Choose Your
   Glossary Mode" cards (same eight modes, copy and ``balanced`` preselected) and
   the target language (``output_language``).
4. **Permissions** - notifications, battery optimisation (Android), the iOS
   background limits note.
5. **Done** - try-chips: Paste text to translate · Import a book · Open Library.

Finishing writes what the desktop welcome's final "Get Started" writes for the
glossary page (``welcome_glossary_updates``) plus ``glossary_mode_dialog_shown``;
"Skip" writes only ``glossary_mode_dialog_shown`` (desktop "Skip Welcome Setup").
Writes are sparse (MobileConfigStore); the HeadlessOwner replays the desktop
startup handlers from these keys at job start.

``WelcomeFlow`` is the pure step/selection state (host-tested); ``WelcomeScreen``
renders it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.welcome_flow import (
    DEFAULT_GLOSSARY_MODE,
    GLOSSARY_MODE_CARDS,
    NOTE,
    OFF_GLOSSARY_MODES,
    STEP_TITLES,
    STEPS,
    TAGLINE,
    WelcomeFlow,
    welcome_glossary_updates,
)

#: Step 1 sign-ins, all equal (U11 item 1): (provider, button label).
SIGN_INS = (("authgpt", "Sign in with ChatGPT"), ("authcd", "Sign in with Claude"),
            ("authgem", "Sign in with Gemini"), ("authgrok", "Sign in with Grok"))
OTHER_SIGN_INS = SIGN_INS[1:]
#: Step 1 keyless routes (U13): (provider, button label). Their models are polled before the user picks one.
#: ocz/ (OpenCode Zen) runs through the desktop OpenCode CLI and cannot run on a phone.
KEYLESS = (("authnd", "NVIDIA Build free models (authnd/, no sign-in)"),)
#: output-token slider range (the mobile default is 16,384; the field takes any value)
TOKEN_SLIDER = (1024, 131072)
#: ModelSheet search that lists a provider's sign-in models (step 2, after signing in): the default
#: model stays ``authgpt/gpt-6-luna`` until another model is chosen.
PROVIDER_MODEL_QUERY = {"authgpt": "authgpt/", "authcd": "authcd/", "authgem": "authgem", "authgrok": "authgrok/"}

__all__ = [
    "GLOSSARY_MODE_CARDS",
    "OTHER_SIGN_INS",
    "SIGN_INS",
    "OFF_GLOSSARY_MODES",
    "PROVIDER_MODEL_QUERY",
    "STEPS",
    "WelcomeFlow",
    "WelcomeScreen",
    "welcome_glossary_updates",
]


class WelcomeScreen(Screen):
    title = "Welcome"

    def __init__(
        self,
        match: Any,
        *,
        flow: Optional[WelcomeFlow] = None,
        login_panel_factory: Optional[Callable[[Callable[[dict], Any]], ft.Control]] = None,
        languages: Sequence[str] = ("English",),
        on_finish: Optional[Callable[[dict], Any]] = None,  # config updates
        on_skip: Optional[Callable[[dict], Any]] = None,
        on_choose_model: Optional[Callable[[], Any]] = None,
        on_use_provider: Optional[Callable[[str], Any]] = None,  # ModelSheet search query
        on_api_key: Optional[Callable[[str], Any]] = None,
        on_request_notifications: Optional[Callable[[], Any]] = None,
        on_request_battery: Optional[Callable[[], Any]] = None,
        on_try: Optional[Callable[[str], Any]] = None,
        is_android: bool = False,
        is_ios: bool = False,
        oauth: Any = None,
        navigate: Optional[Callable[..., Any]] = None,
        copy_text: Optional[Callable[[str], Any]] = None,
        page: Any = None,
        catalog: Any = None,  # ModelCatalogService: the post-sign-in model check (U11 item 1)
        current_model: Optional[Callable[[], str]] = None,
        on_set_model: Optional[Callable[[str], Any]] = None,
        default_max_tokens: Optional[int] = None,
        default_chunk_size: str = "",
    ) -> None:
        super().__init__(match)
        self.flow = flow or WelcomeFlow()
        self.login_panel_factory = login_panel_factory
        self.languages = list(languages) or ["English"]
        self.on_finish = on_finish
        self.on_skip = on_skip
        self.on_choose_model = on_choose_model
        self.on_use_provider = on_use_provider
        self.on_api_key = on_api_key
        self.on_request_notifications = on_request_notifications
        self.on_request_battery = on_request_battery
        self.on_try = on_try
        self.is_android = is_android
        self.is_ios = is_ios
        self.oauth = oauth  # OAuthBridge; else taken from the step-1 LoginPanel
        self.navigate = navigate  # app.navigate_to(route_name)
        self.copy_text = copy_text
        self.page = page
        self.login_sheet: Any = None
        self.provider_status: dict = {}
        self.catalog = catalog
        self.current_model = current_model
        self.on_set_model = on_set_model
        self.default_max_tokens = default_max_tokens  # the saved budget, else the mobile default
        self.default_chunk_size = default_chunk_size
        self.model_notes: dict = {}  # provider -> the model check's line
        self.page_area = ft.Container(expand=True)
        self.dots = ft.Row([], alignment=ft.MainAxisAlignment.CENTER, spacing=6)
        self.back_button = ft.TextButton(content="Back", on_click=lambda e: self.back())
        self.next_button = ft.FilledButton(content="Next", on_click=lambda e: self.next())
        self.skip_button = ft.TextButton(content="Skip", on_click=lambda e: self.skip())
        self.mode_cards: dict = {}
        self.api_key_field = ft.TextField(label="API key", password=True, can_reveal_password=True, dense=True)

    # ---- pages ----------------------------------------------------------------------------

    def _ensure_oauth(self) -> None:
        if self.oauth is None and self.login_panel_factory is not None:
            try:
                self.oauth = getattr(self.login_panel_factory(self._signed_in), "oauth", None)
            except Exception:
                self.oauth = None

    def _page_sign_in(self) -> ft.Control:
        self._ensure_oauth()
        return ft.Column(
            [
                ft.Image(src=HALGAKOS_ASSET, width=96, height=96),
                ft.Text(STEP_TITLES["sign_in"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Text(TAGLINE, theme_style=ft.TextThemeStyle.BODY_MEDIUM, text_align=ft.TextAlign.CENTER),
                ft.Text("Sign in with an AI subscription you already have (no API key needed). Glossarion checks its "
                        "models and picks its most cost-efficient one; you can change it any time.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                        text_align=ft.TextAlign.CENTER),
                *[self._sign_in_row(provider, label) for provider, label in SIGN_INS],
                ft.Text("No account needed", theme_style=ft.TextThemeStyle.TITLE_SMALL),
                *[self._keyless_row(provider, label) for provider, label in KEYLESS],
                ft.TextButton(content="Use an API key or a local model", on_click=lambda e: self.go("providers")),
                ft.TextButton(content="Skip for now", on_click=lambda e: self.next()),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

    def _sign_in_row(self, provider: str, label: str) -> ft.Control:
        done = self.provider_status.get(provider)
        if done:
            who = done.get("email") or done.get("name") or ""
            name = label.replace("Sign in with ", "")
            row: list[ft.Control] = [ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY),
                                     ft.Text(f"{name} signed in" + (f" · {who}" if who else ""))]
            if self.on_use_provider is not None or self.on_choose_model is not None:
                row.append(ft.TextButton(content="Change model", key=f"welcome-use-{provider}",
                                         on_click=lambda e, p=provider: self.use_provider(p)))
            note = self.model_notes.get(provider)
            lines: list[ft.Control] = [ft.Row(row, wrap=True)]
            if note:
                lines.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                     key=f"welcome-model-{provider}"))
            return ft.Column(lines, spacing=2, tight=True, key=f"welcome-signed-{provider}")
        if self.oauth is None:
            return ft.Row([ft.Text(label), ReasonChip(reason="Sign-in unavailable in this build")], wrap=True)
        return ft.OutlinedButton(content=label, on_click=lambda e, p=provider: self.open_sign_in(p),
                                 key=f"welcome-signin-{provider}")

    def _keyless_row(self, provider: str, label: str) -> ft.Control:
        note = self.model_notes.get(provider)
        lines: list[ft.Control] = [ft.OutlinedButton(content=label, key=f"welcome-keyless-{provider}",
                                                     on_click=lambda e, p=provider: call_handler(self.use_keyless, p))]
        if note:
            lines.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 key=f"welcome-model-{provider}"))
        return ft.Column(lines, spacing=2, tight=True)

    async def use_keyless(self, provider: str) -> None:
        """A keyless route: poll its models first, then the ModelSheet on them (the user picks a model that
        exists right now)."""
        self.model_notes[provider] = "Checking the models you can use…"
        self.render()
        count = 0
        if self.catalog is not None:
            try:
                await self.catalog.refresh(provider, explicit=True)
            except Exception:
                pass
            models = tuple(getattr(getattr(self.catalog, "snapshot", None), "models", ()) or ())
            count = sum(1 for m in models if str(m).lower().startswith(provider + "/"))
        self.model_notes[provider] = (f"{count} models available · pick one" if count
                                      else "Couldn't list its models right now; you can still pick one")
        self.flow.signed_in = True  # a usable route: the next step follows
        self.render()
        self.use_provider(provider)

    def _run_budget(self) -> ft.Control:
        """U13: the output-token budget (slider + field; mobile default 16,384) and the chunk size (blank: auto)."""
        current = int(self.flow.max_output_tokens or self.default_max_tokens or 16384)
        low, high = TOKEN_SLIDER
        field = ft.TextField(label="Output token limit", value=str(current), dense=True, width=170,
                             keyboard_type=ft.KeyboardType.NUMBER, key="welcome-max-tokens")
        slider = ft.Slider(min=low, max=high, value=max(low, min(high, current)), expand=True,
                           key="welcome-max-tokens-slider")

        def from_slider(e: Any) -> None:
            value = int(round(float(e.control.value) / 1024.0)) * 1024 or low
            self.flow.max_output_tokens = value
            field.value = str(value)
            try:
                field.update()
            except Exception:
                pass

        def from_field(e: Any) -> None:
            text = str(e.control.value or "").strip()
            if text.isdigit() and int(text) > 0:
                self.flow.max_output_tokens = int(text)
                slider.value = max(low, min(high, int(text)))
                try:
                    slider.update()
                except Exception:
                    pass

        slider.on_change = from_slider
        field.on_change = from_field
        chunk = ft.TextField(label="Chunk size (tokens)", hint_text="Blank = auto", dense=True, width=170,
                             value=self.flow.chunk_size or self.default_chunk_size or "",
                             keyboard_type=ft.KeyboardType.NUMBER, key="welcome-chunk-size",
                             on_change=lambda e: setattr(self.flow, "chunk_size",
                                                         str(e.control.value or "").strip()))
        return ft.Column([
            ft.Text("Request size", theme_style=ft.TextThemeStyle.TITLE_SMALL),
            ft.Row([slider, field], spacing=8),
            chunk,
            ft.Text("Lower limits suit phones and free tiers; leave the chunk size blank to let Glossarion choose.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ], spacing=6, tight=True)

    def _page_providers(self) -> ft.Control:
        local: ft.Control
        if self.navigate is not None:
            local = ft.OutlinedButton(content="Set up Ollama / LM Studio on your network",
                                      on_click=lambda e: self._go("settings.endpoints"), key="welcome-local")
        else:
            local = ft.Row([ft.Text("Ollama / LM Studio on your network"),
                            ReasonChip(reason="Settings › Endpoints")], wrap=True)
        return ft.Column(
            [
                ft.Text(STEP_TITLES["providers"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Text("Use another model with your own API key, or a model on your own network.",
                        theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.FilledTonalButton(content="Choose another model", on_click=lambda e: self._choose_model()),
                self.api_key_field,
                ft.TextButton(content="Save API key", on_click=lambda e: self._save_key()),
                ft.Text("Local", theme_style=ft.TextThemeStyle.TITLE_SMALL),
                local,
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

    def _mode_card(self, card: tuple) -> ft.Control:
        value, emoji, title, subtitle, features, rec = card
        selected = value == self.flow.glossary_mode
        lines: list[ft.Control] = [
            ft.Text(f"{emoji} {title}", weight=ft.FontWeight.W_600),
            ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.PRIMARY),
            *[ft.Text(f, theme_style=ft.TextThemeStyle.BODY_SMALL) for f in features],
        ]
        if rec:
            lines.append(ft.Text(rec, theme_style=ft.TextThemeStyle.LABEL_SMALL))
        control = ft.Container(
            content=ft.Column(lines, spacing=2, tight=True),
            padding=10,
            border_radius=12,
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, v=value: self.select_mode(v),
            key=f"glossary-mode-{value}",
            col={"xs": 12, "sm": 6},
        )
        self.mode_cards[value] = control
        return control

    def _page_language_glossary(self) -> ft.Control:
        self.mode_cards = {}
        language = ft.Dropdown(
            label="Target language",
            value=self.flow.target_language,
            options=[ft.DropdownOption(key=lang, text=lang) for lang in self.languages],
            on_select=lambda e: setattr(self.flow, "target_language", e.control.value or self.flow.target_language),
        )
        return ft.Column(
            [
                ft.Text(STEP_TITLES["language_glossary"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Text("Select how glossary extraction runs when you translate", theme_style=ft.TextThemeStyle.BODY_SMALL),
                language,
                self._run_budget(),
                ft.ResponsiveRow([self._mode_card(card) for card in GLOSSARY_MODE_CARDS], spacing=8, run_spacing=8),
                ft.Text(NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

    def _page_permissions(self) -> ft.Control:
        rows: list[ft.Control] = [
            ft.Text(STEP_TITLES["permissions"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Text("Notifications show translation progress and let you stop a job from the notification.",
                    theme_style=ft.TextThemeStyle.BODY_MEDIUM),
            ft.FilledTonalButton(content="Allow notifications", on_click=lambda e: self._call(self.on_request_notifications)),
        ]
        if self.is_android:
            rows += [
                ft.Text("Long translations keep running with the screen off when battery optimisation is disabled for "
                        "Glossarion.", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.FilledTonalButton(content="Disable battery optimisation", on_click=lambda e: self._call(self.on_request_battery)),
            ]
        rows.append(
            ft.Text(
                "iPhone and iPad: iOS pauses translations about 30 seconds after you leave the app (iOS 26 can "
                "continue in the background). A paused job resumes from its saved progress.",
                theme_style=ft.TextThemeStyle.BODY_SMALL,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
        )
        return ft.Column(rows, spacing=10, scroll=ft.ScrollMode.AUTO)

    def _page_done(self) -> ft.Control:
        chips = [
            ft.Chip(label=ft.Text(label), on_click=lambda e, t=tid: self._try(t), key=f"try-{tid}")
            for tid, label in (("paste", "Paste text to translate"), ("import", "Import a book"), ("library", "Open Library"))
        ]
        return ft.Column(
            [
                ft.Image(src=HALGAKOS_ASSET, width=72, height=72),
                ft.Text(STEP_TITLES["done"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Row(chips, wrap=True, spacing=8, alignment=ft.MainAxisAlignment.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        )

    def _page(self) -> ft.Control:
        return {
            "sign_in": self._page_sign_in,
            "providers": self._page_providers,
            "language_glossary": self._page_language_glossary,
            "permissions": self._page_permissions,
            "done": self._page_done,
        }[self.flow.step_id]()

    def render(self) -> None:
        self.page_area.content = self._page()
        self.dots.controls = [
            ft.Container(width=8, height=8, border_radius=4,
                         bgcolor=ft.Colors.PRIMARY if i == self.flow.step else ft.Colors.OUTLINE_VARIANT)
            for i in range(len(STEPS))
        ]
        self.back_button.visible = self.flow.step > 0
        self.next_button.content = "Get started" if self.flow.is_last else "Next"
        self.skip_button.visible = not self.flow.is_last
        try:
            self.body.update()
        except Exception:
            pass

    def build_body(self) -> ft.Control:
        self.render()
        return ft.Container(
            padding=16,
            content=ft.Column(
                [self.page_area, self.dots, ft.Row([self.skip_button, ft.Container(expand=True), self.back_button,
                                                     self.next_button])],
                expand=True,
            ),
        )

    # ---- actions ----------------------------------------------------------------------------

    def go(self, step_id: str) -> None:
        self.flow.go(step_id)
        self.render()

    def next(self) -> None:
        if self.flow.is_last:
            self.finish()
            return
        self.flow.next()
        self.render()

    def back(self) -> None:
        self.flow.back()
        self.render()

    def select_mode(self, mode: str) -> None:
        self.flow.select_mode(mode)
        self.render()

    def finish(self) -> dict:
        updates = self.flow.finish_updates()
        if self.on_finish is not None:
            self.on_finish(updates)
        return updates

    def skip(self) -> dict:
        updates = self.flow.skip_updates()
        if self.on_skip is not None:
            self.on_skip(updates)
        return updates

    def _signed_in(self, status: dict) -> None:
        self.flow.mark_signed_in()
        self.render()

    def open_sign_in(self, provider: str) -> Any:
        """Step 2: the LoginSheet for Claude / Gemini / Grok (slot #0)."""
        if self.oauth is None:
            return None
        from glossarion_mobile.ui.screens.accounts import LoginSheet

        sheet = LoginSheet(self.oauth, provider=provider, account_id=0, copy_text=self.copy_text,
                           on_done=lambda status, p=provider: self._provider_signed_in(p, status))
        self.login_sheet = sheet
        page = self.page
        if page is None and self.body is not None:
            try:
                page = self.body.page  # raises while the body is not on a page
            except Exception:
                page = None
        if page is not None:
            sheet.show(page)
        return sheet

    def _provider_signed_in(self, provider: str, status: dict) -> None:
        self.provider_status[provider] = dict(status or {})
        self.flow.signed_in = True
        self.flow.skipped_sign_in = False
        self.model_notes[provider] = "Checking the models you can use…"
        self.render()
        call_handler(self.check_model, provider)

    async def check_model(self, provider: str) -> Optional[str]:
        """After a sign-in (U11 item 1): poll the provider's models, keep the current model when it is that
        provider's and still listed (the default GPT-6 Luna for ChatGPT), else switch to the provider's most
        cost-efficient model; nothing found -> the ModelSheet on its models. The model in use, or None."""
        from glossarion_mobile.services import model_catalog as mc

        name = mc.LOGIN_TITLES.get(provider, provider)
        models: tuple = ()
        if self.catalog is not None:
            try:
                await self.catalog.refresh(provider, explicit=True)
            except Exception:
                pass  # the cached catalog still decides
            models = tuple(getattr(getattr(self.catalog, "snapshot", None), "models", ()) or ())
        current = str(self.current_model() if self.current_model is not None else "") or mc.DEFAULT_AUTH_MODEL
        listed = {str(m).casefold() for m in models}
        if mc.login_route(current)[0] == provider and current.casefold() in listed:
            note = f"✓ Using {current}"
            chosen: Optional[str] = current
        else:
            chosen = mc.recommended_model(provider, models)
            if chosen and self.on_set_model is not None:
                call_handler(self.on_set_model, chosen)
                gone = (provider == "authgpt" and current.casefold() == mc.DEFAULT_AUTH_MODEL
                        and current.casefold() not in listed)
                note = (f"{current} is no longer offered · using {chosen}" if gone
                        else f"✓ Using {chosen} ({name}'s most cost-efficient model)")
            else:
                chosen = None
                note = f"Couldn't list {name}'s models: choose one"
        self.model_notes[provider] = note
        self.render()
        if chosen is None:
            self.use_provider(provider)
        return chosen

    def _go(self, route_name: str) -> None:
        if self.navigate is not None:
            self.navigate(route_name)

    def _choose_model(self) -> None:
        self._call(self.on_choose_model)

    def use_provider(self, provider: str) -> None:
        """After a step-2 sign-in: the ModelSheet searching that provider's models (the default
        model stays ``authgpt/gpt-6-luna``, which needs ChatGPT, until another one is chosen)."""
        if self.on_use_provider is not None:
            call_handler(self.on_use_provider, PROVIDER_MODEL_QUERY.get(provider, provider))
        else:
            self._choose_model()

    def _save_key(self) -> None:
        value = (self.api_key_field.value or "").strip()
        if value and self.on_api_key is not None:
            self.on_api_key(value)
            self.flow.api_key_set = True

    def _try(self, try_id: str) -> None:
        self.finish()
        if self.on_try is not None:
            self.on_try(try_id)

    @staticmethod
    def _call(handler: Optional[Callable[[], Any]]) -> None:
        call_handler(handler)  # sync or async (permission requests are coroutines)
