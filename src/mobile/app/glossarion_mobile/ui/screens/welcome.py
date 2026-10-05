"""First-run Welcome (``/welcome``, UI_SPEC §4.17): 5 steps, re-runnable from About.

1. **Welcome · Sign in with ChatGPT** - the default model is ``authgpt/gpt-6-luna``
   (desktop default), so the primary action signs in (OAuthBridge LoginPanel);
   "Use an API key or another provider" goes to step 2; "Skip for now" moves on
   (Send then stays blocked with the "Sign in with ChatGPT" fix action).
2. **Other providers** - choose another model (ModelSheet) or paste an API key
   (the desktop main key field, ``api_key``); other sign-ins arrive in U4.
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

__all__ = [
    "GLOSSARY_MODE_CARDS",
    "OFF_GLOSSARY_MODES",
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
        on_api_key: Optional[Callable[[str], Any]] = None,
        on_request_notifications: Optional[Callable[[], Any]] = None,
        on_request_battery: Optional[Callable[[], Any]] = None,
        on_try: Optional[Callable[[str], Any]] = None,
        is_android: bool = False,
        is_ios: bool = False,
    ) -> None:
        super().__init__(match)
        self.flow = flow or WelcomeFlow()
        self.login_panel_factory = login_panel_factory
        self.languages = list(languages) or ["English"]
        self.on_finish = on_finish
        self.on_skip = on_skip
        self.on_choose_model = on_choose_model
        self.on_api_key = on_api_key
        self.on_request_notifications = on_request_notifications
        self.on_request_battery = on_request_battery
        self.on_try = on_try
        self.is_android = is_android
        self.is_ios = is_ios
        self.page_area = ft.Container(expand=True)
        self.dots = ft.Row([], alignment=ft.MainAxisAlignment.CENTER, spacing=6)
        self.back_button = ft.TextButton(content="Back", on_click=lambda e: self.back())
        self.next_button = ft.FilledButton(content="Next", on_click=lambda e: self.next())
        self.skip_button = ft.TextButton(content="Skip", on_click=lambda e: self.skip())
        self.mode_cards: dict = {}
        self.api_key_field = ft.TextField(label="API key", password=True, can_reveal_password=True, dense=True)

    # ---- pages ----------------------------------------------------------------------------

    def _page_sign_in(self) -> ft.Control:
        login: ft.Control
        if self.login_panel_factory is not None:
            login = self.login_panel_factory(self._signed_in)
        else:
            login = ReasonChip(reason="Sign-in is unavailable in this build")
        return ft.Column(
            [
                ft.Image(src=HALGAKOS_ASSET, width=96, height=96),
                ft.Text(STEP_TITLES["sign_in"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Text(TAGLINE, theme_style=ft.TextThemeStyle.BODY_MEDIUM, text_align=ft.TextAlign.CENTER),
                ft.Text("The default model is GPT-6 Luna (authgpt/gpt-6-luna). Sign in with your ChatGPT account to use it.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                        text_align=ft.TextAlign.CENTER),
                login,
                ft.TextButton(content="Use an API key or another provider", on_click=lambda e: self.go("providers")),
                ft.TextButton(content="Skip for now", on_click=lambda e: self.next()),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

    def _page_providers(self) -> ft.Control:
        return ft.Column(
            [
                ft.Text(STEP_TITLES["providers"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                ft.Text("Use another model with your own API key, or sign in with another provider.",
                        theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.FilledTonalButton(content="Choose another model", on_click=lambda e: self._choose_model()),
                self.api_key_field,
                ft.TextButton(content="Save API key", on_click=lambda e: self._save_key()),
                ft.Row([ft.Text("Claude · Gemini · Grok sign-in"), ReasonChip(reason="Arrives in U4")], wrap=True),
                ft.Row([ft.Text("Ollama / LM Studio on your network"), ReasonChip(reason="Arrives in U4")], wrap=True),
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

    def _choose_model(self) -> None:
        self._call(self.on_choose_model)

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
        if handler is not None:
            handler()
