"""First-run Welcome (``/welcome``, UI_SPEC §4.17): 4 steps, re-runnable from About.

1. **Welcome · Sign in** (one page; the former step 2 "Other providers" merged in, owner 2026-10-16) -
   ChatGPT, Claude, Gemini, Grok, NanoGPT and DeepSeek two per row (U11 item 1: no provider is pushed;
   LoginSheet through the OAuthBridge, Grok shows its device code). After a sign-in the provider's models
   are polled (``ModelCatalogService.refresh``): ChatGPT keeps the default ``authgpt/gpt-6-luna`` while
   the provider still lists it, otherwise (and for the other providers) the model becomes that
   provider's most cost-efficient one (``model_catalog.recommended_model``: Sonnet, the newest Flash,
   the newest Grok); "Change model" opens the ModelSheet on that provider's models. Below them: the
   keyless routes; "Run on this phone (no account needed)": the on-device presets as one-tap buttons
   (size and a RAM hint for this phone from the engine's RAM gate, Light suggested, inline download
   progress; one tap downloads, loads and selects the model through the Model manager's
   ``OnDeviceFlow``); then your own API key (the desktop main key field, ``api_key``), "Choose another
   model" (ModelSheet) and a local Ollama / LM Studio host on your PC (Settings › Endpoints). Next
   moves on (without a sign-in or a model it counts as skipped).
2. **Target language and glossary mode** - the desktop first-run "Choose Your
   Glossary Mode" cards (same eight modes, copy and ``balanced`` preselected) and
   the target language (``output_language``).
3. **Permissions** - notifications, battery optimisation (Android), the iOS
   background limits note.
4. **Done** - try-chips: Paste text to translate · Import a book · Open Library.

Finishing writes what the desktop welcome's final "Get Started" writes for the
glossary page (``welcome_glossary_updates``) plus ``glossary_mode_dialog_shown``;
"Skip" writes only ``glossary_mode_dialog_shown`` (desktop "Skip Welcome Setup").
Writes are sparse (MobileConfigStore); the HeadlessOwner replays the desktop
startup handlers from these keys at job start.

``WelcomeFlow`` is the pure step/selection state (host-tested); ``WelcomeScreen``
renders it.
"""

from __future__ import annotations

import asyncio
import logging
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
from glossarion_mobile.ui.theme import number_keyboard

log = logging.getLogger(__name__)

#: Step 1 sign-ins, all equal (U11 item 1): (provider, button label).
SIGN_INS = (("authgpt", "Sign in with ChatGPT"), ("authcd", "Sign in with Claude"),
            ("authgem", "Sign in with Vertex (Pay as you go)"), ("authgrok", "Sign in with Grok"),
            ("authnan", "Sign in with NanoGPT"), ("authds", "Sign in with DeepSeek"))
#: sign-ins whose long label uses smaller type to fit its half row. authgem/ bills Google Cloud
#: Vertex AI per request; it is not a Gemini subscription, so it is not labelled "Gemini".
SMALL_LABELS = frozenset({"authgem"})
OTHER_SIGN_INS = SIGN_INS[1:]
#: Step 1 keyless routes (U13): (provider, button label). Their models are polled before the user picks one.
#: ocz/ (OpenCode Zen) runs through the desktop OpenCode CLI and cannot run on a phone.
KEYLESS = (("authchatgpt", "ChatGPT web chat"), ("authnd", "NVIDIA free models"))
#: the Halgakos logo beside the step-1 title (was 96 dp above it)
WELCOME_LOGO = 56
#: output-token slider range (the mobile default is 16,384; the field takes any value)
TOKEN_SLIDER = (1024, 131072)
#: batch-size slider range; the field takes any larger value
BATCH_SLIDER = (1, 50)
#: desktop defaults when nothing is saved (``delay`` 5.0 s, ``translation_temperature`` 0.3)
DEFAULT_DELAY = 5.0
DEFAULT_TEMPERATURE = 0.3
RATE_LIMIT_WARNING = ("Higher batch sizes send more requests at once and lower delays send them faster: respect your "
                      "provider's rate limits. authds/ (DeepSeek) and authchatgpt/ (ChatGPT) run on the sites' "
                      "own web chat; keep them at batch size 1 with a delay, or the site may limit or block you.")
#: ModelSheet search that lists a provider's sign-in models (step 2, after signing in): the default
#: model stays ``authgpt/gpt-6-luna`` until another model is chosen.
PROVIDER_MODEL_QUERY = {"authgpt": "authgpt/", "authcd": "authcd/", "authgem": "authgem", "authgrok": "authgrok/",
                        "authnan": "authnan/", "authds": "authds/", "authnd": "authnd/", "authchatgpt": "authchatgpt/"}

#: the on-device preset Welcome suggests (Qwen3.5 2B: quick drafts on most phones)
SUGGESTED_ONDEVICE = "Light"
#: at most this many on-device presets on page 1 (one row)
ONDEVICE_CELLS = 4
#: a cell's second line by ``OnDeviceFlow.row_state`` (short: a quarter of a 360 dp row)
ONDEVICE_LABELS = {"checking": "Checking…", "paused": "Resume", "busy": "Starting…", "error": "Retry",
                   "in_use": "✓ In use", "use": "Tap to use"}

__all__ = [
    "GLOSSARY_MODE_CARDS",
    "ONDEVICE_CELLS",
    "SUGGESTED_ONDEVICE",
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
        config_get: Optional[Callable[[str, Any], Any]] = None,  # saved delay / temperature / batch size
        local_engine: Any = None,  # the app's LocalEngineService (or a callable returning it): "Run on this phone"
        prefs: Any = None,  # the on-device presets (Prefs ``local_model_presets``)
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
        #: async () -> {"notifications": bool, "battery": bool}: the real grants, so step 4 shows ✓ instead of
        #: a button for a permission that is already given (the launch dialog, or an earlier visit)
        self.on_permission_status: Optional[Callable[[], Any]] = None
        self.permission_state: dict = {}
        self._permission_check: Any = None
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
        self.config_get = config_get
        self.model_notes: dict = {}  # provider -> the model check's line
        self.keyless_chosen: set = set()  # keyless routes whose recommended model is in use
        self.changing_provider: Optional[str] = None  # the row whose "Change model" opened the ModelSheet
        self.local_engine = local_engine
        self.prefs = prefs
        self.ondevice: Any = None  # OnDeviceFlow (local_models_panel), made on first render
        self.context_control: Any = None  # ContextControl of the Request size step (on-device models)
        self.ondevice_box = ft.Container(key="welcome-ondevice-box")
        self._ondevice_unsub: Optional[Callable[[], None]] = None
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

    @staticmethod
    def _section(text: str) -> ft.Control:
        return ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.PRIMARY)

    @staticmethod
    def _note(text: str, **kw: Any) -> ft.Control:
        return ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT, **kw)

    def _page_sign_in(self) -> ft.Control:
        """Step 1, one page (owner 2026-10-16: the former step 2 "Other providers" merged in): a compact
        header, the sign-ins two per row, the keyless routes, then your own API key / another model /
        a local host. Small type so it fits a 360-412 dp phone at 100 % text; it scrolls at large text."""
        self._ensure_oauth()
        return ft.Column(
            [
                ft.Row([ft.Image(src=HALGAKOS_ASSET, width=WELCOME_LOGO, height=WELCOME_LOGO),
                        ft.Column([ft.Text(STEP_TITLES["sign_in"], theme_style=ft.TextThemeStyle.TITLE_LARGE),
                                   self._note(TAGLINE)], spacing=2, tight=True, expand=True)],
                       spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, key="welcome-header"),
                self._section("Sign in with an AI subscription · no API key needed"),
                self._note("Glossarion checks its models and picks its most cost-efficient one; "
                           "you can change it any time."),
                ft.ResponsiveRow([self._sign_in_row(provider, label) for provider, label in SIGN_INS],
                                 spacing=8, run_spacing=4, key="welcome-sign-ins"),
                self._section("No account needed"),
                ft.ResponsiveRow([self._keyless_row(provider, label) for provider, label in KEYLESS],
                                 spacing=8, run_spacing=4, key="welcome-keyless"),
                self._section("Run on this phone (no account needed)"),
                self._ondevice_section(),
                self._section("Your own API key or a model on your PC"),
                ft.Row([ft.Container(content=self.api_key_field, expand=True),
                        ft.TextButton(content="Save", tooltip="Save API key", on_click=lambda e: self._save_key(),
                                      key="welcome-save-key")],
                       spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER, key="welcome-api-key"),
                ft.Row([ft.TextButton(content="Choose another model", on_click=lambda e: self._choose_model(),
                                      key="welcome-choose-model"),
                        self._local_control()], wrap=True, spacing=4, run_spacing=0),
            ],
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
        )

    # ---- "Run on this phone": the on-device presets (OnDeviceFlow of the Model manager) ------------------

    def ondevice_flow(self) -> Any:
        """The Model manager's ``OnDeviceFlow`` over the app's engine (None without one)."""
        if self.ondevice is None:
            service = self.local_engine() if callable(self.local_engine) else self.local_engine
            if service is None:
                return None
            from glossarion_mobile.ui.screens.local_models_panel import OnDeviceFlow

            self.ondevice = OnDeviceFlow(service=service, prefs=self.prefs, page=self.page,
                                         use_model=self._use_local_model, current_model=self.current_model,
                                         notify=self._ondevice_note, on_change=self.render_ondevice)
            self._ondevice_unsub = service.downloads.subscribe(lambda state: self.render_ondevice())
        return self.ondevice

    def _ondevice_section(self) -> ft.Control:
        self.ondevice_box.content = self._ondevice_content()
        return self.ondevice_box

    def _ondevice_content(self) -> ft.Control:
        flow = self.ondevice_flow()
        if flow is None:
            return ft.Row([ft.Text("On-device models", theme_style=ft.TextThemeStyle.BODY_SMALL),
                           ReasonChip(reason="Model manager › On-device")], wrap=True, key="welcome-ondevice")
        rows = flow.presets()[:ONDEVICE_CELLS]
        suggested = self.suggested_ondevice(rows)
        cells = [self._ondevice_cell(flow, index, preset, preset.get("name") == suggested)
                 for index, preset in enumerate(rows)]
        lines: list[ft.Control] = [ft.ResponsiveRow(cells, spacing=6, run_spacing=4, key="welcome-ondevice")]
        note = self.model_notes.get("local")
        if note:
            lines.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 key="welcome-model-local", max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        return ft.Column(lines, spacing=2, tight=True) if len(lines) > 1 else lines[0]

    def suggested_ondevice(self, rows: Sequence[dict]) -> Optional[str]:
        """Light, unless it is too big for this phone: then the biggest preset that fits."""
        flow = self.ondevice
        verdicts = {row.get("name"): (flow.ram_hint(row)[0] if flow is not None else "unknown") for row in rows}
        if any(row.get("name") == SUGGESTED_ONDEVICE for row in rows) and verdicts.get(SUGGESTED_ONDEVICE) != "refuse":
            return SUGGESTED_ONDEVICE
        fitting = [row for row in rows if verdicts.get(row.get("name")) == "ok"]
        return str(max(fitting, key=lambda r: int(r.get("size") or 0)).get("name")) if fitting else None

    @staticmethod
    def ondevice_key(preset: dict) -> str:
        slug = "".join(ch if ch.isalnum() else "-" for ch in str(preset.get("name") or "").lower()).strip("-")
        return f"welcome-ondevice-{slug or 'model'}"

    def _ondevice_cell(self, flow: Any, index: int, preset: dict, suggested: bool) -> ft.Control:
        """One preset: its name (⭐ suggested) over its size and RAM mark, or its download progress; one
        tap runs ``OnDeviceFlow.get_preset`` (download incl. image support, load, use)."""
        from glossarion_mobile.services.local_llm import downloads as dl
        from glossarion_mobile.services.local_llm import presets as lp

        stem = flow.stem_of_preset(preset)
        state, status = flow.row_state(preset, stem)
        verdict, mark, words = flow.ram_hint(preset)
        download = flow.active_download(stem)
        if state == "downloading" and download is not None:
            # tapping a downloading cell offers to cancel it (use_ondevice)
            detail = (f"{download.percent}% · ✕" if download.phase == "downloading"
                      else ONDEVICE_LABELS["checking"])
        elif state in ONDEVICE_LABELS:
            detail = ONDEVICE_LABELS[state]
        else:
            detail = " ".join(x for x in (dl.format_bytes(lp.download_size(preset)) if lp.download_size(preset)
                                          else "", mark) if x)
        name = f"{preset.get('name')} ⭐" if suggested else str(preset.get("name"))
        color = {"refuse": ft.Colors.ERROR, "confirm": ft.Colors.TERTIARY}.get(verdict) if state == "download" else None
        tooltip = " · ".join(x for x in (str(preset.get("note") or ""), words if state == "download" else status) if x)
        lines: list[ft.Control] = [
            ft.Text(name, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(detail, theme_style=ft.TextThemeStyle.LABEL_SMALL, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                    color=color or ft.Colors.ON_SURFACE_VARIANT),
        ]
        if state in ("downloading", "paused") and download is not None:
            lines.append(ft.ProgressBar(value=download.fraction, bar_height=3, border_radius=2))
        return ft.Container(
            content=ft.Column(lines, spacing=0, tight=True),
            padding=ft.Padding.symmetric(horizontal=6, vertical=5),
            border_radius=10,
            border=ft.Border.all(2 if suggested else 1, ft.Colors.PRIMARY if suggested else ft.Colors.OUTLINE_VARIANT),
            bgcolor=ft.Colors.SECONDARY_CONTAINER if state == "in_use" else None,
            tooltip=tooltip or None,
            on_click=lambda e, i=index: call_handler(self.use_ondevice, i),
            key=self.ondevice_key(preset),
            col={"xs": 3},
        )

    async def use_ondevice(self, index: int) -> Optional[str]:
        """One tap: download (with its image support), load and select the preset's model; the Welcome
        can then go on (it counts like a sign-in)."""
        flow = self.ondevice_flow()
        if flow is None:
            return None
        self.model_notes.pop("local", None)
        if flow.page is None:  # the big-download / tight-RAM questions need the page (as open_sign_in finds it)
            try:
                flow.page = self.page or (self.body.page if self.body is not None else None)
            except Exception:
                flow.page = None
        rows = flow.presets()
        if 0 <= index < len(rows):
            stem = flow.stem_of_preset(rows[index])
            state, _status = flow.row_state(rows[index], stem)
            download = flow.active_download(stem)
            if state == "downloading" and download is not None:
                # a second tap on a downloading cell is the undo of a mis-tap
                name = str(rows[index].get("name") or "this model")
                if await flow.ask(f"Cancel the {name} download?",
                                  f"{download.percent}% is downloaded. Cancelling deletes the partial file.",
                                  "Cancel download", cancel_label="Keep downloading"):
                    flow.cancel(download.stem)
                    self.render_ondevice()
                return None
        model = await flow.get_preset(index)
        if model:
            self.flow.signed_in = True
            self.flow.skipped_sign_in = False
            self.model_chosen(model)
            self.model_notes["local"] = f"✓ Using {model} · runs on this phone"
            self.render()
        return model

    def _use_local_model(self, model: str) -> None:
        if self.on_set_model is not None:
            call_handler(self.on_set_model, model)

    def _ondevice_note(self, message: str) -> None:
        if not str(message).startswith("Model: "):  # the flow's own "Model: …" line; use_ondevice says it
            self.model_notes["local"] = str(message)

    def render_ondevice(self) -> None:
        """Re-render only the on-device row (download progress arrives several times a second)."""
        if self.flow.step_id != "sign_in" or self.ondevice_box.content is None:
            return
        self.ondevice_box.content = self._ondevice_content()
        try:
            self.ondevice_box.update()
        except Exception:
            pass

    def dispose(self) -> None:
        if self._ondevice_unsub is not None:
            self._ondevice_unsub()
            self._ondevice_unsub = None
        super().dispose()

    def _local_control(self) -> ft.Control:
        if self.navigate is not None:
            return ft.TextButton(content="Ollama / LM Studio", tooltip="Set up Ollama / LM Studio on your PC or network",
                                 on_click=lambda e: self._go("settings.endpoints"), key="welcome-local")
        return ft.Row([ft.Text("Ollama / LM Studio", theme_style=ft.TextThemeStyle.BODY_SMALL),
                       ReasonChip(reason="Settings › Endpoints")], wrap=True, key="welcome-local")

    def _sign_in_row(self, provider: str, label: str) -> ft.Control:
        """One sign-in cell: half the row (two per row on a phone); signed in, the whole row."""
        control = self._sign_in_control(provider, label)
        return ft.Container(content=control, col={"xs": 12 if provider in self.provider_status else 6})

    def _sign_in_control(self, provider: str, label: str) -> ft.Control:
        short = label.replace("Sign in with ", "")
        done = self.provider_status.get(provider) if provider in self.provider_status else None
        if provider in self.provider_status:
            who = done.get("email") or done.get("name") or ""
            row: list[ft.Control] = [ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY),
                                     ft.Text(f"{short} signed in" + (f" · {who}" if who else ""))]
            if self.on_use_provider is not None or self.on_choose_model is not None:
                row.append(ft.TextButton(content="Change model", key=f"welcome-use-{provider}",
                                         on_click=lambda e, p=provider: self.use_provider(p)))
            note = self.model_notes.get(provider)
            lines: list[ft.Control] = [ft.Row(row, wrap=True)]
            if note:
                lines.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                     key=f"welcome-model-{provider}"))
            return ft.Column(lines, spacing=2, tight=True, key=f"welcome-signed-{provider}")
        if provider == "authds":  # DeepSeek's own site in the in-app browser (not an OAuth sign-in)
            from glossarion_mobile.services import webview_bridge

            available, reason = webview_bridge.availability()
            if not available:
                return ft.Row([ft.Text(short, theme_style=ft.TextThemeStyle.BODY_SMALL),
                               ReasonChip(reason="Needs the in-app browser", detail=reason)],
                              wrap=True, key="welcome-signin-authds")
            return self._cell_button(short, label, lambda e: self.open_deepseek_sign_in(), "welcome-signin-authds")
        if self.oauth is None:
            return ft.Row([ft.Text(short, theme_style=ft.TextThemeStyle.BODY_SMALL),
                           ReasonChip(reason="Sign-in unavailable in this build")], wrap=True)
        return self._cell_button(short, label, lambda e, p=provider: self.open_sign_in(p),
                                 f"welcome-signin-{provider}", small=provider in SMALL_LABELS)

    @staticmethod
    def _cell_button(text: str, tooltip: str, on_click: Callable[[Any], Any], key: str,
                     small: bool = False) -> ft.Control:
        """A half-row button: the provider's name only (the section title says what it does).
        ``small`` uses smaller type for a long name."""
        style = ft.TextThemeStyle.LABEL_SMALL if small else None
        return ft.Row([ft.OutlinedButton(content=ft.Text(text, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                                                         theme_style=style),
                                         tooltip=tooltip, on_click=on_click, key=key, expand=True,
                                         style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=8)))])

    def _keyless_row(self, provider: str, label: str) -> ft.Control:
        note = self.model_notes.get(provider)
        lines: list[ft.Control] = [self._cell_button(label, label, lambda e, p=provider: call_handler(
            self.use_keyless, p), f"welcome-keyless-{provider}")]
        if provider in self.keyless_chosen:
            lines.append(ft.TextButton(content="Change model", key=f"welcome-use-{provider}",
                                       on_click=lambda e, p=provider: self.use_provider(p)))
        if note:
            lines.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 key=f"welcome-model-{provider}"))
        return ft.Container(content=ft.Column(lines, spacing=2, tight=True), col={"xs": 6})

    async def use_keyless(self, provider: str) -> None:
        """A keyless route: poll its models first, then use its recommended model (authchatgpt/: auto; authnd/: the highest
        Nemotron) and say which; without one, the ModelSheet on its models (a model that exists right now)."""
        from glossarion_mobile.services import model_catalog as mc

        self.model_notes[provider] = "Checking the models you can use…"
        self.render()
        count = 0
        models: tuple = ()
        if self.catalog is not None:
            try:
                await self.catalog.refresh(provider, explicit=True)
            except Exception:
                pass
            models = tuple(getattr(getattr(self.catalog, "snapshot", None), "models", ()) or ())
            count = sum(1 for m in models if str(m).lower().startswith(provider + "/"))
        chosen = mc.recommended_model(provider, models) if count else None
        self.flow.signed_in = True  # a usable route: the next step follows
        if chosen and self.on_set_model is not None:
            call_handler(self.on_set_model, chosen)
            self.keyless_chosen.add(provider)
            self.model_notes[provider] = f"✓ Using {chosen}"
            self.render()
            return
        self.model_notes[provider] = (f"{count} models available · pick one" if count
                                      else "Couldn't list its models right now; you can still pick one")
        self.render()
        self.use_provider(provider)

    def local_context(self) -> Optional[Any]:
        """The on-device flow when the chosen model is an on-device ``local/`` one (its Context applies)."""
        model = str(self.current_model() if self.current_model is not None else "") or ""
        if not model.lower().startswith("local/"):
            return None
        return self.ondevice_flow()

    def shown_max_output_tokens(self) -> int:
        """The output-token limit Request size shows: half the context for an on-device model, else the
        value set here, else the saved one, else 16,384."""
        ondevice = self.local_context()
        if ondevice is not None:
            return max(256, ondevice.current_context() // 2)
        return int(self.flow.max_output_tokens or self.default_max_tokens or 16384)

    def _run_budget(self) -> ft.Control:
        """U13: the output-token budget (slider + field; mobile default 16,384) and the chunk size (blank: auto).

        With an on-device model: its Context (``ContextControl``, saved at once like Model manager's) above
        the budget, which is then half the context and read-only (jobs on ``local/`` use that; the saved
        budget stays for cloud models)."""
        ondevice = self.local_context()
        current = self.shown_max_output_tokens()
        low, high = TOKEN_SLIDER
        field = ft.TextField(label="Output token limit" + (" (half the context)" if ondevice is not None else ""),
                             value=str(current), dense=True, width=170, disabled=ondevice is not None,
                             keyboard_type=number_keyboard(), key="welcome-max-tokens")
        slider = ft.Slider(min=low, max=high, value=max(low, min(high, current)), expand=True,
                           disabled=ondevice is not None, key="welcome-max-tokens-slider")
        context_rows: list[ft.Control] = []
        if ondevice is not None:
            from glossarion_mobile.ui.screens.local_models_panel import ContextControl

            def on_context(tokens: int) -> None:
                ondevice.service.save_context(tokens)
                half = max(256, tokens // 2)
                field.value = str(half)
                slider.value = max(low, min(high, half))
                for control in (field, slider):
                    try:
                        control.update()
                    except Exception:
                        pass

            self.context_control = ContextControl(ondevice.current_context(), on_context, key="welcome-ctx",
                                                  label="Context (tokens)")
            context_rows = [self.context_control.control,
                            ft.Text("On-device model: the context holds the prompt and the answer; half of it is the "
                                    "output limit, and requests go one at a time.",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)]

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
                             keyboard_type=number_keyboard(), key="welcome-chunk-size",
                             on_change=lambda e: setattr(self.flow, "chunk_size",
                                                         str(e.control.value or "").strip()))
        return ft.Column([
            ft.Text("Request size", theme_style=ft.TextThemeStyle.TITLE_SMALL),
            *context_rows,
            ft.Row([slider, field], spacing=8),
            chunk,
            ft.Text("Lower limits suit phones and free tiers; leave the chunk size blank to let Glossarion choose.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            self._request_pacing(),
            self._batch_size(),
            ft.Text(RATE_LIMIT_WARNING, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR,
                    key="welcome-rate-limit-warning"),
        ], spacing=6, tight=True)

    def _saved(self, key: str, default: Any) -> Any:
        if self.config_get is None:
            return default
        try:
            value = self.config_get(key, None)
        except Exception:
            return default
        return default if value in (None, "") else value

    def default_batch_size(self) -> int:
        """The batch size shown when the user has not set one: the saved one, else the current model's
        default (1 for authds/, authchatgpt/ and on-device local/, else 5; ``setting_writes.default_batch_size``)."""
        from glossarion_mobile.state.setting_writes import default_batch_size
        from glossarion_mobile.ui.chat.send_state import DEFAULT_MODEL

        saved = self._saved("batch_size", None)
        try:
            if saved is not None and int(saved) > 0:
                return int(saved)
        except (TypeError, ValueError):
            pass
        model = str(self.current_model() if self.current_model is not None else "") or DEFAULT_MODEL
        return default_batch_size(model)

    def _request_pacing(self) -> ft.Control:
        """The API call delay, the temperature and Disable temperature (one row, above the batch size)."""
        def number(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        delay = self.flow.delay if self.flow.delay is not None else number(self._saved("delay", DEFAULT_DELAY),
                                                                           DEFAULT_DELAY)
        temperature = (self.flow.temperature if self.flow.temperature is not None
                       else number(self._saved("translation_temperature", DEFAULT_TEMPERATURE), DEFAULT_TEMPERATURE))
        disabled = (self.flow.disable_temperature if self.flow.disable_temperature is not None
                    else bool(self._saved("disable_temperature", False)))
        delay_field = ft.TextField(label="API call delay (s)", value=f"{delay:g}", dense=True, width=150,
                                   keyboard_type=number_keyboard(), key="welcome-delay")
        temperature_field = ft.TextField(label="Temperature", value=f"{temperature:g}", dense=True, width=120,
                                         keyboard_type=number_keyboard(), disabled=disabled,
                                         key="welcome-temperature")
        disable = ft.Checkbox(label="Disable temperature", value=disabled, key="welcome-disable-temperature")

        def on_delay(e: Any) -> None:
            try:
                value = float(str(e.control.value or "").strip())
            except ValueError:
                return
            if value >= 0:
                self.flow.delay = value

        def on_temperature(e: Any) -> None:
            try:
                value = float(str(e.control.value or "").strip())
            except ValueError:
                return
            if 0 <= value <= 2:
                self.flow.temperature = value

        def on_disable(e: Any) -> None:
            self.flow.disable_temperature = bool(e.control.value)
            temperature_field.disabled = self.flow.disable_temperature
            try:
                temperature_field.update()
            except Exception:
                pass

        delay_field.on_change = on_delay
        temperature_field.on_change = on_temperature
        disable.on_change = on_disable
        return ft.Row([delay_field, temperature_field, disable], spacing=8, wrap=True,
                      vertical_alignment=ft.CrossAxisAlignment.CENTER, key="welcome-pacing")

    def _batch_size(self) -> ft.Control:
        """Batch size: a 1-50 slider and a field that takes any larger value."""
        low, high = BATCH_SLIDER
        current = int(self.flow.batch_size or self.default_batch_size())
        field = ft.TextField(label="Batch size", value=str(current), dense=True, width=120,
                             keyboard_type=number_keyboard(), key="welcome-batch-size")
        slider = ft.Slider(min=low, max=high, divisions=high - low, value=max(low, min(high, current)), expand=True,
                           label="{value}", key="welcome-batch-size-slider")

        def from_slider(e: Any) -> None:
            value = max(low, int(round(float(e.control.value))))
            self.flow.batch_size = value
            field.value = str(value)
            try:
                field.update()
            except Exception:
                pass

        def from_field(e: Any) -> None:
            text = str(e.control.value or "").strip()
            if text.isdigit() and int(text) > 0:
                self.flow.batch_size = int(text)  # may go beyond the slider's 50
                slider.value = max(low, min(high, int(text)))
                try:
                    slider.update()
                except Exception:
                    pass

        slider.on_change = from_slider
        field.on_change = from_field
        return ft.Row([slider, field], spacing=8, key="welcome-batch")

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
            on_click=lambda e, v=value: self.select_mode(v),
            key=f"glossary-mode-{value}",
            col={"xs": 12, "sm": 6},
        )
        self._style_mode_card(control, selected)
        self.mode_cards[value] = control
        return control

    @staticmethod
    def _style_mode_card(control: ft.Container, selected: bool) -> None:
        control.bgcolor = ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_HIGHEST
        control.border = ft.Border.all(2, ft.Colors.PRIMARY) if selected else None

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
            self._permission_control("notifications", "Notifications allowed", "Allow notifications",
                                     self.on_request_notifications),
        ]
        if self.is_android:
            rows += [
                ft.Text("Long translations keep running with the screen off when battery optimisation is disabled for "
                        "Glossarion.", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                self._permission_control("battery", "Battery optimisation is off for Glossarion",
                                         "Disable battery optimisation", self.on_request_battery),
            ]
        rows.append(
            ft.Text(
                "iPhone and iPad: iOS pauses translations about 30 seconds after you leave the app (iOS 26 can "
                "continue in the background). A paused job resumes from its saved progress.",
                theme_style=ft.TextThemeStyle.BODY_SMALL,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
        )
        if self.on_permission_status is not None and self._permission_check is None:
            self._permission_check = call_handler(self.refresh_permissions)
        return ft.Column(rows, spacing=10, scroll=ft.ScrollMode.AUTO)

    def _permission_control(self, key: str, granted_label: str, button_label: str,
                            handler: Optional[Callable[[], Any]]) -> ft.Control:
        if self.permission_state.get(key):
            return ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY), ft.Text(granted_label)],
                          key=f"welcome-granted-{key}")
        return ft.FilledTonalButton(content=button_label, key=f"welcome-ask-{key}",
                                    on_click=lambda e: call_handler(self._ask_permission, handler))

    async def _ask_permission(self, handler: Optional[Callable[[], Any]]) -> None:
        if handler is not None:
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("permission request failed")
        await self.refresh_permissions()

    async def refresh_permissions(self) -> None:
        """Read the real grants; re-render step 4 only when they changed."""
        try:
            state = dict(await self.on_permission_status() or {}) if self.on_permission_status else {}
        except Exception:
            log.exception("permission status failed")
            state = {}
        finally:
            self._permission_check = None
        if state and state != self.permission_state:
            self.permission_state = state
            self._permission_check = False  # checked: no re-check on this render
            self.render()
            self._permission_check = None

    def _page_done(self) -> ft.Control:
        chips = [
            ft.Chip(label=ft.Text(label), on_click=lambda e, t=tid: self._try(t), key=f"try-{tid}")
            for tid, label in (("paste", "Paste text to translate"), ("import", "Import a book"), ("library", "Open Library"))
        ]
        return ft.Container(
            content=ft.Column(
                [
                    ft.Image(src=HALGAKOS_ASSET, width=72, height=72),
                    ft.Text(STEP_TITLES["done"], theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
                    ft.Row(chips, wrap=True, spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=12,
                tight=True,
            ),
            alignment=ft.Alignment.CENTER,  # the whole width and height of the page area (was left-aligned)
            expand=True,
        )

    def _page(self) -> ft.Control:
        return {
            "sign_in": self._page_sign_in,
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
        """Highlight the chosen card in place: rebuilding the page (``render``) scrolled it back to the top
        on every tap (owner 2026-10-18)."""
        self.flow.select_mode(mode)
        if self.flow.step_id != "language_glossary" or not self.mode_cards:
            self.render()
            return
        for value, card in self.mode_cards.items():
            self._style_mode_card(card, value == self.flow.glossary_mode)
            try:
                card.update()
            except Exception:  # not on a page (tests)
                pass

    def finish(self) -> dict:
        # Save what Request size showed even when it was left untouched (owner 2026-10-18: an on-device
        # model shows batch size 1 and half the context, yet Run options kept 5 and 128,000).
        if self.flow.batch_size is None:
            self.flow.batch_size = self.default_batch_size()
        if self.flow.max_output_tokens is None or self.local_context() is not None:
            self.flow.max_output_tokens = self.shown_max_output_tokens()
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

    def open_deepseek_sign_in(self) -> Any:
        """DeepSeek: the same in-app browser sheet as Accounts; it closes by itself once signed in."""
        from glossarion_mobile.ui.screens.accounts import show_deepseek_sign_in

        page = self.page
        if page is None and self.body is not None:
            try:
                page = self.body.page
            except Exception:
                page = None
        if page is None:
            return None
        call_handler(self._deepseek_sign_in, page)
        return None

    async def _deepseek_sign_in(self, page: Any) -> None:
        """Already signed in to DeepSeek (Accounts, an earlier visit): use it at once; else the sign-in sheet,
        which reports the sign-in when it closes by itself, or when it is closed after signing in."""
        from glossarion_mobile.ui.screens.accounts import deepseek_session_present, show_deepseek_sign_in

        self.model_notes["authds"] = "Checking your DeepSeek sign-in…"
        self.render()
        if await deepseek_session_present():
            self._provider_signed_in("authds", {})
            return
        self.model_notes.pop("authds", None)
        self.render()
        show_deepseek_sign_in(page, on_signed_in=lambda: self._provider_signed_in("authds", {}))

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
        from glossarion_mobile.ui.chat.send_state import DEFAULT_MODEL

        current = str(self.current_model() if self.current_model is not None else "") or DEFAULT_MODEL
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
                count = sum(1 for m in models if mc.login_route(str(m))[0] == provider)
                note = (f"{count} {name} models available: choose one" if count  # no cost rule for it (NanoGPT)
                        else f"Couldn't list {name}'s models: choose one")
        self.model_notes[provider] = note
        self.render()
        if chosen is None:
            self.use_provider(provider)
        return chosen

    def _go(self, route_name: str) -> None:
        if self.navigate is not None:
            self.navigate(route_name)

    def _choose_model(self) -> None:
        self.changing_provider = None  # "Choose another model": the pick's own provider row
        self._call(self.on_choose_model)

    def use_provider(self, provider: str) -> None:
        """After a step-2 sign-in: the ModelSheet searching that provider's models (the default
        model stays ``authgpt/gpt-6-luna``, which needs ChatGPT, until another one is chosen).
        The pick sets the global model and comes back through :meth:`model_chosen`."""
        if self.on_use_provider is not None:
            call_handler(self.on_use_provider, PROVIDER_MODEL_QUERY.get(provider, provider))
        else:
            self._choose_model()
        self.changing_provider = provider

    def model_chosen(self, model: str) -> None:
        """A model picked in the ModelSheet (the global model now): the row that opened it says so; the
        other rows' "Using …" lines no longer hold (one model is in use)."""
        from glossarion_mobile.services import model_catalog as mc

        model = str(model or "")
        provider = self.changing_provider or mc.login_route(model)[0] or model.split("/", 1)[0]
        self.changing_provider = None
        for other in [p for p, note in self.model_notes.items() if p != provider and str(note).startswith("✓")]:
            self.model_notes.pop(other, None)
        self.model_notes[provider] = f"✓ Using {model}"
        if provider in dict(KEYLESS):
            self.keyless_chosen.add(provider)  # its row keeps "Change model"
        self.render()

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
