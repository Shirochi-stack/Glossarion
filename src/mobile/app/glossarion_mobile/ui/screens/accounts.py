"""Accounts (``/settings/accounts``, UI_SPEC §4.13) and the LoginSheet (§4.13, §2.4).

U3 ships the ChatGPT row (sign in / status / sign out) because the default model is
``authgpt/gpt-6-luna``. Gemini, Claude and Grok sign-ins (and ChatGPT account slots)
arrive in U4: they are listed, disabled, with a ReasonChip. The "Unavailable on
mobile" section always lists the excluded routes, each disabled with its reason.

``LoginPanel`` is the LoginSheet body shared by this screen, Welcome step 1 and the
blocked Send button's "Sign in with ChatGPT" fix action: steps Opening browser ->
Waiting for sign-in -> Exchanging token -> Done, with **Reopen browser**, **Paste
redirect URL / code** and **Cancel** (``services.oauth.OAuthBridge``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.oauth import PROVIDERS, OAuthBridge, SignInState
from glossarion_mobile.ui.components.reason_chip import NOT_ON_MOBILE, ReasonChip
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["AccountsScreen", "LoginPanel", "LoginSheet", "UNAVAILABLE_ACCOUNTS"]

log = logging.getLogger("glossarion.accounts")

#: UI_SPEC §4.13 "Unavailable on mobile" (label, reason).
UNAVAILABLE_ACCOUNTS = (
    ("Antigravity", "Needs the desktop npm/bun proxy"),
    ("OCAGY", "Needs the desktop npm/bun proxy"),
    ("OpenCode Zen (ocz/)", "Needs the desktop npm/bun proxy"),
    ("Z.AI login", "Desktop-only access modes"),
    ("Arena", "Needs a desktop browser session"),
    ("Opera Aria", "Needs the desktop route"),
    ("Tor", "No bundled Tor client on mobile"),
    ("Managed Ollama (ollamapull/)", "Needs a desktop Ollama binary"),
    ("Claude Code CLI import", "Reads the desktop CLI credential store"),
    ("Grok CLI import", "Reads the desktop CLI credential store"),
)

PASTE_HINT = "http://localhost:1455/auth/callback?code=…&state=…"


class LoginPanel(ft.Column):
    """Sign-in progress + paste fallback for one provider (ChatGPT in U3)."""

    def __init__(
        self,
        oauth: OAuthBridge,
        *,
        on_done: Optional[Callable[[dict], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
        autostart: bool = False,
    ) -> None:
        super().__init__(spacing=10, tight=True)
        self.oauth = oauth
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self._task: Any = None
        self.step_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM)
        self.ring = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False,
                                  selectable=True)
        self.start_button = ft.FilledButton(content="Sign in with ChatGPT", on_click=self._start)
        self.reopen_button = ft.TextButton(content="Reopen browser", on_click=lambda e: self.oauth.reopen_browser())
        self.paste_field = ft.TextField(hint_text=PASTE_HINT, dense=True, visible=False, multiline=False)
        self.paste_button = ft.TextButton(content="Paste redirect URL / code", on_click=self._toggle_paste)
        self.finish_button = ft.FilledTonalButton(content="Finish sign-in", visible=False, on_click=self._finish_paste)
        self.cancel_button = ft.TextButton(content="Cancel", on_click=self._cancel)
        self.controls = [
            ft.Row([self.ring, self.step_text], spacing=8),
            self.error_text,
            ft.Row([self.start_button, self.reopen_button], wrap=True, spacing=8),
            self.paste_button,
            self.paste_field,
            ft.Row([self.finish_button, self.cancel_button], wrap=True, spacing=8),
        ]
        self._unsub = oauth.subscribe(self._on_state)
        self.apply(oauth.state)
        self._autostart = autostart

    def did_mount(self) -> None:
        super().did_mount()
        if self._autostart and not self.oauth.state.busy:
            self._autostart = False
            self._task = asyncio.ensure_future(self.run())

    def will_unmount(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        super().will_unmount()

    # ---- state ----------------------------------------------------------------------------

    def apply(self, state: SignInState) -> None:
        busy = state.busy
        self.ring.visible = busy
        if state.step == "done":
            who = f" as {state.email}" if state.email else ""
            self.step_text.value = f"Done · signed in{who}"
        else:
            self.step_text.value = state.label
        self.error_text.visible = state.step == "error"
        self.error_text.value = state.message if state.step == "error" else ""
        self.start_button.visible = not busy and state.step != "done"
        self.start_button.content = "Try again" if state.step in ("error", "cancelled") else "Sign in with ChatGPT"
        self.reopen_button.visible = state.step == "waiting"
        self.cancel_button.visible = busy

    def _on_state(self, state: SignInState) -> None:
        self.apply(state)
        try:
            self.update()
        except Exception:
            pass

    # ---- actions ----------------------------------------------------------------------------

    async def run(self) -> Optional[dict]:
        try:
            status = await self.oauth.sign_in("authgpt", self.oauth.state.account_id)
        except Exception as exc:
            self.error = str(exc)
            log.info("ChatGPT sign-in failed: %s", exc)
            return None
        if status:
            self.result = status
            if self.on_done is not None:
                self.on_done(status)
        return status or None

    def _start(self, e: Any = None) -> None:
        self._task = asyncio.ensure_future(self.run())

    def _toggle_paste(self, e: Any = None) -> None:
        self.paste_field.visible = not self.paste_field.visible
        self.finish_button.visible = self.paste_field.visible
        try:
            self.update()
        except Exception:
            pass

    async def finish_paste(self, text: str) -> Optional[dict]:
        try:
            status = await self.oauth.complete_with_paste(text)
        except Exception as exc:
            self.error = str(exc)
            return None
        self.result = status
        if self.on_done is not None:
            self.on_done(status)
        return status

    async def _finish_paste(self, e: Any = None) -> None:
        await self.finish_paste(self.paste_field.value or "")

    def _cancel(self, e: Any = None) -> None:
        self.oauth.cancel()
        if self.on_cancel is not None:
            self.on_cancel()


class LoginSheet:
    """LoginPanel in a bottom sheet (Send blocked fix action, ModelSheet "Sign in")."""

    def __init__(self, oauth: OAuthBridge, *, on_done: Optional[Callable[[dict], Any]] = None, autostart: bool = True) -> None:
        self._page: Any = None
        self._on_done = on_done
        self.panel = LoginPanel(oauth, on_done=self._done, on_cancel=self.close, autostart=autostart)
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=ft.Column(
                    [ft.Text("Sign in with ChatGPT", theme_style=ft.TextThemeStyle.TITLE_LARGE),
                     ft.Text("Your ChatGPT account powers GPT-6 Luna, the default model.",
                             theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                     self.panel],
                    tight=True,
                    spacing=10,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def _done(self, status: dict) -> None:
        self.close()
        if self._on_done is not None:
            self._on_done(status)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()


class AccountsScreen(Screen):
    title = "Accounts"

    def __init__(
        self,
        match: Any,
        *,
        oauth: OAuthBridge,
        notify: Optional[Callable[..., Any]] = None,
        on_signed_in_changed: Optional[Callable[[bool], Any]] = None,
    ) -> None:
        super().__init__(match)
        self.oauth = oauth
        self.notify = notify
        self.on_signed_in_changed = on_signed_in_changed
        self.status: dict = {}
        self.status_text = ft.Text("Checking…", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                   color=ft.Colors.ON_SURFACE_VARIANT)
        self.sign_out_button = ft.TextButton(content="Sign out", visible=False, on_click=self._sign_out)
        self.login_panel = LoginPanel(oauth, on_done=self._signed_in)
        self._refresh_task: Any = None

    def build_body(self) -> ft.Control:
        chatgpt = ft.Card(
            content=ft.Container(
                padding=16,
                content=ft.Column(
                    [
                        ft.Row([ft.Icon(ft.Icons.ACCOUNT_CIRCLE), ft.Text("ChatGPT", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                                                           expand=True), self.sign_out_button]),
                        self.status_text,
                        self.login_panel,
                        ft.Text("Account slots and authgpt0/ rotation arrive in U4.",
                                theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                    ],
                    spacing=8,
                    tight=True,
                ),
            ),
            key="account-authgpt",
        )
        others = [
            ft.ListTile(
                leading=ft.Icon(ft.Icons.ACCOUNT_CIRCLE_OUTLINED),
                title=ft.Text(label),
                trailing=ReasonChip(reason=reason or ""),
                disabled=True,
                key=f"account-{provider}",
            )
            for provider, label, reason in PROVIDERS
            if provider != "authgpt"
        ]
        unavailable = ft.ExpansionTile(
            title=f"Unavailable on mobile ({len(UNAVAILABLE_ACCOUNTS)})",
            expanded=True,
            controls=[
                ft.ListTile(title=ft.Text(label), trailing=ReasonChip(reason=NOT_ON_MOBILE, detail=reason), disabled=True)
                for label, reason in UNAVAILABLE_ACCOUNTS
            ],
        )
        return ft.ListView(controls=[chatgpt, *others, unavailable], expand=True, padding=12, spacing=8)

    def did_show(self) -> None:
        try:
            self._refresh_task = asyncio.ensure_future(self.refresh())
        except RuntimeError:  # no running loop (tests)
            pass

    async def refresh(self) -> dict:
        self.status = await self.oauth.refresh_status(0)
        self.apply_status(self.status)
        return self.status

    def apply_status(self, status: dict) -> None:
        if status.get("signed_in"):
            parts = ["Signed in"]
            if status.get("email"):
                parts.append(status["email"])
            if status.get("plan"):
                parts.append(f"plan {status['plan']}")
            self.status_text.value = " · ".join(parts) + " ✓"
        else:
            self.status_text.value = "Not signed in"
        self.sign_out_button.visible = bool(status.get("signed_in"))
        self.login_panel.visible = not status.get("signed_in")
        try:
            self.status_text.update()
            self.sign_out_button.update()
            self.login_panel.update()
        except Exception:
            pass

    def _signed_in(self, status: dict) -> None:
        self.status = status
        self.apply_status(status)
        if self.on_signed_in_changed is not None:
            self.on_signed_in_changed(True)

    async def _sign_out(self, e: Any = None) -> None:
        await self.oauth.sign_out(0)
        self.status = {"signed_in": False}
        self.apply_status(self.status)
        if self.on_signed_in_changed is not None:
            self.on_signed_in_changed(False)
        if self.notify is not None:
            self.notify("Signed out of ChatGPT")
