"""Accounts (``/settings/accounts``, UI_SPEC §4.13) and the LoginSheet (§4.13, §2.4).

Provider cards in spec order: **ChatGPT · Grok · Claude · Gemini**. Each card lists
its account slots ("#N · email · ✓ status · token refreshes in 2 h") with ⋯ Re-login /
Log out (/ 📊 Status for Gemini), a "＋ Add account" row that takes the next free slot,
and the rotation note (``authgpt0/`` / ``authgrok0/`` / ``authgem-vertex0/`` use every
slot). Gemini also has the GCP project picker for ``authgem-vertex/`` (config
``authgem_project``, desktop ``authgem_project_combo``). Experimental browser-token
routes (AuthND, Gemini Free) and the "Unavailable on mobile" routes are always listed,
disabled, with a ReasonChip.

``LoginPanel`` is the LoginSheet body shared by this screen, the Welcome flow and the
blocked Send button's "Sign in with ChatGPT" fix action: steps Opening browser ->
Waiting for sign-in -> Exchanging token -> Done, with **Reopen browser**, **Paste
redirect URL / code** and **Cancel** (loopback providers) or the device code with
**Copy code** / **Open verification page** (Grok) - all through
``services.oauth.OAuthBridge``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.oauth import PROVIDER_INFO, PROVIDERS, OAuthBridge, SignInState, provider_for_model
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.reason_chip import NOT_ON_MOBILE, ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = [
    "AccountsScreen",
    "EXPERIMENTAL_ACCOUNTS",
    "LoginPanel",
    "LoginSheet",
    "UNAVAILABLE_ACCOUNTS",
    "expiry_label",
    "slot_status_line",
]

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

#: UI_SPEC §4.13 "Experimental (U9, best effort)": browser-token routes through the WebViewBridge.
EXPERIMENTAL_ACCOUNTS = (
    ("AuthND (NVIDIA Build)", "authnd/", "Experimental · arrives in U9 (off-screen WebView token bridge)"),
    ("Gemini Free (search/gemini)", "search/gemini", "Experimental · arrives in U9 (off-screen WebView token bridge)"),
)

PASTE_HINT = PROVIDER_INFO["authgpt"].paste_hint  # U3 name

#: LoginPanel when a sign-in of its slot was started earlier and its state is still saved.
PENDING_TEXT = ("A sign-in started earlier can still be finished: paste the redirect URL or code the browser "
                "showed. “Start over” begins a new sign-in instead.")


def expiry_label(expires_at: Any, now: Optional[float] = None) -> str:
    """``"token refreshes in 45 min"`` / ``"token expired · refreshes on next use"`` / ``""``."""
    try:
        when = float(expires_at)
    except (TypeError, ValueError):
        return ""
    if when > 10_000_000_000:  # milliseconds
        when /= 1000.0
    left = when - (time.time() if now is None else now)
    if left <= 0:
        return "token expired · refreshes on next use"
    if left < 3600:
        return f"token refreshes in {max(1, int(left // 60))} min"
    if left < 86400 * 2:
        return f"token refreshes in {int(left // 3600)} h"
    return f"token refreshes in {int(left // 86400)} d"


def slot_status_line(status: dict, now: Optional[float] = None) -> str:
    """One slot row's subtitle: ``"✓ Signed in · reader@example.com · plan plus · token refreshes in 2 h"``."""
    if status.get("error"):
        return f"⚠ {status['error']}"
    if not status.get("signed_in"):
        return "Not signed in"
    parts = ["✓ Signed in"]
    who = status.get("email") or status.get("name")
    if who:
        parts.append(str(who))
    if status.get("plan"):
        parts.append(f"plan {status['plan']}")
    if status.get("source") and status.get("source") not in ("glossarion", "glossarion_oauth"):
        parts.append(f"from {status['source']}")
    expiry = expiry_label(status.get("expires_at"), now)
    if expiry:
        parts.append(expiry)
    return " · ".join(parts)


def _slot_title(provider: str, account_id: int) -> str:
    label = PROVIDER_INFO[provider].label
    return f"{label} #{account_id}" if account_id else f"{label} #0 (default)"


def _sign_in_title(state: SignInState) -> str:
    """``"ChatGPT"`` / ``"Gemini #3"`` for a sign-in state."""
    info = PROVIDER_INFO.get(state.provider)
    label = info.label if info is not None else str(state.provider)
    return f"{label} #{state.account_id}" if state.account_id else label


class LoginPanel(ft.Column):
    """Sign-in progress + paste fallback (loopback) or device code (Grok) for one provider slot.

    ``account_id`` None means slot #0. ``autostart`` opens the browser when the panel mounts,
    except when a loopback sign-in of this slot is still saved (the app or its loopback
    listener was killed after the browser opened): starting again would replace the saved
    PKCE verifier and state, so the panel offers the paste first and "Start over".

    While waiting, a ``SignInState.notice`` (the sign-in service is gone, the user came back
    with no callback yet, the listener is gone) is shown and opens the paste field at once.
    """

    def __init__(
        self,
        oauth: OAuthBridge,
        *,
        provider: str = "authgpt",
        account_id: Optional[int] = None,
        on_done: Optional[Callable[[dict], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
        autostart: bool = False,
        copy_text: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__(spacing=10, tight=True)
        self.oauth = oauth
        self.provider = provider
        self.account_id = account_id
        self.info = PROVIDER_INFO[provider]
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.copy_text = copy_text
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.pending = False  # a saved sign-in of this slot waits for its pasted redirect
        self._task: Any = None
        self.step_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM)
        self.ring = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False,
                                  selectable=True)
        # SignInState.notice: still waiting and maybe stuck (sign-in service gone, back in the app with no
        # callback, listener gone); the paste field opens with it.
        self.notice_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                   visible=False, key="sign-in-notice")
        self.start_button = ft.FilledButton(content=self.sign_in_label, on_click=self._start)
        self.reopen_button = ft.TextButton(content="Reopen browser", on_click=lambda e: self.oauth.reopen_browser())
        loopback = self.info.flow == "loopback"
        self.paste_field = ft.TextField(hint_text=self.info.paste_hint or PASTE_HINT, dense=True, visible=False,
                                        multiline=False, on_submit=self._finish_paste)
        self.paste_button = ft.TextButton(content="Paste redirect URL / code", on_click=self._toggle_paste,
                                          visible=loopback)
        # Claude: Anthropic's code page shows code#state to paste when the return to the app fails.
        self.manual_button = ft.TextButton(content="Get a code to paste instead", icon=ft.Icons.PIN_OUTLINED,
                                           on_click=self._open_manual, visible=False)
        self.finish_button = ft.FilledTonalButton(content="Finish sign-in", visible=False, on_click=self._finish_paste)
        self.cancel_button = ft.TextButton(content="Cancel", on_click=self._cancel)
        # Device code (Grok): show the code, copy it, open the verification page.
        self.code_text = ft.Text("", selectable=True, size=24, weight=ft.FontWeight.W_700,
                                 font_family="monospace", key="device-code")
        self.copy_button = ft.TextButton(content="Copy code", icon=ft.Icons.CONTENT_COPY, on_click=self._copy_code)
        self.verify_button = ft.TextButton(content="Open verification page", icon=ft.Icons.OPEN_IN_NEW,
                                           on_click=lambda e: self.oauth.reopen_browser())
        self.device_box = ft.Container(
            visible=False,
            padding=12,
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            content=ft.Column(
                [ft.Text("Approve this sign-in on the xAI page. If it asks for a code, enter:",
                         theme_style=ft.TextThemeStyle.BODY_SMALL),
                 self.code_text,
                 ft.Row([self.copy_button, self.verify_button], wrap=True, spacing=4)],
                spacing=6, tight=True,
            ),
        )
        self.controls = [
            ft.Row([self.ring, self.step_text], spacing=8),
            self.error_text,
            self.notice_text,
            self.device_box,
            ft.Row([self.start_button, self.reopen_button], wrap=True, spacing=8),
            self.manual_button,
            self.paste_button,
            self.paste_field,
            ft.Row([self.finish_button, self.cancel_button], wrap=True, spacing=8),
        ]
        self._unsub = oauth.subscribe(self._on_state)
        self.apply(oauth.state)
        self._autostart = autostart

    @property
    def sign_in_label(self) -> str:
        return f"Sign in with {self.info.label}"

    def _target_account(self) -> int:
        """The slot this panel signs in: the one given, else slot #0 (never the bridge's last slot)."""
        return int(self.account_id or 0)

    def _mine(self, state: SignInState) -> bool:
        return state.provider == self.provider and int(state.account_id or 0) == self._target_account()

    def did_mount(self) -> None:
        super().did_mount()
        if self._autostart and not self.oauth.state.busy:
            self._autostart = False
            self._task = asyncio.ensure_future(self.auto_start())

    async def auto_start(self) -> Optional[dict]:
        """``autostart``: open the browser, unless a sign-in of this slot is still saved (then the
        paste comes first; "Start over" runs ``run``)."""
        if self.info.flow == "loopback" and await self._saved_sign_in():
            self.show_pending()
            return None
        return await self.run()

    async def _saved_sign_in(self) -> bool:
        checker = getattr(self.oauth, "pending_sign_in", None)
        if checker is None:
            return False
        try:
            return bool(await checker(self._target_account(), self.provider))
        except Exception:
            log.debug("checking for a saved %s sign-in failed", self.info.label, exc_info=True)
            return False

    def show_pending(self) -> None:
        self.pending = True
        self.apply(self.oauth.state)
        self._push()

    def _push(self) -> None:
        try:
            self.update()
        except Exception:
            pass

    def will_unmount(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        super().will_unmount()

    # ---- state ----------------------------------------------------------------------------

    def apply(self, state: SignInState) -> None:
        # Another provider's / slot's sign-in in progress: the bridge runs one at a time.
        other = state if state.busy and not self._mine(state) else None
        if not self._mine(state):
            state = SignInState(provider=self.provider, account_id=self._target_account())
        busy = state.busy
        if busy or state.step == "done":
            self.pending = False
        idle = not busy and state.step in ("idle", "cancelled", "error")
        self.ring.visible = busy
        if state.step == "done":
            who = f" as {state.email}" if state.email else ""
            self.step_text.value = f"Done · signed in{who}"
        elif other is not None:
            self.step_text.value = f"Finish or cancel the {_sign_in_title(other)} sign-in first."
        elif self.pending and state.step == "idle":
            self.step_text.value = PENDING_TEXT
        else:
            self.step_text.value = state.label
        self.error_text.visible = state.step == "error"
        self.error_text.value = state.message if state.step == "error" else ""
        self.start_button.visible = not busy and state.step != "done"
        self.start_button.disabled = other is not None
        if self.pending and idle and state.step == "idle":
            self.start_button.content = "Start over"
        else:
            self.start_button.content = "Try again" if state.step in ("error", "cancelled") else self.sign_in_label
        device = bool(state.user_code) and state.step == "waiting"
        self.device_box.visible = device
        self.code_text.value = state.user_code if device else ""
        self.reopen_button.visible = state.step == "waiting" and not device
        self.manual_button.visible = state.step == "waiting" and bool(state.manual_url)
        self.cancel_button.visible = busy
        stalled = state.step == "waiting" and bool(state.notice) and not device
        self.notice_text.visible = stalled
        self.notice_text.value = state.notice if stalled else ""
        if self.info.flow != "loopback":
            self.paste_field.visible = False
            self.finish_button.visible = False
        elif (self.pending and idle) or stalled:
            self.paste_field.visible = True
            self.finish_button.visible = True
        elif state.step == "done":
            self.paste_field.visible = False
            self.finish_button.visible = False

    def _on_state(self, state: SignInState) -> None:
        self.apply(state)
        self._push()

    def _show_error(self, message: str) -> None:
        """An error the bridge raised without a state change (e.g. another sign-in is running)."""
        self.error = message
        self.error_text.value = message
        self.error_text.visible = bool(message)
        self._push()

    # ---- actions ----------------------------------------------------------------------------

    async def run(self) -> Optional[dict]:
        self.pending = False
        try:
            status = await self.oauth.sign_in(self.provider, self._target_account())
        except Exception as exc:
            log.info("%s sign-in failed: %s", self.info.label, exc)
            self._show_error(str(exc))
            return None
        if status:
            self.result = status
            if self.on_done is not None:
                self.on_done(status)
        return status or None

    def _start(self, e: Any = None) -> None:
        self._task = asyncio.ensure_future(self.run())

    def _open_manual(self, e: Any = None) -> None:
        if self.oauth.open_manual_page():
            self.paste_field.visible = True
            self.finish_button.visible = True
            try:
                self.update()
            except Exception:
                pass

    def _toggle_paste(self, e: Any = None) -> None:
        self.paste_field.visible = not self.paste_field.visible
        self.finish_button.visible = self.paste_field.visible
        try:
            self.update()
        except Exception:
            pass

    async def finish_paste(self, text: str) -> Optional[dict]:
        try:
            status = await self.oauth.complete_with_paste(text, self._target_account(), self.provider)
        except Exception as exc:
            self._show_error(str(exc))
            return None
        self.result = status
        if self.on_done is not None:
            self.on_done(status)
        return status

    async def _finish_paste(self, e: Any = None) -> None:
        await self.finish_paste(self.paste_field.value or "")

    async def _copy_code(self, e: Any = None) -> None:
        code = self.code_text.value or self.oauth.state.user_code
        if not code or self.copy_text is None:
            return
        result = self.copy_text(code)
        if asyncio.iscoroutine(result):
            await result

    def _cancel(self, e: Any = None) -> None:
        self.oauth.cancel()
        if self.on_cancel is not None:
            self.on_cancel()


class LoginSheet:
    """LoginPanel in a bottom sheet (Accounts slot rows, Send blocked fix action, ModelSheet "Sign in")."""

    def __init__(
        self,
        oauth: OAuthBridge,
        *,
        on_done: Optional[Callable[[dict], Any]] = None,
        autostart: bool = True,
        provider: str = "authgpt",
        account_id: Optional[int] = None,
        copy_text: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._page: Any = None
        self._on_done = on_done
        info = PROVIDER_INFO[provider]
        self.panel = LoginPanel(oauth, provider=provider, account_id=account_id, on_done=self._done,
                                on_cancel=self.close, autostart=autostart, copy_text=copy_text)
        title = f"Sign in with {info.label}"
        if account_id:
            title += f" · account #{account_id}"
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=ft.Column(
                    [ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE),
                     ft.Text(info.blurb, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
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
        # By identity: pop_dialog() would close a newer SnackBar (e.g. "Copied") instead of the sheet.
        if self._page is not None:
            close_dialog(self._page, self.dialog)


class AccountsScreen(Screen):
    title = "Accounts"

    def __init__(
        self,
        match: Any,
        *,
        oauth: OAuthBridge,
        notify: Optional[Callable[..., Any]] = None,
        on_signed_in_changed: Optional[Callable[[bool], Any]] = None,
        page: Any = None,
        config_get: Optional[Callable[[str, Any], Any]] = None,
        config_set: Optional[Callable[[dict], Any]] = None,
        copy_text: Optional[Callable[[str], Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        tablet: bool = False,
        on_refreshed: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(match)
        self.oauth = oauth
        self.on_refreshed = on_refreshed  # the app mirrors oauth.signed_in into AppState.signed_in
        self.notify = notify
        self.on_signed_in_changed = on_signed_in_changed
        self.page = page
        self.config_get = config_get
        self.config_set = config_set
        self.copy_text = copy_text
        self.run_io = run_io
        self.tablet = tablet
        self.providers = [p for p, _label, _reason in PROVIDERS]
        self.slots: dict[str, list] = {p: [0] for p in self.providers}
        self.statuses: dict[str, dict] = {p: {} for p in self.providers}  # provider -> {slot: status}
        self.status: dict = {}  # ChatGPT #0 (U3 API)
        self.cards: dict[str, ft.Control] = {}
        self.slot_columns: dict[str, ft.Column] = {}
        self.slot_rows: dict[tuple, ft.ListTile] = {}
        self.project_dropdown: Optional[ft.Dropdown] = None
        self.project_field: Optional[ft.TextField] = None
        self.project_note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.projects: list = []
        self.sheet: Any = None
        self._refresh_task: Any = None
        self._unsub: Optional[Callable[[], None]] = None

    # ---- threading ------------------------------------------------------------------------

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _say(self, message: str) -> None:
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:
                log.info("accounts: %s", message)

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.config_get is None:
            return default
        try:
            return self.config_get(key, default)
        except Exception:
            return default

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    def _page(self) -> Any:
        """The page to open sheets on: the one given, else the mounted body's (U3 construction)."""
        if self.page is not None:
            return self.page
        if self.body is None:
            return None
        try:
            return self.body.page  # raises while the body is not on a page
        except Exception:
            return None

    # ---- body ------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        controls: list[ft.Control] = [self._provider_card(provider) for provider in self.providers]
        controls.append(ft.ExpansionTile(
            title=f"Experimental ({len(EXPERIMENTAL_ACCOUNTS)})",
            expanded=False,
            key="accounts-experimental",
            controls=[
                ft.ListTile(title=ft.Text(label), subtitle=ft.Text(route, theme_style=ft.TextThemeStyle.BODY_SMALL),
                            trailing=ReasonChip(reason="Experimental", detail=reason), disabled=True)
                for label, route, reason in EXPERIMENTAL_ACCOUNTS
            ],
        ))
        controls.append(ft.ExpansionTile(
            title=f"Unavailable on mobile ({len(UNAVAILABLE_ACCOUNTS)})",
            expanded=True,
            key="accounts-unavailable",
            controls=[
                ft.ListTile(title=ft.Text(label), trailing=ReasonChip(reason=NOT_ON_MOBILE, detail=reason), disabled=True)
                for label, reason in UNAVAILABLE_ACCOUNTS
            ],
        ))
        return ft.ListView(controls=controls, expand=True, padding=12, spacing=8)

    def _provider_card(self, provider: str) -> ft.Control:
        info = PROVIDER_INFO[provider]
        column = ft.Column(spacing=0, tight=True)
        self.slot_columns[provider] = column
        self._render_slots(provider, push=False)
        rows: list[ft.Control] = [
            ft.Row([ft.Icon(icon_data(info.icon), color=ft.Colors.PRIMARY),
                    ft.Text(info.label, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600,
                            expand=True)],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(info.blurb, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            column,
            ft.TextButton(content="＋ Add account", icon=ft.Icons.PERSON_ADD_ALT, key=f"add-{provider}",
                          on_click=lambda e, p=provider: call_handler(self.add_account, p)),
            ft.Text(info.rotation_note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ]
        if provider == "authcd":
            rows.append(ft.Text("If the browser shows a code instead of returning to the app, use "
                                "“Paste redirect URL / code”.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        if provider == "authgem":
            rows.append(self._project_row())
        card = ft.Container(
            content=ft.Column(rows, spacing=6, tight=True),
            padding=tokens.SPACING["card_padding"],
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            key=f"account-{provider}",
        )
        self.cards[provider] = card
        return card

    def _slot_row(self, provider: str, account_id: int) -> ft.ListTile:
        status = self.statuses.get(provider, {}).get(account_id, {})
        state = self.oauth.state
        busy_here = state.busy and state.provider == provider and int(state.account_id or 0) == account_id
        subtitle = state.label if busy_here else (slot_status_line(status) if status else "Checking…")
        signed = bool(status.get("signed_in"))
        if signed:
            trailing: ft.Control = ft.IconButton(
                icon=ft.Icons.MORE_VERT, tooltip="Account actions", size_constraints=HIT_TARGET,
                on_click=lambda e, p=provider, a=account_id: self.slot_actions(p, a),
            )
        else:
            trailing = ft.FilledTonalButton(content="Sign in", disabled=busy_here,
                                            on_click=lambda e, p=provider, a=account_id: self.open_login(p, a))
        row = ft.ListTile(
            leading=ft.CircleAvatar(content=ft.Text(f"#{account_id}", size=12), radius=16,
                                    bgcolor=ft.Colors.PRIMARY_CONTAINER if signed else ft.Colors.SURFACE_CONTAINER_HIGHEST),
            title=ft.Text(_slot_title(provider, account_id), theme_style=ft.TextThemeStyle.BODY_MEDIUM),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL,
                             color=ft.Colors.ON_SURFACE_VARIANT if not status.get("error") else ft.Colors.ERROR),
            trailing=trailing,
            min_height=tokens.SIZES["hit_target"],
            dense=True,
            key=f"slot-{provider}-{account_id}",
        )
        self.slot_rows[(provider, account_id)] = row
        return row

    def _render_slots(self, provider: str, push: bool = True) -> None:
        column = self.slot_columns.get(provider)
        if column is None:
            return
        column.controls = [self._slot_row(provider, account_id) for account_id in self.slots.get(provider, [0])]
        if push:
            self._push(column)

    # ---- Gemini GCP project picker -----------------------------------------------------------

    def _project_row(self) -> ft.Control:
        current = str(self._cfg("authgem_project", "") or "")
        self.project_dropdown = ft.Dropdown(
            label="GCP project (authgem-vertex/)",
            value=current or None,
            options=[ft.DropdownOption(key=current, text=current)] if current else [],
            on_select=lambda e: self.select_project(e.control.value or ""),
            dense=True,
            expand=True,
            key="authgem-project",
        )
        self.project_field = ft.TextField(hint_text="Or type a project id", dense=True, expand=True,
                                          on_submit=lambda e: self.select_project(e.control.value or ""))
        self.project_note.value = "Vertex AI needs a Google Cloud project with billing. authgem/ (AI Studio) does not."
        return ft.Column(
            [
                ft.Row([self.project_dropdown,
                        ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Load projects", size_constraints=HIT_TARGET,
                                      on_click=self._on_load_projects)],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.project_field,
                self.project_note,
            ],
            spacing=4,
            tight=True,
        )

    def _gemini_slot(self) -> int:
        """Slot the project picker uses: the model's Gemini slot when signed in, else the first signed-in slot."""
        statuses = self.statuses.get("authgem", {})
        route = provider_for_model(str(self._cfg("model", "") or ""))
        if route is not None and route[0] == "authgem" and statuses.get(route[1], {}).get("signed_in"):
            return route[1]
        for account_id in sorted(statuses):
            if statuses[account_id].get("signed_in"):
                return account_id
        return 0

    async def load_projects(self) -> list:
        account_id = self._gemini_slot()
        try:
            projects = await self._io(self.oauth.gemini_projects, account_id)
        except Exception as exc:
            self.project_note.value = f"Could not list projects: {exc}"
            self._push(self.project_note)
            return []
        self.projects = list(projects)
        # The desktop dropdown items and selection rule (authgem_auth.authgem_project_items /
        # choose_authgem_project_index); the local labels only for a build without them.
        items = self.oauth.gemini_project_items(self.projects) if hasattr(self.oauth, "gemini_project_items") else None
        if items is None:
            marks = {"billed": "✅", "unknown": "❔", "unbilled": "⚠️"}
            items = [(f"{marks.get(status, '❔')} {pid}{' (no billing)' if status == 'unbilled' else ''}", pid)
                     for pid, status in self.projects]
        options = [ft.DropdownOption(key=pid, text=label) for label, pid in items]
        if self.project_dropdown is not None:
            self.project_dropdown.options = options
        chosen = None
        if self.projects and hasattr(self.oauth, "gemini_project_choice"):
            try:
                chosen = self.oauth.gemini_project_choice(self.projects, str(self._cfg("authgem_project", "") or ""))
            except Exception:
                log.debug("choosing the GCP project failed", exc_info=True)
        if chosen:
            self.apply_project(chosen)  # desktop: the picker applies its selection at once
        billed = [pid for pid, status in self.projects if status == "billed"]
        self.project_note.value = (f"Found {len(billed)} GCP project(s) with billing enabled" if billed
                                   else "⚠️ No GCP projects with billing found — Vertex AI won't work"
                                   if self.projects else "No projects found; type a project id.")
        self._push(self.project_dropdown, self.project_note)
        return self.projects

    async def _on_load_projects(self, e: Any = None) -> None:
        await self.load_projects()

    def apply_project(self, project_id: str) -> Optional[str]:
        """Save and push ``project_id`` (desktop ``_authgem_project_changed``) and select it in the picker."""
        project_id = str(project_id or "").strip()
        if not project_id:
            return None
        if self.config_set is not None:
            self.config_set({"authgem_project": project_id})
        try:
            self.oauth.set_gemini_project(project_id, self._gemini_slot())
        except Exception:
            log.debug("set_gemini_project failed", exc_info=True)
        if self.project_dropdown is not None:
            if all(getattr(o, "key", None) != project_id for o in self.project_dropdown.options or []):
                self.project_dropdown.options = list(self.project_dropdown.options or []) + [
                    ft.DropdownOption(key=project_id, text=project_id)]
            self.project_dropdown.value = project_id
        return project_id

    def select_project(self, project_id: str) -> Optional[str]:
        project_id = self.apply_project(project_id)
        if not project_id:
            return None
        self.project_note.value = f"📁 AuthGem project set: {project_id}"
        self._push(self.project_dropdown, self.project_note)
        return project_id

    # ---- lifecycle -------------------------------------------------------------------------

    def did_show(self) -> None:
        if self._unsub is None:
            self._unsub = self.oauth.subscribe(self._on_sign_in_state)
        try:
            self._refresh_task = asyncio.ensure_future(self.refresh())
        except RuntimeError:  # no running loop (tests)
            pass

    def dispose(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def _on_sign_in_state(self, state: SignInState) -> None:
        if state.provider in self.slot_columns:
            self._render_slots(state.provider)

    async def refresh(self, provider: Optional[str] = None) -> dict:
        """Re-read the slots and their status (worker thread). Returns ``{provider: [status, ...]}``."""
        out: dict = {}
        for name in ([provider] if provider else self.providers):
            try:
                slots = await self._io(self.oauth.account_slots, name, self.config_get)
                statuses = await self._io(self.oauth.statuses, name, slots)
            except Exception as exc:
                log.warning("reading %s accounts failed: %s", name, exc)
                slots, statuses = [0], [{"signed_in": False, "error": str(exc)}]
            self.slots[name] = list(slots)
            self.statuses[name] = {int(s.get("account_id", slot) or 0): s for slot, s in zip(slots, statuses)}
            out[name] = list(statuses)
            self._render_slots(name)
        chatgpt = self.statuses.get("authgpt", {}).get(0)
        if chatgpt is not None:
            self.status = chatgpt
        if self.on_refreshed is not None:
            try:
                self.on_refreshed()
            except Exception:
                log.debug("on_refreshed failed", exc_info=True)
        return out

    # ---- actions ----------------------------------------------------------------------------

    def open_login(self, provider: str, account_id: int) -> Optional[LoginSheet]:
        sheet = LoginSheet(self.oauth, provider=provider, account_id=account_id, copy_text=self.copy_text,
                           on_done=lambda status, p=provider: self._signed_in(p, status))
        self.sheet = sheet
        page = self._page()
        if page is not None:
            sheet.show(page)
        return sheet

    async def add_account(self, provider: str) -> Optional[int]:
        """"＋ Add account": the next free slot (Grok asks its token stores, off the loop), then its LoginSheet."""
        try:
            account_id = await self._io(self.oauth.next_slot, provider, list(self.slots.get(provider) or [0]))
        except Exception as exc:
            self._say(f"Could not allocate another {PROVIDER_INFO[provider].label} account slot: {exc}")
            return None
        self.oauth.add_slot(provider, account_id)
        if account_id not in self.slots.setdefault(provider, [0]):
            self.slots[provider] = sorted(self.slots[provider] + [account_id])
        self._render_slots(provider)
        self.open_login(provider, account_id)
        return account_id

    def slot_actions(self, provider: str, account_id: int) -> ActionSheet:
        label = _slot_title(provider, account_id)
        items = [
            ActionItem("Re-login", lambda: self.open_login(provider, account_id), icon="LOGIN"),
            ActionItem("Log out", lambda: self.confirm_sign_out(provider, account_id), icon="LOGOUT", destructive=True),
        ]
        if provider == "authgem":
            items.insert(1, ActionItem("📊 Status", lambda: self._spawn(self.show_gemini_status(account_id)),
                                       icon="QUERY_STATS"))
        sheet = ActionSheet(items, title=label, subtitle=slot_status_line(self.statuses.get(provider, {}).get(account_id, {})),
                            tablet=self.tablet)
        self.sheet = sheet
        page = self._page()
        if page is not None:
            sheet.show(page)
        return sheet

    def confirm_sign_out(self, provider: str, account_id: int) -> ConfirmDialog:
        status = self.statuses.get(provider, {}).get(account_id, {})
        who = status.get("email") or status.get("name") or "unknown"
        suffix = f" #{account_id}" if account_id else ""
        dialog = ConfirmDialog(
            title=f"{PROVIDER_INFO[provider].label} Account{suffix}",
            body=f"Currently logged in as: {who}{suffix}\n\nDo you want to log out?",  # desktop question
            confirm_label="Log out",
            cancel_label="No",
            destructive=True,
            on_confirm=lambda: self.sign_out(provider, account_id),
        )
        page = self._page()
        if page is not None:
            dialog.show(page)
        return dialog

    async def sign_out(self, provider: str, account_id: int) -> bool:
        try:
            await self.oauth.sign_out(account_id, provider)
        except Exception as exc:
            self._say(str(exc))
            return False
        self.statuses.setdefault(provider, {})[account_id] = {"provider": provider, "account_id": account_id,
                                                              "signed_in": False}
        if provider == "authgpt" and account_id == 0:
            self.status = self.statuses[provider][account_id]
        self._render_slots(provider)
        if self.on_signed_in_changed is not None:
            self.on_signed_in_changed(False)
        suffix = f" #{account_id}" if account_id else ""
        self._say(f"🔓 {PROVIDER_INFO[provider].label}{suffix}: Logged out")
        return True

    def _signed_in(self, provider: str, status: dict) -> None:
        account_id = int(status.get("account_id", 0) or 0)
        self.statuses.setdefault(provider, {})[account_id] = status
        if account_id not in self.slots.setdefault(provider, [0]):
            self.slots[provider] = sorted(self.slots[provider] + [account_id])
        if provider == "authgpt" and account_id == 0:
            self.status = status
        self._render_slots(provider)
        if self.on_signed_in_changed is not None:
            self.on_signed_in_changed(True)
        if provider == "authgem" and self.project_dropdown is not None and not self.projects:
            self._spawn(self.load_projects())

    async def show_gemini_status(self, account_id: int) -> Optional[InfoSheet]:
        suffix = f" #{account_id}" if account_id else ""
        try:
            status = await self._io(self.oauth.gemini_status, account_id)
        except Exception as exc:
            self._say(f"❌ Gemini status check failed: {exc}")
            return None
        lines: list[str] = []
        actions: list[ft.Control] = []
        if status.get("error"):
            lines.append(f"❌ Gemini{suffix} status: {status['error']}")
        elif not status.get("verified", True):
            lines.append(f"⚠️ Gemini{suffix}: {status.get('verification_message') or 'Account verification required'}")
            url = status.get("verification_url") or ""
            if url:
                actions.append(ft.FilledTonalButton(content="Open verification page",
                                                    on_click=lambda e, u=url: self.oauth.open_url(u)))
            lines.append("💡 After completing verification, check the status again to confirm.")
        else:
            lines.append(f"✅ Gemini{suffix}: Account verified")
            lines.append(f"📊 {status.get('sub_label', '')} | Credits: {status.get('credit_label', '')}")
            if status.get("project"):
                lines.append(f"Project: {status['project']}")
            quota = list(status.get("quota_lines") or [])
            if quota:
                lines.append("⚠️ Daily Quota: EXHAUSTED" if status.get("quota_exhausted") else "📊 Daily Quota:")
                lines.extend(str(q) for q in quota)
            else:
                lines.append("No quota data available")
        sheet = InfoSheet(title=f"Gemini{suffix} status", body="\n".join(lines), actions=actions or None)
        self.sheet = sheet
        page = self._page()
        if page is not None:
            sheet.show(page)
        return sheet

    def _spawn(self, coro: Any) -> Any:
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None
