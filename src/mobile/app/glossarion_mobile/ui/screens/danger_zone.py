"""Danger zone (``/settings/danger``; UI_SPEC §4.16, desktop Other Settings › Danger Zone).

* **Reset settings to defaults** - the desktop ``_reset_config_to_defaults``: the
  same confirmation text and the same preserved keys (API keys and every key pool,
  their toggles, Replicate / Azure / Google credentials, the model, prompt profiles and
  the active profile, the QA scanner's excluded characters). The desktop writes
  config.json with only those keys and restarts; the phone backs up config.json
  first (UI_SPEC "creates an automatic backup first"), replaces the store's content
  with the preserved keys (``MobileConfigStore.revert_to``) and saves - the app keeps
  running and the next job starts from fresh-install defaults.
* **Sign out everywhere** - every saved slot of ChatGPT, Grok, Claude and Gemini
  (``OAuthBridge.sign_out_everywhere``).
* **Wipe app data** - type WIPE to confirm; deletes config.json and its backups,
  mobile_state.json, chats, sign-in tokens, the Inbox, Output and Library, caches and
  temporary files, then closes the app (the in-memory stores must not write the old
  state back). Logs are kept.

``preserved_reset_keys`` is ``config_store.reset_preserved_keys``, the desktop preservation
block moved into the shared core (other_settings calls it too).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Callable, Optional

import flet as ft

# The desktop reset's preserved-key list text and ``keys_to_preserve`` (other_settings
# ``_reset_config_to_defaults``), shared core (backend on sys.path).
from config_store import RESET_PRESERVED_TEXT
from config_store import reset_preserved_keys as preserved_reset_keys
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.screens.page_base import PageScreen, section

__all__ = [
    "DangerZoneScreen",
    "RESET_PRESERVED_TEXT",
    "WIPE_WORD",
    "preserved_reset_keys",
    "wipe_targets",
    "wipe_app_data",
]

log = logging.getLogger("glossarion.danger")

WIPE_WORD = "WIPE"


def reset_settings(store: Any) -> dict:
    """Blocking: backup, then keep only the preserved keys and save. Returns what was kept."""
    try:
        store.backup_now()
    except Exception as exc:
        log.info("backup before reset failed: %s", exc)
    kept = preserved_reset_keys(store.snapshot())
    store.revert_to(kept)
    store.flush()
    return kept


def wipe_targets(paths: Any, *, keep: tuple = ("logs",)) -> list:
    """Everything "Wipe app data" deletes: the children of data / docs / cache / temp, minus ``keep`` in data."""
    targets: list = []
    seen: set = set()
    data = str(getattr(paths, "data", "") or "")
    for root_attr in ("data", "docs", "cache", "temp"):
        root = str(getattr(paths, root_attr, "") or "")
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if root == data and name in keep:
                continue
            target = os.path.join(root, name)
            norm = os.path.normcase(os.path.abspath(target))
            if norm in seen:
                continue
            # Never delete a root that is itself one of the other roots (iOS docs inside data, etc.)
            if any(os.path.normcase(os.path.abspath(str(getattr(paths, a, "") or ""))) == norm
                   for a in ("data", "docs", "cache", "temp", "logs")):
                continue
            seen.add(norm)
            targets.append(target)
    return targets


def wipe_app_data(paths: Any, *, stop: Optional[Callable[[], Any]] = None) -> list:
    """Blocking: stop the stores' savers (``stop``), then delete ``wipe_targets``; returns what failed."""
    if stop is not None:
        try:
            stop()
        except Exception:
            log.exception("stopping the stores before the wipe failed")
    failed = []
    for target in wipe_targets(paths):
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        except OSError as exc:
            failed.append(f"{target}: {exc}")
    return failed


class DangerZoneScreen(PageScreen):
    title = "Danger zone"

    def __init__(self, match: Any, ctx: Any, *, oauth: Any = None, paths: Any = None,
                 stop_stores: Optional[Callable[[], Any]] = None, exit_app: Optional[Callable[[], Any]] = None,
                 on_reset: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(match, ctx)
        self.oauth = oauth
        self.paths = paths
        self.stop_stores = stop_stores
        self.exit_app = exit_app
        self.on_reset = on_reset
        self.wipe_field = ft.TextField(label=f"Type {WIPE_WORD} to confirm", dense=True,
                                       on_change=lambda e: self._wipe_ready(), key="danger-wipe-field")

    def build_body(self) -> ft.Control:
        self.wipe_button = ft.FilledButton(
            content="Wipe app data", disabled=True, key="danger-wipe",
            style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR),
            on_click=lambda e: self.confirm_wipe(),
        )
        return self.scaffold([
            section("Reset settings", [
                ft.Text("Reset all settings to default values. API keys and profiles will be preserved.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.FilledButton(content="⚠️ Reset Settings to Defaults", key="danger-reset",
                                style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR),
                                on_click=lambda e: self.confirm_reset()),
            ], key="danger-reset-card"),
            section("Accounts", [
                ft.Text("Sign every ChatGPT, Grok, Claude and Gemini account slot out of this device.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.FilledTonalButton(content="Sign out everywhere", icon=ft.Icons.LOGOUT, key="danger-signout",
                                     disabled=self.oauth is None, on_click=lambda e: self.confirm_sign_out()),
            ]),
            section("Wipe app data", [
                ft.Text("Deletes settings and backups, chats, sign-ins, imported files, outputs, the Library and caches "
                        "from this device, then closes Glossarion. Logs are kept. This cannot be undone.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                self.wipe_field,
                self.wipe_button,
            ]),
        ])

    # ---- reset ---------------------------------------------------------------------------

    def confirm_reset(self) -> ConfirmDialog:
        return self.show(ConfirmDialog(
            title="Reset to Defaults",
            body="Are you sure you want to reset ALL settings to default values?\n\n"
                 "A backup of your current config.json is made first.\n\n" + RESET_PRESERVED_TEXT,
            confirm_label="Yes", cancel_label="No", destructive=True,
            on_confirm=self.reset,
        ))

    async def reset(self) -> Optional[dict]:
        if self.store is None:
            return None
        try:
            kept = await self.io(reset_settings, self.store)
        except Exception as exc:
            self.say(f"Failed to reset config: {exc}")
            return None
        if self.on_reset is not None:
            self.on_reset()
        self.say("Settings reset to defaults (a backup was made first)")
        return kept

    # ---- sign out ------------------------------------------------------------------------

    def confirm_sign_out(self) -> ConfirmDialog:
        return self.show(ConfirmDialog(
            title="Sign out everywhere",
            body="Log out of every saved ChatGPT, Grok, Claude and Gemini account on this device?",
            confirm_label="Sign out", cancel_label="No", destructive=True,
            on_confirm=self.sign_out_everywhere,
        ))

    async def sign_out_everywhere(self) -> list:
        if self.oauth is None:
            return []
        config_get = getattr(self.store, "get", None)
        try:
            cleared = await self.io(self.oauth.sign_out_everywhere, config_get)
        except Exception as exc:
            self.say(f"Sign-out failed: {exc}")
            return []
        self.say(f"Signed out of {len(cleared)} account{'s' if len(cleared) != 1 else ''}" if cleared
                 else "No account was signed in")
        return cleared

    # ---- wipe ------------------------------------------------------------------------------

    def _wipe_ready(self) -> bool:
        ready = (self.wipe_field.value or "").strip() == WIPE_WORD
        if hasattr(self, "wipe_button"):
            self.wipe_button.disabled = not ready
            self.push(self.wipe_button)
        return ready

    def confirm_wipe(self) -> Optional[ConfirmDialog]:
        if not self._wipe_ready():
            return None
        return self.show(ConfirmDialog(
            title="Wipe app data",
            body="Everything Glossarion stored on this device will be deleted and the app will close. Continue?",
            confirm_label="Wipe", cancel_label="No", destructive=True,
            on_confirm=self.wipe,
        ))

    async def wipe(self) -> list:
        if self.paths is None:
            self.say("App folders are unknown in this session")
            return []
        failed = await self.io(wipe_app_data, self.paths, stop=self.stop_stores)
        if failed:
            log.warning("wipe left %d item(s): %s", len(failed), failed[:5])
        if self.exit_app is not None:
            self.exit_app()
        return failed
