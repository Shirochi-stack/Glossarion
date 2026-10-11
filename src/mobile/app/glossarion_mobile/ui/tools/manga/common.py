"""Small pieces the manga tabs share: the tab base class, the status chip of an ``OptionRow`` /
``ModelEntry`` and the ExportSheet for a produced file."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["CHIP_STATUS", "JobEnds", "MangaTab", "TranslationModel", "export_sheet", "option_chip", "push",
           "reason_or_chip"]

log = logging.getLogger("glossarion.tools.manga")

#: OptionRow / ModelEntry status -> ``tokens.STATUS_STYLES`` key (icon + colour; text from the row).
CHIP_STATUS = {
    "ready": "completed",
    "loaded": "completed",
    "needs_key": "failed",
    "not_downloaded": "pending",
    "missing": "pending",
    "downloading": "in_progress",
    "loading": "in_progress",
    "unavailable": "disabled",
    "error": "failed",
    "off": "skipped",
}


def push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:  # not mounted (host tests) / detached
            pass


class JobEnds:
    """``on_end(snapshot)`` whenever a job of any kind reaches a terminal state (the JobService's
    ``on_transition``; ``JobWatch`` only follows the jobs a tab started). A lookup that a running
    job refused (``services.manga.MangaBusy``: the job owned the process state) runs again once
    that job is over; the JobService reports the end after the job has let go of it."""

    def __init__(self, ctx: Any, on_end: Callable[[Any], Any]) -> None:
        self.ctx = ctx
        self.on_end = on_end
        self._unsub: Optional[Callable[[], Any]] = None

    def start(self) -> None:
        if self._unsub is not None:
            return
        jobs = self.ctx.jobs
        on_transition = getattr(jobs, "on_transition", None) if jobs is not None else None
        if callable(on_transition):
            try:
                self._unsub = on_transition(self._on_transition)
            except Exception:
                log.debug("subscribing to job transitions failed", exc_info=True)

    def stop(self) -> None:
        unsub, self._unsub = self._unsub, None
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass

    def _on_transition(self, snap: Any, previous: Any) -> None:
        if getattr(snap, "is_terminal", False):
            try:
                self.on_end(snap)
            except Exception:
                log.exception("manga job end handler failed")


def option_chip(text: str, status: str, *, key: Optional[str] = None, dark: bool = False) -> ft.Control:
    """Status chip: icon + text + colour (UI_SPEC §5.3), never colour alone."""
    from glossarion_mobile.ui.theme import status_color

    style_key = CHIP_STATUS.get(status, "info")
    color = status_color(style_key, dark)
    return ft.Container(
        content=ft.Row([ft.Icon(icon_data(tokens.status_style(style_key).icon), size=14, color=color),
                        ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=color)],
                       spacing=4, tight=True),
        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        border=ft.Border.all(1, color),
        border_radius=tokens.RADII["chip"],
        key=key,
    )


def reason_or_chip(row: Any, *, key: Optional[str] = None, dark: bool = False) -> ft.Control:
    """A disabled row's ReasonChip, else its status chip."""
    if getattr(row, "reason", None):
        from glossarion_mobile.services.manga import chip_text

        return ReasonChip(reason=str(row.chip or chip_text(row.reason)), detail=row.reason
                          + (f"\n\n{row.detail}" if getattr(row, "detail", "") else ""), key=key)
    return option_chip(str(getattr(row, "chip", "") or ""), str(getattr(row, "status", "info")), key=key, dark=dark)


def export_sheet(ctx: Any, path: str, *, title: Optional[str] = None) -> Optional[ActionSheet]:
    """FileBridge export options for ``path`` (Share… · Save to… · Save to Downloads · Show in Files)."""
    files = ctx.files
    if files is None:
        ctx.say("Exporting is not available")
        return None

    async def run(option_id: str) -> Any:
        try:
            result = await files.export(option_id, path)
        except Exception as exc:
            ctx.say(f"Export failed: {exc}")
            return None
        location = getattr(result, "location", None)
        if getattr(result, "needs_confirm", False):
            from glossarion_mobile.ui.tools.common import ask

            answer = await ask(ctx, "Large file", f"{os.path.basename(path)} is larger than 200 MB. Save it anyway?",
                               (("no", "Cancel", "text"), ("yes", "Save", "filled")), key="manga-save-confirm")
            if answer == "yes":
                result = await files.save_as(path, confirmed=True)
                location = getattr(result, "location", None)
        if location:
            ctx.say(f"Saved {os.path.basename(path)}")
        elif option_id == "downloads" and isinstance(result, str):
            ctx.say(f"Saved to Downloads/Glossarion: {os.path.basename(path)}")
        elif getattr(result, "error", None):
            ctx.say(f"Saving failed: {result.error}")
        return result

    items = [ActionItem(option.label, (lambda oid=option.id: run(oid)), icon=option.icon,
                        disabled_reason=option.disabled_reason, key=f"manga-export-{option.id}")
             for option in files.export_options(path)]
    sheet = ActionSheet(items, title=title or os.path.basename(path), tablet=bool(getattr(ctx, "tablet", False)))
    ctx.extras["manga_last_sheet"] = sheet
    if ctx.page is not None:
        sheet.show(ctx.page)
    return sheet


class MangaTab:
    """A tab of the manga screen: ``build()`` once, ``did_show()`` when selected,
    ``on_session_loaded()`` after the persisted selection is back, ``refresh()`` on outside
    changes, ``dispose()`` when the screen leaves."""

    key = "manga-tab"

    def __init__(self, ctx: Any, session: Any, *, screen: Any = None) -> None:
        self.ctx = ctx
        self.session = session
        self.screen = screen
        self.root: Optional[ft.Control] = None

    def build(self) -> ft.Control:
        raise NotImplementedError

    def did_show(self) -> None:
        pass

    def on_session_loaded(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    def section(self, title: str, controls: list, *, icon: Any = None, key: Optional[str] = None,
                subtitle: Optional[str] = None, trailing: Optional[ft.Control] = None) -> ft.Container:
        from glossarion_mobile.ui.tools.common import card

        return card(title, controls, icon=icon, key=key, subtitle=subtitle, trailing=trailing)



class TranslationModel:
    """The manga translator's own model + API key (``services.manga`` PREF_MODEL / PREF_API_KEY in
    Prefs, the key ``ENC:``), remembered between runs: the shared ``ModelPicker`` (the ModelSheet in
    field mode) and the KeyEditor's ``KeyField`` (Paste · Copy · Test). Empty = the main model /
    API key, which the field shows as its hint; "Use main" forgets both. Start and the editor steps
    send them with every job (``services.manga.translation_params``)."""

    def __init__(self, ctx: Any) -> None:
        from glossarion_mobile.services import manga as svc
        from glossarion_mobile.ui.screens.key_editor import KeyField
        from glossarion_mobile.ui.sheets.model_sheet import ModelPicker, sheet_env

        self.ctx = ctx
        self.prefs = getattr(ctx, "prefs", None)
        env = sheet_env()
        self.env = env
        model = svc.translation_model(self.prefs)
        self.main_model = str(ctx.cfg("model", "") or "")
        self.picker = ModelPicker(value=model or self.main_model, label="Model", env=env,
                                  on_change=self._on_model, key="mf-translation-model")
        key = svc.translation_api_key(self.prefs)
        self.key_field = KeyField(
            value=key, label="API key", on_change=self._on_key,
            hint="Empty: the main API key",
            on_test=(lambda value: env.test_key(value or str(ctx.cfg("api_key", "") or ""), self.model))
            if env.test_key is not None else None,
            read_clipboard=env.read_clipboard, copy_text=env.copy_text)
        self.reset_button = ft.TextButton(content="Use main model and key", icon=ft.Icons.RESTART_ALT,
                                          on_click=lambda e: self.use_main(), key="mf-translation-main")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            key="mf-translation-note")
        self._render_note()

    @property
    def model(self) -> str:
        return str(self.picker.value or "") or self.main_model

    def controls(self) -> list:
        return [self.picker, self.key_field, ft.Row([self.reset_button], wrap=True), self.note]

    def _render_note(self) -> None:
        from glossarion_mobile.services import manga as svc

        own_model = bool(svc.translation_model(self.prefs))
        own_key = bool(svc.translation_key_token(self.prefs))
        if own_model or own_key:
            parts = [("its own model" if own_model else "the main model"),
                     ("its own API key" if own_key else "the main API key")]
            self.note.value = f"The manga translator uses {parts[0]} and {parts[1]}."
        else:
            self.note.value = "The manga translator uses the main model and API key until you pick its own."
        self.reset_button.disabled = not (own_model or own_key)

    def _on_model(self, value: str) -> None:
        from glossarion_mobile.services import manga as svc

        try:
            svc.save_translation_model(self.prefs, value)
        except Exception as exc:
            self.ctx.say(f"Could not save the model: {exc}")
        self._render_note()
        push(self.note, self.reset_button)

    def _on_key(self, value: str) -> None:
        from glossarion_mobile.services import manga as svc

        try:
            svc.save_translation_api_key(self.prefs, value)
        except Exception as exc:  # ShareError: this phone cannot encrypt; nothing stored in plain text
            message = str(getattr(exc, "message", "") or exc)
            if message != getattr(self, "_last_error", None):  # once, not on every keystroke
                self._last_error = message
                self.ctx.say(message)
            return
        self._render_note()
        push(self.note, self.reset_button)

    def use_main(self) -> None:
        from glossarion_mobile.services import manga as svc

        try:
            svc.save_translation_model(self.prefs, "")
            svc.save_translation_api_key(self.prefs, "")
        except Exception as exc:
            self.ctx.say(f"Could not reset: {exc}")
        self.main_model = str(self.ctx.cfg("model", "") or "")
        self.picker.value = self.main_model
        self.picker.field.value = self.main_model
        self.key_field.set_value("", notify=False)
        self._render_note()
        push(self.picker.field, self.note, self.reset_button)

    def params(self) -> dict:
        from glossarion_mobile.services import manga as svc

        return svc.translation_params(self.prefs)
