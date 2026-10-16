"""Model manager › Ollama: Ollama in Termux on this phone, pulls and editable model presets.

The tab body of ``ModelManagerScreen`` (``tab == "ollama"``), over ``services.ollama_termux``:

* **Ollama server**: the address the ``ollamapull/`` route uses (``state.remote_routes``; this phone's
  Termux at ``http://127.0.0.1:11434`` or a PC), "Check" (``/api/version`` + ``/api/tags`` off the loop),
  "Set up in Termux" (one tap through Termux's RUN_COMMAND, else the command is copied and Termux opened)
  and "Use this phone" (points ``ollamapull/`` back at Termux).
* **Termux steps**: the manual commands, each with Copy, plus Open Termux / Get Termux.
* **Download a model**: any Ollama library tag or Hugging Face GGUF id (``hf.co/<user>/<repo>[:<quant>]``,
  validated by the shared ``ollamapull.normalize_pull_name``), pulled by the shared
  ``ollamapull.pull_model`` with live progress and Stop.
* **Presets**: named entries (data in Prefs, ``ollama_termux.PREF_KEY``) the user renames, edits, removes,
  adds and restores; each can be downloaded and used as the model (``ollamapull/<model>``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services import ollama_termux as ot
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["OllamaPanel", "OllamaPresetEditor"]

log = logging.getLogger("glossarion.models")

PULL_HINT = "qwen3.5:2b · hf.co/<user>/<repo>:Q4_K_M"


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


def _small(text: str, **kwargs: Any) -> ft.Text:
    kwargs.setdefault("color", ft.Colors.ON_SURFACE_VARIANT)
    return ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, **kwargs)


class OllamaPresetEditor:
    """Add / edit one preset (name · model · note), validated like the pull field."""

    def __init__(self, *, preset: Optional[dict] = None, on_save: Callable[[dict], Optional[str]]) -> None:
        preset = dict(preset or {})
        self.on_save = on_save
        self.saved = False
        self._page: Any = None
        self.name = ft.TextField(value=str(preset.get("name", "")), label="Name", hint_text="Light", dense=True,
                                 key="ollama-preset-name")
        self.model = ft.TextField(value=str(preset.get("model", "")), label="Ollama tag or Hugging Face id",
                                  hint_text=PULL_HINT, dense=True, key="ollama-preset-model")
        self.note = ft.TextField(value=str(preset.get("note", "")), label="Note (optional)", dense=True,
                                 key="ollama-preset-note")
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, selectable=True)
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=16), content=ft.Column([
                ft.Text("Edit preset" if preset else "New preset", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                        weight=ft.FontWeight.W_600),
                self.name, self.model, self.note, self.error,
                ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close()),
                        ft.FilledButton(content="Save", key="ollama-preset-save", on_click=lambda e: self.save())],
                       alignment=ft.MainAxisAlignment.END, wrap=True),
            ], tight=True, spacing=10)),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def row(self) -> dict:
        return {"name": (self.name.value or "").strip(), "model": (self.model.value or "").strip(),
                "note": (self.note.value or "").strip()}

    def save(self) -> Optional[str]:
        if self.saved:  # a second tap while the sheet closes
            return None
        error = self.on_save(self.row())
        if error:
            self.error.value = error
            self.error.visible = True
            _push(self.error)
            return error
        self.saved = True
        self.close()
        return None

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)


class OllamaPanel:
    """The Ollama tab. ``build()`` returns its control; nothing runs until ``did_show()``."""

    def __init__(self, *, prefs: Any = None, native: Any = None, page: Any = None,
                 copy_text: Optional[Callable[[str], Any]] = None, use_model: Optional[Callable[[str], Any]] = None,
                 add_model: Optional[Callable[[str], Any]] = None, notify: Optional[Callable[..., Any]] = None,
                 spawn: Optional[Callable[[Any], Any]] = None, run_io: Optional[Callable[..., Any]] = None,
                 launch_url: Optional[Callable[[str], Any]] = None, pulls: Optional[ot.PullManager] = None,
                 probe: Optional[Callable[[], dict]] = None) -> None:
        self.prefs = prefs
        self.native = native
        self.page = page
        self.copy_text = copy_text
        self.use_model = use_model
        self.add_model = add_model
        self.notify = notify
        self.spawn_fn = spawn
        self.run_io = run_io
        self.launch_url = launch_url
        self.probe = probe
        self.pulls = pulls or ot.PullManager(run_io=run_io)
        self._unsub = self.pulls.subscribe(self._on_pull)
        self.server: dict = {}
        self.download_rows: dict = {}
        self.editor: Optional[OllamaPresetEditor] = None
        self.last_dialog: Any = None
        self.last_outcome: Optional[ot.TermuxOutcome] = None
        self.control: Optional[ft.Control] = None
        self._checked = False

    # ---- helpers ----------------------------------------------------------------------------------

    def say(self, message: str) -> None:
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:
                pass

    def spawn(self, coro: Any) -> Any:
        return self.spawn_fn(coro) if self.spawn_fn is not None else asyncio.ensure_future(coro)

    def presets(self) -> list:
        return ot.load_presets(self.prefs)

    def address(self) -> str:
        from glossarion_mobile.state import remote_routes

        return remote_routes.saved_url(self.prefs, "ollamapull") or ot.TERMUX_URL

    def on_phone(self) -> bool:
        return ot.is_loopback(self.address())

    # ---- build ------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.address_text = _small("", key="ollama-address", selectable=True)
        self.status_text = ft.Text("Not checked yet", theme_style=ft.TextThemeStyle.BODY_SMALL, key="ollama-status",
                                   color=ft.Colors.ON_SURFACE_VARIANT, selectable=True)
        self.termux_text = _small("", visible=False, key="ollama-termux-message", selectable=True)
        self.check_button = ft.TextButton(content="Check", icon=ft.Icons.REFRESH, key="ollama-check",
                                          on_click=lambda e: self.spawn(self.check_server()))
        self.setup_button = ft.FilledTonalButton(content="Set up in Termux", icon=ft.Icons.TERMINAL,
                                                 key="ollama-setup", on_click=lambda e: self.spawn(self.setup_termux()))
        self.phone_button = ft.TextButton(content="Use this phone", icon=ft.Icons.PHONE_ANDROID,
                                          key="ollama-use-phone", on_click=lambda e: self.use_phone_server())
        server_card = SectionCard(title="🦙 Ollama server", key="ollama-server", children=[
            _small("Run Ollama on this phone with Termux (or use your PC's Ollama from Local AI). "
                   "Models then appear as ollamapull/<name>."),
            self.address_text, self.status_text,
            ft.Row([self.setup_button, self.check_button, self.phone_button], wrap=True, spacing=6, run_spacing=4),
            self.termux_text,
        ])

        steps: list = []
        for number, (title, command) in enumerate(ot.manual_steps(), start=1):
            steps.append(ft.Text(f"{number}. {title}", theme_style=ft.TextThemeStyle.BODY_SMALL))
            if command:
                steps.append(self._command_row(command))
        steps.append(ft.Row([
            ft.TextButton(content="Open Termux", icon=ft.Icons.OPEN_IN_NEW, key="ollama-open-termux",
                          on_click=lambda e: self.spawn(self.open_termux())),
            ft.TextButton(content="Get Termux (F-Droid)", icon=ft.Icons.DOWNLOAD, key="ollama-get-termux",
                          on_click=lambda e: self.open_url(ot.TERMUX_FDROID_URL)),
        ], wrap=True, spacing=6))
        steps_card = SectionCard(title="Termux steps", collapsible=True, expanded=False, key="ollama-steps",
                                 subtitle="Do it by hand, or enable one-tap setup", children=steps)

        self.pull_field = ft.TextField(label="Ollama tag or Hugging Face id", hint_text=PULL_HINT, dense=True,
                                       key="ollama-pull-field", border_radius=tokens.RADII["field"],
                                       on_submit=lambda e: self.spawn(self.pull_typed()))
        self.downloads = ft.Column([], spacing=8, tight=True, key="ollama-downloads")
        pull_card = SectionCard(title="Download a model", key="ollama-pull", children=[
            self.pull_field,
            ft.Row([ft.FilledTonalButton(content="Download", icon=ft.Icons.DOWNLOAD, key="ollama-pull-button",
                                         on_click=lambda e: self.spawn(self.pull_typed()))], wrap=True),
            self.downloads,
        ])

        self.preset_column = ft.Column([], spacing=6, tight=True, key="ollama-presets")
        presets_card = SectionCard(title="Presets", key="ollama-presets-card",
                                   subtitle="Your names for local models. Tap one to rename or edit it.", children=[
            self.preset_column,
            ft.Row([ft.TextButton(content="Add preset", icon=ft.Icons.ADD, key="ollama-add-preset",
                                  on_click=lambda e: self.open_editor(None)),
                    ft.TextButton(content="Restore defaults", icon=ft.Icons.RESTART_ALT, key="ollama-restore",
                                  on_click=lambda e: self.confirm_restore())], wrap=True, spacing=6),
        ])
        self.control = ft.ListView(controls=[server_card, presets_card, pull_card, steps_card], expand=True,
                                   spacing=tokens.SPACING["sm"], key="ollama-panel",
                                   padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"],
                                                                vertical=tokens.SPACING["sm"]))
        self._sync_server()
        self.render_presets(push=False)
        for state in list(self.pulls.states.values()):  # downloads that outlived an earlier visit
            self._on_pull(state)
        return self.control

    def _command_row(self, command: str) -> ft.Control:
        return ft.Container(
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8,
            padding=ft.Padding.only(left=10, top=2, bottom=2, right=0),
            content=ft.Row([
                ft.Text(command, font_family="monospace", size=12, selectable=True, expand=True),
                ft.IconButton(icon=ft.Icons.COPY, tooltip="Copy", size_constraints=HIT_TARGET,
                              on_click=lambda e, c=command: self.spawn(self.copy(c))),
            ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def did_show(self) -> None:
        if self._unsub is None:
            self._unsub = self.pulls.subscribe(self._on_pull)
        if not self._checked:
            self._checked = True
            self.spawn(self.check_server())

    def dispose(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    # ---- server -----------------------------------------------------------------------------------

    def _sync_server(self) -> None:
        where = "this phone (Termux)" if self.on_phone() else "your PC"
        self.address_text.value = f"Address: {self.address()} · {where}"
        self.phone_button.visible = not self.on_phone()

    async def check_server(self) -> dict:
        self._sync_server()
        self.status_text.value = "Checking…"
        self.status_text.color = ft.Colors.ON_SURFACE_VARIANT
        _push(self.address_text, self.status_text, self.phone_button)
        self.server = await ot.server_status(run_io=self.run_io, probe=self.probe)
        if self.server.get("running"):
            count = len(self.server.get("models") or [])
            self.status_text.value = (f"✓ Ollama {self.server.get('version')} is running · "
                                      f"{count} model{'' if count == 1 else 's'} downloaded")
            self.status_text.color = semantic("success")
        else:
            hint = " Tap “Set up in Termux”." if self.on_phone() else ""
            self.status_text.value = f"✗ {self.server.get('error') or 'Ollama is not running'}.{hint}"
            self.status_text.color = ft.Colors.ERROR
        _push(self.status_text)
        self.render_presets()
        return self.server

    def use_phone_server(self) -> str:
        from glossarion_mobile.state import remote_routes

        url = remote_routes.apply_url(self.prefs, "ollamapull", ot.TERMUX_URL, save=True)
        self._sync_server()
        _push(self.address_text, self.phone_button)
        self.say("ollamapull/ now uses Ollama in Termux on this phone")
        self.spawn(self.check_server())
        return url

    async def setup_termux(self) -> ot.TermuxOutcome:
        if not self.on_phone():
            self.use_phone_server()
        outcome = await ot.run_in_termux(self.native, ot.SETUP_COMMAND, label="Ollama server",
                                         copy_text=self.copy_text,
                                         fallback_command=ot.FIRST_RUN_SETUP_COMMAND)
        self.last_outcome = outcome
        self.termux_text.value = outcome.message
        self.termux_text.color = ft.Colors.ERROR if outcome.mode == "missing" else ft.Colors.ON_SURFACE_VARIANT
        self.termux_text.visible = True
        _push(self.termux_text)
        return outcome

    async def open_termux(self) -> bool:
        opened = False
        if self.native is not None:
            try:
                opened = bool(await self.native.open_termux())
            except Exception:
                opened = False
        if not opened:
            self.say("Termux is not installed (or cannot be opened from here)")
        return opened

    def open_url(self, url: str) -> None:
        if self.launch_url is not None:
            try:
                result = self.launch_url(url)
                if asyncio.iscoroutine(result):
                    self.spawn(result)
                return
            except Exception as exc:
                log.info("launch_url failed: %s", exc)
        self.spawn(self.copy(url))

    async def copy(self, text: str) -> None:
        if self.copy_text is None:
            return
        result = self.copy_text(text)
        if asyncio.iscoroutine(result):
            await result

    # ---- pulls ------------------------------------------------------------------------------------

    async def pull_typed(self) -> Optional[ot.PullState]:
        model, error = ot.normalize_model(self.pull_field.value or "")
        self.pull_field.error = error
        if model:
            self.pull_field.value = model
        _push(self.pull_field)
        if error:
            return None
        return await self.pull(model)

    async def pull(self, model: str) -> ot.PullState:
        if self.pulls.running(model):
            return self.pulls.states[model]
        state = await self.pulls.pull(model)
        if state.done:
            self.say(f"Downloaded {model}")
            await self.check_server()
        elif state.error:
            self.say(f"Download failed: {state.error}")
        return state

    def cancel(self, model: str) -> bool:
        return self.pulls.cancel(model)

    def _on_pull(self, state: ot.PullState) -> None:
        if self.control is None:
            return
        row = self.download_rows.get(state.model)
        if row is None:
            bar = ft.ProgressBar(value=None, border_radius=4)
            text = _small(state.status, selectable=True)
            stop = ft.IconButton(icon=ft.Icons.STOP_CIRCLE_OUTLINED, tooltip="Stop", size_constraints=HIT_TARGET,
                                 on_click=lambda e, m=state.model: self.cancel(m))
            box = ft.Container(key=f"ollama-download-{state.model}", content=ft.Column([
                ft.Row([ft.Text(state.model, weight=ft.FontWeight.W_600, expand=True, max_lines=2,
                                overflow=ft.TextOverflow.ELLIPSIS), stop], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bar, text,
            ], spacing=4, tight=True))
            row = (box, bar, text, stop)
            self.download_rows[state.model] = row
            self.downloads.controls.insert(0, box)
            _push(self.downloads)
        box, bar, text, stop = row
        bar.value = state.fraction if state.running else (1.0 if state.done else 0.0)
        bar.color = ft.Colors.ERROR if state.error else None
        text.value = state.status
        text.color = ft.Colors.ERROR if state.error else ft.Colors.ON_SURFACE_VARIANT
        stop.visible = state.running
        _push(box)
        if not state.running:
            self.render_presets()

    # ---- presets ----------------------------------------------------------------------------------

    def render_presets(self, push: bool = True) -> None:
        if self.control is None:
            return
        names = self.server.get("models") or []
        rows: list = []
        presets = self.presets()
        if not presets:
            rows.append(_small("No presets. Add one, or restore the defaults.", key="ollama-presets-empty"))
        for index, preset in enumerate(presets):
            rows.append(self._preset_row(index, preset, ot.installed(preset["model"], names)))
        self.preset_column.controls = rows
        if push:
            _push(self.preset_column)

    def _preset_row(self, index: int, preset: dict, have: bool) -> ft.Control:
        model = preset["model"]
        pulling = self.pulls.running(model)
        lines: list = [
            ft.Text(preset["name"], weight=ft.FontWeight.W_600, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(model, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if preset.get("note"):
            lines.append(_small(preset["note"], max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        if have:
            download: ft.Control = ft.Container(
                content=ft.Icon(ft.Icons.CHECK_CIRCLE, color=semantic("success"), tooltip="Downloaded"),
                width=48, height=48, alignment=ft.Alignment.CENTER, key=f"ollama-preset-have-{index}")
        else:
            download = ft.IconButton(icon=ft.Icons.DOWNLOADING if pulling else ft.Icons.DOWNLOAD,
                                     tooltip="Downloading…" if pulling else "Download", disabled=pulling,
                                     size_constraints=HIT_TARGET, key=f"ollama-preset-pull-{index}",
                                     on_click=lambda e, m=model: self.spawn(self.pull(m)))
        menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", key=f"ollama-preset-menu-{index}", items=[
            ft.PopupMenuItem(content="Use as model", icon=ft.Icons.CHECK, on_click=lambda e, i=index: self.use_preset(i)),
            ft.PopupMenuItem(content="Edit / rename", icon=ft.Icons.EDIT_OUTLINED,
                             on_click=lambda e, i=index: self.open_editor(i)),
            ft.PopupMenuItem(content="Copy pull command", icon=ft.Icons.COPY,
                             on_click=lambda e, m=model: self.spawn(self.copy(ot.pull_command(m)))),
            ft.PopupMenuItem(content="Remove", icon=ft.Icons.DELETE_OUTLINE,
                             on_click=lambda e, i=index: self.confirm_remove(i)),
        ])
        return ft.Container(
            key=f"ollama-preset-{index}", border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.only(left=12, top=6, bottom=6, right=0), on_click=lambda e, i=index: self.open_editor(i),
            content=ft.Row([ft.Column(lines, spacing=0, tight=True, expand=True), download, menu], spacing=0,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def use_preset(self, index: int) -> Optional[str]:
        presets = self.presets()
        if not 0 <= index < len(presets):
            return None
        model = ot.route_model(presets[index]["model"])
        if self.add_model is not None:
            try:
                self.add_model(model)  # in the model list / pickers too; "already in the list" is fine
            except Exception:
                pass
        if self.use_model is not None:
            try:
                self.use_model(model)
            except Exception as exc:
                self.say(f"Could not switch the model: {exc}")
                return None
        self.say(f"Model: {model} ({presets[index]['name']})")
        return model

    def open_editor(self, index: Optional[int]) -> OllamaPresetEditor:
        presets = self.presets()
        current = presets[index] if index is not None and 0 <= index < len(presets) else None
        editor = OllamaPresetEditor(preset=current, on_save=lambda row: self.save_preset(index, row))
        self.editor = editor
        if self.page is not None:
            editor.show(self.page)
        return editor

    def save_preset(self, index: Optional[int], row: dict) -> Optional[str]:
        presets = self.presets()
        editing = index if index is not None and 0 <= index < len(presets) else None
        clean, error = ot.validate_preset(row, presets, editing)
        if error:
            return error
        if editing is not None:
            presets[editing] = clean
        else:
            if len(presets) >= ot.MAX_PRESETS:
                return f"Keep it to {ot.MAX_PRESETS} presets"
            presets.append(clean)
        ot.save_presets(self.prefs, presets)
        self.render_presets()
        return None

    def remove_preset(self, index: int) -> bool:
        presets = self.presets()
        if not 0 <= index < len(presets):
            return False
        presets.pop(index)
        ot.save_presets(self.prefs, presets)
        self.render_presets()
        return True

    def confirm_remove(self, index: int) -> ConfirmDialog:
        presets = self.presets()
        name = presets[index]["name"] if 0 <= index < len(presets) else ""
        dialog = ConfirmDialog(title="Remove preset", body=f"Remove the preset “{name}”? The downloaded model stays "
                                                             "in Ollama.", confirm_label="Remove", destructive=True,
                               on_confirm=lambda: self.remove_preset(index))
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def confirm_restore(self) -> ConfirmDialog:
        def restore() -> None:
            ot.restore_defaults(self.prefs)
            self.render_presets()

        defaults = ", ".join(f"{p['name']} ({p['model']})" for p in ot.DEFAULT_PRESETS)
        dialog = ConfirmDialog(title="Restore default presets",
                               body=f"Replace your presets with the defaults: {defaults}?", confirm_label="Restore",
                               destructive=True, on_confirm=restore)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog
