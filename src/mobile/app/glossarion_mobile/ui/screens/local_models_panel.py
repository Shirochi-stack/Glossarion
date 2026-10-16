"""Model manager › On-device: the built-in llama.cpp engine, its models and the editable presets.

The tab body of ``ModelManagerScreen`` (``tab == "local"``), over ``services.local_llm``:

* **Engine**: Not loaded / Loading… / Ready · <model> / Generating, Unload, the Context and Threads
  defaults (Prefs ``local_llm_settings``) and the engine's system info after the first load. Without
  llama-cpp-python (a desktop dev run) the card says the engine is unavailable; nothing crashes.
* **Presets** (data in Prefs, ``presets.PREF_KEY``; Qwen3.5 0.8B/2B/4B/9B defaults): ONE button per
  model. Download (one tap: Hugging Face file, download with progress, load, use as the model) /
  progress with Pause and Cancel / Resume / Use / In use / Retry. The ⋮ menu edits, removes the preset
  or deletes the file; Add preset and Restore defaults close the card.
* **Add from Hugging Face**: any repo, repo:quant or huggingface.co link; the usual quant is picked
  (else a file chooser), optionally saved as a preset.
* **Downloaded models**: files no preset covers, and paused downloads (Resume / Cancel).

Models are ``local/<stem>`` (the GGUF file name), routed by the shared ``local/`` route to the in-app
endpoint, so the chosen model works for chat, translation, glossary and every other job.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["FileChooser", "LocalModelsPanel", "LocalPresetEditor"]

log = logging.getLogger("glossarion.models")

HF_HINT = "unsloth/Qwen3.5-2B-GGUF · …/resolve/main/x.gguf"
STATE_TEXT = {"idle": "Not loaded", "loading": "Loading…", "ready": "Ready", "generating": "Generating…"}


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


def _preset_ref(preset: dict) -> str:
    return f"{preset.get('repo')}/{preset['file']}" if preset.get("file") else (
        f"{preset.get('repo')}:{preset['quant']}" if preset.get("quant") else str(preset.get("repo") or ""))


class LocalPresetEditor:
    """Add / edit one preset (name · Hugging Face model · file · note · context)."""

    def __init__(self, *, preset: Optional[dict] = None, on_save: Callable[[dict], Optional[str]]) -> None:
        preset = dict(preset or {})
        self.on_save = on_save
        self.saved = False
        self._page: Any = None
        repo = str(preset.get("repo", ""))
        if preset.get("quant") and not preset.get("file"):
            repo = f"{repo}:{preset['quant']}"
        self.name = ft.TextField(value=str(preset.get("name", "")), label="Name", hint_text="Light", dense=True,
                                 key="local-preset-name")
        self.repo = ft.TextField(value=repo, label="Hugging Face model (repo, repo:quant or URL)", hint_text=HF_HINT,
                                 dense=True, key="local-preset-repo")
        self.file = ft.TextField(value=str(preset.get("file", "")), label="File (optional)",
                                 hint_text="Qwen3.5-2B-Q4_K_M.gguf", dense=True, key="local-preset-file")
        self.note = ft.TextField(value=str(preset.get("note", "")), label="Note (optional)", dense=True,
                                 key="local-preset-note")
        self.n_ctx = ft.TextField(value=str(preset.get("n_ctx") or ""), label="Context (tokens, optional)",
                                  hint_text="8192", dense=True, keyboard_type=ft.KeyboardType.NUMBER,
                                  key="local-preset-ctx")
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, selectable=True)
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=16), content=ft.Column([
                ft.Text("Edit preset" if preset else "New preset", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                        weight=ft.FontWeight.W_600),
                self.name, self.repo, self.file, self.note, self.n_ctx, self.error,
                ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close()),
                        ft.FilledButton(content="Save", key="local-preset-save", on_click=lambda e: self.save())],
                       alignment=ft.MainAxisAlignment.END, wrap=True),
            ], tight=True, spacing=10)),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def row(self) -> dict:
        return {"name": (self.name.value or "").strip(), "repo": (self.repo.value or "").strip(),
                "file": (self.file.value or "").strip(), "note": (self.note.value or "").strip(),
                "n_ctx": (self.n_ctx.value or "").strip()}

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


class FileChooser:
    """Pick one GGUF file of a repository (name and size); ``future`` gets the HfFile or None."""

    def __init__(self, repo: str, files: list) -> None:
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._page: Any = None
        rows = [ft.ListTile(title=ft.Text(f.name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                            subtitle=ft.Text(dl.format_bytes(f.size)), dense=True, key=f"local-file-{i}",
                            on_click=lambda e, f=f: self.pick(f)) for i, f in enumerate(files)]
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=8, right=8, bottom=16), content=ft.Column([
                ft.Text(f"Choose a file of {repo}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                        weight=ft.FontWeight.W_600),
                *rows,
                ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.pick(None))],
                       alignment=ft.MainAxisAlignment.END),
            ], tight=True, spacing=4, scroll=ft.ScrollMode.AUTO)),
            show_drag_handle=True, scrollable=True, on_dismiss=lambda e: self.pick(None),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def pick(self, file: Optional[hf.HfFile]) -> None:
        if not self.future.done():
            self.future.set_result(file)
        close_dialog(self._page, self.dialog)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)


class LocalModelsPanel:
    """The On-device tab. ``build()`` returns its control; nothing runs until ``did_show()``.

    ``service``: the app's ``LocalEngineService`` (downloads outlive this screen). ``confirm(title, body,
    label)`` and ``choose(repo, files)`` (tests) replace the dialogs."""

    def __init__(self, *, service: Any, prefs: Any = None, page: Any = None,
                 use_model: Optional[Callable[[str], Any]] = None, add_model: Optional[Callable[[str], Any]] = None,
                 current_model: Optional[Callable[[], str]] = None, notify: Optional[Callable[..., Any]] = None,
                 spawn: Optional[Callable[[Any], Any]] = None,
                 confirm: Optional[Callable[..., Any]] = None, choose: Optional[Callable[..., Any]] = None) -> None:
        self.service = service
        self.prefs = prefs
        self.page = page
        self.use_model = use_model
        self.add_model = add_model
        self.current_model = current_model
        self.notify = notify
        self.spawn_fn = spawn
        self.confirm_fn = confirm
        self.choose_fn = choose
        self.downloads: dl.DownloadManager = service.downloads
        self._unsubs: list = []
        self.errors: dict = {}  # preset key -> last error (the row shows Retry)
        self.preset_stems: dict = {}  # preset key -> stem of its download (file-less presets)
        self.busy: set = set()  # preset keys whose one-tap flow runs
        self.editor: Optional[LocalPresetEditor] = None
        self.last_dialog: Any = None
        self.control: Optional[ft.Control] = None

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
        return presets.load_presets(self.prefs)

    @staticmethod
    def preset_key(preset: dict) -> str:
        return f"{str(preset.get('repo') or '').casefold()}|{preset.get('file') or preset.get('quant') or ''}"

    def stem_of_preset(self, preset: dict) -> Optional[str]:
        """The downloaded (or downloading) stem of ``preset``."""
        registry = self.service.registry
        if preset.get("file"):
            found = registry.find(preset["repo"], preset["file"])
            if found:
                return found
        stem = self.preset_stems.get(self.preset_key(preset))
        if stem:
            return stem
        if preset.get("file"):
            for stem, state in self.downloads.states.items():
                if state.repo.casefold() == str(preset["repo"]).casefold() and state.file == preset["file"]:
                    return stem
        else:
            quant = str(preset.get("quant") or "").lower()
            for stem, entry in registry.load().items():
                if str(entry.get("repo", "")).casefold() == str(preset["repo"]).casefold() and \
                        (not quant or quant in str(entry.get("file", "")).lower()):
                    return stem
        return None

    async def ask(self, title: str, body: str, label: str) -> bool:
        if self.confirm_fn is not None:
            result = self.confirm_fn(title, body, label)
            return bool(await result) if asyncio.iscoroutine(result) else bool(result)
        if self.page is None:
            return True
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        dialog = ConfirmDialog(title=title, body=body, confirm_label=label,
                               on_confirm=lambda: future.done() or future.set_result(True),
                               on_cancel=lambda: future.done() or future.set_result(False))
        self.last_dialog = dialog
        dialog.show(self.page)
        return bool(await future)

    async def choose(self, repo: str, files: list) -> Optional[hf.HfFile]:
        if self.choose_fn is not None:
            result = self.choose_fn(repo, files)
            return await result if asyncio.iscoroutine(result) else result
        if self.page is None:
            return None
        chooser = FileChooser(repo, files)
        self.last_dialog = chooser
        chooser.show(self.page)
        return await chooser.future

    # ---- build ------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        available, reason = self.service.engine_status()
        self.engine_state = ft.Text("Not loaded", weight=ft.FontWeight.W_600, key="local-engine-state",
                                    selectable=True)
        self.engine_detail = _small("", key="local-engine-detail", selectable=True, visible=False)
        self.unavailable_text = ft.Text(reason or "", color=ft.Colors.ERROR, visible=not available,
                                        theme_style=ft.TextThemeStyle.BODY_SMALL, key="local-engine-unavailable",
                                        selectable=True)
        self.unload_button = ft.TextButton(content="Unload", icon=ft.Icons.EJECT, key="local-unload",
                                           on_click=lambda e: self.spawn(self.unload()))
        settings = self.service.settings()
        self.ctx_field = ft.TextField(value=str(settings["n_ctx"] or ""), label="Context", hint_text="auto",
                                      dense=True, expand=True, keyboard_type=ft.KeyboardType.NUMBER, key="local-ctx")
        self.threads_field = ft.TextField(value=str(settings["n_threads"] or ""), label="Threads", hint_text="auto",
                                          dense=True, expand=True, keyboard_type=ft.KeyboardType.NUMBER,
                                          key="local-threads")
        self.settings_error = ft.Text("", color=ft.Colors.ERROR, visible=False, theme_style=ft.TextThemeStyle.BODY_SMALL)
        engine_card = SectionCard(title="🧠 On-device engine", key="local-engine", children=[
            _small("Models run inside Glossarion with llama.cpp: no server, no account. They show up as "
                   "local/<name> for chat, translation and glossaries."),
            self.unavailable_text,
            ft.Row([ft.Container(content=self.engine_state, expand=True), self.unload_button],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
            self.engine_detail,
            ft.Row([self.ctx_field, self.threads_field,
                    ft.TextButton(content="Save", key="local-settings-save", on_click=lambda e: self.save_settings())],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.settings_error,
        ])
        self.preset_column = ft.Column([], spacing=6, tight=True, key="local-presets")
        presets_card = SectionCard(title="Models", key="local-presets-card",
                                   subtitle="One tap downloads a model from Hugging Face and uses it.", children=[
            self.preset_column,
            ft.Row([ft.TextButton(content="Add preset", icon=ft.Icons.ADD, key="local-add-preset",
                                  on_click=lambda e: self.open_editor(None)),
                    ft.TextButton(content="Restore defaults", icon=ft.Icons.RESTART_ALT, key="local-restore",
                                  on_click=lambda e: self.confirm_restore())], wrap=True, spacing=6),
        ])
        self.hf_field = ft.TextField(label="Hugging Face model or link", hint_text=HF_HINT, dense=True,
                                     key="local-download-field", border_radius=tokens.RADII["field"],
                                     on_submit=lambda e: self.spawn(self.download_typed()))
        self.save_as_preset = ft.Checkbox(label="Save as a preset", value=True, key="local-save-preset")
        self.hf_status = _small("", visible=False, key="local-hf-status", selectable=True)
        hf_card = SectionCard(title="Add from Hugging Face", key="local-hf", children=[
            self.hf_field,
            ft.Row([ft.FilledTonalButton(content="Download", icon=ft.Icons.DOWNLOAD, key="local-download-button",
                                         on_click=lambda e: self.spawn(self.download_typed())),
                    self.save_as_preset], wrap=True, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.hf_status,
        ])
        self.other_column = ft.Column([], spacing=6, tight=True, key="local-other")
        self.other_card = SectionCard(title="Other downloads", key="local-downloaded",
                                      subtitle="Models no preset names, and paused downloads", children=[self.other_column])
        self.control = ft.ListView(controls=[engine_card, presets_card, hf_card, self.other_card], expand=True,
                                   spacing=tokens.SPACING["sm"], key="local-panel",
                                   padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"],
                                                                vertical=tokens.SPACING["sm"]))
        self._subscribe()
        self.render_engine(self.service.snapshot(), push=False)
        self.render(push=False)
        return self.control

    def _subscribe(self) -> None:
        if not self._unsubs:
            self._unsubs = [self.downloads.subscribe(self._on_download), self.service.subscribe(self._on_engine)]

    def did_show(self) -> None:
        self._subscribe()
        self.downloads.scan()
        self.render()
        self.render_engine(self.service.snapshot())

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    # ---- engine card --------------------------------------------------------------------------------

    def _on_engine(self, snap: Any) -> None:
        if self.control is None:
            return
        self.render_engine(snap)
        self.render()

    def render_engine(self, snap: Any, push: bool = True) -> None:
        if self.control is None:
            return
        text = STATE_TEXT.get(snap.state, snap.state)
        if snap.model and snap.state != "idle":
            text = f"{text} · {snap.model}"
        if snap.queue:
            text += f" · {snap.queue} waiting"
        self.engine_state.value = text
        self.engine_state.color = semantic("success") if snap.state in ("ready", "generating") else None
        details = []
        if snap.load_seconds is not None and snap.model:
            details.append(f"Loaded in {snap.load_seconds:.1f} s")
        if snap.system_info:
            details.append(snap.system_info)
        if snap.error:
            details.append(snap.error)
        self.engine_detail.value = "\n".join(details)
        self.engine_detail.visible = bool(details)
        self.unload_button.disabled = snap.state not in ("ready",)
        self.unavailable_text.visible = not snap.available
        self.unavailable_text.value = snap.unavailable_reason or ""
        if push:
            _push(self.engine_state, self.engine_detail, self.unload_button, self.unavailable_text)

    def save_settings(self) -> Optional[str]:
        values, error = self.service.save_settings(self.ctx_field.value, self.threads_field.value)
        self.settings_error.value = error or ""
        self.settings_error.visible = bool(error)
        _push(self.settings_error)
        if not error:
            self.say("Saved. The next model load uses these settings.")
        return error

    async def unload(self) -> bool:
        done = await self.service._io(lambda: self.service.unload())
        if not done:
            self.say("The model is busy; it unloads when the answer is done")
        return bool(done)

    # ---- downloads ----------------------------------------------------------------------------------

    def _on_download(self, state: dl.DownloadState) -> None:
        if self.control is None:
            return
        self.render()

    async def fetch(self, ref: hf.HfRef, *, preset: Optional[dict] = None) -> Optional[str]:
        """Resolve ``ref`` on Hugging Face, ask for big files, download; the stem, or None (cancelled /
        failed: the row or status line says why)."""
        try:
            commit, files = await self.service.resolve(ref)
            file = hf.pick_file(files, quant=ref.quant, file=ref.file)
            if file is None:
                file = await self.choose(ref.repo, files)
                if file is None:
                    return None
                if file.split:
                    raise hf.HfError("Split GGUF files are not supported yet")
        except hf.HfError as exc:
            raise _Failed(str(exc)) from exc
        existing = self.service.registry.find(ref.repo, file.path)
        if existing:
            return existing
        spec = self.service.spec_for(ref.repo, commit, file)
        if preset is not None:
            self.preset_stems[self.preset_key(preset)] = spec.stem
        if file.size >= dl.CONFIRM_BYTES and not self.downloads.pending_spec(spec.stem):
            if not await self.ask("Download model", f"Download {dl.format_bytes(file.size)} from Hugging Face?\n\n"
                                                    f"{ref.repo} · {file.name}", "Download"):
                return None
        self.render()
        state = await self.service.download(spec)
        if state.error:
            raise _Failed(state.error)
        return spec.stem if state.done else None

    async def activate(self, stem: str, label: str = "") -> bool:
        """Load ``stem`` (asking when RAM is tight) and make ``local/<stem>`` the model."""
        verdict, message = self.service.ram_check(stem)
        if verdict == "refuse":
            raise _Failed(message)
        force = False
        if verdict == "confirm":
            if not await self.ask("Load model", message, "Load anyway"):
                return False
            force = True
        available, reason = self.service.engine_status()
        if available:
            ok, error = await self.service.load(stem, force=force)
            if not ok:
                raise _Failed(error or "Could not load the model")
        model = f"local/{stem}"
        if self.add_model is not None:
            try:
                self.add_model(model)
            except Exception:
                pass
        if self.use_model is not None:
            try:
                self.use_model(model)
            except Exception as exc:
                raise _Failed(f"Could not switch the model: {exc}") from exc
        suffix = "" if available else " (the engine is not available in this build)"
        self.say(f"Model: {model}{f' ({label})' if label else ''}{suffix}")
        self.render()
        return True

    async def get_preset(self, index: int) -> Optional[str]:
        """The one-tap action: download if needed, load, use. Returns the model id or None."""
        rows = self.presets()
        if not 0 <= index < len(rows):
            return None
        preset = rows[index]
        key = self.preset_key(preset)
        if key in self.busy:
            return None
        self.busy.add(key)
        self.errors.pop(key, None)
        self.render()
        try:
            stem = self.stem_of_preset(preset)
            if not stem or self.service.registry.resolve(stem) is None:
                if stem and self.downloads.running(stem):
                    return None
                ref = hf.HfRef(preset["repo"], preset.get("file") or None, quant=preset.get("quant") or None)
                stem = await self.fetch(ref, preset=preset)
                if not stem:
                    return None
            if await self.activate(stem, preset["name"]):
                return f"local/{stem}"
            return None
        except _Failed as exc:
            self.errors[key] = str(exc)
            self.say(str(exc))
            return None
        finally:
            self.busy.discard(key)
            self.render()

    async def download_typed(self) -> Optional[str]:
        text = (self.hf_field.value or "").strip()
        try:
            ref = hf.parse_ref(text)
        except hf.HfError as exc:
            self.hf_field.error = str(exc)
            _push(self.hf_field)
            return None
        self.hf_field.error = None
        self._status("Looking up the model on Hugging Face…")
        try:
            stem = await self.fetch(ref)
        except _Failed as exc:
            self._status(str(exc), error=True)
            return None
        if not stem:
            self._status("")
            return None
        entry = self.service.registry.load().get(stem) or {}
        if self.save_as_preset.value:
            rows = self.presets()
            if not any(r.get("repo", "").casefold() == ref.repo.casefold() and r.get("file") == entry.get("file")
                       for r in rows) and len(rows) < presets.MAX_PRESETS:
                name = stem
                taken = {str(r.get("name", "")).casefold() for r in rows}
                number = 2
                while name.casefold() in taken:
                    name = f"{stem} {number}"
                    number += 1
                rows.append({"name": name[:60], "repo": ref.repo, "file": entry.get("file", ""), "quant": "",
                             "note": f"{ref.repo}", "n_ctx": 0, "size": entry.get("size"), "sha256": entry.get("sha256")})
                presets.save_presets(self.prefs, rows)
        self._status(f"Downloaded local/{stem} ✓")
        try:
            await self.activate(stem)
        except _Failed as exc:
            self._status(str(exc), error=True)
        self.render()
        return stem

    def _status(self, text: str, *, error: bool = False) -> None:
        self.hf_status.value = text
        self.hf_status.visible = bool(text)
        self.hf_status.color = ft.Colors.ERROR if error else ft.Colors.ON_SURFACE_VARIANT
        _push(self.hf_status, self.hf_field)

    def pause(self, stem: str) -> bool:
        return self.downloads.pause(stem)

    def cancel(self, stem: str) -> bool:
        done = self.downloads.cancel(stem)
        self.render()
        return done

    async def resume(self, stem: str) -> Optional[str]:
        spec = self.downloads.pending_spec(stem)
        if spec is None:
            return None
        state = await self.service.download(spec)
        if state.error:
            self.say(f"Download failed: {state.error}")
        self.render()
        return stem if state.done else None

    def delete_file(self, stem: str) -> bool:
        removed = self.service.delete(stem)
        if removed:
            self.say(f"Deleted local/{stem}")
        self.render()
        return removed

    def confirm_delete(self, stem: str) -> ConfirmDialog:
        entry = self.service.registry.load().get(stem) or {}
        dialog = ConfirmDialog(title="Delete model file",
                               body=f"Delete local/{stem} ({dl.format_bytes(entry.get('size'))}) from this device?",
                               confirm_label="Delete", destructive=True, on_confirm=lambda: self.delete_file(stem))
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    # ---- rows ---------------------------------------------------------------------------------------

    def render(self, push: bool = True) -> None:
        if self.control is None:
            return
        rows: list = []
        listed: set = set()
        preset_rows = self.presets()
        if not preset_rows:
            rows.append(_small("No presets. Add one, or restore the defaults.", key="local-presets-empty"))
        for index, preset in enumerate(preset_rows):
            stem = self.stem_of_preset(preset)
            if stem:
                listed.add(stem)
            rows.append(self._preset_row(index, preset, stem))
        self.preset_column.controls = rows
        others: list = []
        for stem, entry in sorted(self.service.registry.load().items(), key=lambda kv: kv[0].casefold()):
            if stem not in listed:
                others.append(self._other_row(stem, entry))
                listed.add(stem)
        for stem, state in list(self.downloads.states.items()):
            if stem not in listed and (state.running or state.paused):
                others.append(self._download_row(stem, state))
        self.other_column.controls = others
        self.other_card.visible = bool(others)
        if push:
            _push(self.preset_column, self.other_card)

    def row_state(self, preset: dict, stem: Optional[str]) -> tuple:
        """(state, status text) of a preset row: download | downloading | paused | use | in_use | error | busy."""
        key = self.preset_key(preset)
        state = self.downloads.states.get(stem) if stem else None
        current = str(self.current_model() if self.current_model else "") or ""
        if state is not None and state.running:
            return "downloading", state.status
        if stem and self.service.registry.resolve(stem):
            loaded = self.service.loaded_stem == stem
            if current.lower() == f"local/{stem}".lower():
                return "in_use", "Loaded · in use" if loaded else "In use · loads on the first request"
            return "use", "Loaded" if loaded else "On this device"
        if state is not None and state.paused:
            return "paused", state.status
        if key in self.errors:
            return "error", f"Failed: {self.errors[key]}"
        if key in self.busy:
            return "busy", "Looking up the model on Hugging Face…"
        size = preset.get("size")
        return "download", f"Not downloaded · {dl.format_bytes(size)}" if size else "Not downloaded"

    def _preset_row(self, index: int, preset: dict, stem: Optional[str]) -> ft.Control:
        state, status = self.row_state(preset, stem)
        lines: list = [
            ft.Text(preset["name"], weight=ft.FontWeight.W_600, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(_preset_ref(preset), theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if preset.get("note"):
            lines.append(_small(preset["note"], max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        status_text = _small(status, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, key=f"local-preset-status-{index}")
        if state == "error":
            status_text.color = ft.Colors.ERROR
        elif state in ("use", "in_use"):
            status_text.color = semantic("success")
        lines.append(status_text)
        body: list = [ft.Row([ft.Column(lines, spacing=0, tight=True, expand=True)], spacing=0)]
        download = self.downloads.states.get(stem) if stem else None
        if state in ("downloading", "paused") and download is not None:
            body.append(ft.ProgressBar(value=download.fraction, border_radius=4, key=f"local-preset-progress-{index}"))
        trailing: list = [self._action(index, state, stem)]
        if state == "downloading" and stem:
            trailing.append(ft.IconButton(icon=ft.Icons.PAUSE, tooltip="Pause", size_constraints=HIT_TARGET,
                                          key=f"local-preset-pause-{index}", on_click=lambda e, s=stem: self.pause(s)))
        if state in ("downloading", "paused") and stem:
            trailing.append(ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel", size_constraints=HIT_TARGET,
                                          key=f"local-preset-cancel-{index}", on_click=lambda e, s=stem: self.cancel(s)))
        items = [
            ft.PopupMenuItem(content="Edit / rename", icon=ft.Icons.EDIT_OUTLINED,
                             on_click=lambda e, i=index: self.open_editor(i)),
            ft.PopupMenuItem(content="Remove preset", icon=ft.Icons.PLAYLIST_REMOVE,
                             on_click=lambda e, i=index: self.confirm_remove(i)),
        ]
        if state in ("use", "in_use") and stem:
            items.append(ft.PopupMenuItem(content="Delete file", icon=ft.Icons.DELETE_OUTLINE,
                                          on_click=lambda e, s=stem: self.confirm_delete(s)))
        trailing.append(ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", key=f"local-preset-menu-{index}",
                                           items=items))
        return ft.Container(
            key=f"local-preset-{index}", border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.only(left=12, top=6, bottom=6, right=0),
            content=ft.Column([*body, ft.Row(trailing, spacing=0, wrap=True, alignment=ft.MainAxisAlignment.END)],
                              spacing=2, tight=True),
        )

    def _action(self, index: int, state: str, stem: Optional[str]) -> ft.Control:
        key = f"local-preset-action-{index}"
        if state == "download":
            return ft.FilledTonalButton(content="Download", icon=ft.Icons.DOWNLOAD, key=key,
                                        on_click=lambda e, i=index: self.spawn(self.get_preset(i)))
        if state == "error":
            return ft.FilledTonalButton(content="Retry", icon=ft.Icons.REFRESH, key=key,
                                        on_click=lambda e, i=index: self.spawn(self.get_preset(i)))
        if state == "paused":
            return ft.FilledTonalButton(content="Resume", icon=ft.Icons.PLAY_ARROW, key=key,
                                        on_click=lambda e, i=index: self.spawn(self.get_preset_resume(i)))
        if state == "use":
            return ft.FilledButton(content="Use", icon=ft.Icons.CHECK, key=key,
                                   on_click=lambda e, i=index: self.spawn(self.get_preset(i)))
        if state == "in_use":
            return ft.FilledTonalButton(content="In use", icon=ft.Icons.CHECK_CIRCLE, key=key, disabled=True)
        label = "Downloading…" if state == "downloading" else "Working…"
        return ft.FilledTonalButton(content=label, key=key, disabled=True)

    async def get_preset_resume(self, index: int) -> Optional[str]:
        rows = self.presets()
        if not 0 <= index < len(rows):
            return None
        stem = self.stem_of_preset(rows[index])
        if stem and await self.resume(stem):
            return await self.get_preset(index)
        return None

    def _other_row(self, stem: str, entry: dict) -> ft.Control:
        loaded = self.service.loaded_stem == stem
        current = str(self.current_model() if self.current_model else "") or ""
        in_use = current.lower() == f"local/{stem}".lower()
        status = ("Loaded · " if loaded else "") + ("in use · " if in_use else "") + dl.format_bytes(entry.get("size"))
        return ft.Container(
            key=f"local-other-{stem}", border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.only(left=12, top=6, bottom=6, right=0),
            content=ft.Column([
                ft.Text(f"local/{stem}", weight=ft.FontWeight.W_600, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                _small(f"{entry.get('repo', '')} · {status}", max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Row([
                    ft.TextButton(content="In use" if in_use else "Use", disabled=in_use, key=f"local-other-use-{stem}",
                                  on_click=lambda e, s=stem: self.spawn(self._use_other(s))),
                    ft.TextButton(content="Delete", icon=ft.Icons.DELETE_OUTLINE, key=f"local-other-delete-{stem}",
                                  on_click=lambda e, s=stem: self.confirm_delete(s)),
                ], spacing=0, wrap=True, alignment=ft.MainAxisAlignment.END),
            ], spacing=2, tight=True),
        )

    async def _use_other(self, stem: str) -> bool:
        try:
            return await self.activate(stem)
        except _Failed as exc:
            self.say(str(exc))
            return False

    def _download_row(self, stem: str, state: dl.DownloadState) -> ft.Control:
        buttons: list = []
        if state.running:
            buttons.append(ft.TextButton(content="Pause", icon=ft.Icons.PAUSE, key=f"local-dl-pause-{stem}",
                                         on_click=lambda e, s=stem: self.pause(s)))
        else:
            buttons.append(ft.TextButton(content="Resume", icon=ft.Icons.PLAY_ARROW, key=f"local-dl-resume-{stem}",
                                         on_click=lambda e, s=stem: self.spawn(self.resume(s))))
        buttons.append(ft.TextButton(content="Cancel", icon=ft.Icons.CLOSE, key=f"local-dl-cancel-{stem}",
                                     on_click=lambda e, s=stem: self.cancel(s)))
        return ft.Container(
            key=f"local-dl-{stem}", border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.only(left=12, top=6, bottom=6, right=0),
            content=ft.Column([
                ft.Text(f"local/{stem}", weight=ft.FontWeight.W_600, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                _small(state.status, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                ft.ProgressBar(value=state.fraction, border_radius=4),
                ft.Row(buttons, spacing=0, wrap=True, alignment=ft.MainAxisAlignment.END),
            ], spacing=2, tight=True),
        )

    # ---- presets ------------------------------------------------------------------------------------

    def open_editor(self, index: Optional[int]) -> LocalPresetEditor:
        rows = self.presets()
        current = rows[index] if index is not None and 0 <= index < len(rows) else None
        editor = LocalPresetEditor(preset=current, on_save=lambda row: self.save_preset(index, row))
        self.editor = editor
        if self.page is not None:
            editor.show(self.page)
        return editor

    def save_preset(self, index: Optional[int], row: dict) -> Optional[str]:
        rows = self.presets()
        editing = index if index is not None and 0 <= index < len(rows) else None
        clean, error = presets.validate_preset(row, rows, editing)
        if error:
            return error
        if editing is not None:
            rows[editing] = clean
        else:
            if len(rows) >= presets.MAX_PRESETS:
                return f"Keep it to {presets.MAX_PRESETS} presets"
            rows.append(clean)
        presets.save_presets(self.prefs, rows)
        self.render()
        return None

    def remove_preset(self, index: int) -> bool:
        rows = self.presets()
        if not 0 <= index < len(rows):
            return False
        rows.pop(index)
        presets.save_presets(self.prefs, rows)
        self.render()
        return True

    def confirm_remove(self, index: int) -> ConfirmDialog:
        rows = self.presets()
        name = rows[index]["name"] if 0 <= index < len(rows) else ""
        dialog = ConfirmDialog(title="Remove preset", body=f"Remove the preset “{name}”? A downloaded file stays "
                                                             "on this device (Other downloads).",
                               confirm_label="Remove", destructive=True, on_confirm=lambda: self.remove_preset(index))
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def confirm_restore(self) -> ConfirmDialog:
        def restore() -> None:
            presets.restore_defaults(self.prefs)
            self.render()

        defaults = ", ".join(f"{p['name']} ({p['file']})" for p in presets.DEFAULT_PRESETS)
        dialog = ConfirmDialog(title="Restore default presets",
                               body=f"Replace your presets with the defaults: {defaults}?", confirm_label="Restore",
                               destructive=True, on_confirm=restore)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog


class _Failed(Exception):
    """A step of the one-tap flow that failed (the message is shown)."""
