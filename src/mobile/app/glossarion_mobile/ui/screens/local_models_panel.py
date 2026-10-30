"""Model manager › On-device: the built-in llama.cpp engine, its models and the editable presets.

The tab body of ``ModelManagerScreen`` (``tab == "local"``), over ``services.local_llm``:

* **Engine**: Not loaded / Loading… / Ready · <model> / Generating, Unload, the Context slider and field
  (``ContextControl``, also on the Welcome page) and the Threads default (Prefs ``local_llm_settings``) and
  the engine's system info after the first load. Choosing a model resets Context to that model's own
  (8,192 tokens, 4,096 for files of 4 GB and more); a job on it asks for half the context as output. Without
  llama-cpp-python (a desktop dev run) the card says the engine is unavailable; nothing crashes. Below
  the on-device sampling fields, "Block other scripts in translations (adaptive)" (on by default; ``script_ban``)
  and under it the Settings tile of "Allow emoticons and symbols from other scripts (kaomoji)" (config
  ``script_ban_allow_kaomoji``, off by default; the same value as in Settings and on the desktop).
  Where the build has a GPU backend (iOS Metal, Android arm64 OpenCL), "Use GPU (experimental)" (off by
  default): the card then says what runs the model ("Runs on GPU: … (OpenCL)" / "Runs on CPU (…)") and,
  after a GPU failure, "GPU unavailable for <model>: <reason> — using CPU". On Android (OpenCL) the
  ``presets.GPU_PRESETS`` (Q4_0) are in the list whatever the switch says (owner 2026-10-27): added once,
  quietly, the first time the presets show (Model manager, Welcome page, model picker; ``offer_gpu_presets_once``),
  and switching the GPU on adds any missing one; "Add GPU-friendly presets (Q4_0)" shows while one is
  missing (removed, Restore defaults). iOS (Metal) runs the default K-quants: no Q4_0 presets there.
* **Presets** (data in Prefs, ``presets.PREF_KEY``; Qwen3.5 0.8B/2B/4B/9B defaults): ONE button per
  model. Download (one tap: Hugging Face file and, when the preset or repository has one, its vision
  projector ``mmproj-*.gguf``; download with progress, load, use as the model) / progress with Pause and
  Cancel / Resume / Use / In use / Retry. Rows of models that read images say so ("Reads images").
  The ⋮ menu edits, removes the preset, adds the image support to a text-only download or deletes the
  file; Add preset sits in the card's header, Restore defaults closes the card.
* **Add from Hugging Face**: any repo, repo:quant or huggingface.co link. A repo (or repo link) lists
  every GGUF file of the repository with its size (``FileChooser``, the usual quant preselected); a
  quant or file reference goes straight to the download question. Optionally saved as a preset.
* **Downloaded models**: files no preset covers, and paused downloads (Resume / Cancel).

Models are ``local/<stem>`` (the GGUF file name), routed by the shared ``local/`` route to the in-app
endpoint, so the chosen model works for chat, translation, glossary and every other job (a vision model
also for the chat's Vision mode and image jobs).

``OnDeviceFlow`` is the one-tap logic without the tab's controls (presets, download incl. the projector,
RAM hint, load, use): ``LocalModelsPanel`` is built on it, and the Welcome page's "Run on this phone"
buttons use it as it is (``on_change`` re-renders them).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.services.local_llm.service import DEFAULT_CONTEXT_SMALL
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.components.switch_row import switch_row
from glossarion_mobile.ui.components.sheet import (bottom_sheet, fits_compact, page_width, scroll_column, sheet_frame,
                                                   text_height)
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["CONTEXT_SLIDER", "ContextControl", "FileChooser", "LocalModelsPanel", "LocalPresetEditor", "OnDeviceFlow",
           "RAM_HINTS"]

log = logging.getLogger("glossarion.models")

HF_HINT = "unsloth/Qwen3.5-2B-GGUF · …/resolve/main/x.gguf"
STATE_TEXT = {"idle": "Not loaded", "loading": "Loading…", "ready": "Ready", "generating": "Generating…"}
VISION_TEXT = "Reads images"
#: ``service.preset_ram_check`` verdict -> (mark, words) of the RAM hint
RAM_HINTS = {"ok": ("✓", "fits this phone"), "confirm": ("⚠", "tight on free RAM"), "refuse": ("✕", "too big for this phone"),
             "unknown": ("", "")}
#: Context slider range and step (tokens); the field takes 512 up to ``presets.MAX_CTX``
CONTEXT_SLIDER = (2048, 16384, 1024)
#: Prefs flag (mobile state, not config.json): the GPU-friendly Q4_0 presets were offered once.
GPU_PRESETS_OFFERED_KEY = "local_gpu_presets_offered"
#: config.json key of "Allow emoticons and symbols from other scripts (kaomoji)", shown under the script ban
KAOMOJI_KEY = "script_ban_allow_kaomoji"
#: config.json key of "Whitelist source text", shown under it
WHITELIST_SOURCE_KEY = "script_ban_whitelist_source"


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


class ContextControl:
    """The on-device Context: a 2,048-16,384 slider and a field for exact values and larger contexts.

    ``on_change(tokens)`` runs for every valid value (slider steps of 1,024; the field 512 up to
    ``presets.MAX_CTX``); ``set_value`` shows another one (a newly chosen model's own context)."""

    def __init__(self, value: int, on_change: Callable[[int], Any], *, key: str = "local-ctx",
                 label: str = "Context (tokens)") -> None:
        low, high, step = CONTEXT_SLIDER
        self.on_change = on_change
        self.value = int(value)
        self.field = ft.TextField(label=label, value=str(self.value), dense=True, width=150,
                                  keyboard_type=ft.KeyboardType.NUMBER, key=key, on_change=self._from_field)
        self.slider = ft.Slider(min=low, max=high, divisions=(high - low) // step, value=self._clamp(self.value),
                                label="{value}", expand=True, key=f"{key}-slider", on_change=self._from_slider)
        self.control = ft.Row([self.slider, self.field], spacing=8, key=f"{key}-row",
                              vertical_alignment=ft.CrossAxisAlignment.CENTER)

    @staticmethod
    def _clamp(value: int) -> int:
        low, high, _step = CONTEXT_SLIDER
        return max(low, min(high, int(value)))

    def _from_slider(self, e: Any) -> None:
        low, _high, step = CONTEXT_SLIDER
        value = max(low, int(round(float(e.control.value) / step)) * step)
        self.value = value
        self.field.value = str(value)
        _push(self.field)
        self.on_change(value)

    def _from_field(self, e: Any) -> None:
        text = str(e.control.value or "").strip()
        if not text.isdigit() or not 512 <= int(text) <= presets.MAX_CTX:
            return
        self.value = int(text)  # may go beyond the slider's 16,384
        self.slider.value = self._clamp(self.value)
        _push(self.slider)
        self.on_change(self.value)

    def set_value(self, value: int) -> None:
        self.value = int(value)
        self.field.value = str(self.value)
        self.slider.value = self._clamp(self.value)
        _push(self.field, self.slider)


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
                                  hint_text=str(DEFAULT_CONTEXT_SMALL), dense=True, keyboard_type=ft.KeyboardType.NUMBER,
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
    """Pick one GGUF file (quant) of a repository: one radio row per file (name and size; ``selected``,
    the automatic choice, is preselected and marked "Recommended"), Cancel / Next under the list (pinned
    when the list scrolls); ``future`` gets the HfFile or None."""

    def __init__(self, repo: str, files: list, selected: Optional[hf.HfFile] = None) -> None:
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._page: Any = None
        self.files = list(files)
        self.selected = selected if selected in self.files else None
        self.group = ft.RadioGroup(
            value=self.selected.path if self.selected else None, on_change=lambda e: self.select(self.group.value),
            content=ft.Column([self._row(i, f) for i, f in enumerate(self.files)], spacing=0, tight=True),
            key="local-file-group")
        self.next_button = ft.FilledButton(content="Next", disabled=self.selected is None, key="local-file-next",
                                           on_click=lambda e: self.pick(self.chosen()))
        self.title = ft.Text(f"Choose a file of {repo}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                             weight=ft.FontWeight.W_600, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.actions = ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.pick(None)), self.next_button],
                              alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8)
        self.dialog = bottom_sheet(sheet_frame(self._body(compact=False)), key="local-file-sheet")
        self.dialog.on_dismiss = lambda e: self.pick(None)

    def _row(self, index: int, file: hf.HfFile) -> ft.ListTile:
        size = dl.format_bytes(file.size)
        return ft.ListTile(leading=ft.Radio(value=file.path, key=f"local-file-radio-{index}"),
                           title=ft.Text(file.name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                           subtitle=ft.Text(f"{size} · Recommended" if file is self.selected else size),
                           dense=True, key=f"local-file-{index}", on_click=lambda e, f=file: self.select(f.path))

    def _body(self, *, compact: bool) -> ft.Column:
        if compact:
            return scroll_column([self.title, self.group, self.actions], spacing=4, scroll=False)
        return scroll_column([self.title, self.group], footer=[self.actions], spacing=4)

    def select(self, path: Optional[str]) -> None:
        self.group.value = path
        self.next_button.disabled = self.chosen() is None
        _push(self.group, self.next_button)

    def chosen(self) -> Optional[hf.HfFile]:
        return next((f for f in self.files if f.path == self.group.value), None)

    def pick(self, file: Optional[hf.HfFile]) -> None:
        if not self.future.done():
            self.future.set_result(file)
        close_dialog(self._page, self.dialog)

    def show(self, page: Any) -> None:
        self._page = page
        # a short list keeps a compact sheet; a long one opens full height and scrolls under pinned buttons
        width = page_width(page) - 96
        estimate = 140 + sum(text_height(f.name, width) + text_height("0", width, 12) + 16 for f in self.files)
        if fits_compact(page, estimate):
            self.dialog.content = sheet_frame(self._body(compact=True))
        page.show_dialog(self.dialog)


class OnDeviceFlow:
    """The one-tap on-device flow over ``service`` (the app's ``LocalEngineService``; downloads outlive any
    screen): presets, Hugging Face lookup, model + vision projector download, the RAM hint, load and use.

    ``confirm(title, body, label)`` and ``choose(repo, files)`` (tests) replace the dialogs; ``on_change()``
    runs after every state change a screen shows (``LocalModelsPanel`` re-renders itself instead)."""

    def __init__(self, *, service: Any, prefs: Any = None, page: Any = None,
                 use_model: Optional[Callable[[str], Any]] = None, add_model: Optional[Callable[[str], Any]] = None,
                 current_model: Optional[Callable[[], str]] = None, notify: Optional[Callable[..., Any]] = None,
                 spawn: Optional[Callable[[Any], Any]] = None,
                 confirm: Optional[Callable[..., Any]] = None, choose: Optional[Callable[..., Any]] = None,
                 on_change: Optional[Callable[[], Any]] = None) -> None:
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
        self.on_change = on_change
        self.editor: Optional[LocalPresetEditor] = None
        self.last_dialog: Any = None
        self.control: Optional[ft.Control] = None
        self.gpu_presets_button: Optional[ft.TextButton] = None  # built by _gpu_section (Model manager, Welcome)

    def render(self, push: bool = True) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                log.debug("on-device flow listener failed", exc_info=True)

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

    # pause / cancel / resume: the Model manager rows and the Welcome cells
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

    def active_download(self, stem: Optional[str]) -> Optional[dl.DownloadState]:
        """The running or paused download of ``stem``: the model's, else its vision projector's."""
        if not stem:
            return None
        for key in (stem, self.service.registry.projector_key(stem)):
            state = self.downloads.states.get(key)
            if state is not None and (state.running or state.paused):
                return state
        return None

    def ram_hint(self, preset: dict) -> tuple:
        """``(verdict, mark, text)`` of ``preset`` on this phone (``service.preset_ram_check``: the engine's
        RAM gate for the model's size; "unknown" without a known size)."""
        try:
            verdict, message = self.service.preset_ram_check(preset)
        except Exception:
            verdict, message = "unknown", ""
        mark, words = RAM_HINTS.get(verdict, ("", ""))
        return verdict, mark, message or words

    def is_vision_preset(self, preset: dict, stem: Optional[str] = None) -> bool:
        """The preset's model reads images: downloaded with its projector, or the preset names one (it
        downloads with the first image)."""
        stem = stem or self.stem_of_preset(preset)
        if stem and self.service.registry.resolve(stem) and self.service.is_vision(stem):
            return True
        return bool(preset.get("mmproj"))

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

    async def ask(self, title: str, body: str, label: str, cancel_label: str = "Cancel") -> bool:
        if self.confirm_fn is not None:
            result = self.confirm_fn(title, body, label)
            return bool(await result) if asyncio.iscoroutine(result) else bool(result)
        if self.page is None:
            return True
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        dialog = ConfirmDialog(title=title, body=body, confirm_label=label, cancel_label=cancel_label,
                               on_confirm=lambda: future.done() or future.set_result(True),
                               on_cancel=lambda: future.done() or future.set_result(False))
        self.last_dialog = dialog
        dialog.show(self.page)
        return bool(await future)

    async def choose(self, repo: str, files: list, selected: Optional[hf.HfFile] = None) -> Optional[hf.HfFile]:
        if self.choose_fn is not None:
            result = self.choose_fn(repo, files, selected)
            return await result if asyncio.iscoroutine(result) else result
        if self.page is None:
            return selected
        chooser = FileChooser(repo, files, selected)
        self.last_dialog = chooser
        chooser.show(self.page)
        return await chooser.future

    # ---- the one-tap flow -------------------------------------------------------------------------

    async def fetch(self, ref: hf.HfRef, *, preset: Optional[dict] = None, choose: bool = False,
                    image_support: bool = False) -> Optional[str]:
        """Resolve ``ref`` on Hugging Face, ask for big files, download the model; the stem, or None
        (cancelled / failed: the row or status line says why).

        ``image_support`` (Add image support, an image sent to a text-only model): also its vision
        projector, when the preset or the repository has one. Owner 2026-10-29: image support is
        downloaded when an image needs it, not with every model (205-918 MB). A failed projector keeps
        the model (text only).

        ``choose`` (Add from Hugging Face): a ref naming no file and no quant lists every runnable file of
        the repository (smallest first; split GGUF sets are left out) with the automatic pick preselected;
        presets keep the one-tap automatic pick (the chooser then opens only when there is none)."""
        try:
            listing = await self.service.resolve(ref)
            choosing = choose and not (ref.file or ref.quant)
            file = None if choosing else hf.pick_file(listing.files, quant=ref.quant, file=ref.file)
            if file is None:
                runnable = sorted((f for f in listing.files if not f.split), key=lambda f: (f.size or 0, f.name.lower()))
                if not runnable:
                    raise hf.HfError("Split GGUF files are not supported yet" if listing.files
                                     else "This repository has no GGUF model files")
                default = hf.pick_file(runnable)
                file = default if len(runnable) == 1 else await self.choose(ref.repo, runnable, default)
                if file is None:
                    return None
        except hf.HfError as exc:
            raise _Failed(str(exc)) from exc
        projector = hf.pick_projector(listing.projectors, (preset or {}).get("mmproj") or None)
        registry = self.service.registry
        existing = registry.find(ref.repo, file.path)
        stem = existing or self.service.spec_for(ref.repo, listing.commit, file).stem
        if preset is not None:
            self.preset_stems[self.preset_key(preset)] = stem
        if image_support and projector is None:
            self.say(f"{ref.repo} has no vision projector: this model reads text only")
        need_projector = image_support and projector is not None and not (existing and self.service.is_vision(existing))
        total = (0 if existing else int(file.size or 0)) + (int(projector.size or 0) if need_projector else 0)
        pending = self.downloads.pending_spec(stem) or self.downloads.pending_spec(registry.projector_key(stem))
        if total > 0 and not pending:
            # Every download is confirmed (a mis-tap on a preset must never start one); big ones also
            # warn about mobile data.
            names = ([] if existing else [file.name]) + ([f"{projector.name} (image support)"] if need_projector else [])
            label = str((preset or {}).get("name") or "").strip()
            title = "Download image support?" if existing else (f"Download {label}?" if label else "Download model?")
            body = f"Download {dl.format_bytes(total)} from Hugging Face?\n\n{ref.repo} · {' + '.join(names)}"
            if total >= dl.CONFIRM_BYTES:
                body += "\n\nThis is a large download: use Wi-Fi unless your mobile data plan allows it."
            body += "\n\nYou can pause or cancel it while it downloads."
            if not await self.ask(title, body, "Download"):
                return None
        self.render()
        if not existing:
            state = await self.service.download(self.service.spec_for(ref.repo, listing.commit, file))
            if state.error:
                raise _Failed(state.error)
            if not state.done:
                return None
        if need_projector:
            state = await self.service.download(self.service.projector_spec_for(stem, ref.repo, listing.commit,
                                                                                projector))
            if state.error:
                self.say(f"Image support was not downloaded ({state.error}); the model still works for text")
            elif not state.done:
                return None
        return stem

    async def activate(self, stem: str, label: str = "") -> bool:
        """Load ``stem`` and make ``local/<stem>`` the model. Context goes back to the model's own
        (``context_for``: 4,096 for Heavy), which the Context slider then shows. Tight on free RAM only
        says so (owner 2026-10-27: Use always loads, the warning is cosmetic)."""
        self.service.save_context(0)
        self.context_changed(stem)
        verdict, message = self.service.ram_check(stem)
        if verdict == "refuse":
            raise _Failed(message)
        if verdict == "confirm":
            self.say(message)
        available, reason = self.service.engine_status()
        if available:
            ok, error = await self.service.load(stem)
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
        self.render()
        return True

    async def ensure_image_support(self, model: str) -> bool:
        """Before an image goes to ``model``: True when it can take it (not an on-device model, or one that
        reads images), else asks to download the on-device model's image support now (the confirm shows
        its size) - True once it is downloaded, False when declined or when the model has none."""
        if not str(model or "").lower().startswith("local/"):
            return True
        registry = self.service.registry
        stem = registry.resolve(model)
        if not stem:
            return True  # not downloaded: the request's own error says so
        if self.service.is_vision(stem):
            return True
        entry = registry.load().get(stem) or {}
        preset = next((p for p in self.presets() if self.stem_of_preset(p) == stem), None)
        try:
            await self.fetch(hf.HfRef(str(entry.get("repo") or ""), str(entry.get("file") or "") or None),
                             preset=preset, image_support=True)
        except _Failed as exc:
            self.say(str(exc))
            return False
        return bool(self.service.is_vision(stem))

    async def add_vision(self, index: int) -> bool:
        """Menu "Add image support": download the projector of a text-only download (the model reloads
        with it on the next request)."""
        rows = self.presets()
        if not 0 <= index < len(rows):
            return False
        preset = rows[index]
        stem = self.stem_of_preset(preset)
        if not stem or not self.service.registry.resolve(stem):
            return False
        key = self.preset_key(preset)
        self.busy.add(key)
        self.render()
        try:
            ref = hf.HfRef(preset["repo"], preset.get("file") or None, quant=preset.get("quant") or None)
            await self.fetch(ref, preset=preset, image_support=True)
            return bool(self.service.is_vision(stem))
        except _Failed as exc:
            self.say(str(exc))
            return False
        finally:
            self.busy.discard(key)
            self.render()

    def context_changed(self, stem: Optional[str]) -> None:
        """The Context default changed (a model was chosen); a screen showing a ``ContextControl`` updates it."""

    def current_context(self) -> int:
        """The Context the slider shows: the current ``local/`` model's (its preset's, the saved default or
        its own), else the saved default, else the small-model default (``DEFAULT_CONTEXT_SMALL``)."""
        model = str(self.current_model() if self.current_model is not None else "") or ""
        if model.lower().startswith("local/"):
            return self.service.context_for(model[len("local/"):])
        return self.service.settings()["n_ctx"] or DEFAULT_CONTEXT_SMALL

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

    def other_models(self, preset_rows: Optional[list] = None) -> tuple:
        """What no preset row shows: ``([(stem, registry entry)], [(key, running or paused DownloadState)])``
        (user-added Hugging Face files, and their downloads)."""
        listed: set = set()
        for preset in self.presets() if preset_rows is None else preset_rows:
            stem = self.stem_of_preset(preset)
            if stem:
                listed.add(stem)
                if self.row_state(preset, stem)[0] in ("downloading", "paused"):  # the row shows its projector too
                    listed.add(self.service.registry.projector_key(stem))
        downloaded: list = []
        for stem, entry in sorted(self.service.registry.load().items(), key=lambda kv: kv[0].casefold()):
            if stem not in listed:
                downloaded.append((stem, entry))
                listed.add(stem)
        pending = [(stem, state) for stem, state in list(self.downloads.states.items())
                   if stem not in listed and (state.running or state.paused)]
        return downloaded, pending

    async def get_preset_resume(self, index: int) -> Optional[str]:
        """Resume the preset's paused download, then load and use it (the Resume button)."""
        rows = self.presets()
        if not 0 <= index < len(rows):
            return None
        stem = self.stem_of_preset(rows[index])
        paused = self.active_download(stem)
        if paused is not None and await self.resume(paused.stem):
            return await self.get_preset(index)
        return None

    async def _use_other(self, stem: str) -> bool:
        """Load and use a downloaded model no preset names (a user-added Hugging Face file)."""
        try:
            return await self.activate(stem)
        except _Failed as exc:
            self.say(str(exc))
            return False

    def row_state(self, preset: dict, stem: Optional[str]) -> tuple:
        """(state, status text) of a preset row: download | downloading | paused | use | in_use | error | busy."""
        key = self.preset_key(preset)
        state = self.active_download(stem)
        current = str(self.current_model() if self.current_model else "") or ""
        if state is not None and state.running:
            return "downloading", _download_status(state)
        if stem and self.service.registry.resolve(stem):
            loaded = self.service.loaded_stem == stem
            if current.lower() == f"local/{stem}".lower():
                return "in_use", "Loaded · in use" if loaded else "In use · loads on the first request"
            return "use", "Loaded" if loaded else "On this device"
        if state is not None and state.paused:
            return "paused", _download_status(state)
        if key in self.errors:
            return "error", f"Failed: {self.errors[key]}"
        if key in self.busy:
            return "busy", "Looking up the model on Hugging Face…"
        size = presets.download_size(preset)
        return "download", f"Not downloaded · {dl.format_bytes(size)}" if size else "Not downloaded"

    # ---- GPU ------------------------------------------------------------------------------------------

    def _gpu_backend(self) -> Optional[str]:
        return getattr(self.service, "gpu_backend", None)

    def _gpu_section(self, key_prefix: str = "local") -> list:
        """"Use GPU (experimental)" (off by default; only where the build has a GPU backend), its note, the
        fallback line ("GPU unavailable: … — using CPU") and, on Android, "Add GPU-friendly presets (Q4_0)"
        (the fallback while one is missing: switching the GPU on adds them, ``save_gpu``). Shared by the
        Model manager's engine card and the Welcome page ("Run on this phone"); *key_prefix* keeps their
        control keys apart. Building it never changes the presets list."""
        backend = self._gpu_backend()
        use = bool(backend) and bool(getattr(self.service, "use_gpu", lambda: False)())
        self.gpu_switch = ft.Switch(label="Use GPU (experimental)", value=use, key=f"{key_prefix}-gpu", visible=bool(backend),
                                    on_change=lambda e: self.save_gpu())
        if backend == "OpenCL":
            text = ("Runs the model on a Qualcomm Adreno GPU (Snapdragon) through OpenCL. Fastest with Q4_0 models; "
                    "other quantizations can be slower than the CPU. Other GPUs, or a failure, fall back to the "
                    "CPU automatically.")
        else:
            text = ("Runs the model on the iPhone's or iPad's GPU through Metal. A failure falls back to the CPU "
                    "automatically. The GPU shares the device's memory.")
        self.gpu_note = _small(text, key=f"{key_prefix}-gpu-note", visible=bool(backend))
        self.gpu_status = ft.Text("", color=semantic("warning"), visible=False, key=f"{key_prefix}-gpu-status",
                                  theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True)
        self.gpu_presets_button = ft.TextButton(content="Add GPU-friendly presets (Q4_0)", icon=ft.Icons.ADD,
                                                key=f"{key_prefix}-gpu-presets", on_click=lambda e: self.add_gpu_presets())
        self._render_gpu_presets_button()
        # the current model is a K-quant the Adreno kernels run slowly: one tap gets its Q4_0 preset
        self.gpu_swap_text = _small("", key=f"{key_prefix}-gpu-swap-text")
        self.gpu_swap_button = ft.TextButton(content="", icon=ft.Icons.SWAP_HORIZ, key=f"{key_prefix}-gpu-swap-button",
                                             on_click=lambda e: self.spawn(self.swap_to_gpu_quant()))
        self.gpu_swap_box = ft.Column([self.gpu_swap_text, self.gpu_swap_button], spacing=0, tight=True,
                                      key=f"{key_prefix}-gpu-swap")
        self._render_gpu_swap(push=False)
        return [self.gpu_switch, self.gpu_note, self.gpu_status, self.gpu_swap_box, self.gpu_presets_button]

    def _render_gpu_presets_button(self) -> None:
        if self.gpu_presets_button is None:
            return
        missing = presets.missing_gpu_presets(self.presets())
        self.gpu_presets_button.visible = self._gpu_backend() == "OpenCL" and bool(missing)

    def save_gpu(self) -> bool:
        """The switch. Turning it on with Android's OpenCL backend also adds the GPU-friendly Q4_0 presets
        the list lacks (``add_gpu_presets``; owner 2026-10-26: no extra tap). Off keeps every preset and
        file; iOS (Metal) runs the default K-quants, so it adds none there."""
        was_on = bool(getattr(self.service, "use_gpu", lambda: False)())
        value = bool(self.service.save_gpu(self.gpu_switch.value))
        self.gpu_switch.value = value
        if value and not was_on and self._gpu_backend() == "OpenCL":
            # re-renders the list; on the Welcome page that rebuilds this section (self.gpu_* are the new ones)
            self.add_gpu_presets(auto=True)
        self._render_gpu_presets_button()
        self._render_gpu_swap(push=False)
        _push(self.gpu_switch, self.gpu_presets_button, self.gpu_swap_box)
        return value

    def gpu_swap(self) -> Optional[tuple]:
        """Android GPU on and the current ``local/`` model is not a Q4_0 file while its repository has a
        Q4_0 preset (``presets.GPU_PRESETS``): ``(stem, label, quant, gpu preset)``, else None. The model
        still runs (on the GPU, maybe slower than the CPU); only a failure falls back to the CPU."""
        if self._gpu_backend() != "OpenCL" or not self.service.use_gpu():
            return None
        model = str(self.current_model() if self.current_model is not None else "") or ""
        if not model.lower().startswith("local/"):
            return None
        stem = model[len("local/"):]
        entry = self.service.registry.load().get(stem) or {}
        file = entry.get("file")
        if not file or presets.is_gpu_quant(file):
            return None
        gpu_row = presets.gpu_preset_for(entry.get("repo"))
        if gpu_row is None:
            return None
        label = str((self.service.preset_for(stem) or {}).get("name") or stem)
        return stem, label, presets.file_quant(file) or "not Q4_0", gpu_row

    def _render_gpu_swap(self, push: bool = True) -> None:
        box = getattr(self, "gpu_swap_box", None)
        if box is None:
            return
        swap = self.gpu_swap()
        box.visible = swap is not None
        if swap is not None:
            _stem, label, quant, row = swap
            self.gpu_swap_text.value = (f"{label} is {quant}, which can be slow on this GPU: its fast kernels "
                                        "are for Q4_0.")
            if self.service.registry.find(row["repo"], row["file"]):
                self.gpu_swap_button.content = f"Switch to {row['name']}"
            else:
                size = presets.download_size(row)
                self.gpu_swap_button.content = f"Get {row['name']}" + (f" · {dl.format_bytes(size)}" if size else "")
        if push:
            _push(box)

    async def swap_to_gpu_quant(self) -> Optional[str]:
        """The hint's tap: add the model's Q4_0 preset to the list (when missing) and run its one-tap flow
        (confirm the download, load, use). The current model keeps working until the new one is in use."""
        swap = self.gpu_swap()
        if swap is None:
            return None
        _rows, index = presets.add_gpu_preset(self.prefs, swap[3])
        self.render()
        model = await self.get_preset(index)
        self._render_gpu_swap()
        return model

    def add_gpu_presets(self, *, auto: bool = False, quiet: bool = False) -> list:
        """Append the Q4_0 presets (``presets.GPU_PRESETS``) the list does not have yet: the button, and
        ``save_gpu`` when the GPU goes on (``auto``: silent when none was added; ``quiet``: silent)."""
        rows, added = presets.add_gpu_presets(self.prefs)
        if self.prefs is not None:
            self.prefs.set(GPU_PRESETS_OFFERED_KEY, True)  # offered once: a preset removed later stays removed
        self._render_gpu_presets_button()
        _push(self.gpu_presets_button)
        # said before the re-render: the Welcome page shows the note under its cells
        if added and not quiet:
            self.say(f"Added {len(added)} GPU-friendly preset{'s' if len(added) != 1 else ''} (Q4_0)")
        elif not added and not auto:
            self.say(f"Keep it to {presets.MAX_PRESETS} presets: remove one to add the GPU-friendly presets"
                     if presets.missing_gpu_presets(rows) else "The GPU-friendly presets are already in the list")
        if added or not auto:
            self.render()
        return added

    def offer_gpu_presets_once(self) -> list:
        """Owner 2026-10-27: on Android (OpenCL) the Q4_0 presets show whatever "Use GPU" says, so the
        first time the presets show (Model manager, Welcome page, model picker) they are added, without a
        note. After that (or after any add) the list is the user's: a Q4_0 preset removed later is not
        added back (the button offers it)."""
        if self.prefs is None or self.prefs.get(GPU_PRESETS_OFFERED_KEY, False):
            return []
        if self._gpu_backend() != "OpenCL":
            return []
        return self.add_gpu_presets(auto=True, quiet=True)


class LocalModelsPanel(OnDeviceFlow):
    """The On-device tab. ``build()`` returns its control; nothing runs until ``did_show()``."""

    #: the Settings context of the shared config tiles (None: the app's, ``sheet_env().ctx``)
    settings_ctx: Any = None

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
        self.context = ContextControl(self.current_context(), self.save_context)
        self.ctx_field = self.context.field
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
            self.context.control,
            _small("Choosing a model sets its own context (4,096 for Heavy). Jobs on an on-device model ask for "
                   "half the context as output and send one request at a time.", key="local-ctx-note"),
            ft.Row([self.threads_field,
                    ft.TextButton(content="Save", key="local-settings-save", on_click=lambda e: self.save_settings())],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.settings_error,
            *self._gpu_section(),
            *self._sampling_section(),
        ])
        self.preset_column = ft.Column([], spacing=6, tight=True, key="local-presets")
        self.add_preset_button = ft.TextButton(content="Add preset", icon=ft.Icons.ADD, key="local-add-preset",
                                               on_click=lambda e: self.open_editor(None))
        presets_card = SectionCard(title="Models", key="local-presets-card", trailing=self.add_preset_button,
                                   subtitle="One tap downloads a model from Hugging Face and uses it.", children=[
            self.preset_column,
            ft.Row([ft.TextButton(content="Restore defaults", icon=ft.Icons.RESTART_ALT, key="local-restore",
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
        self.offer_gpu_presets_once()
        self.render()
        self.render_engine(self.service.snapshot())

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        self._drop_kaomoji_observer()

    def _drop_kaomoji_observer(self) -> None:
        unsub, self._kaomoji_unsub = getattr(self, "_kaomoji_unsub", None), None
        if callable(unsub):
            unsub()

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
            if getattr(snap, "vision", False):
                text += " · reads images"
        if snap.queue:
            text += f" · {snap.queue} waiting"
        self.engine_state.value = text
        self.engine_state.color = semantic("success") if snap.state in ("ready", "generating") else None
        details = []
        if snap.load_seconds is not None and snap.model:
            details.append(f"Loaded in {snap.load_seconds:.1f} s")
        device = getattr(snap, "device", None)
        if device and snap.model and snap.state != "idle":
            details.append(f"Runs on {device}")
        if snap.system_info:
            details.append(snap.system_info)
        if snap.error:
            details.append(snap.error)
        self.engine_detail.value = "\n".join(details)
        self.engine_detail.visible = bool(details)
        self.unload_button.disabled = snap.state not in ("ready",)
        self.unavailable_text.visible = not snap.available
        self.unavailable_text.value = snap.unavailable_reason or ""
        note = getattr(snap, "gpu_note", None)
        self.gpu_status.value = note or ""
        self.gpu_status.visible = bool(note)
        self._render_gpu_swap(push=False)  # a model was loaded / switched
        if push:
            _push(self.engine_state, self.engine_detail, self.unload_button, self.unavailable_text, self.gpu_status,
                  self.gpu_swap_box)

    def context_changed(self, stem: Optional[str]) -> None:
        if getattr(self, "context", None) is not None:
            self.context.set_value(self.service.context_for(stem) if stem else self.current_context())

    def save_context(self, tokens: int) -> Optional[str]:
        """The Context slider or field: saved at once; a loaded model reloads with it on its next request."""
        _values, error = self.service.save_context(tokens)
        self.settings_error.value = error or ""
        self.settings_error.visible = bool(error)
        _push(self.settings_error)
        return error

    def _sampling_section(self) -> list:
        """On-device sampling (Qwen's recommendation by default): replaces the global Temperature and
        anti-duplicate values for on-device models only."""
        from glossarion_mobile.services.local_llm.service import SAMPLING_LABELS

        values = self.service.sampling()
        self.sampling_fields = {
            key: ft.TextField(value=f"{values[key]:g}", label=label, dense=True, width=118,
                              keyboard_type=ft.KeyboardType.NUMBER, key=f"local-sampling-{key}")
            for key, label in SAMPLING_LABELS.items()
        }
        self.sampling_error = ft.Text("", color=ft.Colors.ERROR, visible=False,
                                      theme_style=ft.TextThemeStyle.BODY_SMALL)
        self.forced_truncation_switch = ft.Switch(value=self.service.forced_truncation(), key="local-forced-truncation",
                                                  on_change=lambda e: self.save_forced_truncation())
        self.forced_truncation_seconds = ft.TextField(value=str(self.service.forced_truncation_seconds()), dense=True,
                                                      label="Seconds", width=118, key="local-forced-truncation-seconds",
                                                      keyboard_type=ft.KeyboardType.NUMBER,
                                                      on_blur=lambda e: self.save_forced_truncation(),
                                                      on_submit=lambda e: self.save_forced_truncation())
        self.mark_glossary_complete_switch = ft.Switch(
            value=self.service.mark_glossary_complete(), key="local-mark-glossary-complete",
            on_change=lambda e: self.service.save_mark_glossary_complete(self.mark_glossary_complete_switch.value))
        self.block_cjk_switch = ft.Switch(value=self.service.block_cjk(),
                                          key="local-block-cjk", on_change=lambda e: self.save_block_cjk())
        return [
            ft.Text("Sampling (on-device models only)", weight=ft.FontWeight.W_600, key="local-sampling-title"),
            _small("Qwen's recommended values and the DRY sampler reduce repetition loops. DRY stops the "
                   "model repeating a stretch of text, more strongly the longer the repeat; DRY multiplier 0 "
                   "turns it off. These replace the anti-duplicate settings for on-device models; cloud "
                   "models keep using those."),
            ft.Row(list(self.sampling_fields.values()), wrap=True, spacing=8, run_spacing=8, key="local-sampling"),
            ft.Row([ft.TextButton(content="Save", key="local-sampling-save", on_click=lambda e: self.save_sampling()),
                    ft.TextButton(content="Qwen defaults", key="local-sampling-reset",
                                  on_click=lambda e: self.reset_sampling())], spacing=4),
            self.sampling_error,
            switch_row(self.forced_truncation_switch, "Forced Truncation"),
            self.forced_truncation_seconds,
            _small("When the model keeps repeating the same text for this many seconds, the answer stops and the "
                   "chapter is marked truncated, so the run moves on. Text the source itself repeats is left alone."),
            switch_row(self.mark_glossary_complete_switch, "Mark glossary forced truncation as complete"),
            _small("A glossary request stopped this way counts as finished, with the entries it has."),
            switch_row(self.block_cjk_switch, "Block other scripts in translations (adaptive)"),
            _small("Stops small models from slipping into another writing system (Chinese, Japanese, Korean, "
                   "Hindi, Thai, Arabic, Cyrillic ...) while translating: only the target language's own "
                   "script, Latin letters, numbers and punctuation can be written. Glossary requests and the "
                   "single-pass auto glossary are never blocked.", key="local-block-cjk-note"),
            *self._kaomoji_rows(),
        ]

    def _kaomoji_rows(self) -> list:
        """"Allow emoticons and symbols from other scripts (kaomoji)" and "Whitelist source text": the Settings tiles of the shared config
        keys (``KAOMOJI_KEY``, ``WHITELIST_SOURCE_KEY``; also Settings › Anti-Duplicate Parameters › Logit Bias and the desktop), so
        both places edit one value. Nothing without the Settings context (a screen built in a test)."""
        self.kaomoji_tile = self.whitelist_tile = None
        ctx = self.settings_ctx
        if ctx is None:
            from glossarion_mobile.ui.sheets.model_sheet import sheet_env

            ctx = sheet_env().ctx
        if ctx is None:
            return []
        rows: list = []
        observers: list = []
        observe = getattr(ctx.store, "observe", None)
        for key, attribute in ((KAOMOJI_KEY, "kaomoji_tile"), (WHITELIST_SOURCE_KEY, "whitelist_tile")):
            try:
                spec = ctx.schema.spec(key)
                if spec is None:
                    continue
                from glossarion_mobile.ui.settings.tiles import make_tile

                tile = make_tile(spec, ctx)
            except Exception:
                log.exception("building the %s tile failed", key)
                continue
            setattr(self, attribute, tile)
            rows.append(tile.control)
            if callable(observe):  # changed in Settings: the tile here follows
                observers.append(observe(key, lambda _key, _value, t=tile: t.refresh()))
        if observers:
            # kept apart from ``_unsubs``: did_show subscribes to downloads / the engine only while that list
            # is empty, so a build-time entry there left the On-device tab without progress updates
            self._drop_kaomoji_observer()  # a rebuilt section replaces its tiles
            self._kaomoji_unsub = lambda: [unsub() for unsub in observers if callable(unsub)]
        return rows

    def save_forced_truncation(self) -> tuple:
        enabled, seconds = self.service.save_forced_truncation(self.forced_truncation_switch.value,
                                                               self.forced_truncation_seconds.value)
        self.forced_truncation_seconds.value = str(seconds)
        return enabled, seconds

    def save_block_cjk(self) -> bool:
        value = self.service.save_block_cjk(self.block_cjk_switch.value)
        return value

    def save_sampling(self) -> Optional[str]:
        values, error = self.service.save_sampling({k: f.value for k, f in self.sampling_fields.items()})
        self.sampling_error.value = error or ""
        self.sampling_error.visible = bool(error)
        _push(self.sampling_error)
        if not error:
            for key, field in self.sampling_fields.items():
                field.value = f"{values[key]:g}"
            _push(*self.sampling_fields.values())
        return error

    def reset_sampling(self) -> Optional[str]:
        for field in self.sampling_fields.values():
            field.value = ""
        return self.save_sampling()

    def save_settings(self) -> Optional[str]:
        values, error = self.service.save_settings(self.ctx_field.value, self.threads_field.value)
        self.settings_error.value = error or ""
        self.settings_error.visible = bool(error)
        _push(self.settings_error)
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
            stem = await self.fetch(ref, choose=True)
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

    def delete_file(self, stem: str) -> bool:
        busy = self.service.busy_with(stem)
        removed = self.service.delete(stem)
        if not removed and busy:
            self.say("The model is busy; delete it when the current answer is done")
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
        preset_rows = self.presets()
        if not preset_rows:
            rows.append(_small("No presets. Add one, or restore the defaults.", key="local-presets-empty"))
        for index, preset in enumerate(preset_rows):
            rows.append(self._preset_row(index, preset, self.stem_of_preset(preset)))
        self.preset_column.controls = rows
        downloaded, pending = self.other_models(preset_rows)
        others: list = [self._other_row(stem, entry) for stem, entry in downloaded]
        others.extend(self._download_row(stem, state) for stem, state in pending)
        self.other_column.controls = others
        self.other_card.visible = bool(others)
        if push:
            _push(self.preset_column, self.other_card)

    def _preset_row(self, index: int, preset: dict, stem: Optional[str]) -> ft.Control:
        state, status = self.row_state(preset, stem)
        if state == "download":
            _verdict, mark, words = self.ram_hint(preset)
            if mark:
                status += f" · {mark} {RAM_HINTS[_verdict][1]}"
        lines: list = [
            ft.Text(preset["name"], weight=ft.FontWeight.W_600, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(_preset_ref(preset), theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if preset.get("note"):
            lines.append(_small(preset["note"], max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        if self.is_vision_preset(preset, stem):
            lines.append(_vision_line(f"local-preset-vision-{index}"))
        status_text = _small(status, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, key=f"local-preset-status-{index}")
        if state == "error":
            status_text.color = ft.Colors.ERROR
        elif state in ("use", "in_use"):
            status_text.color = semantic("success")
        lines.append(status_text)
        body: list = [ft.Row([ft.Column(lines, spacing=0, tight=True, expand=True)], spacing=0)]
        download = self.active_download(stem)
        if state in ("downloading", "paused") and download is not None:
            body.append(ft.ProgressBar(value=download.fraction, border_radius=4, key=f"local-preset-progress-{index}"))
        target = download.stem if download else stem
        trailing: list = [self._action(index, state, stem, target)]
        if state == "downloading" and stem:
            trailing.append(ft.TextButton(content="Pause", icon=ft.Icons.PAUSE, key=f"local-preset-pause-{index}",
                                          on_click=lambda e, s=target: self.pause(s)))
        if state == "paused" and stem:
            trailing.append(ft.TextButton(content="Cancel", icon=ft.Icons.CLOSE, key=f"local-preset-cancel-{index}",
                                          on_click=lambda e, s=target: self.cancel(s)))
        items = [
            ft.PopupMenuItem(content="Edit / rename", icon=ft.Icons.EDIT_OUTLINED,
                             on_click=lambda e, i=index: self.open_editor(i)),
            ft.PopupMenuItem(content="Remove preset", icon=ft.Icons.PLAYLIST_REMOVE,
                             on_click=lambda e, i=index: self.confirm_remove(i)),
        ]
        if state in ("use", "in_use") and stem and not self.service.is_vision(stem):
            items.append(ft.PopupMenuItem(content="Add image support", icon=ft.Icons.IMAGE_OUTLINED,
                                          on_click=lambda e, i=index: self.spawn(self.add_vision(i))))
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

    def _action(self, index: int, state: str, stem: Optional[str], target: Optional[str] = None) -> ft.Control:
        key = f"local-preset-action-{index}"
        if state == "downloading" and stem:
            # the main button while it downloads: an undo right where "Download" was tapped
            return ft.FilledTonalButton(content="Cancel download", icon=ft.Icons.CLOSE, key=key,
                                        on_click=lambda e, s=(target or stem): self.cancel(s))
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

    def _other_row(self, stem: str, entry: dict) -> ft.Control:
        loaded = self.service.loaded_stem == stem
        current = str(self.current_model() if self.current_model else "") or ""
        in_use = current.lower() == f"local/{stem}".lower()
        status = ("Loaded · " if loaded else "") + ("in use · " if in_use else "") + dl.format_bytes(entry.get("size"))
        vision = [_vision_line(f"local-other-vision-{stem}")] if entry.get("mmproj") else []
        return ft.Container(
            key=f"local-other-{stem}", border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.only(left=12, top=6, bottom=6, right=0),
            content=ft.Column([
                ft.Text(f"local/{stem}", weight=ft.FontWeight.W_600, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                _small(f"{entry.get('repo', '')} · {status}", max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                *vision,
                ft.Row([
                    ft.TextButton(content="In use" if in_use else "Use", disabled=in_use, key=f"local-other-use-{stem}",
                                  on_click=lambda e, s=stem: self.spawn(self._use_other(s))),
                    ft.TextButton(content="Delete", icon=ft.Icons.DELETE_OUTLINE, key=f"local-other-delete-{stem}",
                                  on_click=lambda e, s=stem: self.confirm_delete(s)),
                ], spacing=0, wrap=True, alignment=ft.MainAxisAlignment.END),
            ], spacing=2, tight=True),
        )

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
                ft.Text(f"local/{_owner(stem)}", weight=ft.FontWeight.W_600, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                _small(_download_status(state), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
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


def _owner(key: str) -> str:
    """The model stem of a download key (``<stem>.mmproj`` is its vision projector)."""
    return key[: -len(dl.PROJECTOR_SUFFIX)] if key.endswith(dl.PROJECTOR_SUFFIX) else key


def _download_status(state: dl.DownloadState) -> str:
    return f"Image support · {state.status}" if state.stem.endswith(dl.PROJECTOR_SUFFIX) else state.status


def _vision_line(key: str) -> ft.Control:
    """The "Reads images" mark of a vision model's row."""
    return ft.Row([ft.Icon(ft.Icons.IMAGE_OUTLINED, size=14, color=semantic("success")),
                   _small(VISION_TEXT, color=semantic("success"), max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)],
                  spacing=4, tight=True, key=key)
