"""RPG Maker (``/tools/rpgmaker``, UI_SPEC §4.10; desktop "RPG Maker Game (*.exe)" input + GTool).

* **Game**: "Pick game folder" (``FilePicker.get_directory_path``; copied into the Inbox, so
  it is writable) or "Pick a ZIP" (extracted into the RPG Maker work folder under the output
  root) instead of the desktop ``.exe`` filter. Android cannot hand over a picked folder
  (SAF tree), so the folder button then suggests the ZIP.
* **Scan**: ``rpgmaker_job.prepare_rpgmaker_game`` (the shared mobile entry: extract / copy
  into a writable work folder, find the game root) + ``rpgmaker_handler.detect_version`` and
  ``extract_all`` on the io pool: "RPG Maker MV · 1,234 strings".
* **GTool prompts**: the Settings tiles of ``gtool_filter_user_prompt`` /
  ``gtool_scan_user_prompt`` (the desktop "Configure GTool Scan Prompt" sub-dialog).
* **Translate**: an ``rpgmaker`` job (``job_kinds.rpgmaker``: the shared runner applies the
  translation into the game's data folder; Image output mode translates its images). While it
  runs: phase, Stop, the job log. Afterwards "Share game as ZIP" exports the translated game.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.tools.common import JobWatch, action_button, card, hint_text, schema_tiles

__all__ = ["GameScan", "RpgMakerScreen", "WORK_FOLDER", "rpgmaker_spec", "scan_game", "work_dir_for", "zip_game"]

log = logging.getLogger("glossarion.tools.rpgmaker")

KIND = "rpgmaker"
WORK_FOLDER = "RPG Maker"
PROMPT_KEYS = ("gtool_filter_user_prompt", "gtool_scan_user_prompt")
FOLDER_PICK_REASON = "Android does not let apps read a picked folder directly: pick a ZIP of the game instead."
VERSION_LABELS = {"mv": "RPG Maker MV", "mz": "RPG Maker MZ", "vxace": "RPG Maker VX Ace", "vx": "RPG Maker VX",
                  "xp": "RPG Maker XP"}


@dataclass(frozen=True)
class GameScan:
    source: str
    game_dir: str = ""
    version: str = ""
    data_dir: str = ""
    strings: Optional[int] = None
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if self.error:
            return self.error
        label = VERSION_LABELS.get(self.version, self.version.upper() or "Unknown")
        count = f" · {self.strings:,} strings" if self.strings is not None else ""
        return f"{label}{count}"


def work_dir_for(ctx: Any) -> str:
    """Where ZIP games are extracted (and read-only folders copied): ``<output root>/RPG Maker``."""
    root = getattr(ctx, "output_root", "") or getattr(ctx, "data_dir", "") or ""
    return os.path.join(root, WORK_FOLDER) if root else ""


def scan_game(source: str, work_dir: str) -> GameScan:
    """Blocking: the shared game-folder entry + version detection + the translatable string count."""
    try:
        import rpgmaker_handler
        from rpgmaker_job import prepare_rpgmaker_game
    except Exception as exc:
        return GameScan(source, error=f"The RPG Maker runner is not available in this build ({exc})")
    try:
        game_dir = prepare_rpgmaker_game(source, work_dir or None, log=lambda *_a, **_k: None)
        version, data_dir = rpgmaker_handler.detect_version(game_dir)
    except (OSError, ValueError) as exc:
        return GameScan(source, error=str(exc))
    strings = None
    try:
        _version, _data, all_strings = rpgmaker_handler.extract_all(game_dir, lambda *_a, **_k: None)
        strings = len(all_strings or ())
    except Exception:
        log.debug("counting the game strings failed", exc_info=True)
    return GameScan(source, game_dir=str(game_dir), version=str(version or ""), data_dir=str(data_dir or ""),
                    strings=strings)


def zip_game(game_dir: str, cache_dir: str) -> str:
    """Blocking: the translated game folder as ``<cache>/<name>_translated.zip`` (for Share / Save)."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"{os.path.basename(game_dir.rstrip(os.sep)) or 'game'}_translated")
    if os.path.exists(base + ".zip"):
        os.remove(base + ".zip")
    return shutil.make_archive(base, "zip", root_dir=game_dir)


def rpgmaker_spec(source: str, work_dir: str, title: str = "") -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    name = title or os.path.basename(source.rstrip("/\\"))
    return JobSpec(kind=KIND, title=name, inputs=(source,), params={"work_dir": work_dir},
                   origin={"type": "tool", "label": "Tools · RPG Maker"})


class RpgMakerScreen(Screen):
    title = "RPG Maker"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("rpgmaker", {})
        self.source: str = self.state.get("source") or ""
        self.scan: Optional[GameScan] = self.state.get("scan")
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.last_game_dir: str = self.state.get("game_dir") or ""
        # A scan extracts / copies into the same work folder the job prepares: Scan and Translate
        # wait for it (the shared entry also serialises the two on the folder).
        self.scanning = False

    def build_body(self) -> ft.Control:
        self.source_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="rpg-source")
        self.scan_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="rpg-scan")
        folder_reason = FOLDER_PICK_REASON if self.ctx.platform == "android" else None
        pickers = ft.Row([
            ft.FilledTonalButton(content="Pick game folder", icon=ft.Icons.FOLDER_OPEN,
                                 on_click=lambda e: self.ctx.spawn(self.pick_folder()), key="rpg-pick-folder"),
            ft.FilledTonalButton(content="Pick a ZIP", icon=ft.Icons.FOLDER_ZIP,
                                 on_click=lambda e: self.ctx.spawn(self.pick_zip()), key="rpg-pick-zip"),
        ], wrap=True, spacing=8)
        game_controls: list = [self.source_text, pickers]
        if folder_reason:
            game_controls.append(ReasonChip(reason="Folder: pick a ZIP on Android", detail=folder_reason))
        self.scan_button = ft.OutlinedButton(content="Scan", icon=ft.Icons.SEARCH,
                                             on_click=lambda e: self.ctx.spawn(self.run_scan()), key="rpg-scan-button")
        game_controls += [ft.Row([self.scan_button], wrap=True), self.scan_text,
                          hint_text("Replaces the desktop .exe picker: the game folder (www/data or data/) "
                                    "or a ZIP of it.")]
        game = card("Game", game_controls, icon="VIDEOGAME_ASSET", key="rpg-game")
        tiles, self.prompt_tiles = schema_tiles(self.ctx, PROMPT_KEYS)
        prompts = card("GTool prompts", tiles or [hint_text("Settings › Image & vision › GTool scan prompt")],
                       icon="EDIT_NOTE", key="rpg-prompts")
        self.translate_button = ft.FilledButton(content="Translate game", icon=ft.Icons.PLAY_ARROW,
                                                on_click=lambda e: self.ctx.spawn(self.start()), key="rpg-start")
        self.stop_button = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, visible=False,
                                             on_click=self._on_stop, key="rpg-stop")
        self.progress = ft.ProgressBar(visible=False, key="rpg-progress")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="rpg-run-status")
        self.export_button = action_button("Share game as ZIP", "IOS_SHARE", lambda e: self.ctx.spawn(self.export()),
                                           key="rpg-export",
                                           reason=None if self.last_game_dir else "Translate the game first")
        self.export_holder = ft.Container(content=self.export_button, key="rpg-export-holder")
        run = card("Run", [ft.Row([self.translate_button, self.stop_button], wrap=True, spacing=8), self.progress,
                           self.run_status, self.export_holder,
                           hint_text("Text output mode translates the game's text; Image output mode its images.")],
                   icon="PLAY_CIRCLE", key="rpg-run")
        self._render()
        return ft.ListView(controls=[game, prompts, run], expand=True, spacing=tokens.SPACING["md"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="rpg-screen")

    def did_show(self) -> None:
        for snap in self.watch.adopt((KIND,)):
            self._on_job_change(snap)

    def dispose(self) -> None:
        self.watch.stop()

    # ---- state -------------------------------------------------------------------------------------

    def _render(self) -> None:
        self.source_text.value = (f"{'📁' if os.path.isdir(self.source) else '📦'} {os.path.basename(self.source)}"
                                  if self.source else "Pick the game folder or a ZIP of it")
        self.scan_text.value = self.scan.summary if self.scan is not None else ""
        self.scan_text.color = ft.Colors.ERROR if (self.scan is not None and self.scan.error) else None
        busy = self.watch.active() is not None or self.scanning
        self.scan_button.disabled = busy or not self.source
        self.translate_button.disabled = busy or not self.source or (self.scan is not None and bool(self.scan.error))

    def set_source(self, path: str) -> None:
        self.source = path or ""
        self.scan = None
        self.state.update(source=self.source, scan=None)
        self._render()
        self._push(self.body)

    async def pick_folder(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return None
        try:
            folder = await files.pick_folder(dialog_title="Pick the RPG Maker game folder")
        except Exception as exc:  # FolderPickUnavailable (Android SAF trees) and picker errors
            self.ctx.say(f"{exc} · Pick a ZIP instead", "Pick a ZIP", lambda: self.ctx.spawn(self.pick_zip()))
            return None
        self.set_source(folder.path)
        return folder.path

    async def pick_zip(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return None
        picked = await files.pick_files(allowed_extensions=["zip"], allow_multiple=False,
                                        dialog_title="Pick a ZIP of the game")
        if not picked:
            return None
        self.set_source(picked[0].path)
        return picked[0].path

    async def run_scan(self) -> Optional[GameScan]:
        if not self.source or self.scanning:
            return None
        self.scanning = True
        self._render()
        self.scan_text.value = "Scanning…"
        self._push(self.body)
        try:
            scan = await self.ctx.io(scan_game, self.source, work_dir_for(self.ctx))
        finally:
            self.scanning = False
        self.scan = scan
        self.state["scan"] = scan  # its game_dir is only the prepared copy: "game_dir" (Share) is set by a run
        self._render()
        self._push(self.body)
        return scan

    # ---- run ---------------------------------------------------------------------------------------

    async def start(self) -> Optional[str]:
        if not self.source:
            self.ctx.say("Pick the game folder or a ZIP first")
            return None
        if self.scanning:
            self.ctx.say("Wait for the scan to finish")
            return None
        if not self.ctx.has_kind(KIND):
            self.ctx.say("RPG Maker jobs are not available in this build")
            return None
        spec = rpgmaker_spec(self.source, work_dir_for(self.ctx))
        job_id = await self.ctx.submit(spec)
        if job_id:
            self.watch.watch(job_id)
            self.ctx.remember_source("tools.rpgmaker", os.path.basename(self.source))
            self._set_running(True, "Queued…")
        return job_id

    def _set_running(self, running: bool, text: str) -> None:
        self.progress.visible = running
        self.stop_button.visible = running
        self.run_status.value = text
        self._render()
        self._push(self.body)

    def _on_stop(self, e: Any = None) -> None:
        snap = self.watch.active()
        if snap is not None and self.ctx.jobs is not None:
            try:
                self.ctx.jobs.request_stop(snap.id)
            except Exception:
                log.exception("stopping the RPG Maker job failed")

    def _on_job_change(self, snap: Any) -> None:
        if not getattr(snap, "is_terminal", False):
            self._set_running(True, str(getattr(snap, "phase", "") or "Translating…"))

    def _on_job_end(self, snap: Any) -> None:
        result = dict(getattr(snap, "result", {}) or {})
        game_dir = str(result.get("rpgmaker_game_dir") or "")
        if game_dir:
            self.last_game_dir = game_dir
            self.state["game_dir"] = game_dir
        error = getattr(snap, "error", None)
        text = "Stopped" if getattr(snap, "stopped", False) else (f"Failed: {error}" if error else
                                                                  f"Done · applied to {os.path.basename(game_dir)}")
        self.export_button = action_button("Share game as ZIP", "IOS_SHARE", lambda e: self.ctx.spawn(self.export()),
                                           key="rpg-export",
                                           reason=None if self.last_game_dir else "Translate the game first")
        self.export_holder.content = self.export_button
        self._set_running(False, text)

    async def export(self) -> Optional[str]:
        game_dir = self.last_game_dir
        files = self.ctx.files
        if not game_dir or not os.path.isdir(game_dir):
            self.ctx.say("Translate the game first")
            return None
        cache = os.path.join(getattr(self.ctx, "data_dir", "") or os.path.dirname(game_dir), "exports")
        self.ctx.say("Packing the game…")
        try:
            archive = await self.ctx.io(zip_game, game_dir, cache)
        except OSError as exc:
            self.ctx.say(f"Could not pack the game: {exc}")
            return None
        if files is not None:
            await files.share([archive])
        return archive

    def _push(self, *controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass
