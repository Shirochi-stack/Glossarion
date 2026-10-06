"""Scan for raw (``/library/scan-raw``; UI_SPEC §3.4; epub_library ``_ScanForRawDialog``).

Pairs In-progress workspaces whose raw source can't be found with candidate files:

* source folder: Library/Raw, the Inbox, or a picked folder (desktop / iOS; Android
  cannot read a picked folder tree - the chip says so);
* Match: Exact / Fuzzy with a 40-95 similarity slider (default 70), extensions
  Auto (from each workspace's kind) or epub / txt / pdf / html - persisted as
  ``epub_library_scan_raw_mode`` / ``_threshold`` / ``_folder`` / ``_auto`` / ``_exts``;
* **Scan** walks the folder and matches on the io pool (``library_core.RawScanSession``:
  the desktop ``_RawScanWorker``) and lists ☐ · workspace · matched file or
  "— no match" · ratio %, with the session's desktop status line ("ⓘ No missing-raw
  workspaces to pair." / "⚠ No candidate files found…" / "⚠ 0 of N workspaces
  matched (…). Try …" / "✔ H of N workspaces matched (…).");
* **Apply** (``RawScanSession.apply``) writes ``source_epub.txt`` + records the raw in
  the registry, then the Library rescans.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import CoreMissing, first_value
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library.common import LibraryContext
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["RawMatch", "ScanRawScreen", "normalize_matches", "status_line"]

log = logging.getLogger("glossarion.library.ui")

EXTENSIONS = ("epub", "txt", "pdf", "html")
INTRO = ("Pick a folder that contains your raw source files. Glossarion will try to match each in-progress "
         "workspace to one of those files. Matched pairings get written to each workspace's source_epub.txt so "
         "future scans resolve the raw automatically.")


@dataclass
class RawMatch:
    folder: str  # workspace output folder
    path: str  # matched raw file ("" = no match)
    ratio: float  # 0..1
    accepted: bool = True
    label: str = ""

    @property
    def matched(self) -> bool:
        return bool(self.path)


def normalize_matches(matches: Any, books: Sequence[Any] = ()) -> list:
    """``RawMatch`` rows from the shared matcher's result (dict keyed by folder, or a list)."""
    out: list[RawMatch] = []
    names = {str(b.get("output_folder") or ""): str(b.get("name") or "") for b in books}
    items: list = []
    if isinstance(matches, dict):
        items = [(folder, info) for folder, info in matches.items()]
    else:
        for item in matches or ():
            folder = first_value(item, "folder", "output_folder", "workspace", default="")
            items.append((folder, item))
    for folder, info in items:
        path = str(first_value(info, "path", "match", "matched_path", default="") or "")
        ratio = first_value(info, "ratio", "score", default=0) or 0
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = 0.0
        if ratio > 1:
            ratio = ratio / 100.0
        accepted = bool(first_value(info, "accepted", default=bool(path)))
        book = first_value(info, "book", default={}) or {}
        label = (str(book.get("folder_name") or book.get("name") or "") if isinstance(book, dict) else "")
        out.append(RawMatch(str(folder), path, ratio, accepted and bool(path),
                            label or names.get(str(folder)) or os.path.basename(str(folder))))
    return out


def status_line(matches: Sequence[RawMatch], candidate_count: int, mode: str, threshold: int) -> str:
    """The desktop status line (``_ScanForRawDialog._populate_preview``)."""
    mode_label = "exact" if mode == "exact" else f"fuzzy ≥ {threshold}%"
    workspace_count = len(matches)
    hits = sum(1 for m in matches if m.matched)
    plural_ws = "s" if workspace_count != 1 else ""
    plural_c = "s" if candidate_count != 1 else ""
    if workspace_count == 0:
        return "ⓘ No missing-raw workspaces to pair."
    if candidate_count == 0:
        return (f"⚠ No candidate files found in this folder ({mode_label}). Pick a different folder or enable "
                "more extensions.")
    if hits == 0:
        hint = ("try lowering the Similarity slider" if mode != "exact" else
                "switch to Fuzzy match or rename the raw files to match the workspace folder names")
        return (f"⚠ 0 of {workspace_count} workspace{plural_ws} matched ({candidate_count} candidate "
                f"file{plural_c} scanned, {mode_label}). Try {hint}.")
    return (f"✔ {hits} of {workspace_count} workspace{plural_ws} matched ({candidate_count} candidate "
            f"file{plural_c} scanned, {mode_label}).")


class ScanRawScreen(Screen):
    title = "Scan for raw"

    def __init__(self, match: Any, ctx: LibraryContext, *, inbox_dir: Optional[str] = None) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        cfg = self.service.cfg
        self.inbox_dir = inbox_dir
        self.mode = "fuzzy" if str(cfg("epub_library_scan_raw_mode", "exact")) == "fuzzy" else "exact"
        try:
            self.threshold = max(40, min(95, int(cfg("epub_library_scan_raw_threshold", 70))))
        except (TypeError, ValueError):
            self.threshold = 70
        self.auto = bool(cfg("epub_library_scan_raw_auto", True))
        stored = cfg("epub_library_scan_raw_exts", None)
        self.exts = {str(e).lower().lstrip(".") for e in stored if str(e).lower().lstrip(".") in EXTENSIONS} if \
            isinstance(stored, (list, tuple, set)) and stored else set(EXTENSIONS)
        self.folder = str(cfg("epub_library_scan_raw_folder", "") or "")
        if not self.folder or not os.path.isdir(self.folder):
            self.folder = self._raw_dir()
        self.matches: list[RawMatch] = []
        self.session: Any = None  # library_core.RawScanSession
        self.candidate_count = 0
        self.scanning = False
        self.applied = 0

    def _raw_dir(self) -> str:
        root = self.service.library_root()
        return os.path.join(root, "Raw") if root else ""

    # ---- body ---------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.folder_text = ft.Text(self.folder or "—", selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                   key="raw-folder")
        android = self.ctx.platform == "android"
        pick_row: list[ft.Control] = [
            ft.Chip(label=ft.Text("Library Raw folder"), on_click=lambda e: self.set_folder(self._raw_dir()),
                    key="raw-src-library"),
            ft.Chip(label=ft.Text("Inbox"), on_click=lambda e: self.set_folder(self.inbox_dir or ""),
                    disabled=not self.inbox_dir, key="raw-src-inbox"),
            ft.Chip(label=ft.Text("Pick folder…"), on_click=self._pick_folder, disabled=android,
                    key="raw-src-pick"),
        ]
        if android:
            pick_row.append(ReasonChip(reason="Folders: Android", detail="Android does not let apps read a picked "
                                       "folder tree. Share the raw files to Glossarion (they land in the Inbox) or "
                                       "import them into the Library first."))
        self.mode_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value="exact", label=ft.Text("Exact")),
                      ft.Segment(value="fuzzy", label=ft.Text("Fuzzy"))],
            selected=[self.mode], show_selected_icon=False, on_change=self._on_mode, key="raw-mode")
        self.threshold_text = ft.Text(f"{self.threshold}%", key="raw-threshold-text")
        self.slider = ft.Slider(min=40, max=95, divisions=11, value=self.threshold, label="{value}%",
                                disabled=self.mode == "exact", on_change=self._on_threshold,
                                on_change_end=self._on_threshold_end, expand=True, key="raw-threshold")
        self.auto_switch = ft.Switch(label="Auto", value=self.auto, on_change=self._on_auto, key="raw-auto")
        self.ext_chips = {ext: ft.Chip(label=ft.Text(f".{ext}"), selected=ext in self.exts, disabled=self.auto,
                                       show_checkmark=True, on_select=lambda e, x=ext: self._on_ext(x),
                                       key=f"raw-ext-{ext}") for ext in EXTENSIONS}
        self.scan_button = ft.FilledButton(content="Scan", icon=ft.Icons.SEARCH, on_click=self._on_scan,
                                           key="raw-scan")
        self.apply_button = ft.FilledTonalButton(content="Apply", icon=ft.Icons.LINK, on_click=self._on_apply,
                                                 disabled=True, key="raw-apply")
        self.status = ft.Text("", key="raw-status")
        self.progress = ft.ProgressBar(visible=False, key="raw-progress")
        self.results = ft.Column(spacing=2, key="raw-results")
        return ft.ListView([
            ft.Text(INTRO, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text("Folder", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            ft.Row(pick_row, wrap=True, spacing=6, run_spacing=6),
            self.folder_text,
            ft.Text("Match", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            self.mode_buttons,
            ft.Row([ft.Text("Similarity"), self.slider, self.threshold_text], spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text("Extensions", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            ft.Row([self.auto_switch, *self.ext_chips.values()], wrap=True, spacing=6, run_spacing=6),
            ft.Row([self.scan_button, self.apply_button], spacing=8),
            self.progress,
            self.status,
            self.results,
        ], spacing=tokens.SPACING["sm"], padding=16, expand=True)

    # ---- settings ------------------------------------------------------------------------------

    def set_folder(self, folder: str) -> None:
        if not folder:
            return
        self.folder = folder
        self.folder_text.value = folder
        self.service.set_cfg("epub_library_scan_raw_folder", folder)
        self.ctx.push(self.folder_text)

    async def _pick_folder(self, e: Any = None) -> None:
        files = self.ctx.files
        picker = getattr(files, "_get_picker", None) if files is not None else None
        if picker is None:
            return
        try:
            folder = await picker().get_directory_path(dialog_title="Folder with raw files")
        except Exception as exc:
            self.ctx.say(f"Folder picking is not available here ({exc.__class__.__name__})")
            return
        if folder and os.path.isdir(folder):
            self.set_folder(folder)

    def _on_mode(self, e: Any = None) -> None:
        selected = list(getattr(self.mode_buttons, "selected", []) or ["fuzzy"])
        self.mode = selected[0]
        self.slider.disabled = self.mode == "exact"
        self.service.set_cfg("epub_library_scan_raw_mode", self.mode)
        self.ctx.push(self.slider)

    def _on_threshold(self, e: Any = None) -> None:
        self.threshold = int(round(float(self.slider.value or 70)))
        self.threshold_text.value = f"{self.threshold}%"
        self.ctx.push(self.threshold_text)

    def _on_threshold_end(self, e: Any = None) -> None:
        self._on_threshold()
        self.service.set_cfg("epub_library_scan_raw_threshold", self.threshold)

    def _on_auto(self, e: Any = None) -> None:
        self.auto = bool(self.auto_switch.value)
        for chip in self.ext_chips.values():
            chip.disabled = self.auto
        self.service.set_cfg("epub_library_scan_raw_auto", self.auto)
        self.ctx.push(*self.ext_chips.values())

    def _on_ext(self, ext: str) -> None:
        if ext in self.exts:
            self.exts.discard(ext)
        else:
            self.exts.add(ext)
        if not self.exts:
            self.exts = set(EXTENSIONS)
        for key, chip in self.ext_chips.items():
            chip.selected = key in self.exts
        self.service.set_cfg("epub_library_scan_raw_exts", sorted(self.exts))
        self.ctx.push(*self.ext_chips.values())

    # ---- scan / apply ----------------------------------------------------------------------------

    async def scan(self) -> list:
        """``RawScanSession``: configure (persisted ``epub_library_scan_raw_*``), walk + match on the io pool."""
        if self.scanning:
            return self.matches
        self.scanning = True
        self.progress.visible = True
        self.ctx.push(self.progress)
        service = self.service
        folder, mode, threshold, auto, exts = self.folder, self.mode, self.threshold, self.auto, sorted(self.exts)

        def run() -> dict:
            if self.session is None:
                self.session = service.raw_scan_session_blocking()
            self.session.configure(folder=folder, mode=mode, threshold=threshold, auto=auto, exts=exts)
            result = self.session.scan()
            service.save_raw_scan_settings(self.session)
            return result

        try:
            result = await self.ctx.io(run)
        except CoreMissing as exc:
            self.status.value = f"Scan for raw is not available in this build ({exc.name})"
            result = None
        except Exception as exc:
            log.exception("scan for raw failed")
            self.status.value = f"⚠ Could not scan the folder ({exc})"
            result = None
        finally:
            self.scanning = False
            self.progress.visible = False
        if result is not None:
            self.candidate_count = len(result.get("candidates") or [])
            self.matches = normalize_matches(result.get("matches") or {})
            self.status.value = str(result.get("status") or "") or status_line(
                self.matches, self.candidate_count, self.mode, self.threshold)
        self._render_results()
        return self.matches

    def _render_results(self) -> None:
        rows: list[ft.Control] = []
        for index, match in enumerate(self.matches):
            matched = os.path.basename(match.path) if match.path else "— no match"
            ratio = f"{int(round(match.ratio * 100))}%" if match.path else ""
            rows.append(ft.Row([
                ft.Checkbox(value=match.accepted, disabled=not match.matched,
                            on_change=lambda e, m=match: self._toggle(m, bool(e.control.value)), key=f"raw-check-{index}"),
                ft.Column([ft.Text(match.label, weight=ft.FontWeight.W_600, max_lines=2,
                                   overflow=ft.TextOverflow.ELLIPSIS),
                           ft.Text(matched, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                   color=None if match.path else ft.Colors.ON_SURFACE_VARIANT)],
                          spacing=0, expand=True, tight=True),
                ft.Text(ratio, theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
            ], key=f"raw-row-{index}"))
        self.results.controls = rows
        self.apply_button.disabled = not any(m.accepted and m.matched for m in self.matches)
        self.ctx.push(self.results, self.apply_button, self.status, self.progress)

    def _toggle(self, match: RawMatch, value: bool) -> None:
        match.accepted = value and match.matched
        self.apply_button.disabled = not any(m.accepted and m.matched for m in self.matches)
        self.ctx.push(self.apply_button)

    async def apply(self) -> int:
        """``RawScanSession.apply``: ``source_epub.txt`` + the raw registry for every ticked match."""
        session = self.session
        if session is None or not any(m.accepted and m.matched for m in self.matches):
            return 0
        accepted = {m.folder: (m.accepted and m.matched) for m in self.matches}

        def run() -> int:
            for folder, value in accepted.items():
                session.set_accepted(folder, value)
            return int(session.apply() or 0)

        try:
            written = await self.ctx.io(run)
        except Exception as exc:
            self.ctx.say(f"Could not link the raw files: {exc}")
            return 0
        self.applied = written
        self.ctx.say(f"Linked {written} workspace{'s' if written != 1 else ''} to a raw source file. The library will "
                     "refresh in a moment." if written else "No pairings were applied.")
        self.service.mark_dirty()
        await self.service.refresh(reason="scan for raw")
        return written

    async def _on_scan(self, e: Any = None) -> list:
        return await self.scan()

    async def _on_apply(self, e: Any = None) -> int:
        return await self.apply()
