"""BatchPlanCard (UI_SPEC §2.12.5, Appendix A): several files (or a folder) in one translation.

＋ › Files with several files picked, or Files long-press › "Pick folder…", puts a batch plan in
the chat (sidecar ``pending_batch``) instead of the one-file attachment. The card lists the files
as FileChips (each with its glossary from the per-EPUB map: desktop "Map Glossaries to EPUBs"),
the **Include subfolders** switch for a folder (desktop "include subfolders" / ``deep_scan``,
re-walked with the desktop's supported extensions: ``plan_model.batch_files``), the glossary chip
(PlanGlossarySheet with "Map glossaries…"), the same Run options as a single Plan card, and
**Start** / **Clear**. One Start submits one ``translate`` job with every file
(``job_kinds/translate`` runs the desktop multi-file pipeline), so the outputs land in the output
root like the desktop "Run Translation" over a file list.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.direct_text_rules import attachment_icon, format_attachment_size
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["BatchPlanCard", "MAX_CHIPS"]

#: File chips shown at once; the rest are summarised ("+N more").
MAX_CHIPS = 60


class BatchPlanCard(ft.Container):
    def __init__(
        self,
        *,
        files: Sequence[str],
        folder: str = "",
        include_subfolders: bool = False,
        glossary_label: str = "",
        mapped: Optional[Mapping[str, str]] = None,
        run_options: Any = None,
        on_action: Optional[Callable[..., Any]] = None,
        start_reason: Optional[str] = None,
        key: Any = "batch-plan",
    ) -> None:
        super().__init__(key=key)
        self.files = [str(p) for p in files]
        self.folder = str(folder or "")
        self.include_subfolders = bool(include_subfolders)
        self.mapped = dict(mapped or {})
        self.on_action = on_action
        self.run_options = run_options
        count = len(self.files)
        title = f"Batch · {count} file{'s' if count != 1 else ''}"
        subtitle = os.path.basename(os.path.normpath(self.folder)) if self.folder else ""
        size = 0
        for path in self.files:
            try:
                size += os.path.getsize(path)
            except OSError:
                pass
        chips: list = []
        for index, path in enumerate(self.files[:MAX_CHIPS]):
            name = os.path.basename(path)
            glossary = self.mapped.get(os.path.normcase(os.path.abspath(path))) or self.mapped.get(path) or ""
            label = name + (f" · {os.path.basename(glossary)}" if glossary else "")
            chips.append(ft.Chip(
                label=ft.Text(label, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.Icon(icon_data(attachment_icon(os.path.splitext(name)[1].lower())), size=16),
                tooltip=path,
                on_click=(lambda e, p=path: self._act("file", p)),
                key=f"batch-file-{index}",
            ))
        if count > MAX_CHIPS:
            chips.append(ft.Chip(label=ft.Text(f"+{count - MAX_CHIPS} more"), key="batch-more"))
        self.subfolders_switch = ft.Switch(label="Include subfolders", value=self.include_subfolders,
                                           visible=bool(self.folder), key="batch-subfolders",
                                           on_change=lambda e: self._act("subfolders", bool(e.control.value)))
        self.glossary_chip = ft.Chip(label=ft.Text(glossary_label or "Glossary"), leading=ft.Icon(ft.Icons.MENU_BOOK, size=16),
                                     on_click=lambda e: self._act("glossary"), key="batch-glossary")
        epubs = [p for p in self.files if p.lower().endswith(".epub")]
        self.map_button = ft.TextButton(content="Map glossaries…", icon=ft.Icons.ACCOUNT_TREE,
                                        on_click=lambda e: self._act("map"), visible=len(epubs) > 1,
                                        key="batch-map")
        self.start_button = ft.FilledButton(content="Start", icon=ft.Icons.PLAY_ARROW, key="batch-start",
                                            disabled=start_reason is not None, on_click=lambda e: self._act("start"))
        buttons: list = [self.start_button, ft.TextButton(content="Clear", on_click=lambda e: self._act("clear"),
                                                          key="batch-clear")]
        if start_reason:
            buttons.append(ReasonChip(reason=start_reason))
        meta = " · ".join(p for p in (subtitle, format_attachment_size(size) if size else "",
                                      "Save to: Library") if p)
        controls: list = [
            ft.Row([ft.Icon(ft.Icons.LIBRARY_BOOKS, color=ft.Colors.PRIMARY),
                    ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_SMALL, expand=True)],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(meta, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row(chips, wrap=True, spacing=6, run_spacing=6, key="batch-files"),
            self.subfolders_switch,
            ft.Row([self.glossary_chip, self.map_button], wrap=True, spacing=6),
        ]
        if run_options is not None:
            controls.append(run_options.control)
        controls.append(ft.Row(buttons, wrap=True, spacing=8, run_spacing=4))
        self.content = ft.Column(controls, spacing=6, tight=True)
        self.bgcolor = ft.Colors.SURFACE_CONTAINER
        self.border_radius = tokens.RADII["plan_card"]
        self.padding = ft.Padding.all(12)

    def _act(self, action: str, *args: Any) -> Any:
        if self.on_action is None:
            return None
        return self.on_action(action, *args)
