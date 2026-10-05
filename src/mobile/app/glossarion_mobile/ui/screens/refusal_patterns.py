"""Refusal patterns (Multi-Key Manager › Refusal patterns; desktop ``RefusalPatternsDialog``).

Keys: ``refusal_patterns`` (list; absent = the default list), ``disable_refusal_checks``
(default True) and ``refusal_pattern_length_limit`` (default 1000). The reads, the default list
(the single source ``unified_api_client`` uses too), the length-limit normalisation (invalid or
≤ 0 → 1000) and the "Load Patterns" merge (one per line, ``#`` comments skipped, "Loaded N new,
M skipped") are ``key_pool_service`` functions shared with the desktop dialog. Inline add / edit
follow the dialog's tree editor: stored lower-case, empty and duplicate entries ignored, new
patterns at the top. Delete and Reset ask first. Every change is saved at once through
``MobileConfigStore`` (the desktop saves on "Save Refusal Patterns"); deletions offer Undo.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = ["RefusalModel", "RefusalPatternsScreen", "merge_pattern_lines"]

log = logging.getLogger("glossarion.keys")

_KEYS = ("refusal_patterns", "disable_refusal_checks", "refusal_pattern_length_limit")


def _service(module: Any = None) -> Any:
    if module is not None:
        return module
    import key_pool_service  # shared (U4)

    return key_pool_service


def merge_pattern_lines(patterns: Sequence[str], lines: Iterable[str], *, service: Any = None) -> tuple:
    """``key_pool_service.merge_refusal_pattern_lines`` on a copy: (merged list, added, skipped)."""
    merged = list(patterns)
    added, skipped = _service(service).merge_refusal_pattern_lines(merged, list(lines))
    return merged, added, skipped


class RefusalModel:
    """Reads and writes the three refusal keys through the shared helpers (Flet-free)."""

    def __init__(self, store: Any, service: Any = None) -> None:
        self.store = store
        self._service_module = service

    @property
    def service(self) -> Any:
        return _service(self._service_module)

    def _view(self) -> dict:
        return {key: self.store.get(key) for key in _KEYS if self.store.has(key)}

    def defaults(self) -> list:
        return list(self.service.default_refusal_patterns())

    def patterns(self) -> list:
        value = self.service.load_refusal_patterns(self._view())
        return [str(p) for p in value] if isinstance(value, (list, tuple)) else self.defaults()

    def set_patterns(self, patterns: Sequence[str]) -> None:
        self.store.set("refusal_patterns", list(patterns))

    def disabled(self) -> bool:
        return bool(self.service.load_disable_refusal_checks(self._view()))

    def set_disabled(self, value: bool) -> None:
        self.store.set("disable_refusal_checks", bool(value))

    def length_limit(self) -> int:
        return int(self.service.load_refusal_length_limit(self._view()))

    def set_length_limit(self, text: Any) -> int:
        try:
            value = int(self.service.parse_refusal_length_limit(str(text or "")))
        except Exception:  # the dialog's ``except``: 1000
            value = int(getattr(self.service, "DEFAULT_REFUSAL_PATTERN_LENGTH_LIMIT", 1000))
        self.store.set("refusal_pattern_length_limit", value)
        return value

    def add(self, text: str) -> Optional[str]:
        pattern = str(text or "").strip().lower()
        patterns = self.patterns()
        if not pattern or pattern in patterns:
            return None
        self.set_patterns([pattern] + patterns)
        return pattern

    def edit(self, old: str, text: str) -> bool:
        pattern = str(text or "").strip().lower()
        patterns = self.patterns()
        if not pattern or old not in patterns or (pattern != old and pattern in patterns):
            return False
        patterns[patterns.index(old)] = pattern
        self.set_patterns(patterns)
        return True

    def delete(self, selected: Iterable[str]) -> list:
        drop = set(selected)
        before = self.patterns()
        self.set_patterns([p for p in before if p not in drop])
        return before

    def reset(self) -> list:
        before = self.patterns()
        self.set_patterns(self.defaults())
        return before

    def merge_lines(self, lines: Iterable[str]) -> tuple:
        merged, added, skipped = merge_pattern_lines(self.patterns(), lines, service=self.service)
        if added:
            self.set_patterns(merged)
        return added, skipped


try:  # the pure part above must stay importable without Flet (host tests, services)
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog
    from glossarion_mobile.ui.screens.base import Screen
    from glossarion_mobile.ui.theme import HIT_TARGET
except ImportError:  # pragma: no cover - Flet missing
    ft = None  # type: ignore[assignment]
    Screen = object  # type: ignore[assignment,misc]


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


class RefusalPatternsScreen(Screen):  # type: ignore[misc,valid-type]
    title = "Refusal patterns"

    def __init__(self, match: Any = None, *, model: RefusalModel, page: Any = None, files: Any = None,
                 notify: Optional[Callable[..., Any]] = None, run_io: Optional[Callable[..., Any]] = None,
                 spawn: Optional[Callable[[Any], Any]] = None) -> None:
        super().__init__(match)
        self.model = model
        self.page = page
        self.files = files
        self.notify = notify
        self.run_io = run_io
        self.spawn_fn = spawn
        self.selected: set = set()
        self.query = ""
        self.last_dialog: Any = None

    def say(self, message: str, action: Optional[str] = None, on_action: Any = None) -> None:
        if self.notify is None:
            return
        try:
            self.notify(message, action, on_action)
        except TypeError:
            self.notify(message)

    def spawn(self, coro: Any) -> Any:
        return self.spawn_fn(coro) if self.spawn_fn is not None else asyncio.ensure_future(coro)

    # ---- body --------------------------------------------------------------------------------------

    def build_body(self) -> "ft.Control":
        self.disable_switch = ft.Switch(label="Disable refusal pattern checks", value=self.model.disabled(),
                                        on_change=lambda e: self.model.set_disabled(bool(e.control.value)))
        self.limit_field = ft.TextField(value=str(self.model.length_limit()), label="Length limit", suffix=ft.Text("chars"),
                                        width=180, dense=True, keyboard_type=ft.KeyboardType.NUMBER,
                                        on_blur=self._on_limit, on_submit=self._on_limit,
                                        border_radius=tokens.RADII["field"])
        self.new_field = ft.TextField(hint_text="Add a pattern (case-insensitive)", dense=True, expand=True,
                                      on_submit=lambda e: self.add_pattern(), border_radius=tokens.RADII["field"])
        self.search = ft.TextField(hint_text="Filter patterns", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self.set_query(e.control.value or ""),
                                   border_radius=tokens.RADII["field"])
        self.count_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.delete_button = ft.TextButton(content="Delete selected", icon=ft.Icons.DELETE_OUTLINE,
                                           on_click=lambda e: self.confirm_delete())
        self.list_view = ft.ListView(controls=[], expand=True, spacing=0, build_controls_on_demand=True)
        header = ft.Column([
            ft.Text("Responses containing one of these phrases (within the length limit) are treated as refusals "
                    "and retried.", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row([self.disable_switch, self.limit_field], wrap=True, spacing=12,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.new_field, ft.IconButton(icon=ft.Icons.ADD, tooltip="Add pattern", size_constraints=HIT_TARGET,
                                                  on_click=lambda e: self.add_pattern())]),
            ft.Row([
                ft.TextButton(content="Load patterns", icon=ft.Icons.FILE_OPEN_OUTLINED,
                              on_click=lambda e: self.spawn(self.load_from_file())),
                self.delete_button,
                ft.TextButton(content="Reset to defaults", icon=ft.Icons.RESTART_ALT,
                              on_click=lambda e: self.confirm_reset()),
            ], wrap=True, spacing=4),
            self.search,
            self.count_text,
        ], spacing=8, tight=True)
        self.render()
        return ft.Column([ft.Container(content=header, padding=ft.Padding.symmetric(horizontal=12, vertical=8)),
                          self.list_view], expand=True, spacing=0)

    def visible_patterns(self) -> list:
        needle = self.query.strip().lower()
        return [p for p in self.model.patterns() if not needle or needle in p]

    def render(self) -> None:
        patterns = self.visible_patterns()
        total = len(self.model.patterns())
        self.selected &= set(self.model.patterns())
        self.count_text.value = f"{total} patterns" + (f" · {len(patterns)} shown" if self.query.strip() else "") + (
            f" · {len(self.selected)} selected" if self.selected else "")
        self.delete_button.disabled = not self.selected
        self.list_view.controls = [self._row(p) for p in patterns]

    def _row(self, pattern: str) -> "ft.Control":
        return ft.ListTile(
            leading=ft.Checkbox(value=pattern in self.selected, on_change=lambda e, p=pattern: self.toggle(p)),
            title=ft.Text(pattern, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            trailing=ft.IconButton(icon=ft.Icons.EDIT_OUTLINED, tooltip="Edit", size_constraints=HIT_TARGET,
                                   on_click=lambda e, p=pattern: self.open_edit(p)),
            dense=True, min_height=48, on_click=lambda e, p=pattern: self.toggle(p), key=f"refusal-{pattern}",
        )

    def _refresh(self) -> None:
        self.render()
        _push(self.list_view, self.count_text, self.delete_button)

    def set_query(self, query: str) -> None:
        self.query = query
        self._refresh()

    def toggle(self, pattern: str) -> None:
        if pattern in self.selected:
            self.selected.discard(pattern)
        else:
            self.selected.add(pattern)
        self._refresh()

    # ---- edits ---------------------------------------------------------------------------------------

    def _on_limit(self, e: Any = None) -> int:
        value = self.model.set_length_limit(self.limit_field.value)
        self.limit_field.value = str(value)
        _push(self.limit_field)
        return value

    def add_pattern(self, text: Optional[str] = None) -> Optional[str]:
        value = self.model.add(text if text is not None else (self.new_field.value or ""))
        if text is None:
            self.new_field.value = ""
            _push(self.new_field)
        self._refresh()
        return value

    def open_edit(self, pattern: str) -> Any:
        field = ft.TextField(value=pattern, autofocus=True, dense=True)

        def save() -> None:
            if not self.model.edit(pattern, field.value or ""):
                self.say("That pattern is empty or already listed")
            self._refresh()

        dialog = ft.AlertDialog(
            title=ft.Text("Edit pattern"), content=field,
            actions=[ft.TextButton(content="Cancel", on_click=lambda e: self._close(dialog)),
                     ft.FilledButton(content="Save", on_click=lambda e: (self._close(dialog), save()))],
        )
        self.last_dialog = dialog
        self.edit_field = field
        self.edit_save = save
        if self.page is not None:
            self.page.show_dialog(dialog)
        return dialog

    def _close(self, dialog: Any) -> None:
        if self.page is not None and getattr(dialog, "open", False):
            self.page.pop_dialog()

    def confirm_delete(self) -> Any:
        if not self.selected:
            self.say("Please select patterns to delete")
            return None
        selected = set(self.selected)

        def delete() -> None:
            before = self.model.delete(selected)
            self.selected.clear()
            self._refresh()
            self.say(f"Deleted {len(selected)} pattern(s)", "Undo", lambda: self._restore(before))

        dialog = ConfirmDialog(title="Confirm Deletion", body=f"Delete {len(selected)} pattern(s)?",
                               confirm_label="Delete", destructive=True, on_confirm=delete)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def _restore(self, patterns: list) -> None:
        self.model.set_patterns(patterns)
        self._refresh()

    def confirm_reset(self) -> Any:
        def reset() -> None:
            before = self.model.reset()
            self.selected.clear()
            self._refresh()
            self.say("Patterns reset to defaults", "Undo", lambda: self._restore(before))

        dialog = ConfirmDialog(title="Confirm Reset",
                               body="Reset all patterns to defaults? This will replace your current patterns.",
                               confirm_label="Reset", destructive=True, on_confirm=reset)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    async def load_from_file(self, path: Optional[str] = None) -> Optional[tuple]:
        remove_after = False
        if path is None:
            if self.files is None:
                self.say("Picking files is not available in this session")
                return None
            picked = await self.files.pick_files(target="inbox", allowed_extensions=["txt"], allow_multiple=False,
                                                 dialog_title="Load Refusal Patterns")
            if not picked:
                return None
            path = picked[0].path if hasattr(picked[0], "path") else str(picked[0])
            remove_after = True

        def read() -> list:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return handle.readlines()
            finally:
                if remove_after:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        try:
            lines = await (self.run_io(read) if self.run_io is not None else asyncio.to_thread(read))
        except Exception as exc:
            self.say(f"Failed to load file: {exc}")
            return None
        added, skipped = self.model.merge_lines(lines)
        self._refresh()
        self.say(f"✅ Loaded {added} new, {skipped} skipped")
        return added, skipped
