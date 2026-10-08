"""Plan card pieces (UI_SPEC §2.12.1): Run options, Choose chapters, the destination sheet.

``RunOptionsPanel`` - the collapsed "Run options" ``ExpansionTile`` (summary "Batch 10 · Temp 0.3 ·
Rolling summary"): the Settings pages' own schema tiles (``settings.tiles.make_tile``) for
``plan_model.RUN_OPTION_KEYS`` and the switch **Only for this run** (default on). On, the tiles
write into a ``plan_model.RunOverlayStore`` whose values become the JobSpec ``config_overrides``
(merged over the config snapshot at job start, like every other override); off, they write
config.json like Settings. The tiles are rebuilt under a per-build key when the switch flips
(Flet 1.0.3 freezes a replaced subtree that keeps its key).

``ChooseChaptersSheet`` - the range field, the Spine order switch and the live preview of the
files the range translates (``plan_model.range_preview`` on the io pool: the desktop 🔍
``_preview_chapter_range_files`` texts), writing ``chapter_range`` / ``use_spine_order`` through
the same store as the Run options.

``DESTINATIONS`` / ``destination_sheet`` - "Save to: This chat" (Direct Text semantics) or
"Save to: Library" (the normal pipeline: a ``translate`` job, the workspace in the output root).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.plan_model import RUN_OPTION_KEYS, RunOverlayStore, range_label, run_options_summary
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import scroll_sheet

__all__ = ["ChooseChaptersSheet", "DESTINATIONS", "RunOptionsPanel", "destination_sheet"]

log = logging.getLogger("glossarion.chat.plan")

#: Plan destinations: (id, chip label, sheet subtitle)
DESTINATIONS = (
    ("chat", "Save to: This chat", "Direct Text: the run's files stay in this chat's workspace"),
    ("library", "Save to: Library",
     "The normal pipeline: the workspace goes to the output root and the book appears in the Library"),
)


def destination_label(destination: Optional[str]) -> str:
    return next((label for did, label, _s in DESTINATIONS if did == destination), DESTINATIONS[0][1])


def destination_sheet(current: Optional[str], on_choose: Callable[[str], Any], *, tablet: bool = False,
                      library_reason: Optional[str] = None) -> ActionSheet:
    items = []
    for did, label, _subtitle in DESTINATIONS:
        reason = library_reason if did == "library" else None
        items.append(ActionItem(label + (" ✓" if did == (current or "chat") else ""),
                                (lambda d=did: on_choose(d)), icon="CHAT" if did == "chat" else "LOCAL_LIBRARY",
                                disabled_reason=reason, key=f"dest-{did}"))
    return ActionSheet(items, title="Where should the output go?",
                       subtitle=" · ".join(f"{label.split(': ')[-1]}: {sub}" for _d, label, sub in DESTINATIONS),
                       tablet=tablet)


class RunOptionsPanel:
    """"Run options" for one plan (see the module docstring).

    ``ctx``: the SettingsContext (``ChatEnv``) whose ``store`` is the config; ``values``: this run's
    saved overrides; ``on_change(values, only_this_run)`` persists them with the plan."""

    def __init__(self, ctx: Any, *, values: Optional[dict] = None, only_this_run: bool = True,
                 on_change: Optional[Callable[[dict, bool], Any]] = None, keys: Sequence[str] = RUN_OPTION_KEYS,
                 key: str = "run-options") -> None:
        self.ctx = ctx
        self.keys = tuple(keys)
        self.only = bool(only_this_run)
        self.on_change = on_change
        self.overlay = RunOverlayStore(getattr(ctx, "store", None), values or {}, on_change=self._overlay_changed)
        self.tiles: dict = {}
        self._gen = 0
        self._unsub: Any = None
        self.switch = ft.Switch(label="Only for this run", value=self.only, on_change=self._on_switch,
                                key=f"{key}-only")
        self.summary_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS)
        self.body = ft.Container(key=f"{key}-body-0")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.control = ft.ExpansionTile(
            title="Run options", subtitle=self.summary_text, dense=True, maintain_state=True,
            controls=[ft.Container(content=ft.Column([self.switch, self.note, self.body], spacing=6, tight=True),
                                   padding=ft.Padding.only(left=4, right=4, bottom=8))],
            key=key,
        )
        self._key = key
        self.render(push=False)

    # ---- state -------------------------------------------------------------------------------------

    @property
    def values(self) -> dict:
        return dict(self.overlay.values)

    @property
    def store(self) -> Any:
        """Where edits go now: this run's overlay ("Only for this run") or the config."""
        return self.overlay if self.only else getattr(self.ctx, "store", None)

    def config_overrides(self) -> dict:
        """The JobSpec ``config_overrides`` of this plan ({} when the switch is off)."""
        return dict(self.overlay.values) if self.only else {}

    def get(self, key: str) -> Any:
        store = self.store
        if store is None:
            return None
        try:
            return store.effective(key)
        except Exception:
            return None

    def summary(self) -> str:
        return run_options_summary(self.get)

    def range_chip_label(self) -> str:
        return range_label(self.get("chapter_range"), self.get("use_spine_order"))

    # ---- building ----------------------------------------------------------------------------------

    def _tile_ctx(self) -> Any:
        if not self.only:
            return self.ctx
        try:
            return dataclasses.replace(self.ctx, store=self.overlay)
        except Exception:
            return self.ctx

    def build_tiles(self) -> list:
        schema = getattr(self.ctx, "schema", None)
        if schema is None or not getattr(schema, "available", False) or getattr(self.ctx, "store", None) is None:
            return [ft.Text("Run options need the settings schema (not available in this session).",
                            theme_style=ft.TextThemeStyle.BODY_SMALL)]
        from glossarion_mobile.ui.settings.tiles import EffectiveConfig, make_tile

        ctx = self._tile_ctx()
        config = EffectiveConfig(ctx.store)
        controls: list = []
        self.tiles = {}
        for name in self.keys:
            try:
                spec = schema.spec(name)
            except Exception:
                spec = None
            if spec is None:
                continue
            try:
                tile = make_tile(spec, ctx, config=config)
                tile.refresh(push=False)
            except Exception:
                log.exception("building the %s run option failed", name)
                continue
            self.tiles[name] = tile
            controls.append(tile.control)
        return controls

    def render(self, push: bool = True) -> None:
        self._gen += 1
        self.body = ft.Container(content=ft.Column(self.build_tiles(), spacing=4, tight=True),
                                 key=f"{self._key}-body-{self._gen}")
        column = self.control.controls[0].content
        column.controls = [self.switch, self.note, self.body]
        self.note.value = ("These values apply to this run only; Settings keep their own values."
                           if self.only else "Changes here are saved to Settings (all runs).")
        self.summary_text.value = self.summary()
        if push:
            self._push(self.control)

    def refresh_summary(self) -> None:
        self.summary_text.value = self.summary()
        for tile in self.tiles.values():
            try:
                tile.refresh(push=False)
            except Exception:
                pass
        self._push(self.control)

    # ---- events ------------------------------------------------------------------------------------

    def _on_switch(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "value", self.switch.value)
        self.only = bool(value)
        self.render()
        self._notify()

    def _overlay_changed(self, _values: dict) -> None:
        self.summary_text.value = self.summary()
        self._push(self.summary_text)
        self._notify()

    def config_changed(self) -> None:
        """A config change (switch off, or a key this run does not override)."""
        self.refresh_summary()

    def _notify(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(dict(self.overlay.values), self.only)
            except Exception:
                log.exception("saving the run options failed")

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass


class ChooseChaptersSheet:
    """"Choose chapters": range + Spine order + the live preview list (see the module docstring).

    ``store``: where the range is written (the RunOptionsPanel's current store); ``preview(range,
    spine)`` (blocking) returns ``plan_model.range_preview``'s dict; ``io`` runs it off the loop."""

    def __init__(self, *, store: Any, path: str, preview: Callable[[str, bool], dict],
                 io: Callable[..., Any], on_applied: Optional[Callable[[], Any]] = None,
                 spine_available: bool = True) -> None:
        self.store = store
        self.path = path
        self.preview_fn = preview
        self.io = io
        self.on_applied = on_applied
        current = str(store.effective("chapter_range") or "") if store is not None else ""
        spine = bool(store.effective("use_spine_order")) if store is not None else False
        self.range_field = ft.TextField(label="Chapter range", hint_text="e.g. 5 or 5-10 (empty: all chapters)",
                                        value=current, dense=True, on_change=lambda e: self._schedule(),
                                        key="chapters-range", border_radius=tokens.RADII["field"])
        self.spine_switch = ft.Switch(label="Spine order (OPF positions)", value=spine, disabled=not spine_available,
                                      on_change=lambda e: self._schedule(), key="chapters-spine")
        self.header = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY, key="chapters-header")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True, visible=False)
        self.legend = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True, visible=False)
        self.rows = ft.Column([], spacing=0, tight=True, key="chapters-rows")
        self.result: Optional[dict] = None
        self._task: Any = None
        self.dialog = scroll_sheet("Choose chapters", [
            self.range_field, self.spine_switch, self.header, self.note, self.rows, self.legend,
        ], actions=[
            ft.TextButton(content="All chapters", on_click=lambda e: self.apply(""), key="chapters-all"),
            ft.FilledButton(content="Apply", on_click=lambda e: self.apply(), key="chapters-apply"),
        ], key="chapters-sheet")
        self._page: Any = None

    def show(self, page: Any) -> "ChooseChaptersSheet":
        self._page = page
        if page is not None:
            page.show_dialog(self.dialog)
        self._schedule(0.0)
        return self

    def close(self) -> None:
        close_dialog(self._page, self.dialog)

    def _schedule(self, delay: float = 0.35) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        try:
            self._task = asyncio.ensure_future(self.refresh(delay))
        except RuntimeError:
            self._task = None

    async def refresh(self, delay: float = 0.0) -> dict:
        if delay:
            await asyncio.sleep(delay)
        text = str(self.range_field.value or "").strip()
        if not text:
            result = {"ok": True, "header": "All chapters will be translated", "rows": [], "note": "", "legend": ""}
        else:
            try:
                result = await self.io(self.preview_fn, text, bool(self.spine_switch.value))
            except Exception as exc:
                result = {"ok": False, "message": f"Could not preview the range: {exc}", "rows": []}
        self.result = result
        self.render(result)
        return result

    def render(self, result: dict) -> None:
        if not result.get("ok"):
            self.header.value = str(result.get("message") or "")
            self.header.color = ft.Colors.ERROR
            self.rows.controls = []
            self.note.visible = self.legend.visible = False
        else:
            self.header.value = str(result.get("header") or "")
            self.header.color = ft.Colors.PRIMARY
            self.note.value = str(result.get("note") or "")
            self.note.visible = bool(self.note.value)
            self.legend.value = str(result.get("legend") or "")
            self.legend.visible = bool(self.legend.value)
            controls = []
            for label, name, skipped in list(result.get("rows") or [])[:500]:
                text = f"{label}  →  {name}" + ("  ⏩ Skipped (special file)" if skipped else "")
                controls.append(ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                        color=ft.Colors.ON_SURFACE_VARIANT if skipped else None))
            self.rows.controls = controls
        for control in (self.header, self.note, self.legend, self.rows):
            try:
                control.update()
            except Exception:
                pass

    def apply(self, text: Optional[str] = None) -> bool:
        value = str(self.range_field.value if text is None else text or "").strip()
        if value:
            from glossarion_mobile.ui.chat.plan_model import parse_range

            if not parse_range(value):
                self.header.value = "Enter a valid chapter range (e.g. 5 or 5-10) first."
                self.header.color = ft.Colors.ERROR
                try:
                    self.header.update()
                except Exception:
                    pass
                return False
        if self.store is not None:
            self.store.set("chapter_range", value)
            self.store.set("use_spine_order", bool(self.spine_switch.value))
        self.close()
        if self.on_applied is not None:
            self.on_applied()
        return True
