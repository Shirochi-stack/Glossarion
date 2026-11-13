"""Relations Tracker on Glossary › General (UI_SPEC §4.1; desktop GlossaryManager_GUI "Enable Relations Tracker").

The toggle and the "Earlier relations shown" dropdown are schema tiles (``glossary.general``); turning
the toggle on / off adds / removes the relations column in Custom Fields in the same store write
(``state.setting_writes.relations_tracker_writes``). Under the toggle this module adds:

* the prompt check (desktop ``_check_relations_prompts``): when the toggle is turned on, each active
  Balanced/Full glossary prompt, and the Minimal one while "Include Relations Column" is on (turning
  that on with the tracker on checks it too), without ``{relations_instruction}``
  (``relations_tracking.prompts_missing_placeholder``) is offered a reset to default (``ask``). A prompt
  whose active profile is the built-in On-device profile is not offered one, as on desktop;
* the warning notice while the toggle is on and a prompt still lacks the line: which prompt, plus the
  line itself, selectable, with a Copy button (``components.copy_line``). It is recomputed when the tab
  is built and whenever the toggle or a prompt changes;
* "View diagram…" (``relations_diagram.RelationsDiagram``) for the open glossary, else a glossary
  that has a relations tracker.

Every rule is ``relations_tracking``'s (the shared module); nothing here re-implements it.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.copy_line import copy_line
from glossarion_mobile.ui.glossary.common import ask, quiet
from glossarion_mobile.ui.settings.banners import notice

__all__ = ["MINIMAL_KEY", "PROMPT_KEYS", "PROMPT_NAMES", "RelationsPanel", "missing_prompts", "open_diagram", "relations_core",
           "reset_prompt_to_default", "tracker_glossaries"]

log = logging.getLogger("glossarion.glossary.ui")

TOGGLE_KEY = "glossary_enable_relations_tracker"
#: Minimal Generation's "Include Relations Column": the Minimal prompt is checked only with it on
MINIMAL_KEY = "glossary_minimal_include_relations"
#: bucket -> the current prompt key runs read (``glossary_document.glossary_prompt_profile_meta``)
PROMPT_KEYS = {"balanced_full": "manual_glossary_prompt3", "minimal": "unified_auto_glosary_prompt3"}
PROMPT_NAMES = {"balanced_full": "Balanced/Full", "minimal": "Minimal"}
#: the built-in glossary prompt profile for small on-device models (never offered a reset)
ON_DEVICE_PROFILE = "On-device"
PLACEHOLDER_LINE = "- {relations_instruction}"


def relations_core(ctx: Any = None) -> Any:
    """The shared ``relations_tracking`` module (through the service's SharedCore when there is one)."""
    core = getattr(getattr(ctx, "service", None), "core", None)
    module = getattr(core, "module", None)
    if callable(module):
        found = module("relations_tracking")
        if found is not None:
            return found
    try:
        return importlib.import_module("relations_tracking")
    except Exception:
        log.debug("relations_tracking unavailable", exc_info=True)
        return None


def _store(ctx: Any) -> Any:
    return getattr(getattr(ctx, "settings", None), "store", None)


def _default_prompt(ctx: Any, key: str, *, allow_import: bool = False) -> str:
    """The prompt's built-in default; '' while its module is not imported yet (``allow_import``: import it,
    which can take seconds, so only off the UI loop)."""
    schema = getattr(getattr(ctx, "settings", None), "schema", None)
    if schema is None:
        return ""
    try:
        try:
            value = schema.effective_default(key, allow_import=allow_import)
        except TypeError:  # settings_schema itself (no lazy-import guard)
            value = schema.effective_default(key)
    except Exception:
        return ""
    return str(value or "")


def _get(ctx: Any, key: str, default: Any = None) -> Any:
    store = _store(ctx)
    if store is not None:
        return store.get(key, default)
    cfg = getattr(ctx, "cfg", None)
    return cfg(key, default) if callable(cfg) else default


def prompt_texts(ctx: Any) -> dict:
    """``{bucket: prompt}``: the prompt each run sends (the current prompt key, else the built-in default). A
    default that is not loaded yet is left out: the shipped defaults hold the placeholder."""
    texts: dict = {}
    for bucket, key in PROMPT_KEYS.items():
        text = str(_get(ctx, key) or "") or _default_prompt(ctx, key)
        if text:
            texts[bucket] = text
    return texts


def missing_prompts(ctx: Any) -> list:
    """The buckets whose prompt lacks ``{relations_instruction}`` (``prompts_missing_placeholder``); Minimal
    only while its "Include Relations Column" is on (``prompts_to_check``)."""
    core = relations_core(ctx)
    if core is None:
        return []
    texts = core.prompts_to_check(prompt_texts(ctx), {MINIMAL_KEY: bool(_get(ctx, MINIMAL_KEY, False))})
    return list(core.prompts_missing_placeholder(texts))


def _on_device_active(ctx: Any, bucket: str) -> bool:
    active = _get(ctx, "active_glossary_prompt_profiles")
    return isinstance(active, dict) and active.get(bucket) == ON_DEVICE_PROFILE


def reset_prompt_to_default(ctx: Any, bucket: str, default: Optional[str] = None) -> bool:
    """The prompt's Reset to default (desktop ``_reset_glossary_prompt_to_default``): the active profile of
    the bucket gets the default prompt through the shared profile row (``save_prompt_profile_text``)."""
    key = PROMPT_KEYS[bucket]
    default = default or _default_prompt(ctx, key, allow_import=True)
    if not default:
        return False
    service = getattr(ctx, "service", None)
    try:
        _names, selected = service.prompt_profile_choice(bucket, default_text=default)
        error = service.save_prompt_profile_text(bucket, selected, default, default_text=default)
    except Exception:
        log.debug("resetting the %s prompt through its profile row failed", bucket, exc_info=True)
        error = "no profile row"
    if error:
        store = _store(ctx)
        if store is None:
            return False
        store.set(key, default)
    return True


def tracker_glossaries(ctx: Any) -> list:
    """``(path, name)`` of every listed glossary that has a relations tracker (blocking: run on the io pool)."""
    core = relations_core(ctx)
    service = getattr(ctx, "service", None)
    if core is None or service is None:
        return []
    out: list = []
    try:
        rows = list(service.list_glossaries())
    except Exception:
        log.debug("listing the glossaries failed", exc_info=True)
        return []
    for row in rows:
        path = str(getattr(row, "path", "") or "")
        tracker = core.tracker_path_for_glossary(path) if path else ""
        if tracker and os.path.isfile(tracker):
            out.append((path, str(getattr(row, "name", "") or os.path.basename(path))))
    return out


async def open_diagram(ctx: Any, path: Optional[str] = None) -> Any:
    """The relationship diagram of ``path``; without one, of the only glossary with a tracker, or the one
    picked from a sheet (an empty diagram when no glossary has one yet)."""
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    if not path:
        found = await ctx.io(tracker_glossaries, ctx)
        if len(found) > 1:
            return pick_glossary(ctx, found)
        path = found[0][0] if found else None
    return await RelationsDiagram.open(ctx, path)


def pick_glossary(ctx: Any, found: list) -> Any:
    """A sheet listing the glossaries with a relations tracker; a tap opens that one's diagram."""
    from glossarion_mobile.ui.components.dialogs import close_dialog
    from glossarion_mobile.ui.glossary.common import sheet

    holder: dict = {}

    def choose(path: str) -> None:
        close_dialog(getattr(ctx, "page", None), holder.get("sheet"))
        ctx.spawn(open_diagram(ctx, path))

    rows = [ft.ListTile(title=ft.Text(name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                        leading=ft.Icon(ft.Icons.ACCOUNT_TREE_OUTLINED), on_click=lambda e, p=path: choose(p),
                        key=f"rel-pick-{index}")
            for index, (path, name) in enumerate(found)]
    holder["sheet"] = sheet("Relationship diagram", rows, key="rel-pick-sheet")
    page = getattr(ctx, "page", None)
    if page is not None:
        page.show_dialog(holder["sheet"])
    return holder["sheet"]


class RelationsPanel:
    """The rows under the Enable Relations Tracker toggle: the missing-line notice and View diagram…"""

    def __init__(self, ctx: Any, *, glossary_path: Optional[Callable[[], Optional[str]]] = None) -> None:
        self.ctx = ctx
        self.glossary_path = glossary_path
        self.notice = notice("", role="warning", icon="WARNING_AMBER", key="rel-notice")
        self.notice.margin = None
        self.notice_text: ft.Text = self.notice.content.controls[1]
        self.copy_row = copy_line(PLACEHOLDER_LINE, on_copy=getattr(ctx, "copy_text", None), key="rel-line")
        self.warning = ft.Column([self.notice, self.copy_row], spacing=4, tight=True, visible=False,
                                 key="rel-warning")
        self.diagram_button = ft.TextButton(content="View diagram…", icon=ft.Icons.ACCOUNT_TREE_OUTLINED,
                                            key="rel-diagram", on_click=quiet(self.open_diagram))
        self.control = ft.Container(content=ft.Column([self.warning, self.diagram_button], spacing=4, tight=True),
                                    padding=ft.Padding.only(left=12, right=4, bottom=4), key="rel-panel")
        self.missing: list = []
        self.checks = 0  # prompt checks run (the toggle turned on)
        self.check_task: Any = None  # the running prompt check
        self._was_on = bool(_get(ctx, TOGGLE_KEY))
        self._minimal_was_on = bool(_get(ctx, MINIMAL_KEY))
        self._unsub: Any = None
        self._checking = False
        self.refresh(push=False)

    @property
    def enabled(self) -> bool:
        return bool(_get(self.ctx, TOGGLE_KEY))

    def refresh(self, push: bool = True) -> list:
        """Recompute the notice (shown while the toggle is on and a prompt lacks the line)."""
        on = self.enabled
        self.missing = missing_prompts(self.ctx) if on else []
        if self.missing:
            names = [PROMPT_NAMES.get(bucket, bucket) for bucket in self.missing]
            which = (f"The {names[0]} prompt is" if len(names) == 1 else f"The {' and '.join(names)} prompts are")
            them = "it" if len(names) == 1 else "them"
            self.notice_text.value = (f"{which} missing the relations line, so the AI will not return relations. "
                                      f"Add this line to {them}:")
        self.warning.visible = bool(self.missing)
        self.diagram_button.disabled = not on
        if push:
            self.ctx.push(self.control)
        return list(self.missing)

    async def check_prompts(self) -> list:
        """Offer, in one question, a reset to default for the prompts without the line; returns the ones
        still missing."""
        if self._checking:
            return list(self.missing)
        self._checking = True
        self.checks += 1
        try:
            # the built-in On-device profile stays as it is: the notice only
            buckets = [b for b in missing_prompts(self.ctx) if not _on_device_active(self.ctx, b)]
            core = relations_core(self.ctx)
            if buckets and core is not None:
                names = [PROMPT_NAMES.get(bucket, bucket) for bucket in buckets]
                if await ask(self.ctx, title="Relations Tracker", body=core.reset_question(names),
                             confirm="Reset", cancel="Keep it"):
                    for bucket in buckets:
                        # the built-in text may need its module imported (seconds): off the UI loop
                        key = PROMPT_KEYS[bucket]
                        default = await self.ctx.io(lambda k=key: _default_prompt(self.ctx, k, allow_import=True))
                        reset_prompt_to_default(self.ctx, bucket, default)
        finally:
            self._checking = False
        return self.refresh()

    def on_change(self, key: str, value: Any) -> None:
        if key == TOGGLE_KEY:
            on = bool(self.enabled)
            turned_on = on and not self._was_on
            self._was_on = on
            if turned_on:
                self.refresh()
                self.check_task = self.ctx.spawn(self.check_prompts())
                return
        if key == MINIMAL_KEY:
            minimal_on = bool(value)
            turned_on = minimal_on and not self._minimal_was_on
            self._minimal_was_on = minimal_on
            if turned_on and self.enabled:
                self.refresh()
                self.check_task = self.ctx.spawn(self.check_prompts())
                return
        self.refresh()

    def attach(self) -> None:
        store = _store(self.ctx)
        self.refresh()
        if store is None or self._unsub is not None:
            return
        self._was_on = self.enabled
        self._minimal_was_on = bool(_get(self.ctx, MINIMAL_KEY))
        try:
            self._unsub = store.observe_keys((TOGGLE_KEY, MINIMAL_KEY, *PROMPT_KEYS.values()),
                                             lambda key, value: self.ctx.post_ui(self.on_change, key, value))
        except Exception:
            log.debug("observing the relations settings failed", exc_info=True)
            self._unsub = None

    def detach(self) -> None:
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None

    def open_diagram(self) -> Any:
        path = self.glossary_path() if callable(self.glossary_path) else None
        return self.ctx.spawn(open_diagram(self.ctx, path))
