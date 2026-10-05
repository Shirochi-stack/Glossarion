"""PromptEditor for profiles and prompts (UI_SPEC §4.14, §5.5).

Extends the U2 settings ``PromptEditor`` (``ui/settings/editors.py``) instead of
writing a second editor:

* ``PromptEditorPane`` - the in-page editor of the profile and prefill screens:
  full-height mono ``TextField``, placeholder chips (``{target_lang}``,
  ``{split_marker_instruction}``: tapping one inserts it at the cursor) and a token
  count. Counting uses the chat composer's tokenizer
  (``direct_text_rules.count_tokens`` = the shared ``direct_text_stream`` counter:
  tiktoken for the selected model -> o200k_base -> cl100k_base), debounced and run
  off the UI loop; the char / 4 estimate of the U2 editor shows until it returns.
* ``PromptEditorSheet`` - the U2 full-screen ``PromptEditor`` sheet plus the same
  chips and token count (used by the "All prompts" index when a prompt is edited
  in place).

``insert_placeholder`` is the pure insertion rule (host-tested).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.settings.editors import PromptEditor, count_label

__all__ = [
    "PLACEHOLDERS",
    "PromptEditorPane",
    "PromptEditorSheet",
    "TokenCounter",
    "insert_placeholder",
    "token_label",
]

log = logging.getLogger("glossarion.prompts")

#: Placeholders the translation prompts understand (``TransateKRtoEN`` / ``run_env``).
PLACEHOLDERS = ("{target_lang}", "{split_marker_instruction}")

COUNT_DEBOUNCE = 0.45  # seconds (composer draft autosave cadence)


def token_label(text: str, tokens_count: Optional[int]) -> str:
    """``"1,234 chars · 309 tokens"`` (exact) or the U2 ``"… · ≈ N tokens"`` estimate while counting."""
    if tokens_count is None:
        return count_label(text)
    return f"{len(text or ''):,} chars · {int(tokens_count):,} tokens"


def insert_placeholder(text: str, placeholder: str, start: Optional[int] = None,
                       end: Optional[int] = None) -> tuple[str, int]:
    """Insert ``placeholder`` replacing ``text[start:end]`` (cursor/selection); append when unknown.

    Returns ``(new_text, cursor)``. An appended placeholder goes on its own line when the
    text does not already end with whitespace (prompts are line-oriented).
    """
    text = str(text or "")
    if start is None or start < 0 or start > len(text):
        joiner = "" if not text or text[-1].isspace() else "\n"
        new = f"{text}{joiner}{placeholder}"
        return new, len(new)
    if end is None or end < start or end > len(text):
        end = start
    new = text[:start] + placeholder + text[end:]
    return new, start + len(placeholder)


class TokenCounter:
    """Debounced token counts off the UI loop; ``on_count(label)`` gets the new label on the loop."""

    def __init__(
        self,
        *,
        run_io: Optional[Callable[..., Any]] = None,
        model: Callable[[], str] = lambda: "",
        on_count: Optional[Callable[[str], Any]] = None,
        delay: float = COUNT_DEBOUNCE,
        counter: Optional[Callable[[str, str], int]] = None,
    ) -> None:
        self.run_io = run_io
        self.model = model
        self.on_count = on_count
        self.delay = delay
        self._counter = counter
        self._task: Any = None
        self._generation = 0
        self.last: Optional[int] = None

    def _count(self, text: str, model: str) -> int:
        if self._counter is not None:
            return int(self._counter(text, model) or 0)
        from glossarion_mobile.ui.chat.direct_text_rules import count_tokens

        return count_tokens(text, model)

    async def count_now(self, text: str) -> Optional[int]:
        model = ""
        try:
            model = str(self.model() or "")
        except Exception:
            pass
        try:
            if self.run_io is not None:
                value = await self.run_io(self._count, text, model)
            else:
                value = await asyncio.to_thread(self._count, text, model)
        except Exception as exc:
            log.debug("token count failed: %s", exc)
            return None
        self.last = int(value)
        return self.last

    def schedule(self, text: str) -> None:
        self._generation += 1
        generation = self._generation
        if self._task is not None and not self._task.done():
            self._task.cancel()

        async def run() -> None:
            await asyncio.sleep(self.delay)
            value = await self.count_now(text)
            if generation == self._generation and self.on_count is not None:
                self.on_count(token_label(text, value))

        try:
            self._task = asyncio.ensure_future(run())
        except RuntimeError:  # no running loop (tests)
            self._task = None

    def cancel(self) -> None:
        self._generation += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()


class PromptEditorPane(ft.Column):
    """In-page prompt editor: mono field + placeholder chips + token count."""

    def __init__(
        self,
        *,
        value: str = "",
        placeholders: Sequence[str] = PLACEHOLDERS,
        mono: str = "monospace",
        hint: str = "",
        run_io: Optional[Callable[..., Any]] = None,
        model: Callable[[], str] = lambda: "",
        on_change: Optional[Callable[[str], Any]] = None,
        counter: Optional[Callable[[str, str], int]] = None,
        min_lines: int = 14,
        key: Optional[str] = None,
    ) -> None:
        super().__init__(spacing=tokens.SPACING["sm"], expand=True, key=key)
        self.initial = str(value or "")
        self.on_text_change = on_change
        self.field = ft.TextField(
            value=self.initial,
            multiline=True,
            min_lines=min_lines,
            expand=True,
            hint_text=hint or None,
            color=ft.Colors.ON_SURFACE,
            text_style=ft.TextStyle(font_family=mono, size=13),
            border_radius=tokens.RADII["field"],
            on_change=self._on_change,
        )
        self.counter_text = ft.Text(count_label(self.initial), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT)
        self.tokens = TokenCounter(run_io=run_io, model=model, on_count=self._on_count, counter=counter)
        self.chips = [
            ft.Chip(label=ft.Text(p, font_family=mono, size=12), on_click=lambda e, p=p: self.insert(p),
                    key=f"placeholder-{p.strip('{}')}")
            for p in placeholders
        ]
        rows: list[ft.Control] = []
        if self.chips:
            rows.append(ft.Row(self.chips, wrap=True, spacing=6, run_spacing=4))
        rows.append(ft.Container(content=self.field, expand=True))
        rows.append(self.counter_text)
        self.controls = rows

    # ---- value ------------------------------------------------------------------------------

    @property
    def value(self) -> str:
        return self.field.value or ""

    def set_value(self, text: str, *, initial: bool = True) -> None:
        self.field.value = str(text or "")
        if initial:
            self.initial = self.field.value
        self.counter_text.value = count_label(self.field.value)
        self._push(self.field, self.counter_text)
        self.tokens.schedule(self.field.value)

    @property
    def dirty(self) -> bool:
        return self.value != self.initial

    def mark_saved(self) -> None:
        self.initial = self.value

    # ---- events ----------------------------------------------------------------------------

    def did_mount(self) -> None:
        super().did_mount()
        self.tokens.schedule(self.value)

    def will_unmount(self) -> None:
        self.tokens.cancel()
        super().will_unmount()

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    def _on_change(self, e: Any = None) -> None:
        text = self.value
        self.counter_text.value = count_label(text)
        self._push(self.counter_text)
        self.tokens.schedule(text)
        if self.on_text_change is not None:
            self.on_text_change(text)

    def _on_count(self, label: str) -> None:
        self.counter_text.value = label
        self._push(self.counter_text)

    def insert(self, placeholder: str) -> str:
        selection = getattr(self.field, "selection", None)
        start = getattr(selection, "base_offset", None) if selection is not None else None
        end = getattr(selection, "extent_offset", None) if selection is not None else None
        if start is not None and end is not None and end < start:
            start, end = end, start
        text, cursor = insert_placeholder(self.value, placeholder, start, end)
        self.field.value = text
        try:
            self.field.selection = ft.TextSelection(base_offset=cursor, extent_offset=cursor)
        except Exception:
            pass
        self._on_change()
        self._push(self.field)
        return text


class PromptEditorSheet(PromptEditor):
    """The U2 full-screen ``PromptEditor`` with placeholder chips and a real token count."""

    def __init__(self, ctx: Any, *, title: str, value: str, default: Optional[str] = None,
                 on_save: Optional[Callable[[Any], Optional[str]]] = None, subtitle: Optional[str] = None,
                 placeholders: Sequence[str] = PLACEHOLDERS, model: Callable[[], str] = lambda: "",
                 counter: Optional[Callable[[str, str], int]] = None) -> None:
        self.placeholders = tuple(placeholders)
        self._model = model
        self._counter_fn = counter
        super().__init__(ctx, title=title, value=value, default=default, on_save=on_save, subtitle=subtitle)

    def build_body(self) -> ft.Control:
        body = super().build_body()
        self.tokens = TokenCounter(run_io=getattr(self.ctx, "run_io", None), model=self._model,
                                   on_count=self._on_count, counter=self._counter_fn)
        chips = [ft.Chip(label=ft.Text(p, size=12), on_click=lambda e, p=p: self.insert(p)) for p in self.placeholders]
        if not chips:
            return body
        return ft.Column([ft.Row(chips, wrap=True, spacing=6, run_spacing=4), body], spacing=6, expand=True)

    def _on_change(self, e: Any = None) -> None:
        super()._on_change(e)
        self.tokens.schedule(self.field.value or "")

    def _on_count(self, label: str) -> None:
        self.counter.value = label
        self.ctx.push(self.counter)

    def insert(self, placeholder: str) -> str:
        text, _cursor = insert_placeholder(self.field.value or "", placeholder)
        self.field.value = text
        self._on_change()
        self.ctx.push(self.field)
        return text
