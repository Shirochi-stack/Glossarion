"""TranslateSheet (UI_SPEC §3.10): Library-origin "Load N for translation" -> a ``translate`` job.

A ``BottomSheet`` (90%) showing what the run will use - each book's raw source
file, the model, prompt profile, target language and the effective glossary
mode (the chat's ``effective_glossary_label``, the desktop rule) - plus:

* "Review glossary before translating" (off by default: desktop has no gate for
  book-origin jobs; enabled only when the translate job kind supports the pause);
* "Open in chat instead" (one book: a new chat with the raw file attached, via the
  IntentRouter's Translate-in-new-chat handler);
* **Start**: ``LibraryService.translate_spec`` -> ``JobsFeature.submit`` (origin =
  book, so the Book page strip, the JobStrip and Jobs show it).

Raw sources are resolved on the io pool before the sheet opens
(:func:`open_translate_sheet`); books without one are listed with the desktop
"missing raw" reason and are not started.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library.models import size_text

__all__ = ["TranslateSheet", "open_translate_sheet", "review_gate_supported"]

MISSING_RAW_REASON = "The raw source file can't be found (see “Scan for raw”)"


def review_gate_supported() -> bool:
    """Whether the ``translate`` job kind honours ``params["review_glossary"]``."""
    try:
        from glossarion_mobile.job_kinds import translate as translate_kind
    except Exception:
        return False
    return bool(getattr(translate_kind, "SUPPORTS_GLOSSARY_REVIEW", False))


def _glossary_label(config_get: Any) -> str:
    try:
        from glossarion_mobile.ui.chat.direct_text_rules import effective_glossary_label

        return effective_glossary_label("none", config_get, True)
    except Exception:
        mode = str(config_get("auto_glossary_mode", None) or "off")
        return f"Glossary: {mode.replace('_', ' ').title()} (auto)"


class TranslateSheet:
    def __init__(self, ctx: Any, books: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> None:
        self.ctx = ctx
        self.books = [dict(b) for b in books]
        self.sources = list(sources)
        self.started_job: Optional[str] = None
        service = ctx.service
        cfg = service.cfg
        ready = [b for b, s in zip(self.books, self.sources) if s]
        self.ready_books = ready
        self.ready_sources = [s for s in self.sources if s]  # aligned with ready_books
        rows: list[ft.Control] = []
        for index, (book, source) in enumerate(zip(self.books, self.sources)):
            name = str(book.get("name") or os.path.basename(source or ""))
            if source:
                try:
                    meta = f"{os.path.splitext(source)[1].lstrip('.').upper()} · {size_text(os.path.getsize(source))}"
                except OSError:
                    meta = os.path.splitext(source)[1].lstrip(".").upper()
                rows.append(ft.ListTile(leading=ft.Icon(ft.Icons.MENU_BOOK), title=ft.Text(name, max_lines=2),
                                        subtitle=ft.Text(f"{os.path.basename(source)} · {meta}", max_lines=1,
                                                         overflow=ft.TextOverflow.ELLIPSIS),
                                        dense=True, key=f"translate-book-{index}"))
            else:
                rows.append(ft.ListTile(leading=ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ERROR),
                                        title=ft.Text(name, max_lines=2),
                                        trailing=ReasonChip(reason="missing raw", detail=MISSING_RAW_REASON),
                                        dense=True, key=f"translate-book-{index}"))
        model = str(cfg("model", "") or "")
        profile = str(cfg("active_profile", "") or "")
        target = str(cfg("output_language", "") or "English")
        facts = [f"Model: {model}" if model else "Model: (not set)", f"Profile: {profile}" if profile else "",
                 f"→ {target}", _glossary_label(cfg)]
        self.fact_chips = [ft.Chip(label=ft.Text(f), key=f"fact-{i}") for i, f in enumerate(facts) if f]
        supported = review_gate_supported()
        self.review_switch = ft.Switch(label="Review glossary before translating", value=False,
                                       disabled=not supported, key="review-glossary")
        review_row: list[ft.Control] = [self.review_switch]
        if not supported:
            review_row.append(ReasonChip(reason="Not in this build",
                                         detail="The translate job pauses for glossary review only in the chat "
                                                "in this build; book translations run without the gate, as on "
                                                "desktop."))
        jobs_ok = service.jobs is not None and service.has_job_kind("translate")
        self.start_reason = None if jobs_ok and ready else (
            "The job service is not running" if not jobs_ok else "No raw source file resolves for the selection")
        self.start_button = ft.FilledButton(content="Start", icon=ft.Icons.PLAY_ARROW, on_click=self._on_start,
                                            disabled=self.start_reason is not None, key="start")
        chat_handler = self._chat_handler()
        self.chat_reason = None
        if chat_handler is None:
            self.chat_reason = "The chat is not available in this session"
        elif len(ready) != 1:
            self.chat_reason = "One book at a time"
        self.chat_button = ft.TextButton(content="Open in chat instead", icon=ft.Icons.CHAT_BUBBLE_OUTLINE,
                                         on_click=self._on_chat, disabled=self.chat_reason is not None,
                                         tooltip=self.chat_reason, key="open-in-chat")
        title = (f"Load {len(self.books)} for translation" if len(self.books) > 1
                 else f"Translate “{self.books[0].get('name') or ''}”" if self.books else "Translate")
        content = ft.Column([
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600, key="title"),
            *rows,
            ft.Row(self.fact_chips, wrap=True, spacing=6, run_spacing=6),
            ft.Row(review_row, wrap=True),
            ft.Text(self.start_reason or "", color=ft.Colors.ERROR, visible=bool(self.start_reason),
                    theme_style=ft.TextThemeStyle.BODY_SMALL, key="start-reason"),
            ft.Row([self.chat_button, self.start_button], alignment=ft.MainAxisAlignment.END, wrap=True),
        ], tight=True, spacing=tokens.SPACING["sm"], scroll=ft.ScrollMode.AUTO)
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=content, padding=tokens.SPACING["sheet_padding"]),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )
        self._page: Any = None

    def _chat_handler(self) -> Any:
        intents = getattr(self.ctx, "intents", None)
        if intents is None:
            return None
        from glossarion_mobile.services.intents import ACTION_TRANSLATE_NEW_CHAT

        return getattr(intents, "handlers", {}).get(ACTION_TRANSLATE_NEW_CHAT)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    async def start(self) -> Optional[str]:
        if self.start_reason is not None:
            return None
        service = self.ctx.service
        # The sources were resolved on the io pool when the sheet opened: never re-run the
        # registry reads / directory scans of raw_source() on the UI loop.
        spec = service.translate_spec(self.ready_books, review_glossary=bool(self.review_switch.value),
                                      sources=self.ready_sources)
        job_id = await service.submit(spec)
        self.started_job = job_id
        self.close()
        self.ctx.haptic("medium_impact")
        self.ctx.say(f"Translating · {spec.title}", "Jobs", lambda: self.ctx.go("jobs"))
        return job_id

    async def _on_start(self, e: Any = None) -> Optional[str]:
        try:
            return await self.start()
        except Exception as exc:
            self.ctx.say(f"Could not start: {exc}")
            return None

    def open_in_chat(self) -> Any:
        handler = self._chat_handler()
        if handler is None or self.chat_reason is not None:
            return None
        source = next(s for s in self.sources if s)
        imp = SimpleNamespace(imported=SimpleNamespace(path=source, name=os.path.basename(source)))
        self.close()
        return handler(imp)

    def _on_chat(self, e: Any = None) -> Any:
        return self.open_in_chat()


async def open_translate_sheet(ctx: Any, books: Sequence[Mapping[str, Any]]) -> TranslateSheet:
    """Resolve the raw sources on the io pool, then show the sheet."""
    service = ctx.service
    books = [dict(b) for b in books]
    sources = await ctx.io(lambda: [service.raw_source(b) for b in books])
    sheet = TranslateSheet(ctx, books, sources)
    if ctx.page is not None:
        sheet.show(ctx.page)
    return sheet
