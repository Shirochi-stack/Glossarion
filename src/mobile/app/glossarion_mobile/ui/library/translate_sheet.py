"""TranslateSheet (UI_SPEC §3.10): Library-origin "Load N for translation" -> a ``translate`` job.

A ``BottomSheet`` (90%) showing what the run will use - each book's raw source
file, the model, prompt profile, target language and the effective glossary
mode (the chat's ``effective_glossary_label``, the desktop rule) - plus:

* "Review glossary before translating" (off by default: desktop has no gate for
  book-origin jobs; enabled only when the translate job kind supports the pause);
* "Open in chat instead" (one book: a new chat with the raw file attached, via the
  IntentRouter's Translate-in-new-chat handler);
* U9: the chat Plan card's pieces: the glossary chip opens the PlanGlossarySheet (Map glossaries…
  for several EPUBs), "Run options" (``plan_card.RunOptionsPanel``: "Only for this run" values
  become the job's ``config_overrides``) and, for one book, "Choose chapters" (range + Spine order
  + the desktop 🔍 preview);
* **Start**: :func:`confirm_output_root` first (the desktop "Output Folder Mismatch"
  question of Load for translation), then ``LibraryService.translate_spec`` ->
  ``JobsFeature.submit`` (origin = book, so the Book page strip, the JobStrip and Jobs
  show it).

Raw sources are resolved on the io pool before the sheet opens
(:func:`open_translate_sheet`); books without one are listed with the desktop
"missing raw" reason and are not started.

The sheet is for a fresh Translate only: a chat Result card's Resume / Retry failed of a
book that moved into the Library starts the run directly (:func:`resume_translation`).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip, reason_subtitle
from glossarion_mobile.ui.components.sheet import fitted_column, sheet_frame
from glossarion_mobile.ui.components.switch_row import switch_row
from glossarion_mobile.ui.library.models import size_text

__all__ = ["OUTPUT_MISMATCH_TITLE", "TranslateSheet", "confirm_output_root", "open_translate_sheet",
           "resume_translation", "review_gate_supported", "submit_translation"]

log = logging.getLogger("glossarion.library.ui")

MISSING_RAW_REASON = "The raw source file can't be found (see “Scan for raw”)"
OUTPUT_MISMATCH_TITLE = "Output Folder Mismatch"  # desktop _ensure_output_override_matches


async def confirm_output_root(ctx: Any, books: Sequence[Mapping[str, Any]]) -> bool:
    """Desktop ``_ensure_output_override_matches`` (Library "Load for translation"): when a book's
    workspace sits under another output root than the current one, ask "Output Folder Mismatch"
    (the shared prompt text; Cancel / Yes) and, on Yes, switch the root so the run resumes in that
    workspace instead of starting a second one. True when loading can go on (no mismatch, or Yes).

    ``ctx``: a ``LibraryContext`` (``LibraryFeature.context()``): its service's
    ``output_root_mismatch_blocking`` / ``apply_output_override_blocking`` run on the io pool. A
    service without them, or a check that fails, never blocks the run."""
    service = ctx.service
    check = getattr(service, "output_root_mismatch_blocking", None)
    if not callable(check):
        return True
    rows = [dict(book) for book in books or ()]
    adopt = getattr(service, "adopt_workspaces_blocking", None)
    if callable(adopt):
        # Issue 18: a workspace outside the Library's output folder moves (or merges) into it with its
        # glossary, so the run continues there; the question below is left for what could not move
        try:
            adopted = await ctx.io(adopt, rows)
            rows = list(adopted.get("books") or rows)
        except Exception:
            log.debug("moving the workspaces into the output folder failed", exc_info=True)
    try:
        info = await ctx.io(check, rows)
    except Exception:
        log.debug("output root check failed", exc_info=True)
        return True
    if not info:
        return True
    if not str(info.get("new_override") or "") and callable(getattr(service, "apply_output_override_blocking", None)):
        # the books are in the Library's own output folder and only the process root points elsewhere
        # (an earlier switch): back to the Library's folder without asking
        try:
            await ctx.io(service.apply_output_override_blocking, "")
            return True
        except Exception:
            log.debug("switching back to the Library's output folder failed", exc_info=True)
    from glossarion_mobile.ui.tools.common import ask

    answer = await ask(ctx, OUTPUT_MISMATCH_TITLE, str(info.get("prompt") or ""),
                       [("cancel", "Cancel", "text"), ("yes", "Yes", "filled")], key="lib-output-root")
    if answer != "yes":
        return False
    try:
        await ctx.io(service.apply_output_override_blocking, str(info.get("new_override") or ""))
    except Exception as exc:
        ctx.say(f"Could not switch the output folder: {exc}")
        return False
    return True


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


def _settings_ctx() -> Any:
    """The SettingsContext the Run options tiles use (the app's sheet environment), or None."""
    try:
        from glossarion_mobile.ui.sheets.model_sheet import sheet_env

        ctx = sheet_env().ctx
    except Exception:
        return None
    if ctx is None or getattr(ctx, "store", None) is None:
        return None
    return ctx


class TranslateSheet:
    def __init__(self, ctx: Any, books: Sequence[Mapping[str, Any]], sources: Sequence[str],
                 run_config: Optional[Mapping[str, Any]] = None) -> None:
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
                                        subtitle=reason_subtitle(ReasonChip(reason="missing raw",
                                                                            detail=MISSING_RAW_REASON)),
                                        dense=True, key=f"translate-book-{index}"))
        # a tap changes the global value (the chat's ModelSheet); Start reads the model picked last
        self.setting_chips = {tab: ft.Chip(label=ft.Text(""), on_click=lambda e, t=tab: self.choose_setting(t),
                                           key=f"fact-{i}") for i, tab in enumerate(("model", "profile", "language"))}
        self.fact_chips: list = list(self.setting_chips.values())
        self.model_block_chip: Any = None
        self.model_block: Any = None
        self.picked_model = ""
        self._read_settings()
        self.glossary_chip = ft.Chip(label=ft.Text(_glossary_label(cfg)), leading=ft.Icon(ft.Icons.MENU_BOOK, size=16),
                                     on_click=lambda e: self.open_glossary(), key="fact-glossary")
        self.fact_chips.append(self.glossary_chip)
        settings_ctx = _settings_ctx()
        self.run_panel: Any = None
        self.range_chip: Optional[ft.Chip] = None
        run_controls: list[ft.Control] = []
        if settings_ctx is not None and ready:
            from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel

            # ``run_config``: "Only for this run" values the sheet starts switched on with
            self.run_panel = RunOptionsPanel(settings_ctx, values=dict(run_config or {}),
                                             only_this_run=bool(run_config),
                                             on_change=lambda values, only: self._refresh_range(),
                                             key="lib-run-options")
            self.range_chip = ft.Chip(label=ft.Text(self.run_panel.range_chip_label()),
                                      leading=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED, size=16),
                                      on_click=lambda e: self.open_choose_chapters(),
                                      disabled=len(self.ready_sources) != 1, key="fact-range")
            self.fact_chips.append(self.range_chip)
            run_controls.append(self.run_panel.control)
        supported = review_gate_supported()
        self.review_switch = ft.Switch(value=False,
                                       disabled=not supported, key="review-glossary")
        review_row: list[ft.Control] = [switch_row(self.review_switch, "Review glossary before translating")]
        if not supported:
            review_row.append(ReasonChip(reason="Not in this build",
                                         detail="The translate job pauses for glossary review only in the chat "
                                                "in this build; book translations run without the gate, as on "
                                                "desktop."))
        else:  # U9: the job pauses after its glossary phase; the Book page asks Edit · Yes · No
            review_row.append(ft.Text("Pauses after the glossary is generated: Edit · Yes · No on the Book page.",
                                      theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                      key="review-glossary-hint"))
        jobs_ok = service.jobs is not None and service.has_job_kind("translate")
        self.base_reason = None if jobs_ok and ready else (
            "The job service is not running" if not jobs_ok else "No raw source file resolves for the selection")
        self.start_reason = self.base_reason or (self.model_block[0] if self.model_block is not None else None)
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
        self.facts_row = ft.Row(self.fact_chips, wrap=True, spacing=6, run_spacing=6)
        self.start_reason_text = ft.Text(self.start_reason or "", color=ft.Colors.ERROR, visible=bool(self.start_reason),
                                         theme_style=ft.TextThemeStyle.BODY_SMALL, key="start-reason")
        scale = float(getattr(ctx, "text_scale", 1.0) or 1.0)
        content = fitted_column([
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600, key="title"),
            *rows,
            self.facts_row,
            *run_controls,
            ft.Row(review_row, wrap=True),
            self.start_reason_text,
            ft.Row([self.chat_button, self.start_button], alignment=ft.MainAxisAlignment.END, wrap=True),
        ], spacing=tokens.SPACING["sm"], estimate=(360 + 56 * len(rows)) * scale)
        self.sheet = ft.BottomSheet(
            content=sheet_frame(content, padding=tokens.SPACING["sheet_padding"]),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )
        self._page: Any = None

    def _read_settings(self) -> str:
        """The setting chips from the config now, with the excluded-route preflight of the model (U9, chat Send's
        rule and reason). The model: picked last (what the drawer shows and a new chat starts on), else All-chats."""
        cfg = self.ctx.service.cfg
        picked = getattr(self.ctx, "picked_model", None)
        self.picked_model = str(picked() or "") if callable(picked) else ""
        model = self.picked_model or str(cfg("model", "") or "")
        profile = str(cfg("active_profile", "") or "")
        target = str(cfg("output_language", "") or "English")
        self.setting_chips["model"].label = ft.Text(f"Model: {model}" if model else "Model: (not set)")
        self.setting_chips["profile"].label = ft.Text(f"Profile: {profile}" if profile else "Profile: (not set)")
        self.setting_chips["language"].label = ft.Text(f"→ {target}")
        from glossarion_mobile.services.model_catalog import model_block

        self.model_block = model_block(model) if model else None
        if self.model_block_chip in self.fact_chips:
            self.fact_chips.remove(self.model_block_chip)
        self.model_block_chip = None
        if self.model_block is not None:
            self.model_block_chip = ReasonChip(reason=self.model_block[0], detail=self.model_block[1],
                                               key="fact-model-excluded")
            self.fact_chips.insert(1, self.model_block_chip)
        return model

    def choose_setting(self, tab: str) -> Any:
        """A setting chip: the chat's ModelSheet on that tab; the pick is the global value, and shows here."""
        opener = getattr(self.ctx, "choose_setting", None)
        sheet = opener(tab, lambda field_name, value: self._settings_changed()) if callable(opener) else None
        if sheet is None:
            self.ctx.say("These settings can't be changed here in this session")
        return sheet

    def _settings_changed(self) -> None:
        self._read_settings()
        self.start_reason = self.base_reason or (self.model_block[0] if self.model_block is not None else None)
        self.start_button.disabled = self.start_reason is not None
        self.start_reason_text.value = self.start_reason or ""
        self.start_reason_text.visible = bool(self.start_reason)
        self.ctx.push(self.facts_row, self.start_button, self.start_reason_text)

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
        if self.run_panel is not None:
            self.run_panel.release()  # its summary stops following the config

    async def start(self) -> Optional[str]:
        if self.start_reason is not None:
            return None
        if not await confirm_output_root(self.ctx, self.ready_books):
            return None  # Cancel on "Output Folder Mismatch": the sheet stays open
        # The sources were resolved on the io pool when the sheet opened: never re-run the
        # registry reads / directory scans of raw_source() on the UI loop.
        if self.run_panel is not None:
            self.run_panel.commit()  # a number typed and not yet submitted when Start is tapped
        overrides = self.run_panel.config_overrides() if self.run_panel is not None else {}
        if self.picked_model:
            overrides = {"model": self.picked_model, **overrides}
        job_id = await submit_translation(self.ctx, self.ready_books, self.ready_sources,
                                          review_glossary=bool(self.review_switch.value), config_overrides=overrides)
        self.started_job = job_id
        self.close()
        self.ctx.haptic("medium_impact")
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

    # ---- U9: glossary chip, Choose chapters ------------------------------------------------------------

    def _glossary_feature(self) -> Any:
        getter = getattr(self.ctx, "glossary", None)
        return getter() if callable(getter) else None

    def open_glossary(self) -> Any:
        """The glossary chip: the PlanGlossarySheet for this run (Map glossaries… with several EPUBs)."""
        feature = self._glossary_feature()
        if feature is None or not hasattr(feature, "open_plan_glossary_sheet"):
            self.ctx.say("The glossary tools are not available in this session")
            return None
        book = self.ready_books[0] if len(self.ready_books) == 1 else None
        return feature.open_plan_glossary_sheet(book=book, inputs=list(self.ready_sources),
                                                on_changed=self._refresh_glossary)

    def _refresh_glossary(self) -> None:
        self.glossary_chip.label = ft.Text(_glossary_label(self.ctx.service.cfg))
        self.ctx.push(self.glossary_chip)

    def _refresh_range(self) -> None:
        if self.range_chip is not None and self.run_panel is not None:
            self.range_chip.label = ft.Text(self.run_panel.range_chip_label())
            self.ctx.push(self.range_chip)

    def open_choose_chapters(self) -> Any:
        """Choose chapters (one book): range + Spine order + the desktop 🔍 preview, written through the
        Run options' store (this run, or Settings with the switch off)."""
        if self.run_panel is None or len(self.ready_sources) != 1:
            return None
        from glossarion_mobile.ui.chat.plan_card import ChooseChaptersSheet
        from glossarion_mobile.ui.chat.plan_model import range_preview

        panel = self.run_panel
        path = self.ready_sources[0]
        store = panel.store

        def preview(text: str, spine: bool) -> dict:
            snapshot = store.snapshot() if store is not None and hasattr(store, "snapshot") else {}
            return range_preview(snapshot, path, text, spine)

        def applied() -> None:
            panel.refresh_summary()
            self._refresh_range()

        self.chapters_sheet = ChooseChaptersSheet(store=store, path=path, preview=preview, io=self.ctx.io,
                                                  on_applied=applied,
                                                  spine_available=not str(path).lower().endswith(".pdf"))
        self.chapters_sheet.show(self.ctx.page)
        return self.chapters_sheet


async def submit_translation(ctx: Any, books: Sequence[Mapping[str, Any]], sources: Sequence[str], *,
                             review_glossary: bool = False,
                             config_overrides: Optional[Mapping[str, Any]] = None,
                             chat_cid: Optional[str] = None) -> Optional[str]:
    """``LibraryService.translate_spec`` -> ``submit`` (the job service's own preflight, e.g. a missing
    sign-in, says why when it queues nothing): the sheet's Start and :func:`resume_translation`."""
    service = ctx.service
    tied = {"chat_cid": str(chat_cid)} if chat_cid else {}
    spec = service.translate_spec(list(books), review_glossary=review_glossary, sources=list(sources),
                                  config_overrides=dict(config_overrides or {}), **tied)
    return await service.submit(spec)


async def resume_translation(ctx: Any, book: Mapping[str, Any],
                             config_overrides: Optional[Mapping[str, Any]] = None,
                             chat_cid: Optional[str] = None) -> Optional[str]:
    """Resume / Retry failed of a chat book that moved into the Library (issue 4 of the 9.15.7 batch: "clicking
    resume opens a popup instead of just resuming"): the book's translation run starts at once with the
    settings handed over (``config_overrides``: the chat's model / profile / language and the card's Run
    options), and the pipeline continues from the workspace's progress file, like the desktop's resume. No
    sheet: what would block the run is said in one short message instead (no raw source, no job service, a
    model excluded on mobile, Cancel on "Output Folder Mismatch"); the sheet stays for a fresh Translate."""
    service = ctx.service
    book = dict(book)
    name = str(book.get("name") or "this book")
    if service.jobs is None or not service.has_job_kind("translate"):
        ctx.say("The job service is not running")
        return None
    overrides = dict(config_overrides or {})
    model = str(overrides.get("model") or service.cfg("model", "") or "")
    from glossarion_mobile.services.model_catalog import model_block

    block = model_block(model) if model else None
    if block is not None:
        ctx.say(f"Could not resume: {block[0]}")
        return None
    try:
        source = await ctx.io(service.raw_source, book)
    except Exception:
        log.debug("resolving the raw source failed", exc_info=True)
        source = ""
    if not source:
        ctx.say(MISSING_RAW_REASON)
        return None
    if not await confirm_output_root(ctx, [book]):
        return None
    try:
        job_id = await submit_translation(ctx, [book], [source], config_overrides=overrides, chat_cid=chat_cid)
    except Exception as exc:
        ctx.say(f"Could not resume: {exc}")
        return None
    if job_id:
        ctx.haptic("medium_impact")
        ctx.say(f"Resuming “{name}”")
    return job_id


async def open_translate_sheet(ctx: Any, books: Sequence[Mapping[str, Any]],
                               run_config: Optional[Mapping[str, Any]] = None) -> TranslateSheet:
    """Resolve the raw sources on the io pool, then show the sheet (``run_config``: its Run options'
    "Only for this run" values to start with)."""
    service = ctx.service
    books = [dict(b) for b in books]
    sources = await ctx.io(lambda: [service.raw_source(b) for b in books])
    sheet = TranslateSheet(ctx, books, sources, run_config=run_config)
    if ctx.page is not None:
        sheet.show(ctx.page)
    return sheet
