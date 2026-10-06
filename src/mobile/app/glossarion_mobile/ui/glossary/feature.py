"""GlossaryFeature: wires the U6 Glossary Manager into GlossarionApp (UI_SPEC §4.1).

``await GlossaryFeature.install(app)`` - call it in ``GlossarionApp.start`` after
``_install_library()`` (it uses the Library service for book links, the jobs feature for
the glossary jobs and the settings context for the schema tiles):

1. builds the ``GlossaryService`` (app paths, MobileConfigStore, Prefs, FileBridge,
   JobsFeature, LibraryService, io pool) as ``app.glossary``; the feature is
   ``app.glossary_feature``;
2. wraps the shell's screen factory for ``glossary``, ``glossary.detail``,
   ``glossary.unified`` and ``glossary.parallel_pair``, and ``show_sheet`` for the
   ``glossary.entry`` deep link (opens the glossary, then that entry's sheet);
3. hands the Library its glossary hooks (``LibraryService.glossary_hooks = feature``): the
   Book page Glossary tab's Open in editor / Delete glossary files / Restore backup / Load
   as manual glossary / ✨ Refine this / ✨ Refinement, and the Library selection bar's
   Delete glossary files (N) / Restore glossary backup;
4. routes the chat ＋ sheet "Extract glossary" tool to the Extract sheet (with the chat's
   attachment as "This file" when there is one), and gives the chat its table-editor and
   "Add term to glossary" hooks (the Reader's "Add to glossary" calls :meth:`add_term` too).

Cross-screen actions (used by the screens and the hooks) live here: the Extract sheet, the
mode sheet, Use as manual glossary (desktop "Load Glossary" question; the current chat can
use it too), delete / restore glossary files of inputs (desktop "Delete Glossary" /
"Restore Glossary" questions), the refinement sheet and job, the PlanGlossarySheet.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Mapping, Optional, Sequence

from glossarion_mobile.ui.router import RouteMatch

__all__ = ["GlossaryFeature", "IMPLEMENTED_ROUTES", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.glossary")

SCREEN_ROUTES = ("glossary", "glossary.detail", "glossary.unified", "glossary.parallel_pair")
SHEET_ROUTES = ("glossary.entry",)
#: Routes this feature ships (for the drawer / hubs; Integrate merges them).
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES + SHEET_ROUTES)


class GlossaryFeature:
    def __init__(self, app: Any, *, service: Any = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._fallback_sheet: Optional[Callable[[RouteMatch], Any]] = None
        self.screens_built: list = []
        self.listing: list = []
        self.sheets: list = []
        #: Shared by every context this feature builds (scripted answers / asked questions in host tests).
        self.extras: dict = {}
        self.service = service or self._make_service()

    # ---- install ----------------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "GlossaryFeature":
        feature = cls(app, **kwargs)
        feature.attach()
        return feature

    def _make_service(self) -> Any:
        from glossarion_mobile.services.glossary import GlossaryService

        app = self.app
        return GlossaryService(paths=getattr(app, "paths", None), config=getattr(app, "config_store", None),
                               prefs=getattr(app, "prefs", None), files=getattr(app, "files", None),
                               jobs=getattr(app, "jobs", None), library=getattr(app, "library", None),
                               run_io=self.run_io)

    def attach(self) -> None:
        app = self.app
        app.glossary = self.service
        app.glossary_feature = self
        if getattr(self.service, "ask_continue", None) is None:
            self.service.ask_continue = self.ask_continue_blocking
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
            self._fallback_sheet = shell.show_sheet
            shell.show_sheet = self.show_sheet
        library = getattr(app, "library", None)
        if library is not None:
            library.glossary_hooks = self
        chat_view = getattr(app, "chat_view", None)
        if chat_view is not None and hasattr(chat_view, "_on_tool") and not getattr(chat_view, "_glossary_tool", False):
            original = chat_view._on_tool

            def on_tool(tool_id: str) -> Any:
                if tool_id == "extract_glossary":
                    try:
                        chat_view.composer.set_plus_open(False)
                    except Exception:
                        pass
                    return self.open_extract_from_chat()
                return original(tool_id)

            chat_view._on_tool = on_tool
            chat_view._glossary_tool = True
        if chat_view is not None:
            # the approval card's raw editor -> "Open in table editor"; response ⋯ -> Add term to glossary
            chat_view.glossary_table_opener = self.open_editor_for_path
            chat_view.glossary_term_adder = lambda term="", **kw: self.spawn(self.add_term(term, **kw))

    # ---- plumbing ---------------------------------------------------------------------------------

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-glossary")
        return await asyncio.to_thread(fn, *args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def ask_continue_blocking(self, title: str, text: str) -> bool:
        """``GlossaryService.ask_continue``: the desktop "Backup Failed … Continue anyway?" Yes/No.

        Editor actions run on the io pool and the backup step asks in the middle of them, so the
        question is shown on the UI loop and this (io) thread waits for the answer. Without a
        UI loop to ask on (or when called on the loop itself) the answer is No: the destructive
        step is skipped rather than run without its backup.
        """
        from glossarion_mobile.ui.glossary.common import ask

        loop = getattr(self.dispatcher, "loop", None)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or not loop.is_running() or running is loop:
            log.warning("cannot ask %r off the UI loop; answering No", title)
            return False

        async def question() -> bool:
            return await ask(self.context(), title=title, body=text, confirm="Yes", cancel="No", destructive=True)

        try:
            return bool(asyncio.run_coroutine_threadsafe(question(), loop).result(timeout=900))
        except Exception:
            log.exception("asking %r failed", title)
            return False

    def _push_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return
        shell.push_overlay(view)
        try:
            self.page.update()
        except Exception:
            pass

    def _pop_overlay(self) -> None:
        back = getattr(self.app, "back", None)
        if callable(back):
            back()

    def context(self) -> Any:
        from glossarion_mobile.ui import tokens
        from glossarion_mobile.ui.glossary.common import GlossaryContext

        app = self.app
        shell = getattr(app, "shell", None)
        state = getattr(app, "state", None)
        try:
            scale = float(state.text_scale.value) if state is not None else 1.0
        except Exception:
            scale = 1.0
        settings = getattr(app, "settings", None)
        platform = str(getattr(getattr(self.page, "platform", None), "value", None) or "desktop")
        dark = False
        try:
            from glossarion_mobile.ui.theme import is_dark

            dark = bool(is_dark(self.page))
        except Exception:
            dark = False
        library_feature = getattr(app, "library_feature", None)
        ctx = GlossaryContext(
            service=self.service,
            page=self.page,
            dispatcher=self.dispatcher,
            navigate=getattr(app, "navigate_to", None),
            notify=getattr(app, "notify", None),
            files=getattr(app, "files", None),
            jobs=getattr(app, "jobs", None),
            prefs=getattr(app, "prefs", None),
            haptics=getattr(app, "haptics", None),
            shell=shell,
            intents=getattr(app, "intents", None),
            copy_text=getattr(app, "_copy_text", None),
            reader=lambda: getattr(app, "reader", None),
            push_overlay=self._push_overlay,
            pop_overlay=self._pop_overlay,
            foreground=getattr(library_feature, "foreground", lambda: True),
            platform=platform,
            tablet=bool(getattr(shell, "tablet", False)),
            dark=dark,
            text_scale=scale,
            library=getattr(app, "library", None),
            settings=getattr(settings, "ctx", None),
            feature=self,
            extras=self.extras,
        )
        try:
            from glossarion_mobile.ui.theme import mono_family

            ctx.mono = mono_family(self.page)  # type: ignore[attr-defined]
        except Exception:
            ctx.mono = tokens.MONO_FAMILIES.get(platform, "monospace")  # type: ignore[attr-defined]
        return ctx

    # ---- screens -------------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        ctx = self.context()
        name = match.name
        if name == "glossary":
            from glossarion_mobile.ui.glossary.home import GlossariesScreen

            return GlossariesScreen(match, ctx)
        if name == "glossary.detail":
            from glossarion_mobile.ui.glossary.glossary_view import GlossaryScreen

            return GlossaryScreen(match, ctx)
        if name == "glossary.unified":
            from glossarion_mobile.ui.glossary.unified import UnifiedGlossaryScreen

            return UnifiedGlossaryScreen(match, ctx)
        if name == "glossary.parallel_pair":
            from glossarion_mobile.ui.glossary.parallel_pair import ParallelPairScreen

            return ParallelPairScreen(match, ctx)
        return None

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        if match.name in SCREEN_ROUTES:
            try:
                screen = self.make_screen(match)
            except Exception:
                log.exception("building the %s screen failed", match.name)
                screen = None
            if screen is not None:
                self.screens_built.append(match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen

    def show_sheet(self, match: RouteMatch) -> Any:
        """``/glossary/<gid>/entry/<n>``: the glossary view, then the EntrySheet of entry ``n`` (1-based)."""
        if match.name != "glossary.entry":
            return self._fallback_sheet(match) if self._fallback_sheet is not None else None
        gid = match.params.get("gid")
        try:
            number = int(match.params.get("n") or 0)
        except (TypeError, ValueError):
            number = 0
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("glossary.detail", {"gid": gid})
        self.spawn(self._open_entry_when_ready(gid, number))
        return None

    async def _open_entry_when_ready(self, gid: Any, number: int, attempts: int = 100) -> Any:
        for _ in range(attempts):
            screen = getattr(getattr(self.app, "shell", None), "top_screen", None)
            editor = getattr(screen, "editor", None)
            if editor is not None and getattr(screen, "gid", None) == gid and editor.doc is not None:
                spec = next((s for s in editor.specs if s.source_idx == number - 1), None)
                if spec is None:
                    return None
                await editor.jump_to(spec.key)
                return editor.open_entry(spec)
            await asyncio.sleep(0.05)
        return None

    # ---- glossary mode / plan ---------------------------------------------------------------------------

    def open_mode_sheet(self, on_changed: Optional[Callable[[str], Any]] = None) -> Any:
        from glossarion_mobile.ui.glossary.sheets import ModeSheet

        sheet = ModeSheet(self.context(), on_changed=on_changed)
        self.sheets.append(sheet)
        return sheet.show()

    def open_plan_glossary_sheet(self, *, book: Optional[Mapping[str, Any]] = None, effective: str = "",
                                 on_changed: Optional[Callable[[], Any]] = None) -> Any:
        """PlanGlossarySheet for the chat Plan card / the Library TranslateSheet glossary chip."""
        from glossarion_mobile.ui.glossary.sheets import PlanGlossarySheet

        ctx = self.context()

        def done(_value: Any = None) -> None:
            if on_changed is not None:
                on_changed()

        async def load_file() -> None:
            files = ctx.files
            if files is None:
                return
            picked = await files.pick_files(target="inbox", allowed_extensions=["csv", "json", "txt", "md"],
                                            allow_multiple=False, dialog_title="Load Glossary")
            if picked:
                self.service.record_import(picked[0].path)
                await self.use_as_manual(picked[0].path, source_path=self._book_source(book), ask_chat=False)
                done()

        async def use_book() -> None:
            rows = await ctx.io(self.service.glossaries_for_book, book) if book else []
            if not rows:
                ctx.say("This book has no glossary yet")
                return
            await self.use_as_manual(rows[0].path, source_path=self._book_source(book), ask_chat=False)
            done()

        async def clear() -> None:
            await self.clear_manual_glossary()
            done()

        def review() -> None:
            self.spawn(self.open_editor_for_book(book)) if book else ctx.go("glossary")

        sheet = PlanGlossarySheet(ctx, book=book, effective=effective, on_load_file=lambda: self.spawn(load_file()),
                                  on_use_book=(lambda: self.spawn(use_book())) if book else None,
                                  on_clear=lambda: self.spawn(clear()), on_review=review,
                                  on_mode=lambda: self.open_mode_sheet(on_changed=done))
        self.sheets.append(sheet)
        return sheet.show()

    def _book_source(self, book: Optional[Mapping[str, Any]]) -> Optional[str]:
        library = getattr(self.app, "library", None)
        if book is None or library is None:
            return None
        try:
            return library.raw_source(book) or None
        except Exception:
            return None

    # ---- extract -----------------------------------------------------------------------------------------

    def open_extract_sheet(self, source_path: Optional[str] = None, title: Optional[str] = None,
                           origin: Optional[dict] = None) -> Any:
        from glossarion_mobile.ui.glossary.sheets import ExtractSheet

        ctx = self.context()
        sheet = ExtractSheet(ctx, source_path=source_path, title=title,
                             on_submit=lambda inputs, t: self.spawn(self.submit_extract(inputs, t, origin)),
                             on_pair=lambda: ctx.go("glossary.parallel_pair"))
        self.sheets.append(sheet)
        return sheet.show()

    def open_extract_from_chat(self) -> Any:
        """＋ sheet › Extract glossary: the chat's attachment (when there is one) as "This file"."""
        chat_view = getattr(self.app, "chat_view", None)
        record = getattr(getattr(chat_view, "composer", None), "attachment", None) if chat_view is not None else None
        path = str(record.get("path") or "") if isinstance(record, Mapping) else ""
        origin = None
        cid = getattr(chat_view, "cid", None)
        if cid:
            origin = {"type": "chat", "cid": str(cid), "label": "Chat"}
        return self.open_extract_sheet(path if path and os.path.exists(path) else None,
                                       os.path.basename(path) if path else None, origin)

    async def submit_extract(self, inputs: Sequence[str], title: Optional[str] = None,
                             origin: Optional[dict] = None) -> Optional[str]:
        ctx = self.context()
        try:
            spec = self.service.extract_spec(list(inputs), title=title, origin=origin)
            job_id = await self.service.submit(spec)
        except Exception as exc:
            ctx.say(f"Could not start the extraction: {exc}")
            return None
        ctx.say("Extracting the glossary…", "Jobs", lambda: ctx.go("jobs"))
        return job_id

    # ---- manual glossary ------------------------------------------------------------------------------

    async def use_as_manual(self, path: str, *, source_path: Optional[str] = None, ask_chat: bool = True) -> Any:
        """Editor "Load" / "Use as manual glossary": the desktop "Load Glossary" question, then
        ``GlossaryService.load_as_manual`` (manual_glossary_path + Append Glossary; the Manual Glossary Only
        copy to the book's output folder). When a chat is open it can use the glossary instead."""
        from glossarion_mobile.ui.glossary.common import ask

        ctx = self.context()
        chat_view = getattr(self.app, "chat_view", None)
        if ask_chat and chat_view is not None and getattr(chat_view, "bound", False) and getattr(chat_view, "cid", None):
            from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

            loop = asyncio.get_running_loop()
            choice: asyncio.Future = loop.create_future()
            sheet = ActionSheet([
                ActionItem("For the next translation runs", lambda: choice.done() or choice.set_result("runs"),
                           icon="TRANSLATE"),
                ActionItem("For this chat (Force Manual Glossary)", lambda: choice.done() or choice.set_result("chat"),
                           icon="CHAT"),
            ], title="Use as manual glossary", subtitle=os.path.basename(path), tablet=ctx.tablet,
                # Cancel, Android back or an outside tap: nothing is loaded (never awaits forever)
                on_cancel=lambda: choice.done() or choice.set_result(None))
            self.sheets.append(sheet)
            scripted = ctx.extras.get("answers") if isinstance(ctx.extras, dict) else None
            if isinstance(scripted, list) and scripted and isinstance(scripted[0], str):
                choice.set_result(scripted.pop(0))
            else:
                ctx.show(sheet)
            where = await choice
            if where is None:
                return None
            if where == "chat":
                return self.use_for_chat(path)
        mode = self.service.mode()
        text, info = self.service.load_prompt(path, mode)
        if not await ask(ctx, title="Load Glossary", body=f"{text}\n\n{info}", confirm="Yes", cancel="Cancel"):
            return None
        try:
            result = await ctx.io(lambda: self.service.load_as_manual(path, epub_path=source_path))
        except Exception as exc:
            ctx.say(f"Failed to load glossary: {exc}")
            return None
        copied = result.get("copied_to")
        ctx.say(f"\U0001F4D1 Loaded manual glossary: {os.path.basename(path)}" +
                (f" · copied to {os.path.basename(os.path.dirname(copied))}/glossary.csv" if copied else ""))
        return result

    def use_for_chat(self, path: str) -> Optional[str]:
        """The current chat's glossary policy becomes Force Manual Glossary with this file prefilled."""
        chat_view = getattr(self.app, "chat_view", None)
        env = getattr(chat_view, "env", None) if chat_view is not None else None
        chats = getattr(env, "chats", None) if env is not None else None
        cid = getattr(chat_view, "cid", None)
        if chats is None or not cid:
            return None
        try:
            from glossarion_mobile.ui.chat.direct_text_rules import ManualGlossarySource

            chats.set_override(cid, "glossary_override_mode", "manual")
            chats.set_override(cid, "manual_glossary_path", path)
            chat_view.last_manual_glossary = ManualGlossarySource("path", path=path,
                                                                  extension=os.path.splitext(path)[1].lower())
            chat_view.apply_settings_changed()
        except Exception:
            log.exception("using the glossary for the chat failed")
            return None
        self.context().say(f"This chat now uses {os.path.basename(path)} (Force Manual Glossary)")
        return str(cid)

    async def clear_manual_glossary(self) -> bool:
        """Desktop ✕: "Clear Glossary" → manual_glossary_path ''."""
        from glossarion_mobile.ui.glossary.common import ask

        ctx = self.context()
        previous = str(self.service.cfg("manual_glossary_path", "") or "")
        if not previous:
            return False
        if not await ask(ctx, title="Clear Glossary", body=f"Clear the loaded glossary?\n\n{os.path.basename(previous)}",
                         confirm="Yes", cancel="No"):
            return False
        self.service.set_cfg("manual_glossary_path", "")
        self.service.log(f"📑 Cleared glossary: {os.path.basename(previous)}")
        ctx.say(f"Cleared glossary: {os.path.basename(previous)}")
        return True

    # ---- glossary files of inputs (Book page / Library selection bar) ----------------------------------

    async def delete_glossary_files(self, inputs: Sequence[str]) -> Optional[list]:
        """Desktop 🗑️ (``_delete_current_glossary``) for these inputs: the file list, "Delete Glossary"
        question, files moved to ``Backups/<timestamp>/``."""
        from glossarion_mobile.services.glossary import CoreMissing
        from glossarion_mobile.ui.glossary.common import ask

        ctx = self.context()
        inputs = [p for p in inputs if p]
        if not inputs:
            ctx.say("No input file selected.")
            return None
        try:
            plan = await ctx.io(lambda: self.service.delete_plan(inputs))
        except CoreMissing as exc:
            ctx.say(f"Not available in this build ({exc.name})")
            return None
        except Exception as exc:
            ctx.say(f"⚠️ Error deleting glossary: {exc}")
            return None
        if not plan:
            books = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in inputs)
            await ask(ctx, title="Nothing to Delete", body=f"No glossary files found for: {books}", confirm="OK",
                      cancel="Close")
            return []
        if not await ask(ctx, title="Delete Glossary", body=self.service.delete_prompt(plan), confirm="Yes",
                         cancel="No", destructive=True):
            return None
        try:
            deleted = await ctx.io(lambda: self.service.delete_files(plan))
        except Exception as exc:
            ctx.say(f"⚠️ Error deleting glossary: {exc}")
            return None
        ctx.say(f"🗑️ Deleted ({len(deleted)} files backed up)" if deleted else "Nothing was deleted")
        library = getattr(self.app, "library", None)
        if library is not None and hasattr(library, "mark_dirty"):
            library.mark_dirty()
        return deleted

    async def restore_glossary_backup(self, inputs: Sequence[str]) -> Optional[list]:
        """Desktop ↩️ (``_restore_glossary_backup``): the latest backup folder of these inputs, "Restore Glossary"
        question, files copied back."""
        from glossarion_mobile.services.glossary import CoreMissing
        from glossarion_mobile.ui.glossary.common import ask

        ctx = self.context()
        inputs = [p for p in inputs if p]
        try:
            backup_dir, files = await ctx.io(lambda: self.service.latest_backup(inputs))
        except CoreMissing as exc:
            ctx.say(f"Not available in this build ({exc.name})")
            return None
        if not backup_dir or not files:
            ctx.say("No glossary backup found")
            return []
        if not await ask(ctx, title="Restore Glossary", body=self.service.restore_prompt(backup_dir, files),
                         confirm="Yes", cancel="No"):
            return None
        try:
            restored = await ctx.io(lambda: self.service.restore_files(backup_dir, files))
        except Exception as exc:
            ctx.say(f"⚠️ Error restoring glossary: {exc}")
            return None
        ctx.say(f"↩️ Restored from {os.path.basename(backup_dir)}: {len(restored)} file(s)")
        return restored

    # ---- editor / progress navigation ----------------------------------------------------------------------

    async def open_editor_for_book(self, book: Optional[Mapping[str, Any]], *,
                                   glossary_file: Optional[str] = None) -> Optional[str]:
        """Book page › Glossary › Open in editor: the book's glossary (the progress view's file first)."""
        ctx = self.context()
        path = glossary_file if glossary_file and os.path.isfile(glossary_file) else None
        if path is None and book is not None:
            rows = await ctx.io(self.service.glossaries_for_book, book)
            path = rows[0].path if rows else None
        if not path:
            ctx.say("This book has no glossary file yet — extract one first")
            return None
        gid = self.service.gid_for(path)
        ctx.go("glossary.detail", {"gid": gid})
        return gid

    def open_editor_for_path(self, path: str) -> Optional[str]:
        """A glossary file in the Glossary Manager's editor (the chat approval card's table editor)."""
        if not path:
            return None
        gid = self.service.gid_for(path)
        self.context().go("glossary.detail", {"gid": gid})
        return gid

    async def add_term(self, term: str = "", *, book: Optional[Mapping[str, Any]] = None,
                       glossary_path: Optional[str] = None) -> Optional[str]:
        """Reader selection / chat response › Add to glossary: the book's glossary (or ``glossary_path``)
        in the editor, then its new-entry sheet with the raw name filled in (kept once the user Saves).

        On a tablet over a full-screen surface (the Reader) the editor would open in the main area
        behind it, so the new-entry sheet opens over the Reader instead (UI_SPEC §3.11) and its Add
        writes the entry to the file (:meth:`_add_term_in_sheet`)."""
        ctx = self.context()
        path = glossary_path if glossary_path and os.path.isfile(glossary_path) else None
        if path is None and book is not None:
            rows = await ctx.io(self.service.glossaries_for_book, book)
            path = rows[0].path if rows else None
        if not path:
            ctx.say("This book has no glossary file yet — extract one first")
            return None
        gid = self.service.gid_for(path)
        if self._over_fullscreen(ctx):
            return gid if await self._add_term_in_sheet(ctx, path, term) is not None else None
        ctx.go("glossary.detail", {"gid": gid})
        await self._new_entry_when_ready(gid, term)
        return gid

    def _over_fullscreen(self, ctx: Any) -> bool:
        """Tablet with a full-screen View (the Reader) on top: a pushed screen would sit behind it."""
        stack = getattr(getattr(self.app, "shell", None), "stack", None) or []
        return bool(ctx.tablet and stack and getattr(stack[-1], "fullscreen", False))

    async def _add_term_in_sheet(self, ctx: Any, path: str, term: str) -> Any:
        """The new-entry sheet of ``path`` without the editor: the document is read off the UI loop, and the
        sheet's Add appends the row and saves the file (the editor's Save: "before_save" backup, the shared
        writer) off the UI loop. Returns the sheet, or None when the file could not be read."""
        from glossarion_mobile.services.glossary import doc_fields
        from glossarion_mobile.ui.glossary.editor import entry_types, new_entry_form
        from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet

        def read() -> tuple:
            doc = self.service.open_document(path)
            return doc, entry_types(self.service, doc)

        try:
            doc, types = await ctx.io(read)
        except Exception as exc:
            log.info("opening %s for Add to glossary failed: %s", path, exc)
            ctx.say(f"Could not open {os.path.basename(path)}: {exc}")
            return None
        fields, values = new_entry_form(doc, doc_fields(doc), term)
        sheet = EntrySheet(ctx, fields=fields, values=values, types=types, new=True,
                           title=f"Add to {os.path.basename(path)}",
                           on_save=lambda entered: self.spawn(self._save_new_term(ctx, doc, entered)))
        self.sheets.append(sheet)
        return sheet.show()

    async def _save_new_term(self, ctx: Any, doc: Any, values: Mapping[str, Any]) -> Optional[dict]:
        def write() -> dict:
            self.service.add_entry(doc, values)
            # a freshly read document plus one new row: no translated name changed, so no output files
            return self.service.save_edits(doc, update_outputs=False)

        try:
            report = await ctx.io(write)
        except ValueError as exc:
            ctx.say(str(exc))
            return None
        except Exception as exc:
            log.info("adding the entry to %s failed: %s", getattr(doc, "path", ""), exc)
            ctx.say(f"Could not add the entry: {exc}")
            return None
        name = os.path.basename(str(getattr(doc, "path", "") or ""))
        ctx.say(f"Added to {name}" if report.get("saved") else f"The entry was not saved to {name}")
        return report

    async def _new_entry_when_ready(self, gid: Any, term: str, attempts: int = 100) -> Any:
        for _ in range(attempts):
            screen = getattr(getattr(self.app, "shell", None), "top_screen", None)
            editor = getattr(screen, "editor", None)
            if editor is not None and getattr(screen, "gid", None) == gid and editor.doc is not None:
                return editor.open_new_entry(raw_term=term)
            await asyncio.sleep(0.05)
        return None

    def open_progress_for(self, row: Any) -> Optional[str]:
        book = self.service.book_for_glossary(row)
        library = getattr(self.app, "library", None)
        ctx = self.context()
        if book is None or library is None:
            ctx.go("tools.progress.glossary")
            return None
        bid = library.bid_for(book)
        ctx.go("tools.progress.glossary", None, {"out": bid})
        return bid

    def open_progress_for_path(self, path: Optional[str]) -> Optional[str]:
        from glossarion_mobile.services.glossary import GlossaryFile

        if not path:
            self.context().go("tools.progress.glossary")
            return None
        folder = os.path.dirname(path)
        return self.open_progress_for(GlossaryFile(path=path, kind="book", name="", book=os.path.basename(folder),
                                                   folder=folder))

    # ---- refinement -----------------------------------------------------------------------------------------

    def _refine_context(self, book: Mapping[str, Any], view: Any, rows: Sequence[Any] = ()) -> dict:
        """Blocking: glossary file, progress file, active / selected / completed types of a book (the desktop
        ``_find_glossary_for_refinement`` / ``_active_glossary_refinement_types`` /
        ``_normalize_glossary_refinement_selection`` of the Glossary Progress panel)."""
        library = getattr(self.app, "library", None)
        core = library.core if library is not None else self.service.core
        gpc = core.module("glossary_progress_core")
        from glossarion_mobile.ui.library import progress_model as pm

        source = pm._source_for(library, book) if library is not None else ""
        progress_path = getattr(view, "path", None)
        glossary_path = getattr(view, "glossary_file", None)
        active: list = []
        if gpc is not None and hasattr(gpc, "glossary_progress_locator") and library is not None:
            locator = gpc.glossary_progress_locator(pm.make_owner(library), source)
            if not glossary_path:
                glossary_path = locator._find_glossary_for_refinement(source, progress_path)
            active = list(locator._active_glossary_refinement_types())
        keys = [getattr(getattr(r, "raw", None), "key", None) for r in rows if getattr(r, "kind", "") == "refinement"]
        keys = [k for k in keys if k]
        selected = list(active)
        if keys and gpc is not None and hasattr(gpc, "_normalize_glossary_refinement_selection"):
            selected = list(gpc._normalize_glossary_refinement_selection(keys, active))
        completed: list = []
        model = getattr(view, "state", None)
        data = None
        try:
            data = model._current_gp_data() if model is not None else None
        except Exception:
            data = None
        refinement = data.get("refinement", {}) if isinstance(data, dict) else {}
        if isinstance(refinement, dict):
            for key, info in refinement.items():
                if str(key).startswith("type::") and isinstance(info, dict) and str(
                        info.get("status") or "").lower() == "completed":
                    completed.append(str(info.get("entry_type") or str(key).split("::", 1)[1]))
        return {"glossary_path": glossary_path, "progress_path": progress_path, "source_path": source,
                "selected": selected, "completed": completed}

    async def open_refine(self, book: Mapping[str, Any], view: Any, rows: Sequence[Any] = ()) -> Any:
        """Book page Glossary tab: "✨ Refine this" (rows) / "✨ Refinement" (all active types)."""
        ctx = self.context()
        reason = self.service.refine_supported()
        try:
            info = await ctx.io(lambda: self._refine_context(book, view, rows))
        except Exception as exc:
            log.exception("preparing the refinement failed")
            ctx.say(f"Refinement Preview Failed: {exc}")
            return None
        glossary_path = info.get("glossary_path")
        if not glossary_path or not os.path.isfile(str(glossary_path)):
            ctx.say("No saved glossary file was found for this book.")
            return None
        return await self._open_refine_sheet(ctx, str(glossary_path), info, reason, title=str(book.get("name") or ""),
                                             origin=self._book_origin(book))

    def _book_origin(self, book: Mapping[str, Any]) -> Optional[dict]:
        library = getattr(self.app, "library", None)
        if library is None:
            return None
        try:
            return library.origin_for(book)
        except Exception:
            return None

    async def _open_refine_sheet(self, ctx: Any, glossary_path: str, info: Mapping[str, Any],
                                 reason: Optional[str], *, title: str = "", origin: Optional[dict] = None) -> Any:
        from glossarion_mobile.ui.glossary.sheets import RefineSheet

        types = await ctx.io(self.service.refine_types, glossary_path)
        if not any(count for _name, count in types):
            ctx.say("The selected entry type(s) contain no glossary entries.")
            return None

        async def refine(names: list, target: Optional[int]) -> Optional[str]:
            try:
                spec = self.service.refine_spec(glossary_path=glossary_path, progress_path=info.get("progress_path"),
                                                source_path=info.get("source_path"), selected_types=names,
                                                target_chunk_count=target, title=title or None, origin=origin)
                job_id = await self.service.submit(spec)
            except Exception as exc:
                ctx.say(f"Could not start the refinement: {exc}")
                return None
            ctx.say("✨ Refining the glossary…", "Jobs", lambda: ctx.go("jobs"))
            return job_id

        sheet = RefineSheet(ctx, glossary_path=glossary_path, types=types, selected=info.get("selected") or (),
                            completed=info.get("completed") or (), reason=reason,
                            on_refine=lambda names, target: self.spawn(refine(names, target)))
        self.sheets.append(sheet)
        return sheet.show()

    def refine_from_settings(self) -> Any:
        """Settings › Refinement › "✨ Refine a glossary now…": pick a book glossary, then the sheet."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        ctx = self.context()
        rows = [r for r in (self.listing or []) if r.kind in ("book", "output", "manual")]
        if not rows:
            ctx.say("Open the Glossaries list first, or extract a glossary")
            return None

        async def pick(row: Any) -> Any:
            return await self._open_refine_sheet(ctx, row.path, {"progress_path": None, "source_path": None,
                                                                 "selected": [], "completed": []},
                                                 self.service.refine_supported(), title=row.book or row.name)

        sheet = ActionSheet([ActionItem(r.name, lambda r=r: self.spawn(pick(r)), icon="DESCRIPTION") for r in rows[:200]],
                            title="Refine which glossary?", tablet=ctx.tablet)
        ctx.show(sheet)
        return sheet
