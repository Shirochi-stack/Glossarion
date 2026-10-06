"""Text editor (``/tools/text/<fid>?hit=<n>``, UI_SPEC §4.10).

Replaces the desktop "Edit in Notepad" / "Edit File (find QA issue)" / Notepad++ jumps for
files inside the app's storage roots (Output, Library, Inbox, chat workspaces). The route
carries only the file's opaque ``fid`` (Prefs FileRef registry) and the hit number; the
search term and the read-only flag arrive in-process (``request_open``), never in the URL.

* Editor: ``flet_code_editor.CodeEditor`` with the language chosen by extension (the UI_SPEC
  table: html/xhtml/xml/sdlxliff -> XML, css -> CSS, json -> JSON, md -> MARKDOWN, csv/txt/
  srt/ass -> PLAINTEXT) when the package is in the build; otherwise a monospace multiline
  ``TextField`` (the spec's fallback).
* Find: a term field with ▲ / ▼ steppers and an "i/n" counter; the ``hit``-th match of the
  requested term is selected when the file opens (TextField selection).
* Save writes back atomically (temporary file + ``os.replace``) with the file's own encoding
  details: a UTF-8 BOM and CRLF line endings are kept. Files that are not UTF-8 or larger than
  ``EDIT_LIMIT`` open read-only. Unsaved edits ask before the screen is left (Save / Discard /
  Cancel): by Back (Android back, the iOS swipe, the app-bar arrow, the tablet main-area back:
  ``handle_back``) and through the shell's leave guard (``confirm_leave``) for any other navigation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["EDIT_LIMIT", "EditorRequest", "LANGUAGE_BY_EXTENSION", "LoadedText", "TextEditorScreen",
           "code_editor_available", "editor_language", "find_matches", "load_text", "open_text_editor",
           "request_open", "save_text", "take_request"]

log = logging.getLogger("glossarion.tools.text")

#: UI_SPEC §4.10: the code editor has no HTML or CSV language.
LANGUAGE_BY_EXTENSION = {
    "html": "XML", "xhtml": "XML", "htm": "XML", "xml": "XML", "sdlxliff": "XML", "opf": "XML", "ncx": "XML",
    "css": "CSS", "json": "JSON", "md": "MARKDOWN",
    "csv": "PLAINTEXT", "txt": "PLAINTEXT", "srt": "PLAINTEXT", "ass": "PLAINTEXT", "vtt": "PLAINTEXT",
    "lrc": "PLAINTEXT", "log": "PLAINTEXT",
}
TEXT_EXTENSIONS = frozenset(LANGUAGE_BY_EXTENSION)
EDIT_LIMIT = 4 * 1024 * 1024  # larger files open read-only (a TextField this large is not editable)
READ_LIMIT = 16 * 1024 * 1024
NOT_UTF8 = "This file is not UTF-8 text: it opened read-only."
TOO_LARGE = "This file is too large to edit on the device: it opened read-only."
OUTSIDE = "This file is outside the app's storage"


@dataclass
class EditorRequest:
    find: Optional[str] = None
    read_only: bool = False


_REQUESTS: dict = {}
_LOCK = threading.Lock()


def request_open(fid: str, *, find: Optional[str] = None, read_only: bool = False) -> None:
    """In-process options for the next editor opened on ``fid`` (the term never enters the route)."""
    with _LOCK:
        _REQUESTS[str(fid)] = EditorRequest(find=find or None, read_only=bool(read_only))


def take_request(fid: Optional[str]) -> EditorRequest:
    with _LOCK:
        return _REQUESTS.pop(str(fid or ""), None) or EditorRequest()


def open_text_editor(ctx: Any, path: str, *, find: Optional[str] = None, read_only: bool = False,
                     hit: int = 1) -> Optional[str]:
    """Open ``path`` in the editor route (``ctx`` needs ``prefs.file_ref`` and ``go``/``navigate``)."""
    prefs = getattr(ctx, "prefs", None)
    if prefs is None or not path:
        return None
    fid = prefs.file_ref(path)
    request_open(fid, find=find, read_only=read_only)
    query = {"hit": max(1, int(hit or 1))} if find else None
    go = getattr(ctx, "go", None)
    if callable(go):
        go("tools.text", {"fid": fid}, query)
    else:
        navigate = getattr(ctx, "navigate", None)
        if callable(navigate):
            navigate("tools.text", {"fid": fid}, query)
    return fid


def editor_language(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower().lstrip(".")
    return LANGUAGE_BY_EXTENSION.get(ext, "PLAINTEXT")


def is_text_file(path: str) -> bool:
    return os.path.splitext(str(path or ""))[1].lower().lstrip(".") in TEXT_EXTENSIONS


def code_editor_available() -> bool:
    try:
        import flet_code_editor  # noqa: F401
    except Exception:
        return False
    return True


def find_matches(text: str, term: str) -> list:
    """Start offsets of ``term`` in ``text`` (case-insensitive, non-overlapping)."""
    if not term:
        return []
    hay, needle = text.casefold(), term.casefold()
    out, start = [], 0
    while True:
        index = hay.find(needle, start)
        if index < 0:
            return out
        out.append(index)
        start = index + max(1, len(needle))


@dataclass(frozen=True)
class LoadedText:
    text: str
    bom: bool
    crlf: bool
    size: int
    read_only_reason: Optional[str] = None


def load_text(path: str) -> LoadedText:
    """Blocking: the file as text (``\\n`` lines) plus what saving must restore."""
    size = os.path.getsize(path)
    if size > READ_LIMIT:
        raise ValueError(f"The file is larger than {READ_LIMIT // (1024 * 1024)} MB")
    with open(path, "rb") as handle:
        data = handle.read()
    bom = data.startswith(b"\xef\xbb\xbf")
    reason = None
    try:
        text = data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        reason = NOT_UTF8
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    if reason is None and size > EDIT_LIMIT:
        reason = TOO_LARGE
    return LoadedText(text=text, bom=bom, crlf=crlf, size=size, read_only_reason=reason)


def save_text(path: str, text: str, *, bom: bool = False, crlf: bool = False) -> int:
    """Blocking: atomic write-back keeping the BOM and the line endings; returns the bytes written."""
    body = str(text).replace("\r\n", "\n")
    if crlf:
        body = body.replace("\n", "\r\n")
    payload = (b"\xef\xbb\xbf" if bom else b"") + body.encode("utf-8")
    tmp = f"{path}.glossarion-edit.tmp"
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    return len(payload)


def _inside(path: str, root: str) -> bool:
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([real, base]) == base
    except (OSError, ValueError):
        return False


class TextEditorScreen(Screen):
    title = "Text editor"

    def __init__(
        self,
        match: Optional[RouteMatch],
        *,
        roots: Mapping[str, str],
        resolve_ref: Optional[Callable[[str], Optional[str]]] = None,
        files: Any = None,
        page: Any = None,
        notify: Optional[Callable[..., Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        mono_family: str = "monospace",
        tablet: bool = False,
        on_close: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(match)
        self.on_close = on_close  # leaves the screen after Back was confirmed (the app's back)
        self._leave_task: Any = None  # the "Unsaved changes" question being asked
        self.roots = {k: v for k, v in dict(roots).items() if v}
        self.files = files
        self.page = page
        self.notify = notify
        self.run_io = run_io
        self.mono_family = mono_family
        self.tablet = tablet
        params = match.params if match is not None else {}
        query = match.query if match is not None else {}
        self.fid = params.get("fid")
        try:
            self.hit = max(1, int(query.get("hit") or 1))
        except (TypeError, ValueError):
            self.hit = 1
        request = take_request(self.fid)
        self.find_term = request.find or ""
        self.read_only = bool(request.read_only)
        self.path = resolve_ref(self.fid) if (resolve_ref is not None and self.fid) else None
        self.error: Optional[str] = None
        if not self.path:
            self.error = "This file link has expired"
        elif not any(_inside(self.path, root) for root in self.roots.values()):
            self.error = OUTSIDE
        self.loaded: Optional[LoadedText] = None
        self.saved_text = ""
        self.matches: list = []
        self.match_index = -1
        self.title = os.path.basename(self.path) if self.path else "Text editor"
        self.uses_code_editor = False

    # ---- layout -----------------------------------------------------------------------------------

    def actions(self) -> list:
        self.save_action = ft.IconButton(icon=ft.Icons.SAVE, tooltip="Save", disabled=True, on_click=self._on_save,
                                         key="text-save")
        self.share_action = ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip="Share", on_click=self._on_share,
                                          disabled=self.files is None or self.error is not None, key="text-share")
        return [self.share_action, self.save_action]

    def build_body(self) -> ft.Control:
        if not hasattr(self, "save_action"):
            self.actions()
        language = editor_language(self.path or "")
        self.info = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            key="text-info")
        self.read_only_switch = ft.Switch(label="Read-only", value=self.read_only, on_change=self._on_read_only,
                                          key="text-read-only")
        self.notice = ft.Container(visible=False, key="text-notice")
        self.find_field = ft.TextField(hint_text="Find…", value=self.find_term, dense=True, expand=True,
                                       on_change=self._on_find_change, on_submit=lambda e: self._spawn(self.find_step(1)),
                                       key="text-find")
        self.find_count = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key="text-find-count")
        find_row = ft.Row([
            self.find_field, self.find_count,
            ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_UP, tooltip="Previous", size_constraints=HIT_TARGET,
                          on_click=lambda e: self._spawn(self.find_step(-1)), key="text-find-prev"),
            ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_DOWN, tooltip="Next", size_constraints=HIT_TARGET,
                          on_click=lambda e: self._spawn(self.find_step(1)), key="text-find-next"),
        ], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        chips = [ft.Text(language, theme_style=ft.TextThemeStyle.LABEL_SMALL, key="text-language")]
        if not code_editor_available():
            chips.append(ReasonChip(reason="Plain editor", detail="The code editor package is not in this build; "
                                                                  "a plain text field is used."))
        header = ft.Row([self.info, *chips, self.read_only_switch], wrap=True, spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.editor_holder = ft.Container(expand=True, key="text-editor-holder")
        if self.error:
            self.editor_holder.content = EmptyState(icon="INSERT_DRIVE_FILE", title=self.error, key="text-error")
        else:
            self.editor_holder.content = ft.ProgressRing(width=24, height=24, stroke_width=2)
        return ft.Column([
            ft.Container(content=ft.Column([header, find_row, self.notice], spacing=4, tight=True),
                         padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"], vertical=6)),
            self.editor_holder,
        ], spacing=0, expand=True, key="text-editor")

    def _make_editor(self, text: str) -> ft.Control:
        language = editor_language(self.path or "")
        if code_editor_available():
            try:
                import flet_code_editor as fce

                lang = getattr(getattr(fce, "CodeLanguage", None), language, None)
                editor = fce.CodeEditor(value=text, language=lang, read_only=self.read_only,
                                        on_change=self._on_text_change, expand=True)
                self.uses_code_editor = True
                return editor
            except Exception:
                log.info("CodeEditor unavailable; using a TextField", exc_info=True)
        self.uses_code_editor = False
        return ft.TextField(value=text, multiline=True, min_lines=10, expand=True, read_only=self.read_only,
                            text_style=ft.TextStyle(font_family=self.mono_family, size=13),
                            border=ft.InputBorder.NONE, content_padding=ft.Padding.all(12),
                            on_change=self._on_text_change, key="text-field")

    # ---- loading -----------------------------------------------------------------------------------

    def did_show(self) -> None:
        if self.error or not self.path:
            return
        self._spawn(self.load())

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    async def load(self) -> Optional[LoadedText]:
        try:
            loaded = await self._io(load_text, self.path)
        except (OSError, ValueError) as exc:
            self.error = f"This file cannot be opened: {exc}"
            self.editor_holder.content = EmptyState(icon="INSERT_DRIVE_FILE", title=self.error, key="text-error")
            self._update(self.editor_holder)
            return None
        self.loaded = loaded
        self.saved_text = loaded.text
        if loaded.read_only_reason:
            self.read_only = True
            self.read_only_switch.value = True
            self.read_only_switch.disabled = True
            self.notice.content = ft.Text(loaded.read_only_reason, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                          color=ft.Colors.ERROR)
            self.notice.visible = True
        self.editor = self._make_editor(loaded.text)
        self.editor_holder.content = self.editor
        lines = loaded.text.count("\n") + 1
        self.info.value = f"{lines:,} lines · {'CRLF' if loaded.crlf else 'LF'}{' · BOM' if loaded.bom else ''}"
        self._refresh_matches()
        self._update(self.body)
        if self.find_term and self.matches:
            self.match_index = min(self.hit, len(self.matches)) - 2
            await self.find_step(1)
        return loaded

    # ---- editing ------------------------------------------------------------------------------------

    def current_text(self) -> str:
        editor = getattr(self, "editor", None)
        return str(getattr(editor, "value", "") or "") if editor is not None else self.saved_text

    @property
    def dirty(self) -> bool:
        return self.loaded is not None and self.current_text() != self.saved_text

    def _on_text_change(self, e: Any = None) -> None:
        self._sync_save()
        if self.find_term:
            self._refresh_matches()
            self._update(self.find_count)

    def _sync_save(self) -> None:
        if hasattr(self, "save_action"):
            self.save_action.disabled = self.read_only or not self.dirty
            self._update(self.save_action)

    def _on_read_only(self, e: Any = None) -> None:
        if self.loaded is not None and self.loaded.read_only_reason:
            return
        self.read_only = bool(self.read_only_switch.value)
        editor = getattr(self, "editor", None)
        if editor is not None:
            editor.read_only = self.read_only
            self._update(editor)
        self._sync_save()

    async def save(self) -> bool:
        if self.loaded is None or self.read_only or not self.path:
            return False
        text = self.current_text()
        try:
            await self._io(lambda: save_text(self.path, text, bom=self.loaded.bom, crlf=self.loaded.crlf))
        except OSError as exc:
            self._say(f"Could not save: {exc}")
            return False
        self.saved_text = text
        self._sync_save()
        self._say(f"Saved {os.path.basename(self.path)}")
        return True

    def _on_save(self, e: Any = None) -> None:
        self._spawn(self.save())

    def _on_share(self, e: Any = None) -> None:
        if self.files is not None and self.path:
            self._spawn(self.files.share([self.path]))

    def handle_back(self) -> bool:
        """Back with unsaved edits stays and asks first; the screen leaves on Save or Discard."""
        if not self.dirty:
            return False
        self._spawn(self._leave_after_confirm())
        return True

    async def _leave_after_confirm(self) -> None:
        if await self.confirm_leave() and self.on_close is not None:
            self.on_close()

    async def confirm_leave(self) -> bool:
        """Unsaved edits: ask before the screen is left (Save / Discard / Cancel). The shell's leave
        guard and Back share one question while it is open."""
        if not self.dirty:
            return True
        pending = self._leave_task
        if pending is None or pending.done():
            pending = self._leave_task = asyncio.ensure_future(self._ask_leave())
        return bool(await asyncio.shield(pending))

    async def _ask_leave(self) -> bool:
        from glossarion_mobile.ui.tools.common import ChoiceDialog

        dialog = ChoiceDialog("Unsaved changes", f"Save the changes to {os.path.basename(self.path or '')}?",
                              [("cancel", "Cancel", "text"), ("discard", "Discard", "destructive"),
                               ("save", "Save", "filled")], key="text-leave")
        dialog.show(self.page)
        answer = await dialog.wait()
        if answer == "save":
            return await self.save()
        if answer == "discard":
            self.saved_text = self.current_text()  # dropped: nothing is left to ask about
            return True
        return False

    # ---- find ----------------------------------------------------------------------------------------

    def _on_find_change(self, e: Any = None) -> None:
        self.find_term = str(self.find_field.value or "")
        self.match_index = -1
        self._refresh_matches()
        self._update(self.find_count)

    def _refresh_matches(self) -> None:
        self.matches = find_matches(self.current_text(), self.find_term)
        if self.match_index >= len(self.matches):
            self.match_index = -1
        if not self.find_term:
            self.find_count.value = ""
        elif not self.matches:
            self.find_count.value = "0/0"
        else:
            self.find_count.value = f"{max(0, self.match_index) + 1 if self.match_index >= 0 else 0}/{len(self.matches)}"

    async def find_step(self, direction: int) -> Optional[int]:
        if not self.matches:
            self._refresh_matches()
            self._update(self.find_count)
            return None
        self.match_index = (self.match_index + direction) % len(self.matches)
        start = self.matches[self.match_index]
        self.find_count.value = f"{self.match_index + 1}/{len(self.matches)}"
        self._update(self.find_count)
        editor = getattr(self, "editor", None)
        if editor is not None and not self.uses_code_editor:
            try:
                await editor.focus()
            except Exception:
                pass
            editor.selection = ft.TextSelection(base_offset=start, extent_offset=start + len(self.find_term))
            self._update(editor)
        return start

    # ---- plumbing ------------------------------------------------------------------------------------

    def _spawn(self, coro: Any) -> Any:
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def _say(self, message: str) -> None:
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:
                pass

    @staticmethod
    def _update(*controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:  # not mounted yet (tests) or already gone
                pass
