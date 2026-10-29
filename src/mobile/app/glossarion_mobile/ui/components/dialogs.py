"""ConfirmDialog and snackbars (UI_SPEC §5.2).

``ConfirmDialog``: title, body (the verbatim desktop text when one exists), an
optional item list and 2 buttons; the destructive one uses the error colour.
While an async ``on_confirm`` runs, both buttons are disabled and a
ProgressRing shows (state ``running``); once it succeeds the dialog closes and
ignores further taps (state ``closed``). Dialogs open with ``page.show_dialog``
and close with ``close_dialog`` (Flet 1.0.3).

``change_views`` (hotfix 3 issue 1): a change of the top View first closes the dialogs the
client may show on the old one and waits for one client frame (see the section below).
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Callable, Optional, Sequence

import flet as ft
from flet.controls.dialog_control import DialogControl

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components._handlers import await_handler, call_handler
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "FRAME_PROBE_METHOD",
    "VIEWS_WAIT",
    "ConfirmDialog",
    "change_views",
    "close_dialog",
    "dialog_column",
    "install_views_guard",
    "reports_dismissals",
    "show_snackbar",
]

log = logging.getLogger("glossarion.ui")


def close_dialog(page: Any, dialog: Any) -> bool:
    """Close ``dialog`` itself; returns True when it was open.

    ``page.pop_dialog()`` closes the most recently opened dialog that is still open. After a
    handler has shown a SnackBar (also a dialog control in Flet 1.0.3) that is the SnackBar, not
    the sheet or confirm dialog that should close: Add key left its sheet open with no feedback
    and the same key was added again on every tap. Turning ``open`` off closes exactly this
    dialog's route (Flet 1.0.3 ``bottom_sheet.dart`` closes the route it pushed, as does
    ``alert_dialog.dart``); the client then sends ``dismiss`` and the page drops it.

    A sheet a tablet shows in the SidePanel (``components.surface.present_sheet``) closes the
    panel instead, so the sheet's own Save / Cancel paths work on every size class.

    Never raises: it runs in click handlers, and an exception there makes Flet report a crashed
    session (the whole app behind its error screen).
    """
    if dialog is not None and surface.dismiss(page, dialog):
        return True
    if dialog is None or not getattr(dialog, "open", False):
        return False
    if hasattr(dialog, "_frozen"):
        # Flet freezes a dialog in a keyed diff (one shown again under the same key before the
        # first one's dismiss arrived); its own use_dialog hook strips the marker the same way so
        # ``open`` can be turned off. page.pop_dialog() raised here and the dialog stayed open.
        try:
            del dialog._frozen
        except AttributeError:
            pass
    try:
        dialog.open = False
    except RuntimeError:
        return False
    guard = getattr(page, _GUARD, None) if page is not None else None
    if guard is not None and guard.unqueue(page, dialog):
        return True  # it was waiting for a view change and never reached the client
    try:
        dialog.update()
    except Exception:  # not mounted on a page (tests) or the session is gone
        if page is not None:
            try:
                page.update()
            except Exception:
                pass
    return True


def dialog_column(controls: Sequence[ft.Control], spacing: float = 8) -> ft.Column:
    """A dialog body that is only as tall as its content (owner #21).

    Flet 1.0.3 lays a scrolling Column out at least as tall as its bounded parent
    (``scrollable_control.dart`` ``_InnerConstraintsEnforcer``), so a dialog whose body scrolled opened
    screen-high with empty space under a few lines of text. The body is a plain Column and the dialog
    scrolls instead (``AlertDialog(scrollable=True)``, Flutter's own scroll view): it keeps its natural
    height and scrolls only when the content does not fit. No control in it may ``expand``."""
    return ft.Column(list(controls), tight=True, spacing=spacing)


class ConfirmDialog:
    def __init__(
        self,
        *,
        title: str,
        body: str = "",
        confirm_label: str = "OK",
        cancel_label: str = "Cancel",
        destructive: bool = False,
        items: Optional[Sequence[str]] = None,
        on_confirm: Optional[Callable[[], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.title = title
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.state = "idle"
        self.result: Optional[bool] = None
        self._page: Any = None
        self.body = body
        self.items = [str(item) for item in items or ()]
        content: list[ft.Control] = []
        if body:
            content.append(ft.Text(body, theme_style=ft.TextThemeStyle.BODY_MEDIUM))
        for item in items or ():
            content.append(ft.Text(f"• {item}", theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True))
        self.progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        content.append(ft.Row([self.progress], alignment=ft.MainAxisAlignment.CENTER))
        self.cancel_button = ft.TextButton(content=cancel_label, on_click=self._on_cancel)
        confirm_style = (
            ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR) if destructive else None
        )
        self.confirm_button = ft.FilledButton(content=confirm_label, on_click=self._on_confirm, style=confirm_style)
        self.column = dialog_column(content)
        self.dialog = ft.AlertDialog(
            modal=True,
            scrollable=True,
            title=ft.Text(title),
            content=ft.Container(width=tokens.SIZES["dialog_max"], content=self.column),
            actions=[self.cancel_button, self.confirm_button],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def show(self, page: Any) -> None:
        self._page = page
        if self.state == "closed":  # shown again
            self.state = "idle"
            self._set_running(False)
        page.show_dialog(self.dialog)

    def _close(self) -> None:
        close_dialog(self._page, self.dialog)

    def _set_running(self, running: bool) -> None:
        self.state = "running" if running else "idle"
        self.cancel_button.disabled = running
        self.confirm_button.disabled = running
        self.progress.visible = running
        try:
            self.dialog.update()
        except Exception:  # not mounted (tests) or already closed
            pass

    def _on_cancel(self, e: Any = None) -> None:
        if self.state in ("running", "closed"):
            return
        self.state = "closed"
        self.result = False
        self._close()
        call_handler(self.on_cancel)

    async def _on_confirm(self, e: Any = None) -> None:
        if self.state in ("running", "closed"):  # a second tap never runs the action again
            return
        self._set_running(True)
        try:
            await await_handler(self.on_confirm)
        except Exception:
            # A failing action keeps the dialog open with both buttons usable. Raising here made Flet report
            # a crashed session: the whole app behind its error screen, nothing left to close.
            log.exception("confirm action %r failed", self.title)
            self._set_running(False)
            return
        except BaseException:
            self._set_running(False)
            raise
        # Close this dialog by identity: a snackbar the action showed (often with Undo) stays.
        self.state = "closed"
        self.progress.visible = False
        self.result = True
        self._close()


def show_snackbar(
    page: Any,
    message: str,
    *,
    action_label: Optional[str] = None,
    on_action: Optional[Callable[[], Any]] = None,
    duration_ms: int = 4000,
) -> ft.SnackBar:
    """Show a SnackBar (``page.show_dialog``); returns it for tests.

    A bar without an action gets its dismiss ✕ on the LEFT (owner #15): Flutter's ``show_close_icon``
    puts it at the trailing edge, right over the chat composer's Stop button, so a user closing the
    message stopped the job instead."""
    text = ft.Text(message)
    bar = ft.SnackBar(
        content=text,
        action=action_label,
        on_action=(lambda e: call_handler(on_action)) if on_action is not None else None,
        duration=duration_ms,
        behavior=ft.SnackBarBehavior.FLOATING,
        show_close_icon=False,
    )
    if action_label is None:
        text.expand = True
        bar.content = ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.CLOSE, tooltip="Dismiss", icon_color=ft.Colors.ON_INVERSE_SURFACE,
                    size_constraints=HIT_TARGET, key="snackbar-close",
                    on_click=lambda e: close_dialog(page, bar),
                ),
                text,
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    page.show_dialog(bar)
    return bar


# ---- a change of the top View waits for the dialogs on the old one (hotfix 3 issue 1) ------------------
# Flet 1.0.3 builds the page's dialogs inside the LAST View only (view.dart), and a sheet or alert dialog
# keeps the modal route it pushed in its widget State (bottom_sheet.dart ``_sheetRoute``, alert_dialog.dart
# ``_dialogRoute``). A new top View moves the dialog widgets into it: a new State without that route. An
# ``open = False`` the client had not run on the old top View (a menu row that closes its sheet and opens a
# screen in the same burst), or any later one, then closes nothing; the route stays on the Navigator under the
# new screen (a pageless route stays attached to the page it was pushed over), comes back when that screen is
# popped, and neither Close Menu nor Flet's close reaches it - only Android back (owner: "the close menu button
# and slide to close is not working in some edge cases, I have to backspace on my android").
#
# Python cannot see whether the client built a dialog: ``page.show_dialog`` sends it at once and the phone
# draws frames between any two messages (a sheet shown in the same handler pass as a screen change may already
# stand on the old top View), while a dialog shown and closed before the next frame was never built and never
# sends a dismiss. So ``change_views`` (``AppShell`` installs every top View through it) first closes every
# route dialog the client may show on the old top View and waits for ONE client frame (``_ViewsGuard._probe``):
# a new empty Column in ``page.overlay`` - built, like the dialogs, only inside the last View - answers an
# ``invoke_method`` only once a frame built it (control.dart ``invokeMethod`` waits for the control's listener,
# which ScrollableControl registers in initState; column.dart wraps every Column in one), and that frame also
# ran the dialogs' close (their post-frame ``closeModalRoute``). Then the new top View goes out, in one message
# with the dialogs shown meanwhile (they open on it). A closed dialog whose dismiss never comes (it was never
# built) leaves ``page._dialogs`` ``VIEWS_WAIT`` after that frame, its on_dismiss run as for the client's.
#
# A change that takes the old top View off the page (Back, a pop) needs no wait: the client's Navigator removes
# the pageless routes above a page together with it (and reports their dismiss). Only clients that report
# dismissals take part (Flet's own transports); without one (host tests, a disconnected session) views change
# at once, as before.

#: dialogs Flutter's ScaffoldMessenger shows (no route to strand)
_ROUTE_FREE = ("SnackBar", "Banner")
#: longest wait (s) for the client frame (a lost answer) and for the dismiss of a closed dialog after it
VIEWS_WAIT = 1.5
#: the client method the frame probe is called with (no control handles it: ScrollableControl ignores it)
FRAME_PROBE_METHOD = "glossarion_frame"
#: a probe whose answer did not come in time stays mounted this long (the client gives up after 10 s; a late
#: answer for a control Python already dropped raises in Flet's message handler)
PROBE_KEEP = 12.0
_GUARD = "_glossarion_views_guard"
_HOOK = "_glossarion_views_hook"
#: the client ran this dialog's close in a frame on the top View of the time (no route of it can be stranded)
_CLOSE_SEEN = "_glossarion_close_seen"
#: how often the dialog was shown (a dismiss due for an earlier showing never touches a later one)
_SHOWN = "_glossarion_shown"


def _mark(control: Any, name: str, value: Any) -> None:
    try:
        object.__setattr__(control, name, value)
    except Exception:
        pass


def _unmark(control: Any, name: str) -> None:
    if hasattr(control, name):
        try:
            object.__delattr__(control, name)
        except Exception:
            pass


def _route_dialog(dialog: Any) -> bool:
    return isinstance(dialog, DialogControl) and type(dialog).__name__ not in _ROUTE_FREE


def _shown(dialog: Any) -> None:
    """A dialog went out (again): its close is not seen yet."""
    _unmark(dialog, _CLOSE_SEEN)
    _mark(dialog, _SHOWN, int(getattr(dialog, _SHOWN, 0) or 0) + 1)


def _view_key(view: Any) -> Any:
    """What the client's Navigator keys a View's page by (its route; the control id without one)."""
    route = getattr(view, "route", None)
    return route if route is not None else ("id", id(view))


def reports_dismissals(page: Any) -> bool:
    """True when ``page``'s client sends ``dismiss`` for every dialog it closes and answers method calls the
    way Flet's client does (Flet's own transports, or a connection that says so with
    ``confirms_dialog_dismiss``)."""
    try:
        conn = page.session.connection
    except Exception:
        return False
    if conn is None:
        return False
    flag = getattr(conn, "confirms_dialog_dismiss", None)
    if flag is not None:
        return bool(flag)
    module = getattr(type(conn), "__module__", "") or ""
    return module.split(".", 1)[0] in ("flet", "flet_web")


class _ViewsGuard:
    """Per page: the view changes waiting for one client frame, and the dialogs shown meanwhile."""

    def __init__(self) -> None:
        self.pending: list = []  # view changes, in order
        self.waiting: list = []  # the dialogs whose close that frame runs
        self.queued: list = []  # dialogs to show on the new top View
        self.task: Any = None  # the frame probe
        self.views_unsent = False  # the top View changed in this loop pass and may not be on the client yet

    def on_client(self, page: Any) -> list:
        """The route dialogs the client may show on its top View: open ones, and closed ones whose close the
        client has not run in a frame yet (its dismiss is still to come, or the client never built them)."""
        return [dialog for dialog in list(page._dialogs.controls) if _route_dialog(dialog)
                and (getattr(dialog, "open", False) or not getattr(dialog, _CLOSE_SEEN, False))]

    def apply_now(self, page: Any, apply: Callable[[], Any]) -> None:
        """Change the views now. The caller sends them later in this pass (``page.update()``); a dialog shown
        before that sends them first (``show_dialog``): the phone would build it on the old top View."""
        apply()
        if self.views_unsent:
            return
        self.views_unsent = True
        try:
            asyncio.get_running_loop().call_soon(self._sent)
        except RuntimeError:
            self.views_unsent = False

    def _sent(self) -> None:
        self.views_unsent = False

    def hold(self, page: Any, loop: Any, apply: Callable[[], Any], dialogs: list) -> None:
        for dialog in dialogs:
            if getattr(dialog, "open", False):
                close_dialog(page, dialog)
        self.pending = [apply]
        self.waiting = list(dialogs)
        self.task = loop.create_task(self._probe(weakref.ref(page)))

    async def _probe(self, page_ref: Any) -> None:
        """One client frame on the current top View (section comment), then the waiting view changes."""
        page = page_ref()
        probe = ft.Column(controls=[], width=0, height=0)
        answered = False
        try:
            page.overlay.append(probe)
            page._overlay.update()
            await probe._invoke_method(FRAME_PROBE_METHOD, None, timeout=VIEWS_WAIT)
            answered = True
        except asyncio.CancelledError:
            raise
        except Exception:  # TimeoutError: a lost answer, a closed session; the change applies anyway
            log.debug("no client frame for the view change", exc_info=True)
        finally:
            self.flush(page_ref, probe if answered else None)
            if not answered:
                try:
                    asyncio.get_running_loop().call_later(PROBE_KEEP, self._drop_probe, page_ref, probe)
                except RuntimeError:
                    pass

    @staticmethod
    def _drop_probe(page_ref: Any, probe: Any) -> None:
        page = page_ref()
        if page is None or probe not in page.overlay:
            return
        page.overlay.remove(probe)
        try:
            page._overlay.update()
        except Exception:
            log.debug("removing the frame probe failed", exc_info=True)

    def queue(self, page: Any, dialog: Any) -> None:
        if dialog in page._dialogs.controls or dialog in self.queued:
            raise RuntimeError("Dialog is already opened")
        dialog.open = True
        page._prepare_dialog(dialog)  # parent and on_dismiss as page.show_dialog sets them
        self.queued.append(dialog)

    def unqueue(self, page: Any, dialog: Any) -> bool:
        if dialog not in self.queued:
            return False
        self.queued.remove(dialog)
        try:
            page._restore_dialog_on_dismiss(dialog)
        except Exception:
            pass
        return True

    def flush(self, page_ref: Any, probe: Any = None) -> None:
        """The client ran a frame (or the wait is over): the waiting view changes, the dialogs shown meanwhile
        (and the answered probe gone) in one message."""
        applies, waited, self.pending, self.waiting = self.pending, self.waiting, [], []
        shows, self.queued = self.queued, []
        self.task = None
        page = page_ref()
        if page is None:
            return
        for dialog in waited:
            _mark(dialog, _CLOSE_SEEN, True)
            if dialog in page._dialogs.controls and not getattr(dialog, "open", False):
                self._expect_dismiss(page_ref, dialog)
        if probe is not None and probe in page.overlay:
            page.overlay.remove(probe)
        for apply in applies:
            try:
                apply()
            except Exception:
                log.exception("a view change failed")
        controls = page._dialogs.controls
        for dialog in shows:
            if getattr(dialog, "open", False) and dialog not in controls:
                controls.append(dialog)
                _shown(dialog)
        try:
            page.update()  # the new top View and the dialogs on it, in one message
        except Exception:
            log.debug("sending the waiting view change failed", exc_info=True)

    def _expect_dismiss(self, page_ref: Any, dialog: Any) -> None:
        """A closed dialog the client may never have built: no dismiss would ever come. ``VIEWS_WAIT`` after
        the frame it leaves ``page._dialogs`` as a client dismiss would take it (its on_dismiss runs)."""
        shown = getattr(dialog, _SHOWN, 0)
        try:
            asyncio.get_running_loop().call_later(VIEWS_WAIT, self._dismiss_unbuilt, page_ref, dialog, shown)
        except RuntimeError:
            pass

    @staticmethod
    def _dismiss_unbuilt(page_ref: Any, dialog: Any, shown: Any) -> None:
        page = page_ref()
        if (page is None or getattr(dialog, _SHOWN, 0) != shown or getattr(dialog, "open", False)
                or dialog not in page._dialogs.controls):
            return  # dismissed meanwhile, or shown again
        try:
            asyncio.ensure_future(page.session.dispatch_event(dialog._i, "dismiss", None))
        except Exception:
            log.debug("dropping a dialog the client never built failed", exc_info=True)


def _guard(page: Any, create: bool = True) -> Optional[_ViewsGuard]:
    guard = getattr(page, _GUARD, None)
    if guard is None and create:
        guard = _ViewsGuard()
        _mark(page, _GUARD, guard)
    return guard


def change_views(page: Any, apply: Callable[[], Any], *, top: Any = None,
                 views: Optional[Sequence[Any]] = None) -> bool:
    """Run ``apply`` (it changes ``page.views``; ``top`` is the View that will be on top, ``views`` the whole new
    list when known) now, or after one client frame that closed the dialogs on the current top View (section
    comment above). True when it ran now; the caller sends the page as usual."""
    if page is None or not reports_dismissals(page):
        apply()
        return True
    guard = _guard(page)
    if guard.pending:  # keep the order of view changes
        guard.pending.append(apply)
        return False
    try:
        current = page.views[-1] if page.views else None
    except Exception:
        current = None
    if top is not None and top is current:
        apply()  # the top View stays: its dialogs keep their client State
        return True
    leaving = views is not None and current is not None and _view_key(current) not in {_view_key(v) for v in views}
    dialogs = [] if current is None or leaving else guard.on_client(page)
    if not dialogs:  # nothing on the old top View, or it leaves the client's Navigator with its routes
        guard.apply_now(page, apply)
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        apply()
        return True
    guard.hold(page, loop, apply, dialogs)
    return False


def install_views_guard() -> None:
    """Hook ``page.show_dialog``: a dialog shown while a view change waits opens on the new top View, one shown
    right after a view change follows the new top View to the client, and the guard knows what went out; once
    per process."""
    from flet.controls.base_page import BasePage

    show = BasePage.show_dialog
    if getattr(show, _HOOK, False):
        return

    def show_dialog(self: Any, dialog: Any) -> None:
        if _keyed_twin(self, dialog):
            return
        guard = _guard(self) if reports_dismissals(self) else None
        if guard is not None and _route_dialog(dialog):
            if guard.pending or guard.queued:
                guard.queue(self, dialog)
                return
            if guard.views_unsent:  # the new top View first: the client builds dialogs in the last View it has
                guard.views_unsent = False
                try:
                    self.update()
                except Exception:
                    log.debug("sending the new top View failed", exc_info=True)
        show(self, dialog)
        if guard is not None and _route_dialog(dialog):
            _shown(dialog)

    setattr(show_dialog, _HOOK, True)
    BasePage.show_dialog = show_dialog

    restore = BasePage._restore_dialog_on_dismiss

    def restore_dialog_on_dismiss(self: Any, dialog: Any) -> None:
        # a dialog Flet diffed as frozen (a keyed twin) must still be able to leave page._dialogs; Flet's own
        # use_dialog strips the marker the same way before it sets open=False
        _unmark(dialog, "_frozen")
        restore(self, dialog)

    BasePage._restore_dialog_on_dismiss = restore_dialog_on_dismiss


def _keyed_twin(page: Any, dialog: Any) -> bool:
    """True when ``dialog`` must not be shown: another dialog with the same explicit key is already in
    ``page._dialogs`` (v9.15.7 hotfix 3). Two quick taps opened the DeepSeek sign-in sheet twice; Flet diffs a
    keyed pair of different objects as frozen, the second sheet's dismiss then raised "Frozen controls cannot
    be updated" and both sheets stayed stuck. An open twin wins (the second tap does nothing); a closed one
    still waiting for its dismiss leaves first, so the new dialog opens as a fresh control."""
    key = getattr(dialog, "key", None)
    dialogs = getattr(page, "_dialogs", None)
    if key is None or dialogs is None:
        return False
    for other in list(dialogs.controls):
        if other is dialog or getattr(other, "key", None) != key:
            continue
        if getattr(other, "open", False):
            log.info("not showing a second %r dialog (one with the same key is open)", key)
            return True
        try:
            page._restore_dialog_on_dismiss(other)
        except Exception:
            log.debug("restoring the closed twin's on_dismiss failed", exc_info=True)
        dialogs.controls.remove(other)
    return False
