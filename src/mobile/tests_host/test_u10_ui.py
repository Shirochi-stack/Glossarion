"""Host tests for the U10 UI: Settings › Cloud sync & sharing, the Book page Output tab additions, the chat Result
card's Send to cloud / Share file via link, and the U10 texts (``ui/screens/cloud_sync``).

* Texts: destinations, per-output status lines, the book summary line, the activity summary, link lines and the
  notification texts (payloads are routes, never paths).
* ``CloudFacade`` / ``ShareFacade`` over fakes of the services' public surface, and over the REAL
  ``CloudSyncService`` / ``ShareLinkService`` in a scratch folder (an in-memory upload provider: no network) when
  they import, so a drift between the UI and the services fails here.
* ``CloudSyncScreen``: no service (desktop / failed install: everything visible with a ReasonChip), destination
  choice (change asks first), the switch and formats, the queue and Retry now, the re-link banner, Forget, the
  provider switches behind the consent sheet, the pixeldrain key (cleared from the screen, never said), the Send
  options; the body and every sheet serialise in a fake Flet session.
* ``OutputTab``: cloud lines on the right rows, Send now, the per-book Default / Always / Never sheet, Share file
  via link (file choice → services → pre-flight → upload with progress → link sheet), the transfer.it hand-off
  with the pasted link, saved links with Copy / Share / Delete; unavailable actions keep their ReasonChips.
* ``JobCard.set_u10`` and ``ChatFeature.bind_u10_card`` / ``u10_card_action`` (the turn's workspace, in the
  Library or still in Attachments).

No network, no real Library, no real app data (every path is under ``tmp_path``).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u10_ui.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _has("flet"), reason="flet not installed")

if _has("flet"):
    import flet as ft

    from glossarion_mobile.ui.screens import cloud_sync as u10

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_u10", Path(__file__).with_name(
    "test_bootstrap.py"))


def _tb():
    module = importlib.util.module_from_spec(_TB_SPEC)
    _TB_SPEC.loader.exec_module(module)
    return module


def run(coro):
    return asyncio.run(coro)


async def until(predicate: Any, timeout: float = 5.0) -> bool:
    """Wait (polling) until ``predicate()`` holds: work that went to a worker thread (``asyncio.to_thread``)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


async def settle(rounds: int = 30) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    for _ in range(rounds):
        await asyncio.sleep(0)


def texts(control: Any, out: Optional[list] = None) -> list:
    """Every ``Text.value`` / string label under a control (dataclass walk)."""
    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(str(control.value))
    for name in ("content", "controls", "label", "leading", "title", "subtitle", "trailing", "actions"):
        value = getattr(control, name, None)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    texts(item, out)
        elif isinstance(value, ft.BaseControl):
            texts(value, out)
        elif isinstance(value, str) and name in ("content", "label"):
            out.append(value)
    return out


def find_key(control: Any, key: str) -> Any:
    """The first control under ``control`` whose key is (or starts with, for per-build keys) ``key``."""
    if control is None:
        return None
    own = getattr(control, "key", None)
    if isinstance(own, str) and (own == key or own.startswith(key + "-")):
        return control
    for name in ("content", "controls", "leading", "title", "subtitle", "trailing", "actions"):
        value = getattr(control, name, None)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, ft.BaseControl):
                hit = find_key(item, key)
                if hit is not None:
                    return hit
    return None


def chips(control: Any, out: Optional[list] = None) -> list:
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ReasonChip):
        out.append(control.reason)
    for name in ("content", "controls", "leading", "title", "subtitle", "trailing"):
        value = getattr(control, name, None)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, ft.BaseControl):
                chips(item, out)
    return out


class FakePage:
    width = 400
    height = 800

    def __init__(self) -> None:
        self.dialogs: list = []

    def show_dialog(self, dialog: Any) -> None:
        try:
            dialog.open = True
        except Exception:
            pass
        self.dialogs.append(dialog)

    def update(self, *args: Any) -> None:
        pass

    def by_key(self, key: str) -> Any:
        """A dialog by its key, or by its frame's (the U10 sheets: per-build dialog keys, ``U10Actions._sheet``)."""
        return next((d for d in reversed(self.dialogs)
                     if key in (getattr(d, "key", None), getattr(getattr(d, "content", None), "key", None))), None)


def event(**values: Any) -> Any:
    return types.SimpleNamespace(control=types.SimpleNamespace(**values), data=None)


# ---------------------------------------------------------------------------
# Fakes of the services' public surface (services/cloud_sync, services/share_links)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dest:
    id: str
    mode: str
    label: str = ""
    provider_label: str = ""
    can_create: bool = True
    can_write: bool = True
    needs_relink: str = ""
    target: str = "content://secret.tree/uri"
    path: str = ""

    @property
    def display(self) -> str:
        return self.label + (" · one file at a time" if self.mode == "files" else "")


@dataclass(frozen=True)
class Link:
    ok: bool
    reason: str = ""
    destination: Any = None
    note: str = ""
    cancelled: bool = False


@dataclass(frozen=True)
class KS:
    kind: str
    state: str
    text: str
    name: str = ""
    synced_at: float = 0.0
    error: str = ""
    note: str = ""
    progress: Optional[float] = None


class FakeCloud:
    """The ``CloudSyncService`` members the UI uses (``ui_state`` / ``book_state`` and the actions; answers are
    ``{ok, message, …}`` dicts like the service's ``_answer``)."""

    def __init__(self, platform: str = "android") -> None:
        self.platform = platform
        self.supported = platform in ("android", "ios")
        self.enabled = False
        self.kinds = {k: True for k in ("epub", "pdf", "txt", "html")}
        self.dest: Optional[Dest] = None
        self.overrides: dict = {}
        self.files: dict = {}  # identity -> {kind: entry}
        self.waiting_library: set = set()
        self.queue: list = []
        self.recent: list = []
        self.calls: list = []
        self.listeners: list = []
        self.last_saved_at = 0.0
        self.progress: Optional[dict] = None
        self.next_pick = Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")

    @staticmethod
    def _ui(dest: Optional[Dest]) -> Optional[dict]:
        if dest is None:
            return None
        return {"mode": dest.mode, "label": dest.label, "provider_label": dest.provider_label,
                "needs_relink": dest.needs_relink or None, "can_create": dest.can_create, "display": dest.display}

    def ui_state(self) -> dict:
        return {"supported": self.supported, "reason": None if self.supported else "Phone only",
                "enabled": self.enabled, "kinds": dict(self.kinds), "destination": self._ui(self.dest),
                "queue": list(self.queue), "recent": list(self.recent), "last_saved_at": self.last_saved_at or None,
                "failed": sum(1 for q in self.queue if q.get("status") == "failed"), "needs_pick": 0,
                "progress": self.progress, "phone_folder": self.platform == "android" and self.supported,
                "explainer": "service copy"}

    def book_state(self, identity: str) -> dict:
        self.calls.append(("book_state", identity))
        override = self.overrides.get(identity, "default")
        return {"override": override, "auto": override == "always" or (override == "default" and self.enabled),
                "files": dict(self.files.get(identity, {})), "in_library": identity not in self.waiting_library,
                "waiting_library": identity in self.waiting_library}

    def set_book_override(self, identity: str, value: str) -> dict:
        self.calls.append(("set_book_override", identity, value))
        self.overrides[identity] = value
        self.changed()
        return {"ok": True, "message": "", "override": value}

    def set_enabled(self, value: bool) -> dict:
        self.calls.append(("set_enabled", value))
        self.enabled = bool(value)
        return {"ok": True, "message": "", "enabled": bool(value)}

    def set_kind_enabled(self, kind: str, value: bool) -> dict:
        self.calls.append(("set_kind_enabled", kind, value))
        self.kinds[kind] = bool(value)
        return {"ok": True, "message": ""}

    async def pick_folder(self) -> dict:
        self.calls.append(("pick_folder",))
        self.dest = self.next_pick
        return {"ok": True, "message": "", "destination": self._ui(self.dest)}

    async def use_save_locations(self) -> dict:
        self.calls.append(("use_save_locations",))
        self.dest = Dest("files", "files", label="Chosen per file")
        return {"ok": True, "message": "", "destination": self._ui(self.dest)}

    async def use_phone_folder(self) -> dict:
        self.calls.append(("use_phone_folder",))
        if self.platform != "android":
            return {"ok": False, "message": "Android only"}
        self.dest = Dest("phone", "phone", label="Downloads/Glossarion", provider_label="Phone")
        return {"ok": True, "message": "Glossarion keeps one copy of each book there.",
                "destination": self._ui(self.dest)}

    async def forget_destination(self) -> dict:
        self.calls.append(("forget_destination",))
        self.dest = None
        return {"ok": True, "message": "Files already saved stay where they are"}

    async def test_destination(self) -> dict:
        self.calls.append(("test_destination",))
        return {"ok": True, "message": "The folder accepts files"}

    async def send_now(self, identity: str, kinds: Any = None) -> dict:
        self.calls.append(("send_now", identity))
        return {"ok": True, "message": f"Saving to {self.dest.display if self.dest else '?'}…"}

    async def choose_save_location(self, identity: str, kind: str) -> dict:
        self.calls.append(("choose_save_location", identity, kind))
        return {"ok": True, "message": ""}

    def retry_now(self) -> dict:
        self.calls.append(("retry_now",))
        count = len(self.queue)
        return {"ok": True, "message": f"Retrying {count} book{'s' if count != 1 else ''}"}

    def subscribe(self, callback: Any) -> Any:
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback) if callback in self.listeners else None

    def changed(self) -> None:
        for callback in list(self.listeners):
            callback()


class LegacyStore:
    def __init__(self, cloud: "LegacyCloud") -> None:
        self.cloud = cloud

    def queue(self) -> list:
        return [{"identity": i["identity"]} for i in self.cloud.queue]

    def enqueue(self, identity: str, reason: str, **kwargs: Any) -> dict:
        self.cloud.calls.append(("enqueue", identity, reason))
        return {"key": identity}


class LegacyCloud:
    """A cloud-sync service from before the UI contract (``settings`` / ``summary`` / ``status_for`` /
    ``link_folder`` / ``LinkResult``): the facade still drives it."""

    platform = "android"
    supported = True

    def __init__(self) -> None:
        self.dest: Optional[Dest] = None
        self.queue: list = []
        self.statuses: dict = {}
        self.calls: list = []
        self.store = LegacyStore(self)

    def settings(self) -> Any:
        return types.SimpleNamespace(enabled=False, kinds={"pdf": False}, destination=self.dest)

    def summary(self) -> dict:
        return {"waiting": len(self.queue), "failed": 0, "needs_pick": 0, "last_saved_at": 0, "saving": False}

    def queue_items(self) -> list:
        return list(self.queue)

    def status_for(self, identity: str) -> dict:
        return dict(self.statuses.get(identity, {}))

    def override(self, identity: str) -> str:
        return "always"

    async def link_folder(self) -> Link:
        self.calls.append(("link_folder",))
        self.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")
        return Link(True, destination=self.dest)

    def set_kind(self, kind: str, value: bool) -> Any:
        self.calls.append(("set_kind", kind, value))
        return self.settings()

    async def drain_now(self, reason: str = "manual") -> dict:
        self.calls.append(("drain_now", reason))
        return {}


class FakeShareError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _info(pid: str, label: str, *, e2ee: bool = False, direct: bool = True, needs_key: bool = False,
          can_delete: bool = True, max_bytes: Optional[int] = None) -> Any:
    return types.SimpleNamespace(id=pid, label=label, operator=f"{label} operator", country="", site="",
                                 e2ee=e2ee, direct=direct, needs_api_key=needs_key, can_delete=can_delete,
                                 max_bytes=max_bytes, retention=f"{label} keeps links for a while", terms_url="")


class FakeShares:
    """The ``ShareLinkService`` members the UI uses (direct uploads are instant; the URLs are fake)."""

    def __init__(self) -> None:
        self.infos = {
            "transferit": _info("transferit", "transfer.it", direct=False, can_delete=False),
            "gofile": _info("gofile", "Gofile"),
            "send": _info("send", "Send", e2ee=True),
            "pixeldrain": _info("pixeldrain", "pixeldrain", needs_key=True, max_bytes=10),
        }
        self.enabled: dict = {}
        self.consented: dict = {}
        self.key: Optional[str] = None
        self.links: list = []
        self.calls: list = []
        self.listeners: list = []
        self.state = types.SimpleNamespace(phase="idle", provider="", sent=0, total=0)
        self.handoff_opened: list = []
        self.handoff = types.SimpleNamespace(open=self._open)
        self.send = {"expire": 259200, "downloads": 20}
        self.fail_with: Optional[FakeShareError] = None
        self.existing = None

    async def _open(self, plan: Any) -> bool:
        self.handoff_opened.append(plan.url)
        return True

    def provider_states(self) -> list:
        out = []
        for pid in ("transferit", "gofile", "send", "pixeldrain"):
            info = self.infos[pid]
            enabled, consented = self.enabled.get(pid, False), self.consented.get(pid, False)
            has_key = bool(self.key) if pid == "pixeldrain" else False
            reason = None
            if not enabled:
                reason = "Turn it on in Settings › Cloud sync & sharing"
            elif not consented:
                reason = "Read and accept what this service can see first"
            elif info.needs_api_key and not has_key:
                reason = "Add your pixeldrain API key in Settings › Cloud sync & sharing"
            out.append(types.SimpleNamespace(info=info, enabled=enabled, consented=consented, has_key=has_key,
                                             reason=reason))
        return out

    def menu(self, path: Any) -> list:
        return [(state, state.reason) for state in self.provider_states()]

    def consent_text(self, pid: str, path: Any = None, size: Any = None) -> Any:
        label = self.infos[pid].label
        return types.SimpleNamespace(title=f"Share file via link with {label}?",
                                     lines=(f"This sends the file to {label}.", f"{label} can read the file."),
                                     checkbox="I have the right to share this file", confirm="Turn on",
                                     terms_url="")

    async def set_enabled(self, pid: str, enabled: bool, *, consent: bool = False) -> Any:
        self.calls.append(("set_enabled", pid, enabled, consent))
        self.enabled[pid] = bool(enabled)
        if enabled and consent:
            self.consented[pid] = True
        self.emit("providers")
        return self.provider_states()[0]

    async def give_consent(self, pid: str) -> Any:
        self.calls.append(("give_consent", pid))
        self.consented[pid] = True
        return None

    async def set_pixeldrain_key(self, key: str) -> Any:
        self.calls.append(("set_pixeldrain_key", "***"))
        self.key = key
        self.emit("providers")

    async def clear_pixeldrain_key(self) -> Any:
        self.calls.append(("clear_pixeldrain_key",))
        self.key = None

    async def check_pixeldrain_key(self) -> bool:
        self.calls.append(("check_pixeldrain_key",))
        return self.key == "good-key"

    async def forget_gofile_account(self) -> None:
        self.calls.append(("forget_gofile_account",))

    def send_options(self) -> dict:
        return dict(self.send)

    async def set_send_options(self, *, expire: Any = None, downloads: Any = None) -> dict:
        self.calls.append(("set_send_options", expire, downloads))
        if expire is not None:
            self.send["expire"] = expire
        if downloads is not None:
            self.send["downloads"] = downloads
        return dict(self.send)

    async def preflight(self, pid: str, path: str) -> Any:
        self.calls.append(("preflight", pid))
        warnings = ("120 MB will be uploaded; this uses mobile data when you are not on Wi-Fi.",) \
            if os.path.getsize(path) > 5 else ()
        return types.SimpleNamespace(ok=True, message="", warnings=warnings, existing=self.existing,
                                     size=os.path.getsize(path))

    def _link(self, pid: str, path: str, book: str, url: str) -> Any:
        info = self.infos[pid]
        return types.SimpleNamespace(id=f"l{len(self.links) + 1}", provider=pid, url=url, name=os.path.basename(path),
                                     size=os.path.getsize(path), created=time.time(), book=book,
                                     source=os.path.abspath(path), expires=None, downloads_limit=None,
                                     can_delete=info.can_delete, label=info.label, e2ee=info.e2ee, expired=False)

    async def upload(self, pid: str, path: str, *, book: Optional[str] = None, options: Any = None) -> Any:
        self.calls.append(("upload", pid, book))
        size = os.path.getsize(path)
        for sent in (0, size // 2, size):
            self.state = types.SimpleNamespace(phase="uploading", provider=pid, sent=sent, total=size)
            self.emit("upload")
            await asyncio.sleep(0)
        if self.fail_with is not None:
            raise self.fail_with
        link = self._link(pid, path, book or "", f"https://{pid}.example/d/{len(self.links) + 1}#k")
        self.links.append(link)
        self.emit("links")
        return link

    def cancel(self) -> bool:
        self.calls.append(("cancel",))
        return True

    def links_blocking(self, *, book: Optional[str] = None, paths: Any = ()) -> list:
        return [link for link in self.links if not book or link.book == book]

    async def delete_link(self, link_id: str) -> Any:
        self.calls.append(("delete_link", link_id))
        self.links = [link for link in self.links if link.id != link_id]
        return types.SimpleNamespace(removed=True, remote="deleted")

    async def forget_link(self, link_id: str) -> bool:
        self.calls.append(("forget_link", link_id))
        self.links = [link for link in self.links if link.id != link_id]
        return True

    async def start_handoff(self, path: str, *, book: Optional[str] = None) -> Any:
        self.calls.append(("start_handoff", book))
        plan = types.SimpleNamespace(url="https://transfer.it/start", file_name=os.path.basename(path),
                                     location=f"Downloads › Glossarion › {os.path.basename(path)}",
                                     hint="On transfer.it tap Add files.", show_in_files=False)
        await self._open(plan)
        return plan

    async def add_pasted_link(self, text: str, *, path: Optional[str] = None, book: Optional[str] = None) -> Any:
        self.calls.append(("add_pasted_link", book))
        if not str(text).startswith("https://transfer.it/t/"):
            raise FakeShareError("bad_link", "Paste the link transfer.it shows (it starts with https://transfer.it/t/).")
        link = self._link("transferit", path, book or "", str(text))
        self.links.append(link)
        return link

    def subscribe(self, callback: Any) -> Any:
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback) if callback in self.listeners else None

    def emit(self, kind: str) -> None:
        for callback in list(self.listeners):
            callback(kind)


# ---------------------------------------------------------------------------
# texts
# ---------------------------------------------------------------------------


def test_destination_and_status_texts():
    assert u10.destination_text(None) == "Not set"
    # the service composes "Drive › Glossarion": never doubled
    assert u10.destination_text({"mode": "folder", "label": "Drive › Glossarion", "provider_label": "Drive"}) == \
        "Drive › Glossarion"
    assert u10.destination_text({"mode": "folder", "label": "Books", "provider_label": "Nextcloud"}) == \
        "Nextcloud › Books"
    assert u10.destination_text({"mode": "files", "label": "Chosen per file"}) == "Save locations · one file at a time"
    assert u10.destination_text({"mode": "phone", "label": "Downloads/Glossarion"}) == \
        "Phone folder · Downloads/Glossarion"
    now = 10_000.0
    assert u10.ago_text(now - 10, now) == "just now" and u10.ago_text(now - 180, now) == "3 min ago"
    assert u10.ago_text(now - 7200, now) == "2 h ago" and u10.ago_text(0, now) == ""
    line = u10.file_status_line({"status": "ok", "synced_at": now - 120}, dest_label="Drive › Glossarion", now=now)
    assert line == ("CLOUD_DONE", "Saved to Drive › Glossarion · 2 min ago", "ok")
    # the service's own wording wins; a caveat (replaced file) turns it into a warning
    assert u10.file_status_line({"status": "ok", "text": "Saved to X · just now", "warning": "replaced: the link changed"}
                                )[1:] == ("Saved to X · just now · replaced: the link changed", "warn")
    assert u10.file_status_line({"status": "writing", "written": 450, "total": 1000})[1] == "Saving… 45%"
    assert u10.file_status_line({"status": "waiting", "reason": "busy"})[1] == \
        "Waiting: the book is being compiled or translated"
    assert u10.file_status_line({"status": "failed", "text": "Couldn't save: provider_error"})[1:] == \
        ("Couldn't save: provider_error · tap to retry", "error")
    assert u10.file_status_line({"status": "needs_pick"})[2] == "action"
    assert "tap to reconnect" in u10.file_status_line({"status": "needs_relink"})[1]
    assert u10.file_status_line({"status": "format_off"}, kind="pdf")[1] == "PDF is not synced"
    assert u10.file_status_line(None) is None and u10.file_status_line({"status": "??"}) is None
    state = {"available": True, "enabled": True, "destination": {"mode": "folder", "label": "Drive › G"}}
    assert u10.book_status_line(state, {}, in_library=False)[1] == "Copied to Drive › G once the book is in the Library"
    assert u10.book_status_line(dict(state, enabled=False), {}, in_library=False) is None
    assert u10.book_status_line(state, {"files": {"epub": {"status": "ok", "synced_at": now - 60}}}, now=now)[:3] == \
        ("CLOUD_DONE", "Saved to Drive › G · 1 min ago", "ok")
    assert u10.book_status_line(state, {"files": {"epub": {"status": "failed", "error": "e"}}})[3] == "retry"
    assert u10.book_status_line(state, {"files": {"pdf": {"status": "needs_pick"}}})[3] == "book"
    relink = dict(state, destination={"mode": "folder", "label": "Drive › G", "needs_relink": "revoked"})
    assert u10.book_status_line(relink, {})[3] == "settings"
    assert u10.book_status_line({"available": True, "destination": None}, {}) is None
    summary = u10.summary_text({"last_saved_at": now - 180, "queue": [{"status": "waiting"}, {"status": "failed"}],
                                "failed": 1}, now)
    assert summary == "Last saved 3 min ago · 1 waiting · 1 failed"
    assert u10.link_meta_text({"provider_label": "Gofile", "size": 5 * 1024 * 1024, "created_at": now - 7200,
                               "expires_text": "expires in 3 days"}, now) == \
        "Gofile · 5.0 MB · 2 h ago · expires in 3 days"


def test_notification_texts_are_routes_without_paths():
    title, body, route = u10.cloud_failure_notice("Drive › Glossarion", lost_access=True)
    assert title == "Glossarion lost access to Drive › Glossarion" and route == u10.CLOUD_ROUTE
    assert u10.cloud_failure_notice("Drive", count=3)[1] == "3 files waiting · tap for details"
    assert u10.cloud_needs_location_notice(1, "0123456789ab") == (
        "1 book needs a save location", "Tap to choose where to save it", "/library/book/0123456789ab?tab=output")
    assert u10.cloud_needs_location_notice(4)[0:3:2] == ("4 books need a save location", u10.CLOUD_ROUTE)
    assert u10.cloud_progress_text("Book.epub", "Drive", 45, 100) == "Saving Book.epub to Drive · 45%"
    assert u10.share_progress_text("Book.epub", "Gofile") == "Uploading Book.epub to Gofile…"
    failed = u10.share_failed_notice("Book.epub", "Gofile", "x" * 400, "0123456789ab")
    assert len(failed[1]) <= 200 and failed[2].startswith("/library/book/")
    for _title, _body, route in (u10.cloud_failure_notice("D"), u10.cloud_needs_location_notice(2), failed):
        assert route.startswith("/") and "\\" not in route and "content:" not in route
    assert u10.CLOUD_NOTIFICATION_BASE == 42000
    from glossarion_mobile.ui.router import parse_route

    assert parse_route("/library/book/0123456789ab?tab=output").name == "library.book"


# ---------------------------------------------------------------------------
# facades
# ---------------------------------------------------------------------------


def test_cloud_facade_maps_the_service_surface():
    async def scenario():
        cloud = FakeCloud()
        facade = u10.CloudFacade(lambda: cloud)  # read late (installed after the UI)
        state = facade.snapshot()
        assert state["available"] and state["supported"] and state["destination"] is None and not state["enabled"]
        assert "explainer" not in state or state["explainer"] == "service copy"
        assert (await facade.call("pick_folder"))["ok"] and cloud.calls[-1] == ("pick_folder",)
        state = facade.snapshot()
        assert state["destination"]["label"] == "Drive › Glossarion" and "target" not in state["destination"]
        assert "content://" not in repr(state)  # the tree URI never reaches the UI
        cloud.queue = [{"name": "A", "title": "A", "status": "waiting", "reason": "busy", "error": ""},
                       {"name": "B", "title": "B", "status": "failed", "reason": "provider_error",
                        "error": "the cloud app reported an error"}]
        state = facade.snapshot()
        assert [(q["title"], q["status"], q["reason"]) for q in state["queue"]] == [
            ("A", "waiting", "busy"), ("B", "failed", "provider_error")]
        assert u10.file_status_line(state["queue"][1])[2] == "error"
        cloud.files["/w/a"] = {"epub": {"status": "ok", "synced_at": 5.0, "warning": "replaced: the link changed"},
                               "pdf": {"status": "writing", "written": 40, "total": 100}}
        cloud.overrides["/w/a"] = "always"
        book = facade.book("/w/a")
        assert book["override"] == "always" and book["auto"] and book["in_library"]
        assert u10.file_status_line(book["files"]["epub"])[2] == "warn"
        assert u10.file_status_line(book["files"]["pdf"])[1] == "Saving… 40%"
        cloud.waiting_library.add("/w/c")
        assert facade.book("/w/c")["in_library"] is False
        assert (await facade.call("set_kind", "pdf", False))["ok"] and cloud.calls[-1] == ("set_kind_enabled", "pdf",
                                                                                            False)
        assert (await facade.call("set_override", "/w/a", "never"))["ok"]
        assert cloud.calls[-1] == ("set_book_override", "/w/a", "never")
        phone = await facade.call("use_phone_folder")
        assert phone["ok"] and "one copy" in phone["message"]
        cloud.platform = "ios"
        refused = await facade.call("use_phone_folder")
        assert not refused["ok"] and refused["message"] == "Android only"
        result = await facade.retry_now()
        assert result["ok"] and cloud.calls[-1] == ("retry_now",) and result["message"] == "Retrying 2 books"
        missing = u10.CloudFacade(None)
        assert not missing.available and missing.snapshot()["reason"] == u10.NO_SERVICE_REASON
        assert (await missing.call("send_now", "/w"))["unavailable"]

    run(scenario())


def test_cloud_facade_still_drives_a_service_from_before_the_contract():
    async def scenario():
        legacy = LegacyCloud()
        facade = u10.CloudFacade(legacy)
        picked = await facade.call("pick_folder")
        assert picked["ok"] and legacy.calls[-1] == ("link_folder",) and "target" not in picked["destination"]
        legacy.queue = [{"identity": "/w/a", "title": "A", "reason": "job:1", "attempts": 0, "error": "busy",
                         "failed": False}]
        state = facade.snapshot()
        assert state["destination"]["label"] == "Drive › Glossarion" and not state["kinds"]["pdf"]
        assert state["queue"][0]["reason"] == "busy"
        legacy.statuses["/w/a"] = {"epub": KS("epub", "saved", "Saved to Drive › Glossarion · just now", synced_at=5.0),
                                   "pdf": KS("pdf", "saving", "Saving…", progress=0.4),
                                   "html": KS("html", "no_destination", "Cloud sync is not set up")}
        book = facade.book("/w/a")
        assert book["files"]["epub"]["text"] == "Saved to Drive › Glossarion · just now" and book["auto"]
        assert u10.file_status_line(book["files"]["pdf"])[1] == "Saving… 40%" and "html" not in book["files"]
        assert (await facade.call("set_kind", "pdf", True))["ok"] and legacy.calls[-1] == ("set_kind", "pdf", True)
        spawned: list = []
        result = await facade.retry_now(spawn=spawned.append)
        assert result["ok"] and ("enqueue", "/w/a", "retry") in legacy.calls and len(spawned) == 1
        await spawned[0]
        assert legacy.calls[-1] == ("drain_now", "retry")

    run(scenario())


def test_share_facade_maps_the_service_surface(tmp_path):
    book = tmp_path / "Book.epub"
    book.write_bytes(b"abc")

    async def scenario():
        shares = FakeShares()
        facade = u10.ShareFacade(shares)
        providers = facade.providers()
        assert [p["id"] for p in providers] == ["transferit", "gofile", "send", "pixeldrain"]
        assert providers[0]["handoff"] and not providers[1]["handoff"] and providers[2]["e2ee"]
        assert providers[3]["needs_key"] and not any(p["enabled"] for p in providers)
        menu = facade.menu(str(book))
        assert [reason for _p, reason in menu] == [u10.SERVICE_OFF_REASON] * 4
        consent = facade.consent(providers[1], str(book), 3)
        assert consent["checkbox"] and len(consent["lines"]) == 2 and "Gofile can read" in consent["lines"][1]
        assert (await facade.enable("gofile", True, consent=True))["ok"]
        assert shares.calls[-1] == ("set_enabled", "gofile", True, True)
        seen: list = []
        link = await facade.upload("gofile", str(book), workspace=str(tmp_path), progress=lambda s, t: seen.append((s, t)))
        assert link["ok"] and link["url"].startswith("https://gofile.example/") and link["provider_label"] == "Gofile"
        assert seen[-1] == (3, 3) and shares.calls[-1] == ("upload", "gofile", str(tmp_path))
        assert not shares.listeners  # the progress subscription ends with the upload
        assert [entry["id"] for entry in facade.links_for(str(tmp_path))] == ["l1"]
        shares.fail_with = FakeShareError("maybe_uploaded", "The upload may have finished: check before trying again")
        failed = await facade.upload("gofile", str(book), workspace=str(tmp_path))
        assert failed == {"ok": False, "message": "The upload may have finished: check before trying again",
                          "code": "maybe_uploaded"}
        deleted = await facade.delete_link("l1")
        assert deleted["ok"] and deleted["remote"] == "deleted"
        plan = await facade.start_handoff(str(book), str(tmp_path))
        assert plan["ok"] and plan["location"].startswith("Downloads › Glossarion") and plan["opened"]
        assert shares.handoff_opened == ["https://transfer.it/start"]
        assert await facade.reopen_handoff(plan) and shares.handoff_opened[-1] == "https://transfer.it/start"
        bad = await facade.save_pasted_link("https://evil.example/x", path=str(book), workspace=str(tmp_path))
        assert not bad["ok"] and "transfer.it/t/" in bad["message"]
        good = await facade.save_pasted_link("https://transfer.it/t/abcdef12", path=str(book), workspace=str(tmp_path))
        assert good["ok"] and good["url"] == "https://transfer.it/t/abcdef12" and not good["can_delete"]
        assert (await facade.set_key("pixeldrain", "k"))["ok"] and shares.key == "k"
        assert not (await facade.check_key("pixeldrain"))["ok"]  # the fake only accepts "good-key"
        assert facade.send_options() == {"expire": 259200, "downloads": 20}
        assert u10.short_reason("Add your pixeldrain API key in Settings › Cloud sync & sharing") == \
            u10.NEEDS_KEY_REASON and u10.short_reason(None) is None

    run(scenario())


_REAL = _has("glossarion_mobile.services.cloud_sync") and _has("glossarion_mobile.services.share_links")


class Fernet:  # the share service's secret box checks the cipher type name; reversible stand-in
    pass


class _Handler:
    cipher = Fernet()

    @staticmethod
    def encrypt_value(value: str) -> str:
        return "ENC:" + value[::-1]

    @staticmethod
    def decrypt_value(token: str) -> str:
        return token[4:][::-1]


class _MemoryProvider:
    """A direct share provider without network (the real service's provider interface)."""

    def __init__(self, info: Any) -> None:
        self.info = info
        self.deleted: list = []

    def upload(self, path: str, *, name: str, mime: str, size: int, credentials: Any, options: Any, progress: Any,
               cancel: Any) -> Any:
        from glossarion_mobile.services.share_providers import UploadResult

        progress(0, size)
        progress(size, size)
        return UploadResult(url=f"https://share.invalid/{self.info.id}/1", remote_id="1", delete_handle={"id": "1"})

    def delete(self, handle: Any, credentials: Any, cancel: Any = None) -> bool:
        self.deleted.append(handle)
        return True


class _Prefs:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@pytest.mark.skipif(not _REAL, reason="the U10 services are not in this tree")
def test_facades_drive_the_real_services_without_network(tmp_path):
    from glossarion_mobile.services import cloud_sync as cs
    from glossarion_mobile.services import share_links as sl
    from glossarion_mobile.services.share_providers import provider_info
    from glossarion_mobile.state.cloud_records import CloudRecordStore

    library = tmp_path / "Library"
    workspace = library / "Book"
    workspace.mkdir(parents=True)
    epub = workspace / "Book.epub"
    epub.write_bytes(b"PK-book")

    async def scenario():
        store = CloudRecordStore(tmp_path / "mobile_cloud.json")
        store.load()
        service = cs.CloudSyncService(docs=types.SimpleNamespace(available=True), store=store, prefs=_Prefs(),
                                      platform="android", cache_dir=str(tmp_path / "cache"))
        facade = u10.CloudFacade(service)
        try:
            state = facade.snapshot()
            assert state["supported"] and state["destination"] is None and not state["enabled"]
            assert (await facade.call("use_phone_folder"))["ok"]
            state = facade.snapshot()
            assert state["destination"]["mode"] == "phone"
            assert u10.destination_text(state["destination"]) == "Phone folder · Downloads/Glossarion"
            assert (await facade.call("set_kind", "pdf", False))["ok"] and not facade.snapshot()["kinds"]["pdf"]
            assert (await facade.call("set_enabled", True))["ok"] and facade.snapshot()["enabled"]
            assert (await facade.call("set_override", str(workspace), "never"))["ok"]
            book = facade.book(str(workspace))
            assert book["override"] == "never" and not book["auto"]
            assert (await facade.call("set_override", str(workspace), "default"))["ok"]
            assert facade.book(str(workspace))["auto"]
            store.clear_queue()  # the override above queued the book; a queued book reads "Waiting"
            for kind in ("epub", "pdf"):
                store.update_record(service.destination().id, cs._norm(str(workspace)), kind, str(workspace),
                                    status="ok", synced_at=time.time(), name=f"Book.{kind}")
            files = facade.book(str(workspace))["files"]
            line = u10.file_status_line(files["epub"], dest_label=u10.destination_text(state["destination"]))
            assert line[0] == "CLOUD_DONE" and line[1].startswith("Saved to Phone folder")
            assert u10.file_status_line(files["pdf"], kind="pdf")[1] == "PDF is not synced"
            summary = u10.summary_text(facade.snapshot(), time.time())
            assert summary.startswith("Last saved just now")
            assert (await facade.retry_now())["ok"]
            result = await facade.call("send_now", str(workspace))
            assert "ok" in result and "message" in result
            await facade.call("forget_destination")
            assert facade.snapshot()["destination"] is None
        finally:
            service.close()

        providers = {pid: _MemoryProvider(provider_info(pid)) for pid in ("gofile", "send", "pixeldrain")}
        share = sl.ShareLinkService(tmp_path / "mobile_share_links.json", platform="desktop",
                                    allowed_roots=[str(library)], cache_dir=str(tmp_path / "cache"),
                                    providers=providers, secrets=sl.SecretBox(handler=_Handler()))
        await share.load()
        facade = u10.ShareFacade(share)
        listed = facade.providers()
        assert [p["id"] for p in listed][:2] == ["transferit", "gofile"] and not any(p["enabled"] for p in listed)
        assert listed[0]["handoff"] and listed[2]["e2ee"] and listed[3]["needs_key"]
        assert all(reason for _p, reason in facade.menu(str(epub)))  # everything off until turned on
        consent = facade.consent(listed[1], str(epub), 7)
        assert consent["lines"] and consent["checkbox"]
        assert (await facade.enable("gofile", True, consent=True))["ok"]
        reasons = dict((p["id"], r) for p, r in facade.menu(str(epub)))
        assert reasons["gofile"] is None and reasons["pixeldrain"] == u10.SERVICE_OFF_REASON
        seen: list = []
        link = await facade.upload("gofile", str(epub), workspace=str(workspace),
                                   progress=lambda sent, total: seen.append((sent, total)))
        assert link["ok"] and link["url"] == "https://share.invalid/gofile/1" and link["can_delete"]
        assert seen and seen[-1][0] == seen[-1][1] == 7
        saved = facade.links_for(str(workspace))
        assert [entry["url"] for entry in saved] == ["https://share.invalid/gofile/1"]
        raw = (tmp_path / "mobile_share_links.json").read_text(encoding="utf-8")
        assert "share.invalid" not in raw  # the link is stored encrypted
        assert (await facade.delete_link(saved[0]["id"]))["ok"] and providers["gofile"].deleted
        assert facade.links_for(str(workspace)) == []
        assert (await facade.enable("transferit", True, consent=True))["ok"]
        plan = await facade.start_handoff(str(epub), str(workspace))
        assert plan["ok"] and plan["url"] == "https://transfer.it/start"
        bad = await facade.save_pasted_link("not a link", path=str(epub), workspace=str(workspace))
        assert not bad["ok"] and bad["message"]
        good = await facade.save_pasted_link("https://transfer.it/t/abcDEF12", path=str(epub), workspace=str(workspace))
        assert good["ok"] and not good["can_delete"]
        assert (await facade.forget_link(good["id"]))["ok"] and facade.links_for(str(workspace)) == []

    run(scenario())


# ---------------------------------------------------------------------------
# Settings › Cloud sync & sharing
# ---------------------------------------------------------------------------


class FakeCtx:
    def __init__(self, page: Any = None) -> None:
        self.page = page
        self.said: list = []
        self.went: list = []
        self.copied: list = []
        self.tablet = False
        self.dispatcher = None

    async def run_io(self, fn: Any, *args: Any) -> Any:
        return fn(*args)

    def spawn(self, coro: Any) -> Any:
        return asyncio.ensure_future(coro)

    def say(self, message: str, action_label: Any = None, on_action: Any = None) -> None:
        self.said.append(message)

    def go(self, name: str, params: Any = None, **kwargs: Any) -> str:
        self.went.append((name, params))
        return name

    async def copy_text(self, text: str) -> None:
        self.copied.append(text)


def _screen(platform: str = "android", cloud: Any = None, shares: Any = None, page: Any = None) -> Any:
    from glossarion_mobile.ui.router import parse_route

    ctx = FakeCtx(page)
    screen = u10.CloudSyncScreen(parse_route("/settings"), ctx, cloud=cloud, shares=shares, platform=platform)
    screen.get_body()
    return screen, ctx


def test_settings_without_services_keeps_everything_visible_with_reasons():
    async def scenario():
        screen, _ctx = _screen("android")
        await screen.refresh()
        body_chips = chips(screen.body)
        assert u10.NO_SERVICE_REASON in body_chips
        assert screen.auto_switch.disabled and screen.retry_button.visible is False
        assert all(find_key(screen.destinations_column, f"cloud-dest-{m}") is not None for m in ("folder", "files",
                                                                                                "phone"))
        assert screen.summary.value == u10.NO_SERVICE_REASON
        desktop, _ = _screen("desktop", cloud=FakeCloud("desktop"), shares=FakeShares())
        await desktop.refresh()
        assert u10.PHONE_ONLY in chips(desktop.destinations_column)

    run(scenario())


def test_settings_destination_switch_formats_queue_and_forget():
    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        screen, ctx = _screen("android", cloud, shares, page)
        screen.did_show()
        await settle()
        assert screen.destination_text.value == "Not set" and screen.auto_switch.disabled
        assert u10.NO_DESTINATION_REASON in chips(screen.auto_reason)
        assert len(cloud.listeners) == 1 and len(shares.listeners) == 1
        result = await screen.choose("folder")
        assert result["ok"] and cloud.calls[-1] == ("pick_folder",)
        assert screen.destination_text.value == "Drive › Glossarion" and not screen.auto_switch.disabled
        assert "Drive › Glossarion" in " ".join(ctx.said)
        assert find_key(screen.dest_actions, "cloud-test") is not None
        # changing an existing destination asks first
        dialog = await screen.choose("phone")
        assert isinstance(dialog, u10.ConfirmDialog) and cloud.calls[-1] == ("pick_folder",)
        await dialog._on_confirm()
        assert cloud.calls[-1] == ("use_phone_folder",) and screen.destination_text.value.startswith("Phone folder")
        await screen.test_destination()  # folder mode only shows Test; the call still answers
        screen._on_auto(event(value=True))
        await settle()
        assert ("set_enabled", True) in cloud.calls and screen.auto_switch.value is True
        screen._on_kind(event(selected=False), "html")
        await settle()
        assert ("set_kind_enabled", "html", False) in cloud.calls and screen.kind_chips["html"].selected is False
        cloud.queue = [{"name": "A", "title": "A", "status": "waiting", "reason": "busy", "error": ""}]
        cloud.last_saved_at = time.time() - 120
        cloud.changed()  # the service's change event repaints
        await settle()
        assert screen.summary.value == "Last saved 2 min ago · 1 waiting"
        assert any("Waiting: the book is being compiled or translated" in t for t in texts(screen.activity_column))
        assert not screen.retry_button.disabled
        await screen.retry_now()
        await settle()
        assert ("retry_now",) in cloud.calls and ctx.said[-1] == "Retrying 1 book"
        # lost access: the banner and Choose again (no confirmation for the same destination)
        cloud.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", needs_relink="revoked")
        await screen.refresh()
        assert screen.banner.visible and any("lost access" in t for t in texts(screen.banner))
        await screen.choose("folder", confirm=False)
        assert cloud.calls[-1] == ("pick_folder",) and not screen.banner.visible
        dialog = screen.confirm_forget()
        await dialog._on_confirm()
        assert cloud.calls[-1] == ("forget_destination",) and screen.destination_text.value == "Not set"
        screen.dispose()
        assert not cloud.listeners and not shares.listeners

    run(scenario())


def test_ios_settings_offer_folder_and_save_locations_but_no_phone_folder():
    async def scenario():
        screen, _ctx = _screen("ios", FakeCloud("ios"), FakeShares(), FakePage())
        await screen.refresh()
        tile = find_key(screen.destinations_column, "cloud-dest-phone")
        assert u10.ANDROID_ONLY in chips(tile)
        assert find_key(screen.destinations_column, "cloud-dest-folder").on_click is not None
        assert "On My iPhone" in u10.NOT_LISTED_HELP_IOS

    run(scenario())


def test_provider_switches_need_the_consent_sheet_and_keys_never_stay_on_screen(caplog):
    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        screen, ctx = _screen("android", cloud, shares, page)
        await screen.refresh()
        assert set(screen.provider_switches) == {"transferit", "gofile", "send", "pixeldrain"}
        # decline: nothing is turned on and the switch goes back off
        task = asyncio.ensure_future(screen.set_provider("gofile", True))
        await settle()
        sheet = page.by_key("share-consent")
        parts = screen.actions_.last_sheet
        assert sheet is not None and parts["confirm"].disabled
        assert any("Gofile can read the file" in t for t in texts(sheet))
        parts["cancel"].on_click(None)
        assert await task is False and not shares.enabled.get("gofile")
        assert screen.provider_switches["gofile"].value is False
        # accept: the checkbox unlocks the button, the service records the consent with the switch
        task = asyncio.ensure_future(screen.set_provider("gofile", True))
        await settle()
        parts = screen.actions_.last_sheet
        parts["confirm"].on_click(None)  # nothing before the checkbox
        assert not task.done()
        parts["rights"].on_change(event(value=True))
        assert not parts["confirm"].disabled
        parts["confirm"].on_click(None)
        assert await task is True and ("set_enabled", "gofile", True, True) in shares.calls
        assert screen.provider_switches["gofile"].value is True
        assert await screen.set_provider("gofile", False) and shares.enabled["gofile"] is False
        # the pixeldrain key: saved through the service, cleared from the field, never said or logged
        field = screen.key_fields["pixeldrain"]
        assert field.password and field.can_reveal_password
        field.value = "secret-key-123"
        with caplog.at_level(logging.DEBUG):
            assert await screen.save_key("pixeldrain")
        assert shares.key == "secret-key-123" and screen.key_fields["pixeldrain"].value in ("", None)
        assert "secret-key-123" not in " ".join(ctx.said) and "secret-key-123" not in caplog.text
        assert find_key(screen.providers_column, "share-key-check-pixeldrain") is not None
        assert not await screen.check_key("pixeldrain") and ctx.said[-1] == "The service did not accept the key"
        assert await screen.remove_key("pixeldrain") and shares.key is None
        await screen.set_send_options(expire=3600)
        assert ("set_send_options", 3600, None) in shares.calls
        await screen.forget_gofile()
        assert ("forget_gofile_account",) in shares.calls

    run(scenario())


def test_settings_body_and_sheets_serialise_in_a_flet_session(tmp_path):
    tb = _tb()
    book = tmp_path / "Book.epub"
    book.write_bytes(b"0123456789abcdef")

    async def scenario():
        conn, session = tb._fake_session("android")
        page = session.page
        cloud, shares = FakeCloud(), FakeShares()
        cloud.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", needs_relink="revoked")
        cloud.queue = [{"identity": "/w/a", "title": "A", "reason": "manual", "attempts": 1, "error": "x",
                        "failed": True}]
        screen, _ctx = _screen("android", cloud, shares, page)
        page.views[0].controls.append(screen.body)
        page.update()
        await screen.refresh()
        page.update()
        actions = screen.actions_
        actions.page = page
        actions.consent_sheet({"id": "send", "label": "Send", "e2ee": True}, str(book), lambda ok: None)
        actions.link_sheet({"url": "https://x.example/1", "provider_label": "Gofile", "size": 3})
        shares.enabled["transferit"] = shares.consented["transferit"] = True
        await actions.handoff({"id": "transferit", "label": "transfer.it", "handoff": True}, str(book), str(tmp_path))
        sheet = await actions.provider_sheet(str(book), str(tmp_path))
        page.update()
        assert sheet is not None and screen.show_explainer() is not None
        assert conn.bytes_sent > 0

    run(scenario())


# ---------------------------------------------------------------------------
# Book page › Output tab
# ---------------------------------------------------------------------------


class FakeLibrary:
    def __init__(self, outputs: list) -> None:
        self.outputs = outputs
        self.prefs = None

    async def io(self, fn: Any, *args: Any) -> Any:
        return fn(*args)

    def compiled_outputs_blocking(self, book: Any) -> list:
        return list(self.outputs)

    def raw_source(self, book: Any) -> str:
        return ""

    def bid_for(self, book: Any) -> str:
        return "0123456789ab"

    def mark_dirty(self) -> None:
        pass


def _output_tab(tmp_path: Path, *, cloud: Any = None, shares: Any = None, page: Any = None, extras: Any = None):
    from glossarion_mobile.ui.library.common import LibraryContext
    from glossarion_mobile.ui.library.output_tab import OutputTab

    workspace = tmp_path / "Library" / "Novel"
    workspace.mkdir(parents=True, exist_ok=True)
    epub_new = workspace / "Novel New.epub"
    epub_old = workspace / "Novel Old.epub"
    pdf = workspace / "Novel New.pdf"
    for path, data in ((epub_new, b"new-epub"), (epub_old, b"old"), (pdf, b"%PDF-1.7 x")):
        path.write_bytes(data)
    outputs = [(str(epub_old), "epub"), (str(epub_new), "epub"), (str(pdf), "pdf")]
    library = FakeLibrary(outputs)
    said: list = []
    copied: list = []
    shared: list = []

    async def copy_text(text: str) -> None:
        copied.append(text)

    async def share_text(text: str) -> bool:
        shared.append(text)
        return True

    files = types.SimpleNamespace(share_text=share_text, export_options=lambda path: [], show_in_files=None)
    ctx = LibraryContext(service=library, page=page, notify=lambda m, a=None, o=None: said.append(m), files=files,
                         copy_text=copy_text, platform="android",
                         navigate=lambda name, params=None, query=None: said.append(("go", name)),
                         extras=dict(extras or {"cloud_sync": cloud, "share_links": shares}))
    book = {"name": "Novel", "output_folder": str(workspace), "type": "in_progress", "compiled_conflicts": [str(epub_old)]}
    fake_page = types.SimpleNamespace(ctx=ctx, service=library, book=book, bid="0123456789ab", _unsubs=[],
                                      compile=lambda kind: asyncio.sleep(0), open_files=lambda: None)
    tab = OutputTab(fake_page)
    tab.build()
    return tab, types.SimpleNamespace(workspace=str(workspace), epub_new=str(epub_new), epub_old=str(epub_old),
                                      pdf=str(pdf), said=said, copied=copied, shared=shared, page=fake_page)


def test_output_tab_cloud_lines_send_now_and_override(tmp_path):
    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        cloud.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")
        tab, env = _output_tab(tmp_path, cloud=cloud, shares=shares, page=page)
        identity = os.path.abspath(env.workspace)
        cloud.files[identity] = {
            "epub": {"status": "ok", "name": "Novel.epub", "synced_at": time.time(), "source": env.epub_new},
            "pdf": {"status": "needs_pick", "name": ""},
        }
        await tab.reload()
        assert tab.identity == identity and ("book_state", identity) in cloud.calls
        rows = tab.outputs_column.controls
        # the EPUB line sits on the file the record copies (the new title), not on the stale one listed first
        assert not any("Saved to" in t for t in texts(rows[0]))
        assert any("Saved to Drive › Glossarion · just now" in t for t in texts(rows[1]))
        assert any("Choose where to save · tap to choose" in t for t in texts(rows[2]))
        section = texts(tab.cloud_column)
        assert any(t.startswith("Copies go to Drive › Glossarion · only when you tap Send now") for t in section)
        assert find_key(tab.cloud_column, "out-cloud-send") is not None
        assert chips(tab.cloud_column) == [u10.SHARE_OFF_REASON]  # sharing services are all off
        assert (await tab.send_now())["ok"]
        assert ("send_now", identity) in cloud.calls and env.said[-1] == "Saving to Drive › Glossarion…"
        line = find_key(rows[2], "out-cloud-line-2")
        line.on_click(None)
        await settle()
        assert ("choose_save_location", identity, "pdf") in cloud.calls
        sheet = tab.open_override()
        assert [i.key for i in sheet.items] == ["cloud-override-default", "cloud-override-always", "cloud-override-never"]
        sheet.item(next(i.label for i in sheet.items if i.key == "cloud-override-always")).on_select()
        await settle()
        assert ("set_book_override", identity, "always") in cloud.calls
        assert any("Copy this book: Always" in t for t in texts(tab.cloud_column))
        row_sheet = tab.open_sheet(env.pdf, "pdf")
        keys = [i.key for i in row_sheet.items]
        assert "output-cloud-pick" in keys and "output-cloud" in keys and "output-share-link" in keys
        assert row_sheet.items[keys.index("output-share-link")].disabled_reason == u10.SHARE_OFF_REASON
        # lost access: the section says so and Send now carries the reason
        cloud.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", needs_relink="revoked")
        await tab.refresh_cloud()
        assert u10.RELINK_REASON in chips(tab.cloud_column)

    run(scenario())


def test_output_tab_without_services_or_destination_shows_reasons(tmp_path):
    async def scenario():
        tab, _env = _output_tab(tmp_path, extras={})
        await tab.reload()
        assert set(chips(tab.cloud_column)) == {u10.NO_SERVICE_REASON}
        assert not any("Saved" in t for t in texts(tab.outputs_column))
        cloud, shares = FakeCloud(), FakeShares()
        tab2, _env2 = _output_tab(tmp_path / "b", cloud=cloud, shares=shares)
        await tab2.reload()
        assert {u10.NO_DESTINATION_REASON, u10.SHARE_OFF_REASON} <= set(chips(tab2.cloud_column))
        assert find_key(tab2.cloud_column, "out-cloud-setup") is not None

    run(scenario())


def test_output_tab_share_link_upload_and_saved_links(tmp_path, caplog):
    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        shares.enabled.update(gofile=True, pixeldrain=True, transferit=True)
        shares.consented.update(gofile=True, pixeldrain=True, transferit=True)
        tab, env = _output_tab(tmp_path, cloud=cloud, shares=shares, page=page)
        identity = os.path.abspath(env.workspace)
        await tab.reload()
        assert tab.share_reason() is None
        # several compiled files: the file first, EPUBs before the PDF
        choice = await tab.share_link()
        assert [i.key for i in choice.items] == ["share-file-0", "share-file-1", "share-file-2"]
        assert choice.items[2].label.startswith("PDF")
        menu = await tab.actions.provider_sheet(env.epub_new, identity)
        reasons = {i.key: i.disabled_reason for i in menu.items}
        assert reasons["share-provider-gofile"] is None and reasons["share-provider-send"] == u10.SERVICE_OFF_REASON
        assert reasons["share-provider-pixeldrain"] == u10.NEEDS_KEY_REASON or reasons["share-provider-pixeldrain"] \
            .startswith(u10.TOO_LARGE_REASON)
        assert menu.items[-1].key == "share-settings"
        # the upload: pre-flight warning first (a large file), then progress, then the link sheet
        gofile = next(p for p, _r in tab.actions.shares.menu(env.epub_new) if p["id"] == "gofile")
        with caplog.at_level(logging.DEBUG):
            outcome = await tab.actions.upload(gofile, env.epub_new, identity)
            confirm = outcome["confirm"]
            await confirm._on_confirm()
            await settle()
        assert ("upload", "gofile", identity) in shares.calls
        ready = page.by_key("share-link-ready")
        assert ready is not None and any(t.startswith("https://gofile.example/d/1") for t in texts(ready))
        assert "gofile.example" not in caplog.text  # links (Send carries its key) never reach the log
        await tab.refresh_cloud()
        assert any(t == "Saved links (1)" for t in texts(tab.cloud_column))
        copy = find_key(tab.cloud_column, f"out-link-{tab.cloud_builds}-copy-0")
        copy.on_click(None)
        share = find_key(tab.cloud_column, f"out-link-{tab.cloud_builds}-share-0")
        share.on_click(None)
        await settle()
        assert env.copied == ["https://gofile.example/d/1#k"] and env.shared == ["https://gofile.example/d/1#k"]
        delete = find_key(tab.cloud_column, f"out-link-{tab.cloud_builds}-delete-0")
        delete.on_click(None)
        dialog = tab.actions.last_dialog
        assert dialog.title == "Delete the upload?"
        await dialog._on_confirm()
        await settle()
        assert ("delete_link", "l1") in shares.calls and not any(t.startswith("Saved links") for t in
                                                                 texts(tab.cloud_column))
        # an existing live link to the same file: shown first, Upload again on request
        shares.existing = shares._link("gofile", env.epub_new, identity, "https://gofile.example/d/old")
        result = await tab.actions.upload(gofile, env.epub_new, identity)
        assert result["existing"]["url"] == "https://gofile.example/d/old"
        assert page.by_key("share-link-ready") is page.dialogs[-1]
        # a failed upload says why; nothing is retried by itself
        shares.existing = None
        shares.fail_with = FakeShareError("rate_limited", "Gofile is busy. Try again later.")
        result = await tab.actions.run_upload(gofile, env.epub_new, identity)
        assert not result["ok"] and env.said[-1] == "Gofile: Gofile is busy. Try again later."
        assert sum(1 for c in shares.calls if c[0] == "upload") == 2

    run(scenario())


def test_transfer_it_handoff_keeps_the_pasted_link(tmp_path):
    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        shares.enabled["transferit"] = True
        tab, env = _output_tab(tmp_path, cloud=cloud, shares=shares, page=page)
        identity = os.path.abspath(env.workspace)
        await tab.reload()
        provider = next(p for p, _r in tab.actions.shares.menu(env.epub_new) if p["id"] == "transferit")
        task = asyncio.ensure_future(tab.actions.start_provider(provider, env.epub_new, identity))
        await settle()
        consent = tab.actions.last_sheet  # never consented: the consent sheet first
        consent["set_rights"](True)
        consent["confirm"].on_click(None)
        await task
        assert ("give_consent", "transferit") in shares.calls and ("start_handoff", identity) in shares.calls
        assert shares.handoff_opened == ["https://transfer.it/start"]  # the start page only; no API call
        parts = tab.actions.last_sheet
        assert any("Downloads › Glossarion" in t for t in texts(parts["dialog"]))
        parts["field"].value = "see https://evil.example/t/x"
        assert not (await parts["save"]())["ok"] and parts["error"].visible
        parts["field"].value = "https://transfer.it/t/abcdef12"
        assert (await parts["save"]())["ok"]
        assert env.said[-1] == "transfer.it link saved on the book"
        await tab.refresh_cloud()
        assert any(t == "Remove" for t in texts(tab.cloud_column))  # transfer.it links cannot be deleted from the app
        await parts["open"]()
        assert shares.handoff_opened == ["https://transfer.it/start", "https://transfer.it/start"]

    run(scenario())


# ---------------------------------------------------------------------------
# Chat Result card
# ---------------------------------------------------------------------------


def test_job_card_u10_actions_status_and_links():
    from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTION_REASONS, ATTACHMENT_ACTIONS, U10_ACTIONS, JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    assert [a[0] for a in ATTACHMENT_ACTIONS][-2:] == list(U10_ACTIONS) == ["cloud", "share_link"]
    assert ATTACHMENT_ACTION_REASONS["cloud"] == ATTACHMENT_ACTION_REASONS["share_link"] == u10.NO_SERVICE_REASON
    actions_seen: list = []
    card = JobCard(attachment={"name": "Novel.epub", "extension": ".epub", "size": 10}, phase=CardPhase("done"),
                   on_action=actions_seen.append)
    assert u10.NO_SERVICE_REASON in chips(card.buttons)  # unbound: disabled, never hidden
    assert not card.u10_box.visible
    handled: list = []
    u10_actions = u10.U10Actions()
    card.set_u10({"cloud_reason": None, "share_reason": u10.SHARE_OFF_REASON,
                  "status": ("CLOUD_DONE", "Saved to Drive › G · just now", "ok"),
                  "links": [{"id": "l1", "url": "https://x.example/1", "name": "Novel.epub", "provider_label": "Send",
                             "e2ee": True}],
                  "actions": u10_actions}, handler=handled.append)
    assert card.u10_box.visible and "Saved to Drive › G · just now" in texts(card.u10_box)
    assert "Share links (1)" in texts(card.u10_box)
    cloud_button = card.action_buttons["cloud"]
    assert isinstance(cloud_button, ft.FilledTonalButton)
    cloud_button.on_click(None)
    assert handled == ["cloud"] and actions_seen == []  # U10 actions never reach the chat's dispatcher
    assert u10.SHARE_OFF_REASON in chips(card.action_buttons["share_link"])
    card.action_buttons["read"].on_click(None)
    assert actions_seen == ["read"]
    card.set_phase(CardPhase("running"))
    assert not card.u10_box.visible


def _chat_feature(app: Any) -> Any:
    from glossarion_mobile.ui.chat.integration import ChatFeature

    feature = ChatFeature.__new__(ChatFeature)  # no history, runs or sign-in: only the U10 hooks
    feature.app = app
    feature.page = app.page
    feature.dispatcher = None
    feature.is_android, feature.is_ios = True, False
    feature._u10, feature._u10_cards, feature._u10_unsubs, feature._u10_subscribed = None, {}, [], set()
    feature._u10_rebinding = feature._u10_again = False
    feature._unsubs = []
    return feature


def test_chat_feature_binds_result_cards_to_the_turn_workspace(tmp_path):
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    workspace = tmp_path / "Library" / "Novel"
    workspace.mkdir(parents=True)
    (workspace / "Novel.epub").write_bytes(b"epub")

    async def scenario():
        cloud, shares, page = FakeCloud(), FakeShares(), FakePage()
        cloud.dest = Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")
        cloud.enabled = True
        cloud.files[str(workspace)] = {"epub": {"status": "ok", "synced_at": time.time()}}
        shares.enabled["gofile"] = shares.consented["gofile"] = True
        said: list = []
        app = types.SimpleNamespace(page=page, cloud_sync=cloud, share_links=shares, files=None, opener=None,
                                    clipboard=None, shell=None, notify=lambda m, a=None, o=None: said.append(m),
                                    navigate_to=lambda *a, **k: None, _copy_text=None)
        feature = _chat_feature(app)
        card = JobCard(attachment={"name": "Novel.epub", "extension": ".epub"}, phase=CardPhase("done"))
        state = {"in_attachments": False}
        resolve = lambda: (str(workspace), state["in_attachments"])  # noqa: E731
        task = feature.bind_u10_card(card, resolve)
        await task
        assert card.u10_state["cloud_reason"] is None and card.u10_state["share_reason"] is None
        assert card.u10_state["status"][0] == "CLOUD_DONE"
        assert feature.bind_u10_card(card, resolve) is None  # bound a moment ago: not again
        card.action_buttons["cloud"].on_click(None)
        assert await until(lambda: ("send_now", str(workspace)) in cloud.calls)
        assert await until(lambda: said and said[-1] == "Saving to Drive › Glossarion…")
        card.action_buttons["share_link"].on_click(None)
        assert await until(lambda: bool(page.dialogs))
        assert page.dialogs and getattr(feature.u10_actions().last_sheet, "title", "") == u10.SHARE_LABEL
        # still in Attachments: Send to cloud waits for the Library, the line says when it goes
        state["in_attachments"] = True
        await feature.bind_u10_card(card, resolve, force=True)
        assert card.u10_state["cloud_reason"] == u10.NOT_IN_LIBRARY_REASON
        assert card.u10_state["status"][1] == "Copied to Drive › Glossarion once the book is in the Library"
        card.action_buttons["cloud"]  # disabled with its chip
        assert u10.NOT_IN_LIBRARY_REASON in chips(card.action_buttons["cloud"])
        # a service change re-binds the live cards (at most once a second)
        assert len(cloud.listeners) == 1
        cloud.dest = None
        cloud.changed()
        await asyncio.sleep(1.2)
        await settle()
        assert card.u10_state["cloud_reason"] is None and card.u10_state["status"] is None  # the tap picks one
        feature._u10_unsubs and [u() for u in feature._u10_unsubs]

    run(scenario())


def test_chat_view_wires_the_u10_card_binding():
    """``ChatView`` hands every Result card to ``ChatEnv.bind_u10_card`` (patch applied by Integrate) and the
    chat feature puts the hook on the env."""
    view = (APP_DIR / "glossarion_mobile" / "ui" / "chat" / "chat_view.py").read_text(encoding="utf-8")
    integration = (APP_DIR / "glossarion_mobile" / "ui" / "chat" / "integration.py").read_text(encoding="utf-8")
    assert "bind_u10_card" in view and "_job_workspace_state" in view
    assert "env.bind_u10_card = self.bind_u10_card" in integration


def test_ui_sources_follow_the_dialog_and_target_rules():
    source = (APP_DIR / "glossarion_mobile" / "ui" / "screens" / "cloud_sync.py").read_text(encoding="utf-8")
    assert "pop_dialog(" not in source  # dialogs close by identity (components.dialogs.close_dialog)
    assert "import requests" not in source and "httpx" not in source and "urllib" not in source  # no network here
    row = u10.status_row("CLOUD_DONE", "x", "ok", on_click=lambda e: None)
    assert row.padding.top + row.padding.bottom + 16 >= 48  # a tappable status line is a 48 dp target
    data = (APP_DIR / "glossarion_mobile" / "ui" / "screens" / "cloud_sync.py").read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n"))  # one line ending, never mixed
