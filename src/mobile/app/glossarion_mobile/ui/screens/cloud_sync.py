"""Settings › Cloud sync & sharing (``/settings/cloud``, U10) and the U10 actions the Book page and the chat share.

**Cloud sync** (owner decisions 2026-10-09). The user may opt in to have the Library's finished books (their
compiled EPUB / PDF / TXT / HTML) copied into ONE destination picked once with the phone's own picker, and
kept up to date: a recompile replaces the same cloud file. Nothing goes through the developer: no account, no
OAuth client, no server, no telemetry; the user's own cloud app uploads under the user's own account.
Destinations:

* **a folder** (Android Storage Access Framework tree / iOS Files folder: Google Drive, Nextcloud, iCloud
  Drive, On My iPhone, any app the picker lists);
* **save locations** (one file at a time: the system "Save to" picker once per new output, then automatic
  updates; for providers that are not offered as a folder, such as Google Drive on some phones or
  Drive / OneDrive / Dropbox on iPhone);
* **the phone folder** (Android ``Downloads/Glossarion``, overwritten in place; a backup app such as TeraBox,
  FolderSync or Syncthing can upload it).

Opt-in: a global switch (off by default), a per-book Default / Always / Never override and per-format toggles
(EPUB, PDF, TXT, HTML; all on). Notifications: failures and "needs a save location" only; progress shows in the
existing foreground notification. The texts are here (``cloud_*_notice``) so every caller words them alike.

**Share file via link** (tap-only; every service off until turned on here, behind a consent sheet that says the
file leaves the phone and whether the service can read it): the transfer.it browser hand-off (the user uploads on
transfer.it's own page and pastes the link back; Glossarion never calls transfer.it), Gofile, Send (end-to-end
encrypted) and pixeldrain (the user's own API key, stored encrypted by the share-link service). Links are saved
per book and listed on the Book page and the chat Result card with Copy · Share · Delete (where the service can
delete).

What lives here (UI only; the services own every rule, record and network call):

* text helpers: destination, per-file status lines, queue reasons, link lines, notification texts;
* ``CloudFacade`` / ``ShareFacade``: the one place that names the members of the cloud-sync service
  (``app.cloud_sync``) and the share-link service (``app.share_links``), with safe answers when a service is
  missing (desktop, ``flet run``, a failed install), so every U10 control stays visible and disabled with a
  ReasonChip (UI_SPEC §0 item 6);
* ``U10Actions``: Send now, the per-book Default / Always / Never sheet, "Share file via link" (file choice →
  provider sheet → consent → upload progress → link sheet; the transfer.it hand-off with paste-back) and the
  saved-link rows; one implementation for the Book page's Output tab and the chat Result card;
* ``CloudSyncScreen``: the Settings page.

Blocking work never runs on the UI loop: service getters run through the screen's / context's io runner; the
service mutators and async calls are the services' own (they offload their IO).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip, unavailable_tile
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section
from glossarion_mobile.ui.theme import icon_data

__all__ = [
    "ANDROID_ONLY",
    "CLOUD_NOTIFICATION_BASE",
    "CLOUD_ROUTE",
    "CloudFacade",
    "CloudSyncScreen",
    "EXPLAINER",
    "FORMATS",
    "NOT_LISTED_HELP_ANDROID",
    "NOT_LISTED_HELP_IOS",
    "OVERRIDES",
    "ROUTE_NAME",
    "SETTINGS_PLACE",
    "SHARE_EXPLAINER",
    "SHARE_LABEL",
    "ShareFacade",
    "U10Actions",
    "ago_text",
    "book_status_line",
    "cloud_failure_notice",
    "cloud_needs_location_notice",
    "cloud_progress_text",
    "compiled_kinds",
    "destination_text",
    "file_status_line",
    "link_meta_text",
    "queue_reason_text",
    "share_failed_notice",
    "share_progress_text",
    "summary_text",
]

log = logging.getLogger("glossarion.cloud.ui")

# ---------------------------------------------------------------------------
# Texts (pure; the services and the tests use them too)
# ---------------------------------------------------------------------------

ROUTE_NAME = "settings.cloud"
CLOUD_ROUTE = "/settings/cloud"
SETTINGS_PLACE = "Settings › Cloud sync & sharing"
SHARE_LABEL = "Share file via link"
SEND_NOW_LABEL = "Send to cloud now"
#: Notification ids of cloud sync (``services.notifications``: 41100 FGS, 41200 done, 41600 action).
CLOUD_NOTIFICATION_BASE = 42000

#: (kind, label) of the formats cloud sync copies (UI_SPEC §3.9 compiled outputs).
FORMATS = (("epub", "EPUB"), ("pdf", "PDF"), ("txt", "TXT"), ("html", "HTML"))
_KIND_LABELS = dict(FORMATS)
#: Per-book override: (value, label, what it does).
OVERRIDES = (
    ("default", "Default", "Follows “Copy finished books automatically”"),
    ("always", "Always", "Copied after every compile, even with the switch off"),
    ("never", "Never", "Not copied automatically (Send now still copies it once)"),
)
_OVERRIDE_LABELS = {value: label for value, label, _text in OVERRIDES}

PHONE_ONLY = "Phone only"
PHONE_ONLY_DETAIL = ("Cloud sync writes through the Android or iPhone document picker. The desktop app keeps its own "
                     "output folder.")
ANDROID_ONLY = "Android only"
IOS_PHONE_FOLDER_DETAIL = ("On iPhone and iPad your books are already in Files › On My iPhone › Glossarion. To copy "
                           "them into another app, choose a folder or save locations, or use Share.")
NO_SERVICE_REASON = "Not available in this session"
NO_SERVICE_DETAIL = ("Cloud sync and share links did not start in this session (Settings › Logs & diagnostics has the "
                     "details). Your settings and links are kept.")
NO_DESTINATION_REASON = "Choose a destination first"
NO_DESTINATION_DETAIL = f"Choose a cloud folder, save locations or the phone folder in {SETTINGS_PLACE}."
RELINK_REASON = "Lost access"
RELINK_DETAIL = f"Glossarion lost access to the destination. Choose it again in {SETTINGS_PLACE}."
NOT_IN_LIBRARY_REASON = "Copied once the book is in the Library"
NOT_IN_LIBRARY_DETAIL = ("Cloud sync copies the books in your Library. A finished chat book moves into the Library by "
                         "itself, and is copied then. Share file via link works now.")
NO_OUTPUT_REASON = "Compile an EPUB or PDF first"
NO_OUTPUT_DETAIL = "Cloud sync and share links use the book's compiled files (EPUB, PDF, TXT, HTML)."
SHARE_OFF_REASON = "Turn on a service first"
SHARE_OFF_DETAIL = (f"Every sharing service is off until you turn it on in {SETTINGS_PLACE}. Nothing is uploaded "
                    "until you tap a service.")
SERVICE_OFF_REASON = "Off · turn on in Settings"
SERVICE_OFF_DETAIL = (f"This service is off. Turn it on in {SETTINGS_PLACE}: its sheet says who can read the file "
                      "and how long the link lasts.")
NEEDS_KEY_REASON = "Add your API key"
NEEDS_KEY_DETAIL = f"pixeldrain needs your own free API key: add it in {SETTINGS_PLACE}. It is stored encrypted."
TOO_LARGE_REASON = "Too large for this service"
BUSY_UPLOAD_REASON = "An upload is running"

#: The reasons a ReasonChip explains at length.
REASON_DETAILS = {
    PHONE_ONLY: PHONE_ONLY_DETAIL,
    NO_SERVICE_REASON: NO_SERVICE_DETAIL,
    NO_DESTINATION_REASON: NO_DESTINATION_DETAIL,
    RELINK_REASON: RELINK_DETAIL,
    NOT_IN_LIBRARY_REASON: NOT_IN_LIBRARY_DETAIL,
    NO_OUTPUT_REASON: NO_OUTPUT_DETAIL,
    SHARE_OFF_REASON: SHARE_OFF_DETAIL,
    SERVICE_OFF_REASON: SERVICE_OFF_DETAIL,
    NEEDS_KEY_REASON: NEEDS_KEY_DETAIL,
}
#: Reasons fixed in Settings › Cloud sync & sharing (their snackbar offers "Settings").
SETTINGS_REASONS = (NO_DESTINATION_REASON, RELINK_REASON, SHARE_OFF_REASON, SERVICE_OFF_REASON, NEEDS_KEY_REASON)

EXPLAINER_TITLE = "What Glossarion can and can't see"
EXPLAINER = f"""\
**How it works.** Glossarion copies the finished books in your Library into the place you chose with the
phone's own picker. Your cloud app (Google Drive, Nextcloud, iCloud Drive, OneDrive, …) then uploads them under
your account, with your own settings (for example Wi-Fi only).

**Nothing goes through Glossarion's developer.** There is no Glossarion account, server, sign-in or telemetry.
Glossarion never sees your cloud password or your other files: only the folder, or the single files, you picked.

**What it cannot see.** Glossarion only learns whether the phone accepted the file. It cannot tell whether your
cloud app finished the upload: full storage, a signed-out cloud app or uploads limited to Wi-Fi stay invisible.
If a copy is missing, check your cloud app.

**One-way copy.** A recompiled book replaces its cloud copy; changes made to the cloud copy elsewhere are
overwritten. Google Drive and some other services keep earlier versions. A book whose title changes keeps its
first cloud file name. Some providers cannot overwrite a file: Glossarion then saves a new file and removes the old
one, and the cloud link of that file changes (the book's status says so).

**Forget destination** stops the copies and gives back Glossarion's access to the folder or files. Files already
copied stay where they are. Uninstalling the app (or wiping its data) forgets the destination too; after
reinstalling, choose the same folder again: Glossarion reuses the files with the same names instead of making
copies. Deleting a book in the Library keeps its cloud copy.

You can change all of this in {SETTINGS_PLACE}."""

NOT_LISTED_HELP_ANDROID = """\
**Folder backup apps.** TeraBox and some other cloud apps do not offer themselves in the system picker. Choose the
**phone folder** (Downloads/Glossarion) as the destination, then turn on the folder backup of your app for
Download/Glossarion (TeraBox › Automatic backup, FolderSync, Syncthing). On Android 11 and newer, give that app
access to the folder through its own folder picker (or All files access), or it may not see EPUB and PDF files.

**More services without sign-in in Glossarion.** RSAF, an open-source document provider for rclone, adds Dropbox,
OneDrive, pCloud, WebDAV and many others to the folder picker; your sign-in stays inside RSAF. (rclone has no
TeraBox support.)

**Not a folder?** If your cloud app appears only when saving a single file (Google Drive on some phones), choose
**Save each file separately**: you pick the location once per file, later compiles replace it.

**One-off copies** always work with Share on the Book page."""

NOT_LISTED_HELP_IOS = """\
Your books are already in **Files › On My iPhone › Glossarion**.

Choose a **folder** for iCloud Drive or On My iPhone. Google Drive, OneDrive and Dropbox cannot be picked as a
folder by other apps: choose **Save each file separately** (you pick the location once per file, later compiles
replace it).

Apps that do not appear in Files (TeraBox) can still get a book through **Share** on the Book page."""

SHARE_EXPLAINER = f"""\
**Share file via link** uploads one file to a sharing service you turned on, so you can send the link to someone.
It runs only when you tap it; nothing is shared automatically.

- The file leaves your phone. Anyone with the link can download it, and links can be forwarded.
- **Send** is end-to-end encrypted: the key is in the link, the service cannot read the file. The other services
  can read the file.
- The service records your IP address and the upload time. Glossarion's developer receives nothing.
- **transfer.it** has no app integration: Glossarion opens its page, you upload there and paste the link back.
- Only share files you have the right to share. Services remove infringing files and can block accounts.
- Uploads use mobile data on a mobile connection.

Turn services on or off in {SETTINGS_PLACE}."""

#: Queue reasons (the cloud-sync service's queue ``reason`` / ``last_error``) -> the "Waiting: …" text.
QUEUE_REASONS = {
    # why a book is held back
    "busy": "the book is being compiled or translated",
    "compiling": "the book is compiling",
    "not_in_library": "waiting for the book to move to the Library",
    "attachments": "waiting for the book to move to the Library",
    "deferred": "waiting for the book to move to the Library",
    "needs_pick": "needs a save location",
    "needs_relink": "lost access to the destination",
    "permission_lost": "lost access to the destination",
    "revoked": "lost access to the destination",
    "provider_error": "the cloud app reported an error",
    "provider": "the cloud app reported an error",
    "unsupported_mode": "the cloud app could not overwrite the file",
    "size_mismatch": "the cloud app reported a different size",
    "no_space": "not enough free space on the phone",
    "source_missing": "the book's file is gone",
    "source_changed": "the book changed while it was copied",
    "missing": "the cloud file is gone",
    "unavailable": "the cloud app did not answer",
    "offline": "the cloud app is offline",
    "cancelled": "the copy was stopped",
    "background": "continues when Glossarion is open",
    # why a book was queued
    "queued": "waiting its turn",
    "job": "queued after a compile",
    "manual": "queued by Send now",
    "enabled": "queued when copying was turned on",
    "format": "queued when a format was turned on",
    "override": "queued by the book's setting",
    "linked": "queued for the new destination",
    "relinked": "queued for the destination",
    "library": "queued after the move to the Library",
    "save location": "queued for its save location",
    "retry": "retrying",
    "backoff": "retrying later",
    "timer": "retrying later",
    "resume": "left over from last time",
}


def compiled_kinds() -> tuple:
    return tuple(kind for kind, _label in FORMATS)


def _now(now: Optional[float]) -> float:
    return time.time() if now is None else float(now)


def ago_text(ts: Any, now: Optional[float] = None) -> str:
    """"just now" / "3 min ago" / "2 h ago" / "yesterday" / "4 days ago" ("" without a time)."""
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    delta = max(0.0, _now(now) - value)
    if delta < 45:
        return "just now"
    if delta < 3600:
        return f"{max(1, int(round(delta / 60)))} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    if delta < 2 * 86400:
        return "yesterday"
    return f"{int(delta // 86400)} days ago"


def destination_text(dest: Optional[Mapping]) -> str:
    """"Not set" / "Drive › Glossarion" / "Drive · one file at a time" / "Phone folder · Downloads/Glossarion"."""
    if not dest:
        return "Not set"
    mode = str(dest.get("mode") or "folder")
    label = str(dest.get("label") or "").strip()
    provider = str(dest.get("provider_label") or "").strip()
    if mode == "phone":
        return "Phone folder · " + (label or "Downloads/Glossarion")
    if mode == "files":
        base = provider or ("" if label.casefold() in ("", "chosen per file") else label)
        return (base or "Save locations") + " · one file at a time"
    if provider and label and not label.casefold().startswith(provider.casefold()):
        return f"{provider} › {label}"
    return label or provider or "Chosen folder"


def queue_reason_text(reason: Any) -> str:
    key = str(reason or "").strip()
    head = key.split(":", 1)[0]
    return QUEUE_REASONS.get(key) or QUEUE_REASONS.get(head) or (key.replace("_", " ") if key else "waiting")


def _percent(written: Any, total: Any) -> Optional[int]:
    try:
        written_f, total_f = float(written or 0), float(total or 0)
    except (TypeError, ValueError):
        return None
    if total_f <= 0:
        return None
    return max(0, min(100, int(written_f * 100 // total_f)))


def file_status_line(entry: Optional[Mapping], *, dest_label: str = "", kind: str = "",
                     now: Optional[float] = None) -> Optional[tuple]:
    """``(icon, text, tone)`` of one output's cloud state (Output tab rows; tone ok / busy / warn / error /
    action / muted), or None when there is nothing to say.

    ``entry`` (``CloudFacade.book``): ``status`` (ok · writing · waiting · failed · check · needs_pick ·
    needs_relink · missing · source_missing · off · format_off · new), the service's own ``text`` when it has
    one (the wording follows the service), ``synced_at``, ``written`` / ``total`` (while writing), ``reason``
    (waiting), ``error``, ``warning`` (saved with a caveat, such as a replaced file whose link changed). The
    tappable states say what a tap does (retry / choose / reconnect)."""
    if not entry:
        return None
    status = str(entry.get("status") or "")
    dest = dest_label or "the cloud"
    error = str(entry.get("error") or entry.get("message") or "").strip()
    given = str(entry.get("text") or "").strip()
    if status in ("ok", "saved", "done"):
        warning = str(entry.get("warning") or "").strip()
        when = ago_text(entry.get("synced_at") or entry.get("at"), now)
        text = given or f"Saved to {dest}" + (f" · {when}" if when else "")
        if warning:
            return "WARNING_AMBER", f"{text} · {warning}", "warn"
        return "CLOUD_DONE", text, "ok"
    if status == "writing":
        pct = _percent(entry.get("written"), entry.get("total"))
        return "CLOUD_UPLOAD", "Saving…" + (f" {pct}%" if pct is not None else ""), "busy"
    if status in ("pending", "waiting", "queued"):
        reason = entry.get("reason")
        text = ("Waiting: " + queue_reason_text(reason)) if reason else (given or "Waiting to save")
        return "CLOUD_QUEUE", text, "busy"
    if status in ("failed", "error"):
        text = given or ("Couldn't save" + (f": {error}" if error else ""))
        return "SYNC_PROBLEM", text + " · tap to retry", "error"
    if status == "check":
        return "WARNING_AMBER", given or "Saved, but the size did not match: will try again", "warn"
    if status == "needs_pick":
        return "DRIVE_FILE_MOVE", (given or "Choose where to save") + " · tap to choose", "action"
    if status in ("needs_relink", "revoked", "permission_lost"):
        return "SYNC_PROBLEM", (given or "Lost access") + " · tap to reconnect", "error"
    if status == "missing":
        return "CLOUD_QUEUE", given or "The cloud file is gone · saved again on the next copy", "warn"
    if status == "source_missing":
        return "CLOUD_OFF", given or "The book's file is gone", "warn"
    if status == "off":
        return "CLOUD_OFF", given or "Not copied (off for this book)", "muted"
    if status == "format_off":
        label = _KIND_LABELS.get(kind, kind.upper() or "This format")
        return "CLOUD_OFF", given or f"{label} is not synced", "muted"
    if status == "new":
        return "CLOUD_QUEUE", given or "Not sent yet", "muted"
    return None


def book_status_line(state: Optional[Mapping], book: Optional[Mapping], *, in_library: bool = True,
                     has_outputs: bool = True, now: Optional[float] = None) -> Optional[tuple]:
    """``(icon, text, tone, action)`` summing up one book's cloud state (the chat Result card), or None when
    there is nothing to say (no destination: the buttons carry the reason). ``action`` is what a tap on the line
    does: ``settings`` (reconnect) · ``retry`` · ``book`` (choose a save location on the Book page) · None."""
    dest = state.get("destination") if state else None
    if not state or not state.get("available", True) or not isinstance(dest, Mapping) or not dest:
        return None
    label = destination_text(dest)
    if dest.get("needs_relink"):
        return "SYNC_PROBLEM", f"Lost access to {label} · Reconnect", "error", "settings"
    if not in_library:
        if state.get("enabled") and has_outputs:
            return "CLOUD_QUEUE", f"Copied to {label} once the book is in the Library", "muted", None
        return None
    files = [dict(v) for v in dict((book or {}).get("files") or {}).values() if isinstance(v, Mapping)]

    def first(*statuses: str) -> Optional[dict]:
        return next((entry for entry in files if str(entry.get("status") or "") in statuses), None)

    entry = first("writing")
    if entry is not None:
        icon, text, tone = file_status_line(entry, dest_label=label, now=now) or ("CLOUD_UPLOAD", "Saving…", "busy")
        return icon, text, tone, None
    if first("needs_relink", "revoked", "permission_lost") is not None:
        return "SYNC_PROBLEM", f"Lost access to {label} · Reconnect", "error", "settings"
    if first("needs_pick") is not None:
        return "DRIVE_FILE_MOVE", "Choose where to save it (Book page › Output)", "action", "book"
    entry = first("failed", "error")
    if entry is not None:
        error = str(entry.get("error") or entry.get("message") or "").strip()
        return "SYNC_PROBLEM", f"Couldn't save to {label}" + (f": {error}" if error else "") + " · Retry", "error", "retry"
    entry = first("pending", "waiting", "queued")
    if entry is not None:
        reason = entry.get("reason")
        return "CLOUD_QUEUE", ("Waiting: " + queue_reason_text(reason)) if reason else "Waiting to save", "busy", None
    saved = [entry for entry in files if str(entry.get("status") or "") in ("ok", "saved", "done", "check")]
    if saved:
        warned = next((e for e in saved if e.get("warning") or e.get("status") == "check"), None)
        if warned is not None:
            icon, text, tone = file_status_line(warned, dest_label=label, now=now) or ("WARNING_AMBER", label, "warn")
            return icon, text, tone, None
        try:
            newest = max(float(e.get("synced_at") or e.get("at") or 0) for e in saved)
        except (TypeError, ValueError):
            newest = 0.0
        when = ago_text(newest, now)
        return "CLOUD_DONE", f"Saved to {label}" + (f" · {when}" if when else ""), "ok", None
    if str((book or {}).get("override") or "") == "never":
        return "CLOUD_OFF", "Not copied to the cloud (off for this book)", "muted", None
    return None


def summary_text(state: Mapping, now: Optional[float] = None) -> str:
    """"Last saved 3 min ago · 2 waiting · 1 failed" (the Settings activity line)."""
    parts: list = []
    when = ago_text(state.get("last_saved_at"), now)
    parts.append(f"Last saved {when}" if when else "Nothing saved yet")
    queue = list(state.get("queue") or ())
    waiting = sum(1 for item in queue if str(item.get("status") or "waiting") != "failed")
    failed = int(state.get("failed") or 0) or sum(1 for item in queue if str(item.get("status") or "") == "failed")
    if waiting:
        parts.append(f"{waiting} waiting")
    if failed:
        parts.append(f"{failed} failed")
    return " · ".join(parts)


def link_meta_text(link: Mapping, now: Optional[float] = None) -> str:
    """"Gofile · 4.8 MB · 2 h ago · kept while it is downloaded" for a saved link row."""
    parts = [str(link.get("provider_label") or link.get("provider") or "Link")]
    try:
        size = int(link.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size:
        parts.append(human_size(size))
    when = ago_text(link.get("created_at"), now)
    if when:
        parts.append(when)
    expires = str(link.get("expires_text") or "").strip()
    if expires:
        parts.append(expires)
    return " · ".join(parts)


# ---- notification texts (payloads are routes, never paths or URIs) ----------------------------------------------


#: The progress sheet's Cancel once the whole file was sent (``share_providers.MAYBE_CANCELLED``).
STOP_WAITING_LABEL = "Stop waiting"


def cloud_failure_notice(dest_label: str, *, lost_access: bool = False, count: int = 1, unit: str = "file") -> tuple:
    """``(title, body, route)`` of the one failure notification per destination (``count`` ``unit``s waiting:
    the cloud sync counts books, one notification per drain)."""
    dest = dest_label or "your cloud"
    if lost_access:
        return f"Glossarion lost access to {dest}", "Tap to choose it again", CLOUD_ROUTE
    count = max(1, int(count or 1))
    return (f"Couldn't save to {dest}", f"{count} {unit}{'s' if count != 1 else ''} waiting · tap for details",
            CLOUD_ROUTE)


def cloud_needs_location_notice(count: int, bid: Optional[str] = None) -> tuple:
    """``(title, body, route)``: save-locations mode needs the user to pick where a new file goes."""
    count = max(1, int(count or 1))
    title = "1 book needs a save location" if count == 1 else f"{count} books need a save location"
    body = "Tap to choose where to save it" if count == 1 else "Tap to choose where to save them"
    route = f"/library/book/{bid}?tab=output" if count == 1 and bid else CLOUD_ROUTE
    return title, body, route


def cloud_progress_text(name: str, dest_label: str, written: Any = 0, total: Any = 0) -> str:
    """The foreground notification line while a copy runs: "Saving Book.epub to Drive · 45%"."""
    pct = _percent(written, total)
    return f"Saving {name} to {dest_label or 'the cloud'}" + (f" · {pct}%" if pct is not None else "…")


def share_progress_text(name: str, provider_label: str, sent: Any = 0, total: Any = 0) -> str:
    pct = _percent(sent, total)
    return f"Uploading {name} to {provider_label}" + (f" · {pct}%" if pct is not None else "…")


def share_failed_notice(name: str, provider_label: str, error: str = "", bid: Optional[str] = None) -> tuple:
    """``(title, body, route)`` when an upload the user started fails while the app is in the background."""
    route = f"/library/book/{bid}?tab=output" if bid else CLOUD_ROUTE
    body = f"{name}: {error}" if error else f"{name}: tap to try again"
    return f"Upload to {provider_label} failed", body[:200], route


# ---------------------------------------------------------------------------
# Service facades
# ---------------------------------------------------------------------------


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _source(source: Any) -> Any:
    """A service, or a zero-argument function returning it (``ctx.extras`` lambdas, app attributes read late)."""
    if inspect.isfunction(source) or inspect.ismethod(source):
        try:
            return source()
        except Exception:
            log.debug("resolving a U10 service failed", exc_info=True)
            return None
    return source


def _dest_dict(dest: Any) -> Optional[dict]:
    """A ``cloud_sync.Destination`` (or its dict) as the UI's dict (no target URI / bookmark: never shown)."""
    if dest is None:
        return None
    if isinstance(dest, Mapping):
        raw = dict(dest)
    else:
        raw = {name: getattr(dest, name, None) for name in ("id", "mode", "label", "provider_label", "can_create",
                                                            "can_write", "needs_relink", "layout")}
        try:
            raw["display"] = dest.display
        except Exception:
            pass
    raw.pop("target", None)
    raw.pop("path", None)
    raw["needs_relink"] = raw.get("needs_relink") or None
    return raw


def _result(value: Any, default_ok: bool = True) -> dict:
    """A service answer as ``{ok, message, cancelled, …}``: dicts pass through; a ``LinkResult`` maps ``reason``
    / ``note`` to ``message``; True / False / None become ok."""
    if isinstance(value, Mapping):
        out = dict(value)
        out.setdefault("ok", default_ok and not out.get("error"))
        out["message"] = str(out.get("message") or out.get("reason") or "")
        return out
    if value is None:
        return {"ok": default_ok, "message": ""}
    if isinstance(value, bool):
        return {"ok": value, "message": ""}
    if hasattr(value, "ok"):  # cloud_sync.LinkResult and similar dataclasses
        ok = bool(getattr(value, "ok"))
        reason = str(getattr(value, "reason", "") or "")
        note = str(getattr(value, "note", "") or "")
        return {"ok": ok, "message": (note if ok else reason) or reason or note, "note": note,
                "cancelled": bool(getattr(value, "cancelled", False)),
                "destination": _dest_dict(getattr(value, "destination", None)), "value": value}
    return {"ok": True, "message": "", "value": value}


#: ``cloud_sync.KindStatus.state`` -> the UI's entry status (``file_status_line``).
_KIND_STATES = {
    "off": "off", "kind_off": "format_off", "waiting_library": "waiting", "waiting": "waiting", "saving": "writing",
    "saved": "ok", "failed": "failed", "needs_pick": "needs_pick", "needs_relink": "needs_relink", "check": "check",
    "not_sent": "new", "source_missing": "source_missing",
}


class CloudFacade:
    """The one place that names the cloud-sync service's members (``services.cloud_sync.CloudSyncService``,
    ``app.cloud_sync``); a missing service (desktop, ``flet run``, a failed install) answers safely.

    Getters (``snapshot`` → the service's ``ui_state()``, ``book`` → ``book_state(identity)``) run on the UI's io
    runner. ``call(op, …)`` runs an action: ``set_enabled``, ``set_kind`` (``set_kind_enabled``), ``set_override``
    (``set_book_override``), ``pick_folder``, ``use_save_locations``, ``use_phone_folder``, ``forget_destination``,
    ``test_destination``, ``send_now``, ``choose_save_location``, ``choose_existing_file``; answers are
    ``{ok, message, …}`` dicts (``_result`` also maps an older ``LinkResult``). Each op lists the member names it
    accepts, first match wins, so a service from before the UI contract still works."""

    OPS = {
        "set_enabled": ("set_enabled",),
        "set_kind": ("set_kind_enabled", "set_kind"),
        "set_override": ("set_book_override", "set_override"),
        "pick_folder": ("pick_folder", "link_folder"),
        "use_save_locations": ("use_save_locations", "use_files_mode"),
        "use_phone_folder": ("use_phone_folder",),
        "forget_destination": ("forget_destination",),
        "test_destination": ("test_destination",),
        "send_now": ("send_now",),
        "choose_save_location": ("choose_save_location",),
        "choose_existing_file": ("choose_existing_file",),
    }

    def __init__(self, service: Any = None) -> None:
        self._source = service

    @property
    def service(self) -> Any:
        return _source(self._source)

    @property
    def available(self) -> bool:
        return self.service is not None

    def _fn(self, *names: str) -> Optional[Callable[..., Any]]:
        service = self.service
        if service is None:
            return None
        for name in names:
            fn = getattr(service, name, None)
            if callable(fn):
                return fn
        return None

    # ---- getters (blocking-safe: call through io) ----

    @staticmethod
    def _default_state() -> dict:
        return {"available": False, "supported": False, "reason": NO_SERVICE_REASON, "enabled": False,
                "kinds": {kind: True for kind in compiled_kinds()}, "destination": None, "queue": [], "recent": [],
                "last_saved_at": None, "failed": 0, "needs_pick": 0, "progress": None, "phone_folder": False}

    def snapshot(self) -> dict:
        """Settings state: ``available``, ``supported``, ``reason``, ``enabled``, ``kinds``, ``destination``
        (``{mode, label, provider_label, display, needs_relink, can_create}``: never a URI or bookmark),
        ``queue`` / ``recent`` (``{name, title, kind, status, reason, error, at, bid}``), ``last_saved_at``,
        ``failed``, ``needs_pick``, ``progress`` (``{name, written, total}`` or None), ``phone_folder``."""
        state = self._default_state()
        service = self.service
        if service is None:
            return state
        state["available"] = True
        try:
            fn = self._fn("ui_state")
            raw = fn() if fn is not None else self._derived_state(service)
        except Exception:
            log.exception("reading the cloud sync state failed")
            return state
        if not isinstance(raw, Mapping):
            return state
        state.update({k: v for k, v in raw.items() if k in state or k in ("explainer",)})
        state["available"] = True
        supported = bool(raw.get("supported", True))
        state["supported"] = supported
        state["reason"] = None if supported else str(raw.get("reason") or PHONE_ONLY)
        kinds = {kind: True for kind in compiled_kinds()}
        kinds.update({str(k): bool(v) for k, v in dict(raw.get("kinds") or {}).items()})
        state["kinds"] = kinds
        state["destination"] = _dest_dict(raw.get("destination"))
        state["queue"] = [dict(item) for item in raw.get("queue") or () if isinstance(item, Mapping)]
        state["recent"] = [dict(item) for item in raw.get("recent") or () if isinstance(item, Mapping)]
        progress = raw.get("progress")
        state["progress"] = dict(progress) if isinstance(progress, Mapping) else None
        return state

    @staticmethod
    def _derived_state(service: Any) -> dict:
        """``ui_state`` of a service that only has ``settings`` / ``summary`` / ``queue_items``."""
        settings = service.settings()
        summary = dict(service.summary() or {}) if callable(getattr(service, "summary", None)) else {}
        items = []
        for item in (service.queue_items() if callable(getattr(service, "queue_items", None)) else ()) or ():
            if isinstance(item, Mapping):
                error = str(item.get("error") or "")
                items.append({"title": str(item.get("title") or "Book"), "name": str(item.get("title") or "Book"),
                              "status": "failed" if item.get("failed") else "waiting",
                              "reason": error or str(item.get("reason") or ""), "error": "" if error == "busy" else error})
        supported = bool(getattr(service, "supported", True))
        return {"supported": supported, "enabled": bool(getattr(settings, "enabled", False)),
                "kinds": dict(getattr(settings, "kinds", {}) or {}),
                "destination": getattr(settings, "destination", None), "queue": items,
                "last_saved_at": summary.get("last_saved_at") or None, "failed": int(summary.get("failed") or 0),
                "needs_pick": int(summary.get("needs_pick") or 0),
                "progress": {"name": "", "written": 0, "total": 0} if summary.get("saving") else None,
                "phone_folder": getattr(service, "platform", "") == "android" and supported}

    def book(self, identity: str) -> dict:
        """The Book page / chat card state of one book: ``override``, ``auto`` (copied after each compile),
        ``in_library``, ``files`` ``{kind: entry}`` (``file_status_line``)."""
        state: dict = {"override": "default", "auto": False, "files": {}, "in_library": True}
        service = self.service
        if service is None or not identity:
            return state
        try:
            fn = self._fn("book_state")
            raw = fn(identity) if fn is not None else self._derived_book(service, identity)
        except Exception:
            log.exception("reading a book's cloud state failed")
            return state
        if not isinstance(raw, Mapping):
            return state
        state.update(dict(raw))
        state["files"] = {str(k): dict(v) for k, v in dict(raw.get("files") or {}).items() if isinstance(v, Mapping)}
        if state.get("override") not in _OVERRIDE_LABELS:
            state["override"] = "default"
        state["in_library"] = bool(raw.get("in_library", True)) and not raw.get("waiting_library")
        return state

    @staticmethod
    def _derived_book(service: Any, identity: str) -> dict:
        """``book_state`` of a service that only has ``status_for`` (``KindStatus`` per kind)."""
        statuses = dict(service.status_for(identity) or {})
        override = str(service.override(identity) or "default")
        enabled = bool(getattr(service.settings(), "enabled", False))
        files: dict = {}
        in_library = True
        for kind, status in statuses.items():
            def get(name: str, s: Any = status) -> Any:
                return s.get(name) if isinstance(s, Mapping) else getattr(s, name, None)

            raw = str(get("state") or "")
            mapped = _KIND_STATES.get(raw)
            if mapped is None:  # no_destination: the section says so
                continue
            entry = {"status": mapped, "text": str(get("text") or ""), "name": str(get("name") or ""),
                     "synced_at": get("synced_at") or 0, "error": str(get("error") or "")}
            if get("note"):
                entry["warning"] = str(get("note"))
            if get("source"):
                entry["source"] = str(get("source"))
            progress = get("progress")
            if mapped == "writing" and progress is not None:
                entry["written"], entry["total"] = int(float(progress) * 1000), 1000
            if raw == "waiting_library":
                entry["reason"] = "not_in_library"
                in_library = False
            files[str(kind)] = entry
        return {"override": override, "files": files, "in_library": in_library,
                "auto": override == "always" or (override == "default" and enabled)}

    # ---- actions ----

    async def call(self, op: str, *args: Any) -> dict:
        fn = self._fn(*self.OPS.get(op, (op,)))
        if fn is None:
            return {"ok": False, "message": NO_SERVICE_REASON, "unavailable": True}
        try:
            return _result(await _resolve(fn(*args)))
        except Exception as exc:
            log.exception("cloud sync %s failed", op)
            return {"ok": False, "message": str(exc) or exc.__class__.__name__}

    async def retry_now(self, spawn: Optional[Callable[[Any], Any]] = None) -> dict:
        """Settings › Retry now: every waiting book is due now and a drain starts (it is not awaited)."""
        service = self.service
        if service is None:
            return {"ok": False, "message": NO_SERVICE_REASON}
        retry = self._fn("retry_now")
        if retry is not None:
            return _result(await _resolve(retry()))
        store = getattr(service, "store", None)
        count = 0
        try:
            for entry in list(store.queue() if store is not None else ()):
                store.enqueue(entry["identity"], "retry")  # due now, attempts reset; manual stays as it was
                count += 1
        except Exception as exc:
            log.exception("retrying the cloud queue failed")
            return {"ok": False, "message": str(exc)}
        drain = self._fn("drain_now")
        if drain is not None and count:
            coro = drain("retry")
            if spawn is not None:
                spawn(coro)
            else:
                asyncio.ensure_future(coro)
        return {"ok": bool(count), "message": f"Retrying {count} waiting book{'s' if count != 1 else ''}"
                if count else "Nothing is waiting"}

    def subscribe(self, listener: Callable[..., Any]) -> Callable[[], None]:
        fn = self._fn("subscribe")
        if fn is None:
            return lambda: None
        try:
            unsub = fn(listener)
        except Exception:
            log.debug("subscribing to cloud sync failed", exc_info=True)
            return lambda: None
        return unsub if callable(unsub) else (lambda: None)


def _provider_dict(state: Any) -> dict:
    """A ``share_links.ProviderState`` (or a dict) as the UI's provider dict."""
    if isinstance(state, Mapping):
        out = {"label": str(state.get("id") or ""), "operator": "", "e2ee": False, "retention": "", "enabled": False,
               "consented": False, "needs_key": False, "has_key": False, "can_delete": False, "handoff": False,
               "max_bytes": 0, "terms_url": "", "reason": None}
        out.update(dict(state))
        out["id"] = str(state.get("id") or "")
        return out
    info = getattr(state, "info", None)
    return {
        "id": str(getattr(info, "id", "") or getattr(state, "id", "")),
        "label": str(getattr(info, "label", "") or ""),
        "operator": str(getattr(info, "operator", "") or ""),
        "country": str(getattr(info, "country", "") or ""),
        "site": str(getattr(info, "site", "") or ""),
        "e2ee": bool(getattr(info, "e2ee", False)),
        "handoff": not bool(getattr(info, "direct", True)),
        "needs_key": bool(getattr(info, "needs_api_key", False)),
        "can_delete": bool(getattr(info, "can_delete", False)),
        "max_bytes": int(getattr(info, "max_bytes", 0) or 0),
        "retention": str(getattr(info, "retention", "") or ""),
        "terms_url": str(getattr(info, "terms_url", "") or ""),
        "enabled": bool(getattr(state, "enabled", False)),
        "consented": bool(getattr(state, "consented", False)),
        "has_key": bool(getattr(state, "has_key", False)),
        "reason": getattr(state, "reason", None),
    }


def _expires_text(expires: Any, now: float) -> str:
    try:
        value = float(expires or 0)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    left = value - now
    if left <= 0:
        return "expired"
    if left < 3600:
        return f"expires in {max(1, int(left // 60))} min"
    if left < 48 * 3600:
        return f"expires in {int(left // 3600)} h"
    return f"expires in {int(left // 86400)} days"


def _link_dict(link: Any, now: Optional[float] = None) -> dict:
    """A ``share_links.ShareLink`` (or a dict) as the UI's link dict (the URL only goes to the screen)."""
    if isinstance(link, Mapping):
        return dict(link)
    clock = _now(now)
    out = {
        "id": str(getattr(link, "id", "") or ""), "provider": str(getattr(link, "provider", "") or ""),
        "url": str(getattr(link, "url", "") or ""), "name": str(getattr(link, "name", "") or ""),
        "size": int(getattr(link, "size", 0) or 0), "created_at": float(getattr(link, "created", 0) or 0),
        "can_delete": bool(getattr(link, "can_delete", False)), "expires_text": _expires_text(
            getattr(link, "expires", None), clock),
    }
    for name in ("label", "e2ee", "expired"):
        try:
            value = getattr(link, name)
        except Exception:
            continue
        out["provider_label" if name == "label" else name] = value
    limit = getattr(link, "downloads_limit", None)
    if limit:
        out["expires_text"] = " · ".join(t for t in (out["expires_text"], f"{limit} download{'s' if limit != 1 else ''}")
                                         if t)
    return out


#: Long share-service reasons -> short ReasonChip labels (the ActionSheet row keeps its width).
_SHORT_REASONS = (
    ("Turn it on", SERVICE_OFF_REASON),
    ("API key", NEEDS_KEY_REASON),
    ("Another upload", BUSY_UPLOAD_REASON),
    ("accept", "Accept its terms first"),
    ("Only book files", "Not a book file"),
    ("Secure storage", "No secure storage"),
)


def short_reason(reason: Any) -> Optional[str]:
    if not reason:
        return None
    text = str(reason)
    for needle, short in _SHORT_REASONS:
        if needle.lower() in text.lower():
            return short
    return text if len(text) <= 40 else text[:39] + "…"


class ShareFacade:
    """The one place that names the share-link service's members (``services.share_links.ShareLinkService``,
    ``app.share_links``); a missing service answers safely. Provider states, links and consent texts come back
    as plain dicts (``_provider_dict`` / ``_link_dict``); errors (``ShareError``) as ``{ok: False, message,
    code}``. Uploads report progress through the service's ``upload`` events (``state.sent`` / ``total``)."""

    def __init__(self, service: Any = None) -> None:
        self._source = service

    @property
    def service(self) -> Any:
        return _source(self._source)

    @property
    def available(self) -> bool:
        return self.service is not None

    def _fn(self, name: str) -> Optional[Callable[..., Any]]:
        service = self.service
        fn = getattr(service, name, None) if service is not None else None
        return fn if callable(fn) else None

    # ---- getters (blocking-safe: through io) ----

    def providers(self) -> list:
        fn = self._fn("provider_states") or self._fn("providers")
        if fn is None:
            return []
        try:
            raw = fn()
        except Exception:
            log.exception("listing the share-link services failed")
            return []
        if isinstance(raw, Mapping):  # ``providers()`` of the service is its provider objects: not states
            return []
        return [p for p in (_provider_dict(item) for item in raw or ()) if p.get("id")]

    def menu(self, path: str) -> list:
        """``[(provider dict, short reason | None)]`` for "Share file via link" on ``path``."""
        fn = self._fn("menu")
        if fn is not None:
            try:
                return [(_provider_dict(state), short_reason(reason)) for state, reason in fn(path) or ()]
            except Exception:
                log.exception("the share-link menu failed")
        return [(p, short_reason(p.get("reason"))) for p in self.providers()]

    def consent(self, provider: Mapping, path: Optional[str] = None, size: Optional[int] = None) -> dict:
        """``{title, lines, checkbox, confirm, terms_url}`` of a provider's consent sheet (the service's text)."""
        label = str(provider.get("label") or provider.get("id") or "the service")
        fallback = {"title": f"{SHARE_LABEL} · {label}", "lines": [
            f"This sends the file off your phone to {label}. Anyone with the link can download it.",
            "End-to-end encrypted: the service cannot read the file." if provider.get("e2ee") else
            f"Not end-to-end encrypted: {label} can read the file."],
            "checkbox": "I have the right to share files I upload", "confirm": "Continue", "terms_url": ""}
        fn = self._fn("consent_text")
        if fn is None:
            return fallback
        try:
            text = fn(provider.get("id"), path, size)
        except TypeError:
            try:
                text = fn(provider.get("id"), path)
            except Exception:
                return fallback
        except Exception:
            log.debug("consent text failed", exc_info=True)
            return fallback
        if isinstance(text, str):
            return dict(fallback, lines=[text])
        if isinstance(text, Mapping):
            return dict(fallback, **dict(text))
        return {"title": str(getattr(text, "title", "") or fallback["title"]),
                "lines": [str(line) for line in getattr(text, "lines", ()) or () if line] or fallback["lines"],
                "checkbox": str(getattr(text, "checkbox", "") or fallback["checkbox"]),
                "confirm": str(getattr(text, "confirm", "") or fallback["confirm"]),
                "terms_url": str(getattr(text, "terms_url", "") or "")}

    def links_for(self, identity: str) -> list:
        if not identity:
            return []
        fn = self._fn("links_blocking")
        try:
            raw = fn(book=identity) if fn is not None else []
        except Exception:
            log.exception("reading the saved links failed")
            return []
        now = time.time()
        return [d for d in (_link_dict(link, now) for link in raw or ()) if d.get("url")]

    def send_options(self) -> dict:
        fn = self._fn("send_options")
        try:
            return dict(fn() or {}) if fn is not None else {}
        except Exception:
            return {}

    # ---- actions ----

    async def call(self, name: str, *args: Any, **kwargs: Any) -> dict:
        fn = self._fn(name)
        if fn is None:
            return {"ok": False, "message": NO_SERVICE_REASON, "unavailable": True}
        try:
            return _result(await _resolve(fn(*args, **kwargs)))
        except Exception as exc:
            log.info("share links %s failed: %s", name, getattr(exc, "code", type(exc).__name__))
            return {"ok": False, "message": str(getattr(exc, "message", "") or exc) or exc.__class__.__name__,
                    "code": str(getattr(exc, "code", "") or "")}

    async def enable(self, provider_id: str, value: bool, *, consent: bool = False) -> dict:
        return await self.call("set_enabled", provider_id, bool(value), consent=bool(consent))

    async def give_consent(self, provider_id: str) -> dict:
        return await self.call("give_consent", provider_id)

    async def set_key(self, provider_id: str, key: str) -> dict:
        return await self.call(f"set_{provider_id}_key", key)

    async def clear_key(self, provider_id: str) -> dict:
        return await self.call(f"clear_{provider_id}_key")

    async def check_key(self, provider_id: str) -> dict:
        return await self.call(f"check_{provider_id}_key")

    async def preflight(self, provider_id: str, path: str) -> dict:
        """``{ok, message, warnings, existing (link dict | None), size}`` before an upload."""
        fn = self._fn("preflight")
        if fn is None:
            return {"ok": True, "message": "", "warnings": [], "existing": None}
        try:
            pre = await _resolve(fn(provider_id, path))
        except Exception as exc:
            return {"ok": False, "message": str(getattr(exc, "message", "") or exc), "warnings": [], "existing": None}
        existing = getattr(pre, "existing", None)
        return {"ok": bool(getattr(pre, "ok", True)), "message": str(getattr(pre, "message", "") or ""),
                "warnings": [str(w) for w in getattr(pre, "warnings", ()) or ()],
                "existing": _link_dict(existing) if existing is not None else None,
                "size": int(getattr(pre, "size", 0) or 0)}

    async def upload(self, provider_id: str, path: str, *, workspace: str,
                     progress: Optional[Callable[[Any, Any], Any]] = None) -> dict:
        """The new link dict, or ``{ok: False, message, code}`` (``ShareError``: ``cancelled``,
        ``maybe_uploaded`` (timed out after the whole file was sent: ask before uploading again), …)."""
        service = self.service
        fn = self._fn("upload")
        if fn is None:
            return {"ok": False, "message": NO_SERVICE_REASON, "code": "unavailable"}
        unsub = None
        if progress is not None and callable(getattr(service, "subscribe", None)):
            def on_event(kind: Any = None) -> None:
                if kind not in (None, "upload"):
                    return
                state = getattr(service, "state", None)
                if state is not None and getattr(state, "provider", provider_id) == provider_id:
                    progress(getattr(state, "sent", 0), getattr(state, "total", 0))

            try:
                unsub = service.subscribe(on_event)
            except Exception:
                unsub = None
        try:
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                params = {}
            kwargs = {"workspace": workspace} if "workspace" in params else {"book": workspace}
            if "progress" in params and progress is not None:
                kwargs["progress"] = progress
            link = await _resolve(fn(provider_id, path, **kwargs))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"ok": False, "message": str(getattr(exc, "message", "") or exc) or exc.__class__.__name__,
                    "code": str(getattr(exc, "code", "") or "")}
        finally:
            if callable(unsub):
                unsub()
        out = _link_dict(link)
        out.setdefault("ok", bool(out.get("url")))
        if not out.get("url"):
            out.setdefault("message", "The service gave no link")
        return out

    def cancel(self) -> bool:
        fn = self._fn("cancel") or self._fn("cancel_upload")
        try:
            return bool(fn()) if fn is not None else False
        except Exception:
            return False

    async def start_handoff(self, path: str, workspace: str) -> dict:
        """transfer.it: the service makes the file reachable and opens the start page itself."""
        fn = self._fn("start_handoff")
        if fn is None:
            return {"ok": False, "message": NO_SERVICE_REASON}
        try:
            plan = await _resolve(fn(path, book=workspace))
        except Exception as exc:
            return {"ok": False, "message": str(getattr(exc, "message", "") or exc) or exc.__class__.__name__}
        if isinstance(plan, Mapping):
            return dict({"ok": True}, **dict(plan))
        return {"ok": True, "url": str(getattr(plan, "url", "") or ""), "location": str(getattr(plan, "location", "")
                                                                                        or ""),
                "hint": str(getattr(plan, "hint", "") or ""), "show_in_files": bool(getattr(plan, "show_in_files", False)),
                "plan": plan, "opened": True}

    async def reopen_handoff(self, info: Mapping, open_url: Optional[Callable[[str], Any]] = None) -> bool:
        """"Open transfer.it again" (the start page only, never anything else)."""
        handoff = getattr(self.service, "handoff", None)
        plan = info.get("plan")
        if handoff is not None and plan is not None and callable(getattr(handoff, "open", None)):
            try:
                return bool(await _resolve(handoff.open(plan)))
            except Exception:
                log.debug("re-opening the hand-off page failed", exc_info=True)
        url = str(info.get("url") or "")
        if open_url is not None and url:
            await _resolve(open_url(url))
            return True
        return False

    async def save_pasted_link(self, text: str, *, path: str, workspace: str) -> dict:
        fn = self._fn("add_pasted_link")
        if fn is None:
            return {"ok": False, "message": NO_SERVICE_REASON}
        try:
            link = await _resolve(fn(text, path=path, book=workspace))
        except Exception as exc:
            return {"ok": False, "message": str(getattr(exc, "message", "") or exc) or exc.__class__.__name__}
        out = _link_dict(link)
        out.setdefault("ok", True)
        return out

    async def delete_link(self, link_id: str) -> dict:
        """Delete the upload on its service (the record goes with it)."""
        result = await self.call("delete_link", link_id)
        value = result.get("value")
        if value is not None and hasattr(value, "removed"):
            result["ok"] = bool(getattr(value, "removed"))
            result["remote"] = str(getattr(value, "remote", "") or "")
        return result

    async def forget_link(self, link_id: str) -> dict:
        return await self.call("forget_link", link_id)

    def subscribe(self, listener: Callable[..., Any]) -> Callable[[], None]:
        fn = self._fn("subscribe")
        if fn is None:
            return lambda: None
        try:
            unsub = fn(listener)
        except Exception:
            return lambda: None
        return unsub if callable(unsub) else (lambda: None)


# ---------------------------------------------------------------------------
# Shared actions (Book page Output tab + chat Result card)
# ---------------------------------------------------------------------------


def _tone_color(tone: str) -> Any:
    return {
        "ok": ft.Colors.PRIMARY,
        "busy": ft.Colors.ON_SURFACE_VARIANT,
        "warn": ft.Colors.TERTIARY,
        "error": ft.Colors.ERROR,
        "action": ft.Colors.PRIMARY,
        "muted": ft.Colors.ON_SURFACE_VARIANT,
    }.get(tone, ft.Colors.ON_SURFACE_VARIANT)


def status_row(icon: str, text: str, tone: str = "muted", *, key: Optional[str] = None,
               on_click: Optional[Callable[[Any], Any]] = None) -> ft.Control:
    """A status line: icon + text in the tone colour (status is always icon + text + colour, UI_SPEC §5.0); a
    tappable one gets a 48 dp tall target."""
    color = _tone_color(tone)
    row = ft.Row([ft.Icon(icon_data(icon), size=16, color=color),
                  ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=color, expand=True)],
                 spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    if on_click is None:
        return ft.Container(content=row, key=key)
    # Flet 1.0.3 Container has no min-size constraints: the padding makes a one-line row (16 dp) 48 dp tall
    pad = max(4, (tokens.SIZES["hit_target"] - 16) // 2)
    return ft.Container(content=row, on_click=on_click, key=key, padding=ft.Padding.symmetric(vertical=pad))


def reason_detail(reason: Optional[str]) -> Optional[str]:
    return REASON_DETAILS.get(str(reason or "")) if reason else None


class U10Actions:
    """Send now, the per-book override sheet, Share file via link and the saved-link rows.

    ``say(message, action_label=None, on_action=None)``, ``spawn(coro)``, ``io(fn, *args)`` (blocking work off
    the loop), ``go(route_name, params=None)``, ``copy_text(text)``, ``share_text(text)``, ``show_in_files(path)``
    (async or sync), ``open_url(url)`` (the in-app browser) and ``read_clipboard()`` come from the caller's
    context. ``identity`` is the book's identity path (``services.library.book_identity``: its workspace)."""

    def __init__(self, *, cloud: Any = None, shares: Any = None, page: Any = None,
                 say: Optional[Callable[..., Any]] = None, spawn: Optional[Callable[[Any], Any]] = None,
                 io: Optional[Callable[..., Any]] = None, go: Optional[Callable[..., Any]] = None,
                 copy_text: Optional[Callable[[str], Any]] = None, share_text: Optional[Callable[[str], Any]] = None,
                 show_in_files: Optional[Callable[[str], Any]] = None, open_url: Optional[Callable[[str], Any]] = None,
                 read_clipboard: Optional[Callable[[], Any]] = None, tablet: bool = False,
                 platform: str = "desktop") -> None:
        self.cloud = cloud if isinstance(cloud, CloudFacade) else CloudFacade(cloud)
        self.shares = shares if isinstance(shares, ShareFacade) else ShareFacade(shares)
        self.page = page
        self._say = say
        self._spawn = spawn
        self._io = io
        self.go = go
        self.copy_text = copy_text
        self.share_text = share_text
        self.show_in_files = show_in_files
        self.open_url = open_url
        self.read_clipboard = read_clipboard
        self.tablet = tablet
        self.platform = platform
        self.last_sheet: Any = None  # the newest ActionSheet / sheet parts (tests, back handling)
        self.last_dialog: Any = None
        self.uploading: Optional[str] = None  # provider id of the running upload
        self.sheet_builds = 0  # per-build dialog keys (``_sheet``)

    # ---- context helpers ----

    @property
    def native(self) -> bool:
        return self.platform in ("android", "ios")

    def say(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> None:
        if self._say is None:
            log.info("cloud: %s", message)
            return
        try:
            self._say(message, action_label, on_action)
        except TypeError:
            self._say(message)

    def spawn(self, coro: Any) -> Any:
        if self._spawn is not None:
            return self._spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._io is not None:
            return await self._io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def open_settings(self) -> Any:
        if self.go is None:
            return None
        try:
            return self.go(ROUTE_NAME)
        except Exception:
            log.debug("opening %s failed", ROUTE_NAME, exc_info=True)
            return None

    def show(self, dialog: Any) -> Any:
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def _show_sheet(self, sheet: Any) -> Any:
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def _open_dialog(self, dialog: Any) -> None:
        if self.page is not None:
            self.page.show_dialog(dialog)

    def _sheet(self, content: ft.Control, key: str) -> Any:
        """A bottom sheet whose dialog key is new on every build: a sheet shown again before the client's
        dismiss of the previous one arrived (Cancel, then the same switch again) would otherwise be diffed
        under the old one's key and freeze (Flet 1.0.3). ``key`` itself stays on its frame for finders."""
        self.sheet_builds += 1
        return bottom_sheet(sheet_frame(content, key=key), key=f"{key}-{self.sheet_builds}")

    def _close(self, dialog: Any) -> None:
        close_dialog(self.page, dialog)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass

    # ---- reasons ----

    def cloud_reason(self, state: Mapping, *, has_outputs: bool = True, in_library: bool = True) -> Optional[str]:
        """Why "Send to cloud now" cannot run (None: it can)."""
        if not self.cloud.available or not state.get("available", True):
            return NO_SERVICE_REASON
        if not state.get("supported", self.native):
            return PHONE_ONLY
        dest = state.get("destination")
        if not dest:
            return NO_DESTINATION_REASON
        if dest.get("needs_relink"):
            return RELINK_REASON
        if not in_library:
            return NOT_IN_LIBRARY_REASON
        if not has_outputs:
            return NO_OUTPUT_REASON
        return None

    def share_reason(self, providers: Sequence[Mapping], *, has_outputs: bool = True) -> Optional[str]:
        """Why "Share file via link" cannot start (None: it can)."""
        if not self.shares.available:
            return NO_SERVICE_REASON
        if not has_outputs:
            return NO_OUTPUT_REASON
        if not any(p.get("enabled") for p in providers):
            return SHARE_OFF_REASON
        return None

    def explain(self, reason: str) -> None:
        """A disabled action was asked for anyway: its full reason, with "Settings" when that is where it is fixed."""
        detail = REASON_DETAILS.get(reason, reason)
        if reason in SETTINGS_REASONS:
            self.say(detail, "Settings", self.open_settings)
        else:
            self.say(detail)

    # ---- cloud actions ----

    async def send_now(self, identity: str, state: Optional[Mapping] = None) -> dict:
        """"Send to cloud now" for one book. Says what happened; a missing destination offers Settings."""
        if state is None:
            state = await self.io(self.cloud.snapshot)
        reason = self.cloud_reason(state)
        if reason == NO_DESTINATION_REASON:  # owner: pick it right here, then send (no trip to Settings)
            if not await self.ask_destination():
                return {"ok": False, "message": reason, "cancelled": True}
            state = await self.io(self.cloud.snapshot)
            reason = self.cloud_reason(state)
        if reason:
            self.explain(reason)
            return {"ok": False, "message": reason}
        result = await self.cloud.call("send_now", identity)
        dest = destination_text(state.get("destination"))
        if result.get("ok"):
            self.say(str(result.get("message") or f"Copying to {dest}…"))
        else:
            self.say(str(result.get("message") or f"Could not copy to {dest}"))
        return result

    async def ask_destination(self) -> bool:
        """"Send to cloud" with no destination yet: the same three choices as Settings › Cloud sync & sharing,
        as a sheet; True once one was set."""
        loop = asyncio.get_running_loop()
        picked: asyncio.Future = loop.create_future()
        items = [ActionItem(label, (lambda o=op: picked.done() or picked.set_result(o)), icon=icon,
                            key=f"cloud-dest-{mode}")
                 for mode, op, label, _detail, icon in DESTINATIONS
                 if self.native or mode == "folder"]
        sheet = ActionSheet(items, title="Where should books go?",
                            subtitle="Pick once; Settings › Cloud sync & sharing can change it later",
                            tablet=self.tablet,
                            on_cancel=lambda: picked.done() or picked.set_result(None))
        self._show_sheet(sheet)
        op = await picked
        if not op:
            return False
        result = await self.cloud.call(op)
        if result.get("cancelled"):
            return False
        if not result.get("ok"):
            self.say(str(result.get("message") or "The destination was not changed"))
            return False
        self.say(f"Destination: {destination_text(result.get('destination'))}")
        return True

    def override_sheet(self, identity: str, current: str = "default",
                       on_done: Optional[Callable[[str], Any]] = None) -> ActionSheet:
        """"Copy this book to the cloud": Default / Always / Never (✓ on the current one)."""

        async def choose(value: str) -> None:
            result = await self.cloud.call("set_override", identity, value)
            if not result.get("ok"):
                self.say(str(result.get("message") or "Could not change the setting"))
                return
            if on_done is not None:
                outcome = on_done(value)
                if inspect.isawaitable(outcome):
                    await outcome

        items = [ActionItem(f"{label} · {text}", (lambda v=value: self.spawn(choose(v))),
                            icon="CHECK" if value == current else "RADIO_BUTTON_UNCHECKED",
                            key=f"cloud-override-{value}")
                 for value, label, text in OVERRIDES]
        sheet = ActionSheet(items, title="Copy this book to the cloud",
                            subtitle=f"Now: {_OVERRIDE_LABELS.get(current, 'Default')}", tablet=self.tablet)
        return self._show_sheet(sheet)

    async def choose_save_location(self, identity: str, kind: str) -> dict:
        result = await self.cloud.call("choose_save_location", identity, kind)
        if result.get("message") and not result.get("cancelled"):
            self.say(str(result.get("message")))
        return result

    # ---- share links ----

    @staticmethod
    def shareable(outputs: Sequence[tuple]) -> list:
        """The compiled outputs a link can be made for (EPUB first, then PDF, TXT, HTML)."""
        order = {kind: index for index, kind in enumerate(compiled_kinds())}
        rows = [(str(path), str(kind)) for path, kind in outputs or () if str(kind) in order and path]
        return sorted(rows, key=lambda row: order[row[1]])

    async def share_link(self, outputs: Sequence[tuple], identity: str) -> Any:
        """"Share file via link": one file → its provider sheet; several → choose the file first."""
        rows = self.shareable(outputs)
        if not rows:
            self.say(NO_OUTPUT_DETAIL)
            return None
        if len(rows) == 1:
            return await self.provider_sheet(rows[0][0], identity)
        items = [ActionItem(f"{_KIND_LABELS.get(kind, kind.upper())} · {os.path.basename(path)}",
                            (lambda p=path: self.spawn(self.provider_sheet(p, identity))),
                            icon="INSERT_DRIVE_FILE", key=f"share-file-{index}")
                 for index, (path, kind) in enumerate(rows)]
        return self._show_sheet(ActionSheet(items, title=SHARE_LABEL, subtitle="Which file?", tablet=self.tablet))

    async def provider_sheet(self, path: str, identity: str) -> Optional[ActionSheet]:
        """The services the user can send ``path`` to; off ones stay listed with their reason (the service's
        ``menu``: switches, consent, the key, eligibility, a running upload)."""
        try:
            size = int(await self.io(os.path.getsize, path))
        except (OSError, TypeError, ValueError):
            self.say("The file is no longer on the phone")
            return None
        menu = await self.io(self.shares.menu, path)
        items: list = []
        for provider, reason in menu:
            limit = int(provider.get("max_bytes") or 0)
            if reason is None and limit and size > limit:
                reason = f"{TOO_LARGE_REASON} (max {human_size(limit)})"
            if reason is None and self.uploading and not provider.get("handoff"):
                reason = BUSY_UPLOAD_REASON
            if provider.get("handoff"):
                label, icon = f"Open {provider.get('label')}…", "OPEN_IN_BROWSER"
            else:
                label, icon = str(provider.get("label") or provider.get("id")), (
                    "LOCK_OUTLINE" if provider.get("e2ee") else "LINK")
            items.append(ActionItem(label, (lambda p=provider: self.spawn(self.start_provider(p, path, identity))),
                                    icon=icon, disabled_reason=reason, key=f"share-provider-{provider['id']}"))
        if not items:
            items.append(ActionItem("No sharing services", None, icon="LINK_OFF", disabled_reason=NO_SERVICE_REASON,
                                    key="share-provider-none"))
        items.append(ActionItem("Sharing settings…", self.open_settings, icon="SETTINGS_OUTLINED",
                                key="share-settings"))
        sheet = ActionSheet(items, title=SHARE_LABEL, subtitle=f"{os.path.basename(path)} · {human_size(size)}",
                            tablet=self.tablet)
        return self._show_sheet(sheet)

    async def start_provider(self, provider: Mapping, path: str, identity: str) -> Any:
        """Consent first when the current consent text was never accepted, then the upload or the hand-off."""
        if not provider.get("consented"):
            accepted = await self.ask_consent(provider, path)
            if not accepted:
                return None
            result = await self.shares.give_consent(str(provider.get("id")))
            if not result.get("ok"):
                self.say(str(result.get("message") or "Could not save your answer"))
                return None
        if provider.get("handoff"):
            return await self.handoff(provider, path, identity)
        return await self.upload(provider, path, identity)

    def consent_sheet(self, provider: Mapping, path: Optional[str], on_answer: Callable[[bool], Any],
                      *, confirm_label: Optional[str] = None) -> dict:
        """The consent sheet: the service's text (the file leaves the phone; who can read it; what the service
        sees; how long it lasts) and its required checkbox. Returns its parts (tests tap them)."""
        answered = {"done": False}
        size = None
        if path:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
        text = self.shares.consent(provider, path, size)
        rights = ft.Checkbox(label=text["checkbox"], value=False, key="consent-rights")
        confirm = ft.FilledButton(content=confirm_label or text["confirm"], disabled=True, key="consent-confirm")
        cancel = ft.TextButton(content="Cancel", key="consent-cancel")
        holder: dict = {}

        def finish(value: bool) -> None:
            if answered["done"]:
                return
            answered["done"] = True
            self._close(holder.get("dialog"))
            result = on_answer(value)
            if inspect.isawaitable(result):
                self.spawn(result)

        def set_rights(value: bool) -> None:
            rights.value = bool(value)
            confirm.disabled = not rights.value
            self._push(confirm)

        rights.on_change = lambda e: set_rights(bool(getattr(getattr(e, "control", None), "value", rights.value)))
        confirm.on_click = lambda e: finish(True) if rights.value else None
        cancel.on_click = lambda e: finish(False)
        controls: list = [ft.Text(text["title"], theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600)]
        controls.extend(ft.Text(line, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True,
                                key=f"consent-line-{index}") for index, line in enumerate(text["lines"]))
        if text.get("terms_url"):
            controls.append(ft.Text(f"Terms: {text['terms_url']}", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT, selectable=True))
        controls.append(rights)
        dialog = self._sheet(scroll_column(controls, footer=[
            ft.Row([cancel, confirm], alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8)]), "share-consent")
        dialog.on_dismiss = lambda e: finish(False)
        holder["dialog"] = dialog
        parts = {"dialog": dialog, "confirm": confirm, "cancel": cancel, "rights": rights, "finish": finish,
                 "set_rights": set_rights, "text": text}
        self.last_sheet = parts
        self._open_dialog(dialog)
        return parts

    async def ask_consent(self, provider: Mapping, path: Optional[str], *, confirm_label: Optional[str] = None) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def answer(value: bool) -> None:
            if not future.done():
                future.set_result(bool(value))

        self.consent_sheet(provider, path, answer, confirm_label=confirm_label)
        if self.page is None:  # headless: nothing can answer a sheet nobody sees
            return False
        return await future

    async def upload(self, provider: Mapping, path: str, identity: str) -> dict:
        """The service's pre-flight first: blocked (its reason), a live link to the same file (show it, or
        upload again), warnings (a large file on mobile data: confirm); then ``run_upload``."""
        pid = str(provider.get("id"))
        pre = await self.shares.preflight(pid, path)
        if not pre.get("ok"):
            self.say(str(pre.get("message") or "This file cannot be shared here"))
            return {"ok": False, "message": pre.get("message")}
        existing = pre.get("existing")
        if existing:
            self.link_sheet(existing, again=lambda: self.spawn(self._confirm_then_upload(provider, path, identity,
                                                                                          pre.get("warnings") or [])),
                            title="You already have a link for this file")
            return {"ok": True, "existing": existing}
        return await self._confirm_then_upload(provider, path, identity, pre.get("warnings") or [])

    async def _confirm_then_upload(self, provider: Mapping, path: str, identity: str, warnings: Sequence[str]) -> dict:
        if warnings:
            dialog = ConfirmDialog(title=f"Upload to {provider.get('label')}?", body="\n\n".join(warnings),
                                   confirm_label="Upload",
                                   on_confirm=lambda: self.spawn(self.run_upload(provider, path, identity)))
            self.show(dialog)
            return {"ok": True, "confirm": dialog}
        return await self.run_upload(provider, path, identity)

    async def run_upload(self, provider: Mapping, path: str, identity: str) -> dict:
        """Upload with a progress sheet (Cancel; Hide keeps it running), then the link sheet."""
        if self.uploading:
            self.say("An upload is already running")
            return {"ok": False, "message": BUSY_UPLOAD_REASON}
        pid = str(provider.get("id"))
        label = str(provider.get("label") or pid)
        name = os.path.basename(path)
        self.uploading = pid
        loop = asyncio.get_running_loop()
        bar = ft.ProgressBar(value=None, key="share-progress-bar")
        line = ft.Text(f"Preparing {name}…", theme_style=ft.TextThemeStyle.BODY_SMALL, key="share-progress-text")
        state = {"open": True, "last": 0.0}
        holder: dict = {}

        def hide(e: Any = None) -> None:
            state["open"] = False
            self._close(holder.get("dialog"))

        def cancel(e: Any = None) -> None:
            if self.shares.cancel():
                line.value = "Cancelling…"
                self._push(line)

        cancel_button = ft.TextButton(content="Cancel upload", on_click=cancel, key="share-cancel")

        def apply(sent: Any, total: Any) -> None:
            pct = _percent(sent, total)
            bar.value = None if pct is None else pct / 100.0
            if total:
                try:
                    line.value = f"{human_size(int(sent or 0))} of {human_size(int(total))}"
                except (TypeError, ValueError):
                    pass
            if total and sent and int(sent) >= int(total) and cancel_button.content != STOP_WAITING_LABEL:
                # the whole file was sent: Cancel can no longer take it back from the service
                cancel_button.content = STOP_WAITING_LABEL
                cancel_button.tooltip = "The file may already be on the service; its link may not come back"
                self._push(cancel_button)
            self._push(bar, line)

        def progress(sent: Any, total: Any) -> None:
            now = time.monotonic()
            if now - state["last"] < 0.25 and not (total and sent and sent >= total):
                return
            state["last"] = now
            try:
                loop.call_soon_threadsafe(apply, sent, total)
            except RuntimeError:  # the loop is gone (app closing)
                pass

        dialog = self._sheet(scroll_column([
            ft.Text(f"Uploading to {label}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            ft.Text(name, theme_style=ft.TextThemeStyle.LABEL_MEDIUM, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            bar, line,
            ft.Text("You can leave this screen: the upload continues, and its link is saved on the book.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ], footer=[ft.Row([
            cancel_button,
            ft.FilledTonalButton(content="Hide", on_click=hide, key="share-hide"),
        ], alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8)]), "share-progress")
        dialog.on_dismiss = lambda e: state.update(open=False)
        holder["dialog"] = dialog
        self.last_sheet = {"dialog": dialog, "bar": bar, "line": line, "hide": hide, "cancel": cancel,
                           "cancel_button": cancel_button}
        self._open_dialog(dialog)
        try:
            result = await self.shares.upload(pid, path, workspace=identity, progress=progress)
        finally:
            self.uploading = None
        still_open = bool(state["open"])
        if still_open:
            hide()
        if result.get("ok") and result.get("url"):
            if still_open:
                self.link_sheet(result)
            else:  # the user hid the progress sheet: a snackbar, not a sheet over whatever they do now
                self.say(f"Link ready: {label}", "Copy", lambda: self.spawn(self.copy_link(result)))
        elif result.get("code") == "cancelled":
            self.say("Upload cancelled")
        else:
            self.say(f"{label}: {result.get('message') or 'Upload failed'}")
        return result

    def link_sheet(self, link: Mapping, *, again: Optional[Callable[[], Any]] = None,
                   title: str = "Link ready") -> Any:
        """A link: selectable URL, Copy · Share · Done (and Upload again for an existing one). The link is
        already saved on the book."""
        holder: dict = {}

        def done(e: Any = None) -> None:
            self._close(holder.get("dialog"))

        buttons: list = [
            ft.TextButton(content="Copy", icon=ft.Icons.CONTENT_COPY, on_click=lambda e: self.spawn(self.copy_link(link)),
                          key="link-copy"),
            ft.TextButton(content="Share", icon=ft.Icons.SHARE, on_click=lambda e: self.spawn(self.share_url(link)),
                          key="link-share"),
        ]
        if again is not None:
            def upload_again(e: Any = None) -> None:
                done()
                again()

            buttons.append(ft.TextButton(content="Upload again", icon=ft.Icons.REPLAY, on_click=upload_again,
                                         key="link-again"))
        buttons.append(ft.FilledTonalButton(content="Done", on_click=done, key="link-done"))
        dialog = self._sheet(scroll_column([
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            ft.Text(str(link.get("url") or ""), selectable=True, theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                    key="link-url"),
            ft.Text(link_meta_text(link), theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text("Saved on the book: its Output tab lists it.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
        ], footer=[ft.Row(buttons, alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8)]), "share-link-ready")
        holder["dialog"] = dialog
        self.last_sheet = {"dialog": dialog, "link": dict(link), "done": done}
        self._open_dialog(dialog)
        return dialog

    async def handoff(self, provider: Mapping, path: str, identity: str) -> dict:
        """transfer.it: the service makes the file reachable (Android: Downloads/Glossarion; iOS: its Files
        path) and opens the start page; the user uploads there and pastes the link back. Glossarion never
        talks to transfer.it."""
        label = str(provider.get("label") or provider.get("id"))
        info = await self.shares.start_handoff(path, identity)
        if not info.get("ok"):
            self.say(str(info.get("message") or f"Could not prepare the file for {label}"))
            return info
        location = str(info.get("location") or "")
        hint = str(info.get("hint") or "").strip() or (
            f"Tap Add files and choose {os.path.basename(path)}. Keep the page open until {label} shows the link, "
            "then Copy link and come back here.")
        field = ft.TextField(label=f"Paste the {label} link", dense=True, key="handoff-link")
        error = ft.Text("", color=ft.Colors.ERROR, visible=False, key="handoff-error")
        holder: dict = {}

        async def open_page() -> bool:
            return await self.shares.reopen_handoff(info, self.open_url)

        async def paste() -> None:
            if self.read_clipboard is None:
                return
            try:
                text = await _resolve(self.read_clipboard())
            except Exception:
                text = None
            if text:
                field.value = str(text).strip()
                self._push(field)

        async def save() -> dict:
            value = str(field.value or "").strip()
            if not value:
                error.value = "Paste the link first"
                error.visible = True
                self._push(error)
                return {"ok": False}
            result = await self.shares.save_pasted_link(value, path=path, workspace=identity)
            if not result.get("ok"):
                error.value = str(result.get("message") or f"That is not a {label} link")
                error.visible = True
                self._push(error)
                return result
            self._close(holder.get("dialog"))
            self.say(f"{label} link saved on the book")
            return result

        steps: list = [
            ft.Text(f"Share with {label}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            ft.Text(f"{label} has no app integration: you upload the file on its own page, then paste its link here.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ]
        if location:
            steps.append(ft.Text(f"The file: {location}", theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True,
                                 key="handoff-location"))
        steps.append(ft.Text(hint, theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="handoff-hint"))
        buttons: list = [ft.FilledTonalButton(content=f"Open {label} again", icon=ft.Icons.OPEN_IN_BROWSER,
                                              on_click=lambda e: self.spawn(open_page()), key="handoff-open")]
        if info.get("show_in_files") and self.show_in_files is not None:
            buttons.append(ft.TextButton(content="Show in Files", icon=ft.Icons.FOLDER_OPEN,
                                         on_click=lambda e: self.spawn(_resolve(self.show_in_files(path))),
                                         key="handoff-files"))
        steps.append(ft.Row(buttons, wrap=True, spacing=8))
        steps.append(field)
        if self.read_clipboard is not None:
            steps.append(ft.TextButton(content="Paste", icon=ft.Icons.CONTENT_PASTE,
                                       on_click=lambda e: self.spawn(paste()), key="handoff-paste"))
        steps.append(error)
        dialog = self._sheet(scroll_column(steps, footer=[ft.Row([
            ft.TextButton(content="Close", on_click=lambda e: self._close(holder.get("dialog")), key="handoff-close"),
            ft.FilledButton(content="Save link", on_click=lambda e: self.spawn(save()), key="handoff-save"),
        ], alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8)]), "share-handoff")
        holder["dialog"] = dialog
        self.last_sheet = {"dialog": dialog, "field": field, "error": error, "save": save, "open": open_page,
                           "paste": paste, "info": info}
        self._open_dialog(dialog)
        return info

    # ---- saved links ----

    def link_rows(self, links: Sequence[Mapping], *, on_changed: Optional[Callable[[], Any]] = None,
                  key_prefix: str = "link", now: Optional[float] = None) -> list:
        """One block per saved link: file name, the URL (selectable), service · size · when · expiry, and
        Copy · Share · Delete (Remove where the service cannot delete from the app)."""
        rows: list = []
        for index, link in enumerate(links or ()):
            can_delete = bool(link.get("can_delete"))
            name = str(link.get("name") or "")
            buttons = [
                ft.TextButton(content="Copy", icon=ft.Icons.CONTENT_COPY,
                              on_click=lambda e, l=link: self.spawn(self.copy_link(l)), key=f"{key_prefix}-copy-{index}"),
                ft.TextButton(content="Share", icon=ft.Icons.SHARE,
                              on_click=lambda e, l=link: self.spawn(self.share_url(l)), key=f"{key_prefix}-share-{index}"),
                ft.TextButton(content="Delete" if can_delete else "Remove", icon=ft.Icons.DELETE_OUTLINE,
                              on_click=lambda e, l=link: self.confirm_delete(l, on_changed),
                              key=f"{key_prefix}-delete-{index}",
                              tooltip="Delete the upload" if can_delete else "Remove from this list"),
            ]
            icon = "LOCK_OUTLINE" if link.get("e2ee") else "LINK"
            rows.append(ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(icon_data(icon), size=18, color=ft.Colors.PRIMARY),
                            ft.Text(name or str(link.get("provider_label") or "Link"),
                                    theme_style=ft.TextThemeStyle.LABEL_LARGE, expand=True, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS)], spacing=6),
                    ft.Text(str(link.get("url") or ""), selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL,
                            max_lines=3, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(link_meta_text(link, now), theme_style=ft.TextThemeStyle.BODY_SMALL,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Row(buttons, wrap=True, spacing=0, run_spacing=0),
                ], spacing=2, tight=True),
                padding=ft.Padding.symmetric(horizontal=4, vertical=6),
                key=f"{key_prefix}-{index}",
            ))
        return rows

    async def copy_link(self, link: Mapping) -> bool:
        url = str(link.get("url") or "")
        if not url:
            return False
        if self.copy_text is None:
            self.say("Copying is not available in this session")
            return False
        await _resolve(self.copy_text(url))
        return True

    async def share_url(self, link: Mapping) -> bool:
        url = str(link.get("url") or "")
        if not url:
            return False
        if self.share_text is None:
            return await self.copy_link(link)
        try:
            return bool(await _resolve(self.share_text(url)))
        except Exception as exc:
            log.info("sharing a link failed: %s", type(exc).__name__)
            self.say("The share sheet could not open")
            return False

    def confirm_delete(self, link: Mapping, on_changed: Optional[Callable[[], Any]] = None) -> ConfirmDialog:
        can_delete = bool(link.get("can_delete"))
        label = str(link.get("provider_label") or link.get("provider") or "the service")
        name = str(link.get("name") or "this file")
        if can_delete:
            title, confirm = "Delete the upload?", "Delete"
            body = (f"Deletes {name} from {label}. People who have the link can no longer download it. "
                    "This cannot be undone.")
        else:
            title, confirm = "Remove the link?", "Remove"
            body = (f"{label} links can't be deleted from Glossarion. This removes the link from the book; the file "
                    f"stays on {label} until it expires (delete it there if you want it gone sooner).")

        async def run() -> None:
            if can_delete:
                result = await self.shares.delete_link(str(link.get("id")))
            else:
                result = await self.shares.forget_link(str(link.get("id")))
            if not result.get("ok"):
                self.say(str(result.get("message") or ("Could not delete the upload" if can_delete else
                                                       "Could not remove the link")))
                return
            if can_delete and result.get("remote") == "gone":
                self.say("The upload was already gone; the link is removed")
            else:
                self.say("Upload deleted" if can_delete else "Link removed")
            if on_changed is not None:
                outcome = on_changed()
                if inspect.isawaitable(outcome):
                    await outcome

        return self.show(ConfirmDialog(title=title, body=body, confirm_label=confirm, destructive=True, on_confirm=run))


# ---------------------------------------------------------------------------
# Settings › Cloud sync & sharing
# ---------------------------------------------------------------------------

#: (mode, operation, title, subtitle, icon) of the three destinations.
DESTINATIONS = (
    ("folder", "pick_folder", "Choose a folder…",
     "Google Drive, Nextcloud, iCloud Drive, On My iPhone or any app the picker lists. Each book gets its own "
     "folder there; recompiled books replace their copy.", "CREATE_NEW_FOLDER"),
    ("files", "use_save_locations", "Save each file separately",
     "For apps that can't be picked as a folder: you choose where each new file goes once; later compiles replace "
     "it.", "NOTE_ADD"),
    ("phone", "use_phone_folder", "Phone folder (Downloads/Glossarion)",
     "Copies replace themselves in Downloads/Glossarion; a backup app such as TeraBox, FolderSync or Syncthing can "
     "upload that folder.", "PHONE_ANDROID"),
)
#: Send (send.vis.ee) link options: (seconds, label) and download counts (the service clamps them too).
SEND_EXPIRY_CHOICES = ((300, "5 minutes"), (3600, "1 hour"), (86400, "1 day"), (259200, "3 days"))
SEND_DOWNLOAD_CHOICES = (1, 2, 3, 5, 10, 20)


class CloudSyncScreen(PageScreen):
    title = "Cloud sync & sharing"

    def __init__(self, match: Any, ctx: Any, *, cloud: Any = None, shares: Any = None, platform: str = "desktop",
                 open_url: Optional[Callable[[str], Any]] = None, share_text: Optional[Callable[[str], Any]] = None,
                 show_in_files: Optional[Callable[[str], Any]] = None,
                 now: Optional[Callable[[], float]] = None) -> None:
        super().__init__(match, ctx)
        self.platform = platform
        self.facade = CloudFacade(cloud)
        self.share_facade = ShareFacade(shares)
        self.actions_ = U10Actions(
            cloud=self.facade, shares=self.share_facade, page=self.page, say=self.say, spawn=self.spawn, io=self.io,
            go=getattr(ctx, "go", None), copy_text=getattr(ctx, "copy_text", None), share_text=share_text,
            show_in_files=show_in_files, open_url=open_url, read_clipboard=getattr(ctx, "read_clipboard", None),
            tablet=bool(getattr(ctx, "tablet", False)), platform=platform)
        self.clock = now or time.time
        # until the first refresh (the service getters run on the io pool, never here)
        self.state: dict = {"available": self.facade.available, "supported": platform in ("android", "ios"),
                            "reason": None if self.facade.available else NO_SERVICE_REASON, "enabled": False,
                            "kinds": {kind: True for kind in compiled_kinds()}, "destination": None, "queue": [],
                            "recent": [], "progress": None, "phone_folder": platform == "android"}
        self.providers: list = []
        self.send_options: dict = {}
        self.loaded = False
        self.renders = 0
        self._unsubs: list = []
        self.key_fields: dict = {}
        self.provider_switches: dict = {}
        self._signatures: dict = {}
        self._refreshing = False
        self._refresh_again = False

    @property
    def native(self) -> bool:
        return self.platform in ("android", "ios")

    # ---- build ----

    def build_body(self) -> ft.Control:
        self.banner = ft.Container(visible=False, key="cloud-relink",
                                   bgcolor=ft.Colors.ERROR_CONTAINER, border_radius=tokens.RADII["card"],
                                   padding=tokens.SPACING["card_padding"])
        self.destination_text = ft.Text("…", theme_style=ft.TextThemeStyle.BODY_LARGE, key="cloud-destination")
        self.dest_actions = ft.Row([], wrap=True, spacing=8, key="cloud-dest-actions")
        self.destinations_column = ft.Column(spacing=0, key="cloud-destinations")
        self.auto_switch = ft.Switch(label="Copy finished books automatically", value=False,
                                     on_change=self._on_auto, key="cloud-auto")
        self.auto_reason = ft.Row([], wrap=True, key="cloud-auto-reason")
        self.kind_chips = {
            kind: ft.Chip(label=ft.Text(label), selected=True, show_checkmark=True,
                          on_select=lambda e, k=kind: self._on_kind(e, k), key=f"cloud-kind-{kind}")
            for kind, label in FORMATS
        }
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="cloud-summary")
        self.progress_bar = ft.ProgressBar(value=None, visible=False, key="cloud-progress")
        self.progress_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False,
                                     key="cloud-progress-text")
        self.retry_button = ft.FilledTonalButton(content="Retry now", icon=ft.Icons.REFRESH,
                                                 on_click=lambda e: self.spawn(self.retry_now()), key="cloud-retry")
        self.activity_column = ft.Column(spacing=0, key="cloud-activity")
        self.providers_column = ft.Column(spacing=4, key="share-providers")
        platform_help = NOT_LISTED_HELP_IOS if self.platform == "ios" else NOT_LISTED_HELP_ANDROID
        cloud_rows: list = [
            ft.Text("Copies of your Library's finished books go to a place you pick once. Your cloud app uploads "
                    "them; nothing goes through Glossarion's developer.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
            ft.TextButton(content=EXPLAINER_TITLE, icon=ft.Icons.INFO_OUTLINE,
                          on_click=lambda e: self.show_explainer(), key="cloud-explainer"),
            self.banner,
            ft.Row([ft.Text("Destination", theme_style=ft.TextThemeStyle.LABEL_LARGE), self.destination_text],
                   wrap=True, spacing=8),
            self.dest_actions,
            self.destinations_column,
            self.auto_switch,
            self.auto_reason,
            ft.Text("Formats", theme_style=ft.TextThemeStyle.LABEL_LARGE),
            ft.Row(list(self.kind_chips.values()), wrap=True, spacing=8, run_spacing=4, key="cloud-kinds"),
            ft.Text("Each book can also be set to Always or Never on its Output tab.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.ExpansionTile(title=ft.Text("My cloud app isn't listed"), leading=ft.Icon(ft.Icons.HELP_OUTLINE),
                             controls=[ft.Container(content=ft.Markdown(platform_help, selectable=True),
                                                    padding=ft.Padding.only(left=8, right=8, bottom=8))],
                             dense=True, key="cloud-not-listed"),
        ]
        activity_rows: list = [self.summary, self.progress_bar, self.progress_text, self.activity_column,
                               self.retry_button]
        share_rows: list = [
            ft.Text("Make a link for one file to send to someone. Each service stays off until you turn it on, and "
                    "nothing is uploaded until you tap Share file via link on a book.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.TextButton(content="About share links", icon=ft.Icons.INFO_OUTLINE,
                          on_click=lambda e: self.show_share_explainer(), key="share-explainer"),
            self.providers_column,
        ]
        body = self.scaffold([
            section("Copy books to your cloud", cloud_rows, key="cloud-card"),
            section("Activity", activity_rows, key="cloud-activity-card"),
            section(SHARE_LABEL, share_rows, key="share-card"),
        ])
        self._render()
        return body

    def did_show(self) -> None:
        if not self._unsubs:
            self._unsubs.append(self.facade.subscribe(self._on_change))
            self._unsubs.append(self.share_facade.subscribe(self._on_change))
        self.spawn(self.refresh())

    def app_resumed(self) -> None:
        """Back from the system picker or the cloud app: the destination may have changed."""
        self.spawn(self.refresh())

    def dispose(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    def _on_change(self, *args: Any) -> None:
        """A service change (any thread; progress events arrive several times a second): one refresh at a
        time on the loop, another one after it when more changes came in meanwhile."""
        dispatcher = getattr(self.ctx, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(self._schedule_refresh)
            return
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        if self._refreshing:
            self._refresh_again = True
            return
        try:
            self.spawn(self.refresh())
        except Exception:
            log.debug("cloud settings refresh failed", exc_info=True)

    def _read(self) -> tuple:
        """Blocking (io): the cloud state, the share providers and the Send options."""
        return self.facade.snapshot(), self.share_facade.providers(), self.share_facade.send_options()

    async def refresh(self) -> dict:
        if self._refreshing:
            self._refresh_again = True
            return self.state
        self._refreshing = True
        try:
            while True:
                self._refresh_again = False
                self.state, self.providers, self.send_options = await self.io(self._read)
                self.loaded = True
                self._render()
                if not self._refresh_again:
                    break
                await asyncio.sleep(0.25)  # coalesce a burst of progress events
        finally:
            self._refreshing = False
        return self.state

    # ---- render ----

    @property
    def destination(self) -> Optional[dict]:
        dest = self.state.get("destination")
        return dict(dest) if isinstance(dest, Mapping) else None

    def availability_reason(self) -> Optional[str]:
        if not self.facade.available:
            return NO_SERVICE_REASON
        if not self.native or not self.state.get("supported", True):
            return PHONE_ONLY
        return None

    def _changed(self, part: str, signature: Any) -> bool:
        """True when ``part`` must be rebuilt (its inputs changed since the last build). Rebuilt rows get fresh
        keys; unchanged ones keep their controls (a key being typed survives progress refreshes)."""
        if self._signatures.get(part) == signature:
            return False
        self._signatures[part] = signature
        return True

    def _render(self) -> None:
        if getattr(self, "destinations_column", None) is None:
            return
        self.renders += 1
        n = self.renders
        state = self.state
        dest = self.destination
        unavailable = self.availability_reason()
        dest_label = destination_text(dest)
        self.destination_text.value = dest_label
        relink = dest.get("needs_relink") if dest else None
        if self._changed("destination", (unavailable, repr(sorted((dest or {}).items())),
                                         bool(state.get("phone_folder", True)))):
            self.banner.visible = bool(relink)
            self.banner.content = self._relink_banner(dest, dest_label, n) if relink else None
            self.destinations_column.controls = self._destination_tiles(unavailable, dest, n)
            self.dest_actions.controls = self._dest_action_buttons(unavailable, dest, n)
        self.auto_switch.value = bool(state.get("enabled"))
        auto_reason = unavailable or (NO_DESTINATION_REASON if not dest else None)
        self.auto_switch.disabled = bool(auto_reason)
        if self._changed("auto_reason", auto_reason):
            self.auto_reason.controls = ([ReasonChip(reason=auto_reason, detail=reason_detail(auto_reason),
                                                     key=f"cloud-auto-chip-{n}")] if auto_reason else [])
        kinds = dict(state.get("kinds") or {})
        for kind, chip in self.kind_chips.items():
            chip.selected = bool(kinds.get(kind, True))
            chip.disabled = bool(unavailable)
        self._render_activity(n)
        provider_sig = (self.share_facade.available, tuple(
            (p["id"], str(p.get("label")), bool(p.get("needs_key")), bool(p.get("has_key")), bool(p.get("handoff")),
             bool(p.get("e2ee")), str(p.get("operator")), str(p.get("retention"))) for p in self.providers),
            tuple(sorted(self.send_options.items())))
        if self._changed("providers", provider_sig):
            self.provider_switches = {}
            self.key_fields = {}
            self.providers_column.controls = self._provider_tiles(n)
        for provider in self.providers:
            switch = self.provider_switches.get(provider["id"])
            if switch is not None:
                switch.value = bool(provider.get("enabled"))
        self.push(self.body)

    def _relink_banner(self, dest: Mapping, dest_label: str, n: int) -> ft.Control:
        mode = str(dest.get("mode") or "folder")
        return ft.Column([
            ft.Row([ft.Icon(ft.Icons.SYNC_PROBLEM, color=ft.Colors.ON_ERROR_CONTAINER),
                    ft.Text(f"Glossarion lost access to {dest_label}. Choose it again.",
                            color=ft.Colors.ON_ERROR_CONTAINER, expand=True)], spacing=8),
            ft.Row([
                ft.FilledButton(content="Choose again", on_click=lambda e: self.spawn(self.choose(mode, confirm=False)),
                                key=f"cloud-relink-choose-{n}"),
                ft.TextButton(content="Forget", on_click=lambda e: self.confirm_forget(),
                              key=f"cloud-relink-forget-{n}"),
            ], wrap=True, spacing=8),
        ], spacing=8, tight=True)

    def _dest_action_buttons(self, unavailable: Optional[str], dest: Optional[dict], n: int) -> list:
        if not dest:
            return []
        buttons: list = []
        if dest.get("mode") == "folder":
            buttons.append(ft.OutlinedButton(content="Test", icon=ft.Icons.FACT_CHECK_OUTLINED,
                                             on_click=lambda e: self.spawn(self.test_destination()),
                                             disabled=bool(unavailable), key=f"cloud-test-{n}"))
        buttons.append(ft.TextButton(content="Forget destination", icon=ft.Icons.LINK_OFF,
                                     on_click=lambda e: self.confirm_forget(), key=f"cloud-forget-{n}"))
        return buttons

    def _destination_tiles(self, unavailable: Optional[str], dest: Optional[dict], n: int) -> list:
        current = str(dest.get("mode") or "") if dest else ""
        tiles: list = []
        for mode, _op, title, subtitle, icon in DESTINATIONS:
            reason = unavailable
            if not reason and mode == "phone" and (self.platform != "android" or not self.state.get("phone_folder",
                                                                                                    True)):
                reason = ANDROID_ONLY
            if reason:
                detail = IOS_PHONE_FOLDER_DETAIL if reason == ANDROID_ONLY else reason_detail(reason)
                tiles.append(unavailable_tile(title, reason=reason, detail=detail, subtitle=subtitle,
                                              leading=ft.Icon(icon_data(icon)), key=f"cloud-dest-{mode}-{n}",
                                              min_height=tokens.SIZES["hit_target"]))
                continue
            selected = mode == current
            tiles.append(ft.ListTile(
                leading=ft.Icon(icon_data(icon)),
                title=ft.Text(title + (" · in use" if selected else "")),
                subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL),
                trailing=ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY) if selected else None,
                on_click=lambda e, m=mode: self.spawn(self.choose(m)),
                min_height=tokens.SIZES["hit_target"],
                key=f"cloud-dest-{mode}-{n}",
            ))
        return tiles

    def _render_activity(self, n: int) -> None:
        state = self.state
        now = self.clock()
        self.summary.value = summary_text(state, now) if self.facade.available else NO_SERVICE_REASON
        saving = bool(state.get("saving") or state.get("progress"))
        self.progress_bar.visible = saving
        self.progress_text.visible = saving
        if saving:
            progress = state.get("progress") if isinstance(state.get("progress"), Mapping) else {}
            pct = _percent(progress.get("written"), progress.get("total"))
            self.progress_bar.value = None if pct is None else pct / 100.0
            self.progress_text.value = cloud_progress_text(str(progress.get("name") or "books"),
                                                           destination_text(self.destination),
                                                           progress.get("written"), progress.get("total"))
        rows: list = []
        queue = list(state.get("queue") or ())
        for index, item in enumerate(queue[:20]):
            rows.append(self._activity_tile(item, index, n, waiting=True, now=now))
        for index, item in enumerate(list(state.get("recent") or ())[:10]):
            rows.append(self._activity_tile(item, index, n, waiting=False, now=now))
        if not rows:
            rows.append(ft.Text("Nothing waiting." if self.destination else "Choose a destination to start.",
                                theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                key=f"cloud-activity-empty-{n}"))
        self.activity_column.controls = rows
        self.retry_button.disabled = not queue or not self.facade.available
        self.retry_button.visible = self.facade.available

    def _activity_tile(self, item: Mapping, index: int, n: int, *, waiting: bool, now: float) -> ft.Control:
        name = str(item.get("name") or item.get("title") or _KIND_LABELS.get(str(item.get("kind") or ""), "Book"))
        entry = dict(item)
        entry.setdefault("status", "waiting" if waiting else "ok")
        line = file_status_line(entry, dest_label=destination_text(self.destination), kind=str(item.get("kind") or ""),
                                now=now) or ("CLOUD_QUEUE", str(entry.get("status")), "muted")
        icon, text, tone = line
        return ft.ListTile(
            leading=ft.Icon(icon_data(icon), color=_tone_color(tone)),
            title=ft.Text(name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=_tone_color(tone)),
            dense=True, min_height=tokens.SIZES["hit_target"],
            key=f"cloud-{'queue' if waiting else 'recent'}-{index}-{n}",
        )

    def _provider_tiles(self, n: int) -> list:
        tiles: list = []
        if not self.share_facade.available:
            tiles.append(unavailable_tile(SHARE_LABEL, reason=NO_SERVICE_REASON, detail=NO_SERVICE_DETAIL,
                                          key=f"share-unavailable-{n}"))
            return tiles
        for provider in self.providers:
            pid = provider["id"]
            facts = [str(provider.get("operator") or "").strip()]
            if provider.get("handoff"):
                facts.append("you upload on its web page")
            facts.append("end-to-end encrypted" if provider.get("e2ee") else "the service can read the file")
            switch = ft.Switch(value=bool(provider.get("enabled")), on_change=lambda e, p=pid: self._on_provider(e, p),
                               key=f"share-toggle-{pid}-{n}")
            self.provider_switches[pid] = switch
            tiles.append(ft.ListTile(
                leading=ft.Icon(icon_data("OPEN_IN_BROWSER" if provider.get("handoff") else
                                          ("LOCK_OUTLINE" if provider.get("e2ee") else "LINK"))),
                title=ft.Text(str(provider.get("label") or pid)),
                subtitle=ft.Text(" · ".join(f for f in facts if f), theme_style=ft.TextThemeStyle.BODY_SMALL),
                trailing=switch, min_height=tokens.SIZES["hit_target"], key=f"share-provider-{pid}-{n}",
            ))
            retention = str(provider.get("retention") or "").strip()
            if retention:
                tiles.append(ft.Container(content=ft.Text(retention, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                                          color=ft.Colors.ON_SURFACE_VARIANT),
                                          padding=ft.Padding.only(left=56, right=8), key=f"share-retention-{pid}-{n}"))
            if provider.get("needs_key"):
                tiles.append(self._key_row(provider, n))
            if pid == "send" and self.send_options:
                tiles.append(self._send_row(n))
            if pid == "gofile":
                tiles.append(ft.Container(content=ft.TextButton(
                    content="Start a new guest account", icon=ft.Icons.PERSON_OFF_OUTLINED,
                    tooltip="Glossarion forgets its Gofile guest account; links made with it stay deletable",
                    on_click=lambda e: self.spawn(self.forget_gofile()), key=f"share-gofile-reset-{n}"),
                    padding=ft.Padding.only(left=48)))
        if not self.providers:
            tiles.append(ft.Text("No sharing services in this build.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 key=f"share-none-{n}"))
        return tiles

    def _key_row(self, provider: Mapping, n: int) -> ft.Control:
        pid = str(provider["id"])
        has_key = bool(provider.get("has_key"))
        field = ft.TextField(label=f"Your {provider.get('label')} API key", password=True, can_reveal_password=True,
                             dense=True, expand=True, key=f"share-key-{pid}-{n}",
                             hint_text="A key is saved" if has_key else "Stored encrypted on this phone")
        self.key_fields[pid] = field
        status = ft.Text("Key saved (encrypted)" if has_key else "No key yet: the service needs your own free key",
                         theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                         key=f"share-key-status-{pid}-{n}")
        buttons = [ft.FilledTonalButton(content="Save key", on_click=lambda e, p=pid: self.spawn(self.save_key(p)),
                                        key=f"share-key-save-{pid}-{n}")]
        if has_key:
            buttons.append(ft.TextButton(content="Check key", on_click=lambda e, p=pid: self.spawn(self.check_key(p)),
                                         tooltip="Asks the service whether the key works (one request)",
                                         key=f"share-key-check-{pid}-{n}"))
            buttons.append(ft.TextButton(content="Remove key", on_click=lambda e, p=pid: self.spawn(self.remove_key(p)),
                                         key=f"share-key-remove-{pid}-{n}"))
        return ft.Container(content=ft.Column([ft.Row([field], spacing=8), status,
                                               ft.Row(buttons, wrap=True, spacing=8)], spacing=4, tight=True),
                            padding=ft.Padding.only(left=56, right=8, bottom=8), key=f"share-key-row-{pid}-{n}")

    def _send_row(self, n: int) -> ft.Control:
        options = self.send_options
        expire = int(options.get("expire") or SEND_EXPIRY_CHOICES[-1][0])
        downloads = int(options.get("downloads") or SEND_DOWNLOAD_CHOICES[-1])
        self.send_expire = ft.Dropdown(
            label="Link lasts", value=str(expire), dense=True, expand=True, key=f"share-send-expire-{n}",
            options=[ft.DropdownOption(key=str(seconds), text=label) for seconds, label in SEND_EXPIRY_CHOICES],
            on_select=lambda e: self.spawn(self.set_send_options(expire=int(e.control.value or expire))))
        self.send_downloads = ft.Dropdown(
            label="Downloads", value=str(downloads), dense=True, expand=True, key=f"share-send-downloads-{n}",
            options=[ft.DropdownOption(key=str(count), text=str(count)) for count in SEND_DOWNLOAD_CHOICES],
            on_select=lambda e: self.spawn(self.set_send_options(downloads=int(e.control.value or downloads))))
        # no wrap: an expanding child inside a wrapping Row is a Flutter layout error (a grey box on release builds)
        return ft.Container(content=ft.Row([self.send_expire, self.send_downloads], spacing=8),
                            padding=ft.Padding.only(left=56, right=8, bottom=8), key=f"share-send-row-{n}")

    # ---- handlers ----

    def show_explainer(self) -> InfoSheet:
        return self.show(InfoSheet(title=EXPLAINER_TITLE, body=EXPLAINER, markdown=True))

    def show_share_explainer(self) -> InfoSheet:
        return self.show(InfoSheet(title="About share links", body=SHARE_EXPLAINER, markdown=True))

    async def choose(self, mode: str, *, confirm: bool = True) -> Any:
        """A destination tile (or Choose again): the system folder picker, save-locations mode, or the phone
        folder. The service refuses Glossarion's own folder and adopts same-named files it finds.

        Changing an existing destination asks first: copies already made stay where they are and are no longer
        updated (the service releases the old access and retires its records). Returns the dialog then."""
        op = next((o for m, o, *_rest in DESTINATIONS if m == mode), None)
        if op is None:
            return {"ok": False, "message": "Unknown destination"}
        dest = self.destination
        changing = dest and not dest.get("needs_relink") and (mode != dest.get("mode") or mode == "folder")
        if confirm and changing:
            old = destination_text(dest)

            async def run() -> None:
                await self._choose(op)

            return self.show(ConfirmDialog(
                title="Change destination?",
                body=(f"New copies go to the new place. Files already copied to {old} stay there and are no longer "
                      "updated."),
                confirm_label="Change", on_confirm=run))
        return await self._choose(op)

    async def _choose(self, op: str) -> dict:
        result = await self.facade.call(op)
        if not result.get("cancelled"):
            message = str(result.get("message") or "")
            if not result.get("ok"):
                self.say(message or "The destination was not changed")
            else:
                self.say(message or f"Destination: {destination_text(result.get('destination'))}")
        await self.refresh()
        return result

    async def test_destination(self) -> dict:
        self.say("Testing the folder…")
        result = await self.facade.call("test_destination")
        self.say(str(result.get("message") or ("The folder accepts files" if result.get("ok") else
                                               "The folder refused the test file")))
        await self.refresh()
        return result

    def confirm_forget(self) -> ConfirmDialog:
        dest = destination_text(self.destination)

        async def run() -> None:
            result = await self.facade.call("forget_destination")
            self.say("Destination forgotten" if result.get("ok") else str(result.get("message") or
                                                                         "Could not forget the destination"))
            await self.refresh()

        return self.show(ConfirmDialog(
            title="Forget destination?",
            body=(f"Glossarion stops copying books to {dest} and gives back its access to it. Files already copied "
                  "stay where they are."),
            confirm_label="Forget", destructive=True, on_confirm=run))

    def _on_auto(self, e: Any = None) -> None:
        value = bool(getattr(getattr(e, "control", None), "value", self.auto_switch.value))
        self.spawn(self.set_auto(value))

    async def set_auto(self, value: bool) -> dict:
        reason = self.availability_reason() or (NO_DESTINATION_REASON if not self.destination else None)
        if reason and value:
            self.auto_switch.value = False
            self.push(self.auto_switch)
            self.say(REASON_DETAILS.get(reason, reason))
            return {"ok": False, "message": reason}
        result = await self.facade.call("set_enabled", bool(value))
        if not result.get("ok"):
            self.say(str(result.get("message") or "Could not change the setting"))
        elif value:
            self.say("Finished books are copied automatically from now on")
        await self.refresh()
        return result

    def _on_kind(self, e: Any, kind: str) -> None:
        control = getattr(e, "control", None)
        value = bool(getattr(control, "selected", not self.state.get("kinds", {}).get(kind, True)))
        self.spawn(self.set_kind(kind, value))

    async def set_kind(self, kind: str, value: bool) -> dict:
        result = await self.facade.call("set_kind", kind, bool(value))
        if not result.get("ok"):
            self.say(str(result.get("message") or "Could not change the setting"))
        await self.refresh()
        return result

    async def retry_now(self) -> dict:
        result = await self.facade.retry_now(spawn=self.spawn)
        self.say(str(result.get("message") or ("Retrying" if result.get("ok") else "Nothing could be retried")))
        await self.refresh()
        return result

    def provider(self, pid: str) -> Optional[dict]:
        return next((p for p in self.providers if p.get("id") == pid), None)

    def _on_provider(self, e: Any, pid: str) -> None:
        provider = self.provider(pid) or {"id": pid}
        value = bool(getattr(getattr(e, "control", None), "value", not provider.get("enabled")))
        self.spawn(self.set_provider(pid, value))

    async def set_provider(self, pid: str, value: bool) -> bool:
        """Turning a service on shows its consent sheet every time (the user reads what leaves the phone and who
        can read it); off is immediate."""
        provider = self.provider(pid) or {"id": pid, "label": pid}
        consent = False
        if value:
            accepted = await self.actions_.ask_consent(provider, None, confirm_label="Turn on")
            if not accepted:
                switch = self.provider_switches.get(pid)
                if switch is not None:  # the switch goes back off
                    switch.value = False
                    self.push(switch)
                await self.refresh()
                return False
            consent = True
        result = await self.share_facade.enable(pid, bool(value), consent=consent)
        if not result.get("ok"):
            self.say(str(result.get("message") or "Could not change the setting"))
        await self.refresh()
        return bool(result.get("ok"))

    async def save_key(self, pid: str) -> bool:
        field = self.key_fields.get(pid)
        value = str(getattr(field, "value", "") or "").strip()
        if not value:
            self.say("Paste your API key first")
            return False
        result = await self.share_facade.set_key(pid, value)
        if field is not None:
            field.value = ""  # never left on screen
            self.push(field)
        self.say("Key saved (encrypted)" if result.get("ok") else str(result.get("message") or "Could not save the key"))
        await self.refresh()
        return bool(result.get("ok"))

    async def check_key(self, pid: str) -> bool:
        self.say("Checking the key…")
        result = await self.share_facade.check_key(pid)
        works = bool(result.get("ok")) and result.get("value", True) is not False
        self.say("The key works" if works else str(result.get("message") or "The service did not accept the key"))
        return works

    async def remove_key(self, pid: str) -> bool:
        result = await self.share_facade.clear_key(pid)
        self.say("Key removed" if result.get("ok") else str(result.get("message") or "Could not remove the key"))
        await self.refresh()
        return bool(result.get("ok"))

    async def forget_gofile(self) -> bool:
        result = await self.share_facade.call("forget_gofile_account")
        self.say("Gofile guest account forgotten: the next upload starts a new one" if result.get("ok") else
                 str(result.get("message") or "Could not forget the account"))
        return bool(result.get("ok"))

    async def set_send_options(self, *, expire: Optional[int] = None, downloads: Optional[int] = None) -> dict:
        result = await self.share_facade.call("set_send_options", expire=expire, downloads=downloads)
        if not result.get("ok"):
            self.say(str(result.get("message") or "Could not change the Send options"))
        await self.refresh()
        return result
