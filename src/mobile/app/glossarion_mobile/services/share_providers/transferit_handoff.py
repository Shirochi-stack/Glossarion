"""transfer.it: a browser handoff only - Glossarion never calls transfer.it.

transfer.it (Mega Privacy (NZ) Limited, a MEGA group company, on MEGA infrastructure) has no
public API. Its client protocol is private, the client code is under a review-only licence, and
MEGA's Terms ("Our IP") require developer registration and MEGA's approval before an app may use
its API. So the app only:

1. makes the file reachable for the browser's file chooser: Android saves it to
   Downloads/Glossarion (``FileBridge.save_to_downloads``), overwriting the entry an earlier handoff of
   the same file made (``replace_uri``: a recompiled book never leaves the old copy under the name the
   hint shows, nor a second ``name (1).epub``) and naming the entry MediaStore really has; on iOS the
   Library and Output folders are already in Files (On My iPhone › Glossarion), so nothing is copied and
   the path is shown;
2. opens https://transfer.it/start in the in-app browser (Custom Tab / SFSafariViewController),
   with the hint where to find the file;
3. takes the link the user copies there and pastes back (``parse_link``), and saves it on the book.

The upload runs in the browser, so Glossarion's foreground service does not protect it: the hint
asks the user to keep the page open until transfer.it shows the link.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlsplit

from glossarion_mobile.services.share_providers import ShareError, provider_info

__all__ = ["HandoffPlan", "LINK_RE", "START_URL", "TransferItHandoff", "parse_link"]

log = logging.getLogger("glossarion.share.transferit")

START_URL = "https://transfer.it/start"
#: A transfer.it link: https://transfer.it/t/<transfer handle>
LINK_RE = re.compile(r"^https://transfer\.it/t/([A-Za-z0-9_-]{6,64})/?$")
_URL_IN_TEXT = re.compile(r"https://transfer\.it/t/[A-Za-z0-9_-]+/?", re.I)
KEEP_OPEN = "Keep the page open until transfer.it shows the link, then tap Copy link and come back here to paste it."


def parse_link(text: Any) -> str:
    """The transfer.it link in pasted text (``https://transfer.it/t/<handle>``); ``ShareError('bad_link')``."""
    raw = str(text or "").strip()
    match = _URL_IN_TEXT.search(raw)
    candidate = match.group(0) if match else raw
    parts = urlsplit(candidate)
    normalised = f"https://{(parts.hostname or '').lower()}{parts.path}" if parts.scheme.lower() == "https" else ""
    found = LINK_RE.match(normalised)
    if not found:
        raise ShareError("bad_link", "Paste the link transfer.it shows (it starts with https://transfer.it/t/).")
    return f"https://transfer.it/t/{found.group(1)}"


@dataclass(frozen=True)
class HandoffPlan:
    url: str  # the page the app opens
    file_name: str
    location: str  # where the browser's file chooser finds the file (display text)
    hint: str
    saved_to: Optional[str] = None  # Android: the Downloads entry (content URI)
    show_in_files: bool = False  # iOS: "Show in Files" makes sense
    saved_name: Optional[str] = None  # Android: the entry's name in Downloads/Glossarion


def _files_location(path: str, root: Optional[str]) -> Optional[str]:
    """``On My iPhone › Glossarion › Library › … › name`` for a file under the Files-visible root."""
    if not root:
        return None
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    except ValueError:
        return None
    if rel.startswith(os.pardir) or os.path.isabs(rel):
        return None
    parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
    return " › ".join(["On My iPhone", "Glossarion"] + parts)


class TransferItHandoff:
    """The handoff steps. ``open_url(url)`` opens the in-app browser (``InAppUrlOpener.launch``);
    ``save_to_downloads(path)`` is ``FileBridge.save_to_downloads`` (Android)."""

    info = provider_info("transferit")

    def __init__(self, *, platform: str = "desktop", open_url: Optional[Callable[[str], Any]] = None,
                 save_to_downloads: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
                 files_visible_root: Optional[str] = None) -> None:
        self.platform = platform
        self.open_url = open_url
        self.save_to_downloads = save_to_downloads
        self.files_visible_root = files_visible_root

    async def _save(self, path: str, replace_uri: Optional[str] = None) -> tuple:
        """``(uri, name)`` of the Downloads entry (``name`` None when the platform cannot say)."""
        save = self.save_to_downloads
        if save is None:
            return None, None
        kwargs: dict = {}
        try:
            params = inspect.signature(save).parameters
        except (TypeError, ValueError):
            params = {}
        if "replace_uri" in params and replace_uri:
            kwargs["replace_uri"] = replace_uri
        if "entry" in params:
            kwargs["entry"] = True
        result = await save(path, **kwargs)
        if isinstance(result, dict):
            uri = result.get("uri")
            return (str(uri) if uri else None), (str(result.get("name")) if result.get("name") else None)
        return (str(result) if result else None), None

    async def prepare(self, path: str, *, replace_uri: Optional[str] = None,
                      saved_name: Optional[str] = None) -> HandoffPlan:
        """``replace_uri`` / ``saved_name``: the entry an earlier handoff of this file made (overwritten in place
        when it is still Glossarion's; it keeps its first name)."""
        name = os.path.basename(path)
        if self.platform == "android":
            saved, entry_name = await self._save(path, replace_uri)
            if not saved:
                raise ShareError("unsupported", "Could not save the file to Downloads. Use Share… to send it "
                                 "to another app instead.")
            if not entry_name:
                # an entry overwritten in place keeps the name it was given first; a new one has the one asked for
                entry_name = saved_name if (replace_uri and saved == replace_uri and saved_name) else name
            location = f"Downloads › Glossarion › {entry_name}"
            hint = (f"On transfer.it tap Add files, then pick {location} (Glossarion saved it there). {KEEP_OPEN}")
            return HandoffPlan(START_URL, name, location, hint, saved_to=saved, saved_name=entry_name)
        if self.platform == "ios":
            location = _files_location(path, self.files_visible_root)
            if location is None:
                raise ShareError("unsupported", "This file is not in Glossarion's Files folder. Use Share… instead.")
            hint = f"On transfer.it tap Add files › Browse, then pick {location}. {KEEP_OPEN}"
            return HandoffPlan(START_URL, name, location, hint, show_in_files=True)
        location = os.path.dirname(os.path.abspath(path))
        hint = f"On transfer.it add {name} from {location}. {KEEP_OPEN}"
        return HandoffPlan(START_URL, name, location, hint)

    async def open(self, plan: HandoffPlan) -> bool:
        """Open the start page (never anything else); False when no opener is wired."""
        if plan.url != START_URL or self.open_url is None:
            return False
        result = self.open_url(START_URL)
        if asyncio.iscoroutine(result):
            await result
        return True
