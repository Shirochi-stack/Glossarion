"""Generated media and refinement compare for chat cards (pure Python, no Flet, Python 3.10).

UI_SPEC §2.9 item 4 / §2.6: a response may carry ``[GENERATED_IMAGE|VIDEO|AUDIO:<path>]``
markers. Which file belongs to a response is the shared desktop rule
(``direct_text_store.ChatStoreMixin._assistant_generated_media`` /
``_generated_media_references_from_text``, through ``ChatStoreAdapter.message_media``); this
module only shapes the result for the cards:

* ``MediaItem`` / ``media_items`` - ``(kind, path, exists)`` tuples -> items with a file name;
* ``display_content`` - the response text without the markers the cards render (the desktop
  replaces a marker in place with the image or the player; a missing file becomes
  "**Generated <kind> unavailable.**" plus the file name, as on desktop);
* ``format_clock`` - "m:ss" for the AudioCard position/duration;
* ``image_size`` / ``capped_image_height`` - a generated image's size from its header (cached
  per file) so a single image renders at most ``IMAGE_MAX_HEIGHT`` tall (UI_SPEC §2.9);
* ``compare_blocks`` / ``find_unrefined_backup`` - the Refine card's "Compare with original"
  (the run's ``translation_progress.json`` records ``unrefined_backup_file`` for refined
  chapters; the stacked diff pairs paragraphs with ``difflib``);
* ``ocr_entries`` - the Vision card's "OCR" section: the per-image OCR texts a Vision run
  caches in its workspace (``OCR/single`` and ``OCR/chunks``, the folders the EPUB
  converter's gallery filter reads too).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "AUDIO_DEFAULT_VOLUME",
    "GENERATIVE_MODES",
    "IMAGE_MAX_HEIGHT",
    "MARKER_PATTERN",
    "MEDIA_KINDS",
    "MediaItem",
    "VIDEO_ASPECT_RATIO",
    "capped_image_height",
    "compare_blocks",
    "display_content",
    "find_unrefined_backup",
    "format_clock",
    "image_size",
    "media_items",
    "ocr_entries",
    "unavailable_text",
]

MEDIA_KINDS = ("image", "video", "audio")
#: Output modes with "Generate from prompt (no input)" (UI_SPEC §2.6).
GENERATIVE_MODES = ("image", "video", "audio")
#: The desktop media player's default volume (AudioCard, UI_SPEC §2.9).
AUDIO_DEFAULT_VOLUME = 0.75
VIDEO_ASPECT_RATIO = 16 / 9
#: A single generated image is at most this tall in a card (UI_SPEC §2.9).
IMAGE_MAX_HEIGHT = 760.0
#: The desktop marker pattern (``_is_generated_media_only_content``).
MARKER_PATTERN = r"\[GENERATED_(IMAGE|VIDEO|AUDIO):(.+?)\]"


@dataclass(frozen=True)
class MediaItem:
    kind: str  # image | video | audio
    path: str
    exists: bool

    @property
    def name(self) -> str:
        return os.path.basename(self.path) or self.path


def media_items(raw: Iterable[Sequence[Any]]) -> list:
    """``ChatStoreAdapter.message_media`` tuples -> ``MediaItem`` list (unknown kinds dropped)."""
    out = []
    for item in raw or ():
        try:
            kind, path, exists = item[0], item[1], item[2]
        except (IndexError, TypeError):
            continue
        kind = str(kind or "").lower()
        if kind in MEDIA_KINDS and path:
            out.append(MediaItem(kind, str(path), bool(exists)))
    return out


def unavailable_text(kind: str, path: str) -> str:
    """The desktop missing-media text ("**Generated video unavailable.**" + the file)."""
    name = os.path.basename(str(path or "")) or str(path or f"unknown {kind}")
    return f"**Generated {kind} unavailable.**\n\n`{name}`"


def display_content(content: Any, items: Sequence[MediaItem] = ()) -> str:
    """The response text the card shows: markers of rendered media removed, missing ones explained."""
    source = str(content or "")
    if not re.search(MARKER_PATTERN, source, flags=re.IGNORECASE):
        return source
    by_path = {os.path.normcase(os.path.abspath(i.path)): i for i in items}

    def replace(match: "re.Match[str]") -> str:
        kind = match.group(1).lower()
        value = str(match.group(2) or "").strip().strip("\"'")
        path = os.path.abspath(os.path.expanduser(value)) if value else ""
        item = by_path.get(os.path.normcase(path)) if path else None
        if item is not None and item.exists:
            return ""
        if item is None and path and os.path.isfile(path):
            return ""  # a sibling marker the card set shows anyway (gallery)
        return unavailable_text(kind, path or value)

    text = re.sub(MARKER_PATTERN, replace, source, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_SIZE_CACHE: "OrderedDict[tuple, Optional[tuple]]" = OrderedDict()
_SIZE_LOCK = threading.Lock()


def image_size(path: str) -> Optional[tuple]:
    """``(width, height)`` of an image file from its header (Pillow reads only the header), or None.

    Cached per file (path, size, mtime): a card re-render reads nothing again.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)
    with _SIZE_LOCK:
        if key in _SIZE_CACHE:
            _SIZE_CACHE.move_to_end(key)
            return _SIZE_CACHE[key]
    size: Optional[tuple] = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        if width > 0 and height > 0:
            size = (int(width), int(height))
    except Exception:
        size = None
    with _SIZE_LOCK:
        _SIZE_CACHE[key] = size
        while len(_SIZE_CACHE) > 256:
            _SIZE_CACHE.popitem(last=False)
    return size


def capped_image_height(width: float, size: Optional[tuple], cap: float = IMAGE_MAX_HEIGHT) -> Optional[float]:
    """The height of an image shown ``width`` wide, at most ``cap``; None when the size is unknown."""
    if not size or not size[0] or not size[1]:
        return None
    return min(float(cap), float(width) * float(size[1]) / float(size[0]))


def format_clock(milliseconds: Any) -> str:
    """"m:ss" (or "h:mm:ss") for a position / duration in milliseconds."""
    try:
        total = max(0, int(float(milliseconds or 0) // 1000))
    except (TypeError, ValueError):
        total = 0
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Refine: compare with the original (UI_SPEC §2.6 Refine row)
# ---------------------------------------------------------------------------


def _progress_files(folder: str) -> list:
    """``translation_progress.json`` in ``folder`` and its immediate subfolders (attachment workspaces)."""
    found = []
    if not folder or not os.path.isdir(folder):
        return found
    direct = os.path.join(folder, "translation_progress.json")
    if os.path.isfile(direct):
        found.append(direct)
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        return found
    for entry in entries:
        candidate = os.path.join(entry.path, "translation_progress.json")
        if entry.is_dir() and os.path.isfile(candidate):
            found.append(candidate)
    return found


def find_unrefined_backup(output_folder: str, request_label: str = "") -> Optional[str]:
    """Blocking: the ``unrefined_backup_file`` of the chapter this response refined, or None.

    The run's progress file records it per refined chapter (TransateKRtoEN refinement mode); a
    response label naming the chapter file ("… · ch012.xhtml · Request 3") picks that entry,
    otherwise the folder's only refined entry is used.
    """
    label = str(request_label or "").lower()
    for progress_path in _progress_files(output_folder):
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        base = os.path.dirname(progress_path)
        candidates = []
        for entry in (data.get("chapters") or {}).values() if isinstance(data, dict) else ():
            if not isinstance(entry, dict) or not entry.get("unrefined_backup_file"):
                continue
            backup = os.path.normpath(os.path.join(base, str(entry["unrefined_backup_file"])))
            if os.path.isfile(backup):
                candidates.append((entry, backup))
        for entry, backup in candidates:
            names = {os.path.basename(str(entry.get(key) or "")).lower()
                     for key in ("output_file", "original_basename", "source_file") if entry.get(key)}
            if label and any(name and name in label for name in names):
                return backup
        if len(candidates) == 1:
            return candidates[0][1]
    return None


def _paragraphs(text: str) -> list:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "").replace("\r\n", "\n")) if p.strip()]


def compare_blocks(original: str, refined: str) -> list:
    """Stacked paragraph diff: ``[(tag, original paragraph, refined paragraph)]``.

    ``tag`` is ``equal`` / ``replace`` / ``delete`` / ``insert`` (``difflib.SequenceMatcher``
    opcodes over paragraphs); a replaced run pairs its paragraphs in order.
    """
    old, new = _paragraphs(original), _paragraphs(refined)
    blocks: list = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            blocks.extend(("equal", old[i], new[j]) for i, j in zip(range(i1, i2), range(j1, j2)))
        elif tag == "delete":
            blocks.extend(("delete", old[i], "") for i in range(i1, i2))
        elif tag == "insert":
            blocks.extend(("insert", "", new[j]) for j in range(j1, j2))
        else:
            left, right = old[i1:i2], new[j1:j2]
            for k in range(max(len(left), len(right))):
                blocks.append(("replace", left[k] if k < len(left) else "", right[k] if k < len(right) else ""))
    return blocks


# ---------------------------------------------------------------------------
# Vision: the OCR section
# ---------------------------------------------------------------------------


def ocr_entries(workspace: str, limit: int = 200) -> list:
    """Blocking: ``[(file name, text)]`` of a Vision run's cached OCR (``OCR/single``, then
    ``OCR/chunks``; empty files skipped, at most ``limit``)."""
    found: list = []
    base = os.path.join(str(workspace or ""), "OCR")
    for kind in ("single", "chunks"):
        folder = os.path.join(base, kind)
        try:
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".txt"))
        except OSError:
            continue
        for name in names:
            try:
                with open(os.path.join(folder, name), "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read().strip()
            except OSError:
                continue
            if text:
                found.append((name, text))
            if len(found) >= limit:
                return found
    return found
