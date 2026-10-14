"""Document destinations: the contract shared by the Python API, the native code and test fakes.

Glossarion writes finished books into a place the user picked once in the system picker:

* Android: a folder through the Storage Access Framework (``ACTION_OPEN_DOCUMENT_TREE`` +
  ``takePersistableUriPermission``) or a single "save location" (``ACTION_CREATE_DOCUMENT``) when the
  cloud app does not offer folders. Writes go through ``ContentResolver.openFileDescriptor`` with a
  mode chain (``wt`` -> ``rwt`` -> ``w`` by default) and are checked by reading the length back.
* iOS: a folder from the Files picker (iCloud Drive, On My iPhone) or one exported file
  (Drive / OneDrive / Dropbox), both kept as minimal bookmarks; writes are coordinated
  (``NSFileCoordinator``) and swap a staged copy in with ``replaceItemAt``.

This module has no Flet import: the app, the fakes and the tests use it on any host.

References ("refs")
-------------------
Every picked or created item is a plain JSON-friendly dict that the app stores as is and passes back
unchanged. Keys (``REF_KEYS``):

``platform``      ``"android"`` / ``"ios"``
``kind``          ``"folder"`` or ``"file"``
``id``            stable id: the picked folder (tree URI / folder path) for a folder target, else the
                  document; records are keyed by the destination's ``id`` (see :func:`stable_id`)
``uri``           Android: the tree URI when the item lives in a picked folder, else None
``document``      Android: the document URI (folder or file)
``bookmark``      iOS: minimal bookmark (base64) of the item itself (None for a ``.icloud`` placeholder)
``root``          iOS: bookmark of the picked folder for items inside it, else None
``path``          iOS: path relative to ``root`` (``"Book/Book.epub"``), else None
``name``, ``mime``, ``size``, ``mtime`` (ms since the epoch), ``flags`` (Android Document flags)
``provider``      Android content authority; iOS ``icloud`` / ``local`` / ``file_provider``
``provider_label`` the cloud app's name when known (``"Drive"``)
``can_write``, ``can_create``, ``can_delete``
``persisted``     the permission survives a restart (Android grant persisted / iOS bookmark made)
``own_folder``    the item is inside Glossarion's own storage (app container, Download/Glossarion):
                  the app must refuse it as a cloud destination

Results
-------
Every call returns ``{"ok": bool, "error": code | None, "message": str | None, "scope": ...,
"retryable": bool, ...}``. ``scope`` says what a failure affects: ``"target"`` (the whole
destination: re-link), ``"document"`` (only this file: re-create / re-pick it) or ``"source"``
(the local file). Error codes are :class:`DocumentError`.

``missing`` is only reported when the picked folder still answers and the document is not there
(Android: the tree root query returns a row, the document does not); a ``FileNotFoundException``
alone is never taken as "deleted", because cloud providers throw it for an unsupported write mode or
while offline. A per-file grant (no folder to ask) reports ``missing`` with ``proven: False``.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "DEFAULT_MODE_CHAIN",
    "DocumentError",
    "DocumentScope",
    "MAX_WRITE_TIMEOUT",
    "PICKER_TIMEOUT",
    "QUERY_TIMEOUT",
    "REF_KEYS",
    "RETRYABLE_ERRORS",
    "TRUNCATING_MODES",
    "VALID_MODES",
    "error_result",
    "is_retryable",
    "normalize_modes",
    "normalize_result",
    "redact",
    "ref_id",
    "same_destination",
    "stable_id",
    "write_timeout",
]

#: Write modes tried in order (critic: ``wt`` first, it skips Nextcloud's server-change check).
DEFAULT_MODE_CHAIN: tuple[str, ...] = ("wt", "rwt", "w")
#: Modes the native writers accept. ``w`` / ``rw`` do not truncate: the native side only uses them
#: when the new file is at least as long as the cloud copy, or when it can cut the tail itself.
VALID_MODES: tuple[str, ...] = ("wt", "rwt", "w", "rw")
TRUNCATING_MODES: tuple[str, ...] = ("wt", "rwt")

#: Pickers wait for the user (Flet's own FilePicker uses one hour too).
PICKER_TIMEOUT = 3600.0
#: Listing, stat, create, delete: providers may go to the network.
QUERY_TIMEOUT = 120.0
#: Upper bound of :func:`write_timeout`.
MAX_WRITE_TIMEOUT = 7200.0

REF_KEYS: tuple[str, ...] = (
    "platform", "kind", "id", "uri", "document", "bookmark", "root", "path", "name", "mime", "size",
    "mtime", "flags", "provider", "provider_label", "can_write", "can_create", "can_delete",
    "persisted", "own_folder", "virtual",
)


class DocumentError(str, Enum):
    """``error`` codes of document results."""

    CANCELLED = "cancelled"
    """The user closed the picker, the operation was cancelled, or Glossarion was restarted while a
    picker was open."""
    PERMISSION_LOST = "permission_lost"
    """The grant / bookmark no longer works (revoked, app reinstalled, cloud app removed). With
    ``scope == "target"`` the destination must be picked again."""
    MISSING = "missing"
    """The document is not there although its folder answers (deleted or moved out)."""
    UNSUPPORTED_MODE = "unsupported_mode"
    """The cloud app accepted none of the write modes. ``needs_replace`` is True when a
    create-new + delete-old fallback is the only safe way left."""
    PROVIDER_ERROR = "provider_error"
    """Any other provider failure (offline, slow, eventually consistent). Retry later."""
    NO_SPACE = "no_space"
    """No space left (phone or cloud staging). Not retried automatically."""
    SOURCE_MISSING = "source_missing"
    """The local file to upload does not exist."""
    SOURCE_CHANGED = "source_changed"
    """The local file changed while it was copied (use a private snapshot)."""
    READ_ONLY = "read_only"
    """The document or folder does not accept writes / new files."""
    SIZE_MISMATCH = "size_mismatch"
    """The length read back differs from what was written. ``stale_tail`` means an older, longer
    copy left bytes at the end (non-truncating provider): replace the file."""
    EXISTS = "exists"
    """``on_exists="fail"`` and an item with that name is already there (``document`` is it)."""
    BUSY = "busy"
    """Another picker is already open."""
    UNAVAILABLE = "unavailable"
    """No native document support here (desktop, web, companion app, no activity)."""
    TIMEOUT = "timeout"
    """The Python side stopped waiting; a write is cancelled through ``cancel_document_op``."""
    BAD_ARGS = "bad_args"


class DocumentScope(str, Enum):
    """``scope`` of a failure."""

    TARGET = "target"
    DOCUMENT = "document"
    SOURCE = "source"


RETRYABLE_ERRORS = frozenset({
    DocumentError.PROVIDER_ERROR.value,
    DocumentError.SOURCE_CHANGED.value,
    DocumentError.TIMEOUT.value,
    DocumentError.BUSY.value,
})

_ERROR_VALUES = frozenset(e.value for e in DocumentError)
_SCOPE_VALUES = frozenset(s.value for s in DocumentScope)

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def stable_id(identity: str) -> str:
    """``"d"`` + 16 hex digits of FNV-1a/64 over UTF-8 ``identity``.

    The Kotlin and Swift sides compute the same function over ``"android:<uri>"`` /
    ``"ios:<canonical path>"``; tests use it to build refs like the native code does.
    """
    value = _FNV_OFFSET
    for byte in str(identity).encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK64
    return "d%016x" % value


def ref_id(ref: Any) -> Optional[str]:
    """The ref's ``id`` (falls back to hashing its URI / bookmark when an old ref has none)."""
    if not isinstance(ref, Mapping):
        return None
    value = ref.get("id")
    if value:
        return str(value)
    platform = ref.get("platform") or ""
    for key in ("uri", "document", "bookmark"):
        if ref.get(key):
            return stable_id(f"{platform}:{ref[key]}")
    return None


def same_destination(a: Any, b: Any) -> bool:
    """True when two refs name the same picked destination (same ``id``)."""
    first, second = ref_id(a), ref_id(b)
    return bool(first) and first == second


def is_retryable(code: Any) -> bool:
    value = getattr(code, "value", code)
    return value in RETRYABLE_ERRORS


def error_result(code: Any, message: Optional[str] = None, *, scope: Any = None, **extra: Any) -> dict:
    """A failure result in the shared shape."""
    value = getattr(code, "value", code)
    if value not in _ERROR_VALUES:
        value = DocumentError.PROVIDER_ERROR.value
    scope_value = getattr(scope, "value", scope)
    out: dict[str, Any] = {
        "ok": False,
        "error": value,
        "message": message,
        "scope": scope_value if scope_value in _SCOPE_VALUES else None,
        "retryable": value in RETRYABLE_ERRORS,
    }
    out.update(extra)
    return out


def normalize_result(raw: Any, *, message: Optional[str] = None) -> dict:
    """Coerce a native answer into the shared shape (never raises)."""
    if not isinstance(raw, Mapping):
        return error_result(DocumentError.PROVIDER_ERROR, message or "Unexpected answer from the native side")
    out = {str(k): v for k, v in raw.items()}
    if out.get("ok") is True:
        out["ok"] = True
        out["error"] = None
        out.setdefault("message", None)
        out.setdefault("retryable", False)
        return out
    code = out.get("error")
    if code not in _ERROR_VALUES:
        out["message"] = out.get("message") or (f"Unknown error {code!r}" if code else message)
        code = DocumentError.PROVIDER_ERROR.value
    out["ok"] = False
    out["error"] = code
    out.setdefault("message", message)
    if out.get("scope") not in _SCOPE_VALUES:
        out["scope"] = None
    out["retryable"] = code in RETRYABLE_ERRORS
    return out


def normalize_modes(modes: Any) -> tuple[str, ...]:
    """Valid, de-duplicated write modes in the caller's order (empty when none is valid)."""
    if modes is None:
        return DEFAULT_MODE_CHAIN
    if isinstance(modes, str):
        modes = [modes]
    out: list[str] = []
    for mode in modes:
        mode = str(mode).strip().lower()
        if mode in VALID_MODES and mode not in out:
            out.append(mode)
    return tuple(out)


def write_timeout(size_bytes: Optional[int]) -> float:
    """How long Python waits for a write: 120 s + 1 s per MiB, at most two hours."""
    try:
        size = max(int(size_bytes or 0), 0)
    except (TypeError, ValueError):
        size = 0
    return min(120.0 + size / (1024 * 1024), MAX_WRITE_TIMEOUT)


def redact(value: Any) -> str:
    """Log-safe label for a ref, URI or bookmark: provider + short hash, never the path."""
    if isinstance(value, Mapping):
        provider = value.get("provider_label") or value.get("provider") or value.get("platform") or "?"
        ident = value.get("id") or ref_id(value) or ""
        return f"{provider}#{str(ident)[-6:]}"
    text = str(value or "")
    provider = "?"
    if text.startswith("content://"):
        provider = text[len("content://"):].split("/", 1)[0]
    elif text.startswith("file:") or text.startswith("/"):
        provider = "file"
    elif text:
        provider = "bookmark"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return f"{provider}#{digest}"


def names_of(children: Optional[Sequence[Mapping]]) -> list[str]:
    """Display names of a ``list_children`` answer (helper for adopt-by-name)."""
    return [str(c.get("name")) for c in (children or ()) if isinstance(c, Mapping) and c.get("name")]
