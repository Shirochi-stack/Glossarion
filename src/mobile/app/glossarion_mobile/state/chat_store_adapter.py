"""ChatStoreAdapter: the chat's persistence, backed by the shared ``direct_text_store`` (U3).

Data parity (UI_SPEC §2.19, Appendix B):

* ``direct_text_chats.json`` v2 and the ``Direct Text/<chat>/`` tree are read and
  written ONLY by the shared ``direct_text_store.ChatStore`` (the desktop
  ``_InputOutputDialog`` persistence moved verbatim), so the desktop opens a chat
  history written on the phone and vice versa. Mobile writes nothing else into it.
* Mobile-only data (pins, per-chat overrides, "Skip plan", text size, last-activity
  times for the Recents headers, versions, pending plans) lives in the sidecar
  ``direct_text_chats.mobile.json`` next to the history file, keyed by chat id and
  message fingerprints. Orphans are dropped on load.

``ChatStoreBinding`` is the one place that knows the ChatStore API. It calls the
extracted methods by their desktop names (``_load_chat_history``,
``_save_chat_history``, ``_new_chat_session``, ``_assistant_message_text``,
``_ensure_conversation_output_folder_for_session``, ``_validated_chat_output_folder``,
``_conversation_attachment_folders``, ...), accepting the public spelling too.

``ChatStoreAdapter`` exposes:

* the drawer index interface of ``state.chat_index.InMemoryChatIndex`` (``all``,
  ``get``, ``pinned``, ``recents``, ``search``, ``set_pinned``, ``upsert``, ``remove``,
  ``subscribe``) so ``ChatDrawer`` binds to it unchanged;
* session operations with the desktop rules (reuse an empty chat on New chat,
  auto-title on the first send, 120-character rename, validated folder delete,
  draft autosave debounced 450 ms, lazy bodies with a 128-entry cache);
* message fingerprints / opaque ``mid`` ids for routes (Appendix B);
* U7 (mobile-only mutations that keep the v2 file valid, UI_SPEC §2.10 / §2.16 / §2.17 / §2.19):
  scratch chats (``s<uuid>`` ids; each one is a ``ChatStore`` of its own under the scratch
  folder, never written to ``direct_text_chats.json``; Save moves it into the history and its
  folder under ``Direct Text/`` through the shared ``_relocate_session_attachment_paths``),
  Delete message (renames the surviving responses' ``Chat Messages`` files to their new
  indices, re-indexes ``expanded``, remaps the sidecar fingerprints and records the turns of
  the chat's jobs in the sidecar ``job_turns``), message versions
  (sidecar ``versions``), the generated media of a response (the shared
  ``_assistant_generated_media`` / ``_generated_media_references_from_text``) and the desktop
  attachment Migrate (``ChatStore.migrate_attachment``).

Thread-safety: every method takes the adapter lock; listeners run on the thread
that changed the data unless ``post`` is set (the chat feature sets it to
``UiDispatcher.post`` so the drawer always refreshes on the UI loop). Blocking
file I/O (load, save, delete, body reads) belongs on a worker thread. The shared
ChatStore has a lock of its own (held through its long work: ``finish_run``, Migrate);
it is always taken before the adapter lock, never while holding it, and long file moves
(Migrate, delete workspace, Save scratch) run without the adapter lock, so the UI loop
never waits behind them.

Pure Python (3.10); never imports Flet. Backend modules load lazily.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from glossarion_mobile.state.chat_index import ChatSummary, group_recents
from glossarion_mobile.state.config_store import DebouncedSaver
from glossarion_mobile.state.prefs import atomic_write_json

__all__ = [
    "ChatStoreAdapter",
    "ChatStoreBinding",
    "MobileChatSidecar",
    "SCRATCH_DIR_NAME",
    "SIDECAR_NAME",
    "STORAGE_PATH_KEYS",
    "ScratchChat",
    "StoreUnavailable",
    "default_history_path",
    "is_scratch_cid",
    "message_fingerprints",
    "message_id",
    "remap_sidecar_entry",
    "summary_for_scratch",
]

log = logging.getLogger("glossarion.chats")

SIDECAR_NAME = "direct_text_chats.mobile.json"
SIDECAR_VERSION = 1
DRAFT_SAVE_DELAY = 0.45  # desktop _chat_history_save_timer
BODY_CACHE_LIMIT = 128  # desktop _message_text_cache
NEW_CHAT_TITLE = "New chat"
SCRATCH_DIR_NAME = "Direct Text Scratch"  # UI_SPEC §2.16: cache/Direct Text Scratch/<uuid>/
#: v2 storage keys holding history-relative file references (desktop normalizer limits, §2.19).
STORAGE_PATH_KEYS = (
    "content_path", "content_text_path", "content_html_path", "content_xhtml_path",
    "thinking_path", "image_path", "media_path",
)


def is_scratch_cid(cid: Any) -> bool:
    """Scratch chat ids are ``s<uuid hex>`` (route-safe, never a v2 int id)."""
    text = str(cid or "")
    return len(text) > 1 and text[0] == "s" and all(c in "0123456789abcdefABCDEF-" for c in text[1:])


class StoreUnavailable(RuntimeError):
    """The shared direct_text_store is missing or lacks an operation."""


def default_history_path() -> str:
    """``GLOSSARION_DIRECT_TEXT_HISTORY`` or ``direct_text_chats.json`` beside ``CONFIG_FILE``."""
    override = os.environ.get("GLOSSARION_DIRECT_TEXT_HISTORY")
    if override:
        return os.path.abspath(os.path.expanduser(str(override)))
    config_file = os.environ.get("CONFIG_FILE", "").strip()
    if not config_file:
        try:
            import app_paths

            config_file = app_paths.config_file_path()
        except Exception:
            config_file = os.path.join(os.getcwd(), "config.json")
    return os.path.join(os.path.dirname(os.path.abspath(config_file)), "direct_text_chats.json")


# ---------------------------------------------------------------------------
# Fingerprints (Appendix B)
# ---------------------------------------------------------------------------


def _sha12(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8", "surrogatepass")).hexdigest()[:12]


def message_fingerprints(messages: Iterable[Any]) -> list:
    """Stable per-message fingerprints: ``a:<created_at>:<basename(content_path)>``,
    ``u:<sha1(text)[:12]>:<k>``, ``f:<sha1(path)[:12]>:<k>`` (k = occurrence)."""
    seen: dict = {}
    out = []
    for message in messages:
        role = str(message[0]) if isinstance(message, (list, tuple)) and message else ""
        if role == "assistant":
            storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
            created = str(storage.get("created_at", "") or "")
            name = os.path.basename(str(storage.get("content_path", "") or "").replace("\\", "/"))
            base = f"a:{created}:{name}"
        elif role == "user_file":
            base = f"f:{_sha12(message[2] if len(message) > 2 else '')}"
        elif role == "user":
            base = f"u:{_sha12(message[1] if len(message) > 1 else '')}"
        else:
            base = f"x:{role}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        if role == "assistant":
            out.append(base if k == 0 else f"{base}:{k}")
        else:
            out.append(f"{base}:{k}")
    return out


def message_id(fingerprint: str) -> str:
    """Opaque 12-hex id for routes (fingerprints contain file names, ids do not)."""
    return _sha12(fingerprint)


# ---------------------------------------------------------------------------
# Binding to direct_text_store.ChatStore
# ---------------------------------------------------------------------------


class ChatStoreBinding:
    """Adapter over the shared ``direct_text_store.ChatStore`` (desktop method names)."""

    def __init__(self, store: Any = None, *, history_path: Optional[str] = None, output_root: Optional[str] = None) -> None:
        self.history_path = history_path or default_history_path()
        self.output_root = output_root
        self.store = store if store is not None else self._make_store()

    def _make_store(self) -> Any:
        try:
            import direct_text_store
        except Exception as exc:  # not extracted yet / not bundled
            raise StoreUnavailable(f"direct_text_store is unavailable: {exc}") from exc
        cls = getattr(direct_text_store, "ChatStore", None)
        if cls is None:
            raise StoreUnavailable("direct_text_store has no ChatStore")
        attempts = (
            {"history_path": self.history_path, "output_root": self.output_root},
            {"history_path": self.history_path},
            {"path": self.history_path},
        )
        for kwargs in attempts:
            try:
                return cls(**{k: v for k, v in kwargs.items() if v is not None})
            except TypeError:
                continue
        return cls(self.history_path)

    # ---- resolution -----------------------------------------------------------------

    def fn(self, *names: str, required: bool = True) -> Optional[Callable[..., Any]]:
        for name in names:
            for candidate in (name.lstrip("_"), "_" + name.lstrip("_")):
                fn = getattr(self.store, candidate, None)
                if callable(fn):
                    return fn
        if required:
            raise StoreUnavailable(f"ChatStore lacks {names[0]}")
        return None

    def _attr(self, *names: str, default: Any = None) -> Any:
        for name in names:
            for candidate in (name.lstrip("_"), "_" + name.lstrip("_")):
                if hasattr(self.store, candidate):
                    value = getattr(self.store, candidate)
                    if not callable(value):
                        return value
        return default

    # ---- history ------------------------------------------------------------------------

    def load(self) -> tuple:
        """``(sessions, current_chat_id)`` as the desktop ``_load_chat_history`` returns."""
        result = self.fn("load_chat_history", "load")()
        if isinstance(result, tuple) and len(result) == 2:
            sessions, current = result
        else:
            sessions = self._attr("chat_sessions", "sessions", default=[])
            current = self._attr("current_chat_id", default=None)
        sessions = list(sessions or [])
        try:
            current = int(current) if current is not None else None
        except (TypeError, ValueError):
            current = None
        return sessions, current

    def save(self, sessions: list, current_chat_id: Optional[int]) -> None:
        """Externalise bodies and write the v2 file atomically (``_save_chat_history``)."""
        save = self.fn("save_chat_history", "save")
        try:
            save(sessions, current_chat_id)
            return
        except TypeError:
            pass
        for name in ("_chat_sessions", "chat_sessions", "sessions"):
            if hasattr(self.store, name):
                setattr(self.store, name, sessions)
        for name in ("_current_chat_id", "current_chat_id"):
            if hasattr(self.store, name):
                setattr(self.store, name, current_chat_id)
        if hasattr(self.store, "_current_chat_index"):
            index = next((i for i, s in enumerate(sessions) if s.get("id") == current_chat_id), 0)
            self.store._current_chat_index = index
        save()

    def new_session(self, session_id: int) -> dict:
        return self.fn("new_chat_session")(session_id)

    # ---- bodies, folders --------------------------------------------------------------------

    def _wants_session(self, fn: Callable[..., Any]) -> bool:
        """True when a ChatStore method takes the session explicitly (else it works on the current chat)."""
        import inspect

        try:
            params = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
        except (TypeError, ValueError):
            return False
        return bool(params) and params[0].name in ("session", "chat_session", "chat")

    def _make_current(self, sessions: list, session: dict) -> None:
        """Point a dialog-style store at ``session`` (methods that use the current chat)."""
        for name in ("_chat_sessions", "chat_sessions", "sessions"):
            if hasattr(self.store, name):
                setattr(self.store, name, sessions)
        if hasattr(self.store, "_current_chat_index"):
            self.store._current_chat_index = next((i for i, s in enumerate(sessions) if s is session), 0)
        if hasattr(self.store, "_chat_messages") or not hasattr(self.store, "chat_messages"):
            session["messages"] = list(session.get("messages") or [])
            self.store._chat_messages = session["messages"]

    def save_response_edit(self, sessions: list, session: dict, message_index: int, source: str) -> Any:
        """``_save_response_output_edit``: managed response files + the real chapter file."""
        fn = self.fn("save_response_output_edit", "save_response_edit")
        if self._wants_session(fn):
            return fn(session, int(message_index), source)
        self._make_current(sessions, session)
        result = fn(int(message_index), source)
        messages = getattr(self.store, "_chat_messages", None)
        if isinstance(messages, list):
            session["messages"] = messages
        return result

    def resolve_reference(self, reference: str) -> str:
        fn = self.fn("resolve_history_file_reference", required=False)
        if fn is not None:
            return fn(reference)
        value = str(reference or "").strip()
        if not value:
            return ""
        value = value.replace("/", os.sep).replace("\\", os.sep)
        if os.path.isabs(value):
            return os.path.abspath(value)
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(self.history_path)), value))

    def history_reference(self, path: str) -> str:
        """The v2 reference of a file (``_history_file_reference``: relative to the history file)."""
        fn = self.fn("history_file_reference", required=False)
        if fn is not None:
            return fn(path)
        absolute = os.path.abspath(os.path.expanduser(str(path or "")))
        try:
            reference = os.path.relpath(absolute, os.path.dirname(os.path.abspath(self.history_path)))
        except ValueError:
            reference = absolute
        return reference.replace("\\", "/")

    @property
    def lock(self) -> Any:
        """The ChatStore's own lock (its operations hold it), or None for a store without one."""
        lock = getattr(self.store, "lock", None)
        return lock if lock is not None and hasattr(lock, "__enter__") else None

    def ensure_output_folder(self, session: dict) -> str:
        return str(self.fn("ensure_conversation_output_folder_for_session", "ensure_output_folder")(session) or "")

    def validated_output_folder(self, session: dict) -> str:
        return self.fn("validated_chat_output_folder")(session)

    def attachment_folders(self, session: dict) -> list:
        fn = self.fn("conversation_attachment_folders", "attachment_folders", required=False)
        if fn is None:
            return []
        try:
            return list(fn(session) or [])
        except Exception:
            log.debug("attachment folder scan failed", exc_info=True)
            return []

    # ---- U7: media, relocation, migrate -------------------------------------------------------

    def generated_media(self, message: Any, content: str) -> tuple:
        """``(primary, references)`` of one response: the desktop ``_assistant_generated_media``
        (storage ``media_path`` / ``image_path``, else the first existing marker) and every
        ``[GENERATED_IMAGE|VIDEO|AUDIO:<path>]`` marker (``_generated_media_references_from_text``)."""
        primary_fn = self.fn("assistant_generated_media", required=False)
        refs_fn = self.fn("generated_media_references_from_text", required=False)
        primary: tuple = ("", "")
        if primary_fn is not None:
            try:
                primary = tuple(primary_fn(message, content) or ("", ""))
            except Exception:
                log.debug("generated media lookup failed", exc_info=True)
        references: list = []
        if refs_fn is not None:
            try:
                references = [tuple(item) for item in (refs_fn(content) or [])]
            except Exception:
                log.debug("generated media markers failed", exc_info=True)
        return primary, references

    def relocate(self, session: dict, source_folder: str, target_folder: str) -> None:
        """The shared ``_relocate_session_attachment_paths`` (response links follow a moved tree)."""
        self.fn("relocate_session_attachment_paths")(session, source_folder, target_folder)

    def migrate_attachment(self, session: dict, source_folder: str, confirm_merge: Any = None) -> dict:
        """Desktop Migrate (``ChatStore.migrate_attachment``): ``{"ok", "notices"}``."""
        fn = self.fn("migrate_attachment")
        return dict(fn(session, source_folder, confirm_merge=confirm_merge) or {})

    def is_managed_attachment_workspace(self, session: dict, folder: str) -> bool:
        fn = self.fn("is_managed_attachment_workspace", required=False)
        return bool(fn(session, folder)) if fn is not None else False

    def migration_output_root(self, session: dict) -> str:
        """Where Migrate moves a workspace (the desktop ``_direct_text_migration_output_root``)."""
        fn = self.fn("migration_output_root", required=False)
        return str(fn(session) or "") if fn is not None else ""


# ---------------------------------------------------------------------------
# Sidecar: direct_text_chats.mobile.json
# ---------------------------------------------------------------------------

#: Per-chat override keys (Appendix B ``overrides``); None / missing = inherited.
OVERRIDE_KEYS = (
    "model", "profile", "target_language", "output_mode", "attachment_prompt_role",
    "glossary_override_mode", "manual_glossary_path", "force_multipass_off", "disable_thinking",
    "skip_prompt_profile", "disable_auto_scroll", "rendered_card_limit",
)


class MobileChatSidecar:
    """Mobile-only per-chat data; never read by desktop."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: dict = {"version": SIDECAR_VERSION, "chats": {}, "scratch": []}

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning("ignoring unreadable %s: %s", os.path.basename(self.path), exc)
            return
        if isinstance(payload, dict) and isinstance(payload.get("chats"), dict):
            self.data = {
                "version": SIDECAR_VERSION,
                "chats": {str(k): dict(v) for k, v in payload["chats"].items() if isinstance(v, dict)},
                "scratch": list(payload.get("scratch") or []),
            }

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def chat(self, cid: Any, create: bool = False) -> dict:
        chats = self.data["chats"]
        key = str(cid)
        if key not in chats:
            if not create:
                return {}
            chats[key] = {}
        return chats[key]

    def drop_orphans(self, live_ids: Iterable[Any], fingerprints: Optional[dict] = None) -> bool:
        """Remove chats that no longer exist and version/add_only entries whose fingerprint vanished."""
        live = {str(i) for i in live_ids}
        changed = False
        for key in list(self.data["chats"]):
            if key not in live:
                del self.data["chats"][key]
                changed = True
        for key, entry in self.data["chats"].items():
            known = set((fingerprints or {}).get(key, ()))
            if fingerprints is None:
                continue
            versions = entry.get("versions")
            if isinstance(versions, dict):
                for anchor in list(versions):
                    if anchor not in known:
                        del versions[anchor]
                        changed = True
            add_only = entry.get("add_only")
            if isinstance(add_only, list):
                kept = [fp for fp in add_only if fp in known]
                if kept != add_only:
                    entry["add_only"] = kept
                    changed = True
        return changed


def remap_sidecar_entry(entry: dict, fp_map: dict, index_map: dict) -> bool:
    """Rewrite one chat's sidecar data after messages were removed (Delete message, §2.19).

    ``fp_map``: old fingerprint -> new fingerprint (removed messages are absent); ``index_map``:
    old message index -> new index. Versions keep their surviving members (a group needs two;
    a removed anchor is replaced by its first surviving member), ``add_only`` its surviving
    fingerprints, a pending Plan its user turn (dropped when that turn was removed) and
    ``job_turns`` (run key -> the user turn's fingerprint) follows its turns (None once a
    turn is removed).
    """
    changed = False
    job_turns = entry.get("job_turns")
    if isinstance(job_turns, dict):
        for key, fp in list(job_turns.items()):
            moved = fp_map.get(fp) if fp else None
            if moved != fp:
                job_turns[key] = moved  # None: the job's turn was deleted (its card is gone)
                changed = True
    versions = entry.get("versions")
    if isinstance(versions, dict):
        rebuilt: dict = {}
        for anchor, group in versions.items():
            members = [fp_map[fp] for fp in (group or {}).get("members") or () if fp in fp_map]
            if len(members) < 2:
                changed = True
                continue
            old_members = list((group or {}).get("members") or ())
            try:
                selected_fp = old_members[int((group or {}).get("selected", len(old_members) - 1))]
            except (IndexError, TypeError, ValueError):
                selected_fp = None
            new_anchor = fp_map.get(anchor) or members[0]
            selected = members.index(fp_map[selected_fp]) if selected_fp in fp_map else len(members) - 1
            rebuilt[new_anchor] = {"members": members, "selected": selected}
            if new_anchor != anchor or members != old_members or selected != (group or {}).get("selected"):
                changed = True
        if rebuilt:
            entry["versions"] = rebuilt
        else:
            entry.pop("versions", None)
    add_only = entry.get("add_only")
    if isinstance(add_only, list):
        kept = [fp_map[fp] for fp in add_only if fp in fp_map]
        if kept != add_only:
            changed = True
        if kept:
            entry["add_only"] = kept
        else:
            entry.pop("add_only", None)
    plan = entry.get("pending_plan")
    if isinstance(plan, dict) and plan.get("user_index") is not None:
        try:
            old_index = int(plan["user_index"])
        except (TypeError, ValueError):
            old_index = -1
        if old_index in index_map:
            if index_map[old_index] != old_index:
                plan["user_index"] = index_map[old_index]
                changed = True
        else:
            entry.pop("pending_plan", None)
            changed = True
    return changed


@dataclass
class ScratchChat:
    """One unsaved scratch chat (UI_SPEC §2.16): its session, its own ChatStore and folder."""

    cid: str
    root: str  # <scratch dir>/<uuid>: the scratch store's history file and output root
    binding: Any  # ChatStoreBinding over a ChatStore whose history lives inside ``root``
    session: dict
    created: float = 0.0
    meta: dict = field(default_factory=dict)  # the sidecar entry of a v2 chat (never saved)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _parse_iso(value: Any) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    try:
        return created.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


class ChatStoreAdapter:
    def __init__(
        self,
        binding: Optional[ChatStoreBinding] = None,
        *,
        history_path: Optional[str] = None,
        sidecar_path: Optional[str] = None,
        clock: Callable[[], float] = time.time,
        save_delay: float = DRAFT_SAVE_DELAY,
        scratch_dir: Optional[str] = None,
    ) -> None:
        self._binding = binding
        self._history_path = history_path or (binding.history_path if binding is not None else None)
        self._sidecar_path = sidecar_path
        self._scratch_dir = scratch_dir
        self._scratch: "OrderedDict[str, ScratchChat]" = OrderedDict()
        self._clock = clock
        self._lock = threading.RLock()
        self._listeners: list[Callable[[], None]] = []
        self.post: Optional[Callable[..., Any]] = None  # e.g. UiDispatcher.post
        self.sessions: list[dict] = []
        self.current_id: Optional[int] = None
        self.sidecar: Optional[MobileChatSidecar] = None
        self.loaded = False
        self.load_error: Optional[str] = None
        self._bodies: "OrderedDict[tuple, str]" = OrderedDict()
        self._attachments: dict = {}
        self._running: set = set()
        self._saving: set = set()  # scratch chats whose Save is moving files
        self._saver = DebouncedSaver(self._save_now, delay=save_delay, name="gl-chats-save")
        self.saves = 0

    # ---- lifecycle --------------------------------------------------------------------------

    @property
    def binding(self) -> ChatStoreBinding:
        if self._binding is None:
            self._binding = ChatStoreBinding(history_path=self._history_path)
        return self._binding

    @property
    def available(self) -> bool:
        return self.loaded and self.load_error is None

    @property
    def history_path(self) -> str:
        return self._history_path or self.binding.history_path

    @property
    def sidecar_path(self) -> str:
        return self._sidecar_path or os.path.join(os.path.dirname(os.path.abspath(self.history_path)), SIDECAR_NAME)

    @property
    def scratch_dir(self) -> str:
        """``<cache>/Direct Text Scratch`` (the chat feature passes the app cache; default: beside the history)."""
        return os.path.abspath(self._scratch_dir or os.path.join(
            os.path.dirname(os.path.abspath(self.history_path)), "cache", SCRATCH_DIR_NAME))

    def load(self) -> bool:
        """Read history + sidecar (blocking). False (and ``load_error``) when the store is missing."""
        try:
            sessions, current = self.binding.load()
        except Exception as exc:
            with self._lock:
                self.load_error = str(exc)
                self.loaded = True
            log.warning("Direct Text history unavailable: %s", exc)
            self._notify()
            return False
        sidecar = MobileChatSidecar(self.sidecar_path)
        sidecar.load()
        with self._lock:
            self.sessions = sessions
            ids = [s.get("id") for s in sessions]
            self.current_id = current if current in ids else (ids[0] if ids else None)
            self.sidecar = sidecar
            self.load_error = None
            self.loaded = True
            fingerprints = {str(s.get("id")): message_fingerprints(s.get("messages") or []) for s in sessions}
            orphans = sidecar.drop_orphans(ids, fingerprints)
        for session in list(sessions):
            self._scan_attachments(session)
        if self._clear_stale_scratch() or orphans:
            self._save_sidecar()
        self._notify()
        return True

    def _clear_stale_scratch(self) -> bool:
        """Scratch chats are never saved: folders a previous launch left behind are removed."""
        sidecar = self.sidecar
        if sidecar is None:
            return False
        stale = list(sidecar.data.get("scratch") or [])
        live = {entry.root for entry in self._scratch.values()}
        kept = []
        for item in stale:
            root = str((item or {}).get("root") or "") if isinstance(item, dict) else ""
            if root in live:
                kept.append(item)
                continue
            self._remove_scratch_root(root)
        changed = kept != stale
        sidecar.data["scratch"] = kept
        return changed

    def _remove_scratch_root(self, root: str) -> bool:
        """rmtree one scratch folder, only when it is an immediate child of the scratch dir."""
        if not root:
            return False
        folder = os.path.realpath(os.path.abspath(root))
        parent = os.path.realpath(self.scratch_dir)
        if os.path.normcase(os.path.dirname(folder)) != os.path.normcase(parent) or not os.path.isdir(folder):
            return False
        try:
            shutil.rmtree(folder)
            return True
        except OSError:
            log.warning("could not remove the scratch folder %s", folder, exc_info=True)
            return False

    def flush(self) -> bool:
        """Synchronous save of pending changes (lifecycle hide, job start)."""
        if not self.available:
            return False
        self._saver.cancel()
        self._save_now()
        return True

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._saver.wait_idle(timeout)

    def close(self) -> None:
        if self._saver.pending:
            self.flush()
        self._saver.close()

    def _store_lock(self, cid: Any = None) -> Any:
        """The shared ChatStore's own lock for chat ``cid`` (default: the history's store).

        Lock order: the store lock is always taken BEFORE the adapter lock, never while the
        adapter lock is held. The ChatStore holds its lock through long work of its own
        (``finish_run`` persisting a run's output, the desktop Migrate moving a workspace), so a
        thread that waits for it must not park the adapter lock the UI loop reads under.
        """
        binding = self.binding_for(cid) if cid is not None else self._binding
        lock = getattr(binding, "lock", None) if binding is not None else None
        return lock if lock is not None else contextlib.nullcontext()

    def _save_now(self) -> None:
        with self._store_lock():
            with self._lock:
                if not self.available:
                    return
                try:
                    self.binding.save(self.sessions, self.current_id)
                    self.saves += 1
                except Exception:
                    log.exception("saving direct_text_chats.json failed")
        self._save_sidecar()

    def _save_sidecar(self) -> None:
        with self._lock:
            sidecar = self.sidecar
            if sidecar is None:
                return
            try:
                sidecar.save()
            except Exception:
                log.exception("saving %s failed", SIDECAR_NAME)

    def schedule_save(self) -> None:
        if self.available:
            self._saver.schedule()

    # ---- listeners --------------------------------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _notify(self) -> None:
        post = self.post
        for callback in list(self._listeners):
            try:
                if post is not None:
                    post(callback)
                else:
                    callback()
            except Exception:
                log.exception("chat store listener failed")

    def _changed(self, cid: Any = None, *, save: bool = True, touch: bool = True) -> None:
        if is_scratch_cid(cid):
            self._notify()  # scratch chats are never written to the history or the sidecar
            return
        if touch and cid is not None and self.sidecar is not None:
            with self._lock:
                self.sidecar.chat(cid, create=True)["updated_at"] = self._clock()
        if save:
            self.schedule_save()
        self._notify()

    # ---- session lookup ----------------------------------------------------------------------

    @staticmethod
    def _key(cid: Any) -> Optional[int]:
        try:
            return int(str(cid))
        except (TypeError, ValueError):
            return None

    def session(self, cid: Any = None) -> Optional[dict]:
        if is_scratch_cid(cid):
            with self._lock:
                entry = self._scratch.get(str(cid))
                return entry.session if entry is not None else None
        key = self._key(cid if cid is not None else self.current_id)
        with self._lock:
            for session in self.sessions:
                if session.get("id") == key:
                    return session
        return None

    def is_scratch(self, cid: Any) -> bool:
        with self._lock:
            return str(cid) in self._scratch

    def scratch_ids(self) -> list:
        with self._lock:
            return list(self._scratch)

    def binding_for(self, cid: Any) -> ChatStoreBinding:
        """The ChatStore binding that owns chat ``cid`` (a scratch chat has its own store)."""
        if is_scratch_cid(cid):
            with self._lock:
                entry = self._scratch.get(str(cid))
            if entry is not None:
                return entry.binding
        return self.binding

    def current_cid(self) -> str:
        with self._lock:
            return str(self.current_id) if self.current_id is not None else "1"

    def messages(self, cid: Any = None) -> list:
        session = self.session(cid)
        with self._lock:
            return list(session.get("messages") or []) if session else []

    def message(self, cid: Any, index: int) -> Optional[tuple]:
        messages = self.messages(cid)
        return messages[index] if 0 <= index < len(messages) else None

    def select(self, cid: Any) -> bool:
        key = self._key(cid)
        with self._lock:
            if key is None or not any(s.get("id") == key for s in self.sessions) or key == self.current_id:
                return False
            self.current_id = key
        self._changed(save=True, touch=False)
        return True

    # ---- drawer index interface (InMemoryChatIndex) ----------------------------------------

    def _updated_at(self, session: dict) -> float:
        meta = self.sidecar.chat(session.get("id")) if self.sidecar is not None else {}
        try:
            value = float(meta.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value:
            return value
        for message in reversed(session.get("messages") or []):
            if isinstance(message, (list, tuple)) and message and message[0] == "assistant" and len(message) > 6:
                stamp = _parse_iso((message[6] or {}).get("created_at"))
                if stamp:
                    return stamp
        try:
            return os.path.getmtime(self.history_path)
        except OSError:
            return self._clock()

    def _summary(self, session: dict) -> ChatSummary:
        cid = str(session.get("id"))
        meta = self.sidecar.chat(cid) if self.sidecar is not None else {}
        return ChatSummary(
            cid=cid,
            title=str(session.get("title") or NEW_CHAT_TITLE),
            updated_at=self._updated_at(session),
            pinned=bool(meta.get("pinned")),
            pinned_at=float(meta.get("pinned_at") or 0.0),
            attachments=self._attachments.get(cid, 0),
            running=cid in self._running,
        )

    def _scratch_summary(self, entry: ScratchChat) -> ChatSummary:
        title = str(entry.session.get("title") or NEW_CHAT_TITLE)
        title = "Scratch chat" if title == NEW_CHAT_TITLE else f"{title} · scratch"
        summary = summary_for_scratch(entry.cid, title, clock=lambda: entry.created)
        return replace(summary, running=entry.cid in self._running, attachments=self._attachments.get(entry.cid, 0))

    def all(self) -> list:
        with self._lock:
            return [self._summary(s) for s in self.sessions] + [self._scratch_summary(e) for e in self._scratch.values()]

    def get(self, cid: Any) -> Optional[ChatSummary]:
        if is_scratch_cid(cid):
            with self._lock:
                entry = self._scratch.get(str(cid))
                return self._scratch_summary(entry) if entry is not None else None
        session = self.session(cid)
        if session is None:
            return None
        with self._lock:
            return self._summary(session)

    def pinned(self) -> list:
        rows = [c for c in self.all() if c.pinned]
        return sorted(rows, key=lambda c: c.pinned_at, reverse=True)

    def recents(self, now: Optional[float] = None) -> list:
        return group_recents(self.all(), self._clock() if now is None else now)

    def search(self, query: str) -> list:
        needle = str(query or "").strip().casefold()
        if not needle:
            return []
        out = []
        with self._lock:
            for session in self.sessions:
                haystack = [str(session.get("title") or "")]
                for message in session.get("messages") or []:
                    if isinstance(message, (list, tuple)) and len(message) > 1 and message[0] in ("user", "user_file"):
                        haystack.append(str(message[1] or ""))
                        if message[0] == "user_file" and len(message) > 4:
                            haystack.append(str(message[4] or ""))
                if any(needle in text.casefold() for text in haystack):
                    out.append(self._summary(session))
        return sorted(out, key=lambda c: c.updated_at, reverse=True)

    def set_pinned(self, cid: Any, pinned: bool) -> bool:
        if self.sidecar is None or self.session(cid) is None or is_scratch_cid(cid):
            return False
        with self._lock:
            meta = self.sidecar.chat(cid, create=True)
            if bool(meta.get("pinned")) == bool(pinned):
                return False
            meta["pinned"] = bool(pinned)
            meta["pinned_at"] = self._clock() if pinned else 0.0
        self._save_sidecar()
        self._notify()
        return True

    def upsert(self, chat: ChatSummary) -> None:
        """Drawer compatibility: only the title and pin state of an existing chat change."""
        session = self.session(chat.cid)
        if session is None:
            return
        with self._lock:
            session["title"] = chat.title or session.get("title")
        if self.get(chat.cid) and self.get(chat.cid).pinned != chat.pinned:
            self.set_pinned(chat.cid, chat.pinned)
        self._changed(chat.cid)

    def remove(self, cid: Any) -> bool:
        ok, _error = self.delete(cid)
        return ok

    def set_running(self, cid: Optional[Any], running: bool) -> None:
        if cid is None:
            return
        key = str(cid)
        with self._lock:
            changed = (key in self._running) != running
            if running:
                self._running.add(key)
            else:
                self._running.discard(key)
        if changed:
            self._notify()

    # ---- chat lifecycle (desktop rules) -----------------------------------------------------

    @staticmethod
    def _has_content(session: Optional[dict]) -> bool:
        return bool(
            session
            and (
                session.get("messages")
                or str(session.get("draft", "") or "").strip()
                or session.get("attachment")
            )
        )

    def new_chat(self) -> str:
        """_new_chat: a new session only when the current one has content."""
        with self._lock:
            current = self.session()
            if current is None or self._has_content(current):
                next_id = max([int(s.get("id") or 0) for s in self.sessions] + [0]) + 1
                session = self.binding.new_session(next_id)
                self.sessions.append(session)
                self.current_id = session["id"]
            cid = str(self.current_id)
        self._changed(cid)
        return cid

    def rename(self, cid: Any, title: str) -> bool:
        from glossarion_mobile.ui.chat.direct_text_rules import rename_title

        session = self.session(cid)
        value = rename_title(title)
        if session is None or not value:
            return False
        with self._lock:
            session["title"] = value
        self._changed(cid)
        return True

    def can_delete(self, cid: Any) -> bool:
        with self._lock:
            return len(self.sessions) > 1 or self._has_content(self.session(cid))

    def delete(self, cid: Any) -> tuple:
        """_delete_current_chat after its confirmation: validated rmtree, then drop the session.

        Returns ``(ok, error message)``. Blocking (rmtree); call it off the UI loop.
        A scratch chat is discarded (its folder under the scratch dir is removed).
        """
        if is_scratch_cid(cid):
            return (True, "") if self.discard_scratch(cid) else (False, "This chat no longer exists.")
        session = self.session(cid)
        if session is None:
            return False, "This chat no longer exists."
        # One critical section: a debounced save must not re-create the folder (externalising
        # this chat's bodies) between the rmtree and the removal of the session.
        with self._lock:
            output_folder = str(session.get("output_folder", "") or "")
            if output_folder and os.path.exists(output_folder):
                try:
                    import shutil

                    safe_folder = self.binding.validated_output_folder(session)
                    shutil.rmtree(safe_folder)
                except Exception as exc:
                    return False, (
                        "The chat was not deleted because its output folder could not "
                        f"be safely removed.\n\n{exc}"
                    )
            index = self.sessions.index(session)
            self.sessions.pop(index)
            if not self.sessions:
                next_id = int(session.get("id") or 0) + 1
                self.sessions.append(self.binding.new_session(next_id))
            if self.current_id == session.get("id"):
                self.current_id = self.sessions[min(index, len(self.sessions) - 1)]["id"]
            if self.sidecar is not None:
                self.sidecar.data["chats"].pop(str(session.get("id")), None)
            self._attachments.pop(str(session.get("id")), None)
        self._changed()
        return True, ""

    def delete_notice(self, cid: Any) -> tuple:
        """(title, body) of the desktop "Delete chat?" confirmation (scratch: "Discard scratch chat?")."""
        if is_scratch_cid(cid):
            return "Discard scratch chat?", ("This scratch chat was never saved. Discarding it removes its "
                                             "messages and output files.")
        session = self.session(cid) or {}
        title = str(session.get("title", NEW_CHAT_TITLE) or NEW_CHAT_TITLE)
        output_folder = str(session.get("output_folder", "") or "")
        folder_notice = (
            f"\n\nOutput folder that will also be permanently deleted:\n{output_folder}"
            if output_folder
            else "\n\nNo output folder has been created for this chat yet."
        )
        return "Delete chat?", (
            f'Delete "{title}"?\n\n'
            "This permanently removes the conversation, its messages, and every "
            "saved output file for this chat. This cannot be undone."
            f"{folder_notice}"
        )

    # ---- composer state ----------------------------------------------------------------------

    def draft(self, cid: Any = None) -> str:
        session = self.session(cid)
        return str(session.get("draft", "") or "") if session else ""

    def set_draft(self, cid: Any, text: str) -> None:
        """Draft autosave (the 450 ms debounce is the save, like the desktop timer)."""
        session = self.session(cid)
        if session is None:
            return
        with self._lock:
            if session.get("draft", "") == text:
                return
            session["draft"] = text
        self.schedule_save()

    def attachment(self, cid: Any = None) -> Optional[dict]:
        session = self.session(cid)
        record = session.get("attachment") if session else None
        return dict(record) if isinstance(record, dict) else None

    def set_attachment(self, cid: Any, record: Optional[dict]) -> None:
        session = self.session(cid)
        if session is None:
            return
        with self._lock:
            session["attachment"] = dict(record) if record else None
        self.schedule_save()

    # ---- messages ---------------------------------------------------------------------------

    def record_user_turn(self, cid: Any, message: tuple, display_input: str) -> int:
        """Append the user turn, auto-title, clear draft + attachment (``_start_translation``)."""
        from glossarion_mobile.ui.chat.direct_text_rules import auto_title

        session = self.session(cid)
        if session is None:
            raise KeyError(f"no chat {cid}")
        with self._lock:
            title = auto_title(session.get("title"), display_input)
            if title:
                session["title"] = title
            messages = list(session.get("messages") or [])
            messages.append(tuple(message))
            session["messages"] = messages
            session["draft"] = ""
            session["attachment"] = None
            index = len(messages) - 1
        self._changed(cid)
        return index

    def append_messages(self, cid: Any, messages: Iterable[tuple], *, save: bool = True) -> list:
        session = self.session(cid)
        if session is None:
            return []
        with self._lock:
            current = list(session.get("messages") or [])
            start = len(current)
            current.extend(tuple(m) for m in messages)
            session["messages"] = current
            indices = list(range(start, len(current)))
        self._changed(cid, save=save)
        return indices

    def truncate_messages(self, cid: Any, length: int) -> bool:
        """Drop messages from ``length`` on (a cancelled Plan's unsent turn; mobile-only, stays v2-valid).

        The sidecar forgets the dropped messages like Delete message does (``remap_sidecar_entry``):
        a cancelled Run again / Retranslate turn leaves its version group, so a later send of the
        same file (same fingerprint) is not grouped with the turn it was meant to re-run."""
        session = self.session(cid)
        if session is None:
            return False
        with self._lock:
            current = list(session.get("messages") or [])
            if length >= len(current):
                return False
            kept = max(0, int(length))
            old_fps = message_fingerprints(current)
            session["messages"] = current[:kept]
            session["expanded"] = {i for i in (session.get("expanded") or ()) if i < length}
            self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[0] != session.get("id"))
            entry = self._meta_entry(cid)
            remapped = entry is not None and remap_sidecar_entry(
                entry, {fp: fp for fp in old_fps[:kept]}, {i: i for i in range(kept)})
        if remapped:
            self._meta_saved(cid)
        self._changed(cid)
        return True

    def replace_message(self, cid: Any, index: int, message: tuple) -> bool:
        session = self.session(cid)
        if session is None:
            return False
        with self._lock:
            current = list(session.get("messages") or [])
            if not 0 <= index < len(current):
                return False
            current[index] = tuple(message)
            session["messages"] = current
            self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[1] != index or k[0] != session.get("id"))
        self._changed(cid)
        return True

    def request_count(self, cid: Any = None) -> int:
        """Assistant cards whose label carries "Request N" (desktop ``_next_conversation_request_number`` - 1)."""
        import re

        count = 0
        for message in self.messages(cid):
            if (
                isinstance(message, (list, tuple))
                and message
                and message[0] == "assistant"
                and len(message) > 5
                and re.search(r"\bRequest\s+\d+\b", str(message[5] or ""), flags=re.IGNORECASE)
            ):
                count += 1
        return count

    def expanded(self, cid: Any = None) -> set:
        session = self.session(cid)
        return set(session.get("expanded") or ()) if session else set()

    def set_expanded(self, cid: Any, index: int, expanded: bool) -> None:
        session = self.session(cid)
        if session is None:
            return
        with self._lock:
            values = set(session.get("expanded") or ())
            if expanded:
                values.add(int(index))
            else:
                values.discard(int(index))
            session["expanded"] = values
        self.schedule_save()

    def message_text(self, cid: Any, index: int, kind: str = "content") -> str:
        """Lazy body (inline text or the ``Chat Messages/`` file), cached like desktop (128)."""
        session = self.session(cid)
        if session is None:
            return ""
        messages = session.get("messages") or []
        if not 0 <= index < len(messages):
            return ""
        message = messages[index]
        inline_index = 2 if kind == "thinking" else 1
        inline = str(message[inline_index] if len(message) > inline_index else "")
        if inline:
            return inline
        storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
        reference = str(storage.get(f"{kind}_path", "") or "")
        if not reference:
            return ""
        key = (session.get("id"), int(index), kind, reference)
        with self._lock:
            cached = self._bodies.get(key)
            if cached is not None:
                self._bodies.move_to_end(key)
                return cached
        # desktop _assistant_message_text: v2 reference (relative to the history file) -> UTF-8 text
        path = self.binding_for(cid).resolve_reference(reference)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = handle.read()
        except Exception:
            label = "thinking log" if kind == "thinking" else "response"
            value = f"*The saved {label} file is missing or unreadable.*"
        with self._lock:
            self._bodies[key] = value
            while len(self._bodies) > BODY_CACHE_LIMIT:
                self._bodies.popitem(last=False)
        return value

    def save_response_edit(self, cid: Any, index: int, source: str) -> Any:
        """Blocking: write an edited response through the shared store, then save the history."""
        session = self.session(cid)
        if session is None:
            raise IndexError("Response message is no longer available")
        with self._store_lock(cid), self._lock:
            result = self.binding_for(cid).save_response_edit(self.sessions, session, index, source)
        self.forget_bodies(cid)
        self.schedule_save()
        self._notify()
        return result

    def forget_bodies(self, cid: Any = None) -> None:
        key = self._key(cid) if cid is not None else None
        with self._lock:
            if key is None:
                self._bodies.clear()
            else:
                self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[0] != key)

    # ---- folders ------------------------------------------------------------------------------

    def output_folder(self, cid: Any, create: bool = False) -> str:
        session = self.session(cid)
        if session is None:
            return ""
        if create:
            with self._store_lock(cid), self._lock:
                folder = self.binding_for(cid).ensure_output_folder(session)
            self.schedule_save()
            return folder
        return str(session.get("output_folder", "") or "")

    def _scan_attachments(self, session: dict) -> int:
        count = len(self.binding.attachment_folders(session)) if self._binding is not None or self.available else 0
        with self._lock:
            self._attachments[str(session.get("id"))] = count
        return count

    def attachment_folders(self, cid: Any) -> list:
        session = self.session(cid)
        if session is None:
            return []
        folders = self.binding_for(cid).attachment_folders(session)
        with self._lock:
            self._attachments[str(cid) if is_scratch_cid(cid) else str(session.get("id"))] = len(folders)
        return folders

    def attachment_count(self, cid: Any) -> int:
        with self._lock:
            return int(self._attachments.get(str(cid), 0))

    def refresh_attachments(self, cid: Any) -> None:
        self.attachment_folders(cid)
        self._notify()

    # ---- sidecar data --------------------------------------------------------------------------

    def _meta_entry(self, cid: Any, create: bool = False) -> Optional[dict]:
        """The sidecar entry of a v2 chat, or a scratch chat's in-memory entry (never saved)."""
        if is_scratch_cid(cid):
            entry = self._scratch.get(str(cid))
            return entry.meta if entry is not None else ({} if create else None)
        if self.sidecar is None:
            return None
        return self.sidecar.chat(cid, create=create)

    def _meta_saved(self, cid: Any) -> None:
        if not is_scratch_cid(cid):
            self._save_sidecar()

    def meta(self, cid: Any) -> dict:
        with self._lock:
            entry = self._meta_entry(cid)
            return copy.deepcopy(entry) if entry else {}

    def set_meta(self, cid: Any, key: str, value: Any) -> None:
        with self._lock:
            entry = self._meta_entry(cid, create=True)
            if entry is None:
                return
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        self._meta_saved(cid)
        self._notify()

    def overrides(self, cid: Any) -> dict:
        overrides = self.meta(cid).get("overrides")
        return dict(overrides) if isinstance(overrides, dict) else {}

    def set_override(self, cid: Any, key: str, value: Any) -> None:
        if key not in OVERRIDE_KEYS:
            raise KeyError(key)
        with self._lock:
            entry = self._meta_entry(cid, create=True)
            if entry is None:
                return
            overrides = dict(entry.get("overrides") or {})
            if value is None:
                overrides.pop(key, None)
            else:
                overrides[key] = value
            if overrides:
                entry["overrides"] = overrides
            else:
                entry.pop("overrides", None)
        self._meta_saved(cid)
        self._notify()

    def reset_overrides(self, cid: Any) -> None:
        self.set_meta(cid, "overrides", None)

    def fingerprints(self, cid: Any = None) -> list:
        return message_fingerprints(self.messages(cid))

    def index_for_mid(self, cid: Any, mid: str) -> Optional[int]:
        for index, fp in enumerate(self.fingerprints(cid)):
            if message_id(fp) == str(mid).lower():
                return index
        return None

    def mid_for_index(self, cid: Any, index: int) -> Optional[str]:
        fps = self.fingerprints(cid)
        return message_id(fps[index]) if 0 <= index < len(fps) else None

    # ---- U7: generated media of a response -----------------------------------------------------

    def message_media(self, cid: Any, index: int) -> list:
        """``[(kind, path, exists)]`` of one response's generated media (blocking: reads the body).

        The primary item is the desktop ``_assistant_generated_media`` (storage ``media_path`` /
        ``image_path``, else the first existing marker); every other ``[GENERATED_*:<path>]``
        marker follows (an image gallery), missing files included so the card can say so.
        """
        message = self.message(cid, index)
        if not message or str(message[0]) != "assistant":
            return []
        content = self.message_text(cid, index, "content")
        primary, references = self.binding_for(cid).generated_media(message, content)
        out: list = []
        seen: set = set()

        def add(kind: str, path: str) -> None:
            if not kind or not path:
                return
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                return
            seen.add(key)
            out.append((str(kind), os.path.abspath(path), os.path.isfile(path)))

        if primary and len(primary) == 2:
            add(primary[0], primary[1])
        for kind, path in references:
            if primary and primary[1] and kind != primary[0] and kind in ("video", "audio"):
                continue  # one player per response, like the desktop card
            add(kind, path)
        return out

    # ---- U7: delete message (mobile-only, stays v2-valid) ------------------------------------

    def delete_messages(self, cid: Any, indices: Iterable[int], *, jobs: Iterable[tuple] = ()) -> bool:
        """Blocking: remove messages (UI_SPEC §2.19); the history is saved before it returns.

        * The shared store names a response's managed body files after its message index
          (``Chat Messages/NNNNNN-response.{md,txt,html,xhtml}`` / ``-thinking.md``:
          ``_externalize_session_messages`` and ``_editable_response_paths``). The removed
          responses' own files are deleted and every surviving response's files are renamed to
          its new index (``_rehome_bodies``), so a later response or edit at a reused index can
          never write over a surviving one. Files outside the chat's ``Chat Messages`` folder
          (attachment outputs, a duplicated chat's originals) are left alone.
        * ``expanded`` is re-indexed and the sidecar remapped (``remap_sidecar_entry``).
        * ``jobs``: ``[(run key, submitted user index)]`` of the chat's known jobs (the chat's
          ``ChatRuns.chat_job_turns``). Each one's turn is recorded by fingerprint in the sidecar
          ``job_turns`` (read back by ``job_turns()``), so job cards stay on their turns.
        """
        session = self.session(cid)
        if session is None:
            return False
        binding = self.binding_for(cid)
        with self._store_lock(cid), self._lock:
            messages = list(session.get("messages") or [])
            drop = {int(i) for i in indices if 0 <= int(i) < len(messages)}
            if not drop:
                return False
            old_fps = message_fingerprints(messages)
            keep = [i for i in range(len(messages)) if i not in drop]
            kept = self._rehome_bodies(binding, session, messages, keep, drop)
            new_fps = message_fingerprints(kept)
            fp_map = {old_fps[old]: new_fps[new] for new, old in enumerate(keep)}
            index_map = {old: new for new, old in enumerate(keep)}
            session["messages"] = kept
            session["expanded"] = {index_map[i] for i in (session.get("expanded") or ()) if i in index_map}
            key = session.get("id")
            self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[0] != key)
            jobs = [(str(k), i) for k, i in jobs or () if k]
            entry = self._meta_entry(cid, create=bool(jobs))
            if entry is not None:
                turns = entry.get("job_turns") if isinstance(entry.get("job_turns"), dict) else {}
                for job_key, user_index in jobs:
                    if job_key not in turns:  # first delete since the job ran: its turn by fingerprint
                        try:
                            index = int(user_index)
                        except (TypeError, ValueError):
                            index = -1
                        turns[job_key] = old_fps[index] if 0 <= index < len(old_fps) else None
                if turns:
                    entry["job_turns"] = turns
                remap_sidecar_entry(entry, fp_map, index_map)
        self._meta_saved(cid)
        self._changed(cid)
        if not is_scratch_cid(cid):
            self.flush()  # the renamed body files and the v2 references they moved to land together
        return True

    #: Storage keys of a response's managed body files (``Chat Messages/NNNNNN-<suffix>``).
    _BODY_KEYS = ("content_path", "content_text_path", "content_html_path", "content_xhtml_path", "thinking_path")

    def _rehome_bodies(self, binding: ChatStoreBinding, session: dict, messages: list, keep: list, drop: set) -> list:
        """The surviving messages, each response's managed body files moved to its new index.

        Only files a response owns are touched: a storage reference resolving into this chat's
        ``Chat Messages`` folder under the response's own ``NNNNNN-`` prefix. Survivors move in
        ascending order (a new index is never above the old one, so the slot is free by then).
        Called with both locks held.
        """
        kept = [messages[i] for i in keep]
        folder = str(session.get("output_folder", "") or "")
        if not folder:
            return kept
        managed_dir = os.path.normcase(os.path.join(os.path.abspath(folder), "Chat Messages"))

        def own_files(message: Any, index: int) -> list:
            if not (isinstance(message, (list, tuple)) and message and str(message[0]) == "assistant"):
                return []
            storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
            prefix = f"{index + 1:06d}-"
            found = []
            for storage_key in self._BODY_KEYS:
                reference = str(storage.get(storage_key, "") or "")
                if not reference:
                    continue
                path = os.path.abspath(binding.resolve_reference(reference))
                if os.path.normcase(os.path.dirname(path)) == managed_dir and os.path.basename(path).startswith(prefix):
                    found.append((storage_key, path))
            return found

        surviving = {os.path.normcase(path) for i in keep for _k, path in own_files(messages[i], i)}
        for index in sorted(drop):
            for _key, path in own_files(messages[index], index):
                if os.path.normcase(path) in surviving:
                    continue
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    log.warning("could not remove the deleted response file %s", path, exc_info=True)
        rebuilt = []
        for new, old in enumerate(keep):
            message = messages[old]
            files = own_files(message, old) if new != old else []
            if not files:
                rebuilt.append(message)
                continue
            storage = dict(message[6])
            old_prefix, new_prefix = f"{old + 1:06d}-", f"{new + 1:06d}-"
            for storage_key, path in files:
                target = os.path.join(os.path.dirname(path), new_prefix + os.path.basename(path)[len(old_prefix):])
                if os.path.isfile(path):
                    os.replace(path, target)
                storage[storage_key] = binding.history_reference(target)
            values = list(message)
            values[6] = storage
            rebuilt.append(tuple(values))
        return rebuilt

    # ---- U7: job <-> turn (Delete message keeps job cards on their turns) ----------------------

    def job_turns(self, cid: Any) -> dict:
        """``{run key: user turn fingerprint or None}`` recorded by ``delete_messages``.

        A job without an entry still sits at its submitted ``params["user_index"]`` (no message
        before it was deleted since it ran); None: its turn was deleted. ``ChatRuns`` resolves a
        fingerprint to the turn's current index.
        """
        with self._lock:
            entry = self._meta_entry(cid)
            turns = (entry or {}).get("job_turns")
            return dict(turns) if isinstance(turns, dict) else {}

    # ---- U7: versions (edit-and-resend / retranslate variants, sidecar ``versions``) ---------

    def version_groups(self, cid: Any) -> dict:
        """``{anchor fp: {"members": [fp, ...], "selected": i}}`` (Appendix B ``versions``)."""
        versions = self.meta(cid).get("versions")
        return dict(versions) if isinstance(versions, dict) else {}

    def add_version(self, cid: Any, anchor_index: int, new_index: int) -> Optional[str]:
        """Record turn ``new_index`` as the newest version of turn ``anchor_index``; returns the anchor fp."""
        fps = self.fingerprints(cid)
        if not (0 <= anchor_index < len(fps) and 0 <= new_index < len(fps)) or anchor_index == new_index:
            return None
        anchor_fp, new_fp = fps[anchor_index], fps[new_index]
        with self._lock:
            entry = self._meta_entry(cid, create=True)
            if entry is None:
                return None
            versions = dict(entry.get("versions") or {})
            group_key = next((k for k, g in versions.items() if k == anchor_fp or anchor_fp in (g or {}).get("members", ())),
                             anchor_fp)
            group = dict(versions.get(group_key) or {"members": [anchor_fp]})
            members = [fp for fp in group.get("members") or [anchor_fp] if fp != new_fp] + [new_fp]
            versions[group_key] = {"members": members, "selected": len(members) - 1}
            entry["versions"] = versions
        self._meta_saved(cid)
        self._notify()
        return group_key

    def select_version(self, cid: Any, anchor: str, selected: int) -> bool:
        with self._lock:
            entry = self._meta_entry(cid)
            group = ((entry or {}).get("versions") or {}).get(anchor)
            if not group:
                return False
            members = list(group.get("members") or ())
            value = max(0, min(int(selected), len(members) - 1))
            if value == group.get("selected"):
                return False
            group["selected"] = value
        self._meta_saved(cid)
        self._notify()
        return True

    # ---- U7: attachments manager (Migrate, delete workspace) ---------------------------------

    def migrate_attachment(self, cid: Any, folder: str, confirm_merge: Optional[Callable[[str], bool]] = None) -> dict:
        """Blocking: the desktop Migrate of one ``Attachments/<stem>`` workspace (``{"ok", "notices"}``).

        The shared migrate moves (or, for "Merge and replace", copies) the workspace, relocates
        the stored paths and saves the history under the ChatStore's own lock, like
        ``finish_run``; the adapter lock is not held meanwhile, so the UI loop never waits for
        the file work (the chat refuses Migrate while one of its jobs runs).
        """
        session = self.session(cid)
        if session is None:
            return {"ok": False, "notices": [{"level": "warning", "title": "Attachment unavailable",
                                              "text": "This chat no longer exists."}]}
        result = self.binding_for(cid).migrate_attachment(session, folder, confirm_merge)
        with self._lock:
            self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[0] != session.get("id"))
        self.attachment_folders(cid)
        self._changed(cid)
        return result

    def migration_target(self, cid: Any, folder: str) -> str:
        """The folder Migrate would create (exists -> the desktop "Attachment folder already exists" choice)."""
        session = self.session(cid)
        if session is None:
            return ""
        root = self.binding_for(cid).migration_output_root(session)
        return os.path.abspath(os.path.join(root, os.path.basename(os.path.normpath(str(folder))))) if root else ""

    def delete_attachment_workspace(self, cid: Any, folder: str) -> tuple:
        """Blocking: remove one managed ``Attachments/<stem>`` workspace (``(ok, error)``)."""
        session = self.session(cid)
        if session is None:
            return False, "This chat no longer exists."
        binding = self.binding_for(cid)
        if not binding.is_managed_attachment_workspace(session, folder):
            return False, "This folder is no longer a managed attachment for the conversation."
        try:  # no lock held during the rmtree: the UI loop keeps reading the chat meanwhile
            shutil.rmtree(os.path.realpath(os.path.abspath(folder)))
        except OSError as exc:
            return False, f"The attachment workspace could not be deleted.\n\n{exc}"
        self.attachment_folders(cid)
        self._changed(cid)
        return True, ""

    # ---- U7: scratch chats (UI_SPEC §2.16) ------------------------------------------------------

    def _make_scratch(self, title: str = NEW_CHAT_TITLE) -> ScratchChat:
        token = uuid.uuid4().hex
        cid = "s" + token
        root = os.path.join(self.scratch_dir, token)
        os.makedirs(root, exist_ok=True)
        binding = ChatStoreBinding(history_path=os.path.join(root, "direct_text_chats.json"), output_root=root)
        session = binding.new_session(1)
        session["title"] = title or NEW_CHAT_TITLE
        store = binding.store
        try:  # the scratch store holds just this session (its own saves stay inside the scratch folder)
            store._chat_sessions = [session]
            store.set_current_session(session)
        except Exception:
            log.debug("scratch store set-up", exc_info=True)
        entry = ScratchChat(cid=cid, root=root, binding=binding, session=session, created=self._clock())
        return entry

    def _register_scratch(self, entry: ScratchChat) -> None:
        with self._lock:
            self._scratch[entry.cid] = entry
            if self.sidecar is not None:
                items = [i for i in self.sidecar.data.get("scratch") or [] if isinstance(i, dict)]
                items.append({"cid": entry.cid, "root": entry.root, "created": entry.created})
                self.sidecar.data["scratch"] = items
        self._save_sidecar()
        self._notify()

    def new_scratch(self) -> str:
        """Drawer / header "New scratch chat": an unsaved chat (never in ``direct_text_chats.json``)."""
        entry = self._make_scratch()
        self._register_scratch(entry)
        return entry.cid

    def duplicate_as_scratch(self, cid: Any) -> Optional[str]:
        """Blocking: drawer "Duplicate as scratch", a scratch copy of the chat's messages.

        The copy owns its response bodies: the files its responses keep in the original's
        ``Chat Messages`` folder are copied into the scratch chat's own ``Chat Messages`` under the
        same names (the store's index naming), because a later Delete message in the original
        renames or removes those files (``_rehome_bodies``) and an edit rewrites them. Other
        references (attachment outputs, generated media) keep pointing at the original files by
        absolute path: folders are not copied.
        """
        source = self.session(cid)
        if source is None:
            return None
        binding = self.binding_for(cid)
        with self._store_lock(cid):  # the original's bodies are not renamed / rewritten while they are copied
            with self._lock:
                messages = [self._absolute_message(binding, m) for m in source.get("messages") or []]
                title = str(source.get("title") or NEW_CHAT_TITLE)
                expanded = set(source.get("expanded") or ())
                folder = str(source.get("output_folder", "") or "")
            entry = self._make_scratch(title)
            messages = self._copy_managed_bodies(entry, messages, folder)
        entry.session["messages"] = messages
        entry.session["expanded"] = expanded
        self._register_scratch(entry)
        return entry.cid

    @staticmethod
    def _copy_managed_bodies(entry: "ScratchChat", messages: list, folder: str) -> list:
        """``messages`` (absolute references) with every file inside ``<folder>/Chat Messages`` copied
        into the scratch chat's own output folder and its reference pointed at the copy."""
        managed = os.path.normcase(os.path.join(os.path.abspath(folder), "Chat Messages")) if folder else ""
        if not managed:
            return messages
        copies_dir = ""
        out = []
        for message in messages:
            if not (message and str(message[0]) == "assistant" and len(message) > 6 and isinstance(message[6], dict)):
                out.append(message)
                continue
            storage = dict(message[6])
            changed = False
            for key in STORAGE_PATH_KEYS:
                path = str(storage.get(key, "") or "")
                if not path or os.path.normcase(os.path.dirname(os.path.abspath(path))) != managed:
                    continue
                if not os.path.isfile(path):
                    continue  # already missing: the copy shows the same "missing" placeholder
                if not copies_dir:
                    copies_dir = os.path.join(entry.binding.ensure_output_folder(entry.session), "Chat Messages")
                    os.makedirs(copies_dir, exist_ok=True)
                target = os.path.join(copies_dir, os.path.basename(path))
                shutil.copy2(path, target)
                storage[key] = target
                changed = True
            if changed:
                values = list(message)
                values[6] = storage
                out.append(tuple(values))
            else:
                out.append(message)
        return out

    @staticmethod
    def _absolute_message(binding: ChatStoreBinding, message: Any) -> tuple:
        """One message with its storage file references made absolute (store-independent)."""
        values = list(message) if isinstance(message, (list, tuple)) else [message]
        if values and str(values[0]) == "assistant" and len(values) > 6 and isinstance(values[6], dict):
            storage = dict(values[6])
            for key in STORAGE_PATH_KEYS:
                reference = str(storage.get(key, "") or "")
                if reference:
                    storage[key] = binding.resolve_reference(reference)
            values[6] = storage
        return tuple(values)

    def discard_scratch(self, cid: Any) -> bool:
        """Blocking: forget a scratch chat and remove its folder."""
        with self._lock:
            entry = self._scratch.pop(str(cid), None)
            if entry is None:
                return False
            if self.sidecar is not None:
                self.sidecar.data["scratch"] = [i for i in self.sidecar.data.get("scratch") or []
                                                if isinstance(i, dict) and i.get("cid") != entry.cid]
            self._running.discard(entry.cid)
        self._remove_scratch_root(entry.root)
        self._save_sidecar()
        self._notify()
        return True

    def save_scratch(self, cid: Any) -> Optional[str]:
        """Blocking: Save a scratch chat (UI_SPEC §2.16) -> the new v2 chat id.

        Assigns a v2 id, moves the scratch output folder into ``Direct Text/<safe title> - <ts>_<uuid8>/``
        (the shared ``_ensure_conversation_output_folder_for_session`` names it) and rewrites the
        stored paths with the shared ``_relocate_session_attachment_paths``; the scratch folder goes.
        The files move with no lock held (the UI loop keeps reading the chats meanwhile).
        """
        with self._store_lock(), self._lock:
            entry = self._scratch.get(str(cid))
            if entry is None or not self.available or entry.cid in self._saving:
                return None
            self._saving.add(entry.cid)
            session = entry.session
            messages = [self._absolute_message(entry.binding, m) for m in session.get("messages") or []]
            source_folder = str(session.get("output_folder", "") or "")
            next_id = max([int(s.get("id") or 0) for s in self.sessions] + [0]) + 1
            saved = {
                **{k: v for k, v in session.items() if k not in ("id", "output_folder", "output_folder_name")},
                "id": next_id,
                "messages": messages,
                "expanded": set(session.get("expanded") or ()),
                "output_folder": "",
                "output_folder_name": "",
            }
            self.sessions.append(saved)  # first: the shared folder helpers then see a known session
            target = ""
            if source_folder and os.path.isdir(source_folder):
                target = self.binding.ensure_output_folder(saved)
            else:
                saved["next_output_index"] = 1
        try:
            if target:
                for name in os.listdir(source_folder):
                    shutil.move(os.path.join(source_folder, name), os.path.join(target, name))
        finally:
            with self._store_lock(), self._lock:
                if target:
                    self.binding.relocate(saved, source_folder, target)
                self._saving.discard(entry.cid)
        with self._lock:
            self.current_id = next_id
            self._scratch.pop(entry.cid, None)
            if self.sidecar is not None:
                meta = copy.deepcopy(entry.meta)
                if meta:
                    self.sidecar.data["chats"][str(next_id)] = meta
                self.sidecar.data["scratch"] = [i for i in self.sidecar.data.get("scratch") or []
                                                if isinstance(i, dict) and i.get("cid") != entry.cid]
            self._running.discard(entry.cid)
        self._remove_scratch_root(entry.root)
        self._scan_attachments(saved)
        self._changed(str(next_id))
        self.flush()
        return str(next_id)

def summary_for_scratch(cid: str, title: str, clock: Callable[[], float] = time.time) -> ChatSummary:
    """A drawer row for an unsaved scratch chat (never written to the v2 file)."""
    return replace(ChatSummary(cid=cid, title=title or "Scratch", updated_at=clock()), scratch=True)
