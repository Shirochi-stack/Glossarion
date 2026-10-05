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
* message fingerprints / opaque ``mid`` ids for routes (Appendix B).

Thread-safety: every method takes the adapter lock; listeners run on the thread
that changed the data unless ``post`` is set (the chat feature sets it to
``UiDispatcher.post`` so the drawer always refreshes on the UI loop). Blocking
file I/O (load, save, delete, body reads) belongs on a worker thread.

Pure Python (3.10); never imports Flet. Backend modules load lazily.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from glossarion_mobile.state.chat_index import ChatSummary, group_recents
from glossarion_mobile.state.config_store import DebouncedSaver
from glossarion_mobile.state.prefs import atomic_write_json

__all__ = [
    "ChatStoreAdapter",
    "ChatStoreBinding",
    "MobileChatSidecar",
    "SIDECAR_NAME",
    "StoreUnavailable",
    "default_history_path",
    "message_fingerprints",
    "message_id",
]

log = logging.getLogger("glossarion.chats")

SIDECAR_NAME = "direct_text_chats.mobile.json"
SIDECAR_VERSION = 1
DRAFT_SAVE_DELAY = 0.45  # desktop _chat_history_save_timer
BODY_CACHE_LIMIT = 128  # desktop _message_text_cache
NEW_CHAT_TITLE = "New chat"


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
    ) -> None:
        self._binding = binding
        self._history_path = history_path or (binding.history_path if binding is not None else None)
        self._sidecar_path = sidecar_path
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
        if orphans:
            self._save_sidecar()
        self._notify()
        return True

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

    def _save_now(self) -> None:
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
        key = self._key(cid if cid is not None else self.current_id)
        with self._lock:
            for session in self.sessions:
                if session.get("id") == key:
                    return session
        return None

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

    def all(self) -> list:
        with self._lock:
            return [self._summary(s) for s in self.sessions]

    def get(self, cid: Any) -> Optional[ChatSummary]:
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
        if self.sidecar is None or self.session(cid) is None:
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
        """
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
        """(title, body) of the desktop "Delete chat?" confirmation."""
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
        """Drop messages from ``length`` on (a cancelled Plan's unsent turn; mobile-only, stays v2-valid)."""
        session = self.session(cid)
        if session is None:
            return False
        with self._lock:
            current = list(session.get("messages") or [])
            if length >= len(current):
                return False
            session["messages"] = current[: max(0, int(length))]
            session["expanded"] = {i for i in (session.get("expanded") or ()) if i < length}
            self._bodies = OrderedDict((k, v) for k, v in self._bodies.items() if k[0] != session.get("id"))
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
        path = self.binding.resolve_reference(reference)
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
        with self._lock:
            result = self.binding.save_response_edit(self.sessions, session, index, source)
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
            with self._lock:
                folder = self.binding.ensure_output_folder(session)
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
        folders = self.binding.attachment_folders(session)
        with self._lock:
            self._attachments[str(session.get("id"))] = len(folders)
        return folders

    def refresh_attachments(self, cid: Any) -> None:
        self.attachment_folders(cid)
        self._notify()

    # ---- sidecar data --------------------------------------------------------------------------

    def meta(self, cid: Any) -> dict:
        if self.sidecar is None:
            return {}
        with self._lock:
            return copy.deepcopy(self.sidecar.chat(cid))

    def set_meta(self, cid: Any, key: str, value: Any) -> None:
        if self.sidecar is None:
            return
        with self._lock:
            entry = self.sidecar.chat(cid, create=True)
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        self._save_sidecar()
        self._notify()

    def overrides(self, cid: Any) -> dict:
        overrides = self.meta(cid).get("overrides")
        return dict(overrides) if isinstance(overrides, dict) else {}

    def set_override(self, cid: Any, key: str, value: Any) -> None:
        if key not in OVERRIDE_KEYS:
            raise KeyError(key)
        if self.sidecar is None:
            return
        with self._lock:
            entry = self.sidecar.chat(cid, create=True)
            overrides = dict(entry.get("overrides") or {})
            if value is None:
                overrides.pop(key, None)
            else:
                overrides[key] = value
            if overrides:
                entry["overrides"] = overrides
            else:
                entry.pop("overrides", None)
        self._save_sidecar()
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


def summary_for_scratch(cid: str, title: str, clock: Callable[[], float] = time.time) -> ChatSummary:
    """A drawer row for an unsaved scratch chat (never written to the v2 file)."""
    return replace(ChatSummary(cid=cid, title=title or "Scratch", updated_at=clock()), scratch=True)
