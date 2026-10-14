"""ShareLinkService: opt-in "Share file via link" for Library books (U10; mobile only).

What it does, and what it never does:

* Every provider (``share_providers``: the transfer.it browser handoff, Gofile, Send end-to-end
  encrypted, pixeldrain with the user's key) is **off** until the user turns it on in Settings ›
  Cloud sync & sharing and accepts its consent sheet (``consent_text``: the file leaves the phone,
  who can read it, what the service sees, how long it lasts). A new consent text (version) asks again.
* Uploads happen only when the user taps (``upload``): never after a compile, never on launch,
  never retried by themselves (the Gofile 429 rule in ``share_providers.gofile`` is the one
  exception the critic allows). transfer.it is a browser handoff (``start_handoff`` +
  ``add_pasted_link``): Glossarion makes no request to transfer.it.
* Only book outputs qualify (``eligible``): EPUB/PDF/TXT/HTML/CBZ under the Library or Output
  folders - never a config or key export, nothing from the data folder (config, Inbox).
* Before an upload the file is copied to a private snapshot in the app cache (hashing it on the
  way), so a compile that rewrites the EPUB meanwhile cannot tear the upload; the snapshot is
  removed afterwards (and stale ones at the next start).
* One upload at a time, on a worker thread, with progress (``state`` / ``subscribe``), Cancel, and
  the Android foreground service (``native.ServiceLease`` "share-link", shared with a running job)
  or the iOS background task held meanwhile.

Storage (mobile-only; nothing in config.json): ``<data>/mobile_share_links.json`` holds the provider
switches and consents, the Send defaults and the per-book link records. Secrets are Fernet ``ENC:``
values made with the app's API-key encryption (``api_key_encryption``, whose key lives in the
Android Keystore / iOS Keychain): the Gofile guest token, the pixeldrain API key, every link URL
(a Send link carries its decryption key) and every delete handle. Nothing secret is written in
plain text or logged; a device without working encryption cannot use direct providers.

Records are keyed by the book's identity path (``library.book_identity``: its workspace) and the
uploaded file; ``relocate`` follows a chat book that auto-migrates into the Library, and
``forget_book`` / ``wipe`` drop records with the book / the app data.

Pure Python (3.10); no Flet import. Async methods run on the UI loop and do their blocking work
through ``run_io``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import secrets as _secrets
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.services.share_providers import (
    PROVIDER_IDS,
    CancelToken,
    ConsentText,
    ProviderInfo,
    ShareError,
    UploadResult,
    consent_text,
    format_size,
    provider_info,
)
from glossarion_mobile.state.prefs import atomic_write_json

__all__ = [
    "DeleteResult",
    "ELIGIBLE_EXTENSIONS",
    "LARGE_FILE_BYTES",
    "Preflight",
    "ProviderState",
    "SHARE_HOLD",
    "STORE_FILE",
    "SecretBox",
    "ShareLink",
    "ShareLinkService",
    "ShareLinkStore",
    "UploadState",
]

log = logging.getLogger("glossarion.share")

STORE_FILE = "mobile_share_links.json"
STORE_VERSION = 1
#: ``ServiceHolds`` name of an upload (the job runner's is "jobs", a sign-in's "sign-in").
SHARE_HOLD = "share-link"
#: Book outputs that may be shared by link (the Library's compiled kinds + manga CBZ).
ELIGIBLE_EXTENSIONS = (".epub", ".pdf", ".txt", ".html", ".htm", ".cbz")
#: Above this the preflight warns about mobile data.
LARGE_FILE_BYTES = 50 * 1024 * 1024
MAX_LINKS = 500
SNAPSHOT_DIR = "share_uploads"
#: Sidecar key: ``{normalised file path: ENC {"uri", "name"}}`` of the Downloads/Glossarion entry the last
#: transfer.it handoff of that file made (Android), overwritten in place by the next handoff.
HANDOFF_ENTRIES = "handoff_downloads"
_PROGRESS_INTERVAL = 1.0  # seconds between notification updates


def _mime(path: str) -> str:
    from glossarion_mobile.services.files import mime_type_for

    return mime_type_for(path)


def _norm(path: Any) -> str:
    text = os.fspath(path) if path else ""
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(os.path.abspath(text)))


def _under(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# encryption of secrets (the app's API-key encryption)
# ---------------------------------------------------------------------------


class SecretBox:
    """``ENC:`` Fernet values with the app's ``api_key_encryption`` handler (SecureStorage key).

    ``seal`` refuses (``ShareError('secrets_unavailable')``) instead of storing plain text when the
    handler cannot really encrypt (no ``cryptography``, or the mobile keys were not installed), and
    ``open`` refuses instead of returning a value it could not decrypt.
    """

    def __init__(self, *, ready: Optional[Callable[[], bool]] = None, handler: Any = None) -> None:
        self._ready = ready
        self._handler_override = handler
        self._ok: Optional[bool] = None  # the encrypt/decrypt probe, run once (off the UI loop at load)

    def _handler(self) -> Any:
        if self._handler_override is not None:
            return self._handler_override
        import api_key_encryption  # shared; its key comes from the Keystore / Keychain on a phone

        return api_key_encryption.get_handler()

    def available(self) -> bool:
        if self._ready is not None:
            try:
                if not self._ready():
                    return False
            except Exception:
                return False
        if self._ok:
            return True
        self._ok = self._probe()
        return self._ok

    def _probe(self) -> bool:
        try:
            handler = self._handler()
        except Exception:
            return False
        cipher = getattr(handler, "cipher", None)
        if cipher is None or type(cipher).__name__ != "Fernet":
            return False
        probe = "glossarion-share-probe"
        try:
            token = handler.encrypt_value(probe)
            return isinstance(token, str) and token.startswith("ENC:") and token != probe \
                and handler.decrypt_value(token) == probe
        except Exception:
            return False

    def seal(self, value: str) -> str:
        value = str(value)
        if not self.available():
            raise ShareError("secrets_unavailable", "Secure storage is not available on this phone, so Glossarion "
                             "cannot keep share links or keys safely.")
        token = self._handler().encrypt_value(value)
        if not isinstance(token, str) or not token.startswith("ENC:") or token == value:
            raise ShareError("secrets_unavailable", "Could not encrypt the value; nothing was saved.")
        return token

    def open(self, token: Any) -> str:
        text = str(token or "")
        if not text.startswith("ENC:"):
            raise ShareError("secrets_unavailable", "A stored share-link value is not encrypted; it was ignored.")
        if self._ready is not None and not self._ready():
            raise ShareError("secrets_unavailable", "Secure storage is not ready yet.")
        value = self._handler().decrypt_value(text)
        if not isinstance(value, str) or value == text:
            raise ShareError("secrets_unavailable", "A stored share-link value could not be decrypted (the app's "
                             "keys changed).")
        return value

    def seal_json(self, value: Any) -> str:
        return self.seal(json.dumps(value, separators=(",", ":"), sort_keys=True))

    def open_json(self, token: Any) -> Any:
        try:
            return json.loads(self.open(token))
        except ValueError:
            raise ShareError("secrets_unavailable", "A stored share-link value is damaged.") from None


# ---------------------------------------------------------------------------
# the sidecar
# ---------------------------------------------------------------------------


def _empty() -> dict:
    return {"version": STORE_VERSION, "providers": {}, "secrets": {}, "send": {}, "links": []}


class ShareLinkStore:
    """``<data>/mobile_share_links.json``: thread-safe, atomic write per change (blocking: io only)."""

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data = _empty()
        self.loaded = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        data = _empty()
        error = None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("not a JSON object")
            for key in ("providers", "secrets", "send"):
                if isinstance(raw.get(key), dict):
                    data[key] = raw[key]
            if isinstance(raw.get("links"), list):
                data["links"] = [r for r in raw["links"] if isinstance(r, dict) and r.get("id")]
            for key, value in raw.items():  # keys of newer builds are kept
                data.setdefault(key, value)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            error = f"{self.path.name} was unreadable ({type(exc).__name__}); starting fresh"
            log.warning(error)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            try:
                os.replace(self.path, self.path.with_name(f"{self.path.stem}.corrupt-{stamp}{self.path.suffix}"))
            except OSError:
                pass
        with self._lock:
            self._data = data
            self.loaded = True
            self.load_error = error

    def read(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def mutate(self, fn: Callable[[dict], Any]) -> Any:
        """Apply ``fn(data)`` and write the file (atomically) under the lock; returns ``fn``'s result.

        Loads the file first when nobody did (a blocking caller such as the chat's auto-migrate must not
        write an empty store over the saved links)."""
        with self._lock:
            if not self.loaded:
                self.load()
            data = copy.deepcopy(self._data)
            result = fn(data)
            data["version"] = STORE_VERSION
            atomic_write_json(self.path, data)
            self._data = data
            return result

    def remove_file(self) -> None:
        with self._lock:
            self._data = _empty()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# public values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderState:
    info: ProviderInfo
    enabled: bool
    consented: bool  # the current consent version was accepted
    has_key: bool = False  # pixeldrain: an API key is saved
    reason: Optional[str] = None  # why it cannot be used right now (None: ready)

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def ready(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class ShareLink:
    id: str
    provider: str
    url: str
    name: str
    size: int
    created: float
    book: str = ""  # the book's identity path
    source: str = ""  # the uploaded file
    expires: Optional[float] = None
    downloads_limit: Optional[int] = None
    can_delete: bool = False
    sha256: str = ""
    now: float = 0.0  # when this view was made (for ``expired``)

    @property
    def label(self) -> str:
        return provider_info(self.provider).label

    @property
    def e2ee(self) -> bool:
        return provider_info(self.provider).e2ee

    @property
    def expired(self) -> bool:
        return bool(self.expires) and (self.now or time.time()) >= float(self.expires)

    @property
    def summary(self) -> str:
        """``gofile.io link · 4.8 MB`` style line for cards (no URL)."""
        bits = [self.label, format_size(self.size)]
        if self.expired:
            bits.append("expired")
        elif self.expires:
            left = float(self.expires) - (self.now or time.time())
            bits.append(f"expires in {max(1, int(left // 3600))} h" if left < 48 * 3600
                        else f"expires in {int(left // 86400)} days")
        return " · ".join(b for b in bits if b)


@dataclass(frozen=True)
class UploadState:
    phase: str = "idle"  # idle | preparing | uploading | finishing | done | failed | cancelled
    provider: str = ""
    name: str = ""
    sent: int = 0
    total: int = 0
    message: str = ""
    link_id: Optional[str] = None
    error_code: Optional[str] = None
    book: str = ""

    @property
    def active(self) -> bool:
        return self.phase in ("preparing", "uploading", "finishing")

    @property
    def fraction(self) -> float:
        return min(1.0, self.sent / self.total) if self.total else 0.0

    @property
    def text(self) -> str:
        if self.phase == "uploading" and self.total:
            return f"{int(self.fraction * 100)}% · {format_size(self.sent)}/{format_size(self.total)}"
        return self.message


@dataclass(frozen=True)
class Preflight:
    provider: str
    name: str
    size: int
    mime: str
    warnings: tuple = ()
    blocked: Optional[str] = None  # ShareError code when it cannot be uploaded
    message: str = ""  # the reason for ``blocked``
    existing: Optional[ShareLink] = None  # a live link to the same file (same content) on this provider

    @property
    def ok(self) -> bool:
        return self.blocked is None


@dataclass(frozen=True)
class DeleteResult:
    removed: bool  # the record is gone from the app
    remote: str  # "deleted" | "gone" (the service no longer had it) | "kept" (not deletable from the app)


@dataclass
class _Snapshot:
    path: str  # private copy
    folder: str
    source: str
    name: str
    size: int
    sha256: str
    mime: str


# ---------------------------------------------------------------------------
# platform hold (Android foreground service / iOS background task)
# ---------------------------------------------------------------------------


class _UploadHold:
    """Keeps the process alive during an upload: Android shares the job foreground service through
    ``native.ServiceLease`` (the extracted OAuth sign-in logic), iOS asks for a background task."""

    def __init__(self, native: Any, platform: str) -> None:
        self.native = native
        self.platform = platform
        self.lease: Any = None
        self.task_id = -1

    async def acquire(self, title: str, text: str) -> bool:
        if self.native is None:
            return False
        if self.platform == "android":
            try:
                from glossarion_mobile.services.native import ServiceLease
            except ImportError:
                log.info("native.ServiceLease is missing; the upload runs without the foreground service")
                return False
            self.lease = ServiceLease(self.native, SHARE_HOLD)
            return bool(await self.lease.acquire(title, text))
        if self.platform == "ios":
            try:
                result = await self.native.call("begin_background_task", SHARE_HOLD, default=-1,
                                                expiration_title="Glossarion paused",
                                                expiration_body="The share-link upload stopped: open Glossarion to retry")
                self.task_id = int(result)
            except Exception as exc:
                log.info("begin_background_task failed: %s", exc)
                self.task_id = -1
            return self.task_id >= 0
        return False

    async def update(self, text: str, title: Optional[str] = None) -> None:
        if self.lease is not None:
            try:
                await self.lease.update(text, title)
            except Exception:
                log.debug("lease update failed", exc_info=True)

    async def release(self) -> None:
        lease, self.lease = self.lease, None
        if lease is not None:
            try:
                await lease.release()
            except Exception:
                log.debug("lease release failed", exc_info=True)
        task, self.task_id = self.task_id, -1
        if task >= 0 and self.native is not None:
            try:
                await self.native.call("end_background_task", task)
            except Exception:
                log.debug("end_background_task failed", exc_info=True)

    def lost(self) -> None:
        lease = self.lease
        if lease is not None and hasattr(lease, "drop"):
            lease.drop()


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


async def _to_thread(fn: Callable[..., Any], *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


class ShareLinkService:
    def __init__(
        self,
        store_path: Any,
        *,
        run_io: Optional[Callable[..., Awaitable[Any]]] = None,
        post: Optional[Callable[..., Any]] = None,
        platform: str = "desktop",
        native: Any = None,
        open_url: Optional[Callable[[str], Any]] = None,
        save_to_downloads: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
        files_visible_root: Optional[str] = None,
        allowed_roots: Iterable[Any] = (),
        cache_dir: Any = None,
        providers: Optional[Mapping[str, Any]] = None,
        secrets: Optional[SecretBox] = None,
        is_busy: Optional[Callable[[str], bool]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from glossarion_mobile.services.share_providers.transferit_handoff import TransferItHandoff

        self.store = ShareLinkStore(store_path)
        self.run_io = run_io or _to_thread
        self.post = post
        self.platform = platform
        self.native = native
        self.secrets = secrets or SecretBox()
        self.allowed_roots = [r for r in (_norm(p) for p in allowed_roots if p) if r]
        base_cache = os.fspath(cache_dir) if cache_dir else os.path.join(os.fspath(Path(store_path).parent), "cache")
        self.snapshot_root = os.path.join(base_cache, SNAPSHOT_DIR)
        self._providers = dict(providers) if providers is not None else None
        self.handoff = TransferItHandoff(platform=platform, open_url=open_url, save_to_downloads=save_to_downloads,
                                         files_visible_root=files_visible_root)
        self.is_busy = is_busy
        self.clock = clock
        self._listeners: list = []
        self._state = UploadState()
        self._cancel: Optional[CancelToken] = None
        self._hold: Optional[_UploadHold] = None
        self._last_notice = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.pending_handoff: Optional[dict] = None  # {"path", "book", "name"} until a link is pasted

    # ---- wiring ----------------------------------------------------------------------------------

    @classmethod
    def from_app(cls, app: Any) -> "ShareLinkService":
        """The service for ``GlossarionApp`` (Integrate: install after the Library)."""
        paths = getattr(app, "paths", None)
        files = getattr(app, "files", None)
        platform = str(getattr(files, "platform", "") or "desktop")
        dispatcher = getattr(app, "dispatcher", None)
        data = getattr(paths, "data", None) or Path.cwd()
        opener = getattr(app, "opener", None)

        def keys_ready() -> bool:
            status = getattr(app, "key_status", None)
            return bool(status is not None and getattr(status, "installed", False))

        # The Library and Output roots (iOS keeps both inside the Files-visible Documents folder). Not the
        # data dir: on Android it is also "docs" and holds the config, keys and Inbox imports.
        roots = [getattr(paths, name, None) for name in ("library", "output")] if paths is not None else []

        def is_busy(path: str) -> bool:
            """A running or queued job works on the file's folder (the Attachments guard's rule)."""
            service = getattr(app, "job_service", None)
            view = getattr(service, "view", None)
            if not callable(view):
                return False
            from glossarion_mobile.ui.chat.attachments import job_writes_into

            current = view()
            folder = os.path.dirname(os.path.abspath(path))
            snapshots = [getattr(current, "active", None), *getattr(current, "queue", ())]
            return any(job_writes_into(snap, folder) for snap in snapshots if snap is not None)

        async def run_in_thread(fn: Callable[..., Any], *args: Any) -> Any:
            return await dispatcher.run_in_thread(fn, *args, name="gl-share")

        service = cls(
            Path(data) / STORE_FILE,
            run_io=run_in_thread if dispatcher is not None else None,
            post=getattr(dispatcher, "post", None),
            platform=platform,
            native=getattr(app, "native", None),
            open_url=getattr(opener, "launch", None),
            save_to_downloads=getattr(files, "save_to_downloads", None),
            files_visible_root=str(getattr(paths, "docs", "")) if platform == "ios" and paths is not None else None,
            allowed_roots=[r for r in roots if r],
            cache_dir=getattr(paths, "cache", None),
            secrets=SecretBox(ready=keys_ready if platform in ("android", "ios") else None),
            is_busy=is_busy,
        )
        native = getattr(app, "native", None)
        if native is not None and hasattr(native, "add_listener"):
            native.add_listener("foreground", service.on_foreground_event)
        return service

    def providers(self) -> dict:
        """The direct providers (built on first use)."""
        if self._providers is None:
            from glossarion_mobile.services.share_providers.gofile import GofileProvider
            from glossarion_mobile.services.share_providers.pixeldrain import PixeldrainProvider
            from glossarion_mobile.services.share_providers.send_e2ee import SendProvider

            self._providers = {"gofile": GofileProvider(), "send": SendProvider(), "pixeldrain": PixeldrainProvider()}
        return self._providers

    def subscribe(self, callback: Callable[[str], Any]) -> Callable[[], None]:
        """``callback(kind)`` with kind ``upload`` / ``links`` / ``providers`` (on the UI loop when a ``post``
        is wired)."""
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _emit(self, kind: str) -> None:
        def deliver() -> None:
            for callback in list(self._listeners):
                try:
                    result = callback(kind)
                    if asyncio.iscoroutine(result):
                        asyncio.ensure_future(result)
                except Exception:
                    log.exception("share-link listener failed")

        if self.post is not None:
            if self.post(deliver) is not False:
                return
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                if asyncio.get_running_loop() is loop:
                    deliver()
                    return
            except RuntimeError:
                pass
            try:
                loop.call_soon_threadsafe(deliver)
                return
            except RuntimeError:
                pass
        deliver()

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await self.run_io(fn, *args)

    # ---- load ----------------------------------------------------------------------------------------

    def load_blocking(self) -> None:
        self.store.load()
        self._sweep_snapshots()
        self.secrets.available()  # warm the probe here, not on the UI loop

    async def load(self) -> None:
        self._loop = asyncio.get_running_loop()
        if not self.store.loaded:
            await self._io(self.load_blocking)

    def _sweep_snapshots(self) -> None:
        """Remove private upload copies left behind by a killed process."""
        root = self.snapshot_root
        if not os.path.isdir(root) or (self._cancel is not None):
            return
        for name in os.listdir(root):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)

    # ---- provider settings and consent -----------------------------------------------------------------

    def _provider_entry(self, data: Mapping, provider_id: str) -> dict:
        entry = (data.get("providers") or {}).get(provider_id)
        return entry if isinstance(entry, dict) else {}

    def provider_state(self, provider_id: str) -> ProviderState:
        info = provider_info(provider_id)
        data = self.store.read()
        entry = self._provider_entry(data, provider_id)
        enabled = bool(entry.get("enabled"))
        try:
            consented = int(entry.get("consent_version") or 0) >= info.consent_version
        except (TypeError, ValueError):
            consented = False
        has_key = bool((data.get("secrets") or {}).get("pixeldrain_key")) if info.needs_api_key else False
        reason = None
        if not enabled:
            reason = "Turn it on in Settings › Cloud sync & sharing"
        elif not consented:
            reason = "Read and accept what this service can see first"
        elif info.needs_api_key and not has_key:
            reason = f"Add your {info.label} API key in Settings › Cloud sync & sharing"
        elif info.direct and not self.secrets.available():
            reason = "Secure storage is not available on this phone"
        return ProviderState(info, enabled, consented, has_key, reason)

    def provider_states(self) -> list:
        return [self.provider_state(pid) for pid in PROVIDER_IDS]

    def consent_text(self, provider_id: str, path: Optional[str] = None, size: Optional[int] = None) -> ConsentText:
        return consent_text(provider_id, file_name=os.path.basename(path) if path else "", size=size)

    async def set_enabled(self, provider_id: str, enabled: bool, *, consent: bool = False) -> ProviderState:
        """Turn a provider on (``consent=True``: the user accepted its consent sheet just now) or off."""
        info = provider_info(provider_id)
        await self.load()
        now = self.clock()

        def change(data: dict) -> None:
            entry = data.setdefault("providers", {}).setdefault(provider_id, {})
            entry["enabled"] = bool(enabled)
            if enabled and consent:
                entry["consent_version"] = info.consent_version
                entry["consent_at"] = now

        await self._io(self.store.mutate, change)
        self._emit("providers")
        return self.provider_state(provider_id)

    async def give_consent(self, provider_id: str) -> ProviderState:
        info = provider_info(provider_id)
        await self.load()
        now = self.clock()

        def change(data: dict) -> None:
            entry = data.setdefault("providers", {}).setdefault(provider_id, {})
            entry["consent_version"] = info.consent_version
            entry["consent_at"] = now

        await self._io(self.store.mutate, change)
        self._emit("providers")
        return self.provider_state(provider_id)

    async def revoke_consent(self, provider_id: str) -> ProviderState:
        """Withdraw consent (the provider is turned off as well)."""
        provider_info(provider_id)
        await self.load()

        def change(data: dict) -> None:
            entry = data.setdefault("providers", {}).setdefault(provider_id, {})
            entry.update({"enabled": False, "consent_version": 0})
            entry.pop("consent_at", None)

        await self._io(self.store.mutate, change)
        self._emit("providers")
        return self.provider_state(provider_id)

    # ---- keys -------------------------------------------------------------------------------------------

    def has_pixeldrain_key(self) -> bool:
        return bool((self.store.read().get("secrets") or {}).get("pixeldrain_key"))

    async def set_pixeldrain_key(self, key: str) -> ProviderState:
        from glossarion_mobile.services.share_providers.pixeldrain import clean_api_key

        clean = clean_api_key(key)
        await self.load()

        def change(data: dict) -> None:
            data.setdefault("secrets", {})["pixeldrain_key"] = self.secrets.seal(clean)

        await self._io(self.store.mutate, change)
        self._emit("providers")
        return self.provider_state("pixeldrain")

    async def clear_pixeldrain_key(self) -> ProviderState:
        await self.load()
        await self._io(self.store.mutate, lambda data: (data.setdefault("secrets", {}).pop("pixeldrain_key", None),
                                                         None)[1])
        self._emit("providers")
        return self.provider_state("pixeldrain")

    async def check_pixeldrain_key(self) -> bool:
        """Ask pixeldrain whether the saved key works (network; only on the user's tap)."""
        await self.load()
        provider = self.providers()["pixeldrain"]
        creds = await self._io(self._credentials_blocking, "pixeldrain")
        return bool(await self._io(provider.check_key, creds))

    async def forget_gofile_account(self) -> None:
        """Drop the saved Gofile guest token (the next upload starts a new guest account). Links made with
        it stay deletable: each keeps its own handle."""
        await self.load()
        await self._io(self.store.mutate, lambda data: (data.setdefault("secrets", {}).pop("gofile_token", None),
                                                         None)[1])
        self._emit("providers")

    def _credentials_blocking(self, provider_id: str) -> dict:
        secrets = self.store.read().get("secrets") or {}
        if provider_id == "gofile":
            token = secrets.get("gofile_token")
            return {"token": self.secrets.open(token)} if token else {}
        if provider_id == "pixeldrain":
            key = secrets.get("pixeldrain_key")
            if not key:
                raise ShareError("no_key", "Add your pixeldrain API key in Settings › Cloud sync & sharing first.")
            return {"api_key": self.secrets.open(key)}
        return {}

    # ---- Send defaults ----------------------------------------------------------------------------------

    def send_options(self) -> dict:
        from glossarion_mobile.services.share_providers.send_e2ee import SendProvider

        raw = self.store.read().get("send") or {}
        expire, downloads = SendProvider.options(raw)
        return {"expire": expire, "downloads": downloads}

    async def set_send_options(self, *, expire: Optional[int] = None, downloads: Optional[int] = None) -> dict:
        from glossarion_mobile.services.share_providers.send_e2ee import SendProvider

        await self.load()
        current = self.send_options()
        wanted = {"expire": expire if expire is not None else current["expire"],
                  "downloads": downloads if downloads is not None else current["downloads"]}
        clean_expire, clean_downloads = SendProvider.options(wanted)

        def change(data: dict) -> None:
            data["send"] = {"expire": clean_expire, "downloads": clean_downloads}

        await self._io(self.store.mutate, change)
        self._emit("providers")
        return self.send_options()

    # ---- eligibility and preflight -----------------------------------------------------------------------

    def eligible(self, path: Any) -> bool:
        """A book output the user may share by link (no IO beyond path resolution)."""
        text = os.fspath(path) if path else ""
        if not text or os.path.splitext(text)[1].lower() not in ELIGIBLE_EXTENSIONS:
            return False
        if not self.allowed_roots:
            return True
        try:
            real = _norm(os.path.realpath(text))
        except OSError:
            return False
        return any(_under(real, root) or _under(real, _norm(os.path.realpath(root))) for root in self.allowed_roots)

    def menu(self, path: Any) -> list:
        """``[(ProviderState, disabled_reason)]`` for a "Share file via link" menu on ``path``."""
        reason_all = None if self.eligible(path) else "Only book files from the Library can be shared by link"
        out = []
        for state in self.provider_states():
            reason = reason_all or state.reason
            if reason is None and self._cancel is not None and state.info.direct:
                reason = "Another upload is running"
            out.append((state, reason))
        return out

    def _require(self, provider_id: str, path: str) -> ProviderState:
        state = self.provider_state(provider_id)
        if not state.enabled:
            raise ShareError("disabled", f"{state.info.label} is turned off. Turn it on in Settings › Cloud sync & "
                             "sharing.")
        if not state.consented:
            raise ShareError("consent", f"Accept what {state.info.label} can see before using it.")
        if not self.eligible(path):
            raise ShareError("not_allowed", "Only book files from the Library can be shared by link.")
        if state.info.needs_api_key and not state.has_key:
            raise ShareError("no_key", f"Add your {state.info.label} API key in Settings › Cloud sync & sharing first.")
        if not self.secrets.available():
            raise ShareError("secrets_unavailable", "Secure storage is not available on this phone, so Glossarion "
                             "cannot keep share links or keys safely.")
        return state

    def _check_file_blocking(self, provider_id: str, path: str) -> tuple:
        info = provider_info(provider_id)
        try:
            size = os.path.getsize(path)
        except OSError:
            raise ShareError("missing_file", "The file is gone. Compile the book again first.") from None
        if not os.path.isfile(path):
            raise ShareError("missing_file", "The file is gone. Compile the book again first.")
        if size <= 0:
            raise ShareError("empty", "The file is empty.")
        if info.max_bytes and size > info.max_bytes:
            raise ShareError("too_large", f"{info.label} accepts files up to {format_size(info.max_bytes)}; this one "
                             f"is {format_size(size)}.")
        if self.is_busy is not None:
            try:
                busy = bool(self.is_busy(path))
            except Exception:
                busy = False
            if busy:
                raise ShareError("busy", "A job is still writing this book. Try again when it has finished.")
        return size, _mime(path)

    def preflight_blocking(self, provider_id: str, path: str) -> Preflight:
        name = os.path.basename(path)
        try:
            self._require(provider_id, path)
            size, mime = self._check_file_blocking(provider_id, path)
        except ShareError as exc:
            return Preflight(provider_id, name, 0, "", blocked=exc.code, message=exc.message)
        warnings = []
        if size > LARGE_FILE_BYTES and provider_info(provider_id).direct:
            warnings.append(f"{format_size(size)} will be uploaded; this uses mobile data when you are not on Wi-Fi.")
        existing = None
        if provider_info(provider_id).direct:
            digest = _sha256(path)
            for link in self.links_blocking(paths=[path]):
                if link.provider == provider_id and link.sha256 == digest and not link.expired:
                    existing = link
                    break
        return Preflight(provider_id, name, size, mime, tuple(warnings), existing=existing)

    async def preflight(self, provider_id: str, path: str) -> Preflight:
        await self.load()
        return await self._io(self.preflight_blocking, provider_id, path)

    # ---- upload --------------------------------------------------------------------------------------------

    @property
    def state(self) -> UploadState:
        return self._state

    @property
    def uploading(self) -> bool:
        return self._cancel is not None

    def _set_state(self, state: UploadState) -> None:
        self._state = state
        self._emit("upload")

    def cancel(self) -> bool:
        """Stop the running upload (at the next chunk; a blocked socket is shut down)."""
        token = self._cancel
        if token is None:
            return False
        token.cancel()
        return True

    def _snapshot_blocking(self, provider_id: str, path: str) -> _Snapshot:
        size, mime = self._check_file_blocking(provider_id, path)
        folder = os.path.join(self.snapshot_root, _secrets.token_hex(8))
        os.makedirs(folder, exist_ok=True)
        name = os.path.basename(path)
        target = os.path.join(folder, name)
        digest = hashlib.sha256()
        copied = 0
        try:
            with open(path, "rb") as src, open(target, "wb") as dst:
                while True:
                    data = src.read(1024 * 1024)
                    if not data:
                        break
                    digest.update(data)
                    dst.write(data)
                    copied += len(data)
        except OSError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise ShareError("missing_file", f"Could not read the file ({type(exc).__name__}).") from exc
        if copied <= 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise ShareError("empty", "The file is empty.")
        return _Snapshot(target, folder, os.path.abspath(path), name, copied, digest.hexdigest(), mime)

    def _progress(self, provider_id: str, name: str, book: str) -> Callable[[int, int], None]:
        def report(sent: int, total: int) -> None:
            state = UploadState("uploading", provider_id, name, int(sent), int(total), "Uploading…", book=book)
            if total and sent >= total:
                state = replace(state, phase="finishing", message="Waiting for the service…")
            self._state = state
            now = time.monotonic()
            final = bool(total and sent >= total)
            if final or now - self._last_notice >= _PROGRESS_INTERVAL:
                self._last_notice = now
                self._emit("upload")
                hold = self._hold
                if hold is not None and self.post is not None:
                    label = provider_info(provider_id).label
                    self.post(hold.update, f"{label} · {state.text or state.message}", "Share file via link")

        return report

    def _upload_blocking(self, provider: Any, snap: _Snapshot, creds: dict, options: dict, token: CancelToken,
                         book: str) -> UploadResult:
        return provider.upload(snap.path, name=snap.name, mime=snap.mime, size=snap.size, credentials=creds,
                               options=options, progress=self._progress(provider.info.id, snap.name, book),
                               cancel=token)

    async def upload(self, provider_id: str, path: str, *, book: Optional[str] = None,
                     options: Optional[Mapping[str, Any]] = None) -> ShareLink:
        """Upload ``path`` to a direct provider and save the link on ``book`` (its identity path).

        Raises ``ShareError`` (``disabled`` / ``consent`` / ``busy`` / ``cancelled`` / network codes …).
        """
        await self.load()
        info = provider_info(provider_id)
        if not info.direct:
            raise ShareError("unsupported", f"{info.label} works through the browser: use start_handoff.")
        if self._cancel is not None:
            raise ShareError("busy", "Another upload is running. Wait for it or cancel it first.")
        token = self._cancel = CancelToken()
        book_path = os.path.abspath(book) if book else os.path.dirname(os.path.abspath(path))
        name = os.path.basename(path)
        snap: Optional[_Snapshot] = None
        self._set_state(UploadState("preparing", provider_id, name, message="Preparing…", book=book_path))
        try:
            self._require(provider_id, path)
            snap = await self._io(self._snapshot_blocking, provider_id, path)
            token.check()
            creds = await self._io(self._credentials_blocking, provider_id)
            opts = dict(options or {})
            if provider_id == "send":
                opts = {**self.send_options(), **opts}
            provider = self.providers()[provider_id]
            hold = self._hold = _UploadHold(self.native, self.platform)
            try:
                await hold.acquire("Share file via link", f"{info.label} · {name}")
                token.check()
                result = await self._io(self._upload_blocking, provider, snap, creds, opts, token, book_path)
            finally:
                self._hold = None
                await hold.release()
            link = await self._io(self._record_upload_blocking, provider_id, snap, result, book_path)
        except ShareError as exc:
            cancelled = exc.code == "cancelled" or token.cancelled
            if exc.code == "auth" and provider_id == "gofile":
                try:  # the next tap starts a new guest account
                    await self._io(self.store.mutate,
                                   lambda data: (data.setdefault("secrets", {}).pop("gofile_token", None), None)[1])
                except Exception:
                    log.debug("could not drop the Gofile token", exc_info=True)
            self._set_state(UploadState("cancelled" if cancelled else "failed", provider_id, name,
                                        message="Upload cancelled" if cancelled else exc.message,
                                        error_code="cancelled" if cancelled else exc.code, book=book_path))
            log.info("share upload to %s %s (%s)", provider_id, "cancelled" if cancelled else "failed",
                     "cancelled" if cancelled else exc.code)
            if cancelled and exc.code != "cancelled":
                raise ShareError("cancelled", "Upload cancelled") from exc
            raise
        except Exception as exc:
            log.exception("share upload to %s crashed", provider_id)
            self._set_state(UploadState("failed", provider_id, name, message="The upload failed unexpectedly.",
                                        error_code="server", book=book_path))
            raise ShareError("server", "The upload failed unexpectedly.") from exc
        finally:
            self._cancel = None
            if snap is not None:
                folder = snap.folder
                try:
                    await self._io(shutil.rmtree, folder, True)
                except Exception:
                    pass
        self._set_state(UploadState("done", provider_id, name, snap.size, snap.size, "Link ready", link.id,
                                    book=book_path))
        self._emit("links")
        log.info("share link created on %s (%s)", provider_id, format_size(snap.size))
        return link

    def _record_upload_blocking(self, provider_id: str, snap: _Snapshot, result: UploadResult, book: str) -> ShareLink:
        url = self.secrets.seal(result.url)
        handle = self.secrets.seal_json(result.delete_handle) if result.delete_handle else None
        token = self.secrets.seal(result.account_token) if result.account_token else None
        record = {
            "id": _secrets.token_hex(6), "provider": provider_id, "book": book, "source": snap.source,
            "name": snap.name, "size": snap.size, "sha256": snap.sha256, "created": self.clock(),
            "expires": result.expires, "downloads_limit": result.downloads_limit, "url": url, "delete": handle,
        }

        def change(data: dict) -> None:
            links = data.setdefault("links", [])
            links.append(record)
            del links[:-MAX_LINKS]
            if token and not (data.setdefault("secrets", {}).get("gofile_token")):
                data["secrets"]["gofile_token"] = token

        self.store.mutate(change)
        return self._view(record, url=result.url)

    # ---- transfer.it handoff ----------------------------------------------------------------------------------

    async def start_handoff(self, path: str, *, book: Optional[str] = None) -> Any:
        """transfer.it: make the file reachable, open https://transfer.it/start in the in-app browser and
        remember the file for the pasted link. No request to transfer.it is made by the app."""
        await self.load()
        self._require("transferit", path)
        await self._io(self._check_file_blocking, "transferit", path)
        previous_uri = previous_name = None
        if self.platform == "android":  # the Downloads entry an earlier handoff of this file made: overwritten
            previous_uri, previous_name = await self._io(self._downloads_entry_blocking, path)
        plan = await self.handoff.prepare(path, replace_uri=previous_uri, saved_name=previous_name)
        if plan.saved_to:
            await self._io(self._remember_downloads_entry_blocking, path, plan.saved_to, plan.saved_name)
        self.pending_handoff = {"path": os.path.abspath(path), "name": os.path.basename(path),
                                "book": os.path.abspath(book) if book else os.path.dirname(os.path.abspath(path))}
        await self.handoff.open(plan)
        return plan

    def _downloads_entry_blocking(self, path: str) -> tuple:
        """``(uri, name)`` of the Downloads/Glossarion entry the last handoff of ``path`` made (Android)."""
        raw = (self.store.read().get(HANDOFF_ENTRIES) or {}).get(_norm(path))
        if not raw:
            return None, None
        try:
            value = self.secrets.open_json(raw)
        except ShareError:
            return None, None
        if not isinstance(value, dict) or not value.get("uri"):
            return None, None
        return str(value["uri"]), (str(value["name"]) if value.get("name") else None)

    def _remember_downloads_entry_blocking(self, path: str, uri: str, name: Optional[str]) -> None:
        """Keep the handoff's Downloads entry of ``path`` (encrypted like the links) for the next handoff."""
        try:
            sealed = self.secrets.seal_json({"uri": str(uri), "name": name})
        except ShareError:
            return

        def change(data: dict) -> None:
            entries = data.setdefault(HANDOFF_ENTRIES, {})
            entries.pop(_norm(path), None)
            entries[_norm(path)] = sealed
            for stale in list(entries)[:-MAX_LINKS]:
                entries.pop(stale, None)

        self.store.mutate(change)

    async def add_pasted_link(self, text: str, *, path: Optional[str] = None, book: Optional[str] = None) -> ShareLink:
        """Save the transfer.it link the user pasted (validated) on the book."""
        from glossarion_mobile.services.share_providers.transferit_handoff import parse_link

        await self.load()
        url = parse_link(text)
        pending = self.pending_handoff or {}
        source = os.path.abspath(path) if path else str(pending.get("path") or "")
        if not source:
            raise ShareError("bad_link", "Pick the book file first.")
        self._require("transferit", source)
        book_path = os.path.abspath(book) if book else str(pending.get("book") or os.path.dirname(source))

        def save() -> ShareLink:
            try:
                size = os.path.getsize(source)
            except OSError:
                size = 0
            sealed = self.secrets.seal(url)
            record = {"id": _secrets.token_hex(6), "provider": "transferit", "book": book_path, "source": source,
                      "name": os.path.basename(source), "size": size, "sha256": "", "created": self.clock(),
                      "expires": None, "downloads_limit": None, "url": sealed, "delete": None}

            def change(data: dict) -> None:
                links = data.setdefault("links", [])
                links.append(record)
                del links[:-MAX_LINKS]

            self.store.mutate(change)
            return self._view(record, url=url)

        link = await self._io(save)
        self.pending_handoff = None
        self._emit("links")
        return link

    # ---- links -------------------------------------------------------------------------------------------------

    def _view(self, record: Mapping[str, Any], *, url: Optional[str] = None) -> ShareLink:
        info = provider_info(str(record.get("provider")))
        if url is None:
            url = self.secrets.open(record.get("url"))
        return ShareLink(
            id=str(record.get("id")), provider=info.id, url=url, name=str(record.get("name") or ""),
            size=int(record.get("size") or 0), created=float(record.get("created") or 0.0),
            book=str(record.get("book") or ""), source=str(record.get("source") or ""),
            expires=float(record["expires"]) if record.get("expires") else None,
            downloads_limit=int(record["downloads_limit"]) if record.get("downloads_limit") else None,
            can_delete=bool(info.can_delete and record.get("delete")), sha256=str(record.get("sha256") or ""),
            now=self.clock(),
        )

    def links_blocking(self, *, book: Optional[str] = None, paths: Sequence[str] = ()) -> list:
        """Links of a book (its identity path, or anything inside it) and/or of files; newest first.

        With neither filter: every link. Records whose secret cannot be opened are skipped.
        """
        wanted_book = _norm(book) if book else ""
        wanted_paths = {_norm(p) for p in paths if p}
        out = []
        for record in self.store.read().get("links") or ():
            key, source = _norm(record.get("book")), _norm(record.get("source"))
            if wanted_book or wanted_paths:
                hit = bool(wanted_book and (key == wanted_book or _under(source, wanted_book)))
                hit = hit or source in wanted_paths
                if not hit:
                    continue
            try:
                out.append(self._view(record))
            except ShareError as exc:
                log.warning("skipping a share link: %s", exc.code)
        out.sort(key=lambda link: link.created, reverse=True)
        return out

    async def links_for(self, *, book: Optional[str] = None, paths: Sequence[str] = ()) -> list:
        await self.load()
        return await self._io(lambda: self.links_blocking(book=book, paths=list(paths)))

    def _record(self, link_id: str) -> Optional[dict]:
        for record in self.store.read().get("links") or ():
            if str(record.get("id")) == str(link_id):
                return record
        return None

    def _drop_records(self, ids: Iterable[str]) -> None:
        ids = {str(i) for i in ids}

        def change(data: dict) -> None:
            data["links"] = [r for r in data.get("links") or () if str(r.get("id")) not in ids]

        self.store.mutate(change)

    def delete_link_blocking(self, link_id: str, cancel: Optional[CancelToken] = None) -> DeleteResult:
        record = self._record(link_id)
        if record is None:
            return DeleteResult(False, "gone")
        provider_id = str(record.get("provider"))
        info = provider_info(provider_id)
        if not (info.can_delete and record.get("delete")):
            raise ShareError("unsupported", f"{info.label} links cannot be deleted from Glossarion; use Remove "
                             "from list, and delete it on the service.")
        handle = self.secrets.open_json(record.get("delete"))
        try:
            creds = self._credentials_blocking(provider_id)
        except ShareError:
            creds = {}
        deleted = self.providers()[provider_id].delete(handle, creds, cancel=cancel)
        self._drop_records([link_id])
        return DeleteResult(True, "deleted" if deleted else "gone")

    async def delete_link(self, link_id: str) -> DeleteResult:
        """Delete the upload on the service, then the record (kept when the service could not be reached)."""
        await self.load()
        result = await self._io(self.delete_link_blocking, link_id)
        self._emit("links")
        return result

    async def forget_link(self, link_id: str) -> bool:
        """Remove the link from the app only (the upload stays on the service until it expires)."""
        await self.load()
        before = self._record(link_id) is not None
        if before:
            await self._io(self._drop_records, [link_id])
            self._emit("links")
        return before

    # ---- book lifecycle -----------------------------------------------------------------------------------------

    def relocate_blocking(self, old: str, new: str) -> int:
        """A book's workspace (or a file) moved (chat auto-migrate, Organize): move its records along."""
        old_key, new_abs = _norm(old), os.path.abspath(new)
        if not old_key or not new_abs or old_key == _norm(new):
            return 0

        def moved(path: Any) -> Optional[str]:
            text = str(path or "")
            key = _norm(text)
            if not key:
                return None
            if key == old_key:
                return new_abs
            if _under(key, old_key):
                rel = os.path.relpath(os.path.abspath(text), os.path.abspath(old))
                if rel.startswith(os.pardir):  # case-folded match on a case-sensitive path: rebuild from the key
                    rel = os.path.relpath(key, old_key)
                return os.path.join(new_abs, rel)
            return None

        def change(data: dict) -> int:
            count = 0
            for record in data.get("links") or ():
                book, source = moved(record.get("book")), moved(record.get("source"))
                if book or source:
                    count += 1
                    if book:
                        record["book"] = book
                    if source:
                        record["source"] = source
            entries = data.get(HANDOFF_ENTRIES)
            if isinstance(entries, dict):
                for key in list(entries):
                    target = moved(key)
                    if target:
                        entries[_norm(target)] = entries.pop(key)
            return count

        return int(self.store.mutate(change) or 0)

    async def relocate(self, old: str, new: str) -> int:
        await self.load()
        count = await self._io(self.relocate_blocking, old, new)
        if count:
            self._emit("links")
        return count

    def forget_books_blocking(self, identities: Iterable[str], *, delete_remote: bool = False) -> list:
        """Worker thread (``LibraryService.execute_delete_blocking``, after the books are gone): drop the
        link records of the deleted books (their workspace, or files inside it). With ``delete_remote`` the
        deletable uploads are deleted on their services first, best effort. Returns the links whose remote
        delete failed; every record goes either way (also the ones whose secret cannot be opened)."""
        keys = [k for k in (_norm(i) for i in identities or ()) if k]
        if not keys:
            return []
        ids = []
        for record in self.store.read().get("links") or ():
            book, source = _norm(record.get("book")), _norm(record.get("source"))
            if any(book == k or _under(source, k) for k in keys):
                ids.append(str(record.get("id")))
        failed = []
        if delete_remote:
            for link in self.links_blocking():
                if link.id in ids and link.can_delete:
                    try:
                        self.delete_link_blocking(link.id)
                    except ShareError:
                        failed.append(link)
        entries = self.store.read().get(HANDOFF_ENTRIES) or {}
        stale = [k for k in entries if any(_norm(k) == key or _under(_norm(k), key) for key in keys)]
        if stale:
            def forget_entries(data: dict) -> None:
                for key in stale:
                    (data.get(HANDOFF_ENTRIES) or {}).pop(key, None)

            self.store.mutate(forget_entries)
        if ids:
            self._drop_records(ids)
            self._emit("links")
        return failed

    def deletable_links_blocking(self, identities: Iterable[str]) -> list:
        """The live links of these books whose upload Glossarion can delete (the Library's delete asks first)."""
        keys = [k for k in (_norm(i) for i in identities or ()) if k]
        out = []
        for link in self.links_blocking() if keys else ():
            book, source = _norm(link.book), _norm(link.source)
            if link.can_delete and not link.expired and any(book == k or _under(source, k) for k in keys):
                out.append(link)
        return out

    async def forget_book(self, book: str, *, delete_remote: bool = False) -> list:
        """``forget_books_blocking`` for one book, from the UI loop."""
        await self.load()
        return await self._io(lambda: self.forget_books_blocking([book], delete_remote=delete_remote))

    async def wipe(self) -> None:
        """Danger zone: forget every link, key and token (uploads stay on their services until they
        expire)."""
        self.cancel()
        await self._io(self.store.remove_file)
        await self._io(lambda: shutil.rmtree(self.snapshot_root, ignore_errors=True))
        self.pending_handoff = None
        self._emit("links")
        self._emit("providers")

    # ---- native events --------------------------------------------------------------------------------------------

    async def on_foreground_event(self, event: Mapping[str, Any]) -> None:
        """Android foreground-service events: **Stop** cancels the upload when no job holds the service
        (a running job owns Stop); the service ending drops the lease without stopping anything."""
        kind = str((event or {}).get("type") or "")
        if self._cancel is None:
            return
        if kind == "button" and str(event.get("button_id") or "") == "stop":
            from glossarion_mobile.services.native import service_holds

            holds = service_holds(self.native)
            if holds is None or "jobs" not in holds:
                self.cancel()
        elif kind in ("destroyed", "timeout"):
            hold = self._hold
            if hold is not None:
                hold.lost()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()
