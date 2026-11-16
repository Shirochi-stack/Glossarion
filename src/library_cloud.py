"""The desktop Library's cloud setup: Google Drive sync and "Share via link" (GUI-free).

Settings live in their own file beside config.json, ``library_cloud.json`` (never in config.json,
like mobile's ``mobile_share_links.json``). Secrets in it - the Google refresh token, the own OAuth
client's secret, the Gofile guest token, the pixeldrain API key, every share link and delete handle -
are ``ENC:`` values sealed by ``share_hosting.SecretBox`` with the desktop's ``api_key_encryption``
key (the one that encrypts the API keys in config.json); nothing secret is written in plain text.

:class:`LibraryCloudStore` is the file; :class:`LibraryCloud` is what the Library window and the
compile hooks call (sign in / out, sync on compile, "Sync to Drive now", share uploads and their
link records). :func:`instance` gives the one shared controller of the running app, so the
translation run's compile and the Library window use the same background worker.

Nothing contacts Google or a hosting site until the user signs in / enables a site; Drive sync is
off by default. Blocking methods (sign-in wait, uploads) belong on worker threads. No Qt import.
"""

from __future__ import annotations

import copy
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import drive_sync_core as dsc
import google_drive_core as gdc
from share_hosting import (
    CancelToken,
    ConsentText,
    SecretBox,
    ShareError,
    UploadResult,
    consent_text,
    provider_info,
)

__all__ = [
    "DESKTOP_PROVIDERS",
    "DEVICE",
    "LibraryCloud",
    "LibraryCloudStore",
    "ProviderStatus",
    "STORE_FILE",
    "book_key_for",
    "default_store_path",
    "install_compile_hook",
    "instance",
]

log = logging.getLogger("glossarion.library_cloud")

STORE_FILE = "library_cloud.json"
STORE_VERSION = 1
#: The consent wording's word for this device.
DEVICE = "computer"
#: "Share via link" sites on the desktop, in menu order. transfer.it is the browser handoff
#: (Glossarion opens transfer.it/start and shows the file's folder; the user pastes the link back).
DESKTOP_PROVIDERS = ("transferit", "gofile", "send", "pixeldrain", "gdrive")
#: Book outputs that may be shared by link.
ELIGIBLE_EXTENSIONS = (".epub", ".pdf", ".txt", ".html", ".htm", ".cbz")
MAX_LINKS = 500


def default_store_path() -> str:
    import app_paths

    return os.path.join(os.path.dirname(os.path.abspath(app_paths.config_file_path())), STORE_FILE)


def book_key_for(path: Any) -> str:
    """The key link records and sync state use for a book: its output folder (or file) path."""
    text = os.fspath(path) if path else ""
    return os.path.normcase(os.path.normpath(os.path.abspath(text))) if text else ""


def _is_library_shelf(folder: Any) -> bool:
    """True for the Library folder itself or a folder inside it (``Library/Translated``, ``Raw``):
    shelves shared by many books, never one book's output folder."""
    if not folder:
        return False
    try:
        import library_core

        library = os.path.normcase(os.path.abspath(library_core.get_library_dir()))
        target = os.path.normcase(os.path.abspath(os.fspath(folder)))
        return os.path.commonpath([library, target]) == library
    except Exception:
        return False


def shelf_book_key(book: Mapping, output_folder: str, compiled: str) -> str:
    """The link / sync key of a Library book: its output folder, or (no output folder, or a file on
    the shared Library shelf) its own compiled file, so two shelf books never share a key."""
    if output_folder and not _is_library_shelf(output_folder):
        return book_key_for(output_folder)
    if compiled:
        return book_key_for(compiled)
    return book_key_for(book.get("path") or "") if isinstance(book, Mapping) else ""


def _defaults() -> dict:
    return {
        "version": STORE_VERSION,
        "drive": {
            "enabled": False,
            "mode": dsc.DEFAULT_MODE,
            "folder": dsc.DEFAULT_FOLDER,
            "use_own_client": False,
            "client_id": "",
            "client_secret": "",  # ENC:
            "refresh_token": "",  # ENC:
            "sync_state": {},  # remembered Drive folder / file ids (no secrets)
            "last_sync": None,  # SyncResult.as_record()
        },
        "share": {"providers": {}, "secrets": {}, "send": {}, "links": []},
    }


def _merge_defaults(data: Any) -> dict:
    merged = _defaults()
    if not isinstance(data, dict):
        return merged
    for section in ("drive", "share"):
        value = data.get(section)
        if isinstance(value, dict):
            merged[section].update(value)
    merged["version"] = STORE_VERSION
    return merged


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    label: str
    enabled: bool
    consented: bool
    ready: bool  # enabled, consented and has what it needs (pixeldrain key, Drive sign-in)
    reason: str = ""  # why it is not ready (one line)
    direct: bool = True  # False: transfer.it browser handoff


class LibraryCloudStore:
    """``library_cloud.json``: thread-safe, an atomic write per change."""

    def __init__(self, path: Optional[str] = None, *, secrets: Optional[SecretBox] = None) -> None:
        self.path = path or default_store_path()
        self.secrets = secrets or SecretBox(device=DEVICE)
        self._lock = threading.RLock()
        self._data: Optional[dict] = None

    # ---- file -------------------------------------------------------------------------------------------

    def _load(self) -> dict:
        if self._data is None:
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    self._data = _merge_defaults(json.load(handle))
            except FileNotFoundError:
                self._data = _defaults()
            except (OSError, ValueError):
                log.warning("Library cloud settings could not be read; using defaults")
                self._data = _defaults()
        return self._data

    def read(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._load())

    def mutate(self, change: Callable[[dict], Any]) -> dict:
        import app_paths

        with self._lock:
            data = copy.deepcopy(self._load())
            change(data)
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            app_paths._atomic_json_write(self.path, data)
            self._data = data
            return copy.deepcopy(data)

    def reload(self) -> None:
        with self._lock:
            self._data = None

    def _open(self, token: Any) -> str:
        return self.secrets.open(token) if token else ""

    # ---- Google Drive sync settings -----------------------------------------------------------------------

    def drive_settings(self) -> dsc.SyncSettings:
        drive = self.read()["drive"]
        return dsc.SyncSettings(enabled=bool(drive.get("enabled")), mode=dsc.normalize_mode(drive.get("mode")),
                                folder="/".join(dsc.folder_parts(drive.get("folder"))),
                                signed_in=bool(drive.get("refresh_token")))

    def set_drive_options(self, *, enabled: Optional[bool] = None, mode: Optional[str] = None,
                          folder: Optional[str] = None) -> dsc.SyncSettings:
        def change(data: dict) -> None:
            drive = data["drive"]
            if enabled is not None:
                drive["enabled"] = bool(enabled)
            if mode is not None:
                drive["mode"] = dsc.normalize_mode(mode)
            if folder is not None:
                drive["folder"] = "/".join(dsc.folder_parts(folder))

        self.mutate(change)
        return self.drive_settings()

    def own_client(self) -> dict:
        """``{"use_own", "client_id", "has_secret"}`` (the secret itself never leaves the store)."""
        drive = self.read()["drive"]
        return {"use_own": bool(drive.get("use_own_client")), "client_id": str(drive.get("client_id") or ""),
                "has_secret": bool(drive.get("client_secret"))}

    def set_own_client(self, *, use_own: Optional[bool] = None, client_id: Optional[str] = None,
                       client_secret: Optional[str] = None) -> dict:
        """``client_secret``: None keeps the stored one, "" clears it. Changing the client signs out
        (a refresh token belongs to the client that issued it)."""
        sealed = None
        if client_secret:
            sealed = self.secrets.seal(str(client_secret).strip())

        def change(data: dict) -> None:
            drive = data["drive"]
            before = (bool(drive.get("use_own_client")), drive.get("client_id") or "")
            if use_own is not None:
                drive["use_own_client"] = bool(use_own)
            if client_id is not None:
                drive["client_id"] = str(client_id).strip()
            if client_secret is not None:
                drive["client_secret"] = sealed or ""
            after = (bool(drive.get("use_own_client")), drive.get("client_id") or "")
            if before != after and drive.get("refresh_token"):
                drive["refresh_token"] = ""

        self.mutate(change)
        return self.own_client()

    def oauth_client(self, **endpoints: str) -> Optional[gdc.OAuthClient]:
        drive = self.read()["drive"]
        secret = ""
        if drive.get("use_own_client") and drive.get("client_secret"):
            secret = self._open(drive.get("client_secret"))
        return gdc.desktop_client(str(drive.get("client_id") or ""), secret,
                                  use_own=bool(drive.get("use_own_client")), **endpoints)

    def refresh_token(self) -> str:
        return self._open(self.read()["drive"].get("refresh_token"))

    def save_refresh_token(self, token: str) -> None:
        sealed = self.secrets.seal(str(token))
        self.mutate(lambda data: data["drive"].update(refresh_token=sealed))

    def clear_refresh_token(self, only: Optional[str] = None) -> None:
        """Forget the stored token (``only``: just when it is still this sealed value)."""
        def change(data: dict) -> None:
            if only is None or data["drive"].get("refresh_token") == only:
                data["drive"]["refresh_token"] = ""

        self.mutate(change)

    def sync_state(self) -> dict:
        state = self.read()["drive"].get("sync_state")
        return state if isinstance(state, dict) else {}

    def save_sync_state(self, state: Mapping) -> None:
        snapshot = copy.deepcopy(dict(state))
        self.mutate(lambda data: data["drive"].update(sync_state=snapshot))

    def last_sync(self) -> Optional[dict]:
        record = self.read()["drive"].get("last_sync")
        return record if isinstance(record, dict) else None

    def record_sync(self, record: Mapping) -> None:
        snapshot = dict(record)
        self.mutate(lambda data: data["drive"].update(last_sync=snapshot))

    # ---- Share via link ---------------------------------------------------------------------------------

    def _provider_entry(self, data: Mapping, provider_id: str) -> dict:
        entry = (data.get("share", {}).get("providers") or {}).get(provider_id)
        return entry if isinstance(entry, dict) else {}

    def provider_state(self, provider_id: str) -> ProviderStatus:
        info = provider_info(provider_id)
        data = self.read()
        entry = self._provider_entry(data, provider_id)
        enabled = bool(entry.get("enabled"))
        consented = int(entry.get("consent_version") or 0) >= info.consent_version
        reason = ""
        if not enabled:
            reason = "Turned off"
        elif not consented:
            reason = "Accept the terms first"
        elif info.direct and not self.secrets.available():
            reason = "Secure storage is not available on this computer"
        elif provider_id == "pixeldrain" and not (data["share"].get("secrets") or {}).get("pixeldrain_key"):
            reason = "Add your pixeldrain API key"
        elif provider_id == "gdrive" and not data["drive"].get("refresh_token"):
            reason = "Sign in to Google Drive (Google Drive sync above)"
        return ProviderStatus(provider_id, info.label, enabled, consented, not reason, reason, info.direct)

    def provider_states(self) -> list:
        return [self.provider_state(pid) for pid in DESKTOP_PROVIDERS]

    def set_provider_enabled(self, provider_id: str, enabled: bool, *, consent: bool = False) -> ProviderStatus:
        """Turn a site on (``consent``: the user ticked the consent text now) or off."""
        if provider_id not in DESKTOP_PROVIDERS:
            raise ShareError("unsupported", "Unknown sharing service")
        info = provider_info(provider_id)
        current = self.provider_state(provider_id)
        if enabled and not consent and not current.consented:
            raise ShareError("consent", f"Accept the {info.label} terms first.")

        def change(data: dict) -> None:
            providers = data["share"].setdefault("providers", {})
            entry = dict(providers.get(provider_id) or {})
            entry["enabled"] = bool(enabled)
            if enabled and consent:
                entry["consent_version"] = info.consent_version
                entry["consented_at"] = time.time()
            providers[provider_id] = entry

        self.mutate(change)
        return self.provider_state(provider_id)

    def revoke_consent(self, provider_id: str) -> ProviderStatus:
        def change(data: dict) -> None:
            providers = data["share"].setdefault("providers", {})
            providers[provider_id] = {"enabled": False, "consent_version": 0}

        self.mutate(change)
        return self.provider_state(provider_id)

    def _secret(self, name: str) -> str:
        return self._open((self.read()["share"].get("secrets") or {}).get(name))

    def _set_secret(self, name: str, value: Optional[str]) -> None:
        sealed = self.secrets.seal(value) if value else ""

        def change(data: dict) -> None:
            secrets_ = data["share"].setdefault("secrets", {})
            if sealed:
                secrets_[name] = sealed
            else:
                secrets_.pop(name, None)

        self.mutate(change)

    def has_pixeldrain_key(self) -> bool:
        return bool((self.read()["share"].get("secrets") or {}).get("pixeldrain_key"))

    def set_pixeldrain_key(self, key: str) -> None:
        from share_hosting_pixeldrain import clean_api_key

        self._set_secret("pixeldrain_key", clean_api_key(key))

    def clear_pixeldrain_key(self) -> None:
        self._set_secret("pixeldrain_key", None)

    def pixeldrain_key(self) -> str:
        return self._secret("pixeldrain_key")

    def gofile_token(self) -> str:
        return self._secret("gofile_token")

    def save_gofile_token(self, token: Optional[str]) -> None:
        self._set_secret("gofile_token", token)

    def send_options(self) -> dict:
        from share_hosting_send import SendProvider

        expire, downloads = SendProvider.options(self.read()["share"].get("send") or {})
        return {"expire": expire, "downloads": downloads}

    def set_send_options(self, *, expire: Optional[int] = None, downloads: Optional[int] = None) -> dict:
        from share_hosting_send import SendProvider

        current = self.send_options()
        if expire is not None:
            current["expire"] = expire
        if downloads is not None:
            current["downloads"] = downloads
        clean_expire, clean_downloads = SendProvider.options(current)
        self.mutate(lambda data: data["share"].update(send={"expire": clean_expire, "downloads": clean_downloads}))
        return self.send_options()

    # ---- link records -----------------------------------------------------------------------------------

    def add_link(self, book_key: str, path: str, provider_id: str, result: UploadResult) -> dict:
        record = {
            "id": uuid.uuid4().hex, "book": book_key, "file": book_key_for(path), "name": os.path.basename(path),
            "provider": provider_id, "created": time.time(), "expires": result.expires,
            "downloads_limit": result.downloads_limit,
            "url": self.secrets.seal(result.url),
            "handle": self.secrets.seal_json(result.delete_handle) if result.delete_handle else "",
        }

        def change(data: dict) -> None:
            links = [link for link in data["share"].get("links") or () if isinstance(link, dict)]
            links.append(record)
            data["share"]["links"] = links[-MAX_LINKS:]

        self.mutate(change)
        return self._plain(record)

    def _plain(self, record: Mapping) -> dict:
        plain = {k: v for k, v in record.items() if k not in ("url", "handle")}
        try:
            plain["url"] = self._open(record.get("url"))
        except ShareError:
            plain["url"] = ""
        plain["can_delete"] = bool(record.get("handle"))
        return plain

    def links(self, book_key: Optional[str] = None) -> list:
        """Link records (newest first) with the URL decrypted; ``book_key`` limits them to one book."""
        records = [r for r in self.read()["share"].get("links") or () if isinstance(r, dict)]
        if book_key is not None:
            records = [r for r in records if r.get("book") == book_key]
        return [self._plain(r) for r in reversed(records)]

    def link_handle(self, link_id: str) -> tuple:
        for record in self.read()["share"].get("links") or ():
            if isinstance(record, dict) and record.get("id") == link_id:
                handle = self.secrets.open_json(record["handle"]) if record.get("handle") else None
                return record.get("provider"), handle
        raise ShareError("not_found", "This link is no longer in the list.")

    def remove_link(self, link_id: str) -> None:
        self.mutate(lambda data: data["share"].update(
            links=[r for r in data["share"].get("links") or () if isinstance(r, dict) and r.get("id") != link_id]))


class LibraryCloud:
    """What the desktop Library window and the compile hooks call.

    ``endpoints``: overrides for tests (``auth_url``, ``token_url``, ``revoke_url``, ``api_base``,
    ``upload_base``); ``providers``: share-link provider objects by id (tests); ``log``: one-line log
    sink (the main window's log), called from worker threads.
    """

    def __init__(self, store: Optional[LibraryCloudStore] = None, *, http: Any = None,
                 log_line: Optional[Callable[[str], Any]] = None, endpoints: Optional[Mapping[str, str]] = None,
                 providers: Optional[Mapping[str, Any]] = None,
                 on_change: Optional[Callable[[str], Any]] = None) -> None:
        self.store = store or LibraryCloudStore()
        self.http = http
        self.log_line = log_line
        self.on_change = on_change  # "drive" | "sync" | "share" | "links" (worker threads too)
        self.endpoints = dict(endpoints or {})
        self._providers = dict(providers) if providers is not None else None
        self._oauth: Optional[gdc.GoogleOAuth] = None
        self._oauth_lock = threading.Lock()
        self._revoke_thread: Optional[threading.Thread] = None
        self.sync =dsc.DriveSyncService(
            settings=self.store.drive_settings, access=self.drive_access, load_state=self.store.sync_state,
            save_state=self.store.save_sync_state, on_result=self._on_sync_result, log_line=self._log,
            on_sign_in_lost=self._sign_in_lost, http=http, api_base=self.endpoints.get("api_base"),
            upload_base=self.endpoints.get("upload_base"), resolve_source=self._resolve_source,
        )

    # ---- plumbing ---------------------------------------------------------------------------------------

    def _log(self, line: str) -> None:
        if self.log_line is not None:
            try:
                self.log_line(line)
            except Exception:
                pass

    def _changed(self, what: str) -> None:
        if self.on_change is not None:
            try:
                self.on_change(what)
            except Exception:
                log.debug("library cloud change callback failed", exc_info=True)

    def _oauth_endpoints(self) -> dict:
        return {k: v for k, v in self.endpoints.items() if k in ("auth_url", "token_url", "revoke_url")}

    def _google(self) -> gdc.GoogleOAuth:
        client = self.store.oauth_client(**self._oauth_endpoints())
        if client is None:  # a settings problem, not a lost grant: the stored token is kept
            raise ShareError("unsupported", gdc.NO_DESKTOP_CLIENT)
        with self._oauth_lock:
            if self._oauth is None or self._oauth.client != client:
                self._oauth = gdc.GoogleOAuth(self.http, client)
            return self._oauth

    # ---- Google sign-in ---------------------------------------------------------------------------------

    def sign_in_status(self) -> tuple:
        """``(available, reason)``: sign-in needs the built-in or the user's own OAuth client."""
        if self.store.oauth_client() is None:
            return False, gdc.NO_DESKTOP_CLIENT
        if not self.store.secrets.available():
            return False, "Secure storage is not available on this computer, so the sign-in cannot be kept."
        return True, ""

    def signed_in(self) -> bool:
        return self.store.drive_settings().signed_in

    def begin_sign_in(self, timeout: float = 300.0) -> gdc.DesktopSignIn:
        """Start the loopback listener; open the result's ``auth_url`` in the system browser, then call
        :meth:`finish_sign_in` on a worker thread."""
        available, reason = self.sign_in_status()
        if not available:
            raise ShareError("unsupported", reason)
        return gdc.DesktopSignIn(self.store.oauth_client(**self._oauth_endpoints()), http=self.http,
                                 timeout=timeout)

    def finish_sign_in(self, sign_in: gdc.DesktopSignIn, *, timeout: Optional[float] = None,
                       cancel: Optional[CancelToken] = None) -> None:
        result = sign_in.wait(timeout, cancel)
        self.store.save_refresh_token(result.refresh_token)
        with self._oauth_lock:
            self._oauth = sign_in.oauth  # it already holds the fresh access token
        self._log("☁️ Google Drive connected")
        self._changed("drive")

    def sign_out(self) -> bool:
        """Revoke the grant at Google (best effort) and delete it here. True when Google confirmed."""
        token = ""
        try:
            token = self.store.refresh_token()
        except ShareError:
            pass
        revoked = False
        if token:
            try:
                revoked = self._google().revoke(token)
            except ShareError as exc:
                log.info("Google revoke failed (%s); the local sign-out still applies", exc.code)
        self.store.clear_refresh_token()
        with self._oauth_lock:
            if self._oauth is not None:
                self._oauth.forget()
        self._log("☁️ Google Drive disconnected")
        self._changed("drive")
        return revoked

    def set_own_client(self, *, use_own: Optional[bool] = None, client_id: Optional[str] = None,
                       client_secret: Optional[str] = None) -> dict:
        """Save the "own OAuth client" fields. A change of client while signed in is a sign-out: the
        old grant is revoked at Google with the client that issued it (best effort, on a background
        thread) before the store forgets it."""
        before = self.store.own_client()
        after_use = before["use_own"] if use_own is None else bool(use_own)
        after_id = before["client_id"] if client_id is None else str(client_id).strip()
        changes = (before["use_own"], before["client_id"]) != (after_use, after_id)
        token, old_client = "", None
        if changes and self.signed_in():
            try:
                token = self.store.refresh_token()
                old_client = self.store.oauth_client(**self._oauth_endpoints())
            except ShareError:
                token = ""
        result = self.store.set_own_client(use_own=use_own, client_id=client_id, client_secret=client_secret)
        if token and old_client is not None:
            def revoke() -> None:
                try:
                    gdc.GoogleOAuth(self.http, old_client).revoke(token)
                except Exception as exc:
                    log.info("Google revoke after a client change failed (%s)", type(exc).__name__)

            self._revoke_thread = threading.Thread(target=revoke, name="google-revoke", daemon=True)
            self._revoke_thread.start()
        if changes and token:
            with self._oauth_lock:
                if self._oauth is not None:
                    self._oauth.forget()
            self._log("☁️ Google Drive disconnected (the OAuth client changed)")
            self._changed("drive")
        return result

    def drive_access(self) -> Callable[[str], str]:
        """The ``access(rejected)`` token callable for ``DriveClient`` (refreshes with the stored grant)."""
        def access(rejected: str) -> str:
            refresh = self.store.refresh_token()
            if not refresh:
                raise ShareError("sign_in", "Sign in to Google Drive in Library settings.")
            return self._google().token(refresh, rejected)

        return access

    def _sign_in_lost(self) -> None:
        self.store.clear_refresh_token()
        self._log("⚠️ Google Drive sign-in expired or was revoked; sign in again in Library settings.")
        self._changed("drive")

    # ---- Drive sync ---------------------------------------------------------------------------------------

    def _on_sync_result(self, job: dsc.SyncJob, result: dsc.SyncResult) -> None:
        if result.error_code != "disabled":
            self.store.record_sync(result.as_record())
        self._changed("sync")

    @staticmethod
    def _resolve_source(job: dsc.SyncJob) -> str:
        try:
            import library_core

            return library_core._find_raw_source_for_folder(job.output_folder) or ""
        except Exception:
            return ""

    def sync_folder(self, output_folder: str, *, compiled_files: tuple = (), source_file: str = "",
                    label: str = "", reason: str = "manual") -> bool:
        if not output_folder:
            return False
        job = dsc.SyncJob(book_key=book_key_for(output_folder), label=label or os.path.basename(
            os.path.normpath(output_folder)), reason=reason, output_folder=output_folder,
            compiled_files=tuple(p for p in compiled_files if p), source_file=source_file or "")
        return self.sync.request_sync(job)

    def sync_book(self, book: Mapping) -> bool:
        """"Sync to Drive now" for a Library card's book dict (queued; False when Drive is not connected)."""
        try:
            import library_core

            output = library_core._resolve_book_output_folder(dict(book))
            compiled = library_core._resolve_book_translated_file(dict(book))
            source = library_core._resolve_book_source_file(dict(book))
        except Exception:
            log.debug("Library book paths unavailable", exc_info=True)
            return False
        label = str(book.get("title") or book.get("name") or "")
        if not output or _is_library_shelf(output):
            # No output folder (an organized or imported book): only its own file, keyed by that file.
            # Never the folder it sits in: that is the shared Library shelf with every other book.
            if not compiled or not os.path.isfile(compiled):
                return False
            job = dsc.SyncJob(book_key=book_key_for(compiled), label=label or os.path.basename(compiled),
                              reason="manual", output_folder="", compiled_files=(compiled,))
            return self.sync.request_sync(job)
        compiled_files = tuple(dict.fromkeys([p for p in [compiled] + dsc.compiled_files_in(output) if p]))
        return self.sync_folder(output, compiled_files=compiled_files, source_file=source,
                                label=label, reason="manual")

    def on_compile_finished(self, kind: Any, path: Any, error: Any = None) -> bool:
        """The compile hook (Library compile and the compile at the end of a translation run). Queues
        the book when Drive sync is on; never raises."""
        try:
            if error or not path or not os.path.isfile(str(path)):
                return False
            path = str(path)
            if not path.lower().endswith(dsc.COMPILED_EXTENSIONS):
                return False
            settings = self.store.drive_settings()
            if not settings.active:
                return False
            folder = os.path.dirname(os.path.abspath(path))
            if _is_library_shelf(folder):  # a file on the shared Library shelf: only that file
                job = dsc.SyncJob(book_key=book_key_for(path), label=os.path.basename(path), reason="compile",
                                  output_folder="", compiled_files=(path,))
            else:
                job = dsc.SyncJob(book_key=book_key_for(folder), label=os.path.basename(folder),
                                  reason="compile", output_folder=folder, compiled_files=(path,))
            return self.sync.on_compiled(job)
        except Exception:
            log.debug("Drive sync hook failed", exc_info=True)
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """App closing: cancel the running upload and drop queued ones."""
        self.sync.shutdown(timeout)

    # ---- Share via link -------------------------------------------------------------------------------

    def providers(self) -> dict:
        if self._providers is None:
            from share_hosting_gofile import GofileProvider
            from share_hosting_pixeldrain import PixeldrainProvider
            from share_hosting_send import SendProvider

            kwargs = {}
            if self.endpoints.get("api_base"):
                kwargs["api_base"] = self.endpoints["api_base"]
            if self.endpoints.get("upload_base"):
                kwargs["upload_base"] = self.endpoints["upload_base"]
            self._providers = {"gofile": GofileProvider(self.http), "send": SendProvider(self.http),
                               "pixeldrain": PixeldrainProvider(self.http),
                               "gdrive": gdc.GoogleDriveProvider(self.http, **kwargs)}
        return self._providers

    def consent_text(self, provider_id: str, path: Optional[str] = None) -> ConsentText:
        size = None
        if path:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
        return consent_text(provider_id, file_name=os.path.basename(path) if path else "", size=size, device=DEVICE)

    def _credentials(self, provider_id: str) -> dict:
        if provider_id == "gofile":
            token = self.store.gofile_token()
            return {"token": token} if token else {}
        if provider_id == "pixeldrain":
            key = self.store.pixeldrain_key()
            if not key:
                raise ShareError("no_key", "Add your pixeldrain API key in Library settings first.")
            return {"api_key": key}
        if provider_id == "gdrive":
            return {"access_token": self.drive_access()}
        return {}

    def _check_ready(self, provider_id: str) -> None:
        if provider_id not in DESKTOP_PROVIDERS:
            raise ShareError("unsupported", "Unknown sharing service")
        state = self.store.provider_state(provider_id)
        if not state.enabled:
            raise ShareError("disabled", f"Turn on {state.label} in Library settings first.")
        if not state.consented:
            raise ShareError("consent", f"Accept the {state.label} terms in Library settings first.")
        if not state.ready:
            raise ShareError("no_key" if provider_id == "pixeldrain" else "sign_in", state.reason)

    def share_upload(self, provider_id: str, path: str, *, book_key: str = "",
                     progress: Optional[Callable[[int, int], Any]] = None,
                     cancel: Optional[CancelToken] = None) -> dict:
        """Upload ``path`` to an enabled site and record the link (blocking; run on a worker thread).
        Returns the link record with its plain ``url``."""
        if provider_id == "transferit":
            raise ShareError("unsupported", "transfer.it is a browser handoff: use transferit_plan().")
        self._check_ready(provider_id)
        info = provider_info(provider_id)
        if not path or not os.path.isfile(path):
            raise ShareError("missing_file", "The file is not there any more. Compile the book again.")
        if not path.lower().endswith(ELIGIBLE_EXTENSIONS):
            raise ShareError("not_allowed", "Only compiled book files can be shared.")
        size = os.path.getsize(path)
        if size <= 0:
            raise ShareError("empty", "The file is empty.")
        if info.max_bytes and size > info.max_bytes:
            raise ShareError("too_large", f"The file is larger than {info.label} allows.")
        cancel = cancel or CancelToken()
        credentials = self._credentials(provider_id)
        options = self.store.send_options() if provider_id == "send" else {}
        name = os.path.basename(path)
        mime = "application/epub+zip" if name.lower().endswith(".epub") else (
            mimetypes.guess_type(name)[0] or "application/octet-stream")
        # a private snapshot: a compile that rewrites the file meanwhile cannot tear the upload
        snapshot_dir = tempfile.mkdtemp(prefix="glossarion_share_")
        snapshot = os.path.join(snapshot_dir, name)
        try:
            shutil.copyfile(path, snapshot)
            result = self.providers()[provider_id].upload(snapshot, name=name, mime=mime, size=size,
                                                          credentials=credentials, options=options,
                                                          progress=progress, cancel=cancel)
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        if provider_id == "gofile" and result.account_token:
            self.store.save_gofile_token(result.account_token)
        record = self.store.add_link(book_key or book_key_for(os.path.dirname(path)), path, provider_id, result)
        self._log(f"\U0001f517 Share link ready ({info.label}): {name}")
        self._changed("links")
        return record

    def transferit_plan(self, path: str) -> Any:
        """The transfer.it handoff for ``path``: open ``plan.url`` in the browser, show ``plan.hint`` and
        the file's folder (``plan.location``), then take the pasted link with :meth:`add_pasted_link`."""
        import asyncio

        from share_hosting_transferit import TransferItHandoff

        self._check_ready("transferit")
        return asyncio.run(TransferItHandoff(platform="desktop").prepare(path))

    def add_pasted_link(self, path: str, text: str, *, book_key: str = "") -> dict:
        from share_hosting_transferit import parse_link

        self._check_ready("transferit")
        url = parse_link(text)
        record = self.store.add_link(book_key or book_key_for(os.path.dirname(path)), path, "transferit",
                                     UploadResult(url=url))
        self._changed("links")
        return record

    def links_for(self, book_key: str) -> list:
        return self.store.links(book_key)

    def delete_link(self, link_id: str, *, cancel: Optional[CancelToken] = None) -> bool:
        """Delete the upload at the site when it can (else only forget the link). True when deleted there."""
        provider_id, handle = self.store.link_handle(link_id)
        deleted = False
        if handle and provider_id in self.providers():
            try:
                deleted = bool(self.providers()[provider_id].delete(handle, self._credentials(provider_id),
                                                                   cancel or CancelToken()))
            except ShareError as exc:
                if exc.code != "not_found":
                    raise
        self.store.remove_link(link_id)
        self._changed("links")
        return deleted


_INSTANCE: Optional[LibraryCloud] = None
_INSTANCE_LOCK = threading.Lock()


def instance(**kwargs: Any) -> LibraryCloud:
    """The running app's one controller (created on first use; ``kwargs`` only apply then, except
    ``log_line`` / ``on_change``, which replace the current sinks when given)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = LibraryCloud(**kwargs)
        else:
            if kwargs.get("log_line") is not None:
                _INSTANCE.log_line = kwargs["log_line"]
            if kwargs.get("on_change") is not None:
                _INSTANCE.on_change = kwargs["on_change"]
        return _INSTANCE


def reset_instance() -> None:
    """Shut the shared controller down and forget it (app exit, tests)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        current, _INSTANCE = _INSTANCE, None
    if current is not None:
        current.shutdown()


def _compile_listener(kind: str, path: Optional[str], error: Optional[str]) -> None:
    instance().on_compile_finished(kind, path, error)


def install_compile_hook(log_line: Optional[Callable[[str], Any]] = None) -> None:
    """Desktop start-up: every reported compile (the Library / main-window compile through
    ``_notify_compile_result`` and the compile at the end of a translation run) reaches
    :meth:`LibraryCloud.on_compile_finished` through ``drive_sync_core.compile_finished``. Reads no
    file and starts no thread until a compile is reported; ``log_line`` is the main window's log."""
    if log_line is not None:
        instance(log_line=log_line)
    dsc.add_compile_listener(_compile_listener)


def uninstall_compile_hook() -> None:
    dsc.remove_compile_listener(_compile_listener)
