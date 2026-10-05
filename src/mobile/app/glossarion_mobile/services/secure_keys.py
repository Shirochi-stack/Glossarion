"""Backend encryption keys kept in the platform secure storage (plan §3 Keys, §4 bootstrap).

API keys in ``config.json`` are Fernet ``ENC:`` values (``api_key_encryption``)
and OAuth tokens are encrypted by ``token_encryption``. Desktop keeps their keys
in a file next to the app (or the macOS Keychain). On a phone that path is
wrong: Android re-extracts the app dir on every update, so the key and every
saved API key would be lost, and the iOS bundle is read-only.

So the app keeps both keys in Flet ``SecureStorage`` (Android Keystore / iOS
Keychain), creates them on the first run, and hands them to the backend with
``api_key_encryption.set_key_material()`` and
``token_encryption.set_symmetric_key()`` before the warm-import thread starts
(``GlossarionApp.start``). The keys never go into ``os.environ``.

If SecureStorage cannot be used (no platform support, a Keystore error, a
timeout), the keys come from a fallback file in the data dir, which survives
updates, and the status is flagged ``degraded`` (the self-test reports it). A
later run with working SecureStorage moves the fallback keys into it, so values
encrypted while degraded stay readable.

``storage`` is duck-typed (``async get(key)`` / ``async set(key, value)``), so
this module never imports Flet, and host tests can pass a fake.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import importlib
import json
import logging
import os
import secrets
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "API_KEY_NAME",
    "TOKEN_KEY_NAME",
    "FALLBACK_FILE_NAME",
    "KeyStatus",
    "LoadedKeys",
    "current_status",
    "decrypt_with_installed_api_key",
    "fingerprint",
    "install_backend_keys",
    "load_or_create",
    "new_api_key",
    "new_token_key",
    "reset",
    "setup",
]

log = logging.getLogger("glossarion.keys")

API_KEY_NAME = "glossarion.api_key_fernet"  # url-safe base64 Fernet key (44 chars)
TOKEN_KEY_NAME = "glossarion.token_key"  # standard base64 of 32 random bytes
FALLBACK_FILE_NAME = ".glossarion_keys.json"  # <data>/, only while SecureStorage is unusable
STORAGE_TIMEOUT = 20.0  # seconds per SecureStorage call (first Keystore use can be slow)

_LOCK = threading.Lock()
_STATUS: Optional["KeyStatus"] = None
_INSTALLED_API_KEY: Optional[bytes] = None


# --------------------------------------------------------------------------
# Key material
# --------------------------------------------------------------------------


def new_api_key() -> bytes:
    """A new Fernet key (the same format as ``Fernet.generate_key()``)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32))


def new_token_key() -> bytes:
    """A new 32-byte token-encryption key."""
    return secrets.token_bytes(32)


def fingerprint(key: bytes) -> str:
    """Short, non-reversible identifier of a key (for logs and the self-test)."""
    return hashlib.sha256(bytes(key)).hexdigest()[:12]


def _decode_api_key(value: Any) -> bytes:
    """A stored Fernet key (str or bytes) as ASCII bytes; ValueError when malformed."""
    if isinstance(value, str):
        value = value.strip().encode("ascii", "replace")
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("not a string")
    value = bytes(value).strip()
    try:
        raw = base64.urlsafe_b64decode(value)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not base64: {exc}") from None
    if len(raw) != 32:
        raise ValueError(f"decodes to {len(raw)} bytes, not 32")
    return value


def _encode_token_key(key: bytes) -> str:
    return base64.b64encode(key).decode("ascii")


def _decode_token_key(value: Any) -> bytes:
    """A stored token key (base64 str or bytes) as 32 raw bytes; ValueError when malformed."""
    if isinstance(value, str):
        value = value.strip().encode("ascii", "replace")
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("not a string")
    try:
        raw = base64.b64decode(bytes(value).strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not base64: {exc}") from None
    if len(raw) != 32:
        raise ValueError(f"decodes to {len(raw)} bytes, not 32")
    return raw


# --------------------------------------------------------------------------
# Fallback file (degraded mode only)
# --------------------------------------------------------------------------


def _read_fallback(path: Path, notes: list[str]) -> Optional[tuple[bytes, bytes]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        notes.append(f"{path.name} unreadable ({type(exc).__name__}); ignored")
        return None
    try:
        return _decode_api_key(data.get("api_key_fernet")), _decode_token_key(data.get("token_key_b64"))
    except (AttributeError, ValueError) as exc:
        notes.append(f"{path.name} holds malformed keys ({exc}); ignored")
        return None


def _write_fallback(path: Path, api_key: bytes, token_key: bytes) -> None:
    """Atomic write, owner-only permissions."""
    payload = json.dumps(
        {"version": 1, "api_key_fernet": api_key.decode("ascii"), "token_key_b64": _encode_token_key(token_key)}
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))


def _remove_fallback(path: Path, notes: list[str]) -> None:
    try:
        path.unlink()
        notes.append(f"moved the {path.name} keys into SecureStorage")
    except FileNotFoundError:
        pass
    except OSError as exc:
        notes.append(f"could not remove {path.name}: {exc}")


@dataclass
class LoadedKeys:
    api_key: bytes  # Fernet key, url-safe base64 ASCII
    token_key: bytes  # 32 raw bytes
    source: str  # "secure_storage" | "created" | "migrated" | "fallback_file" | "session"
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


def _degraded(path: Path, fallback: Optional[tuple[bytes, bytes]], notes: list[str]) -> LoadedKeys:
    if fallback is not None:
        return LoadedKeys(fallback[0], fallback[1], "fallback_file", True, notes)
    api_key, token_key = new_api_key(), new_token_key()
    try:
        _write_fallback(path, api_key, token_key)
        notes.append(f"created {path.name}")
        source = "fallback_file"
    except OSError as exc:
        notes.append(f"could not write {path.name} ({exc}); keys last for this session only")
        source = "session"
    return LoadedKeys(api_key, token_key, source, True, notes)


async def load_or_create(storage: Any, fallback_path: Path, *, timeout: float = STORAGE_TIMEOUT) -> LoadedKeys:
    """Read both keys from SecureStorage, creating (and storing) the missing ones.

    Never raises: problems fall back to ``fallback_path`` (``degraded``).
    """
    notes: list[str] = []
    fallback = _read_fallback(fallback_path, notes)
    if storage is None:
        notes.append("SecureStorage is not available")
        return _degraded(fallback_path, fallback, notes)
    try:
        stored_api = await asyncio.wait_for(storage.get(API_KEY_NAME), timeout)
        stored_token = await asyncio.wait_for(storage.get(TOKEN_KEY_NAME), timeout)
    except Exception as exc:  # timeout, Keystore/Keychain error, unsupported platform
        notes.append(f"SecureStorage read failed ({type(exc).__name__}: {exc})")
        return _degraded(fallback_path, fallback, notes)
    try:
        api_key = _decode_api_key(stored_api) if stored_api else None
        token_key = _decode_token_key(stored_token) if stored_token else None
    except ValueError as exc:
        # Never overwrite a stored key we do not understand.
        notes.append(f"SecureStorage holds a malformed key ({exc}); left untouched")
        return _degraded(fallback_path, fallback, notes)

    missing: list[tuple[str, str]] = []
    if api_key is None:
        api_key = fallback[0] if fallback is not None else new_api_key()
        missing.append((API_KEY_NAME, api_key.decode("ascii")))
    if token_key is None:
        token_key = fallback[1] if fallback is not None else new_token_key()
        missing.append((TOKEN_KEY_NAME, _encode_token_key(token_key)))
    try:
        for name, value in missing:
            await asyncio.wait_for(storage.set(name, value), timeout)
    except Exception as exc:
        notes.append(f"SecureStorage write failed ({type(exc).__name__}: {exc})")
        try:  # keep the keys for the next run, which moves them into SecureStorage
            _write_fallback(fallback_path, api_key, token_key)
            return LoadedKeys(api_key, token_key, "fallback_file", True, notes)
        except OSError as write_exc:
            notes.append(f"could not write {fallback_path.name} ({write_exc}); keys last for this session only")
            return LoadedKeys(api_key, token_key, "session", True, notes)

    if fallback is None:
        source = "created" if missing else "secure_storage"
    elif fallback == (api_key, token_key):
        source = "migrated" if missing else "secure_storage"
        _remove_fallback(fallback_path, notes)
    else:
        source = "secure_storage"
        notes.append(f"{fallback_path.name} holds different keys than SecureStorage; kept it")
    return LoadedKeys(api_key, token_key, source, False, notes)


# --------------------------------------------------------------------------
# Backend hand-off
# --------------------------------------------------------------------------


@dataclass
class KeyStatus:
    source: str
    degraded: bool = False
    installed: bool = False
    api_key_fingerprint: Optional[str] = None
    token_key_fingerprint: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "degraded": self.degraded,
            "installed": self.installed,
            "api_key_fingerprint": self.api_key_fingerprint,
            "token_key_fingerprint": self.token_key_fingerprint,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def install_backend_keys(
    api_key: bytes,
    token_key: bytes,
    *,
    source: str,
    degraded: bool = False,
    notes: Optional[list[str]] = None,
) -> KeyStatus:
    """Pass the keys to ``api_key_encryption`` and ``token_encryption`` (before the warm import)."""
    global _STATUS, _INSTALLED_API_KEY
    status = KeyStatus(source=source, degraded=degraded, notes=list(notes or ()))
    installed_api: Optional[bytes] = None
    try:
        importlib.import_module("api_key_encryption").set_key_material(api_key)
        installed_api = bytes(api_key)
        status.api_key_fingerprint = fingerprint(api_key)
    except Exception as exc:
        status.errors.append(f"api_key_encryption.set_key_material: {type(exc).__name__}: {exc}")
    try:
        importlib.import_module("token_encryption").set_symmetric_key(token_key)
        status.token_key_fingerprint = fingerprint(token_key)
    except Exception as exc:
        status.errors.append(f"token_encryption.set_symmetric_key: {type(exc).__name__}: {exc}")
    status.installed = not status.errors
    with _LOCK:
        _STATUS = status
        _INSTALLED_API_KEY = installed_api
    if status.errors:
        log.error("encryption keys not installed: %s", "; ".join(status.errors))
    elif degraded:
        log.warning("encryption keys from %s (SecureStorage unusable): %s", source, "; ".join(status.notes))
    else:
        log.info("encryption keys installed from %s", source)
    return status


async def setup(storage: Any, fallback_dir: Path, *, timeout: float = STORAGE_TIMEOUT) -> KeyStatus:
    """``load_or_create`` + ``install_backend_keys``; never raises (startup must go on)."""
    global _STATUS
    try:
        keys = await load_or_create(storage, Path(fallback_dir) / FALLBACK_FILE_NAME, timeout=timeout)
    except Exception as exc:  # pragma: no cover - load_or_create handles its own failures
        log.exception("loading the encryption keys failed")
        status = KeyStatus(source="none", degraded=True, errors=[f"{type(exc).__name__}: {exc}"])
        with _LOCK:
            _STATUS = status
        return status
    return install_backend_keys(keys.api_key, keys.token_key, source=keys.source, degraded=keys.degraded, notes=keys.notes)


def current_status() -> Optional[KeyStatus]:
    with _LOCK:
        return _STATUS


def decrypt_with_installed_api_key(token: bytes) -> bytes:
    """Fernet-decrypt ``token`` with the key given to ``set_key_material`` (self-test)."""
    with _LOCK:
        key = _INSTALLED_API_KEY
    if key is None:
        raise LookupError("no API key was installed")
    from cryptography.fernet import Fernet

    return Fernet(key).decrypt(token)


def reset() -> None:
    """Forget the installed keys (host tests); clears the setters of already-imported modules."""
    global _STATUS, _INSTALLED_API_KEY
    with _LOCK:
        _STATUS = None
        _INSTALLED_API_KEY = None
    for module_name, setter in (("api_key_encryption", "set_key_material"), ("token_encryption", "set_symmetric_key")):
        module = sys.modules.get(module_name)
        if module is not None and callable(getattr(module, setter, None)):
            try:
                getattr(module, setter)(None)
            except Exception:
                pass
