"""Passphrase-protected config export / import (UI_SPEC §4.16 Data › Backup: "Export config (keys
excluded, or encrypted with a passphrase) / Import config"; FEATURE_MAP api-client #87).

On the phone config.json's API keys are encrypted with the device's own key (SecureStorage), so a
plain copy cannot be read on another device. An export re-encrypts every API-key field with a key
derived from a passphrase:

* the ``ENC:<base64 Fernet token>`` value format is the shared ``api_key_encryption.APIKeyEncryption``
  one (``encrypt_config`` / ``decrypt_config`` run on a handler whose cipher is the passphrase key),
  so the fields the desktop encrypts are protected; the other credentials in config.json
  (``secret_fields``: every settings_schema key typed ``secret``, nested ones such as
  ``qa_scanner_settings.ai_truncation_api_key`` included, and the Azure OCR keys) are encrypted
  with the same handler, and an export that would still hold one in plain text is refused;
* the key is ``scrypt(passphrase, 16-byte random salt, n=2**15, r=8, p=1)`` → Fernet;
* a ``check`` token (an encrypted constant) tells a wrong passphrase from a damaged file.

The file is JSON: ``{"format": FORMAT, "version": 1, "kdf": {...}, "check": "ENC:…",
"exported": time, "config": {...}}``. Import returns the decrypted config (the device key
re-encrypts it when the store saves). ``without_secrets`` is the "Export settings (without API
keys)" copy: every one of those fields removed.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import time
from typing import Any, Mapping

__all__ = ["EXTRA_SECRET_KEYS", "FORMAT", "MIN_PASSPHRASE", "WrongPassphrase", "export_config", "import_config",
           "read_export", "secret_fields", "without_secrets", "write_export"]

FORMAT = "glossarion-config-export"
VERSION = 1
MIN_PASSPHRASE = 8
_CHECK_TEXT = "glossarion-config-export-check"
_SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1, "length": 32}


#: Credentials config.json holds outside ``api_key_encryption``'s field lists and the settings
#: schema: the Azure Computer Vision / Document Intelligence keys of the manga OCR settings
#: (services.manga K_AZURE_KEY / K_DOCINTEL_KEY) and ``azure_key``, the OCR config's own name for it.
EXTRA_SECRET_KEYS = ("azure_vision_key", "azure_document_intelligence_key", "azure_key")


class WrongPassphrase(ValueError):
    """The passphrase does not open this export."""


def secret_fields() -> tuple:
    """``(paths, lists)``: the config paths (tuples; ``parent.child`` split) that hold a credential,
    and the key-pool list fields whose entries carry ``api_key``.

    ``paths``: ``api_key_encryption``'s plain field list, every settings_schema key typed
    ``secret`` and ``EXTRA_SECRET_KEYS``; ``lists``: ``APIKeyEncryption.multi_key_list_fields``.
    """
    try:
        import api_key_encryption

        handler = api_key_encryption.get_handler()
        plain = list(getattr(handler, "api_key_fields", None) or api_key_encryption._NullHandler.api_key_fields)
        lists = list(api_key_encryption.APIKeyEncryption.multi_key_list_fields(handler))
    except Exception:
        plain, lists = ["api_key"], []
    paths = [(name,) for name in plain]
    try:
        import settings_schema

        paths += [tuple(spec.path) for spec in settings_schema.all_specs() if spec.type == "secret"]
    except Exception:
        pass
    paths += [(name,) for name in EXTRA_SECRET_KEYS]
    return tuple(dict.fromkeys(paths)), tuple(dict.fromkeys(lists))


def _get(config: Any, path: tuple) -> Any:
    node = config
    for part in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _set(config: dict, path: tuple, value: Any) -> None:
    node = config
    for part in path[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict) and path[-1] in node:
        node[path[-1]] = value


def _pop(config: dict, path: tuple) -> None:
    node = config
    for part in path[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _plain(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not value.startswith("ENC:")


def _plaintext_secrets(config: Mapping[str, Any], paths: tuple, lists: tuple) -> list:
    """The secret fields of ``config`` that still hold a plain-text value."""
    found = [".".join(path) for path in paths if _plain(_get(config, path))]
    for name in lists:
        entries = config.get(name)
        if isinstance(entries, list) and any(isinstance(e, dict) and _plain(e.get("api_key")) for e in entries):
            found.append(name)
    return found


def without_secrets(config: Mapping[str, Any]) -> dict:
    """A deep copy of ``config`` with every ``secret_fields`` value removed (pool entries keep their
    other fields)."""
    out = copy.deepcopy(dict(config or {}))
    paths, lists = secret_fields()
    for path in paths:
        _pop(out, path)
    for name in lists:
        entries = out.get(name)
        if isinstance(entries, list):
            out[name] = [{k: v for k, v in e.items() if k != "api_key"} if isinstance(e, dict) else e for e in entries]
    return out


def _derive_key(passphrase: str, salt: bytes, params: Mapping[str, Any]) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    kdf = Scrypt(salt=salt, length=int(params.get("length", 32)), n=int(params.get("n", _SCRYPT["n"])),
                 r=int(params.get("r", _SCRYPT["r"])), p=int(params.get("p", _SCRYPT["p"])))
    return base64.urlsafe_b64encode(kdf.derive(str(passphrase).encode("utf-8")))


def _handler(key: bytes) -> Any:
    """An ``APIKeyEncryption`` with the passphrase key (its field lists, its ENC: format)."""
    from cryptography.fernet import Fernet

    import api_key_encryption

    handler = api_key_encryption.APIKeyEncryption.__new__(api_key_encryption.APIKeyEncryption)
    handler.cipher = Fernet(key)
    handler.key_file = None
    try:  # the field list of the app's own handler (its __init__ list)
        handler.api_key_fields = list(getattr(api_key_encryption.get_handler(), "api_key_fields", None) or ["api_key"])
    except Exception:
        handler.api_key_fields = ["api_key"]
    return handler


def export_config(config: Mapping[str, Any], passphrase: str) -> dict:
    """The export document for a decrypted ``config`` (API keys re-encrypted with the passphrase)."""
    if len(str(passphrase or "")) < MIN_PASSPHRASE:
        raise ValueError(f"Use a passphrase of at least {MIN_PASSPHRASE} characters")
    salt = os.urandom(16)
    params = dict(_SCRYPT)
    handler = _handler(_derive_key(passphrase, salt, params))
    data = handler.encrypt_config(copy.deepcopy(dict(config or {})))
    paths, lists = secret_fields()
    for path in paths:
        value = _get(data, path)
        if _plain(value):
            _set(data, path, handler.encrypt_value(value))
    leftover = _plaintext_secrets(data, paths, lists)
    if leftover:  # encrypt_value hands the plain value back when encryption fails
        raise ValueError(f"Could not encrypt: {', '.join(leftover)}")
    return {
        "format": FORMAT,
        "version": VERSION,
        "kdf": {"name": "scrypt", "salt": base64.b64encode(salt).decode("ascii"), **params},
        "check": handler.encrypt_value(_CHECK_TEXT),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": data,
    }


def import_config(document: Mapping[str, Any], passphrase: str) -> dict:
    """The decrypted config of an export; ``WrongPassphrase`` / ``ValueError`` otherwise."""
    if not isinstance(document, Mapping) or document.get("format") != FORMAT:
        raise ValueError("This file is not a Glossarion config export")
    kdf = document.get("kdf") if isinstance(document.get("kdf"), Mapping) else {}
    try:
        salt = base64.b64decode(str(kdf.get("salt") or ""))
    except Exception as exc:
        raise ValueError("The export is damaged (salt)") from exc
    handler = _handler(_derive_key(passphrase, salt, kdf))
    if handler.decrypt_value(str(document.get("check") or "")) != _CHECK_TEXT:
        raise WrongPassphrase("Wrong passphrase")
    config = document.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("The export has no config")
    decrypted = handler.decrypt_config(copy.deepcopy(dict(config)))
    paths, lists = secret_fields()
    for path in paths:
        value = _get(decrypted, path)
        if isinstance(value, str) and value.startswith("ENC:"):
            _set(decrypted, path, handler.decrypt_value(value))
    leftover = [".".join(path) for path in paths if str(_get(decrypted, path) or "").startswith("ENC:")]
    leftover += [name for name in lists if isinstance(decrypted.get(name), list) and any(
        isinstance(e, dict) and str(e.get("api_key") or "").startswith("ENC:") for e in decrypted[name])]
    if leftover:
        raise ValueError(f"Could not decrypt: {', '.join(leftover)}")
    return decrypted


def write_export(path: str, document: Mapping[str, Any]) -> str:
    """Blocking: atomic write of an export document."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def read_export(path: str) -> dict:
    """Blocking: an export document from ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("This file is not a Glossarion config export")
    return data
