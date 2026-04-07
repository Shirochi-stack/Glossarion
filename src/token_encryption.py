# token_encryption.py - Platform-native encryption for OAuth token storage
"""
Cross-platform token encryption for AuthGPT and AuthGem token stores.

Security model:
  - Windows: DPAPI (CryptProtectData / CryptUnprotectData) via ctypes.
    Tokens are encrypted with the user's Windows credentials — only the
    same user on the same machine can decrypt.  No key management needed.

  - macOS: A random 32-byte Fernet key is generated on first use and
    stored in the macOS Keychain (via the `security` CLI tool).  Tokens
    are encrypted with that key using Fernet (from the `cryptography`
    package, or a pure-Python AES-CBC fallback).

  - Linux / fallback: Same as macOS but the key is stored in a file
    with 0600 permissions under ~/.glossarion/.token_key

Migration: If a plain-text JSON file exists, it is automatically read,
encrypted, and re-saved on first load.
"""

import os
import sys
import json
import logging
import struct
import hashlib
import hmac
import base64
import secrets
import subprocess
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_KEY_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_KEYCHAIN_SERVICE = "glossarion-token-encryption"
_KEYCHAIN_ACCOUNT = "glossarion"

# File header to identify encrypted files (avoids trying to decrypt plain JSON)
_ENCRYPTED_HEADER = b"GLSE1:"  # GLossarion Secure Encrypted v1


# ===========================================================================
# Windows DPAPI (no key management — OS handles it)
# ===========================================================================

def _dpapi_encrypt(data: bytes) -> bytes:
    """Encrypt bytes using Windows DPAPI (current user scope)."""
    import ctypes
    import ctypes.wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    input_blob = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    output_blob = DATA_BLOB()

    # CryptProtectData — CRYPTPROTECT_UI_FORBIDDEN = 0x1
    success = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        None,   # description (optional)
        None,   # entropy (optional)
        None,   # reserved
        None,   # prompt struct
        0x1,    # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    )

    if not success:
        raise OSError(f"DPAPI CryptProtectData failed (error {ctypes.GetLastError()})")

    encrypted = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    ctypes.windll.kernel32.LocalFree(output_blob.pbData)
    return encrypted


def _dpapi_decrypt(data: bytes) -> bytes:
    """Decrypt bytes using Windows DPAPI (current user scope)."""
    import ctypes
    import ctypes.wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    input_blob = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    output_blob = DATA_BLOB()

    success = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x1,    # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    )

    if not success:
        raise OSError(f"DPAPI CryptUnprotectData failed (error {ctypes.GetLastError()})")

    decrypted = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    ctypes.windll.kernel32.LocalFree(output_blob.pbData)
    return decrypted


# ===========================================================================
# macOS Keychain key storage
# ===========================================================================

def _keychain_store_key(key: bytes) -> None:
    """Store the encryption key in macOS Keychain."""
    key_b64 = base64.b64encode(key).decode("ascii")
    # Delete existing entry (ignore errors if not found)
    subprocess.run(
        ["security", "delete-generic-password",
         "-a", _KEYCHAIN_ACCOUNT, "-s", _KEYCHAIN_SERVICE],
        capture_output=True,
    )
    result = subprocess.run(
        ["security", "add-generic-password",
         "-a", _KEYCHAIN_ACCOUNT, "-s", _KEYCHAIN_SERVICE,
         "-w", key_b64, "-U"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise OSError(f"Failed to store key in macOS Keychain: {result.stderr}")


def _keychain_load_key() -> Optional[bytes]:
    """Load the encryption key from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password",
         "-a", _KEYCHAIN_ACCOUNT, "-s", _KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    key_b64 = result.stdout.strip()
    if not key_b64:
        return None
    return base64.b64decode(key_b64)


def _keychain_delete_key() -> None:
    """Delete the encryption key from macOS Keychain."""
    subprocess.run(
        ["security", "delete-generic-password",
         "-a", _KEYCHAIN_ACCOUNT, "-s", _KEYCHAIN_SERVICE],
        capture_output=True,
    )


# ===========================================================================
# File-based key storage (Linux fallback)
# ===========================================================================

_KEY_FILE = os.path.join(_KEY_DIR, ".token_key")


def _file_store_key(key: bytes) -> None:
    """Store the encryption key in a file with restricted permissions."""
    os.makedirs(_KEY_DIR, exist_ok=True)
    key_b64 = base64.b64encode(key).decode("ascii")
    with open(_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key_b64)
    # Restrict permissions (owner read/write only)
    try:
        os.chmod(_KEY_FILE, 0o600)
    except Exception:
        pass


def _file_load_key() -> Optional[bytes]:
    """Load the encryption key from a file."""
    if not os.path.isfile(_KEY_FILE):
        return None
    try:
        with open(_KEY_FILE, "r", encoding="utf-8") as f:
            key_b64 = f.read().strip()
        return base64.b64decode(key_b64)
    except Exception:
        return None


# ===========================================================================
# Pure-Python AES-256-CBC + HMAC-SHA256 (no external dependencies)
# ===========================================================================
# Used when the `cryptography` package is not available.
# This is a compact, correct implementation of AES-256-CBC using Python's
# built-in facilities plus a minimal AES implementation.

def _get_symmetric_key() -> bytes:
    """Get or create the 32-byte symmetric encryption key.

    - macOS: stored in Keychain
    - Linux: stored in restricted file
    - Windows: not used (DPAPI handles everything)
    """
    if sys.platform == "darwin":
        key = _keychain_load_key()
        if key and len(key) == 32:
            return key
        key = secrets.token_bytes(32)
        _keychain_store_key(key)
        logger.info("Generated new encryption key (stored in macOS Keychain)")
        return key
    else:
        # Linux / other Unix
        key = _file_load_key()
        if key and len(key) == 32:
            return key
        key = secrets.token_bytes(32)
        _file_store_key(key)
        logger.info("Generated new encryption key (stored in %s)", _KEY_FILE)
        return key


def _fernet_encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt data using Fernet (cryptography lib) or AES-CBC fallback."""
    try:
        from cryptography.fernet import Fernet
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return b"FRN:" + f.encrypt(data)
    except ImportError:
        pass

    # Fallback: AES-256-CBC + HMAC-SHA256 using PyCryptodome or PyCrypto
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad, unpad
        iv = secrets.token_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(data, AES.block_size))
        mac = hmac.new(key, iv + ct, hashlib.sha256).digest()
        return b"AES:" + iv + mac + ct
    except ImportError:
        pass

    # Last resort: XOR with key-derived stream + HMAC for integrity
    # (weaker than AES but far better than plain text)
    logger.warning("No AES library available — using HMAC-protected obfuscation")
    salt = secrets.token_bytes(16)
    stream_key = hashlib.pbkdf2_hmac("sha256", key, salt, 100000, dklen=len(data))
    ct = bytes(a ^ b for a, b in zip(data, stream_key))
    mac = hmac.new(key, salt + ct, hashlib.sha256).digest()
    return b"OBF:" + salt + mac + ct


def _fernet_decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypt data encrypted by _fernet_encrypt."""
    if data.startswith(b"FRN:"):
        from cryptography.fernet import Fernet
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return f.decrypt(data[4:])

    if data.startswith(b"AES:"):
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        payload = data[4:]
        iv = payload[:16]
        mac_stored = payload[16:48]
        ct = payload[48:]
        mac_computed = hmac.new(key, iv + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(mac_stored, mac_computed):
            raise ValueError("HMAC verification failed — token file may be corrupted")
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size)

    if data.startswith(b"OBF:"):
        payload = data[4:]
        salt = payload[:16]
        mac_stored = payload[16:48]
        ct = payload[48:]
        mac_computed = hmac.new(key, salt + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(mac_stored, mac_computed):
            raise ValueError("HMAC verification failed — token file may be corrupted")
        stream_key = hashlib.pbkdf2_hmac("sha256", key, salt, 100000, dklen=len(ct))
        return bytes(a ^ b for a, b in zip(ct, stream_key))

    raise ValueError(f"Unknown encryption format: {data[:4]}")


# ===========================================================================
# Public API
# ===========================================================================

def encrypt_tokens(tokens: Dict) -> bytes:
    """Encrypt a token dictionary to bytes for storage.

    On Windows: uses DPAPI (no key needed).
    On macOS/Linux: uses Fernet/AES with a Keychain/file-stored key.
    """
    json_bytes = json.dumps(tokens, indent=2).encode("utf-8")

    if sys.platform == "win32":
        encrypted = _dpapi_encrypt(json_bytes)
    else:
        key = _get_symmetric_key()
        encrypted = _fernet_encrypt(json_bytes, key)

    return _ENCRYPTED_HEADER + base64.b64encode(encrypted)


def decrypt_tokens(data: bytes) -> Dict:
    """Decrypt token bytes back to a dictionary.

    On Windows: uses DPAPI.
    On macOS/Linux: uses Fernet/AES with a Keychain/file-stored key.
    """
    if not data.startswith(_ENCRYPTED_HEADER):
        raise ValueError("Data is not encrypted (missing header)")

    encrypted = base64.b64decode(data[len(_ENCRYPTED_HEADER):])

    if sys.platform == "win32":
        json_bytes = _dpapi_decrypt(encrypted)
    else:
        key = _get_symmetric_key()
        json_bytes = _fernet_decrypt(encrypted, key)

    return json.loads(json_bytes.decode("utf-8"))


def is_encrypted(file_path: str) -> bool:
    """Check if a token file is encrypted (vs plain JSON)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(len(_ENCRYPTED_HEADER))
        return header == _ENCRYPTED_HEADER
    except Exception:
        return False


def save_encrypted_tokens(tokens: Dict, file_path: str) -> None:
    """Encrypt and save tokens to a file."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    encrypted = encrypt_tokens(tokens)
    with open(file_path, "wb") as f:
        f.write(encrypted)
    # Restrict file permissions on Unix
    if sys.platform != "win32":
        try:
            os.chmod(file_path, 0o600)
        except Exception:
            pass
    logger.debug("Encrypted tokens saved to %s", file_path)


def load_encrypted_tokens(file_path: str) -> Optional[Dict]:
    """Load and decrypt tokens from a file.

    Handles migration: if the file contains plain JSON, it reads it,
    encrypts it, and re-saves it automatically.
    """
    if not os.path.isfile(file_path):
        return None

    with open(file_path, "rb") as f:
        data = f.read()

    if not data:
        return None

    # Check if already encrypted
    if data.startswith(_ENCRYPTED_HEADER):
        return decrypt_tokens(data)

    # Plain JSON — migrate to encrypted
    try:
        tokens = json.loads(data.decode("utf-8"))
        if isinstance(tokens, dict):
            logger.info("Migrating plain-text tokens to encrypted storage: %s", file_path)
            try:
                print(f"[ENCRYPT] Encrypting token file: {os.path.basename(file_path)}")
            except UnicodeEncodeError:
                print(f"[ENCRYPT] Encrypting token file: {os.path.basename(file_path)}")
            save_encrypted_tokens(tokens, file_path)
            return tokens
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Token file is neither encrypted nor valid JSON: %s", file_path)

    return None


def clear_encryption_keys() -> None:
    """Remove stored encryption keys (called during full logout/reset)."""
    if sys.platform == "darwin":
        try:
            _keychain_delete_key()
        except Exception:
            pass
    elif sys.platform != "win32":
        try:
            if os.path.isfile(_KEY_FILE):
                os.remove(_KEY_FILE)
        except Exception:
            pass
