"""Transparent workaround for the Windows 32,767-character environment variable limit.

On Windows, individual environment variables cannot exceed 32,767 characters
(including the null terminator).  This module provides an in-memory fallback
store for oversized values, with helpers that mirror os.environ's interface.

Usage
-----
    import large_env

    # Writing (replaces os.environ[key] = value)
    large_env.set_env('SYSTEM_PROMPT', very_long_string)

    # Bulk writing (replaces os.environ.update(mapping))
    large_env.update_env(env_dict)

    # Reading (replaces os.getenv / os.environ.get)
    val = large_env.get_env('SYSTEM_PROMPT', '')

    # Cleanup
    large_env.clear_store()
"""

import os
import sys

# ---------------------------------------------------------------------------
# In-memory fallback store for values that exceed the Windows limit.
# Keys stored here are *not* present in os.environ (or set to '').
# ---------------------------------------------------------------------------
_store: dict[str, str] = {}

# Windows limit is 32,767 chars including the null terminator.
# Use a conservative safe threshold so we never flirt with the boundary.
_MAX_ENV_LEN = 32_000


def set_env(key: str, value) -> None:
    """Set an environment variable, falling back to in-memory store if too large."""
    value = str(value) if value is not None else ''
    if sys.platform == 'win32' and len(value) > _MAX_ENV_LEN:
        _store[key] = value
        # Set a short empty marker so nothing downstream blows up on KeyError.
        try:
            os.environ[key] = ''
        except (ValueError, OSError):
            pass
    else:
        # Value fits – remove any stale in-memory entry and use os.environ.
        _store.pop(key, None)
        os.environ[key] = value


def get_env(key: str, default=None) -> str | None:
    """Get an environment variable, checking the in-memory store first."""
    if key in _store:
        return _store[key]
    return os.environ.get(key, default)


def update_env(mapping: dict) -> None:
    """Like ``os.environ.update()``, but routes oversized values through the store."""
    for key, value in mapping.items():
        set_env(key, value)


def clear_store() -> None:
    """Clear the in-memory fallback store (call during env-restore / cleanup)."""
    _store.clear()
