"""Routes a phone cannot run itself but can use on the user's PC (U13).

``antigravity/`` needs the Antigravity proxy and ``ollamapull/`` an Ollama server; both install and start
programs on the desktop. On mobile the user points the route at the one running on their PC. The address
lives in Prefs (``mobile_state.json``, never config.json) and reaches the shared route module as an
environment variable at start-up and whenever it is saved.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["REMOTE_ROUTES", "REMOTE_SECRETS", "apply_all", "apply_secret", "apply_url", "has_secret", "normalize_url",
           "saved_url"]

#: route id -> (Prefs key, environment variable the shared route module reads)
REMOTE_ROUTES = {
    "antigravity": ("antigravity_proxy_url", "ANTIGRAVITY_PROXY_URL"),
    "ollamapull": ("ollamapull_base_url", "OLLAMAPULL_BASE_URL"),
    "authza": ("authza_proxy_url", "GLM_PROXY_URL"),
}
#: route id -> (Prefs key of the encrypted value, environment variable): the PC proxy's own API key
REMOTE_SECRETS = {
    "authza": ("authza_proxy_key", "GLM_PROXY_API_KEY"),
}


def _box() -> Any:
    from glossarion_mobile.services.share_links import SecretBox  # the app's ENC: codec (SecureStorage key)

    return SecretBox()


def has_secret(prefs: Any, route: str) -> bool:
    try:
        return bool(prefs.get(REMOTE_SECRETS[route][0], "")) if prefs is not None else False
    except Exception:
        return False


def apply_secret(prefs: Any, route: str, value: Any = None, *, save: bool = False) -> bool:
    """Set the route's secret (``value`` given; stored ``ENC:`` only, never plain) or re-apply the saved one.
    True when the environment now holds a value."""
    key, env = REMOTE_SECRETS[route]
    plain = ""
    try:
        if value is not None:
            plain = str(value or "").strip()
            if save and prefs is not None:
                prefs.set(key, _box().seal(plain) if plain else "")
        elif prefs is not None and prefs.get(key, ""):
            plain = _box().open(prefs.get(key, ""))
    except Exception:
        plain = ""
    if plain:
        os.environ[env] = plain
    else:
        os.environ.pop(env, None)
    return bool(plain)


def normalize_url(value: Any) -> str:
    url = str(value or "").strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


def saved_url(prefs: Any, route: str) -> str:
    key = REMOTE_ROUTES[route][0]
    try:
        return str(prefs.get(key, "") or "") if prefs is not None else ""
    except Exception:
        return ""


def apply_url(prefs: Any, route: str, url: Any = None, *, save: bool = False) -> str:
    """Set (``url`` given: save it too when ``save``) or re-apply the route's address; returns it."""
    key, env = REMOTE_ROUTES[route]
    value = saved_url(prefs, route) if url is None else normalize_url(url)
    if save and prefs is not None:
        try:
            prefs.set(key, value)
        except Exception:
            pass
    if value:
        os.environ[env] = value
    else:
        os.environ.pop(env, None)
    return value


def apply_all(prefs: Any) -> None:
    for route in REMOTE_ROUTES:
        apply_url(prefs, route)
    for route in REMOTE_SECRETS:
        apply_secret(prefs, route)
