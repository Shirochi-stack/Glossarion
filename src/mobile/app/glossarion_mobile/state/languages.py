"""Target languages: the shared desktop list plus the user's own (U11 item 9).

Translations go through API requests, so any language a model can write works: the user may add
names the desktop list (``language_options.TARGET_LANGUAGES``) lacks. Added names live in Prefs
(``mobile_state.json``, never config.json) under ``CUSTOM_LANGUAGES_PREF`` and are listed after the
built-in ones wherever a target language is picked (ModelSheet Language tab, Chat settings, Welcome).
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["CUSTOM_LANGUAGES_PREF", "MAX_CUSTOM", "base_languages", "custom_languages", "remember_language",
           "target_languages"]

CUSTOM_LANGUAGES_PREF = "custom_target_languages"
MAX_CUSTOM = 50


def base_languages() -> tuple:
    try:
        from language_options import TARGET_LANGUAGES

        return tuple(TARGET_LANGUAGES)
    except Exception:
        return ("English",)


def _clean(name: Any) -> str:
    return " ".join(str(name or "").split())[:60]


def custom_languages(prefs: Any) -> list:
    if prefs is None:
        return []
    try:
        value = prefs.get(CUSTOM_LANGUAGES_PREF, [])
    except Exception:
        return []
    return [n for n in (_clean(v) for v in value) if n] if isinstance(value, list) else []


def target_languages(prefs: Any = None, base: Iterable[str] = ()) -> tuple:
    """The built-in list (``base`` or the shared desktop one) followed by the user's added names."""
    names = list(base) or list(base_languages())
    seen = {n.casefold() for n in names}
    for name in custom_languages(prefs):
        if name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    return tuple(names)


def remember_language(prefs: Any, name: Any, base: Iterable[str] = ()) -> bool:
    """Keep ``name`` in the user's list when it is not a built-in language. True when it was added."""
    name = _clean(name)
    if prefs is None or not name:
        return False
    known = {n.casefold() for n in (list(base) or list(base_languages()))}
    custom = custom_languages(prefs)
    if name.casefold() in known or name.casefold() in {n.casefold() for n in custom}:
        return False
    try:
        prefs.set(CUSTOM_LANGUAGES_PREF, (custom + [name])[-MAX_CUSTOM:])
    except Exception:
        return False
    return True
