"""Which placeholders a mobile prompt editor marks, and what each says right now (UI_SPEC §5.5).

The mobile half of the desktop's placeholder links (owner 2026-11-14: "can you make it so mobile marks
placeholders?"). Every value comes from the GUI-free core the request itself uses; nothing here changes a
prompt, a setting or a request:

* glossary prompts (Balanced/Full, Minimal, the Parallel EPUB pair and manga glossary prompts):
  ``extract_glossary_from_epub.placeholder_value`` (the real ``build_prompt`` /
  ``GlossaryManager.build_minimal_system_prompt``) with the settings mapping the glossary job exports
  (``run_env.RunEnvMixin._glossary_env_mappings``, read from the live config with the schema defaults);
* glossary refinement prompts: ``glossary_refinement.refinement_placeholder_value``;
* the Single Pass Header Prompt: ``extract_glossary_from_epub.single_pass_header_placeholder_value``;
* the translation system prompt, refinement prompts and Assistant Prompt: ``prompt_defaults``
  (``translation_placeholder_value`` / ``assistant_placeholder_value``), ``{target_lang}`` = the output language;
* per-request inputs (summary, chunk, image/OCR, pair wrapper): ``prompt_defaults.PROMPT_INPUT_PLACEHOLDERS``.

A ``PlaceholderLinks`` is what one editor needs: ``names`` (the placeholders its request replaces; a
callable when the names come from a core that is slow to import, called off the UI loop) and
``value(name, prompt_text)`` (blocking; the editor runs it off the UI loop when the sheet opens). The
config source is read at that moment, so an unsaved entry type, toggle or language change shows at once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

__all__ = [
    "LiveConfig",
    "PlaceholderLinks",
    "assistant_links",
    "described_links",
    "glossary_links",
    "glossary_mode_for_key",
    "glossary_refinement_links",
    "glossary_settings",
    "links_for_setting",
    "refinement_links",
    "single_pass_links",
    "target_lang_links",
    "target_language",
    "translation_links",
]

log = logging.getLogger("glossarion.prompts")

_MISSING = object()


@dataclass(frozen=True)
class PlaceholderLinks:
    """The placeholders one prompt's request replaces and what each says now.

    ``names``: a collection, or a callable returning one (blocking: may import a core). ``value(name, text)``:
    blocking; returns a str (the expansion) or a value dict (``prompt_defaults.placeholder_value_parts``)."""

    names: Any
    value: Callable[[str, str], Any]

    def names_now(self) -> frozenset:
        names = self.names() if callable(self.names) else self.names
        return frozenset(str(n).strip("{}") for n in (names or ()))


class LiveConfig:
    """The live config as the request will see it: the stored value (``MobileConfigStore`` / a dict), else the
    schema default the store shows (``default_for``), else the caller's default. Read when asked, never cached."""

    def __init__(self, source: Any) -> None:
        self.source = source

    def _target(self) -> Any:
        source = self.source() if callable(self.source) and not hasattr(self.source, "get") else self.source
        for attr in ("store", "config"):  # a SettingsContext / a service holding the store
            inner = getattr(source, attr, None)
            if inner is not None and hasattr(inner, "get") and not isinstance(source, Mapping):
                return inner
        return source

    def get(self, key: str, default: Any = None, *, schema: bool = True) -> Any:
        """``schema=False``: the stored value or ``default`` (no schema display default)."""
        target = self._target()
        if target is None:
            return default
        try:
            value = target.get(key, _MISSING)
        except Exception:
            value = _MISSING
        if value is not _MISSING:
            return value
        if not schema:
            return default
        default_for = getattr(target, "default_for", None)
        if callable(default_for):
            try:
                schema_default = default_for(key)
            except Exception:
                schema_default = None
            if schema_default is not None:
                return schema_default
        return default

    def snapshot_store(self) -> Any:
        """The store-like object (``snapshot()``) for the shared profile listing, or None."""
        target = self._target()
        return target if callable(getattr(target, "snapshot", None)) else None


#: Keys an unset value of which must reach the env mapping as unset (its own fallback), not as the schema
#: display default: the schema data keeps dict defaults with sorted keys, while the job's owner seeds
#: ``custom_entry_types`` in the desktop order, which is also ``get_custom_entry_types``' own default for an
#: empty value. Same entry types either way; only this keeps their order (the order {entries},
#: {gender_instruction} and {relations_instruction} list them in).
_UNSET_REACHES_MAPPING = frozenset({"custom_entry_types"})


class _ConfigView(dict):
    """``self.config`` of the env-mapping stub: reads fall through to the live config, writes stay local."""

    def __init__(self, live: LiveConfig) -> None:
        super().__init__()
        self.live = live

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self.live.get(key, default, schema=key not in _UNSET_REACHES_MAPPING)

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return self.get(key, _MISSING) is not _MISSING


def _live(source: Any) -> LiveConfig:
    return source if isinstance(source, LiveConfig) else LiveConfig(source)


def target_language(source: Any) -> str:
    """What ``{target_lang}`` becomes: the output language (run_env / review_generator / local_inpainter)."""
    return str(_live(source).get("output_language", "English") or "English").strip() or "English"


def glossary_settings(source: Any) -> dict:
    """The glossary settings by ENV name, built the way the glossary job exports them
    (``RunEnvMixin._glossary_env_mappings`` over the live config)."""
    from run_env import RunEnvMixin  # shared, GUI-free

    class _Owner:
        pass

    owner = _Owner()
    owner.config = _ConfigView(_live(source))
    return {key: value for key, value in RunEnvMixin._glossary_env_mappings(owner)}


# ---- links per prompt ------------------------------------------------------------------------------------


def glossary_links(mode: str, source: Any) -> PlaceholderLinks:
    """Balanced/Full (``'balanced'``) or Minimal (``'minimal'``) glossary prompt: the placeholders its prompt
    builder replaces; the value is the builder's own output for the editor's text."""

    def names() -> Iterable[str]:
        from extract_glossary_from_epub import known_placeholder_names

        return known_placeholder_names(mode, tokens=False)

    def value(name: str, text: str) -> Any:
        from extract_glossary_from_epub import placeholder_value

        return placeholder_value(name, mode=mode, prompt=text, settings=glossary_settings(source))

    return PlaceholderLinks(names, value)


def glossary_refinement_links(source: Any) -> PlaceholderLinks:
    """Glossary refinement system / user prompt (``glossary_refinement._render_prompt_placeholders``)."""

    def names() -> Iterable[str]:
        from glossary_refinement import REFINEMENT_PROMPT_PLACEHOLDERS

        return REFINEMENT_PROMPT_PLACEHOLDERS

    def value(name: str, text: str) -> Any:
        from glossary_refinement import refinement_placeholder_value

        settings = glossary_settings(source)

        def parsed(key: str, empty: Any) -> Any:
            raw = settings.get(key)
            try:
                return json.loads(raw) if isinstance(raw, str) else (raw if raw is not None else empty)
            except ValueError:
                return empty

        live = _live(source)
        entry_types = parsed("GLOSSARY_CUSTOM_ENTRY_TYPES", {})
        if not entry_types:  # none configured: the run's get_custom_entry_types default
            from extract_glossary_from_epub import DEFAULT_CUSTOM_ENTRY_TYPES

            entry_types = DEFAULT_CUSTOM_ENTRY_TYPES
        return refinement_placeholder_value(
            name, custom_fields=parsed("GLOSSARY_CUSTOM_FIELDS", []),
            entry_types=entry_types,
            chunking_mode=settings.get("GLOSSARY_REFINEMENT_CHUNKING_MODE"),
            type_mode=settings.get("GLOSSARY_REFINEMENT_TYPE_MODE"),
            selected=list(live.get("glossary_refinement_selected_types", []) or []))

    return PlaceholderLinks(names, value)


def _active_system_prompt(source: Any) -> str:
    """The translation system prompt in use (the shared profile listing's active profile)."""
    store = _live(source).snapshot_store()
    if store is None:
        return ""
    from glossarion_mobile.ui.screens.profiles import ProfileService

    listing = ProfileService(store).listing()
    return str(listing.texts.get(listing.active, "") or "")


def single_pass_links(source: Any) -> PlaceholderLinks:
    """Single Pass Header Prompt (``build_single_pass_prompt``)."""

    def names() -> Iterable[str]:
        from extract_glossary_from_epub import SINGLE_PASS_HEADER_PLACEHOLDERS

        return SINGLE_PASS_HEADER_PLACEHOLDERS

    def value(name: str, text: str) -> Any:
        from extract_glossary_from_epub import single_pass_header_placeholder_value

        live = _live(source)
        if name == "glossary_prompt":
            return single_pass_header_placeholder_value(
                name, glossary_prompt=live.get("manual_glossary_prompt3", None) or None,
                settings=glossary_settings(live))
        return single_pass_header_placeholder_value(name, translation_prompt=_active_system_prompt(live),
                                                    target_lang=target_language(live))

    return PlaceholderLinks(names, value)


def translation_links(source: Any, *, names: Optional[Iterable[str]] = None,
                      target_lang: Optional[Callable[[], Any]] = None) -> PlaceholderLinks:
    """Translation system prompt (``{target_lang}``, ``{split_marker_instruction}``); ``target_lang`` overrides
    the language (a chat's own target language)."""
    from prompt_defaults import SYSTEM_PROMPT_PLACEHOLDERS

    def value(name: str, text: str) -> Any:
        from prompt_defaults import translation_placeholder_value

        live = _live(source)
        language = str(target_lang() or "").strip() if target_lang is not None else ""
        return translation_placeholder_value(name, target_lang=language or target_language(live),
                                             request_merging=bool(live.get("request_merging_enabled", False)))

    return PlaceholderLinks(frozenset(names) if names is not None else SYSTEM_PROMPT_PLACEHOLDERS, value)


def refinement_links(source: Any, *, user: bool) -> PlaceholderLinks:
    """Refinement mode system / user prompts (``_format_refinement_prompt``)."""
    from prompt_defaults import REFINEMENT_SYSTEM_PLACEHOLDERS, REFINEMENT_USER_PLACEHOLDERS

    return translation_links(source, names=REFINEMENT_USER_PLACEHOLDERS if user else REFINEMENT_SYSTEM_PLACEHOLDERS)


def assistant_links(source: Any) -> PlaceholderLinks:
    """Assistant Prompt (prefill): replaced in refinement requests only."""
    from prompt_defaults import REFINEMENT_SYSTEM_PLACEHOLDERS, assistant_placeholder_value

    return PlaceholderLinks(REFINEMENT_SYSTEM_PLACEHOLDERS,
                            lambda name, text: assistant_placeholder_value(name, target_lang=target_language(source)))


def target_lang_links(source: Any) -> PlaceholderLinks:
    """Prompts whose request fills only ``{target_lang}`` with the output language (review, metadata, chapter
    headers, book title, image-only title tags, rolling summary system prompt, custom image edit)."""
    return PlaceholderLinks(frozenset({"target_lang"}), lambda name, text: target_language(source))


def described_links(descriptions: Mapping[str, str]) -> PlaceholderLinks:
    """Placeholders filled with data of each request (``descriptions``: name -> what it holds)."""
    from prompt_defaults import described_placeholder_value

    table = {str(name).strip("{}"): text for name, text in dict(descriptions).items()}
    return PlaceholderLinks(frozenset(table), lambda name, text: described_placeholder_value(table, name))


#: Prompt settings (config keys) whose request fills only {target_lang} with the output language.
TARGET_LANG_KEYS = frozenset({
    "rolling_summary_system_prompt", "image_only_title_tag_system_prompt", "book_title_system_prompt",
    "book_title_prompt", "batch_header_system_prompt", "batch_header_prompt", "metadata_batch_prompt",
    "metadata_system_prompt", "review_system_prompt", "review_final_prompt", "custom_image_edit_system_prompt",
    "custom_image_edit_user_prompt",
})
#: Glossary prompt settings -> the prompt builder mode.
GLOSSARY_PROMPT_KEYS = {"manual_glossary_prompt3": "balanced", "manual_glossary_prompt": "balanced",
                        "manga_glossary_prompt": "balanced", "unified_auto_glosary_prompt3": "minimal"}
#: Refinement mode prompt pairs (TranslatorGUI's Refinement Prompt tabs): system -> user.
REFINEMENT_KEYS = {
    "refinement_system_prompt": "refinement_user_prompt",
    "refinement_full_with_raw_system_prompt": "refinement_full_with_raw_user_prompt",
    "refinement_failed_system_prompt": "refinement_failed_user_prompt",
    "refinement_partial_system_prompt": "refinement_partial_user_prompt",
    "refinement_partial_b_system_prompt": "refinement_partial_b_user_prompt",
    "refinement_partial_b2_system_prompt": "refinement_partial_b2_user_prompt",
}


def glossary_mode_for_key(key: str) -> Optional[str]:
    return GLOSSARY_PROMPT_KEYS.get(str(key or ""))


def links_for_setting(key: str, source: Any) -> Optional[PlaceholderLinks]:
    """The links of a prompt setting edited by its settings tile (None: its request replaces no placeholder)."""
    key = str(key or "")
    mode = glossary_mode_for_key(key)
    if mode is not None:
        return glossary_links(mode, source)
    if key in ("glossary_refinement_system_prompt", "glossary_refinement_user_prompt"):
        return glossary_refinement_links(source)
    if key == "single_pass_glossary_header_prompt":
        return single_pass_links(source)
    if key in REFINEMENT_KEYS:
        return refinement_links(source, user=False)
    if key in REFINEMENT_KEYS.values():
        return refinement_links(source, user=True)
    if key == "assistant_prompt":
        return assistant_links(source)
    if key == "system_prompt":  # the stored translation system prompt (Other › Stored)
        return translation_links(source)
    if key in TARGET_LANG_KEYS:
        return target_lang_links(source)
    try:
        from prompt_defaults import PROMPT_INPUT_PLACEHOLDERS
    except Exception:  # a build without the shared defaults: nothing to mark
        return None
    if key in PROMPT_INPUT_PLACEHOLDERS:
        return described_links(PROMPT_INPUT_PLACEHOLDERS[key])
    return None
