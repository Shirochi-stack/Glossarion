"""Direct Text rules the chat applies before a run (pure Python, no Flet, Python 3.10).

The desktop Direct Text dialog (``translator_gui._InputOutputDialog``) inherits its
chat logic from the shared GUI-free modules ``direct_text_store.ChatStoreMixin`` and
``direct_text_stream.DirectTextStreamMixin`` (U3). The rules below call those shared
implementations - attachment size, the glossary policy of one submission, the
rendered-card window, the card timestamp, auto-title, the supported attachment types
and token counting - so the chat and the desktop run the same code. The shared
modules are imported lazily (the backend loads off the UI loop; the warm import has
them in memory before the first send).

Since U7 the rules that were still inline in the dialog's Qt handlers are shared too
(``direct_text_store``: ``chat_rename_title`` = ``_rename_chat``,
``glossary_override_config_updates`` = ``_on_glossary_override_toggled``,
``configured_glossary_override_mode`` = the ``__init__`` read,
``manual_glossary_source_record`` / ``sniff_manual_glossary_extension`` = the "Provide
Manual Glossary" dialog's ``_accept``, ``GLOSSARY_OVERRIDE_MODES`` /
``MANUAL_GLOSSARY_EXTENSIONS``); the chat calls them. What stays here is mobile UI
(icons, Plan cards, token hint, long-output preview), the repaint cadence of the
dialog's Qt ``_schedule_stream_render`` override and the other settings reads of the
dialog ``__init__``; ``tests_host/test_chat.py`` compares those with the dialog source.

Nothing here touches the UI, ``os.environ`` or the backend pipeline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.ui.chat.output_modes import (
    IMAGE_ATTACHMENT_EXTENSIONS,
    VISION_ARCHIVE_ATTACHMENT_EXTENSIONS,
    normalize_mode,
)

__all__ = [
    "ATTACHMENT_PROMPT_ROLES",
    "DEFAULT_RENDERED_CARD_LIMIT",
    "DirectTextSettings",
    "effective_glossary_label",
    "GLOSSARY_OVERRIDE_LABELS",
    "GLOSSARY_OVERRIDE_MODES",
    "HISTORY_CHARACTER_BUDGET",
    "MANUAL_GLOSSARY_EXTENSIONS",
    "MAX_RENDERED_CARD_LIMIT",
    "MIN_RENDERED_CARD_LIMIT",
    "MOBILE_EXTRA_ATTACHMENT_EXTENSIONS",
    "ManualGlossarySource",
    "PLAN_EXTENSIONS",
    "PLAN_TEXT_THRESHOLD",
    "attachment_icon",
    "attachment_kind_label",
    "auto_title",
    "count_tokens",
    "display_markdown",
    "force_no_glossary_for_mode",
    "format_attachment_size",
    "glossary_override_updates",
    "history_window_bounds",
    "is_supported_attachment",
    "manual_glossary_source",
    "needs_plan",
    "normalize_glossary_override_mode",
    "normalize_rendered_card_limit",
    "rename_title",
    "split_long_output",
    "stream_render_interval_ms",
    "timestamp_label",
    "token_hint",
]

# ---------------------------------------------------------------------------
# Shared implementations (the desktop dialog's own code)
# ---------------------------------------------------------------------------


def _store() -> Any:
    import direct_text_store  # shared (U3): ChatStoreMixin + module helpers

    return direct_text_store


def _stream() -> Any:
    import direct_text_stream  # shared (U3): the log-stream model

    return direct_text_stream


# ---------------------------------------------------------------------------
# Attachments (desktop _is_supported_dropped_text_file / _format_attachment_size)
# ---------------------------------------------------------------------------

#: UI_SPEC §2.3: the chat also accepts the main-window inputs Direct Text lacks. The
#: regular pipeline handles them (ZIP -> EPUB, SDLXLIFF, video), so the run hands them to
#: it as-is (``prepare_direct_text_input(extra_pass_through_extensions=...)``; recorded
#: mobile divergence).
MOBILE_EXTRA_ATTACHMENT_EXTENSIONS = frozenset({".sdlxliff", ".zip", ".mp4"})

#: UI_SPEC §2.12.1: attachments that get a Plan card first (unless "Skip plan").
PLAN_EXTENSIONS = frozenset({".epub", ".pdf", ".cbz", ".zip", ".sdlxliff", ".srt", ".ass", ".lrc", ".vtt"})
PLAN_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
PLAN_TEXT_THRESHOLD = 20_000


def _extension(path: Any) -> str:
    return os.path.splitext(str(path or ""))[1].lower()


def is_supported_attachment(path: Any) -> bool:
    """The dialog's drop/attach filter (``_is_supported_dropped_text_file``) + the mobile extras."""
    if _extension(path) in MOBILE_EXTRA_ATTACHMENT_EXTENSIONS:
        return True
    return bool(_store().ChatStoreMixin._is_supported_dropped_text_file(path))


def format_attachment_size(size: Any) -> str:
    """``_format_attachment_size``: "12 B" / "1.5 KB" / "2.0 MB"."""
    return _store().format_attachment_size(size)


def attachment_kind_label(extension: str) -> str:
    """"EPUB" for ``.epub`` (desktop meta line ``EXT · size``)."""
    return str(extension or "").lstrip(".").upper() or "FILE"


def attachment_icon(extension: str) -> str:
    """Material icon name for a file chip / document card (UI_SPEC §2.3 item 1)."""
    ext = str(extension or "").lower()
    if ext == ".epub":
        return "MENU_BOOK"
    if ext == ".pdf":
        return "PICTURE_AS_PDF"
    if ext in IMAGE_ATTACHMENT_EXTENSIONS:
        return "PHOTO"
    if ext in VISION_ARCHIVE_ATTACHMENT_EXTENSIONS:
        return "COLLECTIONS"
    if ext in {".srt", ".ass", ".lrc", ".vtt"}:
        return "SUBTITLES"
    if ext == ".zip":
        return "FOLDER_ZIP"
    if ext == ".mp4":
        return "MOVIE"
    return "DESCRIPTION"


def needs_plan(extension: str, text_chars: int = 0, *, skip_plan: bool = False) -> bool:
    """True when sending this attachment shows a Plan card first (§2.12.1)."""
    if skip_plan:
        return False
    ext = str(extension or "").lower()
    if ext in PLAN_EXTENSIONS:
        return True
    return ext in PLAN_TEXT_EXTENSIONS and int(text_chars or 0) > PLAN_TEXT_THRESHOLD


# ---------------------------------------------------------------------------
# Glossary policy (desktop _force_no_glossary_for_mode / _on_glossary_override_toggled)
# ---------------------------------------------------------------------------

#: The desktop radio labels (Direct Text Settings tab).
GLOSSARY_OVERRIDE_LABELS = {
    "none": "No Override",
    "attachments_only": "No Override (Attachments Only)",
    "no_glossary": "Force No Glossary",
    "manual": "Force Manual Glossary",
}


def __getattr__(name: str) -> Any:
    """``GLOSSARY_OVERRIDE_MODES`` / ``MANUAL_GLOSSARY_EXTENSIONS``: the shared constants, read lazily
    (the backend loads off the UI loop; the sheets that use them import this module late)."""
    if name in ("GLOSSARY_OVERRIDE_MODES", "MANUAL_GLOSSARY_EXTENSIONS"):
        return getattr(_store(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_glossary_override_mode(mode: Any) -> str:
    """The dialog ``__init__`` read (``configured_glossary_override_mode``): unknown or empty values
    fall back to ``attachments_only``."""
    return str(_store().configured_glossary_override_mode(mode))


def force_no_glossary_for_mode(mode: Any, has_attachment: Any) -> bool:
    """``_force_no_glossary_for_mode``: does this submission run without a glossary?"""
    return bool(_store().force_no_glossary_for_mode(mode, has_attachment))


def effective_glossary_label(mode: Any, config_get: Callable[..., Any], has_attachment: bool = True) -> str:
    """The Plan card chip (UI_SPEC §2.12.1): the glossary mode this run really uses.

    The chat override wins ("Off", "Manual"); otherwise the run follows the main settings'
    auto glossary mode, resolved by the desktop rule (``RunEnvMixin._current_auto_glossary_mode``
    on the config: display names, the legacy ``enable_auto_glossary`` fallback) and titled like
    the Welcome glossary cards: "Glossary: Balanced (auto)".
    """
    if force_no_glossary_for_mode(mode, has_attachment):
        return "Glossary: Off"
    if normalize_glossary_override_mode(mode) == "manual":
        return "Glossary: Manual"
    config = {"auto_glossary_mode": config_get("auto_glossary_mode", None),
              "enable_auto_glossary": config_get("enable_auto_glossary", False)}
    try:
        from types import SimpleNamespace

        from run_env import RunEnvMixin  # shared (U2): the desktop's mode resolution

        key = RunEnvMixin._current_auto_glossary_mode(SimpleNamespace(config=config))
    except Exception:
        key = str(config["auto_glossary_mode"] or "off").strip().lower().replace(" ", "_")
    try:
        from glossarion_mobile.ui.screens.welcome_flow import GLOSSARY_MODE_CARDS

        title = next((card[2].title() for card in GLOSSARY_MODE_CARDS if card[0] == key), "")
    except Exception:
        title = ""
    return f"Glossary: {title or str(key).replace('_', ' ').title()} (auto)"


def glossary_override_updates(mode: Any) -> dict:
    """Config writes of the dialog's ``_on_glossary_override_toggled`` (``glossary_override_config_updates``:
    the enum + the two legacy booleans)."""
    return dict(_store().glossary_override_config_updates(mode))


@dataclass(frozen=True)
class ManualGlossarySource:
    """The desktop ``_request_direct_text_manual_glossary`` result record."""

    kind: str  # "path" | "content"
    path: str = ""
    content: str = ""
    extension: str = ".txt"

    def as_dict(self) -> dict:
        if self.kind == "path":
            return {"kind": "path", "path": self.path, "extension": self.extension}
        return {"kind": "content", "content": self.content, "extension": self.extension}


def manual_glossary_source(
    content: str,
    *,
    source_path: str = "",
    source_text: Optional[str] = None,
    source_extension: str = "",
) -> Optional[ManualGlossarySource]:
    """Accept logic of the desktop "Provide Manual Glossary" dialog (``manual_glossary_source_record``).

    A file loaded with Browse… and left unedited is used by path; edited or pasted
    contents are written by the run into its temp folder with a sniffed extension.
    Returns None when the box is empty ("Glossary required").
    """
    content = str(content or "")
    extension = (source_extension or os.path.splitext(str(source_path or ""))[1] or ".txt").lower()
    record = _store().manual_glossary_source_record(
        content, source_path or "", content if source_text is None else source_text, extension,
    )
    if record is None:
        return None
    if record.get("kind") == "path":
        return ManualGlossarySource("path", path=str(record.get("path") or ""), extension=str(record.get("extension") or extension))
    return ManualGlossarySource("content", content=str(record.get("content") or ""),
                                extension=str(record.get("extension") or ".txt"))


# ---------------------------------------------------------------------------
# Chat titles, rendered-card window, stream cadence
# ---------------------------------------------------------------------------


class _TitleHost:
    """The three members ``ChatStoreMixin._title_current_chat_from_text`` uses."""

    def __init__(self, title: Any) -> None:
        self.session = {"title": title}

    def _current_chat_session(self) -> dict:
        return self.session

    def _refresh_chat_list(self) -> None:
        pass

    def _schedule_chat_history_save(self) -> None:
        pass


def auto_title(current_title: Any, text: Any) -> Optional[str]:
    """The first send's auto-title (``_title_current_chat_from_text``); None keeps the current title."""
    host = _TitleHost(current_title)
    _store().ChatStoreMixin._title_current_chat_from_text(host, text)
    title = host.session.get("title")
    return None if title == current_title else title


def rename_title(text: Any) -> str:
    """The dialog's ``_rename_chat`` rule (``chat_rename_title``): whitespace collapsed, at most 120 characters."""
    return str(_store().chat_rename_title(text))


# The dialog's rendered-card limits (ChatStoreMixin._DEFAULT/_MIN/_MAX_RENDERED_CARD_LIMIT;
# test_chat checks they are equal) and its history character budget (dialog __init__).
DEFAULT_RENDERED_CARD_LIMIT = 20
MIN_RENDERED_CARD_LIMIT = 4
MAX_RENDERED_CARD_LIMIT = 200
HISTORY_CHARACTER_BUDGET = 120000


def normalize_rendered_card_limit(value: Any) -> int:
    """``_normalize_rendered_card_limit`` (4..200, default 20)."""
    return int(_store().normalize_rendered_card_limit(value))


def history_window_bounds(total: Any, limit: Any, focus_index: Any = None) -> tuple:
    """``_history_window_bounds``: the saved-message window containing ``focus_index``."""
    return tuple(_store().history_window_bounds(total, limit, focus_index))


def stream_render_interval_ms(active_characters: int, auto_scroll_disabled: bool = False) -> int:
    """The dialog's ``_schedule_stream_render`` cadence for the streaming tail (280-900 ms)."""
    interval = min(900, 280 + int(active_characters or 0) // 350)
    if auto_scroll_disabled:
        interval = max(interval, 450)
    return interval


# ---------------------------------------------------------------------------
# Direct Text settings (desktop dialog __init__ reads of the direct_text_* keys)
# ---------------------------------------------------------------------------

ATTACHMENT_PROMPT_ROLES = (("User", "user"), ("System", "system"), ("Assistant", "assistant"))
_ROLES = {value for _label, value in ATTACHMENT_PROMPT_ROLES}


@dataclass(frozen=True)
class DirectTextSettings:
    """Effective Direct Text settings for one chat (global keys, then chat overrides)."""

    attachment_prompt_role: str = "user"
    glossary_override_mode: str = "attachments_only"
    force_multipass_off: bool = True
    disable_thinking: bool = False
    skip_prompt_profile: bool = False
    disable_auto_scroll: bool = False
    rendered_card_limit: int = DEFAULT_RENDERED_CARD_LIMIT
    output_mode: str = "text"
    skip_plan: bool = False

    @classmethod
    def from_config(cls, get: Callable[[str, Any], Any]) -> "DirectTextSettings":
        """Read the global keys exactly as the desktop dialog initialises its widgets.

        ``get(key, default)`` returns the raw config.json value (MobileConfigStore.get).
        """
        legacy_simple_mode = bool(get('direct_text_force_simple_mode', True))
        skip_profile = get('direct_text_skip_prompt_profile', None)
        if skip_profile is None:
            skip_profile = bool(
                get('direct_text_skip_system_prompt_profile', False)
                or get('direct_text_skip_user_prompt_profile', False)
            )
        role = str(get('direct_text_attachment_prompt_role', 'user') or 'user').strip().lower()
        parent_mode = get('output_mode', 'text')
        configured_mode = str(get('direct_text_output_mode', parent_mode) or parent_mode or 'text').strip().lower()
        return cls(
            attachment_prompt_role=role if role in _ROLES else 'user',
            glossary_override_mode=normalize_glossary_override_mode(get('direct_text_glossary_override_mode', '')),
            force_multipass_off=bool(get('direct_text_force_multipass_off', legacy_simple_mode)),
            disable_thinking=bool(get('direct_text_disable_thinking', False)),
            skip_prompt_profile=bool(skip_profile),
            disable_auto_scroll=bool(get('direct_text_disable_auto_scroll', False)),
            rendered_card_limit=normalize_rendered_card_limit(
                get('direct_text_rendered_card_limit', DEFAULT_RENDERED_CARD_LIMIT)
            ),
            output_mode=normalize_mode(configured_mode),
        )

    def with_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "DirectTextSettings":
        """Apply per-chat overrides (sidecar ``overrides``; None/missing = inherit)."""
        if not overrides:
            return self
        values = dict(self.__dict__)
        for field_name in (
            "attachment_prompt_role", "glossary_override_mode", "force_multipass_off", "disable_thinking",
            "skip_prompt_profile", "disable_auto_scroll", "rendered_card_limit", "output_mode",
        ):
            if overrides.get(field_name) is not None:
                values[field_name] = overrides[field_name]
        if overrides.get("skip_plan") is not None:
            values["skip_plan"] = bool(overrides["skip_plan"])
        values["glossary_override_mode"] = normalize_glossary_override_mode(values["glossary_override_mode"])
        role = str(values["attachment_prompt_role"] or "user").strip().lower()
        values["attachment_prompt_role"] = role if role in _ROLES else "user"
        values["rendered_card_limit"] = normalize_rendered_card_limit(values["rendered_card_limit"])
        values["output_mode"] = normalize_mode(values["output_mode"])
        for flag in ("force_multipass_off", "disable_thinking", "skip_prompt_profile", "disable_auto_scroll"):
            values[flag] = bool(values[flag])
        return DirectTextSettings(**values)

    def config_updates(self) -> dict:
        """Global keys for these values (the desktop persists one key per control)."""
        updates = {
            "direct_text_attachment_prompt_role": self.attachment_prompt_role,
            "direct_text_force_multipass_off": self.force_multipass_off,
            "direct_text_disable_thinking": self.disable_thinking,
            "direct_text_skip_prompt_profile": self.skip_prompt_profile,
            "direct_text_disable_auto_scroll": self.disable_auto_scroll,
            "direct_text_rendered_card_limit": self.rendered_card_limit,
            "direct_text_output_mode": self.output_mode,
        }
        updates.update(glossary_override_updates(self.glossary_override_mode))
        return updates


# ---------------------------------------------------------------------------
# Card header timestamp and content display
# ---------------------------------------------------------------------------


def timestamp_label(created_at: Any) -> str:
    """``_assistant_timestamp_label``: "HH:MM" today, "Mon DD · HH:MM" this year, else with the year."""
    return _store().timestamp_label(created_at)


_HTML_HINT = re.compile(
    r"<(?:!doctype\s+html|html\b|head\b|body\b|p\b|div\b|br\b|h[1-6]\b|ul\b|ol\b|table\b|span\b|em\b|strong\b)",
    re.IGNORECASE,
)


def display_markdown(source: Any) -> str:
    """Markdown for flet ``Markdown`` from a response (Markdown or raw HTML).

    UI_SPEC §2.9: HTML/XHTML responses are sanitised with the dialog's ``_markup_to_html``
    (``direct_text_store.markup_to_html``) and converted back to Markdown with html2text;
    plain Markdown is shown as-is.
    """
    text = str(source or "").replace("\r\n", "\n").replace("\r", "\n")
    if not _HTML_HINT.search(text):
        return text
    try:
        html_source = _store().markup_to_html(text)
    except Exception:
        html_source = text
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_images = False
        converter.ignore_links = False
        return converter.handle(html_source).strip()
    except Exception:
        return re.sub(r"<[^>]+>", "", html_source)


LONG_OUTPUT_CHARS = 6000
LONG_OUTPUT_LINES = 60
LONG_OUTPUT_PREVIEW_LINES = 40


def split_long_output(text: Any) -> tuple:
    """``(preview, truncated)``: outputs over 6,000 chars or 60 lines show their first ~40 lines (§2.9)."""
    value = str(text or "")
    lines = value.split("\n")
    if len(value) <= LONG_OUTPUT_CHARS and len(lines) <= LONG_OUTPUT_LINES:
        return value, False
    preview = "\n".join(lines[:LONG_OUTPUT_PREVIEW_LINES])
    if len(preview) > LONG_OUTPUT_CHARS:
        preview = preview[:LONG_OUTPUT_CHARS]
    return preview, True


# ---------------------------------------------------------------------------
# Token counting (desktop _get_token_encoder / _count_tokens)
# ---------------------------------------------------------------------------


def count_tokens(text: Any, model_name: str = "") -> int:
    """Tokens in ``text`` for ``model_name`` (the dialog's ``_count_tokens``: tiktoken model ->
    o200k_base -> cl100k_base; 0 without tiktoken). Blocking: run off the loop."""
    return int(_stream().count_tokens(text, model_name) or 0)


TOKEN_HINT_MIN_CHARS = 200


def token_hint(count: int) -> str:
    """Composer token hint: "≈1.2k tok" / "≈850 tok" (UI_SPEC §2.3)."""
    if count <= 0:
        return ""
    if count >= 1000:
        return f"≈{count / 1000:.1f}k tok"
    return f"≈{count} tok"
