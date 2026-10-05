"""direct_text_store: Direct Text chat persistence, output folders and attachments (ChatStoreMixin).

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). The desktop Direct Text
dialog (``translator_gui._InputOutputDialog``) kept its chat persistence inside Qt
handlers; those members moved here verbatim (``translator_gui.py`` @ 1719fb59) and the
dialog inherits ``ChatStoreMixin`` (with ``direct_text_stream.DirectTextStreamMixin``)
ahead of ``QDialog``, so its behaviour and rendering are unchanged. ``ChatStore`` is the
GUI-free host the mobile chat uses; ``direct_text_stream.DirectTextStream`` composes both
mixins for one run.

Moved (legacy line ranges at 1719fb59):

* ``direct_text_chats.json`` v2: ``_new_chat_session`` 3330, ``_direct_response_timestamp``,
  ``_resolve_chat_history_path``, ``_load_chat_history`` 3360-3488, ``_normalize_message_storage``,
  ``_history_file_reference`` / ``_resolve_history_file_reference``, ``_save_chat_history``
  4427-4474 (atomic), ``_externalize_session_messages`` 4343-4425 (bodies in
  ``<chat>/Chat Messages/NNNNNN-response.{md,txt,html,xhtml}`` + ``-thinking.md``),
  ``_write_response_files`` 4269-4341, ``_normalize_attachment_record``, lazy bodies
  ``_assistant_message_text`` (+ char counts), timestamps, request numbering;
* generated media references (``[GENERATED_IMAGE|VIDEO|AUDIO:<path>]``) 3556-3651,
  3864-3957 and ``_markup_to_html`` 10417-10551 (the sanitiser);
* output folders 11983-12471: ``Direct Text/<safe title> - <ts>_<uuid8>``, ``Direct Text N``
  (``next_output_index``), attachment trees ``Attachments/<stem>`` (copy + glossary sync),
  output discovery and ``_persist_output_folder`` (temp fallback);
* attachment workspaces + Migrate with path rewrite 6836-7277, the validated delete folder,
  ``_remember_output_folder``, auto-title, response edits 7918-8140 (write-back to the
  real chapter file through ``translation_progress.json``), ``_assistant_source_is_attachment``
  9928-9946;
* class constants: output/Direct Text env keys, media extensions, rendered-card limits.

Edits made while moving (everything else is byte-for-byte):

* ``_InputOutputDialog.<name>`` -> ``ChatStoreMixin.<name>`` (explicit class references);
* ``_migrate_conversation_attachment``: ``QMessageBox.information/warning`` are the
  ``_direct_text_notice(level, parent, title, text)`` hook and the "Attachment folder
  already exists" dialog is the ``_confirm_attachment_merge(parent, target)`` hook (the
  dialog implements both with the original Qt code);
* new: ``_attachment_card_actions`` (the "Attachment actions" card's link decision,
  formerly inline in ``_render_output``), ``_init_chat_sessions`` (the chat-loading block
  of the dialog ``__init__``, 2091-2108, minus the UI flag ``_switching_chat``) and
  ``_prepare_direct_text_input`` (the temp-input block of ``_start_translation``,
  9185-9262, dedented; it imports ``uuid``/``datetime`` itself and returns the run's
  manual glossary path);
* ``_prepare_direct_text_input`` reads two class knobs whose defaults are the dialog's
  code (U3 Integrate): ``mkdtemp(..., dir=_DIRECT_TEXT_TEMP_PARENT)`` (None = the OS temp
  dir) and ``_EXTRA_PASS_THROUGH_EXTENSIONS`` (empty); only ``prepare_direct_text_input``
  (mobile) sets them, on its own throwaway host;
* ``_effective_run_glossary_path`` reads ``MANUAL_GLOSSARY`` from the ``_RUN_ENVIRONMENT``
  knob (U3 fix pass; None = the live ``os.environ``, the dialog's code); only the GUI-free
  ``DirectTextStream.load_run`` sets it, to the values the mobile job recorded;
* ``apply_direct_text_run_environment`` is the run-environment block of
  ``_start_translation`` (9339-9349) with the dialog state as parameters;
* ``_atomic_text_write`` (translator_gui 1321) lives here; translator_gui re-imports it.

Hooks the moved code calls (the dialog implements them with Qt; ``ChatStore`` GUI-free):
``_refresh_chat_list``, ``_render_output``, ``_set_status``, ``_direct_text_notice``,
``_confirm_attachment_merge`` (see ``CHAT_STORE_HOOKS``).

Data format: unchanged (v2, written only through these methods); mobile-only data goes to
the ``direct_text_chats.mobile.json`` sidecar (``SIDECAR_NAME``), never into this file.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import contextlib
import json
import os
import re
import sys
import tempfile
import threading

from app_paths import CONFIG_FILE, _atomic_json_write, _get_app_dir
from translation_pipeline import IMAGE_ATTACHMENT_EXTENSIONS

__all__ = [
    "CHAT_STORE_HOOKS",
    "ChatStore",
    "ChatStoreMixin",
    "DirectTextOwnerView",
    "HISTORY_FILE_NAME",
    "SIDECAR_NAME",
    "apply_direct_text_run_environment",
    "default_history_path",
    "force_no_glossary_for_mode",
    "format_attachment_size",
    "history_window_bounds",
    "markup_to_html",
    "normalize_attachment_record",
    "normalize_rendered_card_limit",
    "prepare_direct_text_input",
    "timestamp_label",
]

HISTORY_FILE_NAME = "direct_text_chats.json"
#: Mobile-only per-chat data (pins, overrides, versions, plans); desktop never reads it.
SIDECAR_NAME = "direct_text_chats.mobile.json"

#: Dialog hooks the moved code calls (implemented by _InputOutputDialog and ChatStore).
CHAT_STORE_HOOKS = (
    "_refresh_chat_list",
    "_render_output",
    "_set_status",
    "_direct_text_notice",
    "_confirm_attachment_merge",
)


def _atomic_text_write(filepath, text):
    """Atomically replace a UTF-8 text file without exposing partial content."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8', newline='') as handle:
            handle.write(str(text or ""))
        os.replace(tmp_path, filepath)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class ChatStoreMixin:
    """Direct Text chat persistence moved verbatim from ``_InputOutputDialog``."""

    _OUTPUT_ENV_KEYS = ('OUTPUT_DIRECTORY', 'OUTPUT_DIR', 'EPUB_OUTPUT_DIR')
    _VIDEO_OUTPUT_EXTENSIONS = {
        '.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.mpeg', '.mpg',
        '.ogv', '.wmv', '.3gp',
    }
    _AUDIO_OUTPUT_EXTENSIONS = {
        '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma',
        '.aiff', '.aif', '.oga', '.pcm',
    }
    _VISION_ARCHIVE_ATTACHMENT_EXTENSIONS = {'.cbz'}
    _DIRECT_TEXT_ENV_KEYS = (
        'DIRECT_TEXT_ACTIVE',
        'DIRECT_TEXT_PRESERVE_MARKUP',
        'OUTPUT_MODE',
        'VISION_OCR_FIRST',
        'ENABLE_IMAGE_TRANSLATION',
        'ENABLE_IMAGE_OUTPUT_MODE',
        'ENABLE_VIDEO_OUTPUT_MODE',
        'ENABLE_AUDIO_OUTPUT_MODE',
        'ENABLE_REFINEMENT_OUTPUT_MODE',
        'MULTIPASS_MODE',
        'AUTO_GLOSSARY_MODE',
        'ENABLE_AUTO_GLOSSARY',
        'SINGLE_PASS_GLOSSARY_MODE',
        'FUZZY_AUTO_MAPPING',
        'APPEND_GLOSSARY',
        'MANUAL_GLOSSARY',
        'DEFER_GLOSSARY_APPEND',
        'ENABLE_ANTHROPIC_THINKING',
        'ANTHROPIC_THINKING_BUDGET',
        'ANTHROPIC_FORCE_ADAPTIVE',
        'ENABLE_GEMINI_THINKING',
        'ENABLE_DEEPSEEK_THINKING',
        'DEEPSEEK_USE_RESPONSES_API',
        'ENABLE_GPT_THINKING',
        'GPT_REASONING_TOKENS',
        'GPT_EFFORT',
        'OPENROUTER_USE_REASONING_TOKENS',
        'PASS_THINKING_TO_OPENAI_COMPATIBLE',
        'FORCE_SERVICE_TIER_UNKNOWN_ROUTES',
        'GEMINI_THINKING_LEVEL',
        'GEMINI_SERVICE_TIER',
        'THINKING_BUDGET',
        'STREAM_THINKING_LOGS',
        'AUTHND_STREAM_THINKING_LOGS',
        'ENABLE_THOUGHTS',
        'DIRECT_TEXT_ATTACHMENT_PROMPT',
        'DIRECT_TEXT_ATTACHMENT_PROMPT_ROLE',
        'DIRECT_TEXT_PROFILE_USER_PROMPT',
        'DIRECT_TEXT_SKIP_PROMPT_PROFILE',
        'DIRECT_TEXT_ORDERED_BATCH',
        'ORDER_BATCH_REQUESTS_BY_SPINE',
        'SYSTEM_PROMPT_TO_USER',
    )
    _DEFAULT_RENDERED_CARD_LIMIT = 20
    _MIN_RENDERED_CARD_LIMIT = 4
    _MAX_RENDERED_CARD_LIMIT = 200
    # One set with translation_pipeline (the dialog keeps the same alias).
    _IMAGE_ATTACHMENT_EXTENSIONS = IMAGE_ATTACHMENT_EXTENSIONS
    # The parent folder of a run's temp root (None: the OS temp dir, as on desktop) and the
    # attachment types a run hands to the pipeline as-is besides the desktop ones; only the
    # mobile chat sets them, per call, through prepare_direct_text_input().
    _DIRECT_TEXT_TEMP_PARENT = None
    _EXTRA_PASS_THROUGH_EXTENSIONS = frozenset()
    # The run environment _effective_run_glossary_path reads MANUAL_GLOSSARY from (None: the
    # live os.environ, as on desktop, where the dialog finishes before its run context is
    # restored). The GUI-free DirectTextStream that finishes a mobile chat run off the job
    # thread sets the values the job recorded (DirectTextStream.load_run), so it never reads
    # the environment of the next queued job.
    _RUN_ENVIRONMENT = None

    @classmethod
    def _normalize_rendered_card_limit(cls, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = cls._DEFAULT_RENDERED_CARD_LIMIT
        return max(
            cls._MIN_RENDERED_CARD_LIMIT,
            min(cls._MAX_RENDERED_CARD_LIMIT, value),
        )

    @staticmethod
    def _history_window_bounds(total, limit, focus_index=None):
        """Return a bounded saved-message window containing ``focus_index``."""
        total = max(0, int(total or 0))
        limit = max(1, int(limit or 1))
        if total <= limit:
            return (0, total)
        if focus_index is None:
            return (total - limit, total)
        focus_index = max(0, min(int(focus_index), total - 1))
        start = max(0, focus_index - (limit // 2))
        end = min(total, start + limit)
        start = max(0, end - limit)
        return (start, end)

    @staticmethod
    def _new_chat_session(session_id):
        """Return one clean, serializable Direct Text conversation record."""
        return {
            "id": int(session_id),
            "title": "New chat",
            "messages": [],
            "draft": "",
            "attachment": None,
            "output_folder": "",
            "output_folder_name": "",
            "next_output_index": 1,
            "expanded": set(),
        }

    @staticmethod
    def _direct_response_timestamp():
        """Return a compact, timezone-aware creation time for persisted cards."""
        from datetime import datetime

        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _resolve_chat_history_path():
        """Locate the durable Direct Text history file beside app configuration."""
        override = os.environ.get("GLOSSARION_DIRECT_TEXT_HISTORY")
        if override:
            return os.path.abspath(os.path.expanduser(str(override)))
        return os.path.join(os.path.dirname(CONFIG_FILE), "direct_text_chats.json")

    def _load_chat_history(self):
        """Load and normalize saved chats without trusting malformed JSON fields."""
        history_path = self._chat_history_path
        if not os.path.isfile(history_path):
            return [self._new_chat_session(1)], 1
        try:
            with open(history_path, 'r', encoding='utf-8') as history_file:
                payload = json.load(history_file)
            raw_sessions = payload.get("sessions", [])
            current_chat_id = payload.get("current_chat_id")
            sessions = []
            used_ids = set()
            next_fallback_id = 1
            for raw_session in raw_sessions:
                if not isinstance(raw_session, dict):
                    continue
                try:
                    session_id = max(1, int(raw_session.get("id", 0)))
                except (TypeError, ValueError):
                    session_id = 0
                while session_id <= 0 or session_id in used_ids:
                    session_id = next_fallback_id
                    next_fallback_id += 1
                used_ids.add(session_id)
                next_fallback_id = max(next_fallback_id, session_id + 1)

                messages = []
                for raw_message in raw_session.get("messages", []):
                    if not isinstance(raw_message, (list, tuple)) or len(raw_message) < 2:
                        continue
                    role = str(raw_message[0] or "")
                    content = str(raw_message[1] or "")
                    if role == "user":
                        messages.append(("user", content))
                    elif role == "user_file":
                        source_path = str(
                            raw_message[2] if len(raw_message) > 2 else ""
                        )
                        try:
                            source_size = max(
                                0, int(raw_message[3] if len(raw_message) > 3 else 0)
                            )
                        except (TypeError, ValueError):
                            source_size = 0
                        attachment_prompt = str(
                            raw_message[4] if len(raw_message) > 4 else ""
                        )
                        attachment_prompt_role = str(
                            raw_message[5] if len(raw_message) > 5 else "user"
                        ).strip().lower()
                        if attachment_prompt_role not in {
                            'system', 'assistant', 'user'
                        }:
                            attachment_prompt_role = 'user'
                        messages.append(
                            (
                                "user_file",
                                content,
                                source_path,
                                source_size,
                                attachment_prompt,
                                attachment_prompt_role,
                            )
                        )
                    elif role == "assistant":
                        thinking = str(
                            raw_message[2] if len(raw_message) > 2 else ""
                        )
                        storage = self._normalize_message_storage(
                            raw_message[6] if len(raw_message) > 6 else None
                        )
                        messages.append(
                            (
                                "assistant",
                                content,
                                thinking,
                                str(raw_message[3] if len(raw_message) > 3 else "Processing"),
                                str(raw_message[4] if len(raw_message) > 4 else ""),
                                str(raw_message[5] if len(raw_message) > 5 else ""),
                                storage,
                            )
                        )

                try:
                    expanded = {
                        int(value)
                        for value in raw_session.get("expanded", [])
                        if int(value) >= 0
                    }
                except (TypeError, ValueError):
                    expanded = set()
                try:
                    next_output_index = max(
                        1, int(raw_session.get("next_output_index", 1))
                    )
                except (TypeError, ValueError):
                    next_output_index = 1
                title = " ".join(
                    str(raw_session.get("title", "New chat") or "New chat").split()
                )[:120]
                sessions.append(
                    {
                        "id": session_id,
                        "title": title or "New chat",
                        "messages": messages,
                        "draft": str(raw_session.get("draft", "") or ""),
                        "attachment": self._normalize_attachment_record(
                            raw_session.get("attachment")
                        ),
                        "output_folder": str(
                            raw_session.get("output_folder", "") or ""
                        ),
                        "output_folder_name": str(
                            raw_session.get("output_folder_name", "") or ""
                        ),
                        "next_output_index": next_output_index,
                        "expanded": expanded,
                    }
                )
            if not sessions:
                return [self._new_chat_session(1)], 1
            try:
                current_chat_id = int(current_chat_id)
            except (TypeError, ValueError):
                current_chat_id = sessions[0]["id"]
            return sessions, current_chat_id
        except Exception as exc:
            print(f"[Direct Text] Could not load saved chats: {exc}")
            return [self._new_chat_session(1)], 1

    def _schedule_chat_history_save(self):
        """Debounce draft/history writes while the user is typing."""
        timer = getattr(self, "_chat_history_save_timer", None)
        if timer is not None:
            timer.start()

    @staticmethod
    def _normalize_message_storage(storage):
        """Return the safe, JSON-serializable metadata for a file-backed reply."""
        if not isinstance(storage, dict):
            return {}
        normalized = {}
        for key in (
            "content_path",
            "content_text_path",
            "content_html_path",
            "content_xhtml_path",
            "thinking_path",
            "image_path",
            "media_path",
            "media_kind",
        ):
            value = str(storage.get(key, "") or "").strip()
            if value:
                normalized[key] = value
        for key in ("content_chars", "thinking_chars"):
            try:
                normalized[key] = max(0, int(storage.get(key, 0) or 0))
            except (TypeError, ValueError):
                normalized[key] = 0
        created_at = str(storage.get("created_at", "") or "").strip()
        if created_at:
            normalized["created_at"] = created_at[:64]
        return normalized

    def _history_file_reference(self, path):
        """Store a portable path relative to the history JSON when possible."""
        absolute = os.path.abspath(os.path.expanduser(str(path or "")))
        history_directory = os.path.dirname(os.path.abspath(self._chat_history_path))
        try:
            reference = os.path.relpath(absolute, history_directory)
        except ValueError:
            reference = absolute
        return reference.replace('\\', '/')

    def _resolve_history_file_reference(self, reference):
        """Resolve either a v2 relative reference or an absolute fallback path."""
        value = str(reference or "").strip()
        if not value:
            return ""
        value = value.replace('/', os.sep).replace('\\', os.sep)
        if os.path.isabs(value):
            return os.path.abspath(value)
        history_directory = os.path.dirname(os.path.abspath(self._chat_history_path))
        return os.path.abspath(os.path.join(history_directory, value))

    @staticmethod
    def _assistant_storage_for(message):
        if (
            isinstance(message, (list, tuple))
            and len(message) > 6
            and isinstance(message[6], dict)
        ):
            return message[6]
        return {}

    @classmethod
    def _generated_image_paths_from_text(cls, content):
        """Extract generated-image sentinel paths without treating normal text as a path."""
        return [
            path
            for kind, path in cls._generated_media_references_from_text(content)
            if kind == 'image'
        ]

    @classmethod
    def _media_kind_for_path(cls, path, marker_kind=''):
        """Classify one generated artifact by extension, then marker type."""
        extension = os.path.splitext(str(path or ''))[1].lower()
        if extension in cls._IMAGE_ATTACHMENT_EXTENSIONS:
            return 'image'
        if extension in cls._VIDEO_OUTPUT_EXTENSIONS:
            return 'video'
        if extension in cls._AUDIO_OUTPUT_EXTENSIONS:
            return 'audio'
        marker_kind = str(marker_kind or '').strip().lower()
        return marker_kind if marker_kind in {'image', 'video', 'audio'} else ''

    @classmethod
    def _generated_media_references_from_text(cls, content):
        """Extract typed image/video/audio sentinel paths from response text."""
        references = []
        for match in re.finditer(
            r'\[GENERATED_(IMAGE|VIDEO|AUDIO):(.+?)\]',
            str(content or ''),
            flags=re.IGNORECASE,
        ):
            marker_kind = str(match.group(1) or '').strip().lower()
            value = str(match.group(2) or '').strip().strip('"\'')
            if not value:
                continue
            path = os.path.abspath(os.path.expanduser(value))
            kind = cls._media_kind_for_path(path, marker_kind)
            if kind:
                references.append((kind, path))
        return references

    def _assistant_generated_media(self, message, content=''):
        """Resolve the durable generated media associated with one response."""
        storage = self._assistant_storage_for(message)
        reference = str(storage.get('media_path', '') or '').strip()
        stored_kind = str(storage.get('media_kind', '') or '').strip().lower()
        if reference:
            candidate = self._resolve_history_file_reference(reference)
            kind = self._media_kind_for_path(candidate, stored_kind)
            if os.path.isfile(candidate) and kind:
                return kind, candidate

        # Image-only histories saved before media_path was introduced remain
        # fully compatible.
        image_reference = str(storage.get('image_path', '') or '').strip()
        if image_reference:
            candidate = self._resolve_history_file_reference(image_reference)
            if (
                os.path.isfile(candidate)
                and self._media_kind_for_path(candidate) == 'image'
            ):
                return 'image', candidate

        for kind, candidate in self._generated_media_references_from_text(content):
            if os.path.isfile(candidate):
                return kind, candidate
        return '', ''

    def _assistant_generated_image_path(self, message, content=''):
        """Resolve the durable image associated with one assistant response."""
        kind, path = self._assistant_generated_media(message, content)
        return path if kind == 'image' else ''

    @staticmethod
    def _is_generated_image_only_content(content):
        """Return whether a response consists solely of generated-image markers."""
        source = str(content or '')
        if not re.search(
            r'\[GENERATED_IMAGE:(.+?)\]', source, flags=re.IGNORECASE
        ):
            return False
        remainder = re.sub(
            r'\[GENERATED_IMAGE:(.+?)\]', '', source, flags=re.IGNORECASE
        )
        return not remainder.strip()

    @staticmethod
    def _is_generated_media_only_content(content):
        """Return whether a response consists solely of generated-media markers."""
        source = str(content or '')
        marker_pattern = r'\[GENERATED_(?:IMAGE|VIDEO|AUDIO):(.+?)\]'
        if not re.search(marker_pattern, source, flags=re.IGNORECASE):
            return False
        return not re.sub(
            marker_pattern, '', source, flags=re.IGNORECASE
        ).strip()

    def _promote_generated_media_reference(self, persistent_path):
        """Point the live response at a persistent Direct Text media artifact."""
        persistent_path = os.path.abspath(str(persistent_path or ''))
        media_kind = ChatStoreMixin._media_kind_for_path(persistent_path)
        if not os.path.isfile(persistent_path) or not media_kind:
            return False

        marker = f"[GENERATED_{media_kind.upper()}:{persistent_path}]"
        marker_pattern = r'\[GENERATED_(?:IMAGE|VIDEO|AUDIO):(.+?)\]'
        stream_promoted = False
        segment_promoted = False

        if re.search(marker_pattern, self._streamed_content, flags=re.IGNORECASE):
            self._streamed_content = re.sub(
                marker_pattern,
                lambda _match: marker,
                self._streamed_content,
                count=1,
                flags=re.IGNORECASE,
            )
            stream_promoted = True

        for segment in self._active_request_segments:
            segment_content = str(segment.get('content', '') or '')
            if re.search(marker_pattern, segment_content, flags=re.IGNORECASE):
                segment['content'] = re.sub(
                    marker_pattern,
                    lambda _match: marker,
                    segment_content,
                    count=1,
                    flags=re.IGNORECASE,
                )
                segment['media_path'] = persistent_path
                segment['media_kind'] = media_kind
                if media_kind == 'image':
                    segment['image_path'] = persistent_path
                segment_promoted = True

        if self._active_request_segments and not segment_promoted:
            segment = self._active_request_segments[-1]
            segment_content = str(segment.get('content', '') or '').rstrip()
            segment['content'] = (
                f"{segment_content}\n\n{marker}" if segment_content else marker
            )
            segment['media_path'] = persistent_path
            segment['media_kind'] = media_kind
            if media_kind == 'image':
                segment['image_path'] = persistent_path
        if not stream_promoted:
            streamed = str(self._streamed_content or '').rstrip()
            self._streamed_content = f"{streamed}\n\n{marker}" if streamed else marker

        self._expected_output = persistent_path
        self._rendered_message_cache.clear()
        return True

    def _promote_generated_image_reference(self, persistent_path):
        """Point the live response at its persistent Direct Text image artifact."""
        return ChatStoreMixin._promote_generated_media_reference(
            self, persistent_path
        )

    def _response_generated_image_path(self, message_index):
        """Return one saved response's renderable generated-image path."""
        try:
            message_index = int(message_index)
            message = self._chat_messages[message_index]
            if message[0] != 'assistant':
                return ''
        except (IndexError, TypeError, ValueError):
            return ''
        content = self._assistant_message_text(
            message, 'content', message_index
        )
        return self._assistant_generated_image_path(message, content)

    def _response_generated_media(self, message_index):
        """Return one saved response's generated media kind and local path."""
        try:
            message_index = int(message_index)
            message = self._chat_messages[message_index]
            if message[0] != 'assistant':
                return '', ''
        except (IndexError, TypeError, ValueError):
            return '', ''
        content = self._assistant_message_text(
            message, 'content', message_index
        )
        return self._assistant_generated_media(message, content)

    def _assistant_message_with_timestamp(self, message, created_at=""):
        """Return an assistant tuple with durable creation metadata."""
        values = list(message[:6])
        while len(values) < 6:
            values.append("")
        storage = self._normalize_message_storage(
            message[6] if len(message) > 6 else None
        )
        if not storage.get("created_at"):
            storage["created_at"] = (
                str(created_at or "").strip()
                or self._direct_response_timestamp()
            )
        if not storage.get("image_path"):
            image_paths = self._generated_image_paths_from_text(values[1])
            if image_paths and os.path.isfile(image_paths[0]):
                storage["image_path"] = self._history_file_reference(
                    image_paths[0]
                )
        if not storage.get("media_path"):
            media_references = self._generated_media_references_from_text(
                values[1]
            )
            if media_references and os.path.isfile(media_references[0][1]):
                media_kind, media_path = media_references[0]
                storage["media_path"] = self._history_file_reference(
                    media_path
                )
                storage["media_kind"] = media_kind
        return tuple(values + [storage])

    def _assistant_timestamp_label(self, message):
        """Format a response timestamp unobtrusively for the card header."""
        from datetime import datetime

        raw = str(
            self._assistant_storage_for(message).get("created_at", "") or ""
        ).strip()
        if not raw:
            return ""
        try:
            created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if created.tzinfo is not None:
                created = created.astimezone()
            now = datetime.now().astimezone()
            if created.date() == now.date():
                return created.strftime("%H:%M")
            if created.year == now.year:
                return created.strftime("%b %d · %H:%M")
            return created.strftime("%Y %b %d · %H:%M")
        except (TypeError, ValueError):
            return ""

    def _next_conversation_request_number(self):
        """Return the next request number across the entire active chat."""
        import re

        request_count = 0
        for message in self._chat_messages:
            if (
                not isinstance(message, (list, tuple))
                or not message
                or str(message[0] or "") != "assistant"
                or len(message) <= 5
            ):
                continue
            if re.search(
                r"\bRequest\s+\d+\b",
                str(message[5] or ""),
                flags=re.IGNORECASE,
            ):
                request_count += 1
        return request_count + 1

    def _assistant_message_char_count(self, message, kind="content"):
        inline_index = 2 if kind == "thinking" else 1
        inline = str(message[inline_index] if len(message) > inline_index else "")
        if inline:
            return len(inline)
        storage = self._assistant_storage_for(message)
        try:
            return max(0, int(storage.get(f"{kind}_chars", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _assistant_message_has_text(self, message, kind="content"):
        inline_index = 2 if kind == "thinking" else 1
        if str(message[inline_index] if len(message) > inline_index else "").strip():
            return True
        storage = self._assistant_storage_for(message)
        return bool(
            storage.get(f"{kind}_path")
            or self._assistant_message_char_count(message, kind) > 0
        )

    def _assistant_message_text(self, message, kind="content", message_index=-1):
        """Lazily load one response or thinking stream from its backing file."""
        inline_index = 2 if kind == "thinking" else 1
        inline = str(message[inline_index] if len(message) > inline_index else "")
        if inline:
            return inline
        storage = self._assistant_storage_for(message)
        reference = str(storage.get(f"{kind}_path", "") or "")
        if not reference:
            return ""
        session = self._current_chat_session()
        session_id = session.get("id") if session is not None else None
        cache_key = (session_id, int(message_index), kind, reference)
        cached = self._message_text_cache.get(cache_key)
        if cached is not None:
            return cached
        path = self._resolve_history_file_reference(reference)
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                value = handle.read()
        except Exception:
            label = "thinking log" if kind == "thinking" else "response"
            value = f"*The saved {label} file is missing or unreadable.*"
        if len(self._message_text_cache) >= 128:
            self._message_text_cache.clear()
        self._message_text_cache[cache_key] = value
        return value

    def _response_html_document(self, source):
        """Return a standalone HTML document matching one editable response."""
        import re

        value = str(source or "").replace("\r\n", "\n").replace("\r", "\n")
        if re.search(
            r"<(?:!doctype\s+html|html\b|head\b|body\b)",
            value,
            flags=re.IGNORECASE,
        ):
            return value
        rendered = self._markup_to_html(value)
        return (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset=\"utf-8\"></head><body>\n"
            f"{rendered}\n"
            "</body></html>\n"
        )

    def _response_xhtml_document(self, source):
        """Return an XML-serialized XHTML counterpart for one response."""
        html_document = self._response_html_document(source)
        try:
            from lxml import etree
            from lxml import html as lxml_html

            root = lxml_html.document_fromstring(html_document)
            root.set("xmlns", "http://www.w3.org/1999/xhtml")
            serialized = etree.tostring(
                root,
                encoding="unicode",
                method="xml",
                pretty_print=True,
            )
            return (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<!DOCTYPE html>\n'
                f"{serialized}\n"
            )
        except Exception:
            import re

            namespaced = re.sub(
                r"<html(\s|>)",
                r'<html xmlns="http://www.w3.org/1999/xhtml"\1',
                html_document,
                count=1,
                flags=re.IGNORECASE,
            )
            return '<?xml version="1.0" encoding="utf-8"?>\n' + namespaced

    def _write_response_files(
        self,
        markdown_path,
        source,
        *,
        text_source=None,
        html_source=None,
        output_artifact_path=None,
    ):
        """Update an editable response and its translated output atomically."""
        markdown_path = os.path.abspath(str(markdown_path or ""))
        response_stem = os.path.splitext(markdown_path)[0]
        text_path = response_stem + ".txt"
        html_path = response_stem + ".html"
        xhtml_path = response_stem + ".xhtml"
        markdown_source = str(source or "")
        plain_text_source = (
            markdown_source if text_source is None else str(text_source or "")
        )
        html_document = self._response_html_document(
            markdown_source if html_source is None else html_source
        )
        xhtml_document = self._response_xhtml_document(html_document)

        targets = [
            (markdown_path, markdown_source),
            (text_path, plain_text_source),
            (html_path, html_document),
            (xhtml_path, xhtml_document),
        ]
        artifact_path = os.path.abspath(str(output_artifact_path or ""))
        if artifact_path and artifact_path not in {path for path, _value in targets}:
            artifact_extension = os.path.splitext(artifact_path)[1].casefold()
            artifact_value = {
                ".html": html_document,
                ".htm": html_document,
                ".xhtml": xhtml_document,
                ".xhtm": xhtml_document,
                ".txt": plain_text_source,
                ".md": markdown_source,
                ".markdown": markdown_source,
            }.get(artifact_extension)
            if artifact_value is not None:
                targets.append((artifact_path, artifact_value))
        previous = {}
        for path, _value in targets:
            existed = os.path.isfile(path)
            old_value = None
            if existed:
                try:
                    with open(path, 'r', encoding='utf-8') as handle:
                        old_value = handle.read()
                except OSError:
                    old_value = None
            previous[path] = (existed, old_value)

        try:
            for path, value in targets:
                _atomic_text_write(path, value)
        except Exception:
            # Best-effort rollback keeps all four representations synchronized
            # if one target cannot be replaced.
            for path, _value in targets:
                existed, old_value = previous[path]
                try:
                    if existed and old_value is not None:
                        _atomic_text_write(path, old_value)
                    elif not existed and os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass
            raise
        return text_path, html_path, xhtml_path

    def _externalize_session_messages(self, session):
        """Move assistant bodies into files and return their compact JSON records."""
        messages = list(session.get("messages", []))
        serialized = []
        normalized_messages = []
        message_directory = ""
        for message_index, message in enumerate(messages):
            if not isinstance(message, (list, tuple)) or not message:
                continue
            if str(message[0] or "") != "assistant":
                normalized = tuple(message)
                normalized_messages.append(normalized)
                serialized.append(list(normalized))
                continue

            values = list(message[:6])
            while len(values) < 6:
                values.append("")
            values[0] = "assistant"
            values[1] = str(values[1] or "")
            values[2] = str(values[2] or "")
            values[3] = str(values[3] or "Processing")
            values[4] = str(values[4] or "")
            values[5] = str(values[5] or "")
            storage = self._normalize_message_storage(
                message[6] if len(message) > 6 else None
            )

            for kind, inline_index, suffix in (
                ("content", 1, "response.md"),
                ("thinking", 2, "thinking.md"),
            ):
                inline_text = values[inline_index]
                if not inline_text:
                    continue
                path_key = f"{kind}_path"
                try:
                    if not message_directory:
                        conversation_folder = (
                            self._ensure_conversation_output_folder_for_session(session)
                        )
                        message_directory = os.path.join(
                            conversation_folder, "Chat Messages"
                        )
                    # Newly completed bodies are written to our managed
                    # directory instead of trusting any path that may have been
                    # hand-edited into the history JSON.
                    target_path = os.path.join(
                        message_directory,
                        f"{message_index + 1:06d}-{suffix}",
                    )
                    if kind == "content":
                        text_path, html_path, xhtml_path = self._write_response_files(
                            target_path, inline_text
                        )
                        storage["content_text_path"] = (
                            self._history_file_reference(text_path)
                        )
                        storage["content_html_path"] = (
                            self._history_file_reference(html_path)
                        )
                        storage["content_xhtml_path"] = (
                            self._history_file_reference(xhtml_path)
                        )
                    else:
                        _atomic_text_write(target_path, inline_text)
                    storage[path_key] = self._history_file_reference(target_path)
                    storage[f"{kind}_chars"] = len(inline_text)
                    values[inline_index] = ""
                except Exception as exc:
                    # Keep this one body inline if its file could not be safely
                    # written; the next save will retry without losing data.
                    print(
                        f"[Direct Text] Could not externalize {kind} for chat "
                        f"{session.get('id')}, message {message_index + 1}: {exc}"
                    )

            normalized = tuple(values + [storage])
            normalized_messages.append(normalized)
            serialized.append(list(normalized))

        session["messages"] = normalized_messages
        return serialized

    def _save_chat_history(self):
        """Atomically persist all retained Direct Text conversations."""
        try:
            current = self._current_chat_session()
            if current is not None and hasattr(self, "input_box"):
                current["draft"] = self.input_box.toPlainText()
                current["attachment"] = self._pending_attachment
                current["expanded"] = set(self._expanded_processing_messages)
            sessions = []
            for session in self._chat_sessions:
                serialized_messages = self._externalize_session_messages(session)
                sessions.append(
                    {
                        "id": int(session.get("id", 0)),
                        "title": str(session.get("title", "New chat") or "New chat"),
                        "messages": serialized_messages,
                        "draft": str(session.get("draft", "") or ""),
                        "attachment": self._normalize_attachment_record(
                            session.get("attachment")
                        ),
                        "output_folder": str(session.get("output_folder", "") or ""),
                        "output_folder_name": str(
                            session.get("output_folder_name", "") or ""
                        ),
                        "next_output_index": max(
                            1, int(session.get("next_output_index", 1))
                        ),
                        "expanded": sorted(
                            int(value) for value in session.get("expanded", set())
                        ),
                    }
                )
            if current is not None:
                self._chat_messages = current["messages"]
            current_chat_id = current.get("id") if current is not None else None
            history_directory = os.path.dirname(self._chat_history_path)
            if history_directory:
                os.makedirs(history_directory, exist_ok=True)
            _atomic_json_write(
                self._chat_history_path,
                {
                    "version": 2,
                    "current_chat_id": current_chat_id,
                    "sessions": sessions,
                },
            )
        except Exception as exc:
            print(f"[Direct Text] Could not save chats: {exc}")

    @staticmethod
    def _normalize_attachment_record(record):
        """Return a portable JSON-safe attachment record, or ``None``."""
        if not isinstance(record, dict):
            return None
        path = os.path.abspath(os.path.expanduser(str(record.get("path", "") or "")))
        if not path:
            return None
        name = str(record.get("name", "") or os.path.basename(path))
        extension = os.path.splitext(name or path)[1].lower()
        try:
            size = max(0, int(record.get("size", 0) or 0))
        except (TypeError, ValueError):
            size = 0
        return {
            "path": path,
            "name": name or os.path.basename(path),
            "extension": extension,
            "size": size,
        }

    @staticmethod
    def _force_no_glossary_for_mode(mode, has_attachment):
        """Resolve a persisted glossary policy for one submission."""
        normalized = str(mode or 'attachments_only').strip().lower()
        return normalized == 'no_glossary' or (
            normalized == 'attachments_only' and not bool(has_attachment)
        )

    @staticmethod
    def _is_supported_dropped_text_file(path):
        extension = os.path.splitext(str(path or ''))[1].lower()
        return extension in {
            '.txt', '.epub', '.pdf', '.md', '.markdown', '.html', '.htm',
            '.xhtml', '.xml', '.json', '.csv', '.tsv', '.srt', '.ass', '.lrc', '.vtt', '.log',
        } or extension in (
            ChatStoreMixin._IMAGE_ATTACHMENT_EXTENSIONS
            | ChatStoreMixin._VISION_ARCHIVE_ATTACHMENT_EXTENSIONS
        )

    @classmethod
    def _is_image_attachment(cls, path):
        return (
            os.path.splitext(str(path or ''))[1].lower()
            in cls._IMAGE_ATTACHMENT_EXTENSIONS
        )

    @classmethod
    def _is_vision_attachment(cls, path):
        extension = os.path.splitext(str(path or ''))[1].lower()
        return extension in (
            cls._IMAGE_ATTACHMENT_EXTENSIONS
            | cls._VISION_ARCHIVE_ATTACHMENT_EXTENSIONS
        )

    @staticmethod
    def _format_attachment_size(size):
        try:
            value = max(0, int(size))
        except (TypeError, ValueError):
            value = 0
        if value < 1024:
            return f"{value} B"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value / (1024 * 1024):.1f} MB"

    def _current_chat_session(self):
        index = int(getattr(self, "_current_chat_index", -1))
        if 0 <= index < len(self._chat_sessions):
            return self._chat_sessions[index]
        return None

    @staticmethod
    def _validated_chat_output_folder(session):
        """Resolve an exact, recognized conversation subfolder for deletion."""
        folder = os.path.realpath(
            os.path.abspath(str(session.get("output_folder", "") or ""))
        )
        if not folder or not os.path.isdir(folder):
            raise ValueError("The conversation output folder no longer exists.")

        parent = os.path.dirname(folder)
        basename = os.path.basename(folder)
        if not basename or folder == parent:
            raise ValueError("Refusing to delete a filesystem root.")

        protected_paths = {
            os.path.normcase(os.path.realpath(os.getcwd())),
            os.path.normcase(os.path.realpath(os.path.expanduser("~"))),
            os.path.normcase(os.path.realpath(_get_app_dir())),
        }
        if os.path.normcase(folder) in protected_paths:
            raise ValueError("Refusing to delete a protected application folder.")

        expected_name = str(session.get("output_folder_name", "") or "")
        is_direct_text_subfolder = (
            os.path.basename(parent).casefold() == "direct text"
            and (not expected_name or basename == expected_name)
        )
        temp_root = os.path.normcase(os.path.realpath(tempfile.gettempdir()))
        is_managed_fallback = (
            os.path.normcase(parent) == temp_root
            and basename.startswith("glossarion_direct_text_chat_")
        )
        if not (is_direct_text_subfolder or is_managed_fallback):
            raise ValueError(
                "Refusing to delete an unrecognized folder outside the managed "
                "Direct Text conversation area."
            )
        return folder

    @staticmethod
    def _managed_conversation_output_folder(session):
        """Return an existing, recognized Direct Text conversation folder."""
        existing = os.path.realpath(
            os.path.abspath(str(session.get("output_folder", "") or ""))
        )
        if not existing or not os.path.isdir(existing):
            return ""

        # Older histories could store an attachment descendant instead of the
        # conversation root. Walk upward until the direct child of Direct Text
        # is found, but never accept an arbitrary directory from edited JSON.
        candidate = existing
        while True:
            parent = os.path.dirname(candidate)
            if os.path.basename(parent).casefold() == "direct text":
                return candidate if os.path.isdir(candidate) else ""
            if not parent or parent == candidate:
                break
            candidate = parent

        temp_root = os.path.normcase(os.path.realpath(tempfile.gettempdir()))
        if (
            os.path.normcase(os.path.dirname(existing)) == temp_root
            and os.path.basename(existing).startswith(
                "glossarion_direct_text_chat_"
            )
        ):
            return existing
        return ""

    def _conversation_attachment_folders(self, session):
        """List the immediate managed attachment workspaces for one chat."""
        conversation_folder = self._managed_conversation_output_folder(session)
        if not conversation_folder:
            return []
        stored_folder = os.path.abspath(
            str(session.get("output_folder", "") or "")
        )
        if os.path.normcase(stored_folder) != os.path.normcase(
            conversation_folder
        ):
            session["output_folder"] = conversation_folder
            self._schedule_chat_history_save()
        attachments_root = os.path.join(conversation_folder, "Attachments")
        if not os.path.isdir(attachments_root):
            return []
        try:
            folders = [
                os.path.abspath(os.path.join(attachments_root, name))
                for name in os.listdir(attachments_root)
                if os.path.isdir(os.path.join(attachments_root, name))
            ]
        except OSError:
            return []
        return sorted(
            folders,
            key=lambda path: os.path.basename(path).casefold(),
        )

    def _is_managed_attachment_workspace(self, session, folder):
        """Return whether a folder is an immediate attachment of one chat."""
        if session is None:
            return False
        conversation_folder = self._managed_conversation_output_folder(session)
        folder = os.path.realpath(os.path.abspath(str(folder or "")))
        if not conversation_folder or not folder or not os.path.isdir(folder):
            return False
        attachments_root = os.path.realpath(
            os.path.join(conversation_folder, "Attachments")
        )
        return (
            os.path.normcase(os.path.dirname(folder))
            == os.path.normcase(attachments_root)
        )

    def _direct_text_migration_output_root(self, conversation_folder):
        """Resolve the same output root used by normal Run Translation."""
        configured_root = (
            os.environ.get("OUTPUT_DIRECTORY")
            or os.environ.get("OUTPUT_DIR")
            or self.translator.config.get("output_directory")
        )
        if configured_root:
            return os.path.abspath(os.path.expanduser(str(configured_root)))

        conversation_folder = os.path.abspath(str(conversation_folder or ""))
        direct_text_root = os.path.dirname(conversation_folder)
        if os.path.basename(direct_text_root).casefold() == "direct text":
            return os.path.dirname(direct_text_root)
        return _get_app_dir() if getattr(sys, "frozen", False) else os.getcwd()

    @staticmethod
    def _path_is_same_or_descendant(path, parent):
        """Return whether an absolute path is equal to or below a parent."""
        try:
            path = os.path.normcase(os.path.abspath(str(path or "")))
            parent = os.path.normcase(os.path.abspath(str(parent or "")))
            return os.path.commonpath([path, parent]) == parent
        except (OSError, TypeError, ValueError):
            return False

    def _relocate_session_attachment_paths(
        self, session, source_folder, target_folder
    ):
        """Keep response links valid after an attachment workspace is moved."""
        source_folder = os.path.abspath(source_folder)
        target_folder = os.path.abspath(target_folder)

        def relocated(path):
            value = str(path or "").strip()
            if not value:
                return ""
            absolute = os.path.abspath(value)
            if not self._path_is_same_or_descendant(absolute, source_folder):
                return ""
            relative = os.path.relpath(absolute, source_folder)
            return os.path.normpath(os.path.join(target_folder, relative))

        updated_messages = []
        for message in session.get("messages", []):
            if not isinstance(message, (list, tuple)) or not message:
                updated_messages.append(message)
                continue
            values = list(message)
            if str(values[0] or "") == "assistant":
                while len(values) < 7:
                    values.append({} if len(values) == 6 else "")
                moved_output = relocated(values[4])
                if moved_output:
                    values[4] = moved_output

                storage = self._normalize_message_storage(values[6])
                for key in (
                    "content_path",
                    "content_text_path",
                    "content_html_path",
                    "content_xhtml_path",
                    "thinking_path",
                    "image_path",
                    "media_path",
                ):
                    reference = str(storage.get(key, "") or "")
                    if not reference:
                        continue
                    old_path = self._resolve_history_file_reference(reference)
                    new_path = relocated(old_path)
                    if new_path:
                        storage[key] = self._history_file_reference(new_path)
                values[6] = storage
            updated_messages.append(tuple(values))
        session["messages"] = updated_messages

        moved_last_output = relocated(self._last_output_folder)
        if moved_last_output and session is self._current_chat_session():
            self._last_output_folder = moved_last_output
            self._chat_messages = session["messages"]

    @staticmethod
    def _preferred_attachment_compiled_documents(folder):
        """Choose the attachment's authoritative top-level EPUB/PDF outputs.

        Attachment workspaces normally contain one compiled document. If an
        interrupted or older run left more than one, the newest top-level file
        is the best representation of the attachment run that is being moved.
        Nested PDFs are resources, not compiled attachment outputs, and are
        deliberately ignored.
        """
        folder = os.path.abspath(str(folder or ""))
        preferred = {}
        try:
            entries = [
                entry
                for entry in os.scandir(folder)
                if entry.is_file(follow_symlinks=False)
            ]
        except OSError:
            return preferred
        for extension in (".epub", ".pdf"):
            candidates = [
                entry
                for entry in entries
                if os.path.splitext(entry.name)[1].casefold() == extension
            ]
            if not candidates:
                continue

            def candidate_key(entry):
                try:
                    modified = float(entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    modified = 0.0
                return modified, entry.name.casefold()

            preferred[extension] = max(candidates, key=candidate_key).name
        return preferred

    @staticmethod
    def _remove_extra_attachment_compiled_documents(folder, preferred):
        """Keep only the attachment-priority compiled EPUB/PDF files."""
        folder = os.path.realpath(os.path.abspath(str(folder or "")))
        for extension, preferred_name in dict(preferred or {}).items():
            keep_path = os.path.join(folder, str(preferred_name))
            if not os.path.isfile(keep_path):
                raise FileNotFoundError(
                    f"The attachment's preferred {extension} output is missing: "
                    f"{keep_path}"
                )
            for entry in os.scandir(folder):
                if not entry.is_file(follow_symlinks=False):
                    continue
                if os.path.splitext(entry.name)[1].casefold() != extension:
                    continue
                if os.path.normcase(entry.name) == os.path.normcase(
                    str(preferred_name)
                ):
                    continue
                os.remove(entry.path)

    def _attachment_compiled_epub_path(self, folder):
        """Return the authoritative compiled EPUB in an attachment workspace."""
        folder = os.path.abspath(str(folder or ""))
        preferred = self._preferred_attachment_compiled_documents(folder)
        epub_name = str(preferred.get(".epub", "") or "")
        epub_path = os.path.join(folder, epub_name) if epub_name else ""
        return epub_path if epub_path and os.path.isfile(epub_path) else ""

    def _migrate_conversation_attachment(
        self, session_index, source_folder, message_parent=None
    ):
        """Move one attachment tree beside Direct Text, confirming conflicts."""
        message_parent = message_parent or self
        if self._active:
            self._direct_text_notice(
                "information",
                message_parent,
                "Attachment migration unavailable",
                "Wait for the current Direct Text run to finish before moving files.",
            )
            return False
        try:
            session_index = int(session_index)
        except (TypeError, ValueError):
            return False
        if not (0 <= session_index < len(self._chat_sessions)):
            return False

        session = self._chat_sessions[session_index]
        conversation_folder = self._managed_conversation_output_folder(session)
        attachments_root = os.path.realpath(
            os.path.join(conversation_folder, "Attachments")
        )
        source_folder = os.path.realpath(os.path.abspath(str(source_folder)))
        if (
            not conversation_folder
            or not os.path.isdir(source_folder)
            or os.path.normcase(os.path.dirname(source_folder))
            != os.path.normcase(attachments_root)
        ):
            self._direct_text_notice(
                "warning",
                message_parent,
                "Attachment unavailable",
                "This folder is no longer a managed attachment for the conversation.",
            )
            return False

        target_root = self._direct_text_migration_output_root(
            conversation_folder
        )
        target_folder = os.path.abspath(
            os.path.join(target_root, os.path.basename(source_folder))
        )
        if os.path.normcase(target_folder) == os.path.normcase(source_folder):
            self._direct_text_notice(
                "information",
                message_parent,
                "Attachment already migrated",
                f"The attachment is already at:\n{target_folder}",
            )
            return False

        preferred_compiled_documents = (
            self._preferred_attachment_compiled_documents(source_folder)
        )
        destination_exists = os.path.exists(target_folder)
        if destination_exists:
            if not self._confirm_attachment_merge(message_parent, target_folder):
                return False

        try:
            import shutil

            os.makedirs(target_root, exist_ok=True)
            if destination_exists and os.path.isdir(target_folder):
                shutil.copytree(
                    source_folder,
                    target_folder,
                    dirs_exist_ok=True,
                )
                self._remove_extra_attachment_compiled_documents(
                    target_folder, preferred_compiled_documents
                )
                # Revalidate the exact recursive-delete target after copying.
                if (
                    os.path.normcase(os.path.dirname(source_folder))
                    != os.path.normcase(attachments_root)
                ):
                    raise ValueError(
                        "The attachment source changed during migration."
                )
                shutil.rmtree(source_folder)
            elif destination_exists:
                self._remove_extra_attachment_compiled_documents(
                    source_folder, preferred_compiled_documents
                )
                os.remove(target_folder)
                shutil.move(source_folder, target_folder)
            else:
                self._remove_extra_attachment_compiled_documents(
                    source_folder, preferred_compiled_documents
                )
                shutil.move(source_folder, target_folder)
        except Exception as exc:
            self._direct_text_notice(
                "warning",
                message_parent,
                "Could not migrate attachment",
                f"The attachment could not be moved.\n\n{exc}",
            )
            return False

        self._relocate_session_attachment_paths(
            session, source_folder, target_folder
        )
        self._message_text_cache.clear()
        self._rendered_message_cache.clear()
        self._save_chat_history()
        self._refresh_chat_list()
        if session_index == self._current_chat_index:
            self._render_output(preserve_viewport=True)
        self._direct_text_notice(
            "information",
            message_parent,
            "Attachment migrated",
            f"The attachment workspace was moved to:\n{target_folder}",
        )
        return True

    def _title_current_chat_from_text(self, text):
        session = self._current_chat_session()
        if session is None or session.get("title") != "New chat":
            return
        compact = " ".join(str(text or "").split())
        if not compact:
            return
        max_title = 42
        session["title"] = (
            compact if len(compact) <= max_title else compact[:max_title - 1] + "…"
        )
        self._refresh_chat_list()
        self._schedule_chat_history_save()

    def _remember_output_folder(self, folder, *, update_conversation_root=True):
        folder = os.path.abspath(str(folder or "")) if folder else ""
        if not folder or not os.path.isdir(folder):
            return ""
        self._last_output_folder = folder
        session = self._current_chat_session()
        if session is not None and update_conversation_root:
            session["output_folder"] = folder
        self._schedule_chat_history_save()
        return folder

    def _editable_response_paths(self, message_index):
        """Return the managed Markdown/text/HTML/XHTML files for one response."""
        session = self._current_chat_session()
        if session is None:
            return "", "", "", ""
        conversation_folder = self._ensure_conversation_output_folder_for_session(
            session
        )
        if not conversation_folder:
            return "", "", "", ""
        message_directory = os.path.join(conversation_folder, "Chat Messages")
        markdown_path = os.path.join(
            message_directory, f"{int(message_index) + 1:06d}-response.md"
        )
        response_stem = os.path.splitext(markdown_path)[0]
        return (
            markdown_path,
            response_stem + ".txt",
            response_stem + ".html",
            response_stem + ".xhtml",
        )

    @staticmethod
    def _response_artifact_match_text(source):
        """Reduce response markup to comparable visible text."""
        import html as html_lib
        import re

        value = str(source or "")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(value, "html.parser")
            for unwanted in soup.find_all(("script", "style")):
                unwanted.decompose()
            value = soup.get_text(" ", strip=True)
        except Exception:
            value = re.sub(r"<[^>]+>", " ", value)
        value = html_lib.unescape(value)
        return " ".join(value.split()).casefold()

    def _response_output_artifact_path(self, message, message_index):
        """Resolve a chapter card to its real translated attachment file."""
        import difflib
        import json as json_lib
        import re

        if not isinstance(message, (list, tuple)):
            return ""
        output_folder = str(message[4] if len(message) > 4 else "").strip()
        request_label = str(message[5] if len(message) > 5 else "").strip()
        chapter_match = re.search(
            r"\b(?:chapter|section)\s+(-?\d+)\b",
            request_label,
            flags=re.IGNORECASE,
        )
        if not output_folder or not chapter_match:
            return ""
        output_folder = os.path.abspath(os.path.expanduser(output_folder))
        if not os.path.isdir(output_folder):
            return ""
        try:
            chapter_number = int(chapter_match.group(1))
        except (TypeError, ValueError):
            return ""

        progress_paths = []
        direct_progress = os.path.join(output_folder, "translation_progress.json")
        if os.path.isfile(direct_progress):
            progress_paths.append(direct_progress)
        else:
            # Some attachment types keep each translated document one level
            # below the card's output folder. Avoid Chat Messages and backup
            # trees while locating their authoritative progress file.
            for root, directories, files in os.walk(output_folder):
                directories[:] = [
                    name
                    for name in directories
                    if name.casefold()
                    not in {"chat messages", "unrefined_backup", "sdlxliff"}
                ]
                if "translation_progress.json" in files:
                    progress_paths.append(
                        os.path.join(root, "translation_progress.json")
                    )
                if len(progress_paths) >= 12:
                    break

        candidates = []
        supported_extensions = {
            ".html",
            ".htm",
            ".xhtml",
            ".xhtm",
            ".txt",
            ".md",
            ".markdown",
        }
        for progress_path in progress_paths:
            try:
                with open(progress_path, "r", encoding="utf-8") as handle:
                    progress = json_lib.load(handle)
            except (OSError, TypeError, ValueError):
                continue
            chapters = progress.get("chapters", {})
            if not isinstance(chapters, dict):
                continue
            progress_folder = os.path.dirname(progress_path)
            for record in chapters.values():
                if not isinstance(record, dict):
                    continue
                try:
                    actual_number = int(record.get("actual_num"))
                except (TypeError, ValueError):
                    continue
                if actual_number != chapter_number:
                    continue
                relative_output = str(record.get("output_file", "") or "").strip()
                if not relative_output:
                    continue
                artifact_path = os.path.abspath(
                    os.path.join(progress_folder, relative_output)
                )
                try:
                    if os.path.commonpath((artifact_path, progress_folder)) != os.path.abspath(
                        progress_folder
                    ):
                        continue
                except ValueError:
                    continue
                if (
                    os.path.splitext(artifact_path)[1].casefold()
                    not in supported_extensions
                    or not os.path.isfile(artifact_path)
                ):
                    continue
                candidates.append(artifact_path)

        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Chapter zero can legitimately contain both a cover and an info page.
        # Match the response's visible text instead of guessing from filenames.
        response_text = self._response_artifact_match_text(
            self._assistant_message_text(message, "content", message_index)
        )
        if not response_text:
            return ""
        ranked = []
        for candidate in candidates:
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    candidate_text = self._response_artifact_match_text(handle.read())
            except OSError:
                continue
            score = difflib.SequenceMatcher(
                None,
                response_text,
                candidate_text,
                autojunk=False,
            ).ratio()
            ranked.append((score, candidate))
        if not ranked:
            return ""
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _save_response_output_edit(
        self,
        message_index,
        edited_source,
        *,
        text_source=None,
        html_source=None,
    ):
        """Persist an edited card to every managed format and refresh its caches."""
        if not (0 <= int(message_index) < len(self._chat_messages)):
            raise IndexError("Response message is no longer available")
        message = self._chat_messages[int(message_index)]
        if not message or message[0] != "assistant":
            raise ValueError("Only assistant responses can be edited")

        (
            markdown_path,
            _text_path,
            _html_path,
            _xhtml_path,
        ) = self._editable_response_paths(message_index)
        if not markdown_path:
            raise OSError("The conversation output folder is unavailable")
        output_artifact_path = self._response_output_artifact_path(
            message, int(message_index)
        )
        text_path, html_path, xhtml_path = self._write_response_files(
            markdown_path,
            edited_source,
            text_source=text_source,
            html_source=html_source,
            output_artifact_path=output_artifact_path,
        )

        values = list(message[:6])
        while len(values) < 6:
            values.append("")
        values[1] = ""
        storage = self._normalize_message_storage(
            message[6] if len(message) > 6 else None
        )
        storage["content_path"] = self._history_file_reference(markdown_path)
        storage["content_text_path"] = self._history_file_reference(text_path)
        storage["content_html_path"] = self._history_file_reference(html_path)
        storage["content_xhtml_path"] = self._history_file_reference(xhtml_path)
        storage["content_chars"] = len(str(edited_source or ""))
        self._chat_messages[int(message_index)] = tuple(values + [storage])

        self._message_text_cache.clear()
        self._rendered_message_cache.clear()
        self._save_chat_history()
        self._render_output(preserve_viewport=True)
        return markdown_path, html_path

    def _assistant_source_is_attachment(self, message_index):
        """Resolve whether an assistant card belongs to a file-attachment turn."""
        try:
            upper_index = min(
                int(message_index) - 1,
                len(self._chat_messages) - 1,
            )
        except (TypeError, ValueError):
            return None
        for prior_index in range(upper_index, -1, -1):
            prior_message = self._chat_messages[prior_index]
            if not isinstance(prior_message, (list, tuple)) or not prior_message:
                continue
            prior_role = str(prior_message[0] or "").strip().lower()
            if prior_role == "user_file":
                return True
            if prior_role == "user":
                return False
        return None

    @staticmethod
    def _markup_to_html(source):
        """Convert mixed Markdown/raw HTML into one renderable HTML fragment."""
        import html as html_lib
        import re

        source = str(source or "").replace("\r\n", "\n").replace("\r", "\n")

        # Models frequently wrap an otherwise valid HTML document in a fenced
        # block.  In this rich-output surface that fence is presentation noise,
        # so unwrap it before handing the text to the Markdown converter.
        fenced_html = re.fullmatch(
            r"\s*```(?:html?|xhtml)\s*\n?(.*?)\n?```\s*",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_html:
            source = fenced_html.group(1)

        # Render escaped, known presentation tags as tags too.  The allow-list
        # deliberately excludes script/object/embed and other active content.
        renderable_tags = (
            r"html|head|body|title|meta|style|h[1-6]|p|div|span|br|hr|"
            r"ul|ol|li|blockquote|pre|code|table|thead|tbody|tfoot|tr|th|td|"
            r"strong|b|em|i|u|s|del|a|img|figure|figcaption|details|summary"
        )

        def _unescape_known_tag(match):
            return "<" + html_lib.unescape(match.group(1)) + ">"

        source = re.sub(
            rf"&lt;(/?(?:{renderable_tags})\b.*?/?)&gt;",
            _unescape_known_tag,
            source,
            flags=re.IGNORECASE,
        )

        def _sanitized_fragment(value, prefer_body=False):
            """Return chat-safe body markup without document-level styling."""
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(str(value or ""), 'html.parser')
                # A model response must not restyle the surrounding chat UI or
                # inject active/external document metadata.
                for tag in soup.find_all(
                    ('script', 'style', 'link', 'meta', 'base', 'object',
                     'embed', 'iframe', 'noscript')
                ):
                    tag.decompose()

                if prefer_body:
                    body = soup.body
                    if body is not None:
                        root = body
                    else:
                        if soup.head is not None:
                            soup.head.decompose()
                        root = soup.html if soup.html is not None else soup
                else:
                    root = soup

                for tag in root.find_all(True):
                    for attribute in list(tag.attrs):
                        attr_low = str(attribute).lower()
                        if (
                            attr_low == 'style'
                            or attr_low == 'bgcolor'
                            or attr_low.startswith('on')
                        ):
                            del tag.attrs[attribute]

                fragment = root.decode_contents()
                block_names = (
                    'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'br', 'hr', 'ul', 'ol', 'li', 'blockquote', 'pre',
                    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
                    'figure', 'figcaption', 'details', 'summary',
                )
                if prefer_body and not root.find(block_names):
                    fragment = fragment.replace("\n", "<br>\n")
                return fragment
            except Exception:
                cleaned = re.sub(
                    r"<(?:script|style|head)\b[^>]*>.*?</(?:script|style|head)\s*>",
                    "",
                    str(value or ""),
                    flags=re.IGNORECASE | re.DOTALL,
                )
                cleaned = re.sub(
                    r"</?(?:!doctype|html|body)\b[^>]*>",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    r"\s(?:style|bgcolor|on\w+)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                return cleaned

        # Parse complete HTML before Markdown. Markdown treats indented HTML
        # source as a code block, which exposes literal tags and creates dark
        # code-background strips. Only the document body belongs in the chat.
        if re.search(
            r"<(?:!doctype\s+html|html\b|head\b|body\b)",
            source,
            flags=re.IGNORECASE,
        ):
            return _sanitized_fragment(source, prefer_body=True)

        try:
            import markdown
            rendered = markdown.markdown(
                source,
                extensions=['extra', 'sane_lists', 'nl2br'],
                output_format='html5',
            )
        except Exception:
            try:
                import markdown2
                rendered = markdown2.markdown(
                    source,
                    extras=['break-on-newline', 'cuddled-lists',
                            'fenced-code-blocks', 'tables'],
                )
            except Exception:
                # Both converters are bundled dependencies, but preserving raw
                # HTML remains more useful than showing literal tags if a
                # minimal/custom installation omitted them.
                rendered = source.replace("\n", "<br>")

        return _sanitized_fragment(rendered)

    def _ensure_conversation_output_folder_for_session(self, session):
        """Return one session's persistent Direct Text output folder."""
        if session is None:
            return ""

        existing = str(session.get("output_folder", "") or "")
        if existing:
            existing = os.path.abspath(existing)
            # Repair histories saved by the old attachment behavior, which
            # could replace the conversation root with a descendant attachment
            # directory. A managed conversation root is always directly below
            # the ``Direct Text`` directory.
            candidate = existing
            while True:
                parent = os.path.dirname(candidate)
                if os.path.basename(parent).casefold() == "direct text":
                    if candidate != existing:
                        existing = candidate
                        session["output_folder"] = existing
                        self._schedule_chat_history_save()
                    break
                if not parent or parent == candidate:
                    break
                candidate = parent
            os.makedirs(existing, exist_ok=True)
            return existing

        configured_root = (
            self._saved_env.get('OUTPUT_DIRECTORY')
            or self._saved_env.get('OUTPUT_DIR')
            or self.translator.config.get('output_directory')
        )
        if configured_root:
            output_root = os.path.abspath(os.path.expanduser(str(configured_root)))
        else:
            output_root = _get_app_dir() if getattr(sys, 'frozen', False) else os.getcwd()

        direct_text_root = os.path.join(output_root, 'Direct Text')
        os.makedirs(direct_text_root, exist_ok=True)

        folder_name = str(session.get("output_folder_name", "") or "")
        if not folder_name:
            import re
            import uuid
            from datetime import datetime

            title = " ".join(str(session.get("title", "") or "").split())
            if not title or title == "New chat":
                title = f"Chat {int(session.get('id', 0) or 0):03d}"
            safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title)
            safe_title = safe_title.rstrip(" .")[:60] or "Chat"
            folder_name = (
                f"{safe_title} - {datetime.now():%Y%m%d_%H%M%S}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            session["output_folder_name"] = folder_name

        target_folder = os.path.join(direct_text_root, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        session["output_folder"] = target_folder
        if session is self._current_chat_session():
            self._last_output_folder = target_folder
        return target_folder

    def _conversation_output_folder(self):
        """Return the current chat's single persistent Direct Text output folder."""
        return self._ensure_conversation_output_folder_for_session(
            self._current_chat_session()
        )

    def _next_indexed_output_path(self, target_folder, extension=".txt"):
        """Return a collision-safe ``Direct Text N`` path for any output type."""
        import re

        extension = str(extension or ".txt").lower()
        if not extension.startswith(".") or any(
            char in extension for char in ('/', '\\', ':')
        ):
            extension = ".txt"

        session = self._current_chat_session()
        try:
            next_index = max(1, int(session.get("next_output_index", 1)))
        except (AttributeError, TypeError, ValueError):
            next_index = 1

        pattern = re.compile(r"^Direct Text (\d+)\.[^.]+$", re.IGNORECASE)
        try:
            for filename in os.listdir(target_folder):
                match = pattern.match(filename)
                if match:
                    next_index = max(next_index, int(match.group(1)) + 1)
        except OSError:
            pass

        while True:
            target_path = os.path.join(
                target_folder, f"Direct Text {next_index}{extension}"
            )
            if not os.path.exists(target_path):
                return next_index, target_path
            next_index += 1

    def _copy_indexed_output_file(self, target_folder):
        """Copy the completed translation using this conversation's next index."""
        if not self._expected_output or not os.path.isfile(self._expected_output):
            return ""
        import shutil

        output_extension = os.path.splitext(self._expected_output)[1].lower() or ".txt"
        output_index, target_path = self._next_indexed_output_path(
            target_folder, output_extension
        )
        shutil.copy2(self._expected_output, target_path)
        self._persisted_output_path = os.path.abspath(target_path)
        session = self._current_chat_session()
        if session is not None:
            session["next_output_index"] = output_index + 1
        self._schedule_chat_history_save()
        return target_path

    def _attachment_output_subfolder(self, conversation_folder):
        """Return the run-style output folder for the current attachment."""
        if not self._run_source_is_attachment or not self._run_source_path:
            return ""
        import re

        source_stem = os.path.splitext(
            os.path.basename(self._run_source_path)
        )[0]
        safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', source_stem)
        safe_stem = safe_stem.rstrip(" .")[:120] or "Attachment"
        # Keep run-style attachment trees in their own namespace so an
        # attachment can never become the destination for ``Direct Text N``
        # files created by ordinary composer-only messages.
        target_folder = os.path.join(
            conversation_folder, "Attachments", safe_stem
        )
        os.makedirs(target_folder, exist_ok=True)
        return target_folder

    def _copy_attachment_output_tree(self, target_folder):
        """Preserve the complete regular-run folder for an attached source."""
        import shutil

        source_stem = os.path.splitext(
            os.path.basename(self._run_source_path)
        )[0]
        generated_folder = os.path.join(self._temp_root, source_stem)
        copied_any = False
        persisted_candidate = ""
        if os.path.isdir(generated_folder):
            shutil.copytree(
                generated_folder,
                target_folder,
                dirs_exist_ok=True,
            )
            copied_any = True

        # Some output modes place their final artifact directly under the
        # override root. Keep that artifact too without flattening a normal
        # generated book folder.
        if self._expected_output and os.path.isfile(self._expected_output):
            expected_abs = os.path.abspath(self._expected_output)
            generated_abs = os.path.abspath(generated_folder)
            try:
                already_in_tree = (
                    os.path.commonpath([expected_abs, generated_abs])
                    == generated_abs
                )
            except ValueError:
                already_in_tree = False
            if not already_in_tree or not copied_any:
                shutil.copy2(
                    expected_abs,
                    os.path.join(target_folder, os.path.basename(expected_abs)),
                )
                persisted_candidate = os.path.join(
                    target_folder, os.path.basename(expected_abs)
                )
                copied_any = True
            elif already_in_tree:
                persisted_candidate = os.path.join(
                    target_folder,
                    os.path.relpath(expected_abs, generated_abs),
                )
        if persisted_candidate and os.path.isfile(persisted_candidate):
            self._persisted_output_path = os.path.abspath(persisted_candidate)
        if copied_any:
            self._sync_attachment_glossary(target_folder)
        return target_folder if copied_any else ""

    def _effective_run_glossary_path(self):
        """Return the authoritative glossary used by the active Direct Text run."""
        try:
            if bool(
                getattr(
                    self.translator,
                    '_direct_text_force_no_glossary',
                    self.force_no_glossary_radio.isChecked(),
                )
            ):
                return ""
        except Exception:
            pass

        # An explicitly supplied Direct Text manual glossary has the highest
        # precedence. For inherited/automatic modes, glossary generation may
        # replace the parent's path during the run, so inspect the live values
        # after that explicit candidate.
        candidates = [
            getattr(self, '_run_manual_glossary_path', ''),
            getattr(self.translator, '_direct_text_manual_glossary_path', ''),
            (os.environ if self._RUN_ENVIRONMENT is None else self._RUN_ENVIRONMENT).get('MANUAL_GLOSSARY', ''),
            getattr(self.translator, 'manual_glossary_path', ''),
        ]
        seen = set()
        for candidate in candidates:
            path = os.path.abspath(str(candidate or '').strip())
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
        return ""

    def _sync_attachment_glossary(self, target_folder):
        """Make the reused attachment tree match this run's effective glossary."""
        import shutil

        canonical_names = ('glossary.csv', 'glossary.json', 'glossary.md')
        glossary_path = self._effective_run_glossary_path()

        # No Glossary is authoritative for the current run. Do not leave a
        # canonical glossary from an earlier attachment run in the reused
        # folder, where it looks active and can be auto-detected later.
        if not glossary_path:
            try:
                force_none = bool(
                    getattr(
                        self.translator,
                        '_direct_text_force_no_glossary',
                        self.force_no_glossary_radio.isChecked(),
                    )
                )
            except Exception:
                force_none = False
            if force_none:
                for name in canonical_names:
                    stale_path = os.path.join(target_folder, name)
                    if os.path.isfile(stale_path):
                        os.remove(stale_path)
            return ""

        extension = os.path.splitext(glossary_path)[1].lower()
        if extension in {'.csv', '.txt'}:
            target_name = 'glossary.csv'
        elif extension == '.md':
            target_name = 'glossary.md'
        elif extension == '.json':
            target_name = 'glossary.json'
        else:
            target_name = 'glossary.csv'
        target_path = os.path.abspath(os.path.join(target_folder, target_name))

        # Remove only the alternate canonical glossary formats managed by the
        # translation pipeline. Other CSV/JSON artifacts are left untouched.
        for name in canonical_names:
            candidate = os.path.abspath(os.path.join(target_folder, name))
            if candidate != target_path and os.path.isfile(candidate):
                os.remove(candidate)
        if os.path.abspath(glossary_path) != target_path:
            shutil.copy2(glossary_path, target_path)
        return target_path

    def _discover_generated_output(self):
        """Find the final translated artifact created inside the scoped temp root."""
        existing_expected = (
            self._expected_output
            if self._expected_output and os.path.isfile(self._expected_output)
            else ""
        )
        if existing_expected:
            expected_extension = os.path.splitext(existing_expected)[1].lower()
            expected_media_extensions = {
                'image': ChatStoreMixin._IMAGE_ATTACHMENT_EXTENSIONS,
                'video': ChatStoreMixin._VIDEO_OUTPUT_EXTENSIONS,
                'audio': ChatStoreMixin._AUDIO_OUTPUT_EXTENSIONS,
            }.get(self._run_output_mode, set())
            # Generated-media runs can first expose a small translated .txt
            # adapter containing only a sentinel. Do not let that adapter hide
            # the actual durable media file saved elsewhere in the run tree.
            if not (
                expected_media_extensions
                and expected_extension not in expected_media_extensions
            ):
                return existing_expected
        if self._run_output_mode in {'image', 'video', 'audio'}:
            sentinel_sources = [str(getattr(self, '_streamed_content', '') or '')]
            sentinel_sources.extend(
                str(segment.get('content', '') or '')
                for segment in getattr(self, '_active_request_segments', [])
                if isinstance(segment, dict)
            )
            for sentinel_source in sentinel_sources:
                for media_kind, media_path in (
                    ChatStoreMixin._generated_media_references_from_text(
                        sentinel_source
                    )
                ):
                    if (
                        media_kind == self._run_output_mode
                        and os.path.isfile(media_path)
                    ):
                        self._expected_output = media_path
                        return media_path
        if not self._temp_root or not os.path.isdir(self._temp_root):
            return existing_expected

        source_ext = str(self._run_source_extension or ".txt").lower()
        generated_image_scores = {
            extension: 125
            for extension in ChatStoreMixin._IMAGE_ATTACHMENT_EXTENSIONS
        }
        generated_video_scores = {
            '.mp4': 140, '.mov': 135, '.webm': 135, '.mkv': 130,
            '.avi': 125, '.m4v': 125, '.mpeg': 120, '.mpg': 120,
            '.ogv': 120, '.wmv': 115, '.3gp': 110,
        }
        generated_audio_scores = {
            '.wav': 140, '.mp3': 135, '.m4a': 130, '.flac': 130,
            '.aac': 125, '.ogg': 125, '.opus': 125, '.wma': 120,
            '.aiff': 120, '.aif': 120, '.oga': 120, '.pcm': 110,
        }
        preferred_by_source = {
            ".epub": {".epub": 140, ".pdf": 100, ".txt": 80, ".html": 60},
            ".cbz": {".epub": 140, ".pdf": 100, ".txt": 90, ".html": 60},
            ".pdf": {".pdf": 140, ".epub": 130, ".txt": 100, ".html": 70},
            ".srt": {".srt": 160, ".txt": 70},
            ".ass": {".ass": 160, ".txt": 70},
            ".lrc": {".lrc": 160, ".txt": 70},
        }
        if self._run_output_mode == 'image':
            preferred = dict(generated_image_scores)
            preferred.update({'.txt': 60, '.html': 55})
        elif self._run_output_mode == 'video':
            preferred = dict(generated_video_scores)
            preferred.update(generated_image_scores)
        elif self._run_output_mode == 'audio':
            preferred = dict(generated_audio_scores)
            preferred.update({'.txt': 70})
        elif source_ext in self._IMAGE_ATTACHMENT_EXTENSIONS:
            preferred = {'.txt': 150, '.html': 135, '.xhtml': 130}
            preferred.update(generated_image_scores)
        else:
            preferred = preferred_by_source.get(
                source_ext,
                {".txt": 140, ".html": 100, ".xhtml": 95, ".epub": 80, ".pdf": 80},
            )
        candidates = []
        for root, _dirs, filenames in os.walk(self._temp_root):
            for filename in filenames:
                path = os.path.join(root, filename)
                extension = os.path.splitext(filename)[1].lower()
                if extension not in preferred:
                    continue
                normalized = path.replace('\\', '/').lower()
                if '/_archive_input/' in normalized:
                    continue
                score = preferred[extension]
                low_name = filename.lower()
                if "translated" in low_name:
                    score += 55
                if "direct text" in low_name:
                    score += 20
                if any(
                    marker in normalized
                    for marker in ("/chunks/", "/responses/", "/payloads/", "/ocr/chunks/")
                ):
                    score -= 120
                if low_name in ("source_epub.txt", "toc.txt"):
                    score -= 200
                try:
                    modified = os.path.getmtime(path)
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if modified >= self._run_started_at - 2.0:
                    score += 10
                if size > 0:
                    score += min(20, int(size).bit_length())
                candidates.append((score, modified, size, path))
        if not candidates:
            return existing_expected
        candidates.sort(reverse=True)
        self._expected_output = candidates[0][3]
        return self._expected_output

    def _persist_output_folder(self):
        """Save a completed run as an indexed artifact in the chat folder."""
        self._persisted_output_path = ""
        self._discover_generated_output()
        if (
            (not self._expected_output or not os.path.isfile(self._expected_output))
            and self._streamed_content.strip()
            and self._temp_root
        ):
            self._expected_output = os.path.join(
                self._temp_root, "direct_text_stream_translated.txt"
            )
            try:
                with open(self._expected_output, 'w', encoding='utf-8') as handle:
                    handle.write(self._streamed_content)
            except OSError:
                self._expected_output = ""
        if not self._expected_output or not os.path.isfile(self._expected_output):
            return ""
        try:
            conversation_folder = self._conversation_output_folder()
            if not conversation_folder:
                return ""
            if self._run_source_is_attachment:
                attachment_folder = self._attachment_output_subfolder(
                    conversation_folder
                )
                if attachment_folder:
                    copied_folder = self._copy_attachment_output_tree(
                        attachment_folder
                    )
                    if copied_folder:
                        return copied_folder
            self._copy_indexed_output_file(conversation_folder)
            return conversation_folder
        except Exception as exc:
            self._append_thinking(f"⚠️ Could not persist Direct Text output: {exc}\n")
            session = self._current_chat_session()
            try:
                fallback_folder = tempfile.mkdtemp(
                    prefix="glossarion_direct_text_chat_"
                )
                if session is not None:
                    session["output_folder"] = fallback_folder
                    session["next_output_index"] = 1
                self._copy_indexed_output_file(fallback_folder)
                self._last_output_folder = fallback_folder
                return fallback_folder
            except Exception as fallback_exc:
                # Last resort: preserve this run so its response link remains valid.
                self._preserve_temp_root = True
                self._persisted_output_path = os.path.abspath(
                    self._expected_output
                )
                self._append_thinking(
                    f"⚠️ Could not create shared fallback output folder: "
                    f"{fallback_exc}\n"
                )
                return os.path.dirname(self._expected_output)

    def _find_direct_run_artifact(self, filename):
        """Find one run artifact, preferring the attached source's output tree."""
        if not self._temp_root or not os.path.isdir(self._temp_root):
            return ""
        source_stem = os.path.splitext(
            os.path.basename(self._run_source_path or "")
        )[0]
        preferred_root = os.path.abspath(
            os.path.join(self._temp_root, source_stem)
        )
        candidates = []
        for root, _dirs, files in os.walk(self._temp_root):
            if filename not in files:
                continue
            path = os.path.abspath(os.path.join(root, filename))
            try:
                in_preferred_tree = (
                    os.path.commonpath([path, preferred_root])
                    == preferred_root
                )
            except ValueError:
                in_preferred_tree = False
            try:
                modified = os.path.getmtime(path)
            except OSError:
                modified = 0.0
            candidates.append((1 if in_preferred_tree else 0, modified, path))
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _attachment_card_actions(self, session, output_folder):
        """Links of an "Attachment actions" card, in desktop order: 'migrate', 'reader'.

        Was inline in ``_render_output``; the dialog renders these ids as its links and
        the mobile chat as buttons (the "Open output folder" link follows any card with
        an output folder).
        """
        actions = []
        if self._is_managed_attachment_workspace(
            session, output_folder
        ):
            actions.append('migrate')
        compiled_epub = self._attachment_compiled_epub_path(
            output_folder
        )
        if compiled_epub:
            actions.append('reader')
        return actions

    def _init_chat_sessions(self):
        """Load the retained chats and select the saved current one (dialog ``__init__``).

        Expects ``_chat_history_path``; returns the saved current chat id.
        """
        self._chat_sessions, current_chat_id = self._load_chat_history()
        self._chat_session_counter = max(
            (int(session.get("id", 0)) for session in self._chat_sessions),
            default=0,
        )
        if not self._chat_sessions:
            self._chat_session_counter = 1
            self._chat_sessions = [self._new_chat_session(self._chat_session_counter)]
        self._current_chat_index = next(
            (
                index
                for index, session in enumerate(self._chat_sessions)
                if session.get("id") == current_chat_id
            ),
            0,
        )
        self._chat_messages = self._chat_sessions[self._current_chat_index]["messages"]
        return current_chat_id

    def _prepare_direct_text_input(self, text, attachment, manual_glossary_source):
        """Create the run's temp root and input file (the ``_start_translation`` block).

        Sets ``_temp_root``, ``_run_manual_glossary_path``, ``_temp_input``,
        ``_run_source_is_attachment``, ``_run_source_path``, ``_run_source_extension``,
        ``_run_started_at`` and ``_expected_output``; returns the manual glossary path.
        """
        import uuid
        from datetime import datetime

        self._temp_root = tempfile.mkdtemp(prefix="glossarion_input_output_", dir=self._DIRECT_TEXT_TEMP_PARENT)
        manual_glossary_path = ""
        if manual_glossary_source:
            if manual_glossary_source.get('kind') == 'path':
                manual_glossary_path = os.path.abspath(
                    str(manual_glossary_source.get('path') or '')
                )
            else:
                glossary_extension = str(
                    manual_glossary_source.get('extension') or '.txt'
                ).lower()
                if glossary_extension not in {'.csv', '.json', '.txt', '.md'}:
                    glossary_extension = '.txt'
                manual_glossary_path = os.path.join(
                    self._temp_root,
                    f"direct_text_manual_glossary{glossary_extension}",
                )
                with open(
                    manual_glossary_path, 'w', encoding='utf-8'
                ) as glossary_file:
                    glossary_file.write(
                        str(manual_glossary_source.get('content') or '')
                    )
            if not manual_glossary_path or not os.path.isfile(manual_glossary_path):
                raise FileNotFoundError(
                    "The selected manual glossary is no longer available."
                )
        # Keep the exact glossary selected for this run independent from
        # the parent window's state. The parent state is temporarily
        # mutated by auto-loading/generation and is restored when Direct
        # Text finishes, so it is not a reliable persistence source.
        self._run_manual_glossary_path = manual_glossary_path
        if attachment:
            attached_path = attachment["path"]
            attached_extension = os.path.splitext(attached_path)[1].lower()
            stem = os.path.splitext(os.path.basename(attached_path))[0]
            if (
                attached_extension in {
                    '.txt', '.epub', '.pdf', '.cbz', '.csv', '.json',
                    '.srt', '.ass', '.lrc',
                }
                or attached_extension in self._IMAGE_ATTACHMENT_EXTENSIONS
                or attached_extension in self._EXTRA_PASS_THROUGH_EXTENSIONS
            ):
                self._temp_input = attached_path
            else:
                # Preserve legacy support for markup/subtitle/log files by
                # adapting them to the regular text pipeline internally,
                # while the composer/history still displays a file card.
                try:
                    with open(attached_path, 'r', encoding='utf-8-sig') as handle:
                        attached_text = handle.read()
                except UnicodeDecodeError:
                    with open(
                        attached_path, 'r', encoding='utf-8', errors='replace'
                    ) as handle:
                        attached_text = handle.read()
                self._temp_input = os.path.join(self._temp_root, f"{stem}.txt")
                with open(self._temp_input, 'w', encoding='utf-8') as handle:
                    handle.write(attached_text)
            self._run_source_is_attachment = True
        else:
            stem = f"direct_text_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
            self._temp_input = os.path.join(self._temp_root, f"{stem}.txt")
            with open(self._temp_input, 'w', encoding='utf-8') as handle:
                handle.write(text)
            self._run_source_is_attachment = False
        self._run_source_path = self._temp_input
        self._run_source_extension = (
            os.path.splitext(self._temp_input)[1].lower() or ".txt"
        )
        import time as _time
        self._run_started_at = _time.time()
        self._expected_output = os.path.join(
            self._temp_root,
            stem,
            f"{stem}_translated"
            f"{self._run_source_extension if self._run_source_extension in {'.srt', '.ass', '.lrc'} else '.txt'}",
        )
        return manual_glossary_path


class DirectTextOwnerView:
    """The ``translator`` a GUI-free host gives the moved code: ``.config`` and ``.model_var``.

    The moved code reads ``config.get('output_directory')`` (chat folders, Migrate),
    ``model_var`` (token counting) and, at the end of a run, the Direct Text glossary
    attributes the pipeline set on the owner (``_direct_text_force_no_glossary``,
    ``_direct_text_manual_glossary_path``, ``manual_glossary_path``).
    """

    def __init__(self, config=None, model=None):
        self.config = config if config is not None else {}
        self.model_var = str(model or "")


class _CheckFlag:
    """``isChecked()`` shim for the dialog radio the moved code reads (getattr default)."""

    def __init__(self, checked=False):
        self.checked = bool(checked)

    def isChecked(self):
        return self.checked


class _SaveTimer:
    """``_chat_history_save_timer`` shim: ``start()`` marks the host's history dirty."""

    def __init__(self, host):
        self._host = host

    def start(self):
        self._host._history_dirty = True
        callback = getattr(self._host, "on_schedule_save", None)
        if callable(callback):
            callback()


def default_history_path():
    """``GLOSSARION_DIRECT_TEXT_HISTORY`` or ``direct_text_chats.json`` beside ``CONFIG_FILE``."""
    return ChatStoreMixin._resolve_chat_history_path()


class ChatStore(ChatStoreMixin):
    """GUI-free host of the Direct Text chat store (mobile ``ChatStoreAdapter``, tests).

    It holds the chat state the dialog keeps (``_chat_history_path``, ``_chat_sessions``,
    ``_current_chat_index``, ``_chat_messages``, caches) and implements the dialog hooks
    without Qt: notices are collected in ``notices``, the merge confirmation asks
    ``confirm_merge(target_folder)`` (default: cancel), status text goes to ``status_text``.

    Session-scoped public methods take the session dict itself (the mobile adapter owns
    the list): the session is made current for the call, the moved method runs, and one
    history save follows when the moved code asked for it (``_save_chat_history``).

    ``output_root`` (default: ``OUTPUT_DIRECTORY`` at construction) plays the role of the
    dialog's ``_saved_env['OUTPUT_DIRECTORY']`` (the user's output root outside the run),
    so new chat folders land in ``<output_root>/Direct Text``.
    """

    def __init__(self, history_path=None, *, output_root=None, config=None, load=False):
        self._chat_history_path = (
            os.path.abspath(os.path.expanduser(str(history_path)))
            if history_path else self._resolve_chat_history_path()
        )
        self.translator = DirectTextOwnerView(config)
        self._saved_env = {key: os.environ.get(key) for key in self._OUTPUT_ENV_KEYS}
        if output_root:
            self._saved_env['OUTPUT_DIRECTORY'] = os.path.abspath(os.path.expanduser(str(output_root)))
        self._chat_sessions = [self._new_chat_session(1)]
        self._chat_session_counter = 1
        self._current_chat_index = 0
        self._chat_messages = self._chat_sessions[0]["messages"]
        self._pending_attachment = None
        self._expanded_processing_messages = set()
        self._message_text_cache = {}
        self._rendered_message_cache = {}
        self._last_output_folder = ""
        self._active = False
        self._chat_history_save_timer = _SaveTimer(self)
        self._history_dirty = False
        self._save_depth = 0
        self._save_pending = False
        self.force_no_glossary_radio = _CheckFlag(False)
        self.status_text = "Ready"
        self.notices = []
        self.confirm_merge = None
        self.on_schedule_save = None
        self.lock = threading.RLock()
        if load:
            self.load_chat_history()

    # ---- dialog hooks (GUI-free) --------------------------------------------------------

    def _refresh_chat_list(self):
        """No chat list widget: listeners refresh from the data."""

    def _render_output(self, active_only=False, preserve_viewport=False):
        """No transcript widget: the mobile chat renders from the data."""

    def _set_status(self, text, output_folder=""):
        self.status_text = str(text)

    def _direct_text_notice(self, level, parent, title, text):
        self.notices.append({"level": str(level), "title": str(title), "text": str(text)})

    def _confirm_attachment_merge(self, message_parent, target_folder):
        confirm = self.confirm_merge
        return bool(confirm(target_folder)) if callable(confirm) else False

    def _save_chat_history(self):
        """Saves requested inside a session scope run once, after the scope."""
        if self._save_depth:
            self._save_pending = True
            return
        ChatStoreMixin._save_chat_history(self)
        self._history_dirty = False

    # ---- session scope ----------------------------------------------------------------------

    @contextlib.contextmanager
    def session_scope(self, session):
        """Make *session* the current chat for one operation (adds it when unknown)."""
        with self.lock:
            previous = self._current_chat_session()
            if session is not None:
                if all(existing is not session for existing in self._chat_sessions):
                    self._chat_sessions.append(session)
                if not isinstance(session.get("messages"), list):
                    session["messages"] = list(session.get("messages") or [])
                self._current_chat_index = next(
                    index for index, existing in enumerate(self._chat_sessions) if existing is session
                )
                self._chat_messages = session["messages"]
            self._save_depth += 1
            try:
                yield session
            finally:
                self._save_depth -= 1
                if previous is not None and any(existing is previous for existing in self._chat_sessions):
                    self._current_chat_index = next(
                        index for index, existing in enumerate(self._chat_sessions) if existing is previous
                    )
                current = self._current_chat_session()
                if current is not None:
                    self._chat_messages = current["messages"]
                if not self._save_depth and self._save_pending:
                    self._save_pending = False
                    self._save_chat_history()

    # ---- history ------------------------------------------------------------------------------

    def load_chat_history(self):
        """``(sessions, current_chat_id)``; the store keeps them (dialog ``__init__`` order)."""
        with self.lock:
            current_chat_id = self._init_chat_sessions()
            self._message_text_cache.clear()
            return self._chat_sessions, current_chat_id

    def save_chat_history(self, sessions=None, current_chat_id=None):
        """Externalise bodies and write the v2 file atomically for *sessions* (default: the store's)."""
        with self.lock:
            if sessions is not None:
                self._chat_sessions = sessions
            if current_chat_id is not None:
                self._current_chat_index = next(
                    (index for index, session in enumerate(self._chat_sessions)
                     if session.get("id") == current_chat_id),
                    self._current_chat_index,
                )
            if not (0 <= self._current_chat_index < len(self._chat_sessions)):
                self._current_chat_index = 0
            current = self._current_chat_session()
            if current is not None:
                self._chat_messages = current["messages"]
            self._save_chat_history()

    def new_chat_session(self, session_id):
        return self._new_chat_session(session_id)

    def current_session(self):
        return self._current_chat_session()

    def set_current_session(self, session):
        """Make *session* the store's current chat (saved as ``current_chat_id``)."""
        with self.lock:
            if all(existing is not session for existing in self._chat_sessions):
                self._chat_sessions.append(session)
            self._current_chat_index = next(
                index for index, existing in enumerate(self._chat_sessions) if existing is session
            )
            self._chat_messages = session["messages"]

    def history_file_reference(self, path):
        return self._history_file_reference(path)

    def resolve_history_file_reference(self, reference):
        return self._resolve_history_file_reference(reference)

    def assistant_message_text(self, session, message_index, kind="content"):
        """Lazy body of one assistant message (inline text or its ``Chat Messages/`` file)."""
        with self.session_scope(session):
            return self._assistant_message_text(self._chat_messages[int(message_index)], kind, int(message_index))

    def title_chat_from_text(self, session, text):
        """Auto-title on the first send (``_title_current_chat_from_text``)."""
        with self.session_scope(session):
            self._title_current_chat_from_text(text)
            return session.get("title")

    # ---- folders and attachments ----------------------------------------------------------------

    def ensure_conversation_output_folder_for_session(self, session):
        with self.session_scope(session):
            return self._ensure_conversation_output_folder_for_session(session)

    def conversation_output_folder(self, session):
        with self.session_scope(session):
            return self._conversation_output_folder()

    @staticmethod
    def validated_chat_output_folder(session):
        return ChatStoreMixin._validated_chat_output_folder(session)

    @staticmethod
    def managed_conversation_output_folder(session):
        return ChatStoreMixin._managed_conversation_output_folder(session)

    def conversation_attachment_folders(self, session):
        with self.session_scope(session):
            return self._conversation_attachment_folders(session)

    def is_managed_attachment_workspace(self, session, folder):
        return self._is_managed_attachment_workspace(session, folder)

    def attachment_compiled_epub_path(self, folder):
        return self._attachment_compiled_epub_path(folder)

    def attachment_card_actions(self, session, output_folder):
        return self._attachment_card_actions(session, output_folder)

    def migration_output_root(self, session):
        conversation_folder = self._managed_conversation_output_folder(session)
        return self._direct_text_migration_output_root(conversation_folder)

    def migrate_attachment(self, session, source_folder, *, confirm_merge=None):
        """Desktop Migrate: move one ``Attachments/<stem>`` workspace to the output root.

        Returns ``{"ok", "notices"}``; a name collision asks ``confirm_merge(target)``
        ("Merge and replace" when it returns True; Cancel otherwise).
        """
        with self.lock:
            self.notices = []
            previous_confirm = self.confirm_merge
            if confirm_merge is not None:
                self.confirm_merge = confirm_merge
            try:
                with self.session_scope(session):
                    index = self._current_chat_index
                    ok = self._migrate_conversation_attachment(index, source_folder)
            finally:
                self.confirm_merge = previous_confirm
            return {"ok": bool(ok), "notices": list(self.notices)}

    # ---- responses ------------------------------------------------------------------------------

    def editable_response_paths(self, session, message_index):
        with self.session_scope(session):
            return self._editable_response_paths(message_index)

    def save_response_output_edit(self, session, message_index, edited_source, *, text_source=None,
                                  html_source=None):
        """Write an edited response (md/txt/html/xhtml + the real chapter file), then save."""
        with self.session_scope(session):
            return self._save_response_output_edit(
                message_index, edited_source, text_source=text_source, html_source=html_source
            )

    # ---- runs -----------------------------------------------------------------------------------

    def commit_request_phase(self, session, stream):
        """The glossary gate: freeze the run's real request cards into *session*, then save.

        The dialog's ``_commit_active_request_phase`` on the chat while the run waits for
        the Edit / Yes / No answer: *stream* is the run's live
        ``direct_text_stream.DirectTextStream`` (JobService feeds it); its cards with
        content, thinking or tokens become v2 assistant messages of the session and the
        live stream starts the next phase empty. Returns the committed messages.
        """
        from direct_text_stream import DirectTextStream

        with self.lock:
            with self.session_scope(session):
                host = DirectTextStream(store=self)
                host.adopt_stream_state(stream)
                before = len(self._chat_messages)
                host._commit_active_request_phase()
                committed = list(self._chat_messages[before:])
            # the live stream drops the committed cards (its own message list is never saved)
            stream.commit_active_request_phase()
            return committed

    def finish_run(self, session, run, segments=None, *, cleanup=True):
        """The dialog's ``_finish_translation`` for one finished chat run, GUI-free.

        *run* holds the run state (``temp_root``, ``source_path``, ``source_extension``,
        ``is_attachment``, ``expected_output``, ``manual_glossary_path``, ``started_at``,
        ``output_mode``; optional ``force_no_glossary``, ``glossary_path`` (the glossary
        the pipeline used), ``created_at``, ``streamed_content``). *segments* is the run's
        ``direct_text_stream.DirectTextStream`` (its full state is used) or a list of its
        request segments. The translated output is persisted into the chat folder, the
        cards (+ "Extraction report" / "Attachment actions") are committed to the session
        and the history is saved. Returns ``{output_folder, status, messages, ...}``.
        """
        from direct_text_stream import DirectTextStream

        with self.lock:
            with self.session_scope(session):
                host = DirectTextStream(store=self, cleanup_temp_root=cleanup)
                host.load_run(run)
                if isinstance(segments, DirectTextStream):
                    host.adopt_stream_state(segments)
                else:
                    host.load_segments(segments or [], streamed_content=(run or {}).get("streamed_content"))
                before = len(self._chat_messages)
                host._active = True
                host._assistant_message_active = True
                host._finish_translation()
                result = {
                    "output_folder": host._last_output_folder,
                    "status": host.status_text,
                    "messages": list(self._chat_messages[before:]),
                    "persisted_output_path": host._persisted_output_path,
                    "expected_output": host._expected_output,
                    "preserve_temp_root": host._preserve_temp_root,
                }
            return result


format_attachment_size = ChatStoreMixin._format_attachment_size
force_no_glossary_for_mode = ChatStoreMixin._force_no_glossary_for_mode
history_window_bounds = ChatStoreMixin._history_window_bounds
markup_to_html = ChatStoreMixin._markup_to_html
normalize_rendered_card_limit = ChatStoreMixin._normalize_rendered_card_limit
normalize_attachment_record = ChatStoreMixin._normalize_attachment_record


def timestamp_label(created_at):
    """Card header time: "HH:MM" today, "Mon DD · HH:MM" this year, else with the year."""
    return ChatStoreMixin._assistant_timestamp_label(
        ChatStoreMixin(), ("assistant", "", "", "", "", "", {"created_at": created_at})
    )


def prepare_direct_text_input(text, attachment=None, manual_glossary_source=None, *, temp_parent=None,
                              extra_pass_through_extensions=()):
    """The send-time input preparation of ``_start_translation`` (temp root + input file).

    *attachment* is a v2 attachment record (``path``, ``name``, ...) or None for typed
    text; *manual_glossary_source* the "Provide Manual Glossary" record (``kind`` 'path'
    with ``path``, or 'content' with ``content`` + ``extension``). Returns the run state
    ``{temp_root, temp_input, source_path, source_extension, is_attachment,
    expected_output, manual_glossary_path, started_at}``.

    Mobile only: *temp_parent* puts the temp root in an app folder that survives a restart
    (Resume continues from its ``translation_progress.json``) instead of the OS temp dir,
    and *extra_pass_through_extensions* are attachment types the chat accepts beyond the
    desktop dialog (handed to the pipeline as-is). Without them this is the dialog's code.
    """
    host = ChatStoreMixin()
    if temp_parent:
        temp_parent = os.path.abspath(os.path.expanduser(str(temp_parent)))
        os.makedirs(temp_parent, exist_ok=True)
        host._DIRECT_TEXT_TEMP_PARENT = temp_parent
    if extra_pass_through_extensions:
        host._EXTRA_PASS_THROUGH_EXTENSIONS = frozenset(
            str(extension).lower() for extension in extra_pass_through_extensions
        )
    attachment = ChatStoreMixin._normalize_attachment_record(attachment)
    manual_glossary_path = host._prepare_direct_text_input(text, attachment, manual_glossary_source)
    return {
        "temp_root": host._temp_root,
        "temp_input": host._temp_input,
        "source_path": host._run_source_path,
        "source_extension": host._run_source_extension,
        "is_attachment": host._run_source_is_attachment,
        "expected_output": host._expected_output,
        "manual_glossary_path": manual_glossary_path,
        "started_at": host._run_started_at,
    }


def apply_direct_text_run_environment(owner, output_root, is_attachment):
    """The Direct Text run environment of ``_start_translation`` (after ``DirectTextRunOptions``).

    Points the run's outputs at *output_root*, marks the run as Direct Text, orders
    attachment batches by spine, then applies the owner's forced streaming and
    ``_apply_direct_text_runtime_environment`` (the same order as the dialog).
    """
    os.environ['OUTPUT_DIRECTORY'] = output_root
    os.environ['OUTPUT_DIR'] = output_root
    os.environ['DIRECT_TEXT_ACTIVE'] = '1'
    os.environ['DIRECT_TEXT_PRESERVE_MARKUP'] = '1'
    os.environ['DIRECT_TEXT_ORDERED_BATCH'] = (
        '1' if is_attachment else '0'
    )
    if is_attachment:
        os.environ['ORDER_BATCH_REQUESTS_BY_SPINE'] = '1'
    owner._apply_forced_streaming_environment()
    owner._apply_direct_text_runtime_environment()
