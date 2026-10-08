"""direct_text_stream: the Direct Text log-stream model (DirectTextStreamMixin) and run host.

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). The desktop Direct Text
dialog (``translator_gui._InputOutputDialog``) builds its chat from the main log stream:
``_on_log_line`` (any thread) tracks a phase per worker thread and queues each record;
``_drain_log_queue`` classifies every line (``_classify_line``) and grows one *request
segment* per outbound API call, ordered by spine (``[spine-order:N]`` dispatch records,
``[DIRECT_TEXT_RESPONSE_PAYLOAD]`` / ``[DIRECT_TEXT_GLOSSARY_STREAM_START]`` markers,
NVIDIA queue / thinking / text phases, tiktoken counts with the ``encoding_for_model`` ->
``o200k_base`` -> ``cl100k_base`` fallback); ``_commit_*`` freezes the segments into v2
assistant messages and ``_finish_translation`` completes a run. Those members moved here
verbatim (``translator_gui.py`` @ 1719fb59: 1934-1993 constants, 4158-4167, 7795-7871,
9393-9926, 9976-10415, 11823-11973, 12473-12877); the dialog inherits
``DirectTextStreamMixin`` first and keeps the Qt rendering (``_render_output``, timers).

``DirectTextStream`` is the GUI-free host (both mixins): one per run on mobile (the chat's
``RunStream`` and JobService feed it raw log lines), and the engine behind
``direct_text_store.ChatStore.finish_run``.

Edits made while moving (everything else is byte-for-byte):

* ``_build_attachment_action_card`` / ``_build_attachment_extraction_summary`` are split
  into data builders (``attachment_action_data``, ``attachment_extraction_report``) and
  the persisted card format (``attachment_action_card``, ``extraction_report_card``);
  the methods gather the same inputs in the same order and return the same tuples.

Hooks the moved code calls (the dialog implements them with Qt; ``DirectTextStream``
GUI-free): ``_schedule_stream_render``, ``_render_output``,
``_update_history_window_after_append``, ``_set_status``, ``_restore_run_context``
(see ``STREAM_HOOKS``).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import os
import re
import shutil
import threading
from collections import deque

from direct_text_store import ChatStoreMixin, DirectTextOwnerView, _CheckFlag, _SaveTimer

__all__ = [
    "DirectTextStream",
    "DirectTextStreamMixin",
    "RequestStream",
    "STREAM_HOOKS",
    "attachment_action_card",
    "attachment_action_data",
    "attachment_extraction_report",
    "classify_request_issue",
    "is_safety_block_line",
    "count_tokens",
    "extraction_report_card",
    "make_stream",
    "request_segment_message",
]

#: Wait states the API client logs while a request cannot proceed (first match wins): every key
#: of a multi-key pool cooling down, a provider rate limit (HTTP 429 / quota) and a lost network.
#: Glossarion Mobile turns them into the running card's issue chips (UI_SPEC §2.12.3); the
#: desktop log is unchanged.
_REQUEST_ISSUE_PATTERNS = (
    ("key_cooling", re.compile(
        r"all keys (?:are )?rate-limited|waiting for (?:key )?cooldown|keys? (?:on|in) cooldown|"
        r"cooling down|cooldown remaining", re.IGNORECASE)),
    ("rate_limited", re.compile(
        r"rate[- ]?limit|\b429\b|quota exhausted|too many requests|resource[_ ]exhausted", re.IGNORECASE)),
    ("network_wait", re.compile(
        r"waiting for network|network (?:is )?unreachable|no internet|internet connection|"
        r"connection (?:error|reset|refused|aborted)|failed to establish a new connection|"
        r"name resolution|getaddrinfo failed|temporary failure in name resolution", re.IGNORECASE)),
)
_ISSUE_WAIT_SECONDS = re.compile(
    r"(?:retrying in|retry in|waiting|sleeping|wait(?:ing)? for|retry-after:?)\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b",
    re.IGNORECASE)


def classify_request_issue(line):
    """``(kind, seconds)`` for an API-client wait line, else None.

    ``kind`` is ``"key_cooling"``, ``"rate_limited"`` or ``"network_wait"``; ``seconds`` is the wait
    the line announces ("waiting 30.0s", "retrying in 12s", "retry-after: 20") or None. Lines that
    report a failure the client does not wait on still match (the caller expires the state)."""
    text = str(line or "")
    if not text.strip():
        return None
    for kind, pattern in _REQUEST_ISSUE_PATTERNS:
        if pattern.search(text):
            seconds = None
            match = _ISSUE_WAIT_SECONDS.search(text)
            if match:
                try:
                    seconds = float(match.group(1))
                except ValueError:
                    seconds = None
            return kind, seconds
    return None


#: Lines the client and the translator log when a provider's safety filter blocked a request:
#: ``UnifiedClientError(error_type="prohibited_content")`` ("Content blocked: Google Generative AI
#: Prohibited Use policy", gemini_policy), a ``prohibited_content`` / ``content_filter`` finish
#: reason, TransateKRtoEN's "hit content filter/prohibited".
_SAFETY_BLOCK_PATTERN = re.compile(
    r"content blocked|prohibited[ _]use policy|prohibited[_ ]content|hit content filter|"
    r"finish[_ ]reason\W{0,3}(?:safety|content_filter|prohibited)|blocked by (?:the )?(?:provider'?s? )?safety",
    re.IGNORECASE)


def is_safety_block_line(line):
    """True for a log line that reports a provider safety / prohibited-use block."""
    return bool(_SAFETY_BLOCK_PATTERN.search(str(line or "")))


#: Dialog hooks the moved code calls (implemented by _InputOutputDialog and DirectTextStream).
STREAM_HOOKS = (
    "_schedule_stream_render",
    "_render_output",
    "_update_history_window_after_append",
    "_set_status",
    "_restore_run_context",
)


class DirectTextStreamMixin:
    """Direct Text log-stream classifier and request segments moved verbatim from ``_InputOutputDialog``."""

    _STATUS_FIRST_CHARS = set(
        "🚀📄📃📜📋✅⚠❌📚📦🔧📊🔍💾🖼🔄📌📸🧠🛰📡⏱⏳"
        "🟢🟡🟠🔴🎯📑📖🌐⚡🧪✨🎨💡🔠🗑🧹📂📁🔁🔂📝🔑🗝"
        "🔒🔓🚫⛔🛑⏹💬🌍🌏🌎🐛📈📉🤖🆕═─=[#"
    )
    _PIPELINE_LOG_PHRASES = (
        'translation stopped',
        'force stop requested',
        'graceful stop',
        'stream finished',
        'stream finished in',
        'stream complete',
        'sdk call finished',
        'received translation from api',
        'received section ',
        'fallback key ',
        'saved text file',
        'total translation time',
        'chapters completed',
        'text file translation complete',
        'translation completed successfully',
    )
    _EMBEDDED_PIPELINE_MARKERS = (
        '⏹️', '⏹', '🛑',
        '📂 Payloads directory:',
        '🛰️ [gemini-native] Stream finished',
        '🛰️ [gemini-grpc] Stream finished',
        '🛰️ [anthropic] SSE stream complete',
        '🛰️ [mistral] SDK stream complete',
        '📡 AuthGPT: Stream finished',
        '📡 AuthGem: Stream finished',
        '📡 AuthCD: Stream finished',
        '📡 AuthND: Stream finished',
        'TRANSLATION_COMPLETE_SIGNAL', 'GLOSSARY_COMPLETE_SIGNAL',
        '✅ Text file translation complete',
    )
    _STREAM_END_LOG_PHRASES = (
        'stream finished',
        'stream complete',
        'sdk call finished',
        'received translation from api',
        'received chapter ',
        'received section ',
        'received merged response',
    )
    _HIDDEN_STREAM_START_LOG_PHRASES = (
        '] streaming on (env=',
        '] stream start (model=',
        '] sse stream start (model=',
        '] sdk stream start (model=',
        '] sdk stream opened in ',
        'sending api call in ',
        ': sending request via codex api (',
        ': sending request via anthropic messages api (',
        'gemini safety status:',
    )
    _DIRECT_RESPONSE_PAYLOAD_PREFIX = '[DIRECT_TEXT_RESPONSE_PAYLOAD] '
    _DIRECT_GLOSSARY_STREAM_START_PREFIX = (
        '[DIRECT_TEXT_GLOSSARY_STREAM_START] '
    )

    def _allocate_active_request_number(self):
        """Reserve one conversation-wide request number for the active run."""
        try:
            request_number = int(self._active_request_next_number)
        except (AttributeError, TypeError, ValueError):
            request_number = 0
        if request_number < 1:
            request_number = self._next_conversation_request_number()
        self._active_request_next_number = request_number + 1
        return request_number

    def _set_thinking_toggle_text(self, text):
        """Update the inline processing disclosure label."""
        self._processing_label_text = str(text)

    def _refresh_processing_label(self):
        """Show the active phase and its tiktoken-counted text volume."""
        if self._active and self._in_thinking:
            label = "Thinking"
            count = self._thinking_token_count
        elif self._active and self._streaming_text:
            label = "Generating Text"
            count = self._generation_token_count
        elif not self._active and (
            self._thinking_token_count or self._generation_token_count
        ):
            self._set_thinking_toggle_text(
                "Token summary  ·  "
                f"Thinking {self._thinking_token_count:,}  ·  "
                f"Text {self._generation_token_count:,}"
            )
            return
        else:
            label = "Processing"
            count = 0
        suffix = f" ({count:,} tokens)" if count else ""
        self._set_thinking_toggle_text(label + suffix)

    def _get_token_encoder(self):
        """Resolve and cache the closest tiktoken encoder for the selected model."""
        if self._token_encoder_initialized:
            return self._token_encoder

        self._token_encoder_initialized = True
        try:
            import tiktoken

            model_name = str(
                getattr(self.translator, 'model_var', '')
                or getattr(self.translator, 'config', {}).get('model', '')
                or os.environ.get('MODEL', '')
            ).strip()
            candidates = [model_name]
            if '/' in model_name:
                candidates.append(model_name.rsplit('/', 1)[-1])

            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    self._token_encoder = tiktoken.encoding_for_model(candidate)
                    return self._token_encoder
                except KeyError:
                    continue

            try:
                self._token_encoder = tiktoken.get_encoding('o200k_base')
            except Exception:
                self._token_encoder = tiktoken.get_encoding('cl100k_base')
        except Exception:
            self._token_encoder = None
        return self._token_encoder

    def _count_tokens(self, text):
        """Count visible phase text with tiktoken; return zero if unavailable."""
        value = str(text or "").replace('\u200b', '')
        if not value.strip():
            return 0
        encoder = self._get_token_encoder()
        if encoder is None:
            return 0
        try:
            return len(encoder.encode(value, disallowed_special=()))
        except Exception:
            return 0

    def _set_thinking_spinner_active(self, active):
        self._processing_spinner_active = bool(active)

    def _request_label_from_log(self, line, fallback_number):
        """Extract the friendly chapter/chunk label from an API-start record."""
        import re

        # Typed Direct Text is adapted to a temporary one-file document only
        # so it can reuse the normal translation pipeline. Its synthetic
        # chapter/chunk identity is an implementation detail, not useful chat
        # metadata. Real attachments retain their spine/chunk labels.
        if not self._run_source_is_attachment:
            return f"Request {int(fallback_number)}"

        value = str(line or "")
        metadata_match = re.search(
            r"\b(?:header|toc)\s+batch\s+\d+\s*/\s*\d+",
            value,
            flags=re.IGNORECASE,
        )
        if metadata_match:
            label = " ".join(metadata_match.group(0).split())
            return label[0].upper() + label[1:]
        if re.search(r"\brequest\s*\(\s*metadata\s*\)", value, re.IGNORECASE):
            # This is a provider lifecycle label, not generated response text.
            # Keep its temporary in-flight card readable; finalization removes
            # it if the provider never emits model content for this channel.
            return "Metadata translation"
        chapter_match = re.search(
            r"\b(chapter|section)\s+([^\s(),:]+)"
            r"(?:\s*(?:\(|,)?\s*chunk\s+(\d+)\s*/\s*(\d+)\s*\)?)?",
            value,
            flags=re.IGNORECASE,
        )
        if chapter_match:
            label = f"{chapter_match.group(1)} {chapter_match.group(2)}"
            if chapter_match.group(3) and chapter_match.group(4):
                label += (
                    f" (chunk {chapter_match.group(3)}/{chapter_match.group(4)})"
                )
            source_match = re.search(
                r"(?:·|\[File:\s*)\s*([^·\[\]\r\n]+?\.(?:xhtml|html|htm))",
                value,
                flags=re.IGNORECASE,
            )
            if source_match:
                source_name = source_match.group(1).strip().replace("\\", "/")
                source_name = source_name.rsplit("/", 1)[-1]
                if source_name:
                    label += f" · {source_name}"
            return label[0].upper() + label[1:]
        merged_match = re.search(
            r"\b(?:merged|request)\s+.+?(?=\s+response\b|\s*[\[(]|$)",
            value,
            flags=re.IGNORECASE,
        )
        if merged_match:
            return " ".join(merged_match.group(0).split())[:80]
        return f"Request {int(fallback_number)}"

    def _request_segment_for_completion_log(self, line, source_thread=None):
        """Resolve a completion record to its request even across worker threads."""
        direct = self._request_segment_for_thread(source_thread, create=False)
        label = self._request_label_from_log(line, 1)
        if label.startswith("Request "):
            return direct

        wanted = " ".join(label.lower().split())
        for segment in reversed(self._active_request_segments):
            existing = " ".join(
                str(segment.get("label", "") or "").lower().split()
            )
            if existing == wanted or existing.startswith(wanted + " "):
                return segment
        return direct

    def _apply_direct_response_payload(self, line, source_thread=None):
        """Install a backend response that cannot be exposed as token logs."""
        import json as json_lib
        import re

        raw = str(line or "")
        if not raw.startswith(self._DIRECT_RESPONSE_PAYLOAD_PREFIX):
            return False
        try:
            payload = json_lib.loads(
                raw[len(self._DIRECT_RESPONSE_PAYLOAD_PREFIX):]
            )
        except (TypeError, ValueError):
            return True
        if not isinstance(payload, dict):
            return True

        label = str(payload.get("label", "") or "").strip()
        content = str(payload.get("content", "") or "")
        thinking = str(payload.get("thinking", "") or "")
        if not label or not content:
            return True
        order = payload.get("order")
        request_number = payload.get("request_number")
        target = None
        normalized_order = None
        try:
            normalized_order = int(order)
        except (TypeError, ValueError):
            pass
        try:
            request_number = int(request_number)
            if request_number < 1:
                request_number = None
        except (TypeError, ValueError):
            request_match = re.fullmatch(
                r"Request\s+(\d+)", label, flags=re.IGNORECASE
            )
            request_number = (
                int(request_match.group(1)) if request_match else None
            )

        payload_source_thread = str(
            payload.get("source_thread", "") or source_thread or ""
        )
        thread_target = self._request_segment_for_thread(
            payload_source_thread, create=False
        )

        # Dispatch order is the stable request identity.  A provider may lose
        # chapter context at completion and call the same response ``Request
        # 11``; bind it back to the existing spine card instead of creating a
        # second card or overwriting the chapter label.
        if normalized_order is not None:
            for segment in self._active_request_segments:
                order_key = tuple(segment.get("order_key", ()))
                if len(order_key) > 1 and order_key[1] == normalized_order:
                    target = segment
                    break
        wanted = " ".join(label.lower().split())
        if target is None:
            for segment in reversed(self._active_request_segments):
                existing = " ".join(
                    str(segment.get("label", "") or "").lower().split()
                )
                if existing == wanted or existing.startswith(wanted + " "):
                    target = segment
                    break
        if target is None and thread_target is not None:
            target = thread_target
        if target is None:
            marker = ""
            if normalized_order is not None:
                marker = f" [spine-order:{normalized_order}]"
            target = self._begin_request_segment(
                f"{label}{marker} Direct Text dispatch", ""
            )

        # Older/provider retry paths can momentarily open a generic card on
        # the API worker after the ordered chapter card already exists.  The
        # final payload supplies both identities, so fold that transient card
        # into the spine card instead of persisting a second ``Request N``.
        if (
            thread_target is not None
            and thread_target is not target
            and str(thread_target.get("label", "") or "")
            .strip().lower().startswith("request ")
        ):
            redundant_text_tokens = int(
                thread_target.get("text_tokens", 0) or 0
            )
            redundant_thinking = str(
                thread_target.get("thinking", "") or ""
            )
            if redundant_thinking:
                existing_thinking = str(target.get("thinking", "") or "")
                if redundant_thinking not in existing_thinking:
                    target["thinking"] = existing_thinking + redundant_thinking
                    target["thinking_tokens"] = int(
                        target.get("thinking_tokens", 0) or 0
                    ) + int(thread_target.get("thinking_tokens", 0) or 0)
            aliases = list(target.get("thread_aliases", []) or [])
            for alias in (
                thread_target.get("thread", ""),
                *(thread_target.get("thread_aliases", []) or []),
            ):
                alias = str(alias or "").strip()
                if alias and alias not in aliases:
                    aliases.append(alias)
            target["thread_aliases"] = aliases
            self._generation_token_count = max(
                0, self._generation_token_count - redundant_text_tokens
            )
            self._active_request_segments.remove(thread_target)

        target_request_number = int(
            target.get("request_number", 0)
            or self._allocate_active_request_number()
        )
        target["request_number"] = target_request_number
        if not self._run_source_is_attachment:
            # Plain Direct Text is translated through a synthetic one-file
            # document. Structured completion payloads can expose that
            # adapter's chapter/chunk identity even though no attachment was
            # submitted. Never persist implementation-only document metadata
            # on a normal chat response.
            label = f"Request {target_request_number}"
        elif label.lower().startswith("request "):
            # Provider payload numbers are local to one backend invocation.
            # The Direct Text card number is conversation-wide.
            label = f"Request {target_request_number}"

        previous_tokens = int(target.get("text_tokens", 0) or 0)
        previous_thinking_tokens = int(
            target.get("thinking_tokens", 0) or 0
        )
        current_tokens = self._count_tokens(content)
        current_thinking_tokens = (
            self._count_tokens(thinking) if thinking else previous_thinking_tokens
        )
        existing_label = str(target.get("label", "") or "").strip()
        incoming_generic = label.lower().startswith("request ")
        existing_generic = existing_label.lower().startswith("request ")
        incoming_has_chunk = "(chunk " in label.lower()
        existing_has_chunk = "(chunk " in existing_label.lower()
        incoming_has_html_file = bool(
            re.search(r"\.(?:xhtml|html|htm)(?:\s|$)", label, re.IGNORECASE)
        )
        existing_has_html_file = bool(
            re.search(
                r"\.(?:xhtml|html|htm)(?:\s|$)",
                existing_label,
                re.IGNORECASE,
            )
        )
        if (
            not existing_label
            or existing_generic
            or (
                not incoming_generic
                and incoming_has_chunk
                and not existing_has_chunk
            )
            or (
                not incoming_generic
                and incoming_has_html_file
                and not existing_has_html_file
            )
        ):
            target["label"] = label
        target["content"] = content
        target["text_tokens"] = current_tokens
        if thinking:
            # The glossary backend publishes the complete captured reasoning
            # channel at request completion.  Replace any partially rendered
            # live copy so retries/line-buffer flushes cannot duplicate it.
            target["thinking"] = thinking
            target["thinking_tokens"] = current_thinking_tokens
        target["status_only"] = False
        target["phase"] = "processing"
        target["complete"] = True
        self._generation_token_count += current_tokens - previous_tokens
        if thinking:
            self._thinking_token_count += (
                current_thinking_tokens - previous_thinking_tokens
            )
        self._sort_active_request_segments()
        return True

    def _apply_direct_glossary_stream_start(self, line, source_thread=None):
        """Bind one live glossary provider stream to a stable response card."""
        import json as json_lib

        raw = str(line or "")
        if not raw.startswith(self._DIRECT_GLOSSARY_STREAM_START_PREFIX):
            return False
        try:
            payload = json_lib.loads(
                raw[len(self._DIRECT_GLOSSARY_STREAM_START_PREFIX):]
            )
        except (TypeError, ValueError):
            return True
        if not isinstance(payload, dict):
            return True

        label = str(payload.get("label", "") or "Glossary request").strip()
        payload_thread = str(
            payload.get("source_thread", "") or source_thread or ""
        ).strip()
        callback_thread = str(source_thread or "").strip()
        segment = self._begin_request_segment(
            f"{label} Direct Text glossary dispatch",
            payload_thread,
        )
        if label and self._run_source_is_attachment:
            segment["label"] = label
        segment["glossary_stream"] = True
        segment["phase"] = "processing"
        segment["complete"] = False

        aliases = list(segment.get("thread_aliases", []) or [])
        for alias in (payload_thread, callback_thread):
            if alias and alias != segment.get("thread", "") and alias not in aliases:
                aliases.append(alias)
        segment["thread_aliases"] = aliases
        self._sort_active_request_segments()
        return True

    @staticmethod
    def _request_order_from_log(line, fallback_number):
        """Return a stable spine/chapter/chunk sort key for a request card."""
        import re

        value = str(line or "")
        dispatch_match = re.search(
            r"\[spine-order:(\d+)\]", value, flags=re.IGNORECASE
        )
        if dispatch_match:
            primary = int(dispatch_match.group(1))
            source_rank = 0
        else:
            chapter_match = re.search(
                r"\b(?:chapter|section|merged)\s+(-?\d+(?:\.\d+)?)",
                value,
                flags=re.IGNORECASE,
            )
            if chapter_match:
                try:
                    primary = float(chapter_match.group(1))
                except ValueError:
                    primary = int(fallback_number)
                source_rank = 1
            else:
                primary = int(fallback_number)
                source_rank = 2
        chunk_match = re.search(
            r"\bchunk\s+(\d+)\s*/\s*(\d+)",
            value,
            flags=re.IGNORECASE,
        )
        chunk = int(chunk_match.group(1)) if chunk_match else 0
        return (source_rank, primary, chunk, int(fallback_number))

    def _sort_active_request_segments(self):
        """Keep cards deterministic without losing their thread lookup."""
        self._active_request_segments.sort(
            key=lambda segment: tuple(
                segment.get("order_key", (2, float("inf"), 0, 0))
            )
        )
        self._request_segment_by_thread = {}
        for index, segment in enumerate(self._active_request_segments):
            thread_keys = [segment.get("thread", "")]
            thread_keys.extend(segment.get("thread_aliases", []) or [])
            for thread_key in thread_keys:
                thread_key = str(thread_key or "").strip()
                if thread_key:
                    self._request_segment_by_thread[thread_key] = index

    @staticmethod
    def _thread_key_from_log(line, source_thread=None):
        import re

        # An API worker is sometimes mentioned inside a record that is emitted
        # by its parent orchestration thread. Prefer that explicit worker ID;
        # generic provider tags such as ``[gemini-native]`` are not request IDs.
        # Browser-backed providers can also consume their stream on an
        # interruptible transport helper.  Those helpers retain the logical
        # worker as ``ProviderTransport[Thread-N (...)]`` so every stream event
        # and the final response resolve to one Direct Text request card.
        transport_match = re.fullmatch(
            r"[^\[\]]*Transport\[(.+)\]",
            str(source_thread or "").strip(),
            flags=re.IGNORECASE,
        )
        if transport_match:
            return transport_match.group(1).strip()
        for candidate in (line, source_thread):
            match = re.search(
                r"\[((?:Thread|Dummy)-[^\]]+)\]",
                str(candidate or ""),
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        return str(source_thread or "").strip()

    def _begin_request_segment(self, line, source_thread=None):
        """Create one independently rendered response for an outbound API call."""
        thread_key = self._thread_key_from_log(line, source_thread)
        segment_number = len(self._active_request_segments) + 1
        request_label = self._request_label_from_log(line, segment_number)
        incoming_generic = request_label.lower().startswith("request ")
        existing_index = self._request_segment_by_thread.get(thread_key)
        if (
            thread_key
            and existing_index is not None
            and 0 <= existing_index < len(self._active_request_segments)
        ):
            existing = self._active_request_segments[existing_index]
            existing_label = str(existing.get("label", "") or "")
            existing_generic = existing_label.lower().startswith("request ")
            # Provider/client layers can report "API call in progress" more
            # than once for the same worker thread. Those are status updates,
            # not new outbound requests, even after thinking has started.
            # Generic retry/status records stay on the same request even after
            # a provider has emitted an early stream-complete boundary.  A
            # genuinely new request supplies a new detailed dispatch label.
            if not existing.get("complete") or incoming_generic:
                incoming_order_key = self._request_order_from_log(
                    line, existing_index + 1
                )
                existing_order_key = tuple(
                    existing.get("order_key", (2, float("inf"), 0, 0))
                )
                order_key_improved = (
                    incoming_order_key[0] < existing_order_key[0]
                    or "[spine-order:" in str(line or "").lower()
                )
                if existing_generic and not incoming_generic:
                    existing["label"] = request_label
                if order_key_improved:
                    existing["order_key"] = incoming_order_key
                if (existing_generic and not incoming_generic) or order_key_improved:
                    self._sort_active_request_segments()
                return existing
        conversation_request_number = self._allocate_active_request_number()
        if incoming_generic:
            request_label = f"Request {conversation_request_number}"
        segment = {
            "label": request_label,
            "request_number": conversation_request_number,
            "thread": thread_key,
            "content": "",
            "thinking": "",
            "phase": "processing",
            "thinking_tokens": 0,
            "text_tokens": 0,
            # Dispatch/progress records create the card before model output is
            # available.  Content, reasoning, or a structured response payload
            # promotes it to a real response card.
            "status_only": True,
            "complete": False,
            "created_at": self._direct_response_timestamp(),
            "source_is_attachment": bool(self._run_source_is_attachment),
            "order_key": self._request_order_from_log(
                line, segment_number
            ),
        }
        self._active_request_segments.append(segment)
        self._sort_active_request_segments()
        segment_index = self._active_request_segments.index(segment)
        if thread_key:
            self._request_segment_by_thread[thread_key] = segment_index
            self._stream_phase_by_thread[thread_key] = "processing"
        return segment

    def _request_segment_for_thread(self, source_thread=None, create=True):
        thread_key = str(source_thread or "")
        segment_index = self._request_segment_by_thread.get(thread_key)
        if segment_index is not None and 0 <= segment_index < len(self._active_request_segments):
            return self._active_request_segments[segment_index]
        if self._active_request_segments and not thread_key:
            return self._active_request_segments[-1]
        if not create:
            return None
        return self._begin_request_segment("", source_thread)

    def _request_segment_message(self, segment, output_folder=""):
        thinking_tokens = int(segment.get("thinking_tokens", 0) or 0)
        text_tokens = int(segment.get("text_tokens", 0) or 0)
        phase = str(segment.get("phase", "processing") or "processing")
        if segment.get("complete"):
            processing_label = (
                "Token summary  ·  "
                f"Thinking {thinking_tokens:,}  ·  Text {text_tokens:,}"
            )
        elif phase == "thinking":
            processing_label = f"Thinking ({thinking_tokens:,} tokens)"
        elif phase == "text":
            processing_label = f"Generating Text ({text_tokens:,} tokens)"
        elif phase == "queue":
            processing_label = str(
                segment.get("status_label", "")
                or "NVIDIA queue / prefill · Waiting for first token"
            )
        else:
            processing_label = "Processing"
        response_label = str(segment.get("label", "") or "").strip()
        try:
            request_number = int(segment.get("request_number", 0) or 0)
        except (TypeError, ValueError):
            request_number = 0
        if (
            segment.get("source_is_attachment") is False
            and request_number > 0
        ):
            # Defense in depth for every live and committed rendering path.
            # Only genuine attachment requests may display chapter, filename,
            # or chunk metadata.
            response_label = f"Request {request_number}"
        label_is_generic = response_label.lower().startswith("request ")
        label_has_request = " · request " in response_label.lower()
        if response_label and not label_is_generic and not label_has_request:
            if request_number > 0:
                response_label = (
                    f"{response_label} · Request {request_number}"
                )
        elif not response_label and request_number > 0:
            response_label = f"Request {request_number}"
        created_at = str(segment.get("created_at", "") or "").strip()
        storage = {"created_at": created_at} if created_at else {}
        image_path = str(segment.get('image_path', '') or '').strip()
        if not image_path:
            image_paths = self._generated_image_paths_from_text(
                segment.get('content', '')
            )
            image_path = image_paths[0] if image_paths else ''
        if image_path and os.path.isfile(image_path):
            storage['image_path'] = self._history_file_reference(image_path)
        media_path = str(segment.get('media_path', '') or '').strip()
        media_kind = str(segment.get('media_kind', '') or '').strip().lower()
        if not media_path:
            media_references = self._generated_media_references_from_text(
                segment.get('content', '')
            )
            if media_references:
                media_kind, media_path = media_references[0]
        media_kind = self._media_kind_for_path(media_path, media_kind)
        if media_path and media_kind and os.path.isfile(media_path):
            storage['media_path'] = self._history_file_reference(media_path)
            storage['media_kind'] = media_kind
        return (
            "assistant",
            str(segment.get("content", "") or ""),
            str(segment.get("thinking", "") or ""),
            processing_label,
            str(output_folder or ""),
            response_label,
            storage,
        )

    def _on_log_line(self, message, source_thread=None):
        """Queue a request-bound, channel-bound stream event for the UI thread."""
        try:
            import threading

            value = str(message)
            callback_thread = str(
                source_thread or threading.current_thread().name or ""
            )
            thread_key = self._thread_key_from_log(value, callback_thread)

            # Tell append_log not to duplicate high-volume token payloads in
            # the main-window QTextEdit. Direct Text still receives and renders
            # them here; status/configuration records remain in the main log.
            phase = self._listener_stream_phase_by_thread.get(thread_key, "")
            low = value.strip().lower()
            is_glossary_stream_start = value.strip().startswith(
                self._DIRECT_GLOSSARY_STREAM_START_PREFIX
            )
            is_direct_response_payload = value.strip().startswith(
                self._DIRECT_RESPONSE_PAYLOAD_PREFIX
            )
            if is_glossary_stream_start:
                # Glossary providers do not consistently emit a separate
                # ``Text streaming`` banner. Treat this explicit request marker
                # as the boundary so raw CSV/JSON chunks are not also appended
                # to the main GUI log one chunk at a time.
                self._listener_glossary_threads.add(thread_key)
                phase = "text"
            is_glossary_thread = thread_key in self._listener_glossary_threads
            if any(phrase in low for phrase in self._STREAM_END_LOG_PHRASES):
                phase = "processing"
                self._listener_glossary_threads.discard(thread_key)
            elif "thinking complete" in low:
                # A glossary response may move directly from thinking to answer
                # chunks without a provider text-start banner.
                phase = "text" if is_glossary_thread else "processing"
            elif (
                (value.strip().startswith("🧠") and "thinking" in low)
                or " thinking..." in low
                or low.endswith("thinking...")
            ):
                phase = "thinking"
            elif (
                "text streaming" in low
                or "first text token" in low
                or ("first token" in low and "streaming" in low)
            ):
                phase = "text"
            elif (
                "sending api call now" in low
                or "api call in progress" in low
                or "preparing ollama request" in low
            ):
                phase = "processing"
            self._listener_stream_phase_by_thread[thread_key] = phase
            self._log_queue.append(
                {
                    "message": value,
                    "thread": thread_key,
                    "channel": phase,
                }
            )
            if is_glossary_stream_start or is_direct_response_payload:
                return "suppress-main-log"
            if phase in ("thinking", "text"):
                stripped = value.strip()
                is_control = (
                    not stripped
                    or self._looks_like_pipeline_status(stripped)
                    or any(
                        phrase in low
                        for phrase in self._HIDDEN_STREAM_START_LOG_PHRASES
                    )
                    or "thinking" in low
                    or "streaming" in low
                    or "stream complete" in low
                )
                if not is_control:
                    return "suppress-main-log"
        except Exception:
            pass
        return None

    @classmethod
    def _looks_like_pipeline_status(cls, line):
        """Identify transport/status records without swallowing translated Markdown."""
        value = str(line or "").strip()
        low = value.lower()
        if not value:
            return False
        if any(phrase in low for phrase in cls._PIPELINE_LOG_PHRASES):
            return True
        if value.startswith(("Traceback", "File \"", "[DEBUG]", "[INFO]", "[WARN", "[ERROR")):
            return True
        if value in ("TRANSLATION_COMPLETE_SIGNAL", "GLOSSARY_COMPLETE_SIGNAL"):
            return True
        status_phrases = (
            "api call in progress", "preparing ollama request", "http request:", "sdk call finished",
            "thinking tokens used:", "received chapter ", "received section ",
            "received translation from api", "received merged response",
            "finish_reason:", "saved ", "temperature:", "max tokens:",
            "output token limit:", "translation time:", "chapters completed:",
            "processing as text file", "using async chapter extraction",
            "text streaming", "first text token", "thinking complete",
            "first token in", "stream finished", "stream complete",
            "skipping image title translation", "translation preview:",
            "output directory:", "payload directory:",
            "payloads directory:", "skipping post-translation scanning",
        )
        return any(phrase in low for phrase in status_phrases)

    def _classify_line(self, line, source_thread=None, channel_hint=None):
        raw = str(line).rstrip("\n")
        stripped = raw.strip()
        low = stripped.lower()
        thread_key = str(source_thread or "")
        phase = self._stream_phase_by_thread.get(thread_key, "processing")
        channel_hint = str(channel_hint or "").strip().lower()
        if channel_hint in ("thinking", "text", "processing"):
            phase = channel_hint
            self._stream_phase_by_thread[thread_key] = phase

        if stripped.startswith(self._DIRECT_RESPONSE_PAYLOAD_PREFIX):
            self._apply_direct_response_payload(stripped, source_thread)
            return "payload"

        if stripped.startswith(self._DIRECT_GLOSSARY_STREAM_START_PREFIX):
            self._apply_direct_glossary_stream_start(stripped, source_thread)
            return "request_start"

        # Provider transport banners are useful in the main application log,
        # but they are neither model reasoning nor generated response text.
        # Do not put them inside the chat request card.
        if any(
            phrase in low for phrase in self._HIDDEN_STREAM_START_LOG_PHRASES
        ):
            return "ignore"

        # This is the real start of one provider request.  Register it before
        # status/thinking lines arrive so retries and repeated progress notices
        # remain inside the same request card.
        if "[spine-order:" in low and "direct text dispatch" in low:
            self._begin_request_segment(raw, source_thread)
            return "ignore"

        if "sending api call now" in low or "preparing ollama request" in low:
            self._begin_request_segment(raw, source_thread)
            self._in_thinking = False
            self._streaming_text = False
            return "log"

        if "api call in progress" in low:
            self._begin_request_segment(raw, source_thread)
            self._in_thinking = False
            self._streaming_text = False
            return "log"

        if "authnd: nvidia queue / prefill" in low:
            segment = self._request_segment_for_thread(source_thread)
            segment["phase"] = "queue"
            segment["complete"] = False
            if "response headers received" in low:
                import re

                elapsed_match = re.search(
                    r"response headers received in\s+([0-9.]+s)",
                    stripped,
                    flags=re.IGNORECASE,
                )
                elapsed = elapsed_match.group(1) if elapsed_match else ""
                segment["status_label"] = (
                    "NVIDIA queue / prefill · Headers in "
                    f"{elapsed} · Waiting for first token"
                    if elapsed
                    else "NVIDIA queue / prefill · Waiting for first token"
                )
            else:
                segment["status_label"] = (
                    "NVIDIA queue / prefill · Waiting for response"
                )
            return "log"

        # A provider completion record is the hard boundary between streamed
        # model text and downstream pipeline logging. Reset the per-thread
        # phase before classifying anything that follows it.
        if any(phrase in low for phrase in self._STREAM_END_LOG_PHRASES):
            self._in_thinking = False
            self._streaming_text = False
            phase = "processing"
            self._stream_phase_by_thread[thread_key] = phase
            segment = self._request_segment_for_thread(
                source_thread, create=False
            )
            if segment is not None:
                segment["phase"] = phase
                segment["complete"] = True
            return "log"

        if any(phrase in low for phrase in self._PIPELINE_LOG_PHRASES):
            if any(
                phrase in low
                for phrase in (
                    'translation stopped', 'force stop requested',
                    'graceful stop', 'stream finished', 'stream complete',
                    'text file translation complete',
                    'translation completed successfully',
                )
            ):
                phase = "processing"
                self._stream_phase_by_thread[thread_key] = phase
                segment = self._request_segment_for_thread(source_thread, create=False)
                if segment is not None and (
                    "stream finished" in low or "stream complete" in low
                ):
                    segment["phase"] = "processing"
            return "log"

        if "thinking complete" in low:
            self._in_thinking = False
            phase = "processing"
            self._stream_phase_by_thread[thread_key] = phase
            segment = self._request_segment_for_thread(source_thread, create=False)
            if segment is not None:
                segment["phase"] = phase
            return "log"
        # Providers may interleave multiple native thought and text blocks in
        # one response. An explicit thought-start record always opens the
        # thinking channel, even if this request previously streamed text.
        if (
            stripped.startswith("🧠")
            or " thinking..." in low
            or low.endswith("thinking...")
        ):
            self._in_thinking = True
            phase = "thinking"
            self._stream_phase_by_thread[thread_key] = phase
            segment = self._request_segment_for_thread(source_thread)
            segment["phase"] = phase
            return "log"
        if phase == "thinking" and (
            not stripped
            or stripped == "\u200b"
            or not self._looks_like_pipeline_status(stripped)
        ):
            return "thinking"

        if (
            "text streaming" in low
            or "first text token" in low
            or ("first token" in low and "streaming" in low)
        ):
            self._in_thinking = False
            self._streaming_text = True
            phase = "text"
            self._stream_phase_by_thread[thread_key] = phase
            segment = self._request_segment_for_thread(source_thread)
            segment["phase"] = phase
            return "log"
        if "stream complete" in low or "translation completed" in low:
            self._streaming_text = False
            phase = "processing"
            self._stream_phase_by_thread[thread_key] = phase
            return "log"

        if not stripped:
            return "content" if phase == "text" else "log"
        if phase == "text" and not self._looks_like_pipeline_status(stripped):
            # Once the provider announces text streaming, Markdown headings,
            # HTML, emoji and other content-looking status characters are all
            # valid model output. Only explicit transport records are removed.
            return "content"
        if self._looks_like_pipeline_status(stripped):
            return "log"
        if stripped.startswith(("Traceback", "File \"", "[DEBUG]", "[INFO]", "[WARN", "[ERROR")):
            return "log"
        if stripped in ("TRANSLATION_COMPLETE_SIGNAL", "GLOSSARY_COMPLETE_SIGNAL"):
            return "log"

        # Some providers omit a text-start banner when a glossary response has
        # no visible reasoning block. In that case the first raw model line
        # arrives while the request still says ``processing``. Limit this
        # fallback to API workers introduced by the explicit glossary marker,
        # so normal pipeline output retains the stricter filtering rules.
        segment = self._request_segment_for_thread(
            source_thread, create=False
        )
        if (
            segment is not None
            and segment.get("glossary_stream")
            and not segment.get("complete")
        ):
            likely_model_text = (
                stripped[0] not in self._STATUS_FIRST_CHARS
                or stripped.startswith(
                    ("#", "<", "*", "_", "`", ">", "-", "+", "[", "{")
                )
            )
            if likely_model_text:
                self._in_thinking = False
                self._streaming_text = True
                self._stream_phase_by_thread[thread_key] = "text"
                segment["phase"] = "text"
                return "content"
        if stripped[0] in self._STATUS_FIRST_CHARS:
            return "log"
        if stripped.startswith("<"):
            # Raw HTML is valid streamed model output only while this exact
            # request is open.  Downstream ``Translation preview`` logs also
            # begin with ``<html>``; treating those as a new stream duplicated
            # every chapter into a generic Request card and stole its label.
            segment = self._request_segment_for_thread(
                source_thread, create=False
            )
            if segment is not None and not segment.get("complete"):
                self._streaming_text = True
                self._stream_phase_by_thread[thread_key] = "text"
                segment["phase"] = "text"
                return "content"
            return "log"
        return "content" if phase == "text" else "log"

    def _split_embedded_pipeline_log(self, line):
        """Separate a status record appended to an unterminated text chunk."""
        raw = str(line)
        marker_positions = [
            raw.find(marker)
            for marker in self._EMBEDDED_PIPELINE_MARKERS
            if raw.find(marker) > 0
        ]
        if not marker_positions:
            return (raw,)
        split_at = min(marker_positions)
        return raw[:split_at], raw[split_at:]

    def _drain_log_queue(self, final=False):
        import time

        drained = 0
        output_changed = False
        processing_changed = False
        generation_batches = {}
        thinking_batches = {}
        deadline = float("inf") if final else time.monotonic() + 0.012
        drain_limit = 100000 if final else 1200
        while self._log_queue and drained < drain_limit and time.monotonic() < deadline:
            try:
                queued = self._log_queue.popleft()
            except IndexError:
                break
            if isinstance(queued, dict):
                message = queued.get("message", "")
                source_thread = str(queued.get("thread", "") or "")
                channel_hint = str(queued.get("channel", "") or "")
            elif isinstance(queued, tuple) and len(queued) == 2:
                message, source_thread = queued
                channel_hint = ""
            else:
                message, source_thread = queued, ""
                channel_hint = ""
            drained += 1
            for raw_line in str(message).split("\n"):
                for line in self._split_embedded_pipeline_log(raw_line):
                    kind = self._classify_line(
                        line, source_thread, channel_hint=channel_hint
                    )
                    segment = self._request_segment_for_thread(
                        source_thread, create=kind in ("content", "thinking")
                    )
                    if kind == "content":
                        chunk = line + "\n"
                        self._streamed_content += chunk
                        if segment is not None:
                            segment["content"] += chunk
                            segment["status_only"] = False
                            segment["phase"] = "text"
                            segment["complete"] = False
                            generation_batches.setdefault(id(segment), [segment, []])[1].append(chunk)
                        output_changed = True
                    elif kind == "thinking":
                        value = line[4:] if line.startswith("    ") else line
                        self._append_thinking(value + "\n", streamed_thinking=True)
                        if segment is not None:
                            segment["thinking"] += value + "\n"
                            segment["status_only"] = False
                            segment["phase"] = "thinking"
                            segment["complete"] = False
                            thinking_batches.setdefault(id(segment), [segment, []])[1].append(
                                value + "\n"
                            )
                        processing_changed = True
                    elif kind == "payload":
                        output_changed = True
                        processing_changed = True
                    elif kind == "request_start":
                        processing_changed = True
                    elif kind == "ignore":
                        continue
                    else:
                        self._append_thinking(line + "\n")
                        if "received " in line.lower() and " response" in line.lower():
                            target = self._request_segment_for_completion_log(
                                line, source_thread
                            )
                            if target is not None:
                                updated_label = self._request_label_from_log(
                                    line,
                                    self._active_request_segments.index(target) + 1,
                                )
                                existing_label = str(
                                    target.get("label", "") or ""
                                )
                                if (
                                    not updated_label.startswith("Request ")
                                    and (
                                        existing_label.startswith("Request ")
                                        or (
                                            "(chunk " in updated_label.lower()
                                            and "(chunk " not in existing_label.lower()
                                        )
                                    )
                                ):
                                    target["label"] = updated_label
                                target["complete"] = True
                                target["phase"] = "processing"
                                self._sort_active_request_segments()
                        processing_changed = True

        for segment, values in generation_batches.values():
            count = self._count_tokens("".join(values))
            segment["text_tokens"] += count
            self._generation_token_count += count
        for segment, values in thinking_batches.values():
            count = self._count_tokens("".join(values))
            segment["thinking_tokens"] += count
            self._thinking_token_count += count
        if drained:
            self._refresh_processing_label()
        if output_changed or processing_changed:
            self._schedule_stream_render()

    def _commit_active_request_phase(self):
        """Persist real request cards without ending the ongoing Direct Text run."""
        previous_message_count = len(self._chat_messages)
        segments_to_commit = [
            segment
            for segment in self._active_request_segments
            if (
                str(segment.get("content", "") or "").strip()
                or str(segment.get("thinking", "") or "").strip()
                or int(segment.get("thinking_tokens", 0) or 0) > 0
                or int(segment.get("text_tokens", 0) or 0) > 0
            )
        ]
        for segment in segments_to_commit:
            segment["complete"] = True
            segment["phase"] = "processing"
            self._chat_messages.append(
                self._request_segment_message(
                    segment, self._last_output_folder
                )
            )
        self._active_request_segments = []
        self._request_segment_by_thread = {}
        self._stream_phase_by_thread = {}
        self._listener_stream_phase_by_thread = {}
        self._listener_glossary_threads = set()
        self._streamed_content = ""
        self._thinking_stream_text = ""
        self._thinking_token_count = 0
        self._generation_token_count = 0
        self._processing_token_count = 0
        self._processing_text = ""
        followed_history_tail = self._update_history_window_after_append(
            previous_message_count
        )
        if segments_to_commit:
            self._save_chat_history()
        if followed_history_tail:
            self._schedule_stream_render(immediate=True)
        else:
            self._render_output(preserve_viewport=True)

    def _commit_assistant_message(self, completion_message=None):
        """Freeze the active streamed response into the visible chat history."""
        if not self._assistant_message_active:
            return
        previous_message_count = len(self._chat_messages)
        if (
            isinstance(completion_message, list)
            and (
                not completion_message
                or isinstance(completion_message[0], (list, tuple))
            )
        ):
            completion_messages = [
                self._assistant_message_with_timestamp(tuple(message))
                for message in completion_message
                if isinstance(message, (list, tuple)) and message
            ]
        elif completion_message:
            completion_messages = [
                self._assistant_message_with_timestamp(completion_message)
            ]
        else:
            completion_messages = []
        if self._active_request_segments:
            segments_to_commit = list(self._active_request_segments)
            if len(segments_to_commit) > 1 or completion_messages:
                # API clients emit lifecycle-only requests for bookkeeping
                # channels such as EPUB metadata.  If that channel produced no
                # model stream at all, it is not a second response and must not
                # become a fabricated "No translated output" card beside the
                # real chapter/header responses.
                segments_to_commit = [
                    segment for segment in segments_to_commit
                    if not (
                        segment.get("status_only", False)
                        and not str(segment.get("content", "") or "").strip()
                        and not str(segment.get("thinking", "") or "").strip()
                        and int(segment.get("thinking_tokens", 0) or 0) == 0
                        and int(segment.get("text_tokens", 0) or 0) == 0
                    )
                ]
            for segment in segments_to_commit:
                segment["complete"] = True
                segment["phase"] = "processing"
                content = str(segment.get("content", "") or "").strip()
                if not content:
                    content = "*No translated output was emitted for this request.*"
                    segment["content"] = content
                self._chat_messages.append(
                    self._request_segment_message(
                        segment, self._last_output_folder
                    )
                )
            self._chat_messages.extend(completion_messages)
            self._assistant_message_active = False
            self._active_request_segments = []
            self._request_segment_by_thread = {}
            self._stream_phase_by_thread = {}
            self._listener_stream_phase_by_thread = {}
            self._listener_glossary_threads = set()
            followed_history_tail = self._update_history_window_after_append(
                previous_message_count
            )
            self._save_chat_history()
            if followed_history_tail:
                self._schedule_stream_render(immediate=True)
            else:
                self._render_output(preserve_viewport=True)
            return
        content = str(self._streamed_content or "").strip()
        if not content:
            content = "*No translated output was produced for this message.*"
        if self._thinking_token_count or self._generation_token_count:
            final_processing_label = (
                "Token summary  ·  "
                f"Thinking {self._thinking_token_count:,}  ·  "
                f"Text {self._generation_token_count:,}"
            )
        else:
            final_processing_label = self._processing_label_text or "Processing"
        fallback_request_number = self._allocate_active_request_number()
        self._chat_messages.append(
            self._assistant_message_with_timestamp(
                (
                    "assistant",
                    content,
                    self._thinking_stream_text,
                    final_processing_label,
                    self._last_output_folder,
                    f"Request {fallback_request_number}",
                ),
                self._active_response_timestamp,
            )
        )
        self._chat_messages.extend(completion_messages)
        self._assistant_message_active = False
        followed_history_tail = self._update_history_window_after_append(
            previous_message_count
        )
        self._save_chat_history()
        self._render_output(preserve_viewport=not followed_history_tail)

    def _append_thinking(self, text, *, streamed_thinking=False):
        value = str(text)
        self._processing_text += value
        if len(self._processing_text) > 20000:
            self._processing_text = self._processing_text[-20000:]
        if streamed_thinking:
            self._thinking_stream_text += value

    def _hydrate_header_toc_response(self):
        """Guarantee that final header/TOC API output owns a response card.

        Streaming providers do not all echo structured JSON chunks through the
        normal text channel.  The header translator nevertheless writes an
        authoritative ``translated_headers.txt`` artifact.  Use that artifact
        as a lossless fallback so this API request is externalized to its own
        ``response.md`` instead of ending as an empty Request card.
        """
        if not self._run_source_is_attachment:
            return
        import json as json_lib
        import re

        headers_path = self._find_direct_run_artifact(
            "translated_headers.txt"
        )
        if not headers_path:
            return
        try:
            with open(headers_path, 'r', encoding='utf-8') as handle:
                artifact_text = handle.read()
        except OSError:
            return

        translations = {}
        current_key = None
        for line in artifact_text.splitlines():
            chapter_match = re.match(r"\s*Chapter\s+([^:]+):\s*$", line)
            if chapter_match:
                current_key = chapter_match.group(1).strip()
                continue
            translated_match = re.match(r"\s*Translated:\s*(.*)$", line)
            if current_key and translated_match:
                translations[current_key] = translated_match.group(1).strip()
                current_key = None
        if not translations:
            return

        content = json_lib.dumps(translations, ensure_ascii=False, indent=2)
        target = next(
            (
                segment
                for segment in reversed(self._active_request_segments)
                if any(
                    token in str(segment.get("label", "") or "").lower()
                    for token in ("header", "toc")
                )
            ),
            None,
        )
        if target is None:
            response_request_number = self._allocate_active_request_number()
            target = {
                "label": "Header / TOC translation",
                "request_number": response_request_number,
                "thread": "",
                "content": "",
                "thinking": "",
                "phase": "processing",
                "thinking_tokens": 0,
                "text_tokens": 0,
                "status_only": False,
                "complete": True,
                "created_at": self._direct_response_timestamp(),
                "order_key": (
                    0,
                    1_000_000,
                    0,
                    len(self._active_request_segments) + 1,
                ),
            }
            self._active_request_segments.append(target)

        if not str(target.get("content", "") or "").strip():
            previous_tokens = int(target.get("text_tokens", 0) or 0)
            current_tokens = self._count_tokens(content)
            target["content"] = content
            target["text_tokens"] = current_tokens
            self._generation_token_count += current_tokens - previous_tokens
        target["label"] = "Header / TOC translation"
        target["phase"] = "processing"
        target["complete"] = True
        self._sort_active_request_segments()

    def _build_attachment_action_card(self, output_folder):
        """Build the final attachment-management card for a completed run."""
        if (
            not self._run_source_is_attachment
            or not output_folder
            or not os.path.isdir(output_folder)
        ):
            return None
        return attachment_action_card(
            attachment_action_data(self._run_source_path, output_folder)
        )

    def _build_attachment_extraction_summary(self, output_folder):
        """Build the final, persisted chat card from real extraction artifacts."""
        if not self._run_source_is_attachment:
            return None
        report = attachment_extraction_report(
            self._find_direct_run_artifact("extraction_report.txt"),
            self._find_direct_run_artifact("metadata.json"),
            self._find_direct_run_artifact("chapters_full.json"),
            self._active_request_segments,
            self._run_started_at,
            self._run_source_path,
        )
        if report is None:
            return None
        return extraction_report_card(report, output_folder)

    @staticmethod
    def _is_expected_cover_chapter(chapter):
        """Identify cover/title-page records that are valid image-only pages."""
        if not isinstance(chapter, dict):
            return False
        if chapter.get('is_cover') is True:
            return True
        import re

        for key in (
            'title', 'filename', 'original_filename', 'original_basename',
            'original_html_file',
        ):
            value = str(chapter.get(key, '') or '').strip().lower()
            if not value:
                continue
            basename = os.path.basename(value.replace('\\', '/'))
            stem = os.path.splitext(basename)[0]
            compact = re.sub(r'[^a-z0-9]+', '', stem)
            normalized = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
            if (
                normalized in {'cover', 'cover_page', 'title_page'}
                or compact in {'cover', 'coverpage', 'titlepage'}
            ):
                return True
        return False

    def _finish_translation(self):
        if not self._active:
            return
        self._drain_log_queue(final=True)

        final_text = ""
        self._discover_generated_output()
        try:
            readable_extensions = {
                '.txt', '.md', '.markdown', '.html', '.htm', '.xhtml', '.xml',
                '.json', '.csv', '.tsv', '.srt', '.ass', '.lrc', '.vtt', '.log',
            }
            if (
                self._expected_output
                and os.path.isfile(self._expected_output)
                and os.path.splitext(self._expected_output)[1].lower()
                in readable_extensions
            ):
                with open(self._expected_output, 'r', encoding='utf-8') as handle:
                    final_text = handle.read()
        except Exception as exc:
            self._append_thinking(f"⚠️ Could not read translated output: {exc}\n")

        output_folder = self._persist_output_folder()
        persisted_output = str(self._persisted_output_path or '')
        if persisted_output and self._media_kind_for_path(persisted_output):
            self._promote_generated_media_reference(persisted_output)
        self._remember_output_folder(
            output_folder,
            update_conversation_root=not self._run_source_is_attachment,
        )
        if final_text.strip():
            # Replace transport fragments with the clean, assembled output file.
            self._streamed_content = final_text
            if len(self._active_request_segments) == 1:
                self._active_request_segments[0]["content"] = final_text
            elif not self._active_request_segments:
                segment = self._begin_request_segment("", "")
                segment["content"] = final_text
                segment["text_tokens"] = self._count_tokens(final_text)
            self._set_status("Ready")
        elif self._streamed_content.strip():
            self._set_status("Ready" if output_folder else "Run ended; showing streamed output")
        elif self._expected_output and os.path.isfile(self._expected_output):
            segment = self._request_segment_for_thread("", create=True)
            artifact_type = (
                os.path.splitext(self._expected_output)[1].lstrip('.').upper()
                or "file"
            )
            segment["content"] = (
                f"**Translation completed.** The translated {artifact_type} file is "
                "available from the output-folder link below."
            )
            self._set_status("Ready")
        else:
            self._set_status("No translated output was produced")

        self._hydrate_header_toc_response()
        completion_messages = [
            message
            for message in (
                self._build_attachment_extraction_summary(output_folder),
                self._build_attachment_action_card(output_folder),
            )
            if message is not None
        ]
        self._commit_assistant_message(
            completion_message=completion_messages
        )
        self._restore_run_context()


def attachment_action_data(source_path, output_folder):
    """The "Attachment actions" card as data: source name, compiled EPUB/PDF names, folder."""
    source_name = os.path.basename(source_path or "Attachment")
    compiled = ChatStoreMixin._preferred_attachment_compiled_documents(output_folder)
    compiled_names = [
        str(compiled[extension])
        for extension in (".epub", ".pdf")
        if compiled.get(extension)
    ]
    return {
        "source_name": source_name,
        "compiled_names": compiled_names,
        "output_folder": os.path.abspath(str(output_folder)),
    }


def attachment_action_card(data):
    """The persisted v2 "Attachment actions" message for :func:`attachment_action_data`."""
    source_name = data["source_name"]
    compiled_names = data["compiled_names"]
    lines = [
        "## Attachment ready",
        "",
        f"**Source:** {source_name}",
        "",
        "The attachment output workspace is ready.",
    ]
    if compiled_names:
        lines.extend(
            (
                "",
                "**Compiled output:** " + " · ".join(compiled_names),
            )
        )
    lines.extend(
        (
            "",
            "Use the actions below to migrate the workspace outside Direct "
            "Text or, when available, open its compiled EPUB in the "
            "integrated reader.",
        )
    )
    return (
        "assistant",
        "\n".join(lines),
        "",
        "Completed",
        data["output_folder"],
        "Attachment actions",
    )


def attachment_extraction_report(report_path, metadata_path, chapters_path, segments, run_started_at,
                                 source_path):
    """The "Extraction report" of one attachment run as data (None without extraction artifacts).

    Inputs: the run's ``extraction_report.txt`` / ``metadata.json`` / ``chapters_full.json``
    (``_find_direct_run_artifact``), its request segments, start time and source path.
    """
    import json as json_lib
    import time

    metadata = {}
    chapters = []
    try:
        if metadata_path:
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                loaded = json_lib.load(handle)
            if isinstance(loaded, dict):
                metadata = loaded
    except (OSError, TypeError, ValueError):
        metadata = {}
    try:
        if chapters_path:
            with open(chapters_path, 'r', encoding='utf-8') as handle:
                loaded = json_lib.load(handle)
            if isinstance(loaded, list):
                chapters = loaded
    except (OSError, TypeError, ValueError):
        chapters = []

    # Do not manufacture an extraction report for single images or plain
    # pasted text. EPUB/PDF extraction leaves at least one of these
    # authoritative artifacts behind.
    if not report_path and not chapters and not metadata.get('extraction_mode'):
        return None

    try:
        chapter_count = int(metadata.get('chapter_count', len(chapters)) or 0)
    except (TypeError, ValueError):
        chapter_count = len(chapters)
    if not chapter_count:
        chapter_count = len(chapters)
    payloads_ready = metadata.get('chapter_payloads_ready')
    try:
        payloads_ready = int(payloads_ready)
    except (TypeError, ValueError):
        payloads_ready = sum(
            1 for chapter in chapters
            if isinstance(chapter, dict)
            and isinstance(chapter.get('body'), str)
        )

    text_only = 0
    image_only = 0
    actionable_image_only = 0
    mixed = 0
    empty_minimal = 0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        has_images = bool(chapter.get('has_images'))
        try:
            file_size = int(chapter.get('file_size', 0) or 0)
        except (TypeError, ValueError):
            file_size = 0
        if file_size < 50:
            empty_minimal += 1
        if bool(chapter.get('is_image_only', False)):
            image_only += 1
            if not DirectTextStreamMixin._is_expected_cover_chapter(chapter):
                actionable_image_only += 1
        elif has_images and file_size >= 500:
            mixed += 1
        elif not has_images and file_size >= 500:
            text_only += 1

    extracted_resources = metadata.get('extracted_resources') or {}
    resource_count = 0
    if isinstance(extracted_resources, dict):
        for value in extracted_resources.values():
            if isinstance(value, (list, tuple, set, dict)):
                resource_count += len(value)
            elif isinstance(value, int):
                resource_count += max(0, value)

    issue_lines = []
    if report_path:
        try:
            with open(report_path, 'r', encoding='utf-8') as handle:
                report_text = handle.read()
            issue_section = report_text.split('POTENTIAL ISSUES:', 1)
            if len(issue_section) == 2:
                for line in issue_section[1].splitlines():
                    value = line.strip().lstrip('•').strip()
                    if 'contain only images' in value.lower():
                        if actionable_image_only == 0:
                            continue
                        chapter_word = (
                            'chapter' if actionable_image_only == 1
                            else 'chapters'
                        )
                        verb = (
                            'contains' if actionable_image_only == 1
                            else 'contain'
                        )
                        value = (
                            f"{actionable_image_only} {chapter_word} {verb} "
                            "only images (may need OCR)"
                        )
                    if value and not value.lower().startswith('none detected'):
                        issue_lines.append(value)
        except OSError:
            pass

    request_count = len(segments)
    thinking_tokens = sum(
        int(segment.get('thinking_tokens', 0) or 0)
        for segment in segments
    )
    text_tokens = sum(
        int(segment.get('text_tokens', 0) or 0)
        for segment in segments
    )
    elapsed_seconds = (
        max(0, int(time.time() - run_started_at))
        if run_started_at
        else 0
    )
    source_name = os.path.basename(source_path or 'Attachment')
    extraction_mode = str(
        metadata.get('extraction_mode', 'unknown') or 'unknown'
    ).title()
    language = str(metadata.get('detected_language', 'unknown') or 'unknown')
    ready_label = (
        "Ready"
        if chapter_count > 0 and payloads_ready == chapter_count
        else "Incomplete"
    )
    return {
        "source_name": source_name,
        "ready_label": ready_label,
        "extraction_mode": extraction_mode,
        "language": language,
        "payloads_ready": payloads_ready,
        "chapter_count": chapter_count,
        "text_only": text_only,
        "image_only": image_only,
        "actionable_image_only": actionable_image_only,
        "mixed": mixed,
        "empty_minimal": empty_minimal,
        "resource_count": resource_count,
        "request_count": request_count,
        "thinking_tokens": thinking_tokens,
        "text_tokens": text_tokens,
        "elapsed_seconds": elapsed_seconds,
        "issues": issue_lines,
        "has_report_file": bool(report_path),
    }


def extraction_report_card(report, output_folder):
    """The persisted v2 "Extraction report" message for :func:`attachment_extraction_report`."""
    source_name = report["source_name"]
    ready_label = report["ready_label"]
    extraction_mode = report["extraction_mode"]
    language = report["language"]
    payloads_ready = report["payloads_ready"]
    chapter_count = report["chapter_count"]
    text_only = report["text_only"]
    image_only = report["image_only"]
    mixed = report["mixed"]
    empty_minimal = report["empty_minimal"]
    resource_count = report["resource_count"]
    request_count = report["request_count"]
    thinking_tokens = report["thinking_tokens"]
    text_tokens = report["text_tokens"]
    elapsed_seconds = report["elapsed_seconds"]
    issue_lines = report["issues"]
    lines = [
        "## Extraction report summary",
        "",
        f"**Source:** {source_name}",
        "",
        f"- **Extraction:** {ready_label} · {extraction_mode} mode · "
        f"{language}",
        f"- **Chapter payloads:** {payloads_ready:,}/{chapter_count:,} ready",
        f"- **Content:** {text_only:,} text · {image_only:,} image-only · "
        f"{mixed:,} mixed · {empty_minimal:,} empty/minimal",
        f"- **Resources extracted:** {resource_count:,}",
        f"- **API requests:** {request_count:,}",
        f"- **Tokens:** Thinking {thinking_tokens:,} · Text {text_tokens:,}",
        f"- **Elapsed:** {elapsed_seconds // 60}m {elapsed_seconds % 60}s",
    ]
    if issue_lines:
        lines.extend(("", "**Potential issues:**"))
        lines.extend(f"- {issue}" for issue in issue_lines[:4])
    else:
        lines.extend(("", "**Potential issues:** None detected."))
    if report["has_report_file"]:
        lines.extend(
            (
                "",
                "The complete extraction_report.txt and extracted metadata "
                "are saved in the attachment output folder.",
            )
        )

    return (
        "assistant",
        "\n".join(lines),
        "",
        "Completed",
        str(output_folder or ""),
        "Extraction report",
    )


class DirectTextStream(DirectTextStreamMixin, ChatStoreMixin):
    """GUI-free host of one Direct Text run's stream (mobile RunStream / JobService / finish).

    It holds the run state the dialog keeps (``_log_queue``, request segments, phases,
    token counts, ``_run_*``) and implements the dialog hooks without Qt. Public API:
    ``feed`` / ``on_log_line`` (any thread; returns the ``"suppress-main-log"`` hint),
    ``drain(final=False)`` (classify queued lines; True when the cards changed),
    ``segments(drain=True)`` / ``ordered_segments()`` (copies; ``drain=False`` skips the
    unbudgeted drain, for repaints),
    ``request_segment_message(segment, output_folder)`` (the v2 assistant tuple),
    ``commit_active_request_phase()`` (the glossary gate) and ``processing_label``.

    Standalone (``store=None``) it commits into its own ``chat_messages`` list and never
    writes the history (the mobile chat's live cards; a run's stream is created when the
    send starts, so the run and its assistant message are active). Bound to a
    ``direct_text_store.ChatStore`` (``store=``) it works on the store's current session,
    shares the store's output-folder link and saves through it - one scoped operation
    such as ``ChatStore.finish_run``.
    """

    @property
    def _last_output_folder(self):
        store = self.__dict__.get("store")
        if store is not None:
            return store._last_output_folder
        return self.__dict__.get("_own_last_output_folder", "")

    @_last_output_folder.setter
    def _last_output_folder(self, value):
        store = self.__dict__.get("store")
        if store is not None:
            store._last_output_folder = value
        else:
            self.__dict__["_own_last_output_folder"] = value

    def __init__(self, *, source_is_attachment=False, request_number=None, model=None, store=None,
                 translator=None, chat_messages=None, history_path=None, cleanup_temp_root=False):
        self.lock = threading.RLock()
        self.store = store
        self.cleanup_temp_root = bool(cleanup_temp_root)
        if store is not None:
            self.translator = translator or store.translator
            self._chat_history_path = store._chat_history_path
            self._chat_sessions = store._chat_sessions
            self._current_chat_index = store._current_chat_index
            self._chat_messages = store._chat_messages
            self._saved_env = store._saved_env
            self._message_text_cache = store._message_text_cache
            self._rendered_message_cache = store._rendered_message_cache
            self._expanded_processing_messages = store._expanded_processing_messages
            self._pending_attachment = store._pending_attachment
            self._chat_history_save_timer = store._chat_history_save_timer
        else:
            self.translator = translator or DirectTextOwnerView(model=model)
            self._chat_history_path = (
                os.path.abspath(os.path.expanduser(str(history_path)))
                if history_path else self._resolve_chat_history_path()
            )
            session = self._new_chat_session(0)
            if chat_messages is not None:
                session["messages"] = chat_messages
            self._chat_sessions = [session]
            self._current_chat_index = 0
            self._chat_messages = session["messages"]
            self._saved_env = {key: os.environ.get(key) for key in self._OUTPUT_ENV_KEYS}
            self._message_text_cache = {}
            self._rendered_message_cache = {}
            self._expanded_processing_messages = set()
            self._pending_attachment = None
            self._chat_history_save_timer = _SaveTimer(self)
        if model is not None and translator is None and store is not None:
            self.translator = DirectTextOwnerView(getattr(store.translator, "config", None), model)
        # Run state (dialog __init__ / _start_translation).
        self._log_queue = deque()
        self._active = True
        self._in_thinking = False
        self._thinking_token_count = 0
        self._generation_token_count = 0
        self._processing_token_count = 0
        self._token_encoder = None
        self._token_encoder_initialized = False
        self._streaming_text = False
        self._streamed_content = ""
        self._temp_root = ""
        self._temp_input = ""
        self._expected_output = ""
        self._persisted_output_path = ""
        # the send starts from the chat's folder (``_start_translation``)
        current_session = self._current_chat_session()
        self._last_output_folder = str(
            current_session.get("output_folder", "") if current_session else ""
        )
        self._preserve_temp_root = False
        self._assistant_message_active = True
        self._active_response_timestamp = self._direct_response_timestamp()
        self._active_request_segments = []
        self._request_segment_by_thread = {}
        self._stream_phase_by_thread = {}
        self._listener_stream_phase_by_thread = {}
        self._listener_glossary_threads = set()
        self._run_output_mode = "text"
        self._run_source_path = ""
        self._run_source_extension = ".txt"
        self._run_source_is_attachment = bool(source_is_attachment)
        self._run_manual_glossary_path = ""
        self._run_started_at = 0.0
        self._processing_text = ""
        self._thinking_stream_text = ""
        self._processing_label_text = "Processing"
        self._processing_spinner_active = False
        self.force_no_glossary_radio = _CheckFlag(False)
        self._active_request_next_number = (
            int(request_number) if request_number else self._next_conversation_request_number()
        )
        self._history_dirty = False
        self.render_requests = 0
        self.status_text = "Ready"
        self.saves = 0
        self.on_schedule_save = None

    # ---- dialog hooks (GUI-free) --------------------------------------------------------

    def _schedule_stream_render(self, immediate=False):
        self.render_requests += 1

    def _render_output(self, active_only=False, preserve_viewport=False):
        self.render_requests += 1

    def _update_history_window_after_append(self, previous_count):
        return True

    def _set_status(self, text, output_folder=""):
        self.status_text = str(text)

    def _restore_run_context(self):
        """End of run: the GUI-free part of the dialog's restore (state + temp root)."""
        temp_root = self._temp_root
        self._active = False
        self._in_thinking = False
        self._refresh_processing_label()
        self._set_thinking_spinner_active(False)
        if self.cleanup_temp_root and temp_root and not self._preserve_temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _refresh_chat_list(self):
        if self.store is not None:
            self.store._refresh_chat_list()

    def _save_chat_history(self):
        self.saves += 1
        if self.store is not None:
            self.store._save_chat_history()
            # A save externalises bodies into a new messages list (the dialog rebinds too).
            current = self._current_chat_session()
            if current is not None:
                self._chat_messages = current["messages"]

    # ---- run set-up -------------------------------------------------------------------------

    def load_run(self, run):
        """Adopt a run's state (``_start_translation`` / mobile ``DirectTextRun.as_dict()``)."""
        run = dict(run or {})
        self._temp_root = str(run.get("temp_root") or "")
        self._run_source_path = str(run.get("source_path") or run.get("temp_input") or "")
        self._temp_input = self._run_source_path
        self._run_source_extension = str(
            run.get("source_extension") or os.path.splitext(self._run_source_path)[1].lower() or ".txt"
        )
        self._run_source_is_attachment = bool(run.get("is_attachment"))
        self._expected_output = str(run.get("expected_output") or "")
        self._run_manual_glossary_path = str(run.get("manual_glossary_path") or "")
        try:
            self._run_started_at = float(run.get("started_at") or 0.0)
        except (TypeError, ValueError):
            self._run_started_at = 0.0
        self._run_output_mode = str(run.get("output_mode") or "text")
        if run.get("created_at"):
            self._active_response_timestamp = str(run["created_at"])
        # A run-local translator view: the glossary attributes the pipeline set on the owner.
        self.translator = DirectTextOwnerView(getattr(self.translator, "config", None),
                                              run.get("model") or getattr(self.translator, "model_var", None))
        if "force_no_glossary" in run:
            force_none = bool(run.get("force_no_glossary"))
            self.force_no_glossary_radio = _CheckFlag(force_none)
            self.translator._direct_text_force_no_glossary = force_none
        if self._run_manual_glossary_path:
            self.translator._direct_text_manual_glossary_path = self._run_manual_glossary_path
        if run.get("glossary_path"):
            self.translator.manual_glossary_path = str(run["glossary_path"])
        # The run's own environment values (recorded by the job before its process state was
        # restored); never the live os.environ, which may belong to the next queued job.
        self._RUN_ENVIRONMENT = {str(key): str(value or "") for key, value in dict(run.get("run_env") or {}).items()}
        session = self._current_chat_session()
        self._last_output_folder = str(session.get("output_folder", "") if session else "")
        return self

    def load_segments(self, segments, streamed_content=None):
        """Adopt request segments produced elsewhere (e.g. the job's own stream)."""
        self._active_request_segments = [dict(segment) for segment in segments or ()]
        self._sort_active_request_segments()
        self._streamed_content = (
            str(streamed_content) if streamed_content is not None
            else "".join(str(segment.get("content", "") or "") for segment in self._active_request_segments)
        )
        self._thinking_stream_text = "".join(
            str(segment.get("thinking", "") or "") for segment in self._active_request_segments
        )
        self._thinking_token_count = sum(
            int(segment.get("thinking_tokens", 0) or 0) for segment in self._active_request_segments
        )
        self._generation_token_count = sum(
            int(segment.get("text_tokens", 0) or 0) for segment in self._active_request_segments
        )
        numbers = [int(segment.get("request_number", 0) or 0) for segment in self._active_request_segments]
        self._active_request_next_number = max(
            [self._next_conversation_request_number()] + [number + 1 for number in numbers]
        )

    def adopt_stream_state(self, other):
        """Continue from another ``DirectTextStream`` of the same run (queue drained first)."""
        with other.lock:
            other._drain_log_queue(final=True)
            for name in (
                "_streamed_content", "_thinking_token_count", "_generation_token_count",
                "_processing_token_count", "_processing_text", "_thinking_stream_text",
                "_processing_label_text", "_in_thinking", "_streaming_text", "_active_request_next_number",
                "_active_response_timestamp", "_token_encoder", "_token_encoder_initialized",
            ):
                setattr(self, name, getattr(other, name))
            # copies: finishing edits the cards; the live stream keeps its own
            self._active_request_segments = [dict(segment) for segment in other._active_request_segments]
            self._stream_phase_by_thread = dict(other._stream_phase_by_thread)
            self._listener_stream_phase_by_thread = dict(other._listener_stream_phase_by_thread)
            self._listener_glossary_threads = set(other._listener_glossary_threads)
            self._sort_active_request_segments()
            self._active_request_next_number = max(
                int(self._active_request_next_number or 0), self._next_conversation_request_number()
            )

    # ---- public API ---------------------------------------------------------------------------

    def feed(self, message, source_thread=None):
        """Queue one raw log record (any thread); returns ``"suppress-main-log"`` for token payloads.

        Like the dialog's log listener it only touches the listener-side phase maps and
        the thread-safe queue, so the logging thread never waits for a drain.
        """
        return self._on_log_line(message, source_thread)

    on_log_line = feed

    def _card_signature(self):
        return (self._processing_label_text,) + tuple(
            (id(segment), segment.get("label"), segment.get("phase"), len(str(segment.get("content", "") or "")),
             len(str(segment.get("thinking", "") or "")), segment.get("complete"), segment.get("status_only"),
             segment.get("text_tokens"), segment.get("thinking_tokens"), segment.get("status_label"))
            for segment in self._active_request_segments
        )

    def drain(self, final=False):
        """Classify the queued records (``final`` = no time budget); True when the cards changed."""
        with self.lock:
            before = (self.render_requests, self._card_signature())
            self._drain_log_queue(final=final)
            return (self.render_requests, self._card_signature()) != before

    drain_log_queue = drain

    @property
    def active_request_segments(self):
        return self._active_request_segments

    def segments(self, drain=True):
        """Copies of the live request segments (spine order).

        ``drain=True`` classifies every queued record first (no time budget: finishing,
        tests); a repaint passes ``drain=False`` and reads what the budgeted ``drain()``
        ticks classified, so a long backlog never runs on the UI loop.
        """
        with self.lock:
            if drain:
                self._drain_log_queue(final=True)
            return [dict(segment) for segment in self._active_request_segments]

    ordered_segments = segments

    def request_segment_message(self, segment, output_folder=""):
        return self._request_segment_message(segment, output_folder)

    segment_message = request_segment_message

    def messages(self, output_folder=""):
        """v2 assistant tuples for the live segments (what the dialog renders while streaming)."""
        return [self._request_segment_message(segment, output_folder) for segment in self.segments()]

    def commit_active_request_phase(self):
        """Freeze the real request cards (glossary gate) into ``chat_messages``."""
        with self.lock:
            self._drain_log_queue(final=True)
            self._commit_active_request_phase()

    @property
    def chat_messages(self):
        return self._chat_messages

    @property
    def processing_label(self):
        return self._processing_label_text

    @property
    def streamed_content(self):
        return self._streamed_content

    def count_tokens(self, text):
        return self._count_tokens(text)


#: JobService looks the request model up by this name.
RequestStream = DirectTextStream


def make_stream(*, source_is_attachment=False, request_number=None, model=None, **kwargs):
    """A ``DirectTextStream`` for one run (the mobile chat's factory entry point)."""
    return DirectTextStream(source_is_attachment=source_is_attachment, request_number=request_number,
                            model=model, **kwargs)


_DEFAULT_STREAM = None
_TOKEN_COUNTERS = {}
_HELPER_LOCK = threading.Lock()


def request_segment_message(segment, output_folder=""):
    """The v2 assistant tuple for a request segment (``_request_segment_message``)."""
    global _DEFAULT_STREAM
    with _HELPER_LOCK:
        if _DEFAULT_STREAM is None:
            _DEFAULT_STREAM = DirectTextStream()
        stream = _DEFAULT_STREAM
    return stream._request_segment_message(segment, output_folder)


def count_tokens(text, model_name=""):
    """Tokens in *text* for *model_name* (``_count_tokens``: tiktoken model -> o200k -> cl100k; 0 without tiktoken)."""
    key = str(model_name or "")
    with _HELPER_LOCK:
        counter = _TOKEN_COUNTERS.get(key)
        if counter is None:
            counter = DirectTextStream(model=key)
            _TOKEN_COUNTERS[key] = counter
    return counter._count_tokens(text)
