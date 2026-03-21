# reader_translator.py
"""
Lightweight single-chapter translation engine for the in-reader translator.
Accepts raw HTML, extracts text (html2text or BS4), calls UnifiedClient,
and captures real-time streaming output via a logging handler on the
unified_api_client logger (which uses _gui_print → logger.log, not stdout).
"""

import os
import sys
import threading
import logging
import re
import time

logger = logging.getLogger(__name__)


def _get_system_prompt(config_data):
    """Build a system prompt from the user's config settings."""
    try:
        from default_prompts import get_prompt
    except ImportError:
        return "Translate the following text to English. Preserve formatting."

    profile = config_data.get('prompt_profile', 'Korean_html2text')
    prompt = config_data.get('system_prompt', '')

    if not prompt:
        prompt = get_prompt(profile)
    if not prompt:
        prompt = get_prompt('Korean_html2text')
    return prompt or "Translate the following text to English. Preserve formatting."


def _html_to_text(raw_html, use_html2text=True):
    """Extract text from a single HTML chapter file.

    Args:
        raw_html: Raw HTML string from the EPUB chapter.
        use_html2text: If True, use html2text for markdown-like output.
                       If False, use BeautifulSoup for plain text.
    Returns:
        Extracted text string.
    """
    if use_html2text:
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0  # No line wrapping
            h.unicode_snob = True
            return h.handle(raw_html).strip()
        except ImportError:
            pass  # Fall through to BS4

    # BeautifulSoup fallback
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html, 'html.parser')
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        body = soup.find('body') or soup
        return body.get_text(separator='\n\n').strip()
    except Exception:
        # Last resort: crude tag stripping
        text = re.sub(r'<[^>]+>', '', raw_html)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def _setup_thinking_env(config_data):
    """Set thinking environment variables based on config.

    Default: Gemini = minimal thinking (budget 1024), GPT = none.
    When enabled: Gemini = larger budget, Claude = thinking enabled.
    """
    enable_thinking = config_data.get('reader_enable_thinking', False)
    model = config_data.get('model', '').lower()

    is_gemini = 'gemini' in model
    is_gpt = 'gpt' in model or 'openai' in model or 'authgpt' in model
    is_claude = 'claude' in model

    if is_gemini:
        # Gemini always gets thinking; control the LEVEL
        os.environ['ENABLE_THINKING'] = '1'
        if enable_thinking:
            os.environ['GEMINI_THINKING_LEVEL'] = 'high'
            os.environ['GEMINI_THINKING_BUDGET'] = '8192'
        else:
            os.environ['GEMINI_THINKING_LEVEL'] = 'minimal'
            os.environ['GEMINI_THINKING_BUDGET'] = '1024'
        os.environ['EXTENDED_THINKING'] = '0'
    elif is_gpt:
        # GPT: no thinking unless explicitly enabled
        os.environ['ENABLE_THINKING'] = '1' if enable_thinking else '0'
        os.environ['EXTENDED_THINKING'] = '0'
        os.environ.pop('GEMINI_THINKING_BUDGET', None)
    elif is_claude:
        # Claude: thinking off by default, enabled with budget when toggled
        os.environ['ENABLE_THINKING'] = '1' if enable_thinking else '0'
        if enable_thinking:
            os.environ['THINKING_BUDGET'] = '4096'
        else:
            os.environ.pop('THINKING_BUDGET', None)
        os.environ['EXTENDED_THINKING'] = '0'
    else:
        # Unknown model: follow the toggle
        os.environ['ENABLE_THINKING'] = '1' if enable_thinking else '0'
        os.environ['EXTENDED_THINKING'] = '0'


class _StreamLogHandler(logging.Handler):
    """Logging handler that intercepts unified_api_client log messages.

    unified_api_client shadows print() with _gui_print() which routes ALL
    output through logger.log().  This means sys.stdout never sees streaming
    tokens.  We attach this handler to the unified_api_client logger to
    capture the actual streamed content and forward it to the on_chunk callback.
    """

    # Prefixes that indicate log/status messages — NOT translation content
    _SKIP_PREFIXES = (
        '[', '🔄', '⚡', '⏳', '📊', '✅', '❌', '⚠️', '🚨', '🛰️', '🧠',
        '───', '---', '===', 'Model:', 'Provider:', 'Streaming',
        'Temperature', 'Max tokens', 'API call', 'HTTP', 'Status',
        'Time:', 'Tokens:', 'Rate limit', 'Retry', 'Error:',
        'response_name', 'Request', 'Using ', 'Sending ', '   ',
        'Content-Type', 'Authorization', 'Bearer', 'x-goog',
        'Total tokens', 'Input tokens', 'Output tokens',
        'finish_reason', 'Prompt tokens', 'Completion tokens',
    )

    def __init__(self, on_chunk, stop_event=None):
        super().__init__()
        self._on_chunk = on_chunk
        self._stop_event = stop_event

    def emit(self, record):
        if self._stop_event and self._stop_event.is_set():
            return
        try:
            msg = self.format(record)
            if not msg or not msg.strip():
                return
            line = msg.strip()
            # Skip log/status lines — only forward actual content
            if any(line.startswith(p) for p in self._SKIP_PREFIXES):
                return
            # Skip very short fragments that are likely log noise
            if len(line) < 2:
                return
            self._on_chunk(msg)
        except Exception:
            pass


def translate_chapter_streaming(
    raw_html,
    config_data,
    on_chunk=None,
    on_thinking=None,
    on_complete=None,
    on_error=None,
    stop_event=None,
):
    """Translate a single HTML chapter using streaming, with callbacks.

    Uses a logging handler on the unified_api_client logger to capture
    streamed tokens (unified_api_client uses _gui_print → logger.log,
    not sys.stdout).
    """
    if stop_event is None:
        stop_event = threading.Event()
    stream_handler = None

    try:
        # Pre-register Android stubs
        import importlib
        _stub_map = {
            'tiktoken': 'tiktoken_stub',
            'ebooklib': 'ebooklib_stub',
            'ebooklib.epub': 'ebooklib_stub',
            'httpx': 'httpx_stub',
            'rapidfuzz': 'rapidfuzz_stub',
            'rapidfuzz.fuzz': 'rapidfuzz_stub',
            'rapidfuzz.process': 'rapidfuzz_stub',
            'langdetect': 'langdetect_stub',
        }
        for mod_name, stub_name in _stub_map.items():
            if mod_name not in sys.modules:
                try:
                    importlib.import_module(mod_name)
                except ImportError:
                    try:
                        stub = importlib.import_module(stub_name)
                        sys.modules[mod_name] = stub
                    except Exception:
                        pass

        # Import the API client
        from unified_api_client import UnifiedClient
        import unified_api_client
        if hasattr(unified_api_client, 'set_stop_flag'):
            unified_api_client.set_stop_flag(False)
        if hasattr(unified_api_client, 'UnifiedClient'):
            unified_api_client.UnifiedClient._global_cancelled = False

        # Build config
        api_key = config_data.get('api_key', '')
        model = config_data.get('model', 'gemini-2.0-flash')
        temperature = config_data.get('temperature', 0.3)
        max_output_tokens = config_data.get('max_output_tokens', 8192)
        system_prompt = _get_system_prompt(config_data)
        use_html2text = config_data.get('extraction_mode', 'html2text') == 'html2text'

        # Check API key (skip for authgpt/antigravity)
        model_lower = model.lower()
        needs_key = not (model_lower.startswith('authgpt/') or
                         model_lower.startswith('antigravity/') or
                         model_lower == 'google-translate-free')
        if needs_key and not api_key:
            if on_error:
                on_error("No API key configured. Set one in Translation Settings.")
            return

        if stop_event.is_set():
            return

        # Extract text from the single HTML chapter
        chapter_text = _html_to_text(raw_html, use_html2text=use_html2text)
        if not chapter_text.strip():
            if on_error:
                on_error("Chapter has no extractable text.")
            return

        # Set up environment
        os.environ['API_KEY'] = api_key
        os.environ['MODEL'] = model
        os.environ['ENABLE_STREAMING'] = '1'
        os.environ['LOG_STREAM_CHUNKS'] = '1'

        # Wire glossary CSV if configured
        glossary_path = config_data.get('reader_glossary_path', '')
        if glossary_path and os.path.isfile(glossary_path):
            os.environ['MANUAL_GLOSSARY'] = glossary_path
        else:
            os.environ.pop('MANUAL_GLOSSARY', None)

        # Set thinking env vars
        _setup_thinking_env(config_data)

        if stop_event.is_set():
            return

        # Install logging handler on the unified_api_client logger
        # to capture streamed tokens (they go through logger, not stdout)
        if on_chunk:
            uac_logger = logging.getLogger('unified_api_client')
            stream_handler = _StreamLogHandler(on_chunk, stop_event)
            stream_handler.setLevel(logging.INFO)
            uac_logger.addHandler(stream_handler)

        # Create client
        client = UnifiedClient(api_key=api_key, model=model)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chapter_text},
        ]

        # Call the API
        response = client._get_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            max_completion_tokens=None,
            response_name="reader_translate",
        )

        if stop_event.is_set():
            return

        if response and response.content:
            translated = response.content.strip()

            # Clean up AI artifacts
            try:
                from TransateKRtoEN import PostProcessor
                translated = PostProcessor.strip_split_markers(translated)
                translated = PostProcessor.clean_ai_artifacts(translated, remove_artifacts=True)
            except Exception:
                pass

            if on_complete:
                on_complete(translated)
        else:
            error_msg = "Empty response from API"
            if response and response.error_details:
                error_msg = f"API error: {response.error_details}"
            if on_error:
                on_error(error_msg)

    except Exception as e:
        import traceback
        err = f"Translation error: {e}\n{traceback.format_exc()}"
        logger.error(err)
        if on_error:
            on_error(str(e))
    finally:
        # Remove the streaming log handler
        if stream_handler:
            try:
                uac_logger = logging.getLogger('unified_api_client')
                uac_logger.removeHandler(stream_handler)
            except Exception:
                pass
        # Reset stop flags
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(False)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
        except Exception:
            pass
