# reader_translator.py
"""
Lightweight single-chapter translation engine for the in-reader translator.
Accepts raw HTML, extracts text (html2text or BS4), calls UnifiedClient,
and captures real-time streaming output via stdout redirection.
"""

import os
import sys
import io
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

    Default: thinking OFF.
    When enabled: Gemini gets minimal budget, GPT gets none.
    """
    enable_thinking = config_data.get('reader_enable_thinking', False)

    if not enable_thinking:
        os.environ['ENABLE_THINKING'] = '0'
        os.environ['EXTENDED_THINKING'] = '0'
        return

    # Thinking enabled — set minimal budgets
    model = config_data.get('model', '').lower()
    os.environ['ENABLE_THINKING'] = '1'

    if 'gemini' in model:
        # Gemini: minimal thinking budget
        os.environ['GEMINI_THINKING_BUDGET'] = '1024'
    elif 'gpt' in model or 'openai' in model or 'authgpt' in model:
        # GPT: no extended thinking
        os.environ['EXTENDED_THINKING'] = '0'
    elif 'claude' in model:
        os.environ['THINKING_BUDGET'] = '1024'


class _StreamCapture:
    """Captures stdout writes during API call to extract streaming tokens.

    The UnifiedClient prints streaming tokens to stdout.
    This wrapper intercepts them and forwards to on_chunk callbacks.
    """
    def __init__(self, original_stdout, on_chunk=None, stop_event=None):
        self._original = original_stdout
        self._on_chunk = on_chunk
        self._stop_event = stop_event
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text):
        # Always write to original stdout for logging
        try:
            self._original.write(text)
        except Exception:
            pass

        # Forward non-empty content to the chunk callback
        if text and self._on_chunk and text.strip():
            # Filter out log-style lines (timestamps, status messages)
            # Only forward actual translation content
            line = text.strip()
            # Skip common log prefixes
            if any(line.startswith(p) for p in [
                '[', '🔄', '⚡', '⏳', '📊', '✅', '❌', '⚠️', '🚨',
                '───', '---', '===', 'Model:', 'Provider:', 'Streaming',
                'Temperature', 'Max tokens', 'API call', 'HTTP', 'Status',
                'Time:', 'Tokens:', 'Rate limit', 'Retry', 'Error:',
                'response_name', 'Request', 'Using ', 'Sending ',
            ]):
                return
            # Forward the text
            self._on_chunk(text)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False

    # Forward any other attribute access to original
    def __getattr__(self, name):
        return getattr(self._original, name)


def translate_chapter_streaming(
    raw_html,
    config_data,
    on_chunk=None,
    on_thinking=None,
    on_complete=None,
    on_error=None,
    stop_event=None,
):
    """
    Translate a single chapter's raw HTML using the configured LLM.
    Runs synchronously (call from a background thread).

    Args:
        raw_html: The raw HTML of the single EPUB chapter to translate.
        config_data: The app's config dict (from android_config).
        on_chunk: Callback(text_fragment) for streaming content tokens.
        on_thinking: Callback(text_fragment) for thinking/reasoning tokens.
        on_complete: Callback(full_translated_text) when done.
        on_error: Callback(error_message) on failure.
        stop_event: threading.Event to signal cancellation.
    """
    if stop_event is None:
        stop_event = threading.Event()

    original_stdout = sys.stdout

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

        # Set thinking env vars
        _setup_thinking_env(config_data)

        if stop_event.is_set():
            return

        # Create client
        client = UnifiedClient(api_key=api_key, model=model)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chapter_text},
        ]

        # Install stdout capture for real-time streaming
        stream_capture = _StreamCapture(
            original_stdout=original_stdout,
            on_chunk=on_chunk,
            stop_event=stop_event,
        )
        sys.stdout = stream_capture

        # Call the API
        response = client._get_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            max_completion_tokens=None,
            response_name="reader_translate",
        )

        # Restore stdout
        sys.stdout = original_stdout

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
        # Always restore stdout
        sys.stdout = original_stdout
        # Reset stop flags
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(False)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
        except Exception:
            pass
