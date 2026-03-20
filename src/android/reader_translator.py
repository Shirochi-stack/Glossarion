# reader_translator.py
"""
Lightweight single-chapter translation engine for the in-reader translator.
Wraps UnifiedApiClient directly with streaming support.
"""

import os
import sys
import threading
import logging
import re

logger = logging.getLogger(__name__)


def _get_system_prompt(config_data):
    """Build a system prompt from the user's config settings."""
    from default_prompts import get_prompt

    # Use the configured profile or fall back to Korean_html2text
    profile = config_data.get('prompt_profile', 'Korean_html2text')
    prompt = config_data.get('system_prompt', '')

    if not prompt:
        prompt = get_prompt(profile)

    if not prompt:
        prompt = get_prompt('Korean_html2text')

    return prompt


def _strip_html_to_text(html_content):
    """Extract raw text from HTML for the reader display."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove scripts/styles
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        body = soup.find('body') or soup
        return body.get_text(separator='\n\n').strip()
    except Exception:
        # Fallback: crude tag stripping
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def translate_chapter_streaming(
    chapter_text,
    config_data,
    on_chunk=None,
    on_thinking=None,
    on_complete=None,
    on_error=None,
    stop_event=None,
):
    """
    Translate a single chapter's text content using the configured LLM.
    Runs synchronously (call from a background thread).

    Args:
        chapter_text: The plain text content of the chapter to translate.
        config_data: The app's config dict (from android_config).
        on_chunk: Callback(text_fragment) for streaming content tokens.
        on_thinking: Callback(text_fragment) for thinking/reasoning tokens.
        on_complete: Callback(full_translated_text) when done.
        on_error: Callback(error_message) on failure.
        stop_event: threading.Event to signal cancellation.
    """
    if stop_event is None:
        stop_event = threading.Event()

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

        # Reset stop flags
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

        if not api_key:
            if on_error:
                on_error("No API key configured. Set one in Translation Settings.")
            return

        if stop_event.is_set():
            return

        # Set up environment for UnifiedClient
        os.environ['API_KEY'] = api_key
        os.environ['MODEL'] = model
        os.environ['ENABLE_STREAMING'] = '1'
        os.environ['LOG_STREAM_CHUNKS'] = '1'

        # Create client
        client = UnifiedClient(
            api_key=api_key,
            model=model,
        )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chapter_text},
        ]

        if stop_event.is_set():
            return

        # Call the API — the streaming output goes to stdout/print
        # We'll capture it via a custom print hook
        full_text_parts = []
        thinking_parts = []

        # Hook into streaming by capturing printed output
        # For now, use a synchronous call and capture the result
        response = client.call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            context="reader_translate",
        )

        if stop_event.is_set():
            return

        if response and response.content:
            translated = response.content.strip()

            # Clean up any AI artifacts
            try:
                from TransateKRtoEN import PostProcessor
                translated = PostProcessor.strip_split_markers(translated)
                translated = PostProcessor.clean_ai_artifacts(translated, remove_artifacts=True)
            except Exception:
                pass

            # Simulate streaming output for the UI
            # Split into chunks and call on_chunk for each
            chunk_size = 50  # chars per chunk
            for i in range(0, len(translated), chunk_size):
                if stop_event.is_set():
                    return
                chunk = translated[i:i + chunk_size]
                full_text_parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)

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
        # Reset stop flags
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(False)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
        except Exception:
            pass
