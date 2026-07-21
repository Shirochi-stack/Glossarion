# extract_glossary_from_epub.py
import os
import json
import re
import argparse
import zipfile
import time
import sys
import unicodedata
import tiktoken
import threading
import queue
import ebooklib
import re
import tempfile
import shutil
from ebooklib import epub
from chapter_splitter import ChapterSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Dict, Tuple
from unified_api_client import UnifiedClient, UnifiedClientError
from glossary_paths import (
    get_book_glossary_dir,
    migrate_all_legacy_glossary_files,
    resolve_shared_glossary_dir,
    sanitize_glossary_folder_name,
)
from glossary_refinement import (
    DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT,
    locked_progress_file as _locked_glossary_progress_file,
    refine_glossary_entries,
    refinement_enabled as _glossary_refinement_enabled,
)
from glossary_usage import compact_extracted_entries

# Thread submission throttling (glossary batch) — mirrors translation behavior
_glossary_thread_submit_lock = threading.Lock()
_glossary_last_thread_submit = 0.0
_gender_tracker_lock = threading.Lock()


class _OrderedGlossaryBatchDispatcher:
    """Release batch units into their first API send in reading order."""

    def __init__(self, enabled=True, stop_check=None):
        self.enabled = bool(enabled)
        self.stop_check = stop_check
        self._condition = threading.Condition()
        self._next_order = 0
        self._released = set()
        self._abandoned = set()
        self._last_release = 0.0

    def _advance_abandoned_locked(self):
        while self._next_order in self._abandoned:
            self._abandoned.remove(self._next_order)
            self._next_order += 1

    def wait_for_turn(self, request_order):
        """Gate only the first send; retries and later chunks pass immediately."""
        if not self.enabled or request_order is None:
            return
        try:
            request_order = int(request_order)
        except (TypeError, ValueError):
            return

        try:
            timeout = max(
                5.0,
                float(os.getenv("ORDERED_BATCH_DISPATCH_TIMEOUT", "120")),
            )
        except (TypeError, ValueError):
            timeout = 120.0
        deadline = time.monotonic() + timeout

        with self._condition:
            if (
                request_order in self._released
                or request_order < self._next_order
            ):
                return
            self._advance_abandoned_locked()
            while request_order != self._next_order:
                if callable(self.stop_check) and self.stop_check():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        "⚠️ Ordered glossary batch dispatch timed out waiting "
                        "for an earlier spine item; releasing this request"
                    )
                    self._next_order = request_order
                    break
                self._condition.wait(timeout=min(0.1, remaining))
                self._advance_abandoned_locked()

            try:
                release_interval = max(
                    0.0,
                    float(os.getenv("ORDERED_BATCH_DISPATCH_INTERVAL", "0.025")),
                )
            except (TypeError, ValueError):
                release_interval = 0.025
            while self._last_release:
                delay = self._last_release + release_interval - time.monotonic()
                if delay <= 0:
                    break
                self._condition.wait(timeout=min(0.05, delay))

            self._released.add(request_order)
            self._last_release = time.monotonic()
            self._next_order = max(self._next_order, request_order + 1)
            self._advance_abandoned_locked()
            self._condition.notify_all()

    def abandon_if_unsent(self, request_order):
        """Prevent one pre-send worker failure from blocking later spine items."""
        if not self.enabled or request_order is None:
            return
        try:
            request_order = int(request_order)
        except (TypeError, ValueError):
            return
        with self._condition:
            if request_order in self._released or request_order < self._next_order:
                return
            self._abandoned.add(request_order)
            self._advance_abandoned_locked()
            self._condition.notify_all()

# Direct Text needs the glossary phase to be a first-class part of the chat,
# not just a collection of provider lifecycle logs.  The API client already
# prints streamed thinking from the actual API worker thread; retain that
# request-local stream until ``send_with_interrupt`` has the exact final
# response and can publish both channels together.
_direct_stream_capture_lock = threading.Lock()
_direct_stream_captures = {}
_DIRECT_TEXT_GLOSSARY_STREAM_START_PREFIX = (
    "[DIRECT_TEXT_GLOSSARY_STREAM_START] "
)


def _capture_direct_text_stream_write(text):
    """Capture streamed glossary thinking for the current API worker."""
    if os.getenv("DIRECT_TEXT_ACTIVE", "0") != "1":
        return
    value = str(text or "")
    if not value:
        return
    thread_id = threading.current_thread().ident
    if thread_id is None:
        return
    with _direct_stream_capture_lock:
        state = _direct_stream_captures.setdefault(
            thread_id, {"phase": "processing", "thinking": []}
        )
        for raw_line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if "thinking complete" in low:
                state["phase"] = "processing"
                continue
            if (
                " thinking..." in low
                or low.endswith("thinking...")
                or (stripped.startswith("🧠") and "thinking" in low)
            ):
                state["phase"] = "thinking"
                continue
            if (
                "text streaming" in low
                or "first text token" in low
                or ("first token" in low and "streaming" in low)
                or "stream finished" in low
                or "stream complete" in low
            ):
                state["phase"] = "processing"
                continue
            if state.get("phase") == "thinking" and stripped != "\u200b":
                # Provider thinking lines are indented for the application log;
                # the saved thinking file should contain the model text itself.
                state["thinking"].append(
                    raw_line[4:] if raw_line.startswith("    ") else raw_line
                )


def _consume_direct_text_thinking(thread_id):
    if thread_id is None:
        return ""
    with _direct_stream_capture_lock:
        state = _direct_stream_captures.pop(thread_id, None)
    if not isinstance(state, dict):
        return ""
    return "\n".join(str(line) for line in state.get("thinking", [])).strip()


def _emit_direct_text_glossary_stream_start(label, source_thread=""):
    """Bind live provider output to its Direct Text glossary request card.

    Some streaming providers do not print a ``Text streaming`` banner when a
    response has no reasoning block. Without an explicit request boundary the
    GUI treats those otherwise-unlabelled lines as application logs and only
    sees the authoritative response payload at completion.
    """
    if os.getenv("DIRECT_TEXT_ACTIVE", "0") != "1":
        return
    payload = json.dumps(
        {
            "label": str(label or "Glossary request"),
            "source_thread": str(source_thread or ""),
        },
        ensure_ascii=False,
    )
    print(f"{_DIRECT_TEXT_GLOSSARY_STREAM_START_PREFIX}{payload}")


def _emit_direct_text_glossary_response(
    label, content, thinking="", source_thread=""
):
    """Publish a completed glossary API response to the Direct Text dialog."""
    if os.getenv("DIRECT_TEXT_ACTIVE", "0") != "1":
        return
    response_text = content
    if isinstance(response_text, tuple):
        response_text = response_text[0] if response_text else ""
    if hasattr(response_text, "content"):
        response_text = response_text.content
    elif hasattr(response_text, "text"):
        response_text = response_text.text
    response_text = str(response_text or "")
    if not response_text:
        return
    payload = json.dumps(
        {
            "label": str(label or "Glossary request"),
            "content": response_text,
            "thinking": str(thinking or ""),
            "phase": "glossary",
            "source_thread": str(source_thread or ""),
        },
        ensure_ascii=False,
    )
    print(f"[DIRECT_TEXT_RESPONSE_PAYLOAD] {payload}")

# Fix for PyInstaller - handle stdout reconfigure more carefully
if sys.platform.startswith("win"):
    try:
        # Try to reconfigure if the method exists
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # If reconfigure doesn't work, try to set up UTF-8 another way
        import io
        import locale
        if sys.stdout and hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MODEL = os.getenv("MODEL", "gemini-2.0-flash")

def _format_sampling_log_float(value):
    try:
        number = round(float(value), 6)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    if "." not in text and "e" not in text.lower():
        text = f"{text}.0"
    return text

def _positive_int(value):
    try:
        if value in (None, ""):
            return None
        value = int(value)
        return value if value > 0 else None
    except Exception:
        return None

def _load_key_pool_from_env_or_config(env_name, config_key, config):
    keys_json = os.getenv(env_name, "[]")
    try:
        if keys_json and keys_json.strip() not in ("", "[]", "null", "None"):
            parsed = json.loads(keys_json)
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass
    keys = config.get(config_key, []) if isinstance(config, dict) else []
    return keys if isinstance(keys, list) else []

def _pool_output_limit_override(keys, global_limit):
    """Return the safe effective output limit for a rotating key pool."""
    enabled_keys = [k for k in (keys or []) if isinstance(k, dict) and k.get("enabled", True)]
    if not enabled_keys:
        return None
    limits = []
    for key_data in enabled_keys:
        limit = _positive_int(key_data.get("individual_output_token_limit"))
        limits.append(limit if limit is not None else global_limit)
    return min(limits) if limits else None

def _peek_live_pool_output_limit():
    """Peek at the live key pool to read the output token limit of the next key
    about to be assigned (based on current rotation index).

    Returns the individual_output_token_limit of that key, or None if no pool
    is active or the next key has no per-key limit configured.
    """
    # 1. Try glossary-specific key pool first
    try:
        _gk_pool = getattr(UnifiedClient, '_glossary_key_pool', None)
        if _gk_pool and hasattr(_gk_pool, 'keys') and _gk_pool.keys:
            _pool_keys = _gk_pool.keys
            _cur_idx = getattr(_gk_pool, 'current_index', 0) % len(_pool_keys)
            # Scan from rotation index to find first enabled key
            for offset in range(len(_pool_keys)):
                _key = _pool_keys[(_cur_idx + offset) % len(_pool_keys)]
                if getattr(_key, 'enabled', True):
                    limit = getattr(_key, 'individual_output_token_limit', None)
                    if limit is not None and int(limit) > 0:
                        return int(limit)
                    # Key is enabled but has no per-key limit — fall through
                    return None
    except Exception:
        pass

    # 2. Fallback: multi-key pool
    try:
        _mk_pool = getattr(UnifiedClient, '_api_key_pool', None)
        if _mk_pool and hasattr(_mk_pool, 'keys') and _mk_pool.keys:
            _pool_keys = _mk_pool.keys
            _cur_idx = getattr(_mk_pool, 'current_index', 0) % len(_pool_keys)
            for offset in range(len(_pool_keys)):
                _key = _pool_keys[(_cur_idx + offset) % len(_pool_keys)]
                if getattr(_key, 'enabled', True):
                    limit = getattr(_key, 'individual_output_token_limit', None)
                    if limit is not None and int(limit) > 0:
                        return int(limit)
                    return None
    except Exception:
        pass

    # 3. Fallback: in-memory glossary keys (raw dicts, before pool is hydrated)
    try:
        _im_gk = getattr(UnifiedClient, '_in_memory_glossary_keys', None)
        if _im_gk:
            for _gk_dict in _im_gk:
                if isinstance(_gk_dict, dict) and _gk_dict.get('enabled', True):
                    limit = _positive_int(_gk_dict.get('individual_output_token_limit'))
                    if limit is not None:
                        return limit
                    return None
    except Exception:
        pass

    return None

def _effective_glossary_output_limit(config, model_name=None):
    raw_output_env = os.getenv("GLOSSARY_MAX_OUTPUT_TOKENS", os.getenv("MAX_OUTPUT_TOKENS", "0"))
    effective = _positive_int(str(raw_output_env).strip())
    if effective is None:
        effective = _positive_int(os.getenv("MAX_OUTPUT_TOKENS", str(config.get("max_tokens", 65536)))) or 65536

    # ── Priority 1: Live key pool (runtime truth for multi-key rotation) ──
    live_limit = _peek_live_pool_output_limit()
    if live_limit is not None:
        effective = live_limit
    else:
        # ── Priority 2: Env / config JSON fallback ──
        if os.getenv("USE_GLOSSARY_KEYS", "0") == "1" or config.get("use_glossary_keys", False):
            glossary_keys = _load_key_pool_from_env_or_config("GLOSSARY_API_KEYS", "glossary_keys", config)
            pool_limit = _pool_output_limit_override(glossary_keys, effective)
            if pool_limit is not None:
                effective = pool_limit
        elif config.get("use_multi_api_keys", False) or os.getenv("USE_MULTI_API_KEYS", "0") == "1":
            multi_keys = _load_key_pool_from_env_or_config("MULTI_API_KEYS", "multi_api_keys", config)
            pool_limit = _pool_output_limit_override(multi_keys, effective)
            if pool_limit is not None:
                effective = pool_limit

    try:
        with UnifiedClient._model_limits_lock:
            cached_limit = getattr(UnifiedClient, "_model_token_limits", {}).get(model_name or MODEL)
        if cached_limit and cached_limit > 0:
            effective = min(effective, cached_limit)
    except Exception:
        pass

    return effective

def interruptible_sleep(duration, check_stop_fn, interval=0.1):
    """Sleep that can be interrupted by stop request or graceful stop"""
    elapsed = 0
    while elapsed < duration:
        if check_stop_fn and check_stop_fn():  # Add safety check for None
            return False  # Interrupted
        # Also bail on graceful stop — no point sleeping if we'll skip anyway
        if os.environ.get('GRACEFUL_STOP') == '1' or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
            return False
        sleep_time = min(interval, duration - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
    return True  # Completed normally
    
def cancel_all_futures(futures):
    """Cancel all pending futures immediately"""
    cancelled_count = 0
    for future in futures:
        if not future.done() and future.cancel():
            cancelled_count += 1
    return cancelled_count


def _is_graceful_stop_skip_error(err: Exception) -> bool:
    """True when a queued API call was prevented from starting due to graceful stop."""
    try:
        s = str(err).lower()
    except Exception:
        return False
    if "graceful stop active - not starting new api call" in s:
        return True
    if not _glossary_is_graceful_stop_active():
        return False
    return (
        "skipped before api call" in s
        or "stopped by user during threading delay" in s
    )


def _graceful_stop_should_drain_after_result(result_committed: bool) -> bool:
    """A committed result starts graceful draining; it must not end it early."""
    return bool(
        result_committed
        and os.environ.get('GRACEFUL_STOP_COMPLETED') == '1'
    )

def create_client_with_multi_key_support(api_key, model, output_dir, config, context='glossary'):
    """Create a UnifiedClient with multi API key support if enabled.
    
    Priority order for key pool:
    1. Glossary-specific keys (USE_GLOSSARY_KEYS=1 + GLOSSARY_API_KEYS / config['glossary_keys'])
    2. Multi-API keys (use_multi_api_keys + config['multi_api_keys'])
    3. Single key mode (main GUI key only)
    """
    
    # ── Step 1: Determine which key pool to use ──────────────────────────
    use_glossary_keys = os.getenv('USE_GLOSSARY_KEYS', '0') == '1'
    use_refinement_keys = os.getenv('USE_GLOSSARY_REFINEMENT_KEYS', '0') == '1' or config.get('use_glossary_refinement_keys', False)
    refinement_keys = []
    if use_refinement_keys:
        refinement_keys_json = os.getenv('GLOSSARY_REFINEMENT_API_KEYS', '[]')
        try:
            if refinement_keys_json and refinement_keys_json.strip() not in ('', '[]', 'null', 'None'):
                refinement_keys = json.loads(refinement_keys_json)
        except Exception:
            refinement_keys = []
        if not refinement_keys:
            refinement_keys = config.get('glossary_refinement_keys', [])
        if refinement_keys:
            os.environ['USE_GLOSSARY_REFINEMENT_KEYS'] = '1'
            os.environ['GLOSSARY_REFINEMENT_API_KEYS'] = json.dumps(refinement_keys)
            try:
                UnifiedClient.set_in_memory_glossary_refinement_keys(
                    refinement_keys,
                    force_rotation=config.get('force_key_rotation', True),
                    rotation_frequency=config.get('rotation_frequency', 1),
                )
            except Exception:
                pass
    
    # Try to load glossary-specific keys
    glossary_keys = []
    if use_glossary_keys:
        # Source 1: GLOSSARY_API_KEYS env var (set by GUI)
        glossary_keys_json = os.getenv('GLOSSARY_API_KEYS', '[]')
        try:
            if glossary_keys_json and glossary_keys_json.strip() not in ('', '[]', 'null', 'None'):
                glossary_keys = json.loads(glossary_keys_json)
        except Exception:
            glossary_keys = []
        
        # Source 2: config['glossary_keys'] (from config file)
        if not glossary_keys:
            glossary_keys = config.get('glossary_keys', [])
    
    if use_glossary_keys and glossary_keys:
        # ── GLOSSARY KEYS MODE ──────────────────────────────────────────
        # Use glossary-specific keys for rotation, NOT the translation multi-keys
        if context == 'glossary':
            print("🔑 Glossary API Key mode enabled for glossary extraction")
        
        os.environ['USE_MULTI_API_KEYS'] = '1'
        os.environ['USE_MULTI_KEYS'] = '1'
        os.environ['USE_GLOSSARY_KEYS'] = '1'
        os.environ['FORCE_KEY_ROTATION'] = '1' if config.get('force_key_rotation', True) else '0'
        os.environ['ROTATION_FREQUENCY'] = str(config.get('rotation_frequency', 1))

        # Store glossary keys in their DEDICATED glossary pool — NOT the shared
        # multi-key pool.  Using set_in_memory_multi_keys here would overwrite
        # the multi-key pool object in-place, corrupting the 6-key pool that
        # the Multi API Key Manager dialog references via _bind_shared_pool.
        try:
            UnifiedClient.set_in_memory_glossary_keys(
                glossary_keys,
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        except Exception:
            pass
        
        # ALSO load the regular multi-keys (if configured) so that
        # UnifiedClient.__init__ can enter multi-key mode for rotation/fallback.
        # This keeps the multi-key pool intact (loaded with the real multi-keys)
        # while the glossary pool holds the glossary-specific keys separately.
        regular_multi_keys = config.get('multi_api_keys', [])
        if regular_multi_keys and config.get('use_multi_api_keys', False):
            try:
                UnifiedClient.set_in_memory_multi_keys(
                    regular_multi_keys,
                    force_rotation=config.get('force_key_rotation', True),
                    rotation_frequency=config.get('rotation_frequency', 1),
                )
            except Exception:
                pass
        
        if context == 'glossary':
            print(f"   • Glossary keys configured: {len(glossary_keys)}")
            print(f"   • Force rotation: {config.get('force_key_rotation', True)}")
            print(f"   • Rotation frequency: every {config.get('rotation_frequency', 1)} request(s)")
    
    elif config.get('use_multi_api_keys', False) and config.get('multi_api_keys'):
        # ── MULTI-KEY MODE (no glossary keys) ────────────────────────────
        # Fall back to regular translation multi-keys
        print("🔑 Multi API Key mode enabled for glossary extraction")
        
        os.environ['USE_MULTI_API_KEYS'] = '1'
        os.environ['USE_MULTI_KEYS'] = '1'
        os.environ['FORCE_KEY_ROTATION'] = '1' if config.get('force_key_rotation', True) else '0'
        os.environ['ROTATION_FREQUENCY'] = str(config.get('rotation_frequency', 1))

        try:
            UnifiedClient.set_in_memory_multi_keys(
                config['multi_api_keys'],
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        except Exception:
            pass
        
        print(f"   • Keys configured: {len(config['multi_api_keys'])}")
        print(f"   • Force rotation: {config.get('force_key_rotation', True)}")
        print(f"   • Rotation frequency: every {config.get('rotation_frequency', 1)} request(s)")
    else:
        # Ensure multi-key mode is disabled in environment
        os.environ['USE_MULTI_API_KEYS'] = '0'
        
    # Create UnifiedClient normally - it will check environment variables
    client = UnifiedClient(api_key=api_key, model=model, output_dir=output_dir)
    client.context = 'glossary'
    
    # Check if fallback keys should be used
    # FORCE enable if env var is '1' (handled by GUI toggle)
    use_fallback_keys = os.getenv('USE_FALLBACK_KEYS', '0') == '1'
    
    if use_fallback_keys:
        # If enabled, ensure the instance knows it
        client.use_fallback_keys = True
    else:
        # If disabled, force disable on the client
        client.use_fallback_keys = False
    
    # Check if we should use main key as fallback
    use_main_key_fb = os.getenv('USE_MAIN_KEY_FALLBACK', '1') == '1'
    
    # If explicitly '0' (False), disable main key fallback
    # If '1' (True), ensure it is enabled
    if not use_main_key_fb:
        client.use_main_key_fallback = False
        client._use_main_key_fallback = False
    else:
        # Explicitly enable if set to '1'
        client.use_main_key_fallback = True
        client._use_main_key_fallback = True

    return client
    
# Log assistant prompt at module initialization if configured
def _log_assistant_prompt_once():
    """Log assistant prompt once at the start of extraction if configured"""
    if not hasattr(_log_assistant_prompt_once, '_logged'):
        _log_assistant_prompt_once._logged = False
    if not _log_assistant_prompt_once._logged:
        assistant_prompt = os.getenv('ASSISTANT_PROMPT', '').strip()
        if assistant_prompt:
            print(f"🤖 Assistant Prompt: {assistant_prompt}")
            _log_assistant_prompt_once._logged = True

def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None, chapter_idx=None, chapter_num=None, chunk_idx=None, total_chunks=None, merged_chapters=None, before_send_callback=None, context='glossary'):
    """Send API request with interrupt capability and optional timeout retry
    
    Args:
        merged_chapters: Optional list of chapter numbers that were merged into this request
    """
    global _glossary_last_thread_submit, _glossary_thread_submit_lock
    
    # Early exit: if stop/graceful-stop is already flagged, skip client init and delays
    if stop_check_fn() or os.environ.get('GRACEFUL_STOP') == '1' or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
        raise UnifiedClientError("Glossary extraction stopped by user (skipped before API call)")
    
    # Mark that an API call is now active (for graceful stop logic)
    os.environ['GRACEFUL_STOP_API_ACTIVE'] = '1'
    
    # Get timeout retry settings
    max_timeout_retries = int(os.getenv('TIMEOUT_RETRY_ATTEMPTS', '2'))
    timeout_retry_count = 0
    
    # Format chapter context for logs
    chapter_label = "API call"
    if merged_chapters and len(merged_chapters) > 1:
        # Build merged label like "Merged 1-3"
        try:
            merged_nums = sorted([int(c) for c in merged_chapters if c is not None])
            if len(merged_nums) == 1:
                chapter_label = f"Merged {merged_nums[0]}"
            else:
                chapter_label = f"Merged {merged_nums[0]}-{merged_nums[-1]}"
            if chunk_idx and total_chunks:
                chapter_label = f"{chapter_label} (chunk {chunk_idx}/{total_chunks})"
        except Exception:
            pass
    elif chapter_idx is not None or chapter_num is not None:
        try:
            chap_num = int(chapter_num) if chapter_num is not None else int(chapter_idx) + 1
        except Exception:
            chap_num = chapter_num if chapter_num is not None else chapter_idx
        if chunk_idx and total_chunks:
            chapter_label = f"Chapter {chap_num} (chunk {chunk_idx}/{total_chunks})"
        else:
            chapter_label = f"Chapter {chap_num}"
    
    result_queue = queue.Queue()

    def _capture_actual_request_metadata():
        actual_model = None
        actual_key = None
        try:
            if hasattr(client, 'get_last_actual_request_model'):
                actual_model = client.get_last_actual_request_model()
        except Exception:
            actual_model = None
        try:
            if hasattr(client, 'get_last_actual_request_key_identifier'):
                actual_key = client.get_last_actual_request_key_identifier()
        except Exception:
            actual_key = None
        return actual_model, actual_key

    def _install_actual_request_metadata(actual_model, actual_key):
        try:
            from unified_api_client import set_current_thread_actual_request_model
            set_current_thread_actual_request_model(actual_model, actual_key)
        except Exception:
            pass

    # Honor runtime toggle: if RETRY_TIMEOUT is off, disable chunk timeout entirely.
    env_retry = os.getenv("RETRY_TIMEOUT")
    if env_retry is not None:
        retry_enabled = env_retry.strip().lower() not in ("0", "false", "off", "")
    else:
        retry_enabled = True  # legacy default

    if not retry_enabled:
        chunk_timeout = None
    else:
        # Allow overriding timeout via env; treat non-positive/blank as disabled
        env_ct = os.getenv("CHUNK_TIMEOUT")
        if env_ct is not None and str(env_ct).strip() not in ("", "none"):
            try:
                ct_val = float(env_ct)
                chunk_timeout = None if ct_val <= 0 else ct_val
            except Exception:
                pass
    
    while True:  # Retry loop for timeout and cancelled errors
        attempt_cancel_event = threading.Event()
        api_call_state = {"tls": None}

        def _close_api_thread_transport() -> bool:
            closed_any = False
            api_tls = api_call_state.get("tls")
            if api_tls is not None:
                for attr in (
                    "current_stream",
                    "current_openai_sdk_client",
                    "current_httpx_client",
                    "current_oai_http_client",
                    "openai_client",
                    "gemini_client",
                ):
                    obj = getattr(api_tls, attr, None)
                    if obj is None:
                        continue
                    try:
                        if attr == "gemini_client" and hasattr(obj, "_client"):
                            http_client = obj._client
                            if hasattr(http_client, "close"):
                                http_client.close()
                            if hasattr(http_client, "_transport"):
                                http_client._transport.close()
                        if hasattr(obj, "close"):
                            obj.close()
                        closed_any = True
                    except Exception:
                        pass
                    try:
                        setattr(api_tls, attr, None)
                    except Exception:
                        pass
                try:
                    openai_clients = getattr(api_tls, "openai_clients", None)
                    if isinstance(openai_clients, dict):
                        for obj in list(openai_clients.values()):
                            try:
                                if hasattr(obj, "close"):
                                    obj.close()
                                    closed_any = True
                            except Exception:
                                pass
                        openai_clients.clear()
                except Exception:
                    pass
            return closed_any

        def _cancel_current_api_call(*, mark_client_cancel: bool = False) -> None:
            attempt_cancel_event.set()
            closed_local = _close_api_thread_transport()
            if closed_local or not mark_client_cancel:
                return
            if hasattr(client, 'cancel_current_operation'):
                client.cancel_current_operation()

        def api_call():
            tls_for_local_cancel = None
            had_local_cancel = False
            previous_local_cancel = None
            try:
                # Apply chapter/chunk context in THIS thread so UnifiedClient's
                # thread-local metadata is visible to watchdog/payloads.
                try:
                    if hasattr(client, 'set_chapter_context'):
                        if chapter_num is not None:
                            chap_val = chapter_num
                        else:
                            chap_val = (chapter_idx + 1) if isinstance(chapter_idx, int) else (
                                int(chapter_idx) + 1 if chapter_idx is not None and str(chapter_idx).isdigit() else chapter_idx
                            )
                        client.set_chapter_context(
                            chapter=chap_val if (chapter_idx is not None or chapter_num is not None) else None,
                            chunk=chunk_idx,
                            total_chunks=total_chunks,
                            merged_chapters=merged_chapters,
                        )
                except Exception:
                    pass
                try:
                    if hasattr(client, '_get_thread_local_client'):
                        tls_for_local_cancel = client._get_thread_local_client()
                        api_call_state["tls"] = tls_for_local_cancel
                        had_local_cancel = hasattr(tls_for_local_cancel, 'local_cancel_check')
                        previous_local_cancel = getattr(tls_for_local_cancel, 'local_cancel_check', None)
                        tls_for_local_cancel.local_cancel_check = attempt_cancel_event.is_set
                except Exception:
                    tls_for_local_cancel = None
                # Reinitialize client if needed (check correct client based on type)
                # Skip in multi-key mode — _ensure_thread_client handles per-thread client setup
                if not getattr(client, '_multi_key_mode', False):
                    client_type = getattr(client, 'client_type', 'unknown')
                    needs_reinit = False
                    
                    if client_type == 'gemini':
                        needs_reinit = hasattr(client, 'gemini_client') and client.gemini_client is None
                    elif client_type == 'openai':
                        needs_reinit = hasattr(client, 'openai_client') and client.openai_client is None
                    
                    if needs_reinit:
                        try:
                            print(f"   🔄 Reinitializing {client_type} client...")
                            client._setup_client()
                        except Exception as reinit_err:
                            print(f"   ⚠️ Failed to reinitialize client: {reinit_err}")
                
                tls_for_callback = None
                if callable(before_send_callback):
                    try:
                        if hasattr(client, '_get_thread_local_client'):
                            tls_for_callback = client._get_thread_local_client()
                            tls_for_callback.pre_api_call_callback = before_send_callback
                        else:
                            before_send_callback()
                    except Exception as cb_err:
                        print(f"⚠️ Failed to register pre-send glossary progress callback: {cb_err}")

                start_time = time.time()
                try:
                    _emit_direct_text_glossary_stream_start(
                        chapter_label,
                        threading.current_thread().name,
                    )
                    result = client.send(messages, temperature=temperature, max_tokens=max_tokens, context=context or 'glossary')
                finally:
                    if tls_for_callback is not None:
                        try:
                            tls_for_callback.pre_api_call_callback = None
                        except Exception:
                            pass
                elapsed = time.time() - start_time

                if attempt_cancel_event.is_set():
                    return
                
                # Capture raw response object for thought signatures (if available)
                raw_obj = None
                if hasattr(client, 'get_last_response_object'):
                    resp_obj = client.get_last_response_object()
                    if resp_obj and hasattr(resp_obj, 'raw_content_object'):
                        raw_obj = resp_obj.raw_content_object
                        # if raw_obj:
                        #     print("🧠 Captured thought signature for glossary extraction")
                
                actual_model, actual_key = _capture_actual_request_metadata()
                # Include raw_obj plus concrete model/key metadata in the result tuple.
                result_queue.put((result, elapsed, raw_obj, actual_model, actual_key))
            except Exception as e:
                if attempt_cancel_event.is_set():
                    return
                actual_model, actual_key = _capture_actual_request_metadata()
                try:
                    e._glossarion_actual_model = actual_model
                    e._glossarion_actual_key = actual_key
                except Exception:
                    pass
                result_queue.put(e)
            finally:
                if tls_for_local_cancel is not None:
                    try:
                        if had_local_cancel:
                            tls_for_local_cancel.local_cancel_check = previous_local_cancel
                        else:
                            tls_for_local_cancel.local_cancel_check = None
                    except Exception:
                        pass
        # Apply submission delay shared across glossary batch threads to space out API launches.
        # Priority: per-key api_call_delay from glossary keys > global SEND_INTERVAL_SECONDS
        try:
            thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", os.getenv("THREAD_SUBMISSION_DELAY", "0.1")))
        except Exception:
            thread_delay = 0.1

        # Check for per-key delay from glossary pool first.
        # IMPORTANT: Key rotation hasn't happened yet (it occurs inside client.send()),
        # so client._per_key_api_delay is stale/None. We must peek at the pool's
        # current rotation index to find the NEXT key's delay.
        _per_key_delay = None

        # 1. Peek at glossary key pool — find the key that will be selected next
        try:
            _ctx_norm = str(context or '').strip().lower().replace(' ', '_').replace('-', '_')
            _gk_pool = (
                getattr(UnifiedClient, '_glossary_refinement_key_pool', None)
                if _ctx_norm == 'glossary_refinement'
                else getattr(UnifiedClient, '_glossary_key_pool', None)
            )
            if _gk_pool and hasattr(_gk_pool, 'keys') and _gk_pool.keys:
                _pool_keys = _gk_pool.keys
                _cur_idx = getattr(_gk_pool, 'current_index', 0) % len(_pool_keys)
                # Read the NEXT key's delay (the one that will be selected)
                _next_key = _pool_keys[_cur_idx]
                _gk_d = getattr(_next_key, 'api_call_delay', 0.0) or 0.0
                if _gk_d > 0:
                    _per_key_delay = _gk_d
        except Exception:
            pass

        # 2. Fallback: peek at multi-key pool if no glossary pool
        if _per_key_delay is None:
            try:
                _mk_pool = getattr(UnifiedClient, '_api_key_pool', None)
                if _mk_pool and hasattr(_mk_pool, 'keys') and _mk_pool.keys:
                    _pool_keys = _mk_pool.keys
                    _cur_idx = getattr(_mk_pool, 'current_index', 0) % len(_pool_keys)
                    _next_key = _pool_keys[_cur_idx]
                    _gk_d = getattr(_next_key, 'api_call_delay', 0.0) or 0.0
                    if _gk_d > 0:
                        _per_key_delay = _gk_d
            except Exception:
                pass

        # 3. Fallback: check in-memory glossary keys (raw dict list)
        if _per_key_delay is None:
            try:
                _im_gk = getattr(UnifiedClient, '_in_memory_glossary_keys', None)
                if _im_gk:
                    for _gk_dict in _im_gk:
                        _gk_d = float(_gk_dict.get('api_call_delay', 0)) if _gk_dict.get('api_call_delay') not in (None, '') else 0.0
                        if _gk_d > 0:
                            _per_key_delay = _gk_d
                            break
            except Exception:
                pass

        # 4. Last resort: check client instance (may be set from a previous rotation)
        if _per_key_delay is None:
            try:
                _per_key_delay = getattr(client, '_per_key_api_delay', None)
            except Exception:
                pass

        if _per_key_delay is not None and _per_key_delay > 0:
            api_delay = float(_per_key_delay)
        else:
            try:
                api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
            except Exception:
                api_delay = 2.0

        # Delay is handled by _apply_api_call_stagger inside client.send()

        api_thread = threading.Thread(target=api_call)
        api_thread.daemon = True
        api_thread.start()
        
        timeout = chunk_timeout  # None means wait indefinitely
        check_interval = 0.1
        elapsed = 0
        
        try:
            while True:
                try:
                    # Check for results with shorter timeout
                    result = result_queue.get(timeout=check_interval)
                    if isinstance(result, Exception):
                        _install_actual_request_metadata(
                            getattr(result, '_glossarion_actual_model', None),
                            getattr(result, '_glossarion_actual_key', None),
                        )
                        _consume_direct_text_thinking(api_thread.ident)
                        raise result
                    if isinstance(result, tuple):
                        # Check if we have the new format with response object and model/key metadata.
                        if len(result) >= 5:
                            api_result, api_time, raw_obj, actual_model, actual_key = result[:5]
                            _install_actual_request_metadata(actual_model, actual_key)
                        elif len(result) == 3:
                            api_result, api_time, raw_obj = result
                            _install_actual_request_metadata(*_capture_actual_request_metadata())
                        else:
                            # Old format without response object
                            api_result, api_time = result
                            raw_obj = None
                            _install_actual_request_metadata(*_capture_actual_request_metadata())
                        
                        if chunk_timeout and api_time > chunk_timeout:
                            if hasattr(client, '_in_cleanup'):
                                client._in_cleanup = True
                            _cancel_current_api_call()
                            raise UnifiedClientError(f"API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                        
                        # client.send() returns (str, Optional[str]) tuple
                        # Extract the content and finish_reason from the tuple
                        if isinstance(api_result, tuple):
                            content, finish_reason = api_result
                        else:
                            # Single string result
                            content = api_result
                            finish_reason = 'stop'
                        
                        # raw_obj was already captured in the API thread and included in result
                        # Mark API call as no longer active
                        os.environ['GRACEFUL_STOP_API_ACTIVE'] = '0'
                        # If graceful stop was requested, mark that an API call completed
                        if os.environ.get('GRACEFUL_STOP') == '1':
                            os.environ['GRACEFUL_STOP_COMPLETED'] = '1'
                        glossary_thinking = _consume_direct_text_thinking(
                            api_thread.ident
                        )
                        _emit_direct_text_glossary_response(
                            chapter_label,
                            content,
                            glossary_thinking,
                            api_thread.name,
                        )
                        return content, finish_reason or 'stop', raw_obj
                except queue.Empty:
                    # During graceful stop, don't cancel the API call - let it complete
                    if os.environ.get('GRACEFUL_STOP') != '1' and stop_check_fn():
                        # More aggressive cancellation
                        # print("🛑 Stop requested - cancelling API call immediately...")  # Redundant
                        
                        # Set cleanup flag
                        if hasattr(client, '_in_cleanup'):
                            client._in_cleanup = True
                        
                        # Try to cancel the operation
                        _cancel_current_api_call(mark_client_cancel=True)
                        
                        # Don't wait for the thread to finish - just raise immediately
                        raise UnifiedClientError("Glossary extraction stopped by user")
                    
                    if timeout is not None:
                        elapsed += check_interval
                        if elapsed >= timeout:
                            if hasattr(client, '_in_cleanup'):
                                client._in_cleanup = True
                            _cancel_current_api_call()
                            try:
                                api_thread.join(timeout=2.0)
                            except Exception:
                                pass
                            raise UnifiedClientError(f"API call timed out after {timeout} seconds") from None
        
        except UnifiedClientError as e:
            error_msg = str(e)

            # A deliberate cancellation is terminal, not a network timeout.
            # Retrying it only repeats the cancelled state and produces a chain
            # of misleading "timeout retry" messages after the Stop button.
            if (
                str(getattr(e, 'error_type', '') or '').lower() == 'cancelled'
                or 'operation cancelled by user' in error_msg.lower()
                or 'glossary extraction stopped by user' in error_msg.lower()
            ):
                raise
            
            # Treat non-user transport cancellations (from a client being
            # closed unexpectedly) as timeouts.
            if "cancelled" in error_msg.lower() or "Gemini client not initialized" in error_msg or "timed out" in error_msg.lower():
                # Check stop flag before retrying
                if stop_check_fn():
                    # print("❌ Glossary extraction stopped by user during timeout retry")  # Redundant
                    raise
                
                if timeout_retry_count < max_timeout_retries:
                    _consume_direct_text_thinking(api_thread.ident)
                    timeout_retry_count += 1
                    # Detailed log with chapter context like TransateKRtoEN.py
                    if "timed out" in error_msg.lower():
                        if chunk_timeout:
                            print(f"⚠️ {chapter_label}: API call timed out after {chunk_timeout} seconds, retrying ({timeout_retry_count}/{max_timeout_retries})...")
                        else:
                            print(f"⚠️ {chapter_label}: API call timed out, retrying ({timeout_retry_count}/{max_timeout_retries})...")
                    elif "Gemini client not initialized" in error_msg:
                        print(f"⚠️ {chapter_label}: {error_msg}, retrying ({timeout_retry_count}/{max_timeout_retries})...")
                    else:
                        print(f"⚠️ {chapter_label}: {error_msg}, retrying ({timeout_retry_count}/{max_timeout_retries})...")

                    # Reinitialize the client if it was closed (check correct client based on type)
                    # Skip in multi-key mode — _ensure_thread_client handles per-thread client setup
                    if not getattr(client, '_multi_key_mode', False):
                        client_type = getattr(client, 'client_type', 'unknown')
                        needs_reinit = False

                        if client_type == 'gemini':
                            needs_reinit = hasattr(client, 'gemini_client') and client.gemini_client is None
                        elif client_type == 'openai':
                            needs_reinit = hasattr(client, 'openai_client') and client.openai_client is None

                        if needs_reinit:
                            try:
                                print(f"   🔄 Reinitializing {client_type} client...")
                                client._setup_client()
                            except Exception as reinit_err:
                                print(f"   ⚠️ Failed to reinitialize client: {reinit_err}")

                    # Add staggered delay before retry
                    # Prefer per-key delay from glossary keys, fall back to SEND_INTERVAL_SECONDS
                    import random
                    _retry_per_key = getattr(client, '_per_key_api_delay', None)
                    base_delay = float(_retry_per_key) if _retry_per_key and float(_retry_per_key) > 0 else float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
                    retry_delay = random.uniform(base_delay / 2, base_delay)
                    print(f"   ⏳ Waiting {retry_delay:.1f}s before retry...")
                    time.sleep(retry_delay)

                    # Clear the queue and continue retry loop
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    continue
                else:
                    print(f"❌ Max timeout retries ({max_timeout_retries}) reached")
                    raise
            else:
                # Other errors, re-raise immediately
                raise

# Parse token limit from environment variable (same logic as translation)
def parse_glossary_token_limit():
    """Parse token limit from environment variable"""
    env_value = os.getenv("GLOSSARY_TOKEN_LIMIT", "1000000").strip()
    
    if not env_value or env_value == "":
        return None, "unlimited"
    
    if env_value.lower() == "unlimited":
        return None, "unlimited"
    
    if env_value.isdigit() and int(env_value) > 0:
        limit = int(env_value)
        return limit, str(limit)
    
    # Default fallback
    return 1000000, "1000000 (default)"

MAX_GLOSSARY_TOKENS, GLOSSARY_LIMIT_STR = parse_glossary_token_limit()
def _compute_safe_input_tokens(max_output_tokens: int, compression_factor: float, *, safety_margin: int = 500, minimum: int = 1000) -> int:
    """
    Mirror the chapter split budgeting used in TransateKRtoEN:
    available_tokens = (effective_output_tokens - safety_margin) / compression_factor, clamped to a minimum.
    """
    try:
        effective = int(max_output_tokens)
    except Exception:
        effective = 65536
    if effective <= 0:
        effective = 65536
    try:
        cf = float(compression_factor)
        if cf <= 0:
            cf = 0.000000000001
    except Exception:
        cf = 1.0
    budget = int((effective - safety_margin) / cf)
    return max(budget, minimum)

# Global stop flag for GUI integration
_stop_requested = False

# Threading locks for atomic glossary saves
_glossary_json_lock = threading.Lock()
_glossary_csv_lock = threading.Lock()
_progress_lock = threading.Lock()
_history_lock = threading.Lock()  # For thread-safe history access in batch mode

def _atomic_replace_file(temp_path: str, target_path: str, max_retries: int = 3, delay: float = 0.5):
    """Atomically replace target_path with temp_path, retrying on Windows lock errors.
    
    Uses os.replace() which is atomic on Windows (unlike os.remove+os.rename).
    Retries with a short delay to handle transient file locks from antivirus,
    editors, or other processes that briefly hold the file open.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return  # Success
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    # All retries exhausted — clean up temp file and raise
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
    raise last_err
# Global book title cache (set in main)
BOOK_TITLE_RAW = None
BOOK_TITLE_TRANSLATED = None
BOOK_TITLE_PRESENT = False
BOOK_TITLE_VALUE = None
GLOSSARY_SOURCE_LANGUAGE = None
GLOSSARY_SOURCE_LANGUAGE_PATH = None
_GLOSSARY_SOURCE_LANGUAGE_LOADED = False
_GLOSSARY_SOURCE_LANGUAGE_LOGGED = False
GLOSSARY_SOURCE_SCRIPT = None
GLOSSARY_SOURCE_SCRIPT_IS_CJK = None
_GLOSSARY_SOURCE_SCRIPT_READY = False
_GLOSSARY_SOURCE_SCRIPT_LOGGED = False
_glossary_source_script_lock = threading.Lock()

def _ensure_user_message(msgs: List[Dict], fallback_text: str) -> List[Dict]:
    """
    Guarantee at least one user message is present before sending to the API.
    If missing, append a user message using the provided fallback text (chapter or chunk).
    """
    if any(m.get("role") == "user" for m in msgs):
        return msgs
    safe_text = fallback_text or ""
    if not safe_text.strip():
        print("⚠️ User prompt was empty or missing; inserting placeholder to avoid empty request")
    else:
        print("⚠️ User prompt missing from messages; auto-inserting chapter text as user message")
    return msgs + [{"role": "user", "content": safe_text}]

def _sanitize_messages_for_api(msgs: List[Dict], fallback_text: str) -> List[Dict]:
    """
    Prepare messages for API/payload: ensure a user message exists and drop non-serializable fields.
    - Guarantees a user role using the provided fallback text.
    - Removes _raw_content_object (often contains bytes/thought signatures) to keep payload JSON valid.
    - Normalizes None content to empty string.
    """
    msgs = _ensure_user_message(msgs, fallback_text)
    sanitized = []
    for m in msgs:
        m2 = dict(m)
        raw_obj = m2.pop("_raw_content_object", None)

        # If content is empty, try to recover text from raw content object parts
        content = m2.get("content")
        if (content is None or content == "") and raw_obj:
            try:
                parts = []
                if hasattr(raw_obj, "parts"):
                    parts = raw_obj.parts or []
                elif isinstance(raw_obj, dict):
                    parts = raw_obj.get("parts") or []
                texts = []
                for p in parts:
                    if hasattr(p, "text") and getattr(p, "text", None):
                        texts.append(getattr(p, "text"))
                    elif isinstance(p, dict) and p.get("text"):
                        texts.append(p.get("text"))
                if texts:
                    m2["content"] = "\n".join(texts)
            except Exception:
                pass

        if m2.get("content") is None:
            m2["content"] = ""
        sanitized.append(m2)
    return sanitized

def _mark_book_title_from_csv(csv_text: str):
    """Detect existing book entry in CSV content and mark presence flag."""
    global BOOK_TITLE_PRESENT
    if not csv_text:
        return
    for line in csv_text.splitlines():
        if line.lower().startswith("book,"):
            BOOK_TITLE_PRESENT = True
            return


def _extract_title_from_metadata(meta: Dict) -> str:
    """Best-effort retrieval of a book title from metadata structures."""
    if not isinstance(meta, dict):
        return None

    title_keys = [
        "title",
        "book_title",
        "bookTitle",
        "title_translated",
        "translated_title",
        "title_en",
    ]
    for key in title_keys:
        val = meta.get(key)
        if val:
            return str(val).strip()

    # Look into common nested objects
    for nested_key in ("metadata", "opf", "info", "data"):
        nested = meta.get(nested_key)
        if isinstance(nested, dict):
            nested_title = _extract_title_from_metadata(nested)
            if nested_title:
                return nested_title
    return None


def _unique_metadata_candidates(output_path: str = None, epub_path: str = None) -> List[str]:
    """Return likely metadata.json paths without duplicate probes."""
    epub_base = os.path.splitext(os.path.basename(epub_path or os.getenv("EPUB_PATH", "") or ""))[0]
    candidates = []

    for env_key in ("METADATA_JSON_PATH", "METADATA_PATH"):
        env_path = (os.getenv(env_key) or "").strip()
        if env_path:
            candidates.append(env_path)

    for env_key in ("EPUB_OUTPUT_DIR", "OUTPUT_DIRECTORY", "OUTPUT_DIR"):
        env_dir = (os.getenv(env_key) or "").strip()
        if not env_dir:
            continue
        candidates.append(os.path.join(env_dir, "metadata.json"))
        if epub_base:
            candidates.append(os.path.join(env_dir, epub_base, "metadata.json"))

    if output_path:
        meta_dir = os.path.abspath(os.path.dirname(output_path) or ".")
        candidates.append(os.path.join(meta_dir, "metadata.json"))
        if epub_base:
            candidates.append(os.path.join(meta_dir, epub_base, "metadata.json"))
        parent = os.path.dirname(meta_dir)
        grandparent = os.path.dirname(parent)
        if parent:
            candidates.append(os.path.join(parent, "metadata.json"))
            if epub_base:
                candidates.append(os.path.join(parent, epub_base, "metadata.json"))
        if grandparent and grandparent != parent:
            candidates.append(os.path.join(grandparent, "metadata.json"))
            if epub_base:
                candidates.append(os.path.join(grandparent, epub_base, "metadata.json"))

    if epub_base:
        candidates.append(os.path.join(os.getcwd(), epub_base, "metadata.json"))
        if epub_path:
            source_dir = os.path.dirname(os.path.abspath(epub_path))
            candidates.append(os.path.join(source_dir, epub_base, "metadata.json"))

    seen = set()
    unique = []
    for path in candidates:
        if not path:
            continue
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            norm = path
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _normalize_detected_language(value):
    lang = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if lang in ("", "unknown", "und", "none", "null"):
        return None
    return lang


def _read_detected_language_from_metadata(output_path: str = None, epub_path: str = None):
    for meta_path in _unique_metadata_candidates(output_path, epub_path):
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                continue
            detected = _normalize_detected_language(meta.get("detected_language") or meta.get("source_language"))
            if detected:
                return detected, meta_path
        except Exception as e:
            print(f"[Warning] Could not read metadata.json for detected language: {e}")
    return None, None


def _metadata_language_is_cjk(language: str) -> bool:
    lang = _normalize_detected_language(language)
    cjk_langs = {
        "korean", "ko", "kor",
        "japanese", "ja", "jpn",
        "chinese", "zh", "zho", "chi",
        "simplified chinese", "traditional chinese",
        "mandarin", "cantonese",
    }
    return lang in cjk_langs


def _set_glossary_source_language_from_metadata(output_path: str = None, epub_path: str = None, log: bool = False):
    global GLOSSARY_SOURCE_LANGUAGE, GLOSSARY_SOURCE_LANGUAGE_PATH
    global _GLOSSARY_SOURCE_LANGUAGE_LOADED, _GLOSSARY_SOURCE_LANGUAGE_LOGGED
    detected, meta_path = _read_detected_language_from_metadata(output_path, epub_path)
    GLOSSARY_SOURCE_LANGUAGE = detected
    GLOSSARY_SOURCE_LANGUAGE_PATH = meta_path
    _GLOSSARY_SOURCE_LANGUAGE_LOADED = True
    if detected and log and not _GLOSSARY_SOURCE_LANGUAGE_LOGGED:
        status = "CJK source confirmed" if _metadata_language_is_cjk(detected) else "non-CJK source, filter skipped"
        source = "metadata.json" if meta_path else "environment"
        print(f"🔍 [CJK Filter] Source language from {source}: {detected} -> {status}")
        _GLOSSARY_SOURCE_LANGUAGE_LOGGED = True
    return detected


def _get_glossary_source_language_from_metadata():
    global GLOSSARY_SOURCE_LANGUAGE, GLOSSARY_SOURCE_LANGUAGE_PATH
    global _GLOSSARY_SOURCE_LANGUAGE_LOADED, _GLOSSARY_SOURCE_LANGUAGE_LOGGED

    env_lang = _normalize_detected_language(os.getenv("GLOSSARY_SOURCE_LANGUAGE") or os.getenv("SOURCE_LANGUAGE"))
    if env_lang:
        GLOSSARY_SOURCE_LANGUAGE = env_lang
        GLOSSARY_SOURCE_LANGUAGE_PATH = None
        _GLOSSARY_SOURCE_LANGUAGE_LOADED = True

    if not _GLOSSARY_SOURCE_LANGUAGE_LOADED:
        _set_glossary_source_language_from_metadata(
            os.getenv("OUTPUT_PATH") or None,
            os.getenv("EPUB_PATH") or None,
            log=False,
        )

    if GLOSSARY_SOURCE_LANGUAGE and not _GLOSSARY_SOURCE_LANGUAGE_LOGGED:
        status = "CJK source confirmed" if _metadata_language_is_cjk(GLOSSARY_SOURCE_LANGUAGE) else "non-CJK source, filter skipped"
        source = "metadata.json" if GLOSSARY_SOURCE_LANGUAGE_PATH else "environment"
        print(f"🔍 [CJK Filter] Source language from {source}: {GLOSSARY_SOURCE_LANGUAGE} -> {status}")
        _GLOSSARY_SOURCE_LANGUAGE_LOGGED = True

    return GLOSSARY_SOURCE_LANGUAGE


def _extract_raw_title_from_epub(epub_path: str) -> str:
    """Extract the raw untranslated title from the input EPUB."""
    if not epub_path or not os.path.exists(epub_path):
        return None
    
    # Skip for non-EPUB files (txt, pdf, etc.)
    if not epub_path.lower().endswith('.epub'):
        return None
    
    print(f"[Metadata] Checking input EPUB for raw title: {epub_path}")
    
    # Try manual parsing first (more robust)
    try:
        import zipfile
        from bs4 import BeautifulSoup
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Find opf
            opf_name = next((n for n in zf.namelist() if n.lower().endswith('.opf')), None)
            if opf_name:
                content = zf.read(opf_name).decode('utf-8', errors='ignore')
                # Use BS4 with xml parser
                try:
                    soup = BeautifulSoup(content, 'xml')
                except Exception:
                    soup = BeautifulSoup(content, 'html.parser')
                    
                # Try dc:title
                title_tag = soup.find('dc:title')
                if not title_tag:
                    # Fallback to any title tag
                    title_tag = soup.find('title')
                
                if title_tag:
                    val = title_tag.get_text(strip=True)
                    if val:
                        return val
    except Exception as e:
        print(f"[Warning] Manual EPUB title extraction failed: {e}")

    # Fallback: ebooklib
    try:
        from ebooklib import epub
        book = epub.read_epub(epub_path)
        titles = book.get_metadata("DC", "title")
        if titles:
            val = titles[0][0]
            if val:
                return str(val).strip()
    except Exception as e:
        print(f"[Warning] Could not read EPUB metadata via ebooklib: {e}")
        
    return None

def _extract_translated_title_from_metadata(output_path: str, epub_path: str) -> str:
    """Extract translated title from metadata.json in output directory."""
    for meta_path in _unique_metadata_candidates(output_path, epub_path):
        # print(f"[Metadata] Checking for translated book title at: {meta_path}")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta_title = _extract_title_from_metadata(meta)
                if meta_title:
                    return meta_title.strip()
            except Exception as e:
                print(f"[Warning] Could not read metadata.json for book title: {e}")
                
    return None


def _ensure_book_title_entry(glossary: List[Dict], context=None) -> List[Dict]:
    """Insert a 'book' entry (raw + translated title) at the top if enabled and not present."""
    global BOOK_TITLE_PRESENT, BOOK_TITLE_VALUE, BOOK_TITLE_RAW, BOOK_TITLE_TRANSLATED
    
    include = os.getenv("GLOSSARY_INCLUDE_BOOK_TITLE", "1").lower() not in ("0", "false", "no")
    
    # Determine titles to use
    # Prefer specific raw/translated values if available
    if isinstance(context, GlossaryProgressContext):
        raw_title = context.book_title_raw
        trans_title = context.book_title_translated
    else:
        raw_title = BOOK_TITLE_RAW
        trans_title = BOOK_TITLE_TRANSLATED
    
    # Fallback logic if one is missing
    if not raw_title and trans_title:
        raw_title = trans_title
    if not trans_title and raw_title:
        trans_title = raw_title
        
    if not include or not raw_title:
        return glossary

    norm_raw = raw_title.lower() if raw_title else ""
    norm_trans = trans_title.lower() if trans_title else ""
    
    for entry in glossary:
        raw = str(entry.get("raw_name", "")).strip().lower()
        trans = str(entry.get("translated_name", "")).strip().lower()
        # Check against both raw and translated to avoid duplicates
        if (raw == norm_raw or trans == norm_trans or 
            raw == norm_trans or trans == norm_raw):
            if isinstance(context, GlossaryProgressContext):
                context.book_title_present = True
                context.book_title_value = entry.get("translated_name") or entry.get("raw_name")
            else:
                BOOK_TITLE_PRESENT = True
                BOOK_TITLE_VALUE = entry.get("translated_name") or entry.get("raw_name")
            return glossary  # Already present

    book_entry = {
        "type": "book",
        "raw_name": raw_title,
        "translated_name": trans_title,
        "gender": ""
    }
    glossary.insert(0, book_entry)
    if isinstance(context, GlossaryProgressContext):
        context.book_title_present = True
        context.book_title_value = trans_title or raw_title
    else:
        BOOK_TITLE_PRESENT = True
        BOOK_TITLE_VALUE = trans_title or raw_title
    return glossary

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = bool(value)
    
    # Keep the extractor and UnifiedClient cancellation lifecycles symmetric.
    # Previously the True path set unified_api_client.global_stop_flag, while
    # the False path only cleared UnifiedClient._global_cancelled. The module
    # flag therefore survived into the next glossary run and every request was
    # rejected as "Operation cancelled by user".
    try:
        import unified_api_client
        if hasattr(unified_api_client, 'set_stop_flag'):
            unified_api_client.set_stop_flag(bool(value))
        elif hasattr(unified_api_client, 'UnifiedClient'):
            unified_api_client.UnifiedClient._global_cancelled = bool(value)
    except Exception:
        pass

    # When clearing the stop flag, also clear the shared environment variable.
    if not value:
        os.environ['TRANSLATION_CANCELLED'] = '0'

def is_stop_requested():
    """Check if stop was requested"""
    global _stop_requested
    return _stop_requested

# ─── resilient tokenizer setup ───
try:
    enc = tiktoken.encoding_for_model(MODEL)
except Exception:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

def count_tokens(text: str) -> int:
    if enc:
        return len(enc.encode(text))
    # crude fallback: assume ~1 token per 4 chars
    return max(1, len(text) // 4)

from ebooklib import epub
from bs4 import BeautifulSoup
from unified_api_client import UnifiedClient
from history_manager import HistoryManager
from typing import List, Dict
import re

PROGRESS_FILE = "glossary_progress.json"
_GLOSSARY_PROGRESS_SESSION_ID = f"{os.getpid()}-{time.time_ns()}"
_GLOSSARY_QA_ISSUES_FOUND = {}
_GLOSSARY_CHAPTER_POSITIONS = {}
_GLOSSARY_CHAPTER_NUMBERS = {}
_GLOSSARY_CHAPTER_FILENAMES = {}
_GLOSSARY_TOTAL_CHAPTERS = 0
_GLOSSARY_OUTPUT_FILE = ""

def _current_glossary_model_name(existing_info=None, *, prefer_thread=False):
    existing_model = ""
    if isinstance(existing_info, dict):
        existing_model = str(existing_info.get("model_name") or existing_info.get("model") or "").strip()
    thread_model = ""
    try:
        from unified_api_client import get_current_thread_actual_request_model
        thread_model = str(get_current_thread_actual_request_model() or "").strip()
    except Exception:
        thread_model = ""
    if prefer_thread and thread_model:
        return thread_model
    return existing_model

def _glossary_key_pool_from_identifier(key_identifier: str) -> str:
    key_identifier = str(key_identifier or "").strip()
    pool_prefixes = (
        ("GlossaryRefinementKey#", "glossary_refinement"),
        ("GlossaryKey#", "glossary"),
        ("MetadataKey#", "metadata"),
        ("VisionKey#", "vision"),
        ("TruncationRetryKey#", "truncation_retry"),
        ("AITruncationDetectionKey#", "ai_truncation_detection"),
        ("ImageGenEditKey#", "inpainter"),
        ("Key#", "multi"),
        ("FALLBACK KEY", "fallback"),
        ("Main Key", "main"),
        ("Single Key", "single"),
    )
    for prefix, pool_name in pool_prefixes:
        if key_identifier.startswith(prefix):
            return pool_name
    return ""

def _current_glossary_key_context(existing_info=None, *, prefer_thread=False):
    existing_identifier = ""
    existing_pool = ""
    if isinstance(existing_info, dict):
        existing_identifier = str(
            existing_info.get("key_identifier")
            or existing_info.get("api_key_context")
            or existing_info.get("key_context")
            or ""
        ).strip()
        existing_pool = str(existing_info.get("key_pool") or "").strip()

    thread_identifier = ""
    try:
        from unified_api_client import get_current_thread_actual_request_key_identifier
        thread_identifier = str(get_current_thread_actual_request_key_identifier() or "").strip()
    except Exception:
        thread_identifier = ""

    key_identifier = thread_identifier if prefer_thread and thread_identifier else existing_identifier
    key_pool = _glossary_key_pool_from_identifier(key_identifier) or existing_pool
    return key_identifier, key_pool

class GlossaryProgressContext:
    """Per-run glossary progress state.

    Keep file paths and chapter row metadata on this object so parallel
    glossary callers do not have to mutate module-level progress globals.
    """
    def __init__(
        self,
        progress_file=None,
        output_file="",
        chapter_positions=None,
        chapter_numbers=None,
        chapter_filenames=None,
        total_chapters=0,
        book_title_raw=None,
        book_title_translated=None,
        book_title_present=False,
        book_title_value=None,
    ):
        self.progress_file = progress_file
        self.output_file = output_file or ""
        self.chapter_positions = {int(k): int(v) for k, v in (chapter_positions or {}).items()}
        self.chapter_numbers = {int(k): int(v) for k, v in (chapter_numbers or {}).items()}
        self.chapter_filenames = {
            int(k): os.path.basename(str(v or ""))
            for k, v in (chapter_filenames or {}).items()
        }
        self.total_chapters = int(total_chapters or 0)
        self.book_title_raw = book_title_raw
        self.book_title_translated = book_title_translated
        self.book_title_present = bool(book_title_present)
        self.book_title_value = book_title_value

def make_glossary_progress_context(**kwargs):
    return GlossaryProgressContext(**kwargs)

def _progress_context_values(context=None):
    if isinstance(context, GlossaryProgressContext):
        return (
            context.progress_file,
            context.output_file,
            context.chapter_positions,
            context.chapter_numbers,
            context.chapter_filenames,
            context.total_chapters,
        )
    return (
        PROGRESS_FILE,
        _GLOSSARY_OUTPUT_FILE,
        _GLOSSARY_CHAPTER_POSITIONS,
        _GLOSSARY_CHAPTER_NUMBERS,
        _GLOSSARY_CHAPTER_FILENAMES,
        _GLOSSARY_TOTAL_CHAPTERS,
    )

def _resolved_glossary_progress_file(context=None) -> str:
    """Return the concrete progress path, never the bare cwd default."""
    progress_file, output_file, _positions, _numbers, _filenames, _total = _progress_context_values(context)
    progress_file = str(progress_file or "").strip()
    if progress_file and (
        os.path.isabs(progress_file)
        or os.path.basename(progress_file).lower() != "glossary_progress.json"
    ):
        return progress_file

    output_file = str(output_file or "").strip()
    if output_file:
        source_path = os.getenv("EPUB_PATH", "").strip()
        try:
            _mac_cwd_unusable = (
                sys.platform == "darwin"
                and (
                    os.path.abspath(os.getcwd()) == os.path.abspath(os.sep)
                    or not os.access(os.getcwd(), os.W_OK)
                )
            )
        except Exception:
            _mac_cwd_unusable = sys.platform == "darwin"
        if not os.path.isabs(output_file) and _mac_cwd_unusable:
            source_dir = os.path.dirname(os.path.abspath(source_path)) if source_path else ""
            output_file = os.path.join(source_dir or resolve_shared_glossary_dir(), output_file)
        elif os.path.isabs(output_file) and sys.platform == "darwin":
            try:
                root_glossary = os.path.abspath(os.path.join(os.path.abspath(os.sep), "Glossary"))
                output_abs = os.path.abspath(output_file)
                if os.path.commonpath([root_glossary, output_abs]) == root_glossary:
                    source_dir = os.path.dirname(os.path.abspath(source_path)) if source_path else ""
                    rel_output = os.path.relpath(output_abs, root_glossary)
                    output_file = os.path.join(source_dir or resolve_shared_glossary_dir(), "Glossary", rel_output)
            except Exception:
                pass
        glossary_dir = os.path.dirname(os.path.abspath(output_file))
        base = os.path.splitext(os.path.basename(output_file))[0]
        if base.lower().endswith("_glossary"):
            base = base[:-len("_glossary")]
        if os.path.basename(glossary_dir).lower() == "glossary":
            migrate_all_legacy_glossary_files(glossary_dir, logger=print)
            glossary_dir = get_book_glossary_dir(glossary_dir, base)
    else:
        glossary_dir = os.getenv("GLOSSARY_SHARED_DIR", "").strip()
        if not glossary_dir:
            glossary_dir = resolve_shared_glossary_dir(fallback_base=os.getenv("EPUB_PATH", "").strip())
        source_path = os.getenv("EPUB_PATH", "").strip()
        base = os.path.splitext(os.path.basename(source_path))[0] if source_path else "book"
        glossary_dir = get_book_glossary_dir(glossary_dir, base, fallback_base=source_path)

    os.makedirs(glossary_dir, exist_ok=True)
    return os.path.join(glossary_dir, f"{base or 'book'}_glossary_progress.json")

def _unique_int_list(values):
    """Return ints in first-seen order, ignoring values that cannot be parsed."""
    seen = set()
    result = []
    for value in values or []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result

def _glossary_chapter_actual_num(idx: int, context=None) -> int:
    """Return the chapter number used by translation progress for a glossary row."""
    _progress_file, _output_file, positions, numbers, _filenames, _total = _progress_context_values(context)
    try:
        idx_int = int(idx)
        return int(numbers.get(
            idx_int,
            positions.get(idx_int, idx_int + 1)
        ))
    except (TypeError, ValueError):
        return int(idx) + 1

def _glossary_chapter_display_total(total_chapters=None, context=None) -> int:
    """Return the visible chapter total from the same numbering used by progress rows."""
    _progress_file, _output_file, positions, numbers, _filenames, context_total = _progress_context_values(context)
    number_values = [value for value in _unique_int_list((numbers or {}).values()) if value > 0]
    if number_values:
        return max(number_values)

    position_values = [value for value in _unique_int_list((positions or {}).values()) if value > 0]
    if position_values:
        return max(position_values)

    try:
        return int(total_chapters or context_total or 0)
    except (TypeError, ValueError):
        return 0

def _glossary_chapter_log_label(chapter_num, total_chapters=None, context=None) -> str:
    """Return the bracketed chapter label used by glossary progress logs."""
    try:
        chapter_num = int(chapter_num)
    except (TypeError, ValueError):
        chapter_num = str(chapter_num)
    total_chapters = _glossary_chapter_display_total(total_chapters, context=context)
    if total_chapters > 0:
        return f"[Chapter {chapter_num}/{total_chapters}]"
    return f"[Chapter {chapter_num}]"

def _parse_glossary_chapter_range(value):
    """Return (start, end), accepting either 'N' or 'N-M'."""
    value = str(value or "").strip()
    if re.match(r"^\d+$", value):
        num = int(value)
        return num, num
    if re.match(r"^\d+\s*-\s*\d+$", value):
        return tuple(map(int, re.split(r"\s*-\s*", value, 1)))
    return None

def _glossary_chapter_key(idx: int) -> str:
    """Build a stable zero-based progress key.

    Display chapter numbers can collide for cover/info files and episode 1, so
    the persisted key must be the internal spine index. The human-facing number
    remains available on each chapter entry as actual_num/chapter_num.
    """
    try:
        return str(int(idx))
    except (TypeError, ValueError):
        return str(idx)

def _glossary_chapter_output_file(idx: int, context=None) -> str:
    """Return the stable filename anchor used by the GUI progress manager."""
    _progress_file, _output_file, _positions, _numbers, filenames, _total = _progress_context_values(context)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return ""
    return _glossary_progress_filename(filenames.get(idx, ""))

def _glossary_progress_filename(value) -> str:
    """Return a chapter filename for progress, rejecting source-book paths."""
    name = os.path.basename(str(value or "").strip())
    if not name:
        return ""
    source_book = os.path.basename(str(os.getenv("EPUB_PATH", "") or "")).lower()
    if source_book and name.lower() == source_book:
        return ""
    if os.path.splitext(name)[1].lower() in (".epub", ".pdf", ".zip", ".cbz"):
        return ""
    return name

def _normalize_glossary_qa_issues(value=None, chapters=None):
    """Normalize glossary QA issue storage to {chapter_index: [issue, ...]}."""
    normalized = {}

    def _add(idx, issues):
        try:
            key = int(idx)
        except (TypeError, ValueError):
            return
        if isinstance(issues, str):
            issues = [issues]
        if not isinstance(issues, list):
            return
        bucket = normalized.setdefault(key, [])
        for issue in issues:
            issue_text = str(issue).strip()
            if issue_text and issue_text not in bucket:
                bucket.append(issue_text)

    if isinstance(value, dict):
        for idx, issues in value.items():
            if isinstance(issues, dict):
                issues = issues.get("qa_issues_found") or issues.get("issues") or []
            _add(idx, issues)

    if isinstance(chapters, dict):
        for key, info in chapters.items():
            if not isinstance(info, dict):
                continue
            idx = info.get("chapter_index", key)
            issues = info.get("qa_issues_found") or []
            _add(idx, issues)

    return normalized

def _glossary_progress_entry_index(info, key=None):
    """Return the internal zero-based chapter index stored in a progress entry."""
    if isinstance(info, dict):
        for idx_key in ("chapter_index", "idx", "index"):
            try:
                return int(info.get(idx_key))
            except (TypeError, ValueError):
                pass
    try:
        return int(key)
    except (TypeError, ValueError):
        return None

def _mark_glossary_failed(failed, idx, issues=None):
    """Mark a glossary chapter failed and optionally attach QA-style issue codes."""
    global _GLOSSARY_QA_ISSUES_FOUND
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return
    if idx not in failed:
        failed.append(idx)
    if issues:
        if isinstance(issues, str):
            issues = [issues]
        bucket = _GLOSSARY_QA_ISSUES_FOUND.setdefault(idx, [])
        for issue in issues:
            issue_text = str(issue).strip()
            if issue_text and issue_text not in bucket:
                bucket.append(issue_text)

def _glossary_issue_from_finish_reason(finish_reason, default_issue="EMPTY_OUTPUT"):
    finish_text = str(finish_reason or "").strip().lower()
    if finish_text in ("length", "max_tokens") or "max_tokens" in finish_text:
        return "TRUNCATED"
    return default_issue


def _confirmed_merged_child_indices(group_result, submitted_indices):
    """Return merged children only after this exact group succeeded completely."""
    try:
        expected = [int(idx) for idx in submitted_indices]
    except (TypeError, ValueError):
        return []
    if len(expected) < 2 or len(set(expected)) != len(expected):
        return []
    if not isinstance(group_result, dict):
        return []

    try:
        claimed_children = [int(idx) for idx in group_result.get("merged_indices", [])]
    except (TypeError, ValueError):
        return []
    if claimed_children != expected[1:]:
        return []

    results = group_result.get("results")
    if not isinstance(results, list) or len(results) != len(expected):
        return []

    results_by_idx = {}
    for result in results:
        if not isinstance(result, dict) or result.get("error"):
            return []
        try:
            idx = int(result.get("idx"))
        except (TypeError, ValueError):
            return []
        if idx in results_by_idx:
            return []
        results_by_idx[idx] = result
    if set(results_by_idx) != set(expected):
        return []

    parent_idx = expected[0]
    parent = results_by_idx[parent_idx]
    parent_data = parent.get("data", [])
    parent_response = str(parent.get("resp") or "").strip()
    if not parent_data and (not parent_response or parent_response in ("[]", "{}")):
        return []
    if _glossary_issue_from_finish_reason(parent.get("finish_reason", "stop"), None):
        return []

    for child_idx in expected[1:]:
        try:
            merged_into = int(results_by_idx[child_idx].get("merged_into"))
        except (TypeError, ValueError):
            return []
        if merged_into != parent_idx:
            return []

    return expected[1:]

def _glossary_restore_in_progress_entry(info):
    """Restore the pre-in-progress chapter entry, or return None for not completed."""
    if not isinstance(info, dict):
        return None
    previous_status = str(info.get("previous_status", "") or "").lower()
    previous_entry = info.get("previous_progress_entry")
    if isinstance(previous_entry, dict):
        restored = dict(previous_entry)
        restored_status = str(restored.get("status", previous_status) or previous_status).lower()
        if restored_status and restored_status not in ("in_progress", "not_completed", "not translated", "not_translated"):
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            return restored

    if previous_status in ("qa_failed", "failed", "error", "pending", "merged", "completed"):
        restored = dict(info)
        restored["status"] = "failed" if previous_status == "error" else previous_status
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if info.get("previous_status_unknown"):
        restored = dict(info)
        restored["status"] = "failed"
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if previous_status in ("not_completed", "not translated", "not_translated", ""):
        # Explicit not-completed snapshots are temporary markers; removing
        # in-progress should delete the row rather than invent a real status.
        if previous_status:
            return None
        # Legacy in-progress entries did not record previous state. If there is
        # an output anchor, fall back to failed so the row is not silently lost.
        if info.get("output_file"):
            restored = dict(info)
            restored["status"] = "failed"
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            restored.pop("previous_status_unknown", None)
            return restored
    return None


def _restore_glossary_in_progress_file(context=None, indices=None):
    """Atomically restore active glossary rows to their pre-run state.

    ``previous_progress_entry`` is intentionally stored in each in-progress
    row so a cancelled run can put the exact prior row back.  Do that directly
    in the progress file instead of relying on a later general save, which may
    preserve an abandoned in-progress row when the stop branch returns early.
    ``indices=None`` restores every active row for the current book.
    """
    progress_file = _resolved_glossary_progress_file(context)
    selected = None if indices is None else set(_unique_int_list(indices))

    try:
        with _progress_lock, _locked_glossary_progress_file(progress_file):
            if not os.path.exists(progress_file):
                return None
            with open(progress_file, "r", encoding="utf-8") as progress_f:
                progress_data = json.load(progress_f)
            if not isinstance(progress_data, dict):
                return None

            chapters = progress_data.get("chapters", {})
            if not isinstance(chapters, dict):
                return None

            changed = False
            for chapter_key, info in list(chapters.items()):
                if not isinstance(info, dict) or str(info.get("status", "")).lower() != "in_progress":
                    continue
                idx = _glossary_progress_entry_index(info, chapter_key)
                if idx is None or (selected is not None and idx not in selected):
                    continue
                restored = _glossary_restore_in_progress_entry(info)
                if restored is None:
                    chapters.pop(chapter_key, None)
                else:
                    chapters[chapter_key] = restored
                changed = True

            if not changed:
                return progress_data

            completed = []
            failed = []
            merged = []
            in_progress = []
            for chapter_key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                idx = _glossary_progress_entry_index(info, chapter_key)
                if idx is None:
                    continue
                status = str(info.get("status", "")).lower()
                if status in ("failed", "qa_failed", "error"):
                    failed.append(idx)
                elif status == "in_progress":
                    in_progress.append(idx)
                elif status == "merged":
                    merged.append(idx)
                    completed.append(idx)
                elif status == "completed":
                    completed.append(idx)

            progress_data["chapters"] = chapters
            progress_data["completed"] = _unique_int_list(completed)
            progress_data["failed"] = _unique_int_list(failed)
            progress_data["merged_indices"] = _unique_int_list(merged)
            progress_data["in_progress"] = _unique_int_list(in_progress)
            progress_data["qa_issues_found"] = {
                str(idx): issues
                for idx, issues in _normalize_glossary_qa_issues(
                    progress_data.get("qa_issues_found"), chapters
                ).items()
                if int(idx) in set(progress_data["failed"])
            }

            progress_dir = os.path.dirname(progress_file) or "."
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=progress_dir,
                delete=False,
                suffix=".tmp",
            ) as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())
            try:
                _atomic_replace_file(temp_path, progress_file)
            except Exception:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise

            _GLOSSARY_DISK_IN_PROGRESS_CACHE.clear()
            return progress_data
    except Exception as exc:
        print(f"⚠️ Could not restore glossary progress after stop: {exc}")
        return None

def _glossary_failed_from_in_progress_entry(info):
    """Convert a glossary in-progress entry to failed without restoring previous state."""
    if not isinstance(info, dict):
        info = {}
    failed = dict(info)
    failed["status"] = "failed"
    failed.pop("previous_status", None)
    failed.pop("previous_progress_entry", None)
    failed.pop("previous_status_unknown", None)
    return failed

_GLOSSARY_DISK_IN_PROGRESS_CACHE = {}

def _glossary_disk_in_progress_snapshot(context=None):
    """Return a cached set of on-disk in-progress indices, or None if the file is gone/unreadable."""
    try:
        progress_file = _resolved_glossary_progress_file(context)
        if not os.path.exists(progress_file):
            return None
        stat = os.stat(progress_file)
        cache_key = (progress_file, stat.st_mtime_ns, stat.st_size)
        cached = _GLOSSARY_DISK_IN_PROGRESS_CACHE.get("snapshot")
        if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
            return cached.get("indices", set())
        with open(progress_file, "r", encoding="utf-8") as f:
            disk_progress = json.load(f)
        if not isinstance(disk_progress, dict):
            return None
        indices = set()
        chapters = disk_progress.get("chapters", {})
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                entry_idx = _glossary_progress_entry_index(info, key)
                if entry_idx is not None and str(info.get("status", "")).lower() == "in_progress":
                    indices.add(int(entry_idx))
        indices.update(_unique_int_list(disk_progress.get("in_progress", [])))
        _GLOSSARY_DISK_IN_PROGRESS_CACHE["snapshot"] = {"cache_key": cache_key, "indices": indices}
        return indices
    except Exception:
        return None

def _glossary_disk_entry_is_still_in_progress(idx, context=None):
    """False means the user deleted or changed the in-progress row on disk."""
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return False
    snapshot = _glossary_disk_in_progress_snapshot(context)
    return bool(snapshot is not None and idx in snapshot)

def _glossary_is_graceful_stop_active():
    return os.environ.get('GRACEFUL_STOP') == '1' or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1'

def _glossary_is_hard_stop_env_active():
    return os.environ.get('TRANSLATION_CANCELLED') == '1' and not _glossary_is_graceful_stop_active()

def _glossary_is_hard_stop_requested(stop_callback=None):
    # First-click graceful stop leaves in-flight requests alive. Second-click
    # force stop clears GRACEFUL_STOP before setting the hard-stop signals.
    if _glossary_is_graceful_stop_active():
        return False
    if os.environ.get('TRANSLATION_CANCELLED') == '1':
        return True
    try:
        if stop_callback and stop_callback():
            return True
    except Exception:
        return False
    try:
        return bool(is_stop_requested())
    except Exception:
        return False

def remove_honorifics(name):
    """Remove common honorifics from names"""
    if not name:
        return name
    
    # Check if honorifics filtering is disabled
    if os.getenv('GLOSSARY_DISABLE_HONORIFICS_FILTER', '0') == '1':
        return name.strip()
    
    # Modern Korean honorifics
    korean_honorifics = [
        '님', '씨', '씨는', '군', '양', '선생님', '선생', '사장님', '사장', 
        '과장님', '과장', '대리님', '대리', '주임님', '주임', '이사님', '이사',
        '부장님', '부장', '차장님', '차장', '팀장님', '팀장', '실장님', '실장',
        '교수님', '교수', '박사님', '박사', '원장님', '원장', '회장님', '회장',
        '소장님', '소장', '전무님', '전무', '상무님', '상무', '이사장님', '이사장'
    ]
    
    # Archaic/Historical Korean honorifics
    korean_archaic = [
        '공', '옹', '어른', '나리', '나으리', '대감', '영감', '마님', '마마',
        '대군', '군', '옹주', '공주', '왕자', '세자', '영애', '영식', '도령',
        '낭자', '낭군', '서방', '영감님', '대감님', '마님', '아씨', '도련님',
        '아가씨', '나으리', '진사', '첨지', '영의정', '좌의정', '우의정',
        '판서', '참판', '정승', '대원군'
    ]
    
    # Modern Japanese honorifics
    japanese_honorifics = [
        'さん', 'さま', '様', 'くん', '君', 'ちゃん', 'せんせい', '先生',
        'どの', '殿', 'たん', 'ぴょん', 'ぽん', 'ちん', 'りん', 'せんぱい',
        '先輩', 'こうはい', '後輩', 'し', '氏', 'ふじん', '夫人', 'かちょう',
        '課長', 'ぶちょう', '部長', 'しゃちょう', '社長'
    ]
    
    # Archaic/Historical Japanese honorifics
    japanese_archaic = [
        'どの', '殿', 'たいゆう', '大夫', 'きみ', '公', 'あそん', '朝臣',
        'おみ', '臣', 'むらじ', '連', 'みこと', '命', '尊', 'ひめ', '姫',
        'みや', '宮', 'おう', '王', 'こう', '侯', 'はく', '伯', 'し', '子',
        'だん', '男', 'じょ', '女', 'ひこ', '彦', 'ひめみこ', '姫御子',
        'すめらみこと', '天皇', 'きさき', '后', 'みかど', '帝'
    ]
    
    # Modern Chinese honorifics
    chinese_honorifics = [
        '先生', '女士', '小姐', '老师', '师傅', '大人', '公', '君', '总',
        '老总', '老板', '经理', '主任', '处长', '科长', '股长', '教授',
        '博士', '院长', '校长', '同志', '师兄', '师姐', '师弟', '师妹',
        '学长', '学姐', '前辈', '阁下'
    ]
    
    # Archaic/Historical Chinese honorifics
    chinese_archaic = [
        '公', '侯', '伯', '子', '男', '王', '君', '卿', '大夫', '士',
        '陛下', '殿下', '阁下', '爷', '老爷', '大人', '夫人', '娘娘',
        '公子', '公主', '郡主', '世子', '太子', '皇上', '皇后', '贵妃',
        '娘子', '相公', '官人', '郎君', '小姐', '姑娘', '公公', '嬷嬷',
        '大侠', '少侠', '前辈', '晚辈', '在下', '足下', '兄台', '仁兄',
        '贤弟', '老夫', '老朽', '本座', '本尊', '真人', '上人', '尊者'
    ]
    
    # Combine all honorifics
    all_honorifics = (
        korean_honorifics + korean_archaic +
        japanese_honorifics + japanese_archaic +
        chinese_honorifics + chinese_archaic
    )
    
    # Remove honorifics from the end of the name
    name_cleaned = name.strip()
    
    # Sort by length (longest first) to avoid partial matches
    sorted_honorifics = sorted(all_honorifics, key=len, reverse=True)
    
    for honorific in sorted_honorifics:
        if name_cleaned.endswith(honorific):
            remainder = name_cleaned[:-len(honorific)].strip()
            # Guard: for single-character honorifics (子, 公, 王, 君, etc.),
            # require the remaining name to be at least 2 characters.
            # This prevents mangling real CJK names like 花子 → 花, 公子 → 公.
            # Multi-character honorifics (선생님, さん, etc.) are unambiguous.
            if len(honorific) == 1 and len(remainder) < 2:
                continue
            name_cleaned = remainder
            # Only remove one honorific per pass
            break
    
    return name_cleaned

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        import sys
        import io
        import threading
        
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                self.buffer = ""
                self.main_thread = threading.main_thread()
                
            def write(self, text):
                _capture_direct_text_stream_write(text)
                if text.strip():
                    # The callback (append_log) is already thread-safe - it handles QTimer internally
                    # So we can call it directly from any thread
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

def load_config(path: str) -> Dict:
    # Gracefully handle missing config file (e.g. when running from Gradio web UI)
    # Instead of crashing, create a sensible default config from environment variables
    if not path or not os.path.exists(path):
        print(f"[Info] Config file not found at '{path}', using environment variables and defaults")
        cfg = {
            'api_key': os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY', ''),
            'model': os.getenv('MODEL', 'gemini-2.0-flash'),
            'temperature': 0.1,
            'max_tokens': 65536,
            'context_limit_chapters': 3,
        }
    else:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

    # override context_limit_chapters if GUI passed GLOSSARY_CONTEXT_LIMIT
    env_limit = os.getenv("GLOSSARY_CONTEXT_LIMIT")
    if env_limit is not None:
        try:
            cfg['context_limit_chapters'] = int(env_limit)
        except ValueError:
            pass  # keep existing config value on parse error

    # override temperature if GUI passed GLOSSARY_TEMPERATURE
    env_temp = os.getenv("GLOSSARY_TEMPERATURE")
    if env_temp is not None:
        try:
            cfg['temperature'] = float(env_temp)
        except ValueError:
            pass  # keep existing config value on parse error

    return cfg

def get_custom_entry_types():
    """Get custom entry types configuration from environment"""
    try:
        types_json = os.getenv('GLOSSARY_CUSTOM_ENTRY_TYPES', '{}')
        result = json.loads(types_json)
        # If empty, return defaults
        if not result:
            return {
                'character': {'enabled': True, 'has_gender': True},
                'term': {'enabled': True, 'has_gender': False},
                'surnames': {'enabled': True, 'has_gender': False},
                'titles': {'enabled': True, 'has_gender': True},
                'locations': {'enabled': True, 'has_gender': False},
                'nicknames': {'enabled': True, 'has_gender': True}
            }
        return result
    except:
        # Default configuration
        return {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False},
            'surnames': {'enabled': True, 'has_gender': False},
            'titles': {'enabled': True, 'has_gender': True},
            'locations': {'enabled': True, 'has_gender': False},
            'nicknames': {'enabled': True, 'has_gender': True}
        }

def _normalize_gender_value(value) -> str:
    gender = str(value or "").strip().lower()
    aliases = {
        "m": "male",
        "man": "male",
        "boy": "male",
        "masc": "male",
        "masculine": "male",
        "f": "female",
        "woman": "female",
        "girl": "female",
        "fem": "female",
        "feminine": "female",
    }
    return aliases.get(gender, gender)

def _gender_is_trackable(gender: str) -> bool:
    return bool(gender) and gender not in {"unknown", "n/a", "na", "none", "-"}

def _gender_is_dedupe_protected(gender: str) -> bool:
    return _normalize_gender_value(gender) in {"male", "female"}

def _entry_type_has_gender(entry: Dict) -> bool:
    try:
        custom_types = get_custom_entry_types()
        entry_type = str(entry.get("type", "character")).strip()
        return bool(custom_types.get(entry_type, {}).get("has_gender", False))
    except Exception:
        return str(entry.get("type", "")).strip().lower() == "character"

def _entry_type_is_active(entry: Dict) -> bool:
    try:
        custom_types = get_custom_entry_types()
        entry_type = str(entry.get("type", "character")).strip()
        cfg = custom_types.get(entry_type)
        if cfg is None:
            return entry_type.strip().lower() in {"character", "term", "terms"}
        return bool(cfg.get("enabled", True))
    except Exception:
        return True

def _entry_type_has_active_gender(entry: Dict) -> bool:
    return _entry_type_is_active(entry) and _entry_type_has_gender(entry)

def _entry_gender(entry: Dict) -> str:
    return _normalize_gender_value(entry.get("gender", ""))

def _raw_exact_key(raw_name) -> str:
    return str(raw_name or "").strip()

def _raw_tracker_key(raw_name) -> str:
    return _raw_exact_key(raw_name).casefold()

def _strip_private_glossary_keys(entry: Dict) -> Dict:
    if not isinstance(entry, dict):
        return entry
    return {k: v for k, v in entry.items() if not str(k).startswith("_gender_tracker_")}

def _gender_tracker_path_for_output(output_path: str) -> str:
    stem, _ext = os.path.splitext(output_path)
    if stem.endswith("_glossary"):
        stem = stem[:-len("_glossary")]
    elif os.path.basename(stem).lower() == "glossary":
        stem = os.path.join(os.path.dirname(stem), "gender")
    return f"{stem}_gender_tracker.json"

def _gender_tracking_disabled() -> bool:
    return os.getenv("GLOSSARY_SKIP_GENDER_TRACKING", "0").strip().lower() in ("1", "true", "yes", "on")

def _partial_ratio_gender_only() -> bool:
    return os.getenv("GLOSSARY_PARTIAL_RATIO_GENDER_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")

def _alias_aware_name_matching_enabled() -> bool:
    return os.getenv("GLOSSARY_ALIAS_AWARE_NAME_MATCHING", "0").strip().lower() in ("1", "true", "yes", "on")

def _alias_aware_gender_only() -> bool:
    return os.getenv("GLOSSARY_ALIAS_AWARE_GENDER_ONLY", "1").strip().lower() in ("1", "true", "yes", "on")

def _alias_entry_allowed(entry: Dict) -> bool:
    if _alias_aware_gender_only():
        return _entry_type_has_active_gender(entry)
    return _entry_type_is_active(entry)

def _load_gender_tracker(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"version": 1, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("version", 1)
            data.setdefault("entries", {})
            return data
    except Exception:
        pass
    return {"version": 1, "entries": {}}

def _tracker_int(value, fallback=10**9):
    try:
        if value is None or value == "":
            return fallback
        return int(value)
    except Exception:
        return fallback

def _tracker_occurrence_sort_key(occurrence: Dict):
    chapter_file = str(occurrence.get("chapter_file", "") or "")
    file_num_match = re.search(r"(\d+)", chapter_file)
    file_num = int(file_num_match.group(1)) if file_num_match else 10**9
    return (
        _tracker_int(occurrence.get("chapter_index")),
        _tracker_int(occurrence.get("chapter_num")),
        file_num,
        chapter_file,
        str(occurrence.get("gender", "") or ""),
    )

def _normalize_gender_tracker_order(tracker: Dict):
    if not isinstance(tracker, dict):
        return tracker
    entries = tracker.get("entries", {})
    if not isinstance(entries, dict):
        return tracker

    def first_occurrence_key(item):
        occurrences = item.get("occurrences", []) if isinstance(item, dict) else []
        occurrences = [o for o in occurrences if isinstance(o, dict)]
        return _tracker_occurrence_sort_key(occurrences[0]) if occurrences else (10**9, 10**9, 10**9, "", "")

    normalized_entries = []
    for key, item in entries.items():
        if not isinstance(item, dict):
            normalized_entries.append((key, item))
            continue
        occurrences = [o for o in item.get("occurrences", []) if isinstance(o, dict)]
        occurrences.sort(key=_tracker_occurrence_sort_key)
        item["occurrences"] = occurrences

        changes = []
        previous = None
        for occurrence in occurrences:
            if previous and previous.get("gender") != occurrence.get("gender"):
                changes.append({
                    "from": previous.get("gender"),
                    "to": occurrence.get("gender"),
                    "chapter_index": occurrence.get("chapter_index"),
                    "chapter_num": occurrence.get("chapter_num"),
                    "chapter_file": occurrence.get("chapter_file", ""),
                })
            previous = occurrence
        item["changes"] = changes

        genders = item.setdefault("genders", {})
        if isinstance(genders, dict):
            for gender, meta in genders.items():
                if not isinstance(meta, dict):
                    continue
                first = next((o for o in occurrences if o.get("gender") == gender), None)
                if first:
                    meta["first_seen_chapter"] = first.get("chapter_num")
                    meta["first_seen_file"] = first.get("chapter_file", "")

        normalized_entries.append((key, item))

    normalized_entries.sort(key=lambda pair: (first_occurrence_key(pair[1]), str(pair[0])))
    tracker["entries"] = {key: item for key, item in normalized_entries}
    return tracker

def _write_gender_tracker(path: str, tracker: Dict):
    if not path or not isinstance(tracker, dict):
        return
    try:
        output_dir = os.path.dirname(path) or "."
        os.makedirs(output_dir, exist_ok=True)
        tracker["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tracker = _normalize_gender_tracker_order(tracker)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False, suffix=".tmp") as temp_f:
            temp_path = temp_f.name
            json.dump(tracker, temp_f, ensure_ascii=False, indent=2)
            temp_f.flush()
            os.fsync(temp_f.fileno())
        _atomic_replace_file(temp_path, path)
    except Exception as e:
        print(f"⚠️ Could not write gender tracker: {e}")

def update_gender_tracker(entries: List[Dict], output_path: str, source_path: str = None,
                          chapter_index=None, chapter_num=None, chapter_file: str = None):
    """Record which chapter/file observed each gendered raw-name variant."""
    if _gender_tracking_disabled() or not entries or not output_path:
        return
    tracker_path = _gender_tracker_path_for_output(output_path)
    source_path = source_path or os.getenv("EPUB_PATH", "")
    with _gender_tracker_lock:
        tracker = _load_gender_tracker(tracker_path)
        tracker["source_path"] = source_path
        tracker["glossary_path"] = output_path
        tracker_entries = tracker.setdefault("entries", {})

        changed = False
        for entry in entries:
            if not isinstance(entry, dict) or not _entry_type_has_gender(entry):
                continue
            raw_name = _raw_exact_key(entry.get("raw_name"))
            gender = _entry_gender(entry)
            if not raw_name or not _gender_is_trackable(gender):
                continue

            key = _raw_tracker_key(raw_name)
            item = tracker_entries.setdefault(key, {
                "raw_name": raw_name,
                "translated_name": entry.get("translated_name", ""),
                "genders": {},
                "occurrences": [],
                "changes": [],
            })
            if entry.get("translated_name") and not item.get("translated_name"):
                item["translated_name"] = entry.get("translated_name", "")
            item.setdefault("genders", {}).setdefault(gender, {
                "first_seen_chapter": chapter_num,
                "first_seen_file": chapter_file or "",
            })

            occurrence = {
                "raw_name": raw_name,
                "gender": gender,
                "chapter_index": chapter_index,
                "chapter_num": chapter_num,
                "chapter_file": os.path.basename(str(chapter_file or "")),
                "source_path": source_path,
                "translated_name": entry.get("translated_name", ""),
            }
            sig = (
                occurrence["gender"],
                str(occurrence["chapter_num"]),
                occurrence["chapter_file"],
            )
            existing_sigs = {
                (o.get("gender"), str(o.get("chapter_num")), o.get("chapter_file", ""))
                for o in item.get("occurrences", [])
                if isinstance(o, dict)
            }
            if sig not in existing_sigs:
                previous = item["occurrences"][-1] if item.get("occurrences") else None
                item.setdefault("occurrences", []).append(occurrence)
                if previous and previous.get("gender") != gender:
                    item.setdefault("changes", []).append({
                        "from": previous.get("gender"),
                        "to": gender,
                        "chapter_num": chapter_num,
                        "chapter_file": occurrence["chapter_file"],
                    })
                changed = True

        if changed:
            _write_gender_tracker(tracker_path, tracker)

def sync_gender_tracker_with_glossary(glossary: List[Dict], output_path: str):
    """Keep tracker raw/translated names aligned with the final saved glossary."""
    if _gender_tracking_disabled() or not glossary or not output_path:
        return
    tracker_path = _gender_tracker_path_for_output(output_path)
    if not os.path.exists(tracker_path):
        return

    canonical_by_key = {}
    for entry in glossary:
        if not isinstance(entry, dict) or not _entry_type_has_gender(entry):
            continue
        raw_name = _raw_exact_key(entry.get("raw_name"))
        translated_name = str(entry.get("translated_name", "") or "").strip()
        if raw_name and translated_name:
            canonical_by_key.setdefault(_raw_tracker_key(raw_name), {
                "raw_name": raw_name,
                "translated_name": translated_name,
            })
    if not canonical_by_key:
        return

    with _gender_tracker_lock:
        tracker = _load_gender_tracker(tracker_path)
        tracker_entries = tracker.get("entries", {})
        changed = False
        for key, canonical in canonical_by_key.items():
            item = tracker_entries.get(key)
            if not isinstance(item, dict):
                continue
            if item.get("raw_name") != canonical["raw_name"]:
                item["raw_name"] = canonical["raw_name"]
                changed = True
            if item.get("translated_name") != canonical["translated_name"]:
                item["translated_name"] = canonical["translated_name"]
                changed = True
            for occurrence in item.get("occurrences", []):
                if not isinstance(occurrence, dict):
                    continue
                if occurrence.get("raw_name") != canonical["raw_name"]:
                    occurrence["raw_name"] = canonical["raw_name"]
                    changed = True
                if occurrence.get("translated_name") != canonical["translated_name"]:
                    occurrence["translated_name"] = canonical["translated_name"]
                    changed = True
        if changed:
            _write_gender_tracker(tracker_path, tracker)

def _align_gender_variant_translation(existing_entry: Dict, new_entry: Dict):
    canonical = existing_entry.get("translated_name") or new_entry.get("translated_name")
    if canonical:
        existing_entry["translated_name"] = canonical
        new_entry["translated_name"] = canonical

def _is_exact_raw_gender_variant(existing_entry: Dict, new_entry: Dict) -> bool:
    if _gender_tracking_disabled():
        return False
    if _raw_exact_key(existing_entry.get("raw_name")) != _raw_exact_key(new_entry.get("raw_name")):
        return False
    if not (_entry_type_has_gender(existing_entry) or _entry_type_has_gender(new_entry)):
        return False
    existing_gender = _entry_gender(existing_entry)
    new_gender = _entry_gender(new_entry)
    return (
        _gender_is_dedupe_protected(existing_gender)
        and _gender_is_dedupe_protected(new_gender)
        and existing_gender != new_gender
    )

def _alias_raw_key(value) -> str:
    text = unicodedata.normalize('NFC', remove_honorifics(str(value or ""))).strip().casefold()
    return re.sub(r"[\s\-_.'’·・・]+", "", text)

def _alias_min_len(key: str) -> int:
    return 4 if key and all(ord(ch) < 128 for ch in key) else 2

def _raw_alias_relation(existing_entry: Dict, new_entry: Dict):
    if not (_alias_aware_name_matching_enabled() and _alias_entry_allowed(existing_entry) and _alias_entry_allowed(new_entry)):
        return None
    existing_raw = _alias_raw_key(existing_entry.get("raw_name"))
    new_raw = _alias_raw_key(new_entry.get("raw_name"))
    if not existing_raw or not new_raw or existing_raw == new_raw:
        return None
    if len(existing_raw) < len(new_raw):
        short_entry, long_entry = existing_entry, new_entry
        short_key, long_key = existing_raw, new_raw
    else:
        short_entry, long_entry = new_entry, existing_entry
        short_key, long_key = new_raw, existing_raw
    if len(short_key) < _alias_min_len(short_key):
        return None
    if long_key.endswith(short_key):
        position = "suffix"
    elif long_key.startswith(short_key):
        position = "prefix"
    else:
        return None
    return short_entry, long_entry, position

def _translation_similarity(a: str, b: str) -> float:
    a_norm = re.sub(r"[^a-z0-9]+", "", str(a or "").casefold())
    b_norm = re.sub(r"[^a-z0-9]+", "", str(b or "").casefold())
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a_norm, b_norm) / 100.0
    except Exception:
        import difflib
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

def _alias_tail_translation(long_translation: str, position: str) -> str:
    text = str(long_translation or "").strip()
    parts = [p for p in re.split(r"\s+", text) if p]
    if len(parts) < 2:
        return ""
    if position == "suffix":
        return " ".join(parts[1:]).strip()
    if position == "prefix":
        return " ".join(parts[:-1]).strip()
    return ""

def _align_alias_variant_translation(existing_entry: Dict, new_entry: Dict) -> bool:
    relation = _raw_alias_relation(existing_entry, new_entry)
    if not relation:
        return False
    short_entry, long_entry, position = relation
    long_translation = str(long_entry.get("translated_name", "") or "").strip()
    short_translation = str(short_entry.get("translated_name", "") or "").strip()
    alias_translation = _alias_tail_translation(long_translation, position)
    if not alias_translation:
        return True
    if short_translation and _translation_similarity(short_translation, alias_translation) < 0.60:
        return True
    if short_entry.get("translated_name") == alias_translation:
        return True
    short_entry["translated_name"] = alias_translation
    return True

def _harmonize_alias_name_translations(glossary: List[Dict]) -> List[Dict]:
    if not _alias_aware_name_matching_enabled():
        return glossary
    entries = [e for e in glossary or [] if isinstance(e, dict) and _alias_entry_allowed(e)]
    for i, entry in enumerate(entries):
        for other in entries[i + 1:]:
            _align_alias_variant_translation(entry, other)
    return glossary

def _harmonize_gender_variant_translations(glossary: List[Dict]) -> List[Dict]:
    if _gender_tracking_disabled():
        return glossary
    groups = {}
    for entry in glossary or []:
        if not isinstance(entry, dict) or not _entry_type_has_gender(entry):
            continue
        raw_name = _raw_exact_key(entry.get("raw_name"))
        gender = _entry_gender(entry)
        if raw_name and _gender_is_dedupe_protected(gender):
            groups.setdefault(raw_name, []).append(entry)
    for entries in groups.values():
        genders = {_entry_gender(e) for e in entries}
        if len(genders) < 2:
            continue
        canonical = next((e.get("translated_name") for e in entries if e.get("translated_name")), "")
        if canonical:
            for entry in entries:
                entry["translated_name"] = canonical
    return glossary

def save_glossary_json(glossary: List[Dict], output_path: str):
    """Save glossary in the new simple format with automatic sorting by type"""
    # Check if legacy JSON output is enabled (default disabled)
    if os.getenv('GLOSSARY_OUTPUT_LEGACY_JSON', '0') != '1':
        return
    glossary = _harmonize_gender_variant_translations(_harmonize_alias_name_translations([
        _strip_private_glossary_keys(e) for e in _ensure_book_title_entry(glossary)
    ]))

    global _glossary_json_lock
    
    # Acquire lock to prevent concurrent writes
    with _glossary_json_lock:
        # Get custom types for sorting order
        custom_types = get_custom_entry_types()
        
        # Create sorting order: character=0, term=1, others alphabetically starting from 2
        type_order = {'character': 0, 'term': 1}
        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
        for i, t in enumerate(other_types):
            type_order[t] = i + 2
        
        # Sort glossary by type order, then by raw_name
        sorted_glossary = sorted(glossary, key=lambda x: (
            type_order.get(x.get('type', 'term'), 999),  # Unknown types go last
            x.get('raw_name', '').lower()
        ))
        
        # Use atomic write to prevent corruption during parallel saves
        try:
            # Create temp file in the same directory for atomic rename
            output_dir = os.path.dirname(output_path) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_dir, delete=False, suffix='.tmp') as temp_f:
                temp_path = temp_f.name
                json.dump(sorted_glossary, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())  # Force immediate disk write
            
            # Atomic replace with retry for Windows file locks
            _atomic_replace_file(temp_path, output_path)
        except Exception as e:
            print(f"[Warning] Atomic write failed for JSON: {e}. Attempting direct write...")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted_glossary, f, ensure_ascii=False, indent=2)
            except Exception as e2:
                print(f"[Error] Failed to save glossary JSON: {e2}")

def _mirror_glossary_outputs_to_backup(output_path: str):
    """Copy current glossary outputs to the optional output-side backup folder."""
    backup_dir = os.getenv("GLOSSARY_OUTPUT_BACKUP_DIR", "").strip()
    if not backup_dir:
        return
    try:
        output_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
        backup_abs = os.path.abspath(backup_dir)
        if os.path.normcase(output_dir) == os.path.normcase(backup_abs):
            return
        os.makedirs(backup_abs, exist_ok=True)
        for src_path in (output_path, output_path.replace('.json', '.csv')):
            if src_path and os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(backup_abs, os.path.basename(src_path)))
    except Exception as e:
        print(f"[Warning] Could not save glossary backup copies: {e}")

def save_glossary_csv(glossary: List[Dict], output_path: str):
    """Save glossary in CSV or token-efficient format based on environment variable"""
    global _glossary_csv_lock
    import csv
    
    with _glossary_csv_lock:
        csv_path = output_path.replace('.json', '.csv')
        # If a glossary already exists on disk, mark whether it already has the book entry
        if not BOOK_TITLE_PRESENT and os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    _mark_book_title_from_csv(f.read())
            except Exception:
                pass

        glossary = _harmonize_gender_variant_translations(_harmonize_alias_name_translations([
            _strip_private_glossary_keys(e) for e in _ensure_book_title_entry(glossary)
        ]))
        custom_types = get_custom_entry_types()
        type_order = {'book': -1, 'character': 0, 'term': 1}
        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
        for i, t in enumerate(other_types):
            type_order[t] = i + 2
        
        sorted_glossary = sorted(glossary, key=lambda x: (
            type_order.get(x.get('type', 'term'), 999),
            x.get('raw_name', '').lower()
        ))
        sync_gender_tracker_with_glossary(sorted_glossary, output_path)
        
        use_legacy_format = os.getenv('GLOSSARY_USE_LEGACY_CSV', '0') == '1'
        csv_dir = os.path.dirname(csv_path) or '.'
        
        try:
            if use_legacy_format:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=csv_dir, delete=False, newline='', suffix='.tmp') as temp_f:
                    temp_path = temp_f.name
                    writer = csv.writer(temp_f)
                    header = ['type', 'raw_name', 'translated_name', 'gender']
                    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                    try:
                        custom_fields = json.loads(custom_fields_json)
                        header.extend(custom_fields)
                    except:
                        custom_fields = []
                    # Custom Fields veto: only emit description column when
                    # the user has "description" in Custom Fields. Preserve
                    # their original casing (e.g. "Description") for the
                    # column header, and use case-insensitive lookup in the
                    # entry dicts.
                    legacy_desc_name = _find_description_field_casing(custom_fields)
                    include_description_legacy = legacy_desc_name is not None
                    if include_description_legacy and legacy_desc_name not in header:
                        header.append(legacy_desc_name)
                    writer.writerow(header)
                    for entry in sorted_glossary:
                        entry_type = entry.get('type', 'term')
                        type_config = custom_types.get(entry_type, {})
                        row = [entry_type, entry.get('raw_name', ''), entry.get('translated_name', '')]
                        # Always emit gender column to keep alignment with header;
                        # non-gender types get an empty string.
                        if type_config.get('has_gender', False):
                            row.append(entry.get('gender', ''))
                        else:
                            row.append('')
                        for field in custom_fields:
                            # Skip description here — handled by the explicit
                            # include_description_legacy block below to avoid duplication.
                            if isinstance(field, str) and field.strip().lower() == 'description':
                                continue
                            row.append(entry.get(field, ''))
                        if include_description_legacy:
                            _dv = ''
                            for _ek, _ev in entry.items():
                                if isinstance(_ek, str) and _ek.strip().lower() == 'description':
                                    _dv = str(_ev or '')
                                    break
                            row.append(_dv)
                        expected_fields = 4 + len(custom_fields) + (1 if include_description_legacy else 0)
                        while len(row) > expected_fields and row[-1] == '':
                            row.pop()
                        while len(row) < 3:
                            row.append('')
                        writer.writerow(row)
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Force immediate disk write
                
                _atomic_replace_file(temp_path, csv_path)
                print(f"✅ Saved legacy CSV format: {csv_path}")
            
            else:
                grouped_entries = {}
                for entry in sorted_glossary:
                    entry_type = entry.get('type', 'term')
                    if entry_type not in grouped_entries:
                        grouped_entries[entry_type] = []
                    grouped_entries[entry_type].append(entry)
                
                custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json)
                except:
                    custom_fields = []
                
                # Custom Fields is the authoritative source for whether description
                # should appear in the output at all. Removing "description" from
                # Custom Fields in the GUI must fully suppress the column even if
                # GLOSSARY_INCLUDE_DESCRIPTION=1 or if the AI leaked description
                # values into the entries (they'd already be stripped by the parser,
                # but this is a second guarantee).
                #
                # Preserve the user's configured casing (e.g. "Description" vs
                # "description") so the column header and dict lookups match
                # what they typed in the GUI.
                desc_field_name = _find_description_field_casing(custom_fields)
                include_description = desc_field_name is not None
                
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=csv_dir, delete=False, suffix='.tmp') as temp_f:
                    temp_path = temp_f.name
                    
                    # Write column header
                    column_headers = ['raw_name', 'translated_name']
                    # Add gender if any type supports it
                    has_gender = any(type_config.get('has_gender', False) for type_config in custom_types.values())
                    if has_gender:
                        column_headers.append('gender')
                    # Description always comes right after gender (rendered as main body text).
                    # Use the user's original casing — ``desc_field_name`` is exactly
                    # what they typed in Custom Fields (e.g. "Description").
                    if include_description:
                        column_headers.append(desc_field_name)
                    # Then remaining custom fields (rendered in parentheses)
                    for f in custom_fields:
                        if str(f).strip().lower() != 'description':
                            column_headers.append(f)
                    temp_f.write(f"Glossary Columns: {', '.join(column_headers)}\n\n")
                    for entry_type in sorted(grouped_entries.keys(), key=lambda x: type_order.get(x, 999)):
                        entries = grouped_entries[entry_type]
                        type_config = custom_types.get(entry_type, {})
                        section_name = entry_type.upper() + 'S' if not entry_type.upper().endswith('S') else entry_type.upper()
                        temp_f.write(f"=== {section_name} ===\n")
                        for entry in entries:
                            raw_name = entry.get('raw_name', '')
                            translated_name = entry.get('translated_name', '')
                            line = f"* {raw_name} = {translated_name}"
                            if type_config.get('has_gender', False):
                                gender = entry.get('gender', '')
                                if gender and gender != 'Unknown':
                                    line += f" [{gender}]"
                            # Gather custom field values
                            custom_field_parts = []
                            for field in custom_fields:
                                # Skip description — it's handled separately below
                                if str(field).strip().lower() == 'description':
                                    continue
                                value = entry.get(field, '').strip()
                                if value:
                                    if field.lower() in ['notes', 'details']:
                                        custom_field_parts.append(f"{field}: {value}")
                                    else:
                                        custom_field_parts.append(f"{field}: {value}")
                            # Write description (main body text after colon).
                            # Case-insensitive dict-key lookup so we find the
                            # value regardless of whether the AI returned it as
                            # "description", "Description", "DESCRIPTION", etc.
                            desc_value = ''
                            if include_description:
                                for _ek, _ev in entry.items():
                                    if isinstance(_ek, str) and _ek.strip().lower() == 'description':
                                        desc_value = str(_ev or '').strip()
                                        break
                            if desc_value:
                                if custom_field_parts:
                                    line += f": {desc_value} ({', '.join(custom_field_parts)})"
                                else:
                                    line += f": {desc_value}"
                            elif custom_field_parts:
                                line += f" ({', '.join(custom_field_parts)})"
                            temp_f.write(line + "\n")
                        temp_f.write("\n")
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Force immediate disk write
                
                _atomic_replace_file(temp_path, csv_path)
                print(f"✅ Saved token-efficient glossary: {csv_path}")
                type_counts = {}
                for entry_type in grouped_entries:
                    type_counts[entry_type] = len(grouped_entries[entry_type])
                total = sum(type_counts.values())
                print(f"   Total entries: {total}")
                for entry_type, count in type_counts.items():
                    safe_entry_type = entry_type.replace(chr(0x1f), '\\x1F')
                    print(f"   - {safe_entry_type}: {count} entries")
        
        except Exception as e:
            print(f"[Warning] Atomic write failed for CSV: {e}. Attempting direct write...")
            try:
                if use_legacy_format:
                    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        header = ['type', 'raw_name', 'translated_name', 'gender']
                        try:
                            custom_fields = json.loads(os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]'))
                            header.extend(custom_fields)
                        except:
                            custom_fields = []
                        # Custom Fields veto (fallback legacy writer path).
                        # Preserves user's casing and is case-insensitive on
                        # entry dict lookup.
                        legacy_desc_name = _find_description_field_casing(custom_fields)
                        include_description_legacy = legacy_desc_name is not None
                        if include_description_legacy and legacy_desc_name not in header:
                            header.append(legacy_desc_name)
                        writer.writerow(header)
                        for entry in sorted_glossary:
                            entry_type = entry.get('type', 'term')
                            type_config = custom_types.get(entry_type, {})
                            row = [entry_type, entry.get('raw_name', ''), entry.get('translated_name', '')]
                            # Always emit gender column for alignment
                            if type_config.get('has_gender', False):
                                row.append(entry.get('gender', ''))
                            else:
                                row.append('')
                            for field in custom_fields:
                                # Skip description — handled explicitly below
                                if isinstance(field, str) and field.strip().lower() == 'description':
                                    continue
                                row.append(entry.get(field, ''))
                            if include_description_legacy:
                                _dv = ''
                                for _ek, _ev in entry.items():
                                    if isinstance(_ek, str) and _ek.strip().lower() == 'description':
                                        _dv = str(_ev or '')
                                        break
                                row.append(_dv)
                            writer.writerow(row)
                else:
                    grouped_entries = {}
                    for entry in sorted_glossary:
                        entry_type = entry.get('type', 'term')
                        if entry_type not in grouped_entries:
                            grouped_entries[entry_type] = []
                        grouped_entries[entry_type].append(entry)
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        # Write column header
                        custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                        try:
                            custom_fields_list = json.loads(custom_fields_json)
                        except:
                            custom_fields_list = []
                        column_headers = ['raw_name', 'translated_name']
                        # Add gender if any type supports it
                        has_gender = any(type_config.get('has_gender', False) for type_config in custom_types.values())
                        if has_gender:
                            column_headers.append('gender')
                        # Custom Fields veto (fallback token-efficient writer path).
                        # Preserves user's casing.
                        fb_desc_field_name = _find_description_field_casing(custom_fields_list)
                        if fb_desc_field_name is not None:
                            include_desc_fallback = True
                            column_headers.append(fb_desc_field_name)
                        else:
                            include_desc_fallback = False
                        # Then remaining custom fields (rendered in parentheses)
                        for f in custom_fields_list:
                            if str(f).strip().lower() != 'description':
                                column_headers.append(f)
                        f.write(f"Glossary Columns: {', '.join(column_headers)}\n\n")
                        for entry_type in sorted(grouped_entries.keys(), key=lambda x: type_order.get(x, 999)):
                            entries = grouped_entries[entry_type]
                            type_config = custom_types.get(entry_type, {})
                            section_name = entry_type.upper() + 'S' if not entry_type.upper().endswith('S') else entry_type.upper()
                            f.write(f"=== {section_name} ===\n")
                            for entry in entries:
                                raw_name = entry.get('raw_name', '')
                                translated_name = entry.get('translated_name', '')
                                line = f"* {raw_name} = {translated_name}"
                                if type_config.get('has_gender', False):
                                    gender = entry.get('gender', '')
                                    if gender and gender != 'Unknown':
                                        line += f" [{gender}]"
                                custom_field_parts = []
                                for field in custom_fields:
                                    if str(field).strip().lower() == 'description':
                                        continue
                                    value = entry.get(field, '').strip()
                                    if value:
                                        custom_field_parts.append(f"{field}: {value}")
                                # Case-insensitive dict-key lookup for description
                                desc_value = ''
                                if include_desc_fallback:
                                    for _ek, _ev in entry.items():
                                        if isinstance(_ek, str) and _ek.strip().lower() == 'description':
                                            desc_value = str(_ev or '').strip()
                                            break
                                if desc_value:
                                    if custom_field_parts:
                                        line += f": {desc_value} ({', '.join(custom_field_parts)})"
                                    else:
                                        line += f": {desc_value}"
                                elif custom_field_parts:
                                    line += f" ({', '.join(custom_field_parts)})"
                                f.write(line + "\n")
                            f.write("\n")
            except Exception as e2:
                print(f"[Error] Failed to save CSV: {e2}")
        _mirror_glossary_outputs_to_backup(output_path)
            
def extract_chapters_from_epub(epub_path: str, return_metadata: bool = False) -> List:
    """Extract chapters from EPUB for glossary extraction.
    
    Args:
        epub_path: Path to the EPUB file
        return_metadata: If True, returns list of (text, filename) tuples.
                        If False (default), returns list of text strings for backward compat.
    """
    chapters = []
    items = []
    
    # Add this helper function
    def is_html_document(item):
        """Check if an EPUB item is an HTML document"""
        if hasattr(item, 'media_type'):
            return item.media_type in [
                'application/xhtml+xml',
                'text/html',
                'application/html+xml',
                'text/xml'
            ]
        # Fallback for items that don't have media_type
        if hasattr(item, 'get_name'):
            name = item.get_name()
            return name.lower().endswith(('.html', '.xhtml', '.htm'))
        return False
    
    try:
        # Add stop check before reading
        if is_stop_requested():
            return []
            
        book = epub.read_epub(epub_path)
        # Prefer spine order to match content.opf reading order
        try:
            spine_ids = [sid for (sid, _linear) in (book.spine or [])]
        except Exception:
            spine_ids = []
        spine_items = []
        if spine_ids:
            for sid in spine_ids:
                try:
                    it = book.get_item_with_id(sid)
                except Exception:
                    it = None
                if it and is_html_document(it):
                    spine_items.append(it)
            items = spine_items
        else:
            # Fallback to manifest order
            items = [item for item in book.get_items() if is_html_document(item)]
    except Exception as e:
        print(f"[Warning] Manifest load failed, falling back to raw EPUB scan: {e}")
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(('.html', '.xhtml'))]
                for name in names:
                    # Add stop check in loop
                    if is_stop_requested():
                        return chapters
                        
                    try:
                        data = zf.read(name)
                        items.append(type('X', (), {
                            'get_content': lambda self, data=data: data,
                            'get_name': lambda self, name=name: name,
                            'media_type': 'text/html'  # Add media_type for consistency
                        })())
                    except Exception:
                        print(f"[Warning] Could not read zip file entry: {name}")
        except Exception as ze:
            print(f"[Fatal] Cannot open EPUB as zip: {ze}")
            return chapters
            
    # Check if special files should be skipped (same logic as TransateKRtoEN)
    translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
    _kw_env = os.getenv('SPECIAL_FILE_KEYWORDS', '')
    special_keywords = [k.strip().lower() for k in _kw_env.split(',') if k.strip()] if _kw_env else [
        'title', 'toc', 'copyright', 'preface', 'nav',
        'message', 'notice', 'colophon', 'dedication', 'epigraph',
        'foreword', 'acknowledgment', 'author', 'appendix',
        'bibliography'
    ]
    _exact_env = os.getenv('SPECIAL_FILE_EXACT', '')
    special_exact = [k.strip().lower() for k in _exact_env.split(',') if k.strip()] if _exact_env else ['index', 'glossary', 'glossary_extension']
    skipped_special = []

    for item in items:
        # Add stop check before processing each chapter
        if is_stop_requested():
            return chapters
            
        try:
            # Skip special files when TRANSLATE_SPECIAL_FILES is disabled
            item_name = item.get_name() if hasattr(item, 'get_name') else ''
            if not translate_special:
                name_noext = os.path.splitext(os.path.basename(item_name))[0] if item_name else ''
                if name_noext:
                    name_lower = name_noext.lower()
                    # Strip trailing digits to catch files like notice01, cover001
                    name_stripped = re.sub(r'\d+$', '', name_lower).rstrip('_- ')
                    has_digits = bool(re.search(r'\d', name_noext))
                    is_special = False
                    # Exact match: these are special only when the basename matches exactly
                    if name_lower in special_exact:
                        is_special = True
                    # Match only configured special keywords, including numbered variants like notice01.
                    elif any(kw in name_lower for kw in special_keywords):
                        # A no-digit name is special only because it already matched a keyword.
                        # If it has digits, keep it special when the stripped base still matches.
                        if not has_digits or any(kw == name_stripped or kw in name_stripped for kw in special_keywords):
                            is_special = True
                    if is_special:
                        skipped_special.append(name_noext)
                        continue

            raw = item.get_content()
            soup = BeautifulSoup(raw, 'html.parser')
            text = soup.get_text("\n", strip=True)
            if text:
                if return_metadata:
                    chapters.append((text, os.path.basename(item_name) if item_name else ''))
                else:
                    chapters.append(text)
        except Exception as e:
            name = item.get_name() if hasattr(item, 'get_name') else repr(item)
            print(f"[Warning] Skipped corrupted chapter {name}: {e}")

    if skipped_special:
        print(f"⏭️ Skipped {len(skipped_special)} special file(s) for glossary extraction: {', '.join(skipped_special)}")
            
    return chapters

def trim_context_history(history: List[Dict], limit: int, rolling_window: bool = False) -> List[Dict]:
    """Convert stored glossary context into messages for API context.
    
    IMPORTANT: For glossary extraction, we should NOT include user messages (source text)
    in the conversation history as that would confuse the model. We only want to provide
    previously extracted glossary entries as context.
    """
    # Count current exchanges (each user+assistant pair is one exchange)
    current_exchanges = len([m for m in history if m.get('role') == 'user'])

    # Handle based on mode
    if limit > 0 and current_exchanges >= limit:
        if rolling_window:
            # Rolling window: keep the most recent exchanges
            print(f"🔄 Rolling glossary context window: keeping last {limit} chapters")
            # Each exchange is 2 messages (user + assistant)
            messages_to_keep = (limit - 1) * 2 if limit > 1 else 0
            history = history[-messages_to_keep:] if messages_to_keep > 0 else []
        else:
            # Reset mode (original behavior)
            print(f"🔄 Reset glossary context after {limit} chapters")
            return []  # Return empty to reset context

    # Check if we're using a model with thought signature support
    # This includes Gemini 3, experimental models, and Vertex AI models
    model = os.getenv("MODEL", "gemini-2.0-flash").lower()
    api_type = os.getenv("API_TYPE", "gemini").lower()
    
    # Check for models that support thought signatures
    is_thought_signature_model = (
        "gemini-3" in model or 
        "gemini-exp-1206" in model or
        "gemini-2.0-flash-thinking" in model or
        (api_type == "vertex" and "thinking" in model)  # Vertex AI thinking models
    )
    
    # Check if including source is enabled (generally not recommended for glossary)
    include_source = os.getenv("INCLUDE_SOURCE_IN_HISTORY", "0") == "1"
    
    if is_thought_signature_model:
        # For models with thought signature support (Gemini 3, Vertex AI thinking models, etc.),
        # we return the actual assistant messages with their preserved _raw_content_object fields
        # containing thought signatures. We skip user messages to avoid confusing the model with source text
        
        result_messages = []
        
        # Process history to extract only assistant messages with their thought signatures and text intact
        for msg in history:
            if msg.get('role') == 'assistant':
                # Create a copy of the message to preserve it
                assistant_msg = {"role": "assistant", "content": msg.get("content", "")}
                
                # Preserve the raw content object with thought signatures AND text (no filtering)
                if '_raw_content_object' in msg:
                    assistant_msg['_raw_content_object'] = msg['_raw_content_object']
                # If raw_content_object already contains text, drop duplicate content field
                try:
                    has_text_part = False
                    ro = assistant_msg.get('_raw_content_object')
                    if ro:
                        parts = []
                        if hasattr(ro, 'parts'):
                            parts = ro.parts or []
                        elif isinstance(ro, dict):
                            parts = ro.get('parts', []) or []
                        for p in parts:
                            if hasattr(p, 'text') and getattr(p, 'text', None):
                                has_text_part = True
                                break
                            if isinstance(p, dict) and p.get('text'):
                                has_text_part = True
                                break
                    if has_text_part and 'content' in assistant_msg and assistant_msg.get('content') == "":
                        assistant_msg.pop('content', None)
                except Exception:
                    pass
                
                result_messages.append(assistant_msg)
        
        if result_messages:
            print(f"📌 Including {len(result_messages)} assistant message(s) with thought signatures from context")
        
        return result_messages
    
    else:
        # For other models, use memory blocks (legacy behavior)
        memory_blocks: List[str] = []
        
        i = 0
        while i < len(history):
            # Get user and assistant messages
            user_msg = history[i] if i < len(history) and history[i].get('role') == 'user' else None
            assistant_msg = history[i+1] if i+1 < len(history) and history[i+1].get('role') == 'assistant' else None
            
            if user_msg and assistant_msg:
                user_text = (user_msg.get("content") or "").strip()
                assistant_text = (assistant_msg.get("content") or "").strip()
                
                # Optionally include previous source text as a MEMORY block
                if include_source and user_text:
                    prefix = (
                        "[MEMORY - PREVIOUS SOURCE TEXT]\n"
                        "This is prior source content provided for context only.\n"
                        "Do NOT extract from this text directly in your response.\n\n"
                    )
                    footer = "\n\n[END MEMORY BLOCK]\n"
                    memory_blocks.append(prefix + user_text + footer)
                
                # Always include previously extracted glossary entries as MEMORY
                if assistant_text:
                    prefix = (
                        "[MEMORY - PREVIOUSLY EXTRACTED GLOSSARY]\n"
                        "These are previously extracted glossary entries provided for context only.\n"
                        "Build upon these but do NOT repeat these entries verbatim in your response.\n\n"
                    )
                    footer = "\n\n[END MEMORY BLOCK]\n"
                    memory_blocks.append(prefix + assistant_text + footer)
            
            i += 2  # Move to next exchange
        
        if not memory_blocks:
            return []
        
        # Return as assistant message with memory blocks
        combined_memory = "\n".join(memory_blocks)
        return [{"role": "assistant", "content": combined_memory}]

def load_progress(context=None) -> Dict:
    progress_file = _resolved_glossary_progress_file(context)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate the structure
                if not isinstance(data, dict):
                    print(f"[Warning] Progress file has invalid structure, resetting...")
                    return {"completed": [], "glossary": [], "in_progress": []}
                chapters = data.get("chapters", {})
                if not isinstance(chapters, dict):
                    chapters = {}
                    data["chapters"] = chapters
                # Glossary field is deprecated but may exist in old progress files
                # We ignore it now since glossary is loaded from output file instead
                if "glossary" in data:
                    # Remove old glossary field to save space (will be ignored anyway)
                    del data["glossary"]

                completed_from_entries = []
                failed_from_entries = []
                merged_from_entries = []
                in_progress_from_entries = []
                for chapter_key, info in chapters.items():
                    if not isinstance(info, dict):
                        continue
                    idx = _glossary_progress_entry_index(info, chapter_key)
                    if idx is None:
                        continue
                    status = str(info.get("status", "")).lower()
                    if status in ("failed", "qa_failed", "error"):
                        failed_from_entries.append(idx)
                    elif status == "in_progress":
                        in_progress_from_entries.append(idx)
                    elif status == "merged":
                        merged_from_entries.append(idx)
                        completed_from_entries.append(idx)
                    elif status == "completed":
                        completed_from_entries.append(idx)

                if completed_from_entries or failed_from_entries or merged_from_entries or in_progress_from_entries:
                    data["completed"] = _unique_int_list(completed_from_entries)
                    data["failed"] = _unique_int_list(failed_from_entries)
                    data["merged_indices"] = _unique_int_list(merged_from_entries)
                    data["in_progress"] = _unique_int_list(in_progress_from_entries)
                else:
                    # Backward compatibility for old progress files that only
                    # had top-level arrays.
                    data["completed"] = _unique_int_list(data.get("completed", []))
                    data["failed"] = _unique_int_list(data.get("failed", []))
                    data["merged_indices"] = _unique_int_list(data.get("merged_indices", []))
                    data["in_progress"] = _unique_int_list(data.get("in_progress", []))

                failed_set = set(data["failed"])
                if failed_set:
                    data["completed"] = [idx for idx in data["completed"] if idx not in failed_set]
                done_set = set(data["completed"]) | failed_set | set(data["merged_indices"])
                if done_set:
                    data["in_progress"] = [idx for idx in data["in_progress"] if idx not in done_set]
                data["qa_issues_found"] = _normalize_glossary_qa_issues(
                    data.get("qa_issues_found"),
                    data.get("chapters")
                )
                
                # Filter text from _raw_content_object in existing history to avoid duplication
                # This cleans up history that was saved before we added filtering
                # EXCEPT for Vertex AI where thinking is embedded in text
                for msg in data.get("context_history", []):
                    if msg.get('role') == 'assistant' and '_raw_content_object' in msg:
                        raw_obj = msg['_raw_content_object']
                        if isinstance(raw_obj, dict) and 'parts' in raw_obj:
                            # Check if this is a Vertex AI response
                            is_vertex = raw_obj.get('_from_vertex', False)
                            # Filter out text field from parts (except for Vertex)
                            filtered_parts = []
                            for part in raw_obj.get('parts', []):
                                if isinstance(part, dict):
                                    filtered_part = {}
                                    # For Vertex AI, keep text field
                                    if is_vertex and 'text' in part:
                                        filtered_part['text'] = part['text']
                                    # Keep thought signatures
                                    if 'thought' in part:
                                        filtered_part['thought'] = part['thought']
                                    if 'thought_signature' in part:
                                        filtered_part['thought_signature'] = part['thought_signature']
                                    if filtered_part:
                                        filtered_parts.append(filtered_part)
                            
                            if filtered_parts:
                                msg['_raw_content_object'] = {
                                    'parts': filtered_parts,
                                    'role': raw_obj.get('role', 'model'),
                                    '_from_vertex': is_vertex  # Preserve the flag
                                }
                            else:
                                # No thought signatures, remove the raw object
                                del msg['_raw_content_object']
                
                return data
        except json.JSONDecodeError as e:
            print(f"[Warning] Progress file is corrupted (JSON error at line {e.lineno}, column {e.colno}): {e.msg}")
            print(f"   -> Creating backup and starting fresh...")
            # Try to backup the corrupted file
            try:
                import shutil
                import time
                backup_name = f"{progress_file}.corrupted.{int(time.time())}"
                shutil.copy2(progress_file, backup_name)
                print(f"   -> Corrupted file backed up to: {backup_name}")
            except:
                pass
            return {"completed": [], "glossary": [], "context_history": [], "in_progress": []}
        except Exception as e:
            print(f"[Warning] Error loading progress file: {e}")
            return {"completed": [], "glossary": [], "context_history": [], "in_progress": []}
    return {"completed": [], "glossary": [], "context_history": [], "merged_indices": [], "in_progress": []}

def _normalize_entry_type(entry_type, enabled_types):
    """Normalize an entry type to match an enabled type if possible.
    
    Always attempts plural normalization regardless of filter mode.
    Returns the best matching enabled type, or the original type if no match found.
    """
    if not entry_type:
        return entry_type
    if entry_type in enabled_types:
        return entry_type
    # Strip apostrophe-s: "term's" → "term"
    cleaned = entry_type.replace("'s", "").replace("\u2019s", "")
    if cleaned in enabled_types:
        return cleaned
    # Strip trailing s: "terms" → "term", "characters" → "character"
    if cleaned.endswith('s'):
        singular = cleaned[:-1]
        if singular and singular in enabled_types:
            return singular
    # Handle -ies → -y: "abilities" → "ability", "entities" → "entity"
    if cleaned.endswith('ies'):
        y_form = cleaned[:-3] + 'y'
        if y_form in enabled_types:
            return y_form
    # Reverse: enabled type is plural, AI returned singular
    if (cleaned + 's') in enabled_types:
        return cleaned + 's'
    # Reverse y→ies: "funny" → "funnies"
    if cleaned.endswith('y'):
        ies_form = cleaned[:-1] + 'ies'
        if ies_form in enabled_types:
            return ies_form
    if (cleaned + 'ies') in enabled_types:
        return cleaned + 'ies'
    return entry_type


def _is_entry_type_accepted(entry_type, enabled_types):
    """Check if entry type is accepted based on the configured filter mode.
    
    Modes (via GLOSSARY_ENTRY_TYPE_FILTER_MODE env var):
      strict - exact match required (e.g. 'terms' is rejected when only 'term' is enabled)
      loose  - normalizes common plurals/variants and accepts matches
      none   - any value in the type column is accepted (still normalizes)
    """
    filter_mode = os.getenv('GLOSSARY_ENTRY_TYPE_FILTER_MODE', 'none').lower()
    if filter_mode == 'none':
        return True
    normalized = _normalize_entry_type(entry_type, enabled_types)
    if normalized in enabled_types:
        return True
    if filter_mode == 'loose':
        return True  # Loose accepts anything that was attempted to normalize
    return False


def _find_description_field_casing(custom_fields):
    """Return the user's original casing of ``description`` in custom_fields.

    Users can configure the field as ``description``, ``Description``,
    ``DESCRIPTION``, etc. — whatever label they typed in the GUI. We want
    the rest of the pipeline to preserve that casing in column headers
    and dict-key lookups, so callers use this to discover the effective
    name. Returns ``None`` when no description field is present.
    """
    if not custom_fields:
        return None
    for f in custom_fields:
        try:
            if str(f).strip().lower() == 'description':
                return f
        except Exception:
            continue
    return None


def _strip_unwanted_description_keys(entries):
    """Remove any description-shaped keys from each entry when description
    is NOT an active custom field.

    Case-insensitive — strips ``description``, ``Description``,
    ``DESCRIPTION``, etc., because AI responses may vary in casing
    regardless of what the prompt asked for.

    Centralised so every return path of ``parse_api_response`` (JSON list,
    JSON single-object, wrapper-object, and CSV) routes through the same
    filter. Without this, AI responses that ignore the prompt and return
    description values anyway would leak through on JSON-shaped responses
    even though the user opted out via Custom Fields.
    """
    if not entries:
        return entries
    try:
        if _glossary_description_active():
            return entries
    except Exception:
        return entries
    for _entry in entries:
        if not isinstance(_entry, dict):
            continue
        # Collect matching keys first to avoid mutating dict during iteration.
        keys_to_drop = [
            k for k in list(_entry.keys())
            if isinstance(k, str) and k.strip().lower() == 'description'
        ]
        for k in keys_to_drop:
            _entry.pop(k, None)
    return entries


def parse_api_response(response_text: str) -> List[Dict]:
    """Parse API response to extract glossary entries - handles custom types"""
    entries = []
    import csv
    
    # Get enabled types from custom configuration
    custom_types = get_custom_entry_types()
    enabled_types = [t for t, cfg in custom_types.items() if cfg.get('enabled', True)]

    def _looks_like_gender_value(value) -> bool:
        normalized = _normalize_gender_value(value)
        return normalized in {
            'male', 'female', 'unknown', 'nonbinary', 'non-binary',
            'ambiguous', 'mixed', 'various', 'n/a', 'na', 'none', '-'
        }
    
    # First try JSON parsing
    try:
        # Clean up response text
        cleaned_text = response_text.strip()
        
        # Strip AI thinking/reasoning blocks that may have leaked into the response.
        # Various providers use different tags: <thinking>, <think>, <thought>, etc.
        import re
        cleaned_text = re.sub(
            r'<(?:thinking|think|thought)>.*?</(?:thinking|think|thought)>',
            '', cleaned_text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        
        # Remove markdown code blocks if present
        if '```json' in cleaned_text or '```' in cleaned_text:
            import re
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if code_block_match:
                cleaned_text = code_block_match.group(1)
        
        # Try to find JSON array or object
        import re
        json_match = re.search(r'[\[\{].*[\]\}]', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Check if entry type is enabled
                        entry_type = item.get('type', '').lower()
                        
                        # Handle legacy format where type is the key
                        if not entry_type:
                            for type_name in enabled_types:
                                if type_name in item:
                                    entry_type = type_name
                                    fixed_entry = {
                                        'type': type_name,
                                        'raw_name': item.get(type_name, ''),
                                        'translated_name': item.get('translated_name', '')
                                    }
                                    
                                    # Add gender if type supports it
                                    if custom_types.get(type_name, {}).get('has_gender', False):
                                        fixed_entry['gender'] = item.get('gender', 'Unknown')
                                    
                                    # Copy other fields
                                    for k, v in item.items():
                                        if k not in [type_name, 'translated_name', 'gender', 'type', 'raw_name']:
                                            fixed_entry[k] = v
                                    
                                    entries.append(fixed_entry)
                                    break
                        else:
                            # Standard format with type field
                            if _is_entry_type_accepted(entry_type, enabled_types):
                                entries.append(item)
                
                return _strip_unwanted_description_keys(entries)
                
            elif isinstance(data, dict):
                # Handle single entry
                entry_type = data.get('type', '').lower()
                if _is_entry_type_accepted(entry_type, enabled_types):
                    return _strip_unwanted_description_keys([data])
                
                # Check for wrapper
                for key in ['entries', 'glossary', 'characters', 'terms', 'data']:
                    if key in data and isinstance(data[key], list):
                        # Recursive call already routes through this function,
                        # which will apply the strip at its own return point.
                        return parse_api_response(json.dumps(data[key]))
                
                return []
                
    except (json.JSONDecodeError, AttributeError):
        # CSV output is expected for the default glossary prompt; fall through
        # to the CSV parser without logging a scary-but-harmless JSON miss.
        pass
    
    # CSV-like format parsing
    lines = response_text.strip().split('\n')
    header_fields = None
    def _is_basic_latin(s: str) -> bool:
        try:
            for ch in (s or ""):
                if ch.isspace():
                    continue
                # Treat any non-ASCII letter/mark as non-Latin
                if ord(ch) > 127 and ch.isalpha():
                    return False
            return True
        except Exception:
            return False



    for line in lines:
        try:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Detect and store header to preserve every returned column
            _GSEP = '\x1F'
            # Normalize literal \x1F text to the actual byte (models may output the escape as text)
            if '\\x1F' in line or '\\x1f' in line:
                line = line.replace('\\x1F', _GSEP).replace('\\x1f', _GSEP)
            # Normalize tab-separated output to \x1F (some models use tabs when they can't produce the control char)
            if _GSEP not in line and '\t' in line and line.count('\t') >= 2:
                line = line.replace('\t', _GSEP)
            # Space-separated fallback: when model can't produce \x1F and uses aligned spaces
            if _GSEP not in line and ',' not in line:
                _low = line.lower()
                _space_prefixes = ('character ', 'term ', 'location ', 'skill ', 'item ', 'organization ', 'title ', 'book ')
                if any(_low.startswith(p) for p in _space_prefixes) or ('type' in _low and 'raw_name' in _low):
                    if '  ' in line:  # 2+ consecutive spaces = column separator
                        line = re.sub(r'  +', _GSEP, line)
            if 'type' in line.lower() and 'raw_name' in line.lower():
                if _GSEP in line:
                    header_fields = [c.strip() for c in line.split(_GSEP) if c.strip()]
                else:
                    try:
                        header_fields = [c.strip() for c in next(csv.reader([line])) if c.strip()]
                    except Exception:
                        header_fields = [c.strip() for c in line.split(',') if c.strip()]
                continue

            if _GSEP in line:
                row = [p.strip() for p in line.split(_GSEP)]
            else:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = [p.strip() for p in line.split(',')]

            # --- NEW CLEANUP LOGIC ---
            if len(row) >= 3:
                raw_check = row[1].strip()
                trans_check = row[2].strip()
                if raw_check == '()' or trans_check == '()':
                    print(f"[Warning] Filtered invalid entry with empty brackets: {line}")
                    continue
                if raw_check.lower() == trans_check.lower() and len(raw_check) > 3:
                    # Keep Latin-only entries even if raw==translated (names/terms in Latin script)
                    if not (_is_basic_latin(raw_check) and _is_basic_latin(trans_check)):
                        print(f"[Warning] Filtered untranslated entry (raw==translated): {line}")
                        continue
            # -------------------------

            # If we saw a header, map every column by name to keep all AI-returned data
            if header_fields:
                if len(row) < len(header_fields):
                    row += [''] * (len(header_fields) - len(row))
                elif len(row) > len(header_fields):
                    # If the model failed to quote a comma-containing description, merge overflow into the last column
                    desc_idx = next((i for i, h in enumerate(header_fields) if h.lower() == 'description'), None)
                    if desc_idx is not None and desc_idx < len(header_fields):
                        # Rejoin with the same separator that was used to split
                        _rejoin_sep = _GSEP if _GSEP in line else ','
                        row = row[:desc_idx] + [_rejoin_sep.join(row[desc_idx:])]
                    else:
                        row = row[:len(header_fields)]
                entry_map = {header_fields[i]: row[i] for i in range(len(header_fields))}
                entry_type = (entry_map.get('type') or '').lower() or 'term'
                if not _is_entry_type_accepted(entry_type, enabled_types):
                    continue
                entry_map['type'] = _normalize_entry_type(entry_type, enabled_types)

                # Default gender if column exists but value missing for gendered types
                if custom_types.get(entry_type, {}).get('has_gender', False):
                    if 'gender' not in entry_map or not entry_map.get('gender'):
                        entry_map['gender'] = 'Unknown'

                # Require essential fields
                if not entry_map.get('raw_name') or not entry_map.get('translated_name'):
                    continue

                # Skip entries where translated_name == raw_name (AI returned unchanged)
                if os.getenv('GLOSSARY_SKIP_IDENTICAL_ENTRIES', '1') == '1':
                    if str(entry_map.get('raw_name', '')).strip() == str(entry_map.get('translated_name', '')).strip():
                        continue

                # CJK/Latin script validation when output language is non-CJK
                if os.getenv('GLOSSARY_CJK_SCRIPT_FILTER', '0') == '1' and _is_known_non_cjk_output_language():
                    _raw = str(entry_map.get('raw_name', '')).strip()
                    _trans = str(entry_map.get('translated_name', '')).strip()
                    # Reject translated_name if it contains CJK characters (should be Latin/romaji)
                    if _trans and _contains_cjk(_trans):
                        print(f"🚫 [CJK Filter] Rejected entry — CJK in translated_name (output is non-CJK): {_raw} → {_trans}")
                        continue

                entries.append(entry_map)
                continue

            # Legacy fallback (no header detected)
            parts = row
            if len(parts) >= 3:
                entry_type = parts[0].lower()
                normalized_entry_type = _normalize_entry_type(entry_type, enabled_types)

                # Check if type is enabled
                if not _is_entry_type_accepted(entry_type, enabled_types):
                    continue

                entry = {
                    'type': normalized_entry_type,
                    'raw_name': parts[1],
                    'translated_name': parts[2]
                }

                # Add gender if type supports it and it's provided
                type_config = custom_types.get(normalized_entry_type, custom_types.get(entry_type, {}))
                has_gender_field = bool(type_config.get('has_gender', False))
                custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json)
                except Exception:
                    custom_fields = []
                description_active = _find_description_field_casing(custom_fields) is not None
                start_idx = 4 if has_gender_field else 3

                if has_gender_field and len(parts) > 3:
                    possible_gender = str(parts[3] or '').strip()
                    if possible_gender and not (
                        description_active
                        and len(parts) == 4
                        and not _looks_like_gender_value(possible_gender)
                    ):
                        entry['gender'] = possible_gender
                    else:
                        entry['gender'] = 'Unknown'
                        if possible_gender:
                            # Gender-enabled type omitted the gender column and
                            # placed description in the fourth position.
                            start_idx = 3
                elif has_gender_field:
                    entry['gender'] = 'Unknown'
                elif len(parts) > 4 and not str(parts[3] or '').strip():
                    # Some no-header responses still follow the universal
                    # type,raw,translated,gender,description layout. For
                    # non-gender types that fourth column is just a blank
                    # placeholder, so custom fields begin after it.
                    start_idx = 4

                # Add any custom fields
                try:
                    for i, field in enumerate(custom_fields):
                        if len(parts) > start_idx + i:
                            field_value = parts[start_idx + i]
                            if field_value:  # Only add if not empty
                                entry[field] = field_value
                except:
                    pass

                entries.append(entry)
        except IndexError as e:
            # If no extra columns are configured, emit a warning; otherwise stay silent.
            try:
                custom_fields = json.loads(os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]'))
            except Exception:
                custom_fields = []
            if not custom_fields:
                print(f"[Warning] Malformed glossary line (IndexError): {line} -> {e}")
            continue

    # Post-filter: reject entries with all-Latin raw_names when the batch
    # is auto-detected as CJK source and the output is a known non-CJK language.
    # Runs AFTER all entries are collected so we can detect the source script
    # from the raw_names themselves - no hardcoded config dependency.
    if entries and os.getenv('GLOSSARY_CJK_SCRIPT_FILTER', '0') == '1' and _is_known_non_cjk_output_language():
        all_raw = [str(e.get('raw_name', '')) for e in entries]
        if _is_cjk_source_detected(all_raw):
            before = len(entries)
            entries = [
                e for e in entries
                if _contains_cjk(str(e.get('raw_name', '')).strip())
            ]
            rejected = before - len(entries)
            if rejected:
                print(f"🚫 [CJK Filter] Rejected {rejected} entries with no CJK in raw_name (auto-detected CJK source, non-CJK output)")

    # Post-filter: if the user did NOT include ``description`` in their
    # custom fields list, strip any description value the AI may have
    # returned anyway. Pairs with the prompt-side description-rule
    # placeholder strips — together they guarantee description data is
    # absent both from what we ask for and from what we accept in
    # downstream CSV / token-efficient output. Uses the shared helper so
    # every return path (JSON + CSV) applies the same filter.
    return _strip_unwanted_description_keys(entries)

def _is_cjk_output_language():
    """Return True when the glossary target language is CJK (Korean/Chinese/Japanese)."""
    lang = os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English').strip().lower()
    cjk_langs = {
        'korean', 'japanese', 'chinese',
        'simplified chinese', 'traditional chinese',
        'mandarin', 'cantonese',
        # Common native names
        '한국어', '日本語', '中文', '中国语',
    }
    return lang in cjk_langs

def _detect_dominant_script_glossary(text, max_chars=10000):
    """Lightweight dominant script detection (same logic as scan_html_folder.detect_dominant_script).

    Returns one of: 'cjk', 'japanese', 'korean', 'cyrillic', 'arabic', 'latin', 'other'
    """
    ranges = [
        ('cjk', [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)]),
        ('japanese', [(0x3040, 0x309F), (0x30A0, 0x30FF)]),
        ('korean', [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
        ('cyrillic', [(0x0400, 0x04FF), (0x0500, 0x052F)]),
        ('arabic', [(0x0600, 0x06FF), (0x0750, 0x077F)]),
        ('latin', [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]),
    ]
    counts = {k: 0 for k, _ in ranges}
    for ch in text[:max_chars]:
        code = ord(ch)
        for key, spans in ranges:
            if any(start <= code <= end for start, end in spans):
                counts[key] += 1
                break
    if not any(counts.values()):
        return 'other'
    return max(counts, key=counts.get)


def _glossary_script_is_cjk(script):
    return script in {'cjk', 'korean', 'japanese'}


def _glossary_script_label(script):
    return {'cjk': 'Chinese/CJK', 'korean': 'Korean', 'japanese': 'Japanese'}.get(script, script or 'other')


def _set_glossary_source_script_detection(script, sample_count, sample_kind="raw entries", log=True):
    global GLOSSARY_SOURCE_SCRIPT, GLOSSARY_SOURCE_SCRIPT_IS_CJK
    global _GLOSSARY_SOURCE_SCRIPT_READY, _GLOSSARY_SOURCE_SCRIPT_LOGGED
    with _glossary_source_script_lock:
        if _GLOSSARY_SOURCE_SCRIPT_READY:
            return bool(GLOSSARY_SOURCE_SCRIPT_IS_CJK)
        GLOSSARY_SOURCE_SCRIPT = script
        GLOSSARY_SOURCE_SCRIPT_IS_CJK = _glossary_script_is_cjk(script)
        _GLOSSARY_SOURCE_SCRIPT_READY = True
        is_cjk = bool(GLOSSARY_SOURCE_SCRIPT_IS_CJK)
        should_log = log and not _GLOSSARY_SOURCE_SCRIPT_LOGGED
        if should_log:
            _GLOSSARY_SOURCE_SCRIPT_LOGGED = True

    if should_log:
        unit = "chapter" if sample_kind == "chapter sample" else "entry"
        suffix = "" if int(sample_count or 0) == 1 else "s"
        status = "CJK source confirmed" if is_cjk else "non-CJK source, filter skipped"
        print(
            f"[CJK Filter] Auto-detected source script: {_glossary_script_label(script)} "
            f"({sample_kind}: {int(sample_count or 0)} {unit}{suffix}) -> {status}"
        )
    return is_cjk


def _get_glossary_source_script_detection():
    with _glossary_source_script_lock:
        if not _GLOSSARY_SOURCE_SCRIPT_READY:
            return None
        return bool(GLOSSARY_SOURCE_SCRIPT_IS_CJK)


def _glossary_cjk_filter_active():
    return os.getenv('GLOSSARY_CJK_SCRIPT_FILTER', '0') == '1' and _is_known_non_cjk_output_language()


def _prime_glossary_source_script_from_chapters(chapters_to_process, sample_size=10, chars_per_chapter=5000):
    if not _glossary_cjk_filter_active():
        return None

    metadata_language = _get_glossary_source_language_from_metadata()
    if metadata_language:
        return _metadata_language_is_cjk(metadata_language)

    sample_parts = []
    for _idx, chapter_text in list(chapters_to_process or [])[:sample_size]:
        text = str(chapter_text or "").strip()
        if text:
            sample_parts.append(text[:chars_per_chapter])

    if not sample_parts:
        return None

    sample = "\n\n".join(sample_parts)
    script = _detect_dominant_script_glossary(sample, max_chars=sample_size * chars_per_chapter)
    return _set_glossary_source_script_detection(script, len(sample_parts), "chapter sample")


def _is_cjk_source_detected(raw_names):
    """Auto-detect whether the source material is CJK by sampling raw_name entries.

    Concatenates the raw_names and runs script detection heuristics.
    Returns True when the dominant script is CJK/Korean/Japanese.
    No hardcoded config dependency — purely content-driven.
    """
    metadata_language = _get_glossary_source_language_from_metadata()
    if metadata_language:
        return _metadata_language_is_cjk(metadata_language)

    cached_script = _get_glossary_source_script_detection()
    if cached_script is not None:
        return cached_script

    if not raw_names:
        return False
    # Sample up to 50 raw names for speed
    sample = ' '.join(str(n) for n in raw_names[:50])
    if not sample.strip():
        return False
    script = _detect_dominant_script_glossary(sample)
    return _set_glossary_source_script_detection(script, len(raw_names), "raw fallback sample")

def _is_known_non_cjk_output_language():
    """Return True only for well-known non-CJK output languages.

    The CJK script filter should only fire when we are confident the output
    language uses a Latin/non-CJK script.  For custom or unrecognised
    languages the filter is disabled to avoid false positives.
    """
    lang = os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English').strip().lower()
    known_non_cjk = {
        'english', 'spanish', 'french', 'german', 'italian', 'portuguese',
        'dutch', 'russian', 'polish', 'swedish', 'norwegian', 'danish',
        'finnish', 'czech', 'hungarian', 'romanian', 'turkish', 'greek',
        'arabic', 'hebrew', 'hindi', 'thai', 'vietnamese', 'bahasa indonesia', 'indonesian',
        'malay', 'tagalog', 'filipino', 'ukrainian', 'bulgarian',
        'croatian', 'serbian', 'slovak', 'slovenian', 'latvian',
        'lithuanian', 'estonian', 'persian', 'farsi', 'urdu', 'bengali',
        'marathi', 'punjabi', 'gujarati', 'tamil', 'telugu', 'kannada',
        'malayalam', 'swahili', 'amharic', 'hausa', 'yoruba', 'zulu',
        'catalan', 'galician', 'basque',
        'icelandic', 'albanian', 'macedonian', 'georgian', 'armenian',
        'azerbaijani', 'kazakh', 'uzbek', 'mongolian', 'nepali',
        'sinhala', 'burmese', 'khmer', 'lao', 'afrikaans',
        'brazilian portuguese', 'latin american spanish',
    }
    return lang in known_non_cjk

def _contains_cjk(s):
    """Return True if the string contains any Hangul, CJK Unified, Hiragana, or Katakana characters."""
    for ch in (s or ""):
        cp = ord(ch)
        if (
            0xAC00 <= cp <= 0xD7AF        # Hangul Syllables
            or 0x1100 <= cp <= 0x11FF      # Hangul Jamo
            or 0x3130 <= cp <= 0x318F      # Hangul Compatibility Jamo
            or 0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF      # CJK Unified Ideographs Extension A
            or 0x20000 <= cp <= 0x2A6DF    # CJK Unified Ideographs Extension B
            or 0xF900 <= cp <= 0xFAFF      # CJK Compatibility Ideographs
            or 0x3040 <= cp <= 0x309F      # Hiragana
            or 0x30A0 <= cp <= 0x30FF      # Katakana
        ):
            return True
    return False

def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields and enabled type"""
    if 'type' not in entry:
        return False
    
    # Check if type is accepted by the current filter mode
    custom_types = get_custom_entry_types()
    entry_type = entry.get('type', '').lower()
    enabled_types = [t for t, cfg in custom_types.items() if cfg.get('enabled', True)]
    
    if not _is_entry_type_accepted(entry_type, enabled_types):
        return False
    
    # Must have raw_name and translated_name
    if 'raw_name' not in entry or not entry['raw_name']:
        return False
    if 'translated_name' not in entry or not entry['translated_name']:
        return False
    
    # Skip entries where translated_name == raw_name (AI returned unchanged)
    if os.getenv('GLOSSARY_SKIP_IDENTICAL_ENTRIES', '1') == '1':
        raw = str(entry.get('raw_name', '')).strip()
        trans = str(entry.get('translated_name', '')).strip()
        if raw and trans and raw == trans:
            return False
    
    # CJK script validation: reject translated_name with CJK when output is non-CJK
    if os.getenv('GLOSSARY_CJK_SCRIPT_FILTER', '0') == '1' and _is_known_non_cjk_output_language():
        trans = str(entry.get('translated_name', '')).strip()
        if trans and _contains_cjk(trans):
            return False
    
    return True

def _glossary_description_active(custom_fields=None):
    """Return True when ``description`` is in the active custom fields list.

    Reads ``GLOSSARY_CUSTOM_FIELDS`` env var when ``custom_fields`` is None.
    Centralised so the prompt builder, AI-response parser, and any future
    callers share one source of truth.
    """
    if custom_fields is None:
        try:
            custom_fields = json.loads(os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]'))
        except Exception:
            custom_fields = []
    return any(
        str(f).strip().lower() == 'description'
        for f in (custom_fields or [])
    )


# Canonical replacements for the description-rule placeholders. When the
# ``description`` custom field is active they expand to these strings; when
# it is not, line-owning placeholders (and their owning line) are stripped
# out entirely and inline placeholders are simply emptied, so the prompt
# doesn't instruct the AI to emit description data the user didn't ask for.
_DESCRIPTION_MANDATORY_TEXT = "The description column is mandatory and must be detailed"
_DESCRIPTION_DETAILED_TEXT = "The description column must contain detailed context/explanation"
# Inline placeholder — sits mid-sentence in the "Critical Requirement" line,
# so it must be replaced in-place (not line-stripped). Expands to " and
# description" (with the leading space) when active so the sentence reads
# "The translated name and description column must be in ..." and reads
# "The translated name column must be in ..." when stripped.
_DESCRIPTION_IN_LANGUAGE_TEXT = " and description"
# Inline placeholder — parenthetical note appended to the REJECT-starters
# rule so the "Me / How / What / ... / But" exclusion rule doesn't
# incorrectly reject description content. Stored with a leading space so
# the rule line reads cleanly in both modes:
#   active:   ..."But" (The description column is excluded from this restriction)
#   inactive: ..."But"
_DESCRIPTION_EXCLUDED_NOTE_TEXT = " (The description column is excluded from this restriction)"
_SUBJECT_TRACKING_INSTRUCTION_TEXT = (
    'Strictly follow a Subject Tracking & Pronoun Resolution process when extracting character data: track omitted or ambiguous subjects/pronouns from surrounding context, titles, relationships, dialogue, and repeated mentions so gender/person fields and descriptions stay consistent instead of defaulting to "he", "she", or "it".'
)

# Example CSV lines for glossary prompts. The description-bearing version is
# used when description is active; the plain version when it is not.
_DESCRIPTION_EXAMPLE_TEXT = (
    'For example:\n'
    'character,이히리ᐐ 나애,Dihirit Ade,female,"The enigmatic guild leader of the Shadow Lotus who operates from the concealed backrooms of the capital, manipulating city politics through commerce and wielding dual daggers with lethal precision"\n'
    'character,뢤사난,Kim Sang-hyu,male,"A master swordsman from the Northern Sect known for his icy demeanor and unparalleled skill with the Frost Blade technique which he uses to defend the border fortress"'
)
_EXAMPLE_TEXT = (
    'For example:\n'
    'character,이히리ᐐ 나애,Dihirit Ade,female\n'
    'character,뢤사난,Kim Sang-hyu,male\n'
    'term,간편헤,Gale Hardest,'
)

# Name-split example lines. The description-bearing version shows
# descriptions in the split entries; the plain version omits them.
_DESCRIPTION_NAME_SPLIT_EXAMPLE_TEXT = (
    'For character entries, the raw_name must contain ONLY the given name (first name), never a full name. '
    'If the text mentions a character by full name (e.g. "김상현"), split it: create one entry with raw_name "상현" '
    '(given name) and a second entry with raw_name "김" (surname). Example output:\n'
    '  character,상현,Sang-hyun,male,"A knight of the royal guard"\n'
    '  character,김,Kim,,"Surname of Sang-hyun"\n'
    '  Do NOT create a single combined full-name entry (e.g. character,김상현,Kim Sang-hyun,male,…) — always split into given name + surname.'
)
_NAME_SPLIT_EXAMPLE_TEXT = (
    'For character entries, the raw_name must contain ONLY the given name (first name), never a full name. '
    'If the text mentions a character by full name (e.g. "김상현"), split it: create one entry with raw_name "상현" '
    '(given name) and a second entry with raw_name "김" (surname). Example output:\n'
    '  character,상현,Sang-hyun,male\n'
    '  character,김,Kim,\n'
    '  Do NOT create a single combined full-name entry (e.g. character,김상현,Kim Sang-hyun,male) — always split into given name + surname.'
)

# ── Canonical default glossary prompt ──────────────────────────────
# This is the **single source of truth** for the Balanced/Full default
# prompt.  GlossaryManager_GUI.py imports this constant rather than
# maintaining its own copy, so edits only need to happen here.
DEFAULT_GLOSSARY_PROMPT = """\
You are a novel glossary extraction assistant.

You must strictly return ONLY CSV format with columns separated by commas.
Columns and entry types in this exact order provided:

{fields}

{gender_instruction}
{description_mandatory}
IMPORTANT: Use commas to separate columns. Wrap a field value in double quotes ONLY when the value itself contains a comma.

Critical Requirement: The translated name{description_in_language} column must be in {language}, While the raw name column must the same as the source language.
The translated_name column must be a direct translation or transliteration of the raw_name ONLY. Do NOT use role labels, descriptions, or invented names as translations.

{description_example}
{example}

CRITICAL EXTRACTION RULES:
- Extract All {entries}
- Strictly follow a Subject Tracking & Pronoun Resolution process when extracting character data: track omitted or ambiguous subjects/pronouns from surrounding context, titles, relationships, dialogue, and repeated mentions so gender/person fields and descriptions stay consistent instead of defaulting to "he", "she", or "it".
- Do NOT extract sentences, dialogue, actions, questions, or statements as glossary entries
- REJECT entries that contain verbs or end with punctuation (?, !, .)
- REJECT entries starting with: "Me", "How", "What", "Why", "I", "He", "She", "They", "That's", "So", "Therefore", "Still", "But"{description_excluded_note}
- Do NOT create entries for common pronouns (나, 저, 너, 그, 그녀, 우리, 私, 僕, 俺, я, etc.) — these are NOT character names. Do NOT translate pronouns as role labels like "Narrator", "Protagonist", "Main Character", or "MC"
- Do NOT output any entries that are rejected by the above rules; skip them entirely
- REJECT generic common nouns, unnamed extras, and bare titles/roles (e.g. "Woman", "Man", "Boy", "Girl", "Villager", "Guard", "Soldier", "Aunt", "Father", "Queen", "Prince", "King", "Princess", "Knight", "Servant", "Maid", 여자, 남자, 소녀, 소년, 아줌마, 아버지, 여왕, 왕자). These are NOT proper nouns and must be skipped.
- REJECT descriptive noun phrases and adjectives attached to generic nouns (e.g. "Blonde Elf Girl", "Orange-eyed Beastman", "White-bearded Merchant", "Fake Couple", "Bespectacled Student"). Only extract actual names or standardized titles.
- If unsure whether something is a proper noun/name, skip it
- {description_name_split_example}
- {name_split_example}
- {description_detailed}
- The translated_name MUST be a strict literal dictionary translation or transliteration of the raw_name ONLY. You are FORBIDDEN from injecting story context, roles, or extra adjectives (e.g., do NOT translate "女学生" as "Female Student Assassin" or "주인님" as "The Protagonist").
- You must include absolutely all characters found in the provided text in your glossary generation. Do not skip any character."""

# ── Canonical default AUTO glossary prompt ─────────────────────────
# Single source of truth for the Minimal/Auto default prompt.
# GlossaryManager_GUI.py, GlossaryManager.py, and translator_gui.py
# import this constant rather than maintaining their own copies.
DEFAULT_AUTO_GLOSSARY_PROMPT = """\
You are a novel glossary extraction assistant.

You must strictly return ONLY CSV format with columns separated by commas.
Columns in this exact order: type,raw_name,translated_name,gender,description
{gender_instruction}
The description column is optional and can contain brief context (role, location, significance).
IMPORTANT: Use commas to separate columns. Wrap a field value in double quotes ONLY when the value itself contains a comma.

Critical Requirement: The translated name and description column must be in {language}, While the raw name column must the same as the source language.
The translated_name column must be a direct translation or transliteration of the raw_name ONLY. Do NOT use role labels, descriptions, or invented names as translations.
{description_example}
{example}


CRITICAL EXTRACTION RULES:
- Extract All Character names, Terms, Location names, Ability/Skill names, Item names, Organization names, and Titles/Ranks
- {subject_tracking_instruction}
- Do NOT extract sentences, dialogue, actions, questions, or statements as glossary entries
- REJECT entries that contain verbs or end with punctuation (?, !, .)
- REJECT entries starting with: "Me", "How", "What", "Why", "I", "He", "She", "They", "That's", "So", "Therefore", "Still", "But" (The description column is excluded from this restriction)
- Do NOT create entries for common pronouns (나, 저, 너, 그, 그녀, 우리, 私, 僕, 俺, я, etc.) — these are NOT character names. Do NOT translate pronouns as role labels like "Narrator", "Protagonist", "Main Character", or "MC"
- Do NOT output any entries that are rejected by the above rules; skip them entirely
- REJECT generic common nouns, unnamed extras, and bare titles/roles (e.g. "Woman", "Man", "Boy", "Girl", "Villager", "Guard", "Soldier", "Aunt", "Father", "Queen", "Prince", "King", "Princess", "Knight", "Servant", "Maid", 여자, 남자, 소녀, 소년, 아줌마, 아버지, 여왕, 왕자). These are NOT proper nouns and must be skipped.
- REJECT descriptive noun phrases and adjectives attached to generic nouns (e.g. "Blonde Elf Girl", "Orange-eyed Beastman", "White-bearded Merchant", "Fake Couple", "Bespectacled Student"). Only extract actual names or standardized titles.
- If unsure whether something is a proper noun/name, skip it
- {description_name_split_example}
- {name_split_example}
- The description column must contain detailed context/explanation
- The translated_name MUST be a strict literal dictionary translation or transliteration of the raw_name ONLY. You are FORBIDDEN from injecting story context, roles, or extra adjectives (e.g., do NOT translate "여학생" as "Female Student Assassin" or "주인님" as "The Protagonist").
- Create at least one glossary entry for EVERY context marker window (lines ending with "=== CONTEXT N END ==="); treat each marker boundary as a required extraction point.
- You must create {marker} glossary entries (one or more per window; do not invent placeholders).
- You must include absolutely all characters found in the provided text in your glossary generation. Do not skip any character."""

# Placeholders that occupy their own line in the prompt. When the description
# field isn't active they get stripped along with the entire line (including
# any leading ``- `` bullet) so we don't leave orphan bullets behind.
_DESCRIPTION_LINE_PLACEHOLDERS = (
    '{description_mandatory}',
    '{description_detailed}',
    '{description_example}',
    '{description_name_split_example}',
)
# Placeholders that appear inline within a larger sentence. When inactive
# they must be replaced with an empty string — NOT stripped by line —
# otherwise the enclosing sentence would be destroyed.
_DESCRIPTION_INLINE_PLACEHOLDERS = (
    '{description_in_language}',
    '{description_excluded_note}',
)
# "No-description" line placeholders — the inverse of _DESCRIPTION_LINE_PLACEHOLDERS.
# These are stripped when description IS active and expanded when inactive.
_NO_DESCRIPTION_LINE_PLACEHOLDERS = (
    '{example}',
    '{name_split_example}',
)


def _apply_subject_tracking_placeholder(prompt_text, include_gender_context=None):
    """Expand subject-tracking guidance only when gender context is enabled."""
    if not isinstance(prompt_text, str) or '{subject_tracking_instruction}' not in prompt_text:
        return prompt_text
    if include_gender_context is None:
        include_gender_context = os.getenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "0") == "1"
    if include_gender_context:
        return prompt_text.replace('{subject_tracking_instruction}', _SUBJECT_TRACKING_INSTRUCTION_TEXT)

    import re as _re_st
    return _re_st.sub(
        r'^[ \t]*-?[ \t]*'
        + _re_st.escape('{subject_tracking_instruction}')
        + r'[ \t]*\r?\n?',
        '',
        prompt_text,
        flags=_re_st.MULTILINE,
    )


def _apply_description_rule_placeholders(prompt_text, custom_fields=None, description_active=None):
    """Replace description-rule placeholders in glossary prompts.

    Line-owning placeholders (`{description_mandatory}`, `{description_detailed}`,
    `{description_example}`, `{description_name_split_example}`):
    expanded when description is active; removed with their entire line when not.

    No-description line placeholders (`{example}`, `{name_split_example}`):
    the inverse — expanded when description is *inactive*; removed when active.

    Inline placeholders (`{description_in_language}`, `{description_excluded_note}`):
    replaced with "" when inactive (sentence stays intact).

    Args:
        prompt_text: The prompt string to process.
        custom_fields: Optional list of custom field names. Used to determine
            if description is active when ``description_active`` is None.
        description_active: Explicit override for the description-active flag.
            When True/False, skips the Custom Fields lookup entirely.
            Useful for the auto/minimal prompt path which uses the
            ``GLOSSARY_INCLUDE_DESCRIPTION`` env var instead.
"""
    if not isinstance(prompt_text, str):
        return prompt_text
    all_placeholders = (
        _DESCRIPTION_LINE_PLACEHOLDERS
        + _DESCRIPTION_INLINE_PLACEHOLDERS
        + _NO_DESCRIPTION_LINE_PLACEHOLDERS
    )
    if not any(p in prompt_text for p in all_placeholders):
        return prompt_text
    if description_active is not None:
        active = bool(description_active)
    else:
        active = _glossary_description_active(custom_fields)
    import re as _re_dr
    if active:
        result = prompt_text
        result = result.replace('{description_mandatory}', _DESCRIPTION_MANDATORY_TEXT)
        result = result.replace('{description_detailed}', _DESCRIPTION_DETAILED_TEXT)
        result = result.replace('{description_in_language}', _DESCRIPTION_IN_LANGUAGE_TEXT)
        result = result.replace('{description_excluded_note}', _DESCRIPTION_EXCLUDED_NOTE_TEXT)
        result = result.replace('{description_example}', _DESCRIPTION_EXAMPLE_TEXT)
        result = result.replace('{description_name_split_example}', _DESCRIPTION_NAME_SPLIT_EXAMPLE_TEXT)
        # Strip the no-description placeholders (they are the inverse)
        for placeholder in _NO_DESCRIPTION_LINE_PLACEHOLDERS:
            result = _re_dr.sub(
                r'^[ \t]*-?[ \t]*'
                + _re_dr.escape(placeholder)
                + r'[ \t]*\r?\n?',
                '',
                result,
                flags=_re_dr.MULTILINE,
            )
        result = _re_dr.sub(r'\n{3,}', '\n\n', result)
        return result
    # Inactive path: strip description line-owners, expand no-description ones.
    cleaned = prompt_text
    for placeholder in _DESCRIPTION_LINE_PLACEHOLDERS:
        cleaned = _re_dr.sub(
            r'^[ \t]*-?[ \t]*'
            + _re_dr.escape(placeholder)
            + r'[ \t]*\r?\n?',
            '',
            cleaned,
            flags=_re_dr.MULTILINE,
        )
    for placeholder in _DESCRIPTION_INLINE_PLACEHOLDERS:
        cleaned = cleaned.replace(placeholder, '')
    # Expand the no-description placeholders
    cleaned = cleaned.replace('{example}', _EXAMPLE_TEXT)
    cleaned = cleaned.replace('{name_split_example}', _NAME_SPLIT_EXAMPLE_TEXT)
    cleaned = _re_dr.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


def build_prompt(chapter_text: str) -> tuple:
    """Build the extraction prompt with custom types - returns (system_prompt, user_prompt)"""
    custom_prompt = os.getenv('GLOSSARY_SYSTEM_PROMPT', '').strip()
    
    # Replace {language} placeholder with target language
    target_language = os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English')
    if custom_prompt and '{language}' in custom_prompt:
        custom_prompt = custom_prompt.replace('{language}', target_language)

    # Build entries phrase from enabled custom entry types (manual prompt only)
    custom_types = get_custom_entry_types()
    def _entries_phrase(types_dict: dict) -> str:
        items = []
        for t_name, cfg in (types_dict or {}).items():
            if cfg is not None and not cfg.get('enabled', True):
                continue
            label = str(t_name).replace('_', ' ').strip()
            if not label:
                continue
            label = label[0].upper() + label[1:]
            items.append(label)
        if not items:
            return "entries"
        if len(items) == 1:
            return f"{items[0]} entries"
        if len(items) == 2:
            return f"{items[0]} & {items[1]} entries"
        return ", ".join(items[:-1]) + f", & {items[-1]} entries"

    entries_str = _entries_phrase(custom_types)

    # Build dynamic gender instruction based on which types have has_gender
    _gender_types = [t for t, cfg in custom_types.items()
                     if cfg.get('enabled', True) and cfg.get('has_gender', False)]
    if _gender_types:
        _gender_labels = ', '.join(_gender_types)
        _gender_instruction = (
            f"The gender column applies only to {_gender_labels} entries — determine gender from context clues, leave empty if insufficient.\n"
            f"For all other entry types, leave gender empty."
        )
    else:
        _gender_instruction = ""
    
    if not custom_prompt:
        # If no custom prompt, use the canonical default defined at module level.
        # GlossaryManager_GUI.py imports this same constant so there is only
        # one authoritative copy of the default prompt.
        custom_prompt = DEFAULT_GLOSSARY_PROMPT

    # Replace {entries} placeholder now that we have the enabled custom entry types
    custom_prompt = custom_prompt.replace('{entries}', entries_str)
    custom_prompt = custom_prompt.replace('{{entries}}', entries_str)

    # Replace {gender_instruction} with dynamic gender rule
    custom_prompt = custom_prompt.replace('{gender_instruction}', _gender_instruction)
    custom_prompt = _apply_subject_tracking_placeholder(custom_prompt)

    # Expand (or strip) the description-rule placeholders. Done BEFORE the
    # {fields}/{fields1} expansion so no subsequent regex or format op can
    # accidentally touch a stripped region.
    custom_prompt = _apply_description_rule_placeholders(custom_prompt)

    # Check if the prompt contains {fields} or {fields1} placeholders
    has_fields = '{fields}' in custom_prompt
    has_fields1 = '{fields1}' in custom_prompt
    if has_fields or has_fields1:
        # Get enabled types
        enabled_types = [(t, cfg) for t, cfg in custom_types.items() if cfg.get('enabled', True)]
        
        # Get custom fields
        custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
        try:
            custom_fields = json.loads(custom_fields_json)
        except:
            custom_fields = []
        
        # Build header parts (shared between {fields} and {fields1})
        header_parts = ['type', 'raw_name', 'translated_name', 'gender']
        if custom_fields:
            header_parts.extend(custom_fields)
        
        # List valid entry types
        type_names = [t[0] for t in enabled_types]

        # {fields1} → Unit Separator (\x1F) separated columns (new default)
        if has_fields1:
            fields1_spec = []
            _sep_joined = '\\x1F'.join(header_parts)
            fields1_spec.append(f"Columns (separated by Unit Separator character \\x1F):\n{_sep_joined}")
            if type_names:
                fields1_spec.append(f"Entry Types:\n{', '.join(type_names)}")
            fields1_str = '\\n'.join(fields1_spec)
            custom_prompt = custom_prompt.replace('{fields1}', fields1_str)

        # {fields} → comma-separated columns (legacy/backward compatible)
        if has_fields:
            if 'CSV' in custom_prompt.upper() or 'COMMA' in custom_prompt.upper() or 'SEPARATOR' in custom_prompt.upper():
                # CSV-style comma-separated list
                fields_spec = []
                fields_spec.append(f"Columns:\n{', '.join(header_parts)}")
                if type_names:
                    fields_spec.append(f"Entry Types:\n{', '.join(type_names)}")
                fields_str = '\\n'.join(fields_spec)
            else:
                # JSON format (fallback for prompts that don't mention CSV)
                fields_spec = []
                fields_spec.append("Extract entities and return as a JSON array.")
                fields_spec.append("Each entry must be a JSON object with these exact fields:")
                fields_spec.append("")
                
                for type_name, type_config in enabled_types:
                    fields_spec.append(f"For {type_name}s:")
                    fields_spec.append(f'  "type": "{type_name}" (required)')
                    fields_spec.append('  "raw_name": the name in original language/script (required)')
                    fields_spec.append('  "translated_name": English translation or romanization (required)')
                    if type_config.get('has_gender', False):
                        fields_spec.append(f'  "gender": "Male", "Female", or "Unknown" (required for {type_name} entries)')
                    fields_spec.append("")
                
                # Add custom fields info
                if custom_fields:
                    fields_spec.append("Additional custom fields to include:")
                    for field in custom_fields:
                        fields_spec.append(f'  "{field}": appropriate value')
                    fields_spec.append("")
                
                # Add example
                if enabled_types:
                    fields_spec.append("Example output format:")
                    fields_spec.append('[')
                    examples = []
                    # Generate an example for the first gender-enabled type
                    _gender_example_types = [t for t, cfg in enabled_types if cfg.get('has_gender', False)]
                    if _gender_example_types:
                        _get = _gender_example_types[0]
                        example = f'  {{"type": "{_get}", "raw_name": "田中太郎", "translated_name": "Tanaka Taro", "gender": "Male"'
                        for field in custom_fields:
                            example += f', "{field}": "example value"'
                        example += '}'
                        examples.append(example)
                    if 'term' in [t[0] for t in enabled_types]:
                        example = '  {"type": "term", "raw_name": "東京駅", "translated_name": "Tokyo Station"'
                        for field in custom_fields:
                            example += f', "{field}": "example value"'
                        example += '}'
                        examples.append(example)
                    fields_spec.append(',\n'.join(examples))
                    fields_spec.append(']')
                
                fields_str = '\n'.join(fields_spec)
            
            custom_prompt = custom_prompt.replace('{fields}', fields_str)
        
        system_prompt = custom_prompt
    else:
        # No {fields} or {fields1} placeholder - use the prompt as-is
        system_prompt = custom_prompt
    
    # Remove any {chapter_text} placeholders from system prompt
    system_prompt = system_prompt.replace('{chapter_text}', '')
    system_prompt = system_prompt.replace('{{chapter_text}}', '')
    system_prompt = system_prompt.replace('{text}', '')
    system_prompt = system_prompt.replace('{{text}}', '')
    
    # Strip any trailing "Text:" or similar
    system_prompt = system_prompt.rstrip()
    if system_prompt.endswith('Text:'):
        system_prompt = system_prompt[:-5].rstrip()
    
    # User prompt is just the chapter text
    user_prompt = chapter_text
    
    return (system_prompt, user_prompt)

DEFAULT_SINGLE_PASS_GLOSSARY_HEADER_PROMPT = """You have two tasks for this request.

1. Extract a translation glossary from the source text according to the glossary instructions.
2. Translate the source text using the generated glossary and the translation instructions.

Rules:
- Output the generated glossary first inside <glossary>...</glossary> tags.
- Inside <glossary>, follow the glossary prompt's requested schema, fields, and formatting exactly.
- Any glossary prompt instruction to return only CSV/JSON applies only inside the <glossary> block.
- After </glossary>, output only the translated content.
- The translation must use the generated glossary consistently while still following the active translation prompt.
- A response that contains only the glossary is invalid; the full translation after </glossary> is mandatory.
- Do not explain the process, mention these tasks, or add notes outside the required glossary block and translation.

[Glossary Prompt]
{glossary_prompt}

[Translation Prompt]
{translation_prompt}

[Final Single-Pass Override]
The final answer must contain exactly:
1. One <glossary>...</glossary> block.
2. The complete translated source text immediately after </glossary>.
Do not stop after the glossary."""

STALE_SINGLE_PASS_GLOSSARY_HEADER_MARKERS = (
    "Balanced/Full glossary logic",
    "standard Balanced/Full extraction",
)


def build_single_pass_translation_system_prompt(translation_prompt: str, source_text: str) -> str:
    """Build a combined glossary-extraction + translation system prompt.

    The glossary extraction half reuses build_prompt(), so custom fields,
    entry types, description rules, and Balanced/Full formatting stay aligned
    with the normal glossary pipeline.
    """
    glossary_system_prompt, _ = build_prompt(source_text or "")
    translation_prompt = translation_prompt or ""
    header = (os.getenv("SINGLE_PASS_GLOSSARY_HEADER_PROMPT", "") or "").strip()
    if not header or any(marker in header for marker in STALE_SINGLE_PASS_GLOSSARY_HEADER_MARKERS):
        header = DEFAULT_SINGLE_PASS_GLOSSARY_HEADER_PROMPT
    final_override = (
        "\n\n[Final Single-Pass Override]\n"
        "The final answer must contain exactly one <glossary>...</glossary> block followed immediately by the complete translated source text. "
        "Any instruction inside the glossary prompt to return only CSV/JSON applies only inside <glossary>; do not stop after the glossary."
    )
    if "{glossary_prompt}" in header or "{translation_prompt}" in header:
        combined = (
            header
            .replace("{glossary_prompt}", glossary_system_prompt)
            .replace("{translation_prompt}", translation_prompt)
        )
    else:
        combined = f"{header}\n\n[Balanced/Full Glossary Prompt]\n{glossary_system_prompt}\n\n[Translation Prompt]\n{translation_prompt}"
    if "[Final Single-Pass Override]" not in combined:
        combined += final_override
    return combined

def split_single_pass_response(response_text: str) -> tuple:
    """Split a single-pass response into (translation_text, glossary_block)."""
    if not isinstance(response_text, str):
        return response_text, ""

    def _extract_csv_glossary_block(text: str, search_start: int = 0):
        header_re = re.compile(
            r'(?im)^[ \t]*(?:```(?:csv)?[ \t]*)?type\s*(?:,|\t|\\x1[fF]|\x1f)\s*'
            r'raw_name\s*(?:,|\t|\\x1[fF]|\x1f)\s*translated_name\b.*$'
        )
        match = header_re.search(text, search_start)
        if not match:
            return None

        line_start = match.start()
        pos = line_start
        rows_seen = 0
        last_nonempty_end = match.end()
        enabled_types = set(get_custom_entry_types().keys()) or {'character', 'term', 'book'}
        enabled_types.update({'character', 'term', 'book'})
        type_re = re.compile(
            r'^\s*(?:' + '|'.join(re.escape(t) for t in sorted(enabled_types, key=len, reverse=True)) + r')\s*(?:,|\t|\\x1[fF]|\x1f)',
            re.IGNORECASE
        )

        while pos < len(text):
            next_nl = text.find('\n', pos)
            line_end = len(text) if next_nl == -1 else next_nl
            line = text[pos:line_end]
            stripped = line.strip()

            if pos == line_start:
                pass
            elif not stripped:
                if rows_seen > 0:
                    break
            elif stripped.startswith('```') and rows_seen > 0:
                last_nonempty_end = line_end
                pos = len(text) if next_nl == -1 else next_nl + 1
                break
            elif type_re.match(stripped):
                rows_seen += 1
            elif rows_seen > 0:
                # Description fields sometimes wrap onto following lines. Keep
                # continuation lines until a blank line or fence closes the block.
                pass
            else:
                break

            if stripped:
                last_nonempty_end = line_end
            if next_nl == -1:
                pos = len(text)
                break
            pos = next_nl + 1

        if rows_seen <= 0:
            return None
        block_end = last_nonempty_end
        block = text[line_start:block_end].strip()
        return line_start, block_end, block

    glossary_open_re = r"(?:<glossary\b[^>]*>|&lt;glossary\b.*?&gt;)"
    glossary_close_re = r"(?:</glossary>|&lt;/glossary&gt;)"
    lower_response = response_text.lower()
    if "<glossary" in lower_response or "&lt;glossary" in lower_response:
        match = re.search(
            glossary_open_re + r"(.*?)" + glossary_close_re,
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            glossary_block = (match.group(1) or "").strip()
            translation = (response_text[:match.start()] + response_text[match.end():]).strip()
            return translation, glossary_block

        open_match = re.search(glossary_open_re, response_text, flags=re.IGNORECASE | re.DOTALL)
        if open_match:
            fallback = _extract_csv_glossary_block(response_text, open_match.end())
            if fallback:
                start, end, glossary_block = fallback
                translation = (response_text[:open_match.start()] + response_text[end:]).strip()
                return translation, glossary_block
            glossary_block = response_text[open_match.end():].strip()
            translation = response_text[:open_match.start()].strip()
            return translation, glossary_block

    fallback = _extract_csv_glossary_block(response_text)
    if fallback:
        start, end, glossary_block = fallback
        translation = (response_text[:start] + response_text[end:]).strip()
        return translation, glossary_block

    return response_text, ""


# Cache the description-active flag at module level so it's evaluated once per
# process and stays consistent across all dedup calls within the same run.
_DEDUP_DESCRIPTION_ACTIVE = None

def _dedup_description_active_cached():
    global _DEDUP_DESCRIPTION_ACTIVE
    if _DEDUP_DESCRIPTION_ACTIVE is None:
        _DEDUP_DESCRIPTION_ACTIVE = _glossary_description_active()
    return bool(_DEDUP_DESCRIPTION_ACTIVE)


def _dedup_field_count(entry):
    """Count non-empty, non-internal fields for dedup comparison.

    Excludes:
    - Keys starting with ``_`` (internal bookkeeping like ``_section``)
    - ``description`` (case-insensitive) when the user has NOT enabled
      description in their Custom Fields.  This prevents a leaked AI
      description from giving a newer entry an unfair field-count advantage
      over a manually-reviewed older entry.
    """
    count = 0
    for k, v in entry.items():
        k_str = str(k)
        if k_str.startswith("_"):
            continue
        if not _dedup_description_active_cached() and k_str.strip().lower() == 'description':
            continue
        if v and str(v).strip():
            count += 1
    return count


def _dedup_preference_score(entry):
    """Return the score used to decide whether a duplicate can replace an older row.

    When the user opted into description, richer rows can contribute newer
    description/custom-column data. Without description, dedup is intentionally
    conservative: only useful gender information can make a newer duplicate
    contribute data.
    """
    if _dedup_description_active_cached():
        return _dedup_field_count(entry)
    return 1 if _gender_is_trackable(_entry_gender(entry)) else 0


def _dedup_preference_label():
    return "fields" if _dedup_description_active_cached() else "gender score"


def _dedup_preference_reason(current_score, existing_score):
    if _dedup_description_active_cached():
        return f"richer entry ({current_score} vs {existing_score} fields)"
    return "adds gender to existing duplicate"


def _dedup_replacement_entry(existing_entry, new_entry):
    """Merge replacement data while preserving the oldest visible name columns.

    Dedup may absorb newer description/custom-field or gender data, but the
    first kept raw/translated names remain authoritative because they may have
    been manually curated.
    """
    replacement = dict(new_entry)
    for field in ("raw_name", "translated_name"):
        if field in existing_entry:
            replacement[field] = existing_entry.get(field, "")
    return replacement


def skip_duplicate_entries(glossary, dry_run=False, output_dir=None):
    """
    Skip entries with duplicate raw names and translated names using 2-pass deduplication.
    
    Pass 1: Remove entries with similar raw names (fuzzy matching)
    Pass 2: Remove entries with identical translated names (exact matching)
    
    Args:
        glossary: List of entry dicts with 'raw_name', 'translated_name', etc.
        dry_run: If True, return (original_entries, dedup_log) without modifying the list.
        output_dir: Accepted for caller compatibility; no report file is written.
    
    Returns:
        list: Deduplicated entries (when dry_run=False)
        tuple: (original_entries, dedup_log) when dry_run=True
    """
    # Try to use RapidFuzz for speed, fallback to difflib
    # Reset the description-active cache so it picks up the current setting
    global _DEDUP_DESCRIPTION_ACTIVE
    _DEDUP_DESCRIPTION_ACTIVE = None
    try:
        from rapidfuzz import fuzz
        use_rapidfuzz = True
    except ImportError:
        import difflib
        use_rapidfuzz = False
    
    # Get configuration
    fuzzy_threshold = float(os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.9'))
    # GLOSSARY_DEDUPE_TRANSLATIONS: "1" = enable Pass 2 (remove entries with identical translations)
    #                              : "0" = disable Pass 2 (only remove entries with similar raw names)
    dedupe_translations = os.getenv('GLOSSARY_DEDUPE_TRANSLATIONS', '1') == '1'
    
    # Structured dedup log — records all decisions for auditing
    dedup_log = []
    
    original_count = len(glossary)
    print(f"[Dedup] Starting 2-pass deduplication with {original_count} entries...")
    
    # Show which algorithm mode is configured
    algo_mode = os.getenv('GLOSSARY_DUPLICATE_ALGORITHM', 'auto')
    use_advanced = os.getenv('GLOSSARY_USE_ADVANCED_DETECTION', '1') == '1'
    if use_advanced:
        print(f"[Dedup] Algorithm Mode: {algo_mode.upper()} (Multi-algorithm detection enabled)")
    else:
        print(f"[Dedup] Algorithm Mode: BASIC (Advanced detection disabled)")
    
    # Show which method we're using
    if use_rapidfuzz:
        print(f"[Dedup] Using RapidFuzz (C++ speed) with threshold {fuzzy_threshold:.2f}")
    else:
        print(f"[Dedup] Using difflib (fallback) with threshold {fuzzy_threshold:.2f}")
    
    if dedupe_translations:
        print(f"[Dedup] Pass 2 (translated name deduplication): ENABLED")
    else:
        print(f"[Dedup] Pass 2 (translated name deduplication): DISABLED")
    
    # PASS 1: Raw name deduplication
    print(f"[Dedup] 🔄 PASS 1: Raw name deduplication...")
    pass1_stats = {"missing_raw_name_count": 0}
    pass1_results = _skip_raw_name_duplicates(
        glossary, fuzzy_threshold, use_rapidfuzz, dedup_log, pass1_stats
    )
    pass1_removed = original_count - len(pass1_results)
    missing_raw_name_count = int(pass1_stats.get("missing_raw_name_count", 0) or 0)
    pass1_duplicates_removed = max(0, pass1_removed - missing_raw_name_count)
    if missing_raw_name_count:
        missing_entry_label = "entry" if missing_raw_name_count == 1 else "entries"
        print(
            f"[Dedup] ✅ PASS 1 complete: {pass1_duplicates_removed} duplicates removed, "
            f"{missing_raw_name_count} {missing_entry_label} skipped due to missing raw_name "
            f"({len(pass1_results)} remaining)"
        )
    else:
        print(f"[Dedup] ✅ PASS 1 complete: {pass1_removed} duplicates removed ({len(pass1_results)} remaining)")
    
    # PASS 2: Translated name deduplication (if enabled)
    if dedupe_translations:
        print(f"[Dedup] 🔄 PASS 2: Translated name deduplication...")
        final_results = _skip_translated_name_duplicates(pass1_results, dedup_log)
    else:
        final_results = pass1_results
        print(f"[Dedup] ⏭️ PASS 2 skipped (translation deduplication disabled)")
    
    total_removed = original_count - len(final_results)
    final_results = _harmonize_gender_variant_translations(_harmonize_alias_name_translations(final_results))
    print(f"[Dedup] ✨ Deduplication complete: {total_removed} total duplicates removed, {len(final_results)} unique entries kept")
    
    if dry_run:
        return glossary, dedup_log
    
    return final_results


def _skip_raw_name_duplicates(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log=None, stats=None):
    """Pass 1: Remove entries with similar raw names using optimized serial processing"""
    # Note: Parallel processing doesn't work well for deduplication because:
    # 1. Order matters - can't determine if A is duplicate of B until we've processed A
    # 2. The "seen" list changes as we process, making parallelization complex
    # 3. The serial version with RapidFuzz batch processing is already very fast
    
    # Use optimized serial version for all sizes
    return _skip_raw_name_duplicates_serial(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log, stats)


def _skip_raw_name_duplicates_matrix(glossary, fuzzy_threshold):
    """Ultra-fast matrix-based deduplication for large datasets.
    
    Uses config-driven similarity scoring and compares entries within
    adjacent length buckets to avoid missing near-boundary pairs.
    """
    from duplicate_detection_config import calculate_similarity_with_config, get_duplicate_detection_config
    config = get_duplicate_detection_config()
    
    print(f"[Dedup] Using matrix-based deduplication (optimized for {len(glossary)} entries)")
    
    # Pre-process all entries with Unicode normalization
    processed = []
    for entry in glossary:
        raw_name = entry.get('raw_name', '')
        if raw_name:
            cleaned_name = unicodedata.normalize('NFC', remove_honorifics(raw_name))
            processed.append((entry, raw_name, cleaned_name))
    
    if not processed:
        return []
    
    # Extract just the cleaned names for comparison
    cleaned_names = [cleaned.lower() for _, _, cleaned in processed]
    
    print(f"[Dedup] Building similarity groups...")
    
    # Group by length bucket for faster comparison (only compare similar lengths)
    length_buckets = {}
    for idx, (entry, raw, cleaned) in enumerate(processed):
        length = len(cleaned)
        bucket = length // 3  # Group by length/3 (e.g., 6-8 chars in same bucket)
        if bucket not in length_buckets:
            length_buckets[bucket] = []
        length_buckets[bucket].append(idx)
    
    # Track duplicates
    is_duplicate = [False] * len(processed)
    duplicate_of = [None] * len(processed)  # Index of the entry this is a duplicate of
    
    total_comparisons = 0
    
    # Build unique bucket pairs to compare: each bucket with itself AND adjacent buckets (±1).
    # This catches near-boundary misses (e.g., 11-char vs 12-char names in different buckets).
    sorted_buckets = sorted(length_buckets.keys())
    bucket_pairs_done = set()
    
    for bucket in sorted_buckets:
        for neighbor in (bucket, bucket + 1):
            if neighbor not in length_buckets:
                continue
            pair_key = (min(bucket, neighbor), max(bucket, neighbor))
            if pair_key in bucket_pairs_done:
                continue
            bucket_pairs_done.add(pair_key)
            
            indices_a = length_buckets[bucket]
            indices_b = length_buckets[neighbor]
            
            if bucket == neighbor:
                # Intra-bucket: compare all pairs within same bucket
                for i, idx1 in enumerate(indices_a):
                    if is_duplicate[idx1]:
                        continue
                    for j in range(i + 1, len(indices_a)):
                        idx2 = indices_a[j]
                        if is_duplicate[idx2]:
                            continue
                        
                        compare_config = config
                        if _partial_ratio_gender_only() and not (
                            _entry_type_has_active_gender(processed[idx1][0])
                            and _entry_type_has_active_gender(processed[idx2][0])
                        ):
                            compare_config = dict(config)
                            compare_config["algorithms"] = [a for a in config.get("algorithms", []) if a != "partial"]
                        score = calculate_similarity_with_config(
                            cleaned_names[idx1], cleaned_names[idx2], compare_config
                        )
                        total_comparisons += 1
                        
                        if score >= fuzzy_threshold:
                            is_duplicate[idx2] = True
                            duplicate_of[idx2] = idx1
            else:
                # Cross-bucket: compare entries from adjacent buckets
                for idx1 in indices_a:
                    if is_duplicate[idx1]:
                        continue
                    for idx2 in indices_b:
                        if is_duplicate[idx2]:
                            continue
                        
                        compare_config = config
                        if _partial_ratio_gender_only() and not (
                            _entry_type_has_active_gender(processed[idx1][0])
                            and _entry_type_has_active_gender(processed[idx2][0])
                        ):
                            compare_config = dict(config)
                            compare_config["algorithms"] = [a for a in config.get("algorithms", []) if a != "partial"]
                        score = calculate_similarity_with_config(
                            cleaned_names[idx1], cleaned_names[idx2], compare_config
                        )
                        total_comparisons += 1
                        
                        if score >= fuzzy_threshold:
                            # Mark the later entry as duplicate
                            if idx2 > idx1:
                                is_duplicate[idx2] = True
                                duplicate_of[idx2] = idx1
                            else:
                                is_duplicate[idx1] = True
                                duplicate_of[idx1] = idx2
        
        if total_comparisons % 1000 == 0 and total_comparisons > 0:
            print(f"[Dedup] Processed {total_comparisons} comparisons...")
    
    print(f"[Dedup] Completed {total_comparisons} comparisons (reduced from {len(processed) * (len(processed)-1) // 2})")
    
    # Build deduplicated list with the same preference rules as the serial path.
    deduplicated = []
    raw_name_to_idx = {}
    skipped_count = 0
    replaced_count = 0
    
    for idx, (entry, raw_name, cleaned_name) in enumerate(processed):
        if is_duplicate[idx]:
            original_idx = duplicate_of[idx]
            original_raw = processed[original_idx][1]
            
            # Check whether the duplicate should replace the original.
            existing_idx = raw_name_to_idx.get(original_raw)
            if existing_idx is not None:
                existing_entry = deduplicated[existing_idx]
                current_score = _dedup_preference_score(entry)
                existing_score = _dedup_preference_score(existing_entry)
                
                if current_score > existing_score:
                    replacement_entry = _dedup_replacement_entry(existing_entry, entry)
                    replacement_raw = replacement_entry.get('raw_name', raw_name)
                    deduplicated[existing_idx] = replacement_entry
                    raw_name_to_idx[replacement_raw] = existing_idx
                    if original_raw in raw_name_to_idx and original_raw != replacement_raw:
                        del raw_name_to_idx[original_raw]
                    replaced_count += 1
                    
            skipped_count += 1
        else:
            raw_name_to_idx[raw_name] = len(deduplicated)
            deduplicated.append(entry)
    
    print(f"[Dedup] ✅ Matrix deduplication complete: {skipped_count} duplicates removed ({replaced_count} replaced), {len(deduplicated)} remaining")
    return deduplicated

def _parse_token_efficient_glossary(text: str) -> List[Dict]:
    """
    Parse token-efficient glossary text (section headers + bullet lines) into entry dicts.
    Supports custom entry types and arbitrary extra columns from the header line.
    """
    lines = text.splitlines()
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if not (first.lower().startswith("glossary columns:") or first.startswith("===") or first.startswith("* ")):
        return []

    header_cols = ['raw_name', 'translated_name', 'gender', 'description']
    section_re = re.compile(r"^===\s*(.+?)\s*===\s*$")
    entries: List[Dict] = []
    current_section = None

    # Map section names back to type using enabled custom types (with plural tolerance)
    custom_types = get_custom_entry_types()
    type_map = {}
    for t in custom_types.keys():
        type_map[t.lower()] = t
        if not t.lower().endswith('s'):
            type_map[f"{t.lower()}s"] = t

    def _split_custom_field_parts(field_text: str, known_fields: list) -> dict:
        values = {}
        if not field_text or not known_fields:
            return values
        known_by_lower = {str(f).strip().lower(): str(f).strip() for f in known_fields if str(f).strip()}
        for part in re.split(r",\s*(?=[^,]+?:)", field_text):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key_l = key.strip().lower()
            if key_l in known_by_lower:
                values[known_by_lower[key_l]] = value.strip()
        return values

    def _parse_token_line(line: str, extra_cols: list):
        body = line[2:].strip()
        custom_values = {}

        def _split_head_desc(text):
            paren_depth = 0
            bracket_depth = 0
            for idx, ch in enumerate(text):
                if ch == '(' and bracket_depth == 0:
                    paren_depth += 1
                elif ch == ')' and bracket_depth == 0 and paren_depth > 0:
                    paren_depth -= 1
                elif ch == '[' and paren_depth == 0:
                    bracket_depth += 1
                elif ch == ']' and paren_depth == 0 and bracket_depth > 0:
                    bracket_depth -= 1
                elif ch == ':' and paren_depth == 0 and bracket_depth == 0:
                    return text[:idx].rstrip(), text[idx + 1:].strip()
            return text, ""

        head, desc = _split_head_desc(body)

        def _pull_custom_tails(text):
            while True:
                custom_tail = re.search(r"\s+\(([^()]*)\)\s*$", text)
                if not custom_tail:
                    return text.rstrip()
                parsed_custom = _split_custom_field_parts(custom_tail.group(1), extra_cols)
                if not parsed_custom:
                    return text.rstrip()
                custom_values.update(parsed_custom)
                text = text[:custom_tail.start()].rstrip()

        head = _pull_custom_tails(head)
        gender = ""
        gender_match = re.search(r"\s*\[([^\]]*)\]\s*$", head)
        if gender_match:
            gender = (gender_match.group(1) or "").strip()
            head = head[:gender_match.start()].rstrip()
            head = _pull_custom_tails(head)

        equal_match = re.match(r"^(?P<raw>.+?)\s*=\s*(?P<translated>.+?)\s*$", head)
        if equal_match:
            raw_name = (equal_match.group("raw") or "").strip()
            translated = (equal_match.group("translated") or "").strip()
        else:
            name_match = re.match(r"^(?P<translated>.*)\s+\((?P<raw>.*?)\)\s*$", head)
            if not name_match:
                return None
            translated = (name_match.group("translated") or "").strip()
            raw_name = (name_match.group("raw") or "").strip()

        # Writer form with description plus custom fields:
        #   : description text (Field: value)
        if desc and extra_cols:
            desc_custom_tail = re.search(r"\s+\(([^()]*)\)\s*$", desc)
            if desc_custom_tail:
                parsed_custom = _split_custom_field_parts(desc_custom_tail.group(1), extra_cols)
                if parsed_custom:
                    custom_values.update(parsed_custom)
                    desc = desc[:desc_custom_tail.start()].strip()

        return translated, raw_name, gender, desc, custom_values

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Header line
        if line.lower().startswith("glossary columns:"):
            cols_text = line.split(":", 1)[1]
            header_cols = [c.strip() for c in cols_text.split(",") if c.strip()]
            if not header_cols:
                header_cols = ['raw_name', 'translated_name', 'gender', 'description']
            continue

        # Section line
        m_sec = section_re.match(line)
        if m_sec:
            sec = m_sec.group(1).strip()
            current_section = sec
            continue

        # Entry line
        if not line.startswith("* "):
            continue

        standard_cols = {"translated_name", "raw_name", "gender", "description"}
        extra_cols = [c for c in header_cols if str(c).strip().lower() not in standard_cols]
        parsed_line = _parse_token_line(line, extra_cols)
        if not parsed_line:
            continue
        translated, raw_name, gender, desc, custom_values = parsed_line

        entry = {
            "type": type_map.get((current_section or "").lower(), type_map.get("terms", "terms")),
            "raw_name": raw_name,
            "translated_name": translated,
        }
        if gender:
            entry["gender"] = gender
        if desc:
            entry["description"] = desc
        for k, v in custom_values.items():
            if v:
                entry[k] = v

        entries.append(entry)

    return entries


def _load_glossary_file(path: str) -> List[Dict]:
    """
    Load a glossary file that may be in token-efficient or legacy CSV format.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith(".json"):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    print(f"📂 Loaded JSON glossary: {len(data)} entries")
                    return [entry for entry in data if isinstance(entry, dict)]
                if isinstance(data, dict) and isinstance(data.get("glossary"), list):
                    entries = [entry for entry in data.get("glossary", []) if isinstance(entry, dict)]
                    print(f"📂 Loaded JSON glossary: {len(entries)} entries")
                    return entries
            except Exception:
                pass
        token_entries = _parse_token_efficient_glossary(text)
        if token_entries:
            print(f"📂 Loaded token-efficient glossary: {len(token_entries)} entries")
            return token_entries
        # Legacy CSV
        import csv
        rows = []
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            rows.append(row)
        print(f"📂 Loaded legacy CSV glossary: {len(rows)} entries")
        return rows
    except Exception as e:
        print(f"⚠️ Could not load glossary file {path}: {e}")
        return []

def _dedupe_worker_chunk(chunk_items, all_items, fuzzy_threshold, use_rapidfuzz):
    """Process a chunk of items in parallel - reduces memory overhead"""
    results = []
    for item in chunk_items:
        result = _dedupe_worker(item, all_items, fuzzy_threshold, use_rapidfuzz)
        results.append(result)
    return results


def _dedupe_worker(item, all_items, fuzzy_threshold, use_rapidfuzz):
    """Worker function for ProcessPoolExecutor - process single item with fast pruning"""
    entry, raw_name, cleaned_name = item
    name_len = len(cleaned_name)
    name_lower = cleaned_name.lower()
    
    # Build candidates list - no filtering for maximum quality
    candidates = []
    for other_entry, other_raw, other_cleaned in all_items:
        if other_raw == raw_name:  # Exclude self
            continue
        
        candidates.append((other_cleaned, other_raw, other_entry))
    
    if not candidates:
        return (entry, raw_name, cleaned_name, False, 0.0, None)
    
    is_dup, best_score, best_match = _find_best_duplicate_match(
        cleaned_name, candidates, fuzzy_threshold, use_rapidfuzz, entry
    )
    
    return (entry, raw_name, cleaned_name, is_dup, best_score, best_match)


def _find_best_duplicate_match(cleaned_name, seen_raw_names, fuzzy_threshold, use_rapidfuzz, current_entry=None,
                                _config=None, _partial_gender_only=None, _config_no_partial=None,
                                _seen_lower_names=None):
    """Find the best duplicate match using multi-algorithm fuzzy matching.
    
    Performance: when RapidFuzz is available, uses C++ batch comparison
    (process.extract) as a pre-filter, reducing full multi-algorithm
    evaluations from O(n) per call to ~20 candidates.
    """
    if not seen_raw_names:
        return (False, 0.0, None)
    
    name_lower = cleaned_name.lower()
    
    # Use advanced multi-algorithm detection if configured
    use_advanced = os.getenv('GLOSSARY_USE_ADVANCED_DETECTION', '1') == '1'
    
    if use_advanced:
        try:
            from duplicate_detection_config import calculate_similarity_with_config, get_duplicate_detection_config
            config = _config if _config is not None else get_duplicate_detection_config()
            partial_gender_only = _partial_gender_only if _partial_gender_only is not None else _partial_ratio_gender_only()
            config_no_partial = _config_no_partial
            
            best_score = 0.0
            best_match = None
            
            # ── Fast path: RapidFuzz C++ batch pre-filter ──
            # Instead of a Python for-loop calling calculate_similarity_with_config
            # N times per entry (O(n²) total), use a single C++ batch call to find
            # the top candidates, then run full multi-algorithm scoring only on those.
            if use_rapidfuzz and len(seen_raw_names) > 10:
                from rapidfuzz import fuzz as rf_fuzz, process as rf_process
                
                # Use pre-built lowercase list if caller maintains one
                candidate_names = _seen_lower_names if _seen_lower_names is not None else [
                    item[0].lower() for item in seen_raw_names
                ]
                
                # Pre-filter cutoff: generous margin below threshold so we don't
                # miss candidates where token_sort or jaro_winkler scores higher
                prefilter_pct = max((fuzzy_threshold - 0.15) * 100, 50)
                
                # Two fast C++ batch calls to cover different matching dimensions:
                # - ratio: character-level similarity (catches join/split variations)
                # - token_sort_ratio: word-reorder matches (e.g. "Park Jisoo" / "Jisoo Park")
                # Both run in C++ — two calls are still orders of magnitude faster
                # than the old Python loop over all candidates.
                candidate_indices = set()
                for scorer in (rf_fuzz.ratio, rf_fuzz.token_sort_ratio):
                    hits = rf_process.extract(
                        name_lower, candidate_names,
                        scorer=scorer,
                        score_cutoff=prefilter_pct,
                        limit=20
                    )
                    for _, _, idx in (hits or []):
                        candidate_indices.add(idx)
                
                if not candidate_indices:
                    return (False, 0.0, None)
                
                # Full multi-algorithm scoring only on pre-filtered candidates
                current_has_gender = _entry_type_has_active_gender(current_entry) if current_entry else False
                for idx in candidate_indices:
                    seen_item = seen_raw_names[idx]
                    seen_clean = seen_item[0]
                    seen_original = seen_item[1]
                    seen_entry = seen_item[2] if len(seen_item) > 2 else None
                    
                    cmp_config = config
                    if partial_gender_only and not (
                        current_has_gender and _entry_type_has_active_gender(seen_entry)
                    ):
                        cmp_config = config_no_partial if config_no_partial is not None else config
                    
                    score = calculate_similarity_with_config(cleaned_name, seen_clean, cmp_config)
                    if score >= fuzzy_threshold and score > best_score:
                        best_score = score
                        best_match = seen_original
                
                return (best_score >= fuzzy_threshold, best_score, best_match)
            
            # ── Small list fallback: Python loop is acceptable for ≤10 items ──
            for seen_item in seen_raw_names:
                seen_clean = seen_item[0]
                seen_original = seen_item[1]
                seen_entry = seen_item[2] if len(seen_item) > 2 else None
                compare_config = config
                if partial_gender_only and not (
                    _entry_type_has_active_gender(current_entry)
                    and _entry_type_has_active_gender(seen_entry)
                ):
                    if config_no_partial is not None:
                        compare_config = config_no_partial
                    else:
                        compare_config = dict(config)
                        compare_config["algorithms"] = [a for a in config.get("algorithms", []) if a != "partial"]
                score = calculate_similarity_with_config(cleaned_name, seen_clean, compare_config)
                
                if score >= fuzzy_threshold and score > best_score:
                    best_score = score
                    best_match = seen_original
            
            return (best_score >= fuzzy_threshold, best_score, best_match)
        except ImportError:
            # Fallback to basic if advanced module not available
            pass
    
    # Basic mode (original logic)
    if use_rapidfuzz:
        from rapidfuzz import fuzz, process
        
        # For large candidate lists, use batch processing (much faster)
        if len(seen_raw_names) > 50:
            # Use pre-built lowercase list if caller maintains one
            candidate_names = _seen_lower_names if _seen_lower_names is not None else [
                seen_item[0].lower() for seen_item in seen_raw_names
            ]
            
            # Use extractOne for fast best-match finding
            result = process.extractOne(
                name_lower, 
                candidate_names, 
                scorer=fuzz.ratio,
                score_cutoff=fuzzy_threshold * 100
            )
            
            if result:
                matched_name, score, idx = result
                best_score = score / 100.0
                best_match = seen_raw_names[idx][1]  # Get original name
                return (True, best_score, best_match)
            return (False, 0.0, None)
        
        # For smaller lists, use simple loop (less overhead)
        best_score = 0.0
        best_match = None
        
        for seen_item in seen_raw_names:
            seen_clean = seen_item[0]
            seen_original = seen_item[1]
            score = fuzz.ratio(name_lower, seen_clean.lower()) / 100.0
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_match = seen_original
        
        return (best_score >= fuzzy_threshold, best_score, best_match)
    
    else:
        # Fallback to difflib (slower)
        import difflib
        best_score = 0.0
        best_match = None
        
        for seen_item in seen_raw_names:
            seen_clean = seen_item[0]
            seen_original = seen_item[1]
            score = difflib.SequenceMatcher(None, name_lower, seen_clean.lower()).ratio()
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_match = seen_original
        
        return (best_score >= fuzzy_threshold, best_score, best_match)


def _skip_raw_name_duplicates_serial(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log=None, stats=None):
    """Serial version of Pass 1: raw name fuzzy deduplication.
    
    Fixes:
    - Stale seen_raw_names: when replacing an entry, update the seen list so
      future fuzzy comparisons use the current cleaned name.
    - Unicode normalization: NFC-normalizes cleaned names before comparison.
    - Structured logging: appends decisions to dedup_log list.
    """
    if use_rapidfuzz:
        from rapidfuzz import fuzz
    else:
        import difflib
    
    seen_raw_names = []  # List of (cleaned_name, original_raw_name) tuples
    seen_lower_names = []  # Pre-lowered names for C++ batch comparison
    raw_name_to_indices = {}  # raw_name -> list of indices in deduplicated
    # Reverse map: for a given index in deduplicated, which index in seen_raw_names?
    dedup_idx_to_seen_idx = {}
    deduplicated = []
    skipped_count = 0
    
    # ── Cache configuration once before the hot loop ──
    # Avoids millions of os.getenv / dict construction calls inside _find_best_duplicate_match
    _cached_config = None
    _cached_config_no_partial = None
    _cached_partial_gender_only = None
    try:
        from duplicate_detection_config import get_duplicate_detection_config
        _cached_config = get_duplicate_detection_config()
        _cached_config_no_partial = dict(_cached_config)
        _cached_config_no_partial["algorithms"] = [
            a for a in _cached_config.get("algorithms", []) if a != "partial"
        ]
        _cached_partial_gender_only = _partial_ratio_gender_only()
    except ImportError:
        pass
    _cached_alias_enabled = _alias_aware_name_matching_enabled()
    
    for entry in glossary:
        # Get raw_name and clean it
        raw_name = entry.get('raw_name', '')
        if not raw_name:
            if stats is not None:
                stats["missing_raw_name_count"] = stats.get("missing_raw_name_count", 0) + 1
            continue
            
        # Remove honorifics + NFC normalize for comparison (unless disabled)
        cleaned_name = unicodedata.normalize('NFC', remove_honorifics(raw_name))
        
        exact_indices = raw_name_to_indices.get(raw_name, [])
        if exact_indices:
            same_gender_idx = None
            variant_idx = None
            current_gender = _entry_gender(entry)
            for candidate_idx in exact_indices:
                candidate = deduplicated[candidate_idx]
                if _entry_gender(candidate) == current_gender:
                    same_gender_idx = candidate_idx
                    break
                if _is_exact_raw_gender_variant(candidate, entry):
                    variant_idx = candidate_idx

            if same_gender_idx is None and variant_idx is not None:
                _align_gender_variant_translation(deduplicated[variant_idx], entry)
                deduplicated.append(entry)
                raw_name_to_indices.setdefault(raw_name, []).append(len(deduplicated) - 1)
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 1,
                        "action": "kept_gender_variant",
                        "kept": raw_name,
                        "gender": current_gender,
                        "reason": "exact raw_name match with different gender",
                    })
                continue

            existing_index = same_gender_idx if same_gender_idx is not None else exact_indices[0]
            existing_entry = deduplicated[existing_index]
            current_score = _dedup_preference_score(entry)
            existing_score = _dedup_preference_score(existing_entry)
            if current_score > existing_score:
                deduplicated[existing_index] = _dedup_replacement_entry(existing_entry, entry)
                skipped_count += 1
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 1, "action": "replaced",
                        "kept": raw_name, "dropped": raw_name,
                        "score": 1.0,
                        "reason": _dedup_preference_reason(current_score, existing_score)
                    })
            else:
                skipped_count += 1
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 1, "action": "dropped",
                        "kept": raw_name, "dropped": raw_name,
                        "score": 1.0,
                        "reason": "exact raw duplicate"
                    })
            continue

        # Check for fuzzy matches with seen names
        alias_matched = False
        if _cached_alias_enabled and _alias_entry_allowed(entry):
            for existing_idx, existing_entry in enumerate(deduplicated):
                if not _alias_entry_allowed(existing_entry):
                    continue
                if _align_alias_variant_translation(existing_entry, entry):
                    alias_matched = True
                    if dedup_log is not None:
                        dedup_log.append({
                            "pass": 1,
                            "action": "kept_alias_variant",
                            "kept": entry.get("raw_name", ""),
                            "matched": existing_entry.get("raw_name", ""),
                            "reason": "alias-aware raw-name containment",
                        })
                    break
            if alias_matched:
                seen_idx = len(seen_raw_names)
                seen_raw_names.append((cleaned_name, raw_name, entry))
                seen_lower_names.append(cleaned_name.lower())
                dedup_idx = len(deduplicated)
                raw_name_to_indices.setdefault(raw_name, []).append(dedup_idx)
                dedup_idx_to_seen_idx[dedup_idx] = seen_idx
                deduplicated.append(entry)
                continue

        is_duplicate, best_score, best_match = _find_best_duplicate_match(
            cleaned_name, seen_raw_names, fuzzy_threshold, use_rapidfuzz, entry,
            _config=_cached_config, _partial_gender_only=_cached_partial_gender_only,
            _config_no_partial=_cached_config_no_partial, _seen_lower_names=seen_lower_names
        )
        
        if is_duplicate:
            # Use O(1) dictionary lookup
            best_indices = raw_name_to_indices.get(best_match, [])
            existing_index = best_indices[0] if best_indices else None
            
            if existing_index is not None:
                existing_entry = deduplicated[existing_index]
                current_score = _dedup_preference_score(entry)
                existing_score = _dedup_preference_score(existing_entry)
                
                # Replace only when the current entry is preferred by active dedup rules.
                if current_score > existing_score:
                    # Replace existing entry in deduplicated list
                    replacement_entry = _dedup_replacement_entry(existing_entry, entry)
                    replacement_raw = replacement_entry.get('raw_name', raw_name)
                    replacement_cleaned = unicodedata.normalize('NFC', remove_honorifics(replacement_raw))
                    deduplicated[existing_index] = replacement_entry
                    if replacement_raw != best_match:
                        raw_name_to_indices[replacement_raw] = [existing_index]
                        del raw_name_to_indices[best_match]
                    elif replacement_raw not in raw_name_to_indices:
                        raw_name_to_indices[replacement_raw] = [existing_index]
                    # FIX: Update seen_raw_names at the correct index so future
                    # fuzzy comparisons use the kept entry's cleaned name / raw name.
                    seen_idx = dedup_idx_to_seen_idx.get(existing_index)
                    if seen_idx is not None:
                        seen_raw_names[seen_idx] = (replacement_cleaned, replacement_raw, replacement_entry)
                        seen_lower_names[seen_idx] = replacement_cleaned.lower()
                    skipped_count += 1
                    if skipped_count <= 10:
                        unit = _dedup_preference_label()
                        print(f"[Skip] Pass 1: Merging fields from {raw_name} ({current_score} {unit}) into {replacement_raw} ({existing_score} {unit}) - {best_score*100:.1f}% match, preserving original names")
                    if dedup_log is not None:
                        dedup_log.append({
                            "pass": 1, "action": "replaced",
                            "kept": replacement_raw, "dropped": raw_name,
                            "score": round(best_score, 4),
                            "reason": _dedup_preference_reason(current_score, existing_score)
                        })
                else:
                    # Keep existing entry
                    skipped_count += 1
                    if dedup_log is not None:
                        dedup_log.append({
                            "pass": 1, "action": "dropped",
                            "kept": best_match, "dropped": raw_name,
                            "score": round(best_score, 4),
                            "reason": "duplicate"
                        })
            else:
                # Fallback if we can't find the existing entry in the index
                skipped_count += 1
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 1, "action": "dropped",
                        "kept": best_match or "?", "dropped": raw_name,
                        "score": round(best_score, 4),
                        "reason": "duplicate (index miss)"
                    })
        else:
            # Add to seen list and keep the entry
            seen_idx = len(seen_raw_names)
            seen_raw_names.append((cleaned_name, raw_name, entry))
            seen_lower_names.append(cleaned_name.lower())
            dedup_idx = len(deduplicated)
            raw_name_to_indices.setdefault(raw_name, []).append(dedup_idx)
            dedup_idx_to_seen_idx[dedup_idx] = seen_idx
            deduplicated.append(entry)
    
    return deduplicated


def _skip_translated_name_duplicates(glossary, dedup_log=None):
    """Pass 2: Remove entries with identical translated names (optimized with indexing)"""
    seen_translations = {}  # translated_name.lower() -> (raw_name, entry, index_in_deduplicated)
    deduplicated = []
    skipped_count = 0
    replaced_count = 0
    
    for entry in glossary:
        raw_name = entry.get('raw_name', '')
        translated_name = entry.get('translated_name', '')
        if not translated_name:
            deduplicated.append(entry)
            continue
        # Pre-compute normalized key once
        translated_lower = translated_name.lower().strip()
        if not translated_lower:
            deduplicated.append(entry)
            continue
        
        # Check if we've seen this translation before
        if translated_lower in seen_translations:
            existing_raw, existing_entry, existing_idx = seen_translations[translated_lower]
            existing_translated = existing_entry.get('translated_name', translated_name)
            if _is_exact_raw_gender_variant(existing_entry, entry):
                _align_gender_variant_translation(existing_entry, entry)
                deduplicated.append(entry)
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 2,
                        "action": "kept_gender_variant",
                        "kept": raw_name,
                        "translation": translated_name,
                        "reason": "same raw_name and translation but different gender",
                    })
                continue
            
            current_score = _dedup_preference_score(entry)
            existing_score = _dedup_preference_score(existing_entry)
            
            # Replace only when the current entry is preferred by active dedup rules.
            if current_score > existing_score:
                replacement_entry = _dedup_replacement_entry(existing_entry, entry)
                replacement_raw = replacement_entry.get('raw_name', raw_name)
                replacement_translated = replacement_entry.get('translated_name', translated_name)
                # Replace in-place using the stored index (faster than list comprehension)
                deduplicated[existing_idx] = replacement_entry
                # Update tracking with new index
                seen_translations[translated_lower] = (replacement_raw, replacement_entry, existing_idx)
                replaced_count += 1
                skipped_count += 1
                if skipped_count <= 10:
                    unit = _dedup_preference_label()
                    print(f"[Skip] Pass 2: Merging fields from '{raw_name}' -> '{translated_name}' ({current_score} {unit}) into '{replacement_raw}' -> '{replacement_translated}' ({existing_score} {unit}), preserving original names")
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 2, "action": "replaced",
                        "kept": replacement_raw, "dropped": raw_name,
                        "translation": replacement_translated,
                        "reason": _dedup_preference_reason(current_score, existing_score)
                    })
            else:
                # Keep existing entry when the duplicate does not add preferred data.
                skipped_count += 1
                if skipped_count <= 10:
                    print(f"[Skip] Pass 2: '{raw_name}' -> '{translated_name}' (duplicate of '{existing_raw}' -> '{existing_translated}')")
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 2, "action": "dropped",
                        "kept": existing_raw, "dropped": raw_name,
                        "translation": translated_name,
                        "reason": "duplicate translation"
                    })
        else:
            # New translation, keep it
            deduplicated.append(entry)
            seen_translations[translated_lower] = (raw_name, entry, len(deduplicated) - 1)
    
    replaced_msg = f" ({replaced_count} replaced with more complete entries, {len(deduplicated)} remaining)" if replaced_count > 0 else f" ({len(deduplicated)} remaining)"
    print(f"[Dedup] ✅ PASS 2 complete: {skipped_count} duplicates removed{replaced_msg}")
    return deduplicated


# Batch processing functions
def process_chapter_batch(chapters_batch: List[Tuple[int, str]], 
                         client: UnifiedClient,
                         config: Dict,
                         contextual_enabled: bool,
                         history: List[Dict],
                         ctx_limit: int,
                         rolling_window: bool,
                         check_stop,
                         chunk_timeout: int = None,
                         history_manager: HistoryManager = None) -> List[Dict]:
    """Process a batch of chapters in parallel with improved interrupt support.

    Contextual mode mirrors the main translator:
    - When CONTEXTUAL is enabled, previous chapters are converted to
      chat messages via trim_context_history(), respecting
      INCLUDE_SOURCE_IN_HISTORY.
    - We also log an approximate combined prompt size per chapter.
    """
    temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
    
    env_max_output = os.getenv("MAX_OUTPUT_TOKENS")
    if env_max_output and env_max_output.isdigit():
        mtoks = int(env_max_output)
    else:
        mtoks = config.get('max_tokens', 4196)
    
    results: List[Dict] = []
    
    with ThreadPoolExecutor(max_workers=len(chapters_batch)) as executor:
        futures: Dict = {}
        
        for idx, chap in chapters_batch:
            if check_stop():
                break
            display_idx = _glossary_chapter_actual_num(idx)
            chapter_label = _glossary_chapter_log_label(display_idx, config.get("total_chapters") or None)
                
            # Get system and user prompts
            system_prompt, user_prompt = build_prompt(chap)

            # Build optional assistant prefill message if configured
            assistant_prefill_msgs = []
            assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
            if assistant_prompt_env:
                assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]

            # Build messages correctly with system and user prompts
            if not contextual_enabled:
                msgs = [
                    {"role": "system", "content": system_prompt},
                ] + assistant_prefill_msgs + [
                    {"role": "user", "content": user_prompt}
                ]
            else:
                # Microsecond lock to prevent race conditions when reading history
                time.sleep(0.000001)
                with _history_lock:
                    msgs = (
                        [{"role": "system", "content": system_prompt}]
                        + trim_context_history(history, ctx_limit, rolling_window)
                        + assistant_prefill_msgs
                        + [{"role": "user", "content": user_prompt}]
                    )

            # Approximate combined prompt tokens for this chapter
            try:
                total_tokens = 0
                assistant_tokens = 0
                for m in msgs:
                    content = m.get("content", "") or ""
                    tokens = count_tokens(content)
                    total_tokens += tokens
                    if m.get("role") == "assistant":
                        assistant_tokens += tokens
                non_assistant = total_tokens - assistant_tokens

                if contextual_enabled and assistant_tokens > 0:
                    print(
                        f"💬 {chapter_label} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant:,}, "
                        f"assistant/memory: {assistant_tokens:,}) / {GLOSSARY_LIMIT_STR}"
                    )
                else:
                    print(
                        f"💬 {chapter_label} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) / {GLOSSARY_LIMIT_STR}"
                    )
            except Exception:
                # Never let logging break batch processing
                pass

            # Submit to thread pool
            future = executor.submit(
                process_single_chapter_api_call,
                idx, chap, msgs, client, temp, mtoks, check_stop, chunk_timeout,
                chapter_num=display_idx,
            )
            futures[future] = (idx, chap, msgs)  # Store messages for history update
        
        # Process results with better cancellation
        for future in as_completed(futures):  # Removed timeout - let futures complete
            # Normal stop check (during graceful stop, check_stop() returns False)
            if check_stop():
                # print("🛑 Stop detected - cancelling all pending operations...")  # Redundant
                # Cancel all pending futures immediately
                cancelled = cancel_all_futures(list(futures.keys()))
                if cancelled > 0:
                    print(f"✅ Cancelled {cancelled} pending API calls")
                # Shutdown executor immediately
                executor.shutdown(wait=False)
                break
                
            idx, chap, msgs = futures[future]  # Get messages too
            try:
                result = future.result(timeout=0.5)  # Short timeout on result retrieval
                # Ensure chap is added to result here if not already present
                if 'chap' not in result:
                    result['chap'] = chap
                    
                # Update history if contextual is enabled and we got a valid response
                if contextual_enabled and 'resp' in result and result['resp']:
                    # Find the user message content from the messages
                    user_content = None
                    for msg in msgs:
                        if msg.get('role') == 'user':
                            user_content = msg.get('content', '')
                            break
                    
                    if user_content:
                        try:
                            if history_manager:
                                history = history_manager.append_to_history(
                                    user_content=user_content,
                                    assistant_content=result['resp'],
                                    hist_limit=ctx_limit,
                                    reset_on_limit=not rolling_window,
                                    rolling_window=rolling_window,
                                    raw_assistant_object=result.get('raw_obj')
                                )
                            else:
                                with _history_lock:
                                    history.append({"role": "user", "content": user_content})
                                    assistant_entry = {"role": "assistant", "content": result['resp']}
                                    if 'raw_obj' in result and result['raw_obj']:
                                        assistant_entry["_raw_content_object"] = result['raw_obj']
                                    history.append(assistant_entry)
                        except Exception as e:
                            print(f"⚠️ Failed to save batch history for chapter {_glossary_chapter_actual_num(idx)}: {e}")
                
                results.append(result)
            except Exception as e:
                if "stopped by user" in str(e).lower():
                    print(f"✅ Chapter {_glossary_chapter_actual_num(idx)} stopped by user")
                else:
                    print(f"Error processing chapter {_glossary_chapter_actual_num(idx)}: {e}")
                results.append({
                    'idx': idx,
                    'data': [],
                    'resp': "",
                    'chap': chap,
                    'error': str(e)
                })
            
            # Graceful stop check AFTER processing: if an API call completed during graceful stop,
            # we've now saved its result, so we can stop
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                print("✅ Graceful stop: Chapter completed and saved, stopping batch processing...")
                # Cancel remaining futures
                cancelled = cancel_all_futures(list(futures.keys()))
                if cancelled > 0:
                    print(f"✅ Cancelled {cancelled} pending API calls")
                executor.shutdown(wait=False)
                break
    
    # Sort results by chapter index
    results.sort(key=lambda x: x['idx'])
    return results

def process_single_chapter_api_call(idx: int, chap: str, msgs: List[Dict], 
                                  client: UnifiedClient, temp: float, mtoks: int,
                                  stop_check_fn, chunk_timeout: int = None,
                                  chunk_idx: int = None, total_chunks: int = None,
                                  chapter_num: int = None,
                                  before_send_callback=None) -> Dict:
    """Process a single chapter API call with thread-safe payload handling"""
    display_chapter_num = chapter_num if chapter_num is not None else idx + 1
    
    # Early exit: skip immediately if stop/graceful-stop is already flagged
    if stop_check_fn() or os.environ.get('GRACEFUL_STOP') == '1' or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,
            'error': "Skipped (stop requested)",
            'graceful_stop_skip': True,
        }
    
    # Ensure the request always contains a user message and no non-serializable blobs
    msgs = _sanitize_messages_for_api(msgs, chap)
    # APPLY INTERRUPTIBLE THREADING DELAY FIRST
    thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
    if thread_delay > 0:
        # Check if we need to wait (same logic as unified_api_client)
        if hasattr(client, '_thread_submission_lock') and hasattr(client, '_last_thread_submission_time'):
            with client._thread_submission_lock:
                current_time = time.time()
                time_since_last = current_time - client._last_thread_submission_time
                
                if time_since_last < thread_delay:
                    sleep_time = thread_delay - time_since_last
                    thread_name = threading.current_thread().name
                    
                    # PRINT BEFORE THE DELAY STARTS
                    print(f"🧵 [{thread_name}] Applying thread delay: {sleep_time:.5f}s for Chapter {display_chapter_num}")
                    
                    # Interruptible sleep - check stop flag every 0.1 seconds
                    elapsed = 0
                    check_interval = 0.1
                    while elapsed < sleep_time:
                        if stop_check_fn() or os.environ.get('GRACEFUL_STOP') == '1' or os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                            raise UnifiedClientError("Glossary extraction stopped by user during threading delay")
                        
                        sleep_chunk = min(check_interval, sleep_time - elapsed)
                        time.sleep(sleep_chunk)
                        elapsed += sleep_chunk
                
                client._last_thread_submission_time = time.time()
                if not hasattr(client, '_thread_submission_count'):
                    client._thread_submission_count = 0
                client._thread_submission_count += 1
    start_time = time.time()
    
    # Thread-safe payload directory
    thread_name = threading.current_thread().name
    thread_id = threading.current_thread().ident
    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
    try:
        os.makedirs(thread_dir, exist_ok=True)
    except (PermissionError, OSError):
        import tempfile
        thread_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "glossary", f"{thread_name}_{thread_id}")
        try:
            os.makedirs(thread_dir, exist_ok=True)
        except Exception:
            thread_dir = None  # Skip payload saving
    
    try:
        # Save request payload before API call
        if thread_dir:
            payload_file = os.path.join(thread_dir, f"chapter_{idx+1}_request.json")
            with open(payload_file, 'w', encoding='utf-8') as f:
                # Clean messages to remove non-serializable objects before saving
                clean_msgs = []
                for msg in msgs:
                    clean_msg = {'role': msg.get('role'), 'content': msg.get('content')}
                    # Don't include _raw_content_object in payload as it's not JSON serializable
                    clean_msgs.append(clean_msg)
                
                json.dump({
                    'chapter': display_chapter_num,
                    'messages': clean_msgs,
                    'temperature': temp,
                    'max_tokens': mtoks,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, ensure_ascii=False)
    except (PermissionError, OSError):
        pass  # Non-fatal: payload saving is debug-only

    try:
        # Use send_with_interrupt for API call
        raw, finish_reason, raw_obj = send_with_interrupt(
            messages=msgs,
            client=client, 
            temperature=temp,
            max_tokens=mtoks,
            stop_check_fn=stop_check_fn,
            chunk_timeout=chunk_timeout,
            chapter_idx=idx,
            chapter_num=display_chapter_num,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            before_send_callback=before_send_callback,
        )
        request_model_name = _current_glossary_model_name({}, prefer_thread=True)
        request_key_identifier, request_key_pool = _current_glossary_key_context({}, prefer_thread=True)

        # Handle the response - it might be a tuple or a string
        if raw is None:
            print(f"⚠️ API returned None for chapter {display_chapter_num}")
            return {
                'idx': idx,
                'data': [],
                'resp': "",
                'chap': chap,
                'model_name': request_model_name,
                'key_identifier': request_key_identifier,
                'key_pool': request_key_pool,
                'error': "API returned None"
            }

        if isinstance(raw, tuple):
            resp = raw[0] if raw[0] is not None else ""
        elif isinstance(raw, str):
            resp = raw
        elif hasattr(raw, 'content'):
            resp = raw.content if raw.content is not None else ""
        elif hasattr(raw, 'text'):
            resp = raw.text if raw.text is not None else ""
        else:
            resp = str(raw) if raw is not None else ""

        # Ensure resp is never None
        if resp is None:
            resp = ""
        
        # Save the raw response in thread-safe location
        if thread_dir:
            try:
                response_file = os.path.join(thread_dir, f"chapter_{idx+1}_response.txt")
                with open(response_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(resp)
            except (PermissionError, OSError):
                pass
        
        # Parse response using the new parser
        data = parse_api_response(resp)
        
        # More detailed debug logging
        print(f"[BATCH] Chapter {display_chapter_num} - Raw response length: {len(resp)} chars")
        print(f"[BATCH] Chapter {display_chapter_num} - Parsed {len(data)} entries before validation")
        
        # Filter out invalid entries
        valid_data = []
        for entry in data:
            if validate_extracted_entry(entry):
                # Clean the raw_name
                if 'raw_name' in entry:
                    entry['raw_name'] = entry['raw_name'].strip()
                valid_data.append(entry)
            else:
                print(f"[BATCH] Chapter {display_chapter_num} - Invalid entry: {entry}")
        
        elapsed = time.time() - start_time
        print(f"[BATCH] Completed Chapter {display_chapter_num} in {elapsed:.1f}s at {time.strftime('%H:%M:%S')} - Extracted {len(valid_data)} valid entries")
        
        return {
            'idx': idx,
            'data': valid_data,
            'resp': resp,
            'chap': chap,  # Include the chapter text in the result
            'raw_obj': raw_obj,  # Include raw object for history (from send_with_interrupt)
            'finish_reason': finish_reason,  # Track truncation ('length'/'MAX_TOKENS' = truncated)
            'model_name': request_model_name,
            'key_identifier': request_key_identifier,
            'key_pool': request_key_pool,
            'error': None
        }
            
    except UnifiedClientError as e:
        # Graceful-stop cancellations are expected when queued calls are prevented from starting.
        # Keep a concise log so it's clear why extraction stopped/skipped without spamming the full error.
        if _is_graceful_stop_skip_error(e):
            if chunk_idx and total_chunks and int(total_chunks) > 1:
                print(f"⏭️ Chapter {display_chapter_num} chunk {chunk_idx}/{total_chunks} skipped (graceful stop)")
            else:
                print(f"⏭️ Chapter {display_chapter_num} skipped (graceful stop)")
        else:
            print(f"[Error] API call interrupted/failed for chapter {display_chapter_num}: {e}")

        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
            'model_name': _current_glossary_model_name({}, prefer_thread=True),
            'key_identifier': _current_glossary_key_context({}, prefer_thread=True)[0],
            'error': str(e),
            'graceful_stop_skip': _is_graceful_stop_skip_error(e),
        }
    except Exception as e:
        print(f"[Error] Unexpected error for chapter {display_chapter_num}: {e}")
        import traceback
        print(f"[Error] Traceback: {traceback.format_exc()}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
            'model_name': _current_glossary_model_name({}, prefer_thread=True),
            'key_identifier': _current_glossary_key_context({}, prefer_thread=True)[0],
            'error': str(e)
        }
def process_single_chapter_with_split(idx: int,
                                      chap: str,
                                      build_prompt_fn,
                                      chapter_splitter,
                                      available_tokens: int,
                                      chapter_split_enabled: bool,
                                      contextual_enabled: bool,
                                      history,
                                      ctx_limit: int,
                                      rolling_window: bool,
                                      client,
                                      temp: float,
                                      mtoks: int,
                                      stop_check_fn,
                                      chunk_timeout: int = None,
                                      before_send_callback=None,
                                      chapter_num: int = None):
    """
    Wrapper that performs chapter-level splitting (using output-limit budget) before calling the API.
    Aggregates all chunk results into a single result dict to keep batch accounting identical.
    """
    display_chapter_num = chapter_num if chapter_num is not None else idx + 1

    # Decide if splitting is needed
    chapter_tokens = chapter_splitter.count_tokens(chap)
    if not (chapter_split_enabled and chapter_tokens > available_tokens):
        # No split needed; build messages as usual
        system_prompt, user_prompt = build_prompt_fn(chap)
        
        # Build optional assistant prefill message if configured
        assistant_prefill_msgs = []
        assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
        if assistant_prompt_env:
            assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]
        
        if not contextual_enabled:
            msgs = [
                {"role": "system", "content": system_prompt},
            ] + assistant_prefill_msgs + [
                {"role": "user", "content": user_prompt}
            ]
        else:
            time.sleep(0.000001)
            with _history_lock:
                msgs = (
                    [{"role": "system", "content": system_prompt}]
                    + trim_context_history(history, ctx_limit, rolling_window)
                    + assistant_prefill_msgs
                    + [{"role": "user", "content": user_prompt}]
                )
        return process_single_chapter_api_call(idx, chap, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout, chapter_num=display_chapter_num, before_send_callback=before_send_callback)

    print(f"⚠️ Chapter {display_chapter_num} exceeds chunk budget ({chapter_tokens:,} > {available_tokens:,}); splitting...")
    # Wrap plain text as simple HTML for splitter
    chapter_html = f"<html><body><p>{chap.replace(chr(10)+chr(10), '</p><p>')}</p></body></html>"
    chunks = chapter_splitter.split_chapter(chapter_html, available_tokens)
    print(f"📄 Chapter split into {len(chunks)} chunks (budget {available_tokens:,})")

    aggregated_data = []
    last_resp = ""
    last_raw_obj = None
    last_model_name = ""
    last_key_identifier = ""
    last_key_pool = ""
    any_chunk_truncated = False
    for chunk_html, chunk_idx, total_chunks in chunks:
        if stop_check_fn():
            print(f"❌ Glossary extraction stopped during chunk {chunk_idx}/{total_chunks} of chapter {display_chapter_num}")
            break
        soup = BeautifulSoup(chunk_html, 'html.parser')
        chunk_text = soup.get_text(strip=True)

        system_prompt, user_prompt = build_prompt_fn(chunk_text)
        
        # Build optional assistant prefill message if configured
        assistant_prefill_msgs = []
        assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
        if assistant_prompt_env:
            assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]
        
        if not contextual_enabled:
            msgs = [
                {"role": "system", "content": system_prompt},
            ] + assistant_prefill_msgs + [
                {"role": "user", "content": user_prompt}
            ]
        else:
            time.sleep(0.000001)
            with _history_lock:
                msgs = (
                    [{"role": "system", "content": system_prompt}]
                    + trim_context_history(history, ctx_limit, rolling_window)
                    + assistant_prefill_msgs
                    + [{"role": "user", "content": user_prompt}]
                )

        print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {display_chapter_num}")
        # Sanitize before delegating (guarantees user + no raw blobs in payload)
        msgs = _sanitize_messages_for_api(msgs, chunk_text)
        result = process_single_chapter_api_call(
            idx, chunk_text, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout,
            chunk_idx=chunk_idx, total_chunks=total_chunks,
            chapter_num=display_chapter_num,
            before_send_callback=before_send_callback,
        )
        if result.get("data"):
            aggregated_data.extend(result["data"])
        last_resp = result.get("resp", last_resp)
        if result.get("raw_obj"):
            last_raw_obj = result.get("raw_obj")
        if result.get("model_name"):
            last_model_name = result.get("model_name")
        if result.get("key_identifier"):
            last_key_identifier = result.get("key_identifier")
        if result.get("key_pool"):
            last_key_pool = result.get("key_pool")
        # Track truncation: if any chunk was truncated, the whole chapter is truncated
        chunk_finish = result.get("finish_reason", "stop")
        if chunk_finish in ("length", "MAX_TOKENS", "max_tokens"):
            any_chunk_truncated = True

    return {
        'idx': idx,
        'data': aggregated_data,
        'resp': last_resp,
        'chap': chap,
        'raw_obj': last_raw_obj,
        'finish_reason': 'length' if any_chunk_truncated else 'stop',
        'model_name': last_model_name,
        'key_identifier': last_key_identifier,
        'key_pool': last_key_pool,
        'error': None
    }

def process_merged_group_api_call(merge_group: list, msgs_builder_fn, 
                                   client, temp: float, mtoks: int,
                                   stop_check_fn, chunk_timeout: int = None,
                                   before_send_callback=None,
                                   chapter_num_map=None) -> Dict:
    """
    Process a merged group of chapters in a single API call.
    
    Args:
        merge_group: List of (idx, chap) tuples
        msgs_builder_fn: Function to build messages, takes (chapter_content, history_context)
        client: UnifiedClient instance
        temp: Temperature for API call
        mtoks: Max tokens for API call
        stop_check_fn: Function to check if stop is requested
        chunk_timeout: Optional timeout for API call
        
    Returns:
        Dict with keys: 'results' (list of per-chapter results), 'merged_indices' (list of child indices)
    """
    # Build optional assistant prefill message if configured
    assistant_prefill_msgs = []
    assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
    if assistant_prompt_env:
        assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]

    if len(merge_group) == 1:
        # Single chapter, use normal processing
        idx, chap = merge_group[0]
        system_prompt, user_prompt = msgs_builder_fn(chap)
        msgs = [{"role": "system", "content": system_prompt}] + assistant_prefill_msgs + [{"role": "user", "content": user_prompt}]
        result = process_single_chapter_api_call(
            idx, chap, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout,
            chapter_num=(chapter_num_map or {}).get(idx, idx + 1),
            before_send_callback=before_send_callback,
        )
        return {'results': [result], 'merged_indices': []}
    
    # Merge chapter contents WITHOUT separators (glossary extraction doesn't need them)
    parent_idx = merge_group[0][0]
    merged_parts = []
    chapter_nums = []
    
    for idx, chap in merge_group:
        chapter_num = (chapter_num_map or {}).get(idx, idx + 1)
        chapter_nums.append(chapter_num)
        merged_parts.append(chap)
    
    merged_content = "\n\n".join(merged_parts)
    
    print(f"\n🔗 Processing MERGED group: Chapters {chapter_nums}")
    print(f"   📊 Merged content: {len(merged_content):,} characters")
    
    # Build messages for merged content
    system_prompt, user_prompt = msgs_builder_fn(merged_content)
    msgs = [{"role": "system", "content": system_prompt}] + assistant_prefill_msgs + [{"role": "user", "content": user_prompt}]
    msgs = _sanitize_messages_for_api(msgs, merged_content)
    
    # Thread-safe payload directory
    thread_name = threading.current_thread().name
    thread_id = threading.current_thread().ident
    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
    try:
        os.makedirs(thread_dir, exist_ok=True)
    except (PermissionError, OSError):
        import tempfile
        thread_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "glossary", f"{thread_name}_{thread_id}")
        try:
            os.makedirs(thread_dir, exist_ok=True)
        except Exception:
            thread_dir = None
    
    start_time = time.time()
    
    try:
        # Save request payload
        if thread_dir:
            try:
                payload_file = os.path.join(thread_dir, f"merged_chapters_{parent_idx+1}_request.json")
                with open(payload_file, 'w', encoding='utf-8') as f:
                    clean_msgs = [{'role': msg.get('role'), 'content': msg.get('content')} for msg in msgs]
                    json.dump({
                        'chapters': chapter_nums,
                        'messages': clean_msgs,
                        'temperature': temp,
                        'max_tokens': mtoks,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f, indent=2, ensure_ascii=False)
            except (PermissionError, OSError):
                pass
        
        # Make API call (use parent chapter idx for logging, pass merged chapter nums for progress bar)
        raw, finish_reason, raw_obj = send_with_interrupt(
            messages=msgs,
            client=client,
            temperature=temp,
            max_tokens=mtoks,
            stop_check_fn=stop_check_fn,
            chunk_timeout=chunk_timeout,
            chapter_idx=parent_idx,
            chapter_num=chapter_nums[0] if chapter_nums else parent_idx + 1,
            merged_chapters=chapter_nums,
            before_send_callback=before_send_callback,
        )
        request_model_name = _current_glossary_model_name({}, prefer_thread=True)
        request_key_identifier, request_key_pool = _current_glossary_key_context({}, prefer_thread=True)
        
        # Extract response text
        resp = ""
        if raw is None:
            print(f"⚠️ API returned None for merged group")
            return {
                'results': [{'idx': idx, 'data': [], 'resp': '', 'chap': chap, 'model_name': request_model_name, 'key_identifier': request_key_identifier, 'key_pool': request_key_pool, 'error': 'API returned None'}
                           for idx, chap in merge_group],
                'merged_indices': []
            }
        
        if isinstance(raw, tuple):
            resp = raw[0] if raw[0] is not None else ""
        elif isinstance(raw, str):
            resp = raw
        elif hasattr(raw, 'content'):
            resp = raw.content if raw.content is not None else ""
        elif hasattr(raw, 'text'):
            resp = raw.text if raw.text is not None else ""
        else:
            resp = str(raw) if raw is not None else ""
        
        if resp is None:
            resp = ""
        
        # Save response
        if thread_dir:
            try:
                response_file = os.path.join(thread_dir, f"merged_chapters_{parent_idx+1}_response.txt")
                with open(response_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(resp)
            except (PermissionError, OSError):
                pass
        
        elapsed = time.time() - start_time
        print(f"   ✅ Received merged response ({len(resp):,} chars) in {elapsed:.1f}s")
        
        # Parse the entire merged response
        all_data = parse_api_response(resp)
        
        # Filter valid entries
        valid_data = []
        for entry in all_data:
            if validate_extracted_entry(entry):
                if 'raw_name' in entry:
                    entry['raw_name'] = entry['raw_name'].strip()
                valid_data.append(entry)
        
        print(f"   📊 Extracted {len(valid_data)} valid entries from merged response")
        
        # Create results for each chapter in the group
        # Since we can't easily attribute entries to specific chapters in a merged response,
        # all entries go to the parent chapter's result, children get empty lists
        results = []
        for i, (idx, chap) in enumerate(merge_group):
            if i == 0:
                # Parent chapter gets all entries
                results.append({
                    'idx': idx,
                    'data': valid_data,
                    'resp': resp,
                    'chap': chap,
                    'raw_obj': raw_obj,
                    'finish_reason': finish_reason,
                    'model_name': request_model_name,
                    'key_identifier': request_key_identifier,
                    'key_pool': request_key_pool,
                    'error': None
                })
            else:
                # Child chapters get empty results (they're merged)
                results.append({
                    'idx': idx,
                    'data': [],
                    'resp': '',
                    'chap': chap,
                    'raw_obj': None,
                    'model_name': request_model_name,
                    'key_identifier': request_key_identifier,
                    'key_pool': request_key_pool,
                    'error': None,
                    'merged_into': parent_idx
                })
        
        return {
            'results': results,
            'merged_indices': [idx for idx, _ in merge_group[1:]]
        }
        
    except UnifiedClientError as e:
        # Check if this is a user stop (not an actual error)
        err_lower = str(e).lower()
        if "stopped by user" in err_lower or "cancelled" in err_lower or "operation cancelled" in err_lower:
            # print(f"🛑 Glossary extraction stopped by user")  # Redundant
            # Re-raise to propagate the stop signal up the call stack
            raise
        else:
            # Actual API error (timeout, etc.)
            print(f"❌ Merged group failed: {e} (NOTE: API Error triggered cancellation logic)")
            
            return {
                'results': [{'idx': idx, 'data': [], 'resp': '', 'chap': chap, 'error': str(e)}
                           for idx, chap in merge_group],
                'merged_indices': []
            }
    except Exception as e:
        print(f"❌ Merged group failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        
        return {
            'results': [{'idx': idx, 'data': [], 'resp': '', 'chap': chap, 'error': str(e)}
                       for idx, chap in merge_group],
            'merged_indices': []
        }


def _extract_pdf_chapters_for_glossary(pdf_path, check_stop=None):
    """Extract text chapters from a PDF for glossary extraction.

    Uses PyMuPDF's fast page.get_text() for plain text extraction — glossary
    extraction only needs text content, not XHTML formatting.  This is the same
    approach the review generator uses and is dramatically faster than the
    full extract_pdf_with_formatting() XHTML rendering pipeline.

    Falls back to extract_pdf_with_formatting only if PyMuPDF is unavailable.
    """
    chapters = []

    # ── Fast path: fitz.get_text() (same as review_generator) ──
    try:
        import fitz
        print(f"📄 Extracting PDF with formatting: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📄 Extracting text from {total_pages} pages via PyMuPDF (fast mode)...")

        for i, page in enumerate(doc):
            if check_stop and check_stop():
                print(f"🛑 PDF extraction stopped at page {i+1}/{total_pages}")
                doc.close()
                return chapters

            text = page.get_text()
            if text and text.strip():
                chapters.append(text.strip())

        doc.close()
        print(f"✅ PDF text extracted: {len(chapters)} pages with content (from {total_pages} total)")
        return chapters

    except ImportError:
        print("⚠️ PyMuPDF not available, falling back to extract_pdf_with_formatting...")
    except Exception as e:
        print(f"⚠️ Fast PDF extraction failed ({e}), falling back...")

    # ── Fallback: extract_pdf_with_formatting (slow but reliable) ──
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="glossarion_pdf_extract_")
    previous_render_mode = os.environ.get("PDF_RENDER_MODE")
    if not previous_render_mode:
        os.environ["PDF_RENDER_MODE"] = "xhtml"

    try:
        from pdf_extractor import extract_pdf_with_formatting
        page_list, _ = extract_pdf_with_formatting(
            pdf_path=pdf_path,
            output_dir=tmp_dir,
            extract_images=False,
            page_by_page=True
        )
        for page_num, page_html in (page_list or []):
            if check_stop and check_stop():
                return chapters
            try:
                page_text = BeautifulSoup(page_html, 'html.parser').get_text("\n", strip=True)
            except Exception:
                page_text = str(page_html)
            if page_text and page_text.strip():
                chapters.append(page_text)
    finally:
        if not previous_render_mode:
            os.environ.pop("PDF_RENDER_MODE", None)

    return chapters


def _extract_sdlxliff_chapters_for_glossary(sdlxliff_path, check_stop=None):
    """Extract eligible SDLXLIFF source segment text for glossary generation."""
    use_async = os.getenv("USE_ASYNC_CHAPTER_EXTRACTION", "0") == "1"
    if not use_async:
        from sdlxliff_extractor import extract_sdlxliff_texts
        chapters = extract_sdlxliff_texts(sdlxliff_path)
        print(f"SDLXLIFF source extracted: {len(chapters)} segment(s)")
        return chapters

    tmp_dir = tempfile.mkdtemp(prefix="glossarion_sdlxliff_glossary_")
    try:
        from sdlxliff_extraction_manager import SdlxliffExtractionManager

        manager = SdlxliffExtractionManager(log_callback=print)
        state = {"done": False, "result": None}

        def _complete(result):
            state["result"] = result
            state["done"] = True

        manager.extract_async(
            sdlxliff_path,
            tmp_dir,
            progress_callback=lambda message: print(f"SDLXLIFF extraction: {message}"),
            completion_callback=_complete,
        )
        while not state["done"]:
            if check_stop and check_stop():
                manager.stop_extraction()
                return []
            time.sleep(0.2)

        result = state["result"] or {}
        if not result.get("success"):
            raise RuntimeError(result.get("error", "SDLXLIFF extraction failed"))
        chapters_path = result.get("chapters_path") or os.path.join(tmp_dir, "chapters_full.json")
        with open(chapters_path, "r", encoding="utf-8") as f:
            chapters_data = json.load(f)
        chapters = [
            str(chapter.get("body", "")).strip()
            for chapter in chapters_data
            if str(chapter.get("body", "")).strip()
        ]
        print(f"SDLXLIFF source extracted: {len(chapters)} segment(s)")
        return chapters
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



# Update main function to support batch processing:
def main(log_callback=None, stop_callback=None):
    # Declare global variables at the very start of the function
    global _skipped_chapters
    global GLOSSARY_SOURCE_LANGUAGE, GLOSSARY_SOURCE_LANGUAGE_PATH
    global _GLOSSARY_SOURCE_LANGUAGE_LOADED, _GLOSSARY_SOURCE_LANGUAGE_LOGGED
    global GLOSSARY_SOURCE_SCRIPT, GLOSSARY_SOURCE_SCRIPT_IS_CJK
    global _GLOSSARY_SOURCE_SCRIPT_READY, _GLOSSARY_SOURCE_SCRIPT_LOGGED
    GLOSSARY_SOURCE_LANGUAGE = None
    GLOSSARY_SOURCE_LANGUAGE_PATH = None
    _GLOSSARY_SOURCE_LANGUAGE_LOADED = False
    _GLOSSARY_SOURCE_LANGUAGE_LOGGED = False
    GLOSSARY_SOURCE_SCRIPT = None
    GLOSSARY_SOURCE_SCRIPT_IS_CJK = None
    _GLOSSARY_SOURCE_SCRIPT_READY = False
    _GLOSSARY_SOURCE_SCRIPT_LOGGED = False

    # Thread identifiers may be reused between runs.  Never let an aborted
    # request's partial thinking stream attach to a later glossary request.
    if os.getenv("DIRECT_TEXT_ACTIVE", "0") == "1":
        with _direct_stream_capture_lock:
            _direct_stream_captures.clear()
    
    # Redirect print/logs to callback if provided
    if log_callback:
        set_output_redirect(log_callback)
    """Modified main function that can accept a logging callback and stop callback"""
    if log_callback:
        set_output_redirect(log_callback)
    
    # Set up stop checking
    def check_stop():
        # During graceful stop, ALWAYS return False to let current chapter complete fully
        # The main loop will check GRACEFUL_STOP at the START of each new chapter
        if _glossary_is_graceful_stop_active():
            return False
        if os.environ.get('TRANSLATION_CANCELLED') == '1':
            return True
        
        if stop_callback and stop_callback():
            # print("❌ Glossary extraction stopped by user request.")  # Redundant - logged elsewhere
            return True
        return is_stop_requested()
        
    start = time.time()
    
    # Handle both command line and GUI calls
    if '--epub' in sys.argv:
        # Command line mode
        parser = argparse.ArgumentParser(description='Extract glossary from EPUB/TXT/SDLXLIFF')
        parser.add_argument('--epub', required=True, help='Path to EPUB/TXT/SDLXLIFF file')
        parser.add_argument('--output', required=True, help='Output glossary path')
        parser.add_argument('--config', help='Config file path')
        
        args = parser.parse_args()
        epub_path = args.epub
    else:
        # GUI mode - get from environment
        epub_path = os.getenv("EPUB_PATH", "")
        if not epub_path and len(sys.argv) > 1:
            epub_path = sys.argv[1]
        
        # Create args object for GUI mode
        import types
        args = types.SimpleNamespace()
        args.epub = epub_path
        args.output = os.getenv("OUTPUT_PATH", "glossary.json")
        args.config = os.getenv("CONFIG_PATH", "config.json")

    is_text_file = epub_path.lower().endswith('.txt')
    is_pdf_file = epub_path.lower().endswith('.pdf')
    is_sdlxliff_file = epub_path.lower().endswith('.sdlxliff')
    
    if is_sdlxliff_file:
        chapters = _extract_sdlxliff_chapters_for_glossary(epub_path, check_stop)
        _chapter_filenames = {}
        file_base = os.path.splitext(os.path.basename(epub_path))[0]
    elif is_text_file:
        # Import text processor
        from extract_glossary_from_txt import extract_chapters_from_txt
        chapters = extract_chapters_from_txt(epub_path)
        _chapter_filenames = {}  # No filename metadata for txt files
        file_base = os.path.splitext(os.path.basename(epub_path))[0]
    elif is_pdf_file:
        # PDF: extract page-by-page using subprocess to prevent GUI lag
        chapters = _extract_pdf_chapters_for_glossary(epub_path, check_stop)
        _chapter_filenames = {}  # No filename metadata for PDF files

        file_base = os.path.splitext(os.path.basename(epub_path))[0]
    else:
        # Existing EPUB code — request metadata so we can show filenames in logs
        _raw_chapters = extract_chapters_from_epub(epub_path, return_metadata=True)
        chapters = [text for text, _fn in _raw_chapters]
        _chapter_filenames = {idx: fn for idx, (text, fn) in enumerate(_raw_chapters)}
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        file_base = epub_base

    # ── Build chapter position mapping ──────────────────────────────────
    # Maps idx → chapter_number matching the same numbering TransateKRtoEN.py uses.
    # This ensures CHAPTER_RANGE selects the same chapters in both scripts.
    _chapter_positions = {}  # idx → chapter number for range filtering
    use_spine_order = os.getenv("USE_SPINE_ORDER", "0") == "1"

    if _chapter_filenames and not is_text_file and not is_pdf_file and not is_sdlxliff_file:
        if use_spine_order:
            # ── Spine order mode: build OPF offset positions from within the EPUB ──
            # Same logic as TransateKRtoEN.py's _spine_pos_by_idx
            try:
                import xml.etree.ElementTree as _ET
                import zipfile as _zf
                translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'

                with _zf.ZipFile(epub_path, 'r') as zf:
                    # Find content.opf inside the EPUB
                    opf_content = None
                    for name in zf.namelist():
                        if name.lower().endswith('.opf'):
                            opf_content = zf.read(name).decode('utf-8')
                            break

                if opf_content:
                    _opf_root = _ET.fromstring(opf_content)
                    _ns = {'opf': 'http://www.idpf.org/2007/opf'}
                    if _opf_root.tag.startswith('{'):
                        _default_ns = _opf_root.tag[1:_opf_root.tag.index('}')]
                        _ns = {'opf': _default_ns}

                    _manifest = {}
                    for _item in _opf_root.findall('.//opf:manifest/opf:item', _ns):
                        _iid = _item.get('id')
                        _href = _item.get('href')
                        _mtype = _item.get('media-type', '')
                        if _iid and _href and (
                            'html' in _mtype.lower() or
                            _href.endswith(('.html', '.xhtml', '.htm'))
                        ):
                            _manifest[_iid] = os.path.basename(_href)

                    _spine_el = _opf_root.find('.//opf:spine', _ns)
                    _all_spine_basenames = []
                    if _spine_el is not None:
                        for _iref in _spine_el.findall('opf:itemref', _ns):
                            _idref = _iref.get('idref')
                            if _idref and _idref in _manifest:
                                _all_spine_basenames.append(_manifest[_idref])

                    _sp_kw_env = os.getenv('SPECIAL_FILE_KEYWORDS', '')
                    _sp_keywords = [k.strip().lower() for k in _sp_kw_env.split(',') if k.strip()] if _sp_kw_env else [
                        'title', 'toc', 'copyright', 'preface', 'nav',
                        'message', 'notice', 'colophon', 'dedication', 'epigraph',
                        'foreword', 'acknowledgment', 'author', 'appendix',
                        'bibliography'
                    ]
                    _sp_exact_env = os.getenv('SPECIAL_FILE_EXACT', '')
                    _sp_exact = [k.strip().lower() for k in _sp_exact_env.split(',') if k.strip()] if _sp_exact_env else ['index', 'glossary', 'glossary_extension']

                    # Build offset positions, skipping only configured special files.
                    # Non-numbered files like info.xhtml may display as Ch.000 in the GUI,
                    # but they must remain normal OPF entries here.
                    def _is_special_spine(fname):
                        fnoext = os.path.splitext(os.path.basename(str(fname or '')).lower())[0]
                        if not fnoext:
                            return False
                        return fnoext in _sp_exact or any(kw in fnoext for kw in _sp_keywords)

                    _offset_by_basename = {}  # basename → offset pos (1-based)
                    _tpos = 0
                    for _sb in _all_spine_basenames:
                        _special = _is_special_spine(_sb)
                        _skip = (not translate_special and _special)
                        if not _skip:
                            _tpos += 1
                            _offset_by_basename[_sb] = _tpos
                            _offset_by_basename[os.path.splitext(_sb)[0]] = _tpos

                    # Map each glossary chapter to its spine offset
                    for _ci, _fn in _chapter_filenames.items():
                        _bn_noext = os.path.splitext(_fn)[0] if _fn else ''
                        _pos = _offset_by_basename.get(_fn) or _offset_by_basename.get(_bn_noext)
                        if _pos is not None:
                            _chapter_positions[_ci] = _pos

                    if _chapter_positions:
                        print(f"📊 Spine order: mapped {len(_chapter_positions)}/{len(chapters)} chapters to OPF positions")
                    else:
                        print("⚠️ Spine order: could not map chapters to OPF positions, falling back to filename numbering")
            except Exception as _e:
                print(f"⚠️ Spine order: failed to read content.opf from EPUB: {_e}")

        # ── Normal mode (or spine order fallback): extract number from filename ──
        # Same logic as extract_chapter_number_from_filename in TransateKRtoEN.py:
        # use the rightmost digit sequence in the filename stem.
        if not _chapter_positions:
            for _ci, _fn in _chapter_filenames.items():
                if _fn:
                    _stem = os.path.splitext(_fn)[0]
                    _nums = re.findall(r'[0-9]+', _stem)
                    if _nums:
                        _chapter_positions[_ci] = int(_nums[-1])
                    else:
                        _chapter_positions[_ci] = _ci + 1  # fallback

    # For txt/pdf, positions are just 1-based index
    if not _chapter_positions:
        _chapter_positions = {i: i + 1 for i in range(len(chapters))}

    # If user didn't override --output, derive it from the EPUB filename:
    if args.output == 'glossary.json':
        args.output = f"{file_base}_glossary.json" 

    try:
        _mac_cwd_unusable = (
            sys.platform == "darwin"
            and (
                os.path.abspath(os.getcwd()) == os.path.abspath(os.sep)
                or not os.access(os.getcwd(), os.W_OK)
            )
        )
    except Exception:
        _mac_cwd_unusable = sys.platform == "darwin"
    if args.output and not os.path.isabs(args.output) and _mac_cwd_unusable:
        source_dir = os.path.dirname(os.path.abspath(epub_path)) if epub_path else ""
        args.output = os.path.join(source_dir or resolve_shared_glossary_dir(), args.output)
    elif args.output and os.path.isabs(args.output) and sys.platform == "darwin":
        try:
            root_glossary = os.path.abspath(os.path.join(os.path.abspath(os.sep), "Glossary"))
            output_abs = os.path.abspath(args.output)
            if os.path.commonpath([root_glossary, output_abs]) == root_glossary:
                source_dir = os.path.dirname(os.path.abspath(epub_path)) if epub_path else ""
                rel_output = os.path.relpath(output_abs, root_glossary)
                args.output = os.path.join(source_dir or resolve_shared_glossary_dir(), "Glossary", rel_output)
        except Exception:
            pass

    # Keep regular EPUB glossaries in the shared Glossary subfolder. Output-side
    # backup copies are written only when the caller passes an explicit target.
    save_glossary_backup_in_output = os.getenv("SAVE_GLOSSARY_IN_OUTPUT", "0").strip().lower() in ("1", "true", "yes", "on")
    base_out_dir = os.path.dirname(args.output)
    base_out_dir_abs = os.path.abspath(base_out_dir or os.getcwd())
    base_out_parent_abs = os.path.dirname(base_out_dir_abs)
    expected_book_folder = sanitize_glossary_folder_name(file_base)
    glossary_dir = None
    if (
        os.path.basename(base_out_parent_abs).lower() == "glossary"
        and os.path.basename(base_out_dir_abs).casefold() == expected_book_folder.casefold()
    ):
        shared_glossary_dir = base_out_parent_abs
        glossary_dir = base_out_dir_abs
    elif os.path.basename(base_out_dir_abs).lower() == "glossary":
        shared_glossary_dir = base_out_dir
    else:
        shared_glossary_dir = os.path.join(base_out_dir, "Glossary")
    migrate_all_legacy_glossary_files(
        shared_glossary_dir,
        logger=print,
    )
    if glossary_dir is None:
        glossary_dir = get_book_glossary_dir(shared_glossary_dir, file_base)
    os.makedirs(glossary_dir, exist_ok=True)
    if not save_glossary_backup_in_output:
        os.environ.pop("GLOSSARY_OUTPUT_BACKUP_DIR", None)

    # override the module‐level PROGRESS_FILE to include epub name
    global PROGRESS_FILE, _GLOSSARY_OUTPUT_FILE
    PROGRESS_FILE = os.path.join(
        glossary_dir,
        f"{file_base}_glossary_progress.json"
    )
    _GLOSSARY_OUTPUT_FILE = os.path.join(glossary_dir, os.path.basename(args.output))
    args.output = _GLOSSARY_OUTPUT_FILE
    _set_glossary_source_language_from_metadata(args.output, epub_path, log=False)
    progress_context = make_glossary_progress_context(
        progress_file=PROGRESS_FILE,
        output_file=_GLOSSARY_OUTPUT_FILE,
    )

    config = load_config(args.config)

    refinement_env_defaults = {
        "GLOSSARY_REFINEMENT_ENABLED": "1" if config.get("glossary_refinement_enabled", False) else "0",
        "GLOSSARY_REFINEMENT_SYSTEM_PROMPT": config.get("glossary_refinement_system_prompt") or DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT,
        "GLOSSARY_REFINEMENT_USER_PROMPT": config.get("glossary_refinement_user_prompt", ""),
        "GLOSSARY_REFINEMENT_TYPE_MODE": config.get("glossary_refinement_type_mode", "all"),
        "GLOSSARY_REFINEMENT_SELECTED_TYPES": ",".join(config.get("glossary_refinement_selected_types", [])),
        "GLOSSARY_REFINEMENT_CHUNKING_MODE": config.get("glossary_refinement_chunking_mode", "separate"),
        "GLOSSARY_REFINEMENT_SKIP_DEDUPE": "1" if config.get("glossary_refinement_skip_dedupe", False) else "0",
    }
    for _env_key, _env_value in refinement_env_defaults.items():
        if os.getenv(_env_key) is None:
            os.environ[_env_key] = str(_env_value or "")
    
    # Log assistant prompt if configured
    _log_assistant_prompt_once()

    # Ensure truncation retry settings use the correct Other Settings keys
    # (only set if not already provided by the environment)
    if os.getenv("RETRY_TRUNCATED") is None:
        os.environ["RETRY_TRUNCATED"] = "1" if config.get("retry_truncated", True) else "0"
    if os.getenv("TRUNCATION_RETRY_ATTEMPTS") is None:
        os.environ["TRUNCATION_RETRY_ATTEMPTS"] = str(config.get("truncation_retry_attempts", 3))
    if os.getenv("MAX_RETRY_TOKENS") is None:
        os.environ["MAX_RETRY_TOKENS"] = str(config.get("max_retry_tokens", -1))

    # Use Gemini thinking settings exactly as saved in Other Settings / config
    enable_thinking = bool(config.get("enable_gemini_thinking", False))
    model_name = (os.getenv("MODEL") or config.get("model", "") or "").lower()
    is_gemini_flash = "gemini-3" in model_name and "flash" in model_name
    is_gemini_pro = "gemini-3" in model_name and "pro" in model_name

    if enable_thinking:
        os.environ["ENABLE_GEMINI_THINKING"] = "1"
        thinking_level = config.get("thinking_level")
        if thinking_level is not None:
            os.environ["GEMINI_THINKING_LEVEL"] = str(thinking_level)
        if "thinking_budget_tokens" in config or "thinking_budget" in config:
            budget_val = config.get("thinking_budget_tokens", config.get("thinking_budget"))
            try:
                os.environ["THINKING_BUDGET"] = str(int(budget_val))
            except Exception:
                pass
    else:
        # Explicitly disable and set fallback level per TransateKRtoEN behavior
        os.environ["ENABLE_GEMINI_THINKING"] = "0"
        if is_gemini_flash:
            os.environ["GEMINI_THINKING_LEVEL"] = "minimal"
        elif is_gemini_pro:
            os.environ["GEMINI_THINKING_LEVEL"] = "low"
        else:
            os.environ.pop("GEMINI_THINKING_LEVEL", None)
        os.environ.pop("THINKING_BUDGET", None)
    
    # Skip thinking / lightweight thinking for title & metadata
    # (env vars may not be set if glossary runs in-process before user opens settings)
    if os.getenv('SKIP_BOOK_TITLE_THINKING') is None:
        os.environ['SKIP_BOOK_TITLE_THINKING'] = '1' if config.get('skip_book_title_thinking', True) else '0'
    if os.getenv('SKIP_METADATA_THINKING') is None:
        os.environ['SKIP_METADATA_THINKING'] = '1' if config.get('skip_metadata_thinking', True) else '0'
    if os.getenv('SKIP_TOC_THINKING') is None:
        os.environ['SKIP_TOC_THINKING'] = '1' if config.get('skip_toc_thinking', False) else '0'
    if os.getenv('LIGHTWEIGHT_THINKING_LEVEL') is None:
        os.environ['LIGHTWEIGHT_THINKING_LEVEL'] = str(config.get('lightweight_thinking_level', 1))
    
    # Retrieve book titles (raw from input, translated from metadata/output)
    global BOOK_TITLE_RAW, BOOK_TITLE_TRANSLATED, BOOK_TITLE_PRESENT, BOOK_TITLE_VALUE
    BOOK_TITLE_RAW = _extract_raw_title_from_epub(epub_path)
    BOOK_TITLE_TRANSLATED = _extract_translated_title_from_metadata(args.output, epub_path)
    progress_file = _resolved_glossary_progress_file(context=progress_context)
    
    # Check progress file for saved book title to avoid re-translation
    if not BOOK_TITLE_TRANSLATED and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                saved_prog = json.load(f)
                saved_title = saved_prog.get('book_title')
                if saved_title:
                    print(f"📂 Loaded translated book title from progress: {saved_title}")
                    BOOK_TITLE_TRANSLATED = saved_title
                    BOOK_TITLE_VALUE = saved_title
                    BOOK_TITLE_PRESENT = saved_prog.get('book_title_present', False)
        except Exception:
            # If reading progress fails, just fall back to standard behavior
            pass
    
    # Get API key from environment variables (set by GUI) or config file
    api_key = (os.getenv("API_KEY") or 
               os.getenv("OPENAI_API_KEY") or 
               os.getenv("OPENAI_OR_Gemini_API_KEY") or
               os.getenv("GEMINI_API_KEY") or
               config.get('api_key'))

    # Get model from environment or config
    model = os.getenv("MODEL") or config.get('model', 'gemini-1.5-flash')

    # Define output directory (use current directory as default)
    out = os.path.dirname(args.output) if hasattr(args, 'output') else os.getcwd()

    # Use the variables we just retrieved
    client = create_client_with_multi_key_support(api_key, model, out, config)
    
    # Translate book title if needed:
    # 1. We have a raw title
    # 2. We don't have a translated title from metadata
    # 3. Translation is enabled
    if BOOK_TITLE_RAW and not BOOK_TITLE_TRANSLATED:
        include_title = os.getenv("GLOSSARY_INCLUDE_BOOK_TITLE", "1").lower() not in ("0", "false", "no")
        if include_title:
            try:
                # Try to import translate_title from TransateKRtoEN
                # Use local import to avoid top-level circular dependencies
                from TransateKRtoEN import translate_title
                
                print(f"📚 Translating book title: {BOOK_TITLE_RAW}")
                translated = translate_title(
                    BOOK_TITLE_RAW, 
                    client, 
                    None, # system_prompt (uses default/env)
                    None, # user_prompt (uses default/env)
                    float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
                )
                if translated and translated != BOOK_TITLE_RAW:
                    print(f"📚 Translated title for glossary: {translated}")
                    BOOK_TITLE_TRANSLATED = translated
                    
                    # Save immediately to progress
                    try:
                        p_data = {}
                        progress_file = _resolved_glossary_progress_file(context=progress_context)
                        if os.path.exists(progress_file):
                            with open(progress_file, 'r', encoding='utf-8') as f:
                                p_data = json.load(f)
                        p_data['book_title'] = translated
                        
                        with open(progress_file, 'w', encoding='utf-8') as f:
                            json.dump(p_data, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"⚠️ Failed to save book title to progress: {e}")
                else:
                    # Translation failed or returned same, assume raw is best we have
                    BOOK_TITLE_TRANSLATED = BOOK_TITLE_RAW
            except ImportError:
                print("⚠️ Could not import translate_title from TransateKRtoEN - using raw title")
                BOOK_TITLE_TRANSLATED = BOOK_TITLE_RAW
            except Exception as e:
                print(f"⚠️ Failed to translate book title: {e} - using raw title")
                BOOK_TITLE_TRANSLATED = BOOK_TITLE_RAW
    
    # Ensure fallback if no translation occurred
    if not BOOK_TITLE_TRANSLATED:
        BOOK_TITLE_TRANSLATED = BOOK_TITLE_RAW
    if not BOOK_TITLE_RAW:
        BOOK_TITLE_RAW = BOOK_TITLE_TRANSLATED
    progress_context.book_title_raw = BOOK_TITLE_RAW
    progress_context.book_title_translated = BOOK_TITLE_TRANSLATED
    progress_context.book_title_present = BOOK_TITLE_PRESENT
    progress_context.book_title_value = BOOK_TITLE_VALUE
    
    # Check for batch mode
    batch_enabled = os.getenv("BATCH_TRANSLATION", "0") == "1"
    batch_size = int(os.getenv("BATCH_SIZE", "5"))
    batching_mode = (os.getenv("BATCHING_MODE", "direct") or "direct").strip().lower()
    batch_group_size = int(os.getenv("BATCH_GROUP_SIZE", "3"))

    # Backward compatibility for CONSERVATIVE_BATCHING
    if os.getenv("CONSERVATIVE_BATCHING", "0") == "1":
        batching_mode = "conservative"
    if batching_mode not in ("direct", "conservative", "aggressive"):
        batching_mode = "direct"

    print(f"[DEBUG] BATCH_TRANSLATION = {os.getenv('BATCH_TRANSLATION')} (enabled: {batch_enabled})")
    print(f"[DEBUG] BATCH_SIZE = {batch_size}")
    log_batching_mode = "no batching" if batching_mode == "aggressive" else batching_mode
    print(f"[DEBUG] BATCHING_MODE = {log_batching_mode}")
    print(f"[DEBUG] BATCH_GROUP_SIZE = {batch_group_size}")

    if batch_enabled:
        display_mode = "No Batching" if batching_mode == "aggressive" else batching_mode.capitalize()
        print(f"🚀 Glossary batch mode enabled with size: {batch_size} (Mode: {display_mode})")
        if batching_mode == 'conservative':
            print(f"   Conservative group size: {batch_group_size}")
        elif batching_mode == 'aggressive':
            print(f"   Aggressive mode: keeps {batch_size} parallel calls and auto-refills")
        print(f"📑 Note: Glossary extraction uses a simplified batching process for API calls.")
    
    #API call delay — show effective delay (per-key > global)
    _display_delay = None
    try:
        _gk_pool = getattr(UnifiedClient, '_glossary_key_pool', None)
        if _gk_pool and hasattr(_gk_pool, 'keys') and _gk_pool.keys:
            for _gk in _gk_pool.keys:
                _gk_d = getattr(_gk, 'api_call_delay', 0.0) or 0.0
                if _gk_d > 0:
                    _display_delay = _gk_d
                    break
    except Exception:
        pass
    if _display_delay is None:
        try:
            _im_gk = getattr(UnifiedClient, '_in_memory_glossary_keys', None)
            if _im_gk:
                for _gk_dict in _im_gk:
                    _gk_d = float(_gk_dict.get('api_call_delay', 0)) if _gk_dict.get('api_call_delay') not in (None, '') else 0.0
                    if _gk_d > 0:
                        _display_delay = _gk_d
                        break
        except Exception:
            pass
    api_delay = float(_display_delay) if _display_delay else float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
    _delay_source = "per-key" if _display_delay else "global"
    print(f"⏱️  API call delay: {api_delay} seconds ({_delay_source})")
    
    # Get compression factor from environment (glossary-specific with fallback)
    compression_factor = float(os.getenv("GLOSSARY_COMPRESSION_FACTOR", os.getenv("COMPRESSION_FACTOR", "1.0")))
    print(f"📐 Compression Factor: {compression_factor}")

    # Toggle for chapter splitting (manual glossary tab)
    chapter_split_enabled = os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "0") == "1"
    if chapter_split_enabled or os.getenv("DEBUG_CHAPTER_SPLIT_LOG", "0") == "1":
        print(f"✂️  Chapter Split Enabled: {'✅' if chapter_split_enabled else '❌'}")

    # Resolve effective output token limit, including active per-key pool overrides.
    effective_output_tokens = _effective_glossary_output_limit(config, model)

    # Budget for chunking, matching TransateKRtoEN safe limit logic
    available_tokens = _compute_safe_input_tokens(effective_output_tokens, compression_factor)
    print(f"📊 Chunk budget: {available_tokens:,} tokens (output limit {effective_output_tokens:,}, margin 500, compression {compression_factor})")

    # Initialize chapter splitter with compression factor
    chapter_splitter = ChapterSplitter(model_name=model, compression_factor=compression_factor)

    refinement_compression_factor = float(os.getenv(
        "GLOSSARY_REFINEMENT_COMPRESSION_FACTOR",
        os.getenv("COMPRESSION_FACTOR", str(compression_factor)),
    ))
    refinement_available_tokens = _compute_safe_input_tokens(effective_output_tokens, refinement_compression_factor)
    refinement_splitter = ChapterSplitter(model_name=model, compression_factor=refinement_compression_factor)

    # Get temperature from environment or config
    temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))

    # Use effective output tokens for API max_tokens
    mtoks = effective_output_tokens
    
    # Get context limit from environment or config
    ctx_limit = int(os.getenv("GLOSSARY_CONTEXT_LIMIT") or config.get('context_limit_chapters', 3))

    # Parse chapter range from environment
    chapter_range = os.getenv("CHAPTER_RANGE", "").strip()
    range_start = None
    range_end = None
    parsed_chapter_range = _parse_glossary_chapter_range(chapter_range)
    if parsed_chapter_range:
        range_start, range_end = parsed_chapter_range
        if use_spine_order:
            print(f"📊 Chapter Range Filter (SPINE ORDER): positions {range_start} to {range_end}")
        else:
            print(f"📊 Chapter Range Filter: {range_start} to {range_end}")
    elif chapter_range:
        print(f"⚠️ Invalid chapter range format: {chapter_range} (use format: 5 or 5-10)")

    # Log settings
    format_parts = ["type", "raw_name", "translated_name", "gender"]
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            format_parts.extend(custom_fields)
    except:
        pass
    print(f"📑 Glossary Format: Simple ({', '.join(format_parts)})")
    
    # Check honorifics filter toggle
    honorifics_disabled = os.getenv('GLOSSARY_DISABLE_HONORIFICS_FILTER', '0') == '1'
    if honorifics_disabled:
        print("📑 Honorifics Filtering: ❌ DISABLED")
    else:
        print("📑 Honorifics Filtering: ✅ ENABLED")

    cjk_filter_env = os.getenv('GLOSSARY_CJK_SCRIPT_FILTER', '0')
    cjk_filter_enabled = cjk_filter_env == '1'
    cjk_target_language = os.getenv('GLOSSARY_TARGET_LANGUAGE', 'English')
    cjk_target_known_non_cjk = _is_known_non_cjk_output_language()
    cjk_filter_active = cjk_filter_enabled and cjk_target_known_non_cjk
    print(
        f"[DEBUG] GLOSSARY_CJK_SCRIPT_FILTER = {cjk_filter_env} "
        f"(enabled: {cjk_filter_enabled}, target: {cjk_target_language}, "
        f"known_non_cjk_target: {cjk_target_known_non_cjk}, active: {cjk_filter_active})"
    )

    # Log glossary anti-duplicate parameters usage (matches GlossaryManager)
    if os.getenv("GLOSSARY_ENABLE_ANTI_DUPLICATE", "0") == "1":
        ad_top_p = _format_sampling_log_float(os.getenv("GLOSSARY_TOP_P", "1.0"))
        ad_min_p = _format_sampling_log_float(os.getenv("GLOSSARY_MIN_P", "0.0"))
        ad_top_k = os.getenv("GLOSSARY_TOP_K", "0")
        ad_freq = _format_sampling_log_float(os.getenv("GLOSSARY_FREQUENCY_PENALTY", "0.0"))
        ad_pres = _format_sampling_log_float(os.getenv("GLOSSARY_PRESENCE_PENALTY", "0.0"))
        ad_rep = _format_sampling_log_float(os.getenv("GLOSSARY_REPETITION_PENALTY", "1.0"))
        print(f"🎯 Anti-duplicate enabled for glossary (top_p={ad_top_p}, min_p={ad_min_p}, top_k={ad_top_k}, freq_penalty={ad_freq}, presence_penalty={ad_pres}, repetition_penalty={ad_rep})")
    
    # Log custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            print(f"📑 Custom Fields: {', '.join(custom_fields)}")
    except:
        pass
    
    # Check if custom prompt is being used
    if os.getenv('GLOSSARY_SYSTEM_PROMPT'):
        print("📑 Using custom extraction prompt")
    else:
        print("📑 Using default extraction prompt")

    if is_sdlxliff_file:
        chapters = _extract_sdlxliff_chapters_for_glossary(args.epub, check_stop)
        _chapter_filenames = {}
    elif is_text_file:
        from extract_glossary_from_txt import extract_chapters_from_txt
        chapters = extract_chapters_from_txt(args.epub)
        _chapter_filenames = {}
    elif args.epub.lower().endswith('.pdf'):
        # PDF: extract page-by-page using subprocess to prevent GUI lag
        chapters = _extract_pdf_chapters_for_glossary(args.epub, check_stop)
        _chapter_filenames = {}
    else:
        _raw_chapters = extract_chapters_from_epub(args.epub, return_metadata=True)
        chapters = [text for text, _fn in _raw_chapters]
        _chapter_filenames = {idx: fn for idx, (text, fn) in enumerate(_raw_chapters)}
    
    # Rebuild chapter positions from the final chapter list
    # (this is the definitive load used for processing)
    if _chapter_filenames and not is_text_file and not is_pdf_file and not is_sdlxliff_file:
        _chapter_positions = {}
        if use_spine_order:
            # Spine order positions were already built from OPF above
            # Rebuild from current filenames using same OPF logic
            try:
                import xml.etree.ElementTree as _ET2
                import zipfile as _zf2
                translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
                with _zf2.ZipFile(epub_path, 'r') as zf:
                    opf_content = None
                    for name in zf.namelist():
                        if name.lower().endswith('.opf'):
                            opf_content = zf.read(name).decode('utf-8')
                            break
                if opf_content:
                    _opf_root2 = _ET2.fromstring(opf_content)
                    _ns2 = {'opf': 'http://www.idpf.org/2007/opf'}
                    if _opf_root2.tag.startswith('{'):
                        _ns2 = {'opf': _opf_root2.tag[1:_opf_root2.tag.index('}')]}
                    _manifest2 = {}
                    for _item in _opf_root2.findall('.//opf:manifest/opf:item', _ns2):
                        _iid = _item.get('id')
                        _href = _item.get('href')
                        _mtype = _item.get('media-type', '')
                        if _iid and _href and ('html' in _mtype.lower() or _href.endswith(('.html', '.xhtml', '.htm'))):
                            _manifest2[_iid] = os.path.basename(_href)
                    _spine_el2 = _opf_root2.find('.//opf:spine', _ns2)
                    _all_spine2 = []
                    if _spine_el2 is not None:
                        for _iref in _spine_el2.findall('opf:itemref', _ns2):
                            _idref = _iref.get('idref')
                            if _idref and _idref in _manifest2:
                                _all_spine2.append(_manifest2[_idref])
                    _sp_kw_env2 = os.getenv('SPECIAL_FILE_KEYWORDS', '')
                    _sp_keywords2 = [k.strip().lower() for k in _sp_kw_env2.split(',') if k.strip()] if _sp_kw_env2 else [
                        'title', 'toc', 'copyright', 'preface', 'nav',
                        'message', 'notice', 'colophon', 'dedication', 'epigraph',
                        'foreword', 'acknowledgment', 'author', 'appendix',
                        'bibliography'
                    ]
                    _sp_exact_env2 = os.getenv('SPECIAL_FILE_EXACT', '')
                    _sp_exact2 = [k.strip().lower() for k in _sp_exact_env2.split(',') if k.strip()] if _sp_exact_env2 else ['index', 'glossary', 'glossary_extension']

                    def _is_special2(fname):
                        fnoext = os.path.splitext(os.path.basename(str(fname or '')).lower())[0]
                        if not fnoext:
                            return False
                        return fnoext in _sp_exact2 or any(kw in fnoext for kw in _sp_keywords2)
                    _off2 = {}
                    _tpos2 = 0
                    for _sb in _all_spine2:
                        if not (not translate_special and _is_special2(_sb)):
                            _tpos2 += 1
                            _off2[_sb] = _tpos2
                            _off2[os.path.splitext(_sb)[0]] = _tpos2
                    for _ci, _fn in _chapter_filenames.items():
                        _bn = os.path.splitext(_fn)[0] if _fn else ''
                        _pos = _off2.get(_fn) or _off2.get(_bn)
                        if _pos is not None:
                            _chapter_positions[_ci] = _pos
            except Exception:
                pass
        # Fallback: extract number from filename (same as first load)
        if not _chapter_positions:
            for _ci, _fn in _chapter_filenames.items():
                if _fn:
                    _stem = os.path.splitext(_fn)[0]
                    _nums = re.findall(r'[0-9]+', _stem)
                    if _nums:
                        _chapter_positions[_ci] = int(_nums[-1])
                    else:
                        _chapter_positions[_ci] = _ci + 1
    if not _chapter_positions:
        _chapter_positions = {i: i + 1 for i in range(len(chapters))}

    global _GLOSSARY_CHAPTER_POSITIONS, _GLOSSARY_CHAPTER_NUMBERS, _GLOSSARY_CHAPTER_FILENAMES, _GLOSSARY_TOTAL_CHAPTERS
    _GLOSSARY_CHAPTER_POSITIONS = {int(k): int(v) for k, v in (_chapter_positions or {}).items()}
    _GLOSSARY_CHAPTER_FILENAMES = {int(k): os.path.basename(str(v or "")) for k, v in (_chapter_filenames or {}).items()}
    _GLOSSARY_CHAPTER_NUMBERS = {}
    for _idx, _fname in _GLOSSARY_CHAPTER_FILENAMES.items():
        _stem = os.path.splitext(_fname)[0]
        _nums = re.findall(r'[0-9]+', _stem) if _stem else []
        if _nums:
            _GLOSSARY_CHAPTER_NUMBERS[_idx] = int(_nums[-1])
    for _idx, _pos in _GLOSSARY_CHAPTER_POSITIONS.items():
        _GLOSSARY_CHAPTER_NUMBERS.setdefault(_idx, _pos)
    _GLOSSARY_TOTAL_CHAPTERS = len(chapters)
    progress_context.chapter_positions = dict(_GLOSSARY_CHAPTER_POSITIONS)
    progress_context.chapter_numbers = dict(_GLOSSARY_CHAPTER_NUMBERS)
    progress_context.chapter_filenames = dict(_GLOSSARY_CHAPTER_FILENAMES)
    progress_context.total_chapters = _GLOSSARY_TOTAL_CHAPTERS

    if not chapters:
        print("No chapters found. Exiting.")
        return

    # Check for stop before starting processing
    if check_stop():
        return

    global _GLOSSARY_QA_ISSUES_FOUND
    prog = load_progress(context=progress_context)
    _GLOSSARY_QA_ISSUES_FOUND = _normalize_glossary_qa_issues(
        prog.get('qa_issues_found'),
        prog.get('chapters')
    )
    completed = prog['completed']
    failed = prog.get('failed', [])
    in_progress = prog.get('in_progress', [])
    # Remove failed chapters from completed so they get retried
    if failed:
        before = len(completed)
        completed[:] = [idx for idx in completed if idx not in failed]
        if before != len(completed):
            print(f"🔄 {len(failed)} previously failed chapter(s) will be retried: {[i+1 for i in sorted(failed)]}")
        failed.clear()  # Reset failed list for this run
        _GLOSSARY_QA_ISSUES_FOUND.clear()
    # Load existing glossary from output file (if it exists) instead of progress file
    # This preserves manual edits to the glossary
    output_glossary_path = args.output
    if os.path.exists(output_glossary_path):
        try:
            with open(output_glossary_path, 'r', encoding='utf-8') as f:
                glossary = json.load(f)
            # Strip description keys from loaded entries if description is
            # no longer in the user's active Custom Fields.  Old glossary
            # files saved when description WAS active would otherwise carry
            # ghost fields that inflate the dedup field count and cause
            # newer entries (without description) to never win — or,
            # conversely, prevent the loaded entry from being correctly
            # treated as equal to a freshly-parsed entry.
            glossary = _strip_unwanted_description_keys(
                [e for e in glossary if isinstance(e, dict)]
            )
            print(f"📂 Loaded existing glossary: {len(glossary)} entries")
        except Exception as e:
            print(f"⚠️ Could not load existing glossary, starting fresh: {e}")
            glossary = []
    else:
        # Try loading CSV if JSON not found (legacy support)
        csv_path = os.path.splitext(output_glossary_path)[0] + '.csv'
        if os.path.exists(csv_path):
            glossary = _load_glossary_file(csv_path)
        else:
            glossary = []
    merged_indices = prog.get('merged_indices', [])

    def _mark_glossary_in_progress(indices):
        changed = False
        model_update_indices = []
        for _idx in indices:
            try:
                _idx = int(_idx)
            except (TypeError, ValueError):
                continue
            if _idx in completed:
                continue
            if _idx in failed:
                failed.remove(_idx)
                _GLOSSARY_QA_ISSUES_FOUND.pop(_idx, None)
                changed = True
            if _idx not in in_progress:
                in_progress.append(_idx)
                changed = True
            model_update_indices.append(_idx)
        if changed or model_update_indices:
            save_progress(
                completed,
                glossary,
                merged_indices,
                failed=failed,
                in_progress=in_progress,
                context=progress_context,
                model_update_indices=model_update_indices,
            )

    def _clear_glossary_in_progress(indices):
        remove = set()
        for _idx in indices:
            try:
                remove.add(int(_idx))
            except (TypeError, ValueError):
                pass
        if not remove:
            return
        before = len(in_progress)
        in_progress[:] = [idx for idx in in_progress if idx not in remove]
        if before != len(in_progress):
            save_progress(completed, glossary, merged_indices, failed=failed, in_progress=in_progress, context=progress_context)

    def _restore_glossary_in_progress_for_hard_stop(indices):
        requested = set(_unique_int_list(list(in_progress) + list(indices or [])))
        disk_in_progress = _glossary_disk_in_progress_snapshot(progress_context)
        if disk_in_progress is not None:
            requested.update(disk_in_progress)

        # A stop abandons the whole run, so restore every active row in this
        # book. This also catches callbacks that reached the API just before a
        # parallel future was removed from the executor's active-future map.
        restored_progress = _restore_glossary_in_progress_file(progress_context)
        if restored_progress is not None:
            completed[:] = _unique_int_list(restored_progress.get("completed", []))
            failed[:] = _unique_int_list(restored_progress.get("failed", []))
            merged_indices[:] = _unique_int_list(restored_progress.get("merged_indices", []))
            in_progress[:] = _unique_int_list(restored_progress.get("in_progress", []))
            return

        # If the progress file disappeared during cancellation there is no
        # previous snapshot to restore. Fail only the active requests; never
        # turn already completed chapters into failures.
        for _idx in sorted(requested):
            if _idx in completed:
                completed.remove(_idx)
            if _idx in merged_indices:
                merged_indices.remove(_idx)
            if _idx in in_progress:
                in_progress.remove(_idx)
            _mark_glossary_failed(failed, _idx)
        if requested:
            save_progress(
                completed,
                glossary,
                merged_indices,
                failed=failed,
                in_progress=in_progress,
                context=progress_context,
            )
    
    # Request merging configuration (glossary-specific with fallback to global)
    request_merging_enabled = os.getenv('GLOSSARY_REQUEST_MERGING_ENABLED', os.getenv('REQUEST_MERGING_ENABLED', '0')) == '1'
    request_merge_count = int(os.getenv('GLOSSARY_REQUEST_MERGE_COUNT', os.getenv('REQUEST_MERGE_COUNT', '3')))
    
    if request_merging_enabled and request_merge_count > 1:
        print(f"\n🔗 REQUEST MERGING ENABLED: Combining up to {request_merge_count} chapters per request")
    
    # Get both settings
    contextual_enabled = os.getenv('CONTEXTUAL', '1') == '1'
    rolling_window = True
    
    # Initialize HistoryManager for context history (separate from progress file)
    # Use source file-based naming like other glossary files
    history_filename = f"{file_base}_glossary_history.json"
    history_manager = HistoryManager(glossary_dir, history_filename)
    history = history_manager.load_history() if contextual_enabled else []
    total_chapters = len(chapters)
    
    # Count chapters that will be processed with range filter
    chapters_to_process = []
    for idx, chap in enumerate(chapters):
        # Skip if chapter is outside the range
        if range_start is not None and range_end is not None:
            chapter_num = _chapter_positions.get(idx, idx + 1)
            if not (range_start <= chapter_num <= range_end):
                continue
        if idx not in completed:
            chapters_to_process.append((idx, chap))
    
    if len(chapters_to_process) < total_chapters:
        print(f"📊 Processing {len(chapters_to_process)} out of {total_chapters} chapters")
    
    _prime_glossary_source_script_from_chapters(chapters_to_process)

    # Get chunk timeout (respect RETRY_TIMEOUT toggle)
    retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
    chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "1800")) if retry_timeout_enabled else None
    
    # Process chapters based on mode
    # Request merging now works with both batch and sequential modes
    use_request_merging = request_merging_enabled and request_merge_count > 1
    
    if batch_enabled and len(chapters_to_process) > 0:
        # BATCH MODE: Process in batches with per-entry saving
        
        # Create merge groups if request merging is enabled
        if use_request_merging:
            # Use the same proximity logic as translation to avoid merging distant chapters
            try:
                from TransateKRtoEN import RequestMerger
            except Exception:
                RequestMerger = None
            if chapter_split_enabled:
                # Budget-aware auto-adjust (only when chapter splitting toggle is ON)
                merge_groups = []
                run_groups = []
                if RequestMerger:
                    run_groups = RequestMerger.create_merge_groups(
                        chapters_to_process,
                        max(1, len(chapters_to_process)),
                    )
                else:
                    run_groups = [chapters_to_process]

                for run in run_groups:
                    if not run:
                        continue
                    i = 0
                    while i < len(run):
                        group = [run[i]]
                        i += 1

                        while i < len(run) and len(group) < request_merge_count:
                            candidate = run[i]
                            merged_preview = "\n\n".join([c for (_, c) in group + [candidate]])
                            merged_tokens = chapter_splitter.count_tokens(merged_preview)

                            if merged_tokens <= available_tokens:
                                group.append(candidate)
                                i += 1
                            else:
                                break

                        merge_groups.append(group)

                print(f"🔗 Created {len(merge_groups)} merge groups from {len(chapters_to_process)} chapters (budget-aware)")
                units_to_process = merge_groups
                is_merged_mode = True
            else:
                # Original simple grouping by count when split toggle is OFF
                # Still enforce token budget to prevent oversized requests
                merge_groups = []
                i = 0
                while i < len(chapters_to_process):
                    group = [chapters_to_process[i]]
                    i += 1
                    while i < len(chapters_to_process) and len(group) < request_merge_count:
                        candidate = chapters_to_process[i]
                        merged_preview = "\n\n".join([c for (_, c) in group + [candidate]])
                        merged_tokens = chapter_splitter.count_tokens(merged_preview)
                        if merged_tokens <= available_tokens:
                            group.append(candidate)
                            i += 1
                        else:
                            break
                    merge_groups.append(group)
                print(f"🔗 Created {len(merge_groups)} merge groups from {len(chapters_to_process)} chapters (count-based)")
                units_to_process = merge_groups
                is_merged_mode = True
        else:
            units_to_process = [[ch] for ch in chapters_to_process]  # Each chapter as single-item group
            is_merged_mode = False

        # ``extract_chapters_from_epub`` returns documents in OPF spine order,
        # and every merge group above preserves that sequence. Futures are also
        # submitted in this order, but worker scheduling can otherwise let a
        # later unit reach client.send() first. Assign a stable ticket to each
        # unit and gate only its first send so parallel requests stay parallel
        # after being released in reading order.
        ordered_batch_dispatch_enabled = (
            os.getenv("ORDER_BATCH_REQUESTS_BY_SPINE", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        ordered_batch_dispatcher = _OrderedGlossaryBatchDispatcher(
            enabled=ordered_batch_dispatch_enabled,
            stop_check=check_stop,
        )
        unit_dispatch_order = {
            int(unit[0][0]): order
            for order, unit in enumerate(units_to_process)
            if unit
        }
        if ordered_batch_dispatch_enabled:
            print("📚 Glossary batch API dispatch: OPF spine order")
        
        aggressive_mode = batching_mode == 'aggressive'
        if batching_mode == 'conservative':
            effective_batch_group_size = max(1, batch_size * max(1, batch_group_size))
        else:
            effective_batch_group_size = max(1, batch_size)

        total_batches = 1 if aggressive_mode else (len(units_to_process) + effective_batch_group_size - 1) // effective_batch_group_size
        
        for batch_num in range(total_batches):
            # Check for graceful stop completion at START of each batch iteration
            # This allows the previous batch to fully complete (including save) before stopping
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                print(f"\u2705 Graceful stop: Previous batch completed and saved, stopping extraction...")
                # Apply deduplication before stopping
                if glossary:
                    print("\U0001F500 Applying deduplication and sorting before exit...")
                    glossary[:] = skip_duplicate_entries(glossary)
                    save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                    save_glossary_json(glossary, args.output)
                    save_glossary_csv(glossary, args.output)
                return
            
            # If graceful stop requested but NO API call is active, stop immediately
            if os.environ.get('GRACEFUL_STOP') == '1' and os.environ.get('GRACEFUL_STOP_API_ACTIVE') != '1':
                print(f"\u2705 Graceful stop: No API call in progress, stopping immediately...")
                # Apply deduplication before stopping
                if glossary:
                    print("\U0001F500 Applying deduplication and sorting before exit...")
                    glossary[:] = skip_duplicate_entries(glossary)
                    
                    custom_types = get_custom_entry_types()
                    type_order = {'book': -1, 'character': 0, 'term': 1}
                    other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                    for i, t in enumerate(other_types):
                        type_order[t] = i + 2
                    glossary.sort(key=lambda x: (
                        type_order.get(x.get('type', 'term'), 999),
                        x.get('raw_name', '').lower()
                    ))
                    
                    save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                    save_glossary_json(glossary, args.output)
                    save_glossary_csv(glossary, args.output)
                    print(f"\u2705 Saved {len(glossary)} entries before graceful exit")
                return
            
            # Check for stop at the beginning of each batch
            if check_stop():
                print(f"❌ Glossary extraction stopped at batch {batch_num+1}")
                # Apply deduplication before stopping
                if glossary:
                    print("🔀 Applying deduplication and sorting before exit...")
                    glossary[:] = skip_duplicate_entries(glossary)
                    
                    # Sort glossary
                    custom_types = get_custom_entry_types()
                    type_order = {'book': -1, 'character': 0, 'term': 1}
                    other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                    for i, t in enumerate(other_types):
                        type_order[t] = i + 2
                    glossary.sort(key=lambda x: (
                        type_order.get(x.get('type', 'term'), 999),
                        x.get('raw_name', '').lower()
                    ))
                    
                    save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                    save_glossary_json(glossary, args.output)
                    save_glossary_csv(glossary, args.output)
                    
                    print(f"✅ Saved {len(glossary)} entries before exit")
                return
            
            # Get current batch of units
            if aggressive_mode:
                current_batch_units = units_to_process
            else:
                batch_start = batch_num * effective_batch_group_size
                batch_end = min(batch_start + effective_batch_group_size, len(units_to_process))
                current_batch_units = units_to_process[batch_start:batch_end]
            
            # Count total chapters in this batch
            chapters_in_batch = sum(len(unit) for unit in current_batch_units)
            
            if is_merged_mode:
                print(f"\n🔄 Processing Batch {batch_num+1}/{total_batches} ({len(current_batch_units)} merged groups, {chapters_in_batch} chapters)")
            else:
                current_batch = [unit[0] for unit in current_batch_units]
                chapter_nums = sorted(idx + 1 for idx, _ in current_batch)
                # Collapse consecutive chapter numbers into ranges (e.g. 168–171)
                ranges = []
                i = 0
                while i < len(chapter_nums):
                    start = chapter_nums[i]
                    while i + 1 < len(chapter_nums) and chapter_nums[i + 1] == chapter_nums[i] + 1:
                        i += 1
                    end = chapter_nums[i]
                    ranges.append(str(start) if start == end else f"{start}–{end}")
                    i += 1
                print(f"\n🔄 Processing Batch {batch_num+1}/{total_batches} (Chapters: {', '.join(ranges)})")
            print(f"[BATCH] Submitting {len(current_batch_units)} work units for parallel processing...")
            batch_start_time = time.time()
            
            # Process batch in parallel BUT handle results as they complete
            temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
            # Use glossary-specific token limit with fallback
            env_max_output = os.getenv("GLOSSARY_MAX_OUTPUT_TOKENS", os.getenv("MAX_OUTPUT_TOKENS"))
            if env_max_output and env_max_output.isdigit():
                mtoks = int(env_max_output)
            else:
                mtoks = config.get('max_tokens', 4196)
            
            batch_entry_count = 0
            stopped_early = False
            
            # Determine number of workers - always cap to batch_size to respect concurrency limit
            num_workers = min(len(current_batch_units), batch_size)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}

                # Collect batch results for history update (same as TransateKRtoEN)
                batch_history_map = {}  # Will store (idx, user_prompt, resp, raw_obj) for each successful chapter

                def _submit_unit(unit):
                    """Submit a single work unit and return its Future."""
                    unit_indices = [u_idx for u_idx, _ in unit]
                    dispatch_order = unit_dispatch_order.get(unit_indices[0])

                    def _mark_unit_progress_on_send(
                        unit_indices=unit_indices,
                        dispatch_order=dispatch_order,
                    ):
                        ordered_batch_dispatcher.wait_for_turn(dispatch_order)
                        _mark_glossary_in_progress(unit_indices)

                    if is_merged_mode:
                        future = executor.submit(
                            process_merged_group_api_call,
                            unit, build_prompt, client, temp, mtoks, check_stop, chunk_timeout,
                            _mark_unit_progress_on_send,
                            chapter_num_map={
                                u_idx: _glossary_chapter_actual_num(u_idx, context=progress_context)
                                for u_idx, _ in unit
                            },
                        )
                    else:
                        idx, chap = unit[0]
                        system_prompt, user_prompt = build_prompt(chap)
                        
                        # Build optional assistant prefill message if configured
                        assistant_prefill_msgs = []
                        assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
                        if assistant_prompt_env:
                            assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]
                        
                        if not contextual_enabled:
                            msgs = [
                                {"role": "system", "content": system_prompt},
                            ] + assistant_prefill_msgs + [
                                {"role": "user", "content": user_prompt}
                            ]
                        else:
                            time.sleep(0.000001)
                            with _history_lock:
                                msgs = [{"role": "system", "content": system_prompt}] \
                                     + trim_context_history(history, ctx_limit, rolling_window) \
                                     + assistant_prefill_msgs \
                                     + [{"role": "user", "content": user_prompt}]
                        future = executor.submit(
                            process_single_chapter_with_split,
                            idx,
                            chap,
                            build_prompt,
                            chapter_splitter,
                            available_tokens,
                            chapter_split_enabled,
                            contextual_enabled,
                            history,
                            ctx_limit,
                            rolling_window,
                            client,
                            temp,
                            mtoks,
                            check_stop,
                            chunk_timeout,
                            _mark_unit_progress_on_send,
                            chapter_num=_glossary_chapter_actual_num(idx, context=progress_context),
                        )
                    futures[future] = unit
                    future.add_done_callback(
                        lambda _future, order=dispatch_order: (
                            ordered_batch_dispatcher.abandon_if_unsent(order)
                        )
                    )
                    # Small yield to keep GUI responsive
                    time.sleep(0.001)
                    return future

                def _handle_future_result(future, unit):
                    nonlocal batch_entry_count, stopped_early
                    unit_indices = [u_idx for u_idx, _ in unit]
                    model_updates = {}
                    key_updates = {}
                    extracted_entry_updates = {}

                    def _collect_request_metadata(result, target_indices=None):
                        if not isinstance(result, dict):
                            return
                        model_name = str(result.get('model_name') or '').strip()
                        key_identifier = str(result.get('key_identifier') or '').strip()
                        if not (model_name or key_identifier):
                            return
                        targets = target_indices
                        if targets is None:
                            try:
                                targets = [int(result.get('idx'))]
                            except (TypeError, ValueError):
                                targets = []
                        for target_idx in targets or []:
                            try:
                                target_idx = int(target_idx)
                            except (TypeError, ValueError):
                                continue
                            if model_name:
                                model_updates[target_idx] = model_name
                            if key_identifier:
                                key_updates[target_idx] = key_identifier

                    def _save_progress_for_unit():
                        save_progress(
                            completed,
                            glossary,
                            merged_indices,
                            failed=failed,
                            in_progress=in_progress,
                            context=progress_context,
                            model_update_indices=sorted(set(model_updates) | set(key_updates)),
                            model_updates=model_updates,
                            key_updates=key_updates,
                            extracted_entries_updates=extracted_entry_updates,
                        )

                    try:
                        if is_merged_mode:
                            # Handle merged group result
                            group_result = future.result(timeout=0.5)
                            results = group_result.get('results', [])
                            group_model = next((str(r.get('model_name') or '').strip() for r in results if isinstance(r, dict) and str(r.get('model_name') or '').strip()), '')
                            group_key = next((str(r.get('key_identifier') or '').strip() for r in results if isinstance(r, dict) and str(r.get('key_identifier') or '').strip()), '')
                            if group_model or group_key:
                                for u_idx in unit_indices:
                                    if group_model:
                                        model_updates[int(u_idx)] = group_model
                                    if group_key:
                                        key_updates[int(u_idx)] = group_key
                            # Never trust a worker's child list until the exact
                            # submitted group has returned a complete, usable result.
                            # Queued graceful-stop skips used to leak their children
                            # into merged_indices before the error was inspected.
                            confirmed_merged_indices = _confirmed_merged_child_indices(
                                group_result,
                                unit_indices,
                            )
                            
                            for result in results:
                                idx = result.get('idx')
                                data = result.get('data', [])
                                resp = result.get('resp', '')
                                error = result.get('error')
                                raw_obj = result.get('raw_obj')
                                chap = result.get('chap')
                                display_idx = _glossary_chapter_actual_num(idx, context=progress_context)
                                
                                if error:
                                    # Suppress expected "graceful stop" pre-send cancellations.
                                    if isinstance(error, str) and _is_graceful_stop_skip_error(error):
                                        return False
                                    print(f"[Chapter {display_idx}] Error: {error}")
                                    _mark_glossary_failed(failed, idx, "API_ERROR")
                                    _clear_glossary_in_progress(unit_indices)
                                    _save_progress_for_unit()
                                    return True
                                
                                # Process entries
                                if data and len(data) > 0:
                                    tracker_buckets = {}
                                    for entry in data:
                                        raw_for_tracker = str(entry.get("raw_name", "")).strip()
                                        matched_idx = idx
                                        if raw_for_tracker:
                                            for cand_idx, cand_chap in unit:
                                                if raw_for_tracker in str(cand_chap or ""):
                                                    matched_idx = cand_idx
                                                    break
                                        tracker_buckets.setdefault(matched_idx, []).append(entry)
                                    for tracker_idx, tracker_entries in tracker_buckets.items():
                                        extracted_entry_updates[int(tracker_idx)] = list(tracker_entries)
                                        update_gender_tracker(
                                            tracker_entries,
                                            args.output,
                                            source_path=epub_path,
                                            chapter_index=tracker_idx,
                                            chapter_num=_chapter_positions.get(tracker_idx, tracker_idx + 1),
                                            chapter_file=_chapter_filenames.get(tracker_idx, ""),
                                        )
                                    total_ent = len(data)
                                    batch_entry_count += total_ent
                                    
                                    for eidx, entry in enumerate(data, start=1):
                                        elapsed = time.time() - start
                                        entry_type = entry.get("type", "?")
                                        raw_name = entry.get("raw_name", "?")
                                        trans_name = entry.get("translated_name", "?")
                                        chapter_label = _glossary_chapter_log_label(display_idx, total_chapters, context=progress_context)
                                        print(f'{chapter_label} [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed) → {entry_type}: {raw_name} ({trans_name})')
                                        glossary.append(entry)
                                
                                # Check if this was actually a failure (empty/refused content)
                                # BUT skip this check for merged children — they intentionally have
                                # empty data/resp because their content was processed via the parent chapter
                                if 'merged_into' in result:
                                    # A child is complete only when its exact parent
                                    # request was confirmed successful above.
                                    if idx in confirmed_merged_indices:
                                        completed.append(idx)
                                else:
                                    _resp_text = resp or ''
                                    _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                                    
                                    if _is_empty_failure:
                                        print(f"⚠️ Chapter {display_idx} returned empty/refused content — marking as failed for retry")
                                        ch_finish = result.get('finish_reason', 'stop')
                                        _mark_glossary_failed(failed, idx, _glossary_issue_from_finish_reason(ch_finish))
                                    else:
                                        extracted_entry_updates.setdefault(int(idx), list(data or []))
                                        completed.append(idx)
                                        
                                        # Mark truncated chapters as failed so they get retried
                                        ch_finish = result.get('finish_reason', 'stop')
                                        if _glossary_issue_from_finish_reason(ch_finish, None) == "TRUNCATED":
                                            print(f"⚠️ Chapter {display_idx} was truncated — entries kept but chapter will be retried")
                                            _mark_glossary_failed(failed, idx, "TRUNCATED")
                                
                                # Store history for parent chapter only
                                if contextual_enabled and resp and chap and 'merged_into' not in result:
                                    system_prompt, user_prompt = build_prompt(chap)
                                    batch_history_map[idx] = (user_prompt, resp, raw_obj)

                            # Commit the merged status only after every result in
                            # this exact group has been processed successfully.
                            for mi in confirmed_merged_indices:
                                if mi not in merged_indices:
                                    merged_indices.append(mi)

                            print(f"✅ Merged group done: {len(results)} chapters")
                        else:
                            # Handle single chapter result
                            idx, chap = unit[0]
                            display_idx = _glossary_chapter_actual_num(idx, context=progress_context)
                            result = future.result(timeout=0.5)
                            _collect_request_metadata(result)
                            
                            # Process this chapter's results immediately
                            data = result.get('data', [])
                            resp = result.get('resp', '')
                            error = result.get('error')
                            raw_obj = result.get('raw_obj')
                            
                            if error:
                                # Suppress expected "graceful stop" pre-send cancellations.
                                if (isinstance(error, str) and _is_graceful_stop_skip_error(error)) or result.get('graceful_stop_skip'):
                                    return False
                                print(f"[Chapter {display_idx}] Error: {error}")
                                _mark_glossary_failed(failed, idx, "API_ERROR")
                                _clear_glossary_in_progress(unit_indices)
                                _save_progress_for_unit()
                                return True
                            
                            # Process entries as each chapter completes
                            if data and len(data) > 0:
                                update_gender_tracker(
                                    data,
                                    args.output,
                                    source_path=epub_path,
                                    chapter_index=idx,
                                    chapter_num=_chapter_positions.get(idx, idx + 1),
                                    chapter_file=_chapter_filenames.get(idx, ""),
                                )
                                total_ent = len(data)
                                batch_entry_count += total_ent
                                
                                for eidx, entry in enumerate(data, start=1):
                                    elapsed = time.time() - start
                                    
                                    # Get entry info
                                    entry_type = entry.get("type", "?")
                                    raw_name = entry.get("raw_name", "?")
                                    trans_name = entry.get("translated_name", "?")
                                    
                                    chapter_label = _glossary_chapter_log_label(display_idx, total_chapters, context=progress_context)
                                    print(f'{chapter_label} [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed) → {entry_type}: {raw_name} ({trans_name})')
                                    
                                    # Add entry immediately WITHOUT deduplication
                                    glossary.append(entry)
                            
                            # Check if this was actually a failure (empty/refused content)
                            _resp_text = resp or ''
                            _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                            
                            if _is_empty_failure:
                                print(f"⚠️ Chapter {display_idx} returned empty/refused content — marking as failed for retry")
                                ch_finish = result.get('finish_reason', 'stop')
                                _mark_glossary_failed(failed, idx, _glossary_issue_from_finish_reason(ch_finish))
                            else:
                                extracted_entry_updates[int(idx)] = list(data or [])
                                completed.append(idx)
                                
                                # Mark truncated chapters as failed so they get retried
                                ch_finish = result.get('finish_reason', 'stop')
                                if _glossary_issue_from_finish_reason(ch_finish, None) == "TRUNCATED":
                                    print(f"⚠️ Chapter {display_idx} was truncated — entries kept but chapter will be retried")
                                    _mark_glossary_failed(failed, idx, "TRUNCATED")
                            
                            # Store history entry for this chapter (will be added after batch completes)
                            if contextual_enabled and resp and chap:
                                system_prompt, user_prompt = build_prompt(chap)
                                batch_history_map[idx] = (user_prompt, resp, raw_obj)
                        
                        # Save progress after each chapter completes (crash-safe with atomic writes)
                        _clear_glossary_in_progress(unit_indices)
                        _save_progress_for_unit()
                        # Also save glossary files for incremental updates
                        save_glossary_json(glossary, args.output)
                        save_glossary_csv(glossary, args.output)
                        return True
                        
                    except Exception as e:
                        # Suppress expected "graceful stop" pre-send cancellations.
                        if _is_graceful_stop_skip_error(e):
                            return False
                        if _glossary_is_hard_stop_requested(stop_callback):
                            stopped_early = True
                            _restore_glossary_in_progress_for_hard_stop(unit_indices)
                            return False
                        if is_merged_mode:
                            # For merged mode, mark all chapters in the unit as failed on error
                            _err_lower = str(e).lower()
                            _is_user_cancel = "stopped by user" in _err_lower or "cancelled by user" in _err_lower or "operation cancelled" in _err_lower
                            for u_idx, u_chap in unit:
                                if not _is_user_cancel:
                                    print(f"Error processing merged chapter {_glossary_chapter_actual_num(u_idx, context=progress_context)}: {e}")
                                if u_idx not in completed and u_idx not in failed:
                                    _mark_glossary_failed(failed, u_idx, "API_ERROR")
                        else:
                            idx, chap = unit[0]
                            _err_lower = str(e).lower()
                            _is_user_cancel = "stopped by user" in _err_lower or "cancelled by user" in _err_lower or "operation cancelled" in _err_lower
                            if not _is_user_cancel:
                                print(f"Error processing chapter {_glossary_chapter_actual_num(idx, context=progress_context)}: {e}")
                            if idx not in completed and idx not in failed:
                                _mark_glossary_failed(failed, idx, "API_ERROR")
                        _clear_glossary_in_progress(unit_indices)
                        _save_progress_for_unit()
                        return True

                if aggressive_mode:
                    # Aggressive mode: keep pool full, auto-refill as futures complete.
                    # Use wait(FIRST_COMPLETED) so newly-submitted futures are also observed promptly.
                    active_futures = {}
                    next_unit_idx = 0
                    graceful_drain_requested = False
                    graceful_drain_announced = False
                    # Ensure batch_size is at least 1 to avoid submission loops never running
                    effective_aggressive_batch_size = max(1, batch_size)

                    def _submit_next():
                        nonlocal next_unit_idx
                        if next_unit_idx >= len(current_batch_units):
                            return False
                        unit = current_batch_units[next_unit_idx]
                        next_unit_idx += 1
                        fut = _submit_unit(unit)
                        active_futures[fut] = unit
                        return True

                    # Prime the executor to fill all slots
                    while len(active_futures) < effective_aggressive_batch_size and _submit_next():
                        pass

                    while active_futures or next_unit_idx < len(current_batch_units):
                        if _glossary_is_graceful_stop_active():
                            graceful_drain_requested = True
                        if check_stop():
                            stopped_early = True
                            for active_unit in list(active_futures.values()):
                                _restore_glossary_in_progress_for_hard_stop([u_idx for u_idx, _ in active_unit])
                            cancelled = cancel_all_futures(list(active_futures.keys()))
                            if cancelled > 0:
                                print(f"✅ Cancelled {cancelled} pending API calls")
                            executor.shutdown(wait=False)
                            active_futures.clear()
                            break

                        # Auto-refill to maintain batch_size parallel calls (but not during graceful stop)
                        if os.environ.get('GRACEFUL_STOP') != '1':
                            while len(active_futures) < effective_aggressive_batch_size and _submit_next():
                                pass

                        # Only break if truly done: no active futures AND nothing left to submit
                        if not active_futures and next_unit_idx >= len(current_batch_units):
                            break
                        
                        # If active_futures is empty but there are items left, submit them now
                        # (This handles edge cases where graceful stop was briefly set then cleared)
                        if not active_futures and next_unit_idx < len(current_batch_units):
                            if os.environ.get('GRACEFUL_STOP') == '1':
                                graceful_drain_requested = True
                                break
                            while len(active_futures) < effective_aggressive_batch_size and _submit_next():
                                pass
                            if not active_futures:
                                # Still empty after trying to submit - something's wrong, break to avoid infinite loop
                                print("⚠️ Warning: Could not submit remaining items, breaking loop")
                                break

                        done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

                        for future in done:
                            unit = active_futures.pop(future, None)
                            if unit is None:
                                continue

                            # Refill freed slot ASAP (unless graceful stop is active)
                            if os.environ.get('GRACEFUL_STOP') != '1':
                                while len(active_futures) < effective_aggressive_batch_size and _submit_next():
                                    pass

                            result_committed = _handle_future_result(future, unit)
                            if stopped_early:
                                break

                            if _graceful_stop_should_drain_after_result(result_committed):
                                graceful_drain_requested = True
                                if not graceful_drain_announced:
                                    print(
                                        "⏳ Graceful stop: Result saved; waiting for all "
                                        "remaining in-flight API calls to finish..."
                                    )
                                    graceful_drain_announced = True

                        if stopped_early:
                            break

                    if graceful_drain_requested and not stopped_early:
                        stopped_early = True
                        print("✅ Graceful stop: All completed in-flight API results were saved.")
                else:
                    # Submit all units in this batch
                    for unit in current_batch_units:
                        if check_stop():
                            stopped_early = True
                            break
                        _submit_unit(unit)

                    # Process results AS THEY COMPLETE, not all at once
                    graceful_drain_requested = False
                    graceful_drain_announced = False
                    for future in as_completed(futures):
                        if _glossary_is_graceful_stop_active():
                            graceful_drain_requested = True
                        if check_stop():
                            # print("🛑 Stop detected - cancelling all pending operations...")  # Redundant
                            stopped_early = True
                            for pending_unit in list(futures.values()):
                                _restore_glossary_in_progress_for_hard_stop([u_idx for u_idx, _ in pending_unit])
                            cancelled = cancel_all_futures(list(futures.keys()))
                            if cancelled > 0:
                                print(f"✅ Cancelled {cancelled} pending API calls")
                            executor.shutdown(wait=False)
                            break

                        unit = futures[future]
                        result_committed = _handle_future_result(future, unit)
                        
                        # Graceful stop blocks new API sends, but every request that
                        # was already in flight must still pass through the result
                        # handler and be committed before this loop exits.
                        if _graceful_stop_should_drain_after_result(result_committed):
                            graceful_drain_requested = True
                            if not graceful_drain_announced:
                                print(
                                    "⏳ Graceful stop: Result saved; waiting for all "
                                    "remaining in-flight API calls to finish..."
                                )
                                graceful_drain_announced = True

                    if graceful_drain_requested and not stopped_early:
                        stopped_early = True
                        print("✅ Graceful stop: All completed in-flight API results were saved.")
            
            # Graceful-stop workers have now been drained. Restore only a request
            # marker that never produced a committable result.
            if stopped_early:
                _restore_glossary_in_progress_for_hard_stop([])

            # After all futures in this batch complete, append history entries in order
            # This matches TransateKRtoEN batch mode behavior
            if contextual_enabled and batch_history_map:
                print(f"\n📝 Updating context history for batch {batch_num+1}...")
                # Flatten units to get all chapters and sort by index
                all_chapters_in_batch = []
                for unit in current_batch_units:
                    all_chapters_in_batch.extend(unit)
                sorted_chapters = sorted(all_chapters_in_batch, key=lambda x: x[0])
                for idx, chap in sorted_chapters:
                    if idx in batch_history_map:
                        user_content, assistant_content, raw_obj = batch_history_map[idx]
                        try:
                            history = history_manager.append_to_history(
                                user_content=user_content,
                                assistant_content=assistant_content,
                                hist_limit=ctx_limit,
                                reset_on_limit=not rolling_window,
                                rolling_window=rolling_window,
                                raw_assistant_object=raw_obj
                            )
                        except Exception as e:
                            print(f"⚠️ Failed to append Chapter {_glossary_chapter_actual_num(idx, context=progress_context)} to glossary history: {e}")
                print(f"💾 Saved glossary history ({len(history)} messages)")
            
            batch_elapsed = time.time() - batch_start_time
            print(f"[BATCH] Batch {batch_num+1} completed in {batch_elapsed:.1f}s total")
            
            # After batch completes, apply deduplication and sorting (only if not stopped early)
            if batch_entry_count > 0 and not stopped_early:
                print(f"\n🔀 Applying deduplication and sorting after batch {batch_num+1}/{total_batches}")
                original_size = len(glossary)
                
                # Apply deduplication to entire glossary
                glossary[:] = skip_duplicate_entries(glossary)
                
                # Sort glossary by type and name
                custom_types = get_custom_entry_types()
                type_order = {'book': -1, 'character': 0, 'term': 1}
                other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                for i, t in enumerate(other_types):
                    type_order[t] = i + 2
                
                glossary.sort(key=lambda x: (
                    type_order.get(x.get('type', 'term'), 999),
                    x.get('raw_name', '').lower()
                ))
                
                deduplicated_size = len(glossary)
                removed = original_size - deduplicated_size
                
                if removed > 0:
                    print(f"✅ Removed {removed} duplicates (fuzzy threshold: {os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.90')})")
                print(f"📊 Glossary size: {deduplicated_size} unique entries")
                
                # Save final deduplicated and sorted glossary
                save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                save_glossary_json(glossary, args.output)
                save_glossary_csv(glossary, args.output)
            
            # Print batch summary
            if batch_entry_count > 0:
                print(f"\n📊 Batch {batch_num+1}/{total_batches} Summary:")
                print(f"   • Chapters processed: {chapters_in_batch}")
                print(f"   • Total entries extracted: {batch_entry_count}")
                print(f"   • Glossary size: {len(glossary)} unique entries")
            
            # If stopped early, deduplicate once and exit
            if stopped_early:
                if glossary:
                    print(f"\n🔀 Deduplicating {len(glossary)} entries before exit...")
                    original_size = len(glossary)
                    glossary[:] = skip_duplicate_entries(glossary)
                    
                    custom_types = get_custom_entry_types()
                    type_order = {'book': -1, 'character': 0, 'term': 1}
                    other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                    for i, t in enumerate(other_types):
                        type_order[t] = i + 2
                    glossary.sort(key=lambda x: (
                        type_order.get(x.get('type', 'term'), 999),
                        x.get('raw_name', '').lower()
                    ))
                    
                    save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                    save_glossary_json(glossary, args.output)
                    save_glossary_csv(glossary, args.output)
                    
                    # Log the final size after deduplication
                    removed = max(0, original_size - len(glossary))
                    print(f"✅ Saved {len(glossary)} entries (after {removed} duplicates removed) before exit")
                return
            
            # Handle context history
            if contextual_enabled:
                if not rolling_window and len(history) >= ctx_limit and ctx_limit > 0:
                    print(f"🔄 Resetting glossary context (reached {ctx_limit} chapter limit)")
                    history = []
                    prog['context_history'] = []
            
            # Add delay between batches (but not after the last batch)
            if batch_num < total_batches - 1:
                print(f"\n⏱️  Waiting {api_delay}s before next batch...")
                if not interruptible_sleep(api_delay, check_stop, 0.1):
                    print(f"❌ Glossary extraction stopped during delay")
                    # Apply deduplication before stopping
                    if glossary:
                        print("🔀 Applying deduplication and sorting before exit...")
                        original_size = len(glossary)
                        glossary[:] = skip_duplicate_entries(glossary)
                        
                        # Sort glossary
                        custom_types = get_custom_entry_types()
                        type_order = {'book': -1, 'character': 0, 'term': 1}
                        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                        for i, t in enumerate(other_types):
                            type_order[t] = i + 2
                        glossary.sort(key=lambda x: (
                            type_order.get(x.get('type', 'term'), 999),
                            x.get('raw_name', '').lower()
                        ))
                        
                        save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)
                        save_glossary_json(glossary, args.output)
                        save_glossary_csv(glossary, args.output)
                    
                        # Log the final size after deduplication
                        removed = max(0, original_size - len(glossary))
                        print(f"✅ Saved {len(glossary)} entries (after {removed} duplicates removed) before exit")
                    return
    
    else:
        # SEQUENTIAL MODE: Original behavior
        
        # Request merging preprocessing
        merge_groups = {}  # Maps parent_idx -> list of (idx, chap) tuples
        merged_children = set()  # Children merged into a parent for this run only
        
        if request_merging_enabled and request_merge_count > 1:
            # Collect chapters that need processing (not completed, not merged)
            chapters_needing_processing = []
            for idx, chap in enumerate(chapters):
                # Skip if already completed or merged
                if idx in completed:
                    continue
                
                # Apply chapter range filter
                if range_start is not None and range_end is not None:
                    chapter_num = _chapter_positions.get(idx, idx + 1)
                    if not (range_start <= chapter_num <= range_end):
                        continue
                
                chapters_needing_processing.append((idx, chap))
            
            if chapter_split_enabled:
                # Budget-aware grouping (auto-adjust to fit available_tokens)
                try:
                    from TransateKRtoEN import RequestMerger
                except Exception:
                    RequestMerger = None

                run_groups = []
                if RequestMerger:
                    run_groups = RequestMerger.create_merge_groups(
                        chapters_needing_processing,
                        max(1, len(chapters_needing_processing)),
                    )
                else:
                    run_groups = [chapters_needing_processing]

                for run in run_groups:
                    if not run:
                        continue
                    i = 0
                    while i < len(run):
                        group = [run[i]]
                        i += 1
                        
                        while i < len(run) and len(group) < request_merge_count:
                            candidate = run[i]
                            merged_preview = "\n\n".join([c for (_, c) in group + [candidate]])
                            merged_tokens = chapter_splitter.count_tokens(merged_preview)
                            
                            if merged_tokens <= available_tokens:
                                group.append(candidate)
                                i += 1
                            else:
                                break
                        
                        parent_idx = group[0][0]
                        merge_groups[parent_idx] = group
                        
                        if len(group) > 1:
                            child_indices = [g[0] for g in group[1:]]
                            for child_idx in child_indices:
                                merged_children.add(child_idx)
            else:
                # Count-based grouping when chapter splitting toggle is OFF
                # Still enforce token budget to prevent oversized requests
                i = 0
                while i < len(chapters_needing_processing):
                    group = [chapters_needing_processing[i]]
                    i += 1
                    while i < len(chapters_needing_processing) and len(group) < request_merge_count:
                        candidate = chapters_needing_processing[i]
                        merged_preview = "\n\n".join([c for (_, c) in group + [candidate]])
                        merged_tokens = chapter_splitter.count_tokens(merged_preview)
                        if merged_tokens <= available_tokens:
                            group.append(candidate)
                            i += 1
                        else:
                            break
                    parent_idx = group[0][0]
                    merge_groups[parent_idx] = group
                    if len(group) > 1:
                        child_indices = [g[0] for g in group[1:]]
                        for child_idx in child_indices:
                            merged_children.add(child_idx)
            
            if merge_groups:
                mode_label = "budget-aware" if chapter_split_enabled else "count-based"
                multi_groups = {p: g for p, g in merge_groups.items() if len(g) > 1}
                if multi_groups:
                    group_descs = [f"{p+1}+{[g[0]+1 for g in grp[1:]]}" for p, grp in sorted(multi_groups.items())]
                    print(f"   📎 {len(multi_groups)} merge groups ({mode_label}): {', '.join(group_descs)}")
                print(f"   📊 Created {len(merge_groups)} merge groups total ({mode_label})")
        
        for idx, chap in enumerate(chapters):
            # Check for graceful stop completion at START of each chapter iteration
            # This allows the previous chapter to fully complete (including save) before stopping
            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                print(f"✅ Graceful stop: Previous chapter completed and saved, stopping extraction...")
                return
            
            # If graceful stop requested but NO API call is active, stop immediately
            # (nothing to wait for - no in-flight calls to complete)
            if os.environ.get('GRACEFUL_STOP') == '1' and os.environ.get('GRACEFUL_STOP_API_ACTIVE') != '1':
                print(f"✅ Graceful stop: No API call in progress, stopping immediately...")
                return
            
            # Check for stop at the beginning of each chapter
            if check_stop():
                print(f"❌ Glossary extraction stopped at chapter {_glossary_chapter_actual_num(idx, context=progress_context)}")
                return
            
            # Skip if this chapter was merged into another (current run only)
            if idx in merged_children:
                continue
            
            # Apply chapter range filter
            if range_start is not None and range_end is not None:
                chapter_num = _chapter_positions.get(idx, idx + 1)
                if not (range_start <= chapter_num <= range_end):
                    # Track skipped chapters for summary (don't print individually)
                    if '_skipped_chapters' not in globals():
                        _skipped_chapters = []
                    is_text_chapter = hasattr(chap, 'filename') and chap.get('filename', '').endswith('.txt')
                    terminology = "Section" if is_text_chapter else "Chapter"
                    _skipped_chapters.append((chapter_num, terminology))
                    continue
                
            if idx in completed:
                # Check if processing text file chapters
                is_text_chapter = hasattr(chap, 'filename') and chap.get('filename', '').endswith('.txt')
                terminology = "section" if is_text_chapter else "chapter"
                # print(f"Skipping {terminology} {idx+1} (already processed)")  # Redundant - already shown in summary
                continue
                    
            # Show filename alongside chapter number when available
            _fname = _chapter_filenames.get(idx, '')
            _chap_num = _glossary_chapter_actual_num(idx, context=progress_context)
            _chap_total = _glossary_chapter_display_total(total_chapters, context=progress_context)
            _chap_label = f"Chapter {_chap_num}/{_chap_total}" if _chap_total > 0 else f"Chapter {_chap_num}"
            if _fname:
                print(f"🔄 Processing {_chap_label} ({_fname})")
            else:
                print(f"🔄 Processing {_chap_label}")
            
            # Request merging: If this is a parent chapter, merge content from child chapters
            chapter_content = chap
            if idx in merge_groups:
                group = merge_groups[idx]
                print(f"\n🔗 MERGING {len(group)} chapters into single request...")
                merged_contents = []
                for g_idx, g_chap in group:
                    # Don't add separators - glossary extraction doesn't need them
                    merged_contents.append(g_chap)
                    if g_idx != idx:
                        print(f"   → Including chapter {_glossary_chapter_actual_num(g_idx, context=progress_context)}")
                
                chapter_content = "\n\n".join(merged_contents)
                print(f"   📊 Merged content: {len(chapter_content):,} characters")
            
            # Build merged chapter nums for watchdog progress bar
            merged_chapter_nums = [_glossary_chapter_actual_num(g_idx, context=progress_context) for g_idx, _ in merge_groups[idx]] if idx in merge_groups else None
            current_progress_indices = [g_idx for g_idx, _ in merge_groups[idx]] if idx in merge_groups else [idx]
            current_model_updates = {}
            current_key_updates = {}
            current_extracted_entry_updates = {}

            def _remember_current_request_metadata():
                model_name = _current_glossary_model_name({}, prefer_thread=True)
                key_identifier, _key_pool = _current_glossary_key_context({}, prefer_thread=True)
                if not (model_name or key_identifier):
                    return
                for progress_idx in current_progress_indices:
                    try:
                        progress_idx = int(progress_idx)
                    except (TypeError, ValueError):
                        continue
                    if model_name:
                        current_model_updates[progress_idx] = model_name
                    if key_identifier:
                        current_key_updates[progress_idx] = key_identifier

            def _mark_current_glossary_progress_on_send():
                _mark_glossary_in_progress(current_progress_indices)

            # Check if history will reset on this chapter
            if contextual_enabled and len(history) >= ctx_limit and ctx_limit > 0 and not rolling_window:
                print(f"  📌 Glossary context will reset after this chapter (current: {len(history)}/{ctx_limit} chapters)")        

            try:
                # Get system and user prompts from build_prompt
                system_prompt, user_prompt = build_prompt(chapter_content)
                
                # Build optional assistant prefill message if configured
                assistant_prefill_msgs = []
                assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
                if assistant_prompt_env:
                    assistant_prefill_msgs = [{"role": "assistant", "content": assistant_prompt_env}]
                
                if not contextual_enabled:
                    # No context at all
                    msgs = [
                        {"role": "system", "content": system_prompt},
                    ] + assistant_prefill_msgs + [
                        {"role": "user", "content": user_prompt}
                    ]
                else:
                    # Get context history (may be natural conversation or memory blocks)
                    # Microsecond lock to prevent race conditions when reading history
                    time.sleep(0.000001)
                    with _history_lock:
                        context_msgs = trim_context_history(history, ctx_limit, rolling_window)
                    
                    # Check if we're using Gemini 3 (natural conversation format)
                    model = os.getenv("MODEL", "gemini-2.0-flash").lower()
                    is_gemini_3 = "gemini-3" in model or "gemini-exp-1206" in model
                    
                    if is_gemini_3:
                        # For Gemini 3, context_msgs is the natural conversation history
                        # Build: system + history + current user prompt
                        msgs = [{"role": "system", "content": system_prompt}] \
                             + context_msgs \
                             + assistant_prefill_msgs \
                             + [{"role": "user", "content": user_prompt}]
                    else:
                        # For other models, context_msgs is memory blocks (assistant messages)
                        # Build: system + memory + current user prompt  
                        msgs = [{"role": "system", "content": system_prompt}] \
                             + context_msgs \
                             + assistant_prefill_msgs \
                             + [{"role": "user", "content": user_prompt}]
                
                # Compute total and assistant/memory tokens for this chapter
                total_tokens = 0
                assistant_tokens = 0
                for m in msgs:
                    content = m.get("content", "") or ""
                    tokens = count_tokens(content)
                    total_tokens += tokens
                    if m.get("role") == "assistant":
                        assistant_tokens += tokens
                non_assistant_tokens = total_tokens - assistant_tokens
                
                # READ THE TOKEN LIMIT
                env_value = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
                if not env_value or env_value == "":
                    token_limit = None
                    limit_str = "unlimited"
                elif env_value.isdigit() and int(env_value) > 0:
                    token_limit = int(env_value)
                    limit_str = str(token_limit)
                else:
                    token_limit = 1000000
                    limit_str = "1000000 (default)"
                
                # Log combined prompt similar to main translator (use safe chunk budget)
                if contextual_enabled and assistant_tokens > 0:
                    print(
                        f"💬 {_chap_label} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant_tokens:,}, "
                        f"assistant/memory: {assistant_tokens:,}) | chunk budget {available_tokens:,}"
                    )
                else:
                    print(
                        f"💬 {_chap_label} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) | chunk budget {available_tokens:,}"
                    )

                # Determine if we need to split based on output-limit budget
                chapter_had_truncated_chunk = False
                chapter_tokens = chapter_splitter.count_tokens(chapter_content)
                if chapter_split_enabled and chapter_tokens > available_tokens:
                    print(f"⚠️ Chapter {_chap_num} exceeds chunk budget: {chapter_tokens:,} > {available_tokens:,}")
                    print(f"📄 Using ChapterSplitter to split into smaller chunks (output-limit safe)...")

                    # Since glossary extraction works with plain text, wrap it in a simple HTML structure
                    chapter_html = f"<html><body><p>{chap.replace(chr(10)+chr(10), '</p><p>')}</p></body></html>"

                    # Use ChapterSplitter to split the chapter
                    # No filename passed as this is EPUB content (not plain text files)
                    chunks = chapter_splitter.split_chapter(chapter_html, available_tokens)
                    print(f"📄 Chapter split into {len(chunks)} chunks (budget {available_tokens:,})")
                    
                    # Process each chunk
                    chapter_glossary_data = []  # Collect data from all chunks
                    
                    for chunk_html, chunk_idx, total_chunks in chunks:
                        if check_stop():
                            print(f"❌ Glossary extraction stopped during chunk {chunk_idx} of chapter {_chap_num}")
                            _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                            return
                            
                        print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {_chap_num}")
                        
                        # Extract text from the chunk HTML
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(chunk_html, 'html.parser')
                        chunk_text = soup.get_text(strip=True)
                        
                        # Get system and user prompts for chunk
                        chunk_system_prompt, chunk_user_prompt = build_prompt(chunk_text)
                        
                        # Build optional assistant prefill message if configured
                        chunk_assistant_prefill_msgs = []
                        chunk_assistant_prompt_env = os.getenv('ASSISTANT_PROMPT', '').strip()
                        if chunk_assistant_prompt_env:
                            chunk_assistant_prefill_msgs = [{"role": "assistant", "content": chunk_assistant_prompt_env}]

                        # Build chunk messages
                        if not contextual_enabled:
                            chunk_msgs = [
                                {"role": "system", "content": chunk_system_prompt},
                            ] + chunk_assistant_prefill_msgs + [
                                {"role": "user", "content": chunk_user_prompt}
                            ]
                        else:
                            # Get context history (may be natural conversation or memory blocks)
                            # Microsecond lock to prevent race conditions when reading history
                            time.sleep(0.000001)
                            with _history_lock:
                                context_msgs = trim_context_history(history, ctx_limit, rolling_window)
                            
                            # Check if we're using Gemini 3 (natural conversation format)
                            model = os.getenv("MODEL", "gemini-2.0-flash").lower()
                            is_gemini_3 = "gemini-3" in model or "gemini-exp-1206" in model
                            
                            if is_gemini_3:
                                # For Gemini 3, context_msgs is the natural conversation history
                                # Build: system + history + current user prompt
                                chunk_msgs = [{"role": "system", "content": chunk_system_prompt}] \
                                           + context_msgs \
                                           + chunk_assistant_prefill_msgs \
                                           + [{"role": "user", "content": chunk_user_prompt}]
                            else:
                                # For other models, context_msgs is memory blocks (assistant messages)
                                # Build: system + memory + current user prompt
                                chunk_msgs = [{"role": "system", "content": chunk_system_prompt}] \
                                           + context_msgs \
                                           + chunk_assistant_prefill_msgs \
                                           + [{"role": "user", "content": chunk_user_prompt}]

                        # Build messages following the translation pattern for _raw_content_object handling
                        filtered_chunk_msgs = []
                        for msg in chunk_msgs:
                            filtered_msg = {}
                            
                            # Check if this message has _raw_content_object with parts
                            has_raw_parts = False
                            if '_raw_content_object' in msg:
                                raw_obj = msg['_raw_content_object']
                                if isinstance(raw_obj, dict) and 'parts' in raw_obj:
                                    has_raw_parts = True
                                elif hasattr(raw_obj, 'parts'):
                                    has_raw_parts = True
                            
                            # For assistant messages with raw parts, include BOTH content and _raw_content_object
                            # The content field is needed as fallback when parts don't have text
                            if msg.get('role') == 'assistant' and has_raw_parts:
                                # Include role, content, AND _raw_content_object
                                filtered_msg['role'] = msg['role']
                                filtered_msg['content'] = msg.get('content', '')
                                filtered_msg['_raw_content_object'] = msg['_raw_content_object']
                            else:
                                # For other messages, copy everything as-is
                                filtered_msg = msg.copy()
                            
                            filtered_chunk_msgs.append(filtered_msg)
                        
                        # API call for chunk with FILTERED + SANITIZED messages
                        send_msgs = _sanitize_messages_for_api(filtered_chunk_msgs, chunk_text)
                        try:
                            chunk_raw, chunk_finish_reason, chunk_raw_obj = send_with_interrupt(
                                messages=send_msgs,  # Use sanitized messages
                                client=client,
                                temperature=temp,
                                max_tokens=mtoks,
                                stop_check_fn=check_stop,
                                chunk_timeout=chunk_timeout,
                                chapter_idx=idx,
                                chapter_num=_chap_num,
                                chunk_idx=chunk_idx,
                                total_chunks=total_chunks,
                                merged_chapters=merged_chapter_nums,
                                before_send_callback=_mark_current_glossary_progress_on_send,
                            )
                            _remember_current_request_metadata()
                            if _glossary_issue_from_finish_reason(chunk_finish_reason, None) == "TRUNCATED":
                                chapter_had_truncated_chunk = True
                        except UnifiedClientError as e:
                            if "stopped by user" in str(e).lower():
                                print(f"❌ Glossary extraction stopped during chunk {chunk_idx} API call")
                                _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                                return
                            elif "timeout" in str(e).lower():
                                print(f"⚠️ Chunk {chunk_idx} API call timed out: {e}")
                                continue  # Skip this chunk
                            else:
                                print(f"❌ Chunk {chunk_idx} API error: {e}")
                                continue  # Skip this chunk
                        except Exception as e:
                            print(f"❌ Unexpected error in chunk {chunk_idx}: {e}")
                            continue  # Skip this chunk
                        
                        # Process chunk response
                        if chunk_raw is None:
                            print(f"❌ API returned None for chunk {chunk_idx}")
                            continue

                        # Handle different response types
                        if isinstance(chunk_raw, tuple):
                            chunk_resp = chunk_raw[0] if chunk_raw[0] is not None else ""
                        elif isinstance(chunk_raw, str):
                            chunk_resp = chunk_raw
                        elif hasattr(chunk_raw, 'content'):
                            chunk_resp = chunk_raw.content if chunk_raw.content is not None else ""
                        elif hasattr(chunk_raw, 'text'):
                            chunk_resp = chunk_raw.text if chunk_raw.text is not None else ""
                        else:
                            print(f"❌ Unexpected response type for chunk {chunk_idx}: {type(chunk_raw)}")
                            chunk_resp = str(chunk_raw) if chunk_raw is not None else ""

                        # Ensure resp is a string
                        if not isinstance(chunk_resp, str):
                            print(f"⚠️ Converting non-string response to string for chunk {chunk_idx}")
                            chunk_resp = str(chunk_resp) if chunk_resp is not None else ""

                        # Check if response is empty
                        if not chunk_resp or chunk_resp.strip() == "":
                            print(f"⚠️ Empty response for chunk {chunk_idx}, skipping...")
                            continue
                        
                        # Save chunk response with thread-safe location
                        thread_name = threading.current_thread().name
                        thread_id = threading.current_thread().ident
                        thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                        try:
                            os.makedirs(thread_dir, exist_ok=True)
                        except (PermissionError, OSError):
                            import tempfile
                            thread_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "glossary", f"{thread_name}_{thread_id}")
                            try:
                                os.makedirs(thread_dir, exist_ok=True)
                            except Exception:
                                thread_dir = None
                        
                        if thread_dir:
                            try:
                                with open(os.path.join(thread_dir, f"chunk_response_chap{idx+1}_chunk{chunk_idx}.txt"), "w", encoding="utf-8", errors="replace") as f:
                                    f.write(chunk_resp)
                            except (PermissionError, OSError):
                                pass
                        
                        # Extract data from chunk
                        chunk_resp_data = parse_api_response(chunk_resp)

                        if not chunk_resp_data:
                            print(f"[Warning] No data found in chunk {chunk_idx}, skipping...")
                            continue

                        # The parse_api_response already returns parsed data, no need to parse again
                        try:
                            # Filter out invalid entries directly from chunk_resp_data
                            valid_chunk_data = []
                            for entry in chunk_resp_data:
                                if validate_extracted_entry(entry):
                                    # Clean the raw_name
                                    if 'raw_name' in entry:
                                        entry['raw_name'] = entry['raw_name'].strip()
                                    valid_chunk_data.append(entry)
                                else:
                                    print(f"[Debug] Skipped invalid entry in chunk {chunk_idx}: {entry}")
                            
                            chapter_glossary_data.extend(valid_chunk_data)
                            print(f"✅ Chunk {chunk_idx}/{total_chunks}: extracted {len(valid_chunk_data)} entries")
                            
                            # Add chunk to history if contextual
                            if contextual_enabled:
                                try:
                                    history = history_manager.append_to_history(
                                        user_content=chunk_user_prompt,
                                        assistant_content=chunk_resp,
                                        hist_limit=ctx_limit,
                                        reset_on_limit=not rolling_window,
                                        rolling_window=rolling_window,
                                        raw_assistant_object=chunk_raw_obj
                                    )
                                except Exception as e:
                                    print(f"⚠️ Failed to save chunk {chunk_idx} history: {e}")

                        except Exception as e:
                            print(f"[Warning] Error processing chunk {chunk_idx} data: {e}")
                            continue
                        
                        # Add delay between chunks (but not after last chunk)
                        if chunk_idx < total_chunks:
                            print(f"⏱️  Waiting {api_delay}s before next chunk...")
                            if not interruptible_sleep(api_delay, check_stop, 0.1):
                                print(f"❌ Glossary extraction stopped during chunk delay")
                                _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                                return
                    
                    # Use the collected data from all chunks
                    data = chapter_glossary_data
                    resp = ""  # Combined response not needed for progress tracking
                    # Set raw_obj to None for chunked processing (history was already saved per chunk)
                    raw_obj = None
                    print(f"✅ Chapter {_chap_num} processed in {len(chunks)} chunks, total entries: {len(data)}")
                    
                else:
                    # Original single-chapter processing
                    # Check for stop before API call
                    if check_stop():
                        print(f"❌ Glossary extraction stopped before API call for chapter {_chap_num}")
                        _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                        return
                
                    # Build messages following the translation pattern for _raw_content_object handling
                    filtered_msgs = []
                    for msg in msgs:
                        filtered_msg = {}
                        
                        # Check if this message has _raw_content_object with parts
                        has_raw_parts = False
                        if '_raw_content_object' in msg:
                            raw_obj = msg['_raw_content_object']
                            if isinstance(raw_obj, dict) and 'parts' in raw_obj:
                                has_raw_parts = True
                            elif hasattr(raw_obj, 'parts'):
                                has_raw_parts = True
                        
                        # For assistant messages with raw parts, include BOTH content and _raw_content_object
                        # The content field is needed as fallback when parts don't have text
                        if msg.get('role') == 'assistant' and has_raw_parts:
                            # Include role, content, AND _raw_content_object
                            filtered_msg['role'] = msg['role']
                            filtered_msg['content'] = msg.get('content', '')
                            filtered_msg['_raw_content_object'] = msg['_raw_content_object']
                        else:
                            # For other messages, copy everything as-is
                            filtered_msg = msg.copy()
                        
                        filtered_msgs.append(filtered_msg)
                    
                    raw_obj = None
                    try:
                        # Use send_with_interrupt for API call with FILTERED messages
                        raw, finish_reason, raw_obj = send_with_interrupt(
                            messages=filtered_msgs,  # Use filtered messages
                            client=client,
                            temperature=temp,
                            max_tokens=mtoks,
                            stop_check_fn=check_stop,
                            chunk_timeout=chunk_timeout,
                            chapter_idx=idx,
                            chapter_num=_chap_num,
                            merged_chapters=merged_chapter_nums,
                            before_send_callback=_mark_current_glossary_progress_on_send,
                        )
                        _remember_current_request_metadata()
                                
                    except UnifiedClientError as e:
                        if "stopped by user" in str(e).lower():
                            print(f"❌ Glossary extraction stopped during API call for chapter {_chap_num}")
                            _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                            return
                        elif "timeout" in str(e).lower():
                            print(f"⚠️ API call timed out for chapter {_chap_num}: {e}")
                            continue
                        else:
                            print(f"❌ API error for chapter {_chap_num}: {e}")
                            continue
                    except Exception as e:
                        print(f"❌ Unexpected error for chapter {_chap_num}: {e}")
                        continue
                    
                    # Handle response
                    if raw is None:
                        print(f"❌ API returned None for chapter {_chap_num}")
                        continue

                    # Handle different response types
                    if isinstance(raw, tuple):
                        resp = raw[0] if raw[0] is not None else ""
                    elif isinstance(raw, str):
                        resp = raw
                    elif hasattr(raw, 'content'):
                        resp = raw.content if raw.content is not None else ""
                    elif hasattr(raw, 'text'):
                        resp = raw.text if raw.text is not None else ""
                    else:
                        print(f"❌ Unexpected response type for chapter {_chap_num}: {type(raw)}")
                        resp = str(raw) if raw is not None else ""

                    # Ensure resp is a string
                    if not isinstance(resp, str):
                        print(f"⚠️ Converting non-string response to string for chapter {_chap_num}")
                        resp = str(resp) if resp is not None else ""

                    # NULL CHECK before checking if response is empty
                    if resp is None:
                        print(f"⚠️ Response is None for chapter {_chap_num}, skipping...")
                        _mark_glossary_failed(failed, idx, _glossary_issue_from_finish_reason(finish_reason))
                        continue

                    # Check if response is empty
                    if not resp or resp.strip() == "":
                        print(f"⚠️ Empty response for chapter {_chap_num}, skipping...")
                        _mark_glossary_failed(failed, idx, _glossary_issue_from_finish_reason(finish_reason))
                        continue

                    # Save the raw response with thread-safe location
                    thread_name = threading.current_thread().name
                    thread_id = threading.current_thread().ident
                    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                    try:
                        os.makedirs(thread_dir, exist_ok=True)
                    except (PermissionError, OSError):
                        import tempfile
                        thread_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "glossary", f"{thread_name}_{thread_id}")
                        try:
                            os.makedirs(thread_dir, exist_ok=True)
                        except Exception:
                            thread_dir = None
                    
                    if thread_dir:
                        try:
                            with open(os.path.join(thread_dir, f"response_chap{idx+1}.txt"), "w", encoding="utf-8", errors="replace") as f:
                                f.write(resp)
                        except (PermissionError, OSError):
                            pass

                    # Parse response using the new parser
                    try:
                        data = parse_api_response(resp)
                    except Exception as e:
                        print(f"❌ Error parsing response for chapter {_chap_num}: {e}")
                        print(f"   Response preview: {resp[:200] if resp else 'None'}...")
                        _mark_glossary_failed(failed, idx, "PARSE_ERROR")
                        continue
                    
                    # Filter out invalid entries
                    valid_data = []
                    for entry in data:
                        if validate_extracted_entry(entry):
                            # Clean the raw_name
                            if 'raw_name' in entry:
                                entry['raw_name'] = entry['raw_name'].strip()
                            valid_data.append(entry)
                        else:
                            print(f"[Debug] Skipped invalid entry: {entry}")
                    
                    data = valid_data
                    total_ent = len(data)
                    
                    # Log entries
                    for eidx, entry in enumerate(data, start=1):
                        if check_stop():
                            print(f"❌ Glossary extraction stopped during entry processing for chapter {_chap_num}")
                            _restore_glossary_in_progress_for_hard_stop(current_progress_indices)
                            return
                            
                        elapsed = time.time() - start
                        if idx == 0 and eidx == 1:
                            eta = 0
                        else:
                            avg = elapsed / ((idx * 100) + eidx)
                            eta = avg * (total_chapters * 100 - ((idx * 100) + eidx))
                        
                        # Get entry info based on new format
                        entry_type = entry.get("type", "?")
                        raw_name = entry.get("raw_name", "?")
                        trans_name = entry.get("translated_name", "?")
                        
                        chapter_label = _glossary_chapter_log_label(_chap_num, total_chapters, context=progress_context)
                        print(f'{chapter_label} [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry_type}: {raw_name} ({trans_name})')
                    
                # Check if this was actually a failure (empty/refused content)
                _resp_text = locals().get('resp', '') or ''
                _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                
                if _is_empty_failure:
                    # Empty/refused response — mark as failed so it gets retried
                    print(f"⚠️ Chapter {_chap_num} returned empty/refused content — marking as failed for retry")
                    _fr = locals().get('finish_reason') or locals().get('chunk_finish_reason', 'stop')
                    if chapter_had_truncated_chunk:
                        _fr = 'length'
                    empty_issue = _glossary_issue_from_finish_reason(_fr)
                    _mark_glossary_failed(failed, idx, empty_issue)
                    # Also mark merged children as failed
                    if idx in merge_groups:
                        for g_idx, _ in merge_groups[idx]:
                            if g_idx != idx:
                                _mark_glossary_failed(failed, g_idx, empty_issue)
                            if g_idx != idx and g_idx not in merged_indices:
                                merged_indices.append(g_idx)
                else:
                    # Apply skip logic and save
                    tracker_units = merge_groups.get(idx, [(idx, chap)]) if isinstance(merge_groups, dict) else [(idx, chap)]
                    tracker_buckets = {}
                    for entry in data:
                        raw_for_tracker = str(entry.get("raw_name", "")).strip()
                        matched_idx = idx
                        if raw_for_tracker:
                            for cand_idx, cand_chap in tracker_units:
                                if raw_for_tracker in str(cand_chap or ""):
                                    matched_idx = cand_idx
                                    break
                        tracker_buckets.setdefault(matched_idx, []).append(entry)
                    for tracker_idx, tracker_entries in tracker_buckets.items():
                        current_extracted_entry_updates[int(tracker_idx)] = list(tracker_entries)
                        update_gender_tracker(
                            tracker_entries,
                            args.output,
                            source_path=epub_path,
                            chapter_index=tracker_idx,
                            chapter_num=_chapter_positions.get(tracker_idx, tracker_idx + 1),
                            chapter_file=_chapter_filenames.get(tracker_idx, ""),
                        )
                    glossary.extend(data)
                    glossary[:] = skip_duplicate_entries(glossary)
                    current_extracted_entry_updates.setdefault(int(idx), list(data or []))
                    completed.append(idx)
                    
                    # Mark truncated chapters as failed so they get retried
                    # finish_reason comes from single-chapter mode, chunk_finish_reason from chunked mode
                    _fr = locals().get('finish_reason') or locals().get('chunk_finish_reason', 'stop')
                    if chapter_had_truncated_chunk:
                        _fr = 'length'
                    if _glossary_issue_from_finish_reason(_fr, None) == "TRUNCATED":
                        print(f"⚠️ Chapter {_chap_num} was truncated — entries kept but chapter will be retried")
                        _mark_glossary_failed(failed, idx, "TRUNCATED")
                    
                    # If this was a merged request, also mark child chapters as completed
                    if idx in merge_groups:
                        marked_children = []
                        for g_idx, _ in merge_groups[idx]:
                            if g_idx != idx and g_idx not in completed:
                                completed.append(g_idx)
                                marked_children.append(_glossary_chapter_actual_num(g_idx, context=progress_context))
                            if g_idx != idx and g_idx not in merged_indices:
                                merged_indices.append(g_idx)
                        if marked_children:
                            print(f"   ✅ Marked chapters {marked_children} as completed (merged with {_chap_num})")

                # Only add to history if contextual is enabled
                if contextual_enabled:
                    # Check if we processed in chunks or single
                    was_chunked = 'chunks' in locals() and isinstance(chunks, list) and len(chunks) > 0
                    
                    if was_chunked:
                        # Already added to history during chunk processing
                        pass
                    elif 'resp' in locals() and resp:
                        try:
                            history = history_manager.append_to_history(
                                user_content=user_prompt,
                                assistant_content=resp,
                                hist_limit=ctx_limit,
                                reset_on_limit=not rolling_window,
                                rolling_window=rolling_window,
                                raw_assistant_object=raw_obj if 'raw_obj' in locals() else None
                            )
                        except Exception as e:
                            print(f"⚠️ Failed to save history for chapter {_chap_num}: {e}")

                _clear_glossary_in_progress(current_progress_indices)
                save_progress(
                    completed,
                    glossary,
                    merged_indices,
                    failed=failed,
                    in_progress=in_progress,
                    context=progress_context,
                    model_update_indices=sorted(set(current_model_updates) | set(current_key_updates)),
                    model_updates=current_model_updates,
                    key_updates=current_key_updates,
                    extracted_entries_updates=current_extracted_entry_updates,
                )
                save_glossary_json(glossary, args.output)
                save_glossary_csv(glossary, args.output)
                
                # Add delay before next API call (but not after the last chapter)
                if idx < len(chapters) - 1:
                    # Check if we're within the range or if there are more chapters to process
                    next_chapter_in_range = True
                    if range_start is not None and range_end is not None:
                        next_chapter_num = _chapter_positions.get(idx + 1, idx + 2)
                        next_chapter_in_range = (range_start <= next_chapter_num <= range_end)
                    else:
                        # No range filter, check if next chapter is already completed
                        next_chapter_in_range = (idx + 1) not in completed
                    
                    if next_chapter_in_range:
                        print(f"⏱️  Waiting {api_delay}s before next chapter...")
                        if not interruptible_sleep(api_delay, check_stop, 0.1):
                            print(f"❌ Glossary extraction stopped during delay")
                            return
                            
                # Check for stop after processing chapter
                if check_stop():
                    print(f"❌ Glossary extraction stopped after processing chapter {_chap_num}")
                    return

            except Exception as e:
                if _glossary_is_hard_stop_requested(stop_callback):
                    _restore_glossary_in_progress_for_hard_stop(locals().get('current_progress_indices', [idx]))
                    print(f"❌ Glossary extraction stopped after error in chapter {locals().get('_chap_num', idx + 1)}")
                    return
                print(f"Error at chapter {locals().get('_chap_num', idx + 1)}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                _mark_glossary_failed(failed, idx, "API_ERROR")
                _clear_glossary_in_progress(locals().get('current_progress_indices', [idx]))
                save_progress(
                    completed,
                    glossary,
                    merged_indices,
                    failed=failed,
                    in_progress=in_progress,
                    context=progress_context,
                    model_update_indices=sorted(set(locals().get('current_model_updates', {})) | set(locals().get('current_key_updates', {}))),
                    model_updates=locals().get('current_model_updates', {}),
                    key_updates=locals().get('current_key_updates', {}),
                )
                # Check for stop even after error
                if check_stop():
                    print(f"❌ Glossary extraction stopped after error in chapter {locals().get('_chap_num', idx + 1)}")
                    return
    
    # Print skip summary if any chapters were skipped
    if '_skipped_chapters' in globals() and _skipped_chapters:
        skipped = _skipped_chapters
        print(f"\n📊 Skipped {len(skipped)} chapters outside range {range_start}-{range_end}")
        if len(skipped) <= 10:
            chapter_list = ', '.join([f"{term} {num}" for num, term in skipped])
            print(f"   Skipped: {chapter_list}")
        else:
            chapter_nums = [num for num, _ in skipped]
            print(f"   Range: {min(chapter_nums)} to {max(chapter_nums)}")
        # Clear the list
        _skipped_chapters = []
    
    # Print failed chapters summary
    if failed:
        issue_map = _normalize_glossary_qa_issues(_GLOSSARY_QA_ISSUES_FOUND)
        print(f"\n⚠️ {len(failed)} chapter(s) failed and will be retried on next run:")
        issue_set = set()
        for idx in sorted(failed):
            issues = issue_map.get(idx) or ["UNKNOWN"]
            issue_set.update(issues)
            try:
                chapter_num = _glossary_chapter_actual_num(idx, context=progress_context)
            except Exception:
                chapter_num = idx + 1
            try:
                chapter_file = _glossary_chapter_output_file(idx, context=progress_context)
            except Exception:
                chapter_file = ""
            file_note = f" ({chapter_file})" if chapter_file else ""
            print(f"   • Chapter {chapter_num}{file_note}: {', '.join(issues)}")

        print("\n   What these QA issues mean:")
        if "TRUNCATED" in issue_set:
            print(
                "   • TRUNCATED: the provider/server ended the response early. "
                "Increase the glossary compression factor or reduce the output token limit, use a different model, "
                "or increase the auto-retry truncated value."
            )
        if "PROHIBITED_CONTENT" in issue_set or "PROHIBITED CONTENT" in issue_set:
            print(
                "   • PROHIBITED_CONTENT: the model/provider likely blocked the request because of safety or censorship. "
                "Use a different model/provider for that chapter."
            )
        if "SPLIT_FAILED" in issue_set:
            print(
                "   • SPLIT_FAILED: the AI ignored or mishandled the split-marker instructions, so the output could not "
                "be safely mapped back to the original split chapters."
            )

        if "API_ERROR" in issue_set:
            print(
                "   - API_ERROR: the provider/API request failed before a usable response was returned. "
                "Retry the chapter, check the API/provider logs, or switch model/provider if it repeats."
            )

        known_issues = {"TRUNCATED", "PROHIBITED_CONTENT", "PROHIBITED CONTENT", "SPLIT_FAILED", "API_ERROR"}
        other_issues = sorted(issue for issue in issue_set if issue not in known_issues)
        if other_issues:
            print(
                f"   • Other issue(s) ({', '.join(other_issues)}): the exact cause is not known from the saved marker. "
                "Retry the chapter, check the surrounding logs, or switch model/provider if it repeats."
            )
        save_progress(completed, glossary, merged_indices, failed=failed, context=progress_context)

    if _glossary_refinement_enabled() and not check_stop():
        print(
            f"📊 Glossary refinement chunk budget: {refinement_available_tokens:,} tokens "
            f"(output limit {effective_output_tokens:,}, margin 500, compression {refinement_compression_factor})"
        )
        before_refinement_count = len(glossary)
        glossary = refine_glossary_entries(
            glossary,
            client=client,
            temp=temp,
            mtoks=mtoks,
            check_stop=check_stop,
            chapter_splitter=refinement_splitter,
            available_tokens=refinement_available_tokens,
            chunk_timeout=chunk_timeout,
            parse_response_fn=parse_api_response,
            dedupe_fn=skip_duplicate_entries,
            custom_entry_types_fn=get_custom_entry_types,
            send_fn=send_with_interrupt,
            progress_file=_resolved_glossary_progress_file(progress_context),
            output_path=args.output,
            atomic_replace_fn=_atomic_replace_file,
        )
        if len(glossary) != before_refinement_count or _glossary_refinement_enabled():
            save_progress(completed, glossary, merged_indices, failed=failed, in_progress=in_progress, context=progress_context)
            save_glossary_json(glossary, args.output)
            save_glossary_csv(glossary, args.output)
    
    print(f"\nDone. Glossary saved to {args.output}")
    
    # Also save as CSV format for compatibility
    try:
        csv_output = args.output.replace('.json', '.csv')
        csv_path = os.path.join(glossary_dir, os.path.basename(csv_output))
        save_glossary_csv(glossary, args.output)
        print(f"Also saved as CSV: {csv_path}")
    except Exception as e:
        print(f"[Warning] Could not save CSV format: {e}")

def save_progress(completed: List[int], glossary: List[Dict], merged_indices: List[int] = None, failed: List[int] = None, in_progress: List[int] = None, context=None, model_update_indices=None, model_updates=None, key_updates=None, extracted_entries_updates=None):
    """Save progress to JSON file (history is now managed separately)
    
    NOTE: We no longer save the glossary itself in the progress file to avoid
    overwriting manual edits. The progress file only tracks which chapters are completed/failed.
    The actual glossary data is saved separately in the output JSON/CSV files.
    """
    global _progress_lock
    
    # Ensure book title entry is present in-memory before recording status
    glossary = _ensure_book_title_entry(glossary, context=context)

    # Refresh book-title status from current glossary snapshot
    def _refresh_book_title_flags():
        global BOOK_TITLE_PRESENT, BOOK_TITLE_VALUE
        for entry in glossary or []:
            if str(entry.get("type", "")).lower() == "book":
                if isinstance(context, GlossaryProgressContext):
                    context.book_title_present = True
                    context.book_title_value = entry.get("translated_name") or entry.get("raw_name")
                else:
                    BOOK_TITLE_PRESENT = True
                    BOOK_TITLE_VALUE = entry.get("translated_name") or entry.get("raw_name")
                return
        if isinstance(context, GlossaryProgressContext):
            context.book_title_present = False
            context.book_title_value = None
        else:
            BOOK_TITLE_PRESENT = False
            BOOK_TITLE_VALUE = None

    _refresh_book_title_flags()
    progress_file = _resolved_glossary_progress_file(context)
    _progress_file, output_file, positions, numbers, filenames, total_chapters = _progress_context_values(context)

    # Acquire local and cross-process locks to prevent concurrent writers from
    # replacing each other's chapter/refinement progress.
    with _progress_lock, _locked_glossary_progress_file(progress_file):
        completed_clean = _unique_int_list(completed)
        failed_clean = _unique_int_list(failed or [])
        requested_failed_set = set(failed_clean)
        merged_clean = _unique_int_list(merged_indices or [])
        failed_set = set(failed_clean)
        merged_set = set(merged_clean)
        requested_in_progress_set = set(_unique_int_list(in_progress)) if in_progress is not None else set()
        requested_output_stems = {}
        for req_idx in sorted(set(completed_clean) | set(failed_clean) | set(merged_clean) | requested_in_progress_set):
            req_file = _glossary_chapter_output_file(req_idx, context=context)
            req_stem = os.path.splitext(os.path.basename(str(req_file or "").lower()))[0]
            if req_stem:
                requested_output_stems[int(req_idx)] = req_stem

        # Failed chapters must not also be persisted as completed. The glossary
        # extractor may keep partial entries, but the chapter itself still needs
        # a retry and should render as failed in the progress dialog immediately.
        completed_clean = [idx for idx in completed_clean if idx not in failed_set]

        if completed is not None:
            completed[:] = completed_clean
        if failed is not None:
            failed[:] = failed_clean
        if merged_indices is not None:
            merged_indices[:] = merged_clean

        existing_chapters_by_idx = {}
        existing_refinement = {}
        existing_extracted_entries = {}
        preserved_in_progress = []
        externally_failed = []
        manual_removed_indices = []
        existing_progress = {}
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r', encoding='utf-8') as existing_f:
                    existing_progress = json.load(existing_f)
                if (
                    isinstance(existing_progress, dict)
                    and existing_progress.get("manual_removed_session_id") in (None, "", _GLOSSARY_PROGRESS_SESSION_ID)
                ):
                    manual_removed_indices = _unique_int_list(existing_progress.get("manual_removed_indices", []))
                if isinstance(existing_progress, dict) and isinstance(existing_progress.get("refinement"), dict):
                    existing_refinement = existing_progress.get("refinement", {})
                if isinstance(existing_progress, dict) and isinstance(existing_progress.get("chapter_extracted_entries"), dict):
                    existing_extracted_entries = {
                        str(k): v
                        for k, v in existing_progress.get("chapter_extracted_entries", {}).items()
                        if isinstance(v, list)
                    }
                existing_chapters = existing_progress.get("chapters", {}) if isinstance(existing_progress, dict) else {}
                if isinstance(existing_chapters, dict):
                    for existing_key, existing_info in existing_chapters.items():
                        if not isinstance(existing_info, dict):
                            continue
                        existing_idx = _glossary_progress_entry_index(existing_info, existing_key)
                        if existing_idx is not None:
                            existing_file = ""
                            for fname_key in ("output_file", "chapter_file", "original_basename", "filename", "source_filename"):
                                existing_file = _glossary_progress_filename(existing_info.get(fname_key, ""))
                                if existing_file:
                                    break
                            existing_stem = os.path.splitext(os.path.basename(str(existing_file or "").lower()))[0]
                            if any(req_idx != int(existing_idx) and req_stem == existing_stem for req_idx, req_stem in requested_output_stems.items()):
                                continue
                            existing_chapters_by_idx[int(existing_idx)] = existing_info
                            existing_status = str(existing_info.get("status", "")).lower()
                            if existing_status == "in_progress":
                                preserved_in_progress.append(int(existing_idx))
                            elif existing_status in ("failed", "qa_failed", "error"):
                                externally_failed.append(int(existing_idx))
                if isinstance(existing_progress, dict):
                    preserved_in_progress.extend(_unique_int_list(existing_progress.get("in_progress", [])))
        except Exception:
            existing_chapters_by_idx = {}

        manual_removed_set = set(manual_removed_indices)
        if manual_removed_set:
            if total_chapters:
                all_chapters_removed = len(manual_removed_set) >= int(total_chapters)
            else:
                all_chapters_removed = bool(completed_clean or failed_clean or merged_clean or requested_in_progress_set) and manual_removed_set.issuperset(
                    set(completed_clean) | set(failed_clean) | set(merged_clean) | requested_in_progress_set
                )
            if all_chapters_removed and (completed_clean or failed_clean or merged_clean or requested_in_progress_set):
                print("⚠️ Ignoring stale glossary progress manual_removed_indices that covered every chapter.")
                manual_removed_indices = []
                manual_removed_set = set()
        if manual_removed_indices:
            completed_clean = [idx for idx in completed_clean if idx not in manual_removed_set]
            failed_clean = [idx for idx in failed_clean if idx not in manual_removed_set]
            merged_clean = [idx for idx in merged_clean if idx not in manual_removed_set]
            failed_set = set(failed_clean)
            merged_set = set(merged_clean)
            if completed is not None:
                completed[:] = completed_clean
            if failed is not None:
                failed[:] = failed_clean
            if merged_indices is not None:
                merged_indices[:] = merged_clean

        if externally_failed:
            failed_clean = _unique_int_list(failed_clean + externally_failed)
            failed_set = set(failed_clean)
            merged_clean = [idx for idx in merged_clean if idx not in failed_set]
            completed_clean = [idx for idx in completed_clean if idx not in failed_set]
            merged_set = set(merged_clean)
            if completed is not None:
                completed[:] = completed_clean
            if failed is not None:
                failed[:] = failed_clean
            if merged_indices is not None:
                merged_indices[:] = merged_clean

        qa_issues_clean = {
            int(idx): issues
            for idx, issues in _normalize_glossary_qa_issues(_GLOSSARY_QA_ISSUES_FOUND).items()
            if int(idx) in failed_set
        }

        current_override_set = set(completed_clean) | set(merged_clean) | requested_in_progress_set
        if current_override_set:
            failed_clean = [
                idx for idx in failed_clean
                if idx not in current_override_set or idx in requested_failed_set
            ]
            failed_set = set(failed_clean)
            qa_issues_clean = {
                idx: issues for idx, issues in qa_issues_clean.items()
                if idx in failed_set
            }
            if failed is not None:
                failed[:] = failed_clean

        if in_progress is None:
            in_progress_clean = _unique_int_list(preserved_in_progress)
        else:
            in_progress_clean = _unique_int_list(in_progress)
        completed_set = set(completed_clean)
        done_set = completed_set | failed_set | merged_set
        in_progress_clean = [idx for idx in in_progress_clean if idx not in done_set]
        if manual_removed_indices:
            in_progress_clean = [idx for idx in in_progress_clean if idx not in manual_removed_set]

        hard_stop_restored_entries = {}
        hard_stop_failed_indices = set()
        if _glossary_is_hard_stop_env_active():
            hard_stop_indices = set(in_progress_clean)
            for existing_idx, existing_info in existing_chapters_by_idx.items():
                if isinstance(existing_info, dict) and str(existing_info.get("status", "")).lower() == "in_progress":
                    hard_stop_indices.add(existing_idx)
            hard_stop_indices -= manual_removed_set
            for stop_idx in sorted(hard_stop_indices):
                existing_info = existing_chapters_by_idx.get(stop_idx)
                restored = _glossary_restore_in_progress_entry(existing_info) if isinstance(existing_info, dict) else None
                if restored:
                    hard_stop_restored_entries[stop_idx] = restored
                else:
                    hard_stop_failed_indices.add(stop_idx)
            if hard_stop_failed_indices:
                failed_clean = _unique_int_list(failed_clean + sorted(hard_stop_failed_indices))
                failed_set = set(failed_clean)
                if failed is not None:
                    failed[:] = failed_clean
            if hard_stop_indices:
                completed_clean = [idx for idx in completed_clean if idx not in hard_stop_indices]
                merged_clean = [idx for idx in merged_clean if idx not in hard_stop_indices]
                in_progress_clean = [idx for idx in in_progress_clean if idx not in hard_stop_indices]
                if completed is not None:
                    completed[:] = completed_clean
                if merged_indices is not None:
                    merged_indices[:] = merged_clean
                completed_set = set(completed_clean)
                merged_set = set(merged_clean)
                done_set = completed_set | failed_set | merged_set

        in_progress_set = set(in_progress_clean)
        model_update_set = set()
        for _idx in model_update_indices or []:
            try:
                model_update_set.add(int(_idx))
            except (TypeError, ValueError):
                pass
        model_update_map = {}
        if isinstance(model_updates, dict):
            for _idx, _model in model_updates.items():
                try:
                    _idx = int(_idx)
                except (TypeError, ValueError):
                    continue
                _model = str(_model or "").strip()
                if _model:
                    model_update_map[_idx] = _model
                    model_update_set.add(_idx)
        key_update_map = {}
        if isinstance(key_updates, dict):
            for _idx, _key in key_updates.items():
                try:
                    _idx = int(_idx)
                except (TypeError, ValueError):
                    continue
                _key = str(_key or "").strip()
                if _key:
                    key_update_map[_idx] = _key
                    model_update_set.add(_idx)

        chapters = {}
        for idx in sorted(completed_set | failed_set | merged_set | in_progress_set):
            existing_info = existing_chapters_by_idx.get(int(idx), {})
            try:
                actual_num = int(existing_info.get("actual_num") or existing_info.get("chapter_num"))
            except (TypeError, ValueError):
                actual_num = _glossary_chapter_actual_num(idx, context=context)
            chapter_key = _glossary_chapter_key(idx)
            chapter_file = _glossary_chapter_output_file(idx, context=context)
            if not chapter_file and isinstance(existing_info, dict):
                for fname_key in ("output_file", "original_basename", "chapter_file", "source_filename", "filename"):
                    chapter_file = _glossary_progress_filename(existing_info.get(fname_key, ""))
                    if chapter_file:
                        break
            issue_list = qa_issues_clean.get(idx, [])
            if idx in failed_set:
                status = "qa_failed" if issue_list else "failed"
            elif idx in merged_set:
                status = "merged"
            elif idx in in_progress_set:
                status = "in_progress"
            else:
                status = "completed"

            chapter_info = {
                "chapter_index": idx,
                "actual_num": actual_num,
                "chapter_num": actual_num,
                "status": status,
                "last_updated": time.time(),
            }
            model_name = model_update_map.get(idx) or _current_glossary_model_name(existing_info, prefer_thread=idx in model_update_set)
            if model_name:
                chapter_info["model_name"] = model_name
            key_identifier, key_pool = _current_glossary_key_context(existing_info, prefer_thread=idx in model_update_set)
            if key_update_map.get(idx):
                key_identifier = key_update_map[idx]
                key_pool = _glossary_key_pool_from_identifier(key_identifier) or key_pool
            if key_identifier:
                chapter_info["key_identifier"] = key_identifier
            if key_pool:
                chapter_info["key_pool"] = key_pool
            if chapter_file:
                # Match TransateKRtoEN.py's progress shape: every chapter gets
                # a stable filename anchor so OPF offsets do not shift rows.
                chapter_info["output_file"] = chapter_file
            if issue_list:
                chapter_info["qa_issues"] = True
                chapter_info["qa_timestamp"] = time.time()
                chapter_info["qa_issues_found"] = issue_list
            if status == "in_progress":
                previous_status = "not_completed"
                previous_entry = None
                if isinstance(existing_info, dict) and existing_info:
                    if str(existing_info.get("status", "")).lower() == "in_progress":
                        copied_previous = False
                        for key in ("previous_status", "previous_progress_entry", "previous_status_unknown"):
                            if key in existing_info:
                                chapter_info[key] = existing_info[key]
                                copied_previous = True
                        previous_entry = chapter_info.get("previous_progress_entry")
                        if "previous_status" not in chapter_info and isinstance(previous_entry, dict):
                            chapter_info["previous_status"] = str(previous_entry.get("status", "not_completed") or "not_completed")
                        if not copied_previous:
                            chapter_info["previous_status_unknown"] = True
                        previous_status = None
                    else:
                        previous_status = str(existing_info.get("status", "not_completed") or "not_completed")
                        previous_entry = {
                            k: v for k, v in existing_info.items()
                            if k not in ("previous_status", "previous_progress_entry", "previous_status_unknown")
                        }
                if previous_status is not None:
                    chapter_info["previous_status"] = previous_status
                if (
                    previous_status is not None
                    and isinstance(previous_entry, dict)
                    and previous_status.lower() not in ("not_completed", "not translated", "not_translated")
                ):
                    chapter_info["previous_progress_entry"] = previous_entry
            if isinstance(existing_info, dict) and isinstance(existing_info.get("ocr_progress"), dict):
                chapter_info["ocr_progress"] = existing_info["ocr_progress"]
            chapters[chapter_key] = chapter_info

        for idx, existing_info in existing_chapters_by_idx.items():
            if idx in manual_removed_set:
                continue
            if idx in done_set or idx in in_progress_set:
                continue
            if not isinstance(existing_info, dict):
                continue
            existing_status = str(existing_info.get("status", "")).lower()
            if existing_status == "in_progress":
                restored = hard_stop_restored_entries.get(idx) or _glossary_restore_in_progress_entry(existing_info)
                if restored:
                    model_name = _current_glossary_model_name(existing_info)
                    if model_name and not (restored.get("model_name") or restored.get("model")):
                        restored = dict(restored)
                        restored["model_name"] = model_name
                    key_identifier, key_pool = _current_glossary_key_context(existing_info)
                    if key_identifier and not restored.get("key_identifier"):
                        restored = dict(restored)
                        restored["key_identifier"] = key_identifier
                    if key_pool and not restored.get("key_pool"):
                        restored = dict(restored)
                        restored["key_pool"] = key_pool
                    chapters[_glossary_chapter_key(idx)] = restored
            elif existing_status in ("qa_failed", "failed", "error", "pending", "merged", "completed"):
                model_name = _current_glossary_model_name(existing_info)
                if model_name and not (existing_info.get("model_name") or existing_info.get("model")):
                    existing_info = dict(existing_info)
                    existing_info["model_name"] = model_name
                key_identifier, key_pool = _current_glossary_key_context(existing_info)
                if key_identifier and not existing_info.get("key_identifier"):
                    existing_info = dict(existing_info)
                    existing_info["key_identifier"] = key_identifier
                if key_pool and not existing_info.get("key_pool"):
                    existing_info = dict(existing_info)
                    existing_info["key_pool"] = key_pool
                chapters[_glossary_chapter_key(idx)] = existing_info

        progress_model_name = next((value for value in model_update_map.values() if value), "") or _current_glossary_model_name(existing_progress, prefer_thread=bool(model_update_set))
        progress_key_identifier, progress_key_pool = _current_glossary_key_context(existing_progress, prefer_thread=bool(model_update_set))
        progress_key_identifier = next((value for value in key_update_map.values() if value), "") or progress_key_identifier
        if progress_key_identifier:
            progress_key_pool = _glossary_key_pool_from_identifier(progress_key_identifier) or progress_key_pool
        progress_data = {
            "book_title_present": bool(context.book_title_present) if isinstance(context, GlossaryProgressContext) else bool(BOOK_TITLE_PRESENT),
            # Use value from entry if present, otherwise fallback to global translated title
            "book_title": (
                context.book_title_value if context.book_title_present else context.book_title_translated
            ) if isinstance(context, GlossaryProgressContext) else (BOOK_TITLE_VALUE if BOOK_TITLE_PRESENT else BOOK_TITLE_TRANSLATED),
            "chapters": chapters,
            "completed": completed_clean,
            "failed": failed_clean,
            "merged_indices": merged_clean,
            "chapter_positions": {str(k): v for k, v in sorted((positions or {}).items())},
            "chapter_numbers": {str(k): v for k, v in sorted((numbers or {}).items())},
            "chapter_filenames": {str(k): v for k, v in sorted((filenames or {}).items())},
            "chapter_count": total_chapters,
            "glossary_output_file": output_file,
            "progress_schema_version": "2.1",
            "indexing": "chapter_index_zero_based",
            "qa_issues_found": {str(idx): issues for idx, issues in sorted(qa_issues_clean.items())},
            "in_progress": in_progress_clean,
            "progress_session_id": _GLOSSARY_PROGRESS_SESSION_ID,
            # Glossary is saved separately to output files, not in progress
            # This prevents the progress file from overwriting manual edits
        }
        if progress_model_name:
            progress_data["model_name"] = progress_model_name
        if progress_key_identifier:
            progress_data["key_identifier"] = progress_key_identifier
        if progress_key_pool:
            progress_data["key_pool"] = progress_key_pool
        if existing_refinement:
            progress_data["refinement"] = existing_refinement
        entry_index = dict(existing_extracted_entries)
        if isinstance(extracted_entries_updates, dict):
            for idx, entries in extracted_entries_updates.items():
                try:
                    key = str(int(idx))
                except (TypeError, ValueError):
                    continue
                entry_index[key] = compact_extracted_entries(entries)
        if entry_index:
            progress_data["chapter_extracted_entries"] = entry_index
        if manual_removed_indices:
            progress_data["manual_removed_indices"] = manual_removed_indices
            progress_data["manual_removed_session_id"] = _GLOSSARY_PROGRESS_SESSION_ID
        
        try:
            # Use atomic write with proper temp file handling
            progress_dir = os.path.dirname(progress_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())  # Ensure data is written to disk
            
            # Atomic replace
            try:
                _atomic_replace_file(temp_path, progress_file)
            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise
            
        except Exception as e:
            print(f"[Warning] Failed to save progress: {e}")

if __name__=='__main__':
    from shutdown_utils import run_cli_main
    run_cli_main(main)
