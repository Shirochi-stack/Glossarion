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
from ebooklib import epub
from chapter_splitter import ChapterSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Dict, Tuple
from unified_api_client import UnifiedClient, UnifiedClientError

# Thread submission throttling (glossary batch) — mirrors translation behavior
_glossary_thread_submit_lock = threading.Lock()
_glossary_last_thread_submit = 0.0

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
    return "graceful stop active - not starting new api call" in s

def create_client_with_multi_key_support(api_key, model, output_dir, config):
    """Create a UnifiedClient with multi API key support if enabled.
    
    Priority order for key pool:
    1. Glossary-specific keys (USE_GLOSSARY_KEYS=1 + GLOSSARY_API_KEYS / config['glossary_keys'])
    2. Multi-API keys (use_multi_api_keys + config['multi_api_keys'])
    3. Single key mode (main GUI key only)
    """
    
    # ── Step 1: Determine which key pool to use ──────────────────────────
    use_glossary_keys = os.getenv('USE_GLOSSARY_KEYS', '0') == '1'
    
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

def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None, chapter_idx=None, chunk_idx=None, total_chunks=None, merged_chapters=None):
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
    elif chapter_idx is not None:
        try:
            chap_num = int(chapter_idx) + 1
        except Exception:
            chap_num = chapter_idx
        if chunk_idx and total_chunks:
            chapter_label = f"Chapter {chap_num} (chunk {chunk_idx}/{total_chunks})"
        else:
            chapter_label = f"Chapter {chap_num}"
    
    result_queue = queue.Queue()

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
        def api_call():
            try:
                # Apply chapter/chunk context in THIS thread so UnifiedClient's
                # thread-local metadata is visible to watchdog/payloads.
                try:
                    if hasattr(client, 'set_chapter_context'):
                        chap_val = (chapter_idx + 1) if isinstance(chapter_idx, int) else (
                            int(chapter_idx) + 1 if chapter_idx is not None and str(chapter_idx).isdigit() else chapter_idx
                        )
                        client.set_chapter_context(
                            chapter=chap_val if chapter_idx is not None else None,
                            chunk=chunk_idx,
                            total_chunks=total_chunks,
                            merged_chapters=merged_chapters,
                        )
                except Exception:
                    pass
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
                
                start_time = time.time()
                result = client.send(messages, temperature=temperature, max_tokens=max_tokens, context='glossary')
                elapsed = time.time() - start_time
                
                # Capture raw response object for thought signatures (if available)
                raw_obj = None
                if hasattr(client, 'get_last_response_object'):
                    resp_obj = client.get_last_response_object()
                    if resp_obj and hasattr(resp_obj, 'raw_content_object'):
                        raw_obj = resp_obj.raw_content_object
                        # if raw_obj:
                        #     print("🧠 Captured thought signature for glossary extraction")
                
                # Include raw_obj in the result tuple
                result_queue.put((result, elapsed, raw_obj))
            except Exception as e:
                result_queue.put(e)
        # Apply submission delay shared across glossary batch threads to space out API launches.
        # Priority: per-key api_call_delay from glossary keys > global SEND_INTERVAL_SECONDS
        try:
            thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", os.getenv("THREAD_SUBMISSION_DELAY", "0.1")))
        except Exception:
            thread_delay = 0.1

        # Check for per-key delay from glossary pool first
        _per_key_delay = None
        try:
            # 1. Check if the client already has a per-key delay set
            _per_key_delay = getattr(client, '_per_key_api_delay', None)
        except Exception:
            pass
        if _per_key_delay is None:
            # 2. Check glossary key pool entries for individual api_call_delay
            try:
                _gk_pool = getattr(UnifiedClient, '_glossary_key_pool', None)
                if _gk_pool and hasattr(_gk_pool, 'keys') and _gk_pool.keys:
                    for _gk in _gk_pool.keys:
                        _gk_d = getattr(_gk, 'api_call_delay', 0.0) or 0.0
                        if _gk_d > 0:
                            _per_key_delay = _gk_d
                            break
            except Exception:
                pass
        if _per_key_delay is None:
            # 3. Check in-memory glossary keys (raw dict list)
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

        if _per_key_delay is not None and _per_key_delay > 0:
            api_delay = float(_per_key_delay)
        else:
            try:
                api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
            except Exception:
                api_delay = 2.0

        enforce_delay = max(thread_delay, api_delay)

        if enforce_delay > 0:
            with _glossary_thread_submit_lock:
                now = time.time()
                elapsed_since_last = now - _glossary_last_thread_submit
                remaining = enforce_delay - elapsed_since_last
                if remaining > 0:
                    # Emit queued log with the actual remaining sleep time
                    if not stop_check_fn() and os.environ.get('GRACEFUL_STOP') != '1':
                        try:
                            thread_name = threading.current_thread().name
                        except Exception:
                            thread_name = "thread"
                        print(f"📤 [{thread_name}] Queued {chapter_label} — Sending API call in {remaining:.1f}s")
                    interruptible_sleep(remaining, stop_check_fn, interval=0.1)
                    _glossary_last_thread_submit = time.time()
                else:
                    _glossary_last_thread_submit = now

        api_thread = threading.Thread(target=api_call)
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
                        raise result
                    if isinstance(result, tuple):
                        # Check if we have the new format with response object
                        if len(result) == 3:
                            api_result, api_time, raw_obj = result
                        else:
                            # Old format without response object
                            api_result, api_time = result
                            raw_obj = None
                        
                        if chunk_timeout and api_time > chunk_timeout:
                            if hasattr(client, '_in_cleanup'):
                                client._in_cleanup = True
                            if hasattr(client, 'cancel_current_operation'):
                                client.cancel_current_operation()
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
                        if hasattr(client, 'cancel_current_operation'):
                            client.cancel_current_operation()
                        
                        # Don't wait for the thread to finish - just raise immediately
                        raise UnifiedClientError("Glossary extraction stopped by user")
                    
                    if timeout is not None:
                        elapsed += check_interval
                        if elapsed >= timeout:
                            if hasattr(client, '_in_cleanup'):
                                client._in_cleanup = True
                            if hasattr(client, 'cancel_current_operation'):
                                client.cancel_current_operation()
                            raise UnifiedClientError(f"API call timed out after {timeout} seconds") from None
        
        except UnifiedClientError as e:
            error_msg = str(e)
            
            # Treat cancelled errors (from client being closed) as timeout
            if "cancelled" in error_msg.lower() or "Gemini client not initialized" in error_msg or "timed out" in error_msg.lower():
                # Check stop flag before retrying
                if stop_check_fn():
                    # print("❌ Glossary extraction stopped by user during timeout retry")  # Redundant
                    raise
                
                if timeout_retry_count < max_timeout_retries:
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
            cf = 1.0
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
    # metadata.json next to the output
    meta_dir = os.path.abspath(os.path.dirname(output_path) or ".")
    epub_base = os.path.splitext(os.path.basename(epub_path or ""))[0] if epub_path else None
    candidates = []
    if epub_base:
        candidates.append(os.path.join(meta_dir, epub_base, "metadata.json"))

    for meta_path in candidates:
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


def _ensure_book_title_entry(glossary: List[Dict]) -> List[Dict]:
    """Insert a 'book' entry (raw + translated title) at the top if enabled and not present."""
    global BOOK_TITLE_PRESENT, BOOK_TITLE_VALUE, BOOK_TITLE_RAW, BOOK_TITLE_TRANSLATED
    
    include = os.getenv("GLOSSARY_INCLUDE_BOOK_TITLE", "1").lower() not in ("0", "false", "no")
    
    # Determine titles to use
    # Prefer specific raw/translated values if available
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
    BOOK_TITLE_PRESENT = True
    BOOK_TITLE_VALUE = trans_title or raw_title
    return glossary

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = value
    
    # When clearing the stop flag, also clear the multi-key environment variable
    if not value:
        os.environ['TRANSLATION_CANCELLED'] = '0'
        
        # Also clear UnifiedClient global flag
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
        except:
            pass
    else:
        # Propagate stop to UnifiedClient (for streaming cancellation)
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            elif hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = True
        except Exception:
            pass

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
                'term': {'enabled': True, 'has_gender': False}
            }
        return result
    except:
        # Default configuration
        return {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False}
        }

def save_glossary_json(glossary: List[Dict], output_path: str):
    """Save glossary in the new simple format with automatic sorting by type"""
    # Check if legacy JSON output is enabled (default disabled)
    if os.getenv('GLOSSARY_OUTPUT_LEGACY_JSON', '0') != '1':
        return
    glossary = _ensure_book_title_entry(glossary)

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

        glossary = _ensure_book_title_entry(glossary)
        custom_types = get_custom_entry_types()
        type_order = {'book': -1, 'character': 0, 'term': 1}
        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
        for i, t in enumerate(other_types):
            type_order[t] = i + 2
        
        sorted_glossary = sorted(glossary, key=lambda x: (
            type_order.get(x.get('type', 'term'), 999),
            x.get('raw_name', '').lower()
        ))
        
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
                        if type_config.get('has_gender', False):
                            row.append(entry.get('gender', ''))
                        for field in custom_fields:
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
                    column_headers = ['translated_name', 'raw_name']
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
                            line = f"* {translated_name} ({raw_name})"
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
                            if type_config.get('has_gender', False):
                                row.append(entry.get('gender', ''))
                            for field in custom_fields:
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
                        column_headers = ['translated_name', 'raw_name']
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
                                line = f"* {translated_name} ({raw_name})"
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
    special_keywords = [
        'title', 'toc', 'cover', 'index', 'copyright', 'preface', 'nav',
        'message', 'info', 'notice', 'colophon', 'dedication', 'epigraph',
        'foreword', 'acknowledgment', 'author', 'appendix', 'glossary',
        'bibliography'
    ]
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
                    # Match if name (with or without trailing digits) contains a special keyword
                    if any(kw in name_lower for kw in special_keywords):
                        # If no digits at all, it's clearly special (e.g. "cover", "notice")
                        # If has digits, still special if the base part matches a keyword (e.g. "notice01")
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

def load_progress() -> Dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate the structure
                if not isinstance(data, dict):
                    print(f"[Warning] Progress file has invalid structure, resetting...")
                    return {"completed": [], "glossary": []}
                # Ensure all required keys exist
                if "completed" not in data:
                    data["completed"] = []
                # Glossary field is deprecated but may exist in old progress files
                # We ignore it now since glossary is loaded from output file instead
                if "glossary" in data:
                    # Remove old glossary field to save space (will be ignored anyway)
                    del data["glossary"]
                if "merged_indices" not in data:
                    data["merged_indices"] = []  # Track which chapters were merged into others
                if "failed" not in data:
                    data["failed"] = []  # Track chapters that had errors (will be retried)
                
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
                backup_name = f"{PROGRESS_FILE}.corrupted.{int(time.time())}"
                shutil.copy2(PROGRESS_FILE, backup_name)
                print(f"   -> Corrupted file backed up to: {backup_name}")
            except:
                pass
            return {"completed": [], "glossary": [], "context_history": []}
        except Exception as e:
            print(f"[Warning] Error loading progress file: {e}")
            return {"completed": [], "glossary": [], "context_history": []}
    return {"completed": [], "glossary": [], "context_history": [], "merged_indices": []}

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
                
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[Debug] JSON parsing failed: {e}")
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

                entries.append(entry_map)
                continue

            # Legacy fallback (no header detected)
            parts = row
            if len(parts) >= 3:
                entry_type = parts[0].lower()

                # Check if type is enabled
                if not _is_entry_type_accepted(entry_type, enabled_types):
                    continue

                entry = {
                    'type': _normalize_entry_type(entry_type, enabled_types),
                    'raw_name': parts[1],
                    'translated_name': parts[2]
                }

                # Add gender if type supports it and it's provided
                type_config = custom_types.get(entry_type, {})
                if type_config.get('has_gender', False) and len(parts) > 3 and parts[3]:
                    entry['gender'] = parts[3]
                elif type_config.get('has_gender', False):
                    entry['gender'] = 'Unknown'

                # Add any custom fields
                custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json)
                    start_idx = 4  # Always 4, not conditional
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

    # Post-filter: if the user did NOT include ``description`` in their
    # custom fields list, strip any description value the AI may have
    # returned anyway. Pairs with the prompt-side description-rule
    # placeholder strips — together they guarantee description data is
    # absent both from what we ask for and from what we accept in
    # downstream CSV / token-efficient output. Uses the shared helper so
    # every return path (JSON + CSV) applies the same filter.
    return _strip_unwanted_description_keys(entries)

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

# Placeholders that occupy their own line in the prompt. When the description
# field isn't active they get stripped along with the entire line (including
# any leading ``- `` bullet) so we don't leave orphan bullets behind.
_DESCRIPTION_LINE_PLACEHOLDERS = (
    '{description_mandatory}',
    '{description_detailed}',
)
# Placeholders that appear inline within a larger sentence. When inactive
# they must be replaced with an empty string — NOT stripped by line —
# otherwise the enclosing sentence would be destroyed.
_DESCRIPTION_INLINE_PLACEHOLDERS = (
    '{description_in_language}',
    '{description_excluded_note}',
)


def _apply_description_rule_placeholders(prompt_text, custom_fields=None):
    """Replace the three description-rule placeholders.

    Line-owning placeholders (`{description_mandatory}`, `{description_detailed}`):
    removed with their entire line when description is inactive.

    Inline placeholder (`{description_in_language}`):
    replaced with "" when inactive (sentence stays intact).

    When description is active, each placeholder is replaced with its
    canonical text.
    """
    if not isinstance(prompt_text, str):
        return prompt_text
    all_placeholders = _DESCRIPTION_LINE_PLACEHOLDERS + _DESCRIPTION_INLINE_PLACEHOLDERS
    if not any(p in prompt_text for p in all_placeholders):
        return prompt_text
    active = _glossary_description_active(custom_fields)
    if active:
        result = prompt_text
        result = result.replace('{description_mandatory}', _DESCRIPTION_MANDATORY_TEXT)
        result = result.replace('{description_detailed}', _DESCRIPTION_DETAILED_TEXT)
        result = result.replace('{description_in_language}', _DESCRIPTION_IN_LANGUAGE_TEXT)
        result = result.replace('{description_excluded_note}', _DESCRIPTION_EXCLUDED_NOTE_TEXT)
        return result
    # Inactive path: strip line-owners by line, empty inline placeholders.
    import re as _re_dr
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
    
    if not custom_prompt:
        # If no custom prompt, create a default
        custom_prompt = """You are a novel glossary extraction assistant.

You must strictly return ONLY CSV format with columns separated by commas.
Columns and entry types in this exact order provided:

{fields}

For character entries, determine gender from context, leave empty if context is insufficient.
For non-character entries, leave gender empty.
{description_mandatory}
IMPORTANT: Use commas to separate columns. Wrap a field value in double quotes ONLY when the value itself contains a comma.

Critical Requirement: The translated name{description_in_language} column must be in {language}, While the raw name column must the same as the source language.
The translated_name column must be a direct translation or transliteration of the raw_name ONLY. Do NOT use role labels, descriptions, or invented names as translations.

For example:
character,이히리ᐐ 나애,Dihirit Ade,female,"The enigmatic guild leader of the Shadow Lotus who operates from the concealed backrooms of the capital, manipulating city politics through commerce and wielding dual daggers with lethal precision"
character,뢤사난,Kim Sang-hyu,male,"A master swordsman from the Northern Sect known for his icy demeanor and unparalleled skill with the Frost Blade technique which he uses to defend the border fortress"
term,간편헤,Gale Hardest,,"A legendary ancient artifact forged by the Wind God said to control the atmospheric currents, currently sought by the Empire's elite guard to quell the rebellion"

CRITICAL EXTRACTION RULES:
- Extract All {entries}
- Do NOT extract sentences, dialogue, actions, questions, or statements as glossary entries
- REJECT entries that contain verbs or end with punctuation (?, !, .)
- REJECT entries starting with: "Me", "How", "What", "Why", "I", "He", "She", "They", "That's", "So", "Therefore", "Still", "But"{description_excluded_note}
- Do NOT create entries for common pronouns (나, 저, 너, 그, 그녀, 우리, 私, 僕, 俺, я, etc.) — these are NOT character names. Do NOT translate pronouns as role labels like "Narrator", "Protagonist", "Main Character", or "MC"
- Do NOT output any entries that are rejected by the above rules; skip them entirely
- REJECT generic common nouns, unnamed extras, and bare titles/roles (e.g. "Woman", "Man", "Boy", "Girl", "Villager", "Guard", "Soldier", "Aunt", "Father", "Queen", "Prince", "King", "Princess", "Knight", "Servant", "Maid", 여자, 남자, 소녀, 소년, 아줌마, 아버지, 여왕, 왕자). These are NOT proper nouns and must be skipped.
- REJECT descriptive noun phrases and adjectives attached to generic nouns (e.g. "Blonde Elf Girl", "Orange-eyed Beastman", "White-bearded Merchant", "Fake Couple", "Bespectacled Student"). Only extract actual names or standardized titles.
- If unsure whether something is a proper noun/name, skip it
- {description_detailed}
- The translated_name MUST be a strict literal dictionary translation or transliteration of the raw_name ONLY. You are FORBIDDEN from injecting story context, roles, or extra adjectives (e.g., do NOT translate "女学生" as "Female Student Assassin" or "주인님" as "The Protagonist").
- You must include absolutely all characters found in the provided text in your glossary generation. Do not skip any character."""

    # Replace {entries} placeholder now that we have the enabled custom entry types
    custom_prompt = custom_prompt.replace('{entries}', entries_str)
    custom_prompt = custom_prompt.replace('{{entries}}', entries_str)

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
                        fields_spec.append('  "gender": "Male", "Female", or "Unknown" (required for characters)')
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
                    if 'character' in [t[0] for t in enabled_types]:
                        example = '  {"type": "character", "raw_name": "田中太郎", "translated_name": "Tanaka Taro", "gender": "Male"'
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


def skip_duplicate_entries(glossary, dry_run=False, output_dir=None):
    """
    Skip entries with duplicate raw names and translated names using 2-pass deduplication.
    
    Pass 1: Remove entries with similar raw names (fuzzy matching)
    Pass 2: Remove entries with identical translated names (exact matching)
    
    Args:
        glossary: List of entry dicts with 'raw_name', 'translated_name', etc.
        dry_run: If True, return (original_entries, dedup_report) without modifying the list.
        output_dir: If provided, write dedup_report.json to this directory.
    
    Returns:
        list: Deduplicated entries (when dry_run=False)
        tuple: (original_entries, dedup_report) when dry_run=True
    """
    # Try to use RapidFuzz for speed, fallback to difflib
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
    pass1_results = _skip_raw_name_duplicates(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log)
    pass1_removed = original_count - len(pass1_results)
    print(f"[Dedup] ✅ PASS 1 complete: {pass1_removed} duplicates removed ({len(pass1_results)} remaining)")
    
    # PASS 2: Translated name deduplication (if enabled)
    if dedupe_translations:
        print(f"[Dedup] 🔄 PASS 2: Translated name deduplication...")
        final_results = _skip_translated_name_duplicates(pass1_results, dedup_log)
    else:
        final_results = pass1_results
        print(f"[Dedup] ⏭️ PASS 2 skipped (translation deduplication disabled)")
    
    total_removed = original_count - len(final_results)
    print(f"[Dedup] ✨ Deduplication complete: {total_removed} total duplicates removed, {len(final_results)} unique entries kept")
    
    # Write structured dedup report if output_dir provided
    if output_dir and dedup_log:
        try:
            report_path = os.path.join(output_dir, "dedup_report.json")
            report = {
                "original_count": original_count,
                "final_count": len(final_results),
                "total_removed": total_removed,
                "pass1_removed": pass1_removed,
                "pass2_removed": total_removed - pass1_removed,
                "threshold": fuzzy_threshold,
                "algorithm_mode": algo_mode,
                "decisions": dedup_log,
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[Dedup] 📄 Dedup report written to {report_path}")
        except Exception as e:
            print(f"[Dedup] ⚠️ Could not write dedup report: {e}")
    
    if dry_run:
        return glossary, dedup_log
    
    return final_results


def _skip_raw_name_duplicates(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log=None):
    """Pass 1: Remove entries with similar raw names using optimized serial processing"""
    # Note: Parallel processing doesn't work well for deduplication because:
    # 1. Order matters - can't determine if A is duplicate of B until we've processed A
    # 2. The "seen" list changes as we process, making parallelization complex
    # 3. The serial version with RapidFuzz batch processing is already very fast
    
    # Use optimized serial version for all sizes
    return _skip_raw_name_duplicates_serial(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log)


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
                        
                        score = calculate_similarity_with_config(
                            cleaned_names[idx1], cleaned_names[idx2], config
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
                        
                        score = calculate_similarity_with_config(
                            cleaned_names[idx1], cleaned_names[idx2], config
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
    
    # Build deduplicated list with field count comparison
    deduplicated = []
    raw_name_to_idx = {}
    skipped_count = 0
    replaced_count = 0
    
    for idx, (entry, raw_name, cleaned_name) in enumerate(processed):
        if is_duplicate[idx]:
            original_idx = duplicate_of[idx]
            original_raw = processed[original_idx][1]
            
            # Check if we should replace the original with this one (more fields)
            existing_idx = raw_name_to_idx.get(original_raw)
            if existing_idx is not None:
                existing_entry = deduplicated[existing_idx]
                current_field_count = len([v for v in entry.values() if v and str(v).strip()])
                existing_field_count = len([v for v in existing_entry.values() if v and str(v).strip()])
                
                if current_field_count > existing_field_count:
                    deduplicated[existing_idx] = entry
                    raw_name_to_idx[raw_name] = existing_idx
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

    header_cols = ['translated_name', 'raw_name', 'gender', 'description']
    section_re = re.compile(r"^===\s*(.+?)\s*===\s*$")
    line_re = re.compile(r"^\*\s+(.*?)\s*(?:\((.*?)\))?\s*(?:\[(.*?)\])?\s*(?::\s*(.*))?$")
    entries: List[Dict] = []
    current_section = None

    # Map section names back to type using enabled custom types (with plural tolerance)
    custom_types = get_custom_entry_types()
    type_map = {}
    for t in custom_types.keys():
        type_map[t.lower()] = t
        if not t.lower().endswith('s'):
            type_map[f"{t.lower()}s"] = t

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Header line
        if line.lower().startswith("glossary columns:"):
            cols_text = line.split(":", 1)[1]
            header_cols = [c.strip() for c in cols_text.split(",") if c.strip()]
            if not header_cols:
                header_cols = ['translated_name', 'raw_name', 'gender', 'description']
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
        m = line_re.match(line)
        if not m:
            continue
        translated, raw_name, gender_field, desc = m.groups()
        translated = (translated or "").strip()
        raw_name = (raw_name or "").strip()
        desc = (desc or "").strip()
        gender = (gender_field or "").strip()

        entry = {
            "type": type_map.get((current_section or "").lower(), type_map.get("terms", "terms")),
            "raw_name": raw_name,
            "translated_name": translated,
        }
        if gender:
            entry["gender"] = gender
        if desc:
            entry["description"] = desc

        # Extra columns beyond the standard four
        extra_cols = header_cols[4:] if len(header_cols) > 4 else []
        if extra_cols and desc and " | " in desc:
            desc_main, *extra_parts = desc.split(" | ")
            entry["description"] = desc_main
            for part in extra_parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in extra_cols:
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
        
        candidates.append((other_cleaned, other_raw))
    
    if not candidates:
        return (entry, raw_name, cleaned_name, False, 0.0, None)
    
    is_dup, best_score, best_match = _find_best_duplicate_match(
        cleaned_name, candidates, fuzzy_threshold, use_rapidfuzz
    )
    
    return (entry, raw_name, cleaned_name, is_dup, best_score, best_match)


def _find_best_duplicate_match(cleaned_name, seen_raw_names, fuzzy_threshold, use_rapidfuzz):
    """Find the best duplicate match using multi-algorithm fuzzy matching"""
    if not seen_raw_names:
        return (False, 0.0, None)
    
    name_lower = cleaned_name.lower()
    
    # Use advanced multi-algorithm detection if configured
    use_advanced = os.getenv('GLOSSARY_USE_ADVANCED_DETECTION', '1') == '1'
    
    if use_advanced:
        try:
            from duplicate_detection_config import calculate_similarity_with_config, get_duplicate_detection_config
            config = get_duplicate_detection_config()
            
            best_score = 0.0
            best_match = None
            
            for seen_clean, seen_original in seen_raw_names:
                # Use multi-algorithm similarity scoring
                score = calculate_similarity_with_config(cleaned_name, seen_clean, config)
                
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
            # Extract just the cleaned names for batch comparison
            candidate_names = [seen_clean.lower() for seen_clean, _ in seen_raw_names]
            
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
        
        for seen_clean, seen_original in seen_raw_names:
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
        
        for seen_clean, seen_original in seen_raw_names:
            score = difflib.SequenceMatcher(None, name_lower, seen_clean.lower()).ratio()
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_match = seen_original
        
        return (best_score >= fuzzy_threshold, best_score, best_match)


def _skip_raw_name_duplicates_serial(glossary, fuzzy_threshold, use_rapidfuzz, dedup_log=None):
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
    raw_name_to_idx = {}  # raw_name -> index in deduplicated (O(1) lookup)
    # Reverse map: for a given index in deduplicated, which index in seen_raw_names?
    dedup_idx_to_seen_idx = {}
    deduplicated = []
    skipped_count = 0
    
    for entry in glossary:
        # Get raw_name and clean it
        raw_name = entry.get('raw_name', '')
        if not raw_name:
            continue
            
        # Remove honorifics + NFC normalize for comparison (unless disabled)
        cleaned_name = unicodedata.normalize('NFC', remove_honorifics(raw_name))
        
        # Check for fuzzy matches with seen names
        is_duplicate, best_score, best_match = _find_best_duplicate_match(
            cleaned_name, seen_raw_names, fuzzy_threshold, use_rapidfuzz
        )
        
        if is_duplicate:
            # Use O(1) dictionary lookup
            existing_index = raw_name_to_idx.get(best_match)
            
            if existing_index is not None:
                existing_entry = deduplicated[existing_index]
                # Count non-empty fields (excluding internal keys starting with _)
                current_field_count = len([v for v in entry.values() if v and str(v).strip()])
                existing_field_count = len([v for v in existing_entry.values() if v and str(v).strip()])
                
                # If current entry has more fields, replace the existing one
                if current_field_count > existing_field_count:
                    # Replace existing entry in deduplicated list
                    deduplicated[existing_index] = entry
                    # Update raw_name_to_idx: add new key, remove old key
                    raw_name_to_idx[raw_name] = existing_index
                    del raw_name_to_idx[best_match]
                    # FIX: Update seen_raw_names at the correct index so future
                    # fuzzy comparisons use the new entry's cleaned name / raw name.
                    seen_idx = dedup_idx_to_seen_idx.get(existing_index)
                    if seen_idx is not None:
                        seen_raw_names[seen_idx] = (cleaned_name, raw_name)
                    skipped_count += 1
                    if skipped_count <= 10:
                        print(f"[Skip] Pass 1: Replacing {best_match} ({existing_field_count} fields) with {raw_name} ({current_field_count} fields) - {best_score*100:.1f}% match, more detailed entry")
                    if dedup_log is not None:
                        dedup_log.append({
                            "pass": 1, "action": "replaced",
                            "kept": raw_name, "dropped": best_match,
                            "score": round(best_score, 4),
                            "reason": f"richer entry ({current_field_count} vs {existing_field_count} fields)"
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
            seen_raw_names.append((cleaned_name, raw_name))
            dedup_idx = len(deduplicated)
            raw_name_to_idx[raw_name] = dedup_idx
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
            
            # Count fields in both entries (more fields = higher priority)
            current_field_count = len([v for v in entry.values() if v and str(v).strip()])
            existing_field_count = len([v for v in existing_entry.values() if v and str(v).strip()])
            
            # If current entry has more fields, replace the existing one
            if current_field_count > existing_field_count:
                # Replace in-place using the stored index (faster than list comprehension)
                deduplicated[existing_idx] = entry
                # Update tracking with new index
                seen_translations[translated_lower] = (raw_name, entry, existing_idx)
                replaced_count += 1
                skipped_count += 1
                if skipped_count <= 10:
                    print(f"[Skip] Pass 2: Replacing '{existing_raw}' -> '{existing_translated}' ({existing_field_count} fields) with '{raw_name}' -> '{translated_name}' ({current_field_count} fields)")
                if dedup_log is not None:
                    dedup_log.append({
                        "pass": 2, "action": "replaced",
                        "kept": raw_name, "dropped": existing_raw,
                        "translation": translated_name,
                        "reason": f"richer entry ({current_field_count} vs {existing_field_count} fields)"
                    })
            else:
                # Keep existing entry (has same or more fields)
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
                        f"💬 Batch Chapter {idx+1} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant:,}, "
                        f"assistant/memory: {assistant_tokens:,}) / {GLOSSARY_LIMIT_STR}"
                    )
                else:
                    print(
                        f"💬 Batch Chapter {idx+1} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) / {GLOSSARY_LIMIT_STR}"
                    )
            except Exception:
                # Never let logging break batch processing
                pass

            # Submit to thread pool
            future = executor.submit(
                process_single_chapter_api_call,
                idx, chap, msgs, client, temp, mtoks, check_stop, chunk_timeout
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
                            print(f"⚠️ Failed to save batch history for chapter {idx+1}: {e}")
                
                results.append(result)
            except Exception as e:
                if "stopped by user" in str(e).lower():
                    print(f"✅ Chapter {idx+1} stopped by user")
                else:
                    print(f"Error processing chapter {idx+1}: {e}")
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
                                  chunk_idx: int = None, total_chunks: int = None) -> Dict:
    """Process a single chapter API call with thread-safe payload handling"""
    
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
                    print(f"🧵 [{thread_name}] Applying thread delay: {sleep_time:.3f}s for Chapter {idx+1}")
                    
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
    print(f"[BATCH] Starting API call for Chapter {idx+1} at {time.strftime('%H:%M:%S')}")
    
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
                    'chapter': idx + 1,
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
            chunk_idx=chunk_idx,
            total_chunks=total_chunks
        )

        # Handle the response - it might be a tuple or a string
        if raw is None:
            print(f"⚠️ API returned None for chapter {idx+1}")
            return {
                'idx': idx,
                'data': [],
                'resp': "",
                'chap': chap,
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
        print(f"[BATCH] Chapter {idx+1} - Raw response length: {len(resp)} chars")
        print(f"[BATCH] Chapter {idx+1} - Parsed {len(data)} entries before validation")
        
        # Filter out invalid entries
        valid_data = []
        for entry in data:
            if validate_extracted_entry(entry):
                # Clean the raw_name
                if 'raw_name' in entry:
                    entry['raw_name'] = entry['raw_name'].strip()
                valid_data.append(entry)
            else:
                print(f"[BATCH] Chapter {idx+1} - Invalid entry: {entry}")
        
        elapsed = time.time() - start_time
        print(f"[BATCH] Completed Chapter {idx+1} in {elapsed:.1f}s at {time.strftime('%H:%M:%S')} - Extracted {len(valid_data)} valid entries")
        
        return {
            'idx': idx,
            'data': valid_data,
            'resp': resp,
            'chap': chap,  # Include the chapter text in the result
            'raw_obj': raw_obj,  # Include raw object for history (from send_with_interrupt)
            'finish_reason': finish_reason,  # Track truncation ('length'/'MAX_TOKENS' = truncated)
            'error': None
        }
            
    except UnifiedClientError as e:
        # Graceful-stop cancellations are expected when queued calls are prevented from starting.
        # Keep a concise log so it's clear why extraction stopped/skipped without spamming the full error.
        if _is_graceful_stop_skip_error(e):
            if chunk_idx and total_chunks and int(total_chunks) > 1:
                print(f"⏭️ Chapter {idx+1} chunk {chunk_idx}/{total_chunks} skipped (graceful stop)")
            else:
                print(f"⏭️ Chapter {idx+1} skipped (graceful stop)")
        else:
            print(f"[Error] API call interrupted/failed for chapter {idx+1}: {e}")

        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
            'error': str(e),
            'graceful_stop_skip': _is_graceful_stop_skip_error(e),
        }
    except Exception as e:
        print(f"[Error] Unexpected error for chapter {idx+1}: {e}")
        import traceback
        print(f"[Error] Traceback: {traceback.format_exc()}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
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
                                      chunk_timeout: int = None):
    """
    Wrapper that performs chapter-level splitting (using output-limit budget) before calling the API.
    Aggregates all chunk results into a single result dict to keep batch accounting identical.
    """
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
        return process_single_chapter_api_call(idx, chap, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout)

    print(f"⚠️ Chapter {idx+1} exceeds chunk budget ({chapter_tokens:,} > {available_tokens:,}); splitting...")
    # Wrap plain text as simple HTML for splitter
    chapter_html = f"<html><body><p>{chap.replace(chr(10)+chr(10), '</p><p>')}</p></body></html>"
    chunks = chapter_splitter.split_chapter(chapter_html, available_tokens)
    print(f"📄 Chapter split into {len(chunks)} chunks (budget {available_tokens:,})")

    aggregated_data = []
    last_resp = ""
    last_raw_obj = None
    any_chunk_truncated = False
    for chunk_html, chunk_idx, total_chunks in chunks:
        if stop_check_fn():
            print(f"❌ Glossary extraction stopped during chunk {chunk_idx}/{total_chunks} of chapter {idx+1}")
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

        print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {idx+1}")
        # Sanitize before delegating (guarantees user + no raw blobs in payload)
        msgs = _sanitize_messages_for_api(msgs, chunk_text)
        result = process_single_chapter_api_call(
            idx, chunk_text, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout,
            chunk_idx=chunk_idx, total_chunks=total_chunks
        )
        if result.get("data"):
            aggregated_data.extend(result["data"])
        last_resp = result.get("resp", last_resp)
        if result.get("raw_obj"):
            last_raw_obj = result.get("raw_obj")
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
        'error': None
    }

def process_merged_group_api_call(merge_group: list, msgs_builder_fn, 
                                   client, temp: float, mtoks: int,
                                   stop_check_fn, chunk_timeout: int = None) -> Dict:
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
        result = process_single_chapter_api_call(idx, chap, msgs, client, temp, mtoks, stop_check_fn, chunk_timeout)
        return {'results': [result], 'merged_indices': []}
    
    # Merge chapter contents WITHOUT separators (glossary extraction doesn't need them)
    parent_idx = merge_group[0][0]
    merged_parts = []
    chapter_nums = []
    
    for idx, chap in merge_group:
        chapter_num = idx + 1  # Fallback for module-level function
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
            merged_chapters=chapter_nums,
        )
        
        # Extract response text
        resp = ""
        if raw is None:
            print(f"⚠️ API returned None for merged group")
            return {
                'results': [{'idx': idx, 'data': [], 'resp': '', 'chap': chap, 'error': 'API returned None'}
                           for idx, chap in merge_group],
                'merged_indices': [idx for idx, _ in merge_group[1:]]
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
                'merged_indices': [idx for idx, _ in merge_group[1:]]
            }
    except Exception as e:
        print(f"❌ Merged group failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        
        return {
            'results': [{'idx': idx, 'data': [], 'resp': '', 'chap': chap, 'error': str(e)}
                       for idx, chap in merge_group],
            'merged_indices': [idx for idx, _ in merge_group[1:]]
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



# Update main function to support batch processing:
def main(log_callback=None, stop_callback=None):
    # Declare global variables at the very start of the function
    global _skipped_chapters
    
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
        if os.environ.get('GRACEFUL_STOP') == '1':
            return False
        
        if stop_callback and stop_callback():
            # print("❌ Glossary extraction stopped by user request.")  # Redundant - logged elsewhere
            return True
        return is_stop_requested()
        
    start = time.time()
    
    # Handle both command line and GUI calls
    if '--epub' in sys.argv:
        # Command line mode
        parser = argparse.ArgumentParser(description='Extract glossary from EPUB/TXT')
        parser.add_argument('--epub', required=True, help='Path to EPUB/TXT file')
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
    
    if is_text_file:
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

    if _chapter_filenames and not is_text_file and not is_pdf_file:
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

                    # Build offset positions, skip special files (same as TransateKRtoEN)
                    def _is_special_spine(fname):
                        fl = fname.lower()
                        fnoext = os.path.splitext(fl)[0]
                        if any(kw in fl for kw in ['nav.', 'toc.', 'cover.']):
                            return True
                        if not re.search(r'\d', fnoext):
                            return True
                        return False

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

    # ensure we have a Glossary subfolder next to the JSON/MD outputs
    base_out_dir = os.path.dirname(args.output)
    if os.path.basename(os.path.abspath(base_out_dir)).lower() == "glossary":
        glossary_dir = base_out_dir
    else:
        glossary_dir = os.path.join(base_out_dir, "Glossary")
    os.makedirs(glossary_dir, exist_ok=True)

    # override the module‐level PROGRESS_FILE to include epub name
    global PROGRESS_FILE
    PROGRESS_FILE = os.path.join(
        glossary_dir,
        f"{file_base}_glossary_progress.json"
    )

    config = load_config(args.config)
    
    # Log assistant prompt if configured
    _log_assistant_prompt_once()

    # Ensure truncation retry settings use the correct Other Settings keys
    # (only set if not already provided by the environment)
    if os.getenv("RETRY_TRUNCATED") is None:
        os.environ["RETRY_TRUNCATED"] = "1" if config.get("retry_truncated", True) else "0"
    if os.getenv("TRUNCATION_RETRY_ATTEMPTS") is None:
        os.environ["TRUNCATION_RETRY_ATTEMPTS"] = str(config.get("truncation_retry_attempts", 1))
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
    
    # Check progress file for saved book title to avoid re-translation
    if not BOOK_TITLE_TRANSLATED and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
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
                        if os.path.exists(PROGRESS_FILE):
                            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                                p_data = json.load(f)
                        p_data['book_title'] = translated
                        
                        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
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
    chapter_split_enabled = os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "1") == "1"
    if chapter_split_enabled or os.getenv("DEBUG_CHAPTER_SPLIT_LOG", "0") == "1":
        print(f"✂️  Chapter Split Enabled: {'✅' if chapter_split_enabled else '❌'}")

    # Resolve effective output token limit (honor -1 as inherit)
    raw_output_env = os.getenv("GLOSSARY_MAX_OUTPUT_TOKENS", os.getenv("MAX_OUTPUT_TOKENS", "0"))
    effective_output_tokens = None
    try:
        raw_val = int(str(raw_output_env).strip())
        if raw_val > 0:
            effective_output_tokens = raw_val
    except Exception:
        effective_output_tokens = None

    if effective_output_tokens is None or effective_output_tokens <= 0:
        try:
            fallback_val = int(os.getenv("MAX_OUTPUT_TOKENS", str(config.get('max_tokens', 65536))))
            effective_output_tokens = fallback_val if fallback_val > 0 else 65536
        except Exception:
            effective_output_tokens = 65536

    # Honor discovered per-model limits from UnifiedClient (if available)
    try:
        with UnifiedClient._model_limits_lock:
            cached_limit = getattr(UnifiedClient, "_model_token_limits", {}).get(model)
        if cached_limit and cached_limit > 0:
            effective_output_tokens = min(effective_output_tokens, cached_limit)
    except Exception:
        pass

    # Budget for chunking, matching TransateKRtoEN safe limit logic
    available_tokens = _compute_safe_input_tokens(effective_output_tokens, compression_factor)
    print(f"📊 Chunk budget: {available_tokens:,} tokens (output limit {effective_output_tokens:,}, margin 500, compression {compression_factor})")

    # Initialize chapter splitter with compression factor
    chapter_splitter = ChapterSplitter(model_name=model, compression_factor=compression_factor)

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
    if chapter_range and re.match(r"^\d+\s*-\s*\d+$", chapter_range):
        range_start, range_end = map(int, chapter_range.split("-", 1))
        if use_spine_order:
            print(f"📊 Chapter Range Filter (SPINE ORDER): positions {range_start} to {range_end}")
        else:
            print(f"📊 Chapter Range Filter: {range_start} to {range_end}")
    elif chapter_range:
        print(f"⚠️ Invalid chapter range format: {chapter_range} (use format: 5-10)")

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

    # Log glossary anti-duplicate parameters usage (matches GlossaryManager)
    if os.getenv("GLOSSARY_ENABLE_ANTI_DUPLICATE", "0") == "1":
        ad_top_p = os.getenv("GLOSSARY_TOP_P", "1.0")
        ad_top_k = os.getenv("GLOSSARY_TOP_K", "0")
        ad_freq = os.getenv("GLOSSARY_FREQUENCY_PENALTY", "0.0")
        ad_pres = os.getenv("GLOSSARY_PRESENCE_PENALTY", "0.0")
        ad_rep = os.getenv("GLOSSARY_REPETITION_PENALTY", "1.0")
        print(f"🎯 Anti-duplicate enabled for glossary (top_p={ad_top_p}, top_k={ad_top_k}, freq_penalty={ad_freq}, presence_penalty={ad_pres}, repetition_penalty={ad_rep})")
    
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

    if is_text_file:
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
    if _chapter_filenames and not is_text_file and not is_pdf_file:
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
                    def _is_special2(fname):
                        fl = fname.lower()
                        fnoext = os.path.splitext(fl)[0]
                        if any(kw in fl for kw in ['nav.', 'toc.', 'cover.']):
                            return True
                        if not re.search(r'\d', fnoext):
                            return True
                        return False
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

    if not chapters:
        print("No chapters found. Exiting.")
        return

    # Check for stop before starting processing
    if check_stop():
        return

    prog = load_progress()
    completed = prog['completed']
    failed = prog.get('failed', [])
    # Remove failed chapters from completed so they get retried
    if failed:
        before = len(completed)
        completed[:] = [idx for idx in completed if idx not in failed]
        if before != len(completed):
            print(f"🔄 {len(failed)} previously failed chapter(s) will be retried: {[i+1 for i in sorted(failed)]}")
        failed.clear()  # Reset failed list for this run
    # Load existing glossary from output file (if it exists) instead of progress file
    # This preserves manual edits to the glossary
    output_glossary_path = os.path.join(glossary_dir, os.path.basename(args.output))
    if os.path.exists(output_glossary_path):
        try:
            with open(output_glossary_path, 'r', encoding='utf-8') as f:
                glossary = json.load(f)
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
    
    # Request merging configuration (glossary-specific with fallback to global)
    request_merging_enabled = os.getenv('GLOSSARY_REQUEST_MERGING_ENABLED', os.getenv('REQUEST_MERGING_ENABLED', '0')) == '1'
    request_merge_count = int(os.getenv('GLOSSARY_REQUEST_MERGE_COUNT', os.getenv('REQUEST_MERGE_COUNT', '3')))
    
    if request_merging_enabled and request_merge_count > 1:
        print(f"\n🔗 REQUEST MERGING ENABLED: Combining up to {request_merge_count} chapters per request")
    
    # Get both settings
    contextual_enabled = os.getenv('CONTEXTUAL', '1') == '1'
    rolling_window = os.getenv('GLOSSARY_HISTORY_ROLLING', '0') == '1'
    
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
                merge_groups = []
                for i in range(0, len(chapters_to_process), request_merge_count):
                    merge_groups.append(chapters_to_process[i:i + request_merge_count])
                print(f"🔗 Created {len(merge_groups)} merge groups from {len(chapters_to_process)} chapters (count-based)")
                units_to_process = merge_groups
                is_merged_mode = True
        else:
            units_to_process = [[ch] for ch in chapters_to_process]  # Each chapter as single-item group
            is_merged_mode = False
        
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
                    save_progress(completed, glossary, merged_indices, failed=failed)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
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
                    
                    save_progress(completed, glossary, merged_indices, failed=failed)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
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
                    
                    save_progress(completed, glossary, merged_indices, failed=failed)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    
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
                print(f"\n🔄 Processing Batch {batch_num+1}/{total_batches} (Chapters: {[idx+1 for idx, _ in current_batch]})")
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
                    if is_merged_mode:
                        future = executor.submit(
                            process_merged_group_api_call,
                            unit, build_prompt, client, temp, mtoks, check_stop, chunk_timeout
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
                            chunk_timeout
                        )
                    futures[future] = unit
                    # Small yield to keep GUI responsive
                    time.sleep(0.001)
                    return future

                def _handle_future_result(future, unit):
                    nonlocal batch_entry_count, stopped_early
                    try:
                        if is_merged_mode:
                            # Handle merged group result
                            group_result = future.result(timeout=0.5)
                            results = group_result.get('results', [])
                            new_merged_indices = group_result.get('merged_indices', [])
                            
                            # Add new merged indices to tracking
                            for mi in new_merged_indices:
                                if mi not in merged_indices:
                                    merged_indices.append(mi)
                            
                            for result in results:
                                idx = result.get('idx')
                                data = result.get('data', [])
                                resp = result.get('resp', '')
                                error = result.get('error')
                                raw_obj = result.get('raw_obj')
                                chap = result.get('chap')
                                
                                if error:
                                    # Suppress expected "graceful stop" pre-send cancellations.
                                    if isinstance(error, str) and _is_graceful_stop_skip_error(error):
                                        stopped_early = True
                                        return
                                    print(f"[Chapter {idx+1}] Error: {error}")
                                    if idx not in failed:
                                        failed.append(idx)
                                    return
                                
                                # Process entries
                                if data and len(data) > 0:
                                    total_ent = len(data)
                                    batch_entry_count += total_ent
                                    
                                    for eidx, entry in enumerate(data, start=1):
                                        elapsed = time.time() - start
                                        entry_type = entry.get("type", "?")
                                        raw_name = entry.get("raw_name", "?")
                                        trans_name = entry.get("translated_name", "?")
                                        print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed) → {entry_type}: {raw_name} ({trans_name})')
                                        glossary.append(entry)
                                
                                # Check if this was actually a failure (empty/refused content)
                                # BUT skip this check for merged children — they intentionally have
                                # empty data/resp because their content was processed via the parent chapter
                                if 'merged_into' in result:
                                    # Merged child: content was handled by parent, just mark completed
                                    completed.append(idx)
                                else:
                                    _resp_text = resp or ''
                                    _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                                    
                                    if _is_empty_failure:
                                        print(f"⚠️ Chapter {idx+1} returned empty/refused content — marking as failed for retry")
                                        if idx not in failed:
                                            failed.append(idx)
                                    else:
                                        completed.append(idx)
                                        
                                        # Mark truncated chapters as failed so they get retried
                                        ch_finish = result.get('finish_reason', 'stop')
                                        if ch_finish in ('length', 'MAX_TOKENS', 'max_tokens'):
                                            print(f"⚠️ Chapter {idx+1} was truncated — entries kept but chapter will be retried")
                                            if idx not in failed:
                                                failed.append(idx)
                                
                                # Store history for parent chapter only
                                if contextual_enabled and resp and chap and 'merged_into' not in result:
                                    system_prompt, user_prompt = build_prompt(chap)
                                    batch_history_map[idx] = (user_prompt, resp, raw_obj)
                            
                            print(f"✅ Merged group done: {len(results)} chapters")
                        else:
                            # Handle single chapter result
                            idx, chap = unit[0]
                            result = future.result(timeout=0.5)
                            
                            # Process this chapter's results immediately
                            data = result.get('data', [])
                            resp = result.get('resp', '')
                            error = result.get('error')
                            raw_obj = result.get('raw_obj')
                            
                            if error:
                                # Suppress expected "graceful stop" pre-send cancellations.
                                if (isinstance(error, str) and _is_graceful_stop_skip_error(error)) or result.get('graceful_stop_skip'):
                                    stopped_early = True
                                    return
                                print(f"[Chapter {idx+1}] Error: {error}")
                                if idx not in failed:
                                    failed.append(idx)
                                return
                            
                            # Process entries as each chapter completes
                            if data and len(data) > 0:
                                total_ent = len(data)
                                batch_entry_count += total_ent
                                
                                for eidx, entry in enumerate(data, start=1):
                                    elapsed = time.time() - start
                                    
                                    # Get entry info
                                    entry_type = entry.get("type", "?")
                                    raw_name = entry.get("raw_name", "?")
                                    trans_name = entry.get("translated_name", "?")
                                    
                                    print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed) → {entry_type}: {raw_name} ({trans_name})')
                                    
                                    # Add entry immediately WITHOUT deduplication
                                    glossary.append(entry)
                            
                            # Check if this was actually a failure (empty/refused content)
                            _resp_text = resp or ''
                            _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                            
                            if _is_empty_failure:
                                print(f"⚠️ Chapter {idx+1} returned empty/refused content — marking as failed for retry")
                                if idx not in failed:
                                    failed.append(idx)
                            else:
                                completed.append(idx)
                                
                                # Mark truncated chapters as failed so they get retried
                                ch_finish = result.get('finish_reason', 'stop')
                                if ch_finish in ('length', 'MAX_TOKENS', 'max_tokens'):
                                    print(f"⚠️ Chapter {idx+1} was truncated — entries kept but chapter will be retried")
                                    if idx not in failed:
                                        failed.append(idx)
                            
                            # Store history entry for this chapter (will be added after batch completes)
                            if contextual_enabled and resp and chap:
                                system_prompt, user_prompt = build_prompt(chap)
                                batch_history_map[idx] = (user_prompt, resp, raw_obj)
                        
                        # Save progress after each chapter completes (crash-safe with atomic writes)
                        save_progress(completed, glossary, merged_indices, failed=failed)
                        # Also save glossary files for incremental updates
                        save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        
                    except Exception as e:
                        # Suppress expected "graceful stop" pre-send cancellations.
                        if _is_graceful_stop_skip_error(e):
                            stopped_early = True
                            return
                        if is_merged_mode:
                            # For merged mode, mark all chapters in the unit as failed on error
                            for u_idx, u_chap in unit:
                                if "stopped by user" in str(e).lower():
                                    print(f"✅ Chapter {u_idx+1} stopped by user")
                                else:
                                    print(f"Error processing merged chapter {u_idx+1}: {e}")
                                if u_idx not in completed and u_idx not in failed:
                                    failed.append(u_idx)
                        else:
                            idx, chap = unit[0]
                            if "stopped by user" in str(e).lower():
                                print(f"✅ Chapter {idx+1} stopped by user")
                            else:
                                print(f"Error processing chapter {idx+1}: {e}")
                            if idx not in completed and idx not in failed:
                                failed.append(idx)

                if aggressive_mode:
                    # Aggressive mode: keep pool full, auto-refill as futures complete.
                    # Use wait(FIRST_COMPLETED) so newly-submitted futures are also observed promptly.
                    active_futures = {}
                    next_unit_idx = 0
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
                        if check_stop():
                            stopped_early = True
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

                            _handle_future_result(future, unit)
                            if stopped_early:
                                break

                            if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                                print("\u2705 Graceful stop: Chapter completed and saved, stopping...")
                                stopped_early = True
                                cancelled = cancel_all_futures(list(active_futures.keys()))
                                if cancelled > 0:
                                    print(f"\u2705 Cancelled {cancelled} pending API calls")
                                executor.shutdown(wait=False)
                                active_futures.clear()
                                break

                        if stopped_early:
                            break
                else:
                    # Submit all units in this batch
                    for unit in current_batch_units:
                        if check_stop():
                            stopped_early = True
                            break
                        _submit_unit(unit)

                    # Process results AS THEY COMPLETE, not all at once
                    for future in as_completed(futures):
                        if check_stop():
                            # print("🛑 Stop detected - cancelling all pending operations...")  # Redundant
                            stopped_early = True
                            cancelled = cancel_all_futures(list(futures.keys()))
                            if cancelled > 0:
                                print(f"✅ Cancelled {cancelled} pending API calls")
                            executor.shutdown(wait=False)
                            break

                        unit = futures[future]
                        _handle_future_result(future, unit)
                        
                        # Check for graceful stop AFTER processing result
                        if os.environ.get('GRACEFUL_STOP_COMPLETED') == '1':
                            print("✅ Graceful stop: Chapter completed and saved, stopping...")
                            stopped_early = True
                            cancelled = cancel_all_futures(list(futures.keys()))
                            if cancelled > 0:
                                print(f"✅ Cancelled {cancelled} pending API calls")
                            executor.shutdown(wait=False)
                            break
            
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
                            print(f"⚠️ Failed to append Chapter {idx+1} to glossary history: {e}")
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
                save_progress(completed, glossary, merged_indices, failed=failed)
                save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            
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
                    
                    save_progress(completed, glossary, merged_indices, failed=failed)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    
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
                        
                        save_progress(completed, glossary, merged_indices, failed=failed)
                        save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    
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
                for i in range(0, len(chapters_needing_processing), request_merge_count):
                    group = chapters_needing_processing[i:i + request_merge_count]
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
                print(f"❌ Glossary extraction stopped at chapter {idx+1}")
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
            _chap_num = _chapter_positions.get(idx, idx + 1)
            if _fname:
                print(f"🔄 Processing Chapter {_chap_num}/{total_chapters} ({_fname})")
            else:
                print(f"🔄 Processing Chapter {_chap_num}/{total_chapters}")
            
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
                        print(f"   → Including chapter {g_idx+1}")
                
                chapter_content = "\n\n".join(merged_contents)
                print(f"   📊 Merged content: {len(chapter_content):,} characters")
            
            # Build merged chapter nums for watchdog progress bar
            merged_chapter_nums = [g_idx + 1 for g_idx, _ in merge_groups[idx]] if idx in merge_groups else None

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
                        f"💬 Chapter {idx+1} combined prompt: "
                        f"{total_tokens:,} tokens (system + user: {non_assistant_tokens:,}, "
                        f"assistant/memory: {assistant_tokens:,}) | chunk budget {available_tokens:,}"
                    )
                else:
                    print(
                        f"💬 Chapter {idx+1} combined prompt: "
                        f"{total_tokens:,} tokens (system + user) | chunk budget {available_tokens:,}"
                    )

                # Determine if we need to split based on output-limit budget
                chapter_tokens = chapter_splitter.count_tokens(chapter_content)
                if chapter_split_enabled and chapter_tokens > available_tokens:
                    print(f"⚠️ Chapter {idx+1} exceeds chunk budget: {chapter_tokens:,} > {available_tokens:,}")
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
                            print(f"❌ Glossary extraction stopped during chunk {chunk_idx} of chapter {idx+1}")
                            return
                            
                        print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {idx+1}")
                        
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
                                chunk_idx=chunk_idx,
                                total_chunks=total_chunks,
                                merged_chapters=merged_chapter_nums
                            )
                        except UnifiedClientError as e:
                            if "stopped by user" in str(e).lower():
                                print(f"❌ Glossary extraction stopped during chunk {chunk_idx} API call")
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
                                return
                    
                    # Use the collected data from all chunks
                    data = chapter_glossary_data
                    resp = ""  # Combined response not needed for progress tracking
                    # Set raw_obj to None for chunked processing (history was already saved per chunk)
                    raw_obj = None
                    print(f"✅ Chapter {idx+1} processed in {len(chunks)} chunks, total entries: {len(data)}")
                    
                else:
                    # Original single-chapter processing
                    # Check for stop before API call
                    if check_stop():
                        print(f"❌ Glossary extraction stopped before API call for chapter {idx+1}")
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
                            merged_chapters=merged_chapter_nums
                        )
                                
                    except UnifiedClientError as e:
                        if "stopped by user" in str(e).lower():
                            print(f"❌ Glossary extraction stopped during API call for chapter {idx+1}")
                            return
                        elif "timeout" in str(e).lower():
                            print(f"⚠️ API call timed out for chapter {idx+1}: {e}")
                            continue
                        else:
                            print(f"❌ API error for chapter {idx+1}: {e}")
                            continue
                    except Exception as e:
                        print(f"❌ Unexpected error for chapter {idx+1}: {e}")
                        continue
                    
                    # Handle response
                    if raw is None:
                        print(f"❌ API returned None for chapter {idx+1}")
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
                        print(f"❌ Unexpected response type for chapter {idx+1}: {type(raw)}")
                        resp = str(raw) if raw is not None else ""

                    # Ensure resp is a string
                    if not isinstance(resp, str):
                        print(f"⚠️ Converting non-string response to string for chapter {idx+1}")
                        resp = str(resp) if resp is not None else ""

                    # NULL CHECK before checking if response is empty
                    if resp is None:
                        print(f"⚠️ Response is None for chapter {idx+1}, skipping...")
                        if idx not in failed:
                            failed.append(idx)
                        continue

                    # Check if response is empty
                    if not resp or resp.strip() == "":
                        print(f"⚠️ Empty response for chapter {idx+1}, skipping...")
                        if idx not in failed:
                            failed.append(idx)
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
                        print(f"❌ Error parsing response for chapter {idx+1}: {e}")
                        print(f"   Response preview: {resp[:200] if resp else 'None'}...")
                        if idx not in failed:
                            failed.append(idx)
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
                            print(f"❌ Glossary extraction stopped during entry processing for chapter {idx+1}")
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
                        
                        print(f'[Chapter {_chapter_positions.get(idx, idx+1)}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry_type}: {raw_name} ({trans_name})')    
                    
                # Check if this was actually a failure (empty/refused content)
                _resp_text = locals().get('resp', '') or ''
                _is_empty_failure = (not data) and (not _resp_text.strip() or _resp_text.strip() in ('[]', '{}'))
                
                if _is_empty_failure:
                    # Empty/refused response — mark as failed so it gets retried
                    print(f"⚠️ Chapter {idx+1} returned empty/refused content — marking as failed for retry")
                    if idx not in failed:
                        failed.append(idx)
                    # Also mark merged children as failed
                    if idx in merge_groups:
                        for g_idx, _ in merge_groups[idx]:
                            if g_idx != idx and g_idx not in failed:
                                failed.append(g_idx)
                            if g_idx != idx and g_idx not in merged_indices:
                                merged_indices.append(g_idx)
                else:
                    # Apply skip logic and save
                    glossary.extend(data)
                    glossary[:] = skip_duplicate_entries(glossary)
                    completed.append(idx)
                    
                    # Mark truncated chapters as failed so they get retried
                    # finish_reason comes from single-chapter mode, chunk_finish_reason from chunked mode
                    _fr = locals().get('finish_reason') or locals().get('chunk_finish_reason', 'stop')
                    if _fr in ('length', 'MAX_TOKENS', 'max_tokens'):
                        print(f"⚠️ Chapter {idx+1} was truncated — entries kept but chapter will be retried")
                        if idx not in failed:
                            failed.append(idx)
                    
                    # If this was a merged request, also mark child chapters as completed
                    if idx in merge_groups:
                        marked_children = []
                        for g_idx, _ in merge_groups[idx]:
                            if g_idx != idx and g_idx not in completed:
                                completed.append(g_idx)
                                marked_children.append(g_idx + 1)
                            if g_idx != idx and g_idx not in merged_indices:
                                merged_indices.append(g_idx)
                        if marked_children:
                            print(f"   ✅ Marked chapters {marked_children} as completed (merged with {idx+1})")

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
                            print(f"⚠️ Failed to save history for chapter {idx+1}: {e}")

                save_progress(completed, glossary, merged_indices, failed=failed)
                save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                
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
                    print(f"❌ Glossary extraction stopped after processing chapter {idx+1}")
                    return

            except Exception as e:
                print(f"Error at chapter {idx+1}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                if idx not in failed:
                    failed.append(idx)
                # Check for stop even after error
                if check_stop():
                    print(f"❌ Glossary extraction stopped after error in chapter {idx+1}")
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
        print(f"\n⚠️ {len(failed)} chapter(s) failed and will be retried on next run: {[i+1 for i in sorted(failed)]}")
        save_progress(completed, glossary, merged_indices, failed=failed)
    
    print(f"\nDone. Glossary saved to {args.output}")
    
    # Also save as CSV format for compatibility
    try:
        csv_output = args.output.replace('.json', '.csv')
        csv_path = os.path.join(glossary_dir, os.path.basename(csv_output))
        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
        print(f"Also saved as CSV: {csv_path}")
    except Exception as e:
        print(f"[Warning] Could not save CSV format: {e}")

def save_progress(completed: List[int], glossary: List[Dict], merged_indices: List[int] = None, failed: List[int] = None):
    """Save progress to JSON file (history is now managed separately)
    
    NOTE: We no longer save the glossary itself in the progress file to avoid
    overwriting manual edits. The progress file only tracks which chapters are completed/failed.
    The actual glossary data is saved separately in the output JSON/CSV files.
    """
    global _progress_lock
    
    # Ensure book title entry is present in-memory before recording status
    glossary = _ensure_book_title_entry(glossary)

    # Refresh book-title status from current glossary snapshot
    def _refresh_book_title_flags():
        global BOOK_TITLE_PRESENT, BOOK_TITLE_VALUE
        for entry in glossary or []:
            if str(entry.get("type", "")).lower() == "book":
                BOOK_TITLE_PRESENT = True
                BOOK_TITLE_VALUE = entry.get("translated_name") or entry.get("raw_name")
                return
        BOOK_TITLE_PRESENT = False
        BOOK_TITLE_VALUE = None

    _refresh_book_title_flags()

    # Acquire lock to prevent concurrent writes
    with _progress_lock:
        progress_data = {
            "completed": completed,
            "book_title_present": bool(BOOK_TITLE_PRESENT),
            # Use value from entry if present, otherwise fallback to global translated title
            "book_title": BOOK_TITLE_VALUE if BOOK_TITLE_PRESENT else BOOK_TITLE_TRANSLATED,
            # Glossary is saved separately to output files, not in progress
            # This prevents the progress file from overwriting manual edits
        }
        
        # Add merged_indices if provided
        if merged_indices is not None:
            progress_data["merged_indices"] = merged_indices
        
        # Add failed chapters list
        if failed is not None:
            progress_data["failed"] = failed
        
        try:
            # Use atomic write with proper temp file handling
            progress_dir = os.path.dirname(PROGRESS_FILE) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())  # Ensure data is written to disk
            
            # Atomic rename
            try:
                if os.path.exists(PROGRESS_FILE):
                    os.remove(PROGRESS_FILE)
                os.rename(temp_path, PROGRESS_FILE)
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
