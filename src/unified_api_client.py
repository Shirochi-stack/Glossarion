# unified_api_client.py - REFACTORED with Enhanced Error Handling and Extended AI Model Support
"""
Key Design Principles:
- The client handles API communication and returns accurate status
- The client must save responses properly for duplicate detection
- The client must return accurate finish_reason for truncation detection
- The client must support cancellation for timeout handling

Supported models and their prefixes (Updated July 2025):
- OpenAI: gpt*, o1*, o3*, o4*, codex* (e.g., gpt-4, gpt-4o, gpt-4o-mini, gpt-4.5, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, o3-mini, o3-pro, o4-mini)
- Google: gemini*, palm*, bard* (e.g., gemini-2.0-flash-exp, gemini-2.5-pro, gemini-2.5-flash)
- Anthropic: claude*, sonnet*, opus*, haiku* (e.g., claude-3.5-sonnet, claude-3.7-sonnet, claude-4-opus, claude-4-sonnet, claude-opus-4-20250514, claude-sonnet-4-20250514)
- DeepSeek: deepseek* (e.g., deepseek-chat, deepseek-vl, deepseek-r1)
- Mistral: mistral*, mixtral*, codestral*
- Cohere: command*, cohere*, aya* (e.g., aya-vision, command-r7b)
- AI21: j2*, jurassic*, jamba*
- Together AI: llama*, together*, alpaca*, vicuna*, wizardlm*, openchat*
- Perplexity: perplexity*, pplx*, sonar*
- Replicate: replicate*
- Yi (01.AI): yi* (e.g., yi-34b-chat-200k, yi-vl)
- Qwen (Alibaba): qwen* (e.g., qwen2.5-vl)
- Baichuan: baichuan*
- Zhipu AI: glm*, chatglm*
- Moonshot: moonshot*, kimi*
- Groq: groq*, llama-groq*, mixtral-groq*
- Baidu: ernie*
- Tencent: hunyuan*
- iFLYTEK: spark*
- ByteDance: doubao*
- MiniMax: minimax*, abab*
- SenseNova: sensenova*, nova*
- InternLM: intern*, internlm*
- TII: falcon* (e.g., falcon-2-11b)
- Microsoft: phi*, orca*
- Azure: azure* (for Azure OpenAI deployments)
- Aleph Alpha: luminous*
- Databricks: dolly*
- HuggingFace: starcoder*
- Salesforce: codegen*
- BigScience: bloom*
- Meta: opt*, galactica*, llama2*, llama3*, llama4*, codellama*
- xAI: grok* (e.g., grok-3, grok-vision)
- Poe: poe/* (e.g., poe/claude-4-opus, poe/gpt-4.5, poe/Assistant)
- OpenRouter: or/*, openrouter/* (e.g., or/anthropic/claude-4-opus, or/openai/gpt-4.5)
- Fireworks AI: fireworks/* (e.g., fireworks/llama-v3-70b)

ELECTRONHUB SUPPORT:
ElectronHub is an API aggregator that provides access to multiple models.
To use ElectronHub, prefix your model name with one of these:
- eh/ (e.g., eh/yi-34b-chat-200k)
- electronhub/ (e.g., electronhub/gpt-4.5)
- electron/ (e.g., electron/claude-4-opus)

ElectronHub allows you to access models from multiple providers using a single API key.

POE SUPPORT:
Poe by Quora provides access to multiple AI models through their platform.
To use Poe, prefix your model name with 'poe/':
- poe/claude-4-opus
- poe/claude-4-sonnet
- poe/gpt-4.5
- poe/gpt-4.1
- poe/Assistant
- poe/gemini-2.5-pro

OPENROUTER SUPPORT:
OpenRouter is a unified interface for 300+ models from various providers.
To use OpenRouter, prefix your model name with 'or/' or 'openrouter/':
- or/anthropic/claude-4-opus
- openrouter/openai/gpt-4.5
- or/google/gemini-2.5-pro
- or/meta-llama/llama-4-70b

Environment Variables:
- SEND_INTERVAL_SECONDS: Delay between API calls (respects GUI settings)
- YI_API_BASE_URL: Custom endpoint for Yi models (optional)
- ELECTRONHUB_API_URL: Custom ElectronHub endpoint (default: https://api.electronhub.ai/v1)
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint (for azure* models)
- AZURE_API_VERSION: Azure API version (default: 2024-02-01)
- DATABRICKS_API_URL: Databricks workspace URL
- SALESFORCE_API_URL: Salesforce API endpoint
- OPENROUTER_REFERER: HTTP referer for OpenRouter (default: https://github.com/Shirochi-stack/Glossarion)
- OPENROUTER_APP_NAME: App name for OpenRouter (default: Glossarion Translation)
- POE_API_KEY: API key for Poe platform
- GROQ_API_URL: Custom Groq endpoint (default: https://api.groq.com/openai/v1)
- FIREWORKS_API_URL: Custom Fireworks AI endpoint (default: https://api.fireworks.ai/inference/v1)
- DISABLE_GEMINI_SAFETY: Set to "true" to disable Gemini safety filters (respects GUI toggle)
- XAI_API_URL: Custom xAI endpoint (default: https://api.x.ai/v1)
- DEEPSEEK_API_URL: Custom DeepSeek endpoint (default: https://api.deepseek.com/v1)

SAFETY SETTINGS:
The client respects the GUI's "Disable Gemini API Safety Filters" toggle via the 
DISABLE_GEMINI_SAFETY environment variable. When enabled, it applies API-level safety 
settings where available:

- Gemini: Sets all harm categories to BLOCK_NONE (most permissive)
- OpenRouter: Disables safe mode via X-Safe-Mode header
- Poe: Disables safe mode via safe_mode parameter
- Other OpenAI-compatible providers: Sets moderation=false where supported

Note: Not all providers support API-level safety toggles. OpenAI and Anthropic APIs
do not have direct safety filter controls. The client only applies settings that are
officially supported by each provider's API.

Note: Many Chinese model providers (Yi, Qwen, Baichuan, etc.) may require
API keys from their respective platforms. Some endpoints might need adjustment
based on your region or deployment.
"""
import os
import json
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging
import re
import base64
import contextlib
from PIL import Image
import io
import time
import random
import csv
from datetime import datetime
import traceback
import hashlib
import html
try:
    from multi_api_key_manager import APIKeyPool, APIKeyEntry, RateLimitCache
except ImportError:
    try:
        from .multi_api_key_manager import APIKeyPool, APIKeyEntry, RateLimitCache
    except ImportError:
        # Fallback classes if module not available
        class APIKeyPool:
            def __init__(self): pass
        class APIKeyEntry:
            def __init__(self): pass
        class RateLimitCache:
            def __init__(self): pass
import threading
import uuid
from threading import RLock
from collections import defaultdict
logger = logging.getLogger(__name__)

# IMPORTANT: This client respects GUI settings via environment variables:
# - SEND_INTERVAL_SECONDS: Delay between API calls (set by GUI)
# All API providers INCLUDING ElectronHub respect this setting for proper GUI integration

# Note: For Yi models through ElectronHub, use eh/yi-34b-chat-200k format
# For direct Yi API access, use yi-34b-chat-200k format

# Set up logging
logger = logging.getLogger(__name__)

# Enable HTTP request logging for debugging
def setup_http_logging():
    """Enable detailed HTTP request/response logging for debugging"""
    import logging
    
    # Enable httpx logging (used by OpenAI SDK)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.INFO)
    
    # Enable requests logging (fallback HTTP calls)
    requests_logger = logging.getLogger("requests.packages.urllib3")
    requests_logger.setLevel(logging.INFO)
    
    # Enable OpenAI SDK logging
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.DEBUG)
    
    # Create console handler if not exists
    if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        console_handler.setFormatter(formatter)
        
        httpx_logger.addHandler(console_handler)
        requests_logger.addHandler(console_handler)
        openai_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        httpx_logger.propagate = False
        requests_logger.propagate = False
        openai_logger.propagate = False

# Enable HTTP logging on module import
setup_http_logging()

# OpenAI SDK
try:
    import openai
    from openai import OpenAIError
except ImportError:
    openai = None
    class OpenAIError(Exception): pass
try:
    import httpx
    from httpx import HTTPStatusError
except ImportError:
    httpx = None
    class HTTPStatusError(Exception): pass
    
# Gemini SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

# Anthropic SDK (optional - can use requests if not installed)
try:
    import anthropic
except ImportError:
    anthropic = None

# Cohere SDK (optional)
try:
    import cohere
except ImportError:
    cohere = None

# Mistral SDK (optional)
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    MistralClient = None
    
# Google Vertex AI API Cloud
try:
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform")

try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    deepl = None
    DEEPL_AVAILABLE = False

try:
    from google.cloud import translate_v2 as google_translate
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    google_translate = None
    GOOGLE_TRANSLATE_AVAILABLE = False
    
from functools import lru_cache
from datetime import datetime, timedelta
 

@dataclass
class UnifiedResponse:
    """Standardized response format for all API providers"""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    error_details: Optional[Dict[str, Any]] = None
    
    @property
    def is_truncated(self) -> bool:
        """Check if the response was truncated
        
        IMPORTANT: This is used by retry logic to detect when to retry with more tokens
        """
        return self.finish_reason in ['length', 'max_tokens', 'stop_sequence_limit', 'truncated', 'incomplete']
    
    @property
    def is_complete(self) -> bool:
        """Check if the response completed normally"""
        return self.finish_reason in ['stop', 'complete', 'end_turn', 'finished', None]
    
    @property
    def is_error(self) -> bool:
        """Check if the response is an error"""
        return self.finish_reason == 'error' or bool(self.error_details)

class UnifiedClientError(Exception):
    """Generic exception for UnifiedClient errors."""
    def __init__(self, message, error_type=None, http_status=None, details=None):
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status
        self.details = details

class UnifiedClient:
    # ----- Helper methods to reduce duplication -----
    @contextlib.contextmanager
    def _model_lock(self):
        """Context manager for thread-safe model access"""
        if hasattr(self, '_instance_model_lock') and self._instance_model_lock is not None:
            with self._instance_model_lock:
                yield
        else:
            # Fallback - create a temporary lock if needed
            if not hasattr(self, '_temp_model_lock'):
                self._temp_model_lock = threading.RLock()
            with self._temp_model_lock:
                yield
    def _get_send_interval(self) -> float:
        try:
            return float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        except Exception:
            return 2.0

    def _safe_len(self, obj, context="unknown"):
        """Safely get length of an object with better error reporting"""
        try:
            if obj is None:
                print(f"⚠️ Warning: Attempting to get length of None in context: {context}")
                return 0
            return len(obj)
        except TypeError as e:
            print(f"❌ TypeError in _safe_len for context '{context}': {e}")
            print(f"❌ Object type: {type(obj)}, Object value: {obj}")
            return 0
        except Exception as e:
            print(f"❌ Unexpected error in _safe_len for context '{context}': {e}")
            return 0

    def _extract_first_image_base64(self, messages) -> Optional[str]:
        if messages is None:
            return None
        for msg in messages:
            if msg is None:
                continue
            content = msg.get('content')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        url = part.get('image_url', {}).get('url', '')
                        if isinstance(url, str):
                            if url.startswith('data:') and ',' in url:
                                return url.split(',', 1)[1]
                            return url
        return None
    

    def _get_timeout_config(self) -> Tuple[bool, int]:
        enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
        window = int(os.getenv("CHUNK_TIMEOUT", "180"))
        return enabled, window
    def _with_attempt_suffix(self, payload_name: str, response_name: str, request_id: str, attempt: int, is_image: bool) -> Tuple[str, str]:
        base_payload, ext_payload = os.path.splitext(payload_name)
        base_response, ext_response = os.path.splitext(response_name)
        unique_suffix = f"_{request_id}_imgA{attempt}" if is_image else f"_{request_id}_A{attempt}"
        return f"{base_payload}{unique_suffix}{ext_payload}", f"{base_response}{unique_suffix}{ext_response}"

    def _maybe_retry_main_key_on_prohibited(self, messages, temperature, max_tokens, max_completion_tokens, context, request_id=None, image_data=None):
        if not (self._multi_key_mode and getattr(self, 'original_api_key', None) and getattr(self, 'original_model', None)):
            return None
        try:
            return self._retry_with_main_key(
                messages, temperature, max_tokens, max_completion_tokens, context,
                request_id=request_id, image_data=image_data
            )
        except Exception:
            return None

    def _detect_safety_filter(self, messages, extracted_content: str, finish_reason: Optional[str], response: Any, provider: str) -> bool:
        # Heuristic patterns consolidated from previous branches
        # 1) Suspicious finish reasons with empty content
        if not extracted_content and finish_reason in ['length', 'stop', 'max_tokens', None]:
            return True
        # 2) Safety indicators in raw response/error details
        response_str = ""
        if response is not None:
            if hasattr(response, 'raw_response') and response.raw_response is not None:
                response_str = str(response.raw_response).lower()
            elif hasattr(response, 'error_details') and response.error_details is not None:
                response_str = str(response.error_details).lower()
            else:
                response_str = str(response).lower()
        safety_indicators = [
            'safety', 'blocked', 'prohibited', 'harmful', 'inappropriate',
            'refused', 'content_filter', 'content policy', 'violation',
            'cannot assist', 'unable to process', 'against guidelines',
            'ethical', 'responsible ai', 'harm_category', 'nsfw',
            'adult content', 'explicit', 'violence', 'disturbing'
        ]
        if any(ind in response_str for ind in safety_indicators):
            return True
        # 3) Safety phrases in extracted content
        if extracted_content:
            content_lower = extracted_content.lower()
            safety_phrases = [
                'blocked', 'safety', 'cannot', 'unable', 'prohibited',
                'content filter', 'refused', 'inappropriate', 'i cannot',
                "i can't", "i'm not able", "not able to", "against my",
                'content policy', 'guidelines', 'ethical',
                'analyze this image', 'process this image', 'describe this image', 'nsfw'
            ]
            if any(p in content_lower for p in safety_phrases):
                return True
        # 4) Provider-specific empty behavior
        if provider in ['openai', 'azure', 'electronhub', 'openrouter', 'poe', 'gemini']:
            if not extracted_content and finish_reason != 'error':
                return True
        # 5) Suspiciously short output vs long input - FIX: Add None checks
        if extracted_content and len(extracted_content) < 50:
            # FIX: Add None check for messages
            if messages is not None:
                input_length = 0
                for m in messages:
                    if m is not None and m.get('role') == 'user':
                        content = m.get('content', '')
                        # FIX: Add None check for content
                        if content is not None:
                            input_length += len(str(content))
                if input_length > 200 and any(w in extracted_content.lower() for w in ['cannot', 'unable', 'sorry', 'assist']):
                    return True
        return False

    def _finalize_empty_response(self, messages, context, response, extracted_content: str, finish_reason: Optional[str], provider: str, request_type: str, start_time: float) -> Tuple[str, str]:
        is_safety = self._detect_safety_filter(messages, extracted_content, finish_reason, response, provider)
        # Always save failure snapshot and log truncation details
        self._save_failed_request(messages, f"Empty {request_type} response from {getattr(self, 'client_type', 'unknown')}", context, response)
        error_details = getattr(response, 'error_details', None)
        if is_safety:
            error_details = {
                'likely_safety_filter': True,
                'original_finish_reason': finish_reason,
                'provider': getattr(self, 'client_type', None),
                'model': self.model,
                'key_identifier': getattr(self, 'key_identifier', None),
                'request_type': request_type
            }
        self._log_truncation_failure(
            messages=messages,
            response_content=extracted_content or "",
            finish_reason='content_filter' if is_safety else (finish_reason or 'error'),
            context=context,
            error_details=error_details
        )
        # Stats
        self._track_stats(context, False, f"empty_{request_type}_response", time.time() - start_time)
        # Fallback message
        if is_safety:
            fb_reason = f"image_safety_filter_{provider}" if request_type == 'image' else f"safety_filter_{provider}"
        else:
            fb_reason = "empty_image" if request_type == 'image' else "empty"
        fallback = self._handle_empty_result(messages, context, getattr(response, 'error_details', fb_reason) if response else fb_reason)
        return fallback, ('content_filter' if is_safety else 'error')

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        s = str(exc).lower()
        if hasattr(exc, 'error_type') and getattr(exc, 'error_type') == 'rate_limit':
            return True
        return ('429' in s) or ('rate limit' in s) or ('quota' in s)

    def _compute_backoff(self, attempt: int, base: float, cap: float) -> float:
        delay = (base * (2 ** attempt)) + random.uniform(0, 1)
        return min(delay, cap)

    def _normalize_token_params(self, max_tokens: Optional[int], max_completion_tokens: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        if self._is_o_series_model():
            mct = max_completion_tokens if max_completion_tokens is not None else (max_tokens or getattr(self, 'default_max_tokens', 8192))
            return None, mct
        else:
            mt = max_tokens if max_tokens is not None else (max_completion_tokens or getattr(self, 'default_max_tokens', 8192))
            return mt, None
    def _apply_api_delay(self) -> None:
        if getattr(self, '_in_cleanup', False):
            print("⚡ Skipping API delay (cleanup mode)")
            return
        try:
            api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        except Exception:
            api_delay = 2.0
        if api_delay > 0:
            print(f"⏳ Waiting {api_delay}s before next API call...")
            time.sleep(api_delay)

    def _set_idempotency_context(self, request_id: str, attempt: int) -> None:
        tls = self._get_thread_local_client()
        tls.idem_request_id = request_id
        tls.idem_attempt = attempt

    def _get_extraction_kwargs(self) -> dict:
        ct = getattr(self, 'client_type', None)
        if ct == 'gemini':
            return {
                'supports_thinking': self._supports_thinking(),
                'thinking_budget': int(os.getenv("THINKING_BUDGET", "-1")),
            }
        return {}

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        s = str(exc).lower()
        if hasattr(exc, 'error_type') and getattr(exc, 'error_type') == 'rate_limit':
            return True
        return ('429' in s) or ('rate limit' in s) or ('quota' in s)

    def _compute_backoff(self, attempt: int, base: float, cap: float) -> float:
        delay = (base * (2 ** attempt)) + random.uniform(0, 1)
        return min(delay, cap)

    def _normalize_token_params(self, max_tokens: Optional[int], max_completion_tokens: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        if self._is_o_series_model():
            mct = max_completion_tokens if max_completion_tokens is not None else (max_tokens or getattr(self, 'default_max_tokens', 8192))
            return None, mct
        else:
            mt = max_tokens if max_tokens is not None else (max_completion_tokens or getattr(self, 'default_max_tokens', 8192))
            return mt, None
    """
    Unified client with fixed thread-safe multi-key support
    
    Key improvements:
    1. Thread-local storage for API clients
    2. Proper key rotation per request
    3. Thread-safe rate limit handling
    4. Cleaner error handling and retry logic
    5. INSTANCE-BASED multi-key mode (not class-based)
    """
    # Thread safety for file operations
    _file_write_lock = RLock()
    _tracker_lock = RLock()
    _model_lock = RLock()
    
    # Class-level shared resources - properly initialized
    _rate_limit_cache = None
    _api_key_pool: Optional[APIKeyPool] = None
    _pool_lock = threading.Lock()
    
    # Request tracking
    _global_request_counter = 0
    _counter_lock = threading.Lock()
    
    # Thread-local storage for clients and key assignments
    _thread_local = threading.local()
    
    # global stop flag
    _global_cancelled = False
    
    # Legacy tracking (for compatibility)
    _key_assignments = {}  # thread_id -> (key_index, key_identifier)
    _assignment_lock = threading.Lock()
    
    # Track displayed log messages to avoid spam
    _displayed_messages = set()
    _message_lock = threading.Lock()
    
    # Your existing MODEL_PROVIDERS and other class variables
    MODEL_PROVIDERS = {
        'vertex/': 'vertex_model_garden',
        '@': 'vertex_model_garden',
        'gpt': 'openai',
        'o1': 'openai',
        'o3': 'openai',
        'o4': 'openai',
        'gemini': 'gemini',
        'claude': 'anthropic',
        'chutes': 'chutes',
        'chutes/': 'chutes',
        'sonnet': 'anthropic',
        'opus': 'anthropic',
        'haiku': 'anthropic',
        'deepseek': 'deepseek',
        'mistral': 'mistral',
        'mixtral': 'mistral',
        'codestral': 'mistral',
        'command': 'cohere',
        'cohere': 'cohere',
        'aya': 'cohere',
        'j2': 'ai21',
        'jurassic': 'ai21',
        'llama': 'together',
        'together': 'together',
        'perplexity': 'perplexity',
        'pplx': 'perplexity',
        'sonar': 'perplexity',
        'replicate': 'replicate',
        'yi': 'yi',
        'qwen': 'qwen',
        'baichuan': 'baichuan',
        'glm': 'zhipu',
        'chatglm': 'zhipu',
        'moonshot': 'moonshot',
        'kimi': 'moonshot',
        'groq': 'groq',
        'llama-groq': 'groq',
        'mixtral-groq': 'groq',
        'ernie': 'baidu',
        'hunyuan': 'tencent',
        'spark': 'iflytek',
        'doubao': 'bytedance',
        'minimax': 'minimax',
        'abab': 'minimax',
        'sensenova': 'sensenova',
        'nova': 'sensenova',
        'intern': 'internlm',
        'internlm': 'internlm',
        'falcon': 'tii',
        'jamba': 'ai21',
        'phi': 'microsoft',
        'azure': 'azure',
        'palm': 'google',
        'bard': 'google',
        'codex': 'openai',
        'luminous': 'alephalpha',
        'alpaca': 'together',
        'vicuna': 'together',
        'wizardlm': 'together',
        'openchat': 'together',
        'orca': 'microsoft',
        'dolly': 'databricks',
        'starcoder': 'huggingface',
        'codegen': 'salesforce',
        'bloom': 'bigscience',
        'opt': 'meta',
        'galactica': 'meta',
        'llama2': 'meta',
        'llama3': 'meta',
        'llama4': 'meta',
        'codellama': 'meta',
        'grok': 'xai',
        'poe': 'poe',
        'or': 'openrouter',
        'openrouter': 'openrouter',
        'fireworks': 'fireworks',
        'eh/': 'electronhub',
        'electronhub/': 'electronhub',
        'electron/': 'electronhub',
        'deepl': 'deepl',
        'google-translate': 'google_translate',
    }
    
    # Model-specific constraints
    MODEL_CONSTRAINTS = {
        'temperature_fixed': ['o4-mini', 'o1-mini', 'o1-preview', 'o3-mini', 'o3', 'o3-pro', 'o4-mini', 'gpt-5-mini','gpt-5','gpt-5-nano'],
        'no_system_message': ['o1', 'o1-preview', 'o3', 'o3-pro'],
        'max_completion_tokens': ['o4', 'o1', 'o3', 'gpt-5-mini','gpt-5','gpt-5-nano'],
        'chinese_optimized': ['qwen', 'yi', 'glm', 'chatglm', 'baichuan', 'ernie', 'hunyuan'],
    }
    
    @classmethod
    def _log_once(cls, message: str, is_debug: bool = False):
        """Log a message only once per session to avoid spam"""
        with cls._message_lock:
            if message not in cls._displayed_messages:
                cls._displayed_messages.add(message)
                if is_debug:
                    print(f"[DEBUG] {message}")
                else:
                    logger.info(message)
                return True
        return False
    
    @classmethod
    def setup_multi_key_pool(cls, keys_list, force_rotation=True, rotation_frequency=1):
        """Setup the shared API key pool"""
        with cls._pool_lock:
            if cls._api_key_pool is None:
                cls._api_key_pool = APIKeyPool()
            
            # Initialize rate limit cache if needed
            if cls._rate_limit_cache is None:
                cls._rate_limit_cache = RateLimitCache()
            
            # Validate and fix encrypted keys
            validated_keys = []
            encrypted_keys_fixed = 0
            
            # FIX 1: Use keys_list parameter instead of undefined 'config'
            for i, key_data in enumerate(keys_list):
                if not isinstance(key_data, dict):
                    continue
                    
                api_key = key_data.get('api_key', '')
                if not api_key:
                    continue
                
                # Fix encrypted keys
                if api_key.startswith('ENC:'):
                    try:
                        from api_key_encryption import get_handler
                        handler = get_handler()
                        decrypted_key = handler.decrypt_value(api_key)
                        
                        if decrypted_key != api_key and not decrypted_key.startswith('ENC:'):
                            # Create a copy with decrypted key
                            fixed_key_data = key_data.copy()
                            fixed_key_data['api_key'] = decrypted_key
                            validated_keys.append(fixed_key_data)
                            encrypted_keys_fixed += 1
                    except Exception:
                        continue
                else:
                    # Key is already decrypted
                    validated_keys.append(key_data)
            
            if not validated_keys:
                return False
            
            # Load the validated keys
            cls._api_key_pool.load_from_list(validated_keys)
            #cls._main_fallback_key = validated_keys[0]['api_key']
            #cls._main_fallback_model = validated_keys[0]['model']
            #print(f"🔑 Using {validated_keys[0]['model']} as main fallback key")

            # FIX 2: Store settings at class level (these affect all instances)
            # These are class variables since pool is shared
            if not hasattr(cls, '_force_rotation'):
                cls._force_rotation = force_rotation
            if not hasattr(cls, '_rotation_frequency'):
                cls._rotation_frequency = rotation_frequency
            
            # Or update if provided
            cls._force_rotation = force_rotation
            cls._rotation_frequency = rotation_frequency
            
            # Single debug message
            if encrypted_keys_fixed > 0:
                print(f"🔑 Multi-key pool: {len(validated_keys)} keys loaded ({encrypted_keys_fixed} required decryption fix)")
            else:
                print(f"🔑 Multi-key pool: {len(validated_keys)} keys loaded")
            
            return True
    
    @classmethod
    def initialize_key_pool(cls, key_list: list):
        """Initialize the shared API key pool (legacy compatibility)"""
        with cls._pool_lock:
            if cls._api_key_pool is None:
                cls._api_key_pool = APIKeyPool()
            cls._api_key_pool.load_from_list(key_list)
    
    @classmethod
    def get_key_pool(cls):
        """Get the shared API key pool (legacy compatibility)"""
        with cls._pool_lock:
            if cls._api_key_pool is None:
                cls._api_key_pool = APIKeyPool()
            return cls._api_key_pool
    
    def _get_max_retries(self) -> int:
        """Get max retry count from environment variable, default to 7"""
        return int(os.getenv('MAX_RETRIES', '7'))
    
    # Class-level cancellation flag for all instances
    _global_cancelled = False
    _global_cancel_lock = threading.RLock()
    
    @classmethod
    def set_global_cancellation(cls, cancelled: bool):
        """Set global cancellation flag for all client instances"""
        with cls._global_cancel_lock:
            cls._global_cancelled = cancelled
    
    @classmethod
    def is_globally_cancelled(cls) -> bool:
        """Check if globally cancelled"""
        with cls._global_cancel_lock:
            return cls._global_cancelled
    
    def __init__(self, api_key: str, model: str, output_dir: str = "Output"):
        """Initialize the unified client with enhanced thread safety"""
        # Store original values
        self.original_api_key = api_key
        self.original_model = model
        
        self._sequential_send_lock = threading.Lock()      
        
        # Thread submission timing controls
        self._thread_submission_lock = threading.Lock()
        self._last_thread_submission_time = 0
        self._thread_submission_count = 0
        
        # Add unique session ID for this client instance
        self.session_id = str(uuid.uuid4())[:8]
        
        # INSTANCE-LEVEL multi-key configuration
        self._multi_key_mode = False  # INSTANCE variable, not class!
        self._force_rotation = True
        self._rotation_frequency = 1
        
        # Instance variables
        self.output_dir = output_dir
        self._cancelled = False
        self._in_cleanup = False
        self.conversation_message_count = 0
        self.context = None
        self.current_session_context = None
        
        # Request tracking (from first init)
        self._request_count = 0
        self._thread_request_count = 0
        
        # Thread coordination for key assignment
        self._key_assignment_lock = RLock()
        self._thread_key_assignments = {}  # {thread_id: (key_index, timestamp)}
        self._instance_model_lock = threading.RLock()
        
        # Thread-local storage for client instances
        self._thread_local = threading.local()
        
        # Thread-specific request counters
        self._thread_request_counters = defaultdict(int)
        self._counter_lock = RLock()
        
        # File write coordination
        self._file_write_locks = {}  # {filepath: RLock}
        self._file_write_locks_lock = RLock()
        if not hasattr(self, '_instance_model_lock'):
            self._instance_model_lock = threading.RLock()
        
        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'errors': defaultdict(int),
            'response_times': [],
            'empty_results': 0  # Add this for completeness
        }
        
        # Pattern recognition attributes
        self.pattern_counts = {}  # Track pattern frequencies for reinforcement
        self.last_pattern = None  # Track last seen pattern
        
        # Call reset_stats if it exists (from first init)
        if hasattr(self, 'reset_stats'):
            self.reset_stats()
        
        # File tracking for duplicate prevention
        self._active_files = set()  # Track files being written
        self._file_lock = RLock()
        
        # Timeout configuration
        enabled, window = self._get_timeout_config()
        self.request_timeout = int(os.getenv("CHUNK_TIMEOUT", "900")) if enabled else 36000  # 10 hours
        
        # Initialize client references
        self.api_key = api_key
        self.model = model
        self.key_identifier = "Single Key"
        self.current_key_index = None
        self.openai_client = None
        self.gemini_client = None
        self.mistral_client = None
        self.cohere_client = None
        self._actual_output_filename = None
        self._current_output_file = None
        self._last_response_filename = None
        
        
        # Store Google Cloud credentials path if available
        self.google_creds_path = None
        # Store current key's Google credentials, Azure endpoint, and Google region
        self.current_key_google_creds = None
        self.current_key_azure_endpoint = None
        self.current_key_google_region = None
        
        # Azure-specific flags
        self.is_azure = False
        self.azure_endpoint = None
        self.azure_api_version = None

        self.translator_config = {
            'use_fallback_keys': os.getenv('USE_FALLBACK_KEYS', '0') == '1',
            'fallback_keys': json.loads(os.getenv('FALLBACK_KEYS', '[]'))
        }
        
        # Debug print to verify
        if self.translator_config['use_fallback_keys']:
            num_fallbacks = len(self.translator_config['fallback_keys'])
            print(f"🔑 Fallback keys loaded: {num_fallbacks} keys")
        
        # Check if multi-key mode should be enabled FOR THIS INSTANCE
        use_multi_keys_env = os.getenv('USE_MULTI_API_KEYS', '0') == '1'
        print(f"[DEBUG] USE_MULTI_API_KEYS env var: {os.getenv('USE_MULTI_API_KEYS')}")
        print(f"[DEBUG] Creating new instance - multi-key mode from env: {use_multi_keys_env}")
        
        if use_multi_keys_env:
            # Initialize from environment
            multi_keys_json = os.getenv('MULTI_API_KEYS', '[]')
            print(f"[DEBUG] Loading multi-keys config...")
            force_rotation = os.getenv('FORCE_KEY_ROTATION', '1') == '1'
            rotation_frequency = int(os.getenv('ROTATION_FREQUENCY', '1'))
            
            try:
                multi_keys = json.loads(multi_keys_json)
                if multi_keys:
                    # Setup the shared pool
                    self.setup_multi_key_pool(multi_keys, force_rotation, rotation_frequency)
                    
                    # Enable multi-key mode FOR THIS INSTANCE
                    self._multi_key_mode = True
                    self._force_rotation = force_rotation
                    self._rotation_frequency = rotation_frequency
                    
                    print(f"[DEBUG] ✅ This instance has multi-key mode ENABLED")
                else:
                    print(f"[DEBUG] ❌ No keys found in config, staying in single-key mode")
                    self._multi_key_mode = False
            except Exception as e:
                print(f"Failed to load multi-key config: {e}")
                self._multi_key_mode = False
                print(f"[DEBUG] ❌ Error loading config, falling back to single-key mode")
        else:
            #print(f"[DEBUG] ❌ Multi-key mode is DISABLED for this instance (env var = 0)")
            self._multi_key_mode = False
        
        # Initial setup based on THIS INSTANCE's mode
        if not self._multi_key_mode:
            self.api_key = api_key
            self.model = model
            self.key_identifier = "Single Key"
            self._setup_client()
        
        # Check for Vertex AI Model Garden models (contain @ symbol)
        # NOTE: This happens AFTER the initial setup, as in the second version
        if '@' in self.model or self.model.startswith('vertex/'):
            # For Vertex AI, we need Google Cloud credentials, not API key
            self.client_type = 'vertex_model_garden'
            
            # Try to find Google Cloud credentials
            # 1. Check environment variable
            self.google_creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            
            # 2. Check if passed as api_key (for compatibility)
            if not self.google_creds_path and api_key and os.path.exists(api_key):
                self.google_creds_path = api_key
                # Use logger if available, otherwise print
                if hasattr(self, 'logger'):
                    self.logger.info("Using API key parameter as Google Cloud credentials path")
                else:
                    print("Using API key parameter as Google Cloud credentials path")
            
            # 3. Will check GUI config later during send if needed
            
            if self.google_creds_path:
                msg = f"Vertex AI Model Garden: Using credentials from {self.google_creds_path}"
                if hasattr(self, 'logger'):
                    self.logger.info(msg)
                else:
                    print(msg)
            else:
                print("Vertex AI Model Garden: Google Cloud credentials not yet configured")
        else:
            # Only set up client if not in multi-key mode
            # Multi-key mode will set up the client when a key is selected
            if not self._multi_key_mode:
                # NOTE: This is a SECOND call to _setup_client() in the else branch
                # Determine client type from model name
                self._setup_client()
                print(f"[DEBUG] After setup - client_type: {getattr(self, 'client_type', None)}, openai_client: {self.openai_client}")
                
                # FORCE OPENAI CLIENT IF CUSTOM BASE URL IS SET AND ENABLED
                use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
                custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')

                # Force OpenAI client when custom endpoint is enabled
                if custom_base_url and use_custom_endpoint and self.openai_client is None:
                    original_client_type = self.client_type
                    print(f"[DEBUG] Custom base URL detected and enabled, overriding {original_client_type or 'unmatched'} model to use OpenAI client: {self.model}")
                    self.client_type = 'openai'
                    
                    # Check if openai module is available
                    try:
                        import openai
                    except ImportError:
                        raise ImportError("OpenAI library not installed. Install with: pip install openai")
                    
                    # Validate URL has protocol
                    if not custom_base_url.startswith(('http://', 'https://')):
                        print(f"[WARNING] Custom base URL missing protocol, adding https://")
                        custom_base_url = 'https://' + custom_base_url
                    
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=custom_base_url
                    )
                    print(f"[DEBUG] OpenAI client created with custom base URL: {custom_base_url}")
                elif custom_base_url and not use_custom_endpoint:
                    print(f"[DEBUG] Custom base URL detected but disabled via toggle, using standard client")
 
    def _apply_thread_submission_delay(self):
        # Get threading delay from environment (default 0.5)
        thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
        
        if thread_delay <= 0:
            return
        
        sleep_time = 0
        should_log = False
        log_message = ""
        
        # HOLD LOCK ONLY BRIEFLY to check timing and update counter
        with self._thread_submission_lock:
            current_time = time.time()
            time_since_last_submission = current_time - self._last_thread_submission_time
            
            if time_since_last_submission < thread_delay:
                sleep_time = thread_delay - time_since_last_submission
                # Update the timestamp NOW while we have the lock
                self._last_thread_submission_time = time.time()
                
                # Determine if we should log (but don't log yet)
                if self._thread_submission_count < 3:
                    should_log = True
                    log_message = f"🧵 [{threading.current_thread().name}] Thread delay: {sleep_time:.1f}s"
                elif self._thread_submission_count == 3:
                    should_log = True
                    log_message = f"🧵 [Subsequent thread delays: {thread_delay}s each...]"
            
            self._thread_submission_count += 1
        # LOCK RELEASED HERE
        
        # NOW do the sleep OUTSIDE the lock
        if sleep_time > 0:
            if should_log:
                print(log_message)
            
            # Interruptible sleep
            elapsed = 0
            check_interval = 0.1
            while elapsed < sleep_time:
                if self._cancelled:
                    print(f"🛑 Threading delay cancelled")
                    return  # Exit early if cancelled
                
                time.sleep(min(check_interval, sleep_time - elapsed))
                elapsed += check_interval
        
    def _get_thread_local_client(self):
        """Get or create thread-local client"""
        thread_id = threading.current_thread().ident
        
        # Check if we need a new client for this thread
        if not hasattr(self._thread_local, 'initialized'):
            self._thread_local.initialized = False
            self._thread_local.api_key = None
            self._thread_local.model = None
            self._thread_local.key_index = None
            self._thread_local.key_identifier = None
            self._thread_local.request_count = 0
            self._thread_local.openai_client = None
            self._thread_local.gemini_client = None
            self._thread_local.mistral_client = None
            self._thread_local.cohere_client = None
            self._thread_local.client_type = None
            self._thread_local.current_request_label = None
            
            # THREAD-LOCAL CACHE
            self._thread_local.request_cache = {}  # Each thread gets its own cache!
            self._thread_local.cache_hits = 0
            self._thread_local.cache_misses = 0
        
        return self._thread_local
    
    def _ensure_thread_client(self):
        """Ensure the current thread has a properly initialized client with thread safety"""
        # Check if cancelled before proceeding
        if self._cancelled:
            raise UnifiedClientError("Operation cancelled", error_type="cancelled")
            
        tls = self._get_thread_local_client()
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        # Multi-key mode
        if self._multi_key_mode:
            # Check if we need to rotate
            should_rotate = False
            
            if not tls.initialized:
                should_rotate = True
                print(f"[Thread-{thread_name}] Initializing with multi-key mode")
            elif self._force_rotation:
                tls.request_count = getattr(tls, 'request_count', 0) + 1
                if tls.request_count >= self._rotation_frequency:
                    should_rotate = True
                    tls.request_count = 0
                    print(f"[Thread-{thread_name}] Rotating key (reached {self._rotation_frequency} requests)")
            
            if should_rotate:
                # Release previous thread assignment to avoid stale usage tracking
                if hasattr(self._api_key_pool, 'release_thread_assignment'):
                    try:
                        self._api_key_pool.release_thread_assignment(thread_id)
                    except Exception:
                        pass
                
                # Get a key using thread-safe method with timeout
                key_info = None
                
                # Add timeout protection for key retrieval
                start_time = time.time()
                max_wait = 120  # 120 seconds max to get a key
                
                # First try using the pool's method if available
                if hasattr(self._api_key_pool, 'get_key_for_thread'):
                    try:
                        key_info = self._api_key_pool.get_key_for_thread(
                            force_rotation=should_rotate,
                            rotation_frequency=self._rotation_frequency
                        )
                        if key_info:
                            key, key_index, key_id = key_info
                            # Convert to tuple format expected below
                            key_info = (key, key_index)
                    except Exception as e:
                        print(f"[Thread-{thread_name}] Error getting key from pool: {e}")
                        key_info = None
                
                # Fallback to our method with timeout check
                if not key_info:
                    if time.time() - start_time > max_wait:
                        raise UnifiedClientError(f"Timeout getting key for thread after {max_wait}s", error_type="timeout")
                    key_info = self._get_next_available_key_for_thread()
                
                if key_info:
                    key, key_index = key_info[:2]  # Handle both tuple formats
                    
                    # Generate key identifier
                    key_id = f"Key#{key_index+1} ({key.model})"
                    if hasattr(key, 'identifier') and key.identifier:
                        key_id = key.identifier
                    
                    # Update thread-local state (no lock needed, thread-local is safe)
                    tls.api_key = key.api_key
                    tls.model = key.model
                    tls.key_index = key_index
                    tls.key_identifier = key_id
                    tls.google_credentials = getattr(key, 'google_credentials', None)
                    tls.azure_endpoint = getattr(key, 'azure_endpoint', None)
                    tls.azure_api_version = getattr(key, 'azure_api_version', None)
                    tls.google_region = getattr(key, 'google_region', None)
                    tls.use_individual_endpoint = getattr(key, 'use_individual_endpoint', False)
                    tls.initialized = True
                    tls.last_rotation = time.time()
                    
                    # MICROSECOND LOCK: Only when copying to instance variables
                    with self._model_lock:
                        # Copy to instance for compatibility
                        self.api_key = tls.api_key
                        self.model = tls.model
                        self.key_identifier = tls.key_identifier
                        self.current_key_index = key_index
                        self.current_key_google_creds = tls.google_credentials
                        self.current_key_azure_endpoint = tls.azure_endpoint
                        self.current_key_azure_api_version = tls.azure_api_version
                        self.current_key_google_region = tls.google_region
                        self.current_key_use_individual_endpoint = tls.use_individual_endpoint
                    
                    # Log key assignment - FIX: Add None check for api_key
                    if self.api_key and len(self.api_key) > 12:
                        masked_key = self.api_key[:4] + "..." + self.api_key[-4:]
                    elif self.api_key and len(self.api_key) > 5:
                        masked_key = self.api_key[:3] + "..." + self.api_key[-2:]
                    else:
                        masked_key = "***"
                    
                    print(f"[Thread-{thread_name}] 🔑 Using {self.key_identifier} - {masked_key}")
                    
                    # Setup client with new key (might need lock if it modifies instance state)
                    self._setup_client()
                    
                    # CRITICAL FIX: Apply individual key's Azure endpoint like single-key mode does
                    self._apply_individual_key_endpoint_if_needed()
                    return
                else:
                    # No keys available
                    raise UnifiedClientError("No available API keys for thread", error_type="no_keys")
            else:
                # Not rotating, ensure instance variables match thread-local
                if tls.initialized:
                    # MICROSECOND LOCK: When syncing instance variables
                    with self._model_lock:
                        self.api_key = tls.api_key
                        self.model = tls.model
                        self.key_identifier = tls.key_identifier
                        self.current_key_index = getattr(tls, 'key_index', None)
                        self.current_key_google_creds = getattr(tls, 'google_credentials', None)
                        self.current_key_azure_endpoint = getattr(tls, 'azure_endpoint', None)
                        self.current_key_azure_api_version = getattr(tls, 'azure_api_version', None)
                        self.current_key_google_region = getattr(tls, 'google_region', None)
                        self.current_key_use_individual_endpoint = getattr(tls, 'use_individual_endpoint', False)
        
        # Single key mode
        elif not tls.initialized:
            tls.api_key = self.original_api_key
            tls.model = self.original_model
            tls.key_identifier = "Single Key"
            tls.initialized = True
            tls.request_count = 0
            
            # MICROSECOND LOCK: When setting instance variables
            with self._model_lock:
                self.api_key = tls.api_key
                self.model = tls.model
                self.key_identifier = tls.key_identifier
            
            logger.debug(f"[Thread-{thread_name}] Single-key mode: Using {self.model}")
            self._setup_client()

    def _get_thread_key(self) -> Optional[Tuple[str, int]]:
        """Get the API key assigned to current thread"""
        thread_id = threading.current_thread().ident
        
        with self._assignment_lock:
            if thread_id in self._key_assignments:
                return self._key_assignments[thread_id]
        
        return None

    def _assign_thread_key(self):
        """Assign a key to the current thread"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name

        # Check if cancelled at start
        if self._cancelled:
            raise UnifiedClientError("Operation cancelled", error_type="cancelled")
            
        # Check if thread already has a key
        existing = self._get_thread_key()
        if existing and not self._should_rotate_thread_key():
            # Thread already has a key and doesn't need rotation
            key_index, key_identifier = existing
            self.current_key_index = key_index
            self.key_identifier = key_identifier
            
            # Apply the key settings
            if key_index < len(self._api_key_pool.keys):
                key = self._api_key_pool.keys[key_index]
                self.api_key = key.api_key
                self.model = key.model
            return
        
        # Get next available key for this thread
        max_retries = self._get_max_retries()
        retry_count = 0
        
        while retry_count <= max_retries:
            with self._pool_lock:
                key_info = self._get_next_available_key_for_thread()
                if key_info:
                    key, key_index = key_info
                    self.api_key = key.api_key
                    self.model = key.model
                    self.current_key_index = key_index
                    self.key_identifier = f"Key#{key_index+1} ({self.model})"
                    
                    # Store assignment
                    with self._assignment_lock:
                        self._key_assignments[thread_id] = (key_index, self.key_identifier)
                    
                    # FIX: Add None check for api_key
                    if self.api_key and len(self.api_key) > 12:
                        masked_key = self.api_key[:8] + "..." + self.api_key[-4:]
                    else:
                        masked_key = self.api_key or "***"
                    print(f"[THREAD-{thread_name}] 🔑 Assigned {self.key_identifier} - {masked_key}")
                    
                    # Setup client for this key
                    self._setup_client()
                    self._apply_custom_endpoint_if_needed()
                    print(f"[THREAD-{thread_name}] 🔄 Key assignment: Client setup completed, ready for requests...")
                    time.sleep(0.1)  # Brief pause after key assignment for stability
                    return
            
            # No key available - all are on cooldown
            if retry_count < max_retries:
                wait_time = self._get_shortest_cooldown_time()
                print(f"[THREAD-{thread_name}] No keys available, waiting {wait_time}s (retry {retry_count + 1}/{max_retries})")
                
                # Wait with cancellation check
                for i in range(wait_time):
                    if hasattr(self, '_cancelled') and self._cancelled:
                        raise UnifiedClientError("Operation cancelled while waiting for key", error_type="cancelled")
                    time.sleep(1)
                    if i % 10 == 0 and i > 0:
                        print(f"[THREAD-{thread_name}] Still waiting... {wait_time - i}s remaining")
                
                # Clear expired entries before next attempt
                if hasattr(self, '_rate_limit_cache') and self._rate_limit_cache:
                    self._rate_limit_cache.clear_expired()
                print(f"[THREAD-{thread_name}] 🔄 Cooldown wait: Cache cleared, attempting next key assignment...")
                time.sleep(0.1)  # Brief pause after cooldown wait for retry stability
            
            retry_count += 1
        
        # If we've exhausted all retries, raise error
        raise UnifiedClientError(f"No available API keys for thread after {max_retries} retries", error_type="no_keys")

    def _get_next_available_key_for_thread(self) -> Optional[Tuple]:
        """Get next available key for thread assignment with proper thread safety"""
        if not self._api_key_pool:
            return None
        
        thread_name = threading.current_thread().name
        
        # Stop check
        if self._cancelled:
            raise UnifiedClientError("Operation cancelled", error_type="cancelled")
        
        # Use the APIKeyPool's built-in thread-safe method
        if hasattr(self._api_key_pool, 'get_key_for_thread'):
            # Let the pool handle all the thread assignment logic
            key_info = self._api_key_pool.get_key_for_thread(
                force_rotation=getattr(self, '_force_rotation', True),
                rotation_frequency=getattr(self, '_rotation_frequency', 1)
            )
            
            if key_info:
                key, key_index, key_id = key_info
                print(f"[{thread_name}] Got {key_id} from pool")
                return (key, key_index)
            else:
                # Pool couldn't provide a key, all are on cooldown
                print(f"[{thread_name}] No keys available from pool")
                return None
        
        # Fallback: If pool doesn't have the method, use simpler logic
        print("APIKeyPool missing get_key_for_thread method, using fallback")
        
        with self.__class__._pool_lock:
            # Simple round-robin without complex thread tracking
            for _ in range(len(self._api_key_pool.keys)):
                current_idx = getattr(self._api_key_pool, 'current_index', 0)
                
                # Ensure index is valid
                if current_idx >= len(self._api_key_pool.keys):
                    current_idx = 0
                    self._api_key_pool.current_index = 0
                
                key = self._api_key_pool.keys[current_idx]
                key_id = f"Key#{current_idx+1} ({key.model})"
                
                # Advance index for next call
                self._api_key_pool.current_index = (current_idx + 1) % len(self._api_key_pool.keys)
                
                # Check availability
                if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                    print(f"[{thread_name}] Assigned {key_id} (fallback)")
                    return (key, current_idx)
            
            # No available keys
            print(f"[{thread_name}] All keys unavailable in fallback")
            return None

    def _wait_for_available_key(self) -> Optional[Tuple]:
        """Wait for a key to become available (called outside lock)"""
        thread_name = threading.current_thread().name
        
        # Check if cancelled first
        if self._cancelled:
            if not self._is_stop_requested():
                logger.info(f"[Thread-{thread_name}] Operation cancelled, not waiting for key")
            return None
        
        # Get shortest cooldown time with timeout protection
        wait_time = self._get_shortest_cooldown_time()
        
        # Cap maximum wait time to prevent infinite waits
        max_wait_time = 120  # 2 minutes max
        if wait_time > max_wait_time:
            print(f"[Thread-{thread_name}] Cooldown time {wait_time}s exceeds max {max_wait_time}s")
            wait_time = max_wait_time
        
        if wait_time <= 0:
            # Keys should be available now
            with self.__class__._pool_lock:
                for i, key in enumerate(self._api_key_pool.keys):
                    key_id = f"Key#{i+1} ({key.model})"
                    if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                        return (key, i)
        
        print(f"[Thread-{thread_name}] All keys on cooldown. Waiting {wait_time}s...")
        
        # Wait with cancellation check
        wait_start = time.time()
        while time.time() - wait_start < wait_time:
            if self._cancelled:
                print(f"[Thread-{thread_name}] Wait cancelled by user")
                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
            
            # Check every second if a key became available early
            with self.__class__._pool_lock:
                for i, key in enumerate(self._api_key_pool.keys):
                    key_id = f"Key#{i+1} ({key.model})"
                    if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                        print(f"[Thread-{thread_name}] Key became available early: {key_id}")
                        print(f"[Thread-{thread_name}] 🔄 Early key availability: Key ready for immediate use...")
                        time.sleep(0.1)  # Brief pause after early detection for stability
                        return (key, i)
            
            time.sleep(1)
            
            # Progress indicator
            elapsed = int(time.time() - wait_start)
            if elapsed % 10 == 0 and elapsed > 0:
                remaining = wait_time - elapsed
                print(f"[Thread-{thread_name}] Still waiting... {remaining}s remaining")
        
        # Clear expired entries from cache
        self._rate_limit_cache.clear_expired()
        
        # Final attempt after wait
        with self.__class__._pool_lock:
            # Try to find an available key
            for i, key in enumerate(self._api_key_pool.keys):
                key_id = f"Key#{i+1} ({key.model})"
                if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                    return (key, i)
            
            # Still no keys? Return the first enabled one (last resort)
            for i, key in enumerate(self._api_key_pool.keys):
                if key.enabled:
                    print(f"[Thread-{thread_name}] WARNING: Using potentially rate-limited key as last resort")
                    return (key, i)
        
        return None

    def _should_rotate_thread_key(self) -> bool:
        """Check if current thread should rotate its key"""
        if not self._force_rotation:
            return False
        
        # Check thread-local request count
        if not hasattr(self._thread_local, 'request_count'):
            self._thread_local.request_count = 0
        
        self._thread_local.request_count += 1
        
        if self._thread_local.request_count >= self._rotation_frequency:
            self._thread_local.request_count = 0
            return True
        
        return False

    def _handle_rate_limit_for_thread(self):
        """Handle rate limit by marking current thread's key and getting a new one (thread-safe)"""
        if not self._multi_key_mode:  # Check INSTANCE variable
            return
            
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        # Get thread-local state first (thread-safe by nature)
        tls = self._get_thread_local_client()
        
        # Store the current key info before we change anything
        current_key_index = None
        current_key_identifier = None
        
        # Safely get current key information from thread-local storage
        if hasattr(tls, 'key_index') and tls.key_index is not None:
            current_key_index = tls.key_index
            current_key_identifier = getattr(tls, 'key_identifier', f"Key#{current_key_index+1}")
        elif hasattr(self, 'current_key_index') and self.current_key_index is not None:
            # Fallback to instance variable if thread-local not set
            current_key_index = self.current_key_index
            current_key_identifier = self.key_identifier
        
        # Mark the current key as rate limited (if we have one)
        if current_key_index is not None and self._api_key_pool:
            # Use the pool's thread-safe method to mark the error
            self._api_key_pool.mark_key_error(current_key_index, 429)
            
            # Get cooldown value safely
            cooldown = 60  # Default
            with self.__class__._pool_lock:
                if current_key_index < len(self._api_key_pool.keys):
                    key = self._api_key_pool.keys[current_key_index]
                    cooldown = getattr(key, 'cooldown', 60)
            
            print(f"[THREAD-{thread_name}] 🕐 Marking {current_key_identifier} for cooldown ({cooldown}s)")
            
            # Add to rate limit cache (this is already thread-safe)
            if hasattr(self.__class__, '_rate_limit_cache') and self.__class__._rate_limit_cache:
                self.__class__._rate_limit_cache.add_rate_limit(current_key_identifier, cooldown)
        
        # Clear thread-local state to force new key assignment
        tls.initialized = False
        tls.api_key = None
        tls.model = None
        tls.key_index = None
        tls.key_identifier = None
        tls.request_count = 0
        
        # Remove any legacy assignments (thread-safe with lock)
        if hasattr(self, '_assignment_lock') and hasattr(self, '_key_assignments'):
            with self._assignment_lock:
                if thread_id in self._key_assignments:
                    del self._key_assignments[thread_id]
        
        # Release thread assignment in the pool (if pool supports it)
        if hasattr(self._api_key_pool, 'release_thread_assignment'):
            self._api_key_pool.release_thread_assignment(thread_id)
        
        # Now force getting a new key
        # This will call _ensure_thread_client which will get a new key
        print(f"[THREAD-{thread_name}] 🔄 Requesting new key after rate limit...")
        
        try:
            # Ensure we get a new client with a new key
            self._ensure_thread_client()
            
            # Verify we got a different key
            new_key_index = getattr(tls, 'key_index', None)
            new_key_identifier = getattr(tls, 'key_identifier', 'Unknown')
            
            if new_key_index != current_key_index:
                print(f"[THREAD-{thread_name}] ✅ Successfully rotated from {current_key_identifier} to {new_key_identifier}")
            else:
                print(f"[THREAD-{thread_name}] ⚠️ Warning: Got same key back: {new_key_identifier}")
                
        except Exception as e:
            print(f"[THREAD-{thread_name}] ❌ Failed to get new key after rate limit: {e}")
            raise UnifiedClientError(f"Failed to rotate key after rate limit: {e}", error_type="no_keys")
    
    # Helper methods that need to check instance state
    def _count_available_keys(self) -> int:
        """Count how many keys are currently available"""
        if not self._multi_key_mode or not self.__class__._api_key_pool:
            return 0
        
        count = 0
        for i, key in enumerate(self.__class__._api_key_pool.keys):
            if key.enabled:
                key_id = f"Key#{i+1} ({key.model})"
                # Check both rate limit cache AND key's own cooling status
                is_rate_limited = self.__class__._rate_limit_cache.is_rate_limited(key_id)
                is_cooling = key.is_cooling_down  # Also check the key's own status
                
                if not is_rate_limited and not is_cooling:
                    count += 1
        return count
        
    def _mark_key_success(self):
        """Mark the current key as successful (thread-safe)"""
        # Check both instance and class-level cancellation
        if (hasattr(self, '_cancelled') and self._cancelled) or self.__class__._global_cancelled:
            # Don't mark success if we're cancelled
            return
            
        if not self._multi_key_mode:
            return
        
        # Get thread-local state
        tls = self._get_thread_local_client()
        key_index = getattr(tls, 'key_index', None)
        
        # Fallback to instance variable if thread-local not set
        if key_index is None:
            key_index = getattr(self, 'current_key_index', None)
        
        if key_index is not None and self.__class__._api_key_pool:
            # Use the pool's thread-safe method
            self.__class__._api_key_pool.mark_key_success(key_index)
    
    def _mark_key_error(self, error_code: int = None):
        """Mark current key as having an error and apply cooldown if rate limited (thread-safe)"""
        # Check both instance and class-level cancellation
        if (hasattr(self, '_cancelled') and self._cancelled) or self.__class__._global_cancelled:
            # Don't mark error if we're cancelled
            return
            
        if not self._multi_key_mode:
            return
        
        # Get thread-local state
        tls = self._get_thread_local_client()
        key_index = getattr(tls, 'key_index', None)
        
        # Fallback to instance variable if thread-local not set
        if key_index is None:
            key_index = getattr(self, 'current_key_index', None)
        
        if key_index is not None and self.__class__._api_key_pool:
            # Use the pool's thread-safe method
            self.__class__._api_key_pool.mark_key_error(key_index, error_code)
            
            # If it's a rate limit error, also add to rate limit cache
            if error_code == 429:
                # Get key identifier safely
                with self.__class__._pool_lock:
                    if key_index < len(self.__class__._api_key_pool.keys):
                        key = self.__class__._api_key_pool.keys[key_index]
                        key_id = f"Key#{key_index+1} ({key.model})"
                        cooldown = getattr(key, 'cooldown', 60)
                        
                        # Add to rate limit cache (already thread-safe)
                        if hasattr(self.__class__, '_rate_limit_cache'):
                            self.__class__._rate_limit_cache.add_rate_limit(key_id, cooldown)
    
    def _apply_custom_endpoint_if_needed(self):
        """Apply custom endpoint configuration if needed"""
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
        
        if custom_base_url and use_custom_endpoint:
            if not custom_base_url.startswith(('http://', 'https://')):
                custom_base_url = 'https://' + custom_base_url
            
            # Don't override Gemini models - they have their own separate endpoint toggle
            if self.client_type == 'gemini':
                # Only log if Gemini OpenAI endpoint is not also enabled
                use_gemini_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
                if not use_gemini_endpoint:
                    self._log_once("Gemini model detected, not overriding with custom OpenAI endpoint (use USE_GEMINI_OPENAI_ENDPOINT instead)", is_debug=True)
                return
            
            # Override other model types to use OpenAI client when custom endpoint is enabled
            original_client_type = self.client_type
            self.client_type = 'openai'
            
            try:
                import openai
                # MICROSECOND LOCK: Create custom endpoint client with thread safety
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=custom_base_url
                    )
            except ImportError:
                print(f"[ERROR] OpenAI library not installed, cannot use custom endpoint")
                self.client_type = original_client_type  # Restore original type
    
    def _apply_individual_key_endpoint_if_needed(self):
        """Apply individual key endpoint if configured (multi-key mode) - works independently of global toggle"""
        # Check if this key has an individual endpoint enabled AND configured
        has_individual_endpoint = (hasattr(self, 'current_key_azure_endpoint') and 
                                 hasattr(self, 'current_key_use_individual_endpoint') and 
                                 self.current_key_use_individual_endpoint and 
                                 self.current_key_azure_endpoint)
        
        if has_individual_endpoint:
            # Use individual endpoint - works independently of global custom endpoint toggle
            individual_endpoint = self.current_key_azure_endpoint
            
            if not individual_endpoint.startswith(('http://', 'https://')):
                individual_endpoint = 'https://' + individual_endpoint
            
            # Don't override Gemini models - they have their own separate endpoint toggle
            if self.client_type == 'gemini':
                # Only log if Gemini OpenAI endpoint is not also enabled
                use_gemini_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
                if not use_gemini_endpoint:
                    self._log_once("Gemini model detected, not overriding with individual endpoint (use USE_GEMINI_OPENAI_ENDPOINT instead)", is_debug=True)
                return
            
            # Detect Azure endpoints and route via Azure handler instead of generic OpenAI base_url
            url_l = individual_endpoint.lower()
            is_azure = (".openai.azure.com" in url_l) or (".cognitiveservices" in url_l) or ("/openai/deployments/" in url_l)
            if is_azure:
                # Normalize to plain Azure base (strip any trailing /openai/... if present)
                azure_base = individual_endpoint.split('/openai')[0] if '/openai' in individual_endpoint else individual_endpoint.rstrip('/')
                with self._model_lock:
                    # Switch this instance to Azure mode for correct routing
                    self.client_type = 'azure'
                    self.azure_endpoint = azure_base
                    # Prefer per-key Azure API version if available
                    self.azure_api_version = getattr(self, 'current_key_azure_api_version', None) or os.getenv('AZURE_API_VERSION', '2024-02-01')
                    # Mark that we applied an individual (per-key) endpoint
                    self._individual_endpoint_applied = True
                # Also update TLS so subsequent calls on this thread know it's Azure
                try:
                    tls = self._get_thread_local_client()
                    with self._model_lock:
                        tls.azure_endpoint = azure_base
                        tls.azure_api_version = self.azure_api_version
                        tls.client_type = 'azure'
                except Exception:
                    pass
                print(f"[DEBUG] Individual Azure endpoint applied: {azure_base} (api-version={self.azure_api_version})")
                return  # Handled; do not fall through to custom endpoint logic
            
            # Non-Azure: Override to use OpenAI-compatible client against the provided base URL
            original_client_type = self.client_type
            self.client_type = 'openai'
            
            try:
                import openai
                
                # MICROSECOND LOCK: Create individual endpoint client with thread safety
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=individual_endpoint
                    )
                
                # Set flag to prevent _setup_client from overriding this client
                self._individual_endpoint_applied = True
                
                # CRITICAL: Update thread-local storage with our correct client
                tls = self._get_thread_local_client()
                with self._model_lock:
                    tls.openai_client = self.openai_client
                    tls.client_type = 'openai'
                
                return  # Individual endpoint applied - don't check global custom endpoint
            except ImportError:
                self.client_type = original_client_type  # Restore original type
                return
            except Exception as e:
                print(f"[ERROR] Failed to create individual endpoint client: {e}")
                self.client_type = original_client_type  # Restore original type
                return
        
        # If no individual endpoint, check global custom endpoint (but only if global toggle is enabled)
        self._apply_custom_endpoint_if_needed()
    
    # Properties for backward compatibility
    @property
    def use_multi_keys(self):
        """Property for backward compatibility"""
        return self._multi_key_mode
    
    @use_multi_keys.setter
    def use_multi_keys(self, value):
        """Property setter for backward compatibility"""
        self._multi_key_mode = value

    def _ensure_key_rotation(self):
        """Ensure we have a key selected and rotate if in multi-key mode"""
        if not self.use_multi_keys:
            return
        
        # Force rotation to next key on every request
        if self.current_key_index is not None:
            # We already have a key, rotate to next
            print(f"[DEBUG] Rotating from {self.key_identifier} to next key")
            self._force_next_key()
        else:
            # First request, get initial key
            print(f"[DEBUG] First request, selecting initial key")
            key_info = self._get_next_available_key()
            if key_info:
                self._apply_key_change(key_info, "Initial")
            else:
                raise UnifiedClientError("No available API keys", error_type="no_keys")

    def _force_next_key(self):
        """Force rotation to the next key in the pool"""
        if not self.use_multi_keys or not self._api_key_pool:
            return
        
        old_key_identifier = self.key_identifier
        
        # Use force_rotate method to always get next key
        key_info = self._api_key_pool.force_rotate_to_next_key()
        if key_info:
            # Check if it's available
            if not key_info[0].is_available():
                print(f"[WARNING] Next key in rotation is on cooldown, but using it anyway")
            
            self._apply_key_change(key_info, old_key_identifier)
            print(f"🔄 Force key rotation: Key change completed, system ready...")
            time.sleep(0.1)  # Brief pause after force rotation for system stability
        else:
            print(f"[ERROR] Failed to rotate to next key")
    
    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available key and reinitialize client - THREAD SAFE"""
        if not self.use_multi_keys or not self._api_key_pool:
            return False
        
        old_key_identifier = self.key_identifier
        
        key_info = self._get_next_available_key()
        if key_info:
            # MICROSECOND LOCK: Protect all instance variable modifications
            with self._model_lock:
                # Update key and model
                self.api_key = key_info[0].api_key
                self.model = key_info[0].model
                self.current_key_index = key_info[1]
                
                # Update key identifier
                self.key_identifier = f"Key#{key_info[1]+1} ({self.model})"
                
                # Reset clients (these are instance variables too!)
                self.openai_client = None
                self.gemini_client = None
                self.mistral_client = None
                self.cohere_client = None
            
            # Logging (outside lock - just reading) - FIX: Add None check
            if self.api_key and len(self.api_key) > 12:
                masked_key = self.api_key[:8] + "..." + self.api_key[-4:]
            else:
                masked_key = self.api_key or "***"
            print(f"[DEBUG] 🔄 Rotating from {old_key_identifier} to {self.key_identifier} - {masked_key}")
            
            # Re-setup the client with new key
            self._setup_client()
            
            # Re-apply individual endpoint if needed (this takes priority over global custom endpoint)
            self._apply_individual_key_endpoint_if_needed()
            
            print(f"🔄 Key rotation: Endpoint setup completed, rotation successful...")
            time.sleep(0.1)  # Brief pause after rotation for system stability
            return True
        
        print(f"[WARNING] No available keys to rotate to")
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about API usage"""
        stats = dict(self.stats)
        
        # Add multi-key stats if in multi-key mode
        if self._multi_key_mode:  # Use instance variable
            stats['multi_key_enabled'] = True
            stats['force_rotation'] = self._force_rotation  # Use instance variable
            stats['rotation_frequency'] = self._rotation_frequency  # Use instance variable
            
            if hasattr(self, '_api_key_pool') and self._api_key_pool:
                stats['total_keys'] = len(self._api_key_pool.keys)
                stats['active_keys'] = sum(1 for k in self._api_key_pool.keys if k.enabled and k.is_available())
                stats['keys_on_cooldown'] = sum(1 for k in self._api_key_pool.keys if k.is_cooling_down)
                
                # Per-key stats
                key_stats = []
                for i, key in enumerate(self._api_key_pool.keys):
                    key_stat = {
                        'index': i,
                        'model': key.model,
                        'enabled': key.enabled,
                        'available': key.is_available(),
                        'success_count': key.success_count,
                        'error_count': key.error_count,
                        'cooling_down': key.is_cooling_down
                    }
                    key_stats.append(key_stat)
                stats['key_details'] = key_stats
        else:
            stats['multi_key_enabled'] = False
        
        return stats
    
    def diagnose_custom_endpoint(self) -> Dict[str, Any]:
        """Diagnose custom endpoint configuration for troubleshooting"""
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'client_type': getattr(self, 'client_type', None),
            'multi_key_mode': getattr(self, '_multi_key_mode', False),
            'environment_variables': {
                'USE_CUSTOM_OPENAI_ENDPOINT': os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', 'not_set'),
                'OPENAI_CUSTOM_BASE_URL': os.getenv('OPENAI_CUSTOM_BASE_URL', 'not_set'),
                'OPENAI_API_BASE': os.getenv('OPENAI_API_BASE', 'not_set'),
            },
            'client_status': {
                'openai_client_exists': hasattr(self, 'openai_client') and self.openai_client is not None,
                'gemini_client_exists': hasattr(self, 'gemini_client') and self.gemini_client is not None,
                'current_api_key_length': len(self.api_key) if hasattr(self, 'api_key') and self.api_key else 0,
            }
        }
        
        # Check if custom endpoint should be applied
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
        
        diagnosis['custom_endpoint_analysis'] = {
            'toggle_enabled': use_custom_endpoint,
            'custom_url_provided': bool(custom_base_url),
            'should_use_custom_endpoint': use_custom_endpoint and bool(custom_base_url),
            'would_override_model_type': True,  # With our fix, it always overrides now
        }
        
        # Determine if there are any issues
        issues = []
        if use_custom_endpoint and not custom_base_url:
            issues.append("Custom endpoint enabled but no URL provided in OPENAI_CUSTOM_BASE_URL")
        if custom_base_url and not use_custom_endpoint:
            issues.append("Custom URL provided but toggle USE_CUSTOM_OPENAI_ENDPOINT is disabled")
        if not openai and use_custom_endpoint:
            issues.append("OpenAI library not installed - cannot use custom endpoints")
            
        diagnosis['issues'] = issues
        diagnosis['status'] = 'OK' if not issues else 'ISSUES_FOUND'
        
        return diagnosis
    
    def print_custom_endpoint_diagnosis(self):
        """Print a user-friendly diagnosis of custom endpoint configuration"""
        diagnosis = self.diagnose_custom_endpoint()
        
        print("\n🔍 Custom OpenAI Endpoint Diagnosis:")
        print(f"   Model: {diagnosis['model']}")
        print(f"   Client Type: {diagnosis['client_type']}")
        print(f"   Multi-Key Mode: {diagnosis['multi_key_mode']}")
        print("\n📋 Environment Variables:")
        for key, value in diagnosis['environment_variables'].items():
            print(f"   {key}: {value}")
        
        print("\n🔧 Custom Endpoint Analysis:")
        analysis = diagnosis['custom_endpoint_analysis']
        print(f"   Toggle Enabled: {analysis['toggle_enabled']}")
        print(f"   Custom URL Provided: {analysis['custom_url_provided']}")
        print(f"   Should Use Custom Endpoint: {analysis['should_use_custom_endpoint']}")
        
        if diagnosis['issues']:
            print("\n⚠️  Issues Found:")
            for issue in diagnosis['issues']:
                print(f"   • {issue}")
        else:
            print("\n✅ No configuration issues detected")
            
        print(f"\n📊 Status: {diagnosis['status']}\n")
        
        return diagnosis
    
    def reset_stats(self):
        """Reset usage statistics and pattern tracking"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'errors': defaultdict(int),
            'response_times': [],
            'empty_results': 0
        }
        
        # Reset pattern tracking
        self.pattern_counts = {}
        self.last_pattern = None
        
        # Reset conversation tracking if not already set
        if not hasattr(self, 'conversation_message_count'):
            self.conversation_message_count = 0
        
        # Log if logger is available
        if hasattr(self, 'logger'):
            self.logger.info("Statistics and pattern tracking reset")
        else:
            print("Statistics and pattern tracking reset")
    
    def _rotate_to_next_available_key(self, skip_current: bool = False) -> bool:
        """
        Rotate to the next available key that's not rate limited
        
        Args:
            skip_current: If True, skip the current key even if it becomes available
        """
        if not self._multi_key_mode or not self._api_key_pool:  # Use instance variable
            return False
        
        old_key_identifier = self.key_identifier
        start_index = self._api_key_pool.current_index
        max_attempts = len(self._api_key_pool.keys)
        attempts = 0
        
        while attempts < max_attempts:
            # Get next key from pool
            key_info = self._get_next_available_key()
            if not key_info:
                attempts += 1
                continue
            
            # Check if this is the same key we started with
            potential_key_id = f"Key#{key_info[1]+1} ({key_info[0].model})"
            if skip_current and potential_key_id == old_key_identifier:
                attempts += 1
                continue
            
            # Check if this key is rate limited
            if not self._rate_limit_cache.is_rate_limited(potential_key_id):
                # This key is available, use it
                self._apply_key_change(key_info, old_key_identifier)
                return True
            else:
                print(f"[DEBUG] Skipping {potential_key_id} (in cooldown)")
            
            attempts += 1
        
        print(f"[DEBUG] No available keys found after checking all {max_attempts} keys")
        
        # All keys are on cooldown - wait for shortest cooldown
        wait_time = self._get_shortest_cooldown_time()
        print(f"[DEBUG] All keys on cooldown. Waiting {wait_time}s...")
        
        # Wait with cancellation check
        for i in range(wait_time):
            if hasattr(self, '_cancelled') and self._cancelled:
                print(f"[DEBUG] Wait cancelled by user")
                return False
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"[DEBUG] Still waiting... {wait_time - i}s remaining")
        
        # Clear expired entries and try again
        self._rate_limit_cache.clear_expired()
        
        # Try one more time to find an available key
        attempts = 0
        while attempts < max_attempts:
            key_info = self._get_next_available_key()
            if key_info:
                potential_key_id = f"Key#{key_info[1]+1} ({key_info[0].model})"
                if not self._rate_limit_cache.is_rate_limited(potential_key_id):
                    self._apply_key_change(key_info, old_key_identifier)
                    return True
            attempts += 1
        
        return False
    
    def _apply_key_change(self, key_info: tuple, old_key_identifier: str):
        """Apply the key change and reinitialize clients"""
        self.api_key = key_info[0].api_key
        self.model = key_info[0].model
        self.current_key_index = key_info[1]
        self.key_identifier = f"Key#{key_info[1]+1} ({key_info[0].model})"
        
        # MICROSECOND LOCK: Atomic update of all key-related variables
        with self._model_lock:
            self.api_key = key_info[0].api_key
            self.model = key_info[0].model
            self.current_key_index = key_info[1]
            self.key_identifier = f"Key#{key_info[1]+1} ({key_info[0].model})"
            
            # Reset clients atomically
            self.openai_client = None
            self.gemini_client = None
            self.mistral_client = None
            self.cohere_client = None
        
        # Logging OUTSIDE the lock - FIX: Add None check
        if self.api_key and len(self.api_key) > 8:
            masked_key = self.api_key[:8] + "..." + self.api_key[-4:]
        else:
            masked_key = self.api_key or "***"
        print(f"[DEBUG] 🔄 Switched from {old_key_identifier} to {self.key_identifier}")
        
        # Reset clients
        self.openai_client = None
        self.gemini_client = None
        self.mistral_client = None
        self.cohere_client = None
        
        # Re-setup the client with new key
        self._setup_client()
        
        # Re-apply custom endpoint if needed
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
        
        if custom_base_url and use_custom_endpoint and self.client_type == 'openai':
            if not custom_base_url.startswith(('http://', 'https://')):
                custom_base_url = 'https://' + custom_base_url
            
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=custom_base_url
            )
            print(f"[DEBUG] Re-created OpenAI client with custom base URL")
    
    def _force_rotate_to_untried_key(self, attempted_keys: set) -> bool:
        """
        Force rotation to any key that hasn't been tried yet, ignoring cooldown
        
        Args:
            attempted_keys: Set of key identifiers that have already been attempted
        """
        if not self._multi_key_mode or not self._api_key_pool:  # Use instance variable
            return False
        
        old_key_identifier = self.key_identifier
        
        # Try each key in the pool
        for i in range(len(self._api_key_pool.keys)):
            key = self._api_key_pool.keys[i]
            potential_key_id = f"Key#{i+1} ({key.model})"
            
            # Skip if already tried
            if potential_key_id in attempted_keys:
                continue
            
            # Found an untried key - use it regardless of cooldown
            key_info = (key, i)
            self._apply_key_change(key_info, old_key_identifier)
            print(f"[DEBUG] 🔄 Force-rotated to untried key: {self.key_identifier}")
            return True
        
        return False
    
    def get_current_key_info(self) -> str:
        """Get information about the currently active key"""
        if self._multi_key_mode and self.current_key_index is not None:  # Use instance variable
            key = self._api_key_pool.keys[self.current_key_index]
            status = "Active" if key.is_available() else "Cooling Down"
            return f"{self.key_identifier} - Status: {status}, Success: {key.success_count}, Errors: {key.error_count}"
        else:
            return "Single Key Mode"

    def _generate_unique_thread_dir(self, context: str) -> str:
        """Generate a truly unique thread directory with session ID and timestamp"""
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        # Include timestamp and session ID for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        unique_id = f"{thread_name}_{thread_id}_{self.session_id}_{timestamp}"
        
        thread_dir = os.path.join("Payloads", context, unique_id)
        os.makedirs(thread_dir, exist_ok=True)
        return thread_dir

    def _get_request_hash(self, messages) -> str:
        """Generate a STABLE hash for request deduplication - THREAD-SAFE VERSION
        WITH MICROSECOND LOCKING for thread safety."""
        
        # MICROSECOND LOCK: Ensure atomic hash generation
        with self._model_lock:
            # Get thread-specific identifier to prevent cross-thread cache collisions
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            
            # REMOVED: request_uuid, request_timestamp, request_timestamp_micro
            # We want STABLE hashes for caching to work!
        
        # Create normalized representation (can be done outside lock)
        normalized_messages = []
        
        for msg in messages:
            normalized_msg = {
                'role': msg.get('role', ''),
                'content': msg.get('content', '')
            }
            
            # For image messages, include image size/hash instead of full data
            if isinstance(normalized_msg['content'], list):
                content_parts = []
                for part in normalized_msg['content']:
                    if isinstance(part, dict) and 'image_url' in part:
                        # Hash the image data
                        image_data = part.get('image_url', {}).get('url', '')
                        if image_data.startswith('data:'):
                            # Extract just the data part
                            image_hash = hashlib.md5(image_data.encode()).hexdigest()
                            content_parts.append(f"image:{image_hash}")
                        else:
                            content_parts.append(f"image_url:{image_data}")
                    else:
                        content_parts.append(str(part))
                normalized_msg['content'] = '|'.join(content_parts)
            
            normalized_messages.append(normalized_msg)
        
        # MICROSECOND LOCK: Ensure atomic hash generation
        with self._model_lock:
            # Include thread_id but NO request-specific IDs for stable caching
            hash_data = {
                'thread_id': thread_id,  # THREAD ISOLATION
                'thread_name': thread_name,  # Additional context for debugging
                # REMOVED: request_uuid, request_time, request_time_ns
                'messages': normalized_messages,
                'model': self.model,
                'temperature': getattr(self, 'temperature', 0.3),
                'max_tokens': getattr(self, 'max_tokens', 8192)
            }
            
            # Debug logging if needed
            if os.getenv("DEBUG_HASH", "0") == "1":
                print(f"[HASH] Thread: {thread_name} (ID: {thread_id})")
                print(f"[HASH] Model: {self.model}")
            
            # Create stable JSON representation
            hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
            
            # Use SHA256 for better distribution
            final_hash = hashlib.sha256(hash_str.encode()).hexdigest()
        
        if os.getenv("DEBUG_HASH", "0") == "1":
            print(f"[HASH] Generated stable hash: {final_hash[:16]}...")
        
        return final_hash

    def _get_request_hash_with_context(self, messages, context=None) -> str:
        """
        Generate a STABLE hash that includes context AND thread info for better deduplication.
        WITH MICROSECOND LOCKING for thread safety.
        """
        
        # MICROSECOND LOCK: Ensure atomic reading of model/settings
        with self._model_lock:
            # Get thread-specific identifier
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            
            # REMOVED: request_uuid, request_timestamp, request_timestamp_micro
            # We want STABLE hashes for caching to work!
        
        # Create normalized representation (can be done outside lock)
        normalized_messages = []
        
        for msg in messages:
            normalized_msg = {
                'role': msg.get('role', ''),
                'content': msg.get('content', '')
            }
            
            # Handle image messages
            if isinstance(normalized_msg['content'], list):
                content_parts = []
                for part in normalized_msg['content']:
                    if isinstance(part, dict) and 'image_url' in part:
                        image_data = part.get('image_url', {}).get('url', '')
                        if image_data.startswith('data:'):
                            # Use first 1000 chars of image data for hash
                            image_sample = image_data[:1000]
                            image_hash = hashlib.md5(image_sample.encode()).hexdigest()
                            content_parts.append(f"image:{image_hash}")
                        else:
                            content_parts.append(f"image_url:{image_data}")
                    elif isinstance(part, dict):
                        content_parts.append(json.dumps(part, sort_keys=True))
                    else:
                        content_parts.append(str(part))
                normalized_msg['content'] = '|'.join(content_parts)
            
            normalized_messages.append(normalized_msg)
        
        # MICROSECOND LOCK: Ensure atomic hash generation
        with self._instance_model_lock:
            # Include context, thread info, but NO request-specific IDs
            hash_data = {
                'thread_id': thread_id,  # THREAD ISOLATION
                'thread_name': thread_name,  # Additional thread context
                # REMOVED: request_uuid, request_time, request_time_ns
                'context': context,  # Include context (e.g., 'translation', 'glossary', etc.)
                'messages': normalized_messages,
                'model': self.model,
                'temperature': getattr(self, 'temperature', 0.3),
                'max_tokens': getattr(self, 'max_tokens', 8192)
            }
            
            # Debug logging if needed
            if os.getenv("DEBUG_HASH", "0") == "1":
                print(f"[HASH_CONTEXT] Thread: {thread_name} (ID: {thread_id})")
                print(f"[HASH_CONTEXT] Context: {context}")
                print(f"[HASH_CONTEXT] Model: {self.model}")
            
            # Create stable JSON representation
            hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
            
            # Use SHA256 for better distribution
            final_hash = hashlib.sha256(hash_str.encode()).hexdigest()
        
        if os.getenv("DEBUG_HASH", "0") == "1":
            print(f"[HASH_CONTEXT] Generated stable hash: {final_hash[:16]}...")
        
        return final_hash

    def _get_unique_file_suffix(self, attempt: int = 0) -> str:
        """Generate a unique suffix for file names to prevent overwrites
        WITH MICROSECOND LOCKING for thread safety."""
        
        # MICROSECOND LOCK: Ensure atomic generation of unique identifiers
        with self._instance_model_lock:
            thread_id = threading.current_thread().ident
            timestamp = datetime.now().strftime("%H%M%S%f")[:10]
            request_uuid = str(uuid.uuid4())[:8]
            
            # Create unique suffix for files
            suffix = f"_T{thread_id}_A{attempt}_{timestamp}_{request_uuid}"
        
        return suffix

    def _get_request_hash_with_request_id(self, messages, request_id: str) -> str:
        """Generate hash WITH request ID for per-call caching
        WITH MICROSECOND LOCKING for thread safety."""
        
        # MICROSECOND LOCK: Ensure atomic hash generation
        with self._instance_model_lock:
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
        
        # Create normalized representation
        normalized_messages = []
        
        for msg in messages:
            normalized_msg = {
                'role': msg.get('role', ''),
                'content': msg.get('content', '')
            }
            
            # For image messages, include image size/hash instead of full data
            if isinstance(normalized_msg['content'], list):
                content_parts = []
                for part in normalized_msg['content']:
                    if isinstance(part, dict) and 'image_url' in part:
                        image_data = part.get('image_url', {}).get('url', '')
                        if image_data.startswith('data:'):
                            image_hash = hashlib.md5(image_data.encode()).hexdigest()
                            content_parts.append(f"image:{image_hash}")
                        else:
                            content_parts.append(f"image_url:{image_data}")
                    else:
                        content_parts.append(str(part))
                normalized_msg['content'] = '|'.join(content_parts)
            
            normalized_messages.append(normalized_msg)
        
        # MICROSECOND LOCK: Ensure atomic hash generation
        with self._instance_model_lock:
            hash_data = {
                'thread_id': thread_id,
                'thread_name': thread_name,
                'request_id': request_id,  # THIS MAKES EACH send() CALL UNIQUE
                'messages': normalized_messages,
                'model': self.model,
                'temperature': getattr(self, 'temperature', 0.3),
                'max_tokens': getattr(self, 'max_tokens', 8192)
            }
            
            if os.getenv("DEBUG_HASH", "0") == "1":
                print(f"[HASH] Thread: {thread_name} (ID: {thread_id})")
                print(f"[HASH] Request ID: {request_id}")  # Debug the request ID
                print(f"[HASH] Model: {self.model}")
            
            hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
            final_hash = hashlib.sha256(hash_str.encode()).hexdigest()
        
        if os.getenv("DEBUG_HASH", "0") == "1":
            print(f"[HASH] Generated hash for request {request_id}: {final_hash[:16]}...")
        
        return final_hash

    def _check_duplicate_request(self, request_hash: str, context: str) -> bool:
        """
        Enhanced duplicate detection that properly handles parallel requests.
        Returns True only if this exact request is actively being processed.
        """
        
        # Only check for duplicates in specific contexts
        if context not in ['translation', 'glossary', 'image_translation']:
            return False
        
        thread_name = threading.current_thread().name
        
        # This method is now deprecated in favor of the active_requests tracking
        # We keep it for backward compatibility but it just returns False
        # The real duplicate detection happens in the send() method using _active_requests
        return False

    def _debug_active_requests(self):
        """Debug method to show current active requests"""
        pass

    def _ensure_thread_safety_init(self):
        """
        Ensure all thread safety structures are properly initialized.
        Call this during __init__ or before parallel processing.
        """
        
        # Thread-local storage
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
        
        # File operation locks
        if not hasattr(self, '_file_write_locks'):
            self._file_write_locks = {}
        if not hasattr(self, '_file_write_locks_lock'):
            self._file_write_locks_lock = RLock()
        
        # Legacy tracker (for backward compatibility)
        if not hasattr(self, '_tracker_lock'):
            self._tracker_lock = RLock()

    def _periodic_cache_cleanup(self):
        """
        Periodically clean up expired cache entries and active requests.
        Should be called periodically or scheduled with a timer.
        """
        pass

    def _get_thread_status(self) -> dict:
        """
        Get current status of thread-related structures for debugging.
        """
        status = {
            'thread_name': threading.current_thread().name,
            'thread_id': threading.current_thread().ident,
            'cache_size': len(self._request_cache) if hasattr(self, '_request_cache') else 0,
            'active_requests': len(self._active_requests) if hasattr(self, '_active_requests') else 0,
            'multi_key_mode': self._multi_key_mode,
            'current_key': self.key_identifier if hasattr(self, 'key_identifier') else 'Unknown'
        }
        
        # Add thread-local info if available
        if hasattr(self, '_thread_local'):
            tls = self._get_thread_local_client()
            status['thread_local'] = {
                'initialized': getattr(tls, 'initialized', False),
                'key_index': getattr(tls, 'key_index', None),
                'request_count': getattr(tls, 'request_count', 0)
            }
        
        return status

    def cleanup(self):
        """
        Enhanced cleanup method to properly release all resources.
        Should be called when done with the client or on shutdown.
        """
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Cleaning up UnifiedClient resources")
        
        # Release thread key assignment if in multi-key mode
        if self._multi_key_mode and self._api_key_pool:
            thread_id = threading.current_thread().ident
            self._api_key_pool.release_thread_assignment(thread_id)      
        
        # Clear thread-local storage
        if hasattr(self, '_thread_local'):
            # Reset thread-local state
            self._thread_local.initialized = False
            self._thread_local.api_key = None
            self._thread_local.model = None
            self._thread_local.key_index = None
            self._thread_local.request_count = 0
        
        logger.info(f"[{thread_name}] Cleanup complete")
    
    def _get_safe_filename(self, base_filename: str, content_hash: str = None) -> str:
        """Generate a safe, unique filename"""
        # Add content hash if provided
        if content_hash:
            name, ext = os.path.splitext(base_filename)
            return f"{name}_{content_hash[:8]}{ext}"
        return base_filename

    def _is_file_being_written(self, filepath: str) -> bool:
        """Check if a file is currently being written by another thread"""
        with self._file_lock:
            return filepath in self._active_files

    def _mark_file_active(self, filepath: str):
        """Mark a file as being written"""
        with self._file_lock:
            self._active_files.add(filepath)

    def _mark_file_complete(self, filepath: str):
        """Mark a file write as complete"""
        with self._file_lock:
            self._active_files.discard(filepath)

    def _extract_chapter_info(self, messages) -> dict:
        """Extract chapter and chunk information from messages and progress file
        
        Args:
            messages: The messages to search for chapter/chunk info
        
        Returns:
            dict with 'chapter', 'chunk', 'total_chunks'
        """
        info = {
            'chapter': None,
            'chunk': None,
            'total_chunks': None
        }
        
        messages_str = str(messages)
        
        # First extract chapter number from messages
        chapter_match = re.search(r'Chapter\s+(\d+)', messages_str, re.IGNORECASE)
        if not chapter_match:
            # Try Section pattern for text files
            chapter_match = re.search(r'Section\s+(\d+)', messages_str, re.IGNORECASE)
        
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            info['chapter'] = str(chapter_num)
            
            # Now try to get more accurate info from progress file
            # Look for translation_progress.json in common locations
            possible_paths = [
                'translation_progress.json',
                os.path.join('Payloads', 'translation_progress.json'),
                os.path.join(os.getcwd(), 'Payloads', 'translation_progress.json')
            ]
            
            # Check environment variable for output directory
            output_dir = os.getenv('OUTPUT_DIRECTORY', '')
            if output_dir:
                possible_paths.insert(0, os.path.join(output_dir, 'translation_progress.json'))
            
            progress_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    progress_file = path
                    break
            
            if progress_file:
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        prog = json.load(f)
                    
                    # Look through chapters for matching actual_num
                    for chapter_key, chapter_info in prog.get("chapters", {}).items():
                        if chapter_info.get('actual_num') == chapter_num:
                            # Found it! Get chunk info if available
                            if chapter_key in prog.get("chapter_chunks", {}):
                                chunk_data = prog["chapter_chunks"][chapter_key]
                                info['total_chunks'] = chunk_data.get('total')
                                
                                # Get current/latest chunk
                                completed = chunk_data.get('completed', [])
                                if completed:
                                    info['chunk'] = str(max(completed) + 1)  # Next chunk to process
                                else:
                                    info['chunk'] = '1'  # First chunk
                            break
                except:
                    pass  # Fallback to regex parsing
        
        # If we didn't get chunk info from progress file, try regex
        if not info['chunk']:
            chunk_match = re.search(r'Chunk\s+(\d+)/(\d+)', messages_str)
            if chunk_match:
                info['chunk'] = chunk_match.group(1)
                info['total_chunks'] = chunk_match.group(2)
        
        return info


    def get_current_key_info(self) -> str:
        """Get information about the currently active key"""
        if self.use_multi_keys and self.current_key_index is not None:
            key = self._api_key_pool.keys[self.current_key_index]
            status = "Active" if key.is_available() else "Cooling Down"
            return f"{self.key_identifier} - Status: {status}, Success: {key.success_count}, Errors: {key.error_count}"
        else:
            return "Single Key Mode"
 
    def _should_rotate(self) -> bool:
        """Check if we should rotate keys based on settings"""
        if not self.use_multi_keys:
            return False
        
        if not self._force_rotation:
            # Only rotate on errors
            return False
        
        # Check frequency
        with self._counter_lock:
            self._request_counter += 1
            
            # Check if it's time to rotate
            if self._request_counter >= self._rotation_frequency:
                self._request_counter = 0
                return True
            else:
                return False

    def _get_shortest_cooldown_time(self) -> int:
        """Get the shortest cooldown time among all keys"""
        # Check if cancelled at start
        if self._cancelled:
            return 0  # Return immediately if cancelled
            
        if not self._multi_key_mode or not self.__class__._api_key_pool:
            return 60  # Default cooldown
            
        min_cooldown = float('inf')
        now = time.time()
        
        for i, key in enumerate(self.__class__._api_key_pool.keys):
            if key.enabled:
                key_id = f"Key#{i+1} ({key.model})"
                
                # Check rate limit cache
                cache_cooldown = self.__class__._rate_limit_cache.get_remaining_cooldown(key_id)
                if cache_cooldown > 0:
                    min_cooldown = min(min_cooldown, cache_cooldown)
                
                # Also check key's own cooldown
                if key.is_cooling_down and key.last_error_time:
                    remaining = key.cooldown - (now - key.last_error_time)
                    if remaining > 0:
                        min_cooldown = min(min_cooldown, remaining)
        
        # Add random jitter to prevent thundering herd (0-5 seconds)
        jitter = random.randint(0, 5)
        
        # Return the minimum wait time plus jitter, capped at 60 seconds
        base_time = int(min_cooldown) if min_cooldown != float('inf') else 30
        return min(base_time + jitter, 60)
        
    def _execute_with_retry(self,
                             perform,
                             messages,
                             temperature,
                             max_tokens,
                             max_completion_tokens,
                             context,
                             request_id,
                             is_image: bool = False) -> Tuple[str, Optional[str]]:
        """
        Simplified shim retained for compatibility. Executes once without internal retry.
        """
        result = perform(messages, temperature, max_tokens, max_completion_tokens, context, None, request_id)
        if not result or not isinstance(result, tuple):
            raise UnifiedClientError("Invalid result from perform()", error_type="unexpected")
        return result
    def _send_core(self,
                   messages,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   max_completion_tokens: Optional[int] = None,
                   context: Optional[str] = None,
                   image_data: Any = None) -> Tuple[str, Optional[str]]:
        """
        Unified front for send and send_image. Includes multi-key retry wrapper.
        """
        batch_mode = os.getenv("BATCH_TRANSLATION", "0") == "1"
        if not batch_mode:
            self._sequential_send_lock.acquire()
        try:
            self.reset_cleanup_state()
            # Pre-stagger log so users see what's being sent before delay
            self._log_pre_stagger(messages, context or ('image_translation' if image_data else 'translation'))
            self._apply_thread_submission_delay()
            request_id = str(uuid.uuid4())[:8]
            
            # Multi-key retry wrapper
            if self._multi_key_mode:
                # Check if indefinite retry is enabled for multi-key mode too
                indefinite_retry_enabled = os.getenv("INDEFINITE_RATE_LIMIT_RETRY", "1") == "1"
                last_error = None
                attempt = 0
                
                while True:  # Indefinite retry loop when enabled
                    try:
                        if image_data is None:
                            return self._send_internal(messages, temperature, max_tokens, max_completion_tokens, context, retry_reason=None, request_id=request_id)
                        else:
                            return self._send_image_internal(messages, image_data, temperature, max_tokens, max_completion_tokens, context, retry_reason=None, request_id=request_id)
                    
                    except UnifiedClientError as e:
                        last_error = e
                        
                        # Handle rate limit errors with key rotation
                        if e.error_type == "rate_limit" or self._is_rate_limit_error(e):
                            attempt += 1
                            
                            if indefinite_retry_enabled:
                                print(f"🔄 Multi-key mode: Rate limit hit, attempting key rotation (indefinite retry, attempt {attempt})")
                            else:
                                # Limited retry mode - respect max attempts per key
                                num_keys = len(self._api_key_pool.keys) if self._api_key_pool else 3
                                max_attempts = num_keys * 2  # Allow 2 attempts per key
                                print(f"🔄 Multi-key mode: Rate limit hit, attempting key rotation (attempt {attempt}/{max_attempts})")
                                
                                if attempt >= max_attempts:
                                    print(f"❌ Multi-key mode: Exhausted {max_attempts} attempts, giving up")
                                    raise
                            
                            try:
                                # Rotate to next key
                                self._handle_rate_limit_for_thread()
                                print(f"🔄 Multi-key retry: Key rotation completed, preparing for next attempt...")
                                time.sleep(0.1)  # Brief pause after key rotation for system stability
                                
                                # Check if we have any available keys left after rotation
                                available_keys = self._count_available_keys()
                                if available_keys == 0:
                                    print(f"🔄 Multi-key mode: All keys rate-limited, waiting for cooldown...")
                                    # Wait a bit before trying again
                                    wait_time = min(60 + random.uniform(1, 10), 120)  # 60-70 seconds
                                    print(f"🔄 Multi-key mode: Waiting {wait_time:.1f}s for keys to cool down")
                                    
                                    wait_start = time.time()
                                    while time.time() - wait_start < wait_time:
                                        if self._cancelled:
                                            raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                        time.sleep(0.5)
                                
                                # Continue to next attempt with rotated key
                                continue
                                
                            except Exception as rotation_error:
                                print(f"❌ Multi-key mode: Key rotation failed: {rotation_error}")
                                # If rotation fails, we can't continue with multi-key retry
                                if indefinite_retry_enabled:
                                    # In indefinite mode, try to continue with any available key
                                    print(f"🔄 Multi-key mode: Key rotation failed, but indefinite retry enabled - continuing...")
                                    time.sleep(5)  # Brief pause before trying again
                                    continue
                                else:
                                    break
                        else:
                            # Non-rate-limit error, don't retry with different keys
                            raise
                
                # This point is only reached in non-indefinite mode when giving up
                if last_error:
                    print(f"❌ Multi-key mode: All retry attempts failed")
                    raise last_error
                else:
                    raise UnifiedClientError("All multi-key attempts failed", error_type="no_keys")
            else:
                # Single key mode - direct call
                if image_data is None:
                    return self._send_internal(messages, temperature, max_tokens, max_completion_tokens, context, retry_reason=None, request_id=request_id)
                else:
                    return self._send_image_internal(messages, image_data, temperature, max_tokens, max_completion_tokens, context, retry_reason=None, request_id=request_id)
        finally:
            if not batch_mode:
                self._sequential_send_lock.release()
        
    def _get_thread_assigned_key(self) -> Optional[int]:
        """Get the key index assigned to current thread"""
        thread_id = threading.current_thread().ident
        
        with self._key_assignment_lock:
            if thread_id in self._thread_key_assignments:
                key_index, timestamp = self._thread_key_assignments[thread_id]
                # Check if assignment is still valid (not expired)
                if time.time() - timestamp < 300:  # 5 minute expiry
                    return key_index
                else:
                    # Expired, remove it
                    del self._thread_key_assignments[thread_id]
        
        return None

    def _assign_key_to_thread(self, key_index: int):
        """Assign a key to the current thread"""
        thread_id = threading.current_thread().ident
        
        with self._key_assignment_lock:
            self._thread_key_assignments[thread_id] = (key_index, time.time())
            
            # Cleanup old assignments
            current_time = time.time()
            expired_threads = [
                tid for tid, (_, ts) in self._thread_key_assignments.items()
                if current_time - ts > 300
            ]
            for tid in expired_threads:
                del self._thread_key_assignments[tid]
                
                
    def _setup_client(self):
        """Setup the appropriate client based on model type"""
        model_lower = self.model.lower()
        tls = self._get_thread_local_client()
        
        # Determine client_type (no lock needed, just reading)
        self.client_type = None
        for prefix, provider in self.MODEL_PROVIDERS.items():
            if model_lower.startswith(prefix):
                self.client_type = provider
                break
        
        # Check if we're using a custom OpenAI base URL
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', ''))
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        
        # Apply custom endpoint logic when enabled - override any model type (except Gemini which has its own toggle)
        if custom_base_url and custom_base_url != 'https://api.openai.com/v1' and use_custom_endpoint:
            if not self.client_type:
                # No prefix matched - assume it's a custom model that should use OpenAI endpoint
                self.client_type = 'openai'
                logger.info(f"Using OpenAI client for custom endpoint with unmatched model: {self.model}")
            elif self.client_type == 'openai':
                logger.info(f"Using custom OpenAI endpoint for OpenAI model: {self.model}")
            elif self.client_type == 'gemini':
                # Don't override Gemini - it has its own separate endpoint toggle
                # Only log if Gemini OpenAI endpoint is not also enabled
                use_gemini_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
                if not use_gemini_endpoint:
                    self._log_once(f"Gemini model detected, not overriding with custom OpenAI endpoint (use USE_GEMINI_OPENAI_ENDPOINT instead)")
            else:
                # Override other model types to use custom OpenAI endpoint when toggle is enabled
                original_client_type = self.client_type
                self.client_type = 'openai'
                print(f"[DEBUG] Custom endpoint override: {original_client_type} -> openai for model '{self.model}'")
                logger.info(f"Custom endpoint enabled: Overriding {original_client_type} model {self.model} to use OpenAI client")
        elif not use_custom_endpoint and custom_base_url and self.client_type == 'openai':
            logger.info("Custom OpenAI endpoint disabled via toggle, using default endpoint")
        
        # If still no client type, show error with suggestions
        if not self.client_type:
            # Provide helpful suggestions
            suggestions = []
            for prefix in self.MODEL_PROVIDERS.keys():
                if prefix in model_lower or model_lower[:3] in prefix:
                    suggestions.append(prefix)
            
            error_msg = f"Unsupported model: {self.model}. "
            if suggestions:
                error_msg += f"Did you mean to use one of these prefixes? {suggestions}. "
            else:
                # Check if it might be an aggregator model
                if any(provider in model_lower for provider in ['yi', 'qwen', 'llama', 'gpt', 'claude']):
                    error_msg += f"If using ElectronHub, prefix with 'eh/' (e.g., eh/{self.model}). "
                    error_msg += f"If using OpenRouter, prefix with 'or/' (e.g., or/{self.model}). "
                    error_msg += f"If using Poe, prefix with 'poe/' (e.g., poe/{self.model}). "
            error_msg += f"Supported prefixes: {list(self.MODEL_PROVIDERS.keys())}"
            raise ValueError(error_msg)
        
        # Initialize variables at method scope for all client types
        base_url = None
        use_gemini_endpoint = False
        gemini_endpoint = ""
        
        # Prepare provider-specific settings (but don't create clients yet)
        if self.client_type == 'openai':
            if openai is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
        elif self.client_type == 'gemini':
            # Check if we should use OpenAI-compatible endpoint for Gemini
            use_gemini_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
            gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
            
            if use_gemini_endpoint and gemini_endpoint:
                # Use OpenAI client for Gemini with custom endpoint
                #print(f"[DEBUG] Preparing Gemini with OpenAI-compatible endpoint")
                pass
                if openai is None:
                    raise ImportError("OpenAI library not installed. Install with: pip install openai")
                
                # Ensure endpoint has proper format
                if not gemini_endpoint.endswith('/openai/'):
                    if gemini_endpoint.endswith('/'):
                        gemini_endpoint = gemini_endpoint + 'openai/'
                    else:
                        gemini_endpoint = gemini_endpoint + '/openai/'
                
                # Set base_url for Gemini OpenAI endpoint
                base_url = gemini_endpoint
                
                print(f"[DEBUG] Gemini will use OpenAI-compatible endpoint: {gemini_endpoint}")
                
                disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
                
                config_data = {
                    "type": "GEMINI_OPENAI_ENDPOINT_REQUEST",
                    "model": self.model,
                    "endpoint": gemini_endpoint,
                    "safety_enabled": not disable_safety,
                    "safety_settings": "DISABLED_VIA_OPENAI_ENDPOINT" if disable_safety else "DEFAULT",
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Just call the existing save method
                self._save_gemini_safety_config(config_data, None)
            else:
                # Use native Gemini client
                #print(f"[DEBUG] Preparing native Gemini client")
                if not GENAI_AVAILABLE:
                    raise ImportError(
                        "Google Gen AI library not installed. Install with: "
                        "pip install google-genai"
                    )
        
        elif self.client_type == 'electronhub':
            # ElectronHub uses OpenAI SDK if available
            if openai is not None:
                logger.info("ElectronHub will use OpenAI SDK for API calls")
            else:
                logger.info("ElectronHub will use HTTP API for API calls")

        elif self.client_type == 'chutes':
            # chutes uses OpenAI-compatible endpoint
            if openai is not None:
                chutes_base_url = os.getenv("CHUTES_API_URL", "https://llm.chutes.ai/v1")
                
                # MICROSECOND LOCK for chutes client
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=chutes_base_url
                    )
                logger.info(f"chutes client configured with endpoint: {chutes_base_url}")
            else:
                logger.info("chutes will use HTTP API")
        
        elif self.client_type == 'mistral':
            if MistralClient is None:
                # Fall back to HTTP API if SDK not installed
                logger.info("Mistral SDK not installed, will use HTTP API")
        
        elif self.client_type == 'cohere':
            if cohere is None:
                logger.info("Cohere SDK not installed, will use HTTP API")
        
        elif self.client_type == 'anthropic':
            if anthropic is None:
                logger.info("Anthropic SDK not installed, will use HTTP API")
            else:
                # Store API key for HTTP fallback
                self.anthropic_api_key = self.api_key
                logger.info("Anthropic client configured")
        
        elif self.client_type == 'deepseek':
            # DeepSeek typically uses OpenAI-compatible endpoint
            if openai is None:
                logger.info("DeepSeek will use HTTP API")
            else:
                base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
                logger.info(f"DeepSeek will use endpoint: {base_url}")
        
        elif self.client_type == 'groq':
            # Groq uses OpenAI-compatible endpoint
            if openai is None:
                logger.info("Groq will use HTTP API")
            else:
                base_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
                logger.info(f"Groq will use endpoint: {base_url}")
        
        elif self.client_type == 'fireworks':
            # Fireworks uses OpenAI-compatible endpoint
            if openai is None:
                logger.info("Fireworks will use HTTP API")
            else:
                base_url = os.getenv("FIREWORKS_API_URL", "https://api.fireworks.ai/inference/v1")
                logger.info(f"Fireworks will use endpoint: {base_url}")
        
        elif self.client_type == 'xai':
            # xAI (Grok) uses OpenAI-compatible endpoint
            if openai is None:
                logger.info("xAI will use HTTP API")
            else:
                base_url = os.getenv("XAI_API_URL", "https://api.x.ai/v1")
                logger.info(f"xAI will use endpoint: {base_url}")
        
        # =====================================================
        # MICROSECOND LOCK: Create ALL clients with thread safety
        # =====================================================
        
        if self.client_type == 'openai':
            # Skip if individual endpoint already applied
            if hasattr(self, '_individual_endpoint_applied') and self._individual_endpoint_applied:
                return
                
            # MICROSECOND LOCK for OpenAI client
            with self._model_lock:
                # Use regular OpenAI client - individual endpoint will be set later
                self.openai_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url='https://api.openai.com/v1'  # Default, will be overridden by individual endpoint
                )
        
        elif self.client_type == 'gemini':
            if use_gemini_endpoint and gemini_endpoint:
                # Use OpenAI client for Gemini endpoint
                if base_url is None:
                    base_url = gemini_endpoint
                
                # MICROSECOND LOCK for Gemini with OpenAI endpoint
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                    self._original_client_type = 'gemini'
                    self.client_type = 'openai'
                print(f"[DEBUG] Gemini using OpenAI-compatible endpoint: {base_url}")
            else:
                # MICROSECOND LOCK for native Gemini client
                # Check if this key has Google credentials (multi-key mode)
                google_creds = None
                if hasattr(self, 'current_key_google_creds') and self.current_key_google_creds:
                    google_creds = self.current_key_google_creds
                    print(f"[DEBUG] Using key-specific Google credentials: {os.path.basename(google_creds)}")
                    # Set environment variable for this request
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                elif hasattr(self, 'google_creds_path') and self.google_creds_path:
                    google_creds = self.google_creds_path
                    print(f"[DEBUG] Using default Google credentials: {os.path.basename(google_creds)}")
                
                with self._model_lock:
                    self.gemini_client = genai.Client(api_key=self.api_key)
                if hasattr(tls, 'model'):
                    tls.gemini_configured = True
                    tls.gemini_api_key = self.api_key
                    tls.gemini_client = self.gemini_client
                
                #print(f"[DEBUG] Created native Gemini client for model: {self.model}")
        
        elif self.client_type == 'mistral':
            if MistralClient is not None:
                # MICROSECOND LOCK for Mistral client
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.mistral_client = MistralClient(api_key=self.api_key)
                else:
                    self.mistral_client = MistralClient(api_key=self.api_key)
                logger.info("Mistral client created")
        
        elif self.client_type == 'cohere':
            if cohere is not None:
                # MICROSECOND LOCK for Cohere client
                with self._model_lock:
                    self.cohere_client = cohere.Client(self.api_key)
                logger.info("Cohere client created")
        
        elif self.client_type == 'deepseek':
            if openai is not None:
                if base_url is None:
                    base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
                
                # MICROSECOND LOCK for DeepSeek client
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                logger.info(f"DeepSeek client configured with endpoint: {base_url}")
        
        elif self.client_type == 'groq':
            if openai is not None:
                if base_url is None:
                    base_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
                
                # MICROSECOND LOCK for Groq client
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                logger.info(f"Groq client configured with endpoint: {base_url}")
        
        elif self.client_type == 'fireworks':
            if openai is not None:
                if base_url is None:
                    base_url = os.getenv("FIREWORKS_API_URL", "https://api.fireworks.ai/inference/v1")
                
                # MICROSECOND LOCK for Fireworks client
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                logger.info(f"Fireworks client configured with endpoint: {base_url}")
        
        elif self.client_type == 'xai':
            if openai is not None:
                if base_url is None:
                    base_url = os.getenv("XAI_API_URL", "https://api.x.ai/v1")
                
                # MICROSECOND LOCK for xAI client
                with self._model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                logger.info(f"xAI client configured with endpoint: {base_url}")
 
        elif self.client_type == 'deepl' or self.model.startswith('deepl'):
            self.client_type = 'deepl'
            self.client = None  # No persistent client needed
            return 
            
        elif self.client_type == 'google_translate' or self.model.startswith('google-translate'):
            self.client_type = 'google_translate'
            self.client = None  # No persistent client needed
            return            
 
        elif self.client_type == 'vertex_model_garden':
            # Vertex AI doesn't need a client created here
            logger.info("Vertex AI Model Garden will initialize on demand")
        
        elif self.client_type in ['yi', 'qwen', 'baichuan', 'zhipu', 'moonshot', 'baidu', 
                                  'tencent', 'iflytek', 'bytedance', 'minimax', 
                                  'sensenova', 'internlm', 'tii', 'microsoft', 
                                  'azure', 'google', 'alephalpha', 'databricks', 
                                  'huggingface', 'salesforce', 'bigscience', 'meta',
                                  'electronhub', 'poe', 'openrouter', 'chutes']:
            # These providers will use HTTP API or OpenAI-compatible endpoints
            # No client initialization needed here
            logger.info(f"{self.client_type} will use HTTP API or compatible endpoint")
        
        # Store thread-local client reference if in multi-key mode
        if self._multi_key_mode and hasattr(tls, 'model'):
            # MICROSECOND LOCK for thread-local storage
            with self._model_lock:
                tls.client_type = self.client_type
                if hasattr(self, 'openai_client'):
                    tls.openai_client = self.openai_client
                if hasattr(self, 'gemini_client'):
                    tls.gemini_client = self.gemini_client
                if hasattr(self, 'mistral_client'):
                    tls.mistral_client = self.mistral_client
                if hasattr(self, 'cohere_client'):
                    tls.cohere_client = self.cohere_client
        else:
            tls.client_type = self.client_type
            if hasattr(self, 'openai_client'):
                tls.openai_client = self.openai_client
            if hasattr(self, 'gemini_client'):
                tls.gemini_client = self.gemini_client
            if hasattr(self, 'mistral_client'):
                tls.mistral_client = self.mistral_client
            if hasattr(self, 'cohere_client'):
                tls.cohere_client = self.cohere_client
        
        # Log retry feature support
        logger.info(f"✅ Initialized {self.client_type} client for model: {self.model}")
        logger.debug("✅ GUI retry features supported: truncation detection, timeout handling, duplicate detection")
    
    def send(self, messages, temperature=None, max_tokens=None, 
             max_completion_tokens=None, context=None) -> Tuple[str, Optional[str]]:
        """Backwards-compatible public API; now delegates to unified _send_core."""
        return self._send_core(messages, temperature, max_tokens, max_completion_tokens, context, image_data=None)

    def _send_internal(self, messages, temperature=None, max_tokens=None, 
                       max_completion_tokens=None, context=None, retry_reason=None,
                       request_id=None, image_data=None) -> Tuple[str, Optional[str]]:
        """
        Unified internal send implementation for both text and image requests.
        Pass image_data=None for text requests, or image bytes/base64 for image requests.
        """
        # Determine if this is an image request
        is_image_request = image_data is not None
        
        # Use appropriate context default
        if context is None:
            context = 'image_translation' if is_image_request else 'translation'
        
        # Always ensure per-request key assignment/rotation for multi-key mode
        # This guarantees forced rotation when rotation_frequency == 1
        if getattr(self, '_multi_key_mode', False):
            try:
                self._ensure_thread_client()
            except UnifiedClientError:
                # Propagate known client errors
                raise
            except Exception as e:
                # Normalize unexpected failures
                raise UnifiedClientError(f"Failed to acquire API key for thread: {e}", error_type="no_keys")
        
        # Handle refactored mode with disabled internal retry
        if getattr(self, '_disable_internal_retry', False):
            t0 = time.time()
            
            # For image requests, prepare messages with embedded image
            if image_data:
                messages = self._prepare_image_messages(messages, image_data)
            
            # Validate request
            valid, error_msg = self._validate_request(messages, max_tokens)
            if not valid:
                raise UnifiedClientError(f"Invalid request: {error_msg}", error_type="validation")
            
            # File names and payload save
            payload_name, response_name = self._get_file_names(messages, context=self.context or context)
            self._save_payload(messages, payload_name, retry_reason=retry_reason)
            
            # Get response via provider router
            response = self._get_response(messages, temperature, max_tokens, max_completion_tokens, response_name)
            
            # Extract text uniformly
            extracted_content, finish_reason = self._extract_response_text(response, provider=getattr(self, 'client_type', 'unknown'))
            
            # Save response if any
            if extracted_content:
                self._save_response(extracted_content, response_name)
            
            # Stats and success mark
            self._track_stats(context, True, None, time.time() - t0)
            self._mark_key_success()
            
            # API delay between calls (respects GUI setting)
            self._apply_api_delay()
            
            return extracted_content, finish_reason

        # Main implementation with retry logic
        start_time = time.time()
        
        # Generate request hash WITH request ID if provided
        if image_data:
            image_size = len(image_data) if isinstance(image_data, (bytes, str)) else 0
            if request_id:
                messages_hash = self._get_request_hash_with_request_id(messages, request_id)
            else:
                request_id = str(uuid.uuid4())[:8]
                messages_hash = self._get_request_hash_with_request_id(messages, request_id)
            request_hash = f"{messages_hash}_img{image_size}"
        else:
            if request_id:
                request_hash = self._get_request_hash_with_request_id(messages, request_id)
            else:
                request_id = str(uuid.uuid4())[:8]
                request_hash = self._get_request_hash_with_request_id(messages, request_id)
        
        thread_name = threading.current_thread().name
        
        # Log with hash for tracking
        logger.debug(f"  Request ID: {request_id}")
        logger.debug(f"  Hash: {request_hash[:8]}...")
        logger.debug(f"  Retry reason: {retry_reason}")
        
        # Log with hash for tracking
        logger.debug(f"[{thread_name}] _send_internal starting for {context} (hash: {request_hash[:8]}...) retry_reason: {retry_reason}")
        
        # Reset cancelled flag
        self._cancelled = False
        
        # Reset counters when context changes
        if context != self.current_session_context:
            self.reset_conversation_for_new_context(context)
        
        self.context = context or 'translation'
        self.conversation_message_count += 1
        
        # Internal retry logic for 500 errors - now optionally disabled (centralized retry handles it)
        internal_retries = self._get_max_retries()
        base_delay = 5  # Base delay for exponential backoff
        
        # Track if we've tried main key for prohibited content
        main_key_attempted = False
        
        # Initialize variables that might be referenced in exception handlers
        extracted_content = ""
        finish_reason = 'error'
        
        for attempt in range(internal_retries):
            try:
                # Validate request
                valid, error_msg = self._validate_request(messages, max_tokens)
                if not valid:
                    raise UnifiedClientError(f"Invalid request: {error_msg}", error_type="validation")
                
                os.makedirs("Payloads", exist_ok=True)
                
                # Apply reinforcement
                messages = self._apply_pure_reinforcement(messages)
                
                # For image requests, prepare messages with embedded image
                if image_data:
                    messages = self._prepare_image_messages(messages, image_data)
                
                # Get file names - now unique per request AND attempt
                payload_name, response_name = self._get_file_names(messages, context=self.context)
                
                # Add request ID and attempt to filename for complete isolation
                payload_name, response_name = self._with_attempt_suffix(payload_name, response_name, request_id, attempt, is_image=bool(image_data))
                
                # Save payload with retry reason
                # On internal retries (500 errors), add that info too
                if attempt > 0:
                    internal_retry_reason = f"500_error_attempt_{attempt}"
                    if retry_reason:
                        combined_reason = f"{retry_reason}_{internal_retry_reason}"
                    else:
                        combined_reason = internal_retry_reason
                    self._save_payload(messages, payload_name, retry_reason=combined_reason)
                else:
                    self._save_payload(messages, payload_name, retry_reason=retry_reason)
                
                # FIX: Define payload_messages BEFORE using it
                # Create a sanitized version for payload (without actual image data)
                payload_messages = [
                    {**msg, 'content': 'IMAGE_DATA_OMITTED' if isinstance(msg.get('content'), list) else msg.get('content')}
                    for msg in messages
                ]
                
                # Now save the payload (payload_messages is now defined)
                #self._save_payload(payload_messages, payload_name)
                
                
                # Set idempotency context for downstream calls
                self._set_idempotency_context(request_id, attempt)
                
                # Unified provider dispatch: for image requests, messages already embed the image.
                # Route via the same _get_response used for text; Gemini handler internally detects images.
                response = self._get_response(messages, temperature, max_tokens, max_completion_tokens, response_name)
                
                # Check for cancellation (from timeout or stop button)
                if self._cancelled:
                    if not self._is_stop_requested():
                        logger.info("Operation cancelled (timeout or user stop)")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                # ====== UNIVERSAL EXTRACTION INTEGRATION ======
                # Use universal extraction instead of assuming response.content exists
                extracted_content = ""
                finish_reason = 'stop'

                if response:
                    # Prepare provider-specific parameters
                    extraction_kwargs = {}
                    
                    # Add provider-specific parameters if applicable
                    extraction_kwargs.update(self._get_extraction_kwargs())
                    
# Try universal extraction with provider-specific parameters
                    extracted_content, finish_reason = self._extract_response_text(
                        response, 
                        provider=getattr(self, 'client_type', 'unknown'),
                        **extraction_kwargs
                    )
                    
                    # If extraction failed but we have a response object
                    if not extracted_content and response:
                        print(f"⚠️ Failed to extract text from {getattr(self, 'client_type', 'unknown')} response")
                        print(f"   Response type: {type(response)}")
                        
                        # Provider-specific guidance
                        if getattr(self, 'client_type', None) == 'gemini':
                            print(f"   Consider checking Gemini response structure")
                            print(f"   Response attributes: {dir(response)[:5]}...")  # Show first 5 attributes
                        else:
                            print(f"   Consider checking response extraction for this provider")
                        
                        # Log the response structure for debugging
                        self._save_failed_request(messages, "Extraction failed", context, response)
                        
                        # Check if response has any common attributes we missed
                        if hasattr(response, 'content') and response.content:
                            extracted_content = str(response.content)
                            print(f"   Fallback: Using response.content directly")
                        elif hasattr(response, 'text') and response.text:
                            extracted_content = str(response.text)
                            print(f"   Fallback: Using response.text directly")
                    
                    # Update response object with extracted content
                    if extracted_content and hasattr(response, 'content'):
                        response.content = extracted_content
                    elif extracted_content:
                        # Create a new response object if needed
                        response = UnifiedResponse(
                            content=extracted_content,
                            finish_reason=finish_reason,
                            raw_response=response
                        )
                
                # CRITICAL: Save response for duplicate detection
                # This must happen even for truncated/empty responses
                if extracted_content:
                    self._save_response(extracted_content, response_name)

                # Handle empty responses
                if not extracted_content or extracted_content.strip() in ["", "[]", "[IMAGE TRANSLATION FAILED]"]:
                    is_likely_safety_filter = self._detect_safety_filter(messages, extracted_content, finish_reason, response, getattr(self, 'client_type', 'unknown'))
                    if is_likely_safety_filter and not main_key_attempted and self._multi_key_mode and getattr(self, 'original_api_key', None) and getattr(self, 'original_model', None):
                        main_key_attempted = True
                        try:
                            retry_res = self._retry_with_main_key(messages, temperature, max_tokens, max_completion_tokens, context, request_id=request_id, image_data=image_data)
                            if retry_res:
                                content, fr = retry_res
                                if content and content.strip() and len(content) > 10:
                                    return content, fr
                        except Exception:
                            pass
                    # Finalize empty handling
                    req_type = 'image' if image_data else 'text'
                    return self._finalize_empty_response(messages, context, response, extracted_content or "", finish_reason, getattr(self, 'client_type', 'unknown'), req_type, start_time)
                                
                # Track success
                self._track_stats(context, True, None, time.time() - start_time)
                
                # Mark key as successful in multi-key mode
                self._mark_key_success()
                
                # Check for truncation and handle retry if enabled
                if finish_reason in ['length', 'max_tokens']:
                    print(f"Response was truncated: {finish_reason}")
                    print(f"⚠️ Response truncated (finish_reason: {finish_reason})")
                    
                    # ALWAYS log truncation failures
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=extracted_content,
                        finish_reason=finish_reason,
                        context=context,
                        error_details=getattr(response, 'error_details', None) if response else None
                    )
                    
                    # Check if retry on truncation is enabled
                    retry_truncated_enabled = os.getenv("RETRY_TRUNCATED", "0") == "1"
                    
                    if retry_truncated_enabled:
                        print(f"  🔄 RETRY_TRUNCATED enabled - attempting to retry with increased token limit")
                        
                        # Get the max retry tokens limit
                        max_retry_tokens = int(os.getenv("MAX_RETRY_TOKENS", "16384"))
                        current_max_tokens = max_tokens or 8192
                        
                        if current_max_tokens < max_retry_tokens:
                            new_max_tokens = min(current_max_tokens * 2, max_retry_tokens)
                            print(f"  📊 Retrying with increased tokens: {current_max_tokens} → {new_max_tokens}")
                            
                            try:
                                # Recursive call with increased token limit
                                retry_content, retry_finish_reason = self._send_internal(
                                    messages=messages,
                                    temperature=temperature, 
                                    max_tokens=new_max_tokens,
                                    max_completion_tokens=max_completion_tokens,
                                    context=context,
                                    retry_reason=f"truncation_retry_{finish_reason}",
                                    request_id=request_id,
                                    image_data=image_data
                                )
                                
                                # Check if retry succeeded (not truncated)
                                if retry_finish_reason not in ['length', 'max_tokens']:
                                    print(f"  ✅ Truncation retry succeeded: {len(retry_content)} chars")
                                    return retry_content, retry_finish_reason
                                else:
                                    print(f"  ⚠️ Retry was also truncated, returning original response")
                                    
                            except Exception as retry_error:
                                print(f"  ❌ Truncation retry failed: {retry_error}")
                        else:
                            print(f"  📊 Already at max retry tokens ({current_max_tokens}), not retrying")
                    else:
                        print(f"  📋 RETRY_TRUNCATED disabled - accepting truncated response")
                
                # Apply API delay after successful call (even if truncated)
                # SKIP DELAY DURING CLEANUP
                
                self._apply_api_delay()
                
                # Brief stability pause after API call completion
                if not getattr(self, '_in_cleanup', False):
                    time.sleep(0.1)  # System stability pause after API completion
                
                # If the provider signaled a content filter, elevate to prohibited_content to trigger retries
                if finish_reason == 'content_filter':
                    raise UnifiedClientError(
                        "Content blocked by provider",
                        error_type="prohibited_content",
                        http_status=400
                    )
                
                # Return the response with accurate finish_reason
                # This is CRITICAL for retry mechanisms to work
                return extracted_content, finish_reason
                
            except UnifiedClientError as e:
                # Handle cancellation specially for timeout support
                if e.error_type == "cancelled" or "cancelled" in str(e):
                    self._in_cleanup = False  # Ensure cleanup flag is set
                    if not self._is_stop_requested():
                        logger.info("Propagating cancellation to caller")
                    # Re-raise so send_with_interrupt can handle it
                    raise
                
                print(f"UnifiedClient error: {e}")
                
                # Check if it's a rate limit error - handle according to mode
                error_str = str(e).lower()
                if self._is_rate_limit_error(e):
                    # In multi-key mode, always re-raise to let _send_core handle key rotation
                    if self._multi_key_mode:
                        print(f"🔄 Rate limit error - multi-key mode active, re-raising for key rotation")
                        raise
                    
                    # In single-key mode, check if indefinite retry is enabled
                    indefinite_retry_enabled = os.getenv("INDEFINITE_RATE_LIMIT_RETRY", "1") == "1"
                    
                    if indefinite_retry_enabled:
                        # Calculate wait time from Retry-After header if available
                        retry_after_seconds = 60  # Default wait time
                        if hasattr(e, 'http_status') and e.http_status == 429:
                            # Try to extract Retry-After from the error if it contains header info
                            error_details = str(e)
                            if 'retry-after' in error_details.lower():
                                import re
                                match = re.search(r'retry-after[:\s]+([0-9]+)', error_details.lower())
                                if match:
                                    retry_after_seconds = int(match.group(1))
                        
                        # Add some jitter and cap the wait time
                        wait_time = min(retry_after_seconds + random.uniform(1, 10), 300)  # Max 5 minutes
                        
                        print(f"🔄 Rate limit error - single-key indefinite retry, waiting {wait_time:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        wait_start = time.time()
                        while time.time() - wait_start < wait_time:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        
                        # For rate limit errors, continue retrying without counting against max retries
                        # Reset attempt counter to avoid exhausting retries on rate limits
                        attempt = max(0, attempt - 1)  # Don't count rate limit waits against retry budget
                        continue  # Retry the attempt
                    else:
                        print(f"❌ Rate limit error - single-key mode, indefinite retry disabled, re-raising")
                        raise
                
                # Check for prohibited content — treat any HTTP 400 as prohibited to force fallback
                if (
                    e.error_type == "prohibited_content"
                    or getattr(e, 'http_status', None) == 400
                    or " 400 " in error_str
                    or self._detect_safety_filter(messages, extracted_content or "", finish_reason, None, getattr(self, 'client_type', 'unknown'))
                ):
                    print(f"❌ Prohibited content detected: {error_str[:200]}")
                    
                    # Different behavior based on mode
                    if self._multi_key_mode:
                        # Multi-key mode: Attempt main key retry once, then fall through to fallback
                        if not main_key_attempted:
                            main_key_attempted = True
                            retry_res = self._maybe_retry_main_key_on_prohibited(
                                messages, temperature, max_tokens, max_completion_tokens, context, request_id=request_id, image_data=image_data
                            )
                            if retry_res:
                                res_content, res_fr = retry_res
                                if res_content and res_content.strip():
                                    return res_content, res_fr
                    else:
                        # Single-key mode: Check if fallback keys are enabled
                        use_fallback_keys = os.getenv('USE_FALLBACK_KEYS', '0') == '1'
                        if use_fallback_keys:
                            print(f"[FALLBACK DIRECT] Using fallback keys")
                            # Try fallback keys directly without retrying main key
                            retry_res = self._try_fallback_keys_direct(
                                messages, temperature, max_tokens, max_completion_tokens, context, request_id=request_id, image_data=image_data
                            )
                            if retry_res:
                                res_content, res_fr = retry_res
                                if res_content and res_content.strip():
                                    return res_content, res_fr
                        else:
                            print(f"[SINGLE-KEY MODE] Fallback keys disabled - no retry available")
                    
                    # Fallthrough: record and return generic fallback
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for retryable server errors (500, 502, 503, 504)
                http_status = getattr(e, 'http_status', None)
                retryable_errors = ["500", "502", "503", "504", "api_error", "internal server error", "bad gateway", "service unavailable", "gateway timeout"]
                
                if (http_status in [500, 502, 503, 504] or 
                    any(err in error_str for err in retryable_errors)):
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = self._compute_backoff(attempt, base_delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Server error ({http_status or 'API error'}) - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)  # Check every 0.5 seconds
                        print(f"🔄 Server error retry: Backoff completed, initiating retry attempt...")
                        time.sleep(0.1)  # Brief pause after backoff for retry stability
                        continue  # Retry the attempt
                    else:
                        print(f"❌ Server error ({http_status or 'API error'}) - exhausted {internal_retries} retries")
                
                # Check for other retryable errors (timeouts, connection issues)
                timeout_errors = ["timeout", "timed out", "connection reset", "connection aborted", "connection error", "network error"]
                if any(err in error_str for err in timeout_errors):
                    if attempt < internal_retries - 1:
                        delay = self._compute_backoff(attempt, base_delay/2, 30)  # Shorter delay for timeouts
                        
                        print(f"🔄 Network/timeout error - retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        print(f"🔄 Timeout error retry: Backoff completed, initiating retry attempt...")
                        time.sleep(0.1)  # Brief pause after backoff for retry stability
                        continue  # Retry the attempt
                    else:
                        print(f"❌ Network/timeout error - exhausted {internal_retries} retries")
                
                # If we get here, this is the last attempt or a non-retryable error
                # Save failed request and return fallback only if we've exhausted retries
                if attempt >= internal_retries - 1:
                    print(f"❌ Final attempt failed, returning fallback response")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                else:
                    # For other errors, try again with a short delay
                    delay = self._compute_backoff(attempt, base_delay/4, 15)  # Short delay for other errors
                    print(f"🔄 API error - retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries}): {str(e)[:100]}")
                    
                    wait_start = time.time()
                    while time.time() - wait_start < delay:
                        if self._cancelled:
                            raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                        time.sleep(0.5)
                    print(f"🔄 API error retry: Backoff completed, initiating retry attempt...")
                    time.sleep(0.1)  # Brief pause after backoff for retry stability
                    continue  # Retry the attempt
                
            except Exception as e:
                # COMPREHENSIVE ERROR HANDLING FOR NoneType and other issues
                error_str = str(e).lower()
                print(f"Unexpected error: {e}")
                
                # Special handling for NoneType length errors
                if "nonetype" in error_str and "len" in error_str:
                    print(f"🚨 Detected NoneType length error - likely caused by None message content")
                    print(f"🔍 Error details: {type(e).__name__}: {e}")
                    print(f"🔍 Context: {context}, Messages count: {self._safe_len(messages, 'unexpected_error_messages')}")
                    
                    # Log the actual traceback for debugging
                    import traceback
                    print(f"🔍 Traceback: {traceback.format_exc()}")
                    
                    # Return a safe fallback
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "nonetype_length_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, "NoneType length error")
                    return fallback_content, 'error'
                
                # For unexpected errors, check if it's a timeout
                if "timed out" in error_str:
                    # Re-raise timeout errors so the retry logic can handle them
                    raise UnifiedClientError(f"Request timed out: {e}", error_type="timeout")
                
                # Check if it's a rate limit error - handle according to mode
                if self._is_rate_limit_error(e):
                    # In multi-key mode, always re-raise to let _send_core handle key rotation
                    if self._multi_key_mode:
                        print(f"🔄 Unexpected rate limit error - multi-key mode active, re-raising for key rotation")
                        raise
                    
                    # In single-key mode, check if indefinite retry is enabled
                    indefinite_retry_enabled = os.getenv("INDEFINITE_RATE_LIMIT_RETRY", "1") == "1"
                    
                    if indefinite_retry_enabled:
                        # Calculate wait time from Retry-After header if available
                        retry_after_seconds = 60  # Default wait time
                        if hasattr(e, 'http_status') and e.http_status == 429:
                            # Try to extract Retry-After from the error if it contains header info
                            error_details = str(e)
                            if 'retry-after' in error_details.lower():
                                import re
                                match = re.search(r'retry-after[:\s]+([0-9]+)', error_details.lower())
                                if match:
                                    retry_after_seconds = int(match.group(1))
                        
                        # Add some jitter and cap the wait time
                        wait_time = min(retry_after_seconds + random.uniform(1, 10), 300)  # Max 5 minutes
                        
                        print(f"🔄 Unexpected rate limit error - single-key indefinite retry, waiting {wait_time:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        wait_start = time.time()
                        while time.time() - wait_start < wait_time:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        
                        # For rate limit errors, continue retrying without counting against max retries
                        # Reset attempt counter to avoid exhausting retries on rate limits
                        attempt = max(0, attempt - 1)  # Don't count rate limit waits against retry budget
                        continue  # Retry the attempt
                    else:
                        print(f"❌ Unexpected rate limit error - single-key mode, indefinite retry disabled, re-raising")
                        raise  # Re-raise for higher-level handling
                
                # Check for prohibited content in unexpected errors
                if self._detect_safety_filter(messages, extracted_content or "", finish_reason, None, getattr(self, 'client_type', 'unknown')):
                    print(f"❌ Content prohibited in unexpected error: {error_str[:200]}")
                    
                    # If we're in multi-key mode and haven't tried the main key yet
                    if (self._multi_key_mode and not main_key_attempted and getattr(self, 'original_api_key', None) and getattr(self, 'original_model', None)):
                        main_key_attempted = True
                        try:
                            retry_res = self._retry_with_main_key(messages, temperature, max_tokens, max_completion_tokens, context)
                            if retry_res:
                                content, fr = retry_res
                                return content, fr
                        except Exception:
                            pass
                    
                    # Fall through to normal error handling
                    print(f"❌ Content prohibited - not retrying")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for retryable server errors
                retryable_server_errors = ["500", "502", "503", "504", "internal server error", "bad gateway", "service unavailable", "gateway timeout"]
                if any(err in error_str for err in retryable_server_errors):
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = self._compute_backoff(attempt, base_delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Server error - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue  # Retry the attempt
                
                # Check for other transient errors with exponential backoff
                transient_errors = ["connection reset", "connection aborted", "connection error", "network error", "timeout", "timed out"]
                if any(err in error_str for err in transient_errors):
                    if attempt < internal_retries - 1:
                        # Use a slightly less aggressive backoff for transient errors
                        delay = self._compute_backoff(attempt, base_delay/2, 30)  # Max 30 seconds
                        
                        print(f"🔄 Transient error - retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue  # Retry the attempt
                
                # If we get here, either we've exhausted retries or it's a non-retryable error
                if attempt >= internal_retries - 1:
                    print(f"❌ Unexpected error - final attempt failed, returning fallback")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                else:
                    # For other unexpected errors, try again with a short delay
                    delay = self._compute_backoff(attempt, base_delay/4, 15)  # Short delay
                    print(f"🔄 Unexpected error - retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries}): {str(e)[:100]}")
                    
                    wait_start = time.time()
                    while time.time() - wait_start < delay:
                        if self._cancelled:
                            raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                        time.sleep(0.5)
                    continue  # Retry the attempt

                    
    def _retry_with_main_key(self, messages, temperature=None, max_tokens=None, 
                            max_completion_tokens=None, context=None,
                            request_id=None, image_data=None) -> Optional[Tuple[str, Optional[str]]]: 
        """
        Unified retry method for both text and image requests with main/fallback keys.
        Pass image_data=None for text requests, or image bytes/base64 for image requests.
        Returns None when fallbacks are disabled.
        """
        # Determine if this is an image request
        is_image_request = image_data is not None
        
        # THREAD-SAFE RECURSION CHECK: Use thread-local storage
        tls = self._get_thread_local_client()
        
        # Check if THIS THREAD is already in a retry (unified check for both text and image)
        retry_flag = 'in_image_retry' if image_data else 'in_retry'
        if getattr(tls, retry_flag, False):
            retry_type = "IMAGE " if image_data else ""
            print(f"[{retry_type}MAIN KEY RETRY] Thread {threading.current_thread().name} already in retry, preventing recursion")
            return None
        
        # CHECK: Verify multi-key mode is actually enabled
        if not self._multi_key_mode:
            print(f"[MAIN KEY RETRY] Not in multi-key mode, skipping retry")
            return None
        
        # CHECK: Multi-key mode is already verified above via self._multi_key_mode
        # DO NOT gate main-GUI-key retry on fallback toggle; only use toggle for additional fallback keys
        use_fallback_keys = os.getenv('USE_FALLBACK_KEYS', '0') == '1'
        
        # CHECK: Verify we have the necessary attributes
        if not (hasattr(self, 'original_api_key') and 
                hasattr(self, 'original_model') and
                self.original_api_key and 
                self.original_model):
            print(f"[MAIN KEY RETRY] Missing original key/model attributes, skipping retry")
            return None
        
        # Mark THIS THREAD as being in retry
        setattr(tls, retry_flag, True)
        
        try:
            fallback_keys = []
            
            # FIRST: Always add the MAIN GUI KEY as the first fallback
            fallback_keys.append({
                'api_key': self.original_api_key,
                'model': self.original_model,
                'label': 'MAIN GUI KEY'
            })
            print(f"[MAIN KEY RETRY] Using main GUI key with model: {self.original_model}")
            
            # Add configured fallback keys only if toggle is enabled
            fallback_keys_json = os.getenv('FALLBACK_KEYS', '[]')
            
            if use_fallback_keys and fallback_keys_json != '[]':
                try:
                    configured_fallbacks = json.loads(fallback_keys_json)
                    print(f"[DEBUG] Loaded {len(configured_fallbacks)} fallback keys from environment")
                    for fb in configured_fallbacks:
                        fallback_keys.append({
                            'api_key': fb.get('api_key'),
                            'model': fb.get('model'),
                            'google_credentials': fb.get('google_credentials'),
                            'azure_endpoint': fb.get('azure_endpoint'),
                            'google_region': fb.get('google_region'),
                            'label': 'FALLBACK KEY'
                        })
                except Exception as e:
                    print(f"[DEBUG] Failed to parse FALLBACK_KEYS: {e}")
            elif not use_fallback_keys:
                print("[MAIN KEY RETRY] Fallback keys toggle is OFF — will try main GUI key only")
            
            print(f"[MAIN KEY RETRY] Total keys to try: {len(fallback_keys)}")
            
            # Try each fallback key in the list
            max_attempts = min(len(fallback_keys), self._get_max_retries())
            for idx, fallback_data in enumerate(fallback_keys[:max_attempts]):
                label = fallback_data.get('label', 'Fallback')
                fallback_key = fallback_data.get('api_key')
                fallback_model = fallback_data.get('model')
                fallback_google_creds = fallback_data.get('google_credentials')
                fallback_azure_endpoint = fallback_data.get('azure_endpoint')
                fallback_google_region = fallback_data.get('google_region')
                
                print(f"[{label} {idx+1}/{max_attempts}] Trying {fallback_model}")
                print(f"[{label} {idx+1}] Failed multi-key model was: {self.model}")
                
                try:
                    # Create a new temporary UnifiedClient instance with the fallback key
                    temp_client = UnifiedClient(
                        api_key=fallback_key,  
                        model=fallback_model,   
                        output_dir=self.output_dir
                    )
                    
                    # Set key-specific credentials for the temp client
                    if fallback_google_creds:
                        temp_client.current_key_google_creds = fallback_google_creds
                        temp_client.google_creds_path = fallback_google_creds
                        print(f"[{label} {idx+1}] Using fallback Google credentials: {os.path.basename(fallback_google_creds)}")
                    
                    if fallback_google_region:
                        temp_client.current_key_google_region = fallback_google_region
                        print(f"[{label} {idx+1}] Using fallback Google region: {fallback_google_region}")
                    
                    if fallback_azure_endpoint:
                        temp_client.current_key_azure_endpoint = fallback_azure_endpoint
                        # Set up Azure-specific configuration
                        temp_client.is_azure = True
                        temp_client.azure_endpoint = fallback_azure_endpoint
                        temp_client.azure_api_version = os.getenv('AZURE_API_VERSION', '2024-08-01-preview')
                        print(f"[{label} {idx+1}] Using fallback Azure endpoint: {fallback_azure_endpoint}")
                        print(f"[{label} {idx+1}] Azure API version: {temp_client.azure_api_version}")

                    # Don't override with main client's base_url if we have fallback Azure endpoint
                    if hasattr(self, 'base_url') and self.base_url and not fallback_azure_endpoint:
                        temp_client.base_url = self.base_url
                        temp_client.openai_base_url = self.base_url
                        
                    if hasattr(self, 'api_version') and not fallback_azure_endpoint:
                        temp_client.api_version = self.api_version
                        
                    # Only inherit Azure settings if fallback doesn't have its own Azure endpoint
                    if hasattr(self, 'is_azure') and self.is_azure and not fallback_azure_endpoint:
                        temp_client.is_azure = self.is_azure
                        temp_client.azure_endpoint = getattr(self, 'azure_endpoint', None)
                        temp_client.azure_api_version = getattr(self, 'azure_api_version', '2024-08-01-preview')
                        
                    # Force the client to reinitialize with Azure settings
                    temp_client._setup_client()
                    
                    # FORCE single-key mode after initialization
                    temp_client._multi_key_mode = False
                    temp_client.use_multi_keys = False
                    temp_client.key_identifier = f"{label} ({fallback_model})"
                    temp_client._is_retry_client = True
                    
                    # The client should already be set up from __init__, but verify
                    if not hasattr(temp_client, 'client_type') or temp_client.client_type is None:
                        temp_client.api_key = fallback_key
                        temp_client.model = fallback_model
                        temp_client._setup_client()
                    
                    # Copy relevant state BUT NOT THE CANCELLATION FLAG
                    temp_client.context = context
                    temp_client._cancelled = False
                    temp_client._in_cleanup = False
                    temp_client.current_session_context = self.current_session_context
                    temp_client.conversation_message_count = self.conversation_message_count
                    temp_client.request_timeout = self.request_timeout
                    
                    print(f"[{label} {idx+1}] Created temp client with model: {temp_client.model}")
                    print(f"[{label} {idx+1}] Multi-key mode: {temp_client._multi_key_mode}")
                    
                    # Get file names for response tracking
                    payload_name, response_name = self._get_file_names(messages, context=context)
                    
                    request_type = "image " if image_data else ""
                    print(f"[{label} {idx+1}] Sending {request_type}request...")
                    
                    # Use unified internal method to avoid nested retry loops
                    result = temp_client._send_internal(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_completion_tokens=max_completion_tokens,
                        context=context,
                        retry_reason=f"{request_type.replace(' ', '')}{label.lower().replace(' ', '_')}_{idx+1}",
                        request_id=request_id,
                        image_data=image_data
                    )
                    
                    # Check the result
                    if result and isinstance(result, tuple):
                        content, finish_reason = result
                        
                        # Check if content is an error message
                        if content and "[AI RESPONSE UNAVAILABLE]" in content:
                            print(f"[{label} {idx+1}] ❌ Got error message: {content}")
                            continue
                        
                        # Check if content is valid - FIX: Add None check
                        if content and self._safe_len(content, "main_key_retry_content") > 50:
                            print(f"[{label} {idx+1}] ✅ SUCCESS! Got content of length: {len(content)}")
                            self._save_response(content, response_name)
                            return content, finish_reason
                        else:
                            print(f"[{label} {idx+1}] ❌ Content too short or empty: {len(content) if content else 0} chars")
                            continue
                    else:
                        print(f"[{label} {idx+1}] ❌ Unexpected result type: {type(result)}")
                        continue
                        
                except UnifiedClientError as e:
                    if e.error_type == "cancelled":
                        print(f"[{label} {idx+1}] Operation was cancelled during retry")
                        return None
                    
                    error_str = str(e).lower()
                    if ("azure" in error_str and "content" in error_str) or e.error_type == "prohibited_content":
                        print(f"[{label} {idx+1}] ❌ Content filter error: {str(e)[:100]}")
                        continue
                    
                    print(f"[{label} {idx+1}] ❌ UnifiedClientError: {str(e)[:200]}")
                    continue
                    
                except Exception as e:
                    print(f"[{label} {idx+1}] ❌ Exception: {str(e)[:200]}")
                    continue
            
            print(f"[MAIN KEY RETRY] ❌ All {max_attempts} fallback keys failed")
            return None
            
        finally:
            # ALWAYS clear the thread-local flag
            setattr(tls, retry_flag, False)
    
    def _try_fallback_keys_direct(self, messages, temperature=None, max_tokens=None, 
                                  max_completion_tokens=None, context=None,
                                  request_id=None, image_data=None) -> Optional[Tuple[str, Optional[str]]]: 
        """
        Try fallback keys directly in single-key mode without retrying main key.
        Used when fallback keys are enabled in single-key mode.
        """
        # Check if fallback keys are enabled
        use_fallback_keys = os.getenv('USE_FALLBACK_KEYS', '0') == '1'
        if not use_fallback_keys:
            print(f"[FALLBACK DIRECT] Fallback keys not enabled, skipping")
            return None
        
        # Load fallback keys from environment
        fallback_keys_json = os.getenv('FALLBACK_KEYS', '[]')
        if fallback_keys_json == '[]':
            print(f"[FALLBACK DIRECT] No fallback keys configured")
            return None
        
        try:
            configured_fallbacks = json.loads(fallback_keys_json)
            print(f"[FALLBACK DIRECT] Loaded {len(configured_fallbacks)} fallback keys")
            
            # Try each fallback key
            max_attempts = min(len(configured_fallbacks), 3)  # Limit attempts
            for idx, fb in enumerate(configured_fallbacks[:max_attempts]):
                fallback_key = fb.get('api_key')
                fallback_model = fb.get('model')
                fallback_google_creds = fb.get('google_credentials')
                fallback_azure_endpoint = fb.get('azure_endpoint')
                fallback_google_region = fb.get('google_region')
                fallback_azure_api_version = fb.get('azure_api_version')
                
                if not fallback_key or not fallback_model:
                    print(f"[FALLBACK DIRECT {idx+1}] Invalid key data, skipping")
                    continue
                
                print(f"[FALLBACK DIRECT {idx+1}/{max_attempts}] Trying {fallback_model}")
                
                try:
                    # Create temporary client for fallback key
                    temp_client = UnifiedClient(
                        api_key=fallback_key,  
                        model=fallback_model,   
                        output_dir=self.output_dir
                    )
                    
                    # Set key-specific credentials
                    if fallback_google_creds:
                        temp_client.current_key_google_creds = fallback_google_creds
                        temp_client.google_creds_path = fallback_google_creds
                        print(f"[FALLBACK DIRECT {idx+1}] Using Google credentials: {os.path.basename(fallback_google_creds)}")
                    
                    if fallback_google_region:
                        temp_client.current_key_google_region = fallback_google_region
                        print(f"[FALLBACK DIRECT {idx+1}] Using Google region: {fallback_google_region}")
                    
                    if fallback_azure_endpoint:
                        temp_client.current_key_azure_endpoint = fallback_azure_endpoint
                        # Set up Azure-specific configuration
                        temp_client.is_azure = True
                        temp_client.azure_endpoint = fallback_azure_endpoint
                        # Use per-key Azure API version, fallback to environment, then default
                        temp_client.azure_api_version = fallback_azure_api_version or os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
                        print(f"[FALLBACK DIRECT {idx+1}] Using Azure endpoint: {fallback_azure_endpoint}")
                        print(f"[FALLBACK DIRECT {idx+1}] Azure API version: {temp_client.azure_api_version}")
                    
                    # Force single-key mode
                    temp_client._multi_key_mode = False
                    temp_client.key_identifier = f"FALLBACK KEY ({fallback_model})"
                    temp_client._is_retry_client = True
                    
                    # Setup the client
                    temp_client._setup_client()
                    
                    # Copy relevant state
                    temp_client.context = context
                    temp_client._cancelled = False
                    temp_client._in_cleanup = False
                    temp_client.current_session_context = self.current_session_context
                    temp_client.conversation_message_count = self.conversation_message_count
                    temp_client.request_timeout = self.request_timeout
                    
                    print(f"[FALLBACK DIRECT {idx+1}] Sending request...")
                    
                    # Use internal method to avoid nested retry loops
                    result = temp_client._send_internal(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_completion_tokens=max_completion_tokens,
                        context=context,
                        retry_reason=f"single_key_fallback_{idx+1}",
                        request_id=request_id,
                        image_data=image_data
                    )
                    
                    # Check the result
                    if result and isinstance(result, tuple):
                        content, finish_reason = result
                        
                        # Check if content is valid
                        if content and "[AI RESPONSE UNAVAILABLE]" not in content and len(content) > 50:
                            print(f"[FALLBACK DIRECT {idx+1}] ✅ SUCCESS! Got content of length: {len(content)}")
                            return content, finish_reason
                        else:
                            print(f"[FALLBACK DIRECT {idx+1}] ❌ Content too short or error: {len(content) if content else 0} chars")
                            continue
                    else:
                        print(f"[FALLBACK DIRECT {idx+1}] ❌ Unexpected result type: {type(result)}")
                        continue
                        
                except Exception as e:
                    print(f"[FALLBACK DIRECT {idx+1}] ❌ Failed: {e}")
                    continue
            
            print(f"[FALLBACK DIRECT] All fallback keys failed")
            return None
            
        except Exception as e:
            print(f"[FALLBACK DIRECT] Failed to parse fallback keys: {e}")
            return None
    
    # Image handling methods
    def send_image(self, messages: List[Dict[str, Any]], image_data: Any,
                  temperature: Optional[float] = None, 
                  max_tokens: Optional[int] = None,
                  max_completion_tokens: Optional[int] = None,
                  context: str = 'image_translation') -> Tuple[str, str]:
        """Backwards-compatible public API; now delegates to unified _send_core."""
        return self._send_core(messages, temperature, max_tokens, max_completion_tokens, context, image_data=image_data)

    def _send_image_internal(self, messages: List[Dict[str, Any]], image_data: Any,
                            temperature: Optional[float] = None, 
                            max_tokens: Optional[int] = None,
                            max_completion_tokens: Optional[int] = None,
                            context: str = 'image_translation',
                            retry_reason: Optional[str] = None, 
                            request_id=None) -> Tuple[str, str]:
        """
        Image send internal - backwards compatibility wrapper
        """
        return self._send_internal(
            messages, temperature, max_tokens, max_completion_tokens,
            context or 'image_translation', retry_reason, request_id, image_data=image_data
        )
        
    def _prepare_image_messages(self, messages: List[Dict[str, Any]], image_data: Any) -> List[Dict[str, Any]]:
        """
        Helper method to prepare messages with embedded image for providers that accept image_url parts
        """
        embedded_messages = []
        # Prepare base64 string
        try:
            if isinstance(image_data, (bytes, bytearray)):
                b64 = base64.b64encode(image_data).decode('ascii')
            else:
                b64 = str(image_data)
        except Exception:
            b64 = str(image_data)
        
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, list):
                    new_parts = list(content)
                    new_parts.append(image_part)
                    embedded_messages.append({"role": "user", "content": new_parts})
                else:
                    embedded_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content},
                            image_part
                        ]
                    })
            else:
                embedded_messages.append(msg)
        
        if not any(m.get('role') == 'user' for m in embedded_messages):
            embedded_messages.append({"role": "user", "content": [image_part]})
        
        return embedded_messages
    
    def _retry_image_with_main_key(self, messages, image_data, temperature=None, max_tokens=None, 
                                   max_completion_tokens=None, context=None, request_id=None) -> Optional[Tuple[str, Optional[str]]]:
        """
        Image retry method - backwards compatibility wrapper
        """
        return self._retry_with_main_key(
            messages, temperature, max_tokens, max_completion_tokens,
            context or 'image_translation', request_id, image_data=image_data
        )
 
    def reset_conversation_for_new_context(self, new_context):
        """Reset conversation state when context changes"""
        with self._model_lock:
            self.current_session_context = new_context
            self.conversation_message_count = 0
            self.pattern_counts.clear()
            self.last_pattern = None
        
        logger.info(f"Reset conversation state for new context: {new_context}")
    
    def _apply_pure_reinforcement(self, messages):
        """Apply PURE frequency-based reinforcement pattern"""
        
        # DISABLE in batch mode
        #if os.getenv('BATCH_TRANSLATION', '0') == '1':
        #    return messages
        
        # Skip if not enough messages
        if self.conversation_message_count < 4:
            return messages
        
        # Create pattern from last 2 user messages
        if len(messages) >= 2:
            pattern = []
            for msg in messages[-2:]:
                if msg.get('role') == 'user':
                    content = msg['content']
                    pattern.append(len(content))
            
            if len(pattern) >= 2:
                pattern_key = f"reinforcement_{pattern[0]}_{pattern[1]}"
                
                # MICROSECOND LOCK: When modifying pattern_counts
                with self._model_lock:
                    self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1
                    count = self.pattern_counts[pattern_key]
                
                # Just track patterns, NO PROMPT INJECTION
                if count >= 3:
                    logger.info(f"Pattern {pattern_key} detected (count: {count})")
                    # NO [PATTERN REINFORCEMENT ACTIVE] - KEEP IT GONE
        
        return messages
    
    def _validate_and_clean_messages(self, messages):
        """Validate and clean messages, removing None entries and fixing content issues"""
        if messages is None:
            return []
        
        cleaned_messages = []
        for msg in messages:
            if msg is None:
                continue
            
            # Ensure the message is a dict
            if not isinstance(msg, dict):
                continue
            
            # Ensure content is not None
            if msg.get('content') is None:
                msg = dict(msg)  # Make a copy
                msg['content'] = ''
            
            cleaned_messages.append(msg)
        
        return cleaned_messages
    
    def _validate_request(self, messages, max_tokens=None):
        """Validate request parameters before sending"""
        # Clean messages first
        messages = self._validate_and_clean_messages(messages)
        
        if not messages:
            return False, "Empty messages list"
        
        # Check message content isn't empty - FIX: Add None checks
        total_chars = 0
        for msg in messages:
            if msg is not None and msg.get('role') == 'user':
                content = msg.get('content', '')
                if content is not None:
                    total_chars += len(str(content))
        if total_chars == 0:
            return False, "Empty request content"
        
        # Handle None max_tokens
        if max_tokens is None:
            max_tokens = getattr(self, 'max_tokens', 8192)  # Use instance default or 8192
        
        # Estimate tokens (rough approximation)
        estimated_tokens = total_chars / 4
        if estimated_tokens > max_tokens * 2:
            print(f"Request might be too long: ~{estimated_tokens} tokens vs {max_tokens} max")
        
        # Check for valid roles
        valid_roles = {'system', 'user', 'assistant'}
        for msg in messages:
            if msg.get('role') not in valid_roles:
                return False, f"Invalid role: {msg.get('role')}"
        
        return True, None
    
    def _track_stats(self, context, success, error_type=None, response_time=None):
        """Track API call statistics"""
        self.stats['total_requests'] += 1
        
        if not success:
            self.stats['empty_results'] += 1
            error_key = f"{getattr(self, 'client_type', 'unknown')}_{context}_{error_type}"
            self.stats['errors'][error_key] = self.stats['errors'].get(error_key, 0) + 1
        
        if response_time:
            self.stats['response_times'].append(response_time)
        
        # Save stats periodically
        if self.stats['total_requests'] % 10 == 0:
            self._save_stats()
    
    def _save_stats(self):
        """Save statistics to file"""
        stats_file = "api_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Failed to save stats: {e}")
    
    def _save_failed_request(self, messages, error, context, response=None):
        """Save failed requests for debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_dir = "Payloads/failed_requests"
        os.makedirs(failed_dir, exist_ok=True)
        
        failure_data = {
            'timestamp': timestamp,
            'context': context,
            'error': str(error),
            'error_type': type(error).__name__,
            'messages': messages,
            'model': getattr(self, 'model', None),
            'client_type': getattr(self, 'client_type', None),
            'response': str(response) if response else None,
            'traceback': traceback.format_exc()
        }
        
        filename = f"{failed_dir}/failed_{context}_{getattr(self, 'client_type', 'unknown')}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(failure_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved failed request to: {filename}")
    
    def _handle_empty_result(self, messages, context, error_info):
        """Handle empty results with context-aware fallbacks"""
        print(f"Handling empty result for context: {context}, error: {error_info}")
        
        # Log detailed error information for debugging
        if isinstance(error_info, dict):
            error_type = error_info.get('error', 'unknown')
            error_details = error_info.get('details', '')
        else:
            error_type = str(error_info)
            error_details = ''
        
        # Check if this is an extraction failure vs actual empty response
        is_extraction_failure = 'extract' in error_type.lower() or 'parse' in error_type.lower()
        
        if context == 'glossary':
            # For glossary, we might have partial data in error_info
            if is_extraction_failure and isinstance(error_info, dict):
                # Check if raw response is available
                raw_response = error_info.get('raw_response', '')
                if raw_response and 'character' in str(raw_response):
                    # Log that data exists but couldn't be extracted
                    print("⚠️ Glossary data exists in response but extraction failed!")
                    print("   Consider checking response extraction for this provider")
            
            # Return empty but valid JSON
            return "[]"
            
        elif context == 'translation':
            # Extract the original text and return it with a marker
            original_text = self._extract_user_content(messages)
            
            # Add more specific error info if available
            if is_extraction_failure:
                return f"[EXTRACTION FAILED - ORIGINAL TEXT PRESERVED]\n{original_text}"
            elif 'rate' in error_type.lower():
                return f"[RATE LIMITED - ORIGINAL TEXT PRESERVED]\n{original_text}"
            elif 'safety' in error_type.lower() or 'prohibited' in error_type.lower():
                return f"[CONTENT BLOCKED - ORIGINAL TEXT PRESERVED]\n{original_text}"
            else:
                return f"[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]\n{original_text}"
                
        elif context == 'image_translation':
            # Provide more specific error messages for image translation
            if 'size' in error_type.lower():
                return "[IMAGE TOO LARGE - TRANSLATION FAILED]"
            elif 'format' in error_type.lower():
                return "[UNSUPPORTED IMAGE FORMAT - TRANSLATION FAILED]"
            elif is_extraction_failure:
                return "[RESPONSE EXTRACTION FAILED]"
            else:
                return "[IMAGE TRANSLATION FAILED]"
                
        elif context == 'manga':
            # Add manga-specific handling
            return "[MANGA TRANSLATION FAILED]"
            
        elif context == 'metadata':
            # For metadata extraction
            return "{}"
            
        else:
            # Generic fallback with error type
            if is_extraction_failure:
                return "[RESPONSE EXTRACTION FAILED]"
            elif 'rate' in error_type.lower():
                return "[RATE LIMITED - PLEASE RETRY]"
            else:
                return "[AI RESPONSE UNAVAILABLE]"

    def _extract_response_text(self, response, provider=None, **kwargs):
        """
        Universal response text extraction that works across all providers.
        Includes enhanced OpenAI-specific handling and proper Gemini support.
        """
        result = ""
        finish_reason = 'stop'
        
        # Determine provider if not specified
        if provider is None:
            provider = self.client_type
        
        print(f"   🔍 Extracting text from {provider} response...")
        print(f"   🔍 Response type: {type(response)}")
        
        # Handle UnifiedResponse objects
        if isinstance(response, UnifiedResponse):
            # Check if content is a string (even if empty)
            if response.content is not None and isinstance(response.content, str):
                # Always return the content from UnifiedResponse
                if len(response.content) > 0:
                    print(f"   ✅ Got text from UnifiedResponse.content: {len(response.content)} chars")
                else:
                    print(f"   ⚠️ UnifiedResponse has empty content (finish_reason: {response.finish_reason})")
                return response.content, response.finish_reason or 'stop'
            elif response.error_details:
                print(f"   ⚠️ UnifiedResponse has error_details: {response.error_details}")
                return "", response.finish_reason or 'error'
            else:
                # Only try to extract from raw_response if content is actually None
                print(f"   ⚠️ UnifiedResponse.content is None, checking raw_response...")
                if hasattr(response, 'raw_response') and response.raw_response:
                    print(f"   🔍 Found raw_response, attempting extraction...")
                    response = response.raw_response
                else:
                    print(f"   ⚠️ No raw_response found")
                    return "", 'error'
        
        # ========== GEMINI-SPECIFIC HANDLING ==========
        if provider == 'gemini':
            print(f"   🔍 [Gemini] Attempting specialized extraction...")
            
            # Check for Gemini-specific response structure
            if hasattr(response, 'candidates'):
                print(f"   🔍 [Gemini] Found candidates attribute")
                if response.candidates:
                    candidate = response.candidates[0]
                    
                    # Check finish reason
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason).lower()
                        print(f"   🔍 [Gemini] Finish reason: {finish_reason}")
                        
                        # Map Gemini finish reasons
                        if 'max_tokens' in finish_reason:
                            finish_reason = 'length'
                        elif 'safety' in finish_reason or 'blocked' in finish_reason:
                            finish_reason = 'content_filter'
                        elif 'stop' in finish_reason:
                            finish_reason = 'stop'
                    
                    # Extract content from candidate
                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        print(f"   🔍 [Gemini] Content object: {content}")
                        print(f"   🔍 [Gemini] Content type: {type(content)}")
                        print(f"   🔍 [Gemini] Content attributes: {[attr for attr in dir(content) if not attr.startswith('_')][:10]}")
                        
                        # NEW: Try to access content as string directly first
                        try:
                            content_str = str(content)
                            if content_str and len(content_str) > 20 and 'role=' not in content_str:
                                print(f"   ✅ [Gemini] Got content from string conversion: {len(content_str)} chars")
                                return content_str, finish_reason
                        except Exception as e:
                            print(f"   ⚠️ [Gemini] String conversion failed: {e}")
                        
                        # Content might have parts - FIX: Add None check for parts
                        if hasattr(content, 'parts') and content.parts is not None:
                            parts_count = self._safe_len(content.parts, "gemini_content_parts")
                            print(f"   🔍 [Gemini] Found {parts_count} parts in content")
                            text_parts = []
                            
                            for i, part in enumerate(content.parts):
                                part_text = self._extract_part_text(part, provider='gemini', part_index=i+1)
                                if part_text:
                                    text_parts.append(part_text)
                            
                            if text_parts:
                                result = ''.join(text_parts)
                                print(f"   ✅ [Gemini] Extracted from parts: {len(result)} chars")
                                return result, finish_reason
                                
                            else:
                                # NEW: Handle case where parts exist but contain no text
                                parts_count = self._safe_len(content.parts, "gemini_empty_parts")
                                print(f"   ⚠️ [Gemini] Parts found but no text extracted from {parts_count} parts")           
                                # Don't return here, try other methods
                        
                        # Try direct text access on content
                        elif hasattr(content, 'text'):
                            if content.text:
                                print(f"   ✅ [Gemini] Got text from content.text: {len(content.text)} chars")
                                return content.text, finish_reason
                        
                        # NEW: Try accessing raw content data
                        for attr in ['text', 'content', 'data', 'message', 'response']:
                            if hasattr(content, attr):
                                try:
                                    value = getattr(content, attr)
                                    if value and isinstance(value, str) and len(value) > 10:
                                        print(f"   ✅ [Gemini] Got text from content.{attr}: {len(value)} chars")
                                        return value, finish_reason
                                except Exception as e:
                                    print(f"   ⚠️ [Gemini] Failed to get content.{attr}: {e}")
                    
                    # Try to get text directly from candidate
                    if hasattr(candidate, 'text'):
                        if candidate.text:
                            print(f"   ✅ [Gemini] Got text from candidate.text: {len(candidate.text)} chars")
                            return candidate.text, finish_reason
            
            # Alternative Gemini response structure (for native SDK)
            if hasattr(response, 'text'):
                try:
                    # This might be a property that needs to be called
                    text = response.text
                    if text:
                        print(f"   ✅ [Gemini] Got text via response.text property: {len(text)} chars")
                        return text, finish_reason
                except Exception as e:
                    print(f"   ⚠️ [Gemini] Error accessing response.text: {e}")
            
            # Try parts directly on response - FIX: Add None check
            if hasattr(response, 'parts') and response.parts is not None:
                print(f"   🔍 [Gemini] Found parts directly on response")
                text_parts = []
                for i, part in enumerate(response.parts):
                    part_text = self._extract_part_text(part, provider='gemini', part_index=i+1)
                    if part_text:
                        text_parts.append(part_text)
                
                if text_parts:
                    result = ''.join(text_parts)
                    print(f"   ✅ [Gemini] Extracted from direct parts: {len(result)} chars")
                    return result, finish_reason
            
            print(f"   ⚠️ [Gemini] Specialized extraction failed, trying generic methods...")
        
        # ========== ENHANCED OPENAI HANDLING ==========
        elif provider == 'openai':
            print(f"   🔍 [OpenAI] Attempting specialized extraction...")
            
            # Check if it's an OpenAI ChatCompletion object
            if hasattr(response, 'choices') and response.choices is not None:
                choices_count = self._safe_len(response.choices, "openai_response_choices")
                print(f"   🔍 [OpenAI] Found choices attribute, {choices_count} choices")
                
                if response.choices:
                    choice = response.choices[0]
                    
                    # Log choice details
                    print(f"   🔍 [OpenAI] Choice type: {type(choice)}")
                    
                    # Get finish reason
                    if hasattr(choice, 'finish_reason'):
                        finish_reason = choice.finish_reason
                        print(f"   🔍 [OpenAI] Finish reason: {finish_reason}")
                        
                        # Normalize finish reasons
                        if finish_reason == 'max_tokens':
                            finish_reason = 'length'
                        elif finish_reason == 'content_filter':
                            finish_reason = 'content_filter'
                    
                    # Extract message content
                    if hasattr(choice, 'message'):
                        message = choice.message
                        print(f"   🔍 [OpenAI] Message type: {type(message)}")
                        
                        # Check for refusal first
                        if hasattr(message, 'refusal') and message.refusal:
                            print(f"   🚫 [OpenAI] Message was refused: {message.refusal}")
                            return f"[REFUSED]: {message.refusal}", 'content_filter'
                        
                        # Try to get content
                        if hasattr(message, 'content'):
                            content = message.content
                            
                            # Handle None content
                            if content is None:
                                print(f"   ⚠️ [OpenAI] message.content is None")
                                
                                # Check if it's a function call instead
                                if hasattr(message, 'function_call'):
                                    print(f"   🔍 [OpenAI] Found function_call instead of content")
                                    return "", 'function_call'
                                elif hasattr(message, 'tool_calls'):
                                    print(f"   🔍 [OpenAI] Found tool_calls instead of content")
                                    return "", 'tool_call'
                                else:
                                    print(f"   ⚠️ [OpenAI] No content, refusal, or function calls found")
                                    return "", finish_reason or 'error'
                            
                            # Handle empty string content
                            elif content == "":
                                print(f"   ⚠️ [OpenAI] message.content is empty string")
                                if finish_reason == 'length':
                                    print(f"   ⚠️ [OpenAI] Empty due to length limit (tokens too low)")
                                return "", finish_reason or 'error'
                            
                            # Valid content found
                            else:
                                print(f"   ✅ [OpenAI] Got content: {len(content)} chars")
                                return content, finish_reason
                        
                        # Try alternative attributes
                        elif hasattr(message, 'text'):
                            print(f"   🔍 [OpenAI] Trying message.text...")
                            if message.text:
                                print(f"   ✅ [OpenAI] Got text: {len(message.text)} chars")
                                return message.text, finish_reason
                        
                        # Try dict access if message is dict-like
                        elif hasattr(message, 'get'):
                            content = message.get('content') or message.get('text')
                            if content:
                                print(f"   ✅ [OpenAI] Got content via dict access: {len(content)} chars")
                                return content, finish_reason
                        
                        # Log all available attributes for debugging
                        print(f"   ⚠️ [OpenAI] Message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                else:
                    print(f"   ⚠️ [OpenAI] Empty choices array")
                    
                    # Check if there's metadata about why it's empty
                    if hasattr(response, 'model'):
                        print(f"   Model used: {response.model}")
                    if hasattr(response, 'id'):
                        print(f"   Response ID: {response.id}")
                    if hasattr(response, 'usage'):
                        print(f"   Token usage: {response.usage}")
            
            # If OpenAI extraction failed, continue to generic methods
            print(f"   ⚠️ [OpenAI] Specialized extraction failed, trying generic methods...")
        
        # ========== GENERIC EXTRACTION METHODS ==========
        
        # Method 1: Direct text attributes (common patterns)
        text_attributes = ['text', 'content', 'message', 'output', 'response', 'answer', 'reply']
        
        for attr in text_attributes:
            if hasattr(response, attr):
                try:
                    value = getattr(response, attr)
                    if value is not None and isinstance(value, str) and len(value) > 0:
                        result = value
                        print(f"   ✅ Got text from response.{attr}: {len(result)} chars")
                        return result, finish_reason
                except Exception as e:
                    print(f"   ⚠️ Failed to get response.{attr}: {e}")
        
        # Method 2: Common nested patterns
        nested_patterns = [
            # OpenAI/Mistral pattern
            lambda r: r.choices[0].message.content if hasattr(r, 'choices') and r.choices and hasattr(r.choices[0], 'message') and hasattr(r.choices[0].message, 'content') else None,
            # Alternative OpenAI pattern
            lambda r: r.choices[0].text if hasattr(r, 'choices') and r.choices and hasattr(r.choices[0], 'text') else None,
            # Anthropic SDK pattern
            lambda r: r.content[0].text if hasattr(r, 'content') and r.content and hasattr(r.content[0], 'text') else None,
            # Gemini pattern - candidates structure
            lambda r: r.candidates[0].content.parts[0].text if hasattr(r, 'candidates') and r.candidates and hasattr(r.candidates[0], 'content') and hasattr(r.candidates[0].content, 'parts') and r.candidates[0].content.parts else None,
            # Cohere pattern
            lambda r: r.text if hasattr(r, 'text') else None,
            # JSON response pattern
            lambda r: r.get('choices', [{}])[0].get('message', {}).get('content') if isinstance(r, dict) else None,
            lambda r: r.get('content') if isinstance(r, dict) else None,
            lambda r: r.get('text') if isinstance(r, dict) else None,
            lambda r: r.get('output') if isinstance(r, dict) else None,
        ]
        
        for i, pattern in enumerate(nested_patterns):
            try:
                extracted = pattern(response)
                if extracted is not None and isinstance(extracted, str) and len(extracted) > 0:
                    result = extracted
                    print(f"   ✅ Extracted via nested pattern {i+1}: {len(result)} chars")
                    return result, finish_reason
            except Exception as e:
                # Log pattern failures for debugging
                if provider in ['openai', 'gemini'] and i < 4:  # First patterns are provider-specific
                    print(f"   ⚠️ [{provider}] Pattern {i+1} failed: {e}")
        
        # Method 3: String representation extraction (last resort)
        if not result:
            print(f"   🔍 Attempting string extraction as last resort...")
            result = self._extract_from_string(response, provider=provider)
            if result:
                print(f"   🔧 Extracted from string representation: {len(result)} chars")
                return result, finish_reason
        
        # Method 4: AGGRESSIVE GEMINI FALLBACK - Parse response string manually
        if provider == 'gemini' and not result:
            print(f"   🔍 [Gemini] Attempting aggressive manual parsing...")
            try:
                response_str = str(response)
                
                # Look for common patterns in Gemini response strings
                import re
                patterns = [
                    r'text["\']([^"\'].*?)["\']',  # text="content" or text='content'
                    r'text=([^,\)\]]+)',  # text=content
                    r'content["\']([^"\'].*?)["\']',  # content="text"
                    r'>([^<>{},\[\]]+)<',  # HTML-like tags
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response_str, re.DOTALL)
                    for match in matches:
                        if match and len(match.strip()) > 20:
                            # Clean up the match
                            clean_match = match.strip()
                            clean_match = clean_match.replace('\\n', '\n').replace('\\t', '\t')
                            if len(clean_match) > 20:
                                print(f"   🔧 [Gemini] Extracted via regex pattern: {len(clean_match)} chars")
                                return clean_match, finish_reason
                
                # If no patterns match, try to find the largest text block
                words = response_str.split()
                text_blocks = []
                current_block = []
                
                for word in words:
                    if len(word) > 2 and word.isalpha() or any(c.isalpha() for c in word):
                        current_block.append(word)
                    else:
                        if len(current_block) > 5:  # At least 5 words
                            text_blocks.append(' '.join(current_block))
                        current_block = []
                
                if current_block and len(current_block) > 5:
                    text_blocks.append(' '.join(current_block))
                
                if text_blocks:
                    # Return the longest text block
                    longest_block = max(text_blocks, key=len)
                    if len(longest_block) > 50:
                        print(f"   🔧 [Gemini] Extracted longest text block: {len(longest_block)} chars")
                        return longest_block, finish_reason
                
            except Exception as e:
                print(f"   ⚠️ [Gemini] Aggressive parsing failed: {e}")
        
        # Final failure - log detailed debug info
        print(f"   ❌ Failed to extract text from {provider} response")
        
        # Log the full response structure for debugging
        print(f"   🔍 [{provider}] Full response structure:")
        print(f"   Type: {type(response)}")
        
        # Log available attributes
        if hasattr(response, '__dict__'):
            attrs = list(response.__dict__.keys())[:20]
            print(f"   Attributes: {attrs}")
        else:
            attrs = [attr for attr in dir(response) if not attr.startswith('_')][:20]
            print(f"   Dir attributes: {attrs}")
        
        # Try to get any text representation as absolute last resort
        try:
            response_str = str(response)
            if len(response_str) > 100 and len(response_str) < 100000:  # Reasonable size
                print(f"   🔍 Response string representation: {response_str[:500]}...")
        except:
            pass
        
        return "", 'error'


    def _extract_part_text(self, part, provider=None, part_index=None):
        """
        Extract text from a part object (handles various formats).
        Enhanced with provider-specific handling and aggressive extraction.
        """
        if provider == 'gemini' and part_index:
            print(f"   🔍 [Gemini] Part {part_index} type: {type(part)}")
            print(f"   🔍 [Gemini] Part {part_index} attributes: {[attr for attr in dir(part) if not attr.startswith('_')][:10]}")
        
        # Direct text attribute
        if hasattr(part, 'text'):
            try:
                text = part.text
                if text:
                    if provider == 'gemini' and part_index:
                        print(f"   ✅ [Gemini] Part {part_index} has text via direct access: {len(text)} chars")
                    return text
            except Exception as e:
                if provider == 'gemini' and part_index:
                    print(f"   ⚠️ [Gemini] Failed direct access on part {part_index}: {e}")
        
        # NEW: Try direct string conversion of the part
        try:
            part_str = str(part)
            if part_str and len(part_str) > 10 and 'text=' not in part_str.lower():
                if provider == 'gemini' and part_index:
                    print(f"   ✅ [Gemini] Part {part_index} extracted as string: {len(part_str)} chars")
                return part_str
        except Exception as e:
            if provider == 'gemini' and part_index:
                print(f"   ⚠️ [Gemini] Part {part_index} string conversion failed: {e}")
        
        # Use getattr with fallback
        try:
            text = getattr(part, 'text', None)
            if text:
                if provider == 'gemini' and part_index:
                    print(f"   ✅ [Gemini] Part {part_index} has text via getattr: {len(text)} chars")
                return text
        except Exception as e:
            if provider == 'gemini' and part_index:
                print(f"   ⚠️ [Gemini] Failed getattr on part {part_index}: {e}")
        
        # String representation extraction
        part_str = str(part)
        
        if provider == 'gemini' and part_index:
            print(f"   🔍 [Gemini] Part {part_index} string representation length: {len(part_str)}")
        
        if 'text=' in part_str or 'text":' in part_str:
            import re
            patterns = [
                r'text="""(.*?)"""',  # Triple quotes (common in Gemini)
                r'text="([^"]*(?:\\.[^"]*)*)"',  # Double quotes with escaping
                r"text='([^']*(?:\\.[^']*)*)'",  # Single quotes
                r'text=([^,\)]+)',  # Unquoted text (last resort)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, part_str, re.DOTALL)
                if match:
                    text = match.group(1)
                    # Unescape common escape sequences
                    text = text.replace('\\n', '\n')
                    text = text.replace('\\t', '\t')
                    text = text.replace('\\r', '\r')
                    text = text.replace('\\"', '"')
                    text = text.replace("\\'", "'")
                    text = text.replace('\\\\', '\\')
                    
                    if provider == 'gemini' and part_index:
                        #print(f"   🔧 [Gemini] Part {part_index} extracted via regex pattern: {len(text)} chars")
                        pass
                    
                    return text
        
        # Part is itself a string
        if isinstance(part, str):
            if provider == 'gemini' and part_index:
                print(f"   ✅ [Gemini] Part {part_index} is a string: {len(part)} chars")
            return part
        
        if provider == 'gemini' and part_index:
            print(f"   ⚠️ [Gemini] Failed string extraction on part {part_index}")
        
        return None


    def _extract_from_string(self, response, provider=None):
        """
        Extract text from string representation of response.
        Enhanced with provider-specific patterns.
        """
        try:
            response_str = str(response)
            import re
            
            # Common patterns in string representations
            patterns = [
                r'text="""(.*?)"""',  # Triple quotes (Gemini often uses this)
                r'text="([^"]*(?:\\.[^"]*)*)"',  # Double quotes
                r"text='([^']*(?:\\.[^']*)*)'",  # Single quotes  
                r'content="([^"]*(?:\\.[^"]*)*)"',  # Content field
                r'content="""(.*?)"""',  # Triple quoted content
                r'"text":\s*"([^"]*(?:\\.[^"]*)*)"',  # JSON style
                r'"content":\s*"([^"]*(?:\\.[^"]*)*)"',  # JSON content
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_str, re.DOTALL)
                if match:
                    text = match.group(1)
                    # Unescape common escape sequences
                    text = text.replace('\\n', '\n')
                    text = text.replace('\\t', '\t')
                    text = text.replace('\\r', '\r')
                    text = text.replace('\\"', '"')
                    text = text.replace("\\'", "'")
                    text = text.replace('\\\\', '\\')
                    
                    if provider == 'gemini':
                        #print(f"   🔧 [Gemini] Extracted from string using pattern: {pattern[:30]}...")
                        pass
                    
                    return text
        except Exception as e:
            if provider == 'gemini':
                print(f"   ⚠️ [Gemini] Error during string extraction: {e}")
            else:
                print(f"   ⚠️ Error during string extraction: {e}")
        
        return None
    
    def _extract_user_content(self, messages):
        """Extract user content from messages"""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return msg.get('content', '')
        return ''
    
    def _get_file_names(self, messages, context=None):
        """Generate appropriate file names based on context
        
        IMPORTANT: File naming must support duplicate detection across chapters
        """
        if context == 'glossary':
            payload_name = f"glossary_payload_{self.conversation_message_count}.json"
            response_name = f"glossary_response_{self.conversation_message_count}.txt"
        elif context == 'translation':
            # Extract chapter info if available - CRITICAL for duplicate detection
            content_str = str(messages)
            # Remove any rolling summary blocks to avoid picking previous chapter numbers
            try:
                content_str = re.sub(r"\[Rolling Summary of Chapter \d+\][\s\S]*?\[End of Rolling Summary\]", "", content_str, flags=re.IGNORECASE)
            except Exception:
                pass
            chapter_match = re.search(r'Chapter (\d+)', content_str)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                # Use standard naming that duplicate detection expects
                payload_name = f"translation_chapter_{chapter_num}_payload.json"
                response_name = f"response_{chapter_num}.html"  # This format is expected by duplicate detection
            else:
                # Check for chunk information
                chunk_match = re.search(r'Chunk (\d+)/(\d+)', str(messages))
                if chunk_match:
                    chunk_num = chunk_match.group(1)
                    total_chunks = chunk_match.group(2)
                    # Extract chapter from fuller context
                    chapter_in_chunk = re.search(r'Chapter (\d+)', str(messages))
                    if chapter_in_chunk:
                        chapter_num = chapter_in_chunk.group(1)
                        payload_name = f"translation_chapter_{chapter_num}_chunk_{chunk_num}_payload.json"
                        response_name = f"response_{chapter_num}_chunk_{chunk_num}.html"
                    else:
                        payload_name = f"translation_chunk_{chunk_num}_of_{total_chunks}_payload.json"
                        response_name = f"response_chunk_{chunk_num}_of_{total_chunks}.html"
                else:
                    payload_name = f"translation_payload_{self.conversation_message_count}.json"
                    response_name = f"response_{self.conversation_message_count}.html"
        else:
            payload_name = f"{context or 'general'}_payload_{self.conversation_message_count}.json"
            response_name = f"{context or 'general'}_response_{self.conversation_message_count}.txt"
        self._last_response_filename = response_name
        return payload_name, response_name
    
    def _save_payload(self, messages, filename, retry_reason=None):
        """Save request payload for debugging with retry reason tracking"""
        
        # Get stable thread directory
        thread_dir = self._get_thread_directory()
        
        # Generate request hash for the filename (to make it unique)
        request_hash = self._get_request_hash(messages)
        
        # Add hash and retry info to filename
        base_name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Include retry reason in filename if provided
        if retry_reason:
            # Sanitize retry reason for filename
            safe_reason = retry_reason.replace(" ", "_").replace("/", "_")[:20]
            unique_filename = f"{base_name}_{timestamp}_{safe_reason}_{request_hash[:6]}{ext}"
        else:
            unique_filename = f"{base_name}_{timestamp}_{request_hash[:6]}{ext}"
        
        filepath = os.path.join(thread_dir, unique_filename)
        
        try:
            # Thread-safe file writing
            with self._file_write_lock:
                thread_name = threading.current_thread().name
                thread_id = threading.current_thread().ident
                
                # Extract chapter info for better tracking
                chapter_info = self._extract_chapter_info(messages)
                
                # Include debug info with retry reason
                debug_info = {
                    'system_prompt_present': any(msg.get('role') == 'system' for msg in messages),
                    'system_prompt_length': 0,
                    'request_hash': request_hash,
                    'thread_name': thread_name,
                    'thread_id': thread_id,
                    'session_id': self.session_id,
                    'chapter_info': chapter_info,
                    'timestamp': datetime.now().isoformat(),
                    'key_identifier': self.key_identifier,
                    'retry_reason': retry_reason,  # Track why this payload was saved
                    'is_retry': retry_reason is not None
                }
                
                for msg in messages:
                    if msg.get('role') == 'system':
                        debug_info['system_prompt_length'] = len(msg.get('content', ''))
                        break
                
                # Write the payload
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'model': getattr(self, 'model', None),
                        'client_type': getattr(self, 'client_type', None),
                        'messages': messages,
                        'timestamp': datetime.now().isoformat(),
                        'debug': debug_info,
                        'key_identifier': getattr(self, 'key_identifier', None),
                        'retry_info': {
                            'reason': retry_reason,
                            'attempt': getattr(self, '_current_retry_attempt', 0),
                            'max_retries': getattr(self, '_max_retries', 7)
                        } if retry_reason else None
                    }, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"[{thread_name}] Saved payload to: {filepath} (reason: {retry_reason or 'initial'})")
                
        except Exception as e:
            print(f"Failed to save payload: {e}")


    def _save_response(self, content: str, filename: str):
        """Save API response with enhanced thread safety and deduplication"""
        if not content or not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
        
        # ONLY save JSON files to Payloads folder
        if not filename.endswith('.json'):
            logger.debug(f"Skipping HTML response save to Payloads: {filename}")
            return
        
        # Get thread-specific directory
        thread_dir = self._get_thread_directory()
        thread_id = threading.current_thread().ident
        
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
            
            # Clean up filename
            safe_filename = os.path.basename(filename)
            base_name, ext = os.path.splitext(safe_filename)
            
            # Create unique filename with thread ID and content hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # Include microseconds
            unique_filename = f"{base_name}_T{thread_id}_{timestamp}_{content_hash}{ext}"
            filepath = os.path.join(thread_dir, unique_filename)
            
            # Get file-specific lock
            file_lock = self._get_file_lock(filepath)
            
            with file_lock:
                # Check if this exact content was already saved (deduplication)
                if self._is_duplicate_file(thread_dir, content_hash):
                    logger.debug(f"Skipping duplicate response save: {content_hash[:8]}")
                    return
                
                # Write atomically with temp file
                temp_filepath = filepath + '.tmp'
                
                try:
                    os.makedirs(thread_dir, exist_ok=True)
                    
                    if filename.endswith('.json'):
                        try:
                            json_content = json.loads(content) if isinstance(content, str) else content
                            with open(temp_filepath, 'w', encoding='utf-8') as f:
                                json.dump(json_content, f, indent=2, ensure_ascii=False)
                        except json.JSONDecodeError:
                            with open(temp_filepath, 'w', encoding='utf-8') as f:
                                f.write(content)
                    else:
                        with open(temp_filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                    
                    # Atomic rename
                    os.replace(temp_filepath, filepath)
                    logger.debug(f"Saved response: {filepath}")
                    
                except Exception as e:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    raise
                    
        except Exception as e:
            print(f"Failed to save response: {e}")

    def _get_file_lock(self, filepath: str) -> RLock:
        """Get or create a lock for a specific file"""
        with self._file_write_locks_lock:
            if filepath not in self._file_write_locks:
                self._file_write_locks[filepath] = RLock()
            return self._file_write_locks[filepath]

    def _is_duplicate_file(self, directory: str, content_hash: str) -> bool:
        """Check if a file with this content hash already exists"""
        try:
            for filename in os.listdir(directory):
                if content_hash in filename and filename.endswith('.json'):
                    return True
        except:
            pass
        return False

    def set_output_filename(self, filename: str):
        """Set the actual output filename for truncation logging
        
        This should be called before sending a request to inform the client
        about the actual chapter output filename (e.g., response_001_Chapter_1.html)
        
        Args:
            filename: The actual output filename that will be created in the book folder
        """
        self._actual_output_filename = filename
        logger.debug(f"Set output filename for truncation logging: {filename}")

    def set_output_directory(self, directory: str):
        """Set the output directory for truncation logs
        
        Args:
            directory: The output directory path (e.g., the book folder)
        """
        self.output_dir = directory
        logger.debug(f"Set output directory: {directory}")
    
    def cancel_current_operation(self):
        """Mark current operation as cancelled
        
        IMPORTANT: Called by send_with_interrupt when timeout occurs
        """
        self._cancelled = True
        self._in_cleanup = True  # Set cleanup flag correctly
        # Show cancellation messages before setting global flag (to avoid circular check)
        print("🛑 Operation cancelled (timeout or user stop)")
        print("🛑 API operation cancelled")
        # Set global cancellation to affect all instances
        self.set_global_cancellation(True)
        # Suppress httpx logging when cancelled
        self._suppress_http_logs()

    def _suppress_http_logs(self):
        """Suppress HTTP and API logging during cancellation"""
        import logging
        # Suppress httpx logs (used by OpenAI client)
        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.WARNING)
        
        # Suppress OpenAI client logs
        openai_logger = logging.getLogger('openai')
        openai_logger.setLevel(logging.WARNING)
        
        # Suppress our own API client logs  
        unified_logger = logging.getLogger('unified_api_client')
        unified_logger.setLevel(logging.WARNING)
    
    def _reset_http_logs(self):
        """Reset HTTP and API logging levels for new operations"""
        import logging
        # Reset httpx logs back to INFO
        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.INFO)
        
        # Reset OpenAI client logs back to INFO
        openai_logger = logging.getLogger('openai')
        openai_logger.setLevel(logging.INFO)
        
        # Reset our own API client logs back to INFO
        unified_logger = logging.getLogger('unified_api_client')
        unified_logger.setLevel(logging.INFO)
    
    def reset_cleanup_state(self):
            """Reset cleanup state for new operations"""
            self._in_cleanup = False
            self._cancelled = False
            # Reset global cancellation flag for new operations
            self.set_global_cancellation(False)
            # Reset logging levels for new operations
            self._reset_http_logs()

    def _send_vertex_model_garden(self, messages, temperature=0.7, max_tokens=None, stop_sequences=None, response_name=None):
        """Send request to Vertex AI Model Garden models (including Claude)"""
        response = None
        try:
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            import google.auth.transport.requests
            import vertexai
            import json
            import os
            import re
            import traceback
            import logging
            
            # Get logger
            logger = logging.getLogger(__name__)
            
            # Import or define UnifiedClientError
            try:
                # Try to import from the module if it exists
                from unified_api_client import UnifiedClientError, UnifiedResponse
            except ImportError:
                # Define them locally if import fails
                class UnifiedClientError(Exception):
                    def __init__(self, message, error_type=None):
                        super().__init__(message)
                        self.error_type = error_type
                
                from dataclasses import dataclass
                @dataclass
                class UnifiedResponse:
                    content: str
                    usage: dict = None
                    finish_reason: str = 'stop'
                    raw_response: object = None
            
            # Import your global stop check function
            try:
                from TranslateKRtoEN import is_stop_requested
            except ImportError:
                # Fallback to checking _cancelled flag
                def is_stop_requested():
                    return self._cancelled
            
            # Use the same credentials as Cloud Vision (comes from GUI config)
            google_creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if not google_creds_path:
                # Try to get from config
                if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
                    google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or \
                                      self.main_gui.config.get('google_cloud_credentials', '')
            
            if not google_creds_path or not os.path.exists(google_creds_path):
                raise ValueError("Google Cloud credentials not found. Please set up credentials.")
            
            # Load credentials with proper scopes
            credentials = service_account.Credentials.from_service_account_file(
                google_creds_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Extract project ID from credentials
            with open(google_creds_path, 'r') as f:
                creds_data = json.load(f)
                project_id = creds_data.get('project_id')
            
            if not project_id:
                raise ValueError("Project ID not found in credentials file")
            
            logger.info(f"Using project ID: {project_id}")
            
            # Parse model name
            model_name = self.model
            if model_name.startswith('vertex_ai/'):
                model_name = model_name[10:]  # Remove "vertex_ai/" prefix
            elif model_name.startswith('vertex/'):
                model_name = model_name[7:]  # Remove "vertex/" prefix
            
            logger.info(f"Using model: {model_name}")
            
            # For Claude models, use the Anthropic SDK with Vertex AI
            if 'claude' in model_name.lower():
                # Import Anthropic exceptions
                try:
                    from anthropic import AnthropicVertex
                    import anthropic
                    import httpx
                except ImportError:
                    raise UnifiedClientError("Anthropic SDK not installed. Run: pip install anthropic")
                
                # Use the region from environment variable (which comes from GUI)
                region = os.getenv('VERTEX_AI_LOCATION', 'us-east5')
                
                # CHECK STOP FLAG
                if is_stop_requested():
                    logger.info("Stop requested, cancelling")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                
                # Initialize Anthropic client for Vertex AI
                client = AnthropicVertex(
                    project_id=project_id,
                    region=region
                )
                
                # Convert messages to Anthropic format
                anthropic_messages = []
                system_prompt = ""
                
                for msg in messages:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    else:
                        anthropic_messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
                
                # Create message with Anthropic client
                kwargs = {
                    "model": model_name,
                    "messages": anthropic_messages,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature,
                }
                
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                if stop_sequences:
                    kwargs["stop_sequences"] = stop_sequences
                
                # CHECK STOP FLAG BEFORE API CALL
                if is_stop_requested():
                    logger.info("Stop requested, cancelling API call")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")

                
                try:
                    message = client.messages.create(**kwargs)
                    
                except httpx.HTTPStatusError as e:
                    # Handle HTTP status errors from the Anthropic SDK
                    status_code = e.response.status_code if hasattr(e.response, 'status_code') else 0
                    error_body = e.response.text if hasattr(e.response, 'text') else str(e)
                    
                    # Check if it's an HTML error page
                    if '<!DOCTYPE html>' in error_body or '<html' in error_body:
                        if '404' in error_body:
                            # Extract the region from the error
                            import re
                            region_match = re.search(r'/locations/([^/]+)/', error_body)
                            bad_region = region_match.group(1) if region_match else region
                            
                            raise UnifiedClientError(
                                f"Invalid region: {bad_region}\n\n"
                                f"This region does not exist. Common regions:\n"
                                f"• us-east5 (for Claude models)\n"
                                f"• us-central1\n"
                                f"• europe-west4\n"
                                f"• asia-southeast1\n\n"
                                f"Please check the region spelling in the text box."
                            )
                        else:
                            raise UnifiedClientError(
                                "Connection error to Vertex AI.\n"
                                "Please check your region and try again."
                            )
                    
                    if status_code == 429:
                        raise UnifiedClientError(
                            f"Quota exceeded for Vertex AI model: {model_name}\n\n"
                            "You need to request quota increase in Google Cloud Console:\n"
                            "1. Go to IAM & Admin → Quotas\n"
                            "2. Search for 'online_prediction_requests_per_base_model'\n"
                            "3. Request increase for your model\n\n"
                            "Or use the model directly with provider's API key instead of Vertex AI."
                        )
                    elif status_code == 404:
                        raise UnifiedClientError(
                            f"Model {model_name} not found in region {region}.\n\n"
                            "Try changing the region in the text box next to Google Cloud Credentials button.\n"
                            "Claude models are typically available in us-east5."
                        )
                    elif status_code == 403:
                        raise UnifiedClientError(
                            f"Permission denied for model {model_name}.\n\n"
                            "Make sure:\n"
                            "1. The model is enabled in Vertex AI Model Garden\n"
                            "2. Your service account has the necessary permissions\n"
                            "3. You have accepted any required terms for Claude models"
                        )
                    elif status_code == 500:
                        raise UnifiedClientError(
                            f"Vertex AI internal error.\n\n"
                            "This is a temporary issue with Google's servers.\n"
                            "Please try again in a few moments."
                        )
                    else:
                        raise UnifiedClientError(f"HTTP {status_code} error")
                        
                except anthropic.APIError as e:
                    # Handle Anthropic-specific API errors
                    error_str = str(e)
                    
                    # Check for HTML in error message
                    if '<!DOCTYPE html>' in error_str or '<html' in error_str:
                        if '404' in error_str:
                            import re
                            region_match = re.search(r'/locations/([^/]+)/', error_str)
                            bad_region = region_match.group(1) if region_match else region
                            
                            raise UnifiedClientError(
                                f"Invalid region: {bad_region}\n\n"
                                f"This region does not exist. Try:\n"
                                f"• us-east5\n"
                                f"• us-central1\n"
                                f"• europe-west4"
                            )
                        else:
                            raise UnifiedClientError("Connection error. Check your region.")
                    
                    if hasattr(e, 'status_code'):
                        status_code = e.status_code
                        if status_code == 429:
                            raise UnifiedClientError(
                                f"Quota exceeded for Vertex AI model: {model_name}\n\n"
                                "Request quota increase in Google Cloud Console."
                            )
                    raise UnifiedClientError(f"API error: {error_str[:200]}")  # Limit error length
                    
                except Exception as e:
                    # Catch any other errors
                    error_str = str(e)
                    
                    # Check if it's an HTML error page
                    if '<!DOCTYPE html>' in error_str or '<html' in error_str:
                        if '404' in error_str:
                            # Extract the region from the error
                            import re
                            region_match = re.search(r'/locations/([^/]+)/', error_str)
                            bad_region = region_match.group(1) if region_match else region
                            
                            raise UnifiedClientError(
                                f"Invalid region: {bad_region}\n\n"
                                f"This region does not exist. Common regions:\n"
                                f"• us-east5 (for Claude models)\n"
                                f"• us-central1\n"
                                f"• europe-west4\n"
                                f"• asia-southeast1\n\n"
                                f"Please check the region spelling in the text box."
                            )
                        else:
                            # Generic HTML error
                            raise UnifiedClientError(
                                "Connection error to Vertex AI.\n"
                                "Please check your region and try again."
                            )
                    elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        raise UnifiedClientError(
                            f"Quota exceeded for Vertex AI model: {model_name}\n\n"
                            "Request quota increase in Google Cloud Console."
                        )
                    elif "404" in error_str or "NOT_FOUND" in error_str:
                        raise UnifiedClientError(
                            f"Model {model_name} not found in region {region}.\n\n"
                            "Try changing the region."
                        )
                    else:
                        # For any other error, show a clean message
                        if len(error_str) > 200:
                            raise UnifiedClientError(f"Vertex AI error: Request failed. Check your region and model name.")
                        else:
                            raise UnifiedClientError(f"Vertex AI error: {error_str}")
                
                # CHECK STOP FLAG AFTER RESPONSE
                if is_stop_requested():
                    logger.info("Stop requested after response, discarding result")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                # Success! Convert response to UnifiedResponse
                print(f"Successfully got response from {region}")
                return UnifiedResponse(
                    content=message.content[0].text if message.content else "",
                    usage={
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "total_tokens": message.usage.input_tokens + message.usage.output_tokens
                    } if hasattr(message, 'usage') else None,
                    finish_reason=message.stop_reason if hasattr(message, 'stop_reason') else 'stop',
                    raw_response=message
                )
            
            else:
                # For Gemini models on Vertex AI, we need to use Vertex AI SDK
                location = os.getenv('VERTEX_AI_LOCATION', 'us-east5')
                
                # Check stop flag before Gemini call
                if is_stop_requested():
                    logger.info("Stop requested, cancelling Vertex AI Gemini request")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                # Initialize Vertex AI
                vertexai.init(project=project_id, location=location, credentials=credentials)
                
                # Import GenerativeModel from vertexai
                from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
                
                # Create model instance
                vertex_model = GenerativeModel(model_name)
                
                # Format messages for Vertex AI Gemini using existing formatter
                formatted_prompt = self._format_prompt(messages, style='gemini')
                
                # Check if safety settings are disabled via config (from GUI)
                disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
                
                # Get thinking budget from environment (though Vertex AI may not support it)
                thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))
                enable_thinking = os.getenv("ENABLE_GEMINI_THINKING", "0") == "1"
                
                # Log configuration
                print(f"\n🔧 Vertex AI Gemini Configuration:")
                print(f"   Model: {model_name}")
                print(f"   Region: {location}")
                print(f"   Project: {project_id}")
                
                # Configure generation parameters using passed parameters
                generation_config_dict = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                }
                
                # Add user-configured anti-duplicate parameters if enabled
                if os.getenv("ENABLE_ANTI_DUPLICATE", "0") == "1":
                    # Get all anti-duplicate parameters from environment
                    if os.getenv("TOP_P"):
                        top_p = float(os.getenv("TOP_P", "1.0"))
                        if top_p < 1.0:  # Only add if not default
                            generation_config_dict["top_p"] = top_p
                    
                    if os.getenv("TOP_K"):
                        top_k = int(os.getenv("TOP_K", "0"))
                        if top_k > 0:  # Only add if not default
                            generation_config_dict["top_k"] = top_k
                    
                    # Note: Vertex AI Gemini may not support all parameters like frequency_penalty
                    # Add only supported parameters
                    if os.getenv("CANDIDATE_COUNT"):
                        candidate_count = int(os.getenv("CANDIDATE_COUNT", "1"))
                        if candidate_count > 1:
                            generation_config_dict["candidate_count"] = candidate_count
                    
                    # Add custom stop sequences if provided
                    custom_stops = os.getenv("CUSTOM_STOP_SEQUENCES", "").strip()
                    if custom_stops:
                        additional_stops = [s.strip() for s in custom_stops.split(",") if s.strip()]
                        if stop_sequences:
                            stop_sequences.extend(additional_stops)
                        else:
                            stop_sequences = additional_stops
                
                if stop_sequences:
                    generation_config_dict["stop_sequences"] = stop_sequences
                
                # Create generation config
                generation_config = GenerationConfig(**generation_config_dict)
                
                # Configure safety settings based on GUI toggle
                safety_settings = None
                if disable_safety:
                    # Import SafetySetting from vertexai
                    from vertexai.generative_models import SafetySetting
                    
                    # Create list of SafetySetting objects (same format as regular Gemini)
                    safety_settings = [
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=HarmBlockThreshold.BLOCK_NONE
                        ),
                    ]
                    # Only log if not stopping
                    if not self._is_stop_requested():
                        print(f"🔒 Vertex AI Gemini Safety Status: DISABLED - All categories set to BLOCK_NONE")
                else:
                    # Only log if not stopping
                    if not self._is_stop_requested():
                        print(f"🔒 Vertex AI Gemini Safety Status: ENABLED - Using default Gemini safety settings")
                    
                # SAVE SAFETY CONFIGURATION FOR VERIFICATION
                if safety_settings:
                    safety_status = "DISABLED - All categories set to BLOCK_NONE"
                    readable_safety = {
                        "HATE_SPEECH": "BLOCK_NONE",
                        "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARASSMENT": "BLOCK_NONE",
                        "DANGEROUS_CONTENT": "BLOCK_NONE",
                        "CIVIC_INTEGRITY": "BLOCK_NONE"
                    }
                else:
                    safety_status = "ENABLED - Using default Gemini safety settings"
                    readable_safety = "DEFAULT"
                
                # Save configuration to file
                config_data = {
                    "type": "VERTEX_AI_GEMINI_REQUEST",
                    "model": model_name,
                    "project_id": project_id,
                    "location": location,
                    "safety_enabled": not disable_safety,
                    "safety_settings": readable_safety,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Save configuration to file with thread isolation
                self._save_gemini_safety_config(config_data, response_name)
            
                # Retry logic
                attempts = self._get_max_retries()
                attempt = 0
                result_text = ""
                
                while attempt < attempts and not result_text:
                    try:
                        # Update max_output_tokens for this attempt
                        generation_config_dict["max_output_tokens"] = max_tokens or 8192
                        generation_config = GenerationConfig(**generation_config_dict)
                        
                        # Only log if not stopping
                        if not self._is_stop_requested():
                            print(f"   📊 Temperature: {temperature}, Max tokens: {max_tokens or 8192}")
                        
                        # Generate content with optional safety settings
                        if safety_settings:
                            response = vertex_model.generate_content(
                                formatted_prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                            )
                        else:
                            response = vertex_model.generate_content(
                                formatted_prompt,
                                generation_config=generation_config
                            )
                        
                        # Extract text from response
                        if response.candidates:
                            for candidate in response.candidates:
                                if candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text'):
                                            result_text += part.text
                        
                        # Check if we got content
                        if result_text and result_text.strip():
                            break
                        else:
                            raise Exception("Empty response from Vertex AI")
                            
                    except Exception as e:
                        print(f"Vertex AI Gemini attempt {attempt+1} failed: {e}")
                        
                        # Check for quota errors
                        error_str = str(e)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            raise UnifiedClientError(
                                f"Quota exceeded for Vertex AI Gemini model: {model_name}\n\n"
                                "Request quota increase in Google Cloud Console."
                            )
                        elif "404" in error_str or "NOT_FOUND" in error_str:
                            raise UnifiedClientError(
                                f"Model {model_name} not found in region {location}.\n\n"
                                "Available Gemini models on Vertex AI:\n"
                                "• gemini-1.5-flash-002\n"
                                "• gemini-1.5-pro-002\n"
                                "• gemini-1.0-pro-002"
                            )
                        
                        # No automatic retry - let higher level handle retries
                        #attempt += 1
                        #if attempt < attempts:
                        #    print(f"❌ Gemini attempt {attempt} failed, no automatic retry")
                        #    break  # Exit the retry loop
                    
                # Check stop flag after response
                if is_stop_requested():
                    logger.info("Stop requested after Vertex AI Gemini response")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                if not result_text:
                    raise UnifiedClientError("All Vertex AI Gemini attempts failed to produce content")
                
                return UnifiedResponse(
                    content=result_text,
                    finish_reason='stop',
                    raw_response=response 
                )
                
        except UnifiedClientError:
            # Re-raise our own errors without modification
            raise
        except Exception as e:
            # Handle any other unexpected errors
            error_str = str(e)
            # Don't print HTML errors
            if '<!DOCTYPE html>' not in error_str and '<html' not in error_str:
                print(f"Vertex AI Model Garden error: {str(e)}")
                print(f"Full traceback: {traceback.format_exc()}")
            raise UnifiedClientError(f"Vertex AI Model Garden error: {str(e)[:200]}")  # Limit length
            
    def _convert_messages_for_vertex(self, messages):
        """Convert OpenAI-style messages to Vertex AI Model Garden format"""
        converted = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # Map roles for Claude in Vertex AI
            if role == 'system':
                converted.append({
                    "role": "system",
                    "content": content
                })
            elif role == 'user':
                converted.append({
                    "role": "user", 
                    "content": content
                })
            elif role == 'assistant':
                converted.append({
                    "role": "assistant",
                    "content": content
                })
        
        return converted

    def _apply_api_call_stagger(self):
        """Stagger API calls to prevent simultaneous requests"""
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        if api_delay <= 0:
            return
        
        thread_name = threading.current_thread().name
        
        # Initialize class-level tracking if needed
        if not hasattr(self.__class__, '_last_api_call_start'):
            self.__class__._last_api_call_start = 0
            self.__class__._api_stagger_lock = threading.Lock()
        
        with self.__class__._api_stagger_lock:
            current_time = time.time()
            
            # Calculate next available slot (ensures exact intervals)
            next_available = self.__class__._last_api_call_start + api_delay
            
            if current_time < next_available:
                sleep_time = next_available - current_time
                # Reserve this slot
                self.__class__._last_api_call_start = next_available
                
                # Check stop flag before logging stagger message
                if not self._is_stop_requested():
                    print(f"⏳ [{thread_name}] Staggering API call by {sleep_time:.1f}s")
                
                # Sleep outside lock
                time.sleep(sleep_time)
                
                # Immediately after stagger completes, indicate what is being sent
                if not self._is_stop_requested():
                    try:
                        tls = self._get_thread_local_client()
                        label = getattr(tls, 'current_request_label', None)
                        if label:
                            print(f"📤 [{thread_name}] Sending {label} to API...")
                    except Exception:
                        pass
            else:
                # This thread gets to go immediately
                self.__class__._last_api_call_start = current_time

    def _get_timeouts(self):
        """Return (connect_timeout, read_timeout) from environment, with sane defaults.
        Respects master toggle ENABLE_HTTP_TUNING (defaults to disabled)."""
        enabled = os.getenv("ENABLE_HTTP_TUNING", "0") == "1"
        if not enabled:
            # Use conservative, very high read timeout to avoid request timeouts (e.g., Gemini)
            connect = 10.0
            read = float(os.getenv("CHUNK_TIMEOUT", "900"))
            return (connect, read)
        connect = float(os.getenv("CONNECT_TIMEOUT", "10"))
        read = float(os.getenv("READ_TIMEOUT", os.getenv("CHUNK_TIMEOUT", "180")))
        return (connect, read)

    def _parse_retry_after(self, value: str) -> int:
        """Parse Retry-After header (seconds or HTTP-date) into seconds."""
        if not value:
            return 0
        value = value.strip()
        if value.isdigit():
            try:
                return max(0, int(value))
            except Exception:
                return 0
        try:
            import email.utils
            dt = email.utils.parsedate_to_datetime(value)
            if dt is None:
                return 0
            now = datetime.now(dt.tzinfo)
            secs = int((dt - now).total_seconds())
            return max(0, secs)
        except Exception:
            return 0

    def _get_session(self, base_url: str):
        """Get or create a thread-local requests.Session for a base_url with connection pooling."""
        tls = self._get_thread_local_client()
        if not hasattr(tls, "session_map"):
            tls.session_map = {}
        session = tls.session_map.get(base_url)
        if session is None:
            session = requests.Session()
            try:
                adapter = HTTPAdapter(
                    pool_connections=int(os.getenv("HTTP_POOL_CONNECTIONS", "20")),
                    pool_maxsize=int(os.getenv("HTTP_POOL_MAXSIZE", "50")),
                    max_retries=Retry(total=0) if Retry is not None else 0
                )
            except Exception:
                adapter = HTTPAdapter(
                    pool_connections=int(os.getenv("HTTP_POOL_CONNECTIONS", "20")),
                    pool_maxsize=int(os.getenv("HTTP_POOL_MAXSIZE", "50"))
                )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            tls.session_map[base_url] = session
        return session

    def _http_request_with_retries(self, method: str, url: str, headers: dict = None, json: dict = None,
                                   expected_status: tuple = (200,), max_retries: int = 3,
                                   provider_name: str = None, use_session: bool = False):
        """
        Generic HTTP requester with standardized retry behavior.
        - Handles cancellation, rate limits (429 with Retry-After), 5xx with backoff, and generic errors.
        - Returns the requests.Response object when a successful status is received.
        """
        api_delay = self._get_send_interval()
        provider = provider_name or "HTTP"
        for attempt in range(max_retries):
            if self._cancelled:
                raise UnifiedClientError("Operation cancelled")
            
            # Toggle to ignore server-provided Retry-After headers
            ignore_retry_after = (os.getenv("ENABLE_HTTP_TUNING", "0") == "1") and (os.getenv("IGNORE_RETRY_AFTER", "0") == "1")
            try:
                if use_session:
                    # Reuse pooled session based on the base URL
                    try:
                        from urllib.parse import urlsplit
                    except Exception:
                        urlsplit = None
                    base_for_session = None
                    if urlsplit is not None:
                        parts = urlsplit(url)
                        base_for_session = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else None
                    session = self._get_session(base_for_session) if base_for_session else requests
                    timeout = self._get_timeouts() if base_for_session else self.request_timeout
                    resp = session.request(method, url, headers=headers, json=json, timeout=timeout)
                else:
                    resp = requests.request(method, url, headers=headers, json=json, timeout=self.request_timeout)
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"{provider} network error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                raise UnifiedClientError(f"{provider} network error: {e}")

            status = resp.status_code
            if status in expected_status:
                return resp

            # Rate limit handling (429)
            if status == 429:
                # Check if indefinite rate limit retry is enabled
                indefinite_retry_enabled = os.getenv("INDEFINITE_RATE_LIMIT_RETRY", "1") == "1"
                
                retry_after_val = resp.headers.get('Retry-After', '')
                retry_secs = self._parse_retry_after(retry_after_val)
                if ignore_retry_after:
                    wait_time = api_delay * 10
                else:
                    wait_time = retry_secs if retry_secs > 0 else api_delay * 10
                
                # Add jitter and cap wait time
                wait_time = min(wait_time + random.uniform(1, 5), 300)  # Max 5 minutes
                
                if indefinite_retry_enabled:
                    # For indefinite retry, don't count against max_retries
                    print(f"{provider} rate limit ({status}), indefinite retry enabled - waiting {wait_time:.1f}s")
                    waited = 0.0
                    while waited < wait_time:
                        if self._cancelled:
                            raise UnifiedClientError("Operation cancelled", error_type="cancelled")
                        time.sleep(0.5)
                        waited += 0.5
                    # Don't increment attempt counter for rate limits when indefinite retry is enabled
                    attempt = max(0, attempt - 1)
                    continue
                elif attempt < max_retries - 1:
                    # Standard retry behavior when indefinite retry is disabled
                    print(f"{provider} rate limit ({status}), waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    waited = 0.0
                    while waited < wait_time:
                        if self._cancelled:
                            raise UnifiedClientError("Operation cancelled", error_type="cancelled")
                        time.sleep(0.5)
                        waited += 0.5
                    continue
                
                # If we reach here, indefinite retry is disabled and we've exhausted max_retries
                raise UnifiedClientError(f"{provider} rate limit: {resp.text}", error_type="rate_limit", http_status=429)

            # Transient server errors with optional Retry-After
            if status in (500, 502, 503, 504) and attempt < max_retries - 1:
                retry_after_val = resp.headers.get('Retry-After', '')
                retry_secs = self._parse_retry_after(retry_after_val)
                if ignore_retry_after:
                    retry_secs = 0
                if retry_secs:
                    sleep_for = retry_secs + random.uniform(0, 1)
                else:
                    base_delay = 5.0
                    sleep_for = min(base_delay * (2 ** attempt) + random.uniform(0, 1), 60.0)
                print(f"{provider} {status} - retrying in {sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})")
                waited = 0.0
                while waited < sleep_for:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled", error_type="cancelled")
                    time.sleep(0.5)
                    waited += 0.5
                continue

            # Other non-success statuses
            if attempt < max_retries - 1:
                print(f"{provider} API error: {status} - {resp.text} (attempt {attempt + 1})")
                time.sleep(api_delay)
                continue
            raise UnifiedClientError(f"{provider} API error: {status} - {resp.text}", http_status=status)

    def _extract_openai_json(self, json_resp: dict):
        """Extract content, finish_reason, and usage from OpenAI-compatible JSON."""
        content = ""
        finish_reason = 'stop'
        choices = json_resp.get('choices', [])
        if choices:
            choice = choices[0]
            finish_reason = choice.get('finish_reason') or 'stop'
            message = choice.get('message')
            if isinstance(message, dict):
                content = message.get('content') or message.get('text') or ""
            elif isinstance(message, str):
                content = message
            else:
                # As a fallback, try 'text' field directly on choice
                content = choice.get('text', "")
        # Normalize finish reasons
        if finish_reason in ['max_tokens', 'max_length']:
            finish_reason = 'length'
        usage = None
        if 'usage' in json_resp:
            u = json_resp['usage'] or {}
            pt = u.get('prompt_tokens', 0)
            ct = u.get('completion_tokens', 0)
            tt = u.get('total_tokens', pt + ct)
            usage = {'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': tt}
        return content, finish_reason, usage

    def _with_sdk_retries(self, provider_name: str, max_retries: int, call):
        """Run an SDK call with standardized retry behavior and error wrapping."""
        api_delay = self._get_send_interval()
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                return call()
            except UnifiedClientError:
                # Already normalized; propagate
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"{provider_name} SDK error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"{provider_name} SDK error after all retries: {e}")
                raise UnifiedClientError(f"{provider_name} SDK error: {e}")

    def _build_openai_headers(self, provider: str, api_key: str, headers: Optional[dict]) -> dict:
        """Construct standard headers for OpenAI-compatible HTTP calls without altering behavior."""
        h = dict(headers) if headers else {}
        # Only set Authorization if not already present and not Azure special-case (Azure handled earlier)
        if 'Authorization' not in h and provider not in ('azure',):
            h['Authorization'] = f'Bearer {api_key}'
        # Ensure content type
        if 'Content-Type' not in h:
            h['Content-Type'] = 'application/json'
        return h

    def _apply_openai_safety(self, provider: str, disable_safety: bool, payload: dict, headers: dict):
        """Apply safety flags for providers that support them (no behavior change)."""
        if not disable_safety:
            return
        if provider in ["openai", "groq", "fireworks", "together"]:
            payload["moderation"] = False
        elif provider == "poe":
            payload["safe_mode"] = False
        elif provider == "openrouter":
            headers['X-Safe-Mode'] = 'false'

    def _build_anthropic_payload(self, formatted_messages: list, temperature: float, max_tokens: int, anti_dupe_params: dict, system_message: Optional[str] = None) -> dict:
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **(anti_dupe_params or {})
        }
        if system_message:
            data["system"] = system_message
        return data

    def _parse_anthropic_json(self, json_resp: dict):
        content_parts = json_resp.get("content", [])
        if isinstance(content_parts, list):
            content = "".join(part.get("text", "") for part in content_parts)
        else:
            content = str(content_parts)
        finish_reason = json_resp.get("stop_reason")
        if finish_reason == "max_tokens":
            finish_reason = "length"
        elif finish_reason == "stop_sequence":
            finish_reason = "stop"
        usage = json_resp.get("usage")
        if usage:
            usage = {
                'prompt_tokens': usage.get('input_tokens', 0),
                'completion_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            }
        else:
            usage = None
        return content, finish_reason, usage

    def _get_idempotency_key(self) -> str:
        """Build an idempotency key from the current request context."""
        tls = self._get_thread_local_client()
        req_id = getattr(tls, "idem_request_id", None) or uuid.uuid4().hex[:8]
        attempt = getattr(tls, "idem_attempt", 0)
        return f"{req_id}-a{attempt}"

    def _get_openai_client(self, base_url: str, api_key: str):
        """Get or create a thread-local OpenAI client for a base_url."""
        if openai is None:
            raise UnifiedClientError("OpenAI SDK not installed. Install with: pip install openai")
            
        # CRITICAL: If individual endpoint is applied, use our existing client instead of creating new one
        if (hasattr(self, '_individual_endpoint_applied') and self._individual_endpoint_applied and 
            hasattr(self, 'openai_client') and self.openai_client):
            return self.openai_client
            
        tls = self._get_thread_local_client()
        if not hasattr(tls, "openai_clients"):
            tls.openai_clients = {}
        map_key = f"{base_url}|{bool(api_key)}"
        client = tls.openai_clients.get(map_key)
        if client is None:
            timeout_obj = None
            try:
                if httpx is not None:
                    connect, read = self._get_timeouts()
                    timeout_obj = httpx.Timeout(connect=connect, read=read, write=read, pool=connect)
                else:
                    # Fallback: use read timeout as a single float
                    _, read = self._get_timeouts()
                    timeout_obj = float(read)
            except Exception:
                _, read = self._get_timeouts()
                timeout_obj = float(read)
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_obj
            )
            tls.openai_clients[map_key] = client
        return client
    
    def _get_response(self, messages, temperature, max_tokens, max_completion_tokens, response_name) -> UnifiedResponse:
        """
        Route to appropriate AI provider and get response
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens (for non-o series models)
            max_completion_tokens: Maximum completion tokens (for o-series models)
            response_name: Name for saving response
        """
        self._apply_api_call_stagger()
        # Ensure client_type is initialized before routing (important for multi-key mode)
        try:
            if not hasattr(self, 'client_type') or self.client_type is None:
                self._ensure_thread_client()
        except Exception:
            # Guard against missing attribute in extreme early paths
            if not hasattr(self, 'client_type'):
                self.client_type = None

        # FIX: Ensure max_tokens has a value before passing to handlers
        if max_tokens is None and max_completion_tokens is None:
            # Use instance default or standard default
            max_tokens = getattr(self, 'max_tokens', 8192)
        elif max_tokens is None and max_completion_tokens is not None:
            # For o-series models, use max_completion_tokens as fallback
            max_tokens = max_completion_tokens
        # Check if this is actually Gemini (including when using OpenAI endpoint)
        actual_provider = self._get_actual_provider()
        
        # Detect if this is an image request (messages contain image parts)
        has_images = False
        for _m in messages:
            c = _m.get('content')
            if isinstance(c, list) and any(isinstance(p, dict) and p.get('type') == 'image_url' for p in c):
                has_images = True
                break

        # If image request, route to image handlers only for providers that require it
        if has_images:
            img_b64 = self._extract_first_image_base64(messages)
            if actual_provider == 'gemini':
                return self._send_gemini(messages, temperature, max_tokens or max_completion_tokens, response_name, image_base64=img_b64)
            if actual_provider == 'anthropic':
                return self._send_anthropic_image(messages, img_b64, temperature, max_tokens or max_completion_tokens, response_name)
            if actual_provider == 'vertex_model_garden':
                return self._send_vertex_model_garden_image(messages, img_b64, temperature, max_tokens or max_completion_tokens, response_name)
            if actual_provider == 'poe':
                return self._send_poe_image(messages, img_b64, temperature, max_tokens or max_completion_tokens, response_name)
            # Otherwise fall through to default handler below (OpenAI-compatible providers handle images in messages)

        # Map client types to their handler methods
        handlers = {
            'openai': self._send_openai,
            'gemini': self._send_gemini,
            'deepseek': self._send_openai_provider_router,  # Consolidated
            'anthropic': self._send_anthropic,
            'mistral': self._send_mistral,
            'cohere': self._send_cohere,
            'chutes': self._send_openai_provider_router,  # Consolidated
            'ai21': self._send_ai21,
            'together': self._send_openai_provider_router,  # Already in router
            'perplexity': self._send_openai_provider_router,  # Consolidated
            'replicate': self._send_replicate,
            'yi': self._send_openai_provider_router,
            'qwen': self._send_openai_provider_router,
            'baichuan': self._send_openai_provider_router,
            'zhipu': self._send_openai_provider_router,
            'moonshot': self._send_openai_provider_router,
            'groq': self._send_openai_provider_router,
            'baidu': self._send_openai_provider_router,
            'tencent': self._send_openai_provider_router,
            'iflytek': self._send_openai_provider_router,
            'bytedance': self._send_openai_provider_router,
            'minimax': self._send_openai_provider_router,
            'sensenova': self._send_openai_provider_router,
            'internlm': self._send_openai_provider_router,
            'tii': self._send_openai_provider_router,
            'microsoft': self._send_openai_provider_router,
            'azure': self._send_azure,
            'google': self._send_google_palm,
            'alephalpha': self._send_alephalpha,
            'databricks': self._send_openai_provider_router,
            'huggingface': self._send_huggingface,
            'openrouter': self._send_openai_provider_router,  # OpenRouter aggregator
            'poe': self._send_poe,  # POE platform (restored)
            'electronhub': self._send_electronhub,  # ElectronHub aggregator (restored)
            'fireworks': self._send_openai_provider_router,
            'xai': self._send_openai_provider_router,  # xAI Grok models
            'salesforce': self._send_openai_provider_router,  # Consolidated
            'vertex_model_garden': self._send_vertex_model_garden,
            'deepl': self._send_deepl,  # DeepL translation service
            'google_translate': self._send_google_translate,  # Google Cloud Translate
        }
        
        # IMPORTANT: Use actual_provider for routing, not client_type
        # This ensures Gemini always uses its native handler even when using OpenAI endpoint
        handler = handlers.get(actual_provider)
        
        if not handler:
            # Fallback to client_type if no actual_provider match
            handler = handlers.get(self.client_type)
        
        if not handler:
            # Try fallback to Together AI for open models
            if self.client_type in ['bigscience', 'meta']:
                logger.info(f"Using Together AI for {self.client_type} model")
                return self._send_openai_provider_router(messages, temperature, max_tokens, response_name)
            raise UnifiedClientError(f"No handler for client type: {getattr(self, 'client_type', 'unknown')}")

        if self.client_type in ['deepl', 'google_translate']:
            # These services don't use temperature or token limits
            # They just translate the text directly
            return handler(messages, None, None, response_name)
        
        # Route based on actual provider (handles Gemini with OpenAI endpoint correctly)
        elif actual_provider == 'gemini':
            # Always use Gemini handler for Gemini models, regardless of transport
            logger.debug(f"Routing to Gemini handler (actual provider: {actual_provider}, client_type: {self.client_type})")
            return self._send_gemini(messages, temperature, max_tokens, response_name)
        elif actual_provider == 'openai' or self.client_type == 'openai':
            # For OpenAI, pass the max_completion_tokens parameter
            return handler(messages, temperature, max_tokens, max_completion_tokens, response_name)
        elif self.client_type == 'vertex_model_garden':
            # Vertex AI doesn't use response_name parameter
            return handler(messages, temperature, max_tokens or max_completion_tokens, None, response_name)
        else:
            # Other providers don't use max_completion_tokens
            return handler(messages, temperature, max_tokens, response_name)


    def _get_actual_provider(self) -> str:
        """
        Get the actual provider name, accounting for Gemini using OpenAI endpoint.
        This is used for proper routing and detection.
        """
        # Check if this is Gemini using OpenAI endpoint
        if hasattr(self, '_original_client_type') and self._original_client_type:
            return self._original_client_type
        return getattr(self, 'client_type', 'openai')

    def _extract_chapter_label(self, messages) -> str:
        """Extract a concise chapter/chunk label from messages for logging."""
        try:
            s = str(messages)
            import re
            chap = None
            m = re.search(r'Chapter\s+(\d+)', s)
            if m:
                chap = f"Chapter {m.group(1)}"
            chunk = None
            mc = re.search(r'Chunk\s+(\d+)/(\d+)', s)
            if mc:
                chunk = f"{mc.group(1)}/{mc.group(2)}"
            if chap and chunk:
                return f"{chap} (chunk {chunk})"
            if chap:
                return chap
            if chunk:
                return f"chunk {chunk}"
        except Exception:
            pass
        return "request"

    def _log_pre_stagger(self, messages, context: Optional[str] = None) -> None:
        """Emit a pre-stagger log line so users see what's being sent before delay."""
        try:
            thread_name = threading.current_thread().name
            label = self._extract_chapter_label(messages)
            ctx = context or 'translation'
            print(f"📤 [{thread_name}] Sending {label} ({ctx}) — queuing staggered API call...")
            # Stash label so stagger logger can show what is being translated
            try:
                tls = self._get_thread_local_client()
                tls.current_request_label = label
            except Exception:
                pass
        except Exception:
            # Never block on logging
            pass

    def _is_gemini_request(self) -> bool:
        """
        Check if this is a Gemini request (native or via OpenAI endpoint)
        """
        return self._get_actual_provider() == 'gemini'
    
    def _is_stop_requested(self) -> bool:
        """
        Check if stop was requested by checking global flag, local cancelled flag, and class-level cancellation
        """
        # Check class-level global cancellation first
        if self.is_globally_cancelled():
            return True
            
        # Check local cancelled flag (more reliable in threading context)
        if getattr(self, '_cancelled', False):
            return True
            
        try:
            # Import the stop check function from the main translation module
            from TransateKRtoEN import is_stop_requested
            return is_stop_requested()
        except ImportError:
            # Fallback if import fails
            return False
    
    def _get_anti_duplicate_params(self, temperature):
        """Get user-configured anti-duplicate parameters from GUI settings"""
        # Check if user enabled anti-duplicate
        if os.getenv("ENABLE_ANTI_DUPLICATE", "0") != "1":
            return {}
        
        # Get user's exact values from GUI (via environment variables)
        top_p = float(os.getenv("TOP_P", "1.0"))
        top_k = int(os.getenv("TOP_K", "0"))
        frequency_penalty = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
        presence_penalty = float(os.getenv("PRESENCE_PENALTY", "0.0"))
        
        # Apply parameters based on provider capabilities
        params = {}
        
        if self.client_type in ['openai', 'deepseek', 'groq', 'electronhub', 'openrouter']:
            # OpenAI-compatible providers
            if frequency_penalty > 0:
                params["frequency_penalty"] = frequency_penalty
            if presence_penalty > 0:
                params["presence_penalty"] = presence_penalty
            if top_p < 1.0:
                params["top_p"] = top_p
                
        elif self.client_type == 'gemini':
            # Gemini supports both top_p and top_k
            if top_p < 1.0:
                params["top_p"] = top_p
            if top_k > 0:
                params["top_k"] = top_k
                
        elif self.client_type == 'anthropic':
            # Claude supports top_p and top_k
            if top_p < 1.0:
                params["top_p"] = top_p
            if top_k > 0:
                params["top_k"] = top_k
        
        # Log applied parameters
        if params:
            logger.info(f"Applying anti-duplicate params for {self.client_type}: {list(params.keys())}")
        
        return params
    
    def _detect_silent_truncation(self, content: str, messages: List[Dict], context: str = None) -> bool:
        """
        Detect silent truncation where APIs (especially ElectronHub) cut off content
        without setting proper finish_reason.
        
        Common patterns:
        - Sentences ending abruptly without punctuation
        - Content significantly shorter than expected
        - Missing closing tags in structured content
        - Sudden topic changes or incomplete thoughts
        """
        if not content:
            return False
        
        content_stripped = content.strip()
        if not content_stripped:
            return False
        
        # Pattern 1: Check for incomplete sentence endings (with improved logic)
        # Skip this check for code contexts, JSON, or when content contains code blocks
        if context not in ['code', 'json', 'data', 'list', 'python', 'javascript', 'programming']:
            # Also skip if content appears to contain code
            if '```' in content or 'def ' in content or 'class ' in content or 'import ' in content or 'function ' in content:
                pass  # Skip punctuation check for code content
            else:
                last_char = content_stripped[-1]
                # Valid endings for PROSE/NARRATIVE text only
                # Removed quotes since they're common in code
                valid_endings = [
                    ".", "!", "?", "»", "】", "）", ")", 
                    "。", "！", "？", ":", ";", "]", "}",
                    "…", "—", "–", "*", "/", ">", "~", "%"
                ]
                
                # Check if ends with incomplete sentence (no proper punctuation)
                if last_char not in valid_endings:
                    # Look at the last few characters for better context
                    last_segment = content_stripped[-50:] if len(content_stripped) > 50 else content_stripped
                    
                    # Check for common false positive patterns
                    false_positive_patterns = [
                        # Lists or enumerations often don't end with punctuation
                        r'\n\s*[-•*]\s*[^.!?]+$',  # Bullet points
                        r'\n\s*\d+\)\s*[^.!?]+$',   # Numbered lists
                        r'\n\s*[a-z]\)\s*[^.!?]+$', # Letter lists
                        # Code or technical content
                        r'```[^`]*$',                # Inside code block
                        r'\$[^$]+$',                 # Math expressions
                        # Single words or short phrases (likely labels/headers)
                        r'^\w+$',                    # Single word
                        r'^[\w\s]{1,15}$',          # Very short content
                    ]
                    
                    import re
                    is_false_positive = any(re.search(pattern, last_segment) for pattern in false_positive_patterns)
                    
                    if not is_false_positive:
                        # Additional check: is the last word incomplete?
                        words = content_stripped.split()
                        if words and len(words) > 3:  # Only check if we have enough content
                            last_word = words[-1]
                            # Check for common incomplete patterns
                            # But exclude common abbreviations
                            common_abbreviations = {'etc', 'vs', 'eg', 'ie', 'vol', 'no', 'pg', 'ch', 'pt'}
                            if (len(last_word) > 2 and 
                                last_word[-1].isalpha() and 
                                last_word.lower() not in common_abbreviations and
                                not last_word.isupper()):  # Exclude acronyms
                                
                                # Final check: does it look like mid-sentence?
                                # Look for sentence starters before the last segment
                                preceding_text = ' '.join(words[-10:-1]) if len(words) > 10 else ' '.join(words[:-1])
                                sentence_starters = ['the', 'a', 'an', 'and', 'but', 'or', 'so', 'because', 'when', 'if', 'that']
                                
                                # Check if we're likely mid-sentence
                                if any(starter in preceding_text.lower().split() for starter in sentence_starters):
                                    print(f"Possible silent truncation detected: incomplete sentence ending")
                                    return True
        
        # Pattern 2: Check for significantly short responses (with improved thresholds)
        if context == 'translation':
            # Calculate input length more accurately
            input_content = []
            for msg in messages:
                if msg.get('role') == 'user':
                    msg_content = msg.get('content', '')
                    # Handle both string and list content formats
                    if isinstance(msg_content, list):
                        for item in msg_content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                input_content.append(item.get('text', ''))
                    else:
                        input_content.append(msg_content)
            
            input_length = sum(len(text) for text in input_content)
            
            # Adjusted threshold - translations can legitimately be shorter
            # Only flag if output is less than 20% of input AND input is substantial
            if input_length > 1000 and len(content_stripped) < input_length * 0.2:
                # Additional check: does the content seem complete?
                if not content_stripped.endswith(('.', '!', '?', '"', "'", '。', '！', '？')):
                    print(f"Possible silent truncation: output ({len(content_stripped)} chars) much shorter than input ({input_length} chars)")
                    return True
        
        # Pattern 3: Check for incomplete HTML/XML structures (improved)
        if '<' in content and '>' in content:
            # More sophisticated tag matching
            import re
            
            # Find all opening tags (excluding self-closing)
            opening_tags = re.findall(r'<([a-zA-Z][^/>]*?)(?:\s[^>]*)?>',content)
            closing_tags = re.findall(r'</([a-zA-Z][^>]*?)>', content)
            self_closing = re.findall(r'<[^>]*?/>', content)
            
            # Count tag mismatches
            from collections import Counter
            open_counts = Counter(opening_tags)
            close_counts = Counter(closing_tags)
            
            # Check for significant mismatches
            unclosed_tags = []
            for tag, count in open_counts.items():
                # Ignore void elements that don't need closing
                void_elements = {'br', 'hr', 'img', 'input', 'meta', 'area', 'base', 'col', 'embed', 'link', 'param', 'source', 'track', 'wbr'}
                if tag.lower() not in void_elements:
                    close_count = close_counts.get(tag, 0)
                    if count > close_count + 1:  # Allow 1 tag mismatch
                        unclosed_tags.append(tag)
            
            if len(unclosed_tags) > 2:  # Multiple unclosed tags indicate truncation
                print(f"Possible silent truncation: unclosed HTML tags detected: {unclosed_tags}")
                return True
        
        # Pattern 4: Check for mature content indicators (reduced false positives)
        # Only check if the content is suspiciously short
        if len(content_stripped) < 200:
            mature_indicators = [
                'cannot provide explicit', 'cannot generate adult',
                'unable to create sexual', 'cannot assist with mature',
                'against my guidelines to create explicit'
            ]
            content_lower = content_stripped.lower()
            
            for indicator in mature_indicators:
                if indicator in content_lower:
                    # This is likely a refusal, not truncation
                    # Don't mark as truncation, let the calling code handle it
                    print(f"Content appears to be refused (contains '{indicator[:20]}...')")
                    return False  # This is a refusal, not truncation
        
        # Pattern 5: Check for incomplete code blocks
        if '```' in content:
            code_block_count = content.count('```')
            if code_block_count % 2 != 0:  # Odd number means unclosed
                # Additional check: is there actual code content?
                last_block_pos = content.rfind('```')
                content_after_block = content[last_block_pos + 3:].strip()
                
                # Only flag if there's substantial content after the opening ```
                if len(content_after_block) > 10:
                    print(f"Possible silent truncation: unclosed code block")
                    return True
        
        # Pattern 6: For glossary/JSON context, check for incomplete JSON (improved)
        if context in ['glossary', 'json', 'data']:
            # Try to detect JSON-like content
            if content_stripped.startswith(('[', '{')):
                # Check for matching brackets
                open_brackets = content_stripped.count('[') + content_stripped.count('{')
                close_brackets = content_stripped.count(']') + content_stripped.count('}')
                
                if open_brackets > close_brackets:
                    # Additional validation: try to parse as JSON
                    import json
                    try:
                        json.loads(content_stripped)
                        # It's valid JSON, not truncated
                        return False
                    except json.JSONDecodeError as e:
                        # Check if the error is at the end (indicating truncation)
                        if e.pos >= len(content_stripped) - 10:
                            print(f"Possible silent truncation: incomplete JSON structure")
                            return True
        
        # Pattern 7: Check for sudden endings in long content
        if len(content_stripped) > 500:
            # Look for patterns that indicate mid-thought truncation
            last_100_chars = content_stripped[-100:]
            
            # Check for incomplete patterns at the end
            incomplete_patterns = [
                r',\s*$',                    # Ends with comma
                r';\s*$',                    # Ends with semicolon  
                r'\w+ing\s+$',               # Ends with -ing word (often mid-action)
                r'\b(and|or|but|with|for|to|in|on|at)\s*$',  # Ends with conjunction/preposition
                r'\b(the|a|an)\s*$',        # Ends with article
            ]
            
            import re
            for pattern in incomplete_patterns:
                if re.search(pattern, last_100_chars, re.IGNORECASE):
                    # Double-check this isn't a false positive
                    # Look at the broader context
                    sentences = content_stripped.split('.')
                    if len(sentences) > 3:  # Has multiple sentences
                        last_sentence = sentences[-1].strip()
                        if len(last_sentence) > 20:  # Substantial incomplete sentence
                            print(f"Possible silent truncation: content ends mid-thought")
                            return True
        
        return False

    def _enhance_electronhub_response(self, response: UnifiedResponse, messages: List[Dict], 
                                     context: str = None) -> UnifiedResponse:
        """
        Enhance ElectronHub responses with better truncation detection and handling.
        ElectronHub sometimes silently truncates without proper finish_reason.
        """
        # If already marked as truncated, no need to check further
        if response.is_truncated:
            return response
        
        # Check for silent truncation
        if self._detect_silent_truncation(response.content, messages, context):
            print(f"Silent truncation detected for {self.model} via ElectronHub")
            
            # Check if it's likely censorship vs length limit
            content_lower = response.content.lower()
            censorship_phrases = [
                "i cannot", "i can't", "inappropriate", "unable to process",
                "against my guidelines", "cannot assist", "not able to",
                "i'm not able", "i am not able", "cannot provide", "can't provide"
            ]
            
            is_censorship = any(phrase in content_lower for phrase in censorship_phrases)
            
            if is_censorship:
                # This is content refusal, not truncation
                logger.info("Detected content refusal rather than truncation")
                response.finish_reason = 'content_filter'
                response.error_details = {
                    'type': 'content_refused',
                    'provider': 'electronhub',
                    'model': self.model,
                    'detection': 'silent_censorship'
                }
            else:
                # This is actual truncation
                response.finish_reason = 'length'  # Mark as truncated for retry logic
                response.error_details = {
                    'type': 'silent_truncation',
                    'provider': 'electronhub', 
                    'model': self.model,
                    'detection': 'pattern_analysis'
                }
            
            # Add warning to content for translation context
            if context == 'translation' and not is_censorship:
                response.content += "\n[WARNING: Response may be truncated]"
        
        return response
 
    def _send_electronhub(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to ElectronHub API aggregator with enhanced truncation detection
        
        ElectronHub provides access to multiple AI models through a unified endpoint.
        Model names should be prefixed with 'eh/', 'electronhub/', or 'electron/'.
        
        Examples:
        - eh/yi-34b-chat-200k
        - electronhub/gpt-4.5
        - electron/claude-4-opus
        
        Note: ElectronHub uses OpenAI-compatible API format.
        This version includes silent truncation detection for mature content.
        """
        # Get ElectronHub endpoint (can be overridden via environment)
        base_url = os.getenv("ELECTRONHUB_API_URL", "https://api.electronhub.ai/v1")
        
        # Store original model name for error messages and restoration
        original_model = self.model
        
        # Strip the ElectronHub prefix from the model name
        # This is critical - ElectronHub expects the model name WITHOUT the prefix
        actual_model = self.model
        
        # Define prefixes to strip (in order of likelihood)
        electronhub_prefixes = ['eh/', 'electronhub/', 'electron/']
        
        # Strip the first matching prefix
        for prefix in electronhub_prefixes:
            if actual_model.startswith(prefix):
                actual_model = actual_model[len(prefix):]
                logger.info(f"Stripped '{prefix}' prefix from model name: '{original_model}' -> '{actual_model}'")
                print(f"🔌 ElectronHub: Using model '{actual_model}' (stripped from '{original_model}')")
                break
        else:
            # No prefix found - this shouldn't happen if routing worked correctly
            print(f"No ElectronHub prefix found in model '{self.model}', using as-is")
            print(f"⚠️ ElectronHub: No prefix found in '{self.model}', using as-is")
        
        # Log the API call details
        logger.info(f"Sending to ElectronHub API: model='{actual_model}', endpoint='{base_url}'")
        
        # Debug: Log system prompt if present
        for msg in messages:
            if msg.get('role') == 'system':
                logger.debug(f"ElectronHub - System prompt detected: {len(msg.get('content', ''))} chars")
                print(f"📝 ElectronHub: Sending system prompt ({len(msg.get('content', ''))} characters)")
                break
        else:
            print("ElectronHub - No system prompt found in messages")
            print("⚠️ ElectronHub: No system prompt in messages")
        
        # Check if we should warn about potentially problematic models
        #problematic_models = ['claude', 'gpt-4', 'gpt-3.5', 'gemini']
        #if any(model in actual_model.lower() for model in problematic_models):
            #print(f"⚠️ ElectronHub: Model '{actual_model}' may have strict content filters")
            
            # Check for mature content indicators
            all_content = ' '.join(msg.get('content', '') for msg in messages).lower()
            mature_indicators = ['mature', 'adult', 'explicit', 'sexual', 'violence', 'intimate']
            #if any(indicator in all_content for indicator in mature_indicators):
                #print(f"💡 ElectronHub: Consider using models like yi-34b-chat, deepseek-chat, or llama-2-70b for this content")
        
        # Temporarily update self.model for the API call
        # This is necessary because _send_openai_compatible uses self.model
        self.model = actual_model
        
        try:
            # Make the API call using OpenAI-compatible format
            result = self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url=base_url,
                response_name=response_name,
                provider="electronhub"
            )
            
            # ENHANCEMENT: Check for silent truncation/censorship
            enhanced_result = self._enhance_electronhub_response(result, messages, self.context)
            
            if enhanced_result.finish_reason in ['length', 'content_filter']:
                self._log_truncation_failure(
                    messages=messages,
                    response_content=enhanced_result.content,
                    finish_reason=enhanced_result.finish_reason,
                    context=self.context,
                    error_details=enhanced_result.error_details
                )
            
            # Log if truncation was detected
            if enhanced_result.finish_reason == 'length' and result.finish_reason != 'length':
                print(f"🔍 ElectronHub: Silent truncation detected and corrected")
            elif enhanced_result.finish_reason == 'content_filter' and result.finish_reason != 'content_filter':
                print(f"🚫 ElectronHub: Silent content refusal detected")
            
            return enhanced_result
            
        except UnifiedClientError as e:
            # Enhance error messages for common ElectronHub issues
            error_str = str(e)
            
            if "Invalid model" in error_str or "400" in error_str or "model not found" in error_str.lower():
                # Provide helpful error message for invalid models
                error_msg = (
                    f"ElectronHub rejected model '{actual_model}' (original: '{original_model}').\n"
                    f"\nCommon ElectronHub model names:\n"
                    f"  • OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-4.5, gpt-4.1\n"
                    f"  • Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-4-opus, claude-4-sonnet\n"
                    f"  • Meta: llama-2-70b-chat, llama-2-13b-chat, llama-2-7b-chat, llama-3-70b, llama-4-70b\n"
                    f"  • Mistral: mistral-large, mistral-medium, mixtral-8x7b\n"
                    f"  • Google: gemini-pro, gemini-1.5-pro, gemini-2.5-pro\n"
                    f"  • Yi: yi-34b-chat, yi-6b-chat\n"
                    f"  • Others: deepseek-coder-33b, qwen-72b-chat, grok-3\n"
                    f"\nNote: Do not include version suffixes like ':latest' or ':safe'"
                )
                print(f"\n❌ {error_msg}")
                raise UnifiedClientError(error_msg, error_type="invalid_model", details={"attempted_model": actual_model})
                
            elif "unauthorized" in error_str.lower() or "401" in error_str:
                error_msg = (
                    f"ElectronHub authentication failed. Please check your API key.\n"
                    f"Make sure you're using an ElectronHub API key, not a key from the underlying provider."
                )
                print(f"\n❌ {error_msg}")
                raise UnifiedClientError(error_msg, error_type="auth_error")
                
            elif "rate limit" in error_str.lower() or "429" in error_str:
                error_msg = f"ElectronHub rate limit exceeded. Please wait before retrying."
                print(f"\n⏳ {error_msg}")
                raise UnifiedClientError(error_msg, error_type="rate_limit")
                
            else:
                # Re-raise original error with context
                print(f"ElectronHub API error for model '{actual_model}': {e}")
                raise
                
        finally:
            # Always restore the original model name
            # This ensures subsequent calls work correctly
            self.model = original_model
 
    def _parse_poe_tokens(self, key_str: str) -> dict:
        """Parse POE cookies from a single string.
        Returns a dict that always includes 'p-b' (required) and may include 'p-lat' and any
        other cookies present (e.g., 'cf_clearance', '__cf_bm'). Unknown cookies are forwarded as-is.
        
        Accepted input formats:
        - "p-b:AAA|p-lat:BBB"
        - "p-b=AAA; p-lat=BBB"
        - Raw cookie header with or without the "Cookie:" prefix
        - Just the p-b value (long string) when no delimiter is present
        """
        import re
        s = (key_str or "").strip()
        if s.lower().startswith("cookie:"):
            s = s.split(":", 1)[1].strip()
        tokens: dict = {}
        # Split on | ; , or newlines
        parts = re.split(r"[|;,\n]+", s)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                k, v = part.split(":", 1)
            elif "=" in part:
                k, v = part.split("=", 1)
            else:
                # If no delimiter and p-b not set, treat entire string as p-b
                if 'p-b' not in tokens and len(part) > 20:
                    tokens['p-b'] = part
                continue
            k = k.strip().lower()
            v = v.strip()
            # Normalize key names
            if k in ("p-b", "p_b", "pb", "p.b"):
                tokens['p-b'] = v
            elif k in ("p-lat", "p_lat", "plat", "p.lat"):
                tokens['p-lat'] = v
            else:
                # Forward any additional cookie that looks valid
                if re.match(r"^[a-z0-9_\-\.]+$", k):
                    tokens[k] = v
        return tokens

    def _send_poe(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request using poe-api-wrapper"""
        try:
            from poe_api_wrapper import PoeApi
        except ImportError:
            raise UnifiedClientError(
                "poe-api-wrapper not installed. Run: pip install poe-api-wrapper"
            )
        
        # Parse cookies using robust parser
        tokens = self._parse_poe_tokens(self.api_key)
        if 'p-b' not in tokens or not tokens['p-b']:
            raise UnifiedClientError(
                "POE tokens missing. Provide cookies as 'p-b:VALUE|p-lat:VALUE' or 'p-b=VALUE; p-lat=VALUE'",
                error_type="auth_error"
            )
        
        # Some wrapper versions require p-lat present (empty is allowed but may reduce success rate)
        if 'p-lat' not in tokens:
            logger.info("No p-lat cookie provided; proceeding without it")
            tokens['p-lat'] = ''
        
        logger.info(f"Tokens being sent: p-b={len(tokens.get('p-b', ''))} chars, p-lat={len(tokens.get('p-lat', ''))} chars")
        
        try:
            # Create Poe client (try to pass proxy/headers if supported)
            poe_kwargs = {}
            ua = os.getenv("POE_USER_AGENT") or os.getenv("HTTP_USER_AGENT")
            if ua:
                poe_kwargs["headers"] = {"User-Agent": ua, "Referer": "https://poe.com/", "Origin": "https://poe.com"}
            proxy = os.getenv("POE_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
            if proxy:
                poe_kwargs["proxy"] = proxy
            try:
                poe_client = PoeApi(tokens=tokens, **poe_kwargs)
            except TypeError:
                # Older versions may not support headers/proxy kwargs
                poe_client = PoeApi(tokens=tokens)
                # Best-effort header update if client exposes httpx session
                try:
                    if ua and hasattr(poe_client, "session") and hasattr(poe_client.session, "headers"):
                        poe_client.session.headers.update({"User-Agent": ua, "Referer": "https://poe.com/", "Origin": "https://poe.com"})
                except Exception:
                    pass
            
            # Get bot name
            requested_model = self.model.replace('poe/', '', 1)
            bot_map = {
                # GPT models
                'gpt-4': 'beaver',
                'gpt-4o': 'GPT-4o',
                'gpt-3.5-turbo': 'chinchilla',
                
                # Claude models
                'claude': 'a2',
                'claude-instant': 'a2',
                'claude-2': 'claude_2',
                'claude-3-opus': 'claude_3_opus',
                'claude-3-sonnet': 'claude_3_sonnet',
                'claude-3-haiku': 'claude_3_haiku',
                
                # Gemini models
                'gemini-2.5-flash': 'gemini_1_5_flash',
                'gemini-2.5-pro': 'gemini_1_5_pro',
                'gemini-pro': 'gemini_pro',
                
                # Other models
                'assistant': 'assistant',
                'web-search': 'web_search',
            }
            bot_name = bot_map.get(requested_model.lower(), requested_model)
            logger.info(f"Using bot name: {bot_name}")
            
            # Send message
            prompt = self._messages_to_prompt(messages)
            full_response = ""
            
            # Handle temperature and max_tokens if supported
            # Note: poe-api-wrapper might not support these parameters directly
            for chunk in poe_client.send_message(bot_name, prompt):
                if 'response' in chunk:
                    full_response = chunk['response']
            
            # Get the final text
            final_text = chunk.get('text', full_response) if 'chunk' in locals() else full_response
            
            if not final_text:
                raise UnifiedClientError("POE returned empty response", error_type="empty")
            
            return UnifiedResponse(
                content=final_text,
                finish_reason="stop",
                raw_response=chunk if 'chunk' in locals() else {"response": full_response}
            )
            
        except Exception as e:
            print(f"Poe API error details: {str(e)}")
            # Check for specific errors
            error_str = str(e).lower()
            if "403" in error_str or "forbidden" in error_str or "auth" in error_str or "unauthorized" in error_str:
                raise UnifiedClientError(
                    "POE authentication failed (403). Your cookies may be invalid or expired. "
                    "Copy fresh cookies (p-b and p-lat) from an active poe.com session.",
                    error_type="auth_error"
                )
            if "rate limit" in error_str or "429" in error_str:
                raise UnifiedClientError(
                    "POE rate limit exceeded. Please wait before trying again.",
                    error_type="rate_limit"
                )
            raise UnifiedClientError(f"Poe API error: {e}")
            
    def _save_openrouter_config(self, config_data: dict, response_name: str = None):
        """Save OpenRouter configuration - simplified"""
        if not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
        
        # Handle None or empty response_name
        if not response_name:
            response_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize response_name
        import re
        response_name = re.sub(r'[<>:"/\\|?*]', '_', str(response_name))
        
        # Determine context from thread
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        # Create directory
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{thread_id}")
        os.makedirs(thread_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%H%M%S")
        config_filename = f"openrouter_config_{timestamp}_{response_name}.json"
        config_path = os.path.join(thread_dir, config_filename)
        
        try:
            # Use file lock if available
            if hasattr(self, '_file_write_lock'):
                with self._file_write_lock:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved OpenRouter config to: {config_path}")
        except Exception as e:
            print(f"Failed to save OpenRouter config: {e}")


    def _send_fireworks(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenAI API with o-series model support"""
        # Check if this is actually Azure
        if os.getenv('IS_AZURE_ENDPOINT') == '1':
            # Route to Azure-compatible handler
            base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            return self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url=base_url,
                response_name=response_name,
                provider="azure"
            )
        
        max_retries = self._get_max_retries()
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        # Track what fixes we've already tried
        fixes_attempted = {
            'temperature': False,
            'system_message': False,
            'max_tokens_param': False
        }
        
        for attempt in range(max_retries):
            try:
                params = self._build_openai_params(messages, temperature, max_tokens, max_completion_tokens)
                
                # Get user-configured anti-duplicate parameters
                anti_dupe_params = self._get_anti_duplicate_params(temperature)
                params.update(anti_dupe_params)
                
                # Apply any fixes from previous attempts
                if fixes_attempted['temperature'] and 'temperature_override' in fixes_attempted:
                    params['temperature'] = fixes_attempted['temperature_override']
                
                if fixes_attempted['system_message']:
                    # Convert system messages to user messages
                    new_messages = []
                    for msg in params.get('messages', []):
                        if msg['role'] == 'system':
                            new_messages.append({
                                'role': 'user',
                                'content': f"Instructions: {msg['content']}"
                            })
                        else:
                            new_messages.append(msg)
                    params['messages'] = new_messages
                
                if fixes_attempted['max_tokens_param']:
                    if 'max_tokens' in params:
                        params['max_completion_tokens'] = params.pop('max_tokens')
                
                # Check for cancellation
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                # Log the request for debugging
                logger.debug(f"OpenAI request - Model: {self.model}, Params: {list(params.keys())}")
                
                
                # Make the API call
                resp = self.openai_client.chat.completions.create(
                    **params,
                    timeout=self.request_timeout,
                    idempotency_key=self._get_idempotency_key()
                )
                
                # Enhanced response validation with detailed logging
                if not resp:
                    print("OpenAI returned None response")
                    raise UnifiedClientError("OpenAI returned empty response object")
                
                if not hasattr(resp, 'choices'):
                    print(f"OpenAI response missing 'choices'. Response type: {type(resp)}")
                    print(f"Response attributes: {dir(resp)[:10]}")  # Log first 10 attributes
                    raise UnifiedClientError("Invalid OpenAI response structure - missing choices")
                
                if not resp.choices:
                    print("OpenAI response has empty choices array")
                    # Check if this is a content filter issue
                    if hasattr(resp, 'model') and hasattr(resp, 'id'):
                        print(f"Response ID: {resp.id}, Model: {resp.model}")
                    raise UnifiedClientError("OpenAI returned empty choices array")
                
                choice = resp.choices[0]
                
                # Enhanced choice validation
                if not hasattr(choice, 'message'):
                    print(f"OpenAI choice missing 'message'. Choice type: {type(choice)}")
                    print(f"Choice attributes: {dir(choice)[:10]}")
                    raise UnifiedClientError("OpenAI choice missing message")
                
                # Check if this is actually Gemini using OpenAI endpoint
                is_gemini_via_openai = False
                if hasattr(self, '_original_client_type') and self._original_client_type == 'gemini':
                    is_gemini_via_openai = True
                    logger.info("This is Gemini using OpenAI-compatible endpoint")
                elif self.model.lower().startswith('gemini'):
                    is_gemini_via_openai = True
                    logger.info("Detected Gemini model via OpenAI endpoint")
                
                if not choice.message:
                    # Gemini via OpenAI sometimes returns None message
                    if is_gemini_via_openai:
                        print("Gemini via OpenAI returned None message - creating empty message")
                        # Create a mock message object
                        class MockMessage:
                            content = ""
                            refusal = None
                        choice.message = MockMessage()
                    else:
                        print("OpenAI choice.message is None")
                        raise UnifiedClientError("OpenAI message is empty")
                
                # Check for content with detailed debugging
                content = None
                
                # Try different ways to get content
                if hasattr(choice.message, 'content'):
                    content = choice.message.content
                elif hasattr(choice.message, 'text'):
                    content = choice.message.text
                elif isinstance(choice.message, dict):
                    content = choice.message.get('content') or choice.message.get('text')
                
                # Log what we found
                if content is None:
                    # For Gemini via OpenAI, None content is common and not an error
                    if is_gemini_via_openai:
                        print("Gemini via OpenAI returned None content - likely a safety filter")
                        content = ""  # Set to empty string instead of raising error
                        finish_reason = 'content_filter'
                    else:
                        print(f"OpenAI message has no content. Message type: {type(choice.message)}")
                        print(f"Message attributes: {dir(choice.message)[:20]}")
                        print(f"Message representation: {str(choice.message)[:200]}")
                    
                    # Check if this is a refusal (only if not already handled by Gemini)
                    if content is None and hasattr(choice.message, 'refusal') and choice.message.refusal:
                        print(f"OpenAI refused: {choice.message.refusal}")
                        # Return the refusal as content
                        content = f"[REFUSED BY OPENAI]: {choice.message.refusal}"
                        finish_reason = 'content_filter'
                    elif hasattr(choice, 'finish_reason'):
                        finish_reason = choice.finish_reason
                        print(f"Finish reason: {finish_reason}")
                        
                        # Check for specific finish reasons
                        if finish_reason == 'content_filter':
                            content = "[CONTENT BLOCKED BY OPENAI SAFETY FILTER]"
                        elif finish_reason == 'length':
                            content = ""  # Empty but will be marked as truncated
                        else:
                            # Try to extract any available info
                            content = f"[EMPTY RESPONSE - Finish reason: {finish_reason}]"
                    else:
                        content = "[EMPTY RESPONSE FROM OPENAI]"
                
                # Handle empty string content
                elif content == "":
                    print("OpenAI returned empty string content")
                    finish_reason = getattr(choice, 'finish_reason', 'unknown')
                    
                    if finish_reason == 'length':
                        logger.info("Empty content due to length limit")
                        # This is a truncation at the start - token limit too low
                        return UnifiedResponse(
                            content="",
                            finish_reason='length',
                            error_details={
                                'error': 'Response truncated - increase max_completion_tokens',
                                'finish_reason': 'length',
                                'token_limit': params.get('max_completion_tokens') or params.get('max_tokens')
                            }
                        )
                    elif finish_reason == 'content_filter':
                        content = "[CONTENT BLOCKED BY OPENAI]"
                    else:
                        print(f"Empty content with finish_reason: {finish_reason}")
                        content = f"[EMPTY - Reason: {finish_reason}]"
                
                # Get finish reason (with fallback)
                finish_reason = getattr(choice, 'finish_reason', 'stop')
                
                # Normalize OpenAI finish reasons
                if finish_reason == "max_tokens":
                    finish_reason = "length"
                
                # Special handling for Gemini empty responses
                if is_gemini_via_openai and content == "" and finish_reason == 'stop':
                    # Empty content with 'stop' from Gemini usually means safety filter
                    print("Empty Gemini response with finish_reason='stop' - likely safety filter")
                    content = "[BLOCKED BY GEMINI SAFETY FILTER]"
                    finish_reason = 'content_filter'
                
                # Extract usage
                usage = None
                if hasattr(resp, 'usage') and resp.usage:
                    usage = {
                        'prompt_tokens': resp.usage.prompt_tokens,
                        'completion_tokens': resp.usage.completion_tokens,
                        'total_tokens': resp.usage.total_tokens
                    }
                    logger.debug(f"Token usage: {usage}")
                
                # Log successful response
                logger.info(f"OpenAI response - Content length: {len(content) if content else 0}, Finish reason: {finish_reason}")
                
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_response=resp
                )
                
            except OpenAIError as e:
                error_str = str(e)
                error_dict = None
                
                # Try to extract error details
                try:
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        error_dict = e.response.json()
                        print(f"OpenAI error details: {error_dict}")
                except:
                    pass
                
                # Check if we can fix the error and retry
                should_retry = False
                
                # Handle temperature constraints reactively
                if not fixes_attempted['temperature'] and "temperature" in error_str and ("does not support" in error_str or "unsupported_value" in error_str):
                    # Extract what temperature the model wants
                    default_temp = 1  # Default fallback
                    if "Only the default (1)" in error_str:
                        default_temp = 1
                    elif error_dict and 'error' in error_dict:
                        # Try to parse the required temperature from error message
                        import re
                        temp_match = re.search(r'default \((\d+(?:\.\d+)?)\)', error_dict['error'].get('message', ''))
                        if temp_match:
                            default_temp = float(temp_match.group(1))
                    
                    # Send message to GUI
                    print(f"🔄 Model {self.model} requires temperature={default_temp}, retrying...")
                    
                    print(f"Model {self.model} requires temperature={default_temp}, will retry...")
                    fixes_attempted['temperature'] = True
                    fixes_attempted['temperature_override'] = default_temp
                    should_retry = True
                
                # Handle system message constraints reactively
                elif not fixes_attempted['system_message'] and "system" in error_str.lower() and ("not supported" in error_str or "unsupported" in error_str):
                    print(f"Model {self.model} doesn't support system messages, will convert and retry...")
                    fixes_attempted['system_message'] = True
                    should_retry = True
                
                # Handle max_tokens vs max_completion_tokens reactively
                elif not fixes_attempted['max_tokens_param'] and "max_tokens" in error_str and ("not supported" in error_str or "max_completion_tokens" in error_str):
                    print(f"Switching from max_tokens to max_completion_tokens for model {self.model}")
                    fixes_attempted['max_tokens_param'] = True
                    should_retry = True
                    time.sleep(api_delay)
                    continue
                
                # Handle rate limits
                elif "rate limit" in error_str.lower() or "429" in error_str:
                    # In multi-key mode, don't retry here - let outer handler rotate keys
                    if self._multi_key_mode:
                        print(f"OpenAI rate limit hit in multi-key mode - passing to key rotation")
                        raise UnifiedClientError(f"OpenAI rate limit: {e}", error_type="rate_limit")
                    elif attempt < max_retries - 1:
                        # Single key mode - wait and retry
                        wait_time = api_delay * 10
                        print(f"Rate limit hit, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                
                # If we identified a fix, retry immediately
                if should_retry and attempt < max_retries - 1:
                    time.sleep(api_delay)
                    continue
                
                # Other errors or no retries left
                if attempt < max_retries - 1:
                    print(f"OpenAI error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(api_delay)
                    continue
                    
                print(f"OpenAI error after all retries: {e}")
                raise UnifiedClientError(f"OpenAI error: {e}", error_type="api_error")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(api_delay)
                    continue
                raise UnifiedClientError(f"OpenAI error: {e}", error_type="unknown")
        
        raise UnifiedClientError("OpenAI API failed after all retries")
    
    def _build_openai_params(self, messages, temperature, max_tokens, max_completion_tokens=None):
        """Build parameters for OpenAI API call"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        # Determine which token parameter to use based on model
        if self._is_o_series_model():
            # o-series models use max_completion_tokens
            # The manga translator passes the actual value as max_tokens for now
            if max_completion_tokens is not None:
                params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
                logger.debug(f"Using max_completion_tokens={max_tokens} for o-series model {self.model}")
        else:
            # Regular models use max_tokens
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
                
        return params
    
    def _supports_thinking(self) -> bool:
        """Check if the current Gemini model supports thinking parameter"""
        if not self.model:
            return False
        
        model_lower = self.model.lower()
        
        # According to Google documentation, thinking is supported on:
        # 1. All Gemini 2.5 series models (Pro, Flash, Flash-Lite)
        # 2. Gemini 2.0 Flash Thinking Experimental model
        
        # Check for Gemini 2.5 series
        if 'gemini-2.5' in model_lower:
            return True
        
        # Check for Gemini 2.0 Flash Thinking model variants
        thinking_models = [
            'gemini-2.0-flash-thinking-exp',
            'gemini-2.0-flash-thinking-experimental',
            'gemini-2.0-flash-thinking-exp-1219',
            'gemini-2.0-flash-thinking-exp-01-21',
        ]
        
        for thinking_model in thinking_models:
            if thinking_model in model_lower:
                return True
        
        return False

    def _get_thread_directory(self):
        """Get thread-specific directory for payload storage"""
        thread_name = threading.current_thread().name
        # Prefer the client's explicit context if available
        explicit = getattr(self, 'context', None)
        if explicit in ('translation', 'glossary', 'summary'):
            context = explicit
        else:
            if 'Translation' in thread_name:
                context = 'translation'
            elif 'Glossary' in thread_name:
                context = 'glossary'
            elif 'Summary' in thread_name:
                context = 'summary'
            else:
                context = 'general'
        
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{threading.current_thread().ident}")
        os.makedirs(thread_dir, exist_ok=True)
        return thread_dir

    def _save_gemini_safety_config(self, config_data: dict, response_name: str = None):
        """Save Gemini safety configuration next to the current request payloads"""
        if not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
        
        # Handle None or empty response_name
        if not response_name:
            response_name = f"safety_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize response_name to ensure it's filesystem-safe
        # Remove or replace invalid characters
        import re
        response_name = re.sub(r'[<>:\"/\\|?*]', '_', str(response_name))
        
        # Reuse the same payload directory as other saves
        thread_dir = self._get_thread_directory()
        os.makedirs(thread_dir, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Ensure response_name doesn't already contain timestamp to avoid duplication
        if timestamp not in response_name:
            config_filename = f"gemini_safety_{timestamp}_{response_name}.json"
        else:
            config_filename = f"gemini_safety_{response_name}.json"
        
        config_path = os.path.join(thread_dir, config_filename)
        
        try:
            with self._file_write_lock:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                # Only log if not stopping
                if not self._is_stop_requested():
                    print(f"Saved Gemini safety status to: {config_path}")
        except Exception as e:
            # Only log errors if not stopping
            if not self._is_stop_requested():
                print(f"Failed to save Gemini safety config: {e}")

    def _send_gemini(self, messages, temperature, max_tokens, response_name, image_base64=None) -> UnifiedResponse:
        """Send request to Gemini API with support for both text and multi-image messages"""
        response = None
        
        # Check if we should use OpenAI-compatible endpoint
        use_openai_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
        gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
        
        # Import types at the top
        from google.genai import types
        
        # Check if this contains images
        has_images = image_base64 is not None  # Direct image parameter
        if not has_images:
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    for part in msg['content']:
                        if part.get('type') == 'image_url':
                            has_images = True
                            break
                    if has_images:
                        break
        
        # Check if safety settings are disabled
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Get thinking budget from environment
        thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))  
        
        # Check if this model supports thinking
        supports_thinking = self._supports_thinking()
        
        # Configure safety settings
        safety_settings = None
        if disable_safety:
            # Set all safety categories to BLOCK_NONE (most permissive)
            safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
            ]
            if not self._is_stop_requested():
                logger.info("Gemini safety settings disabled - using BLOCK_NONE for all categories")
        else:
            if not self._is_stop_requested():
                logger.info("Using default Gemini safety settings")

        # Define retry attempts
        attempts = self._get_max_retries()
        attempt = 0
        error_details = {}
        
        # Prepare configuration data
        readable_safety = "DEFAULT"
        safety_status = "ENABLED - Using default Gemini safety settings"
        if disable_safety:
            safety_status = "DISABLED - All categories set to BLOCK_NONE"
            readable_safety = {
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARASSMENT": "BLOCK_NONE",
                "DANGEROUS_CONTENT": "BLOCK_NONE",
                "CIVIC_INTEGRITY": "BLOCK_NONE"
            }
        
        # Log to console with thinking status - only if not stopping
        if not self._is_stop_requested():
            endpoint_info = f" (via OpenAI endpoint: {gemini_endpoint})" if use_openai_endpoint else " (native API)"
            print(f"🔒 Gemini Safety Status: {safety_status}{endpoint_info}")
            
            thinking_status = ""
            if supports_thinking:
                if thinking_budget == 0:
                    thinking_status = " (thinking disabled)"
                elif thinking_budget == -1:
                    thinking_status = " (dynamic thinking)"
                elif thinking_budget > 0:
                    thinking_status = f" (thinking budget: {thinking_budget})"
            else:
                thinking_status = " (thinking not supported)"
            
            print(f"🧠 Thinking Status: {thinking_status}")
        
        # Save configuration to file
        request_type = "IMAGE_REQUEST" if has_images else "TEXT_REQUEST"
        if use_openai_endpoint:
            request_type = "GEMINI_OPENAI_ENDPOINT_" + request_type
        config_data = {
            "type": request_type,
            "model": self.model,
            "endpoint": gemini_endpoint if use_openai_endpoint else "native",
            "safety_enabled": not disable_safety,
            "safety_settings": readable_safety,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "thinking_supported": supports_thinking,
            "thinking_budget": thinking_budget if supports_thinking else None,
            "timestamp": datetime.now().isoformat(),
        }

        # Save configuration to file with thread isolation
        self._save_gemini_safety_config(config_data, response_name)
        
        # Main attempt loop - SAME FOR BOTH ENDPOINTS
        while attempt < attempts:
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                # Get user-configured anti-duplicate parameters
                anti_dupe_params = self._get_anti_duplicate_params(temperature)

                # Build generation config with anti-duplicate parameters
                generation_config_params = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **anti_dupe_params  # Add user's custom parameters
                }
                
                # Log the request - only if not stopping
                if not self._is_stop_requested():
                    print(f"   📊 Temperature: {temperature}, Max tokens: {max_tokens}")

                # ========== MAKE THE API CALL - DIFFERENT FOR EACH ENDPOINT ==========
                if use_openai_endpoint and gemini_endpoint:
                    # Ensure the endpoint ends with /openai/ for compatibility
                    if not gemini_endpoint.endswith('/openai/'):
                        if gemini_endpoint.endswith('/'):
                            gemini_endpoint = gemini_endpoint + 'openai/'
                        else:
                            gemini_endpoint = gemini_endpoint + '/openai/'
                    
                    # Call OpenAI-compatible endpoint
                    response = self._send_openai_compatible(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=gemini_endpoint,
                        response_name=response_name,
                        provider="gemini-openai"
                    )
                    
                    # For OpenAI endpoint, we already have a UnifiedResponse
                    # Extract any thinking tokens if available
                    thinking_tokens_displayed = False
                    
                    if hasattr(response, 'raw_response'):
                        raw_resp = response.raw_response
                        
                        # Check multiple possible locations for thinking tokens
                        thinking_tokens = 0
                        
                        # Check in usage object
                        if hasattr(raw_resp, 'usage'):
                            usage = raw_resp.usage
                            
                            # Try various field names that might contain thinking tokens
                            if hasattr(usage, 'thoughts_token_count'):
                                thinking_tokens = usage.thoughts_token_count or 0
                            elif hasattr(usage, 'thinking_tokens'):
                                thinking_tokens = usage.thinking_tokens or 0
                            elif hasattr(usage, 'reasoning_tokens'):
                                thinking_tokens = usage.reasoning_tokens or 0
                            
                            # Also check if there's a breakdown in the usage
                            if hasattr(usage, 'completion_tokens_details'):
                                details = usage.completion_tokens_details
                                if hasattr(details, 'reasoning_tokens'):
                                    thinking_tokens = details.reasoning_tokens or 0
                        
                        # Check in the raw response itself
                        if thinking_tokens == 0 and hasattr(raw_resp, '__dict__'):
                            # Look for thinking-related fields in the response
                            for field_name in ['thoughts_token_count', 'thinking_tokens', 'reasoning_tokens']:
                                if field_name in raw_resp.__dict__:
                                    thinking_tokens = raw_resp.__dict__[field_name] or 0
                                    if thinking_tokens > 0:
                                        break
                        
                        # Display thinking tokens if found or if thinking was requested - only if not stopping
                        if supports_thinking and not self._is_stop_requested():
                            if thinking_tokens > 0:
                                print(f"   💭 Thinking tokens used: {thinking_tokens}")
                                thinking_tokens_displayed = True
                            elif thinking_budget == 0:
                                print(f"   ✅ Thinking successfully disabled (0 thinking tokens)")
                                thinking_tokens_displayed = True
                            elif thinking_budget == -1:
                                # Dynamic thinking - might not be reported
                                print(f"   💭 Thinking: Dynamic mode (tokens may not be reported via OpenAI endpoint)")
                                thinking_tokens_displayed = True
                            elif thinking_budget > 0:
                                # Specific budget requested but not reported
                                print(f"   ⚠️ Thinking budget set to {thinking_budget} but tokens not reported via OpenAI endpoint")
                                thinking_tokens_displayed = True
                    
                    # If we haven't displayed thinking status yet and it's supported, show a message
                    if not thinking_tokens_displayed and supports_thinking:
                        logger.debug("Thinking tokens may have been used but are not reported via OpenAI endpoint")
                    
                    # Check finish reason for prohibited content
                    if response.finish_reason == 'content_filter' or response.finish_reason == 'prohibited_content':
                        raise UnifiedClientError(
                            "Content blocked by Gemini OpenAI endpoint",
                            error_type="prohibited_content",
                            details={"endpoint": "openai", "finish_reason": response.finish_reason}
                        )
                    
                    return response
                    
                else:
                    # Native Gemini API call
                    # Prepare content based on whether we have images
                    if has_images:
                        # Handle image content
                        contents = self._prepare_gemini_image_content(messages, image_base64)
                    else:
                        # text-only: use formatted prompt
                        formatted_prompt = self._format_prompt(messages, style='gemini')
                        contents = formatted_prompt
                    # Only add thinking_config if the model supports it
                    if supports_thinking:
                        # Create thinking config separately
                        thinking_config = types.ThinkingConfig(
                            thinking_budget=thinking_budget
                        )
                        
                        # Create generation config with thinking_config as a parameter
                        generation_config = types.GenerateContentConfig(
                            thinking_config=thinking_config,
                            **generation_config_params
                        )
                    else:
                        # Create generation config without thinking_config
                        generation_config = types.GenerateContentConfig(
                            **generation_config_params
                        )
                    
                    # Add safety settings to config if they exist
                    if safety_settings:
                        generation_config.safety_settings = safety_settings

                    # Make the native API call
                    # Make the native API call with proper error handling
                    try:
                        # Check if gemini_client exists and is not None
                        if not hasattr(self, 'gemini_client') or self.gemini_client is None:
                            print("⚠️ Gemini client is None. This typically happens when stop was requested.")
                            raise UnifiedClientError("Gemini client not initialized - operation may have been cancelled", error_type="cancelled")
                        
                        response = self.gemini_client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config=generation_config
                        )
                    except AttributeError as e:
                        if "'NoneType' object has no attribute 'models'" in str(e):
                            print("⚠️ Gemini client is None or invalid. This typically happens when stop was requested.")
                            raise UnifiedClientError("Gemini client not initialized - operation may have been cancelled", error_type="cancelled")
                        else:
                            raise
                    
                    # Check for blocked content in prompt_feedback
                    if hasattr(response, 'prompt_feedback'):
                        feedback = response.prompt_feedback
                        if hasattr(feedback, 'block_reason') and feedback.block_reason:
                            error_details['block_reason'] = str(feedback.block_reason)
                            if disable_safety:
                                print(f"Content blocked despite safety disabled: {feedback.block_reason}")
                            else:
                                print(f"Content blocked: {feedback.block_reason}")
                            
                            # Raise as UnifiedClientError with prohibited_content type
                            raise UnifiedClientError(
                                f"Content blocked: {feedback.block_reason}",
                                error_type="prohibited_content",
                                details={"block_reason": str(feedback.block_reason)}
                            )
                    
                    # Check if response has candidates with prohibited content finish reason
                    prohibited_detected = False
                    finish_reason = 'stop'  # Default
                    
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason_str = str(candidate.finish_reason)
                                if 'PROHIBITED_CONTENT' in finish_reason_str:
                                    prohibited_detected = True
                                    finish_reason = 'prohibited_content'
                                    print(f"   🚫 Candidate has prohibited content finish reason: {finish_reason_str}")
                                    break
                                elif 'MAX_TOKENS' in finish_reason_str:
                                    finish_reason = 'length'
                                elif 'SAFETY' in finish_reason_str:
                                    finish_reason = 'safety'
                    
                    # If prohibited content detected, raise error for retry logic
                    if prohibited_detected:
                        # Get thinking tokens if available for debugging
                        thinking_tokens_wasted = 0
                        if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'thoughts_token_count'):
                            thinking_tokens_wasted = response.usage_metadata.thoughts_token_count or 0
                            if thinking_tokens_wasted > 0:
                                print(f"   ⚠️ Wasted {thinking_tokens_wasted} thinking tokens on prohibited content")
                        
                        raise UnifiedClientError(
                            "Content blocked: FinishReason.PROHIBITED_CONTENT",
                            error_type="prohibited_content",
                            details={
                                "finish_reason": "PROHIBITED_CONTENT",
                                "thinking_tokens_wasted": thinking_tokens_wasted
                            }
                        )
                    
                    # Log thinking token usage if available - only if not stopping
                    if hasattr(response, 'usage_metadata') and not self._is_stop_requested():
                        usage = response.usage_metadata
                        if supports_thinking and hasattr(usage, 'thoughts_token_count'):
                            if usage.thoughts_token_count and usage.thoughts_token_count > 0:
                                print(f"   💭 Thinking tokens used: {usage.thoughts_token_count}")
                            else:
                                print(f"   ✅ Thinking successfully disabled (0 thinking tokens)")
                    
                    # Extract text from the Gemini response - FIXED LOGIC HERE
                    text_content = ""
                    
                    # Try the simple .text property first (most common)
                    if hasattr(response, 'text'):
                        try:
                            text_content = response.text
                            if text_content:
                                print(f"   ✅ Extracted {len(text_content)} chars from response.text")
                        except Exception as e:
                            print(f"   ⚠️ Could not access response.text: {e}")
                    
                    # If that didn't work or returned empty, try extracting from candidates
                    if not text_content:
                        # CRITICAL FIX: Check if candidates exists AND is not None before iterating
                        if hasattr(response, 'candidates') and response.candidates is not None:
                            print(f"   🔍 Extracting from candidates...")
                            try:
                                for candidate in response.candidates:
                                    if hasattr(candidate, 'content') and candidate.content:
                                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                            for part in candidate.content.parts:
                                                if hasattr(part, 'text') and part.text:
                                                    text_content += part.text
                                        elif hasattr(candidate.content, 'text') and candidate.content.text:
                                            text_content += candidate.content.text
                                
                                if text_content:
                                    print(f"   ✅ Extracted {len(text_content)} chars from candidates")
                            except TypeError as e:
                                print(f"   ⚠️ Error iterating candidates: {e}")
                                print(f"   🔍 Candidates type: {type(response.candidates)}")
                        else:
                            print(f"   ⚠️ No candidates found in response or candidates is None")
                    
                    # Log if we still have no content
                    if not text_content:
                        print(f"   ⚠️ Warning: No text content extracted from Gemini response")
                        print(f"   🔍 Response attributes: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'No __dict__'}")
                    
                    # Return with the actual content populated
                    return UnifiedResponse(
                        content=text_content,  # Properly populated with the actual response text
                        finish_reason=finish_reason,
                        raw_response=response,
                        error_details=error_details if error_details else None
                    )
                # ========== END OF API CALL SECTION ==========
                    
            except UnifiedClientError as e:
                # Re-raise UnifiedClientErrors (including prohibited content)
                # This will trigger main key retry in the outer send() method
                raise
                
            except Exception as e:
                print(f"Gemini attempt {attempt+1} failed: {e}")
                error_details[f'attempt_{attempt+1}'] = str(e)
                
                # Check if this is a prohibited content error
                error_str = str(e).lower()
                if any(indicator in error_str for indicator in [
                    "content blocked", "prohibited_content", "blockedreason",
                    "content_filter", "safety filter", "harmful content"
                ]):
                    # Re-raise as UnifiedClientError with proper type
                    raise UnifiedClientError(
                        str(e),
                        error_type="prohibited_content",
                        details=error_details
                    )
                
                # Check if this is a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Re-raise for multi-key handling
                    raise UnifiedClientError(
                        f"Rate limit exceeded: {e}", 
                        error_type="rate_limit",
                        http_status=429
                    )
                
                # For other errors, we might want to retry
                if attempt < attempts - 1:
                    attempt += 1
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff with max 10s
                    print(f"⏳ Retrying Gemini in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, re-raise
                    raise
        
        # If we exhausted all attempts without success
        print(f"❌ All {attempts} Gemini attempts failed")
        
        # Log the failure
        self._log_truncation_failure(
            messages=messages,
            response_content="",
            finish_reason='error',
            context=self.context,
            error_details={'error': 'all_retries_failed', 'provider': 'gemini', 'attempts': attempts, 'details': error_details}
        )
        
        # Return error response
        return UnifiedResponse(
            content="",
            finish_reason='error',
            raw_response=response,
            error_details={'error': 'all_retries_failed', 'attempts': attempts, 'details': error_details}
        )
    
    def _format_prompt(self, messages, *, style: str) -> str:
        """
        Format messages into a single prompt string.
        style:
          - 'gemini': SYSTEM lines as 'INSTRUCTIONS: ...', others 'ROLE: ...'
          - 'ai21': SYSTEM as 'Instructions: ...', USER as 'User: ...', ASSISTANT as 'Assistant: ...', ends with 'Assistant: '
          - 'replicate': simple concatenation of SYSTEM, USER, ASSISTANT with labels, no trailing assistant line
        """
        formatted_parts = []
        for msg in messages:
            role = (msg.get('role') or 'user').upper()
            content = msg.get('content', '')
            if style == 'gemini':
                if role == 'SYSTEM':
                    formatted_parts.append(f"INSTRUCTIONS: {content}")
                else:
                    formatted_parts.append(f"{role}: {content}")
            elif style in ('ai21', 'replicate'):
                if role == 'SYSTEM':
                    label = 'Instructions' if style == 'ai21' else 'System'
                    formatted_parts.append(f"{label}: {content}")
                elif role == 'USER':
                    formatted_parts.append(f"User: {content}")
                elif role == 'ASSISTANT':
                    formatted_parts.append(f"Assistant: {content}")
                else:
                    formatted_parts.append(f"{role.title()}: {content}")
            else:
                formatted_parts.append(str(content))
        prompt = "\n\n".join(formatted_parts)
        if style == 'ai21':
            prompt += "\nAssistant: "
        return prompt
    
    def _send_anthropic(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Anthropic API"""
        max_retries = self._get_max_retries()
        
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Format messages for Anthropic
        system_message = None
        formatted_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                formatted_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Get user-configured anti-duplicate parameters
        anti_dupe_params = self._get_anti_duplicate_params(temperature)
        data = self._build_anthropic_payload(formatted_messages, temperature, max_tokens, anti_dupe_params, system_message)
        
        resp = self._http_request_with_retries(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            expected_status=(200,),
            max_retries=max_retries,
            provider_name="Anthropic"
        )
        json_resp = resp.json()
        content, finish_reason, usage = self._parse_anthropic_json(json_resp)
        return UnifiedResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=json_resp
        )
    
    def _send_mistral(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Mistral API"""
        max_retries = self._get_max_retries()
        api_delay = self._get_send_interval()
        
        if MistralClient and hasattr(self, 'mistral_client'):
            # Use SDK if available
            def _do():
                chat_messages = []
                for msg in messages:
                    chat_messages.append(ChatMessage(role=msg['role'], content=msg['content']))
                response = self.mistral_client.chat(
                    model=self.model,
                    messages=chat_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content if response.choices else ""
                finish_reason = response.choices[0].finish_reason if response.choices else 'stop'
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    raw_response=response
                )
            return self._with_sdk_retries("Mistral", max_retries, _do)
        else:
            # Use HTTP API
            return self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url="https://api.mistral.ai/v1",
                response_name=response_name,
                provider="mistral"
            )
    
    def _send_cohere(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Cohere API"""
        max_retries = self._get_max_retries()
        api_delay = self._get_send_interval()
        
        if cohere and hasattr(self, 'cohere_client'):
            # Use SDK with standardized retry wrapper
            def _do():
                # Format messages for Cohere
                chat_history = []
                message = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        message = msg['content']
                    elif msg['role'] == 'assistant':
                        chat_history.append({"role": "CHATBOT", "message": msg['content']})
                    elif msg['role'] == 'system':
                        message = msg['content'] + "\n\n" + message
                response = self.cohere_client.chat(
                    model=self.model,
                    message=message,
                    chat_history=chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.text
                finish_reason = 'stop'
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    raw_response=response
                )
            return self._with_sdk_retries("Cohere", max_retries, _do)
        else:
            # Use HTTP API with retry logic
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format for HTTP API
            chat_history = []
            message = ""
            
            for msg in messages:
                if msg['role'] == 'user':
                    message = msg['content']
                elif msg['role'] == 'assistant':
                    chat_history.append({"role": "CHATBOT", "message": msg['content']})
            
            data = {
                "model": self.model,
                "message": message,
                "chat_history": chat_history,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            resp = self._http_request_with_retries(
                method="POST",
                url="https://api.cohere.ai/v1/chat",
                headers=headers,
                json=data,
                expected_status=(200,),
                max_retries=max_retries,
                provider_name="Cohere"
            )
            json_resp = resp.json()
            content = json_resp.get("text", "")
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
    
    def _send_ai21(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to AI21 API"""
        max_retries = self._get_max_retries()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages for AI21
        prompt = self._format_prompt(messages, style='ai21')
        
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": max_tokens
        }
        
        resp = self._http_request_with_retries(
            method="POST",
            url=f"https://api.ai21.com/studio/v1/{self.model}/complete",
            headers=headers,
            json=data,
            expected_status=(200,),
            max_retries=max_retries,
            provider_name="AI21"
        )
        json_resp = resp.json()
        completions = json_resp.get("completions", [])
        content = completions[0].get("data", {}).get("text", "") if completions else ""
        return UnifiedResponse(
            content=content,
            finish_reason='stop',
            raw_response=json_resp
        )
    
    
    def _send_replicate(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Replicate API"""
        max_retries = self._get_max_retries()
        api_delay = self._get_send_interval()
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages as single prompt
        prompt = self._format_prompt(messages, style='replicate')
        
        # Replicate uses versioned models
        data = {
            "version": self.model,  # Model should be the version ID
            "input": {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
        # Create prediction
        resp = self._http_request_with_retries(
            method="POST",
            url="https://api.replicate.com/v1/predictions",
            headers=headers,
            json=data,
            expected_status=(201,),
            max_retries=max_retries,
            provider_name="Replicate"
        )
        prediction = resp.json()
        prediction_id = prediction['id']
        
        # Poll for result with GUI delay between polls
        poll_count = 0
        max_polls = 300  # Maximum 5 minutes at 1 second intervals
        
        while poll_count < max_polls:
            if self._cancelled:
                raise UnifiedClientError("Operation cancelled")
            
            resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
                timeout=self.request_timeout  # Use configured timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"Replicate polling error: {resp.status_code}")
            
            result = resp.json()
            
            if result['status'] == 'succeeded':
                content = result.get('output', '')
                if isinstance(content, list):
                    content = ''.join(content)
                break
            elif result['status'] == 'failed':
                raise UnifiedClientError(f"Replicate prediction failed: {result.get('error')}")
            
            # Use GUI delay for polling interval
            time.sleep(min(api_delay, 1))  # But at least 1 second
            poll_count += 1
        
        if poll_count >= max_polls:
            raise UnifiedClientError("Replicate prediction timed out")
        
        return UnifiedResponse(
            content=content,
            finish_reason='stop',
            raw_response=result
        )
    
    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url, 
                                response_name, provider="generic", headers=None, model_override=None) -> UnifiedResponse:
        """Send request to OpenAI-compatible APIs with safety settings"""
        max_retries = self._get_max_retries()
        api_delay = self._get_send_interval()
        
        # Determine effective model for this call (do not rely on shared self.model)
        if model_override is not None:
            effective_model = model_override
        else:
            # Read instance model under microsecond lock to avoid cross-thread contamination
            with self._model_lock:
                effective_model = self.model
        # Provider-specific model normalization (transport-only)
        if provider == 'openrouter':
            for prefix in ('or/', 'openrouter/'):
                if effective_model.startswith(prefix):
                    effective_model = effective_model[len(prefix):]
                    break
            effective_model = effective_model.strip()
        elif provider == 'fireworks':
            if effective_model.startswith('fireworks/'):
                effective_model = effective_model[len('fireworks/') :]
            if not effective_model.startswith('accounts/'):
                effective_model = f"accounts/fireworks/models/{effective_model}"
        elif provider == 'chutes':
            # Strip the 'chutes/' prefix from the model name if present
            if effective_model.startswith('chutes/'):
                effective_model = effective_model[7:]  # Remove 'chutes/' prefix
        
        # CUSTOM ENDPOINT OVERRIDE - Check if enabled and override base_url
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        actual_api_key = self.api_key
        
        # Determine if this is a local endpoint that doesn't need a real API key
        is_local_endpoint = False
        
        # Never override OpenRouter base_url with custom endpoint
        if use_custom_endpoint and provider not in ("gemini-openai", "openrouter"):
            custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            if custom_base_url:
                # Check if it's Azure
                if '.azure.com' in custom_base_url or '.cognitiveservices' in custom_base_url:
                    # Azure needs special client
                    from openai import AzureOpenAI
                    
                    deployment = effective_model  # Use override or instance model as deployment name
                    api_version = os.getenv('AZURE_API_VERSION', '2024-12-01-preview')
                    
                    # Azure endpoint should be just the base URL
                    azure_endpoint = custom_base_url.split('/openai')[0] if '/openai' in custom_base_url else custom_base_url
                    
                    print(f"🔷 Azure endpoint detected")
                    print(f"   Endpoint: {azure_endpoint}")
                    print(f"   Deployment: {deployment}")
                    print(f"   API Version: {api_version}")
                    
                    # Create Azure client
                    for attempt in range(max_retries):
                        try:
                            client = AzureOpenAI(
                                api_key=actual_api_key,
                                api_version=api_version,
                                azure_endpoint=azure_endpoint
                            )
                            
                            # Build params with correct token parameter based on model
                            params = {
                                "model": deployment,
                                "messages": messages,
                                "temperature": temperature
                            }

                            # Normalize token parameter for Azure endpoint
                            norm_max_tokens, norm_max_completion_tokens = self._normalize_token_params(max_tokens, None)
                            if norm_max_completion_tokens is not None:
                                params["max_completion_tokens"] = norm_max_completion_tokens
                            elif norm_max_tokens is not None:
                                params["max_tokens"] = norm_max_tokens

                            # Use Idempotency-Key via headers for compatibility
                            idem_key = self._get_idempotency_key()
                            response = client.chat.completions.create(
                                **params,
                                extra_headers={"Idempotency-Key": idem_key}
                            )
                            
                            # Extract response
                            content = response.choices[0].message.content if response.choices else ""
                            finish_reason = response.choices[0].finish_reason if response.choices else "stop"
                            
                            return UnifiedResponse(
                                content=content,
                                finish_reason=finish_reason,
                                raw_response=response
                            )
                            
                        except Exception as e:
                            error_str = str(e).lower()
                            
                            # Check if this is a content filter error FIRST
                            if ("content_filter" in error_str or 
                                "responsibleaipolicyviolation" in error_str or
                                "content management policy" in error_str or
                                "the response was filtered" in error_str):
                                
                                # This is a content filter error - raise it immediately as prohibited_content
                                print(f"Azure content filter detected: {str(e)[:100]}")
                                raise UnifiedClientError(
                                    f"Azure content blocked: {e}",
                                    error_type="prohibited_content",
                                    http_status=400,
                                    details={"provider": "azure", "original_error": str(e)}
                                )
                            
                            # Only retry for non-content-filter errors
                            if attempt < max_retries - 1:
                                print(f"Azure error (attempt {attempt + 1}): {e}")
                                time.sleep(api_delay)
                                continue
                            
                            raise UnifiedClientError(f"Azure error: {e}")
                
                # Not Azure, continue with regular custom endpoint
                base_url = custom_base_url
                print(f"🔄 Custom endpoint enabled: Overriding {provider} endpoint")
                print(f"   Original: {base_url}")
                print(f"   Override: {custom_base_url}")
                
                # Check if it's Azure
                if '.azure.com' in custom_base_url or '.cognitiveservices' in custom_base_url:
                    # Azure needs special handling
                    deployment = self.model  # Use model as deployment name
                    api_version = os.getenv('AZURE_API_VERSION', '2024-08-01-preview')
                    
                    # Fix Azure URL format
                    if '/openai/deployments/' not in custom_base_url:
                        custom_base_url = f"{custom_base_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                    
                    # Azure uses different auth header
                    if headers is None:
                        headers = {}
                    headers['api-key'] = actual_api_key
                    headers.pop('Authorization', None)  # Remove OpenAI auth
                    
                    print(f"🔷 Azure endpoint detected: {custom_base_url}")
                
                base_url = custom_base_url
                
                # Check if it's a local endpoint
                local_indicators = [
                    'localhost', '127.0.0.1', '0.0.0.0',
                    '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.',
                    '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
                    '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
                    ':11434',  # Ollama default port
                    ':8080',   # Common local API port
                    ':5000',   # Common local API port
                    ':8000',   # Common local API port
                    ':1234',   # LM Studio default port
                    'host.docker.internal',  # Docker host
                ]
                
                # Also check if user explicitly marked it as local
                is_local_llm_env = os.getenv('IS_LOCAL_LLM', '0') == '1'
                
                is_local_endpoint = is_local_llm_env or any(indicator in custom_base_url.lower() for indicator in local_indicators)
                
                if is_local_endpoint:
                    actual_api_key = "dummy-key-for-local-llm"
                    #print(f"   📍 Detected local endpoint, using dummy API key")
                else:
                    #print(f"   ☁️  Using actual API key for cloud endpoint")
                    pass
        
        # For all other providers, use the actual API key
        # Remove the special case for gemini-openai - it needs the real API key
        if not is_local_endpoint:
            #print(f"   Using actual API key for {provider}")
            pass
        
        # Check if safety settings are disabled via GUI toggle
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Debug logging for ElectronHub
        if provider == "electronhub":
            logger.debug(f"ElectronHub API call - Messages structure:")
            for i, msg in enumerate(messages):
                logger.debug(f"  Message {i}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}")
                if msg.get('role') == 'system':
                    logger.debug(f"  System prompt preview: {msg.get('content', '')[:100]}...")
        
        # Use OpenAI SDK for providers known to work well with it
        sdk_compatible = ['deepseek', 'together', 'mistral', 'yi', 'qwen', 'moonshot', 'groq', 
                         'electronhub', 'openrouter', 'fireworks', 'xai', 'gemini-openai', 'chutes']
        
        if openai and provider in sdk_compatible:
            # Use OpenAI SDK with custom base URL
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    client = self._get_openai_client(base_url=base_url, api_key=actual_api_key)
                    
                    # Check if this is Gemini via OpenAI endpoint
                    is_gemini_endpoint = provider == "gemini-openai" or effective_model.lower().startswith('gemini')
                    
                    # Get user-configured anti-duplicate parameters
                    anti_dupe_params = self._get_anti_duplicate_params(temperature)

                    norm_max_tokens, norm_max_completion_tokens = self._normalize_token_params(max_tokens, None)
                    params = {
                        "model": effective_model,
                        "messages": messages,
                        "temperature": temperature,
                        **anti_dupe_params
                    }
                    if norm_max_completion_tokens is not None:
                        params["max_completion_tokens"] = norm_max_completion_tokens
                    elif norm_max_tokens is not None:
                        params["max_tokens"] = norm_max_tokens
                    
                    # Add safety parameters for providers that support them
                    # Note: Together AI doesn't support the 'moderation' parameter
                    if disable_safety and provider in ["groq", "fireworks"]:
                        params["moderation"] = False
                        logger.info(f"🔓 Safety moderation disabled for {provider}")
                    elif disable_safety and provider == "together":
                        # Together AI handles safety differently - no moderation parameter
                        logger.info(f"🔓 Safety settings note: {provider} doesn't support moderation parameter")
                    
                    # Use Idempotency-Key header to avoid unsupported kwarg on some endpoints
                    idem_key = self._get_idempotency_key()
                    resp = client.chat.completions.create(
                        **params,
                        extra_headers={"Idempotency-Key": idem_key}
                    )
                    
                    # Enhanced extraction for Gemini endpoints
                    content = None
                    finish_reason = 'stop'
                    
                    # Extract content with Gemini awareness
                    if hasattr(resp, 'choices') and resp.choices:
                        choice = resp.choices[0]
                        
                        if hasattr(choice, 'finish_reason'):
                            finish_reason = choice.finish_reason or 'stop'
                        
                        if hasattr(choice, 'message'):
                            message = choice.message
                            if message is None:
                                content = ""
                                if is_gemini_endpoint:
                                    content = "[GEMINI RETURNED NULL MESSAGE]"
                                    finish_reason = 'content_filter'
                            elif hasattr(message, 'content'):
                                content = message.content or ""
                                if content is None and is_gemini_endpoint:
                                    content = "[BLOCKED BY GEMINI SAFETY FILTER]"
                                    finish_reason = 'content_filter'
                            elif hasattr(message, 'text'):
                                content = message.text
                            elif isinstance(message, str):
                                content = message
                            else:
                                content = str(message) if message else ""
                        elif hasattr(choice, 'text'):
                            content = choice.text
                        else:
                            content = ""
                    else:
                        content = ""
                    
                    # Normalize finish reasons
                    if finish_reason in ["max_tokens", "max_length"]:
                        finish_reason = "length"
                    
                    usage = None
                    if hasattr(resp, 'usage'):
                        usage = {
                            'prompt_tokens': resp.usage.prompt_tokens,
                            'completion_tokens': resp.usage.completion_tokens,
                            'total_tokens': resp.usage.total_tokens
                        }
                    
                    self._save_response(content, response_name)
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw_response=resp
                    )
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        raise UnifiedClientError(f"{provider} rate limit: {e}", error_type="rate_limit")
                    if not self._multi_key_mode and attempt < max_retries - 1:
                        print(f"{provider} SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    elif self._multi_key_mode:
                        raise UnifiedClientError(f"{provider} error: {e}", error_type="api_error")
                    raise UnifiedClientError(f"{provider} SDK error: {e}")
        else:
            # Use HTTP API with retry logic
            headers = self._build_openai_headers(provider, actual_api_key, headers)
            # Provider-specific header tweaks
            if provider == 'openrouter':
                headers['HTTP-Referer'] = os.getenv('OPENROUTER_REFERER', 'https://github.com/Shirochi-stack/Glossarion')
                headers['X-Title'] = os.getenv('OPENROUTER_APP_NAME', 'Glossarion Translation')
                headers['X-Proxy-TTL'] = '0'
            elif provider == 'zhipu':
                headers["Authorization"] = f"Bearer {actual_api_key}"
            elif provider == 'baidu':
                headers["Content-Type"] = "application/json"
            data = {
                "model": effective_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add Perplexity-specific options for Sonar models
            if provider == 'perplexity' and 'sonar' in effective_model.lower():
                data['search_domain_filter'] = ['perplexity.ai']
                data['return_citations'] = True
                data['search_recency_filter'] = 'month'
            
            # Apply safety flags
            self._apply_openai_safety(provider, disable_safety, data, headers)
            # Save OpenRouter config if requested
            if provider == 'openrouter' and os.getenv("SAVE_PAYLOAD", "1") == "1":
                cfg = {
                    "provider": "openrouter",
                    "timestamp": datetime.now().isoformat(),
                    "model": effective_model,
                    "safety_disabled": disable_safety,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                self._save_openrouter_config(cfg, response_name)
            # Endpoint and idempotency
            endpoint = "/chat/completions"
            headers["Idempotency-Key"] = self._get_idempotency_key()
            resp = self._http_request_with_retries(
                method="POST",
                url=f"{base_url}{endpoint}",
                headers=headers,
                json=data,
                expected_status=(200,),
                max_retries=max_retries,
                provider_name=provider,
                use_session=True
            )
            json_resp = resp.json()
            choices = json_resp.get("choices", [])
            if not choices:
                raise UnifiedClientError(f"{provider} API returned no choices")
            content, finish_reason, usage = self._extract_openai_json(json_resp)
            # ElectronHub truncation detection
            if provider == "electronhub" and content:
                if len(content) < 50 and "cannot" in content.lower():
                    finish_reason = "content_filter"
                    print(f"ElectronHub likely refused content: {content[:100]}")
                elif finish_reason == "stop":
                    if self._detect_silent_truncation(content, messages, self.context):
                        finish_reason = "length"
                        print("ElectronHub reported 'stop' but content appears truncated")
                        print(f"🔍 ElectronHub: Detected silent truncation despite 'stop' status")
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=json_resp
            )
    
    def _send_openai(self, messages, temperature, max_tokens, max_completion_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenAI API with proper token parameter handling"""
        # CRITICAL: Check if individual endpoint is applied first
        if (hasattr(self, '_individual_endpoint_applied') and self._individual_endpoint_applied and 
            hasattr(self, 'openai_client') and self.openai_client):
            individual_base_url = getattr(self.openai_client, 'base_url', None)
            if individual_base_url:
                base_url = str(individual_base_url).rstrip('/')
            else:
                base_url = 'https://api.openai.com/v1'
        else:
            # Fallback to global custom endpoint logic
            custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
            
            if custom_base_url and use_custom_endpoint:
                base_url = custom_base_url
            else:
                base_url = 'https://api.openai.com/v1'
        
        # For OpenAI, we need to handle max_completion_tokens properly
        return self._send_openai_compatible(
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens if not max_completion_tokens else max_completion_tokens,
            base_url=base_url,
            response_name=response_name,
            provider="openai"
        )

    def _send_openai_provider_router(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Generic router for many OpenAI-compatible providers to reduce wrapper duplication."""
        provider = self._get_actual_provider()
        
        # Provider URL mapping dictionary
        provider_urls = {
            'yi': lambda: os.getenv("YI_API_BASE_URL", "https://api.01.ai/v1"),
            'qwen': "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            'baichuan': "https://api.baichuan-ai.com/v1",
            'zhipu': "https://open.bigmodel.cn/api/paas/v4",
            'moonshot': "https://api.moonshot.cn/v1",
            'groq': lambda: os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1"),
            'baidu': "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            'tencent': "https://hunyuan.cloud.tencent.com/v1",
            'iflytek': "https://spark-api.xf-yun.com/v1",
            'bytedance': "https://maas-api.vercel.app/v1",
            'minimax': "https://api.minimax.chat/v1",
            'sensenova': "https://api.sensenova.cn/v1",
            'internlm': "https://api.internlm.org/v1",
            'tii': "https://api.tii.ae/v1",
            'microsoft': "https://api.microsoft.com/v1",
            'databricks': lambda: f"{os.getenv('DATABRICKS_API_URL', 'https://YOUR-WORKSPACE.databricks.com')}/serving/endpoints",
            'together': "https://api.together.xyz/v1",
            'openrouter': "https://openrouter.ai/api/v1",
            'fireworks': lambda: os.getenv("FIREWORKS_API_URL", "https://api.fireworks.ai/inference/v1"),
            'xai': lambda: os.getenv("XAI_API_URL", "https://api.x.ai/v1"),
            'deepseek': lambda: os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),
            'perplexity': "https://api.perplexity.ai",
            'chutes': lambda: os.getenv("CHUTES_API_URL", "https://llm.chutes.ai/v1"),
            'salesforce': lambda: os.getenv("SALESFORCE_API_URL", "https://api.salesforce.com/v1"),
            'bigscience': "https://api.together.xyz/v1",  # Together AI fallback
            'meta': "https://api.together.xyz/v1"  # Together AI fallback
        }
        
        # Get base URL from mapping
        url_spec = provider_urls.get(provider)
        if url_spec:
            base_url = url_spec() if callable(url_spec) else url_spec
        else:
            # Fallback to base OpenAI-compatible flow if unknown
            base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', 'https://api.openai.com/v1')
        
        return self._send_openai_compatible(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            response_name=response_name,
            provider=provider
        )
    
    def _send_azure(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Azure OpenAI"""
        # Prefer per-key (individual) endpoint/version when present, then fall back to env vars
        endpoint = getattr(self, 'azure_endpoint', None) or \
                   getattr(self, 'current_key_azure_endpoint', None) or \
                   os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com")
        api_version = getattr(self, 'azure_api_version', None) or \
                      getattr(self, 'current_key_azure_api_version', None) or \
                      os.getenv("AZURE_API_VERSION", "2024-02-01")
        
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = "https://" + endpoint
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Azure uses a different URL structure
        base_url = f"{endpoint.rstrip('/')}/openai/deployments/{self.model}"
        url = f"{base_url}/chat/completions?api-version={api_version}"
        
        data = {
            "messages": messages,
            "temperature": temperature
        }
        
        # Use _is_o_series_model to determine which token parameter to use
        if self._is_o_series_model():
            data["max_completion_tokens"] = max_tokens
        else:
            data["max_tokens"] = max_tokens
        
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                # Treat all 400s as prohibited_content to trigger fallback keys cleanly
                if resp.status_code == 400:
                    raise UnifiedClientError(
                        f"Azure OpenAI error: {resp.status_code} - {resp.text}",
                        error_type="prohibited_content",
                        http_status=400
                    )
                # Other errors propagate normally with status code
                raise UnifiedClientError(
                    f"Azure OpenAI error: {resp.status_code} - {resp.text}",
                    http_status=resp.status_code
                )
            
            json_resp = resp.json()
            content, finish_reason, usage = self._extract_openai_json(json_resp)
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Azure OpenAI error: {e}")
            raise UnifiedClientError(f"Azure OpenAI error: {e}")
    
    def _send_google_palm(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Google PaLM API"""
        # PaLM is being replaced by Gemini, but included for completeness
        return self._send_gemini(messages, temperature, max_tokens, response_name)
    
    def _send_alephalpha(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Aleph Alpha API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages for Aleph Alpha (simple concatenation)
        prompt = self._format_prompt(messages, style='replicate')
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "maximum_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            resp = self._http_request_with_retries(
                method="POST",
                url="https://api.aleph-alpha.com/complete",
                headers=headers,
                json=data,
                expected_status=(200,),
                max_retries=3,
                provider_name="AlephAlpha"
            )
            json_resp = resp.json()
            content = json_resp.get('completions', [{}])[0].get('completion', '')
            
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Aleph Alpha error: {e}")
            raise UnifiedClientError(f"Aleph Alpha error: {e}")
      
    def _send_huggingface(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to HuggingFace Inference API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages for HuggingFace (simple concatenation)
        prompt = self._format_prompt(messages, style='replicate')
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            resp = self._http_request_with_retries(
                method="POST",
                url=f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=data,
                expected_status=(200,),
                max_retries=3,
                provider_name="HuggingFace"
            )
            json_resp = resp.json()
            content = ""
            if isinstance(json_resp, list) and json_resp:
                content = json_resp[0].get('generated_text', '')
            
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"HuggingFace error: {e}")
            raise UnifiedClientError(f"HuggingFace error: {e}")
    
    def _send_vertex_model_garden_image(self, messages, image_base64, temperature, max_tokens, response_name):
        """Send image request to Vertex AI Model Garden"""
        # For now, we can just call the regular send method since Vertex AI 
        # handles images in the message format
        
        # Convert image to message format that Vertex AI expects
        image_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": messages[-1]['content'] if messages else ""},
                {"type": "image", "image": {"base64": image_base64}}
            ]
        }
        
        # Replace last message with image message
        messages_with_image = messages[:-1] + [image_message]
        
        # Use the regular Vertex AI send method
        return self._send_vertex_model_garden(messages_with_image, temperature, max_tokens, response_name=response_name)

    def _is_o_series_model(self) -> bool:
        """Check if the current model is an o-series model (o1, o3, o4, etc.) or GPT-5"""
        if not self.model:
            return False
        
        model_lower = self.model.lower()
        
        # Check for specific patterns
        if 'o1-preview' in model_lower or 'o1-mini' in model_lower:
            return True
        
        # Check for o3 models
        if 'o3-mini' in model_lower or 'o3-pro' in model_lower:
            return True
        
        # Check for o4 models
        if 'o4-mini' in model_lower:
            return True
        
        # Check for GPT-5 models (including variants)
        if 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return True
        
        # Check if it starts with o followed by a digit
        if len(model_lower) >= 2 and model_lower[0] == 'o' and model_lower[1].isdigit():
            return True
        
        return False

    def _prepare_gemini_image_content(self, messages, image_base64):
        """Prepare image content for Gemini API - supports both single and multiple images"""

        # Check if this is a multi-image request (messages contain content arrays)
        is_multi_image = False
        for msg in messages:
            if isinstance(msg.get('content'), list):
                for part in msg['content']:
                    if part.get('type') == 'image_url':
                        is_multi_image = True
                        break
        
        if is_multi_image:
            # Handle multi-image format
            contents = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"Instructions: {msg['content']}"}]
                    })
                elif msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        contents.append({
                            "role": "user",
                            "parts": [{"text": msg['content']}]
                        })
                    elif isinstance(msg['content'], list):
                        parts = []
                        for part in msg['content']:
                            if part['type'] == 'text':
                                parts.append({"text": part['text']})
                            elif part['type'] == 'image_url':
                                image_data = part['image_url']['url']
                                if image_data.startswith('data:'):
                                    base64_data = image_data.split(',')[1]
                                else:
                                    base64_data = image_data
                                
                                mime_type = "image/png"
                                if 'jpeg' in image_data or 'jpg' in image_data:
                                    mime_type = "image/jpeg"
                                elif 'webp' in image_data:
                                    mime_type = "image/webp"
                                
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64_data
                                    }
                                })
                        
                        contents.append({
                            "role": "user",
                            "parts": parts
                        })
        else:
            # Handle single image format (backward compatibility)
            formatted_parts = []
            for msg in messages:
                if msg.get('role') == 'system':
                    formatted_parts.append(f"Instructions: {msg['content']}")
                elif msg.get('role') == 'user':
                    formatted_parts.append(f"User: {msg['content']}")
            
            text_prompt = "\n\n".join(formatted_parts)
            
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": text_prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }}
                    ]
                }
            ]
        
        return contents
        
    # Removed: _send_openai_image
    # OpenAI-compatible providers handle images within messages via _get_response and _send_openai_compatible
    
    def _send_anthropic_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Anthropic API"""
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Format messages with image
        system_message = None
        formatted_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            elif msg['role'] == 'user':
                # Add image to user message
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg['content']
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                })
            else:
                formatted_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Get user-configured anti-duplicate parameters
        anti_dupe_params = self._get_anti_duplicate_params(temperature)
        data.update(anti_dupe_params)  # Add user's custom parameters

        if system_message:
            data["system"] = system_message
            
        try:
            resp = self._http_request_with_retries(
                method="POST",
                url="https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                expected_status=(200,),
                max_retries=3,
                provider_name="Anthropic Image"
            )
            json_resp = resp.json()
            content, finish_reason, usage = self._parse_anthropic_json(json_resp)
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Anthropic Vision API error: {e}")
            raise UnifiedClientError(f"Anthropic Vision API error: {e}")
    
    # Removed: _send_electronhub_image (handled via _send_openai_compatible in _get_response)
    
    def _send_poe_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request using poe-api-wrapper"""
        try:
            from poe_api_wrapper import PoeApi
        except ImportError:
            raise UnifiedClientError(
                "poe-api-wrapper not installed. Run: pip install poe-api-wrapper"
            )
        
        # Parse cookies using robust parser
        tokens = self._parse_poe_tokens(self.api_key)
        if 'p-b' not in tokens or not tokens['p-b']:
            raise UnifiedClientError(
                "POE tokens missing. Provide cookies as 'p-b:VALUE|p-lat:VALUE' or 'p-b=VALUE; p-lat=VALUE'",
                error_type="auth_error"
            )
        if 'p-lat' not in tokens:
            tokens['p-lat'] = ''
            logger.info("No p-lat cookie provided; proceeding without it")
        
        logger.info(f"Tokens being sent for image: p-b={len(tokens.get('p-b', ''))} chars, p-lat={len(tokens.get('p-lat', ''))} chars")
        
        try:
            # Create Poe client (try to pass proxy/headers if supported)
            poe_kwargs = {}
            ua = os.getenv("POE_USER_AGENT") or os.getenv("HTTP_USER_AGENT")
            if ua:
                poe_kwargs["headers"] = {"User-Agent": ua, "Referer": "https://poe.com/", "Origin": "https://poe.com"}
            proxy = os.getenv("POE_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
            if proxy:
                poe_kwargs["proxy"] = proxy
            try:
                poe_client = PoeApi(tokens=tokens, **poe_kwargs)
            except TypeError:
                poe_client = PoeApi(tokens=tokens)
                try:
                    if ua and hasattr(poe_client, "session") and hasattr(poe_client.session, "headers"):
                        poe_client.session.headers.update({"User-Agent": ua, "Referer": "https://poe.com/", "Origin": "https://poe.com"})
                except Exception:
                    pass
            
            # Get bot name - use vision-capable bots
            requested_model = self.model.replace('poe/', '', 1)
            bot_map = {
                # Vision-capable models
                'gpt-4-vision': 'GPT-4V',
                'gpt-4v': 'GPT-4V',
                'claude-3-opus': 'claude_3_opus',  # Claude 3 models support vision
                'claude-3-sonnet': 'claude_3_sonnet',
                'claude-3-haiku': 'claude_3_haiku',
                'gemini-pro-vision': 'gemini_pro_vision',
                'gemini-2.5-flash': 'gemini_1_5_flash',  # Gemini 1.5 supports vision
                'gemini-2.5-pro': 'gemini_1_5_pro',
                
                # Fallback to regular models
                'gpt-4': 'beaver',
                'claude': 'a2',
                'assistant': 'assistant',
            }
            bot_name = bot_map.get(requested_model.lower(), requested_model)
            logger.info(f"Using bot name for vision: {bot_name}")
            
            # Convert messages to prompt
            prompt = self._format_prompt(messages, style='replicate')
            
            # Note: poe-api-wrapper's image support varies by version
            # Some versions support file_path parameter, others need different approaches
            full_response = ""
            
            # POE file_path support is inconsistent; fall back to plain prompt
            for chunk in poe_client.send_message(bot_name, prompt):
                if 'response' in chunk:
                    full_response = chunk['response']
            
        except Exception as img_error:
            print(f"Image handling error: {img_error}")
            # Fall back to text-only message
            print("Falling back to text-only message due to image error")
            for chunk in poe_client.send_message(bot_name, prompt):
                if 'response' in chunk:
                    full_response = chunk['response']
            
            # Get the final text
            final_text = chunk.get('text', full_response) if 'chunk' in locals() else full_response
            
            if not final_text:
                raise UnifiedClientError(
                    "POE returned empty response for image. "
                    "The bot may not support image inputs or the image format is unsupported."
                )
            
            return UnifiedResponse(
                content=final_text,
                finish_reason="stop",
                raw_response=chunk if 'chunk' in locals() else {"response": full_response}
            )
            
        except Exception as e:
            print(f"Poe image API error details: {str(e)}")
            error_str = str(e).lower()
            
            if self._is_rate_limit_error(e):
                raise UnifiedClientError(
                    "POE rate limit exceeded. Please wait before trying again.",
                    error_type="rate_limit"
                )
            elif "auth" in error_str or "unauthorized" in error_str:
                raise UnifiedClientError(
                    "POE authentication failed. Your cookies may be expired.",
                    error_type="auth_error"
                )
            elif "not support" in error_str or "vision" in error_str:
                raise UnifiedClientError(
                    f"The selected POE bot '{requested_model}' may not support image inputs. "
                    "Try using a vision-capable model like gpt-4-vision or claude-3-opus.",
                    error_type="capability_error"
                )
            
            raise UnifiedClientError(f"Poe image API error: {e}")
    
    def _log_truncation_failure(self, messages, response_content, finish_reason, context=None, attempts=None, error_details=None):
        """Log truncation failures for analysis - saves to CSV, TXT, and HTML in truncation_logs subfolder"""
        try:
            # Use output directory if provided, otherwise current directory
            base_dir = self.output_dir if self.output_dir else "."
            
            # Create truncation_logs subfolder inside the output directory
            log_dir = os.path.join(base_dir, "truncation_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate log filename with date
            log_date = datetime.now().strftime("%Y%m")
            
            # CSV log file (keeping for compatibility)
            csv_log_file = os.path.join(log_dir, f"truncation_failures_{log_date}.csv")
            
            # TXT log file (human-readable format)
            txt_log_file = os.path.join(log_dir, f"truncation_failures_{log_date}.txt")
            
            # HTML log file (web-viewable format)
            html_log_file = os.path.join(log_dir, f"truncation_failures_{log_date}.html")
            
            # Summary file to track truncated outputs
            summary_file = os.path.join(log_dir, f"truncation_summary_{log_date}.json")
            
            # Check if CSV file exists to determine if we need headers
            csv_file_exists = os.path.exists(csv_log_file)
            
            # Extract output filename - UPDATED LOGIC
            output_filename = 'unknown'
            
            # PRIORITY 1: Use the actual output filename if set via set_output_filename()
            if hasattr(self, '_actual_output_filename') and self._actual_output_filename:
                output_filename = self._actual_output_filename
            # PRIORITY 2: Use current output file if available
            elif hasattr(self, '_current_output_file') and self._current_output_file:
                output_filename = self._current_output_file
            # PRIORITY 3: Use tracked response filename from _save_response
            elif hasattr(self, '_last_response_filename') and self._last_response_filename:
                # Skip if it's a generic Payloads filename
                if not self._last_response_filename.startswith(('response_', 'translation_')):
                    output_filename = self._last_response_filename
            
            # FALLBACK: Try to extract from context/messages if no filename was set
            if output_filename == 'unknown':
                if context == 'translation':
                    # Try to extract chapter/response filename
                    chapter_match = re.search(r'Chapter (\d+)', str(messages))
                    if chapter_match:
                        chapter_num = chapter_match.group(1)
                        # Use the standard format that matches book output
                        safe_title = f"Chapter_{chapter_num}"
                        output_filename = f"response_{chapter_num.zfill(3)}_{safe_title}.html"
                    else:
                        # Try chunk pattern
                        chunk_match = re.search(r'Chunk (\d+)/(\d+).*Chapter (\d+)', str(messages))
                        if chunk_match:
                            chunk_num = chunk_match.group(1)
                            chapter_num = chunk_match.group(3)
                            safe_title = f"Chapter_{chapter_num}"
                            output_filename = f"response_{chapter_num.zfill(3)}_{safe_title}_chunk_{chunk_num}.html"
                elif context == 'image_translation':
                    # Extract image filename if available
                    img_match = re.search(r'([\w\-]+\.(jpg|jpeg|png|gif|webp))', str(messages), re.IGNORECASE)
                    if img_match:
                        output_filename = f"image_{img_match.group(1)}"
                        
            # Load or create summary tracking
            summary_data = {"truncated_files": set(), "total_truncations": 0, "by_type": {}}
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                        summary_data["truncated_files"] = set(loaded_data.get("truncated_files", []))
                        summary_data["total_truncations"] = loaded_data.get("total_truncations", 0)
                        summary_data["by_type"] = loaded_data.get("by_type", {})
                except:
                    pass
            
            # Update summary
            summary_data["truncated_files"].add(output_filename)
            summary_data["total_truncations"] += 1
            truncation_type_key = f"{finish_reason}_{context or 'unknown'}"
            summary_data["by_type"][truncation_type_key] = summary_data["by_type"].get(truncation_type_key, 0) + 1
            
            # Save summary
            save_summary = {
                "truncated_files": sorted(list(summary_data["truncated_files"])),
                "total_truncations": summary_data["total_truncations"],
                "by_type": summary_data["by_type"],
                "last_updated": datetime.now().isoformat()
            }
            with self._file_write_lock:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(save_summary, f, indent=2, ensure_ascii=False)
            
            # Prepare log entry
            # Compute safe input length and serialize error details safely
            def _safe_content_len(c):
                if isinstance(c, str):
                    return len(c)
                if isinstance(c, list):
                    total = 0
                    for p in c:
                        if isinstance(p, dict):
                            if isinstance(p.get('text'), str):
                                total += len(p['text'])
                            elif isinstance(p.get('image_url'), dict):
                                url = p['image_url'].get('url')
                                if isinstance(url, str):
                                    total += len(url)
                        elif isinstance(p, str):
                            total += len(p)
                    return total
                return len(str(c)) if c is not None else 0
            
            input_length_value = sum(_safe_content_len(msg.get('content')) for msg in (messages or []))
            
            truncation_type_label = 'explicit' if finish_reason == 'length' else 'silent'
            
            error_details_str = ''
            if error_details is not None:
                try:
                    error_details_str = json.dumps(error_details, ensure_ascii=False)
                except Exception:
                    error_details_str = str(error_details)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'provider': self.client_type,
                'context': context or 'unknown',
                'finish_reason': finish_reason,
                'attempts': attempts or 1,
                'input_length': input_length_value,
                'output_length': len(response_content) if response_content else 0,
                'truncation_type': truncation_type_label,
                'content_refused': 'yes' if finish_reason == 'content_filter' else 'no',
                'last_50_chars': response_content[-50:] if response_content else '',
                'error_details': error_details_str,
                'input_preview': self._get_safe_preview(messages),
                'output_preview': response_content[:200] if response_content else '',
                'output_filename': output_filename  # Add output filename to log entry
            }
            
            # Write to CSV
            with self._file_write_lock:
                with open(csv_log_file, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = [
                        'timestamp', 'model', 'provider', 'context', 'finish_reason',
                        'attempts', 'input_length', 'output_length', 'truncation_type',
                        'content_refused', 'last_50_chars', 'error_details',
                        'input_preview', 'output_preview', 'output_filename'
                    ]
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # Write header if new file
                    if not csv_file_exists:
                        writer.writeheader()
                    
                    writer.writerow(log_entry)
            
            # Write to TXT file with human-readable format
            with self._file_write_lock:
                with open(txt_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TRUNCATION LOG ENTRY - {log_entry['timestamp']}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Output File: {log_entry['output_filename']}\n")
                    f.write(f"Model: {log_entry['model']}\n")
                    f.write(f"Provider: {log_entry['provider']}\n")
                    f.write(f"Context: {log_entry['context']}\n")
                    f.write(f"Finish Reason: {log_entry['finish_reason']}\n")
                    f.write(f"Attempts: {log_entry['attempts']}\n")
                    f.write(f"Input Length: {log_entry['input_length']} chars\n")
                    f.write(f"Output Length: {log_entry['output_length']} chars\n")
                    f.write(f"Truncation Type: {log_entry['truncation_type']}\n")
                    f.write(f"Content Refused: {log_entry['content_refused']}\n")
                    
                    if log_entry['error_details']:
                        f.write(f"Error Details: {log_entry['error_details']}\n")
                    
                    f.write(f"\n--- Input Preview ---\n")
                    f.write(f"{log_entry['input_preview']}\n")
                    
                    f.write(f"\n--- Output Preview ---\n")
                    f.write(f"{log_entry['output_preview']}\n")
                    
                    if log_entry['last_50_chars']:
                        f.write(f"\n--- Last 50 Characters ---\n")
                        f.write(f"{log_entry['last_50_chars']}\n")
                    
                    f.write(f"\n{'='*80}\n")
            
            # Write to HTML file with nice formatting
            html_file_exists = os.path.exists(html_log_file)
            
            # Create or update HTML file
            if not html_file_exists:
                # Create new HTML file with header
                html_content = ('<!DOCTYPE html>\n'
                    '<html>\n'
                    '<head>\n'
                    '<meta charset="UTF-8">\n'
                    '<title>Truncation Failures Log</title>\n'
                    '<style>\n'
                    'body { font-family: Arial, sans-serif; background-color: #f5f5f5; margin: 20px; line-height: 1.6; }\n'
                    '.summary { background-color: #e3f2fd; border: 2px solid #1976d2; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n'
                    '.summary h2 { color: #1976d2; margin-top: 0; }\n'
                    '.summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }\n'
                    '.stat-box { background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #ddd; }\n'
                    '.stat-label { font-size: 12px; color: #666; text-transform: uppercase; }\n'
                    '.stat-value { font-size: 24px; font-weight: bold; color: #333; }\n'
                    '.truncated-files { background-color: white; padding: 15px; border-radius: 4px; border: 1px solid #ddd; max-height: 200px; overflow-y: auto; }\n'
                    '.truncated-files h3 { margin-top: 0; color: #333; }\n'
                    '.file-list { display: flex; flex-wrap: wrap; gap: 8px; }\n'
                    '.file-badge { background-color: #ffecb3; border: 1px solid #ffc107; padding: 4px 8px; border-radius: 4px; font-size: 13px; font-family: \'Courier New\', monospace; }\n'
                    '.log-entry { background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n'
                    '.timestamp { color: #666; font-size: 14px; margin-bottom: 10px; }\n'
                    '.metadata { display: grid; grid-template-columns: 200px 1fr; gap: 10px; margin-bottom: 15px; }\n'
                    '.label { font-weight: bold; color: #333; }\n'
                    '.value { color: #555; }\n'
                    '.content-preview { background-color: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; margin: 10px 0; font-family: \'Courier New\', monospace; font-size: 13px; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; }\n'
                    '.error { color: #d9534f; }\n'
                    '.warning { color: #f0ad4e; }\n'
                    '.section-title { font-weight: bold; color: #2c5aa0; margin-top: 15px; margin-bottom: 5px; }\n'
                    'h1 { color: #333; border-bottom: 2px solid #2c5aa0; padding-bottom: 10px; }\n'
                    '.truncation-type-silent { background-color: #fff3cd; border-left: 4px solid #ffc107; }\n'
                    '.truncation-type-explicit { background-color: #f8d7da; border-left: 4px solid #dc3545; }\n'
                    '</style>\n'
                    '</head>\n'
                    '<body>\n'
                    '<h1>Truncation Failures Log</h1>\n'
                    '<div id="summary-container">\n'
                    '<!-- Summary will be inserted here -->\n'
                    '</div>\n'
                    '<div id="entries-container">\n'
                    '<!-- Log entries will be inserted here -->\n'
                    '</div>\n')
                # Write initial HTML structure
                with self._file_write_lock:
                    with open(html_log_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                # Make sure HTML is properly closed
                if not html_content.rstrip().endswith('</html>'):
                    with self._file_write_lock:
                        with open(html_log_file, 'a', encoding='utf-8') as f:
                            f.write('\n</body>\n</html>')
            
            # Read existing HTML content
            with self._file_write_lock:
                with open(html_log_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            
            # Generate summary HTML
            summary_html = f"""
            <div class=\"summary\">
            <h2>Summary</h2>
            <div class=\"summary-stats\">
            <div class=\"stat-box\">
            <div class=\"stat-label\">Total Truncations</div>
            <div class=\"stat-value\">{summary_data['total_truncations']}</div>
            </div>
            <div class=\"stat-box\">
            <div class=\"stat-label\">Affected Files</div>
            <div class=\"stat-value\">{len(summary_data['truncated_files'])}</div>
            </div>
            </div>
            <div class=\"truncated-files\">
            <h3>Truncated Output Files:</h3>
            <div class=\"file-list\">
            """
            
            # Add file badges
            for filename in sorted(summary_data['truncated_files']):
                summary_html += f'                <span class=\"file-badge\">{html.escape(filename)}</span>\n'
            
            summary_html += """</div>
            </div>
            </div>
            """
            
            # Update summary in HTML
            if '<div id="summary-container">' in html_content:
                # Replace existing summary between summary-container and entries-container
                start_marker = '<div id="summary-container">'
                end_marker = '<div id="entries-container">'
                start = html_content.find(start_marker) + len(start_marker)
                end = html_content.find(end_marker, start)
                if end != -1:
                    html_content = html_content[:start] + '\n' + summary_html + '\n' + html_content[end:]
                else:
                    # Fallback: insert before closing </body>
                    tail_idx = html_content.rfind('</body>')
                    if tail_idx != -1:
                        html_content = html_content[:start] + '\n' + summary_html + '\n' + html_content[tail_idx:]
            
            # Generate new log entry HTML
            truncation_class = 'truncation-type-silent' if log_entry['truncation_type'] == 'silent' else 'truncation-type-explicit'
            
            entry_html = f"""<div class=\"log-entry {truncation_class}\">\n            <div class=\"timestamp\">{log_entry["timestamp"]} - Output: {html.escape(output_filename)}</div>\n            <div class=\"metadata\">\n            <span class=\"label\">Model:</span><span class=\"value\">{html.escape(str(log_entry["model"]))}</span>\n            <span class=\"label\">Provider:</span><span class=\"value\">{html.escape(str(log_entry["provider"]))}</span>\n            <span class=\"label\">Context:</span><span class=\"value\">{html.escape(str(log_entry["context"]))}</span>\n            <span class=\"label\">Finish Reason:</span><span class=\"value {("error" if log_entry["finish_reason"] == "content_filter" else "warning")}">{html.escape(str(log_entry["finish_reason"]))}</span>\n            <span class=\"label\">Attempts:</span><span class=\"value\">{log_entry["attempts"]}</span>\n            <span class=\"label\">Input Length:</span><span class=\"value\">{log_entry["input_length"]:,} chars</span>\n            <span class=\"label\">Output Length:</span><span class=\"value\">{log_entry["output_length"]:,} chars</span>\n            <span class=\"label\">Truncation Type:</span><span class=\"value\">{html.escape(str(log_entry["truncation_type"]))}</span>\n            <span class=\"label\">Content Refused:</span><span class=\"value {("error" if log_entry["content_refused"] == "yes" else "")}">{html.escape(str(log_entry["content_refused"]))}</span>\n            """
            
            if log_entry['error_details']:
                entry_html += f'            <span class=\"label\">Error Details:</span><span class=\"value error\">{html.escape(str(log_entry["error_details"]))}</span>\n'
            
            entry_html += f"""</div>
                <div class=\"section-title\">Input Preview</div>
                <div class=\"content-preview\">{html.escape(str(log_entry["input_preview"]))}</div>
                <div class=\"section-title\">Output Preview</div>
                <div class=\"content-preview\">{html.escape(str(log_entry["output_preview"]))}</div>
                """
            
            if log_entry['last_50_chars']:
                entry_html += f"""<div class=\"section-title\">Last 50 Characters</div>
                <div class=\"content-preview\">{html.escape(str(log_entry["last_50_chars"]))}</div>
                """
            
            entry_html += """</div>"""                                   
            
            # Insert new entry
            if '<div id="entries-container">' in html_content:
                insert_pos = html_content.find('<div id="entries-container">') + len('<div id="entries-container">')
                # Find the next newline after the container div
                newline_pos = html_content.find('\n', insert_pos)
                if newline_pos != -1:
                    insert_pos = newline_pos + 1
                html_content = html_content[:insert_pos] + entry_html + html_content[insert_pos:]
            else:
                # Fallback: append before closing body tag
                insert_pos = html_content.rfind('</body>')
                html_content = html_content[:insert_pos] + entry_html + '\n' + html_content[insert_pos:]
            
            # Write updated HTML
            with self._file_write_lock:
                with open(html_log_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            # Log to console with FULL PATH so user knows where to look
            csv_log_path = os.path.abspath(csv_log_file)
            txt_log_path = os.path.abspath(txt_log_file)
            html_log_path = os.path.abspath(html_log_file)
            
            if finish_reason == 'content_filter':
                print(f"⛔ Content refused by {self.model}")
                print(f"   📁 CSV log: {csv_log_path}")
                print(f"   📁 TXT log: {txt_log_path}")
                print(f"   📁 HTML log: {html_log_path}")
            else:
                print(f"✂️ Response truncated by {self.model}")
                print(f"   📁 CSV log: {csv_log_path}")
                print(f"   📁 TXT log: {txt_log_path}")
                print(f"   📁 HTML log: {html_log_path}")
            
        except Exception as e:
            # Don't crash the translation just because logging failed
            print(f"Failed to log truncation failure: {e}")

    def _get_safe_preview(self, messages: List[Dict], max_length: int = 100) -> str:
        """Get a safe preview of the input messages for logging"""
        try:
            # Get the last user message
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if len(content) > max_length:
                        return content[:max_length] + "..."
                    return content
            return "No user content found"
        except:
            return "Error extracting preview"

    def _send_deepl(self, messages, temperature=None, max_tokens=None, response_name=None) -> UnifiedResponse:
        """
        Send messages to DeepL API for translation
        
        Args:
            messages: List of message dicts
            temperature: Not used by DeepL (included for signature compatibility)
            max_tokens: Not used by DeepL (included for signature compatibility)
            response_name: Name for saving response (for debugging/logging)
        
        Returns:
            UnifiedResponse object
        """
        
        if not DEEPL_AVAILABLE:
            raise UnifiedClientError("DeepL library not installed. Run: pip install deepl")
        
        try:
            # Get DeepL API key
            deepl_api_key = os.getenv('DEEPL_API_KEY') or self.api_key
            
            if not deepl_api_key or deepl_api_key == 'dummy':
                raise UnifiedClientError("DeepL API key not found. Set DEEPL_API_KEY environment variable or configure in settings.")
            
            # Initialize DeepL translator
            translator = deepl.Translator(deepl_api_key)
            
            # Extract ONLY user content to translate - ignore AI system prompts
            text_to_translate = ""
            source_lang = None
            target_lang = "EN-US"  # Default to US English
            
            # Extract only user messages, ignore system prompts completely
            for msg in messages:
                if msg['role'] == 'user':
                    text_to_translate = msg['content']
                    # Simple language detection from content patterns
                    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in text_to_translate[:100]):
                        source_lang = 'KO'  # Korean
                    elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in text_to_translate[:100]) or \
                         any(ord(char) >= 0x30A0 and ord(char) <= 0x30FF for char in text_to_translate[:100]):
                        source_lang = 'JA'  # Japanese
                    elif any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text_to_translate[:100]):
                        source_lang = 'ZH'  # Chinese
                    break  # Take only the first user message
            
            if not text_to_translate:
                raise UnifiedClientError("No text to translate found in messages")
            
            # Log the translation request
            logger.info(f"DeepL: Translating {len(text_to_translate)} characters")
            if source_lang:
                logger.info(f"DeepL: Source language: {source_lang}")
            
            # Perform translation
            start_time = time.time()
            
            # DeepL API call
            if source_lang:
                result = translator.translate_text(
                    text_to_translate, 
                    source_lang=source_lang,
                    target_lang=target_lang,
                    preserve_formatting=True,
                    tag_handling='html' if '<' in text_to_translate else None
                )
            else:
                result = translator.translate_text(
                    text_to_translate,
                    target_lang=target_lang,
                    preserve_formatting=True,
                    tag_handling='html' if '<' in text_to_translate else None
                )
            
            elapsed_time = time.time() - start_time
            
            # Get the translated text
            translated_text = result.text
            
            # Create UnifiedResponse object
            response = UnifiedResponse(
                content=translated_text,
                finish_reason='complete',
                usage={
                    'characters': len(text_to_translate),
                    'detected_source_lang': result.detected_source_lang if hasattr(result, 'detected_source_lang') else source_lang
                },
                raw_response={'result': result}
            )
            
            logger.info(f"DeepL: Translation completed in {elapsed_time:.2f}s")
            
            return response
            
        except Exception as e:
            error_msg = f"DeepL API error: {str(e)}"
            logger.error(f"ERROR: {error_msg}")
            raise UnifiedClientError(error_msg)

    def _send_google_translate(self, messages, temperature=None, max_tokens=None, response_name=None):
        """Send messages to Google Translate API with markdown/HTML structure fixes"""
        
        if not GOOGLE_TRANSLATE_AVAILABLE:
            raise UnifiedClientError(
                "Google Cloud Translate not installed. Run: pip install google-cloud-translate\n"
                "Also ensure you have Google Cloud credentials configured."
            )
        
        # Import HTML output fixer for Google Translate's structured HTML
        try:
            from translate_output_fix import fix_google_translate_html
        except ImportError:
            # Fallback: create inline HTML structure fix
            import re
            def fix_google_translate_html(html_content):
                """Simple fallback: fix HTML structure issues where everything is in one header tag"""
                if not html_content:
                    return html_content
                
                # Check if everything is wrapped in a single header tag
                single_header = re.match(r'^<(h[1-6])>(.*?)</\1>$', html_content.strip(), re.DOTALL)
                if single_header:
                    tag = single_header.group(1)
                    content = single_header.group(2).strip()
                    
                    # Simple pattern: "Number. Title Name was/were..." -> "Number. Title" + "Name was/were..."
                    chapter_match = re.match(r'^(\d+\.\s+[^A-Z]*[A-Z][^A-Z]*?)\s+([A-Z][a-z]+\s+(?:was|were|had|did|is|are)\s+.*)$', content, re.DOTALL)
                    if chapter_match:
                        title = chapter_match.group(1).strip()
                        body = chapter_match.group(2).strip()
                        # Create properly structured HTML
                        paragraphs = re.split(r'\n\s*\n', body)
                        formatted_paragraphs = [f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()]
                        return f'<{tag}>{title}</{tag}>\n\n' + '\n\n'.join(formatted_paragraphs)
                
                return html_content
        
        try:
            # Check for Google Cloud credentials with better error messages
            google_creds_path = None
            
            # Try multiple possible locations for credentials
            possible_paths = [
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                os.getenv('GOOGLE_CLOUD_CREDENTIALS'),
                self.config.get('google_cloud_credentials') if hasattr(self, 'config') else None,
                self.config.get('google_vision_credentials') if hasattr(self, 'config') else None,
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    google_creds_path = path
                    break
            
            if not google_creds_path:
                raise UnifiedClientError(
                    "Google Cloud credentials not found.\n\n"
                    "To use Google Translate, you need to:\n"
                    "1. Create a Google Cloud service account\n"
                    "2. Download the JSON credentials file\n"
                    "3. Set it up in Glossarion:\n"
                    "   - For GUI: Use the 'Set up Google Cloud Translate Credentials' button\n"
                    "   - For CLI: Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n\n"
                    "The same credentials work for both Google Translate and Cloud Vision (manga OCR)."
                )
            
            # Set the environment variable for the Google client library
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path
            logger.info(f"Using Google Cloud credentials: {os.path.basename(google_creds_path)}")
            
            # Initialize the client
            translate_client = google_translate.Client()
            
            # Extract ONLY user content to translate - ignore AI system prompts
            text_to_translate = ""
            source_lang = None
            target_lang = 'en'  # Default to English
            
            # Extract only user messages, ignore system prompts completely
            for msg in messages:
                if msg['role'] == 'user':
                    text_to_translate = msg['content']
                    # Simple language detection from content patterns
                    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in text_to_translate[:100]):
                        source_lang = 'ko'  # Korean
                    elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in text_to_translate[:100]) or \
                         any(ord(char) >= 0x30A0 and ord(char) <= 0x30FF for char in text_to_translate[:100]):
                        source_lang = 'ja'  # Japanese
                    elif any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text_to_translate[:100]):
                        source_lang = 'zh'  # Chinese
                    break  # Take only the first user message
            
            if not text_to_translate:
                # Return empty response instead of error
                return UnifiedResponse(
                    content="",
                    finish_reason='complete',
                    usage={'characters': 0},
                    raw_response={}
                )
            
            # Log the translation request
            logger.info(f"Google Translate: Translating {len(text_to_translate)} characters")
            if source_lang:
                logger.info(f"Google Translate: Source language: {source_lang}")
            
            # Perform translation
            start_time = time.time()
            
            # Google Translate API call - force text format for markdown content
            # Detect if this is markdown from html2text (starts with #)
            is_markdown = text_to_translate.strip().startswith('#')
            translate_format = 'text' if is_markdown else ('html' if '<' in text_to_translate else 'text')
            
            
            if source_lang:
                result = translate_client.translate(
                    text_to_translate,
                    source_language=source_lang,
                    target_language=target_lang,
                    format_=translate_format
                )
            else:
                # Auto-detect source language
                result = translate_client.translate(
                    text_to_translate,
                    target_language=target_lang,
                    format_=translate_format
                )
            
            elapsed_time = time.time() - start_time
            
            # Handle both single result and list of results
            if isinstance(result, list):
                result = result[0] if result else {}
            
            translated_text = result.get('translatedText', '')
            detected_lang = result.get('detectedSourceLanguage', source_lang)
            
            import re
            
            # Fix markdown structure issues in Google Translate text output
            original_text = translated_text
            
            if is_markdown and translate_format == 'text':
                # Google Translate in text mode removes line breaks from markdown
                # Need to restore proper markdown structure
                
                # Pattern: "#6. Title Content goes here" -> "#6. Title\n\nContent goes here"
                markdown_fix = re.match(r'^(#{1,6}[^\n]*?)([A-Z][^#]+)$', translated_text.strip(), re.DOTALL)
                if markdown_fix:
                    header_part = markdown_fix.group(1).strip()
                    content_part = markdown_fix.group(2).strip()
                    
                    # Try to split header from content intelligently
                    # Look for patterns like "6. Title Name was" -> "6. Title" + "Name was"
                    title_content_match = re.match(r'^(.*?)([A-Z][a-z]+\s+(?:was|were|had|did|is|are)\s+.*)$', content_part, re.DOTALL)
                    if title_content_match:
                        title_end = title_content_match.group(1).strip()
                        content_start = title_content_match.group(2).strip()
                        
                        # Restore paragraph structure in the content
                        paragraphs = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content_start)
                        formatted_content = '\n\n'.join(paragraphs)
                        
                        translated_text = f"{header_part} {title_end}\n\n{formatted_content}"
                    else:
                        # Fallback: try to split at reasonable word boundary
                        words = content_part.split()
                        if len(words) > 3:
                            for i in range(2, min(6, len(words)-2)):
                                if words[i][0].isupper():
                                    title_words = ' '.join(words[:i])
                                    content_words = ' '.join(words[i:])
                                    
                                    # Restore paragraph structure in the content
                                    paragraphs = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content_words)
                                    formatted_content = '\n\n'.join(paragraphs)
                                    
                                    translated_text = f"{header_part} {title_words}\n\n{formatted_content}"
                                    break
            
            if translate_format == 'html':
                # Apply HTML structure fixes for HTML mode
                translated_text = fix_google_translate_html(translated_text)
            
            
            # Create UnifiedResponse object
            response = UnifiedResponse(
                content=translated_text,
                finish_reason='complete',
                usage={
                    'characters': len(text_to_translate),
                    'detected_source_lang': detected_lang
                },
                raw_response=result
            )
            
            logger.info(f"Google Translate: Translation completed in {elapsed_time:.2f}s")
            
            return response
            
        except UnifiedClientError:
            # Re-raise our custom errors with helpful messages
            raise
        except Exception as e:
            # Provide more helpful error messages for common issues
            error_msg = str(e)
            
            if "403" in error_msg or "permission" in error_msg.lower():
                raise UnifiedClientError(
                    "Google Translate API permission denied.\n\n"
                    "Please ensure:\n"
                    "1. Cloud Translation API is enabled in your Google Cloud project\n"
                    "2. Your service account has the 'Cloud Translation API User' role\n"
                    "3. Billing is enabled for your project (required for Translation API)\n\n"
                    f"Original error: {error_msg}"
                )
            elif "billing" in error_msg.lower():
                raise UnifiedClientError(
                    "Google Cloud billing not enabled.\n\n"
                    "The Translation API requires billing to be enabled on your project.\n"
                    "Visit: https://console.cloud.google.com/billing\n\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise UnifiedClientError(f"Google Translate API error: {error_msg}")
