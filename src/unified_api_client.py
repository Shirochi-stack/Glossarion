# unified_api_client.py - REFACTORED with Enhanced Error Handling and Extended AI Model Support
"""
Key Design Principles:
- The client handles API communication and returns accurate status
- Retry logic is implemented in the translation layer (TranslateKRtoEN.py)
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
- OPENROUTER_REFERER: HTTP referer for OpenRouter (default: https://github.com/your-app)
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
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging
import re
import base64
from PIL import Image
import io
import time
import csv
from datetime import datetime
import traceback
import hashlib
import html
from multi_api_key_manager import APIKeyPool, APIKeyEntry, RateLimitCache
import threading
from collections import defaultdict
logger = logging.getLogger(__name__)

# IMPORTANT: This client respects GUI settings via environment variables:
# - SEND_INTERVAL_SECONDS: Delay between API calls (set by GUI)
# All API providers INCLUDING ElectronHub respect this setting for proper GUI integration

# Note: For Yi models through ElectronHub, use eh/yi-34b-chat-200k format
# For direct Yi API access, use yi-34b-chat-200k format

# Set up logging
logger = logging.getLogger(__name__)

# OpenAI SDK
try:
    import openai
    from openai import OpenAIError
except ImportError:
    openai = None
    class OpenAIError(Exception): pass

# Gemini SDK
from google import genai
from google.genai import types

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
    logger.warning("Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform")

from functools import lru_cache
from datetime import datetime, timedelta
import threading
            
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
    """
    Unified client with fixed thread-safe multi-key support
    
    Key improvements:
    1. Thread-local storage for API clients
    2. Proper key rotation per request
    3. Thread-safe rate limit handling
    4. Cleaner error handling and retry logic
    """
    
    # Class-level shared resources - properly initialized
    _rate_limit_cache = None
    _api_key_pool: Optional[APIKeyPool] = None
    _pool_lock = threading.Lock()
    
    # Multi-key configuration
    _multi_key_mode = False
    _multi_key_config = []  # Will be populated from config
    _force_rotation = True
    _rotation_frequency = 1
    
    # Request tracking
    _global_request_counter = 0
    _counter_lock = threading.Lock()
    
    # Thread-local storage for clients and key assignments
    _thread_local = threading.local()
    
    # Legacy tracking (for compatibility)
    _key_assignments = {}  # thread_id -> (key_index, key_identifier)
    _assignment_lock = threading.Lock()
    
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
    }
    
    # Model-specific constraints
    MODEL_CONSTRAINTS = {
        'temperature_fixed': ['o4-mini', 'o1-mini', 'o1-preview', 'o3-mini', 'o3', 'o3-pro', 'o4-mini'],
        'no_system_message': ['o1', 'o1-preview', 'o3', 'o3-pro'],
        'max_completion_tokens': ['o4', 'o1', 'o3'],
        'chinese_optimized': ['qwen', 'yi', 'glm', 'chatglm', 'baichuan', 'ernie', 'hunyuan'],
    }
    
    @classmethod
    def setup_multi_key_pool(cls, config: List[Dict], force_rotation: bool = True, 
                           rotation_frequency: int = 1):
        """Setup the shared multi-key pool"""
        with cls._pool_lock:
            if cls._api_key_pool is None:
                cls._api_key_pool = APIKeyPool()
            
            # Initialize rate limit cache if needed
            if cls._rate_limit_cache is None:
                cls._rate_limit_cache = RateLimitCache()
            
            cls._api_key_pool.load_from_list(config)
            
            # Update configuration
            cls._multi_key_mode = True
            cls._multi_key_config = config
            cls._force_rotation = force_rotation
            cls._rotation_frequency = rotation_frequency
            
            print(f"[DEBUG] Multi-key pool initialized with {len(config)} keys")
            print(f"[DEBUG] Rotation: {'Forced' if force_rotation else 'On-Error'}, "
                  f"Frequency: every {rotation_frequency} request(s)")
    
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
    
    def __init__(self, api_key: str, model: str, output_dir: str = None):
        """Initialize the unified client"""
        # Store original values
        self.original_api_key = api_key
        self.original_model = model
        
        # Instance variables
        self.output_dir = output_dir
        self._cancelled = False
        self._in_cleanup = False
        self.conversation_message_count = 0
        self.context = None
        self.current_session_context = None
        
        # Request tracking
        self._request_count = 0
        self._thread_request_count = 0
        
        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'errors': defaultdict(int),
            'response_times': []
        }
        
        # Timeout configuration
        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
        if retry_timeout_enabled:
            self.request_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))
        else:
            self.request_timeout = 36000  # 10 hours
        
        # Check if multi-key mode should be enabled
        use_multi_keys_env = os.getenv('USE_MULTI_API_KEYS', '0') == '1'
        print(f"[DEBUG] USE_MULTI_API_KEYS env var: {os.getenv('USE_MULTI_API_KEYS')}")
        print(f"[DEBUG] MULTI_API_KEYS env var exists: {os.getenv('MULTI_API_KEYS') is not None}")
        print(f"[DEBUG] use_multi_keys_env: {use_multi_keys_env}")
        print(f"[DEBUG] self._multi_key_mode: {self._multi_key_mode}")
        if use_multi_keys_env and not self._multi_key_mode:
            # Initialize from environment
            multi_keys_json = os.getenv('MULTI_API_KEYS', '[]')
            print(f"[DEBUG] multi_keys_json: {multi_keys_json[:100]}...")  # First 100 chars
            force_rotation = os.getenv('FORCE_KEY_ROTATION', '1') == '1'
            rotation_frequency = int(os.getenv('ROTATION_FREQUENCY', '1'))
            
            try:
                multi_keys = json.loads(multi_keys_json)
                if multi_keys:
                    self.setup_multi_key_pool(multi_keys, force_rotation, rotation_frequency)
            except Exception as e:
                logger.error(f"Failed to load multi-key config: {e}")
        
        # Initial setup if not in multi-key mode
        if not self._multi_key_mode:
            self.api_key = api_key
            self.model = model
            self.key_identifier = "Single Key"
            self._setup_client()
    
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
        
        return self._thread_local
    
    def _ensure_thread_client(self):
        """Ensure the current thread has a properly initialized client"""
        tls = self._get_thread_local_client()
        thread_name = threading.current_thread().name
        
        # Multi-key mode
        if self._multi_key_mode:
            # Check if we need to rotate
            should_rotate = False
            
            if not tls.initialized:
                should_rotate = True
                print(f"[Thread-{thread_name}] Initializing with multi-key mode")
            elif self._force_rotation:
                tls.request_count += 1
                if tls.request_count >= self._rotation_frequency:
                    should_rotate = True
                    tls.request_count = 0
                    print(f"[Thread-{thread_name}] Rotating key (reached {self._rotation_frequency} requests)")
            
            if should_rotate:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    # Get a key from the pool
                    key_info = self._api_key_pool.get_key_for_thread(
                        force_rotation=should_rotate,
                        rotation_frequency=self._rotation_frequency
                    )
                    
                    if key_info:
                        # Successfully got a key
                        key, key_index, key_id = key_info  # Changed from 2 to 3 values
                        
                        # Update thread-local state
                        tls.api_key = key.api_key
                        tls.model = key.model
                        tls.key_index = key_index
                        tls.key_identifier = key_id
                        tls.initialized = True
                        
                        # Copy to instance for compatibility
                        self.api_key = tls.api_key
                        self.model = tls.model
                        self.key_identifier = tls.key_identifier
                        self.current_key_index = key_index
                        
                        if len(self.api_key) > 12:
                            masked_key = self.api_key[:4] + "..." + self.api_key[-4:]
                        else:
                            # For short keys, show less characters
                            masked_key = self.api_key[:3] + "..." + self.api_key[-2:] if len(self.api_key) > 5 else "***"

                        print(f"[Thread-{thread_name}] 🔑 Using {self.key_identifier} - {masked_key}")
                        
                        # Setup client
                        self._setup_client()
                        return  # Success!
                    
                    # No key available - check why
                    if not self._api_key_pool or not self._api_key_pool.keys:
                        raise UnifiedClientError("No API keys configured", error_type="no_keys")
                    
                    # All keys must be cooling down
                    print(f"[Thread-{thread_name}] No available keys, all cooling down")
                    
                    # Get shortest cooldown time
                    cooldown_time = self._get_shortest_cooldown_time()
                    
                    if retry_count < max_retries - 1:
                        print(f"[Thread-{thread_name}] Waiting {cooldown_time}s for key to become available...")
                        
                        # Wait with cancellation check
                        for i in range(cooldown_time):
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled", error_type="cancelled")
                            time.sleep(1)
                            if i % 10 == 0 and i > 0:
                                print(f"[Thread-{thread_name}] Still waiting... {cooldown_time - i}s remaining")
                        
                        retry_count += 1
                        continue
                    else:
                        # Final attempt - try to get ANY key, even if on cooldown
                        print(f"[Thread-{thread_name}] Final attempt - trying to get any key")
                        
                        # Find key with shortest remaining cooldown
                        best_key_index = None
                        min_cooldown = float('inf')
                        
                        for i, key in enumerate(self._api_key_pool.keys):
                            if key.enabled:  # At least check if enabled
                                key_id = f"Key#{i+1} ({key.model})"
                                remaining = self._rate_limit_cache.get_remaining_cooldown(key_id)
                                if remaining < min_cooldown:
                                    min_cooldown = remaining
                                    best_key_index = i
                        
                        if best_key_index is not None:
                            key = self._api_key_pool.keys[best_key_index]
                            print(f"[Thread-{thread_name}] Using key on cooldown (remaining: {min_cooldown:.1f}s)")
                            
                            # Force assign this key
                            tls.api_key = key.api_key
                            tls.model = key.model
                            tls.key_index = best_key_index
                            tls.key_identifier = f"Key#{best_key_index+1} ({key.model})"
                            tls.initialized = True
                            
                            self.api_key = tls.api_key
                            self.model = tls.model
                            self.key_identifier = tls.key_identifier
                            self.current_key_index = best_key_index
                            
                            if len(self.api_key) > 12:
                                masked_key = self.api_key[:4] + "..." + self.api_key[-4:]
                            else:
                                masked_key = self.api_key[:3] + "..." + self.api_key[-2:] if len(self.api_key) > 5 else "***"
                            
                            print(f"[Thread-{thread_name}] 🔑 Forced using {self.key_identifier} - {masked_key}")
                            
                            self._setup_client()
                            return
                        
                        # Really no keys available at all
                        raise UnifiedClientError("No available API keys for thread", error_type="no_keys")
        
        # Single key mode
        elif not tls.initialized:
            tls.api_key = self.original_api_key
            tls.model = self.original_model
            tls.key_identifier = "Single Key"
            tls.initialized = True
            
            self.api_key = tls.api_key
            self.model = tls.model
            self.key_identifier = tls.key_identifier
            
            #print(f"🔑 Single-key mode: Using {self.model}")
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
        max_retries = 3
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
                    
                    masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else self.api_key
                    print(f"[THREAD-{thread_name}] 🔑 Assigned {self.key_identifier} - {masked_key}")
                    
                    # Setup client for this key
                    self._setup_client()
                    self._apply_custom_endpoint_if_needed()
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
            
            retry_count += 1
        
        # If we've exhausted all retries, raise error
        raise UnifiedClientError(f"No available API keys for thread after {max_retries} retries", error_type="no_keys")

    def _get_next_available_key_for_thread(self) -> Optional[Tuple]:
        """Get next available key for thread assignment (thread-safe)"""
        if not self._api_key_pool:
            return None
        
        # Try each key starting from current pool index
        start_index = self._api_key_pool.current_index
        attempts = 0
        
        while attempts < len(self._api_key_pool.keys):
            key = self._api_key_pool.keys[self._api_key_pool.current_index]
            key_index = self._api_key_pool.current_index
            
            # Always advance for next thread
            self._api_key_pool.current_index = (self._api_key_pool.current_index + 1) % len(self._api_key_pool.keys)
            
            # Check if key is available
            key_id = f"Key#{key_index+1} ({key.model})"
            if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                return (key, key_index)
            
            attempts += 1
        
        # No available keys found, try to find any key not on cooldown
        for i, key in enumerate(self._api_key_pool.keys):
            key_id = f"Key#{i+1} ({key.model})"
            if not self._rate_limit_cache.is_rate_limited(key_id):
                return (key, i)
        
        # All keys are rate limited - wait for shortest cooldown
        wait_time = self._get_shortest_cooldown_time()
        thread_name = threading.current_thread().name
        
        print(f"[Thread-{thread_name}] All keys on cooldown. Waiting {wait_time}s...")
        
        # Wait with cancellation check
        for i in range(wait_time):
            if hasattr(self, '_cancelled') and self._cancelled:
                print(f"[Thread-{thread_name}] Wait cancelled by user")
                return None
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"[Thread-{thread_name}] Still waiting... {wait_time - i}s remaining")
        
        # Clear expired entries from cache
        self._rate_limit_cache.clear_expired()
        
        # Try again to find an available key
        for i, key in enumerate(self._api_key_pool.keys):
            key_id = f"Key#{i+1} ({key.model})"
            if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                return (key, i)
        
        # Still no keys? Return the first enabled one
        for i, key in enumerate(self._api_key_pool.keys):
            if key.enabled:
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

    def _ensure_key_selection(self):
        """Ensure we have a key selected for this thread"""
        if not self.use_multi_keys:
            return
        
        thread_name = threading.current_thread().name
        
        # Assign or rotate key for this thread
        self._assign_thread_key()
        
        # Clear any expired rate limits
        if self._rate_limit_cache:
            self._rate_limit_cache.clear_expired()

    def _handle_rate_limit_for_thread(self):
        """Handle rate limit by marking current thread's key and getting a new one"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        # Mark current key as rate limited
        if self.current_key_index is not None:
            key = self._api_key_pool.keys[self.current_key_index]
            cooldown = getattr(key, 'cooldown', 60)
            
            print(f"[THREAD-{thread_name}] 🕐 Marking {self.key_identifier} for cooldown ({cooldown}s)")
            self._rate_limit_cache.add_rate_limit(self.key_identifier, cooldown)
            self._mark_key_error(429)
        
        # Remove current assignment
        with self._assignment_lock:
            if thread_id in self._key_assignments:
                del self._key_assignments[thread_id]
        
        # Force reassignment
        self._thread_local.request_count = 0  # Reset rotation counter
        self._assign_thread_key()

    def _apply_custom_endpoint_if_needed(self):
        """Apply custom endpoint configuration if needed"""
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
        
        if custom_base_url and use_custom_endpoint and self.client_type == 'openai':
            if not custom_base_url.startswith(('http://', 'https://')):
                custom_base_url = 'https://' + custom_base_url
            
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=custom_base_url
            )
            print(f"[DEBUG] Applied custom OpenAI endpoint: {custom_base_url}")
    

    def __init__(self, api_key: str, model: str, output_dir: str = None):
        self._current_output_file = None
        self.original_api_key = api_key  # Store original
        self.original_model = model      # Store original
        self.api_key = api_key
        self.model = model
        self.conversation_message_count = 0
        self.pattern_counts = {}
        self.last_pattern = None
        self.context = None
        self.current_session_context = None
        self._cancelled = False
        self.output_dir = output_dir
        self._actual_output_filename = None
        self._in_cleanup = False
        self.openai_client = None
        self.gemini_client = None
        self.mistral_client = None
        self.cohere_client = None
        
        # Multi-key support
        self.use_multi_keys = False
        self.current_key_index = None
        self.key_identifier = "Single Key"
        self.request_count = 0  # Instance request counter
        
        print(f"[DEBUG] Initializing UnifiedClient with model: {model}")
        
        # Get timeout configuration from GUI (moved up)
        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
        if retry_timeout_enabled:
            self.request_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))
            logger.info(f"Using GUI-configured timeout: {self.request_timeout}s")
        else:
            self.request_timeout = 36000  # 10 hour default
            logger.info(f"Using default timeout: {self.request_timeout}s")
        
        # Stats tracking (moved up)
        self.stats = {
            'total_requests': 0,
            'empty_results': 0,
            'errors': {},
            'response_times': [],
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        # Store Google Cloud credentials path if available
        self.google_creds_path = None
        
        # Check if multi-key mode is enabled
        use_multi_keys_env = os.getenv('USE_MULTI_API_KEYS', '0') == '1'
        print(f"[DEBUG] USE_MULTI_API_KEYS env var: {os.getenv('USE_MULTI_API_KEYS')}")
        print(f"[DEBUG] MULTI_API_KEYS env var exists: {os.getenv('MULTI_API_KEYS') is not None}")
        print(f"[DEBUG] use_multi_keys_env: {use_multi_keys_env}")
        print(f"[DEBUG] self._multi_key_mode: {self._multi_key_mode}")
        if use_multi_keys_env and not self._multi_key_mode:
            # Initialize from environment if not already done
            multi_keys_json = os.getenv('MULTI_API_KEYS', '[]')
            print(f"[DEBUG] multi_keys_json: {multi_keys_json[:100]}...")
            force_rotation = os.getenv('FORCE_KEY_ROTATION', '1') == '1'
            rotation_frequency = int(os.getenv('ROTATION_FREQUENCY', '1'))
            
            try:
                multi_keys = json.loads(multi_keys_json)
                if multi_keys:
                    self.setup_multi_key_pool(multi_keys, force_rotation, rotation_frequency)
                    self.use_multi_keys = True
                    print(f"[DEBUG] Multi-key mode enabled with {len(multi_keys)} keys")
                    # Don't select a key yet - wait until first request
                    # But DON'T return early - we need to finish initialization
            except Exception as e:
                print(f"[ERROR] Failed to load multi-key config: {e}")
                self.use_multi_keys = False
        
        # Check for Vertex AI Model Garden models (contain @ symbol)
        if '@' in self.model or self.model.startswith('vertex/'):
            # For Vertex AI, we need Google Cloud credentials, not API key
            self.client_type = 'vertex_model_garden'
            
            # Try to find Google Cloud credentials
            # 1. Check environment variable
            self.google_creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            
            # 2. Check if passed as api_key (for compatibility)
            if not self.google_creds_path and api_key and os.path.exists(api_key):
                self.google_creds_path = api_key
                logger.info("Using API key parameter as Google Cloud credentials path")
            
            # 3. Will check GUI config later during send if needed
            
            if self.google_creds_path:
                logger.info(f"Vertex AI Model Garden: Using credentials from {self.google_creds_path}")
            else:
                logger.warning("Vertex AI Model Garden: Google Cloud credentials not yet configured")
        else:
            # Only set up client if not in multi-key mode
            # Multi-key mode will set up the client when a key is selected
            if not self.use_multi_keys:
                # Determine client type from model name
                self._setup_client()
                print(f"[DEBUG] After setup - client_type: {self.client_type}, openai_client: {self.openai_client}")
                
                # FORCE OPENAI CLIENT IF CUSTOM BASE URL IS SET AND ENABLED
                use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
                custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')

                # Only force OpenAI client if:
                # 1. Custom endpoint is enabled via toggle
                # 2. We have a custom URL
                # 3. No client was set up (or it's already openai)
                if custom_base_url and use_custom_endpoint and self.openai_client is None:
                    if self.client_type is None or self.client_type == 'openai':
                        print(f"[DEBUG] Custom base URL detected and enabled, using OpenAI client for model: {self.model}")
                        self.client_type = 'openai'
                        
                        if openai is None:
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
                    else:
                        print(f"[DEBUG] Custom base URL set but model {self.model} uses {self.client_type}, not overriding")
                elif custom_base_url and not use_custom_endpoint:
                    print(f"[DEBUG] Custom base URL detected but disabled via toggle, using standard client")
    
    def _get_next_available_key(self) -> Optional[Tuple]:
        """Get the next available key from the pool"""
        if self._api_key_pool:
            return self._api_key_pool.get_next_available_key()
        return None

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
        else:
            print(f"[ERROR] Failed to rotate to next key")
    
    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available key and reinitialize client"""
        if not self.use_multi_keys or not self._api_key_pool:
            return False
        
        old_key_identifier = self.key_identifier
        
        key_info = self._get_next_available_key()
        if key_info:
            # Update key and model
            self.api_key = key_info[0].api_key
            self.model = key_info[0].model
            self.current_key_index = key_info[1]
            
            # Update key identifier
            self.key_identifier = f"Key#{key_info[1]+1} ({self.model})"
            masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else self.api_key
            
            print(f"[DEBUG] 🔄 Rotating from {old_key_identifier} to {self.key_identifier} - {masked_key}")
            
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
                print(f"[DEBUG] Rotated key: Re-created OpenAI client with custom base URL")
            
            return True
        
        print(f"[WARNING] No available keys to rotate to")
        return False
    
    def _mark_key_success(self):
        """Mark current key as successful"""
        if self.use_multi_keys and self.current_key_index is not None and self._api_key_pool:
            self._api_key_pool.mark_key_success(self.current_key_index)
    
    def _mark_key_error(self, error_code: int = None):
        """Mark current key as having an error and apply cooldown if rate limited"""
        if self.use_multi_keys and self.current_key_index is not None and self._api_key_pool:
            self._api_key_pool.mark_key_error(self.current_key_index, error_code)
    
    def get_stats(self):
        """Get enhanced statistics including per-thread information"""
        stats = self.stats.copy()
        
        if self._multi_key_mode and self._api_key_pool:
            # Collect key pool statistics
            key_stats = []
            for i, key in enumerate(self._api_key_pool.keys):
                key_info = {
                    'index': i + 1,
                    'model': key.model,
                    'available': key.is_available(),
                    'cooling_down': key.is_cooling_down,
                    'success_count': key.success_count,
                    'error_count': key.error_count,
                    'last_used': key.last_used_time
                }
                key_stats.append(key_info)
            
            # Thread assignment info
            thread_info = {}
            for thread in threading.enumerate():
                if hasattr(thread, 'ident'):
                    # Check if this thread has an assignment
                    with self._api_key_pool.lock:
                        if thread.ident in self._api_key_pool._thread_assignments:
                            key_index, _ = self._api_key_pool._thread_assignments[thread.ident]
                            key = self._api_key_pool.keys[key_index]
                            thread_info[thread.name] = f"Key#{key_index+1} ({key.model})"
            
            stats.update({
                'multi_key_enabled': True,
                'total_keys': len(self._api_key_pool.keys),
                'available_keys': sum(1 for k in self._api_key_pool.keys if k.is_available()),
                'key_stats': key_stats,
                'thread_assignments': thread_info,
                'rotation_config': {
                    'force_rotation': self._force_rotation,
                    'rotation_frequency': self._rotation_frequency
                }
            })
        else:
            stats['multi_key_enabled'] = False
        
        return stats
    
    def cleanup(self):
        """Cleanup thread resources"""
        if self._multi_key_mode and self._api_key_pool:
            self._api_key_pool.release_thread_assignment()

            
    def _setup_client(self):
        """Setup the appropriate client based on model type"""
        model_lower = self.model.lower()
        tls = self._get_thread_local_client()
        print(f"[DEBUG] _setup_client called with model: {self.model}")
        
        # Check model prefixes FIRST to determine provider
        self.client_type = None
        for prefix, provider in self.MODEL_PROVIDERS.items():
            if model_lower.startswith(prefix):
                self.client_type = provider
                print(f"[DEBUG] Matched prefix '{prefix}' -> provider '{provider}'")
                break
        
        # Check if we're using a custom OpenAI base URL
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', ''))
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        
        # Only apply custom endpoint logic for OpenAI models or unmatched models
        if custom_base_url and custom_base_url != 'https://api.openai.com/v1' and use_custom_endpoint:
            if not self.client_type:
                # No prefix matched - assume it's a custom model that should use OpenAI endpoint
                self.client_type = 'openai'
                logger.info(f"Using OpenAI client for custom endpoint with unmatched model: {self.model}")
            elif self.client_type == 'openai':
                logger.info(f"Using custom OpenAI endpoint for OpenAI model: {self.model}")
            else:
                logger.info(f"Model {self.model} matched to {self.client_type}, not using custom OpenAI endpoint")
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
        
        # Initialize provider-specific settings
        if self.client_type == 'openai':
            print(f"[DEBUG] Setting up OpenAI client")
            if openai is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
            # Check if custom endpoints are enabled
            use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
            
            # Check for custom base URL
            if use_custom_endpoint:
                base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'))
                
                # Validate URL has protocol if it's not the default
                if base_url != 'https://api.openai.com/v1' and not base_url.startswith(('http://', 'https://')):
                    print(f"[WARNING] Custom base URL missing protocol, adding https://")
                    base_url = 'https://' + base_url
            else:
                # Force default endpoint when toggle is off
                base_url = 'https://api.openai.com/v1'
                print(f"[DEBUG] Custom endpoints disabled, using default OpenAI endpoint")
            
            print(f"[DEBUG] Using base URL: {base_url}")
            
            # Create OpenAI client with custom base URL support
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
            print(f"[DEBUG] OpenAI client created: {self.openai_client}")
            
        elif self.client_type == 'gemini':
            if genai is None:
                raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
            self.gemini_client = genai.Client(api_key=self.api_key)
            
        elif self.client_type == 'electronhub':
            # ElectronHub uses OpenAI SDK if available
            if openai is not None:
                logger.info("ElectronHub will use OpenAI SDK for API calls")
            else:
                logger.info("ElectronHub will use HTTP API for API calls")
            
        elif self.client_type == 'mistral':
            if MistralClient is None:
                # Fall back to HTTP API if SDK not installed
                logger.info("Mistral SDK not installed, will use HTTP API")
            else:
                self.mistral_client = MistralClient(api_key=self.api_key)
                
        elif self.client_type == 'cohere':
            if cohere is not None:
                self.cohere_client = cohere.Client(self.api_key)
            else:
                logger.info("Cohere SDK not installed, will use HTTP API")
        
        # Log retry feature support
        logger.info(f"✅ Initialized {self.client_type} client for model: {self.model}")
        logger.debug("✅ GUI retry features supported: truncation detection, timeout handling, duplicate detection")
    
    def reset_conversation_for_new_context(self, new_context):
        """Reset conversation state when context changes"""
        self.current_session_context = new_context
        self.conversation_message_count = 0
        self.pattern_counts.clear()
        self.last_pattern = None
        logger.info(f"Reset conversation state for new context: {new_context}")
    
    def _apply_pure_reinforcement(self, messages):
        """Apply PURE frequency-based reinforcement pattern"""
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
                self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1
                
                # Apply reinforcement if pattern occurs frequently
                if self.pattern_counts[pattern_key] >= 3:
                    logger.info(f"Applying reinforcement for pattern: {pattern_key}")
                    # Add reinforcement to system message
                    for msg in messages:
                        if msg.get('role') == 'system':
                            msg['content'] += "\n\n[PATTERN REINFORCEMENT ACTIVE]"
                            break
        
        return messages
    
    def _validate_request(self, messages, max_tokens):
        """Validate request parameters before sending"""
        if not messages:
            return False, "Empty messages list"
        
        # Check message content isn't empty
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        if total_chars == 0:
            return False, "Empty request content"
        
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
            error_key = f"{self.client_type}_{context}_{error_type}"
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
            'model': self.model,
            'client_type': self.client_type,
            'response': str(response) if response else None,
            'traceback': traceback.format_exc()
        }
        
        filename = f"{failed_dir}/failed_{context}_{self.client_type}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(failure_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved failed request to: {filename}")
    
    def _handle_empty_result(self, messages, context, error_info):
        """Handle empty results with context-aware fallbacks"""
        print(f"Handling empty result for context: {context}, error: {error_info}")
        
        if context == 'glossary':
            # Return empty but valid JSON
            return "[]"
        elif context == 'translation':
            # Extract the original text and return it with a marker
            original_text = self._extract_user_content(messages)
            return f"[TRANSLATION FAILED - ORIGINAL TEXT PRESERVED]\n{original_text}"
        elif context == 'image_translation':
            return "[IMAGE TRANSLATION FAILED]"
        else:
            # Generic fallback
            return "[AI RESPONSE UNAVAILABLE]"
    
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
            chapter_match = re.search(r'Chapter (\d+)', str(messages))
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
    
    def _save_payload(self, messages, filename):
        
         # Thread isolation
        thread_name = threading.current_thread().name
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{threading.current_thread().ident}")
        os.makedirs(thread_dir, exist_ok=True)
        """Save request payload for debugging"""
        filepath = os.path.join(thread_dir, filename)
        try:
            # Include debug info about system prompt
            debug_info = {
                'system_prompt_present': any(msg.get('role') == 'system' for msg in messages),
                'system_prompt_length': 0
            }
            
            for msg in messages:
                if msg.get('role') == 'system':
                    debug_info['system_prompt_length'] = len(msg.get('content', ''))
                    break
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': self.model,
                    'client_type': self.client_type,
                    'messages': messages,
                    'timestamp': datetime.now().isoformat(),
                    'debug': debug_info
                }, f, indent=2, ensure_ascii=False)
                
            if self.client_type == 'electronhub' and debug_info['system_prompt_present']:
                logger.info(f"ElectronHub payload saved with system prompt ({debug_info['system_prompt_length']} chars)")
        except Exception as e:
            print(f"Failed to save payload: {e}")
    

    def _save_response(self, content: str, filename: str):
        """Save API response to file with proper path handling
        
        IMPORTANT: Only save JSON payloads, not HTML responses
        HTML responses are saved in the book output folder, not Payloads
        """
        if not content or not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
        
        # ONLY save JSON files to Payloads folder
        # Skip HTML files - they belong in the book output folder
        if not filename.endswith('.json'):
            logger.debug(f"Skipping HTML response save to Payloads: {filename}")
            return
        
        # ADD: Thread isolation
        thread_name = threading.current_thread().name
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{threading.current_thread().ident}")
        os.makedirs(thread_dir, exist_ok=True)
            
        try:
            # REST OF YOUR CODE STAYS EXACTLY THE SAME
            # Use forward slashes for consistency
            safe_filename = filename.replace("\\", "/")
            if "/" in safe_filename:
                safe_filename = safe_filename.split("/")[-1]
            
            # CHANGE: Use thread directory instead of "Payloads"
            filepath = os.path.join(thread_dir, safe_filename)  # CHANGED from: os.path.join("Payloads", safe_filename)
            
            # For JSON responses, ensure proper formatting
            try:
                # Try to parse and pretty-print JSON
                json_content = json.loads(content)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # If not valid JSON, save as-is
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            logger.debug(f"Saved JSON payload to: {filepath}")
            
        except Exception as e:
            print(f"Failed to save response to {filename}: {e}")
            # Don't raise - this is not critical functionality

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
        self._in_cleanup = False  # Set cleanup flag
        print("🛑 Operation cancelled (timeout or user stop)")
        print("🛑 API operation cancelled")

    def reset_cleanup_state(self):
            """Reset cleanup state for new operations"""
            self._in_cleanup = False
            self._cancelled = False

    def _send_vertex_model_garden(self, messages, temperature=0.7, max_tokens=None, stop_sequences=None, response_name=None):
        """Send request to Vertex AI Model Garden models (including Claude)"""
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
                
                print(f"Using Vertex AI region: {region}")
                
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
                
                print(f"Sending request to {model_name} in region {region}")
                
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
                
                # Format messages for Vertex AI Gemini
                formatted_prompt = self._format_gemini_prompt_simple(messages)
                
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
                    print(f"🔒 Vertex AI Gemini Safety Status: DISABLED - All categories set to BLOCK_NONE")
                else:
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
                if response_name:  # Only save if response_name is provided
                    self._save_gemini_safety_config(config_data, response_name)
            
                # Retry logic with token reduction
                BOOST_FACTOR = 1
                attempts = 4
                attempt = 0
                result_text = ""
                current_tokens = (max_tokens or 8192) * BOOST_FACTOR
                
                while attempt < attempts and not result_text:
                    try:
                        # Update max_output_tokens for this attempt
                        generation_config_dict["max_output_tokens"] = current_tokens
                        generation_config = GenerationConfig(**generation_config_dict)
                        
                        print(f"   📊 Temperature: {temperature}, Max tokens: {current_tokens}")
                        
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
                        attempt += 1
                        if attempt < attempts:
                            print(f"❌ Gemini attempt {attempt} failed, no automatic retry")
                            break  # Exit the retry loop
                    
                # Check stop flag after response
                if is_stop_requested():
                    logger.info("Stop requested after Vertex AI Gemini response")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                if not result_text:
                    raise UnifiedClientError("All Vertex AI Gemini attempts failed to produce content")
                
                return UnifiedResponse(
                    content=result_text,
                    finish_reason='stop',
                    raw_response=response if 'response' in locals() else None
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
                
    def send(self, messages, temperature=None, max_tokens=None, 
             max_completion_tokens=None, context=None) -> Tuple[str, Optional[str]]:
        """Thread-safe send with proper key management for batch translation"""
        thread_name = threading.current_thread().name
        
        # Ensure thread has a client
        self._ensure_thread_client()
        
        logger.info(f"[{thread_name}] Using {self.key_identifier} for {context or 'unknown'}")
        
        max_retries = 3
        retry_count = 0
        last_error = None
        
        # Track which keys we've already tried to avoid infinite loops
        attempted_keys = set()
        
        while retry_count < max_retries:
            try:
                # Track current key
                attempted_keys.add(self.key_identifier)
                
                # Call the actual implementation
                result = self._send_internal(messages, temperature, max_tokens, 
                                           max_completion_tokens, context)
                
                # Mark success
                if self._multi_key_mode:
                    tls = self._get_thread_local_client()
                    if tls.key_index is not None:
                        self._api_key_pool.mark_key_success(tls.key_index)
                
                logger.info(f"[{thread_name}] ✓ Request completed with {self.key_identifier}")
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.error(f"[{thread_name}] ✗ {self.key_identifier} error: {error_str[:100]}")
                
                # Check for rate limit
                if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                    if self._multi_key_mode:
                        print(f"[Thread-{thread_name}] Rate limit hit on {self.key_identifier}")
                        
                        # Handle rate limit for this thread
                        self._handle_rate_limit_for_thread()
                        
                        # Check if we have any available keys
                        available_count = self._count_available_keys()
                        if available_count == 0:
                            logger.error(f"[{thread_name}] All API keys are cooling down")
                            
                            # If we still have retries left, wait for the shortest cooldown
                            if retry_count < max_retries - 1:
                                cooldown_time = self._get_shortest_cooldown_time()
                                print(f"⏳ [Thread-{thread_name}] All keys cooling down, waiting {cooldown_time}s...")
                                
                                # Wait with cancellation check
                                for i in range(cooldown_time):
                                    if self._cancelled:
                                        raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                    time.sleep(1)
                                    if i % 10 == 0 and i > 0:
                                        print(f"⏳ [Thread-{thread_name}] Still waiting... {cooldown_time - i}s remaining")
                                
                                # Force re-initialization after cooldown
                                tls = self._get_thread_local_client()
                                tls.initialized = False
                                self._ensure_thread_client()
                                
                                retry_count += 1
                                continue
                            else:
                                # No more retries, raise the error
                                raise UnifiedClientError("All API keys are cooling down", error_type="rate_limit")
                        
                        # Check if we've tried too many keys
                        if len(attempted_keys) >= len(self._api_key_pool.keys):
                            logger.error(f"[{thread_name}] Attempted all {len(self._api_key_pool.keys)} keys")
                            raise UnifiedClientError("All API keys rate limited", error_type="rate_limit")
                        
                        retry_count += 1
                        logger.info(f"[{thread_name}] Retrying with new key, attempt {retry_count}/{max_retries}")
                        continue
                    else:
                        # Single key mode - wait and retry
                        if retry_count < max_retries - 1:
                            wait_time = min(30 * (retry_count + 1), 120)
                            logger.info(f"[{thread_name}] Rate limit, waiting {wait_time}s")
                            
                            for i in range(wait_time):
                                if self._cancelled:
                                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                time.sleep(1)
                                
                            retry_count += 1
                            continue
                        else:
                            raise UnifiedClientError("Rate limit exceeded", error_type="rate_limit")
                
                # Check for cancellation
                elif isinstance(e, UnifiedClientError) and e.error_type in ["cancelled", "timeout"]:
                    raise
                
                # Other errors
                elif retry_count < max_retries - 1:
                    if self._multi_key_mode:
                        tls = self._get_thread_local_client()
                        if tls.key_index is not None:
                            self._api_key_pool.mark_key_error(tls.key_index)
                        
                        if not self._force_rotation:
                            # Error-based rotation - try a different key
                            logger.info(f"[{thread_name}] Error occurred, rotating to new key...")
                            
                            # Force reassignment
                            tls.initialized = False
                            tls.request_count = 0
                            self._ensure_thread_client()
                            
                            retry_count += 1
                            logger.info(f"[{thread_name}] Rotated to {self.key_identifier} after error")
                            continue
                    
                    # Retry with same key (or if rotation disabled)
                    retry_count += 1
                    time.sleep(2)
                    continue
                
                # Can't retry
                raise
        
        # Exhausted retries
        if last_error:
            raise last_error
        else:
            raise Exception(f"Failed after {max_retries} attempts")
        
    def _rotate_to_next_available_key(self, skip_current: bool = False) -> bool:
        """
        Rotate to the next available key that's not rate limited
        
        Args:
            skip_current: If True, skip the current key even if it becomes available
        """
        if not self.use_multi_keys or not self._api_key_pool:
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


    def _force_rotate_to_untried_key(self, attempted_keys: set) -> bool:
        """
        Force rotation to any key that hasn't been tried yet, ignoring cooldown
        
        Args:
            attempted_keys: Set of key identifiers that have already been attempted
        """
        if not self.use_multi_keys or not self._api_key_pool:
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


    def _apply_key_change(self, key_info: tuple, old_key_identifier: str):
        """Apply the key change and reinitialize clients"""
        self.api_key = key_info[0].api_key
        self.model = key_info[0].model
        self.current_key_index = key_info[1]
        self.key_identifier = f"Key#{key_info[1]+1} ({key_info[0].model})"
        
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else self.api_key
        print(f"[DEBUG] 🔄 Switched from {old_key_identifier} to {self.key_identifier} - {masked_key}")
        
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


    def _count_available_keys(self) -> int:
        """Count how many keys are currently available (not on cooldown)"""
        if not self.use_multi_keys or not self._api_key_pool:
            return 0
        
        available = 0
        for i, key in enumerate(self._api_key_pool.keys):
            key_id = f"Key#{i+1} ({key.model})"
            if not self._rate_limit_cache.is_rate_limited(key_id) and key.is_available():
                available += 1
        
        return available
    
    def _get_shortest_cooldown_time(self) -> int:
        """Get the shortest time until a key becomes available"""
        if not self.use_multi_keys or not self._api_key_pool:
            return 30  # Default wait time
        
        min_wait = float('inf')
        now = time.time()
        
        # Check each key's cooldown
        for i, key in enumerate(self._api_key_pool.keys):
            if key.enabled:  # Only check enabled keys
                key_id = f"Key#{i+1} ({key.model})"
                
                # Check rate limit cache first
                cache_cooldown = self._rate_limit_cache.get_remaining_cooldown(key_id)
                if cache_cooldown > 0:
                    min_wait = min(min_wait, cache_cooldown)
                
                # Also check key's own cooldown
                if key.is_cooling_down and key.last_error_time:
                    remaining = key.cooldown - (now - key.last_error_time)
                    if remaining > 0:
                        min_wait = min(min_wait, remaining)
        
        # Return the minimum wait time, capped at 60 seconds
        return min(int(min_wait) if min_wait != float('inf') else 30, 60)
    
    def _send_internal(self, messages, temperature=None, max_tokens=None, max_completion_tokens=None, context=None) -> Tuple[str, Optional[str]]:
        """
        Internal send implementation with integrated 500 error retry logic
        """
        start_time = time.time()
        
        # Reset cancelled flag
        self._cancelled = False
        
        # Reset counters when context changes
        if context != self.current_session_context:
            self.reset_conversation_for_new_context(context)
        
        self.context = context or 'translation'
        self.conversation_message_count += 1
        
        # Internal retry logic for 500 errors
        internal_retries = 3
        for attempt in range(internal_retries):
            try:
                # Validate request
                valid, error_msg = self._validate_request(messages, max_tokens)
                if not valid:
                    raise UnifiedClientError(f"Invalid request: {error_msg}", error_type="validation")
                
                os.makedirs("Payloads", exist_ok=True)
                
                # Apply reinforcement
                messages = self._apply_pure_reinforcement(messages)
                
                # Get file names - IMPORTANT for duplicate detection
                payload_name, response_name = self._get_file_names(messages, context=self.context)
                
                # Save payload for debugging
                self._save_payload(messages, payload_name)
                
                # Check for timeout toggle from GUI
                retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
                if retry_timeout_enabled:
                    timeout_seconds = int(os.getenv("CHUNK_TIMEOUT", "180"))
                    logger.info(f"Timeout monitoring enabled: {timeout_seconds}s limit")
                
                # Get response
                response = self._get_response(messages, temperature, max_tokens, max_completion_tokens, response_name)
                
                # Check for cancellation (from timeout or stop button)
                if self._cancelled:
                    logger.info("Operation cancelled (timeout or user stop)")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                # CRITICAL: Save response for duplicate detection
                # This must happen even for truncated/empty responses
                if response.content:
                    self._save_response(response.content, response_name)
                
                # Handle empty responses
                if response.is_error or not response.content or response.content.strip() in ["", "[]"]:
                    print(f"Empty or error response: {response.finish_reason}")
                    self._save_failed_request(messages, "Empty response", context, response.raw_response)
                    # ALWAYS log these failures too
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=response.content or "",
                        finish_reason=response.finish_reason or 'error',
                        context=context,
                        error_details=response.error_details
                    )
                    self._track_stats(context, False, "empty_response", time.time() - start_time)
                    
                    # Use fallback
                    fallback_content = self._handle_empty_result(messages, context, response.error_details or "empty")
                    return fallback_content, 'error'
                
                # Track success
                self._track_stats(context, True, None, time.time() - start_time)
                
                # Mark key as successful in multi-key mode
                self._mark_key_success()
                
                # Log important info for retry mechanisms
                if response.is_truncated:
                    print(f"Response was truncated: {response.finish_reason}")
                    print(f"⚠️ Response truncated (finish_reason: {response.finish_reason})")
                    
                    # ALWAYS log truncation failures
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=response.content,
                        finish_reason=response.finish_reason,
                        context=context,
                        error_details=response.error_details
                    )
                    # The calling code will check finish_reason=='length' for retry
                
                # Apply API delay after successful call (even if truncated)
                # SKIP DELAY DURING CLEANUP
                if not self._in_cleanup:
                    api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
                    if api_delay > 0:
                        print(f"⏳ Waiting {api_delay}s before next API call...")
                        time.sleep(api_delay)
                else:
                    print("⚡ Skipping API delay (cleanup mode)")
                
                # Return the response with accurate finish_reason
                # This is CRITICAL for retry mechanisms to work
                return response.content, response.finish_reason
                
            except UnifiedClientError as e:
                # Handle cancellation specially for timeout support
                if e.error_type == "cancelled" or "cancelled" in str(e):
                    self._in_cleanup = False  # Ensure cleanup flag is set
                    logger.info("Propagating cancellation to caller")
                    # Re-raise so send_with_interrupt can handle it
                    raise
                
                print(f"UnifiedClient error: {e}")
                
                # Check if it's a rate limit error and re-raise for retry logic
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic in outer send() method
                
                # Check for prohibited content - never retry these
                content_filter_indicators = [
                    "content_filter", "content was blocked", "response was blocked",
                    "safety filter", "content policy", "harmful content",
                    "blocked by safety", "harm_category"
                ]
                
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Content prohibited - not retrying: {error_str[:100]}")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors - retry these
                http_status = getattr(e, 'http_status', None)
                if http_status == 500 or "500" in error_str or "api_error" in error_str:
                    if attempt < internal_retries - 1:
                        wait_time = min(5 * (attempt + 1), 15)  # 5s, 10s, 15s backoff
                        print(f"🔄 Server error (500) - auto-retrying in {wait_time}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        for i in range(wait_time):
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(1)
                        continue
                    else:
                        print(f"❌ Server error (500) - exhausted {internal_retries} retries")
                
                # Save failed request and return fallback
                self._save_failed_request(messages, e, context)
                self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                fallback_content = self._handle_empty_result(messages, context, str(e))
                return fallback_content, 'error'
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                error_str = str(e).lower()
                
                # For unexpected errors, check if it's a timeout
                if "timed out" in error_str:
                    # Re-raise timeout errors so the retry logic can handle them
                    raise UnifiedClientError(f"Request timed out: {e}", error_type="timeout")
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic
                
                # Check for prohibited content in unexpected errors
                content_filter_indicators = [
                    "content_filter", "content was blocked", "response was blocked",
                    "safety filter", "content policy", "harmful content",
                    "blocked by safety", "harm_category"
                ]
                
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Content prohibited - not retrying")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors in unexpected exceptions
                if "500" in error_str or "internal server error" in error_str:
                    if attempt < internal_retries - 1:
                        wait_time = min(5 * (attempt + 1), 15)
                        print(f"🔄 Server error (500) - auto-retrying in {wait_time}s (attempt {attempt + 1}/{internal_retries})")
                        
                        for i in range(wait_time):
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(1)
                        continue
                
                # Check for other transient errors
                transient_errors = ["502", "503", "504", "connection reset", "connection aborted"]
                if any(err in error_str for err in transient_errors):
                    if attempt < internal_retries - 1:
                        wait_time = min(3 * (attempt + 1), 10)
                        print(f"🔄 Transient error - retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                
                # Save failed request and return fallback for other errors
                self._save_failed_request(messages, e, context)
                self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                fallback_content = self._handle_empty_result(messages, context, str(e))
                return fallback_content, 'error'
    
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
        # Map client types to their handler methods
        handlers = {
            'openai': self._send_openai,
            'gemini': self._send_gemini,
            'deepseek': self._send_deepseek,
            'anthropic': self._send_anthropic,
            'mistral': self._send_mistral,
            'cohere': self._send_cohere,
            'ai21': self._send_ai21,
            'together': self._send_together,
            'perplexity': self._send_perplexity,
            'replicate': self._send_replicate,
            'yi': self._send_yi,
            'qwen': self._send_qwen,
            'baichuan': self._send_baichuan,
            'zhipu': self._send_zhipu,
            'moonshot': self._send_moonshot,
            'groq': self._send_groq,
            'baidu': self._send_baidu,
            'tencent': self._send_tencent,
            'iflytek': self._send_iflytek,
            'bytedance': self._send_bytedance,
            'minimax': self._send_minimax,
            'sensenova': self._send_sensenova,
            'internlm': self._send_internlm,
            'tii': self._send_tii,
            'microsoft': self._send_microsoft,
            'azure': self._send_azure,
            'google': self._send_google_palm,
            'alephalpha': self._send_alephalpha,
            'databricks': self._send_databricks,
            'huggingface': self._send_huggingface,
            'salesforce': self._send_salesforce,
            'bigscience': self._send_together,  # Usually through Together/HF
            'meta': self._send_together,  # Meta models usually through Together
            'electronhub': self._send_electronhub,  # ElectronHub API aggregator
            'poe': self._send_poe,  # Poe platform
            'openrouter': self._send_openrouter,  # OpenRouter aggregator
            'fireworks': self._send_fireworks,  # Fireworks AI
            'xai': self._send_xai,  # xAI Grok models
            'vertex_model_garden': self._send_vertex_model_garden,
        }
        
        handler = handlers.get(self.client_type)
        if not handler:
            # Try fallback to Together AI for open models
            if self.client_type in ['bigscience', 'meta', 'databricks', 'huggingface', 'salesforce']:
                logger.info(f"Using Together AI for {self.client_type} model")
                return self._send_together(messages, temperature, max_tokens, response_name)
            raise UnifiedClientError(f"No handler for client type: {self.client_type}")
        
        # For OpenAI, pass the max_completion_tokens parameter
        if self.client_type == 'openai':
            return handler(messages, temperature, max_tokens, max_completion_tokens, response_name)
        elif self.client_type == 'vertex_model_garden':
            # Vertex AI doesn't use response_name parameter
            return handler(messages, temperature, max_tokens or max_completion_tokens, None, response_name)
        else:
            # Other providers don't use max_completion_tokens yet
            return handler(messages, temperature, max_tokens, response_name)

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
        
        # Pattern 1: Check for incomplete sentence endings
        # Most complete responses end with proper punctuation
        content_stripped = content.strip()
        if content_stripped:
            last_char = content_stripped[-1]
            # Define valid ending punctuation marks
            valid_endings = [
                '.', '!', '?', '"', "'", '»', '】', '）', ')', 
                '。', '！', '？', '"', ''', '】', '"', '''
            ]
            # Check if ends with incomplete sentence (no proper punctuation)
            if last_char not in valid_endings:
                # Additional check: is the last word incomplete?
                words = content_stripped.split()
                if words:
                    last_word = words[-1]
                    # Check for common incomplete patterns
                    if len(last_word) > 2 and last_word[-1].isalpha():
                        print(f"Possible silent truncation detected: incomplete sentence ending")
                        return True
        
        # Pattern 2: Check for significantly short responses
        if context == 'translation':
            # Estimate expected length based on input
            input_length = sum(len(msg.get('content', '')) for msg in messages if msg.get('role') == 'user')
            if input_length > 500 and len(content_stripped) < input_length * 0.3:
                print(f"Possible silent truncation: output ({len(content_stripped)} chars) much shorter than input ({input_length} chars)")
                return True
        
        # Pattern 3: Check for incomplete HTML/XML structures
        if '<' in content and '>' in content:
            # Count opening and closing tags
            opening_tags = content.count('<') - content.count('</')
            closing_tags = content.count('</')
            if opening_tags > closing_tags + 2:  # Allow small mismatch
                print(f"Possible silent truncation: unclosed HTML tags detected")
                return True
        
        # Pattern 4: Check for mature content indicators followed by abrupt ending
        mature_indicators = [
            'mature content', 'explicit', 'sexual', 'violence', 'adult',
            'inappropriate', 'sensitive', 'censored', 'restricted'
        ]
        content_lower = content_stripped.lower()
        for indicator in mature_indicators:
            if indicator in content_lower:
                # Check if response is suspiciously short after mentioning mature content
                indicator_pos = content_lower.rfind(indicator)
                remaining_content = content_stripped[indicator_pos + len(indicator):]
                if len(remaining_content) < 50:  # Very little content after indicator
                    print(f"Possible censorship truncation: content ends shortly after '{indicator}'")
                    return True
        
        # Pattern 5: Check for incomplete code blocks or quotes
        if '```' in content:
            code_block_count = content.count('```')
            if code_block_count % 2 != 0:  # Odd number means unclosed
                print(f"Possible silent truncation: unclosed code block")
                return True
        
        # Pattern 6: For glossary context, check for incomplete JSON
        if context == 'glossary' and content_stripped.startswith('['):
            if not content_stripped.endswith(']') and not content_stripped.endswith('],'):
                print(f"Possible silent truncation: incomplete JSON array")
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
 
    def _send_poe(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request using poe-api-wrapper"""
        try:
            from poe_api_wrapper import PoeApi
        except ImportError:
            raise UnifiedClientError(
                "poe-api-wrapper not installed. Run: pip install poe-api-wrapper"
            )
        
        # Parse cookies
        tokens = {}
        if '|' in self.api_key:
            for pair in self.api_key.split('|'):
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    tokens[k.strip()] = v.strip()
        elif ':' in self.api_key:
            k, v = self.api_key.split(':', 1)
            tokens[k.strip()] = v.strip()
        else:
            tokens['p-b'] = self.api_key.strip()
        
        # If no p-lat provided, add empty string (some versions of poe-api-wrapper need this)
        if 'p-lat' not in tokens:
            tokens['p-lat'] = ''
            logger.info("No p-lat cookie provided, using empty string")
        
        logger.info(f"Tokens being sent: p-b={len(tokens.get('p-b', ''))} chars, p-lat={len(tokens.get('p-lat', ''))} chars")
        
        try:
            # Create Poe client
            poe_client = PoeApi(tokens=tokens)
            
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
            
            return UnifiedResponse(
                content=final_text,
                finish_reason="stop",
                raw_response=chunk if 'chunk' in locals() else {"response": full_response}
            )
            
        except Exception as e:
            print(f"Poe API error details: {str(e)}")
            # Check for specific errors
            error_str = str(e).lower()
            if "rate limit" in error_str:
                raise UnifiedClientError(
                    "POE rate limit exceeded. Please wait before trying again.",
                    error_type="rate_limit"
                )
            elif "auth" in error_str or "unauthorized" in error_str:
                raise UnifiedClientError(
                    "POE authentication failed. Your cookies may be expired. "
                    "Please get fresh cookies from poe.com.",
                    error_type="auth_error"
                )
            raise UnifiedClientError(f"Poe API error: {e}")
            
    def _send_openrouter(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenRouter API with safety settings"""
        # Check if safety settings are disabled via GUI toggle
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # OpenRouter uses OpenAI-compatible format
        # Strip 'or/' or 'openrouter/' prefix
        model_name = self.model
        for prefix in ['or/', 'openrouter/']:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
                break
        
        # OpenRouter specific headers
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': os.getenv('OPENROUTER_REFERER', 'https://github.com/your-app'),
            'X-Title': os.getenv('OPENROUTER_APP_NAME', 'Glossarion Translation')
        }
        
        # Add safety header if disabled
        if disable_safety:
            headers['X-Safe-Mode'] = 'false'
            logger.info("🔓 Safety toggle enabled for OpenRouter")
            print("🔓 OpenRouter Safety: Disabled via X-Safe-Mode header")
        
        # Store original model and update for API call
        original_model = self.model
        self.model = model_name
        
        try:
            return self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url="https://openrouter.ai/api/v1",
                response_name=response_name,
                provider="openrouter",
                headers=headers
            )
        finally:
            self.model = original_model
    
    def _send_fireworks(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Fireworks AI API"""
        # Fireworks uses accounts/ prefix in model names
        model_name = self.model.replace('fireworks/', '', 1)
        if not model_name.startswith('accounts/'):
            model_name = f'accounts/fireworks/models/{model_name}'
        
        # Store original model
        original_model = self.model
        self.model = model_name
        
        try:
            return self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url=os.getenv("FIREWORKS_API_URL", "https://api.fireworks.ai/inference/v1"),
                response_name=response_name,
                provider="fireworks"
            )
        finally:
            self.model = original_model
    
    def _send_xai(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to xAI API (Grok models)"""
        # xAI uses OpenAI-compatible format
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=os.getenv("XAI_API_URL", "https://api.x.ai/v1"),
            response_name=response_name,
            provider="xai"
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages array to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                prompt_parts.append(f"Human: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        return "\n\n".join(prompt_parts)
    
    def _send_openai(self, messages, temperature, max_tokens, max_completion_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenAI API with o-series model support"""
        max_retries = 3
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
                params.update(anti_dupe_params)  # Add user's custom parameters
                
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
                
                # Make the API call
                resp = self.openai_client.chat.completions.create(
                    **params,
                    timeout=self.request_timeout
                )
                
                # Validate response
                if not resp or not hasattr(resp, 'choices') or not resp.choices:
                    raise UnifiedClientError("Invalid OpenAI response structure")
                
                choice = resp.choices[0]
                if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                    raise UnifiedClientError("OpenAI response missing content")
                
                content = choice.message.content or ""
                finish_reason = choice.finish_reason
                
                if not content and finish_reason == 'length':
                    print(f"OpenAI vision API returned empty content with finish_reason='length'")
                    print(f"This usually means the token limit is too low. Current limit: {params.get('max_completion_tokens') or params.get('max_tokens', 'not set')}")
                    # Return with error details
                    return UnifiedResponse(
                        content="",
                        finish_reason='error',
                        error_details={'error': 'Response truncated - increase max_completion_tokens', 
                                     'finish_reason': 'length',
                                     'token_limit': params.get('max_completion_tokens') or params.get('max_tokens')}
                    )
                    
                # Normalize OpenAI finish reasons for retry mechanisms
                if finish_reason == "max_tokens":
                    finish_reason = "length"  # Standard truncation indicator
                
                # Extract usage
                usage = None
                if hasattr(resp, 'usage') and resp.usage:
                    usage = {
                        'prompt_tokens': resp.usage.prompt_tokens,
                        'completion_tokens': resp.usage.completion_tokens,
                        'total_tokens': resp.usage.total_tokens
                    }
                
                # Don't save here - the main send() method handles saving
                
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
                elif not fixes_attempted['max_tokens_param'] and "max_tokens" in error_str and ("not supported" in error_str or "unsupported" in error_str):
                    print(f"Model {self.model} requires max_completion_tokens instead of max_tokens, will retry...")
                    fixes_attempted['max_tokens_param'] = True
                    should_retry = True
                
                # Handle rate limits
                elif "rate limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
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

    def _extract_gemini_response_text(self, response, supports_thinking=False, thinking_budget=-1):
        """
        Enhanced extraction method that handles various response structures,
        including when thinking is enabled.
        """
        result = ""
        finish_reason = 'stop'
        
        # Method 1: Try direct text access first
        try:
            if hasattr(response, 'text') and response.text:
                result = response.text
                print(f"   ✅ Got text directly: {len(result)} chars")
                return result, finish_reason
        except Exception as e:
            print(f"   ⚠️ Failed to get text directly: {e}")
        
        # Method 2: Try candidates with enhanced extraction
        if hasattr(response, 'candidates') and response.candidates:
            print(f"   🔍 Number of candidates: {len(response.candidates)}")
            
            for i, candidate in enumerate(response.candidates):
                print(f"   🔍 Checking candidate {i+1}")
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason_str = str(candidate.finish_reason)
                    print(f"   🔍 Finish reason: {finish_reason_str}")
                    if 'MAX_TOKENS' in finish_reason_str:
                        finish_reason = 'length'
                    elif 'SAFETY' in finish_reason_str:
                        finish_reason = 'safety'
                
                # Method 2a: Try candidate.text directly (some models provide this)
                if hasattr(candidate, 'text'):
                    try:
                        if candidate.text:
                            result = candidate.text
                            print(f"   ✅ Got text from candidate.text: {len(result)} chars")
                            return result, finish_reason
                    except:
                        pass
                
                # Method 2b: Extract from content.parts
                if hasattr(candidate, 'content'):
                    # Try direct text on content
                    if hasattr(candidate.content, 'text'):
                        try:
                            if candidate.content.text:
                                result = candidate.content.text
                                print(f"   ✅ Got text from candidate.content.text: {len(result)} chars")
                                return result, finish_reason
                        except:
                            pass
                    
                    # Try parts
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        parts = candidate.content.parts
                        print(f"   🔍 Candidate has {len(parts)} parts")
                        
                        text_parts = []
                        for j, part in enumerate(parts):
                            # Skip thinking parts if they're marked differently
                            if hasattr(part, 'thinking') and part.thinking:
                                print(f"   🔍 Part {j+1} is a thinking part, skipping")
                                continue
                            
                            # Extract text from part
                            if hasattr(part, 'text') and part.text:
                                print(f"   🔍 Part {j+1} has text: {len(part.text)} chars")
                                text_parts.append(part.text)
                        
                        if text_parts:
                            result = ''.join(text_parts)
                            print(f"   ✅ Extracted text from parts: {len(result)} chars")
                            return result, finish_reason
        
        # Method 3: Check for thinking-specific response structure
        if supports_thinking and thinking_budget != 0:
            print("   🔍 Checking for thinking-specific response structure...")
            
            # Some models might have a separate 'output' or 'response' field when thinking is enabled
            if hasattr(response, 'output') and response.output:
                result = str(response.output)
                print(f"   ✅ Got text from response.output: {len(result)} chars")
                return result, finish_reason
            
            if hasattr(response, 'response') and response.response:
                result = str(response.response)
                print(f"   ✅ Got text from response.response: {len(result)} chars")
                return result, finish_reason
        
        # Method 4: Last resort - inspect all attributes
        if not result:
            print("   🔍 Last resort: inspecting all response attributes...")
            attrs = dir(response)
            text_attrs = [attr for attr in attrs if 'text' in attr.lower() or 'content' in attr.lower() or 'output' in attr.lower()]
            print(f"   🔍 Potential text attributes: {text_attrs}")
            
            for attr in text_attrs:
                if not attr.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(response, attr)
                        if value and isinstance(value, str) and len(value) > 10:  # Likely actual content
                            result = value
                            print(f"   ✅ Got text from response.{attr}: {len(result)} chars")
                            return result, finish_reason
                    except:
                        pass
        
        print(f"   ❌ Failed to extract any text from response")
        return result, finish_reason

    def _get_thread_directory(self):
        """Get thread-specific directory for payload storage"""
        thread_name = threading.current_thread().name
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{threading.current_thread().ident}")
        os.makedirs(thread_dir, exist_ok=True)
        return thread_dir

    def _save_gemini_safety_config(self, config_data: dict, response_name: str):
        """Save Gemini safety configuration with thread isolation"""
        if not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
            
        thread_dir = self._get_thread_directory()
        config_filename = f"gemini_safety_{response_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_path = os.path.join(thread_dir, config_filename)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved Gemini safety config to: {config_path}")
        except Exception as e:
            print(f"Failed to save Gemini safety config: {e}")

    def _send_gemini(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Gemini API with support for both text and multi-image messages"""
        
        # Check if we should use OpenAI-compatible endpoint
        use_openai_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
        gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
        
        if use_openai_endpoint and gemini_endpoint:
            # Ensure the endpoint ends with /openai/ for compatibility
            if not gemini_endpoint.endswith('/openai/'):
                if gemini_endpoint.endswith('/'):
                    gemini_endpoint = gemini_endpoint + 'openai/'
                else:
                    gemini_endpoint = gemini_endpoint + '/openai/'
            
            print(f"🔄 Using OpenAI-compatible endpoint for Gemini: {gemini_endpoint}")
            
            # SAVE SAFETY CONFIGURATION FOR GEMINI OPENAI ENDPOINT
            # Check if safety settings are disabled
            disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
            
            # Prepare safety configuration data
            if disable_safety:
                safety_status = "DISABLED - Using OpenAI-compatible endpoint"
                readable_safety = "DISABLED_VIA_OPENAI_ENDPOINT"
            else:
                safety_status = "ENABLED - Using default settings via OpenAI endpoint"
                readable_safety = "DEFAULT"
            
            print(f"🔒 Gemini OpenAI Endpoint Safety Status: {safety_status}")
            
            # Save configuration to file
            config_data = {
                "type": "GEMINI_OPENAI_ENDPOINT_REQUEST",
                "model": self.model,
                "endpoint": gemini_endpoint,
                "safety_enabled": not disable_safety,
                "safety_settings": readable_safety,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Handle None response_name
            if not response_name:
                response_name = f"gemini_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save configuration to file with thread isolation
            self._save_gemini_safety_config(config_data, response_name)
            
            # Route to OpenAI-compatible handler
            return self._send_openai_compatible(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=gemini_endpoint,
                response_name=response_name,
                provider="gemini-openai"
            )
        
        # Import types at the top
        from google.genai import types
        
        # Check if this contains images
        has_images = False
        for msg in messages:
            if isinstance(msg.get('content'), list):
                for part in msg['content']:
                    if part.get('type') == 'image_url':
                        has_images = True
                        break
                if has_images:
                    break
        
        if has_images:
            # Handle as image request - the method now handles both single and multi
            return self._send_gemini_image(messages, None, temperature, max_tokens, response_name)
        
        # text-only logic
        formatted_prompt = self._format_gemini_prompt_simple(messages)
        
        # Check if safety settings are disabled via config
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Get thinking budget from environment
        thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))  
        
        # Check if this model supports thinking
        supports_thinking = self._supports_thinking()
        
        # Configure safety settings based on toggle
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
            logger.info("Gemini safety settings disabled - using BLOCK_NONE for all categories")
        else:
            # Use default safety settings (let Gemini decide)
            safety_settings = None
            logger.info("Using default Gemini safety settings")

        # Define BOOST_FACTOR and current_tokens FIRST
        BOOST_FACTOR = 1
        attempts = 4
        attempt = 0
        result = None
        current_tokens = max_tokens * BOOST_FACTOR
        finish_reason = None
        error_details = {}
        
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
        
        # Log to console with thinking status
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
            
        print(f"🔒 Gemini Safety Status: {safety_status}")
        
        # Save configuration to file
        config_data = {
            "type": "TEXT_REQUEST",
            "model": self.model,
            "safety_enabled": not disable_safety,
            "safety_settings": readable_safety,
            "temperature": temperature,
            "max_output_tokens": current_tokens,
            "thinking_supported": supports_thinking,
            "thinking_budget": thinking_budget if supports_thinking else None,
            "timestamp": datetime.now().isoformat(),
        }

        # Save configuration to file with thread isolation
        self._save_gemini_safety_config(config_data, response_name)
               
        while attempt < attempts:
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                # Get user-configured anti-duplicate parameters
                anti_dupe_params = self._get_anti_duplicate_params(temperature)

                # Build generation config with anti-duplicate parameters
                generation_config_params = {
                    "temperature": temperature,
                    "max_output_tokens": current_tokens,
                    **anti_dupe_params  # Add user's custom parameters
                }
                
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
                
                # Log the request with thinking info
                #print(f"   📤 Sending text request to Gemini{thinking_status}")
                print(f"   📊 Temperature: {temperature}, Max tokens: {current_tokens}")
                
                if supports_thinking:
                    if thinking_budget == 0:
                        print(f"   🧠 Thinking: DISABLED")
                    elif thinking_budget == -1:
                        print(f"   🧠 Thinking: DYNAMIC (model decides)")
                    else:
                        print(f"   🧠 Thinking Budget: {thinking_budget} tokens")
                else:
                    #print(f"   🧠 Model does not support thinking parameter")
                    pass

                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=formatted_prompt,
                    config=generation_config
                )
                
                # Check for blocked content
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        error_details['block_reason'] = str(feedback.block_reason)
                        if disable_safety:
                            print(f"Content blocked despite safety disabled: {feedback.block_reason}")
                        else:
                            print(f"Content blocked: {feedback.block_reason}")
                        raise Exception(f"Content blocked: {feedback.block_reason}")
                
                # Extract text
                try:
                    result = response.text
                    if not result or result.strip() == "":
                        raise Exception("Empty text in response")
                    finish_reason = 'stop'
                except Exception as text_error:
                    print(f"Failed to extract text: {text_error}")
                    
                    # Try to extract from candidates
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            result = ''.join(part.text for part in (parts or []) if hasattr(part, 'text'))
                        
                        # Check finish reason
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = str(candidate.finish_reason)
                            if 'MAX_TOKENS' in finish_reason:
                                finish_reason = 'length'
                
                # Check usage metadata for thinking tokens
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    
                    # Check if thinking tokens were actually disabled (only if model supports thinking)
                    if supports_thinking and hasattr(usage, 'thoughts_token_count'):
                        if usage.thoughts_token_count and usage.thoughts_token_count > 0:
                            print(f"   Thinking tokens used: {usage.thoughts_token_count}")
                        else:
                            print(f"   ✅ Thinking successfully disabled (0 thinking tokens)")
                
                if result:
                    break
                    
            except Exception as e:
                print(f"Gemini attempt {attempt+1} failed: {e}")
                error_details[f'attempt_{attempt+1}'] = str(e)
            
            # No automatic retry - let higher level handle retries
            attempt += 1
            if attempt < attempts:
                print(f"❌ Gemini attempt {attempt} failed, no automatic retry")
                break  # Exit the retry loop
                
        # After getting the response, use the enhanced extraction method
        result, finish_reason = self._extract_gemini_response_text(
            response, 
            supports_thinking=supports_thinking,
            thinking_budget=thinking_budget
        )
        
        if not result:
            print("All Gemini retries failed")
            self._log_truncation_failure(
                messages=messages,
                response_content="",
                finish_reason='error',
                context=self.context,
                error_details={'error': 'all_retries_failed', 'provider': 'gemini', 'attempts': attempt}
            )
            result = "[]" if self.context == 'glossary' else ""
            finish_reason = 'error'
        
        return UnifiedResponse(
            content=result,
            finish_reason=finish_reason,
            raw_response=response if 'response' in locals() else None,
            error_details=error_details if error_details else None
        )
    
    def _format_gemini_prompt_simple(self, messages) -> str:
        """Format messages for Gemini"""
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg['content']
            
            if role == 'SYSTEM':
                formatted_parts.append(f"INSTRUCTIONS: {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def _send_anthropic(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Anthropic API"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
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

        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **anti_dupe_params  # Add user's custom parameters
        }
        
        if system_message:
            data["system"] = system_message
            
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout  # Use configured timeout
                )
                
                if resp.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = api_delay * 10
                        print(f"Anthropic rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                elif resp.status_code != 200:
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                    if attempt < max_retries - 1:
                        print(f"Anthropic API error (attempt {attempt + 1}): {error_msg}")
                        time.sleep(api_delay)
                        continue
                    raise UnifiedClientError(error_msg, http_status=resp.status_code)

                json_resp = resp.json()
                
                # Extract content
                content_parts = json_resp.get("content", [])
                if isinstance(content_parts, list):
                    content = "".join(part.get("text", "") for part in content_parts)
                else:
                    content = str(content_parts)
                
                # Extract finish reason
                finish_reason = json_resp.get("stop_reason")
                # Map Anthropic finish reasons to standard ones
                if finish_reason == "max_tokens":
                    finish_reason = "length"  # Standard truncation indicator
                elif finish_reason == "stop_sequence":
                    finish_reason = "stop"
                
                # Extract usage
                usage = json_resp.get("usage")
                if usage:
                    usage = {
                        'prompt_tokens': usage.get('input_tokens', 0),
                        'completion_tokens': usage.get('output_tokens', 0),
                        'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                    }
                    
                # Don't save here - the main send() method handles saving
                
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_response=json_resp
                )
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Anthropic API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"Anthropic API error after all retries: {e}")
                raise UnifiedClientError(f"Anthropic API error: {e}")
    
    def _send_mistral(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Mistral API"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        if MistralClient and hasattr(self, 'mistral_client'):
            # Use SDK if available
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    chat_messages = []
                    for msg in messages:
                        chat_messages.append(ChatMessage(role=msg['role'], content=msg['content']))
                    
                    response = self.mistral_client.chat(
                        model=self.model,
                        messages=chat_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    
                    # Don't save here - the main send() method handles saving
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason=finish_reason,
                        raw_response=response
                    )
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Mistral SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    print(f"Mistral SDK error after all retries: {e}")
                    raise UnifiedClientError(f"Mistral SDK error: {e}")
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
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        if cohere and hasattr(self, 'cohere_client'):
            # Use SDK
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    # Format messages for Cohere
                    chat_history = []
                    message = ""
                    
                    for msg in messages:
                        if msg['role'] == 'user':
                            message = msg['content']
                        elif msg['role'] == 'assistant':
                            chat_history.append({"role": "CHATBOT", "message": msg['content']})
                        elif msg['role'] == 'system':
                            # Prepend system message to user message
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
                    
                    # Don't save here - the main send() method handles saving
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason=finish_reason,
                        raw_response=response
                    )
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Cohere SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    print(f"Cohere SDK error after all retries: {e}")
                    raise UnifiedClientError(f"Cohere SDK error: {e}")
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
            
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    resp = requests.post(
                        "https://api.cohere.ai/v1/chat",
                        headers=headers,
                        json=data
                    )
                    
                    if resp.status_code != 200:
                        raise UnifiedClientError(f"Cohere API error: {resp.status_code} - {resp.text}")
                    
                    json_resp = resp.json()
                    content = json_resp.get("text", "")
                    
                    # Don't save here - the main send() method handles saving
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason='stop',
                        raw_response=json_resp
                    )
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Cohere API error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    print(f"Cohere API error after all retries: {e}")
                    raise UnifiedClientError(f"Cohere API error: {e}")
    
    def _send_ai21(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to AI21 API"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages for AI21
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += "Assistant: "
        
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": max_tokens
        }
        
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                resp = requests.post(
                    f"https://api.ai21.com/studio/v1/{self.model}/complete",
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout  # Use configured timeout
                )
                
                if resp.status_code != 200:
                    error_msg = f"AI21 API error: {resp.status_code} - {resp.text}"
                    if resp.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = api_delay * 10
                            print(f"AI21 rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    raise UnifiedClientError(error_msg)
                
                json_resp = resp.json()
                completions = json_resp.get("completions", [])
                
                if completions:
                    content = completions[0].get("data", {}).get("text", "")
                else:
                    content = ""
                
                # Don't save here - the main send() method handles saving
                
                return UnifiedResponse(
                    content=content,
                    finish_reason='stop',
                    raw_response=json_resp
                )
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"AI21 API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"AI21 API error after all retries: {e}")
                raise UnifiedClientError(f"AI21 API error: {e}")
    
    def _send_together(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Together AI API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.together.xyz/v1",
            response_name=response_name,
            provider="together"
        )
    
    def _send_perplexity(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Perplexity API with Sonar models"""
        # Check for safety settings
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Use chat completions endpoint
        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            'model': self.model,
            'messages': messages
        }
        
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens
        
        # Add search options for Sonar models
        if 'sonar' in self.model.lower():
            payload['search_domain_filter'] = ['perplexity.ai']
            payload['return_citations'] = True
            payload['search_recency_filter'] = 'month'
        
        # Get response using HTTP request
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                
                if resp.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = api_delay * 10
                        print(f"Perplexity rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                elif resp.status_code != 200:
                    error_msg = f"Perplexity API error: {resp.status_code} - {resp.text}"
                    if attempt < max_retries - 1:
                        print(f"{error_msg} (attempt {attempt + 1})")
                        time.sleep(api_delay)
                        continue
                    raise UnifiedClientError(error_msg, http_status=resp.status_code)
                
                json_resp = resp.json()
                
                # Extract content
                choices = json_resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason", "stop")
                else:
                    content = ""
                    finish_reason = "error"
                
                # Normalize finish reasons
                if finish_reason in ["max_tokens", "max_length"]:
                    finish_reason = "length"
                
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    raw_response=json_resp
                )
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Perplexity API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"Perplexity API error after all retries: {e}")
                raise UnifiedClientError(f"Perplexity API error: {e}")
    
    def _send_replicate(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Replicate API"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages as single prompt
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"{msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n"
        
        # Replicate uses versioned models
        data = {
            "version": self.model,  # Model should be the version ID
            "input": {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                # Create prediction
                resp = requests.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout  # Use configured timeout
                )
                
                if resp.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = api_delay * 10
                        print(f"Replicate rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                elif resp.status_code != 201:
                    error_msg = f"Replicate API error: {resp.status_code} - {resp.text}"
                    if attempt < max_retries - 1:
                        print(f"{error_msg} (attempt {attempt + 1})")
                        time.sleep(api_delay)
                        continue
                    raise UnifiedClientError(error_msg)
                
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
                
                # Don't save here - the main send() method handles saving
                
                return UnifiedResponse(
                    content=content,
                    finish_reason='stop',
                    raw_response=result
                )
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Replicate API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"Replicate API error after all retries: {e}")
                raise UnifiedClientError(f"Replicate API error: {e}")
    
    def _send_deepseek(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to DeepSeek API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1"),
            response_name=response_name,
            provider="deepseek"
        )
    
    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url, 
                                response_name, provider="generic", headers=None) -> UnifiedResponse:
        """Send request to OpenAI-compatible APIs with safety settings"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        # CUSTOM ENDPOINT OVERRIDE - Check if enabled and override base_url
        use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
        actual_api_key = self.api_key  # Store original API key
        
        # Determine if this is a local endpoint that doesn't need a real API key
        is_local_endpoint = False
        
        if use_custom_endpoint and provider != "gemini-openai":
            custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            if custom_base_url:
                print(f"🔄 Custom endpoint enabled: Overriding {provider} endpoint")
                print(f"   Original: {base_url}")
                print(f"   Override: {custom_base_url}")
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
                         'electronhub', 'openrouter', 'fireworks', 'xai', 'gemini-openai']
        
        if openai and provider in sdk_compatible:
            # Use OpenAI SDK with custom base URL
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    client = openai.OpenAI(
                        api_key=actual_api_key,  # Uses real key for cloud, dummy for local
                        base_url=base_url,
                        timeout=float(self.request_timeout)
                    )
                    
                    # Get user-configured anti-duplicate parameters
                    anti_dupe_params = self._get_anti_duplicate_params(temperature)

                    params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **anti_dupe_params  # Add user's custom parameters
                    }
                    
                    # Add safety parameters for providers that support them
                    if disable_safety and provider in ["groq", "fireworks", "together"]:
                        params["moderation"] = False
                        logger.info(f"🔓 Safety moderation disabled for {provider}")
                    
                    resp = client.chat.completions.create(**params)
                    
                    content = resp.choices[0].message.content
                    finish_reason = resp.choices[0].finish_reason
                    
                    # ADD ELECTRONHUB TRUNCATION DETECTION HERE TOO!
                    if provider == "electronhub" and content:
                        # Additional validation for ElectronHub responses
                        if len(content) < 50 and "cannot" in content.lower():
                            # Very short response with "cannot" - likely refused
                            finish_reason = "content_filter"
                            print(f"ElectronHub likely refused content: {content[:100]}")
                            self._log_truncation_failure(
                                messages=messages,
                                response_content=content,
                                finish_reason='content_filter',
                                context=self.context,
                                error_details={'type': 'content_refused', 'provider': 'electronhub'}
                            )
                        elif finish_reason == "stop":
                            # Check if content looks truncated despite "stop" status
                            if self._detect_silent_truncation(content, messages, self.context):
                                finish_reason = "length"
                                print("ElectronHub reported 'stop' but content appears truncated")
                                print(f"🔍 ElectronHub: Detected silent truncation despite 'stop' status")
                                self._log_truncation_failure(
                                    messages=messages,
                                    response_content=content,
                                    finish_reason='length',
                                    context=self.context,
                                    error_details={'type': 'silent_truncation', 'provider': 'electronhub'}
                                )
                                
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
                    
                    # For rate limits, immediately re-raise to let multi-key system handle it
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        print(f"{provider} rate limit hit - passing to multi-key handler")
                        raise UnifiedClientError(f"{provider} rate limit: {e}", error_type="rate_limit")
                    
                    # For other errors, retry at this level
                    if attempt < max_retries - 1:
                        print(f"{provider} SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                        
                    # Final attempt failed
                    print(f"{provider} SDK error after all retries: {e}")
                    raise UnifiedClientError(f"{provider} SDK error: {e}")
        else:
            # Use HTTP API with retry logic
            if headers is None:
                headers = {
                    "Authorization": f"Bearer {actual_api_key}",  # Uses real key for cloud, dummy for local
                    "Content-Type": "application/json"
                }
            
            # Add safety-related headers for providers that support them
            if disable_safety:
                # OpenRouter specific safety settings
                if provider == "openrouter" and "X-Safe-Mode" not in headers:
                    headers['X-Safe-Mode'] = 'false'
                    logger.info(f"🔓 {provider} Safety: Disabled via X-Safe-Mode header")
            
            # Some providers need special headers
            if provider == 'zhipu':
                headers["Authorization"] = f"Bearer {actual_api_key}"
            elif provider == 'baidu':
                # Baidu might need special auth handling
                headers["Content-Type"] = "application/json"
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add safety parameters for compatible providers
            if disable_safety:
                if provider in ["openai", "groq", "fireworks", "together"]:
                    data["moderation"] = False
                    logger.info(f"🔓 {provider} Safety: Moderation disabled")
                elif provider == "poe":
                    data["safe_mode"] = False
                    logger.info(f"🔓 {provider} Safety: Safe mode disabled")
            
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    # Some providers use different endpoints
                    endpoint = "/chat/completions"
                    if provider == 'zhipu':
                        endpoint = "/chat/completions"
                    
                    resp = requests.post(
                        f"{base_url}{endpoint}",
                        headers=headers,
                        json=data,
                        timeout=self.request_timeout  # Use configured timeout
                    )
                    
                    # Check for rate limit FIRST, before other status codes
                    if resp.status_code == 429:
                        # Extract any retry information if available
                        retry_after = resp.headers.get('Retry-After', '60')
                        print(f"{provider} rate limit hit (429) - passing to multi-key handler")
                        raise UnifiedClientError(
                            f"{provider} rate limit: {resp.text}", 
                            error_type="rate_limit",
                            http_status=429
                        )
                    
                    # Handle other error status codes
                    if resp.status_code != 200:
                        error_msg = f"{provider} API error: {resp.status_code} - {resp.text}"
                        if attempt < max_retries - 1:
                            print(f"{error_msg} (attempt {attempt + 1})")
                            time.sleep(api_delay)
                            continue
                        raise UnifiedClientError(error_msg, http_status=resp.status_code)
                    
                    json_resp = resp.json()
                    choices = json_resp.get("choices", [])
                    
                    if not choices:
                        raise UnifiedClientError(f"{provider} API returned no choices")
                    
                    content = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason", "stop")
                    
                    # ElectronHub truncation detection (already in HTTP branch)
                    if provider == "electronhub" and content:
                        # Additional validation for ElectronHub responses
                        if len(content) < 50 and "cannot" in content.lower():
                            # Very short response with "cannot" - likely refused
                            finish_reason = "content_filter"
                            print(f"ElectronHub likely refused content: {content[:100]}")
                        elif finish_reason == "stop":
                            # Check if content looks truncated despite "stop" status
                            if self._detect_silent_truncation(content, messages, self.context):
                                finish_reason = "length"
                                print("ElectronHub reported 'stop' but content appears truncated")
                                print(f"🔍 ElectronHub: Detected silent truncation despite 'stop' status")                    
                    
                    usage = json_resp.get("usage")
                    if usage:
                        usage = {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0)
                        }
                    
                    # Don't save here - the main send() method handles saving
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw_response=json_resp
                    )
                    
                except UnifiedClientError:
                    # Re-raise our own errors immediately (including rate limits)
                    raise
                    
                except requests.RequestException as e:
                    # For connection errors, retry at this level
                    if attempt < max_retries - 1:
                        print(f"{provider} API error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    print(f"{provider} API error after all retries: {e}")
                    raise UnifiedClientError(f"{provider} API error: {e}")
    
    # Provider-specific implementations using generic handler
    def _send_yi(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Yi API"""
        base_url = os.getenv("YI_API_BASE_URL", "https://api.01.ai/v1")
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=base_url,
            response_name=response_name,
            provider="yi"
        )
    
    def _send_qwen(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Qwen API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            response_name=response_name,
            provider="qwen"
        )
    
    def _send_baichuan(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Baichuan API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.baichuan-ai.com/v1",
            response_name=response_name,
            provider="baichuan"
        )
    
    def _send_zhipu(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Zhipu AI (GLM/ChatGLM)"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            response_name=response_name,
            provider="zhipu"
        )
    
    def _send_moonshot(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Moonshot API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.moonshot.cn/v1",
            response_name=response_name,
            provider="moonshot"
        )
    
    def _send_groq(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Groq API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1"),
            response_name=response_name,
            provider="groq"
        )
    
    def _send_baidu(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Baidu Ernie API"""
        # Baidu ERNIE has a specific auth flow - this is simplified
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            response_name=response_name,
            provider="baidu"
        )
    
    def _send_tencent(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Tencent Hunyuan API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://hunyuan.cloud.tencent.com/v1",
            response_name=response_name,
            provider="tencent"
        )
    
    def _send_iflytek(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to iFLYTEK Spark API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://spark-api.xf-yun.com/v1",
            response_name=response_name,
            provider="iflytek"
        )
    
    def _send_bytedance(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to ByteDance Doubao API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://maas-api.vercel.app/v1",
            response_name=response_name,
            provider="bytedance"
        )
    
    def _send_minimax(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to MiniMax API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.minimax.chat/v1",
            response_name=response_name,
            provider="minimax"
        )
    
    def _send_sensenova(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to SenseNova API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.sensenova.cn/v1",
            response_name=response_name,
            provider="sensenova"
        )
    
    def _send_internlm(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to InternLM API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.internlm.org/v1",
            response_name=response_name,
            provider="internlm"
        )
    
    def _send_tii(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to TII Falcon API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.tii.ae/v1",
            response_name=response_name,
            provider="tii"
        )
    
    def _send_microsoft(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Microsoft API (Phi, Orca)"""
        # Microsoft models often through Azure
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.microsoft.com/v1",
            response_name=response_name,
            provider="microsoft"
        )
    
    def _send_azure(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Azure OpenAI"""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com")
        api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Azure uses a different URL structure
        base_url = f"{endpoint}/openai/deployments/{self.model}"
        url = f"{base_url}/chat/completions?api-version={api_version}"
        
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"Azure OpenAI error: {resp.status_code} - {resp.text}")
            
            json_resp = resp.json()
            content = json_resp['choices'][0]['message']['content']
            finish_reason = json_resp['choices'][0]['finish_reason']
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
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
        
        # Format messages for Aleph Alpha
        prompt = self._messages_to_prompt(messages)
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "maximum_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            resp = requests.post(
                "https://api.aleph-alpha.com/complete",
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"Aleph Alpha error: {resp.status_code} - {resp.text}")
            
            json_resp = resp.json()
            content = json_resp['completions'][0]['completion']
            
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Aleph Alpha error: {e}")
            raise UnifiedClientError(f"Aleph Alpha error: {e}")
    
    def _send_databricks(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Databricks API"""
        workspace_url = os.getenv("DATABRICKS_API_URL", "https://YOUR-WORKSPACE.databricks.com")
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=f"{workspace_url}/serving/endpoints",
            response_name=response_name,
            provider="databricks"
        )
    
    def _send_huggingface(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to HuggingFace Inference API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format messages for HuggingFace
        prompt = self._messages_to_prompt(messages)
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"HuggingFace error: {resp.status_code} - {resp.text}")
            
            json_resp = resp.json()
            content = json_resp[0]['generated_text']
            
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"HuggingFace error: {e}")
            raise UnifiedClientError(f"HuggingFace error: {e}")
    
    def _send_salesforce(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Salesforce CodeGen API"""
        api_url = os.getenv("SALESFORCE_API_URL", "https://api.salesforce.com/v1")
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=api_url,
            response_name=response_name,
            provider="salesforce"
        )
    
    # Image handling methods
    def send_image(self, messages: List[Dict[str, Any]], image_data: Any,
                  temperature: Optional[float] = None, 
                  max_tokens: Optional[int] = None,
                  max_completion_tokens: Optional[int] = None,
                  context: str = 'image_translation') -> Tuple[str, str]:
        """Thread-safe image send with proper key management for batch translation"""
        thread_name = threading.current_thread().name
        
        # Ensure thread has a client
        self._ensure_thread_client()
        
        logger.info(f"[{thread_name}] Using {self.key_identifier} for image: {context}")
        
        max_retries = 3
        retry_count = 0
        last_error = None
        
        # Track which keys we've already tried
        attempted_keys = set()
        
        while retry_count < max_retries:
            try:
                # Track current key
                attempted_keys.add(self.key_identifier)
                
                # Call the actual implementation
                result = self._send_image_internal(messages, image_data, temperature,
                                                 max_tokens, max_completion_tokens, context)
                
                # Mark success
                if self._multi_key_mode:
                    tls = self._get_thread_local_client()
                    if tls.key_index is not None:
                        self._api_key_pool.mark_key_success(tls.key_index)
                
                logger.info(f"[{thread_name}] ✓ Image request completed")
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Log the error with key info
                logger.error(f"[{thread_name}] ✗ {self.key_identifier} image error: {error_str[:100]}")
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                    if self._multi_key_mode:
                        print(f"[Thread-{thread_name}] Image rate limit hit on {self.key_identifier}")
                        
                        # Handle rate limit for this thread
                        self._handle_rate_limit_for_thread()
                        
                        # Check if we have any available keys
                        available_count = self._count_available_keys()
                        if available_count == 0:
                            logger.error(f"[{thread_name}] All API keys are cooling down for images")
                            
                            # If we still have retries left, wait for the shortest cooldown
                            if retry_count < max_retries - 1:
                                cooldown_time = self._get_shortest_cooldown_time()
                                print(f"⏳ [Thread-{thread_name}] All keys cooling down for image request, waiting {cooldown_time}s...")
                                
                                # Wait with cancellation check
                                for i in range(cooldown_time):
                                    if self._cancelled:
                                        raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                    time.sleep(1)
                                    if i % 10 == 0 and i > 0:
                                        print(f"⏳ [Thread-{thread_name}] Still waiting for image API... {cooldown_time - i}s remaining")
                                
                                # Force re-initialization after cooldown
                                tls = self._get_thread_local_client()
                                tls.initialized = False
                                self._ensure_thread_client()
                                
                                retry_count += 1
                                continue
                            else:
                                # No more retries, raise the error
                                raise UnifiedClientError("All API keys are cooling down", error_type="rate_limit")
                        
                        # Check if we've tried too many keys
                        if len(attempted_keys) >= len(self._api_key_pool.keys):
                            logger.error(f"[{thread_name}] Attempted all {len(self._api_key_pool.keys)} keys for image")
                            raise UnifiedClientError("All API keys rate limited for image requests", error_type="rate_limit")
                        
                        retry_count += 1
                        logger.info(f"[{thread_name}] Retrying image with new key, attempt {retry_count}/{max_retries}")
                        continue
                    else:
                        # Single key mode
                        if retry_count < max_retries - 1:
                            wait_time = min(30 * (retry_count + 1), 120)
                            logger.info(f"[{thread_name}] Single key image rate limit, waiting {wait_time}s")
                            
                            for i in range(wait_time):
                                if self._cancelled:
                                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                time.sleep(1)
                                
                            retry_count += 1
                            continue
                        else:
                            raise UnifiedClientError(f"Image rate limit exceeded after {max_retries} attempts", error_type="rate_limit")
                
                # For cancellation or timeout, don't retry
                elif isinstance(e, UnifiedClientError) and e.error_type in ["cancelled", "timeout"]:
                    raise
                
                # For other errors in multi-key mode
                elif self._multi_key_mode and retry_count < max_retries - 1:
                    tls = self._get_thread_local_client()
                    if tls.key_index is not None:
                        self._api_key_pool.mark_key_error(tls.key_index)
                    
                    if not self._force_rotation:
                        # Error-based rotation mode
                        logger.info(f"[{thread_name}] Image error, rotating to new key...")
                        
                        # Force reassignment
                        tls.initialized = False
                        tls.request_count = 0
                        self._ensure_thread_client()
                        
                        retry_count += 1
                        continue
                    else:
                        # Force rotation mode - retry same key
                        retry_count += 1
                        time.sleep(2)
                        continue
                
                # Can't retry, raise the error
                raise
        
        # Should not reach here
        if last_error:
            raise last_error
        else:
            raise Exception(f"Image request failed after {max_retries} attempts")


    def _send_image_internal(self, messages: List[Dict[str, Any]], image_data: Any,
                            temperature: Optional[float] = None, 
                            max_tokens: Optional[int] = None,
                            max_completion_tokens: Optional[int] = None,
                            context: str = 'image_translation') -> Tuple[str, str]:
        """
        Internal implementation of send_image with integrated 500 error retry logic
        """
        self._cancelled = False
        self.context = context or 'image_translation'
        self.conversation_message_count += 1
        
        # Use GUI values if not explicitly overridden
        if temperature is None:
            temperature = getattr(self, 'default_temperature', 0.3)
            logger.debug(f"Using default temperature: {temperature}")
        
        # Determine if this is an o-series model
        is_o_series = self._is_o_series_model()
        
        # Handle token limits based on model type
        if is_o_series:
            # o-series models use max_completion_tokens
            if max_completion_tokens is None:
                max_completion_tokens = max_tokens if max_tokens is not None else getattr(self, 'default_max_tokens', 8192)
            max_tokens = None  # Clear max_tokens for o-series
            logger.info(f"Using o-series model {self.model} with max_completion_tokens: {max_completion_tokens}")
        else:
            # Regular models use max_tokens
            if max_tokens is None:
                max_tokens = max_completion_tokens if max_completion_tokens is not None else getattr(self, 'default_max_tokens', 8192)
            max_completion_tokens = None  # Clear max_completion_tokens for regular models
            logger.debug(f"Using regular model {self.model} with max_tokens: {max_tokens}")
        
        # Convert to base64 if needed
        if isinstance(image_data, bytes):
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            logger.debug(f"Converted {len(image_data)} bytes to base64")
        else:
            image_base64 = image_data
        
        # Internal retry logic for 500 errors
        internal_retries = 3
        for attempt in range(internal_retries):
            try:
                os.makedirs("Payloads", exist_ok=True)
                
                messages = self._apply_pure_reinforcement(messages)
                
                # Use proper naming for duplicate detection
                payload_name, response_name = self._get_file_names(messages, context=self.context)
                
                # Log the request details
                logger.info(f"Sending image request to {self.client_type} ({self.model})")
                logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens or max_completion_tokens}")
                
                # Check provider vision support with latest models (2025)
                vision_providers = {
                    'openai': ['gpt-4-vision', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.5', 'gpt-4.1', 
                              'gpt-4.1-mini', 'o1-vision', 'o3', 'o3-mini', 'o3-pro', 'o4', 'o4-mini'],
                    'anthropic': ['claude-3', 'claude-3.5', 'claude-3-opus', 'claude-3.5-sonnet', 'claude-3-haiku', 
                                 'claude-3.5-haiku', 'claude-3.7-sonnet', 'claude-4-opus', 'claude-4-sonnet',
                                 'claude-opus-4', 'claude-sonnet-4'],
                    'gemini': ['gemini-pro-vision', 'gemini-1.5', 'gemini-2.0', 'gemini-2.5', 'gemini-flash', 
                              'gemini-flash-lite', 'gemini-2.5-pro', 'gemini-2.5-flash'],
                    'poe': ['claude-3-opus', 'claude-4-opus', 'claude-4-sonnet', 'gpt-4', 'gpt-4o', 'gpt-4.5', 
                           'gemini-pro', 'claude-3.5-sonnet', 'gemini-2.5-pro'],
                    'openrouter': ['any'],  # Supports routing to any vision model
                    'groq': ['llava', 'vision'],  # Groq supports some vision models
                    'fireworks': ['firellava', 'vision'],  # Fireworks vision models
                    'together': ['llava', 'fuyu', 'cogvlm'],
                    'replicate': ['blip', 'clip', 'llava', 'minigpt4'],
                    'huggingface': ['vision-transformer', 'vit', 'clip', 'blip'],
                    'deepseek': ['deepseek-vl', 'deepseek-r1-vl'],  # DeepSeek vision language models
                    'qwen': ['qwen-vl', 'qwen2-vl', 'qwen2.5-vl'],  # Qwen vision models
                    'yi': ['yi-vl'],  # Yi vision models
                    'moonshot': ['moonshot-v1-vision'],
                    'electronhub': ['any'],  # Can route to any vision model
                    'perplexity': [],  # Perplexity doesn't support direct image input
                    'cohere': ['aya-vision'],  # Cohere's multimodal Aya Vision
                    'tii': ['falcon-2-11b'],  # Falcon 2 with vision support
                    'xai': ['grok-3', 'grok-vision'],  # Grok models with vision
                    'meta': ['llama-4-vision'],  # Meta's Llama 4 with vision
                    'vertex_model_garden': ['gemini', 'imagen', 'claude'],  # Vertex AI Model Garden vision models
                }
                
                # Check if provider supports vision
                if self.client_type not in vision_providers:
                    raise UnifiedClientError(f"Provider {self.client_type} does not support image input")
                
                # Check if specific model supports vision
                supported_models = vision_providers.get(self.client_type, [])
                if supported_models != ['any']:
                    model_supported = any(model in self.model.lower() for model in supported_models)
                    if not model_supported:
                        raise UnifiedClientError(f"Model {self.model} does not support image input")
                
                # Route to appropriate handler based on client type
                if self.client_type == 'gemini':
                    response = self._send_gemini_image(messages, image_base64, temperature, 
                                                     max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'openai':
                    response = self._send_openai_image(messages, image_base64, temperature, 
                                                 max_tokens, max_completion_tokens, response_name)
                elif self.client_type == 'anthropic':
                    response = self._send_anthropic_image(messages, image_base64, temperature, 
                                                        max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'electronhub':
                    response = self._send_electronhub_image(messages, image_base64, temperature, 
                                                          max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'poe':
                    response = self._send_poe_image(messages, image_base64, temperature,
                                                  max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'openrouter':
                    response = self._send_openrouter_image(messages, image_base64, temperature,
                                                         max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'cohere':
                    response = self._send_cohere_image(messages, image_base64, temperature,
                                                     max_tokens or max_completion_tokens, response_name)
                elif self.client_type == 'vertex_model_garden':
                    response = self._send_vertex_model_garden_image(messages, image_base64, temperature, 
                                                                   max_tokens or max_completion_tokens, response_name)
                else:
                    raise UnifiedClientError(f"Image input not supported for {self.client_type}")
                
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled by user")
                
                # Save response for duplicate detection
                if response.content:
                    self._save_response(response.content, response_name)
                    logger.debug(f"Saved response to: {response_name}")
                
                # Handle empty responses
                if not response.content or response.content.strip() == "":
                    print(f"Empty response from {self.client_type}")
                    
                    # Log empty image responses
                    self._log_truncation_failure(
                        messages=messages,
                        response_content="",
                        finish_reason='error',
                        context=context or 'image_translation',
                        error_details={'error': 'empty_image_response'}
                    )
                    
                    fallback = self._handle_empty_result(messages, context, "empty_image_response")
                    return fallback, 'error'
                
                # Mark key as successful for image request
                if self.use_multi_keys:
                    self._mark_key_success()
                
                # Log truncation for retry mechanism
                if response.is_truncated:
                    print(f"Image response was truncated: {response.finish_reason}")
                    print(f"⚠️ Image response truncated (finish_reason: {response.finish_reason})")
                    
                    # Log image truncation failures
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=response.content,
                        finish_reason=response.finish_reason,
                        context=context or 'image_translation',
                        error_details=response.error_details
                    )               

                # Apply API delay after successful image call
                # SKIP DELAY DURING CLEANUP
                if not self._in_cleanup:
                    api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
                    if api_delay > 0:
                        print(f"⏳ Waiting {api_delay}s before next API call...")
                        time.sleep(api_delay)
                else:
                    print("⚡ Skipping API delay (cleanup mode)")

                return response.content, response.finish_reason
                    
            except UnifiedClientError as e:
                # Re-raise our own errors
                if e.error_type == "cancelled" or "cancelled" in str(e):
                    self._in_cleanup = False  # Ensure cleanup flag is set
                    print(f"Image processing cancelled: {e}")
                    raise
                
                print(f"Image processing error: {e}")
                error_str = str(e).lower()
                
                # Check if it's a rate limit that needs to be handled by outer retry logic
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic in outer send_image() method
                
                # Check for prohibited content - never retry these
                content_filter_indicators = [
                    "content_filter", "content was blocked", "response was blocked",
                    "safety filter", "content policy", "harmful content",
                    "blocked by safety", "harm_category", "inappropriate image"
                ]
                
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Image content prohibited - not retrying: {error_str[:100]}")
                    self._save_failed_request(messages, e, context)
                    raise  # Re-raise without retry
                
                # Check for 500 errors - retry these
                http_status = getattr(e, 'http_status', None)
                if http_status == 500 or "500" in error_str or "api_error" in error_str:
                    if attempt < internal_retries - 1:
                        wait_time = min(5 * (attempt + 1), 15)  # 5s, 10s, 15s backoff
                        print(f"🔄 Image server error (500) - auto-retrying in {wait_time}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        for i in range(wait_time):
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(1)
                        continue
                    else:
                        print(f"❌ Image server error (500) - exhausted {internal_retries} retries")
                
                # Save failed request and raise
                self._save_failed_request(messages, e, context)
                raise
                
            except Exception as e:
                # Wrap other errors
                print(f"Unexpected image processing error: {e}")
                error_str = str(e).lower()
                
                # Check if it's a rate limit that wasn't caught
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic
                
                # Check for timeout errors
                if "timed out" in error_str:
                    raise UnifiedClientError(f"Image request timed out: {e}", error_type="timeout")
                
                # Check for prohibited content in unexpected errors
                content_filter_indicators = [
                    "content_filter", "content was blocked", "response was blocked",
                    "safety filter", "content policy", "harmful content",
                    "blocked by safety", "harm_category", "inappropriate"
                ]
                
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Image content prohibited - not retrying")
                    self._save_failed_request(messages, e, context)
                    fallback = self._handle_empty_result(messages, context, str(e))
                    return fallback, 'error'
                
                # Check for 500 errors in unexpected exceptions
                if "500" in error_str or "internal server error" in error_str:
                    if attempt < internal_retries - 1:
                        wait_time = min(5 * (attempt + 1), 15)
                        print(f"🔄 Image server error (500) - auto-retrying in {wait_time}s (attempt {attempt + 1}/{internal_retries})")
                        
                        for i in range(wait_time):
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(1)
                        continue
                
                # Check for other transient errors
                transient_errors = ["502", "503", "504", "connection reset", "connection aborted"]
                if any(err in error_str for err in transient_errors):
                    if attempt < internal_retries - 1:
                        wait_time = min(3 * (attempt + 1), 10)
                        print(f"🔄 Image transient error - retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                
                # Save failed request and return fallback
                self._save_failed_request(messages, e, context)
                fallback = self._handle_empty_result(messages, context, str(e))
                return fallback, 'error'

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
        """Check if the current model is an o-series model (o1, o3, o4, etc.)"""
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
        
        # Check if it starts with o followed by a digit
        if len(model_lower) >= 2 and model_lower[0] == 'o' and model_lower[1].isdigit():
            return True
        
        return False

    def _send_gemini_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Gemini API - supports both single and multiple images"""
        try:
            # Check if we should use OpenAI-compatible endpoint
            use_openai_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
            gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
            
            if use_openai_endpoint and gemini_endpoint:
                # Ensure the endpoint ends with /openai/ for compatibility
                if not gemini_endpoint.endswith('/openai/'):
                    if gemini_endpoint.endswith('/'):
                        gemini_endpoint = gemini_endpoint + 'openai/'
                    else:
                        gemini_endpoint = gemini_endpoint + '/openai/'
                
                print(f"🔄 Using OpenAI-compatible endpoint for Gemini image: {gemini_endpoint}")
                
                # SAVE SAFETY CONFIGURATION FOR GEMINI OPENAI ENDPOINT (IMAGE)
                # Check if safety settings are disabled
                disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
                
                # Prepare safety configuration data
                if disable_safety:
                    safety_status = "DISABLED - Using OpenAI-compatible endpoint"
                    readable_safety = "DISABLED_VIA_OPENAI_ENDPOINT"
                else:
                    safety_status = "ENABLED - Using default settings via OpenAI endpoint"
                    readable_safety = "DEFAULT"
                
                print(f"🔒 Gemini OpenAI Endpoint Safety Status (Image): {safety_status}")
                
                # Save configuration to file
                config_data = {
                    "type": "GEMINI_OPENAI_ENDPOINT_IMAGE_REQUEST",
                    "model": self.model,
                    "endpoint": gemini_endpoint,
                    "safety_enabled": not disable_safety,
                    "safety_settings": readable_safety,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Handle None response_name
                if not response_name:
                    response_name = f"gemini_openai_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Save configuration to file with thread isolation
                self._save_gemini_safety_config(config_data, response_name)
                
                # Route to OpenAI-compatible handler
                return self._send_openai_compatible(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=gemini_endpoint,
                    response_name=response_name,
                    provider="gemini-openai"
                )
        
            # Import types at the top
            from google.genai import types
            
            # Check if safety settings are disabled
            disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
           
            # Get thinking budget from environment
            thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))  
            
            # Check if this model supports thinking
            supports_thinking = self._supports_thinking()
            
            # Configure safety settings
            safety_settings = None
            if disable_safety:
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
                print(f"🔒 Gemini Safety Status: DISABLED - All categories set to BLOCK_NONE")
            else:
                print(f"🔒 Gemini Safety Status: ENABLED - Using default settings")
                pass
            
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
            
            # Get anti-duplicate params
            anti_dupe_params = self._get_anti_duplicate_params(temperature)
            
            # Build generation config params dictionary FIRST
            generation_config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **anti_dupe_params
            }
            
            # Only add thinking_config if the model supports it
            if supports_thinking:
                # Create thinking config separately
                thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
                
                # Create generation config with all parameters INCLUDING thinking_config
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
            
            # Log the request
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
                
            if is_multi_image:
                print(f"   📤 Sending multi-image request to Gemini{thinking_status}")
            else:
                print(f"   📤 Sending single image request to Gemini{thinking_status}")
            print(f"   📊 Temperature: {temperature}, Max tokens: {max_tokens}")
            
            if supports_thinking:
                if thinking_budget == 0:
                    print(f"   🧠 Thinking: DISABLED")
                elif thinking_budget == -1:
                    print(f"   🧠 Thinking: DYNAMIC (model decides)")
                else:
                    print(f"   🧠 Thinking Budget: {thinking_budget} tokens")
            else:
                #print(f"   🧠 Model does not support thinking parameter")
                pass
            
            # Make the API call
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config
            )
            print(f"   🔍 Raw response type: {type(response)}")
            
            # Check prompt feedback first
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = response.prompt_feedback
                print(f"   🔍 Prompt feedback: {feedback}")
                if hasattr(feedback, 'block_reason') and feedback.block_reason:
                    print(f"   ❌ Content blocked by Gemini: {feedback.block_reason}")
                    return UnifiedResponse(
                        content="",
                        finish_reason='safety',
                        error_details={'block_reason': str(feedback.block_reason)}
                    )
            # Use the enhanced extraction method
            result, finish_reason = self._extract_gemini_response_text(
                response,
                supports_thinking=supports_thinking,
                thinking_budget=thinking_budget
            )
            
            # Check usage metadata for debugging
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                
                # Check if thinking tokens were actually disabled/limited (only if model supports thinking)
                if supports_thinking and hasattr(usage, 'thoughts_token_count'):
                    if usage.thoughts_token_count and usage.thoughts_token_count > 0:
                        if thinking_budget > 0 and usage.thoughts_token_count > thinking_budget:
                            print(f"   ⚠️ WARNING: Thinking tokens exceeded budget: {usage.thoughts_token_count} > {thinking_budget}")
                        elif thinking_budget == 0:
                            print(f"   ⚠️ WARNING: Thinking tokens still used despite being disabled: {usage.thoughts_token_count}")
                        else:
                            print(f"   ✅ Thinking tokens used: {usage.thoughts_token_count}")
                    else:
                        print(f"   ✅ Thinking successfully disabled (0 thinking tokens)")
            
            if not result:
                print(f"   ❌ Gemini returned empty response")
                print(f"   🔍 Response attributes: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'No __dict__'}")
                result = ""
                finish_reason = 'error'
            else:
                print(f"   ✅ Successfully extracted {len(result)} characters")
            
            return UnifiedResponse(
                content=result,
                finish_reason=finish_reason,
                raw_response=response,
                usage=None
            )
            
        except Exception as e:
            print(f"   ❌ Gemini image processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return UnifiedResponse(
                content="",
                finish_reason='error',
                error_details={'error': str(e)}
            )
        
    def _send_openai_image(self, messages, image_base64, temperature, 
                          max_tokens, max_completion_tokens, response_name) -> UnifiedResponse:
        """
        Refactored OpenAI image handler with o-series model support
        """
        try:
            # Format messages with image
            vision_messages = []
            
            for msg in messages:
                if msg['role'] == 'user':
                    # Add image to user message
                    vision_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg['content']},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    })
                else:
                    vision_messages.append(msg)
            
            # Build API parameters
            api_params = {
                "model": self.model,
                "messages": vision_messages,
                "temperature": temperature
            }

            # Get user-configured anti-duplicate parameters
            anti_dupe_params = self._get_anti_duplicate_params(temperature)
            api_params.update(anti_dupe_params)  # Add user's custom parameters

            # Use the appropriate token parameter based on model type
            if self._is_o_series_model():
                # o-series models use max_completion_tokens
                token_limit = max_completion_tokens or max_tokens or 16384  # Higher default for vision
                api_params["max_completion_tokens"] = token_limit
                logger.info(f"Using max_completion_tokens={token_limit} for o-series vision model {self.model}")
            else:
                # Regular models use max_tokens
                if max_tokens:
                    api_params["max_tokens"] = max_tokens
                    logger.debug(f"Using max_tokens: {max_tokens} for {self.model}")
            
            logger.info(f"Calling OpenAI vision API with model: {self.model}")
            
            response = self.openai_client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                logger.debug(f"Token usage: {usage}")
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=response
            )
          
        except Exception as e:
            error_msg = str(e)
            
            # Check for specific o-series model errors
            if "max_tokens" in error_msg and "not supported" in error_msg:
                print(f"Token parameter error for {self.model}: {error_msg}")
                logger.info("Retrying without token limits...")
                
                # Build new params without token limits
                retry_params = {
                    "model": self.model,
                    "messages": vision_messages,
                    "temperature": temperature
                }

                # Get user-configured anti-duplicate parameters for retry
                anti_dupe_params = self._get_anti_duplicate_params(temperature)
                retry_params.update(anti_dupe_params)  # Add user's custom parameters

                try:
                    response = self.openai_client.chat.completions.create(**retry_params)
                    content = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    
                    return UnifiedResponse(
                        content=content,
                        finish_reason=finish_reason,
                        usage=None,
                        raw_response=response
                    )
                except Exception as retry_error:
                    print(f"Retry failed: {retry_error}")
                    raise UnifiedClientError(f"OpenAI Vision API error: {retry_error}")
            
            print(f"OpenAI Vision API error: {e}")
            raise UnifiedClientError(f"OpenAI Vision API error: {e}")
    
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
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
                raise UnifiedClientError(error_msg, http_status=resp.status_code)

            json_resp = resp.json()
            
            # Extract content
            content_parts = json_resp.get("content", [])
            if isinstance(content_parts, list):
                content = "".join(part.get("text", "") for part in content_parts)
            else:
                content = str(content_parts)
            
            finish_reason = json_resp.get("stop_reason")
            if finish_reason == "max_tokens":
                finish_reason = "length"
            
            # Don't save here - send_image() method handles saving
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Anthropic Vision API error: {e}")
            raise UnifiedClientError(f"Anthropic Vision API error: {e}")
    
    def _send_electronhub_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request through ElectronHub API
        
        ElectronHub uses OpenAI-compatible format for vision models.
        The model name has already been stripped of the eh/ prefix in __init__.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Get ElectronHub endpoint
        base_url = os.getenv("ELECTRONHUB_API_URL", "https://api.electronhub.ai/v1")
        
        # Format messages with image using OpenAI format
        vision_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                # Add image to user message
                vision_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg['content']},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            else:
                vision_messages.append(msg)
        
        # Store original model and strip prefix for API call
        original_model = self.model
        actual_model = self.model
        
        # Strip ElectronHub prefixes
        electronhub_prefixes = ['eh/', 'electronhub/', 'electron/']
        for prefix in electronhub_prefixes:
            if actual_model.startswith(prefix):
                actual_model = actual_model[len(prefix):]
                logger.info(f"ElectronHub image: Using model '{actual_model}' (stripped from '{original_model}')")
                break
        
        payload = {
            "model": actual_model,
            "messages": vision_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the request
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        for attempt in range(max_retries):
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"ElectronHub Vision API error: {response.status_code}"
                    if response.text:
                        try:
                            error_data = response.json()
                            error_msg += f" - {error_data.get('error', {}).get('message', response.text)}"
                        except:
                            error_msg += f" - {response.text}"
                    
                    if attempt < max_retries - 1:
                        print(f"{error_msg} (attempt {attempt + 1})")
                        time.sleep(api_delay)
                        continue
                    raise UnifiedClientError(error_msg)
                
                json_resp = response.json()
                choice = json_resp['choices'][0]
                content = choice['message']['content']
                finish_reason = choice.get('finish_reason', 'stop')
                
                usage = None
                if 'usage' in json_resp:
                    usage = {
                        'prompt_tokens': json_resp['usage'].get('prompt_tokens', 0),
                        'completion_tokens': json_resp['usage'].get('completion_tokens', 0),
                        'total_tokens': json_resp['usage'].get('total_tokens', 0)
                    }
                
                return UnifiedResponse(
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_response=json_resp
                )
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"ElectronHub Vision API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                print(f"ElectronHub Vision API error after all retries: {e}")
                raise UnifiedClientError(f"ElectronHub Vision API error: {e}")
    
    def _send_poe_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request using poe-api-wrapper"""
        try:
            from poe_api_wrapper import PoeApi
        except ImportError:
            raise UnifiedClientError(
                "poe-api-wrapper not installed. Run: pip install poe-api-wrapper"
            )
        
        # Parse cookies (same as _send_poe)
        tokens = {}
        if '|' in self.api_key:
            for pair in self.api_key.split('|'):
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    tokens[k.strip()] = v.strip()
        elif ':' in self.api_key:
            k, v = self.api_key.split(':', 1)
            tokens[k.strip()] = v.strip()
        else:
            tokens['p-b'] = self.api_key.strip()
        
        if 'p-lat' not in tokens:
            tokens['p-lat'] = ''
            logger.info("No p-lat cookie provided, using empty string")
        
        logger.info(f"Tokens being sent for image: p-b={len(tokens.get('p-b', ''))} chars, p-lat={len(tokens.get('p-lat', ''))} chars")
        
        try:
            # Create Poe client
            poe_client = PoeApi(tokens=tokens)
            
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
            prompt = self._messages_to_prompt(messages)
            
            # Note: poe-api-wrapper's image support varies by version
            # Some versions support file_path parameter, others need different approaches
            full_response = ""
            
            try:
                # First, try to save the base64 image temporarily
                import tempfile
                import base64
                
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(base64.b64decode(image_base64))
                    tmp_path = tmp_file.name
                
                logger.info(f"Saved temporary image to: {tmp_path}")
                
                # Try sending with file_path if supported
                try:
                    for chunk in poe_client.send_message(bot_name, prompt, file_path=tmp_path):
                        if 'response' in chunk:
                            full_response = chunk['response']
                except TypeError:
                    # If file_path not supported, try alternative method
                    print("file_path parameter not supported, trying without image")
                    # Fall back to text-only
                    for chunk in poe_client.send_message(bot_name, prompt):
                        if 'response' in chunk:
                            full_response = chunk['response']
                
                # Clean up temp file
                try:
                    import os
                    os.unlink(tmp_path)
                except:
                    pass
                    
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
            
            if "rate limit" in error_str:
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
    
    def _send_openrouter_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request through OpenRouter"""
        # OpenRouter uses OpenAI-compatible format
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Strip prefix
        model_name = self.model
        for prefix in ['or/', 'openrouter/']:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
                break
        
        # Format messages with image
        vision_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                vision_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg['content']},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            else:
                vision_messages.append(msg)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': os.getenv('OPENROUTER_REFERER', 'https://github.com/your-app'),
            'X-Title': os.getenv('OPENROUTER_APP_NAME', 'Glossarion Translation')
        }
        
        if disable_safety:
            headers['X-Safe-Mode'] = 'false'
        
        payload = {
            "model": model_name,
            "messages": vision_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"OpenRouter Vision API error: {resp.status_code} - {resp.text}")
            
            json_resp = resp.json()
            content = json_resp['choices'][0]['message']['content']
            finish_reason = json_resp['choices'][0].get('finish_reason', 'stop')
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"OpenRouter Vision API error: {e}")
            raise UnifiedClientError(f"OpenRouter Vision API error: {e}")
    
    def _send_cohere_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Cohere Aya Vision API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format prompt
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"{msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt += msg['content']
        
        # Cohere Aya Vision uses a different format
        data = {
            "model": self.model,
            "prompt": prompt,
            "image": f"data:image/jpeg;base64,{image_base64}",
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            resp = requests.post(
                "https://api.cohere.ai/v1/vision",
                headers=headers,
                json=data,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                raise UnifiedClientError(f"Cohere Vision API error: {resp.status_code} - {resp.text}")
            
            json_resp = resp.json()
            content = json_resp.get("text", "")
            
            return UnifiedResponse(
                content=content,
                finish_reason='stop',
                raw_response=json_resp
            )
            
        except Exception as e:
            print(f"Cohere Vision API error: {e}")
            raise UnifiedClientError(f"Cohere Vision API error: {e}")
            
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
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(save_summary, f, indent=2, ensure_ascii=False)
            
            # Prepare log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'provider': self.client_type,
                'context': context or 'unknown',
                'finish_reason': finish_reason,
                'attempts': attempts or 1,
                'input_length': sum(len(msg.get('content', '')) for msg in messages),
                'output_length': len(response_content) if response_content else 0,
                'truncation_type': 'silent' if finish_reason == 'length' else 'explicit',
                'content_refused': 'yes' if finish_reason == 'content_filter' else 'no',
                'last_50_chars': response_content[-50:] if response_content else '',
                'error_details': json.dumps(error_details) if error_details else '',
                'input_preview': self._get_safe_preview(messages),
                'output_preview': response_content[:200] if response_content else '',
                'output_filename': output_filename  # Add output filename to log entry
            }
            
            # Write to CSV
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
                html_content = """<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Truncation Failures Log</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 20px;
                line-height: 1.6;
            }
            .summary {
                background-color: #e3f2fd;
                border: 2px solid #1976d2;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary h2 {
                color: #1976d2;
                margin-top: 0;
            }
            .summary-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-box {
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            .stat-label {
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
            .truncated-files {
                background-color: white;
                padding: 15px;
                border-radius: 4px;
                border: 1px solid #ddd;
                max-height: 200px;
                overflow-y: auto;
            }
            .truncated-files h3 {
                margin-top: 0;
                color: #333;
            }
            .file-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .file-badge {
                background-color: #ffecb3;
                border: 1px solid #ffc107;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 13px;
                font-family: 'Courier New', monospace;
            }
            .log-entry {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .timestamp {
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
            }
            .metadata {
                display: grid;
                grid-template-columns: 200px 1fr;
                gap: 10px;
                margin-bottom: 15px;
            }
            .label {
                font-weight: bold;
                color: #333;
            }
            .value {
                color: #555;
            }
            .content-preview {
                background-color: #f8f8f8;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                white-space: pre-wrap;
                word-break: break-word;
                max-height: 200px;
                overflow-y: auto;
            }
            .error {
                color: #d9534f;
            }
            .warning {
                color: #f0ad4e;
            }
            .section-title {
                font-weight: bold;
                color: #2c5aa0;
                margin-top: 15px;
                margin-bottom: 5px;
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #2c5aa0;
                padding-bottom: 10px;
            }
            .truncation-type-silent {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
            }
            .truncation-type-explicit {
                background-color: #f8d7da;
                border-left: 4px solid #dc3545;
            }
        </style>
    </head>
    <body>
        <h1>Truncation Failures Log</h1>
        <div id="summary-container">
            <!-- Summary will be inserted here -->
        </div>
        <div id="entries-container">
            <!-- Log entries will be inserted here -->
        </div>
    """
                # Write initial HTML structure
                with open(html_log_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                # Make sure HTML is properly closed
                if not html_content.rstrip().endswith('</html>'):
                    with open(html_log_file, 'a', encoding='utf-8') as f:
                        f.write('\n</body>\n</html>')
            
            # Read existing HTML content
            with open(html_log_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Generate summary HTML
            summary_html = f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-label">Total Truncations</div>
                    <div class="stat-value">{summary_data['total_truncations']}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Affected Files</div>
                    <div class="stat-value">{len(summary_data['truncated_files'])}</div>
                </div>
            </div>
            <div class="truncated-files">
                <h3>Truncated Output Files:</h3>
                <div class="file-list">
    """
            
            # Add file badges
            for filename in sorted(summary_data['truncated_files']):
                summary_html += f'                <span class="file-badge">{html.escape(filename)}</span>\n'
            
            summary_html += """            </div>
            </div>
        </div>
    """
            
            # Update summary in HTML
            if '<div id="summary-container">' in html_content:
                # Replace existing summary
                start = html_content.find('<div id="summary-container">') + len('<div id="summary-container">')
                end = html_content.find('</div>', start) 
                html_content = html_content[:start] + '\n' + summary_html + '\n    ' + html_content[end:]
            
            # Generate new log entry HTML
            truncation_class = 'truncation-type-silent' if log_entry['truncation_type'] == 'silent' else 'truncation-type-explicit'
            
            entry_html = f"""    <div class="log-entry {truncation_class}">
            <div class="timestamp">{log_entry["timestamp"]} - Output: {html.escape(output_filename)}</div>
            <div class="metadata">
                <span class="label">Model:</span><span class="value">{html.escape(str(log_entry["model"]))}</span>
                <span class="label">Provider:</span><span class="value">{html.escape(str(log_entry["provider"]))}</span>
                <span class="label">Context:</span><span class="value">{html.escape(str(log_entry["context"]))}</span>
                <span class="label">Finish Reason:</span><span class="value {("error" if log_entry["finish_reason"] == "content_filter" else "warning")}">{html.escape(str(log_entry["finish_reason"]))}</span>
                <span class="label">Attempts:</span><span class="value">{log_entry["attempts"]}</span>
                <span class="label">Input Length:</span><span class="value">{log_entry["input_length"]:,} chars</span>
                <span class="label">Output Length:</span><span class="value">{log_entry["output_length"]:,} chars</span>
                <span class="label">Truncation Type:</span><span class="value">{html.escape(str(log_entry["truncation_type"]))}</span>
                <span class="label">Content Refused:</span><span class="value {("error" if log_entry["content_refused"] == "yes" else "")}">{html.escape(str(log_entry["content_refused"]))}</span>
    """
            
            if log_entry['error_details']:
                entry_html += f'            <span class="label">Error Details:</span><span class="value error">{html.escape(str(log_entry["error_details"]))}</span>\n'
            
            entry_html += f"""        </div>
            <div class="section-title">Input Preview</div>
            <div class="content-preview">{html.escape(str(log_entry["input_preview"]))}</div>
            <div class="section-title">Output Preview</div>
            <div class="content-preview">{html.escape(str(log_entry["output_preview"]))}</div>
    """
            
            if log_entry['last_50_chars']:
                entry_html += f"""        <div class="section-title">Last 50 Characters</div>
            <div class="content-preview">{html.escape(str(log_entry["last_50_chars"]))}</div>
    """
            
            entry_html += """    </div>
    """
            
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
