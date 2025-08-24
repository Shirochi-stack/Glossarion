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
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import logging
import re
import base64
from PIL import Image
import io
import time
import random
import csv
from datetime import datetime
import traceback
import hashlib
import html
from multi_api_key_manager import APIKeyPool, APIKeyEntry, RateLimitCache
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
    _chapter_request_tracker = {}  # Track chapter requests to prevent duplication
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
        'temperature_fixed': ['o4-mini', 'o1-mini', 'o1-preview', 'o3-mini', 'o3', 'o3-pro', 'o4-mini', 'gpt-5-mini','gpt-5','gpt-5-nano'],
        'no_system_message': ['o1', 'o1-preview', 'o3', 'o3-pro'],
        'max_completion_tokens': ['o4', 'o1', 'o3', 'gpt-5-mini','gpt-5','gpt-5-nano'],
        'chinese_optimized': ['qwen', 'yi', 'glm', 'chatglm', 'baichuan', 'ernie', 'hunyuan'],
    }
    
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
            cls._main_fallback_key = validated_keys[0]['api_key']
            cls._main_fallback_model = validated_keys[0]['model']
            print(f"🔑 Using {validated_keys[0]['model']} as main fallback key")

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
    
    def __init__(self, api_key: str, model: str, output_dir: str = None):
        """Initialize the unified client with enhanced thread safety"""
        # Store original values
        self.original_api_key = api_key
        self.original_model = model
        
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
        if not hasattr(self, '_model_lock'):
            self._model_lock = threading.RLock()
        
        # Duplicate request tracking (existing but enhanced)
        self._chapter_request_tracker = {}  # {request_hash: {timestamp, thread, context}}
        self._tracker_lock = RLock()
        
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
        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
        if retry_timeout_enabled:
            self.request_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))
        else:
            self.request_timeout = 36000  # 10 hours
        
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

                # Only force OpenAI client if:
                # 1. Custom endpoint is enabled via toggle
                # 2. We have a custom URL
                # 3. No client was set up (or it's already openai)
                if custom_base_url and use_custom_endpoint and self.openai_client is None:
                    if self.client_type is None or self.client_type == 'openai':
                        print(f"[DEBUG] Custom base URL detected and enabled, using OpenAI client for model: {self.model}")
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
                    else:
                        print(f"[DEBUG] Custom base URL set but model {self.model} uses {self.client_type}, not overriding")
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
                    tls.initialized = True
                    tls.last_rotation = time.time()
                    
                    # MICROSECOND LOCK: Only when copying to instance variables
                    if hasattr(self, '_instance_model_lock'):
                        with self._instance_model_lock:
                            # Copy to instance for compatibility
                            self.api_key = tls.api_key
                            self.model = tls.model
                            self.key_identifier = tls.key_identifier
                            self.current_key_index = key_index
                    else:
                        # No lock available, fall back
                        self.api_key = tls.api_key
                        self.model = tls.model
                        self.key_identifier = tls.key_identifier
                        self.current_key_index = key_index
                    
                    # Log key assignment
                    if len(self.api_key) > 12:
                        masked_key = self.api_key[:4] + "..." + self.api_key[-4:]
                    else:
                        masked_key = self.api_key[:3] + "..." + self.api_key[-2:] if len(self.api_key) > 5 else "***"
                    
                    print(f"[Thread-{thread_name}] 🔑 Using {self.key_identifier} - {masked_key}")
                    
                    # Setup client with new key (might need lock if it modifies instance state)
                    self._setup_client()
                    return
                else:
                    # No keys available
                    raise UnifiedClientError("No available API keys for thread", error_type="no_keys")
            else:
                # Not rotating, ensure instance variables match thread-local
                if tls.initialized:
                    # MICROSECOND LOCK: When syncing instance variables
                    if hasattr(self, '_instance_model_lock'):
                        with self._instance_model_lock:
                            self.api_key = tls.api_key
                            self.model = tls.model
                            self.key_identifier = tls.key_identifier
                            self.current_key_index = getattr(tls, 'key_index', None)
                    else:
                        self.api_key = tls.api_key
                        self.model = tls.model
                        self.key_identifier = tls.key_identifier
                        self.current_key_index = getattr(tls, 'key_index', None)
        
        # Single key mode
        elif not tls.initialized:
            tls.api_key = self.original_api_key
            tls.model = self.original_model
            tls.key_identifier = "Single Key"
            tls.initialized = True
            tls.request_count = 0
            
            # MICROSECOND LOCK: When setting instance variables
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
                    self.api_key = tls.api_key
                    self.model = tls.model
                    self.key_identifier = tls.key_identifier
            else:
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
        max_retries = 7
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
            # ← ADD THIS CHECK
            if self._cancelled:
                print(f"[Thread-{thread_name}] Wait cancelled by user")
                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
            
            # Check every second if a key became available early
            with self.__class__._pool_lock:
                for i, key in enumerate(self._api_key_pool.keys):
                    key_id = f"Key#{i+1} ({key.model})"
                    if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                        print(f"[Thread-{thread_name}] Key became available early: {key_id}")
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

    def _check_and_wait_for_duplicate(self, request_hash: str, context: str) -> Optional[Tuple[str, str]]:
        """Check for duplicate requests and wait for results if found"""
        thread_name = threading.current_thread().name
        
        # First check cache
        cached_result = self._get_cached_response(request_hash)
        if cached_result:
            print(f"[{thread_name}] Using cached response for {request_hash[:8]}")
            return cached_result
        
        # Check if another thread is processing this request
        #with self._active_requests_lock:
        #    if request_hash in self._active_requests:
                # Another thread is processing, get the event
        #        event = self._active_requests[request_hash]
        #        print(f"[{thread_name}] Waiting for another thread to complete request {request_hash[:8]}")
        #    else:
        #        # We're the first, create an event for others to wait on
        #        event = threading.Event()
        #        self._active_requests[request_hash] = event
        #        return None  # We should process this request
        
        # Wait for the other thread to complete (outside the lock)
        if event.wait(timeout=300):  # 5 minute timeout
            # Check cache again after waiting
            cached_result = self._get_cached_response(request_hash)
            if cached_result:
                print(f"[{thread_name}] Got result from other thread for {request_hash[:8]}")
                return cached_result
        
        # Timeout or no result, we'll process it ourselves
        print(f"[{thread_name}] Other thread didn't complete, processing request ourselves")
        return None

    def _get_cached_response(self, request_hash: str) -> Optional[Tuple[str, str]]:
        """Get cached response if available and not expired"""
        with self._request_cache_lock:
            if request_hash in self._request_cache:
                content, finish_reason, timestamp = self._request_cache[request_hash]
                if time.time() - timestamp < self._cache_expiry_seconds:
                    return content, finish_reason
                else:
                    # Expired, remove it
                    del self._request_cache[request_hash]
        return None

    def _cache_response(self, request_hash: str, content: str, finish_reason: str):
        """Cache a response with timestamp"""
        with self._request_cache_lock:
            self._request_cache[request_hash] = (content, finish_reason, time.time())
            
            # Cleanup old entries if cache is too large
            if len(self._request_cache) > 1000:
                # Remove oldest 100 entries
                sorted_items = sorted(
                    self._request_cache.items(),
                    key=lambda x: x[1][2]  # Sort by timestamp
                )
                for key, _ in sorted_items[:100]:
                    del self._request_cache[key]

    def _complete_request(self, request_hash: str):
        """Mark a request as complete and notify waiting threads"""
        with self._active_requests_lock:
            if request_hash in self._active_requests:
                event = self._active_requests[request_hash]
                event.set()  # Wake up waiting threads
                
                # Schedule cleanup after a delay
                threading.Timer(5.0, self._cleanup_active_request, args=[request_hash]).start()

    def _cleanup_active_request(self, request_hash: str):
        """Remove completed request from active tracking after delay"""
        with self._active_requests_lock:
            self._active_requests.pop(request_hash, None)
            logger.debug(f"Cleaned up active request {request_hash[:8]}")
        
    
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
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
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
            else:
                # No lock available, fall back (not thread-safe!)
                self.api_key = key_info[0].api_key
                self.model = key_info[0].model
                self.current_key_index = key_info[1]
                self.key_identifier = f"Key#{key_info[1]+1} ({self.model})"
                self.openai_client = None
                self.gemini_client = None
                self.mistral_client = None
                self.cohere_client = None
            
            # Logging (outside lock - just reading)
            masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else self.api_key
            print(f"[DEBUG] 🔄 Rotating from {old_key_identifier} to {self.key_identifier} - {masked_key}")
            
            # Re-setup the client with new key
            self._setup_client()
            
            # Re-apply custom endpoint if needed
            use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
            custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            
            if custom_base_url and use_custom_endpoint and self.client_type == 'openai':
                if not custom_base_url.startswith(('http://', 'https://')):
                    custom_base_url = 'https://' + custom_base_url
                
                # MICROSECOND LOCK: When modifying client instance
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=custom_base_url
                        )
                else:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=custom_base_url
                    )
                
                print(f"[DEBUG] Rotated key: Re-created OpenAI client with custom base URL")
            
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
        with self._instance_model_lock:
            self.api_key = key_info[0].api_key
            self.model = key_info[0].model
            self.current_key_index = key_info[1]
            self.key_identifier = f"Key#{key_info[1]+1} ({key_info[0].model})"
            
            # Reset clients atomically
            self.openai_client = None
            self.gemini_client = None
            self.mistral_client = None
            self.cohere_client = None
        
        # Logging OUTSIDE the lock
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:]
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
        with self._instance_model_lock:
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
        with self._instance_model_lock:
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
        with self._instance_model_lock:
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
        #with self._active_requests_lock:
        #    active_count = len(self._active_requests)
        #    if active_count > 0:
        #        logger.debug(f"Active requests: {active_count}")
        #        for hash_key in list(self._active_requests.keys())[:5]:  # Show first 5
        #            logger.debug(f"  - {hash_key[:8]}...")
        pass

    def _ensure_thread_safety_init(self):
        """
        Ensure all thread safety structures are properly initialized.
        Call this during __init__ or before parallel processing.
        """
        # Request deduplication structures
        #if not hasattr(self, '_request_cache'):
        #    self._request_cache = {}
        #if not hasattr(self, '_request_cache_lock'):
        #    self._request_cache_lock = RLock()
        #if not hasattr(self, '_cache_expiry_seconds'):
        #    self._cache_expiry_seconds = 300  # 5 minutes
        
        # Active request tracking
        #if not hasattr(self, '_active_requests'):
        #    self._active_requests = {}  # {request_hash: threading.Event}
        #if not hasattr(self, '_active_requests_lock'):
        #    self._active_requests_lock = RLock()
        
        # Thread-local storage
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
        
        # File operation locks
        if not hasattr(self, '_file_write_locks'):
            self._file_write_locks = {}
        if not hasattr(self, '_file_write_locks_lock'):
            self._file_write_locks_lock = RLock()
        
        # Legacy tracker (for backward compatibility)
        if not hasattr(self, '_chapter_request_tracker'):
            self._chapter_request_tracker = {}
        if not hasattr(self, '_tracker_lock'):
            self._tracker_lock = RLock()

    def _periodic_cache_cleanup(self):
        """
        Periodically clean up expired cache entries and active requests.
        Should be called periodically or scheduled with a timer.
        """
        current_time = time.time()
        
        # Clean up expired cache entries
        with self._request_cache_lock:
            expired_keys = [
                key for key, (_, _, timestamp) in self._request_cache.items()
                if current_time - timestamp > self._cache_expiry_seconds
            ]
            for key in expired_keys:
                del self._request_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        # Clean up stuck active requests (older than 5 minutes)
        #with self._active_requests_lock:
        #    stuck_timeout = 300  # 5 minutes
        #    stuck_requests = []
        #    
        #    for request_hash, event in list(self._active_requests.items()):
                # Note: We can't easily track creation time of events,
                # so we rely on the cleanup timer approach in the main send() method
                # This is just a safety cleanup for any that got stuck
        #        if not event.is_set():
                    # Check if any thread is waiting on this event
                    # If no waiters after timeout, remove it
                    # This is a simplified check - in production you might track timestamps
        #            pass

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
        
        # Clear any pending active requests for this client
        #with self._active_requests_lock:
            # Set all events to release waiting threads
        #    for event in self._active_requests.values():
        #        event.set()
            # Note: We don't clear the dict here as other threads might still be using it
        
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
        
        # Initialize variables at method scope for all client types
        base_url = None
        use_gemini_endpoint = False
        gemini_endpoint = ""
        
        # Prepare provider-specific settings (but don't create clients yet)
        if self.client_type == 'openai':
            #print(f"[DEBUG] Preparing OpenAI client setup")
            pass
            if openai is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
            # Check if custom endpoints are enabled
            use_custom_endpoint = os.getenv('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
            
            # Initialize base_url with default value
            base_url = 'https://api.openai.com/v1'  # Default OpenAI endpoint
            
            # Check for custom base URL
            if use_custom_endpoint:
                custom_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', ''))
                if custom_url:  # Only override if custom URL is provided
                    base_url = custom_url
                    
                    # Validate URL has protocol
                    if not base_url.startswith(('http://', 'https://')):
                        print(f"[WARNING] Custom base URL missing protocol, adding https://")
                        base_url = 'https://' + base_url
                    print(f"[DEBUG] Custom endpoints enabled, using: {base_url}")
                else:
                    print(f"[DEBUG] Custom endpoints enabled but no URL provided, using default")
            else:
                # Use default endpoint when toggle is off
                print(f"[DEBUG] Custom endpoints disabled, using default OpenAI endpoint")
            
            print(f"[DEBUG] Will use base URL: {base_url}")
            
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
            # Ensure base_url is set
            if base_url is None:
                base_url = 'https://api.openai.com/v1'
            
            # MICROSECOND LOCK for OpenAI client
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
            else:
                self.openai_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=base_url
                )
            print(f"[DEBUG] OpenAI client created with base_url: {base_url}")
        
        elif self.client_type == 'gemini':
            if use_gemini_endpoint and gemini_endpoint:
                # Use OpenAI client for Gemini endpoint
                if base_url is None:
                    base_url = gemini_endpoint
                
                # MICROSECOND LOCK for Gemini with OpenAI endpoint
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url
                        )
                        self._original_client_type = 'gemini'
                        self.client_type = 'openai'
                else:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                    self._original_client_type = 'gemini'
                    self.client_type = 'openai'
                print(f"[DEBUG] Gemini using OpenAI-compatible endpoint: {base_url}")
            else:
                # MICROSECOND LOCK for native Gemini client
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.gemini_client = genai.Client(api_key=self.api_key)
                        if hasattr(tls, 'model'):
                            tls.gemini_configured = True
                            tls.gemini_api_key = self.api_key
                            tls.gemini_client = self.gemini_client
                else:
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
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.cohere_client = cohere.Client(self.api_key)
                else:
                    self.cohere_client = cohere.Client(self.api_key)
                logger.info("Cohere client created")
        
        elif self.client_type == 'deepseek':
            if openai is not None:
                if base_url is None:
                    base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
                
                # MICROSECOND LOCK for DeepSeek client
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url
                        )
                else:
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
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url
                        )
                else:
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
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url
                        )
                else:
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
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.openai_client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=base_url
                        )
                else:
                    self.openai_client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url
                    )
                logger.info(f"xAI client configured with endpoint: {base_url}")
        
        elif self.client_type == 'vertex_model_garden':
            # Vertex AI doesn't need a client created here
            logger.info("Vertex AI Model Garden will initialize on demand")
        
        elif self.client_type in ['yi', 'qwen', 'baichuan', 'zhipu', 'moonshot', 'baidu', 
                                  'tencent', 'iflytek', 'bytedance', 'minimax', 
                                  'sensenova', 'internlm', 'tii', 'microsoft', 
                                  'azure', 'google', 'alephalpha', 'databricks', 
                                  'huggingface', 'salesforce', 'bigscience', 'meta',
                                  'electronhub', 'poe', 'openrouter']:
            # These providers will use HTTP API or OpenAI-compatible endpoints
            # No client initialization needed here
            logger.info(f"{self.client_type} will use HTTP API or compatible endpoint")
        
        # Store thread-local client reference if in multi-key mode
        if self._multi_key_mode and hasattr(tls, 'model'):
            # MICROSECOND LOCK for thread-local storage
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
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
        """Thread-safe send with proper key management and deduplication for batch translation"""
        self._apply_thread_submission_delay()
        thread_name = threading.current_thread().name
        
        # GENERATE UNIQUE REQUEST ID FOR THIS CALL
        request_id = str(uuid.uuid4())[:8]
        
        # Generate request hash WITH request ID (unique per send() call)
        request_hash = self._get_request_hash_with_request_id(messages, request_id)
        
        # Extract chapter info for better logging
        chapter_info = self._extract_chapter_info(messages)
        context_str = context or 'unknown'
        if chapter_info['chapter']:
            context_str = f"Chapter {chapter_info['chapter']}"
            if chapter_info['chunk']:
                context_str += f" Chunk {chapter_info['chunk']}/{chapter_info['total_chunks']}"
        
        # === IMPROVED DEDUPLICATION WITH PROPER SYNCHRONIZATION ===
        
        # Only check for duplicates in specific contexts
        if context in ['translation', 'glossary', 'image_translation']:
            # Get thread-local storage
            tls = self._get_thread_local_client()
            
            # Ensure thread has a cache
            if not hasattr(tls, 'request_cache'):
                tls.request_cache = {}
            
            # Step 1: Try to get from THREAD-LOCAL cache
            if request_hash in tls.request_cache:  # Changed from self._request_cache
                content, finish_reason, timestamp = tls.request_cache[request_hash]
                if time.time() - timestamp < self._cache_expiry_seconds:
                    logger.info(f"[{thread_name}] Thread-local cache HIT for {context_str}")
                    return content, finish_reason
                else:
                    # Expired, remove it
                    del tls.request_cache[request_hash]
            
            # Step 2: Check if another thread is processing this request (atomic check-and-set)
            processing_by_other = False
            event_to_wait = None
            
            
            # Step 3: If another thread is processing, wait for it
            if processing_by_other and event_to_wait:
                logger.info(f"[{thread_name}] Waiting for {context_str} to be processed by another thread...")
                
                # Wait with timeout and periodic cache checks
                wait_timeout = 60  # Maximum wait time
                check_interval = 0.5
                total_waited = 0
                
                while total_waited < wait_timeout:
                    # Check if event is set
                    if event_to_wait.wait(timeout=check_interval):
                        # Event was set, check cache
                        with self._request_cache_lock:
                            if request_hash in self._request_cache:
                                content, finish_reason, timestamp = self._request_cache[request_hash]
                                if time.time() - timestamp < self._cache_expiry_seconds:
                                    logger.info(f"[{thread_name}] Got cached result after waiting for {context_str}")
                                    return content, finish_reason
                        
                        # No cache entry found despite event being set, break and process ourselves
                        print(f"[{thread_name}] Event set but no cache entry, processing {context_str} ourselves")
                        break
                    
                    total_waited += check_interval
                    
                    # Periodic cache check even if event not set
                    if total_waited % 2 == 0:  # Check every 2 seconds
                        with self._request_cache_lock:
                            if request_hash in self._request_cache:
                                content, finish_reason, timestamp = self._request_cache[request_hash]
                                if time.time() - timestamp < self._cache_expiry_seconds:
                                    logger.info(f"[{thread_name}] Found cached result while waiting for {context_str}")
                                    return content, finish_reason
                
                if total_waited >= wait_timeout:
                    print(f"[{thread_name}] Timeout waiting for {context_str}, processing ourselves")
        
        # === END OF IMPROVED DEDUPLICATION ===
        
        # Call debug if enabled via environment variable
        if os.getenv("DEBUG_PARALLEL_REQUESTS", "0") == "1":
            self._debug_active_requests()
        
        # Ensure thread has a client
        self._ensure_thread_client()
        
        # Log the processing
        logger.info(f"[{thread_name}] Processing {context_str} with {self.key_identifier}")
        
        max_retries = 7
        retry_count = 0
        last_error = None
        retry_reason = None
        successful_response = None
        
        # Track current attempt for payload
        self._current_retry_attempt = 0
        self._max_retries = max_retries
        
        # Track which keys we've already tried
        attempted_keys = set()
        
        # Flag to track if we should try main key for prohibited content
        should_try_main_key = False
        
        # Define content filter indicators
        content_filter_indicators = [
            "content_filter", "content was blocked", "response was blocked",
            "safety filter", "content policy", "harmful content",
            "blocked by safety", "harm_category", "content_policy_violation",
            "unsafe content", "violates our usage policies",
            "prohibited_content", "blockedreason", "content blocked"
        ]
        
        try:
            while retry_count < max_retries:
                try:
                    # Update current attempt
                    self._current_retry_attempt = retry_count
                    
                    # Track current key
                    attempted_keys.add(self.key_identifier)
                    
                    # Call the actual implementation with retry reason
                    result = self._send_internal(messages, temperature, max_tokens, 
                                               max_completion_tokens, context, retry_reason=retry_reason,request_id=request_id)
                    
                    # Mark success
                    if self._multi_key_mode:
                        tls = self._get_thread_local_client()
                        if tls.key_index is not None:
                            self._api_key_pool.mark_key_success(tls.key_index)
                    self.reset_cleanup_state()
                    
                    logger.info(f"[{thread_name}] ✓ Request completed with {self.key_identifier}")
                    successful_response = result
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    print(f"[{thread_name}] ✗ {self.key_identifier} error: {error_str[:100]}")
                    
                    # Check for prohibited content FIRST
                    if any(indicator in error_str.lower() for indicator in content_filter_indicators):
                        retry_reason = "prohibited_content"
                        print(f"[Thread-{thread_name}] Prohibited content detected on {self.key_identifier}")
                        
                        # Try main key if conditions are met
                        if (self._multi_key_mode and 
                            hasattr(self, 'original_api_key') and 
                            hasattr(self, 'original_model') and
                            self.original_api_key and 
                            self.original_model and
                            not should_try_main_key):
                            
                            print(f"[Thread-{thread_name}] Will retry with main key for prohibited content")
                            should_try_main_key = True
                            retry_count += 1
                            continue
                        else:
                            print(f"[Thread-{thread_name}] Prohibited content - cannot retry")
                            raise
                    
                    # Check for rate limit
                    elif "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                        retry_reason = "rate_limit"
                        if self._multi_key_mode:
                            print(f"[Thread-{thread_name}] Rate limit hit on {self.key_identifier}")
                            
                            # Handle rate limit for this thread
                            self._handle_rate_limit_for_thread()
                            
                            # Check if we have any available keys
                            available_count = self._count_available_keys()
                            if available_count == 0:
                                print(f"[{thread_name}] All API keys are cooling down")
                                
                                # Get the shortest cooldown time from all keys
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
                                
                                # DON'T increment retry_count - we want to retry indefinitely
                                # Just continue the loop without counting this as a retry
                                continue
                                # REMOVED the else clause here - it would never execute
                            
                            # Check if we've tried too many keys
                            if len(attempted_keys) >= len(self._api_key_pool.keys):
                                print(f"[{thread_name}] Attempted all {len(self._api_key_pool.keys)} keys")
                                # Instead of raising error, wait for cooldown
                                cooldown_time = self._get_shortest_cooldown_time()
                                print(f"⏳ [Thread-{thread_name}] All keys attempted, waiting {cooldown_time}s for cooldown...")
                                
                                for i in range(cooldown_time):
                                    if self._cancelled:
                                        raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                    time.sleep(1)
                                
                                # Clear attempted keys and try again
                                attempted_keys.clear()
                                tls = self._get_thread_local_client()
                                tls.initialized = False
                                self._ensure_thread_client()
                                continue
                            
                            #retry_count += 1
                            #logger.info(f"[{thread_name}] Retrying with new key, attempt {retry_count}/{max_retries}")
                            continue
                        else:
                            # Single key mode - wait and retry
                            wait_time = min(60 * (retry_count + 1), 120)
                            print(f"[{thread_name}] Rate limit, waiting {wait_time}s")
                            
                            for i in range(wait_time):
                                if self._cancelled:
                                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                time.sleep(1)
                            
                            #retry_count += 1
                            continue
                    
                    # Check for cancellation
                    elif isinstance(e, UnifiedClientError) and e.error_type in ["cancelled", "timeout"]:
                        retry_reason = f"cancelled_{e.error_type}"
                        raise
                    
                    # Other errors - retry logic
                    elif retry_count < max_retries - 1:
                        # Determine retry reason
                        if "timeout" in error_str.lower():
                            retry_reason = "timeout_error"
                        elif "connection" in error_str.lower():
                            retry_reason = "connection_error"
                        elif "500" in error_str or "502" in error_str or "503" in error_str:
                            retry_reason = f"server_error_{error_str[:3]}"
                        else:
                            retry_reason = f"error_{type(e).__name__}"[:30]
                        
                        if self._multi_key_mode:
                            tls = self._get_thread_local_client()
                            if tls.key_index is not None:
                                self._api_key_pool.mark_key_error(tls.key_index)
                            
                            if not self._force_rotation:
                                # Error-based rotation
                                logger.info(f"[{thread_name}] Error occurred ({retry_reason}), rotating to new key...")
                                tls.initialized = False
                                tls.request_count = 0
                                self._ensure_thread_client()
                                retry_count += 1
                                print(f"[{thread_name}] Rotated to {self.key_identifier} after error")
                                continue
                        
                        # Retry with same key
                        print(f"[{thread_name}] Retrying after {retry_reason} (attempt {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(2)
                        continue
                    
                    # Can't retry - final error
                    else:
                        retry_reason = f"final_error_{type(e).__name__}"
                        print(f"[{thread_name}] Cannot retry request: {retry_reason}")
                        raise
            
            # === IMPROVED CACHE UPDATE AND CLEANUP ===
            if successful_response and context in ['translation', 'glossary', 'image_translation']:
                content, finish_reason = successful_response
                
                
                logger.info(f"[{thread_name}] Cached successful response for {context_str}")
                
            
            if successful_response:
                return successful_response
            
            # Exhausted retries
            if last_error:
                print(f"[{thread_name}] Exhausted {max_retries} retries, last reason: {retry_reason}")
                
                raise last_error
            else:
                raise Exception(f"Failed after {max_retries} attempts")
                
        except Exception as e:
            raise

    def _send_internal(self, messages, temperature=None, max_tokens=None, 
                       max_completion_tokens=None, context=None, retry_reason=None,
                       request_id=None) -> Tuple[str, Optional[str]]:  # ADD request_id parameter
        """
        Internal send implementation with integrated 500 error retry logic and prohibited content handling
        """
        start_time = time.time()
        
        # Generate request hash WITH request ID if provided
        if request_id:
            request_hash = self._get_request_hash_with_request_id(messages, request_id)
        else:
            # Fallback for direct calls (shouldn't happen in normal flow)
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
        
        # Internal retry logic for 500 errors - INCREASED TO 7
        internal_retries = 7
        base_delay = 5  # Base delay for exponential backoff
        
        # Track if we've tried main key for prohibited content
        main_key_attempted = False
        
        # Define content filter indicators locally for consistency
        content_filter_indicators = [
            "content_filter", "content was blocked", "response was blocked",
            "safety filter", "content policy", "harmful content",
            "blocked by safety", "harm_category", "content_policy_violation",
            "unsafe content", "violates our usage policies",
            "prohibited_content", "blockedreason", "content blocked"
        ]
        
        for attempt in range(internal_retries):
            try:
                # Validate request
                valid, error_msg = self._validate_request(messages, max_tokens)
                if not valid:
                    raise UnifiedClientError(f"Invalid request: {error_msg}", error_type="validation")
                
                os.makedirs("Payloads", exist_ok=True)
                
                # Apply reinforcement
                messages = self._apply_pure_reinforcement(messages)
                
                # Get file names - now unique per request AND attempt
                payload_name, response_name = self._get_file_names(messages, context=self.context)
                
                # Add request ID and attempt to filename for complete isolation
                base_payload, ext_payload = os.path.splitext(payload_name)
                base_response, ext_response = os.path.splitext(response_name)
                
                # Include request_id and attempt in filename
                unique_suffix = f"_{request_id}_A{attempt}"
                payload_name = f"{base_payload}{unique_suffix}{ext_payload}"
                response_name = f"{base_response}{unique_suffix}{ext_response}"
                
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
                
                # ====== UNIVERSAL EXTRACTION INTEGRATION ======
                # Use universal extraction instead of assuming response.content exists
                extracted_content = ""
                finish_reason = 'stop'

                if response:
                    # Prepare provider-specific parameters
                    extraction_kwargs = {}
                    
                    # Add Gemini-specific parameters if applicable
                    if self.client_type == 'gemini':
                        # Check if this model supports thinking
                        extraction_kwargs['supports_thinking'] = self._supports_thinking()
                        # Get thinking budget from environment
                        extraction_kwargs['thinking_budget'] = int(os.getenv("THINKING_BUDGET", "-1"))
                    
                    # Try universal extraction with provider-specific parameters
                    extracted_content, finish_reason = self._extract_response_text(
                        response, 
                        provider=self.client_type,
                        **extraction_kwargs
                    )
                    
                    # If extraction failed but we have a response object
                    if not extracted_content and response:
                        print(f"⚠️ Failed to extract text from {self.client_type} response")
                        print(f"   Response type: {type(response)}")
                        
                        # Provider-specific guidance
                        if self.client_type == 'gemini':
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
                    logger.info(f"Saving response for {context} ({len(extracted_content)} chars)")
                    self._save_response(extracted_content, response_name)
                    logger.info(f"Response saved successfully for {context}")
                else:
                    print(f"No content to save for {context}")

                # Handle empty responses
                if not extracted_content or extracted_content.strip() in ["", "[]", "[IMAGE TRANSLATION FAILED]"]:
                    print(f"Empty or error response: {finish_reason}")
                    
                    # Check if this is likely a safety filter issue (for ALL providers, not just Gemini)
                    is_likely_safety_filter = False
                    
                    # Pattern 1: Empty content with suspicious finish_reasons (applies to all providers)
                    if not extracted_content and finish_reason in ['length', 'stop', 'max_tokens', None]:
                        print(f"⚠️ Suspicious empty response from {self.client_type} (finish_reason={finish_reason}) - possible safety filter")
                        is_likely_safety_filter = True
                    
                    # Pattern 2: Check for safety-related content in raw response
                    if response:
                        response_str = ""
                        if hasattr(response, 'raw_response'):
                            response_str = str(response.raw_response).lower()
                        elif hasattr(response, 'error_details'):
                            response_str = str(response.error_details).lower()
                        else:
                            response_str = str(response).lower()
                        
                        safety_indicators = [
                            'safety', 'blocked', 'prohibited', 'harmful', 'inappropriate',
                            'refused', 'content_filter', 'content_policy', 'violation',
                            'cannot assist', 'unable to process', 'against guidelines',
                            'ethical', 'responsible ai', 'harm_category'
                        ]
                        
                        if any(indicator in response_str for indicator in safety_indicators):
                            print(f"❌ Safety indicators found in response from {self.client_type}")
                            is_likely_safety_filter = True
                    
                    # Pattern 3: Check for specific safety filter messages in extracted content
                    if extracted_content:
                        content_lower = extracted_content.lower()
                        safety_phrases = [
                            'blocked', 'safety', 'cannot', 'unable', 'prohibited',
                            'content filter', 'refused', 'inappropriate', 'i cannot',
                            "i can't", "i'm not able", "not able to", "against my",
                            'content policy', 'guidelines', 'ethical'
                        ]
                        if any(phrase in content_lower for phrase in safety_phrases):
                            print(f"❌ Safety filter phrases detected in content from {self.client_type}")
                            is_likely_safety_filter = True
                    
                    # Pattern 4: Provider-specific patterns
                    actual_provider = self._get_actual_provider()

                    if actual_provider in ['openai', 'azure', 'electronhub', 'openrouter', 'poe', 'gemini']:
                        # These providers often return empty on safety issues
                        if not extracted_content and finish_reason != 'error':
                            print(f"⚠️ {actual_provider} returned empty content - likely safety filter")
                            is_likely_safety_filter = True
                    
                    # Pattern 5: Check content length vs input length
                    if extracted_content and len(extracted_content) < 50:
                        input_length = sum(len(msg.get('content', '')) for msg in messages if msg.get('role') == 'user')
                        if input_length > 200:  # Substantial input but tiny output
                            print(f"⚠️ Suspiciously short response ({len(extracted_content)} chars) for {input_length} char input")
                            if any(word in extracted_content.lower() for word in ['cannot', 'unable', 'sorry', 'assist']):
                                is_likely_safety_filter = True
                    
                    # If it's likely a safety filter and we haven't tried main key yet
                    if is_likely_safety_filter and not main_key_attempted:
                        # Only try main key if conditions are met
                        if (self._multi_key_mode and 
                            hasattr(self, 'original_api_key') and 
                            hasattr(self, 'original_model') and
                            self.original_api_key and 
                            self.original_model):
                            
                            print(f"🔄 Empty/blocked response likely due to safety filter - attempting main key fallback")
                            print(f"   Provider: {self.client_type}")
                            print(f"   Current model: {self.model} ({self.key_identifier})")
                            print(f"   Main key model: {self.original_model}")
                            
                            main_key_attempted = True
                            
                            try:
                                # Create temporary client with main key
                                main_response = self._retry_with_main_key(
                                    messages, temperature, max_tokens, max_completion_tokens, context, request_id=request_id
                                )
                                
                                if main_response:
                                    content, finish_reason = main_response
                                    if content and content.strip() and len(content) > 10:  # Make sure we got actual content
                                        print(f"✅ Main key succeeded! Got {len(content)} chars")
                                        return content, finish_reason
                                    else:
                                        print(f"❌ Main key also returned empty/minimal content: {len(content) if content else 0} chars")
                                else:
                                    print(f"❌ Main key returned None")
                                    
                            except Exception as main_error:
                                print(f"❌ Main key error: {str(main_error)[:200]}")
                                # Check if main key also hit content filter
                                main_error_str = str(main_error).lower()
                                if any(indicator in main_error_str for indicator in content_filter_indicators):
                                    print(f"❌ Main key also hit content filter")
                                # Continue to normal error handling
                        else:
                            if not self._multi_key_mode:
                                print(f"❌ Not in multi-key mode, cannot retry with main key")
                            else:
                                print(f"❌ Main key not available for retry (check configuration)")
                    elif main_key_attempted:
                        print(f"❌ Already attempted main key for this request")
                    
                    # If we couldn't retry or retry failed, continue with normal error handling
                    self._save_failed_request(messages, f"Empty response from {self.client_type} (possible safety filter)", context, response)
                    
                    # Log the failure
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=extracted_content or "",
                        finish_reason='content_filter' if is_likely_safety_filter else (finish_reason or 'error'),
                        context=context,
                        error_details={
                            'likely_safety_filter': is_likely_safety_filter,
                            'original_finish_reason': finish_reason,
                            'provider': self.client_type,
                            'model': self.model,
                            'key_identifier': self.key_identifier
                        } if is_likely_safety_filter else getattr(response, 'error_details', None)
                    )
                    
                    self._track_stats(context, False, "empty_response", time.time() - start_time)
                    
                    # Use fallback with appropriate reason
                    if is_likely_safety_filter:
                        fallback_reason = f"safety_filter_{self.client_type}"
                    else:
                        fallback_reason = "empty"
                    
                    fallback_content = self._handle_empty_result(messages, context, 
                        getattr(response, 'error_details', fallback_reason) if response else fallback_reason)
                    
                    # Return with appropriate finish_reason
                    return fallback_content, 'content_filter' if is_likely_safety_filter else 'error'
                                
                # Track success
                self._track_stats(context, True, None, time.time() - start_time)
                
                # Mark key as successful in multi-key mode
                self._mark_key_success()
                
                # Log important info for retry mechanisms
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
                return extracted_content, finish_reason
                
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
                if e.error_type == "rate_limit" or "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic in outer send() method
                
                # Check for prohibited content - check BOTH error_type AND error string
                if e.error_type == "prohibited_content" or any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Prohibited content detected: {error_str[:200]}")
                    
                    # Only try main key if conditions are met
                    if (self._multi_key_mode and 
                        not main_key_attempted and 
                        hasattr(self, 'original_api_key') and 
                        hasattr(self, 'original_model') and
                        self.original_api_key and 
                        self.original_model):
                        
                        print(f"🔄 Attempting main key fallback for prohibited content")
                        print(f"   Current key: {self.key_identifier}")
                        print(f"   Main key model: {self.original_model}")
                        
                        main_key_attempted = True
                        
                        try:
                            # Create temporary client with main key
                            main_response = self._retry_with_main_key(
                                messages, temperature, max_tokens, max_completion_tokens, context
                            )
                            
                            if main_response:
                                content, finish_reason = main_response
                                print(f"✅ Main key succeeded! Returning response")
                                return content, finish_reason
                            else:
                                print(f"❌ Main key returned None")
                                
                        except Exception as main_error:
                            print(f"❌ Main key error: {str(main_error)[:200]}")
                            # Check if main key also hit content filter
                            main_error_str = str(main_error).lower()
                            if any(indicator in main_error_str for indicator in content_filter_indicators):
                                print(f"❌ Main key also hit content filter")
                            # Continue to normal error handling
                    
                    # Normal prohibited content handling
                    print(f"❌ Content prohibited - not retrying further")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors - retry these with exponential backoff
                http_status = getattr(e, 'http_status', None)
                if http_status == 500 or "500" in error_str or "api_error" in error_str:
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        # Cap the maximum delay to prevent extremely long waits
                        delay = min(delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Server error (500) - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)  # Check every 0.5 seconds
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
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Content prohibited in unexpected error: {error_str[:200]}")
                    
                    # Debug current state
                    #self._debug_multi_key_state()
                    
                    # If we're in multi-key mode and haven't tried the main key yet
                    if (self._multi_key_mode and 
                        not main_key_attempted and 
                        hasattr(self, 'original_api_key') and 
                        hasattr(self, 'original_model') and
                        self.original_api_key and 
                        self.original_model):
                        
                        print(f"🔄 Attempting main key fallback for prohibited content (from unexpected error)")
                        print(f"   Current key: {self.key_identifier}")
                        print(f"   Main key model: {self.original_model}")
                        
                        main_key_attempted = True
                        
                        try:
                            # Create temporary client with main key
                            main_response = self._retry_with_main_key(
                                messages, temperature, max_tokens, max_completion_tokens, context
                            )
                            
                            if main_response:
                                content, finish_reason = main_response
                                print(f"✅ Main key succeeded! Returning response")
                                return content, finish_reason
                            else:
                                print(f"❌ Main key returned None")
                                
                        except Exception as main_error:
                            print(f"❌ Main key error: {str(main_error)[:200]}")
                            # Check if main key also hit content filter
                            main_error_str = str(main_error).lower()
                            if any(indicator in main_error_str for indicator in content_filter_indicators):
                                print(f"❌ Main key also hit content filter")
                    
                    # Fall through to normal error handling
                    print(f"❌ Content prohibited - not retrying")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors in unexpected exceptions
                if "500" in error_str or "internal server error" in error_str:
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        delay = min(delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Server error (500) - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue
                
                # Check for other transient errors with exponential backoff
                transient_errors = ["502", "503", "504", "connection reset", "connection aborted"]
                if any(err in error_str for err in transient_errors):
                    if attempt < internal_retries - 1:
                        # Use a slightly less aggressive backoff for transient errors
                        delay = (base_delay/2 * (2 ** attempt)) + random.uniform(0, 1)
                        delay = min(delay, 30)  # Max 30 seconds for transient errors
                        
                        print(f"🔄 Transient error - retrying in {delay:.1f}s")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue
                
                # Save failed request and return fallback for other errors
                self._save_failed_request(messages, e, context)
                self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                fallback_content = self._handle_empty_result(messages, context, str(e))
                return fallback_content, 'error'

                    
    def _retry_with_main_key(self, messages, temperature=None, max_tokens=None, 
                            max_completion_tokens=None, context=None,
                            request_id=None) -> Optional[Tuple[str, Optional[str]]]: 
        """
        Create a temporary client with the main key and retry the request.
        This is used for prohibited content errors in multi-key mode.
        """
        # Use the class-level fallback key if available, otherwise use original
        fallback_key = getattr(self.__class__, '_main_fallback_key', self.original_api_key)
        fallback_model = getattr(self.__class__, '_main_fallback_model', self.original_model)
        
        # Don't retry with the same key that just failed
        if fallback_model == self.model and fallback_key == self.api_key:
            print(f"[MAIN KEY RETRY] Fallback is same as current key ({self.model}), skipping retry")
            return None
        
        print(f"[MAIN KEY RETRY] Starting retry with main key")
        print(f"[MAIN KEY RETRY] Current failing model: {self.model}")
        print(f"[MAIN KEY RETRY] Fallback model if main key retry fails: {fallback_model}")
        
        try:
            # Create a new temporary UnifiedClient instance with the fallback key
            temp_client = UnifiedClient(
                api_key=fallback_key,  # Use fallback instead of original
                model=fallback_model,   # Use fallback instead of original
                output_dir=self.output_dir
            )
            
            # FORCE single-key mode after initialization
            temp_client._multi_key_mode = False
            temp_client.use_multi_keys = False
            temp_client.key_identifier = "Main Key (Fallback)"
            
            # The client should already be set up from __init__, but verify
            if not hasattr(temp_client, 'client_type') or temp_client.client_type is None:
                # Force setup if needed
                temp_client.api_key = self.original_api_key
                temp_client.model = self.original_model
                temp_client._setup_client()
            
            # Copy relevant state BUT NOT THE CANCELLATION FLAG
            temp_client.context = context
            # DON'T COPY THE CANCELLED FLAG - This is the bug!
            # temp_client._cancelled = self._cancelled  # REMOVE THIS LINE
            temp_client._cancelled = False  # ALWAYS start fresh for main key retry
            temp_client._in_cleanup = False  # Reset cleanup state too
            temp_client.current_session_context = self.current_session_context
            temp_client.conversation_message_count = self.conversation_message_count
            temp_client.request_timeout = self.request_timeout  # Copy timeout settings
            
            print(f"[MAIN KEY RETRY] Created temp client with model: {temp_client.model}")
            print(f"[MAIN KEY RETRY] Temp client type: {getattr(temp_client, 'client_type', 'NOT SET')}")
            print(f"[MAIN KEY RETRY] Multi-key mode: {temp_client._multi_key_mode}")
            print(f"[MAIN KEY RETRY] Cancelled flag: {temp_client._cancelled}")  # Debug log
            
            # Get file names for response tracking
            payload_name, response_name = self._get_file_names(messages, context=context)
            
            # Try to send the request using _send_internal instead of send
            # This avoids the outer retry loop and goes directly to the implementation
            print(f"[MAIN KEY RETRY] Sending request...")
            
            # Use _send_internal directly to avoid nested retry loops
            result = temp_client._send_internal(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                context=context,
                retry_reason="main_key_fallback",
                request_id=request_id
            )
            
            # Check the result
            if result and isinstance(result, tuple):
                content, finish_reason = result
                if content:
                    print(f"[MAIN KEY RETRY] Success! Got content of length: {len(content)}")
                    # Save the response using our instance's method
                    self._save_response(content, response_name)
                    return content, finish_reason
                else:
                    print(f"[MAIN KEY RETRY] Empty content returned")
                    return None
            else:
                print(f"[MAIN KEY RETRY] Unexpected result type: {type(result)}")
                return None
            
        except UnifiedClientError as e:
            # Check if it's a cancellation from the temp client
            if e.error_type == "cancelled":
                print(f"[MAIN KEY RETRY] Operation was cancelled during main key retry")
                # Don't propagate cancellation - just return None
                return None
            
            print(f"[MAIN KEY RETRY] UnifiedClientError: {type(e).__name__}: {str(e)[:500]}")
            
            # Check if it's also a content filter error
            error_str = str(e).lower()
            content_filter_indicators = [
                "content_filter", "content was blocked", "response was blocked",
                "safety filter", "content policy", "harmful content",
                "blocked by safety", "harm_category", "content_policy_violation",
                "unsafe content", "violates our usage policies",
                "prohibited_content", "blockedreason", "content blocked"
            ]
            
            if any(indicator in error_str for indicator in content_filter_indicators):
                print(f"[MAIN KEY RETRY] Main key also hit content filter")
            
            # Re-raise other errors so the calling method can handle them
            raise
            
        except Exception as e:
            print(f"[MAIN KEY RETRY] Exception: {type(e).__name__}: {str(e)[:500]}")
            
            # Check if it's also a content filter error
            error_str = str(e).lower()
            content_filter_indicators = [
                "content_filter", "content was blocked", "response was blocked",
                "safety filter", "content policy", "harmful content",
                "blocked by safety", "harm_category", "content_policy_violation",
                "unsafe content", "violates our usage policies",
                "prohibited_content", "blockedreason", "content blocked"
            ]
            
            if any(indicator in error_str for indicator in content_filter_indicators):
                print(f"[MAIN KEY RETRY] Main key also hit content filter")
            
            # Re-raise so the calling method can handle it
            raise
    
    
    # Image handling methods
    def send_image(self, messages: List[Dict[str, Any]], image_data: Any,
                  temperature: Optional[float] = None, 
                  max_tokens: Optional[int] = None,
                  max_completion_tokens: Optional[int] = None,
                  context: str = 'image_translation') -> Tuple[str, str]:
        """Thread-safe image send with proper key management and deduplication for batch translation"""
        self._apply_thread_submission_delay()
        thread_name = threading.current_thread().name
        
        # GENERATE UNIQUE REQUEST ID FOR THIS CALL
        request_id = str(uuid.uuid4())[:8]
        
        # Generate request hash for duplicate detection with better image hashing
        image_size = len(image_data) if isinstance(image_data, (bytes, str)) else 0
        
        # Create a more unique hash by including actual image data hash
        if image_data:
            if isinstance(image_data, bytes):
                # Hash the actual bytes
                image_hash = hashlib.md5(image_data).hexdigest()[:12]
            else:
                # Hash string representation (base64)
                image_hash = hashlib.md5(str(image_data).encode()).hexdigest()[:12]
        else:
            image_hash = "empty"
        
        # Include image hash AND request ID in request hash for better uniqueness
        messages_hash = self._get_request_hash_with_request_id(messages, request_id)  # USE REQUEST ID
        request_hash = f"{messages_hash}_img_{image_size}_{image_hash}"
        
        # Extract any chapter/context info from messages
        chapter_info = self._extract_chapter_info(messages)
        context_str = context or 'image_translation'
        
        # Try to get image filename if available
        if chapter_info['chapter']:
            context_str = f"Image Chapter {chapter_info['chapter']}"
        else:
            # Try to extract filename from messages
            messages_str = str(messages)
            if 'image' in messages_str.lower():
                import re
                filename_match = re.search(r'(\w+\.(png|jpg|jpeg|gif|bmp|webp))', messages_str, re.IGNORECASE)
                if filename_match:
                    context_str = f"Image: {filename_match.group(1)}"
        
        # === IMPROVED IMAGE DEDUPLICATION WITH PROPER SYNCHRONIZATION ===
        
        # Only check for duplicates in specific contexts
        if context in ['translation', 'glossary', 'image_translation']:
            # Get thread-local storage
            tls = self._get_thread_local_client()
            
            # Ensure thread has a cache
            if not hasattr(tls, 'request_cache'):
                tls.request_cache = {}
            
            # Step 1: Try to get from THREAD-LOCAL cache
            if request_hash in tls.request_cache:  # Changed from self._request_cache
                content, finish_reason, timestamp = tls.request_cache[request_hash]
                if time.time() - timestamp < self._cache_expiry_seconds:
                    logger.info(f"[{thread_name}] Thread-local cache HIT for {context_str}")
                    return content, finish_reason
                else:
                    # Expired, remove it
                    del tls.request_cache[request_hash]
            
            # Step 2: Check if another thread is processing this image (atomic check-and-set)
            processing_by_other = False
            event_to_wait = None
            
            # Step 3: If another thread is processing, wait for it
            if processing_by_other and event_to_wait:
                logger.info(f"[{thread_name}] Waiting for {context_str} to be processed by another thread...")
                
                # Wait with timeout and periodic cache checks
                wait_timeout = 60  # Maximum wait time for images
                check_interval = 0.5
                total_waited = 0
                
                while total_waited < wait_timeout:
                    # Check if event is set
                    if event_to_wait.wait(timeout=check_interval):
                        # Event was set, check cache
                        with self._request_cache_lock:
                            if request_hash in self._request_cache:
                                content, finish_reason, timestamp = self._request_cache[request_hash]
                                if time.time() - timestamp < self._cache_expiry_seconds:
                                    logger.info(f"[{thread_name}] Got cached image result after waiting")
                                    return content, finish_reason
                        
                        # No cache entry found despite event being set
                        print(f"[{thread_name}] Event set but no cache entry for image, processing ourselves")
                        break
                    
                    total_waited += check_interval
                    
                    # Periodic cache check
                    if total_waited % 2 == 0:  # Check every 2 seconds
                        with self._request_cache_lock:
                            if request_hash in self._request_cache:
                                content, finish_reason, timestamp = self._request_cache[request_hash]
                                if time.time() - timestamp < self._cache_expiry_seconds:
                                    logger.info(f"[{thread_name}] Found cached image result while waiting")
                                    return content, finish_reason
                
                if total_waited >= wait_timeout:
                    print(f"[{thread_name}] Timeout waiting for image {context_str}, processing ourselves")
        
        # === END OF IMPROVED IMAGE DEDUPLICATION ===
        
        # Call debug if enabled
        if os.getenv("DEBUG_PARALLEL_REQUESTS", "0") == "1":
            self._debug_active_requests()
        
        # Ensure thread has a client
        self._ensure_thread_client()
        
        logger.info(f"[{thread_name}] Processing {context_str} with {self.key_identifier}")

        max_retries = 7
        retry_count = 0
        last_error = None
        retry_reason = None
        successful_response = None
        
        # Track current attempt for payload
        self._current_retry_attempt = 0
        self._max_retries = max_retries
        
        # Track which keys we've already tried to avoid infinite loops
        attempted_keys = set()
        
        # Flag to track if we should try main key for prohibited content
        should_try_main_key = False
        
        # Define content filter indicators at method level
        content_filter_indicators = [
            "content_filter", "content was blocked", "response was blocked",
            "safety filter", "content policy", "harmful content",
            "blocked by safety", "harm_category", "content_policy_violation",
            "unsafe content", "violates our usage policies",
            "prohibited_content", "blockedreason", "content blocked",
            "inappropriate image", "inappropriate content"
        ]
        
        try:
            while retry_count < max_retries:
                try:
                    # Update current attempt
                    self._current_retry_attempt = retry_count
                    
                    # Track current key
                    attempted_keys.add(self.key_identifier)
                    
                    # Call the actual implementation with retry reason
                    result = self._send_image_internal(messages, image_data, temperature,
                                                     max_tokens, max_completion_tokens, context,
                                                     retry_reason=retry_reason, request_id=request_id)
                    
                    # Mark success
                    if self._multi_key_mode:
                        tls = self._get_thread_local_client()
                        if tls.key_index is not None:
                            self._api_key_pool.mark_key_success(tls.key_index)
                    self.reset_cleanup_state()
                    
                    logger.info(f"[{thread_name}] ✓ Image request completed with {self.key_identifier}")
                    successful_response = result
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    print(f"[{thread_name}] ✗ {self.key_identifier} image error: {error_str[:100]}")
                    
                    # Check for prohibited content FIRST (before rate limit check)
                    if any(indicator in error_str.lower() for indicator in content_filter_indicators):
                        retry_reason = "prohibited_image_content"
                        print(f"[Thread-{thread_name}] Prohibited image content detected on {self.key_identifier}")
                        
                        # If we're in multi-key mode and haven't tried main key yet
                        if (self._multi_key_mode and 
                            hasattr(self, 'original_api_key') and 
                            hasattr(self, 'original_model') and
                            self.original_api_key and 
                            self.original_model and
                            not should_try_main_key):
                            
                            print(f"[Thread-{thread_name}] Will retry with main key for prohibited image content")
                            should_try_main_key = True
                            
                            # Try with main key
                            try:
                                main_response = self._retry_image_with_main_key(
                                    messages, image_data, temperature, max_tokens, max_completion_tokens, context
                                )
                                
                                if main_response:
                                    content, finish_reason = main_response
                                    print(f"✅ Main key succeeded for image! Returning response")
                                    successful_response = main_response
                                    break  # Exit retry loop with success
                                else:
                                    print(f"❌ Main key returned None for image")
                                    
                            except Exception as main_error:
                                print(f"❌ Main key image error: {str(main_error)[:200]}")
                                # Check if main key also hit content filter
                                main_error_str = str(main_error).lower()
                                if any(indicator in main_error_str for indicator in content_filter_indicators):
                                    print(f"❌ Main key also hit content filter for image")
                            
                            # Don't count this as a retry, just continue
                            retry_count += 1
                            continue
                        else:
                            # Either not in multi-key mode, or already tried main key
                            print(f"[Thread-{thread_name}] Prohibited image content - cannot retry")
                            raise
                    
                    # Check for rate limit
                    elif "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                        retry_reason = "rate_limit"
                        if self._multi_key_mode:
                            print(f"[Thread-{thread_name}] Rate limit hit on {self.key_identifier}")
                            
                            # Handle rate limit for this thread
                            self._handle_rate_limit_for_thread()
                            
                            # Check if we have any available keys
                            available_count = self._count_available_keys()
                            if available_count == 0:
                                print(f"[{thread_name}] All API keys are cooling down")
                                
                                # Get the shortest cooldown time from all keys
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
                                
                                # DON'T increment retry_count - we want to retry indefinitely
                                # Just continue the loop without counting this as a retry
                                continue
                                # REMOVED the else clause here - it would never execute
                            
                            # Check if we've tried too many keys
                            if len(attempted_keys) >= len(self._api_key_pool.keys):
                                print(f"[{thread_name}] Attempted all {len(self._api_key_pool.keys)} keys")
                                # Instead of raising error, wait for cooldown
                                cooldown_time = self._get_shortest_cooldown_time()
                                print(f"⏳ [Thread-{thread_name}] All keys attempted, waiting {cooldown_time}s for cooldown...")
                                
                                for i in range(cooldown_time):
                                    if self._cancelled:
                                        raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                    time.sleep(1)
                                
                                # Clear attempted keys and try again
                                attempted_keys.clear()
                                tls = self._get_thread_local_client()
                                tls.initialized = False
                                self._ensure_thread_client()
                                continue
                            
                            #retry_count += 1
                            #logger.info(f"[{thread_name}] Retrying with new key, attempt {retry_count}/{max_retries}")
                            continue
                        else:
                            # Single key mode - wait and retry indefinitely
                            wait_time = min(60 * (retry_count + 1), 120)
                            print(f"[{thread_name}] Rate limit, waiting {wait_time}s")
                            
                            for i in range(wait_time):
                                if self._cancelled:
                                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                                time.sleep(1)
                            
                            #retry_count += 1
                            continue
                    
                    # Check for cancellation
                    elif isinstance(e, UnifiedClientError) and e.error_type in ["cancelled", "timeout"]:
                        retry_reason = f"cancelled_{e.error_type}"
                        raise
                    
                    # Other errors
                    elif retry_count < max_retries - 1:
                        # Determine retry reason based on error type
                        if "timeout" in error_str.lower():
                            retry_reason = "timeout_error"
                        elif "connection" in error_str.lower():
                            retry_reason = "connection_error"
                        elif "500" in error_str or "502" in error_str or "503" in error_str:
                            retry_reason = f"server_error_{error_str[:3]}"
                        else:
                            # Generic error with exception type
                            retry_reason = f"error_{type(e).__name__}"[:30]  # Limit length for filename
                        
                        if self._multi_key_mode:
                            tls = self._get_thread_local_client()
                            if tls.key_index is not None:
                                self._api_key_pool.mark_key_error(tls.key_index)
                            
                            if not self._force_rotation:
                                # Error-based rotation - try a different key
                                print(f"[{thread_name}] Image error occurred ({retry_reason}), rotating to new key...")
                                
                                # Force reassignment
                                tls.initialized = False
                                tls.request_count = 0
                                self._ensure_thread_client()
                                
                                retry_count += 1
                                print(f"[{thread_name}] Rotated to {self.key_identifier} after image error")
                                continue
                        
                        # Retry with same key (or if rotation disabled)
                        logger.info(f"[{thread_name}] Retrying image after {retry_reason} (attempt {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(2)
                        continue
                    
                    # Can't retry - final error
                    else:
                        retry_reason = f"final_error_{type(e).__name__}"
                        print(f"[{thread_name}] Cannot retry image request: {retry_reason}")
                        raise
            
            # === IMPROVED CACHE UPDATE AND CLEANUP FOR IMAGES ===
            if successful_response and context in ['image_translation', 'glossary']:
                content, finish_reason = successful_response
                
                
                logger.info(f"[{thread_name}] Cached successful image response for {context_str}")
                

            if successful_response:
                return successful_response
            
            # Exhausted retries
            if last_error:
                print(f"[{thread_name}] Exhausted {max_retries} image retries, last reason: {retry_reason}")
                raise last_error
            else:
                raise Exception(f"Image request failed after {max_retries} attempts")
                
        except Exception as e:
            raise

    def _send_image_internal(self, messages: List[Dict[str, Any]], image_data: Any,
                            temperature: Optional[float] = None, 
                            max_tokens: Optional[int] = None,
                            max_completion_tokens: Optional[int] = None,
                            context: str = 'image_translation',
                            retry_reason: Optional[str] = None, 
                            request_id=None) -> Tuple[str, str]:  # request_id already in signature
        """
        Internal implementation of send_image with integrated 500 error retry logic and universal extraction
        """
        start_time = time.time()
        
        # Generate request hash WITH request ID if provided
        image_size = len(image_data) if isinstance(image_data, (bytes, str)) else 0
        
        if request_id:
            messages_hash = self._get_request_hash_with_request_id(messages, request_id)
        else:
            # Fallback for direct calls (shouldn't happen in normal flow)
            request_id = str(uuid.uuid4())[:8]
            messages_hash = self._get_request_hash_with_request_id(messages, request_id)
        
        request_hash = f"{messages_hash}_img{image_size}"
        thread_name = threading.current_thread().name
        
        # Log with hash for tracking
        logger.debug(f"[{thread_name}] _send_image_internal starting for {context} (hash: {request_hash[:8]}...) retry_reason: {retry_reason}")
        
        # Reset cancelled flag
        self._cancelled = False
        
        # Reset counters when context changes
        if context != self.current_session_context:
            self.reset_conversation_for_new_context(context)
        
        self.context = context or 'image_translation'
        self.conversation_message_count += 1
        
        # Internal retry logic for 500 errors - INCREASED TO 7
        internal_retries = 7
        base_delay = 5  # Base delay for exponential backoff
        
        # Track if we've tried main key for prohibited content
        main_key_attempted = False
        
        # Define content filter indicators locally for consistency
        content_filter_indicators = [
            "content_filter", "content was blocked", "response was blocked",
            "safety filter", "content policy", "harmful content",
            "blocked by safety", "harm_category", "content_policy_violation",
            "unsafe content", "violates our usage policies",
            "prohibited_content", "blockedreason", "content blocked",
            "inappropriate image", "inappropriate content"
        ]
        
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
                
        for attempt in range(internal_retries):
            try:
                # Validate request (basic validation for images)
                if not messages:
                    raise UnifiedClientError("No messages provided for image request", error_type="validation")

                os.makedirs("Payloads", exist_ok=True)

                # Apply reinforcement
                messages = self._apply_pure_reinforcement(messages)

                # Get file names - now unique per request AND attempt
                payload_name, response_name = self._get_file_names(messages, context=self.context)
                
                # Add request ID and attempt to filename for complete isolation
                base_payload, ext_payload = os.path.splitext(payload_name)
                base_response, ext_response = os.path.splitext(response_name)
                
                # Include request_id and attempt in filename
                unique_suffix = f"_{request_id}_A{attempt}"
                payload_name = f"{base_payload}{unique_suffix}{ext_payload}"
                response_name = f"{base_response}{unique_suffix}{ext_response}"

                # Create a sanitized version for payload (without actual image data)
                payload_messages = [
                    {**msg, 'content': 'IMAGE_DATA_OMITTED' if isinstance(msg.get('content'), list) else msg.get('content')}
                    for msg in messages
                ]

                # Save payload with retry reason
                # On internal retries (500 errors), add that info too
                if attempt > 0:
                    internal_retry_reason = f"500_error_imageattempt{attempt}"
                    if retry_reason:
                        combined_reason = f"{retry_reason}_{internal_retry_reason}"
                    else:
                        combined_reason = internal_retry_reason
                    self._save_payload(payload_messages, payload_name, retry_reason=combined_reason)
                else:
                    self._save_payload(payload_messages, payload_name, retry_reason=retry_reason)

                # Now save the payload (payload_messages is now defined)
                #self._save_payload(payload_messages, payload_name)
        
                # Check for timeout toggle from GUI
                retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
                if retry_timeout_enabled:
                    timeout_seconds = int(os.getenv("CHUNK_TIMEOUT", "180"))
                    logger.info(f"Image timeout monitoring enabled: {timeout_seconds}s limit")
                
                # Log the request details
                logger.info(f"Sending image request to {self.client_type} ({self.model})")
                logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens or max_completion_tokens}")
                
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
                    # Try OpenAI-compatible endpoint as fallback
                    print(f"⚠️ No specific image handler for {self.client_type}, trying OpenAI-compatible endpoint")
                    response = self._send_openai_image(messages, image_base64, temperature,
                                                     max_tokens, max_completion_tokens, response_name)
                
                # Check for cancellation (from timeout or stop button)
                if self._cancelled:
                    logger.info("Image operation cancelled (timeout or user stop)")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                # ====== UNIVERSAL EXTRACTION INTEGRATION ======
                # Use universal extraction instead of assuming response.content exists
                extracted_content = ""
                finish_reason = 'stop'
                
                if response:
                    # Prepare provider-specific parameters
                    extraction_kwargs = {}
                    
                    # Add Gemini-specific parameters if applicable
                    if self.client_type == 'gemini':
                        # Check if this model supports thinking
                        extraction_kwargs['supports_thinking'] = self._supports_thinking()
                        # Get thinking budget from environment
                        extraction_kwargs['thinking_budget'] = int(os.getenv("THINKING_BUDGET", "-1"))
                    
                    # Try universal extraction with provider-specific parameters
                    extracted_content, finish_reason = self._extract_response_text(
                        response, 
                        provider=self.client_type,
                        **extraction_kwargs
                    )
                    
                    # If extraction failed but we have a response object
                    if not extracted_content and response:
                        print(f"⚠️ Failed to extract text from {self.client_type} image response")
                        print(f"   Response type: {type(response)}")
                        print(f"   Consider checking image response extraction for this provider")
                        
                        # Log the response structure for debugging
                        self._save_failed_request(messages, "Image extraction failed", context, response)
                        
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
                    logger.debug(f"Saved image response to: {response_name}")
                
                # Handle empty responses
                if not extracted_content or extracted_content.strip() in ["", "[]", "[IMAGE TRANSLATION FAILED]"]:
                    print(f"Empty or error image response: {finish_reason}")
                    
                    # Check if this is likely a safety filter issue (for ALL providers, not just Gemini)
                    is_likely_safety_filter = False
                    
                    # Pattern 1: Empty content with suspicious finish_reasons (applies to all providers)
                    if not extracted_content and finish_reason in ['length', 'stop', 'max_tokens', None]:
                        print(f"⚠️ Suspicious empty image response from {self.client_type} (finish_reason={finish_reason}) - possible safety filter")
                        is_likely_safety_filter = True
                    
                    # Pattern 2: Check for safety-related content in raw response
                    if response:
                        response_str = ""
                        if hasattr(response, 'raw_response'):
                            response_str = str(response.raw_response).lower()
                        elif hasattr(response, 'error_details'):
                            response_str = str(response.error_details).lower()
                        else:
                            response_str = str(response).lower()
                        
                        safety_indicators = [
                            'safety', 'blocked', 'prohibited', 'harmful', 'inappropriate',
                            'refused', 'content_filter', 'content_policy', 'violation',
                            'cannot assist', 'unable to process', 'against guidelines',
                            'ethical', 'responsible ai', 'harm_category', 'nsfw',
                            'adult content', 'explicit', 'violence', 'disturbing'
                        ]
                        
                        if any(indicator in response_str for indicator in safety_indicators):
                            print(f"❌ Safety indicators found in image response from {self.client_type}")
                            is_likely_safety_filter = True
                    
                    # Pattern 3: Check for specific safety filter messages in extracted content
                    if extracted_content:
                        content_lower = extracted_content.lower()
                        safety_phrases = [
                            'blocked', 'safety', 'cannot', 'unable', 'prohibited',
                            'content filter', 'refused', 'inappropriate', 'i cannot',
                            "i can't", "i'm not able", "not able to", "against my",
                            'content policy', 'guidelines', 'ethical', 'analyze this image',
                            'process this image', 'describe this image', 'nsfw'
                        ]
                        if any(phrase in content_lower for phrase in safety_phrases):
                            print(f"❌ Safety filter phrases detected in image content from {self.client_type}")
                            is_likely_safety_filter = True
                    
                    # Pattern 4: Provider-specific patterns for vision models
                    actual_provider = self._get_actual_provider()

                    if actual_provider in ['openai', 'azure', 'electronhub', 'openrouter', 'poe', 'gemini']:
                        # These providers often return empty on safety issues
                        if not extracted_content and finish_reason != 'error':
                            print(f"⚠️ {actual_provider} returned empty content - likely safety filter")
                            is_likely_safety_filter = True
                    
                    # Pattern 5: Image-specific safety checks
                    # Images are more likely to trigger safety filters
                    if not extracted_content:
                        print(f"⚠️ Empty response for image request - checking for safety filter")
                        # For image requests, empty responses are almost always safety filters
                        is_likely_safety_filter = True
                    
                    # If it's likely a safety filter and we haven't tried main key yet
                    if is_likely_safety_filter and not main_key_attempted:
                        # Only try main key if conditions are met
                        if (self._multi_key_mode and 
                            hasattr(self, 'original_api_key') and 
                            hasattr(self, 'original_model') and
                            self.original_api_key and 
                            self.original_model):
                            
                            print(f"🔄 Empty/blocked image response likely due to safety filter - attempting main key fallback")
                            print(f"   Provider: {self.client_type}")
                            print(f"   Current model: {self.model} ({self.key_identifier})")
                            print(f"   Main key model: {self.original_model}")
                            
                            main_key_attempted = True
                            
                            try:
                                # Create temporary client with main key for image
                                main_response = self._retry_image_with_main_key(
                                    messages, image_data, temperature, max_tokens, max_completion_tokens, context, request_id=request_id
                                )
                                
                                if main_response:
                                    content, finish_reason = main_response
                                    if content and content.strip() and len(content) > 10:  # Make sure we got actual content
                                        print(f"✅ Main key succeeded for image! Got {len(content)} chars")
                                        return content, finish_reason
                                    else:
                                        print(f"❌ Main key also returned empty/minimal image content: {len(content) if content else 0} chars")
                                else:
                                    print(f"❌ Main key returned None for image")
                                    
                            except Exception as main_error:
                                print(f"❌ Main key image error: {str(main_error)[:200]}")
                                # Check if main key also hit content filter
                                main_error_str = str(main_error).lower()
                                if any(indicator in main_error_str for indicator in content_filter_indicators):
                                    print(f"❌ Main key also hit content filter for image")
                                # Continue to normal error handling
                        else:
                            if not self._multi_key_mode:
                                print(f"❌ Not in multi-key mode, cannot retry image with main key")
                            else:
                                print(f"❌ Main key not available for image retry (check configuration)")
                    elif main_key_attempted:
                        print(f"❌ Already attempted main key for this image request")
                    
                    # If we couldn't retry or retry failed, continue with normal error handling
                    self._save_failed_request(messages, f"Empty image response from {self.client_type} (possible safety filter)", context, response)
                    
                    # Log the failure
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=extracted_content or "",
                        finish_reason='content_filter' if is_likely_safety_filter else (finish_reason or 'error'),
                        context=context,
                        error_details={
                            'likely_safety_filter': is_likely_safety_filter,
                            'original_finish_reason': finish_reason,
                            'provider': self.client_type,
                            'model': self.model,
                            'key_identifier': self.key_identifier,
                            'request_type': 'image'
                        } if is_likely_safety_filter else getattr(response, 'error_details', None)
                    )
                    
                    self._track_stats(context, False, "empty_image_response", time.time() - start_time)
                    
                    # Use fallback with appropriate reason
                    if is_likely_safety_filter:
                        fallback_reason = f"image_safety_filter_{self.client_type}"
                    else:
                        fallback_reason = "empty_image"
                    
                    fallback_content = self._handle_empty_result(messages, context, 
                        getattr(response, 'error_details', fallback_reason) if response else fallback_reason)
                    
                    # Return with appropriate finish_reason
                    return fallback_content, 'content_filter' if is_likely_safety_filter else 'error'
                
                # Track success
                self._track_stats(context, True, None, time.time() - start_time)
                
                # Mark key as successful in multi-key mode
                self._mark_key_success()
                
                # Log important info for retry mechanisms
                if finish_reason in ['length', 'max_tokens']:
                    print(f"Image response was truncated: {finish_reason}")
                    print(f"⚠️ Image response truncated (finish_reason: {finish_reason})")
                    
                    # ALWAYS log truncation failures
                    self._log_truncation_failure(
                        messages=messages,
                        response_content=extracted_content,
                        finish_reason=finish_reason,
                        context=context,
                        error_details=getattr(response, 'error_details', None) if response else None
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
                return extracted_content, finish_reason
                
            except UnifiedClientError as e:
                # Handle cancellation specially for timeout support
                if e.error_type == "cancelled" or "cancelled" in str(e):
                    self._in_cleanup = False  # Ensure cleanup flag is set
                    logger.info("Propagating image cancellation to caller")
                    # Re-raise so send_with_interrupt can handle it
                    raise
                
                print(f"UnifiedClient image error: {e}")
                
                # Check if it's a rate limit error and re-raise for retry logic
                error_str = str(e).lower()
                if e.error_type == "rate_limit" or "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic in outer send_image() method
                
                # Check for prohibited content - check BOTH error_type AND error string
                if e.error_type == "prohibited_content" or any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Prohibited image content detected: {error_str[:200]}")
                    
                    # Only try main key if conditions are met
                    if (self._multi_key_mode and 
                        not main_key_attempted and 
                        hasattr(self, 'original_api_key') and 
                        hasattr(self, 'original_model') and
                        self.original_api_key and 
                        self.original_model):
                        
                        print(f"🔄 Attempting main key fallback for prohibited image content")
                        print(f"   Current key: {self.key_identifier}")
                        print(f"   Main key model: {self.original_model}")
                        
                        main_key_attempted = True
                        
                        try:
                            # Create temporary client with main key for image
                            main_response = self._retry_image_with_main_key(
                                messages, image_data, temperature, max_tokens, max_completion_tokens, context
                            )
                            
                            if main_response:
                                content, finish_reason = main_response
                                print(f"✅ Main key succeeded for image! Returning response")
                                return content, finish_reason
                            else:
                                print(f"❌ Main key returned None for image")
                                
                        except Exception as main_error:
                            print(f"❌ Main key image error: {str(main_error)[:200]}")
                            # Check if main key also hit content filter
                            main_error_str = str(main_error).lower()
                            if any(indicator in main_error_str for indicator in content_filter_indicators):
                                print(f"❌ Main key also hit content filter for image")
                            # Continue to normal error handling
                    
                    # Normal prohibited content handling
                    print(f"❌ Image content prohibited - not retrying further")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors - retry these with exponential backoff
                http_status = getattr(e, 'http_status', None)
                if http_status == 500 or "500" in error_str or "api_error" in error_str:
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        # Cap the maximum delay to prevent extremely long waits
                        delay = min(delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Image server error (500) - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        # Wait with cancellation check
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)  # Check every 0.5 seconds
                        continue
                    else:
                        print(f"❌ Image server error (500) - exhausted {internal_retries} retries")
                
                # Save failed request and return fallback
                self._save_failed_request(messages, e, context)
                self._track_stats(context, False, type(e).__name__, time.time() - start_time)
                fallback_content = self._handle_empty_result(messages, context, str(e))
                return fallback_content, 'error'
                
            except Exception as e:
                print(f"Unexpected image error: {e}")
                error_str = str(e).lower()
                
                # For unexpected errors, check if it's a timeout
                if "timed out" in error_str:
                    # Re-raise timeout errors so the retry logic can handle them
                    raise UnifiedClientError(f"Image request timed out: {e}", error_type="timeout")
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    raise  # Re-raise for multi-key retry logic
                
                # Check for prohibited content in unexpected errors
                if any(indicator in error_str for indicator in content_filter_indicators):
                    print(f"❌ Image content prohibited in unexpected error: {error_str[:200]}")
                    
                    # Debug current state
                    #self._debug_multi_key_state()
                    
                    # If we're in multi-key mode and haven't tried the main key yet
                    if (self._multi_key_mode and 
                        not main_key_attempted and 
                        hasattr(self, 'original_api_key') and 
                        hasattr(self, 'original_model') and
                        self.original_api_key and 
                        self.original_model):
                        
                        print(f"🔄 Attempting main key fallback for prohibited image (from unexpected error)")
                        print(f"   Current key: {self.key_identifier}")
                        print(f"   Main key model: {self.original_model}")
                        
                        main_key_attempted = True
                        
                        try:
                            # Create temporary client with main key for image
                            main_response = self._retry_image_with_main_key(
                                messages, image_data, temperature, max_tokens, max_completion_tokens, context
                            )
                            
                            if main_response:
                                content, finish_reason = main_response
                                print(f"✅ Main key succeeded for image! Returning response")
                                return content, finish_reason
                            else:
                                print(f"❌ Main key returned None for image")
                                
                        except Exception as main_error:
                            print(f"❌ Main key image error: {str(main_error)[:200]}")
                            # Check if main key also hit content filter
                            main_error_str = str(main_error).lower()
                            if any(indicator in main_error_str for indicator in content_filter_indicators):
                                print(f"❌ Main key also hit content filter for image")
                    
                    # Fall through to normal error handling
                    print(f"❌ Image content prohibited - not retrying")
                    self._save_failed_request(messages, e, context)
                    self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                    fallback_content = self._handle_empty_result(messages, context, str(e))
                    return fallback_content, 'error'
                
                # Check for 500 errors in unexpected exceptions
                if "500" in error_str or "internal server error" in error_str:
                    if attempt < internal_retries - 1:
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        delay = min(delay, 60)  # Max 60 seconds
                        
                        print(f"🔄 Image server error (500) - auto-retrying in {delay:.1f}s (attempt {attempt + 1}/{internal_retries})")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue
                
                # Check for other transient errors with exponential backoff
                transient_errors = ["502", "503", "504", "connection reset", "connection aborted"]
                if any(err in error_str for err in transient_errors):
                    if attempt < internal_retries - 1:
                        # Use a slightly less aggressive backoff for transient errors
                        delay = (base_delay/2 * (2 ** attempt)) + random.uniform(0, 1)
                        delay = min(delay, 30)  # Max 30 seconds for transient errors
                        
                        print(f"🔄 Image transient error - retrying in {delay:.1f}s")
                        
                        wait_start = time.time()
                        while time.time() - wait_start < delay:
                            if self._cancelled:
                                raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                            time.sleep(0.5)
                        continue
                
                # Save failed request and return fallback for other errors
                self._save_failed_request(messages, e, context)
                self._track_stats(context, False, "unexpected_error", time.time() - start_time)
                fallback_content = self._handle_empty_result(messages, context, str(e))
                return fallback_content, 'error'

    def _retry_image_with_main_key(self, messages, image_data, temperature=None, max_tokens=None, 
                                   max_completion_tokens=None, context=None, request_id=None) -> Optional[Tuple[str, Optional[str]]]:
        """
        Create a temporary client with the main key and retry the image request.
        This is used for prohibited content errors in multi-key mode.
        """
        # Use the class-level fallback key if available, otherwise use original
        fallback_key = getattr(self.__class__, '_main_fallback_key', self.original_api_key)
        fallback_model = getattr(self.__class__, '_main_fallback_model', self.original_model)
        
        # Don't retry with the same key that just failed
        if fallback_model == self.model and fallback_key == self.api_key:
            print(f"[MAIN KEY IMAGE RETRY] Fallback is same as current key ({self.model}), skipping retry")
            return None
        
        print(f"[MAIN KEY IMAGE RETRY] Starting image retry with main key")
        print(f"[MAIN KEY IMAGE RETRY] Current failing model: {self.model}")
        print(f"[MAIN KEY RETRY] Fallback model if main key retry fails: {fallback_model}")
        
        try:
            # Create a new temporary UnifiedClient instance with the fallback key
            temp_client = UnifiedClient(
                api_key=fallback_key,  # Use fallback instead of original
                model=fallback_model,   # Use fallback instead of original
                output_dir=self.output_dir
            )
            
            # FORCE single-key mode after initialization
            temp_client._multi_key_mode = False
            temp_client.use_multi_keys = False
            temp_client.key_identifier = "Main Key (Image Fallback)"
            
            # The client should already be set up from __init__, but verify
            if not hasattr(temp_client, 'client_type') or temp_client.client_type is None:
                # Force setup if needed
                temp_client.api_key = self.original_api_key
                temp_client.model = self.original_model
                temp_client._setup_client()
            
            # Copy relevant state BUT NOT THE CANCELLATION FLAG
            temp_client.context = context or 'image_translation'
            # DON'T COPY THE CANCELLED FLAG - This is the bug!
            temp_client._cancelled = False  # ALWAYS start fresh for main key retry
            temp_client._in_cleanup = False
            temp_client.current_session_context = self.current_session_context
            temp_client.conversation_message_count = self.conversation_message_count
            
            # Copy image-specific settings if they exist
            temp_client.default_temperature = getattr(self, 'default_temperature', 0.3)
            temp_client.default_max_tokens = getattr(self, 'default_max_tokens', 8192)
            temp_client.request_timeout = self.request_timeout
            
            print(f"[MAIN KEY IMAGE RETRY] Created temp client with model: {temp_client.model}")
            print(f"[MAIN KEY IMAGE RETRY] Temp client type: {getattr(temp_client, 'client_type', 'NOT SET')}")
            print(f"[MAIN KEY IMAGE RETRY] Multi-key mode: {temp_client._multi_key_mode}")
            print(f"[MAIN KEY IMAGE RETRY] Cancelled flag: {temp_client._cancelled}")  # Debug log
            
            # Get file names for response tracking
            payload_name, response_name = self._get_file_names(messages, context=context)
            
            # Try to send the image request using _send_image_internal
            print(f"[MAIN KEY IMAGE RETRY] Sending image request...")
            
            # Use _send_image_internal directly to avoid nested retry loops
            result = temp_client._send_image_internal(
                messages=messages,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                context=context,
                retry_reason="main_key_image_fallback",
                request_id=request_id
            )
            
            # Check the result
            if result and isinstance(result, tuple):
                content, finish_reason = result
                if content:
                    print(f"[MAIN KEY IMAGE RETRY] Success! Got image response of length: {len(content)}")
                    # Save the response using our instance's method
                    self._save_response(content, response_name)
                    return content, finish_reason
                else:
                    print(f"[MAIN KEY IMAGE RETRY] Empty content returned for image")
                    return None
            else:
                print(f"[MAIN KEY IMAGE RETRY] Unexpected result type: {type(result)}")
                return None
            
        except UnifiedClientError as e:
            # Check if it's a cancellation from the temp client
            if e.error_type == "cancelled":
                print(f"[MAIN KEY IMAGE RETRY] Operation was cancelled during main key retry")
                # Don't propagate cancellation - just return None
                return None
                
            print(f"[MAIN KEY IMAGE RETRY] UnifiedClientError: {type(e).__name__}: {str(e)[:500]}")
            
            # Check if it's also a content filter error
            error_str = str(e).lower()
            content_filter_indicators = [
                "content_filter", "content was blocked", "response was blocked",
                "safety filter", "content policy", "harmful content",
                "blocked by safety", "harm_category", "content_policy_violation",
                "unsafe content", "violates our usage policies",
                "prohibited_content", "blockedreason", "content blocked",
                "inappropriate image", "inappropriate content"
            ]
            
            if any(indicator in error_str for indicator in content_filter_indicators):
                print(f"[MAIN KEY IMAGE RETRY] Main key also hit content filter for image")
            
            # Re-raise other errors
            raise
            
        except Exception as e:
            print(f"[MAIN KEY IMAGE RETRY] Exception: {type(e).__name__}: {str(e)[:500]}")
            
            # Check if it's also a content filter error
            error_str = str(e).lower()
            content_filter_indicators = [
                "content_filter", "content was blocked", "response was blocked",
                "safety filter", "content policy", "harmful content",
                "blocked by safety", "harm_category", "content_policy_violation",
                "unsafe content", "violates our usage policies",
                "prohibited_content", "blockedreason", "content blocked",
                "inappropriate image", "inappropriate content"
            ]
            
            if any(indicator in error_str for indicator in content_filter_indicators):
                print(f"[MAIN KEY IMAGE RETRY] Main key also hit content filter for image")
            
            # Re-raise so the calling method can handle it
            raise
 
    def reset_conversation_for_new_context(self, new_context):
        """Reset conversation state when context changes"""
        if hasattr(self, '_instance_model_lock'):
            with self._instance_model_lock:
                self.current_session_context = new_context
                self.conversation_message_count = 0
                self.pattern_counts.clear()
                self.last_pattern = None
        else:
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
                if hasattr(self, '_instance_model_lock'):
                    with self._instance_model_lock:
                        self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1
                        count = self.pattern_counts[pattern_key]
                else:
                    self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1
                    count = self.pattern_counts[pattern_key]
                
                # Just track patterns, NO PROMPT INJECTION
                if count >= 3:
                    logger.info(f"Pattern {pattern_key} detected (count: {count})")
                    # NO [PATTERN REINFORCEMENT ACTIVE] - KEEP IT GONE
        
        return messages
    
    def _validate_request(self, messages, max_tokens=None):
        """Validate request parameters before sending"""
        if not messages:
            return False, "Empty messages list"
        
        # Check message content isn't empty
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
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
            # Check if content exists and is not empty
            if response.content and isinstance(response.content, str) and len(response.content) > 0:
                print(f"   ✅ Got text from UnifiedResponse.content: {len(response.content)} chars")
                return response.content, response.finish_reason or 'stop'
            elif response.error_details:
                # Handle error responses
                print(f"   ⚠️ UnifiedResponse has error_details: {response.error_details}")
                return "", response.finish_reason or 'error'
            else:
                # Content is None or empty, try to extract from raw_response if available
                print(f"   ⚠️ UnifiedResponse.content is empty or None, checking raw_response...")
                if hasattr(response, 'raw_response') and response.raw_response:
                    print(f"   🔍 Found raw_response, attempting extraction...")
                    # Continue to provider-specific extraction using raw_response
                    response = response.raw_response
                else:
                    print(f"   ⚠️ No raw_response found in UnifiedResponse")
                    # Continue with the UnifiedResponse object itself
        
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
                        
                        # Content might have parts
                        if hasattr(content, 'parts'):
                            print(f"   🔍 [Gemini] Found {len(content.parts)} parts in content")
                            text_parts = []
                            
                            for i, part in enumerate(content.parts):
                                part_text = self._extract_part_text(part, provider='gemini', part_index=i+1)
                                if part_text:
                                    text_parts.append(part_text)
                            
                            if text_parts:
                                result = ''.join(text_parts)
                                print(f"   ✅ [Gemini] Extracted from parts: {len(result)} chars")
                                return result, finish_reason
                        
                        # Try direct text access on content
                        elif hasattr(content, 'text'):
                            if content.text:
                                print(f"   ✅ [Gemini] Got text from content.text: {len(content.text)} chars")
                                return content.text, finish_reason
                    
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
            
            # Try parts directly on response
            if hasattr(response, 'parts'):
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
            if hasattr(response, 'choices'):
                print(f"   🔍 [OpenAI] Found choices attribute, {len(response.choices)} choices")
                
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
        Enhanced with provider-specific handling.
        """
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
                        'model': self.model,
                        'client_type': self.client_type,
                        'messages': messages,
                        'timestamp': datetime.now().isoformat(),
                        'debug': debug_info,
                        'key_identifier': self.key_identifier,
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
        print("🛑 Operation cancelled (timeout or user stop)")
        print("🛑 API operation cancelled")

    def reset_cleanup_state(self):
            """Reset cleanup state for new operations"""
            self._in_cleanup = False
            self._cancelled = False

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
        # FIX: Ensure max_tokens has a value before passing to handlers
        if max_tokens is None and max_completion_tokens is None:
            # Use instance default or standard default
            max_tokens = getattr(self, 'max_tokens', 8192)
        elif max_tokens is None and max_completion_tokens is not None:
            # For o-series models, use max_completion_tokens as fallback
            max_tokens = max_completion_tokens
        # Check if this is actually Gemini (including when using OpenAI endpoint)
        actual_provider = self._get_actual_provider()
        
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
        
        # IMPORTANT: Use actual_provider for routing, not client_type
        # This ensures Gemini always uses its native handler even when using OpenAI endpoint
        handler = handlers.get(actual_provider)
        
        if not handler:
            # Fallback to client_type if no actual_provider match
            handler = handlers.get(self.client_type)
        
        if not handler:
            # Try fallback to Together AI for open models
            if self.client_type in ['bigscience', 'meta', 'databricks', 'huggingface', 'salesforce']:
                logger.info(f"Using Together AI for {self.client_type} model")
                return self._send_together(messages, temperature, max_tokens, response_name)
            raise UnifiedClientError(f"No handler for client type: {self.client_type}")
        
        # Route based on actual provider (handles Gemini with OpenAI endpoint correctly)
        if actual_provider == 'gemini':
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
        return self.client_type

    def _is_gemini_request(self) -> bool:
        """
        Check if this is a Gemini request (native or via OpenAI endpoint)
        """
        return self._get_actual_provider() == 'gemini'
    
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


    def _send_openrouter(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenRouter API - simplified version that trusts user input"""
        # Check if safety settings are disabled via GUI toggle
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Just use the model as provided by the user
        model_name = self.model
        
        # Only strip OpenRouter prefixes if present (or/ and openrouter/)
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
        
        # Save configuration if needed
        if os.getenv("SAVE_PAYLOAD", "1") == "1":
            config_data = {
                "provider": "openrouter",
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "safety_disabled": disable_safety,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            self._save_openrouter_config(config_data, response_name)
        
        # MICROSECOND LOCK: Only lock when setting model, not during API call
        if hasattr(self, '_instance_model_lock'):
            with self._instance_model_lock:
                original_model = self.model
                self.model = model_name
        else:
            original_model = self.model
            self.model = model_name
        
        try:
            # API call happens OUTSIDE the lock - allows parallelism!
            result = self._send_openai_compatible(
                messages, temperature, max_tokens,
                base_url="https://openrouter.ai/api/v1",
                response_name=response_name,
                provider="openrouter",
                headers=headers
            )
            return result
        finally:
            # Lock again just to restore
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
                    self.model = original_model
            else:
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
        max_retries = 7
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
                    timeout=self.request_timeout
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
                elif not fixes_attempted['max_tokens_param'] and "max_tokens" in error_str and ("not supported" in error_str or "unsupported" in error_str):
                    print(f"Model {self.model} requires max_completion_tokens instead of max_tokens, will retry...")
                    fixes_attempted['max_tokens_param'] = True
                    should_retry = True
                
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
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{threading.current_thread().ident}")
        os.makedirs(thread_dir, exist_ok=True)
        return thread_dir

    def _save_gemini_safety_config(self, config_data: dict, response_name: str = None):
        """Save Gemini safety configuration with proper thread organization"""
        if not os.getenv("SAVE_PAYLOAD", "1") == "1":
            return
        
        # Handle None or empty response_name
        if not response_name:
            response_name = f"safety_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize response_name to ensure it's filesystem-safe
        # Remove or replace invalid characters
        import re
        response_name = re.sub(r'[<>:"/\\|?*]', '_', str(response_name))
        
        # Determine context from thread name
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        if 'Translation' in thread_name:
            context = 'translation'
        elif 'Glossary' in thread_name:
            context = 'glossary'
        else:
            context = 'general'
        
        # Use STABLE thread directory (same as other save methods)
        thread_dir = os.path.join("Payloads", context, f"{thread_name}_{thread_id}")
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
                print(f"Saved Gemini safety status to: {config_path}")
        except Exception as e:
            print(f"Failed to save Gemini safety config: {e}")

    def _send_gemini(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Gemini API with support for both text and multi-image messages"""
        response = None
        
        # Check if we should use OpenAI-compatible endpoint
        use_openai_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
        gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
        
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
        
        # Configure safety settings based on toggle (SAME FOR BOTH ENDPOINTS)
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
        current_tokens = max_tokens * BOOST_FACTOR
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
        config_data = {
            "type": "GEMINI_OPENAI_ENDPOINT_REQUEST" if use_openai_endpoint else "TEXT_REQUEST",
            "model": self.model,
            "endpoint": gemini_endpoint if use_openai_endpoint else "native",
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
                    "max_output_tokens": current_tokens,
                    **anti_dupe_params  # Add user's custom parameters
                }
                
                # Log the request
                print(f"   📊 Temperature: {temperature}, Max tokens: {current_tokens}")

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
                        max_tokens=current_tokens,
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
                        
                        # Display thinking tokens if found or if thinking was requested
                        if supports_thinking:
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
                    response = self.gemini_client.models.generate_content(
                        model=self.model,
                        contents=formatted_prompt,
                        config=generation_config
                    )
                    
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
                    
                    # Log thinking token usage if available
                    if hasattr(response, 'usage_metadata'):
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
        max_retries = 7
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
                    
                    # Check if this is Gemini via OpenAI endpoint
                    is_gemini_endpoint = provider == "gemini-openai" or self.model.lower().startswith('gemini')
                    
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
                    
                    # Enhanced extraction for Gemini endpoints
                    content = None
                    finish_reason = 'stop'
                    
                    # Log response structure for debugging
                    if is_gemini_endpoint:
                        logger.debug(f"Gemini endpoint response type: {type(resp)}")
                    
                    # Extract content with Gemini awareness
                    if hasattr(resp, 'choices') and resp.choices:
                        choice = resp.choices[0]
                        
                        if hasattr(choice, 'finish_reason'):
                            finish_reason = choice.finish_reason or 'stop'
                        
                        if hasattr(choice, 'message'):
                            message = choice.message
                            
                            # Handle None message (can happen with Gemini)
                            if message is None:
                                content = ""
                                if is_gemini_endpoint:
                                    print("Gemini returned None message")
                                    content = "[GEMINI RETURNED NULL MESSAGE]"
                                    finish_reason = 'content_filter'
                            # Try content attribute
                            elif hasattr(message, 'content'):
                                content = message.content
                                # Gemini might return None instead of empty string
                                if content is None:
                                    content = ""
                                    if is_gemini_endpoint:
                                        print("Gemini returned None content - likely safety filter")
                                        content = "[BLOCKED BY GEMINI SAFETY FILTER]"
                                        finish_reason = 'content_filter'
                            # Try text attribute
                            elif hasattr(message, 'text'):
                                content = message.text
                            # Message might be a string directly (some Gemini endpoints)
                            elif isinstance(message, str):
                                content = message
                            else:
                                # Try to extract from string representation
                                msg_str = str(message)
                                if msg_str and not msg_str.startswith('<'):
                                    content = msg_str
                                else:
                                    content = ""
                        
                        # Fallback: try text directly on choice
                        elif hasattr(choice, 'text'):
                            content = choice.text
                        else:
                            content = ""
                    else:
                        content = ""
                        if is_gemini_endpoint:
                            print("Gemini endpoint returned no choices")
                    
                    # Handle None content (final check)
                    if content is None:
                        content = ""
                        if is_gemini_endpoint:
                            print("Gemini final content is None")
                            content = "[GEMINI RETURNED NULL CONTENT]"
                            finish_reason = 'content_filter'
                    
                    # Check for Gemini safety blocks
                    if is_gemini_endpoint and content == "" and finish_reason == 'stop':
                        print("Empty Gemini response with finish_reason='stop' - likely safety filter")
                        content = "[BLOCKED BY GEMINI SAFETY FILTER]"
                        finish_reason = 'content_filter'
                    
                    #  ELECTRONHUB TRUNCATION
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
                    
                    # For rate limits, ALWAYS immediately re-raise to let multi-key system handle it
                    # Don't retry at this level - let the outer send() method handle key rotation
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        print(f"{provider} rate limit hit - passing to multi-key handler")
                        raise UnifiedClientError(f"{provider} rate limit: {e}", error_type="rate_limit")
                    
                    # For other errors, retry at this level ONLY if not in multi-key mode
                    if not self._multi_key_mode and attempt < max_retries - 1:
                        print(f"{provider} SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    elif self._multi_key_mode:
                        # In multi-key mode, let outer handler manage retries with different keys
                        print(f"{provider} error in multi-key mode - passing to outer handler")
                        raise UnifiedClientError(f"{provider} error: {e}", error_type="api_error")
                        
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

    def _send_gemini_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Gemini API - supports both single and multiple images"""
        try:
            response = None
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
                
                # Get thinking budget and check if model supports thinking
                thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))
                supports_thinking = self._supports_thinking()
                
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
                
                # Log thinking status
                if supports_thinking:
                    if thinking_budget == 0:
                        print(f"🧠 Thinking Status: Disabled")
                    elif thinking_budget == -1:
                        print(f"🧠 Thinking Status: Dynamic (model decides)")
                    elif thinking_budget > 0:
                        print(f"🧠 Thinking Status: Budget of {thinking_budget} tokens")
                else:
                    print(f"🧠 Thinking Status: Not supported by model")
                
                # Save configuration to file
                config_data = {
                    "type": "GEMINI_OPENAI_ENDPOINT_IMAGE_REQUEST",
                    "model": self.model,
                    "endpoint": gemini_endpoint,
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
                
                # Route to OpenAI-compatible handler
                response = self._send_openai_compatible(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=gemini_endpoint,
                    response_name=response_name,
                    provider="gemini-openai"
                )
                
                # Extract and display thinking tokens if available
                if supports_thinking and hasattr(response, 'raw_response'):
                    raw_resp = response.raw_response
                    thinking_tokens = 0
                    
                    # Check in usage object
                    if hasattr(raw_resp, 'usage'):
                        usage = raw_resp.usage
                        
                        # Try various field names
                        if hasattr(usage, 'thoughts_token_count'):
                            thinking_tokens = usage.thoughts_token_count or 0
                        elif hasattr(usage, 'thinking_tokens'):
                            thinking_tokens = usage.thinking_tokens or 0
                        elif hasattr(usage, 'reasoning_tokens'):
                            thinking_tokens = usage.reasoning_tokens or 0
                        
                        # Check completion_tokens_details
                        if hasattr(usage, 'completion_tokens_details'):
                            details = usage.completion_tokens_details
                            if hasattr(details, 'reasoning_tokens'):
                                thinking_tokens = details.reasoning_tokens or 0
                    
                    # Display thinking tokens status
                    if thinking_tokens > 0:
                        print(f"   💭 Thinking tokens used: {thinking_tokens}")
                    elif thinking_budget == 0:
                        print(f"   ✅ Thinking successfully disabled (0 thinking tokens)")
                    elif thinking_budget == -1:
                        print(f"   💭 Thinking: Dynamic mode (tokens may not be reported via OpenAI endpoint)")
                    elif thinking_budget > 0:
                        print(f"   ⚠️ Thinking budget set to {thinking_budget} but tokens not reported via OpenAI endpoint")
                
                return response
        
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
            
            # Extract text from the Gemini response - FIXED LOGIC HERE
            text_content = ""
            finish_reason = 'stop'  # Default
            
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
                            # Check finish reason
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason_str = str(candidate.finish_reason)
                                if 'PROHIBITED_CONTENT' in finish_reason_str:
                                    finish_reason = 'prohibited_content'
                                elif 'MAX_TOKENS' in finish_reason_str:
                                    finish_reason = 'length'
                                elif 'SAFETY' in finish_reason_str:
                                    finish_reason = 'safety'
                                elif 'STOP' in finish_reason_str:
                                    finish_reason = 'stop'
                            
                            # Extract content
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
            
            # Log if we still have no content
            if not text_content:
                print(f"   ❌ Gemini returned empty response")
                print(f"   🔍 Response attributes: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'No __dict__'}")
                text_content = ""
                finish_reason = 'error'
            else:
                print(f"   ✅ Successfully extracted {len(text_content)} characters")
            
            return UnifiedResponse(
                content=text_content,  # Properly populated with the actual response text
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
        """Send image request through OpenRouter - simplified version that trusts user input"""
        # Check if safety settings are disabled
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Just use the model as provided by the user
        model_name = self.model
        
        # Only strip OpenRouter prefixes if present
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
            logger.info("🔓 Safety toggle enabled for OpenRouter Vision")
            print("🔓 OpenRouter Vision Safety: Disabled via X-Safe-Mode header")
        
        # Save configuration if needed
        if os.getenv("SAVE_PAYLOAD", "1") == "1":
            config_data = {
                "provider": "openrouter",
                "type": "vision_request",
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "safety_disabled": disable_safety,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "has_image": True
            }
            self._save_openrouter_config(config_data, f"vision_{response_name}" if response_name else "vision")
        
        # MICROSECOND LOCK: Only lock when setting model, not during API call
        if hasattr(self, '_instance_model_lock'):
            with self._instance_model_lock:
                original_model = self.model
                self.model = model_name
        else:
            original_model = self.model
            self.model = model_name
        
        try:
            # API call happens OUTSIDE the lock - allows parallelism!
            payload = {
                "model": self.model,  # Uses the temporarily set model
                "messages": vision_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            if resp.status_code != 200:
                error_msg = f"OpenRouter Vision API error: {resp.status_code} - {resp.text}"
                raise UnifiedClientError(error_msg)
            
            json_resp = resp.json()
            content = json_resp['choices'][0]['message']['content']
            finish_reason = json_resp['choices'][0].get('finish_reason', 'stop')
            
            return UnifiedResponse(
                content=content,
                finish_reason=finish_reason,
                raw_response=json_resp
            )
            
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter Vision API network error: {e}")
            raise UnifiedClientError(f"OpenRouter Vision API network error: {e}")
        except Exception as e:
            print(f"OpenRouter Vision API error: {e}")
            raise UnifiedClientError(f"OpenRouter Vision API error: {e}")
        finally:
            # Lock again just to restore
            if hasattr(self, '_instance_model_lock'):
                with self._instance_model_lock:
                    self.model = original_model
            else:
                self.model = original_model
    
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
            
    def _debug_multi_key_state(self):
        """Debug method to show current multi-key configuration state"""
        print(f"\n[DEBUG] Multi-Key State Information:")
        print(f"  Instance multi-key mode: {self._multi_key_mode}")
        print(f"  Has original_api_key: {hasattr(self, 'original_api_key')}")
        print(f"  Has original_model: {hasattr(self, 'original_model')}")
        
        if hasattr(self, 'original_api_key') and self.original_api_key:
            masked_key = self.original_api_key[:8] + "..." + self.original_api_key[-4:] if len(self.original_api_key) > 12 else "***"
            print(f"  Original API key: {masked_key}")
        else:
            print(f"  Original API key: None")
        
        if hasattr(self, 'original_model') and self.original_model:
            print(f"  Original model: {self.original_model}")
        else:
            print(f"  Original model: None")
        
        print(f"  Current key identifier: {self.key_identifier if hasattr(self, 'key_identifier') else 'None'}")
        print(f"  Current model: {self.model if hasattr(self, 'model') else 'None'}")
        
        # Check API key pool status
        if hasattr(self.__class__, '_api_key_pool') and self.__class__._api_key_pool:
            print(f"  API key pool exists: Yes")
            print(f"  Number of keys in pool: {len(self.__class__._api_key_pool.keys)}")
        else:
            print(f"  API key pool exists: No")
        
        # Check environment variables
        print(f"  USE_MULTI_API_KEYS env: {os.getenv('USE_MULTI_API_KEYS', 'Not set')}")
        print(f"  MULTI_API_KEYS env exists: {'Yes' if os.getenv('MULTI_API_KEYS') else 'No'}")
        
        # Thread-local state
        if hasattr(self, '_thread_local'):
            tls = self._get_thread_local_client()
            print(f"  Thread-local initialized: {getattr(tls, 'initialized', False)}")
            print(f"  Thread-local key_index: {getattr(tls, 'key_index', None)}")
        
        print("[DEBUG] End of Multi-Key State\n")
