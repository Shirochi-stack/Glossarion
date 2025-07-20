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
    """Unified client for multiple AI model APIs with enhanced error handling
    
    ELECTRONHUB USAGE:
    For ElectronHub API aggregator, prefix model names with 'eh/', 'electronhub/', or 'electron/'
    Examples:
    - eh/yi-34b-chat-200k (Yi model via ElectronHub)
    - electronhub/gpt-4.5 (OpenAI model via ElectronHub)
    - electron/claude-4-opus (Anthropic model via ElectronHub)
    
    POE USAGE:
    For Poe platform models, prefix with 'poe/'
    Examples:
    - poe/gpt-4.5
    - poe/claude-4-opus
    - poe/Assistant (Poe's default assistant)
    
    OPENROUTER USAGE:
    For OpenRouter aggregator, prefix with 'or/' or 'openrouter/'
    Examples:
    - or/openai/gpt-4.5
    - openrouter/anthropic/claude-4-opus
    - or/meta-llama/llama-4-70b
    
    Timeout Behavior:
    - Respects GUI "Auto-retry Slow Chunks" timeout setting
    - If enabled: Uses CHUNK_TIMEOUT (default 900s/15min) from GUI
    - If disabled: Uses 1 hour default timeout
    - The timeout applies to ALL API requests (HTTP and SDK)
    - Works with send_with_interrupt() wrapper for proper retry handling
    """
    
    # Supported model prefixes and their providers (Updated July 2025)
    MODEL_PROVIDERS = {
        'vertex/': 'vertex_model_garden',  # For Vertex AI Model Garden
        '@': 'vertex_model_garden',  # For models with @ symbol (claude-sonnet-4@20250514)
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
        # New aggregators and providers
        'poe': 'poe',
        'or': 'openrouter',
        'openrouter': 'openrouter',
        'fireworks': 'fireworks',
        # ElectronHub support - prefix for ElectronHub-routed models
        'eh/': 'electronhub',
        'electronhub/': 'electronhub',
        'electron/': 'electronhub',
    }
    
    # Model-specific constraints (for reference and logging only - handled reactively)
    MODEL_CONSTRAINTS = {
        'temperature_fixed': ['o4-mini', 'o1-mini', 'o1-preview', 'o3-mini', 'o3', 'o3-pro', 'o4-mini'],  # Models that only support specific temperatures
        'no_system_message': ['o1', 'o1-preview', 'o3', 'o3-pro'],    # Models that don't support system messages
        'max_completion_tokens': ['o4', 'o1', 'o3'],        # Models using max_completion_tokens
        'chinese_optimized': ['qwen', 'yi', 'glm', 'chatglm', 'baichuan', 'ernie', 'hunyuan'],  # Models optimized for Chinese
    }
    
    def __init__(self, api_key: str, model: str, output_dir: str = None):
        self._current_output_file = None
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
        self._in_cleanup = False  # Flag to indicate cleanup/cancellation mode
        self.openai_client = None
        self.gemini_client = None
        self.mistral_client = None
        self.cohere_client = None
        print(f"[DEBUG] Initializing UnifiedClient with model: {model}")
        # Get timeout configuration from GUI
        # IMPORTANT: This respects the "Auto-retry Slow Chunks" timeout setting
        # If enabled, chunks can run up to CHUNK_TIMEOUT seconds before being cancelled
        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "0") == "1"
        if retry_timeout_enabled:
            self.request_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))
            logger.info(f"Using GUI-configured timeout: {self.request_timeout}s")
        else:
            self.request_timeout = 3600  # 1 hour default
            logger.info(f"Using default timeout: {self.request_timeout}s")
        
        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'empty_results': 0,
            'errors': {},
            'response_times': []
        }
        
        # Store Google Cloud credentials path if available
        self.google_creds_path = None
        
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
            # Determine client type from model name (existing logic)
            self._setup_client()
            print(f"[DEBUG] After setup - client_type: {self.client_type}, openai_client: {self.openai_client}")
            
            # FORCE OPENAI CLIENT IF CUSTOM BASE URL IS SET
            custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', '')
            if custom_base_url and self.openai_client is None:
                print(f"[DEBUG] Custom base URL detected, forcing OpenAI client for model: {self.model}")
                self.client_type = 'openai'
                
                if openai is None:
                    raise ImportError("OpenAI library not installed. Install with: pip install openai")
                
                self.openai_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=custom_base_url
                )
                print(f"[DEBUG] OpenAI client created with custom base URL: {custom_base_url}")

            
    def _setup_client(self):
        """Setup the appropriate client based on model type"""
        model_lower = self.model.lower()
        print(f"[DEBUG] _setup_client called with model: {self.model}")
        
        # Check if we're using a custom OpenAI base URL
        custom_base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', ''))
        if custom_base_url and custom_base_url != 'https://api.openai.com/v1':
            # Force OpenAI client type for custom endpoints
            self.client_type = 'openai'
            logger.info(f"Using OpenAI client for custom endpoint with model: {self.model}")
            return
        
        # Check model prefixes (existing code)
        self.client_type = None
        for prefix, provider in self.MODEL_PROVIDERS.items():
            if model_lower.startswith(prefix):
                self.client_type = provider
                print(f"[DEBUG] Matched prefix '{prefix}' -> provider '{provider}'")
                break
        
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
            
            # Check for custom base URL
            base_url = os.getenv('OPENAI_CUSTOM_BASE_URL', os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'))
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
        """Save request payload for debugging"""
        filepath = os.path.join("Payloads", filename)
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
            
        try:
            # Ensure Payloads directory exists
            os.makedirs("Payloads", exist_ok=True)
            
            # Use forward slashes for consistency
            safe_filename = filename.replace("\\", "/")
            if "/" in safe_filename:
                safe_filename = safe_filename.split("/")[-1]
            
            filepath = os.path.join("Payloads", safe_filename)
            
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
        self._in_cleanup = True  # Set cleanup flag
        print("🛑 Operation cancelled (timeout or user stop)")
        print("🛑 API operation cancelled")

    def reset_cleanup_state(self):
            """Reset cleanup state for new operations"""
            self._in_cleanup = False
            self._cancelled = False

    def _send_vertex_model_garden(self, messages, temperature=0.7, max_tokens=None, stop_sequences=None):
        """Send request to Vertex AI Model Garden models (including Claude)"""
        try:
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            import google.auth.transport.requests
            import vertexai
            
            # Import your global stop check function
            try:
                from TranslateKRtoEN import is_stop_requested
            except ImportError:
                # Fallback to checking _cancelled flag
                def is_stop_requested():
                    return self._cancelled
            
            # Use the same credentials as Cloud Vision
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
                
                # Use the region from environment variable
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
                # For Gemini models on Vertex AI, use standard Vertex AI SDK
                location = os.getenv('VERTEX_AI_LOCATION', 'us-east5')
                
                # Check stop flag before Gemini call
                if is_stop_requested():
                    logger.info("Stop requested, cancelling Vertex AI Gemini request")
                    raise UnifiedClientError("Operation cancelled by user", error_type="cancelled")
                
                vertexai.init(project=project_id, location=location, credentials=credentials)
                
                # Use the existing Gemini implementation
                return self._send_gemini(messages, temperature, max_tokens, stop_sequences)
                
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
    
    def send(self, messages, temperature=None, max_tokens=None, max_completion_tokens=None, context=None) -> Tuple[str, Optional[str]]:
        """
        Send messages to the API with enhanced error handling
        Returns: (content, finish_reason) tuple for backward compatibility
        
        IMPORTANT: This method supports GUI retry mechanisms:
        - Truncated responses: Returns finish_reason='length' when response is cut off
        - Timeout handling: Respects cancellation from send_with_interrupt wrapper
        - Duplicate detection: Saves all responses for comparison
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens (for non-o series models)
            max_completion_tokens: Maximum completion tokens (for o-series models)
            context: Context identifier
        """
        start_time = time.time()
        
        # Reset cancelled flag
        self._cancelled = False
        
        # Reset counters when context changes
        if context != self.current_session_context:
            self.reset_conversation_for_new_context(context)
        
        self.context = context or 'translation'
        self.conversation_message_count += 1
        
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
                self._in_cleanup = True  # Ensure cleanup flag is set
                logger.info("Propagating cancellation to caller")
                # Re-raise so send_with_interrupt can handle it
                raise
            
            print(f"UnifiedClient error: {e}")
            self._save_failed_request(messages, e, context)
            self._track_stats(context, False, type(e).__name__, time.time() - start_time)
            
            # Return fallback
            fallback_content = self._handle_empty_result(messages, context, str(e))
            return fallback_content, 'error'
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            self._save_failed_request(messages, e, context)
            self._track_stats(context, False, "unexpected_error", time.time() - start_time)
            
            # For unexpected errors, check if it's a timeout
            if "timed out" in str(e).lower():
                # Re-raise timeout errors so the retry logic can handle them
                raise UnifiedClientError(f"Request timed out: {e}", error_type="timeout")
            
            # Return fallback for other errors
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
        elif self.client_type == 'vertex_model_garden':  # ADD THIS ELIF BLOCK
            # Vertex AI doesn't use response_name parameter
            return handler(messages, temperature, max_tokens or max_completion_tokens)
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

    def _send_gemini(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Gemini API with support for both text and multi-image messages"""
        
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
            
        #print(f"🔒 Gemini Safety Status: {safety_status}")
        
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
        
        # Save to Payloads folder
        os.makedirs("Payloads", exist_ok=True)
        config_filename = f"gemini_safety_{response_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_path = os.path.join("Payloads", config_filename)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Could not save safety config: {e}")
               
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
                print(f"   📤 Sending text request to Gemini{thinking_status}")
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
                            result = ''.join(part.text for part in parts if hasattr(part, 'text'))
                        
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
            
            # Reduce tokens and retry
            current_tokens = max(256, current_tokens // 2)
            attempt += 1
            if attempt < attempts:
                logger.info(f"Retrying with max_output_tokens={current_tokens}")
                print(f"🔄 Retrying Gemini with reduced tokens: {current_tokens}")

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

        # Don't save here - the main send() method handles saving
        
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
                         'electronhub', 'openrouter', 'fireworks', 'xai']
        
        if openai and provider in sdk_compatible:
            # Use OpenAI SDK with custom base URL
            for attempt in range(max_retries):
                try:
                    if self._cancelled:
                        raise UnifiedClientError("Operation cancelled")
                    
                    client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=base_url,
                        timeout=float(self.request_timeout)  # Use configured timeout
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
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = api_delay * 10
                            print(f"{provider} rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    elif attempt < max_retries - 1:
                        print(f"{provider} SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    print(f"{provider} SDK error after all retries: {e}")
                    raise UnifiedClientError(f"{provider} SDK error: {e}")
        else:
            # Use HTTP API with retry logic
            if headers is None:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
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
                headers["Authorization"] = f"Bearer {self.api_key}"
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
                    
                    if resp.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = api_delay * 10
                            print(f"{provider} rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    elif resp.status_code != 200:
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
                    
                except requests.RequestException as e:
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
            base_url="https://dashscope.aliyuncs.com/api/v1",
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
        """
        Send messages with image to vision-capable APIs
        
        REFACTORED VERSION with:
        - Proper o-series model support (o1, o3, o4, etc.)
        - Support for new providers with vision capabilities
        - Better error handling and fallbacks
        - GUI value respect for temperature and tokens
        - Enhanced error handling
        - Better logging
        
        Args:
            messages: List of message dicts
            image_data: Raw image bytes or base64 string
            temperature: Temperature for generation (None = use default)
            max_tokens: Max tokens for non-o models (None = use default)
            max_completion_tokens: Max tokens for o-series models (None = use default)
            context: Context identifier for logging
            
        Returns:
            Tuple of (content, finish_reason)
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
                        self._in_cleanup = True  # Ensure cleanup flag is set
                    print(f"Image processing error: {e}")
                    self._save_failed_request(messages, e, context)
                    raise
            
        except Exception as e:
            # Wrap other errors
            print(f"Unexpected image processing error: {e}", exc_info=True)
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
        return self._send_vertex_model_garden(messages_with_image, temperature, max_tokens)

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
            # Extract response text with error handling
            result = ""
            finish_reason = 'stop'
            
            try:
                if hasattr(response, 'text') and response.text:
                    result = response.text
                    print(f"   ✅ Got text directly: {len(result)} chars")
                else:
                    raise AttributeError("No text attribute or empty text")
            except Exception as e:
                print(f"   ⚠️ Failed to get text directly: {e}")
                
                # Enhanced candidate debugging
                if hasattr(response, 'candidates'):
                    print(f"   🔍 Number of candidates: {len(response.candidates) if response.candidates else 0}")
                    
                    if response.candidates:
                        for i, candidate in enumerate(response.candidates):
                            print(f"   🔍 Checking candidate {i+1}")
                            
                            # Check finish reason
                            if hasattr(candidate, 'finish_reason'):
                                print(f"   🔍 Finish reason: {candidate.finish_reason}")
                            
                            # Check safety ratings
                            if hasattr(candidate, 'safety_ratings'):
                                print(f"   🔍 Safety ratings: {candidate.safety_ratings}")
                            
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                parts = candidate.content.parts
                                print(f"   🔍 Candidate has {len(parts) if parts else 0} parts")
                                
                                if parts:
                                    text_parts = []
                                    for j, part in enumerate(parts):
                                        if hasattr(part, 'text') and part.text:
                                            print(f"   🔍 Part {j+1} has text: {len(part.text)} chars")
                                            text_parts.append(part.text)
                                        else:
                                            print(f"   🔍 Part {j+1} has no text")
                                    
                                    if text_parts:
                                        result = ''.join(text_parts)
                                        print(f"   ✅ Extracted text from candidate {i+1}: {len(result)} chars")
                                        break
                            else:
                                print(f"   ❌ Candidate {i+1} has no content or parts")
                            
                            # Update finish reason
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason_str = str(candidate.finish_reason)
                                if 'MAX_TOKENS' in finish_reason_str:
                                    finish_reason = 'length'
                                elif 'SAFETY' in finish_reason_str:
                                    finish_reason = 'safety'
                    else:
                        print("   ❌ No candidates in response")
                else:
                    print("   ❌ Response has no candidates attribute")
            
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
