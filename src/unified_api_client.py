# unified_api_client.py - REFACTORED with Enhanced Error Handling and Extended AI Model Support
"""
Key Design Principles:
- The client handles API communication and returns accurate status
- Retry logic is implemented in the translation layer (TranslateKRtoEN.py)
- The client must save responses properly for duplicate detection
- The client must return accurate finish_reason for truncation detection
- The client must support cancellation for timeout handling

Supported models and their prefixes:
- OpenAI: gpt*, o1*, o3*, o4*, codex* (e.g., gpt-4, o3, o4-mini)
- Google: gemini*, palm*, bard* (e.g., gemini-2.0-flash-exp)
- Anthropic: claude*, sonnet*, opus*, haiku*
- DeepSeek: deepseek* (e.g., deepseek-chat)
- Mistral: mistral*, mixtral*, codestral*
- Cohere: command*, cohere*
- AI21: j2*, jurassic*, jamba*
- Together AI: llama*, together*, alpaca*, vicuna*, wizardlm*, openchat*
- Perplexity: perplexity*, pplx*
- Replicate: replicate*
- Yi (01.AI): yi* (e.g., yi-34b-chat-200k)
- Qwen (Alibaba): qwen*
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
- TII: falcon*
- Microsoft: phi*, orca*
- Azure: azure* (for Azure OpenAI deployments)
- Aleph Alpha: luminous*
- Databricks: dolly*
- HuggingFace: starcoder*
- Salesforce: codegen*
- BigScience: bloom*
- Meta: opt*, galactica*, llama2*, codellama*

ELECTRONHUB SUPPORT:
ElectronHub is an API aggregator that provides access to multiple models.
To use ElectronHub, prefix your model name with one of these:
- eh/ (e.g., eh/yi-34b-chat-200k)
- electronhub/ (e.g., electronhub/gpt-4)
- electron/ (e.g., electron/claude-3-opus)

ElectronHub allows you to access models from multiple providers using a single API key.

Environment Variables:
- SEND_INTERVAL_SECONDS: Delay between API calls (respects GUI settings)
- YI_API_BASE_URL: Custom endpoint for Yi models (optional)
- ELECTRONHUB_API_URL: Custom ElectronHub endpoint (default: https://api.electronhub.ai/v1)
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint (for azure* models)
- AZURE_API_VERSION: Azure API version (default: 2024-02-01)
- DATABRICKS_API_URL: Databricks workspace URL
- SALESFORCE_API_URL: Salesforce API endpoint

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
from datetime import datetime
import traceback

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
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # ADD THIS LINE
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None 

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
    - electronhub/gpt-4 (OpenAI model via ElectronHub)
    - electron/claude-3-opus (Anthropic model via ElectronHub)
    
    Timeout Behavior:
    - Respects GUI "Auto-retry Slow Chunks" timeout setting
    - If enabled: Uses CHUNK_TIMEOUT (default 900s/15min) from GUI
    - If disabled: Uses 1 hour default timeout
    - The timeout applies to ALL API requests (HTTP and SDK)
    - Works with send_with_interrupt() wrapper for proper retry handling
    """
    
    # Supported model prefixes and their providers
    MODEL_PROVIDERS = {
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
        'j2': 'ai21',
        'jurassic': 'ai21',
        'llama': 'together',
        'together': 'together',
        'perplexity': 'perplexity',
        'pplx': 'perplexity',
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
        'codellama': 'meta',
        # ElectronHub support - prefix for ElectronHub-routed models
        'eh/': 'electronhub',
        'electronhub/': 'electronhub',
        'electron/': 'electronhub',
    }
    
    # Model-specific constraints (for reference and logging only - handled reactively)
    MODEL_CONSTRAINTS = {
        'temperature_fixed': ['o4-mini', 'o1-mini', 'o1-preview'],  # Models that only support specific temperatures
        'no_system_message': ['o1', 'o1-preview', 'o3'],    # Models that don't support system messages
        'max_completion_tokens': ['o4', 'o1', 'o3'],        # Models using max_completion_tokens
        'chinese_optimized': ['qwen', 'yi', 'glm', 'chatglm', 'baichuan', 'ernie', 'hunyuan'],  # Models optimized for Chinese
    }
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.conversation_message_count = 0
        self.pattern_counts = {}
        self.last_pattern = None
        self.context = None
        self.current_session_context = None
        self._cancelled = False
        
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
        
        # Determine client type from model name
        self._setup_client()
            
    def _setup_client(self):
        """Setup the appropriate client based on model type"""
        model_lower = self.model.lower()
        
        # Check model prefixes
        self.client_type = None
        for prefix, provider in self.MODEL_PROVIDERS.items():
            if model_lower.startswith(prefix):
                self.client_type = provider
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
                # Check if it might be an ElectronHub model
                if any(provider in model_lower for provider in ['yi', 'qwen', 'llama', 'gpt', 'claude']):
                    error_msg += f"If using ElectronHub, prefix with 'eh/' (e.g., eh/{self.model}). "
            error_msg += f"Supported prefixes: {list(self.MODEL_PROVIDERS.keys())}"
            raise ValueError(error_msg)
        
        # Initialize provider-specific settings
        if self.client_type == 'openai':
            if openai is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            openai.api_key = self.api_key
            
        elif self.client_type == 'gemini':
            if genai is None:
                raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
            genai.configure(api_key=self.api_key)
            
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
            logger.warning(f"Request might be too long: ~{estimated_tokens} tokens vs {max_tokens} max")
        
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
            logger.error(f"Failed to save stats: {e}")
    
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
        logger.warning(f"Handling empty result for context: {context}, error: {error_info}")
        
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
            logger.error(f"Failed to save payload: {e}")
    
    def _save_response(self, content, filename):
        """Save response for debugging and duplicate detection
        
        IMPORTANT: Responses must be saved in Payloads/ directory for duplicate detection
        """
        filepath = os.path.join("Payloads", filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"Saved response to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save response: {e}")
    
    def cancel_current_operation(self):
        """Mark current operation as cancelled
        
        IMPORTANT: Called by send_with_interrupt when timeout occurs
        """
        self._cancelled = True
        logger.info("🛑 Operation cancelled (timeout or user stop)")
        print("🛑 API operation cancelled")
    
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
        
        self.context = context or self.context
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
            payload_name, response_name = self._get_file_names(messages, context)
            
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
                logger.warning(f"Empty or error response: {response.finish_reason}")
                self._save_failed_request(messages, "Empty response", context, response.raw_response)
                self._track_stats(context, False, "empty_response", time.time() - start_time)
                
                # Use fallback
                fallback_content = self._handle_empty_result(messages, context, response.error_details or "empty")
                return fallback_content, 'error'
            
            # Track success
            self._track_stats(context, True, None, time.time() - start_time)
            
            # Log important info for retry mechanisms
            if response.is_truncated:
                logger.warning(f"Response was truncated: {response.finish_reason}")
                print(f"⚠️ Response truncated (finish_reason: {response.finish_reason})")
                # The calling code will check finish_reason=='length' for retry
            
            # Return the response with accurate finish_reason
            # This is CRITICAL for retry mechanisms to work
            return response.content, response.finish_reason
            
        except UnifiedClientError as e:
            # Handle cancellation specially for timeout support
            if e.error_type == "cancelled" or "cancelled" in str(e):
                logger.info("Propagating cancellation to caller")
                # Re-raise so send_with_interrupt can handle it
                raise
            
            logger.error(f"UnifiedClient error: {e}")
            self._save_failed_request(messages, e, context)
            self._track_stats(context, False, type(e).__name__, time.time() - start_time)
            
            # Return fallback
            fallback_content = self._handle_empty_result(messages, context, str(e))
            return fallback_content, 'error'
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
        else:
            # Other providers don't use max_completion_tokens yet
            return handler(messages, temperature, max_tokens, response_name)
    
    def _send_electronhub(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to ElectronHub API aggregator
        
        ElectronHub provides access to multiple AI models through a unified endpoint.
        Model names should be prefixed with 'eh/', 'electronhub/', or 'electron/'.
        
        Examples:
        - eh/yi-34b-chat-200k
        - electronhub/gpt-4
        - electron/claude-3-opus
        
        Note: ElectronHub uses OpenAI-compatible API format.
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
            logger.warning(f"No ElectronHub prefix found in model '{self.model}', using as-is")
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
            logger.warning("ElectronHub - No system prompt found in messages")
            print("⚠️ ElectronHub: No system prompt in messages")
        
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
            return result
            
        except UnifiedClientError as e:
            # Enhance error messages for common ElectronHub issues
            error_str = str(e)
            
            if "Invalid model" in error_str or "400" in error_str or "model not found" in error_str.lower():
                # Provide helpful error message for invalid models
                error_msg = (
                    f"ElectronHub rejected model '{actual_model}' (original: '{original_model}').\n"
                    f"\nCommon ElectronHub model names:\n"
                    f"  • OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o\n"
                    f"  • Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku\n"
                    f"  • Meta: llama-2-70b-chat, llama-2-13b-chat, llama-2-7b-chat\n"
                    f"  • Mistral: mistral-large, mistral-medium, mixtral-8x7b\n"
                    f"  • Google: gemini-pro\n"
                    f"  • Yi: yi-34b-chat, yi-6b-chat\n"
                    f"  • Others: deepseek-coder-33b, qwen-72b-chat\n"
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
                logger.error(f"ElectronHub API error for model '{actual_model}': {e}")
                raise
                
        finally:
            # Always restore the original model name
            # This ensures subsequent calls work correctly
            self.model = original_model
        
    def _send_openai(self, messages, temperature, max_tokens, max_completion_tokens, response_name) -> UnifiedResponse:
        """Send request to OpenAI API with o-series model support"""
        """Send request to OpenAI API with retry logic"""
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
                params = self._build_openai_params(messages, temperature, max_tokens)
                
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
                resp = openai.chat.completions.create(
                    **params,
                    timeout=self.request_timeout  # Use configured timeout
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
                    logger.warning(f"OpenAI vision API returned empty content with finish_reason='length'")
                    logger.warning(f"This usually means the token limit is too low. Current limit: {api_params.get('max_completion_tokens') or api_params.get('max_tokens', 'not set')}")
                    # Return with error details
                    return UnifiedResponse(
                        content="",
                        finish_reason='error',
                        error_details={'error': 'Response truncated - increase max_completion_tokens', 
                                     'finish_reason': 'length',
                                     'token_limit': api_params.get('max_completion_tokens') or api_params.get('max_tokens')}
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
                    
                    logger.warning(f"Model {self.model} requires temperature={default_temp}, will retry...")
                    fixes_attempted['temperature'] = True
                    fixes_attempted['temperature_override'] = default_temp
                    should_retry = True
                
                # Handle system message constraints reactively
                elif not fixes_attempted['system_message'] and "system" in error_str.lower() and ("not supported" in error_str or "unsupported" in error_str):
                    logger.warning(f"Model {self.model} doesn't support system messages, will convert and retry...")
                    fixes_attempted['system_message'] = True
                    should_retry = True
                
                # Handle max_tokens vs max_completion_tokens reactively
                elif not fixes_attempted['max_tokens_param'] and "max_tokens" in error_str and ("not supported" in error_str or "unsupported" in error_str):
                    logger.warning(f"Model {self.model} requires max_completion_tokens instead of max_tokens, will retry...")
                    fixes_attempted['max_tokens_param'] = True
                    should_retry = True
                
                # Handle rate limits
                elif "rate limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = api_delay * 10
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                
                # If we identified a fix, retry immediately
                if should_retry and attempt < max_retries - 1:
                    time.sleep(api_delay)
                    continue
                
                # Other errors or no retries left
                if attempt < max_retries - 1:
                    logger.warning(f"OpenAI error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(api_delay)
                    continue
                    
                logger.error(f"OpenAI error after all retries: {e}")
                raise UnifiedClientError(f"OpenAI error: {e}", error_type="api_error")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
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
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
                logger.debug(f"Using max_completion_tokens={max_tokens} for o-series model {self.model}")
        else:
            # Regular models use max_tokens
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
                
        return params
    
    def _send_gemini(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Gemini API with enhanced error handling and configurable safety settings"""
        formatted_prompt = self._format_gemini_prompt_simple(messages)
        
        # Check if safety settings are disabled via config
        disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
        
        # Configure safety settings based on toggle
        if disable_safety:
            # Set all safety categories to BLOCK_NONE (most permissive)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            logger.info("Gemini safety settings disabled - using BLOCK_NONE for all categories")
        else:
            # Use default safety settings (let Gemini decide)
            safety_settings = None
            logger.info("Using default Gemini safety settings")
        
        # Create model with safety settings
        model = genai.GenerativeModel(
            self.model,
            safety_settings=safety_settings if safety_settings else None
        )
        # Define BOOST_FACTOR and current_tokens FIRST
        BOOST_FACTOR = 4
        attempts = 4
        attempt = 0
        result = None
        current_tokens = max_tokens * BOOST_FACTOR  # <-- Define current_tokens HERE
        finish_reason = None
        error_details = {}
        
        # SAVE SAFETY CONFIGURATION FOR VERIFICATION
        if safety_settings:
            safety_status = "DISABLED - All categories set to BLOCK_NONE"
            readable_safety = {
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARASSMENT": "BLOCK_NONE",
                "DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        else:
            safety_status = "ENABLED - Using default Gemini safety settings"
            readable_safety = "DEFAULT"
        
        # Log to console
        print(f"🔒 Gemini Safety Status: {safety_status}")
        
        # Save configuration to file
        config_data = {
            "type": "TEXT_REQUEST",
            "model": self.model,
            "safety_enabled": not disable_safety,
            "safety_settings": readable_safety,
            "temperature": temperature,
            "max_output_tokens": current_tokens,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Payloads folder
        os.makedirs("Payloads", exist_ok=True)
        config_filename = f"gemini_safety_{response_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_path = os.path.join("Payloads", config_filename)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            #print(f"📄 Safety config saved: {config_filename}")
        except Exception as e:
            logger.warning(f"Could not save safety config: {e}")
        
        # Log to console for immediate verification
        #if disable_safety and safety_settings:
         #   print(f"✅ SAFETY DISABLED - Request includes safety_settings with BLOCK_NONE")
          #  print(f"   Safety config: {json.dumps(safety_settings, indent=2)}")
        #else:
        #    print(f"⚠️ SAFETY ENABLED - Using default Gemini safety settings")
               


        while attempt < attempts:
            try:
                if self._cancelled:
                    raise UnifiedClientError("Operation cancelled")
                
                response = model.generate_content(
                    formatted_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": current_tokens
                    }
                )
                
                # Check for blocked content
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        error_details['block_reason'] = str(feedback.block_reason)
                        if disable_safety:
                            logger.warning(f"Content blocked despite safety disabled: {feedback.block_reason}")
                        else:
                            logger.warning(f"Content blocked: {feedback.block_reason}")
                        raise Exception(f"Content blocked: {feedback.block_reason}")
                
                # Extract text
                try:
                    result = response.text
                    if not result or result.strip() == "":
                        raise Exception("Empty text in response")
                    finish_reason = 'stop'
                except Exception as text_error:
                    logger.warning(f"Failed to extract text: {text_error}")
                    
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
                
                if result:
                    break
                    
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
                error_details[f'attempt_{attempt+1}'] = str(e)
            
            # Reduce tokens and retry
            current_tokens = max(256, current_tokens // 2)
            attempt += 1
            if attempt < attempts:
                logger.info(f"Retrying with max_output_tokens={current_tokens}")
                print(f"🔄 Retrying Gemini with reduced tokens: {current_tokens}")

        if not result:
            logger.error("All Gemini retries failed")
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
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
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
                        logger.warning(f"Anthropic rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                elif resp.status_code != 200:
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                    if attempt < max_retries - 1:
                        logger.warning(f"Anthropic API error (attempt {attempt + 1}): {error_msg}")
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
                    logger.warning(f"Anthropic API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                logger.error(f"Anthropic API error after all retries: {e}")
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
                        logger.warning(f"Mistral SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    logger.error(f"Mistral SDK error after all retries: {e}")
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
                        logger.warning(f"Cohere SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    logger.error(f"Cohere SDK error after all retries: {e}")
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
                        logger.warning(f"Cohere API error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    logger.error(f"Cohere API error after all retries: {e}")
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
                            logger.warning(f"AI21 rate limit hit, waiting {wait_time}s")
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
                    logger.warning(f"AI21 API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                logger.error(f"AI21 API error after all retries: {e}")
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
        """Send request to Perplexity API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.perplexity.ai",
            response_name=response_name,
            provider="perplexity"
        )
    
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
                        logger.warning(f"Replicate rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                elif resp.status_code != 201:
                    error_msg = f"Replicate API error: {resp.status_code} - {resp.text}"
                    if attempt < max_retries - 1:
                        logger.warning(f"{error_msg} (attempt {attempt + 1})")
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
                    logger.warning(f"Replicate API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                logger.error(f"Replicate API error after all retries: {e}")
                raise UnifiedClientError(f"Replicate API error: {e}")
    
    def _send_deepseek(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to DeepSeek API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.deepseek.com/v1",
            response_name=response_name,
            provider="deepseek"
        )
    
    def _send_openai_compatible(self, messages, temperature, max_tokens, base_url, 
                                response_name, provider="generic") -> UnifiedResponse:
        """Send request to OpenAI-compatible APIs"""
        max_retries = 3
        api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        # Debug logging for ElectronHub
        if provider == "electronhub":
            logger.debug(f"ElectronHub API call - Messages structure:")
            for i, msg in enumerate(messages):
                logger.debug(f"  Message {i}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}")
                if msg.get('role') == 'system':
                    logger.debug(f"  System prompt preview: {msg.get('content', '')[:100]}...")
        
        # Use OpenAI SDK for providers known to work well with it
        sdk_compatible = ['deepseek', 'together', 'mistral', 'yi', 'qwen', 'moonshot', 'groq', 'electronhub']
        
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
                    
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    content = resp.choices[0].message.content
                    finish_reason = resp.choices[0].finish_reason
                    
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
                            logger.warning(f"{provider} rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    elif attempt < max_retries - 1:
                        logger.warning(f"{provider} SDK error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    logger.error(f"{provider} SDK error after all retries: {e}")
                    raise UnifiedClientError(f"{provider} SDK error: {e}")
        else:
            # Use HTTP API with retry logic
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
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
                            logger.warning(f"{provider} rate limit hit, waiting {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    elif resp.status_code != 200:
                        error_msg = f"{provider} API error: {resp.status_code} - {resp.text}"
                        if attempt < max_retries - 1:
                            logger.warning(f"{error_msg} (attempt {attempt + 1})")
                            time.sleep(api_delay)
                            continue
                        raise UnifiedClientError(error_msg, http_status=resp.status_code)
                    
                    json_resp = resp.json()
                    choices = json_resp.get("choices", [])
                    
                    if not choices:
                        raise UnifiedClientError(f"{provider} API returned no choices")
                    
                    content = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason", "stop")
                    
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
                        logger.warning(f"{provider} API error (attempt {attempt + 1}): {e}")
                        time.sleep(api_delay)
                        continue
                    logger.error(f"{provider} API error after all retries: {e}")
                    raise UnifiedClientError(f"{provider} API error: {e}")
    
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
            payload_name, response_name = self._get_file_names(messages, context)
            
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
                logger.warning(f"Empty response from {self.client_type}")
                fallback = self._handle_empty_result(messages, context, "empty_image_response")
                return fallback, 'error'
            
            # Log truncation for retry mechanism
            if response.is_truncated:
                logger.warning(f"Image response was truncated: {response.finish_reason}")
                print(f"⚠️ Image response truncated (finish_reason: {response.finish_reason})")
                
            return response.content, response.finish_reason
                
        except UnifiedClientError as e:
            # Re-raise our own errors
            logger.error(f"Image processing error: {e}")
            self._save_failed_request(messages, e, context)
            raise
            
        except Exception as e:
            # Wrap other errors
            logger.error(f"Unexpected image processing error: {e}", exc_info=True)
            self._save_failed_request(messages, e, context)
            fallback = self._handle_empty_result(messages, context, str(e))
            return fallback, 'error'

    def _is_o_series_model(self) -> bool:
        """Check if the current model is an o-series model (o1, o3, o4, etc.)"""
        if not self.model:
            return False
        
        model_lower = self.model.lower()
        
        # Check for specific patterns
        if 'o1-preview' in model_lower:
            return True
        
        # Check if it starts with o followed by a digit
        if len(model_lower) >= 2 and model_lower[0] == 'o' and model_lower[1].isdigit():
            return True
        
        return False

    def _send_gemini_image(self, messages, image_base64, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send image request to Gemini API with configurable safety settings"""
        try:
            # Format prompt
            formatted_parts = []
            for msg in messages:
                if msg.get('role') == 'system':
                    formatted_parts.append(f"Instructions: {msg['content']}")
                elif msg.get('role') == 'user':
                    formatted_parts.append(f"User: {msg['content']}")
            
            text_prompt = "\n\n".join(formatted_parts)
            
            # Check if safety settings are disabled
            disable_safety = os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true"
            
            # Configure safety settings
            if disable_safety:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = None
            
            model = genai.GenerativeModel(
                self.model,
                safety_settings=safety_settings if safety_settings else None
            )
             # SAVE SAFETY CONFIGURATION FOR VERIFICATION
            if safety_settings:
                safety_status = "DISABLED - All categories set to BLOCK_NONE"
                readable_safety = {
                    "HATE_SPEECH": "BLOCK_NONE",
                    "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARASSMENT": "BLOCK_NONE",
                    "DANGEROUS_CONTENT": "BLOCK_NONE"
                }
            else:
                safety_status = "ENABLED - Using default Gemini safety settings"
                readable_safety = "DEFAULT"
            
            # Log to console
            print(f"\n📷 Gemini Image Safety Status: {safety_status}")
            
            # Save configuration to file
            config_data = {
                "type": "IMAGE_REQUEST",
                "model": self.model,
                "safety_enabled": not disable_safety,
                "safety_settings": readable_safety,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "has_image": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to Payloads folder
            os.makedirs("Payloads", exist_ok=True)
            config_filename = f"gemini_image_safety_{response_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            config_path = os.path.join("Payloads", config_filename)
            
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2)
               # print(f"📄 Image safety config saved: {config_filename}")
            except Exception as e:
                logger.warning(f"Could not save image safety config: {e}")    
                
            # Decode image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))

            start_time = time.time()
            response = model.generate_content(
                [text_prompt, image],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            elapsed = time.time() - start_time
            logger.info(f"Vision API call took {elapsed:.1f} seconds")
            
            # Extract response with error handling
            try:
                result = response.text
                finish_reason = 'stop'
            except Exception as e:
                logger.warning(f"Failed to extract image response text: {e}")
                result = ""
                finish_reason = 'error'
                
                # Try extracting from candidates with improved logic
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            candidate_text = ''.join(part.text for part in parts if hasattr(part, 'text'))
                            if candidate_text:
                                result = candidate_text
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = str(candidate.finish_reason)
                                    if 'MAX_TOKENS' in finish_reason:
                                        finish_reason = 'length'
                                break
            
            return UnifiedResponse(
                content=result,
                finish_reason=finish_reason,
                raw_response=response,
                usage=None  # Gemini doesn't provide usage info for images
            )
            
        except Exception as e:
            logger.error(f"Gemini image processing error: {e}")
            error_msg = str(e)
            
            # Check for safety blocks
            if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                if disable_safety:
                    logger.warning("Content blocked despite safety settings disabled")
                return UnifiedResponse(
                    content="",
                    finish_reason='safety',
                    error_details={'error': error_msg}
                )
            
            return UnifiedResponse(
                content="",
                finish_reason='error',
                error_details={'error': error_msg}
        )
            
        except Exception as e:
            logger.error(f"Gemini image API error: {e}")
            raise UnifiedClientError(f"Gemini image API error: {e}")
    

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
                
                response = openai.chat.completions.create(**api_params)
                
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
                    logger.error(f"Token parameter error for {self.model}: {error_msg}")
                    logger.info("Retrying without token limits...")
                    
                    # Build new params without token limits
                    retry_params = {
                        "model": self.model,
                        "messages": vision_messages,
                        "temperature": temperature
                    }
                    
                    try:
                        response = openai.chat.completions.create(**retry_params)
                        content = response.choices[0].message.content
                        finish_reason = response.choices[0].finish_reason
                        
                        return UnifiedResponse(
                            content=content,
                            finish_reason=finish_reason,
                            usage=None,
                            raw_response=response
                        )
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        raise UnifiedClientError(f"OpenAI Vision API error: {retry_error}")
                
                logger.error(f"OpenAI Vision API error: {e}")
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
            logger.error(f"Anthropic Vision API error: {e}")
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
                        logger.warning(f"{error_msg} (attempt {attempt + 1})")
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
                    logger.warning(f"ElectronHub Vision API error (attempt {attempt + 1}): {e}")
                    time.sleep(api_delay)
                    continue
                logger.error(f"ElectronHub Vision API error after all retries: {e}")
                raise UnifiedClientError(f"ElectronHub Vision API error: {e}")
   
    def generate_image(self, prompt: str, size: str = "1024x1024", n: int = 1, 
                      quality: str = "standard", style: str = "vivid", 
                      model: str = "dall-e-3") -> Dict[str, Any]:
        """
        Generate images using DALL-E
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (1024x1024, 1024x1792, 1792x1024 for DALL-E 3)
            n: Number of images to generate (1 for DALL-E 3, 1-10 for DALL-E 2)
            quality: "standard" or "hd" (DALL-E 3 only)
            style: "vivid" or "natural" (DALL-E 3 only)
            model: "dall-e-3" or "dall-e-2"
        
        Returns:
            Dict with 'images' list containing URLs or base64 data
        """
        if self.client_type != 'openai':
            raise UnifiedClientError(f"DALL-E is only available for OpenAI clients, not {self.client_type}")
        
        if not openai:
            raise UnifiedClientError("OpenAI library not installed. Install with: pip install openai")
        
        try:
            # Initialize OpenAI client if needed
            client = openai.OpenAI(api_key=self.api_key, timeout=float(self.request_timeout))
            
            # Adjust parameters based on model
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": n
            }
            
            if model == "dall-e-3":
                params["quality"] = quality
                params["style"] = style
                if n > 1:
                    logger.warning("DALL-E 3 only supports n=1, adjusting")
                    params["n"] = 1
            
            # Make the API call
            response = client.images.generate(**params)
            
            # Extract image data
            images = []
            for img_data in response.data:
                image_info = {
                    "url": img_data.url if hasattr(img_data, 'url') else None,
                    "b64_json": img_data.b64_json if hasattr(img_data, 'b64_json') else None,
                    "revised_prompt": img_data.revised_prompt if hasattr(img_data, 'revised_prompt') else None
                }
                images.append(image_info)
            
            # Track stats
            self.stats['total_requests'] += 1
            
            return {
                "images": images,
                "model": model,
                "created": datetime.now().isoformat()
            }
            
        except openai.OpenAIError as e:
            logger.error(f"DALL-E generation error: {e}")
            self.stats['errors']['dalle_generate'] = self.stats['errors'].get('dalle_generate', 0) + 1
            raise UnifiedClientError(f"DALL-E generation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected DALL-E error: {e}")
            raise UnifiedClientError(f"Unexpected DALL-E error: {e}")
    
    def edit_image(self, image: Any, mask: Any, prompt: str, 
                   size: str = "1024x1024", n: int = 1, model: str = "dall-e-2") -> Dict[str, Any]:
        """
        Edit images using DALL-E (image editing is primarily supported by dall-e-2)
        
        Args:
            image: PIL Image, file path, or bytes of the image to edit
            mask: PIL Image, file path, or bytes of the mask (transparent areas will be edited)
            prompt: Text description of what should be generated in the masked area
            size: Output size (256x256, 512x512, or 1024x1024)
            n: Number of images to generate (1-10)
            model: Model to use (default: dall-e-2, but configurable for future models)
        
        Returns:
            Dict with 'images' list containing URLs or base64 data
        """
        if self.client_type != 'openai':
            raise UnifiedClientError(f"DALL-E is only available for OpenAI clients, not {self.client_type}")
        
        if not openai:
            raise UnifiedClientError("OpenAI library not installed. Install with: pip install openai")
        
        try:
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=self.api_key, timeout=float(self.request_timeout))
            
            # Convert image to bytes if needed
            logger.info(f"Preparing image for DALL-E edit (type: {type(image)})")
            image_bytes = self._prepare_image_bytes(image, "image")
            
            logger.info(f"Preparing mask for DALL-E edit (type: {type(mask)})")
            mask_bytes = self._prepare_image_bytes(mask, "mask")
            
            # Log details for debugging
            logger.debug(f"Image bytes size: {len(image_bytes.getvalue())} bytes")
            logger.debug(f"Mask bytes size: {len(mask_bytes.getvalue())} bytes")
            logger.debug(f"Using model: {model}, size: {size}")
            
            # Make the API call with specified model
            response = client.images.edit(
                model=model,  # Use the configurable model
                image=image_bytes,
                mask=mask_bytes,
                prompt=prompt,
                size=size,
                n=n
            )
            
            # Extract image data
            images = []
            for img_data in response.data:
                image_info = {
                    "url": img_data.url if hasattr(img_data, 'url') else None,
                    "b64_json": img_data.b64_json if hasattr(img_data, 'b64_json') else None
                }
                images.append(image_info)
            
            # Track stats
            self.stats['total_requests'] += 1
            
            logger.info(f"DALL-E edit successful, received {len(images)} images")
            
            return {
                "images": images,
                "model": model,
                "operation": "edit",
                "created": datetime.now().isoformat()
            }
            
        except openai.OpenAIError as e:
            logger.error(f"DALL-E edit error with model {model}: {e}")
            self.stats['errors']['dalle_edit'] = self.stats['errors'].get('dalle_edit', 0) + 1
            raise UnifiedClientError(f"DALL-E edit error: {e}")
        except Exception as e:
            logger.error(f"Unexpected DALL-E edit error with model {model}: {e}")
            raise UnifiedClientError(f"Unexpected DALL-E edit error: {e}")
    
    def create_variation(self, image: Any, size: str = "1024x1024", n: int = 1) -> Dict[str, Any]:
        """
        Create variations of an image using DALL-E 2
        
        Args:
            image: PIL Image, file path, or bytes of the image to create variations of
            size: Output size (256x256, 512x512, or 1024x1024)
            n: Number of variations to generate (1-10)
        
        Returns:
            Dict with 'images' list containing URLs or base64 data
        """
        if self.client_type != 'openai':
            raise UnifiedClientError(f"DALL-E is only available for OpenAI clients, not {self.client_type}")
        
        if not openai:
            raise UnifiedClientError("OpenAI library not installed. Install with: pip install openai")
        
        try:
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=self.api_key, timeout=float(self.request_timeout))
            
            # Convert image to bytes if needed
            image_bytes = self._prepare_image_bytes(image, "image")
            
            # Make the API call
            response = client.images.create_variation(
                model="dall-e-2",
                image=image_bytes,
                size=size,
                n=n
            )
            
            # Extract image data
            images = []
            for img_data in response.data:
                image_info = {
                    "url": img_data.url if hasattr(img_data, 'url') else None,
                    "b64_json": img_data.b64_json if hasattr(img_data, 'b64_json') else None
                }
                images.append(image_info)
            
            # Track stats
            self.stats['total_requests'] += 1
            
            return {
                "images": images,
                "model": "dall-e-2",
                "operation": "variation",
                "created": datetime.now().isoformat()
            }
            
        except openai.OpenAIError as e:
            logger.error(f"DALL-E variation error: {e}")
            self.stats['errors']['dalle_variation'] = self.stats['errors'].get('dalle_variation', 0) + 1
            raise UnifiedClientError(f"DALL-E variation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected DALL-E variation error: {e}")
            raise UnifiedClientError(f"Unexpected DALL-E variation error: {e}")
    
    def _prepare_image_bytes(self, image: Any, image_type: str = "image") -> io.BytesIO:
        """
        Convert various image inputs to bytes for DALL-E API
        
        Args:
            image: PIL Image, file path string, or bytes
            image_type: "image" or "mask" for error messages
        
        Returns:
            BytesIO object ready for API
        """
        try:
            if isinstance(image, str):
                # File path
                with Image.open(image) as img:
                    img_bytes = io.BytesIO()
                    # Always save as PNG for DALL-E
                    if img.mode == 'RGBA':
                        img.save(img_bytes, format='PNG')
                    else:
                        # Convert to RGB for non-transparent images
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Save as PNG (DALL-E prefers PNG)
                        img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    # Set proper name for the BytesIO object
                    img_bytes.name = f'{image_type}.png'
                    return img_bytes
            
            elif isinstance(image, bytes):
                # Already bytes - load and ensure it's PNG
                img = Image.open(io.BytesIO(image))
                img_bytes = io.BytesIO()
                if img.mode == 'RGBA':
                    img.save(img_bytes, format='PNG')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                img_bytes.name = f'{image_type}.png'
                return img_bytes
            
            elif hasattr(image, 'save'):
                # PIL Image or similar
                img_bytes = io.BytesIO()
                # Ensure RGBA for masks, RGB for images
                if image_type == "mask":
                    if image.mode != 'RGBA' and image.mode != 'L':
                        # Convert to RGBA for masks
                        if image.mode == 'L':
                            # Grayscale mask - keep as is
                            image.save(img_bytes, format='PNG')
                        else:
                            image = image.convert('RGBA')
                            image.save(img_bytes, format='PNG')
                    else:
                        image.save(img_bytes, format='PNG')
                else:
                    # Regular image
                    if image.mode == 'RGBA':
                        image.save(img_bytes, format='PNG')
                    else:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.save(img_bytes, format='PNG')
                
                img_bytes.seek(0)
                img_bytes.name = f'{image_type}.png'
                return img_bytes
            
            elif isinstance(image, io.BytesIO):
                # Already BytesIO - ensure it has a name
                image.seek(0)
                # Load and re-save as PNG to ensure format
                img = Image.open(image)
                img_bytes = io.BytesIO()
                if img.mode == 'RGBA' or (image_type == "mask" and img.mode == 'L'):
                    img.save(img_bytes, format='PNG')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                img_bytes.name = f'{image_type}.png'
                return img_bytes
            
            else:
                raise ValueError(f"Unsupported {image_type} type: {type(image)}")
                
        except Exception as e:
            raise UnifiedClientError(f"Failed to prepare {image_type} for DALL-E: {e}")
    
    def download_dalle_image(self, url: str, save_path: Optional[str] = None) -> bytes:
        """
        Download image from DALL-E URL
        
        Args:
            url: The image URL from DALL-E response
            save_path: Optional path to save the image
            
        Returns:
            Image bytes
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            image_bytes = response.content
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
                logger.info(f"Saved DALL-E image to: {save_path}")
            
            return image_bytes
            
        except requests.RequestException as e:
            raise UnifiedClientError(f"Failed to download DALL-E image: {e}")

            
    # Additional provider methods for extended model support
    def _send_yi(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Yi API (01.AI)
        
        Note: If you're using a custom Yi deployment or different endpoint,
        you may need to modify the base_url. The official Yi API is at:
        https://api.lingyiwanwu.com/v1
        
        Some alternative endpoints:
        - Local deployment: http://localhost:8000/v1
        - Custom cloud: https://your-yi-endpoint.com/v1
        
        IMPORTANT: Yi API requests will timeout based on your GUI settings:
        - With "Auto-retry Slow Chunks" enabled: Your configured timeout (e.g., 900s)
        - Without it: 1 hour default timeout
        This prevents hanging on slow/unresponsive endpoints.
        """
        # Check for custom Yi endpoint in environment
        custom_yi_url = os.getenv("YI_API_BASE_URL")
        base_url = custom_yi_url if custom_yi_url else "https://api.lingyiwanwu.com/v1"
        
        if custom_yi_url:
            logger.info(f"Using custom Yi API endpoint: {base_url}")
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=base_url,
            response_name=response_name,
            provider="yi"
        )
    
    def _send_qwen(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Qwen API (Alibaba)"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
        """Send request to Zhipu AI (GLM models)"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            response_name=response_name,
            provider="zhipu"
        )
    
    def _send_moonshot(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Moonshot AI (Kimi)"""
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
            base_url="https://api.groq.com/openai/v1",
            response_name=response_name,
            provider="groq"
        )
    
    def _send_baidu(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Baidu ERNIE API"""
        # Baidu has a different API structure, but we can try OpenAI-compatible endpoint
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1",
            response_name=response_name,
            provider="baidu"
        )
    
    def _send_tencent(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Tencent Hunyuan API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://hunyuan.tencentcloudapi.com",
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
            base_url="https://ark.cn-beijing.volcanicengine.com/api/v3",
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
            base_url="https://api.sensenova.com/v1",
            response_name=response_name,
            provider="sensenova"
        )
    
    def _send_internlm(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to InternLM API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1",
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
        """Send request to Microsoft Phi API"""
        # Note: Phi models are often accessed through Azure OpenAI
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.microsoft.com/v1",  # Placeholder - usually through Azure
            response_name=response_name,
            provider="microsoft"
        )
    
    def _send_azure(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Azure OpenAI"""
        # Azure needs special handling - endpoint from environment
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
        api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
        
        base_url = f"{azure_endpoint}/openai/deployments/{self.model}"
        
        # Azure uses different auth header
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=base_url,
            response_name=response_name,
            provider="azure"
        )
    
    def _send_google_palm(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Google PaLM/Bard API"""
        # PaLM is being deprecated in favor of Gemini
        logger.warning("PaLM API is deprecated. Consider using Gemini models instead.")
        return self._send_gemini(messages, temperature, max_tokens, response_name)
    
    def _send_alephalpha(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Aleph Alpha API"""
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api.aleph-alpha.com/v1",
            response_name=response_name,
            provider="alephalpha"
        )
    
    def _send_databricks(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Databricks API"""
        # Databricks endpoint from environment
        databricks_url = os.getenv("DATABRICKS_API_URL", "https://your-workspace.databricks.com/api/2.0")
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=f"{databricks_url}/serving-endpoints/{self.model}/invocations",
            response_name=response_name,
            provider="databricks"
        )
    
    def _send_huggingface(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to HuggingFace Inference API"""
        # HuggingFace models typically through Inference API
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url="https://api-inference.huggingface.co/models",
            response_name=response_name,
            provider="huggingface"
        )
    
    def _send_salesforce(self, messages, temperature, max_tokens, response_name) -> UnifiedResponse:
        """Send request to Salesforce CodeGen API"""
        # Salesforce models often through custom endpoints
        salesforce_url = os.getenv("SALESFORCE_API_URL", "https://api.salesforce.com/v1")
        
        return self._send_openai_compatible(
            messages, temperature, max_tokens,
            base_url=salesforce_url,
            response_name=response_name,
            provider="salesforce"
        )
