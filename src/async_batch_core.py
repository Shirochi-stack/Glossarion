# async_batch_core.py
"""GUI-free core of Glossarion's async (batch API, 50% discount) processing.

Moved in milestone U7 out of ``async_api_processor.py`` (which re-exports every name below and
keeps only the Qt dialog as a thin view):

* ``AsyncAPIStatus``, ``AsyncJobInfo`` and ``AsyncAPIProcessor`` (request building per provider,
  submission, status checks, result download and the job list file), byte for byte, except that
  ``AsyncAPIProcessor(gui, jobs_file=None)`` takes the job list path as a parameter (the default
  is unchanged: ``async_jobs.json`` next to this module);
* ``AsyncBatchJobMixin``: the ``AsyncProcessingDialog`` workflow methods (prepare the run
  environment, extract chapters, submit, poll, cancel, estimate, download and apply the results
  to the output workspace). Their bodies are the dialog's, with each Qt call replaced by a hook
  (``_async_msgbox``, ``_async_single_shot``, ``_async_set_cost_info``, ...; the table is
  ``tests/test_async_batch_core.py::SUBSTITUTIONS``). The mixin's hooks are GUI-free; the desktop
  dialog overrides them with the original Qt statements, so desktop behaviour is unchanged;
* the dialog's job-row / model-status / auto-refresh logic (``job_display_row``,
  ``selected_job_progress``, ``async_support_status``, ``gui_model_name``,
  ``refresh_pending_job_statuses``);
* ``HeadlessAsyncBatch``: the workflow without Qt (Glossarion Mobile's Tools > Async batch):
  questions go to ``host.ask('async_batch_question', ...)`` and notices to
  ``host.emit('async_batch_message', ...)``.

Python 3.10 compatible; never imports PySide6, translator_gui or dpi_setup.
"""

import os
import sys
import re
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import json
import time
import threading
import logging
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import uuid
from pathlib import Path
from html_output_utils import ensure_utf8_html_document
from epub_package import find_epub_opf_member

try:
    from antigravity_proxy import clamp_output_tokens_for_model as _clamp_antigravity_output_tokens
except Exception:
    def _clamp_antigravity_output_tokens(model, max_tokens, default=8192):
        try:
            requested = int(max_tokens)
        except Exception:
            requested = int(default)
        if requested <= 0:
            return requested
        model_lower = str(model or "").lower()
        if "claude" in model_lower:
            return min(requested, 64000)
        if "gemini" in model_lower:
            return min(requested, 64000)
        return requested


def _is_antigravity_model_name(model) -> bool:
    return str(model or "").strip().lower().startswith("antigravity")


def _clamp_output_tokens_for_selected_model(model, max_tokens, default=65536):
    if _is_antigravity_model_name(model):
        return _clamp_antigravity_output_tokens(model, max_tokens, default=default)
    try:
        return int(max_tokens)
    except Exception:
        return int(default)

try:
    import tiktoken
except ImportError:
    tiktoken = None
    
# For TXT file processing
try:
    from txt_processor import TextFileProcessor
except ImportError:
    TextFileProcessor = None
    print("txt_processor not available - TXT file support disabled")
# For provider-specific implementations
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


#: The desktop logger name is kept: the log file's format prints ``%(name)s``.
logger = logging.getLogger("async_api_processor")

class AsyncAPIStatus(Enum):
    """Status states for async API jobs"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass
class AsyncJobInfo:
    """Information about an async API job"""
    job_id: str
    provider: str
    model: str
    status: AsyncAPIStatus
    created_at: datetime
    updated_at: datetime
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    cost_estimate: float = 0.0
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncJobInfo':
        """Create from dictionary"""
        data['status'] = AsyncAPIStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('metadata') is None:
            data['metadata'] = {}
        return cls(**data)

class AsyncAPIProcessor:
    """Handles asynchronous batch API processing for supported providers"""
    
    # Provider configurations
    PROVIDER_CONFIGS = {
        'gemini': {
            'batch_endpoint': 'native_sdk',  # Uses native SDK instead of REST
            'status_endpoint': 'native_sdk',
            'max_requests_per_batch': 10000,
            'supports_chunking': False,
            'discount': 0.5,
            'available': True  # Now available!
        },
        'anthropic': {
            'batch_endpoint': 'https://api.anthropic.com/v1/messages/batches',
            'status_endpoint': 'https://api.anthropic.com/v1/messages/batches/{job_id}',
            'max_requests_per_batch': 10000,
            'supports_chunking': False,
            'discount': 0.5
        },
        'openai': {
            'batch_endpoint': 'https://api.openai.com/v1/batches',
            'status_endpoint': 'https://api.openai.com/v1/batches/{job_id}',
            'cancel_endpoint': 'https://api.openai.com/v1/batches/{job_id}/cancel',
            'max_requests_per_batch': 50000,
            'supports_chunking': False,
            'discount': 0.5
        },
        'mistral': {
            'batch_endpoint': 'https://api.mistral.ai/v1/batch/jobs',
            'status_endpoint': 'https://api.mistral.ai/v1/batch/jobs/{job_id}',
            'max_requests_per_batch': 10000,
            'supports_chunking': False,
            'discount': 0.5
        },
        'bedrock': {
            'batch_endpoint': 'batch-inference',  # AWS SDK specific
            'max_requests_per_batch': 10000,
            'supports_chunking': False,
            'discount': 0.5
        },
        'groq': {
            'batch_endpoint': 'https://api.groq.com/openai/v1/batch',
            'status_endpoint': 'https://api.groq.com/openai/v1/batch/{job_id}',
            'max_requests_per_batch': 1000,
            'supports_chunking': False,
            'discount': 0.5
        }
    }
    
    def __init__(self, gui_instance, jobs_file=None):
        """Initialize the async processor
        
        Args:
            gui_instance: Reference to TranslatorGUI instance
            jobs_file: Job list file (default: async_jobs.json next to this module, as
                before the U7 move; Glossarion Mobile passes its app-data file)
        """
        self.gui = gui_instance
        self.jobs_file = jobs_file or os.path.join(os.path.dirname(__file__), 'async_jobs.json')
        self.jobs: Dict[str, AsyncJobInfo] = {}
        self.stop_flag = threading.Event()
        self.processing_thread = None
        self._load_jobs()
        
    def _load_jobs(self):
        """Load saved async jobs from file"""
        try:
            if os.path.exists(self.jobs_file):
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for job_id, job_data in data.items():
                        try:
                            self.jobs[job_id] = AsyncJobInfo.from_dict(job_data)
                        except Exception as e:
                            print(f"Failed to load job {job_id}: {e}")
        except Exception as e:
            print(f"Failed to load async jobs: {e}")
            
    def _save_jobs(self):
        """Save async jobs to file"""
        try:
            data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            with open(self.jobs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save async jobs: {e}")
            
    def get_provider_from_model(self, model: str) -> Optional[str]:
        """Determine provider from model name"""
        model_lower = model.lower()
        
        # Check prefixes
        if model_lower.startswith(('gpt', 'o1', 'o3', 'o4')):
            return 'openai'
        elif model_lower.startswith('gemini'):
            return 'gemini'
        elif model_lower.startswith(('claude', 'sonnet', 'opus', 'haiku')):
            return 'anthropic'
        elif model_lower.startswith((
            'mistral', 'open-mistral', 'mixtral', 'codestral', 'devstral',
            'pixtral', 'voxtral', 'magistral', 'ministral', 'labs-leanstral'
        )):
            return 'mistral'
        elif model_lower.startswith('groq'):
            return 'groq'
        elif model_lower.startswith('bedrock'):
            return 'bedrock'
            
        # Check for aggregator prefixes that might support async
        if model_lower.startswith(('eh/', 'electronhub/', 'electron/', 'authgpt/', 'authgem/', 'authgem-key/', 'authgem-vertex/')):
            # Extract actual model after prefix
            actual_model = model.split('/', 1)[1] if '/' in model else model
            return self.get_provider_from_model(actual_model)
            
        return None
        
    def supports_async(self, model: str) -> bool:
        """Check if model supports async processing"""
        provider = self.get_provider_from_model(model)
        return provider in self.PROVIDER_CONFIGS
        
    def estimate_cost(self, num_chapters: int, avg_tokens_per_chapter: int, model: str, compression_factor: float = 1.0) -> Tuple[float, float]:
        """Estimate costs for async vs regular processing
        
        Returns:
            Tuple of (async_cost, regular_cost)
        """
        provider = self.get_provider_from_model(model)
        if not provider:
            return (0.0, 0.0)
            
        # UPDATED PRICING AS OF JULY 2025
        # Prices are (input_price, output_price) per 1M tokens
        token_prices = {
            'openai': {
                # GPT-5 Series (Standard pricing per 1M tokens)
                'gpt-5.2-pro': (21.0, 168.0),
                'gpt-5-pro': (15.0, 120.0),
                'gpt-5.2': (1.75, 14.0),
                'gpt-5.1': (1.25, 10.0),
                'gpt-5': (1.25, 10.0),
                'gpt-5-mini': (0.25, 2.0),
                'gpt-5-nano': (0.05, 0.40),
                'gpt-5.2-chat-latest': (1.75, 14.0),
                'gpt-5.1-chat-latest': (1.25, 10.0),
                'gpt-5-chat-latest': (1.25, 10.0),
                'gpt-5.1-codex-max': (1.25, 10.0),
                'gpt-5.1-codex': (1.25, 10.0),
                'gpt-5-codex': (1.25, 10.0),
                # GPT-4.1 Series (Latest - June 2024 knowledge)
                'gpt-4.1': (2.0, 8.0),
                'gpt-4.1-mini': (0.4, 1.6),
                'gpt-4.1-nano': (0.1, 0.4),
                
                # GPT-4.5 Preview
                'gpt-4.5-preview': (75.0, 150.0),
                
                # GPT-4o Series
                'gpt-4o': (2.5, 10.0),
                'gpt-4o-mini': (0.15, 0.6),
                'gpt-4o-audio': (2.5, 10.0),
                'gpt-4o-audio-preview': (2.5, 10.0),
                'gpt-4o-realtime': (5.0, 20.0),
                'gpt-4o-realtime-preview': (5.0, 20.0),
                'gpt-4o-mini-audio': (0.15, 0.6),
                'gpt-4o-mini-audio-preview': (0.15, 0.6),
                'gpt-4o-mini-realtime': (0.6, 2.4),
                'gpt-4o-mini-realtime-preview': (0.6, 2.4),
                
                # GPT-4 Legacy
                'gpt-4': (30.0, 60.0),
                'gpt-4-turbo': (10.0, 30.0),
                'gpt-4-32k': (60.0, 120.0),
                'gpt-4-0613': (30.0, 60.0),
                'gpt-4-0314': (30.0, 60.0),
                
                # GPT-3.5
                'gpt-3.5-turbo': (0.5, 1.5),
                'gpt-3.5-turbo-instruct': (1.5, 2.0),
                'gpt-3.5-turbo-16k': (3.0, 4.0),
                'gpt-3.5-turbo-0125': (0.5, 1.5),
                
                # O-series Reasoning Models (NOT batch compatible usually)
                'o1': (15.0, 60.0),
                'o1-pro': (150.0, 600.0),
                'o1-mini': (1.1, 4.4),
                'o3': (1.0, 4.0),
                'o3-pro': (20.0, 80.0),
                'o3-deep-research': (10.0, 40.0),
                'o3-mini': (1.1, 4.4),
                'o4-mini': (1.1, 4.4),
                'o4-mini-deep-research': (2.0, 8.0),
                
                # Special models
                'chatgpt-4o-latest': (5.0, 15.0),
                'computer-use-preview': (3.0, 12.0),
                'gpt-4o-search-preview': (2.5, 10.0),
                'gpt-4o-mini-search-preview': (0.15, 0.6),
                'codex-mini-latest': (1.5, 6.0),
                
                # Small models
                'davinci-002': (2.0, 2.0),
                'babbage-002': (0.4, 0.4),
                
                'default': (2.5, 10.0)
            },
            'anthropic': {
                # Claude 4.5 series
                'claude-opus-4.5': (5.0, 25.0),
                'claude-opus-4.1': (15.0, 75.0),
                'claude-opus-4': (15.0, 75.0),
                # Claude Sonnet 4.x
                'claude-sonnet-4.5': (3.0, 15.0),
                'claude-sonnet-4': (3.0, 15.0),
                'claude-sonnet-3.7': (3.0, 15.0),
                # Claude Haiku 4.x / 3.x
                'claude-haiku-4.5': (1.0, 5.0),
                'claude-haiku-3.5': (0.80, 4.0),
                'claude-haiku-3': (0.25, 1.25),
                # Deprecated / legacy
                'claude-opus-3': (15.0, 75.0),
                'claude-2.1': (8.0, 24.0),
                'claude-2': (8.0, 24.0),
                'claude-instant': (0.8, 2.4),
                'default': (3.0, 15.0)
            },
            'gemini': {
                # Gemini 3 Series (Preview pricing from Google AI Studio)
                'gemini-3-pro-preview': (2.0, 12.0),       # ≤200k tokens tier
                'gemini-3-pro': (2.0, 12.0),
                'gemini-3-flash-preview': (0.5, 3.0),      # text/image/video tier
                'gemini-3-flash': (0.5, 3.0),
                # Gemini 2.5 Series (Latest)
                'gemini-2.5-pro': (1.25, 10.0),      # ≤200k tokens
                'gemini-2.5-flash': (0.3, 2.5),
                'gemini-2.5-flash-lite': (0.1, 0.4),
                'gemini-2.5-flash-lite-preview': (0.1, 0.4),
                'gemini-2.5-flash-lite-preview-06-17': (0.1, 0.4),
                'gemini-2.5-flash-native-audio': (0.5, 12.0),  # Audio output
                'gemini-2.5-flash-preview-native-audio-dialog': (0.5, 12.0),
                'gemini-2.5-flash-exp-native-audio-thinking-dialog': (0.5, 12.0),
                'gemini-2.5-flash-preview-tts': (0.5, 10.0),
                'gemini-2.5-pro-preview-tts': (1.0, 20.0),
                
                # Gemini 2.0 Series
                'gemini-2.0-flash': (0.1, 0.4),
                'gemini-2.0-flash-lite': (0.075, 0.3),
                'gemini-2.0-flash-live': (0.35, 1.5),
                'gemini-2.0-flash-live-001': (0.35, 1.5),
                'gemini-live-2.5-flash-preview': (0.35, 1.5),
                
                # Gemini 1.5 Series
                'gemini-1.5-flash': (0.075, 0.3),    # ≤128k tokens
                'gemini-1.5-flash-8b': (0.0375, 0.15),
                'gemini-1.5-pro': (1.25, 5.0),
                
                # Legacy/Deprecated
                'gemini-1.0-pro': (0.5, 1.5),
                'gemini-pro': (0.5, 1.5),
                
                # Experimental
                'gemini-exp': (1.25, 5.0),
                
                'default': (0.3, 2.5)
            },
            'mistral': {
                'mistral-large': (3.0, 9.0),
                'mistral-large-2': (3.0, 9.0),
                'mistral-medium': (0.4, 2.0),
                'mistral-medium-3': (0.4, 2.0),
                'mistral-small': (1.0, 3.0),
                'mistral-small-v24.09': (1.0, 3.0),
                'mistral-nemo': (0.3, 0.3),
                'mixtral-8x7b': (0.24, 0.24),
                'mixtral-8x22b': (1.0, 3.0),
                'codestral': (0.1, 0.3),
                'ministral': (0.1, 0.3),
                'default': (0.4, 2.0)
            },
            'groq': {
                # Grok 4.1 and 4 fast
                'grok-4-1-fast-reasoning': (0.20, 0.50),
                'grok-4-1-fast-non-reasoning': (0.20, 0.50),
                'grok-code-fast-1': (0.20, 1.50),
                'grok-4-fast-reasoning': (0.20, 0.50),
                'grok-4-fast-non-reasoning': (0.20, 0.50),
                'grok-4-0709': (3.00, 15.00),
                # Grok 3 series
                'grok-3-mini': (0.30, 0.50),
                'grok-3': (3.00, 15.00),
                # Grok 2 series
                'grok-2-vision-1212': (2.00, 10.00),
                # Legacy/default fallback
                'llama-4-scout': (0.11, 0.34),
                'llama-4-maverick': (0.5, 0.77),
                'llama-3.1-405b': (2.5, 2.5),
                'llama-3.1-70b': (0.59, 0.79),
                'llama-3.1-8b': (0.05, 0.1),
                'llama-3-70b': (0.59, 0.79),
                'llama-3-8b': (0.05, 0.1),
                'mixtral-8x7b': (0.24, 0.24),
                'gemma-7b': (0.07, 0.07),
                'gemma2-9b': (0.1, 0.1),
                'default': (0.3, 0.3)
            },
            'deepseek': {
                'deepseek-v3': (0.27, 1.09),         # Regular price
                'deepseek-v3-promo': (0.14, 0.27),   # Promo until Feb 8
                'deepseek-chat': (0.27, 1.09),
                'deepseek-r1': (0.27, 1.09),
                'deepseek-reasoner': (0.27, 1.09),
                'deepseek-coder': (0.14, 0.14),
                'default': (0.27, 1.09)
            },
            'cohere': {
                'command-a': (2.5, 10.0),
                'command-r-plus': (2.5, 10.0),
                'command-r+': (2.5, 10.0),
                'command-r': (0.15, 0.6),
                'command-r7b': (0.0375, 0.15),
                'command': (1.0, 3.0),
                'default': (0.5, 2.0)
            }
        }
        
        provider_prices = token_prices.get(provider, {'default': (2.5, 10.0)})
        
        # Find the right price for this model
        price_tuple = provider_prices.get('default', (2.5, 10.0))
        model_lower = model.lower()
        
        # Try exact match first
        if model_lower in provider_prices:
            price_tuple = provider_prices[model_lower]
        else:
            # Try prefix matching
            for model_key, price in provider_prices.items():
                if model_key == 'default':
                    continue
                # Remove version numbers for matching
                model_key_clean = model_key.replace('-', '').replace('.', '')
                model_lower_clean = model_lower.replace('-', '').replace('.', '')
                
                if (model_lower.startswith(model_key) or 
                    model_lower_clean.startswith(model_key_clean) or
                    model_key in model_lower):
                    price_tuple = price
                    break
        
        # Calculate weighted average price based on compression_factor
        input_price, output_price = price_tuple
        input_ratio = 1 / (1 + compression_factor)
        output_ratio = compression_factor / (1 + compression_factor)
        price_per_million = (input_ratio * input_price) + (output_ratio * output_price)
        
        # Calculate total tokens
        # For translation: output is typically 1.2-1.5x input length
        output_multiplier = compression_factor   # Conservative estimate
        total_tokens_per_chapter = avg_tokens_per_chapter * (1 + output_multiplier)
        total_tokens = num_chapters * total_tokens_per_chapter
        
        # Convert to cost
        regular_cost = (total_tokens / 1_000_000) * price_per_million
        
        # Batch API discount (50% off)
        discount = self.PROVIDER_CONFIGS.get(provider, {}).get('discount', 0.5)
        async_cost = regular_cost * discount
        
        # Log for debugging
        logger.info(f"Cost calculation for {model}:")
        logger.info(f"  Provider: {provider}")
        logger.info(f"  Input price: ${input_price:.4f}/1M tokens")
        logger.info(f"  Output price: ${output_price:.4f}/1M tokens")
        logger.info(f"  Compression factor: {compression_factor}")
        logger.info(f"  Weighted avg price: ${price_per_million:.4f}/1M tokens")
        logger.info(f"  Chapters: {num_chapters}")
        logger.info(f"  Avg input tokens/chapter: {avg_tokens_per_chapter:,}")
        logger.info(f"  Total tokens (input+output): {total_tokens:,}")
        logger.info(f"  Regular cost: ${regular_cost:.4f}")
        logger.info(f"  Async cost (50% off): ${async_cost:.4f}")
        
        return (async_cost, regular_cost)
        
    def prepare_batch_request(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare batch request for provider
        
        Args:
            chapters: List of chapter data with prompts
            model: Model name
            
        Returns:
            Provider-specific batch request format
        """
        provider = self.get_provider_from_model(model)
        
        if provider == 'openai':
            return self._prepare_openai_batch(chapters, model)
        elif provider == 'anthropic':
            return self._prepare_anthropic_batch(chapters, model)
        elif provider == 'gemini':
            return self._prepare_gemini_batch(chapters, model)
        elif provider == 'mistral':
            return self._prepare_mistral_batch(chapters, model)
        elif provider == 'groq':
            return self._prepare_groq_batch(chapters, model)
        else:
            raise ValueError(f"Unsupported provider for async: {provider}")
            
    def _prepare_openai_batch(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare OpenAI batch format"""
        
        # Allow any model to be used
        actual_model = model
        
        # Check if model is in our known supported list just for logging, but don't restrict it
        supported_batch_models = {
            # Current models (as of July 2025)
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4-turbo': 'gpt-4-turbo',
            'gpt-4-turbo-preview': 'gpt-4-turbo',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-3.5': 'gpt-3.5-turbo',
            
            # New GPT-4.1 models (if available in your region)
            'gpt-4.1': 'gpt-4.1',
            'gpt-4.1-mini': 'gpt-4.1-mini',
            'gpt-4o-nano': 'gpt-4o-nano',
            
            # Legacy models (may still work)
            'gpt-4': 'gpt-4',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0314': 'gpt-4-0314',
        }
        
        model_lower = model.lower()
        known_mapping = None
        for key, value in supported_batch_models.items():
            if model_lower == key.lower() or model_lower.startswith(key.lower()):
                known_mapping = value
                break
        
        if known_mapping:
            actual_model = known_mapping
            logger.info(f"Mapped '{model}' to known batch model '{actual_model}'")
        else:
            logger.info(f"Using unmapped model '{model}' for batch processing")
        
        requests = []
        
        for chapter in chapters:
            # Validate messages
            messages = chapter.get('messages', [])
            if not messages:
                print(f"Chapter {chapter['id']} has no messages!")
                continue
                
            # Ensure all messages have required fields
            valid_messages = []
            for msg in messages:
                if not msg.get('role') or not msg.get('content'):
                    print(f"Skipping invalid message: {msg}")
                    continue
                
                # Ensure content is string and not empty
                content = str(msg['content']).strip()
                if not content:
                    print(f"Skipping message with empty content")
                    continue
                    
                valid_messages.append({
                    'role': msg['role'],
                    'content': content
                })
            
            if not valid_messages:
                print(f"No valid messages for chapter {chapter['id']}")
                continue
            
            # Decide correct token param name: newer O-series / GPT-5+ require max_completion_tokens
            model_is_o_or_5 = self._is_o_series_model(actual_model)
            token_param_name = "max_completion_tokens" if model_is_o_or_5 else "max_tokens"

            # Honor the requested limit without capping so large outputs aren't truncated
            requested_max_tokens = int(chapter.get('max_tokens', 65536))

            request = {
                "custom_id": chapter['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": actual_model,
                    "messages": valid_messages,
                    "temperature": float(chapter.get('temperature', 0.3)),
                    token_param_name: requested_max_tokens
                }
            }

            # Optional Gemini thinking budget (expects tokens count in chapter['thinking_budget_tokens'])
            # Note: This is specific to Gemini, but keeping logic structure consistent.
            # OpenAI typically uses 'reasoning_effort' or different params for o1/o3 models if supported in batch.
            # If this was intended for OpenAI 'thinking' parameters, it might be incorrect here as OpenAI doesn't use "generateContentRequest" structure.
            # However, if this code block is generic or copy-pasted, we should be careful.
            # Since this is _prepare_openai_batch, "generateContentRequest" key is definitely WRONG for OpenAI.
            # OpenAI Batch API expects standard chat completion body.
            
            # Remove the incorrect Gemini-style structure injection for OpenAI
            # If you need to support reasoning models (o1/o3), they use standard parameters or don't support temperature.
            # For now, stripping the invalid key insertion.
            # LOG THE FIRST REQUEST COMPLETELY
            if len(requests) == 0:
                print(f"=== FIRST REQUEST ===")
                print(json.dumps(request, indent=2))
                print(f"=== END FIRST REQUEST ===")
            
            requests.append(request)
            
        return {"requests": requests}
        
    def _prepare_anthropic_batch(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare Anthropic batch format"""
        requests = []
        
        for chapter in chapters:
            # Extract system message if present
            system = None
            messages = []
            
            for msg in chapter['messages']:
                if msg['role'] == 'system':
                    system = msg['content']
                else:
                    messages.append(msg)
            
            request = {
                "custom_id": chapter['id'],
                "params": {
                    "model": model,
                    "messages": messages,
                    "max_tokens": chapter.get('max_tokens', 65536),
                    "temperature": chapter.get('temperature', 0.3)
                }
            }
            
            if system:
                request["params"]["system"] = system
                
            requests.append(request)
            
        return {"requests": requests}
        
    def _prepare_gemini_batch(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare Gemini batch format"""
        requests = []
        
        for chapter in chapters:
            # Format messages for Gemini
            prompt = self._format_messages_for_gemini(chapter['messages'])
            
            request = {
                "custom_id": chapter['id'],
                "generateContentRequest": {
                    "model": f"models/{model}",
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": chapter.get('temperature', 0.3),
                        "maxOutputTokens": chapter.get('max_tokens', 65536)
                    }
                }
            }
            
            # Optional Gemini thinking config (Gemini 3 supports level and/or budgetTokens)
            thinking_enabled = os.getenv("ENABLE_GEMINI_THINKING", "1").lower() not in ("0", "false")
            if thinking_enabled:
                thinking_cfg = {}
                thinking_level = chapter.get('thinking_level') or os.getenv("GEMINI_THINKING_LEVEL")
                if thinking_level:
                    thinking_cfg["level"] = str(thinking_level)
                thinking_budget = chapter.get('thinking_budget_tokens')
                if thinking_budget is None:
                    env_budget = os.getenv("THINKING_BUDGET")
                    if env_budget is not None:
                        try:
                            thinking_budget = int(env_budget)
                        except Exception:
                            thinking_budget = None
                if thinking_budget is not None:
                    thinking_cfg["budgetTokens"] = int(thinking_budget)
                if thinking_cfg:
                    request["generateContentRequest"]["generationConfig"]["thinking"] = thinking_cfg

            # Add safety settings if disabled
            if os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true":
                request["generateContentRequest"]["safetySettings"] = [
                    {"category": cat, "threshold": "BLOCK_NONE"}
                    for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                               "HARM_CATEGORY_CIVIC_INTEGRITY"]
                ]
                
            requests.append(request)
            
        return {"requests": requests}
        
    def _prepare_mistral_batch(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare Mistral batch format"""
        requests = []
        
        for chapter in chapters:
            request = {
                "custom_id": chapter['id'],
                "model": model,
                "messages": chapter['messages'],
                "temperature": chapter.get('temperature', 0.3),
                "max_tokens": chapter.get('max_tokens', 65536)
            }
            requests.append(request)
            
        return {"requests": requests}
        
    def _prepare_groq_batch(self, chapters: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Prepare Groq batch format (OpenAI-compatible)"""
        return self._prepare_openai_batch(chapters, model)
        
    def _format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Gemini prompt"""
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg['content']
            
            if role == 'SYSTEM':
                formatted_parts.append(f"INSTRUCTIONS: {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
                
        return "\n\n".join(formatted_parts)

    def _is_o_series_model(self, model: str) -> bool:
        """Detect OpenAI o-series and GPT-5+ models that require max_completion_tokens."""
        ml = model.lower()
        return ml.startswith(('o1', 'o3', 'o4', 'gpt-5'))
        
    async def submit_batch(self, batch_data: Dict[str, Any], model: str, api_key: str) -> AsyncJobInfo:
        """Submit batch to provider and create job entry"""
        provider = self.get_provider_from_model(model)
        
        if provider == 'openai':
            return await self._submit_openai_batch(batch_data, model, api_key)
        elif provider == 'anthropic':
            return await self._submit_anthropic_batch(batch_data, model, api_key)
        elif provider == 'gemini':
            return await self._submit_gemini_batch(batch_data, model, api_key)
        elif provider == 'mistral':
            return await self._submit_mistral_batch(batch_data, model, api_key)
        elif provider == 'groq':
            return await self._submit_groq_batch(batch_data, model, api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    def _submit_openai_batch_sync(self, batch_data, model, api_key):
        """Submit OpenAI batch synchronously"""
        try:
            # Remove aiofiles import - not needed for sync operations
            import tempfile
            import json
            
            # Create temporary file for batch data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                # Write each request as JSONL
                for request in batch_data['requests']:
                    json.dump(request, f)
                    f.write('\n')
                temp_path = f.name
            
            try:
                # Upload file to OpenAI
                headers = {'Authorization': f'Bearer {api_key}'}
                
                with open(temp_path, 'rb') as f:
                    files = {'file': ('batch.jsonl', f, 'application/jsonl')}
                    data = {'purpose': 'batch'}
                    
                    response = requests.post(
                        'https://api.openai.com/v1/files',
                        headers=headers,
                        files=files,
                        data=data
                    )
                    
                if response.status_code != 200:
                    raise Exception(f"File upload failed: {response.text}")
                    
                file_id = response.json()['id']
                
                # Create batch job
                batch_request = {
                    'input_file_id': file_id,
                    'endpoint': '/v1/chat/completions',
                    'completion_window': '24h'
                }
                
                response = requests.post(
                    'https://api.openai.com/v1/batches',
                    headers={**headers, 'Content-Type': 'application/json'},
                    json=batch_request
                )
                
                if response.status_code != 200:
                    raise Exception(f"Batch creation failed: {response.text}")
                    
                batch_info = response.json()
                
                # Calculate cost estimate
                total_tokens = sum(r.get('token_count', 15000) for r in batch_data['requests'])
                async_cost, _ = self.estimate_cost(
                    len(batch_data['requests']), 
                    total_tokens // len(batch_data['requests']), 
                    model
                )
                
                job = AsyncJobInfo(
                    job_id=batch_info['id'],
                    provider='openai',
                    model=model,
                    status=AsyncAPIStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    total_requests=len(batch_data['requests']),
                    cost_estimate=async_cost,
                    metadata={'file_id': file_id, 'batch_info': batch_info}
                )
                
                return job
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            print(f"OpenAI batch submission failed: {e}")
            raise
            
    def _submit_anthropic_batch_sync(self, batch_data: Dict[str, Any], model: str, api_key: str) -> AsyncJobInfo:
        """Submit Anthropic batch (synchronous version)"""
        try:
            headers = {
                'X-API-Key': api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'anthropic-beta': 'message-batches-2024-09-24'
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages/batches',
                headers=headers,
                json=batch_data
            )
            
            if response.status_code != 200:
                raise Exception(f"Batch creation failed: {response.text}")
                
            batch_info = response.json()
            
            job = AsyncJobInfo(
                job_id=batch_info['id'],
                provider='anthropic',
                model=model,
                status=AsyncAPIStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                total_requests=len(batch_data['requests']),
                metadata={'batch_info': batch_info}
            )
            
            return job
            
        except Exception as e:
            print(f"Anthropic batch submission failed: {e}")
            raise
            
    def check_job_status(self, job_id: str) -> AsyncJobInfo:
        """Check the status of a batch job"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
            
        try:
            provider = job.provider
            
            if provider == 'openai':
                self._check_openai_status(job)
            elif provider == 'gemini':
                self._check_gemini_status(job)
            elif provider == 'anthropic':
                self._check_anthropic_status(job)
            else:
                print(f"Unknown provider: {provider}")
                
            # Update timestamp
            job.updated_at = datetime.now()
            self._save_jobs()
            
        except Exception as e:
            print(f"Error checking job status: {e}")
            job.metadata['last_error'] = str(e)
            
        return job

    def _check_gemini_status(self, job: AsyncJobInfo):
        """Check Gemini batch status"""
        try:
            # First try the Python SDK approach
            try:
                from google import genai
                
                api_key = self._get_api_key()
                client = genai.Client(api_key=api_key)
                
                # Get batch job status
                batch_job = client.batches.get(name=job.job_id)
                
                # Log the actual response for debugging
                logger.info(f"Gemini batch job state: {batch_job.state.name if hasattr(batch_job, 'state') else 'Unknown'}")
                
                # Map Gemini states to our status
                state_map = {
                    'JOB_STATE_PENDING': AsyncAPIStatus.PENDING,
                    'JOB_STATE_RUNNING': AsyncAPIStatus.PROCESSING,
                    'JOB_STATE_SUCCEEDED': AsyncAPIStatus.COMPLETED,
                    'JOB_STATE_FAILED': AsyncAPIStatus.FAILED,
                    'JOB_STATE_CANCELLED': AsyncAPIStatus.CANCELLED,
                    'JOB_STATE_CANCELLING': AsyncAPIStatus.PROCESSING
                }
                
                job.status = state_map.get(batch_job.state.name, AsyncAPIStatus.PENDING)
                
                # Update metadata
                if not job.metadata:
                    job.metadata = {}
                if 'batch_info' not in job.metadata:
                    job.metadata['batch_info'] = {}
                    
                job.metadata['batch_info']['state'] = batch_job.state.name
                job.metadata['raw_state'] = batch_job.state.name
                job.metadata['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Try to get progress information
                if hasattr(batch_job, 'completed_count'):
                    job.completed_requests = batch_job.completed_count
                elif job.status == AsyncAPIStatus.PROCESSING:
                    # If processing but no progress info, show as 1 to indicate it started
                    job.completed_requests = 1
                elif job.status == AsyncAPIStatus.COMPLETED:
                    # If completed, all requests are done
                    job.completed_requests = job.total_requests
                    
                # If completed, store the result file info
                if batch_job.state.name == 'JOB_STATE_SUCCEEDED' and hasattr(batch_job, 'dest'):
                    job.output_file = batch_job.dest.file_name if hasattr(batch_job.dest, 'file_name') else None
                    
            except Exception as sdk_error:
                # Fallback to REST API if SDK fails
                print(f"Gemini SDK failed, trying REST API: {sdk_error}")
                
                api_key = self._get_api_key()
                headers = {'x-goog-api-key': api_key}
                
                batch_name = job.job_id if job.job_id.startswith('batches/') else f'batches/{job.job_id}'
                
                response = requests.get(
                    f'https://generativelanguage.googleapis.com/v1beta/{batch_name}',
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update job status
                    state = data.get('metadata', {}).get('state', 'JOB_STATE_PENDING')
                    
                    # Map states
                    state_map = {
                        'JOB_STATE_PENDING': AsyncAPIStatus.PENDING,
                        'JOB_STATE_RUNNING': AsyncAPIStatus.PROCESSING,
                        'JOB_STATE_SUCCEEDED': AsyncAPIStatus.COMPLETED,
                        'JOB_STATE_FAILED': AsyncAPIStatus.FAILED,
                        'JOB_STATE_CANCELLED': AsyncAPIStatus.CANCELLED,
                    }
                    
                    job.status = state_map.get(state, AsyncAPIStatus.PENDING)
                    
                    # Extract progress from metadata
                    metadata = data.get('metadata', {})
                    
                    # Gemini might provide progress info
                    if 'completedRequestCount' in metadata:
                        job.completed_requests = metadata['completedRequestCount']
                    if 'failedRequestCount' in metadata:
                        job.failed_requests = metadata['failedRequestCount']
                    if 'totalRequestCount' in metadata:
                        job.total_requests = metadata['totalRequestCount']
                        
                    # Store raw state
                    if not job.metadata:
                        job.metadata = {}
                    job.metadata['raw_state'] = state
                    job.metadata['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check if completed
                    if state == 'JOB_STATE_SUCCEEDED' and 'response' in data:
                        job.status = AsyncAPIStatus.COMPLETED
                        if 'responsesFile' in data.get('response', {}):
                            job.output_file = data['response']['responsesFile']
                else:
                    print(f"Gemini status check failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"Gemini status check failed: {e}")
            if not job.metadata:
                job.metadata = {}
            job.metadata['last_error'] = str(e)

    def _check_openai_status(self, job: AsyncJobInfo):
        """Check OpenAI batch status"""
        try:
            api_key = self._get_api_key()
            headers = {'Authorization': f'Bearer {api_key}'}
            
            response = requests.get(
                f'https://api.openai.com/v1/batches/{job.job_id}',
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Status check failed: {response.text}")
                return
                
            data = response.json()
            
            # Log the full response for debugging
            logger.debug(f"OpenAI batch status response: {json.dumps(data, indent=2)}")
            # Check for high failure rate while in progress
            request_counts = data.get('request_counts', {})
            total = request_counts.get('total', 0)
            failed = request_counts.get('failed', 0)
            completed = request_counts.get('completed', 0)
            
            # Map OpenAI status to our status
            status_map = {
                'validating': AsyncAPIStatus.PENDING,
                'in_progress': AsyncAPIStatus.PROCESSING,
                'finalizing': AsyncAPIStatus.PROCESSING,
                'completed': AsyncAPIStatus.COMPLETED,
                'failed': AsyncAPIStatus.FAILED,
                'expired': AsyncAPIStatus.EXPIRED,
                'cancelled': AsyncAPIStatus.CANCELLED,
                'cancelling': AsyncAPIStatus.CANCELLED,
            }
            
            job.status = status_map.get(data['status'], AsyncAPIStatus.PENDING)
            
            # Update progress
            request_counts = data.get('request_counts', {})
            job.completed_requests = request_counts.get('completed', 0)
            job.failed_requests = request_counts.get('failed', 0)
            job.total_requests = request_counts.get('total', job.total_requests)
            
            # Store metadata
            if not job.metadata:
                job.metadata = {}
            job.metadata['raw_state'] = data['status']
            job.metadata['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle completion
            if data['status'] == 'completed':
                # Check if all requests failed
                if job.failed_requests > 0 and job.completed_requests == 0:
                    print(f"OpenAI job completed but all {job.failed_requests} requests failed")
                    job.status = AsyncAPIStatus.FAILED
                    job.metadata['all_failed'] = True
                    
                    # Store error file if available
                    if data.get('error_file_id'):
                        job.metadata['error_file_id'] = data['error_file_id']
                        logger.info(f"Error file available: {data['error_file_id']}")
                else:
                    # Normal completion with some successes
                    if 'output_file_id' in data and data['output_file_id']:
                        job.output_file = data['output_file_id']
                        logger.info(f"OpenAI job completed with output file: {job.output_file}")
                        
                        # If there were also failures, note that
                        if job.failed_requests > 0:
                            job.metadata['partial_failure'] = True
                            print(f"Job completed with {job.failed_requests} failed requests out of {job.total_requests}")
                    else:
                        print(f"OpenAI job marked as completed but no output_file_id found: {data}")
                        
            # Always store error file if present
            if data.get('error_file_id'):
                job.metadata['error_file_id'] = data['error_file_id']
                
        except Exception as e:
            print(f"OpenAI status check failed: {e}")
            if not job.metadata:
                job.metadata = {}
            job.metadata['last_error'] = str(e)
                
    def _check_anthropic_status(self, job: AsyncJobInfo):
        """Check Anthropic batch status"""
        try:
            api_key = self._get_api_key()
            headers = {
                'X-API-Key': api_key,
                'anthropic-version': '2023-06-01',
                'anthropic-beta': 'message-batches-2024-09-24'
            }
            
            response = requests.get(
                f'https://api.anthropic.com/v1/messages/batches/{job.job_id}',
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Status check failed: {response.text}")
                return
                
            data = response.json()
            
            # Map Anthropic status
            status_map = {
                'created': AsyncAPIStatus.PENDING,
                'processing': AsyncAPIStatus.PROCESSING,
                'ended': AsyncAPIStatus.COMPLETED,
                'failed': AsyncAPIStatus.FAILED,
                'expired': AsyncAPIStatus.EXPIRED,
                'canceled': AsyncAPIStatus.CANCELLED,
            }
            
            job.status = status_map.get(data['processing_status'], AsyncAPIStatus.PENDING)
            
            # Update progress
            results_summary = data.get('results_summary', {})
            job.completed_requests = results_summary.get('succeeded', 0)
            job.failed_requests = results_summary.get('failed', 0)
            job.total_requests = results_summary.get('total', job.total_requests)
            
            # Store metadata
            if not job.metadata:
                job.metadata = {}
            job.metadata['raw_state'] = data['processing_status']
            job.metadata['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if data.get('results_url'):
                job.output_file = data['results_url']
                
        except Exception as e:
            print(f"Anthropic status check failed: {e}")
            if not job.metadata:
                job.metadata = {}
            job.metadata['last_error'] = str(e)
            
    def _get_api_key(self) -> str:
        """Get API key from GUI settings, with AuthGPT OAuth fallback"""
        # Try standard API key first
        key = ''
        if hasattr(self.gui, 'api_key_entry'):
            if hasattr(self.gui.api_key_entry, 'text'):
                key = self.gui.api_key_entry.text().strip()
            else:
                key = self.gui.api_key_entry.get().strip()
        elif hasattr(self.gui, 'api_key_var'):
            key = self.gui.api_key_var.get().strip()
        else:
            key = os.getenv('API_KEY', '') or os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')
        
        if key:
            return key
        
        # Fallback: try AuthGPT OAuth token if model is authgpt/*
        try:
            model = ''
            if hasattr(self.gui, 'model_var'):
                if hasattr(self.gui.model_var, 'get'):
                    model = self.gui.model_var.get()
                else:
                    model = str(self.gui.model_var) if self.gui.model_var else ''
            
            if model.lower().startswith('authgpt/'):
                try:
                    from authgpt_auth import get_default_store
                    token = get_default_store().get_valid_access_token(auto_login=True)
                    if token:
                        return token
                except Exception as e:
                    print(f"[ASYNC] AuthGPT OAuth token retrieval failed: {e}")
        except Exception:
            pass
        
        return key
    
    def _get_api_key_from_gui(self) -> str:
        """Wrapper for _get_api_key for consistency"""
        return self._get_api_key()
        
    def retrieve_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve results from a completed batch job"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
            
        if job.status != AsyncAPIStatus.COMPLETED:
            raise ValueError(f"Job is not completed. Current status: {job.status.value}")
        
        # If output file is missing, try to refresh status first
        if not job.output_file:
            print(f"No output file for completed job {job_id}, refreshing status...")
            self.check_job_status(job_id)
            
            # Re-check after status update
            if not job.output_file:
                # Log the job details for debugging
                print(f"Job details: {json.dumps(job.to_dict(), indent=2)}")
                raise ValueError(f"No output file available for job {job_id} even after status refresh")
        
        provider = job.provider
        
        if provider == 'openai':
            return self._retrieve_openai_results(job)
        elif provider == 'gemini':
            return self._retrieve_gemini_results(job)
        elif provider == 'anthropic':
            return self._retrieve_anthropic_results(job)
        else:
            raise ValueError(f"Unknown provider: {provider}")
 
    def _retrieve_gemini_results(self, job: AsyncJobInfo) -> List[Dict[str, Any]]:
        """Retrieve Gemini batch results"""
        try:
            from google import genai
            
            api_key = self._get_api_key()
            
            # Create client with API key
            client = genai.Client(api_key=api_key)
            
            # Get the batch job
            batch_job = client.batches.get(name=job.job_id)
            
            if batch_job.state != 'JOB_STATE_SUCCEEDED':
                raise ValueError(f"Batch job not completed: {batch_job.state}")
            
            # Download results
            if hasattr(batch_job, 'dest') and batch_job.dest:
                # Extract the file name from the destination object
                if hasattr(batch_job.dest, 'output_uri'):
                    # For BigQuery or Cloud Storage destinations
                    file_name = batch_job.dest.output_uri
                elif hasattr(batch_job.dest, 'file_name'):
                    # For file-based destinations
                    file_name = batch_job.dest.file_name
                else:
                    # Try to get any file reference from the dest object
                    # Log the object to understand its structure
                    logger.info(f"BatchJobDestination object: {batch_job.dest}")
                    logger.info(f"BatchJobDestination attributes: {dir(batch_job.dest)}")
                    raise ValueError(f"Cannot extract file name from destination: {batch_job.dest}")
                
                # Download the results file
                results_content_bytes = client.files.download(file=file_name)
                results_content = results_content_bytes.decode('utf-8')
                
                results = []
                # Parse JSONL results
                for line in results_content.splitlines():
                    if line.strip():
                        result_data = json.loads(line)
                        
                        # Extract the response content
                        text_content = ""
                        
                        # Handle different response formats
                        if 'response' in result_data:
                            response = result_data['response']
                            
                            # Check for different content structures
                            if isinstance(response, dict):
                                if 'candidates' in response and response['candidates']:
                                    candidate = response['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                text_content += part['text']
                                    elif 'text' in candidate:
                                        text_content = candidate['text']
                                elif 'text' in response:
                                    text_content = response['text']
                                elif 'content' in response:
                                    text_content = response['content']
                            elif isinstance(response, str):
                                text_content = response
                        
                        results.append({
                            'custom_id': result_data.get('key', ''),
                            'content': text_content,
                            'finish_reason': 'stop'
                        })
                            
                return results
            else:
                raise ValueError("No output file available for completed job")
                
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
        except Exception as e:
            print(f"Failed to retrieve Gemini results: {e}")
            raise
 
    def _retrieve_openai_results(self, job: AsyncJobInfo) -> List[Dict[str, Any]]:
        """Retrieve OpenAI batch results"""
        if not job.output_file:
            # Try one more status check
            self._check_openai_status(job)
            if not job.output_file:
                raise ValueError(f"No output file available for OpenAI job {job.job_id}")
        
        try:
            api_key = self._get_api_key()
            headers = {'Authorization': f'Bearer {api_key}'}
            
            # Download results file
            response = requests.get(
                f'https://api.openai.com/v1/files/{job.output_file}/content',
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to download results: {response.status_code} - {response.text}")
                
            # Parse JSONL results
            results = []
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        result = json.loads(line)
                        # Extract the actual response content
                        if 'response' in result and 'body' in result['response']:
                            results.append({
                                'custom_id': result.get('custom_id', ''),
                                'content': result['response']['body']['choices'][0]['message']['content'],
                                'finish_reason': result['response']['body']['choices'][0].get('finish_reason', 'stop')
                            })
                        else:
                            print(f"Unexpected result format: {result}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse result line: {line} - {e}")
                        
            return results
            
        except Exception as e:
            print(f"Failed to retrieve OpenAI results: {e}")
            print(f"Job details: {json.dumps(job.to_dict(), indent=2)}")
            raise
        
    def _retrieve_anthropic_results(self, job: AsyncJobInfo) -> List[Dict[str, Any]]:
        """Retrieve Anthropic batch results"""
        if not job.output_file:
            raise ValueError("No output file available")
            
        api_key = self._get_api_key()
        headers = {
            'X-API-Key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        # Download results
        response = requests.get(job.output_file, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download results: {response.text}")
            
        # Parse JSONL results
        results = []
        for line in response.text.strip().split('\n'):
            if line:
                result = json.loads(line)
                if result['result']['type'] == 'succeeded':
                    message = result['result']['message']
                    results.append({
                        'custom_id': result['custom_id'],
                        'content': message['content'][0]['text'],
                        'finish_reason': message.get('stop_reason', 'stop')
                    })
                    
        return results


# ---------------------------------------------------------------------------
# Dialog view logic shared with the mobile screen (moved from AsyncProcessingDialog)
# ---------------------------------------------------------------------------

def gui_model_name(gui):
    """The model the dialog shows: ``gui.model_var`` ("Not selected" without one)."""
    # Get model name from GUI - handle both tkinter and PySide6
    if hasattr(gui, 'model_var'):
        if hasattr(gui.model_var, 'get'):
            model_name = gui.model_var.get()
        else:
            model_name = str(gui.model_var) if gui.model_var else "Not selected"
    else:
        model_name = "Not selected"
    return model_name


def async_support_status(processor, model_name):
    """``(supported, text)`` of the dialog's model row: "✓ Supported (OPENAI)" / "✗ Not supported for async"."""
    # Check if model supports async
    provider = processor.get_provider_from_model(model_name)
    if provider and provider in processor.PROVIDER_CONFIGS:
        status_text = f"✓ Supported ({provider.upper()})"
        return True, status_text
    else:
        status_text = "✗ Not supported for async"
        return False, status_text


def job_display_row(job_id, job):
    """One row of the dialog's job list (``_refresh_jobs_list``) as a dict of display strings."""
    # Calculate progress percentage and format progress text
    if job.total_requests > 0:
        progress_pct = int((job.completed_requests / job.total_requests) * 100)
        progress_text = f"{progress_pct}% ({job.completed_requests}/{job.total_requests})"
    else:
        progress_pct = 0
        progress_text = "0% (0/0)"

    # Override progress text for completed/failed/cancelled statuses
    if job.status == AsyncAPIStatus.COMPLETED:
        progress_text = "100% (Complete)"
    elif job.status == AsyncAPIStatus.FAILED:
        progress_text = f"{progress_pct}% (Failed)"
    elif job.status == AsyncAPIStatus.CANCELLED:
        progress_text = f"{progress_pct}% (Cancelled)"
    elif job.status == AsyncAPIStatus.PENDING:
        progress_text = "0% (Waiting)"

    created = job.created_at.strftime("%Y-%m-%d %H:%M")
    cost = f"${job.cost_estimate:.2f}" if job.cost_estimate else "N/A"

    # Determine status style
    status_text = job.status.value.capitalize()

    # Shorten job ID for display
    display_id = job_id[:20] + "..." if len(job_id) > 20 else job_id

    source_file = ""
    try:
        src_path = job.metadata.get('source_file') if job.metadata else ""
        if src_path:
            source_file = os.path.basename(src_path)
    except Exception:
        source_file = ""

    return {
        "job_id": job_id,
        "display_id": display_id,
        "provider": job.provider.upper(),
        "model": job.model[:15] + "..." if len(job.model) > 15 else job.model,  # Shorten model name
        "status": status_text,
        "progress": progress_text,  # Now shows percentage and counts
        "progress_pct": progress_pct,
        "created": created,
        "source_file": source_file,
        "cost": cost,
        "state": job.status.value,
    }


def selected_job_progress(job):
    """``(percent, label)`` of the dialog's "Selected Job Progress" bar for ``job``."""
    if job.total_requests > 0:
        progress = int((job.completed_requests / job.total_requests) * 100)
        return progress, f"{progress}% ({job.completed_requests}/{job.total_requests} chapters)"
    return 0, "0% (Waiting)"


def refresh_pending_job_statuses(processor):
    """Check every pending / processing job once (the dialog's 30 s auto-refresh)."""
    # Refresh all jobs
    for job_id in list(processor.jobs.keys()):
        try:
            job = processor.jobs[job_id]
            if job.status in [AsyncAPIStatus.PENDING, AsyncAPIStatus.PROCESSING]:
                processor.check_job_status(job_id)
        except:
            pass


#: ``QMessageBox.StandardButton`` values the hooks answer with (the desktop dialog uses the real enums).
MB_OK = 0x00000400
MB_YES = 0x00004000
MB_NO = 0x00010000
MB_CANCEL = 0x00400000
#: Button flag -> answer name of ``host.ask('async_batch_question', ...)``.
_ANSWER_NAMES = ((MB_YES, "yes"), (MB_NO, "no"), (MB_CANCEL, "cancel"))


class AsyncBatchJobMixin:
    """The ``AsyncProcessingDialog`` workflow (U7 move), GUI-free.

    The owner state is the dialog's: ``self.gui`` (TranslatorGUI on desktop, a
    ``headless_owner.HeadlessOwner`` on mobile), ``self.processor`` (``AsyncAPIProcessor``),
    ``self.selected_job_id``, ``self.polling_jobs`` and ``self.dialog`` (the QDialog, or None).

    The methods below the hook section are the dialog's, moved verbatim except for the Qt
    statements, which call these hooks instead:

    ============================================================  ==========================================
    dialog code                                                    hook
    ============================================================  ==========================================
    ``QMessageBox.<kind>(self.dialog, *args)``                     ``self._async_msgbox('<kind>', *args)``
    ``QMessageBox.Yes`` / ``No`` / ``Cancel``                      ``self._MB_YES`` / ``_MB_NO`` / ``_MB_CANCEL``
    ``QTimer.singleShot(*args)``                                   ``self._async_single_shot(*args)``
    ``QApplication.processEvents()``                               ``self._async_process_events()``
    ``self.cost_info_label.setText(text)``                         ``self._async_set_cost_info(text)``
    ``self.start_button.setEnabled(flag)``                         ``self._async_set_start_enabled(flag)``
    ``self.wait_for_completion_checkbox.isChecked()``              ``self._async_wait_for_completion()``
    ``self.poll_interval_spinbox.value()``                         ``self._async_poll_interval()``
    ``self.dialog.setCursor(Qt.WaitCursor / Qt.ArrowCursor)``      ``self._async_set_wait_cursor(True / False)``
    ``hasattr(self, 'dialog') and self.dialog.isVisible()``        ``self._async_dialog_visible()``
    ============================================================  ==========================================

    ``AsyncProcessingDialog`` (async_api_processor) defines every hook with the original Qt
    statement, plus its own ``_refresh_jobs_list`` / ``_get_selected_job_ids`` (the job tree) and
    ``_log`` / ``_show_error`` / ``_show_info`` / ``_show_warning`` (thread-safe Qt helpers);
    its class body wins over these GUI-free defaults. ``HeadlessAsyncBatch`` uses the defaults.
    """

    _MB_OK = MB_OK
    _MB_YES = MB_YES
    _MB_NO = MB_NO
    _MB_CANCEL = MB_CANCEL

    # ---- GUI-free hook defaults (the desktop dialog overrides every one) --------------------
    def _async_emit(self, kind, **data):
        """``host.emit(kind, **data)`` when a host is attached."""
        emit = getattr(getattr(self, 'host', None), 'emit', None)
        if callable(emit):
            try:
                emit(kind, **data)
            except Exception:
                pass

    def _async_stop_requested(self):
        """The host's stop latch (``job_runner.JobHost.is_stop_requested``)."""
        check = getattr(getattr(self, 'host', None), 'is_stop_requested', None)
        try:
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def _async_answer(self, kind, title, text, names):
        """The answer name ('yes' / 'no' / 'cancel') to a dialog question.

        ``self.answers[title]`` (preset by the caller, e.g. after its own confirmation sheet)
        wins; else ``host.ask('async_batch_question', ...)``; without either the answer is
        'no' (or the last button), so nothing destructive happens unasked.
        """
        fallback = 'no' if 'no' in names else names[-1]
        preset = (getattr(self, 'answers', None) or {}).get(title)
        if preset is None:
            ask = getattr(getattr(self, 'host', None), 'ask', None)
            if callable(ask):
                try:
                    preset = ask('async_batch_question', level=kind, title=title, text=text,
                                 buttons=list(names), default=fallback)
                except Exception:
                    preset = None
        if preset is True:
            preset = 'yes'
        elif preset is False:
            preset = 'no'
        preset = str(preset).strip().lower() if preset is not None else ''
        return preset if preset in names else fallback

    def _async_msgbox(self, kind, title, text, buttons=None):
        """``QMessageBox.<kind>(self.dialog, title, text[, buttons])`` without Qt.

        Without buttons it is a notice (recorded in ``self.messages`` and emitted as
        ``async_batch_message``); with buttons it is a question (``_async_answer``) and returns
        the chosen button flag, like the Qt call.
        """
        record = {"level": kind, "title": title, "text": text}
        if buttons is None:
            getattr(self, 'messages', []).append(record)
            self._async_emit('async_batch_message', **record)
            return self._MB_OK
        names = [name for flag, name in ((self._MB_YES, 'yes'), (self._MB_NO, 'no'), (self._MB_CANCEL, 'cancel'))
                 if buttons & flag]
        answer = self._async_answer(kind, title, text, names or ['no'])
        record["answer"] = answer
        getattr(self, 'messages', []).append(record)
        return {'yes': self._MB_YES, 'no': self._MB_NO, 'cancel': self._MB_CANCEL}[answer]

    def _async_single_shot(self, msec, *args):
        """``QTimer.singleShot(msec[, context], callback)`` without an event loop.

        A zero delay calls back at once (the caller's thread). A positive delay (the
        "Wait for completion" polling) waits on the caller's thread and stops when the host's
        stop latch is set; nested waits are queued so polling never deepens the stack.
        """
        callback = args[-1]
        if not msec or msec <= 0:
            callback()
            return
        pending = getattr(self, '_async_timer_queue', None)
        if pending is not None:
            pending.append((msec, callback))
            return
        self._async_timer_queue = pending = [(msec, callback)]
        try:
            while pending:
                delay, fn = pending.pop(0)
                if not self._async_wait(delay / 1000.0):
                    self._log("⏹️ Async polling stopped", level="warning")
                    break
                fn()
        finally:
            self._async_timer_queue = None

    def _async_wait(self, seconds):
        """Wait ``seconds``; False as soon as the host's stop latch is set."""
        deadline = time.monotonic() + max(0.0, float(seconds))
        event = threading.Event()
        while True:
            if self._async_stop_requested():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            event.wait(min(0.5, remaining))

    def _async_process_events(self):
        """``QApplication.processEvents()``: nothing to pump without Qt."""
        return None

    def _async_set_cost_info(self, text):
        """The cost label: kept as ``self.cost_info`` and emitted as ``async_batch_cost``."""
        self.cost_info = text
        self._async_emit('async_batch_cost', text=text)

    def _async_set_start_enabled(self, enabled):
        """The Start button's enabled state (``self.start_enabled``)."""
        self.start_enabled = bool(enabled)

    def _async_wait_for_completion(self):
        """The "Wait for completion" checkbox: config ``async_wait_for_completion`` (dialog default False)."""
        return bool(self.gui.config.get('async_wait_for_completion', False))

    def _async_poll_interval(self):
        """The poll interval spin box: config ``async_poll_interval`` (default 60) within its 10-600 s range."""
        try:
            value = int(self.gui.config.get('async_poll_interval', 60))
        except Exception:
            value = 60
        return min(600, max(10, value))

    def _async_set_wait_cursor(self, waiting):
        """The dialog's busy cursor: nothing to show without Qt."""
        return None

    def _async_dialog_visible(self):
        """Whether the dialog is shown (never, without Qt)."""
        return False

    def _refresh_jobs_list(self):
        """The job tree: emits ``async_batch_jobs`` with ``job_display_row`` rows."""
        self._async_emit('async_batch_jobs', rows=[job_display_row(job_id, job)
                                                   for job_id, job in self.processor.jobs.items()])

    def _get_selected_job_ids(self) -> List[str]:
        """The job tree selection: ``self.selected_job_ids`` (set by the caller)."""
        return [job_id for job_id in (getattr(self, 'selected_job_ids', None) or []) if job_id]

    # Helper methods for thread-safe UI updates (GUI-free versions of the dialog's)
    def _log(self, message, level="info"):
        """The dialog's levels: errors / warnings are printed, info goes to the owner's log."""
        # Log based on level
        if level == "error":
            print(f"❌ {message}")
        elif level == "warning":
            print(f"⚠️ {message}")
        else:
            logger.info(message)  # This only goes to log file
            # Also display info messages in GUI
            if hasattr(self.gui, 'append_log'):
                self.gui.append_log(message)
        self._async_note_log(message)

    def _async_note_log(self, message):
        """Remember the output folder ``_handle_completed_job`` reports ("✅ Saved N chapters to: <dir>")."""
        match = re.match(r"✅ Saved (\d+) chapters to: (.+)$", str(message), re.DOTALL)
        if match:
            saved = getattr(self, 'saved_output_dirs', None)
            if saved is not None:
                saved.append(match.group(2))

    def _show_error(self, message):
        """Error notice: logged like the dialog's (log line + "❌" log entry) and a critical message."""
        self._log(f"Error: {message}", level="error")
        # Also show in the GUI log panel so it's visible even if dialog fails
        if hasattr(self.gui, 'append_log'):
            try:
                self.gui.append_log(f"❌ {message}")
            except Exception:
                pass
        # Truncate for dialog display (very long messages can cause rendering issues)
        display_msg = message if len(message) <= 800 else message[:800] + '\n... (truncated)'
        self._async_msgbox('critical', "Error", display_msg)

    def _show_info(self, title, message):
        """Info notice (logged, then an information message)."""
        self._log(f"{title}: {message}", level="info")
        self._async_msgbox('information', title, message)

    def _show_warning(self, message):
        """Warning notice (logged only, like the dialog's)."""
        self._log(f"Warning: {message}", level="warning")

    # ---- moved from AsyncProcessingDialog (bodies verbatim apart from the hook table) --------

    def _get_opf_spine_map(self, epub_path: str):
        """Return mapping of href/basename/slug -> spine position from content.opf"""
        try:
            import zipfile
            import xml.etree.ElementTree as ET

            spine_map = {}
            with zipfile.ZipFile(epub_path, "r") as zf:
                opf_name = find_epub_opf_member(zf)
                if not opf_name:
                    return spine_map

                opf_content = zf.read(opf_name).decode("utf-8", errors="ignore")

            root = ET.fromstring(opf_content)
            ns = {"opf": "http://www.idpf.org/2007/opf"}
            if root.tag.startswith("{"):
                ns = {"opf": root.tag[1 : root.tag.index("}")]}

            manifest = {}
            for item in root.findall(".//opf:manifest/opf:item", ns):
                item_id = item.get("id")
                href = item.get("href")
                if item_id and href:
                    manifest[item_id] = href

            spine = []
            spine_elem = root.find(".//opf:spine", ns)
            if spine_elem is not None:
                for itemref in spine_elem.findall("opf:itemref", ns):
                    idref = itemref.get("idref")
                    if idref and idref in manifest:
                        spine.append(manifest[idref])

            for idx, href in enumerate(spine):
                base = os.path.basename(href)
                slug = os.path.splitext(base)[0]
                spine_map[href] = idx
                spine_map[base] = idx
                spine_map[slug] = idx

            return spine_map
        except Exception as e:
            print(f"⚠️ Failed to parse OPF spine: {e}")
            return {}

    def _check_selected_status(self):
        """Check status of selected job"""
        if not self.selected_job_id:
            self._async_msgbox('warning', "No Selection", "Please select a job to check status")
            return
            
        try:
            job = self.processor.check_job_status(self.selected_job_id)
            self._refresh_jobs_list()
            
            # Build detailed status message
            status_text = f"Job ID: {job.job_id}\n"
            status_text += f"Provider: {job.provider.upper()}\n"
            status_text += f"Status: {job.status.value}\n"
            status_text += f"State: {job.metadata.get('raw_state', 'Unknown')}\n\n"
            
            # Progress information
            if job.completed_requests > 0 or job.status == AsyncAPIStatus.PROCESSING:
                status_text += f"Progress: {job.completed_requests}/{job.total_requests}\n"
            else:
                status_text += f"Progress: Waiting to start (0/{job.total_requests})\n"
                
            status_text += f"Failed: {job.failed_requests}\n\n"
            
            # Time information
            status_text += f"Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            status_text += f"Last Updated: {job.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if 'last_check' in job.metadata:
                status_text += f"Last Checked: {job.metadata['last_check']}\n"
            # If OpenAI provided an error file, fetch a brief excerpt so the user sees the actual failure reason
            if job.provider == 'openai' and job.metadata.get('error_file_id'):
                snippet = self._fetch_openai_error_snippet(job.metadata['error_file_id'])
                if snippet:
                    status_text += f"\nErrors (first 5):\n{snippet}\n"
                
            # Show output file if available
            if job.output_file:
                status_text += f"\nOutput Ready: {job.output_file}\n"
            
            self._async_msgbox('information', "Job Status", status_text)
            
        except Exception as e:
            self._async_msgbox('critical', "Error", f"Failed to check status: {str(e)}")

    def _fetch_openai_error_snippet(self, error_file_id: str) -> str:
        """
        Download the OpenAI batch error file and return the first few error messages.
        Returns empty string on failure.
        """
        try:
            api_key = self._get_api_key_from_gui()
            if not api_key:
                return ""
            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.get(
                f'https://api.openai.com/v1/files/{error_file_id}/content',
                headers=headers,
                timeout=15
            )
            if response.status_code != 200:
                return ""
            lines = response.text.strip().split('\n')
            snippets = []
            for line in lines[:5]:
                try:
                    obj = json.loads(line)
                    msg = obj.get('error', {}).get('message') or obj.get('message') or str(obj)
                    snippets.append(f"• {msg}")
                except Exception:
                    snippets.append(f"• {line}")
            if len(lines) > 5:
                snippets.append(f"... and {len(lines) - 5} more")
            return "\n".join(snippets)
        except Exception:
            return ""

    def _retrieve_selected_results(self):
        """Retrieve results from selected job"""
        job_ids = self._get_selected_job_ids()
        if not job_ids:
            self._async_msgbox('warning', "No Selection", "Please select one or more jobs to retrieve results")
            return

        # Partition by status
        incompletes = []
        not_found = []
        for job_id in job_ids:
            job = self.processor.jobs.get(job_id)
            if not job:
                not_found.append(job_id)
            elif job.status != AsyncAPIStatus.COMPLETED:
                incompletes.append(job_id)

        if not_found:
            self._async_msgbox('critical', "Error", f"{len(not_found)} selected job(s) not found locally.")
            return
        if incompletes:
            self._async_msgbox('warning', "Job Not Complete",
                f"{len(incompletes)} selected job(s) are not completed yet and were skipped."
            )

        completed_jobs = [jid for jid in job_ids if jid not in incompletes and jid not in not_found]
        if not completed_jobs:
            return

        try:
            if self._async_dialog_visible():
                self._async_set_wait_cursor(True)

            for jid in completed_jobs:
                self._handle_completed_job(jid)

        except Exception as e:
            self._log(f"❌ Error retrieving results: {e}")
            self._async_msgbox('critical', "Error", f"Failed to retrieve some results: {str(e)}")
        finally:
            if self._async_dialog_visible():
                self._async_set_wait_cursor(False)

    def _cancel_selected_job(self):
        """Cancel selected job"""
        job_ids = self._get_selected_job_ids()
        if not job_ids:
            self._async_msgbox('warning', "No Selection", "Please select one or more jobs to cancel")
            return

        cancellable = []
        skipped = []
        for jid in job_ids:
            job = self.processor.jobs.get(jid)
            if not job or job.status in [AsyncAPIStatus.COMPLETED, AsyncAPIStatus.FAILED, AsyncAPIStatus.CANCELLED]:
                skipped.append(jid)
            else:
                cancellable.append(job)

        if not cancellable:
            self._async_msgbox('information', "Nothing to Cancel", "No selected jobs can be cancelled.")
            return

        reply = self._async_msgbox('question', "Cancel Jobs",
            f"Cancel {len(cancellable)} selected job(s)?",
            self._MB_YES | self._MB_NO
        )
        if reply != self._MB_YES:
            return

        # Explicitly RE-ENABLE button here since we are just canceling a remote job
        # and not running a local blocking process that needs the UI disabled.
        # Although cancelling is quick, we don't want to leave the button disabled if it was.
        # But wait, this method (_cancel_selected_job) doesn't disable the start button.
        # The issue might be that the user thinks "Start Async Processing" is disabled *while* a job is running?
        # Or if they cancel, they expect to start a new one immediately?
        
        # If the start button was disabled because a job was running/submitting, ensure it's enabled.
        if hasattr(self, 'start_button'):
            self._async_set_start_enabled(True)

        api_key = self._get_api_key_from_gui()
        successes, failures = 0, []

        for job in cancellable:
            try:
                if job.provider == 'openai':
                    headers = {'Authorization': f'Bearer {api_key}'}
                    response = requests.post(f'https://api.openai.com/v1/batches/{job.job_id}/cancel', headers=headers)
                    if response.status_code == 200:
                        job.status = AsyncAPIStatus.CANCELLED
                        successes += 1
                    else:
                        failures.append((job.job_id, response.text))
                elif job.provider == 'gemini':
                    headers = {'x-goog-api-key': api_key}
                    batch_name = job.job_id if job.job_id.startswith('batches/') else f'batches/{job.job_id}'
                    response = requests.post(f'https://generativelanguage.googleapis.com/v1beta/{batch_name}:cancel', headers=headers)
                    if response.status_code == 200:
                        job.status = AsyncAPIStatus.CANCELLED
                        successes += 1
                    else:
                        failures.append((job.job_id, response.text))
                elif job.provider == 'anthropic':
                    job.status = AsyncAPIStatus.CANCELLED
                    successes += 1
                else:
                    job.status = AsyncAPIStatus.CANCELLED
                    successes += 1

                job.updated_at = datetime.now()
            except Exception as e:
                failures.append((job.job_id, str(e)))

        self.processor._save_jobs()
        self._refresh_jobs_list()

        summary = f"Cancelled: {successes}"
        if skipped:
            summary += f"\nSkipped (already done/failed): {len(skipped)}"
        if failures:
            summary += f"\nFailed: {len(failures)}"
        self._async_msgbox('information', "Cancel Jobs", summary)

    def _cancel_openai_job(self, job, api_key):
        """Cancel OpenAI batch job"""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # OpenAI has a specific cancel endpoint
        cancel_url = f"https://api.openai.com/v1/batches/{job.job_id}/cancel"
        
        response = requests.post(cancel_url, headers=headers)
        
        if response.status_code not in [200, 204]:
            raise Exception(f"OpenAI cancellation failed: {response.text}")
            
        logger.info(f"OpenAI job {job.job_id} cancelled successfully")

    def _cancel_anthropic_job(self, job, api_key):
        """Cancel Anthropic batch job"""
        headers = {
            'X-API-Key': api_key,
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'message-batches-2024-09-24'
        }
        
        # Anthropic uses DELETE method for cancellation
        cancel_url = f"https://api.anthropic.com/v1/messages/batches/{job.job_id}"
        
        response = requests.delete(cancel_url, headers=headers)
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Anthropic cancellation failed: {response.text}")
            
        logger.info(f"Anthropic job {job.job_id} cancelled successfully")

    def _cancel_gemini_job(self, job, api_key):
        """Cancel Gemini batch job"""
        try:
            from google import genai
            
            # Create client
            client = genai.Client(api_key=api_key)
            
            # Try to cancel using the SDK
            # Note: The SDK might not have a cancel method yet
            if hasattr(client.batches, 'cancel'):
                client.batches.cancel(name=job.job_id)
                logger.info(f"Gemini job {job.job_id} cancelled successfully")
            else:
                # If SDK doesn't support cancellation, inform the user
                raise Exception(
                    "Gemini batch cancellation is not supported yet.\n"
                    "The job will continue to run and complete within 24 hours.\n"
                    "You can check the status later to retrieve results."
                )
                
        except AttributeError:
            # SDK doesn't have cancel method
            raise Exception(
                "Gemini batch cancellation is not available in the current SDK.\n"
                "The job will continue to run and complete within 24 hours."
            )
        except Exception as e:
            # Check if it's a permission error
            if "403" in str(e) or "PERMISSION_DENIED" in str(e):
                raise Exception(
                    "Gemini batch jobs cannot be cancelled once submitted.\n"
                    "The job will complete within 24 hours and you can retrieve results then."
                )
            else:
                # Re-raise other errors
                raise

    def _cancel_mistral_job(self, job, api_key):
        """Cancel Mistral batch job"""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Mistral batch cancellation endpoint
        cancel_url = f"https://api.mistral.ai/v1/batch/jobs/{job.job_id}/cancel"
        
        response = requests.post(cancel_url, headers=headers)
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Mistral cancellation failed: {response.text}")
            
        logger.info(f"Mistral job {job.job_id} cancelled successfully")

    def _cancel_groq_job(self, job, api_key):
        """Cancel Groq batch job"""
        # Groq uses OpenAI-compatible endpoints
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        cancel_url = f"https://api.groq.com/openai/v1/batch/{job.job_id}/cancel"
        
        response = requests.post(cancel_url, headers=headers)
        
        if response.status_code not in [200, 204]:
            raise Exception(f"Groq cancellation failed: {response.text}")
            
        logger.info(f"Groq job {job.job_id} cancelled successfully")

    def _estimate_cost(self):
        """Estimate cost for current file"""
        # Get current file info
        if not hasattr(self.gui, 'file_path') or not self.gui.file_path:
            self._async_msgbox('warning', "No File", "Please select a file first")
            return
        
        try:
            # Show analyzing message
            self._async_set_cost_info("Analyzing file...")
            self._async_process_events()
            
            file_path = self.gui.file_path
            # Get model name - handle both tkinter and PySide6
            if hasattr(self.gui.model_var, 'get'):
                model = self.gui.model_var.get()
            else:
                model = str(self.gui.model_var) if self.gui.model_var else ""
            
            # Calculate overhead tokens (system prompt + glossary)
            overhead_tokens = 0
            
            # Count system prompt tokens
            # Get text from QTextEdit (PySide6) or Text widget (tkinter)
            if hasattr(self.gui.prompt_text, 'toPlainText'):
                system_prompt = self.gui.prompt_text.toPlainText().strip()
            else:
                system_prompt = self.gui.prompt_text.get("1.0", "end").strip()
            if system_prompt:
                overhead_tokens += self.count_tokens(system_prompt, model)
                logger.info(f"System prompt tokens: {overhead_tokens}")
            
            # Count glossary tokens if enabled
            glossary_tokens = 0
            
            # Check if glossary should be appended - match the logic from _prepare_environment_variables
            append_glossary = False
            if hasattr(self.gui, 'append_glossary_var'):
                if hasattr(self.gui.append_glossary_var, 'get'):
                    append_glossary = self.gui.append_glossary_var.get()
                else:
                    append_glossary = bool(self.gui.append_glossary_var)
            
            if (hasattr(self.gui, 'manual_glossary_path') and 
                self.gui.manual_glossary_path and 
                append_glossary):  # This is the key check!
                
                try:
                    glossary_path = self.gui.manual_glossary_path
                    logger.info(f"Loading glossary from: {glossary_path}")
                    
                    if os.path.exists(glossary_path):
                        with open(glossary_path, 'r', encoding='utf-8') as f:
                            glossary_data = json.load(f)
                        
                        # Format glossary same way as in translation
                        #glossary_text = self._format_glossary_for_prompt(glossary_data)
                        
                        # Add append prompt if available
                        append_prompt = self.gui.append_glossary_prompt if hasattr(self.gui, 'append_glossary_prompt') else ''
                        
                        if append_prompt:
                            if '{glossary}' in append_prompt:
                                glossary_text = append_prompt.replace('{glossary}', glossary_text)
                            else:
                                glossary_text = f"{append_prompt}\n{glossary_text}"
                        else:
                            glossary_text = f"Glossary:\n{glossary_text}"
                        
                        glossary_tokens = self.count_tokens(glossary_text, model)
                        overhead_tokens += glossary_tokens
                        logger.info(f"Loaded glossary with {glossary_tokens} tokens")
                    else:
                        print(f"Glossary file not found: {glossary_path}")
                        
                except Exception as e:
                    print(f"Failed to load glossary: {e}")
            
            logger.info(f"Total overhead per chapter: {overhead_tokens} tokens")
            
            # Actually extract chapters and count tokens
            num_chapters = 0
            total_content_tokens = 0  # Just the chapter content
            chapters_needing_chunking = 0
            
            if file_path.lower().endswith('.epub'):
                # Import and use EPUB extraction
                try:
                    import ebooklib
                    from ebooklib import epub
                    from bs4 import BeautifulSoup
                    
                    book = epub.read_epub(file_path)
                    chapters = []
                    
                    # Extract text chapters
                    for item in book.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            soup = BeautifulSoup(item.get_content(), 'html.parser')
                            text = soup.get_text(separator='\n').strip()
                            if len(text) > 500:  # Minimum chapter length
                                chapters.append(text)
                                
                    num_chapters = len(chapters)
                    
                    # Count tokens for each chapter (sample more for better accuracy)
                    sample_size = min(20, num_chapters)  # Sample up to 20 chapters for better accuracy
                    sampled_content_tokens = 0

                    # Resolve output token limit for chunking (honor disable flag)
                    try:
                        output_limit_disabled = bool(getattr(self.gui, 'token_limit_disabled', False))
                    except Exception:
                        output_limit_disabled = False
                    output_limit_disabled = output_limit_disabled or bool(self.gui.config.get('token_limit_disabled', False))
                    
                    if output_limit_disabled:
                        output_limit = 0  # unlimited
                    else:
                        try:
                            output_limit = int(env_vars.get('MAX_OUTPUT_TOKENS', 65536))
                        except Exception:
                            output_limit = 65536
                        # Count just the content tokens
                        content_tokens = self.count_tokens(chapter_text, model)
                        sampled_content_tokens += content_tokens
                        
                        # Check if needs chunking (including overhead)
                        total_chapter_tokens = content_tokens + overhead_tokens
                        if output_limit == 0:
                            needs_chunking = False
                        else:
                            needs_chunking = total_chapter_tokens > output_limit * 0.8

                        if needs_chunking:
                            chapters_needing_chunking += 1
                        # Update progress
                        if i % 5 == 0:
                            self._async_set_cost_info(f"Analyzing chapters... {i+1}/{sample_size}")
                            self._async_process_events()
                            
                    # Calculate average based on actual sample
                    if sample_size > 0:
                        avg_content_tokens_per_chapter = sampled_content_tokens // sample_size
                        # Extrapolate chunking needs if we didn't sample all
                        if num_chapters > sample_size:
                            chapters_needing_chunking = int(chapters_needing_chunking * (num_chapters / sample_size))
                    else:
                        avg_content_tokens_per_chapter = 15000  # Default
                        
                except Exception as e:
                    print(f"Failed to analyze EPUB: {e}")
                    # Fall back to estimates
                    num_chapters = 50
                    avg_content_tokens_per_chapter = 15000
                    
            elif file_path.lower().endswith('.txt'):
                # Import and use TXT extraction
                try:
                    from txt_processor import TextFileProcessor
                    
                    processor = TextFileProcessor(file_path, '')
                    chapters = processor.extract_chapters()
                    num_chapters = len(chapters)
                    
                    # Count tokens
                    sample_size = min(20, num_chapters)  # Sample up to 20 chapters
                    sampled_content_tokens = 0

                    # Resolve token limit for chunking (honor disable flag)
                    token_limit_disabled = bool(getattr(self.gui, 'token_limit_disabled', False)) or bool(self.gui.config.get('token_limit_disabled', False))
                    if token_limit_disabled:
                        token_limit = 0  # unlimited
                    else:
                        try:
                            raw_limit = self.gui.token_limit_entry.text() if hasattr(self.gui.token_limit_entry, 'text') else self.gui.token_limit_entry.get()
                        except Exception:
                            raw_limit = ''
                        raw_limit = (raw_limit or '').strip()
                        if raw_limit:
                            try:
                                token_limit = int(raw_limit)
                            except Exception:
                                token_limit = 65536
                        else:
                            try:
                                token_limit = int(env_vars.get('MAX_OUTPUT_TOKENS', 65536))
                            except Exception:
                                token_limit = 65536
                            try:
                                cfg_limit = self.gui.config.get('token_limit')
                                if cfg_limit:
                                    token_limit = int(cfg_limit)
                            except Exception:
                                pass

                    for i, chapter_text in enumerate(chapters[:sample_size]):
                        # Count just the content tokens
                        content_tokens = self.count_tokens(chapter_text, model)
                        sampled_content_tokens += content_tokens
                        
                        # Check if needs chunking (including overhead)
                        total_chapter_tokens = content_tokens + overhead_tokens
                        if token_limit == 0:
                            needs_chunking = False
                        else:
                            needs_chunking = total_chapter_tokens > token_limit * 0.8
                        if needs_chunking:
                            chapters_needing_chunking += 1
                        
                        # Update progress
                        if i % 5 == 0:
                            self._async_set_cost_info(f"Analyzing chapters... {i+1}/{sample_size}")
                            self._async_process_events()
                            
                    # Calculate average based on actual sample
                    if sample_size > 0:
                        avg_content_tokens_per_chapter = sampled_content_tokens // sample_size
                        # Extrapolate chunking needs
                        if num_chapters > sample_size:
                            chapters_needing_chunking = int(chapters_needing_chunking * (num_chapters / sample_size))
                    else:
                        avg_content_tokens_per_chapter = 15000  # Default
                        
                except Exception as e:
                    print(f"Failed to analyze TXT: {e}")
                    # Fall back to estimates
                    num_chapters = 50
                    avg_content_tokens_per_chapter = 15000
            else:
                # Unsupported format
                self._async_set_cost_info(
                    "Unsupported file format. Only EPUB and TXT are supported."
                )
                return
            
            # Calculate costs
            processable_chapters = num_chapters - chapters_needing_chunking
            
            if processable_chapters <= 0:
                self._async_set_cost_info(
                    f"Warning: All {num_chapters} chapters require chunking.\n"
                    f"Async APIs do not support chunked chapters.\n"
                    f"Consider using regular batch translation instead."
                )
                return
            
            # Add overhead to get total average tokens per chapter
            avg_total_tokens_per_chapter = avg_content_tokens_per_chapter + overhead_tokens
            
            # Get the translation compression factor from GUI
            if hasattr(self.gui.compression_factor_var, 'get'):
                compression_factor = float(self.gui.compression_factor_var.get() or 1.0)
            else:
                compression_factor = float(self.gui.compression_factor_var or 1.0)
            
            # Get accurate cost estimate
            async_cost, regular_cost = self.processor.estimate_cost(
                processable_chapters, 
                avg_total_tokens_per_chapter,  # Now includes content + system prompt + glossary
                model,
                compression_factor
            )
            
            # Update any existing jobs for this file with the accurate estimate
            current_file = self.gui.file_path
            for job_id, job in self.processor.jobs.items():
                # Check if this job is for the current file and model
                if (job.metadata and 
                    job.metadata.get('source_file') == current_file and 
                    job.model == model and 
                    job.status in [AsyncAPIStatus.PENDING, AsyncAPIStatus.PROCESSING]):
                    # Update the cost estimate
                    job.cost_estimate = async_cost
                    job.updated_at = datetime.now()
            
            # Save updated jobs
            self.processor._save_jobs()
            
            # Refresh the display
            self._refresh_jobs_list()
            
            # Build detailed message
            cost_text = f"File analysis complete!\n\n"
            cost_text += f"Total chapters: {num_chapters}\n"
            cost_text += f"Average content tokens per chapter: {avg_content_tokens_per_chapter:,}\n"
            cost_text += f"Overhead per chapter: {overhead_tokens:,} tokens"
            if glossary_tokens > 0:
                cost_text += f" (system: {overhead_tokens - glossary_tokens:,}, glossary: {glossary_tokens:,})"
            cost_text += f"\nTotal input tokens per chapter: {avg_total_tokens_per_chapter:,}\n"
            
            if chapters_needing_chunking > 0:
                cost_text += f"\nChapters requiring chunking: {chapters_needing_chunking} (will be skipped)\n"
                cost_text += f"Processable chapters: {processable_chapters}\n"
            
            cost_text += f"\nEstimated cost for {processable_chapters} chapters:\n"
            cost_text += f"Regular processing: ${regular_cost:.2f}\n"
            cost_text += f"Async processing: ${async_cost:.2f} (50% savings: ${regular_cost - async_cost:.2f})"
            
            # Add note about token calculation
            cost_text += f"\n\nNote: Costs include input (~{avg_total_tokens_per_chapter:,}) and "
            cost_text += f"output (~{int(avg_content_tokens_per_chapter * compression_factor):,}) tokens per chapter."

            
            self._async_set_cost_info(cost_text)
            
        except Exception as e:
            self._async_set_cost_info(
                f"Error estimating cost: {str(e)}"
            )
            print(f"Cost estimation error: {traceback.format_exc()}")

    def _estimate_batch_cost(self):
        """
        Run the same detailed estimation used by the 'Estimate Cost Only' button.
        This is invoked automatically right after a batch is submitted so the
        job list shows the accurate cost (including system prompt/glossary
        overhead and the user-selected compression factor).
        """
        return self._estimate_cost()

    def count_tokens(self, text, model):
        """Count tokens in text (content only - system prompt and glossary are counted separately)"""
        try:
            import tiktoken
            
            # Get base encoding for model
            if model.startswith(('gpt-4', 'gpt-3')):
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
            elif model.startswith('claude'):
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Just count the text tokens - don't include system/glossary here
            # They are counted separately in _estimate_cost to avoid confusion
            text_tokens = len(encoding.encode(text))
            
            return text_tokens
            
        except Exception as e:
            # Fallback: estimate ~4 characters per token
            return len(text) // 4

    def _start_processing(self):
        """Start async processing"""
        # Get model name - handle both tkinter and PySide6
        if hasattr(self.gui.model_var, 'get'):
            model = self.gui.model_var.get()
        else:
            model = str(self.gui.model_var) if self.gui.model_var else ""
        
        if not self.processor.supports_async(model):
            reply = self._async_msgbox('warning', "Possibly Unsupported",
                f"Model '{model}' may not support async processing.\n"
                "Known supported providers: Gemini, Anthropic, OpenAI, Mistral, Groq\n\n"
                "Would you like to try anyway?",
                self._MB_YES | self._MB_NO
            )
            if reply != self._MB_YES:
                return
        
        # Add special check for Gemini
        if model.lower().startswith('gemini'):
            reply = self._async_msgbox('question', "Gemini Batch API",
                "Note: Gemini's batch API may not be publicly available yet.\n"
                "This feature is experimental for Gemini models.\n\n"
                "Would you like to try anyway?",
                self._MB_YES | self._MB_NO
            )
            if reply != self._MB_YES:
                return
        

            
        if not hasattr(self.gui, 'file_path') or not self.gui.file_path:
            self._async_msgbox('warning', "No File", "Please select a file to translate first")
            return
            
        # Confirm start
        reply = self._async_msgbox('question', "Start Async Processing",
            "Start async batch processing?\n\n"
            "This will submit all chapters for processing at 50% discount.\n"
            "Processing may take up to 24 hours.",
            self._MB_YES | self._MB_NO
        )
        if reply != self._MB_YES:
            return
        
        # Disable buttons during processing
        self._async_set_start_enabled(False)
        
        # Start processing in background thread
        self.processing_thread = threading.Thread(
            target=self._async_processing_worker,
            daemon=True
        )
        self.processing_thread.start()

    def _async_processing_worker(self):
        """Worker thread for async processing"""
        try:
            self._log("Starting async processing preparation...")
            
            # Get all settings from GUI
            file_path = self.gui.file_path
            # Get model name - handle both tkinter and PySide6
            if hasattr(self.gui.model_var, 'get'):
                model = self.gui.model_var.get()
            else:
                model = str(self.gui.model_var) if self.gui.model_var else ""
            
            # Get API key (AuthGPT models will use OAuth token automatically)
            api_key = self._get_api_key_from_gui()
            
            # AuthGPT OAuth tokens are NOT accepted by OpenAI's Batch API
            # (requires sk-* secret keys). If user wants async with an authgpt/ model,
            # they need a real OpenAI API key entered in the API key field.
            is_authgpt = model.lower().startswith(('authgpt/', 'authgem/', 'authgem-key/', 'authgem-vertex/'))
            if is_authgpt:
                if not api_key or not api_key.startswith('sk-'):
                    self._show_error(
                        "AuthGPT models cannot use async batch mode with OAuth tokens.\n\n"
                        "The OpenAI Batch API requires a secret API key (sk-...).\n"
                        "ChatGPT OAuth tokens are only accepted by the ChatGPT backend,\n"
                        "which does not support batch processing.\n\n"
                        "To use async mode, enter your OpenAI API key in the API key field."
                    )
                    return
                # User provided a real sk-* key — strip authgpt/ prefix and use OpenAI Batch API
                model = model.split('/', 1)[1]
                self._log(f"AuthGPT model with OpenAI key: submitting '{model}' to Batch API")
            
            if not api_key:
                self._show_error("API key is required")
                return
                
            # Prepare environment variables like the main translation
            env_vars = self._prepare_environment_variables()
            
            # Extract chapters
            self._log("Extracting chapters from file...")
            chapters, chapter_mapping = self._extract_chapters_for_async(file_path, env_vars)  # CHANGED: Now unpacking both values
            
            if not chapters:
                self._show_error("No chapters found in file")
                return
                
            self._log(f"Found {len(chapters)} chapters to process")
            
            # Check for chapters that need chunking
            chapters_to_process = []
            skipped_count = 0
            
            for chapter in chapters:
                if chapter.get('needs_chunking', False):
                    skipped_count += 1
                    self._log(f"Skipping chapter {chapter['number']} - requires chunking")
                else:
                    chapters_to_process.append(chapter)
                    
            if skipped_count > 0:
                self._log(f"⚠️ Skipped {skipped_count} chapters that require chunking")
                
            if not chapters_to_process:
                self._show_error("All chapters require chunking. Async APIs don't support chunked chapters.")
                # Re-enable button before returning
                self._async_single_shot(0, lambda: self._async_set_start_enabled(True))
                return

            # Prepare batch request
            self._log("Preparing batch request...")
            batch_data = self.processor.prepare_batch_request(chapters_to_process, model)
            
            # Submit batch
            self._log("Submitting batch to API...")
            job = self._submit_batch_sync(batch_data, model, api_key)
            
            # Save job with chapter mapping in metadata
            job.metadata = job.metadata or {}
            job.metadata['chapter_mapping'] = chapter_mapping  # ADDED: Store mapping for later use
            job.metadata['env'] = env_vars  # preserve env to know extraction mode on save
            job.metadata['source_file'] = file_path  # ensure job remembers the originating input file
            
            # Save job
            self.processor.jobs[job.job_id] = job
            self.processor._save_jobs()
            
            # Update UI on the GUI thread using dialog affinity
            self._async_single_shot(0, self.dialog, self._refresh_jobs_list)
            
            self._log(f"✅ Batch submitted successfully! Job ID: {job.job_id}")
            
            # Show success message
            self._show_info(
                "Batch Submitted",
                f"Successfully submitted {len(chapters_to_process)} chapters for async processing.\n\n"
                f"Job ID: {job.job_id}\n\n"
                "You can close this dialog and check back later for results.\n\n"
                "Tip: Use the 'Estimate Cost Only' button to get accurate cost estimates before submitting."
            )
            
            # Run immediate cost estimate for the newly created job
            # This populates the Cost Estimate column without waiting for user action
            try:
                # We need to run this on the main thread because it updates UI.
                # Pass the dialog as the context so the callback is invoked on the GUI thread.
                self._async_single_shot(0, self.dialog, lambda: self._estimate_batch_cost())
            except Exception as e:
                print(f"Failed to auto-run cost estimate: {e}")
            
            # Start polling if requested
            if self._async_wait_for_completion():
                self._start_polling(job.job_id)
                
        except Exception as e:
            error_msg = str(e)
            # Truncate very long error messages (e.g. full JSON responses)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + '\n... (truncated)'
            self._log(f"❌ Error: {error_msg}")
            print(f"Async processing error: {traceback.format_exc()}")
            self._show_error(f"Failed to start async processing:\n\n{error_msg}")
        finally:
            # Always re-enable the button on completion/failure
            try:
                self._async_single_shot(0, self.dialog, lambda: self._async_set_start_enabled(True))
            except Exception:
                try:
                    self._async_set_start_enabled(True)
                except Exception:
                    pass

    def _prepare_environment_variables(self):
        """Prepare environment variables from GUI settings"""
        def _val(obj, default=None):
            try:
                if hasattr(obj, "isChecked"):
                    return obj.isChecked()
                return obj.get()
            except Exception:
                return obj if obj is not None else default

        def _bool_val(obj, default=False):
            value = _val(obj, default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def _text(widget, default=""):
            if widget is None:
                return default
            if hasattr(widget, "text"):
                return widget.text()
            if hasattr(widget, "get"):
                return widget.get()
            return str(widget) if widget else default

        env_vars = {}
        
        # Core settings - handle both PySide6 and tkinter
        if hasattr(self.gui.model_var, 'get'):
            env_vars['MODEL'] = self.gui.model_var.get()
        else:
            env_vars['MODEL'] = str(self.gui.model_var) if self.gui.model_var else ""
            
        if hasattr(self.gui.api_key_entry, 'get'):
            env_vars['API_KEY'] = self.gui.api_key_entry.get().strip()
        else:
            env_vars['API_KEY'] = self.gui.api_key_entry.text().strip()
            
        env_vars['OPENAI_API_KEY'] = env_vars['API_KEY']
        env_vars['OPENAI_OR_Gemini_API_KEY'] = env_vars['API_KEY']
        env_vars['GEMINI_API_KEY'] = env_vars['API_KEY']
        
        if hasattr(self.gui.lang_var, 'get'):
            env_vars['PROFILE_NAME'] = self.gui.lang_var.get().lower()
        else:
            env_vars['PROFILE_NAME'] = str(self.gui.lang_var).lower() if self.gui.lang_var else ""
            
        if hasattr(self.gui.contextual_var, 'get'):
            env_vars['CONTEXTUAL'] = '1' if self.gui.contextual_var.get() else '0'
        else:
            env_vars['CONTEXTUAL'] = '1' if self.gui.contextual_var else '0'
            
        raw_max_output_tokens = getattr(self.gui, 'max_output_tokens', 65536)
        clamped_max_output_tokens = _clamp_output_tokens_for_selected_model(
            env_vars.get('MODEL', ''),
            raw_max_output_tokens,
            default=65536,
        )
        env_vars['MAX_OUTPUT_TOKENS'] = str(clamped_max_output_tokens)
        if str(raw_max_output_tokens) != str(clamped_max_output_tokens):
            logger.info(
                "[ASYNC] Clamped MAX_OUTPUT_TOKENS for %s: %s -> %s",
                env_vars.get('MODEL', ''),
                raw_max_output_tokens,
                clamped_max_output_tokens,
            )
        
        # Resolve target language for prompt substitution
        target_lang = self.gui.config.get('output_language') or ''
        if not target_lang and hasattr(self.gui, 'lang_var'):
            try:
                target_lang = self.gui.lang_var.get()
            except Exception:
                target_lang = str(self.gui.lang_var) if self.gui.lang_var else ''
        target_lang = target_lang or 'English'

        if hasattr(self.gui.prompt_text, 'get'):
            system_prompt = self.gui.prompt_text.get("1.0", "end").strip()
        else:
            system_prompt = self.gui.prompt_text.toPlainText().strip()
        env_vars['SYSTEM_PROMPT'] = system_prompt.replace('{target_lang}', target_lang).replace('{split_marker_instruction}', '')
        
        # Async processing does not support request merging logic, so force it off
        env_vars['REQUEST_MERGING_ENABLED'] = '0'
            
        env_vars['TRANSLATION_TEMPERATURE'] = _text(getattr(self.gui, 'trans_temp', None), '0.3')
        env_vars['TRANSLATION_HISTORY_LIMIT'] = _text(getattr(self.gui, 'trans_history', None), '8')
        # Explicitly disable thought streaming for async jobs
        env_vars['ENABLE_STREAMING'] = "1" if _val(getattr(self.gui, 'enable_streaming_var', None), self.gui.config.get('enable_streaming', False)) else "0"
        env_vars['ALLOW_BATCH_STREAM_LOGS'] = "1" if _val(getattr(self.gui, 'allow_batch_stream_logs_var', None), self.gui.config.get('allow_batch_stream_logs', False)) else "0"
        env_vars['ALLOW_AUTHGPT_BATCH_STREAM_LOGS'] = "1" if _val(getattr(self.gui, 'allow_authgpt_batch_stream_logs_var', None), self.gui.config.get('allow_authgpt_batch_stream_logs', False)) else "0"
        env_vars['STREAM_THINKING_LOGS'] = "1" if _val(getattr(self.gui, 'stream_thinking_logs_var', None), self.gui.config.get('stream_thinking_logs', False)) else "0"
        env_vars['ENABLE_THOUGHTS'] = '0'
        
        # API settings - handle both PySide6 and tkinter
        if hasattr(self.gui.delay_entry, 'get'):
            env_vars['SEND_INTERVAL_SECONDS'] = str(self.gui.delay_entry.get())
        else:
            env_vars['SEND_INTERVAL_SECONDS'] = str(self.gui.delay_entry.text() if hasattr(self.gui.delay_entry, 'text') else '2')
            
        # Token limit resolution order:
        # 0) if token_limit_disabled -> unlimited (0)
        # 1) token_limit_entry if provided by user
        # 2) GUI max_output_tokens / env_vars['MAX_OUTPUT_TOKENS'] (default 65536)
        # 3) config token_limit
        # 4) hard default 65536
        if hasattr(self.gui, 'token_limit_entry'):
            raw_limit = _text(self.gui.token_limit_entry, '').strip()
        else:
            raw_limit = ''

        token_limit_disabled = False
        try:
            token_limit_disabled = bool(getattr(self.gui, 'token_limit_disabled', False))
        except Exception:
            pass
        token_limit_disabled = token_limit_disabled or bool(self.gui.config.get('token_limit_disabled', False))

        if token_limit_disabled:
            resolved_token_limit = 0  # unlimited
            source = "disabled (unlimited)"
        elif raw_limit:
            resolved_token_limit = int(raw_limit) if str(raw_limit).lstrip('+-').isdigit() else 65536
            source = "token_limit_entry"
        else:
            try:
                gui_max = int(env_vars.get('MAX_OUTPUT_TOKENS', 0))
            except Exception:
                gui_max = 0
            if gui_max > 0:
                resolved_token_limit = gui_max
                source = "max_output_tokens"
            else:
                cfg_limit = self.gui.config.get('token_limit')
                resolved_token_limit = int(cfg_limit) if cfg_limit else 65536
                source = "config_token_limit" if cfg_limit else "hard_default"

        env_vars['TOKEN_LIMIT'] = str(resolved_token_limit)
        env_vars['TOKEN_LIMIT_SOURCE'] = source
        logger.info(f"[ASYNC] TOKEN_LIMIT={env_vars['TOKEN_LIMIT']} (source={source}, raw_field='{raw_limit}', max_output_tokens={getattr(self.gui, 'max_output_tokens', None)}, config_token_limit={self.gui.config.get('token_limit')}, token_limit_disabled={token_limit_disabled})")

        # Book title translation - replace {target_lang} with output language
        env_vars['TRANSLATE_BOOK_TITLE'] = "1" if _val(self.gui.translate_book_title_var, False) else "0"
        output_lang = self.gui.config.get('output_language', 'English')
        book_title_prompt = self.gui.book_title_prompt if hasattr(self.gui, 'book_title_prompt') else ''
        book_title_system_prompt = self.gui.config.get('book_title_system_prompt', 
            "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content.")
        env_vars['BOOK_TITLE_PROMPT'] = book_title_prompt.replace('{target_lang}', output_lang)
        env_vars['BOOK_TITLE_SYSTEM_PROMPT'] = book_title_system_prompt.replace('{target_lang}', output_lang)
        
        # Processing options
        env_vars['CHAPTER_RANGE'] = _text(getattr(self.gui, 'chapter_range_entry', None), '').strip()
        env_vars['USE_SPINE_ORDER'] = "1" if _val(
            getattr(self.gui, 'use_spine_order_checkbox', None), False
        ) else "0"
        env_vars['REMOVE_AI_ARTIFACTS'] = str(getattr(self.gui, 'REMOVE_AI_ARTIFACTS_var', 'off') or 'off')
        env_vars['BATCH_TRANSLATION'] = "1" if _val(self.gui.batch_translation_var, False) else "0"
        env_vars['BATCH_SIZE'] = _val(self.gui.batch_size_var, 1)
        env_vars['API_QUEUE_SIZE'] = str(
            _val(getattr(self.gui, 'api_queue_var', 4), 4)
        )
        env_vars['BATCHING_MODE'] = str(_val(getattr(self.gui, 'batch_mode_var', 'direct'), 'direct'))
        env_vars['BATCH_GROUP_SIZE'] = str(_val(getattr(self.gui, 'batch_group_size_var', 3), 3))
        # Backward compatibility for downstream components expecting CONSERVATIVE_BATCHING
        env_vars['CONSERVATIVE_BATCHING'] = "1" if env_vars['BATCHING_MODE'] == 'conservative' else "0"
        
        # Anti-duplicate parameters
        env_vars['ENABLE_ANTI_DUPLICATE'] = '1' if hasattr(self.gui, 'enable_anti_duplicate_var') and _val(self.gui.enable_anti_duplicate_var, False) else '0'
        env_vars['TOP_P'] = str(_val(self.gui.top_p_var, 1.0)) if hasattr(self.gui, 'top_p_var') else '1.0'
        env_vars['MIN_P'] = str(_val(self.gui.min_p_var, 0.0)) if hasattr(self.gui, 'min_p_var') else '0.0'
        env_vars['BYPASS_MIN_P_ALLOWLIST'] = '1' if hasattr(self.gui, 'bypass_min_p_allowlist_var') and _val(self.gui.bypass_min_p_allowlist_var, False) else '0'
        env_vars['TOP_K'] = str(_val(self.gui.top_k_var, 0)) if hasattr(self.gui, 'top_k_var') else '0'
        env_vars['FREQUENCY_PENALTY'] = str(_val(self.gui.frequency_penalty_var, 0.0)) if hasattr(self.gui, 'frequency_penalty_var') else '0.0'
        env_vars['PRESENCE_PENALTY'] = str(_val(self.gui.presence_penalty_var, 0.0)) if hasattr(self.gui, 'presence_penalty_var') else '0.0'
        env_vars['REPETITION_PENALTY'] = str(_val(self.gui.repetition_penalty_var, 1.0)) if hasattr(self.gui, 'repetition_penalty_var') else '1.0'
        env_vars['CANDIDATE_COUNT'] = str(_val(self.gui.candidate_count_var, 1)) if hasattr(self.gui, 'candidate_count_var') else '1'
        env_vars['CUSTOM_STOP_SEQUENCES'] = _val(self.gui.custom_stop_sequences_var, '') if hasattr(self.gui, 'custom_stop_sequences_var') else ''
        env_vars['LOGIT_BIAS_ENABLED'] = '1' if hasattr(self.gui, 'logit_bias_enabled_var') and _val(self.gui.logit_bias_enabled_var, False) else '0'
        env_vars['LOGIT_BIAS_STRENGTH'] = str(_val(self.gui.logit_bias_strength_var, -0.5)) if hasattr(self.gui, 'logit_bias_strength_var') else '-0.5'
        env_vars['BIAS_COMMON_WORDS'] = '1' if hasattr(self.gui, 'bias_common_words_var') and _val(self.gui.bias_common_words_var, False) else '0'
        env_vars['BIAS_REPETITIVE_PHRASES'] = '1' if hasattr(self.gui, 'bias_repetitive_phrases_var') and _val(self.gui.bias_repetitive_phrases_var, False) else '0'
        # Glossary settings
        env_vars['MANUAL_GLOSSARY'] = self.gui.manual_glossary_path if hasattr(self.gui, 'manual_glossary_path') and self.gui.manual_glossary_path else ''
        env_vars['DISABLE_AUTO_GLOSSARY'] = "0" if _val(self.gui.enable_auto_glossary_var, False) else "1"
        env_vars['DISABLE_GLOSSARY_TRANSLATION'] = "0" if _val(self.gui.enable_auto_glossary_var, False) else "1"
        env_vars['APPEND_GLOSSARY'] = "1" if _val(self.gui.append_glossary_var, False) else "0"
        env_vars['APPEND_GLOSSARY_PROMPT'] = self.gui.append_glossary_prompt if hasattr(self.gui, 'append_glossary_prompt') else ''
        auto_glossary_mode = self.gui.config.get('auto_glossary_mode', 'off')
        env_vars['AUTO_GLOSSARY_MODE'] = auto_glossary_mode
        env_vars['SINGLE_PASS_GLOSSARY_MODE'] = '1' if auto_glossary_mode == 'single_pass' else ''
        env_vars['SINGLE_PASS_GLOSSARY_HEADER_PROMPT'] = self.gui.config.get('single_pass_glossary_header_prompt', '')
        env_vars['GLOSSARY_CUSTOM_ENTRY_TYPES'] = json.dumps(
            getattr(self.gui, 'custom_entry_types', self.gui.config.get('custom_entry_types', {}))
        )
        env_vars['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(
            getattr(self.gui, 'custom_glossary_fields', self.gui.config.get('custom_glossary_fields', []))
        )
        env_vars['GLOSSARY_ENTRY_TYPE_FILTER_MODE'] = self.gui.config.get('glossary_entry_type_filter_mode', 'none')
        env_vars['GLOSSARY_MIN_FREQUENCY'] = _val(self.gui.glossary_min_frequency_var, 0)
        env_vars['GLOSSARY_MAX_NAMES'] = _val(self.gui.glossary_max_names_var, 0)
        env_vars['GLOSSARY_MAX_TITLES'] = _val(self.gui.glossary_max_titles_var, 0)
        env_vars['GLOSSARY_BATCH_SIZE'] = _val(getattr(self.gui, 'glossary_batch_size_var', None), 0)
        env_vars['GLOSSARY_DUPLICATE_KEY_MODE'] = self.gui.config.get('glossary_duplicate_key_mode', 'auto')
        env_vars['GLOSSARY_DUPLICATE_CUSTOM_FIELD'] = self.gui.config.get('glossary_duplicate_custom_field', '')
        # Compress glossary toggle (was missing, so async path wasn't compressing)
        if hasattr(self.gui, 'compress_glossary_prompt_var'):
            env_vars['COMPRESS_GLOSSARY_PROMPT'] = "1" if _val(self.gui.compress_glossary_prompt_var, False) else "0"
        else:
            env_vars['COMPRESS_GLOSSARY_PROMPT'] = "1" if self.gui.config.get('compress_glossary_prompt', False) else "0"
        # Two checkboxes resolve into one engine value; 'new' wins over 'shadow'.
        if hasattr(self.gui, 'compress_glossary_precise_matching_var'):
            precise_matching = _val(self.gui.compress_glossary_precise_matching_var, True)
        elif hasattr(self.gui, 'precise_matching_checkbox'):
            precise_matching = _val(self.gui.precise_matching_checkbox, False)
        else:
            precise_matching = self.gui.config.get('compress_glossary_precise_matching', True)
        if hasattr(self.gui, 'compress_glossary_shadow_log_var'):
            shadow_log = _val(self.gui.compress_glossary_shadow_log_var, False)
        elif hasattr(self.gui, 'shadow_log_matching_checkbox'):
            shadow_log = _val(self.gui.shadow_log_matching_checkbox, False)
        else:
            shadow_log = self.gui.config.get('compress_glossary_shadow_log', False)
        env_vars['GLOSSARY_MATCH_ENGINE'] = (
            "new" if precise_matching else ("shadow" if shadow_log else "legacy")
        )
        if hasattr(self.gui, 'compress_glossary_consider_translated_column_var'):
            consider_translated_column = _val(self.gui.compress_glossary_consider_translated_column_var, False)
        elif hasattr(self.gui, 'consider_translated_compression_checkbox'):
            consider_translated_column = _val(self.gui.consider_translated_compression_checkbox, False)
        else:
            consider_translated_column = self.gui.config.get('compress_glossary_consider_translated_column', False)
        env_vars['COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN'] = "1" if consider_translated_column else "0"
        if hasattr(self.gui, 'compress_glossary_multipass_exclude_matching_var'):
            multipass_exclude_matching = _val(self.gui.compress_glossary_multipass_exclude_matching_var, True)
        elif hasattr(self.gui, 'multipass_exclude_matching_checkbox'):
            multipass_exclude_matching = _val(self.gui.multipass_exclude_matching_checkbox, True)
        else:
            multipass_exclude_matching = self.gui.config.get('compress_glossary_multipass_exclude_matching', True)
        env_vars['COMPRESS_GLOSSARY_MULTIPASS_EXCLUDE_MATCHING'] = "1" if multipass_exclude_matching else "0"
        # Unified glossary (cross-novel glossary_unified.csv); the GUI helper
        # reads the live checkboxes with config as the fallback.
        if hasattr(self.gui, '_strict_matching_env_dict'):
            env_vars.update(self.gui._strict_matching_env_dict())
        if hasattr(self.gui, '_unified_glossary_env_dict'):
            env_vars.update(self.gui._unified_glossary_env_dict())
        else:
            env_vars['ENABLE_UNIFIED_GLOSSARY'] = "1" if self.gui.config.get('enable_unified_glossary', False) else "0"
            env_vars['GENERATE_UNIFIED_GLOSSARY'] = "1" if self.gui.config.get('generate_unified_glossary', False) else "0"
            env_vars['UNIFIED_GLOSSARY_SOURCE_LANGUAGE'] = str(self.gui.config.get('unified_glossary_source_language', 'auto') or 'auto')
            env_vars['UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES'] = "1" if self.gui.config.get('unified_glossary_combine_all_languages', False) else "0"
            env_vars['UNIFIED_GLOSSARY_EXCLUDE_GENDER_ENTRIES'] = "1" if self.gui.config.get('unified_glossary_exclude_gender_entries', True) else "0"

        # History and summary settings
        env_vars['TRANSLATION_HISTORY_ROLLING'] = "1"
        env_vars['USE_ROLLING_SUMMARY'] = "1" if self.gui.config.get('use_rolling_summary') else "0"
        env_vars['SUMMARY_ROLE'] = self.gui.config.get('summary_role', 'system')
        env_vars['ROLLING_SUMMARY_EXCHANGES'] = _val(self.gui.rolling_summary_exchanges_var, 0)
        env_vars['ROLLING_SUMMARY_MODE'] = _val(self.gui.rolling_summary_mode_var, '')
        env_vars['ROLLING_SUMMARY_SYSTEM_PROMPT'] = self.gui.rolling_summary_system_prompt if hasattr(self.gui, 'rolling_summary_system_prompt') else ''
        env_vars['ROLLING_SUMMARY_USER_PROMPT'] = self.gui.rolling_summary_user_prompt if hasattr(self.gui, 'rolling_summary_user_prompt') else ''
        env_vars['ROLLING_SUMMARY_MAX_ENTRIES'] = _val(self.gui.rolling_summary_max_entries_var, '10') if hasattr(self.gui, 'rolling_summary_max_entries_var') else '10'
        env_vars['ROLLING_SUMMARY_MAX_TOKENS'] = _val(self.gui.rolling_summary_max_tokens_var, '-1') if hasattr(self.gui, 'rolling_summary_max_tokens_var') else '-1'
        
        # Retry and error handling settings
        env_vars['EMERGENCY_PARAGRAPH_RESTORE'] = "1" if _val(self.gui.emergency_restore_var, False) else "0"
        env_vars['RETRY_TRUNCATED'] = "1" if _val(self.gui.retry_truncated_var, False) else "0"
        try:
            _raw_retry_tokens = self.gui.max_retry_tokens_var.get()
            _resolved_retry_tokens = int(_raw_retry_tokens)
        except Exception:
            _resolved_retry_tokens = int(getattr(self.gui, 'max_output_tokens', 65536))
        try:
            if _resolved_retry_tokens <= 0:
                _resolved_retry_tokens = int(getattr(self.gui, 'max_output_tokens', 65536))
        except Exception:
            _resolved_retry_tokens = int(getattr(self.gui, 'max_output_tokens', 65536))
        _resolved_retry_tokens = _clamp_output_tokens_for_selected_model(
            env_vars.get('MODEL', ''),
            _resolved_retry_tokens,
            default=65536,
        )
        env_vars['MAX_RETRY_TOKENS'] = str(_resolved_retry_tokens)

        # Truncation and silent-truncation retries
        env_vars['TRUNCATION_RETRY_ATTEMPTS'] = str(_val(getattr(self.gui, 'truncation_retry_attempts_var', '3'), '3'))
        env_vars['USE_TRUNCATION_RETRY_KEYS'] = "1" if _val(getattr(self.gui, 'use_truncation_retry_keys_var', False), False) else "0"
        try:
            env_vars['TRUNCATION_RETRY_API_KEYS'] = json.dumps(getattr(self.gui, 'config', {}).get('truncation_retry_keys', []))
        except Exception:
            env_vars['TRUNCATION_RETRY_API_KEYS'] = "[]"
        env_vars['USE_ROLLING_SUMMARY_KEYS'] = "1" if _val(getattr(self.gui, 'use_rolling_summary_keys_var', False), False) else "0"
        try:
            env_vars['ROLLING_SUMMARY_API_KEYS'] = json.dumps(getattr(self.gui, 'config', {}).get('rolling_summary_keys', []))
        except Exception:
            env_vars['ROLLING_SUMMARY_API_KEYS'] = "[]"
        env_vars['CHAR_RATIO_TRUNCATION_ENABLED'] = "1" if _val(getattr(self.gui, 'char_ratio_truncation_var', True), True) else "0"
        env_vars['CHAR_RATIO_TRUNCATION_PERCENT'] = str(_val(getattr(self.gui, 'char_ratio_truncation_percent_var', '50'), '50'))
        env_vars['CHAR_RATIO_TRUNCATION_ATTEMPTS'] = str(_val(getattr(self.gui, 'char_ratio_truncation_attempts_var', '1'), '1'))
        env_vars['CHAR_RATIO_MIN_OUTPUT_CHARS'] = str(_val(getattr(self.gui, 'char_ratio_min_output_chars_var', '100'), '100'))

        env_vars['RETRY_DUPLICATE_BODIES'] = "1" if _val(self.gui.retry_duplicate_var, False) else "0"
        env_vars['RETRY_TIMEOUT'] = "1" if _val(self.gui.retry_timeout_var, False) else "0"
        env_vars['CHUNK_TIMEOUT'] = _val(self.gui.chunk_timeout_var, '')
        
        # Image processing
        env_vars['ENABLE_IMAGE_TRANSLATION'] = "1" if _val(self.gui.enable_image_translation_var, False) else "0"
        env_vars['PROCESS_WEBNOVEL_IMAGES'] = "1" if _val(self.gui.process_webnovel_images_var, False) else "0"
        env_vars['WEBNOVEL_MIN_HEIGHT'] = _val(self.gui.webnovel_min_height_var, 0)
        env_vars['MAX_IMAGES_PER_CHAPTER'] = _val(self.gui.max_images_per_chapter_var, -1)
        env_vars['IMAGE_API_DELAY'] = '1.0'
        env_vars['SAVE_IMAGE_TRANSLATIONS'] = '1'
        env_vars['IMAGE_CHUNK_HEIGHT'] = _val(self.gui.image_chunk_height_var, 0)
        env_vars['IMAGE_CHUNK_OVERLAP_PERCENT'] = _val(getattr(self.gui, 'image_chunk_overlap_var', None), 3)
        env_vars['IMAGE_CHUNK_MIN_OVERLAP_PIXELS'] = _val(getattr(self.gui, 'image_chunk_min_overlap_pixels_var', None), 80)
        env_vars['IMAGE_SMART_CHUNKING'] = "1" if _val(getattr(self.gui, 'image_smart_chunking_var', None), True) else "0"
        env_vars['VISION_OCR_BATCH_TRANSLATION'] = "1" if _bool_val(getattr(self.gui, 'vision_ocr_batch_translation_var', None), self.gui.config.get('vision_ocr_batch_translation', True)) else "0"
        env_vars['VISION_OCR_BATCH_SIZE'] = str(_val(getattr(self.gui, 'vision_ocr_batch_size_var', None), self.gui.config.get('vision_ocr_batch_size', '-1')))
        env_vars['VISION_OCR_FUZZY_CHUNK_DEDUPE'] = "1" if _val(getattr(self.gui, 'vision_ocr_fuzzy_chunk_dedupe_var', None), False) else "0"
        env_vars['HIDE_IMAGE_TRANSLATION_LABEL'] = "1" if _val(self.gui.hide_image_translation_label_var, False) else "0"
        output_mode = self.gui._get_output_mode() if hasattr(self.gui, '_get_output_mode') else _val(getattr(self.gui, 'output_mode_var', 'text'), 'text')
        env_vars['OUTPUT_MODE'] = output_mode
        env_vars['VISION_OCR_FIRST'] = "1" if output_mode == "vision" else "0"
        glossary_request_merge_count = str(self.gui.config.get('glossary_request_merge_count', 10) or 10)
        if output_mode == "vision" and auto_glossary_mode == "balanced":
            env_vars['GLOSSARY_REQUEST_MERGING_ENABLED'] = "1"
            env_vars['GLOSSARY_ENABLE_CHAPTER_SPLIT'] = "1" if self.gui.config.get('glossary_enable_chapter_split', False) else "0"
        else:
            env_vars['GLOSSARY_REQUEST_MERGING_ENABLED'] = "1" if self.gui.config.get('glossary_request_merging_enabled', False) else "0"
            env_vars['GLOSSARY_ENABLE_CHAPTER_SPLIT'] = "1" if self.gui.config.get('glossary_enable_chapter_split', False) else "0"
        env_vars['GLOSSARY_REQUEST_MERGE_COUNT'] = glossary_request_merge_count
        
        # Advanced settings

        env_vars['RESET_FAILED_CHAPTERS'] = "1" if _val(getattr(self.gui, 'reset_failed_chapters_var', None), False) else "0"
        env_vars['DUPLICATE_LOOKBACK_CHAPTERS'] = _val(self.gui.duplicate_lookback_var, 0)
        env_vars['DUPLICATE_DETECTION_MODE'] = _val(self.gui.duplicate_detection_mode_var, '')
        env_vars['CHAPTER_NUMBER_OFFSET'] = str(_val(self.gui.chapter_number_offset_var, 0))
        env_vars['COMPRESSION_FACTOR'] = _val(self.gui.compression_factor_var, 0)
        extraction_mode = _val(self.gui.extraction_mode_var, 'smart') if hasattr(self.gui, 'extraction_mode_var') else 'smart'
        text_extraction_method = _val(getattr(self.gui, 'text_extraction_method_var', None), 'standard') if hasattr(self.gui, 'text_extraction_method_var') else ('enhanced' if extraction_mode == 'enhanced' else 'standard')
        enhanced_filtering = _val(getattr(self.gui, 'enhanced_filtering_var', None), 'smart')
        if output_mode == "vision":
            extraction_mode = "enhanced"
            text_extraction_method = "enhanced"
            enhanced_filtering = _val(getattr(self.gui, 'file_filtering_level_var', None), enhanced_filtering)
        env_vars['COMPREHENSIVE_EXTRACTION'] = "1" if extraction_mode in ['comprehensive', 'full'] else "0"
        env_vars['EXTRACTION_MODE'] = extraction_mode
        env_vars['TEXT_EXTRACTION_METHOD'] = text_extraction_method
        env_vars['ENHANCED_FILTERING'] = enhanced_filtering
        env_vars['USE_HTML2TEXT'] = "1" if output_mode == "vision" or text_extraction_method in ('enhanced', 'html2text', 'markdown') or extraction_mode == 'enhanced' else "0"
        env_vars['CONVERT_BR_TO_PARAGRAPHS'] = (
            "1"
            if _bool_val(
                getattr(self.gui, 'convert_br_to_paragraphs_var', None),
                self.gui.config.get('convert_br_to_paragraphs', True),
            )
            else "0"
        )
        env_vars['PRESERVE_ASTERISK_SEPARATOR_LINES'] = (
            "1"
            if _bool_val(
                getattr(self.gui, 'preserve_asterisk_separator_lines_var', None),
                self.gui.config.get('preserve_asterisk_separator_lines', True),
            )
            else "0"
        )
        env_vars['FIX_STRAY_P_GT_EPUB'] = "1" if _bool_val(getattr(self.gui, 'fix_stray_p_gt_epub_var', None), self.gui.config.get('fix_stray_p_gt_epub', False)) else "0"
        env_vars['FIX_STRAY_P_GT_BS'] = "1" if _bool_val(getattr(self.gui, 'fix_stray_p_gt_bs_var', None), self.gui.config.get('fix_stray_p_gt_bs', False)) else "0"
        env_vars['DISABLE_ZERO_DETECTION'] = "1" if _val(self.gui.disable_zero_detection_var, False) else "0"
        env_vars['USE_HEADER_AS_OUTPUT'] = "0"
        env_vars['ENABLE_DECIMAL_CHAPTERS'] = "1" if _val(self.gui.enable_decimal_chapters_var, False) else "0"
        env_vars['ENABLE_WATERMARK_REMOVAL'] = "1" if _val(self.gui.enable_watermark_removal_var, False) else "0"
        env_vars['ADVANCED_WATERMARK_REMOVAL'] = "1" if _val(self.gui.advanced_watermark_removal_var, False) else "0"
        env_vars['SAVE_CLEANED_IMAGES'] = "1" if _val(self.gui.save_cleaned_images_var, False) else "0"
        
        # EPUB specific settings
        env_vars['DISABLE_EPUB_GALLERY'] = "1" if _val(self.gui.disable_epub_gallery_var, False) else "0"
        env_vars['SKIP_NON_SPINE_SPECIAL_FILES'] = "1" if _val(
            getattr(self.gui, 'skip_non_spine_special_files_var', False),
            False,
        ) else "0"
        env_vars['SKIP_UNREFERENCED_EPUB_IMAGES'] = "1" if _val(
            getattr(self.gui, 'skip_unreferenced_epub_images_var', False),
            False,
        ) else "0"
        env_vars['FORCE_NCX_ONLY'] = '1' if _val(self.gui.force_ncx_only_var, False) else '0'
        
        # Special handling for Gemini safety filters
        env_vars['DISABLE_GEMINI_SAFETY'] = str(self.gui.config.get('disable_gemini_safety', False)).lower()
        
        # AI Hunter settings (if enabled)
        if 'ai_hunter_config' in self.gui.config:
            env_vars['AI_HUNTER_CONFIG'] = json.dumps(self.gui.config['ai_hunter_config'])
        
        # Output settings
        env_vars['EPUB_OUTPUT_DIR'] = os.getcwd()
        try:
            from glossary_paths import resolve_shared_glossary_dir
            env_vars['GLOSSARY_SHARED_DIR'] = resolve_shared_glossary_dir(
                os.environ.get('GLOSSARY_SHARED_DIR', ''),
                fallback_base=getattr(self.gui, 'file_path', ''),
            )
        except Exception:
            env_vars['GLOSSARY_SHARED_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Glossary')
        output_path = self.gui.output_entry.get().strip() if hasattr(self.gui, 'output_entry') else ''
        if not output_path:
            try:
                output_path = str(self.gui.config.get('output_directory', '') or '').strip()
            except Exception:
                output_path = ''
        if output_path:
            output_path = os.path.abspath(output_path)
            env_vars['OUTPUT_DIR'] = output_path
            env_vars['OUTPUT_DIRECTORY'] = output_path
            env_vars['GLOSSARY_SHARED_DIR'] = os.path.join(output_path, 'Glossary')
        
        # File path (needed by some modules)
        env_vars['EPUB_PATH'] = self.gui.file_path
        
        return env_vars

    def _safe_int(self, value, default: int, allow_zero: bool = False) -> int:
        """Safely parse int, falling back to default on errors/blank.
        If allow_zero is True, 0 is treated as valid."""
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                iv = int(value)
            else:
                txt = str(value).strip().replace(',', '')
                iv = int(txt) if txt else default
            if iv > 0:
                return iv
            if allow_zero and iv == 0:
                return 0
            return default
        except Exception:
            return default

    def _extract_chapters_for_async(self, file_path, env_vars):
        """Extract chapters and prepare them for async processing"""
        chapters = []
        original_basename = None
        chapter_mapping = {}  # Map custom_id to chapter info
        markdown_provenance_by_file = {}
        html2text_blocks_by_file = {}
        html2text_block_hash_by_file = {}

        # Respect GUI extraction mode/radio
        extraction_method = env_vars.get('TEXT_EXTRACTION_METHOD', env_vars.get('EXTRACTION_MODE', 'standard')).lower()
        enhanced_filtering = env_vars.get('ENHANCED_FILTERING', 'smart')
        preserve_structure = env_vars.get('ENHANCED_PRESERVE_STRUCTURE', True)
        use_html2text = extraction_method in ['enhanced', 'html2text', 'markdown']
        extractor = None
        if use_html2text:
            try:
                from enhanced_text_extractor import EnhancedTextExtractor
                extractor = EnhancedTextExtractor(filtering_mode=enhanced_filtering, preserve_structure=preserve_structure)
            except Exception as e:
                print(f"⚠️ Falling back to BeautifulSoup extraction; failed to init EnhancedTextExtractor: {e}")
                use_html2text = False
        
        try:
            if file_path.lower().endswith('.epub'):
                # Use direct ZIP reading to avoid ebooklib's manifest validation
                import zipfile
                from bs4 import BeautifulSoup
                opf_spine_map = self._get_opf_spine_map(file_path)
                
                raw_chapters = []
                
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        # Get all HTML/XHTML files
                        html_files = [f for f in zf.namelist() if f.endswith(('.html', '.xhtml', '.htm')) and not f.startswith('__MACOSX')]
                        # Order by OPF spine if available, otherwise fallback to name sort
                        if opf_spine_map:
                            html_files.sort(key=lambda f: opf_spine_map.get(f) or opf_spine_map.get(os.path.basename(f)) or opf_spine_map.get(os.path.splitext(os.path.basename(f))[0]) or float('inf'))
                        else:
                            html_files.sort()
                        
                        for idx, html_file in enumerate(html_files):
                            try:
                                content = zf.read(html_file)
                                soup = BeautifulSoup(content, 'html.parser')

                                # Keep full HTML (including images and links) for translation unless user chose html2text
                                chapter_html = str(soup)
                                chapter_text = soup.get_text(separator='\n').strip()

                                spine_pos = None
                                if opf_spine_map:
                                    spine_pos = (
                                        opf_spine_map.get(html_file)
                                        or opf_spine_map.get(os.path.basename(html_file))
                                        or opf_spine_map.get(os.path.splitext(os.path.basename(html_file))[0])
                                    )
                                chapter_num = (spine_pos + 1) if spine_pos is not None else (idx + 1)

                                # Try to extract chapter number from content
                                for element in soup.find_all(['h1', 'h2', 'h3', 'title']):
                                    text = element.get_text().strip()
                                    match = re.search(r'chapter\\s*(\\d+)', text, re.IGNORECASE)
                                    if match:
                                        chapter_num = int(match.group(1))
                                        break

                                # Apply extraction mode
                                if use_html2text and extractor:
                                    try:
                                        cleaned_text, _, _ = extractor.extract_chapter_content(chapter_html, extraction_mode=extraction_method)
                                        chapter_payload = cleaned_text
                                        markdown_provenance_by_file[html_file] = getattr(extractor, "last_markdown_provenance", {})
                                        html2text_blocks_by_file[html_file] = getattr(extractor, "last_html2text_blocks", []) or []
                                        html2text_block_hash_by_file[html_file] = hashlib.sha256(
                                            chapter_payload.encode('utf-8', errors='ignore')
                                        ).hexdigest()
                                    except Exception as e:
                                        print(f"⚠️ html2text extraction failed, using HTML: {e}")
                                        chapter_payload = chapter_html
                                else:
                                    chapter_payload = chapter_html
                                raw_chapters.append((chapter_num, chapter_payload, html_file, spine_pos))
                                    
                            except Exception as e:
                                print(f"Error reading {html_file}: {e}")
                                continue
                                
                except Exception as e:
                    print(f"Failed to read EPUB as ZIP: {e}")
                    raise ValueError(f"Cannot read EPUB file: {str(e)}")
                    
            elif file_path.lower().endswith('.txt'):
                # Import TXT processing
                from txt_processor import TextFileProcessor
                
                processor = TextFileProcessor(file_path, '')
                txt_chapters = processor.extract_chapters()
                raw_chapters = [(i+1, text, f"section_{i+1:04d}.txt") for i, text in enumerate(txt_chapters)]
                
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            if not raw_chapters:
                raise ValueError("No valid chapters found in file")
            # Reorder chapters using OPF spine if available
            if file_path.lower().endswith('.epub') and 'opf_spine_map' in locals() and opf_spine_map:
                raw_chapters.sort(
                    key=lambda ch: (
                        ch[3] if ch[3] is not None else float("inf"),
                        ch[0]
                    )
                )
            else:
                raw_chapters.sort(key=lambda ch: ch[0])
                
            # Process each chapter to prepare for API
            # Initialize splitter once
            splitter = None

            for idx, (chapter_num, content, original_filename, spine_pos) in enumerate(raw_chapters):
                # Count tokens (content only)
                token_count = self.count_tokens(content, env_vars['MODEL'])
                ordered_num = (spine_pos + 1) if spine_pos is not None else chapter_num

                # Determine output token limit (use MAX_OUTPUT_TOKENS; allow zero = unlimited)
                output_limit = self._safe_int(env_vars.get('MAX_OUTPUT_TOKENS', '65536'), 65536, allow_zero=True)

                # Estimate overhead (system prompt + optional glossary if appended)
                overhead_tokens = 0
                try:
                    overhead_tokens += self.count_tokens(env_vars.get('SYSTEM_PROMPT', ''), env_vars['MODEL'])
                except Exception:
                    pass
                if env_vars.get('MANUAL_GLOSSARY') and env_vars.get('APPEND_GLOSSARY') == '1':
                    try:
                        with open(env_vars['MANUAL_GLOSSARY'], 'r', encoding='utf-8') as f:
                            glossary_preview = f.read(40000)  # cap read for speed
                        overhead_tokens += self.count_tokens(glossary_preview, env_vars['MODEL'])
                    except Exception:
                        pass

                if output_limit == 0:
                    # Unlimited: never mark for chunking
                    effective_limit = float('inf')
                    threshold = float('inf')
                    needs_chunking = False
                else:
                    # Allow for request envelope (~2k) and overhead
                    effective_limit = max(0, output_limit - overhead_tokens - 2000)
                    threshold = max(effective_limit, int(output_limit * 0.8))
                    needs_chunking = token_count > threshold

                # Always log the decision so users can see why a chapter is (or isn't) skipped
                safe_name = os.path.basename(original_filename) if original_filename else "<unknown>"
                self._log(
                    f"[ASYNC] Chapter {ordered_num} ({safe_name}): content_tokens={token_count}, "
                    f"overhead≈{overhead_tokens}, token_limit={output_limit}, "
                    f"threshold={threshold}, needs_chunking={needs_chunking}")

                # If chunking needed, split instead of skipping
                if needs_chunking and output_limit != float('inf'):
                    try:
                        try:
                            compression_factor = float(env_vars.get('COMPRESSION_FACTOR', 1.0) or 1.0)
                        except Exception:
                            compression_factor = 1.0
                        if compression_factor <= 0:
                            compression_factor = 0.000000000001
                        if splitter is None:
                            chunk_target = max(1024, int(output_limit / compression_factor))
                            splitter = ChapterSplitter(model_name=env_vars.get('MODEL', 'gpt-4'), target_tokens=chunk_target, compression_factor=compression_factor)
                        chunk_target = max(1024, int(output_limit / compression_factor))
                        block_chunks = []
                        blocks = html2text_blocks_by_file.get(original_filename)
                        block_hash = html2text_block_hash_by_file.get(original_filename)
                        try:
                            content_hash_for_blocks = hashlib.sha256(
                                str(content or "").encode('utf-8', errors='ignore')
                            ).hexdigest()
                        except Exception:
                            content_hash_for_blocks = None
                        if isinstance(blocks, list) and blocks and block_hash and content_hash_for_blocks == block_hash:
                            block_chunks = splitter.split_blocks(blocks, max_tokens=chunk_target)
                        chunk_list = block_chunks or splitter.split_chapter(content, max_tokens=chunk_target, filename=original_filename)
                        total_chunks = len(chunk_list)
                        base_slug = Path(original_filename).stem if original_filename else f"ch{ordered_num}"
                        base_provenance = markdown_provenance_by_file.get(original_filename, {})
                        base_atx_headings = base_provenance.get('atx_headings', []) if isinstance(base_provenance, dict) else []
                        chunk_heading_offset = 0
                        for ci, (chunk_html, chunk_idx, total) in enumerate(chunk_list, start=1):
                            chunk_heading_count = len(re.findall(r'(?m)^\s{0,3}#{1,6}(?:\s+|$)', chunk_html))
                            chunk_provenance = {
                                'version': base_provenance.get('version', 1) if isinstance(base_provenance, dict) else 1,
                                'atx_headings': base_atx_headings[chunk_heading_offset:chunk_heading_offset + chunk_heading_count],
                            } if base_atx_headings else {}
                            chunk_heading_offset += chunk_heading_count
                            part_custom_id = f"{ordered_num:04d}_{base_slug}_part{chunk_idx}"
                            messages = self._prepare_chapter_messages(chunk_html, env_vars)
                            chapter_data = {
                                'id': part_custom_id,
                                'number': ordered_num + ci * 0.001,  # slight offset to preserve order
                                'detected_number': chapter_num,
                                'content': chunk_html,
                                'messages': messages,
                                'temperature': float(env_vars.get('TRANSLATION_TEMPERATURE', '0.3')),
                                'max_tokens': int(env_vars['MAX_OUTPUT_TOKENS']),
                                'needs_chunking': False,
                                'token_count': self.count_tokens(chunk_html, env_vars['MODEL']),
                                'original_basename': original_filename,
                                'original_filename': original_filename,
                                'extraction_method': extraction_method,
                                'opf_spine_position': spine_pos,
                                'chunk_index': chunk_idx,
                                'chunk_total': total
                            }
                            chapters.append(chapter_data)
                            chapter_mapping[part_custom_id] = {
                                'original_filename': original_filename,
                                'chapter_num': ordered_num,
                                'extraction_method': extraction_method,
                                'preserve_structure': preserve_structure,
                                'markdown_provenance': chunk_provenance,
                                'opf_spine_position': spine_pos,
                                'detected_chapter_num': chapter_num,
                                'chunk_index': chunk_idx,
                                'chunk_total': total
                            }
                        continue  # handled splitting; skip default add
                    except Exception as split_err:
                        self._log(f"[ASYNC] Chunk splitting failed for chapter {ordered_num}: {split_err}", level="warning")
                        # fall through to add original as-is (will be marked needs_chunking)

                # Prepare messages format
                messages = self._prepare_chapter_messages(content, env_vars)
                # Use ordered number (spine-aware) for the custom id to keep IDs unique and aligned with spine order
                base_slug = Path(original_filename).stem if original_filename else f"ch{ordered_num}"
                custom_id = f"{ordered_num:04d}_{base_slug}"

                chapter_data = {
                    'id': custom_id,
                    'number': ordered_num,
                    'detected_number': chapter_num,
                    'content': content,
                    'messages': messages,
                    'temperature': float(env_vars.get('TRANSLATION_TEMPERATURE', '0.3')),
                    'max_tokens': int(env_vars['MAX_OUTPUT_TOKENS']),
                    'needs_chunking': needs_chunking,
                    'token_count': token_count,
                    'original_basename': original_filename,  # Use original_filename instead of undefined original_basename
                    'original_filename': original_filename,  # preserve full original filename for saving
                    'extraction_method': extraction_method,
                    'opf_spine_position': spine_pos
                }

                chapters.append(chapter_data)

                # Store mapping
                chapter_mapping[custom_id] = {
                    'original_filename': original_filename,
                    'chapter_num': ordered_num,
                    'extraction_method': extraction_method,
                    'preserve_structure': preserve_structure,
                    'markdown_provenance': markdown_provenance_by_file.get(original_filename, {}),
                    'opf_spine_position': spine_pos,
                    'detected_chapter_num': chapter_num
                }
                
        except Exception as e:
            print(f"Failed to extract chapters: {e}")
            raise
            
        # Return both chapters and mapping
        return chapters, chapter_mapping

    def _delete_selected_job(self):
        """Delete selected job from the list"""
        job_ids = self._get_selected_job_ids()
        if not job_ids:
            self._async_msgbox('warning', "No Selection", "Please select one or more jobs to delete")
            return

        reply = self._async_msgbox('question', "Confirm Delete",
            f"Delete {len(job_ids)} selected job(s) from the local list?\n\n"
            "Note: This does not stop running jobs on the provider.",
            self._MB_YES | self._MB_NO
        )

        if reply != self._MB_YES:
            return

        for jid in job_ids:
            if jid in self.processor.jobs:
                del self.processor.jobs[jid]

        self.processor._save_jobs()
        self.selected_job_id = None
        self._refresh_jobs_list()
        self._async_msgbox('information', "Job Deleted", f"Removed {len(job_ids)} job(s) from the local list.")

    def _clear_completed_jobs(self):
        """Clear all completed/failed/cancelled jobs"""
        # Get list of jobs to remove
        jobs_to_remove = []
        for job_id, job in self.processor.jobs.items():
            if job.status in [AsyncAPIStatus.COMPLETED, AsyncAPIStatus.FAILED, 
                             AsyncAPIStatus.CANCELLED, AsyncAPIStatus.EXPIRED]:
                jobs_to_remove.append(job_id)
        
        if not jobs_to_remove:
            self._async_msgbox('information', "No Jobs to Clear", "No completed/failed/cancelled jobs to clear.")
            return
        
        # Confirm
        reply = self._async_msgbox('question', "Clear Completed Jobs",
            f"Remove {len(jobs_to_remove)} completed/failed/cancelled jobs from the list?\n\n"
            "This will not affect any running jobs.",
            self._MB_YES | self._MB_NO
        )
        
        if reply == self._MB_YES:
            # Remove jobs
            for job_id in jobs_to_remove:
                del self.processor.jobs[job_id]
            
            # Save
            self.processor._save_jobs()
            
            # Refresh
            self._refresh_jobs_list()
            
            self._async_msgbox('information', "Jobs Cleared", f"Removed {len(jobs_to_remove)} jobs from the list.")

    def _prepare_chapter_messages(self, content, env_vars):
        """Prepare messages array for a chapter"""
        messages = []
        
        # System prompt
        system_prompt = env_vars.get('SYSTEM_PROMPT', '')
        
        # DEBUG: Log what we're sending
        logger.info(f"Model: {env_vars.get('MODEL')}")
        logger.info(f"System prompt length: {len(system_prompt)}")
        logger.info(f"Content length: {len(content)}")
        
        # Log the system prompt (first 200 chars)
        logger.info(f"Using system prompt: {system_prompt[:200]}...")
        
        # Add glossary if enabled
        if (env_vars.get('MANUAL_GLOSSARY') and 
            env_vars.get('APPEND_GLOSSARY') == '1' and 
            env_vars.get('DISABLE_GLOSSARY_TRANSLATION') != '1'):
            try:
                glossary_path = env_vars['MANUAL_GLOSSARY']
                with open(glossary_path, 'r', encoding='utf-8') as f:
                    glossary_data = json.load(f)
                
                # TRUE BRUTE FORCE: Just dump the entire JSON
                glossary_text = json.dumps(glossary_data, ensure_ascii=False, indent=2)
                original_glossary_text = glossary_text  # Store for compression stats
                
                # Apply glossary compression if enabled
                compress_glossary_enabled = env_vars.get('COMPRESS_GLOSSARY_PROMPT') == '1'
                glossary_compression_logged = False
                if compress_glossary_enabled and content:
                    try:
                        from glossary_compressor import compress_glossary
                        original_length = len(glossary_text)
                        glossary_text = compress_glossary(
                            glossary_text,
                            content,
                            glossary_format='auto',
                            glossary_path=glossary_path,
                        )
                        compressed_length = len(glossary_text)
                        reduction_pct = ((original_length - compressed_length) / original_length * 100) if original_length > 0 else 0
                        glossary_compression_logged = True
                        
                        # Calculate token savings if tiktoken is available
                        try:
                            import tiktoken
                            try:
                                enc = tiktoken.encoding_for_model(env_vars.get('MODEL', 'gpt-4'))
                            except:
                                enc = tiktoken.get_encoding('cl100k_base')
                            
                            original_tokens = len(enc.encode(original_glossary_text))
                            compressed_tokens = len(enc.encode(glossary_text))
                            token_reduction_pct = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0
                            whole_term_scope = env_vars.get('COMPRESS_GLOSSARY_STRICT_MATCHING_MODE', 'all')
                            translated_state = "ON" if env_vars.get('COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN') == '1' else "OFF"
                            
                            logger.info(f"🗜️ Glossary: {original_tokens}→{compressed_tokens} tokens ({token_reduction_pct:.1f}%) (whole term: {whole_term_scope}, translated column {translated_state})")
                        except ImportError:
                            logger.info(f"🗜️ Glossary compressed: {original_length} → {compressed_length} chars ({reduction_pct:.1f}% reduction)")
                    except Exception as e:
                        logger.warning(f"⚠️ Glossary compression failed: {e}")
                
                # Use the append prompt format if provided
                append_prompt = env_vars.get('APPEND_GLOSSARY_PROMPT', '')
                if glossary_text and glossary_text.strip():
                    if append_prompt:
                        # Replace placeholder with actual glossary
                        if '{glossary}' in append_prompt:
                            glossary_section = append_prompt.replace('{glossary}', glossary_text)
                        else:
                            glossary_section = f"{append_prompt}\n{glossary_text}"
                        system_prompt = f"{system_prompt}\n\n{glossary_section}"
                    else:
                        # Default format
                        system_prompt = f"{system_prompt}\n\nGlossary:\n{glossary_text}"
                    
                    logger.info(f"✅ Glossary appended ({len(glossary_text)} characters)")
                else:
                    logger.info("ℹ️ Glossary skipped for this chapter (no matching entries after compression)")
                
                # Log preview for debugging
                if glossary_text and len(glossary_text) > 200:
                    logger.info(f"Glossary preview: {glossary_text[:200]}...")
                elif glossary_text:
                    logger.info(f"Glossary: {glossary_text}")
                        
            except FileNotFoundError:
                print(f"Glossary file not found: {env_vars.get('MANUAL_GLOSSARY')}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in glossary file")
            except Exception as e:
                print(f"Failed to load glossary: {e}")
        else:
            # Log why glossary wasn't added
            if not env_vars.get('MANUAL_GLOSSARY'):
                logger.info("No glossary path specified")
            elif env_vars.get('APPEND_GLOSSARY') != '1':
                logger.info("Glossary append is disabled")
            elif env_vars.get('DISABLE_GLOSSARY_TRANSLATION') == '1':
                logger.info("Glossary translation is disabled")
        
        messages.append({
            'role': 'system',
            'content': system_prompt
        })
        
        # Add context if enabled
        if env_vars.get('CONTEXTUAL') == '1':
            # This would need to load context from history
            # For async, we might need to pre-generate context
            logger.info("Note: Contextual mode enabled but not implemented for async yet")
        
        # User message with chapter content
        messages.append({
            'role': 'user',
            'content': content
        })
        
        return messages

    def _submit_batch_sync(self, batch_data, model, api_key):
        """Submit batch synchronously (wrapper for async method)"""
        provider = self.processor.get_provider_from_model(model)
        
        if provider == 'openai':
            return self.processor._submit_openai_batch_sync(batch_data, model, api_key)
        elif provider == 'anthropic':
            return self.processor._submit_anthropic_batch_sync(batch_data, model, api_key)
        elif provider == 'gemini':
            return self._submit_gemini_batch_sync(batch_data, model, api_key)
        elif provider == 'mistral':
            return self._submit_mistral_batch_sync(batch_data, model, api_key)
        elif provider == 'groq':
            return self._submit_groq_batch_sync(batch_data, model, api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _submit_gemini_batch_sync(self, batch_data, model, api_key):
        """Submit Gemini batch using the official Batch Mode API"""
        try:
            # Use the new Google Gen AI SDK
            from google import genai
            from google.genai import types
            
            # Configure client with API key
            client = genai.Client(api_key=api_key)
            
            # Log for debugging
            logger.info(f"Submitting Gemini batch with model: {model}")
            logger.info(f"Number of requests: {len(batch_data['requests'])}")
            
            # Create JSONL file for batch requests
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
                for request in batch_data['requests']:
                    # Format for Gemini batch API
                    gen_cfg = request['generateContentRequest'].get('generationConfig', {}).copy()
                    # Google Batch API rejects unknown fields; remove 'thinking' (only realtime API supports it)
                    gen_cfg.pop('thinking', None)

                    batch_line = {
                        "key": request['custom_id'],
                        "request": {
                            "contents": request['generateContentRequest']['contents'],
                            "generation_config": gen_cfg
                        }
                    }
                    # Add safety settings if present
                    if 'safetySettings' in request['generateContentRequest']:
                        batch_line['request']['safety_settings'] = request['generateContentRequest']['safetySettings']
                        
                    f.write(json.dumps(batch_line) + '\n')
                
                batch_file_path = f.name
            
            # Upload the batch file with explicit mime type
            logger.info("Uploading batch file...")
            
            # Use the upload config to specify mime type
            upload_config = types.UploadFileConfig(
                mime_type='application/jsonl',  # Explicit JSONL mime type
                display_name=f"batch_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )

            uploaded_file = client.files.upload(
                file=batch_file_path,
                config=upload_config
            )
            
            logger.info(f"File uploaded: {uploaded_file.name}")
            
            # Create batch job
            batch_job = client.batches.create(
                model=model,
                src=uploaded_file.name,
                config={
                    'display_name': f"glossarion_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
            
            logger.info(f"Gemini batch job created: {batch_job.name}")
            
            # Clean up temp file
            os.unlink(batch_file_path)
            
            # Calculate cost estimate
            total_tokens = sum(r.get('token_count', 15000) for r in batch_data['requests'])
            async_cost, _ = self.processor.estimate_cost(
                len(batch_data['requests']), 
                total_tokens // len(batch_data['requests']), 
                model
            )
            
            # Create job info
            job = AsyncJobInfo(
                job_id=batch_job.name,
                provider='gemini',
                model=model,
                status=AsyncAPIStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                total_requests=len(batch_data['requests']),
                cost_estimate=0.0,  # No estimate initially
                metadata={
                    'batch_info': {
                        'name': batch_job.name,
                        'state': batch_job.state.name if hasattr(batch_job, 'state') else 'PENDING',
                        'src_file': uploaded_file.name
                    },
                    'source_file': self.gui.file_path  # Add this to track which file this job is for
                }
            )
            
            return job
            
        except ImportError:
            print("Google Gen AI SDK not installed. Run: pip install google-genai")
            raise Exception("Google Gen AI SDK not installed. Please run: pip install google-genai")
        except Exception as e:
            print(f"Gemini batch submission failed: {e}")
            print(f"Full error: {traceback.format_exc()}")
            raise

    def _submit_mistral_batch_sync(self, batch_data, model, api_key):
        """Submit Mistral batch (synchronous version)"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                'https://api.mistral.ai/v1/batch/jobs',
                headers=headers,
                json=batch_data
            )
            
            if response.status_code != 200:
                raise Exception(f"Batch creation failed: {response.text}")
                
            batch_info = response.json()
            
            # Calculate cost estimate
            total_tokens = sum(r.get('token_count', 15000) for r in batch_data['requests'])
            async_cost, _ = self.processor.estimate_cost(
                len(batch_data['requests']), 
                total_tokens // len(batch_data['requests']), 
                model
            )
            
            job = AsyncJobInfo(
                job_id=batch_info['id'],
                provider='mistral',
                model=model,
                status=AsyncAPIStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                total_requests=len(batch_data['requests']),
                cost_estimate=async_cost,
                metadata={'batch_info': batch_info}
            )
            
            return job
            
        except Exception as e:
            print(f"Mistral batch submission failed: {e}")
            raise

    def _submit_groq_batch_sync(self, batch_data, model, api_key):
        """Submit Groq batch (synchronous version)"""
        # Groq uses OpenAI-compatible format
        return self.processor._submit_openai_batch_sync(batch_data, model, api_key)

    def _start_polling(self, job_id):
        """Start polling for job completion with progress updates"""
        def poll():
            try:
                job = self.processor.check_job_status(job_id)
                self._refresh_jobs_list()
                
                # Update progress message
                if job.total_requests > 0:
                    progress_pct = int((job.completed_requests / job.total_requests) * 100)
                    self._log(f"Progress: {progress_pct}% ({job.completed_requests}/{job.total_requests} chapters)")
                
                if job.status == AsyncAPIStatus.COMPLETED:
                    self._log(f"✅ Job {job_id} completed!")
                    self._handle_completed_job(job_id)
                elif job.status in [AsyncAPIStatus.FAILED, AsyncAPIStatus.CANCELLED]:
                    self._log(f"❌ Job {job_id} {job.status.value}")
                else:
                    # Continue polling with progress update
                    poll_interval = self._async_poll_interval() * 1000
                    self._async_single_shot(poll_interval, poll)
                    
            except Exception as e:
                self._log(f"❌ Polling error: {e}")
                
        # Start polling
        poll()

    def _handle_completed_job(self, job_id):
        """Handle a completed job - retrieve results and save"""
        try:
            job = self.processor.jobs.get(job_id)
            # Retrieve results
            results = self.processor.retrieve_results(job_id)
            
            if not results:
                self._log("❌ No results retrieved from completed job")
                return
                
            # Determine source file strictly from job metadata to avoid using current GUI selection
            source_path = ""
            if job and job.metadata:
                source_path = job.metadata.get('source_file') or ""
                # Fallback to stored env path if source_file wasn't persisted
                if not source_path and isinstance(job.metadata.get('env'), dict):
                    source_path = job.metadata['env'].get('EPUB_PATH') or job.metadata['env'].get('SOURCE_FILE') or ""
            if not source_path:
                raise ValueError("Source file path missing from job metadata. Please resubmit the job.")
            if not os.path.isfile(source_path):
                raise ValueError(f"Source file not found: {source_path}")
            self._log(f"Using source file for results: {source_path}")

            # Get output directory
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            
            # Check for override
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.gui.config.get('output_directory')
            if override_dir:
                output_dir = os.path.join(override_dir, base_name)
            else:
                # Default: same name as source file, in exe location
                if getattr(sys, 'frozen', False):
                    # Running as compiled exe - use exe directory
                    app_dir = os.path.dirname(sys.executable)
                else:
                    # Running as script - use script directory
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(app_dir, base_name)
            
            # Handle existing directory
            if os.path.exists(output_dir):
                reply = self._async_msgbox('question', "Directory Exists",
                    f"The output directory already exists:\n{output_dir}\n\n"
                    "Overwrite = Yes\n"
                    "Create new = No\n"
                    "Cancel = Cancel",
                    self._MB_YES | self._MB_NO | self._MB_CANCEL
                )
                
                if reply == self._MB_CANCEL:
                    return
                elif reply == self._MB_NO:
                    counter = 1
                    while os.path.exists(f"{output_dir}_{counter}"):
                        counter += 1
                    output_dir = f"{output_dir}_{counter}"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract ALL resources from EPUB (CSS, fonts, images)
            self._log("📦 Extracting EPUB resources...")
            import zipfile
            
            with zipfile.ZipFile(source_path, 'r') as zf:
                # Create resource directories
                for res_type in ['css', 'fonts', 'images']:
                    os.makedirs(os.path.join(output_dir, res_type), exist_ok=True)
                
                # Extract all resources, flatten images into images/
                for file_path in zf.namelist():
                    if file_path.endswith('/'):
                        continue
                        
                    file_lower = file_path.lower()
                    file_name = os.path.basename(file_path)
                    
                    # Skip empty filenames
                    if not file_name:
                        continue
                    
                    if file_lower.endswith('.css'):
                        zf.extract(file_path, os.path.join(output_dir, 'css'))
                    elif file_lower.endswith(('.ttf', '.otf', '.woff', '.woff2')):
                        zf.extract(file_path, os.path.join(output_dir, 'fonts'))
                    elif file_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp')):
                        # Flatten: copy image into output_dir/images with basename only
                        dest = os.path.join(output_dir, 'images', file_name)
                        with open(dest, 'wb') as img_out:
                            img_out.write(zf.read(file_path))
            
            # Extract chapter info and metadata from source EPUB
            self._log("📋 Extracting metadata from source EPUB...")
            
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            from TransateKRtoEN import get_content_hash, should_retain_source_extension
            spine_map = self._get_opf_spine_map(source_path)
            
            # Extract metadata
            metadata = {}
            book = epub.read_epub(source_path)
            
            # Get book metadata
            if book.get_metadata('DC', 'title'):
                metadata['title'] = book.get_metadata('DC', 'title')[0][0]
            if book.get_metadata('DC', 'creator'):
                metadata['creator'] = book.get_metadata('DC', 'creator')[0][0]
            if book.get_metadata('DC', 'language'):
                metadata['language'] = book.get_metadata('DC', 'language')[0][0]
            
            # Save metadata.json
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Map chapter numbers to original info
            chapter_map = {}
            chapter_map_by_spine = {}
            chapters_info = []
            actual_chapter_num = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    original_name = item.get_name()
                    original_basename = os.path.splitext(os.path.basename(original_name))[0]
                    
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text().strip()
                    
                    # Keep even very short documents (e.g., covers/credits) to preserve filenames
                    actual_chapter_num += 1
                    
                    # Try to find chapter number from headings
                    chapter_num = actual_chapter_num
                    for element in soup.find_all(['h1', 'h2', 'h3', 'title']):
                        element_text = element.get_text().strip()
                        match = re.search(r'chapter\\s*(\\d+)', element_text, re.IGNORECASE)
                        if match:
                            chapter_num = int(match.group(1))
                            break
                    
                    # Calculate real content hash
                    content_hash = get_content_hash(text)
                    spine_pos = None
                    if spine_map:
                        spine_pos = (
                            spine_map.get(original_name)
                            or spine_map.get(original_basename)
                            or spine_map.get(os.path.splitext(original_basename)[0])
                        )
                    order_num = (spine_pos + 1) if spine_pos is not None else chapter_num
                    
                    info = {
                        'original_basename': original_basename,
                        'original_extension': os.path.splitext(original_name)[1],
                        'content_hash': content_hash,
                        'text_length': len(text),
                        'has_images': bool(soup.find_all('img')),
                        'opf_spine_position': spine_pos,
                        'detected_chapter_num': chapter_num,
                        'original_filename': original_name,
                        'title': element_text if 'element_text' in locals() else f"Chapter {chapter_num}"
                    }
                    
                    chapter_map[order_num] = info
                    if spine_pos is not None:
                        chapter_map_by_spine[spine_pos] = info
                    
                    chapters_info.append({
                        'num': order_num,
                        'title': info['title'],
                        'original_filename': original_name,
                        'original_basename': original_basename,
                        'has_images': info['has_images'],
                        'text_length': info['text_length'],
                        'content_hash': content_hash,
                        'opf_spine_position': spine_pos,
                        'detected_chapter_num': chapter_num
                    })
            
            # Save chapters_info.json
            chapters_info_path = os.path.join(output_dir, 'chapters_info.json')
            with open(chapters_info_path, 'w', encoding='utf-8') as f:
                json.dump(chapters_info, f, ensure_ascii=False, indent=2)
            
            # Create realistic progress tracking
            progress_data = {
                "version": "3.0",
                "chapters": {},
                "chapter_chunks": {},
                "content_hashes": {},
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_chapters": len(results),
                "completed_chapters": len(results),
                "failed_chapters": 0,
                "async_translated": True
            }
            
            chapter_mapping = {}
            if hasattr(self, 'processor'):
                job = self.processor.jobs.get(job_id)
                if job and job.metadata:
                    chapter_mapping = job.metadata.get('chapter_mapping', {})

            def _result_sort_key(res):
                meta = chapter_mapping.get(res.get('custom_id'), {}) if chapter_mapping else {}
                spine_pos = meta.get('opf_spine_position')
                if spine_pos is None:
                    chapter_num = self._extract_chapter_number(res.get('custom_id', ''))
                    spine_pos = chapter_map.get(chapter_num, {}).get('opf_spine_position')
                if spine_pos is None:
                    spine_pos = float('inf')
                return (spine_pos, self._extract_chapter_number(res.get('custom_id', '')))

            # Sort results and save with proper filenames (OPF spine aware)
            sorted_results = sorted(results, key=_result_sort_key)
            
            self._log("💾 Saving translated chapters...")
            for result in sorted_results:
                chapter_num = self._extract_chapter_number(result['custom_id'])
                chapter_meta = chapter_mapping.get(result['custom_id'], {}) if chapter_mapping else {}
                spine_pos = chapter_meta.get('opf_spine_position')
                if spine_pos is not None:
                    chapter_num = spine_pos + 1
                
                # Get chapter info
                chapter_info = {}
                if spine_pos is not None and spine_pos in chapter_map_by_spine:
                    chapter_info = chapter_map_by_spine.get(spine_pos, {})
                elif chapter_num in chapter_map:
                    chapter_info = chapter_map.get(chapter_num, {})
                original_basename = chapter_info.get('original_basename', f"{chapter_num:04d}")
                content_hash = chapter_info.get('content_hash', hashlib.sha256(f"chapter_{chapter_num}".encode()).hexdigest())
                
                # Save file with correct name (only once!)
                # Async: always retain original source extension to keep filenames consistent
                retain_ext = True
                # Preserve compound extensions like .htm.xhtml when retaining
                orig_name = chapter_info.get('original_filename') or chapter_info.get('original_basename')
                if retain_ext and orig_name:
                    # Compute full extension suffix and a base with ALL extensions stripped
                    full = os.path.basename(orig_name)
                    bn, ext1 = os.path.splitext(full)
                    full_ext = ''
                    while ext1:
                        full_ext = ext1 + full_ext
                        bn, ext1 = os.path.splitext(bn)
                    base_no_ext = bn if bn else os.path.splitext(full)[0]
                    # If no extension found, default to .html
                    suffix = full_ext if full_ext else '.html'
                    filename = f"{base_no_ext}{suffix}"
                elif retain_ext:
                    filename = f"{original_basename}.html"
                else:
                    filename = f"response_{original_basename}.html"
                file_path = os.path.join(output_dir, filename)
                
                # Determine extraction method for this chapter (per-chapter > env > default)
                job_obj = self.processor.jobs.get(job_id) if hasattr(self, 'processor') else None
                job_env = job_obj.metadata.get('env', {}) if job_obj and job_obj.metadata else {}
                chapter_meta = {}
                if job_obj and job_obj.metadata and job_obj.metadata.get('chapter_mapping'):
                    chapter_meta = job_obj.metadata['chapter_mapping'].get(result['custom_id'], {})
                extraction_method = str(
                    chapter_meta.get('extraction_method')
                    or job_env.get('TEXT_EXTRACTION_METHOD')
                    or job_env.get('EXTRACTION_MODE')
                    or 'standard'
                ).lower()

                # Convert plain/markdown back to HTML when html2text/enhanced was used
                content = result.get('content', '')
                if extraction_method in ['enhanced', 'html2text', 'markdown']:
                    try:
                        from TransateKRtoEN import convert_enhanced_text_to_html
                        preserve_structure = chapter_meta.get('preserve_structure', True)
                        content = convert_enhanced_text_to_html(content, {
                            'preserve_structure': preserve_structure,
                            'markdown_provenance': chapter_meta.get('markdown_provenance', {}),
                        })
                    except Exception as e:
                        print(f"⚠️ Could not convert enhanced text to HTML: {e}")
                        if content and '<' not in content[:200].lower():
                            body = ''.join(f"<p>{line}</p>" for line in content.splitlines() if line.strip())
                            content = f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>{body}</body></html>"
                elif content and '<' not in content[:200].lower():
                    # Provider returned plain text in standard mode; wrap minimally
                    body = ''.join(f"<p>{line}</p>" for line in content.splitlines() if line.strip())
                    content = f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>{body}</body></html>"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(ensure_utf8_html_document(content))
                
                # Add realistic progress entry
                progress_data["chapters"][content_hash] = {
                    "status": "completed",
                    "output_file": filename,
                    "actual_num": chapter_num,
                    "chapter_num": chapter_num,
                    "content_hash": content_hash,
                    "original_basename": original_basename,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "translation_time": 2.5,  # Fake but realistic
                    "token_count": chapter_info.get('text_length', 5000) // 4,  # Rough estimate
                    # model_var can be a Tk variable or plain string depending on context
                    "model": (
                        self.gui.model_var.get()
                        if hasattr(getattr(self.gui, "model_var", None), "get")
                        else (self.gui.model_var if hasattr(self.gui, "model_var") else getattr(self.gui, "model", ""))
                    ),
                    "from_async": True
                }
                
                # Add content hash tracking
                progress_data["content_hashes"][content_hash] = {
                    "chapter_key": content_hash,
                    "chapter_num": chapter_num,
                    "status": "completed",
                    "index": chapter_num - 1
                }
            
            # Save realistic progress file
            progress_file = os.path.join(output_dir, 'translation_progress.json')
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            self._log(f"✅ Saved {len(sorted_results)} chapters to: {output_dir}")
            
            self._async_msgbox('information', "Async Translation Complete",
                f"Successfully saved {len(sorted_results)} translated chapters to:\n{output_dir}\n\n"
                "Ready for EPUB conversion or further processing."
            )
                
        except Exception as e:
            self._log(f"❌ Error handling completed job: {e}")
            import traceback
            self._log(traceback.format_exc())
            self._async_msgbox('critical', "Error", f"Failed to process results: {str(e)}")

    def _show_error_details(self, job):
        """Show details from error file"""
        if not job.metadata.get('error_file_id'):
            return
            
        try:
            # Get API key using the helper method
            api_key = self._get_api_key_from_gui()
            headers = {'Authorization': f'Bearer {api_key}'}
            
            # Download error file
            response = requests.get(
                f'https://api.openai.com/v1/files/{job.metadata["error_file_id"]}/content',
                headers=headers
            )
            
            if response.status_code == 200:
                # Parse first few errors
                errors = []
                for i, line in enumerate(response.text.strip().split('\n')[:5]):  # Show first 5 errors
                    if line:
                        try:
                            error_data = json.loads(line)
                            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                            errors.append(f"• {error_msg}")
                        except:
                            pass
                            
                error_text = '\n'.join(errors)
                if len(response.text.strip().split('\n')) > 5:
                    newline = '\n'
                    error_text += f"\n\n... and {len(response.text.strip().split(newline)) - 5} more errors"
                    
                self._async_msgbox('critical', "Batch Processing Errors",
                    f"All requests failed with errors:\n\n{error_text}\n\n"
                    "Common causes:\n"
                    "• Invalid API key or insufficient permissions\n"
                    "• Model not available in your region\n"
                    "• Malformed request format"
                )
            
        except Exception as e:
            print(f"Failed to retrieve error details: {e}")

    def _extract_chapter_number(self, custom_id):
        """Extract chapter number from custom ID"""
        match = re.search(r'chapter[_-](\d+)', custom_id, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _get_api_key_from_gui(self) -> str:
        """Retrieve API key using same logic as processor"""
        try:
            # Prefer the processor's validated helper when available
            if hasattr(self, "processor") and hasattr(self.processor, "_get_api_key"):
                return self.processor._get_api_key()
        except Exception:
            pass

        # Fallback to direct GUI inspection to avoid failure
        if hasattr(self.gui, "api_key_entry"):
            if hasattr(self.gui.api_key_entry, "text"):
                return self.gui.api_key_entry.text().strip()
            return self.gui.api_key_entry.get().strip()
        if hasattr(self.gui, "api_key_var"):
            return self.gui.api_key_var.get().strip()
        return os.getenv("API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def default_jobs_file():
    """The job list of a GUI-free front end: ``async_jobs.json`` in the app-data folder.

    ``mobile_runtime.data_dir`` returns ``GLOSSARION_DATA_DIR`` on Glossarion Mobile and the
    given default elsewhere, so on desktop this is the processor's own default (next to the
    module; the frozen desktop dialog uses the exe folder instead).
    """
    default_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        from mobile_runtime import data_dir
        folder = data_dir(default_dir)
    except Exception:
        folder = default_dir
    return os.path.join(folder, 'async_jobs.json')


class HeadlessAsyncBatch(AsyncBatchJobMixin):
    """The async batch workflow without Qt (Glossarion Mobile's Tools > Async batch, tests).

    ``owner`` is what the dialog reads as ``self.gui`` (``headless_owner.HeadlessOwner`` on
    mobile; its ``file_path`` is the input to submit). ``host`` (optional, a
    ``job_runner.JobHost``) gets the questions (``ask('async_batch_question', level=, title=,
    text=, buttons=, default=)``), notices (``emit('async_batch_message', level=, title=, text=)``),
    cost text (``async_batch_cost``), job rows (``async_batch_jobs``) and polls the stop latch.
    ``answers`` presets answers by dialog title (``{"Start Async Processing": "yes"}``).

    Every public method runs the dialog's own handler, so the checks, messages, job list
    writes and output files are the desktop's.
    """

    def __init__(self, owner, *, host=None, jobs_file=None, answers=None):
        self.parent = None
        self.gui = owner
        self.host = host
        self.processor = AsyncAPIProcessor(owner, jobs_file=jobs_file or default_jobs_file())
        self.selected_job_id = None
        self.selected_job_ids = []
        self.polling_jobs = set()  # Track which jobs are being polled
        self.dialog = None
        self.processing_thread = None
        self.answers = dict(answers or {})
        self.messages = []
        self.saved_output_dirs = []
        self.cost_info = "Select chapters to see cost estimate"
        self.start_enabled = True

    # ---- read side ---------------------------------------------------------------------
    @property
    def jobs(self):
        """``{job_id: AsyncJobInfo}`` of the job list file."""
        return self.processor.jobs

    def rows(self):
        """The job list as the dialog shows it (``job_display_row`` per job, file order)."""
        return [job_display_row(job_id, job) for job_id, job in self.processor.jobs.items()]

    def model_status(self):
        """``{"model", "supported", "text"}`` of the dialog's "Current Model" row."""
        model_name = gui_model_name(self.gui)
        supported, text = async_support_status(self.processor, model_name)
        return {"model": model_name, "supported": supported, "text": text}

    def reload(self):
        """Re-read the job list file (another front end may have changed it)."""
        self.processor.jobs = {}
        self.processor._load_jobs()
        return self.rows()

    # ---- actions (the dialog's handlers) -------------------------------------------------
    def _run(self, handler, job_ids=None):
        self.messages = []
        if job_ids is not None:
            self.selected_job_ids = [str(job_id) for job_id in job_ids if job_id]
            self.selected_job_id = self.selected_job_ids[0] if self.selected_job_ids else None
        handler()
        return list(self.messages)

    def refresh_statuses(self):
        """Poll every pending / processing job once (the dialog's auto-refresh); returns the rows."""
        refresh_pending_job_statuses(self.processor)
        self._refresh_jobs_list()
        return self.rows()

    def estimate(self):
        """"Estimate Cost Only" for ``owner.file_path``; returns the cost text."""
        self._run(self._estimate_cost)
        return self.cost_info

    def submit(self):
        """"Start Async Processing" for ``owner.file_path``; waits for the submission thread.

        Returns the new ``AsyncJobInfo`` (None when a check or question stopped it).
        """
        before = set(self.processor.jobs)
        self.processing_thread = None
        self._run(self._start_processing)
        thread = self.processing_thread
        if thread is not None:
            thread.join()
        new_ids = [job_id for job_id in self.processor.jobs if job_id not in before]
        return self.processor.jobs[new_ids[-1]] if new_ids else None

    def check_status(self, job_id):
        """"Check Status" for one job; returns the messages shown (the status text)."""
        return self._run(self._check_selected_status, [job_id])

    def retrieve(self, job_ids):
        """"Retrieve Results": downloads completed jobs and writes their output folders.

        Returns the output folders written (in order).
        """
        self.saved_output_dirs = []
        self._run(self._retrieve_selected_results, job_ids)
        return list(self.saved_output_dirs)

    def cancel(self, job_ids):
        """"Cancel Job" for the given jobs; returns the messages shown."""
        return self._run(self._cancel_selected_job, job_ids)

    def delete(self, job_ids):
        """"Delete Selected" (local list only); returns the messages shown."""
        return self._run(self._delete_selected_job, job_ids)

    def clear_completed(self):
        """"Clear Completed"; returns the messages shown."""
        return self._run(self._clear_completed_jobs)


__all__ = [
    "AsyncAPIProcessor",
    "AsyncAPIStatus",
    "AsyncBatchJobMixin",
    "AsyncJobInfo",
    "HeadlessAsyncBatch",
    "MB_CANCEL",
    "MB_NO",
    "MB_OK",
    "MB_YES",
    "async_support_status",
    "default_jobs_file",
    "gui_model_name",
    "job_display_row",
    "refresh_pending_job_statuses",
    "selected_job_progress",
]
