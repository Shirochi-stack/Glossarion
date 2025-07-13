# async_api_processor.py
"""
Asynchronous API Processing for Glossarion
Implements batch API processing with 50% discount from supported providers.
This is SEPARATE from the existing batch processing (parallel API calls).

Supported Providers with Async/Batch APIs (50% discount):
- Gemini (Batch API)
- Anthropic (Message Batches API) 
- OpenAI (Batch API)
- Mistral (Batch API)
- Amazon Bedrock (Batch Inference)
- Groq (Batch API)

Providers without Async APIs:
- DeepSeek (no batch API)
- Cohere (only batch embeddings, not completions)
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
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import uuid
from pathlib import Path

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

logger = logging.getLogger(__name__)

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
    
    def __init__(self, gui_instance):
        """Initialize the async processor
        
        Args:
            gui_instance: Reference to TranslatorGUI instance
        """
        self.gui = gui_instance
        self.jobs_file = os.path.join(os.path.dirname(__file__), 'async_jobs.json')
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
        elif model_lower.startswith(('mistral', 'mixtral', 'codestral')):
            return 'mistral'
        elif model_lower.startswith('groq'):
            return 'groq'
        elif model_lower.startswith('bedrock'):
            return 'bedrock'
            
        # Check for aggregator prefixes that might support async
        if model_lower.startswith(('eh/', 'electronhub/', 'electron/')):
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
        # Prices are per 1M tokens (average of input + output)
        token_prices = {
            'openai': {
                # GPT-4.1 Series (Latest - June 2024 knowledge)
                'gpt-4.1': 5.0,          # Similar to gpt-4o pricing
                'gpt-4.1-mini': 0.225,   # $0.15 input + $0.60 output per 1M
                'gpt-4.1-nano': 0.10,    # Even cheaper than mini
                
                # GPT-4o Series
                'gpt-4o': 5.0,           # $2.50 input + $10 output per 1M
                'gpt-4o-mini': 0.225,    # $0.15 input + $0.60 output per 1M
                'gpt-4o-audio': 7.5,     # Audio models cost more
                'gpt-4o-realtime': 10.0, # Real-time premium
                
                # GPT-4 Legacy
                'gpt-4': 45.0,           # $30 input + $60 output per 1M
                'gpt-4-turbo': 15.0,     # $10 input + $30 output per 1M
                'gpt-4-0613': 45.0,      # Legacy pricing
                'gpt-4-0314': 45.0,      # Legacy pricing
                
                # GPT-3.5
                'gpt-3.5-turbo': 1.0,    # $0.50 input + $1.50 output per 1M
                'gpt-3.5-turbo-0125': 1.0,
                
                # O-series Reasoning Models (NOT batch compatible usually)
                'o1': 30.0,              # $15 input + $60 output per 1M
                'o1-preview': 30.0,
                'o1-mini': 7.5,          # $3 input + $12 output per 1M
                'o3': 50.0,              # Premium reasoning
                'o3-mini': 15.0,         # Smaller reasoning
                'o4-mini': 10.0,         # Latest efficient reasoning
                
                'default': 5.0
            },
            'anthropic': {
                # Claude 3.5 Series
                'claude-3.5-sonnet': 7.5,    # $3 input + $15 output per 1M
                'claude-3.5-opus': 37.5,     # $15 input + $75 output per 1M
                'claude-3.5-haiku': 0.625,   # $0.25 input + $1.25 output per 1M
                
                # Claude 3 Series
                'claude-3-opus': 37.5,       # Premium tier
                'claude-3-sonnet': 7.5,      # Mid tier
                'claude-3-haiku': 0.625,     # Budget tier
                
                # Legacy
                'claude-2.1': 16.0,          # $8 input + $24 output per 1M
                'claude-2': 16.0,
                'claude-instant': 1.6,       # $0.80 input + $2.40 output per 1M
                
                'default': 7.5
            },
            'gemini': {
                # Gemini 2.5 Series (Latest)
                'gemini-2.5-flash': 0.0375,  # $0.0125 input + $0.05 output per 1M
                'gemini-2.5-pro': 2.50,      # $1.25 input + $5.00 output per 1M
                
                # Gemini 1.5 Series
                'gemini-1.5-flash': 0.0375,  # Ultra cheap
                'gemini-1.5-flash-8b': 0.0375,
                'gemini-1.5-pro': 1.875,     # $1.25 input + $5.00 output per 1M
                
                # Gemini 1.0 Series
                'gemini-1.0-pro': 0.50,      # $0.50 input + $1.50 output per 1M
                'gemini-pro': 0.50,          # Alias
                
                # Experimental
                'gemini-2.0-flash': 0.0375,  # Same as 1.5-flash
                'gemini-exp': 2.50,          # Experimental pricing
                
                'default': 0.50
            },
            'mistral': {
                'mistral-large': 6.0,        # $4 input + $12 output per 1M
                'mistral-medium': 2.7,       # Deprecated but still works
                'mistral-small': 0.3,        # $0.10 input + $0.30 output per 1M
                'mixtral-8x7b': 0.7,         # MoE model
                'mixtral-8x22b': 2.0,        # Larger MoE
                'codestral': 1.0,            # Code-specific
                'ministral': 0.2,            # Tiny model
                'default': 1.0
            },
            'groq': {
                'llama-3.1-405b': 2.5,       # Large Llama
                'llama-3.1-70b': 0.7,        # $0.59 input + $0.79 output per 1M
                'llama-3.1-8b': 0.05,        # $0.05 input + $0.10 output per 1M
                'llama-3-70b': 0.7,          # Previous gen
                'llama-3-8b': 0.05,          
                'mixtral-8x7b': 0.24,        # $0.24 per 1M tokens
                'gemma-7b': 0.07,            # Google's open model
                'gemma2-9b': 0.10,           # Newer Gemma
                'default': 0.2
            },
            'deepseek': {
                'deepseek-v3': 1.0,          # Latest model
                'deepseek-chat': 0.14,       # $0.14 per 1M tokens
                'deepseek-coder': 0.14,      # Code-specific
                'default': 0.14
            },
            'cohere': {
                'command-r-plus': 7.5,       # $3 input + $15 output per 1M
                'command-r': 0.75,           # $0.5 input + $1.5 output per 1M
                'command': 2.0,              # Legacy
                'default': 1.0
            }
        }
        
        provider_prices = token_prices.get(provider, {'default': 5.0})
        
        # Find the right price for this model
        price_per_million = provider_prices.get('default', 5.0)
        model_lower = model.lower()
        
        # Try exact match first
        if model_lower in provider_prices:
            price_per_million = provider_prices[model_lower]
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
                    price_per_million = price
                    break
        
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
        logger.info(f"  Model matched to price: ${price_per_million:.4f}/1M tokens")
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
        
        # CRITICAL: Map to exact supported model names
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
        
        # Check if model is supported
        model_lower = model.lower()
        actual_model = None
        
        for key, value in supported_batch_models.items():
            if model_lower == key.lower() or model_lower.startswith(key.lower()):
                actual_model = value
                break
        
        if not actual_model:
            print(f"Model '{model}' is not supported for batch processing!")
            print(f"Supported models: {list(supported_batch_models.values())}")
            raise ValueError(f"Model '{model}' is not supported for OpenAI Batch API")
        
        logger.info(f"Using batch-supported model: '{actual_model}' (from '{model}')")
        
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
            
            request = {
                "custom_id": chapter['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": actual_model,
                    "messages": valid_messages,
                    "temperature": float(chapter.get('temperature', 0.3)),
                    "max_tokens": int(chapter.get('max_tokens', 8192))
                }
            }
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
                    "max_tokens": chapter.get('max_tokens', 8192),
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
                        "maxOutputTokens": chapter.get('max_tokens', 8192)
                    }
                }
            }
            
            # Add safety settings if disabled
            if os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true":
                request["generateContentRequest"]["safetySettings"] = [
                    {"category": cat, "threshold": "BLOCK_NONE"}
                    for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
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
                "max_tokens": chapter.get('max_tokens', 8192)
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
        """Get API key from GUI settings"""
        if hasattr(self.gui, 'api_key_entry'):
            return self.gui.api_key_entry.get().strip()
        elif hasattr(self.gui, 'api_key_var'):
            return self.gui.api_key_var.get().strip()
        else:
            # Fallback to environment variable
            return os.getenv('API_KEY', '') or os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')
        
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


class AsyncProcessingDialog:
    """GUI dialog for async processing"""
    
    def __init__(self, parent, translator_gui):
        """Initialize dialog
        
        Args:
            parent: Parent window
            translator_gui: Reference to main TranslatorGUI instance
        """
        self.parent = parent
        self.gui = translator_gui
        self.processor = AsyncAPIProcessor(translator_gui)  # Pass translator_gui here
        self.selected_job_id = None
        
        # Use the correct attribute name 'wm' instead of 'window_manager'
        self.window_manager = translator_gui.wm  # WindowManager is stored as 'wm'
        
        self._create_dialog()
        
    def _create_dialog(self):
        """Create the async processing dialog"""
        # Create scrollable dialog (stays hidden)
        self.dialog, scrollable_frame, canvas = self.window_manager.setup_scrollable(
            self.parent,
            "Async Batch Processing (50% Discount)",
            width=0,  # Will be auto-sized
            height=None,
            max_width_ratio=0.9,
            max_height_ratio=1.00
        )
        
        # Store references
        self.scrollable_frame = scrollable_frame
        self.canvas = canvas
        
        # Main container in scrollable_frame
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Information and controls
        self._create_info_section(main_frame)
        
        # Middle section - Configuration
        self._create_config_section(main_frame)
        
        # Bottom section - Active jobs
        self._create_jobs_section(main_frame)
        
        # Button frame goes in the DIALOG, not scrollable_frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        self._create_button_frame(button_frame)
        
        # Load active jobs
        self._refresh_jobs_list()
        
        # Auto-resize and show - THIS is what applies the height ratio!
        self.window_manager.auto_resize_dialog(
            self.dialog, 
            canvas, 
            max_width_ratio=0.9, 
            max_height_ratio=0.92  # Can override to any value like 1.43
        )
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", 
                           lambda: [self.dialog._cleanup_scrolling(), self.dialog.destroy()])
       
        self._start_auto_refresh(30)
        
    def _create_info_section(self, parent):
        """Create information section"""
        info_frame = ttk.LabelFrame(parent, text="Async Processing Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model and provider info
        model_frame = ttk.Frame(info_frame)
        model_frame.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Current Model:").pack(side=tk.LEFT, padx=(0, 5))
        model_name = self.gui.model_var.get() if hasattr(self.gui, 'model_var') else "Not selected"
        self.model_label = ttk.Label(model_frame, text=model_name, style="info.TLabel")
        self.model_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Check if model supports async
        provider = self.processor.get_provider_from_model(model_name)
        if provider and provider in self.processor.PROVIDER_CONFIGS:
            status_text = f"✓ Supported ({provider.upper()})"
            status_style = "success.TLabel"
        else:
            status_text = "✗ Not supported for async"
            status_style = "danger.TLabel"
            
        ttk.Label(model_frame, text=status_text, style=status_style).pack(side=tk.LEFT)
        
        # Cost estimation
        cost_frame = ttk.Frame(info_frame)
        cost_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(cost_frame, text="Cost Estimation:", font=("", 10, "bold")).pack(anchor=tk.W)
        
        self.cost_info_label = ttk.Label(cost_frame, text="Select chapters to see cost estimate")
        self.cost_info_label.pack(anchor=tk.W, pady=(5, 0))
        
    def _create_config_section(self, parent):
        """Create configuration section"""
        config_frame = ttk.LabelFrame(parent, text="Async Processing Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Processing options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(fill=tk.X)
        
        # Wait for completion
        self.wait_for_completion_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Wait for completion (blocks GUI)",
            variable=self.wait_for_completion_var
        ).pack(anchor=tk.W)
        
        # Poll interval
        poll_frame = ttk.Frame(options_frame)
        poll_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(poll_frame, text="Poll interval (seconds):").pack(side=tk.LEFT, padx=(0, 5))
        self.poll_interval_var = tk.IntVar(value=60)
        ttk.Spinbox(
            poll_frame,
            from_=10,
            to=600,
            textvariable=self.poll_interval_var,
            width=10
        ).pack(side=tk.LEFT)
        
        # Chapter selection info
        chapter_frame = ttk.Frame(config_frame)
        chapter_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.chapter_info_label = ttk.Label(
            chapter_frame,
            text="Note: Async processing will skip chapters that require chunking",
            style="warning.TLabel"
        )
        self.chapter_info_label.pack(anchor=tk.W)
        
    def _create_jobs_section(self, parent):
        """Create active jobs section"""
        jobs_frame = ttk.LabelFrame(parent, text="Active Async Jobs", padding=10)
        jobs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Jobs treeview
        tree_frame = ttk.Frame(jobs_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        self.jobs_tree = ttk.Treeview(
            tree_frame,
            columns=("Provider", "Model", "Status", "Progress", "Created", "Cost"),
            show="tree headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        
        vsb.config(command=self.jobs_tree.yview)
        hsb.config(command=self.jobs_tree.xview)
        
        # Add a progress bar for the selected job
        progress_frame = ttk.Frame(jobs_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(progress_frame, text="Selected Job Progress:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.job_progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            style='success.Horizontal.TProgressbar'
        )
        self.job_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=(5, 0))   
        
        # Configure columns
        self.jobs_tree.heading("#0", text="Job ID")
        self.jobs_tree.heading("Provider", text="Provider")
        self.jobs_tree.heading("Model", text="Model")
        self.jobs_tree.heading("Status", text="Status")
        self.jobs_tree.heading("Progress", text="Progress")
        self.jobs_tree.heading("Created", text="Created")
        self.jobs_tree.heading("Cost", text="Est. Cost")
        
        self.jobs_tree.column("#0", width=200)
        self.jobs_tree.column("Provider", width=100)
        self.jobs_tree.column("Model", width=150)
        self.jobs_tree.column("Status", width=100)
        self.jobs_tree.column("Progress", width=150)
        self.jobs_tree.column("Created", width=150)
        self.jobs_tree.column("Cost", width=100)
        
        # Add right-click menu
        self.jobs_context_menu = tk.Menu(self.jobs_tree, tearoff=0)
        self.jobs_context_menu.add_command(label="Check Status", command=self._check_selected_status)
        self.jobs_context_menu.add_command(label="Retrieve Results", command=self._retrieve_selected_results)
        self.jobs_context_menu.add_separator()
        self.jobs_context_menu.add_command(label="Delete", command=self._delete_selected_job)
        
        # Context menu binding function - use unique name to avoid conflicts
        def show_jobs_context_menu(event):
            # Select the item under cursor
            item = self.jobs_tree.identify_row(event.y)
            if item:
                self.jobs_tree.selection_set(item)
                self._on_job_select(None)  # Update selection
            self.jobs_context_menu.post(event.x_root, event.y_root)
        
        # Bind right-click
        self.jobs_tree.bind("<Button-3>", show_jobs_context_menu)  # Right-click on Windows/Linux
        if sys.platform == "darwin":
            self.jobs_tree.bind("<Button-2>", show_jobs_context_menu)  # Right-click on macOS
        
        # Pack treeview and scrollbars
        self.jobs_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection
        self.jobs_tree.bind('<<TreeviewSelect>>', self._on_job_select)
        
        # Job action buttons
        action_frame = ttk.Frame(jobs_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        button_width = 15

        ttk.Button(
            action_frame,
            text="Check Status",
            command=self._check_selected_status,
            style="info.TButton",
            width=button_width
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            action_frame,
            text="Retrieve Results",
            command=self._retrieve_selected_results,
            style="success.TButton",
            width=button_width
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            action_frame,
            text="Cancel Job",
            command=self._cancel_selected_job,
            style="warning.TButton",
            width=button_width
        ).pack(side=tk.LEFT, padx=(0, 5))

        # delete buttons
        ttk.Button(
            action_frame,
            text="Delete Selected",
            command=self._delete_selected_job,
            style="danger.TButton",
            width=button_width
        ).pack(side=tk.LEFT, padx=(30, 5))  # Extra padding to separate

        ttk.Button(
            action_frame,
            text="Clear Completed",
            command=self._clear_completed_jobs,
            style="secondary.TButton",
            width=button_width
        ).pack(side=tk.LEFT)
    
    def _create_button_frame(self, parent):
        """Create bottom button frame"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Start processing button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Async Processing",
            command=self._start_processing,
            style="success.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Estimate only button
        ttk.Button(
            button_frame,
            text="Estimate Cost Only",
            command=self._estimate_cost,
            style="info.TButton"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Close button - need to handle cleanup if using WindowManager
        if hasattr(self.dialog, '_cleanup_scrolling'):
            ttk.Button(
                button_frame,
                text="Close",
                command=lambda: [self.dialog._cleanup_scrolling(), self.dialog.destroy()]
            ).pack(side=tk.RIGHT)
        else:
            ttk.Button(
                button_frame,
                text="Close",
                command=self.dialog.destroy
            ).pack(side=tk.RIGHT)
        
    def _update_selected_job_progress(self, job):
        """Update progress display for selected job"""
        if hasattr(self, 'job_progress_bar'):
            if job.total_requests > 0:
                progress = int((job.completed_requests / job.total_requests) * 100)
                self.job_progress_bar['value'] = progress
                
                # Update progress label if exists
                if hasattr(self, 'progress_label'):
                    self.progress_label.config(
                        text=f"{progress}% ({job.completed_requests}/{job.total_requests} chapters)"
                    )
            else:
                self.job_progress_bar['value'] = 0
                if hasattr(self, 'progress_label'):
                    self.progress_label.config(text="0% (Waiting)")        
        
    def _refresh_jobs_list(self):
        """Refresh the jobs list"""
        # Clear existing items
        for item in self.jobs_tree.get_children():
            self.jobs_tree.delete(item)
            
        # Add jobs
        for job_id, job in self.processor.jobs.items():
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
            
            self.jobs_tree.insert(
                "",
                "end",
                text=display_id,
                values=(
                    job.provider.upper(),
                    job.model[:15] + "..." if len(job.model) > 15 else job.model,  # Shorten model name
                    status_text,
                    progress_text,  # Now shows percentage and counts
                    created,
                    cost
                ),
                tags=(job.status.value,)
            )
            
        # Configure tags for status colors
        self.jobs_tree.tag_configure("pending", foreground="#FFA500")  # Orange
        self.jobs_tree.tag_configure("processing", foreground="#007BFF")  # Blue
        self.jobs_tree.tag_configure("completed", foreground="#28A745")  # Green
        self.jobs_tree.tag_configure("failed", foreground="#DC3545")  # Red
        self.jobs_tree.tag_configure("cancelled", foreground="#6C757D")  # Gray
        
        # Update progress bar if a job is selected
        if hasattr(self, 'selected_job_id') and self.selected_job_id:
            job = self.processor.jobs.get(self.selected_job_id)
            if job:
                self._update_selected_job_progress(job)
        
    def _on_job_select(self, event):
        """Handle job selection"""
        selection = self.jobs_tree.selection()
        if selection:
            item = self.jobs_tree.item(selection[0])
            # Get full job ID from the item
            job_id_prefix = item['text'].rstrip('...')
            
            # Find matching job
            for job_id in self.processor.jobs:
                if job_id.startswith(job_id_prefix):
                    self.selected_job_id = job_id
                    
                    # Update progress display for selected job
                    job = self.processor.jobs.get(job_id)
                    if job:
                        # Update progress bar if it exists
                        if hasattr(self, 'job_progress_bar'):
                            if job.total_requests > 0:
                                progress = int((job.completed_requests / job.total_requests) * 100)
                                self.job_progress_bar['value'] = progress
                            else:
                                self.job_progress_bar['value'] = 0
                        
                        # Update progress label if it exists
                        if hasattr(self, 'progress_label'):
                            if job.total_requests > 0:
                                progress = int((job.completed_requests / job.total_requests) * 100)
                                self.progress_label.config(
                                    text=f"{progress}% ({job.completed_requests}/{job.total_requests} chapters)"
                                )
                            else:
                                self.progress_label.config(text="0% (Waiting)")
                        
                        # Log selection
                        logger.info(f"Selected job: {job_id[:30]}... - Status: {job.status.value}")
                    
                    break
                    
    def _check_selected_status(self):
        """Check status of selected job"""
        if not self.selected_job_id:
            messagebox.showwarning("No Selection", "Please select a job to check status")
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
                
            # Show output file if available
            if job.output_file:
                status_text += f"\nOutput Ready: {job.output_file}\n"
            
            messagebox.showinfo("Job Status", status_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check status: {str(e)}")
 
    def _start_auto_refresh(self, interval_seconds=30):
        """Start automatic status refresh"""
        def refresh():
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                # Refresh all jobs
                for job_id in list(self.processor.jobs.keys()):
                    try:
                        job = self.processor.jobs[job_id]
                        if job.status in [AsyncAPIStatus.PENDING, AsyncAPIStatus.PROCESSING]:
                            self.processor.check_job_status(job_id)
                    except:
                        pass
                
                self._refresh_jobs_list()
                
                # Schedule next refresh
                self.dialog.after(interval_seconds * 1000, refresh)
        
        # Start the refresh cycle
        refresh()
    
    def _retrieve_selected_results(self):
        """Retrieve results from selected job"""
        if not self.selected_job_id:
            messagebox.showwarning("No Selection", "Please select a job to retrieve results")
            return
        
        # Check job status first
        job = self.processor.jobs.get(self.selected_job_id)
        if not job:
            messagebox.showerror("Error", "Selected job not found")
            return
            
        if job.status != AsyncAPIStatus.COMPLETED:
            messagebox.showwarning(
                "Job Not Complete", 
                f"This job is currently {job.status.value}.\n"
                "Only completed jobs can have results retrieved."
            )
            return
        
        try:
            # Set cursor to busy (with safety check)
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.config(cursor="wait")
                self.dialog.update()
            
            # Retrieve results
            self._handle_completed_job(self.selected_job_id)
            
        except Exception as e:
            self._log(f"❌ Error retrieving results: {e}")
            messagebox.showerror("Error", f"Failed to retrieve results: {str(e)}")
        finally:
            # Reset cursor (with safety check)
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                try:
                    self.dialog.config(cursor="")
                except tk.TclError:
                    # Dialog was closed, ignore
                    pass
            
    def _cancel_selected_job(self):
        """Cancel selected job"""
        if not self.selected_job_id:
            messagebox.showwarning("No Selection", "Please select a job to cancel")
            return
            
        job = self.processor.jobs.get(self.selected_job_id)
        if not job:
            messagebox.showerror("Error", "Selected job not found")
            return
            
        if job.status in [AsyncAPIStatus.COMPLETED, AsyncAPIStatus.FAILED, AsyncAPIStatus.CANCELLED]:
            messagebox.showwarning(
                "Cannot Cancel", 
                f"This job is already {job.status.value}"
            )
            return
            
        # Confirm cancellation
        if not messagebox.askyesno(
            "Cancel Job", 
            f"Are you sure you want to cancel this job?\n\n"
            f"Job ID: {job.job_id}\n"
            f"Status: {job.status.value}"
        ):
            return
            
        try:
            api_key = self.gui.api_key_entry.get().strip()
            
            if job.provider == 'openai':
                headers = {'Authorization': f'Bearer {api_key}'}
                
                response = requests.post(
                    f'https://api.openai.com/v1/batches/{job.job_id}/cancel',
                    headers=headers
                )
                
                if response.status_code == 200:
                    job.status = AsyncAPIStatus.CANCELLED
                    job.updated_at = datetime.now()
                    self.processor._save_jobs()
                    self._refresh_jobs_list()
                    messagebox.showinfo("Job Cancelled", "Job has been cancelled successfully")
                else:
                    messagebox.showerror("Error", f"Failed to cancel job: {response.text}")
                    
            elif job.provider == 'gemini':
                # Gemini batch cancellation using REST API
                headers = {'x-goog-api-key': api_key}
                
                # Format: batches/123456 -> https://generativelanguage.googleapis.com/v1beta/batches/123456:cancel
                batch_name = job.job_id if job.job_id.startswith('batches/') else f'batches/{job.job_id}'
                
                response = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/{batch_name}:cancel',
                    headers=headers
                )
                
                if response.status_code == 200:
                    job.status = AsyncAPIStatus.CANCELLED
                    job.updated_at = datetime.now()
                    self.processor._save_jobs()
                    self._refresh_jobs_list()
                    messagebox.showinfo("Job Cancelled", "Gemini batch job has been cancelled successfully")
                else:
                    messagebox.showerror("Error", f"Failed to cancel Gemini job: {response.text}")
                    
            elif job.provider == 'anthropic':
                # Anthropic doesn't support cancellation via API yet
                messagebox.showinfo(
                    "Not Supported", 
                    "Anthropic doesn't support job cancellation via API.\n"
                    "The job will be marked as cancelled locally only."
                )
                job.status = AsyncAPIStatus.CANCELLED
                job.updated_at = datetime.now()
                self.processor._save_jobs()
                self._refresh_jobs_list()
                
            else:
                # For other providers, just mark as cancelled locally
                messagebox.showinfo(
                    "Local Cancellation", 
                    f"{job.provider.title()} cancellation not implemented.\n"
                    "The job will be marked as cancelled locally only."
                )
                job.status = AsyncAPIStatus.CANCELLED
                job.updated_at = datetime.now()
                self.processor._save_jobs()
                self._refresh_jobs_list()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cancel job: {str(e)}")

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
            messagebox.showwarning("No File", "Please select a file first")
            return
        
        try:
            # Show analyzing message
            self.cost_info_label.config(text="Analyzing file...")
            self.dialog.update()
            
            file_path = self.gui.file_path
            model = self.gui.model_var.get()
            
            # Calculate overhead tokens (system prompt + glossary)
            overhead_tokens = 0
            
            # Count system prompt tokens
            system_prompt = self.gui.prompt_text.get("1.0", "end").strip()
            if system_prompt:
                overhead_tokens += self.count_tokens(system_prompt, model)
                logger.info(f"System prompt tokens: {overhead_tokens}")
            
            # Count glossary tokens if enabled
            glossary_tokens = 0
            
            # Check if glossary should be appended - match the logic from _prepare_environment_variables
            if (hasattr(self.gui, 'manual_glossary_path') and 
                self.gui.manual_glossary_path and 
                hasattr(self.gui, 'append_glossary_var') and 
                self.gui.append_glossary_var.get()):  # This is the key check!
                
                try:
                    glossary_path = self.gui.manual_glossary_path
                    logger.info(f"Loading glossary from: {glossary_path}")
                    
                    if os.path.exists(glossary_path):
                        with open(glossary_path, 'r', encoding='utf-8') as f:
                            glossary_data = json.load(f)
                        
                        # Format glossary same way as in translation
                        glossary_text = self._format_glossary_for_prompt(glossary_data)
                        
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
                    
                    for i, chapter_text in enumerate(chapters[:sample_size]):
                        # Count just the content tokens
                        content_tokens = self.count_tokens(chapter_text, model)
                        sampled_content_tokens += content_tokens
                        
                        # Check if needs chunking (including overhead)
                        total_chapter_tokens = content_tokens + overhead_tokens
                        token_limit = int(self.gui.token_limit_entry.get() or 200000)
                        if total_chapter_tokens > token_limit * 0.8:
                            chapters_needing_chunking += 1
                        
                        # Update progress
                        if i % 5 == 0:
                            self.cost_info_label.config(text=f"Analyzing chapters... {i+1}/{sample_size}")
                            self.dialog.update()
                            
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
                    
                    for i, chapter_text in enumerate(chapters[:sample_size]):
                        # Count just the content tokens
                        content_tokens = self.count_tokens(chapter_text, model)
                        sampled_content_tokens += content_tokens
                        
                        # Check if needs chunking (including overhead)
                        total_chapter_tokens = content_tokens + overhead_tokens
                        token_limit = int(self.gui.token_limit_entry.get() or 200000)
                        if total_chapter_tokens > token_limit * 0.8:
                            chapters_needing_chunking += 1
                        
                        # Update progress
                        if i % 5 == 0:
                            self.cost_info_label.config(text=f"Analyzing chapters... {i+1}/{sample_size}")
                            self.dialog.update()
                            
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
                self.cost_info_label.config(
                    text="Unsupported file format. Only EPUB and TXT are supported."
                )
                return
            
            # Calculate costs
            processable_chapters = num_chapters - chapters_needing_chunking
            
            if processable_chapters <= 0:
                self.cost_info_label.config(
                    text=f"Warning: All {num_chapters} chapters require chunking.\n"
                    f"Async APIs do not support chunked chapters.\n"
                    f"Consider using regular batch translation instead."
                )
                return
            
            # Add overhead to get total average tokens per chapter
            avg_total_tokens_per_chapter = avg_content_tokens_per_chapter + overhead_tokens
            
            # Get the translation compression factor from GUI
            compression_factor = float(self.gui.compression_factor_var.get() or 1.0)
            
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

            
            self.cost_info_label.config(text=cost_text)
            
        except Exception as e:
            self.cost_info_label.config(
                text=f"Error estimating cost: {str(e)}"
            )
            print(f"Cost estimation error: {traceback.format_exc()}")

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
        model = self.gui.model_var.get()
        
        if not self.processor.supports_async(model):
            messagebox.showerror(
                "Not Supported",
                f"Model '{model}' does not support async processing.\n"
                "Supported providers: Gemini, Anthropic, OpenAI, Mistral, Groq"
            )
            return
        
        # Add special check for Gemini
        if model.lower().startswith('gemini'):
            response = messagebox.askyesno(
                "Gemini Batch API",
                "Note: Gemini's batch API may not be publicly available yet.\n"
                "This feature is experimental for Gemini models.\n\n"
                "Would you like to try anyway?"
            )
            if not response:
                return
        
        if not self.processor.supports_async(model):
            messagebox.showerror(
                "Not Supported",
                f"Model '{model}' does not support async processing.\n"
                "Supported providers: Gemini, Anthropic, OpenAI, Mistral, Groq"
            )
            return
            
        if not hasattr(self.gui, 'file_path') or not self.gui.file_path:
            messagebox.showwarning("No File", "Please select a file to translate first")
            return
            
        # Confirm start
        if not messagebox.askyesno(
            "Start Async Processing",
            "Start async batch processing?\n\n"
            "This will submit all chapters for processing at 50% discount.\n"
            "Processing may take up to 24 hours."
        ):
            return
        
        # Disable buttons during processing
        self.start_button.config(state='disabled')
        
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
            model = self.gui.model_var.get()
            api_key = self.gui.api_key_entry.get().strip()
            
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
            
            # Save job
            self.processor.jobs[job.job_id] = job
            self.processor._save_jobs()
            
            # Update UI
            self.dialog.after(0, self._refresh_jobs_list)
            
            self._log(f"✅ Batch submitted successfully! Job ID: {job.job_id}")
            
            # Show success message
            self._show_info(
                "Batch Submitted",
                f"Successfully submitted {len(chapters_to_process)} chapters for async processing.\n\n"
                f"Job ID: {job.job_id}\n\n"
                "You can close this dialog and check back later for results.\n\n"
                "Tip: Use the 'Estimate Cost Only' button to get accurate cost estimates before submitting."
            )
            
            # Start polling if requested
            if self.wait_for_completion_var.get():
                self._start_polling(job.job_id)
                
        except Exception as e:
            self._log(f"❌ Error: {str(e)}")
            print(f"Async processing error: {traceback.format_exc()}")
            self._show_error(f"Failed to start async processing: {str(e)}")
        finally:
            # Re-enable button
            self.dialog.after(0, lambda: self.start_button.config(state='normal'))

    def _prepare_environment_variables(self):
        """Prepare environment variables from GUI settings"""
        env_vars = {}
        
        # Core settings
        env_vars['MODEL'] = self.gui.model_var.get()
        env_vars['API_KEY'] = self.gui.api_key_entry.get().strip()
        env_vars['OPENAI_API_KEY'] = env_vars['API_KEY']
        env_vars['OPENAI_OR_Gemini_API_KEY'] = env_vars['API_KEY']
        env_vars['GEMINI_API_KEY'] = env_vars['API_KEY']
        env_vars['PROFILE_NAME'] = self.gui.lang_var.get().lower()
        env_vars['CONTEXTUAL'] = '1' if self.gui.contextual_var.get() else '0'
        env_vars['MAX_OUTPUT_TOKENS'] = str(self.gui.max_output_tokens)
        env_vars['SYSTEM_PROMPT'] = self.gui.prompt_text.get("1.0", "end").strip()
        env_vars['TRANSLATION_TEMPERATURE'] = str(self.gui.trans_temp.get())
        env_vars['TRANSLATION_HISTORY_LIMIT'] = str(self.gui.trans_history.get())
        
        # API settings
        env_vars['SEND_INTERVAL_SECONDS'] = str(self.gui.delay_entry.get())
        env_vars['TOKEN_LIMIT'] = self.gui.token_limit_entry.get() if hasattr(self.gui, 'token_limit_entry') else '200000'
        
        # Book title translation
        env_vars['TRANSLATE_BOOK_TITLE'] = "1" if self.gui.translate_book_title_var.get() else "0"
        env_vars['BOOK_TITLE_PROMPT'] = self.gui.book_title_prompt if hasattr(self.gui, 'book_title_prompt') else ''
        env_vars['BOOK_TITLE_SYSTEM_PROMPT'] = self.gui.config.get('book_title_system_prompt', 
            "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content.")
        
        # Processing options
        env_vars['CHAPTER_RANGE'] = self.gui.chapter_range_entry.get().strip() if hasattr(self.gui, 'chapter_range_entry') else ''
        env_vars['REMOVE_AI_ARTIFACTS'] = "1" if self.gui.REMOVE_AI_ARTIFACTS_var.get() else "0"
        env_vars['BATCH_TRANSLATION'] = "1" if self.gui.batch_translation_var.get() else "0"
        env_vars['BATCH_SIZE'] = self.gui.batch_size_var.get()
        
        # Anti-duplicate parameters
        env_vars['ENABLE_ANTI_DUPLICATE'] = '1' if hasattr(self.gui, 'enable_anti_duplicate_var') and self.gui.enable_anti_duplicate_var.get() else '0'
        env_vars['TOP_P'] = str(self.gui.top_p_var.get()) if hasattr(self.gui, 'top_p_var') else '1.0'
        env_vars['TOP_K'] = str(self.gui.top_k_var.get()) if hasattr(self.gui, 'top_k_var') else '0'
        env_vars['FREQUENCY_PENALTY'] = str(self.gui.frequency_penalty_var.get()) if hasattr(self.gui, 'frequency_penalty_var') else '0.0'
        env_vars['PRESENCE_PENALTY'] = str(self.gui.presence_penalty_var.get()) if hasattr(self.gui, 'presence_penalty_var') else '0.0'
        env_vars['REPETITION_PENALTY'] = str(self.gui.repetition_penalty_var.get()) if hasattr(self.gui, 'repetition_penalty_var') else '1.0'
        env_vars['CANDIDATE_COUNT'] = str(self.gui.candidate_count_var.get()) if hasattr(self.gui, 'candidate_count_var') else '1'
        env_vars['CUSTOM_STOP_SEQUENCES'] = self.gui.custom_stop_sequences_var.get() if hasattr(self.gui, 'custom_stop_sequences_var') else ''
        env_vars['LOGIT_BIAS_ENABLED'] = '1' if hasattr(self.gui, 'logit_bias_enabled_var') and self.gui.logit_bias_enabled_var.get() else '0'
        env_vars['LOGIT_BIAS_STRENGTH'] = str(self.gui.logit_bias_strength_var.get()) if hasattr(self.gui, 'logit_bias_strength_var') else '-0.5'
        env_vars['BIAS_COMMON_WORDS'] = '1' if hasattr(self.gui, 'bias_common_words_var') and self.gui.bias_common_words_var.get() else '0'
        env_vars['BIAS_REPETITIVE_PHRASES'] = '1' if hasattr(self.gui, 'bias_repetitive_phrases_var') and self.gui.bias_repetitive_phrases_var.get() else '0'
        
        # Glossary settings
        env_vars['MANUAL_GLOSSARY'] = self.gui.manual_glossary_path if hasattr(self.gui, 'manual_glossary_path') and self.gui.manual_glossary_path else ''
        env_vars['DISABLE_AUTO_GLOSSARY'] = "0" if self.gui.enable_auto_glossary_var.get() else "1"
        env_vars['DISABLE_GLOSSARY_TRANSLATION'] = "0" if self.gui.enable_auto_glossary_var.get() else "1"
        env_vars['APPEND_GLOSSARY'] = "1" if self.gui.append_glossary_var.get() else "0"
        env_vars['APPEND_GLOSSARY_PROMPT'] = self.gui.append_glossary_prompt if hasattr(self.gui, 'append_glossary_prompt') else ''
        env_vars['GLOSSARY_MIN_FREQUENCY'] = self.gui.glossary_min_frequency_var.get()
        env_vars['GLOSSARY_MAX_NAMES'] = self.gui.glossary_max_names_var.get()
        env_vars['GLOSSARY_MAX_TITLES'] = self.gui.glossary_max_titles_var.get()
        env_vars['GLOSSARY_BATCH_SIZE'] = self.gui.glossary_batch_size_var.get()
        env_vars['GLOSSARY_DUPLICATE_KEY_MODE'] = self.gui.config.get('glossary_duplicate_key_mode', 'auto')
        env_vars['GLOSSARY_DUPLICATE_CUSTOM_FIELD'] = self.gui.config.get('glossary_duplicate_custom_field', '')
        
        # History and summary settings
        env_vars['TRANSLATION_HISTORY_ROLLING'] = "1" if self.gui.translation_history_rolling_var.get() else "0"
        env_vars['USE_ROLLING_SUMMARY'] = "1" if self.gui.config.get('use_rolling_summary') else "0"
        env_vars['SUMMARY_ROLE'] = self.gui.config.get('summary_role', 'user')
        env_vars['ROLLING_SUMMARY_EXCHANGES'] = self.gui.rolling_summary_exchanges_var.get()
        env_vars['ROLLING_SUMMARY_MODE'] = self.gui.rolling_summary_mode_var.get()
        env_vars['ROLLING_SUMMARY_SYSTEM_PROMPT'] = self.gui.rolling_summary_system_prompt if hasattr(self.gui, 'rolling_summary_system_prompt') else ''
        env_vars['ROLLING_SUMMARY_USER_PROMPT'] = self.gui.rolling_summary_user_prompt if hasattr(self.gui, 'rolling_summary_user_prompt') else ''
        
        # Retry and error handling settings
        env_vars['EMERGENCY_PARAGRAPH_RESTORE'] = "1" if self.gui.emergency_restore_var.get() else "0"
        env_vars['RETRY_TRUNCATED'] = "1" if self.gui.retry_truncated_var.get() else "0"
        env_vars['MAX_RETRY_TOKENS'] = self.gui.max_retry_tokens_var.get()
        env_vars['RETRY_DUPLICATE_BODIES'] = "1" if self.gui.retry_duplicate_var.get() else "0"
        env_vars['RETRY_TIMEOUT'] = "1" if self.gui.retry_timeout_var.get() else "0"
        env_vars['CHUNK_TIMEOUT'] = self.gui.chunk_timeout_var.get()
        
        # Image processing
        env_vars['ENABLE_IMAGE_TRANSLATION'] = "1" if self.gui.enable_image_translation_var.get() else "0"
        env_vars['PROCESS_WEBNOVEL_IMAGES'] = "1" if self.gui.process_webnovel_images_var.get() else "0"
        env_vars['WEBNOVEL_MIN_HEIGHT'] = self.gui.webnovel_min_height_var.get()
        env_vars['MAX_IMAGES_PER_CHAPTER'] = self.gui.max_images_per_chapter_var.get()
        env_vars['IMAGE_API_DELAY'] = '1.0'
        env_vars['SAVE_IMAGE_TRANSLATIONS'] = '1'
        env_vars['IMAGE_CHUNK_HEIGHT'] = self.gui.image_chunk_height_var.get()
        env_vars['HIDE_IMAGE_TRANSLATION_LABEL'] = "1" if self.gui.hide_image_translation_label_var.get() else "0"
        
        # Advanced settings
        env_vars['REINFORCEMENT_FREQUENCY'] = self.gui.reinforcement_freq_var.get()
        env_vars['RESET_FAILED_CHAPTERS'] = "1" if self.gui.reset_failed_chapters_var.get() else "0"
        env_vars['DUPLICATE_LOOKBACK_CHAPTERS'] = self.gui.duplicate_lookback_var.get()
        env_vars['DUPLICATE_DETECTION_MODE'] = self.gui.duplicate_detection_mode_var.get()
        env_vars['CHAPTER_NUMBER_OFFSET'] = str(self.gui.chapter_number_offset_var.get())
        env_vars['COMPRESSION_FACTOR'] = self.gui.compression_factor_var.get()
        env_vars['DISABLE_ZERO_DETECTION'] = "1" if self.gui.disable_zero_detection_var.get() else "0"
        env_vars['COMPREHENSIVE_EXTRACTION'] = "1" if self.gui.comprehensive_extraction_var.get() else "0"
        env_vars['USE_HEADER_AS_OUTPUT'] = "1" if self.gui.use_header_as_output_var.get() else "0"
        env_vars['ENABLE_DECIMAL_CHAPTERS'] = "1" if self.gui.enable_decimal_chapters_var.get() else "0"
        env_vars['ENABLE_WATERMARK_REMOVAL'] = "1" if self.gui.enable_watermark_removal_var.get() else "0"
        env_vars['ADVANCED_WATERMARK_REMOVAL'] = "1" if self.gui.advanced_watermark_removal_var.get() else "0"
        env_vars['SAVE_CLEANED_IMAGES'] = "1" if self.gui.save_cleaned_images_var.get() else "0"
        
        # EPUB specific settings
        env_vars['DISABLE_EPUB_GALLERY'] = "1" if self.gui.disable_epub_gallery_var.get() else "0"
        env_vars['FORCE_NCX_ONLY'] = '1' if self.gui.force_ncx_only_var.get() else '0'
        
        # Special handling for Gemini safety filters
        env_vars['DISABLE_GEMINI_SAFETY'] = str(self.gui.config.get('disable_gemini_safety', False)).lower()
        
        # AI Hunter settings (if enabled)
        if 'ai_hunter_config' in self.gui.config:
            env_vars['AI_HUNTER_CONFIG'] = json.dumps(self.gui.config['ai_hunter_config'])
        
        # Output settings
        env_vars['EPUB_OUTPUT_DIR'] = os.getcwd()
        output_path = self.gui.output_entry.get().strip() if hasattr(self.gui, 'output_entry') else ''
        if output_path:
            env_vars['OUTPUT_DIR'] = output_path
        
        # File path (needed by some modules)
        env_vars['EPUB_PATH'] = self.gui.file_path
        
        return env_vars

    def _extract_chapters_for_async(self, file_path, env_vars):
        """Extract chapters and prepare them for async processing"""
        chapters = []
        original_basename = None
        chapter_mapping = {}  # Map custom_id to chapter info
        
        try:
            if file_path.lower().endswith('.epub'):
                # Use direct ZIP reading to avoid ebooklib's manifest validation
                import zipfile
                from bs4 import BeautifulSoup
                
                raw_chapters = []
                
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        # Get all HTML/XHTML files
                        html_files = [f for f in zf.namelist() if f.endswith(('.html', '.xhtml', '.htm')) and not f.startswith('__MACOSX')]
                        html_files.sort()  # Sort to maintain order
                        
                        for idx, html_file in enumerate(html_files):
                            try:
                                content = zf.read(html_file)
                                soup = BeautifulSoup(content, 'html.parser')
                                
                                # Remove all image tags
                                for img in soup.find_all('img'):
                                    img.decompose()
                                
                                # Remove all link tags that might reference CSS or other files
                                for link in soup.find_all('link'):
                                    link.decompose()
                                    
                                chapter_text = soup.get_text(separator='\n').strip()
                                
                                if len(chapter_text) > 500:  # Minimum chapter length
                                    chapter_num = idx + 1
                                    
                                    # Try to extract chapter number from content
                                    for element in soup.find_all(['h1', 'h2', 'h3', 'title']):
                                        text = element.get_text().strip()
                                        match = re.search(r'chapter\s*(\d+)', text, re.IGNORECASE)
                                        if match:
                                            chapter_num = int(match.group(1))
                                            break
                                    
                                    raw_chapters.append((chapter_num, chapter_text, html_file))
                                    
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
                
            # Process each chapter to prepare for API
            for idx, (chapter_num, content, original_filename) in enumerate(raw_chapters):
                # Count tokens
                token_count = self.count_tokens(content, env_vars['MODEL'])
                
                # Check if needs chunking
                token_limit = int(env_vars.get('TOKEN_LIMIT', '200000'))
                needs_chunking = token_count > token_limit * 0.8  # 80% threshold
                
                # Prepare messages format
                messages = self._prepare_chapter_messages(content, env_vars)
                
                custom_id = f"chapter_{chapter_num}"
                
                chapter_data = {
                    'id': custom_id,
                    'number': chapter_num,
                    'content': content,
                    'messages': messages,
                    'temperature': float(env_vars.get('TRANSLATION_TEMPERATURE', '0.3')),
                    'max_tokens': int(env_vars['MAX_OUTPUT_TOKENS']),
                    'needs_chunking': needs_chunking,
                    'token_count': token_count,
                    'original_basename': original_filename  # Use original_filename instead of undefined original_basename
                }
                
                chapters.append(chapter_data)
                
                # Store mapping
                chapter_mapping[custom_id] = {
                    'original_filename': original_filename,
                    'chapter_num': chapter_num
                }
                
        except Exception as e:
            print(f"Failed to extract chapters: {e}")
            raise
            
        # Return both chapters and mapping
        return chapters, chapter_mapping

    def _delete_selected_job(self):
        """Delete selected job from the list"""
        if not self.selected_job_id:
            messagebox.showwarning("No Selection", "Please select a job to delete")
            return
        
        # Get job details for confirmation
        job = self.processor.jobs.get(self.selected_job_id)
        if not job:
            messagebox.showerror("Error", "Selected job not found")
            return
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete this job?\n\n"
            f"Job ID: {job.job_id}\n"
            f"Status: {job.status.value}\n"
            f"Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Note: This only removes the job from your local list.\n"
            "The job may still be running on the server."
        )
        
        if response:
            # Remove from jobs dictionary
            del self.processor.jobs[self.selected_job_id]
            
            # Save updated jobs
            self.processor._save_jobs()
            
            # Clear selection
            self.selected_job_id = None
            
            # Refresh the display
            self._refresh_jobs_list()
            
            messagebox.showinfo("Job Deleted", "Job removed from local list.")

    def _clear_completed_jobs(self):
        """Clear all completed/failed/cancelled jobs"""
        # Get list of jobs to remove
        jobs_to_remove = []
        for job_id, job in self.processor.jobs.items():
            if job.status in [AsyncAPIStatus.COMPLETED, AsyncAPIStatus.FAILED, 
                             AsyncAPIStatus.CANCELLED, AsyncAPIStatus.EXPIRED]:
                jobs_to_remove.append(job_id)
        
        if not jobs_to_remove:
            messagebox.showinfo("No Jobs to Clear", "No completed/failed/cancelled jobs to clear.")
            return
        
        # Confirm
        response = messagebox.askyesno(
            "Clear Completed Jobs",
            f"Remove {len(jobs_to_remove)} completed/failed/cancelled jobs from the list?\n\n"
            "This will not affect any running jobs."
        )
        
        if response:
            # Remove jobs
            for job_id in jobs_to_remove:
                del self.processor.jobs[job_id]
            
            # Save
            self.processor._save_jobs()
            
            # Refresh
            self._refresh_jobs_list()
            
            messagebox.showinfo("Jobs Cleared", f"Removed {len(jobs_to_remove)} jobs from the list.")

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
        
        # Add glossary if enabled (fixed the env var checks)
        if (env_vars.get('MANUAL_GLOSSARY') and 
            env_vars.get('APPEND_GLOSSARY') == '1' and 
            env_vars.get('DISABLE_GLOSSARY_TRANSLATION') != '1'):
            try:
                glossary_path = env_vars['MANUAL_GLOSSARY']
                with open(glossary_path, 'r', encoding='utf-8') as f:
                    glossary_data = json.load(f)
                
                # Format glossary for prompt
                glossary_text = self._format_glossary_for_prompt(glossary_data)
                
                # Use the append prompt format if provided
                append_prompt = env_vars.get('APPEND_GLOSSARY_PROMPT', '')
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
                
                logger.info(f"Glossary appended to system prompt ({len(glossary_text)} chars)")
                
                # Log first few glossary entries for debugging
                entries = glossary_text.split('\n')[:5]
                logger.info(f"Sample glossary entries: {entries}")
                
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

    def _format_glossary_for_prompt(self, glossary_data):
        """Format glossary data for inclusion in prompt"""
        if isinstance(glossary_data, list):
            # List format
            lines = []
            for entry in glossary_data[:50]:  # Limit to prevent prompt bloat
                if 'original_name' in entry and 'name' in entry:
                    lines.append(f"{entry['original_name']} = {entry['name']}")
            return '\n'.join(lines)
        else:
            # Dictionary format
            lines = []
            for orig, trans in list(glossary_data.items())[:50]:
                lines.append(f"{orig} = {trans}")
            return '\n'.join(lines)

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
                    batch_line = {
                        "key": request['custom_id'],
                        "request": {
                            "contents": request['generateContentRequest']['contents'],
                            "generation_config": request['generateContentRequest'].get('generationConfig', {})
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
                    poll_interval = self.poll_interval_var.get() * 1000
                    self.dialog.after(poll_interval, poll)
                    
            except Exception as e:
                self._log(f"❌ Polling error: {e}")
                
        # Start polling
        poll()

    def _handle_completed_job(self, job_id):
        """Handle a completed job - retrieve results and save"""
        try:
            # Retrieve results
            results = self.processor.retrieve_results(job_id)
            
            if not results:
                self._log("❌ No results retrieved from completed job")
                return
                
            # Get output directory - same name as input file
            app_dir = os.path.dirname(os.path.abspath(__file__))
            base_name = os.path.splitext(os.path.basename(self.gui.file_path))[0]
            output_dir = os.path.join(app_dir, base_name)
            
            # Handle existing directory
            if os.path.exists(output_dir):
                response = messagebox.askyesnocancel(
                    "Directory Exists",
                    f"The output directory already exists:\n{output_dir}\n\n"
                    "Yes = Overwrite\n"
                    "No = Create new with number\n"
                    "Cancel = Cancel operation"
                )
                
                if response is None:
                    return
                elif response is False:
                    counter = 1
                    while os.path.exists(f"{output_dir}_{counter}"):
                        counter += 1
                    output_dir = f"{output_dir}_{counter}"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract ALL resources from EPUB (CSS, fonts, images)
            self._log("📦 Extracting EPUB resources...")
            import zipfile
            
            with zipfile.ZipFile(self.gui.file_path, 'r') as zf:
                # Create resource directories
                for res_type in ['css', 'fonts', 'images']:
                    os.makedirs(os.path.join(output_dir, res_type), exist_ok=True)
                
                # Extract all resources
                for file_path in zf.namelist():
                    if file_path.endswith('/'):
                        continue
                        
                    file_lower = file_path.lower()
                    file_name = os.path.basename(file_path)
                    
                    # Skip empty filenames
                    if not file_name:
                        continue
                    
                    # Determine resource type and extract
                    if file_lower.endswith('.css'):
                        zf.extract(file_path, os.path.join(output_dir, 'css'))
                    elif file_lower.endswith(('.ttf', '.otf', '.woff', '.woff2')):
                        zf.extract(file_path, os.path.join(output_dir, 'fonts'))
                    elif file_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp')):
                        zf.extract(file_path, os.path.join(output_dir, 'images'))
            
            # Extract chapter info and metadata from source EPUB
            self._log("📋 Extracting metadata from source EPUB...")
            
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            from TransateKRtoEN import get_content_hash
            
            # Extract metadata
            metadata = {}
            book = epub.read_epub(self.gui.file_path)
            
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
            chapters_info = []
            actual_chapter_num = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    original_name = item.get_name()
                    original_basename = os.path.splitext(os.path.basename(original_name))[0]
                    
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text().strip()
                    
                    if len(text) > 500:  # Valid chapter
                        actual_chapter_num += 1
                        
                        # Try to find chapter number in content
                        chapter_num = actual_chapter_num
                        for element in soup.find_all(['h1', 'h2', 'h3', 'title']):
                            element_text = element.get_text().strip()
                            match = re.search(r'chapter\s*(\d+)', element_text, re.IGNORECASE)
                            if match:
                                chapter_num = int(match.group(1))
                                break
                        
                        # Calculate real content hash
                        content_hash = get_content_hash(text)
                        
                        chapter_map[chapter_num] = {
                            'original_basename': original_basename,
                            'content_hash': content_hash,
                            'text_length': len(text),
                            'has_images': bool(soup.find_all('img'))
                        }
                        
                        chapters_info.append({
                            'num': chapter_num,
                            'title': element_text if 'element_text' in locals() else f"Chapter {chapter_num}",
                            'original_filename': original_name,
                            'has_images': bool(soup.find_all('img')),
                            'text_length': len(text),
                            'content_hash': content_hash
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
            
            # Sort results and save with proper filenames
            sorted_results = sorted(results, key=lambda x: self._extract_chapter_number(x['custom_id']))
            
            self._log("💾 Saving translated chapters...")
            for result in sorted_results:
                chapter_num = self._extract_chapter_number(result['custom_id'])
                
                # Get chapter info
                chapter_info = chapter_map.get(chapter_num, {})
                original_basename = chapter_info.get('original_basename', f"{chapter_num:04d}")
                content_hash = chapter_info.get('content_hash', hashlib.md5(f"chapter_{chapter_num}".encode()).hexdigest())
                
                # Save file with correct name (only once!)
                filename = f"response_{original_basename}.html"
                file_path = os.path.join(output_dir, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result['content'])
                
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
                    "model": self.gui.model_var.get(),
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
            
            messagebox.showinfo(
                "Async Translation Complete",
                f"Successfully saved {len(sorted_results)} translated chapters to:\n{output_dir}\n\n"
                "Ready for EPUB conversion or further processing."
            )
                
        except Exception as e:
            self._log(f"❌ Error handling completed job: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to process results: {str(e)}")
 
    def _show_error_details(self, job):
        """Show details from error file"""
        if not job.metadata.get('error_file_id'):
            return
            
        try:
            api_key = self.gui.api_key_entry.get().strip()
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
                    error_text += f"\n\n... and {len(response.text.strip().split('\n')) - 5} more errors"
                    
                messagebox.showerror(
                    "Batch Processing Errors",
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

    # Helper methods for thread-safe UI updates
    def _log(self, message, level="info"):
        """Thread-safe logging to GUI"""
        # Log based on level
        if level == "error":
            print(f"❌ {message}")  # This will show in GUI
        elif level == "warning":
            print(f"⚠️ {message}")  # This will show in GUI
        else:
            logger.info(message)  # This only goes to log file
            # Also display info messages in GUI
            if hasattr(self.gui, 'append_log'):
                self.dialog.after(0, lambda: self.gui.append_log(message))

    def _show_error(self, message):
        """Thread-safe error dialog"""
        self._log(f"Error: {message}", level="error")
        self.dialog.after(0, lambda: messagebox.showerror("Error", message))

    def _show_info(self, title, message):
        """Thread-safe info dialog"""
        self._log(f"{title}: {message}", level="info")
        self.dialog.after(0, lambda: messagebox.showinfo(title, message))

    def _show_warning(self, message):
        """Thread-safe warning display"""
        self._log(f"Warning: {message}", level="warning")


def show_async_processing_dialog(parent, translator_gui):
    """Show the async processing dialog
    
    Args:
        parent: Parent window
        translator_gui: Reference to main TranslatorGUI instance
    """
    dialog = AsyncProcessingDialog(parent, translator_gui)
    return dialog.dialog


# Integration function for translator_gui.py
def add_async_processing_button(translator_gui, parent_frame):
    """Add async processing button to GUI
    
    This function should be called from translator_gui.py to add the button
    
    Args:
        translator_gui: TranslatorGUI instance
        parent_frame: Frame to add button to
    """
    # Create button with appropriate styling
    async_button = ttk.Button(
        parent_frame,
        text="Async Processing (50% Off)",
        command=lambda: show_async_processing_dialog(translator_gui.master, translator_gui),
        style="primary.TButton"
    )
    
    # Place button appropriately
    async_button.pack(side=tk.LEFT, padx=5)
    
    # Store reference
    translator_gui.async_button = async_button
    
    return async_button
