# model_options.py
"""
Centralized model catalog for Glossarion UIs.
Returned list should mirror the main GUI model dropdown.
Updated: 2026-03-10
"""
import concurrent.futures
import json
import os
import platform
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

def _get_static_model_options() -> List[str]:
    return [
    
        # OpenAI Models (as of March 2026)
        # - GPT-4o/4o-mini/4-turbo/4.1/3.5-turbo retired from ChatGPT Feb 13 2026; 4o still on API but legacy
        # - GPT-5.1 retiring March 11 2026, removed
        # - GPT-5.4 released March 5 2026, GPT-5.3 Instant released March 3 2026
        "gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-5.5","gpt-5.4", "gpt-5.4-pro",
        "gpt-5.3-codex", "gpt-5.3-codex-spark",
        "gpt-5.2", "gpt-5.2-pro", "gpt-5.2-chat-latest",
        "gpt-5-mini","gpt-5","gpt-5-nano", "gpt-5-chat-latest", "gpt-5-codex", "gpt-5-pro", "gpt-5-pro-2025-10-06",
        "gpt-4.1-nano",
        "gpt-4o-mini",  # Still on API, legacy
        "o3", "chatgpt-image-latest",
        
        # Google Gemini Models (as of March 2026)
        # - gemini-3-pro-preview shut down March 9 2026, removed
        # - gemini-pro / gemini-pro-vision are legacy 1.0 models, removed
        # - gemini-2.0-flash/lite scheduled shutdown June 1 2026, still available
        "gemini-3.6-flash","gemini-3.5-flash","gemini-3-flash-preview",
        "gemini-3.1-pro-preview","gemini-3.5-flash-lite","gemini-3.1-flash-lite",
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro",
        "gemini-2.0-flash","gemini-2.0-flash-lite",
        # Gemma models (served via the Gemini API endpoint)
        "gemma-4-31b-it", "gemma-3-27b-it", "gemma-3-12b-it",
        "gemma-3-4b-it", "gemma-3-1b-it", "gemma-3n-e4b-it", "gemma-3n-e2b-it",
        "gemma-2-27b-it", "gemma-2-9b-it", "gemma-2-2b-it",
        
        # Anthropic Claude Models
        "claude-fable-5","claude-opus-4-8","claude-opus-4-7","claude-opus-4-6", "claude-opus-4-5-20251101", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-6", 
        "claude-sonnet-5","claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
        "claude-3-haiku-20240307",       
        
        # Grok Models
        "grok-4.5","grok-4.3","grok-4.20-beta-0309-reasoning","grok-4.20-beta-0309-non-reasoning", "grok-4.20-multi-agent-beta-0309",
        "grok-4.20-multi-agent-experimental-beta-0304","grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning","grok-4-0709", "grok-4-fast",
        "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",  "grok-4-fast-reasoning-latest", "grok-3", "grok-3-mini",        
        
        # Alternative format with vertex_ai prefix
        "vertex/claude-3-7-sonnet@20250219",
        "vertex/claude-3-5-sonnet@20240620",
        "vertex/claude-3-opus@20240229",
        "vertex/claude-4-opus@20250514",
        "vertex/claude-4-sonnet@20250514",
        "vertex/gemini-2.0-flash",
        "vertex/gemini-2.5-pro",
        "vertex/gemini-2.5-flash",
        "vertex/gemini-2.5-flash-lite",
        "vertex/gemini-3.1-pro-preview",
        "vertex/gemini-3.1-flash-lite",
        "vertex/gemini-3-flash-preview",
        "vertex/gemini-3.1-flash-image-preview",
        "vertex/gemini-3-pro-image-preview",

        # Chute AI
        "chutes/openai/gpt-oss-120b",
        "chutes/deepseek-ai/DeepSeek-V3.2",
        "chutes/deepseek-ai/DeepSeek-V3.2-TEE",
        "chutes/deepseek-ai/DeepSeek-V3.1",
        "chutes/deepseek-ai/DeepSeek-V3-0324",
        "chutes/deepseek-ai/DeepSeek-V3",
        "chutes/deepseek-ai/DeepSeek-R1-0528",
        "chutes/moonshotai/Kimi-K2-Thinking",
        "chutes/zai-org/GLM-4.6-TEE", "chutes/zai-org/GLM-4.7-TEE",
        "chutes/moonshotai/Kimi-K2.6-TEE",
        
        # DeepSeek Models (api.deepseek.com)
        "deepseek-v4-flash", "deepseek-v4-pro",
        "deepseek-chat","deepseek-reasoner", "deepseek-coder", "deepseek-coder-33b-instruct",
        
        # Mistral Models
        "codestral-2508", "codestral-embed",
        "devstral-2512", "devstral-medium-2507", "devstral-small-2507",
        "labs-leanstral-2603",
        "magistral-medium-2509", "magistral-small-2509",
        "ministral-14b-2512", "ministral-3b-2512", "ministral-8b-2512",
        "mistral-embed-2312",
        "mistral-large-2411", "mistral-large-2512",
        "mistral-medium-2505", "mistral-medium-2508", "mistral-medium-3-5",
        "mistral-moderation-2411", "mistral-moderation-2603",
        "mistral-ocr-latest",
        "mistral-ocr-2505", "mistral-ocr-2512",
        "mistral-small-2506", "mistral-small-2603",
        "open-mistral-nemo",
        "pixtral-large-2411",
        "voxtral-mini-2507", "voxtral-mini-2602",
        "voxtral-mini-transcribe-2507", "voxtral-mini-transcribe-realtime-2602",
        "voxtral-mini-tts-2603", "voxtral-small-2507",
        
        # Meta Llama Models (via Together/other providers)
        "llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat",
        "llama-3-8b-instruct", "llama-3-70b-instruct", "codellama-34b-instruct",
        
        # Yi Models
        "yi-34b-chat", "yi-34b-chat-200k", "yi-6b-chat",
        
        # Qwen Models
        "qwen-72b-chat", "qwen-14b-chat", "qwen-7b-chat", "qwen-plus", "qwen-turbo",
        
        # Cohere Models
        "command", "command-light", "command-nightly", "command-r", "command-r-plus",
        
        # AI21 Models
        "j2-ultra", "j2-mid", "j2-light", "jamba-instruct",
        
        # Perplexity Models
        "perplexity-70b-online", "perplexity-7b-online", "pplx-70b-online", "pplx-7b-online",
        
        # Groq Models
        "groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct", "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/meta-llama/llama-prompt-guard-2-22m", "groq/meta-llama/llama-prompt-guard-2-86m",
        "groq/meta-llama/llama-guard-4-12b",
        "groq/moonshotai/kimi-k2-instruct-0905",
        "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-safeguard-20b",
        "groq/qwen/qwen3-32b",
        "groq/playai-tts", "groq/playai-tts-arabic",
        "groq/whisper-large-v3", "groq/whisper-large-v3-turbo",
        "groq/groq/compound", "groq/groq/compound-mini",
        
        # Chinese Models
        "chatglm-6b", "chatglm2-6b", "chatglm3-6b",
        "baichuan-13b-chat", "baichuan2-13b-chat",
        "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
        
        # GLM Models
        # Text / Reasoning
        "GLM-5.1", "GLM-5", "GLM-4.7",
        "GLM-4.7-Flash", "GLM-4.6", "GLM-4.5", "GLM-4.5-Flash",

        # Vision / Multimodal
        "GLM-5V-Turbo",
        "GLM-4.6V", "GLM-4.6V-Flash", "GLM-4.6V-FlashX",
        "GLM-4.5V",

        # Specialized Variants
        "GLM-4.1V-Thinking", "GLM-4-Voice", "GLM-4-Plus",
        "glm-4", "glm-3-turbo", 

        # Text / Reasoning (za prefix)
        "za/GLM-5.1", "za/GLM-5", "za/GLM-4.7",
        "za/GLM-4.7-Flash", "za/GLM-4.6", "za/GLM-4.5", "za/GLM-4.5-Flash",

        # Vision / Multimodal (za prefix)
        "za/GLM-5V-Turbo",
        "za/GLM-4.6V", "za/GLM-4.6V-Flash", "za/GLM-4.6V-FlashX",
        "za/GLM-4.5V",

        # Specialized Variants (za prefix)
        "za/GLM-4.1V-Thinking", "za/GLM-4-Voice", "za/GLM-4-Plus",
        "za/glm-4", "za/glm-3-turbo", 
        
        # Other Models
        "falcon-40b-instruct", "falcon-7b-instruct",
        "phi-2", "phi-3-mini", "phi-3-small", "phi-3-medium",
        "orca-2-13b", "orca-2-7b",
        "vicuna-13b", "vicuna-7b",
        "alpaca-7b",
        "wizardlm-70b", "wizardlm-13b",
        "openchat-3.5",
        
        # For POE, prefix with 'poe/' (NO LONGER WORKS)
       # "poe/gpt-4", "poe/gpt-4o", "poe/gpt-4.5", "poe/gpt-4.1",
       # "poe/claude-3-opus", "poe/claude-4-opus", "poe/claude-3-sonnet", "poe/claude-4-sonnet",
       # "poe/claude", "poe/Assistant",
       # "poe/gemini-2.5-flash", "poe/gemini-2.5-pro",

        # LiteRouter (lr/ prefix)
        "lr/deepseek-v3.2:free", "lr/deepseek-chat", "lr/deepseek-r1",
        "lr/deepseek-r1-0528", "lr/deepseek-reasoner", "lr/deepseek-reasoner-official",
        "lr/deepseek-v3", "lr/deepseek-v3-0324", "lr/deepseek-v3-0324-fp8",
        "lr/deepseek-v3-fp8", "lr/deepseek-v3.1", "lr/deepseek-v3.1-fp8",
        "lr/deepseek-v3.1-nex-n1", "lr/deepseek-v3.1-terminus",
        "lr/deepseek-v3.1-terminus-fp8", "lr/deepseek-v3.2",
        "lr/deepseek-v3.2-exp", "lr/deepseek-v3.2-fp8", "lr/deepseek-v3.2-official",
        "lr/deepseek-v4-flash", "lr/deepseek-v4-flash-official",
        "lr/deepseek-v4-flash-thinking", "lr/deepseek-v4-flash-thinking-official",
        "lr/devstral-small-2507:free",
        "lr/gemini-2.0-flash-lite-001:free", "lr/gemini-2.5-flash",
        "lr/gemini-2.5-flash-lite", "lr/gemini-2.5-flash-thinking",
        "lr/gemini-3-flash-preview", "lr/gemini-3-flash-preview-thinking",
        "lr/gemini-3.1-flash-lite-preview", "lr/gemini-3.1-flash-lite-preview-thinking",
        "lr/gemma-3-27b-it:free", "lr/gemma-3-27b-it",
        "lr/gemma-4-26b-a4b", "lr/gemma-4-31b", "lr/gemma-4-31b-non-reasoning",
        "lr/glm-4-32b:free", "lr/gpt-3.5-turbo", "lr/gpt-4.1",
        "lr/gpt-4.1-mini", "lr/gpt-4.1-nano", "lr/gpt-4o-mini",
        "lr/gpt-4o-mini-search-preview", "lr/gpt-5-mini", "lr/gpt-5-nano",
        "lr/gpt-5.4-mini", "lr/gpt-5.4-nano", "lr/gpt-oss-120b:free",
        "lr/gpt-oss-120b", "lr/gpt-oss-20b:free", "lr/gpt-oss-20b",
        "lr/grok-4.1-fast-reasoning:free", "lr/kimi-k2.6",
        "lr/l3-8b-lunaris:free", "lr/llama-3-8b-instruct:free",
        "lr/llama-3.1-8b-instruct-turbo:free", "lr/llama-3.1-8b-instruct:free",
        "lr/llama-3.2-3b-instruct:free", "lr/llama-3.3-70b-instruct-turbo:free",
        "lr/mimo-v2-flash:free", "lr/ministral-3b-2512:free",
        "lr/mistral-large-3", "lr/mistral-nemo-instruct-2407:free",
        "lr/mistral-small-24b-instruct-2501:free",
        "lr/mythomax-l2-13b:free", "lr/nemotron-nano-9b-v2:free",
        "lr/openrouter:free:full-context", "lr/owl-alpha:free:full-context",
        "lr/pixtral-large-2411", "lr/pixtral-large-latest",
        "lr/qwen3-4b-fp8:free", "lr/trinity-large-thinking", "lr/trinity-mini:free",

        # OpenCode Go (oc/ prefix) - OpenAI-compatible /chat/completions models
        "oc/glm-5.1", "oc/glm-5",
        "oc/kimi-k2.6", "oc/kimi-k2.5",
        "oc/deepseek-v4-pro", "oc/deepseek-v4-flash",
        "oc/mimo-v2.5-pro", "oc/mimo-v2.5",
        
        # For OR, prefix with 'or/'
        "or/openrouter/free",
        "or/anthropic/claude-sonnet-4.6","or/anthropic/claude-sonnet-4.5", "or/anthropic/claude-sonnet-4",
        "or/google/gemini-3.1-flash-lite-preview", "or/google/gemini-3-flash-preview", "or/google/gemini-3.1-pro-preview",
        "or/openai/gpt-5.4","or/openai/gpt-5.4-mini", "or/openai/gpt-5.4-nano",
        "or/google/gemini-2.5-pro",
        "or/google/gemini-2.5-flash",
        "or/google/gemini-2.5-flash-preview-09-2025",
        "or/google/gemini-2.5-flash-lite",
        "or/openai/gpt-oss-20b:free","or/openai/gpt-5",
        "or/openai/gpt-5-mini",
        "or/openai/gpt-5-nano",
        "or/openai/chatgpt-4o-latest", "or/deepseek/deepseek-chat-v3-0324:free",
        "or/deepseek/deepseek-v4-flash","or/deepseek/deepseek-v4-pro",
        "or/deepseek/deepseek-r1-distill-llama-70b:free", "or/deepseek/deepseek-v3.2", "or/deepseek/deepseek-v3.2-speciale",
        "or/deepseek/deepseek-chat-v3.1", "or/deepseek/deepseek-r1-0528", "or/deepseek/deepseek-r1", "or/deepseek/deepseek-chat",
        "or/deepseek/deepseek-r1:free","or/deepseek/deepseek-r1-0528:free", "or/deepseek/deepseek-chat-v3.1:free", "or/deepseek/deepseek-r1-0528-qwen3-8b:free",
        "or/tngtech/deepseek-r1t2-chimera:free","or/tngtech/deepseek-r1t-chimera:free", "or/deepseek/deepseek-v4-flash-0731",
        "or/google/gemma-3-27b-it:free", "or/google/gemma-3-27b-it",
        "or/qwen/qwen3-235b-a22b", "or/qwen/qwen3-235b-a22b-thinking-2507", 
        
        # For ElectronHub, prefix with 'eh/'
        "eh/claude-sonnet-4-6","eh/claude-sonnet-4-6-thinking","eh/claude-opus-4-6","eh/claude-opus-4-6-thinking",
        "eh/claude-sonnet-4-5-20250929", "eh/claude-sonnet-4-5-20250929-thinking",        
        "eh/claude-sonnet-4-20250514:aws-bedrock", "eh/claude-sonnet-4-20250514-thinking:aws-bedrock",
        "eh/claude-sonnet-4-20250514", "eh/claude-sonnet-4-20250514-thinking",       
        "eh/claude-opus-4-1-20250805-thinking","eh/claude-opus-4-1-20250805:aws-bedrock",
        "eh/claude-opus-4-1-20250805", "eh/claude-opus-4-20250514:aws-bedrock",
        "eh/gpt-5-chat-latest:free","eh/gpt-5-high","eh/gpt-5-low","eh/gpt-5-chat-latest","eh/gpt-5-minimal", 
        "eh/gpt-5-mini:free","eh/gpt-5-mini-minimal","eh/gpt-5-nano:free","eh/gpt-4o",
        "eh/gpt-4", "eh/gpt-3.5-turbo", "eh/claude-3-opus", "eh/claude-3-sonnet",
        "eh/gemini-2.5-flash","eh/gemini-2.5-flash-thinking", "eh/gemini-2.5-flash-preview-05-20",
        "eh/gemini-2.5-flash-preview-05-20-thinking","eh/gemini-2.5-flash-preview-09-2025",
        "eh/gemini-2.5-flash-lite", "eh/gemini-2.5-pro","eh/gemini-2.5-pro-thinking",
        "eh/gemini-2.5-pro-preview-06-05","eh/gemini-2.5-pro-preview-05-06", "eh/gemini-2.5-pro-preview-03-25",
        "eh/gemini-3.1-pro-preview", "eh/gemini-3.1-pro-preview-medium", "eh/gemini-3.1-pro-preview-low",
        "eh/gemini-2.0-flash-001","eh/gemini-2.0-flash-exp","eh/gemini-2.0-flash-thinking-exp", "eh/grok-4-fast",
        "eh/grok-4-0709", "eh/grok-3", "eh/grok-3-mini-fast", "eh/grok-3-mini", "eh/grok-3-fast",
        "eh/grok-code-fast-1","eh/llama-2-70b-chat", "eh/yi-34b-chat-200k", "eh/mistral-large", "eh/deepseek-v3-0324:free",
        "eh/deepseek-v3.1:free", "eh/deepseek-v3.1", "eh/deepseek-v3.2-exp:free", "eh/deepseek-v3.2-exp" , "eh/deepseek-v3.2-exp-thinking" ,
        "eh/gemini-pro", "eh/deepseek-coder-33b", "eh/gemma-3-27b-it", "eh/glm-4.6", "eh/glm-4.7",

        # AuthGPT – ChatGPT subscription via OAuth (Codex Responses endpoint)
        # Only models supported by /backend-api/codex/responses are listed.
        # GPT-5.1 retiring March 11 2026, removed. GPT-5.4 released March 5 2026.
        "authgpt/gpt-5.6-luna","authgpt/gpt-5.6-sol","authgpt/gpt-5.6-terra","authgpt/gpt-5.5","authgpt/gpt-5.4", "authgpt/gpt-5.4-pro",
        "authgpt/gpt-5.3-codex", "authgpt/gpt-5.3-codex-spark",
        "authgpt/gpt-5.2", "authgpt/gpt-5.2-codex",

        # AuthGrok – xAI account via OAuth (catalog is account-specific)
        "authgrok/grok-4.5", "authgrok/grok-4.3", "authgrok/grok-build",
        "authgrok/grok-composer-2.5-fast",
        "authgrok/grok-4.20-0309-reasoning",
        "authgrok/grok-4.20-0309-non-reasoning",
        "authgrok/grok-4.20-multi-agent-0309",

        # AuthCD – Claude subscription via OAuth (Anthropic Messages API)
        "authcd/claude-sonnet-5", "authcd/claude-fable-5", "authcd/claude-sonnet-4-6", "authcd/claude-sonnet-4-5", "authcd/claude-opus-4-8",
        "authcd/claude-opus-4-7", "authcd/claude-opus-4-6", "authcd/claude-haiku-4-5",

        # AuthGem – Gemini-cli via Google OAuth (no API key needed)
        "authgem/gemini-2.5-flash", "authgem/gemini-2.5-flash-lite",
        "authgem/gemini-2.5-pro",
        "authgem/gemini-2.0-flash", "authgem/gemini-2.0-flash-lite",
        "authgem/gemini-3.1-pro-preview", "authgem/gemini-3.1-flash-lite", "authgem/gemini-3-flash-preview",

        # AuthGem-Key - Gemini AI Studio using the API-key route
        "authgem-key/gemini-2.5-flash", "authgem-key/gemini-2.5-flash-lite",
        "authgem-key/gemini-2.5-pro", "authgem-key/gemini-2.0-flash",
        "authgem-key/gemini-2.0-flash-lite", "authgem-key/gemini-3.1-pro-preview",
        "authgem-key/gemini-3.1-flash-lite", "authgem-key/gemini-3-flash-preview",

        # nano-gpt provider models
        "nan/deepseek/deepseek-v4-flash", "nan/deepseek/deepseek-v4-flash:thinking", "nan/deepseek/deepseek-v4-pro",
        "nan/deepseek/deepseek-v4-pro:thinking", "nan/TEE/kimi-k2.6", "nan/TEE/glm-5.1", "nan/TEE/glm-5.1-thinking",
        "nan/moonshotai/kimi-k2.6:thinking","nan/moonshotai/kimi-k2.6","nan/anthropic/claude-opus-4.7", "nan/anthropic/claude-opus-4.7:thinking",
        "nan/qwen-3.6-plus", "nan/google/gemini-pro-latest","nan/google/gemini-flash-latest", "nan/google/gemini-flash-lite-latest", "nan/anthropic/claude-haiku-latest",
        "nan/anthropic/claude-opus-latest", "nan/openai/gpt-latest", "nan/gpt-image-2","nan/step-image-edit-2","nan/kling-v3-4k","nan/bytedance-seedance-2-0", "nan/wan-2.6-image-edit","nan/wan-2.7-video", "nan/veo3-1-lite-video", "nan/grok-imagine-video-extend", "nan/grok-imagine-video-edit",
        
        # SambaNova Cloud (api.sambanova.ai)
        "sam/DeepSeek-R1-Distill-Llama-70B", "sam/DeepSeek-V3.1-cb", "sam/DeepSeek-V3.1", "sam/DeepSeek-V3.2",
        "sam/gemma-3-12b-it", "sam/gpt-oss-120b",
        "sam/Llama-4-Maverick-17B-128E-Instruct", "sam/Meta-Llama-3.3-70B-Instruct",
        "sam/MiniMax-M2.5",

        # AuthGem – Gemini-cli via Google OAuth (Uses Vertex AI)
        "authgem-vertex/gemini-2.5-flash", "authgem-vertex/gemini-2.5-flash-lite",
        "authgem-vertex/gemini-2.5-pro",
        "authgem-vertex/gemini-2.0-flash", "authgem-vertex/gemini-2.0-flash-lite",
        "authgem-vertex/gemini-3.1-pro-preview", "authgem-vertex/gemini-3.1-flash-lite", "authgem-vertex/gemini-3-flash-preview",

        # OpenCode + opencode-antigravity-auth (OAuth; no API key needed)
        "ocagy/gemini-3.1-pro-high",
        "ocagy/gemini-3.1-pro-low",
        "ocagy/gemini-3-pro-high",
        "ocagy/gemini-3-pro-low",
        "ocagy/gemini-3-flash-minimal",
        "ocagy/gemini-3-flash-low",
        "ocagy/gemini-3-flash-medium",
        "ocagy/gemini-3-flash-high",
        "ocagy/claude-sonnet-4-6",
        "ocagy/claude-opus-4-6-thinking-low",
        "ocagy/claude-opus-4-6-thinking-max",

        # Antigravity Cloud Code proxy (frieser/antigravity-proxy dashboard catalog)
        "antigravity/gemini-3-flash",
        "antigravity/gemini-3-flash-agent",
        "antigravity/gemini-3.1-flash-image",
        "antigravity/gemini-3.1-flash-lite",
        "antigravity/gemini-3.6-flash-low",
        "antigravity/gemini-3.6-flash-medium",
        "antigravity/gemini-3.6-flash-high",
        "antigravity/gemini-3.5-flash-extra-low",
        "antigravity/gemini-3.5-flash-low",
        "antigravity/gemini-3.1-pro-low",
        "antigravity/gemini-pro-agent",
        "antigravity/gemini-2.5-flash",
        "antigravity/gemini-2.5-flash-lite",
        "antigravity/gemini-2.5-flash-thinking",
        "antigravity/gemini-2.5-pro",
        "antigravity/claude-opus-4-6-thinking",
        "antigravity/claude-sonnet-4-6",
        "antigravity/gpt-oss-120b-medium",

        # Google Search / Gemini browser-backed route (no API key needed)
        "search/gemini",

        # NVIDIA Build browser-backed route (no API key needed) - chat-tagged catalog models
        "authnd/nvidia/nemotron-3-ultra-550b-a55b",
        "authnd/mistralai/mistral-medium-3.5-128b",
        "authnd/deepseek-ai/deepseek-v4-flash",
        "authnd/deepseek-ai/deepseek-v4-pro",
        "authnd/minimaxai/minimax-m2.7",
        "authnd/google/gemma-4-31b-it",
        "authnd/minimaxai/minimax-m2.5",
        "authnd/qwen/qwen3-coder-480b-a35b-instruct",
        "authnd/sarvamai/sarvam-m",
        "authnd/mistralai/magistral-small-2506",
        "authnd/moonshotai/kimi-k2.6",
        "authnd/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "authnd/mistralai/mistral-small-4-119b-2603",
        "authnd/nvidia/nemotron-3-super-120b-a12b",
        "authnd/qwen/qwen3.5-122b-a10b",
        "authnd/qwen/qwen3.5-397b-a17b",
        "authnd/stepfun-ai/step-3.5-flash",
        "authnd/nvidia/nemotron-3-nano-30b-a3b",
        "authnd/mistralai/mistral-large-3-675b-instruct-2512",
        "authnd/mistralai/ministral-14b-instruct-2512",
        "authnd/nvidia/nemotron-nano-12b-v2-vl",
        "authnd/qwen/qwen3-next-80b-a3b-instruct",
        "authnd/qwen/qwen3-next-80b-a3b-thinking",
        "authnd/bytedance/seed-oss-36b-instruct",
        "authnd/nvidia/nvidia-nemotron-nano-9b-v2",
        "authnd/openai/gpt-oss-120b",
        "authnd/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "authnd/google/gemma-3n-e4b-it",
        "authnd/google/gemma-3n-e2b-it",
        "authnd/mistralai/mistral-nemotron",
        "authnd/nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "authnd/mistralai/mistral-medium-3-instruct",
        "authnd/meta/llama-4-maverick-17b-128e-instruct",
        "authnd/nvidia/llama-3.3-nemotron-super-49b-v1",
        "authnd/nvidia/llama-3.1-nemotron-nano-8b-v1",
        "authnd/microsoft/phi-4-mini-instruct",
        "authnd/qwen/qwen2.5-coder-32b-instruct",
        "authnd/meta/llama-3.2-3b-instruct",
        "authnd/meta/llama-3.2-11b-vision-instruct",
        "authnd/meta/llama-3.2-90b-vision-instruct",
        "authnd/meta/llama-3.2-1b-instruct",
        "authnd/abacusai/dracarys-llama-3.1-70b-instruct",
        "authnd/nvidia/nemotron-mini-4b-instruct",
        "authnd/google/gemma-2-2b-it",
        "authnd/meta/llama-3.1-70b-instruct",
        "authnd/meta/llama-3.1-8b-instruct",
        "authnd/mistralai/mistral-7b-instruct-v0.3",
        "authnd/mistralai/mixtral-8x22b-instruct-v0.1",
        "authnd/mistralai/mixtral-8x7b-instruct-v0.1",
        "authnd/z-ai/glm-5.2",

        # NVIDIA Integrate (OpenAI-compatible) — models from UI dropdown
        "nd/mistralai/mistral-medium-3.5-128b",
        "nd/deepseek-ai/deepseek-v4-flash",
        "nd/deepseek-ai/deepseek-v4-pro",
        "nd/z-ai/glm-5.2",
        "nd/minimaxai/minimax-m2.7",
        "nd/google/gemma-4-31b-it",
        "nd/minimaxai/minimax-m2.5",
        "nd/qwen/qwen3-coder-480b-a35b-instruct",
        "nd/sarvamai/sarvam-m",
        "nd/mistralai/magistral-small-2506",
        "nd/moonshotai/kimi-k2.6",
        "nd/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nd/mistralai/mistral-small-4-119b-2603",
        "nd/nvidia/nemotron-3-super-120b-a12b",
        "nd/qwen/qwen3.5-122b-a10b",
        "nd/qwen/qwen3.5-397b-a17b",
        "nd/stepfun-ai/step-3.5-flash",
        "nd/nvidia/nemotron-3-nano-30b-a3b",
        "nd/mistralai/mistral-large-3-675b-instruct-2512",
        "nd/mistralai/ministral-14b-instruct-2512",
        "nd/nvidia/nemotron-nano-12b-v2-vl",
        "nd/qwen/qwen3-next-80b-a3b-instruct",
        "nd/qwen/qwen3-next-80b-a3b-thinking",
        "nd/bytedance/seed-oss-36b-instruct",
        "nd/nvidia/nvidia-nemotron-nano-9b-v2",
        "nd/openai/gpt-oss-120b",
        "nd/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nd/google/gemma-3n-e4b-it",
        "nd/google/gemma-3n-e2b-it",
        "nd/mistralai/mistral-nemotron",
        "nd/nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "nd/mistralai/mistral-medium-3-instruct",
        "nd/meta/llama-4-maverick-17b-128e-instruct",
        "nd/nvidia/llama-3.3-nemotron-super-49b-v1",
        "nd/nvidia/llama-3.1-nemotron-nano-8b-v1",
        "nd/microsoft/phi-4-mini-instruct",
        "nd/qwen/qwen2.5-coder-32b-instruct",
        "nd/meta/llama-3.2-3b-instruct",
        "nd/meta/llama-3.2-11b-vision-instruct",
        "nd/meta/llama-3.2-90b-vision-instruct",
        "nd/meta/llama-3.2-1b-instruct",
        "nd/abacusai/dracarys-llama-3.1-70b-instruct",
        "nd/nvidia/nemotron-mini-4b-instruct",
        "nd/google/gemma-2-2b-it",
        "nd/meta/llama-3.1-70b-instruct",
        "nd/meta/llama-3.1-8b-instruct",
        "nd/mistralai/mistral-7b-instruct-v0.3",
        "nd/mistralai/mixtral-8x22b-instruct-v0.1",
        "nd/mistralai/mixtral-8x7b-instruct-v0.1",
        "nd/deepseek-ai/deepseek-v3.2",
        "nd/deepseek-ai/deepseek-v3.1",
        "nd/deepseek-ai/deepseek-v3.1-terminus",
        "nd/moonshotai/kimi-k2-thinking",
        "nd/meta/llama-4-scout-17b-16e-instruct",
        "nd/meta/llama-3.3-70b-instruct",
        
        # Last Resort
        "deepl",  # Will use DeepL API
        "google-translate-free",  # Uses free web endpoint (no key)
        "google-translate",  # Will use Google Cloud Translate
    ]


@dataclass(frozen=True)
class ProviderCatalogSpec:
    """Description of a provider's read-only model catalog endpoint."""

    name: str
    prefix: str
    models_url: str
    api_key_envs: Tuple[str, ...] = ()
    public: bool = False
    auth_style: str = "bearer"
    base_url_env: str = ""
    models_path: str = "/models"
    response_keys: Tuple[str, ...] = ("data", "models")
    id_fields: Tuple[str, ...] = ("id", "model", "name", "slug")
    allowed_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelCatalogRefreshResult:
    """Result returned by a provider-catalog refresh."""

    models: List[str]
    provider_models: Dict[str, List[str]]
    statuses: Dict[str, str]
    requested_provider: Optional[str] = None


_MODEL_CATALOG_CACHE_VERSION = 1
_MODEL_CATALOG_CACHE_TTL_SECONDS = 24 * 60 * 60
_MODEL_CATALOG_LOCK = threading.RLock()
_MODEL_CATALOG_MEMORY_CACHE: Optional[dict] = None


# These routes expose a model-list operation compatible with the current
# Glossarion transport. AuthGrok and OcAgy are handled separately because their
# OAuth sessions use provider-specific catalog mechanisms.
PROVIDER_CATALOG_SPECS: Tuple[ProviderCatalogSpec, ...] = (
    ProviderCatalogSpec(
        "openrouter", "or/", "https://openrouter.ai/api/v1/models?output_modalities=text",
        ("OPENROUTER_API_KEY",), True,
    ),
    ProviderCatalogSpec(
        "openai", "", "https://api.openai.com/v1/models", ("OPENAI_API_KEY",),
        base_url_env="OPENAI_API_BASE",
    ),
    ProviderCatalogSpec(
        "anthropic", "", "https://api.anthropic.com/v1/models?limit=1000", ("ANTHROPIC_API_KEY",),
        auth_style="anthropic", base_url_env="ANTHROPIC_BASE_URL", models_path="/v1/models",
    ),
    ProviderCatalogSpec(
        "gemini", "", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"), auth_style="query",
    ),
    ProviderCatalogSpec(
        "authgem_key", "authgem-key/",
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"), auth_style="query",
    ),
    ProviderCatalogSpec(
        "mistral", "", "https://api.mistral.ai/v1/models", ("MISTRAL_API_KEY",),
    ),
    ProviderCatalogSpec(
        "deepseek", "", "https://api.deepseek.com/models", ("DEEPSEEK_API_KEY",),
        base_url_env="DEEPSEEK_API_URL",
    ),
    ProviderCatalogSpec(
        "xai", "", "https://api.x.ai/v1/models", ("XAI_API_KEY",),
        base_url_env="XAI_API_URL",
    ),
    ProviderCatalogSpec(
        "groq", "groq/", "https://api.groq.com/openai/v1/models", ("GROQ_API_KEY",),
        base_url_env="GROQ_API_URL",
    ),
    ProviderCatalogSpec(
        "chutes", "chutes/", "https://llm.chutes.ai/v1/models", ("CHUTES_API_KEY",),
        base_url_env="CHUTES_API_URL",
    ),
    ProviderCatalogSpec(
        "nvidia", "nd/", "https://integrate.api.nvidia.com/v1/models", ("NVIDIA_API_KEY",),
        base_url_env="NVIDIA_API_URL",
    ),
    ProviderCatalogSpec(
        "literouter", "lr/", "https://api.literouter.com/v1/models", ("LITEROUTER_API_KEY",),
        base_url_env="LITEROUTER_API_URL",
    ),
    ProviderCatalogSpec(
        "opencode", "oc/", "https://opencode.ai/zen/go/v1/models", ("OPENCODE_API_KEY",),
        base_url_env="OPENCODE_API_URL",
    ),
    ProviderCatalogSpec(
        "electronhub", "eh/", "https://api.electronhub.ai/v1/models", ("ELECTRONHUB_API_KEY",),
        base_url_env="ELECTRONHUB_API_URL",
    ),
    ProviderCatalogSpec(
        "nanogpt", "nan/", "https://nano-gpt.com/api/v1/models", ("NANOGPT_API_KEY",),
        base_url_env="NANOGPT_API_URL", models_path="/api/v1/models",
    ),
    ProviderCatalogSpec(
        "sambanova", "sam/", "https://api.sambanova.ai/v1/models", ("SAMBANOVA_API_KEY",),
        base_url_env="SAMBANOVA_API_URL",
    ),
    ProviderCatalogSpec(
        "together", "together/", "https://api.together.xyz/v1/models", ("TOGETHER_API_KEY",),
        base_url_env="TOGETHER_API_URL", allowed_types=("chat",),
    ),
    ProviderCatalogSpec(
        "zai", "za/", "https://api.z.ai/api/paas/v4/models", ("ZAI_API_KEY", "ZA_API_KEY"),
        base_url_env="ZA_API_URL",
    ),
    ProviderCatalogSpec(
        "zhipu", "", "https://open.bigmodel.cn/api/paas/v4/models", ("ZHIPU_API_KEY",),
        base_url_env="ZHIPU_API_URL",
    ),
    ProviderCatalogSpec(
        "fireworks", "fireworks/", "https://api.fireworks.ai/inference/v1/models",
        ("FIREWORKS_API_KEY",), base_url_env="FIREWORKS_API_URL",
    ),
    ProviderCatalogSpec(
        "cohere", "cohere/", "https://api.cohere.com/v1/models?page_size=1000", ("COHERE_API_KEY",),
        response_keys=("models", "data"),
    ),
    ProviderCatalogSpec(
        "moonshot", "", "https://api.moonshot.cn/v1/models", ("MOONSHOT_API_KEY",),
        base_url_env="MOONSHOT_API_URL",
    ),
    # The local proxy is queried only if it is already running. Catalog
    # discovery must never start the proxy or open an OAuth browser window.
    ProviderCatalogSpec(
        "antigravity", "antigravity/", "http://localhost:3000/v1/models",
        public=True, base_url_env="ANTIGRAVITY_PROXY_URL", models_path="/v1/models",
    ),
)


STATIC_ONLY_PROVIDER_PREFIXES: Mapping[str, str] = {
    "authgpt/": "OAuth backend does not expose a stable general catalog endpoint",
    "authcd/": "OAuth backend does not expose a stable general catalog endpoint",
    "authgem/": "OAuth account and project context are required",
    "authgem-vertex/": "Vertex catalogs are project and region specific",
    "authnd/": "Browser-backed NVIDIA route uses the curated chat catalog",
    "authza/": "Browser-backed model selector does not expose a stable catalog endpoint",
    "vertex/": "Vertex catalogs are project and region specific",
    "search/": "Search route is a fixed service rather than a model catalog",
    "perplexity/": "The provider catalog currently lists Agent API models, not this chat route",
}


_PREFIX_PROVIDER_MAP: Tuple[Tuple[str, str], ...] = (
    ("authgem-vertex/", "static"),
    ("authgpt/", "static"),
    ("authgrok/", "authgrok"),
    ("authgem-key/", "authgem_key"),
    ("authgem/", "static"),
    ("authcd/", "static"),
    ("authnd/", "static"),
    ("authza/", "static"),
    ("ocagy/", "ocagy"),
    ("antigravity/", "antigravity"),
    ("vertex/", "static"),
    ("search/", "static"),
    ("chutes/", "chutes"),
    ("groq/", "groq"),
    ("fireworks/", "fireworks"),
    ("or/", "openrouter"),
    ("openrouter/", "openrouter"),
    ("lr/", "literouter"),
    ("oc/", "opencode"),
    ("opencode/", "opencode"),
    ("opencode-go/", "opencode"),
    ("eh/", "electronhub"),
    ("electronhub/", "electronhub"),
    ("electron/", "electronhub"),
    ("nan/", "nanogpt"),
    ("sam/", "sambanova"),
    ("nd/", "nvidia"),
    ("za/", "zai"),
    ("together/", "together"),
    ("cohere/", "cohere"),
)


_BARE_PROVIDER_PREFIXES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("openai", ("gpt-", "chatgpt-", "o1", "o3", "o4")),
    ("anthropic", ("claude-",)),
    ("gemini", ("gemini-", "gemma-")),
    ("xai", ("grok-",)),
    ("deepseek", ("deepseek-",)),
    ("mistral", (
        "mistral-", "open-mistral-", "mixtral-", "codestral-", "devstral-", "pixtral-",
        "voxtral-", "magistral-", "ministral-", "labs-leanstral-",
    )),
    ("cohere", ("command", "aya-")),
    ("moonshot", ("moonshot-", "kimi-")),
    ("zhipu", ("glm-", "chatglm")),
    ("together", (
        "llama-", "llama2", "llama3", "llama4", "codellama-", "alpaca-",
        "vicuna-", "wizardlm-",
    )),
)


def _catalog_provider_for_model(model: str) -> Optional[str]:
    """Resolve a dropdown model to the catalog that owns its namespace."""
    value = str(model or "").strip().lower()
    authgrok_match = re.match(r"^authgrok(\d{1,4})/", value)
    if authgrok_match:
        return f"authgrok:{int(authgrok_match.group(1))}"
    for prefix, provider in _PREFIX_PROVIDER_MAP:
        if value.startswith(prefix):
            return provider
    for provider, prefixes in _BARE_PROVIDER_PREFIXES:
        if value.startswith(prefixes):
            return provider
    return None


def _model_catalog_cache_path() -> str:
    override = str(os.getenv("GLOSSARION_MODEL_CATALOG_CACHE", "") or "").strip()
    if override:
        return os.path.abspath(override)
    if platform.system() == "Windows":
        root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or tempfile.gettempdir()
    elif platform.system() == "Darwin":
        root = os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        root = os.getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "Glossarion", "model_catalog_cache.json")


def _empty_model_catalog_cache() -> dict:
    return {
        "version": _MODEL_CATALOG_CACHE_VERSION,
        "providers": {},
        "last_successful": {},
        "attempts": {},
    }


def _load_model_catalog_cache(*, force_disk: bool = False) -> dict:
    global _MODEL_CATALOG_MEMORY_CACHE
    with _MODEL_CATALOG_LOCK:
        if _MODEL_CATALOG_MEMORY_CACHE is not None and not force_disk:
            return json.loads(json.dumps(_MODEL_CATALOG_MEMORY_CACHE))
        path = _model_catalog_cache_path()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or loaded.get("version") != _MODEL_CATALOG_CACHE_VERSION:
                loaded = _empty_model_catalog_cache()
            if not isinstance(loaded.get("providers"), dict):
                loaded["providers"] = {}
            if not isinstance(loaded.get("last_successful"), dict):
                loaded["last_successful"] = {}
            # Migrate successful records written before persistent marker state
            # was introduced. Unlike the active 24-hour cache, these records
            # are retained across later failed refresh attempts.
            for name, record in loaded["providers"].items():
                if isinstance(record, dict) and isinstance(record.get("models"), list):
                    loaded["last_successful"].setdefault(name, record)
            if not isinstance(loaded.get("attempts"), dict):
                loaded["attempts"] = {}
        except (OSError, ValueError, TypeError):
            loaded = _empty_model_catalog_cache()
        _MODEL_CATALOG_MEMORY_CACHE = loaded
        return json.loads(json.dumps(loaded))


def _write_model_catalog_cache(cache: dict) -> None:
    global _MODEL_CATALOG_MEMORY_CACHE
    path = _model_catalog_cache_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix="model_catalog_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    with _MODEL_CATALOG_LOCK:
        _MODEL_CATALOG_MEMORY_CACHE = json.loads(json.dumps(cache))


def _cached_provider_models(*, max_age: int = _MODEL_CATALOG_CACHE_TTL_SECONDS) -> Dict[str, List[str]]:
    now = time.time()
    providers = _load_model_catalog_cache().get("providers", {})
    result: Dict[str, List[str]] = {}
    for name, record in providers.items():
        if not isinstance(record, dict):
            continue
        try:
            age = now - float(record.get("fetched_at", 0))
        except (TypeError, ValueError):
            continue
        models = record.get("models")
        if age <= max_age and isinstance(models, list):
            cleaned = [str(model) for model in models if isinstance(model, str) and model.strip()]
            if cleaned:
                result[str(name)] = cleaned
    return result


def get_last_successful_provider_models() -> Dict[str, List[str]]:
    """Return each provider's last successful catalog, regardless of age."""
    records = _load_model_catalog_cache().get("last_successful", {})
    result: Dict[str, List[str]] = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            continue
        models = record.get("models")
        if not isinstance(models, list):
            continue
        cleaned = [
            str(model) for model in models
            if isinstance(model, str) and model.strip()
        ]
        if cleaned:
            result[str(name)] = cleaned
    return result


def _deduplicate_models(models: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for model in models:
        value = str(model or "").strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _merge_dynamic_model_options(
    static_models: Sequence[str],
    provider_models: Mapping[str, Sequence[str]],
) -> List[str]:
    """Replace each successfully fetched provider section, preserving catalog order."""
    replacements = {
        str(provider): _deduplicate_models(models)
        for provider, models in provider_models.items()
        if models
    }
    if not replacements:
        return _deduplicate_models(static_models)

    merged: List[str] = []
    emitted = set()
    for model in static_models:
        provider = _catalog_provider_for_model(model)
        if provider in replacements:
            if provider not in emitted:
                merged.extend(replacements[provider])
                emitted.add(provider)
            continue
        merged.append(model)
    for provider, models in replacements.items():
        if provider not in emitted:
            merged.extend(models)
    return _deduplicate_models(merged)


def get_model_options() -> List[str]:
    """Return cached live provider catalogs with the built-in list as fallback."""
    static_models = _get_static_model_options()
    dynamic = _cached_provider_models()
    return _merge_dynamic_model_options(static_models, dynamic)


def _provider_models_url(spec: ProviderCatalogSpec) -> str:
    base = str(os.getenv(spec.base_url_env, "") or "").strip() if spec.base_url_env else ""
    if not base:
        return spec.models_url
    base = base.rstrip("/")
    if base.lower().endswith("/models"):
        return base
    if spec.name == "anthropic":
        for suffix in ("/v1/messages", "/v1/chat/completions", "/v1", "/chat/completions"):
            if base.lower().endswith(suffix):
                base = base[:-len(suffix)]
                break
        return f"{base}/v1/models?limit=1000"
    # NanoGPT's configured root normally omits /api/v1, while the other
    # configured provider roots already include their API version.
    return f"{base}{spec.models_path}"


def _append_query_parameter(url: str, name: str, value: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append((name, value))
    return urllib.parse.urlunsplit((
        parts.scheme, parts.netloc, parts.path,
        urllib.parse.urlencode(query), parts.fragment,
    ))


def _http_get_json(url: str, headers: Mapping[str, str], timeout: float) -> object:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _catalog_error_status(error: BaseException) -> str:
    """Return a useful, bounded catalog error without exposing request secrets."""
    if isinstance(error, urllib.error.HTTPError):
        code = int(getattr(error, "code", 0) or 0)
        reason = str(getattr(error, "reason", "") or "").strip()
        detail = ""
        try:
            raw_body = error.read(4096)
            body = raw_body.decode("utf-8", errors="replace").strip()
            if body:
                try:
                    payload = json.loads(body)
                    candidate = payload.get("error", payload) if isinstance(payload, dict) else payload
                    if isinstance(candidate, dict):
                        detail = str(
                            candidate.get("message")
                            or candidate.get("detail")
                            or candidate.get("error")
                            or ""
                        ).strip()
                    elif isinstance(candidate, str):
                        detail = candidate.strip()
                except (TypeError, ValueError):
                    detail = body
        except (OSError, ValueError):
            pass

        # Keep provider responses readable in the GUI log and avoid reflecting
        # long HTML error pages. The URL is intentionally excluded because a
        # Gemini catalog URL can contain its key as a query parameter.
        reason_detail = re.sub(r"\s+", " ", detail or reason).strip()[:240]
        suffix = f" — {reason_detail}" if reason_detail else ""
        return f"static fallback (HTTP {code}{suffix})"

    reason = re.sub(r"\s+", " ", str(error or "")).strip()[:160]
    suffix = f" — {reason}" if reason else ""
    return f"static fallback ({type(error).__name__}{suffix})"


def _extract_catalog_entries(payload: object, spec: ProviderCatalogSpec) -> List[object]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in spec.response_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_catalog_model_id(spec: ProviderCatalogSpec, model_id: object) -> Optional[str]:
    value = str(model_id or "").strip()
    if spec.name in ("gemini", "authgem_key") and value.startswith("models/"):
        value = value[len("models/"):]
    if not value or len(value) > 300 or any(ch.isspace() for ch in value):
        return None
    if value.lower().startswith(("http://", "https://")):
        return None
    if spec.prefix and not value.casefold().startswith(spec.prefix.casefold()):
        value = f"{spec.prefix}{value}"
    return value


def _fetch_provider_catalog(
    spec: ProviderCatalogSpec,
    api_key: str = "",
    timeout: float = 8.0,
) -> List[str]:
    url = _provider_models_url(spec)
    headers = {
        "Accept": "application/json",
        "User-Agent": "Glossarion/ModelCatalog",
    }
    if api_key:
        if spec.auth_style == "query":
            url = _append_query_parameter(url, "key", api_key)
        elif spec.auth_style == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    payload = _http_get_json(url, headers, timeout)
    entries = _extract_catalog_entries(payload, spec)
    models: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            model_id = entry
        elif isinstance(entry, dict):
            if entry.get("active") is False or entry.get("archived") is True:
                continue
            if spec.allowed_types:
                model_type = str(entry.get("type", "") or "").strip().casefold()
                if model_type and model_type not in spec.allowed_types:
                    continue
            if spec.name in ("gemini", "authgem_key"):
                actions = entry.get("supportedGenerationMethods", entry.get("supported_actions", []))
                if isinstance(actions, list) and actions and "generateContent" not in actions:
                    continue
            if spec.name == "cohere":
                endpoints = entry.get("endpoints", [])
                if isinstance(endpoints, list) and endpoints and not {"chat", "generate"}.intersection(endpoints):
                    continue
            model_id = next((entry.get(field) for field in spec.id_fields if entry.get(field)), None)
        else:
            model_id = None
        normalized = _normalize_catalog_model_id(spec, model_id)
        if normalized:
            models.append(normalized)
    return _deduplicate_models(models)


def _authgrok_catalog_target(active_model: str) -> Optional[Tuple[str, str, int]]:
    """Return (cache name, dropdown prefix, account id) for an AuthGrok route."""
    value = str(active_model or "").strip().lower()
    match = re.match(r"^authgrok(\d{0,4})/", value)
    if not match:
        return None
    account_id = int(match.group(1) or 0)
    if account_id:
        return f"authgrok:{account_id}", f"authgrok{account_id}/", account_id
    return "authgrok", "authgrok/", 0


def _fetch_authgrok_catalog(active_model: str, timeout: float) -> Optional[Tuple[str, List[str]]]:
    """Use AuthGrok's existing account catalog without initiating login."""
    target = _authgrok_catalog_target(active_model)
    if target is None:
        return None
    provider_name, prefix, account_id = target
    import authgrok_auth

    store = authgrok_auth.get_store(account_id)
    access_token = store.get_valid_access_token(auto_login=False)
    raw_models = authgrok_auth.fetch_available_models(
        access_token,
        timeout=max(1, int(round(timeout))),
    )
    models = _deduplicate_models(
        model if str(model).casefold().startswith(prefix.casefold()) else f"{prefix}{model}"
        for model in raw_models
    )
    if not models:
        raise ValueError("AuthGrok returned no usable model IDs")
    return provider_name, models


def _ocagy_has_account() -> bool:
    """Check OcAgy's local OAuth store without exposing or refreshing tokens."""
    import ocagy_cli

    summary = ocagy_cli.get_account_summary()
    return bool(int(summary.get("account_count", 0) or 0))


def _fetch_ocagy_catalog(timeout: float) -> List[str]:
    """Poll plugin-backed models through OpenCode's existing OAuth session."""
    import ocagy_cli

    models = _deduplicate_models(ocagy_cli.poll_models(timeout=timeout))
    if not models:
        raise ValueError("OcAgy returned no usable model IDs")
    return models


def _provider_key(
    spec: ProviderCatalogSpec,
    active_provider: Optional[str],
    active_api_key: str,
    provider_keys: Mapping[str, str],
) -> str:
    if spec.public:
        return ""
    explicit = str(provider_keys.get(spec.name, "") or "").strip()
    if explicit:
        return explicit
    for env_name in spec.api_key_envs:
        value = str(os.getenv(env_name, "") or "").strip()
        if value:
            return value
    if active_provider == spec.name:
        return str(active_api_key or "").strip()
    return ""


def _custom_catalog_specs(custom_routes: object, active_model: str) -> List[ProviderCatalogSpec]:
    if not isinstance(custom_routes, list):
        return []
    model_value = str(active_model or "").strip().casefold()
    result: List[ProviderCatalogSpec] = []
    for entry in custom_routes:
        if not isinstance(entry, dict):
            continue
        prefix = str(entry.get("prefix", "") or "").strip().replace("\\", "/").lstrip("/")
        routing = str(entry.get("routing", entry.get("base_url", "")) or "").strip().rstrip("/")
        endpoint_type = str(entry.get("endpoint_type", "/chat/completions") or "").strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        if (
            not prefix or not routing.startswith(("http://", "https://"))
            or endpoint_type not in ("/chat/completions", "chat/completions", "openai_chat")
            or not model_value.startswith(prefix.casefold())
        ):
            continue
        result.append(ProviderCatalogSpec(
            name=f"custom:{prefix.casefold()}",
            prefix=prefix,
            models_url=f"{routing}/models",
        ))
    return result


def catalog_provider_for_model(model: str, custom_routes: object = None) -> Optional[str]:
    """Return the pollable catalog owner for a model ID, including custom routes."""
    provider = _catalog_provider_for_model(model)
    if provider is not None:
        return None if provider == "static" else provider
    custom_specs = _custom_catalog_specs(custom_routes, model)
    return custom_specs[0].name if custom_specs else None


def provider_model_catalog_refresh_due(
    provider: str,
    *,
    max_age: int = _MODEL_CATALOG_CACHE_TTL_SECONDS,
    successful_only: bool = False,
) -> bool:
    """Return whether an automatic provider poll is due under the persisted TTL.

    ``successful_only`` is used when a previously unavailable local service has
    just become ready. In that case, an earlier failed attempt must not suppress
    the first usable poll, while a successful catalog from the last 24 hours
    should still be reused.
    """
    provider = str(provider or "").strip()
    if not provider:
        return False
    cache = _load_model_catalog_cache()
    attempts = cache.get("attempts", {})
    providers = cache.get("providers", {})
    last_successful = cache.get("last_successful", {})
    timestamps: List[float] = []
    if not successful_only:
        try:
            timestamps.append(float(attempts.get(provider, 0) or 0))
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        timestamps.append(float((providers.get(provider, {}) or {}).get("fetched_at", 0) or 0))
    except (AttributeError, TypeError, ValueError):
        pass
    if successful_only:
        try:
            timestamps.append(
                float((last_successful.get(provider, {}) or {}).get("fetched_at", 0) or 0)
            )
        except (AttributeError, TypeError, ValueError):
            pass
    last_attempt = max(timestamps or [0.0])
    return (time.time() - last_attempt) >= max(0, int(max_age))


def due_provider_catalog_for_model(
    active_model: str,
    active_api_key: str = "",
    custom_routes: object = None,
    *,
    max_age: int = _MODEL_CATALOG_CACHE_TTL_SECONDS,
) -> Optional[str]:
    """Return the selected provider when it is credentialed and due for auto-polling."""
    provider = catalog_provider_for_model(active_model, custom_routes)
    # The Antigravity catalog lives behind a local proxy. Typing/selecting its
    # prefix must not spend the 24-hour attempt TTL while that proxy is offline;
    # its dedicated proxy-start hook polls once the service is healthy instead.
    if provider == "antigravity":
        return None
    if not provider or not provider_model_catalog_refresh_due(provider, max_age=max_age):
        return None

    # AuthGrok uses its existing account session and checks it without opening
    # a login window. The actual session validation remains in the worker.
    if provider == "authgrok" or provider.startswith("authgrok:"):
        return provider

    if provider == "ocagy":
        try:
            return provider if _ocagy_has_account() else None
        except Exception:
            return None

    specs = list(PROVIDER_CATALOG_SPECS)
    specs.extend(_custom_catalog_specs(custom_routes, active_model))
    spec = next((item for item in specs if item.name == provider), None)
    if spec is None:
        return None
    key = _provider_key(spec, provider, active_api_key, {})
    return provider if spec.public or bool(key) else None


def refresh_provider_model_catalogs(
    *,
    active_model: str = "",
    active_api_key: str = "",
    provider_keys: Optional[Mapping[str, str]] = None,
    custom_routes: object = None,
    timeout: float = 8.0,
    max_workers: int = 6,
    only_provider: Optional[str] = None,
) -> ModelCatalogRefreshResult:
    """Fetch eligible provider catalogs and atomically update the local cache.

    The generic active key is sent only to the provider resolved from
    ``active_model``. Other providers are queried only when they are public or
    have a provider-specific credential.
    """
    active_provider = catalog_provider_for_model(active_model, custom_routes)
    only_provider = str(only_provider or "").strip() or None
    provider_keys = dict(provider_keys or {})
    specs = list(PROVIDER_CATALOG_SPECS)
    custom_specs = _custom_catalog_specs(custom_routes, active_model)
    specs.extend(custom_specs)
    if only_provider:
        specs = [spec for spec in specs if spec.name == only_provider]

    statuses: Dict[str, str] = {}
    eligible: List[Tuple[ProviderCatalogSpec, str]] = []
    for spec in specs:
        key = _provider_key(spec, active_provider, active_api_key, provider_keys)
        if spec.name.startswith("custom:") and active_api_key:
            key = str(active_api_key).strip()
        if not spec.public and not key:
            statuses[spec.name] = "static fallback (no provider credential)"
            continue
        eligible.append((spec, key))

    successful: Dict[str, List[str]] = {}
    failed = set()
    attempted = {spec.name for spec, _key in eligible}

    authgrok_target = _authgrok_catalog_target(active_model)
    if (
        authgrok_target is not None
        and (only_provider is None or authgrok_target[0] == only_provider)
    ):
        authgrok_name = authgrok_target[0]
        attempted.add(authgrok_name)
        try:
            authgrok_result = _fetch_authgrok_catalog(active_model, timeout)
            if authgrok_result is not None:
                name, models = authgrok_result
                successful[name] = models
                statuses[name] = f"online ({len(models)} models)"
        except Exception as error:
            failed.add(authgrok_name)
            statuses[authgrok_name] = _catalog_error_status(error)

    if only_provider is None or only_provider == "ocagy":
        try:
            ocagy_authenticated = _ocagy_has_account()
        except Exception as error:
            ocagy_authenticated = False
            statuses["ocagy"] = _catalog_error_status(error)
        if ocagy_authenticated:
            attempted.add("ocagy")
            try:
                ocagy_models = _fetch_ocagy_catalog(timeout)
                successful["ocagy"] = ocagy_models
                statuses["ocagy"] = f"online ({len(ocagy_models)} models)"
            except Exception as error:
                failed.add("ocagy")
                statuses["ocagy"] = _catalog_error_status(error)
        elif "ocagy" not in statuses:
            statuses["ocagy"] = "static fallback (no provider credential)"

    def fetch(item: Tuple[ProviderCatalogSpec, str]):
        spec, key = item
        return spec, _fetch_provider_catalog(spec, key, timeout)

    if eligible:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(max_workers), len(eligible))),
            thread_name_prefix="model-catalog",
        ) as executor:
            futures = {executor.submit(fetch, item): item[0] for item in eligible}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    models = future.result()[1]
                    if not models:
                        raise ValueError("provider returned no usable model IDs")
                    successful[spec.name] = models
                    statuses[spec.name] = f"online ({len(models)} models)"
                except Exception as error:
                    failed.add(spec.name)
                    statuses[spec.name] = _catalog_error_status(error)

    try:
        cache = _load_model_catalog_cache(force_disk=True)
        providers = cache.setdefault("providers", {})
        last_successful = cache.setdefault("last_successful", {})
        attempts = cache.setdefault("attempts", {})
        now = time.time()
        built_in_names = {spec.name for spec in PROVIDER_CATALOG_SPECS}
        built_in_names.add("ocagy")
        built_in_names.update(spec.name for spec in custom_specs)
        built_in_names.update(name for name in successful if name.startswith("authgrok"))
        built_in_names.update(name for name in failed if name.startswith("authgrok"))
        for name in failed:
            if name in built_in_names:
                providers.pop(name, None)
        for name, models in successful.items():
            if name in built_in_names:
                record = {"fetched_at": now, "models": models}
                providers[name] = record
                last_successful[name] = record
        for name in attempted:
            attempts[name] = now
        cache["version"] = _MODEL_CATALOG_CACHE_VERSION
        cache["updated_at"] = now
        _write_model_catalog_cache(cache)
    except OSError as error:
        # A read-only or unavailable cache directory must not discard a
        # catalog that was fetched successfully for the current session.
        statuses["cache"] = f"memory only ({type(error).__name__})"

    # A full refresh uses only catalogs fetched successfully in this run. A
    # provider-scoped automatic refresh preserves other still-fresh cached
    # catalogs while replacing (or clearing) the selected provider.
    if only_provider:
        runtime_models = _cached_provider_models()
        runtime_models.pop(only_provider, None)
        runtime_models.update(successful)
    else:
        runtime_models = dict(successful)
    options = _merge_dynamic_model_options(_get_static_model_options(), runtime_models)
    return ModelCatalogRefreshResult(options, successful, statuses, only_provider)


def start_provider_model_catalog_refresh(
    callback: Optional[Callable[[ModelCatalogRefreshResult], None]] = None,
    **kwargs,
) -> threading.Thread:
    """Start a daemon refresh thread and optionally receive its result."""
    def worker() -> None:
        try:
            result = refresh_provider_model_catalogs(**kwargs)
        except Exception as error:
            result = ModelCatalogRefreshResult(
                get_model_options(),
                {},
                {"catalog": f"static fallback ({type(error).__name__})"},
            )
        if callback is not None:
            try:
                callback(result)
            except Exception:
                # The GUI may have closed while this daemon worker was active.
                pass

    thread = threading.Thread(target=worker, name="provider-model-catalog", daemon=True)
    thread.start()
    return thread
