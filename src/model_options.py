# model_options.py
"""
Centralized model catalog for Glossarion UIs.
Returned list should mirror the main GUI model dropdown.
"""
from typing import List

def get_model_options() -> List[str]:
    return [
    
        # OpenAI Models
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k","gpt-5.2", "gpt-5.2-pro", "gpt-5.2-chat-latest",
        "gpt-5-mini","gpt-5","gpt-5-nano", "gpt-5-chat-latest", "gpt-5-codex", "gpt-5-pro", "gpt-5-pro-2025-10-06",
        "gpt-5.1", "gpt-5.1-chat-latest", "gpt-5.1-codex", "gpt-5.1-codex-mini",
        "o1-preview", "o1-mini", "o3", "o4-mini",
        
        # Google Gemini Models
        "gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro",
        "gemini-2.0-flash","gemini-2.0-flash-lite", "gemini-pro", "gemini-pro-vision",
        "gemini-3-pro-preview","gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3-pro-image-preview",
        
        # Anthropic Claude Models
        "claude-opus-4-6", "claude-opus-4-5-20251101", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-6", 
        "claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
        "claude-3-haiku-20240307",       
        
        # Grok Models
        "grok-4-0709", "grok-4-fast", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",  "grok-4-fast-reasoning-latest", "grok-3", "grok-3-mini",
        "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning",
        
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
        "vertex/gemini-3-pro-preview",
        "vertex/gemini-3-flash-preview",
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
        
        # DeepSeek Models
        "deepseek-chat","deepseek-reasoner", "deepseek-coder", "deepseek-coder-33b-instruct",
        
        # Mistral Models
        "mistral-large", "mistral-medium", "mistral-small", "mistral-tiny",
        "mixtral-8x7b-instruct", "mixtral-8x22b", "codestral-latest",
        
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
        
        # Chinese Models
        "glm-4", "glm-3-turbo", "chatglm-6b", "chatglm2-6b", "chatglm3-6b",
        "baichuan-13b-chat", "baichuan2-13b-chat",
        "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
        
        # Other Models
        "falcon-40b-instruct", "falcon-7b-instruct",
        "phi-2", "phi-3-mini", "phi-3-small", "phi-3-medium",
        "orca-2-13b", "orca-2-7b",
        "vicuna-13b", "vicuna-7b",
        "alpaca-7b",
        "wizardlm-70b", "wizardlm-13b",
        "openchat-3.5",
        
        # For POE, prefix with 'poe/'
        "poe/gpt-4", "poe/gpt-4o", "poe/gpt-4.5", "poe/gpt-4.1",
        "poe/claude-3-opus", "poe/claude-4-opus", "poe/claude-3-sonnet", "poe/claude-4-sonnet",
        "poe/claude", "poe/Assistant",
        "poe/gemini-2.5-flash", "poe/gemini-2.5-pro",
        
        # For OR, prevfix with 'or/'
        "or/openrouter/free",
        "or/google/gemini-2.5-pro",
        "or/google/gemini-2.5-flash",
        "or/google/gemini-2.5-flash-preview-09-2025",
        "or/google/gemini-2.5-flash-lite",
        "or/openai/gpt-oss-20b:free","or/openai/gpt-5",
        "or/openai/gpt-5-mini",
        "or/openai/gpt-5-nano",
        "or/openai/chatgpt-4o-latest", "or/deepseek/deepseek-chat-v3-0324:free", "or/deepseek/deepseek-r1-distill-llama-70b:free",
        "or/deepseek/deepseek-r1:free","or/deepseek/deepseek-r1-0528:free", "or/deepseek/deepseek-chat-v3.1:free", "or/deepseek/deepseek-r1-0528-qwen3-8b:free",
        "or/tngtech/deepseek-r1t2-chimera:free","or/tngtech/deepseek-r1t-chimera:free",
        "or/google/gemma-3-27b-it:free", "or/google/gemma-3-27b-it",
        "or/qwen/qwen3-235b-a22b", "or/qwen/qwen3-235b-a22b-thinking-2507",
        
        # For ElectronHub, prefix with 'eh/'
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

        # AuthGPT – ChatGPT subscription via OAuth (no API key needed)
        "authgpt/gpt-5.2", "authgpt/gpt-5.2-pro", "authgpt/gpt-5.2-chat-latest",
        "authgpt/gpt-5.1", "authgpt/gpt-5.1-codex", "authgpt/gpt-5.1-codex-max",
        "authgpt/gpt-5.2-codex", "authgpt/gpt-5.3-codex",
        "authgpt/gpt-5-mini", "authgpt/gpt-5-codex-mini",

        # NVIDIA Integrate (OpenAI-compatible) — models from UI dropdown
        "nd/deepseek-ai/deepseek-v3.2",
        "nd/deepseek-ai/deepseek-r1",
        "nd/moonshotai/kimi-k2-thinking",
        "nd/meta/llama-4-maverick-17b-128e-instruct",
        "nd/meta/llama-4-scout-17b-16e-instruct",
        "nd/meta/llama-3.3-70b-instruct",
        "nd/qwen/qwen2.5-coder-32b-instruct",
        
        # Last Resort
        "deepl",  # Will use DeepL API
        "google-translate-free",  # Uses free web endpoint (no key)
        "google-translate",  # Will use Google Cloud Translate
    ]
