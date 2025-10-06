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
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
        "gpt-5-mini","gpt-5","gpt-5-nano",
        "o1-preview", "o1-mini", "o3", "o4-mini",
        
        # Google Gemini Models
        "gemini-2.0-flash","gemini-2.0-flash-lite",
        "gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-pro", "gemini-pro-vision",
        
        # Anthropic Claude Models
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        "claude-2.1", "claude-2", "claude-instant-1.2",
        
        # Grok Models
        "grok-4-0709", "grok-4-fast", "grok-4-fast-reasoning",  "grok-4-fast-reasoning-latest", "grok-3", "grok-3-mini",
        
        # Vertex AI Model Garden - Claude models (confirmed)
        "claude-4-opus@20250514",
        "claude-4-sonnet@20250514",
        "claude-opus-4@20250514",
        "claude-sonnet-4@20250514",
        "claude-3-7-sonnet@20250219",
        "claude-3-5-sonnet@20240620",
        "claude-3-5-sonnet-v2@20241022",
        "claude-3-opus@20240229",
        "claude-3-sonnet@20240229",
        "claude-3-haiku@20240307",

        
        # Alternative format with vertex_ai prefix
        "vertex/claude-3-7-sonnet@20250219",
        "vertex/claude-3-5-sonnet@20240620",
        "vertex/claude-3-opus@20240229",
        "vertex/claude-4-opus@20250514",
        "vertex/claude-4-sonnet@20250514",
        "vertex/gemini-1.5-pro",
        "vertex/gemini-1.5-flash",
        "vertex/gemini-2.0-flash",
        "vertex/gemini-2.5-pro",
        "vertex/gemini-2.5-flash",
        "vertex/gemini-2.5-flash-lite",

        # Chute AI
        "chutes/openai/gpt-oss-120b",
        "chutes/deepseek-ai/DeepSeek-V3.1",
        
        # DeepSeek Models
        "deepseek-chat", "deepseek-coder", "deepseek-coder-33b-instruct",
        
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
        
        # Groq Models (usually with suffix)
        "llama-3-70b-groq", "llama-3-8b-groq", "mixtral-8x7b-groq",
        
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
        "or/google/gemini-2.5-pro",
        "or/google/gemini-2.5-flash",
        "or/google/gemini-2.5-flash-lite",
        "or/openai/gpt-5",
        "or/openai/gpt-5-mini",
        "or/openai/gpt-5-nano",
        "or/openai/chatgpt-4o-latest",   
        "or/deepseek/deepseek-r1-0528:free", 
        "or/google/gemma-3-27b-it:free",
        
        # For ElectronHub, prefix with 'eh/'
        "eh/gpt-4", "eh/gpt-3.5-turbo", "eh/claude-3-opus", "eh/claude-3-sonnet",
        "eh/llama-2-70b-chat", "eh/yi-34b-chat-200k", "eh/mistral-large",
        "eh/gemini-pro", "eh/deepseek-coder-33b",
        
        # Last Resort
        "deepl",  # Will use DeepL API
        "google-translate-free",  # Uses free web endpoint (no key)
        "google-translate",  # Will use Google Cloud Translate
    ]
