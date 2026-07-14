# model_options.py
"""
Centralized model catalog for Glossarion UIs.
Returned list should mirror the main GUI model dropdown.
Updated: 2026-03-10
"""
from typing import List

def get_model_options() -> List[str]:
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
        "gemini-3.5-flash","gemini-3-flash-preview",
        "gemini-3.1-pro-preview","gemini-3.1-flash-lite",
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
        "grok-4.3","grok-4.20-beta-0309-reasoning","grok-4.20-beta-0309-non-reasoning", "grok-4.20-multi-agent-beta-0309",
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
        "or/openai/gpt-5.4","or/openai/gpt-5.4-mini", "or/openai/gpt-5.4-nano"
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
        "or/tngtech/deepseek-r1t2-chimera:free","or/tngtech/deepseek-r1t-chimera:free",
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

        # AuthCD – Claude subscription via OAuth (Anthropic Messages API)
        "authcd/claude-sonnet-5", "authcd/claude-fable-5", "authcd/claude-sonnet-4-6", "authcd/claude-sonnet-4-5", "authcd/claude-opus-4-8",
        "authcd/claude-opus-4-7", "authcd/claude-opus-4-6", "authcd/claude-haiku-4-5",

        # AuthGem – Gemini-cli via Google OAuth (no API key needed)
        "authgem/gemini-2.5-flash", "authgem/gemini-2.5-flash-lite",
        "authgem/gemini-2.5-pro",
        "authgem/gemini-2.0-flash", "authgem/gemini-2.0-flash-lite",
        "authgem/gemini-3.1-pro-preview", "authgem/gemini-3.1-flash-lite", "authgem/gemini-3-flash-preview",

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

        # Antigravity Cloud Code proxy (frieser/antigravity-proxy dashboard catalog)
        "antigravity/gemini-3-flash",
        "antigravity/gemini-3-flash-agent",
        "antigravity/gemini-3.1-flash-image",
        "antigravity/gemini-3.1-flash-lite",
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
