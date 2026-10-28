"""The built-in on-device LLM engine (Model manager › On-device; the shared ``local/`` route).

One tap on a model downloads its GGUF from Hugging Face and runs it inside the app with llama.cpp
(llama-cpp-python, ``engine.ENGINE_VERSION``). Every job kind then reaches it like any other model:
``UnifiedClient`` sends ``local/<stem>`` to the in-app OpenAI-compatible endpoint on 127.0.0.1
(``server.LocalLLMServer``, published as ``GLOSSARION_LOCAL_LLM_URL``).

* ``presets``: the editable preset rows (Prefs ``local_model_presets``; Qwen3.5 0.8B/2B/4B/9B defaults);
* ``hf``: Hugging Face references (repo, repo:quant, file paths and URLs), file listing and auto-pick;
* ``downloads``: resumable, cancellable GGUF downloads with sha256 verification and the registry;
* ``engine``: ``LlamaEngine`` (llama_cpp, imported lazily) and ``FakeEngine`` (tests, offline e2e);
* ``script_ban``: the adaptive script ban (which requests, which scripts, which token ids, the logits
  processor);
* ``server``: the loopback endpoint (``/v1/models``, ``/v1/chat/completions``, ``/health``);
* ``service``: ``LocalEngineService`` (one model loaded at a time, RAM gate, unload policy, lifecycle).

Nothing here imports Flet. ``llama_cpp`` is optional: without it the app says the engine is unavailable.
"""

from __future__ import annotations

__all__ = ["LOCAL_PREFIX", "PROVIDER", "is_local_model", "stem_of"]

#: The shared route prefix (``UnifiedClient._LOCAL_OPENAI_ROUTES``) and catalog provider name.
LOCAL_PREFIX = "local/"
PROVIDER = "local"


def is_local_model(model: object) -> bool:
    return str(model or "").strip().lower().startswith(LOCAL_PREFIX)


def stem_of(model: object) -> str:
    """``local/Qwen3.5-2B-Q4_K_M`` -> ``Qwen3.5-2B-Q4_K_M`` (a bare stem stays as it is)."""
    text = str(model or "").strip()
    return text[len(LOCAL_PREFIX):].strip() if text.lower().startswith(LOCAL_PREFIX) else text
