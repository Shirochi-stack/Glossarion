"""Exact, provider-specific Gemini policy refusal artifacts."""

GOOGLE_PROHIBITED_USE_POLICY_MESSAGE = (
    "The prompt could not be submitted. The prompt contains sensitive words that "
    "violate Google's [Generative AI Prohibited Use policy]"
    "(https://policies.google.com/terms/generative-ai/use-policy). Try rephrasing "
    "the prompt. If you think this was an error, [send feedback]"
    "(https://ai.google.dev/gemini-api/docs/troubleshooting)."
)


def is_google_prohibited_use_policy_refusal(content) -> bool:
    """Return True only for Google's complete canonical policy refusal text."""
    return (
        isinstance(content, str)
        and content.strip() == GOOGLE_PROHIBITED_USE_POLICY_MESSAGE
    )


#: Adjustable Gemini harm categories (API names) with their Other Settings labels, in send order.
GEMINI_SAFETY_CATEGORIES = (
    ("HARM_CATEGORY_HATE_SPEECH", "Hate speech"),
    ("HARM_CATEGORY_DANGEROUS_CONTENT", "Dangerous content"),
    ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "Sexually explicit"),
    ("HARM_CATEGORY_HARASSMENT", "Harassment"),
    ("HARM_CATEGORY_CIVIC_INTEGRITY", "Civic integrity"),
    ("HARM_CATEGORY_JAILBREAK", "Jailbreak"),
)
GEMINI_SAFETY_THRESHOLDS = ("OFF", "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE")


def gemini_safety_thresholds(default="OFF"):
    """``[(category, threshold), ...]`` for every adjustable category: its own threshold from
    ``GEMINI_SAFETY_CATEGORY_THRESHOLDS`` (JSON ``{category: threshold}``, set in Other Settings),
    else ``GEMINI_SAFETY_THRESHOLD``, else ``default``. Unknown values fall back the same way."""
    import json
    import os
    base = str(os.getenv("GEMINI_SAFETY_THRESHOLD", default) or default).strip().upper()
    if base not in GEMINI_SAFETY_THRESHOLDS:
        base = default
    try:
        overrides = json.loads(os.getenv("GEMINI_SAFETY_CATEGORY_THRESHOLDS", "") or "{}")
    except (TypeError, ValueError):
        overrides = {}
    if not isinstance(overrides, dict):
        overrides = {}
    result = []
    for category, _label in GEMINI_SAFETY_CATEGORIES:
        value = str(overrides.get(category) or "").strip().upper()
        result.append((category, value if value in GEMINI_SAFETY_THRESHOLDS else base))
    return result


def describe_gemini_safety(pairs):
    """``(status, readable)`` for the safety log and payload record of ``gemini_safety_thresholds()``:
    "DISABLED - All categories set to X" when they agree, else the per-category values."""
    readable = {category.replace("HARM_CATEGORY_", ""): threshold for category, threshold in pairs}
    values = set(readable.values())
    if len(values) == 1:
        return f"DISABLED - All categories set to {values.pop()}", readable
    return "DISABLED - Per category: " + ", ".join(f"{k}={v}" for k, v in readable.items()), readable


def _gemini_version(model):
    """``(major, minor)`` of a numbered Gemini model (``gemini-3-flash`` -> (3, 0)), else None."""
    import re
    match = re.search(r"(?:^|/)gemini-(\d+)(?:\.(\d+))?(?=[.-]|$)", str(model or "").lower())
    return (int(match.group(1)), int(match.group(2) or 0)) if match else None


def uses_gemini_thinking_level(model) -> bool:
    """Gemini 3 and all later numbered generations use levels, not budgets."""
    version = _gemini_version(model)
    return bool(version and version[0] >= 3)


GEMINI_THINKING_LEVELS = ('minimal', 'low', 'medium', 'high', 'xhigh')


def gemini_thinking_level(model, level) -> str:
    """The level to send for *model*: unknown values become ``high``; ``xhigh`` exists only on
    Gemini 4 Neon/Hydrogen (``high`` elsewhere); ``minimal`` is rejected by Gemini 3 Pro,
    Gemini 4 and gemini-3.8-flash (``low`` there)."""
    level = str(level or '').strip().lower()
    if level not in GEMINI_THINKING_LEVELS:
        level = 'high'
    name = str(model or '').lower()
    version = _gemini_version(name)
    if level == 'xhigh' and not ('gemini-4-neon' in name or 'gemini-4-hydrogen' in name):
        level = 'high'
    if level == 'minimal' and version and (
            version[0] >= 4 or version == (3, 8)
            or (version[0] == 3 and 'pro' in name and 'flash' not in name)):
        level = 'low'
    return level


def gemini_ignores_sampling(model) -> bool:
    """Gemini 3.5 and later ignore temperature, top_p and top_k (deprecated 2026-07-21; Gemini 4
    Argon: "If provided, they are ignored"). Gemini 3 and 3.1 still use them."""
    version = _gemini_version(model)
    return bool(version and version >= (3, 5))


#: google-genai 3.0 moved the Gemini Developer API default from v1beta to v1. The -latest
#: aliases, most -preview models, the Live API (Lyria RealTime), caches and batch jobs on
#: preview models exist only on v1beta, so every Developer-API client stays on v1beta (the
#: google-genai 1.x behaviour). Vertex clients (vertexai=True) keep the SDK default (v1beta1).
GEMINI_DEVELOPER_API_VERSION = 'v1beta'


def new_gemini_api_client(api_key, genai_module=None):
    """``genai.Client`` for the Gemini Developer API, pinned to GEMINI_DEVELOPER_API_VERSION.
    Not for Vertex clients. ``genai_module`` defaults to ``google.genai``."""
    if genai_module is None:
        from google import genai as genai_module
    types = getattr(genai_module, 'types', None)
    http_options_cls = getattr(types, 'HttpOptions', None) if types is not None else None
    if http_options_cls is None:
        return genai_module.Client(api_key=api_key)
    return genai_module.Client(api_key=api_key,
                               http_options=http_options_cls(api_version=GEMINI_DEVELOPER_API_VERSION))


#: Other Settings "Long Decode" values (config ``gemini_long_decode``, env GEMINI_LONG_DECODE).
GEMINI_LONG_DECODE_MODES = ('auto', 'off', 'on')
#: One Gemini 4 decode hop ends at this many output tokens; more needs continuation.
GEMINI_LONG_DECODE_HOP_TOKENS = 65536


def uses_gemini_long_decode(model) -> bool:
    """Gemini 4 models (carbon/argon/neon/hydrogen) continue past one hop with a continuation token."""
    version = _gemini_version(model)
    return bool(version and version[0] >= 4)


def gemini_long_decode(model, max_output_tokens, mode=None):
    """``(enabled, reason)``: whether to let google-genai continue a Gemini 4 answer past
    GEMINI_LONG_DECODE_HOP_TOKENS (automatic continuation). ``mode`` defaults to
    GEMINI_LONG_DECODE ('off' when unset or unknown). Auto turns on when the output token
    limit is above one hop. ``reason`` is None when off."""
    import os
    if mode is None:
        mode = os.getenv('GEMINI_LONG_DECODE', 'off')
    mode = str(mode or 'off').strip().lower()
    if mode not in GEMINI_LONG_DECODE_MODES:
        mode = 'off'
    if mode == 'off' or not uses_gemini_long_decode(model):
        return False, None
    if mode == 'on':
        return True, "Long Decode is On"
    try:
        limit = int(max_output_tokens or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit > GEMINI_LONG_DECODE_HOP_TOKENS:
        return True, f"Long Decode is Auto and the output token limit {limit:,} is above {GEMINI_LONG_DECODE_HOP_TOKENS:,}"
    return False, None


_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'topP', 'topK')


def omit_gemini_legacy_parameters(payload, model=None):
    """Remove the thinking budget from a Gemini 3+ payload (they use levels), and for Gemini 3.5+
    (``gemini_ignores_sampling``) the sampling parameters too. Without ``model`` both go."""
    if not isinstance(payload, dict):
        return
    keys = ('thinking_budget', 'thinkingBudget')
    if model is None or gemini_ignores_sampling(model):
        keys += _SAMPLING_KEYS
    for key in keys:
        payload.pop(key, None)
    for key in ('extra_body', 'google', 'generation_config', 'generationConfig', 'parameters', 'thinkingConfig', 'thinking_config'):
        omit_gemini_legacy_parameters(payload.get(key), model)
