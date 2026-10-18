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


def gemini_ignores_sampling(model) -> bool:
    """Gemini 3.5 and later ignore temperature, top_p and top_k (deprecated 2026-07-21; Gemini 4
    Argon: "If provided, they are ignored"). Gemini 3 and 3.1 still use them."""
    version = _gemini_version(model)
    return bool(version and version >= (3, 5))


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
