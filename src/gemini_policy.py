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
