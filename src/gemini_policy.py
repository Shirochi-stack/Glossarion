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


def uses_gemini_thinking_level(model) -> bool:
    """Gemini 3 and all later numbered generations use levels, not budgets."""
    import re
    match = re.search(r"(?:^|/)gemini-(\d+)(?=[.-]|$)", str(model or "").lower())
    return bool(match and int(match.group(1)) >= 3)


def omit_gemini_legacy_parameters(payload):
    """Remove deprecated sampling and budget controls from Gemini 3+ payloads."""
    if not isinstance(payload, dict):
        return
    for key in ('temperature', 'top_p', 'top_k', 'topP', 'topK', 'thinking_budget', 'thinkingBudget'):
        payload.pop(key, None)
    for key in ('extra_body', 'google', 'generation_config', 'generationConfig', 'parameters', 'thinkingConfig', 'thinking_config'):
        omit_gemini_legacy_parameters(payload.get(key))
