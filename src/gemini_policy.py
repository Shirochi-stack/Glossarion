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
