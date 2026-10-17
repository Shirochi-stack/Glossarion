"""Session-wide Claude temperature/thinking exclusions and HTTP 400 compatibility."""
import re
import threading

_rejected_models = set()
_adaptive_required_models = set()
_lock = threading.Lock()


def is_temperature_rejection(status, detail):
    return status == 400 and "temperature" in str(detail or "").lower()


def _is_mythos_preview(name):
    return bool(re.search(r"(?:^|[/.])claude-mythos-preview(?=$|[-:@.])", name))


def _claude_family_version(name):
    """(family, (major, minor)) for a Claude model id, or None.

    Handles provider/Bedrock prefixes, dated aliases, and both Claude naming orders.
    """
    families = r"sonnet|opus|haiku|fable|mythos"
    match = re.search(
        rf"(?:^|[/.])claude-(?:({families})-(\d+)(?:[.-](\d{{1,2}})(?!\d))?"
        rf"|(\d+)(?:[.-](\d{{1,2}})(?!\d))?-({families}))(?=$|[-:@])",
        name,
    )
    if not match:
        return None
    family = match[1] or match[6]
    return family, (int(match[2] or match[4]), int(match[3] or match[5] or 0))


def claude_omits_sampling(model):
    """Claude models whose documented API omits temperature, top_p and top_k.

    See https://platform.claude.com/docs/en/about-claude/model-deprecations
    and the family migration guides. Older families remain configurable.
    """
    name = str(model or "").strip().lower()
    if _is_mythos_preview(name):
        return True
    parsed = _claude_family_version(name)
    if not parsed:
        return False
    family, version = parsed
    minimum = {"sonnet": (5, 0), "opus": (4, 7), "haiku": (5, 5),
               "fable": (5, 0), "mythos": (5, 0)}[family]
    return version >= minimum


def claude_requires_adaptive_thinking(model):
    """Claude models that reject thinking.type "enabled" (budget_tokens).

    Opus 4.6 / Sonnet 4.6 are the last models that accept a thinking budget;
    every Claude model after 4.6 (Opus 4.7+, Sonnet 5+, Haiku 5.5+, Fable,
    Mythos) only takes {"type": "adaptive"} + output_config.effort. Models
    learned from the Models API catalog or a 400 are cached for the session.
    """
    name = str(model or "").strip().lower()
    if _is_mythos_preview(name):
        return True
    parsed = _claude_family_version(name)
    if parsed and parsed[1] > (4, 6):
        return True
    with _lock:
        return name in _adaptive_required_models


def claude_uses_adaptive_thinking(model):
    """Adaptive-only models plus Opus/Sonnet 4.6, where adaptive is recommended."""
    if claude_requires_adaptive_thinking(model):
        return True
    parsed = _claude_family_version(str(model or "").strip().lower())
    return bool(parsed) and parsed[0] in ("opus", "sonnet") and parsed[1] == (4, 6)


def is_adaptive_required_rejection(status, detail):
    text = str(detail or "").lower()
    return status == 400 and "thinking.type.enabled" in text and "not supported" in text


def remember_adaptive_required(model):
    name = str(model or "").strip().lower()
    if name:
        with _lock:
            _adaptive_required_models.add(name)


def model_rejects_temperature(model):
    name = str(model or "").strip().lower()
    if claude_omits_sampling(name):
        return True
    with _lock:
        return name in _rejected_models


def omit_claude_sampling_parameters(payload, model):
    """Keep learned temperature-only exclusions separate from known sampling rules."""
    if not claude_omits_sampling(model) or not isinstance(payload, dict):
        return
    for key in ('temperature', 'top_p', 'top_k', 'topP', 'topK'):
        payload.pop(key, None)
    for key in ('parameters', 'input', 'generationConfig', 'generation_config', 'extra_body'):
        omit_claude_sampling_parameters(payload.get(key), model)


def remember_temperature_rejection(model):
    name = str(model or "").strip().lower()
    if name:
        with _lock:
            _rejected_models.add(name)


def strip_temperature(payload):
    """Remove sampling temperature from common provider payload shapes."""
    if not isinstance(payload, dict):
        return False
    removed = "temperature" in payload
    payload.pop("temperature", None)
    for key in ("parameters", "input", "generationConfig", "generation_config", "extra_body"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            removed = strip_temperature(nested) or removed
    return removed
