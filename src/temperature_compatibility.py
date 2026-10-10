"""Session-wide model temperature exclusions and HTTP 400 compatibility."""
import re
import threading

_rejected_models = set()
_lock = threading.Lock()


def is_temperature_rejection(status, detail):
    return status == 400 and "temperature" in str(detail or "").lower()


def claude_omits_sampling(model):
    """Claude models whose documented API omits temperature, top_p and top_k.

    See https://platform.claude.com/docs/en/about-claude/model-deprecations
    and the family migration guides. Older families remain configurable.
    """
    name = str(model or "").strip().lower()
    if re.search(r"(?:^|[/.])claude-mythos-preview(?=$|[-:@.])", name):
        return True
    # Provider/Bedrock prefixes, dated aliases, and both Claude naming orders.
    families = r"sonnet|opus|haiku|fable|mythos"
    match = re.search(
        rf"(?:^|[/.])claude-(?:({families})-(\d+)(?:[.-](\d{{1,2}})(?!\d))?"
        rf"|(\d+)(?:[.-](\d{{1,2}})(?!\d))?-({families}))(?=$|[-:@])",
        name,
    )
    if not match:
        return False
    family = match[1] or match[6]
    version = (int(match[2] or match[4]), int(match[3] or match[5] or 0))
    minimum = {"sonnet": (5, 0), "opus": (4, 7), "haiku": (5, 5),
               "fable": (5, 0), "mythos": (5, 0)}[family]
    return version >= minimum


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
